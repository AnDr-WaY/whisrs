//! Meeting mode in the daemon: recording sessions, the processing queue,
//! crash recovery and graceful shutdown.
//!
//! A meeting lives entirely outside the dictation state machine and its
//! lock: dictation keeps working while a meeting records, and a long
//! transcription runs on its own task so IPC never waits for it. Status goes
//! out on a separate watch channel that only the tray reads, so a meeting
//! never animates the overlay or fires recording hooks.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Local};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use whisrs::audio::capture::AudioCaptureHandle;
use whisrs::audio::device::default_monitor_source;
use whisrs::meeting::process::process_meeting;
use whisrs::meeting::recorder::{spawn_track_writer, TrackStats};
use whisrs::meeting::wav_repair::repair_wav;
use whisrs::meeting::{
    store, EndReason, MeetingAction, MeetingMeta, MeetingPhase, MeetingState, MeetingStatus,
    TrackKind, TrackMeta,
};
use whisrs::{Config, MeetingConfig, Response};

use crate::context::DaemonContext;
use crate::notify::{send_notification, truncate_preview};

/// How often a track writer rewrites its WAV header, bounding what a crash
/// can leave outside the header's length.
const FLUSH_EVERY: Duration = Duration::from_secs(2);

/// How long a stop waits for each track writer to finalize its file.
const STOP_TIMEOUT: Duration = Duration::from_secs(10);

/// How long shutdown waits for a recording to be finalized, so a SIGTERM
/// from systemd is answered well inside its own timeout.
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(3);

/// Longest error text put in a notification.
const NOTIFY_ERROR_MAX: usize = 200;

/// A meeting that is recording, processing or queued, as shown in status.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ActiveMeeting {
    id: String,
    title: Option<String>,
    started_at: DateTime<Local>,
    dir: PathBuf,
}

impl ActiveMeeting {
    fn from_meta(dir: &Path, meta: &MeetingMeta) -> Self {
        Self {
            id: meta.id.clone(),
            title: meta.title.clone(),
            started_at: meta.started_at,
            dir: dir.to_path_buf(),
        }
    }
}

/// What the daemon is doing with meetings, the source of [`MeetingStatus`].
#[derive(Debug, Default)]
struct MeetingView {
    recording: Option<ActiveMeeting>,
    processing: Option<ActiveMeeting>,
    queued: Vec<ActiveMeeting>,
}

impl MeetingView {
    /// A recording wins over processing: it is what the user can still act
    /// on. With nothing recording or processing yet, the next queued meeting
    /// is shown, so a stop reports the meeting it just queued.
    fn status(&self) -> MeetingStatus {
        let (phase, active) = match (&self.recording, &self.processing) {
            (Some(rec), _) => (MeetingPhase::Recording, Some(rec)),
            (None, Some(proc)) => (MeetingPhase::Processing, Some(proc)),
            (None, None) => match self.queued.first() {
                Some(next) => (MeetingPhase::Processing, Some(next)),
                None => (MeetingPhase::Idle, None),
            },
        };
        MeetingStatus {
            phase,
            id: active.map(|a| a.id.clone()),
            title: active.and_then(|a| a.title.clone()),
            started_at: active.map(|a| a.started_at),
            dir: active.map(|a| a.dir.clone()),
            queued: self.queued.iter().map(|q| q.id.clone()).collect(),
        }
    }

    fn is_pending(&self, id: &str) -> bool {
        self.processing.as_ref().is_some_and(|p| p.id == id)
            || self.queued.iter().any(|q| q.id == id)
    }
}

/// One meeting waiting for the processing worker.
#[derive(Debug, Clone)]
pub(crate) struct QueuedMeeting {
    meeting: ActiveMeeting,
}

/// One track being recorded.
struct TrackCapture {
    kind: TrackKind,
    handle: AudioCaptureHandle,
    writer: JoinHandle<Result<TrackStats>>,
}

/// The meeting being recorded.
pub(crate) struct MeetingSession {
    dir: PathBuf,
    meta: MeetingMeta,
    /// Taken just before the captures start; track offsets are relative to it.
    started: Instant,
    tracks: Vec<TrackCapture>,
    /// The max-duration timer, aborted when the meeting stops another way.
    timer: Option<JoinHandle<()>>,
}

/// Meeting state shared by every connection. Deliberately separate from
/// `DaemonState`: dictation start holds that lock for seconds while a device
/// opens, and a meeting start must not wait on it (nor it on a meeting).
pub(crate) struct MeetingRuntime {
    session: tokio::sync::Mutex<Option<MeetingSession>>,
    view: std::sync::Mutex<MeetingView>,
    status_tx: watch::Sender<MeetingStatus>,
    queue_tx: mpsc::UnboundedSender<QueuedMeeting>,
}

impl MeetingRuntime {
    /// The runtime, the status receiver for the tray, and the queue receiver
    /// for [`run_worker`].
    pub(crate) fn new() -> (
        Self,
        watch::Receiver<MeetingStatus>,
        mpsc::UnboundedReceiver<QueuedMeeting>,
    ) {
        let (status_tx, status_rx) = watch::channel(MeetingStatus::default());
        let (queue_tx, queue_rx) = mpsc::unbounded_channel();
        let runtime = Self {
            session: tokio::sync::Mutex::new(None),
            view: std::sync::Mutex::new(MeetingView::default()),
            status_tx,
            queue_tx,
        };
        (runtime, status_rx, queue_rx)
    }

    fn status(&self) -> MeetingStatus {
        self.view.lock().unwrap_or_else(|e| e.into_inner()).status()
    }

    /// Change the view and broadcast the new status. `send_replace` updates
    /// the value even with no receiver (tray disabled), so `status` stays
    /// truthful.
    fn update(&self, f: impl FnOnce(&mut MeetingView)) {
        let status = {
            let mut view = self.view.lock().unwrap_or_else(|e| e.into_inner());
            f(&mut view);
            view.status()
        };
        self.status_tx.send_replace(status);
    }

    /// Queue a meeting for processing, refusing one that is already queued
    /// or being processed.
    fn enqueue(&self, meeting: ActiveMeeting) -> Result<()> {
        {
            let mut view = self.view.lock().unwrap_or_else(|e| e.into_inner());
            if view.is_pending(&meeting.id) {
                bail!("meeting {} is already queued for transcription", meeting.id);
            }
            view.queued.push(meeting.clone());
        }
        let id = meeting.id.clone();
        if self.queue_tx.send(QueuedMeeting { meeting }).is_err() {
            self.update(|v| v.queued.retain(|q| q.id != id));
            bail!("the meeting processing worker is not running");
        }
        self.update(|_| {});
        Ok(())
    }
}

/// Meeting notifications are progress the user asked for, so they follow
/// `[general] notify` alone: the overlay never shows meetings, so unlike
/// dictation toasts they are not suppressed when it is enabled.
fn notify(context: &DaemonContext, body: &str) {
    if context.notify {
        send_notification("whisrs", body);
    }
}

fn meetings_root(config: &Config) -> PathBuf {
    store::meetings_dir_from(MeetingConfig::effective(config).dir.as_deref())
}

/// The mic track's device: `[meeting] mic_device`, else `[audio] device`.
fn mic_device(meeting: &MeetingConfig, audio_device: &str) -> String {
    meeting
        .mic_device
        .as_deref()
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .unwrap_or(audio_device)
        .to_string()
}

/// Where the system track comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
enum SystemSource {
    /// The monitor of the default output device, looked up at start.
    DefaultMonitor,
    /// `"none"`: record the mic only.
    Off,
    /// A specific source, opened strictly.
    Named(String),
}

fn system_source(setting: Option<&str>) -> SystemSource {
    match setting.map(str::trim) {
        None | Some("") => SystemSource::DefaultMonitor,
        Some(s) if s.eq_ignore_ascii_case("default") => SystemSource::DefaultMonitor,
        Some(s) if s.eq_ignore_ascii_case("none") => SystemSource::Off,
        Some(s) => SystemSource::Named(s.to_string()),
    }
}

/// A track's first chunk relative to the moment the captures started.
fn offset_ms(started: Instant, first_chunk_at: Option<Instant>) -> i64 {
    first_chunk_at
        .map(|t| t.saturating_duration_since(started).as_millis() as i64)
        .unwrap_or(0)
}

fn track_meta(kind: TrackKind, device: String) -> TrackMeta {
    TrackMeta {
        kind,
        file: format!("audio/{}.wav", kind.file_stem()),
        device,
        offset_ms: 0,
        samples: 0,
    }
}

fn meeting_response(context: &DaemonContext) -> Response {
    Response::Meeting {
        meeting: context.meeting.status(),
    }
}

fn error_response(message: impl Into<String>) -> Response {
    Response::Error {
        message: message.into(),
    }
}

/// Handle `Command::Meeting`.
pub(crate) async fn handle_meeting(
    context: Arc<DaemonContext>,
    action: MeetingAction,
    title: Option<String>,
    id: Option<String>,
) -> Response {
    let result = match action {
        MeetingAction::Start => start(&context, title).await,
        MeetingAction::Stop => stop(&context, None, EndReason::Stopped).await,
        MeetingAction::Toggle => {
            let recording = context.meeting.session.lock().await.is_some();
            if recording {
                stop(&context, None, EndReason::Stopped).await
            } else {
                start(&context, title).await
            }
        }
        MeetingAction::Cancel => cancel(&context).await,
        MeetingAction::Status => Ok(()),
        MeetingAction::Retry => retry(&context, id.as_deref().unwrap_or("last")).await,
        MeetingAction::Copy => copy(&context, id.as_deref().unwrap_or("last")).await,
    };
    match result {
        Ok(()) => meeting_response(&context),
        Err(e) => {
            warn!("meeting {action:?} failed: {e:#}");
            error_response(format!("{e:#}"))
        }
    }
}

/// Open the mic (required) and the system source (best effort), each with a
/// writer started as soon as its capture is up so the first-chunk times are
/// not skewed by the other device's start. Blocking: runs on the blocking
/// pool, where the runtime handle is available for the writers.
fn open_tracks(
    dir: &Path,
    mic: &str,
    system: SystemSource,
) -> Result<(Vec<TrackCapture>, Vec<TrackMeta>, Option<String>)> {
    let mut captures = Vec::new();
    let mut metas = Vec::new();

    let mut handle = AudioCaptureHandle::start_strict(mic)
        .with_context(|| format!("failed to open the microphone ({mic})"))?;
    let rx = handle
        .take_receiver()
        .context("mic capture has no receiver")?;
    let writer = spawn_track_writer(rx, store::audio_path(dir, TrackKind::Mic), FLUSH_EVERY);
    captures.push(TrackCapture {
        kind: TrackKind::Mic,
        handle,
        writer,
    });
    metas.push(track_meta(TrackKind::Mic, mic.to_string()));

    let system_name = match system {
        SystemSource::Off => return Ok((captures, metas, None)),
        SystemSource::DefaultMonitor => default_monitor_source(),
        SystemSource::Named(name) => Ok(name),
    };
    // The strict start's errors already name the device.
    let opened =
        system_name.and_then(|name| AudioCaptureHandle::start_strict(&name).map(|h| (name, h)));
    match opened {
        Ok((name, mut handle)) => {
            let rx = handle
                .take_receiver()
                .context("system capture has no receiver")?;
            let writer =
                spawn_track_writer(rx, store::audio_path(dir, TrackKind::System), FLUSH_EVERY);
            captures.push(TrackCapture {
                kind: TrackKind::System,
                handle,
                writer,
            });
            metas.push(track_meta(TrackKind::System, name));
            Ok((captures, metas, None))
        }
        Err(e) => Ok((captures, metas, Some(format!("{e:#}")))),
    }
}

async fn start(context: &Arc<DaemonContext>, title: Option<String>) -> Result<()> {
    let mut slot = context.meeting.session.lock().await;
    if let Some(session) = slot.as_ref() {
        bail!("a meeting is already recording ({})", session.meta.id);
    }

    let meeting_cfg = MeetingConfig::effective(&context.config);
    let root = meetings_root(&context.config);
    let title = title
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty());
    let started_at = Local::now();
    let (id, dir) = store::create(&root, started_at, title.as_deref())?;

    let mic = mic_device(&meeting_cfg, &context.config.audio.device);
    let system = system_source(meeting_cfg.system_device.as_deref());
    let started = Instant::now();
    let open_dir = dir.clone();
    let opened = tokio::task::spawn_blocking(move || open_tracks(&open_dir, &mic, system))
        .await
        .context("audio start panicked")
        .and_then(|r| r);
    let (tracks, track_metas, system_error) = match opened {
        Ok(opened) => opened,
        Err(e) => {
            remove_meeting_dir(&dir).await;
            return Err(e);
        }
    };

    let mut meta = MeetingMeta::new(id.clone(), title, started_at);
    meta.tracks = track_metas;
    if let Some(e) = &system_error {
        warn!("meeting {id}: system audio unavailable, recording the mic only: {e}");
        meta.errors.push(format!(
            "System audio unavailable, recorded the mic only: {e}"
        ));
    }
    if let Err(e) = store::write_meta(&dir, &meta) {
        drop(finish_tracks(tracks, STOP_TIMEOUT).await);
        remove_meeting_dir(&dir).await;
        return Err(e);
    }

    let minutes = meeting_cfg.max_duration_minutes.max(1);
    let timer = {
        let ctx = Arc::clone(context);
        let id = id.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(u64::from(minutes) * 60)).await;
            info!("meeting {id} reached the {minutes} minute limit, stopping");
            if let Err(e) = stop(&ctx, Some(&id), EndReason::MaxDuration).await {
                warn!("failed to stop meeting {id} at its time limit: {e:#}");
            }
        })
    };

    let active = ActiveMeeting::from_meta(&dir, &meta);
    info!("meeting {id} recording to {}", dir.display());
    *slot = Some(MeetingSession {
        dir,
        meta,
        started,
        tracks,
        timer: Some(timer),
    });
    drop(slot);
    context.meeting.update(|v| v.recording = Some(active));

    notify(context, "Meeting recording started");
    if let Some(e) = system_error {
        notify(
            context,
            &format!(
                "System audio unavailable, recording your mic only: {}",
                truncate_preview(&e, NOTIFY_ERROR_MAX)
            ),
        );
    }
    Ok(())
}

/// Result of finalizing one track.
#[derive(Debug)]
struct TrackOutcome {
    kind: TrackKind,
    stats: Result<TrackStats, String>,
}

/// Stop every capture, then wait (bounded) for each writer to finalize its
/// WAV. The capture threads are joined on the blocking pool without waiting:
/// they exit on their own after the stop signal.
async fn finish_tracks(mut tracks: Vec<TrackCapture>, limit: Duration) -> Vec<TrackOutcome> {
    for track in &mut tracks {
        track.handle.stop();
    }
    let deadline = tokio::time::Instant::now() + limit;
    let mut outcomes = Vec::new();
    for track in tracks {
        let TrackCapture {
            kind,
            handle,
            writer,
        } = track;
        let stats = match tokio::time::timeout_at(deadline, writer).await {
            Ok(Ok(Ok(stats))) => Ok(stats),
            Ok(Ok(Err(e))) => Err(format!("{e:#}")),
            Ok(Err(e)) => Err(format!("writer task failed: {e}")),
            Err(_) => Err("the track did not finish writing in time".to_string()),
        };
        tokio::task::spawn_blocking(move || drop(handle));
        outcomes.push(TrackOutcome { kind, stats });
    }
    outcomes
}

/// Fold the writers' results into the meta: sample counts, offsets, and a
/// note for any track that did not finalize cleanly.
fn apply_outcomes(meta: &mut MeetingMeta, started: Instant, outcomes: &[TrackOutcome]) {
    for outcome in outcomes {
        let Some(track) = meta.tracks.iter_mut().find(|t| t.kind == outcome.kind) else {
            continue;
        };
        match &outcome.stats {
            Ok(stats) => {
                track.samples = stats.samples;
                track.offset_ms = offset_ms(started, stats.first_chunk_at);
            }
            Err(e) => {
                let note = format!("{} track did not finalize: {e}", outcome.kind.file_stem());
                warn!("meeting {}: {note}", meta.id);
                meta.errors.push(note);
            }
        }
    }
}

/// Mark a recording as ended and waiting for processing.
fn end_recording(meta: &mut MeetingMeta, reason: EndReason, ended_at: DateTime<Local>) {
    meta.status = MeetingState::Processing;
    meta.ended_at = Some(ended_at);
    meta.end_reason = Some(reason);
}

/// Finalize a track that a writer could not: fix its header and read the
/// sample count off the file.
fn repair_unfinished(dir: &Path, meta: &mut MeetingMeta, outcomes: &[TrackOutcome]) {
    for outcome in outcomes.iter().filter(|o| o.stats.is_err()) {
        let path = store::audio_path(dir, outcome.kind);
        match repair_wav(&path) {
            Ok(rep) => {
                if let Some(t) = meta.tracks.iter_mut().find(|t| t.kind == outcome.kind) {
                    t.samples = rep.samples;
                }
            }
            Err(e) => warn!("could not repair {}: {e:#}", path.display()),
        }
    }
}

/// Stop the recording. `expected_id` (from the max-duration timer) makes
/// the stop a no-op when a different meeting is recording by now.
async fn stop(
    context: &Arc<DaemonContext>,
    expected_id: Option<&str>,
    reason: EndReason,
) -> Result<()> {
    let session = {
        let mut slot = context.meeting.session.lock().await;
        match slot.as_ref() {
            None if expected_id.is_some() => return Ok(()),
            None => bail!("no meeting is recording"),
            Some(s) if expected_id.is_some_and(|id| id != s.meta.id) => return Ok(()),
            Some(_) => slot.take().expect("checked above"),
        }
    };
    let MeetingSession {
        dir,
        mut meta,
        started,
        tracks,
        timer,
    } = session;
    // The timer is the caller on a max-duration stop: aborting it there
    // would cancel this very stop at its next await.
    if let Some(timer) = timer {
        if reason != EndReason::MaxDuration {
            timer.abort();
        }
    }

    let outcomes = finish_tracks(tracks, STOP_TIMEOUT).await;
    apply_outcomes(&mut meta, started, &outcomes);
    repair_unfinished(&dir, &mut meta, &outcomes);
    end_recording(&mut meta, reason, Local::now());
    let written = store::write_meta(&dir, &meta);
    let active = ActiveMeeting::from_meta(&dir, &meta);
    context.meeting.update(|v| v.recording = None);
    written?;
    info!("meeting {} stopped ({reason:?})", meta.id);

    let queued = context.meeting.enqueue(active);
    match reason {
        EndReason::MaxDuration => notify(
            context,
            &format!(
                "Meeting stopped at the {} minute limit. Transcribing...",
                MeetingConfig::effective(&context.config).max_duration_minutes
            ),
        ),
        _ => notify(context, "Meeting recording stopped. Transcribing..."),
    }
    queued
}

async fn cancel(context: &Arc<DaemonContext>) -> Result<()> {
    let session = context
        .meeting
        .session
        .lock()
        .await
        .take()
        .context("no meeting is recording")?;
    if let Some(timer) = &session.timer {
        timer.abort();
    }
    drop(finish_tracks(session.tracks, STOP_TIMEOUT).await);
    remove_meeting_dir(&session.dir).await;
    context.meeting.update(|v| v.recording = None);
    info!("meeting {} cancelled and deleted", session.meta.id);
    notify(context, "Meeting recording cancelled and deleted");
    Ok(())
}

async fn remove_meeting_dir(dir: &Path) {
    let dir = dir.to_path_buf();
    let result = tokio::task::spawn_blocking(move || std::fs::remove_dir_all(&dir).map(|_| dir))
        .await
        .map_err(anyhow::Error::from)
        .and_then(|r| r.map_err(anyhow::Error::from));
    if let Err(e) = result {
        warn!("failed to delete meeting folder: {e:#}");
    }
}

/// Whether any of the meeting's track files is still on disk.
fn has_audio(dir: &Path, meta: &MeetingMeta) -> bool {
    !meta.audio_deleted && meta.tracks.iter().any(|t| dir.join(&t.file).exists())
}

async fn retry(context: &Arc<DaemonContext>, query: &str) -> Result<()> {
    let root = meetings_root(&context.config);
    let dir = store::resolve(&root, query)?;
    let mut meta = store::read_meta(&dir)?;
    let recording_id = context
        .meeting
        .session
        .lock()
        .await
        .as_ref()
        .map(|s| s.meta.id.clone());
    if recording_id.as_deref() == Some(meta.id.as_str()) {
        bail!("meeting {} is still recording", meta.id);
    }
    if !has_audio(&dir, &meta) {
        bail!(
            "meeting {} has no audio left to transcribe (deleted by audio_retention_days)",
            meta.id
        );
    }
    meta.status = MeetingState::Processing;
    meta.error = None;
    store::write_meta(&dir, &meta)?;
    context
        .meeting
        .enqueue(ActiveMeeting::from_meta(&dir, &meta))?;
    info!("meeting {} queued for transcription again", meta.id);
    Ok(())
}

async fn copy(context: &Arc<DaemonContext>, query: &str) -> Result<()> {
    let root = meetings_root(&context.config);
    let dir = store::resolve(&root, query)?;
    let meta = store::read_meta(&dir)?;
    let path = store::transcript_path(&dir, &meta.id);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let state = serde_json::to_string(&meta.status).unwrap_or_default();
            let mut message = format!(
                "meeting {} has no transcript yet (status {})",
                meta.id,
                state.trim_matches('"')
            );
            if meta.status == MeetingState::Failed {
                message.push_str(&format!(". Retry with: whisrs meeting retry {}", meta.id));
            }
            bail!(message);
        }
        Err(e) => return Err(e).with_context(|| format!("failed to read {}", path.display())),
    };
    tokio::task::spawn_blocking(move || xkb_type::default_clipboard().set_text(&text))
        .await
        .context("clipboard task panicked")?
        .context("failed to copy the transcript to the clipboard")?;
    notify(context, "Meeting transcript copied to the clipboard");
    Ok(())
}

/// The processing worker: one meeting at a time, in queue order, on its own
/// task so a long transcription never holds up IPC or dictation.
pub(crate) async fn run_worker(
    context: Arc<DaemonContext>,
    mut queue: mpsc::UnboundedReceiver<QueuedMeeting>,
) {
    while let Some(item) = queue.recv().await {
        process_one(&context, item).await;
    }
}

async fn process_one(context: &Arc<DaemonContext>, item: QueuedMeeting) {
    let QueuedMeeting { meeting } = item;
    let id = meeting.id.clone();
    let dir = meeting.dir.clone();
    context.meeting.update(|v| {
        v.queued.retain(|q| q.id != id);
        v.processing = Some(meeting);
    });

    let result = process_meeting(
        &dir,
        &context.config,
        Some(Arc::clone(&context.transcription_backend)),
    )
    .await;
    match result {
        Ok(meta) => {
            let md = store::transcript_path(&dir, &meta.id);
            info!("meeting {id} transcript ready: {}", md.display());
            notify(
                context,
                &format!("Meeting transcript ready: {}", md.display()),
            );
        }
        Err(e) => {
            let message = format!("{e:#}");
            error!("meeting {id} transcription failed: {message}");
            if let Err(write_err) = mark_failed(&dir, &message) {
                error!("failed to record the failure of meeting {id}: {write_err:#}");
            }
            notify(
                context,
                &format!(
                    "Meeting transcription failed: {}. Retry with: whisrs meeting retry {id}",
                    truncate_preview(&message, NOTIFY_ERROR_MAX)
                ),
            );
        }
    }
    context.meeting.update(|v| v.processing = None);
    sweep(context).await;
}

fn mark_failed(dir: &Path, message: &str) -> Result<()> {
    let mut meta = store::read_meta(dir)?;
    meta.status = MeetingState::Failed;
    meta.error = Some(message.to_string());
    store::write_meta(dir, &meta)
}

async fn sweep(context: &DaemonContext) {
    let root = meetings_root(&context.config);
    let days = MeetingConfig::effective(&context.config).audio_retention_days;
    let swept =
        tokio::task::spawn_blocking(move || store::sweep_retention(&root, days, Local::now()))
            .await
            .unwrap_or_default();
    for dir in swept {
        info!("deleted meeting audio past retention: {}", dir.display());
    }
}

/// Update an interrupted meeting's meta from its repaired tracks: sample
/// counts from the files, and a recording that never stopped becomes a
/// `Recovered` one ending at `ended_fallback`. A meeting already stopped
/// (`Processing`, e.g. ended by a shutdown) keeps its end.
fn recover_meta(
    meta: &mut MeetingMeta,
    repaired: &[(TrackKind, u64)],
    ended_fallback: DateTime<Local>,
) {
    for (kind, samples) in repaired {
        if let Some(track) = meta.tracks.iter_mut().find(|t| t.kind == *kind) {
            track.samples = *samples;
        }
    }
    if meta.status == MeetingState::Recording {
        meta.end_reason = Some(EndReason::Recovered);
        if meta.ended_at.is_none() {
            meta.ended_at = Some(ended_fallback);
        }
    }
    meta.status = MeetingState::Processing;
    meta.error = None;
}

/// Repair an interrupted meeting's WAVs and update its meta. Blocking.
fn recover_dir(dir: &Path) -> Result<MeetingMeta> {
    let mut meta = store::read_meta(dir)?;
    let mut repaired = Vec::new();
    let mut newest: Option<std::time::SystemTime> = None;
    if !meta.audio_deleted {
        for track in meta.tracks.clone() {
            let path = dir.join(&track.file);
            if !path.exists() {
                continue;
            }
            match repair_wav(&path) {
                Ok(outcome) => {
                    if outcome.changed {
                        info!("repaired {}", path.display());
                    }
                    repaired.push((track.kind, outcome.samples));
                }
                Err(e) => {
                    let note = format!("{} track could not be repaired: {e:#}", track.file);
                    warn!("meeting {}: {note}", meta.id);
                    meta.errors.push(note);
                }
            }
            if let Ok(modified) = std::fs::metadata(&path).and_then(|m| m.modified()) {
                newest = Some(newest.map_or(modified, |n| n.max(modified)));
            }
        }
    }
    let ended = newest
        .map(DateTime::<Local>::from)
        .unwrap_or_else(Local::now);
    recover_meta(&mut meta, &repaired, ended);
    store::write_meta(dir, &meta)?;
    Ok(meta)
}

/// At startup: requeue every meeting a crash, reboot or shutdown left
/// recording or processing, then apply audio retention. Spawned, so the
/// IPC server comes up without waiting for it.
pub(crate) async fn recover_on_startup(context: Arc<DaemonContext>) {
    let root = meetings_root(&context.config);
    let scan_root = root.clone();
    let recovered = tokio::task::spawn_blocking(move || {
        store::interrupted(&scan_root)
            .into_iter()
            .filter_map(|dir| match recover_dir(&dir) {
                Ok(meta) => Some((dir, meta)),
                Err(e) => {
                    warn!("failed to recover meeting {}: {e:#}", dir.display());
                    None
                }
            })
            .collect::<Vec<_>>()
    })
    .await
    .unwrap_or_default();

    // Oldest first, so they are processed in the order they happened.
    for (dir, meta) in recovered.into_iter().rev() {
        info!(
            "resuming interrupted meeting {} ({:?})",
            meta.id, meta.end_reason
        );
        if let Err(e) = context
            .meeting
            .enqueue(ActiveMeeting::from_meta(&dir, &meta))
        {
            warn!("failed to queue meeting {}: {e:#}", meta.id);
        }
    }
    sweep(&context).await;
}

/// On SIGTERM/SIGINT: finalize a recording meeting (bounded) and leave it
/// `processing` with `end_reason: shutdown`, so the next start transcribes
/// it. A meeting mid-processing is already `processing` on disk and is
/// resumed the same way.
pub(crate) async fn shutdown(context: &DaemonContext) {
    let session = match tokio::time::timeout(SHUTDOWN_TIMEOUT, context.meeting.session.lock()).await
    {
        Ok(mut slot) => slot.take(),
        Err(_) => {
            warn!("meeting busy at shutdown; it will be recovered on the next start");
            return;
        }
    };
    let Some(session) = session else {
        return;
    };
    let MeetingSession {
        dir,
        mut meta,
        started,
        tracks,
        timer,
    } = session;
    if let Some(timer) = timer {
        timer.abort();
    }
    let outcomes = finish_tracks(tracks, SHUTDOWN_TIMEOUT).await;
    apply_outcomes(&mut meta, started, &outcomes);
    end_recording(&mut meta, EndReason::Shutdown, Local::now());
    match store::write_meta(&dir, &meta) {
        Ok(()) => info!(
            "meeting {} saved at shutdown; it will be transcribed on the next start",
            meta.id
        ),
        Err(e) => error!("failed to save meeting {} at shutdown: {e:#}", meta.id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use whisrs::meeting::Segment;
    use whisrs::transcription::{Transcript, TranscriptionBackend, TranscriptionConfig};
    use whisrs::State;

    fn active(id: &str) -> ActiveMeeting {
        ActiveMeeting {
            id: id.into(),
            title: Some(format!("{id} title")),
            started_at: Local::now(),
            dir: PathBuf::from(format!("/m/{id}")),
        }
    }

    #[test]
    fn status_idle_by_default() {
        assert_eq!(MeetingView::default().status(), MeetingStatus::default());
    }

    #[test]
    fn status_recording_wins_over_processing() {
        let view = MeetingView {
            recording: Some(active("rec")),
            processing: Some(active("proc")),
            queued: vec![active("q")],
        };
        let status = view.status();
        assert_eq!(status.phase, MeetingPhase::Recording);
        assert_eq!(status.id.as_deref(), Some("rec"));
        assert_eq!(status.title.as_deref(), Some("rec title"));
        assert_eq!(status.queued, vec!["q".to_string()]);
    }

    #[test]
    fn status_processing_and_queue_only() {
        let view = MeetingView {
            recording: None,
            processing: Some(active("proc")),
            queued: vec![],
        };
        assert_eq!(view.status().phase, MeetingPhase::Processing);
        assert_eq!(view.status().id.as_deref(), Some("proc"));

        let queued_only = MeetingView {
            queued: vec![active("a"), active("b2")],
            ..Default::default()
        };
        assert_eq!(queued_only.status().phase, MeetingPhase::Processing);
        assert_eq!(queued_only.status().id.as_deref(), Some("a"));
        assert_eq!(queued_only.status().queued, vec!["a", "b2"]);
        assert!(queued_only.is_pending("a"));
        assert!(!queued_only.is_pending("b"));
    }

    #[test]
    fn device_selection() {
        let mut cfg = MeetingConfig::default();
        assert_eq!(mic_device(&cfg, "default"), "default");
        cfg.mic_device = Some("  ".into());
        assert_eq!(mic_device(&cfg, "hw:1"), "hw:1");
        cfg.mic_device = Some("my-mic".into());
        assert_eq!(mic_device(&cfg, "hw:1"), "my-mic");

        assert_eq!(system_source(None), SystemSource::DefaultMonitor);
        assert_eq!(system_source(Some("Default")), SystemSource::DefaultMonitor);
        assert_eq!(system_source(Some("none")), SystemSource::Off);
        assert_eq!(
            system_source(Some("x.monitor")),
            SystemSource::Named("x.monitor".into())
        );
    }

    #[test]
    fn offsets_are_relative_to_the_start() {
        let start = Instant::now();
        assert_eq!(offset_ms(start, None), 0);
        assert_eq!(
            offset_ms(start, Some(start + Duration::from_millis(250))),
            250
        );
        // A clock read before the start never goes negative.
        assert_eq!(offset_ms(start + Duration::from_secs(1), Some(start)), 0);
    }

    #[test]
    fn apply_outcomes_fills_tracks_and_notes_failures() {
        let start = Instant::now();
        let mut meta = MeetingMeta::new("m".into(), None, Local::now());
        meta.tracks = vec![
            track_meta(TrackKind::Mic, "mic".into()),
            track_meta(TrackKind::System, "sys".into()),
        ];
        let outcomes = vec![
            TrackOutcome {
                kind: TrackKind::Mic,
                stats: Ok(TrackStats {
                    samples: 32_000,
                    first_chunk_at: Some(start + Duration::from_millis(40)),
                }),
            },
            TrackOutcome {
                kind: TrackKind::System,
                stats: Err("boom".into()),
            },
        ];
        apply_outcomes(&mut meta, start, &outcomes);
        assert_eq!(meta.tracks[0].samples, 32_000);
        assert_eq!(meta.tracks[0].offset_ms, 40);
        assert_eq!(meta.tracks[1].samples, 0);
        assert_eq!(meta.errors.len(), 1);
        assert!(meta.errors[0].contains("system"));

        end_recording(&mut meta, EndReason::Stopped, Local::now());
        assert_eq!(meta.status, MeetingState::Processing);
        assert_eq!(meta.end_reason, Some(EndReason::Stopped));
        assert!(meta.ended_at.is_some());
    }

    #[test]
    fn recover_meta_marks_recording_as_recovered() {
        let ended = Local::now();
        let mut meta = MeetingMeta::new("m".into(), None, Local::now());
        meta.tracks = vec![track_meta(TrackKind::Mic, "mic".into())];
        recover_meta(&mut meta, &[(TrackKind::Mic, 48_000)], ended);
        assert_eq!(meta.status, MeetingState::Processing);
        assert_eq!(meta.end_reason, Some(EndReason::Recovered));
        assert_eq!(meta.ended_at, Some(ended));
        assert_eq!(meta.tracks[0].samples, 48_000);
    }

    #[test]
    fn recover_meta_keeps_a_shutdown_end() {
        let shutdown_at = Local::now() - chrono::Duration::minutes(5);
        let mut meta = MeetingMeta::new("m".into(), None, Local::now());
        meta.status = MeetingState::Processing;
        end_recording(&mut meta, EndReason::Shutdown, shutdown_at);
        recover_meta(&mut meta, &[], Local::now());
        assert_eq!(meta.end_reason, Some(EndReason::Shutdown));
        assert_eq!(meta.ended_at, Some(shutdown_at));
        assert_eq!(meta.status, MeetingState::Processing);
    }

    /// Every loud chunk becomes one segment with fixed text.
    struct FakeBackend {
        text: &'static str,
        fail: bool,
        calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl TranscriptionBackend for FakeBackend {
        async fn transcribe(&self, _audio: &[u8], _config: &TranscriptionConfig) -> Result<String> {
            unreachable!("meetings ask for segments")
        }
        fn sends_prompt(&self, _config: &TranscriptionConfig) -> bool {
            false
        }
        async fn transcribe_segments(
            &self,
            _audio: &[u8],
            _config: &TranscriptionConfig,
        ) -> Result<Transcript> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if self.fail {
                bail!("fake backend is down");
            }
            Ok(Transcript {
                segments: vec![Segment {
                    start_ms: 0,
                    end_ms: 1000,
                    text: self.text.to_string(),
                    speaker: None,
                    words: None,
                }],
            })
        }
        fn max_segment_chunk_secs(&self, _config: &TranscriptionConfig) -> Option<u32> {
            None
        }
    }

    fn test_context(
        root: &Path,
        backend: Arc<FakeBackend>,
    ) -> (Arc<DaemonContext>, mpsc::UnboundedReceiver<QueuedMeeting>) {
        let config: Config = toml::from_str(&format!(
            "[general]\nbackend = \"groq\"\n[meeting]\nuser_name = \"Tester\"\ndir = \"{}\"\n",
            root.display()
        ))
        .unwrap();
        let (meeting, _status_rx, queue_rx) = MeetingRuntime::new();
        let context = DaemonContext {
            config,
            window_tracker: Arc::new(whisrs::window::NoopTracker),
            transcription_backend: backend,
            notify: false,
            state_tx: watch::channel(State::Idle).0,
            overlay_level_tx: None,
            overlay_enabled: false,
            meeting,
        };
        (Arc::new(context), queue_rx)
    }

    /// A meeting a crash left `recording`: a mic WAV whose header was never
    /// finalized (sizes still zero) holding one second of loud audio.
    fn crashed_meeting(root: &Path) -> PathBuf {
        let (id, dir) = store::create(root, Local::now(), Some("crash")).unwrap();
        let mut meta = MeetingMeta::new(id, Some("crash".into()), Local::now());
        meta.tracks = vec![track_meta(TrackKind::Mic, "default".into())];
        store::write_meta(&dir, &meta).unwrap();

        let samples: Vec<i16> = (0..16_000)
            .map(|i| if i % 2 == 0 { 8_000 } else { -8_000 })
            .collect();
        let mut bytes = whisrs::audio::wav::encode_wav(&samples).unwrap();
        // Zero the RIFF and data sizes, as an unflushed writer leaves them.
        bytes[4..8].copy_from_slice(&0u32.to_le_bytes());
        let data = bytes.windows(4).position(|w| w == b"data").unwrap();
        bytes[data + 4..data + 8].copy_from_slice(&0u32.to_le_bytes());
        std::fs::write(store::audio_path(&dir, TrackKind::Mic), bytes).unwrap();
        dir
    }

    #[tokio::test]
    async fn crashed_meeting_is_recovered_and_transcribed() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = crashed_meeting(tmp.path());
        let backend = Arc::new(FakeBackend {
            text: "hello from the crash",
            fail: false,
            calls: AtomicUsize::new(0),
        });
        let (context, mut queue) = test_context(tmp.path(), Arc::clone(&backend));

        recover_on_startup(Arc::clone(&context)).await;
        let meta = store::read_meta(&dir).unwrap();
        assert_eq!(meta.status, MeetingState::Processing);
        assert_eq!(meta.end_reason, Some(EndReason::Recovered));
        assert_eq!(meta.tracks[0].samples, 16_000);
        assert_eq!(context.meeting.status().queued, vec![meta.id.clone()]);

        let item = queue.try_recv().expect("recovery queued the meeting");
        process_one(&context, item).await;

        let meta = store::read_meta(&dir).unwrap();
        assert_eq!(meta.status, MeetingState::Done);
        assert!(backend.calls.load(Ordering::Relaxed) > 0);
        let md = std::fs::read_to_string(store::transcript_path(&dir, &meta.id)).unwrap();
        assert!(md.contains("hello from the crash"), "{md}");
        assert_eq!(context.meeting.status(), MeetingStatus::default());
    }

    #[tokio::test]
    async fn failed_processing_is_marked_and_retryable() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = crashed_meeting(tmp.path());
        let backend = Arc::new(FakeBackend {
            text: "",
            fail: true,
            calls: AtomicUsize::new(0),
        });
        let (context, mut queue) = test_context(tmp.path(), backend);

        recover_on_startup(Arc::clone(&context)).await;
        process_one(&context, queue.try_recv().unwrap()).await;
        let meta = store::read_meta(&dir).unwrap();
        assert_eq!(meta.status, MeetingState::Failed);
        assert!(meta
            .error
            .as_deref()
            .is_some_and(|e| e.contains("fake backend is down")));

        // Retry requeues it, and a second retry while queued is refused.
        retry(&context, "last").await.unwrap();
        assert_eq!(
            store::read_meta(&dir).unwrap().status,
            MeetingState::Processing
        );
        assert!(retry(&context, "last").await.is_err());
        assert!(queue.try_recv().is_ok());
    }

    #[tokio::test]
    async fn stop_and_cancel_without_a_meeting_are_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let backend = Arc::new(FakeBackend {
            text: "",
            fail: false,
            calls: AtomicUsize::new(0),
        });
        let (context, _queue) = test_context(tmp.path(), backend);
        let resp = handle_meeting(Arc::clone(&context), MeetingAction::Stop, None, None).await;
        assert!(matches!(resp, Response::Error { ref message } if message.contains("no meeting")));
        let resp = handle_meeting(Arc::clone(&context), MeetingAction::Cancel, None, None).await;
        assert!(matches!(resp, Response::Error { .. }));
        let resp = handle_meeting(context, MeetingAction::Status, None, None).await;
        assert!(matches!(
            resp,
            Response::Meeting { meeting } if meeting.phase == MeetingPhase::Idle
        ));
        // Shutdown with nothing recording is a no-op.
    }
}
