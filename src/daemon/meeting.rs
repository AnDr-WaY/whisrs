//! Meeting mode in the daemon: recording sessions, the processing queue,
//! crash recovery and graceful shutdown.
//!
//! A meeting lives entirely outside the dictation state machine and its
//! lock, and a long transcription runs on its own task so IPC never waits
//! for it. While a meeting records, dictation and voice commands refuse to
//! start ([`refuse_while_recording`]); one already recording finishes as
//! usual. Status goes out on a separate watch channel that only the tray
//! reads, so a meeting never animates the overlay or fires recording hooks.

use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Local};
use futures_util::FutureExt;
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

/// How much longer a writer that missed its timeout gets once its capture
/// (the sender side of its channel) has been dropped.
const STOP_GRACE: Duration = Duration::from_secs(5);

/// How long shutdown waits for a recording to be finalized, so a SIGTERM
/// from systemd is answered well inside its own timeout.
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(3);

/// Grace after [`SHUTDOWN_TIMEOUT`]; recovery repairs whatever is left.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(1);

/// How long shutdown waits for the session lock. A start holds it while
/// it opens two devices (up to ~5 s each), and must either finish or be
/// left with a `meeting.json` that recovery picks up.
const SHUTDOWN_LOCK_TIMEOUT: Duration = Duration::from_secs(12);

/// How long shutdown waits for the aborted processing worker to unwind.
const WORKER_ABORT_TIMEOUT: Duration = Duration::from_secs(2);

/// Size of the header the track writer (`hound`, 16-bit mono PCM) writes
/// before any sample; a track no longer than this holds no audio.
const WAV_HEADER_LEN: u64 = 44;

/// Processing attempts after which recovery stops requeueing a meeting: a
/// meeting whose processing crashes the daemon would otherwise be retried
/// on every restart, and systemd restarts it forever.
const MAX_ATTEMPTS: u32 = 3;

/// Longest error text put in a notification.
const NOTIFY_ERROR_MAX: usize = 200;

/// Start of the error a stop returns when it saved the meeting but could
/// not queue it. [`stop`] notifies it itself, since a compositor-bound
/// `whisrs meeting toggle` shows nobody its stderr.
const NOT_TRANSCRIBED: &str = "meeting stopped but not transcribed";

/// Whether a meeting error was already notified where it happened, so the
/// hotkey and tray dispatcher does not toast it a second time.
pub(crate) fn error_already_notified(message: &str) -> bool {
    message.starts_with(NOT_TRANSCRIBED)
}

/// A start that opens the mic to type what the user says, which a recording
/// meeting refuses.
#[derive(Debug, Clone, Copy)]
pub(crate) enum DictationStart<'a> {
    /// `toggle`: the CLI, its hotkey, or a tray left-click.
    Dictation,
    /// `command`.
    CommandMode,
    /// A named `[[llm_commands]]` entry.
    LlmCommand(&'a str),
}

/// Refuse `start` while a meeting records, returning the error to send back.
///
/// The mic is shared with the call: the others would hear the dictation, and
/// the meeting transcript would record it as the user's speech. Only starts
/// are refused: a session already recording when the meeting started keeps
/// its own capture and stops, cancels and types as usual.
///
/// Notified like any other failure (`notify_error`), since a hotkey or tray
/// press shows nobody the response and no overlay appears for a start that
/// did not happen.
pub(crate) fn refuse_while_recording(
    context: &DaemonContext,
    start: DictationStart<'_>,
) -> Option<Response> {
    if !context.meeting.is_recording() {
        return None;
    }
    let (what, subject) = match start {
        DictationStart::Dictation => ("dictation".to_string(), "Dictation".to_string()),
        DictationStart::CommandMode => ("command mode".to_string(), "Command mode".to_string()),
        DictationStart::LlmCommand(name) => (format!("llm-command '{name}'"), format!("'{name}'")),
    };
    info!("{what} not started: a meeting is recording");
    let message = format!("A meeting is recording. {subject} is off until you stop it.");
    if context.notify_error() {
        send_notification("whisrs", &message);
    }
    Some(error_response(message))
}

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
    /// `None` only in tests, which feed the writer's channel directly.
    handle: Option<AudioCaptureHandle>,
    writer: JoinHandle<Result<TrackStats>>,
    /// Set when the writer ended before the meeting was stopped: the device
    /// stopped delivering audio (unplugged, server restart) or a write failed.
    ended_early: Arc<AtomicBool>,
}

impl TrackCapture {
    fn new(
        kind: TrackKind,
        handle: Option<AudioCaptureHandle>,
        writer: JoinHandle<Result<TrackStats>>,
    ) -> Self {
        Self {
            kind,
            handle,
            writer,
            ended_early: Arc::new(AtomicBool::new(false)),
        }
    }
}

/// What a track is called in notifications and notes.
fn track_label(kind: TrackKind) -> &'static str {
    match kind {
        TrackKind::Mic => "microphone",
        TrackKind::System => "system audio",
    }
}

/// Wrap a track's writer so that its ending before `stopping` is set is
/// noticed: logged, flagged for the meta, and notified once.
fn watch_writer(track: &mut TrackCapture, id: &str, stopping: &Arc<AtomicBool>, notify: bool) {
    let placeholder = tokio::spawn(async { Ok(TrackStats::default()) });
    let writer = std::mem::replace(&mut track.writer, placeholder);
    let kind = track.kind;
    let early = Arc::clone(&track.ended_early);
    let stopping = Arc::clone(stopping);
    let id = id.to_string();
    track.writer = tokio::spawn(async move {
        let result = match writer.await {
            Ok(result) => result,
            Err(e) => Err(anyhow::anyhow!("writer task failed: {e}")),
        };
        if !stopping.load(Ordering::Acquire) {
            early.store(true, Ordering::Release);
            match &result {
                Ok(_) => warn!(
                    "meeting {id}: the {} stopped delivering audio before the meeting was stopped",
                    track_label(kind)
                ),
                Err(e) => warn!(
                    "meeting {id}: the {} track stopped early: {e:#}",
                    track_label(kind)
                ),
            }
            if notify {
                send_notification(
                    "whisrs",
                    &format!("Meeting: {} stopped delivering audio", track_label(kind)),
                );
            }
        }
        result
    });
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
    /// Set before the captures are stopped, so a writer ending after it is
    /// a normal end and one ending before it is a device that died.
    stopping: Arc<AtomicBool>,
}

/// Meeting state shared by every connection. Deliberately separate from
/// `DaemonState`: dictation start holds that lock for seconds while a device
/// opens, and a meeting start must not wait on it (nor it on a meeting).
pub(crate) struct MeetingRuntime {
    session: tokio::sync::Mutex<Option<MeetingSession>>,
    view: std::sync::Mutex<MeetingView>,
    status_tx: watch::Sender<MeetingStatus>,
    queue_tx: mpsc::UnboundedSender<QueuedMeeting>,
    /// Set by [`shutdown`], so a start queued behind it on the session lock
    /// does not open devices while the daemon exits.
    shutting_down: AtomicBool,
    /// The [`run_worker`] task, aborted by [`shutdown`] so an interrupted
    /// transcription does not count as a crashed attempt.
    worker: std::sync::Mutex<Option<JoinHandle<()>>>,
    /// The folder of a start in progress, so a shutdown that cannot wait for
    /// the start can still tell whether it recorded anything.
    starting: std::sync::Mutex<Option<PathBuf>>,
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
            shutting_down: AtomicBool::new(false),
            worker: std::sync::Mutex::new(None),
            starting: std::sync::Mutex::new(None),
        };
        (runtime, status_rx, queue_rx)
    }

    /// Hand over the [`run_worker`] task, for [`shutdown`] to abort.
    pub(crate) fn set_worker(&self, worker: JoinHandle<()>) {
        *self.worker.lock().unwrap_or_else(|e| e.into_inner()) = Some(worker);
    }

    fn status(&self) -> MeetingStatus {
        self.view.lock().unwrap_or_else(|e| e.into_inner()).status()
    }

    /// Whether a meeting is recording. Reads the view, never the session
    /// lock, so it answers at once while a start or stop holds that lock for
    /// seconds opening or closing devices. A start counts once its devices
    /// are open, a stop until its tracks are finalized.
    pub(crate) fn is_recording(&self) -> bool {
        self.view
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .recording
            .is_some()
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

    /// Take meeting `id` off the view as recording, unless another meeting
    /// is shown by now: a stop finalizes its tracks (up to 15 s) after
    /// releasing the session lock, so a meeting may start meanwhile, and an
    /// unconditional clear would show it, and refuse dictation, as idle.
    fn clear_recording(&self, id: &str) {
        self.update(|v| {
            if v.recording.as_ref().is_some_and(|r| r.id == id) {
                v.recording = None;
            }
        });
    }

    /// Queue a meeting for processing, refusing one that is already queued
    /// or being processed.
    fn enqueue(&self, meeting: ActiveMeeting) -> Result<()> {
        self.reserve(&meeting)?;
        self.dispatch(meeting)
    }

    /// Claim a place in the queue, refusing a meeting that is already queued
    /// or being processed. The check and the claim are one step, so two
    /// retries of one meeting cannot both pass. Follow with [`Self::dispatch`]
    /// or [`Self::unreserve`].
    fn reserve(&self, meeting: &ActiveMeeting) -> Result<()> {
        let mut view = self.view.lock().unwrap_or_else(|e| e.into_inner());
        if view.is_pending(&meeting.id) {
            bail!("meeting {} is already queued for transcription", meeting.id);
        }
        view.queued.push(meeting.clone());
        Ok(())
    }

    fn unreserve(&self, id: &str) {
        self.update(|v| v.queued.retain(|q| q.id != id));
    }

    /// Hand a reserved meeting to the worker.
    fn dispatch(&self, meeting: ActiveMeeting) -> Result<()> {
        let id = meeting.id.clone();
        if self.queue_tx.send(QueuedMeeting { meeting }).is_err() {
            self.unreserve(&id);
            bail!("the meeting processing worker is not running");
        }
        self.update(|_| {});
        Ok(())
    }
}

/// Clears [`MeetingRuntime::starting`] however a start returns.
struct StartingGuard<'a>(&'a MeetingRuntime);

impl<'a> StartingGuard<'a> {
    fn new(runtime: &'a MeetingRuntime, dir: &Path) -> Self {
        *runtime.starting.lock().unwrap_or_else(|e| e.into_inner()) = Some(dir.to_path_buf());
        Self(runtime)
    }
}

impl Drop for StartingGuard<'_> {
    fn drop(&mut self) {
        *self.0.starting.lock().unwrap_or_else(|e| e.into_inner()) = None;
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

/// Find the system track's source name. Blocking. `None` when the meeting
/// records the mic only.
fn resolve_system(system: SystemSource) -> Option<Result<String>> {
    match system {
        SystemSource::Off => None,
        SystemSource::DefaultMonitor => Some(default_monitor_source()),
        SystemSource::Named(name) => Some(Ok(name)),
    }
}

/// Open the mic (required) and the system source (best effort), each with a
/// writer started as soon as its capture is up so the first-chunk times are
/// not skewed by the other device's start. Blocking: runs on the blocking
/// pool, where the runtime handle is available for the writers.
fn open_tracks(
    dir: &Path,
    mic: &str,
    system: Option<Result<String>>,
) -> Result<(Vec<TrackCapture>, Vec<TrackMeta>, Option<String>)> {
    let mut captures = Vec::new();
    let mut metas = Vec::new();

    let mut handle = AudioCaptureHandle::start_strict(mic)
        .with_context(|| format!("failed to open the microphone ({mic})"))?;
    let rx = handle
        .take_receiver()
        .context("mic capture has no receiver")?;
    let writer = spawn_track_writer(rx, store::audio_path(dir, TrackKind::Mic), FLUSH_EVERY);
    captures.push(TrackCapture::new(TrackKind::Mic, Some(handle), writer));
    metas.push(track_meta(TrackKind::Mic, mic.to_string()));

    let Some(system_name) = system else {
        return Ok((captures, metas, None));
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
            captures.push(TrackCapture::new(TrackKind::System, Some(handle), writer));
            metas.push(track_meta(TrackKind::System, name));
            Ok((captures, metas, None))
        }
        Err(e) => Ok((captures, metas, Some(format!("{e:#}")))),
    }
}

async fn start(context: &Arc<DaemonContext>, title: Option<String>) -> Result<()> {
    let mut slot = context.meeting.session.lock().await;
    if context.meeting.shutting_down.load(Ordering::Acquire) {
        bail!("the daemon is shutting down");
    }
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
    let _starting = StartingGuard::new(&context.meeting, &dir);

    let mic = mic_device(&meeting_cfg, &context.config.audio.device);
    let system = system_source(meeting_cfg.system_device.as_deref());
    let system = tokio::task::spawn_blocking(move || resolve_system(system))
        .await
        .unwrap_or_else(|e| Some(Err(anyhow::anyhow!("system source lookup panicked: {e}"))));

    // Written before any audio exists: a crash or SIGTERM while the devices
    // open leaves a meeting recovery can find, never an orphan audio folder.
    let mut meta = MeetingMeta::new(id.clone(), title, started_at);
    meta.tracks = vec![track_meta(TrackKind::Mic, mic.clone())];
    if let Some(Ok(name)) = &system {
        meta.tracks
            .push(track_meta(TrackKind::System, name.clone()));
    }
    if let Err(e) = store::write_meta(&dir, &meta) {
        remove_meeting_dir(&dir).await;
        return Err(e);
    }

    let started = Instant::now();
    let open_dir = dir.clone();
    let opened = tokio::task::spawn_blocking(move || open_tracks(&open_dir, &mic, system))
        .await
        .context("audio start panicked")
        .and_then(|r| r);
    let (mut tracks, track_metas, system_error) = match opened {
        Ok(opened) => opened,
        Err(e) => {
            remove_meeting_dir(&dir).await;
            return Err(e);
        }
    };
    let stopping = Arc::new(AtomicBool::new(false));
    for track in &mut tracks {
        watch_writer(track, &id, &stopping, context.notify);
    }

    meta.tracks = track_metas;
    if let Some(e) = &system_error {
        warn!("meeting {id}: system audio unavailable, recording the mic only: {e}");
        meta.errors.push(format!(
            "System audio unavailable, recorded the mic only: {e}"
        ));
    }
    if let Err(e) = store::write_meta(&dir, &meta) {
        drop(finish_tracks(tracks, &stopping, STOP_TIMEOUT, STOP_GRACE).await);
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
                let message = format!("{e:#}");
                warn!("failed to stop meeting {id} at its time limit: {message}");
                if !error_already_notified(&message) {
                    notify(
                        &ctx,
                        &format!(
                            "Meeting: failed to stop at the time limit: {}",
                            truncate_preview(&message, NOTIFY_ERROR_MAX)
                        ),
                    );
                }
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
        stopping,
    });
    // Shown before the session lock is released, so a stop that takes this
    // meeting right after cannot have its clear overwritten by this. When
    // both are held, the session lock is taken first; the view lock is never
    // held across an await.
    context.meeting.update(|v| v.recording = Some(active));
    drop(slot);

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
    /// The writer was still running after its timeout and grace: its file
    /// may still change, so nothing may repair or read it yet.
    stuck: bool,
    /// The writer ended before the stop (see [`TrackCapture::ended_early`]).
    ended_early: bool,
}

fn joined(
    result: Result<Result<TrackStats>, tokio::task::JoinError>,
) -> Result<TrackStats, String> {
    match result {
        Ok(Ok(stats)) => Ok(stats),
        Ok(Err(e)) => Err(format!("{e:#}")),
        Err(e) => Err(format!("writer task failed: {e}")),
    }
}

/// Mark the meeting as stopping, stop every capture, then wait (bounded)
/// for each writer to finalize its WAV. A writer that misses `limit` has its
/// capture dropped, which closes its channel, and gets `grace` more; one
/// still running after that is reported `stuck`. Capture threads are joined
/// on the blocking pool without waiting: they exit after the stop signal.
async fn finish_tracks(
    mut tracks: Vec<TrackCapture>,
    stopping: &AtomicBool,
    limit: Duration,
    grace: Duration,
) -> Vec<TrackOutcome> {
    stopping.store(true, Ordering::Release);
    for track in &mut tracks {
        if let Some(handle) = track.handle.as_mut() {
            handle.stop();
        }
    }
    let deadline = tokio::time::Instant::now() + limit;
    let mut outcomes = Vec::new();
    for track in tracks {
        let TrackCapture {
            kind,
            mut handle,
            mut writer,
            ended_early,
        } = track;
        let mut stuck = false;
        let stats = match tokio::time::timeout_at(deadline, &mut writer).await {
            Ok(result) => joined(result),
            Err(_) => {
                warn!(
                    "the {} track did not finish writing in {} s; closing its capture",
                    kind.file_stem(),
                    limit.as_secs()
                );
                if let Some(handle) = handle.take() {
                    tokio::task::spawn_blocking(move || drop(handle));
                }
                match tokio::time::timeout(grace, &mut writer).await {
                    Ok(result) => joined(result),
                    Err(_) => {
                        stuck = true;
                        Err("the track was still being written after the stop".to_string())
                    }
                }
            }
        };
        if let Some(handle) = handle {
            tokio::task::spawn_blocking(move || drop(handle));
        }
        outcomes.push(TrackOutcome {
            kind,
            stats,
            stuck,
            ended_early: ended_early.load(Ordering::Acquire),
        });
    }
    outcomes
}

/// Fold the writers' results into the meta: sample counts, offsets, and a
/// note for any track that did not finalize cleanly.
fn apply_outcomes(meta: &mut MeetingMeta, started: Instant, outcomes: &[TrackOutcome]) {
    for outcome in outcomes {
        if outcome.ended_early {
            let note = format!(
                "The {} stopped delivering audio before the meeting ended",
                track_label(outcome.kind)
            );
            if !meta.errors.contains(&note) {
                meta.errors.push(note);
            }
        }
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
/// sample count off the file. A stuck writer's file is left alone: it may
/// still be written to.
fn repair_unfinished(dir: &Path, meta: &mut MeetingMeta, outcomes: &[TrackOutcome]) {
    for outcome in outcomes.iter().filter(|o| o.stats.is_err() && !o.stuck) {
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
        stopping,
    } = session;
    // The timer is the caller on a max-duration stop: aborting it there
    // would cancel this very stop at its next await.
    if let Some(timer) = timer {
        if reason != EndReason::MaxDuration {
            timer.abort();
        }
    }

    let outcomes = finish_tracks(tracks, &stopping, STOP_TIMEOUT, STOP_GRACE).await;
    apply_outcomes(&mut meta, started, &outcomes);
    repair_unfinished(&dir, &mut meta, &outcomes);
    end_recording(&mut meta, reason, Local::now());
    let stuck: Vec<&str> = outcomes
        .iter()
        .filter(|o| o.stuck)
        .map(|o| track_label(o.kind))
        .collect();
    if !stuck.is_empty() {
        // Processing now would read a file that is still being written.
        let message = format!(
            "{NOT_TRANSCRIBED}: the {} track was still being written {} s \
             after the stop; retry with: whisrs meeting retry {}",
            stuck.join(" and "),
            (STOP_TIMEOUT + STOP_GRACE).as_secs(),
            meta.id
        );
        meta.status = MeetingState::Failed;
        meta.error = Some(message.clone());
        let written = store::write_meta(&dir, &meta);
        context.meeting.clear_recording(&meta.id);
        written?;
        error!("meeting {}: {message}", meta.id);
        // Notified here for every caller (CLI, hotkey, tray, time limit);
        // they skip it via [`error_already_notified`], so one stop gives one
        // toast.
        notify(
            context,
            &format!("Meeting: {}", truncate_preview(&message, NOTIFY_ERROR_MAX)),
        );
        bail!(message);
    }
    let written = store::write_meta(&dir, &meta);
    let active = ActiveMeeting::from_meta(&dir, &meta);
    context.meeting.clear_recording(&meta.id);
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
    drop(finish_tracks(session.tracks, &session.stopping, STOP_TIMEOUT, STOP_GRACE).await);
    remove_meeting_dir(&session.dir).await;
    context.meeting.clear_recording(&session.meta.id);
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
    let meta = store::read_meta(&dir)?;
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
    // Claimed before the meta is touched: a retry of a meeting that is
    // queued or being processed must not rewrite it under the worker.
    let active = ActiveMeeting::from_meta(&dir, &meta);
    context.meeting.reserve(&active)?;
    // Repair both tracks first (cheap and idempotent: a crash leftover may
    // still have stale headers), then mark it `processing`. A manual retry
    // is a fresh start for the crash-loop guard.
    let repair_dir = dir.clone();
    let prepared = tokio::task::spawn_blocking(move || recover_dir(&repair_dir, true))
        .await
        .context("meeting repair panicked")
        .and_then(|r| r);
    if let Err(e) = prepared {
        context.meeting.unreserve(&meta.id);
        return Err(e);
    }
    context.meeting.dispatch(active)?;
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

    // Counted before the attempt, so one that takes the daemon down with it
    // still counts towards [`MAX_ATTEMPTS`] at the next start. A graceful
    // shutdown gives it back ([`stop_worker`]).
    if let Err(e) = count_attempt(&dir) {
        error!("failed to record a processing attempt for meeting {id}: {e:#}");
    }
    // A panic in a backend or in the processing itself must not leave the
    // meeting shown as processing forever, nor kill the worker and with it
    // every later meeting. Caught in this task (not a spawned one) so a
    // shutdown that aborts the worker still drops the work with it.
    let result = AssertUnwindSafe(process_meeting(
        &dir,
        &context.config,
        Some(Arc::clone(&context.transcription_backend)),
    ))
    .catch_unwind()
    .await
    .unwrap_or_else(|panic| {
        Err(anyhow::anyhow!(
            "processing crashed: {}",
            panic_message(&*panic)
        ))
    });
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

/// The text of a caught panic's payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string())
}

fn count_attempt(dir: &Path) -> Result<()> {
    let mut meta = store::read_meta(dir)?;
    meta.attempts = meta.attempts.saturating_add(1);
    store::write_meta(dir, &meta)
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
    if meta.attempts >= MAX_ATTEMPTS {
        meta.status = MeetingState::Failed;
        meta.error = Some(format!(
            "processing was interrupted {} times; retry with whisrs meeting retry {}",
            meta.attempts, meta.id
        ));
    }
}

/// Repair an interrupted meeting's WAVs and update its meta. Blocking.
/// `reset_attempts` (a manual retry) zeroes the crash-loop count first.
fn recover_dir(dir: &Path, reset_attempts: bool) -> Result<MeetingMeta> {
    let mut meta = store::read_meta(dir)?;
    if reset_attempts {
        meta.attempts = 0;
    }
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

/// Recover one interrupted meeting and queue it. It is reserved in the
/// queue before anything on disk is touched, like a retry: a retry that
/// already holds it makes this skip it, and a retry arriving meanwhile is
/// refused, so the two never repair or requeue it twice.
async fn recover_one(context: &Arc<DaemonContext>, dir: PathBuf) {
    let meta = match store::read_meta(&dir) {
        Ok(meta) => meta,
        Err(e) => {
            warn!("failed to recover meeting {}: {e:#}", dir.display());
            return;
        }
    };
    let active = ActiveMeeting::from_meta(&dir, &meta);
    if context.meeting.reserve(&active).is_err() {
        info!(
            "meeting {} is already queued (retried meanwhile); not recovering it",
            meta.id
        );
        return;
    }
    let id = meta.id;
    let repair_dir = dir.clone();
    let recovered = tokio::task::spawn_blocking(move || recover_interrupted(&repair_dir))
        .await
        .context("meeting recovery panicked")
        .and_then(|r| r);
    let meta = match recovered {
        Ok(Some(meta)) => meta,
        Ok(None) => {
            // A retry finished it between the listing and the reservation.
            context.meeting.unreserve(&id);
            return;
        }
        Err(e) => {
            context.meeting.unreserve(&id);
            warn!("failed to recover meeting {}: {e:#}", dir.display());
            return;
        }
    };
    if meta.status == MeetingState::Failed {
        context.meeting.unreserve(&id);
        warn!(
            "not resuming meeting {id}: {}",
            meta.error.as_deref().unwrap_or_default()
        );
        notify(
            context,
            &format!(
                "Meeting transcription stopped after {} interrupted attempts. \
                 Retry with: whisrs meeting retry {id}",
                meta.attempts
            ),
        );
        return;
    }
    info!("resuming interrupted meeting {id} ({:?})", meta.end_reason);
    if let Err(e) = context
        .meeting
        .dispatch(ActiveMeeting::from_meta(&dir, &meta))
    {
        warn!("failed to queue meeting {id}: {e:#}");
    }
}

/// [`recover_dir`] for a meeting still interrupted on disk. `None` when it
/// no longer is (a retry processed it after the startup listing), so a
/// finished meeting is never flipped back to processing. Blocking.
fn recover_interrupted(dir: &Path) -> Result<Option<MeetingMeta>> {
    let meta = store::read_meta(dir)?;
    if !matches!(
        meta.status,
        MeetingState::Recording | MeetingState::Processing
    ) {
        return Ok(None);
    }
    recover_dir(dir, false).map(Some)
}

/// At startup: requeue every meeting a crash, reboot or shutdown left
/// recording or processing, then apply audio retention. Spawned, so the
/// IPC server comes up without waiting for it.
pub(crate) async fn recover_on_startup(context: Arc<DaemonContext>) {
    let root = meetings_root(&context.config);
    // Listed under the session lock: a start in progress finishes first and
    // its meeting is skipped below, and one that begins after the listing
    // creates a folder the listing does not have. Recovering it would repair
    // and requeue a recording that is still being written.
    let (interrupted, recording) = {
        let slot = context.meeting.session.lock().await;
        let recording = slot.as_ref().map(|s| s.dir.clone());
        let scan_root = root.clone();
        let interrupted = tokio::task::spawn_blocking(move || store::interrupted(&scan_root))
            .await
            .unwrap_or_default();
        (interrupted, recording)
    };
    // Oldest first, so they are processed in the order they happened.
    for dir in interrupted.into_iter().rev() {
        if recording.as_ref() == Some(&dir) {
            continue;
        }
        recover_one(&context, dir).await;
    }
    sweep(&context).await;
}

/// On SIGTERM/SIGINT: finalize a recording meeting (bounded) and leave it
/// `processing` with `end_reason: shutdown`, so the next start transcribes
/// it. A meeting mid-processing is already `processing` on disk and is
/// resumed the same way.
///
/// A start in progress holds the session lock while its devices open; this
/// waits for it ([`SHUTDOWN_LOCK_TIMEOUT`]) and then stops what it started.
/// A start that was waiting behind it on the lock refuses to run.
pub(crate) async fn shutdown(context: &DaemonContext) {
    context.meeting.shutting_down.store(true, Ordering::Release);
    stop_worker(&context.meeting).await;
    let mut slot =
        match tokio::time::timeout(SHUTDOWN_LOCK_TIMEOUT, context.meeting.session.lock()).await {
            Ok(slot) => slot,
            Err(_) => {
                abandon_start(&context.meeting).await;
                return;
            }
        };
    let Some(session) = slot.take() else {
        return;
    };
    let MeetingSession {
        dir,
        mut meta,
        started,
        tracks,
        timer,
        stopping,
    } = session;
    if let Some(timer) = timer {
        timer.abort();
    }
    let outcomes = finish_tracks(tracks, &stopping, SHUTDOWN_TIMEOUT, SHUTDOWN_GRACE).await;
    apply_outcomes(&mut meta, started, &outcomes);
    end_recording(&mut meta, EndReason::Shutdown, Local::now());
    match store::write_meta(&dir, &meta) {
        Ok(()) => info!(
            "meeting {} saved at shutdown; it will be transcribed on the next start",
            meta.id
        ),
        Err(e) => error!("failed to save meeting {} at shutdown: {e:#}", meta.id),
    }
    context.meeting.clear_recording(&meta.id);
}

/// Abort the processing worker, so an interrupted transcription is resumed
/// on the next start without counting as a crashed attempt: the attempt it
/// counted up front ([`process_one`]) is given back once the task is gone.
async fn stop_worker(runtime: &MeetingRuntime) {
    let worker = runtime
        .worker
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take();
    let Some(worker) = worker else {
        return;
    };
    worker.abort();
    match tokio::time::timeout(WORKER_ABORT_TIMEOUT, worker).await {
        Err(_) => {
            // Still running: giving the attempt back could race its writes.
            warn!("the meeting processing worker did not stop in time");
            return;
        }
        // Only an attempt this abort interrupted is given back. A worker
        // that panicked was not interrupted by the shutdown: its attempt is
        // the crash the crash-loop guard counts.
        Ok(Err(e)) if e.is_cancelled() => {}
        Ok(Err(e)) => {
            warn!("the meeting processing worker had failed: {e}");
            return;
        }
        // The queue closed and the worker ended on its own: nothing was
        // interrupted.
        Ok(Ok(())) => return,
    }
    // Read only now, with the task gone, so the view is final.
    let processing = runtime
        .view
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .processing
        .as_ref()
        .map(|p| (p.id.clone(), p.dir.clone()));
    let Some((id, dir)) = processing else {
        return;
    };
    match tokio::task::spawn_blocking(move || uncount_attempt(&dir)).await {
        Ok(Ok(true)) => info!(
            "meeting {id}: transcription interrupted by shutdown; it resumes on the next start"
        ),
        Ok(Ok(false)) => {}
        Ok(Err(e)) => warn!("meeting {id}: failed to give back the interrupted attempt: {e:#}"),
        Err(e) => warn!("meeting {id}: attempt bookkeeping panicked: {e}"),
    }
}

/// Undo [`count_attempt`] for a meeting a shutdown interrupted. Only a
/// meeting still `processing` is touched: one that finished or failed
/// before the abort keeps its count. Blocking.
fn uncount_attempt(dir: &Path) -> Result<bool> {
    let mut meta = store::read_meta(dir)?;
    if meta.status != MeetingState::Processing || meta.attempts == 0 {
        return Ok(false);
    }
    meta.attempts = meta.attempts.saturating_sub(1);
    store::write_meta(dir, &meta)?;
    Ok(true)
}

/// Whether any track of the meeting in `dir` holds a sample on disk.
fn has_audio_frames(dir: &Path) -> bool {
    [TrackKind::Mic, TrackKind::System].iter().any(|kind| {
        std::fs::metadata(store::audio_path(dir, *kind)).is_ok_and(|m| m.len() > WAV_HEADER_LEN)
    })
}

/// Shutdown could not wait for a start still opening its devices. With no
/// audio written yet there is nothing to transcribe, so its folder is
/// deleted instead of being left for recovery to fail on; with audio, the
/// `meeting.json` the start wrote first lets the next start recover it.
async fn abandon_start(runtime: &MeetingRuntime) {
    let starting = runtime
        .starting
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    match starting {
        Some(dir) if !has_audio_frames(&dir) => {
            warn!(
                "a meeting start was still opening devices at shutdown; deleting {} (no audio yet)",
                dir.display()
            );
            remove_meeting_dir(&dir).await;
        }
        _ => warn!(
            "a meeting start is still in progress at shutdown; it will be recovered on the next start"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command_mode::{handle_command_mode, handle_llm_command};
    use crate::context::{CommandModeContext, DaemonState, LlmCommandContext};
    use crate::dictation::{handle_cancel, handle_toggle};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use whisrs::meeting::Segment;
    use whisrs::state::Action;
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
                stuck: false,
                ended_early: false,
            },
            TrackOutcome {
                kind: TrackKind::System,
                stats: Err("boom".into()),
                stuck: false,
                ended_early: false,
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
        backend: Arc<dyn TranscriptionBackend>,
    ) -> (Arc<DaemonContext>, mpsc::UnboundedReceiver<QueuedMeeting>) {
        test_context_with(root, backend, "")
    }

    /// [`test_context`] with `extra` appended to its config.
    fn test_context_with(
        root: &Path,
        backend: Arc<dyn TranscriptionBackend>,
        extra: &str,
    ) -> (Arc<DaemonContext>, mpsc::UnboundedReceiver<QueuedMeeting>) {
        let config: Config = toml::from_str(&format!(
            "[general]\nbackend = \"groq\"\n[meeting]\nuser_name = \"Tester\"\ndir = \"{}\"\n{extra}",
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
        let (context, mut queue) = test_context(tmp.path(), backend.clone());

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
    }

    fn working_backend() -> Arc<FakeBackend> {
        Arc::new(FakeBackend {
            text: "fine",
            fail: false,
            calls: AtomicUsize::new(0),
        })
    }

    #[tokio::test]
    async fn shutdown_with_nothing_recording_is_a_no_op() {
        let tmp = tempfile::tempdir().unwrap();
        let (context, mut queue) = test_context(tmp.path(), working_backend());
        tokio::time::timeout(Duration::from_secs(1), shutdown(&context))
            .await
            .expect("an idle shutdown returns at once");
        assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 0);
        assert!(queue.try_recv().is_err());
        assert_eq!(context.meeting.status(), MeetingStatus::default());
        // A start after shutdown began is refused.
        let err = start(&context, None).await.unwrap_err();
        assert!(err.to_string().contains("shutting down"), "{err:#}");
    }

    #[tokio::test]
    async fn shutdown_saves_a_recording_for_the_next_start() {
        let tmp = tempfile::tempdir().unwrap();
        let (context, _queue) = test_context(tmp.path(), working_backend());
        let (id, dir) = store::create(tmp.path(), Local::now(), Some("standup")).unwrap();
        let mut meta = MeetingMeta::new(id.clone(), Some("standup".into()), Local::now());
        meta.tracks = vec![track_meta(TrackKind::Mic, "default".into())];
        store::write_meta(&dir, &meta).unwrap();

        // A track fed by the test instead of a device.
        let (tx, rx) = mpsc::unbounded_channel();
        let writer = spawn_track_writer(rx, store::audio_path(&dir, TrackKind::Mic), FLUSH_EVERY);
        tx.send(vec![1000i16; 3200]).unwrap();
        drop(tx);
        let started = Instant::now();
        let stopping = Arc::new(AtomicBool::new(false));
        *context.meeting.session.lock().await = Some(MeetingSession {
            dir: dir.clone(),
            meta: meta.clone(),
            started,
            tracks: vec![TrackCapture::new(TrackKind::Mic, None, writer)],
            timer: None,
            stopping,
        });
        context
            .meeting
            .update(|v| v.recording = Some(ActiveMeeting::from_meta(&dir, &meta)));

        shutdown(&context).await;

        let saved = store::read_meta(&dir).unwrap();
        assert_eq!(saved.status, MeetingState::Processing);
        assert_eq!(saved.end_reason, Some(EndReason::Shutdown));
        assert!(saved.ended_at.is_some());
        assert_eq!(saved.tracks[0].samples, 3200);
        assert!(context.meeting.session.lock().await.is_none());
        assert_eq!(context.meeting.status().phase, MeetingPhase::Idle);
        // And the next start picks it up.
        let interrupted = store::interrupted(tmp.path());
        assert_eq!(interrupted, vec![dir]);
    }

    #[tokio::test]
    async fn repeated_interruptions_end_in_failed() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = crashed_meeting(tmp.path());
        let backend = working_backend();
        let (context, mut queue) = test_context(tmp.path(), backend.clone());

        // Each "crash" is: the worker counts an attempt, then the daemon dies
        // before finishing; the next start recovers it.
        for attempt in 1..=MAX_ATTEMPTS {
            recover_on_startup(Arc::clone(&context)).await;
            let item = queue.try_recv().expect("requeued while under the limit");
            context.meeting.update(|v| v.queued.clear());
            count_attempt(&item.meeting.dir).unwrap();
            assert_eq!(store::read_meta(&dir).unwrap().attempts, attempt);
        }
        recover_on_startup(Arc::clone(&context)).await;
        assert!(queue.try_recv().is_err(), "not requeued after the limit");
        let meta = store::read_meta(&dir).unwrap();
        assert_eq!(meta.status, MeetingState::Failed);
        assert_eq!(
            meta.error.as_deref(),
            Some(
                format!(
                    "processing was interrupted 3 times; retry with whisrs meeting retry {}",
                    meta.id
                )
                .as_str()
            )
        );
        assert_eq!(backend.calls.load(Ordering::Relaxed), 0);

        // A manual retry resets the count and processes it.
        retry(&context, "last").await.unwrap();
        assert_eq!(store::read_meta(&dir).unwrap().attempts, 0);
        process_one(&context, queue.try_recv().unwrap()).await;
        let meta = store::read_meta(&dir).unwrap();
        assert_eq!(meta.status, MeetingState::Done);
        assert_eq!(meta.attempts, 1);
    }

    #[tokio::test]
    async fn stuck_writer_is_reported_not_raced() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("mic.wav");
        // The test holds the sender: with no capture to drop, the writer
        // cannot finish.
        let (tx, rx) = mpsc::unbounded_channel();
        let writer = spawn_track_writer(rx, path.clone(), FLUSH_EVERY);
        let stopping = AtomicBool::new(false);
        let outcomes = finish_tracks(
            vec![TrackCapture::new(TrackKind::Mic, None, writer)],
            &stopping,
            Duration::from_millis(50),
            Duration::from_millis(50),
        )
        .await;
        assert!(stopping.load(Ordering::Acquire));
        assert_eq!(outcomes.len(), 1);
        assert!(outcomes[0].stuck);
        assert!(outcomes[0].stats.is_err());

        // repair_unfinished leaves a stuck track's file alone.
        let mut meta = MeetingMeta::new("m".into(), None, Local::now());
        meta.tracks = vec![track_meta(TrackKind::Mic, "mic".into())];
        let before = std::fs::read(&path).unwrap_or_default();
        repair_unfinished(tmp.path(), &mut meta, &outcomes);
        assert_eq!(std::fs::read(&path).unwrap_or_default(), before);
        drop(tx);
    }

    #[tokio::test]
    async fn a_track_that_ends_before_the_stop_is_noted() {
        let tmp = tempfile::tempdir().unwrap();
        let (tx, rx) = mpsc::unbounded_channel();
        let writer = spawn_track_writer(rx, tmp.path().join("system.wav"), FLUSH_EVERY);
        let mut track = TrackCapture::new(TrackKind::System, None, writer);
        let stopping = Arc::new(AtomicBool::new(false));
        watch_writer(&mut track, "m", &stopping, false);
        tx.send(vec![5i16; 1600]).unwrap();
        // The device goes away mid-meeting.
        drop(tx);
        let early = Arc::clone(&track.ended_early);
        let deadline = Instant::now() + Duration::from_secs(5);
        while !early.load(Ordering::Acquire) {
            assert!(Instant::now() < deadline, "early end never noticed");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let outcomes = finish_tracks(vec![track], &stopping, STOP_TIMEOUT, STOP_GRACE).await;
        assert!(outcomes[0].ended_early);
        assert!(!outcomes[0].stuck);
        let mut meta = MeetingMeta::new("m".into(), None, Local::now());
        meta.tracks = vec![track_meta(TrackKind::System, "sys".into())];
        apply_outcomes(&mut meta, Instant::now(), &outcomes);
        assert_eq!(meta.tracks[0].samples, 1600);
        assert_eq!(
            meta.errors,
            ["The system audio stopped delivering audio before the meeting ended"]
        );

        // A writer that ends after the stop is a normal end.
        let (tx, rx) = mpsc::unbounded_channel::<whisrs::audio::AudioChunk>();
        let writer = spawn_track_writer(rx, tmp.path().join("mic.wav"), FLUSH_EVERY);
        let mut track = TrackCapture::new(TrackKind::Mic, None, writer);
        let stopping = Arc::new(AtomicBool::new(false));
        watch_writer(&mut track, "m", &stopping, false);
        stopping.store(true, Ordering::Release);
        drop(tx);
        let outcomes = finish_tracks(vec![track], &stopping, STOP_TIMEOUT, STOP_GRACE).await;
        assert!(!outcomes[0].ended_early);
    }

    /// A backend whose requests never finish, like a long transcription.
    struct HangingBackend;

    #[async_trait::async_trait]
    impl TranscriptionBackend for HangingBackend {
        async fn transcribe(&self, _audio: &[u8], _config: &TranscriptionConfig) -> Result<String> {
            std::future::pending().await
        }
        fn sends_prompt(&self, _config: &TranscriptionConfig) -> bool {
            false
        }
        async fn transcribe_segments(
            &self,
            _audio: &[u8],
            _config: &TranscriptionConfig,
        ) -> Result<Transcript> {
            std::future::pending().await
        }
        fn max_segment_chunk_secs(&self, _config: &TranscriptionConfig) -> Option<u32> {
            None
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_mid_processing_gives_the_attempt_back() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = crashed_meeting(tmp.path());
        let (context, queue) = test_context(tmp.path(), Arc::new(HangingBackend));
        recover_on_startup(Arc::clone(&context)).await;
        context
            .meeting
            .set_worker(tokio::spawn(run_worker(Arc::clone(&context), queue)));

        // Wait until the worker has counted its attempt and is transcribing.
        let deadline = Instant::now() + Duration::from_secs(5);
        while store::read_meta(&dir).unwrap().attempts != 1
            || context.meeting.status().queued.len() == 1
        {
            assert!(Instant::now() < deadline, "the worker never started");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;

        shutdown(&context).await;
        let meta = store::read_meta(&dir).unwrap();
        assert_eq!(meta.status, MeetingState::Processing);
        assert_eq!(meta.attempts, 0, "a graceful shutdown is not a crash");
        assert!(context.meeting.worker.lock().unwrap().is_none());

        // The next start resumes it and it completes.
        let (next, mut queue) = test_context(tmp.path(), working_backend());
        recover_on_startup(Arc::clone(&next)).await;
        process_one(&next, queue.try_recv().expect("resumed")).await;
        let meta = store::read_meta(&dir).unwrap();
        assert_eq!(meta.status, MeetingState::Done);
        assert_eq!(meta.attempts, 1);
    }

    #[tokio::test]
    async fn retry_of_a_pending_meeting_leaves_its_meta_alone() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = crashed_meeting(tmp.path());
        let (context, mut queue) = test_context(tmp.path(), working_backend());
        recover_on_startup(Arc::clone(&context)).await;
        let item = queue.try_recv().unwrap();
        // The worker took it and counted two attempts so far.
        context.meeting.update(|v| {
            v.queued.clear();
            v.processing = Some(item.meeting.clone());
        });
        let mut meta = store::read_meta(&dir).unwrap();
        meta.attempts = 2;
        store::write_meta(&dir, &meta).unwrap();

        let err = retry(&context, "last").await.unwrap_err();
        assert!(err.to_string().contains("already queued"), "{err:#}");
        assert_eq!(store::read_meta(&dir).unwrap(), meta);
        assert!(queue.try_recv().is_err());
    }

    #[tokio::test]
    async fn retry_repairs_the_tracks_first() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = crashed_meeting(tmp.path());
        // A crash leftover that was marked failed without a repair.
        let mut meta = store::read_meta(&dir).unwrap();
        meta.status = MeetingState::Failed;
        meta.attempts = 3;
        store::write_meta(&dir, &meta).unwrap();
        let (context, mut queue) = test_context(tmp.path(), working_backend());

        retry(&context, "last").await.unwrap();
        let meta = store::read_meta(&dir).unwrap();
        assert_eq!(meta.status, MeetingState::Processing);
        assert_eq!(meta.attempts, 0);
        assert_eq!(meta.tracks[0].samples, 16_000);
        let reader = hound::WavReader::open(store::audio_path(&dir, TrackKind::Mic)).unwrap();
        assert_eq!(reader.duration(), 16_000);
        assert!(queue.try_recv().is_ok());
    }

    #[tokio::test]
    async fn recovery_skips_the_meeting_recording_now() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = crashed_meeting(tmp.path());
        let (context, mut queue) = test_context(tmp.path(), working_backend());
        let meta = store::read_meta(&dir).unwrap();
        *context.meeting.session.lock().await = Some(MeetingSession {
            dir: dir.clone(),
            meta: meta.clone(),
            started: Instant::now(),
            tracks: Vec::new(),
            timer: None,
            stopping: Arc::new(AtomicBool::new(false)),
        });

        recover_on_startup(Arc::clone(&context)).await;
        assert!(queue.try_recv().is_err());
        assert_eq!(store::read_meta(&dir).unwrap(), meta);
    }

    /// A backend that panics mid-transcription.
    struct PanickingBackend;

    #[async_trait::async_trait]
    impl TranscriptionBackend for PanickingBackend {
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
            panic!("backend exploded");
        }
        fn max_segment_chunk_secs(&self, _config: &TranscriptionConfig) -> Option<u32> {
            None
        }
    }

    #[tokio::test]
    async fn a_panic_in_processing_fails_the_meeting_and_keeps_the_worker() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = crashed_meeting(tmp.path());
        let (context, queue) = test_context(tmp.path(), Arc::new(PanickingBackend));
        recover_on_startup(Arc::clone(&context)).await;
        let worker = tokio::spawn(run_worker(Arc::clone(&context), queue));

        let deadline = Instant::now() + Duration::from_secs(5);
        while store::read_meta(&dir).unwrap().status != MeetingState::Failed
            || context.meeting.status().phase != MeetingPhase::Idle
        {
            assert!(Instant::now() < deadline, "the panic was never handled");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let meta = store::read_meta(&dir).unwrap();
        let error = meta.error.as_deref().unwrap_or_default();
        assert!(
            error.starts_with("processing crashed: ") && error.contains("backend exploded"),
            "{error}"
        );
        assert_eq!(meta.attempts, 1);
        assert!(!worker.is_finished(), "the worker survives the panic");

        // And it still takes work: a retry is processed (and panics) again.
        retry(&context, "last").await.unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while store::read_meta(&dir).unwrap().status != MeetingState::Failed {
            assert!(Instant::now() < deadline, "the retry was never processed");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        worker.abort();
    }

    #[tokio::test]
    async fn shutdown_after_a_worker_panic_keeps_the_attempt() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = crashed_meeting(tmp.path());
        let (context, mut queue) = test_context(tmp.path(), working_backend());
        recover_on_startup(Arc::clone(&context)).await;
        let item = queue.try_recv().unwrap();
        context.meeting.update(|v| {
            v.queued.clear();
            v.processing = Some(item.meeting.clone());
        });
        count_attempt(&dir).unwrap();

        let worker = tokio::spawn(async { panic!("worker died") });
        while !worker.is_finished() {
            tokio::task::yield_now().await;
        }
        context.meeting.set_worker(worker);
        stop_worker(&context.meeting).await;
        assert_eq!(
            store::read_meta(&dir).unwrap().attempts,
            1,
            "a panic is not an interruption"
        );
    }

    #[tokio::test]
    async fn recovery_skips_a_meeting_a_retry_holds() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = crashed_meeting(tmp.path());
        let (context, mut queue) = test_context(tmp.path(), working_backend());
        let before = store::read_meta(&dir).unwrap();
        // A retry reserved it and is repairing it.
        context
            .meeting
            .reserve(&ActiveMeeting::from_meta(&dir, &before))
            .unwrap();

        recover_on_startup(Arc::clone(&context)).await;
        assert!(queue.try_recv().is_err(), "not queued twice");
        assert_eq!(
            store::read_meta(&dir).unwrap(),
            before,
            "not repaired twice"
        );
        assert_eq!(context.meeting.status().queued, vec![before.id.clone()]);

        // A meeting recovery holds refuses a retry.
        context.meeting.unreserve(&before.id);
        recover_on_startup(Arc::clone(&context)).await;
        let err = retry(&context, "last").await.unwrap_err();
        assert!(err.to_string().contains("already queued"), "{err:#}");
        assert!(queue.try_recv().is_ok());
        assert!(queue.try_recv().is_err());
    }

    #[test]
    fn recovery_leaves_a_finished_meeting_alone() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = crashed_meeting(tmp.path());
        let mut meta = store::read_meta(&dir).unwrap();
        meta.status = MeetingState::Done;
        store::write_meta(&dir, &meta).unwrap();
        assert!(recover_interrupted(&dir).unwrap().is_none());
        assert_eq!(store::read_meta(&dir).unwrap(), meta);
    }

    #[test]
    fn a_not_transcribed_stop_is_notified_once() {
        let message = format!("{NOT_TRANSCRIBED}: the microphone track was still being written");
        assert!(error_already_notified(&message));
        assert!(!error_already_notified("no meeting is recording"));
    }

    #[tokio::test]
    async fn an_abandoned_start_is_deleted_only_without_audio() {
        let tmp = tempfile::tempdir().unwrap();
        let (context, _queue) = test_context(tmp.path(), working_backend());

        // Devices still opening: meeting.json and a header-only track.
        let (_, empty) = store::create(tmp.path(), Local::now(), Some("empty")).unwrap();
        std::fs::write(
            store::audio_path(&empty, TrackKind::Mic),
            whisrs::audio::wav::encode_wav(&[]).unwrap(),
        )
        .unwrap();
        {
            let _starting = StartingGuard::new(&context.meeting, &empty);
            abandon_start(&context.meeting).await;
        }
        assert!(!empty.exists());

        // Audio already written: kept for recovery.
        let with_audio = crashed_meeting(tmp.path());
        {
            let _starting = StartingGuard::new(&context.meeting, &with_audio);
            abandon_start(&context.meeting).await;
        }
        assert!(with_audio.exists());
        assert!(context.meeting.starting.lock().unwrap().is_none());
    }

    /// One `[[llm_commands]]` entry, for the dictation refusal tests.
    const GERMAN: &str = "[[llm_commands]]\nname = \"german\"\n\
                          hotkey = \"Super+Shift+G\"\ninstruction = \"Translate to German.\"\n";

    fn idle_dictation() -> Arc<tokio::sync::Mutex<DaemonState>> {
        Arc::new(tokio::sync::Mutex::new(DaemonState::new()))
    }

    fn assert_refused(response: &Response, subject: &str) {
        let Response::Error { message } = response else {
            panic!("expected a refusal, got {response:?}");
        };
        assert_eq!(
            message,
            &format!("A meeting is recording. {subject} is off until you stop it.")
        );
    }

    /// A toggle from idle while a meeting records is refused before it
    /// opens the mic or moves the state machine. It carries a bad `-l`, the
    /// next check after the refusal, so a broken refusal fails here on the
    /// language error instead of opening the host's mic.
    #[tokio::test]
    async fn dictation_is_refused_while_a_meeting_records() {
        let tmp = tempfile::tempdir().unwrap();
        let (context, _queue) = test_context(tmp.path(), working_backend());
        context
            .meeting
            .update(|v| v.recording = Some(active("rec")));
        let ds = idle_dictation();

        let response = handle_toggle(
            Arc::clone(&ds),
            Arc::clone(&context),
            Some("english".into()),
        )
        .await;
        assert_refused(&response, "Dictation");
        let ds = ds.lock().await;
        assert_eq!(ds.state_machine.state(), State::Idle);
        assert!(ds.audio_capture.is_none(), "no capture is opened");
        assert!(ds.recording_started_at.is_none());
        assert!(ds.session_language.is_none());
    }

    /// The refusal for every mode: an error naming the mode while a meeting
    /// records, nothing while it is idle or being transcribed. Command mode
    /// and llm-command are tested here and by source order below, not
    /// through their handlers: with no seam before the selection and the
    /// mic, a broken refusal would copy the host's selection (sending Ctrl+C
    /// when it is empty) and open its mic.
    #[test]
    fn the_refusal_names_each_mode_and_fires_only_while_recording() {
        let tmp = tempfile::tempdir().unwrap();
        let (context, _queue) = test_context(tmp.path(), working_backend());
        let modes = [
            (DictationStart::Dictation, "Dictation"),
            (DictationStart::CommandMode, "Command mode"),
            (DictationStart::LlmCommand("german"), "'german'"),
        ];

        for (start, _) in modes {
            assert!(
                refuse_while_recording(&context, start).is_none(),
                "{start:?}"
            );
        }
        context
            .meeting
            .update(|v| v.recording = Some(active("rec")));
        for (start, subject) in modes {
            let response = refuse_while_recording(&context, start)
                .unwrap_or_else(|| panic!("{start:?} was not refused"));
            assert_refused(&response, subject);
        }
        context.meeting.update(|v| {
            v.recording = None;
            v.processing = Some(active("rec"));
        });
        for (start, _) in modes {
            assert!(
                refuse_while_recording(&context, start).is_none(),
                "{start:?}"
            );
        }
    }

    /// The llm-command refusal comes after the entry lookup, so a misspelled
    /// name keeps its config error during a meeting. An unknown name never
    /// reaches a device, so the handler is safe to call here.
    #[tokio::test]
    async fn an_unknown_llm_command_keeps_its_config_error() {
        let tmp = tempfile::tempdir().unwrap();
        let (context, _queue) = test_context_with(tmp.path(), working_backend(), GERMAN);
        context
            .meeting
            .update(|v| v.recording = Some(active("rec")));
        let ds = idle_dictation();

        let response =
            handle_llm_command(Arc::clone(&ds), Arc::clone(&context), "nope".into()).await;
        assert!(
            matches!(&response, Response::Error { message } if message.contains("no llm_commands entry")),
            "{response:?}"
        );
        assert_eq!(ds.lock().await.state_machine.state(), State::Idle);
    }

    /// Calls a refused start must not reach: the mic, the start cue, the
    /// selection (whose capture may send Ctrl+C), the focused window, and
    /// the two start fns that make those calls.
    const DEVICE_CALLS: &[&str] = &[
        "AudioCaptureHandle::",
        "feedback::play_start(",
        "capture_selection(",
        "acquire_selected_text(",
        "get_focused_window(",
        "command_mode_start(",
        "llm_command_start(",
    ];

    /// `source` before its test module, without comment lines, so neither
    /// prose nor the strings in a test can match.
    fn production_code(source: &str) -> String {
        source
            .split("#[cfg(test)]")
            .next()
            .expect("the file has a production half")
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The top-level fn starting at `signature`, up to its closing brace.
    fn fn_body<'a>(code: &'a str, signature: &str) -> &'a str {
        let start = code
            .find(signature)
            .unwrap_or_else(|| panic!("{signature} not found"));
        let rest = &code[start..];
        &rest[..rest.find("\n}").expect("the fn ends")]
    }

    /// Each start handler calls `refuse_while_recording` before anything in
    /// [`DEVICE_CALLS`]. This pins command mode and llm-command, whose
    /// handlers no test may run with a broken refusal; toggle is pinned too.
    /// The two start fns are private and called once each, from their
    /// handler, so nothing reaches them around the refusal.
    #[test]
    fn every_start_is_refused_before_it_touches_a_device() {
        let dictation = production_code(include_str!("dictation.rs"));
        let command_mode = production_code(include_str!("command_mode.rs"));
        let cases = [
            (&dictation, "fn handle_toggle(", "DictationStart::Dictation"),
            (
                &command_mode,
                "fn handle_command_mode(",
                "DictationStart::CommandMode",
            ),
            (
                &command_mode,
                "fn handle_llm_command(",
                "DictationStart::LlmCommand(",
            ),
        ];
        for (code, signature, mode) in cases {
            let body = fn_body(code, signature);
            let refusal = body
                .find(&format!("refuse_while_recording(&context, {mode}"))
                .unwrap_or_else(|| panic!("{signature} does not refuse {mode}"));
            for call in DEVICE_CALLS {
                assert!(
                    !body[..refusal].contains(call),
                    "{signature} reaches {call} before the meeting refusal"
                );
            }
            assert!(
                DEVICE_CALLS
                    .iter()
                    .any(|call| body[refusal..].contains(call)),
                "{signature}: the start path after the refusal was not found"
            );
        }
        for start_fn in ["command_mode_start(", "llm_command_start("] {
            assert_eq!(
                command_mode.matches(start_fn).count(),
                2,
                "{start_fn} must be defined once and called once, by its handler"
            );
            assert!(command_mode.contains(&format!("async fn {start_fn}")));
        }
    }

    /// Only starts are refused. A dictation recording when the meeting
    /// started stops and goes on to transcription: with no device in the
    /// test it has no capture, so the batch path fails to collect audio and
    /// returns to idle. Command-mode and llm-command sessions still stop on
    /// their second press and cancel.
    #[tokio::test]
    async fn a_session_started_before_the_meeting_still_stops() {
        let tmp = tempfile::tempdir().unwrap();
        let (context, _queue) = test_context_with(tmp.path(), working_backend(), GERMAN);
        let ds = idle_dictation();
        ds.lock()
            .await
            .state_machine
            .transition(Action::Toggle)
            .unwrap();
        context
            .meeting
            .update(|v| v.recording = Some(active("rec")));

        let response = handle_toggle(Arc::clone(&ds), Arc::clone(&context), None).await;
        assert!(
            matches!(response, Response::Ok { state: State::Idle }),
            "{response:?}"
        );
        assert_eq!(ds.lock().await.state_machine.state(), State::Idle);

        {
            let mut ds = ds.lock().await;
            ds.state_machine.transition(Action::Toggle).unwrap();
            ds.command_mode = Some(CommandModeContext {
                selected_text: "selected".into(),
                llm_config: Default::default(),
            });
        }
        let response = handle_command_mode(Arc::clone(&ds), Arc::clone(&context)).await;
        assert!(
            matches!(
                response,
                Response::Ok {
                    state: State::Recording
                }
            ),
            "{response:?}"
        );
        let response = handle_cancel(Arc::clone(&ds), Arc::clone(&context)).await;
        assert!(
            matches!(response, Response::Ok { state: State::Idle }),
            "{response:?}"
        );

        {
            let mut ds = ds.lock().await;
            ds.state_machine.transition(Action::Toggle).unwrap();
            ds.llm_command = Some(LlmCommandContext {
                name: "german".into(),
                instruction: "Translate to German.".into(),
                llm_config: Default::default(),
            });
        }
        let response =
            handle_llm_command(Arc::clone(&ds), Arc::clone(&context), "german".into()).await;
        assert!(
            matches!(
                response,
                Response::Ok {
                    state: State::Recording
                }
            ),
            "{response:?}"
        );
        let response = handle_cancel(Arc::clone(&ds), Arc::clone(&context)).await;
        assert!(
            matches!(response, Response::Ok { state: State::Idle }),
            "{response:?}"
        );
        assert!(context.meeting.is_recording(), "the meeting is untouched");
    }

    /// Only a recording meeting refuses: one being transcribed or queued
    /// does not. The start carries a bad `-l`, so it fails at the language
    /// check right after the refusal, before any device opens; that error,
    /// not the meeting's, shows the start got past the refusal.
    #[tokio::test]
    async fn a_meeting_being_transcribed_does_not_block_dictation() {
        let tmp = tempfile::tempdir().unwrap();
        let (context, _queue) = test_context(tmp.path(), working_backend());
        let ds = idle_dictation();

        context
            .meeting
            .update(|v| v.processing = Some(active("proc")));
        assert_start_reaches_the_language_check(&ds, &context).await;

        context.meeting.update(|v| {
            v.processing = None;
            v.queued = vec![active("next")];
        });
        assert_start_reaches_the_language_check(&ds, &context).await;
    }

    async fn assert_start_reaches_the_language_check(
        ds: &Arc<tokio::sync::Mutex<DaemonState>>,
        context: &Arc<DaemonContext>,
    ) {
        assert_eq!(context.meeting.status().phase, MeetingPhase::Processing);
        let response =
            handle_toggle(Arc::clone(ds), Arc::clone(context), Some("english".into())).await;
        let Response::Error { message } = &response else {
            panic!("expected the language error, got {response:?}");
        };
        assert!(
            message.starts_with("invalid language 'english'"),
            "{message}"
        );
        assert_eq!(ds.lock().await.state_machine.state(), State::Idle);
    }

    /// A recording session with its folder and meta on disk and a mic track
    /// fed by the returned sender instead of a device: its writer finishes
    /// only once the sender is dropped, like a slow device at a stop.
    fn fed_session(
        root: &Path,
        title: &str,
    ) -> (
        MeetingSession,
        mpsc::UnboundedSender<whisrs::audio::AudioChunk>,
    ) {
        let (id, dir) = store::create(root, Local::now(), Some(title)).unwrap();
        let mut meta = MeetingMeta::new(id, Some(title.into()), Local::now());
        meta.tracks = vec![track_meta(TrackKind::Mic, "default".into())];
        store::write_meta(&dir, &meta).unwrap();
        let (tx, rx) = mpsc::unbounded_channel();
        let writer = spawn_track_writer(rx, store::audio_path(&dir, TrackKind::Mic), FLUSH_EVERY);
        let session = MeetingSession {
            dir,
            meta,
            started: Instant::now(),
            tracks: vec![TrackCapture::new(TrackKind::Mic, None, writer)],
            timer: None,
            stopping: Arc::new(AtomicBool::new(false)),
        };
        (session, tx)
    }

    /// Install `session` the way `start` does once its devices are open.
    async fn show_recording(context: &DaemonContext, session: MeetingSession) {
        let mut slot = context.meeting.session.lock().await;
        let active = ActiveMeeting::from_meta(&session.dir, &session.meta);
        *slot = Some(session);
        context.meeting.update(|v| v.recording = Some(active));
    }

    /// A stop finalizes its tracks after releasing the session lock, so the
    /// next meeting can start meanwhile. The first stop's late clear must
    /// leave that meeting shown as recording, and dictation refused.
    #[tokio::test]
    async fn a_slow_stop_does_not_hide_the_next_meeting() {
        let tmp = tempfile::tempdir().unwrap();
        let (context, mut queue) = test_context(tmp.path(), working_backend());
        let (first, first_tx) = fed_session(tmp.path(), "first");
        let first_id = first.meta.id.clone();
        show_recording(&context, first).await;

        let stopping = tokio::spawn({
            let context = Arc::clone(&context);
            async move { stop(&context, None, EndReason::Stopped).await }
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let taken = context.meeting.session.lock().await.is_none();
            if taken {
                break;
            }
            assert!(Instant::now() < deadline, "the stop never took the session");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        // The next meeting starts while the first one's track is finalizing.
        let (next, next_tx) = fed_session(tmp.path(), "next");
        let next_id = next.meta.id.clone();
        show_recording(&context, next).await;
        drop(first_tx);
        stopping.await.unwrap().unwrap();

        assert_eq!(queue.try_recv().unwrap().meeting.id, first_id);
        assert!(context.meeting.is_recording(), "the next meeting is hidden");
        assert_eq!(
            context.meeting.status().id.as_deref(),
            Some(next_id.as_str())
        );
        assert!(refuse_while_recording(&context, DictationStart::Dictation).is_some());

        // Its own stop still clears it.
        drop(next_tx);
        stop(&context, None, EndReason::Stopped).await.unwrap();
        assert!(!context.meeting.is_recording());
    }
}
