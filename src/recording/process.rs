//! Recording post-processing: transcribe both tracks and build the transcript.
//!
//! [`process_recording`] runs once a recording has stopped (and its WAVs have
//! been repaired by the caller). The speakers' echo is first taken out of
//! the mic track ([`super::echo`]). Only the speech in each track is sent
//! ([`super::speech`]): its active stretches are packed into condensed,
//! backend-sized chunks, each chunk is transcribed into timed segments, and
//! the segments are mapped back, filtered and moved onto the recording
//! timeline. Mic segments that still echo the system track are dropped, and
//! the two channels are labelled and merged. The result lands in
//! `recording.json` and `<id>.md`.

use std::ffi::CStr;
use std::fs;
use std::io::Write;
use std::ops::Range;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context};
use audio_silence_gate::rms_energy;
use filler_remove::FillerFilter;
use prompt_echo::is_prompt_echo;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use super::echo::{cancel_echo, EchoOutcome};
use super::render::render_markdown;
use super::speech::{self, TimeMap};
use super::split::frame_len;
use super::store::{self, CACHE_DIR};
use super::{RecordingMeta, RecordingState, Segment, Speaker, TrackKind, TranscriptSegment};
use crate::audio::wav::encode_wav;
use crate::replacements::Replacer;
use crate::transcription::factory::{
    backend_identity, canonical_backend_name, create_backend_named, recording_backend_name,
    recording_model_label, recording_transcription_config,
};
use crate::transcription::retry::{
    is_network_unavailable, retry_rate_limited, RateLimitExhausted, RATE_LIMIT_BUDGET,
};
use crate::transcription::{TranscriptionBackend, TranscriptionConfig};
use crate::{Config, RecordingConfig};

/// Recording tracks are recorded at this rate (and `encode_wav` writes it).
const SAMPLE_RATE: u32 = super::SAMPLE_RATE;

/// How far apart a mic segment and the system audio it echoes may be.
pub const ECHO_TOLERANCE_MS: u64 = 1500;

/// The echo-cancelled mic track, under [`CACHE_DIR`]: a retry reuses it,
/// and it goes with the chunk cache once the transcript is written.
const ECHO_CANCELLED_MIC: &str = "mic-echo-cancelled.wav";

/// Its near-end mask ([`speech::encode_mask`]), beside it in the cache.
const ECHO_CANCELLED_MASK: &str = "mic-echo-cancelled.mask";

/// Transcribe a stopped recording and write its transcript.
///
/// Marks the recording `processing`, transcribes every recorded track with the
/// recording backend ([`recording_backend_name`]), each in the language the
/// recording stored for it, and on success writes `recording.json` with
/// status `done`, the backend and the model, plus `<id>.md`, returning the
/// new meta. `dictation_backend` is reused when the recording backend is the
/// dictation one, so a loaded whisper model is not loaded twice. On error the
/// meta is left as `processing`; the caller marks it failed.
pub async fn process_recording(
    dir: &Path,
    config: &Config,
    dictation_backend: Option<Arc<dyn TranscriptionBackend>>,
) -> anyhow::Result<RecordingMeta> {
    let mut meta = store::read_meta(dir)?;
    meta.status = RecordingState::Processing;
    meta.error = None;
    fill_languages(&mut meta, config);
    store::write_meta(dir, &meta)?;

    let name = recording_backend_name(config)?;
    let backend = match dictation_backend {
        // `local` and `local-whisper` are one backend: reuse its loaded model.
        Some(backend)
            if canonical_backend_name(&name) == canonical_backend_name(&config.general.backend) =>
        {
            backend
        }
        _ => {
            // Loading a local model can take seconds; keep it off the runtime.
            let owned = config.clone();
            let backend_name = name.clone();
            tokio::task::spawn_blocking(move || create_backend_named(&backend_name, &owned))
                .await
                .context("backend construction panicked")?
        }
    };
    // Both are set by `fill_languages`; the fallback is never taken.
    let language = |kind| {
        meta.language(kind)
            .unwrap_or(&config.general.language)
            .to_string()
    };
    let mic_request = TrackRequest::new(&name, config, &language(TrackKind::Mic), dir);
    let system_request = TrackRequest::new(&name, config, &language(TrackKind::System), dir);
    // Only the language differs between the two requests, so the model and
    // the prompt are the mic's.
    let model = recording_model_label(&name, config, &mic_request.tcfg);
    info!(
        "processing recording {} with {name} (model {model}, mic in {}, system audio in {})",
        meta.id, mic_request.tcfg.language, system_request.tcfg.language
    );

    let filler = if config.general.remove_filler_words {
        Some(
            FillerFilter::new(&config.general.filler_words)
                .context("invalid custom filler word in configuration")?,
        )
    } else {
        None
    };
    let replacer = Replacer::new(&config.replacements)
        .context("invalid [replacements] key in configuration")?;
    let filters = SegmentFilters {
        prompt: backend
            .sends_prompt(&mic_request.tcfg)
            .then(|| mic_request.tcfg.prompt.clone())
            .flatten(),
        filler,
        replacer,
    };

    let echo_cache = mic_request.cache.echo_cancelled();
    let cancelled_mic = echo_cancelled_mic(dir, &meta).await;
    let mut mic = Vec::new();
    let mut system = Vec::new();
    let mut readable_tracks = 0usize;
    for track in meta.tracks.clone() {
        let path = dir.join(&track.file);
        if !path.exists() {
            let note = format!("{} audio missing: {}", track.kind.file_stem(), track.file);
            warn!("recording {}: {note}", meta.id);
            if !meta.errors.contains(&note) {
                meta.errors.push(note);
            }
            continue;
        }
        readable_tracks += 1;
        let request = match track.kind {
            TrackKind::Mic => &mic_request,
            TrackKind::System => &system_request,
        };
        // Same length as the raw mic, so the track offset still holds.
        let (path, track_cache, near_end) = match (&cancelled_mic, track.kind) {
            (Some(cancelled), TrackKind::Mic) => (
                cancelled.path.clone(),
                &echo_cache,
                Some(cancelled.near_end.as_slice()),
            ),
            _ => (path, &request.cache, None),
        };
        let job = TrackJob {
            path: &path,
            kind: track.kind,
            offset_ms: track.offset_ms,
            near_end,
        };
        let segments =
            transcribe_track(backend.as_ref(), &request.tcfg, &filters, track_cache, &job)
                .await
                .with_context(|| {
                    format!("failed to transcribe the {} track", track.kind.file_stem())
                })?;
        info!(
            "recording {}: {} track ({}) gave {} segment(s)",
            meta.id,
            track.kind.file_stem(),
            request.tcfg.language,
            segments.len()
        );
        match track.kind {
            TrackKind::Mic => mic.extend(segments),
            TrackKind::System => system.extend(segments),
        }
    }
    if readable_tracks == 0 {
        bail!("no recorded audio found for recording {}", meta.id);
    }

    let before = mic.len();
    let mic = drop_echo_segments(mic, &system, ECHO_TOLERANCE_MS);
    if mic.len() < before {
        info!(
            "recording {}: dropped {} mic segment(s) echoing the system audio",
            meta.id,
            before - mic.len()
        );
    }

    meta.speakers = speakers_for(&meta, config);
    meta.segments = merge_channels(mic, system);
    meta.backend = Some(name);
    meta.model = Some(model);
    meta.status = RecordingState::Done;
    meta.error = None;
    meta.network_wait = None;

    write_private(
        &store::transcript_path(dir, &meta.id),
        render_markdown(&meta).as_bytes(),
    )?;
    store::write_meta(dir, &meta)?;
    // One folder holds both tracks' chunks.
    mic_request.cache.clear();
    Ok(meta)
}

/// Give a track with no stored language (a recording made before languages
/// were stored) the one the config gives it ([`RecordingConfig::language_for`]),
/// so `recording.json` says what the transcript was made in and a later
/// retry keeps it.
fn fill_languages(meta: &mut RecordingMeta, config: &Config) {
    let recording = RecordingConfig::effective(config);
    let general = &config.general.language;
    meta.mic_language
        .get_or_insert_with(|| recording.language_for(TrackKind::Mic, general));
    meta.system_language
        .get_or_insert_with(|| recording.language_for(TrackKind::System, general));
}

/// One track's request and the cache of its finished chunks. The two
/// tracks' requests differ only in their language, which the fingerprint
/// covers: a change to one track's language leaves the other track's
/// cached chunks valid.
struct TrackRequest {
    tcfg: TranscriptionConfig,
    cache: ChunkCache,
}

impl TrackRequest {
    fn new(name: &str, config: &Config, language: &str, dir: &Path) -> Self {
        let tcfg = recording_transcription_config(name, config, language);
        let cache = ChunkCache {
            dir: dir.join(CACHE_DIR),
            fingerprint: cache_fingerprint(name, config, &tcfg),
        };
        Self { tcfg, cache }
    }
}

/// The echo-cancelled mic and where the user speaks in it.
struct CancelledMic {
    path: PathBuf,
    /// [`speech::FRAME`] activity from the canceller's near-end detector.
    near_end: Vec<bool>,
}

/// The mic track with the speakers' echo removed ([`cancel_echo`]), or
/// `None` to transcribe the mic as recorded: with no system track, when the
/// system track is silent or the mic does not pick it up, and on any error,
/// which is logged rather than failing the recording. `audio/mic.wav` itself
/// is never modified.
async fn echo_cancelled_mic(dir: &Path, meta: &RecordingMeta) -> Option<CancelledMic> {
    let track = |kind| meta.tracks.iter().find(|t| t.kind == kind);
    let (mic, system) = (track(TrackKind::Mic)?, track(TrackKind::System)?);
    let mic_path = dir.join(&mic.file);
    let system_path = dir.join(&system.file);
    if !mic_path.exists() || !system_path.exists() {
        return None;
    }
    // Mic sample `u` and system sample `u - lag` were captured together.
    let timeline_lag = (system.offset_ms - mic.offset_ms) * i64::from(SAMPLE_RATE) / 1000;
    let out = dir.join(CACHE_DIR).join(ECHO_CANCELLED_MIC);
    let step = {
        let out = out.clone();
        tokio::task::spawn_blocking(move || {
            cancel_into_cache(&mic_path, &system_path, timeline_lag, &out)
        })
        .await
        .context("echo cancellation panicked")
        .and_then(|step| step)
    };
    let id = &meta.id;
    match step {
        Ok(EchoStep::Reused(near_end)) => {
            info!("recording {id}: reusing the echo-cancelled mic track");
            Some(CancelledMic {
                path: out,
                near_end,
            })
        }
        Ok(EchoStep::Ran(EchoOutcome::Cancelled(done))) => {
            let report = &done.report;
            let ms = |samples: i64| samples * 1000 / i64::from(SAMPLE_RATE);
            info!(
                "recording {id}: removed the speaker echo from the mic track \
                 (lag {} ms, drift {} ms over {} block(s), echo path {:.1} dB)",
                ms(report.lag),
                ms(report.lag_drift),
                report.aligned_blocks,
                report.echo_path_db
            );
            Some(CancelledMic {
                path: out,
                near_end: done.near_end,
            })
        }
        Ok(EchoStep::Ran(EchoOutcome::Skipped(reason))) => {
            info!("recording {id}: no speaker echo to remove: {reason}");
            None
        }
        Err(e) => {
            warn!(
                "recording {id}: echo cancellation failed, transcribing the mic as recorded: {e:#}"
            );
            None
        }
    }
}

/// What [`cancel_into_cache`] found or did.
enum EchoStep {
    /// An earlier attempt finished the cancelled track and its near-end mask.
    Reused(Vec<bool>),
    Ran(EchoOutcome),
}

/// Reuse `out` and its mask when an earlier attempt finished both (each the
/// mic's length, the mask from this detector version), else run
/// [`cancel_echo`] into a temporary file, rename the track into place, and
/// only then write the mask (atomically, 0600). A stale mask is removed
/// first, so a crash at any point leaves at most a track without a mask,
/// which is made again, never a mask beside a track it was not made from.
/// Blocking.
fn cancel_into_cache(
    mic: &Path,
    system: &Path,
    timeline_lag: i64,
    out: &Path,
) -> anyhow::Result<EchoStep> {
    let mic_len = open_track(mic)?.duration();
    let mask_path = out.with_file_name(ECHO_CANCELLED_MASK);
    if open_track(out).is_ok_and(|done| done.duration() == mic_len) {
        let mask = fs::read(&mask_path)
            .ok()
            .and_then(|bytes| speech::decode_mask(&bytes, mic_len as usize));
        if let Some(mask) = mask {
            return Ok(EchoStep::Reused(mask));
        }
    }
    let cache_dir = out.parent().context("cache path has no parent")?;
    fs::create_dir_all(cache_dir)
        .with_context(|| format!("failed to create {}", cache_dir.display()))?;
    fs::set_permissions(cache_dir, fs::Permissions::from_mode(0o700))?;
    match fs::remove_file(&mask_path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e).context(format!("failed to delete {}", mask_path.display())),
    }
    let mut tmp_name = out.file_name().unwrap_or_default().to_os_string();
    tmp_name.push(".tmp");
    let tmp = out.with_file_name(tmp_name);
    let outcome = cancel_echo(mic, system, timeline_lag, &tmp).and_then(|outcome| {
        if let EchoOutcome::Cancelled(done) = &outcome {
            fs::rename(&tmp, out)
                .with_context(|| format!("failed to rename {} into place", tmp.display()))?;
            write_private(
                &mask_path,
                &speech::encode_mask(mic_len as usize, &done.near_end),
            )?;
        }
        Ok(outcome)
    });
    // Gone after a rename; otherwise never leave a mic-sized file behind.
    if let Err(e) = fs::remove_file(&tmp) {
        if e.kind() != std::io::ErrorKind::NotFound {
            warn!("failed to delete {}: {e}", tmp.display());
        }
    }
    Ok(EchoStep::Ran(outcome?))
}

/// What every transcribed segment is checked against.
struct SegmentFilters {
    /// The prompt as sent on the wire, `None` when the backend does not send
    /// one (then nothing can echo it).
    prompt: Option<String>,
    filler: Option<FillerFilter>,
    /// `[replacements]`, applied after filler removal as on the dictation
    /// paths.
    replacer: Option<Replacer>,
}

impl SegmentFilters {
    /// The cleaned text to keep, or `None` to drop the segment.
    fn apply(&self, text: &str) -> Option<String> {
        let text = text.trim();
        if text.is_empty() {
            return None;
        }
        if let Some(prompt) = &self.prompt {
            if is_prompt_echo(text, prompt) {
                debug!("dropping recording segment as a prompt echo: {text:?}");
                return None;
            }
        }
        let text = match &self.filler {
            Some(filter) => filter.apply(text),
            None => text.to_string(),
        };
        let text = match &self.replacer {
            Some(replacer) => replacer.apply(text.trim()),
            None => text,
        };
        let text = text.trim();
        (!text.is_empty()).then(|| text.to_string())
    }
}

/// Open a 16 kHz mono 16-bit track.
pub(super) fn open_track(
    path: &Path,
) -> anyhow::Result<hound::WavReader<std::io::BufReader<fs::File>>> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != SAMPLE_RATE || spec.bits_per_sample != 16 {
        bail!(
            "{} is {} ch / {} Hz / {} bit, expected mono 16 kHz 16-bit",
            path.display(),
            spec.channels,
            spec.sample_rate,
            spec.bits_per_sample
        );
    }
    Ok(reader)
}

/// What one pass over a track learns, without keeping its samples: a
/// three-hour track is ~350 MB of samples but ~1 MB of frame energies.
#[derive(Debug, Clone, PartialEq)]
struct TrackScan {
    /// Samples in the track.
    len: usize,
    /// RMS of every frame ([`frame_len`]), the last one partial when the
    /// track does not end on a frame: one per [`speech::FRAME`] of activity,
    /// and what [`speech::pack`] cuts long regions by.
    energies: Vec<f64>,
}

/// Read a track once, computing its length and energies. Blocking.
fn scan_track(path: &Path) -> anyhow::Result<TrackScan> {
    let mut reader = open_track(path)?;
    let frame = frame_len(SAMPLE_RATE as usize);
    let mut buf = Vec::with_capacity(frame);
    let mut energies = Vec::new();
    let mut len = 0usize;
    for sample in reader.samples::<i16>() {
        let sample = sample.with_context(|| format!("failed to read {}", path.display()))?;
        len += 1;
        buf.push(sample);
        if buf.len() == frame {
            energies.push(rms_energy(&buf));
            buf.clear();
        }
    }
    if !buf.is_empty() {
        energies.push(rms_energy(&buf));
    }
    Ok(TrackScan { len, energies })
}

/// Read samples `range` of a track. Blocking.
fn read_range(path: &Path, range: Range<usize>) -> anyhow::Result<Vec<i16>> {
    let mut reader = open_track(path)?;
    let start = u32::try_from(range.start).context("track too long")?;
    reader
        .seek(start)
        .with_context(|| format!("failed to seek in {}", path.display()))?;
    reader
        .samples::<i16>()
        .take(range.len())
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("failed to read {}", path.display()))
}

/// Run a blocking track read off the async runtime.
async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> anyhow::Result<T> + Send + 'static,
) -> anyhow::Result<T> {
    tokio::task::spawn_blocking(f)
        .await
        .context("track read panicked")?
}

/// The condensed audio of one chunk: its pieces read from the track, with
/// [`speech::JOIN_GAP`] of digital silence between them. Blocking.
fn read_condensed(path: &Path, map: &TimeMap) -> anyhow::Result<Vec<i16>> {
    let mut out = Vec::with_capacity(map.condensed_len());
    for piece in &map.pieces {
        out.resize(piece.condensed, 0);
        let samples = read_range(path, piece.original..piece.original + piece.len)?;
        anyhow::ensure!(
            samples.len() == piece.len,
            "{} ended before sample {}",
            path.display(),
            piece.original + piece.len
        );
        out.extend(samples);
    }
    Ok(out)
}

/// Where a track holds speech, per [`speech::FRAME`]: its frame energy at
/// the gate's level, and for the echo-cancelled mic also its near-end mask.
/// The mask alone passes the canceller's in-band floor, which hiss from a
/// noisy mic can clear while the far side is quiet; the energy gate is the
/// one every other track is held to, so the cancelled mic is held to it too.
fn speech_activity(near_end: Option<&[bool]>, energies: &[f64], stem: &str) -> Vec<bool> {
    let energy = speech::energy_mask(energies);
    match near_end {
        Some(mask) if mask.len() == energy.len() => mask
            .iter()
            .zip(&energy)
            .map(|(&near, &loud)| near && loud)
            .collect(),
        Some(mask) => {
            warn!(
                "{stem} near-end mask has {} frames for a track of {}; finding speech by \
                 energy alone",
                mask.len(),
                energy.len()
            );
            energy
        }
        None => energy,
    }
}

/// The track being transcribed.
struct TrackJob<'a> {
    path: &'a Path,
    kind: TrackKind,
    offset_ms: i64,
    /// Where the user speaks, from the echo canceller, for the cancelled
    /// mic; every other track goes by frame energy.
    near_end: Option<&'a [bool]>,
}

/// Transcribe the speech in one track into filtered segments on the
/// recording timeline.
///
/// Peak memory is one chunk: the track is scanned once for its frame
/// energies, its speech found ([`speech::regions`]) and packed into
/// condensed chunks ([`speech::pack`]), and each chunk is read from disk
/// only when it is sent. A track without speech sends nothing. A chunk
/// already in `cache` is not sent again.
async fn transcribe_track(
    backend: &dyn TranscriptionBackend,
    tcfg: &TranscriptionConfig,
    filters: &SegmentFilters,
    cache: &ChunkCache,
    job: &TrackJob<'_>,
) -> anyhow::Result<Vec<Segment>> {
    let path: PathBuf = job.path.to_path_buf();
    let scan = {
        let path = path.clone();
        blocking(move || scan_track(&path)).await?
    };
    let stem = job.kind.file_stem();
    let active = speech_activity(job.near_end, &scan.energies, stem);
    let regions = speech::regions(&active, scan.len);
    if regions.is_empty() {
        debug!("{stem} track holds no speech; skipping");
        return Ok(Vec::new());
    }
    let chunks = speech::pack(
        &regions,
        &scan.energies,
        backend
            .max_segment_chunk_secs(tcfg)
            .map(|secs| secs as usize),
    );
    let speech_samples: usize = regions.iter().map(|r| r.len()).sum();
    info!(
        "{stem} track: {:.1} s of speech in {} region(s) of {:.1} s, {} chunk(s)",
        speech_samples as f64 / f64::from(SAMPLE_RATE),
        regions.len(),
        scan.len as f64 / f64::from(SAMPLE_RATE),
        chunks.len()
    );
    drop((scan, active, regions));

    let mut out = Vec::new();
    for map in chunks {
        let raw = match cache.load(job.kind, &map) {
            Some(segments) => {
                debug!(
                    "reusing a cached {stem} chunk from sample {}",
                    map.pieces[0].original
                );
                segments
            }
            None => {
                let chunk = {
                    let path = path.clone();
                    let map = map.clone();
                    blocking(move || read_condensed(&path, &map)).await?
                };
                let wav = encode_wav(&chunk)?;
                drop(chunk);
                let transcript = transcribe_with_retry(backend, &wav, tcfg).await?;
                if let Err(e) = cache.store(job.kind, &map, &transcript.segments) {
                    warn!("could not cache a transcribed chunk: {e:#}");
                }
                transcript.segments
            }
        };
        let offset_ms = job.offset_ms;
        let shift = |ms: u64| (offset_ms + ms as i64).max(0) as u64;
        for seg in raw.into_iter().flat_map(|seg| map.map_segment(seg)) {
            let Some(text) = filters.apply(&seg.text) else {
                continue;
            };
            out.push(Segment {
                start_ms: shift(seg.start_ms),
                end_ms: shift(seg.end_ms),
                text,
                speaker: seg.speaker,
                words: seg.words.map(|words| {
                    words
                        .into_iter()
                        .map(|mut w| {
                            w.start_ms = shift(w.start_ms);
                            w.end_ms = shift(w.end_ms);
                            w
                        })
                        .collect()
                }),
            });
        }
    }
    Ok(out)
}

/// Transcribe one chunk, waiting out rate limits within
/// [`RATE_LIMIT_BUDGET`] and retrying a server or network error a few times
/// ([`retry_rate_limited`]). Phrase backends retry each phrase inside
/// `transcribe_segments` and give up with a [`RateLimitExhausted`] or a
/// [`NetworkUnavailable`](crate::transcription::retry::NetworkUnavailable),
/// which this layer does not retry again, so the two waits never multiply.
async fn transcribe_with_retry(
    backend: &dyn TranscriptionBackend,
    wav: &[u8],
    tcfg: &TranscriptionConfig,
) -> anyhow::Result<crate::transcription::Transcript> {
    retry_rate_limited(RATE_LIMIT_BUDGET, || backend.transcribe_segments(wav, tcfg))
        .await
        .map_err(|err| {
            if err.downcast_ref::<RateLimitExhausted>().is_some() || is_network_unavailable(&err) {
                err.context(
                    "chunks finished before the failure are saved; retry resumes from the failed chunk",
                )
            } else {
                err
            }
        })
}

/// A stable 64-bit FNV-1a hash, so cache keys do not depend on the
/// standard library's per-process hasher seeds.
fn fnv1a64(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Everything that decides what a chunk transcribes to: the canonical
/// backend, its model identity ([`backend_identity`]: the model file for
/// local whisper, the URL for the sidecar) and every request field. Hashed
/// rather than stored, so the prompt or a URL with a token in it is not
/// copied into the cache.
fn cache_fingerprint(name: &str, config: &Config, tcfg: &TranscriptionConfig) -> String {
    let key = serde_json::json!({
        // 2: chunks are condensed speech, keyed by their pieces, so entries
        // from the earlier range chunking never match.
        "chunking": 2,
        "backend": canonical_backend_name(name),
        "identity": backend_identity(name, config),
        "language": tcfg.language,
        "model": tcfg.model,
        "prompt": tcfg.prompt,
        "keyterms": tcfg.keyterms,
    });
    format!("{:016x}", fnv1a64(key.to_string().as_bytes()))
}

/// The raw segments of one finished chunk, as cached on disk.
#[derive(Debug, Serialize, Deserialize)]
struct CachedChunk {
    /// [`cache_fingerprint`] of the request that produced it.
    fingerprint: String,
    segments: Vec<Segment>,
}

/// Cache of finished chunks for one track of a recording and its request.
/// Entries hold the backend's segments before filtering, relative to the
/// chunk, so a config change to the filters still applies on reuse. An entry
/// is reused only when its fingerprint matches exactly: another backend,
/// model file, language or vocabulary ignores it (recordings send no other
/// prompt).
struct ChunkCache {
    dir: PathBuf,
    fingerprint: String,
}

impl ChunkCache {
    /// One file per track and chunk: a hash of the chunk's pieces, so the
    /// same speech packed the same way finds its entry again.
    fn path(&self, kind: TrackKind, map: &TimeMap) -> PathBuf {
        let pieces: Vec<u8> = map
            .pieces
            .iter()
            .flat_map(|p| [p.condensed, p.original, p.len])
            .flat_map(|n| (n as u64).to_le_bytes())
            .collect();
        self.dir.join(format!(
            "{}-{:016x}.json",
            kind.file_stem(),
            fnv1a64(&pieces)
        ))
    }

    fn load(&self, kind: TrackKind, map: &TimeMap) -> Option<Vec<Segment>> {
        let data = fs::read(self.path(kind, map)).ok()?;
        let cached: CachedChunk = serde_json::from_slice(&data).ok()?;
        (cached.fingerprint == self.fingerprint).then_some(cached.segments)
    }

    fn store(&self, kind: TrackKind, map: &TimeMap, segments: &[Segment]) -> anyhow::Result<()> {
        fs::create_dir_all(&self.dir)
            .with_context(|| format!("failed to create {}", self.dir.display()))?;
        fs::set_permissions(&self.dir, fs::Permissions::from_mode(0o700))?;
        let cached = CachedChunk {
            fingerprint: self.fingerprint.clone(),
            segments: segments.to_vec(),
        };
        write_private(&self.path(kind, map), &serde_json::to_vec(&cached)?)
    }

    /// The cache for chunks of the echo-cancelled mic. Its own fingerprint,
    /// so chunks transcribed from the raw mic in an earlier attempt are not
    /// reused for the cancelled one, or the other way round.
    fn echo_cancelled(&self) -> ChunkCache {
        let key = format!("{}/echo-cancelled", self.fingerprint);
        ChunkCache {
            dir: self.dir.clone(),
            fingerprint: format!("{:016x}", fnv1a64(key.as_bytes())),
        }
    }

    /// Drop the cache once the transcript is written.
    fn clear(&self) {
        match fs::remove_dir_all(&self.dir) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => warn!("failed to delete {}: {e}", self.dir.display()),
        }
    }
}

/// Drop mic segments that are the system audio leaking into the microphone.
///
/// Without headphones the far side of the call plays through the speakers and
/// is recorded twice. A mic segment is dropped when its text is an echo
/// ([`is_prompt_echo`]) of the system segments overlapping it within
/// `tol_ms`. Short mic segments are never dropped: `is_prompt_echo` ignores
/// anything under 8 characters, so a real "yes" over the other side survives.
pub fn drop_echo_segments(mic: Vec<Segment>, system: &[Segment], tol_ms: u64) -> Vec<Segment> {
    mic.into_iter()
        .filter(|m| {
            let lo = m.start_ms.saturating_sub(tol_ms);
            let hi = m.end_ms.saturating_add(tol_ms);
            let overlapping: Vec<&str> = system
                .iter()
                .filter(|s| s.start_ms <= hi && s.end_ms >= lo)
                .map(|s| s.text.as_str())
                .collect();
            overlapping.is_empty() || !is_prompt_echo(&m.text, &overlapping.join(" "))
        })
        .collect()
}

/// Label and merge both channels into the transcript, ordered by start time.
fn merge_channels(mic: Vec<Segment>, system: Vec<Segment>) -> Vec<TranscriptSegment> {
    let label = |seg: Segment, kind: TrackKind| TranscriptSegment {
        start_ms: seg.start_ms,
        end_ms: seg.end_ms,
        speaker: speaker_id(kind).to_string(),
        channel: kind,
        text: seg.text,
    };
    let mut merged: Vec<TranscriptSegment> = mic
        .into_iter()
        .map(|s| label(s, TrackKind::Mic))
        .chain(system.into_iter().map(|s| label(s, TrackKind::System)))
        .collect();
    merged.sort_by_key(|s| (s.start_ms, s.end_ms));
    merged
}

fn speaker_id(kind: TrackKind) -> &'static str {
    match kind {
        TrackKind::Mic => "me",
        TrackKind::System => "others",
    }
}

/// One speaker per recorded channel; the system one is named by the
/// recording's others label.
fn speakers_for(meta: &RecordingMeta, config: &Config) -> Vec<Speaker> {
    let has = |kind| meta.tracks.iter().any(|t| t.kind == kind);
    let mut speakers = Vec::new();
    if has(TrackKind::Mic) {
        speakers.push(Speaker {
            id: speaker_id(TrackKind::Mic).into(),
            name: user_name(&RecordingConfig::effective(config)),
            source: "self".into(),
        });
    }
    if has(TrackKind::System) {
        speakers.push(Speaker {
            id: speaker_id(TrackKind::System).into(),
            name: meta.others_label().into(),
            source: "channel".into(),
        });
    }
    speakers
}

/// Name for the "me" speaker: `[recording] user_name`, else the passwd full
/// name, else `$USER`, else "Me".
fn user_name(recording: &RecordingConfig) -> String {
    recording
        .user_name
        .as_deref()
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .map(str::to_string)
        .or_else(passwd_full_name)
        .or_else(|| std::env::var("USER").ok().filter(|u| !u.trim().is_empty()))
        .unwrap_or_else(|| "Me".to_string())
}

/// The full-name field of the current user's passwd entry.
fn passwd_full_name() -> Option<String> {
    let mut pwd: libc::passwd = unsafe { std::mem::zeroed() };
    let mut buf = vec![0 as libc::c_char; 16 * 1024];
    let mut result: *mut libc::passwd = std::ptr::null_mut();
    // SAFETY: every pointer is valid for the call, `buf` outlives every use of
    // the strings `pwd` points into, and `result` is checked before use.
    let rc = unsafe {
        libc::getpwuid_r(
            libc::getuid(),
            &mut pwd,
            buf.as_mut_ptr(),
            buf.len(),
            &mut result,
        )
    };
    if rc != 0 || result.is_null() || pwd.pw_gecos.is_null() {
        return None;
    }
    // SAFETY: non-null, NUL-terminated, and backed by `buf`.
    let gecos = unsafe { CStr::from_ptr(pwd.pw_gecos) }.to_string_lossy();
    gecos_full_name(&gecos)
}

/// GECOS is `Full Name,room,work phone,home phone,other`.
fn gecos_full_name(gecos: &str) -> Option<String> {
    gecos
        .split(',')
        .next()
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .map(str::to_string)
}

/// Write a file atomically with mode 0600, like `recording.json`.
fn write_private(path: &Path, data: &[u8]) -> anyhow::Result<()> {
    let mut tmp_name = path.file_name().unwrap_or_default().to_os_string();
    tmp_name.push(".tmp");
    let tmp = path.with_file_name(tmp_name);
    {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp)
            .with_context(|| format!("failed to open {}", tmp.display()))?;
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
        file.write_all(data)?;
        file.sync_all()?;
    }
    fs::rename(&tmp, path).with_context(|| format!("failed to rename {} into place", tmp.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recording::echo::fixtures;
    use crate::recording::split::split_by_energies;
    use crate::recording::TrackMeta;
    use crate::transcription::{RetryableHttpError, Transcript};
    use audio_silence_gate::SILENCE_RMS_THRESHOLD;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    const RATE: usize = SAMPLE_RATE as usize;

    /// Fake backend: every run of loud 100 ms frames in a chunk becomes one
    /// segment, and the run's peak amplitude picks its text from `texts`.
    /// Times are relative to the chunk, like a real backend's.
    struct FakeBackend {
        texts: Vec<(i16, &'static str)>,
        chunk_secs: Option<u32>,
        calls: AtomicUsize,
        /// Samples in each request that got through.
        sent: std::sync::Mutex<Vec<usize>>,
        /// The language of each request that got through.
        languages: std::sync::Mutex<Vec<String>>,
        /// Calls with this index or later fail (a dead network).
        fail_from: Option<usize>,
        /// The first this many calls answer HTTP 429.
        rate_limited: usize,
    }

    impl FakeBackend {
        fn new(texts: &[(i16, &'static str)], chunk_secs: Option<u32>) -> Arc<Self> {
            Self::with_failures(texts, chunk_secs, None, 0)
        }

        fn with_failures(
            texts: &[(i16, &'static str)],
            chunk_secs: Option<u32>,
            fail_from: Option<usize>,
            rate_limited: usize,
        ) -> Arc<Self> {
            Arc::new(Self {
                texts: texts.to_vec(),
                chunk_secs,
                calls: AtomicUsize::new(0),
                sent: Default::default(),
                languages: Default::default(),
                fail_from,
                rate_limited,
            })
        }
    }

    #[async_trait::async_trait]
    impl TranscriptionBackend for FakeBackend {
        async fn transcribe(
            &self,
            _audio: &[u8],
            _config: &TranscriptionConfig,
        ) -> anyhow::Result<String> {
            unreachable!("recordings ask for segments")
        }
        fn sends_prompt(&self, _config: &TranscriptionConfig) -> bool {
            false
        }
        async fn transcribe_segments(
            &self,
            audio: &[u8],
            config: &TranscriptionConfig,
        ) -> anyhow::Result<Transcript> {
            let call = self.calls.fetch_add(1, Ordering::Relaxed);
            if call < self.rate_limited {
                return Err(RetryableHttpError {
                    status: 429,
                    retry_after: Some(Duration::from_secs(7)),
                    message: "slow down".into(),
                }
                .into());
            }
            if self.fail_from.is_some_and(|n| call >= n) {
                bail!("network is down");
            }
            let samples = crate::transcription::decode_wav_samples(audio)?;
            self.sent.lock().unwrap().push(samples.len());
            self.languages.lock().unwrap().push(config.language.clone());
            let frame = RATE / 10;
            let mut segments = Vec::new();
            let mut run: Option<(usize, i16)> = None;
            let frames = samples.len().div_ceil(frame);
            for f in 0..=frames {
                let slice =
                    &samples[(f * frame).min(samples.len())..((f + 1) * frame).min(samples.len())];
                let peak = slice.iter().map(|s| s.saturating_abs()).max().unwrap_or(0);
                match (run, peak > 100) {
                    (None, true) => run = Some((f, peak)),
                    (Some((s, p)), true) => run = Some((s, p.max(peak))),
                    (Some((s, p)), false) => {
                        let text = self
                            .texts
                            .iter()
                            .find(|(amp, _)| *amp == p)
                            .map_or("unknown", |(_, t)| t);
                        segments.push(Segment {
                            start_ms: (s * 100) as u64,
                            end_ms: (f * 100) as u64,
                            text: text.to_string(),
                            speaker: None,
                            words: None,
                        });
                        run = None;
                    }
                    (None, false) => {}
                }
            }
            Ok(Transcript { segments })
        }
        fn max_segment_chunk_secs(&self, _config: &TranscriptionConfig) -> Option<u32> {
            self.chunk_secs
        }
    }

    /// `secs` of silence with `(start_s, len_s, amplitude)` bursts of a
    /// 2 kHz square wave: in the speech band, where the echo canceller's
    /// near-end detector listens, with RMS and peak both `amplitude`.
    fn track(secs: f64, bursts: &[(f64, f64, i16)]) -> Vec<i16> {
        let mut out = vec![0i16; (secs * RATE as f64) as usize];
        for &(start, len, amp) in bursts {
            let from = (start * RATE as f64) as usize;
            let to = ((start + len) * RATE as f64) as usize;
            for (i, s) in out[from..to].iter_mut().enumerate() {
                *s = if i % 8 < 4 { amp } else { -amp };
            }
        }
        out
    }

    fn write_wav(path: &Path, samples: &[i16]) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, encode_wav(samples).unwrap()).unwrap();
    }

    /// A stopped recording in a temp dir with the given tracks.
    fn recording(
        tracks: &[(TrackKind, i64, Option<Vec<i16>>)],
    ) -> (tempfile::TempDir, RecordingMeta) {
        let tmp = tempfile::tempdir().unwrap();
        let mut meta = RecordingMeta::new("2026-10-02_1400".into(), None, chrono::Local::now());
        for (kind, offset_ms, samples) in tracks {
            let file = format!("audio/{}.wav", kind.file_stem());
            if let Some(samples) = samples {
                write_wav(&tmp.path().join(&file), samples);
            }
            meta.tracks.push(TrackMeta {
                kind: *kind,
                file,
                device: "test".into(),
                offset_ms: *offset_ms,
                samples: samples.as_ref().map_or(0, |s| s.len() as u64),
            });
        }
        store::write_meta(tmp.path(), &meta).unwrap();
        (tmp, meta)
    }

    fn config() -> Config {
        toml::from_str(
            "[general]\nbackend = \"groq\"\nremove_filler_words = true\n\
             [recording]\nuser_name = \"Tester\"\n",
        )
        .unwrap()
    }

    async fn run(dir: &Path, backend: Arc<FakeBackend>) -> RecordingMeta {
        process_recording(
            dir,
            &config(),
            Some(backend as Arc<dyn TranscriptionBackend>),
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn two_tracks_are_merged_sorted_and_labelled() {
        let backend = FakeBackend::new(
            &[(3000, "um, hello from me"), (6000, "hello from them")],
            None,
        );
        let (tmp, _) = recording(&[
            (
                TrackKind::Mic,
                0,
                Some(track(8.0, &[(1.0, 1.0, 3000), (5.0, 1.0, 3000)])),
            ),
            (
                TrackKind::System,
                200,
                Some(track(8.0, &[(3.0, 1.0, 6000)])),
            ),
        ]);
        let meta = run(tmp.path(), backend).await;

        assert_eq!(meta.status, RecordingState::Done);
        assert_eq!(meta.backend.as_deref(), Some("groq"));
        let lines: Vec<_> = meta
            .segments
            .iter()
            .map(|s| (s.start_ms, s.speaker.as_str(), s.text.as_str()))
            .collect();
        assert_eq!(
            lines,
            [
                (1000, "me", "hello from me"),
                (3200, "others", "hello from them"),
                (5000, "me", "hello from me"),
            ],
            "filler stripped, system offset applied, sorted by start"
        );
        assert_eq!(meta.speakers[0].name, "Tester");
        assert_eq!(meta.speakers[1].id, "others");

        let md = fs::read_to_string(store::transcript_path(tmp.path(), &meta.id)).unwrap();
        assert!(md.contains("## Transcript"));
        assert!(md.contains("**[00:00:03] Others:** hello from them"));
        // Persisted, not just returned.
        assert_eq!(store::read_meta(tmp.path()).unwrap(), meta);
    }

    /// `[replacements]` reach recording transcripts, after filler removal like
    /// dictation: the key below only matches once "uh" is gone, and a segment
    /// a replacement empties is dropped rather than kept as a blank line.
    #[tokio::test]
    async fn replacements_apply_to_segments_after_filler_removal() {
        let backend = FakeBackend::new(
            &[
                (3000, "um, edit the package uh build file"),
                (6000, "drop me"),
            ],
            None,
        );
        let (tmp, _) = recording(&[
            (TrackKind::Mic, 0, Some(track(4.0, &[(1.0, 1.0, 3000)]))),
            (TrackKind::System, 0, Some(track(4.0, &[(2.5, 1.0, 6000)]))),
        ]);
        let mut cfg = config();
        cfg.replacements = [
            ("package build".to_string(), "PKGBUILD".to_string()),
            ("drop me".to_string(), String::new()),
        ]
        .into();
        let meta = process_recording(
            tmp.path(),
            &cfg,
            Some(backend as Arc<dyn TranscriptionBackend>),
        )
        .await
        .unwrap();

        let lines: Vec<_> = meta
            .segments
            .iter()
            .map(|s| (s.speaker.as_str(), s.text.as_str()))
            .collect();
        assert_eq!(lines, [("me", "edit the PKGBUILD file")]);
        let md = fs::read_to_string(store::transcript_path(tmp.path(), &meta.id)).unwrap();
        assert!(md.contains("edit the PKGBUILD file"), "{md}");
    }

    #[tokio::test]
    async fn offsets_hold_across_chunks() {
        // A 3 s cap holds one padded 1 s burst (1.6 s) but not two.
        let backend = FakeBackend::new(
            &[(3000, "first"), (4000, "second"), (5000, "third")],
            Some(3),
        );
        let (tmp, _) = recording(&[(
            TrackKind::Mic,
            500,
            Some(track(
                27.0,
                &[(2.0, 1.0, 3000), (12.0, 1.0, 4000), (22.0, 1.0, 5000)],
            )),
        )]);
        let meta = run(tmp.path(), backend.clone()).await;

        assert_eq!(
            backend.calls.load(Ordering::Relaxed),
            3,
            "one chunk per burst"
        );
        let starts: Vec<_> = meta
            .segments
            .iter()
            .map(|s| (s.start_ms, s.text.as_str()))
            .collect();
        assert_eq!(
            starts,
            [(2500, "first"), (12_500, "second"), (22_500, "third")]
        );
    }

    /// The text backstop for echo the audio cancellation misses. The mic's
    /// copy is 1.2 s late, beyond any speaker-to-mic lag, so the canceller
    /// leaves it and only the text match can drop it.
    #[tokio::test]
    async fn mic_echo_of_system_audio_is_dropped() {
        let line = "we should ship the release on friday morning";
        let backend = FakeBackend::new(
            &[(2000, line), (6000, line), (3000, "sounds good to me")],
            None,
        );
        let (tmp, _) = recording(&[
            (
                TrackKind::Mic,
                0,
                Some(track(10.0, &[(2.2, 2.0, 2000), (6.0, 1.0, 3000)])),
            ),
            (TrackKind::System, 0, Some(track(10.0, &[(1.0, 2.0, 6000)]))),
        ]);
        let meta = run(tmp.path(), backend).await;

        let lines: Vec<_> = meta
            .segments
            .iter()
            .map(|s| (s.speaker.as_str(), s.text.as_str()))
            .collect();
        assert_eq!(lines, [("others", line), ("me", "sounds good to me")]);
    }

    #[tokio::test]
    async fn silent_and_missing_tracks_are_skipped() {
        let backend = FakeBackend::new(&[(3000, "only me")], None);
        let (tmp, _) = recording(&[
            (TrackKind::Mic, 0, Some(track(4.0, &[(1.0, 1.0, 3000)]))),
            (TrackKind::System, 0, Some(vec![0; RATE * 4])),
        ]);
        let meta = run(tmp.path(), backend.clone()).await;
        assert_eq!(meta.segments.len(), 1);
        assert_eq!(
            backend.calls.load(Ordering::Relaxed),
            1,
            "the silent system track never reaches the backend"
        );

        let (tmp, _) = recording(&[
            (TrackKind::Mic, 0, Some(track(4.0, &[(1.0, 1.0, 3000)]))),
            (TrackKind::System, 0, None),
        ]);
        let meta = run(tmp.path(), FakeBackend::new(&[(3000, "only me")], None)).await;
        assert_eq!(meta.segments.len(), 1);
        assert_eq!(meta.errors, ["system audio missing: audio/system.wav"]);
    }

    #[tokio::test]
    async fn a_few_words_in_a_quiet_track_are_sent() {
        // One second of speech in two quiet minutes: far below the gate
        // threshold on average, so a whole-track RMS check dropped it.
        let samples = track(120.0, &[(50.0, 1.0, 300)]);
        assert!(rms_energy(&samples) < SILENCE_RMS_THRESHOLD);
        let backend = FakeBackend::new(&[(300, "just one question")], Some(60));
        let (tmp, _) = recording(&[(TrackKind::Mic, 0, Some(samples))]);
        let meta = run(tmp.path(), backend.clone()).await;
        let texts: Vec<_> = meta.segments.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, ["just one question"]);
        assert_eq!(
            backend.calls.load(Ordering::Relaxed),
            1,
            "only the chunk holding the words is sent"
        );

        // A 200 ms click is still below the gate.
        let backend = FakeBackend::new(&[(300, "click")], None);
        let (tmp, _) = recording(&[(TrackKind::Mic, 0, Some(track(120.0, &[(50.0, 0.2, 300)])))]);
        let meta = run(tmp.path(), backend.clone()).await;
        assert!(meta.segments.is_empty());
        assert_eq!(backend.calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn no_audio_at_all_is_an_error() {
        let (tmp, _) = recording(&[(TrackKind::Mic, 0, None)]);
        let err = process_recording(
            tmp.path(),
            &config(),
            Some(FakeBackend::new(&[], None) as Arc<dyn TranscriptionBackend>),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("no recorded audio"));
        // Left as processing for the caller to mark failed.
        assert_eq!(
            store::read_meta(tmp.path()).unwrap().status,
            RecordingState::Processing
        );
    }

    #[tokio::test]
    async fn realtime_recording_backend_is_refused_before_transcribing() {
        let (tmp, _) = recording(&[(TrackKind::Mic, 0, Some(track(2.0, &[(0.5, 1.0, 3000)])))]);
        let mut cfg = config();
        cfg.general.backend = "openai-realtime".into();
        cfg.recording = None;
        let backend = FakeBackend::new(&[], None);
        let err = process_recording(
            tmp.path(),
            &cfg,
            Some(backend.clone() as Arc<dyn TranscriptionBackend>),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("batch backend"), "{err}");
        assert_eq!(backend.calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn finished_chunks_are_cached_and_reused_on_retry() {
        let texts = [(3000, "first"), (4000, "second"), (5000, "third")];
        let samples = track(
            27.0,
            &[(2.0, 1.0, 3000), (12.0, 1.0, 4000), (22.0, 1.0, 5000)],
        );
        let (tmp, _) = recording(&[(TrackKind::Mic, 0, Some(samples))]);

        // One chunk per burst at a 3 s cap. The third chunk's request fails:
        // the first two are cached.
        let failing = FakeBackend::with_failures(&texts, Some(3), Some(2), 0);
        let err = process_recording(
            tmp.path(),
            &config(),
            Some(failing.clone() as Arc<dyn TranscriptionBackend>),
        )
        .await
        .unwrap_err();
        assert!(format!("{err:#}").contains("network is down"), "{err:#}");
        let cache_dir = tmp.path().join(CACHE_DIR);
        assert_eq!(fs::read_dir(&cache_dir).unwrap().count(), 2);
        let entry = fs::read_dir(&cache_dir).unwrap().next().unwrap().unwrap();
        assert_eq!(
            entry.metadata().unwrap().permissions().mode() & 0o777,
            0o600
        );

        // The retry sends only the chunk that failed.
        let working = FakeBackend::new(&texts, Some(3));
        let meta = run(tmp.path(), working.clone()).await;
        assert_eq!(working.calls.load(Ordering::Relaxed), 1);
        let texts: Vec<_> = meta.segments.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, ["first", "second", "third"]);
        assert!(!cache_dir.exists(), "the cache is dropped once done");
    }

    /// Records the audio of every request and hears no speech in it.
    #[derive(Default)]
    struct RecordingBackend {
        sent: std::sync::Mutex<Vec<Vec<i16>>>,
        /// Requests with this index or later fail.
        fail_from: Option<usize>,
    }

    #[async_trait::async_trait]
    impl TranscriptionBackend for RecordingBackend {
        async fn transcribe(
            &self,
            _audio: &[u8],
            _config: &TranscriptionConfig,
        ) -> anyhow::Result<String> {
            unreachable!("recordings ask for segments")
        }
        fn sends_prompt(&self, _config: &TranscriptionConfig) -> bool {
            false
        }
        async fn transcribe_segments(
            &self,
            audio: &[u8],
            _config: &TranscriptionConfig,
        ) -> anyhow::Result<Transcript> {
            let mut sent = self.sent.lock().unwrap();
            if self.fail_from.is_some_and(|n| sent.len() >= n) {
                bail!("network is down");
            }
            sent.push(crate::transcription::decode_wav_samples(audio)?);
            Ok(Transcript {
                segments: Vec::new(),
            })
        }
        fn max_segment_chunk_secs(&self, _config: &TranscriptionConfig) -> Option<u32> {
            None
        }
    }

    async fn run_recording(
        dir: &Path,
        backend: &Arc<RecordingBackend>,
    ) -> anyhow::Result<RecordingMeta> {
        let backend = backend.clone() as Arc<dyn TranscriptionBackend>;
        process_recording(dir, &config(), Some(backend)).await
    }

    /// A chunk of `(original start, len)` pieces, packed as
    /// [`speech::pack`] packs them.
    fn map(pieces: &[(usize, usize)]) -> TimeMap {
        let regions: Vec<Range<usize>> = pieces
            .iter()
            .map(|&(start, len)| start..start + len)
            .collect();
        speech::pack(&regions, &[], None).remove(0)
    }

    /// A call over speakers: the mic and system tracks of
    /// [`fixtures::speaker_call`], as recorded.
    fn speaker_call() -> (Vec<i16>, Vec<i16>) {
        let (mic, system) = fixtures::speaker_call();
        (fixtures::quantize(&mic), fixtures::quantize(&system))
    }

    /// What one uncapped request carries for the track at `path`: its
    /// speech, found by `near_end` when given, else by frame energy, packed
    /// into one condensed chunk.
    fn condensed(path: &Path, near_end: Option<&[bool]>) -> Vec<i16> {
        let scan = scan_track(path).unwrap();
        let active = speech_activity(near_end, &scan.energies, "test");
        let regions = speech::regions(&active, scan.len);
        let chunks = speech::pack(&regions, &scan.energies, None);
        assert_eq!(chunks.len(), 1, "{} regions", regions.len());
        read_condensed(path, &chunks[0]).unwrap()
    }

    /// Cancel the recording's mic into `expected.wav` beside it, as
    /// `process_recording` should have, and return what that left to send.
    fn expected_mic(dir: &Path, timeline_lag: i64) -> Vec<i16> {
        let expected = dir.join("expected.wav");
        let outcome = cancel_echo(
            &store::audio_path(dir, TrackKind::Mic),
            &store::audio_path(dir, TrackKind::System),
            timeline_lag,
            &expected,
        )
        .unwrap();
        let EchoOutcome::Cancelled(done) = outcome else {
            panic!("expected a cancellation, got {outcome:?}");
        };
        condensed(&expected, Some(&done.near_end))
    }

    /// With both tracks the mic request carries the user's speech from the
    /// echo-cancelled track (what [`cancel_echo`] writes, condensed by its
    /// near-end mask), and `audio/mic.wav` is left as recorded.
    #[tokio::test]
    async fn the_echo_cancelled_mic_is_sent_when_both_tracks_exist() {
        let (mic, system) = speaker_call();
        let (tmp, _) = recording(&[
            (TrackKind::Mic, 0, Some(mic.clone())),
            (TrackKind::System, 0, Some(system.clone())),
        ]);
        let mic_path = store::audio_path(tmp.path(), TrackKind::Mic);
        let system_path = store::audio_path(tmp.path(), TrackKind::System);
        let recorded = fs::read(&mic_path).unwrap();
        let backend = Arc::new(RecordingBackend::default());
        run_recording(tmp.path(), &backend).await.unwrap();

        let sent = backend.sent.lock().unwrap();
        assert_eq!(sent.len(), 2);
        // `assert!`, not `assert_eq!`: a failure would print every sample.
        assert!(
            sent[0] == expected_mic(tmp.path(), 0),
            "mic is not the cancelled speech"
        );
        assert!(
            sent[0] != condensed(&mic_path, None),
            "the mic went out uncancelled"
        );
        // Only the user's 1.4 s (padded), not the 8.5 s of echo around it.
        assert!(sent[0].len() < 3 * RATE, "{} samples", sent[0].len());
        assert!(
            sent[1] == condensed(&system_path, None),
            "the system track is sent as recorded"
        );
        assert_eq!(fs::read(&mic_path).unwrap(), recorded);
        assert!(
            !tmp.path().join(CACHE_DIR).exists(),
            "the cancelled track goes with the cache"
        );
    }

    /// No system track (`system_device = "none"`), or a silent one: the mic
    /// is sent as recorded, condensed by frame energy.
    #[tokio::test]
    async fn the_raw_mic_is_sent_without_system_audio() {
        let (mic, _) = speaker_call();
        let (tmp, _) = recording(&[(TrackKind::Mic, 0, Some(mic.clone()))]);
        let mic_path = store::audio_path(tmp.path(), TrackKind::Mic);
        let backend = Arc::new(RecordingBackend::default());
        run_recording(tmp.path(), &backend).await.unwrap();
        let raw = condensed(&mic_path, None);
        assert!(
            *backend.sent.lock().unwrap() == [raw.clone()],
            "no system track"
        );

        let (tmp, _) = recording(&[
            (TrackKind::Mic, 0, Some(mic.clone())),
            (TrackKind::System, 0, Some(vec![0; mic.len()])),
        ]);
        let backend = Arc::new(RecordingBackend::default());
        run_recording(tmp.path(), &backend).await.unwrap();
        assert!(
            *backend.sent.lock().unwrap() == [raw],
            "silent system track"
        );
    }

    /// Two short utterances a minute apart in two quiet minutes: one request
    /// of just them with a second of silence between, and the segments land
    /// back where they were said.
    #[tokio::test]
    async fn only_the_speech_is_sent() {
        let backend = FakeBackend::new(&[(3000, "first"), (4000, "second")], Some(600));
        let samples = track(120.0, &[(30.0, 1.0, 3000), (90.0, 1.0, 4000)]);
        let (tmp, _) = recording(&[(TrackKind::Mic, 250, Some(samples))]);
        let meta = run(tmp.path(), backend.clone()).await;

        // Each second of speech padded by 0.3 s a side, 1 s apart.
        let region = (1.6 * RATE as f64) as usize;
        assert_eq!(*backend.sent.lock().unwrap(), [2 * region + RATE]);
        let lines: Vec<_> = meta
            .segments
            .iter()
            .map(|s| (s.start_ms, s.end_ms, s.text.as_str()))
            .collect();
        assert_eq!(
            lines,
            [(30_250, 31_250, "first"), (90_250, 91_250, "second")]
        );
    }

    #[tokio::test]
    async fn a_silent_track_sends_nothing() {
        let backend = FakeBackend::new(&[], None);
        let mut quiet = vec![0i16; 30 * RATE];
        // Room noise well under the gate.
        for (i, s) in quiet.iter_mut().enumerate() {
            *s = [20, -20, 15, -15][i % 4];
        }
        let (tmp, _) = recording(&[(TrackKind::Mic, 0, Some(quiet))]);
        let meta = run(tmp.path(), backend.clone()).await;
        assert_eq!(meta.status, RecordingState::Done);
        assert!(meta.segments.is_empty());
        assert_eq!(backend.calls.load(Ordering::Relaxed), 0);
    }

    /// A hissy mic (white noise at -54 dBFS) clears the near-end detector's
    /// in-band floor whenever the far side is quiet; the energy gate keeps
    /// those stretches from being sent as speech.
    #[tokio::test]
    async fn mic_hiss_is_not_sent_as_speech() {
        let len = fixtures::at(30.0);
        let mut system = fixtures::speech(len, &[(1.0, 8.0), (20.0, 27.0)], 0.1, 31);
        let near = fixtures::speech(len, &[(10.0, 12.0), (15.0, 16.0)], 0.01, 32);
        let mut mic = fixtures::mix(&fixtures::echo(&system, 1000, 33), &near);
        fixtures::add_noise(&mut mic, 0.002, 34);
        fixtures::add_noise(&mut system, 1e-4, 35);
        let (tmp, _) = recording(&[
            (TrackKind::Mic, 0, Some(fixtures::quantize(&mic))),
            (TrackKind::System, 0, Some(fixtures::quantize(&system))),
        ]);
        let backend = Arc::new(RecordingBackend::default());
        run_recording(tmp.path(), &backend).await.unwrap();
        // The user's 3 s, padded (2.6 + 1.6 s) and 1 s apart: not the 12 s
        // of hiss between the far side's turns.
        let sent = backend.sent.lock().unwrap()[0].len();
        assert!(
            (4 * RATE..6 * RATE).contains(&sent),
            "{:.1} s of mic sent",
            sent as f64 / RATE as f64
        );
    }

    /// A backend that runs a whole chunk into one sentence, with a word per
    /// [`FakeBackend`] segment: like Whisper running two regions together
    /// across a join.
    struct OneSentence(Arc<FakeBackend>);

    #[async_trait::async_trait]
    impl TranscriptionBackend for OneSentence {
        async fn transcribe(
            &self,
            _audio: &[u8],
            _config: &TranscriptionConfig,
        ) -> anyhow::Result<String> {
            unreachable!("recordings ask for segments")
        }
        fn sends_prompt(&self, _config: &TranscriptionConfig) -> bool {
            false
        }
        async fn transcribe_segments(
            &self,
            audio: &[u8],
            config: &TranscriptionConfig,
        ) -> anyhow::Result<Transcript> {
            let parts = self.0.transcribe_segments(audio, config).await?.segments;
            let (Some(first), Some(last)) = (parts.first(), parts.last()) else {
                return Ok(Transcript { segments: parts });
            };
            let texts: Vec<&str> = parts.iter().map(|s| s.text.as_str()).collect();
            let sentence = Segment {
                start_ms: first.start_ms,
                end_ms: last.end_ms,
                text: texts.join(" "),
                speaker: None,
                words: Some(
                    parts
                        .iter()
                        .map(|s| crate::recording::Word {
                            start_ms: s.start_ms,
                            end_ms: s.end_ms,
                            text: s.text.clone(),
                            speaker: None,
                        })
                        .collect(),
                ),
            };
            Ok(Transcript {
                segments: vec![sentence],
            })
        }
        fn max_segment_chunk_secs(&self, config: &TranscriptionConfig) -> Option<u32> {
            self.0.max_segment_chunk_secs(config)
        }
    }

    /// One sentence across a join comes back as two segments at the times
    /// each part was said, words and track offset included.
    #[tokio::test]
    async fn a_sentence_across_a_join_is_split_by_its_words() {
        let fake = FakeBackend::new(&[(3000, "First."), (4000, "Second.")], Some(600));
        let samples = track(120.0, &[(30.0, 1.0, 3000), (90.0, 1.0, 4000)]);
        let (tmp, _) = recording(&[(TrackKind::Mic, 250, Some(samples))]);
        let meta = process_recording(
            tmp.path(),
            &config(),
            Some(Arc::new(OneSentence(fake.clone())) as Arc<dyn TranscriptionBackend>),
        )
        .await
        .unwrap();
        assert_eq!(fake.calls.load(Ordering::Relaxed), 1);
        let lines: Vec<_> = meta
            .segments
            .iter()
            .map(|s| (s.start_ms, s.end_ms, s.text.as_str()))
            .collect();
        assert_eq!(
            lines,
            [(30_250, 31_250, "First."), (90_250, 91_250, "Second.")]
        );
    }

    /// The system track has no near-end detector: its speech is found by
    /// frame energy and condensed the same way.
    #[tokio::test]
    async fn the_system_track_is_condensed_by_energy() {
        let system = track(120.0, &[(10.0, 2.0, 6000), (100.0, 1.0, 6000)]);
        let (tmp, _) = recording(&[
            (TrackKind::Mic, 0, Some(vec![0; 120 * RATE])),
            (TrackKind::System, 0, Some(system)),
        ]);
        let backend = Arc::new(RecordingBackend::default());
        run_recording(tmp.path(), &backend).await.unwrap();
        let sent = backend.sent.lock().unwrap();
        assert_eq!(sent.len(), 1, "the silent mic sends nothing");
        let (first, second) = ((2.6 * RATE as f64) as usize, (1.6 * RATE as f64) as usize);
        assert_eq!(sent[0].len(), first + RATE + second);
        let system_path = store::audio_path(tmp.path(), TrackKind::System);
        assert!(sent[0] == condensed(&system_path, None));
    }

    /// A failed attempt keeps the cancelled mic in the cache, private, and
    /// the retry sends that file instead of cancelling again.
    #[tokio::test]
    async fn a_retry_reuses_the_cancelled_mic() {
        let (mic, system) = speaker_call();
        let (tmp, _) = recording(&[
            (TrackKind::Mic, 0, Some(mic.clone())),
            (TrackKind::System, 0, Some(system)),
        ]);
        let failing = Arc::new(RecordingBackend {
            fail_from: Some(0),
            ..Default::default()
        });
        run_recording(tmp.path(), &failing).await.unwrap_err();
        let cached = tmp.path().join(CACHE_DIR).join(ECHO_CANCELLED_MIC);
        let mask = tmp.path().join(CACHE_DIR).join(ECHO_CANCELLED_MASK);
        for file in [&cached, &mask] {
            let mode = fs::metadata(file).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "{}", file.display());
        }
        assert_eq!(fs::read_dir(tmp.path().join(CACHE_DIR)).unwrap().count(), 2);
        let near_end = speech::decode_mask(&fs::read(&mask).unwrap(), mic.len()).unwrap();

        // A marker of the same length: sent only if the file is reused.
        write_wav(&cached, &vec![1234i16; mic.len()]);
        let working = Arc::new(RecordingBackend::default());
        run_recording(tmp.path(), &working).await.unwrap();
        let sent = &working.sent.lock().unwrap()[0];
        assert!(sent.contains(&1234), "cached track not reused");
        assert!(sent.iter().all(|&s| s == 1234 || s == 0));
        assert_eq!(sent.len(), condensed_len(&near_end, mic.len()));
    }

    /// Condensed length of a `len`-sample track sent by `active`.
    fn condensed_len(active: &[bool], len: usize) -> usize {
        speech::pack(&speech::regions(active, len), &[], None)
            .iter()
            .map(TimeMap::condensed_len)
            .sum()
    }

    /// The cancelled track is reused only with its mask: without the mask
    /// both are made again.
    #[tokio::test]
    async fn a_cached_track_without_its_mask_is_redone() {
        let (mic, system) = speaker_call();
        let (tmp, _) = recording(&[
            (TrackKind::Mic, 0, Some(mic.clone())),
            (TrackKind::System, 0, Some(system)),
        ]);
        let failing = Arc::new(RecordingBackend {
            fail_from: Some(0),
            ..Default::default()
        });
        run_recording(tmp.path(), &failing).await.unwrap_err();
        let cache = tmp.path().join(CACHE_DIR);
        write_wav(&cache.join(ECHO_CANCELLED_MIC), &vec![1234i16; mic.len()]);
        fs::remove_file(cache.join(ECHO_CANCELLED_MASK)).unwrap();

        let working = Arc::new(RecordingBackend::default());
        run_recording(tmp.path(), &working).await.unwrap();
        let sent = &working.sent.lock().unwrap()[0];
        assert!(
            *sent == expected_mic(tmp.path(), 0),
            "stale cached track sent"
        );
    }

    /// The system track started 400 ms after the mic, so system sample `v`
    /// was captured with mic sample `v + 6400`, and the mic hears it 1000
    /// samples after that. With the offsets read the wrong way round, the
    /// echo would fall outside the lag search and the mic would go out
    /// uncancelled.
    #[tokio::test]
    async fn track_offsets_place_the_echo_search() {
        let len = fixtures::at(12.0);
        let mut system = fixtures::speech(len, &[(0.5, 5.0), (7.0, 11.0)], 0.1, 1);
        let near = fixtures::speech(len, &[(5.6, 6.8)], 0.01, 2);
        let mut mic = fixtures::mix(&fixtures::echo(&system, 6400 + 1000, 3), &near);
        fixtures::add_noise(&mut mic, 3e-4, 4);
        fixtures::add_noise(&mut system, 1e-4, 5);
        let (mic, system) = (fixtures::quantize(&mic), fixtures::quantize(&system));
        let (tmp, _) = recording(&[
            (TrackKind::Mic, 0, Some(mic.clone())),
            (TrackKind::System, 400, Some(system)),
        ]);
        let backend = Arc::new(RecordingBackend::default());
        run_recording(tmp.path(), &backend).await.unwrap();

        let outcome = cancel_echo(
            &store::audio_path(tmp.path(), TrackKind::Mic),
            &store::audio_path(tmp.path(), TrackKind::System),
            6400,
            &tmp.path().join("check.wav"),
        )
        .unwrap();
        let EchoOutcome::Cancelled(done) = outcome else {
            panic!("expected a cancellation, got {outcome:?}");
        };
        assert!((done.report.lag - 7400).abs() <= 2, "{:?}", done.report);
        let mic_path = store::audio_path(tmp.path(), TrackKind::Mic);
        let sent = backend.sent.lock().unwrap();
        assert!(
            sent[0] != condensed(&mic_path, None),
            "the mic went out uncancelled"
        );
        assert!(
            sent[0] == expected_mic(tmp.path(), 6400),
            "mic is not the cancelled speech"
        );
    }

    /// The echo step never fails a recording: with `.cache` a plain file it
    /// cannot write the cancelled track (nor cache chunks), and the recording
    /// still finishes from the mic as recorded.
    #[tokio::test]
    async fn a_failed_echo_step_falls_back_to_the_raw_mic() {
        let (mic, system) = speaker_call();
        let (tmp, _) = recording(&[
            (TrackKind::Mic, 0, Some(mic.clone())),
            (TrackKind::System, 0, Some(system)),
        ]);
        fs::write(tmp.path().join(CACHE_DIR), "not a directory").unwrap();
        let backend = Arc::new(RecordingBackend::default());
        let meta = run_recording(tmp.path(), &backend).await.unwrap();
        assert_eq!(meta.status, RecordingState::Done);
        let raw = condensed(&store::audio_path(tmp.path(), TrackKind::Mic), None);
        assert!(backend.sent.lock().unwrap()[0] == raw, "raw mic not sent");
    }

    /// A rename that fails after a full cancellation (here the target is a
    /// directory) leaves no mic-sized temporary in the cache.
    #[tokio::test]
    async fn a_failed_rename_leaves_no_temporary() {
        let (mic, system) = speaker_call();
        let (tmp, _) = recording(&[
            (TrackKind::Mic, 0, Some(mic)),
            (TrackKind::System, 0, Some(system)),
        ]);
        let cached = tmp.path().join(CACHE_DIR).join(ECHO_CANCELLED_MIC);
        fs::create_dir_all(cached.join("occupied")).unwrap();
        // Fail the mic request so the cache is kept for inspection.
        let failing = Arc::new(RecordingBackend {
            fail_from: Some(0),
            ..Default::default()
        });
        run_recording(tmp.path(), &failing).await.unwrap_err();
        let leftover = tmp
            .path()
            .join(CACHE_DIR)
            .join("mic-echo-cancelled.wav.tmp");
        assert!(!leftover.exists(), "temporary left behind");
        // The mask is written only beside the track it was made from.
        let mask = tmp.path().join(CACHE_DIR).join(ECHO_CANCELLED_MASK);
        assert!(!mask.exists(), "a mask without its track");
    }

    /// A system track listed in `recording.json` but missing on disk is not an
    /// echo failure: the step is skipped before it touches the cache.
    #[tokio::test]
    async fn a_missing_system_file_skips_the_echo_step() {
        let (mic, _) = speaker_call();
        let (tmp, _) = recording(&[(TrackKind::Mic, 0, Some(mic)), (TrackKind::System, 0, None)]);
        // Fail the mic request, so nothing else creates the cache dir.
        let failing = Arc::new(RecordingBackend {
            fail_from: Some(0),
            ..Default::default()
        });
        run_recording(tmp.path(), &failing).await.unwrap_err();
        assert!(!tmp.path().join(CACHE_DIR).exists());
    }

    /// A cached cancelled track whose length is not the mic's (another
    /// recording, or a stray file) is cancelled again rather than sent.
    #[tokio::test]
    async fn a_cached_track_of_the_wrong_length_is_redone() {
        let (mic, system) = speaker_call();
        let (tmp, _) = recording(&[
            (TrackKind::Mic, 0, Some(mic.clone())),
            (TrackKind::System, 0, Some(system)),
        ]);
        let cached = tmp.path().join(CACHE_DIR).join(ECHO_CANCELLED_MIC);
        write_wav(&cached, &vec![1234i16; mic.len() - 1]);
        let backend = Arc::new(RecordingBackend::default());
        run_recording(tmp.path(), &backend).await.unwrap();
        let sent = &backend.sent.lock().unwrap()[0];
        assert!(
            *sent == expected_mic(tmp.path(), 0),
            "stale cached track sent"
        );
    }

    #[test]
    fn cancelled_mic_chunks_are_cached_apart_from_raw_ones() {
        let tmp = tempfile::tempdir().unwrap();
        let raw = ChunkCache {
            dir: tmp.path().join(CACHE_DIR),
            fingerprint: "fp".into(),
        };
        let cancelled = raw.echo_cancelled();
        let chunk = map(&[(0, 16_000)]);
        raw.store(TrackKind::Mic, &chunk, &[seg(0, 900, "raw")])
            .unwrap();
        assert_eq!(cancelled.load(TrackKind::Mic, &chunk), None);
        cancelled
            .store(TrackKind::Mic, &chunk, &[seg(0, 900, "cancelled")])
            .unwrap();
        assert_eq!(raw.load(TrackKind::Mic, &chunk), None);
        assert_eq!(
            cancelled.load(TrackKind::Mic, &chunk),
            Some(vec![seg(0, 900, "cancelled")])
        );
    }

    #[test]
    fn cache_ignores_another_backend_or_model() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = |backend: &str, model: &str| ChunkCache {
            dir: tmp.path().join(CACHE_DIR),
            fingerprint: format!("{backend}/{model}"),
        };
        let segments = vec![seg(0, 1000, "hi")];
        let chunk = map(&[(0, 16_000), (48_000, 3_200)]);
        cache("groq", "m1")
            .store(TrackKind::System, &chunk, &segments)
            .unwrap();
        assert_eq!(
            cache("groq", "m1").load(TrackKind::System, &chunk),
            Some(segments)
        );
        assert_eq!(cache("openai", "m1").load(TrackKind::System, &chunk), None);
        assert_eq!(cache("groq", "m2").load(TrackKind::System, &chunk), None);
        assert_eq!(cache("groq", "m1").load(TrackKind::Mic, &chunk), None);
        // Other speech, or the same speech packed otherwise, is another chunk.
        for other in [
            map(&[(0, 16_000)]),
            map(&[(0, 16_000), (48_000, 1_600)]),
            map(&[(1_600, 16_000), (48_000, 3_200)]),
        ] {
            assert_eq!(cache("groq", "m1").load(TrackKind::System, &other), None);
        }
        let mut moved = chunk.clone();
        moved.pieces[1].condensed += 1;
        assert_eq!(cache("groq", "m1").load(TrackKind::System, &moved), None);
    }

    /// Real-data check of what gets sent, run by hand on copies of a
    /// real recording (never in the recording folder itself):
    ///
    /// ```text
    /// WHISRS_ECHO_MIC=mic.wav WHISRS_ECHO_SYSTEM=system.wav \
    /// WHISRS_ECHO_OUT=cancelled.wav WHISRS_ECHO_LAG=<(system - mic offset_ms) * 16> \
    /// WHISRS_SPEECH_LABELS=deepgram.json \
    /// cargo test --release --lib speech_on_a_real_recording -- --ignored --nocapture
    /// ```
    ///
    /// The labels are a Deepgram response for the cancelled mic: the words
    /// in `results.utterances[].words[]` are the user's. Prints the share of
    /// STFT frames (256-sample hop, as the prototype counted them) inside
    /// those words that are sent, and of all other frames, both for the
    /// padded near-end mask alone and for the final regions, and how long
    /// each track is once condensed.
    #[test]
    #[ignore = "needs a real recording, see the doc comment"]
    fn speech_on_a_real_recording() {
        let var = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("set {name}"));
        let (mic, system, out) = (
            PathBuf::from(var("WHISRS_ECHO_MIC")),
            PathBuf::from(var("WHISRS_ECHO_SYSTEM")),
            PathBuf::from(var("WHISRS_ECHO_OUT")),
        );
        let lag: i64 = var("WHISRS_ECHO_LAG").parse().unwrap();
        let labels: serde_json::Value =
            serde_json::from_slice(&fs::read(var("WHISRS_SPEECH_LABELS")).unwrap()).unwrap();
        let started = std::time::Instant::now();
        let EchoOutcome::Cancelled(done) = cancel_echo(&mic, &system, lag, &out).unwrap() else {
            panic!("no echo cancelled");
        };
        println!("{done:?} in {:.2?}", started.elapsed());

        let scan = scan_track(&out).unwrap();
        let active = speech_activity(Some(&done.near_end), &scan.energies, "mic");
        let regions = speech::regions(&active, scan.len);
        let mut words: Vec<(f64, f64)> = labels["results"]["utterances"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|u| u["words"].as_array().unwrap().iter())
            .map(|w| (w["start"].as_f64().unwrap(), w["end"].as_f64().unwrap()))
            .collect();
        words.sort_by(|a, b| a.0.total_cmp(&b.0));
        let hop = 256;
        let frames = (scan.len - 512) / hop + 1;
        let mut next_word = 0;
        let mut labelled = vec![false; frames];
        for (i, label) in labelled.iter_mut().enumerate() {
            let t = (i * hop) as f64 / f64::from(SAMPLE_RATE);
            while next_word < words.len() && words[next_word].1 < t {
                next_word += 1;
            }
            *label = words[next_word..]
                .iter()
                .take_while(|w| w.0 <= t)
                .any(|w| t <= w.1);
        }
        let padded = |mask: &[bool], at: usize| {
            let f = at / speech::FRAME;
            mask[f.saturating_sub(3)..(f + 4).min(mask.len())]
                .iter()
                .any(|&a| a)
        };
        let detector_alone = |at: usize| padded(&done.near_end, at);
        let with_energy = |at: usize| padded(&active, at);
        let in_regions = |at: usize| regions.iter().any(|r| r.contains(&at));
        let share = |kept: &dyn Fn(usize) -> bool, want: bool| {
            let picked: Vec<usize> = (0..frames).filter(|&i| labelled[i] == want).collect();
            let n = picked.iter().filter(|&&i| kept(i * hop)).count();
            100.0 * n as f64 / picked.len() as f64
        };
        println!(
            "label frames {} of {frames}",
            labelled.iter().filter(|&&l| l).count()
        );
        for (name, kept) in [
            (
                "near-end mask alone, padded",
                &detector_alone as &dyn Fn(usize) -> bool,
            ),
            ("near-end and energy, padded", &with_energy),
            ("final regions", &in_regions),
        ] {
            println!(
                "{name}: user's word frames kept {:.1}%, other frames kept {:.1}%",
                share(kept, true),
                share(kept, false)
            );
        }

        // The user over the far side: word frames where the system track,
        // shifted by the lag, is above the gate.
        let system_scan = scan_track(&system).unwrap();
        let far_active = |at: usize| {
            let at = at as i64 - done.report.lag;
            at >= 0
                && system_scan
                    .energies
                    .get(at as usize / speech::FRAME)
                    .is_some_and(|&e| e >= SILENCE_RMS_THRESHOLD)
        };
        let over: Vec<usize> = (0..frames)
            .filter(|&i| labelled[i] && far_active(i * hop))
            .collect();
        let kept = over.iter().filter(|&&i| in_regions(i * hop)).count();
        println!(
            "user's word frames over far-end audio: {}, kept {:.1}%",
            over.len(),
            100.0 * kept as f64 / over.len().max(1) as f64
        );

        let secs = |samples: usize| samples as f64 / f64::from(SAMPLE_RATE);
        let report = |name: &str, len: usize, regions: &[Range<usize>], energies: &[f64]| {
            let speech_len: usize = regions.iter().map(|r| r.len()).sum();
            let one: usize = speech::pack(regions, energies, None)
                .iter()
                .map(TimeMap::condensed_len)
                .sum();
            let groq = speech::pack(regions, energies, Some(600));
            println!(
                "{name}: {:.1} s track, {} region(s), {:.1} s of speech, condensed {:.1} s \
                 ({} chunk(s) at a 600 s cap)",
                secs(len),
                regions.len(),
                secs(speech_len),
                secs(one),
                groq.len()
            );
        };
        report("mic", scan.len, &regions, &scan.energies);
        let system_regions =
            speech::regions(&speech::energy_mask(&system_scan.energies), system_scan.len);
        report(
            "system",
            system_scan.len,
            &system_regions,
            &system_scan.energies,
        );
    }

    fn fingerprint_of(toml_text: &str) -> String {
        let config: Config = toml::from_str(toml_text).unwrap();
        let name = recording_backend_name(&config).unwrap();
        let tcfg = recording_transcription_config(&name, &config, &config.general.language);
        cache_fingerprint(&name, &config, &tcfg)
    }

    #[test]
    fn fingerprint_changes_with_what_decides_the_output() {
        let base = "[general]\nbackend = \"local-whisper\"\nlanguage = \"en\"\n\
                    [local-whisper]\nmodel_path = \"/m/base.en.bin\"\n";
        let fp = fingerprint_of(base);
        // Stable across calls (and processes: no random hasher seed).
        assert_eq!(fp, fingerprint_of(base));
        // `local` is the same backend as `local-whisper`.
        assert_eq!(
            fp,
            fingerprint_of(&base.replace("\"local-whisper\"", "\"local\""))
        );
        // A language change invalidates the cache.
        assert_ne!(fp, fingerprint_of(&base.replace("\"en\"", "\"de\"")));
        // A model file change invalidates it, though the model name is the same.
        assert_ne!(
            fp,
            fingerprint_of(&base.replace("base.en.bin", "small.en.bin"))
        );
        // So does the vocabulary. The dictation prompt is not sent with a
        // recording, so it does not.
        let vocab = base.replace("language", "vocabulary = [\"whisrs\"]\nlanguage");
        assert_ne!(fp, fingerprint_of(&vocab));
        let prompted = base.replace("language", "prompt = \"Standup\"\nlanguage");
        assert_eq!(fp, fingerprint_of(&prompted));
        // Phrase cuts and flash attention change what whisper decodes.
        let silence = format!("{base}phrase_silence_ms = 900\n");
        assert_ne!(fp, fingerprint_of(&silence));
        // Recordings cap the phrase cut at 900 ms, so longer ones decode alike.
        for longer in [1200, 5000] {
            let cut = format!("{base}phrase_silence_ms = {longer}\n");
            assert_eq!(fingerprint_of(&silence), fingerprint_of(&cut), "{longer}");
        }
        let flash = format!("{base}flash_attn = true\n");
        assert_ne!(fp, fingerprint_of(&flash));
        // Segmentation does not: recordings always decode phrase by phrase.
        for mode in ["none", "window", "silence"] {
            let segmented = format!("{base}segmentation = \"{mode}\"\n");
            assert_eq!(fp, fingerprint_of(&segmented), "segmentation = {mode}");
        }

        // The sidecar URL is part of its identity.
        let sidecar = "[general]\nbackend = \"asr-sidecar\"\n\
                       [asr-sidecar]\nurl = \"http://127.0.0.1:1/t\"\n";
        assert_ne!(
            fingerprint_of(sidecar),
            fingerprint_of(&sidecar.replace(":1/t", ":2/t"))
        );
        assert_eq!(
            fingerprint_of(sidecar),
            fingerprint_of(&sidecar.replace("\"asr-sidecar\"\n[", "\"vibevoice\"\n["))
        );
    }

    /// A recording whose first attempt failed after one chunk, which is
    /// cached under English, and the backend texts for its two chunks.
    async fn english_chunk_cached() -> (tempfile::TempDir, [(i16, &'static str); 2]) {
        let texts = [(3000, "first"), (4000, "second")];
        let samples = track(17.0, &[(2.0, 1.0, 3000), (12.0, 1.0, 4000)]);
        let (tmp, _) = recording(&[(TrackKind::Mic, 0, Some(samples))]);
        let failing = FakeBackend::with_failures(&texts, Some(3), Some(1), 0);
        process_recording(
            tmp.path(),
            &config(),
            Some(failing as Arc<dyn TranscriptionBackend>),
        )
        .await
        .unwrap_err();
        assert_eq!(fs::read_dir(tmp.path().join(CACHE_DIR)).unwrap().count(), 1);
        let meta = store::read_meta(tmp.path()).unwrap();
        assert_eq!(meta.mic_language.as_deref(), Some("en"), "stored");
        (tmp, texts)
    }

    /// A retry that changes the language (`retry -l` rewrites the stored
    /// one) sends every chunk again.
    #[tokio::test]
    async fn a_language_change_retranscribes_cached_chunks() {
        let (tmp, texts) = english_chunk_cached().await;
        let mut meta = store::read_meta(tmp.path()).unwrap();
        meta.mic_language = Some("de".into());
        store::write_meta(tmp.path(), &meta).unwrap();
        let working = FakeBackend::new(&texts, Some(3));
        run(tmp.path(), working.clone()).await;
        // Both chunks sent again: nothing English is reused.
        assert_eq!(*working.languages.lock().unwrap(), ["de", "de"]);
    }

    /// A recording keeps the language it started with: a config change
    /// after that does not reach it.
    #[tokio::test]
    async fn a_config_change_does_not_reach_a_started_recording() {
        let (tmp, texts) = english_chunk_cached().await;
        let mut german = config();
        german.general.language = "de".into();
        german.recording.as_mut().unwrap().language = Some("de".into());
        let working = FakeBackend::new(&texts, Some(3));
        let meta = process_recording(
            tmp.path(),
            &german,
            Some(working.clone() as Arc<dyn TranscriptionBackend>),
        )
        .await
        .unwrap();
        assert_eq!(
            *working.languages.lock().unwrap(),
            ["en"],
            "the cached chunk reused, the other sent in English"
        );
        assert_eq!(meta.mic_language.as_deref(), Some("en"));
    }

    /// Each track is sent in its own language, and changing one track's
    /// language sends only that track again: the other's cached chunks are
    /// reused. The transcript names both languages, the model and the
    /// others label.
    #[tokio::test]
    async fn each_track_is_sent_in_its_own_language() {
        let texts = [(3000, "hallo"), (6000, "first"), (7000, "second")];
        let (tmp, _) = recording(&[
            (TrackKind::Mic, 0, Some(track(17.0, &[(6.0, 1.0, 3000)]))),
            (
                TrackKind::System,
                0,
                Some(track(17.0, &[(1.0, 1.0, 6000), (12.0, 1.0, 7000)])),
            ),
        ]);
        let mut meta = store::read_meta(tmp.path()).unwrap();
        meta.mic_language = Some("de".into());
        meta.system_language = Some("en".into());
        meta.others_label = Some("Video".into());
        store::write_meta(tmp.path(), &meta).unwrap();

        // One chunk for the mic, two for the system track; the last fails.
        let failing = FakeBackend::with_failures(&texts, Some(3), Some(2), 0);
        process_recording(
            tmp.path(),
            &config(),
            Some(failing.clone() as Arc<dyn TranscriptionBackend>),
        )
        .await
        .unwrap_err();
        assert_eq!(*failing.languages.lock().unwrap(), ["de", "en"]);

        // A retry with the system track in another language.
        let mut meta = store::read_meta(tmp.path()).unwrap();
        meta.system_language = Some("auto".into());
        store::write_meta(tmp.path(), &meta).unwrap();
        let working = FakeBackend::new(&texts, Some(3));
        let meta = run(tmp.path(), working.clone()).await;
        assert_eq!(
            *working.languages.lock().unwrap(),
            ["auto", "auto"],
            "both system chunks sent again, the cached German mic chunk reused"
        );

        assert_eq!(meta.mic_language.as_deref(), Some("de"));
        assert_eq!(meta.system_language.as_deref(), Some("auto"));
        assert_eq!(meta.model.as_deref(), Some("whisper-large-v3-turbo"));
        let lines: Vec<_> = meta
            .segments
            .iter()
            .map(|s| (s.speaker.as_str(), s.text.as_str()))
            .collect();
        assert_eq!(
            lines,
            [("others", "first"), ("me", "hallo"), ("others", "second")]
        );
        let md = fs::read_to_string(store::transcript_path(tmp.path(), &meta.id)).unwrap();
        for line in [
            "backend: \"groq\"\n",
            "model: \"whisper-large-v3-turbo\"\n",
            "mic_language: \"de\"\n",
            "system_language: \"auto\"\n",
            "  - \"Video\"\n",
            "**[00:00:01] Video:** first",
            "**[00:00:06] Tester:** hallo",
        ] {
            assert!(md.contains(line), "{line:?} missing from:\n{md}");
        }
        assert!(!md.contains("Others"), "{md}");
    }

    /// A `recording.json` from before languages, the others label and the
    /// model were stored loads and processes: both tracks take the config's
    /// language, which is then stored, and the system track is "Others".
    #[tokio::test]
    async fn an_old_recording_json_still_processes() {
        let (tmp, _) = recording(&[
            (TrackKind::Mic, 0, Some(track(8.0, &[(1.0, 1.0, 3000)]))),
            (TrackKind::System, 0, Some(track(8.0, &[(5.0, 1.0, 6000)]))),
        ]);
        let path = tmp.path().join("recording.json");
        let mut json: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        let fields = json.as_object_mut().unwrap();
        for key in ["mic_language", "system_language", "others_label", "model"] {
            assert!(fields.remove(key).is_some(), "{key} is written today");
        }
        fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();

        let mut cfg = config();
        cfg.recording.as_mut().unwrap().language = Some("de".into());
        let backend = FakeBackend::new(&[(3000, "ich"), (6000, "them")], None);
        let meta = process_recording(
            tmp.path(),
            &cfg,
            Some(backend.clone() as Arc<dyn TranscriptionBackend>),
        )
        .await
        .unwrap();
        assert_eq!(*backend.languages.lock().unwrap(), ["de", "de"]);
        assert_eq!(meta.mic_language.as_deref(), Some("de"));
        assert_eq!(meta.system_language.as_deref(), Some("de"));
        assert_eq!(meta.speakers[1].name, "Others");
        let md = fs::read_to_string(store::transcript_path(tmp.path(), &meta.id)).unwrap();
        assert!(md.contains("\nlanguage: \"de\"\n"), "{md}");
        assert!(md.contains("**[00:00:05] Others:** them"), "{md}");
    }

    /// A phrase backend (no timestamps) that is always rate limited.
    struct LimitedPhrases {
        calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl TranscriptionBackend for LimitedPhrases {
        async fn transcribe(
            &self,
            _audio: &[u8],
            _config: &TranscriptionConfig,
        ) -> anyhow::Result<String> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Err(RetryableHttpError {
                status: 429,
                retry_after: Some(Duration::from_secs(7)),
                message: "slow down".into(),
            }
            .into())
        }
        fn sends_prompt(&self, _config: &TranscriptionConfig) -> bool {
            false
        }
        async fn transcribe_segments(
            &self,
            audio: &[u8],
            config: &TranscriptionConfig,
        ) -> anyhow::Result<Transcript> {
            crate::transcription::segments_by_phrase(self, audio, config).await
        }
        fn max_segment_chunk_secs(&self, _config: &TranscriptionConfig) -> Option<u32> {
            None
        }
    }

    #[tokio::test(start_paused = true)]
    async fn phrase_backend_rate_limits_are_not_retried_twice() {
        let backend = Arc::new(LimitedPhrases {
            calls: AtomicUsize::new(0),
        });
        let (tmp, _) = recording(&[(TrackKind::Mic, 0, Some(track(4.0, &[(1.0, 1.0, 3000)])))]);
        let started = tokio::time::Instant::now();
        let err = process_recording(
            tmp.path(),
            &config(),
            Some(backend.clone() as Arc<dyn TranscriptionBackend>),
        )
        .await
        .unwrap_err();
        // One phrase budget (300 s at 7 s a wait is 43 retries), not one
        // per chunk-level retry on top.
        assert!(
            started.elapsed() <= RATE_LIMIT_BUDGET,
            "{:?}",
            started.elapsed()
        );
        assert_eq!(backend.calls.load(Ordering::Relaxed), 43);
        let text = format!("{err:#}");
        assert!(
            text.contains("HTTP 429") && text.contains("retry resumes from the failed chunk"),
            "{text}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn rate_limits_are_waited_out() {
        let backend = FakeBackend::with_failures(&[(3000, "patience")], None, None, 2);
        let (tmp, _) = recording(&[(TrackKind::Mic, 0, Some(track(4.0, &[(1.0, 1.0, 3000)])))]);
        let started = tokio::time::Instant::now();
        let meta = run(tmp.path(), backend.clone()).await;
        assert_eq!(backend.calls.load(Ordering::Relaxed), 3);
        assert_eq!(meta.segments[0].text, "patience");
        // Two 7 s Retry-After waits on the paused clock.
        assert!(started.elapsed() >= Duration::from_secs(14));
    }

    #[tokio::test(start_paused = true)]
    async fn endless_rate_limits_fail_with_a_clear_error() {
        let backend = FakeBackend::with_failures(&[(3000, "never")], None, None, usize::MAX);
        let (tmp, _) = recording(&[(TrackKind::Mic, 0, Some(track(4.0, &[(1.0, 1.0, 3000)])))]);
        let err = process_recording(
            tmp.path(),
            &config(),
            Some(backend as Arc<dyn TranscriptionBackend>),
        )
        .await
        .unwrap_err();
        let text = format!("{err:#}");
        assert!(
            text.contains("HTTP 429") && text.contains("slow down"),
            "{text}"
        );
    }

    #[test]
    fn streamed_scan_splits_like_the_in_memory_track() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("mic.wav");
        let mut samples = track(
            95.0,
            &[
                (1.0, 9.0, 3000),
                (12.0, 20.0, 2000),
                (40.0, 30.0, 4000),
                (75.0, 15.0, 600),
            ],
        );
        samples.extend([9i16; 123]); // a partial last frame
        write_wav(&path, &samples);

        let scan = scan_track(&path).unwrap();
        assert_eq!(scan.len, samples.len());
        // Every frame, the partial last one included.
        assert_eq!(scan.energies.len(), speech::frame_count(samples.len()));
        assert_eq!(scan.energies.last(), Some(&rms_energy(&[9i16; 123])));
        for cap in [10, 25, 60] {
            let expected = crate::recording::split::split_at_silence(
                &samples,
                RATE,
                cap,
                SILENCE_RMS_THRESHOLD,
            );
            let ranges =
                split_by_energies(&scan.energies, scan.len, RATE, cap, SILENCE_RMS_THRESHOLD);
            assert_eq!(ranges, expected, "cap {cap}");
            for r in ranges {
                assert_eq!(read_range(&path, r.clone()).unwrap(), samples[r]);
            }
        }
    }

    fn seg(start_ms: u64, end_ms: u64, text: &str) -> Segment {
        Segment {
            start_ms,
            end_ms,
            text: text.into(),
            speaker: None,
            words: None,
        }
    }

    #[test]
    fn echo_drop_needs_overlap_and_matching_text() {
        let system = [seg(
            10_000,
            14_000,
            "the quarterly numbers look much better than last year",
        )];
        let mic = vec![
            // Echo, slightly late.
            seg(
                10_400,
                14_300,
                "quarterly numbers look much better than last year",
            ),
            // Same words but far away in time: a real repeat, kept.
            seg(
                60_000,
                64_000,
                "the quarterly numbers look much better than last year",
            ),
            // Overlapping but different speech, kept.
            seg(12_000, 13_000, "can you share the slides please"),
            // Short interjection over the other side, kept.
            seg(11_000, 11_500, "yes"),
        ];
        let kept: Vec<_> = drop_echo_segments(mic, &system, ECHO_TOLERANCE_MS)
            .into_iter()
            .map(|s| s.start_ms)
            .collect();
        assert_eq!(kept, [60_000, 12_000, 11_000]);
    }

    #[test]
    fn gecos_full_name_is_the_first_field() {
        assert_eq!(
            gecos_full_name("Yosif Soliman,,,").as_deref(),
            Some("Yosif Soliman")
        );
        assert_eq!(gecos_full_name(""), None);
        assert_eq!(gecos_full_name(" ,room"), None);
    }

    #[test]
    fn configured_user_name_wins() {
        let recording = RecordingConfig {
            user_name: Some(" Ada ".into()),
            ..Default::default()
        };
        assert_eq!(user_name(&recording), "Ada");
        // Without one, something non-empty is always found.
        assert!(!user_name(&RecordingConfig::default()).is_empty());
    }
}
