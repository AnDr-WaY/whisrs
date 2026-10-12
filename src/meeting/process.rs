//! Meeting post-processing: transcribe both tracks and build the transcript.
//!
//! [`process_meeting`] runs once a recording has stopped (and its WAVs have
//! been repaired by the caller). Each track is cut into backend-sized chunks
//! at silence ([`super::split`]), each chunk is transcribed into timed
//! segments, segments are filtered and moved onto the meeting timeline, mic
//! segments that merely echo the system track are dropped, and the two
//! channels are labelled and merged. The result lands in `meeting.json` and
//! `<id>.md`.

use std::ffi::CStr;
use std::fs;
use std::io::Write;
use std::ops::Range;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context};
use audio_silence_gate::{audio_gate_reason, rms_energy, SILENCE_RMS_THRESHOLD};
use filler_remove::FillerFilter;
use prompt_echo::is_prompt_echo;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use super::render::render_markdown;
use super::split::{frame_len, split_by_energies};
use super::store::{self, CACHE_DIR};
use super::{MeetingMeta, MeetingState, Segment, Speaker, TrackKind, TranscriptSegment};
use crate::audio::wav::encode_wav;
use crate::replacements::Replacer;
use crate::transcription::factory::{
    backend_identity, canonical_backend_name, create_backend_named, meeting_backend_name,
    meeting_transcription_config,
};
use crate::transcription::retry::{retry_rate_limited, RateLimitExhausted, RATE_LIMIT_BUDGET};
use crate::transcription::{TranscriptionBackend, TranscriptionConfig};
use crate::{Config, MeetingConfig};

/// Meeting tracks are recorded at this rate (and `encode_wav` writes it).
const SAMPLE_RATE: u32 = super::SAMPLE_RATE;

/// Chunks shorter than this are skipped like an accidental dictation tap.
const GATE_MIN_MS: u64 = 300;

/// How far apart a mic segment and the system audio it echoes may be.
pub const ECHO_TOLERANCE_MS: u64 = 1500;

/// Transcribe a stopped meeting and write its transcript.
///
/// Marks the meeting `processing`, transcribes every recorded track with the
/// meeting backend ([`meeting_backend_name`]), and on success writes
/// `meeting.json` with status `done` plus `<id>.md`, returning the new meta.
/// `dictation_backend` is reused when the meeting backend is the dictation
/// one, so a loaded whisper model is not loaded twice. On error the meta is
/// left as `processing`; the caller marks it failed.
pub async fn process_meeting(
    dir: &Path,
    config: &Config,
    dictation_backend: Option<Arc<dyn TranscriptionBackend>>,
) -> anyhow::Result<MeetingMeta> {
    let mut meta = store::read_meta(dir)?;
    meta.status = MeetingState::Processing;
    meta.error = None;
    store::write_meta(dir, &meta)?;

    let name = meeting_backend_name(config)?;
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
    let tcfg = meeting_transcription_config(&name, config);
    info!(
        "processing meeting {} with {name} (model {})",
        meta.id, tcfg.model
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
            .sends_prompt(&tcfg)
            .then(|| tcfg.prompt.clone())
            .flatten(),
        filler,
        replacer,
    };

    let cache = ChunkCache {
        dir: dir.join(CACHE_DIR),
        fingerprint: cache_fingerprint(&name, config, &tcfg),
    };
    let mut mic = Vec::new();
    let mut system = Vec::new();
    let mut readable_tracks = 0usize;
    for track in meta.tracks.clone() {
        let path = dir.join(&track.file);
        if !path.exists() {
            let note = format!("{} audio missing: {}", track.kind.file_stem(), track.file);
            warn!("meeting {}: {note}", meta.id);
            if !meta.errors.contains(&note) {
                meta.errors.push(note);
            }
            continue;
        }
        readable_tracks += 1;
        let job = TrackJob {
            path: &path,
            kind: track.kind,
            offset_ms: track.offset_ms,
        };
        let segments = transcribe_track(backend.as_ref(), &tcfg, &filters, &cache, &job)
            .await
            .with_context(|| {
                format!("failed to transcribe the {} track", track.kind.file_stem())
            })?;
        info!(
            "meeting {}: {} track gave {} segment(s)",
            meta.id,
            track.kind.file_stem(),
            segments.len()
        );
        match track.kind {
            TrackKind::Mic => mic.extend(segments),
            TrackKind::System => system.extend(segments),
        }
    }
    if readable_tracks == 0 {
        bail!("no recorded audio found for meeting {}", meta.id);
    }

    let before = mic.len();
    let mic = drop_echo_segments(mic, &system, ECHO_TOLERANCE_MS);
    if mic.len() < before {
        info!(
            "meeting {}: dropped {} mic segment(s) echoing the system audio",
            meta.id,
            before - mic.len()
        );
    }

    meta.speakers = speakers_for(&meta, config);
    meta.segments = merge_channels(mic, system);
    meta.backend = Some(name);
    meta.status = MeetingState::Done;
    meta.error = None;

    write_private(
        &store::transcript_path(dir, &meta.id),
        render_markdown(&meta).as_bytes(),
    )?;
    store::write_meta(dir, &meta)?;
    cache.clear();
    Ok(meta)
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
                debug!("dropping meeting segment as a prompt echo: {text:?}");
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
fn open_track(path: &Path) -> anyhow::Result<hound::WavReader<std::io::BufReader<fs::File>>> {
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
    /// RMS of the whole track, as [`rms_energy`] computes it.
    rms: f64,
    /// RMS of every whole frame ([`frame_len`]), for [`split_by_energies`].
    energies: Vec<f64>,
}

impl TrackScan {
    /// The whole-track check of [`audio_gate_reason`]: nothing to send when
    /// the track is empty, shorter than [`GATE_MIN_MS`], or silent.
    fn is_skippable(&self) -> bool {
        let duration_ms = self.len as u64 * 1000 / SAMPLE_RATE as u64;
        self.len == 0 || duration_ms < GATE_MIN_MS || self.rms < SILENCE_RMS_THRESHOLD
    }
}

/// Read a track once, computing its length and energies. Blocking.
fn scan_track(path: &Path) -> anyhow::Result<TrackScan> {
    let mut reader = open_track(path)?;
    let frame = frame_len(SAMPLE_RATE as usize);
    let mut buf = Vec::with_capacity(frame);
    let mut energies = Vec::new();
    let mut sum_squares = 0.0f64;
    let mut len = 0usize;
    for sample in reader.samples::<i16>() {
        let sample = sample.with_context(|| format!("failed to read {}", path.display()))?;
        sum_squares += f64::from(sample) * f64::from(sample);
        len += 1;
        buf.push(sample);
        if buf.len() == frame {
            energies.push(rms_energy(&buf));
            buf.clear();
        }
    }
    let rms = if len == 0 {
        0.0
    } else {
        (sum_squares / len as f64).sqrt() / f64::from(i16::MAX)
    };
    Ok(TrackScan { len, rms, energies })
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

/// The track being transcribed.
struct TrackJob<'a> {
    path: &'a Path,
    kind: TrackKind,
    offset_ms: i64,
}

/// Transcribe one track into filtered segments on the meeting timeline.
///
/// Peak memory is one chunk: the track is scanned once for its frame
/// energies, cut by [`split_by_energies`], and each chunk is read from disk
/// only when it is sent. A chunk already in `cache` is not sent again.
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
    if scan.is_skippable() {
        debug!("track is empty or silent; skipping");
        return Ok(Vec::new());
    }

    let rate = SAMPLE_RATE as usize;
    let chunks = match backend.max_segment_chunk_secs(tcfg) {
        Some(secs) => split_by_energies(
            &scan.energies,
            scan.len,
            rate,
            secs as usize,
            SILENCE_RMS_THRESHOLD,
        ),
        None => std::iter::once(0..scan.len).collect(),
    };
    drop(scan);

    let mut out = Vec::new();
    for range in chunks {
        let raw = match cache.load(job.kind, &range) {
            Some(segments) => {
                debug!(
                    "reusing cached {} chunk {}..{}",
                    job.kind.file_stem(),
                    range.start,
                    range.end
                );
                segments
            }
            None => {
                let chunk = {
                    let path = path.clone();
                    let range = range.clone();
                    blocking(move || read_range(&path, range)).await?
                };
                if let Some(reason) =
                    audio_gate_reason(&chunk, SAMPLE_RATE, GATE_MIN_MS, SILENCE_RMS_THRESHOLD)
                {
                    debug!(
                        "skipping {} chunk at sample {}",
                        reason.as_str(),
                        range.start
                    );
                    continue;
                }
                let wav = encode_wav(&chunk)?;
                drop(chunk);
                let transcript = transcribe_with_retry(backend, &wav, tcfg).await?;
                if let Err(e) = cache.store(job.kind, &range, &transcript.segments) {
                    warn!("could not cache a transcribed chunk: {e:#}");
                }
                transcript.segments
            }
        };
        let chunk_ms = (range.start as u64 * 1000 / SAMPLE_RATE as u64) as i64;
        let offset_ms = job.offset_ms;
        for seg in raw {
            let Some(text) = filters.apply(&seg.text) else {
                continue;
            };
            let shift = |ms: u64| (offset_ms + chunk_ms + ms as i64).max(0) as u64;
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
/// [`RATE_LIMIT_BUDGET`]. Phrase backends retry each phrase inside
/// `transcribe_segments` and give up with a [`RateLimitExhausted`], which
/// this layer does not retry again, so the two waits never multiply.
async fn transcribe_with_retry(
    backend: &dyn TranscriptionBackend,
    wav: &[u8],
    tcfg: &TranscriptionConfig,
) -> anyhow::Result<crate::transcription::Transcript> {
    retry_rate_limited(RATE_LIMIT_BUDGET, || backend.transcribe_segments(wav, tcfg))
        .await
        .map_err(|err| {
            if err.downcast_ref::<RateLimitExhausted>().is_some() {
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

/// Cache of finished chunks for one meeting and backend. Entries hold the
/// backend's segments before filtering, relative to the chunk, so a config
/// change to the filters still applies on reuse. An entry is reused only
/// when its fingerprint matches exactly: another backend, model file,
/// language, prompt or vocabulary ignores it.
struct ChunkCache {
    dir: PathBuf,
    fingerprint: String,
}

impl ChunkCache {
    fn path(&self, kind: TrackKind, range: &Range<usize>) -> PathBuf {
        self.dir.join(format!(
            "{}-{}-{}.json",
            kind.file_stem(),
            range.start,
            range.end
        ))
    }

    fn load(&self, kind: TrackKind, range: &Range<usize>) -> Option<Vec<Segment>> {
        let data = fs::read(self.path(kind, range)).ok()?;
        let cached: CachedChunk = serde_json::from_slice(&data).ok()?;
        (cached.fingerprint == self.fingerprint).then_some(cached.segments)
    }

    fn store(
        &self,
        kind: TrackKind,
        range: &Range<usize>,
        segments: &[Segment],
    ) -> anyhow::Result<()> {
        fs::create_dir_all(&self.dir)
            .with_context(|| format!("failed to create {}", self.dir.display()))?;
        fs::set_permissions(&self.dir, fs::Permissions::from_mode(0o700))?;
        let cached = CachedChunk {
            fingerprint: self.fingerprint.clone(),
            segments: segments.to_vec(),
        };
        write_private(&self.path(kind, range), &serde_json::to_vec(&cached)?)
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

/// One speaker per recorded channel.
fn speakers_for(meta: &MeetingMeta, config: &Config) -> Vec<Speaker> {
    let has = |kind| meta.tracks.iter().any(|t| t.kind == kind);
    let mut speakers = Vec::new();
    if has(TrackKind::Mic) {
        speakers.push(Speaker {
            id: speaker_id(TrackKind::Mic).into(),
            name: user_name(&MeetingConfig::effective(config)),
            source: "self".into(),
        });
    }
    if has(TrackKind::System) {
        speakers.push(Speaker {
            id: speaker_id(TrackKind::System).into(),
            name: "Others".into(),
            source: "channel".into(),
        });
    }
    speakers
}

/// Name for the "me" speaker: `[meeting] user_name`, else the passwd full
/// name, else `$USER`, else "Me".
fn user_name(meeting: &MeetingConfig) -> String {
    meeting
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

/// Write a file atomically with mode 0600, like `meeting.json`.
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
    use crate::meeting::TrackMeta;
    use crate::transcription::{RetryableHttpError, Transcript};
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
            unreachable!("meetings ask for segments")
        }
        fn sends_prompt(&self, _config: &TranscriptionConfig) -> bool {
            false
        }
        async fn transcribe_segments(
            &self,
            audio: &[u8],
            _config: &TranscriptionConfig,
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

    /// `secs` of silence with `(start_s, len_s, amplitude)` square bursts.
    fn track(secs: f64, bursts: &[(f64, f64, i16)]) -> Vec<i16> {
        let mut out = vec![0i16; (secs * RATE as f64) as usize];
        for &(start, len, amp) in bursts {
            let from = (start * RATE as f64) as usize;
            let to = ((start + len) * RATE as f64) as usize;
            for (i, s) in out[from..to].iter_mut().enumerate() {
                *s = if i % 2 == 0 { amp } else { -amp };
            }
        }
        out
    }

    fn write_wav(path: &Path, samples: &[i16]) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, encode_wav(samples).unwrap()).unwrap();
    }

    /// A stopped meeting in a temp dir with the given tracks.
    fn meeting(tracks: &[(TrackKind, i64, Option<Vec<i16>>)]) -> (tempfile::TempDir, MeetingMeta) {
        let tmp = tempfile::tempdir().unwrap();
        let mut meta = MeetingMeta::new("2026-10-02_1400".into(), None, chrono::Local::now());
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
             [meeting]\nuser_name = \"Tester\"\n",
        )
        .unwrap()
    }

    async fn run(dir: &Path, backend: Arc<FakeBackend>) -> MeetingMeta {
        process_meeting(
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
        let (tmp, _) = meeting(&[
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

        assert_eq!(meta.status, MeetingState::Done);
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

    /// `[replacements]` reach meeting transcripts, after filler removal like
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
        let (tmp, _) = meeting(&[
            (TrackKind::Mic, 0, Some(track(4.0, &[(1.0, 1.0, 3000)]))),
            (TrackKind::System, 0, Some(track(4.0, &[(2.5, 1.0, 6000)]))),
        ]);
        let mut cfg = config();
        cfg.replacements = [
            ("package build".to_string(), "PKGBUILD".to_string()),
            ("drop me".to_string(), String::new()),
        ]
        .into();
        let meta = process_meeting(
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
        let backend = FakeBackend::new(
            &[(3000, "first"), (4000, "second"), (5000, "third")],
            Some(10),
        );
        let (tmp, _) = meeting(&[(
            TrackKind::Mic,
            500,
            Some(track(
                27.0,
                &[(2.0, 1.0, 3000), (12.0, 1.0, 4000), (22.0, 1.0, 5000)],
            )),
        )]);
        let meta = run(tmp.path(), backend.clone()).await;

        assert!(
            backend.calls.load(Ordering::Relaxed) >= 3,
            "27 s at a 10 s cap is chunked"
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

    #[tokio::test]
    async fn mic_echo_of_system_audio_is_dropped() {
        let line = "we should ship the release on friday morning";
        let backend = FakeBackend::new(
            &[(2000, line), (6000, line), (3000, "sounds good to me")],
            None,
        );
        let (tmp, _) = meeting(&[
            (
                TrackKind::Mic,
                0,
                Some(track(10.0, &[(1.2, 2.0, 2000), (6.0, 1.0, 3000)])),
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
        let (tmp, _) = meeting(&[
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

        let (tmp, _) = meeting(&[
            (TrackKind::Mic, 0, Some(track(4.0, &[(1.0, 1.0, 3000)]))),
            (TrackKind::System, 0, None),
        ]);
        let meta = run(tmp.path(), FakeBackend::new(&[(3000, "only me")], None)).await;
        assert_eq!(meta.segments.len(), 1);
        assert_eq!(meta.errors, ["system audio missing: audio/system.wav"]);
    }

    #[tokio::test]
    async fn no_audio_at_all_is_an_error() {
        let (tmp, _) = meeting(&[(TrackKind::Mic, 0, None)]);
        let err = process_meeting(
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
            MeetingState::Processing
        );
    }

    #[tokio::test]
    async fn realtime_meeting_backend_is_refused_before_transcribing() {
        let (tmp, _) = meeting(&[(TrackKind::Mic, 0, Some(track(2.0, &[(0.5, 1.0, 3000)])))]);
        let mut cfg = config();
        cfg.general.backend = "openai-realtime".into();
        cfg.meeting = None;
        let backend = FakeBackend::new(&[], None);
        let err = process_meeting(
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
        let (tmp, _) = meeting(&[(TrackKind::Mic, 0, Some(samples))]);

        // The third chunk's request fails: the first two are cached.
        let failing = FakeBackend::with_failures(&texts, Some(10), Some(2), 0);
        let err = process_meeting(
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
        let working = FakeBackend::new(&texts, Some(10));
        let meta = run(tmp.path(), working.clone()).await;
        assert_eq!(working.calls.load(Ordering::Relaxed), 1);
        let texts: Vec<_> = meta.segments.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, ["first", "second", "third"]);
        assert!(!cache_dir.exists(), "the cache is dropped once done");
    }

    #[test]
    fn cache_ignores_another_backend_or_model() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = |backend: &str, model: &str| ChunkCache {
            dir: tmp.path().join(CACHE_DIR),
            fingerprint: format!("{backend}/{model}"),
        };
        let segments = vec![seg(0, 1000, "hi")];
        cache("groq", "m1")
            .store(TrackKind::System, &(0..16_000), &segments)
            .unwrap();
        assert_eq!(
            cache("groq", "m1").load(TrackKind::System, &(0..16_000)),
            Some(segments)
        );
        assert_eq!(
            cache("openai", "m1").load(TrackKind::System, &(0..16_000)),
            None
        );
        assert_eq!(
            cache("groq", "m2").load(TrackKind::System, &(0..16_000)),
            None
        );
        assert_eq!(cache("groq", "m1").load(TrackKind::Mic, &(0..16_000)), None);
        assert_eq!(
            cache("groq", "m1").load(TrackKind::System, &(0..8_000)),
            None
        );
    }

    fn fingerprint_of(toml_text: &str) -> String {
        let config: Config = toml::from_str(toml_text).unwrap();
        let name = meeting_backend_name(&config).unwrap();
        let tcfg = meeting_transcription_config(&name, &config);
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
        // So do a prompt and vocabulary.
        let prompted = base.replace("language", "prompt = \"Standup\"\nlanguage");
        assert_ne!(fp, fingerprint_of(&prompted));
        let vocab = base.replace("language", "vocabulary = [\"whisrs\"]\nlanguage");
        assert_ne!(fp, fingerprint_of(&vocab));

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

    #[tokio::test]
    async fn a_language_change_retranscribes_cached_chunks() {
        let texts = [(3000, "first"), (4000, "second")];
        let samples = track(17.0, &[(2.0, 1.0, 3000), (12.0, 1.0, 4000)]);
        let (tmp, _) = meeting(&[(TrackKind::Mic, 0, Some(samples))]);
        // The second chunk fails: the first is cached under English.
        let failing = FakeBackend::with_failures(&texts, Some(10), Some(1), 0);
        process_meeting(
            tmp.path(),
            &config(),
            Some(failing as Arc<dyn TranscriptionBackend>),
        )
        .await
        .unwrap_err();
        assert_eq!(fs::read_dir(tmp.path().join(CACHE_DIR)).unwrap().count(), 1);

        let mut german = config();
        german.general.language = "de".into();
        let working = FakeBackend::new(&texts, Some(10));
        process_meeting(
            tmp.path(),
            &german,
            Some(working.clone() as Arc<dyn TranscriptionBackend>),
        )
        .await
        .unwrap();
        // Both chunks sent again: nothing English is reused.
        assert_eq!(working.calls.load(Ordering::Relaxed), 2);
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
        let (tmp, _) = meeting(&[(TrackKind::Mic, 0, Some(track(4.0, &[(1.0, 1.0, 3000)])))]);
        let started = tokio::time::Instant::now();
        let err = process_meeting(
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
        let (tmp, _) = meeting(&[(TrackKind::Mic, 0, Some(track(4.0, &[(1.0, 1.0, 3000)])))]);
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
        let (tmp, _) = meeting(&[(TrackKind::Mic, 0, Some(track(4.0, &[(1.0, 1.0, 3000)])))]);
        let err = process_meeting(
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
        assert_eq!(scan.rms, rms_energy(&samples));
        for cap in [10, 25, 60] {
            let expected =
                crate::meeting::split::split_at_silence(&samples, RATE, cap, SILENCE_RMS_THRESHOLD);
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
        let meeting = MeetingConfig {
            user_name: Some(" Ada ".into()),
            ..Default::default()
        };
        assert_eq!(user_name(&meeting), "Ada");
        // Without one, something non-empty is always found.
        assert!(!user_name(&MeetingConfig::default()).is_empty());
    }
}
