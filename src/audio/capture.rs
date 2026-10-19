//! Audio capture using the `cpal` crate.
//!
//! Opens the configured input device (see [`super::device`]) at 16kHz mono
//! 16-bit and pushes audio chunks into a tokio mpsc channel for downstream
//! processing. A recording's system track can follow the default output
//! ([`AudioCaptureHandle::start_following_monitor`]): its capture moves to
//! the new default's monitor, with the gap filled with zeros so the track
//! keeps its timeline.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{BufferSize, SampleFormat, StreamConfig};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use super::device::{self, InputSource, ResolvedInput};
use super::{block_off_runtime, AudioChunk};

/// Desired sample rate for speech recognition.
pub const SAMPLE_RATE: u32 = 16_000;

/// Number of channels (mono).
pub(super) const CHANNELS: u16 = 1;

/// How long the caller waits for the capture thread to report that the
/// stream is running. Longer than [`device::PULSE_BUILD_TIMEOUT`], so a
/// stream-creation timeout surfaces as its own error first. It also covers
/// the cases cpal does not bound: a PulseAudio `play()` blocks until the
/// first data arrives, and listing sources on a hung server never returns.
const START_TIMEOUT: Duration = Duration::from_secs(5);

/// How often a following capture asks the sound server for the default
/// output. A query on the capture thread, never in a stream callback.
const FOLLOW_POLL: Duration = Duration::from_secs(2);

/// How many of a stream's latest chunks [`StreamClock`] keeps: about 3 s of
/// PulseAudio chunks, short enough that clock drift does not matter.
const CLOCK_WINDOW: usize = 32;

/// How much of a new stream's audio [`Splice`] holds before it decides where
/// that audio began: a stream may hand over what it buffered while starting
/// in one quick burst.
const SPLICE_HOLD: Duration = Duration::from_millis(600);

fn samples_to_duration(samples: usize) -> Duration {
    Duration::from_secs_f64(samples as f64 / f64::from(SAMPLE_RATE))
}

/// When a stream's audio began, estimated from its chunks: each chunk's
/// arrival minus the audio the stream delivered up to it is that start plus
/// the chunk's delivery delay, so the smallest is the best estimate. Old and
/// new stream are measured alike, so their common delay cancels out.
fn earliest_start(arrived: Instant, delivered: usize) -> Instant {
    arrived
        .checked_sub(samples_to_duration(delivered))
        .unwrap_or(arrived)
}

/// Where the current stream of a capture is on the wall clock, written by
/// its callbacks and read by the capture thread at a switch, without a lock:
/// the start estimates of its last [`CLOCK_WINDOW`] chunks and the samples
/// it delivered.
struct StreamClock {
    base: Instant,
    /// Nanoseconds after `base` (a start may precede it: signed), or
    /// `i64::MAX` for an empty slot.
    starts: [AtomicI64; CLOCK_WINDOW],
    next: AtomicUsize,
    samples: AtomicU64,
}

impl StreamClock {
    fn new() -> Self {
        Self {
            base: Instant::now(),
            starts: std::array::from_fn(|_| AtomicI64::new(i64::MAX)),
            next: AtomicUsize::new(0),
            samples: AtomicU64::new(0),
        }
    }

    /// A chunk of `len` samples arrived at `at`. Called by the one callback
    /// of the current stream.
    fn mark(&self, at: Instant, len: usize) {
        let delivered = self.samples.fetch_add(len as u64, Ordering::AcqRel) + len as u64;
        let start = earliest_start(at, delivered as usize);
        let nanos = if start >= self.base {
            i64::try_from(start.duration_since(self.base).as_nanos()).unwrap_or(i64::MAX - 1)
        } else {
            -i64::try_from(self.base.duration_since(start).as_nanos()).unwrap_or(i64::MAX)
        };
        let slot = self.next.fetch_add(1, Ordering::AcqRel) % CLOCK_WINDOW;
        self.starts[slot].store(nanos, Ordering::Release);
    }

    /// Where the stream's audio ended: its start estimate plus the audio it
    /// delivered. `None` before its first chunk.
    fn audio_end(&self) -> Option<Instant> {
        let start = self
            .starts
            .iter()
            .map(|s| s.load(Ordering::Acquire))
            .filter(|&n| n != i64::MAX)
            .min()?;
        let start = if start >= 0 {
            self.base + Duration::from_nanos(start as u64)
        } else {
            self.base - Duration::from_nanos(start.unsigned_abs())
        };
        Some(start + samples_to_duration(self.samples.load(Ordering::Acquire) as usize))
    }

    /// Forget the stream that was just dropped, before the next one starts.
    fn reset(&self) {
        for start in &self.starts {
            start.store(i64::MAX, Ordering::Release);
        }
        self.samples.store(0, Ordering::Release);
    }
}

/// Zero samples that put audio beginning at `began` right after audio that
/// ended at `resume_at`. An overlap (it began before the other ended) needs
/// none.
pub(crate) fn gap_fill(resume_at: Instant, began: Instant) -> usize {
    let gap = began.saturating_duration_since(resume_at);
    (gap.as_secs_f64() * f64::from(SAMPLE_RATE)).round() as usize
}

/// `samples` zeros, a second at a time.
fn silence(samples: usize) -> Vec<AudioChunk> {
    let second = SAMPLE_RATE as usize;
    (0..samples.div_ceil(second))
        .map(|i| vec![0; second.min(samples - i * second)])
        .collect()
}

/// Joins a following capture's new stream onto its track: holds the new
/// stream's first [`SPLICE_HOLD`] of audio, estimates where it began
/// ([`earliest_start`]), and lets it out behind the zeros that fill the time
/// since the old stream's audio ended ([`gap_fill`]), so the track keeps the
/// wall-clock timeline.
pub(crate) struct Splice {
    resume_at: Instant,
    held: Vec<AudioChunk>,
    held_samples: usize,
    began: Option<Instant>,
}

impl Splice {
    pub(crate) fn new(resume_at: Instant) -> Self {
        Self {
            resume_at,
            held: Vec::new(),
            held_samples: 0,
            began: None,
        }
    }

    /// Take a chunk that arrived `at`. Once [`SPLICE_HOLD`] of audio is in,
    /// returns the zeros and the held chunks to send, in order.
    pub(crate) fn push(&mut self, chunk: AudioChunk, at: Instant) -> Option<Vec<AudioChunk>> {
        self.held_samples += chunk.len();
        let began = earliest_start(at, self.held_samples);
        self.began = Some(self.began.map_or(began, |b| b.min(began)));
        self.held.push(chunk);
        (samples_to_duration(self.held_samples) >= SPLICE_HOLD).then(|| self.release())
    }

    /// The zeros and the held chunks, whatever was held so far.
    pub(crate) fn release(&mut self) -> Vec<AudioChunk> {
        let fill = self.began.map_or(0, |b| gap_fill(self.resume_at, b));
        let mut out = silence(fill);
        out.append(&mut self.held);
        self.held_samples = 0;
        out
    }
}

/// Where a stream's callback sends its chunks: through a [`Splice`] first
/// when the stream replaced another. A following capture shares it with its
/// thread, which retires it before dropping the stream, at a switch and at
/// the stop: cpal's PulseAudio host can still run the callback once after
/// the drop, and a chunk sent then, after the stream's end was read, would
/// put everything after it one chunk late. Retiring also lets out what the
/// splice holds, so a stop right after a switch loses nothing.
struct Feed {
    tx: mpsc::UnboundedSender<AudioChunk>,
    splice: Option<Splice>,
    retired: bool,
}

type SharedFeed = Arc<std::sync::Mutex<Feed>>;

fn lock_feed(feed: &SharedFeed) -> std::sync::MutexGuard<'_, Feed> {
    feed.lock().unwrap_or_else(|e| e.into_inner())
}

impl Feed {
    fn new(tx: mpsc::UnboundedSender<AudioChunk>, splice: Option<Splice>) -> Self {
        Self {
            tx,
            splice,
            retired: false,
        }
    }

    /// Send a chunk that arrived `at` and mark it on `clock`, together, so
    /// the capture thread never sees one without the other.
    fn send_marked(&mut self, chunk: AudioChunk, at: Instant, clock: &StreamClock) {
        if self.retired {
            return;
        }
        let len = chunk.len();
        self.send(chunk, at);
        if len > 0 {
            clock.mark(at, len);
        }
    }

    /// Let out what the splice holds and ignore every later chunk.
    fn retire(&mut self) {
        if let Some(mut splice) = self.splice.take() {
            for chunk in splice.release() {
                let _ = self.tx.send(chunk);
            }
        }
        self.retired = true;
    }

    fn send(&mut self, chunk: AudioChunk, at: Instant) {
        match self.splice.as_mut() {
            Some(splice) => {
                if let Some(out) = splice.push(chunk, at) {
                    self.splice = None;
                    for chunk in out {
                        // A closed channel means the capture is stopping.
                        let _ = self.tx.send(chunk);
                    }
                }
            }
            None => {
                let _ = self.tx.send(chunk);
            }
        }
    }
}

/// What every stream of one capture shares with its callbacks.
struct StreamHooks {
    /// Non-empty chunks delivered so far.
    chunks: Arc<AtomicU64>,
    clock: Arc<StreamClock>,
    /// Set by a stream's error callback, so a following capture reopens.
    failed: Arc<AtomicBool>,
}

/// A handle to a running audio capture session.
///
/// The `cpal::Stream` lives on a dedicated capture thread.
/// This handle provides only the receiver and a stop signal.
pub struct AudioCaptureHandle {
    /// Receiver end of the audio channel.
    receiver: Option<mpsc::UnboundedReceiver<AudioChunk>>,
    /// Signal to stop the capture thread.
    stop_signal: Arc<AtomicBool>,
    /// Join handle for the capture thread.
    thread_handle: Option<std::thread::JoinHandle<()>>,
}

impl Drop for AudioCaptureHandle {
    fn drop(&mut self) {
        // Signal the capture thread to stop.
        self.stop_signal.store(true, Ordering::Release);
        // Wait for the thread to finish (non-async, best-effort).
        if let Some(handle) = self.thread_handle.take() {
            handle.join().ok();
        }
    }
}

/// How a capture resolves its device name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CaptureMode {
    /// Dictation: `[audio] device`, an unknown name falls back to the default.
    Dictation,
    /// A recording track: an unknown name is an error.
    Strict,
}

impl CaptureMode {
    fn resolve(self, device: &str) -> anyhow::Result<ResolvedInput> {
        match self {
            Self::Dictation => device::resolve_input(device),
            Self::Strict => device::resolve_input_strict(device),
        }
    }

    /// What to suggest when a named device misbehaves.
    fn hint(self) -> &'static str {
        match self {
            Self::Dictation => "set [audio] device = \"default\"",
            Self::Strict => "check the [recording] mic_device and system_device settings",
        }
    }
}

impl AudioCaptureHandle {
    /// Start capturing from `device` (the `[audio] device` value) and
    /// optionally publish a normalized volume level.
    ///
    /// The capture runs on a dedicated thread. Audio chunks are sent through
    /// the internal channel; call `take_receiver()` to get the receiving end.
    pub fn start_with_level_tx(
        device: &str,
        level_tx: Option<tokio::sync::watch::Sender<f32>>,
    ) -> anyhow::Result<Self> {
        Self::start(device, level_tx, CaptureMode::Dictation, false)
    }

    /// Start capturing a recording track from `device`: a PulseAudio source
    /// (such as [`device::default_monitor_source`]), an ALSA PCM, or
    /// `"default"`. Unlike [`Self::start_with_level_tx`] an unknown name is
    /// an error, never the default input, and no level is published.
    pub fn start_strict(device: &str) -> anyhow::Result<Self> {
        Self::start(device, None, CaptureMode::Strict, false)
    }

    /// Start capturing the system track from `monitor` (the monitor of the
    /// default output, [`device::default_monitor_source`]) and follow the
    /// default output: every [`FOLLOW_POLL`] the capture thread asks for the
    /// default again, and when it changed, or the stream failed, it moves to
    /// the new default's monitor. The time between the old stream's audio
    /// and the new stream's is filled with zeros ([`Splice`]), so the track
    /// stays aligned with the mic.
    pub fn start_following_monitor(monitor: &str) -> anyhow::Result<Self> {
        Self::start(monitor, None, CaptureMode::Strict, true)
    }

    fn start(
        device: &str,
        level_tx: Option<tokio::sync::watch::Sender<f32>>,
        mode: CaptureMode,
        follow: bool,
    ) -> anyhow::Result<Self> {
        let device_name = device.to_string();
        let device = device_name.clone();
        let (tx, rx) = mpsc::unbounded_channel::<AudioChunk>();
        let stop_signal = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop_signal);

        // Channel to send back any initialization error from the thread.
        let (init_tx, init_rx) = std::sync::mpsc::channel::<anyhow::Result<()>>();

        let thread_handle = std::thread::Builder::new()
            .name("whisrs-audio".into())
            .spawn(move || {
                run_capture(&device, tx, stop_clone, init_tx, level_tx, mode, follow);
            })
            .context("failed to spawn audio capture thread")?;

        // Wait for initialization result. Opening a device can take seconds
        // (a PulseAudio handshake times out at 2 s), so keep it off the
        // async workers, and never wait unbounded: callers hold the daemon
        // state lock across this.
        let init_result = match block_off_runtime(|| init_rx.recv_timeout(START_TIMEOUT)) {
            Ok(result) => result,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // The thread may be stuck inside cpal. Tell it to stop and
                // detach it (dropping the JoinHandle): it exits on its own
                // once cpal returns, and nothing here ever joins it.
                stop_signal.store(true, Ordering::Release);
                drop(thread_handle);
                return Err(start_timeout_error(&device_name, mode));
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                anyhow::bail!("audio capture thread exited unexpectedly")
            }
        };
        init_result?;

        Ok(Self {
            receiver: Some(rx),
            stop_signal,
            thread_handle: Some(thread_handle),
        })
    }

    /// Take the receiver end of the audio channel.
    pub fn take_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<AudioChunk>> {
        self.receiver.take()
    }

    /// Signal the capture thread to stop (async-friendly).
    /// The channel will close once the thread exits. Callers reading
    /// from the receiver will see `None` after remaining chunks drain.
    pub fn stop(&mut self) {
        self.stop_signal.store(true, Ordering::Release);
    }

    /// Stop the audio capture and return all accumulated samples from the channel.
    pub async fn stop_and_collect(mut self) -> anyhow::Result<Vec<i16>> {
        // Signal the capture thread to stop.
        self.stop_signal.store(true, Ordering::Release);

        // Wait for the thread to finish.
        if let Some(handle) = self.thread_handle.take() {
            // Use spawn_blocking to avoid blocking the tokio runtime.
            tokio::task::spawn_blocking(move || {
                handle.join().ok();
            })
            .await?;
        }

        let mut all_samples = Vec::new();

        if let Some(mut rx) = self.receiver.take() {
            // Drain all remaining chunks from the channel.
            rx.close();
            while let Ok(chunk) = rx.try_recv() {
                all_samples.extend_from_slice(&chunk);
            }
        }

        info!("captured {} audio samples", all_samples.len());
        Ok(all_samples)
    }
}

/// The error for a device that did not start within [`START_TIMEOUT`].
fn start_timeout_error(device: &str, mode: CaptureMode) -> anyhow::Error {
    let secs = START_TIMEOUT.as_secs();
    if device::is_default_name(device) {
        anyhow::anyhow!(
            "the default audio input device did not start within {secs}s; \
             check that the microphone is connected and the sound server is running"
        )
    } else {
        anyhow::anyhow!(
            "audio device {} did not start within {secs}s; {}",
            device.trim(),
            mode.hint()
        )
    }
}

/// How long after `play()` to wait for the first callback before warning.
const FIRST_CHUNK_TIMEOUT: Duration = Duration::from_secs(1);

/// Upper bound on [`drain_after_stop`]: two of PipeWire-ALSA's 128 ms
/// default periods (the `BufferSize::Default` fallback), so a stream that
/// stopped delivering cannot hold up the stop.
const STOP_DRAIN_TIMEOUT: Duration = Duration::from_millis(250);

/// After the stop signal, keep the stream until one more chunk arrives.
/// Audio reaches the callback a whole period at a time, so dropping the
/// stream right away loses the partial period in the device buffer: the
/// end of the last word. The next chunk covers the moment stop was seen.
fn drain_after_stop(chunks: &AtomicU64) {
    let seen = chunks.load(Ordering::Acquire);
    let started = Instant::now();
    while chunks.load(Ordering::Acquire) == seen && started.elapsed() < STOP_DRAIN_TIMEOUT {
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Run the audio capture on the current thread.
///
/// Sends the initialization result through `init_tx`, then blocks until the
/// stop signal is set. The cpal Stream lives on this thread.
fn run_capture(
    device: &str,
    tx: mpsc::UnboundedSender<AudioChunk>,
    stop_signal: Arc<AtomicBool>,
    init_tx: std::sync::mpsc::Sender<anyhow::Result<()>>,
    level_tx: Option<tokio::sync::watch::Sender<f32>>,
    mode: CaptureMode,
    follow: bool,
) {
    let result = setup_and_run(device, tx, stop_signal, &init_tx, level_tx, mode, follow);
    if let Err(e) = result {
        // If init_tx hasn't been used yet, send the error.
        init_tx.send(Err(e)).ok();
    }
}

/// Resolve `device_name` and start a stream on it; `resume_at` is where the
/// capture's previous stream stopped, for the zero fill. A stale cached
/// PulseAudio connection is replaced once.
fn open_input(
    device_name: &str,
    tx: &mpsc::UnboundedSender<AudioChunk>,
    level_tx: &Option<tokio::sync::watch::Sender<f32>>,
    hooks: &StreamHooks,
    mode: CaptureMode,
    resume_at: Option<Instant>,
) -> anyhow::Result<(cpal::Stream, String, InputSource, SharedFeed)> {
    let resolved = mode.resolve(device_name)?;
    let cached_pulse_host = resolved.cached_pulse_host;
    match build_and_play(resolved, tx, level_tx, hooks, mode, resume_at) {
        Ok(opened) => Ok(opened),
        Err(e) if cached_pulse_host && device::is_host_level_error(&e) => {
            // The cached PulseAudio connection is stale (the server
            // restarted under it): reconnect once. A device-level error
            // (busy, unsupported config, timeout) keeps the connection.
            debug!(
                "opening input on the cached PulseAudio connection failed ({e:#}), reconnecting"
            );
            device::invalidate_pulse_host();
            let resolved = mode.resolve(device_name)?;
            build_and_play(resolved, tx, level_tx, hooks, mode, resume_at)
        }
        Err(e) => Err(e),
    }
}

/// A capture that follows the monitor of the default output.
struct Follower {
    /// The monitor being recorded (or, after a failed switch, the one last
    /// recorded).
    current: String,
    last_poll: Instant,
    /// Where the last stream's audio ended, kept across failed switches.
    resume_at: Option<Instant>,
    /// The current stream's feed, retired before the stream is dropped.
    feed: Option<SharedFeed>,
}

impl Follower {
    /// Retire the current stream's feed (see [`Feed`]).
    fn retire(&mut self) {
        if let Some(feed) = self.feed.take() {
            lock_feed(&feed).retire();
        }
    }

    /// Every [`FOLLOW_POLL`]: when the default output changed, or the stream
    /// failed or is gone, move to the default's monitor. The old stream is
    /// dropped first, so where its audio ended is final when the new stream
    /// is spliced on. A switch that fails leaves no stream and is tried
    /// again at the next poll; the fill then covers the whole outage.
    fn poll(
        &mut self,
        stream: &mut Option<cpal::Stream>,
        tx: &mpsc::UnboundedSender<AudioChunk>,
        hooks: &StreamHooks,
    ) {
        if self.last_poll.elapsed() < FOLLOW_POLL {
            return;
        }
        self.last_poll = Instant::now();
        let target = match device::default_monitor_source() {
            Ok(target) => target,
            Err(e) => {
                debug!("could not read the default output: {e:#}");
                return;
            }
        };
        let failed = hooks.failed.swap(false, Ordering::AcqRel);
        if target == self.current && !failed && stream.is_some() {
            return;
        }
        if target != self.current {
            info!(
                "the default output changed: recording the system audio from {target} \
                 instead of {}",
                self.current
            );
        } else {
            info!("reopening the system audio from {target}");
        }
        self.retire();
        if let Some(old) = stream.take() {
            drop(old);
            self.resume_at = hooks.clock.audio_end().or(self.resume_at);
            hooks.clock.reset();
        }
        let resume_at = self.resume_at.unwrap_or_else(Instant::now);
        match open_input(
            &target,
            tx,
            &None,
            hooks,
            CaptureMode::Strict,
            Some(resume_at),
        ) {
            Ok((opened, _, _, feed)) => {
                *stream = Some(opened);
                self.feed = Some(feed);
                self.current = target;
            }
            Err(e) => warn!("could not record the system audio from {target}: {e:#}; retrying"),
        }
    }
}

fn setup_and_run(
    device_name: &str,
    tx: mpsc::UnboundedSender<AudioChunk>,
    stop_signal: Arc<AtomicBool>,
    init_tx: &std::sync::mpsc::Sender<anyhow::Result<()>>,
    level_tx: Option<tokio::sync::watch::Sender<f32>>,
    mode: CaptureMode,
    follow: bool,
) -> anyhow::Result<()> {
    let hooks = StreamHooks {
        chunks: Arc::new(AtomicU64::new(0)),
        clock: Arc::new(StreamClock::new()),
        failed: Arc::new(AtomicBool::new(false)),
    };
    let chunks = Arc::clone(&hooks.chunks);
    let (stream, device, source, feed) =
        open_input(device_name, &tx, &level_tx, &hooks, mode, None)?;
    debug!("audio capture started at {SAMPLE_RATE}Hz mono i16");

    // Signal successful initialization. If the caller already gave up
    // (START_TIMEOUT), it set the stop signal and dropped the receiver:
    // release the stream and exit.
    if stop_signal.load(Ordering::Acquire) || init_tx.send(Ok(())).is_err() {
        debug!("audio capture started after the caller gave up; stopping");
        drop(stream);
        return Ok(());
    }

    // Block until stop is signaled. Keep the stream alive, and for a
    // following capture move it when the default output changes.
    let started = Instant::now();
    let mut watchdog_armed = true;
    let mut stream = Some(stream);
    let mut follower = follow.then(|| Follower {
        current: device_name.trim().to_string(),
        last_poll: Instant::now(),
        resume_at: None,
        feed: Some(feed),
    });
    while !stop_signal.load(Ordering::Acquire) {
        std::thread::sleep(std::time::Duration::from_millis(50));
        if watchdog_armed && started.elapsed() >= FIRST_CHUNK_TIMEOUT {
            watchdog_armed = false;
            if chunks.load(Ordering::Acquire) == 0 {
                warn_no_audio(&device, &source, mode);
            }
        }
        if let Some(follower) = follower.as_mut() {
            follower.poll(&mut stream, &tx, &hooks);
        }
    }

    debug!("audio capture stopping");
    if stream.is_some() {
        drain_after_stop(&chunks);
    }
    if let Some(follower) = follower.as_mut() {
        follower.retire();
    }
    if let Some(level_tx) = &level_tx {
        let _ = level_tx.send(0.0);
    }
    drop(stream);

    Ok(())
}

/// A stream that started but has delivered nothing. A stalled PulseAudio
/// source never gets here (its `play()` blocks until the first data, which
/// [`START_TIMEOUT`] reports), so this is about the device itself.
fn warn_no_audio(device: &str, source: &InputSource, mode: CaptureMode) {
    let secs = FIRST_CHUNK_TIMEOUT.as_secs();
    match source {
        InputSource::Named(..) => warn!(
            "no audio from input device {device} ({source}) {secs} s after starting; \
             check that the device is connected and not muted, or {}",
            mode.hint()
        ),
        _ => warn!(
            "no audio from input device {device} ({source}) {secs} s after starting; \
             check that the microphone is connected and not muted"
        ),
    }
}

/// Build and start the input stream for a resolved device. With `resume_at`
/// (a switch of a following capture) its audio is spliced on behind the
/// zeros that keep the timeline ([`Splice`]).
fn build_and_play(
    resolved: ResolvedInput,
    tx: &mpsc::UnboundedSender<AudioChunk>,
    level_tx: &Option<tokio::sync::watch::Sender<f32>>,
    hooks: &StreamHooks,
    mode: CaptureMode,
    resume_at: Option<Instant>,
) -> anyhow::Result<(cpal::Stream, String, InputSource, SharedFeed)> {
    let device = resolved.device;
    let device_label = device.to_string();
    info!(
        "using audio input device: {device_label} ({})",
        resolved.source
    );

    // Verify device support.
    let supported = device
        .supported_input_configs()
        .context("failed to query supported input configs")?;

    let mut found_match = false;
    for range in supported {
        if range.channels() == CHANNELS
            && range.min_sample_rate() <= SAMPLE_RATE
            && range.max_sample_rate() >= SAMPLE_RATE
            && range.sample_format() == SampleFormat::I16
        {
            found_match = true;
            break;
        }
    }

    if !found_match {
        warn!(
            "device may not natively support {SAMPLE_RATE}Hz mono i16; \
             cpal will attempt conversion"
        );
    }

    let feed: SharedFeed = Arc::new(std::sync::Mutex::new(Feed::new(
        tx.clone(),
        resume_at.map(Splice::new),
    )));
    let build = |buffer_size: BufferSize| {
        let config = StreamConfig {
            channels: CHANNELS,
            sample_rate: SAMPLE_RATE,
            buffer_size,
        };
        let feed = Arc::clone(&feed);
        let callback_level_tx = level_tx.clone();
        let chunks = Arc::clone(&hooks.chunks);
        let clock = Arc::clone(&hooks.clock);
        let failed = Arc::clone(&hooks.failed);
        device.build_input_stream(
            config,
            move |data: &[i16], _info: &cpal::InputCallbackInfo| {
                let now = Instant::now();
                if let Some(level_tx) = &callback_level_tx {
                    let _ = level_tx.send(audio_level(data));
                }
                lock_feed(&feed).send_marked(data.to_vec(), now, &clock);
                // Counted after the send, so a drain that sees the chunk
                // knows it is already queued for the collector.
                if !data.is_empty() {
                    chunks.fetch_add(1, Ordering::Release);
                }
            },
            move |err: cpal::Error| {
                error!("audio stream error: {err}");
                failed.store(true, Ordering::Release);
            },
            resolved.build_timeout,
        )
    };

    let build_started = Instant::now();
    let stream = match build(resolved.buffer_size) {
        Err(e) if retry_with_default_buffer(&resolved.source, resolved.buffer_size) => {
            debug!(
                "opening the default input with {:?} failed ({e}), retrying with the device's default buffer",
                resolved.buffer_size
            );
            build(BufferSize::Default)
        }
        result => result,
    };
    let stream = stream.map_err(|e| match resolved.build_timeout {
        // cpal gave up waiting on the server (a hung PulseAudio).
        Some(limit) if build_started.elapsed() >= limit => anyhow::Error::new(e).context(format!(
            "audio device {} did not start within {}s; {}",
            match &resolved.source {
                InputSource::Named(_, id) => id.as_str(),
                _ => device_label.as_str(),
            },
            limit.as_secs(),
            mode.hint()
        )),
        _ => anyhow::Error::new(e).context("failed to build audio input stream"),
    })?;

    stream.play().context("failed to start audio stream")?;
    Ok((stream, device_label, resolved.source, feed))
}

/// Whether a failed open of an ALSA input should be retried with
/// `BufferSize::Default`: the fixed period ([`device::ALSA_DEFAULT_INPUT_FRAMES`])
/// is a latency choice, never a reason for an ALSA device to fail.
fn retry_with_default_buffer(source: &InputSource, buffer_size: BufferSize) -> bool {
    matches!(buffer_size, BufferSize::Fixed(_))
        && matches!(
            source,
            InputSource::Default
                | InputSource::Fallback
                | InputSource::Named(device::DeviceHost::Alsa, _)
        )
}

fn audio_level(data: &[i16]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    let sum_squares: f32 = data
        .iter()
        .map(|sample| {
            let normalized = *sample as f32 / i16::MAX as f32;
            normalized * normalized
        })
        .sum();
    let rms = (sum_squares / data.len() as f32).sqrt();

    // Soft compressor: 1 - exp(-k*rms). k=18 maps typical speech RMS
    // (~0.05–0.15) to the 0.6–0.95 range, so the visualizer reaches the
    // top of its dynamic range during normal speech instead of hovering
    // around 30 % deflection.
    (1.0 - (-rms * 18.0).exp()).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use device::DeviceHost;

    #[test]
    fn default_input_retries_with_default_buffer() {
        let fixed = BufferSize::Fixed(device::ALSA_DEFAULT_INPUT_FRAMES);
        assert!(retry_with_default_buffer(&InputSource::Default, fixed));
        assert!(retry_with_default_buffer(&InputSource::Fallback, fixed));
        // Already on the default buffer: nothing to retry.
        assert!(!retry_with_default_buffer(
            &InputSource::Default,
            BufferSize::Default
        ));
        // A named ALSA device also falls back; PulseAudio keeps its own error.
        assert!(retry_with_default_buffer(
            &InputSource::Named(DeviceHost::Alsa, "x".into()),
            fixed
        ));
        assert!(!retry_with_default_buffer(
            &InputSource::Named(DeviceHost::PulseAudio, "x".into()),
            fixed
        ));
    }

    #[test]
    fn dictation_hint_is_unchanged() {
        // The dictation messages predate the recording feature; keep their wording.
        assert_eq!(
            CaptureMode::Dictation.hint(),
            "set [audio] device = \"default\""
        );
        let msg = start_timeout_error("usb-mic", CaptureMode::Dictation).to_string();
        assert_eq!(
            msg,
            "audio device usb-mic did not start within 5s; set [audio] device = \"default\""
        );
        let msg = start_timeout_error("x.monitor", CaptureMode::Strict).to_string();
        assert!(msg.contains("[recording]"), "{msg}");
        assert!(!msg.contains("[audio]"), "{msg}");
    }

    const MS: Duration = Duration::from_millis(1);

    #[test]
    fn gap_fill_counts_the_time_between_two_streams() {
        let t0 = Instant::now();
        assert_eq!(gap_fill(t0, t0 + 250 * MS), 4000);
        assert_eq!(gap_fill(t0, t0), 0);
        // The new audio began before the old one ended: nothing to fill.
        assert_eq!(gap_fill(t0 + 100 * MS, t0), 0);
        let zeros = silence(40_000);
        assert_eq!(
            zeros.iter().map(Vec::len).collect::<Vec<_>>(),
            [16_000, 16_000, 8000]
        );
        assert!(zeros.iter().flatten().all(|&s| s == 0));
        assert!(silence(0).is_empty());
    }

    /// The old stream's end comes from its least delayed recent chunk, not
    /// its last one, and a reset forgets it.
    #[test]
    fn stream_clock_estimates_where_the_audio_ended() {
        let clock = StreamClock::new();
        let t0 = clock.base;
        assert_eq!(clock.audio_end(), None);
        // 40 chunks of 100 ms; audio began at t0, each arrives 30 ms late
        // plus up to 15 ms of jitter, the least delayed one 30 ms late.
        for k in 1..=40u64 {
            let jitter = k * 7919 % 16;
            clock.mark(t0 + Duration::from_millis(100 * k + 30 + jitter), 1600);
        }
        let end = clock.audio_end().unwrap();
        assert_eq!(
            end.duration_since(t0),
            4030 * MS,
            "4 s of audio plus the delay"
        );
        clock.reset();
        assert_eq!(clock.audio_end(), None);
        // A stream whose audio began before the clock was made.
        let early = StreamClock::new();
        early.mark(early.base + 50 * MS, 1600);
        assert_eq!(early.audio_end(), Some(early.base + 50 * MS));
    }

    /// Feed `chunks` 100 ms chunks arriving at `arrivals` (ms after `t0`)
    /// to `splice`; what it let out once it had [`SPLICE_HOLD`].
    fn splice_out(splice: &mut Splice, t0: Instant, arrivals: &[u32]) -> Vec<AudioChunk> {
        let mut out = None;
        for (i, &at) in arrivals.iter().enumerate() {
            assert!(out.is_none(), "let out before the hold was full");
            out = splice.push(vec![i as i16 + 1; 1600], t0 + at * MS);
        }
        out.expect("not let out after the hold")
    }

    /// The splice holds its first 600 ms, then lets the zeros and the audio
    /// out in order; a stream that starts with a burst of buffered chunks is
    /// not over-filled.
    #[test]
    fn splice_places_the_new_audio_where_it_began() {
        let t0 = Instant::now();
        // Steady: the new audio began 250 ms after the old ended, each chunk
        // arriving 30 ms after its audio, like the old stream's.
        let mut steady = Splice::new(t0 + 30 * MS);
        let out = splice_out(&mut steady, t0, &[380, 480, 580, 680, 780, 880]);
        let zeros: usize = out.iter().take_while(|c| c[0] == 0).map(Vec::len).sum();
        assert_eq!(zeros, 4000, "250 ms");
        let audio: Vec<i16> = out.iter().skip(1).map(|c| c[0]).collect();
        assert_eq!(audio, [1, 2, 3, 4, 5, 6], "every chunk, in order");

        // Burst: the audio began 50 ms after the old ended, and its first
        // three chunks were buffered and arrive at once. Judged by the first
        // chunk alone it would begin 200 ms later than it did; the third
        // chunk and the steady ones after it show where it began.
        let mut burst = Splice::new(t0 + 30 * MS);
        let out = splice_out(&mut burst, t0, &[380, 381, 382, 480, 580, 680]);
        let zeros: usize = out.iter().filter(|c| c[0] == 0).map(Vec::len).sum();
        assert_eq!(zeros, 800, "50 ms, not 250 ms");
    }

    /// A retired feed lets out what it held, behind its zeros (a stop right
    /// after a switch), and ignores a callback that comes after its stream
    /// was dropped, neither sending nor counting it.
    #[test]
    fn a_retired_feed_ignores_late_chunks() {
        let t0 = Instant::now();
        let clock = StreamClock::new();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut feed = Feed::new(tx, Some(Splice::new(t0)));
        feed.send_marked(vec![5; 1600], t0 + 200 * MS, &clock);
        let end = clock.audio_end();
        assert!(rx.try_recv().is_err(), "held");
        feed.retire();
        assert_eq!(rx.try_recv().unwrap(), vec![0; 1600], "the zeros");
        assert_eq!(rx.try_recv().unwrap(), vec![5; 1600], "the held chunk");
        feed.send_marked(vec![6; 1600], t0 + 300 * MS, &clock);
        assert!(rx.try_recv().is_err(), "a late chunk is not sent");
        assert_eq!(clock.audio_end(), end, "nor counted");
    }

    /// End to end on synthetic timestamps: an old stream, a switch with an
    /// outage and a bursty start, and the track holds every sample at its
    /// place on the wall clock.
    #[test]
    fn a_switch_keeps_the_track_on_the_timeline() {
        let clock = StreamClock::new();
        let t0 = clock.base;
        let (tx, mut rx) = mpsc::unbounded_channel();
        // Old stream: audio from t0, 100 ms chunks arriving 40 ms late with
        // jitter, for 3 s.
        let mut old = Feed::new(tx.clone(), None);
        for k in 1..=30u64 {
            let at = t0 + Duration::from_millis(100 * k + 40 + k % 3 * 5);
            old.send_marked(vec![1; 1600], at, &clock);
        }
        // Retired at the switch: a callback after the drop is ignored.
        old.retire();
        old.send_marked(vec![1; 1600], t0 + 3145 * MS, &clock);
        let resume_at = clock.audio_end().unwrap();
        clock.reset();
        // The new stream's audio begins at 3.4 s (a 400 ms outage) and its
        // first 200 ms come in one burst.
        let mut new = Feed::new(tx, Some(Splice::new(resume_at)));
        let arrivals = [3640, 3641, 3740, 3840, 3940];
        for at in arrivals {
            new.send(vec![2; 1600], t0 + at * MS);
        }
        // The stop comes before the splice let it out.
        new.retire();
        let mut track = Vec::new();
        while let Ok(chunk) = rx.try_recv() {
            track.extend(chunk);
        }
        let first_new = track.iter().position(|&s| s == 2).unwrap();
        // 3.4 s into the track, as on the wall clock.
        assert_eq!(first_new, 3400 * 16);
        assert!(track[3000 * 16..first_new].iter().all(|&s| s == 0));
        assert_eq!(track.len(), (3400 + 500) * 16);
    }

    #[test]
    fn drain_returns_on_next_chunk() {
        let chunks = Arc::new(AtomicU64::new(3));
        let feeder = Arc::clone(&chunks);
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            feeder.fetch_add(1, Ordering::Release);
        });
        let started = Instant::now();
        drain_after_stop(&chunks);
        t.join().unwrap();
        assert!(started.elapsed() < STOP_DRAIN_TIMEOUT);
        assert_eq!(chunks.load(Ordering::Acquire), 4);
    }

    #[test]
    fn drain_is_bounded_without_audio() {
        let started = Instant::now();
        drain_after_stop(&AtomicU64::new(0));
        assert!(started.elapsed() >= STOP_DRAIN_TIMEOUT);
        assert!(started.elapsed() < STOP_DRAIN_TIMEOUT + Duration::from_millis(200));
    }
}
