//! Transcription backends: trait definition and implementations.

pub mod asr_sidecar;
pub mod deepgram;
pub mod groq;
pub mod local_parakeet;
pub mod local_vosk;
#[cfg(feature = "local-whisper")]
pub mod local_whisper;
#[cfg(not(feature = "local-whisper"))]
pub mod local_whisper {
    //! Stub when local-whisper feature is disabled.
    pub struct LocalWhisperBackend;
    impl LocalWhisperBackend {
        pub fn new(_model_path: String) -> Self {
            Self
        }
        pub fn with_flash_attn(_model_path: String, _flash_attn: bool) -> Self {
            Self
        }
        pub fn with_segmentation(self, _mode: &str, _phrase_silence_ms: u64) -> Self {
            self
        }
    }
    #[async_trait::async_trait]
    impl super::TranscriptionBackend for LocalWhisperBackend {
        async fn transcribe(
            &self,
            _audio: &[u8],
            _config: &super::TranscriptionConfig,
        ) -> anyhow::Result<String> {
            anyhow::bail!("local-whisper feature not enabled — rebuild with: cargo build --features local-whisper")
        }

        // Unreachable: `transcribe` bails before a request exists. `false`
        // is what the rule above the trait method yields — read the answer off
        // the request-building code, and this build has none — and it is the
        // safe direction, since a stale `true` is the exact shape of #133.
        // The real backend answers `true` off its own `set_initial_prompt`
        // call; the divergence is deliberate and unobservable.
        fn sends_prompt(&self, _config: &super::TranscriptionConfig) -> bool {
            false
        }

        async fn transcribe_segments(
            &self,
            _audio: &[u8],
            _config: &super::TranscriptionConfig,
        ) -> anyhow::Result<super::Transcript> {
            anyhow::bail!("local-whisper feature not enabled — rebuild with: cargo build --features local-whisper")
        }

        // Matches the real backend so chunking does not depend on the build.
        fn max_segment_chunk_secs(&self, _config: &super::TranscriptionConfig) -> Option<u32> {
            Some(super::LOCAL_CHUNK_SECS)
        }
    }
}
pub mod factory;
pub mod openai_compatible_realtime;
pub mod openai_realtime;
pub mod openai_realtime_protocol;
pub mod openai_rest;
pub mod phrase_split;
pub mod verbose_json;

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::audio::AudioChunk;

pub use crate::meeting::{Segment, Word};

/// A timed transcript of one audio chunk, as returned by
/// [`TranscriptionBackend::transcribe_segments`]. Times are milliseconds
/// relative to the start of the chunk.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Transcript {
    pub segments: Vec<Segment>,
}

/// Chunk length for the 25 MB upload cap of Groq and OpenAI: 16 kHz mono i16
/// WAV is 32 KB/s, so 10 minutes is ~19 MB with headroom.
pub const UPLOAD_CHUNK_SECS: u32 = 600;

/// Chunk length for local decoding. whisper.cpp still decodes phrase by
/// phrase inside a chunk; the cap only bounds how much audio sits in memory.
pub const LOCAL_CHUNK_SECS: u32 = 600;

/// Sample rate of the WAV handed to [`TranscriptionBackend::transcribe_segments`].
const SEGMENT_SAMPLE_RATE: usize = 16_000;

/// Silence that ends a phrase in [`segments_by_phrase`]. Matches the
/// local-whisper default so both phrase paths cut alike.
const FALLBACK_PHRASE_SILENCE_MS: u64 = 400;

/// Convert seconds (as reported by every cloud API) to whole milliseconds.
pub(crate) fn secs_to_ms(secs: f64) -> u64 {
    if secs.is_finite() && secs > 0.0 {
        (secs * 1000.0).round() as u64
    } else {
        0
    }
}

/// Convert a sample index at [`SEGMENT_SAMPLE_RATE`] to milliseconds.
pub(crate) fn samples_to_ms(samples: u64) -> u64 {
    samples * 1000 / SEGMENT_SAMPLE_RATE as u64
}

/// Decode a 16-bit PCM WAV buffer into samples.
pub(crate) fn decode_wav_samples(audio: &[u8]) -> anyhow::Result<Vec<i16>> {
    let reader = hound::WavReader::new(std::io::Cursor::new(audio))?;
    Ok(reader.into_samples::<i16>().collect::<Result<_, _>>()?)
}

/// Phrase fallback for backends that return no timestamps.
///
/// Splits the chunk at pauses with [`phrase_split::PhraseSplitter`], sends each
/// phrase through the backend's plain [`TranscriptionBackend::transcribe`], and
/// returns one segment per non-empty phrase, timed by where the phrase sits in
/// the chunk. Not a trait default on purpose: each backend that relies on it
/// calls it explicitly from its own `transcribe_segments`.
pub async fn segments_by_phrase<B>(
    backend: &B,
    audio: &[u8],
    config: &TranscriptionConfig,
) -> anyhow::Result<Transcript>
where
    B: TranscriptionBackend + ?Sized,
{
    use crate::audio::wav::encode_wav;

    let samples = decode_wav_samples(audio)?;
    let mut splitter = phrase_split::PhraseSplitter::new(
        SEGMENT_SAMPLE_RATE,
        audio_silence_gate::SILENCE_RMS_THRESHOLD,
        FALLBACK_PHRASE_SILENCE_MS,
    );
    let mut phrases = splitter.feed(&samples);
    phrases.extend(splitter.flush());

    let mut segments = Vec::new();
    for phrase in phrases {
        let wav = encode_wav(&phrase.samples)?;
        let text = backend.transcribe(&wav, config).await?;
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        segments.push(Segment {
            start_ms: samples_to_ms(phrase.start_sample),
            end_ms: samples_to_ms(phrase.start_sample + phrase.samples.len() as u64),
            text: text.to_string(),
            speaker: None,
            words: None,
        });
    }
    Ok(Transcript { segments })
}

/// Joins `prompt` and `vocabulary` with `". "`, prompt first. Blank prompts
/// are treated as absent; returns `None` only when both inputs are empty so
/// backends skip the hint entirely rather than receiving an empty string.
pub fn transcription_prompt(prompt: Option<&str>, vocabulary: &[String]) -> Option<String> {
    let prompt = prompt.map(str::trim).filter(|s| !s.is_empty());
    let vocab = if vocabulary.is_empty() {
        None
    } else {
        Some(vocabulary.join(", "))
    };
    match (prompt, vocab) {
        (Some(p), Some(v)) => Some(format!("{p}. {v}")),
        (Some(p), None) => Some(p.to_string()),
        (None, Some(v)) => Some(v),
        (None, None) => None,
    }
}

/// Configuration for a transcription request.
#[derive(Debug, Clone)]
pub struct TranscriptionConfig {
    /// Language code (ISO 639-1), e.g. "en", or "auto" for auto-detection.
    pub language: String,
    /// Model identifier (backend-specific).
    pub model: String,
    /// Optional prompt hint for the transcription model (vocabulary, context).
    pub prompt: Option<String>,
    /// Backend-specific keyword-biasing terms — Deepgram keyterm prompting.
    ///
    /// Carried alongside `prompt` rather than on the backend struct so both
    /// forms of "bias the model toward these words" travel the same channel:
    /// a path that deliberately sends no prompt (command mode, where the
    /// recording is a spoken instruction rather than content) sends no
    /// keyterms either. Backends with no equivalent parameter ignore these —
    /// they receive the same terms folded into `prompt`.
    pub keyterms: Vec<String>,
}

/// Trait for transcription backends.
///
/// Each backend takes WAV-encoded audio bytes and returns the transcribed text.
/// Backends that support streaming override `transcribe_stream`.
#[async_trait]
pub trait TranscriptionBackend: Send + Sync {
    /// Transcribe a complete WAV-encoded audio buffer, returning the text.
    async fn transcribe(
        &self,
        audio: &[u8],
        config: &TranscriptionConfig,
    ) -> anyhow::Result<String>;

    /// Streaming transcription: receive audio chunks and send text incrementally.
    ///
    /// The default implementation collects all audio, encodes to WAV, and calls
    /// `transcribe()` — so non-streaming backends work without overriding this.
    async fn transcribe_stream(
        &self,
        mut audio_rx: mpsc::Receiver<AudioChunk>,
        text_tx: mpsc::Sender<String>,
        config: &TranscriptionConfig,
    ) -> anyhow::Result<()> {
        use crate::audio::wav::encode_wav;

        // Collect all audio chunks.
        let mut all_samples: Vec<i16> = Vec::new();
        while let Some(chunk) = audio_rx.recv().await {
            all_samples.extend_from_slice(&chunk);
        }

        if all_samples.is_empty() {
            return Ok(());
        }

        // Encode to WAV and use the non-streaming method.
        let wav_data = encode_wav(&all_samples)?;
        let text = self.transcribe(&wav_data, config).await?;

        if !text.is_empty() {
            text_tx.send(text).await.ok();
        }

        Ok(())
    }

    /// Whether this backend supports true/chunked streaming.
    ///
    /// When true, the daemon will use `transcribe_stream` during recording
    /// rather than waiting for recording to finish.
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Whether this backend actually transmits [`TranscriptionConfig::prompt`]
    /// to the provider for *this* request.
    ///
    /// Gates the prompt-echo filter in the batch pipeline (issue #133).
    /// `config.prompt` is built for *every* backend from `[general] prompt` +
    /// `[general] vocabulary`, so a backend that silently drops it would
    /// otherwise have genuine speech that happens to resemble the user's own
    /// vocabulary discarded as an "echo" of a prompt that never went on the
    /// wire.
    ///
    /// Required rather than defaulted on purpose: a default of `true` is
    /// exactly what let `openai-compatible-realtime` inherit the wrong answer
    /// in the first cut of the #133 fix. Answer it from the request-building
    /// code, not from what the backend "should" do.
    ///
    /// Takes the request because for the realtime profiles the answer is not a
    /// property of the backend at all: the model is. `openai-realtime` asks
    /// `openai_model_supports_prompt(config.model)`, and
    /// `OpenAiSessionUpdate::new` drops the prompt for gpt-realtime-whisper,
    /// so one backend struct sends it for one model and not for another.
    /// Backends whose answer is fixed ignore the argument.
    fn sends_prompt(&self, config: &TranscriptionConfig) -> bool;

    /// Transcribe one chunk of a meeting track into timed segments.
    ///
    /// `audio` is a 16 kHz mono i16 WAV of at most
    /// [`Self::max_segment_chunk_secs`]; segment times are relative to its
    /// start. Backends with native timestamps parse them; the rest call
    /// [`segments_by_phrase`] explicitly. Live-only backends return an error
    /// naming the batch backends to use instead.
    ///
    /// Required, like `sends_prompt`: a default would let a new backend ship
    /// meeting support nobody decided on.
    async fn transcribe_segments(
        &self,
        audio: &[u8],
        config: &TranscriptionConfig,
    ) -> anyhow::Result<Transcript>;

    /// Longest chunk, in seconds, [`Self::transcribe_segments`] accepts, or
    /// `None` to send a whole track at once.
    fn max_segment_chunk_secs(&self, config: &TranscriptionConfig) -> Option<u32>;
}

/// Error for a backend that only transcribes live audio.
pub(crate) fn live_only_error(backend: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "the {backend} backend only transcribes live audio and cannot transcribe a recorded \
         meeting. Set [meeting] backend to a batch backend: groq, openai, deepgram, \
         local-whisper or asr-sidecar"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Canned `transcribe` that reports what it was sent, for the fallback.
    struct EchoLength;

    #[async_trait]
    impl TranscriptionBackend for EchoLength {
        async fn transcribe(
            &self,
            audio: &[u8],
            _config: &TranscriptionConfig,
        ) -> anyhow::Result<String> {
            let samples = decode_wav_samples(audio)?;
            Ok(format!("{} samples", samples.len()))
        }
        fn sends_prompt(&self, _config: &TranscriptionConfig) -> bool {
            false
        }
        async fn transcribe_segments(
            &self,
            audio: &[u8],
            config: &TranscriptionConfig,
        ) -> anyhow::Result<Transcript> {
            segments_by_phrase(self, audio, config).await
        }
        fn max_segment_chunk_secs(&self, _config: &TranscriptionConfig) -> Option<u32> {
            None
        }
    }

    fn cfg() -> TranscriptionConfig {
        TranscriptionConfig {
            language: "en".into(),
            model: "m".into(),
            prompt: None,
            keyterms: Vec::new(),
        }
    }

    /// 1 s of a 440 Hz tone at moderate level.
    fn tone(secs: f64) -> Vec<i16> {
        let n = (secs * SEGMENT_SAMPLE_RATE as f64) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / SEGMENT_SAMPLE_RATE as f64;
                ((t * 440.0 * std::f64::consts::TAU).sin() * 6000.0) as i16
            })
            .collect()
    }

    #[tokio::test]
    async fn phrase_fallback_times_segments_by_phrase_position() {
        let mut audio = vec![0i16; 32_000]; // 2 s silence
        audio.extend(tone(1.0));
        audio.extend(vec![0i16; 32_000]);
        audio.extend(tone(1.5));
        audio.extend(vec![0i16; 16_000]);
        let wav = crate::audio::wav::encode_wav(&audio).unwrap();

        let t = EchoLength.transcribe_segments(&wav, &cfg()).await.unwrap();
        assert_eq!(t.segments.len(), 2, "{t:?}");
        // Onset minus one 100 ms pad frame.
        assert_eq!(t.segments[0].start_ms, 1900);
        assert_eq!(t.segments[1].start_ms, 4900);
        assert!(t.segments[0].end_ms > 3000 && t.segments[0].end_ms <= 3200);
        assert!(t.segments[1].end_ms > 6500 && t.segments[1].end_ms <= 6700);
        assert!(t.segments.iter().all(|s| s.text.ends_with("samples")));
    }

    #[tokio::test]
    async fn phrase_fallback_on_silence_is_empty() {
        let wav = crate::audio::wav::encode_wav(&vec![0i16; 48_000]).unwrap();
        let t = EchoLength.transcribe_segments(&wav, &cfg()).await.unwrap();
        assert!(t.segments.is_empty());
    }

    #[test]
    fn transcription_prompt_joins_prompt_and_vocabulary() {
        let vocab = vec!["Hyprland".to_string(), "whisrs".to_string()];
        assert_eq!(
            transcription_prompt(Some(" Notes. "), &vocab).as_deref(),
            Some("Notes.. Hyprland, whisrs")
        );
        assert_eq!(transcription_prompt(Some("  "), &[]), None);
        assert_eq!(
            transcription_prompt(None, &vocab).as_deref(),
            Some("Hyprland, whisrs")
        );
    }

    #[test]
    fn secs_to_ms_rounds_and_clamps() {
        assert_eq!(secs_to_ms(1.2345), 1235);
        assert_eq!(secs_to_ms(-1.0), 0);
        assert_eq!(secs_to_ms(f64::NAN), 0);
    }
}
