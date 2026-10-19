//! The recording feature: record mic + system audio to disk, transcribe
//! afterwards.
//!
//! Each recording lives in its own directory under the recordings root (see
//! [`store`]): `recording.json` holds [`RecordingMeta`], `audio/` holds one WAV
//! per track, and `<id>.md` is the rendered transcript.

pub mod echo;
pub mod process;
pub mod recorder;
pub mod render;
pub mod speech;
pub mod split;
pub mod store;
pub mod wav_repair;

use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::ipc::validate_language_override;

/// Label of the system track in a transcript when the recording names none.
pub const DEFAULT_OTHERS_LABEL: &str = "Others";

/// Longest label of the system track (`--others`), in characters.
pub const OTHERS_LABEL_MAX: usize = 64;

/// Validate and normalize the label of the system track (`--others`):
/// trimmed, non-empty, one line, at most [`OTHERS_LABEL_MAX`] characters.
pub fn validate_others_label(label: &str) -> Result<String, String> {
    let trimmed = label.trim();
    if trimmed.is_empty() {
        return Err("the others label is empty: name the system track, e.g. 'Video'".to_string());
    }
    if trimmed.chars().any(char::is_control) {
        return Err("the others label must be a single line of text".to_string());
    }
    let len = trimmed.chars().count();
    if len > OTHERS_LABEL_MAX {
        return Err(format!(
            "the others label is {len} characters long: use at most {OTHERS_LABEL_MAX}"
        ));
    }
    Ok(trimmed.to_string())
}

/// Per-recording settings a `recording` command may carry, all optional.
/// The languages apply to `start`, a `toggle` that starts, and `retry`;
/// `others` to `start` and `toggle`. What is left out keeps the config
/// default, or for a retry what the recording stored.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingOptions {
    /// Language of both tracks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Language of the mic track, over `language`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mic_language: Option<String>,
    /// Language of the system track, over `language`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_language: Option<String>,
    /// Label of the system track in the transcript.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub others: Option<String>,
}

impl RecordingOptions {
    /// Every given value checked and normalized: the languages by the `-l`
    /// rule ([`validate_language_override`]), `others` by
    /// [`validate_others_label`]. The daemon runs this on every request: the
    /// socket is open to any client, not only the CLI that checks first.
    pub fn validated(self) -> Result<Self, String> {
        let language =
            |value: Option<String>| value.map(|v| validate_language_override(&v)).transpose();
        Ok(Self {
            language: language(self.language)?,
            mic_language: language(self.mic_language)?,
            system_language: language(self.system_language)?,
            others: self.others.map(|o| validate_others_label(&o)).transpose()?,
        })
    }

    /// The language these options give track `kind`: its own, else
    /// `language`, else `None`.
    pub fn language_for(&self, kind: TrackKind) -> Option<&str> {
        let own = match kind {
            TrackKind::Mic => &self.mic_language,
            TrackKind::System => &self.system_language,
        };
        own.as_deref().or(self.language.as_deref())
    }
}

/// What a `recording` IPC command asks the daemon to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RecordingAction {
    Start,
    Stop,
    Cancel,
    Status,
    Toggle,
    Retry,
    Copy,
}

/// Daemon-level recording phase, reported over IPC and to the tray.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RecordingPhase {
    #[default]
    Idle,
    Recording,
    Processing,
}

/// Snapshot of the daemon's recording activity.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RecordingStatus {
    pub phase: RecordingPhase,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<DateTime<Local>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dir: Option<PathBuf>,
    /// Ids of recordings waiting for processing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub queued: Vec<String>,
}

/// Lifecycle of a single recording, stored as `status` in `recording.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RecordingState {
    Recording,
    Processing,
    Done,
    Failed,
}

/// Which audio source a track was captured from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrackKind {
    Mic,
    System,
}

impl TrackKind {
    /// File stem used for this track under `audio/`.
    pub fn file_stem(self) -> &'static str {
        match self {
            TrackKind::Mic => "mic",
            TrackKind::System => "system",
        }
    }
}

/// Why a recording ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndReason {
    Stopped,
    MaxDuration,
    Shutdown,
    Recovered,
}

/// One recorded audio track.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackMeta {
    pub kind: TrackKind,
    /// Path relative to the recording directory, e.g. `audio/mic.wav`.
    pub file: String,
    pub device: String,
    /// Time of the first captured chunk relative to `started_at`.
    pub offset_ms: i64,
    pub samples: u64,
}

/// A timed piece of transcribed text from a backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<Word>>,
}

/// A single timed word, when the backend reports word timings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Word {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
}

/// A speaker in the transcript.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Speaker {
    /// `"me"` or `"others"`.
    pub id: String,
    pub name: String,
    /// `"self"` or `"channel"`.
    pub source: String,
}

/// One line of the final transcript, on the recording timeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub start_ms: u64,
    pub end_ms: u64,
    /// A [`Speaker::id`].
    pub speaker: String,
    pub channel: TrackKind,
    pub text: String,
}

/// Contents of `recording.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordingMeta {
    pub version: u32,
    pub id: String,
    #[serde(default)]
    pub title: Option<String>,
    pub status: RecordingState,
    pub started_at: DateTime<Local>,
    #[serde(default)]
    pub ended_at: Option<DateTime<Local>>,
    #[serde(default)]
    pub end_reason: Option<EndReason>,
    pub sample_rate: u32,
    #[serde(default)]
    pub tracks: Vec<TrackMeta>,
    /// Language the mic track is transcribed in, set at start (a retry may
    /// change it). `None` in a recording made before this was stored, until
    /// processing fills it from the config.
    #[serde(default)]
    pub mic_language: Option<String>,
    /// Language the system track is transcribed in, like `mic_language`.
    #[serde(default)]
    pub system_language: Option<String>,
    /// Label of the system track in the transcript, set at start. `None`
    /// (a recording made before this was stored) is [`DEFAULT_OTHERS_LABEL`].
    #[serde(default)]
    pub others_label: Option<String>,
    /// Transcription backend, set by processing.
    #[serde(default)]
    pub backend: Option<String>,
    /// Model the backend transcribed with, set by processing: the request's
    /// model, or the model file for local whisper. Both tracks use the same
    /// one; only their language differs.
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub speakers: Vec<Speaker>,
    #[serde(default)]
    pub segments: Vec<TranscriptSegment>,
    /// Non-fatal notes, e.g. "system audio unavailable".
    #[serde(default)]
    pub errors: Vec<String>,
    /// Fatal processing error when `status` is `Failed`.
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub audio_deleted: bool,
    /// Processing attempts started since the last manual retry. Written
    /// before each attempt, so a recording whose processing keeps crashing the
    /// daemon is not requeued on every start forever.
    #[serde(default)]
    pub attempts: u32,
}

/// Current `recording.json` schema version.
pub const META_VERSION: u32 = 1;

/// Sample rate of recording tracks.
pub const SAMPLE_RATE: u32 = 16_000;

impl RecordingMeta {
    /// A fresh meta for a recording that just started.
    pub fn new(id: String, title: Option<String>, started_at: DateTime<Local>) -> Self {
        Self {
            version: META_VERSION,
            id,
            title,
            status: RecordingState::Recording,
            started_at,
            ended_at: None,
            end_reason: None,
            sample_rate: SAMPLE_RATE,
            tracks: Vec::new(),
            mic_language: None,
            system_language: None,
            others_label: None,
            backend: None,
            model: None,
            speakers: Vec::new(),
            segments: Vec::new(),
            errors: Vec::new(),
            error: None,
            audio_deleted: false,
            attempts: 0,
        }
    }

    /// The stored language of track `kind`.
    pub fn language(&self, kind: TrackKind) -> Option<&str> {
        match kind {
            TrackKind::Mic => self.mic_language.as_deref(),
            TrackKind::System => self.system_language.as_deref(),
        }
    }

    /// Replace the stored language of each track `options` gives one
    /// ([`RecordingOptions::language_for`]); the others keep theirs.
    pub fn apply_languages(&mut self, options: &RecordingOptions) {
        if let Some(language) = options.language_for(TrackKind::Mic) {
            self.mic_language = Some(language.to_string());
        }
        if let Some(language) = options.language_for(TrackKind::System) {
            self.system_language = Some(language.to_string());
        }
    }

    /// The label of the system track in the transcript.
    pub fn others_label(&self) -> &str {
        self.others_label.as_deref().unwrap_or(DEFAULT_OTHERS_LABEL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enums_use_contract_wire_names() {
        assert_eq!(
            serde_json::to_string(&EndReason::MaxDuration).unwrap(),
            r#""max_duration""#
        );
        assert_eq!(
            serde_json::to_string(&RecordingState::Done).unwrap(),
            r#""done""#
        );
        assert_eq!(
            serde_json::to_string(&TrackKind::System).unwrap(),
            r#""system""#
        );
        assert_eq!(RecordingStatus::default().phase, RecordingPhase::Idle);
    }

    #[test]
    fn meta_json_roundtrip() {
        let mut meta = RecordingMeta::new("2026-10-02_1400".into(), None, Local::now());
        meta.tracks.push(TrackMeta {
            kind: TrackKind::Mic,
            file: "audio/mic.wav".into(),
            device: "default".into(),
            offset_ms: 12,
            samples: 16_000,
        });
        meta.end_reason = Some(EndReason::Stopped);
        meta.mic_language = Some("de".into());
        meta.system_language = Some("auto".into());
        meta.others_label = Some("Video".into());
        meta.model = Some("ggml-large-v3-turbo.bin".into());
        let json = serde_json::to_string(&meta).unwrap();
        let back: RecordingMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(back, meta);
    }

    /// A `recording.json` written before the languages, the others label
    /// and the model were stored still loads, with all four unset.
    #[test]
    fn meta_without_languages_still_loads() {
        let old = r#"{"version":1,"id":"2026-10-02_1400","title":null,
            "status":"done","started_at":"2026-10-02T14:00:00+02:00",
            "sample_rate":16000,"tracks":[],"backend":"groq"}"#;
        let meta: RecordingMeta = serde_json::from_str(old).unwrap();
        assert_eq!(meta.backend.as_deref(), Some("groq"));
        assert_eq!(meta.language(TrackKind::Mic), None);
        assert_eq!(meta.language(TrackKind::System), None);
        assert_eq!(meta.model, None);
        assert_eq!(meta.others_label(), DEFAULT_OTHERS_LABEL);
    }

    #[test]
    fn options_are_validated_and_normalized() {
        let options = RecordingOptions {
            language: Some(" DE ".into()),
            mic_language: Some("en_us".into()),
            system_language: Some("Auto".into()),
            others: Some("  Video  ".into()),
        }
        .validated()
        .unwrap();
        assert_eq!(
            options,
            RecordingOptions {
                language: Some("de".into()),
                mic_language: Some("en-US".into()),
                system_language: Some("auto".into()),
                others: Some("Video".into()),
            }
        );
        assert_eq!(
            RecordingOptions::default().validated(),
            Ok(RecordingOptions::default())
        );

        for bad in [
            RecordingOptions {
                language: Some("english".into()),
                ..Default::default()
            },
            RecordingOptions {
                mic_language: Some("".into()),
                ..Default::default()
            },
            RecordingOptions {
                system_language: Some("1".into()),
                ..Default::default()
            },
        ] {
            let err = bad.clone().validated().unwrap_err();
            assert!(err.starts_with("invalid language"), "{bad:?}: {err}");
        }
        for (others, want) in [
            ("   ", "empty"),
            ("Video\nme", "single line"),
            (&"x".repeat(OTHERS_LABEL_MAX + 1), "at most"),
        ] {
            let err = RecordingOptions {
                others: Some(others.to_string()),
                ..Default::default()
            }
            .validated()
            .unwrap_err();
            assert!(err.contains(want), "{others:?}: {err}");
        }
        let longest = "é".repeat(OTHERS_LABEL_MAX);
        assert_eq!(validate_others_label(&longest), Ok(longest.clone()));
    }

    /// A track's own language wins over `language`, which covers the
    /// other track; tracks the options leave out keep what is stored.
    #[test]
    fn track_languages_from_options() {
        let both = RecordingOptions {
            language: Some("de".into()),
            ..Default::default()
        };
        let split = RecordingOptions {
            language: Some("de".into()),
            system_language: Some("auto".into()),
            ..Default::default()
        };
        assert_eq!(both.language_for(TrackKind::Mic), Some("de"));
        assert_eq!(both.language_for(TrackKind::System), Some("de"));
        assert_eq!(split.language_for(TrackKind::Mic), Some("de"));
        assert_eq!(split.language_for(TrackKind::System), Some("auto"));
        assert_eq!(
            RecordingOptions::default().language_for(TrackKind::Mic),
            None
        );

        let mut meta = RecordingMeta::new("m".into(), None, Local::now());
        meta.mic_language = Some("en".into());
        meta.system_language = Some("en".into());
        meta.apply_languages(&RecordingOptions {
            system_language: Some("fr".into()),
            ..Default::default()
        });
        assert_eq!(meta.language(TrackKind::Mic), Some("en"));
        assert_eq!(meta.language(TrackKind::System), Some("fr"));
        meta.apply_languages(&RecordingOptions::default());
        assert_eq!(meta.language(TrackKind::System), Some("fr"));
        meta.apply_languages(&both);
        assert_eq!(meta.language(TrackKind::Mic), Some("de"));
        assert_eq!(meta.language(TrackKind::System), Some("de"));
    }
}
