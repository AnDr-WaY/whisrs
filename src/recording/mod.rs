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
    /// Transcription backend, set by processing.
    #[serde(default)]
    pub backend: Option<String>,
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
            backend: None,
            speakers: Vec::new(),
            segments: Vec::new(),
            errors: Vec::new(),
            error: None,
            audio_deleted: false,
            attempts: 0,
        }
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
        let json = serde_json::to_string(&meta).unwrap();
        let back: RecordingMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(back, meta);
    }
}
