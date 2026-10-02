//! Whisper `verbose_json` responses with segment timestamps.
//!
//! Groq and OpenAI `whisper-1` answer `response_format=verbose_json` with the
//! same shape: `segments[]` carrying start/end seconds plus Whisper's own
//! confidence numbers, and, when word granularity is requested, a top-level
//! `words[]`. Both meeting paths parse it here.

use serde::Deserialize;

use super::{secs_to_ms, Segment, Transcript, Word};

/// A segment is dropped as non-speech when Whisper is fairly sure there was
/// no speech *and* the decoded tokens were improbable. Same rule (and
/// thresholds) as Whisper's own reference decoder uses to skip silent windows.
pub const NO_SPEECH_PROB_THRESHOLD: f64 = 0.6;
/// See [`NO_SPEECH_PROB_THRESHOLD`].
pub const LOW_AVG_LOGPROB_THRESHOLD: f64 = -1.0;

/// Whisper's no-speech heuristic: both conditions must hold, so a quiet but
/// confidently decoded segment survives.
pub fn is_no_speech(no_speech_prob: Option<f64>, avg_logprob: Option<f64>) -> bool {
    match (no_speech_prob, avg_logprob) {
        (Some(p), Some(lp)) => p > NO_SPEECH_PROB_THRESHOLD && lp < LOW_AVG_LOGPROB_THRESHOLD,
        _ => false,
    }
}

#[derive(Debug, Deserialize)]
struct VerboseResponse {
    #[serde(default)]
    text: String,
    #[serde(default)]
    segments: Option<Vec<VerboseSegment>>,
    #[serde(default)]
    words: Vec<VerboseWord>,
}

#[derive(Debug, Deserialize)]
struct VerboseSegment {
    start: f64,
    end: f64,
    #[serde(default)]
    text: String,
    #[serde(default)]
    no_speech_prob: Option<f64>,
    #[serde(default)]
    avg_logprob: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct VerboseWord {
    word: String,
    start: f64,
    end: f64,
}

/// Parse a `verbose_json` body into timed segments.
///
/// Segments flagged by [`is_no_speech`] and segments with blank text are
/// dropped. Words are attached to the segment containing their midpoint, and
/// a segment with words is narrowed to the span they cover; a segment with no
/// words gets `words: None`. A body with text but no
/// `segments` array (a server that ignored the granularity) becomes a single
/// segment spanning its words, or `0..0` when there are none.
pub fn parse_verbose_segments(body: &str) -> anyhow::Result<Transcript> {
    let parsed: VerboseResponse = serde_json::from_str(body)?;

    let words: Vec<Word> = parsed
        .words
        .iter()
        .map(|w| Word {
            start_ms: secs_to_ms(w.start),
            end_ms: secs_to_ms(w.end),
            text: w.word.trim().to_string(),
            speaker: None,
        })
        .collect();

    let Some(raw_segments) = parsed.segments else {
        let text = parsed.text.trim();
        if text.is_empty() {
            return Ok(Transcript::default());
        }
        let start_ms = words.first().map_or(0, |w| w.start_ms);
        let end_ms = words.last().map_or(start_ms, |w| w.end_ms);
        return Ok(Transcript {
            segments: vec![Segment {
                start_ms,
                end_ms,
                text: text.to_string(),
                speaker: None,
                words: (!words.is_empty()).then_some(words),
            }],
        });
    };

    let mut segments = Vec::with_capacity(raw_segments.len());
    for seg in raw_segments {
        let text = seg.text.trim();
        if text.is_empty() || is_no_speech(seg.no_speech_prob, seg.avg_logprob) {
            continue;
        }
        let start_ms = secs_to_ms(seg.start);
        let end_ms = secs_to_ms(seg.end).max(start_ms);
        let seg_words: Vec<Word> = words
            .iter()
            .filter(|w| {
                let mid = (w.start_ms + w.end_ms) / 2;
                mid >= start_ms && mid < end_ms.max(start_ms + 1)
            })
            .cloned()
            .collect();
        // Whisper stretches a segment back over leading silence (a first
        // segment starts at 0 even when speech begins seconds in), so its
        // words, timed independently, give the tighter span when present.
        let (start_ms, end_ms) = match (seg_words.first(), seg_words.last()) {
            (Some(first), Some(last)) => {
                let start = first.start_ms.clamp(start_ms, end_ms);
                (start, last.end_ms.clamp(start, end_ms))
            }
            _ => (start_ms, end_ms),
        };
        segments.push(Segment {
            start_ms,
            end_ms,
            text: text.to_string(),
            speaker: None,
            words: (!seg_words.is_empty()).then_some(seg_words),
        });
    }
    Ok(Transcript { segments })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trimmed from a real Groq `whisper-large-v3-turbo` response with
    /// `timestamp_granularities[]=segment` and `=word`.
    const GROQ_VERBOSE: &str = r#"{
        "task": "transcribe",
        "language": "English",
        "duration": 7.4,
        "text": " Hello there. How are you doing today?",
        "segments": [
            {"id": 0, "seek": 0, "start": 0.0, "end": 1.52, "text": " Hello there.",
             "tokens": [50365, 2425], "temperature": 0.0, "avg_logprob": -0.21,
             "compression_ratio": 0.8, "no_speech_prob": 0.01},
            {"id": 1, "seek": 0, "start": 2.0, "end": 4.1, "text": " How are you doing today?",
             "tokens": [50465], "temperature": 0.0, "avg_logprob": -0.3,
             "compression_ratio": 0.9, "no_speech_prob": 0.02},
            {"id": 2, "seek": 0, "start": 5.0, "end": 7.4, "text": " Thank you.",
             "tokens": [50615], "temperature": 0.0, "avg_logprob": -1.4,
             "compression_ratio": 0.6, "no_speech_prob": 0.93}
        ],
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.6},
            {"word": "there.", "start": 0.6, "end": 1.5},
            {"word": "How", "start": 2.0, "end": 2.3},
            {"word": "are", "start": 2.3, "end": 2.5},
            {"word": "you", "start": 2.5, "end": 2.8},
            {"word": "doing", "start": 2.8, "end": 3.3},
            {"word": "today?", "start": 3.3, "end": 4.1}
        ]
    }"#;

    /// OpenAI `whisper-1` verbose_json: same schema, and with segment-only
    /// granularity no `words` key at all.
    const WHISPER1_VERBOSE: &str = r#"{
        "task": "transcribe",
        "language": "english",
        "duration": 3.2,
        "text": "Ship it on Friday.",
        "segments": [
            {"id": 0, "seek": 0, "start": 0.24, "end": 3.2, "text": " Ship it on Friday.",
             "tokens": [1], "temperature": 0.0, "avg_logprob": -0.18,
             "compression_ratio": 0.7, "no_speech_prob": 0.004}
        ]
    }"#;

    #[test]
    fn groq_segments_carry_times_and_words() {
        let t = parse_verbose_segments(GROQ_VERBOSE).unwrap();
        assert_eq!(t.segments.len(), 2, "the no-speech segment must be dropped");
        let first = &t.segments[0];
        assert_eq!(
            (first.start_ms, first.end_ms),
            (0, 1500),
            "narrowed to its words"
        );
        assert_eq!(first.text, "Hello there.");
        let words: Vec<_> = first
            .words
            .as_ref()
            .unwrap()
            .iter()
            .map(|w| w.text.as_str())
            .collect();
        assert_eq!(words, ["Hello", "there."]);
        let second = &t.segments[1];
        assert_eq!((second.start_ms, second.end_ms), (2000, 4100));
        assert_eq!(second.words.as_ref().unwrap().len(), 5);
    }

    #[test]
    fn whisper1_segments_without_words() {
        let t = parse_verbose_segments(WHISPER1_VERBOSE).unwrap();
        assert_eq!(t.segments.len(), 1);
        assert_eq!(t.segments[0].start_ms, 240);
        assert_eq!(t.segments[0].end_ms, 3200);
        assert_eq!(t.segments[0].text, "Ship it on Friday.");
        assert!(t.segments[0].words.is_none());
    }

    #[test]
    fn no_speech_needs_both_conditions() {
        assert!(is_no_speech(Some(0.9), Some(-1.5)));
        // Confident decode of quiet speech survives.
        assert!(!is_no_speech(Some(0.9), Some(-0.2)));
        // Low logprob but clearly speech survives.
        assert!(!is_no_speech(Some(0.1), Some(-1.5)));
        assert!(!is_no_speech(None, Some(-3.0)));
    }

    #[test]
    fn missing_segments_becomes_one_segment() {
        let body = r#"{"text": " hi all ", "words": [
            {"word": "hi", "start": 1.0, "end": 1.2},
            {"word": "all", "start": 1.3, "end": 1.6}]}"#;
        let t = parse_verbose_segments(body).unwrap();
        assert_eq!(t.segments.len(), 1);
        assert_eq!((t.segments[0].start_ms, t.segments[0].end_ms), (1000, 1600));
        assert_eq!(t.segments[0].text, "hi all");
    }

    #[test]
    fn empty_response_is_empty_transcript() {
        assert!(parse_verbose_segments(r#"{"text": ""}"#)
            .unwrap()
            .segments
            .is_empty());
        assert!(parse_verbose_segments(r#"{"text": "", "segments": []}"#)
            .unwrap()
            .segments
            .is_empty());
    }

    #[test]
    fn segment_is_narrowed_to_its_words() {
        // Real Groq shape: speech starts at 7 s, the segment claims 0 s.
        let body = r#"{"text": "Thanks.", "segments": [
            {"start": 0.0, "end": 10.4, "text": " Thanks. Ready to ship.",
             "avg_logprob": -0.2, "no_speech_prob": 0.05}],
            "words": [{"word": "Thanks.", "start": 7.0, "end": 7.6},
                      {"word": "Ready", "start": 8.0, "end": 8.3},
                      {"word": "to", "start": 8.3, "end": 8.4},
                      {"word": "ship.", "start": 8.4, "end": 9.9}]}"#;
        let t = parse_verbose_segments(body).unwrap();
        assert_eq!((t.segments[0].start_ms, t.segments[0].end_ms), (7000, 9900));
    }
}
