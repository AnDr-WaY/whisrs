//! Rendering a processed meeting as Markdown or plain text.
//!
//! Markdown is the `<id>.md` written next to `meeting.json`: flat YAML front
//! matter (readable by Obsidian and other note tools) followed by the
//! transcript as speaker turns. Plain text is what `whisrs meeting copy`
//! puts on the clipboard.

use super::MeetingMeta;

/// Consecutive segments from one speaker, merged.
struct Turn<'a> {
    start_ms: u64,
    speaker: &'a str,
    text: String,
}

/// Merge consecutive same-speaker segments into turns.
fn turns(meta: &MeetingMeta) -> Vec<Turn<'_>> {
    let mut out: Vec<Turn<'_>> = Vec::new();
    for seg in &meta.segments {
        let text = seg.text.trim();
        if text.is_empty() {
            continue;
        }
        match out.last_mut() {
            Some(turn) if turn.speaker == seg.speaker => {
                turn.text.push(' ');
                turn.text.push_str(text);
            }
            _ => out.push(Turn {
                start_ms: seg.start_ms,
                speaker: &seg.speaker,
                text: text.to_string(),
            }),
        }
    }
    out
}

/// Display name for a speaker id, falling back to the id itself.
fn speaker_name<'a>(meta: &'a MeetingMeta, id: &'a str) -> &'a str {
    meta.speakers
        .iter()
        .find(|s| s.id == id)
        .map_or(id, |s| s.name.as_str())
}

/// `HH:MM:SS` for a millisecond offset.
fn clock(ms: u64) -> String {
    let secs = ms / 1000;
    format!(
        "{:02}:{:02}:{:02}",
        secs / 3600,
        (secs / 60) % 60,
        secs % 60
    )
}

/// A double-quoted YAML scalar. Double quotes are the one YAML style where
/// every character can be expressed, so titles with `:`, `#`, quotes or
/// newlines cannot break the front matter.
fn yaml_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04X}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The meeting title, or a dated default.
fn title(meta: &MeetingMeta) -> String {
    meta.title
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| format!("Meeting {}", meta.started_at.format("%Y-%m-%d %H:%M")))
}

/// Recording length: `ended_at - started_at`, else the longest track.
fn duration_ms(meta: &MeetingMeta) -> u64 {
    if let Some(ended) = meta.ended_at {
        let ms = (ended - meta.started_at).num_milliseconds();
        if ms > 0 {
            return ms as u64;
        }
    }
    let rate = u64::from(meta.sample_rate.max(1));
    meta.tracks
        .iter()
        .map(|t| t.offset_ms.max(0) as u64 + t.samples * 1000 / rate)
        .max()
        .unwrap_or(0)
}

/// Render the meeting as Markdown with YAML front matter.
pub fn render_markdown(meta: &MeetingMeta) -> String {
    let mut out = String::from("---\n");
    out.push_str(&format!("title: {}\n", yaml_str(&title(meta))));
    out.push_str(&format!(
        "date: {}\n",
        yaml_str(&meta.started_at.format("%Y-%m-%d").to_string())
    ));
    out.push_str(&format!(
        "start: {}\n",
        yaml_str(&meta.started_at.format("%H:%M").to_string())
    ));
    out.push_str(&format!(
        "duration: {}\n",
        yaml_str(&clock(duration_ms(meta)))
    ));
    if meta.speakers.is_empty() {
        out.push_str("speakers: []\n");
    } else {
        out.push_str("speakers:\n");
        for speaker in &meta.speakers {
            out.push_str(&format!("  - {}\n", yaml_str(&speaker.name)));
        }
    }
    out.push_str("tags: [meeting]\n");
    out.push_str(&format!("whisrs_id: {}\n", yaml_str(&meta.id)));
    if let Some(backend) = &meta.backend {
        out.push_str(&format!("backend: {}\n", yaml_str(backend)));
    }
    out.push_str("---\n\n## Transcript\n\n");

    let turns = turns(meta);
    if turns.is_empty() {
        out.push_str("_No speech was transcribed._\n");
    }
    for turn in turns {
        out.push_str(&format!(
            "**[{}] {}:** {}\n\n",
            clock(turn.start_ms),
            speaker_name(meta, turn.speaker),
            turn.text
        ));
    }
    if out.ends_with("\n\n") {
        out.pop();
    }
    out
}

/// Render the transcript as plain `[HH:MM:SS] Name: text` lines.
pub fn render_text(meta: &MeetingMeta) -> String {
    turns(meta)
        .into_iter()
        .map(|turn| {
            format!(
                "[{}] {}: {}\n",
                clock(turn.start_ms),
                speaker_name(meta, turn.speaker),
                turn.text
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meeting::{Speaker, TrackKind, TranscriptSegment};
    use chrono::{Local, TimeZone};

    fn seg(start_ms: u64, speaker: &str, text: &str) -> TranscriptSegment {
        TranscriptSegment {
            start_ms,
            end_ms: start_ms + 1000,
            speaker: speaker.to_string(),
            channel: if speaker == "me" {
                TrackKind::Mic
            } else {
                TrackKind::System
            },
            text: text.to_string(),
        }
    }

    fn meta() -> MeetingMeta {
        let started = Local.with_ymd_and_hms(2026, 10, 2, 14, 5, 0).unwrap();
        let mut meta = MeetingMeta::new(
            "2026-10-02_1405_standup".into(),
            Some("Standup: \"Q4\" #plan".into()),
            started,
        );
        meta.ended_at = Some(started + chrono::Duration::seconds(3725));
        meta.backend = Some("groq".into());
        meta.speakers = vec![
            Speaker {
                id: "me".into(),
                name: "Yosif".into(),
                source: "self".into(),
            },
            Speaker {
                id: "others".into(),
                name: "Others".into(),
                source: "channel".into(),
            },
        ];
        meta.segments = vec![
            seg(1_000, "me", "Hi all."),
            seg(3_000, "me", "Quick update."),
            seg(65_000, "others", "Sounds good."),
            seg(3_700_000, "me", "Bye."),
        ];
        meta
    }

    #[test]
    fn markdown_has_front_matter_and_merged_turns() {
        let md = render_markdown(&meta());
        let expected = "---\n\
            title: \"Standup: \\\"Q4\\\" #plan\"\n\
            date: \"2026-10-02\"\n\
            start: \"14:05\"\n\
            duration: \"01:02:05\"\n\
            speakers:\n  - \"Yosif\"\n  - \"Others\"\n\
            tags: [meeting]\n\
            whisrs_id: \"2026-10-02_1405_standup\"\n\
            backend: \"groq\"\n\
            ---\n\n## Transcript\n\n\
            **[00:00:01] Yosif:** Hi all. Quick update.\n\n\
            **[00:01:05] Others:** Sounds good.\n\n\
            **[01:01:40] Yosif:** Bye.\n";
        assert_eq!(md, expected);
    }

    #[test]
    fn text_is_one_line_per_turn() {
        assert_eq!(
            render_text(&meta()),
            "[00:00:01] Yosif: Hi all. Quick update.\n\
             [00:01:05] Others: Sounds good.\n\
             [01:01:40] Yosif: Bye.\n"
        );
    }

    #[test]
    fn yaml_strings_escape_everything_risky() {
        assert_eq!(yaml_str("a\"b\\c\nd"), r#""a\"b\\c\nd""#);
        assert_eq!(yaml_str("x\u{7}"), r#""x\u0007""#);
        assert_eq!(yaml_str("key: value # c"), r#""key: value # c""#);
    }

    #[test]
    fn untitled_empty_meeting_still_renders() {
        let mut m = meta();
        m.title = None;
        m.segments.clear();
        m.backend = None;
        let md = render_markdown(&m);
        assert!(md.contains("title: \"Meeting 2026-10-02 14:05\"\n"));
        assert!(md.contains("_No speech was transcribed._"));
        assert!(!md.contains("backend:"));
        assert_eq!(render_text(&m), "");
    }

    #[test]
    fn duration_falls_back_to_tracks() {
        let mut m = meta();
        m.ended_at = None;
        m.tracks.push(crate::meeting::TrackMeta {
            kind: TrackKind::Mic,
            file: "audio/mic.wav".into(),
            device: "default".into(),
            offset_ms: 500,
            samples: 16_000 * 90,
        });
        assert_eq!(duration_ms(&m), 90_500);
    }
}
