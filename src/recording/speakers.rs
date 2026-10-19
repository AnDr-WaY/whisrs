//! Speakers: who said each line of a recording.
//!
//! Without speaker separation each track is one speaker: the mic is the user
//! (`me`) and the system audio is everyone else (`others`, named by the
//! recording's others label). With it ([`plan`]), the voices on the system
//! track, and with `shared_mic` those on the mic, are told apart. Every word
//! takes a voice label from the backend, a segment splits where its words
//! change voice ([`split_segment`]), and once both tracks are transcribed the
//! voices become speakers, numbered by first appearance ([`assign`]).
//!
//! Nothing is guessed: a voice is named only when the track says who it is
//! (a single remote voice keeps the others label, a single mic voice is the
//! user); every other voice is `Speaker N`.

use tracing::warn;

use super::speech::join_words;
use super::{Segment, Speaker, TrackKind, Word};
use crate::transcription::factory::canonical_backend_name;
use crate::{Config, DiarizationMode, RecordingConfig};

/// A voice with less talk time than this ...
pub const MINOR_TALK_MS: u64 = 5_000;

/// ... and less than this share of its track's talk time, in percent, is
/// minor ([`is_minor`]).
pub const MINOR_SHARE_PERCENT: u64 = 2;

/// Id of the user when the mic is one speaker.
pub const ME: &str = "me";

/// Id of the system track when it is one speaker, and of its speech that no
/// voice was found for.
pub const OTHERS: &str = "others";

/// Id of a shared mic's speech that no voice was found for.
pub const MIC: &str = "mic";

/// Name of [`MIC`].
pub const MIC_LABEL: &str = "Mic";

/// [`Speaker::source`] of the user.
pub const SOURCE_SELF: &str = "self";

/// [`Speaker::source`] of a speaker named by its track's label.
pub const SOURCE_CHANNEL: &str = "channel";

/// [`Speaker::source`] of an unnamed speaker, shown as `Speaker N`.
pub const SOURCE_NONE: &str = "none";

/// How the voices of a recording are told apart.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Method {
    /// They are not: one speaker per track.
    Off,
    /// The transcription backend labels each word or segment with a voice.
    /// `label` is what `recording.json` records as `diarization`.
    Backend { label: String },
}

impl Method {
    /// What `recording.json` records as `diarization`.
    pub fn label(&self) -> &str {
        match self {
            Method::Off => "off",
            Method::Backend { label } => label,
        }
    }
}

/// How one recording's speakers are found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeakerPlan {
    pub method: Method,
    /// The mic's voices are told apart too.
    pub shared_mic: bool,
}

impl SpeakerPlan {
    /// Whether the voices of track `kind` are told apart: the system track
    /// whenever speakers are separated, the mic only when it is shared.
    pub fn separates(&self, kind: TrackKind) -> bool {
        self.method != Method::Off && (kind == TrackKind::System || self.shared_mic)
    }
}

/// What backend `name` records as `diarization` when it labels speakers
/// itself, or `None` when it does not: an `asr-sidecar` may return
/// `segments[].speaker`.
pub fn backend_speakers(name: &str, _config: &Config) -> Option<String> {
    match canonical_backend_name(name) {
        "asr-sidecar" => Some("asr-sidecar".to_string()),
        _ => None,
    }
}

/// How a recording transcribed with backend `backend` finds its speakers,
/// from `[recording] diarization`: `backend` when the backend labels
/// speakers ([`backend_speakers`]), `off` when it cannot or when asked to.
/// `auto` is `off` for every backend that does not separate voices by
/// default.
pub fn plan(config: &Config, backend: &str, shared_mic: bool) -> SpeakerPlan {
    let recording = RecordingConfig::effective(config);
    let method = match recording.diarization {
        DiarizationMode::Off | DiarizationMode::Auto => Method::Off,
        DiarizationMode::Backend => match backend_speakers(backend, config) {
            Some(label) => Method::Backend { label },
            None => {
                warn!(
                    "[recording] diarization = \"backend\", but the {backend} backend does not \
                     label speakers; each track stays one speaker"
                );
                Method::Off
            }
        },
    };
    SpeakerPlan { method, shared_mic }
}

/// Where a track's voice labels come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Labels {
    /// Its voices are not told apart: labels are dropped.
    None,
    /// The labels the backend put on words and segments.
    Backend,
}

impl Labels {
    fn of_segment(&self, seg: &Segment) -> Option<String> {
        match self {
            Labels::None => None,
            Labels::Backend => seg.speaker.clone(),
        }
    }

    fn of_word(&self, word: &Word) -> Option<String> {
        match self {
            Labels::None => None,
            Labels::Backend => word.speaker.clone(),
        }
    }
}

/// Every word takes the label of the last labelled word before it, or of
/// the first one when none comes before: a word the diarizer left out
/// stays with its neighbours rather than splitting the line.
fn fill_gaps(labels: &mut [Option<String>]) {
    let Some(mut last) = labels.iter().flatten().next().cloned() else {
        return;
    };
    for label in labels.iter_mut() {
        match label {
            Some(l) => last = l.clone(),
            None => *label = Some(last.clone()),
        }
    }
}

/// `seg` with each part labelled by its voice: split where its words change
/// voice, each part keeping its words. The parts keep the backend's
/// punctuated text when it has one token per word, else they join the
/// words' own text ([`join_words`]). A segment without word timings takes
/// the segment's own label. Labels are dropped with [`Labels::None`].
pub fn split_segment(mut seg: Segment, labels: Labels) -> Vec<Segment> {
    if labels == Labels::None {
        seg.speaker = None;
        for word in seg.words.iter_mut().flatten() {
            word.speaker = None;
        }
        return vec![seg];
    }
    let segment_label = labels.of_segment(&seg);
    let words = match seg.words.take() {
        Some(words) if !words.is_empty() => words,
        _ => {
            seg.speaker = segment_label;
            return vec![seg];
        }
    };
    let mut per_word: Vec<Option<String>> = words.iter().map(|w| labels.of_word(w)).collect();
    fill_gaps(&mut per_word);
    if per_word.iter().all(Option::is_none) {
        per_word.fill(segment_label);
    }
    let words: Vec<Word> = words
        .into_iter()
        .zip(per_word)
        .map(|(word, speaker)| Word { speaker, ..word })
        .collect();
    let groups: Vec<&[Word]> = words.chunk_by(|a, b| a.speaker == b.speaker).collect();
    if groups.len() == 1 {
        seg.speaker = words[0].speaker.clone();
        seg.words = Some(words);
        return vec![seg];
    }
    let tokens: Vec<&str> = seg.text.split_whitespace().collect();
    let keep_tokens = tokens.len() == words.len();
    let mut at = 0;
    groups
        .into_iter()
        .map(|group| {
            let text = if keep_tokens {
                tokens[at..at + group.len()].join(" ")
            } else {
                join_words(&seg.text, group.iter().map(|w| w.text.as_str()))
            };
            at += group.len();
            let start_ms = group[0].start_ms;
            Segment {
                start_ms,
                end_ms: group[group.len() - 1].end_ms.max(start_ms),
                text,
                speaker: group[0].speaker.clone(),
                words: Some(group.to_vec()),
            }
        })
        .collect()
}

/// Whether a voice that talks `talk_ms` of its track's `track_talk_ms` is
/// minor: under [`MINOR_TALK_MS`] and under [`MINOR_SHARE_PERCENT`] of the
/// track. A minor voice keeps its words; it is only never named after a
/// participant.
pub fn is_minor(talk_ms: u64, track_talk_ms: u64) -> bool {
    talk_ms < MINOR_TALK_MS && talk_ms * 100 < track_talk_ms * MINOR_SHARE_PERCENT
}

/// What [`assign`] needs to know about the recording.
#[derive(Debug, Clone, Copy)]
pub struct Context<'a> {
    /// The recording has a mic track ...
    pub has_mic: bool,
    /// ... and a system track. A track that is listed gets its speaker even
    /// when it holds no speech, unless its voices are told apart.
    pub has_system: bool,
    /// [`SpeakerPlan::separates`] for the mic ...
    pub separate_mic: bool,
    /// ... and the system track.
    pub separate_system: bool,
    /// The user's name ([`super::process`]'s `user_name`).
    pub user_name: &'a str,
    /// The label of the system track ([`super::RecordingMeta::others_label`]).
    pub others_label: &'a str,
}

/// One voice found on a track.
struct Voice {
    label: String,
    first_ms: u64,
    talk_ms: u64,
}

fn talk(seg: &Segment) -> u64 {
    seg.end_ms.saturating_sub(seg.start_ms)
}

/// Turn the voice labels of both tracks into speakers, and every segment's
/// `speaker` into its speaker's id. Returns the speakers, the mic's first,
/// each track's in order of first appearance.
///
/// A track whose voices are not told apart, or where no voice was found, is
/// one speaker: `me` (the user) or `others` (the others label). Otherwise
/// its voices are numbered by first appearance: `S1`, `S2`, ... on the
/// system track, `M1`, `M2`, ... on the mic. A track with exactly one voice
/// that is not minor ([`is_minor`]) names it: on the system track it keeps
/// the others label, on the mic it is the user (id `me`), and the track's
/// speech without a voice goes to it. With several, that speech is the
/// track's own speaker (`others`, or `mic` named [`MIC_LABEL`]), and every
/// voice left unnamed is `Speaker N`, numbered by first appearance across
/// both tracks.
pub fn assign(mic: &mut [Segment], system: &mut [Segment], ctx: &Context<'_>) -> Vec<Speaker> {
    let mut found = channel(mic, TrackKind::Mic, ctx);
    found.extend(channel(system, TrackKind::System, ctx));
    let mut unnamed: Vec<usize> = (0..found.len())
        .filter(|&i| found[i].0.source == SOURCE_NONE)
        .collect();
    unnamed.sort_by_key(|&i| (found[i].1, i));
    for (n, i) in unnamed.into_iter().enumerate() {
        found[i].0.name = format!("Speaker {}", n + 1);
    }
    found.into_iter().map(|(speaker, _)| speaker).collect()
}

/// The speakers of one track with the time each first speaks, and its
/// segments relabelled with their ids ([`assign`]).
fn channel(segments: &mut [Segment], kind: TrackKind, ctx: &Context<'_>) -> Vec<(Speaker, u64)> {
    let (present, separate) = match kind {
        TrackKind::Mic => (ctx.has_mic, ctx.separate_mic),
        TrackKind::System => (ctx.has_system, ctx.separate_system),
    };
    let speaker = |id: &str, name: &str, source: &str, talk_ms: u64| Speaker {
        id: id.to_string(),
        name: name.to_string(),
        source: source.to_string(),
        evidence: None,
        talk_ms,
        channel: Some(kind),
        minor: false,
    };
    let (whole_id, whole_name, whole_source) = match kind {
        TrackKind::Mic => (ME, ctx.user_name, SOURCE_SELF),
        TrackKind::System => (OTHERS, ctx.others_label, SOURCE_CHANNEL),
    };
    let track_talk: u64 = segments.iter().map(talk).sum();

    let mut order: Vec<usize> = (0..segments.len()).collect();
    order.sort_by_key(|&i| (segments[i].start_ms, i));
    let mut voices: Vec<Voice> = Vec::new();
    if separate {
        for &i in &order {
            let Some(label) = &segments[i].speaker else {
                continue;
            };
            match voices.iter_mut().find(|v| &v.label == label) {
                Some(voice) => voice.talk_ms += talk(&segments[i]),
                None => voices.push(Voice {
                    label: label.clone(),
                    first_ms: segments[i].start_ms,
                    talk_ms: talk(&segments[i]),
                }),
            }
        }
    }

    if voices.is_empty() {
        for seg in segments.iter_mut() {
            seg.speaker = Some(whole_id.to_string());
        }
        if !present || (separate && segments.is_empty()) {
            return Vec::new();
        }
        let first = order.first().map_or(0, |&i| segments[i].start_ms);
        return vec![(
            speaker(whole_id, whole_name, whole_source, track_talk),
            first,
        )];
    }

    let minor: Vec<bool> = voices
        .iter()
        .map(|v| is_minor(v.talk_ms, track_talk))
        .collect();
    let mut major = (0..voices.len()).filter(|&i| !minor[i]);
    let main = match (major.next(), major.next()) {
        (Some(only), None) => Some(only),
        _ => None,
    };
    let prefix = match kind {
        TrackKind::Mic => "M",
        TrackKind::System => "S",
    };
    let mut n = 0;
    let ids: Vec<String> = (0..voices.len())
        .map(|i| {
            if kind == TrackKind::Mic && Some(i) == main {
                ME.to_string()
            } else {
                n += 1;
                format!("{prefix}{n}")
            }
        })
        .collect();
    let (fallback_id, fallback_name) = match kind {
        TrackKind::Mic => (MIC, MIC_LABEL),
        TrackKind::System => (OTHERS, ctx.others_label),
    };

    let mut unattributed_talk = 0;
    let mut unattributed_first: Option<u64> = None;
    for &i in &order {
        let seg = &mut segments[i];
        let id = match &seg.speaker {
            Some(label) => {
                let voice = voices.iter().position(|v| &v.label == label);
                ids[voice.expect("every label is a voice")].clone()
            }
            None => {
                unattributed_talk += talk(seg);
                unattributed_first.get_or_insert(seg.start_ms);
                match main {
                    Some(main) => ids[main].clone(),
                    None => fallback_id.to_string(),
                }
            }
        };
        seg.speaker = Some(id);
    }

    let mut out: Vec<(Speaker, u64)> = voices
        .iter()
        .enumerate()
        .map(|(i, voice)| {
            let mut s = if Some(i) == main {
                let talk_ms = voice.talk_ms + unattributed_talk;
                speaker(&ids[i], whole_name, whole_source, talk_ms)
            } else {
                speaker(&ids[i], "", SOURCE_NONE, voice.talk_ms)
            };
            s.minor = minor[i];
            (s, voice.first_ms)
        })
        .collect();
    if let (None, Some(first)) = (main, unattributed_first) {
        out.push((
            speaker(
                fallback_id,
                fallback_name,
                SOURCE_CHANNEL,
                unattributed_talk,
            ),
            first,
        ));
        out.sort_by_key(|(_, first)| *first);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(start_ms: u64, end_ms: u64, text: &str, speaker: Option<&str>) -> Word {
        Word {
            start_ms,
            end_ms,
            text: text.into(),
            speaker: speaker.map(str::to_string),
        }
    }

    fn seg(start_ms: u64, end_ms: u64, text: &str, speaker: Option<&str>) -> Segment {
        Segment {
            start_ms,
            end_ms,
            text: text.into(),
            speaker: speaker.map(str::to_string),
            words: None,
        }
    }

    fn lines(segments: &[Segment]) -> Vec<(u64, u64, &str, Option<&str>)> {
        segments
            .iter()
            .map(|s| (s.start_ms, s.end_ms, s.text.as_str(), s.speaker.as_deref()))
            .collect()
    }

    /// One utterance holding two voices splits where the voice changes,
    /// each part with its own words and the backend's punctuation.
    #[test]
    fn a_segment_splits_where_its_words_change_voice() {
        let mut s = seg(1000, 4000, "Ready? Yes, go ahead.", Some("0"));
        s.words = Some(vec![
            word(1000, 1500, "ready", Some("0")),
            word(2000, 2400, "yes", Some("1")),
            word(2500, 2900, "go", Some("1")),
            word(3000, 4000, "ahead", Some("1")),
        ]);
        let out = split_segment(s, Labels::Backend);
        assert_eq!(
            lines(&out),
            [
                (1000, 1500, "Ready?", Some("0")),
                (2000, 4000, "Yes, go ahead.", Some("1")),
            ]
        );
        let words = out[1].words.as_ref().unwrap();
        assert_eq!(words.len(), 3);
        assert!(words.iter().all(|w| w.speaker.as_deref() == Some("1")));
    }

    /// Tokens that disagree with the words fall back to the words' own
    /// text, joined as the backend joined them: `y` and `es` run together
    /// in its text.
    #[test]
    fn a_split_without_matching_tokens_joins_the_words() {
        let mut s = seg(0, 3000, "ok so, yes", None);
        s.words = Some(vec![
            word(0, 500, "ok", Some("A")),
            word(600, 900, "so,", Some("A")),
            word(1000, 1500, "y", Some("B")),
            word(1500, 2000, "es", Some("B")),
        ]);
        let texts: Vec<String> = split_segment(s, Labels::Backend)
            .into_iter()
            .map(|s| s.text)
            .collect();
        assert_eq!(texts, ["ok so,", "yes"]);
    }

    /// A word without a label stays with its neighbours; a segment
    /// without words takes its own label.
    #[test]
    fn unlabelled_words_stay_with_their_neighbours() {
        let mut s = seg(0, 3000, "a b c d", Some("seg"));
        s.words = Some(vec![
            word(0, 500, "a", None),
            word(600, 900, "b", Some("1")),
            word(1000, 1500, "c", None),
            word(1600, 2000, "d", Some("2")),
        ]);
        let out = split_segment(s, Labels::Backend);
        assert_eq!(
            lines(&out),
            [(0, 1500, "a b c", Some("1")), (1600, 2000, "d", Some("2"))]
        );

        // No word has a label: the segment's own label covers them all.
        let mut s = seg(0, 3000, "a b", Some("seg"));
        s.words = Some(vec![word(0, 500, "a", None), word(600, 900, "b", None)]);
        let out = split_segment(s, Labels::Backend);
        assert_eq!(lines(&out), [(0, 3000, "a b", Some("seg"))]);
        assert!(out[0].words.as_ref().unwrap()[0].speaker.as_deref() == Some("seg"));

        let out = split_segment(seg(0, 900, "hi", Some("2")), Labels::Backend);
        assert_eq!(lines(&out), [(0, 900, "hi", Some("2"))]);
    }

    /// A track whose voices are not told apart drops the backend's labels.
    #[test]
    fn labels_are_dropped_when_not_separating() {
        let mut s = seg(0, 3000, "a b", Some("0"));
        s.words = Some(vec![
            word(0, 500, "a", Some("0")),
            word(600, 900, "b", Some("1")),
        ]);
        let out = split_segment(s, Labels::None);
        assert_eq!(lines(&out), [(0, 3000, "a b", None)]);
        assert!(out[0]
            .words
            .as_ref()
            .unwrap()
            .iter()
            .all(|w| w.speaker.is_none()));
    }

    fn ctx(separate_mic: bool, separate_system: bool) -> Context<'static> {
        Context {
            has_mic: true,
            has_system: true,
            separate_mic,
            separate_system,
            user_name: "Yosif",
            others_label: "Others",
        }
    }

    fn ids(segments: &[Segment]) -> Vec<&str> {
        segments
            .iter()
            .map(|s| s.speaker.as_deref().unwrap())
            .collect()
    }

    fn names(speakers: &[Speaker]) -> Vec<(&str, &str, &str)> {
        speakers
            .iter()
            .map(|s| (s.id.as_str(), s.name.as_str(), s.source.as_str()))
            .collect()
    }

    /// Without separation each track is one speaker, listed even when it
    /// holds no speech, and the backend's labels play no part.
    #[test]
    fn unseparated_tracks_are_one_speaker_each() {
        let mut mic = vec![seg(0, 1000, "hi", None)];
        let mut system: Vec<Segment> = Vec::new();
        let speakers = assign(&mut mic, &mut system, &ctx(false, false));
        assert_eq!(
            names(&speakers),
            [("me", "Yosif", "self"), ("others", "Others", "channel")]
        );
        assert_eq!(ids(&mic), ["me"]);
        assert_eq!(speakers[0].talk_ms, 1000);
        assert_eq!(speakers[0].channel, Some(TrackKind::Mic));
        assert_eq!(speakers[1].talk_ms, 0);

        // No system track recorded: no system speaker.
        let only_mic = Context {
            has_system: false,
            ..ctx(false, false)
        };
        let speakers = assign(&mut mic, &mut [], &only_mic);
        assert_eq!(names(&speakers), [("me", "Yosif", "self")]);
    }

    /// Voices are numbered by first appearance, not by the backend's own
    /// labels, and every unnamed voice is `Speaker N` in that order.
    #[test]
    fn system_voices_get_ids_by_first_appearance() {
        let mut mic = vec![seg(500, 1500, "hello", None)];
        let mut system = vec![
            seg(30_000, 40_000, "third", Some("c2:0")),
            seg(2_000, 12_000, "first", Some("c1:1")),
            seg(14_000, 24_000, "second", Some("c1:0")),
            seg(25_000, 29_000, "first again", Some("c1:1")),
        ];
        let speakers = assign(&mut mic, &mut system, &ctx(false, true));
        assert_eq!(ids(&system), ["S3", "S1", "S2", "S1"]);
        assert_eq!(
            names(&speakers),
            [
                ("me", "Yosif", "self"),
                ("S1", "Speaker 1", "none"),
                ("S2", "Speaker 2", "none"),
                ("S3", "Speaker 3", "none"),
            ]
        );
        assert_eq!(speakers[1].talk_ms, 14_000);
        assert!(speakers.iter().all(|s| !s.minor));
    }

    /// One remote voice keeps the others label, and the system speech no
    /// voice was found for goes to it.
    #[test]
    fn one_remote_voice_keeps_the_others_label() {
        let mut system = vec![
            seg(0, 10_000, "a", Some("0")),
            seg(11_000, 12_000, "b", None),
        ];
        let video = Context {
            others_label: "Video",
            ..ctx(false, true)
        };
        let speakers = assign(&mut [], &mut system, &video);
        assert_eq!(
            names(&speakers),
            [("me", "Yosif", "self"), ("S1", "Video", "channel")]
        );
        assert_eq!(ids(&system), ["S1", "S1"]);
        assert_eq!(speakers[1].talk_ms, 11_000);
    }

    /// With several voices, system speech without a voice is the track's
    /// own speaker, under the others label.
    #[test]
    fn unattributed_speech_of_several_voices_is_the_channel() {
        let mut system = vec![
            seg(0, 10_000, "a", Some("0")),
            seg(10_000, 20_000, "b", Some("1")),
            seg(21_000, 22_000, "c", None),
        ];
        let speakers = assign(&mut [], &mut system, &ctx(false, true));
        assert_eq!(ids(&system), ["S1", "S2", "others"]);
        assert_eq!(
            names(&speakers),
            [
                ("me", "Yosif", "self"),
                ("S1", "Speaker 1", "none"),
                ("S2", "Speaker 2", "none"),
                ("others", "Others", "channel"),
            ]
        );
    }

    /// A voice under 5 s and under 2 % of its track is minor: kept, with
    /// its words, but it does not count as a second voice.
    #[test]
    fn a_short_voice_is_minor() {
        assert!(is_minor(4_999, 250_000));
        assert!(!is_minor(4_999, 249_950), "2 % of the track");
        assert!(!is_minor(5_000, 1_000_000), "5 s");
        assert!(!is_minor(0, 0), "a silent track's only voice");

        let mut system = vec![
            seg(0, 300_000, "the talk", Some("0")),
            seg(300_000, 304_000, "a question", Some("1")),
        ];
        let speakers = assign(&mut [], &mut system, &ctx(false, true));
        assert_eq!(
            names(&speakers),
            [
                ("me", "Yosif", "self"),
                ("S1", "Others", "channel"),
                ("S2", "Speaker 1", "none"),
            ]
        );
        assert!(!speakers[1].minor);
        assert!(speakers[2].minor);
        assert_eq!(ids(&system), ["S1", "S2"]);
    }

    /// A shared mic with one voice is the user; with several, its voices
    /// are `M1`, `M2`, ... and unnamed, numbered with the system's voices
    /// by first appearance.
    #[test]
    fn a_shared_mic_is_the_user_only_with_one_voice() {
        let mut mic = vec![seg(0, 9_000, "hello", Some("0"))];
        let speakers = assign(&mut mic, &mut [], &ctx(true, true));
        assert_eq!(names(&speakers), [("me", "Yosif", "self")]);
        assert_eq!(ids(&mic), ["me"]);

        let mut mic = vec![
            seg(0, 9_000, "one", Some("0")),
            seg(20_000, 29_000, "two", Some("1")),
            seg(30_000, 31_000, "?", None),
        ];
        let mut system = vec![
            seg(10_000, 19_000, "far one", Some("0")),
            seg(40_000, 49_000, "far two", Some("1")),
        ];
        let speakers = assign(&mut mic, &mut system, &ctx(true, true));
        assert_eq!(ids(&mic), ["M1", "M2", "mic"]);
        assert_eq!(
            names(&speakers),
            [
                ("M1", "Speaker 1", "none"),
                ("M2", "Speaker 3", "none"),
                ("mic", "Mic", "channel"),
                ("S1", "Speaker 2", "none"),
                ("S2", "Speaker 4", "none"),
            ]
        );
    }

    /// A separated track where no voice was found is one speaker, as if not
    /// separated; a separated track without speech lists nobody.
    #[test]
    fn a_track_without_voices_is_one_speaker() {
        let mut system = vec![seg(0, 1000, "a", None)];
        let speakers = assign(&mut [], &mut system, &ctx(false, true));
        assert_eq!(
            names(&speakers),
            [("me", "Yosif", "self"), ("others", "Others", "channel")]
        );
        let speakers = assign(&mut [], &mut [], &ctx(false, true));
        assert_eq!(names(&speakers), [("me", "Yosif", "self")]);
    }
}
