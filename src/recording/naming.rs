//! Naming the speakers of a recording.
//!
//! [`super::speakers::assign`] tells the voices apart and numbers them; this
//! module names them ([`name_speakers`]). The steps run in order, and the
//! first that names a speaker wins: a later step never overrides an earlier
//! one.
//!
//! 1. **User.** A name given with `whisrs recording name`, kept in
//!    `recording.json` `speaker_names` apart from the computed list, so a
//!    retry keeps it (`user`).
//! 2. **Self.** The mic, when it is one voice (`me`), is the user
//!    (`[recording] user_name`, `self`).
//! 3. **Participants.** With exactly one remote voice and exactly one
//!    participant (`--with`), the voice is that participant (`with`).
//! 4. **The last remaining voice.** When one remote voice is left unnamed,
//!    one participant is left unassigned, there are as many remote voices as
//!    participants and the mic is not shared, that voice is that participant
//!    (`last`).
//! 5. Everyone else is `Speaker N` (`none`), numbered by first appearance
//!    among the unnamed, except a single remote voice, which keeps the
//!    others label (`channel`).
//!
//! Nothing is guessed: a voice is named only when the steps say who it is. A
//! remote voice is one on the system track (`S1`, `S2`, ...); a minor one
//! ([`super::speakers::is_minor`]) is never named after a participant and
//! does not count as a voice.

use std::collections::BTreeMap;

use super::speakers::{ME, MIC, MIC_LABEL, OTHERS, SOURCE_CHANNEL, SOURCE_NONE, SOURCE_SELF};
use super::{RecordingMeta, Speaker, TrackKind, TranscriptSegment};

/// Most people `--with` may name.
pub const PARTICIPANTS_MAX: usize = 20;

/// Longest name of a person (`--with`, `whisrs recording name`), in
/// characters.
pub const PERSON_NAME_MAX: usize = 64;

/// [`Speaker::source`] of a name the user gave (`whisrs recording name`).
pub const SOURCE_USER: &str = "user";

/// [`Speaker::source`] of the one remote voice named after the one
/// participant.
pub const SOURCE_WITH: &str = "with";

/// [`Speaker::source`] of the last remote voice named after the last
/// participant.
pub const SOURCE_LAST: &str = "last";

/// Validate and trim the name of a person: non-empty, one line, at most
/// [`PERSON_NAME_MAX`] characters.
pub fn validate_person_name(name: &str) -> Result<String, String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("a name is empty".to_string());
    }
    if trimmed.chars().any(char::is_control) {
        return Err(format!(
            "the name {trimmed:?} must be a single line of text"
        ));
    }
    let len = trimmed.chars().count();
    if len > PERSON_NAME_MAX {
        return Err(format!(
            "the name {trimmed:?} is {len} characters long: use at most {PERSON_NAME_MAX}"
        ));
    }
    Ok(trimmed.to_string())
}

/// Parse `--with "Sarah, Omar"`: names separated by commas, each checked by
/// [`validate_participants`]. A blank list names nobody, which lets a retry
/// clear the participants.
pub fn parse_participants(list: &str) -> Result<Vec<String>, String> {
    if list.trim().is_empty() {
        return Ok(Vec::new());
    }
    validate_participants(list.split(',').map(str::to_string).collect())
}

/// Check and trim the participants of a recording: at most
/// [`PARTICIPANTS_MAX`], each a valid name ([`validate_person_name`])
/// without a comma (the separator of `--with`), none listed twice.
pub fn validate_participants(names: Vec<String>) -> Result<Vec<String>, String> {
    if names.len() > PARTICIPANTS_MAX {
        return Err(format!(
            "{} participants: name at most {PARTICIPANTS_MAX}",
            names.len()
        ));
    }
    let mut out: Vec<String> = Vec::with_capacity(names.len());
    for name in names {
        let name = validate_person_name(&name).map_err(|e| {
            format!("participants: {e}; separate names with commas, e.g. \"Sarah, Omar\"")
        })?;
        if name.contains(',') {
            return Err(format!(
                "participants: {name:?} holds a comma, which separates names"
            ));
        }
        if out.iter().any(|n| normalize(n) == normalize(&name)) {
            return Err(format!("participants: {name:?} is listed twice"));
        }
        out.push(name);
    }
    Ok(out)
}

/// Lowercased, with each run of whitespace one space and none at the ends.
pub(crate) fn normalize(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// The first word of a normalized name.
fn first_word(normalized: &str) -> &str {
    normalized.split(' ').next().unwrap_or("")
}

/// Whether `name` names participant `person`: their full name, their first
/// name, or a short form of 3 or more letters that starts their first name
/// ("Sam" for "Samantha Reed"). Case and spacing do not count.
pub fn names_participant(name: &str, person: &str) -> bool {
    let name = normalize(name);
    let person = normalize(person);
    if name.is_empty() {
        return false;
    }
    let first = first_word(&person);
    name == person || name == first || (name.chars().count() >= 3 && first.starts_with(&name))
}

/// What names one recording's speakers, besides its speakers and segments.
#[derive(Debug, Clone, Copy)]
pub struct Naming<'a> {
    /// The user's name, for the mic when it is the user.
    pub user_name: &'a str,
    /// The label of the system track ([`RecordingMeta::others_label`]).
    pub others_label: &'a str,
    /// Who the recording is with (`--with`).
    pub participants: &'a [String],
    /// Several people share the mic.
    pub shared_mic: bool,
    /// Names the user gave, by speaker id (`whisrs recording name`).
    pub user_names: &'a BTreeMap<String, String>,
}

impl<'a> Naming<'a> {
    /// What names `meta`'s speakers, for a user called `user_name`.
    pub fn of(meta: &'a RecordingMeta, user_name: &'a str) -> Self {
        Naming {
            user_name,
            others_label: meta.others_label(),
            participants: &meta.participants,
            shared_mic: meta.shared_mic.unwrap_or(false),
            user_names: &meta.speaker_names,
        }
    }

    /// The participants other than the user, who is on the mic: a listed
    /// name equal to the user's (case and spacing aside) is the user.
    fn remote_participants(&self) -> Vec<&'a str> {
        let user = normalize(self.user_name);
        self.participants
            .iter()
            .map(String::as_str)
            .filter(|p| normalize(p) != user)
            .collect()
    }
}

/// A speaker's name and how it was found.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Named {
    name: String,
    source: String,
    evidence: Option<String>,
}

impl Named {
    fn new(name: &str, source: &str) -> Self {
        Named {
            name: name.to_string(),
            source: source.to_string(),
            evidence: None,
        }
    }
}

/// The track of voice id `id`: `S<n>` on the system track, `M<n>` on a
/// shared mic. `None` for a track's own speaker (`me`, `others`, `mic`).
pub fn voice_track(id: &str) -> Option<TrackKind> {
    let (prefix, number) = id.split_at_checked(1)?;
    let track = match prefix {
        "S" => TrackKind::System,
        "M" => TrackKind::Mic,
        _ => return None,
    };
    (!number.is_empty() && number.bytes().all(|b| b.is_ascii_digit())).then_some(track)
}

/// A voice on the system track that is not minor: one the participants
/// can be.
fn is_remote_voice(speaker: &Speaker) -> bool {
    voice_track(&speaker.id) == Some(TrackKind::System) && !speaker.minor
}

/// What each track says of a speaker before anyone names it: the mic's one
/// voice is the user, a track's own speaker is its label, a voice is
/// unnamed. A speaker of a kind this does not know keeps what it has.
fn base(speaker: &Speaker, naming: &Naming<'_>) -> Option<Named> {
    match speaker.id.as_str() {
        ME => Some(Named::new(naming.user_name, SOURCE_SELF)),
        OTHERS => Some(Named::new(naming.others_label, SOURCE_CHANNEL)),
        MIC => Some(Named::new(MIC_LABEL, SOURCE_CHANNEL)),
        id if voice_track(id).is_some() => None,
        _ => Some(Named {
            name: speaker.name.clone(),
            source: speaker.source.clone(),
            evidence: speaker.evidence.clone(),
        }),
    }
}

/// Whether a speaker already goes by participant `person`'s name. Only the
/// names given to people count: a track's label or the user's own name never
/// assigns a participant.
fn assigned(person: &str, named: &[Option<Named>]) -> bool {
    named.iter().flatten().any(|n| {
        matches!(n.source.as_str(), SOURCE_USER | SOURCE_WITH | SOURCE_LAST)
            && names_participant(&n.name, person)
    })
}

/// When each speaker first speaks in `segments`; never for one without a
/// segment.
fn first_appearance(speakers: &[Speaker], segments: &[TranscriptSegment]) -> Vec<u64> {
    speakers
        .iter()
        .map(|s| {
            segments
                .iter()
                .filter(|seg| seg.speaker == s.id)
                .map(|seg| seg.start_ms)
                .min()
                .unwrap_or(u64::MAX)
        })
        .collect()
}

/// The steps of the module docs, user to last voice: each speaker's name
/// and source, `None` when no step names it.
fn named_by_steps(speakers: &[Speaker], naming: &Naming<'_>) -> Vec<Option<Named>> {
    let mut named: Vec<Option<Named>> = speakers.iter().map(|s| base(s, naming)).collect();

    // 1. The user, over everything else.
    for (i, speaker) in speakers.iter().enumerate() {
        if let Some(name) = naming.user_names.get(&speaker.id) {
            named[i] = Some(Named::new(name, SOURCE_USER));
        }
    }

    let remote: Vec<usize> = (0..speakers.len())
        .filter(|&i| is_remote_voice(&speakers[i]))
        .collect();
    let participants = naming.remote_participants();

    // 3. One remote voice, one participant. Not with others on a shared
    // mic: the one participant may be in the room.
    let others_on_mic = speakers
        .iter()
        .any(|s| voice_track(&s.id) == Some(TrackKind::Mic) && !s.minor);
    if let ([voice], [person]) = (remote.as_slice(), participants.as_slice()) {
        if !others_on_mic && named[*voice].is_none() && !assigned(person, &named) {
            named[*voice] = Some(Named::new(person, SOURCE_WITH));
        }
    }

    // 4. The last remaining voice is the last remaining participant, when
    // the voices and the participants are as many and nobody else is in the
    // room.
    if !naming.shared_mic && remote.len() == participants.len() {
        let unnamed: Vec<usize> = remote
            .iter()
            .copied()
            .filter(|&i| named[i].is_none())
            .collect();
        let left: Vec<&str> = participants
            .iter()
            .copied()
            .filter(|p| !assigned(p, &named))
            .collect();
        if let ([voice], [person]) = (unnamed.as_slice(), left.as_slice()) {
            named[*voice] = Some(Named::new(person, SOURCE_LAST));
        }
    }
    named
}

/// Every speaker's name and source ([`named_by_steps`]), the unnamed ones
/// as `Speaker N` by first appearance in `segments`, or a single remote
/// voice by the others label.
fn decide(speakers: &[Speaker], segments: &[TranscriptSegment], naming: &Naming<'_>) -> Vec<Named> {
    let mut named = named_by_steps(speakers, naming);
    let mut remote = (0..speakers.len()).filter(|&i| is_remote_voice(&speakers[i]));
    let single_remote = match (remote.next(), remote.next()) {
        (Some(only), None) => Some(only),
        _ => None,
    };
    let first = first_appearance(speakers, segments);
    let mut unnamed: Vec<usize> = (0..speakers.len())
        .filter(|&i| named[i].is_none())
        .collect();
    unnamed.sort_by_key(|&i| (first[i], i));
    let mut n = 0;
    for i in unnamed {
        named[i] = Some(if Some(i) == single_remote {
            Named::new(naming.others_label, SOURCE_CHANNEL)
        } else {
            n += 1;
            Named::new(&format!("Speaker {n}"), SOURCE_NONE)
        });
    }
    named
        .into_iter()
        .map(|n| n.expect("every speaker is named"))
        .collect()
}

/// Name `meta`'s speakers, from scratch: every step of the module docs, for
/// a user called `user_name`. Their ids, talk times and order stay.
pub fn name_speakers(meta: &mut RecordingMeta, user_name: &str) {
    let named = decide(&meta.speakers, &meta.segments, &Naming::of(meta, user_name));
    for (speaker, named) in meta.speakers.iter_mut().zip(named) {
        speaker.name = named.name;
        speaker.source = named.source;
        speaker.evidence = named.evidence;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speaker(id: &str, channel: TrackKind) -> Speaker {
        Speaker {
            id: id.into(),
            name: String::new(),
            source: String::new(),
            evidence: None,
            talk_ms: 60_000,
            channel: Some(channel),
            minor: false,
        }
    }

    fn minor(id: &str) -> Speaker {
        Speaker {
            minor: true,
            talk_ms: 1_000,
            ..speaker(id, TrackKind::System)
        }
    }

    fn seg(start_ms: u64, speaker: &str) -> TranscriptSegment {
        TranscriptSegment {
            start_ms,
            end_ms: start_ms + 1000,
            speaker: speaker.into(),
            channel: if speaker.starts_with('S') || speaker == OTHERS {
                TrackKind::System
            } else {
                TrackKind::Mic
            },
            text: "words".into(),
        }
    }

    /// A call: the user, then the system voices in the order given, each
    /// first heard a second after the one before.
    fn call(voices: &[Speaker]) -> (Vec<Speaker>, Vec<TranscriptSegment>) {
        let mut speakers = vec![speaker(ME, TrackKind::Mic)];
        speakers.extend(voices.iter().cloned());
        let segments = speakers
            .iter()
            .enumerate()
            .map(|(i, s)| seg(i as u64 * 1000, &s.id))
            .collect();
        (speakers, segments)
    }

    fn people(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    struct Case {
        participants: Vec<String>,
        shared_mic: bool,
        user_names: BTreeMap<String, String>,
    }

    impl Case {
        fn with(names: &[&str]) -> Self {
            Case {
                participants: people(names),
                shared_mic: false,
                user_names: BTreeMap::new(),
            }
        }

        fn user(mut self, id: &str, name: &str) -> Self {
            self.user_names.insert(id.into(), name.into());
            self
        }

        fn naming(&self) -> Naming<'_> {
            Naming {
                user_name: "Yosif",
                others_label: "Others",
                participants: &self.participants,
                shared_mic: self.shared_mic,
                user_names: &self.user_names,
            }
        }

        fn run(
            &self,
            speakers: &[Speaker],
            segments: &[TranscriptSegment],
        ) -> Vec<(String, String)> {
            decide(speakers, segments, &self.naming())
                .into_iter()
                .zip(speakers)
                .map(|(n, s)| (s.id.clone(), format!("{} ({})", n.name, n.source)))
                .collect()
        }
    }

    fn expect(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect()
    }

    /// The mic is the user; one remote voice keeps the others label, and
    /// more are `Speaker N` by first appearance, without participants.
    #[test]
    fn without_participants_nobody_is_named() {
        let (speakers, segments) = call(&[speaker("S1", TrackKind::System)]);
        assert_eq!(
            Case::with(&[]).run(&speakers, &segments),
            expect(&[("me", "Yosif (self)"), ("S1", "Others (channel)")])
        );
        let (speakers, mut segments) = call(&[
            speaker("S1", TrackKind::System),
            speaker("S2", TrackKind::System),
        ]);
        // S2 speaks first: it is Speaker 1.
        segments[2].start_ms = 100;
        assert_eq!(
            Case::with(&[]).run(&speakers, &segments),
            expect(&[
                ("me", "Yosif (self)"),
                ("S1", "Speaker 2 (none)"),
                ("S2", "Speaker 1 (none)"),
            ])
        );
    }

    /// Only a name given to a person assigns a participant: the user, here
    /// "Sam", is not the participant "Samantha" whose name theirs starts.
    #[test]
    fn the_users_own_name_assigns_nobody() {
        let (speakers, segments) = call(&[speaker("S1", TrackKind::System)]);
        let case = Case::with(&["Samantha"]);
        let naming = Naming {
            user_name: "Sam",
            ..case.naming()
        };
        let named = decide(&speakers, &segments, &naming);
        assert_eq!(
            (named[1].name.as_str(), named[1].source.as_str()),
            ("Samantha", "with")
        );
    }

    /// A track's own speaker holds speech no voice was found for: it is
    /// never named after a participant, and never counts as a voice.
    #[test]
    fn a_tracks_own_speaker_is_never_a_participant() {
        let (speakers, segments) = call(&[
            speaker("S1", TrackKind::System),
            speaker("S2", TrackKind::System),
            speaker(OTHERS, TrackKind::System),
        ]);
        let case = Case::with(&["Sarah", "Omar"]).user("S1", "Sarah");
        assert_eq!(
            case.run(&speakers, &segments),
            expect(&[
                ("me", "Yosif (self)"),
                ("S1", "Sarah (user)"),
                ("S2", "Omar (last)"),
                ("others", "Others (channel)"),
            ])
        );
        let speakers = vec![
            speaker("M1", TrackKind::Mic),
            speaker(MIC, TrackKind::Mic),
            speaker("S1", TrackKind::System),
        ];
        let segments: Vec<_> = speakers
            .iter()
            .enumerate()
            .map(|(i, s)| seg(i as u64 * 1000, &s.id))
            .collect();
        let mut case = Case::with(&["Sarah"]);
        case.shared_mic = true;
        assert_eq!(
            case.run(&speakers, &segments),
            expect(&[
                ("M1", "Speaker 1 (none)"),
                ("mic", "Mic (channel)"),
                ("S1", "Others (channel)"),
            ])
        );
    }

    /// One remote voice and one participant: the voice is them. A minor
    /// voice beside it does not count, and is not named.
    #[test]
    fn one_voice_and_one_participant_is_with() {
        let (speakers, segments) = call(&[speaker("S1", TrackKind::System)]);
        assert_eq!(
            Case::with(&["Sarah"]).run(&speakers, &segments),
            expect(&[("me", "Yosif (self)"), ("S1", "Sarah (with)")])
        );
        let (speakers, segments) = call(&[speaker("S1", TrackKind::System), minor("S2")]);
        assert_eq!(
            Case::with(&["Sarah"]).run(&speakers, &segments),
            expect(&[
                ("me", "Yosif (self)"),
                ("S1", "Sarah (with)"),
                ("S2", "Speaker 1 (none)"),
            ])
        );
        // The user listed among the participants is the user, on the mic.
        let (speakers, segments) = call(&[speaker("S1", TrackKind::System)]);
        assert_eq!(
            Case::with(&["yosif", "Sarah"]).run(&speakers, &segments),
            expect(&[("me", "Yosif (self)"), ("S1", "Sarah (with)")])
        );
    }

    /// Two participants for one voice, or two voices for one participant:
    /// nobody is named.
    #[test]
    fn with_needs_one_of_each() {
        let (speakers, segments) = call(&[speaker("S1", TrackKind::System)]);
        assert_eq!(
            Case::with(&["Sarah", "Omar"]).run(&speakers, &segments),
            expect(&[("me", "Yosif (self)"), ("S1", "Others (channel)")])
        );
        let (speakers, segments) = call(&[
            speaker("S1", TrackKind::System),
            speaker("S2", TrackKind::System),
        ]);
        assert_eq!(
            Case::with(&["Sarah"]).run(&speakers, &segments),
            expect(&[
                ("me", "Yosif (self)"),
                ("S1", "Speaker 1 (none)"),
                ("S2", "Speaker 2 (none)"),
            ])
        );
        // An unseparated system track is not a voice: it keeps its label.
        let (speakers, segments) = call(&[speaker(OTHERS, TrackKind::System)]);
        assert_eq!(
            Case::with(&["Sarah"]).run(&speakers, &segments),
            expect(&[("me", "Yosif (self)"), ("others", "Others (channel)")])
        );
    }

    /// A shared mic with other voices on it: the one participant may be in
    /// the room, so the one remote voice is not named after them.
    #[test]
    fn with_skips_a_shared_mic_with_other_voices() {
        let speakers = vec![
            speaker("M1", TrackKind::Mic),
            speaker("M2", TrackKind::Mic),
            speaker("S1", TrackKind::System),
        ];
        let segments: Vec<_> = speakers
            .iter()
            .enumerate()
            .map(|(i, s)| seg(i as u64 * 1000, &s.id))
            .collect();
        let mut case = Case::with(&["Sarah"]);
        case.shared_mic = true;
        assert_eq!(
            case.run(&speakers, &segments),
            expect(&[
                ("M1", "Speaker 1 (none)"),
                ("M2", "Speaker 2 (none)"),
                ("S1", "Others (channel)"),
            ])
        );
        // A shared mic that held one voice is the user alone.
        let (speakers, segments) = call(&[speaker("S1", TrackKind::System)]);
        assert_eq!(
            case.run(&speakers, &segments),
            expect(&[("me", "Yosif (self)"), ("S1", "Sarah (with)")])
        );
    }

    fn three_voices() -> (Vec<Speaker>, Vec<TranscriptSegment>) {
        call(&[
            speaker("S1", TrackKind::System),
            speaker("S2", TrackKind::System),
            speaker("S3", TrackKind::System),
        ])
    }

    /// The user's names win over every step, on any speaker, and the
    /// unnamed are numbered among themselves.
    #[test]
    fn the_users_names_win() {
        let (speakers, segments) = three_voices();
        let case = Case::with(&[])
            .user("S2", "Dana")
            .user("me", "Yo")
            .user("S9", "Ghost");
        assert_eq!(
            case.run(&speakers, &segments),
            expect(&[
                ("me", "Yo (user)"),
                ("S1", "Speaker 1 (none)"),
                ("S2", "Dana (user)"),
                ("S3", "Speaker 2 (none)"),
            ])
        );
        // Over `with` too.
        let (speakers, segments) = call(&[speaker("S1", TrackKind::System)]);
        assert_eq!(
            Case::with(&["Sarah"])
                .user("S1", "Omar")
                .run(&speakers, &segments),
            expect(&[("me", "Yosif (self)"), ("S1", "Omar (user)")])
        );
    }

    /// With every remote voice but one named and one participant left,
    /// the last voice is them.
    #[test]
    fn the_last_voice_is_the_last_participant() {
        let (speakers, segments) = three_voices();
        let case = Case::with(&["Sarah", "Omar Haddad", "Dana"])
            .user("S1", "Sarah")
            // A first name assigns the participant it starts.
            .user("S3", "omar");
        assert_eq!(
            case.run(&speakers, &segments),
            expect(&[
                ("me", "Yosif (self)"),
                ("S1", "Sarah (user)"),
                ("S2", "Dana (last)"),
                ("S3", "omar (user)"),
            ])
        );
    }

    /// The last voice is skipped unless voices and participants are as
    /// many, the mic is not shared, one voice and one participant are
    /// left, and minor voices do not count.
    #[test]
    fn the_last_voice_has_exclusions() {
        let (speakers, segments) = three_voices();
        let named = |case: &Case| {
            case.run(&speakers, &segments)
                .into_iter()
                .find(|(id, _)| id == "S2")
                .unwrap()
                .1
        };
        let two_named = |names: &[&str]| Case::with(names).user("S1", "Sarah").user("S3", "Omar");

        // Three voices, four participants: someone did not speak.
        assert_eq!(
            named(&two_named(&["Sarah", "Omar", "Dana", "Lee"])),
            "Speaker 1 (none)"
        );
        // Three voices, two participants: someone was not listed (here
        // Bob), so the last voice need not be the last participant.
        let case = Case::with(&["Sarah", "Dana"])
            .user("S1", "Sarah")
            .user("S3", "Bob");
        assert_eq!(named(&case), "Speaker 1 (none)");
        // A shared mic: the participants may be in the room.
        let mut shared = two_named(&["Sarah", "Omar", "Dana"]);
        shared.shared_mic = true;
        assert_eq!(named(&shared), "Speaker 1 (none)");
        // The user named S3 after a participant who is already S1: two
        // participants are left for one voice.
        let case = Case::with(&["Sarah", "Omar", "Dana"])
            .user("S1", "Sarah")
            .user("S3", "Sarah");
        assert_eq!(named(&case), "Speaker 1 (none)");
        // Named, but not after a participant: two are left.
        let case = Case::with(&["Sarah", "Omar", "Dana"])
            .user("S1", "Sarah")
            .user("S3", "Bob");
        assert_eq!(named(&case), "Speaker 1 (none)");
        // The baseline: all four hold.
        assert_eq!(named(&two_named(&["Sarah", "Omar", "Dana"])), "Dana (last)");

        // A minor voice is not a voice: two voices for two participants.
        let (speakers, segments) = call(&[
            speaker("S1", TrackKind::System),
            minor("S2"),
            speaker("S3", TrackKind::System),
        ]);
        let case = Case::with(&["Sarah", "Omar"]).user("S1", "Sarah");
        assert_eq!(
            case.run(&speakers, &segments),
            expect(&[
                ("me", "Yosif (self)"),
                ("S1", "Sarah (user)"),
                ("S2", "Speaker 1 (none)"),
                ("S3", "Omar (last)"),
            ])
        );
    }

    /// A participant's first name, full name or a 3+ letter start of the
    /// first name names them; nothing shorter, nothing else.
    #[test]
    fn names_match_participants_loosely_but_not_too_loosely() {
        for (name, person) in [
            ("Sarah", "Sarah"),
            ("sarah  smith", "Sarah Smith"),
            ("Sarah", "Sarah Smith"),
            ("Sam", "Samantha Reed"),
            ("SAMANT", "Samantha"),
        ] {
            assert!(names_participant(name, person), "{name} / {person}");
        }
        for (name, person) in [
            ("Sa", "Sarah"),
            ("Smith", "Sarah Smith"),
            ("Sarah Jones", "Sarah Smith"),
            ("Sarahs", "Sarah"),
            ("", "Sarah"),
        ] {
            assert!(!names_participant(name, person), "{name} / {person}");
        }
    }

    /// `--with` takes names separated by commas, trimmed, each one line
    /// and at most 64 characters, at most 20 of them, none twice; a blank
    /// list is nobody.
    #[test]
    fn participants_are_validated() {
        assert_eq!(
            parse_participants(" Sarah ,Omar Haddad"),
            Ok(people(&["Sarah", "Omar Haddad"]))
        );
        assert_eq!(parse_participants("  "), Ok(Vec::new()));
        let longest = "é".repeat(PERSON_NAME_MAX);
        assert_eq!(parse_participants(&longest), Ok(vec![longest.clone()]));
        let twenty = vec!["x"; PARTICIPANTS_MAX]
            .iter()
            .enumerate()
            .map(|(i, x)| format!("{x}{i}"))
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(parse_participants(&twenty).unwrap().len(), PARTICIPANTS_MAX);

        for (list, want) in [
            ("Sarah,,Omar", "a name is empty"),
            ("Sarah,", "a name is empty"),
            (&format!("{longest}x"), "at most 64"),
            ("Sarah\u{7}", "single line"),
            ("Sarah, sarah", "listed twice"),
            (&format!("{twenty},y"), "at most 20"),
        ] {
            let err = parse_participants(list).unwrap_err();
            assert!(err.contains(want), "{list:?}: {err}");
        }
        // Over IPC a name arrives whole: a comma in it is refused.
        let err = validate_participants(people(&["Sarah, Omar"])).unwrap_err();
        assert!(err.contains("comma"), "{err}");
        assert_eq!(
            validate_participants(people(&[" Sarah\t"])),
            Ok(people(&["Sarah"]))
        );
    }

    #[test]
    fn voice_ids_name_their_track() {
        assert_eq!(voice_track("S1"), Some(TrackKind::System));
        assert_eq!(voice_track("M12"), Some(TrackKind::Mic));
        for id in ["me", "others", "mic", "S", "Sx", "s1", "", "Ş1"] {
            assert_eq!(voice_track(id), None, "{id}");
        }
    }

    /// Naming from scratch: names, sources and evidence from an earlier
    /// run go, ids, talk times and order stay.
    #[test]
    fn name_speakers_renames_in_place() {
        let mut meta = RecordingMeta::new("m".into(), None, chrono::Local::now());
        let (speakers, segments) = call(&[speaker("S1", TrackKind::System)]);
        meta.speakers = speakers;
        meta.segments = segments;
        meta.speakers[1].name = "Stale".into();
        meta.speakers[1].source = "llm".into();
        meta.speakers[1].evidence = Some("old".into());
        meta.others_label = Some("Video".into());
        name_speakers(&mut meta, "Tester");
        let got: Vec<_> = meta
            .speakers
            .iter()
            .map(|s| {
                (
                    s.id.as_str(),
                    s.name.as_str(),
                    s.source.as_str(),
                    s.evidence.clone(),
                )
            })
            .collect();
        assert_eq!(
            got,
            [
                ("me", "Tester", "self", None),
                ("S1", "Video", "channel", None)
            ]
        );
        meta.participants = people(&["Sarah"]);
        name_speakers(&mut meta, "Tester");
        assert_eq!(
            (
                meta.speakers[1].name.as_str(),
                meta.speakers[1].source.as_str()
            ),
            ("Sarah", "with")
        );
        assert_eq!(meta.speakers[1].talk_ms, 60_000);
    }
}
