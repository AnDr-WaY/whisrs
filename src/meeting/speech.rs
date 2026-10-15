//! Sending only the speech in a meeting track.
//!
//! Whisper backends normalise their input, so a long quiet stretch (what is
//! left of the far side after echo removal, room noise) comes back as
//! invented sentences. A track is therefore sent as its speech only: every
//! 100 ms frame is marked active ([`energy_mask`], or the echo canceller's
//! near-end detector for the mic), active frames grow into padded
//! [`regions`], and the regions are packed back to back with a second of
//! digital silence between them into condensed chunks ([`pack`]). Each
//! chunk keeps a [`TimeMap`] from its condensed audio to the track, which
//! [`TimeMap::map_segment`] uses to put the backend's segments back on the
//! track's own timeline.

use std::ops::Range;

use audio_silence_gate::SILENCE_RMS_THRESHOLD;

use super::split::split_by_energies;
use super::{Segment, Word};

const RATE: usize = super::SAMPLE_RATE as usize;

/// Samples in one activity frame (100 ms), the frame of the track scans.
pub const FRAME: usize = RATE / 10;

/// Each run of active frames is widened by this many frames (300 ms) on
/// both sides, so word onsets and tails stay in.
const PAD_FRAMES: usize = 3;

/// Regions closer than this many frames (1 s) are merged.
const MERGE_FRAMES: usize = 10;

/// A region with less active time than this is a click, not speech: the
/// dictation gate's shortest recording.
const MIN_SPEECH_MS: usize = 300;

/// Digital silence between packed regions (1 s): longer than the default
/// `phrase_silence_ms` (400 ms), so local whisper's phrase split and
/// Deepgram's utterances both break at every join.
pub const JOIN_GAP: usize = RATE;

/// Activity from frame RMS ([`audio_silence_gate::rms_energy`] scale): a
/// frame is active at [`SILENCE_RMS_THRESHOLD`] or above.
pub fn energy_mask(energies: &[f64]) -> Vec<bool> {
    energies
        .iter()
        .map(|&e| e >= SILENCE_RMS_THRESHOLD)
        .collect()
}

/// Frames in a `len`-sample track, the last one possibly partial.
pub fn frame_count(len: usize) -> usize {
    len.div_ceil(FRAME)
}

/// The sample ranges of a `len`-sample track to send, from its per-frame
/// activity: each run of active frames widened by [`PAD_FRAMES`], regions
/// less than [`MERGE_FRAMES`] apart merged, and regions holding less than
/// [`MIN_SPEECH_MS`] of active frames dropped. Disjoint, in order, and
/// within the track. Empty when there is nothing to send.
pub fn regions(active: &[bool], len: usize) -> Vec<Range<usize>> {
    let frames = frame_count(len).min(active.len());
    // (frame range, active frames in it)
    let mut merged: Vec<(Range<usize>, usize)> = Vec::new();
    let mut i = 0;
    while i < frames {
        if !active[i] {
            i += 1;
            continue;
        }
        let run_start = i;
        while i < frames && active[i] {
            i += 1;
        }
        let padded = run_start.saturating_sub(PAD_FRAMES)..(i + PAD_FRAMES).min(frames);
        match merged.last_mut() {
            Some((last, count)) if padded.start < last.end + MERGE_FRAMES => {
                last.end = padded.end;
                *count += i - run_start;
            }
            _ => merged.push((padded, i - run_start)),
        }
    }
    let frame_ms = FRAME * 1000 / RATE;
    merged
        .into_iter()
        .filter(|(_, count)| count * frame_ms >= MIN_SPEECH_MS)
        .map(|(r, _)| r.start * FRAME..(r.end * FRAME).min(len))
        .collect()
}

/// One piece of a condensed chunk: `len` samples at `condensed` in the
/// chunk are samples `original..original + len` of the track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    pub condensed: usize,
    pub original: usize,
    pub len: usize,
}

/// A condensed chunk: its pieces in order, [`JOIN_GAP`] apart.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TimeMap {
    pub pieces: Vec<Piece>,
}

/// Pack `regions` in order into condensed chunks of at most `cap_secs`
/// each (`None`: one chunk), with [`JOIN_GAP`] of silence between regions.
/// A region longer than the cap is first cut where [`split_by_energies`]
/// cuts a track, in the quietest stretch before the cap, using the track's
/// frame `energies` (one per [`FRAME`]).
pub fn pack(regions: &[Range<usize>], energies: &[f64], cap_secs: Option<usize>) -> Vec<TimeMap> {
    let cap = cap_secs.map(|secs| (secs * RATE).max(FRAME));
    let parts = regions.iter().flat_map(|region| -> Vec<Range<usize>> {
        match cap_secs {
            Some(secs) if region.len() > secs * RATE => {
                // Regions start on a frame, so the slice lines up.
                let from = (region.start / FRAME).min(energies.len());
                split_by_energies(
                    &energies[from..],
                    region.len(),
                    RATE,
                    secs,
                    SILENCE_RMS_THRESHOLD,
                )
                .into_iter()
                .map(|r| region.start + r.start..region.start + r.end)
                .collect()
            }
            _ => vec![region.clone()],
        }
    });
    let mut chunks = Vec::new();
    let mut chunk = TimeMap::default();
    for part in parts {
        let at = if chunk.pieces.is_empty() {
            0
        } else {
            chunk.condensed_len() + JOIN_GAP
        };
        if !chunk.pieces.is_empty() && cap.is_some_and(|cap| at + part.len() > cap) {
            chunks.push(std::mem::take(&mut chunk));
            chunk.pieces.push(Piece {
                condensed: 0,
                original: part.start,
                len: part.len(),
            });
            continue;
        }
        chunk.pieces.push(Piece {
            condensed: at,
            original: part.start,
            len: part.len(),
        });
    }
    if !chunk.pieces.is_empty() {
        chunks.push(chunk);
    }
    chunks
}

fn ms_to_samples(ms: u64) -> usize {
    (ms as usize).saturating_mul(RATE / 1000)
}

fn samples_to_ms(samples: usize) -> u64 {
    (samples * 1000 / RATE) as u64
}

impl TimeMap {
    /// Length of the condensed audio in samples.
    pub fn condensed_len(&self) -> usize {
        self.pieces.last().map_or(0, |p| p.condensed + p.len)
    }

    /// The piece holding condensed sample `at`: in a gap, the nearer one.
    fn locate(&self, at: usize) -> usize {
        let i = self.pieces.partition_point(|p| p.condensed + p.len <= at);
        if i == self.pieces.len() {
            return i.saturating_sub(1);
        }
        if i == 0 || at >= self.pieces[i].condensed {
            return i;
        }
        let before = at - (self.pieces[i - 1].condensed + self.pieces[i - 1].len);
        let after = self.pieces[i].condensed - at;
        if before <= after {
            i - 1
        } else {
            i
        }
    }

    /// Condensed time `ms` on the track's timeline through piece `piece`,
    /// clamped into that piece.
    fn to_track(&self, ms: u64, piece: usize) -> u64 {
        let p = self.pieces[piece];
        let into = ms_to_samples(ms).saturating_sub(p.condensed).min(p.len);
        samples_to_ms(p.original + into)
    }

    /// `seg`, timed in this chunk's condensed audio, timed on the track.
    ///
    /// A segment whose words all fall in one piece is mapped through it. One
    /// whose words span several pieces (a backend that ran two regions into
    /// one sentence across a join) is split, each word going to the piece
    /// holding its midpoint, or the nearer piece when it lies in a gap. The
    /// split keeps the backend's punctuated text when it has one token per
    /// word, else uses the words' own text. Without word timings start and
    /// end are mapped on their own, and an end that lands in a later piece
    /// is clamped to the end of the start's piece, so the segment never
    /// spans the silence the join stood for.
    pub fn map_segment(&self, seg: Segment) -> Vec<Segment> {
        if self.pieces.is_empty() {
            return Vec::new();
        }
        let words = match &seg.words {
            Some(words) if !words.is_empty() => words.clone(),
            _ => {
                // Both ends through the start's piece: an end past it is
                // clamped to its end.
                let piece = self.locate(ms_to_samples(seg.start_ms));
                let start_ms = self.to_track(seg.start_ms, piece);
                let end_ms = self.to_track(seg.end_ms, piece);
                return vec![Segment {
                    start_ms,
                    end_ms: end_ms.max(start_ms),
                    words: None,
                    ..seg
                }];
            }
        };
        let homes: Vec<usize> = words
            .iter()
            .map(|w| self.locate(ms_to_samples((w.start_ms + w.end_ms) / 2)))
            .collect();
        let map_word = |w: &Word, piece: usize| {
            let start_ms = self.to_track(w.start_ms, piece);
            Word {
                start_ms,
                end_ms: self.to_track(w.end_ms, piece).max(start_ms),
                ..w.clone()
            }
        };
        if homes.iter().all(|&h| h == homes[0]) {
            let piece = homes[0];
            let start_ms = self.to_track(seg.start_ms, piece);
            return vec![Segment {
                start_ms,
                end_ms: self.to_track(seg.end_ms, piece).max(start_ms),
                words: Some(words.iter().map(|w| map_word(w, piece)).collect()),
                ..seg
            }];
        }

        let tokens: Vec<&str> = seg.text.split_whitespace().collect();
        let keep_tokens = tokens.len() == words.len();
        let mut out = Vec::new();
        let mut i = 0;
        while i < words.len() {
            let piece = homes[i];
            let mut j = i;
            while j < words.len() && homes[j] == piece {
                j += 1;
            }
            let text = if keep_tokens {
                tokens[i..j].join(" ")
            } else {
                let texts: Vec<&str> = words[i..j].iter().map(|w| w.text.as_str()).collect();
                texts.join(" ")
            };
            let mapped: Vec<Word> = words[i..j].iter().map(|w| map_word(w, piece)).collect();
            let start_ms = mapped[0].start_ms;
            let end_ms = mapped[mapped.len() - 1].end_ms.max(start_ms);
            out.push(Segment {
                start_ms,
                end_ms,
                text,
                speaker: seg.speaker.clone(),
                words: Some(mapped),
            });
            i = j;
        }
        out
    }
}

/// The on-disk form of a mic's near-end mask: the track's sample count
/// (little-endian u64), then one bit per [`FRAME`], least significant bit
/// first.
pub fn encode_mask(samples: usize, active: &[bool]) -> Vec<u8> {
    let mut out = (samples as u64).to_le_bytes().to_vec();
    out.extend(active.chunks(8).map(|bits| {
        bits.iter()
            .enumerate()
            .fold(0u8, |byte, (i, &on)| byte | (u8::from(on) << i))
    }));
    out
}

/// Read back [`encode_mask`] for a track of `samples` samples, or `None`
/// when it was written for another length or is cut short.
pub fn decode_mask(bytes: &[u8], samples: usize) -> Option<Vec<bool>> {
    let (head, bits) = bytes.split_at_checked(8)?;
    if u64::from_le_bytes(head.try_into().ok()?) != samples as u64 {
        return None;
    }
    let frames = frame_count(samples);
    if bits.len() != frames.div_ceil(8) {
        return None;
    }
    Some(
        (0..frames)
            .map(|i| bits[i / 8] & (1 << (i % 8)) != 0)
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meeting::split::frame_len;

    const SEC: usize = RATE;

    /// Activity of a `secs`-second track with `(start_s, end_s)` spans on.
    fn mask(secs: f64, spans: &[(f64, f64)]) -> Vec<bool> {
        let frames = (secs * 10.0).round() as usize;
        (0..frames)
            .map(|f| {
                let t = f as f64 / 10.0 + 0.05;
                spans.iter().any(|&(a, b)| t > a && t < b)
            })
            .collect()
    }

    #[test]
    fn frame_matches_the_scans() {
        assert_eq!(FRAME, frame_len(RATE));
    }

    #[test]
    fn regions_pad_merge_and_drop_clicks() {
        let len = 60 * SEC;
        // Speech 10-12 s; another word 0.5 s later (merged); a 1.2 s gap
        // later still is its own region. A 200 ms click at 40 s is dropped,
        // and two of them 0.4 s apart still make only 200 ms of sound.
        let active = mask(
            60.0,
            &[
                (10.0, 12.0),
                (12.5, 13.0),
                (14.8, 15.4),
                (40.0, 40.2),
                (50.0, 50.1),
                (50.5, 50.6),
            ],
        );
        let got = regions(&active, len);
        let secs: Vec<(f64, f64)> = got
            .iter()
            .map(|r| (r.start as f64 / SEC as f64, r.end as f64 / SEC as f64))
            .collect();
        assert_eq!(secs, [(9.7, 13.3), (14.5, 15.7)]);
    }

    #[test]
    fn regions_stay_inside_the_track() {
        // Active at both ends of a track that ends mid-frame.
        let len = 5 * SEC + 700;
        let mut active = vec![false; frame_count(len)];
        active[..5].fill(true);
        let last = active.len() - 1;
        active[last - 4..].fill(true);
        assert_eq!(regions(&active, len), [0..8 * FRAME, 43 * FRAME..len]);
        assert!(regions(&[], len).is_empty());
        assert!(regions(&vec![false; frame_count(len)], len).is_empty());
        assert!(regions(&[true; 3], 0).is_empty(), "an empty track");
    }

    #[test]
    fn packing_keeps_gaps_and_the_cap() {
        let regions = [SEC..3 * SEC, 10 * SEC..12 * SEC, 20 * SEC..25 * SEC];
        let energies = vec![0.1; 300];
        let one = pack(&regions, &energies, None);
        assert_eq!(one.len(), 1);
        assert_eq!(
            one[0].pieces,
            [
                Piece {
                    condensed: 0,
                    original: SEC,
                    len: 2 * SEC
                },
                Piece {
                    condensed: 3 * SEC,
                    original: 10 * SEC,
                    len: 2 * SEC
                },
                Piece {
                    condensed: 6 * SEC,
                    original: 20 * SEC,
                    len: 5 * SEC
                },
            ]
        );
        assert_eq!(one[0].condensed_len(), 11 * SEC);

        // A 6 s cap: the first two fit (2 + 1 + 2 s), the third starts anew.
        let capped = pack(&regions, &energies, Some(6));
        assert_eq!(capped.len(), 2);
        assert_eq!(capped[0].condensed_len(), 5 * SEC);
        assert_eq!(
            capped[1].pieces,
            [Piece {
                condensed: 0,
                original: 20 * SEC,
                len: 5 * SEC
            }]
        );
        assert!(capped.iter().all(|c| c.condensed_len() <= 6 * SEC));
    }

    #[test]
    fn a_region_over_the_cap_is_cut_at_its_quietest_point() {
        // 30 s of speech with one quiet frame at 7.0 s, at a 10 s cap.
        let mut energies = vec![0.1; 400];
        energies[100 + 70] = 0.0;
        let region = 10 * SEC..40 * SEC;
        let chunks = pack(std::slice::from_ref(&region), &energies, Some(10));
        let cuts: Vec<usize> = chunks.iter().map(|c| c.pieces[0].original).collect();
        assert_eq!(cuts[1], 17 * SEC, "cut in the quiet frame");
        assert!(chunks
            .iter()
            .all(|c| c.condensed_len() <= 10 * SEC && c.pieces.len() == 1));
        let total: usize = chunks.iter().map(TimeMap::condensed_len).sum();
        assert_eq!(total, region.len(), "every sample sent once");
    }

    fn word(start_ms: u64, end_ms: u64, text: &str) -> Word {
        Word {
            start_ms,
            end_ms,
            text: text.into(),
            speaker: None,
        }
    }

    fn segment(start_ms: u64, end_ms: u64, text: &str, words: Option<Vec<Word>>) -> Segment {
        Segment {
            start_ms,
            end_ms,
            text: text.into(),
            speaker: Some("0".into()),
            words,
        }
    }

    /// Two regions, 0.5-2.5 s and 60-62 s of the track, condensed to 0-2 s
    /// and 3-5 s.
    fn two_pieces() -> TimeMap {
        pack(&[SEC / 2..5 * SEC / 2, 60 * SEC..62 * SEC], &[], None).remove(0)
    }

    #[test]
    fn a_segment_across_a_join_is_split_at_its_words() {
        let map = two_pieces();
        let seg = segment(
            200,
            4600,
            "Yeah. So, about the release.",
            Some(vec![
                word(200, 600, "yeah"),
                word(3100, 3400, "so"),
                word(3500, 3900, "about"),
                word(3900, 4100, "the"),
                word(4100, 4600, "release"),
            ]),
        );
        let out = map.map_segment(seg);
        let lines: Vec<(u64, u64, &str)> = out
            .iter()
            .map(|s| (s.start_ms, s.end_ms, s.text.as_str()))
            .collect();
        assert_eq!(
            lines,
            [
                (700, 1100, "Yeah."),
                (60_100, 61_600, "So, about the release.")
            ]
        );
        assert_eq!(out[1].words.as_ref().unwrap()[0].start_ms, 60_100);
        assert_eq!(out[1].speaker.as_deref(), Some("0"));

        // Token counts that disagree with the words fall back to the words.
        let seg = segment(
            200,
            4600,
            "Yeah, so about the release",
            Some(vec![
                word(200, 600, "Yeah,"),
                word(3100, 3400, "so"),
                word(3500, 3900, "it"),
            ]),
        );
        let texts: Vec<String> = map.map_segment(seg).into_iter().map(|s| s.text).collect();
        assert_eq!(texts, ["Yeah,", "so it"]);
    }

    #[test]
    fn a_segment_in_one_piece_maps_through_it() {
        let map = two_pieces();
        // Starts in the gap (a backend stretching back over silence):
        // clamped to the piece its words are in.
        let seg = segment(
            2500,
            4200,
            "about the release",
            Some(vec![word(3200, 4200, "x")]),
        );
        let out = map.map_segment(seg);
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].start_ms, out[0].end_ms), (60_000, 61_200));
        // A word in the gap goes to the nearer piece.
        let seg = segment(
            100,
            2700,
            "hi there",
            Some(vec![word(100, 500, "hi"), word(2100, 2300, "there")]),
        );
        let out = map.map_segment(seg);
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(out[0].words.as_ref().unwrap()[1].end_ms, 2500);
    }

    #[test]
    fn without_words_the_end_is_clamped_to_the_start_piece() {
        let map = two_pieces();
        let out = map.map_segment(segment(500, 4000, "yeah so", None));
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].start_ms, out[0].end_ms), (1000, 2500));
        let out = map.map_segment(segment(3500, 4000, "so", None));
        assert_eq!((out[0].start_ms, out[0].end_ms), (60_500, 61_000));
    }

    #[test]
    fn masks_round_trip_and_reject_another_length() {
        let len = 3 * SEC + 5;
        let active: Vec<bool> = (0..frame_count(len)).map(|i| i % 3 == 0).collect();
        let bytes = encode_mask(len, &active);
        assert_eq!(bytes.len(), 8 + 4);
        assert_eq!(decode_mask(&bytes, len), Some(active));
        assert_eq!(decode_mask(&bytes, len + FRAME), None);
        assert_eq!(decode_mask(&bytes[..bytes.len() - 1], len), None);
        assert_eq!(decode_mask(&[], len), None);
    }
}
