//! Telling voices apart with a diarization sidecar
//! (`[recording] diarization_url`): a local HTTP server that only finds who
//! speaks when, while any backend transcribes the words.
//!
//! The contract:
//!
//! ```text
//! POST {diarization_url}   multipart: file=<16 kHz mono WAV>,
//!                          num_speakers (only when > 0)
//! 200 {"segments": [{"start": 1.20, "end": 4.85, "speaker": "SPEAKER_00"}]}
//! ```
//!
//! Times are seconds into the WAV; `speaker` is a string or an integer
//! (NeMo-Speech.cpp's `nemo-speech serve` numbers its voices 1, 2, ...), and
//! only has to be consistent within one request. A track goes in one request
//! (its labels must hold across the whole track), as its speech only,
//! condensed the way transcript chunks are ([`super::speech`]); the caller
//! maps the turns back onto the track. A failure is the caller's to absorb:
//! speakers are never worth a recording.

use std::time::Duration;

use anyhow::Context;
use bytes::Bytes;
use reqwest::multipart;
use serde::Deserialize;
use tracing::debug;

use super::speakers::Turn;
use crate::transcription::retry::{
    final_if_crashed, is_network_unavailable, retry_with_backoff, NETWORK_ERROR_BACKOFF,
    RATE_LIMIT_BUDGET, SERVER_ERROR_BACKOFF,
};
use crate::transcription::{http_error, retry_after_header, secs_to_ms};

/// The shortest wait for an answer. A sidecar on a laptop CPU takes about
/// as long as the audio, so the wait grows with it ([`timeout_for`]).
pub const MIN_TIMEOUT: Duration = Duration::from_secs(10 * 60);

/// Bytes of a 16 kHz mono 16-bit WAV per second of audio.
const WAV_BYTES_PER_SEC: usize = 32_000;

/// How long the sidecar may take per second of audio ([`timeout_for`]).
pub const TIMEOUT_PER_AUDIO_SECOND: f64 = 2.0;

/// How long to wait for the sidecar to answer for `audio_secs` of audio:
/// `per_second` times the audio, and never less than `min`.
pub fn timeout_for(audio_secs: f64, per_second: f64, min: Duration) -> Duration {
    let scaled = Duration::from_secs_f64((audio_secs * per_second).max(0.0));
    scaled.max(min)
}

/// The waits of a [`DiarizationSidecar`]: the real ones by default; tests
/// shorten them.
#[derive(Debug, Clone, Copy)]
pub struct Timing {
    /// The shortest answer timeout ([`timeout_for`]).
    pub min_timeout: Duration,
    /// The answer timeout per second of audio ([`TIMEOUT_PER_AUDIO_SECOND`]).
    pub per_audio_second: f64,
    /// The waits before retrying a gateway error (502, 504, 520 to 524).
    pub server_backoff: &'static [Duration],
    /// The waits before retrying a request that got no answer at all.
    pub network_backoff: &'static [Duration],
}

impl Default for Timing {
    fn default() -> Self {
        Self {
            min_timeout: MIN_TIMEOUT,
            per_audio_second: TIMEOUT_PER_AUDIO_SECOND,
            server_backoff: &SERVER_ERROR_BACKOFF,
            network_backoff: &NETWORK_ERROR_BACKOFF,
        }
    }
}

/// A diarization sidecar at a URL.
pub struct DiarizationSidecar {
    client: reqwest::Client,
    url: String,
    timing: Timing,
}

impl DiarizationSidecar {
    pub fn new(url: &str, timing: Timing) -> Self {
        Self {
            client: reqwest::Client::new(),
            url: url.trim().to_string(),
            timing,
        }
    }

    /// The URL it posts to.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// The turns of each voice in `wav` (a 16 kHz mono WAV), in its own
    /// time. `num_speakers` > 0 tells the sidecar how many voices to find.
    ///
    /// Retried like a transcript chunk: a sidecar that cannot be reached
    /// (not running yet, the network) or a gateway error in front of it is
    /// tried again a few times. Its own HTTP 500, a crash on the request
    /// ([`final_if_crashed`]) and a timeout are final: the same audio would
    /// fail or take as long again.
    pub async fn diarize(&self, wav: Vec<u8>, num_speakers: u32) -> anyhow::Result<Vec<Turn>> {
        let secs = wav.len().saturating_sub(44) as f64 / WAV_BYTES_PER_SEC as f64;
        let timeout = timeout_for(secs, self.timing.per_audio_second, self.timing.min_timeout);
        debug!(
            "sending {:.1} s of speech to the diarization sidecar (num_speakers {num_speakers}, \
             timeout {} s)",
            secs,
            timeout.as_secs()
        );
        let wav = Bytes::from(wav);
        let body = retry_with_backoff(
            RATE_LIMIT_BUDGET,
            self.timing.server_backoff,
            self.timing.network_backoff,
            || self.post(wav.clone(), num_speakers, timeout),
        )
        .await
        .map_err(|err| {
            if is_network_unavailable(&err) {
                err.context("the diarization sidecar could not be reached")
            } else {
                err
            }
        })?;
        parse_turns(&body)
    }

    async fn post(
        &self,
        wav: Bytes,
        num_speakers: u32,
        timeout: Duration,
    ) -> anyhow::Result<String> {
        if self.url.is_empty() {
            anyhow::bail!("no diarization_url configured");
        }
        let length = wav.len() as u64;
        let file = multipart::Part::stream_with_length(reqwest::Body::from(wav), length)
            .file_name("audio.wav")
            .mime_str("audio/wav")?;
        let mut form = multipart::Form::new().part("file", file);
        if num_speakers > 0 {
            form = form.text("num_speakers", num_speakers.to_string());
        }
        // Without the URL, which can hold a token: the error reaches
        // recording.json, the transcript's notes and a notification.
        let error = |e: reqwest::Error| -> anyhow::Error {
            let e = e.without_url();
            if e.is_timeout() {
                anyhow::anyhow!(
                    "the diarization sidecar gave no answer within {} s",
                    timeout.as_secs()
                )
            } else {
                final_if_crashed(e, "the diarization sidecar")
            }
        };
        let response = self
            .client
            .post(&self.url)
            .timeout(timeout)
            .multipart(form)
            .send()
            .await
            .map_err(error)?;
        let status = response.status();
        let retry_after = retry_after_header(response.headers());
        let body = response.text().await.map_err(error)?;
        if !status.is_success() {
            let code = status.as_u16();
            let detail: String = body.trim().chars().take(300).collect();
            let message = format!("diarization sidecar error ({code}): {detail}");
            // Its own 500 is an error in the sidecar, which the same audio
            // would hit again; a gateway in front of it may pass.
            if code == 500 {
                return Err(anyhow::anyhow!(message));
            }
            return Err(http_error(code, retry_after, message));
        }
        Ok(body)
    }
}

#[derive(Debug, Deserialize)]
struct TurnsResponse {
    segments: Vec<RawTurn>,
}

#[derive(Debug, Deserialize)]
struct RawTurn {
    start: f64,
    end: f64,
    speaker: serde_json::Value,
}

/// Parse the sidecar's answer into turns in milliseconds, in order of
/// start. A turn without a usable speaker label is an error: an answer that
/// does not follow the contract is not guessed at.
pub fn parse_turns(body: &str) -> anyhow::Result<Vec<Turn>> {
    let parsed: TurnsResponse = serde_json::from_str(body).with_context(|| {
        let start: String = body.trim().chars().take(120).collect();
        format!("the diarization sidecar's answer is not {{\"segments\": [...]}}: {start}")
    })?;
    let mut turns = parsed
        .segments
        .into_iter()
        .map(|t| {
            let speaker = match &t.speaker {
                serde_json::Value::String(s) if !s.trim().is_empty() => s.trim().to_string(),
                serde_json::Value::Number(n) => integer_label(n).ok_or_else(|| {
                    anyhow::anyhow!("the diarization sidecar sent a speaker that is not an id: {n}")
                })?,
                other => {
                    anyhow::bail!("the diarization sidecar sent a turn without a speaker: {other}")
                }
            };
            let start_ms = secs_to_ms(t.start);
            Ok(Turn {
                start_ms,
                end_ms: secs_to_ms(t.end).max(start_ms),
                speaker,
            })
        })
        .collect::<anyhow::Result<Vec<Turn>>>()?;
    turns.sort_by_key(|t| (t.start_ms, t.end_ms));
    Ok(turns)
}

/// A numeric speaker id as a label: any JSON integer as written (`1`, `-1`),
/// and a float that holds one (`2.0`, as a sidecar built on numpy can send)
/// as that integer, so `2` and `2.0` are one voice. A fraction is not an id.
fn integer_label(n: &serde_json::Number) -> Option<String> {
    if n.is_i64() || n.is_u64() {
        return Some(n.to_string());
    }
    let f = n.as_f64()?;
    // Below 2^53 every integer is exact in an f64, and the cast is lossless.
    (f.fract() == 0.0 && f.abs() < 9_007_199_254_740_992.0).then(|| (f as i64).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcription::test_server::{
        accept_request, crash_after_request, serve_once, serve_responses,
    };
    use tokio::net::TcpListener;

    const TWO_VOICES: &str = r#"{"segments": [
        {"start": 1.2, "end": 4.85, "speaker": "SPEAKER_00"},
        {"start": 0.0, "end": 1.0, "speaker": 1}
    ]}"#;

    /// Short waits, so retries and timeouts run in a test's time.
    const QUICK: Timing = Timing {
        min_timeout: Duration::from_millis(500),
        per_audio_second: 0.0,
        server_backoff: &[Duration::from_millis(1); 3],
        network_backoff: &[Duration::from_millis(1); 4],
    };

    fn wav(secs: usize) -> Vec<u8> {
        crate::audio::wav::encode_wav(&vec![300i16; secs * 16_000]).unwrap()
    }

    async fn local() -> (TcpListener, DiarizationSidecar) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/diarize", listener.local_addr().unwrap());
        (listener, DiarizationSidecar::new(&url, QUICK))
    }

    #[test]
    fn turns_parse_in_order_with_string_or_number_speakers() {
        assert_eq!(
            parse_turns(TWO_VOICES).unwrap(),
            [
                Turn {
                    start_ms: 0,
                    end_ms: 1000,
                    speaker: "1".into()
                },
                Turn {
                    start_ms: 1200,
                    end_ms: 4850,
                    speaker: "SPEAKER_00".into()
                },
            ]
        );
        assert!(parse_turns(r#"{"segments": []}"#).unwrap().is_empty());
        for bad in [
            "not json",
            r#"{"text": "hello"}"#,
            r#"{"segments": [{"start": 0, "end": 1}]}"#,
            r#"{"segments": [{"start": 0, "end": 1, "speaker": null}]}"#,
        ] {
            assert!(parse_turns(bad).is_err(), "{bad}");
        }
    }

    /// The first six turns `nemo-speech serve` (NeMo-Speech.cpp 0.2.0,
    /// Nemotron-3-Diarization) answered for an AMI meeting, verbatim: voices
    /// as 1-based integers, times with float noise, keys in its own order.
    const NEMO_SPEECH_SERVE: &str = r#"{"segments":[{"end":11.508999744519592,"speaker":1,"start":10.720999755248428},{"end":15.358999658465386,"speaker":1,"start":11.900999728873373},{"end":18.268999593421817,"speaker":2,"start":17.650999600350858},{"end":20.358999546706677,"speaker":1,"start":18.660999577775598},{"end":23.828999469146133,"speaker":1,"start":22.120999500438572},{"end":26.618999406784773,"speaker":3,"start":24.920999437853695}]}"#;

    #[test]
    fn nemo_speech_serve_answers_parse_with_integer_voices() {
        let turns = parse_turns(NEMO_SPEECH_SERVE).unwrap();
        let got: Vec<(u64, u64, &str)> = turns
            .iter()
            .map(|t| (t.start_ms, t.end_ms, t.speaker.as_str()))
            .collect();
        assert_eq!(
            got,
            [
                (10_721, 11_509, "1"),
                (11_901, 15_359, "1"),
                (17_651, 18_269, "2"),
                (18_661, 20_359, "1"),
                (22_121, 23_829, "1"),
                (24_921, 26_619, "3"),
            ]
        );
    }

    /// The whole client against a server answering as `nemo-speech serve`
    /// does: three voices come back, labelled by their numbers.
    #[tokio::test]
    async fn a_nemo_speech_serve_answer_comes_back_as_turns() {
        let (listener, sidecar) = local().await;
        let server = tokio::spawn(serve_once(listener, NEMO_SPEECH_SERVE));
        let turns = sidecar.diarize(wav(1), 0).await.unwrap();
        server.await.unwrap();
        let mut voices: Vec<&str> = turns.iter().map(|t| t.speaker.as_str()).collect();
        voices.dedup();
        assert_eq!(voices, ["1", "2", "1", "3"]);
    }

    /// Any JSON integer is a voice id, and a float holding one is the same
    /// voice; a fraction is no id, and the answer is refused.
    #[test]
    fn numeric_voices_are_integers() {
        let label = |speaker: &str| {
            parse_turns(&format!(
                r#"{{"segments": [{{"start": 0, "end": 1, "speaker": {speaker}}}]}}"#
            ))
            .map(|turns| turns[0].speaker.clone())
        };
        assert_eq!(label("0").unwrap(), "0");
        assert_eq!(label("-1").unwrap(), "-1");
        assert_eq!(
            label("18446744073709551615").unwrap(),
            "18446744073709551615"
        );
        assert_eq!(label("2.0").unwrap(), "2", "2.0 is voice 2");
        assert_eq!(label("1e1").unwrap(), "10");
        for bad in ["1.5", "1e300"] {
            let err = label(bad).unwrap_err();
            assert!(format!("{err:#}").contains("not an id"), "{bad}: {err:#}");
        }
    }

    #[test]
    fn the_timeout_grows_with_the_audio() {
        let real = Timing::default();
        let timeout = |secs| timeout_for(secs, real.per_audio_second, real.min_timeout);
        assert_eq!(
            timeout(60.0),
            Duration::from_secs(600),
            "ten minutes at least"
        );
        assert_eq!(
            timeout(3600.0),
            Duration::from_secs(7200),
            "an hour of speech gets two"
        );
        assert_eq!(timeout(301.0), Duration::from_secs(602));
    }

    /// What goes on the wire: the WAV as `file`, and `num_speakers` only
    /// when there is a number to give.
    #[tokio::test]
    async fn the_request_follows_the_contract() {
        let (listener, sidecar) = local().await;
        let server = tokio::spawn(serve_once(listener, TWO_VOICES));
        let audio = wav(2);
        let turns = sidecar.diarize(audio.clone(), 3).await.unwrap();
        assert_eq!(turns.len(), 2);
        let request = server.await.unwrap();
        assert!(
            request.head.starts_with("POST /diarize "),
            "{}",
            request.head
        );
        let (headers, data) = request.file_part();
        assert!(headers.contains("filename=\"audio.wav\""), "{headers}");
        assert!(headers.contains("audio/wav"), "{headers}");
        assert!(data.starts_with(&audio), "the WAV as sent");
        let body = String::from_utf8_lossy(&request.body);
        assert!(
            body.contains("name=\"num_speakers\"\r\n\r\n3\r\n"),
            "{body}"
        );

        let (listener, sidecar) = local().await;
        let server = tokio::spawn(serve_once(listener, TWO_VOICES));
        sidecar.diarize(wav(1), 0).await.unwrap();
        let body = String::from_utf8_lossy(&server.await.unwrap().body).into_owned();
        assert!(!body.contains("num_speakers"), "0 lets the sidecar decide");
    }

    /// Its own 500 is final (one request), a gateway's 502 is retried.
    #[tokio::test]
    async fn a_500_is_final_and_a_gateway_error_is_retried() {
        let (listener, sidecar) = local().await;
        let server = tokio::spawn(serve_responses(
            listener,
            vec![(500, r#"{"detail": "CUDA out of memory"}"#)],
        ));
        let err = sidecar.diarize(wav(1), 0).await.unwrap_err();
        assert!(format!("{err:#}").contains("(500)"), "{err:#}");
        assert_eq!(server.await.unwrap().len(), 1);

        let (listener, sidecar) = local().await;
        let server = tokio::spawn(serve_responses(
            listener,
            vec![(502, "bad gateway"), (200, TWO_VOICES)],
        ));
        assert_eq!(sidecar.diarize(wav(1), 0).await.unwrap().len(), 2);
        assert_eq!(server.await.unwrap().len(), 2);
    }

    /// No answer within the timeout is final: the same audio would take as
    /// long again.
    #[tokio::test]
    async fn a_timeout_is_final() {
        let (listener, sidecar) = local().await;
        let server = tokio::spawn(async move {
            let (stream, _) = accept_request(&listener).await;
            // Hold the connection open, unanswered, past the timeout.
            tokio::time::sleep(Duration::from_secs(3)).await;
            drop(stream);
        });
        let started = std::time::Instant::now();
        let err = sidecar.diarize(wav(1), 0).await.unwrap_err();
        assert!(
            format!("{err:#}").contains("gave no answer within"),
            "{err:#}"
        );
        assert!(started.elapsed() < Duration::from_secs(2), "not retried");
        server.abort();
    }

    /// A sidecar that is not running is retried, then given up on as no
    /// network; one that crashed on the request is not sent it again.
    #[tokio::test]
    async fn an_unreachable_sidecar_is_retried_and_a_crash_is_final() {
        let (listener, sidecar) = local().await;
        drop(listener);
        let err = sidecar.diarize(wav(1), 0).await.unwrap_err();
        assert!(is_network_unavailable(&err), "{err:#}");
        assert!(
            format!("{err:#}").starts_with("the diarization sidecar could not be reached: "),
            "{err:#}"
        );

        let (listener, sidecar) = local().await;
        let server = tokio::spawn(async move { crash_after_request(&listener).await });
        let err = sidecar.diarize(wav(1), 0).await.unwrap_err();
        assert!(format!("{err:#}").contains("likely crashed"), "{err:#}");
        server.await.unwrap();
    }

    /// The sidecar's URL, which can hold a token, stays out of its errors:
    /// they reach recording.json, the transcript and a notification. A
    /// crash on the request is the error that would carry it.
    #[tokio::test]
    async fn errors_leave_the_url_out() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!(
            "http://{}/diarize?token=s3cret",
            listener.local_addr().unwrap()
        );
        let server = tokio::spawn(async move { crash_after_request(&listener).await });
        let err = DiarizationSidecar::new(&url, QUICK)
            .diarize(wav(1), 0)
            .await
            .unwrap_err();
        assert!(format!("{err:#}").contains("likely crashed"), "{err:#}");
        assert!(!format!("{err:#}").contains("s3cret"), "{err:#}");
        assert!(!format!("{err:?}").contains("s3cret"), "{err:?}");
        server.await.unwrap();
    }

    #[tokio::test]
    async fn a_bad_answer_is_an_error() {
        let (listener, sidecar) = local().await;
        let server = tokio::spawn(serve_once(listener, r#"{"text": "hello"}"#));
        let err = sidecar.diarize(wav(1), 0).await.unwrap_err();
        assert!(format!("{err:#}").contains("segments"), "{err:#}");
        server.await.unwrap();
    }
}
