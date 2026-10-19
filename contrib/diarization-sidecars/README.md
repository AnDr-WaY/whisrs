# Diarization sidecars

A diarization sidecar is a local HTTP server that only finds who speaks
when. whisrs asks it about each track of a recording
(`[recording] diarization_url`) while any backend transcribes the words, then
gives every word the voice speaking at that moment.

| Recipe | What runs | 15 minutes of audio on a laptop | Voices found | Error rate |
|---|---|---|---|---|
| [`nemotron`](nemotron/) (recommended) | `nemo-speech serve`, one binary | 49 s on the CPU, 17 s on Vulkan | 4, 4, 4; 1; 2 | 17.9 % |
| [`sherpa-onnx`](sherpa-onnx/) | `uv run server.py` | 170 s on 4 CPU threads | 4, 4, 3; 1; 2 | 11.8 % |

Measured on an Intel Core Ultra 9 185H on three 15-minute AMI meetings
(error rate with a 0.25 s collar, overlapping speech included), a lesson
video with one teacher and a two-person call. The meetings have 4 people
each, but the fourth person of the third one says 2 s. Nemotron also gave
the call a third voice of 3 s, which whisrs marks minor. Nemotron mixes up
voices least but leaves more quiet speech unlabelled; sherpa-onnx labels
more of the speech but needs its threshold and merge step to count the
voices. Deepgram's speaker separation, which comes free with its
transcription, scored 25 to 30 % on the same meetings.

## Contract

```text
POST {diarization_url}   multipart form: file = the track's speech as a 16 kHz mono WAV,
                         num_speakers = N (only when N > 0)
200 {"segments": [{"start": 1.20, "end": 4.85, "speaker": "SPEAKER_00"}]}
```

`start` and `end` are seconds into the WAV; `speaker` is a string or an
integer that only has to be consistent within one request. See "A
diarization sidecar" in [docs/configuration.md](../../docs/configuration.md)
for the timeouts, retries and caching.
