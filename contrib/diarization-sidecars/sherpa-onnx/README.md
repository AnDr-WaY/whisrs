# sherpa-onnx diarization sidecar

[`server.py`](server.py) tells apart the voices of a whisrs recording with
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), on the CPU, in a
small Python environment that `uv` builds on first run. Two ONNX models do
the work: pyannote segmentation-3.0 finds the speech and where the speaker
changes, and 3D-Speaker's ERes2Net (trained on English VoxCeleb) tells the
voices apart. In the laptop measurements below it gave more of the speech
the right voice than the [Nemotron recipe](../nemotron/), but it is 3.5
times slower than Nemotron on a CPU, and it counts the voices right only
thanks to its threshold and merge step.

- sherpa-onnx: Apache-2.0.
- pyannote segmentation-3.0, converted to ONNX by sherpa-onnx: MIT (the
  `LICENSE` in its tarball).
- 3D-Speaker ERes2Net, English VoxCeleb (`3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx`):
  Apache-2.0.
- No account, token or GPU needed. Avoid the `sherpa-onnx-reverb-diarization-v1`
  model that sits next to these downloads: its licence is non-commercial.

## Measured on a laptop

Intel Core Ultra 9 185H, 4 CPU threads, three 15-minute AMI meetings scored
against the AMI reference, plus two real recordings.

| | |
|---|---|
| Time for 15 minutes of audio | about 2 min 50 s (0.19 x real time) |
| Peak memory | 570 MB (680 MB for a 28-minute call) |
| Error rate (0.25 s collar, overlapping speech included) | 11.8 % |
| Voices found, after the merge | 4, 4 and 3 for the meetings (4 in each, but the fourth person of the third meeting says 2 s); 1 for a one-teacher lesson video; 2 for a two-person call |
| Voices found, before the merge | 8, 8 and 8; 1; 9 |

The environment takes about 100 MB (sherpa-onnx bundles onnxruntime), and the
two models 32 MB.

## Why the threshold matters

Without a speaker count, sherpa-onnx groups the voices by a distance
threshold. Its usual default, 0.5, found 60 voices in a 15-minute meeting of
four people. On a separate AMI meeting used only for tuning, 1.1 found the
voices best; 1.2 already merged four people into two. So `--threshold 1.1` is
the default here, set for this embedding model on meetings: another
embedding model needs its own value.

Even at 1.1, a long recording keeps a few small clusters: a cough, two people
at once, one voice split in pieces. Unmerged, each would show up in the
transcript as one more "Speaker N" with a line or two. So every cluster
holding under 5 % of the speech (`--min-share`), smallest first, is merged
into the voice it sounds closest to (the cosine similarity of their mean
embeddings). Only a small cluster with 10 s of speech or more
(`--keep-seconds`) that sounds unlike every other voice (similarity under
0.6, `--min-similarity`) stays a voice of its own: someone who says little.
That guard kept two people of the third meeting apart, who talk for 35 and
11 s there; merging every small cluster left that meeting with one voice
(and a worse error rate, 13.0 % against 11.8 % over the three meetings).

With `num_speakers` (whisrs' `num_speakers` or `--num-speakers N`),
sherpa-onnx groups the speech into that many voices (one can come back empty
on a short clip) and nothing is merged. In the measurements the true count
did worse than the threshold and merge (13.5 % against 11.8 %), so leave it
at 0 unless you know.

## Install

The models, with the checksums of the files measured here:

```sh
mkdir -p ~/.local/share/sherpa-onnx && cd ~/.local/share/sherpa-onnx
rel=https://github.com/k2-fsa/sherpa-onnx/releases/download
curl -LO "$rel/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
curl -LO "$rel/speaker-recongition-models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx"
sha256sum --check <<'EOF'
24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488  sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
c59158379255ad66e161679cca6af8d52d51e389e3224ab7d7a7baae295c2db5  3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx
EOF
tar xjf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
```

(`recongition` is how the sherpa-onnx release is spelled.) Then
[uv](https://docs.astral.sh/uv/), which reads the dependencies from the top of
`server.py`.

## Run

From this folder:

```sh
uv run server.py \
  --seg ~/.local/share/sherpa-onnx/sherpa-onnx-pyannote-segmentation-3-0/model.onnx \
  --emb ~/.local/share/sherpa-onnx/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx
```

It listens on `http://127.0.0.1:8770/diarize` (`--host`, `--port`) and uses 4
CPU threads (`--threads`). Keep it on 127.0.0.1: anyone who can reach the
port can use it. It answers one recording at a time and logs one line per
request: the audio length, the time taken and the voices found.

## Check it

```sh
curl -s http://127.0.0.1:8770/health
ffmpeg -loglevel error -i some-talk.m4a -t 120 -ac 1 -ar 16000 /tmp/talk.wav
curl -s -F file=@/tmp/talk.wav http://127.0.0.1:8770/diarize | head -c 300
```

Or without the server: `uv run server.py --seg ... --emb ... --input /tmp/talk.wav`
prints the same answer, and on stderr how many voices it found before and
after the merge.

## whisrs config

```toml
[recording]
diarization_url = "http://127.0.0.1:8770/diarize"
```

`diarization = "auto"`, the default, uses the sidecar once `diarization_url`
is set. whisrs waits at least 10 minutes, and twice the length of the speech
it sends, which leaves room at 0.19 x real time.
