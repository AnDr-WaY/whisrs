# Nemotron diarization sidecar

NVIDIA's [Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)
model, served by [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp)
(`nemo-speech serve`), tells apart the voices of a whisrs recording on your
machine. It is the recommended diarization sidecar: a single prebuilt binary
(no Python), small, fast on a laptop CPU and faster still on a GPU through
Vulkan, and the best of those measured at counting the voices. Its server
speaks whisrs' diarization sidecar contract as it is, so there is nothing
to wrap.

- Runtime: NeMo-Speech.cpp 0.2.0, Apache-2.0 (the libraries it bundles carry
  their own licences, listed under `share/licenses/` in the tarball).
- Model: Nemotron 3 Diarization, q8_0 GGUF (107 MB), under OpenMDW-1.1. Not
  gated: no account or token needed.
- Up to 8 voices per recording.
- Trained mostly on English conversations; it also kept a German lesson as
  one voice.

## Measured on a laptop

Intel Core Ultra 9 185H, Intel Arc iGPU, 15-minute AMI meetings (three test
meetings, scored against the AMI reference) plus two real recordings.

| | CPU build | Vulkan build (Arc iGPU) |
|---|---|---|
| Time for 15 minutes of audio | 49 s (0.054 x real time, about 4 cores) | 17 s (0.019 x real time, CPU nearly idle) |
| Peak memory | 380 MB (620 MB for a 28-minute call) | 300 MB (540 MB for 28 minutes) |

Speakers found were 4, 4 and 4 for the three meetings (4 each in the
reference; the Vulkan build found 3 in the third, whose fourth person says
2 s), 1 for a lesson video with one teacher, and 2 for a 28-minute
two-person call (plus a 3-second voice, which whisrs marks minor). The
diarization error rate was 17.9 % (0.25 s collar, overlapping speech
included). It almost never mixes two people up (2 % of the speech), but it
leaves more quiet or short speech unlabelled (15 %) than the sherpa-onnx
recipe does. whisrs gives such words the nearest voice within a second, and
the rest stay under the others label. Deepgram's own speaker separation
scored 25 to 30 % on the same meetings.

The `v3-offline` preset in `diarization.yaml` matters: the default streaming
preset gives the same result but took 14 times longer on the CPU (11 minutes
for 15 minutes of audio) and 10 times longer on Vulkan.

## Install

1. The runtime. Pick the CPU build, or the Vulkan build for a GPU with a
   Vulkan driver (`vulkan-intel`, `vulkan-radeon` or the NVIDIA driver).

   ```sh
   mkdir -p ~/.local/share/nemo-speech && cd ~/.local/share/nemo-speech
   base=https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.2.0
   curl -LO "$base/nemo-speech-0.2.0-linux-x86_64-cpu.tar.gz"      # 6.7 MB
   # or: curl -LO "$base/nemo-speech-0.2.0-linux-x86_64-vulkan.tar.gz"   # 16.6 MB
   curl -LO "$base/SHA256SUMS"
   sha256sum --check --ignore-missing SHA256SUMS
   tar xzf nemo-speech-0.2.0-linux-x86_64-cpu.tar.gz
   ```

   The checksums of 0.2.0 are
   `f396057150f1b774935c7414fd32ebc04195a795a34120d78d8c9ecefa1b7507` (cpu) and
   `c8d0ed5b9ac5fb6d5c460015ba38ea217415c44edc1e6d49120646781009aee6` (vulkan).
   On a GPU, `nemo-speech-0.2.0-linux-x86_64-vulkan/bin/nemo-speech doctor`
   should list it.

2. The model, pinned to the revision measured here:

   ```sh
   cd ~/.local/share/nemo-speech
   curl -LO https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/f667ed73aee57d40cc39428eb768b4fd87a0a29e/Nemotron-3-Diarization.q8_0.gguf
   echo "08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1  Nemotron-3-Diarization.q8_0.gguf" | sha256sum --check
   ```

3. The server config, [`diarization.yaml`](diarization.yaml), with your home
   in the model path (run from this folder):

   ```sh
   mkdir -p ~/.config/nemo-speech
   sed "s|/home/YOU|$HOME|" diarization.yaml > ~/.config/nemo-speech/diarization.yaml
   ```

## Run

```sh
~/.local/share/nemo-speech/nemo-speech-0.2.0-linux-x86_64-cpu/bin/nemo-speech serve \
  --host 127.0.0.1 --port 8771 --no-ui \
  --config ~/.config/nemo-speech/diarization.yaml --device cpu
```

With the Vulkan build, use its directory and `--device vulkan:0`. Keep
`--host 127.0.0.1`: anyone who can reach the port can use the server.

To start it with your session instead, install the user unit
[`nemotron-diarization.service`](nemotron-diarization.service) (edit its
`ExecStart` for the Vulkan build):

```sh
mkdir -p ~/.config/systemd/user
cp nemotron-diarization.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now nemotron-diarization.service
```

It holds about 120 MB of memory when it starts, and some 220 MB after a
recording.

## Check it

```sh
curl -s http://127.0.0.1:8771/health
ffmpeg -loglevel error -i some-talk.m4a -t 120 -ac 1 -ar 16000 /tmp/talk.wav
curl -s -F file=@/tmp/talk.wav http://127.0.0.1:8771/v1/audio/diarizations | head -c 300
```

The answer looks like
`{"segments":[{"end":11.508999744519592,"speaker":1,"start":10.720999755248428},...]}`:
seconds into the WAV, and the voices numbered 1, 2, ... in the order they
first speak.

## whisrs config

```toml
[recording]
diarization_url = "http://127.0.0.1:8771/v1/audio/diarizations"
```

`diarization = "auto"`, the default, uses the sidecar once
`diarization_url` is set (`"sidecar"` says so explicitly). whisrs sends each
separated track's speech once, as a 16 kHz mono WAV in the `file` field,
the same way the recipe's `curl` does, and turns the numbers into its own
speaker ids.

`num_speakers` (and `--num-speakers`) has no effect here: `nemo-speech serve`
takes no speaker count and ignores the field (a count of 1, of 3 or a word
gave the very same answer), so the model always finds the voices itself.
Leave it at 0.
