#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sherpa-onnx>=1.13.8,<2",
#     "numpy",
#     "soundfile",
# ]
# ///
"""Speaker diarization sidecar for whisrs, on sherpa-onnx.

pyannote segmentation-3.0 finds the speech and the speaker changes, a speaker
embedding model (3D-Speaker ERes2Net by default) tells the voices apart, and
the turns go back to whisrs in its diarization sidecar contract:

    POST /diarize   multipart form: file = 16 kHz mono WAV,
                    num_speakers = N (optional, N > 0)
    200 {"segments": [{"start": 1.2, "end": 4.85, "speaker": "S0"}, ...]}
    GET  /health    {"ok": true, ...}

Without num_speakers, voices are found by a clustering threshold, which
also leaves a few small clusters on long recordings: a cough, two people at
once, one voice split in pieces. Each cluster holding less than --min-share of
the speech, smallest first, is merged into the cluster whose voice is closest
(cosine similarity of their embedding centroids). Only a small cluster with
--keep-seconds of speech or more that sounds like nobody else (below
--min-similarity) stays a voice of its own: someone who says little. A
cluster too short to embed goes to the voice next to it in time. With
num_speakers, sherpa-onnx groups the speech into that many clusters (one
can come back empty on short audio) and nothing is merged.

    uv run server.py --seg model.onnx --emb eres2net.onnx
    uv run server.py --seg model.onnx --emb eres2net.onnx --input talk.wav
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import sherpa_onnx
import soundfile as sf

RATE = 16000
# A voice's centroid: the mean embedding of its longest turns, each cut to
# PIECE_SECONDS (as long as the pipeline's own windows, so memory stays the
# same), at most PIECES of them.
PIECE_SECONDS = 8.0
PIECES = 4


class Diarizer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.sd = sherpa_onnx.OfflineSpeakerDiarization(self._config(-1))
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=args.emb, num_threads=args.threads
            )
        )
        self.lock = threading.Lock()

    def _config(self, num_clusters: int) -> sherpa_onnx.OfflineSpeakerDiarizationConfig:
        a = self.args
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=a.seg),
                num_threads=a.threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=a.emb, num_threads=a.threads
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_clusters, threshold=a.threshold
            ),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        if not config.validate():
            raise SystemExit("invalid sherpa-onnx config: check --seg and --emb")
        return config

    def diarize(self, audio: np.ndarray, num_speakers: int | None) -> tuple[list[dict], dict]:
        """Turns as {start, end, speaker} labelled S0, S1, ... by first appearance."""
        with self.lock:
            segs = self.raw(audio, num_speakers)
            stats = {"voices_found": len({s for _, _, s in segs}), "merged": {}}
            if not num_speakers and self.args.min_share > 0:
                segs, stats["merged"] = merge_minor(
                    segs,
                    lambda spk: self.centroid(audio, segs, spk),
                    self.args.min_share,
                    self.args.min_similarity,
                    self.args.keep_seconds,
                )
        segs = _union(segs)
        labels: dict[int, str] = {}
        out = [
            {
                "start": round(s, 3),
                "end": round(e, 3),
                "speaker": labels.setdefault(spk, f"S{len(labels)}"),
            }
            for s, e, spk in segs
        ]
        stats["voices"] = len(labels)
        return out, stats

    def raw(self, audio: np.ndarray, num_speakers: int | None) -> list[tuple]:
        """sherpa-onnx's own turns: (start, end, cluster)."""
        if num_speakers:
            self.sd.set_config(self._config(num_speakers))
        try:
            result = self.sd.process(audio).sort_by_start_time()
        finally:
            if num_speakers:
                self.sd.set_config(self._config(-1))
        return [(r.start, r.end, r.speaker) for r in result]

    def centroid(self, audio: np.ndarray, segs: list[tuple], spk) -> np.ndarray | None:
        """The unit-length mean embedding of a voice's longest turns."""
        mine = sorted(((e - s, s) for s, e, k in segs if k == spk), reverse=True)
        vecs = []
        for d, s in mine[:PIECES]:
            piece = audio[int(s * RATE): int((s + min(d, PIECE_SECONDS)) * RATE)]
            stream = self.extractor.create_stream()
            stream.accept_waveform(sample_rate=RATE, waveform=piece)
            stream.input_finished()
            if self.extractor.is_ready(stream):
                v = np.asarray(self.extractor.compute(stream), dtype=np.float32)
                n = float(np.linalg.norm(v))
                if n > 0:
                    vecs.append(v / n)
        if not vecs:
            return None
        m = np.mean(vecs, axis=0)
        n = float(np.linalg.norm(m))
        return m / n if n > 0 else None


def merge_minor(segs, centroid_of, min_share: float, min_similarity: float,
                keep_seconds: float):
    """Merge the small clusters (see the module doc). Returns the turns with
    their new clusters and, per merged or kept cluster, what happened."""
    talk: dict = {}
    for s, e, spk in segs:
        talk[spk] = talk.get(spk, 0.0) + (e - s)
    total = sum(talk.values())
    if total <= 0 or len(talk) < 2:
        return segs, {}
    small = sorted((k for k in talk if talk[k] / total < min_share), key=talk.get)
    if len(small) == len(talk):  # nothing is big: keep the biggest
        small.remove(max(talk, key=talk.get))
    if not small:
        return segs, {}
    cent = {k: centroid_of(k) for k in talk}
    into: dict = {}
    how: dict[str, str] = {}
    for c in small:
        live = [k for k in talk if k != c and k not in into]
        own = talk[c]
        scored = [(float(np.dot(cent[c], cent[k])), k) for k in live
                  if cent[c] is not None and cent[k] is not None]
        if scored:
            sim, target = max(scored)
            if sim < min_similarity and own >= keep_seconds:
                how[str(c)] = f"kept: closest voice {target} at cosine {sim:.2f}, {own:.1f} s"
                continue
            why = f"cosine {sim:.2f}"
            if cent[target] is not None:
                w_t, w_c = min(talk[target], PIECES * PIECE_SECONDS), min(own, PIECES * PIECE_SECONDS)
                v = w_t * cent[target] + w_c * cent[c]
                n = float(np.linalg.norm(v))
                cent[target] = v / n if n > 0 else cent[target]
        else:
            target, why = _time_neighbour(segs, c, live), "next in time"
        into[c] = target
        talk[target] += own
        how[str(c)] = f"-> {target} ({why}, {own:.1f} s)"

    def root(k):
        while k in into:
            k = into[k]
        return k

    return [(s, e, root(spk)) for s, e, spk in segs], how


def _time_neighbour(segs, spk, live):
    """The live voice whose turn comes nearest in time to one of spk's turns."""
    best = (float("inf"), live[0])
    for s, e, k in segs:
        if k != spk:
            continue
        for s2, e2, k2 in segs:
            if k2 in live:
                gap = max(s2 - e, s - e2, 0.0)
                best = min(best, (gap, k2))
    return best[1]


def _union(segs):
    """Join a voice's turns that touch or overlap (after a merge)."""
    out: list[list] = []
    last: dict[int, list] = {}
    for s, e, spk in sorted(segs):
        prev = last.get(spk)
        if prev is not None and s <= prev[1]:
            prev[1] = max(prev[1], e)
            continue
        cur = [s, e, spk]
        out.append(cur)
        last[spk] = cur
    return [tuple(x) for x in out]


def read_wav(data: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if sr != RATE:
        raise ValueError(f"expected a {RATE} Hz WAV, got {sr} Hz")
    return np.ascontiguousarray(audio[:, 0])


def make_handler(diarizer: Diarizer, args: argparse.Namespace):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                return self._json(200, {"ok": True, "threshold": args.threshold,
                                        "min_share": args.min_share})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path.split("?")[0].rstrip("/") != "/diarize":
                return self._json(404, {"error": "POST /diarize"})
            length = int(self.headers.get("content-length") or 0)
            if length <= 0 or length > args.max_upload_mb * 1024 * 1024:
                return self._json(413 if length > 0 else 411, {"error": "bad length"})
            body = self.rfile.read(length)
            ctype = self.headers.get("content-type", "")
            if not ctype.startswith("multipart/form-data"):
                return self._json(400, {"error": "send multipart/form-data with a file part"})
            msg = BytesParser(policy=HTTP).parsebytes(
                b"content-type: " + ctype.encode() + b"\r\n\r\n" + body
            )
            wav, num = None, None
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if name == "file":
                    wav = part.get_payload(decode=True)
                elif name == "num_speakers":
                    try:
                        num = int(part.get_payload(decode=True).decode().strip() or 0) or None
                    except ValueError:
                        return self._json(400, {"error": "num_speakers must be an integer"})
            if not wav:
                return self._json(400, {"error": "missing the file part"})
            try:
                audio = read_wav(wav)
            except Exception as e:  # noqa: BLE001 - any unreadable upload
                return self._json(400, {"error": f"bad WAV: {e}"})
            started = time.monotonic()
            segments, stats = diarizer.diarize(audio, num)
            took = time.monotonic() - started
            print(f"{len(audio) / RATE:.0f} s of audio in {took:.1f} s: {stats['voices']} voice(s)"
                  f" ({stats['voices_found']} found{', asked for ' + str(num) if num else ''})",
                  file=sys.stderr, flush=True)
            self._json(200, {"segments": segments})

        def log_message(self, fmt, *a):
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seg", required=True, help="pyannote segmentation-3.0 model.onnx")
    ap.add_argument("--emb", required=True, help="speaker embedding .onnx (ERes2Net)")
    ap.add_argument("--threshold", type=float, default=1.1,
                    help="clustering threshold without num_speakers: higher finds fewer "
                         "voices (default 1.1, tuned for ERes2Net on meetings)")
    ap.add_argument("--min-share", type=float, default=0.05,
                    help="a voice holding less than this share of the speech is merged "
                         "into the closest other voice (default 0.05; 0 keeps all)")
    ap.add_argument("--min-similarity", type=float, default=0.6,
                    help="a small voice with --keep-seconds of speech or more stays apart "
                         "when its cosine similarity to every other voice is below this "
                         "(default 0.6)")
    ap.add_argument("--keep-seconds", type=float, default=10.0,
                    help="small voices with less speech than this are always merged "
                         "(default 10)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--max-upload-mb", type=int, default=512)
    ap.add_argument("--input", help="diarize this WAV once, print the JSON answer and exit")
    ap.add_argument("--num-speakers", type=int, default=0, help="with --input")
    args = ap.parse_args()

    diarizer = Diarizer(args)
    if args.input:
        with open(args.input, "rb") as f:
            audio = read_wav(f.read())
        started = time.monotonic()
        segments, stats = diarizer.diarize(audio, args.num_speakers or None)
        stats["seconds"] = round(time.monotonic() - started, 1)
        stats["audio_seconds"] = round(len(audio) / RATE, 1)
        print(json.dumps(stats), file=sys.stderr)
        print(json.dumps({"segments": segments}))
        return
    server = ThreadingHTTPServer((args.host, args.port), make_handler(diarizer, args))
    print(f"listening on http://{args.host}:{args.port}/diarize", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
