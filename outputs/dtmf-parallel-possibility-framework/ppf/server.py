"""Zero-framework local UI and inference server."""

from __future__ import annotations

import argparse
import io
import json
import wave
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from .engine import load_checkpoint


class Handler(SimpleHTTPRequestHandler):
    model = None
    ui_dir = Path(__file__).resolve().parent.parent / "ui"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.ui_dir), **kwargs)

    def do_POST(self):
        if self.path != "/api/predict": self.send_error(404); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            with wave.open(io.BytesIO(self.rfile.read(length)), "rb") as wav:
                rate, channels = wav.getframerate(), wav.getnchannels()
                audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
            if channels > 1: audio = audio.reshape(-1, channels).mean(1)
            if rate != 8000:
                old = np.arange(len(audio)); new = np.linspace(0, len(audio) - 1, int(len(audio) * 8000 / rate))
                audio = np.interp(new, old, audio).astype(np.float32)
            audio = np.pad(audio[:2400], (0, max(0, 2400 - len(audio))))
            if self.model is None: raise RuntimeError("Server started without --checkpoint")
            possibilities = self.model.possibilities(torch.from_numpy(audio).unsqueeze(0), 5)[0]
            body = json.dumps({"possibilities": possibilities}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body)); self.end_headers(); self.wfile.write(body)
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode()
            self.send_response(400); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body)); self.end_headers(); self.wfile.write(body)


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", type=Path)
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)
    if a.checkpoint:
        Handler.model, _ = load_checkpoint(a.checkpoint, expected_type="ppf"); Handler.model.eval()
    print(f"DTMF PPF UI: http://{a.host}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()

