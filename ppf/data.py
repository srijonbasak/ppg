"""Deterministic synthetic DTMF data generation and loading."""

from __future__ import annotations

import json
import math
import random
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DTMF_FREQUENCIES = {
    "1": (697, 1209), "2": (697, 1336), "3": (697, 1477),
    "4": (770, 1209), "5": (770, 1336), "6": (770, 1477),
    "7": (852, 1209), "8": (852, 1336), "9": (852, 1477),
    "0": (941, 1336),
}


@dataclass
class SynthesisConfig:
    sample_rate: int = 8000
    tone_ms: tuple[int, int] = (70, 260)
    gap_ms: tuple[int, int] = (15, 100)
    snr_db: tuple[float, float] = (4.0, 35.0)
    max_sequence_length: int = 8


def _fade(signal: np.ndarray, sample_rate: int, ms: float = 5.0) -> np.ndarray:
    count = min(len(signal) // 2, max(1, int(sample_rate * ms / 1000)))
    ramp = np.linspace(0, 1, count, dtype=np.float32)
    signal[:count] *= ramp
    signal[-count:] *= ramp[::-1]
    return signal


def synthesize_sequence(sequence: str, rng: np.random.Generator,
                        cfg: SynthesisConfig | None = None,
                        corruption: bool = True) -> tuple[np.ndarray, dict]:
    cfg = cfg or SynthesisConfig()
    pieces, events = [], []
    cursor = 0
    for digit in sequence:
        low, high = DTMF_FREQUENCIES[digit]
        tone_ms = int(rng.integers(cfg.tone_ms[0], cfg.tone_ms[1] + 1))
        gap_ms = int(rng.integers(cfg.gap_ms[0], cfg.gap_ms[1] + 1))
        n = max(1, int(cfg.sample_rate * tone_ms / 1000))
        t = np.arange(n, dtype=np.float32) / cfg.sample_rate
        jitter = float(rng.normal(0, 0.006)) if corruption else 0.0
        twist_db = float(rng.uniform(-6, 6)) if corruption else 0.0
        a_low = 10 ** (twist_db / 40)
        a_high = 10 ** (-twist_db / 40)
        phase_a, phase_b = rng.uniform(0, 2 * math.pi, size=2)
        tone = a_low * np.sin(2 * math.pi * low * (1 + jitter) * t + phase_a)
        tone += a_high * np.sin(2 * math.pi * high * (1 + jitter) * t + phase_b)
        tone = _fade((0.36 * tone).astype(np.float32), cfg.sample_rate)
        pieces.append(tone)
        events.append({"digit": int(digit), "start_sample": cursor,
                       "end_sample": cursor + n, "tone_ms": tone_ms,
                       "gap_ms": gap_ms, "low_hz": low, "high_hz": high,
                       "frequency_jitter": jitter, "twist_db": twist_db})
        cursor += n
        gap_n = int(cfg.sample_rate * gap_ms / 1000)
        pieces.append(np.zeros(gap_n, dtype=np.float32))
        cursor += gap_n
    audio = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    metadata = {"sequence": sequence, "sample_rate": cfg.sample_rate, "events": events}
    if corruption:
        clean_rms = float(np.sqrt(np.mean(audio ** 2) + 1e-9))
        snr = float(rng.uniform(*cfg.snr_db))
        noise_rms = clean_rms / (10 ** (snr / 20))
        noise = rng.normal(0, noise_rms, len(audio)).astype(np.float32)
        # Add low-frequency hum and occasional impulsive clicks/dropout.
        hum_hz = float(rng.choice([50.0, 60.0]))
        hum = np.sin(2 * math.pi * hum_hz * np.arange(len(audio)) / cfg.sample_rate)
        audio = audio + noise + (noise_rms * 0.35 * hum).astype(np.float32)
        if rng.random() < 0.25 and len(audio) > 32:
            pos = int(rng.integers(8, len(audio) - 8))
            audio[pos:pos + 4] += rng.uniform(-0.8, 0.8)
            metadata["click_sample"] = pos
        if rng.random() < 0.20 and len(audio) > 80:
            width = int(rng.integers(8, min(80, len(audio))))
            pos = int(rng.integers(0, len(audio) - width + 1))
            audio[pos:pos + width] *= float(rng.uniform(0, 0.15))
            metadata["dropout"] = [pos, pos + width]
        drive = float(rng.uniform(0.9, 1.8))
        audio = np.tanh(audio * drive) / max(drive, 1.0)
        metadata.update({"snr_db": snr, "drive": drive})
    peak = max(float(np.max(np.abs(audio))), 1e-6)
    return np.clip(audio / max(1.0, peak / 0.98), -1, 1).astype(np.float32), metadata


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        rate, channels, width = handle.getframerate(), handle.getnchannels(), handle.getsampwidth()
        if width != 2:
            raise ValueError("Only 16-bit PCM WAV files are supported")
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return (pcm.astype(np.float32) / 32768.0), rate


def generate_dataset(output: Path, count: int, seed: int = 7,
                     max_sequence_length: int = 8) -> Path:
    output = Path(output)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    cfg = SynthesisConfig(max_sequence_length=max_sequence_length)
    records = []
    split_cutoffs = (int(count * 0.8), int(count * 0.9))
    for idx in range(count):
        length = int(rng.integers(1, max_sequence_length + 1))
        sequence = "".join(py_rng.choice("0123456789") for _ in range(length))
        audio, meta = synthesize_sequence(sequence, rng, cfg, corruption=True)
        split = "train" if idx < split_cutoffs[0] else "val" if idx < split_cutoffs[1] else "test"
        rel = Path(split) / f"sample_{idx:06d}.wav"
        write_wav(output / rel, audio, cfg.sample_rate)
        records.append({"id": idx, "path": str(rel), "split": split, **meta})
    manifest = output / "manifest.jsonl"
    output.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return manifest


def load_manifest(path: Path, split: str | None = None) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [row for row in rows if split is None or row["split"] == split]

