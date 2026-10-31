from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import ModelConfig
from .data import DTMF_FREQUENCIES, load_manifest, read_wav
from .model import CNNBaseline, ParallelPossibilityFramework


class ToneEventDataset(Dataset):
    """Turns sequence WAVs into ground-truth digit events."""
    def __init__(self, manifest: Path, split="train", samples=2400):
        self.root, self.samples = Path(manifest).parent, samples
        self.items = []
        for row in load_manifest(manifest, split):
            for event in row["events"]:
                self.items.append((row["path"], event))

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        rel, event = self.items[index]
        audio, _ = read_wav(self.root / rel)
        segment = audio[event["start_sample"]:event["end_sample"]]
        if len(segment) < self.samples:
            segment = np.pad(segment, (0, self.samples - len(segment)))
        else:
            segment = segment[:self.samples]
        return torch.from_numpy(segment.copy()), int(event["digit"])


def save_checkpoint(path, model, cfg, epoch=0, metrics=None, model_type="ppf"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                "epoch": epoch, "metrics": metrics or {}, "model_type": model_type}, path)


def load_checkpoint(path, device="cpu", expected_type=None):
    payload = torch.load(path, map_location=device, weights_only=True)
    cfg = ModelConfig.from_dict(payload["config"])
    model_type = payload.get("model_type", "ppf")
    if expected_type and model_type != expected_type:
        raise ValueError(f"Expected a {expected_type} checkpoint, got {model_type}")
    model = (CNNBaseline(cfg) if model_type == "cnn" else ParallelPossibilityFramework(cfg)).to(device)
    model.load_state_dict(payload["model"])
    return model, payload


def train(manifest, output, epochs=5, batch_size=32, lr=3e-4,
          device="cpu", seed=7, baseline=False):
    torch.manual_seed(seed)
    cfg = ModelConfig()
    model = (CNNBaseline(cfg) if baseline else ParallelPossibilityFramework(cfg)).to(device)
    loader = DataLoader(ToneEventDataset(Path(manifest), "train"), batch_size=batch_size,
                        shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); total = correct = seen = 0
        for audio, labels in loader:
            audio, labels = audio.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            if baseline:
                logits = model(audio); loss = F.cross_entropy(logits, labels)
            else:
                result = model(audio, labels); logits = result["digit_logits"]
                loss = result["losses"]["total"]
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * len(labels); seen += len(labels)
            correct += int((logits.argmax(-1) == labels).sum())
        row = {"epoch": epoch, "loss": total / max(seen, 1),
               "train_accuracy": correct / max(seen, 1)}
        history.append(row); print(json.dumps(row), flush=True)
    save_checkpoint(output, model, cfg, epochs, history[-1] if history else {},
                    "cnn" if baseline else "ppf")
    return history


@torch.no_grad()
def evaluate_neural(model, dataset, batch_size=32, device="cpu", top_k=3):
    model.eval(); correct1 = correctk = count = nll = 0.0
    for audio, labels in DataLoader(dataset, batch_size=batch_size):
        audio, labels = audio.to(device), labels.to(device)
        if isinstance(model, ParallelPossibilityFramework):
            generated, _ = model.predict_latent(audio)
            logits = model.decoder(generated)
        else:
            logits = model(audio)
        probs = logits.softmax(-1)
        top = probs.topk(top_k, dim=-1).indices
        correct1 += float((top[:, 0] == labels).sum())
        correctk += float((top == labels[:, None]).any(-1).sum())
        nll += float(F.nll_loss(probs.clamp_min(1e-8).log(), labels, reduction="sum"))
        count += len(labels)
    return {"samples": int(count), "top1_accuracy": correct1 / max(count, 1),
            f"top{top_k}_accuracy": correctk / max(count, 1), "nll": nll / max(count, 1)}


def goertzel_power(audio, frequency, sample_rate=8000):
    n = len(audio); k = int(0.5 + n * frequency / sample_rate)
    omega = 2 * np.pi * k / n; coeff = 2 * np.cos(omega)
    q0 = q1 = q2 = 0.0
    for sample in audio:
        q0 = coeff * q1 - q2 + float(sample); q2, q1 = q1, q0
    return q1 * q1 + q2 * q2 - coeff * q1 * q2


def goertzel_predict(audio):
    scores = []
    for digit, (low, high) in DTMF_FREQUENCIES.items():
        scores.append((goertzel_power(audio, low) * goertzel_power(audio, high), int(digit)))
    return max(scores)[1]


def evaluate_goertzel(dataset):
    correct = 0
    for audio, label in dataset:
        correct += goertzel_predict(audio.numpy()) == label
    return {"samples": len(dataset), "top1_accuracy": correct / max(len(dataset), 1),
            "baseline": "Goertzel frequency detector"}

