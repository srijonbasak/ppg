from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


def _device(value):
    import torch
    if value == "auto": return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def generate_main(argv=None):
    from .data import generate_dataset
    p = argparse.ArgumentParser(description="Generate labeled corrupted DTMF WAV data")
    p.add_argument("--output", type=Path, default=Path("data/dtmf"))
    p.add_argument("--count", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-sequence-length", type=int, default=8)
    a = p.parse_args(argv)
    print(generate_dataset(a.output, a.count, a.seed, a.max_sequence_length))


def train_main(argv=None):
    from .engine import train
    p = argparse.ArgumentParser(description="Train PPF or direct CNN baseline")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("checkpoints/ppf.pt"))
    p.add_argument("--epochs", type=int, default=5); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto"); p.add_argument("--baseline", action="store_true")
    a = p.parse_args(argv)
    train(a.manifest, a.output, a.epochs, a.batch_size, a.lr, _device(a.device), a.seed, a.baseline)


def evaluate_main(argv=None):
    import torch
    from .engine import ToneEventDataset, evaluate_goertzel, evaluate_neural, load_checkpoint
    p = argparse.ArgumentParser(description="Evaluate PPF plus classical baseline")
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--checkpoint", type=Path)
    p.add_argument("--cnn-checkpoint", type=Path)
    p.add_argument("--split", default="test"); p.add_argument("--device", default="auto")
    p.add_argument("--top-k", type=int, default=3)
    a = p.parse_args(argv); dataset = ToneEventDataset(a.manifest, a.split)
    result = {"goertzel": evaluate_goertzel(dataset)}
    if a.checkpoint:
        model, _ = load_checkpoint(a.checkpoint, _device(a.device))
        result["ppf"] = evaluate_neural(model, dataset, device=_device(a.device), top_k=a.top_k)
    if a.cnn_checkpoint:
        model, _ = load_checkpoint(a.cnn_checkpoint, _device(a.device), expected_type="cnn")
        result["cnn"] = evaluate_neural(model, dataset, device=_device(a.device), top_k=a.top_k)
    print(json.dumps(result, indent=2))


def smoke_main(argv=None):
    import torch
    from .config import ModelConfig
    from .data import generate_dataset
    from .engine import ToneEventDataset, evaluate_goertzel
    from .model import ParallelPossibilityFramework
    p = argparse.ArgumentParser(); p.add_argument("--workdir", type=Path); a = p.parse_args(argv)
    base = a.workdir or Path(tempfile.mkdtemp(prefix="dtmf-ppf-"))
    manifest = generate_dataset(base / "data", 12, seed=123, max_sequence_length=2)
    dataset = ToneEventDataset(manifest, "train")
    audio, label = dataset[0]
    cfg = ModelConfig(model_dim=32, code_dim=32, codebook_size=16,
                      transformer_layers=1, transformer_heads=4)
    model = ParallelPossibilityFramework(cfg)
    result = model(audio.unsqueeze(0), torch.tensor([label]))
    result["losses"]["total"].backward()
    summary = {"status": "ok", "manifest": str(manifest), "events": len(dataset),
               "waveform_shape": list(audio.shape),
               "level_shapes": [list(x.shape) for x in result["level_logits"]],
               "loss": float(result["losses"]["total"].detach()),
               "goertzel": evaluate_goertzel(dataset)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": smoke_main()

