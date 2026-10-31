import json

import numpy as np
import torch

from ppf.config import ModelConfig
from ppf.data import SynthesisConfig, generate_dataset, load_manifest, synthesize_sequence
from ppf.engine import ToneEventDataset, goertzel_predict
from ppf.model import ParallelPossibilityFramework


def tiny_config():
    return ModelConfig(model_dim=32, code_dim=32, codebook_size=16,
                       transformer_layers=1, transformer_heads=4)


def test_synthesis_is_labeled_and_reproducible():
    cfg = SynthesisConfig(tone_ms=(100, 100), gap_ms=(20, 20))
    a, ma = synthesize_sequence("507", np.random.default_rng(4), cfg)
    b, mb = synthesize_sequence("507", np.random.default_rng(4), cfg)
    assert np.array_equal(a, b)
    assert ma["sequence"] == "507" and len(ma["events"]) == 3
    assert ma == mb and a.dtype == np.float32


def test_manifest_and_event_dataset(tmp_path):
    manifest = generate_dataset(tmp_path / "dtmf", 10, seed=9, max_sequence_length=2)
    rows = load_manifest(manifest)
    assert len(rows) == 10 and {r["split"] for r in rows} == {"train", "val", "test"}
    event_data = ToneEventDataset(manifest, "train")
    audio, label = event_data[0]
    assert audio.shape == (2400,) and 0 <= label <= 9


def test_multiscale_parallel_model_forward_and_generation():
    model = ParallelPossibilityFramework(tiny_config())
    waveform = torch.randn(2, 2400) * .02
    result = model(waveform, torch.tensor([1, 2]))
    lengths = [x.shape[1] for x in result["level_logits"]]
    assert lengths[0] < lengths[1] < lengths[2]
    assert all(x.shape[0] == 2 and x.shape[-1] == 16 for x in result["level_logits"])
    result["losses"]["total"].backward()
    generated, codes = model.predict_latent(waveform)
    assert generated.shape[:2] == (2, lengths[-1])
    assert len(codes) == 3


def test_goertzel_baseline_on_clean_digit():
    cfg = SynthesisConfig(tone_ms=(180, 180), gap_ms=(20, 20))
    audio, _ = synthesize_sequence("8", np.random.default_rng(2), cfg, corruption=False)
    assert goertzel_predict(audio) == 8


def test_evaluation_uses_generated_codes(monkeypatch):
    from ppf.engine import evaluate_neural
    model = ParallelPossibilityFramework(tiny_config())
    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation must not use the tokenizer/teacher path")
    monkeypatch.setattr(model.tokenizer, "forward", forbidden)
    result = evaluate_neural(model, [(torch.zeros(2400), 0)], top_k=5)
    assert result["samples"] == 1

