"""Controlled comparison of the EXISTING event model, not the final architecture.
Run from repository root: PYTHONPATH=. .venv/bin/python experiments/audit_comparison.py
"""
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ppf.config import ModelConfig
from ppf.data import synthesize_sequence, write_wav, load_manifest
from ppf.engine import ToneEventDataset, train, load_checkpoint
from ppf.model import ParallelPossibilityFramework

ROOT = Path('audit_results')
torch.set_num_threads(1)
DATA_SEED = 20261031
TRAIN_SEEDS = [7, 17, 27]
EPOCHS = 6

def make_data():
    rng = np.random.default_rng(DATA_SEED)
    rows = []
    hashes = {}
    for split, count in [('train', 400), ('val', 80), ('test', 120)]:
        for j in range(count):
            seq = ''.join(str(v) for v in rng.integers(0, 10, 4))
            audio, meta = synthesize_sequence(seq, rng)
            rel = f'{split}/{j:04d}.wav'
            path = ROOT / 'data' / rel
            write_wav(path, audio, 8000)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in hashes:
                raise AssertionError('Duplicate waveform')
            hashes[digest] = split
            rows.append({'path': rel, 'split': split, **meta})
    manifest = ROOT / 'data/manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return manifest

UNIVERSE = torch.cartesian_prod(*[torch.arange(10)] * 4)

def summary(values):
    return {'p50': float(np.percentile(values, 50)), 'p95': float(np.percentile(values, 95))}

@torch.inference_mode()
def assess(model, ds, samples):
    model.eval()
    # Oracle event boundaries are used for BOTH models. Each latent hierarchy
    # here is a tuple of four independent event-model hierarchies.
    seqs = torch.stack([ds[i][0] for i in range(len(ds))]).reshape(-1, 4, 2400)
    labels = torch.tensor([ds[i][1] for i in range(len(ds))]).reshape(-1, 4)
    stats = {'digit_accuracy': [], 'sequence_top1': [], 'sequence_top5': [],
             'sequence_nll': [], 'sequence_brier': [], 'confidence': [],
             'latency_ms': []}
    recon = [[] for _ in range(3)]
    usages = [set() for _ in range(3)]
    # Warmup excludes file I/O and includes frontend, generation and ranking.
    for _ in range(3):
        if isinstance(model, ParallelPossibilityFramework):
            h, _ = model.predict_latent(seqs[0], sample=True)
            model.decoder(h)
        else:
            model(seqs[0])
    for audio, truth in zip(seqs, labels):
        start = time.perf_counter()
        if isinstance(model, ParallelPossibilityFramework):
            repeated = audio.repeat(samples, 1)
            h, _ = model.predict_latent(repeated, sample=True, temperature=1.)
            probs = model.decoder(h).softmax(-1).reshape(samples, 4, 10)
        else:
            probs = model(audio).softmax(-1).unsqueeze(0)
        # Exact enumeration of the finite mixture; no candidate truncation.
        score = torch.ones(len(probs), 10000)
        for pos in range(4):
            score *= probs[:, pos, UNIVERSE[:, pos]]
        score = score.mean(0)
        top = score.topk(5).indices
        predicted = UNIVERSE[top]
        stats['latency_ms'].append((time.perf_counter() - start) * 1000)
        hit = bool((predicted[0] == truth).all())
        target_index = int(truth @ torch.tensor([1000, 100, 10, 1]))
        stats['digit_accuracy'].append(float((probs.mean(0).argmax(-1) == truth).float().mean()))
        stats['sequence_top1'].append(hit)
        stats['sequence_top5'].append(bool((predicted == truth).all(-1).any()))
        stats['sequence_nll'].append(float(-score[target_index].clamp_min(1e-30).log()))
        stats['sequence_brier'].append(float(score.square().sum() - 2*score[target_index] + 1))
        stats['confidence'].append(float(score[top[0]]))
        if isinstance(model, ParallelPossibilityFramework):
            tok = model.tokenizer(model.encoder(audio))
            accum = torch.zeros_like(tok.reconstructed)
            for level, q in enumerate(tok.quantized_levels):
                accum += model.tokenizer._resize(q, accum.shape[1])
                recon[level].append(float((model.decoder(accum).argmax(-1) == truth).float().mean()))
                usages[level].update(tok.codes[level].flatten().tolist())
    result = {k: float(np.mean(v)) for k,v in stats.items() if k not in ('confidence','latency_ms')}
    conf = np.array(stats['confidence']); hits = np.array(stats['sequence_top1'])
    ece = 0.
    for lo in np.arange(0,1,.1):
        mask = (conf >= lo) & (conf < lo + .1 if lo < .9 else conf <= 1.)
        if mask.any(): ece += mask.mean() * abs(conf[mask].mean() - hits[mask].mean())
    result.update(sequence_ece_10_bins=float(ece), latency_ms=summary(stats['latency_ms']),
                  parameters=sum(p.numel() for p in model.parameters()), sequences=len(seqs),
                  hierarchy_samples=samples, sequential_generator_levels=3 if recon[0] else 0,
                  gpu_memory_bytes=None)
    if recon[0]:
        result['audio_tokenizer_teacher_digit_accuracy_by_level'] = [float(np.mean(v)) for v in recon]
        result['test_codebook_unique_codes_by_level'] = [len(v) for v in usages]
    return result

def main():
    ROOT.mkdir(exist_ok=True)
    manifest = make_data()
    ds = ToneEventDataset(manifest, 'test')
    result = {'scope': 'Existing isolated-tone audio-tokenizer prototype; oracle segmentation; not final label-tokenizer PPF',
              'torch': torch.__version__, 'device': 'cpu', 'threads': 1,
              'data_seed': DATA_SEED, 'training_seeds': TRAIN_SEEDS, 'epochs': EPOCHS,
              'splits_sequences': {'train':400,'val':80,'test':120},
              'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'duplicate_waveforms': 0, 'runs': []}
    for seed in TRAIN_SEEDS:
        for baseline in [True, False]:
            name = 'cnn' if baseline else 'ppf'
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            checkpoint = ROOT / f'{name}_{seed}.pt'
            begin = time.perf_counter()
            history = train(manifest, checkpoint, epochs=EPOCHS, seed=seed, baseline=baseline)
            elapsed = time.perf_counter() - begin
            model, _ = load_checkpoint(checkpoint)
            for m in ([1] if baseline else [1,4,8,16]):
                torch.manual_seed(seed + 1000)
                metrics = assess(model, ds, m)
                row = {'model': name, 'seed':seed, 'training_seconds':elapsed,
                       'history':history, 'metrics':metrics}
                result['runs'].append(row)
                print(json.dumps({'model': name,'seed':seed,'metrics':metrics}), flush=True)
                (ROOT/'comparison.json').write_text(json.dumps(result, indent=2))
    print('Saved audit_results/comparison.json', flush=True)

if __name__ == '__main__': main()
