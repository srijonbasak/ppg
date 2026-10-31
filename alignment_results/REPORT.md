# Adapter and code-prediction alignment study

Three fixed seeds (317, 419, 523); all percentages are mean ± sample SD. These are development measurements, not an independent confirmatory test. No CNN, tokenizer, candidate policy, or existing checkpoint was replaced.

## Main findings

The current adapter retains the coarse-code target: its linear probe reaches 94.48 ± 0.60%, while the original jointly trained NCM reaches 40.37 ± 6.83%. Directly aligning code training raises the latter to 92.13 ± 1.14%, but protected sequence top-1 changes from 69.38 ± 0.77% to 66.91 ± 1.19%. Better canonical-code prediction does not establish better recognition.

The reversible adapter retains native stem/final values but does not establish a recognition advantage. The results localize the coarse-code discrepancy primarily to readout/training alignment, rather than absent coarse-code information in the adapter. They do not prove that hierarchical generation itself is useful.

## Controlled protocol

- Frozen corrected-frontend CNN and frozen per-seed acoustic residual VQ books; 1→2→4 slots, 64 codes/level, dimension 32. Targets are codes of the same observed final CNN features, not clean-label codes. This makes coarse information recoverable from the original adapter by construction.
- Original train/validation/test splits: 1,200/184/270 recordings. Training pairs original corrupted recordings with clean renderings (seed 817); 9,600 local digit events. Validation pairs clean seed 619. Whole combinations are disjoint; 266 distinct test labels. Test clean seed 318; row-only missing-frequency seed 818. Noise (68) and partial-erasure (67) are overlapping subsets of the 270-recording corrupted test.
- All new models use 16 epochs × 75 batches = 1,200 AdamW updates, batch 128, lr 0.001, gradient norm cap 1. Fixed final epoch, no test selection. Original-joint reuses the exact preserved encoder-study checkpoint trained with this protocol. Temperature uses only the existing validation fitting half (92 recordings plus paired clean).
- All NCM variants retain 34,570 parameters, one decoder block, four heads, width 32, feedforward 128, dropout 0.1. Lossless memory is 128 tokens versus 4: equal update budgets are not equal compute budgets.
- Original-joint loss: mean digit CE over three generated prefixes plus 0.1×mean code CE. Hard code forward uses straight-through soft gradients. Aligned loss: unit-weight mean code CE on generated prefixes; separate digit decoder CE on canonical observed prefixes. Digit gradients do not change predictor code identities. This bundles code weight and gradient-routing changes; their individual effects are not isolated.
- Target-free evaluation generates every earlier level. Oracle level-1 is explicitly diagnostic. The coarse-only Transformer diagnostic trains only level 1; its fine-code/recognition outputs are not evaluated.
- Protected ranking uses identical frozen routing, acceptance masks, and candidate lists from the encoder study, verified live and by saved policy hashes. Unaccepted positions use the CNN. Raw ranking independently enumerates all 10,000 sequences from four digit distributions. No test labels enter prediction or temperature fitting; training labels supervise losses, validation labels fit temperatures, and test labels calculate metrics only.

## Actual path and shapes

| Operation per digit event | Output |
|---|---|
| Audio at 8 kHz | N×2400 |
| Hann STFT 256, hop 80; 40 mel filters; relative-floor log; event normalization | N×40×31 |
| Conv1d 40→64, k5/p2; three residual blocks (dilations 1,2,4), GroupNorm/GELU | N×64×31 |
| Native final features, adaptive average pool 31→4 | N×4×64 |
| Train-only channel mean/std, PCA 64→32, projected standard deviation whitening | N×4×32 |
| Add learned audio positions, Linear 32→32 | N×4×32 |
| Level query: sinusoidal slot positions + learned level embedding + projected generated prior codes (levels 2/3) | N×1/2/4×32 |
| Pre-norm Transformer decoder: LayerNorm, self/cross attention, feedforward/ReLU, dropout and residuals | N×1/2/4×32 |
| Level-specific linear head, CE over code identities 0…63 | N×1/2/4×64 |
| Lookup frozen books, interpolate prefixes to 4, sum residuals; latent-only digit ranker | N×10 |

The preserved legacy tokenizer pools before affine projection; the conditioning implementation projects before pooling. The operations commute up to floating-point roundoff; code targets remain identical. There is no CNN temporal downsampling before the adapter. No causal, padding, or ignored-label masks are used: all positions within each level run in parallel. Batch/event and slot order is preserved.

The new adapter keeps four original anchors exactly, plus native stem and final tensors normalized by training statistics, split into ordered 32-channel halves with fixed reversible time/type offsets. This packs 2×64×31 values into 124 additional tokens. Learned NCM audio positions are interpolated to 128. It adds no trainable parameters. These are local time positions inside a single digit event; the current NCM is not a four-digit sequence Transformer.

## Identical coarse linear probes

All probes are standardized Linear(512,64), 32,832 parameters, same seeds/data/updates. Native inputs use four pooled time slots: final=256 active dimensions; stem+final=512. Current adapter and actual projected memory use flattened 128 dimensions padded to 512. Hidden uses 32 padded dimensions. The lossless probe uses the fixed inverse and the identical stem+final summary; it therefore verifies preservation rather than a learned token aggregation. Active information dimensions differ despite identical trainable heads. These are new coarse-only heads, not replayed weights of the earlier 94.17% multi-level probe.

| Probe | Corrupted | Clean | Noise | Partial erasure | Missing frequency |
|---|---:|---:|---:|---:|---:|
| native_final | 92.84 ± 0.75 | 100.00 ± 0.00 | 93.14 ± 0.56 | 88.18 ± 2.19 | 100.00 ± 0.00 |
| native_stem_final | 92.16 ± 0.68 | 100.00 ± 0.00 | 93.01 ± 1.33 | 87.31 ± 2.96 | 100.00 ± 0.00 |
| current_adapter | 94.48 ± 0.60 | 100.00 ± 0.00 | 94.36 ± 0.56 | 90.80 ± 1.41 | 100.00 ± 0.00 |
| lossless_adapter | 92.16 ± 0.68 | 100.00 ± 0.00 | 93.01 ± 1.33 | 87.31 ± 2.96 | 100.00 ± 0.00 |
| actual_memory | 93.24 ± 0.65 | 100.00 ± 0.00 | 92.77 ± 1.81 | 89.05 ± 2.48 | 100.00 ± 0.00 |
| transformer_hidden | 73.27 ± 2.88 | 100.00 ± 0.00 | 76.96 ± 3.86 | 65.05 ± 6.29 | 63.09 ± 5.51 |

## Generated codes and raw recognition

| Condition / model | Code L1 % | L2 % | L3 % | Digit % | Sequence top-1 % | Top-5 % | ECE % | NLL | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| corrupted / original_joint | 40.37 ± 6.83 | 65.15 ± 3.00 | 51.40 ± 0.36 | 88.95 ± 0.23 | 71.98 ± 0.93 | 83.95 ± 1.67 | 2.94 ± 0.90 | 0.33 ± 0.04 | 0.15 ± 0.01 |
| corrupted / lossless_joint | 38.46 ± 3.45 | 62.93 ± 4.70 | 52.62 ± 1.05 | 86.88 ± 0.70 | 67.90 ± 0.57 | 81.48 ± 0.74 | 3.06 ± 1.70 | 0.39 ± 0.05 | 0.18 ± 0.02 |
| corrupted / original_aligned | 92.13 ± 1.14 | 86.84 ± 1.48 | 67.31 ± 1.10 | 83.36 ± 1.52 | 63.46 ± 1.50 | 76.17 ± 2.60 | 2.64 ± 1.11 | 0.44 ± 0.02 | 0.21 ± 0.01 |
| corrupted / lossless_aligned | 87.65 ± 1.31 | 84.81 ± 2.59 | 64.92 ± 0.07 | 82.96 ± 0.72 | 63.95 ± 2.41 | 75.80 ± 1.40 | 2.46 ± 0.67 | 0.43 ± 0.02 | 0.21 ± 0.01 |
| clean / original_joint | 91.14 ± 9.08 | 100.00 ± 0.00 | 98.33 ± 0.22 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.26 ± 0.22 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| clean / lossless_joint | 82.93 ± 11.01 | 98.60 ± 2.43 | 98.33 ± 0.22 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.15 ± 0.11 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| clean / original_aligned | 100.00 ± 0.00 | 100.00 ± 0.00 | 98.33 ± 0.22 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.20 ± 0.08 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| clean / lossless_aligned | 100.00 ± 0.00 | 100.00 ± 0.00 | 98.33 ± 0.22 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.37 ± 0.14 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| noisy / original_joint | 30.64 ± 2.81 | 62.01 ± 6.10 | 39.77 ± 1.53 | 97.79 ± 0.64 | 91.18 ± 2.55 | 97.55 ± 0.85 | 1.96 ± 0.53 | 0.12 ± 0.03 | 0.04 ± 0.01 |
| noisy / lossless_joint | 32.97 ± 3.53 | 61.09 ± 7.82 | 40.50 ± 2.66 | 95.71 ± 0.93 | 85.29 ± 2.94 | 94.12 ± 1.47 | 2.18 ± 0.86 | 0.21 ± 0.03 | 0.07 ± 0.01 |
| noisy / original_aligned | 91.42 ± 0.77 | 81.86 ± 1.97 | 52.30 ± 2.05 | 96.32 ± 0.74 | 85.78 ± 2.25 | 94.61 ± 0.85 | 1.68 ± 0.54 | 0.17 ± 0.04 | 0.06 ± 0.01 |
| noisy / lossless_aligned | 89.71 ± 2.24 | 79.04 ± 3.06 | 49.45 ± 0.48 | 96.57 ± 0.56 | 87.75 ± 3.40 | 95.59 ± 1.47 | 2.68 ± 0.35 | 0.15 ± 0.03 | 0.06 ± 0.01 |
| partial_erasure / original_joint | 44.28 ± 11.44 | 71.33 ± 1.55 | 59.45 ± 2.73 | 95.65 ± 0.43 | 82.59 ± 1.72 | 95.52 ± 1.49 | 2.16 ± 0.59 | 0.21 ± 0.06 | 0.07 ± 0.00 |
| partial_erasure / lossless_joint | 39.18 ± 7.21 | 67.35 ± 1.87 | 60.23 ± 1.52 | 93.28 ± 1.12 | 73.13 ± 4.48 | 95.02 ± 2.28 | 2.97 ± 1.62 | 0.27 ± 0.04 | 0.10 ± 0.02 |
| partial_erasure / original_aligned | 88.43 ± 2.24 | 83.52 ± 2.10 | 69.37 ± 1.38 | 91.04 ± 1.35 | 64.18 ± 5.38 | 92.54 ± 2.99 | 3.36 ± 0.53 | 0.33 ± 0.03 | 0.13 ± 0.02 |
| partial_erasure / lossless_aligned | 80.22 ± 4.24 | 81.72 ± 2.59 | 67.79 ± 1.23 | 91.17 ± 2.12 | 64.68 ± 8.49 | 94.53 ± 0.86 | 3.67 ± 1.80 | 0.32 ± 0.08 | 0.13 ± 0.03 |
| missing_frequency / original_joint | 19.97 ± 34.59 | 16.81 ± 18.21 | 16.15 ± 0.88 | 6.82 ± 5.96 | 0.00 ± 0.00 | 0.00 ± 0.00 | 55.04 ± 8.63 | 5.71 ± 1.19 | 1.38 ± 0.09 |
| missing_frequency / lossless_joint | 27.28 ± 16.09 | 27.33 ± 6.31 | 25.88 ± 9.00 | 12.78 ± 5.53 | 0.00 ± 0.00 | 0.12 ± 0.21 | 50.93 ± 19.37 | 5.43 ± 2.09 | 1.32 ± 0.25 |
| missing_frequency / original_aligned | 76.17 ± 12.38 | 58.53 ± 7.22 | 33.43 ± 8.01 | 20.34 ± 9.45 | 0.12 ± 0.21 | 0.25 ± 0.43 | 38.02 ± 14.97 | 4.95 ± 1.38 | 1.22 ± 0.18 |
| missing_frequency / lossless_aligned | 93.64 ± 5.51 | 65.22 ± 14.28 | 22.01 ± 10.91 | 10.52 ± 1.09 | 0.00 ± 0.00 | 0.12 ± 0.21 | 36.11 ± 3.95 | 4.42 ± 0.64 | 1.13 ± 0.05 |

## Protected pipeline and CPU latency

End-to-end single-recording wall time includes CNN, evidence/gate evaluation, accepted-digit NCM and candidate ranking. One CPU thread, three warmups, one timed pass per variant; machine load/cache can affect timing. p50/p95 values below are the mean of per-seed percentiles; no throughput claim. Missing-frequency uses the unchanged study policy without an analytic missing-band guard. It is an out-of-distribution stress test, not a claim to infer absent columns.

| Condition / model | Digit % | Seq top-1 % | Top-5 % | ECE % | NLL | Brier | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| corrupted / original_joint | 87.65 ± 0.60 | 69.38 ± 0.77 | 78.27 ± 1.54 | 4.35 ± 2.05 | 0.46 ± 0.12 | 0.20 ± 0.04 | 1.39 ± 0.45 | 2.00 ± 0.04 |
| corrupted / lossless_joint | 87.19 ± 0.79 | 68.64 ± 1.30 | 77.28 ± 0.21 | 4.83 ± 2.65 | 0.49 ± 0.11 | 0.21 ± 0.03 | 1.44 ± 0.50 | 2.19 ± 0.11 |
| corrupted / original_aligned | 85.28 ± 1.37 | 66.91 ± 1.19 | 75.06 ± 1.19 | 3.77 ± 1.30 | 0.51 ± 0.08 | 0.22 ± 0.03 | 1.42 ± 0.46 | 2.05 ± 0.05 |
| corrupted / lossless_aligned | 84.85 ± 1.26 | 67.41 ± 1.11 | 74.81 ± 1.92 | 3.72 ± 1.48 | 0.51 ± 0.07 | 0.22 ± 0.02 | 1.45 ± 0.51 | 2.20 ± 0.10 |
| clean / original_joint | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 3.99 ± 0.98 | 0.04 ± 0.01 | 0.01 ± 0.00 | 1.14 ± 0.10 | 1.42 ± 0.50 |
| clean / lossless_joint | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 3.99 ± 0.97 | 0.04 ± 0.01 | 0.01 ± 0.00 | 1.18 ± 0.11 | 1.81 ± 0.37 |
| clean / original_aligned | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 3.99 ± 0.97 | 0.04 ± 0.01 | 0.01 ± 0.00 | 1.16 ± 0.11 | 1.58 ± 0.41 |
| clean / lossless_aligned | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 4.02 ± 0.92 | 0.04 ± 0.01 | 0.01 ± 0.00 | 1.16 ± 0.07 | 1.61 ± 0.44 |
| noisy / original_joint | 97.43 ± 0.97 | 90.20 ± 4.25 | 98.04 ± 2.25 | 4.39 ± 1.38 | 0.16 ± 0.05 | 0.06 ± 0.02 | 1.45 ± 0.45 | 1.98 ± 0.04 |
| noisy / lossless_joint | 96.57 ± 0.56 | 88.24 ± 2.94 | 94.61 ± 1.70 | 3.75 ± 2.03 | 0.20 ± 0.03 | 0.08 ± 0.02 | 1.55 ± 0.57 | 2.23 ± 0.10 |
| noisy / original_aligned | 97.18 ± 0.93 | 88.73 ± 3.70 | 96.57 ± 3.06 | 4.42 ± 2.54 | 0.18 ± 0.04 | 0.07 ± 0.02 | 1.47 ± 0.47 | 2.02 ± 0.04 |
| noisy / lossless_aligned | 97.30 ± 1.12 | 89.71 ± 4.41 | 98.04 ± 1.70 | 4.87 ± 1.96 | 0.16 ± 0.04 | 0.07 ± 0.02 | 1.53 ± 0.57 | 2.13 ± 0.10 |
| partial_erasure / original_joint | 93.66 ± 0.65 | 74.63 ± 2.59 | 84.58 ± 2.28 | 2.48 ± 0.29 | 0.22 ± 0.05 | 0.09 ± 0.01 | 1.13 ± 0.09 | 1.66 ± 0.47 |
| partial_erasure / lossless_joint | 93.03 ± 0.43 | 72.14 ± 1.72 | 83.58 ± 2.59 | 2.10 ± 0.31 | 0.22 ± 0.00 | 0.10 ± 0.00 | 1.15 ± 0.10 | 1.77 ± 0.56 |
| partial_erasure / original_aligned | 93.16 ± 0.22 | 72.64 ± 0.86 | 84.58 ± 2.28 | 2.46 ± 0.22 | 0.22 ± 0.01 | 0.10 ± 0.00 | 1.14 ± 0.10 | 1.70 ± 0.51 |
| partial_erasure / lossless_aligned | 93.16 ± 0.43 | 72.64 ± 1.72 | 84.58 ± 3.45 | 2.19 ± 0.39 | 0.21 ± 0.01 | 0.10 ± 0.00 | 1.14 ± 0.09 | 1.77 ± 0.47 |
| missing_frequency / original_joint | 16.91 ± 6.27 | 0.00 ± 0.00 | 0.37 ± 0.37 | 53.82 ± 1.57 | 4.40 ± 0.96 | 1.27 ± 0.03 | 1.87 ± 0.08 | 2.01 ± 0.03 |
| missing_frequency / lossless_joint | 20.15 ± 10.61 | 0.25 ± 0.43 | 0.37 ± 0.37 | 53.34 ± 12.33 | 4.44 ± 1.64 | 1.23 ± 0.20 | 2.07 ± 0.16 | 2.68 ± 0.69 |
| missing_frequency / original_aligned | 20.12 ± 9.79 | 0.12 ± 0.21 | 0.62 ± 1.07 | 43.53 ± 10.84 | 4.43 ± 1.64 | 1.20 ± 0.20 | 1.90 ± 0.08 | 2.05 ± 0.05 |
| missing_frequency / lossless_aligned | 21.05 ± 10.95 | 0.37 ± 0.37 | 0.99 ± 1.13 | 36.01 ± 7.14 | 3.72 ± 0.94 | 1.08 ± 0.07 | 2.02 ± 0.13 | 2.47 ± 0.50 |

## Oracle level-1 diagnostic

The first code is supplied from the observed acoustic tokenizer; later codes remain generated. This is not ordinary inference and not a ground-truth digit oracle. It tests compatibility with canonical earlier codes; it does not remove all acoustic ambiguity.

| Corrupted model | Normal L2/L3 % | Oracle L2/L3 % | Normal digit % | Oracle digit % | Normal seq top-1 % | Oracle seq top-1 % |
|---|---:|---:|---:|---:|---:|---:|
| original_joint | 65.15 ± 3.00 / 51.40 ± 0.36 | 55.42 ± 5.00 / 50.84 ± 0.79 | 88.95 ± 0.23 | 83.49 ± 1.77 | 71.98 ± 0.93 | 63.33 ± 4.86 |
| lossless_joint | 62.93 ± 4.70 / 52.62 ± 1.05 | 55.26 ± 5.43 / 50.66 ± 0.79 | 86.88 ± 0.70 | 81.05 ± 1.96 | 67.90 ± 0.57 | 58.52 ± 1.34 |
| original_aligned | 86.84 ± 1.48 / 67.31 ± 1.10 | 87.44 ± 1.40 / 67.63 ± 1.17 | 83.36 ± 1.52 | 83.83 ± 2.13 | 63.46 ± 1.50 | 64.81 ± 1.70 |
| lossless_aligned | 84.81 ± 2.59 / 64.92 ± 0.07 | 85.65 ± 2.35 / 65.35 ± 0.12 | 82.96 ± 0.72 | 83.12 ± 0.77 | 63.95 ± 2.41 | 63.58 ± 2.47 |

## Audit and uncertainty

- Seed 317: analytic nearest-code readout at adapter 100.00%, after undoing learned audio projection 100.00%; projection singular values [0.007789, 1.629294]. Native reconstruction max error 9.54e-07; anchors error 0.0; traced level-1 logits error 0.0; code indices [0, 63].
- Seed 419: analytic nearest-code readout at adapter 100.00%, after undoing learned audio projection 100.00%; projection singular values [0.010026, 1.612505]. Native reconstruction max error 9.54e-07; anchors error 0.0; traced level-1 logits error 0.0; code indices [0, 63].
- Seed 523: analytic nearest-code readout at adapter 100.00%, after undoing learned audio projection 100.00%; projection singular values [0.052193, 1.579491]. Native reconstruction max error 9.54e-07; anchors error 0.0; traced level-1 logits error 0.0; code indices [0, 63].

Level-1-only Transformer coarse accuracy: 91.67 ± 0.93%. This keeps the Transformer architecture and input unchanged; loss focus alone closes much of the coarse-code gap.

Paired label-cluster bootstrap (10,000 resamples; average seed effect; 266 unique sequence labels, repeated labels clustered). Intervals quantify test-label uncertainty conditional on the three trained seeds, not population training uncertainty:

- lossless_joint minus original_joint protected top1: -0.74 percentage points, 95% interval [-2.35, +0.85].
- lossless_joint minus original_joint protected top5: -0.99 percentage points, 95% interval [-2.85, +0.86].
- original_aligned minus original_joint protected top1: -2.47 percentage points, 95% interval [-4.29, -0.74].
- original_aligned minus original_joint protected top5: -3.21 percentage points, 95% interval [-5.06, -1.48].
- lossless_aligned minus original_joint protected top1: -1.98 percentage points, 95% interval [-3.83, -0.25].
- lossless_aligned minus original_joint protected top5: -3.46 percentage points, 95% interval [-5.54, -1.49].

No gross index, masking, shape or normalization error was found. The original hidden state permits a stronger newly fitted coarse readout than its original head, but still underperforms adapter probes. This supports a readout/objective/optimization explanation; it does not uniquely identify one internal Transformer operation as causal.

Coarse codes are nearest centroids of the observed pooled input. An analytic 100% code readout is expected and is not digit-recognition success. The codes themselves can preserve nuisance variation or ambiguous audio; improving their fidelity can hurt the digit objective. Oracle-prefix results must also be interpreted in light of code aliases learned by the joint objective.

The reversible adapter is a controlled information-preservation test, not evidence of an efficient superior adapter. Its long memory changes attention distribution and cost. Code-aligned training also changes the decoder training distribution, so a recognition regression is not proof that canonical codes can never help. The evidence supports keeping the existing CNN while resolving objective alignment; it does not establish NCM-specific advantage or justify a larger model.

## Reproduce and artifacts

```bash
PYTHONPATH=. .venv/bin/python experiments/adapter_alignment.py
PYTHONPATH=. .venv/bin/python experiments/report_alignment.py
PYTHONPATH=. .venv/bin/python -m pytest -q
```

Configuration: experiments/alignment_config.json. Raw measurements: alignment_results/metrics.json; bootstrap summary: alignment_results/summary.json. Source hashes and original-checkpoint identity checks are recorded in alignment_results/source_manifest.json. Separate per-seed probe/model/adapter checkpoints and candidate policies are in alignment_results/seed_*. Original files under representation_results, gate_results and encoder_results remain preserved. All five-condition oracle metrics and per-position digit metrics are in raw JSON. Tests: 42 passed, including reversible preservation, exact NCM trace, explicit oracle prefix and frozen-codebook gradients.
