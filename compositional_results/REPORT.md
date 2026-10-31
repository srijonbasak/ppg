# Compositional-generalization investigation — 2026-10-31

## Outcome

The hypothesis is partly supported: canonical fine-code identities in the original residual hierarchy depend on other digits, and a local structured representation makes them easier to predict. However, the original full hierarchy already supports perfect linear digit recovery, so it is not merely a whole-sequence lookup table. Making fine codes local did not produce a single correct complete held-out sequence in this fixed generator experiment.

## Data and controls

Existing audio conditions and tensors were retained. Training: 1200 recordings (1127 distinct sequences); validation: 184 recordings; test: 270 recordings (266 distinct sequences). Validation excludes training combinations; test excludes both training and validation combinations. The original raw test contains 300 recordings; 30 overlapping combinations were excluded identically for both variants. New subsets and exact source indices are saved in split_indices.json.

All ten digits at every position and all 100 digit pairs at each of the six position pairs appear in training. Thus the held-out combinations consist entirely of individually familiar digits and pairs. Some test combinations have multiple independent waveform renderings; recordings are not 270 independent unique label combinations.

The original frozen tokenizer is retained unchanged. It was previously trained on 8,000 clean label combinations; the new tokenizer trains only on the current 1,200 training labels. The original tokenizer therefore has a different training history and may have seen held-out label combinations during its earlier representation training. Neither generator receives held-out labels in its prediction path. This is a development comparison with a retained frozen control, not a fully matched tokenizer-training-history trial.

Both generator variants use the same existing frozen CNN, identical fresh predictor/event-position initialization, identical example order, 16 epochs, AdamW rate 0.0003, batch 32, teacher forcing, and fixed final-epoch evaluation. Tokenizers/decoders stay frozen. Both have 232,330 total parameters and 154,752 trainable parameters; no model growth, prefix-training intervention or sampling increase. Inference uses deterministic argmax, M=1, and exhaustive ranking of 10,000 decoded sequence products. CNN and tokenizer tensors were checked unchanged after training.

## Equivalent linear digit probes

All digit probes have 256 inputs and 40 outputs (four ten-way classifiers), 10,280 parameters. Features use training-only mean/std normalization. They train for 600 Adam updates at 0.01; there is no test-based hyperparameter or checkpoint selection. CNN features are ordered per-event temporal means. Token features are flattened 4×64 reconstructed prefix latents. Label-code probes receive clean target representations, while the CNN receives noisy audio; absolute accuracy between those sources is not a matched-noise comparison.

| Source | Position 1 | Position 2 | Position 3 | Position 4 | All four correct |
|---|---:|---:|---:|---:|---:|
| Frozen CNN | 82.59% | 82.59% | 80.00% | 83.33% | 59.63% |
| original, prefix 1 | 10.00% | 48.52% | 56.67% | 7.41% | 0.00% |
| original, prefix 2 | 85.19% | 85.56% | 85.93% | 82.22% | 52.59% |
| original, prefix 3 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| structured, prefix 1 | 28.89% | 24.81% | 30.74% | 30.00% | 0.00% |
| structured, prefix 2 | 46.30% | 42.22% | 48.52% | 44.44% | 7.41% |
| structured, prefix 3 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

Both full representations generalize perfectly under linear digit decoding. The original coarse level is strongly asymmetric. Source inspection explains why: its pointwise label encoder followed by linear interpolation from four positions to one samples only the central two positions. Changing position 1 or 4 cannot alter this coarse target. This is a coarse-target construction limitation, not proof of sequence memorization.

## Predicting codes directly from frozen audio features

Linear code probes use the same 256-dimensional ordered CNN features and 64 outputs per code slot. Parameter counts are equal between variants for a given level (16,448 / 32,896 / 65,792). These probes receive no earlier code levels. Accuracies below average the code slots at a level.

| Target representation | Level 1 | Level 2 | Level 3 |
|---|---:|---:|---:|
| original | 50.00% | 46.30% | 43.70% |
| structured | 41.11% | 57.78% | 81.02% |

The new fine codes are easier to predict, but the new global codes are not. Code accuracy is representation-dependent: a 10-class local code target is simpler than a context-dependent 64-code target. These percentages are not a common recognition metric.

## Single-digit changes and code similarity

5,000 random anchors per position; exactly one digit is replaced by a different digit. The unrelated control changes all four digits. Code identity is measured within each codebook; arbitrary integer-code numeric distances are not used. Full code-change maps and codebook embedding MSE are saved in *_probes.json.

| Representation | Mean fine-slot change, one digit edited | Fine-slot change, all four edited | Neighbor / unrelated embedding MSE ratio |
|---|---:|---:|---:|
| original | 55.93% | 99.77% | 0.302 |
| structured | 25.00% | 100.00% | 0.250 |

For example, changing digit 2 changes the original fine code at untouched position 1 in 86.86% of cases, position 3 in 40.16%, and position 4 in 26.12%. In the structured tokenizer, only the edited fine slot changes (100%); unchanged fine slots change 0%. Pair-code changes remain within the edited pair. Its global code responds to changes at all four positions. The original outer-position edits change its coarse code 0% of the time.

## Structured tokenizer definition and validation

ppf/compositional.py adds a controlled subspace residual-VQ variant. A shared digit encoder produces 64 channels. Global mean context occupies channels 0–15, pair mean context channels 16–31, and local digit features channels 32–63. Each level subtracts the earlier upsampled reconstruction, then its codebook quantizes the residual projected to its channel block. Learned codes remain at 1→2→4 positions, are summed coarse-to-fine, and use the same latent-only decoder and nested-prefix loss. The fine target is position-independent and unaffected by earlier global/pair codes.

This is a structured/product-subspace residual quantizer: residuals are projected to disjoint blocks rather than permitting every level to revise every channel. The global context is a bag-like digit summary, not an encoding of all sequence order. This is an explicit architectural modification, not a claim of exact dense NCM equivalence. Generator architecture and size are unchanged.

Structured tokenizer checkpoint selected at epoch 5 using validation full-sequence reconstruction, earliest tie, from a fixed 30-epoch cosine-decay run. Validation and test full reconstruction are both 100%, so it passes the predeclared 99% validation gate before conditional training. Test code usage is [18, 38, 10] of 64 per level. Ten fine codes are enough because they represent reusable digit identities; using all 64 is not required.

Important control: the new decoder reconstructs 100% of test sequences from fine codes alone, and removing or swapping coarse/pair components changes none of its generated digit decisions. Each digit uses exactly one fine code. The original uses 5–10 fine codes per digit in this test and fine-only oracle reconstruction is 88.15%. Therefore any eventual benefit of the new representation could reflect a simpler local digit target, rather than a demonstrated benefit from hierarchical refinement. No hierarchy advantage is established here.

## Held-out generator comparison

| Model | Digit accuracy | Sequence top-1 | Sequence top-5 | Sequence NLL | CPU p50 / p95 ms |
|---|---:|---:|---:|---:|---:|
| CNN | 79.91% | 59.26% | 74.44% | — | 0.89 / 0.98 |
| original PPF | 28.15% | 0.00% | 0.00% | 11.238 | 2.04 / 2.41 |
| structured PPF | 32.50% | 0.00% | 0.00% | 8.752 | 2.03 / 2.26 |

| Generator | Oracle-prefix codes L1 / L2 / L3 | Generated-prefix codes L1 / L2 / L3 |
|---|---|---|
| original | 64.81% / 25.56% / 45.09% | 64.81% / 22.04% / 13.15% |
| structured | 51.48% / 26.85% / 52.13% | 51.48% / 25.93% / 32.50% |

| Condition | Recordings | Original sequence top-1 | Structured sequence top-1 |
|---|---:|---:|---:|
| interference | 67 | 0.00% | 0.00% |
| noise | 68 | 0.00% | 0.00% |
| partial_erasure | 67 | 0.00% | 0.00% |
| reference | 68 | 0.00% | 0.00% |

Latency includes four waveform windows through the existing CNN, deterministic hierarchy generation, 10,000-sequence scoring and top-5 retrieval. File loading, oracle segmentation and code diagnostics are excluded; three warmup calls precede single-sequence CPU timings. One thread, GPU unavailable. CPU timing variation is not a demonstrated efficiency difference.

## Interpretation and limits

1. The original full representation already contains linearly reusable digit information. The broad claim that it mainly memorizes complete sequences is not supported.
2. Canonical residual-code identities are context-coupled, and the original coarse target omits the outer digits. The structured intervention makes fine codes local and substantially more predictable.
3. The fixed generator still fails on complete unseen combinations, even with ten digit-like fine codes. Representation organization alone does not resolve the audio-to-position/code learning problem. The existing order-conditioning and generated-prefix weaknesses remain relevant; this experiment does not isolate a sole cause.
4. There is an observed increase in target-free digit accuracy and lower NLL, but no complete-sequence or top-5 gain. No successful sequence recognition or hierarchical advantage is claimed.

This is one seed, synthetic in-distribution corruption families, oracle event windows, and a previously inspected development test pool. The filtered test has 266 distinct combinations and 270 renderings; uncertainty is not estimated from independent replications. Frozen CNN features were previously trained on the same audio training samples. Tokenizer training histories differ as stated above. New independent tests and matched-history controls are needed before a causal or superiority claim.

## Reproduce and verify

```bash
PYTHONPATH=. .venv/bin/python experiments/compositional_probes.py
PYTHONPATH=. .venv/bin/python experiments/compositional_comparison.py
PYTHONPATH=. .venv/bin/python experiments/compositional_hierarchy_use.py
.venv/bin/python experiments/report_compositional.py
.venv/bin/python -m pytest -q
```

Configuration: experiments/compositional_config.json. Full metrics and slot-change matrices: original_probes.json, structured_probes.json, comparison.json, hierarchy_use.json. Structured checkpoint and both generator checkpoints are preserved alongside logs. Original artifacts remain unchanged. All 14 tests pass, including fine-code locality, codebook subspace constraints, frozen-state behavior and compatibility with the existing generator.
