# Hierarchy-contribution experiment — 2026-10-31

## Outcome

With exactly frozen candidate support and override decisions, continuous-feature reranking remains strongest. Multiscale observed quantization improves acoustic reconstruction and some ranking metrics over flat fine-only quantization, but full learned next-level refinement does not establish an advantage over the simpler controlled alternatives. No NCM-specific benefit is claimed.

## Frozen controls

CNN weights and the corrected relative log-Mel floor 0.001 are unchanged. Same 1,200 training, 184 validation and 270 test recordings; test has 266 distinct digit combinations. Three seeds: 317, 419, 523. Clean and corrupted recordings are identical to the preceding shared-gate experiment.

For each seed, the preserved conventional local_gated system supplies both route and acceptance outputs. Both are frozen before new model training. Factored branch lists explicitly record all ten digits at each routed position and the fixed CNN choice elsewhere. All variants use these same lists and score-replacement positions; neither their confidences nor their predictions may change the gate. This is stricter than merely sharing gate parameters. Each list has a saved SHA-256 hash.

The fixed policy is based on the conventional reference model, so this experiment is conditional on that selection policy; it is not an independently optimized gate for every representation. The purpose is component isolation. During full-pipeline timing the original reference policy is recomputed and checked for exact agreement with the saved route/accept masks, then the frozen masks are used for all variants.

## Four variants

| Variant | Representation scored by the same 128→64→10 ranking head |
|---|---|
| A | Continuous projected local CNN features, four vectors of dimension 32. |
| B | A separately fit, one-level 64-entry quantizer at four temporal positions. |
| C | Observed 1→2→4 multiscale residual codes, summed into a four-position reconstruction. |
| D | The existing-size learned 1→2→4 next-level predictor, followed by the same ranking head. |

B is a standalone fine-resolution quantizer of full acoustic features. It does not misuse residual fine codes that depend on omitted coarse context. C includes the intermediate scale as part of its coarse-to-fine information; it uses exactly the same acoustic tokenizer as D. Its codes are computed from observed input features and are not oracle clean codes or labels. D instead learns to predict that hierarchy from the same input, using generated earlier embeddings and parallel within-level prediction.

A/B/C heads have 8,906 trainable parameters; D remains 25,834, unchanged from APPT. Ranking heads start from identical weights within each seed. All see the same four-position 32-dimensional acoustic evidence, labels and shuffled training data. Quantizers are trained without labels and frozen. D retains its existing mean three-prefix digit loss plus 0.1 mean code CE; A/B/C train their final local digit CE. D-versus-C therefore tests the learned prediction/training mechanism jointly, not only one abstract factorization.

## Training budget controls

Two predetermined comparisons are reported: equal 16-epoch optimizer-update budgets, and approximately matched profiled training FLOPs. The latter gives the cheaper models more passes over the same fixed training data. No test score selects epochs, models or temperatures. Per-variant temperatures use only the unchanged gate’s validation fitting half.

| Variant | Profiled training FLOPs per batch 128 | Equal-update epochs | FLOP-matched final epoch touched |
|---|---:|---:|---:|
| A | 4,685,824 | 16 | 239 |
| B | 6,930,432 | 16 | 162 |
| C | 8,663,040 | 16 | 130 |
| D | 69,911,379 | 16 | 16 |

The common target is approximately 41.95 billion profiled training FLOPs per model/seed, with stopping within one minibatch. The profiler captures supported PyTorch operation FLOPs, not all distance, indexing, softmax, memory or CPU costs; this is approximate FLOP matching, not exact wall-clock/FLOP equality. Longer A/B/C exposure can affect optimization, so equal-update results are retained. Recorded training_seconds_so_far includes intervening checkpoint evaluation/probes and must not be interpreted as pure training time. Gate/frontend inference costs are common and reported separately through full-pipeline latency.

## Corrupted test: matched profiled-FLOP comparison

| Variant | Digit accuracy | Sequence top-1 | Sequence top-5 | Digit ECE | Wrong→correct / correct→wrong | CPU p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| A: Continuous local reranker | 90.56 ± 1.63% | 73.09 ± 2.47% | 82.84 ± 3.44% | 0.0417 | 113.33 / 8.33 | 1.08 / 1.28 |
| B: Flat fine-only acoustic VQ | 84.72 ± 1.21% | 67.41 ± 0.74% | 74.20 ± 1.40% | 0.0476 | 61.67 / 19.67 | 1.10 / 1.34 |
| C: Observed multiscale acoustic VQ | 87.59 ± 1.30% | 68.64 ± 1.71% | 78.02 ± 1.83% | 0.0445 | 86.67 / 13.67 | 1.17 / 1.45 |
| D: Learned NCM-inspired refinement | 87.84 ± 1.10% | 69.38 ± 2.11% | 76.79 ± 1.40% | 0.0555 | 88.67 / 13.00 | 1.18 / 1.43 |

Values are means ± sample standard deviations across the three trained seeds. Correction counts are per 1,080 test digit positions. Latency includes the common CNN/frontend, local projection, verifier, both frozen-policy gate decisions, selected variant scoring and exact candidate ranking. Three warmup calls, one CPU thread; loading, labels and offline diagnostic probes are excluded. Percentiles are averaged across seeds, not pooled. These timings are for the common frozen conventional-reference policy, which decides acceptance before variant scoring; they are not directly comparable with earlier live APPT gates that needed an APPT proposal before accepting it.

## Equal-update sensitivity

| Variant | Digit accuracy | Sequence top-1 | Sequence top-5 |
|---|---:|---:|---:|
| A | 89.38% | 71.23% | 81.23% |
| B | 84.23% | 66.91% | 73.95% |
| C | 87.07% | 68.27% | 76.91% |
| D | 87.84% | 69.38% | 76.79% |

The conclusion that A is the strongest practical ranking option is not dependent on giving it the larger matched-FLOP update count.

## Candidate coverage and clean controls

| Seed | Expanded positions | Score-overridden positions | Candidate leaves | Oracle candidate-support coverage |
|---|---:|---:|---:|---:|
| 317 | 3.90 | 1.90 | 9066.7 | 92.96% |
| 419 | 3.94 | 0.48 | 9433.3 | 95.19% |
| 523 | 1.63 | 0.49 | 2430.9 | 89.26% |

Every number and every branch list in this table is identical across A–D within its seed. Oracle support coverage asks only whether the true full sequence is present anywhere in the fixed candidate universe. Top-5 coverage additionally measures ranking. None of A–D can improve their available candidate support; their differences are score-ordering differences within identical sets.

All four protected systems retain 100% clean digit and sequence accuracy. This does not demonstrate that every raw refiner recognizes clean audio: the frozen gate rejects all score overrides on clean samples, leaving the corrected CNN in control. All-event ungated candidate accuracy is saved separately in metrics.json. Clean protection is a property of the common frozen policy here, not a hierarchy achievement.

## Statistical uncertainty

Paired percentile 95% intervals use 2,000 bootstrap resamples of distinct digit-combination clusters, retaining all acoustic renderings and averaging outcomes over the same three trained seeds. Seed standard deviations are separate. Intervals condition on these models and the reused development pool; they are not independent confirmation on new audio.

| Matched-FLOP top-1 comparison | Delta | Paired cluster-bootstrap 95% interval |
|---|---:|---:|
| D-A | -3.70 points | [-5.58, -1.87] |
| D-B | 1.98 points | [0.36, 3.70] |
| D-C | 0.74 points | [-0.62, 2.09] |
| C-B | 1.23 points | [-0.37, 2.83] |

Intervals for equal-update and clean comparisons are also saved in summary.json. More test reuse, seeds or percentile precision cannot substitute for independent external validation.

## What information the hierarchy contributes

No variant receives new test observations. Its code hierarchy is a deterministic function of the same fixed local acoustic features and learned parameters. It can reorganize/compress that evidence or make a decision function easier to learn, but cannot create independent acoustic information or recover an erased independent digit from no evidence.

To distinguish representation quality from ranking-head capacity, a diagnostic linear probe (128 inputs, ten outputs) is trained on each frozen representation using the same 600 updates and training labels. These probes are not deployed, do not change candidate/gate lists, and are evaluated on all corrupted events. Feature MSE measures distortion from the shared projected acoustic input.

| Variant | Feature reconstruction MSE | Linear-probe digit accuracy |
|---|---:|---:|
| A | 0.0000 | 91.20% |
| B | 0.3792 | 81.11% |
| C | 0.2497 | 86.57% |
| D | 0.9062 | 88.89% |

C preserves more of the input than B and gives more accessible class information. However, C uses three codebooks and seven code indices per sound, while B uses one codebook and four indices: a C-versus-B gain alone cannot distinguish a multiscale benefit from additional quantization capacity/bitrate. D-versus-C holds the acoustic tokenizer fixed and tests learned next-level generation. Any change in linear decodability is a nonlinear remapping of the input plus learned training priors, not new observed evidence.

## Component conclusions

- A: continuous local features and a simple reranker remain the strongest baseline in this controlled task.
- B: flat fine-only quantization loses information and does not improve the continuous-feature baseline.
- C: extra residual scales improve acoustic reconstruction relative to B, but do not establish an advantage over A; capacity/bitrate is a confound for attributing C-versus-B gains specifically to scale.
- D: it beats B by 1.98 top-1 points within this pool (paired interval 0.36–3.70), but does not clearly beat C (0.74 points, interval −0.62–2.09) and loses to A. Its top-5 coverage is also below C. This does not establish an NCM-specific advantage over the strong simpler alternatives.

This is evidence about this small local MLP NCM-inspired adaptation on segmented DTMF, not a claim about every NCM architecture or longer overlapping sounds. The result does not justify further model-size, sampling-count or gate tuning on this pool.

## Verification and reproduction

27 tests pass. New checks verify that B is independent of coarse residual books, C equals observed cumulative reconstruction with frozen books, D uses the shared head initialization, and score reranking cannot alter frozen candidate support. Full-pipeline evaluation checks every reference route/accept mask against its saved version. Prior checkpoints are untouched.

```bash
PYTHONPATH=. .venv/bin/python experiments/hierarchy_contribution.py
# Optional saved-checkpoint full-pipeline re-evaluation, without retraining:
PYTHONPATH=. .venv/bin/python experiments/retime_hierarchy.py
.venv/bin/python experiments/report_hierarchy.py
.venv/bin/python -m pytest -q
```

Configuration: experiments/hierarchy_config.json. Frozen per-position candidate choices and acceptance masks: hierarchy_results/seed_*/clean_branches.json and corrupted_branches.json. Full raw metrics, calibration, paired intervals, linear probes, checkpoints, traces and source hashes are preserved under hierarchy_results/. This remains a three-seed developmental synthetic experiment with supplied event boundaries, selection conditional on the conventional reference gate and a repeatedly inspected test pool.
