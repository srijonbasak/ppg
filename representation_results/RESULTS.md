# Staged experiment results — 2026-10-31

## Clean-label reconstruction gate

Gate passed: True. Selected epoch: 1 by validation full-sequence accuracy (earliest tie). Validation and test each contain 1,000 digit combinations excluded from tokenizer training.

| Code prefix | Test digit reconstruction | Test sequence reconstruction | Used codes / 64 |
|---|---:|---:|---:|
| Through 1 positions | 32.17% | 0.20% | 43 |
| Through 2 positions | 55.47% | 3.50% | 61 |
| Through 4 positions | 100.00% | 100.00% | 63 |

The initial fixed-rate run failed the final-checkpoint validation gate (98.8%). It is preserved. The revised validation-selection/cosine-decay run passed. Because the best checkpoint was epoch 1, the results do not establish that cosine decay caused the gain. EMA, normalization, initialization and dropout were not separately ablated. Later epochs remained recorded; full reconstruction alone was the selection criterion.

## Controlled recognition comparison

One seed; 300 identical test recordings; 75 per condition. Four oracle-separated event windows per recording. PPF M=4, with no increase in sample count or Transformer depth.

| Model | Digit accuracy¹ | Sequence top-1 | Sequence top-5 | Sequence NLL | CPU p50 / p95 ms |
|---|---:|---:|---:|---:|---:|
| CNN | 79.83% | 58.67% | 74.67% | 2.757 | 0.96 / 1.12 |
| PPF | 14.42% | 0.00% | 0.00% | 11.159 | 3.37 / 3.49 |

¹ Digit accuracy is evaluated from the highest-ranked complete sequence, rather than independently marginalized digit predictions. For the CNN factorized baseline these coincide.

| Condition | CNN sequence top-1 / top-5 | PPF sequence top-1 / top-5 |
|---|---:|---:|
| reference | 98.67% / 100.00% | 0.00% / 0.00% |
| noise | 64.00% / 92.00% | 0.00% / 0.00% |
| interference | 5.33% / 9.33% | 0.00% / 0.00% |
| partial_erasure | 66.67% / 97.33% | 0.00% / 0.00% |

## Generator diagnosis

| Level positions | Teacher-forced code accuracy | Generated-earlier-level code accuracy² |
|---|---:|---:|---:|
| 1 | 10.33% | 10.33% |
| 2 | 13.33% | 5.83% |
| 4 | 38.50% | 5.17% |

² Greedy rollout, separate from stochastic mixture recognition. Each level is compared with canonical target-tokenizer codes; another code hierarchy can sometimes decode the same digits, so code accuracy is diagnostic rather than equivalent to recognition accuracy.

The tokenizer and decoder state remained exactly unchanged throughout conditional training. Code prediction loss decreased from 4.099 to 2.720, so the earlier exploding-loss behavior was not reproduced. Recognition remains poor: reliable clean-label reconstruction has not translated into reliable audio-to-code prediction. The large teacher-versus-generated gap at finer levels supports investigating conditional prediction and sensitivity to earlier mistakes; it does not isolate a cause.

## Verification and limitations

All 9 tests pass, covering the original pipeline, evaluator generation path, 1/2/4 shapes, encoder gradients, EMA buffer behavior, frozen target state, target-free generation, generated-prior propagation, and exact joint-mixture mass aggregation. The reconstruction gate was respected: generator training was deferred after the first failed tokenizer run.

No recognition advantage is demonstrated. This is a single-seed development experiment. PPF receives extra training and starts from the trained CNN encoder; compute is not matched. No automatic hyperparameter selection used the audio test set. Both splits use the same corruption families; OOD and real-recording evaluation remain untested. Partial erasure leaves evidence and is not a complete-erasure control. GPU memory is not applicable on this CPU host. Latencies exclude file loading and oracle segmentation.

Next diagnostic work should focus on conditional-code learning and generated-prefix sensitivity, not larger M or more Transformer layers. Detailed protocols and implemented mechanisms: README.md. Full metrics: tokenizer_metrics.json and comparison.json. Historical failed run: tokenizer_initial_failed.json.
