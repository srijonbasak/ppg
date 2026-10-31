# Measured comparison

Mean ± sample standard deviation across three training seeds; same 120 test sequences.

| Model | Digit accuracy | Sequence top-1 | Sequence top-5 | Sequence NLL | CPU p50 / p95 ms |
|---|---:|---:|---:|---:|---:|
| CNN | 100.00 ± 0.00% | 100.00 ± 0.00% | 100.00 ± 0.00% | 0.03 ± 0.00 | 0.80 / 0.88 |
| PPF M=1 | 11.53 ± 0.94% | 0.00 ± 0.00% | 0.00 ± 0.00% | 9.22 ± 0.01 | 3.45 / 3.76 |
| PPF M=4 | 11.94 ± 0.73% | 0.00 ± 0.00% | 0.00 ± 0.00% | 9.21 ± 0.01 | 8.88 / 9.92 |
| PPF M=8 | 12.22 ± 0.52% | 0.00 ± 0.00% | 0.00 ± 0.00% | 9.21 ± 0.01 | 16.25 / 17.29 |
| PPF M=16 | 12.01 ± 0.73% | 0.00 ± 0.00% | 0.00 ± 0.00% | 9.21 ± 0.01 | 30.32 / 31.33 |

Latency columns average per-run percentiles; they are not pooled percentiles. GPU unavailable.

## Reconstruction and level diagnostics

These concern the existing audio tokenizer, not the specified clean-label tokenizer. Greedy generated-prefix accuracy is diagnostic; the main comparison uses stochastic sampling.

- Seed 7: teacher digit accuracy by level [0.12708333333333333, 0.12708333333333333, 0.125]; latent MSE [4245.899104817709, 4239.785302734375, 4233.57177734375]; greedy generated accuracy [0.12708333333333333, 0.12708333333333333, 0.12708333333333333]; unique codes [3, 3, 4].
- Seed 17: teacher digit accuracy by level [0.12708333333333333, 0.12708333333333333, 0.1125]; latent MSE [3643.6063151041667, 3638.07802734375, 3632.573795572917]; greedy generated accuracy [0.12708333333333333, 0.12291666666666666, 0.12291666666666666]; unique codes [2, 2, 4].
- Seed 27: teacher digit accuracy by level [0.12083333333333333, 0.11666666666666667, 0.11666666666666667]; latent MSE [3648.41748046875, 3641.6673502604167, 3634.9373860677083]; greedy generated accuracy [0.12083333333333333, 0.11875, 0.11875]; unique codes [2, 2, 4].

The current PPF did not show a benefit in this fixed short training protocol. This is not evidence against the finalized architecture, which is not implemented. CNN reached a ceiling on this synthetic distribution. Training logs show severe PPF loss growth; causes have not been isolated. No hyperparameters were changed after seeing test results.

Full numerical outputs: comparison.json and diagnostics.json. Scope, audit findings, reproducibility and limitations: AUDIT.md.
