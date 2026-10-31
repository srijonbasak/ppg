# Frozen decoder/hierarchy sensitivity and recognition-aware experiment

Three seeds, fixed previously used splits. Percentages are mean ± sample SD. Oracle interventions are diagnostics and are not deployable recognition results. Existing models/checkpoints were preserved.

## Findings

The frozen decoder is sensitive to code identity, especially level 1. In the original-joint model, injecting canonical L1 lowers raw digit accuracy from 88.95% to 83.49%; fixing the original suffix makes it fall further to 77.01%. Regenerating later levels partly compensates for the canonical coarse code. All canonical codes yield only 72.44%, versus 88.95% generated. This supports a learned code/decoder compatibility mismatch; canonical acoustic code fidelity is not the same objective as digit recognition.

The code-aligned model behaves differently: canonical L3 raises raw digit accuracy from 83.36% to 84.32%, and all canonical codes raise it to 86.14%. Its protected sequence top-1 rises from 66.91% to 68.64% with all oracle levels, still below the ordinary original-joint result (69.38%). Canonical fine-code errors therefore matter, but correcting them alone does not establish an NCM recognition advantage.

One recognition-aware variant was trained after each seed met the validation diagnostic threshold. Protected corrupted-test sequence top-1 is 69.14%, compared with original-joint 69.38% and code-aligned 66.91%. The difference from original-joint is −0.25 pp (95% paired interval −1.62 to +1.22); the difference from code-aligned is +2.22 pp (+0.49 to +4.18). Recognition-aware supervision recovers the code-aligned regression but does not improve on the original NCM. Its all-canonical raw accuracy falls to 55.86%, showing that the added ranking loss does not resolve canonical-code compatibility.

Validation selects full NCM weight in every seed for original-joint and recognition-aware. Code-aligned selects [1,.75,1], giving protected top-1 67.04% versus 66.91% at full weight. Thus this small fusion grid does not identify score overweighting as the main bottleneck. It cannot rule out better fusion strategies.

No NCM-specific recognition improvement is established. These are three-seed development results on reused held-out combinations; code substitutions are diagnostic interventions, and individual code correctness does not causally guarantee a better rank.

## Protocol and interpretation

- Frozen CNN, original four-token adapter, 1→2→4 acoustic VQ books, 34,570-parameter NCM architecture. Original-joint and original-aligned checkpoints are loaded unchanged. These codes describe the observed audio, not clean-label digit sequences.
- An individual oracle replaces exactly one acoustic-code level. All later levels are regenerated conditional on the actual earlier generated/injected codes; unused target levels never enter their queries. The all-oracle case injects all levels. Fixed-suffix controls replace one level but keep every other originally generated code fixed, separating direct decoder effects from downstream changes.
- Categorical perturbation changes only the first slot of one generated level to (index+1) modulo 64 and keeps other codes fixed. Index proximity has no semantic meaning: this measures one deterministic perturbation, not a code-distance curve or exhaustive sensitivity bound.
- Per-level latent MSE compares reconstructed prefixes with the observed unquantized adapter input. Code-vector MSE compares lookup embeddings to canonical lookup embeddings. These are acoustic representation metrics, not recognition objectives.
- Raw digit predictions enumerate all 10,000 four-digit sequences independently. Protected predictions keep the exact previous CNN, frozen gate/acceptance and candidate policies. Scores at accepted positions are replaced by NCM digit probabilities; there is no learned additive fusion weight in the current implementation. Candidate sequence scores multiply digit probabilities. Rejected positions retain CNN scores.
- Probability interpolation alpha in {0,.25,.5,.75,1} is a diagnostic fusion control. Alpha is selected by digit NLL using only the existing validation fitting half. Test results report that fixed validation-selected alpha. Thus alpha=0 uses CNN scores within the same candidate list, not a new CNN/candidate baseline.
- Training eligibility is determined before training from the other validation half: oracle L2/L3/all improves raw digit accuracy by at least 0.5 percentage points for either frozen model. This is a weak diagnostic signal, not evidence of deployable improvement. Seeds that qualify receive exactly one fixed recognition-aware variant, with no weight search or test tuning.
- Recognition-aware loss = 0.1 mean code CE + mean generated-prefix digit CE + 0.25 mean generated-prefix ranking loss. Ranking uses softplus(1+highest incorrect logit−correct logit), only at training positions accepted by the frozen audio-only gate. Target digits supervise losses only. Hard argmax codes are used forward, with straight-through soft-embedding gradients. Earlier levels are generated during both training and inference.
- Paired training data (1,200 corrupted recordings plus clean seed817), batch128, 16 epochs/1,200 updates, AdamW lr.001 and norm cap1 match the original experiment. Training begins from the same per-seed initialization rather than warm-starting a checkpoint. The new ranking term is the controlled change relative to original-joint. Temperatures use the fixed validation fitting half; oracle diagnostics reuse each model’s ordinary-inference temperature, without oracle-specific calibration.
- Test: corrupted270, clean270 (seed318), noise68 and partial-erasure67 subsets, row-only missing-frequency270 (seed818). Reused development test combinations limit confirmatory claims. No absent frequency information can be reconstructed by a code oracle.

## Validation decision

- Seed 317: train=True; {"original_joint": {"generated_digit_accuracy": 0.9633151888847351, "oracle_delta": {"oracle1": -0.004076063632965088, "oracle2": 0.0, "oracle_all": -0.05842387676239014}}, "original_aligned": {"generated_digit_accuracy": 0.9375, "oracle_delta": {"oracle1": -0.0013586878776550293, "oracle2": 0.0027173757553100586, "oracle_all": 0.00951087474822998}}}
- Seed 419: train=True; {"original_joint": {"generated_digit_accuracy": 0.9510869383811951, "oracle_delta": {"oracle1": -0.008152127265930176, "oracle2": -0.004076063632965088, "oracle_all": -0.1086956262588501}}, "original_aligned": {"generated_digit_accuracy": 0.9347826242446899, "oracle_delta": {"oracle1": -0.0013586878776550293, "oracle2": 0.00951087474822998, "oracle_all": 0.01086956262588501}}}
- Seed 523: train=True; {"original_joint": {"generated_digit_accuracy": 0.9633151888847351, "oracle_delta": {"oracle1": -0.008152127265930176, "oracle2": -0.0013586878776550293, "oracle_all": -0.08423912525177002}}, "original_aligned": {"generated_digit_accuracy": 0.9375, "oracle_delta": {"oracle1": -0.0013586878776550293, "oracle2": 0.008152186870574951, "oracle_all": 0.008152186870574951}}}

## Frozen/generated hierarchy interventions on corrupted audio

| Model / intervention | Raw digit % | Raw sequence top-1 % | Top-5 % | Correct-digit probability | Δ probability | Protected top-1 % | Top-5 % | ECE % | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original_joint / generated | 88.95 ± 0.23 | 71.98 ± 0.93 | 83.95 ± 1.67 | 0.86 ± 0.00 | 0.00 ± 0.00 | 69.38 ± 0.77 | 78.27 ± 1.54 | 4.35 ± 2.05 | 0.46 ± 0.12 |
| original_joint / oracle0 | 83.49 ± 1.77 | 63.33 ± 4.86 | 77.41 ± 1.70 | 0.79 ± 0.01 | -0.08 ± 0.01 | 67.28 ± 0.57 | 75.68 ± 0.57 | 3.74 ± 1.63 | 0.55 ± 0.09 |
| original_joint / oracle1 | 87.01 ± 1.33 | 69.63 ± 1.11 | 79.14 ± 3.32 | 0.84 ± 0.01 | -0.02 ± 0.01 | 68.64 ± 1.13 | 76.30 ± 1.34 | 4.63 ± 2.13 | 0.50 ± 0.12 |
| original_joint / oracle2 | 88.70 ± 0.19 | 71.11 ± 1.61 | 83.70 ± 0.37 | 0.86 ± 0.00 | -0.01 ± 0.00 | 69.01 ± 1.07 | 77.41 ± 1.34 | 4.44 ± 2.87 | 0.47 ± 0.11 |
| original_joint / oracle_all | 72.44 ± 4.48 | 43.95 ± 8.07 | 67.28 ± 5.50 | 0.67 ± 0.03 | -0.20 ± 0.03 | 64.69 ± 1.19 | 72.35 ± 0.93 | 5.30 ± 1.54 | 0.69 ± 0.08 |
| original_aligned / generated | 83.36 ± 1.52 | 63.46 ± 1.50 | 76.17 ± 2.60 | 0.80 ± 0.01 | 0.00 ± 0.00 | 66.91 ± 1.19 | 75.06 ± 1.19 | 3.77 ± 1.30 | 0.51 ± 0.08 |
| original_aligned / oracle0 | 83.83 ± 2.13 | 64.81 ± 1.70 | 77.41 ± 3.87 | 0.81 ± 0.01 | 0.00 ± 0.01 | 66.79 ± 1.13 | 75.43 ± 2.04 | 3.64 ± 1.22 | 0.50 ± 0.09 |
| original_aligned / oracle1 | 83.86 ± 1.07 | 63.46 ± 1.90 | 77.65 ± 2.38 | 0.81 ± 0.01 | 0.00 ± 0.00 | 66.91 ± 0.93 | 75.31 ± 1.13 | 3.68 ± 1.39 | 0.50 ± 0.08 |
| original_aligned / oracle2 | 84.32 ± 0.84 | 64.32 ± 2.04 | 78.15 ± 1.61 | 0.81 ± 0.01 | 0.00 ± 0.00 | 67.41 ± 0.98 | 75.80 ± 0.77 | 3.86 ± 1.79 | 0.50 ± 0.08 |
| original_aligned / oracle_all | 86.14 ± 0.96 | 67.90 ± 3.15 | 82.47 ± 0.86 | 0.82 ± 0.01 | 0.02 ± 0.00 | 68.64 ± 1.90 | 76.91 ± 0.43 | 3.96 ± 1.62 | 0.47 ± 0.10 |
| recognition / generated | 87.62 ± 1.26 | 67.53 ± 1.83 | 82.35 ± 1.86 | 0.85 ± 0.01 | 0.00 ± 0.00 | 69.14 ± 1.54 | 77.78 ± 1.70 | 4.79 ± 2.33 | 0.47 ± 0.11 |
| recognition / oracle0 | 80.09 ± 4.50 | 55.19 ± 13.92 | 72.72 ± 6.72 | 0.72 ± 0.08 | -0.13 ± 0.08 | 67.65 ± 1.54 | 75.19 ± 1.11 | 4.62 ± 2.82 | 0.55 ± 0.07 |
| recognition / oracle1 | 86.08 ± 0.96 | 65.06 ± 2.17 | 78.64 ± 3.60 | 0.83 ± 0.01 | -0.02 ± 0.01 | 68.52 ± 2.06 | 76.79 ± 1.50 | 4.95 ± 2.51 | 0.50 ± 0.10 |
| recognition / oracle2 | 86.94 ± 1.20 | 66.17 ± 2.68 | 80.86 ± 1.30 | 0.84 ± 0.01 | -0.01 ± 0.00 | 68.89 ± 2.25 | 76.91 ± 1.19 | 4.74 ± 1.93 | 0.48 ± 0.11 |
| recognition / oracle_all | 55.86 ± 17.94 | 21.85 ± 18.31 | 39.01 ± 26.81 | 0.49 ± 0.16 | -0.36 ± 0.16 | 61.23 ± 4.60 | 69.88 ± 3.42 | 5.49 ± 0.58 | 0.77 ± 0.12 |

## Direct decoder versus downstream sensitivity

| Model / fixed replacement or perturbation | Raw digit % | Δ correct probability | Repairs | Regressions |
|---|---:|---:|---:|---:|
| original_joint / fixed0 | 77.01 ± 2.29 | -0.14 ± 0.02 | 22.00 ± 9.17 | 151.00 ± 25.12 |
| original_joint / fixed1 | 86.76 ± 1.03 | -0.02 ± 0.01 | 13.00 ± 1.00 | 36.67 ± 11.02 |
| original_joint / fixed2 | 88.70 ± 0.19 | -0.01 ± 0.00 | 10.67 ± 0.58 | 13.33 ± 4.16 |
| original_joint / perturb0 | 20.62 ± 10.12 | -0.68 ± 0.05 | 20.67 ± 5.13 | 758.67 ± 105.19 |
| original_joint / perturb1 | 84.51 ± 2.29 | -0.09 ± 0.02 | 15.67 ± 2.08 | 63.67 ± 24.01 |
| original_joint / perturb2 | 88.40 ± 0.65 | -0.00 ± 0.00 | 8.67 ± 2.08 | 14.67 ± 4.16 |
| original_aligned / fixed0 | 83.64 ± 2.00 | 0.00 ± 0.00 | 14.67 ± 4.16 | 11.67 ± 3.06 |
| original_aligned / fixed1 | 84.04 ± 1.11 | 0.00 ± 0.00 | 21.00 ± 7.55 | 13.67 ± 2.08 |
| original_aligned / fixed2 | 84.32 ± 0.84 | 0.00 ± 0.00 | 29.33 ± 7.23 | 19.00 ± 5.00 |
| original_aligned / perturb0 | 19.01 ± 7.24 | -0.65 ± 0.03 | 24.00 ± 10.54 | 719.00 ± 70.15 |
| original_aligned / perturb1 | 77.65 ± 1.51 | -0.10 ± 0.02 | 35.33 ± 6.11 | 97.00 ± 7.94 |
| original_aligned / perturb2 | 82.99 ± 0.54 | -0.00 ± 0.00 | 24.00 ± 5.57 | 28.00 ± 9.64 |
| recognition / fixed0 | 69.48 ± 10.37 | -0.23 ± 0.11 | 28.67 ± 8.50 | 224.67 ± 110.02 |
| recognition / fixed1 | 86.42 ± 1.12 | -0.02 ± 0.01 | 18.67 ± 2.08 | 31.67 ± 2.52 |
| recognition / fixed2 | 86.94 ± 1.20 | -0.01 ± 0.00 | 12.67 ± 4.16 | 20.00 ± 3.61 |
| recognition / perturb0 | 24.72 ± 8.31 | -0.65 ± 0.07 | 26.33 ± 4.73 | 705.67 ± 91.66 |
| recognition / perturb1 | 86.20 ± 0.88 | -0.05 ± 0.02 | 24.33 ± 3.06 | 39.67 ± 8.50 |
| recognition / perturb2 | 87.01 ± 1.07 | -0.00 ± 0.00 | 9.67 ± 3.79 | 16.33 ± 6.11 |

## Target-free comparison across conditions

| Condition / model | Codes L1 / L2 / L3 % | Raw digit % | Protected digit % | Sequence top-1 % | Top-5 % | ECE % | NLL | Brier | CPU p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| corrupted / original_joint | 40.37 ± 6.83 / 65.15 ± 3.00 / 51.40 ± 0.36 | 88.95 ± 0.23 | 87.65 ± 0.60 | 69.38 ± 0.77 | 78.27 ± 1.54 | 4.35 ± 2.05 | 0.46 ± 0.12 | 0.20 ± 0.04 | 1.29 ± 0.44 / 1.96 ± 0.09 |
| corrupted / original_aligned | 92.13 ± 1.14 / 86.84 ± 1.48 / 67.31 ± 1.10 | 83.36 ± 1.52 | 85.28 ± 1.37 | 66.91 ± 1.19 | 75.06 ± 1.19 | 3.77 ± 1.30 | 0.51 ± 0.08 | 0.22 ± 0.03 | 1.31 ± 0.43 / 2.00 ± 0.10 |
| corrupted / recognition | 24.23 ± 13.42 / 52.72 ± 9.78 / 44.90 ± 5.00 | 87.62 ± 1.26 | 87.69 ± 1.30 | 69.14 ± 1.54 | 77.78 ± 1.70 | 4.79 ± 2.33 | 0.47 ± 0.11 | 0.20 ± 0.04 | 1.29 ± 0.44 / 2.05 ± 0.28 |
| clean / original_joint | 91.14 ± 9.08 / 100.00 ± 0.00 / 98.33 ± 0.22 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 3.99 ± 0.98 | 0.04 ± 0.01 | 0.01 ± 0.00 | 1.03 ± 0.11 / 1.43 ± 0.42 |
| clean / original_aligned | 100.00 ± 0.00 / 100.00 ± 0.00 / 98.33 ± 0.22 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 3.99 ± 0.97 | 0.04 ± 0.01 | 0.01 ± 0.00 | 1.04 ± 0.11 / 1.52 ± 0.40 |
| clean / recognition | 70.68 ± 26.30 / 98.60 ± 2.43 / 97.55 ± 1.17 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 3.99 ± 0.98 | 0.04 ± 0.01 | 0.01 ± 0.00 | 1.04 ± 0.12 / 1.44 ± 0.47 |
| noisy / original_joint | 30.64 ± 2.81 / 62.01 ± 6.10 / 39.77 ± 1.53 | 97.79 ± 0.64 | 97.43 ± 0.97 | 90.20 ± 4.25 | 98.04 ± 2.25 | 4.39 ± 1.38 | 0.16 ± 0.05 | 0.06 ± 0.02 | 1.35 ± 0.46 / 1.98 ± 0.19 |
| noisy / original_aligned | 91.42 ± 0.77 / 81.86 ± 1.97 / 52.30 ± 2.05 | 96.32 ± 0.74 | 97.18 ± 0.93 | 88.73 ± 3.70 | 96.57 ± 3.06 | 4.42 ± 2.54 | 0.18 ± 0.04 | 0.07 ± 0.02 | 1.35 ± 0.46 / 1.99 ± 0.19 |
| noisy / recognition | 10.29 ± 6.41 / 51.84 ± 12.29 / 32.69 ± 5.83 | 96.69 ± 0.37 | 96.69 ± 0.97 | 88.24 ± 3.89 | 98.04 ± 0.85 | 3.45 ± 2.47 | 0.16 ± 0.05 | 0.07 ± 0.03 | 1.36 ± 0.47 / 2.06 ± 0.29 |
| partial_erasure / original_joint | 44.28 ± 11.44 / 71.33 ± 1.55 / 59.45 ± 2.73 | 95.65 ± 0.43 | 93.66 ± 0.65 | 74.63 ± 2.59 | 84.58 ± 2.28 | 2.48 ± 0.29 | 0.22 ± 0.05 | 0.09 ± 0.01 | 1.02 ± 0.09 / 1.66 ± 0.41 |
| partial_erasure / original_aligned | 88.43 ± 2.24 / 83.52 ± 2.10 / 69.37 ± 1.38 | 91.04 ± 1.35 | 93.16 ± 0.22 | 72.64 ± 0.86 | 84.58 ± 2.28 | 2.46 ± 0.22 | 0.22 ± 0.01 | 0.10 ± 0.00 | 1.04 ± 0.11 / 1.76 ± 0.44 |
| partial_erasure / recognition | 29.23 ± 20.01 / 58.33 ± 8.72 / 54.48 ± 2.57 | 92.54 ± 0.75 | 93.78 ± 0.43 | 75.12 ± 1.72 | 84.08 ± 3.11 | 2.46 ± 0.22 | 0.22 ± 0.01 | 0.10 ± 0.00 | 1.03 ± 0.10 / 1.75 ± 0.27 |
| missing_frequency / original_joint | 19.97 ± 34.59 / 16.81 ± 18.21 / 16.15 ± 0.88 | 6.82 ± 5.96 | 16.91 ± 6.27 | 0.00 ± 0.00 | 0.37 ± 0.37 | 53.82 ± 1.57 | 4.40 ± 0.96 | 1.27 ± 0.03 | 1.76 ± 0.08 / 2.14 ± 0.10 |
| missing_frequency / original_aligned | 76.17 ± 12.38 / 58.53 ± 7.22 / 33.43 ± 8.01 | 20.34 ± 9.45 | 20.12 ± 9.79 | 0.12 ± 0.21 | 0.62 ± 1.07 | 43.53 ± 10.84 | 4.43 ± 1.64 | 1.20 ± 0.20 | 1.75 ± 0.08 / 2.09 ± 0.07 |
| missing_frequency / recognition | 10.46 ± 18.12 / 12.05 ± 10.71 / 13.01 ± 11.90 | 10.68 ± 1.66 | 17.96 ± 7.49 | 0.12 ± 0.21 | 0.37 ± 0.37 | 46.15 ± 15.35 | 4.87 ± 0.20 | 1.26 ± 0.09 | 1.78 ± 0.06 / 2.22 ± 0.13 |

Latency includes CNN, live gate, accepted-position NCM and candidate ranking, one CPU thread and one pass per recording. No oracle extraction or ground-truth labels enter the timed path. Noisy timing is not a rigorous speed comparison; reported p50/p95 are per-seed percentiles averaged across seeds.

## Score fusion control

| Model | Validation-selected alpha (seed order) | Corrupted top-1 % | Top-5 % | NLL |
|---|---|---:|---:|---:|
| original_joint | [1.0, 1.0, 1.0] | 69.38 ± 0.77 | 78.27 ± 1.54 | 0.46 ± 0.12 |
| original_aligned | [1.0, 0.75, 1.0] | 67.04 ± 0.98 | 75.43 ± 0.57 | 0.51 ± 0.08 |
| recognition | [1.0, 1.0, 1.0] | 69.14 ± 1.54 | 77.78 ± 1.70 | 0.47 ± 0.11 |

## Paired uncertainty

10,000 label-cluster bootstrap resamples over 266 unique corrupted-test sequence labels, averaging seed effects. Intervals condition on these trained seeds and do not include full training-population uncertainty.

- Recognition minus original_joint, protected top1: -0.25 pp, 95% interval [-1.62, +1.22].
- Recognition minus original_joint, protected top5: -0.49 pp, 95% interval [-1.87, +0.98].
- Recognition minus original_aligned, protected top1: +2.22 pp, 95% interval [+0.49, +4.18].
- Recognition minus original_aligned, protected top5: +2.72 pp, 95% interval [+0.99, +4.55].


## Prefix contribution and quantization detail

| Model | Generated prefix L1 / L1+L2 / all digit accuracy % | Canonical prefix acoustic MSE |
|---|---:|---:|
| original_joint | 87.10 ± 0.60 / 88.92 ± 0.35 / 88.95 ± 0.23 | 0.62 ± 0.00 / 0.41 ± 0.00 / 0.25 ± 0.00 |
| original_aligned | 78.95 ± 1.66 / 81.57 ± 1.09 / 83.36 ± 1.52 | 0.62 ± 0.00 / 0.41 ± 0.00 / 0.25 ± 0.00 |
| recognition | 86.30 ± 1.09 / 87.04 ± 1.29 / 87.62 ± 1.26 | 0.62 ± 0.00 / 0.41 ± 0.00 / 0.25 ± 0.00 |

These prefix accuracies show how the shared decoder uses progressively accumulated generated representations. They do not isolate a causal benefit over a matched flat model. Decreasing acoustic quantization error alone does not guarantee improved digit ranking.

| Model / intervention | Correct mean rank | Rank change (negative improves) | Correct-logit change | Code-vector MSE L1 / L2 / L3 |
|---|---:|---:|---:|---:|
| original_joint / generated | 1.20 ± 0.02 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.35 ± 0.08 / 0.16 ± 0.02 / 0.14 ± 0.01 |
| original_joint / oracle0 | 1.38 ± 0.07 | 0.18 ± 0.05 | -1.14 ± 0.21 | 0.00 ± 0.00 / 0.23 ± 0.04 / 0.14 ± 0.00 |
| original_joint / oracle1 | 1.24 ± 0.03 | 0.04 ± 0.01 | -0.43 ± 0.05 | 0.35 ± 0.08 / 0.00 ± 0.00 / 0.14 ± 0.01 |
| original_joint / oracle2 | 1.20 ± 0.01 | 0.00 ± 0.01 | -0.22 ± 0.05 | 0.35 ± 0.08 / 0.16 ± 0.02 / 0.00 ± 0.00 |
| original_joint / oracle_all | 1.70 ± 0.13 | 0.50 ± 0.11 | -2.32 ± 0.35 | 0.00 ± 0.00 / 0.00 ± 0.00 / 0.00 ± 0.00 |
| original_aligned / generated | 1.31 ± 0.03 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.02 ± 0.00 / 0.05 ± 0.00 / 0.09 ± 0.00 |
| original_aligned / oracle0 | 1.29 ± 0.06 | -0.02 ± 0.02 | 0.04 ± 0.03 | 0.00 ± 0.00 / 0.05 ± 0.00 / 0.09 ± 0.00 |
| original_aligned / oracle1 | 1.30 ± 0.02 | -0.02 ± 0.02 | -0.07 ± 0.06 | 0.02 ± 0.00 / 0.00 ± 0.00 / 0.09 ± 0.00 |
| original_aligned / oracle2 | 1.28 ± 0.01 | -0.03 ± 0.02 | -0.13 ± 0.01 | 0.02 ± 0.00 / 0.05 ± 0.00 / 0.00 ± 0.00 |
| original_aligned / oracle_all | 1.22 ± 0.02 | -0.10 ± 0.02 | -0.14 ± 0.03 | 0.00 ± 0.00 / 0.00 ± 0.00 / 0.00 ± 0.00 |
| recognition / generated | 1.23 ± 0.02 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.46 ± 0.10 / 0.26 ± 0.08 / 0.16 ± 0.02 |
| recognition / oracle0 | 1.45 ± 0.03 | 0.22 ± 0.01 | -1.64 ± 0.77 | 0.00 ± 0.00 / 0.34 ± 0.07 / 0.19 ± 0.04 |
| recognition / oracle1 | 1.29 ± 0.03 | 0.06 ± 0.01 | -0.72 ± 0.21 | 0.46 ± 0.10 / 0.00 ± 0.00 / 0.17 ± 0.02 |
| recognition / oracle2 | 1.25 ± 0.03 | 0.01 ± 0.00 | -0.27 ± 0.11 | 0.46 ± 0.10 / 0.26 ± 0.08 / 0.00 ± 0.00 |
| recognition / oracle_all | 2.37 ± 0.45 | 1.13 ± 0.44 | -3.42 ± 0.97 | 0.00 ± 0.00 / 0.00 ± 0.00 / 0.00 ± 0.00 |

## Reproduction and verification

```bash
PYTHONPATH=. .venv/bin/python experiments/decoder_sensitivity.py
PYTHONPATH=. .venv/bin/python experiments/decoder_detail.py
PYTHONPATH=. .venv/bin/python experiments/report_decoder.py
PYTHONPATH=. .venv/bin/python experiments/verify_decoder_results.py
PYTHONPATH=. .venv/bin/python -m pytest -q
```

46 tests passed. New checks verify unchanged generated inference, isolation of injected levels, fixed-suffix behavior, generated-path gradients and frozen codebooks. metrics.json contains all five-condition interventions, calibration, per-position digit scores, repairs/regressions, latent errors and fusion controls. detail.json contains corrupted-test correct-logit/rank/margin changes, code-vector quantization errors and coarse-correct/incorrect strata. These strata are observational, not causal effects. All oracle interventions are marked separately. Existing checkpoints are never overwritten.
