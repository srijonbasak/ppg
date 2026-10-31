# Adaptive Parallel Possibility Tree — 2026-10-31

## Outcome

A separate local APPT prototype is implemented. It preserves the working CNN encoder/head and all earlier PPF checkpoints. It does not predict complete-sequence label codes. Local refinement improves recognition over the existing CNN on the corrupted development test, but the experiment does not establish an NCM-specific advantage over conventional local reranking. Ideal-clean and distribution-shift controls fail badly and are reported below.

## Research and tokenizer location

[NCM, sections 3.1–3.4](https://arxiv.org/html/2610.14162v1) describes residual discrete hierarchies and learned next-level prediction, with code positions within a level predicted in parallel. [Its repository](https://github.com/HazyResearch/ncm) implements a two-stage tokenizer/generator framework. [SoundStream](https://arxiv.org/abs/2107.03312) demonstrates residual quantization of encoded audio representations. These sources motivate a local acoustic experiment; they do not establish the best tokenizer placement for DTMF or prove this APPT design.

| Placement | Decision for this prototype |
|---|---|
| Raw local waveform | Valid future option, but requires a new audio encoder/decoder and broader reconstruction training. |
| Frozen CNN features for each sound segment | Selected: retains the existing acoustic frontend, avoids cross-digit label codes, and shares acoustic work across all ten candidates. |
| Candidate-conditioned latent representation | Not selected initially: adds candidate-specific computation and a risk of scoring a candidate primarily from its own embedding. Requires its own evidence-grounding controls. |

The selected placement is an engineering hypothesis, not a literature-established optimum. The conventional control uses exactly the same continuous local features, so it controls for using a richer local head rather than attributing every gain to quantization.

## Implemented architecture

1. Four supplied tone windows, each 2,400 samples, pass through the unchanged CNN. Its original head supplies ten digit probabilities per position.
2. Each local CNN stream [64,31] is average-pooled into four temporal feature vectors. A training-only standardized PCA projection reduces each vector to 32 dimensions. No complete-sequence label tokenizer is used.
3. An EMA residual acoustic tokenizer quantizes local features at 1→2→4 positions. At each level it pools the remaining feature residual, quantizes it and adds the upsampled code contribution. Random prefix cutoffs update finer books less often during tokenizer fitting. The projection and codebooks are then frozen.
4. A small learned predictor generates the local code levels sequentially, using audio and previous-level reconstruction; all positions within a level are produced together. Hard argmax inference uses generated earlier codes. Straight-through hard embeddings provide gradients during supervised refiner training.
5. A shared local ranker maps each accumulated acoustic hierarchy to ten digit-candidate scores in parallel. The acoustic hierarchy is shared by all ten candidates; this version does not generate a separate code hierarchy for each candidate.
6. A calibrated confidence detector selects uncertain positions. A separately trained, cheap continuous-feature verifier runs on all positions and can reopen a confident position if it disagrees with confidence at least 0.8. Its computation is included in adaptive-method latency.
7. Each selected position contributes all ten digits to the tree. Unselected digits are provisional CNN argmax choices, re-evaluated on every call. Candidate sequences are scored by products of per-position acoustic probabilities. There are no learned arbitrary four-digit patterns.

Important deviations: the level predictor is an MLP, not the NCM paper’s Transformer; the learned hierarchy describes one observed sound rather than text or a complete digit sequence; the acoustic feature encoder is frozen with PCA rather than a jointly trained waveform VQ-VAE. Tokenizer targets are the observed, possibly corrupted acoustic features, not paired clean targets. This is not a demonstrated denoising generative model or an official NCM implementation. Branching itself is ordinary candidate search, not an NCM-specific mechanism.

## Data, calibration and controls

Same 1,200 training, 184 validation and 270 test recordings used by the prior compositional experiment. Full test combinations are absent from training and validation; all position-specific digits and pairs are familiar. Test contains 266 distinct combinations. Tone boundaries are supplied, not detected. No test labels enter prediction; labels are used only for training, validation selection and evaluation.

Local training uses 4,800 individual sound events, not whole-sequence labels. All supervised variants use the same 16 epochs, batch 128, AdamW rate 0.001 and fixed final checkpoint. The 8,906-parameter continuous reranker is smaller than the 25,834-parameter APPT refiner. The no-prior ablation has the same stored parameter architecture as APPT but does not use its prior projection. Training compute is not matched across these sizes.

The acoustic tokenizer is fit without digit labels for ten epochs. APPT trains mean digit CE over all three generated local prefixes plus 0.1 times mean code CE. The no-prior ablation uses the same targets, losses, data order and initialization, but predicts each level without earlier embeddings. It therefore tests prior-level conditioning, not every possible hierarchy mechanism or quantization benefit.

Validation-only temperature fits: {'cnn': 0.7079457640647888, 'local': 0.7079457640647888, 'appt': 1.0, 'no_prior': 1.035142183303833}. The confidence cutoff 0.859576 targets 90% validation CNN-error recall. The independent-verifier cutoff is fixed at 0.8. These do not guarantee calibrated probabilities or 90% recall under test distribution shifts.

Perfectly zero waveform windows use a common explicit no-evidence rule: exactly uniform probability over all ten digits, always expanded and not sent through the refinement network. This applies to every method, including CNN. It is not a learned NCM capability. Partially erased signals retain acoustic evidence and do not use this override.

## Acoustic tokenizer diagnostics

Test feature reconstruction MSE after levels 1/2/3: [0.615005, 0.408851, 0.232998]. Test code usage: [63, 63, 64] of 64 per level. MSE refers to normalized projected CNN features, not waveform reconstruction or recognition accuracy.

| Refiner | Generated code accuracy, levels 1 / 2 / 3 | Digit accuracy after prefixes 1 / 2 / 3 |
|---|---|---|
| appt | 18.33% / 69.68% / 65.16% | 88.80% / 90.19% / 90.46% |
| no_prior | 7.78% / 64.91% / 64.42% | 88.33% / 89.91% / 89.54% |

These diagnostics run the refiner on every test event, before adaptive gating. Exact acoustic-code identity and digit identity are different metrics; low coarse canonical-code accuracy does not imply that all acoustic digit information is absent.

## Corrupted test comparison

| Method | Digit accuracy | Sequence top-1 | Sequence top-5 | Digit ECE | CPU p50 / p95 ms | Mean candidate leaves |
|---|---:|---:|---:|---:|---:|---:|
| CNN full top-K | 79.91% | 59.26% | 74.44% | 0.0481 | 0.89 / 0.95 | 10000.0 |
| CNN adaptive search | 79.91% | 59.26% | 74.44% | 0.0481 | 0.81 / 0.97 | 1182.8 |
| Conventional local reranker | 91.67% | 75.19% | 88.15% | 0.0208 | 0.84 / 1.06 | 1182.8 |
| NCM-inspired APPT | 90.09% | 75.93% | 84.07% | 0.0219 | 1.09 / 1.27 | 1182.8 |
| APPT without prior-level conditioning | 88.89% | 73.33% | 82.22% | 0.0285 | 1.07 / 1.24 | 1182.8 |

APPT top-1 is 205/270 versus conventional reranking 203/270 and CNN 160/270. APPT top-5 is worse than the conventional reranker (227/270 versus 238/270). APPT also has lower digit accuracy and higher NLL than the conventional control. Thus the results do not establish a useful NCM-specific advantage.

| Adaptive search measure | Result on corrupted test |
|---|---:|
| CNN errors selected by confidence alone | 81.57% |
| CNN errors selected after verifier reopening | 84.79% |
| Average positions refined per four-digit sequence | 1.44 |
| Average candidate leaves | 1182.8 |
| Worst-case candidate leaves | 10000 |
| Mean retained probability mass in adaptive search | 0.9052 |

The average leaf count falls by about 88% relative to full enumeration, but APPT is slower than conventional reranking because it adds hierarchy prediction. Candidate count is not wall-clock compute. The four-digit Cartesian search is cheap enough that fewer leaves alone do not imply an efficiency gain.

## Corruption and robustness controls

| Scenario | Recordings | CNN sequence top-1 | Local reranker top-1 | APPT top-1 | No-prior APPT top-1 |
|---|---:|---:|---:|---:|---:|
| corrupted | 270 | 59.26% | 75.19% | 75.93% | 73.33% |
| noise | 68 | 64.71% | 94.12% | 94.12% | 91.18% |
| interference | 67 | 5.97% | 23.88% | 22.39% | 19.40% |
| partial_erasure | 67 | 67.16% | 83.58% | 86.57% | 85.07% |
| clean | 270 | 5.93% | 1.11% | 0.74% | 0.74% |
| two_erased | 270 | 0.74% | 0.37% | 0.37% | 0.37% |
| confident_error | 270 | 0.00% | 72.96% | 73.70% | 71.11% |
| confident_error_clean_ood | 270 | 0.00% | 1.11% | 0.74% | 0.37% |

The noise/interference/partial-erasure rows are subsets of the corrupted test; they are not additional independent recordings. All variants are evaluated on identical waveforms within each row.

Ideal-clean control: re-synthesized pure tones with no noise, jitter or clipping. The unchanged CNN reaches only 50% digit accuracy; the local reranker and APPT fall to 30.28%. This is a serious robustness failure, not evidence of success on clean audio. All 1,080 clean digits were independently verified correct by Goertzel. The failure is consistent with a change from the augmented training distribution, but its specific cause has not been isolated. The earlier tiny-run metadata also recorded weak initial clean CNN accuracy (53.91%); clean generalization was never established by the corrupted-task score.

Two-erased control: exactly two complete event waveforms are zeroed before the frontend. Every method returns exactly uniform probabilities at those positions (maximum deviation 0). Performance is below the 1% top-1 chance limit that would apply if the other two digits were known perfectly; here the other two digits also suffer from the ideal-clean CNN failure. No erased-digit recovery claim is made.

## Mistaken high-confidence predictions

An evaluation-only fault injector changes one CNN position to a deliberately wrong digit with probability 99.5%. Its true digit is used to create the fault, never passed to the inference system. The main confident_error control retains the original corrupted waveform, isolating the confidence fault from the clean distribution shift. A separate confident_error_clean_ood row combines both stressors.

| In-distribution injected error measure | Conventional reranker | APPT |
|---|---:|---:|
| Injected positions reopened | 88.89% | 88.89% |
| Injected positions recovered correctly | 87.41% | 85.19% |

Reopening is implemented but not guaranteed. When the verifier is also wrong, the tree can retain a mistaken confident choice. Under the combined clean-shift/confidence fault, APPT reopening falls to 69.26% and recovery to 19.26%. The conventional verifier’s success is not an NCM-specific capability.

## What the ablation establishes

APPT with prior-level conditioning reaches 75.93% sequence top-1 versus 73.33% without that conditioning in this single fixed run. The acoustic VQ hierarchy, digit head and three output levels remain in both. This controls one hierarchical dependency, but it does not isolate the benefit of residual VQ versus flat quantization, establish an advantage over continuous local classification, or provide independent repeated-seed evidence. The conventional control has better top-5, digit accuracy and speed. No NCM-specific benefit is claimed.

## Search semantics and API

For u selected positions, every digit remains available there, giving exactly 10^u complete candidate leaves. Unselected positions use provisional CNN argmax choices. The returned per-position distributions still contain probabilities for all ten digits, and each call can reopen a provisional choice. Restricted candidate support is not the entire posterior: retained_probability_mass reports what the search includes. Returned candidate probabilities are unconditional products of the local distributions; omitted mass is not silently renormalized away.

A saved example in example_prediction.json expands only position 2, returns ten candidate leaves, and includes four ten-way distributions plus ranked complete hypotheses. Reference labels in that example file are audit metadata, not inference inputs.

```python
from ppf.appt import load_appt
system = load_appt("appt_results")
result = system.predict(events, top_k=5)  # events: float tensor [4, 2400], 8 kHz
```

The existing UI remains attached to the old prototype. This research API requires four supplied event windows; automatic segmentation and a new APPT UI were not implemented.

## Reproduce, tests and limitations

```bash
PYTHONPATH=. .venv/bin/python experiments/appt_experiment.py
# Or re-evaluate saved models without retraining:
PYTHONPATH=. .venv/bin/python experiments/appt_experiment.py --evaluate-only
.venv/bin/python experiments/report_appt.py
.venv/bin/python -m pytest -q
```

All 20 tests pass, including the original pipelines, a ten-branch single-uncertainty example, provisional high-confidence reopening, local hierarchy shape/gradient consistency, frozen tokenizer state, and exact uniform erasure handling. The saved-model loader was exercised on a real test example. Original CNN and prior PPF checkpoint hashes are preserved.

Temperatures and gate selection use validation only. All test evaluations are developmental: one seed, a previously inspected synthetic pool, oracle segmentation and dependent variants of the same sequences. No confidence interval from independent repeated runs is claimed. Coarse code targets represent observed corrupted audio, not latent clean sounds. Conventional and hierarchical model parameter/FLOP budgets differ. The no-prior ablation still retains hierarchical VQ. Calibration collapses on the clean distribution shift despite good in-distribution ECE. Positive findings therefore do not support deployment or a general hierarchy-specific recognition claim.

Latency is measured with the actual audio frontend, calibration, verifier where applicable, selected local refinement and tree ranking, with three warmup calls. CNN full ranking skips PCA/verifier/refinement. All timings use one CPU thread and exclude file loading and injected-fault construction. Final timings were rerun without concurrent diagnostic workloads. Model state, calibration, source hashes, full per-scenario calibration/NLL/Brier/coverage metrics, tokenizer statistics and per-sample correctness traces are saved under appt_results/.
