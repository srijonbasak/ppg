# Graph-conditioned NCM possibility refinement — 2026-10-31

## Outcome

A minimal local acoustic graph and an actual graph-conditioned NCM Transformer are implemented, with NCM retained as the default research inference path. The combination does not outperform graph-only reranking, and the physical graph does not clearly beat its rewired learned control. This experiment does not establish a graph-specific or NCM-specific recognition benefit.

## Minimal graph and acoustic evidence

Each graph describes one digit sound, not an entire four-digit sequence. It has 42 nodes: four local CNN-time nodes, 28 time/frequency evidence nodes (four windows × seven frequencies), and ten possible digit identities. There are no learned digit-sequence edges or co-occurrence assumptions.

Frequency nodes measure log spectral peak energy around the DTMF bands 697, 770, 852, 941, 1209, 1336 and 1477 Hz, using ±2.5% bands and normalized local power. The excluded 1633 Hz column belongs to symbols outside this experiment’s 0–9 alphabet. Time nodes hold frozen CNN acoustic features. Same-frequency edges join neighboring local windows; fixed adjacent-window weights encode continuity. Digit nodes connect only to the two frequencies in the standard DTMF definition.

Fixed message aggregation takes the minimum of the two incident log-frequency supports, so one strong frequency alone does not provide strong pair support. A training-only normalized PCA projects local CNN features, measured frequencies and candidate-support messages into four 32-dimensional vectors. This is a small graph-derived vector embedding with learned unsupervised projection, not a trainable GNN. A large GNN was deliberately not introduced before this minimal relationship test.

| Digit | Frequency pair, Hz |
|---|---|
| 0 | 941, 1336 |
| 1 | 697, 1209 |
| 2 | 697, 1336 |
| 3 | 697, 1477 |
| 4 | 770, 1209 |
| 5 | 770, 1336 |
| 6 | 770, 1477 |
| 7 | 852, 1209 |
| 8 | 852, 1336 |
| 9 | 852, 1477 |

All edges come from known acoustic definitions. Correct sample labels, corruption flags and sample IDs are absent from graph construction and prediction. Class identity nodes are the fixed candidate vocabulary, not the correct answers.

## Placement of NCM and controls

The default D path is waveform → frozen corrected CNN → local acoustic graph → graph-conditioned vectors → multiscale residual VQ targets → conditional Transformer prediction → cumulative acoustic decoder → ten digit scores → the unchanged adaptive possibility tree. The quantizer represents graph-conditioned acoustic vectors, not whole-sequence digit labels.

Codebooks train without digit labels using EMA residual quantization at 1→2→4 positions and random nested-prefix update cutoffs. The acoustic feature projection and codebooks are frozen for conditional training. A one-block Transformer decoder (width 32, four heads, feed-forward width 128) predicts each level from acoustic memory and generated earlier embeddings. All positions within a level are produced in parallel. Training uses hard-forward straight-through embeddings, prefix digit CE and code CE. No true codes enter target-free prediction.

This replaces the prior APPT MLP level predictor only in a new matched Transformer experiment; all previous MLP checkpoints remain intact. C and D have identical 34,570-parameter Transformer/ranker architectures, differing in input/target representation. A/B rerankers have 8,906 trainable parameters. The graph aggregation has no supervised message weights; fitting a trainable GNN is outside this first experiment.

Primary variants: A is the original CNN-only continuous representation; B adds the physical graph; C uses the CNN-only acoustic representation with the Transformer; D combines the graph representation and Transformer. Additional graph-free frequency controls give the continuous reranker and Transformer the same seven explicit spectral measurements, so graph gains cannot automatically be credited to additional input features. A degree-preserving rewiring control permutes high-frequency assignments across digit nodes. A D no-prior control keeps the graph tokenizer/Transformer architecture but removes earlier-level conditioning.

## Frozen protocol and budget

The CNN weights, relative log-Mel floor 0.001, 1,200/184/270 train/validation/test splits, gate weights, route/acceptance masks and per-position candidate lists are unchanged. The 270 test recordings contain 266 unique combinations. Identical branch-file hashes are checked for every variant within each seed. The preserved conventional reference gate determines both route and score-override positions; model confidences cannot change candidate support. Clean cases are protected by that policy.

Three seeds: 317, 419, 523. All ranking heads share their initialization within a seed. We report fixed 16-epoch equal-update results and approximately matched supported-operation supervised-training FLOPs. The latter gives each model about 91.45 billion profiled FLOPs; simple heads receive more updates on the same fixed examples. Temperatures use only the unchanged gate’s validation fitting half. Test scores select no model or checkpoint.

| Model family | Trainable parameters | Equal-update epochs | Matched-FLOP epochs touched |
|---|---:|---:|---:|
| Continuous A/B and rewired controls | 8,906 | 16 | 521 |
| CNN-only NCM Transformer | 34,570 | 16 | 16 |
| Graph-conditioned NCM Transformer | 34,570 | 16 | 16 |
| Graph Transformer, no prior | 34,570 | 16 | 17 |

Matching covers profiler-supported supervised training operations, not exact total runtime. PCA fitting and ten-epoch label-free EMA tokenizer fitting are separate preprocessing stages and are not included in this matched budget. Profiling omits some FFT/distance/indexing/softmax costs. The longer simple-model exposure and different supervised objectives are why equal-update results are also retained. These controls are stronger than equal epochs alone but are not perfectly compute/parameter matched end-to-end pipelines.

## Are the relationships useful before fitting a graph model?

| Simple physical pair-score check | Physical graph digit accuracy | Rewired assignments |
|---|---:|---:|
| corrupted | 89.72% | 22.59% |
| clean | 100.00% | 18.52% |

This validates acoustic measurement and the fixed DTMF mapping. It is a physics/template sanity check, not evidence that a learned graph embedding or NCM is superior. The learned rewired-graph control below is the relevant structure comparison after training.

## Corrupted test: matched profiled-training-FLOP results

| Variant | Digit accuracy | Sequence top-1 | Sequence top-5 | Digit ECE | CPU p50 / p95 ms |
|---|---:|---:|---:|---:|---:|
| A: CNN + continuous reranker | 90.34 ± 1.59% | 72.47 ± 1.71% | 82.22 ± 2.94% | 0.0398 | 1.38 / 1.55 |
| B: CNN + graph embedding | 91.23 ± 1.77% | 73.70 ± 3.21% | 84.07 ± 4.17% | 0.0449 | 1.38 / 1.53 |
| C: CNN + NCM Transformer | 87.01 ± 0.84% | 69.26 ± 1.61% | 77.53 ± 1.50% | 0.0491 | 1.61 / 2.23 |
| D: CNN + graph + NCM Transformer | 88.21 ± 0.33% | 69.88 ± 1.54% | 79.01 ± 1.30% | 0.0480 | 1.62 / 2.26 |

Means ± sample standard deviation across seeds. These are controlled frozen-policy workloads, not independently tuned gates for each model.

| Additional control | Sequence top-1 | Sequence top-5 |
|---|---:|---:|
| Continuous + matched frequency features | 72.47% | 83.21% |
| NCM + matched frequency features | 69.26% | 77.16% |
| Rewired graph + continuous reranker | 73.83% | 83.83% |
| Graph + NCM without prior-level conditioning | 70.25% | 79.26% |

## Equal-update sensitivity

| Variant | Sequence top-1 | Sequence top-5 |
|---|---:|---:|
| A: CNN + continuous reranker | 71.23% | 81.23% |
| B: CNN + graph embedding | 71.73% | 81.36% |
| C: CNN + NCM Transformer | 69.26% | 77.53% |
| D: CNN + graph + NCM Transformer | 69.88% | 79.01% |

## Paired statistical uncertainty

2,000 percentile bootstrap resamples group all renderings of the same digit combination, average across the three fixed trained seeds, and preserve pairings across variants. Intervals condition on these models and the reused development test; they are not independent external confirmation.

| Top-1 comparison | Delta | Paired cluster-bootstrap 95% interval |
|---|---:|---:|
| D-B | -3.83 points | [-5.98, -1.85] |
| D-C_cnn | 0.62 points | [-1.10, 2.23] |
| D-C | 0.62 points | [-0.87, 2.11] |
| B-A_cnn | 1.23 points | [0.12, 2.37] |
| B-A | 1.23 points | [0.12, 2.35] |
| B-rewired | -0.12 points | [-1.00, 0.74] |
| D-D_no_prior | -0.37 points | [-1.85, 1.00] |
| D-A_cnn | -2.59 points | [-4.55, -0.74] |

The combined D model does not beat graph-only B. Correct graph B does not establish an advantage over the trained rewired graph. The prior-conditioning ablation does not support crediting NCM’s next-level dependency. Raw physical pair relationships are meaningful, but the extra embedding/refinement machinery is not shown to use them better than the controls.

## Branch efficiency, calibration and clean audio

All variants preserve the same candidate universes: average expanded positions 3.15 and about 6,977 candidate leaves across seeds. Worst case remains 10,000. Those branch savings belong to the unchanged gate, not the graph or NCM. The same frozen score-override positions make ranking differences attributable to scoring within identical support, conditional on the conventional-reference selection policy.

Every protected variant retains 100% clean digit and sequence accuracy because the fixed gate leaves the corrected CNN in control. This does not prove that every raw refiner recognizes ideal-clean sounds. Full per-condition digit NLL, Brier score, ECE, harmful/helpful changes, latency, seed variation and policy hashes are in metrics.json. No probability-calibration guarantee is inferred from a low development ECE.

## Missing frequency and insufficient evidence

Raw model probabilities were tested on pure 697/770/852 Hz tones with no high-frequency component. Under the known uniform 0–9 DTMF alphabet, each identifies a row of three compatible digits, not a unique digit. No hidden digit label is created for this test. Raw maximum confidences, averaged over phases, positions and seeds:

| Variant | Unguarded mean maximum digit probability |
|---|---:|
| A: CNN + continuous reranker | 0.7970 |
| B: CNN + graph embedding | 0.7818 |
| C: CNN + NCM Transformer | 0.6133 |
| D: CNN + graph + NCM Transformer | 0.5993 |

The raw neural models do not reliably preserve the required ambiguity. Those results remain in uncertainty_checks_unguarded.json; they are not replaced by the guarded result.

An explicit, shared acoustic guard is therefore provided for clearly missing bands. It uses only measured spectra: retain frames with at least 80% of the event’s maximum frame energy, require a present-group normalized spectral peak above 0.01 and an opposite-group peak below 1e-5 of it. This avoids boundary/padding leakage masquerading as a second component. It assigns equal mass to row/column-compatible digits and a 1e-6 floor to every other digit, so all ten remain possible. Both fully absent groups retain the existing exactly uniform rule. For 941 Hz alone, the restricted alphabet has only digit 0; this alphabet-specific conclusion would change for a full telephone keypad.

The guard was designed after the synthetic missing-band diagnostic. It is a transparent DTMF rule, not learned NCM uncertainty and not an independently validated detector for noisy weak bands. Weak but recoverable evidence can require a different decision. All models share it; no benefit is attributed to a graph model or NCM. Guarded row tests give maximum probability 0.333331, and fully zero events remain exactly uniform. Runtime evaluation asserts that the guard fires on none of the main clean/corrupted test windows, so it changes no frozen candidate set or main accuracy result. Its cost is included in final timings.

## Inference and reproduction

```python
from ppf.graph_ncm import load_graph_tree
system = load_graph_tree("graph_results", seed=317, variant="D")
result = system.predict(events, top_k=5)  # four supplied 8 kHz tone windows [4,2400]
```

D is the default research path and contains the graph-conditioned Transformer. Choosing B or another variant is an explicit experimental control, not a replacement of the required NCM research component. Output includes four ten-way distributions, ranked complete hypotheses, expanded/accepted positions, missing-band guard flags, candidate count and retained search mass. example_prediction.json was produced through this audio-only API. Original UI/PPF interfaces are unchanged.

```bash
PYTHONPATH=. .venv/bin/python experiments/graph_experiment.py
PYTHONPATH=. .venv/bin/python experiments/graph_cnn_controls.py
.venv/bin/python experiments/graph_uncertainty.py
PYTHONPATH=. .venv/bin/python experiments/retime_graph.py
.venv/bin/python experiments/report_graph.py
.venv/bin/python -m pytest -q
```

32 tests pass: graph incidence and physical pair support, a single peak’s lack of pair specificity, finite zero-evidence embedding, target-free generated-level Transformer conditioning, hard/straight-through forward agreement, gradient flow, frozen codebooks, missing-band ambiguity and two-valid-band preservation, plus all earlier tests. Full timing includes corrected CNN, common reference verifier/gate, spectral measurements and absence guard, selected embedding/Transformer scoring and exact tree ranking. One CPU thread, three warmups; file loading/diagnostics excluded; per-run percentiles are averaged across seeds.

## Limits and decision

This is a fixed-message graph with a learned PCA vector embedding, not a trained GNN. Transformer C/D are new matched controls rather than the older APPT MLP. The graph changes both conditional embeddings and acoustic code targets; the matched-frequency and rewired controls are needed to interpret it. Three seeds, synthetic segmented audio, a conventional-reference selection policy, nonidentical total preprocessing costs, and repeated use of this development pool limit conclusions. Longer overlapping sounds, learned graph edges and animal sounds remain untested.

The experiment supports keeping the graph small and retaining the strong conventional reranker as the practical reference. It does not justify increasing graph layers or claiming that graph conditioning makes NCM superior. NCM remains available for research, with its negative and uncertainty results preserved.
