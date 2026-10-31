# Auxiliary digit supervision — 2026-10-31

## Outcome

Capacity-aware auxiliary supervision did not establish improved complete-sequence recognition. Test top-1 stayed 0/270 in both matched arms. Digit accuracy rose by 0.83 percentage points in this single development run; validation complete-sequence accuracy decreased. The frozen structured decoder continues to bypass coarse and intermediate components in its final digit decisions.

## Fixed conditions and comparison

Existing structured tokenizer, decoder and CNN checkpoints are unchanged and frozen. Same 1,200 training recordings, 184 validation recordings and 270 test recordings as the compositional experiment. Test combinations are absent from training and validation; all position-specific digits and pairs are familiar. Test contains 266 distinct combinations, with some independent acoustic renderings. Oracle tone boundaries and the same reference/noise/interference/partial-erasure conditions are retained.

Both arms use identical fresh generator/event-position initialization, seed 211, identical data order, 16 epochs, 608 optimizer steps, AdamW learning rate 0.0003, batch 32, and fixed final-epoch evaluation. Model dimensions, 1→2→4 code positions, parallel prediction within a level, parameter count and M=1 remain unchanged. Both arms have 232,330 total and 154,752 trainable parameters. No test-based tuning was performed.

Both arms run the same teacher-forced code path and target-free differentiable rollout each update, consuming the same sequence of dropout draws. The baseline backpropagates code CE only. The auxiliary arm adds digit losses. Executing the extra forward path changes dropout draw timing relative to the earlier 32.50% checkpoint, so the newly matched baseline (33.52%) is the valid treatment comparison; the old checkpoint is separately audited as a reference.

Budgets match examples, epochs and optimizer updates, not backward FLOPs. Training wall times were 41.90s baseline versus 53.12s auxiliary. There is no efficiency advantage claim.

## Ordered audio audit

For eight checked examples: event input [8,4,2400] -> CNN features [8,4,64,31] -> Transformer audio memory [8,64,124]. The four event streams remain in order, with learned event offsets; no global temporal pooling occurs before cross-attention. Reshaping back exactly recovers each event feature stream. The prior ordering-sensitivity failure was learned behavior, not accidental global pooling. Ordered storage alone does not force attention to use the offsets.

| Position | Existing CNN digit accuracy | Earlier structured fine-code accuracy | Matched baseline fine-code accuracy | Auxiliary fine-code accuracy |
|---|---:|---:|---:|---:|
| 1 | 82.96% | 33.70% | 34.07% | 36.67% |
| 2 | 79.26% | 32.22% | 32.96% | 32.22% |
| 3 | 80.74% | 35.19% | 35.93% | 33.70% |
| 4 | 76.67% | 28.89% | 31.11% | 34.81% |

All columns use identical test recordings. Fine slots correspond to individual positions in this structured tokenizer. CNN features contain much more usable digit evidence than either generator extracts.

## Differentiable strategy and capacity-aware targets

SequencePPF.differentiable_generation accepts audio only. At each level it predicts logits, chooses hard argmax codes, embeds those codes and carries the generated embeddings to the next level. It decodes the accumulated hierarchy after each prefix using the existing frozen decoder. No correct target codes enter this rollout.

For logits z, p=softmax(z), a=one_hot(argmax(z)), training uses a+(p-stop_gradient(p)). Forward values are exactly hard one-hot values; backward uses the softmax Jacobian. This is a biased straight-through gradient estimator, not the true derivative of argmax. Frozen decoder parameters permit gradients to generated latent inputs. In evaluation mode, audited hard codes and final probabilities exactly match deterministic production inference (maximum difference 0); regression tests also check every prefix.

Both arms retain the original mean teacher-forced code CE across three levels. The auxiliary objective adds (0.1*KL1 + 0.25*KL2 + CE3)/1.35. KL1 and KL2 match digit distributions produced by the frozen decoder from oracle target-code prefixes, averaged over four positions. CE3 uses the actual four digit labels. These early losses distill the limited-capacity oracle-prefix distributions; they do not require a single coarse code to identify all four digits. Labels and oracle codes define training losses only. Prediction, ranking and timing use generated codes exclusively.

Initial same-level auxiliary-loss gradients to code logits were 6.85e-05, 0.000292, 0.00634 for levels 1–3. All are finite and nonzero. Early oracle-prefix decoder distributions are weak, and their KL losses/gradients are small. This negative result applies to this frozen-decoder distillation recipe; it does not test all possible direct digit-supervision strategies.

## Recognition and code prediction

| Arm | Digit accuracy | Test sequence top-1 | Test top-5 | Test NLL | CPU p50 / p95 ms |
|---|---:|---:|---:|---:|---:|
| Matched baseline | 33.52% | 0.00% | 0.00% | 8.618 | 2.09 / 2.59 |
| Auxiliary | 34.35% | 0.00% | 0.37% | 8.447 | 2.08 / 2.23 |

Auxiliary top-5 contains the true sequence for 1/270 recordings, versus 0/270 in the baseline. Validation top-1 is 1/184 versus 2/184. Neither establishes a reliable recognition gain; validation generated fine-code accuracy is identical at 30.84%.

| Arm | Generated code accuracy L1 / L2 / L3 | Oracle-prefix code accuracy L1 / L2 / L3 |
|---|---|---|
| Matched baseline | 54.44% / 25.56% / 33.52% | 54.44% / 27.22% / 50.93% |
| Auxiliary | 55.93% / 26.11% / 34.35% | 55.93% / 27.41% / 49.44% |

## Generated prefix reconstruction

| Arm | Prefix | Position 1 | Position 2 | Position 3 | Position 4 | Full sequence |
|---|---|---:|---:|---:|---:|---:|
| Matched baseline | 1 | 15.19% | 15.56% | 15.19% | 16.30% | 0.00% |
| Matched baseline | 2 | 28.15% | 20.00% | 25.19% | 23.70% | 0.00% |
| Matched baseline | 3 | 34.07% | 32.96% | 35.93% | 31.11% | 0.00% |
| Auxiliary | 1 | 14.81% | 15.93% | 15.19% | 16.30% | 0.00% |
| Auxiliary | 2 | 25.93% | 20.74% | 24.44% | 23.33% | 0.00% |
| Auxiliary | 3 | 36.67% | 32.22% | 33.70% | 34.81% | 0.00% |

## Corrections and regressions

Counts are paired changes on the same 1,080 digit positions (270 recordings × four digits). They describe changing predictions after adding a level, not a guarantee that earlier errors are explicitly revised.

| Arm | Transition | Wrong → correct digits | Correct → wrong digits | Wrong → correct sequences | Correct → wrong sequences |
|---|---|---:|---:|---:|---:|
| Matched baseline | 1 → 2 | 178 | 84 | 0 | 0 |
| Matched baseline | 2 → 3 | 180 | 80 | 0 | 0 |
| Auxiliary | 1 → 2 | 172 | 85 | 0 | 0 |
| Auxiliary | 2 → 3 | 212 | 96 | 0 | 0 |

Per-position transition counts, per-condition measurements and oracle-prefix reconstruction diagnostics are recorded in comparison.json.

## Coarse/intermediate contribution

Removing or swapping generated coarse/intermediate components at the final decoder changed 0/1,080 digit decisions in either arm. Thus those decoder components still make no measurable contribution to final argmax decisions on this test. This does not assert that the logits are exactly independent of those channels.

| Arm | Normal digit accuracy | Zero earlier-level conditioning | Normal sequence accuracy | Zero-conditioning sequence accuracy |
|---|---:|---:|---:|---:|
| Matched baseline | 33.52% | 28.89% | 0.00% | 0.37% |
| Auxiliary | 34.35% | 32.04% | 0.00% | 0.00% |

Zeroing earlier embeddings only at the generator’s conditioning input lowers average digit accuracy by 4.63 points in the baseline and 2.31 points with auxiliary supervision. Earlier levels therefore influence useful digit predictions, but this inference perturbation is not a retrained no-hierarchy ablation and does not prove a benefit from residual VQ or coarse-to-fine generation. The baseline even obtains one complete sequence after zeroing conditioning while losing digit accuracy; no sequence-level hierarchy advantage is established.

## Timing, verification and limitations

Latency uses the ordinary hard production generation path (not the straight-through training path), waveform features, exhaustive 10,000-sequence scoring and top-5 retrieval. Three warmup calls; single CPU thread; file I/O, supplied segmentation and diagnostic metrics excluded. GPU unavailable. Small timing differences are not an efficiency finding.

All 16 tests pass. Added tests verify target-free hard/ST forward equivalence for every prefix, generated-code carry-forward, gradient flow to every level with a frozen decoder, and unchanged tokenizer state. Training asserts exact frozen tokenizer/decoder and CNN state equality. Original checkpoint files remain untouched.

This is one seed and the same previously inspected synthetic development pool. The controls do not resolve real-world generalization, learned segmentation, OOD corruption or statistical robustness. Small digit/top-5 changes require independent tests and repeated seeds before a positive claim. No tokenizer redesign, additional layers, larger model or increased M was used. The two questions remain separate: reliable sequence recognition is still missing, and a hierarchy-specific recognition benefit remains unestablished.

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python experiments/auxiliary_experiment.py
.venv/bin/python experiments/report_auxiliary.py
.venv/bin/python -m pytest -q
```

Configuration: experiments/auxiliary_config.json. Raw metrics, correction counts and audits: auxiliary_results/comparison.json. Both checkpoints, training log and source hashes are saved in auxiliary_results/.
