# Conditional-generation diagnosis — 2026-10-31

## Main findings

The unchanged architecture can memorize 32 clean examples. Oracle decoding, target-code lookup and sample ordering agree exactly. On held-out data the first level is underlearned, the original model responds weakly to event order, and errors grow under generated prefixes. These are observed failure points; a single exclusive root cause has not been established.

No tokenizer redesign, model-size increase, extra Transformer layers or larger M was used. Original tokenizer/CNN/checkpoint files were preserved. All recognition results below use target-free deterministic argmax generation; teacher-forced metrics are separate diagnostics.

## Wiring and alignment checks

- All 1,700 saved training/validation/test tensors match their manifest sample index, waveform hash, label order and corruption condition. No duplicate waveform hashes were found.
- The 32 clean examples have 128/128 correct Goertzel digit detections, independently checking their audio/label alignment.
- Reshaping CNN features into sequence memory preserves event order exactly.
- Forcing oracle target codes through the actual inference loop gives exactly the same probabilities as direct target-code decoding, including three samples per example.
- Teacher logits equal explicit inference-level calls with the same oracle prefixes: maximum absolute difference 0 at every level.
- The first level has no prefix input. Later levels consume previous-level embeddings using matching codebook indices in both training and inference.
- Tests verify first-level categorical sampling against a fixed 25%/75% conditional distribution. No sampling defect was reproduced.

## Tiny clean-data memorization

32 fixed, distinct four-digit sequences; baseline-pretrained CNN encoder, fresh generator initialization, unchanged model dimensions/dropout, AdamW learning rate 0.001, equal mean cross-entropy weighting across the three levels. All trainable audio/generator parameters could update. Tokenizer and decoder stayed frozen. Maximum 2,000 updates; evaluation every 50; stopped at update 300 when every generated code and sequence was correct.

| Evaluation | Sequence accuracy | Digit accuracy | Generated code accuracy, levels 1 / 2 / 3 |
|---|---:|---:|---|
| 32 training examples | 100.00% | 100.00% | 100.00% / 100.00% / 100.00% |
| Same sequences, new clean phases | 100.00% | 100.00% | 100.00% / 100.00% / 100.00% |
| 32 unseen clean sequences | 0.00% | 19.53% | 28.12% / 7.81% / 8.59% |

This proves a mapping can be learned; it does not prove compositional generalization. The same-label phase control succeeds, while new digit combinations fail. New phases are the only changed acoustic factor in that control.

A linear diagnostic probe on ordered frozen original CNN event means memorizes all seven target codes for all 32 examples (100%). This is a training-set information check, not a held-out result. A separate held-out bridge takes the existing CNN’s **predicted** digits, tokenizes those predictions, and reaches code accuracies 70.33% / 70.50% / 67.75%. It bypasses the generator and does not count as a PPF recognition result. Its 58.67% sequence accuracy is the existing CNN result. It shows usable code-relevant information exists in the CNN pathway.

## Original held-out model: deterministic diagnosis

| Level | Oracle-prefix code accuracy | Generated-prefix code accuracy | Oracle / generated-prefix code CE |
|---|---:|---:|---:|
| 1 | 10.33% | 10.33% | 3.298 / 3.298 |
| 2 | 13.33% | 5.83% | 3.042 / 4.308 |
| 3 | 38.50% | 5.17% | 1.991 / 5.140 |

Level 1: accuracy 10.33%, target majority frequency 5.67%, top-5 code accuracy 40.33%, mean maximum probability 15.24%, entropy 3.276 nats. Failure therefore begins before prefix exposure. Sampling is not needed to reproduce the failure.

## Event-order control

Rotate each recording’s four event windows by one place, and recompute the correct code from the correspondingly rotated labels. This preserves the same audio content and changes which events occupy each position.

| Checkpoint | Coarse prediction unchanged | Correct coarse target unchanged | Mean absolute logit change |
|---|---:|---:|---:|
| original | 92.33% | 13.67% | 0.049 |
| teacher_continuation | 28.33% | 13.67% | 1.029 |
| mixed_prefix | 28.33% | 13.67% | 0.950 |

The original model largely ignores order relevant to its target codes. Learned event-offset RMS was 0.036 versus acoustic feature RMS 1.883 in the clean wiring check. After continuation it grows to approximately 0.100, and predictions become more order-sensitive. Magnitude and sensitivity are diagnostics, not proof that offset scaling alone causes the recognition failure. No positional-embedding redesign was applied.

## Gradient and optimization inspection

On the first tiny-set update, nonzero gradients reached event positions (norm 0.0201), acoustic projection weights (0.6102), level embeddings (0.0795), and previous-level conditioning projection weights (0.2813). The diagnostic trace records subsequent gradients, losses, and clipping norms. No disconnected conditioning gradient was reproduced.
Original audio training used 0.0003 for eight epochs (roughly 304 updates, only eight exposures per example). The tiny test uses 0.001 and 300 updates on the same 32 examples. These budgets differ in exposures and learning rate; success does not isolate which optimization setting matters. Each level’s code CE is averaged over its positions and the three level losses are equally weighted. EMA codebooks are frozen buffers, so codebook optimizer gradients are intentionally absent.

## Paired continuation experiment

Both arms start from the same original PPF checkpoint, use the same shuffled example order, AdamW rate 0.0003, batch 32, and eight additional epochs on the same 1,200 examples. One retains teacher forcing; the other replaces each earlier level with its detached argmax prediction with probability 0.5 per example/level. Canonical target codes remain the supervision. Neither arm changes the tokenizer or model size. This is a single-seed development comparison, not an isolated causal proof; mask sampling also changes the training random-number stream.

| Checkpoint | Test digit accuracy | Test sequence top-1 | Test sequence top-5 | Generated codes, levels 1 / 2 / 3 |
|---|---:|---:|---:|---|
| Original | 17.08% | 0.00% | 0.00% | 10.33% / 5.83% / 5.17% |
| teacher_continuation | 24.33% | 0.00% | 0.00% | 52.00% / 15.00% / 9.50% |
| mixed_prefix | 25.75% | 0.00% | 0.33% | 52.67% / 18.00% / 13.17% |

The matched continuation improves coarse-code learning, but neither arm recovers a full test sequence (0/300). Mixed prefixes improve generated fine-code accuracy from 9.50% to 13.17% relative to continuation, while oracle-prefix fine-code accuracy drops from 44.00% to 33.25%. One mixed-arm true sequence enters the top five (1/300); this is not evidence of a dependable ranking advantage. Validation top-1 is 1/200 in both arms. No checkpoint is selected using test results.

Canonical residual codes describe the correct earlier hierarchy. Training against those same fine codes after deliberately changing a coarse prefix need not teach compensation for that wrong prefix. This is a limitation of the tested intervention, not evidence that mixing guarantees correction.

## Code changes and reproducibility

No indexing, codebook lookup, decoding or sampling defect was demonstrated, so no such bug fix is claimed. The code change adds SequencePPF.training_logits with an optional generated-prefix probability. Existing teacher_logits delegates to this shared path with probability zero. Generation remains unchanged. Tests establish equivalence at probabilities zero and one, oracle inference alignment, and sampling behavior. All 12 tests pass.

Run from repository root after the saved staged experiment artifacts exist:

```bash
PYTHONPATH=. .venv/bin/python experiments/debug_conditional.py
PYTHONPATH=. .venv/bin/python experiments/debug_prefix_training.py
PYTHONPATH=. .venv/bin/python experiments/debug_alignment.py
PYTHONPATH=. .venv/bin/python experiments/debug_tiny_generalization.py
.venv/bin/python -m pytest -q
```

Configuration: conditional_debug/config.json. Sample hashes: tiny_manifest.json and alignment_controls.json. Full traces: tiny_metrics.json, original_deterministic.json, prefix_comparison.json. Original reconstruction and CNN files are preserved. All paths are relative to the repository root.

The next unresolved issue is generalization of the ordered audio-to-code mapping and robustness of later levels to incorrect prefixes. The evidence does not support increasing M, enlarging the model, or declaring improved complete-sequence recognition. Oracle event boundaries, synthetic data, a single seed and previously inspected test samples limit this development investigation. Independent data and multiple seeds are still required for a confirmatory claim.
