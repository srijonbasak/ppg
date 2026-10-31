import json
from pathlib import Path
r=Path('representation_results')
t=json.loads((r/'tokenizer_metrics.json').read_text())
d=json.loads((r/'comparison.json').read_text())
lines=['# Staged experiment results — 2026-10-31','','## Clean-label reconstruction gate','',
       f'Gate passed: {t["validation_gate_passed"]}. Selected epoch: {t["selected_epoch"]} by validation full-sequence accuracy (earliest tie). Validation and test each contain 1,000 digit combinations excluded from tokenizer training.', '',
       '| Code prefix | Test digit reconstruction | Test sequence reconstruction | Used codes / 64 |',
       '|---|---:|---:|---:|']
for i,n in enumerate([1,2,4]):
    m=t['test'];lines.append(f'| Through {n} positions | {100*m["digit_accuracy_by_level"][i]:.2f}% | {100*m["sequence_accuracy_by_level"][i]:.2f}% | {m["usage"][i]["unique_codes"]} |')
lines+=['','The initial fixed-rate run failed the final-checkpoint validation gate (98.8%). It is preserved. The revised validation-selection/cosine-decay run passed. Because the best checkpoint was epoch 1, the results do not establish that cosine decay caused the gain. EMA, normalization, initialization and dropout were not separately ablated. Later epochs remained recorded; full reconstruction alone was the selection criterion.', '',
'## Controlled recognition comparison','',
'One seed; 300 identical test recordings; 75 per condition. Four oracle-separated event windows per recording. PPF M=4, with no increase in sample count or Transformer depth.', '',
'| Model | Digit accuracy¹ | Sequence top-1 | Sequence top-5 | Sequence NLL | CPU p50 / p95 ms |',
'|---|---:|---:|---:|---:|---:|']
for name in ['cnn','ppf']:
    out=d['test'][name];m=out['overall'];lat=out['latency_ms']
    lines.append(f'| {name.upper()} | {m["digit_accuracy"]*100:.2f}% | {m["sequence_top1"]*100:.2f}% | {m["sequence_top5"]*100:.2f}% | {m["sequence_nll"]:.3f} | {lat["p50"]:.2f} / {lat["p95"]:.2f} |')
lines+=['','¹ Digit accuracy is evaluated from the highest-ranked complete sequence, rather than independently marginalized digit predictions. For the CNN factorized baseline these coincide.','','| Condition | CNN sequence top-1 / top-5 | PPF sequence top-1 / top-5 |','|---|---:|---:|']
for c in d['config']['conditions']:
    a=d['test']['cnn']['by_condition'][c];b=d['test']['ppf']['by_condition'][c]
    lines.append(f'| {c} | {100*a["sequence_top1"]:.2f}% / {100*a["sequence_top5"]:.2f}% | {100*b["sequence_top1"]:.2f}% / {100*b["sequence_top5"]:.2f}% |')
p=d['test']['ppf']
lines+=['','## Generator diagnosis','','| Level positions | Teacher-forced code accuracy | Generated-earlier-level code accuracy² |','|---|---:|---:|---:|']
for i,n in enumerate([1,2,4]):
    lines.append(f'| {n} | {100*p["teacher_forced_code_accuracy_by_level"][i]:.2f}% | {100*p["generated_code_accuracy_by_level"][i]:.2f}% |')
lines+=['','² Greedy rollout, separate from stochastic mixture recognition. Each level is compared with canonical target-tokenizer codes; another code hierarchy can sometimes decode the same digits, so code accuracy is diagnostic rather than equivalent to recognition accuracy.', '',
'The tokenizer and decoder state remained exactly unchanged throughout conditional training. Code prediction loss decreased from 4.099 to 2.720, so the earlier exploding-loss behavior was not reproduced. Recognition remains poor: reliable clean-label reconstruction has not translated into reliable audio-to-code prediction. The large teacher-versus-generated gap at finer levels supports investigating conditional prediction and sensitivity to earlier mistakes; it does not isolate a cause.', '',
'## Verification and limitations','',
'All 9 tests pass, covering the original pipeline, evaluator generation path, 1/2/4 shapes, encoder gradients, EMA buffer behavior, frozen target state, target-free generation, generated-prior propagation, and exact joint-mixture mass aggregation. The reconstruction gate was respected: generator training was deferred after the first failed tokenizer run.', '',
'No recognition advantage is demonstrated. This is a single-seed development experiment. PPF receives extra training and starts from the trained CNN encoder; compute is not matched. No automatic hyperparameter selection used the audio test set. Both splits use the same corruption families; OOD and real-recording evaluation remain untested. Partial erasure leaves evidence and is not a complete-erasure control. GPU memory is not applicable on this CPU host. Latencies exclude file loading and oracle segmentation.', '',
'Next diagnostic work should focus on conditional-code learning and generated-prefix sensitivity, not larger M or more Transformer layers. Detailed protocols and implemented mechanisms: README.md. Full metrics: tokenizer_metrics.json and comparison.json. Historical failed run: tokenizer_initial_failed.json.']
(r/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
