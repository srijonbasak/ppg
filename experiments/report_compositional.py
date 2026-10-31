import hashlib
import json
from pathlib import Path
import numpy as np
r=Path('compositional_results')
D=json.loads((r/'comparison.json').read_text());P={n:json.loads((r/f'{n}_probes.json').read_text()) for n in ['original','structured']}
S=json.loads((r/'split_indices.json').read_text());T=json.loads((r/'structured_tokenizer_metrics.json').read_text());H=json.loads((r/'hierarchy_use.json').read_text())
cnnprobe=json.loads((r/'cnn_digit_probe.json').read_text())
lines=['# Compositional-generalization investigation — 2026-10-31','','## Outcome','',
'The hypothesis is partly supported: canonical fine-code identities in the original residual hierarchy depend on other digits, and a local structured representation makes them easier to predict. However, the original full hierarchy already supports perfect linear digit recovery, so it is not merely a whole-sequence lookup table. Making fine codes local did not produce a single correct complete held-out sequence in this fixed generator experiment.', '',
'## Data and controls','',
f'Existing audio conditions and tensors were retained. Training: {S["counts"]["train"]} recordings ({S["distinct_label_sequences"]["train"]} distinct sequences); validation: {S["counts"]["val"]} recordings; test: {S["counts"]["test"]} recordings ({S["distinct_label_sequences"]["test"]} distinct sequences). Validation excludes training combinations; test excludes both training and validation combinations. The original raw test contains 300 recordings; 30 overlapping combinations were excluded identically for both variants. New subsets and exact source indices are saved in split_indices.json.', '',
'All ten digits at every position and all 100 digit pairs at each of the six position pairs appear in training. Thus the held-out combinations consist entirely of individually familiar digits and pairs. Some test combinations have multiple independent waveform renderings; recordings are not 270 independent unique label combinations.', '',
'The original frozen tokenizer is retained unchanged. It was previously trained on 8,000 clean label combinations; the new tokenizer trains only on the current 1,200 training labels. The original tokenizer therefore has a different training history and may have seen held-out label combinations during its earlier representation training. Neither generator receives held-out labels in its prediction path. This is a development comparison with a retained frozen control, not a fully matched tokenizer-training-history trial.', '',
'Both generator variants use the same existing frozen CNN, identical fresh predictor/event-position initialization, identical example order, 16 epochs, AdamW rate 0.0003, batch 32, teacher forcing, and fixed final-epoch evaluation. Tokenizers/decoders stay frozen. Both have 232,330 total parameters and 154,752 trainable parameters; no model growth, prefix-training intervention or sampling increase. Inference uses deterministic argmax, M=1, and exhaustive ranking of 10,000 decoded sequence products. CNN and tokenizer tensors were checked unchanged after training.', '',
'## Equivalent linear digit probes','',
'All digit probes have 256 inputs and 40 outputs (four ten-way classifiers), 10,280 parameters. Features use training-only mean/std normalization. They train for 600 Adam updates at 0.01; there is no test-based hyperparameter or checkpoint selection. CNN features are ordered per-event temporal means. Token features are flattened 4×64 reconstructed prefix latents. Label-code probes receive clean target representations, while the CNN receives noisy audio; absolute accuracy between those sources is not a matched-noise comparison.', '',
'| Source | Position 1 | Position 2 | Position 3 | Position 4 | All four correct |','|---|---:|---:|---:|---:|---:|']
def row(name,p):
    a=p['test_accuracy_by_position'];return f'| {name} | '+' | '.join(f'{100*v:.2f}%' for v in a)+f' | {100*p["test_all_positions_correct"]:.2f}% |'
lines.append(row('Frozen CNN',cnnprobe))
for name in ['original','structured']:
    for l,p in enumerate(P[name]['tokenizer_digit_probes']):lines.append(row(f'{name}, prefix {l+1}',p))
lines+=['','Both full representations generalize perfectly under linear digit decoding. The original coarse level is strongly asymmetric. Source inspection explains why: its pointwise label encoder followed by linear interpolation from four positions to one samples only the central two positions. Changing position 1 or 4 cannot alter this coarse target. This is a coarse-target construction limitation, not proof of sequence memorization.', '',
'## Predicting codes directly from frozen audio features','',
'Linear code probes use the same 256-dimensional ordered CNN features and 64 outputs per code slot. Parameter counts are equal between variants for a given level (16,448 / 32,896 / 65,792). These probes receive no earlier code levels. Accuracies below average the code slots at a level.', '',
'| Target representation | Level 1 | Level 2 | Level 3 |','|---|---:|---:|---:|']
for name in ['original','structured']:
    vals=[np.mean(v['test_accuracy_by_position']) for v in P[name]['audio_to_code_probes']]
    lines.append(f'| {name} | '+' | '.join(f'{100*v:.2f}%' for v in vals)+' |')
lines+=['','The new fine codes are easier to predict, but the new global codes are not. Code accuracy is representation-dependent: a 10-class local code target is simpler than a context-dependent 64-code target. These percentages are not a common recognition metric.', '',
'## Single-digit changes and code similarity','',
'5,000 random anchors per position; exactly one digit is replaced by a different digit. The unrelated control changes all four digits. Code identity is measured within each codebook; arbitrary integer-code numeric distances are not used. Full code-change maps and codebook embedding MSE are saved in *_probes.json.', '',
'| Representation | Mean fine-slot change, one digit edited | Fine-slot change, all four edited | Neighbor / unrelated embedding MSE ratio |','|---|---:|---:|---:|']
for name in ['original','structured']:
    neigh=P[name]['neighbors'];near=[neigh[f'change_position_{p+1}'][2] for p in range(4)];far=neigh['unrelated_all_four_different'][2]
    lines.append(f'| {name} | {100*np.mean([np.mean(v["changed_code_fraction_by_slot"]) for v in near]):.2f}% | {100*np.mean(far["changed_code_fraction_by_slot"]):.2f}% | {np.mean([v["mean_squared_embedding_distance"] for v in near])/far["mean_squared_embedding_distance"]:.3f} |')
lines+=['','For example, changing digit 2 changes the original fine code at untouched position 1 in 86.86% of cases, position 3 in 40.16%, and position 4 in 26.12%. In the structured tokenizer, only the edited fine slot changes (100%); unchanged fine slots change 0%. Pair-code changes remain within the edited pair. Its global code responds to changes at all four positions. The original outer-position edits change its coarse code 0% of the time.', '',
'## Structured tokenizer definition and validation','',
'ppf/compositional.py adds a controlled subspace residual-VQ variant. A shared digit encoder produces 64 channels. Global mean context occupies channels 0–15, pair mean context channels 16–31, and local digit features channels 32–63. Each level subtracts the earlier upsampled reconstruction, then its codebook quantizes the residual projected to its channel block. Learned codes remain at 1→2→4 positions, are summed coarse-to-fine, and use the same latent-only decoder and nested-prefix loss. The fine target is position-independent and unaffected by earlier global/pair codes.', '',
'This is a structured/product-subspace residual quantizer: residuals are projected to disjoint blocks rather than permitting every level to revise every channel. The global context is a bag-like digit summary, not an encoding of all sequence order. This is an explicit architectural modification, not a claim of exact dense NCM equivalence. Generator architecture and size are unchanged.', '',
f'Structured tokenizer checkpoint selected at epoch {T["selected_epoch"]} using validation full-sequence reconstruction, earliest tie, from a fixed 30-epoch cosine-decay run. Validation and test full reconstruction are both 100%, so it passes the predeclared 99% validation gate before conditional training. Test code usage is {[v["unique_codes"] for v in T["test"]["usage"]]} of 64 per level. Ten fine codes are enough because they represent reusable digit identities; using all 64 is not required.', '',
'Important control: the new decoder reconstructs 100% of test sequences from fine codes alone, and removing or swapping coarse/pair components changes none of its generated digit decisions. Each digit uses exactly one fine code. The original uses 5–10 fine codes per digit in this test and fine-only oracle reconstruction is 88.15%. Therefore any eventual benefit of the new representation could reflect a simpler local digit target, rather than a demonstrated benefit from hierarchical refinement. No hierarchy advantage is established here.', '',
'## Held-out generator comparison','',
'| Model | Digit accuracy | Sequence top-1 | Sequence top-5 | Sequence NLL | CPU p50 / p95 ms |','|---|---:|---:|---:|---:|---:|']
b=D['cnn_baseline']['test'];h=H['cnn_baseline']
lines.append(f'| CNN | {100*np.mean(b["digit_accuracy_by_position"]):.2f}% | {100*h["sequence_accuracy"]:.2f}% | {100*h["sequence_top5"]:.2f}% | — | {h["latency_ms"]["p50"]:.2f} / {h["latency_ms"]["p95"]:.2f} |')
for name,rowd in D['runs'].items():
    out=rowd['test'];m=out['overall'];lat=out['latency_ms']
    lines.append(f'| {name} PPF | {100*np.mean(m["digit_accuracy_by_position"]):.2f}% | {100*m["sequence_accuracy"]:.2f}% | {100*m["sequence_top5"]:.2f}% | {m["sequence_nll"]:.3f} | {lat["p50"]:.2f} / {lat["p95"]:.2f} |')
lines+=['','| Generator | Oracle-prefix codes L1 / L2 / L3 | Generated-prefix codes L1 / L2 / L3 |','|---|---|---|']
for name,rowd in D['runs'].items():
    out=rowd['test'];a=' / '.join(f'{100*v:.2f}%' for v in out['teacher_code_accuracy']);b=' / '.join(f'{100*v:.2f}%' for v in out['generated_code_accuracy'])
    lines.append(f'| {name} | {a} | {b} |')
lines+=['','| Condition | Recordings | Original sequence top-1 | Structured sequence top-1 |','|---|---:|---:|---:|']
for cond,m in D['runs']['original']['test']['by_condition'].items():
    other=D['runs']['structured']['test']['by_condition'][cond]
    lines.append(f'| {cond} | {m["samples"]} | {100*m["sequence_accuracy"]:.2f}% | {100*other["sequence_accuracy"]:.2f}% |')
lines+=['','Latency includes four waveform windows through the existing CNN, deterministic hierarchy generation, 10,000-sequence scoring and top-5 retrieval. File loading, oracle segmentation and code diagnostics are excluded; three warmup calls precede single-sequence CPU timings. One thread, GPU unavailable. CPU timing variation is not a demonstrated efficiency difference.', '',
'## Interpretation and limits','',
'1. The original full representation already contains linearly reusable digit information. The broad claim that it mainly memorizes complete sequences is not supported.',
'2. Canonical residual-code identities are context-coupled, and the original coarse target omits the outer digits. The structured intervention makes fine codes local and substantially more predictable.',
'3. The fixed generator still fails on complete unseen combinations, even with ten digit-like fine codes. Representation organization alone does not resolve the audio-to-position/code learning problem. The existing order-conditioning and generated-prefix weaknesses remain relevant; this experiment does not isolate a sole cause.',
'4. There is an observed increase in target-free digit accuracy and lower NLL, but no complete-sequence or top-5 gain. No successful sequence recognition or hierarchical advantage is claimed.', '',
'This is one seed, synthetic in-distribution corruption families, oracle event windows, and a previously inspected development test pool. The filtered test has 266 distinct combinations and 270 renderings; uncertainty is not estimated from independent replications. Frozen CNN features were previously trained on the same audio training samples. Tokenizer training histories differ as stated above. New independent tests and matched-history controls are needed before a causal or superiority claim.', '',
'## Reproduce and verify','',
'```bash',
'PYTHONPATH=. .venv/bin/python experiments/compositional_probes.py',
'PYTHONPATH=. .venv/bin/python experiments/compositional_comparison.py',
'PYTHONPATH=. .venv/bin/python experiments/compositional_hierarchy_use.py',
'.venv/bin/python experiments/report_compositional.py',
'.venv/bin/python -m pytest -q',
'```','',
'Configuration: experiments/compositional_config.json. Full metrics and slot-change matrices: original_probes.json, structured_probes.json, comparison.json, hierarchy_use.json. Structured checkpoint and both generator checkpoints are preserved alongside logs. Original artifacts remain unchanged. All 14 tests pass, including fine-code locality, codebook subspace constraints, frozen-state behavior and compatibility with the existing generator.']
(r/'REPORT.md').write_text('\n'.join(lines)+'\n')
paths=['ppf/sequence.py','ppf/compositional.py','experiments/compositional_config.json','experiments/compositional_probes.py','experiments/compositional_comparison.py','experiments/compositional_hierarchy_use.py','representation_results/tokenizer.pt','representation_results/cnn.pt']
(r/'source_hashes.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},indent=2))
print('Saved compositional_results/REPORT.md')
