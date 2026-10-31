import hashlib,json
from pathlib import Path
import numpy as np
R=Path('graph_results');D=json.loads((R/'metrics.json').read_text());seeds=list(D['seeds']);rng=np.random.default_rng(1611)
primary={'A_cnn':'A: CNN + continuous reranker','B':'B: CNN + graph embedding','C_cnn':'C: CNN + NCM Transformer','D':'D: CNN + graph + NCM Transformer'}
extras={'A':'Continuous + matched frequency features','C':'NCM + matched frequency features','rewired':'Rewired graph + continuous reranker','D_no_prior':'Graph + NCM without prior-level conditioning'}
summary={};pairs={}
for scenario in ['clean','corrupted']:
    for mode in ['equal_updates','matched_flops']:
        key=f'{scenario}/{mode}';summary[key]={};outcomes={}
        for kind in [*primary,*extras]:
            rows=[D['seeds'][s]['variants'][kind][mode]['scenarios'][scenario] for s in seeds]
            result={k:{'mean':float(np.mean([v[k] for v in rows])),'seed_std':float(np.std([v[k] for v in rows],ddof=1))}
                    for k in ['digit_accuracy','sequence_top1','sequence_top5','digit_ece_10_bins','digit_nll','wrong_to_correct','correct_to_wrong','expanded_positions','candidate_leaves']}
            result['latency_ms']={p:float(np.mean([v['latency_ms'][p] for v in rows])) for p in ['p50','p95']}
            traces=[]
            for seed in seeds:
                path=R/f'seed_{seed}/{kind}_{mode}_{scenario}_trace.json'
                if not path.exists():path=R/f'seed_{seed}/{kind}_equal_updates_{scenario}_trace.json'
                traces.append(json.loads(path.read_text()))
            outcomes[kind]=np.array([[r['top1'] for r in t] for t in traces],dtype=float).mean(0)
            summary[key][kind]=result
        labels=[tuple(r['label']) for r in traces[0]];groups={v:[i for i,t in enumerate(labels) if t==v] for v in set(labels)}
        ids=list(groups.values());size=np.array([len(v) for v in ids]);draw=rng.integers(len(ids),size=(2000,len(ids)));den=size[draw].sum(1)
        pairs[key]={}
        for a,b in [('D','B'),('D','C_cnn'),('D','C'),('B','A_cnn'),('B','A'),('B','rewired'),('D','D_no_prior'),('D','A_cnn')]:
            delta=outcomes[a]-outcomes[b];sums=np.array([delta[v].sum() for v in ids]);bootstrap=sums[draw].sum(1)/den
            pairs[key][f'{a}-{b}']={'mean_delta':float(delta.mean()),'cluster_bootstrap_95_ci':np.percentile(bootstrap,[2.5,97.5]).tolist()}
(R/'summary.json').write_text(json.dumps({'summary':summary,'paired':pairs},indent=2))
raw=json.loads((R/'uncertainty_checks_unguarded.json').read_text());guarded=json.loads((R/'uncertainty_checks.json').read_text())
lines=['# Graph-conditioned NCM possibility refinement — 2026-10-31','','## Outcome','',
'A minimal local acoustic graph and an actual graph-conditioned NCM Transformer are implemented, with NCM retained as the default research inference path. The combination does not outperform graph-only reranking, and the physical graph does not clearly beat its rewired learned control. This experiment does not establish a graph-specific or NCM-specific recognition benefit.', '',
'## Minimal graph and acoustic evidence','',
'Each graph describes one digit sound, not an entire four-digit sequence. It has 42 nodes: four local CNN-time nodes, 28 time/frequency evidence nodes (four windows × seven frequencies), and ten possible digit identities. There are no learned digit-sequence edges or co-occurrence assumptions.', '',
'Frequency nodes measure log spectral peak energy around the DTMF bands 697, 770, 852, 941, 1209, 1336 and 1477 Hz, using ±2.5% bands and normalized local power. The excluded 1633 Hz column belongs to symbols outside this experiment’s 0–9 alphabet. Time nodes hold frozen CNN acoustic features. Same-frequency edges join neighboring local windows; fixed adjacent-window weights encode continuity. Digit nodes connect only to the two frequencies in the standard DTMF definition.', '',
'Fixed message aggregation takes the minimum of the two incident log-frequency supports, so one strong frequency alone does not provide strong pair support. A training-only normalized PCA projects local CNN features, measured frequencies and candidate-support messages into four 32-dimensional vectors. This is a small graph-derived vector embedding with learned unsupervised projection, not a trainable GNN. A large GNN was deliberately not introduced before this minimal relationship test.', '',
'| Digit | Frequency pair, Hz |','|---|---|']
from ppf.data import DTMF_FREQUENCIES
for digit in range(10):lines.append(f'| {digit} | {DTMF_FREQUENCIES[str(digit)][0]}, {DTMF_FREQUENCIES[str(digit)][1]} |')
lines+=['','All edges come from known acoustic definitions. Correct sample labels, corruption flags and sample IDs are absent from graph construction and prediction. Class identity nodes are the fixed candidate vocabulary, not the correct answers.', '',
'## Placement of NCM and controls','',
'The default D path is waveform → frozen corrected CNN → local acoustic graph → graph-conditioned vectors → multiscale residual VQ targets → conditional Transformer prediction → cumulative acoustic decoder → ten digit scores → the unchanged adaptive possibility tree. The quantizer represents graph-conditioned acoustic vectors, not whole-sequence digit labels.', '',
'Codebooks train without digit labels using EMA residual quantization at 1→2→4 positions and random nested-prefix update cutoffs. The acoustic feature projection and codebooks are frozen for conditional training. A one-block Transformer decoder (width 32, four heads, feed-forward width 128) predicts each level from acoustic memory and generated earlier embeddings. All positions within a level are produced in parallel. Training uses hard-forward straight-through embeddings, prefix digit CE and code CE. No true codes enter target-free prediction.', '',
'This replaces the prior APPT MLP level predictor only in a new matched Transformer experiment; all previous MLP checkpoints remain intact. C and D have identical 34,570-parameter Transformer/ranker architectures, differing in input/target representation. A/B rerankers have 8,906 trainable parameters. The graph aggregation has no supervised message weights; fitting a trainable GNN is outside this first experiment.', '',
'Primary variants: A is the original CNN-only continuous representation; B adds the physical graph; C uses the CNN-only acoustic representation with the Transformer; D combines the graph representation and Transformer. Additional graph-free frequency controls give the continuous reranker and Transformer the same seven explicit spectral measurements, so graph gains cannot automatically be credited to additional input features. A degree-preserving rewiring control permutes high-frequency assignments across digit nodes. A D no-prior control keeps the graph tokenizer/Transformer architecture but removes earlier-level conditioning.', '',
'## Frozen protocol and budget','',
'The CNN weights, relative log-Mel floor 0.001, 1,200/184/270 train/validation/test splits, gate weights, route/acceptance masks and per-position candidate lists are unchanged. The 270 test recordings contain 266 unique combinations. Identical branch-file hashes are checked for every variant within each seed. The preserved conventional reference gate determines both route and score-override positions; model confidences cannot change candidate support. Clean cases are protected by that policy.', '',
'Three seeds: 317, 419, 523. All ranking heads share their initialization within a seed. We report fixed 16-epoch equal-update results and approximately matched supported-operation supervised-training FLOPs. The latter gives each model about 91.45 billion profiled FLOPs; simple heads receive more updates on the same fixed examples. Temperatures use only the unchanged gate’s validation fitting half. Test scores select no model or checkpoint.', '',
'| Model family | Trainable parameters | Equal-update epochs | Matched-FLOP epochs touched |','|---|---:|---:|---:|']
for kind,label in [('A_cnn','Continuous A/B and rewired controls'),('C_cnn','CNN-only NCM Transformer'),('D','Graph-conditioned NCM Transformer'),('D_no_prior','Graph Transformer, no prior')]:
    row=D['seeds'][seeds[0]]['variants'][kind]['matched_flops'];lines.append(f'| {label} | {row["parameters"]:,} | 16 | {row["epochs_touched"]} |')
lines+=['','Matching covers profiler-supported supervised training operations, not exact total runtime. PCA fitting and ten-epoch label-free EMA tokenizer fitting are separate preprocessing stages and are not included in this matched budget. Profiling omits some FFT/distance/indexing/softmax costs. The longer simple-model exposure and different supervised objectives are why equal-update results are also retained. These controls are stronger than equal epochs alone but are not perfectly compute/parameter matched end-to-end pipelines.', '',
'## Are the relationships useful before fitting a graph model?','',
'| Simple physical pair-score check | Physical graph digit accuracy | Rewired assignments |','|---|---:|---:|']
for name,v in D['physics'].items():lines.append(f'| {name} | {100*v["physical_pair_score_digit_accuracy"]:.2f}% | {100*v["rewired_pair_score_digit_accuracy"]:.2f}% |')
lines+=['','This validates acoustic measurement and the fixed DTMF mapping. It is a physics/template sanity check, not evidence that a learned graph embedding or NCM is superior. The learned rewired-graph control below is the relevant structure comparison after training.', '',
'## Corrupted test: matched profiled-training-FLOP results','',
'| Variant | Digit accuracy | Sequence top-1 | Sequence top-5 | Digit ECE | CPU p50 / p95 ms |','|---|---:|---:|---:|---:|---:|']
for kind,label in primary.items():
    v=summary['corrupted/matched_flops'][kind]
    def pct(k):return f'{100*v[k]["mean"]:.2f} ± {100*v[k]["seed_std"]:.2f}%'
    lines.append(f'| {label} | {pct("digit_accuracy")} | {pct("sequence_top1")} | {pct("sequence_top5")} | {v["digit_ece_10_bins"]["mean"]:.4f} | {v["latency_ms"]["p50"]:.2f} / {v["latency_ms"]["p95"]:.2f} |')
lines+=['','Means ± sample standard deviation across seeds. These are controlled frozen-policy workloads, not independently tuned gates for each model.', '',
'| Additional control | Sequence top-1 | Sequence top-5 |','|---|---:|---:|']
for kind,label in extras.items():
    v=summary['corrupted/matched_flops'][kind];lines.append(f'| {label} | {100*v["sequence_top1"]["mean"]:.2f}% | {100*v["sequence_top5"]["mean"]:.2f}% |')
lines+=['','## Equal-update sensitivity','',
'| Variant | Sequence top-1 | Sequence top-5 |','|---|---:|---:|']
for kind,label in primary.items():
    v=summary['corrupted/equal_updates'][kind];lines.append(f'| {label} | {100*v["sequence_top1"]["mean"]:.2f}% | {100*v["sequence_top5"]["mean"]:.2f}% |')
lines+=['','## Paired statistical uncertainty','',
'2,000 percentile bootstrap resamples group all renderings of the same digit combination, average across the three fixed trained seeds, and preserve pairings across variants. Intervals condition on these models and the reused development test; they are not independent external confirmation.', '',
'| Top-1 comparison | Delta | Paired cluster-bootstrap 95% interval |','|---|---:|---:|']
for key,v in pairs['corrupted/matched_flops'].items():
    lo,hi=v['cluster_bootstrap_95_ci'];lines.append(f'| {key} | {100*v["mean_delta"]:.2f} points | [{100*lo:.2f}, {100*hi:.2f}] |')
lines+=['','The combined D model does not beat graph-only B. Correct graph B does not establish an advantage over the trained rewired graph. The prior-conditioning ablation does not support crediting NCM’s next-level dependency. Raw physical pair relationships are meaningful, but the extra embedding/refinement machinery is not shown to use them better than the controls.', '',
'## Branch efficiency, calibration and clean audio','',
'All variants preserve the same candidate universes: average expanded positions 3.15 and about 6,977 candidate leaves across seeds. Worst case remains 10,000. Those branch savings belong to the unchanged gate, not the graph or NCM. The same frozen score-override positions make ranking differences attributable to scoring within identical support, conditional on the conventional-reference selection policy.', '',
'Every protected variant retains 100% clean digit and sequence accuracy because the fixed gate leaves the corrected CNN in control. This does not prove that every raw refiner recognizes ideal-clean sounds. Full per-condition digit NLL, Brier score, ECE, harmful/helpful changes, latency, seed variation and policy hashes are in metrics.json. No probability-calibration guarantee is inferred from a low development ECE.', '',
'## Missing frequency and insufficient evidence','',
'Raw model probabilities were tested on pure 697/770/852 Hz tones with no high-frequency component. Under the known uniform 0–9 DTMF alphabet, each identifies a row of three compatible digits, not a unique digit. No hidden digit label is created for this test. Raw maximum confidences, averaged over phases, positions and seeds:', '',
'| Variant | Unguarded mean maximum digit probability |','|---|---:|']
for kind,label in primary.items():
    value=np.mean([raw[s][kind]['row_only_mean_max_probability'] for s in seeds]);lines.append(f'| {label} | {value:.4f} |')
lines+=['','The raw neural models do not reliably preserve the required ambiguity. Those results remain in uncertainty_checks_unguarded.json; they are not replaced by the guarded result.', '',
'An explicit, shared acoustic guard is therefore provided for clearly missing bands. It uses only measured spectra: retain frames with at least 80% of the event’s maximum frame energy, require a present-group normalized spectral peak above 0.01 and an opposite-group peak below 1e-5 of it. This avoids boundary/padding leakage masquerading as a second component. It assigns equal mass to row/column-compatible digits and a 1e-6 floor to every other digit, so all ten remain possible. Both fully absent groups retain the existing exactly uniform rule. For 941 Hz alone, the restricted alphabet has only digit 0; this alphabet-specific conclusion would change for a full telephone keypad.', '',
'The guard was designed after the synthetic missing-band diagnostic. It is a transparent DTMF rule, not learned NCM uncertainty and not an independently validated detector for noisy weak bands. Weak but recoverable evidence can require a different decision. All models share it; no benefit is attributed to a graph model or NCM. Guarded row tests give maximum probability 0.333331, and fully zero events remain exactly uniform. Runtime evaluation asserts that the guard fires on none of the main clean/corrupted test windows, so it changes no frozen candidate set or main accuracy result. Its cost is included in final timings.', '',
'## Inference and reproduction','',
'```python',
'from ppf.graph_ncm import load_graph_tree',
'system = load_graph_tree("graph_results", seed=317, variant="D")',
'result = system.predict(events, top_k=5)  # four supplied 8 kHz tone windows [4,2400]',
'```','',
'D is the default research path and contains the graph-conditioned Transformer. Choosing B or another variant is an explicit experimental control, not a replacement of the required NCM research component. Output includes four ten-way distributions, ranked complete hypotheses, expanded/accepted positions, missing-band guard flags, candidate count and retained search mass. example_prediction.json was produced through this audio-only API. Original UI/PPF interfaces are unchanged.', '',
'```bash',
'PYTHONPATH=. .venv/bin/python experiments/graph_experiment.py',
'PYTHONPATH=. .venv/bin/python experiments/graph_cnn_controls.py',
'.venv/bin/python experiments/graph_uncertainty.py',
'PYTHONPATH=. .venv/bin/python experiments/retime_graph.py',
'.venv/bin/python experiments/report_graph.py',
'.venv/bin/python -m pytest -q',
'```','',
'32 tests pass: graph incidence and physical pair support, a single peak’s lack of pair specificity, finite zero-evidence embedding, target-free generated-level Transformer conditioning, hard/straight-through forward agreement, gradient flow, frozen codebooks, missing-band ambiguity and two-valid-band preservation, plus all earlier tests. Full timing includes corrected CNN, common reference verifier/gate, spectral measurements and absence guard, selected embedding/Transformer scoring and exact tree ranking. One CPU thread, three warmups; file loading/diagnostics excluded; per-run percentiles are averaged across seeds.', '',
'## Limits and decision','',
'This is a fixed-message graph with a learned PCA vector embedding, not a trained GNN. Transformer C/D are new matched controls rather than the older APPT MLP. The graph changes both conditional embeddings and acoustic code targets; the matched-frequency and rewired controls are needed to interpret it. Three seeds, synthetic segmented audio, a conventional-reference selection policy, nonidentical total preprocessing costs, and repeated use of this development pool limit conclusions. Longer overlapping sounds, learned graph edges and animal sounds remain untested.', '',
'The experiment supports keeping the graph small and retaining the strong conventional reranker as the practical reference. It does not justify increasing graph layers or claiming that graph conditioning makes NCM superior. NCM remains available for research, with its negative and uncertainty results preserved.']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
# Persist current guard policy explicitly without pretending it trained the models.
config=json.loads(Path('experiments/graph_config.json').read_text())
config['missing_band_guard']={'scope':'shared analytic postprocessing; no main-test activation','strong_frame_energy_fraction':.8,'presence_peak':.01,'absence_ratio':1e-5,'other_digit_floor':1e-6}
Path('experiments/graph_config.json').write_text(json.dumps(config,indent=2)+'\n');(R/'config.json').write_text(json.dumps(config,indent=2))
paths=['ppf/graph_ncm.py','experiments/graph_experiment.py','experiments/graph_config.json','representation_results/cnn.pt','gate_results/metrics.json']
(R/'source_hashes.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},indent=2))
print('Saved graph_results/REPORT.md')
for k,label in primary.items():print(label,summary['corrupted/matched_flops'][k]['sequence_top1'],summary['corrupted/matched_flops'][k]['sequence_top5'])
print(pairs['corrupted/matched_flops'])
