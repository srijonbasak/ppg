import hashlib,json
from pathlib import Path
import numpy as np
R=Path('hierarchy_results');D=json.loads((R/'metrics.json').read_text());seeds=list(D['seeds']);rng=np.random.default_rng(1201)
summary={};pairs={};coverage={}
for scenario in ['clean','corrupted']:
    coverage[scenario]={}
    for seed in seeds:
        choices=json.loads((R/f'seed_{seed}/{scenario}_branches.json').read_text())
        trace=json.loads((R/f'seed_{seed}/A_equal_updates_{scenario}_trace.json').read_text())
        included=[all(int(row['label'][j]) in c['choices'][j] for j in range(4)) for row,c in zip(trace,choices)]
        coverage[scenario][seed]={'oracle_candidate_coverage':float(np.mean(included)),
            'branches_hash':D['seeds'][seed]['policy_hashes'][scenario]}
    for mode in ['equal_updates','matched_flops']:
        key=f'{scenario}/{mode}';summary[key]={};values={}
        for kind in ['A','B','C','D']:
            rows=[D['seeds'][s]['variants'][kind][mode]['scenarios'][scenario] for s in seeds]
            out={k:{'mean':float(np.mean([v[k] for v in rows])),'seed_std':float(np.std([v[k] for v in rows],ddof=1))}
                 for k in ['digit_accuracy','sequence_top1','sequence_top5','digit_ece_10_bins','digit_nll','wrong_to_correct','correct_to_wrong','expanded_positions','accepted_positions','candidate_leaves','feature_mse','all_event_candidate_digit_accuracy']}
            out['latency_ms']={p:float(np.mean([v['latency_ms'][p] for v in rows])) for p in ['p50','p95']}
            out['linear_probe_digit_accuracy']=float(np.mean([D['seeds'][s]['variants'][kind][mode]['linear_probe_digit_accuracy'] for s in seeds]))
            traces=[]
            for s in seeds:
                path=R/f'seed_{s}/{kind}_{mode}_{scenario}_trace.json'
                if not path.exists():path=R/f'seed_{s}/{kind}_equal_updates_{scenario}_trace.json'
                traces.append(json.loads(path.read_text()))
            labels=[tuple(r['label']) for r in traces[0]];groups={v:[i for i,t in enumerate(labels) if t==v] for v in set(labels)}
            values[kind]=np.array([[float(r['top1']) for r in t] for t in traces]).mean(0)
            summary[key][kind]=out
        ids=list(groups.values());sizes=np.array([len(v) for v in ids]);draw=rng.integers(len(ids),size=(2000,len(ids)));den=sizes[draw].sum(1)
        pairs[key]={}
        for a,b in [('D','A'),('D','B'),('D','C'),('C','B')]:
            delta=values[a]-values[b];sums=np.array([delta[i].sum() for i in ids]);boot=sums[draw].sum(1)/den
            pairs[key][f'{a}-{b}']={'mean_delta':float(delta.mean()),'cluster_bootstrap_95_ci':np.percentile(boot,[2.5,97.5]).tolist()}
(R/'summary.json').write_text(json.dumps({'summary':summary,'paired':pairs,'candidate_coverage':coverage},indent=2))
names={'A':'Continuous local reranker','B':'Flat fine-only acoustic VQ','C':'Observed multiscale acoustic VQ','D':'Learned NCM-inspired refinement'}
lines=['# Hierarchy-contribution experiment — 2026-10-31','','## Outcome','',
'With exactly frozen candidate support and override decisions, continuous-feature reranking remains strongest. Multiscale observed quantization improves acoustic reconstruction and some ranking metrics over flat fine-only quantization, but full learned next-level refinement does not establish an advantage over the simpler controlled alternatives. No NCM-specific benefit is claimed.', '',
'## Frozen controls','',
'CNN weights and the corrected relative log-Mel floor 0.001 are unchanged. Same 1,200 training, 184 validation and 270 test recordings; test has 266 distinct digit combinations. Three seeds: 317, 419, 523. Clean and corrupted recordings are identical to the preceding shared-gate experiment.', '',
'For each seed, the preserved conventional local_gated system supplies both route and acceptance outputs. Both are frozen before new model training. Factored branch lists explicitly record all ten digits at each routed position and the fixed CNN choice elsewhere. All variants use these same lists and score-replacement positions; neither their confidences nor their predictions may change the gate. This is stricter than merely sharing gate parameters. Each list has a saved SHA-256 hash.', '',
'The fixed policy is based on the conventional reference model, so this experiment is conditional on that selection policy; it is not an independently optimized gate for every representation. The purpose is component isolation. During full-pipeline timing the original reference policy is recomputed and checked for exact agreement with the saved route/accept masks, then the frozen masks are used for all variants.', '',
'## Four variants','',
'| Variant | Representation scored by the same 128→64→10 ranking head |','|---|---|',
'| A | Continuous projected local CNN features, four vectors of dimension 32. |',
'| B | A separately fit, one-level 64-entry quantizer at four temporal positions. |',
'| C | Observed 1→2→4 multiscale residual codes, summed into a four-position reconstruction. |',
'| D | The existing-size learned 1→2→4 next-level predictor, followed by the same ranking head. |', '',
'B is a standalone fine-resolution quantizer of full acoustic features. It does not misuse residual fine codes that depend on omitted coarse context. C includes the intermediate scale as part of its coarse-to-fine information; it uses exactly the same acoustic tokenizer as D. Its codes are computed from observed input features and are not oracle clean codes or labels. D instead learns to predict that hierarchy from the same input, using generated earlier embeddings and parallel within-level prediction.', '',
'A/B/C heads have 8,906 trainable parameters; D remains 25,834, unchanged from APPT. Ranking heads start from identical weights within each seed. All see the same four-position 32-dimensional acoustic evidence, labels and shuffled training data. Quantizers are trained without labels and frozen. D retains its existing mean three-prefix digit loss plus 0.1 mean code CE; A/B/C train their final local digit CE. D-versus-C therefore tests the learned prediction/training mechanism jointly, not only one abstract factorization.', '',
'## Training budget controls','',
'Two predetermined comparisons are reported: equal 16-epoch optimizer-update budgets, and approximately matched profiled training FLOPs. The latter gives the cheaper models more passes over the same fixed training data. No test score selects epochs, models or temperatures. Per-variant temperatures use only the unchanged gate’s validation fitting half.', '',
'| Variant | Profiled training FLOPs per batch 128 | Equal-update epochs | FLOP-matched final epoch touched |','|---|---:|---:|---:|']
for k in ['A','B','C','D']:
    seed=D['seeds'][seeds[0]];epochs=[D['seeds'][s]['variants'][k]['matched_flops']['epochs_touched'] for s in seeds]
    lines.append(f'| {k} | {seed["profiled_training_flops_per_128"][k]:,.0f} | 16 | {epochs[0]} |')
lines+=['','The common target is approximately 41.95 billion profiled training FLOPs per model/seed, with stopping within one minibatch. The profiler captures supported PyTorch operation FLOPs, not all distance, indexing, softmax, memory or CPU costs; this is approximate FLOP matching, not exact wall-clock/FLOP equality. Longer A/B/C exposure can affect optimization, so equal-update results are retained. Recorded training_seconds_so_far includes intervening checkpoint evaluation/probes and must not be interpreted as pure training time. Gate/frontend inference costs are common and reported separately through full-pipeline latency.', '',
'## Corrupted test: matched profiled-FLOP comparison','',
'| Variant | Digit accuracy | Sequence top-1 | Sequence top-5 | Digit ECE | Wrong→correct / correct→wrong | CPU p50 / p95 ms |','|---|---:|---:|---:|---:|---:|---:|']
for k,v in summary['corrupted/matched_flops'].items():
    def val(metric):return f'{100*v[metric]["mean"]:.2f} ± {100*v[metric]["seed_std"]:.2f}%'
    lines.append(f'| {k}: {names[k]} | {val("digit_accuracy")} | {val("sequence_top1")} | {val("sequence_top5")} | {v["digit_ece_10_bins"]["mean"]:.4f} | {v["wrong_to_correct"]["mean"]:.2f} / {v["correct_to_wrong"]["mean"]:.2f} | {v["latency_ms"]["p50"]:.2f} / {v["latency_ms"]["p95"]:.2f} |')
lines+=['','Values are means ± sample standard deviations across the three trained seeds. Correction counts are per 1,080 test digit positions. Latency includes the common CNN/frontend, local projection, verifier, both frozen-policy gate decisions, selected variant scoring and exact candidate ranking. Three warmup calls, one CPU thread; loading, labels and offline diagnostic probes are excluded. Percentiles are averaged across seeds, not pooled. These timings are for the common frozen conventional-reference policy, which decides acceptance before variant scoring; they are not directly comparable with earlier live APPT gates that needed an APPT proposal before accepting it.', '',
'## Equal-update sensitivity','',
'| Variant | Digit accuracy | Sequence top-1 | Sequence top-5 |','|---|---:|---:|---:|']
for k,v in summary['corrupted/equal_updates'].items():
    lines.append(f'| {k} | {100*v["digit_accuracy"]["mean"]:.2f}% | {100*v["sequence_top1"]["mean"]:.2f}% | {100*v["sequence_top5"]["mean"]:.2f}% |')
lines+=['','The conclusion that A is the strongest practical ranking option is not dependent on giving it the larger matched-FLOP update count.', '',
'## Candidate coverage and clean controls','',
'| Seed | Expanded positions | Score-overridden positions | Candidate leaves | Oracle candidate-support coverage |','|---|---:|---:|---:|---:|']
for seed in seeds:
    v=D['seeds'][seed]['variants']['A']['equal_updates']['scenarios']['corrupted']
    lines.append(f'| {seed} | {v["expanded_positions"]:.2f} | {v["accepted_positions"]:.2f} | {v["candidate_leaves"]:.1f} | {100*coverage["corrupted"][seed]["oracle_candidate_coverage"]:.2f}% |')
lines+=['','Every number and every branch list in this table is identical across A–D within its seed. Oracle support coverage asks only whether the true full sequence is present anywhere in the fixed candidate universe. Top-5 coverage additionally measures ranking. None of A–D can improve their available candidate support; their differences are score-ordering differences within identical sets.', '',
'All four protected systems retain 100% clean digit and sequence accuracy. This does not demonstrate that every raw refiner recognizes clean audio: the frozen gate rejects all score overrides on clean samples, leaving the corrected CNN in control. All-event ungated candidate accuracy is saved separately in metrics.json. Clean protection is a property of the common frozen policy here, not a hierarchy achievement.', '',
'## Statistical uncertainty','',
'Paired percentile 95% intervals use 2,000 bootstrap resamples of distinct digit-combination clusters, retaining all acoustic renderings and averaging outcomes over the same three trained seeds. Seed standard deviations are separate. Intervals condition on these models and the reused development pool; they are not independent confirmation on new audio.', '',
'| Matched-FLOP top-1 comparison | Delta | Paired cluster-bootstrap 95% interval |','|---|---:|---:|']
for key,v in pairs['corrupted/matched_flops'].items():
    lo,hi=v['cluster_bootstrap_95_ci'];lines.append(f'| {key} | {100*v["mean_delta"]:.2f} points | [{100*lo:.2f}, {100*hi:.2f}] |')
lines+=['','Intervals for equal-update and clean comparisons are also saved in summary.json. More test reuse, seeds or percentile precision cannot substitute for independent external validation.', '',
'## What information the hierarchy contributes','',
'No variant receives new test observations. Its code hierarchy is a deterministic function of the same fixed local acoustic features and learned parameters. It can reorganize/compress that evidence or make a decision function easier to learn, but cannot create independent acoustic information or recover an erased independent digit from no evidence.', '',
'To distinguish representation quality from ranking-head capacity, a diagnostic linear probe (128 inputs, ten outputs) is trained on each frozen representation using the same 600 updates and training labels. These probes are not deployed, do not change candidate/gate lists, and are evaluated on all corrupted events. Feature MSE measures distortion from the shared projected acoustic input.', '',
'| Variant | Feature reconstruction MSE | Linear-probe digit accuracy |','|---|---:|---:|']
for k,v in summary['corrupted/matched_flops'].items():lines.append(f'| {k} | {v["feature_mse"]["mean"]:.4f} | {100*v["linear_probe_digit_accuracy"]:.2f}% |')
lines+=['','C preserves more of the input than B and gives more accessible class information. However, C uses three codebooks and seven code indices per sound, while B uses one codebook and four indices: a C-versus-B gain alone cannot distinguish a multiscale benefit from additional quantization capacity/bitrate. D-versus-C holds the acoustic tokenizer fixed and tests learned next-level generation. Any change in linear decodability is a nonlinear remapping of the input plus learned training priors, not new observed evidence.', '',
'## Component conclusions','',
'- A: continuous local features and a simple reranker remain the strongest baseline in this controlled task.',
'- B: flat fine-only quantization loses information and does not improve the continuous-feature baseline.',
'- C: extra residual scales improve acoustic reconstruction relative to B, but do not establish an advantage over A; capacity/bitrate is a confound for attributing C-versus-B gains specifically to scale.',
'- D: it beats B by 1.98 top-1 points within this pool (paired interval 0.36–3.70), but does not clearly beat C (0.74 points, interval −0.62–2.09) and loses to A. Its top-5 coverage is also below C. This does not establish an NCM-specific advantage over the strong simpler alternatives.', '',
'This is evidence about this small local MLP NCM-inspired adaptation on segmented DTMF, not a claim about every NCM architecture or longer overlapping sounds. The result does not justify further model-size, sampling-count or gate tuning on this pool.', '',
'## Verification and reproduction','',
'27 tests pass. New checks verify that B is independent of coarse residual books, C equals observed cumulative reconstruction with frozen books, D uses the shared head initialization, and score reranking cannot alter frozen candidate support. Full-pipeline evaluation checks every reference route/accept mask against its saved version. Prior checkpoints are untouched.', '',
'```bash',
'PYTHONPATH=. .venv/bin/python experiments/hierarchy_contribution.py',
'# Optional saved-checkpoint full-pipeline re-evaluation, without retraining:',
'PYTHONPATH=. .venv/bin/python experiments/retime_hierarchy.py',
'.venv/bin/python experiments/report_hierarchy.py',
'.venv/bin/python -m pytest -q',
'```','',
'Configuration: experiments/hierarchy_config.json. Frozen per-position candidate choices and acceptance masks: hierarchy_results/seed_*/clean_branches.json and corrupted_branches.json. Full raw metrics, calibration, paired intervals, linear probes, checkpoints, traces and source hashes are preserved under hierarchy_results/. This remains a three-seed developmental synthetic experiment with supplied event boundaries, selection conditional on the conventional reference gate and a repeatedly inspected test pool.']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
paths=['ppf/hierarchy_study.py','experiments/hierarchy_contribution.py','experiments/hierarchy_config.json','representation_results/cnn.pt','gate_results/metrics.json']
(R/'source_hashes.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},indent=2))
print('Saved hierarchy_results/REPORT.md')
for k,v in summary['corrupted/matched_flops'].items():print(k,v['digit_accuracy'],v['sequence_top1'],v['sequence_top5'],v['feature_mse'],v['linear_probe_digit_accuracy'])
print(pairs['corrupted/matched_flops'])
