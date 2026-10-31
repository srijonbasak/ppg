import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('gate_results');D=json.loads((ROOT/'metrics.json').read_text());seeds=list(D['seeds']);methods=list(D['seeds'][seeds[0]]['scenarios']['clean']);scenarios=['clean','corrupted','confident_error','two_erased']
C=D['config'];rng=np.random.default_rng(901);summary={};paired={}
for scenario in scenarios:
    traces={m:[json.loads((ROOT/f'seed_{s}/traces'/f'{scenario}_{m}.json').read_text()) for s in seeds] for m in methods}
    labels=[tuple(r['label']) for r in traces[methods[0]][0]]
    groups={label:[i for i,v in enumerate(labels) if v==label] for label in set(labels)}
    ids=list(groups.values());sizes=np.array([len(i) for i in ids]);draw=rng.integers(len(ids),size=(C['bootstrap_samples'],len(ids)))
    denominator=sizes[draw].sum(1)
    def cluster_ci(values):
        sums=np.array([values[i].sum() for i in ids]);boot=sums[draw].sum(1)/denominator
        return np.percentile(boot,[2.5,97.5]).tolist()
    summary[scenario]={}
    values={}
    for method in methods:
        rows=[D['seeds'][s]['scenarios'][scenario][method] for s in seeds]
        result={k:{'mean':float(np.mean([r[k] for r in rows])),'seed_std':float(np.std([r[k] for r in rows],ddof=1))}
                for k in ['digit_accuracy','sequence_top1','sequence_top5','digit_ece_10_bins','wrong_to_correct','correct_to_wrong','expanded_positions','candidate_leaves','refined_positions','cnn_error_recall']}
        result['latency_ms']={p:float(np.mean([r['latency_ms'][p] for r in rows])) for p in ['p50','p95']}
        if 'injected_recovery' in rows[0]:
            for k in ['injected_recovery','injected_reopening']:result[k]={'mean':float(np.mean([r[k] for r in rows])),'seed_std':float(np.std([r[k] for r in rows],ddof=1))}
        per={}
        for metric,tracekey in [('sequence_top1','top1'),('sequence_top5','top5')]:
            a=np.array([[row[tracekey] for row in trace] for trace in traces[method]],dtype=float).mean(0)
            result[metric]['cluster_bootstrap_95_ci']=cluster_ci(a);per[metric]=a
        a=np.array([[np.mean(row['digit_correct']) for row in trace] for trace in traces[method]]).mean(0)
        result['digit_accuracy']['cluster_bootstrap_95_ci']=cluster_ci(a);values[method]=per
        summary[scenario][method]=result
    paired[scenario]={}
    for a,b in [('appt_gated','local_gated'),('appt_gated','appt_old'),('local_gated','local_old')]:
        delta=values[a]['sequence_top1']-values[b]['sequence_top1']
        paired[scenario][f'{a}_minus_{b}']={'mean_delta':float(delta.mean()),'cluster_bootstrap_95_ci':cluster_ci(delta)}
(ROOT/'summary.json').write_text(json.dumps({'summary':summary,'paired':paired},indent=2))
A=json.loads((ROOT/'clean_audit.json').read_text())
lines=['# Clean-audio diagnosis and shared calibrated gate — 2026-10-31','','## Outcome','',
'The ideal-clean CNN failure is corrected by a validation-selected relative log-Mel floor, without changing CNN weights. A separately trained route-and-accept gate preserves all correct clean CNN decisions in all three seeds. It reduces harmful changes on corrupted audio but also rejects useful corrections. Conventional reranking remains stronger than APPT under the same gate, and recovery from artificial confident errors is unstable. No NCM-specific benefit is claimed.', '',
'## Clean-audio investigation','',
'Both generators produce correctly labeled 180 ms tones followed by 960 zero padding samples, giving the same 2,400-sample event shape. Clean tones omit the noise, jitter, twist and tanh soft clipping that are always/variably present in training synthesis. Four event windows and their label order were traced; the earlier independent Goertzel audit correctly recognized all 1,080 ideal-clean test digits. No corruption metadata or correct labels enter inference.', '',
'Validation-only interventions on the unchanged CNN:', '',
'| Intervention | Clean digit accuracy | Corrupted digit accuracy |','|---|---:|---:|']
lines.append(f'| Original input/frontend | {100*A["original_clean"]["digit_accuracy"]:.2f}% | {100*A["original_corrupted"]["digit_accuracy"]:.2f}% |')
for key,label in [('pcm16_roundtrip','PCM16 conversion'),('soft_clip_0.9','Training-style soft clip, drive 0.9'),('noise_ratio_0.03','Added noise, RMS ratio 0.03'),('mel_floor_0.001','Relative log-Mel floor 0.001')]:
    v=A['ablations'][key];corr=f'{100*v["corrupted"]["digit_accuracy"]:.2f}%' if 'corrupted' in v else 'not evaluated'
    lines.append(f'| {label} | {100*v["clean"]["digit_accuracy"]:.2f}% | {corr} |')
lines+=['','Raw clean log-Mel energies hit the old absolute floor in 54.18% of bins versus 35.48% for corrupted validation audio. Both adding a small noise floor and restoring training-style nonlinear harmonics rescue accuracy; PCM conversion does not. This establishes a spectral-distribution compatibility problem, while the exact learned cue responsible is not uniquely isolated.', '',
'The chosen frontend uses log(max(mel + 0.001*mean(mel), 1e-6)) before the existing per-event mean/std normalization. The relative floor is selected from validation interventions for at least 99.5% clean accuracy and no more than one percentage-point corrupted loss. It adds no waveform information, uses no corruption detector and is applied identically to clean/corrupted audio and every method. Original frontend/checkpoints remain available; RelativeFloorLogMel is a separate implementation. The projected acoustic tokenizer and local heads are retrained on the corrected features because the old PCA statistics are no longer compatible.', '',
'On the fixed test, corrected CNN clean digit/sequence accuracy is 100%/100%; corrupted digit accuracy is 80.83% and sequence accuracy 63.70%, versus the earlier 79.91%/59.26%. The frontend change is separate from any gate or hierarchy effect.', '',
'## Shared learned routing and acceptance','',
'Decision A: a cheap conventional local proposal and a 42-parameter three-class linear gate estimate whether additional analysis is useful. Gate features contain calibrated CNN/proposal confidence, entropy, margin, agreement, Jensen–Shannon divergence, waveform RMS and local latent energy. They do not contain digit labels, sample indices, clean/corruption flags or generator metadata.', '',
'Decision B: only routed positions run the selected refinement mechanism. The same gate evaluates the actual candidate distribution and accepts it only above a validation-selected safety threshold. Otherwise the CNN distribution is retained. Thus routing alone does not authorize an override. Every routed position still exposes all ten candidate digits. Exact-zero audio retains the common uniform no-evidence rule.', '',
'One shared gate is fit to pooled conventional/APPT proposals without a method-identity feature. Both arms use identical gate weights, route threshold, acceptance threshold and input-feature definitions. Actual candidate scores can differ, so their acceptance decisions can differ. The conventional model supplies the cheap route proposal in both arms; APPT cost is incurred only after routing.', '',
'The gate predicts harmful/neutral/helpful changes using validation labels only. Its score is P(helpful)-P(harmful), not a guaranteed calibrated probability of correctness. Validation temperatures calibrate digit scores. Empirical held-out safety-threshold selection provides risk control on the validation fold, not a formal guarantee on future audio.', '',
'Validation is split by recording into 92 fit and 92 safety-check recordings, with paired clean/corrupted views kept together. Temperatures and gate fitting use the fit half. Routing targets 90% CNN-error recall on the check half. A common acceptance threshold maximizes net corrections while limiting aggregate harmful changes to at most 0.5% of correct CNN decisions across both proposal types on that check half. The frontend intervention was previously selected using the full validation set, so the safety-check fold is not independent of frontend selection. All new-system test choices are fixed before evaluating these test outputs.', '',
'## Fixed multi-seed experiment','',
'Seeds 317, 419, 523. Same 1,200 training, 184 validation and 270 test recordings; the latter have 266 distinct combinations. The original CNN weights are frozen in every run. Each seed retrains the local acoustic tokenizer and the unchanged-size local/APPT heads for 16 epochs on the same corrupted training events. No ideal-clean labels are added to classifier training; paired clean validation is used only for calibration/gating. No larger NCM model, extra hierarchy levels or increased M is used.', '',
'Old policy controls use the corrected CNN/newly trained same-seed heads with unconditional override at routed positions, so old-versus-learned-policy comparisons isolate policy behavior rather than comparing different feature distributions. CNN gated search is a no-override control. Evaluation accepts only waveforms and optional tester-created probability faults. Labels are used outside prediction for metrics and validation fitting.', '',
'## Results across three seeds','',
'| Scenario / method | Digit accuracy | Sequence top-1 | Sequence top-5 | Digit ECE | CPU p50 / p95 ms |','|---|---:|---:|---:|---:|---:|']
for scenario in ['clean','corrupted']:
    for method in ['cnn','local_old','appt_old','local_gated','appt_gated']:
        v=summary[scenario][method]
        def metric(k):return f'{100*v[k]["mean"]:.2f} ± {100*v[k]["seed_std"]:.2f}%'
        lines.append(f'| {scenario}: {method} | {metric("digit_accuracy")} | {metric("sequence_top1")} | {metric("sequence_top5")} | {v["digit_ece_10_bins"]["mean"]:.4f} | {v["latency_ms"]["p50"]:.2f} / {v["latency_ms"]["p95"]:.2f} |')
lines+=['','Values are mean ± sample standard deviation across training seeds. The CNN is frozen and identical, so its seed standard deviation is zero. Old APPT changes 112 clean digits incorrectly in each of two seeds (112/1,080); the third seed makes none. Every erroneous clean change in those two APPT seeds affects true digit 3: counts by positions 1–4 are 28/28/31/25. Thus it is a systematic local class failure across all positions, not a single misaligned position. The learned gate prevents all those clean changes. Both local-policy variants preserve every clean CNN digit in this experiment.', '',
'## Corrections, regressions and computation on corrupted test','',
'| Method | Wrong → correct digits | Correct → wrong digits | Expanded positions | Candidate leaves | Refiner positions |','|---|---:|---:|---:|---:|---:|']
for method in ['local_old','appt_old','local_gated','appt_gated']:
    v=summary['corrupted'][method]
    lines.append(f'| {method} | {v["wrong_to_correct"]["mean"]:.2f} | {v["correct_to_wrong"]["mean"]:.2f} | {v["expanded_positions"]["mean"]:.2f} | {v["candidate_leaves"]["mean"]:.1f} | {v["refined_positions"]["mean"]:.2f} |')
lines+=['','Correction/regression counts are per 1,080 test digit positions, averaged over seeds. Smaller harm comes with fewer recovered errors; this is a safety/recall tradeoff, not an across-the-board accuracy improvement. Candidate leaves and routing are identical between local/APPT learned-gate arms within a seed. An actual proposal can be computed but rejected, so expanded/refined positions need not equal accepted changes.', '',
'## Confident errors and erased evidence','',
'One position in every corrupted test recording receives a deliberately wrong 99.5% CNN probability. Labels define this evaluation fault only; the inference system sees the fault probabilities and unchanged audio, not the correct answer.', '',
'| Method | Injected-error recovery mean ± seed SD | Seed values |','|---|---:|---|']
for method in ['local_old','appt_old','local_gated','appt_gated']:
    v=summary['confident_error'][method]['injected_recovery']
    vals=[D['seeds'][s]['scenarios']['confident_error'][method]['injected_recovery'] for s in seeds]
    lines.append(f'| {method} | {100*v["mean"]:.2f} ± {100*v["seed_std"]:.2f}% | '+', '.join(f'{100*a:.2f}%' for a in vals)+' |')
lines+=['','Learned gating supports confident-error reopening, but it is not robust to the artificial 99.5% fault distribution: one seed rejects nearly all such corrections. This failure is reported rather than fixed by tuning on test faults. The earlier disagreement rule recovers confident faults more reliably. Validation safety calibration does not imply robustness to adversarial confidence overrides.', '',
'Two completely erased windows still return exact uniform digit distributions. The gate preserves the correct known clean digits and cannot infer erased values. Observed sequence top-1 can exceed 1% in this small finite sample due to chance and deterministic tie-breaking; scores are identical across erased-digit candidates. Differences in top-K under those ties are not evidence of recovering missing information.', '',
'## Statistical uncertainty','',
'Paired 95% percentile bootstrap intervals use 2,000 resamples of distinct label-combination clusters, preserving all renderings and averaging each cluster’s recording outcomes across the three fixed model seeds. Seed variability is reported separately. These intervals condition on these three trained models and the reused development pool; they do not quantify all training-population uncertainty. When all clean samples are correct, the empirical bootstrap interval is degenerate and is not proof of perfect population accuracy.', '',
'| Corrupted comparison | Top-1 delta | Paired cluster-bootstrap 95% interval |','|---|---:|---:|']
for key,v in paired['corrupted'].items():
    lo,hi=v['cluster_bootstrap_95_ci'];lines.append(f'| {key} | {100*v["mean_delta"]:.2f} points | [{100*lo:.2f}, {100*hi:.2f}] |')
lines+=['','Per-method accuracy intervals and all seed-level calibration, top-5, routing, confident-error and latency metrics are in summary.json and metrics.json. The same gate with conventional reranking beats APPT in this experiment. No NCM-specific advantage is established.', '',
'## Verification, timing and limits','',
'24 tests pass. New tests cover unchanged behavior at zero relative floor, finite silent-input preprocessing, separation of routing from override acceptance, identical gate policy across proposal types, confident-choice reopening, and uniform erased probabilities. Prior APPT/PPF tests remain intact. The new prediction method accepts no labels or corruption metadata; the test fault hook accepts only a replacement probability matrix.', '',
'Latency includes the corrected CNN frontend, local projection/verifier, route gate, selected refiner, acceptance gate and candidate ranking. Pure CNN control skips the gate and verifier. Three warmup calls, one CPU thread; file I/O, test-fault construction and metric calculations are excluded. Reported percentiles are averaged across seeds rather than pooled. No GPU is available.', '',
'Important limits: synthetic in-distribution corrupted training, an ideal-clean distribution-shift view, supplied tone boundaries, three seeds, a repeatedly inspected test pool and a small validation safety fold. The frontend selection was exploratory on validation. The 0.5% validation harm cap is not guaranteed on test. Conventional and hierarchical heads have different parameter/FLOP counts. More stable confident-error recovery and independent evaluation remain unresolved.', '',
'## Reproduce','',
'```bash',
'PYTHONPATH=. .venv/bin/python experiments/clean_failure_audit.py',
'PYTHONPATH=. .venv/bin/python experiments/safe_gate_experiment.py',
'.venv/bin/python experiments/report_safe_gate.py',
'.venv/bin/python -m pytest -q',
'```','',
'Code: ppf/safe_gate.py. Configuration: experiments/safe_gate_config.json. Raw clean intervention/confusion matrices: gate_results/clean_audit.json. Gate thresholds and validation decisions: metrics.json. All seed checkpoints and per-recording correctness traces are preserved. Original CNN/APPT/PPF checkpoint files remain untouched.']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
paths=['ppf/safe_gate.py','experiments/safe_gate_experiment.py','experiments/safe_gate_config.json','representation_results/cnn.pt']
(ROOT/'source_hashes.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},indent=2))
print('Saved gate_results/REPORT.md')
for m in ['cnn','local_old','appt_old','local_gated','appt_gated']:
 v=summary['corrupted'][m];print(m,v['digit_accuracy'],v['sequence_top1'],v['sequence_top5'])
print('paired',paired['corrupted'])
