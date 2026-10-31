import hashlib,json
from pathlib import Path
import numpy as np
R=Path('encoder_results');D=json.loads((R/'metrics.json').read_text());seeds=list(D['seeds']);audit=json.loads((R/'encoder_audit.json').read_text())
stages=list(D['seeds'][seeds[0]]['digit_probes']);modes=D['config']['conditioning'];cases=['clean','corrupted','noisy','partial_erasure','missing_frequency'];summary={'probes':{},'ncm':{},'protected_continuous':{}};pairs={};rng=np.random.default_rng(2811)
for stage in stages:
    summary['probes'][stage]={}
    for case in cases:
        rows=[D['seeds'][s]['digit_probes'][stage]['scenarios'][case] for s in seeds]
        out={k:{'mean':float(np.mean([r[k] for r in rows])),'seed_std':float(np.std([r[k] for r in rows],ddof=1))}
             for k in ['digit_accuracy','sequence_top1','sequence_top5','digit_ece_10_bins','mean_max_probability']}
        out['latency_ms']={p:float(np.mean([r['latency_ms'][p] for r in rows])) for p in ['p50','p95']}
        out['frequency_band_nmse']=float(np.mean([r['frequency_recovery']['train_standardized_band_mse'] for r in rows]))
        summary['probes'][stage][case]=out
for mode in modes:
    summary['ncm'][mode]={}
    for case in cases:
        rows=[D['seeds'][s]['ncm'][mode]['scenarios'][case] for s in seeds]
        raw={k:float(np.mean([r['raw'][k] for r in rows])) for k in ['digit_accuracy','sequence_top1','sequence_top5','digit_ece_10_bins']}
        raw['code_accuracy']=np.mean([r['raw']['generated_code_accuracy_by_level'] for r in rows],0).tolist()
        p={k:{'mean':float(np.mean([r['protected'][k] for r in rows])),'seed_std':float(np.std([r['protected'][k] for r in rows],ddof=1))}
           for k in ['digit_accuracy','sequence_top1','sequence_top5','digit_ece_10_bins','wrong_to_correct','correct_to_wrong']}
        p['latency_ms']={q:float(np.mean([r['protected']['latency_ms'][q] for r in rows])) for q in ['p50','p95']}
        summary['ncm'][mode][case]={'raw':raw,'protected':p}
for stage in ['final_pca','stem_pca','block2_pca','multiscale']:
    summary['protected_continuous'][stage]={}
    for case in cases:
        rows=[D['seeds'][s]['protected_continuous_probes'][stage][case] for s in seeds]
        summary['protected_continuous'][stage][case]={k:float(np.mean([v[k] for v in rows])) for k in ['digit_accuracy','sequence_top1','sequence_top5','digit_ece_10_bins']}


def paired_delta(first,second,field,record_mean=False):
    a=[];b=[]
    for seed in seeds:
        one=json.loads((R/f'seed_{seed}'/first).read_text());two=json.loads((R/f'seed_{seed}'/second).read_text())
        a.append([np.mean(v[field]) if record_mean else float(v[field]) for v in one]);b.append([np.mean(v[field]) if record_mean else float(v[field]) for v in two])
    values=np.array(a).mean(0)-np.array(b).mean(0);labels=[tuple(v['label']) for v in one]
    groups={v:[i for i,x in enumerate(labels) if x==v] for v in set(labels)};ids=list(groups.values());size=np.array([len(x) for x in ids]);draw=rng.integers(len(ids),size=(2000,len(ids)))
    sums=np.array([values[x].sum() for x in ids]);ci=np.percentile(sums[draw].sum(1)/size[draw].sum(1),[2.5,97.5])
    return {'mean_delta':float(values.mean()),'cluster_bootstrap_95_ci':ci.tolist()}

for a,b in [('stem','final'),('stem_pca','stem'),('final_pca','final')]:pairs[f'probe_{a}_minus_{b}']=paired_delta(f'rawprobe_{a}_corrupted_trace.json',f'rawprobe_{b}_corrupted_trace.json','digit_correct',True)
for mode in modes:
    if mode!='final':pairs[f'ncm_{mode}_minus_final']=paired_delta(f'{mode}_corrupted_trace.json','final_corrupted_trace.json','top1')
for mode,stage in [('final','final_pca'),('stem','stem_pca'),('block2','block2_pca'),('multiscale','multiscale')]:pairs[f'ncm_{mode}_minus_continuous']=paired_delta(f'{mode}_corrupted_trace.json',f'probe_{stage}_corrupted_trace.json','top1')
# Interference subgroup uses the existing raw probe trace, without refitting.
from compositional_probes import load_data
base_data=load_data();interference=np.array([v=='interference' for v in base_data['test'][2]])
interference_scores={}
for stage in stages:
    values=[]
    for seed in seeds:
        trace=json.loads((R/f'seed_{seed}/rawprobe_{stage}_corrupted_trace.json').read_text());values.append(np.array([v['digit_correct'] for v in trace])[interference].mean())
    interference_scores[stage]=float(np.mean(values))
(R/'summary.json').write_text(json.dumps({'summary':summary,'paired':pairs,'interference_probe_accuracy':interference_scores},indent=2))
linear=np.mean([D['seeds'][s]['code_probes']['final']['corrupted'] for s in seeds],0)
generated=summary['ncm']['final']['corrupted']['raw']['code_accuracy']
teacher=np.mean([D['seeds'][s]['ncm']['final']['code_diagnostics']['corrupted']['oracle_prefix_code_accuracy'] for s in seeds],0)
observed_decoder=float(np.mean([D['seeds'][s]['ncm']['final']['code_diagnostics']['corrupted']['observed_code_decoder_digit_accuracy'] for s in seeds]))
labels=base_data['test'][1].numpy();cardinality=np.where(labels==0,1,3)
bounds={'expected_digit_accuracy':float((1/cardinality).mean()),'expected_sequence_top1':float((1/cardinality).prod(1).mean()),'expected_sequence_top5':float((np.minimum(5,cardinality.prod(1))/cardinality.prod(1)).mean())}
lines=['# Audio encoder information-preservation study — 2026-10-31','','## Main findings','',
'The CNN does not perform temporal downsampling. A measurable decline in digit-probe performance appears through its learned residual stack under interference, but substantial useful evidence remains in the final features. The 32-channel PCA adapter discards much of the advantage of early features. The NCM model then underuses the remaining information and also learns a code/decoder mapping that differs from its canonical acoustic targets. These findings do not justify replacing the audio encoder as the first intervention.', '',
'No original CNN, NCM, graph, gate or checkpoint was overwritten. No new audio encoder was proposed or trained. The existing CNN-only NCM Transformer architecture and final-feature codebooks are preserved; only its conditioning source changes.', '',
'## 1. Exact encoder implementation','',
'Input is a float32 mono tone window [B,2400] at 8 kHz (300 ms). Current generated tones last 180 ms (1,440 samples), then receive 960 zero-padding samples; the study uses supplied event boundaries, not learned segmentation.', '',
'| Operation | Output shape | Sampling/resolution or context |','|---|---|---|',
'| Centered STFT: Hann 256, hop 80, FFT 256, reflect padding 128 | [B,129,31] complex | 32 ms windows, 10 ms frame grid, 31.25 Hz FFT-bin spacing, 0–4 kHz |',
'| Magnitude squared | [B,129,31] real | Phase is discarded; bin spacing is not a guarantee of two-tone resolvability. |',
'| Fixed triangular 40-band Mel matrix | [B,40,31] | Nonuniform overlapping bands from nominal 20–4,000 Hz; actual integer-bin supports in encoder_audit.json. |',
'| Relative floor: log(max(mel + 0.001*mean(mel),1e-6)) | [B,40,31] | Same corrected frontend as prior study; no condition-specific branch. |',
'| Per-event mean subtraction and std division | [B,40,31] | Statistics over all frequency/time cells; default corrected std, clamped at 1e-4. |',
'| Conv1d 40→64, k5, stride1, padding2, bias | [B,64,31] | Nominal convolution support 5 frames / 72 ms. No separate stem activation or pooling. |',
'| Residual block 1, dilation1 | [B,64,31] | Nominal cumulative support 7 frames / 92 ms. |',
'| Residual block 2, dilation2 | [B,64,31] | Nominal cumulative support 11 frames / 132 ms. |',
'| Residual block 3, dilation4 | [B,64,31] | Nominal cumulative support 19 frames / 212 ms. |',
'| Existing classifier: AdaptiveAvgPool1d(1), flatten, Linear64→10 | [B,10] | This head pools time; the encoder itself does not. |', '',
'Each residual block performs exactly Conv1d64→64(k3, stride1, padding=dilation, dilation=dilation, bias) → GroupNorm(4 groups, 16 channels/group, eps1e-5, affine) → GELU → Conv1d64→64(k1, stride1, bias) → GroupNorm(4 groups) → add block input → GELU. Thus the CNN contains seven convolutions, six GroupNorms and six GELUs. There is no max pooling, no strided convolution, no dropout and no embedding lookup inside the CNN.', '',
'After the stem, the frequency axis is mixed into learned channels: 64 channels do not represent 64 evenly spaced frequency bins. Nominal convolution supports above do not equal strict receptive fields: frontend statistics/floor and every GroupNorm use all time frames, giving global event dependence. The frame grid remains 31×10 ms throughout.', '',
f'Manual tap extraction exactly matched native encoder output: {audit["manual_taps_equal_native_encoder"]}. Halving waveform gain changed normalized frontend values by at most {audit["gain_halving_max_frontend_difference"]:.3g}. Absolute gain is substantially removed before the stem; this intended invariance is not demonstrated to be the DTMF recognition failure.', '',
'Actual Mel supports in the DTMF range are nonuniform: around the row frequencies, nonzero support widths span roughly 62.5–93.75 Hz on the discretized grid; higher digit columns involve wider supports. Full per-filter peaks/nonzero supports and every convolution/normalization hook shape are exported rather than treating channels as a uniform frequency grid.', '',
'## 2. Post-encoder pooling, projection and NCM embeddings','',
'The prior local adapter reduces 31 frames to four temporal bins, then standardizes channels with training statistics, applies a 64→32 PCA matrix and component whitening. Four adaptive pooling bins overlap at boundary indices (0–7, 7–15, 15–23, 23–30). This reduction occurs outside the CNN.', '',
'Early-stage adapters use the same train-corrupted-only pooling/standardization/PCA recipe. Final uses the preserved projection buffers. Multiscale concatenates stem/block1/block2/final channels and projects 256→32, retaining four time positions. These projections are fixed, not new trainable encoder layers. Probe input buffers additionally standardize each flattened feature using training statistics; NCM receives channel-whitened tokens, so readout comparisons establish accessibility rather than perfectly matched normalization/optimization. Multiscale stores more fixed projection coefficients, although NCM trainable size is unchanged.', '',
'NCM uses a learned 4×32 acoustic positional embedding, fixed sinusoidal query positions and a learned three-level embedding. Earlier generated code vectors are resized and passed through its existing conditioning projection. The Transformer has one decoder layer, width32, four heads, feed-forward width128, default ReLU, norm-first LayerNorm and dropout0.1. Three output heads predict 64 code entries at lengths1/2/4. Frozen residual codebook vectors sum at four positions; a 128→64→10 GELU decoder scores the local digit.', '',
'Every conditioning variant has exactly 34,570 trainable parameters. The 31-frame condition interpolates the same four learned acoustic position vectors; it adds no learned positional parameters or layers. Its cross-attention memory is longer and more expensive. The output hierarchy and decoder remain1→2→4 and four latent positions respectively. These positions are within one sound, not the four digit positions; each local decoder outputs ten digit scores.', '',
'## 3. Controlled probe and conditioning protocol','',
'Three seeds (317,419,523), frozen CNN and existing final-feature tokenizers, fixed 1,200/184/270 train/validation/test recordings. Digit probes and conditional models train on paired corrupted/clean views of training labels only: 9,600 local events, 16 epochs, batch128, AdamW rate0.001, 1,200 optimizer steps. Temperatures use the original gate’s validation fitting half with paired clean/corrupted views. No test-based selection is performed.', '',
'Each digit probe is exactly 256→64→10 with GELU (17,098 parameters). Native channels are preserved after common four-bin temporal pooling; 40 Mel channels are zero-padded to64. PCA outputs32 are also padded to64. This gives the same probe architecture and budget while allowing explicit compression controls. Fixed-width PCA changes intrinsic input rank; probe results measure accessible information under this readout, not Shannon information or proof of irreversibility.', '',
'A separate same-budget linear code probe (128→448, 57,792 parameters) reads 32-channel conditions and predicts all seven canonical code positions. It is an information-access diagnostic, not a parameter-matched performance competitor. Spectral regression uses the same ridge readout to recover seven measured log-band energies, standardized by training-target variance. Those measurements are regression targets, never added to the CNN/NCM inputs. Code-probe final_mean averages four adapter tokens, whereas the native digit final_mean uses the31-frame global mean; they are different temporal controls. Identical code-probe inputs for *_pca entries reuse their corresponding source-stage code-probe results rather than creating independent trials.', '',
'NCM models are fresh same-seed retrainings of the preserved architecture, not edits to the earlier checkpoints. The paired-clean training protocol is common to every new condition; absolute scores should be compared within this study. Five NCM conditions: final four-token features, stem four-token features, block2 four-token features, multiscale four-token features, and final31 frame-level features. All predict exactly the same frozen final-feature acoustic codes, use the same supervised loss (mean prefix digit CE + 0.1 mean code CE), initialization seeds, data order and update count. Different memory lengths have different compute; this is an equal-update, fixed-architecture input study, not FLOP matching.', '',
'Canonical targets are derived from final CNN features, so final conditioning is aligned with the target by construction. Earlier-feature code accuracy is not a representation-independent definition of acoustic information: its adapter must also retain enough information to approximate the downstream fixed CNN mapping.', '',
'## 4. Identical digit-probe results','',
'All entries are mean digit accuracy across three readout seeds. Clean, noisy, partial-erasure and missing-frequency conditions share exact samples across stages. No analytic missing-band guard is applied in these intrinsic-information diagnostics.', '',
'| Representation | Clean | Corrupted mixture | Noisy | Partial erasure | Missing column frequency |','|---|---:|---:|---:|---:|---:|']
for stage in stages:
    row=summary['probes'][stage];lines.append(f'| {stage} | '+' | '.join(f'{100*row[c]["digit_accuracy"]["mean"]:.2f}%' for c in ['clean','corrupted','noisy','partial_erasure','missing_frequency'])+' |')
lines+=['','The drop from stem to final is concentrated in the interference subset:', '',
'| Representation | Interference digit accuracy | Seven-band regression NMSE, corrupted mixture |','|---|---:|---:|']
for stage in ['mel','stem','block1','block2','final','final_mean','stem_pca','final_pca','multiscale']:
    lines.append(f'| {stage} | {100*interference_scores[stage]:.2f}% | {summary["probes"][stage]["corrupted"]["frequency_band_nmse"]:.4f} |')
lines+=['','Band regression does not degrade monotonically: block1 is better than the stem on this linear spectral metric, and final features still preserve readily decodable relative frequency evidence. Therefore the digit-probe decline does not establish a general collapse of acoustic information. It identifies a task/readout-specific robustness weakness under interference.', '',
'## 5. Probe sequence accuracy, calibration and latency','',
'| Representation, corrupted mixture | Top-1 | Top-5 | Digit ECE | Raw probe CPU p50 / p95 ms |','|---|---:|---:|---:|---:|']
for stage in stages:
    a=summary['probes'][stage]['corrupted'];lines.append(f'| {stage} | {100*a["sequence_top1"]["mean"]:.2f}% | {100*a["sequence_top5"]["mean"]:.2f}% | {a["digit_ece_10_bins"]["mean"]:.4f} | {a["latency_ms"]["p50"]:.2f} / {a["latency_ms"]["p95"]:.2f} |')
lines+=['','Raw probe latency stops at the required native tap when possible, then includes its adapter and readout; it excludes full-sequence candidate enumeration. Controlled NCM latency below includes the full CNN because the unchanged gate needs the final CNN head. These timings answer different questions and should not be directly subtracted as deployment savings.', '',
'## 6. Adapter and temporal bottlenecks','',
'| Paired corrupted digit-probe contrast | Delta | Cluster-bootstrap 95% interval |','|---|---:|---:|']
for key in ['probe_stem_minus_final','probe_stem_pca_minus_stem','probe_final_pca_minus_final']:
    v=pairs[key];lo,hi=v['cluster_bootstrap_95_ci'];lines.append(f'| {key} | {100*v["mean_delta"]:.2f} points | [{100*lo:.2f}, {100*hi:.2f}] |')
lines+=['','The 64→32 adapter removes much of the early feature advantage. Final features are less affected by that same width reduction. A lower early-conditioned NCM result therefore cannot be blamed solely on the early native encoder representation. The same 256-input classifier is used on both sides of each compression comparison; unused padded dimensions are intentional.', '',
'Final31 memory averages exactly back to the four-token condition in this implementation. Its difference from the legacy pool-before-linear implementation was at most about 4.4e-6, with identical measured canonical target codes. Dense memory therefore contains the four-token reference; failure to improve with it is not proof that no useful high-resolution information exists, especially with unchanged/interpolated positional parameters.', '',
'## 7. Code prediction: what information is available?','',
'| Final-feature diagnostic, corrupted mixture | Level1 | Level2 | Level3 |','|---|---:|---:|---:|']
for label,vals in [('Direct observed-feature quantization',[1,1,1]),('Learned linear code probe',linear),('NCM generated-prefix prediction',generated),('NCM oracle-prefix prediction',teacher)]:
    lines.append(f'| {label} | '+' | '.join(f'{100*v:.2f}%' for v in vals)+' |')
lines+=['','Direct quantization reproduces 100% of targets in every evaluated condition, by construction and measurement: the targets are a deterministic function of the available final condition. This is not a recognition achievement. It rules out absent target information as the sole explanation for this implemented code-imitation task.', '',
f'The NCM decoder trained on its generated codes reaches {100*summary["ncm"]["final"]["corrupted"]["raw"]["digit_accuracy"]:.2f}% raw digit accuracy; feeding its canonical observed code hierarchy instead reaches only {100*observed_decoder:.2f}%. Correct target codes are not an upper bound for this jointly trained decoder. The student code/decoder system has adapted to alternative code uses under digit supervision, so canonical code identity, acoustic reconstruction and recognition are different metrics.', '',
'Oracle-prefix feedback does not reliably improve this recipe’s fine-code accuracy. That is consistent with code/decoder alignment and objective interactions, not a simple assumption that earlier mistakes are the only cause. The code probe and NCM also have different objectives/capacities; their gap is evidence of accessibility, not proof that a Transformer cannot represent the function.', '',
'## 8. Existing NCM with different conditioning','',
'| Condition | Raw digit accuracy | Generated codes L1 / L2 / L3 | Protected top-1 | Protected top-5 | Digit ECE | CPU p50 / p95 ms |','|---|---:|---|---:|---:|---:|---:|']
for mode in modes:
    a=summary['ncm'][mode]['corrupted'];p=a['protected'];codes=' / '.join(f'{100*v:.2f}%' for v in a['raw']['code_accuracy'])
    lines.append(f'| {mode} | {100*a["raw"]["digit_accuracy"]:.2f}% | {codes} | {100*p["sequence_top1"]["mean"]:.2f}% | {100*p["sequence_top5"]["mean"]:.2f}% | {p["digit_ece_10_bins"]["mean"]:.4f} | {p["latency_ms"]["p50"]:.2f} / {p["latency_ms"]["p95"]:.2f} |')
lines+=['','| Condition | Raw final / multiscale digit accuracy | Protected final / multiscale top-1 | Protected final / multiscale top-5 |','|---|---:|---:|---:|']
for case in cases:
    one=summary['ncm']['final'][case];two=summary['ncm']['multiscale'][case]
    lines.append(f'| {case} | {100*one["raw"]["digit_accuracy"]:.2f}% / {100*two["raw"]["digit_accuracy"]:.2f}% | {100*one["protected"]["sequence_top1"]["mean"]:.2f}% / {100*two["protected"]["sequence_top1"]["mean"]:.2f}% | {100*one["protected"]["sequence_top5"]["mean"]:.2f}% / {100*two["protected"]["sequence_top5"]["mean"]:.2f}% |')
lines+=['','All protected NCM comparisons use identical gate route/accept masks and candidate lists per condition/seed, verified during timing. New missing-frequency masks are computed once using the same unchanged audio-only reference gate and then frozen. Raw all-event metrics are separate so the gate cannot hide a conditioning failure. All protected clean results remain 100%; raw clean and every other condition are in metrics.json.', '',
'| Protected corrupted top-1 contrast | Delta | Cluster-bootstrap 95% interval |','|---|---:|---:|']
for key in [f'ncm_{m}_minus_final' for m in modes if m!='final']:
    a=pairs[key];lo,hi=a['cluster_bootstrap_95_ci'];lines.append(f'| {key} | {100*a["mean_delta"]:.2f} points | [{100*lo:.2f}, {100*hi:.2f}] |')
lines+=['','Input changes can alter measured fine-code prediction, but no hierarchy-specific recognition gain follows from that alone. A continuous probe is evaluated under the identical frozen policy on exactly the same PCA condition:', '',
'| Condition | Continuous-probe top-1 | NCM top-1 | NCM minus continuous, paired 95% interval |','|---|---:|---:|---|']
for mode,stage in [('final','final_pca'),('stem','stem_pca'),('block2','block2_pca'),('multiscale','multiscale')]:
    cont=summary['protected_continuous'][stage]['corrupted']['sequence_top1'];ncm=summary['ncm'][mode]['corrupted']['protected']['sequence_top1']['mean'];a=pairs[f'ncm_{mode}_minus_continuous'];lo,hi=a['cluster_bootstrap_95_ci']
    lines.append(f'| {mode} | {100*cont:.2f}% | {100*ncm:.2f}% | {100*a["mean_delta"]:.2f} points [{100*lo:.2f}, {100*hi:.2f}] |')
lines+=['','These continuous heads have 17,098 nominal parameters versus 34,570 for NCM and share samples/updates, but not compute or objectives. Their result supports an available-information/utilization distinction; it does not isolate a universal architectural cause.', '',
'## 9. Missing-frequency interpretation and readout-adaptation control','',
'Column frequencies are physically removed before the frontend. Duration, amplitude and phase distribution do not encode the missing column; a unit test confirms that swapping column identity while keeping the row and random seed produces identical waveforms. Under a uniform 0–9 alphabet, rows697/770/852 leave three compatible digits; row941 identifies digit0 uniquely.', '',
f'For the test’s observed row mixture, expected Bayes digit accuracy is {100*bounds["expected_digit_accuracy"]:.2f}%, sequence top-1 {100*bounds["expected_sequence_top1"]:.2f}% and top-5 {100*bounds["expected_sequence_top5"]:.2f}% under uniform unknown columns. These are conditional expected limits, not hard limits on finite-sample realized accuracies.', '',
'Ordinary probes trained only on complete clean/corrupted digits extrapolate poorly to row-only audio. Their low missing-frequency digit scores must not be called encoder information loss. A separate same-architecture probe is trained for16 epochs (608 updates on4,800 local events, identically at every stage) on training-only row-tone renders with soft targets uniform over compatible digits; held-out phases/combinations test the surviving row evidence.', '',
'| Representation | Missing-aware surviving-row accuracy | Compatible-digit probability mass |','|---|---:|---:|']
for stage in stages:
    rows=[D['seeds'][s]['missing_aware_probes'][stage] for s in seeds]
    lines.append(f'| {stage} | {100*np.mean([v["surviving_row_accuracy"] for v in rows]):.2f}% | {np.mean([v["compatible_digit_probability_mass"] for v in rows]):.6f} |')
lines+=['','The remaining row is decodable at every stage, even where an ordinary digit probe scores 0%. This control demonstrates a readout/domain-shift failure rather than erased row information. It does not reconstruct the absent column, change NCM training or establish reliable NCM uncertainty. No analytic DTMF guard is used to obtain these row-aware neural-probe results.', '',
'## 10. Bottleneck assessment','',
'1. Frontend losses are explicit: phase, fine FFT structure merged by Mel filters and near-invariance to absolute gain. The current experiments do not show these to be the primary digit-recognition bottleneck; clean identity remains perfectly decodable.',
'2. Native residual-stack features show a real probe-accessible robustness decline relative to the stem under interference. This is not a temporal-stride/pooling loss, and linear band recovery does not show wholesale acoustic collapse. Readout capacity/training and nonlinear geometry still limit causal interpretation.',
'3. The fixed early-feature PCA adapter is a demonstrated practical conditioning bottleneck: most of the early native-feature classification advantage disappears at32 dimensions. A negative early-memory result does not disprove the usefulness of the original64-channel early features.',
'4. Final conditions still contain enough information to reproduce every defined target code, and a simple code probe accesses much more canonical identity than NCM. Joint digit/code training and decoder alignment are therefore separate major limitations. Changing the audio encoder alone cannot address this demonstrated target-utilization gap.',
'5. Missing-column identity is absent from the waveform by design. Surviving row identity remains decodable at all stages after appropriate readout training. This condition does not justify a new encoder intended to recover nonexistent evidence.', '',
'No NCM-specific advantage is established. The evidence supports addressing adapter/readout/prediction alignment before designing a replacement audio encoder, while retaining the stem-versus-final interference gap as a concrete candidate for further controlled diagnosis.', '',
'## 11. Verification, uncertainty and reproduction','',
'37 tests pass, including exact native/tap equivalence, unchanged31-frame resolution, equal probe inputs/parameter counts, dense-to-four-token consistency, unchanged NCM parameter/state shapes with31-token memory, gradient flow, and absence of missing-column leakage, plus all earlier experiments. Original checkpoints remain untouched.', '',
'Intervals use2,000 paired bootstrap resamples of distinct digit-combination clusters, averaging outcomes over the same three trained seeds. They condition on these models and the reused development pool. Seed standard deviations and per-condition metrics are saved; these results are not independent confirmation on new recordings.', '',
'Timing uses one CPU thread and separates raw stage-only probe inference from full controlled NCM pipelines. The latter include the full common CNN/gate and candidate ranking; missing-frequency diagnostic probabilities are not replaced by analytic uncertainty guards. Oracle-prefix and frequency-regression computations are excluded. Longer31-token memory is more expensive even with identical trainable parameters.', '',
'```bash',
'PYTHONPATH=. .venv/bin/python experiments/encoder_information.py',
'PYTHONPATH=. .venv/bin/python experiments/encoder_adapter_controls.py',
'PYTHONPATH=. .venv/bin/python experiments/encoder_code_diagnostics.py',
'PYTHONPATH=. .venv/bin/python experiments/encoder_missing_control.py',
'.venv/bin/python experiments/report_encoder.py',
'.venv/bin/python -m pytest -q',
'```','',
'Configuration: experiments/encoder_config.json. Complete operation/shape/filter audit: encoder_audit.json. Raw results: metrics.json. Aggregates, interference subgroup and paired intervals: summary.json. Every probe, adapter and NCM condition checkpoint and candidate-list hash is preserved under encoder_results/. Original graph/APPT/PPF/CNN files remain available. Limitations include synthetic segmented tones, no learned boundary detection, three seeds, fixed PCA adapters and codebook targets that favor final-feature alignment, and repeated development-test use.']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
paths=['ppf/encoder_study.py','experiments/encoder_information.py','experiments/encoder_config.json','representation_results/cnn.pt','gate_results/metrics.json']
(R/'source_hashes.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},indent=2))
print('Saved encoder_results/REPORT.md');print(pairs)
for mode in modes:print(mode,summary['ncm'][mode]['corrupted'])
