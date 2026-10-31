# Audio encoder information-preservation study — 2026-10-31

## Main findings

The CNN does not perform temporal downsampling. A measurable decline in digit-probe performance appears through its learned residual stack under interference, but substantial useful evidence remains in the final features. The 32-channel PCA adapter discards much of the advantage of early features. The NCM model then underuses the remaining information and also learns a code/decoder mapping that differs from its canonical acoustic targets. These findings do not justify replacing the audio encoder as the first intervention.

No original CNN, NCM, graph, gate or checkpoint was overwritten. No new audio encoder was proposed or trained. The existing CNN-only NCM Transformer architecture and final-feature codebooks are preserved; only its conditioning source changes.

## 1. Exact encoder implementation

Input is a float32 mono tone window [B,2400] at 8 kHz (300 ms). Current generated tones last 180 ms (1,440 samples), then receive 960 zero-padding samples; the study uses supplied event boundaries, not learned segmentation.

| Operation | Output shape | Sampling/resolution or context |
|---|---|---|
| Centered STFT: Hann 256, hop 80, FFT 256, reflect padding 128 | [B,129,31] complex | 32 ms windows, 10 ms frame grid, 31.25 Hz FFT-bin spacing, 0–4 kHz |
| Magnitude squared | [B,129,31] real | Phase is discarded; bin spacing is not a guarantee of two-tone resolvability. |
| Fixed triangular 40-band Mel matrix | [B,40,31] | Nonuniform overlapping bands from nominal 20–4,000 Hz; actual integer-bin supports in encoder_audit.json. |
| Relative floor: log(max(mel + 0.001*mean(mel),1e-6)) | [B,40,31] | Same corrected frontend as prior study; no condition-specific branch. |
| Per-event mean subtraction and std division | [B,40,31] | Statistics over all frequency/time cells; default corrected std, clamped at 1e-4. |
| Conv1d 40→64, k5, stride1, padding2, bias | [B,64,31] | Nominal convolution support 5 frames / 72 ms. No separate stem activation or pooling. |
| Residual block 1, dilation1 | [B,64,31] | Nominal cumulative support 7 frames / 92 ms. |
| Residual block 2, dilation2 | [B,64,31] | Nominal cumulative support 11 frames / 132 ms. |
| Residual block 3, dilation4 | [B,64,31] | Nominal cumulative support 19 frames / 212 ms. |
| Existing classifier: AdaptiveAvgPool1d(1), flatten, Linear64→10 | [B,10] | This head pools time; the encoder itself does not. |

Each residual block performs exactly Conv1d64→64(k3, stride1, padding=dilation, dilation=dilation, bias) → GroupNorm(4 groups, 16 channels/group, eps1e-5, affine) → GELU → Conv1d64→64(k1, stride1, bias) → GroupNorm(4 groups) → add block input → GELU. Thus the CNN contains seven convolutions, six GroupNorms and six GELUs. There is no max pooling, no strided convolution, no dropout and no embedding lookup inside the CNN.

After the stem, the frequency axis is mixed into learned channels: 64 channels do not represent 64 evenly spaced frequency bins. Nominal convolution supports above do not equal strict receptive fields: frontend statistics/floor and every GroupNorm use all time frames, giving global event dependence. The frame grid remains 31×10 ms throughout.

Manual tap extraction exactly matched native encoder output: True. Halving waveform gain changed normalized frontend values by at most 4.77e-07. Absolute gain is substantially removed before the stem; this intended invariance is not demonstrated to be the DTMF recognition failure.

Actual Mel supports in the DTMF range are nonuniform: around the row frequencies, nonzero support widths span roughly 62.5–93.75 Hz on the discretized grid; higher digit columns involve wider supports. Full per-filter peaks/nonzero supports and every convolution/normalization hook shape are exported rather than treating channels as a uniform frequency grid.

## 2. Post-encoder pooling, projection and NCM embeddings

The prior local adapter reduces 31 frames to four temporal bins, then standardizes channels with training statistics, applies a 64→32 PCA matrix and component whitening. Four adaptive pooling bins overlap at boundary indices (0–7, 7–15, 15–23, 23–30). This reduction occurs outside the CNN.

Early-stage adapters use the same train-corrupted-only pooling/standardization/PCA recipe. Final uses the preserved projection buffers. Multiscale concatenates stem/block1/block2/final channels and projects 256→32, retaining four time positions. These projections are fixed, not new trainable encoder layers. Probe input buffers additionally standardize each flattened feature using training statistics; NCM receives channel-whitened tokens, so readout comparisons establish accessibility rather than perfectly matched normalization/optimization. Multiscale stores more fixed projection coefficients, although NCM trainable size is unchanged.

NCM uses a learned 4×32 acoustic positional embedding, fixed sinusoidal query positions and a learned three-level embedding. Earlier generated code vectors are resized and passed through its existing conditioning projection. The Transformer has one decoder layer, width32, four heads, feed-forward width128, default ReLU, norm-first LayerNorm and dropout0.1. Three output heads predict 64 code entries at lengths1/2/4. Frozen residual codebook vectors sum at four positions; a 128→64→10 GELU decoder scores the local digit.

Every conditioning variant has exactly 34,570 trainable parameters. The 31-frame condition interpolates the same four learned acoustic position vectors; it adds no learned positional parameters or layers. Its cross-attention memory is longer and more expensive. The output hierarchy and decoder remain1→2→4 and four latent positions respectively. These positions are within one sound, not the four digit positions; each local decoder outputs ten digit scores.

## 3. Controlled probe and conditioning protocol

Three seeds (317,419,523), frozen CNN and existing final-feature tokenizers, fixed 1,200/184/270 train/validation/test recordings. Digit probes and conditional models train on paired corrupted/clean views of training labels only: 9,600 local events, 16 epochs, batch128, AdamW rate0.001, 1,200 optimizer steps. Temperatures use the original gate’s validation fitting half with paired clean/corrupted views. No test-based selection is performed.

Each digit probe is exactly 256→64→10 with GELU (17,098 parameters). Native channels are preserved after common four-bin temporal pooling; 40 Mel channels are zero-padded to64. PCA outputs32 are also padded to64. This gives the same probe architecture and budget while allowing explicit compression controls. Fixed-width PCA changes intrinsic input rank; probe results measure accessible information under this readout, not Shannon information or proof of irreversibility.

A separate same-budget linear code probe (128→448, 57,792 parameters) reads 32-channel conditions and predicts all seven canonical code positions. It is an information-access diagnostic, not a parameter-matched performance competitor. Spectral regression uses the same ridge readout to recover seven measured log-band energies, standardized by training-target variance. Those measurements are regression targets, never added to the CNN/NCM inputs. Code-probe final_mean averages four adapter tokens, whereas the native digit final_mean uses the31-frame global mean; they are different temporal controls. Identical code-probe inputs for *_pca entries reuse their corresponding source-stage code-probe results rather than creating independent trials.

NCM models are fresh same-seed retrainings of the preserved architecture, not edits to the earlier checkpoints. The paired-clean training protocol is common to every new condition; absolute scores should be compared within this study. Five NCM conditions: final four-token features, stem four-token features, block2 four-token features, multiscale four-token features, and final31 frame-level features. All predict exactly the same frozen final-feature acoustic codes, use the same supervised loss (mean prefix digit CE + 0.1 mean code CE), initialization seeds, data order and update count. Different memory lengths have different compute; this is an equal-update, fixed-architecture input study, not FLOP matching.

Canonical targets are derived from final CNN features, so final conditioning is aligned with the target by construction. Earlier-feature code accuracy is not a representation-independent definition of acoustic information: its adapter must also retain enough information to approximate the downstream fixed CNN mapping.

## 4. Identical digit-probe results

All entries are mean digit accuracy across three readout seeds. Clean, noisy, partial-erasure and missing-frequency conditions share exact samples across stages. No analytic missing-band guard is applied in these intrinsic-information diagnostics.

| Representation | Clean | Corrupted mixture | Noisy | Partial erasure | Missing column frequency |
|---|---:|---:|---:|---:|---:|
| mel | 100.00% | 88.27% | 96.94% | 94.53% | 39.35% |
| stem | 100.00% | 95.43% | 99.14% | 98.01% | 13.80% |
| block1 | 100.00% | 94.17% | 99.51% | 98.26% | 14.97% |
| block2 | 100.00% | 92.96% | 99.02% | 97.76% | 10.37% |
| final | 100.00% | 92.41% | 98.16% | 97.89% | 0.00% |
| final_mean | 100.00% | 91.85% | 97.79% | 96.39% | 0.00% |
| final_pca | 100.00% | 92.35% | 98.53% | 96.39% | 7.16% |
| multiscale | 100.00% | 93.64% | 98.77% | 97.89% | 13.21% |
| stem_pca | 100.00% | 91.94% | 97.67% | 96.14% | 30.65% |
| block2_pca | 100.00% | 93.30% | 98.53% | 96.77% | 0.00% |

The drop from stem to final is concentrated in the interference subset:

| Representation | Interference digit accuracy | Seven-band regression NMSE, corrupted mixture |
|---|---:|---:|
| mel | 61.32% | 0.0180 |
| stem | 84.58% | 0.0173 |
| block1 | 78.73% | 0.0088 |
| block2 | 74.88% | 0.0119 |
| final | 73.38% | 0.0176 |
| final_mean | 73.01% | 0.0476 |
| stem_pca | 73.76% | 0.0186 |
| final_pca | 74.25% | 0.0368 |
| multiscale | 77.74% | 0.0152 |

Band regression does not degrade monotonically: block1 is better than the stem on this linear spectral metric, and final features still preserve readily decodable relative frequency evidence. Therefore the digit-probe decline does not establish a general collapse of acoustic information. It identifies a task/readout-specific robustness weakness under interference.

## 5. Probe sequence accuracy, calibration and latency

| Representation, corrupted mixture | Top-1 | Top-5 | Digit ECE | Raw probe CPU p50 / p95 ms |
|---|---:|---:|---:|---:|
| mel | 70.74% | 86.17% | 0.0288 | 0.22 / 0.24 |
| stem | 85.80% | 97.04% | 0.0105 | 0.30 / 0.33 |
| block1 | 82.72% | 95.56% | 0.0188 | 0.43 / 0.46 |
| block2 | 79.26% | 93.70% | 0.0193 | 0.55 / 0.59 |
| final | 78.27% | 92.22% | 0.0170 | 0.66 / 0.71 |
| final_mean | 77.16% | 92.10% | 0.0118 | 0.67 / 0.83 |
| final_pca | 77.90% | 91.85% | 0.0223 | 0.68 / 0.71 |
| multiscale | 81.48% | 94.69% | 0.0171 | 0.72 / 0.88 |
| stem_pca | 76.91% | 93.33% | 0.0229 | 0.31 / 0.33 |
| block2_pca | 80.00% | 94.69% | 0.0168 | 0.55 / 0.61 |

Raw probe latency stops at the required native tap when possible, then includes its adapter and readout; it excludes full-sequence candidate enumeration. Controlled NCM latency below includes the full CNN because the unchanged gate needs the final CNN head. These timings answer different questions and should not be directly subtracted as deployment savings.

## 6. Adapter and temporal bottlenecks

| Paired corrupted digit-probe contrast | Delta | Cluster-bootstrap 95% interval |
|---|---:|---:|
| probe_stem_minus_final | 3.02 points | [1.97, 4.15] |
| probe_stem_pca_minus_stem | -3.49 points | [-4.93, -2.17] |
| probe_final_pca_minus_final | -0.06 points | [-0.77, 0.65] |

The 64→32 adapter removes much of the early feature advantage. Final features are less affected by that same width reduction. A lower early-conditioned NCM result therefore cannot be blamed solely on the early native encoder representation. The same 256-input classifier is used on both sides of each compression comparison; unused padded dimensions are intentional.

Final31 memory averages exactly back to the four-token condition in this implementation. Its difference from the legacy pool-before-linear implementation was at most about 4.4e-6, with identical measured canonical target codes. Dense memory therefore contains the four-token reference; failure to improve with it is not proof that no useful high-resolution information exists, especially with unchanged/interpolated positional parameters.

## 7. Code prediction: what information is available?

| Final-feature diagnostic, corrupted mixture | Level1 | Level2 | Level3 |
|---|---:|---:|---:|
| Direct observed-feature quantization | 100.00% | 100.00% | 100.00% |
| Learned linear code probe | 94.17% | 89.54% | 75.76% |
| NCM generated-prefix prediction | 40.37% | 65.15% | 51.40% |
| NCM oracle-prefix prediction | 40.37% | 55.42% | 50.53% |

Direct quantization reproduces 100% of targets in every evaluated condition, by construction and measurement: the targets are a deterministic function of the available final condition. This is not a recognition achievement. It rules out absent target information as the sole explanation for this implemented code-imitation task.

The NCM decoder trained on its generated codes reaches 88.95% raw digit accuracy; feeding its canonical observed code hierarchy instead reaches only 72.44%. Correct target codes are not an upper bound for this jointly trained decoder. The student code/decoder system has adapted to alternative code uses under digit supervision, so canonical code identity, acoustic reconstruction and recognition are different metrics.

Oracle-prefix feedback does not reliably improve this recipe’s fine-code accuracy. That is consistent with code/decoder alignment and objective interactions, not a simple assumption that earlier mistakes are the only cause. The code probe and NCM also have different objectives/capacities; their gap is evidence of accessibility, not proof that a Transformer cannot represent the function.

## 8. Existing NCM with different conditioning

| Condition | Raw digit accuracy | Generated codes L1 / L2 / L3 | Protected top-1 | Protected top-5 | Digit ECE | CPU p50 / p95 ms |
|---|---:|---|---:|---:|---:|---:|
| final | 88.95% | 40.37% / 65.15% / 51.40% | 69.38% | 78.27% | 0.0435 | 1.39 / 2.03 |
| stem | 88.30% | 30.74% / 62.25% / 50.58% | 70.62% | 79.01% | 0.0416 | 1.38 / 2.03 |
| block2 | 89.14% | 30.62% / 66.54% / 53.42% | 69.88% | 78.89% | 0.0453 | 1.38 / 2.01 |
| multiscale | 88.98% | 38.89% / 65.05% / 53.97% | 70.62% | 79.01% | 0.0478 | 1.39 / 2.08 |
| final31 | 87.53% | 40.77% / 66.64% / 52.52% | 69.51% | 77.16% | 0.0467 | 1.42 / 2.12 |

| Condition | Raw final / multiscale digit accuracy | Protected final / multiscale top-1 | Protected final / multiscale top-5 |
|---|---:|---:|---:|
| clean | 100.00% / 100.00% | 100.00% / 100.00% | 100.00% / 100.00% |
| corrupted | 88.95% / 88.98% | 69.38% / 70.62% | 78.27% / 79.01% |
| noisy | 97.79% / 98.77% | 90.20% / 92.16% | 98.04% / 99.02% |
| partial_erasure | 95.65% / 96.27% | 74.63% / 75.62% | 84.58% / 84.58% |
| missing_frequency | 6.82% / 9.26% | 0.00% / 0.00% | 0.37% / 0.12% |

All protected NCM comparisons use identical gate route/accept masks and candidate lists per condition/seed, verified during timing. New missing-frequency masks are computed once using the same unchanged audio-only reference gate and then frozen. Raw all-event metrics are separate so the gate cannot hide a conditioning failure. All protected clean results remain 100%; raw clean and every other condition are in metrics.json.

| Protected corrupted top-1 contrast | Delta | Cluster-bootstrap 95% interval |
|---|---:|---:|
| ncm_stem_minus_final | 1.23 points | [-0.74, 3.21] |
| ncm_block2_minus_final | 0.49 points | [-0.99, 1.98] |
| ncm_multiscale_minus_final | 1.23 points | [-0.25, 2.96] |
| ncm_final31_minus_final | 0.12 points | [-1.37, 1.71] |

Input changes can alter measured fine-code prediction, but no hierarchy-specific recognition gain follows from that alone. A continuous probe is evaluated under the identical frozen policy on exactly the same PCA condition:

| Condition | Continuous-probe top-1 | NCM top-1 | NCM minus continuous, paired 95% interval |
|---|---:|---:|---|
| final | 72.47% | 69.38% | -3.09 points [-4.83, -1.37] |
| stem | 72.22% | 70.62% | -1.60 points [-3.44, 0.12] |
| block2 | 72.96% | 69.88% | -3.09 points [-4.82, -1.36] |
| multiscale | 73.58% | 70.62% | -2.96 points [-4.73, -1.13] |

These continuous heads have 17,098 nominal parameters versus 34,570 for NCM and share samples/updates, but not compute or objectives. Their result supports an available-information/utilization distinction; it does not isolate a universal architectural cause.

## 9. Missing-frequency interpretation and readout-adaptation control

Column frequencies are physically removed before the frontend. Duration, amplitude and phase distribution do not encode the missing column; a unit test confirms that swapping column identity while keeping the row and random seed produces identical waveforms. Under a uniform 0–9 alphabet, rows697/770/852 leave three compatible digits; row941 identifies digit0 uniquely.

For the test’s observed row mixture, expected Bayes digit accuracy is 39.69%, sequence top-1 2.47% and top-5 12.10% under uniform unknown columns. These are conditional expected limits, not hard limits on finite-sample realized accuracies.

Ordinary probes trained only on complete clean/corrupted digits extrapolate poorly to row-only audio. Their low missing-frequency digit scores must not be called encoder information loss. A separate same-architecture probe is trained for16 epochs (608 updates on4,800 local events, identically at every stage) on training-only row-tone renders with soft targets uniform over compatible digits; held-out phases/combinations test the surviving row evidence.

| Representation | Missing-aware surviving-row accuracy | Compatible-digit probability mass |
|---|---:|---:|
| mel | 100.00% | 0.999655 |
| stem | 100.00% | 0.999880 |
| block1 | 100.00% | 0.999897 |
| block2 | 100.00% | 0.999902 |
| final | 100.00% | 0.999904 |
| final_mean | 100.00% | 0.999907 |
| final_pca | 100.00% | 0.999833 |
| multiscale | 100.00% | 0.999827 |
| stem_pca | 100.00% | 0.999807 |
| block2_pca | 100.00% | 0.999830 |

The remaining row is decodable at every stage, even where an ordinary digit probe scores 0%. This control demonstrates a readout/domain-shift failure rather than erased row information. It does not reconstruct the absent column, change NCM training or establish reliable NCM uncertainty. No analytic DTMF guard is used to obtain these row-aware neural-probe results.

## 10. Bottleneck assessment

1. Frontend losses are explicit: phase, fine FFT structure merged by Mel filters and near-invariance to absolute gain. The current experiments do not show these to be the primary digit-recognition bottleneck; clean identity remains perfectly decodable.
2. Native residual-stack features show a real probe-accessible robustness decline relative to the stem under interference. This is not a temporal-stride/pooling loss, and linear band recovery does not show wholesale acoustic collapse. Readout capacity/training and nonlinear geometry still limit causal interpretation.
3. The fixed early-feature PCA adapter is a demonstrated practical conditioning bottleneck: most of the early native-feature classification advantage disappears at32 dimensions. A negative early-memory result does not disprove the usefulness of the original64-channel early features.
4. Final conditions still contain enough information to reproduce every defined target code, and a simple code probe accesses much more canonical identity than NCM. Joint digit/code training and decoder alignment are therefore separate major limitations. Changing the audio encoder alone cannot address this demonstrated target-utilization gap.
5. Missing-column identity is absent from the waveform by design. Surviving row identity remains decodable at all stages after appropriate readout training. This condition does not justify a new encoder intended to recover nonexistent evidence.

No NCM-specific advantage is established. The evidence supports addressing adapter/readout/prediction alignment before designing a replacement audio encoder, while retaining the stem-versus-final interference gap as a concrete candidate for further controlled diagnosis.

## 11. Verification, uncertainty and reproduction

37 tests pass, including exact native/tap equivalence, unchanged31-frame resolution, equal probe inputs/parameter counts, dense-to-four-token consistency, unchanged NCM parameter/state shapes with31-token memory, gradient flow, and absence of missing-column leakage, plus all earlier experiments. Original checkpoints remain untouched.

Intervals use2,000 paired bootstrap resamples of distinct digit-combination clusters, averaging outcomes over the same three trained seeds. They condition on these models and the reused development pool. Seed standard deviations and per-condition metrics are saved; these results are not independent confirmation on new recordings.

Timing uses one CPU thread and separates raw stage-only probe inference from full controlled NCM pipelines. The latter include the full common CNN/gate and candidate ranking; missing-frequency diagnostic probabilities are not replaced by analytic uncertainty guards. Oracle-prefix and frequency-regression computations are excluded. Longer31-token memory is more expensive even with identical trainable parameters.

```bash
PYTHONPATH=. .venv/bin/python experiments/encoder_information.py
PYTHONPATH=. .venv/bin/python experiments/encoder_adapter_controls.py
PYTHONPATH=. .venv/bin/python experiments/encoder_code_diagnostics.py
PYTHONPATH=. .venv/bin/python experiments/encoder_missing_control.py
.venv/bin/python experiments/report_encoder.py
.venv/bin/python -m pytest -q
```

Configuration: experiments/encoder_config.json. Complete operation/shape/filter audit: encoder_audit.json. Raw results: metrics.json. Aggregates, interference subgroup and paired intervals: summary.json. Every probe, adapter and NCM condition checkpoint and candidate-list hash is preserved under encoder_results/. Original graph/APPT/PPF/CNN files remain available. Limitations include synthetic segmented tones, no learned boundary detection, three seeds, fixed PCA adapters and codebook targets that favor final-feature alignment, and repeated development-test use.
