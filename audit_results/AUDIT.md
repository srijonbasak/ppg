# Implementation audit — 2026-10-31

This audits the existing repository against the user-approved specification. The source architecture has been preserved. Experiment results are in comparison.json; the experiment is an existing-prototype comparison, not a completed test of the approved architecture.

## Findings

| Requirement | Status | Evidence |
|---|---|---|
| CNN audio conditioning | Pass | AcousticEncoder: log-Mel frontend, residual temporal convolutions; predictor cross-attends to audio memory. |
| Clean-label target tokenizer | Fail | model.py forward tokenizes the same noisy audio CNN latent. No label encoder exists. |
| Residual vector quantization | Pass structurally | Each codebook quantizes the interpolated target minus previous-level reconstruction. Training effectiveness is separate. |
| Nested dropout | Fail | No level dropout or prefix sampling exists. Ordinary Transformer dropout is unrelated. |
| Fixed 1/2/4 code positions | Fail | Level positions depend on waveform frame count: factors 4/2/1, not fixed counts. |
| Two-stage frozen tokenizer training | Fail | engine.train jointly optimizes all parameters from initialization. |
| Conditional next-level prediction | Pass structurally | forward_level uses audio and earlier-level code embeddings. |
| Parallel predictions within a level | Pass structurally | Unmasked Transformer decoder predicts all positions in one pass. |
| Independent conditional sampling | Partial | predict_latent(sample=True, temperature=1) uses Categorical sampling, sequential levels. Production possibilities() uses argmax, not samples. Experiment invokes eval mode and independent batched draws. |
| Latent-only decoder | Pass, wrong output scope | Decoder receives latent reconstruction only, returns one digit per isolated tone, not four-digit outputs. |
| Mixture sequence ranking | Fail in application | possibilities() returns top digits from one deterministic hierarchy. Added experiment adapter enumerates 10,000 sequences and averages full-sequence probability products. |
| Target-free generated inference | Pass for predict_latent | Does not call tokenizer.forward or accept labels. |
| Correct original evaluator | Fail, corrected | evaluate_neural previously called forward(), conditioning later levels on tokenizer-derived codes from audio. This is teacher-code dependence and a deployment mismatch; not direct ground-truth label leakage. It now uses predict_latent then decoder. Regression test prevents tokenizer access. |
| CNN exhaustive sequence ranking | Missing in application; added in experiment | Existing CNN is a single-tone classifier. Adapter groups four events and enumerates all 10,000 products. |
| Dataset integrity | Partial | Seeded synthesis and file-level split membership exist. No built-in duplicate guard, strict erasure controls, or distinct OOD corruptions. Experiment verifies no duplicate WAV hashes across its 600 samples. |
| Reproducible evaluation | Partial | Fixed data/training seeds and saved checkpoints exist. Original train has no validation selection, calibration, latency, or multiple-seed report. Added harness records these metrics without test-based selection. |

## Tensor flow

Default configuration, a batch of isolated 2,400-sample events:

- waveform: [B, 2400]
- log-Mel: [B, 40, 31]
- CNN: [B, 64, 31]
- tokenizer target: [B, 31, 64] (audio features, not labels)
- coarse/middle/fine codes: [B, 8], [B, 16], [B, 31]
- code logits: [B, 8, 64], [B, 16, 64], [B, 31, 64]
- summed reconstructed latent: [B, 31, 64]
- decoder digit logits: [B, 10]

Training uses tokenizer earlier levels and expected code embeddings from softmax distributions. Inference uses generated discrete earlier levels. This introduces both teacher-forcing and soft-to-discrete mismatch.

Dataset loading uses manifest start/end positions to crop each true tone and pads to 2,400 samples. This is oracle segmentation, not end-to-end sequence recognition. server.py additionally truncates an uploaded recording to its first 2,400 samples; it cannot recognize a whole four-digit recording.

## Verification

Original tests: 4 passed. After evaluator fix and regression test: 5 passed. Smoke command passed forward/backward, generation wiring and Goertzel detection for 13 events. Smoke level lengths were 8/16/31 (small codebook size 16). Passing these tests does not establish specification compliance or recognition quality.

## Controlled experiment protocol

Run: `PYTHONPATH=. .venv/bin/python experiments/audit_comparison.py`

- Data seed 20261031; 400 training, 80 validation and 120 test sequences, exactly four uniformly random digits each.
- Identical noisy WAVs and oracle event crops for both models. No waveform duplicates by SHA-256.
- Training seeds 7, 17, 27; six epochs fixed before testing; default models, batch 32, AdamW, learning rate 0.0003. No tuning or early stopping on test data. Validation split is reserved but unused for model selection or calibration.
- Both models train on 1,600 isolated tone events. Test set has 480 events grouped into 120 sequences.
- CNN ranks all 10,000 products. PPF samples M=1/4/8/16 full tuples of four independent event hierarchies at temperature 1 in evaluation mode; each tuple defines a four-digit factorized distribution. Full sequence probabilities are averaged across tuples. This is a measured adapter over the current model, not the approved clean-label sequence hierarchy.
- CPU, one PyTorch thread. Latency includes waveform-to-probability inference and exhaustive sequence ranking for one four-digit sequence, excludes file loading, diagnostics and model initialization. Three warmup calls. GPU unavailable; GPU memory is not applicable.
- NLL uses natural logarithms. Sequence Brier score sums squared errors across 10,000 outcomes. ECE uses ten equal-width confidence bins. Scores are uncalibrated; no validation temperature fitting.
- Teacher reconstruction diagnostics measure digit accuracy from audio-tokenizer latents at each prefix. They are not clean-label tokenizer reconstruction metrics.

## Limits and next implementation milestone

The present experiment cannot establish whether the finalized NCM-inspired hierarchy is beneficial: its target tokenizer, output scope and training protocol differ. It can detect failures in the existing prototype and establish a CNN reference. The test set is small, synthetic, in-distribution and uses oracle segmentation. No strict erasure/OOD study, matched-compute tuning, or real audio study has been performed. Seed variability on the same test set is not an independent-data confidence interval.

Priority implementation gaps: clean-label sequence tokenizer and four-position decoder; nested dropout and fixed scales; staged training; production sampled-mixture ranking; sequence-level data/inference; strict information-loss and OOD controls. These are implementation gaps in the agreed contract, not proposed additions to it.
