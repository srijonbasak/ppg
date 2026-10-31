# Staged representation-learning experiment

Run from the repository root:

```
PYTHONPATH=. .venv/bin/python experiments/staged_experiment.py --stage tokenizer
PYTHONPATH=. .venv/bin/python experiments/staged_experiment.py --stage audio
.venv/bin/python -m pytest -q
```

Configuration: experiments/staged_config.json. New modules live in ppf/sequence.py. The original CNN, log-Mel frontend, parallel Transformer predictor, original event model and UI remain available. The UI still uses the old event-model checkpoint format; it is not connected to the new sequence model.

## Implemented training path

Clean labels [B,4] -> digit/position embeddings -> bounded LayerNorm encoder [B,4,64] -> residual code levels [B,1], [B,2], [B,4] -> summed upsampled embeddings [B,4,64] -> latent-only decoder [B,4,10].

Residual quantization subtracts the earlier reconstructed representation at each scale. The encoder receives straight-through reconstruction gradients. Codebooks use K-means initialization and EMA counts/sums, with replacement of codes whose EMA count is below 0.1. There are no optimizer gradients for EMA codebooks. Loss combines full-hierarchy reconstruction, a uniformly sampled prefix reconstruction loss (nested dropout), and residual commitment MSE. The full hierarchy remains supervised each batch; dropout is on the auxiliary decoded prefix, not the full loss.

The target tokenizer/decoder train independently. Audio stage refuses to run without a saved validation-gate pass. Frozen target parameters and EMA buffers are checked for exact equality before and after conditional training.

Audio event windows [B,4,2400] -> existing residual CNN -> concatenated temporal memory with learned event-position offsets -> existing parallel level predictor -> 1/2/4 code positions. Teacher forcing uses clean target earlier codes during conditional training only. Generation samples earlier levels and carries those embeddings forward. Decoder receives latent codes only.

At inference, independent samples are drawn from the conditional categorical distributions at each level in evaluation mode, temperature 1. Equal-weight mixtures are exhaustively scored over all 10,000 sequences. M=4 is fixed before this experiment; no M or layer-count search is performed.

## Reconstruction gate and recorded failed run

Initial fixed-learning-rate run: final validation sequence reconstruction 98.8%, below the predeclared 99% gate. Generator training was blocked. Its checkpoint, log and measurements are preserved as tokenizer_initial_failed.*.

Revised protocol adds cosine learning-rate decay and selects the checkpoint with the best validation full-sequence reconstruction, earliest in a tie. The change was motivated by unstable validation reconstruction; the initial run's test metrics were already visible, so this is a development iteration rather than an untouched confirmatory study. No selection uses test metrics. Selected epoch and per-epoch validation histories are recorded in tokenizer_metrics.json.

Selected epoch 1: validation sequence reconstruction 100%, test sequence reconstruction 100% (1,000 held-out digit combinations each). Test sequence reconstruction after prefixes 1/2/3: 0.2%, 3.5%, 100%. Test digit reconstruction: 32.175%, 55.475%, 100%. Test unique codes: 43/61/63 of 64. Effective codebook perplexities: approximately 38.2/49.1/54.7. Low coarse reconstruction reflects limited information at a prefix and is reported rather than assumed to improve monotonically for every example.

Normalization, EMA initialization/updates, dead-code replacement and nested dropout are implemented together. This study does not isolate their individual effects. Greater code usage plus accurate reconstruction shows that this tokenizer avoids the old failure in this run; it does not prove every code is useful. The new representation targets differ from the previous audio tokenizer, so raw quantization losses are not directly comparable.

## Controlled audio protocol and limits

One seed (41), CPU, one thread, fixed eight epochs per audio training stage. Splits: 1,200 training, 200 validation, 300 test sequences. Each contains four independently random digits. Every split contains equal numbers of reference, additive noise, interfering-frequency, and partial-erasure recordings. The reference condition retains ordinary synthesis noise, jitter and distortion. All modifications occur before spectrogram extraction. Partial erasure replaces the center 60% of one tone and adds attenuation/noise; surviving information is intentional. This is not a complete-erasure chance-limit experiment.

Both models use exactly the same recordings and oracle tone boundaries. There is no learned segmenter. Waveform tensor hashes are checked globally for duplicates and stored in audio_manifest.jsonl. audio_data.pt preserves the actual tensors. Clean target labels are used to define representation codes, not as prediction inputs. Teacher-forced code accuracy and generated code accuracy are measured separately after recognition scoring.

CNN parameters are trained on the complete mixed-corruption training set. PPF begins from that CNN encoder, then trains for eight additional epochs. This is an explicit warm start with extra compute, not a matched-compute comparison. It is meant to test the repaired pipeline, not establish an efficiency advantage. Parameter counts include frozen tokenizer parameters. GPU memory is inapplicable on this CPU host.

Latency measures a single sequence from four event tensors through feature extraction and exhaustive ranking, including top-5 retrieval. It excludes file I/O and teacher diagnostics; three warmup calls precede timing. Scores are uncalibrated; ECE, Brier and NLL are evaluation metrics, not a calibration guarantee. Only one training seed is used in this repair experiment; no confidence or superiority claim should be based on it. There are no OOD corruptions, real recordings, complete-erasure controls, or compute-matched tuning in this run.

This is a newly implemented NCM-inspired audio adaptation. It retains hierarchical residual discrete codes and parallel level generation, while adding a clean-label encoder, EMA codebooks, prefix reconstruction loss and acoustic cross-attention. It is not an official NCM implementation or a verified reproduction of the original paper's exact mathematics.
