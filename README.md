# DTMF Parallel Possibility Framework

This is a **working NCM-inspired research prototype**, not an official NCM
implementation and not a claim of architectural equivalence. It borrows the
ideas most useful for this experiment—hierarchical discrete targets,
coarse-to-fine conditioning, parallel prediction within a scale, and multiple
ranked outcomes—and applies them to DTMF digits 0–9.

## What is real here

The model is more than a renamed CNN/Transformer:

1. A torch-only log-Mel front end feeds a dilated residual CNN encoder.
2. `MultiscaleResidualVQ` creates discrete targets at 1/4, 1/2, and full time
   resolution. The coarse codebook explains the first approximation; every
   finer codebook quantizes the **remaining residual**, and the upsampled sum
   reconstructs the encoder latent.
3. `ParallelLevelPredictor` predicts one code distribution per time position.
   It receives acoustic memory and already-generated coarser levels. There is
   no causal mask within a level, so all codes at that level are produced in a
   single Transformer pass.
4. In generation, predicted code embeddings—not tokenizer targets—are carried
   from coarse to fine and passed to the latent digit decoder. The decoder
   exposes calibrated-style top-K possibilities rather than only an argmax.

Teacher targets from the residual tokenizer supervise each level during
training. The digit decoder is also supervised directly, while VQ commitment,
codebook, and reconstruction losses keep the discrete hierarchy meaningful.

## Layout

```text
ppf/data.py       synthetic WAVs, corruption, manifest, PCM I/O
ppf/features.py   log-Mel transform without torchaudio
ppf/model.py      CNN, residual VQ hierarchy, parallel Transformer, decoder
ppf/engine.py     event dataset, training/evaluation, CNN and Goertzel baselines
ppf/cli.py        generate/train/evaluate/smoke commands
ppf/server.py     dependency-free local UI and checkpoint inference API
ui/               responsive keypad, synthesis, mic capture, dataset download
tests/            data, baseline, hierarchy, backward and generation tests
```

## Setup

Use Python 3.10–3.13 (PyTorch wheels may not yet exist for newer Python):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Quick start

Generate reproducible train/validation/test WAVs and JSONL ground truth:

```bash
ppf-generate --output data/dtmf --count 2000 --seed 7
```

Every sequence uses variable tone/gap lengths. Training samples additionally
include frequency jitter, row/column twist, Gaussian noise at randomized SNR,
50/60 Hz hum, soft clipping, and occasional clicks/dropouts. Each event records
its digit, exact sample bounds, frequencies, tone length, and gap length.

Train the framework and a direct CNN baseline:

```bash
ppf-train --manifest data/dtmf/manifest.jsonl --output checkpoints/ppf.pt --epochs 10
ppf-train --manifest data/dtmf/manifest.jsonl --output checkpoints/cnn.pt --epochs 10 --baseline
```

Evaluate the PPF alongside the no-training Goertzel baseline:

```bash
ppf-evaluate --manifest data/dtmf/manifest.jsonl --checkpoint checkpoints/ppf.pt \
  --cnn-checkpoint checkpoints/cnn.pt --top-k 3
```

The evaluator reports top-1, top-K and NLL for the neural model, and top-1 for
the classical two-frequency Goertzel detector. The direct CNN is supplied as an
ablation baseline in `CNNBaseline`.

Run the keypad and recording UI:

```bash
ppf-serve --checkpoint checkpoints/ppf.pt --port 8000
# open http://127.0.0.1:8000
```

The browser can compose/replay sequences, randomize durations and noise,
download synthetic WAV + JSON labels, capture labeled microphone samples, and
show the trained model's top five possibilities. Microphone access works on
localhost in current browsers.

To explore synthesis/recording without installing PyTorch, serve only the
static UI with `python -m http.server 8000 --directory ui`; prediction requires
`ppf-serve` and a trained PPF checkpoint.

## Verification

```bash
pytest -q
ppf-smoke
```

The smoke command builds a tiny dataset, runs a complete forward/backward pass,
checks the generated coarse-to-fine code shapes, and evaluates Goertzel. A smoke
pass proves wiring and tensor compatibility; it does not prove model quality.

## Scope and honest limitations

- The first prototype classifies isolated labeled tone events. Sequence WAVs
  retain event boundaries so a later detector/segmenter can be added cleanly.
- Synthetic corruption approximates telephony conditions; real device and room
  recordings from the UI are still important for domain-shift testing.
- Top-K scores are ordinary softmax probabilities. For deployment, fit
  temperature scaling on validation data and report ECE alongside accuracy.
- “NCM-inspired” describes the design influence only. This repository is
  independent experimental code and is not endorsed by an NCM author/team.


## Interactive research workflow app

Run `.venv/bin/python -m ppf.lab_server --port 8765`, then open http://127.0.0.1:8765.
This app uses the latest preserved CNN and NCM research checkpoints, with controllable synthetic audio, waveform/playback, generated hierarchy inspection, fixed-gate rankings, and measured study results. See [app setup and packaging notes](ui/lab/README.md). The original synthesis/recording UI remains available through the legacy server.
