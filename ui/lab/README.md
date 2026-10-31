# Possibility Lab app

From the repository root:

```bash
.venv/bin/python -m ppf.lab_server --port 8765
```

Open http://127.0.0.1:8765. The frontend uses existing trained checkpoints; no training or checkpoint changes occur. First prediction loads models. The legacy UI at `ui/index.html` is preserved.

Enter exactly four digits, select a condition and digit position, choose one of three NCM checkpoints and click **Run workflow**. Listen to the actual generated input, inspect CNN probabilities, generated 1/2/4 codes and prefix predictions, then compare CNN exhaustive ranking with conventional and NCM gated rankings. Change training seed or audio seed to explore reproducibility. Export stores settings and predictions as JSON. Synthetic corruptions here are interactive controls, not identical to the study's corruption generator. Samples saturate at ±1 before prediction and WAV playback; this bounds strong noise/interference. Audio uses supplied event boundaries; live microphone recognition and file segmentation are not implemented.

## Application structure

- `ui/lab/`: plain HTML, CSS and JavaScript; no CDN, build process or external font required.
- `ppf/lab_server.py`: loopback-only Python HTTP API and static hosting.
- `GET /api/status`: service status; models load lazily on the first request.
- `GET /api/results`: actual three-seed aggregate results from the completed study.
- `POST /api/run`: JSON settings (`sequence`, `condition`, `position` 0–3, `strength` 0–1, `audio_seed`, checkpoint `seed`, `variant`). Returns audio, wave preview, per-digit probabilities, generated codes, gate decisions, top sequences and inspection timings.
- `predict(events, seed, variant)` receives audio only; the sequence text is used solely by the synthesizer. No oracle mode is exposed. All-position NCM computation enables inspection even where its scores are rejected.

The manifest and service worker provide a standalone app shell in supporting browsers. Inference still needs the local Python service and model files. Offline shell caching does **not** make the PyTorch model run inside the browser. Desktop packaging can bundle this folder, the Python service/runtime and checkpoints behind a webview. This deliverable is not a compiled desktop or mobile binary. Mobile/off-device use needs an explicit service/deployment design; the current server deliberately binds only to this computer.

Do not use individual examples as accuracy evidence. The benchmark panel reports measured research results, and the interface distinguishes full CNN ranking from gate-constrained lists. Code IDs are categorical. Prefix confidence uses the final decoder temperature, not a separately calibrated prefix model. Sequence product scores are not calibrated probabilities of correctness.

## Simplified keypad (current homepage)

The homepage now has a 0–9 keypad, a noise slider, sound toggle and an animated data-flow diagram. Pressing a digit automatically invokes real inference and displays the first event of the existing independent-event API; fixed padding events are not displayed or used as context. Previous digit presses do not condition the model. Rapid presses queue independently and Clear cancels pending visual updates. Actual first-event audio is played after the result arrives. Original seed317 is fixed to avoid model-selection controls.

The diagram replays audio → CNN scores → generated coarse/refine/fine codes → audio-only gate → final possibilities. Each hierarchy level animates all its code slots together. Animation timing is illustrative, not model scheduling. Multiple guesses appear if CNN or final top score is below80%, or its top-two gap below20 percentage points. This is an explicit display heuristic, not a trained rejection rule. The alternatives use final model digit scores and may remain sharply peaked even when CNN evidence is weak. The detailed previous interface is preserved at `/advanced/`.

The mechanism diagram now expands the real log-mel frontend, seven-convolution residual CNN, digit-head/adapter split, Transformer, codebook lookup, residual sum, decoder and audio-only gate. A sequence is capped at four entries; its completed result stays visible, then the fifth accepted key resets and starts the next sequence. Presses while a full sequence is still processing ask the user to wait rather than discarding pending predictions. Replay sound replays the last actual first-event waveform. Playback slices by duration at the decoded sample rate (300 ms), avoiding the former 2,400-resampled-sample truncation.

## Presentation view and complete-sequence alternatives

The main view uses plain-language stages: hear the sound, spot familiar patterns, build pattern pieces in parallel, and rank the answers. Pattern glyphs correspond to actual generated code IDs (available in their accessible labels/tooltips); they are not extra sampled hierarchies. Groups retain the actual 1→2→4 progression, with parallel prediction within each group.

At four completed digit predictions, any uncertainty flag produces the top three complete sequence combinations from the displayed per-digit possibilities. Uncertain positions contribute three alternatives; other positions keep their leading digit. Products of digit scores rank the combinations. These are constrained factorized candidate scores, not calibrated sequence confidence or independent hierarchy samples. The left panel explicitly labels its single leading sequence “Best of 3 guesses” and lists the three full alternatives below. The main result shows all three sequence guesses, even if the final digit itself was confident. Clear and the next-sequence reset remove prior alternatives.
