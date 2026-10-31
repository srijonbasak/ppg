# Smoke-test record

Date: 2026-10-31

## Passed in the project `.venv`

- Python source compilation: passed
- Browser JavaScript syntax check: passed
- Synthetic dataset generation: 20/20 WAV files created
- Dataset split: 16 train, 2 validation, 2 test
- Manifest labels and event sample bounds: passed
- Clean-tone Goertzel baseline: 10/10 digits correct
- Pytest suite: 4/4 passed
- Neural forward and backward pass: passed
- Target-free coarse-to-fine generation: passed
- Predicted level shapes: 8, 16, 31 positions (16 code logits each)
- Full smoke dataset: 13 labeled digit events

The generated smoke data was temporary and was not included in the package.

## Re-run

```bash
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
ppf-smoke
```
