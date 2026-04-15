# Baseline Reproduction Fix

Date: 2026-04-15

## Hypothesis

Official InCTRL checkpoints should be evaluated with the paper baseline forward path. Local Visual Adapter branches and altered text prompt normalization can change anomaly scores even when the checkpoint is official.

## Changes

- Set `VISUAL_ADAPTER.ENABLE` default to `False` so the official baseline path is the default model construction behavior.
- Restored the official text prompt embedding behavior: normalize each prompt embedding, average prompts, then normalize the averaged normal/anomaly text embedding again.
- Hardened `test_baseline.py` so baseline evaluation refuses to run if a Visual Adapter is instantiated and raises an error on checkpoint/model key mismatches.

## Config Knobs

- `cfg.VISUAL_ADAPTER.ENABLE = False` for paper-consistent baseline evaluation.
- VA experiments must explicitly set `cfg.VISUAL_ADAPTER.ENABLE = True`.

## Expected Upside

`test_baseline.py` can batch-evaluate the official 2/4/8-shot baseline checkpoints without silently using local VA modules or altered text score scaling.

## Risks And Confounders

- Patch scoring is still vectorized in local `model.py`; it should be mathematically equivalent to the official nested-loop implementation but may not be bitwise identical.
- Metric aggregation in `test_baseline.py` reports macro and micro results; paper table reproduction must use the same aggregation convention as the paper.

## Validation

Executed minimal validation:

```powershell
python -m py_compile open_clip\model.py open_clip\config\defaults.py test_baseline.py
```

Result: passed.

Full validation:

```powershell
python test_baseline.py
```

Status: not run in this change because it evaluates 27 model-shot/eval-shot/dataset combinations.
