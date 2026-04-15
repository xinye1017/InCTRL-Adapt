# Visual Adapter Global/Local Ablation

Date: 2026-04-15

## Hypothesis

The visual adapter can be treated as a feature-space optimizer for InCTRL's original anomaly scoring mechanism, not as a replacement for the scoring mechanism. The safest way to study local patch feature perturbation is to separate global and local adapter routes:

- `global_only`: adapt global query/reference tokens; keep patch features in the frozen backbone space.
- `local_only`: keep global query/reference tokens in the frozen backbone space; adapt patch features before reference matching.
- `global_local`: adapt both branches, matching the current full VA idea.

## Changes

- Default VA residual adapters now use identity-style initialization through `VISUAL_ADAPTER.ZERO_INIT = True`.
- `VISUAL_ADAPTER.MODE` controls VA routing: `global_only`, `local_only`, or `global_local`.
- Disabled VA branches are pass-through and frozen, so inactive branch parameters do not enter optimization.
- `train_va.py` now trains the full MVTec ablation matrix by default: 3 VA modes x 2/4/8 shots.
- `train_va.py` strongly validates the base checkpoint load: frozen CLIP/InCTRL weights must load; only newly trained heads/adapters may be missing.
- Checkpoints are saved under `checkpoints/InCTRL_trained_on_MVTec_VA_ablation/<mode>/<shot>/checkpoint` with per-run `metadata.json`.

## Config Knobs

- `cfg.VISUAL_ADAPTER.ENABLE = True` in VA training.
- `cfg.VISUAL_ADAPTER.ZERO_INIT = True` for identity initialization.
- `cfg.VISUAL_ADAPTER.MODE in {"global_only", "local_only", "global_local"}`.

## Expected Upside

- `global_only` should test classification/reference-score gains without disturbing local patch geometry.
- `local_only` should isolate whether patch adaptation improves or destabilizes localization/reference matching.
- `global_local` should show whether combining both branches gives additive benefit or over-transforms features.

## Risks And Confounders

- `local_only` uses raw global query/reference token differences rather than the legacy non-VA `Adapter`, so it isolates VA local effects but is not identical to the old InCTRL global branch.
- Existing VA checkpoints trained with non-identity initialization or a different norm layer may not be directly comparable.
- Full default training runs nine models and may be long; use CLI filters for smoke runs.

## Validation

Minimal syntax validation:

```powershell
python -m py_compile open_clip\visual_adapter.py open_clip\model.py open_clip\config\defaults.py train_va.py tests\test_visual_adapter.py
```

Focused smoke command:

```powershell
python train_va.py --modes global_only --shots 2 --epochs 1 --steps-per-epoch 1
```

Full training command:

```powershell
python train_va.py
```
