# train_local Cross-Domain Shot Policy

Date: 2026-04-15

## Root Cause

The previous `train_local.py` workflow mixed training and evaluation shots by training one source model and then evaluating all `2/4/8` few-shot settings against the same checkpoint. That does not match the intended InCTRL evaluation protocol.

## Proposed Changes

- Train separate checkpoints for `2-shot`, `4-shot`, and `8-shot`.
- Keep the source-repo train/monitor design during training:
  - train on `MVTec` -> monitor on `MVTec`
  - train on `VisA` -> monitor on `VisA`
- Evaluate each checkpoint only on the matching few-shot setting after training.
- Route final cross-domain test datasets by source-domain policy:
  - `train on MVTec` -> final test on `VisA`, `AITEX`, `ELPV`
  - `train on VisA` -> final test on `MVTec`
- Resolve legacy absolute image paths at dataset load time so cloud runs can reuse the same JSON files after extraction.
- Skip datasets that do not have local few-shot prompt files.

## Files To Modify

- `train_local.py`

## Validation Plan

```powershell
C:\Users\dex\miniconda3\envs\dexter\python.exe -m py_compile train_local.py
```

```powershell
C:\Users\dex\miniconda3\envs\dexter\python.exe -c "import train_local; reg=train_local.prepare_dataset_registry(['mvtec','visa'], ['visa','aitex','elpv','mvtec']); cfg=train_local.prepare_single_experiment_config(reg, 'mvtec', ['visa','aitex','elpv'], 2, 10, 1e-3, 100, 48, 0.0); print(cfg['monitor_dataset']); print(len(cfg['monitor_normal_jsons']), len(cfg['monitor_outlier_jsons']))"
```

```powershell
C:\Users\dex\miniconda3\envs\dexter\python.exe -c "from datasets.IC_dataset_new import IC_dataset; ds=IC_dataset('.', ['data/AD_json/mvtec/bottle_normal.json'], ['data/AD_json/mvtec/bottle_outlier.json'], shot=2); print(ds._resolve_image_path(r'D:\Data\Downloads\InCTRL\data\mvtec\bottle\train\good\000.png')); print(ds._resolve_image_path('/workspace/InCTRL/data/mvtec/bottle/train/good/000.png'))"
```

## Expected Risks

- The default plan now schedules both `MVTec`-trained and `VisA`-trained runs, and each epoch also runs same-domain monitoring, so total compute is higher than the old single-source script.
- Resume support is only practical when the run plan is reduced to a single `source/shot` experiment.
