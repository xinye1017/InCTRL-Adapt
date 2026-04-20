# AGENTS.md

## Project mission
This repository trains and evaluates an industrial anomaly detection system based on InCTRL (CVPR 2024).

Primary goals:
- reduce false positives on real industrial data
- preserve or improve anomaly localization quality
- keep experiments reproducible and easy to review
- prefer practical improvements over novelty for novelty's sake

## Entry points
- `main.py` — full training + testing pipeline (multi-GPU capable via `launch_job`)
- `test.py` — inference-only, accepts `--category`, `--few_shot_dir`, `--shot`, `--image_size`
- `train_local.py` — single-process local training (no `launch_job` overhead)
- `engine_IC.py` — core train/test loop logic; imports the active model from `open_clip`
- `engine_test.py` — test-only engine

## Config
- `open_clip/config/defaults.py` — **the single source of truth** for all hyperparameters (fvcore CfgNode)
- Config is patched via CLI arguments (e.g. `--shot`, `--image_size`, `--model`) or `opts` list, not via config files
- Hardcoded defaults include `CUDA_VISIBLE_DEVICES` assignment — **do not hard-code device IDs** in scripts you write; respect the env var

## Model
- `open_clip/model.py` — base `CLIP` and `InCTRL` class definitions
- `open_clip/inctrl_pqa.py` — `InCTRLPQA` (PQA-only architecture, current active model on `feat/inctrl-pqa-fused-restructure`)
- `open_clip/__init__.py` exports the active model class — it changed from `InCTRLWithAdapters` → `InCTRLPQA` on the feature branch; `git diff main` confirms the current import
- `open_clip/visual_adapter.py` — `VisualAdapter` (used in hybrid branch, disabled in PQA-only)

## Run commands
```bash
# Training (full pipeline)
python main.py --normal_json_path <train_normal.json> --outlier_json_path <train_outlier.json> \
  --val_normal_json_path <val_normal.json> --val_outlier_json_path <val_outlier.json> \
  --shot 2 --image_size 240

# Testing
python test.py --val_normal_json_path <test_normal.json> --val_outlier_json_path <test_outlier.json> \
  --category <name> --few_shot_dir <path> --shot 2 --image_size 240

# Smoke test / compile check
python -m py_compile train_local.py open_clip/inctrl_pqa.py open_clip/config/defaults.py
```

## Dependencies
- torch/torchvision are **NOT** in requirements.txt — assumed pre-installed in the CUDA environment
- Key libs: `fvcore`, `yacs`, `timm`, `transformers`, `open_clip`, `scikit-learn`, `scipy`, `py7zr`
- `open_clip` is a local package (editable install or `PYTHONPATH` required)

## Testing
```bash
pytest tests/ -v
# Focused tests relevant to current architecture:
pytest tests/test_inctrl_pqa.py tests/test_inctrl_pqa_losses.py tests/test_train_local_loss.py -v
```
Tests live in `tests/` and mirror the module structure under `open_clip/`.

## Data conventions
- Raw data: MVTec AD format (train/good + test/good + test/defect_N/)
- JSON manifests: `datasets/AD_json/` — one JSON per split per category
- **Never modify raw data files in place**
- Required Google Drive resources (pre-trained models, few-shot samples) are documented in README

## Output conventions
- Reports → `reports/`
- Experiment notes → `experiments/*.md`
- Analysis scripts → `tools/`
- Checkpoints → `checkpoints/` (local runs) or user-specified `OUTPUT_DIR`
- Result artifacts → `results/`

## Baseline comparison (smoke test rule)
Every smoke test result must be compared against the canonical baseline before claiming regressions or improvements.

**Baseline numbers (`reports/original_inctrl_baseline.md`):**

| Setting | ELPV | AITEX | VisA | MVTec AD |
| --- | ---: | ---: | ---: | ---: |
| 0-shot | 0.733 | 0.733 | 0.781 | 0.912 |
| 2-shot | 0.839 | 0.790 | 0.858 | 0.940 |
| 4-shot | 0.846 | 0.790 | 0.877 | 0.945 |
| 8-shot | 0.872 | 0.806 | 0.887 | 0.953 |

**Per-class VisA baseline (`reports/original_inctrl_visa_4shot_per_class.md`):** mean AUROC=0.890, mean AUPR=0.893.

**Smoke test comparison:** After any `train_local.py` smoke run, report:
1. Absolute AUROC/AUPR for the tested category
2. Delta versus the matching baseline row

## Active branch note
`feat/inctrl-pqa-fused-restructure` replaced the hybrid three-adapter path (`InCTRLWithAdapters`) with a clean PQA-only path (`InCTRLPQA`). The `open_clip/__init__.py` export was updated accordingly. Existing experiment notes in `experiments/` describe the rationale for this transition.

## Before committing
- Run `python -m py_compile <changed_files>` for compile check
- Run relevant pytest tests
- Record validation commands and outputs in the experiment note
