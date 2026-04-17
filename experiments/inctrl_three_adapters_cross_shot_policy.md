# InCTRL Three Adapters Cross-Shot Policy

Date: 2026-04-17

## Hypothesis

Training separate 2/4/8-shot checkpoints is unnecessarily expensive for the current comparison. A single 4-shot source-domain model should provide a practical midpoint for adapter learning, while cross-shot evaluation at 2/4/8 shots measures prompt sensitivity without multiplying training cost.

## Proposed Changes

- Train one `InCTRLWithAdapters` checkpoint per source dataset.
- Default source datasets are `MVTec` and `VisA`.
- Default training shot is `4`.
- Default evaluation shots are `2`, `4`, and `8`.
- Keep VA/TA alternating learning:
  - even epochs train visual-side parameters
  - odd epochs train text-side parameters
- Avoid modifying raw dataset JSON files in place; path compatibility is handled in the dataset loader.

## Files Modified

- `train_local.py`
- `experiments/inctrl_three_adapters_cross_shot_policy.md`

## Config Knobs

- `--train-datasets`, default `mvtec visa`
- `--train-shot`, default `4`
- `--eval-shots`, default `2 4 8`
- `--batch-size`, default `48`
- `--test-batch-size`, default `1`
- `--steps-per-epoch`, default `100`
- `--epochs`, default `10`
- `--num-workers`, default `0`
- `--max-test-categories`, default unset, only for fast validation

## Validation Plan

```powershell
C:\Users\dex\miniconda3\envs\dexter\python.exe -m py_compile train_local.py open_clip\inctrl_three_adapters.py engine_IC.py
```

```powershell
C:\Users\dex\miniconda3\envs\dexter\python.exe -c "import tests.test_inctrl_three_adapters as t; [getattr(t, n)() for n in dir(t) if n.startswith('test_')]; print('manual tests passed')"
```

```powershell
C:\Users\dex\miniconda3\envs\dexter\python.exe train_local.py --train-datasets mvtec --train-shot 4 --eval-shots 4 --batch-size 1 --test-batch-size 4 --steps-per-epoch 1 --epochs 1 --max-test-categories 1
```

## Validation Results

- `py_compile` passed for `train_local.py`, `open_clip/inctrl_three_adapters.py`, and `engine_IC.py`.
- Manual execution of `tests/test_inctrl_three_adapters.py` test functions passed in the `dexter` conda environment.
- Dataset registry check passed:
  - MVTec train JSON: 15 normal, 15 outlier.
  - VisA train JSON: 12 normal, 12 outlier.
  - VisA test categories: 12.
  - MVTec test categories: 15.
  - MVTec and VisA 4-shot few-shot `.pt` files resolve correctly.
- Fast end-to-end smoke passed on RTX 3060 Laptop GPU:
  - command: `python train_local.py --train-datasets mvtec --train-shot 4 --eval-shots 4 --batch-size 1 --test-batch-size 4 --steps-per-epoch 1 --epochs 1 --max-test-categories 1`
  - checkpoint: `checkpoints/trained_on_mvtec/shot_4/checkpoint`
  - result: VisA `candle`, eval 4-shot, AUROC `0.8037`, AUPR `0.7977`.
  - note: one-category evaluation took about 3.4 minutes, so full cross-shot evaluation is expected to take hours without prompt feature caching.

## Evaluation Speed Optimization

Root cause:

- The original three-adapter evaluation path re-encoded the same few-shot prompt images for every query batch.
- It also rebuilt the same category text prototypes for every query batch.

Implemented optimization:

- Cache prompt-side visual features once per `(dataset, category, eval_shot)` via `model.build_prompt_feature_cache()`.
- Cache category text prototypes once per category via `model.build_text_prototype_cache()`.
- Reuse both caches in `model.forward()` through `prompt_feature_cache` and `text_prototype_cache`.

Validation:

- Direct prompt path and cached prompt path produce matching `final_score` and `patch_map` in unit tests.
- Direct text path and cached text path produce matching `final_score` and `text_score` in unit tests.
- VisA `candle`, eval 4-shot, `test_batch_size=4`, same checkpoint:
  - uncached prompt/text path: about `203.55s`, AUROC `0.8038`, AUPR `0.7977`.
  - prompt-cache only: about `100.60s`, AUROC `0.8038`, AUPR `0.7977`.
  - prompt-cache + text-cache: about `16.58s`, AUROC `0.8038`, AUPR `0.7977`.

Expected impact:

- Full cross-shot evaluation should now be practical enough for routine comparisons against `reports/original_inctrl_baseline.md`.

## Risks And Confounders

- Cross-shot evaluation changes only prompt reference count, not model weights, so it measures few-shot prompt robustness rather than independent few-shot training capacity.
- 4-shot may be a good compute compromise but can hide cases where 2-shot training regularizes better.
- Alternating VA/TA learning is stable and interpretable, but the epoch-level schedule may be coarse for very short smoke runs.
