# InCTRL PQA-Only Ablation - Phase 3

Date: 2026-04-18

## Root Cause

The previous hybrid path mixed three behavior changes at once:

- AdaptCLIP visual adapter
- AdaptCLIP textual adapter
- prompt-query adapter (PQA)

This made the regression hard to attribute and kept non-original score branches alive in both the forward path and the training loss. The goal of this phase is to isolate one change only: keep the original InCTRL backbone and add only PQA.

## Proposed Changes

- Remove VA and TA from the runtime path.
- Keep the original InCTRL residual map as the final `patch_map`.
- Keep the fixed InCTRL / WinCLIP text prior instead of trainable text prototypes.
- Restore the original image residual direction: `prompt_proto - query`.
- Keep PQA as the only extra branch.
- Use a single training phase and a single optimizer.
- Align `engine_IC.py` with the same single-phase optimizer path.
- Stop reusing stale loss history from old result JSON files unless training is resumed from a checkpoint.
- Train with:

```text
L = focal(final_logit, y)
  + image_loss_weight * focal(image_logit, y)
  + pqa_loss_weight * focal(pqa_logit, y)
  + mask_loss_weight * pqa_mask_loss
```

## Files Changed

- `open_clip/inctrl_three_adapters.py`
- `open_clip/config/defaults.py`
- `train_local.py`
- `engine_IC.py`
- `tests/test_inctrl_three_adapters.py`
- `tests/test_train_local_loss.py`

## Validation Plan

1. Compile the modified model, training script, config, and focused tests.
2. Run focused PQA-only behavior tests.
3. Run a 1-epoch / 1-step smoke command with one category per target dataset.

## Expected Risks

- PQA still receives weak supervision on datasets without pixel masks.
- The final score now uses the original InCTRL branch as the anchor, so any PQA gains may appear first in branch metrics before they improve final AUROC.
- Existing three-adapter checkpoints are not directly comparable to this path.

## Validation

Compile check:

```text
C:\Users\dex\miniconda3\envs\dexter\python.exe -m py_compile train_local.py open_clip\inctrl_three_adapters.py open_clip\config\defaults.py tests\test_train_local_loss.py tests\test_inctrl_three_adapters.py
```

Focused tests:

```text
tests.test_train_local_loss
tests.test_inctrl_three_adapters
```

Observed result: all selected tests passed.

Smoke command:

```text
C:\Users\dex\miniconda3\envs\dexter\python.exe train_local.py --train-datasets mvtec --train-shot 4 --eval-shots 4 --epochs 1 --steps-per-epoch 1 --batch-size 2 --test-batch-size 1 --max-test-categories 1 --num-workers 0
```

Observed smoke result:

```text
Epoch 1 完成 | avg_loss=2.1954
AITEX 4-shot: AUROC=0.7619, AUPR=0.4151
ELPV 4-shot: AUROC=0.8795, AUPR=0.9420
VISA candle 4-shot: AUROC=0.9261, AUPR=0.9323
branches:
  AITEX -> base=0.7619, text=0.7385, pqa=0.3377, image=0.4474, holistic=0.5862
  ELPV -> base=0.8795, text=0.7332, pqa=0.4955, image=0.6067, holistic=0.5478
  candle -> base=0.9261, text=0.9661, pqa=0.6590, image=0.7446, holistic=0.4814
```

This smoke run is only a path check and must not be treated as a trained-model conclusion.
