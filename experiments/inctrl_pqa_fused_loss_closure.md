# InCTRL PQA Fused Loss Closure

Date: 2026-04-19

## Hypothesis

The fused PQA model should train the same decision surface it uses at inference. `final_logit` is the primary fused decision, `pqa_logit` keeps the prompt-query global branch directly supervised, and `pqa_local_logits` receives pixel supervision whenever masks are available.

## Proposed Changes

- Use `open_clip.inctrl_pqa_fused.InCTRLPQA` in both local and `engine_IC.py` training entrypoints.
- Move fused objective code into `open_clip/inctrl_pqa_losses.py` so both entrypoints use the same loss semantics.
- Default image residual auxiliary supervision to off. The image residual branch still participates in `final_logit` through the decision fusion head.
- Keep PQA mask supervision as focal loss over 2-channel local logits plus anomaly/normal Dice loss.

## Files Changed

- `open_clip/inctrl_pqa_fused.py`
- `open_clip/inctrl_pqa_losses.py`
- `open_clip/config/defaults.py`
- `train_local.py`
- `engine_IC.py`
- `tests/test_train_local_loss.py`
- `tests/test_inctrl_pqa_fused.py`

## Config Knobs

- `cfg.PQA.GLOBAL_LOSS_WEIGHT = 1.0`
- `cfg.PQA.MASK_LOSS_WEIGHT = 1.0`
- `cfg.PQA.IMAGE_LOSS_WEIGHT = 0.0`
- `train_local.py --pqa-loss-weight 1.0`
- `train_local.py --mask-loss-weight 1.0`
- `train_local.py --image-loss-weight 0.0`

## Training Objective

```text
L = focal(final_logit, y)
  + pqa_loss_weight * focal(pqa_logit, y)
  + mask_loss_weight * L_pqa_seg
  + image_loss_weight * focal(image_logit, y)
```

Default `image_loss_weight` is `0.0`; it remains only as an ablation knob.

```text
L_pqa_seg = multiclass_focal(pqa_local_logits, mask_0_1)
          + dice(anomaly_prob, mask)
          + dice(normal_prob, 1 - mask)
```

If masks are unavailable or `mask_loss_weight == 0`, `L_pqa_seg` is skipped.

## Expected Upside

- The decision fusion head is trained directly on the deployed `final_logit`.
- PQA global and local branches remain constrained instead of relying only on indirect fusion gradients.
- The ablation is easier to compare because image-branch direct supervision is disabled by default.

## Risks And Confounders

- If an anomalous sample has no real mask, the current dataset path returns an all-zero mask; this may suppress local anomaly activations for that sample.
- Text prior is supervised only through `final_logit`, not with a separate text loss.
- Existing checkpoints from `open_clip.inctrl_pqa` and `open_clip.inctrl_pqa_fused` should be treated as separate ablations because state dicts differ.
- Short smoke runs validate path correctness only and should not be interpreted as trained model quality.

## Validation Plan

```text
C:\Users\dex\miniconda3\envs\dexter\python.exe -m py_compile train_local.py engine_IC.py open_clip\inctrl_pqa_fused.py open_clip\inctrl_pqa_losses.py open_clip\config\defaults.py
```

```text
C:\Users\dex\miniconda3\envs\dexter\python.exe -m pytest tests\test_train_local_loss.py tests\test_inctrl_pqa_fused.py -q
```

```text
C:\Users\dex\miniconda3\envs\dexter\python.exe train_local.py --train-datasets mvtec --train-shot 4 --eval-shots 4 --epochs 1 --steps-per-epoch 1 --batch-size 2 --test-batch-size 1 --max-test-categories 1 --num-workers 0 --image-loss-weight 0.0 --pqa-loss-weight 1.0 --mask-loss-weight 1.0
```

## Validation Results

Compile check passed:

```text
C:\Users\dex\miniconda3\envs\dexter\python.exe -m py_compile train_local.py engine_IC.py open_clip\inctrl_pqa_fused.py open_clip\inctrl_pqa_losses.py open_clip\config\defaults.py
```

Focused tests passed:

```text
C:\Users\dex\miniconda3\envs\dexter\python.exe -m pytest tests\test_train_local_loss.py tests\test_inctrl_pqa_fused.py tests\test_engine_ic_training.py tests\test_inctrl_pqa.py -q
21 passed in 7.87s
```

Train-local smoke passed:

```text
C:\Users\dex\miniconda3\envs\dexter\python.exe train_local.py --train-datasets mvtec --train-shot 4 --eval-shots 4 --epochs 1 --steps-per-epoch 1 --batch-size 2 --test-batch-size 1 --max-test-categories 1 --num-workers 0 --image-loss-weight 0.0 --pqa-loss-weight 1.0 --mask-loss-weight 1.0
Epoch 1 完成 | avg_loss=1.9237
loss=1.9237, final=0.1733, image=0.0000, pqa=0.1845, mask=1.5659
AITEX 4-shot AUROC=0.5219, AUPR=0.3806
ELPV 4-shot AUROC=0.5249, AUPR=0.7092
VISA candle 4-shot AUROC=0.6771, AUPR=0.7456
```

`engine_IC.py` path check through `main.py` passed:

```text
C:\Users\dex\miniconda3\envs\dexter\python.exe main.py --normal_json_path data\AD_json\mvtec\bottle_normal.json --outlier_json_path data\AD_json\mvtec\bottle_outlier.json --val_normal_json_path data\AD_json\mvtec\bottle_val_normal.json --val_outlier_json_path data\AD_json\mvtec\bottle_val_outlier.json --shot 2 --steps_per_epoch 1 TRAIN.BATCH_SIZE 2 TEST.BATCH_SIZE 1 SOLVER.MAX_EPOCH 1 DATA_LOADER.NUM_WORKERS 0 PQA.IMAGE_LOSS_WEIGHT 0.0 PQA.GLOBAL_LOSS_WEIGHT 1.0 PQA.MASK_LOSS_WEIGHT 1.0 TEST.ENABLE False
train_loss: total=1.8100, final=0.1733, pqa=0.1745, mask=1.4622, image=0.0000
Predict train set: AUC-ROC=0.0000, AUC-PR=0.5000
Predict test set: AUC-ROC=0.7159, AUC-PR=0.9176
```
