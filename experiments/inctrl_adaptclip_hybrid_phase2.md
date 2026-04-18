# InCTRL + AdaptCLIP Hybrid Architecture - Phase 2

Date: 2026-04-18

## Hypothesis

The model should keep InCTRL's multi-layer nearest-neighbor patch residual as the main anomaly backbone, while AdaptCLIP-style VA, TA, and PQA improve representation and few-shot comparison quality. VA changes visual features, TA learns a residual over static InCTRL text prototypes, and PQA contributes an image-level comparison branch instead of replacing the base residual map.

## Implemented Design

- Patch residual now follows InCTRL scale: `residual = 1 - max_cosine`.
- Image residual follows the paper direction: `adapted_query_global - prompt_global_proto`.
- Default phase is `visual`; training alternates `visual` and `text` by epoch.
- Visual adapter keeps residual global adaptation and now uses per-layer local adapters by default.
- Base residual and PQA share parent-level learnable layer weights.
- PQA always runs when enabled and is included in visual-phase parameters even when its local map is not used as the final `patch_map`.
- PQA now returns local patch logits/scores and a global image logit/score from avg/max pooled context-residual features.
- Textual adapter default is `static_residual`: static InCTRL text prototypes plus two AdaptCLIP-style learnable binary prompts through a zero-init delta projector.
- Classification branches are logits-first. Evaluation still uses sigmoid scores.
- Default fusion is additive and InCTRL-preserving:

```text
M_plus = M + lambda_g * image_score + lambda_t * text_score + lambda_p * pqa_score
base_logit = holistic_head(M_plus) + alpha * max(M)
final_logit = softmax(branch_logits) dot [base_logit, text_logit, pqa_logit]
```

## Config Knobs

- `cfg.TEXTUAL_ADAPTER.ENABLE = True`
- `cfg.TEXTUAL_ADAPTER.MODE = "static_residual"`
- `cfg.TEXTUAL_ADAPTER.MAX_PROMPTS_PER_STATE = 32`
- `cfg.INCTRL_ADAPTER.FUSION_MODE = "paper_additive"`
- `cfg.INCTRL_ADAPTER.USE_PQA_IN_FINAL_MAP = False`
- `cfg.INCTRL_ADAPTER.USE_BRANCH_FUSION = True`
- `cfg.INCTRL_ADAPTER.LEARNABLE_LAYER_WEIGHTS = True`
- `cfg.INCTRL_ADAPTER.VISUAL_LOCAL_PER_LAYER = True`
- CLI loss overrides:

```text
python train_local.py --image-loss-weight 1.0 --pqa-loss-weight 1.0 --text-reg-weight 0.01
```

## Training Objective

Visual phase:

```text
L_visual = focal(final_logit, y)
         + focal(base_logit, y)
         + image_loss_weight * focal(image_logit, y)
         + pqa_loss_weight * focal(pqa_logit, y)
```

Text phase:

```text
L_text = focal(text_logit, y) + text_reg_weight * text_static_reg
```

The current dataset loader provides image labels but not masks, so PQA local segmentation loss remains disabled.

## Validation Notes

Run selected behavior tests with the local `dexter` environment because `pytest` is not installed:

```text
C:\Users\dex\miniconda3\envs\dexter\python.exe -m py_compile train_local.py open_clip\inctrl_three_adapters.py open_clip\config\defaults.py
```

```text
tests/test_inctrl_three_adapters.py
tests/test_train_local_loss.py
```

Expected covered behaviors:

- residual equals `1 - max_cosine`
- default phase is `visual`
- PQA participates in visual phase even when not used as final local map
- output contains `final_logit`, `base_logit`, `image_logit`, `text_logit`, `pqa_logit`
- image residual direction is query minus prompt prototype
- TA zero-init residual starts from static text prototypes
- PQA global head receives gradients in visual phase
- prompt and text caches match direct forward outputs
- visual/text phase losses use the intended logit branches

## Smoke Command

```text
python train_local.py --train-datasets mvtec --train-shot 4 --eval-shots 4 --epochs 1 --steps-per-epoch 1 --batch-size 2 --test-batch-size 1 --max-test-categories 1 --num-workers 0
```

This smoke run is only a path check and must not be treated as a trained model.

Actual local smoke result:

```text
Epoch 1 完成 | avg_loss=0.6980
train=MVTEC train_shot=4 -> test=VISA eval_shot=4 | AUROC=0.8619, AUPR=0.8795
loss=0.6980, final=0.1668, base=0.1758, image=0.1872, pqa=0.1682, text=0.0000
local RTX 3060 Laptop GPU speed: about 2.37s/batch for the single visual-phase batch
```

## Cloud Smoke Failure Follow-up

Cloud run reported a severe drop on a tiny VisA two-category smoke:

```text
train=MVTEC train_shot=4 -> test=VISA eval_shot=4
epochs=1, steps_per_epoch=1, batch_size=48, num_workers=4
candle AUROC=0.3705, AUPR=0.4858
capsules AUROC=0.5087, AUPR=0.6154
mean AUROC=0.4396, AUPR=0.5506
```

This result is not acceptable and should stop full 10-epoch training until the failing branch is identified. Evaluation now records branch-level metrics for `final`, `base`, `text`, `pqa`, `image`, `holistic`, and `max_patch`.

Decision rules for the next smoke:

- If `max_patch` is strong but `base`/`final` are weak, the residual backbone still ranks anomalies but holistic/additive fusion is damaging it.
- If `base` is strong but `final` is weak, branch fusion is the main failure point and should be disabled or made more base-biased.
- If `text` or `pqa` is weak while `base` is usable, those auxiliary branches should be down-weighted before longer training.
- If all branches, including `max_patch`, are weak, the feature adaptation or residual computation itself is broken and the architecture should be rolled back to the last faithful InCTRL residual baseline.
