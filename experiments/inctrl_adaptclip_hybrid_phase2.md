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
- PQA now follows the AdaptCLIP shape more closely: each layer aligns query patches to nearest prompt patches, forms `query + beta * abs(query - aligned_prompt)`, applies shared BN, uses a convolution/deconvolution local head with 2-channel normal/anomaly logits, and uses an MLP global head with 2-channel logits from mean plus top-k pooled features.
- Textual adapter default is `static_residual`: static InCTRL text prototypes plus two AdaptCLIP-style learnable binary prompts through a zero-init delta projector.
- Classification branches are logits-first. Evaluation still uses sigmoid scores.
- Default final scoring is now protected by raw InCTRL residuals:

```text
M_raw = nearest_neighbor_residual(raw_clip_query_tokens, raw_clip_prompt_tokens)
final_logit = alpha * max(M_raw)
```

The additive hybrid score is still produced as `base_logit` for ablations and auxiliary training, but it no longer overwrites the final evaluation score by default. This protects the original InCTRL geometry from being damaged by undertrained VA/PQA/holistic branches in early or low-step runs.

## Config Knobs

- `cfg.TEXTUAL_ADAPTER.ENABLE = True`
- `cfg.TEXTUAL_ADAPTER.MODE = "static_residual"`
- `cfg.TEXTUAL_ADAPTER.MAX_PROMPTS_PER_STATE = 32`
- `cfg.INCTRL_ADAPTER.FUSION_MODE = "paper_additive"`
- `cfg.INCTRL_ADAPTER.USE_PQA_IN_FINAL_MAP = False`
- `cfg.INCTRL_ADAPTER.USE_BRANCH_FUSION = True`
- `cfg.INCTRL_ADAPTER.LEARNABLE_LAYER_WEIGHTS = True`
- `cfg.INCTRL_ADAPTER.VISUAL_LOCAL_PER_LAYER = True`
- `cfg.INCTRL_ADAPTER.FINAL_SCORE_MODE = "raw_max_patch"`
- `cfg.INCTRL_ADAPTER.MASK_LOSS_WEIGHT = 1.0`
- CLI loss overrides:

```text
python train_local.py --image-loss-weight 1.0 --pqa-loss-weight 1.0 --mask-loss-weight 1.0 --text-reg-weight 0.01
```

## Training Objective

Visual phase:

```text
L_visual = focal(final_logit, y)
         + focal(base_logit, y)
         + image_loss_weight * focal(image_logit, y)
         + pqa_loss_weight * focal(pqa_logit, y)
         + mask_loss_weight * pqa_mask_loss
```

Text phase:

```text
L_text = focal(text_logit, y) + text_reg_weight * text_static_reg
```

PQA local mask supervision is enabled when masks are available. MVTec and VisA mask paths are derived from the query image path when JSON files do not include explicit mask paths. Normal samples and datasets without masks use an all-zero mask, so the same batch shape works for AITEX/ELPV.

PQA mask loss:

```text
pqa_mask_loss = multiclass_focal(pqa_local_logits, mask_0_1)
              + dice(anomaly_prob, mask)
              + dice(normal_prob, 1 - mask)
```

## Validation Notes

Run selected behavior tests with the local `dexter` environment because `pytest` is not installed:

```text
C:\Users\dex\miniconda3\envs\dexter\python.exe -m py_compile train_local.py datasets\IC_dataset_new.py datasets\loader.py open_clip\inctrl_three_adapters.py open_clip\config\defaults.py
```

```text
tests/test_dataset_path_resolution.py
tests/test_train_local_loss.py
tests/test_inctrl_three_adapters.py
```

Expected covered behaviors:

- residual equals `1 - max_cosine`
- default phase is `visual`
- PQA participates in visual phase even when not used as final local map
- output contains `final_logit`, `base_logit`, `image_logit`, `text_logit`, `pqa_logit`
- image residual direction is query minus prompt prototype
- TA zero-init residual starts from static text prototypes
- PQA global head receives gradients in visual phase
- PQA local head returns 2-channel full-resolution logits/scores
- visual phase adds PQA mask focal/dice loss when masks are present
- MVTec and VisA defect masks are derived from image paths
- raw InCTRL `max(M_raw)` is the default `final_logit`
- prompt and text caches match direct forward outputs
- visual/text phase losses use the intended logit branches

## Smoke Command

```text
python train_local.py --train-datasets mvtec --train-shot 4 --eval-shots 4 --epochs 1 --steps-per-epoch 1 --batch-size 2 --test-batch-size 1 --max-test-categories 1 --num-workers 0
```

This smoke run is only a path check and must not be treated as a trained model.

Actual local smoke result:

```text
Epoch 1 完成 | avg_loss=2.0941
loss=2.0941, final=0.1667, base=0.1667, image=0.1183, pqa=0.1958, mask=1.4466, text=0.0000
train=MVTEC train_shot=4 -> test=AITEX eval_shot=4 | AUROC=0.7632, AUPR=0.4161
train=MVTEC train_shot=4 -> test=ELPV eval_shot=4 | AUROC=0.8790, AUPR=0.9418
train=MVTEC train_shot=4 -> test=VISA eval_shot=4 | AUROC=0.9253, AUPR=0.9311
local RTX 3060 Laptop GPU speed: about 2.98s/batch for the single visual-phase batch
```

The smoke result is only a path and tensor-shape check. It should not be treated as a trained model, but it confirms that the new 4-tuple dataset output, PQA mask loss, raw residual final score, and cross-domain evaluation targets can run end-to-end.

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

Follow-up cloud smoke with branch diagnostics showed the first failure mode:

```text
candle final AUROC=0.7508
branches: base=0.6716, text=0.9660, pqa=0.6796, image=0.8774, holistic=0.3171, max_patch=0.9323

capsules final AUROC=0.7377
branches: base=0.7247, text=0.7937, pqa=0.4390, image=0.6973, holistic=0.5372, max_patch=0.7992

mean final AUROC=0.7442, AUPR=0.8050
```

Conclusion: the InCTRL nearest-neighbor residual signal is still useful, but the random/undertrained holistic and PQA branches are hurting early final scoring. The next patch protects the residual backbone by zero-initializing the final holistic layer and adding `alpha * max(base_residual_map)` as an explicit max-patch fallback branch in final fusion. The fallback branch is initially dominant; auxiliary branches still receive their own losses and can earn weight through training.

AdaptCLIP reference check:

- Original AdaptCLIP PQA global feature uses mean pooling plus top-k mean pooling over patch tokens, not mean plus a single max token.
- This repo now follows that more stable pooling form through `INCTRL_ADAPTER.PQA_GLOBAL_TOPK = 10`.
- The original AdaptCLIP visual adapter uses the last-stage patch feature for local text-alignment scoring. We keep multi-layer InCTRL residuals because they are the backbone of this hybrid experiment, but the fallback branch makes the final score preserve the strongest residual cue during early training.

Cross-domain evaluation mapping:

- Training on MVTec evaluates on AITEX, ELPV, and VisA.
- Training on VisA evaluates only on MVTec.
- This keeps the MVTec-trained model aligned with the broader industrial target-domain comparison while preserving the original VisA-to-MVTec reverse-domain check.
