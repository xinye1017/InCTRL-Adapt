# InCTRL Context Residual PQA Lite

## Hypothesis

以 InCTRL 少样本上下文残差为主干，保留图像级残差和多层 patch 级残差；增加轻量视觉适配器、对象不可知文本语义分支、提示-查询对齐分割头后，应提升像素级定位能力，并减少由纹理、光照、材料差异导致的 false positives。

## Architecture

- Backbone: frozen CLIP visual/text towers.
- InCTRL evidence: image residual score and multi-layer nearest-neighbor patch residual map.
- Added modules: `VisualAdapter`, `ObjectAgnosticTextBranch`, `PromptQuerySegHead`.
- Final image score: fixed-weight fusion of image residual, max patch residual, PQA global logit, and object-agnostic text logit.
- Final map: fixed-weight fusion of patch residual map, PQA segmentation map, and object-agnostic text map.

## Validation Commands

```bash
python3 -m py_compile train_local.py open_clip/visual_adapter.py open_clip/object_agnostic_text.py open_clip/prompt_query_head.py open_clip/inctrl_pqa.py open_clip/inctrl_pqa_losses.py open_clip/config/defaults.py engine_IC.py engine_test.py
python3 -m pytest tests/test_visual_adapter.py tests/test_object_agnostic_text.py tests/test_prompt_query_head.py tests/test_inctrl_pqa.py tests/test_inctrl_pqa_losses.py tests/test_engine_ic_training.py -v
```

## Smoke Result Template

| Dataset | Category | Shot | AUROC | AUPR | Baseline AUROC | Delta AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| VisA | candle | 2 |  |  | 0.858 |  |

Baseline source: `reports/original_inctrl_baseline.md`.
