# InCTRL Textual Adapter Ablation

Date: 2026-04-15

## Hypothesis

The lowest-risk way to extend restored baseline InCTRL is to keep the original residual scoring path unchanged and add a learnable text-context side branch that only adapts the prompt embeddings. This should improve image-level anomaly discrimination on domain-shifted industrial data without disturbing the original patch residual geometry.

## Files Changed

- `open_clip/config/defaults.py`
- `open_clip/model.py`
- `engine_IC.py`
- `train_local.py`

## Config Knobs

- `cfg.TEXTUAL_ADAPTER.ENABLE`: enable or disable the TA side branch.
- `cfg.TEXTUAL_ADAPTER.N_CTX`: number of learnable context tokens prepended to each handcrafted prompt.
- `cfg.TEXTUAL_ADAPTER.POS_INIT_TEXT`: initialization text for the normal prompt context.
- `cfg.TEXTUAL_ADAPTER.NEG_INIT_TEXT`: initialization text for the anomaly prompt context.
- `cfg.TEXTUAL_ADAPTER.FUSION_WEIGHT`: linear fusion weight applied to `ta_score`.
- `cfg.TEXTUAL_ADAPTER.LOSS_WEIGHT`: auxiliary TA classification loss weight during training.
- `cfg.TEXTUAL_ADAPTER.LOGIT_SCALE`: text-image similarity scale for the TA branch.

## Expected Upside

- Keep official InCTRL baseline behavior when `TEXTUAL_ADAPTER.ENABLE = False`.
- Allow a scoped prompt-learning ablation without introducing new visual adapters or changing the original patch min-distance route.
- Give training scripts a consistent way to include TA parameters and TA auxiliary loss.

## Risks And Confounders

- TA currently uses shared learnable context for all categories, not category-specific context parameters.
- The TA branch is fused only at image-score level in this first version; it does not change localization output.
- The new prompt cache is in-memory only and does not reduce first-batch prompt tokenization cost.

## Validation Plan

Syntax and import validation:

```powershell
python -m py_compile open_clip\model.py open_clip\config\defaults.py engine_IC.py train_local.py
```

Forward-path smoke check with TA enabled:

```powershell
python -c "import json, torch, open_clip; from open_clip.config.defaults import get_cfg; from open_clip.model import InCTRL, get_cast_dtype; cfg=get_cfg(); cfg.TEXTUAL_ADAPTER.ENABLE=True; f=open('open_clip/model_configs/ViT-B-16-plus-240.json', encoding='utf-8'); mc=json.load(f); f.close(); model=InCTRL(cfg, mc['embed_dim'], mc['vision_cfg'], mc['text_cfg'], quick_gelu=False, cast_dtype=get_cast_dtype('fp32')).cuda().eval(); tokenizer=open_clip.get_tokenizer('ViT-B-16-plus-240'); batch=[torch.randn(1,3,240,240) for _ in range(3)]; out=model(tokenizer, batch, ('candle',), None, return_aux=True); print(type(out), len(out), out[0].shape, out[1].shape, sorted(out[2].keys()), out[2]['ta_logits'].shape)"
```

## Validation Status

- `python -m py_compile open_clip\model.py open_clip\config\defaults.py engine_IC.py train_local.py`
  - Result: passed.
- Forward-path smoke in `C:\Users\dex\miniconda3\envs\dexter\python.exe`
  - Config: `TEXTUAL_ADAPTER.ENABLE=True`, `model.cuda().eval()`
  - Result:
    - `final_score.shape == torch.Size([1])`
    - `img_ref_score.shape == torch.Size([1])`
    - `aux.keys() == ['base_final_score', 'ta_logits', 'ta_score']`
    - `aux['ta_logits'].shape == torch.Size([1, 2])`
