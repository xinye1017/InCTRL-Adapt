# InCTRL Train Speed Optimization

Date: 2026-04-16

## Hypothesis

The main training slowdown comes from two hotspots:
- patch-level anomaly scoring in `open_clip/model.py` uses nested Python loops over samples, feature levels, and patches
- `datasets/IC_dataset_new.py` repeatedly opens the same support images for few-shot prompts

## Files Changed

- `open_clip/model.py`
- `datasets/IC_dataset_new.py`

## Config Knobs

- `INCTRL_SUPPORT_IMAGE_CACHE_SIZE`
  - default: `512`
  - meaning: maximum number of decoded support images cached per dataset worker process
- `TEXTUAL_ADAPTER.MAX_PROMPTS_PER_STATE`
  - default: `8`
  - meaning: maximum normal/anomaly prompt templates used by the trainable TA branch per object type; the frozen baseline text branch still uses the full handcrafted prompt set

## Expected Upside

- lower Python overhead in the patch reference map computation
- fewer repeated `Image.open(...)` calls for support images
- smoother GPU utilization when training with few-shot support sets
- much lower TA memory usage by avoiding hundreds of trainable text-transformer prompt activations per object type

## Risks And Confounders

- support image cache increases RAM usage on the host, especially with many dataloader workers
- if transforms become stochastic in the future, caching decoded images remains safe, but caching transformed tensors would not be
- vectorized patch scoring should be numerically equivalent, but needs shape-level validation
- compact TA prompts are an intentional memory/compute tradeoff; compare against full-prompt TA only if hardware allows

## Validation Command

```powershell
C:\Users\dex\miniconda3\envs\dexter\python.exe -m py_compile open_clip\model.py datasets\IC_dataset_new.py
```

```powershell
C:\Users\dex\miniconda3\envs\dexter\python.exe -c "import torch; from open_clip.model import compute_patch_reference_scores; q=torch.randn(2,3,5,7); n=torch.randn(2,3,11,7); ref_map, score=compute_patch_reference_scores(q,n); print(ref_map.shape, score.shape)"
```

```powershell
C:\Users\dex\miniconda3\envs\dexter\python.exe -c "from datasets.IC_dataset_new import IC_dataset; ds=IC_dataset('.', ['data/AD_json/mvtec/bottle_normal.json'], ['data/AD_json/mvtec/bottle_outlier.json'], transform=lambda x: x, shot=2); sample=ds[0]; print(sample[1], sample[2], len(sample[0]), len(ds._support_image_cache))"
```
