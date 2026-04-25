# InCTRLv2

This implementation adds an isolated `models/inctrlv2/` package on top of the original InCTRL code. It keeps image-level residual learning and multi-layer patch residual learning, then adds DASL and OASL with independent patch adapters.

## Data Layout

Use MVTec-style directories for both MVTec AD and VisA:

```text
DATA_ROOT/
  mvtec/
    bottle/
      train/good/*.png
      test/good/*.png
      test/broken_large/*.png
      ground_truth/broken_large/*_mask.png
  visa/
    candle/
      train/good/*.png
      test/good/*.png
      test/defect/*.png
      ground_truth/defect/*_mask.png
```

The dataset also accepts legacy JSON manifests through `--train_normal_json`, `--train_outlier_json`, `--normal_json`, and `--outlier_json`. If `mask_path` is missing, the loader tries to infer it from `ground_truth/<defect>/<image_stem>_mask.*`. Normal masks are always returned as all-zero tensors.

## Training

MVTec to VisA:

```bash
python3 train_inctrlv2.py \
  --train_dataset mvtec \
  --test_datasets visa \
  --data_root /path/to/DATA_ROOT \
  --shots 1 2 4 \
  --seeds 0 1 2 \
  --epochs 10 \
  --batch_size 48 \
  --lr 1e-3 \
  --backbone ViT-B-16-plus-240 \
  --clip_checkpoint /path/to/vit_b_16_plus_240-laion400m_e32-699c4b84.pt \
  --input_size 240 \
  --alpha 0.5 \
  --beta 0.75 \
  --num_workers 8 \
  --amp \
  --device cuda:0
```

VisA to MVTec:

```bash
python3 train_inctrlv2.py \
  --train_dataset visa \
  --test_datasets mvtec \
  --data_root /path/to/DATA_ROOT \
  --shots 1 2 4 \
  --seeds 0 1 2 \
  --clip_checkpoint /path/to/vit_b_16_plus_240-laion400m_e32-699c4b84.pt \
  --device cuda:0
```

Checkpoints are saved as:

```text
results/inctrlv2/trained_on_<dataset>/shot_<K>/seed_<S>/last.pth
```

Only the final checkpoint is saved by default.

## Testing

```bash
python3 test_inctrlv2.py \
  --checkpoint results/inctrlv2/trained_on_mvtec/shot_4/seed_0/last.pth \
  --test_dataset visa \
  --shot 4 \
  --data_root /path/to/DATA_ROOT \
  --metrics image_auroc image_ap pixel_auroc pixel_pro \
  --device cuda:0
```

Outputs are written to:

```text
results/inctrlv2_eval/test_on_<dataset>/shot_<K>/seed_<S>/
  metrics.json
  anomaly_scores.csv
  pixel_metrics.json
  visualizations/
```

Use `--save_maps` to write predicted anomaly maps.

## Ablations

Disable DASL:

```bash
python3 train_inctrlv2.py --train_dataset mvtec --data_root /path/to/DATA_ROOT --shots 4 --disable_dasl
```

Disable OASL:

```bash
python3 train_inctrlv2.py --train_dataset mvtec --data_root /path/to/DATA_ROOT --shots 4 --disable_oasl
```

Disable pixel losses:

```bash
python3 train_inctrlv2.py --train_dataset mvtec --data_root /path/to/DATA_ROOT --shots 4 --disable_pixel_loss
```

Alpha and beta sweeps:

```bash
for alpha in 0.0 0.25 0.5 0.75 1.0; do
  python3 train_inctrlv2.py --train_dataset mvtec --data_root /path/to/DATA_ROOT --shots 4 --alpha "$alpha"
done

for beta in 0.0 0.25 0.5 0.75 1.0; do
  python3 train_inctrlv2.py --train_dataset mvtec --data_root /path/to/DATA_ROOT --shots 4 --beta "$beta"
done
```

## Cloud Experiment Matrix

Run these on the cloud machine, not on the local laptop:

```text
train_on_mvtec -> test_on_visa, shots 1/2/4, seeds 0/1/2
train_on_visa  -> test_on_mvtec, shots 1/2/4, seeds 0/1/2
```

For each checkpoint, run `test_inctrlv2.py` and return `metrics.json`, `pixel_metrics.json`, and `anomaly_scores.csv`.

## OOM Handling

The phase-1 target is ViT-B/16+ at 240x240 on a single RTX 3090 with batch size 48. If training OOMs:

1. Keep `--input_size 240` and `--backbone ViT-B-16-plus-240`.
2. Keep `--amp` enabled.
3. Lower `--oasl_batch_size` first, for example `--oasl_batch_size 24`.
4. Lower physical `--batch_size` only if needed, then report the effective batch size used in the experiment note.
5. Reduce `--num_workers` if shared memory or dataloader workers fail.

ViT-L/14 is intentionally not prioritized in this phase.
