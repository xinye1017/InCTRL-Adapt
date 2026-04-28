# InCTRL-Adapt: Integrating AdaptCLIP Adapters into InCTRL for Generalist Anomaly Detection

This repository extends the [InCTRL (CVPR 2024)](https://arxiv.org/abs/2403.06495) generalist anomaly detection framework with [AdaptCLIP](https://arxiv.org/abs/xxxxx)-style adapter modules — **VisualAdapter**, **TextualAdapter**, and **PQAdapter** — to enhance the model's few-shot anomaly detection capability across diverse domains.

## Background

**InCTRL** (In-context Residual Learning) trains a single model to detect anomalies across multiple datasets without dataset-specific fine-tuning. Given a query image and a few normal reference images (few-shot sample prompts), it computes feature residuals between the query and the normal samples. The core insight: anomalies produce larger residuals than normal samples, enabling cross-domain generalization.

This repository builds on InCTRL by integrating three AdaptCLIP-style adapter modules:

| Adapter | Purpose |
|---------|---------|
| **VisualAdapter** | Bottleneck residual adaptation of global image features and multi-layer patch features |
| **TextualAdapter** | Learnable prompt contexts for object-aware text descriptions |
| **PQAdapter** | Patch-Query alignment mechanism for finer anomaly localization |

## Key Features (beyond original InCTRL)

- **VisualAdapter**: Lightweight bottleneck adapter applied to both global CLS tokens and per-layer patch features before residual computation
- **TextualAdapter**: AdaptCLIP-style learnable prompt contexts replacing fixed text templates
- **PQAdapter**: Patch-to-query cross-attention for enhanced anomaly scoring
- **Alternating training**: Phase-switching between visual adapter and text adapter optimization
- **Multi-dataset evaluation**: Slash-separated dataset spec (e.g., `visa/aitex/elpv`)
- **Full-category training**: `--train_dataset` shorthand for training on all categories at once
- **Cached text features**: Speed up evaluation by caching CLIP text embeddings
- **Structured CSV results**: Per-category evaluation metrics exported to CSV

## Setup

```
python >= 3.10
torch >= 1.13.0
torchvision >= 0.14.0
scipy >= 1.10.1
scikit-image >= 0.21.0
numpy >= 1.24.3
tqdm >= 4.64.0
```

## Dataset Preparation

Supported datasets: MVTec AD, VisA, AITEX, ELPV, SDD, BrainMRI, HeadCT, MNIST, CIFAR-10

1. Download datasets and convert to MVTec AD format:

```
DATA_PATH/
    subset_1/
        train/good/
        test/good/
        test/defect_class_1/
        test/defect_class_2/
        ...
    ...
```

2. Generate JSON split files using the preprocessing scripts in `datasets/preprocess/`
3. Download few-shot normal samples and pretrained models (see original InCTRL Google Drive)

## Quick Start (Evaluation)

```bash
python test.py \
  --val_normal_json_path /path/to/normal.json \
  --val_outlier_json_path /path/to/outlier.json \
  --category candle \
  --few_shot_dir /path/to/few_shot_samples/visa/2/
```

Multi-dataset evaluation:

```bash
python test.py --test_dataset visa/aitex/elpv --few_shot_dir /path/to/few_shot_samples/
```

## Training

```bash
# Standard training
python main.py \
  --normal_json_path /path/to/train_normal.json \
  --outlier_json_path /path/to/train_outlier.json \
  --val_normal_json_path /path/to/val_normal.json \
  --val_outlier_json_path /path/to/val_outlier.json

# Full-dataset training with 2-shot
python main.py \
  --train_dataset visa \
  --dataset_path /path/to/datasets/visa \
  --few_shot 2
```

## Model Variants

The repository supports multiple model configurations via `open_clip/factory.py`:

- `InCTRL` — Original in-context residual learning model
- `InCTRLAdapt` — InCTRL + VisualAdapter + TextualAdapter
- `InCTRLPQA` — InCTRLAdapt + PQAdapter (full AdaptCLIP integration)
- `InCTRLPQA-fused` — Fused variant with alternative head design

## Project Structure

```
open_clip/
    model.py                    # Core CLIP + InCTRL model
    inctrl_adapt.py             # InCTRLAdapt with VisualAdapter + TextualAdapter
    visual_adapter.py           # VisualAdapter (bottleneck residual)
    adaptclip_textual_adapter.py # TextualAdapter (learnable prompt contexts)
    pqa_adapter.py              # PQAdapter (patch-query alignment)
    inctrl_pqa_losses.py        # PQA-specific loss functions
    object_agnostic_text.py     # Object-agnostic text prompt templates
main.py                         # Training entry point
test.py / test_baseline.py      # Evaluation entry points
engine_IC.py                    # Training loop
```

## Citation

```bibtex
@inproceedings{zhu2024toward,
  title={Toward generalist anomaly detection via in-context residual learning with few-shot sample prompts},
  author={Zhu, Jiawen and Pang, Guansong},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={17826--17836},
  year={2024}
}
```

## Related

- Original InCTRL: [mala-lab/InCTRL](https://github.com/mala-lab/InCTRL)
- WinCLIP reproduction (competing method): [mala-lab/WinCLIP](https://github.com/mala-lab/WinCLIP)
- This repository: [xinye1017/InCTRL-Adapt](https://github.com/xinye1017/InCTRL-Adapt)
