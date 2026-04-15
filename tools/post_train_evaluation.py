#!/usr/bin/env python3
"""Post-training evaluation, visualization, and model selection for InCTRL VA runs."""

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import open_clip  # noqa: E402
from datasets import loader as ds_loader  # noqa: E402
from open_clip.config.defaults import get_cfg  # noqa: E402
from open_clip.model import get_cast_dtype  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "data"
FEW_SHOT_ROOT = PROJECT_ROOT / "few-shot samples"
REPORTS_ROOT = PROJECT_ROOT / "reports"
FIGURES_ROOT = REPORTS_ROOT / "figures"
RESULTS_ROOT = REPORTS_ROOT / "post_train_evaluation"
MODEL_CONFIG_PATH = PROJECT_ROOT / "open_clip" / "model_configs" / "ViT-B-16-plus-240.json"

DEFAULT_CHECKPOINT_ROOTS = [
    PROJECT_ROOT / "checkpoints" / "InCTRL_trained_on_MVTec_VA_ablation",
    PROJECT_ROOT / "checkpoints" / "InCTRL_trained_on_MVTec_VA",
    PROJECT_ROOT / "checkpoints" / "InCTRL_trained_on_MVTec",
]

DATASET_SPECS = {
    "aitex": {
        "json_dir": DATA_ROOT / "AD_json" / "aitex",
        "few_shot_dir": FEW_SHOT_ROOT / "AITEX" / "AITEX",
        "single_category": "AITEX",
    },
    "elpv": {
        "json_dir": DATA_ROOT / "AD_json" / "elpv",
        "few_shot_dir": FEW_SHOT_ROOT / "elpv" / "elpv",
        "single_category": "elpv",
    },
    "visa": {
        "json_dir": DATA_ROOT / "AD_json" / "visa",
        "few_shot_dir": FEW_SHOT_ROOT / "visa" / "visa",
        "single_category": None,
    },
}

CHECKPOINT_FILENAMES = {"checkpoint", "checkpoint.pth", "checkpoint.pyth"}


@dataclass
class ExperimentSpec:
    experiment_name: str
    checkpoint_path: str
    checkpoint_root: str
    train_dataset: str
    train_shot: Optional[int]
    adapter_enabled: bool
    adapter_mode: str
    zero_init: Optional[bool]
    training_strategy: str
    loss_settings: str
    metadata_path: Optional[str]


def _convert_to_rgb(image):
    return image.convert("RGB") if hasattr(image, "convert") else image


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize(240, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(240),
            transforms.Lambda(_convert_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_categories(dataset_name: str) -> List[str]:
    spec = DATASET_SPECS[dataset_name]
    if spec["single_category"] is not None:
        return [spec["single_category"]]
    return sorted(
        path.name.replace("_val_normal.json", "")
        for path in spec["json_dir"].glob("*_val_normal.json")
    )


def find_few_shot_pt(dataset_name: str, category_name: str, shot: int) -> Path:
    shot_dir = DATASET_SPECS[dataset_name]["few_shot_dir"] / str(shot)
    if not shot_dir.exists():
        raise FileNotFoundError(f"few-shot directory does not exist: {shot_dir}")

    direct = shot_dir / f"{category_name}.pt"
    if direct.exists():
        return direct

    for candidate in shot_dir.glob("*.pt"):
        if candidate.stem.lower() == category_name.lower():
            return candidate

    raise FileNotFoundError(
        "few-shot pt not found: "
        f"dataset={dataset_name}, category={category_name}, shot={shot}, dir={shot_dir}"
    )


def build_cached_normal_list(few_shot_pt: Path, device: str):
    tensors = torch.load(few_shot_pt, map_location=device)
    return [tensor.to(device) for tensor in tensors]


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def infer_int_from_parts(parts: Sequence[str]) -> Optional[int]:
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return None


def infer_adapter_mode(root: Path, checkpoint_path: Path, metadata: Dict[str, Any]) -> str:
    if metadata.get("visual_adapter_mode"):
        return str(metadata["visual_adapter_mode"])

    relative_parts = checkpoint_path.parent.relative_to(root).parts
    for part in relative_parts:
        if part in {"global_only", "local_only", "global_local"}:
            return part

    root_name = root.name.lower()
    if "va" in root_name:
        return "legacy_va"
    return "baseline"


def infer_adapter_enabled(root: Path, adapter_mode: str, metadata: Dict[str, Any]) -> bool:
    if "visual_adapter_enabled" in metadata:
        return bool(metadata["visual_adapter_enabled"])
    if adapter_mode in {"global_only", "local_only", "global_local", "legacy_va"}:
        return True
    return "va" in root.name.lower()


def make_experiment_name(root: Path, checkpoint_path: Path, adapter_mode: str, shot: Optional[int]) -> str:
    if shot is None:
        return f"{root.name}/{checkpoint_path.parent.name}/{adapter_mode}"
    return f"{root.name}/{adapter_mode}/{shot}-shot"


def checkpoint_candidates(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (
        path
        for path in root.rglob("*")
        if path.is_file() and path.name in CHECKPOINT_FILENAMES
    )


def discover_checkpoints(checkpoint_roots: Sequence[Path]) -> List[ExperimentSpec]:
    experiments: List[ExperimentSpec] = []
    seen = set()

    for root in checkpoint_roots:
        root = root.resolve()
        for checkpoint_path in sorted(checkpoint_candidates(root)):
            if checkpoint_path in seen:
                continue
            seen.add(checkpoint_path)

            metadata_path = checkpoint_path.parent / "metadata.json"
            metadata = load_json(metadata_path)
            relative_parts = checkpoint_path.parent.relative_to(root).parts
            train_shot = metadata.get("shot") or infer_int_from_parts(relative_parts)
            train_shot = int(train_shot) if train_shot is not None else None
            adapter_mode = infer_adapter_mode(root, checkpoint_path, metadata)
            adapter_enabled = infer_adapter_enabled(root, adapter_mode, metadata)
            zero_init = metadata.get("visual_adapter_zero_init")

            experiments.append(
                ExperimentSpec(
                    experiment_name=metadata.get(
                        "experiment_name",
                        make_experiment_name(root, checkpoint_path, adapter_mode, train_shot),
                    ),
                    checkpoint_path=str(checkpoint_path),
                    checkpoint_root=str(root),
                    train_dataset=metadata.get("train_dataset", "mvtec"),
                    train_shot=train_shot,
                    adapter_enabled=adapter_enabled,
                    adapter_mode=adapter_mode,
                    zero_init=bool(zero_init) if zero_init is not None else None,
                    training_strategy=metadata.get("training_strategy", "mvtec_va_ablation"),
                    loss_settings=metadata.get(
                        "loss_settings",
                        "BinaryFocalLoss(final_score)+BinaryFocalLoss(img_ref_score)",
                    ),
                    metadata_path=str(metadata_path) if metadata_path.exists() else None,
                )
            )

    return experiments


def build_model_for_experiment(experiment: ExperimentSpec, device: str):
    with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)

    cfg = get_cfg()
    cfg.VISUAL_ADAPTER.ENABLE = bool(experiment.adapter_enabled)
    if experiment.adapter_enabled and experiment.adapter_mode != "legacy_va":
        cfg.VISUAL_ADAPTER.MODE = experiment.adapter_mode
    if experiment.zero_init is not None:
        cfg.VISUAL_ADAPTER.ZERO_INIT = bool(experiment.zero_init)

    from open_clip import model as _model_mod

    model = _model_mod.InCTRL(
        cfg,
        model_config["embed_dim"],
        model_config["vision_cfg"],
        model_config["text_cfg"],
        quick_gelu=False,
        cast_dtype=get_cast_dtype("fp32"),
    )

    checkpoint = torch.load(experiment.checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    load_info = model.load_state_dict(state_dict, strict=False)
    if load_info.missing_keys or load_info.unexpected_keys:
        print(
            "[WARN] Non-strict checkpoint load differences for "
            f"{experiment.experiment_name}: missing={load_info.missing_keys}, "
            f"unexpected={load_info.unexpected_keys}"
        )

    return model.to(device)


def safe_divide(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def threshold_from_policy(labels: np.ndarray, scores: np.ndarray, policy: str) -> float:
    if policy == "fixed_0.5":
        return 0.5

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return 0.5

    f1_values = []
    for p, r in zip(precision[:-1], recall[:-1]):
        f1_values.append(0.0 if (p + r) == 0 else 2 * p * r / (p + r))
    best_idx = int(np.argmax(f1_values))
    return float(thresholds[best_idx])


def image_metrics(labels: Sequence[int], scores: Sequence[float], threshold_policy: str) -> Dict[str, Any]:
    labels_np = np.asarray(labels).astype(int)
    scores_np = np.asarray(scores).astype(float)

    metrics: Dict[str, Any] = {
        "image_auroc": None,
        "image_aupr": None,
        "image_f1": None,
        "threshold": None,
        "false_positive_rate": None,
        "false_negative_rate": None,
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
    }

    if labels_np.size == 0 or np.unique(labels_np).size < 2:
        return metrics

    threshold = threshold_from_policy(labels_np, scores_np, threshold_policy)
    pred = (scores_np >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels_np, pred, labels=[0, 1]).ravel()

    metrics.update(
        {
            "image_auroc": float(roc_auc_score(labels_np, scores_np)),
            "image_aupr": float(average_precision_score(labels_np, scores_np)),
            "image_f1": float(f1_score(labels_np, pred, zero_division=0)),
            "threshold": threshold,
            "false_positive_rate": safe_divide(fp, fp + tn),
            "false_negative_rate": safe_divide(fn, fn + tp),
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
        }
    )
    return metrics


def empty_pixel_metrics() -> Dict[str, Any]:
    return {
        "pixel_auroc": None,
        "pixel_aupr": None,
        "pixel_f1": None,
        "pro": None,
        "pixel_metrics_available": False,
    }


def try_collect_pixel_arrays(batch, outputs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if len(batch) < 4 or not isinstance(outputs, (tuple, list)) or len(outputs) < 3:
        return [], []
    masks = batch[3]
    maps = outputs[2]
    if masks is None or maps is None:
        return [], []
    try:
        masks_np = masks.detach().cpu().numpy()
        maps_np = maps.detach().cpu().float().numpy()
    except AttributeError:
        return [], []
    return [masks_np], [maps_np]


def pro_like_score(pixel_labels: np.ndarray, pixel_scores: np.ndarray, max_fpr: float = 0.3) -> Optional[float]:
    if np.unique(pixel_labels).size < 2:
        return None
    thresholds = np.linspace(float(pixel_scores.max()), float(pixel_scores.min()), num=100)
    points = []
    negatives = pixel_labels == 0
    positives = pixel_labels == 1
    for threshold in thresholds:
        pred = pixel_scores >= threshold
        fpr = safe_divide(float(np.logical_and(pred, negatives).sum()), float(negatives.sum()))
        recall = safe_divide(float(np.logical_and(pred, positives).sum()), float(positives.sum()))
        if fpr is not None and recall is not None and fpr <= max_fpr:
            points.append((fpr, recall))
    if len(points) < 2:
        return None
    points = sorted(points)
    xs = np.asarray([point[0] for point in points])
    ys = np.asarray([point[1] for point in points])
    return float(np.trapz(ys, xs) / max_fpr)


def pixel_metrics(pixel_labels: Sequence[np.ndarray], pixel_scores: Sequence[np.ndarray]) -> Dict[str, Any]:
    if not pixel_labels or not pixel_scores:
        return empty_pixel_metrics()

    labels_np = np.concatenate([np.asarray(item).reshape(-1) for item in pixel_labels]).astype(int)
    scores_np = np.concatenate([np.asarray(item).reshape(-1) for item in pixel_scores]).astype(float)
    if labels_np.size == 0 or np.unique(labels_np).size < 2:
        return empty_pixel_metrics()

    threshold = threshold_from_policy(labels_np, scores_np, "max_f1")
    pred = (scores_np >= threshold).astype(int)
    return {
        "pixel_auroc": float(roc_auc_score(labels_np, scores_np)),
        "pixel_aupr": float(average_precision_score(labels_np, scores_np)),
        "pixel_f1": float(f1_score(labels_np, pred, zero_division=0)),
        "pro": pro_like_score(labels_np, scores_np),
        "pixel_metrics_available": True,
    }


@torch.no_grad()
def collect_scores(model, tokenizer, loader, device: str, cached_normal_list):
    model.eval()
    labels_all: List[int] = []
    scores_all: List[float] = []
    pixel_labels: List[np.ndarray] = []
    pixel_scores: List[np.ndarray] = []
    failure_rows: List[Dict[str, Any]] = []

    for batch_index, batch in enumerate(tqdm(loader, desc="[TEST] Batch", leave=False)):
        inputs, types, labels = batch[:3]
        labels = labels.to(device)
        outputs = model(tokenizer, inputs, types, cached_normal_list)
        scores = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        batch_pixel_labels, batch_pixel_scores = try_collect_pixel_arrays(batch, outputs)
        pixel_labels.extend(batch_pixel_labels)
        pixel_scores.extend(batch_pixel_scores)
        scores_np = scores.detach().cpu().float().numpy()
        labels_np = labels.detach().cpu().numpy().astype(int)

        labels_all.extend(labels_np.tolist())
        scores_all.extend(scores_np.tolist())

        for row_index, (label, score) in enumerate(zip(labels_np.tolist(), scores_np.tolist())):
            failure_rows.append(
                {
                    "batch_index": batch_index,
                    "row_index": row_index,
                    "label": int(label),
                    "score": float(score),
                    "type": types[row_index] if isinstance(types, (list, tuple)) else str(types),
                }
            )

    return np.asarray(labels_all), np.asarray(scores_all), pixel_labels, pixel_scores, failure_rows


def evaluate_dataset(
    model,
    tokenizer,
    transform,
    dataset_name: str,
    eval_shot: int,
    device: str,
    threshold_policy: str,
) -> Dict[str, Any]:
    cfg = get_cfg()
    cfg.NUM_GPUS = 1
    cfg.TEST.BATCH_SIZE = 8
    cfg.DATA_LOADER.NUM_WORKERS = 0
    cfg.DATA_LOADER.PIN_MEMORY = device == "cuda"
    cfg.shot = eval_shot

    all_labels: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    all_pixel_labels: List[np.ndarray] = []
    all_pixel_scores: List[np.ndarray] = []
    category_results: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []

    for category in find_categories(dataset_name):
        json_dir = DATASET_SPECS[dataset_name]["json_dir"]
        cfg.val_normal_json_path = [str(json_dir / f"{category}_val_normal.json")]
        cfg.val_outlier_json_path = [str(json_dir / f"{category}_val_outlier.json")]

        loader = ds_loader.construct_loader(cfg, "test", transform)
        few_shot_pt = find_few_shot_pt(dataset_name, category, eval_shot)
        cached_normal = build_cached_normal_list(few_shot_pt, device)

        labels, scores, pixel_labels, pixel_scores, category_failures = collect_scores(
            model, tokenizer, loader, device, cached_normal
        )
        metrics = image_metrics(labels, scores, threshold_policy)
        metrics.update(pixel_metrics(pixel_labels, pixel_scores))
        metrics.update(
            {
                "category": category,
                "n_samples": int(labels.shape[0]),
                "few_shot_path": str(few_shot_pt),
            }
        )
        category_results.append(metrics)
        all_labels.append(labels)
        all_scores.append(scores)
        all_pixel_labels.extend(pixel_labels)
        all_pixel_scores.extend(pixel_scores)

        for failure in category_failures:
            failure.update({"dataset": dataset_name, "category": category})
        failure_rows.extend(category_failures)

    labels = np.concatenate(all_labels) if all_labels else np.asarray([])
    scores = np.concatenate(all_scores) if all_scores else np.asarray([])
    aggregate_metrics = image_metrics(labels, scores, threshold_policy)
    aggregate_metrics.update(pixel_metrics(all_pixel_labels, all_pixel_scores))

    valid_aurocs = [row["image_auroc"] for row in category_results if row["image_auroc"] is not None]
    valid_auprs = [row["image_aupr"] for row in category_results if row["image_aupr"] is not None]
    aggregate_metrics.update(
        {
            "macro_image_auroc": float(np.mean(valid_aurocs)) if valid_aurocs else None,
            "macro_image_aupr": float(np.mean(valid_auprs)) if valid_auprs else None,
            "n_samples": int(labels.shape[0]),
            "category_results": category_results,
            "failure_rows": failure_rows,
        }
    )
    return aggregate_metrics


def evaluate_checkpoint(
    experiment: ExperimentSpec,
    datasets: Sequence[str],
    eval_shots: Sequence[int],
    device: str,
    threshold_policy: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    print(f"[INFO] Loading {experiment.experiment_name}: {experiment.checkpoint_path}")
    model = build_model_for_experiment(experiment, device)
    tokenizer = open_clip.get_tokenizer("ViT-B-16-plus-240")
    transform = get_transform()
    rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []

    for eval_shot in eval_shots:
        for dataset in datasets:
            print(f"[INFO] Evaluating {experiment.experiment_name} | {dataset} | eval_shot={eval_shot}")
            result = evaluate_dataset(
                model,
                tokenizer,
                transform,
                dataset,
                eval_shot,
                device,
                threshold_policy,
            )
            row = {
                **asdict(experiment),
                "dataset": dataset,
                "eval_shot": eval_shot,
                "n_samples": result["n_samples"],
                "image_auroc": result["image_auroc"],
                "image_aupr": result["image_aupr"],
                "image_f1": result["image_f1"],
                "macro_image_auroc": result["macro_image_auroc"],
                "macro_image_aupr": result["macro_image_aupr"],
                "pixel_auroc": result["pixel_auroc"],
                "pixel_aupr": result["pixel_aupr"],
                "pixel_f1": result["pixel_f1"],
                "pro": result["pro"],
                "false_positive_rate": result["false_positive_rate"],
                "false_negative_rate": result["false_negative_rate"],
                "true_positive": result["true_positive"],
                "false_positive": result["false_positive"],
                "true_negative": result["true_negative"],
                "false_negative": result["false_negative"],
                "threshold": result["threshold"],
                "pixel_metrics_available": result["pixel_metrics_available"],
            }
            rows.append(row)

            for failure in result["failure_rows"]:
                failure.update(
                    {
                        "experiment_name": experiment.experiment_name,
                        "adapter_mode": experiment.adapter_mode,
                        "train_shot": experiment.train_shot,
                        "eval_shot": eval_shot,
                        "checkpoint_path": experiment.checkpoint_path,
                    }
                )
            failure_rows.extend(result["failure_rows"])

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return rows, failure_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def metric_value(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return float(value)


def nanmean_metric(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    values = [metric_value(row, key, np.nan) for row in rows]
    if not values or np.all(np.isnan(values)):
        return None
    return float(np.nanmean(values))


def aggregate_experiment_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["experiment_name"], []).append(row)

    aggregates: List[Dict[str, Any]] = []
    for experiment_name, group in grouped.items():
        first = group[0]
        aggregates.append(
            {
                "experiment_name": experiment_name,
                "checkpoint_path": first["checkpoint_path"],
                "adapter_mode": first["adapter_mode"],
                "train_shot": first["train_shot"],
                "adapter_enabled": first["adapter_enabled"],
                "training_strategy": first["training_strategy"],
                "mean_image_auroc": nanmean_metric(group, "image_auroc"),
                "mean_image_aupr": nanmean_metric(group, "image_aupr"),
                "mean_image_f1": nanmean_metric(group, "image_f1"),
                "mean_false_positive_rate": nanmean_metric(group, "false_positive_rate"),
                "mean_false_negative_rate": nanmean_metric(group, "false_negative_rate"),
                "datasets": ",".join(sorted({row["dataset"] for row in group})),
                "eval_shots": ",".join(str(item) for item in sorted({row["eval_shot"] for row in group})),
            }
        )
    return aggregates


def balanced_score(row: Dict[str, Any]) -> float:
    auroc = metric_value(row, "mean_image_auroc")
    aupr = metric_value(row, "mean_image_aupr")
    f1 = metric_value(row, "mean_image_f1")
    fpr = metric_value(row, "mean_false_positive_rate")
    fnr = metric_value(row, "mean_false_negative_rate")
    return (0.35 * auroc) + (0.25 * aupr) + (0.20 * f1) + (0.10 * (1 - fpr)) + (0.10 * (1 - fnr))


def select_best_model(aggregate_rows: Sequence[Dict[str, Any]], priority: str) -> List[Dict[str, Any]]:
    scored_rows = []
    for row in aggregate_rows:
        enriched = dict(row)
        enriched["balanced_score"] = balanced_score(row)
        scored_rows.append(enriched)

    if priority == "highest_auroc":
        key = lambda row: metric_value(row, "mean_image_auroc")
    elif priority == "lowest_fpr":
        key = lambda row: -metric_value(row, "mean_false_positive_rate", 1.0)
    elif priority == "balanced":
        key = lambda row: metric_value(row, "balanced_score")
    else:
        raise ValueError(f"Unsupported selection priority: {priority}")

    ranked = sorted(scored_rows, key=key, reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def infer_visual_adapter_gain(mode_summary: Dict[str, Dict[str, Any]]) -> Optional[bool]:
    baseline = mode_summary.get("baseline")
    va_modes = [
        summary for mode, summary in mode_summary.items()
        if mode in {"global_only", "local_only", "global_local", "legacy_va"}
    ]
    if baseline is None or not va_modes:
        return None
    return max(metric_value(summary, "mean_auroc") for summary in va_modes) > metric_value(baseline, "mean_auroc")


def infer_complexity_justification(mode_summary: Dict[str, Dict[str, Any]]) -> str:
    global_only = mode_summary.get("global_only")
    global_local = mode_summary.get("global_local")
    if not global_only or not global_local:
        return "insufficient_data"
    auroc_gain = metric_value(global_local, "mean_auroc") - metric_value(global_only, "mean_auroc")
    fpr_gain = metric_value(global_only, "mean_fpr") - metric_value(global_local, "mean_fpr")
    if auroc_gain >= 0.005 and fpr_gain >= -0.01:
        return "global_local_complexity_justified"
    return "prefer_simpler_global_only_until_more_evidence"


def compare_ablation(aggregate_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for row in aggregate_rows:
        by_mode.setdefault(str(row["adapter_mode"]), []).append(row)

    mode_summary = {}
    for mode, rows in by_mode.items():
        mode_summary[mode] = {
            "n_models": len(rows),
            "mean_auroc": nanmean_metric(rows, "mean_image_auroc"),
            "mean_aupr": nanmean_metric(rows, "mean_image_aupr"),
            "mean_f1": nanmean_metric(rows, "mean_image_f1"),
            "mean_fpr": nanmean_metric(rows, "mean_false_positive_rate"),
            "mean_fnr": nanmean_metric(rows, "mean_false_negative_rate"),
        }

    best_mode = None
    if mode_summary:
        best_mode = max(mode_summary, key=lambda mode: metric_value(mode_summary[mode], "mean_auroc"))

    return {
        "mode_summary": mode_summary,
        "best_mode_by_mean_auroc": best_mode,
        "visual_adapter_improves": infer_visual_adapter_gain(mode_summary),
        "complexity_justified": infer_complexity_justification(mode_summary),
    }


def import_matplotlib():
    import matplotlib.pyplot as plt

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("ggplot")
    return plt


def save_bar_chart(path: Path, labels: List[str], values: List[float], title: str, ylabel: str) -> None:
    plt = import_matplotlib()
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4.8))
    ax.bar(labels, values, color="#3A7CA5")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def visualize_results(rows: Sequence[Dict[str, Any]], ranked_rows: Sequence[Dict[str, Any]], figures_dir: Path) -> List[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: List[str] = []
    if not ranked_rows:
        return figure_paths

    labels = [row["experiment_name"] for row in ranked_rows]
    plots = [
        ("ablation_auroc_bar.png", "Ablation Comparison: Mean Image AUROC", "Mean AUROC", "mean_image_auroc"),
        ("metric_f1_bar.png", "Metric Comparison: Mean Image F1", "Mean F1", "mean_image_f1"),
        ("false_positive_rate_bar.png", "False Positive Rate Comparison", "Mean FPR", "mean_false_positive_rate"),
        ("false_negative_rate_bar.png", "False Negative Rate Comparison", "Mean FNR", "mean_false_negative_rate"),
    ]
    for filename, title, ylabel, key in plots:
        path = figures_dir / filename
        save_bar_chart(path, labels, [metric_value(row, key) for row in ranked_rows], title, ylabel)
        figure_paths.append(str(path))

    plt = import_matplotlib()
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
    for metric_key, label in [
        ("mean_image_auroc", "AUROC"),
        ("mean_image_aupr", "AUPR"),
        ("mean_image_f1", "F1"),
    ]:
        ax.plot(labels, [metric_value(row, metric_key) for row in ranked_rows], marker="o", label=label)
    ax.set_title("Metric Trend Across Ranked Checkpoints")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "metric_comparison_line.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(str(path))

    confusion_labels = [f"{row['experiment_name']}|{row['dataset']}|{row['eval_shot']}" for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(confusion_labels) * 0.8), 5))
    bottom = np.zeros(len(confusion_labels))
    for key, label, color in [
        ("true_positive", "TP", "#2E7D32"),
        ("false_positive", "FP", "#C62828"),
        ("true_negative", "TN", "#1565C0"),
        ("false_negative", "FN", "#EF6C00"),
    ]:
        values = [metric_value(row, key) for row in rows]
        ax.bar(confusion_labels, values, bottom=bottom, label=label, color=color)
        bottom += np.asarray(values)
    ax.set_title("Confusion Summary by Evaluation Row")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "confusion_summary_stacked.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(str(path))

    return figure_paths


def top_k_failures(failure_rows: Sequence[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    false_positive_candidates = [row for row in failure_rows if int(row["label"]) == 0]
    false_negative_candidates = [row for row in failure_rows if int(row["label"]) == 1]
    top_fp = sorted(false_positive_candidates, key=lambda row: row["score"], reverse=True)[:k]
    top_fn = sorted(false_negative_candidates, key=lambda row: row["score"])[:k]
    for row in top_fp:
        row["failure_type"] = "false_positive_candidate"
    for row in top_fn:
        row["failure_type"] = "false_negative_candidate"
    return top_fp + top_fn


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[Tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(fmt(row.get(key)) for _, key in columns) + " |")
    return "\n".join([header, sep] + body)


def generate_report(
    report_path: Path,
    ranked_rows: Sequence[Dict[str, Any]],
    ablation: Dict[str, Any],
    figure_paths: Sequence[str],
    selection_priority: str,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    best = ranked_rows[0] if ranked_rows else None

    ranking_table = markdown_table(
        ranked_rows,
        [
            ("Rank", "rank"),
            ("Experiment", "experiment_name"),
            ("Mode", "adapter_mode"),
            ("Train Shot", "train_shot"),
            ("AUROC", "mean_image_auroc"),
            ("AUPR", "mean_image_aupr"),
            ("F1", "mean_image_f1"),
            ("FPR", "mean_false_positive_rate"),
            ("FNR", "mean_false_negative_rate"),
            ("Balanced", "balanced_score"),
        ],
    )

    mode_rows = [
        {"mode": mode, **summary}
        for mode, summary in sorted(ablation["mode_summary"].items())
    ]
    mode_table = markdown_table(
        mode_rows,
        [
            ("Mode", "mode"),
            ("N", "n_models"),
            ("Mean AUROC", "mean_auroc"),
            ("Mean AUPR", "mean_aupr"),
            ("Mean F1", "mean_f1"),
            ("Mean FPR", "mean_fpr"),
            ("Mean FNR", "mean_fnr"),
        ],
    )

    lines = [
        "# Final Model Selection Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Selection priority: `{selection_priority}`",
        "",
        "## Metric Ranking",
        "",
        ranking_table if ranked_rows else "No checkpoints were evaluated.",
        "",
        "## Ablation Conclusions",
        "",
        mode_table if mode_rows else "No ablation groups were available.",
        "",
        f"- Best adapter mode by mean AUROC: `{ablation.get('best_mode_by_mean_auroc')}`.",
        f"- Visual adapter improves over baseline: `{ablation.get('visual_adapter_improves')}`.",
        f"- Complexity judgment: `{ablation.get('complexity_justified')}`.",
        "",
        "## Visualization Summary",
        "",
    ]
    lines.extend([f"- `{path}`" for path in figure_paths] if figure_paths else ["No figures were generated."])
    lines.extend(
        [
            "",
            "## Recommended Final Training Strategy",
            "",
            (
                f"Use `{best['adapter_mode']}` with train shot `{best['train_shot']}`."
                if best else
                "No recommendation available because no valid checkpoints were evaluated."
            ),
            "",
            "## Recommended Production Model Checkpoint",
            "",
            f"`{best['checkpoint_path']}`" if best else "N/A",
            "",
            "## Notes",
            "",
            "- Image-level metrics are computed from model anomaly scores.",
            "- Pixel-level metrics and PRO are `N/A` unless the dataloader/model exposes masks and anomaly maps.",
            "- Top-k failure candidates are saved as score rows; raw image visualization requires sample paths from the dataset.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-roots",
        nargs="+",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOTS,
        help="Directories to scan recursively for checkpoint files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_SPECS.keys()),
        default=["aitex", "elpv", "visa"],
    )
    parser.add_argument(
        "--eval-shots",
        nargs="+",
        type=int,
        default=None,
        help="Few-shot prompt counts for evaluation. Defaults to each checkpoint train shot.",
    )
    parser.add_argument(
        "--selection-priority",
        choices=["highest_auroc", "lowest_fpr", "balanced"],
        default="balanced",
    )
    parser.add_argument(
        "--threshold-policy",
        choices=["max_f1", "fixed_0.5"],
        default="max_f1",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    experiments = discover_checkpoints(args.checkpoint_roots)
    if not experiments:
        raise RuntimeError(f"No checkpoints found under: {args.checkpoint_roots}")

    all_rows: List[Dict[str, Any]] = []
    all_failures: List[Dict[str, Any]] = []

    for experiment in experiments:
        eval_shots = args.eval_shots or ([experiment.train_shot] if experiment.train_shot else [2, 4, 8])
        rows, failure_rows = evaluate_checkpoint(
            experiment,
            datasets=args.datasets,
            eval_shots=[int(shot) for shot in eval_shots],
            device=args.device,
            threshold_policy=args.threshold_policy,
        )
        all_rows.extend(rows)
        all_failures.extend(failure_rows)

    aggregate_rows = aggregate_experiment_rows(all_rows)
    ranked_rows = select_best_model(aggregate_rows, args.selection_priority)
    ablation = compare_ablation(aggregate_rows)
    failures = top_k_failures(all_failures, args.top_k)

    write_csv(RESULTS_ROOT / "experiment_results.csv", all_rows)
    write_csv(RESULTS_ROOT / "experiment_ranking.csv", ranked_rows)
    write_csv(RESULTS_ROOT / "top_failure_candidates.csv", failures)
    write_json(
        RESULTS_ROOT / "summary.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "selection_priority": args.selection_priority,
            "threshold_policy": args.threshold_policy,
            "experiments": [asdict(experiment) for experiment in experiments],
            "results": all_rows,
            "ranking": ranked_rows,
            "ablation": ablation,
            "top_failure_candidates": failures,
        },
    )

    figure_paths = []
    if not args.skip_figures:
        figure_paths = visualize_results(all_rows, ranked_rows, FIGURES_ROOT)

    generate_report(
        REPORTS_ROOT / "final_model_selection_report.md",
        ranked_rows,
        ablation,
        figure_paths,
        args.selection_priority,
    )
    print(f"[INFO] Results saved to {RESULTS_ROOT}")
    print(f"[INFO] Report saved to {REPORTS_ROOT / 'final_model_selection_report.md'}")


if __name__ == "__main__":
    main()
