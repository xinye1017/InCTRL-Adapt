from __future__ import annotations

import math
from typing import Dict

import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_metric(fn, labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if len(np.unique(labels)) < 2:
        return math.nan
    return float(fn(labels, scores))


def compute_image_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    return {
        "image_auroc": _safe_metric(roc_auc_score, labels, scores),
        "image_ap": _safe_metric(average_precision_score, labels, scores),
    }


def compute_pixel_auroc(pixel_maps: np.ndarray, masks: np.ndarray) -> float:
    masks = np.asarray(masks).astype(np.uint8).reshape(-1)
    pixel_maps = np.asarray(pixel_maps).astype(np.float32).reshape(-1)
    return _safe_metric(roc_auc_score, masks, pixel_maps)


def compute_pixel_pro(pixel_maps: np.ndarray, masks: np.ndarray, max_fpr: float = 0.3, num_thresholds: int = 200) -> float:
    """Compute per-region-overlap up to max FPR.

    Returns NaN when no positive mask region is available.
    """
    pixel_maps = np.asarray(pixel_maps, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.uint8)
    if masks.sum() == 0:
        return math.nan

    min_score = float(pixel_maps.min())
    max_score = float(pixel_maps.max())
    if max_score == min_score:
        thresholds = np.array([max_score], dtype=np.float32)
    else:
        thresholds = np.linspace(max_score, min_score, num_thresholds, dtype=np.float32)

    labeled_masks = []
    total_regions = 0
    for mask in masks:
        labeled, num_regions = ndimage.label(mask > 0)
        labeled_masks.append((labeled, num_regions))
        total_regions += num_regions
    if total_regions == 0:
        return math.nan

    pros = []
    fprs = []
    background = masks == 0
    background_count = int(background.sum())
    for threshold in thresholds:
        prediction = pixel_maps >= threshold
        if background_count == 0:
            fpr = 0.0
        else:
            fpr = float(np.logical_and(prediction, background).sum() / background_count)

        overlaps = []
        for pred_one, (labeled, num_regions) in zip(prediction, labeled_masks):
            for region_idx in range(1, num_regions + 1):
                region = labeled == region_idx
                overlaps.append(float(np.logical_and(pred_one, region).sum() / region.sum()))
        pros.append(float(np.mean(overlaps)))
        fprs.append(fpr)

    fprs = np.asarray(fprs)
    pros = np.asarray(pros)
    valid = fprs <= max_fpr
    if not np.any(valid):
        return 0.0
    if np.all(fprs[valid] == 0):
        return float(pros[valid].max())

    order = np.argsort(fprs[valid])
    x = fprs[valid][order]
    y = pros[valid][order]
    if x[0] > 0:
        x = np.concatenate([[0.0], x])
        y = np.concatenate([[y[0]], y])
    if x[-1] < max_fpr:
        x = np.concatenate([x, [max_fpr]])
        y = np.concatenate([y, [y[-1]]])
    return float(np.trapz(y, x) / max_fpr)


def compute_pixel_metrics(pixel_maps: np.ndarray, masks: np.ndarray) -> Dict[str, float]:
    return {
        "pixel_auroc": compute_pixel_auroc(pixel_maps, masks),
        "pixel_pro": compute_pixel_pro(pixel_maps, masks),
    }
