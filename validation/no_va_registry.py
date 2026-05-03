#!/usr/bin/env python3
"""Registry for the six no-VA final checkpoints used in visualization export."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


DEFAULT_SEEDS = (42, 123, 7)

DATASET_CATEGORIES = {
    "aitex": ("AITEX",),
    "elpv": ("elpv",),
    "visa": (
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ),
    "mvtec": (
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ),
}


@dataclass(frozen=True)
class NoVAModelSpec:
    name: str
    train_dataset: str
    shot: int
    checkpoint_path: str
    test_datasets: tuple[str, ...]


NO_VA_MODELS = {
    "mvtec_2shot": NoVAModelSpec(
        name="mvtec_2shot",
        train_dataset="mvtec",
        shot=2,
        checkpoint_path="/root/InCTRL/results/mvtec_va_ta_pqa_2shot_15ep/checkpoint_best.pyth",
        test_datasets=("visa", "aitex", "elpv"),
    ),
    "mvtec_4shot": NoVAModelSpec(
        name="mvtec_4shot",
        train_dataset="mvtec",
        shot=4,
        checkpoint_path="/root/InCTRL/results/mvtec_va_ta_pqa_4shot_15ep/checkpoint_best.pyth",
        test_datasets=("visa", "aitex", "elpv"),
    ),
    "mvtec_8shot": NoVAModelSpec(
        name="mvtec_8shot",
        train_dataset="mvtec",
        shot=8,
        checkpoint_path="/root/InCTRL/results/mvtec_va_ta_pqa_8shot_15ep/checkpoint_best.pyth",
        test_datasets=("visa", "aitex", "elpv"),
    ),
    "visa_2shot": NoVAModelSpec(
        name="visa_2shot",
        train_dataset="visa",
        shot=2,
        checkpoint_path="/root/InCTRL/results/visa_va_ta_pqa_2shot_15ep/checkpoint_best.pyth",
        test_datasets=("mvtec",),
    ),
    "visa_4shot": NoVAModelSpec(
        name="visa_4shot",
        train_dataset="visa",
        shot=4,
        checkpoint_path="/root/InCTRL/results/visa_va_ta_pqa_4shot_15ep/checkpoint_best.pyth",
        test_datasets=("mvtec",),
    ),
    "visa_8shot": NoVAModelSpec(
        name="visa_8shot",
        train_dataset="visa",
        shot=8,
        checkpoint_path="/root/InCTRL/results/visa_va_ta_pqa_8shot_15ep/checkpoint_best.pyth",
        test_datasets=("mvtec",),
    ),
}


def get_model_specs(model_names: Iterable[str] | None = None) -> list[NoVAModelSpec]:
    """Return model specs, preserving user-requested order when provided."""
    if model_names is None:
        return list(NO_VA_MODELS.values())

    specs = []
    unknown = []
    for name in model_names:
        spec = NO_VA_MODELS.get(name)
        if spec is None:
            unknown.append(name)
        else:
            specs.append(spec)
    if unknown:
        valid = ", ".join(NO_VA_MODELS)
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}. Valid models: {valid}")
    return specs


def filter_datasets(spec: NoVAModelSpec, requested: Iterable[str] | None = None) -> list[str]:
    """Return valid target datasets for a model."""
    if requested is None:
        return list(spec.test_datasets)

    requested_set = {dataset.lower() for dataset in requested}
    selected = [dataset for dataset in spec.test_datasets if dataset in requested_set]
    unknown = sorted(requested_set - set(DATASET_CATEGORIES))
    if unknown:
        valid = ", ".join(DATASET_CATEGORIES)
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}. Valid datasets: {valid}")
    return selected


def filter_categories(dataset: str, requested: Iterable[str] | None = None) -> list[str]:
    """Return categories for a dataset, optionally filtered by requested category names."""
    dataset_key = dataset.lower()
    if dataset_key not in DATASET_CATEGORIES:
        valid = ", ".join(DATASET_CATEGORIES)
        raise ValueError(f"Unknown dataset: {dataset}. Valid datasets: {valid}")

    categories = list(DATASET_CATEGORIES[dataset_key])
    if requested is None:
        return categories

    requested_set = set(requested)
    return [category for category in categories if category in requested_set]
