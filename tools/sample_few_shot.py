#!/usr/bin/env python3
"""
Few-shot sampling script for InCTRL evaluation.

From each category's normal samples, randomly selects N images,
applies the standard CLIP preprocessing transform, and saves
the result as .pt files (list of tensors) compatible with
test_baseline.py and test_all_models.py.

Output format:  torch.save([tensor_1, tensor_2, ...], path)
  where each tensor has shape [3, 240, 240] after CLIP normalization.

Output directory structure (matches test_baseline.py expectations):
  <output_dir>/<DATASET>/<DATASET>/<shot>/<category>.pt

Examples:
  # Generate 2/4/8-shot samples for all datasets
  python tools/sample_few_shot.py --shot 2 4 8 --seed 42

  # Generate only 4-shot for AITEX and Visa
  python tools/sample_few_shot.py --shot 4 --datasets aitex visa --seed 0

  # Sample from val normal instead of train normal
  python tools/sample_few_shot.py --shot 2 --source val --seed 123

  # Custom output directory
  python tools/sample_few_shot.py --shot 4 --output_dir "few-shot samples" --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ── Project layout ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_JSON_ROOT = DATA_ROOT / "AD_json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "few-shot samples"

KNOWN_DATASETS = ("mvtec", "visa", "aitex", "elpv", "sdd")

# Maps CLI dataset name -> manifest subdir under AD_json/ and output folder name
# Output folder name is UPPER-CASE for aitex (matching test_baseline.py's
# FEW_SHOT_ROOT / "AITEX" / "AITEX") and lower-case for others.
DATASET_REGISTRY: dict[str, dict] = {
    "aitex": {
        "json_subdir": "aitex",
        "output_name": "AITEX",
        "single_category": "AITEX",      # only one category
    },
    "elpv": {
        "json_subdir": "elpv",
        "output_name": "elpv",
        "single_category": "elpv",
    },
    "visa": {
        "json_subdir": "visa",
        "output_name": "visa",
        "single_category": None,          # multi-category (candle, capsules, ...)
    },
    "mvtec": {
        "json_subdir": "mvtec",
        "output_name": "mvtec",
        "single_category": None,          # multi-category (bottle, cable, ...)
    },
    "sdd": {
        "json_subdir": "sdd",
        "output_name": "sdd",
        "single_category": "SDD",
    },
}


# ── Image transform (identical to test_baseline.py) ─────────────────────────

def _convert_to_rgb(image):
    return image.convert("RGB") if hasattr(image, "convert") else image


def get_clip_transform(image_size: int = 240):
    """Standard CLIP preprocessing used by InCTRL."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.Lambda(_convert_to_rgb),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


# ── Path resolution (replicates datasets/IC_dataset_new.py) ─────────────────

def resolve_image_path(raw_path: str) -> Path:
    """Resolve a legacy Windows/absolute path to the local data directory.

    Tries multiple heuristics (mirrors ``IC_dataset_new.py``):
    1. Raw path as-is (already correct on this machine).
    2. Relative to PROJECT_ROOT.
    3. Strip everything before the known dataset name (aitex, visa, ...).
    4. Strip everything after ``/data/``.
    """
    normalized = str(raw_path).replace("\\", "/")

    # 1. Direct
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    # 2. Relative to project root
    relative = PROJECT_ROOT / normalized
    if relative.exists():
        return relative

    # 3. data/ prefix
    if normalized.startswith("data/"):
        data_path = PROJECT_ROOT / normalized
        if data_path.exists():
            return data_path

    # 4. Find known dataset name in path
    parts = [p for p in normalized.split("/") if p]
    lowered = [p.lower() for p in parts]
    for ds_name in KNOWN_DATASETS:
        if ds_name not in lowered:
            continue
        idx = lowered.index(ds_name)
        data_path = DATA_ROOT.joinpath(*parts[idx:])
        if data_path.exists():
            return data_path

    # 5. /data/ marker
    marker = "/data/"
    marker_idx = normalized.lower().rfind(marker)
    if marker_idx != -1:
        suffix = normalized[marker_idx + len(marker):]
        data_path = DATA_ROOT / suffix
        if data_path.exists():
            return data_path

    # Return best guess (caller will check .exists())
    return Path(normalized)


# ── Category discovery ──────────────────────────────────────────────────────

def discover_categories(json_dir: Path, dataset_name: str, source: str = "train") -> list[str]:
    """Return sorted list of category names found in *json_dir*.

    When *source* is ``'train'``, matches ``{cat}_normal.json`` but excludes
    ``{cat}_val_normal.json``.  When ``'val'``, matches ``{cat}_val_normal.json``.
    """
    spec = DATASET_REGISTRY[dataset_name]
    if spec["single_category"] is not None:
        return [spec["single_category"]]

    if source == "val":
        suffix = "_val_normal.json"
        cats = []
        for f in sorted(json_dir.glob(f"*{suffix}")):
            cat = f.name[: -len(suffix)]
            if cat:
                cats.append(cat)
    else:
        # source == "train": match {cat}_normal.json but EXCLUDE {cat}_val_normal.json
        cats = []
        for f in sorted(json_dir.glob("*_normal.json")):
            if f.name.endswith("_val_normal.json"):
                continue
            cat = f.name[: -len("_normal.json")]
            if cat:
                cats.append(cat)
    return sorted(set(cats))


def load_normal_json(json_path: Path) -> list[dict]:
    """Load a JSON manifest (list of {image_path, target, type})."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Core sampling logic ─────────────────────────────────────────────────────

def sample_images(
    normal_samples: list[dict],
    n: int,
    rng: np.random.RandomState,
) -> list[torch.Tensor]:
    """Randomly pick *n* samples, load & transform, return list of tensors.

    Falls back to with-replacement sampling if the pool is smaller than *n*
    or if some images fail to load.
    """
    pool_size = len(normal_samples)
    replace = pool_size < n
    indices = rng.choice(pool_size, size=n, replace=replace).tolist()

    tensors: list[torch.Tensor] = []
    transform = get_clip_transform()

    for idx in indices:
        entry = normal_samples[idx]
        img_path = resolve_image_path(entry["image_path"])

        if not img_path.exists():
            print(f"    [SKIP] file not found: {img_path}")
            continue

        try:
            img = Image.open(img_path)
            tensors.append(transform(img))
        except Exception as exc:
            print(f"    [SKIP] failed to load {img_path}: {exc}")

    # If we lost some samples due to load failures, top up from the remaining pool
    if len(tensors) < n:
        used = set(indices)
        remaining = [i for i in range(pool_size) if i not in used]
        if not remaining:
            remaining = list(range(pool_size))
        extra_replace = len(remaining) < (n - len(tensors))
        extra_indices = rng.choice(
            remaining, size=n - len(tensors), replace=extra_replace
        ).tolist()
        for idx in extra_indices:
            if len(tensors) >= n:
                break
            entry = normal_samples[idx]
            img_path = resolve_image_path(entry["image_path"])
            if not img_path.exists():
                continue
            try:
                img = Image.open(img_path)
                tensors.append(transform(img))
            except Exception:
                continue

    return tensors


def save_few_shot_pt(
    tensors: list[torch.Tensor],
    output_path: Path,
    dataset_name: str,
    category: str,
    shot: int,
    expected_n: int,
) -> None:
    """Save tensors as .pt and print a summary line."""
    if not tensors:
        print(f"    [ERROR] {dataset_name}/{category}: no images loaded, skipping")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, str(output_path))

    status = "OK" if len(tensors) == expected_n else f"OK (only {len(tensors)}/{expected_n})"
    print(
        f"    [{status}] {dataset_name}/{category} ({shot}-shot): "
        f"{len(tensors)} tensors -> {output_path.relative_to(PROJECT_ROOT)}"
    )


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Few-shot sampling: from each category's normal samples, "
                    "randomly select N images, save as .pt (tensor list) "
                    "expected by test_baseline.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--shot", type=int, nargs="+", default=[2, 4, 8],
        help="Number of samples per category; can specify multiple (default: 2 4 8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output root directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--json_root", type=str, default=str(DEFAULT_JSON_ROOT),
        help=f"JSON manifests root directory (default: {DEFAULT_JSON_ROOT})",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=["aitex", "elpv", "visa"],
        choices=list(DATASET_REGISTRY.keys()),
        help="Datasets to process (default: aitex elpv visa)",
    )
    parser.add_argument(
        "--source", type=str, default="train",
        choices=["train", "val"],
        help="Sample from train normal (_normal.json) or val normal (_val_normal.json) "
             "(default: train)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only print the sampling plan; do not load images or write files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    json_root = Path(args.json_root)
    shots = sorted(set(args.shot))

    # ── Header ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Few-shot Sampling Script")
    print("=" * 70)
    print(f"  Shots    : {shots}")
    print(f"  Seed     : {args.seed}")
    print(f"  Source   : {args.source} normal")
    print(f"  Datasets : {args.datasets}")
    print(f"  JSON root: {json_root}")
    print(f"  Output   : {output_dir}")
    if args.dry_run:
        print("  ** DRY RUN -- no files will be written **")
    print("=" * 70)
    print()

    total_saved = 0
    total_failed = 0

    for ds_name in args.datasets:
        spec = DATASET_REGISTRY[ds_name]
        json_dir = json_root / spec["json_subdir"]

        if not json_dir.exists():
            print(f"[SKIP] {ds_name}: JSON directory not found: {json_dir}")
            continue

        categories = discover_categories(json_dir, ds_name, source=args.source)
        if not categories:
            print(f"[SKIP] {ds_name}: no categories found in {json_dir}")
            continue

        print(f"[{ds_name.upper()}] {len(categories)} categories: {categories}")

        for category in categories:
            # Build JSON path
            suffix = "_normal.json" if args.source == "train" else "_val_normal.json"
            json_path = json_dir / f"{category}{suffix}"

            if not json_path.exists():
                print(f"  [{category}] manifest not found: {json_path.name}")
                total_failed += 1
                continue

            normal_samples = load_normal_json(json_path)
            print(f"  [{category}] {len(normal_samples)} normal samples ({json_path.name})")

            for shot in shots:
                # Use a deterministic seed derived from (base_seed, dataset, category, shot)
                # so results are reproducible regardless of which subsets are requested.
                category_hash = hash(f"{ds_name}/{category}") & 0xFFFFFFFF
                derived_seed = (args.seed ^ category_hash ^ shot) & 0xFFFFFFFF
                rng = np.random.RandomState(derived_seed)

                out_path = (
                    output_dir
                    / spec["output_name"]
                    / spec["output_name"]
                    / str(shot)
                    / f"{category}.pt"
                )

                if args.dry_run:
                    print(
                        f"    [DRY] {ds_name}/{category} ({shot}-shot): "
                        f"would save -> {out_path.relative_to(PROJECT_ROOT)}"
                    )
                    continue

                tensors = sample_images(normal_samples, n=shot, rng=rng)
                save_few_shot_pt(
                    tensors=tensors,
                    output_path=out_path,
                    dataset_name=ds_name,
                    category=category,
                    shot=shot,
                    expected_n=shot,
                )
                if tensors:
                    total_saved += 1
                else:
                    total_failed += 1

        print()

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"Done.  Saved: {total_saved}  Failed: {total_failed}")
    if args.dry_run:
        print("(dry run -- no files were written)")
    print("=" * 70)


if __name__ == "__main__":
    main()
