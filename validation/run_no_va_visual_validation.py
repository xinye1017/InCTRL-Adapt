#!/usr/bin/env python3
"""Run no-VA final visualization export across configured models and datasets."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

import open_clip
from validation.export_gallery_index import write_category_manifest, write_gallery_index
from validation.export_heatmap_overlays import export_heatmap_overlays
from validation.export_residual_maps import export_residual_maps
from validation.export_sample_panels import export_sample_panels
from validation.no_va_registry import DEFAULT_SEEDS, filter_categories, filter_datasets, get_model_specs
from validation.visual_utils import (
    build_cfg,
    build_transform,
    collect_category_samples,
    load_model,
    resolve_output_dir,
    select_representative_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export no-VA final visual validation gallery")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model keys, e.g. mvtec_2shot visa_8shot. Default: all six no-VA final models.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Target datasets to include, e.g. visa aitex. Default: each model's registry targets.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Category names to include. Default: all categories in the selected target datasets.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Few-shot support seeds. Default: 42 123 7.",
    )
    parser.add_argument("--n_examples", type=int, default=6, help="Selected examples per category per seed.")
    parser.add_argument(
        "--output_dir",
        default="reports/no_va_visual_gallery",
        help="Gallery output directory. Relative paths are resolved from the repository root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = build_transform(240)
    tokenizer = open_clip.get_tokenizer("ViT-B-16-plus-240")
    all_manifest_rows = []

    for spec in get_model_specs(args.models):
        print(f"[MODEL] {spec.name}: loading {spec.checkpoint_path}")
        cfg = build_cfg(spec.shot, device)
        model = load_model(cfg, spec.checkpoint_path, device)

        for test_dataset in filter_datasets(spec, args.datasets):
            categories = filter_categories(test_dataset, args.categories)
            if not categories:
                print(f"[SKIP] {spec.name}/{test_dataset}: no selected categories")
                continue

            for category in categories:
                category_dir = output_dir / spec.name / test_dataset / category
                category_rows = []
                for seed in args.seeds:
                    print(
                        f"[EXPORT] model={spec.name} dataset={test_dataset} "
                        f"category={category} seed={seed}"
                    )
                    samples = collect_category_samples(
                        model=model,
                        tokenizer=tokenizer,
                        cfg=cfg,
                        transform=transform,
                        model_name=spec.name,
                        train_dataset=spec.train_dataset,
                        test_dataset=test_dataset,
                        category=category,
                        shot=spec.shot,
                        seed=seed,
                        device=device,
                    )
                    selected = select_representative_samples(samples, args.n_examples, seed)
                    print(f"[SELECTED] {len(selected)} / {len(samples)} samples")

                    category_rows.extend(export_heatmap_overlays(selected, category_dir))
                    category_rows.extend(export_residual_maps(selected, category_dir))
                    category_rows.extend(export_sample_panels(selected, category_dir))

                write_category_manifest(category_dir, category_rows)
                all_manifest_rows.extend(category_rows)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    write_gallery_index(output_dir, all_manifest_rows)
    print(f"[DONE] Gallery written to {output_dir}")


if __name__ == "__main__":
    main()
