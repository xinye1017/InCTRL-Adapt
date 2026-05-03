#!/usr/bin/env python3
"""Export original image plus pixel anomaly heatmap overlays."""
from __future__ import annotations

from pathlib import Path

from validation.visual_utils import GallerySample, safe_filename, sample_manifest_row, save_heatmap_overlay_grid


def export_heatmap_overlays(samples: list[GallerySample], category_dir: Path) -> list[dict]:
    if not samples:
        return []

    image_dir = category_dir / "heatmap_overlays"
    seed = samples[0].seed
    title = (
        f"{samples[0].model_name} | {samples[0].test_dataset}/{samples[0].category} | "
        f"{samples[0].shot}-shot seed={seed}"
    )
    filename = f"heatmap_overlay_seed_{seed}.png"
    save_path = image_dir / filename
    save_heatmap_overlay_grid(samples, title, save_path)
    return [
        sample_manifest_row(
            sample,
            image_type="heatmap_overlay",
            relative_path=Path("heatmap_overlays") / filename,
        )
        for sample in samples
    ]


if __name__ == "__main__":
    raise SystemExit(
        "This module is called by validation/run_no_va_visual_validation.py; "
        "use that entry script for CLI execution."
    )
