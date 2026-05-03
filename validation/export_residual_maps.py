#!/usr/bin/env python3
"""Export normal references, query images, and patch residual maps."""
from __future__ import annotations

from pathlib import Path

from validation.visual_utils import GallerySample, sample_manifest_row, save_residual_map_grid


def export_residual_maps(samples: list[GallerySample], category_dir: Path) -> list[dict]:
    if not samples:
        return []

    image_dir = category_dir / "residual_maps"
    seed = samples[0].seed
    title = (
        f"{samples[0].model_name} | {samples[0].test_dataset}/{samples[0].category} | "
        f"{samples[0].shot}-shot residual maps seed={seed}"
    )
    filename = f"residual_map_seed_{seed}.png"
    save_path = image_dir / filename
    save_residual_map_grid(samples, title, save_path)
    return [
        sample_manifest_row(
            sample,
            image_type="residual_map",
            relative_path=Path("residual_maps") / filename,
        )
        for sample in samples
    ]


if __name__ == "__main__":
    raise SystemExit(
        "This module is called by validation/run_no_va_visual_validation.py; "
        "use that entry script for CLI execution."
    )
