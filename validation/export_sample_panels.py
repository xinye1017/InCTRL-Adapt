#!/usr/bin/env python3
"""Export one compact panel per selected sample."""
from __future__ import annotations

from pathlib import Path

from validation.visual_utils import GallerySample, safe_filename, sample_manifest_row, save_sample_panel


def export_sample_panels(samples: list[GallerySample], category_dir: Path) -> list[dict]:
    rows = []
    if not samples:
        return rows

    image_dir = category_dir / "sample_panels"
    for rank, sample in enumerate(samples):
        case_name = safe_filename(sample.case_type)
        filename = f"seed_{sample.seed}_rank_{rank:02d}_sample_{sample.sample_index:05d}_{case_name}.png"
        save_path = image_dir / filename
        title = (
            f"{sample.model_name} | {sample.test_dataset}/{sample.category} | "
            f"{sample.shot}-shot seed={sample.seed} rank={rank}"
        )
        save_sample_panel(sample, title, save_path)
        rows.append(
            sample_manifest_row(
                sample,
                image_type="sample_panel",
                relative_path=Path("sample_panels") / filename,
            )
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(
        "This module is called by validation/run_no_va_visual_validation.py; "
        "use that entry script for CLI execution."
    )
