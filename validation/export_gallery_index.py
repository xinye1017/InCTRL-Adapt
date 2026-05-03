#!/usr/bin/env python3
"""Write category manifests and a root gallery index."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from validation.visual_utils import write_csv


def write_category_manifest(category_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(category_dir / "manifest.csv", rows)


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_name"], row["test_dataset"], row["category"])].append(row)

    summary_rows = []
    for (model_name, test_dataset, category), group in sorted(grouped.items()):
        case_counts = Counter(row["case_type"] for row in group if row["image_type"] == "sample_panel")
        summary_rows.append(
            {
                "model_name": model_name,
                "test_dataset": test_dataset,
                "category": category,
                "total_files": len(group),
                "sample_panels": sum(1 for row in group if row["image_type"] == "sample_panel"),
                "heatmap_overlays": sum(1 for row in group if row["image_type"] == "heatmap_overlay"),
                "residual_maps": sum(1 for row in group if row["image_type"] == "residual_map"),
                "true_positive": case_counts.get("True Positive", 0),
                "false_positive": case_counts.get("False Positive", 0),
                "true_negative": case_counts.get("True Negative", 0),
                "false_negative": case_counts.get("False Negative", 0),
            }
        )
    write_csv(output_dir / "summary.csv", summary_rows)
    return summary_rows


def write_index(output_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# No-VA Final Visual Gallery",
        "",
        "This gallery contains heatmap overlays, patch residual maps, and per-sample panels for the six final no-VA checkpoints.",
        "",
        "| Model | Test dataset | Category | Files | Panels | Heatmaps | Residual maps | TP | FP | TN | FN |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {model_name} | {test_dataset} | {category} | {total_files} | {sample_panels} | "
            "{heatmap_overlays} | {residual_maps} | {true_positive} | {false_positive} | "
            "{true_negative} | {false_negative} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Directory Layout",
            "",
            "- `<model_name>/<test_dataset>/<category>/heatmap_overlays/`",
            "- `<model_name>/<test_dataset>/<category>/residual_maps/`",
            "- `<model_name>/<test_dataset>/<category>/sample_panels/`",
            "- each category directory contains `manifest.csv`",
            "- root directory contains `summary.csv` and this `index.md`",
        ]
    )
    (output_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gallery_index(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary_rows = write_summary(output_dir, rows)
    write_index(output_dir, summary_rows)


if __name__ == "__main__":
    raise SystemExit(
        "This module is called by validation/run_no_va_visual_validation.py; "
        "use that entry script for CLI execution."
    )
