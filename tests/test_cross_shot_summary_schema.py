"""Schema-level invariants for cross-shot training summaries.

Phase 0 (2026-04-21): enforce strict cross-domain evaluation protocol.
Same-domain evaluation rows (where ``train_dataset == test_dataset``)
must never appear in the artifacts produced by
``train_local.run_all_experiments`` — they indicate same-domain
leakage and make baseline comparison ambiguous (the published InCTRL
numbers are leave-one-out).

These tests exercise pure-python invariants and do not require GPU
or data.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from train_local import TEST_DATASETS_BY_TRAIN


def test_cross_domain_mapping_has_no_same_domain_entries():
    for train_ds, test_dss in TEST_DATASETS_BY_TRAIN.items():
        assert train_ds not in test_dss, (
            f"Same-domain eval leak: train={train_ds} must not appear in {test_dss}"
        )


def test_cross_domain_mapping_covers_baseline_four_datasets():
    """Baseline reports (MVTec, VisA, AITEX, ELPV) must each appear as a test
    target under some training source. If this fails, part of the baseline
    table cannot be reproduced under the current mapping."""
    all_test_datasets = set()
    for _, test_dss in TEST_DATASETS_BY_TRAIN.items():
        all_test_datasets.update(test_dss)

    required = {"mvtec", "visa", "aitex", "elpv"}
    missing = required - all_test_datasets
    assert not missing, f"Cross-domain mapping is missing target datasets: {sorted(missing)}"


@pytest.mark.parametrize(
    "summary",
    [
        {
            "train_datasets": ["mvtec"],
            "summary_rows": [
                {"train_dataset": "mvtec", "test_dataset": "visa"},
                {"train_dataset": "mvtec", "test_dataset": "aitex"},
            ],
        },
        {
            "train_datasets": ["mvtec", "visa"],
            "summary_rows": [
                {"train_dataset": "mvtec", "test_dataset": "visa"},
                {"train_dataset": "visa", "test_dataset": "mvtec"},
            ],
        },
    ],
)
def test_valid_cross_domain_summary_passes_schema(summary):
    _assert_no_same_domain_rows(summary)


def test_same_domain_summary_row_is_rejected():
    bad_summary = {
        "train_datasets": ["mvtec"],
        "summary_rows": [
            {"train_dataset": "mvtec", "test_dataset": "mvtec"},  # leak
        ],
    }
    with pytest.raises(AssertionError, match="Same-domain"):
        _assert_no_same_domain_rows(bad_summary)


def test_existing_summary_files_respect_cross_domain_protocol():
    """If historical summary JSONs live under ``results/``, they must also
    be free of same-domain rows after Phase 0 lands."""
    results_dir = Path(__file__).resolve().parent.parent / "results"
    if not results_dir.exists():
        pytest.skip("results/ directory not populated in this checkout")
    summary_files = sorted(results_dir.glob("cross_shot_train_shot_*_summary.json"))
    if not summary_files:
        pytest.skip("no cross_shot summary files to validate")
    for summary_path in summary_files:
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        _assert_no_same_domain_rows(summary, label=str(summary_path))


def _assert_no_same_domain_rows(summary, label: str | None = None):
    rows = summary.get("summary_rows", [])
    for row in rows:
        train_ds = row.get("train_dataset")
        test_ds = row.get("test_dataset")
        assert train_ds != test_ds, (
            f"Same-domain row detected in {label or summary}: "
            f"train={train_ds} test={test_ds}"
        )
