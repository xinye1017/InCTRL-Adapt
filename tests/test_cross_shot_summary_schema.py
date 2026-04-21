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

from train_local import TEST_DATASETS_BY_TRAIN, _build_summary_analytics


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


# ---------------------------------------------------------------------------
# _build_summary_analytics — pure-python aggregation helper
# ---------------------------------------------------------------------------

def _make_rows(pairs_and_values):
    """Turn [(train, test, shot, auroc, aupr), ...] into summary_rows dicts."""
    return [
        {
            "train_dataset": train,
            "test_dataset": test,
            "eval_shot": shot,
            "auroc": auroc,
            "aupr": aupr,
        }
        for train, test, shot, auroc, aupr in pairs_and_values
    ]


def test_analytics_aggregates_rows_into_per_pair_shot_pivot():
    rows = _make_rows([
        ("mvtec", "aitex", 2, 0.65, 0.55),
        ("mvtec", "aitex", 4, 0.68, 0.59),
        ("mvtec", "aitex", 8, 0.74, 0.63),
        ("mvtec", "elpv", 4, 0.89, 0.94),
    ])
    analytics = _build_summary_analytics(rows, train_datasets=["mvtec"])

    assert set(analytics["aggregated"].keys()) == {"mvtec->aitex", "mvtec->elpv"}
    aitex = analytics["aggregated"]["mvtec->aitex"]
    assert aitex["auroc"] == {2: 0.65, 4: 0.68, 8: 0.74}
    assert aitex["aupr"] == {2: 0.55, 4: 0.59, 8: 0.63}
    assert analytics["aggregated"]["mvtec->elpv"]["auroc"] == {4: 0.89}


def test_analytics_computes_baseline_deltas_with_known_references():
    """Delta must be ours - published_in_domain, rounded to 4 decimals."""
    rows = _make_rows([
        ("mvtec", "aitex", 4, 0.676, 0.59),  # baseline 0.790 -> delta -0.114
        ("mvtec", "aitex", 2, 0.654, 0.56),  # baseline 0.761 -> delta -0.107
    ])
    analytics = _build_summary_analytics(rows, train_datasets=["mvtec"])

    deltas = {(d["pair"], d["shot"]): d for d in analytics["baseline_deltas"]}
    aitex_4 = deltas[("mvtec->aitex", 4)]
    assert aitex_4["published_in_domain_auroc"] == 0.790
    assert aitex_4["delta_vs_in_domain"] == pytest.approx(-0.114, abs=1e-4)
    assert aitex_4["published_zero_shot_auroc"] == 0.733
    assert aitex_4["delta_vs_zero_shot"] == pytest.approx(-0.057, abs=1e-4)

    # AITEX 2-shot baseline must be 0.761 (paper), not a copy of 4-shot.
    aitex_2 = deltas[("mvtec->aitex", 2)]
    assert aitex_2["published_in_domain_auroc"] == 0.761


def test_analytics_baseline_delta_none_when_reference_missing():
    rows = _make_rows([("mvtec", "unknown_dataset", 4, 0.5, 0.5)])
    analytics = _build_summary_analytics(rows, train_datasets=["mvtec"])
    delta_row = analytics["baseline_deltas"][0]
    assert delta_row["published_in_domain_auroc"] is None
    assert delta_row["delta_vs_in_domain"] is None
    assert delta_row["published_zero_shot_auroc"] is None
    assert delta_row["delta_vs_zero_shot"] is None


def test_analytics_phase1_exit_reports_pass_and_fail():
    rows = _make_rows([
        ("mvtec", "aitex", 4, 0.676, 0.59),  # threshold 0.73 -> FAIL
        ("mvtec", "elpv",  4, 0.886, 0.94),  # threshold 0.82 -> PASS
        ("mvtec", "visa",  4, 0.783, 0.81),  # threshold 0.80 -> FAIL
        ("mvtec", "aitex", 8, 0.742, 0.63),  # ignored (not shot=4)
    ])
    exit_block = _build_summary_analytics(rows, train_datasets=["mvtec"])["phase1_exit"]

    assert exit_block["mvtec->aitex"]["passes"] is False
    assert exit_block["mvtec->aitex"]["margin"] == pytest.approx(-0.054, abs=1e-4)
    assert exit_block["mvtec->elpv"]["passes"] is True
    assert exit_block["mvtec->visa"]["passes"] is False
    assert exit_block["_summary"]["all_pass"] is False
    assert exit_block["_summary"]["failing_pairs"] == ["mvtec->aitex", "mvtec->visa"]


def test_analytics_phase1_exit_all_pass_when_all_thresholds_met():
    rows = _make_rows([
        ("mvtec", "aitex", 4, 0.80, 0.70),
        ("mvtec", "elpv",  4, 0.90, 0.95),
        ("mvtec", "visa",  4, 0.85, 0.87),
    ])
    exit_block = _build_summary_analytics(rows, train_datasets=["mvtec"])["phase1_exit"]
    assert exit_block["_summary"]["all_pass"] is True
    assert exit_block["_summary"]["failing_pairs"] == []


def test_analytics_phase1_exit_is_empty_for_non_mvtec_runs():
    """Thresholds are MVTec-specific; VisA-only runs should skip phase1_exit."""
    rows = _make_rows([("visa", "mvtec", 4, 0.90, 0.93)])
    analytics = _build_summary_analytics(rows, train_datasets=["visa"])
    assert analytics["phase1_exit"] == {}
    # baseline_deltas and aggregated still populate for VisA-trained runs.
    assert analytics["aggregated"]["visa->mvtec"]["auroc"] == {4: 0.90}


def test_analytics_returns_plain_json_serializable_payload():
    """Guard: analytics must round-trip through json without tensors/Path objects."""
    rows = _make_rows([
        ("mvtec", "aitex", 4, 0.676, 0.59),
        ("mvtec", "elpv",  4, 0.886, 0.94),
    ])
    analytics = _build_summary_analytics(rows, train_datasets=["mvtec"])
    encoded = json.dumps(analytics)
    decoded = json.loads(encoded)
    assert "aggregated" in decoded
    assert "baseline_deltas" in decoded
    assert "phase1_exit" in decoded
