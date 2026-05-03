from validation.no_va_registry import DEFAULT_SEEDS, NO_VA_MODELS, get_model_specs
from validation.visual_utils import GallerySample, select_representative_samples


def _sample(index: int, label: int, score: float) -> GallerySample:
    return GallerySample(
        model_name="mvtec_2shot",
        train_dataset="mvtec",
        test_dataset="visa",
        category="pcb1",
        shot=2,
        seed=42,
        sample_index=index,
        query_path=f"/tmp/query_{index}.png",
        prompt_paths=[],
        label=label,
        score=score,
        prediction=1 if score >= 0.5 else 0,
        case_type="",
        border_color="",
        query_display=None,
        prompt_displays=[],
        residual_grid=None,
        final_map=None,
    )


def test_no_va_registry_uses_six_cloud_checkpoints_without_local_paths():
    assert DEFAULT_SEEDS == (42, 123, 7)
    assert set(NO_VA_MODELS) == {
        "mvtec_2shot",
        "mvtec_4shot",
        "mvtec_8shot",
        "visa_2shot",
        "visa_4shot",
        "visa_8shot",
    }
    assert NO_VA_MODELS["mvtec_2shot"].test_datasets == ("visa", "aitex", "elpv")
    assert NO_VA_MODELS["visa_8shot"].test_datasets == ("mvtec",)
    assert all(
        spec.checkpoint_path.startswith("/root/InCTRL/results/")
        and spec.checkpoint_path.endswith("/checkpoint_best.pyth")
        for spec in NO_VA_MODELS.values()
    )


def test_get_model_specs_preserves_requested_order():
    specs = get_model_specs(["visa_8shot", "mvtec_2shot"])
    assert [spec.name for spec in specs] == ["visa_8shot", "mvtec_2shot"]


def test_select_representative_samples_prefers_all_confusion_types_once():
    samples = [
        _sample(0, label=1, score=0.91),  # TP
        _sample(1, label=0, score=0.83),  # FP
        _sample(2, label=0, score=0.04),  # TN
        _sample(3, label=1, score=0.12),  # FN
        _sample(4, label=1, score=0.62),
    ]

    selected = select_representative_samples(samples, n_examples=4, seed=42)

    assert [sample.sample_index for sample in selected] == [0, 1, 2, 3]
    assert [sample.case_type for sample in selected] == [
        "True Positive",
        "False Positive",
        "True Negative",
        "False Negative",
    ]
