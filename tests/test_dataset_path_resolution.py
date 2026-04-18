import json
from pathlib import Path

from datasets.IC_dataset_new import DATA_ROOT, IC_dataset, PROJECT_ROOT
from train_local import collect_test_categories, prepare_dataset_registry


def _dataset_without_init():
    dataset = object.__new__(IC_dataset)
    dataset._resolved_path_cache = {}
    return dataset


def test_train_local_uses_repo_relative_ad_json_paths():
    registry = prepare_dataset_registry(["mvtec"])

    assert sorted(registry["test_categories"]) == ["aitex", "elpv", "visa"]
    assert collect_test_categories("aitex") == ["AITEX"]
    assert collect_test_categories("elpv") == ["elpv"]


def test_legacy_windows_json_image_paths_resolve_under_current_repo_data_root():
    sample_json = PROJECT_ROOT / "data" / "AD_json" / "aitex" / "AITEX_val_normal.json"
    sample_path = json.load(open(sample_json, encoding="utf-8"))[0]["image_path"]
    dataset = _dataset_without_init()

    resolved = dataset._resolve_image_path(sample_path)

    assert str(sample_path).startswith("D:\\")
    assert resolved.exists()
    assert resolved.is_relative_to(DATA_ROOT)


def test_cloud_absolute_json_image_paths_resolve_under_current_repo_data_root():
    dataset = _dataset_without_init()
    cloud_path = "/root/InCTRL/data/visa/candle/test/good/0011.JPG"

    resolved = dataset._resolve_image_path(cloud_path)

    assert resolved == PROJECT_ROOT / "data" / "visa" / "candle" / "test" / "good" / "0011.JPG"
    assert resolved.exists()


def test_repo_relative_json_image_paths_resolve_under_current_repo_root():
    dataset = _dataset_without_init()
    relative_path = Path("data") / "elpv" / "test" / "good" / "cell0075.png"

    resolved = dataset._resolve_image_path(str(relative_path))

    assert resolved == PROJECT_ROOT / relative_path
    assert resolved.exists()


def test_mvtec_outlier_mask_path_is_derived_from_image_path():
    dataset = _dataset_without_init()
    image_path = PROJECT_ROOT / "data" / "mvtec" / "bottle" / "test" / "broken_large" / "000.png"

    mask_path = dataset._resolve_mask_path(str(image_path), label=1)

    assert mask_path == PROJECT_ROOT / "data" / "mvtec" / "bottle" / "ground_truth" / "broken_large" / "000_mask.png"
    assert mask_path.exists()


def test_visa_outlier_mask_path_is_derived_from_image_path():
    dataset = _dataset_without_init()
    image_path = PROJECT_ROOT / "data" / "visa" / "candle" / "test" / "defect" / "000.JPG"

    mask_path = dataset._resolve_mask_path(str(image_path), label=1)

    assert mask_path == PROJECT_ROOT / "data" / "visa" / "candle" / "ground_truth" / "defect" / "000.png"
    assert mask_path.exists()


def test_normal_sample_uses_empty_mask():
    dataset = _dataset_without_init()
    image_path = PROJECT_ROOT / "data" / "visa" / "candle" / "test" / "good" / "0011.JPG"

    mask_path = dataset._resolve_mask_path(str(image_path), label=0)

    assert mask_path is None
