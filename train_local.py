import argparse
import csv
import logging as py_logging
import os
import warnings
from collections import defaultdict

from open_clip.config.defaults import assert_and_infer_cfg, get_cfg


TEST_DATASETS_BY_TRAIN = {
    "mvtec": ["visa", "aitex", "elpv"],
    "visa": ["mvtec", "aitex", "elpv"],
    "aitex": ["mvtec", "visa", "elpv"],
    "elpv": ["mvtec", "visa", "aitex"],
}

DATASET_CATEGORIES = {
    "aitex": ["AITEX"],
    "elpv": ["elpv"],
    "visa": [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ],
    "mvtec": [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ],
}

# Original InCTRL paper results — these are CROSS-DOMAIN evaluation numbers
# (e.g. "visa" means train on VisA, test on that row's dataset).
# Do NOT use for in-domain comparison (train and test on the same dataset).
PUBLISHED_CROSS_DOMAIN_AUROC = {
    "elpv": {0: 0.733, 2: 0.839, 4: 0.846, 8: 0.872},
    "aitex": {0: 0.733, 2: 0.761, 4: 0.790, 8: 0.806},
    "visa": {0: 0.781, 2: 0.858, 4: 0.877, 8: 0.887},
    "mvtec": {0: 0.912, 2: 0.940, 4: 0.945, 8: 0.953},
}

# Keep old name as alias for any external code referencing it.
PUBLISHED_IN_DOMAIN_AUROC = PUBLISHED_CROSS_DOMAIN_AUROC

PHASE1_MVTEC_THRESHOLDS = {
    "mvtec->aitex": 0.73,
    "mvtec->elpv": 0.82,
    "mvtec->visa": 0.80,
}


def collect_test_categories(dataset_name):
    return list(DATASET_CATEGORIES.get(dataset_name.lower(), [dataset_name]))


def prepare_dataset_registry(train_datasets):
    train_datasets = [dataset.lower() for dataset in train_datasets]
    test_datasets = []
    for train_dataset in train_datasets:
        for test_dataset in TEST_DATASETS_BY_TRAIN.get(train_dataset, []):
            if test_dataset not in test_datasets:
                test_datasets.append(test_dataset)
    return {
        "train_datasets": train_datasets,
        "test_categories": test_datasets,
    }


def _round_delta(value):
    return None if value is None else round(value, 4)


def _baseline_delta(row):
    test_dataset = row.get("test_dataset")
    shot = int(row.get("eval_shot"))
    auroc = row.get("auroc")
    baselines = PUBLISHED_CROSS_DOMAIN_AUROC.get(test_dataset, {})
    in_domain = baselines.get(shot)
    zero_shot = baselines.get(0)
    return {
        "pair": f"{row.get('train_dataset')}->{test_dataset}",
        "train_dataset": row.get("train_dataset"),
        "test_dataset": test_dataset,
        "shot": shot,
        "auroc": auroc,
        "aupr": row.get("aupr"),
        "published_in_domain_auroc": in_domain,
        "delta_vs_in_domain": _round_delta(None if in_domain is None else auroc - in_domain),
        "published_zero_shot_auroc": zero_shot,
        "delta_vs_zero_shot": _round_delta(None if zero_shot is None else auroc - zero_shot),
    }


def _build_summary_analytics(summary_rows, train_datasets):
    aggregated = defaultdict(lambda: {"auroc": {}, "aupr": {}})
    for row in summary_rows:
        pair = f"{row.get('train_dataset')}->{row.get('test_dataset')}"
        shot = int(row.get("eval_shot"))
        aggregated[pair]["auroc"][shot] = row.get("auroc")
        aggregated[pair]["aupr"][shot] = row.get("aupr")

    baseline_deltas = [_baseline_delta(row) for row in summary_rows]
    phase1_exit = {}
    if "mvtec" in [dataset.lower() for dataset in train_datasets]:
        for row in summary_rows:
            if row.get("train_dataset") != "mvtec" or int(row.get("eval_shot")) != 4:
                continue
            pair = f"{row.get('train_dataset')}->{row.get('test_dataset')}"
            threshold = PHASE1_MVTEC_THRESHOLDS.get(pair)
            if threshold is None:
                continue
            margin = round(row.get("auroc") - threshold, 4)
            phase1_exit[pair] = {
                "threshold": threshold,
                "auroc": row.get("auroc"),
                "margin": margin,
                "passes": margin >= 0,
            }
        failing_pairs = [pair for pair, payload in phase1_exit.items() if not payload["passes"]]
        if phase1_exit:
            phase1_exit["_summary"] = {
                "all_pass": not failing_pairs,
                "failing_pairs": failing_pairs,
            }

    return {
        "aggregated": dict(aggregated),
        "baseline_deltas": baseline_deltas,
        "phase1_exit": phase1_exit,
    }


def _as_cfg_path_list(path_value):
    return path_value if isinstance(path_value, list) else [path_value]


def _expand_dataset_jsons(dataset_name, split="train"):
    """Expand a dataset name to per-category JSON paths.

    split='train' → {cat}_normal.json / {cat}_outlier.json
    split='val'    → {cat}_val_normal.json / {cat}_val_outlier.json
    """
    dataset_name = dataset_name.lower()
    categories = DATASET_CATEGORIES.get(dataset_name)
    if categories is None:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: {list(DATASET_CATEGORIES.keys())}")
    json_dir = os.path.join("data", "AD_json", dataset_name)
    if split == "train":
        normals = [os.path.join(json_dir, f"{cat}_normal.json") for cat in categories]
        outliers = [os.path.join(json_dir, f"{cat}_outlier.json") for cat in categories]
    else:
        normals = [os.path.join(json_dir, f"{cat}_val_normal.json") for cat in categories]
        outliers = [os.path.join(json_dir, f"{cat}_val_outlier.json") for cat in categories]
    return normals, outliers


def _experiment_name_from_cfg(cfg):
    if getattr(cfg.MODEL, "ACTIVE_MODEL", "InCTRL") == "InCTRL":
        return "inctrl"
    parts = []
    if bool(getattr(cfg.VISUAL_ADAPTER, "ENABLE", False)):
        parts.append("va")
    if bool(getattr(cfg.TEXT_BRANCH, "ENABLE", False)):
        parts.append("ta")
    if bool(getattr(cfg.PQA, "ENABLE", False)):
        parts.append("pqa")
    return "_".join(parts) if parts else "baseline"


def _default_output_dir(cfg):
    return os.path.join("results", _experiment_name_from_cfg(cfg), str(int(cfg.shot)))


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Single-process InCTRL PQA Lite smoke runner.")
    parser.add_argument("--train_dataset", default=None,
                        help="Dataset name for full-category training (mvtec, visa, aitex, elpv). Overrides --normal/outlier_json_path.")
    parser.add_argument("--normal_json_path", default=None)
    parser.add_argument("--outlier_json_path", default=None)
    parser.add_argument("--val_normal_json_path", default=None)
    parser.add_argument("--val_outlier_json_path", default=None)
    parser.add_argument("--shot", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=240)
    parser.add_argument("--max_epoch", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--eval_period", type=int, default=1,
                        help="Evaluate every N epochs. Default 1 for clear local metric trends.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm training progress bars.")
    parser.add_argument("--show_warnings", action="store_true", help="Show Python warnings during training.")
    parser.add_argument("--early_stop_patience", type=int, default=5,
                        help="Stop if val AUROC does not improve for N eval epochs. 0=disabled.")
    parser.add_argument("--test_dataset", default=None,
                        help="Dataset(s) for val + per-category eval. Single name or slash-separated (e.g. visa or visa/aitex/elpv)")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def configure_train_local_output(cfg):
    if not getattr(cfg.TRAIN, "SUPPRESS_WARNINGS", False):
        return

    warnings.filterwarnings("ignore")
    py_logging.captureWarnings(True)
    py_logging.getLogger("py.warnings").setLevel(py_logging.ERROR)


def main():
    args = parse_args()
    cfg = get_cfg()
    if args.opts:
        cfg.merge_from_list(args.opts)

    if args.train_dataset:
        train_normals, train_outliers = _expand_dataset_jsons(args.train_dataset, split="train")
        cfg.normal_json_path = train_normals
        cfg.outlier_json_path = train_outliers
    else:
        if not args.normal_json_path or not args.outlier_json_path:
            raise ValueError("Must provide --train_dataset or --normal_json_path + --outlier_json_path")
        cfg.normal_json_path = _as_cfg_path_list(args.normal_json_path)
        cfg.outlier_json_path = _as_cfg_path_list(args.outlier_json_path)

    if args.test_dataset:
        test_datasets = [d.strip() for d in args.test_dataset.split("/")]
        # First dataset used for val during training
        first_val_normals, first_val_outliers = _expand_dataset_jsons(test_datasets[0], split="val")
        cfg.val_normal_json_path = first_val_normals
        cfg.val_outlier_json_path = first_val_outliers
        cfg.eval_dataset_name = test_datasets[0].lower()
    else:
        if not args.val_normal_json_path or not args.val_outlier_json_path:
            raise ValueError("Must provide --test_dataset or --val_normal_json_path + --val_outlier_json_path")
        cfg.val_normal_json_path = _as_cfg_path_list(args.val_normal_json_path)
        cfg.val_outlier_json_path = _as_cfg_path_list(args.val_outlier_json_path)
        cfg.eval_dataset_name = "custom"
    cfg.train_dataset_name = (args.train_dataset or "custom").lower()
    # Only use published baselines for cross-domain evaluation (train ≠ test).
    # For in-domain (train == test), no published cross-domain baseline applies.
    if cfg.train_dataset_name != cfg.eval_dataset_name:
        cfg.eval_baseline_auroc = PUBLISHED_CROSS_DOMAIN_AUROC.get(cfg.eval_dataset_name, {}).get(args.shot, -1.0)
    else:
        cfg.eval_baseline_auroc = -1.0
    cfg.shot = args.shot
    cfg.image_size = args.image_size
    cfg.steps_per_epoch = args.steps_per_epoch
    cfg.OUTPUT_DIR = args.output_dir or _default_output_dir(cfg)
    cfg.SOLVER.MAX_EPOCH = args.max_epoch
    cfg.TRAIN.EVAL_PERIOD = args.eval_period
    cfg.TRAIN.SHOW_PROGRESS = not args.no_progress
    cfg.TRAIN.SUPPRESS_WARNINGS = not args.show_warnings
    cfg.TRAIN.EARLY_STOP_PATIENCE = args.early_stop_patience
    cfg.NUM_GPUS = 1
    cfg.NUM_SHARDS = 1
    cfg.SHARD_ID = 0
    cfg = assert_and_infer_cfg(cfg)
    configure_train_local_output(cfg)
    from engine_IC import test, train

    model, tokenizer, transform = train(cfg)
    if not args.test_dataset:
        test(cfg)

    if args.test_dataset:
        test_datasets = [d.strip() for d in args.test_dataset.split("/")]
        from engine_IC import eval_per_category
        result_rows = []
        for ds in test_datasets:
            category_results, mean_auroc, mean_aupr = eval_per_category(model, tokenizer, transform, cfg, ds)
            for row in category_results:
                result_rows.append({
                    "train_dataset": args.train_dataset or "custom",
                    "test_dataset": ds,
                    "shot": args.shot,
                    "row_type": "category",
                    "category": row["category"],
                    "auroc": row["auroc"],
                    "aupr": row["aupr"],
                    "active_model": cfg.MODEL.ACTIVE_MODEL,
                    "visual_adapter": cfg.VISUAL_ADAPTER.ENABLE,
                    "text_branch": cfg.TEXT_BRANCH.ENABLE,
                    "pqa": cfg.PQA.ENABLE,
                })
            result_rows.append({
                "train_dataset": args.train_dataset or "custom",
                "test_dataset": ds,
                "shot": args.shot,
                "row_type": "mean",
                "category": "MEAN",
                "auroc": mean_auroc,
                "aupr": mean_aupr,
                "active_model": cfg.MODEL.ACTIVE_MODEL,
                "visual_adapter": cfg.VISUAL_ADAPTER.ENABLE,
                "text_branch": cfg.TEXT_BRANCH.ENABLE,
                "pqa": cfg.PQA.ENABLE,
            })
        _write_csv(
            os.path.join(cfg.OUTPUT_DIR, "test_results.csv"),
            result_rows,
            [
                "train_dataset", "test_dataset", "shot", "row_type", "category",
                "auroc", "aupr", "active_model", "visual_adapter", "text_branch", "pqa",
            ],
        )


if __name__ == "__main__":
    main()
