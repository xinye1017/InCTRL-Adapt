import argparse
from collections import defaultdict

from engine_IC import test, train
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

PUBLISHED_IN_DOMAIN_AUROC = {
    "elpv": {0: 0.733, 2: 0.839, 4: 0.846, 8: 0.872},
    "aitex": {0: 0.733, 2: 0.761, 4: 0.790, 8: 0.806},
    "visa": {0: 0.781, 2: 0.858, 4: 0.877, 8: 0.887},
    "mvtec": {0: 0.912, 2: 0.940, 4: 0.945, 8: 0.953},
}

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
    baselines = PUBLISHED_IN_DOMAIN_AUROC.get(test_dataset, {})
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


def parse_args():
    parser = argparse.ArgumentParser(description="Single-process InCTRL PQA Lite smoke runner.")
    parser.add_argument("--normal_json_path", required=True)
    parser.add_argument("--outlier_json_path", required=True)
    parser.add_argument("--val_normal_json_path", required=True)
    parser.add_argument("--val_outlier_json_path", required=True)
    parser.add_argument("--shot", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=240)
    parser.add_argument("--max_epoch", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--output_dir", default="./tmp/inctrl_pqa_lite_smoke")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_cfg()
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.normal_json_path = _as_cfg_path_list(args.normal_json_path)
    cfg.outlier_json_path = _as_cfg_path_list(args.outlier_json_path)
    cfg.val_normal_json_path = _as_cfg_path_list(args.val_normal_json_path)
    cfg.val_outlier_json_path = _as_cfg_path_list(args.val_outlier_json_path)
    cfg.shot = args.shot
    cfg.image_size = args.image_size
    cfg.steps_per_epoch = args.steps_per_epoch
    cfg.OUTPUT_DIR = args.output_dir
    cfg.SOLVER.MAX_EPOCH = args.max_epoch
    cfg.NUM_GPUS = 1
    cfg.NUM_SHARDS = 1
    cfg.SHARD_ID = 0
    cfg = assert_and_infer_cfg(cfg)
    train(cfg)
    test(cfg)


if __name__ == "__main__":
    main()
