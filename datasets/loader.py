#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.

"""Data loader."""

import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from .Sampler import DistributedSamplerWrapper
from .new_utlis import worker_init_fn_seed, BalancedBatchSampler

from .build import build_dataset
import open_clip.utils.misc as misc
import numpy as np


def multiple_samples_collate(batch):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    has_masks = len(batch[0]) == 4
    if has_masks:
        inputs, targets, labels, masks = zip(*batch)
    else:
        inputs, targets, labels = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    targets = [item for sublist in targets for item in sublist]
    labels = [item for sublist in labels for item in sublist]

    inputs, targets, labels = default_collate(inputs), default_collate(targets), default_collate(labels)
    if has_masks:
        masks = default_collate(masks)
        return inputs, targets, labels, masks

    return inputs, targets, labels


def construct_loader(cfg, split, transform):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """

    assert split in ["train", "val", "test"]
    data_path = cfg.DATA_LOADER.data_path
    data_name = cfg.TRAIN.DATASET
    shot = cfg.shot
    transform = transform

    normal_json_path = None
    outlier_json_path = None
    if split in ["train"]:
        normal_json_path = cfg.normal_json_path
        outlier_json_path = cfg.outlier_json_path
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))

    elif split in ["test"]:
        normal_json_path = cfg.val_normal_json_path
        outlier_json_path = cfg.val_outlier_json_path
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))

    # Construct the dataset
    few_shot_seed = int(getattr(cfg, "FEW_SHOT_SEED", 42))
    dataset = build_dataset(data_name, data_path, normal_json_path, outlier_json_path, transform, shot, few_shot_seed=few_shot_seed)

    # Create a sampler for multi-process training
    if cfg.AUG.NUM_SAMPLE > 1 and split in ["train"]:
        collate_func = multiple_samples_collate
    else:
        collate_func = None

    num_workers = int(cfg.DATA_LOADER.NUM_WORKERS)
    pin_memory = bool(cfg.DATA_LOADER.PIN_MEMORY)
    worker_kwargs = {}
    if num_workers > 0:
        # 对 few-shot 大样本批次使用更保守的预取，避免共享内存映射压力过大。
        worker_kwargs["persistent_workers"] = num_workers > 1
        worker_kwargs["prefetch_factor"] = 1

    # Create a loader
    if split in ["train"]:
        loader = torch.utils.data.DataLoader(
            dataset,
            worker_init_fn=worker_init_fn_seed,
            batch_sampler = BalancedBatchSampler(cfg, dataset),  # sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **worker_kwargs,
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate_func,
            **worker_kwargs,
        )

    return loader


def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the dataset.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    sampler = loader.sampler
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler.py type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)
