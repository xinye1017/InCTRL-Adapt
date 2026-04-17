# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import os
import os.path
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
import copy
import cv2
from .build import DATASET_REGISTRY
import logging

from PIL import Image
import numpy as np

from torchvision import utils as vutils

import torch
from torchvision.datasets.vision import VisionDataset, StandardTransform

logging.getLogger('PIL').setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
KNOWN_DATASETS = ("mvtec", "visa", "aitex", "elpv", "sdd")
DEFAULT_SUPPORT_CACHE_SIZE = max(0, int(os.environ.get("INCTRL_SUPPORT_IMAGE_CACHE_SIZE", "512")))

def tile_image(img, stride_ratio=0.8):
    height, width, _ = img.shape
    shorter_edge = min(height, width)
    tile_size = shorter_edge
    stride = int(tile_size * stride_ratio)

    tile_image = []
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            # Extract a tile from the image
            tile = img[y:y + tile_size, x:x + tile_size]
            tile_image.append(tile)

    return tile_image


@DATASET_REGISTRY.register()
class IC_dataset(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        normal_json_path_list: list,
        outlier_json_path_list: list,
        transform: Optional[Callable] = None,
        shot = None
    ) -> None:
        super().__init__(root)

        self.normal_samples = []
        self.outlier_samples = []
        self.image = []
        self.total_n = 0
        self.total_o = 0
        self.shot = shot

        if len(normal_json_path_list) == 1:
            cur_normal = json.load(open(normal_json_path_list[0]))
            self.normal_samples.extend(cur_normal)
        else:
            for idx, json_path in enumerate(normal_json_path_list):
                cur_normal = json.load(open(json_path))
                self.normal_samples.extend(cur_normal)

        if len(outlier_json_path_list) == 1:
            cur_outlier = json.load(open(outlier_json_path_list[0]))
            self.outlier_samples.extend(cur_outlier)
        else:
            for idx, json_path in enumerate(outlier_json_path_list):
                cur_outlier = json.load(open(json_path))
                self.outlier_samples.extend(cur_outlier)

        self.transform = transform
        self.total_n, self.total_o = len(self.normal_samples), len(self.outlier_samples)
        self.image = self.normal_samples + self.outlier_samples
        self.support_cache_size = DEFAULT_SUPPORT_CACHE_SIZE
        self._support_image_cache = OrderedDict()
        self._resolved_path_cache = {}

        # ========== 优化1: 预计算每个样本的shot采样索引 ==========
        # 构建 type -> normal_indices 映射
        self.type_to_normal_indices = {}
        for idx, sample in enumerate(self.normal_samples):
            sample_type = sample['type']
            if sample_type not in self.type_to_normal_indices:
                self.type_to_normal_indices[sample_type] = []
            self.type_to_normal_indices[sample_type].append(idx)

        # 为每个样本预计算shot索引（使用固定seed确保可复现）
        # 这样每个epoch采样结果相同，但避免重复IO
        self.precomputed_shot_indices = []
        rng = np.random.default_rng(42)  # 固定seed保证可复现
        for idx, sample in enumerate(self.image):
            sample_type = sample['type']
            normal_indices = self.type_to_normal_indices.get(sample_type, [])
            if not normal_indices:
                raise ValueError(f"No normal reference samples found for type '{sample_type}'.")

            # 对 normal query，尽量避免把样本自身放进 few-shot 参考集中。
            current_normal_idx = idx if idx < len(self.normal_samples) else None
            candidate_indices = [
                normal_idx for normal_idx in normal_indices if normal_idx != current_normal_idx
            ]

            if len(candidate_indices) >= shot:
                shot_indices = rng.choice(candidate_indices, size=shot, replace=False).tolist()
            elif len(candidate_indices) > 0:
                shot_indices = rng.choice(candidate_indices, size=shot, replace=True).tolist()
            else:
                # 极端情况下某一类只有当前这 1 张 normal 图，只能退回到该类自身采样。
                shot_indices = rng.choice(normal_indices, size=shot, replace=True).tolist()
            self.precomputed_shot_indices.append(shot_indices)

    def _clone_loaded_image(self, image):
        if isinstance(image, np.ndarray):
            return image.copy()
        return image.copy()

    def _load_uncached_image(self, path: str):
        if path[-3:] == 'npy':
            return np.load(path).copy()
        with Image.open(path) as image:
            return image.copy()

    def _load_image(self, path: str, use_cache: bool = False) -> Image.Image:
        path = str(self._resolve_image_path(path))
        if use_cache and self.support_cache_size > 0:
            cached_image = self._support_image_cache.get(path)
            if cached_image is not None:
                self._support_image_cache.move_to_end(path)
                return self._clone_loaded_image(cached_image)

        image = self._load_uncached_image(path)
        if use_cache and self.support_cache_size > 0:
            self._support_image_cache[path] = self._clone_loaded_image(image)
            if len(self._support_image_cache) > self.support_cache_size:
                self._support_image_cache.popitem(last=False)
        return image

    def _resolve_image_path(self, raw_path: str) -> Path:
        """Resolve legacy absolute paths to the current repo-local data directory."""
        raw_path = str(raw_path)
        cached_path = self._resolved_path_cache.get(raw_path)
        if cached_path is not None:
            return cached_path

        candidate = Path(raw_path)
        if candidate.exists():
            self._resolved_path_cache[raw_path] = candidate
            return candidate

        normalized = raw_path.replace("\\", "/")
        relative_candidate = PROJECT_ROOT / normalized
        if relative_candidate.exists():
            self._resolved_path_cache[raw_path] = relative_candidate
            return relative_candidate

        if normalized.startswith("data/"):
            data_candidate = PROJECT_ROOT / normalized
            if data_candidate.exists():
                self._resolved_path_cache[raw_path] = data_candidate
                return data_candidate

        parts = [part for part in normalized.split("/") if part]
        lowered_parts = [part.lower() for part in parts]
        for dataset_name in KNOWN_DATASETS:
            if dataset_name not in lowered_parts:
                continue
            dataset_idx = lowered_parts.index(dataset_name)
            data_candidate = DATA_ROOT.joinpath(*parts[dataset_idx:])
            if data_candidate.exists():
                self._resolved_path_cache[raw_path] = data_candidate
                return data_candidate

        data_marker = "/data/"
        lowered = normalized.lower()
        marker_idx = lowered.rfind(data_marker)
        if marker_idx != -1:
            suffix = normalized[marker_idx + len(data_marker):]
            data_candidate = DATA_ROOT / Path(suffix)
            if data_candidate.exists():
                self._resolved_path_cache[raw_path] = data_candidate
                return data_candidate

        self._resolved_path_cache[raw_path] = candidate
        return candidate

    def _combine_images(self, image, image2):
        h, w = image.shape[1], image.shape[2]
        dst = torch.cat([image, image2], dim=1)
        return dst

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample = self.image[index]
        image = self._load_image(sample['image_path'])
        label = sample['target']
        sample_type = sample['type']

        # decide mode for interpolation
        cur_transforms = self.transform
        image = cur_transforms(image)

        # ========== 优化1: 使用预计算的shot索引 ==========
        # 直接查表，避免每次random.sample + 重复IO
        shot_indices = self.precomputed_shot_indices[index]

        image_list = list()
        image_list.append(image)
        for normal_idx in shot_indices:
            n_img = self._load_image(
                self.normal_samples[normal_idx]['image_path'],
                use_cache=True,
            )
            n_img = cur_transforms(n_img)
            image_list.append(n_img)

        image_type = sample_type

        return image_list, image_type, label

    def __len__(self) -> int:
        return len(self.image)
