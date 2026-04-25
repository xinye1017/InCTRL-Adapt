from __future__ import annotations

import json
import random
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Sampler
from torchvision import transforms

OPENCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENCLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}
DATASET_ALIASES = {
    "mvtec": ("mvtec", "mvtecad", "MVTec", "MVTecAD", "mvtec_anomaly_detection"),
    "visa": ("visa", "VisA", "visa_anomaly_detection"),
}


def _convert_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def _binarize_mask(mask: torch.Tensor) -> torch.Tensor:
    return (mask[:1] > 0.5).float()


@dataclass(frozen=True)
class AnomalySample:
    image_path: Path
    label: int
    class_name: str
    mask_path: Optional[Path] = None
    split: str = "test"
    defect_type: str = "good"


def resolve_dataset_root(data_root: str | Path, dataset_name: str) -> Path:
    root = Path(data_root).expanduser()
    if root.name.lower() in {dataset_name.lower(), *[alias.lower() for alias in DATASET_ALIASES.get(dataset_name, ())]}:
        return root
    candidates = DATASET_ALIASES.get(dataset_name.lower(), (dataset_name,))
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    return root / dataset_name


def build_image_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(size=input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=(input_size, input_size)),
            transforms.Lambda(_convert_rgb),
            transforms.ToTensor(),
            transforms.Normalize(mean=OPENCLIP_MEAN, std=OPENCLIP_STD),
        ]
    )


def build_mask_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(size=input_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(size=(input_size, input_size)),
            transforms.ToTensor(),
            transforms.Lambda(_binarize_mask),
        ]
    )


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _load_image(path: Path) -> Image.Image:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        return Image.fromarray(array.astype(np.uint8)).convert("RGB")
    with Image.open(path) as image:
        return image.convert("RGB")


def _load_mask(path: Optional[Path], input_size: int, mask_transform) -> torch.Tensor:
    if path is None or not path.exists():
        return torch.zeros(1, input_size, input_size, dtype=torch.float32)
    if path.suffix.lower() == ".npy":
        array = np.load(path)
        if array.ndim == 3:
            array = array[..., 0]
        image = Image.fromarray(array.astype(np.uint8))
    else:
        with Image.open(path) as mask_image:
            image = mask_image.convert("L")
    return mask_transform(image)


def _infer_mask_path(image_path: Path, class_root: Path, defect_type: str) -> Optional[Path]:
    if defect_type == "good":
        return None
    mask_dir = class_root / "ground_truth" / defect_type
    if not mask_dir.exists():
        return None
    candidates = []
    for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"):
        candidates.append(mask_dir / f"{image_path.stem}_mask{suffix}")
        candidates.append(mask_dir / f"{image_path.stem}{suffix}")
    candidates.append(mask_dir / image_path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(mask_dir.glob(f"{image_path.stem}*"))
    return matches[0] if matches else None


def _category_roots(dataset_root: Path) -> List[Path]:
    if (dataset_root / "train").exists() and (dataset_root / "test").exists():
        return [dataset_root]
    return sorted(path for path in dataset_root.iterdir() if path.is_dir() and (path / "test").exists())


def _scan_directory_samples(
    dataset_root: Path,
    split: str,
    normal_only: bool = False,
    include_test_good_for_train: bool = True,
) -> List[AnomalySample]:
    samples: List[AnomalySample] = []
    for class_root in _category_roots(dataset_root):
        class_name = class_root.name
        train_good = class_root / "train" / "good"
        test_root = class_root / "test"
        test_good = test_root / "good"

        normal_dirs = []
        if split == "train":
            if train_good.exists():
                normal_dirs.append((train_good, "train"))
            if include_test_good_for_train and test_good.exists() and not normal_only:
                normal_dirs.append((test_good, "test"))
        else:
            if test_good.exists():
                normal_dirs.append((test_good, "test"))

        for normal_dir, source_split in normal_dirs:
            for image_path in sorted(path for path in normal_dir.iterdir() if _is_image(path)):
                samples.append(
                    AnomalySample(
                        image_path=image_path,
                        label=0,
                        class_name=class_name,
                        split=source_split,
                        defect_type="good",
                    )
                )

        if normal_only:
            continue
        if test_root.exists():
            for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir() and path.name != "good"):
                for image_path in sorted(path for path in defect_dir.iterdir() if _is_image(path)):
                    samples.append(
                        AnomalySample(
                            image_path=image_path,
                            label=1,
                            class_name=class_name,
                            mask_path=_infer_mask_path(image_path, class_root, defect_dir.name),
                            split="test",
                            defect_type=defect_dir.name,
                        )
                    )
    return samples


def _read_json_samples(paths: Iterable[str | Path], label: int, data_root: Path) -> List[AnomalySample]:
    samples: List[AnomalySample] = []
    for json_path in paths:
        with open(json_path, "r", encoding="utf-8") as handle:
            raw_samples = json.load(handle)
        for raw in raw_samples:
            image_path = Path(raw["image_path"])
            if not image_path.exists():
                image_path = data_root / image_path
            class_name = raw.get("class_name") or raw.get("type") or image_path.parents[2].name
            mask_value = raw.get("mask_path") or raw.get("label_path")
            mask_path = Path(mask_value) if mask_value else None
            if mask_path is not None and not mask_path.exists():
                mask_path = data_root / mask_path
            if label == 1 and mask_path is None:
                class_root = image_path
                for parent in image_path.parents:
                    if parent.name == "test":
                        class_root = parent.parent
                        break
                mask_path = _infer_mask_path(image_path, class_root, image_path.parent.name)
            samples.append(
                AnomalySample(
                    image_path=image_path,
                    label=label,
                    class_name=str(class_name),
                    mask_path=mask_path,
                    split=raw.get("split", "json"),
                    defect_type=raw.get("defect_type", image_path.parent.name),
                )
            )
    return samples


class InCTRLv2DirectoryDataset(torch.utils.data.Dataset):
    """MVTec-format InCTRLv2 dataset with few-shot normal prompts."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        shots: int,
        input_size: int = 240,
        seed: int = 0,
        normal_only: bool = False,
        normal_json_paths: Optional[Iterable[str | Path]] = None,
        outlier_json_paths: Optional[Iterable[str | Path]] = None,
        return_normal_keys: bool = False,
    ):
        self.dataset_root = Path(dataset_root).expanduser()
        self.split = split
        self.shots = int(shots)
        self.input_size = int(input_size)
        self.seed = int(seed)
        self.normal_only = normal_only
        self.return_normal_keys = return_normal_keys
        self.image_transform = build_image_transform(self.input_size)
        self.mask_transform = build_mask_transform(self.input_size)

        if normal_json_paths:
            normal_samples = _read_json_samples(normal_json_paths, label=0, data_root=self.dataset_root)
            outlier_samples = [] if normal_only else _read_json_samples(outlier_json_paths or [], label=1, data_root=self.dataset_root)
            self.samples = normal_samples + outlier_samples
        else:
            self.samples = _scan_directory_samples(self.dataset_root, split=split, normal_only=normal_only)

        if not self.samples:
            raise ValueError(f"No samples found under {self.dataset_root} for split={split}")

        self.prompt_pool = self._build_prompt_pool()
        self.labels = [sample.label for sample in self.samples]

    def _build_prompt_pool(self) -> dict[str, List[AnomalySample]]:
        prompt_samples = _scan_directory_samples(self.dataset_root, split="train", normal_only=True)
        if not prompt_samples:
            prompt_samples = [sample for sample in self.samples if sample.label == 0]
        prompt_pool: dict[str, List[AnomalySample]] = {}
        for sample in prompt_samples:
            prompt_pool.setdefault(sample.class_name, []).append(sample)
        return prompt_pool

    def _select_prompts(self, sample: AnomalySample, index: int) -> List[AnomalySample]:
        candidates = list(self.prompt_pool.get(sample.class_name, []))
        if not candidates:
            candidates = [candidate for candidate in self.samples if candidate.label == 0]
        if not candidates:
            raise ValueError(f"No normal prompt candidates found for class {sample.class_name}")
        candidates_without_self = [candidate for candidate in candidates if candidate.image_path != sample.image_path]
        if candidates_without_self:
            candidates = candidates_without_self
        rng = random.Random(self.seed + index)
        if len(candidates) >= self.shots:
            return rng.sample(candidates, self.shots)
        return [rng.choice(candidates) for _ in range(self.shots)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        query_image = self.image_transform(_load_image(sample.image_path))
        mask = _load_mask(sample.mask_path, self.input_size, self.mask_transform)
        prompt_images = torch.stack(
            [self.image_transform(_load_image(prompt.image_path)) for prompt in self._select_prompts(sample, index)],
            dim=0,
        )
        label = torch.tensor(sample.label, dtype=torch.long)
        item = {
            "query_image": query_image,
            "prompt_images": prompt_images,
            "label": label,
            "mask": mask,
            "class_name": sample.class_name,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path) if sample.mask_path is not None else "",
        }
        if self.return_normal_keys:
            item.update(
                {
                    "normal_image": query_image,
                    "normal_mask": mask,
                }
            )
        return item


class InCTRLv2NormalDataset(InCTRLv2DirectoryDataset):
    def __init__(self, *args, **kwargs):
        kwargs["normal_only"] = True
        kwargs["return_normal_keys"] = True
        super().__init__(*args, **kwargs)


class BalancedBatchSampler(Sampler[List[int]]):
    """Half-normal, half-anomaly batches for DASL training."""

    def __init__(self, labels: List[int], batch_size: int, steps_per_epoch: int, seed: int = 0):
        self.normal_indices = [idx for idx, label in enumerate(labels) if int(label) == 0]
        self.outlier_indices = [idx for idx, label in enumerate(labels) if int(label) == 1]
        if not self.normal_indices or not self.outlier_indices:
            raise ValueError("BalancedBatchSampler requires both normal and abnormal samples")
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed)
        normal_cycle = cycle(rng.sample(self.normal_indices, len(self.normal_indices)))
        outlier_cycle = cycle(rng.sample(self.outlier_indices, len(self.outlier_indices)))
        normal_count = self.batch_size // 2
        outlier_count = self.batch_size - normal_count
        for _ in range(self.steps_per_epoch):
            batch = [next(normal_cycle) for _ in range(normal_count)]
            batch.extend(next(outlier_cycle) for _ in range(outlier_count))
            rng.shuffle(batch)
            yield batch
