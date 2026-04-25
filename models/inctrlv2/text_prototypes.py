from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

import torch
import torch.nn.functional as F

from open_clip.model import get_texts


class TextPrototypeBuilder:
    """Build and cache WinCLIP/InCTRL normal and abnormal text prototypes."""

    def __init__(self, tokenizer: Callable, encode_text: Callable):
        self.tokenizer = tokenizer
        self.encode_text = encode_text
        self._cache: Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def normalize_class_name(class_name: str) -> str:
        return str(class_name).replace("_", " ").strip()

    @torch.no_grad()
    def build_one(self, class_name: str, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_name = self.normalize_class_name(class_name)
        cache_key = (normalized_name, str(device), str(dtype))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        normal_texts, abnormal_texts = get_texts(normalized_name)
        normal_tokens = self.tokenizer(normal_texts).to(device)
        abnormal_tokens = self.tokenizer(abnormal_texts).to(device)
        normal_features = self.encode_text(normal_tokens, normalize=True).to(dtype=dtype)
        abnormal_features = self.encode_text(abnormal_tokens, normalize=True).to(dtype=dtype)
        normal_proto = F.normalize(normal_features.mean(dim=0), dim=-1)
        abnormal_proto = F.normalize(abnormal_features.mean(dim=0), dim=-1)
        self._cache[cache_key] = (normal_proto, abnormal_proto)
        return normal_proto, abnormal_proto

    def build(self, class_names: Iterable[str] | str, device: torch.device, dtype: torch.dtype) -> dict:
        if isinstance(class_names, str):
            class_names = [class_names]
        normal_protos = []
        abnormal_protos = []
        for class_name in class_names:
            normal_proto, abnormal_proto = self.build_one(class_name, device, dtype)
            normal_protos.append(normal_proto)
            abnormal_protos.append(abnormal_proto)
        return {
            "normal_proto": torch.stack(normal_protos, dim=0),
            "abnormal_proto": torch.stack(abnormal_protos, dim=0),
        }
