from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageAdapter(nn.Module):
    """InCTRL image-level adapter psi."""

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, dim // reduction)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualScoreHead(nn.Module):
    """Small image-level residual scorer eta with sigmoid output."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.LayerNorm(64),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ImageResidualLearner(nn.Module):
    """Image-level in-context residual learning from InCTRL."""

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.adapter = ImageAdapter(dim, reduction)
        self.score_head = ResidualScoreHead(dim)

    def forward(self, query_cls: torch.Tensor, prompt_cls: torch.Tensor) -> dict:
        if prompt_cls.dim() != 3:
            raise ValueError(f"prompt_cls must be [B,K,D], got {tuple(prompt_cls.shape)}")
        query_adapted = self.adapter(query_cls)
        prompt_adapted = self.adapter(prompt_cls)
        prompt_proto = prompt_adapted.mean(dim=1)
        residual = prompt_proto - query_adapted
        score = self.score_head(residual)
        return {
            "image_residual_feature": residual,
            "s_I": score,
        }


def compute_patch_residual(
    query_patch_tokens: List[torch.Tensor],
    prompt_patch_tokens: List[torch.Tensor],
    residual_scale: str = "half",
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """Multi-layer patch nearest-neighbor residual map.

    Args:
        query_patch_tokens: list of [B,N,D] tensors.
        prompt_patch_tokens: list of [B,K*N,D] tensors.
        residual_scale: "half" uses 0.5 * (1 - cosine), matching original InCTRL.
    """
    if len(query_patch_tokens) != len(prompt_patch_tokens):
        raise ValueError("query and prompt patch token layer counts must match")

    layer_maps = []
    for query_layer, prompt_layer in zip(query_patch_tokens, prompt_patch_tokens):
        query_norm = F.normalize(query_layer, dim=-1)
        prompt_norm = F.normalize(prompt_layer, dim=-1)
        similarity = torch.bmm(query_norm, prompt_norm.transpose(1, 2))
        nearest_similarity = similarity.max(dim=-1).values
        residual = 1.0 - nearest_similarity
        if residual_scale == "half":
            residual = 0.5 * residual
        layer_maps.append(residual.clamp(0.0, 1.0))

    residual_map = torch.stack(layer_maps, dim=0).mean(dim=0)
    patch_score = residual_map.max(dim=-1).values
    return residual_map, patch_score, layer_maps


def flat_to_spatial(flat_map: torch.Tensor) -> torch.Tensor:
    batch_size, num_patches = flat_map.shape
    grid = int(num_patches ** 0.5)
    if grid * grid != num_patches:
        raise ValueError(f"flat map patch count must be square, got {num_patches}")
    return flat_map.view(batch_size, 1, grid, grid)


def upsample_flat_map(flat_map: torch.Tensor, size: int | tuple[int, int]) -> torch.Tensor:
    if isinstance(size, int):
        size = (size, size)
    return F.interpolate(flat_to_spatial(flat_map), size=size, mode="bilinear", align_corners=False)
