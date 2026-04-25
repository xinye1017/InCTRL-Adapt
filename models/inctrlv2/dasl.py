from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import fuse_image_score


class PatchTextAdapter(nn.Module):
    """Project CLIP visual patch tokens into text embedding space."""

    def __init__(self, patch_dim: int, text_dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, patch_dim // reduction)
        self.net = nn.Sequential(
            nn.Linear(patch_dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, text_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def semantic_image_score(query_cls: torch.Tensor, normal_proto: torch.Tensor, abnormal_proto: torch.Tensor) -> torch.Tensor:
    query = F.normalize(query_cls, dim=-1)
    normal_logit = (query * normal_proto).sum(dim=-1)
    abnormal_logit = (query * abnormal_proto).sum(dim=-1)
    return torch.stack([normal_logit, abnormal_logit], dim=-1).softmax(dim=-1)[:, 1]


class DASLBranch(nn.Module):
    """Discriminative Anomaly Score Learning branch."""

    def __init__(self, patch_dim: int, text_dim: int, selected_layers: Iterable[int]):
        super().__init__()
        self.selected_layers = list(selected_layers)
        self.patch_adapters = nn.ModuleList(
            PatchTextAdapter(patch_dim=patch_dim, text_dim=text_dim) for _ in self.selected_layers
        )

    def semantic_maps(
        self,
        query_patch_tokens: List[torch.Tensor],
        normal_proto: torch.Tensor,
        abnormal_proto: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(query_patch_tokens) != len(self.patch_adapters):
            raise ValueError("query_patch_tokens must match selected_layers")

        normal_maps = []
        abnormal_maps = []
        for patch_tokens, adapter in zip(query_patch_tokens, self.patch_adapters):
            projected = adapter(patch_tokens)
            normal_logit = torch.einsum("bnd,bd->bn", projected, normal_proto)
            abnormal_logit = torch.einsum("bnd,bd->bn", projected, abnormal_proto)
            probs = torch.stack([normal_logit, abnormal_logit], dim=-1).softmax(dim=-1)
            normal_maps.append(probs[..., 0])
            abnormal_maps.append(probs[..., 1])

        return torch.stack(normal_maps, dim=0).mean(dim=0), torch.stack(abnormal_maps, dim=0).mean(dim=0)

    def forward(
        self,
        query_cls: torch.Tensor,
        query_patch_tokens: List[torch.Tensor],
        residual_map: torch.Tensor,
        s_i: torch.Tensor,
        s_p: torch.Tensor,
        normal_proto: torch.Tensor,
        abnormal_proto: torch.Tensor,
        alpha: float = 0.5,
    ) -> dict:
        s_q = semantic_image_score(query_cls, normal_proto, abnormal_proto)
        s_n_map, s_a_map = self.semantic_maps(query_patch_tokens, normal_proto, abnormal_proto)
        pixel_map = 0.5 * (residual_map + s_a_map)
        image_score = fuse_image_score(s_i=s_i, s_q=s_q, s_p=s_p, alpha=alpha)
        return {
            "semantic_score": s_q,
            "image_score": image_score.clamp(0.0, 1.0),
            "S_n": s_n_map.clamp(0.0, 1.0),
            "S_a": s_a_map.clamp(0.0, 1.0),
            "pixel_map_dasl": pixel_map.clamp(0.0, 1.0),
        }
