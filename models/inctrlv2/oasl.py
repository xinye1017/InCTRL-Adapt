from typing import Iterable, List

import torch
import torch.nn as nn

from .dasl import PatchTextAdapter


class OASLBranch(nn.Module):
    """One-class Anomaly Score Learning branch with independent patch adapters."""

    def __init__(self, patch_dim: int, text_dim: int, selected_layers: Iterable[int]):
        super().__init__()
        self.selected_layers = list(selected_layers)
        self.patch_adapters = nn.ModuleList(
            PatchTextAdapter(patch_dim=patch_dim, text_dim=text_dim) for _ in self.selected_layers
        )

    def semantic_maps(
        self,
        patch_tokens: List[torch.Tensor],
        normal_proto: torch.Tensor,
        abnormal_proto: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(patch_tokens) != len(self.patch_adapters):
            raise ValueError("patch_tokens must match selected_layers")

        normal_maps = []
        abnormal_maps = []
        for layer_tokens, adapter in zip(patch_tokens, self.patch_adapters):
            projected = adapter(layer_tokens)
            normal_logit = torch.einsum("bnd,bd->bn", projected, normal_proto)
            abnormal_logit = torch.einsum("bnd,bd->bn", projected, abnormal_proto)
            probs = torch.stack([normal_logit, abnormal_logit], dim=-1).softmax(dim=-1)
            normal_maps.append(probs[..., 0])
            abnormal_maps.append(probs[..., 1])

        return torch.stack(normal_maps, dim=0).mean(dim=0), torch.stack(abnormal_maps, dim=0).mean(dim=0)

    def forward(
        self,
        patch_tokens: List[torch.Tensor],
        residual_map: torch.Tensor,
        normal_proto: torch.Tensor,
        abnormal_proto: torch.Tensor,
    ) -> dict:
        normal_map, abnormal_map = self.semantic_maps(patch_tokens, normal_proto, abnormal_proto)
        pixel_map = 0.5 * (residual_map + abnormal_map)
        return {
            "S_hat_n": normal_map.clamp(0.0, 1.0),
            "S_hat_a": abnormal_map.clamp(0.0, 1.0),
            "pixel_map_oasl": pixel_map.clamp(0.0, 1.0),
        }
