# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Visual Adapter for AdaptCLIP-style visual feature adaptation."""

import torch
from torch import nn


VALID_VISUAL_ADAPTER_MODES = ("global_only", "local_only", "global_local")


def normalize_visual_adapter_mode(mode: str) -> str:
    aliases = {
        "both": "global_local",
        "global+local": "global_local",
        "global-local": "global_local",
        "full": "global_local",
    }
    normalized = aliases.get(str(mode).lower(), str(mode).lower())
    if normalized not in VALID_VISUAL_ADAPTER_MODES:
        raise ValueError(
            f"Unsupported visual adapter mode: {mode}. "
            f"Expected one of {VALID_VISUAL_ADAPTER_MODES}."
        )
    return normalized


class ResMLP(nn.Module):
    """Residual MLP with bottleneck structure.

    Reduces input dimension by reduction ratio, applies ReLU, then projects back.
    Residual connection preserves base representation capacity.
    """

    def __init__(self, c_in, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.LayerNorm(c_in // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
        )

    def forward(self, x):
        if x.dim() == 2:
            out = self.fc(x)
            return x + out
        else:
            batch_size, seq_len, feature_dim = x.shape
            x_flat = x.reshape(batch_size * seq_len, feature_dim)
            out_flat = self.fc(x_flat)
            out = out_flat.reshape(batch_size, seq_len, feature_dim)
            return x + out


class VisualAdapter(nn.Module):
    """Visual Adapter for adapting CLIP visual features.

    Contains separate adapters for global image features and local patch features.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        global_input_dim: int,
        local_input_dim: int,
        reduction: int = 4,
        zero_init: bool = True,
        mode: str = "global_local",
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.global_input_dim = global_input_dim
        self.local_input_dim = local_input_dim
        self.reduction = reduction
        self.zero_init = zero_init
        self.mode = normalize_visual_adapter_mode(mode)

        self.global_adapter = ResMLP(global_input_dim, reduction=reduction)
        self.local_adapter = ResMLP(local_input_dim, reduction=reduction)

        if zero_init:
            self._zero_init_last_layer()

        self._freeze_disabled_branches()

    @property
    def use_global(self) -> bool:
        return self.mode in ("global_only", "global_local")

    @property
    def use_local(self) -> bool:
        return self.mode in ("local_only", "global_local")

    def _zero_init_last_layer(self):
        """Zero-initialize the last linear layer for identity-like residual."""
        self.global_adapter.fc[-1].weight.data.zero_()
        self.local_adapter.fc[-1].weight.data.zero_()

    def _freeze_disabled_branches(self):
        """Keep inactive ablation branches out of optimization."""
        if not self.use_global:
            for parameter in self.global_adapter.parameters():
                parameter.requires_grad = False
        if not self.use_local:
            for parameter in self.local_adapter.parameters():
                parameter.requires_grad = False

    def adapt_global(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global adapter to token features.

        Args:
            x: Tensor of shape [B, C] or [B, S, C] (2D or 3D)

        Returns:
            Adapted tensor with same shape as input
        """
        if not self.use_global:
            return x
        return self.global_adapter(x)

    def adapt_local(self, patch_tokens) -> list:
        """Apply local adapter to patch features from multiple layers.

        Args:
            patch_tokens: List of tensors, each [B, N, C] where N = num_patches

        Returns:
            List of adapted tensors with same shapes as input
        """
        if not self.use_local:
            return patch_tokens
        adapted = []
        for patch_feat in patch_tokens:
            adapted.append(self.local_adapter(patch_feat))
        return adapted
