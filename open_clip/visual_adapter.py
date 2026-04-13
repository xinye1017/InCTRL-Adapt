# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Visual Adapter for AdaptCLIP-style visual feature adaptation."""

import torch
from torch import nn


class ResMLP(nn.Module):
    """Residual MLP with bottleneck structure.

    Reduces input dimension by reduction ratio, applies ReLU, then projects back.
    Residual connection preserves base representation capacity.
    """

    def __init__(self, c_in, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.BatchNorm1d(c_in // reduction),
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
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.global_input_dim = global_input_dim
        self.local_input_dim = local_input_dim
        self.reduction = reduction
        self.zero_init = zero_init

        self.global_adapter = ResMLP(global_input_dim, reduction=reduction)
        self.local_adapter = ResMLP(local_input_dim, reduction=reduction)

        if zero_init:
            self._zero_init_last_layer()

    def _zero_init_last_layer(self):
        """Zero-initialize the last linear layer for identity-like residual."""
        self.global_adapter.fc[-1].weight.data.zero_()
        self.local_adapter.fc[-1].weight.data.zero_()

    def adapt_global(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global adapter to token features.

        Args:
            x: Tensor of shape [B, C] or [B, S, C] (2D or 3D)

        Returns:
            Adapted tensor with same shape as input
        """
        return self.global_adapter(x)

    def adapt_local(self, patch_tokens) -> list:
        """Apply local adapter to patch features from multiple layers.

        Args:
            patch_tokens: List of tensors, each [B, N, C] where N = num_patches

        Returns:
            List of adapted tensors with same shapes as input
        """
        adapted = []
        for patch_feat in patch_tokens:
            adapted.append(self.local_adapter(patch_feat))
        return adapted
