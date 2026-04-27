from typing import Optional

import torch
from torch import nn


class ResidualMLPAdapter(nn.Module):
    def __init__(self, dim: int, reduction: int = 4, zero_init: bool = True):
        super().__init__()
        hidden = max(dim // reduction, 1)
        self.down = nn.Linear(dim, hidden)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(hidden, dim)
        self.scale = nn.Parameter(torch.tensor(0.1))

        if zero_init:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.up(self.act(self.down(x)))


class VisualAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        reduction: int = 4,
        zero_init: bool = True,
        local_dim: Optional[int] = None,
    ):
        super().__init__()
        local_dim = dim if local_dim is None else local_dim
        self.global_adapter = ResidualMLPAdapter(dim, reduction, zero_init)
        self.local_adapter = ResidualMLPAdapter(local_dim, reduction, zero_init)

    def forward(
        self,
        global_feat: torch.Tensor,
        patch_feats: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        adapted_global = self.global_adapter(global_feat)
        adapted_patches = [self.local_adapter(feat) for feat in patch_feats]
        return adapted_global, adapted_patches
