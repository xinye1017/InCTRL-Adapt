import math

import torch
from torch import nn
import torch.nn.functional as F


def align_prompt_to_query(
    query_tokens: torch.Tensor,
    prompt_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_norm = F.normalize(query_tokens, dim=-1)
    prompt_norm = F.normalize(prompt_tokens, dim=-1)

    sim = torch.matmul(query_norm, prompt_norm.transpose(-1, -2))
    max_sim, indices = sim.max(dim=-1)
    aligned = torch.gather(
        prompt_tokens,
        dim=1,
        index=indices.unsqueeze(-1).expand(-1, -1, prompt_tokens.shape[-1]),
    )
    residual_map = 0.5 * (1.0 - max_sim)
    return aligned, indices, residual_map


class PQAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        image_size: int = 240,
        num_layers: int = 3,
        topk: int = 10,
        beta: float = 1.0,
        enable_seg_head: bool = True,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.num_layers = int(num_layers)
        self.topk = int(topk)
        self.beta = float(beta)
        self.enable_seg_head = enable_seg_head

        # Per-layer BatchNorm on context fusion features (AdaptCLIP's sharebn).
        self.sharebn = nn.ModuleList([nn.BatchNorm2d(dim) for _ in range(num_layers)])

        # Local decoder: ConvTranspose2d learned upsample + 2-class output.
        # Two ConvTranspose2d(2x) stages: grid -> 2*grid -> 4*grid,
        # then F.interpolate for remainder to image_size.
        self.local_head = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 2, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(hidden_dim // 2, 2, kernel_size=1, stride=1, padding=0),
        )

        # Global head: 3-layer MLP with BN, 2-class output.
        self.global_head = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2, bias=False),
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
        layer_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        aligned, indices, residual_map = align_prompt_to_query(query_tokens, prompt_tokens)
        context_tokens = query_tokens + self.beta * torch.abs(query_tokens - aligned)

        batch, patches, dim = context_tokens.shape
        grid = int(math.sqrt(patches))
        if grid * grid != patches:
            raise ValueError(f"Patch count {patches} is not a square grid.")

        if self.enable_seg_head:
            # Reshape to spatial map and apply per-layer sharebn.
            context_map = context_tokens.transpose(1, 2).reshape(batch, dim, grid, grid)
            layer_idx_clamped = min(layer_idx, self.num_layers - 1)
            context_map = self.sharebn[layer_idx_clamped](context_map)

            # CNN decoder with learned upsample.
            low_res_logits = self.local_head(context_map)  # [B, 2, 4*grid, 4*grid]
            pqa_seg_logits = F.interpolate(
                low_res_logits,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            # Anomaly probability from 2-class softmax (channel 1 = anomaly).
            pqa_patch_probs = low_res_logits.softmax(dim=1)
            pqa_patch_map = pqa_patch_probs[:, 1, :, :].flatten(1)  # at upsampled grid res

            # Global head: pool context features → 2-class logit.
            topk = min(self.topk, patches)
            avg_pool = context_tokens.mean(dim=1)
            topk_index = residual_map.topk(topk, dim=-1).indices.unsqueeze(-1).expand(-1, -1, dim)
            topk_tokens = torch.gather(context_tokens, dim=1, index=topk_index).mean(dim=1)
            pooled = 0.5 * avg_pool + 0.5 * topk_tokens
            pqa_global_logits = self.global_head(pooled)  # [B, 2]
            # Binary logit: anomaly - normal (for BCE-style loss).
            pqa_logit = pqa_global_logits[:, 1] - pqa_global_logits[:, 0]
        else:
            pqa_seg_logits = query_tokens.new_zeros(batch, 2, self.image_size, self.image_size)
            pqa_global_logits = query_tokens.new_zeros(batch, 2)
            pqa_logit = query_tokens.new_zeros(batch)
            pqa_patch_map = residual_map

        return {
            "pqa_seg_logits": pqa_seg_logits,
            "pqa_global_logits": pqa_global_logits,
            "pqa_logit": pqa_logit,
            "pqa_score": torch.sigmoid(pqa_logit),
            "pqa_patch_map": pqa_patch_map,
            "inctrl_patch_map": residual_map,
            "context_tokens": context_tokens,
            "aligned_prompt_tokens": aligned,
            "aligned_indices": indices,
        }
