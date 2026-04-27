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


class PromptQuerySegHead(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        image_size: int = 240,
        topk: int = 10,
        beta: float = 1.0,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.topk = int(topk)
        self.beta = float(beta)
        self.local_head = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.global_head = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_tokens: torch.Tensor, prompt_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        aligned, indices, residual_map = align_prompt_to_query(query_tokens, prompt_tokens)
        context_tokens = query_tokens + self.beta * torch.abs(query_tokens - aligned)

        batch, patches, dim = context_tokens.shape
        grid = int(math.sqrt(patches))
        if grid * grid != patches:
            raise ValueError(f"Patch count {patches} is not a square grid.")

        context_map = context_tokens.transpose(1, 2).reshape(batch, dim, grid, grid)
        low_res_logits = self.local_head(context_map)
        pqa_seg_logits = F.interpolate(
            low_res_logits,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        pqa_patch_map = torch.sigmoid(low_res_logits.flatten(2).squeeze(1))

        topk = min(self.topk, patches)
        avg_pool = context_tokens.mean(dim=1)
        topk_index = residual_map.topk(topk, dim=-1).indices.unsqueeze(-1).expand(-1, -1, dim)
        topk_tokens = torch.gather(context_tokens, dim=1, index=topk_index).mean(dim=1)
        pooled = 0.5 * avg_pool + 0.5 * topk_tokens
        pqa_logit = self.global_head(pooled).squeeze(-1)

        return {
            "pqa_seg_logits": pqa_seg_logits,
            "pqa_logit": pqa_logit,
            "pqa_score": torch.sigmoid(pqa_logit),
            "pqa_patch_map": pqa_patch_map,
            "inctrl_patch_map": residual_map,
            "context_tokens": context_tokens,
            "aligned_prompt_tokens": aligned,
            "aligned_indices": indices,
        }
