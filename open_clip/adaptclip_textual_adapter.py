import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class BinaryPromptLearner(nn.Module):
    def __init__(
        self,
        ctx_dim: int,
        n_ctx: int = 12,
        normal_suffix: str = "normal object.",
        abnormal_suffix: str = "damaged object.",
        init_std: float = 0.02,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.n_ctx = int(n_ctx)
        self.ctx_dim = int(ctx_dim)
        self.normal_suffix = str(normal_suffix)
        self.abnormal_suffix = str(abnormal_suffix)

        self.ctx_pos = nn.Parameter(torch.empty(1, self.n_ctx, self.ctx_dim, dtype=dtype))
        self.ctx_neg = nn.Parameter(torch.empty(1, self.n_ctx, self.ctx_dim, dtype=dtype))
        nn.init.normal_(self.ctx_pos, std=float(init_std))
        nn.init.normal_(self.ctx_neg, std=float(init_std))

        self.register_buffer("token_prefix_pos", torch.empty(0), persistent=False)
        self.register_buffer("token_prefix_neg", torch.empty(0), persistent=False)
        self.register_buffer("token_suffix_pos", torch.empty(0), persistent=False)
        self.register_buffer("token_suffix_neg", torch.empty(0), persistent=False)
        self.register_buffer("tokenized_prompts_pos", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("tokenized_prompts_neg", torch.empty(0, dtype=torch.long), persistent=False)

    @property
    def is_initialized(self):
        return self.token_prefix_pos.numel() > 0

    def _tokenize(self, tokenizer, text: str):
        tokenized = tokenizer(text)
        if tokenized.dim() == 1:
            tokenized = tokenized.unsqueeze(0)
        return tokenized

    def initialize_prompt_buffers(self, token_embedding, tokenizer):
        prompt_prefix = " ".join(["X"] * self.n_ctx)
        prompt_pos = f"{prompt_prefix} {self.normal_suffix}"
        prompt_neg = f"{prompt_prefix} {self.abnormal_suffix}"
        tokenized_pos = self._tokenize(tokenizer, prompt_pos).to(token_embedding.weight.device)
        tokenized_neg = self._tokenize(tokenizer, prompt_neg).to(token_embedding.weight.device)
        dtype = self.ctx_pos.dtype

        with torch.no_grad():
            emb_pos = token_embedding(tokenized_pos).to(dtype)
            emb_neg = token_embedding(tokenized_neg).to(dtype)

        self.token_prefix_pos = emb_pos[:, :1, :]
        self.token_prefix_neg = emb_neg[:, :1, :]
        self.token_suffix_pos = emb_pos[:, 1 + self.n_ctx :, :]
        self.token_suffix_neg = emb_neg[:, 1 + self.n_ctx :, :]
        self.tokenized_prompts_pos = tokenized_pos
        self.tokenized_prompts_neg = tokenized_neg

    def forward(self, batch_size: int, device: torch.device, token_embedding=None, tokenizer=None):
        if not self.is_initialized:
            if token_embedding is None or tokenizer is None:
                raise ValueError("token_embedding and tokenizer are required before prompt buffers are initialized.")
            self.initialize_prompt_buffers(token_embedding, tokenizer)

        prefix_pos = self.token_prefix_pos.to(device).expand(batch_size, -1, -1)
        prefix_neg = self.token_prefix_neg.to(device).expand(batch_size, -1, -1)
        suffix_pos = self.token_suffix_pos.to(device).expand(batch_size, -1, -1)
        suffix_neg = self.token_suffix_neg.to(device).expand(batch_size, -1, -1)
        ctx_pos = self.ctx_pos.to(device).expand(batch_size, -1, -1)
        ctx_neg = self.ctx_neg.to(device).expand(batch_size, -1, -1)

        prompt_pos = torch.cat([prefix_pos, ctx_pos, suffix_pos], dim=1)
        prompt_neg = torch.cat([prefix_neg, ctx_neg, suffix_neg], dim=1)
        prompts = torch.cat([prompt_pos, prompt_neg], dim=0)

        tokenized_pos = self.tokenized_prompts_pos.to(device).expand(batch_size, -1)
        tokenized_neg = self.tokenized_prompts_neg.to(device).expand(batch_size, -1)
        tokenized_prompts = torch.cat([tokenized_pos, tokenized_neg], dim=0)
        return prompts, tokenized_prompts


class AdaptCLIPTextualAdapter(nn.Module):
    def __init__(
        self,
        ctx_dim: int,
        image_size: int = 240,
        n_ctx: int = 12,
        normal_suffix: str = "normal object.",
        abnormal_suffix: str = "damaged object.",
        init_std: float = 0.02,
        logit_scale: float = 100.0,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.logit_scale = float(logit_scale)
        self.prompt_learner = BinaryPromptLearner(
            ctx_dim=ctx_dim,
            n_ctx=n_ctx,
            normal_suffix=normal_suffix,
            abnormal_suffix=abnormal_suffix,
            init_std=init_std,
            dtype=dtype,
        )

    def build_text_features(self, encode_text_prompted, token_embedding, tokenizer, batch_size: int, device: torch.device):
        prompts, tokenized_prompts = self.prompt_learner(
            batch_size=batch_size,
            device=device,
            token_embedding=token_embedding,
            tokenizer=tokenizer,
        )
        text_features = encode_text_prompted(prompts, tokenized_prompts, normalize=True)
        normal_feat, abnormal_feat = torch.chunk(text_features, chunks=2, dim=0)
        return torch.stack([normal_feat, abnormal_feat], dim=1)

    def forward(self, encode_text_prompted, token_embedding, tokenizer, global_feat: torch.Tensor, patch_feat: torch.Tensor):
        batch, patches, _ = patch_feat.shape
        text_features = self.build_text_features(
            encode_text_prompted,
            token_embedding,
            tokenizer,
            batch,
            global_feat.device,
        )

        global_feat = F.normalize(global_feat, dim=-1)
        patch_feat = F.normalize(patch_feat, dim=-1)

        text_logits = self.logit_scale * torch.einsum("bd,bcd->bc", global_feat, text_features)
        text_logit = text_logits[:, 1] - text_logits[:, 0]
        text_score = text_logits.softmax(dim=-1)[:, 1]

        patch_text_logits = self.logit_scale * torch.einsum("bnd,bcd->bnc", patch_feat, text_features)
        patch_text_logit = patch_text_logits[..., 1] - patch_text_logits[..., 0]
        grid = int(math.sqrt(patches))
        if grid * grid != patches:
            raise ValueError(f"Patch count {patches} is not a square grid.")
        text_map_logits = patch_text_logit.reshape(batch, 1, grid, grid)
        text_map_logits = F.interpolate(
            text_map_logits,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        text_map = torch.sigmoid(text_map_logits)

        return {
            "text_features": text_features,
            "text_logits": text_logits,
            "text_logit": text_logit,
            "text_score": text_score,
            "patch_text_logits": patch_text_logits,
            "text_map_logits": text_map_logits,
            "text_map": text_map,
        }
