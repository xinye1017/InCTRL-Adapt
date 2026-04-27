import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .model import _build_vision_tower_Mul, _build_text_tower
from .object_agnostic_text import ObjectAgnosticTextBranch
from .prompt_query_head import PromptQuerySegHead
from .visual_adapter import VisualAdapter


def _score_to_logit(score: torch.Tensor) -> torch.Tensor:
    score = score.clamp(min=1e-6, max=1.0 - 1e-6)
    return torch.log(score / (1.0 - score))


def _fuse_scores(
    image_logit: torch.Tensor,
    patch_logit: torch.Tensor,
    pqa_logit: torch.Tensor,
    text_logit: torch.Tensor,
    weights: tuple[float, float, float, float],
) -> torch.Tensor:
    image_w, patch_w, pqa_w, text_w = weights
    return (
        image_w * image_logit
        + patch_w * patch_logit
        + pqa_w * pqa_logit
        + text_w * text_logit
    )


def _fuse_maps(
    residual_map: torch.Tensor,
    pqa_map: torch.Tensor,
    text_map: torch.Tensor,
    weights: tuple[float, float, float],
) -> torch.Tensor:
    residual_w, pqa_w, text_w = weights
    return residual_w * residual_map + pqa_w * pqa_map + text_w * text_map


def _get_vision_width(vision_cfg, fallback: int) -> int:
    if isinstance(vision_cfg, dict):
        return int(vision_cfg.get("width", fallback))
    return int(getattr(vision_cfg, "width", fallback))


class ImageResidualHead(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        query_global: torch.Tensor,
        prompt_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_proto = prompt_global.mean(dim=1)
        residual = prompt_proto - query_global
        logit = self.net(residual).squeeze(-1)
        return torch.sigmoid(logit), logit


class InCTRLPQA(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        vision_cfg,
        text_cfg,
        quick_gelu=False,
        cast_dtype=None,
        output_dict=False,
    ):
        super().__init__()
        self.args = args
        self.output_dict = output_dict
        self.image_size = int(getattr(args, "image_size", 240))
        self.patch_layers = list(getattr(args.PQA, "PATCH_LAYERS", [7, 9, 11]))
        self.use_visual_adapter = bool(getattr(args.VISUAL_ADAPTER, "ENABLE", True))
        self.use_text_branch = bool(getattr(args.TEXT_BRANCH, "ENABLE", True))
        self.use_pqa = bool(getattr(args.PQA, "ENABLE", True))
        hidden_dim = int(getattr(args.PQA, "HIDDEN_DIM", 128))
        patch_dim = _get_vision_width(vision_cfg, fallback=embed_dim)

        self.visual = _build_vision_tower_Mul(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer("attn_mask", text.attn_mask, persistent=False)

        for param in self.visual.parameters():
            param.requires_grad = False
        for param in text.parameters():
            param.requires_grad = False

        reduction = int(getattr(args.VISUAL_ADAPTER, "REDUCTION", 4))
        zero_init = bool(getattr(args.VISUAL_ADAPTER, "ZERO_INIT", True))
        self.visual_adapter = VisualAdapter(
            embed_dim,
            reduction=reduction,
            zero_init=zero_init,
            local_dim=patch_dim,
        )
        self.image_head = ImageResidualHead(embed_dim, hidden_dim=hidden_dim)
        self.pqa_head = PromptQuerySegHead(
            dim=patch_dim,
            hidden_dim=hidden_dim,
            image_size=self.image_size,
            topk=int(getattr(args.PQA, "GLOBAL_TOPK", 10)),
            beta=float(getattr(args.PQA, "CONTEXT_BETA", 1.0)),
        )
        self.patch_text_projection = (
            nn.Identity() if patch_dim == embed_dim else nn.Linear(patch_dim, embed_dim)
        )
        self.text_branch = ObjectAgnosticTextBranch(
            templates=list(getattr(args.TEXT_BRANCH, "TEMPLATES")),
            logit_scale=float(getattr(args.TEXT_BRANCH, "LOGIT_SCALE", 100.0)),
        )

    def _normalize_image_features(self, features):
        global_feat, patch_levels, fp = features
        normalized_patches = [F.normalize(level, dim=-1) for level in patch_levels]
        normalized_fp = F.normalize(fp, dim=-1) if torch.is_tensor(fp) else fp
        return F.normalize(global_feat, dim=-1), normalized_patches, normalized_fp

    def encode_image(self, image: torch.Tensor, normalize: bool = False):
        features = self.visual.forward(image, self.patch_layers)
        return self._normalize_image_features(features) if normalize else features

    def encode_text(self, text: torch.Tensor, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def _flatten_prompt_tokens(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        batch, shot, patches, dim = prompt_tokens.shape
        return prompt_tokens.reshape(batch, shot * patches, dim)

    def _compute_patch_residual(
        self,
        query_tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
    ) -> torch.Tensor:
        query_norm = F.normalize(query_tokens, dim=-1)
        prompt_norm = F.normalize(prompt_tokens, dim=-1)
        max_cosine = torch.matmul(query_norm, prompt_norm.transpose(-1, -2)).max(dim=-1).values
        return 0.5 * (1.0 - max_cosine)

    def _upsample_patch_map(self, patch_map: torch.Tensor) -> torch.Tensor:
        batch, patches = patch_map.shape
        grid = int(math.sqrt(patches))
        if grid * grid != patches:
            raise ValueError(f"Patch count {patches} is not a square grid.")
        patch_map = patch_map.reshape(batch, 1, grid, grid)
        return F.interpolate(
            patch_map,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

    def _zero_text_outputs(self, query_global: torch.Tensor, patch_tokens: torch.Tensor):
        text_logit = query_global.new_zeros(query_global.shape[0])
        text_map = self._upsample_patch_map(patch_tokens.new_zeros(patch_tokens.shape[0], patch_tokens.shape[1]))
        return {
            "text_logit": text_logit,
            "text_score": torch.sigmoid(text_logit),
            "text_map": text_map,
            "text_prototypes": None,
        }

    def _zero_pqa_outputs(self, query_tokens: torch.Tensor):
        pqa_logit = query_tokens.new_zeros(query_tokens.shape[0])
        pqa_patch_map = query_tokens.new_zeros(query_tokens.shape[0], query_tokens.shape[1])
        return {
            "pqa_seg_logits": self._upsample_patch_map(pqa_patch_map),
            "pqa_logit": pqa_logit,
            "pqa_score": torch.sigmoid(pqa_logit),
            "pqa_patch_map": pqa_patch_map,
        }

    def _pqa_map_from_logits(self, pqa_seg_logits: torch.Tensor) -> torch.Tensor:
        if not self.use_pqa:
            return torch.zeros_like(pqa_seg_logits)
        return torch.sigmoid(pqa_seg_logits)

    def _legacy_inputs_to_new(self, image, normal_list):
        if torch.is_tensor(image):
            if image.dim() != 5:
                raise ValueError("Legacy tensor image input must have shape [shot+1, B, C, H, W].")
            query_image = image[0]
            prompt_images = image[1:].permute(1, 0, 2, 3, 4)
            return query_image, prompt_images

        query_image = image[0]
        batch = query_image.shape[0]
        if normal_list is None:
            prompt_images = torch.stack(image[1:], dim=1)
        else:
            normal_images = torch.stack(normal_list) if isinstance(normal_list, (list, tuple)) else normal_list
            normal_images = normal_images.to(device=query_image.device, dtype=query_image.dtype)
            if normal_images.dim() == 4:
                prompt_images = normal_images.unsqueeze(0).expand(batch, -1, -1, -1, -1)
            elif normal_images.dim() == 5 and normal_images.shape[0] == batch:
                prompt_images = normal_images
            elif normal_images.dim() == 5 and normal_images.shape[1] == batch:
                prompt_images = normal_images.permute(1, 0, 2, 3, 4)
            else:
                raise ValueError(
                    "normal_list must have shape [shot, C, H, W], [B, shot, C, H, W], "
                    "or [shot, B, C, H, W]."
                )
        return query_image, prompt_images

    def forward(
        self,
        tokenizer=None,
        image: Optional[torch.Tensor] = None,
        text: Optional[list] = None,
        normal_list=None,
        query_image: Optional[torch.Tensor] = None,
        prompt_images: Optional[torch.Tensor] = None,
        obj_types: Optional[list] = None,
        return_aux: bool = False,
        return_dict: bool = False,
    ):
        legacy_tuple = query_image is None
        if legacy_tuple:
            if image is None:
                raise ValueError("Either query_image/prompt_images or legacy image input must be provided.")
            query_image, prompt_images = self._legacy_inputs_to_new(image, normal_list)
            obj_types = text

        if prompt_images is None:
            raise ValueError("prompt_images must be provided.")
        if self.use_text_branch and tokenizer is None:
            raise ValueError("tokenizer is required for the object-agnostic text branch.")

        del obj_types

        batch, shot, channels, height, width = prompt_images.shape
        prompt_images = prompt_images.to(device=query_image.device, dtype=query_image.dtype)
        prompt_flat = prompt_images.reshape(batch * shot, channels, height, width)

        query_global, query_patch_levels, _ = self.encode_image(query_image, normalize=False)
        prompt_global_flat, prompt_patch_levels_flat, _ = self.encode_image(prompt_flat, normalize=False)
        prompt_global = prompt_global_flat.reshape(batch, shot, -1)

        query_patch_levels = [level[:, 1:, :] for level in query_patch_levels]
        prompt_patch_levels = [
            level[:, 1:, :].reshape(batch, shot, level.shape[1] - 1, level.shape[-1])
            for level in prompt_patch_levels_flat
        ]

        if self.use_visual_adapter:
            query_global, query_patch_levels = self.visual_adapter(query_global, query_patch_levels)
            prompt_global = self.visual_adapter.global_adapter(prompt_global)
            prompt_patch_levels = [
                self.visual_adapter.local_adapter(
                    level.reshape(batch * shot, level.shape[2], level.shape[3])
                ).reshape(batch, shot, level.shape[2], level.shape[3])
                for level in prompt_patch_levels
            ]

        image_score, image_logit = self.image_head(query_global, prompt_global)

        residual_maps = []
        pqa_maps = []
        pqa_logits = []
        pqa_seg_logits_per_layer = []
        for query_tokens, prompt_tokens in zip(query_patch_levels, prompt_patch_levels):
            prompt_flat_tokens = self._flatten_prompt_tokens(prompt_tokens)
            residual_map = self._compute_patch_residual(query_tokens, prompt_flat_tokens)
            pqa_out = (
                self.pqa_head(query_tokens, prompt_flat_tokens)
                if self.use_pqa
                else self._zero_pqa_outputs(query_tokens)
            )
            residual_maps.append(residual_map)
            pqa_maps.append(pqa_out["pqa_patch_map"])
            pqa_logits.append(pqa_out["pqa_logit"])
            pqa_seg_logits_per_layer.append(pqa_out["pqa_seg_logits"])

        patch_residual_map = torch.stack(residual_maps, dim=0).mean(dim=0)
        patch_score = patch_residual_map.max(dim=-1).values
        patch_logit = _score_to_logit(patch_score)
        pqa_patch_map = torch.stack(pqa_maps, dim=0).mean(dim=0)
        pqa_logit = torch.stack(pqa_logits, dim=0).mean(dim=0)
        pqa_score = torch.sigmoid(pqa_logit)
        pqa_seg_logits = torch.stack(pqa_seg_logits_per_layer, dim=0).mean(dim=0)

        text_patch_feat = self.patch_text_projection(query_patch_levels[-1])
        if self.use_text_branch:
            text_out = self.text_branch(
                encode_text=self.encode_text,
                tokenizer=tokenizer,
                global_feat=query_global,
                patch_feat=text_patch_feat,
                image_size=self.image_size,
            )
        else:
            text_out = self._zero_text_outputs(query_global, text_patch_feat)
        text_logit = text_out["text_logit"]
        text_score = text_out["text_score"]
        text_map = text_out["text_map"]

        final_logit = _fuse_scores(
            image_logit,
            patch_logit,
            pqa_logit,
            text_logit,
            weights=(
                float(self.args.FUSION.IMAGE_WEIGHT),
                float(self.args.FUSION.PATCH_WEIGHT),
                float(self.args.FUSION.PQA_WEIGHT),
                float(self.args.FUSION.TEXT_WEIGHT),
            ),
        )
        final_score = torch.sigmoid(final_logit)

        residual_map_up = self._upsample_patch_map(patch_residual_map)
        pqa_map = self._pqa_map_from_logits(pqa_seg_logits)
        final_map = _fuse_maps(
            residual_map_up,
            pqa_map,
            text_map,
            weights=(
                float(self.args.FUSION.MAP_RES_WEIGHT),
                float(self.args.FUSION.MAP_PQA_WEIGHT),
                float(self.args.FUSION.MAP_TEXT_WEIGHT),
            ),
        )

        outputs = {
            "final_score": final_score,
            "final_logit": final_logit,
            "image_score": image_score,
            "image_logit": image_logit,
            "patch_score": patch_score,
            "patch_logit": patch_logit,
            "patch_residual_map": patch_residual_map,
            "pqa_score": pqa_score,
            "pqa_logit": pqa_logit,
            "pqa_patch_map": pqa_patch_map,
            "pqa_seg_logits": pqa_seg_logits,
            "text_score": text_score,
            "text_logit": text_logit,
            "text_map": text_map,
            "final_map": final_map,
        }
        if return_aux:
            outputs["aux"] = {
                "residual_maps": residual_maps,
                "pqa_maps": pqa_maps,
                "pqa_logits": pqa_logits,
            }
        if legacy_tuple and not return_dict:
            return outputs["final_score"], outputs["image_score"]
        return outputs
