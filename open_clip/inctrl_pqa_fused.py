from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from .model import (
    CLIPTextCfg,
    CLIPVisionCfg,
    _build_text_tower,
    _build_vision_tower_Mul,
    get_texts,
)
from .tokenizer import tokenize


def _as_cfg(cfg_obj, cfg_type):
    if isinstance(cfg_obj, dict):
        return cfg_type(**cfg_obj)
    return cfg_obj


class ImageResidualHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PQAConvLocalHead(nn.Module):
    """Local segmentation head for prompt-query fusion features."""

    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        mid_dim = max(hidden_dim // 2, 1)
        self.net = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(hidden_dim, mid_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_dim, mid_dim, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(mid_dim, 2, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPAdapter(nn.Module):
    """Small-batch-safe MLP: Linear -> LayerNorm -> ReLU -> Linear."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PQAGlobalHead(nn.Module):
    """Global PQA head using learnable GAP/GMP fusion for image-level scoring."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.mlp_adapter = MLPAdapter(feature_dim, hidden_dim)
        self.pool_fusion_logits = nn.Parameter(torch.zeros(2))

    def get_pool_weights(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        weights = torch.softmax(self.pool_fusion_logits, dim=0)
        if device is not None or dtype is not None:
            weights = weights.to(device=device if device is not None else weights.device, dtype=dtype or weights.dtype)
        return weights

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_features: [B, N, D] raw patch features (e.g. context_map flattened)
        Returns:
            global_logit: [B,] anomaly logit (anomaly_score - normal_score)
        """
        feat_2d = patch_features.transpose(1, 2).contiguous()
        gap = feat_2d.mean(dim=-1)  # [B, D]
        gmp = feat_2d.max(dim=-1).values  # [B, D]
        pool_weights = self.get_pool_weights(device=feat_2d.device, dtype=feat_2d.dtype)
        pooled = pool_weights[0] * gap + pool_weights[1] * gmp
        logits_2c = self.mlp_adapter(pooled)  # [B, 2]
        return logits_2c[:, 1] - logits_2c[:, 0]


class ScalarFusionHead(nn.Module):
    """Fuse multiple scalar anomaly cues into one final logit."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = max(hidden_dim, 16)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PQAdapter(nn.Module):
    """Prompt-query adapter that emits residual, local, and global PQA cues."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        beta: float = 1.0,
        image_size: int = 240,
    ):
        super().__init__()
        self.beta = beta
        self.image_size = int(image_size)
        self.sharebn = nn.ModuleList(
            nn.BatchNorm2d(feature_dim) for _ in range(num_layers)
        )
        self.local_heads = nn.ModuleList(
            PQAConvLocalHead(feature_dim, hidden_dim) for _ in range(num_layers)
        )
        self.global_heads = nn.ModuleList(
            PQAGlobalHead(feature_dim, hidden_dim) for _ in range(num_layers)
        )

    def _flatten_prompt_level(self, prompt_level: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, feature_dim = prompt_level.shape
        return prompt_level.reshape(batch_size, -1, feature_dim)

    def _match_prompt_patches(
        self,
        query_level: torch.Tensor,
        prompt_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_dim = query_level.shape[-1]
        query_norm = F.normalize(query_level, dim=-1)
        prompt_norm = F.normalize(prompt_flat, dim=-1)
        similarity = torch.einsum("bnc,bmc->bnm", query_norm, prompt_norm)
        max_cosine, best_indices = similarity.max(dim=-1)
        gather_index = best_indices.unsqueeze(-1).expand(-1, -1, feature_dim)
        aligned_prompt = torch.gather(prompt_flat, 1, gather_index)
        residual = 0.5 * (1.0 - max_cosine)
        return residual, best_indices, aligned_prompt

    def _build_context_map(
        self,
        query_level: torch.Tensor,
        aligned_prompt: torch.Tensor,
        beta_value: Union[float, torch.Tensor],
        layer_idx: int,
    ) -> torch.Tensor:
        batch_size, num_patches, feature_dim = query_level.shape
        grid_side = int(num_patches ** 0.5)
        if grid_side * grid_side != num_patches:
            raise ValueError(f"PQA expects a square patch grid, got {num_patches} patches.")
        query_level = F.normalize(query_level, dim=-1)
        aligned_prompt = F.normalize(aligned_prompt, dim=-1)
        context_feat = query_level + beta_value * torch.abs(query_level - aligned_prompt)
        context_map = context_feat.permute(0, 2, 1).reshape(batch_size, feature_dim, grid_side, grid_side)
        return self.sharebn[layer_idx](context_map)

    def _compute_local_outputs(
        self,
        context_map: torch.Tensor,
        layer_idx: int,
        grid_side: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_local_logits = self.local_heads[layer_idx](context_map)
        patch_logits_2c = F.adaptive_avg_pool2d(raw_local_logits, output_size=(grid_side, grid_side))
        patch_scores_2c = torch.softmax(patch_logits_2c, dim=1)
        patch_logits = (patch_logits_2c[:, 1] - patch_logits_2c[:, 0]).flatten(1)
        patch_scores = patch_scores_2c[:, 1].flatten(1)
        local_logits = F.interpolate(
            raw_local_logits,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        local_scores = torch.softmax(local_logits, dim=1)
        return local_logits, local_scores, patch_logits, patch_scores

    def forward(
        self,
        query_patch_levels: Sequence[torch.Tensor],
        prompt_patch_levels: Sequence[torch.Tensor],
        beta: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        query_levels = list(query_patch_levels)
        prompt_levels = list(prompt_patch_levels)
        if not query_levels:
            raise ValueError("query_patch_levels must contain at least one layer.")
        if len(query_levels) != len(prompt_levels):
            raise ValueError(
                "query_patch_levels and prompt_patch_levels must have the same number of layers."
            )
        if len(query_levels) != len(self.local_heads):
            raise ValueError(
                f"Expected {len(self.local_heads)} layers, got {len(query_levels)}."
            )

        inctrl_patch_maps = []
        pqa_patch_maps = []
        pqa_global_logits = []
        pqa_local_logits = []
        patch_logits_list = []
        patch_scores_list = []
        global_pool_weights = []
        aligned_indices = []
        aligned_prompt_features = []
        beta_value = self.beta if beta is None else beta

        for layer_idx, (query_level, prompt_level) in enumerate(zip(query_levels, prompt_levels)):
            num_patches = query_level.shape[1]
            grid_side = int(num_patches ** 0.5)
            prompt_flat = self._flatten_prompt_level(prompt_level)
            residual, best_indices, aligned_prompt = self._match_prompt_patches(query_level, prompt_flat)
            context_map = self._build_context_map(query_level, aligned_prompt, beta_value, layer_idx)
            local_logits_2c, _, patch_logits_from_pool, patch_scores_from_pool = self._compute_local_outputs(
                context_map=context_map,
                layer_idx=layer_idx,
                grid_side=grid_side,
            )
            # Reshape [B, D, H, W] -> [B, N, D] for AdaptCLIP-style global head
            context_flat = context_map.flatten(2).transpose(1, 2)  # [B, N, D]
            pqa_global_logit = self.global_heads[layer_idx](context_flat)  # [B,] image-level anomaly logit

            inctrl_patch_maps.append(residual)
            pqa_patch_maps.append(patch_scores_from_pool)
            pqa_global_logits.append(pqa_global_logit)
            pqa_local_logits.append(local_logits_2c)
            patch_logits_list.append(patch_logits_from_pool)
            patch_scores_list.append(patch_scores_from_pool)
            global_pool_weights.append(
                self.global_heads[layer_idx].get_pool_weights(
                    device=context_map.device,
                    dtype=context_map.dtype,
                )
            )
            aligned_indices.append(best_indices)
            aligned_prompt_features.append(aligned_prompt)

        return {
            "residual_maps": inctrl_patch_maps,
            "inctrl_patch_maps": inctrl_patch_maps,
            "pqa_patch_maps": pqa_patch_maps,
            "patch_logits": patch_logits_list,
            "patch_scores": patch_scores_list,
            "pqa_global_logits": pqa_global_logits,
            "pqa_local_logits": pqa_local_logits,
            "global_pool_weights": global_pool_weights,
            "aligned_indices": aligned_indices,
            "aligned_prompt_features": aligned_prompt_features,
        }


class InCTRLPQA(nn.Module):
    """Frozen CLIP backbone with fused residual, PQA, image, and text anomaly cues."""

    def __init__(
        self,
        args,
        embed_dim: int,
        vision_cfg: Union[CLIPVisionCfg, Dict[str, object]],
        text_cfg: Union[CLIPTextCfg, Dict[str, object]],
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        output_dict: bool = True,
        patch_layers: Tuple[int, ...] = (7, 9, 11),
        beta: float = 1.0,
        hidden_dim: int = 256,
        patch_has_cls_token: bool = True,
        feature_is_projected: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.embed_dim = embed_dim
        self.patch_layers = tuple(patch_layers)
        self.patch_has_cls_token = patch_has_cls_token
        self.feature_is_projected = feature_is_projected

        self.vision_cfg = _as_cfg(vision_cfg, CLIPVisionCfg)
        self.text_cfg = _as_cfg(text_cfg, CLIPTextCfg)
        self.image_size = getattr(args, "image_size", self.vision_cfg.image_size)
        self.beta = float(beta)

        self.visual = _build_vision_tower_Mul(embed_dim, self.vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, self.text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer("attn_mask", text.attn_mask, persistent=False)

        patch_grid = self.image_size // self.vision_cfg.patch_size
        self.num_patches = patch_grid * patch_grid
        if feature_is_projected:
            self.patch_projection = nn.Identity()
        else:
            patch_input_dim = getattr(self.vision_cfg, "width", embed_dim)
            self.patch_projection = (
                nn.Identity()
                if patch_input_dim == embed_dim
                else nn.Linear(patch_input_dim, embed_dim)
            )
        patch_feature_dim = embed_dim

        self.prompt_query_adapter = PQAdapter(
            feature_dim=patch_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=len(self.patch_layers),
            beta=beta,
            image_size=self.image_size,
        )
        self.image_head = ImageResidualHead(embed_dim, hidden_dim)
        self.holistic_head = ScalarFusionHead(
            input_dim=self.num_patches,
            hidden_dim=max(hidden_dim // 2, 32),
        )
        self.patch_map_fusion_logits = nn.Parameter(torch.zeros(2))
        self.final_logit_fusion_logits = nn.Parameter(torch.zeros(2))
        self.layer_weights_logits = nn.Parameter(torch.zeros(len(self.patch_layers)))

        for parameter in self.visual.parameters():
            parameter.requires_grad = False
        for parameter in self.transformer.parameters():
            parameter.requires_grad = False
        for parameter in self.token_embedding.parameters():
            parameter.requires_grad = False
        for parameter in self.ln_final.parameters():
            parameter.requires_grad = False
        self.positional_embedding.requires_grad = False
        self.text_projection.requires_grad = False

        self.enable_pqa_training()

    @staticmethod
    def _score_to_logit(score: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        score = score.clamp(min=eps, max=1.0 - eps)
        return torch.log(score) - torch.log1p(-score)

    def _get_layer_weights(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.softmax(self.layer_weights_logits, dim=0).to(device=device, dtype=dtype)

    def encode_text_prompted(
        self,
        prompts: torch.Tensor,
        tokenized_prompts: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        cast_dtype = self.transformer.get_cast_dtype()
        x = prompts.to(cast_dtype)
        x = x + self.positional_embedding[: x.shape[1]].to(cast_dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def _encode_text_descriptors(
        self,
        descriptors: Sequence[str],
        device: torch.device,
    ) -> torch.Tensor:
        tokenized = tokenize(list(descriptors), context_length=self.context_length).to(device)
        token_embeddings = self.token_embedding(tokenized).to(self.transformer.get_cast_dtype())
        features = self.encode_text_prompted(token_embeddings, tokenized, normalize=True)
        return F.normalize(features.mean(dim=0, keepdim=True), dim=-1).squeeze(0)

    def _compute_patch_residuals(
        self,
        query_patch_levels: Sequence[torch.Tensor],
        prompt_patch_levels: Sequence[torch.Tensor],
    ) -> Dict[str, List[torch.Tensor]]:
        residual_maps = []
        for query_level, prompt_level in zip(query_patch_levels, prompt_patch_levels):
            batch_size, _, feature_dim = query_level.shape
            prompt_reshaped = prompt_level.reshape(batch_size, -1, feature_dim)
            query_norm = F.normalize(query_level, dim=-1)
            prompt_norm = F.normalize(prompt_reshaped, dim=-1)
            similarity = torch.einsum("bnc,bmc->bnm", query_norm, prompt_norm)
            max_cosine = similarity.max(dim=-1).values
            residual = 0.5 * (1.0 - max_cosine)
            residual_maps.append(residual)
        return {"residual_maps": residual_maps}

    def _parse_visual_outputs(
        self,
        outputs: Union[torch.Tensor, Tuple[torch.Tensor, object], Tuple[torch.Tensor, object, object]],
    ) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
        if isinstance(outputs, tuple):
            if len(outputs) >= 2:
                return outputs[0], outputs[1]
        raise ValueError("Visual tower must return (global_token, patch_tokens, ...).")

    def _encode_visual_features(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
        outputs = self.visual(images, list(self.patch_layers))
        return self._parse_visual_outputs(outputs)

    def _prepare_patch_levels(
        self,
        patch_tokens: Sequence[torch.Tensor],
        batch_size: int,
        num_shots: int = 1,
    ) -> torch.Tensor:
        stacked_levels = torch.stack(list(patch_tokens), dim=1)
        if self.patch_has_cls_token:
            stacked_levels = stacked_levels[:, :, 1:, :]
        if self.feature_is_projected:
            if stacked_levels.shape[-1] != self.embed_dim:
                raise ValueError(
                    f"Expected projected patch tokens with dim={self.embed_dim}, got {stacked_levels.shape[-1]}. "
                    "Set feature_is_projected=False to enable lazy projection."
                )
        else:
            stacked_levels = self.patch_projection(stacked_levels)
        if num_shots == 1:
            return stacked_levels.reshape(batch_size, len(self.patch_layers), -1, self.embed_dim)
        stacked_levels = stacked_levels.reshape(batch_size, num_shots, len(self.patch_layers), -1, self.embed_dim)
        return stacked_levels.permute(0, 2, 1, 3, 4).contiguous()

    def _coerce_prompt_images(
        self,
        query_image: torch.Tensor,
        prompt_images: Optional[torch.Tensor] = None,
        normal_list: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        batch_size = query_image.size(0)

        def _normalize_batched_prompts(prompt_tensor: torch.Tensor, source_name: str) -> torch.Tensor:
            if prompt_tensor.dim() != 5:
                raise ValueError(f"{source_name} must be [B, S, C, H, W].")
            prompt_batch = prompt_tensor.size(0)
            if prompt_batch == 1:
                return prompt_tensor.expand(batch_size, -1, -1, -1, -1)
            if prompt_batch == batch_size:
                return prompt_tensor
            raise ValueError(
                f"{source_name} batch dimension must be 1 or match query batch size {batch_size}; got {prompt_batch}."
            )

        def _normalize_shared_prompts(prompt_tensor: torch.Tensor, source_name: str) -> torch.Tensor:
            if prompt_tensor.dim() == 4:
                return prompt_tensor.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            if prompt_tensor.dim() == 5:
                return _normalize_batched_prompts(prompt_tensor, source_name)
            raise ValueError(f"{source_name} must be [S, C, H, W] or [B, S, C, H, W].")

        if prompt_images is not None:
            if prompt_images.dim() == 4:
                return _normalize_shared_prompts(prompt_images, "prompt_images")
            return _normalize_batched_prompts(prompt_images, "prompt_images")

        if normal_list is None:
            raise ValueError("Either prompt_images or normal_list must be provided.")

        if isinstance(normal_list, (tuple, list)):
            prompt_images = torch.stack(list(normal_list), dim=0)
        else:
            prompt_images = normal_list

        return _normalize_shared_prompts(prompt_images, "normal_list")

    def _as_query_level_list(self, patch_levels: Union[torch.Tensor, Sequence[torch.Tensor]]) -> List[torch.Tensor]:
        if isinstance(patch_levels, torch.Tensor):
            if patch_levels.dim() != 4:
                raise ValueError("Query patch levels must be [B, L, N, D].")
            return [patch_levels[:, level_idx, :, :] for level_idx in range(patch_levels.shape[1])]
        return list(patch_levels)

    def _as_prompt_level_list(self, patch_levels: Union[torch.Tensor, Sequence[torch.Tensor]]) -> List[torch.Tensor]:
        if isinstance(patch_levels, torch.Tensor):
            if patch_levels.dim() != 5:
                raise ValueError("Prompt patch levels must be [B, L, K, N, D].")
            return [patch_levels[:, level_idx, :, :, :] for level_idx in range(patch_levels.shape[1])]
        return list(patch_levels)

    def _freeze_backbone_parameters(self) -> None:
        for parameter in self.visual.parameters():
            parameter.requires_grad = False
        for parameter in self.transformer.parameters():
            parameter.requires_grad = False
        for parameter in self.token_embedding.parameters():
            parameter.requires_grad = False
        for parameter in self.ln_final.parameters():
            parameter.requires_grad = False
        self.positional_embedding.requires_grad = False
        self.text_projection.requires_grad = False

    def _encode_prompt_features(
        self,
        query_image: torch.Tensor,
        prompt_images: Optional[torch.Tensor] = None,
        normal_list: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
        prompt_images = self._coerce_prompt_images(query_image, prompt_images, normal_list)
        batch_size, num_shots = prompt_images.shape[:2]
        prompt_images = prompt_images.to(query_image.device, dtype=query_image.dtype)
        flat_prompt_images = prompt_images.reshape(batch_size * num_shots, *prompt_images.shape[2:])
        prompt_global, prompt_patch_tokens = self._encode_visual_features(flat_prompt_images)
        prompt_patch_levels = self._prepare_patch_levels(
            prompt_patch_tokens,
            batch_size=batch_size,
            num_shots=num_shots,
        )
        prompt_global = prompt_global.reshape(batch_size, num_shots, -1)
        return prompt_global, self._as_prompt_level_list(prompt_patch_levels), num_shots

    @staticmethod
    def _normalize_obj_type(obj_type: object) -> str:
        return str(obj_type).replace("_", " ").strip().lower()

    @classmethod
    def _normalize_obj_types(cls, obj_types: Sequence[str]) -> List[str]:
        return [cls._normalize_obj_type(obj_type) for obj_type in obj_types]

    def _resolve_prompt_cache_category(
        self,
        category: Optional[str],
        obj_types: Optional[Sequence[str]],
    ) -> Optional[str]:
        normalized_category = self._normalize_obj_type(category) if category is not None else None
        if obj_types is None:
            return normalized_category
        normalized_obj_types = self._normalize_obj_types(obj_types)
        unique_obj_types = set(normalized_obj_types)
        if len(unique_obj_types) != 1:
            raise ValueError("Prompt feature cache can only be built for one category.")
        obj_category = normalized_obj_types[0]
        if normalized_category is not None and normalized_category != obj_category:
            raise ValueError(
                f"prompt_feature_cache category '{normalized_category}' does not match obj_types category '{obj_category}'."
            )
        return normalized_category or obj_category

    def _prepare_prompt_feature_cache(
        self,
        prompt_feature_cache: Dict[str, object],
        obj_types: Sequence[str],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        required_keys = {"prompt_global", "prompt_patch_levels", "num_shots"}
        missing_keys = required_keys.difference(prompt_feature_cache)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"prompt_feature_cache missing required keys: {missing}.")

        prompt_global = prompt_feature_cache["prompt_global"]
        if not isinstance(prompt_global, torch.Tensor) or prompt_global.dim() != 2:
            raise ValueError("prompt_feature_cache['prompt_global'] must have shape [S, D].")
        num_shots = int(prompt_feature_cache["num_shots"])
        if prompt_global.shape != (num_shots, self.embed_dim):
            raise ValueError(
                "prompt_feature_cache['prompt_global'] shape must be "
                f"[num_shots={num_shots}, embed_dim={self.embed_dim}], got {tuple(prompt_global.shape)}."
            )

        prompt_patch_levels = prompt_feature_cache["prompt_patch_levels"]
        if not isinstance(prompt_patch_levels, (list, tuple)):
            raise ValueError("prompt_feature_cache['prompt_patch_levels'] must be a list of tensors.")
        if len(prompt_patch_levels) != len(self.patch_layers):
            raise ValueError(
                "prompt_feature_cache prompt_patch_levels layers must match "
                f"len(self.patch_layers)={len(self.patch_layers)}, got {len(prompt_patch_levels)}."
            )

        prepared_levels = []
        for level_idx, level in enumerate(prompt_patch_levels):
            if not isinstance(level, torch.Tensor) or level.dim() != 3:
                raise ValueError(
                    f"prompt_feature_cache['prompt_patch_levels'][{level_idx}] must have shape [S, N, D]."
                )
            expected_shape = (num_shots, self.num_patches, self.embed_dim)
            if tuple(level.shape) != expected_shape:
                raise ValueError(
                    "prompt_feature_cache prompt patch level shape must be "
                    f"{expected_shape}, got {tuple(level.shape)} at level {level_idx}."
                )
            prepared_levels.append(
                level.to(device=device, dtype=dtype)
                .unsqueeze(0)
                .expand(batch_size, -1, -1, -1)
            )

        normalized_obj_types = self._normalize_obj_types(obj_types)
        unique_obj_types = set(normalized_obj_types)
        cache_category = prompt_feature_cache.get("category")
        if cache_category is None:
            if len(unique_obj_types) > 1:
                raise ValueError(
                    "prompt_feature_cache without category metadata cannot be used for mixed-category batches; "
                    "rebuild one cache per category."
                )
        else:
            normalized_cache_category = self._normalize_obj_type(cache_category)
            if unique_obj_types != {normalized_cache_category}:
                raise ValueError(
                    f"prompt_feature_cache category '{normalized_cache_category}' does not match "
                    f"batch categories {sorted(unique_obj_types)}."
                )

        prepared_global = prompt_global.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        return prepared_global, prepared_levels

    def _prepare_text_prototype_cache(
        self,
        text_prototype_cache: Dict[str, torch.Tensor],
        batch_size: int,
        feature_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        required_keys = {"normal_proto", "anomaly_proto"}
        missing_keys = required_keys.difference(text_prototype_cache)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"text_prototype_cache missing required keys: {missing}.")

        normal_proto = text_prototype_cache["normal_proto"]
        anomaly_proto = text_prototype_cache["anomaly_proto"]
        if not isinstance(normal_proto, torch.Tensor) or not isinstance(anomaly_proto, torch.Tensor):
            raise ValueError("text_prototype_cache values must be tensors.")
        if normal_proto.dim() != 2 or anomaly_proto.dim() != 2 or normal_proto.shape != anomaly_proto.shape:
            raise ValueError(
                "text_prototype_cache normal_proto and anomaly_proto shape must match [1, D] or [B, D]."
            )
        if normal_proto.size(-1) != feature_dim:
            raise ValueError(
                "text_prototype_cache feature dim does not match query_global; "
                f"expected {feature_dim}, got {normal_proto.size(-1)}."
            )

        normal_proto = normal_proto.to(device=device, dtype=dtype)
        anomaly_proto = anomaly_proto.to(device=device, dtype=dtype)
        if normal_proto.size(0) == 1:
            return normal_proto.expand(batch_size, -1), anomaly_proto.expand(batch_size, -1)
        if normal_proto.size(0) != batch_size:
            raise ValueError(
                "text_prototype_cache batch dimension must be 1 or match "
                f"batch_size={batch_size}, got {normal_proto.size(0)}."
            )
        return normal_proto, anomaly_proto

    @torch.no_grad()
    def build_prompt_feature_cache(
        self,
        prompt_images: Optional[torch.Tensor] = None,
        normal_list: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        obj_types: Optional[Sequence[str]] = None,
        category: Optional[str] = None,
    ) -> Dict[str, object]:
        """Pre-encode shared few-shot prompt images for fast evaluation."""
        if prompt_images is None:
            if normal_list is None:
                raise ValueError("Either prompt_images or normal_list must be provided.")
            if isinstance(normal_list, (tuple, list)):
                prompt_images = torch.stack(list(normal_list), dim=0)
            else:
                prompt_images = normal_list

        if prompt_images.dim() == 5:
            if prompt_images.size(0) != 1:
                raise ValueError("Prompt feature cache only supports shared prompts with batch size 1.")
            prompt_images = prompt_images.squeeze(0)
        if prompt_images.dim() != 4:
            raise ValueError("prompt_images must be [S, C, H, W] or [1, S, C, H, W].")

        device = next(self.parameters()).device
        prompt_images = prompt_images.to(device)
        num_shots = prompt_images.shape[0]
        prompt_global, prompt_patch_tokens = self._encode_visual_features(prompt_images)
        prompt_patch_levels = self._prepare_patch_levels(
            prompt_patch_tokens,
            batch_size=1,
            num_shots=num_shots,
        )
        prompt_global = prompt_global.reshape(1, num_shots, -1)
        prompt_patch_level_list = self._as_prompt_level_list(prompt_patch_levels)
        cache_category = self._resolve_prompt_cache_category(category=category, obj_types=obj_types)

        cache: Dict[str, object] = {
            "prompt_global": prompt_global.squeeze(0).detach(),
            "prompt_patch_levels": [
                level.squeeze(0).detach()
                for level in prompt_patch_level_list
            ],
            "num_shots": num_shots,
        }
        if cache_category is not None:
            cache["category"] = cache_category
        return cache

    @torch.no_grad()
    def _build_static_text_prototypes(
        self,
        obj_types: Sequence[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the fixed InCTRL / WinCLIP text prior without trainable prompts."""
        if not hasattr(self, "_static_text_cache"):
            self._static_text_cache = {}

        normal_prototypes = []
        anomaly_prototypes = []
        unique_obj_types = []
        inverse_indices = []
        unique_index = {}
        for obj_type in obj_types:
            obj_key = str(obj_type).replace("_", " ")
            if obj_key not in unique_index:
                unique_index[obj_key] = len(unique_obj_types)
                unique_obj_types.append(obj_key)
            inverse_indices.append(unique_index[obj_key])

        for obj_type in unique_obj_types:
            if obj_type not in self._static_text_cache:
                normal_descriptors, anomaly_descriptors = get_texts(obj_type)
                n_proto = self._encode_text_descriptors(normal_descriptors, device)
                a_proto = self._encode_text_descriptors(anomaly_descriptors, device)
                self._static_text_cache[obj_type] = (n_proto.cpu(), a_proto.cpu())

            n_proto, a_proto = self._static_text_cache[obj_type]
            normal_prototypes.append(n_proto.to(device))
            anomaly_prototypes.append(a_proto.to(device))

        normal_stack = torch.stack(normal_prototypes, dim=0)
        anomaly_stack = torch.stack(anomaly_prototypes, dim=0)
        index_tensor = torch.tensor(inverse_indices, device=device, dtype=torch.long)
        return normal_stack[index_tensor], anomaly_stack[index_tensor]

    @torch.no_grad()
    def build_text_prototype_cache(
        self,
        obj_types: Sequence[str],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        normal_proto, anomaly_proto = self._build_static_text_prototypes(
            obj_types=obj_types,
            device=device,
        )
        return {
            "normal_proto": normal_proto.detach(),
            "anomaly_proto": anomaly_proto.detach(),
        }

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        modules = [
            self.patch_projection,
            self.image_head,
            self.holistic_head,
            self.prompt_query_adapter,
        ]
        parameters: List[nn.Parameter] = []
        for module in modules:
            parameters.extend(list(module.parameters()))
        parameters.append(self.patch_map_fusion_logits)
        parameters.append(self.final_logit_fusion_logits)
        parameters.append(self.layer_weights_logits)
        return parameters

    def enable_pqa_training(self) -> None:
        for parameter in self.get_trainable_parameters():
            parameter.requires_grad = True
        self._freeze_backbone_parameters()

    def forward(
        self,
        query_image: torch.Tensor,
        prompt_images: Optional[torch.Tensor] = None,
        normal_list: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        prompt_feature_cache: Optional[Dict[str, object]] = None,
        obj_types: Optional[Sequence[str]] = None,
        text_prototype_cache: Optional[Dict[str, torch.Tensor]] = None,
        return_aux: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Dict[str, object], Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = query_image.size(0)
        if obj_types is None:
            obj_types = ["object"] * batch_size
        if len(obj_types) != batch_size:
            raise ValueError(f"obj_types length must match batch_size={batch_size}, got {len(obj_types)}.")

        query_global, query_patch_tokens = self._encode_visual_features(query_image)
        query_patch_levels = self._prepare_patch_levels(query_patch_tokens, batch_size=batch_size, num_shots=1)
        query_patch_level_list = self._as_query_level_list(query_patch_levels)

        if prompt_feature_cache is not None:
            prompt_global, prompt_patch_levels = self._prepare_prompt_feature_cache(
                prompt_feature_cache=prompt_feature_cache,
                obj_types=obj_types,
                batch_size=batch_size,
                device=query_image.device,
                dtype=query_patch_level_list[0].dtype,
            )
        else:
            prompt_global, prompt_patch_levels, _ = self._encode_prompt_features(
                query_image=query_image,
                prompt_images=prompt_images,
                normal_list=normal_list,
            )

        layer_weights = self._get_layer_weights(
            device=query_image.device,
            dtype=query_patch_level_list[0].dtype,
        )
        # Insert PQA immediately after visual feature extraction and before the raw residual branch.
        beta_value = torch.tensor(
            self.beta,
            device=query_image.device,
            dtype=query_patch_level_list[0].dtype,
        )
        pq_outputs = self.prompt_query_adapter(
            query_patch_levels=query_patch_level_list,
            prompt_patch_levels=prompt_patch_levels,
            beta=beta_value,
        )
        raw_residual_maps = pq_outputs["residual_maps"]
        raw_base_patch_map = sum(
            weight * residual
            for weight, residual in zip(layer_weights, raw_residual_maps)
        )
        pqa_patch_map = sum(
            weight * patch_score
            for weight, patch_score in zip(layer_weights, pq_outputs["pqa_patch_maps"])
        )
        pqa_patch_logit = sum(
            weight * patch_logit
            for weight, patch_logit in zip(layer_weights, pq_outputs["patch_logits"])
        )
        pqa_patch_score = sum(
            weight * patch_score
            for weight, patch_score in zip(layer_weights, pq_outputs["patch_scores"])
        )
        pqa_logit = sum(
            weight * global_logit
            for weight, global_logit in zip(layer_weights, pq_outputs["pqa_global_logits"])
        )
        pqa_local_logits = sum(
            weight * local_logit
            for weight, local_logit in zip(layer_weights, pq_outputs["pqa_local_logits"])
        )
        base_patch_map = sum(
            weight * residual
            for weight, residual in zip(layer_weights, pq_outputs["residual_maps"])
        )
        patch_map_fusion_weights = torch.softmax(self.patch_map_fusion_logits, dim=0)
        hybrid_patch_map = (
            patch_map_fusion_weights[0] * base_patch_map
            + patch_map_fusion_weights[1] * pqa_patch_map
        )
        pqa_score = torch.sigmoid(pqa_logit)
        pqa_local_scores = torch.softmax(pqa_local_logits, dim=1)
        final_patch_map = hybrid_patch_map

        prompt_global_proto = F.normalize(prompt_global.mean(dim=1), dim=-1)
        query_global_norm_for_image = F.normalize(query_global, dim=-1)
        image_residual = torch.abs(prompt_global_proto - query_global_norm_for_image)
        image_logit = self.image_head(image_residual)
        image_score = torch.sigmoid(image_logit)

        if text_prototype_cache is not None:
            normal_proto, anomaly_proto = self._prepare_text_prototype_cache(
                text_prototype_cache=text_prototype_cache,
                batch_size=batch_size,
                feature_dim=query_global.shape[-1],
                device=query_image.device,
                dtype=query_global.dtype,
            )
        else:
            normal_proto, anomaly_proto = self._build_static_text_prototypes(
                obj_types=obj_types,
                device=query_image.device,
            )

        query_global_norm = F.normalize(query_global, dim=-1)
        normal_proto = F.normalize(normal_proto, dim=-1)
        anomaly_proto = F.normalize(anomaly_proto, dim=-1)
        normal_logit = 100.0 * torch.sum(query_global_norm * normal_proto, dim=-1)
        anomaly_logit = 100.0 * torch.sum(query_global_norm * anomaly_proto, dim=-1)
        text_logits_2c = torch.stack([normal_logit, anomaly_logit], dim=-1)
        text_score = torch.softmax(text_logits_2c, dim=-1)[:, 1]
        text_logit = anomaly_logit - normal_logit

        holistic_map = (
            final_patch_map
            + image_score.unsqueeze(-1)
            + text_score.unsqueeze(-1)
        )
        holistic_logit = self.holistic_head(holistic_map)
        holistic_score = torch.sigmoid(holistic_logit)
        max_base_patch_score = base_patch_map.max(dim=-1).values
        raw_max_patch_score = raw_base_patch_map.max(dim=-1).values
        max_hybrid_patch_score = hybrid_patch_map.max(dim=-1).values
        max_patch_score = final_patch_map.max(dim=-1).values

        max_base_patch_logit = self._score_to_logit(max_base_patch_score)
        raw_max_patch_logit = self._score_to_logit(raw_max_patch_score)
        max_hybrid_patch_logit = self._score_to_logit(max_hybrid_patch_score)
        max_patch_logit = self._score_to_logit(max_patch_score)

        base_logit = 0.5 * (holistic_logit + max_patch_logit)
        final_logit_fusion_weights = torch.softmax(self.final_logit_fusion_logits, dim=0)
        final_logit = (
            final_logit_fusion_weights[0] * base_logit
            + final_logit_fusion_weights[1] * pqa_logit
        )
        final_score = torch.sigmoid(final_logit)
        base_score = torch.sigmoid(base_logit)

        patch_side = int(self.num_patches ** 0.5)
        aux = {}
        if return_aux:
            aux = {
                "patch_map_2d": final_patch_map.reshape(batch_size, patch_side, patch_side),
                "raw_base_patch_map_2d": raw_base_patch_map.reshape(batch_size, patch_side, patch_side),
                "base_patch_map_2d": base_patch_map.reshape(batch_size, patch_side, patch_side),
                "hybrid_patch_map_2d": hybrid_patch_map.reshape(batch_size, patch_side, patch_side),
                "per_layer_pqa_patch_map": pq_outputs["pqa_patch_maps"],
                "per_layer_pqa_patch_logit": pq_outputs["patch_logits"],
                "per_layer_residual": pq_outputs["residual_maps"],
                "per_layer_raw_residual": raw_residual_maps,
                "aligned_indices": pq_outputs["aligned_indices"],
                "raw_query_global": query_global,
                "prompt_global_proto": prompt_global_proto,
                "image_residual": image_residual,
                "layer_weights": layer_weights,
                "patch_map_fusion_weights": patch_map_fusion_weights,
                "pqa_global_pool_weights": pq_outputs["global_pool_weights"],
                "final_logit_fusion_weights": final_logit_fusion_weights,
                "text_prototypes": {
                    "normal": normal_proto,
                    "anomaly": anomaly_proto,
                },
            }

        result = {
            "final_score": final_score,
            "final_logit": final_logit,
            "base_score": base_score,
            "base_logit": base_logit,
            "holistic_score": holistic_score,
            "holistic_logit": holistic_logit,
            "image_score": image_score,
            "image_logit": image_logit,
            "text_score": text_score,
            "text_logit": text_logit,
            "pqa_score": pqa_score,
            "pqa_logit": pqa_logit,
            "pqa_patch_score": pqa_patch_score,
            "pqa_patch_logit": pqa_patch_logit,
            "pqa_local_scores": pqa_local_scores,
            "pqa_local_logits": pqa_local_logits,
            "patch_map": final_patch_map,
            "raw_base_patch_map": raw_base_patch_map,
            "base_patch_map": base_patch_map,
            "hybrid_patch_map": hybrid_patch_map,
            "max_patch_score": max_patch_score,
            "max_patch_logit": max_patch_logit,
            "raw_max_patch_score": raw_max_patch_score,
            "raw_max_patch_logit": raw_max_patch_logit,
            "max_base_patch_score": max_base_patch_score,
            "max_base_patch_logit": max_base_patch_logit,
            "max_hybrid_patch_score": max_hybrid_patch_score,
            "max_hybrid_patch_logit": max_hybrid_patch_logit,
            "patch_map_fusion_weights": patch_map_fusion_weights,
            "pqa_global_pool_weights": pq_outputs["global_pool_weights"],
            "final_logit_fusion_weights": final_logit_fusion_weights,
            "aux": aux,
        }
        use_return_dict = self.output_dict if return_dict is None else return_dict
        if use_return_dict:
            return result
        return final_score, final_patch_map

    def forward_legacy(
        self,
        tokenizer,
        image: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        text: Optional[Sequence[str]] = None,
        normal_list: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compatibility shim for the original InCTRL forward signature."""
        del tokenizer
        if image is None:
            raise ValueError("forward_legacy requires image input.")

        prompt_images = None
        normal_source = normal_list
        if isinstance(image, (tuple, list)):
            if not image:
                raise ValueError("forward_legacy image sequence must not be empty.")
            query_image = image[0]
            if normal_source is None:
                prompt_items = list(image[1:])
                if not prompt_items:
                    raise ValueError("forward_legacy requires prompt images or normal_list.")
                if prompt_items[0].dim() == 4:
                    prompt_images = torch.stack(prompt_items, dim=1)
                else:
                    prompt_images = torch.stack(prompt_items, dim=0)
        else:
            query_image = image
            if normal_source is None:
                raise ValueError("forward_legacy tensor image input requires normal_list.")

        outputs = self.forward(
            query_image=query_image,
            prompt_images=prompt_images,
            normal_list=normal_source,
            obj_types=text,
            return_aux=False,
            return_dict=True,
        )
        if not isinstance(outputs, dict):
            raise TypeError("forward_legacy expected dict outputs from fused forward.")
        return outputs["final_score"], outputs["image_score"]


__all__ = [
    "InCTRLPQA",
    "ImageResidualHead",
    "MLPAdapter",
    "PQAdapter",
    "PQAGlobalHead",
    "ScalarFusionHead",
]
