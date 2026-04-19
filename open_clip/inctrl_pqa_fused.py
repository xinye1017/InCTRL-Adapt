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


class PQAGlobalHead(nn.Module):
    """Patch-wise classification adapter followed by GAP and MIL pooling."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        global_topk: int = 10,
        projector_layers: int = 2,
    ):
        super().__init__()
        if projector_layers < 1:
            raise ValueError("projector_layers must be >= 1")
        self.global_topk = max(int(global_topk), 1)
        if projector_layers == 1:
            adapter_layers = [nn.Conv1d(feature_dim, 2, kernel_size=1)]
        else:
            adapter_layers = [
                nn.Conv1d(feature_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
            for _ in range(projector_layers - 2):
                adapter_layers.extend(
                    [
                        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(inplace=True),
                    ]
                )
            adapter_layers.append(nn.Conv1d(hidden_dim, 2, kernel_size=1))
        self.classification_adapter = nn.Sequential(*adapter_layers)

    def project_patch_logits(self, patch_features: torch.Tensor) -> torch.Tensor:
        return self.classification_adapter(patch_features)

    def pool_projected_logits(self, projected_logits: torch.Tensor) -> torch.Tensor:
        topk = min(self.global_topk, projected_logits.shape[-1])
        gap_logits = projected_logits.mean(dim=-1)
        topk_logits = projected_logits.topk(topk, dim=-1).values.mean(dim=-1)
        return 0.5 * (gap_logits + topk_logits)

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        projected_logits = self.project_patch_logits(patch_features)
        return self.pool_projected_logits(projected_logits)


class HolisticScoringHead(nn.Module):
    def __init__(self, patch_dim: int, hidden_dim: int):
        super().__init__()
        mid_dim = max(hidden_dim // 2, 1)
        self.net = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


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
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PQAdapter(nn.Module):
    """Safe prompt-query adapter aligned with the InCTRLPQA interface."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        beta: float = 1.0,
        learnable_layer_weights: bool = False,
        global_topk: int = 10,
        image_size: int = 240,
    ):
        super().__init__()
        self.beta = beta
        self.learnable_layer_weights = learnable_layer_weights
        self.global_topk = max(int(global_topk), 1)
        self.image_size = int(image_size)
        self.sharebn = nn.ModuleList(
            nn.BatchNorm2d(feature_dim) for _ in range(num_layers)
        )
        self.local_heads = nn.ModuleList(
            PQAConvLocalHead(feature_dim, hidden_dim) for _ in range(num_layers)
        )
        self.global_heads = nn.ModuleList(
            PQAGlobalHead(feature_dim, hidden_dim, global_topk=self.global_topk) for _ in range(num_layers)
        )
        if learnable_layer_weights:
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        else:
            self.register_parameter("layer_weights", None)

    def _get_layer_weights(self, device: torch.device) -> torch.Tensor:
        if self.layer_weights is None:
            return torch.ones(len(self.local_heads), device=device) / len(self.local_heads)
        return torch.softmax(self.layer_weights, dim=0)

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
        local_logits = self.local_heads[layer_idx](context_map)
        local_logits = F.interpolate(
            local_logits,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        local_scores = torch.softmax(local_logits, dim=1)
        patch_logits_2c = F.adaptive_avg_pool2d(local_logits, output_size=(grid_side, grid_side))
        patch_scores_2c = torch.softmax(patch_logits_2c, dim=1)
        patch_logits = (patch_logits_2c[:, 1] - patch_logits_2c[:, 0]).flatten(1)
        patch_scores = patch_scores_2c[:, 1].flatten(1)
        return local_logits, local_scores, patch_logits, patch_scores

    def forward(
        self,
        query_patch_levels: Sequence[torch.Tensor],
        prompt_patch_levels: Sequence[torch.Tensor],
        beta: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[torch.Tensor]]:
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

        patch_logits = []
        patch_scores = []
        pqa_patch_maps = []
        pqa_global_logits = []
        pqa_global_logits_2c = []
        pqa_global_scores = []
        pqa_local_logits = []
        pqa_local_scores = []
        residual_maps = []
        aligned_indices = []
        aligned_prompt_features = []
        beta_value = self.beta if beta is None else beta

        for layer_idx, (query_level, prompt_level) in enumerate(zip(query_levels, prompt_levels)):
            num_patches = query_level.shape[1]
            grid_side = int(num_patches ** 0.5)
            prompt_flat = self._flatten_prompt_level(prompt_level)
            residual, best_indices, aligned_prompt = self._match_prompt_patches(query_level, prompt_flat)
            context_map = self._build_context_map(query_level, aligned_prompt, beta_value, layer_idx)
            local_logit_2c, local_score_2c, patch_logit, patch_score = self._compute_local_outputs(
                context_map=context_map,
                layer_idx=layer_idx,
                grid_side=grid_side,
            )
            pqa_global_logit_2c = self.global_heads[layer_idx](context_map.flatten(2))
            pqa_global_logit = pqa_global_logit_2c[:, 1] - pqa_global_logit_2c[:, 0]

            patch_logits.append(patch_logit)
            patch_scores.append(patch_score)
            pqa_patch_maps.append(patch_score)
            pqa_global_logits.append(pqa_global_logit)
            pqa_global_logits_2c.append(pqa_global_logit_2c)
            pqa_global_scores.append(torch.sigmoid(pqa_global_logit))
            pqa_local_logits.append(local_logit_2c)
            pqa_local_scores.append(local_score_2c)
            residual_maps.append(residual)
            aligned_indices.append(best_indices)
            aligned_prompt_features.append(aligned_prompt)

        return {
            "patch_logits": patch_logits,
            "patch_scores": patch_scores,
            "pqa_patch_maps": pqa_patch_maps,
            "pqa_global_logits": pqa_global_logits,
            "pqa_global_logits_2c": pqa_global_logits_2c,
            "pqa_global_scores": pqa_global_scores,
            "pqa_local_logits": pqa_local_logits,
            "pqa_local_scores": pqa_local_scores,
            "residual_maps": residual_maps,
            "aligned_indices": aligned_indices,
            "aligned_prompt_features": aligned_prompt_features,
            "layer_weights": self._get_layer_weights(query_levels[0].device),
        }

class InCTRLPQA(nn.Module):
    """Original InCTRL backbone with a single prompt-query branch."""

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
        self.shot = getattr(args, "shot", 1)
        pqa_cfg = getattr(args, "PQA", None)
        pqa_global_topk = int(getattr(pqa_cfg, "GLOBAL_TOPK", 10))

        self.beta = float(beta)
        self.pqa_global_topk = pqa_global_topk

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
            learnable_layer_weights=False,
            global_topk=pqa_global_topk,
            image_size=self.image_size,
        )
        self.image_head = ImageResidualHead(embed_dim, hidden_dim)
        self.holistic_head = HolisticScoringHead(self.num_patches, hidden_dim)
        self.decision_head = ScalarFusionHead(input_dim=5, hidden_dim=max(hidden_dim // 2, 32))
        self.patch_map_fusion_logits = nn.Parameter(torch.zeros(2))

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

    def _get_layer_weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.full(
            (len(self.patch_layers),),
            1.0 / len(self.patch_layers),
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _score_to_logit(score: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        score = score.clamp(min=eps, max=1.0 - eps)
        return torch.log(score) - torch.log1p(-score)

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
        aligned_indices = []
        aligned_prompt_features = []

        for query_level, prompt_level in zip(query_patch_levels, prompt_patch_levels):
            _, _, dim = query_level.shape
            prompt_flat = prompt_level.reshape(query_level.size(0), -1, dim)

            query_norm = F.normalize(query_level, dim=-1)
            prompt_norm = F.normalize(prompt_flat, dim=-1)
            similarity = torch.einsum("bnc,bmc->bnm", query_norm, prompt_norm)
            max_cosine, best_indices = similarity.max(dim=-1)
            residual = 0.5 * (1.0 - max_cosine)

            gather_index = best_indices.unsqueeze(-1).expand(-1, -1, dim)
            aligned_prompt = torch.gather(prompt_flat, 1, gather_index)

            residual_maps.append(residual)
            aligned_indices.append(best_indices)
            aligned_prompt_features.append(aligned_prompt)

        return {
            "residual_maps": residual_maps,
            "aligned_indices": aligned_indices,
            "aligned_prompt_features": aligned_prompt_features,
        }

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
        if prompt_images is not None:
            if prompt_images.dim() == 4:
                prompt_images = prompt_images.unsqueeze(0).expand(query_image.size(0), -1, -1, -1, -1)
            return prompt_images

        if normal_list is None:
            raise ValueError("Either prompt_images or normal_list must be provided.")

        if isinstance(normal_list, (tuple, list)):
            prompt_images = torch.stack(list(normal_list), dim=0)
        else:
            prompt_images = normal_list

        if prompt_images.dim() == 4:
            prompt_images = prompt_images.unsqueeze(0).expand(query_image.size(0), -1, -1, -1, -1)
        elif prompt_images.dim() != 5:
            raise ValueError("normal_list must be [S, C, H, W] or [B, S, C, H, W].")
        return prompt_images

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

    @torch.no_grad()
    def build_prompt_feature_cache(
        self,
        prompt_images: Optional[torch.Tensor] = None,
        normal_list: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
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

        return {
            "prompt_global": prompt_global.squeeze(0).detach(),
            "prompt_patch_levels": [
                level.squeeze(0).detach()
                for level in prompt_patch_level_list
            ],
            "num_shots": num_shots,
        }

    def _build_static_text_prototypes(
        self,
        obj_types: Sequence[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the fixed InCTRL / WinCLIP text prior without trainable prompts."""
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
            normal_descriptors, anomaly_descriptors = get_texts(obj_type)
            normal_prototypes.append(self._encode_text_descriptors(normal_descriptors, device))
            anomaly_prototypes.append(self._encode_text_descriptors(anomaly_descriptors, device))

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
            self.decision_head,
            self.prompt_query_adapter,
        ]
        parameters: List[nn.Parameter] = []
        for module in modules:
            parameters.extend(list(module.parameters()))
        parameters.append(self.patch_map_fusion_logits)
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
        return_dict: bool = True,
    ) -> Dict[str, object]:
        if obj_types is None:
            obj_types = ["object"] * query_image.size(0)

        query_global, query_patch_tokens = self._encode_visual_features(query_image)
        batch_size = query_image.size(0)
        query_patch_levels = self._prepare_patch_levels(query_patch_tokens, batch_size=batch_size, num_shots=1)
        query_patch_level_list = self._as_query_level_list(query_patch_levels)

        if prompt_feature_cache is not None:
            prompt_global = prompt_feature_cache["prompt_global"].to(
                query_image.device,
                dtype=query_global.dtype,
            )
            prompt_global = prompt_global.unsqueeze(0).expand(batch_size, -1, -1)
            prompt_patch_levels = [
                level.to(query_image.device, dtype=query_patch_levels[0].dtype)
                .unsqueeze(0)
                .expand(batch_size, -1, -1, -1)
                for level in prompt_feature_cache["prompt_patch_levels"]
            ]
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
        raw_residual_outputs = self._compute_patch_residuals(
            query_patch_levels=query_patch_level_list,
            prompt_patch_levels=prompt_patch_levels,
        )
        raw_base_patch_map = sum(
            weight * residual
            for weight, residual in zip(layer_weights, raw_residual_outputs["residual_maps"])
        )
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
            normal_proto = text_prototype_cache["normal_proto"].to(
                query_image.device,
                dtype=query_global.dtype,
            )
            anomaly_proto = text_prototype_cache["anomaly_proto"].to(
                query_image.device,
                dtype=query_global.dtype,
            )
            if normal_proto.size(0) == 1 and batch_size != 1:
                normal_proto = normal_proto.expand(batch_size, -1)
                anomaly_proto = anomaly_proto.expand(batch_size, -1)
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

        holistic_input = (
            final_patch_map
            + image_score.unsqueeze(-1)
            + text_score.unsqueeze(-1)
            + pqa_score.unsqueeze(-1)
        )
        holistic_logit = self.holistic_head(holistic_input)
        holistic_score = torch.sigmoid(holistic_logit)
        max_base_patch_score = base_patch_map.max(dim=-1).values
        raw_max_patch_score = raw_base_patch_map.max(dim=-1).values
        max_hybrid_patch_score = hybrid_patch_map.max(dim=-1).values
        max_patch_score = final_patch_map.max(dim=-1).values

        max_base_patch_logit = self._score_to_logit(max_base_patch_score)
        raw_max_patch_logit = self._score_to_logit(raw_max_patch_score)
        max_hybrid_patch_logit = self._score_to_logit(max_hybrid_patch_score)
        max_patch_logit = self._score_to_logit(max_patch_score)

        base_logit = 0.5 * (holistic_logit + max_base_patch_logit)
        decision_input = torch.stack(
            [
                holistic_logit,
                max_patch_logit,
                pqa_logit,
                image_logit,
                text_logit,
            ],
            dim=-1,
        )
        final_logit = self.decision_head(decision_input)
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
                "per_layer_raw_residual": raw_residual_outputs["residual_maps"],
                "aligned_indices": pq_outputs["aligned_indices"],
                "raw_query_global": query_global,
                "prompt_global_proto": prompt_global_proto,
                "image_residual": image_residual,
                "layer_weights": layer_weights,
                "patch_map_fusion_weights": patch_map_fusion_weights,
                "decision_input": decision_input,
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
            "aux": aux,
        }
        if return_dict or self.output_dict:
            return result
        return final_score, final_patch_map


__all__ = [
    "HolisticScoringHead",
    "InCTRLPQA",
    "ImageResidualHead",
    "PQAdapter",
    "PQAGlobalHead",
    "ScalarFusionHead",
]
