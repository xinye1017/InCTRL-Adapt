from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

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


def _inverse_softplus(value: float) -> torch.Tensor:
    value_tensor = torch.tensor(float(value), dtype=torch.float32)
    return torch.log(torch.expm1(value_tensor))


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return residual + x


class ResidualProjectionMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.net(x)
        return F.normalize(x, dim=-1)


class ZeroInitProjectionMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContextResidualPatchHead(nn.Module):
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
    """AdaptCLIP-style local segmentation head for prompt-query fusion features."""

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
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        mid_dim = max(hidden_dim // 2, 1)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, mid_dim, bias=False),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, 2, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


class TextPriorHead(nn.Module):
    def __init__(self, logit_scale_init: float = 1 / 0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(logit_scale_init)))

    def forward(
        self,
        query_global: torch.Tensor,
        normal_proto: torch.Tensor,
        anomaly_proto: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        query_global = F.normalize(query_global, dim=-1)
        normal_proto = F.normalize(normal_proto, dim=-1)
        anomaly_proto = F.normalize(anomaly_proto, dim=-1)

        scale = self.logit_scale.exp()
        normal_logit = scale * torch.sum(query_global * normal_proto, dim=-1)
        anomaly_logit = scale * torch.sum(query_global * anomaly_proto, dim=-1)
        text_logit = anomaly_logit - normal_logit
        return text_logit, {
            "normal_logit": normal_logit,
            "anomaly_logit": anomaly_logit,
            "text_logit": text_logit,
            "text_score": torch.sigmoid(text_logit),
        }


class VisualAdapter(nn.Module):
    def __init__(
        self,
        global_dim: int,
        local_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        local_per_layer: bool = True,
    ):
        super().__init__()
        self.global_adapter = ResidualMLP(global_dim, hidden_dim)
        self.local_per_layer = local_per_layer
        if local_per_layer:
            self.local_adapters = nn.ModuleList(
                ResidualMLP(local_dim, hidden_dim) for _ in range(num_layers)
            )
            self.local_adapter = None
        else:
            self.local_adapters = None
            self.local_adapter = ResidualMLP(local_dim, hidden_dim)

    def adapt_global(self, x: torch.Tensor) -> torch.Tensor:
        return self.global_adapter(x)

    def adapt_local(self, patch_levels: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        adapted_levels = []
        for layer_idx, level in enumerate(patch_levels):
            adapted_levels.append(self.adapt_local_level(level, layer_idx))
        return adapted_levels

    def adapt_local_level(self, level: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if self.local_per_layer:
            return self.local_adapters[layer_idx](level)
        return self.local_adapter(level)


class TextualAdapter(nn.Module):
    def __init__(
        self,
        context_dim: int,
        feature_dim: int,
        context_length: int = 12,
        hidden_dim: int = 256,
        max_prompts_per_state: Optional[int] = 32,
        mode: str = "static_residual",
    ):
        super().__init__()
        self.context_length = context_length
        self.max_prompts_per_state = max_prompts_per_state
        self.mode = str(mode).lower()
        if self.mode not in {"static_residual", "descriptor_context"}:
            raise ValueError("TextualAdapter mode must be static_residual or descriptor_context.")
        self.normal_ctx = nn.Parameter(torch.empty(context_length, context_dim))
        self.anomaly_ctx = nn.Parameter(torch.empty(context_length, context_dim))
        nn.init.normal_(self.normal_ctx, std=0.02)
        nn.init.normal_(self.anomaly_ctx, std=0.02)
        self.prototype_adapter = ResidualProjectionMLP(feature_dim, hidden_dim)
        self.text_delta_adapter = ZeroInitProjectionMLP(feature_dim, hidden_dim)
        self.static_normal = [
            "{}",
            "flawless {}",
            "perfect {}",
            "unblemished {}",
            "{} without flaw",
            "{} without defect",
            "{} without damage",
        ]
        self.static_anomaly = [
            "damaged {}",
            "{} with flaw",
            "{} with defect",
            "{} with damage",
        ]

    def _build_prompt_embeddings(
        self,
        model: "InCTRLWithAdapters",
        descriptors: Sequence[str],
        context_embeddings: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_prefix = " ".join(["X"] * self.context_length)
        prompt_texts = [f"{prompt_prefix} {descriptor}" for descriptor in descriptors]
        tokenized = tokenize(prompt_texts, context_length=model.context_length).to(device)
        token_embeddings = model.token_embedding(tokenized).to(model.transformer.get_cast_dtype())
        prefix = token_embeddings[:, :1, :]
        suffix = token_embeddings[:, 1 + self.context_length :, :]
        learned_ctx = context_embeddings.unsqueeze(0).expand(len(descriptors), -1, -1)
        prompted = torch.cat([prefix, learned_ctx, suffix], dim=1)
        return prompted, tokenized

    def _build_binary_prompt_embedding(
        self,
        model: "InCTRLWithAdapters",
        context_embeddings: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_prefix = " ".join(["X"] * self.context_length)
        tokenized = tokenize([prompt_prefix], context_length=model.context_length).to(device)
        token_embeddings = model.token_embedding(tokenized).to(model.transformer.get_cast_dtype())
        prefix = token_embeddings[:, :1, :]
        suffix = token_embeddings[:, 1 + self.context_length :, :]
        learned_ctx = context_embeddings.unsqueeze(0)
        prompted = torch.cat([prefix, learned_ctx, suffix], dim=1)
        return prompted, tokenized

    def build_static_residual_prototypes(
        self,
        model: "InCTRLWithAdapters",
        static_normal: torch.Tensor,
        static_anomaly: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompted_normal, tokenized_normal = self._build_binary_prompt_embedding(
            model=model,
            context_embeddings=self.normal_ctx,
            device=device,
        )
        prompted_anomaly, tokenized_anomaly = self._build_binary_prompt_embedding(
            model=model,
            context_embeddings=self.anomaly_ctx,
            device=device,
        )

        normal_prompt = model.encode_text_prompted(prompted_normal, tokenized_normal, normalize=True)
        anomaly_prompt = model.encode_text_prompted(prompted_anomaly, tokenized_anomaly, normalize=True)
        normal_delta = self.text_delta_adapter(normal_prompt)
        anomaly_delta = self.text_delta_adapter(anomaly_prompt)

        normal_proto = F.normalize(static_normal + normal_delta.expand_as(static_normal), dim=-1)
        anomaly_proto = F.normalize(static_anomaly + anomaly_delta.expand_as(static_anomaly), dim=-1)
        text_static_reg = (
            2.0
            - (normal_proto * F.normalize(static_normal, dim=-1)).sum(dim=-1)
            - (anomaly_proto * F.normalize(static_anomaly, dim=-1)).sum(dim=-1)
        ).mean()
        return normal_proto, anomaly_proto, text_static_reg

    def _limit_descriptors(self, descriptors: Sequence[str]) -> List[str]:
        descriptors = list(descriptors)
        if self.max_prompts_per_state is None or self.max_prompts_per_state <= 0:
            return descriptors
        return descriptors[: self.max_prompts_per_state]

    def _resolve_descriptors(
        self,
        obj_type: str,
        text_inputs: Optional[Union[Dict[str, object], Sequence[object]]] = None,
        index: Optional[int] = None,
    ) -> Tuple[List[str], List[str]]:
        if text_inputs is None:
            return get_texts(obj_type.replace("_", " "))

        if isinstance(text_inputs, dict):
            if "normal" in text_inputs and "anomaly" in text_inputs:
                return list(text_inputs["normal"]), list(text_inputs["anomaly"])
            if obj_type in text_inputs:
                per_type = text_inputs[obj_type]
                return list(per_type["normal"]), list(per_type["anomaly"])

        if isinstance(text_inputs, Sequence) and index is not None:
            sample_item = text_inputs[index]
            if isinstance(sample_item, dict):
                return list(sample_item["normal"]), list(sample_item["anomaly"])
            if isinstance(sample_item, (tuple, list)) and len(sample_item) == 2:
                return list(sample_item[0]), list(sample_item[1])

        return get_texts(obj_type.replace("_", " "))

    def _build_single_prototype(
        self,
        model: "InCTRLWithAdapters",
        obj_type: str,
        device: torch.device,
        text_inputs: Optional[Union[Dict[str, object], Sequence[object]]] = None,
        index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normal_descriptors, anomaly_descriptors = self._resolve_descriptors(obj_type, text_inputs, index)
        normal_descriptors = self._limit_descriptors(normal_descriptors)
        anomaly_descriptors = self._limit_descriptors(anomaly_descriptors)

        prompted_normal, tokenized_normal = self._build_prompt_embeddings(
            model=model,
            descriptors=normal_descriptors,
            context_embeddings=self.normal_ctx,
            device=device,
        )
        prompted_anomaly, tokenized_anomaly = self._build_prompt_embeddings(
            model=model,
            descriptors=anomaly_descriptors,
            context_embeddings=self.anomaly_ctx,
            device=device,
        )

        normal_features = model.encode_text_prompted(prompted_normal, tokenized_normal, normalize=True)
        anomaly_features = model.encode_text_prompted(prompted_anomaly, tokenized_anomaly, normalize=True)

        normal_proto = self.prototype_adapter(normal_features.mean(dim=0, keepdim=True)).squeeze(0)
        anomaly_proto = self.prototype_adapter(anomaly_features.mean(dim=0, keepdim=True)).squeeze(0)
        return normal_proto, anomaly_proto

    def build_prototypes(
        self,
        model: "InCTRLWithAdapters",
        obj_types: Sequence[str],
        device: torch.device,
        text_inputs: Optional[Union[Dict[str, object], Sequence[object]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normal_prototypes = []
        anomaly_prototypes = []

        if text_inputs is None or isinstance(text_inputs, dict):
            unique_obj_types = []
            inverse_indices = []
            unique_index = {}
            for obj_type in obj_types:
                obj_key = str(obj_type)
                if obj_key not in unique_index:
                    unique_index[obj_key] = len(unique_obj_types)
                    unique_obj_types.append(obj_type)
                inverse_indices.append(unique_index[obj_key])

            for obj_type in unique_obj_types:
                normal_proto, anomaly_proto = self._build_single_prototype(
                    model=model,
                    obj_type=obj_type,
                    device=device,
                    text_inputs=text_inputs,
                )
                normal_prototypes.append(normal_proto)
                anomaly_prototypes.append(anomaly_proto)

            normal_unique = torch.stack(normal_prototypes, dim=0)
            anomaly_unique = torch.stack(anomaly_prototypes, dim=0)
            inverse = torch.tensor(inverse_indices, device=device, dtype=torch.long)
            return normal_unique.index_select(0, inverse), anomaly_unique.index_select(0, inverse)

        for index, obj_type in enumerate(obj_types):
            normal_proto, anomaly_proto = self._build_single_prototype(
                model=model,
                obj_type=obj_type,
                device=device,
                text_inputs=text_inputs,
                index=index,
            )
            normal_prototypes.append(normal_proto)
            anomaly_prototypes.append(anomaly_proto)

        normal_prototypes = torch.stack(normal_prototypes, dim=0)
        anomaly_prototypes = torch.stack(anomaly_prototypes, dim=0)
        return normal_prototypes, anomaly_prototypes


class PromptQueryAdapter(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        beta: float = 1.0,
        gamma_r: float = 0.5,
        gamma_c: float = 0.5,
        learnable_layer_weights: bool = False,
        global_topk: int = 10,
        image_size: int = 240,
    ):
        super().__init__()
        self.beta = beta
        self.gamma_r = gamma_r
        self.gamma_c = gamma_c
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
            PQAGlobalHead(feature_dim, hidden_dim) for _ in range(num_layers)
        )
        if learnable_layer_weights:
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        else:
            self.register_parameter("layer_weights", None)

    def _get_layer_weights(self, device: torch.device) -> torch.Tensor:
        if self.layer_weights is None:
            return torch.ones(len(self.local_heads), device=device) / len(self.local_heads)
        return torch.softmax(self.layer_weights, dim=0)

    def forward(
        self,
        query_patch_levels: Sequence[torch.Tensor],
        prompt_patch_levels: Sequence[torch.Tensor],
        beta: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        patch_evidence_maps = []
        patch_logits = []
        patch_scores = []
        pqa_global_logits = []
        pqa_global_logits_2c = []
        pqa_global_scores = []
        pqa_local_logits = []
        pqa_local_scores = []
        residual_maps = []
        context_scores = []
        aligned_indices = []
        aligned_prompt_features = []
        beta_value = self.beta if beta is None else beta

        for layer_idx, (query_level, prompt_level) in enumerate(zip(query_patch_levels, prompt_patch_levels)):
            batch_size, num_patches, dim = query_level.shape
            prompt_flat = prompt_level.reshape(batch_size, -1, dim)

            query_norm = F.normalize(query_level, dim=-1)
            prompt_norm = F.normalize(prompt_flat, dim=-1)
            similarity = torch.einsum("bnc,bmc->bnm", query_norm, prompt_norm)
            max_cosine, best_indices = similarity.max(dim=-1)
            residual = 1.0 - max_cosine

            gather_index = best_indices.unsqueeze(-1).expand(-1, -1, dim)
            aligned_prompt = torch.gather(prompt_flat, 1, gather_index)
            context_feat = query_level + beta_value * torch.abs(query_level - aligned_prompt)
            grid_side = int(num_patches ** 0.5)
            context_map = context_feat.permute(0, 2, 1).reshape(batch_size, dim, grid_side, grid_side)
            context_map = self.sharebn[layer_idx](context_map)

            local_logit_2c = self.local_heads[layer_idx](context_map)
            local_logit_2c = F.interpolate(
                local_logit_2c,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            local_score_2c = torch.softmax(local_logit_2c, dim=1)
            patch_logit_2c = F.interpolate(
                local_logit_2c,
                size=(grid_side, grid_side),
                mode="bilinear",
                align_corners=False,
            )
            patch_score_2c = torch.softmax(patch_logit_2c, dim=1)
            patch_logit = (patch_logit_2c[:, 1] - patch_logit_2c[:, 0]).flatten(1)
            context_score = patch_score_2c[:, 1].flatten(1)
            patch_evidence = self.gamma_r * residual + self.gamma_c * context_score
            fusion_img_feat = context_map.reshape(batch_size, dim, -1)
            topk = min(self.global_topk, fusion_img_feat.shape[-1])
            topk_feat = fusion_img_feat.topk(topk, dim=-1).values.mean(dim=-1)
            global_feat = 0.5 * (fusion_img_feat.mean(dim=-1) + topk_feat)
            pqa_global_logit_2c = self.global_heads[layer_idx](global_feat)
            pqa_global_logit = pqa_global_logit_2c[:, 1] - pqa_global_logit_2c[:, 0]

            patch_evidence_maps.append(patch_evidence)
            patch_logits.append(patch_logit)
            patch_scores.append(context_score)
            pqa_global_logits.append(pqa_global_logit)
            pqa_global_logits_2c.append(pqa_global_logit_2c)
            pqa_global_scores.append(torch.sigmoid(pqa_global_logit))
            pqa_local_logits.append(local_logit_2c)
            pqa_local_scores.append(local_score_2c)
            residual_maps.append(residual)
            context_scores.append(context_score)
            aligned_indices.append(best_indices)
            aligned_prompt_features.append(aligned_prompt)

        return {
            "patch_evidence_maps": patch_evidence_maps,
            "patch_logits": patch_logits,
            "patch_scores": patch_scores,
            "pqa_global_logits": pqa_global_logits,
            "pqa_global_logits_2c": pqa_global_logits_2c,
            "pqa_global_scores": pqa_global_scores,
            "pqa_local_logits": pqa_local_logits,
            "pqa_local_scores": pqa_local_scores,
            "residual_maps": residual_maps,
            "context_scores": context_scores,
            "aligned_indices": aligned_indices,
            "aligned_prompt_features": aligned_prompt_features,
            "layer_weights": self._get_layer_weights(query_patch_levels[0].device),
        }


class InCTRLWithAdapters(nn.Module):
    """
    InCTRL backbone with AdaptCLIP-style Textual / Visual / Prompt-Query adapters.

    The architecture intentionally stays InCTRL-shaped:
    - multi-layer patch residual learning remains the main local anomaly signal
    - image residual learning remains a dedicated global branch
    - text prior remains an auxiliary branch instead of a separate classifier
    - all branches fuse through one holistic scoring head plus max-patch compensation

    Alternating learning is exposed through set_train_phase():
    optimizing text and visual adapters together can cause mutual interference,
    overfit faster on the auxiliary training domain, and distort CLIP's original
    vision-language alignment. The phase API lets the training script alternate
    updates explicitly while forward() stays identical across phases.
    """

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
        gamma_r: float = 0.5,
        gamma_c: float = 0.5,
        lambda_g: float = 1.0,
        lambda_t: float = 1.0,
        lambda_p: float = 1.0,
        alpha: float = 0.5,
        use_text_adapter: Optional[bool] = None,
        use_visual_adapter: Optional[bool] = None,
        use_prompt_query_adapter: Optional[bool] = None,
        use_pqa_in_final_map: Optional[bool] = None,
        use_branch_fusion: Optional[bool] = None,
        use_max_patch_fallback: Optional[bool] = None,
        final_score_mode: Optional[str] = None,
        fusion_mode: Optional[str] = None,
        text_adapter_ctx_len: int = 12,
        adapter_hidden_dim: int = 256,
        patch_has_cls_token: bool = True,
        feature_is_projected: bool = True,
        learnable_layer_weights: Optional[bool] = None,
        visual_adapter_local_per_layer: Optional[bool] = None,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.embed_dim = embed_dim
        self.patch_layers = tuple(patch_layers)
        self.gamma_r = gamma_r
        self.gamma_c = gamma_c
        self.patch_has_cls_token = patch_has_cls_token
        self.feature_is_projected = feature_is_projected

        self.vision_cfg = _as_cfg(vision_cfg, CLIPVisionCfg)
        self.text_cfg = _as_cfg(text_cfg, CLIPTextCfg)
        self.image_size = getattr(args, "image_size", self.vision_cfg.image_size)
        self.shot = getattr(args, "shot", 1)
        textual_adapter_cfg = getattr(args, "TEXTUAL_ADAPTER", None)
        inctrl_adapter_cfg = getattr(args, "INCTRL_ADAPTER", None)
        if use_text_adapter is None:
            use_text_adapter = bool(getattr(textual_adapter_cfg, "ENABLE", False))
        if use_visual_adapter is None:
            use_visual_adapter = True
        if use_prompt_query_adapter is None:
            use_prompt_query_adapter = True
        if use_pqa_in_final_map is None:
            use_pqa_in_final_map = bool(
                getattr(inctrl_adapter_cfg, "USE_PQA_IN_FINAL_MAP", False)
            )
        if use_branch_fusion is None:
            use_branch_fusion = bool(getattr(inctrl_adapter_cfg, "USE_BRANCH_FUSION", True))
        if use_max_patch_fallback is None:
            use_max_patch_fallback = bool(
                getattr(inctrl_adapter_cfg, "USE_MAX_PATCH_FALLBACK", True)
            )
        if final_score_mode is None:
            final_score_mode = getattr(inctrl_adapter_cfg, "FINAL_SCORE_MODE", "raw_max_patch")
        final_score_mode = str(final_score_mode).lower()
        if final_score_mode not in {"raw_max_patch", "branch_fusion", "base"}:
            raise ValueError("final_score_mode must be one of: raw_max_patch, branch_fusion, base")
        if learnable_layer_weights is None:
            learnable_layer_weights = bool(
                getattr(inctrl_adapter_cfg, "LEARNABLE_LAYER_WEIGHTS", True)
            )
        if visual_adapter_local_per_layer is None:
            visual_adapter_local_per_layer = bool(
                getattr(inctrl_adapter_cfg, "VISUAL_LOCAL_PER_LAYER", True)
            )
        pqa_global_topk = int(getattr(inctrl_adapter_cfg, "PQA_GLOBAL_TOPK", 10))
        if fusion_mode is None:
            fusion_mode = getattr(inctrl_adapter_cfg, "FUSION_MODE", "paper_additive")
        fusion_mode = str(fusion_mode).lower()
        if fusion_mode not in {"inctrl", "paper_additive", "legacy_hybrid"}:
            raise ValueError(
                "fusion_mode must be one of: inctrl, paper_additive, legacy_hybrid"
            )

        self.alpha_raw = nn.Parameter(_inverse_softplus(alpha))
        self.lambda_g_raw = nn.Parameter(_inverse_softplus(lambda_g))
        self.lambda_t_raw = nn.Parameter(_inverse_softplus(lambda_t))
        self.lambda_p_raw = nn.Parameter(_inverse_softplus(lambda_p))
        self.beta_raw = nn.Parameter(_inverse_softplus(beta))
        self.branch_logits = nn.Parameter(torch.tensor([0.0, -2.0, -4.0, 4.0], dtype=torch.float32))
        if learnable_layer_weights:
            self.layer_logits = nn.Parameter(torch.zeros(len(self.patch_layers)))
        else:
            self.register_buffer("layer_logits", torch.zeros(len(self.patch_layers)))
        self.use_text_adapter = use_text_adapter
        self.use_visual_adapter = use_visual_adapter
        self.use_prompt_query_adapter = use_prompt_query_adapter
        self.use_pqa_in_final_map = use_pqa_in_final_map
        self.use_branch_fusion = use_branch_fusion
        self.use_max_patch_fallback = use_max_patch_fallback
        self.final_score_mode = final_score_mode
        self.fusion_mode = fusion_mode
        max_prompts_per_state = getattr(
            textual_adapter_cfg,
            "MAX_PROMPTS_PER_STATE",
            32,
        )
        text_adapter_mode = getattr(textual_adapter_cfg, "MODE", "static_residual")

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
        raw_patch_dim = self.vision_cfg.width
        if feature_is_projected and raw_patch_dim == embed_dim:
            self.patch_projection = nn.Identity()
            patch_feature_dim = embed_dim
        else:
            self.patch_projection = nn.Linear(raw_patch_dim, embed_dim)
            patch_feature_dim = embed_dim

        self.textual_adapter = TextualAdapter(
            context_dim=self.text_cfg.width,
            feature_dim=embed_dim,
            context_length=text_adapter_ctx_len,
            hidden_dim=adapter_hidden_dim,
            max_prompts_per_state=max_prompts_per_state,
            mode=text_adapter_mode,
        )
        self.visual_adapter = VisualAdapter(
            global_dim=embed_dim,
            local_dim=patch_feature_dim,
            hidden_dim=adapter_hidden_dim,
            num_layers=len(self.patch_layers),
            local_per_layer=visual_adapter_local_per_layer,
        )
        self.prompt_query_adapter = PromptQueryAdapter(
            feature_dim=patch_feature_dim,
            hidden_dim=adapter_hidden_dim,
            num_layers=len(self.patch_layers),
            beta=beta,
            gamma_r=gamma_r,
            gamma_c=gamma_c,
            learnable_layer_weights=learnable_layer_weights,
            global_topk=pqa_global_topk,
            image_size=self.image_size,
        )
        self.image_head = ImageResidualHead(embed_dim, adapter_hidden_dim)
        self.text_prior_head = TextPriorHead()
        self.holistic_head = HolisticScoringHead(self.num_patches, adapter_hidden_dim)

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

        self.set_train_phase("visual")

    def _positive_scalar(self, raw_value: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw_value)

    def _get_layer_weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.softmax(self.layer_logits.to(device=device, dtype=dtype), dim=0)

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
            residual = 1.0 - max_cosine

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

    def _adapt_patch_levels(self, patch_levels: torch.Tensor) -> List[torch.Tensor]:
        level_list = [patch_levels[:, level_idx, :, :] for level_idx in range(patch_levels.shape[1])]
        if not self.use_visual_adapter:
            return level_list
        return self.visual_adapter.adapt_local(level_list)

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

    def _adapt_prompt_patch_levels(self, patch_levels: torch.Tensor) -> List[torch.Tensor]:
        adapted_levels = []
        for level_idx in range(patch_levels.shape[1]):
            level = patch_levels[:, level_idx, :, :, :]
            if self.use_visual_adapter:
                flat_level = level.reshape(-1, level.shape[-2], level.shape[-1])
                adapted = self.visual_adapter.adapt_local_level(flat_level, level_idx)
                adapted = adapted.reshape(level.shape[0], level.shape[1], level.shape[2], level.shape[3])
            else:
                adapted = level
            adapted_levels.append(adapted)
        return adapted_levels

    @torch.no_grad()
    def build_prompt_feature_cache(
        self,
        prompt_images: Optional[torch.Tensor] = None,
        normal_list: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    ) -> Dict[str, object]:
        """Pre-encode shared few-shot prompt images for fast evaluation.

        The cache stores prompt-side features after visual adaptation, so each
        category/shot only pays the prompt visual-tower cost once.
        """
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

        if self.use_visual_adapter:
            adapted_prompt_global = self.visual_adapter.adapt_global(prompt_global)
        else:
            adapted_prompt_global = prompt_global
        adapted_prompt_patch_levels = self._adapt_prompt_patch_levels(prompt_patch_levels)
        prompt_patch_level_list = self._as_prompt_level_list(prompt_patch_levels)

        return {
            "prompt_global": prompt_global.squeeze(0).detach(),
            "prompt_patch_levels": [
                level.squeeze(0).detach()
                for level in prompt_patch_level_list
            ],
            "adapted_prompt_global": adapted_prompt_global.squeeze(0).detach(),
            "adapted_prompt_patch_levels": [
                level.squeeze(0).detach()
                for level in adapted_prompt_patch_levels
            ],
            "num_shots": num_shots,
        }

    def _build_adapted_text_prototypes(
        self,
        obj_types: Sequence[str],
        device: torch.device,
        text_inputs: Optional[Union[Dict[str, object], Sequence[object]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normal_proto, anomaly_proto, _ = self._build_text_prototypes_with_reg(
            obj_types=obj_types,
            device=device,
            text_inputs=text_inputs,
        )
        return normal_proto, anomaly_proto

    def _build_text_prototypes_with_reg(
        self,
        obj_types: Sequence[str],
        device: torch.device,
        text_inputs: Optional[Union[Dict[str, object], Sequence[object]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            static_normal, static_anomaly = self._build_static_text_prototypes(obj_types, device, text_inputs)

        if not self.use_text_adapter:
            return static_normal, static_anomaly, static_normal.new_zeros(())
        if self.textual_adapter.mode == "descriptor_context":
            normal_proto, anomaly_proto = self.textual_adapter.build_prototypes(self, obj_types, device, text_inputs)
            return normal_proto, anomaly_proto, normal_proto.new_zeros(())
        return self.textual_adapter.build_static_residual_prototypes(
            model=self,
            static_normal=static_normal,
            static_anomaly=static_anomaly,
            device=device,
        )

    def _build_static_text_prototypes(
        self,
        obj_types: Sequence[str],
        device: torch.device,
        text_inputs: Optional[Union[Dict[str, object], Sequence[object]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the fixed InCTRL / WinCLIP text prior without trainable prompts."""
        normal_prototypes = []
        anomaly_prototypes = []

        if text_inputs is None or isinstance(text_inputs, dict):
            unique_obj_types = []
            inverse_indices = []
            unique_index = {}
            for obj_type in obj_types:
                obj_key = str(obj_type)
                if obj_key not in unique_index:
                    unique_index[obj_key] = len(unique_obj_types)
                    unique_obj_types.append(obj_type)
                inverse_indices.append(unique_index[obj_key])

            for obj_type in unique_obj_types:
                normal_descriptors, anomaly_descriptors = self.textual_adapter._resolve_descriptors(
                    str(obj_type),
                    text_inputs,
                    None,
                )
                normal_prototypes.append(self._encode_text_descriptors(normal_descriptors, device))
                anomaly_prototypes.append(self._encode_text_descriptors(anomaly_descriptors, device))

            normal_stack = torch.stack(normal_prototypes, dim=0)
            anomaly_stack = torch.stack(anomaly_prototypes, dim=0)
            index_tensor = torch.tensor(inverse_indices, device=device, dtype=torch.long)
            return normal_stack[index_tensor], anomaly_stack[index_tensor]

        for index, obj_type in enumerate(obj_types):
            normal_descriptors, anomaly_descriptors = self.textual_adapter._resolve_descriptors(
                str(obj_type),
                text_inputs,
                index,
            )
            normal_prototypes.append(self._encode_text_descriptors(normal_descriptors, device))
            anomaly_prototypes.append(self._encode_text_descriptors(anomaly_descriptors, device))

        return torch.stack(normal_prototypes, dim=0), torch.stack(anomaly_prototypes, dim=0)

    @torch.no_grad()
    def build_text_prototype_cache(
        self,
        obj_types: Sequence[str],
        device: torch.device,
        text_inputs: Optional[Union[Dict[str, object], Sequence[object]]] = None,
    ) -> Dict[str, torch.Tensor]:
        normal_proto, anomaly_proto = self._build_adapted_text_prototypes(
            obj_types=obj_types,
            device=device,
            text_inputs=text_inputs,
        )
        return {
            "normal_proto": normal_proto.detach(),
            "anomaly_proto": anomaly_proto.detach(),
        }

    def get_visual_parameters(self) -> List[nn.Parameter]:
        modules = [
            self.patch_projection,
            self.image_head,
            self.holistic_head,
        ]
        if self.use_visual_adapter:
            modules.append(self.visual_adapter)
        if self.use_prompt_query_adapter:
            modules.append(self.prompt_query_adapter)
        parameters: List[nn.Parameter] = []
        for module in modules:
            parameters.extend(list(module.parameters()))
        parameters.extend([
            self.alpha_raw,
            self.lambda_g_raw,
            self.lambda_t_raw,
            self.lambda_p_raw,
            self.beta_raw,
            self.branch_logits,
        ])
        if isinstance(self.layer_logits, nn.Parameter):
            parameters.append(self.layer_logits)
        return parameters

    def get_text_parameters(self) -> List[nn.Parameter]:
        modules = [self.text_prior_head]
        if self.use_text_adapter:
            modules.append(self.textual_adapter)
        parameters: List[nn.Parameter] = []
        for module in modules:
            parameters.extend(list(module.parameters()))
        return parameters

    def set_train_phase(self, phase: str) -> None:
        if phase not in {"visual", "text", "joint"}:
            raise ValueError(f"Unsupported phase: {phase}")

        for parameter in self.get_visual_parameters():
            parameter.requires_grad = phase in {"visual", "joint"}
        for parameter in self.get_text_parameters():
            parameter.requires_grad = phase in {"text", "joint"}

        if not self.use_visual_adapter:
            for parameter in self.visual_adapter.parameters():
                parameter.requires_grad = False
        if not self.use_prompt_query_adapter:
            for parameter in self.prompt_query_adapter.parameters():
                parameter.requires_grad = False
        if not self.use_text_adapter:
            for parameter in self.textual_adapter.parameters():
                parameter.requires_grad = False

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

        self.train_phase = phase

    def forward(
        self,
        query_image: torch.Tensor,
        prompt_images: Optional[torch.Tensor] = None,
        normal_list: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        prompt_feature_cache: Optional[Dict[str, object]] = None,
        obj_types: Optional[Sequence[str]] = None,
        text_inputs: Optional[Union[Dict[str, object], Sequence[object]]] = None,
        text_prototype_cache: Optional[Dict[str, torch.Tensor]] = None,
        return_aux: bool = False,
        return_dict: bool = True,
    ) -> Dict[str, object]:
        if obj_types is None:
            obj_types = ["object"] * query_image.size(0)

        query_global, query_patch_tokens = self._encode_visual_features(query_image)
        batch_size = query_image.size(0)
        query_patch_levels = self._prepare_patch_levels(query_patch_tokens, batch_size=batch_size, num_shots=1)

        if self.use_visual_adapter:
            adapted_query_global = self.visual_adapter.adapt_global(query_global)
        else:
            adapted_query_global = query_global

        adapted_query_patch_levels = self._adapt_patch_levels(query_patch_levels)

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
            adapted_prompt_global = prompt_feature_cache["adapted_prompt_global"].to(
                query_image.device,
                dtype=adapted_query_global.dtype,
            )
            adapted_prompt_global = adapted_prompt_global.unsqueeze(0).expand(batch_size, -1, -1)
            adapted_prompt_patch_levels = [
                level.to(query_image.device, dtype=adapted_query_patch_levels[0].dtype)
                .unsqueeze(0)
                .expand(batch_size, -1, -1, -1)
                for level in prompt_feature_cache["adapted_prompt_patch_levels"]
            ]
        else:
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

            if self.use_visual_adapter:
                adapted_prompt_global = self.visual_adapter.adapt_global(prompt_global)
            else:
                adapted_prompt_global = prompt_global
            adapted_prompt_patch_levels = self._adapt_prompt_patch_levels(prompt_patch_levels)

        layer_weights = self._get_layer_weights(
            device=query_image.device,
            dtype=adapted_query_patch_levels[0].dtype,
        )
        raw_query_patch_levels = self._as_query_level_list(query_patch_levels)
        raw_prompt_patch_levels = self._as_prompt_level_list(prompt_patch_levels)
        raw_residual_outputs = self._compute_patch_residuals(
            query_patch_levels=raw_query_patch_levels,
            prompt_patch_levels=raw_prompt_patch_levels,
        )
        raw_base_patch_map = sum(
            weight * residual
            for weight, residual in zip(layer_weights, raw_residual_outputs["residual_maps"])
        )
        beta_value = self._positive_scalar(self.beta_raw).to(
            device=query_image.device,
            dtype=adapted_query_patch_levels[0].dtype,
        )
        if self.use_prompt_query_adapter:
            pq_outputs = self.prompt_query_adapter(
                query_patch_levels=adapted_query_patch_levels,
                prompt_patch_levels=adapted_prompt_patch_levels,
                beta=beta_value,
            )
            hybrid_patch_map = sum(
                weight * evidence
                for weight, evidence in zip(layer_weights, pq_outputs["patch_evidence_maps"])
            )
            pqa_patch_logit = sum(
                weight * patch_logit
                for weight, patch_logit in zip(layer_weights, pq_outputs["patch_logits"])
            )
            pqa_logit = sum(
                weight * global_logit
                for weight, global_logit in zip(layer_weights, pq_outputs["pqa_global_logits"])
            )
            pqa_local_logits = sum(
                weight * local_logit
                for weight, local_logit in zip(layer_weights, pq_outputs["pqa_local_logits"])
            )
        else:
            residual_outputs = self._compute_patch_residuals(
                query_patch_levels=adapted_query_patch_levels,
                prompt_patch_levels=adapted_prompt_patch_levels,
            )
            pq_outputs = {
                "patch_evidence_maps": [],
                "patch_logits": [],
                "patch_scores": [],
                "pqa_global_logits": [],
                "pqa_global_logits_2c": [],
                "pqa_global_scores": [],
                "pqa_local_logits": [],
                "pqa_local_scores": [],
                "residual_maps": residual_outputs["residual_maps"],
                "context_scores": [],
                "aligned_indices": residual_outputs["aligned_indices"],
                "aligned_prompt_features": residual_outputs["aligned_prompt_features"],
                "layer_weights": layer_weights,
            }
            hybrid_patch_map = None
            pqa_patch_logit = torch.zeros_like(pq_outputs["residual_maps"][0])
            pqa_logit = torch.zeros(batch_size, device=query_image.device, dtype=adapted_query_global.dtype)
            pqa_local_logits = torch.zeros(
                batch_size,
                2,
                self.image_size,
                self.image_size,
                device=query_image.device,
                dtype=adapted_query_global.dtype,
            )
        base_patch_map = sum(
            weight * residual
            for weight, residual in zip(layer_weights, pq_outputs["residual_maps"])
        )
        if hybrid_patch_map is None:
            hybrid_patch_map = base_patch_map
        pqa_patch_score = torch.sigmoid(pqa_patch_logit)
        pqa_score = torch.sigmoid(pqa_logit)
        pqa_local_scores = torch.softmax(pqa_local_logits, dim=1)
        final_patch_map = hybrid_patch_map if self.use_pqa_in_final_map else base_patch_map

        prompt_global_proto = adapted_prompt_global.mean(dim=1)
        image_residual = adapted_query_global - prompt_global_proto
        image_logit = self.image_head(image_residual)
        image_score = torch.sigmoid(image_logit)

        text_static_reg = image_logit.new_zeros(())
        if text_prototype_cache is not None:
            normal_proto = text_prototype_cache["normal_proto"].to(
                query_image.device,
                dtype=adapted_query_global.dtype,
            )
            anomaly_proto = text_prototype_cache["anomaly_proto"].to(
                query_image.device,
                dtype=adapted_query_global.dtype,
            )
            if normal_proto.size(0) == 1 and batch_size != 1:
                normal_proto = normal_proto.expand(batch_size, -1)
                anomaly_proto = anomaly_proto.expand(batch_size, -1)
        else:
            normal_proto, anomaly_proto, text_static_reg = self._build_text_prototypes_with_reg(
                obj_types=obj_types,
                device=query_image.device,
                text_inputs=text_inputs,
            )
        text_logit, text_aux = self.text_prior_head(adapted_query_global, normal_proto, anomaly_proto)
        text_score = torch.sigmoid(text_logit)

        alpha = self._positive_scalar(self.alpha_raw)
        lambda_g = self._positive_scalar(self.lambda_g_raw)
        lambda_t = self._positive_scalar(self.lambda_t_raw)
        lambda_p = self._positive_scalar(self.lambda_p_raw)

        holistic_input = (
            base_patch_map
            + lambda_g * image_score.unsqueeze(-1)
            + lambda_t * text_score.unsqueeze(-1)
            + lambda_p * pqa_score.unsqueeze(-1)
        )
        holistic_logit = self.holistic_head(holistic_input)
        holistic_score = torch.sigmoid(holistic_logit)
        max_base_patch_score = base_patch_map.max(dim=-1).values
        raw_max_patch_score = raw_base_patch_map.max(dim=-1).values
        max_hybrid_patch_score = hybrid_patch_map.max(dim=-1).values
        max_patch_score = final_patch_map.max(dim=-1).values

        if self.fusion_mode == "inctrl":
            base_logit = 0.5 * (holistic_logit + max_base_patch_score)
        elif self.fusion_mode == "paper_additive":
            base_logit = holistic_logit + alpha * max_base_patch_score
        else:
            base_logit = holistic_logit + alpha * max_patch_score
        max_patch_logit = alpha * max_base_patch_score
        raw_max_patch_logit = alpha * raw_max_patch_score

        branch_weights = torch.softmax(self.branch_logits, dim=0)
        if self.final_score_mode == "raw_max_patch":
            final_logit = raw_max_patch_logit
        elif self.final_score_mode == "base":
            final_logit = base_logit
        elif self.use_branch_fusion:
            if self.use_max_patch_fallback:
                final_logit = (
                    branch_weights[0] * base_logit
                    + branch_weights[1] * text_logit
                    + branch_weights[2] * pqa_logit
                    + branch_weights[3] * max_patch_logit
                )
            else:
                branch_weights_3 = torch.softmax(self.branch_logits[:3], dim=0)
                final_logit = (
                    branch_weights_3[0] * base_logit
                    + branch_weights_3[1] * text_logit
                    + branch_weights_3[2] * pqa_logit
                )
        else:
            final_logit = base_logit
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
                "per_layer_patch_evidence": pq_outputs["patch_evidence_maps"],
                "per_layer_residual": pq_outputs["residual_maps"],
                "per_layer_raw_residual": raw_residual_outputs["residual_maps"],
                "per_layer_context_score": pq_outputs["context_scores"],
                "aligned_indices": pq_outputs["aligned_indices"],
                "adapted_query_global": adapted_query_global,
                "prompt_global_proto": prompt_global_proto,
                "image_residual": image_residual,
                "layer_weights": layer_weights,
                "branch_weights": branch_weights,
                "text_prototypes": {
                    "normal": normal_proto,
                    "anomaly": anomaly_proto,
                },
                "text_logits": text_aux,
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
            "max_hybrid_patch_score": max_hybrid_patch_score,
            "branch_weights": branch_weights,
            "text_static_reg": text_static_reg,
            "aux": aux,
        }
        return result


__all__ = [
    "ContextResidualPatchHead",
    "HolisticScoringHead",
    "ImageResidualHead",
    "InCTRLWithAdapters",
    "PromptQueryAdapter",
    "TextPriorHead",
    "TextualAdapter",
    "VisualAdapter",
]
