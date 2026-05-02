import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .model import _build_vision_tower_Mul, _build_text_tower
from .adaptclip_textual_adapter import AdaptCLIPTextualAdapter
from .pqa_adapter import PQAdapter
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
    visual_logit: torch.Tensor | None = None,
    visual_weight: float = 0.0,
) -> torch.Tensor:
    image_w, patch_w, pqa_w, text_w = weights
    out = (
        image_w * image_logit
        + patch_w * patch_logit
        + pqa_w * pqa_logit
        + text_w * text_logit
    )
    if visual_logit is not None and visual_weight > 0.0:
        out = out + visual_weight * visual_logit
    return out


def _fuse_maps(
    residual_map: torch.Tensor,
    pqa_map: torch.Tensor,
    weights: tuple[float, float],
    text_map: torch.Tensor | None = None,
    text_w: float = 0.0,
    visual_map: torch.Tensor | None = None,
    visual_w: float = 0.0,
    pixel_fusion: str = "weighted",
) -> torch.Tensor:
    maps = [residual_map, pqa_map]
    if text_map is not None and text_w > 0.0:
        maps.append(text_map)
    if visual_map is not None and visual_w > 0.0:
        maps.append(visual_map)
    if pixel_fusion != "weighted":
        return _fuse_tensor_list(maps, mode=pixel_fusion)

    residual_w, pqa_w = weights
    out = residual_w * residual_map + pqa_w * pqa_map
    if text_map is not None and text_w > 0.0:
        out = out + text_w * text_map
    if visual_map is not None and visual_w > 0.0:
        out = out + visual_w * visual_map
    return out


def _harmonic_mean(tensor_list: list[torch.Tensor]) -> torch.Tensor:
    """Harmonic mean: penalizes low scores more than arithmetic mean.

    Any branch can 'veto' a normal prediction, which is desirable for
    anomaly detection where missing a defect is costly.
    """
    stacked = torch.stack(tensor_list, dim=0)  # [N, ...]
    stacked = stacked.clamp(min=1e-6)
    return stacked.shape[0] / (1.0 / stacked).sum(dim=0)


def _average_mean(tensor_list: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(tensor_list, dim=0).mean(dim=0)


def _fuse_tensor_list(tensor_list: list[torch.Tensor], mode: str) -> torch.Tensor:
    if not tensor_list:
        raise ValueError("tensor_list must contain at least one tensor.")
    if len(tensor_list) == 1:
        return tensor_list[0]
    if mode == "average_mean":
        return _average_mean(tensor_list)
    if mode == "harmonic_mean":
        return _harmonic_mean(tensor_list)
    raise ValueError(f"Unsupported fusion mode: {mode}")


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


class InCTRLAdapt(nn.Module):
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
        self.use_visual_branch = bool(getattr(args.FUSION, "USE_VISUAL_BRANCH", True)) if self.use_visual_adapter else False
        self.use_text_branch = bool(getattr(args.TEXT_BRANCH, "ENABLE", True))
        self.use_pqa = bool(getattr(args.PQA, "ENABLE", True))
        self.use_seg_head = bool(getattr(args.PQA, "ENABLE_SEG_HEAD", True))
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
        self.pq_adapter = PQAdapter(
            dim=patch_dim,
            hidden_dim=hidden_dim,
            image_size=self.image_size,
            num_layers=len(self.patch_layers),
            topk=int(getattr(args.PQA, "GLOBAL_TOPK", 10)),
            beta=float(getattr(args.PQA, "CONTEXT_BETA", 1.0)),
            enable_seg_head=self.use_seg_head,
        )
        self.patch_text_projection = (
            nn.Identity() if patch_dim == embed_dim else nn.Linear(patch_dim, embed_dim)
        )
        self.text_branch = AdaptCLIPTextualAdapter(
            ctx_dim=self.ln_final.weight.shape[0],
            image_size=self.image_size,
            n_ctx=int(getattr(args.TEXT_BRANCH, "N_CTX", 12)),
            normal_suffix=str(getattr(args.TEXT_BRANCH, "NORMAL_SUFFIX", "normal object.")),
            abnormal_suffix=str(getattr(args.TEXT_BRANCH, "ABNORMAL_SUFFIX", "damaged object.")),
            init_std=float(getattr(args.TEXT_BRANCH, "CTX_INIT_STD", 0.02)),
            logit_scale=float(getattr(args.TEXT_BRANCH, "LOGIT_SCALE", 100.0)),
            dtype=cast_dtype,
        )

        # Static text features for the visual-text branch (frozen, not learnable).
        # These are the AdaptCLIP "normal"/"anomaly" template embeddings.
        self._static_text_features: torch.Tensor | None = None
        self._visual_logit_scale = float(getattr(args.TEXT_BRANCH, "LOGIT_SCALE", 100.0))

    def get_visual_parameters(self):
        params = []
        if self.use_visual_adapter:
            params.extend(self.visual_adapter.parameters())
        params.extend(self.image_head.parameters())
        if self.use_pqa:
            params.extend(self.pq_adapter.parameters())
        if self.patch_text_projection is not None:
            params.extend(self.patch_text_projection.parameters())
        return params

    def _visual_branch_is_active(self) -> bool:
        if not self.use_visual_branch:
            return False
        fusion_visual_w = float(getattr(self.args.FUSION, "VISUAL_WEIGHT", 0.0))
        loss_visual_w = float(getattr(self.args.LOSS, "VISUAL_WEIGHT", 0.0))
        loss_visual_mask_w = float(getattr(self.args.LOSS, "VISUAL_MASK_WEIGHT", 0.0))
        return fusion_visual_w > 0.0 or loss_visual_w > 0.0 or loss_visual_mask_w > 0.0

    def _project_patch_to_text_dim(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return self.patch_text_projection(patch_tokens)

    def _get_static_text_features(self, tokenizer, device) -> torch.Tensor:
        """Build or return cached static text features for VA branch."""
        if self._static_text_features is not None:
            return self._static_text_features.to(device)
        templates = list(getattr(self.args.TEXT_BRANCH, "TEMPLATES", [
            "a photo of a normal object.",
            "a photo of a damaged object.",
        ]))
        tokens = torch.cat([tokenizer(t) for t in templates]).to(device)
        with torch.no_grad():
            feats = self.encode_text(tokens, normalize=True)  # [2, D]
        self._static_text_features = feats.detach()
        return self._static_text_features

    def _visual_branch(
        self,
        query_global_va: torch.Tensor,
        query_patch_va: torch.Tensor,
        static_text: torch.Tensor,
    ) -> dict:
        """AdaptCLIP-style visual-text alignment branch.

        Uses VA-adapted visual features + frozen static text features to
        compute image-level logit and pixel-level anomaly map.
        """
        batch = query_global_va.shape[0]
        global_feat = F.normalize(query_global_va, dim=-1)
        patch_feat = F.normalize(query_patch_va, dim=-1)
        # static_text: [2, D] → [B, 2, D]
        text_feat = static_text.unsqueeze(0).expand(batch, -1, -1)

        # Image-level
        logits = self._visual_logit_scale * torch.einsum("bd,bcd->bc", global_feat, text_feat)
        visual_logit = logits[:, 1] - logits[:, 0]  # anomaly - normal
        visual_score = logits.softmax(dim=-1)[:, 1]

        # Patch-level
        patches = patch_feat.shape[1]
        patch_logits = self._visual_logit_scale * torch.einsum("bnd,bcd->bnc", patch_feat, text_feat)
        patch_logit = patch_logits[..., 1] - patch_logits[..., 0]
        grid = int(math.sqrt(patches))
        visual_map_logits = patch_logit.reshape(batch, 1, grid, grid)
        visual_map_logits = F.interpolate(
            visual_map_logits,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        visual_map = torch.sigmoid(visual_map_logits)

        return {
            "visual_logits": logits,
            "visual_logit": visual_logit,
            "visual_score": visual_score,
            "visual_map_logits": visual_map_logits,
            "visual_map": visual_map,
        }

    def _zero_visual_outputs(self, query_global: torch.Tensor, patch_tokens: torch.Tensor):
        visual_logit = query_global.new_zeros(query_global.shape[0])
        visual_map = self._upsample_patch_map(
            patch_tokens.new_zeros(patch_tokens.shape[0], patch_tokens.shape[1])
        )
        return {
            "visual_logits": None,
            "visual_logit": visual_logit,
            "visual_score": torch.sigmoid(visual_logit),
            "visual_map_logits": visual_map,
            "visual_map": visual_map,
        }

    def get_text_parameters(self):
        if not self.use_text_branch:
            return []
        return list(self.text_branch.parameters())

    def set_train_phase(self, phase: str):
        has_visual = len(self.get_visual_parameters()) > 0
        has_text = len(self.get_text_parameters()) > 0
        visual_grad = phase in {"visual", "joint"} or not has_text
        text_grad = phase in {"text", "joint"} or not has_visual
        for p in self.get_visual_parameters():
            p.requires_grad = visual_grad
        for p in self.get_text_parameters():
            p.requires_grad = text_grad

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

    def encode_text_prompted(
        self,
        prompts: torch.Tensor,
        tokenized_prompts: torch.Tensor,
        normalize: bool = False,
    ):
        cast_dtype = self.transformer.get_cast_dtype()
        x = prompts.to(cast_dtype) + self.positional_embedding.to(device=prompts.device, dtype=cast_dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        eot_indices = tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=x.device), eot_indices] @ self.text_projection
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
            "text_logits": None,
            "text_map_logits": text_map,
            "patch_text_logits": None,
            "text_features": None,
            "text_prototypes": None,
        }

    def _zero_pqa_outputs(self, query_tokens: torch.Tensor):
        batch = query_tokens.shape[0]
        pqa_logit = query_tokens.new_zeros(batch)
        pqa_patch_map = query_tokens.new_zeros(batch, query_tokens.shape[1])
        return {
            "pqa_seg_logits": query_tokens.new_zeros(batch, 2, self.image_size, self.image_size),
            "pqa_global_logits": query_tokens.new_zeros(batch, 2),
            "pqa_logit": pqa_logit,
            "pqa_score": torch.sigmoid(pqa_logit),
            "pqa_patch_map": pqa_patch_map,
        }

    def _pqa_map_from_logits(self, pqa_seg_logits: torch.Tensor) -> torch.Tensor:
        if not self.use_pqa:
            return pqa_seg_logits.new_zeros(
                pqa_seg_logits.shape[0],
                1,
                pqa_seg_logits.shape[-2],
                pqa_seg_logits.shape[-1],
            )
        if pqa_seg_logits.shape[1] == 2:
            return pqa_seg_logits.softmax(dim=1)[:, 1:2]
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

        # Frozen features for InCTRL residual/PQA path and TA.
        # VA output is ONLY used for the visual-text branch — never enters
        # image_head / _compute_patch_residual / pq_adapter.
        query_global_frozen = query_global
        query_patch_frozen = query_patch_levels

        if self.use_visual_adapter:
            query_global_va, query_patch_va = self.visual_adapter(
                query_global.detach(), [l.detach() for l in query_patch_levels]
            )
        else:
            query_global_va = query_global
            query_patch_va = query_patch_levels

        # InCTRL residual path uses frozen (un-adapted) features.
        image_score, image_logit = self.image_head(query_global_frozen, prompt_global)

        residual_maps = []
        align_maps = []
        pqa_maps = []
        pqa_logits = []
        pqa_global_logits_list = []
        pqa_seg_logits_per_layer = []
        for layer_idx, (query_tokens, prompt_tokens) in enumerate(
            zip(query_patch_frozen, prompt_patch_levels)
        ):
            prompt_flat_tokens = self._flatten_prompt_tokens(prompt_tokens)
            residual_map = self._compute_patch_residual(query_tokens, prompt_flat_tokens)
            pqa_out = (
                self.pq_adapter(query_tokens, prompt_flat_tokens, layer_idx=layer_idx)
                if self.use_pqa
                else self._zero_pqa_outputs(query_tokens)
            )
            residual_maps.append(residual_map)
            align_maps.append(self._upsample_patch_map(residual_map))
            pqa_maps.append(pqa_out["pqa_patch_map"])
            pqa_logits.append(pqa_out["pqa_logit"])
            pqa_global_logits_list.append(pqa_out.get("pqa_global_logits"))
            pqa_seg_logits_per_layer.append(pqa_out["pqa_seg_logits"])

        align_fusion = str(getattr(self.args.FUSION, "ALIGN_FUSION", "harmonic_mean"))
        align_score_map = _fuse_tensor_list(align_maps, mode=align_fusion)

        patch_residual_map = torch.stack(residual_maps, dim=0).mean(dim=0)
        patch_score = patch_residual_map.max(dim=-1).values
        patch_logit = _score_to_logit(patch_score)
        pqa_patch_map = torch.stack(pqa_maps, dim=0).mean(dim=0)
        pqa_logit = torch.stack(pqa_logits, dim=0).mean(dim=0)
        pqa_score = torch.sigmoid(pqa_logit)
        pqa_seg_logits = torch.stack(pqa_seg_logits_per_layer, dim=0).mean(dim=0)
        # PQA global logits: average across layers → [B, 2].
        pqa_global_logits = (
            torch.stack(pqa_global_logits_list, dim=0).mean(dim=0)
            if pqa_global_logits_list[0] is not None
            else None
        )

        # Visual-text branch: VA-adapted features + frozen static text.
        if self._visual_branch_is_active() and tokenizer is not None:
            static_text = self._get_static_text_features(tokenizer, query_global.device)
            visual_patch_feat = self._project_patch_to_text_dim(query_patch_va[-1])
            visual_out = self._visual_branch(
                query_global_va,
                visual_patch_feat,
                static_text,
            )
        else:
            visual_out = self._zero_visual_outputs(query_global_frozen, query_patch_frozen[-1])
        visual_logit = visual_out["visual_logit"]
        visual_score = visual_out["visual_score"]
        visual_map = visual_out["visual_map"]

        # Text branch uses frozen (un-adapted) visual features,
        # matching AdaptCLIP's alternating learning: TA fixes vision, learns text.
        text_patch_feat = self.patch_text_projection(query_patch_frozen[-1])
        if self.use_text_branch:
            text_out = self.text_branch(
                encode_text_prompted=self.encode_text_prompted,
                token_embedding=self.token_embedding,
                tokenizer=tokenizer,
                global_feat=query_global_frozen,
                patch_feat=text_patch_feat,
            )
        else:
            text_out = self._zero_text_outputs(query_global, text_patch_feat)
        text_logit = text_out["text_logit"]
        text_score = text_out["text_score"]
        text_map = text_out["text_map"]

        visual_weight = float(getattr(self.args.FUSION, "VISUAL_WEIGHT", 0.0))
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
            visual_logit=visual_logit,
            visual_weight=visual_weight,
        )
        final_score = torch.sigmoid(final_logit)

        # Pixel-level fusion: only include text/visual maps when their
        # corresponding mask loss weight > 0 (i.e. they have supervision).
        residual_map_up = self._upsample_patch_map(patch_residual_map)
        pqa_map = self._pqa_map_from_logits(pqa_seg_logits)
        text_mask_w = float(getattr(self.args.LOSS, "TEXT_MASK_WEIGHT", 0.0))
        visual_mask_w = float(getattr(self.args.LOSS, "VISUAL_MASK_WEIGHT", 0.0))
        map_text_w = float(getattr(self.args.FUSION, "MAP_TEXT_WEIGHT", 0.2)) if text_mask_w > 0 else 0.0
        map_visual_w = float(getattr(self.args.FUSION, "MAP_VISUAL_WEIGHT", 0.1)) if visual_mask_w > 0 else 0.0
        branch_map = _fuse_maps(
            residual_map_up,
            pqa_map,
            weights=(
                float(getattr(self.args.FUSION, "MAP_RES_WEIGHT", 0.4)),
                float(getattr(self.args.FUSION, "MAP_PQA_WEIGHT", 0.4)),
            ),
            text_map=text_map if map_text_w > 0 else None,
            text_w=map_text_w,
            visual_map=visual_map if map_visual_w > 0 else None,
            visual_w=map_visual_w,
            pixel_fusion=str(getattr(self.args.FUSION, "PIXEL_FUSION", "weighted")),
        )
        final_map = _fuse_tensor_list([branch_map, align_score_map], mode=align_fusion)

        # Image-pixel coupling: anomaly_map_max feeds back into image score
        # via harmonic_mean (AdaptCLIP design).
        anomaly_map_max, _ = final_map.view(batch, -1).max(dim=-1)
        if bool(getattr(self.args.FUSION, "IMAGE_PIXEL_COUPLING", True)):
            coupled_image_score = _harmonic_mean([final_score, anomaly_map_max])
        else:
            coupled_image_score = final_score

        outputs = {
            "final_score": final_score,
            "final_logit": final_logit,
            "coupled_score": coupled_image_score,
            "image_score": image_score,
            "image_logit": image_logit,
            "patch_score": patch_score,
            "patch_logit": patch_logit,
            "patch_residual_map": patch_residual_map,
            "pqa_score": pqa_score,
            "pqa_logit": pqa_logit,
            "pqa_global_logits": pqa_global_logits,
            "pqa_patch_map": pqa_patch_map,
            "pqa_seg_logits": pqa_seg_logits,
            "text_score": text_score,
            "text_logit": text_logit,
            "text_map": text_map,
            "text_logits": text_out.get("text_logits"),
            "patch_text_logits": text_out.get("patch_text_logits"),
            "text_map_logits": text_out.get("text_map_logits"),
            "text_features": text_out.get("text_features"),
            "visual_score": visual_score,
            "visual_logit": visual_logit,
            "visual_logits": visual_out.get("visual_logits"),
            "visual_map": visual_map,
            "visual_map_logits": visual_out.get("visual_map_logits"),
            "align_score_map": align_score_map,
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
