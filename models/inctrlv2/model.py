from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip
from open_clip.model import CLIPTextCfg, CLIPVisionCfg, _build_text_tower, _build_vision_tower_Mul, get_cast_dtype

from .dasl import DASLBranch
from .fusion import fuse_image_score, fuse_pixel_maps
from .oasl import OASLBranch
from .residual import ImageResidualLearner, compute_patch_residual, upsample_flat_map
from .text_prototypes import TextPrototypeBuilder


DEFAULT_SELECTED_LAYERS = [7, 9, 11]


class InCTRLv2Model(nn.Module):
    """InCTRLv2 with frozen OpenCLIP, residual learning, DASL, and OASL."""

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: dict | CLIPVisionCfg,
        text_cfg: dict | CLIPTextCfg,
        tokenizer,
        selected_layers: Optional[Iterable[int]] = None,
        alpha: float = 0.5,
        beta: float = 0.75,
        freeze_clip: bool = True,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        disable_dasl: bool = False,
        disable_oasl: bool = False,
        residual_scale: str = "half",
    ):
        super().__init__()
        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = CLIPTextCfg(**text_cfg)

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

        self.embed_dim = embed_dim
        self.patch_dim = vision_cfg.width
        self.text_dim = embed_dim
        self.image_size = vision_cfg.image_size[0] if isinstance(vision_cfg.image_size, tuple) else vision_cfg.image_size
        self.patch_size = vision_cfg.patch_size[0] if isinstance(vision_cfg.patch_size, tuple) else vision_cfg.patch_size
        self.selected_layers = list(selected_layers or DEFAULT_SELECTED_LAYERS)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.disable_dasl = bool(disable_dasl)
        self.disable_oasl = bool(disable_oasl)
        self.residual_scale = residual_scale

        self.residual = ImageResidualLearner(dim=self.text_dim)
        self.dasl = DASLBranch(patch_dim=self.patch_dim, text_dim=self.text_dim, selected_layers=self.selected_layers)
        self.oasl = OASLBranch(patch_dim=self.patch_dim, text_dim=self.text_dim, selected_layers=self.selected_layers)
        self.text_prototypes = TextPrototypeBuilder(tokenizer=tokenizer, encode_text=self.encode_text)

        if freeze_clip:
            self.freeze_clip()
        if self.disable_dasl:
            for param in self.dasl.parameters():
                param.requires_grad = False
        if self.disable_oasl:
            for param in self.oasl.parameters():
                param.requires_grad = False

    def freeze_clip(self) -> None:
        for param in self.visual.parameters():
            param.requires_grad = False
        text_modules = [
            self.transformer,
            self.token_embedding,
            self.ln_final,
        ]
        for module in text_modules:
            for param in module.parameters():
                param.requires_grad = False
        self.positional_embedding.requires_grad = False
        self.text_projection.requires_grad = False

    def encode_image(self, image: torch.Tensor):
        features = self.visual.forward(image, self.selected_layers)
        if not isinstance(features, tuple) or len(features) < 2:
            raise RuntimeError("Vision tower must return pooled features and multi-layer patch tokens")
        return features

    def encode_text(self, text: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    @staticmethod
    def _normalize_class_names(class_name, batch_size: int) -> List[str]:
        if isinstance(class_name, str):
            return [class_name] * batch_size
        if isinstance(class_name, tuple):
            class_name = list(class_name)
        if isinstance(class_name, list):
            if len(class_name) != batch_size:
                raise ValueError(f"class_name length {len(class_name)} does not match batch size {batch_size}")
            return [str(name) for name in class_name]
        raise TypeError(f"Unsupported class_name type: {type(class_name)!r}")

    def _encode_query_and_prompts(self, query_image: torch.Tensor, prompt_images: torch.Tensor) -> dict:
        if prompt_images.dim() == 4:
            prompt_images = prompt_images.unsqueeze(0)
        if prompt_images.dim() != 5:
            raise ValueError(f"prompt_images must be [B,K,C,H,W] or [K,C,H,W], got {tuple(prompt_images.shape)}")
        if prompt_images.shape[0] != query_image.shape[0]:
            if prompt_images.shape[1] == query_image.shape[0]:
                prompt_images = prompt_images.permute(1, 0, 2, 3, 4)
            else:
                raise ValueError("prompt_images batch dimension must match query_image")

        batch_size, shots = prompt_images.shape[:2]
        query_cls, query_patch_list, _ = self.encode_image(query_image)
        flat_prompts = prompt_images.reshape(batch_size * shots, *prompt_images.shape[2:])
        prompt_cls, prompt_patch_list, _ = self.encode_image(flat_prompts)
        prompt_cls = prompt_cls.reshape(batch_size, shots, -1)

        query_patches = []
        prompt_patches = []
        for query_layer, prompt_layer in zip(query_patch_list, prompt_patch_list):
            query_layer = query_layer[:, 1:, :]
            prompt_layer = prompt_layer[:, 1:, :]
            num_patches = query_layer.shape[1]
            prompt_layer = prompt_layer.reshape(batch_size, shots, num_patches, self.patch_dim)
            prompt_layer = prompt_layer.reshape(batch_size, shots * num_patches, self.patch_dim)
            query_patches.append(query_layer)
            prompt_patches.append(prompt_layer)

        return {
            "query_cls": query_cls,
            "prompt_cls": prompt_cls,
            "query_patches": query_patches,
            "prompt_patches": prompt_patches,
        }

    def forward_main(self, query_image: torch.Tensor, prompt_images: torch.Tensor, class_name) -> dict:
        features = self._encode_query_and_prompts(query_image, prompt_images)
        batch_size = query_image.shape[0]
        class_names = self._normalize_class_names(class_name, batch_size)
        prototypes = self.text_prototypes.build(class_names, device=query_image.device, dtype=features["query_cls"].dtype)

        image_residual = self.residual(features["query_cls"], features["prompt_cls"])
        residual_map, patch_score, _ = compute_patch_residual(
            features["query_patches"],
            features["prompt_patches"],
            residual_scale=self.residual_scale,
        )
        s_i = image_residual["s_I"]

        if self.disable_dasl:
            s_q = s_i
            s_n = (1.0 - residual_map).clamp(0.0, 1.0)
            s_a = residual_map.clamp(0.0, 1.0)
            pixel_map_dasl = residual_map.clamp(0.0, 1.0)
            image_score = fuse_image_score(s_i=s_i, s_q=s_q, s_p=patch_score, alpha=self.alpha)
        else:
            dasl_out = self.dasl(
                query_cls=features["query_cls"],
                query_patch_tokens=features["query_patches"],
                residual_map=residual_map,
                s_i=s_i,
                s_p=patch_score,
                normal_proto=prototypes["normal_proto"],
                abnormal_proto=prototypes["abnormal_proto"],
                alpha=self.alpha,
            )
            s_q = dasl_out["semantic_score"]
            s_n = dasl_out["S_n"]
            s_a = dasl_out["S_a"]
            pixel_map_dasl = dasl_out["pixel_map_dasl"]
            image_score = dasl_out["image_score"]

        dasl_map_up = upsample_flat_map(pixel_map_dasl, self.image_size)
        residual_map_up = upsample_flat_map(residual_map, self.image_size)
        return {
            "s_I": s_i,
            "s_q": s_q,
            "s_p": patch_score,
            "s_final": image_score.clamp(0.0, 1.0),
            "image_score": image_score.clamp(0.0, 1.0),
            "M_x": residual_map,
            "S_a": s_a,
            "S_n": s_n,
            "M_p": pixel_map_dasl,
            "pixel_map_dasl": pixel_map_dasl,
            "dasl_map": dasl_map_up,
            "residual_map": residual_map_up,
            "query_patches": features["query_patches"],
            "normal_proto": prototypes["normal_proto"],
            "abnormal_proto": prototypes["abnormal_proto"],
        }

    def forward_oasl(
        self,
        normal_image: torch.Tensor,
        class_name,
        prompt_images: Optional[torch.Tensor] = None,
        residual_map: Optional[torch.Tensor] = None,
        query_patches: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        batch_size = normal_image.shape[0]
        class_names = self._normalize_class_names(class_name, batch_size)

        if query_patches is None:
            if prompt_images is not None:
                features = self._encode_query_and_prompts(normal_image, prompt_images)
                query_patches = features["query_patches"]
                if residual_map is None:
                    residual_map, _, _ = compute_patch_residual(
                        features["query_patches"],
                        features["prompt_patches"],
                        residual_scale=self.residual_scale,
                    )
                query_dtype = features["query_cls"].dtype
            else:
                query_cls, patch_list, _ = self.encode_image(normal_image)
                query_patches = [patches[:, 1:, :] for patches in patch_list]
                query_dtype = query_cls.dtype
        else:
            query_dtype = query_patches[0].dtype

        if residual_map is None:
            residual_map = torch.zeros(
                batch_size,
                query_patches[0].shape[1],
                device=normal_image.device,
                dtype=query_patches[0].dtype,
            )

        prototypes = self.text_prototypes.build(class_names, device=normal_image.device, dtype=query_dtype)
        if self.disable_oasl:
            s_hat_a = residual_map.clamp(0.0, 1.0)
            s_hat_n = (1.0 - residual_map).clamp(0.0, 1.0)
            pixel_map_oasl = residual_map.clamp(0.0, 1.0)
        else:
            oasl_out = self.oasl(
                patch_tokens=query_patches,
                residual_map=residual_map,
                normal_proto=prototypes["normal_proto"],
                abnormal_proto=prototypes["abnormal_proto"],
            )
            s_hat_a = oasl_out["S_hat_a"]
            s_hat_n = oasl_out["S_hat_n"]
            pixel_map_oasl = oasl_out["pixel_map_oasl"]

        oasl_map_up = upsample_flat_map(pixel_map_oasl, self.image_size)
        return {
            "S_hat_a": s_hat_a,
            "S_hat_n": s_hat_n,
            "M_n": pixel_map_oasl,
            "pixel_map_oasl": pixel_map_oasl,
            "oasl_map": oasl_map_up,
        }

    def forward_inference(self, query_image: torch.Tensor, prompt_images: torch.Tensor, class_name) -> dict:
        main_out = self.forward_main(query_image=query_image, prompt_images=prompt_images, class_name=class_name)
        oasl_out = self.forward_oasl(
            normal_image=query_image,
            class_name=class_name,
            residual_map=main_out["M_x"],
            query_patches=main_out["query_patches"],
        )
        if self.disable_oasl:
            pixel_map = main_out["dasl_map"]
        else:
            pixel_map = fuse_pixel_maps(main_out["dasl_map"], oasl_out["oasl_map"], beta=self.beta)
        return {
            "image_score": main_out["image_score"],
            "pixel_map": pixel_map.clamp(0.0, 1.0),
            "dasl_map": main_out["dasl_map"],
            "oasl_map": oasl_out["oasl_map"],
            "residual_map": main_out["residual_map"],
            **{f"main_{key}": value for key, value in main_out.items() if key not in {"query_patches", "normal_proto", "abnormal_proto"}},
        }

    def forward(self, query_image: torch.Tensor, prompt_images: torch.Tensor, class_name) -> dict:
        return self.forward_inference(query_image=query_image, prompt_images=prompt_images, class_name=class_name)


def _load_model_config(backbone: str) -> dict:
    config_path = Path(__file__).resolve().parents[2] / "open_clip" / "model_configs" / f"{backbone}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"OpenCLIP model config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_checkpoint_into_model(model: InCTRLv2Model, checkpoint_path: str | Path, strict: bool = False):
    checkpoint_path = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    return model.load_state_dict(state_dict, strict=strict)


def build_inctrlv2_model(
    backbone: str = "ViT-B-16-plus-240",
    clip_checkpoint: Optional[str | Path] = None,
    device: str | torch.device = "cpu",
    precision: str = "fp32",
    selected_layers: Optional[Iterable[int]] = None,
    alpha: float = 0.5,
    beta: float = 0.75,
    disable_dasl: bool = False,
    disable_oasl: bool = False,
    allow_random_init: bool = False,
) -> InCTRLv2Model:
    model_config = _load_model_config(backbone)
    tokenizer = open_clip.get_tokenizer(backbone)
    model = InCTRLv2Model(
        embed_dim=model_config["embed_dim"],
        vision_cfg=model_config["vision_cfg"],
        text_cfg=model_config["text_cfg"],
        tokenizer=tokenizer,
        selected_layers=selected_layers,
        alpha=alpha,
        beta=beta,
        cast_dtype=get_cast_dtype(precision),
        disable_dasl=disable_dasl,
        disable_oasl=disable_oasl,
    )

    if clip_checkpoint:
        checkpoint_path = Path(clip_checkpoint).expanduser()
        if checkpoint_path.exists():
            _load_checkpoint_into_model(model, checkpoint_path, strict=False)
        elif not allow_random_init:
            raise FileNotFoundError(f"CLIP checkpoint not found: {checkpoint_path}")
    elif not allow_random_init:
        default_checkpoint = Path("vit_b_16_plus_240-laion400m_e32-699c4b84.pt")
        if default_checkpoint.exists():
            _load_checkpoint_into_model(model, default_checkpoint, strict=False)
        else:
            raise FileNotFoundError(
                "CLIP checkpoint is required. Pass --clip_checkpoint or --allow_random_init for smoke tests."
            )

    return model.to(device)


def namespace_from_args(args) -> SimpleNamespace:
    return SimpleNamespace(**vars(args))
