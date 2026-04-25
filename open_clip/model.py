""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from dataclasses import dataclass
import logging
import math
from typing import Optional, Tuple, Union
from collections import OrderedDict
import re
from sklearn.metrics import pairwise

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch import Tensor
import open_clip.utils.misc as misc
import argparse
from functools import partial
from open_clip.utils.env import checkpoint_pathmgr as pathmgr

from .hf_model import HFTextEncoder
from .modified_resnet import ModifiedResNet
from .timm_model import TimmModel
from .transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer, VisionTransformer_Mul
from .new_utils import to_2tuple

from .vp import (
    PadPrompter,
    RandomPatchPrompter,
    FixedPatchPrompter
)

from torch.autograd import Variable, grad

PROMPT_TYPES = {
    "padding": PadPrompter,
    "random_patch": RandomPatchPrompter,
    "fixed_patch": FixedPatchPrompter
}


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    input_patchnorm: bool = False  # whether to use dual patchnorm - would only apply the input layernorm on each patch, as post-layernorm already exist in original clip vit design
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    output_tokens: bool = True

    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype

state_level = {
               "normal":["{}", "flawless {}", "perfect {}", "unblemished {}",
                         "{} without flaw", "{} without defect", "{} without damage"],
                "anomaly":["damaged {}", "{} with flaw", "{} with defect", "{} with damage"]
}
template_level = [
                  "a cropped photo of the {}.",
                  "a cropped photo of a {}.",
                  "a close-up photo of a {}.",
                  "a close-up photo of the {}.",
                  "a bright photo of a {}.",
                  "a bright photo of the {}.",
                  "a dark photo of a {}.",
                  "a dark photo of the {}.",
                  "a jpeg corrupted photo of a {}.",
                  "a jpeg corrupted photo of the {}.",
                  "a blurry photo of the {}.",
                  "a blurry photo of a {}.",
                  "a photo of the {}.",
                  "a photo of a {}.",
                  "a photo of a small {}.",
                  "a photo of the small {}.",
                  "a photo of a large {}.",
                  "a photo of the large {}.",
                  "a photo of a {} for visual inspection.",
                  "a photo of the {} for visual inspection.",
                  "a photo of a {} for anomaly detection.",
                  "a photo of the {} for anomaly detection."
]

def get_texts(obj_name):

    l = ["airplane", "automobile", "bird",
         "cat", "deer", "dog", "frog", "horse", "ship", "truck", "animal"]

    if obj_name in l:
        normal_texts = []
        anomaly_texts = []
        normal = "a photo of " + obj_name + " for anomaly detection."
        normal_texts.append(normal)
        anomaly = "a photo without " + obj_name + " for anomaly detection."
        anomaly_texts.append(anomaly)
    else:
        normal_states = [s.format(obj_name) for s in state_level["normal"]]
        anomaly_states = [s.format(obj_name) for s in state_level["anomaly"]]

        normal_texts = [t.format(state) for state in normal_states for t in template_level]
        anomaly_texts = [t.format(state) for state in anomaly_states for t in template_level]

    return normal_texts, anomaly_texts


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return visual

def _build_vision_tower_Mul(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = VisionTransformer_Mul(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return visual

def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj=text_cfg.proj,
            pooler_type=text_cfg.pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            output_tokens=text_cfg.output_tokens,
            pad_id=text_cfg.pad_id,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text

class BatchNormPoint(nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        self.feat_size = feat_size
        self.bn = nn.BatchNorm1d(feat_size)

    def forward(self, x):
        assert len(x.shape) == 3
        s1, s2, s3 = x.shape[0], x.shape[1], x.shape[2]
        assert s3 == self.feat_size
        x = x.view(s1 * s2, self.feat_size)
        x = self.bn(x)
        return x.view(s1, s2, s3)

class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()

class TransformerBasicHead(nn.Module):
    """
    Basic Transformer Head. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes
    ):
        super(TransformerBasicHead, self).__init__()
        self.projection1 = nn.Linear(dim_in, 128, bias=True)
        self.projection2 = nn.Linear(128, 64, bias=True)
        self.projection3 = nn.Linear(64, num_classes, bias=True)
        self.bn1 = nn.BatchNorm1d(dim_in)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.projection1(x)
        x = F.relu(x, inplace=True)
        x = self.bn2(x)
        x = self.projection2(x)
        x = F.relu(x, inplace=True)
        x = self.bn3(x)
        x = self.projection3(x)
        return torch.sigmoid(x)

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class PatchTextAdapter(nn.Module):
    """Project patch tokens (vision width) into text-aligned space (embed_dim)."""
    def __init__(self, patch_dim, text_dim, reduction=4):
        super().__init__()
        hidden = patch_dim // reduction
        self.net = nn.Sequential(
            nn.Linear(patch_dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, text_dim, bias=False),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class InCTRL(nn.Module):
    def __init__(
            self,
            args,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower_Mul(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.adapter = Adapter(640, 4)
        self.diff_head = TransformerBasicHead(225, 1)
        self.diff_head_ref = TransformerBasicHead(640, 1)

        for p in self.visual.parameters():
            p.requires_grad = False

        for p in text.parameters():
            p.requires_grad = False

    def encode_image(self, image, out_layers: list = [7, 9, 11], normalize: bool = False):
        features = self.visual.forward(image, out_layers)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def forward(self, tokenizer, image: Optional[torch.Tensor] = None, text: Optional[torch.Tensor] = None, normal_list = None):
        if normal_list == None:
            img = image[0].cuda(non_blocking=True)
            normal_image = image[1:]
            normal_image = torch.stack(normal_image)
            shot, b, _, _, _ = normal_image.shape
            normal_image = normal_image.reshape(-1, 3, 240, 240).cuda(non_blocking=True)
        else:
            img = image[0].cuda(non_blocking=True)
            normal_image = normal_list
            normal_image = torch.stack(normal_image)
            normal_image = normal_image.unsqueeze(1)
            b = len(img)
            normal_image = normal_image.repeat(1, b, 1, 1, 1)
            shot, _, _, _, _ = normal_image.shape
            normal_image = normal_image.reshape(-1, 3, 240, 240).cuda(non_blocking=True)

        token, Fp_list, Fp = self.encode_image(img, normalize=False)
        token_n, Fp_list_n, Fp_n = self.encode_image(normal_image, normalize=False)

        Fp_list = torch.stack(Fp_list)
        Fp_list_n = torch.stack(Fp_list_n)

        Fp_list = Fp_list[:, :, 1:, :]
        Fp_list_n = Fp_list_n[:, :, 1:, :]

        Fp_list = Fp_list.reshape(b, 3, 225, -1)
        Fp_list_n = Fp_list_n.reshape(b, 3, 225 * shot, -1)

        token_n = token_n.reshape(b, shot, -1)

        token_ad = self.adapter.forward(token)
        token_n = self.adapter.forward(token_n)
        token_n = torch.mean(token_n, dim=1)
        token_ref = token_n - token_ad

        text_score = []
        max_diff_score = []
        patch_ref_map = []
        for i in range(len(token)):
            Fp = Fp_list[i, :, :, :]
            Fp_n = Fp_list_n[i, :, :, :]

            Fp_map = list()
            for j in range(len(Fp)):
                tmp_x = Fp[j, :, :]
                tmp_n = Fp_n[j, :, :]
                am_fp = list()
                for k in range(len(tmp_x)):
                    tmp = tmp_x[k]
                    tmp = tmp.unsqueeze(0)
                    tmp_n = tmp_n / tmp_n.norm(dim=-1, keepdim=True)
                    tmp = tmp / tmp.norm(dim=-1, keepdim=True)
                    s = (0.5 * (1 - (tmp @ tmp_n.T))).min(dim=1).values
                    am_fp.append(s)
                am_fp = torch.stack(am_fp)
                Fp_map.append(am_fp)
            Fp_map = torch.stack(Fp_map)
            Fp_map = torch.mean(Fp_map.squeeze(2), dim=0)
            patch_ref_map.append(Fp_map)
            score = Fp_map.max(dim=0).values
            max_diff_score.append(score)

            # zero shot
            image_feature = token[i]
            image_feature = image_feature.unsqueeze(0)
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

            obj_type = text[i]
            normal_texts, anomaly_texts = get_texts(obj_type.replace('_', " "))
            pos_features = tokenizer(normal_texts).cuda()
            neg_features = tokenizer(anomaly_texts).cuda()
            pos_features = self.encode_text(pos_features)
            neg_features = self.encode_text(neg_features)
            pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
            pos_features = torch.mean(pos_features, dim=0, keepdim=True)
            neg_features = torch.mean(neg_features, dim=0, keepdim=True)
            pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
            text_features = torch.cat([pos_features, neg_features], dim=0)
            score = (100 * image_feature @ text_features.T).softmax(dim=-1)
            tmp = score[0, 1]
            text_score.append(tmp)

        text_score = torch.stack(text_score).unsqueeze(1)
        img_ref_score = self.diff_head_ref.forward(token_ref)
        patch_ref_map = torch.stack(patch_ref_map)
        holistic_map = text_score + img_ref_score + patch_ref_map
        hl_score = self.diff_head.forward(holistic_map)

        hl_score = hl_score.squeeze(1)
        fg_score = torch.stack(max_diff_score)
        final_score = (hl_score + fg_score) / 2

        img_ref_score = img_ref_score.squeeze(1)

        return final_score, img_ref_score

class InCTRLv2(nn.Module):
    """
    InCTRL v2 with DASL + OASL dual semantic branches.
    Paper-faithful reimplementation based on InCTRLv2 description.
    Not official code.

    Branches:
      - InCTRL residual branch  (image-level sI + patch-level Mx)
      - DASL  (semantic discriminative, phi1, uses normal+abnormal data)
      - OASL  (one-class normality,    phi2, uses normal-only data)
    """

    def __init__(
            self,
            args,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        # ---------- frozen CLIP backbone ----------
        self.visual = _build_vision_tower_Mul(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        for p in self.visual.parameters():
            p.requires_grad = False
        for p in text.parameters():
            p.requires_grad = False

        # ---------- dimensions ----------
        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)
        self.patch_dim = vision_cfg.width          # 896 for ViT-B-16-plus
        self.text_dim = embed_dim                  # 640
        self.image_size = vision_cfg.image_size    # 240
        self.patch_size = vision_cfg.patch_size    # 16
        self.grid_h = self.image_size // self.patch_size  # 15
        self.num_patches = self.grid_h ** 2               # 225
        self.out_layers = getattr(args, 'patch_layers', [7, 9, 11])
        self.num_layers = len(self.out_layers)

        # ---------- hyperparams ----------
        self.alpha = getattr(args, 'alpha', 0.5)
        self.beta = getattr(args, 'beta', 0.75)
        self.residual_scale = getattr(args, 'residual_scale', 'half')

        # ---------- InCTRL residual branch (trainable) ----------
        self.adapter = Adapter(self.text_dim, 4)                   # psi: image-level adapter
        self.diff_head_ref = TransformerBasicHead(self.text_dim, 1) # eta: residual score head

        # ---------- DASL branch ----------
        self.phi1 = PatchTextAdapter(self.patch_dim, self.text_dim)

        # ---------- OASL branch ----------
        self.phi2 = PatchTextAdapter(self.patch_dim, self.text_dim)

        # ---------- text feature cache ----------
        self._text_cache = {}

    # ------------------------------------------------------------------ #
    #  frozen encoders                                                     #
    # ------------------------------------------------------------------ #
    def encode_image(self, image, normalize=False):
        features = self.visual.forward(image, self.out_layers)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize=False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    # ------------------------------------------------------------------ #
    #  text prototype builder (cached)                                     #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _build_text_prototypes(self, tokenizer, class_names):
        """
        Returns:
            Fn: [B, D]  normal text prototype
            Fa: [B, D]  abnormal text prototype
        """
        Fn_list, Fa_list = [], []
        for name in class_names:
            key = name.replace('_', ' ')
            if key not in self._text_cache:
                normal_texts, anomaly_texts = get_texts(key)
                tok_n = tokenizer(normal_texts).cuda()
                tok_a = tokenizer(anomaly_texts).cuda()
                feat_n = self.encode_text(tok_n)
                feat_a = self.encode_text(tok_a)
                feat_n = F.normalize(feat_n.mean(dim=0), dim=-1)
                feat_a = F.normalize(feat_a.mean(dim=0), dim=-1)
                self._text_cache[key] = (feat_n, feat_a)
            fn, fa = self._text_cache[key]
            Fn_list.append(fn)
            Fa_list.append(fa)
        Fn = torch.stack(Fn_list)  # [B, D]
        Fa = torch.stack(Fa_list)  # [B, D]
        return Fn, Fa

    # ------------------------------------------------------------------ #
    #  input preparation                                                   #
    # ------------------------------------------------------------------ #
    def _prepare_inputs(self, image, normal_list):
        """Parse legacy input format into (query [B,3,H,W], prompts [K,B,3,H,W])."""
        if normal_list is None:
            query = image[0].cuda(non_blocking=True)
            prompts = torch.stack(image[1:]).cuda(non_blocking=True)  # [K, B, 3, H, W]
        else:
            query = image[0].cuda(non_blocking=True)
            prompts = torch.stack(normal_list).unsqueeze(1)           # [K, 1, 3, H, W]
            B = query.shape[0]
            prompts = prompts.expand(-1, B, -1, -1, -1).cuda(non_blocking=True)
        return query, prompts

    # ------------------------------------------------------------------ #
    #  image-level residual  (InCTRL core)                                 #
    # ------------------------------------------------------------------ #
    def _compute_image_residual(self, q_cls, p_cls_per_shot):
        """
        Args:
            q_cls:            [B, D]  query class token (projected)
            p_cls_per_shot:   [B, K, D]  prompt class tokens
        Returns:
            sI:  [B]   image-level residual score
            Fx:  [B, D] residual vector
        """
        q_ad = self.adapter(q_cls)                            # [B, D]
        p_ad = self.adapter(p_cls_per_shot)                   # [B, K, D]
        p_proto = p_ad.mean(dim=1)                            # [B, D]
        Fx = p_proto - q_ad                                   # residual (v1 direction)
        sI = self.diff_head_ref(Fx).squeeze(-1)               # [B]
        return sI, Fx

    # ------------------------------------------------------------------ #
    #  patch-level residual  (vectorised)                                  #
    # ------------------------------------------------------------------ #
    def _compute_patch_residual(self, q_patches, p_patches):
        """
        Args:
            q_patches: list of L tensors, each [B, N, D_patch]
            p_patches: list of L tensors, each [B, K*N, D_patch]
        Returns:
            Mx:      [B, N]   multi-layer averaged residual map (flat)
            sp:      [B]      max-patch residual score
            layer_maps: list of [B, N]
        """
        layer_maps = []
        for q_l, p_l in zip(q_patches, p_patches):
            q_norm = F.normalize(q_l, dim=-1)      # [B, N, D]
            p_norm = F.normalize(p_l, dim=-1)       # [B, KN, D]
            sim = torch.bmm(q_norm, p_norm.transpose(1, 2))  # [B, N, KN]
            max_sim = sim.max(dim=-1).values        # [B, N]
            if self.residual_scale == 'half':
                res = 0.5 * (1.0 - max_sim)
            else:
                res = 1.0 - max_sim
            layer_maps.append(res)
        Mx = torch.stack(layer_maps, dim=0).mean(dim=0)      # [B, N]
        sp = Mx.max(dim=-1).values                            # [B]
        return Mx, sp, layer_maps

    # ------------------------------------------------------------------ #
    #  semantic image score  (DASL s_q)                                    #
    # ------------------------------------------------------------------ #
    def _compute_semantic_image_score(self, q_cls, Fn, Fa):
        """
        Args:
            q_cls: [B, D]   raw class token (before adapter)
            Fn:    [B, D]   normal text prototype
            Fa:    [B, D]   abnormal text prototype
        Returns:
            sq: [B]   P(abnormal) from CLIP semantic space
        """
        q = F.normalize(q_cls, dim=-1)                        # [B, D]
        logit_n = (q * Fn).sum(dim=-1)                        # [B]
        logit_a = (q * Fa).sum(dim=-1)                        # [B]
        logits = torch.stack([logit_n, logit_a], dim=-1)      # [B, 2]
        sq = logits.softmax(dim=-1)[:, 1]                     # [B]
        return sq

    # ------------------------------------------------------------------ #
    #  semantic patch maps  (shared logic for DASL / OASL)                 #
    # ------------------------------------------------------------------ #
    def _compute_semantic_maps(self, q_patches, Fn, Fa, adapter):
        """
        Args:
            q_patches: list of L tensors, each [B, N, D_patch]
            Fn:        [B, D_text]
            Fa:        [B, D_text]
            adapter:   PatchTextAdapter (phi1 or phi2)
        Returns:
            Sn: [B, N]  normality-oriented map   (multi-layer avg)
            Sa: [B, N]  abnormality-oriented map  (multi-layer avg)
        """
        Sn_layers, Sa_layers = [], []
        for q_l in q_patches:
            z = adapter(q_l)                                  # [B, N, D_text]
            logit_n = torch.einsum('bnd,bd->bn', z, Fn)       # [B, N]
            logit_a = torch.einsum('bnd,bd->bn', z, Fa)       # [B, N]
            logits = torch.stack([logit_n, logit_a], dim=-1)  # [B, N, 2]
            probs = logits.softmax(dim=-1)
            Sn_layers.append(probs[..., 0])                   # [B, N]
            Sa_layers.append(probs[..., 1])                   # [B, N]
        Sn = torch.stack(Sn_layers, dim=0).mean(dim=0)        # [B, N]
        Sa = torch.stack(Sa_layers, dim=0).mean(dim=0)        # [B, N]
        return Sn, Sa

    # ------------------------------------------------------------------ #
    #  upsample flat map → [B, 1, H, W]                                   #
    # ------------------------------------------------------------------ #
    def _upsample_map(self, flat_map, target_h=None, target_w=None):
        """flat_map: [B, N] → [B, 1, H, W]"""
        B, N = flat_map.shape
        gh = int(N ** 0.5)
        m = flat_map.view(B, 1, gh, gh)
        h = target_h or self.image_size
        w = target_w or self.image_size
        return F.interpolate(m, size=(h, w), mode='bilinear', align_corners=False)

    # ------------------------------------------------------------------ #
    #  main forward                                                        #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        tokenizer,
        image=None,
        text=None,
        normal_list=None,
        masks=None,
        mode='train',
    ):
        """
        Args:
            tokenizer:    CLIP tokenizer
            image:        legacy format [query_batch, *prompt_batches] or just query
            text:         list of class name strings, length B
            normal_list:  alternative prompt images (for test-time few-shot)
            masks:        [B, H, W] pixel GT masks (optional, for loss outside)
            mode:         'train' | 'eval'

        Returns:  dict with all intermediate scores and maps
        """
        # ---- 1. parse inputs ----
        query, prompts = self._prepare_inputs(image, normal_list)
        B = query.shape[0]
        K = prompts.shape[0]

        # ---- 2. encode images ----
        q_cls, q_patch_list, _ = self.encode_image(query)
        flat_prompts = prompts.reshape(-1, *prompts.shape[2:])     # [K*B, 3, H, W]
        p_cls, p_patch_list, _ = self.encode_image(flat_prompts)

        # reshape prompt features
        p_cls = p_cls.reshape(K, B, -1).permute(1, 0, 2)          # [B, K, D]

        # patch tokens: strip class token, reshape
        q_patches = []
        p_patches = []
        for l_idx in range(self.num_layers):
            ql = q_patch_list[l_idx][:, 1:, :]                    # [B, N, D_patch]
            pl = p_patch_list[l_idx][:, 1:, :]                    # [K*B, N, D_patch]
            pl = pl.reshape(K, B, -1, self.patch_dim).permute(1, 0, 2, 3)  # [B, K, N, D_patch]
            pl = pl.reshape(B, K * self.num_patches, self.patch_dim)        # [B, K*N, D_patch]
            q_patches.append(ql)
            p_patches.append(pl)

        # ---- 3. text prototypes (cached) ----
        Fn, Fa = self._build_text_prototypes(tokenizer, text)

        # ---- 4. InCTRL residual branch ----
        sI, Fx = self._compute_image_residual(q_cls, p_cls)
        Mx, sp, _ = self._compute_patch_residual(q_patches, p_patches)

        # ---- 5. DASL branch ----
        sq = self._compute_semantic_image_score(q_cls, Fn, Fa)
        Sn, Sa = self._compute_semantic_maps(q_patches, Fn, Fa, self.phi1)
        Mp = 0.5 * (Mx + Sa)

        # ---- 6. OASL branch (uses phi2, same query in eval; normal-only in train) ----
        Shat_n, Shat_a = self._compute_semantic_maps(q_patches, Fn, Fa, self.phi2)
        Mn = 0.5 * (Mx + Shat_a)

        # ---- 7. image-level final score ----
        score = (1.0 - self.alpha) * (sI + sq) / 2.0 + self.alpha * sp

        # ---- 8. pixel-level final map ----
        Mp_up = self._upsample_map(Mp)       # [B, 1, H, W]
        Mn_up = self._upsample_map(Mn)       # [B, 1, H, W]
        M_final = (1.0 - self.beta) * Mp_up + self.beta * Mn_up  # [B, 1, H, W]

        return {
            # image-level
            'score': score,       # [B]
            'sI': sI,             # [B]
            'sq': sq,             # [B]
            'sp': sp,             # [B]
            # patch-level flat
            'Mx': Mx,             # [B, N]
            'Sn': Sn,             # [B, N]  DASL normality
            'Sa': Sa,             # [B, N]  DASL abnormality
            'Mp': Mp,             # [B, N]  DASL fused
            'Shat_n': Shat_n,     # [B, N]  OASL normality
            'Shat_a': Shat_a,     # [B, N]  OASL abnormality
            'Mn': Mn,             # [B, N]  OASL fused
            # upsampled maps
            'map': M_final,       # [B, 1, H, W]
            'Mp_up': Mp_up,       # [B, 1, H, W]
            'Mn_up': Mn_up,       # [B, 1, H, W]
        }

    # ------------------------------------------------------------------ #
    #  legacy interface for backward-compatible testing                     #
    # ------------------------------------------------------------------ #
    def forward_legacy(self, tokenizer, image=None, text=None, normal_list=None):
        """Drop-in replacement for original InCTRL forward signature.
        Returns (final_score, img_ref_score) like v1."""
        out = self.forward(tokenizer, image, text, normal_list, mode='eval')
        return out['score'], out['sI']


class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, (CLIP, TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(l, "text_projection", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

        if isinstance(l, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(l, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed

