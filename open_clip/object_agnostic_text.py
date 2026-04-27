import math

import torch
import torch.nn.functional as F


class ObjectAgnosticTextBranch(torch.nn.Module):
    def __init__(self, templates: list[str], logit_scale: float = 100.0):
        super().__init__()
        if len(templates) != 2:
            raise ValueError("ObjectAgnosticTextBranch requires exactly two templates.")
        self.templates = templates
        self.logit_scale = float(logit_scale)

    def build_prototypes(self, encode_text, tokenizer, device):
        tokens = tokenizer(self.templates).to(device)
        prototypes = encode_text(tokens, normalize=True)
        return F.normalize(prototypes, dim=-1)

    def forward(self, encode_text, tokenizer, global_feat, patch_feat, image_size):
        prototypes = self.build_prototypes(encode_text, tokenizer, global_feat.device)

        global_feat = F.normalize(global_feat, dim=-1)
        logits = self.logit_scale * global_feat @ prototypes.t()
        text_logit = logits[:, 1] - logits[:, 0]
        text_score = torch.sigmoid(text_logit)

        patch_feat = F.normalize(patch_feat, dim=-1)
        patch_logits = self.logit_scale * torch.matmul(patch_feat, prototypes.t())
        patch_text_logit = patch_logits[..., 1] - patch_logits[..., 0]
        grid_size = int(math.sqrt(patch_text_logit.shape[1]))
        if grid_size * grid_size != patch_text_logit.shape[1]:
            raise ValueError(f"Patch count {patch_text_logit.shape[1]} is not a square grid.")
        text_map = patch_text_logit.reshape(patch_text_logit.shape[0], 1, grid_size, grid_size)
        text_map = F.interpolate(
            text_map,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        text_map = torch.sigmoid(text_map)

        return {
            "text_logit": text_logit,
            "text_score": text_score,
            "text_map": text_map,
            "text_prototypes": prototypes,
        }
