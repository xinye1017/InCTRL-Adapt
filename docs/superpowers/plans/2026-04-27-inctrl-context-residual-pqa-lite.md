# InCTRL Context Residual PQA Lite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an InCTRL-first anomaly model that preserves image-level residual learning and multi-layer patch residual maps, then adds a lightweight visual adapter, object-agnostic text semantics, and a prompt-query alignment segmentation head for stronger residual representation and pixel localization.

**Architecture:** Keep InCTRL as the outer scaffold: frozen CLIP visual/text towers, few-shot normal references, image residual branch, patch residual branch, text prior, and final score/map fusion. Add three isolated, switchable modules: `VisualAdapter` adapts global and patch features before residuals, `ObjectAgnosticTextBranch` supplies normal/abnormal semantic priors without object names, and `PromptQuerySegHead` uses `query + |query - aligned_prompt|` features to produce segmentation logits and a PQA image logit.

**Tech Stack:** Python, PyTorch, OpenCLIP local package, fvcore `CfgNode`, pytest

---

## Scope Check

This is one model-development plan, not three unrelated projects, because all requested modules share the same InCTRL forward path and must be fused/tested together. Each module is still introduced as a separate ablation so a bad result can be attributed:

- **Ablation A:** InCTRL baseline plus visual adapters.
- **Ablation B:** Ablation A plus object-agnostic text branch.
- **Ablation C:** Ablation B plus PQA segmentation/global head.
- **Final:** C plus explicit image/map fusion losses and smoke-test baseline comparison.

Current checkout note from live inspection on 2026-04-27: the working tree is on `main`, `open_clip/inctrl_pqa.py` and `open_clip/inctrl_three_adapters.py` are absent, while `engine_IC.py` imports `InCTRLWithAdapters`. This plan therefore starts by creating a clean active model module instead of assuming the feature-branch files exist.

## File Structure

- Create: `/Users/xinye/Desktop/InCTRL/open_clip/visual_adapter.py`  
  Responsibility: residual bottleneck adapters for global CLS features and local patch tokens.
- Create: `/Users/xinye/Desktop/InCTRL/open_clip/object_agnostic_text.py`  
  Responsibility: fixed object-agnostic normal/abnormal text prototypes and global/patch text scores.
- Create: `/Users/xinye/Desktop/InCTRL/open_clip/prompt_query_head.py`  
  Responsibility: nearest-neighbor prompt-query patch alignment, context-residual feature construction, local segmentation logits, PQA image logit.
- Create: `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa.py`  
  Responsibility: InCTRL-compatible model class `InCTRLPQA` that combines image residuals, multi-layer patch residuals, visual adapters, object-agnostic text, and PQA segmentation.
- Create: `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_losses.py`  
  Responsibility: focal image losses plus Dice/BCE segmentation loss when masks are available.
- Modify: `/Users/xinye/Desktop/InCTRL/open_clip/config/defaults.py`  
  Responsibility: single source of truth for model switches and fusion weights.
- Modify: `/Users/xinye/Desktop/InCTRL/open_clip/__init__.py`  
  Responsibility: export `InCTRLPQA`.
- Modify: `/Users/xinye/Desktop/InCTRL/engine_IC.py`  
  Responsibility: instantiate `InCTRLPQA`, unpack optional masks, train with the new loss contract, preserve eval AUROC/AUPR behavior.
- Modify: `/Users/xinye/Desktop/InCTRL/engine_test.py`  
  Responsibility: instantiate `InCTRLPQA` for inference-only path while preserving few-shot reference loading.
- Modify: `/Users/xinye/Desktop/InCTRL/main.py` and `/Users/xinye/Desktop/InCTRL/test.py`  
  Responsibility: remove hard-coded `CUDA_VISIBLE_DEVICES` assignment and rely on the caller environment.
- Create: `/Users/xinye/Desktop/InCTRL/train_local.py` if still absent after branch setup  
  Responsibility: single-process smoke training/evaluation entrypoint without `launch_job`.
- Create: `/Users/xinye/Desktop/InCTRL/tests/test_visual_adapter.py`
- Create: `/Users/xinye/Desktop/InCTRL/tests/test_object_agnostic_text.py`
- Create: `/Users/xinye/Desktop/InCTRL/tests/test_prompt_query_head.py`
- Create: `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa.py`
- Create: `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_losses.py`
- Modify: `/Users/xinye/Desktop/InCTRL/tests/test_engine_ic_training.py`
- Create: `/Users/xinye/Desktop/InCTRL/experiments/2026-04-27-inctrl-context-residual-pqa-lite.md`  
  Responsibility: record hypothesis, commands, smoke metrics, and deltas versus canonical baselines.

## Architecture Contract

Stable forward call:

```python
outputs = model(
    query_image=query_image,
    prompt_images=prompt_images,
    obj_types=types,
    return_aux=True,
    return_dict=True,
)
```

Stable primary outputs:

```python
{
    "final_score": Tensor[B],
    "final_logit": Tensor[B],
    "image_score": Tensor[B],
    "image_logit": Tensor[B],
    "patch_score": Tensor[B],
    "patch_residual_map": Tensor[B, N],
    "pqa_score": Tensor[B],
    "pqa_logit": Tensor[B],
    "pqa_seg_logits": Tensor[B, 1, H, W],
    "text_score": Tensor[B],
    "text_logit": Tensor[B],
    "text_map": Tensor[B, 1, H, W],
    "final_map": Tensor[B, 1, H, W],
}
```

Default tuple compatibility:

```python
preds, image_score = model(tokenizer, inputs, types, normal_list)
```

This tuple path is retained for legacy `engine_test.py`-style callers.

## Hypotheses

- **Visual adapter hypothesis:** residual bottleneck adapters make frozen CLIP global and patch features more anomaly-sensitive without damaging InCTRL's nearest-neighbor residual geometry.
- **Object-agnostic text hypothesis:** fixed normal/abnormal object-agnostic prompts reduce object-name semantic noise and provide a stable abnormality prior for both global and patch tokens.
- **PQA segmentation hypothesis:** `query + |query - aligned_prompt|` context features improve pixel localization because the local head sees both query appearance and deviation from few-shot normal references.
- **Fusion hypothesis:** fixed initial fusion weights keep InCTRL evidence dominant while allowing PQA/text branches to help localization and reduce false positives.

### Task 1: Branch Hygiene And Config Knobs

**Files:**
- Modify: `/Users/xinye/Desktop/InCTRL/open_clip/config/defaults.py`
- Modify: `/Users/xinye/Desktop/InCTRL/main.py`
- Modify: `/Users/xinye/Desktop/InCTRL/test.py`
- Test: `/Users/xinye/Desktop/InCTRL/tests/test_engine_ic_training.py`

- [ ] **Step 1: Write the failing config test**

Append this to `/Users/xinye/Desktop/InCTRL/tests/test_engine_ic_training.py`:

```python
from open_clip.config.defaults import get_cfg


def test_pqa_lite_config_defaults_are_available():
    cfg = get_cfg()

    assert cfg.MODEL.ACTIVE_MODEL == "InCTRLPQA"
    assert cfg.VISUAL_ADAPTER.ENABLE is True
    assert cfg.VISUAL_ADAPTER.REDUCTION == 4
    assert cfg.TEXT_BRANCH.ENABLE is True
    assert cfg.TEXT_BRANCH.TEMPLATES == [
        "a photo of a normal object.",
        "a photo of a damaged object.",
    ]
    assert cfg.PQA.ENABLE is True
    assert cfg.PQA.PATCH_LAYERS == [7, 9, 11]
    assert cfg.PQA.CONTEXT_BETA == 1.0
    assert cfg.FUSION.IMAGE_WEIGHT == 0.35
    assert cfg.FUSION.PATCH_WEIGHT == 0.25
    assert cfg.FUSION.PQA_WEIGHT == 0.25
    assert cfg.FUSION.TEXT_WEIGHT == 0.15
    assert cfg.FUSION.MAP_RES_WEIGHT == 0.4
    assert cfg.FUSION.MAP_PQA_WEIGHT == 0.4
    assert cfg.FUSION.MAP_TEXT_WEIGHT == 0.2
    assert cfg.LOSS.IMAGE_WEIGHT == 1.0
    assert cfg.LOSS.PQA_WEIGHT == 0.5
    assert cfg.LOSS.MASK_WEIGHT == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_engine_ic_training.py::test_pqa_lite_config_defaults_are_available -q
```

Expected: FAIL with `AttributeError` for `ACTIVE_MODEL`, `VISUAL_ADAPTER`, `TEXT_BRANCH`, `PQA`, `FUSION`, or `LOSS`.

- [ ] **Step 3: Add config defaults**

Add this block to `/Users/xinye/Desktop/InCTRL/open_clip/config/defaults.py` after the existing `_C.MODEL` section:

```python
_C.MODEL.ACTIVE_MODEL = "InCTRLPQA"

_C.VISUAL_ADAPTER = CfgNode()
_C.VISUAL_ADAPTER.ENABLE = True
_C.VISUAL_ADAPTER.REDUCTION = 4
_C.VISUAL_ADAPTER.ZERO_INIT = True

_C.TEXT_BRANCH = CfgNode()
_C.TEXT_BRANCH.ENABLE = True
_C.TEXT_BRANCH.TEMPLATES = [
    "a photo of a normal object.",
    "a photo of a damaged object.",
]
_C.TEXT_BRANCH.LOGIT_SCALE = 100.0

_C.PQA = CfgNode()
_C.PQA.ENABLE = True
_C.PQA.PATCH_LAYERS = [7, 9, 11]
_C.PQA.CONTEXT_BETA = 1.0
_C.PQA.HIDDEN_DIM = 128
_C.PQA.GLOBAL_TOPK = 10

_C.FUSION = CfgNode()
_C.FUSION.IMAGE_WEIGHT = 0.35
_C.FUSION.PATCH_WEIGHT = 0.25
_C.FUSION.PQA_WEIGHT = 0.25
_C.FUSION.TEXT_WEIGHT = 0.15
_C.FUSION.MAP_RES_WEIGHT = 0.4
_C.FUSION.MAP_PQA_WEIGHT = 0.4
_C.FUSION.MAP_TEXT_WEIGHT = 0.2

_C.LOSS = CfgNode()
_C.LOSS.IMAGE_WEIGHT = 1.0
_C.LOSS.PQA_WEIGHT = 0.5
_C.LOSS.MASK_WEIGHT = 1.0
_C.LOSS.TEXT_WEIGHT = 0.0
```

Remove these two assignments from `/Users/xinye/Desktop/InCTRL/main.py`:

```python
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
```

Remove these two assignments from `/Users/xinye/Desktop/InCTRL/test.py`:

```python
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_engine_ic_training.py::test_pqa_lite_config_defaults_are_available -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /Users/xinye/Desktop/InCTRL add open_clip/config/defaults.py main.py test.py tests/test_engine_ic_training.py
git -C /Users/xinye/Desktop/InCTRL commit -m "config: add pqa lite model switches"
```

### Task 2: Lightweight Visual Adapter

**Files:**
- Create: `/Users/xinye/Desktop/InCTRL/open_clip/visual_adapter.py`
- Create: `/Users/xinye/Desktop/InCTRL/tests/test_visual_adapter.py`
- Test: `/Users/xinye/Desktop/InCTRL/tests/test_visual_adapter.py`

- [ ] **Step 1: Write the failing tests**

Create `/Users/xinye/Desktop/InCTRL/tests/test_visual_adapter.py`:

```python
import torch

from open_clip.visual_adapter import ResidualMLPAdapter, VisualAdapter


def test_residual_mlp_adapter_preserves_shape_and_starts_as_identity():
    torch.manual_seed(7)
    adapter = ResidualMLPAdapter(dim=8, reduction=4, zero_init=True)
    x = torch.randn(2, 5, 8)

    y = adapter(x)

    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6)


def test_visual_adapter_updates_global_and_patch_tokens_when_not_zero_init():
    torch.manual_seed(11)
    adapter = VisualAdapter(dim=8, reduction=4, zero_init=False)
    global_feat = torch.randn(2, 8)
    patch_feats = [torch.randn(2, 4, 8), torch.randn(2, 4, 8), torch.randn(2, 4, 8)]

    adapted_global, adapted_patches = adapter(global_feat, patch_feats)

    assert adapted_global.shape == global_feat.shape
    assert len(adapted_patches) == 3
    assert adapted_patches[0].shape == patch_feats[0].shape
    assert not torch.allclose(adapted_global, global_feat)
    assert not torch.allclose(adapted_patches[0], patch_feats[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_visual_adapter.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'open_clip.visual_adapter'`.

- [ ] **Step 3: Create the adapter module**

Create `/Users/xinye/Desktop/InCTRL/open_clip/visual_adapter.py`:

```python
import torch
from torch import nn


class ResidualMLPAdapter(nn.Module):
    def __init__(self, dim: int, reduction: int = 4, zero_init: bool = True):
        super().__init__()
        hidden = max(dim // reduction, 1)
        self.down = nn.Linear(dim, hidden)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(hidden, dim)
        self.scale = nn.Parameter(torch.tensor(0.1))
        if zero_init:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.up(self.act(self.down(x)))


class VisualAdapter(nn.Module):
    def __init__(self, dim: int, reduction: int = 4, zero_init: bool = True):
        super().__init__()
        self.global_adapter = ResidualMLPAdapter(dim, reduction, zero_init)
        self.local_adapter = ResidualMLPAdapter(dim, reduction, zero_init)

    def forward(self, global_feat: torch.Tensor, patch_feats: list[torch.Tensor]):
        adapted_global = self.global_adapter(global_feat)
        adapted_patches = [self.local_adapter(feat) for feat in patch_feats]
        return adapted_global, adapted_patches
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_visual_adapter.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /Users/xinye/Desktop/InCTRL add open_clip/visual_adapter.py tests/test_visual_adapter.py
git -C /Users/xinye/Desktop/InCTRL commit -m "feat: add lightweight visual adapter"
```

### Task 3: Object-Agnostic Text Branch

**Files:**
- Create: `/Users/xinye/Desktop/InCTRL/open_clip/object_agnostic_text.py`
- Create: `/Users/xinye/Desktop/InCTRL/tests/test_object_agnostic_text.py`
- Test: `/Users/xinye/Desktop/InCTRL/tests/test_object_agnostic_text.py`

- [ ] **Step 1: Write the failing tests**

Create `/Users/xinye/Desktop/InCTRL/tests/test_object_agnostic_text.py`:

```python
import torch
import torch.nn.functional as F

from open_clip.object_agnostic_text import ObjectAgnosticTextBranch


class FakeTextEncoder:
    def __call__(self, tokens, normalize=False):
        feats = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        return F.normalize(feats, dim=-1) if normalize else feats


def fake_tokenizer(texts):
    assert texts == ["a photo of a normal object.", "a photo of a damaged object."]
    return torch.ones(len(texts), 77, dtype=torch.long)


def test_object_agnostic_text_branch_scores_global_and_patch_features():
    branch = ObjectAgnosticTextBranch(
        templates=["a photo of a normal object.", "a photo of a damaged object."],
        logit_scale=10.0,
    )
    global_feat = F.normalize(torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]), dim=-1)
    patch_feat = F.normalize(torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]), dim=-1)

    outputs = branch(
        encode_text=FakeTextEncoder(),
        tokenizer=fake_tokenizer,
        global_feat=global_feat,
        patch_feat=patch_feat,
        image_size=32,
    )

    assert outputs["text_logit"].shape == (2,)
    assert outputs["text_score"].shape == (2,)
    assert outputs["text_map"].shape == (1, 1, 32, 32)
    assert outputs["text_score"][0] > 0.99
    assert outputs["text_score"][1] < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_object_agnostic_text.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'open_clip.object_agnostic_text'`.

- [ ] **Step 3: Create the text branch**

Create `/Users/xinye/Desktop/InCTRL/open_clip/object_agnostic_text.py`:

```python
import math

import torch
import torch.nn.functional as F


class ObjectAgnosticTextBranch(torch.nn.Module):
    def __init__(self, templates: list[str], logit_scale: float = 100.0):
        super().__init__()
        if len(templates) != 2:
            raise ValueError("ObjectAgnosticTextBranch requires [normal_template, abnormal_template].")
        self.templates = templates
        self.logit_scale = float(logit_scale)

    def build_prototypes(self, encode_text, tokenizer, device: torch.device) -> torch.Tensor:
        tokens = tokenizer(self.templates).to(device)
        prototypes = encode_text(tokens, normalize=True)
        return F.normalize(prototypes, dim=-1)

    def forward(
        self,
        encode_text,
        tokenizer,
        global_feat: torch.Tensor,
        patch_feat: torch.Tensor,
        image_size: int,
    ) -> dict[str, torch.Tensor]:
        prototypes = self.build_prototypes(encode_text, tokenizer, global_feat.device)
        global_feat = F.normalize(global_feat, dim=-1)
        logits = self.logit_scale * global_feat @ prototypes.t()
        text_logit = logits[:, 1] - logits[:, 0]
        text_score = torch.sigmoid(text_logit)

        patch_feat = F.normalize(patch_feat, dim=-1)
        patch_logits = self.logit_scale * torch.matmul(patch_feat, prototypes.t())
        patch_text_logit = patch_logits[..., 1] - patch_logits[..., 0]
        grid = int(math.sqrt(patch_text_logit.shape[-1]))
        if grid * grid != patch_text_logit.shape[-1]:
            raise ValueError(f"Patch count {patch_text_logit.shape[-1]} is not a square grid.")
        text_map = patch_text_logit.reshape(patch_text_logit.shape[0], 1, grid, grid)
        text_map = F.interpolate(text_map, size=(image_size, image_size), mode="bilinear", align_corners=False)
        text_map = torch.sigmoid(text_map)

        return {
            "text_logit": text_logit,
            "text_score": text_score,
            "text_map": text_map,
            "text_prototypes": prototypes,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_object_agnostic_text.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /Users/xinye/Desktop/InCTRL add open_clip/object_agnostic_text.py tests/test_object_agnostic_text.py
git -C /Users/xinye/Desktop/InCTRL commit -m "feat: add object agnostic text branch"
```

### Task 4: Prompt-Query Alignment Segmentation Head

**Files:**
- Create: `/Users/xinye/Desktop/InCTRL/open_clip/prompt_query_head.py`
- Create: `/Users/xinye/Desktop/InCTRL/tests/test_prompt_query_head.py`
- Test: `/Users/xinye/Desktop/InCTRL/tests/test_prompt_query_head.py`

- [ ] **Step 1: Write the failing tests**

Create `/Users/xinye/Desktop/InCTRL/tests/test_prompt_query_head.py`:

```python
import torch

from open_clip.prompt_query_head import align_prompt_to_query, PromptQuerySegHead


def test_align_prompt_to_query_selects_nearest_prompt_patch():
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    p = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]]])

    aligned, indices, residual_map = align_prompt_to_query(q, p)

    assert indices.tolist() == [[1, 0]]
    assert torch.allclose(aligned, q)
    assert torch.allclose(residual_map, torch.zeros(1, 2), atol=1e-6)


def test_prompt_query_seg_head_outputs_segmentation_and_image_logit():
    torch.manual_seed(23)
    head = PromptQuerySegHead(dim=8, hidden_dim=4, image_size=32, topk=2)
    query = torch.randn(2, 4, 8)
    prompt = torch.randn(2, 8, 8)

    outputs = head(query, prompt)

    assert outputs["pqa_seg_logits"].shape == (2, 1, 32, 32)
    assert outputs["pqa_logit"].shape == (2,)
    assert outputs["pqa_score"].shape == (2,)
    assert outputs["pqa_patch_map"].shape == (2, 4)
    assert outputs["context_tokens"].shape == (2, 4, 8)
    assert outputs["aligned_indices"].shape == (2, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_prompt_query_head.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'open_clip.prompt_query_head'`.

- [ ] **Step 3: Create the PQA head**

Create `/Users/xinye/Desktop/InCTRL/open_clip/prompt_query_head.py`:

```python
import math

import torch
from torch import nn
import torch.nn.functional as F


def align_prompt_to_query(query_tokens: torch.Tensor, prompt_tokens: torch.Tensor):
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
    def __init__(self, dim: int, hidden_dim: int = 128, image_size: int = 240, topk: int = 10, beta: float = 1.0):
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_prompt_query_head.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /Users/xinye/Desktop/InCTRL add open_clip/prompt_query_head.py tests/test_prompt_query_head.py
git -C /Users/xinye/Desktop/InCTRL commit -m "feat: add prompt query segmentation head"
```

### Task 5: InCTRLPQA Model Wrapper

**Files:**
- Create: `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa.py`
- Modify: `/Users/xinye/Desktop/InCTRL/open_clip/__init__.py`
- Create: `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa.py`
- Test: `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa.py`

- [ ] **Step 1: Write the failing tests**

Create `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa.py`:

```python
import torch

from open_clip.inctrl_pqa import _score_to_logit, _fuse_scores, _fuse_maps


def test_score_to_logit_is_finite_at_edges():
    scores = torch.tensor([0.0, 0.5, 1.0])
    logits = _score_to_logit(scores)

    assert torch.isfinite(logits).all()
    assert logits[0] < 0
    assert logits[1].abs() < 1e-6
    assert logits[2] > 0


def test_fuse_scores_uses_config_weights_without_learning_competition():
    fused = _fuse_scores(
        image_logit=torch.tensor([1.0]),
        patch_logit=torch.tensor([2.0]),
        pqa_logit=torch.tensor([3.0]),
        text_logit=torch.tensor([4.0]),
        weights=(0.35, 0.25, 0.25, 0.15),
    )

    assert torch.allclose(fused, torch.tensor([2.3]))


def test_fuse_maps_preserves_shape_and_weighting():
    residual = torch.ones(1, 1, 4, 4)
    pqa = torch.ones(1, 1, 4, 4) * 2
    text = torch.ones(1, 1, 4, 4) * 3

    fused = _fuse_maps(residual, pqa, text, weights=(0.4, 0.4, 0.2))

    assert fused.shape == (1, 1, 4, 4)
    assert torch.allclose(fused, torch.ones(1, 1, 4, 4) * 1.8)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'open_clip.inctrl_pqa'`.

- [ ] **Step 3: Create model helpers and class skeleton**

Create `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa.py` with this top section:

```python
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .model import _build_vision_tower_Mul, _build_text_tower, get_cast_dtype
from .visual_adapter import VisualAdapter
from .object_agnostic_text import ObjectAgnosticTextBranch
from .prompt_query_head import PromptQuerySegHead


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
    return image_w * image_logit + patch_w * patch_logit + pqa_w * pqa_logit + text_w * text_logit


def _fuse_maps(
    residual_map: torch.Tensor,
    pqa_map: torch.Tensor,
    text_map: torch.Tensor,
    weights: tuple[float, float, float],
) -> torch.Tensor:
    residual_w, pqa_w, text_w = weights
    return residual_w * residual_map + pqa_w * pqa_map + text_w * text_map


class ImageResidualHead(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_global: torch.Tensor, prompt_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_proto = prompt_global.mean(dim=1)
        residual = prompt_proto - query_global
        logit = self.net(residual).squeeze(-1)
        return torch.sigmoid(logit), logit
```

Add this class shell below the helpers:

```python
class InCTRLPQA(nn.Module):
    def __init__(self, args, embed_dim, vision_cfg, text_cfg, quick_gelu=False, cast_dtype=None, output_dict=False):
        super().__init__()
        self.args = args
        self.output_dict = output_dict
        self.image_size = int(getattr(args, "image_size", 240))
        self.patch_layers = list(getattr(args.PQA, "PATCH_LAYERS", [7, 9, 11]))
        hidden_dim = int(getattr(args.PQA, "HIDDEN_DIM", 128))

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
        self.visual_adapter = VisualAdapter(embed_dim, reduction=reduction, zero_init=zero_init)
        self.image_head = ImageResidualHead(embed_dim, hidden_dim=hidden_dim)
        self.pqa_head = PromptQuerySegHead(
            dim=embed_dim,
            hidden_dim=hidden_dim,
            image_size=self.image_size,
            topk=int(getattr(args.PQA, "GLOBAL_TOPK", 10)),
            beta=float(getattr(args.PQA, "CONTEXT_BETA", 1.0)),
        )
        self.text_branch = ObjectAgnosticTextBranch(
            templates=list(getattr(args.TEXT_BRANCH, "TEMPLATES")),
            logit_scale=float(getattr(args.TEXT_BRANCH, "LOGIT_SCALE", 100.0)),
        )
```

Add this export to `/Users/xinye/Desktop/InCTRL/open_clip/__init__.py`:

```python
from .inctrl_pqa import InCTRLPQA
```

- [ ] **Step 4: Run helper tests to verify they pass**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /Users/xinye/Desktop/InCTRL add open_clip/inctrl_pqa.py open_clip/__init__.py tests/test_inctrl_pqa.py
git -C /Users/xinye/Desktop/InCTRL commit -m "feat: add inctrl pqa model shell"
```

### Task 6: Full Forward Path Preserving InCTRL Residuals

**Files:**
- Modify: `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa.py`
- Modify: `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa.py`
- Test: `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa.py`

- [ ] **Step 1: Add forward-helper tests**

Append this to `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa.py`:

```python
from types import SimpleNamespace

from open_clip.inctrl_pqa import InCTRLPQA


def _cfg():
    return SimpleNamespace(
        image_size=32,
        VISUAL_ADAPTER=SimpleNamespace(REDUCTION=4, ZERO_INIT=True),
        TEXT_BRANCH=SimpleNamespace(
            TEMPLATES=["a photo of a normal object.", "a photo of a damaged object."],
            LOGIT_SCALE=10.0,
        ),
        PQA=SimpleNamespace(PATCH_LAYERS=[7, 9, 11], HIDDEN_DIM=8, GLOBAL_TOPK=2, CONTEXT_BETA=1.0),
        FUSION=SimpleNamespace(
            IMAGE_WEIGHT=0.35,
            PATCH_WEIGHT=0.25,
            PQA_WEIGHT=0.25,
            TEXT_WEIGHT=0.15,
            MAP_RES_WEIGHT=0.4,
            MAP_PQA_WEIGHT=0.4,
            MAP_TEXT_WEIGHT=0.2,
        ),
    )


def test_prepare_prompt_tokens_shapes():
    model = InCTRLPQA.__new__(InCTRLPQA)
    query = torch.randn(2, 4, 8)
    prompts = torch.randn(2, 3, 4, 8)

    flat = InCTRLPQA._flatten_prompt_tokens(model, prompts)

    assert flat.shape == (2, 12, 8)


def test_patch_residual_map_matches_nearest_prompt_residual():
    model = InCTRLPQA.__new__(InCTRLPQA)
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    p = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    residual = InCTRLPQA._compute_patch_residual(model, q, p)

    assert torch.allclose(residual, torch.zeros(1, 2), atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa.py::test_prepare_prompt_tokens_shapes /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa.py::test_patch_residual_map_matches_nearest_prompt_residual -q
```

Expected: FAIL with `AttributeError` for `_flatten_prompt_tokens` and `_compute_patch_residual`.

- [ ] **Step 3: Add forward helper methods**

Add these methods inside `InCTRLPQA` in `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa.py`:

```python
def encode_image(self, image: torch.Tensor, normalize: bool = False):
    features = self.visual.forward(image, self.patch_layers)
    return F.normalize(features, dim=-1) if normalize else features

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

def _compute_patch_residual(self, query_tokens: torch.Tensor, prompt_tokens: torch.Tensor) -> torch.Tensor:
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
    return F.interpolate(patch_map, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
```

- [ ] **Step 4: Run tests to verify helper behavior**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa.py -q
```

Expected: PASS.

- [ ] **Step 5: Add the full forward path**

Add this `forward()` method inside `InCTRLPQA`:

```python
def forward(
    self,
    tokenizer=None,
    image: Optional[torch.Tensor] = None,
    text: Optional[list[str]] = None,
    normal_list=None,
    query_image: Optional[torch.Tensor] = None,
    prompt_images: Optional[torch.Tensor] = None,
    obj_types: Optional[list[str]] = None,
    return_aux: bool = False,
    return_dict: bool = False,
):
    legacy_tuple = query_image is None
    if query_image is None:
        query_image = image[0]
        obj_types = text
        if normal_list is None:
            prompt_images = torch.stack(image[1:], dim=1)
        else:
            prompt_images = torch.stack(normal_list).unsqueeze(0).repeat(query_image.shape[0], 1, 1, 1, 1)

    batch, shot, channels, height, width = prompt_images.shape
    device = query_image.device
    prompt_flat = prompt_images.reshape(batch * shot, channels, height, width)

    query_global, query_patch_levels, _ = self.encode_image(query_image, normalize=False)
    prompt_global_flat, prompt_patch_levels_flat, _ = self.encode_image(prompt_flat, normalize=False)
    prompt_global = prompt_global_flat.reshape(batch, shot, -1)

    query_patch_levels = [level[:, 1:, :] for level in query_patch_levels]
    prompt_patch_levels = [
        level[:, 1:, :].reshape(batch, shot, level.shape[1] - 1, level.shape[-1])
        for level in prompt_patch_levels_flat
    ]

    query_global, query_patch_levels = self.visual_adapter(query_global, query_patch_levels)
    prompt_global_adapted = self.visual_adapter.global_adapter(prompt_global)
    prompt_patch_levels = [
        self.visual_adapter.local_adapter(level.reshape(batch * shot, level.shape[2], level.shape[3])).reshape(
            batch, shot, level.shape[2], level.shape[3]
        )
        for level in prompt_patch_levels
    ]

    image_score, image_logit = self.image_head(query_global, prompt_global_adapted)

    residual_maps = []
    pqa_maps = []
    pqa_logits = []
    pqa_seg_logits = []
    for query_tokens, prompt_tokens in zip(query_patch_levels, prompt_patch_levels):
        prompt_flat_tokens = self._flatten_prompt_tokens(prompt_tokens)
        residual_map = self._compute_patch_residual(query_tokens, prompt_flat_tokens)
        pqa_out = self.pqa_head(query_tokens, prompt_flat_tokens)
        residual_maps.append(residual_map)
        pqa_maps.append(pqa_out["pqa_patch_map"])
        pqa_logits.append(pqa_out["pqa_logit"])
        pqa_seg_logits.append(pqa_out["pqa_seg_logits"])

    patch_residual_map = torch.stack(residual_maps, dim=0).mean(dim=0)
    patch_score = patch_residual_map.max(dim=-1).values
    patch_logit = _score_to_logit(patch_score)
    pqa_patch_map = torch.stack(pqa_maps, dim=0).mean(dim=0)
    pqa_logit = torch.stack(pqa_logits, dim=0).mean(dim=0)
    pqa_score = torch.sigmoid(pqa_logit)
    pqa_seg_logits = torch.stack(pqa_seg_logits, dim=0).mean(dim=0)

    text_out = self.text_branch(
        encode_text=self.encode_text,
        tokenizer=tokenizer,
        global_feat=query_global,
        patch_feat=query_patch_levels[-1],
        image_size=self.image_size,
    )
    text_logit = text_out["text_logit"]
    text_score = text_out["text_score"]
    text_map = text_out["text_map"]

    fusion_weights = (
        float(self.args.FUSION.IMAGE_WEIGHT),
        float(self.args.FUSION.PATCH_WEIGHT),
        float(self.args.FUSION.PQA_WEIGHT),
        float(self.args.FUSION.TEXT_WEIGHT),
    )
    final_logit = _fuse_scores(image_logit, patch_logit, pqa_logit, text_logit, fusion_weights)
    final_score = torch.sigmoid(final_logit)

    residual_map_up = self._upsample_patch_map(patch_residual_map)
    pqa_map = torch.sigmoid(pqa_seg_logits)
    final_map = _fuse_maps(
        residual_map_up,
        pqa_map,
        text_map,
        (
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
```

- [ ] **Step 6: Commit**

```bash
git -C /Users/xinye/Desktop/InCTRL add open_clip/inctrl_pqa.py tests/test_inctrl_pqa.py
git -C /Users/xinye/Desktop/InCTRL commit -m "feat: combine inctrl residuals with pqa lite forward"
```

### Task 7: Loss Contract With Mask Supervision

**Files:**
- Create: `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_losses.py`
- Create: `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_losses.py`
- Test: `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_losses.py`

- [ ] **Step 1: Write failing loss tests**

Create `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_losses.py`:

```python
from types import SimpleNamespace

import torch

from open_clip.inctrl_pqa_losses import dice_loss, compute_inctrl_pqa_loss


def _cfg():
    return SimpleNamespace(LOSS=SimpleNamespace(IMAGE_WEIGHT=1.0, PQA_WEIGHT=0.5, MASK_WEIGHT=1.0, TEXT_WEIGHT=0.0))


def test_dice_loss_is_near_zero_for_perfect_mask():
    logits = torch.full((1, 1, 4, 4), 20.0)
    masks = torch.ones(1, 1, 4, 4)

    loss = dice_loss(logits, masks)

    assert loss < 1e-4


def test_compute_inctrl_pqa_loss_uses_image_pqa_and_mask_terms():
    outputs = {
        "final_logit": torch.tensor([0.0, 1.0]),
        "image_logit": torch.tensor([0.0, 1.0]),
        "pqa_logit": torch.tensor([0.0, 1.0]),
        "text_logit": torch.tensor([0.0, 1.0]),
        "pqa_seg_logits": torch.randn(2, 1, 8, 8),
    }
    labels = torch.tensor([0.0, 1.0])
    masks = torch.zeros(2, 1, 8, 8)
    masks[1, :, 2:5, 2:5] = 1.0

    loss, parts = compute_inctrl_pqa_loss(outputs, labels, masks, _cfg())

    assert loss.requires_grad is False
    assert set(parts.keys()) == {"final", "image", "pqa", "text", "mask", "total"}
    assert parts["total"] == loss.item()
    assert parts["mask"] > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_losses.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'open_clip.inctrl_pqa_losses'`.

- [ ] **Step 3: Create loss module**

Create `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_losses.py`:

```python
import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    masks = masks.float()
    intersection = (probs * masks).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + eps) / (union + eps)).mean()


def segmentation_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    masks = masks.float()
    bce = F.binary_cross_entropy_with_logits(logits, masks)
    return bce + dice_loss(logits, masks)


def compute_inctrl_pqa_loss(outputs: dict, labels: torch.Tensor, masks: torch.Tensor | None, cfg):
    labels = labels.float()
    final = F.binary_cross_entropy_with_logits(outputs["final_logit"], labels)
    image = F.binary_cross_entropy_with_logits(outputs["image_logit"], labels)
    pqa = F.binary_cross_entropy_with_logits(outputs["pqa_logit"], labels)
    text = F.binary_cross_entropy_with_logits(outputs["text_logit"], labels)
    if masks is None:
        mask = outputs["final_logit"].sum() * 0.0
    else:
        mask = segmentation_loss(outputs["pqa_seg_logits"], masks.to(outputs["pqa_seg_logits"].device))

    total = (
        final
        + float(cfg.LOSS.IMAGE_WEIGHT) * image
        + float(cfg.LOSS.PQA_WEIGHT) * pqa
        + float(cfg.LOSS.TEXT_WEIGHT) * text
        + float(cfg.LOSS.MASK_WEIGHT) * mask
    )
    parts = {
        "final": final.item(),
        "image": image.item(),
        "pqa": pqa.item(),
        "text": text.item(),
        "mask": mask.item(),
        "total": total.item(),
    }
    return total, parts
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_losses.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /Users/xinye/Desktop/InCTRL add open_clip/inctrl_pqa_losses.py tests/test_inctrl_pqa_losses.py
git -C /Users/xinye/Desktop/InCTRL commit -m "feat: add inctrl pqa loss contract"
```

### Task 8: Wire Training And Inference Entrypoints

**Files:**
- Modify: `/Users/xinye/Desktop/InCTRL/engine_IC.py`
- Modify: `/Users/xinye/Desktop/InCTRL/engine_test.py`
- Modify: `/Users/xinye/Desktop/InCTRL/tests/test_engine_ic_training.py`
- Test: `/Users/xinye/Desktop/InCTRL/tests/test_engine_ic_training.py`

- [ ] **Step 1: Write failing engine utility tests**

Append this to `/Users/xinye/Desktop/InCTRL/tests/test_engine_ic_training.py`:

```python
import torch

from engine_IC import _split_batch_with_optional_masks, _build_active_model
from open_clip.config.defaults import get_cfg


def test_split_batch_with_optional_masks_supports_four_item_batch():
    inputs = [torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)]
    types = ["candle", "candle"]
    labels = torch.tensor([0, 1])
    masks = torch.randn(2, 1, 32, 32)

    query, prompts, out_types, out_labels, out_masks = _split_batch_with_optional_masks((inputs, types, labels, masks))

    assert query.shape == (2, 3, 32, 32)
    assert prompts.shape == (2, 2, 3, 32, 32)
    assert out_types == types
    assert torch.equal(out_labels, labels)
    assert torch.equal(out_masks, masks)


def test_build_active_model_returns_inctrl_pqa_for_default_cfg():
    cfg = get_cfg()
    model_cfg = {
        "embed_dim": 8,
        "vision_cfg": {"image_size": 32, "layers": 2, "width": 8, "patch_size": 16, "head_width": 4, "mlp_ratio": 2.0},
        "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 8, "heads": 2, "layers": 1},
    }

    model = _build_active_model(cfg, model_cfg, cast_dtype=None, quick_gelu=False)

    assert model.__class__.__name__ == "InCTRLPQA"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_engine_ic_training.py::test_split_batch_with_optional_masks_supports_four_item_batch /Users/xinye/Desktop/InCTRL/tests/test_engine_ic_training.py::test_build_active_model_returns_inctrl_pqa_for_default_cfg -q
```

Expected: FAIL with `ImportError` or `AttributeError` for the new helper names.

- [ ] **Step 3: Add engine helpers**

Add this import to `/Users/xinye/Desktop/InCTRL/engine_IC.py`:

```python
from open_clip.inctrl_pqa import InCTRLPQA
from open_clip.inctrl_pqa_losses import compute_inctrl_pqa_loss
```

Add these helpers near `_split_query_prompt_batch()`:

```python
def _split_batch_with_optional_masks(batch, device=None):
    if len(batch) == 4:
        inputs, types, labels, masks = batch
    else:
        inputs, types, labels = batch
        masks = None
    query_image, prompt_images = _split_query_prompt_batch(inputs, device)
    if device is not None:
        labels = labels.to(device)
        if masks is not None:
            masks = masks.to(device)
    return query_image, prompt_images, types, labels, masks


def _build_active_model(cfg, model_cfg, cast_dtype, quick_gelu):
    embed_dim = model_cfg["embed_dim"]
    vision_cfg = model_cfg["vision_cfg"]
    text_cfg = model_cfg["text_cfg"]
    if cfg.MODEL.ACTIVE_MODEL == "InCTRLPQA":
        return InCTRLPQA(cfg, embed_dim, vision_cfg, text_cfg, quick_gelu, cast_dtype=cast_dtype)
    return open_clip.model.InCTRL(cfg, embed_dim, vision_cfg, text_cfg, quick_gelu, cast_dtype=cast_dtype)
```

Replace model construction in `train(cfg)` with:

```python
model = _build_active_model(cfg, model_cfg, cast_dtype=cast_dtype, quick_gelu=quick_gelu)
```

Change the loop header and batch unpacking in `train_epoch()` to:

```python
for cur_iter, batch in enumerate(train_loader):
    query_image, prompt_images, types, labels, masks = _split_batch_with_optional_masks(
        batch,
        device=torch.cuda.current_device() if cfg.NUM_GPUS else None,
    )
```

Replace loss computation with:

```python
outputs = model(
    query_image=query_image,
    prompt_images=prompt_images,
    obj_types=types,
    return_aux=False,
    return_dict=True,
)
loss, loss_parts = compute_inctrl_pqa_loss(outputs, labels.float(), masks, cfg)
```

Apply the same `_split_batch_with_optional_masks()` loop pattern in `eval_epoch()`, using:

```python
preds = outputs["final_score"]
```

- [ ] **Step 4: Update inference engine**

In `/Users/xinye/Desktop/InCTRL/engine_test.py`, import `InCTRLPQA`:

```python
from open_clip.inctrl_pqa import InCTRLPQA
```

Replace model construction with:

```python
if cfg.MODEL.ACTIVE_MODEL == "InCTRLPQA":
    model = InCTRLPQA(cfg, embed_dim, vision_cfg, text_cfg, quick_gelu, cast_dtype=cast_dtype)
else:
    model = open_clip.model.InCTRL(cfg, embed_dim, vision_cfg, text_cfg, quick_gelu, cast_dtype=cast_dtype)
```

Replace the `eval_epoch()` loop with this four-value aware form:

```python
for cur_iter, batch in enumerate(val_loader):
    if len(batch) == 4:
        inputs, types, labels, masks = batch
    else:
        inputs, types, labels = batch
    if cfg.NUM_GPUS:
        labels = labels.cuda()
    preds, _ = model(tokenizer, inputs, types, normal_list)
```

- [ ] **Step 5: Run focused engine tests**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_engine_ic_training.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git -C /Users/xinye/Desktop/InCTRL add engine_IC.py engine_test.py tests/test_engine_ic_training.py
git -C /Users/xinye/Desktop/InCTRL commit -m "feat: wire pqa lite model into engines"
```

### Task 9: Local Smoke Entrypoint And Experiment Note

**Files:**
- Create: `/Users/xinye/Desktop/InCTRL/train_local.py`
- Create: `/Users/xinye/Desktop/InCTRL/experiments/2026-04-27-inctrl-context-residual-pqa-lite.md`
- Test: `/Users/xinye/Desktop/InCTRL/train_local.py`

- [ ] **Step 1: Create single-process local runner**

Create `/Users/xinye/Desktop/InCTRL/train_local.py`:

```python
import argparse

from open_clip.config.defaults import assert_and_infer_cfg, get_cfg
from engine_IC import train, test


def parse_args():
    parser = argparse.ArgumentParser(description="Single-process InCTRL PQA Lite smoke runner.")
    parser.add_argument("--normal_json_path", required=True)
    parser.add_argument("--outlier_json_path", required=True)
    parser.add_argument("--val_normal_json_path", required=True)
    parser.add_argument("--val_outlier_json_path", required=True)
    parser.add_argument("--shot", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=240)
    parser.add_argument("--max_epoch", type=int, default=1)
    parser.add_argument("--output_dir", default="./tmp/inctrl_pqa_lite_smoke")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_cfg()
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.normal_json_path = args.normal_json_path
    cfg.outlier_json_path = args.outlier_json_path
    cfg.val_normal_json_path = args.val_normal_json_path
    cfg.val_outlier_json_path = args.val_outlier_json_path
    cfg.shot = args.shot
    cfg.image_size = args.image_size
    cfg.OUTPUT_DIR = args.output_dir
    cfg.SOLVER.MAX_EPOCH = args.max_epoch
    cfg.NUM_GPUS = 1
    cfg.NUM_SHARDS = 1
    cfg.SHARD_ID = 0
    cfg = assert_and_infer_cfg(cfg)
    train(cfg)
    test(cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create experiment note**

Create `/Users/xinye/Desktop/InCTRL/experiments/2026-04-27-inctrl-context-residual-pqa-lite.md`:

```markdown
# InCTRL Context Residual PQA Lite

## Hypothesis

以 InCTRL 少样本上下文残差为主干，保留图像级残差和多层 patch 级残差；增加轻量视觉适配器、对象不可知文本语义分支、提示-查询对齐分割头后，应提升像素级定位能力，并减少由纹理、光照、材料差异导致的 false positives。

## Architecture

- Backbone: frozen CLIP visual/text towers.
- InCTRL evidence: image residual score and multi-layer nearest-neighbor patch residual map.
- Added modules: `VisualAdapter`, `ObjectAgnosticTextBranch`, `PromptQuerySegHead`.
- Final image score: fixed-weight fusion of image residual, max patch residual, PQA global logit, and object-agnostic text logit.
- Final map: fixed-weight fusion of patch residual map, PQA segmentation map, and object-agnostic text map.

## Validation Commands

```bash
python -m py_compile train_local.py open_clip/visual_adapter.py open_clip/object_agnostic_text.py open_clip/prompt_query_head.py open_clip/inctrl_pqa.py open_clip/inctrl_pqa_losses.py open_clip/config/defaults.py
python -m pytest tests/test_visual_adapter.py tests/test_object_agnostic_text.py tests/test_prompt_query_head.py tests/test_inctrl_pqa.py tests/test_inctrl_pqa_losses.py tests/test_engine_ic_training.py -v
```

## Smoke Result Template

| Dataset | Category | Shot | AUROC | AUPR | Baseline AUROC | Delta AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| VisA | candle | 2 |  |  | 0.858 |  |

Baseline source: `reports/original_inctrl_baseline.md`.
```

- [ ] **Step 3: Run compile check**

Run:

```bash
python -m py_compile /Users/xinye/Desktop/InCTRL/train_local.py
```

Expected: no output and exit code 0.

- [ ] **Step 4: Commit**

```bash
git -C /Users/xinye/Desktop/InCTRL add train_local.py experiments/2026-04-27-inctrl-context-residual-pqa-lite.md
git -C /Users/xinye/Desktop/InCTRL commit -m "chore: add pqa lite smoke runner and experiment note"
```

### Task 10: Validation, Smoke Run, And Baseline Delta

**Files:**
- Modify: `/Users/xinye/Desktop/InCTRL/experiments/2026-04-27-inctrl-context-residual-pqa-lite.md`
- Test: full focused validation set

- [ ] **Step 1: Run compile validation**

Run:

```bash
python -m py_compile \
  /Users/xinye/Desktop/InCTRL/train_local.py \
  /Users/xinye/Desktop/InCTRL/open_clip/visual_adapter.py \
  /Users/xinye/Desktop/InCTRL/open_clip/object_agnostic_text.py \
  /Users/xinye/Desktop/InCTRL/open_clip/prompt_query_head.py \
  /Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa.py \
  /Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_losses.py \
  /Users/xinye/Desktop/InCTRL/open_clip/config/defaults.py
```

Expected: no output and exit code 0.

- [ ] **Step 2: Run focused tests**

Run:

```bash
python -m pytest \
  /Users/xinye/Desktop/InCTRL/tests/test_visual_adapter.py \
  /Users/xinye/Desktop/InCTRL/tests/test_object_agnostic_text.py \
  /Users/xinye/Desktop/InCTRL/tests/test_prompt_query_head.py \
  /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa.py \
  /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_losses.py \
  /Users/xinye/Desktop/InCTRL/tests/test_engine_ic_training.py \
  -v
```

Expected: PASS.

- [ ] **Step 3: Run one short smoke experiment**

Run this only in a CUDA environment with the expected checkpoint and data present:

```bash
mkdir -p /Users/xinye/Desktop/InCTRL/tmp/inctrl_pqa_lite_candle_2shot
python /Users/xinye/Desktop/InCTRL/train_local.py \
  --normal_json_path /Users/xinye/Desktop/InCTRL/datasets/AD_json/visa/candle_train_normal.json \
  --outlier_json_path /Users/xinye/Desktop/InCTRL/datasets/AD_json/visa/candle_train_outlier.json \
  --val_normal_json_path /Users/xinye/Desktop/InCTRL/datasets/AD_json/visa/candle_val_normal.json \
  --val_outlier_json_path /Users/xinye/Desktop/InCTRL/datasets/AD_json/visa/candle_val_outlier.json \
  --shot 2 \
  --image_size 240 \
  --max_epoch 1 \
  --output_dir /Users/xinye/Desktop/InCTRL/tmp/inctrl_pqa_lite_candle_2shot \
  2>&1 | tee /Users/xinye/Desktop/InCTRL/tmp/inctrl_pqa_lite_candle_2shot/smoke.log
```

Expected: command prints `AUC-ROC` and `AUC-PR` for the validation split.

- [ ] **Step 4: Record baseline delta**

Run this parser to append the actual smoke values from the log:

```bash
python - <<'PY'
from pathlib import Path
import re

repo = Path("/Users/xinye/Desktop/InCTRL")
log_path = repo / "tmp/inctrl_pqa_lite_candle_2shot/smoke.log"
note_path = repo / "experiments/2026-04-27-inctrl-context-residual-pqa-lite.md"
text = log_path.read_text()
matches = re.findall(r"AUC-ROC:\s*([0-9.]+),\s*AUC-PR:\s*([0-9.]+)", text)
if not matches:
    raise SystemExit(f"No AUC line found in {log_path}")
auroc, aupr = map(float, matches[-1])
baseline = 0.858
delta = auroc - baseline
block = f"""

## Smoke Result

Command log: `tmp/inctrl_pqa_lite_candle_2shot/smoke.log`

| Dataset | Category | Shot | AUROC | AUPR | Baseline AUROC | Delta AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| VisA | candle | 2 | {auroc:.4f} | {aupr:.4f} | {baseline:.3f} | {delta:+.4f} |
"""
note_path.write_text(note_path.read_text().rstrip() + block + "\n")
PY
```

- [ ] **Step 5: Commit validation note**

```bash
git -C /Users/xinye/Desktop/InCTRL add experiments/2026-04-27-inctrl-context-residual-pqa-lite.md
git -C /Users/xinye/Desktop/InCTRL commit -m "docs: record pqa lite validation results"
```

## Self-Review

1. **Spec coverage:**  
   - InCTRL image residual preserved: Tasks 5-6 via `ImageResidualHead` and `image_score/image_logit`.  
   - Multi-layer patch residual preserved: Task 6 via `_compute_patch_residual()` over `PATCH_LAYERS`.  
   - Lightweight visual adapter added: Task 2 and Task 6.  
   - Object-agnostic text branch added: Task 3 and Task 6.  
   - Prompt-query alignment segmentation head added: Task 4 and Task 6.  
   - Pixel-level localization loss added: Task 7.  
   - Entry points wired: Task 8 and Task 9.  
   - Baseline comparison rule covered: Task 10.

2. **Placeholder scan:**  
   The plan uses concrete file paths, concrete commands, and concrete code snippets. Smoke metrics are appended by a parser from `tmp/inctrl_pqa_lite_candle_2shot/smoke.log`, so the experiment note is filled from command output.

3. **Type consistency:**  
   `InCTRLPQA.forward()` returns dict outputs consumed by `compute_inctrl_pqa_loss()`. Legacy tuple output returns `(final_score, image_score)`, matching existing inference callers. Config names match the test in Task 1 and all later references.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-27-inctrl-context-residual-pqa-lite.md`. Two execution options:

**1. Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints.
