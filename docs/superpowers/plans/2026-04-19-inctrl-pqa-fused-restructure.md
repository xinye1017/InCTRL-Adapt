# InCTRL PQA Fused Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `InCTRLPQA` so it keeps InCTRL's image/text/patch decision shape while making PQA the single patch-analysis branch and cleaning up the forward and loss contracts.

**Architecture:** Keep the frozen CLIP backbone, the image residual head, and the static text prototype branch. Collapse patch analysis into one coherent branch owned by `PQAdapter`, fuse only `inctrl_patch_map` and `pqa_patch_map`, reduce that once into `patch_logit`, then let `decision_head` fuse `[patch_logit, pqa_logit, image_logit, text_logit]` without `holistic_head` or duplicate residual paths.

**Tech Stack:** Python, PyTorch, OpenCLIP, pytest

---

## File Structure

- Modify: `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_fused.py`
  Responsibility: remove duplicate patch residual paths, simplify `PQAdapter`, delete `holistic_head`, add `fused_patch_map -> patch_logit` reduction, and return the new stable forward contract.
- Modify: `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_losses.py`
  Responsibility: align training losses with `final_logit`, `pqa_logit`, `pqa_local_logits`, and optional `image_logit` only.
- Modify: `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_fused.py`
  Responsibility: lock the new forward contract, gradient flow, and cache behavior.
- Create: `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_losses.py`
  Responsibility: lock the simplified loss contract and zero-loss behavior when masks are absent.

### Task 1: Simplify the top-level forward contract

**Files:**
- Modify: `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_fused.py`
- Modify: `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_fused.py`
- Test: `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_fused.py`

- [ ] **Step 1: Write the failing test**

Replace the existing forward-contract assertions in `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_fused.py` with these tests:

```python
import torch

from open_clip.config.defaults import get_cfg
from open_clip.inctrl_pqa_fused import InCTRLPQA
from open_clip.inctrl_pqa_losses import compute_pqa_mask_loss


def _build_args(image_size=32, shot=2):
    cfg = get_cfg()
    cfg.image_size = image_size
    cfg.shot = shot
    return cfg


def _build_model():
    args = _build_args()
    vision_cfg = {
        "image_size": 32,
        "layers": 12,
        "width": 64,
        "patch_size": 16,
        "head_width": 32,
        "mlp_ratio": 4.0,
        "output_tokens": True,
    }
    text_cfg = {
        "context_length": 77,
        "vocab_size": 49408,
        "width": 32,
        "heads": 4,
        "layers": 2,
    }
    return InCTRLPQA(
        args,
        embed_dim=32,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=False,
        cast_dtype=None,
        patch_layers=(7, 9, 11),
        hidden_dim=16,
        feature_is_projected=False,
    )


def _forward(model):
    query_image = torch.randn(2, 3, 32, 32)
    prompt_images = torch.randn(2, 2, 3, 32, 32)
    return model(
        query_image=query_image,
        prompt_images=prompt_images,
        obj_types=["candle", "candle"],
        return_aux=True,
        return_dict=True,
    )


def test_fused_forward_returns_simplified_contract():
    torch.manual_seed(31)
    model = _build_model()
    model.eval()

    outputs = _forward(model)

    assert set(outputs.keys()) == {
        "final_score",
        "final_logit",
        "patch_score",
        "patch_logit",
        "pqa_score",
        "pqa_logit",
        "image_score",
        "image_logit",
        "text_score",
        "text_logit",
        "fused_patch_map",
        "pqa_local_logits",
        "patch_map_fusion_weights",
        "aux",
    }
    assert outputs["final_logit"].shape == (2,)
    assert outputs["patch_logit"].shape == (2,)
    assert outputs["pqa_logit"].shape == (2,)
    assert outputs["image_logit"].shape == (2,)
    assert outputs["text_logit"].shape == (2,)
    assert outputs["fused_patch_map"].shape == (2, 4)
    assert outputs["pqa_local_logits"].shape == (2, 2, 32, 32)
    assert outputs["aux"]["decision_input"].shape == (2, 4)
    assert "raw_base_patch_map" not in outputs
    assert "base_patch_map" not in outputs
    assert "hybrid_patch_map" not in outputs
    assert "holistic_logit" not in outputs


def test_patch_logit_backprop_updates_patch_fusion_weights():
    torch.manual_seed(47)
    model = _build_model()
    model.train()

    outputs = _forward(model)
    outputs["patch_logit"].sum().backward()

    assert model.patch_map_fusion_logits.grad is not None
    assert model.patch_map_fusion_logits.grad.abs().sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_fused.py -q
```

Expected: FAIL because the current model still returns `base_logit`, `holistic_logit`, `base_patch_map`, `hybrid_patch_map`, and `decision_input` with width 5.

- [ ] **Step 3: Write minimal implementation**

In `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_fused.py`, remove `self.holistic_head`, shrink `decision_head` to width 4, add a helper to reduce the fused patch map, and rewrite the top-level result dictionary.

Change `__init__()` to:

```python
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
self.decision_head = ScalarFusionHead(input_dim=4, hidden_dim=max(hidden_dim // 2, 32))
self.patch_map_fusion_logits = nn.Parameter(torch.zeros(2))
```

Add this helper inside `InCTRLPQA`:

```python
def _reduce_patch_map_to_logit(self, fused_patch_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    patch_score = fused_patch_map.clamp(min=1e-6, max=1.0 - 1e-6)
    topk = min(self.pqa_global_topk, patch_score.shape[-1])
    max_patch_score = patch_score.max(dim=-1).values
    topk_patch_score = patch_score.topk(topk, dim=-1).values.mean(dim=-1)
    patch_score = 0.5 * max_patch_score + 0.5 * topk_patch_score
    patch_logit = self._score_to_logit(patch_score)
    return patch_score, patch_logit
```

Replace the old `holistic_head` path in `forward()` with:

```python
patch_map_fusion_weights = torch.softmax(self.patch_map_fusion_logits, dim=0)
fused_patch_map = (
    patch_map_fusion_weights[0] * inctrl_patch_map
    + patch_map_fusion_weights[1] * pqa_patch_map
)
patch_score, patch_logit = self._reduce_patch_map_to_logit(fused_patch_map)

decision_input = torch.stack(
    [patch_logit, pqa_logit, image_logit, text_logit],
    dim=-1,
)
final_logit = self.decision_head(decision_input)
final_score = torch.sigmoid(final_logit)
```

Return this new main contract:

```python
result = {
    "final_score": final_score,
    "final_logit": final_logit,
    "patch_score": patch_score,
    "patch_logit": patch_logit,
    "pqa_score": torch.sigmoid(pqa_logit),
    "pqa_logit": pqa_logit,
    "image_score": image_score,
    "image_logit": image_logit,
    "text_score": text_score,
    "text_logit": text_logit,
    "fused_patch_map": fused_patch_map,
    "pqa_local_logits": pqa_local_logits,
    "patch_map_fusion_weights": patch_map_fusion_weights,
    "aux": aux,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_fused.py -q
```

Expected: PASS for the new simplified contract test and the new patch-fusion gradient test, while old assertions about `base_patch_map`, `hybrid_patch_map`, and `(2, 5)` no longer exist.

- [ ] **Step 5: Commit**

```bash
git -C /Users/xinye/Desktop/InCTRL add /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_fused.py /Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_fused.py
git -C /Users/xinye/Desktop/InCTRL commit -m "refactor: simplify fused pqa scoring contract"
```

### Task 2: Make `PQAdapter` own the patch branch

**Files:**
- Modify: `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_fused.py`
- Modify: `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_fused.py`
- Test: `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_fused.py`

- [ ] **Step 1: Write the failing test**

Append these adapter-focused tests to `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_fused.py`:

```python
from open_clip.inctrl_pqa_fused import PQAdapter


def test_pq_adapter_returns_single_patch_branch_payload():
    adapter = PQAdapter(feature_dim=8, hidden_dim=4, num_layers=2, image_size=16)
    query_patch_levels = [torch.randn(2, 4, 8), torch.randn(2, 4, 8)]
    prompt_patch_levels = [torch.randn(2, 2, 4, 8), torch.randn(2, 2, 4, 8)]

    outputs = adapter(query_patch_levels=query_patch_levels, prompt_patch_levels=prompt_patch_levels)

    assert set(outputs.keys()) == {
        "inctrl_patch_maps",
        "pqa_patch_maps",
        "pqa_global_logits",
        "pqa_local_logits",
        "aligned_indices",
        "aligned_prompt_features",
        "layer_weights",
    }
    assert len(outputs["inctrl_patch_maps"]) == 2
    assert len(outputs["pqa_patch_maps"]) == 2
    assert len(outputs["pqa_global_logits"]) == 2
    assert len(outputs["pqa_local_logits"]) == 2
    assert outputs["inctrl_patch_maps"][0].shape == (2, 4)
    assert outputs["pqa_patch_maps"][0].shape == (2, 4)
    assert outputs["pqa_global_logits"][0].shape == (2,)
    assert outputs["pqa_local_logits"][0].shape == (2, 2, 16, 16)


def test_forward_aux_keeps_patch_diagnostics_without_duplicate_raw_path():
    torch.manual_seed(53)
    model = _build_model()
    model.eval()

    outputs = _forward(model)
    aux = outputs["aux"]

    assert "inctrl_patch_map" in aux
    assert "pqa_patch_map" in aux
    assert "aligned_indices" in aux
    assert "aligned_prompt_features" in aux
    assert "patch_fusion_weights" in aux
    assert "per_layer_inctrl_patch_map" in aux
    assert "per_layer_pqa_patch_map" in aux
    assert "per_layer_pqa_global_logit" in aux
    assert "per_layer_raw_residual" not in aux
    assert "raw_base_patch_map_2d" not in aux
    assert "base_patch_map_2d" not in aux
    assert "hybrid_patch_map_2d" not in aux
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_fused.py -q
```

Expected: FAIL because `PQAdapter.forward()` still returns `patch_logits`, `patch_scores`, `residual_maps`, and the old aux payload still includes raw/base/hybrid duplicate diagnostics.

- [ ] **Step 3: Write minimal implementation**

In `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_fused.py`, rewrite `PQAdapter.forward()` so it only returns one coherent patch payload.

Replace the tail of `PQAdapter.forward()` with:

```python
inctrl_patch_maps = []
pqa_patch_maps = []
pqa_global_logits = []
pqa_local_logits = []
aligned_indices = []
aligned_prompt_features = []

for layer_idx, (query_level, prompt_level) in enumerate(zip(query_levels, prompt_levels)):
    num_patches = query_level.shape[1]
    grid_side = int(num_patches ** 0.5)
    prompt_flat = self._flatten_prompt_level(prompt_level)
    residual, best_indices, aligned_prompt = self._match_prompt_patches(query_level, prompt_flat)
    context_map = self._build_context_map(query_level, aligned_prompt, beta_value, layer_idx)
    local_logit_2c, _, _, patch_score = self._compute_local_outputs(
        context_map=context_map,
        layer_idx=layer_idx,
        grid_side=grid_side,
    )
    pqa_global_logit_2c = self.global_heads[layer_idx](context_map.flatten(2))
    pqa_global_logit = pqa_global_logit_2c[:, 1] - pqa_global_logit_2c[:, 0]

    inctrl_patch_maps.append(residual)
    pqa_patch_maps.append(patch_score)
    pqa_global_logits.append(pqa_global_logit)
    pqa_local_logits.append(local_logit_2c)
    aligned_indices.append(best_indices)
    aligned_prompt_features.append(aligned_prompt)

return {
    "inctrl_patch_maps": inctrl_patch_maps,
    "pqa_patch_maps": pqa_patch_maps,
    "pqa_global_logits": pqa_global_logits,
    "pqa_local_logits": pqa_local_logits,
    "aligned_indices": aligned_indices,
    "aligned_prompt_features": aligned_prompt_features,
    "layer_weights": self._get_layer_weights(query_levels[0].device),
}
```

Then remove `_compute_patch_residuals()` entirely and aggregate directly from `pq_outputs` in `InCTRLPQA.forward()`:

```python
pq_outputs = self.prompt_query_adapter(
    query_patch_levels=query_patch_level_list,
    prompt_patch_levels=prompt_patch_levels,
    beta=beta_value,
)
layer_weights = pq_outputs["layer_weights"].to(query_patch_level_list[0].dtype)
inctrl_patch_map = sum(
    weight * patch_map
    for weight, patch_map in zip(layer_weights, pq_outputs["inctrl_patch_maps"])
)
pqa_patch_map = sum(
    weight * patch_map
    for weight, patch_map in zip(layer_weights, pq_outputs["pqa_patch_maps"])
)
pqa_logit = sum(
    weight * global_logit
    for weight, global_logit in zip(layer_weights, pq_outputs["pqa_global_logits"])
)
pqa_local_logits = sum(
    weight * local_logit
    for weight, local_logit in zip(layer_weights, pq_outputs["pqa_local_logits"])
)
```

Replace the aux payload with:

```python
aux = {
    "inctrl_patch_map": inctrl_patch_map,
    "pqa_patch_map": pqa_patch_map,
    "per_layer_inctrl_patch_map": pq_outputs["inctrl_patch_maps"],
    "per_layer_pqa_patch_map": pq_outputs["pqa_patch_maps"],
    "per_layer_pqa_global_logit": pq_outputs["pqa_global_logits"],
    "aligned_indices": pq_outputs["aligned_indices"],
    "aligned_prompt_features": pq_outputs["aligned_prompt_features"],
    "patch_fusion_weights": patch_map_fusion_weights,
    "decision_input": decision_input,
    "text_prototypes": {"normal": normal_proto, "anomaly": anomaly_proto},
}
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_fused.py -q
```

Expected: PASS for the adapter payload test and the aux diagnostic cleanup test.

- [ ] **Step 5: Commit**

```bash
git -C /Users/xinye/Desktop/InCTRL add /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_fused.py /Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_fused.py
git -C /Users/xinye/Desktop/InCTRL commit -m "refactor: streamline pqa patch branch"
```

### Task 3: Align the loss contract with the simplified model

**Files:**
- Create: `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_losses.py`
- Modify: `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_losses.py`
- Test: `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_losses.py`

- [ ] **Step 1: Write the failing test**

Create `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_losses.py` with:

```python
import torch

from open_clip.inctrl_pqa_losses import compute_pqa_mask_loss, compute_training_loss


def test_compute_training_loss_uses_simplified_logits_only():
    outputs = {
        "final_logit": torch.tensor([0.2, -0.1], requires_grad=True),
        "patch_logit": torch.tensor([0.4, -0.5], requires_grad=True),
        "pqa_logit": torch.tensor([0.3, -0.2], requires_grad=True),
        "image_logit": torch.tensor([0.1, -0.4], requires_grad=True),
        "pqa_local_logits": torch.randn(2, 2, 8, 8, requires_grad=True),
    }
    labels = torch.tensor([1.0, 0.0])
    masks = torch.zeros(2, 1, 8, 8)
    masks[0, :, 2:6, 2:6] = 1.0
    loss_fn = torch.nn.BCEWithLogitsLoss()

    total_loss, metrics = compute_training_loss(
        outputs=outputs,
        labels=labels,
        loss_fn=loss_fn,
        masks=masks,
        pqa_loss_weight=1.0,
        mask_loss_weight=1.0,
        image_loss_weight=0.5,
    )

    assert total_loss.requires_grad
    assert set(metrics.keys()) == {
        "final_loss",
        "image_loss",
        "pqa_loss",
        "pqa_mask_loss",
        "total_loss",
    }
    total_loss.backward()
    assert outputs["final_logit"].grad is not None
    assert outputs["pqa_logit"].grad is not None
    assert outputs["image_logit"].grad is not None
    assert outputs["pqa_local_logits"].grad is not None


def test_compute_pqa_mask_loss_returns_zero_without_masks():
    outputs = {
        "final_logit": torch.tensor([0.1], requires_grad=True),
        "pqa_local_logits": torch.randn(1, 2, 8, 8, requires_grad=True),
    }

    loss = compute_pqa_mask_loss(outputs, masks=None)

    assert torch.equal(loss, outputs["final_logit"].new_zeros(()))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_losses.py -q
```

Expected: FAIL until the simplified training loss path is confirmed and the new test file exists.

- [ ] **Step 3: Write minimal implementation**

Update `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_losses.py` so `compute_training_loss()` depends only on the simplified outputs:

```python
def compute_training_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    phase: Optional[str] = None,
    masks: Optional[torch.Tensor] = None,
    pqa_loss_weight: float = 1.0,
    mask_loss_weight: float = 1.0,
    image_loss_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    del phase
    labels = labels.float()
    zero = outputs["final_logit"].new_zeros(())
    final_loss = loss_fn(outputs["final_logit"], labels)
    pqa_loss = loss_fn(outputs["pqa_logit"], labels) if pqa_loss_weight > 0 else zero
    image_loss = loss_fn(outputs["image_logit"], labels) if image_loss_weight > 0 else zero
    pqa_mask_loss = compute_pqa_mask_loss(outputs, masks) if masks is not None and mask_loss_weight > 0 else zero
    total_loss = final_loss + pqa_loss_weight * pqa_loss + image_loss_weight * image_loss + mask_loss_weight * pqa_mask_loss

    return total_loss, {
        "final_loss": final_loss.detach(),
        "image_loss": image_loss.detach(),
        "pqa_loss": pqa_loss.detach(),
        "pqa_mask_loss": pqa_mask_loss.detach(),
        "total_loss": total_loss.detach(),
    }
```

Do not add `base_logit` or `holistic_logit` back into the loss contract.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_losses.py -q
```

Expected: PASS for both simplified loss tests.

- [ ] **Step 5: Commit**

```bash
git -C /Users/xinye/Desktop/InCTRL add /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_losses.py /Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_losses.py
git -C /Users/xinye/Desktop/InCTRL commit -m "test: cover simplified fused pqa losses"
```

### Task 4: Lock cache behavior and run the focused regression suite

**Files:**
- Modify: `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_fused.py`
- Modify: `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_fused.py`
- Test: `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_fused.py`
- Test: `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_losses.py`

- [ ] **Step 1: Write the failing test**

Append these cache regressions to `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_fused.py`:

```python
def test_prompt_feature_cache_round_trip_still_works():
    torch.manual_seed(59)
    model = _build_model()
    model.eval()

    prompt_images = torch.randn(2, 3, 32, 32)
    cache = model.build_prompt_feature_cache(prompt_images=prompt_images)

    assert cache["prompt_global"].shape == (2, 32)
    assert len(cache["prompt_patch_levels"]) == 3
    assert cache["prompt_patch_levels"][0].shape == (2, 4, 32)
    assert cache["num_shots"] == 2


def test_text_prototype_cache_round_trip_still_works():
    torch.manual_seed(61)
    model = _build_model()
    model.eval()

    cache = model.build_text_prototype_cache(obj_types=["candle", "capsule"], device=torch.device("cpu"))

    assert cache["normal_proto"].shape == (2, 32)
    assert cache["anomaly_proto"].shape == (2, 32)
```

- [ ] **Step 2: Run test to verify it fails if cache behavior drifted**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_fused.py -q
```

Expected: PASS if cache behavior is still intact. If it fails, fix the cache shape regressions before moving on.

- [ ] **Step 3: Write minimal implementation**

Keep `build_prompt_feature_cache()` and `build_text_prototype_cache()` stable. If Task 1 or Task 2 changed tensor ranks or dtypes, adjust only the shape-handling lines needed to preserve these outputs.

The end state must keep:

```python
return {
    "prompt_global": prompt_global.squeeze(0).detach(),
    "prompt_patch_levels": [level.squeeze(0).detach() for level in prompt_patch_level_list],
    "num_shots": num_shots,
}
```

and:

```python
return {
    "normal_proto": normal_proto.detach(),
    "anomaly_proto": anomaly_proto.detach(),
}
```

Also remove dead code and exports tied to the deleted holistic path:

```python
__all__ = [
    "InCTRLPQA",
    "ImageResidualHead",
    "PQAdapter",
    "PQAGlobalHead",
    "ScalarFusionHead",
]
```

- [ ] **Step 4: Run the focused regression suite**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_fused.py /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_losses.py -q
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git -C /Users/xinye/Desktop/InCTRL add /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_fused.py /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_losses.py /Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_fused.py /Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_losses.py
git -C /Users/xinye/Desktop/InCTRL commit -m "refactor: finish fused pqa branch cleanup"
```

## Self-Review

### Spec coverage

- Preserve InCTRL image branch: covered in Task 1 by keeping `image_head` in the final decision contract.
- Preserve InCTRL text branch: covered in Task 1 by keeping `text_logit` and `text_score` in the final decision contract.
- Make PQA the single patch branch: covered in Task 2 by rewriting `PQAdapter.forward()` and removing `_compute_patch_residuals()`.
- Remove duplicated residual paths: covered in Task 2 by deleting raw/base duplicate patch diagnostics.
- Remove `holistic_head`: covered in Task 1 by deleting it from `__init__()`, `get_trainable_parameters()`, and `forward()`.
- Simplify loss contract: covered in Task 3.
- Preserve cache builders: covered in Task 4.

### Placeholder scan

- No `TBD`, `TODO`, or "implement later" markers remain.
- Every code step includes concrete code snippets.
- Every test step includes concrete commands and expected outcomes.

### Type consistency

- `decision_input` width is 4 in tests, implementation, and acceptance criteria.
- The patch branch names are consistent across plan steps: `inctrl_patch_map`, `pqa_patch_map`, `fused_patch_map`, `patch_logit`.
- Cache keys stay consistent across tests and implementation snippets.
