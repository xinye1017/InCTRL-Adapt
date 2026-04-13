# InCTRL AdaptCLIP Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align InCTRL with the AdaptCLIP paper design by replacing the text post-MLP with prompt learning, routing the visual adapter through global and three patch levels, and training TA/VA with epoch-level alternating optimization.

**Architecture:** Keep InCTRL's few-shot residual and holistic fusion framework, but change the feature sources so text scores come from a prompt learner and visual residuals come from adapter-updated global and patch features. Alternate optimization by phase rather than optional feature toggles so the full forward graph stays stable while gradient targets switch between prompt learning and visual adaptation.

**Tech Stack:** Python, PyTorch, OpenCLIP, unittest/pytest

---

## File Structure

- Modify: `/Users/xinye/Desktop/InCTRL/open_clip/model.py`
  Responsibility: add `PromptLearner`, prompted text encoding, phase control, and multi-level visual-adapter routing.
- Modify: `/Users/xinye/Desktop/InCTRL/engine_IC.py`
  Responsibility: select `ta`/`va` phase per epoch, freeze/unfreeze trainable modules, and rebuild the optimizer from phase-active parameters.
- Modify: `/Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py`
  Responsibility: add regression tests for prompted text features, patch adapter coverage, and alternating phase parameter control.

### Task 1: Replace Text Post-MLP With Prompt Learner

**Files:**
- Modify: `/Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py`
- Modify: `/Users/xinye/Desktop/InCTRL/open_clip/model.py`
- Test: `/Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py`

- [ ] **Step 1: Write the failing test**

Add these tests to `/Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py`:

```python
class PromptLearnerTests(unittest.TestCase):
    def test_prompted_text_feature_bank_uses_prompt_encoding(self):
        model = InCTRL.__new__(InCTRL)

        class FakePromptLearner:
            def __call__(self, obj_types, device):
                prompts = torch.randn(2, 4, 2, dtype=torch.float32, device=device)
                tokenized = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device=device)
                return prompts, tokenized

        seen = {}

        def fake_encode_text_prompted(prompts, tokenized_prompts, normalize=False):
            seen["prompts_shape"] = tuple(prompts.shape)
            seen["tokenized_shape"] = tuple(tokenized_prompts.shape)
            feats = torch.tensor([[3.0, 0.0], [0.0, 4.0]], dtype=torch.float32)
            return torch.nn.functional.normalize(feats, dim=-1) if normalize else feats

        model.prompt_learner = FakePromptLearner()
        model.encode_text_prompted = fake_encode_text_prompted

        text_features = model._build_prompted_text_features(
            obj_types=["bottle"],
            device=torch.device("cpu"),
        )

        self.assertEqual(seen["prompts_shape"], (2, 4, 2))
        self.assertEqual(seen["tokenized_shape"], (2, 4))
        expected = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        self.assertTrue(torch.allclose(text_features, expected))

    def test_prompt_learner_returns_positive_and_negative_prompt_batches(self):
        learner = PromptLearner.__new__(PromptLearner)
        learner.token_prefix_pos = torch.zeros(1, 1, 1, 2)
        learner.token_suffix_pos = torch.zeros(1, 1, 2, 2)
        learner.token_prefix_neg = torch.zeros(1, 1, 1, 2)
        learner.token_suffix_neg = torch.zeros(1, 1, 2, 2)
        learner.tokenized_prompts_pos = torch.tensor([[[1, 2, 3, 4]]])
        learner.tokenized_prompts_neg = torch.tensor([[[5, 6, 7, 8]]])
        learner.ctx_pos = torch.nn.Parameter(torch.ones(1, 1, 1, 2))
        learner.ctx_neg = torch.nn.Parameter(torch.ones(1, 1, 1, 2) * 2)

        prompts, tokenized_prompts = PromptLearner.forward(
            learner,
            obj_types=["bottle"],
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(prompts.shape), (2, 4, 2))
        self.assertEqual(tuple(tokenized_prompts.shape), (2, 4))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py -q
```

Expected: FAIL with errors such as `AttributeError: 'InCTRL' object has no attribute '_build_prompted_text_features'` or `NameError: name 'PromptLearner' is not defined`.

- [ ] **Step 3: Write minimal implementation**

In `/Users/xinye/Desktop/InCTRL/open_clip/model.py`, remove `self.use_textual_adapter` and `self.text_adapter`, then add a `PromptLearner` and prompted text helpers:

```python
class PromptLearner(nn.Module):
    def __init__(self, clip_model, n_ctx: int = 12):
        super().__init__()
        dtype = clip_model.transformer.get_cast_dtype()
        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.ctx_pos = nn.Parameter(torch.empty(1, 1, n_ctx, ctx_dim, dtype=dtype))
        self.ctx_neg = nn.Parameter(torch.empty(1, 1, n_ctx, ctx_dim, dtype=dtype))
        nn.init.normal_(self.ctx_pos, std=0.02)
        nn.init.normal_(self.ctx_neg, std=0.02)

        prompt_prefix = " ".join(["X"] * n_ctx) + "."
        tokenized_pos = tokenize(prompt_prefix)
        tokenized_neg = tokenize(prompt_prefix)
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_pos).type(dtype).reshape(1, 1, -1, ctx_dim)
            embedding_neg = clip_model.token_embedding(tokenized_neg).type(dtype).reshape(1, 1, -1, ctx_dim)

        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :])
        self.register_buffer("token_suffix_pos", embedding_pos[:, :, 1 + n_ctx :, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx :, :])
        self.register_buffer("tokenized_prompts_pos", tokenized_pos.reshape(1, 1, -1))
        self.register_buffer("tokenized_prompts_neg", tokenized_neg.reshape(1, 1, -1))

    def forward(self, obj_types, device):
        batch_size = len(obj_types)
        prefix_pos = self.token_prefix_pos.expand(batch_size, -1, -1, -1)
        suffix_pos = self.token_suffix_pos.expand(batch_size, -1, -1, -1)
        prefix_neg = self.token_prefix_neg.expand(batch_size, -1, -1, -1)
        suffix_neg = self.token_suffix_neg.expand(batch_size, -1, -1, -1)
        ctx_pos = self.ctx_pos.expand(batch_size, -1, -1, -1)
        ctx_neg = self.ctx_neg.expand(batch_size, -1, -1, -1)

        prompts_pos = torch.cat([prefix_pos, ctx_pos, suffix_pos], dim=2).reshape(batch_size, -1, prefix_pos.shape[-1])
        prompts_neg = torch.cat([prefix_neg, ctx_neg, suffix_neg], dim=2).reshape(batch_size, -1, prefix_neg.shape[-1])
        prompts = torch.cat([prompts_pos, prompts_neg], dim=0).to(device)

        tokenized_pos = self.tokenized_prompts_pos.expand(batch_size, -1, -1).reshape(batch_size, -1)
        tokenized_neg = self.tokenized_prompts_neg.expand(batch_size, -1, -1).reshape(batch_size, -1)
        tokenized_prompts = torch.cat([tokenized_pos, tokenized_neg], dim=0).to(device)
        return prompts, tokenized_prompts
```

Add the new methods inside `InCTRL`:

```python
def encode_text_prompted(self, prompts, tokenized_prompts, normalize: bool = False):
    cast_dtype = self.transformer.get_cast_dtype()
    x = prompts + self.positional_embedding.to(cast_dtype)
    x = x.permute(1, 0, 2)
    x = self.transformer(x, attn_mask=self.attn_mask)
    x = x.permute(1, 0, 2)
    x = self.ln_final(x)
    x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
    return F.normalize(x, dim=-1) if normalize else x

def _build_prompted_text_features(self, obj_types, device):
    prompts, tokenized_prompts = self.prompt_learner(obj_types, device)
    text_features = self.encode_text_prompted(prompts, tokenized_prompts, normalize=True)
    pos_features, neg_features = torch.chunk(text_features, chunks=2, dim=0)
    return torch.stack([pos_features, neg_features], dim=1)
```

Update `__init__`:

```python
self.visual_adapter = Adapter(embed_dim, 4)
self.prompt_learner = PromptLearner(self, n_ctx=12)
```

Update forward to replace:

```python
text_features = self._build_text_feature_bank(...)
```

with:

```python
text_features = self._build_prompted_text_features(
    obj_types=obj_types,
    device=token.device,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py -q
```

Expected: PASS for the two new prompt-learner tests, while the old `test_text_feature_bank_skips_prompted_text_when_textual_adapter_disabled` is removed or replaced.

- [ ] **Step 5: Commit**

```bash
git -C /Users/xinye/Desktop/InCTRL add /Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py /Users/xinye/Desktop/InCTRL/open_clip/model.py
git -C /Users/xinye/Desktop/InCTRL commit -m "feat: add prompt learner textual adapter"
```

### Task 2: Route Visual Adapter Through Global And All Three Patch Levels

**Files:**
- Modify: `/Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py`
- Modify: `/Users/xinye/Desktop/InCTRL/open_clip/model.py`
- Test: `/Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py`

- [ ] **Step 1: Write the failing test**

Add these tests to `/Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py`:

```python
class VisualAdapterPatchTests(unittest.TestCase):
    def test_visual_adapter_updates_all_patch_levels(self):
        model = InCTRL.__new__(InCTRL)

        class CountingAdapter:
            def __init__(self):
                self.calls = []

            def __call__(self, x):
                self.calls.append(tuple(x.shape))
                return x + 1

        model.visual_adapter = CountingAdapter()

        query_patch_levels = torch.zeros(2, 3, 4, 5)
        ref_patch_levels = torch.zeros(2, 3, 8, 5)

        adapted_query, adapted_ref = model._adapt_patch_levels(query_patch_levels, ref_patch_levels)

        self.assertEqual(len(model.visual_adapter.calls), 6)
        self.assertEqual(tuple(adapted_query.shape), (2, 3, 4, 5))
        self.assertEqual(tuple(adapted_ref.shape), (2, 3, 8, 5))
        self.assertTrue(torch.all(adapted_query == 1))
        self.assertTrue(torch.all(adapted_ref == 1))

    def test_patch_residual_map_uses_adapted_patch_levels(self):
        model = InCTRL.__new__(InCTRL)

        def fake_adapt_patch_levels(query_patch_levels, ref_patch_levels):
            return query_patch_levels + 2, ref_patch_levels + 3

        model._adapt_patch_levels = fake_adapt_patch_levels

        query_patch_levels = torch.zeros(1, 3, 225, 2)
        ref_patch_levels = torch.zeros(1, 3, 225, 2)

        patch_map = model._compute_patch_residual_map(query_patch_levels, ref_patch_levels)

        self.assertEqual(tuple(patch_map.shape), (1, 225))
        self.assertTrue(torch.all(patch_map >= 0))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py -q
```

Expected: FAIL with `AttributeError` for `_adapt_patch_levels` and `_compute_patch_residual_map`.

- [ ] **Step 3: Write minimal implementation**

In `/Users/xinye/Desktop/InCTRL/open_clip/model.py`, rename `self.adapter` to `self.visual_adapter` and add two helpers:

```python
def _adapt_patch_levels(self, query_patch_levels, ref_patch_levels):
    adapted_query = []
    adapted_ref = []
    for level_idx in range(query_patch_levels.shape[1]):
        adapted_query.append(self.visual_adapter(query_patch_levels[:, level_idx, :, :]))
        adapted_ref.append(self.visual_adapter(ref_patch_levels[:, level_idx, :, :]))
    return torch.stack(adapted_query, dim=1), torch.stack(adapted_ref, dim=1)

def _compute_patch_residual_map(self, query_patch_levels, ref_patch_levels):
    adapted_query, adapted_ref = self._adapt_patch_levels(query_patch_levels, ref_patch_levels)
    patch_maps = []
    for level_idx in range(adapted_query.shape[1]):
        q = F.normalize(adapted_query[:, level_idx, :, :], dim=-1)
        r = F.normalize(adapted_ref[:, level_idx, :, :], dim=-1)
        sim = 0.5 * (1 - (q @ r.transpose(-2, -1)))
        patch_maps.append(sim.min(dim=2).values)
    return torch.stack(patch_maps, dim=0).mean(dim=0)
```

Update the global path in `forward`:

```python
token_ad = self.visual_adapter(token)
token_n_ad = self.visual_adapter(token_n)
```

Update the patch path in `forward`:

```python
patch_ref_map = self._compute_patch_residual_map(Fp_list, Fp_list_n)
max_diff_score = patch_ref_map.max(dim=1).values
```

Delete the `if self.use_visual_adapter` branch entirely.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py -q
```

Expected: PASS for the new visual-adapter patch tests and existing holistic shape tests.

- [ ] **Step 5: Commit**

```bash
git -C /Users/xinye/Desktop/InCTRL add /Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py /Users/xinye/Desktop/InCTRL/open_clip/model.py
git -C /Users/xinye/Desktop/InCTRL commit -m "feat: route visual adapter through patch levels"
```

### Task 3: Add Alternating Phase Controls On The Model

**Files:**
- Modify: `/Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py`
- Modify: `/Users/xinye/Desktop/InCTRL/open_clip/model.py`
- Test: `/Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py`

- [ ] **Step 1: Write the failing test**

Add these tests to `/Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py`:

```python
class AlternatingPhaseTests(unittest.TestCase):
    def test_ta_phase_only_enables_prompt_learner(self):
        model = InCTRL.__new__(InCTRL)
        torch.nn.Module.__init__(model)
        model.prompt_learner = torch.nn.Linear(4, 4)
        model.visual_adapter = torch.nn.Linear(4, 4)
        model.diff_head = torch.nn.Linear(4, 1)
        model.diff_head_ref = torch.nn.Linear(4, 1)
        model.visual = torch.nn.Linear(4, 4)
        model.transformer = torch.nn.Linear(4, 4)
        model.token_embedding = torch.nn.Embedding(8, 4)
        model.ln_final = torch.nn.LayerNorm(4)

        model.set_alternating_phase("ta")

        self.assertTrue(all(p.requires_grad for p in model.prompt_learner.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.visual_adapter.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.diff_head.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.diff_head_ref.parameters()))

    def test_va_phase_enables_visual_adapter_and_heads(self):
        model = InCTRL.__new__(InCTRL)
        torch.nn.Module.__init__(model)
        model.prompt_learner = torch.nn.Linear(4, 4)
        model.visual_adapter = torch.nn.Linear(4, 4)
        model.diff_head = torch.nn.Linear(4, 1)
        model.diff_head_ref = torch.nn.Linear(4, 1)
        model.visual = torch.nn.Linear(4, 4)
        model.transformer = torch.nn.Linear(4, 4)
        model.token_embedding = torch.nn.Embedding(8, 4)
        model.ln_final = torch.nn.LayerNorm(4)

        model.set_alternating_phase("va")

        self.assertTrue(all(not p.requires_grad for p in model.prompt_learner.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.visual_adapter.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.diff_head.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.diff_head_ref.parameters()))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py -q
```

Expected: FAIL with `AttributeError: 'InCTRL' object has no attribute 'set_alternating_phase'`.

- [ ] **Step 3: Write minimal implementation**

In `/Users/xinye/Desktop/InCTRL/open_clip/model.py`, add phase helpers:

```python
def _set_requires_grad(self, module, enabled: bool):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = enabled

def set_alternating_phase(self, phase: str):
    if phase not in {"ta", "va"}:
        raise ValueError(f"Unsupported phase: {phase}")
    self.training_phase = phase

    self._set_requires_grad(self.visual, False)
    self._set_requires_grad(self.prompt_learner, phase == "ta")
    self._set_requires_grad(self.visual_adapter, phase == "va")
    self._set_requires_grad(self.diff_head, phase == "va")
    self._set_requires_grad(self.diff_head_ref, phase == "va")
```

In `__init__`, initialize the default phase:

```python
self.training_phase = "ta"
self.set_alternating_phase(self.training_phase)
```

If `self.transformer`, `self.token_embedding`, `self.ln_final`, or `self.text_projection` are standalone modules with trainable params, explicitly set them frozen once during init:

```python
for p in self.transformer.parameters():
    p.requires_grad = False
for p in self.token_embedding.parameters():
    p.requires_grad = False
for p in self.ln_final.parameters():
    p.requires_grad = False
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py -q
```

Expected: PASS for the new alternating-phase tests.

- [ ] **Step 5: Commit**

```bash
git -C /Users/xinye/Desktop/InCTRL add /Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py /Users/xinye/Desktop/InCTRL/open_clip/model.py
git -C /Users/xinye/Desktop/InCTRL commit -m "feat: add alternating phase controls"
```

### Task 4: Rebuild Optimizer Per Epoch And Alternate Training In Engine

**Files:**
- Modify: `/Users/xinye/Desktop/InCTRL/engine_IC.py`
- Modify: `/Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py`
- Test: `/Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py`

- [ ] **Step 1: Write the failing test**

Add a lightweight engine-facing regression test to `/Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py`:

```python
from engine_IC import build_phase_optimizer, get_training_phase


class EngineAlternatingTests(unittest.TestCase):
    def test_get_training_phase_alternates_by_epoch(self):
        self.assertEqual(get_training_phase(0), "ta")
        self.assertEqual(get_training_phase(1), "va")
        self.assertEqual(get_training_phase(2), "ta")

    def test_build_phase_optimizer_uses_only_trainable_parameters(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 1))
        for param in model[0].parameters():
            param.requires_grad = False

        optimizer = build_phase_optimizer(model, lr=1e-3)

        params = optimizer.param_groups[0]["params"]
        self.assertEqual(sum(p.numel() for p in params), sum(p.numel() for p in model[1].parameters()))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py -q
```

Expected: FAIL with `ImportError` or `AttributeError` because `build_phase_optimizer` and `get_training_phase` do not exist.

- [ ] **Step 3: Write minimal implementation**

In `/Users/xinye/Desktop/InCTRL/engine_IC.py`, add:

```python
def get_training_phase(epoch_idx: int) -> str:
    return "ta" if epoch_idx % 2 == 0 else "va"


def build_phase_optimizer(model, lr: float = 1e-3):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable_params, lr=lr, betas=[0.9, 0.999])
```

Update the training loop:

```python
for cur_epoch in range(start_epoch, 10):
    phase = get_training_phase(cur_epoch)
    model_ref = model.module if hasattr(model, "module") else model
    model_ref.set_alternating_phase(phase)
    optimizer = build_phase_optimizer(model_ref, lr=1e-3)

    trainable_names = [n for n, p in model_ref.named_parameters() if p.requires_grad]
    print(f"Epoch {cur_epoch} phase: {phase}")
    for name in trainable_names:
        print(f"  trainable: {name}")

    epoch_loss = train_epoch(
        train_loader,
        model,
        optimizer,
        tokenizer,
        cfg,
    )
```

Delete the one-time optimizer construction before the epoch loop:

```python
optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, betas=[0.9, 0.999])
```

and replace the old trainable-parameter logging block with phase-aware logging inside the loop.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py -q
```

Expected: PASS for the engine alternating tests.

- [ ] **Step 5: Run final verification**

Run:

```bash
python -m pytest /Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py -q
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git -C /Users/xinye/Desktop/InCTRL add /Users/xinye/Desktop/InCTRL/engine_IC.py /Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py
git -C /Users/xinye/Desktop/InCTRL commit -m "feat: alternate TA and VA training by epoch"
```

## Self-Review

- Spec coverage: the plan covers prompt learning, three-level patch adaptation, toggle removal, phase-based freezing, and epoch-level alternating optimization.
- Gaps scan: no deferred implementation notes remain.
- Type consistency: the plan uses the same names throughout: `PromptLearner`, `encode_text_prompted`, `_build_prompted_text_features`, `_adapt_patch_levels`, `_compute_patch_residual_map`, `set_alternating_phase`, `get_training_phase`, and `build_phase_optimizer`.
