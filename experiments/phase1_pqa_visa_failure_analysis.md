# Phase 1 — PQA Cross-Domain Failure Analysis (VisA / AITEX)

**Date**: 2026-04-21
**Branch**: `feat/inctrl-pqa-fused-restructure`
**Scope**: Diagnose why the `InCTRLPQA` fused model trained on MVTec fails to recover the published baseline on VisA and AITEX even after Phase 1 regularization (prior KL loss).
**Status**: Diagnostic — drives Phase 2 direction. No immediate code fix proposed here; this note is reference material for subsequent phases.

---

## 1. Numeric Evidence

All runs use identical data / seed / model class (`InCTRLPQA` in `open_clip/inctrl_pqa_fused.py`). They differ only in training duration and `--prior-loss-weight`.

| Setting | AITEX-2 | **AITEX-4** | AITEX-8 | ELPV-2 | **ELPV-4** | ELPV-8 | VisA-2 | **VisA-4** | VisA-8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step 1 (40 steps, prior=0) | 0.681 | 0.696 | 0.755 | 0.854 | 0.843 | 0.846 | 0.830 | **0.832** | 0.828 |
| Step 2 (1000 steps, prior=0) | 0.654 | **0.676** | 0.742 | 0.887 | **0.886** | 0.885 | 0.788 | **0.783** | 0.781 |
| Step 3 (1000 steps, prior=0.1) | 0.699 | **0.702** | 0.757 | 0.884 | **0.884** | 0.883 | 0.771 | **0.779** | 0.774 |
| **Baseline (in-domain)** | 0.761 | **0.790** | 0.806 | 0.839 | **0.846** | 0.872 | 0.858 | **0.877** | 0.887 |
| **Phase 1 exit (4-shot)** | — | **0.73** | — | — | **0.82** | — | — | **0.80** | — |

(Baselines sourced from `reports/original_inctrl_baseline.md`.)

### 1.1 Critical observation — longer training *hurts* AITEX and VisA

Step 1 (40 training steps ≈ near the pretrained starting point) already clears the VisA 4-shot exit threshold (0.832 vs 0.80). Step 2 (1000 steps) *drops 5 points* to 0.783. Step 3's prior anchor only partially compensates (0.779).

**This is the signature of single-source overfitting, not of insufficient capacity.**

### 1.2 `delta_vs_zero_shot` as health indicator

From `analytics.baseline_deltas` in Step 3:

| Pair | shot=2 | shot=4 | shot=8 | Diagnosis |
| --- | ---: | ---: | ---: | --- |
| MVTec → AITEX | -0.034 | -0.031 | +0.024 | Training pulls below CLIP zero-shot at low shot, only recovers at 8-shot |
| MVTec → ELPV | +0.151 | +0.151 | +0.150 | Healthy, consistent +15pt lift — MVTec features transfer |
| MVTec → VisA | -0.010 | -0.002 | -0.007 | **Stuck on the zero-shot line, training produces ~0 net value** |

VisA's `delta_vs_zero_shot ≈ 0` across all shots means the learned PQA/image/fused cues are *net-neutral* — whatever signal the training extracts is cancelled out by domain-shift-induced noise.

### 1.3 Why prior=0.1 only moves AITEX

Step 2 → Step 3 deltas at 4-shot:
- AITEX: 0.676 → 0.702 (**+0.026**, ~50% of the gap to threshold closed)
- ELPV: 0.886 → 0.884 (-0.002, noise)
- VisA: 0.783 → 0.779 (-0.004, noise)

The KL anchor pulls `final_logit` toward `text_score` (CLIP zero-shot). This helps AITEX (whose zero-shot 0.733 > our result 0.676). It cannot help VisA because VisA's zero-shot (0.781) ≈ our result (0.783) — **we are already at the prior.** The prior anchor has no gradient to apply where it matters.

---

## 2. PQA Failure Mechanisms (Code-Grounded)

Three specific mechanisms, in decreasing order of confidence.

### 2.1 BatchNorm distribution memorization (highest confidence)

`@/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_fused.py:199-214` — `_build_context_map`:

```python
query_level = F.normalize(query_level, dim=-1)
aligned_prompt = F.normalize(aligned_prompt, dim=-1)
context_feat = query_level + beta_value * torch.abs(query_level - aligned_prompt)
context_map = context_feat.permute(0, 2, 1).reshape(batch_size, feature_dim, grid_side, grid_side)
return self.sharebn[layer_idx](context_map)
```

`self.sharebn[layer_idx]` is a `BatchNorm2d` per layer (`:170-172`). After 1000 steps on MVTec the running mean/var converge to MVTec's `context_map` statistics. At inference on VisA:

- VisA's normalized query features have different per-channel statistics than MVTec's
- BN normalizes with the *wrong* mean/var
- Downstream `PQAConvLocalHead` and `PQAGlobalHead` see inputs from a different distribution than they were trained on
- Predictions degrade

**Why this is the primary suspect**: BN affects *all* downstream learnable heads at *every* forward pass, and it is completely unreachable by the `prior_loss` KL anchor (which only touches `final_logit`). This matches the observed pattern where VisA degrades *regardless* of prior strength.

### 2.2 Cosine nearest-neighbor signal collapse on heterogeneous textures (medium confidence)

`@/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_fused.py:184-197` — `_match_prompt_patches`:

```python
similarity = torch.einsum("bnc,bmc->bnm", query_norm, prompt_norm)
max_cosine, best_indices = similarity.max(dim=-1)
...
residual = 0.5 * (1.0 - max_cosine)
```

The residual signal assumes that *for a normal query patch there exists a highly-similar prompt patch* (so `max_cosine → 1, residual → 0`) and *for an anomalous query patch no prompt patch matches well* (so `max_cosine → 0, residual → 0.5`).

This assumption holds on MVTec textures (carpet / leather / tile), which are quasi-stationary — most normal patches look alike, and anomalies are clear outliers.

It breaks on VisA:
- PCB / capsule / candle categories are *highly heterogeneous*: normal patches vary substantially across the image (different components, regions, colors)
- For any query patch there exists some "coincidentally similar" prompt patch somewhere in the prompt set (because the prompt-patch pool is large: `num_prompts × num_patches_per_prompt`, e.g. 4 × 256 = 1024 candidates)
- `max_cosine` stays high for *both* normal and anomalous queries → residual signal drowns in noise

This is an *algorithmic* limitation, not a training artifact. No amount of regularization fixes it.

### 2.3 Over-capacity learnable heads specialize to MVTec anomaly morphology (medium confidence)

`@/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_fused.py:173-178` — PQAdapter has 3 layers × 2 heads (`PQAConvLocalHead` + `PQAGlobalHead`). Rough parameter count:

- Each `PQAConvLocalHead`: Conv(768→128) + Conv(128→64) + Conv(64→2) ≈ 1M params
- Each `PQAGlobalHead.mlp_adapter`: Linear(768→128) + LN + Linear(128→2) ≈ 100K params
- × 3 layers → **~3.3M trainable params in PQA alone**

MVTec anomalies are mostly small blobs, scratches, holes, discoloration — stereotyped visual primitives. 3M convolutional parameters trained on MVTec converge to filters tuned for these primitives.

VisA anomalies are different: bent pins, missing components, wrong color, misaligned parts, missing text. These don't match MVTec's anomaly filter bank. The conv heads produce noise instead of signal.

**Evidence for this hypothesis**: if it were *only* the BN issue, we would expect roughly uniform degradation across tasks. We see instead that ELPV (whose anomalies — solar panel cracks, dark bands — are visually closer to MVTec scratch/discoloration) *improves* with longer training (+0.04), while VisA regresses. This correlation between "visual similarity to MVTec anomalies" and "longer training helps" is consistent with conv-filter specialization.

---

## 3. Why AdaptCLIP Outperforms (Code Contrast)

AdaptCLIP (`@/Users/xinye/Desktop/InCTRL/open_clip/adaptclip.py:715-800`) uses a structurally similar `PQAdapter` — same cosine alignment, same `abs(query - aligned_prompt)` context, same per-layer BN + conv. The algorithm is not the differentiator. Three system-level differences matter.

### 3.1 Learnable textual adapter for cross-domain text alignment

`@/Users/xinye/Desktop/InCTRL/open_clip/adaptclip.py:846` — `TextualAdapter` introduces learnable context tokens (`n_ctx_pos`, `n_ctx_neg`) that adapt text prompts across domains during training.

Our `InCTRLPQA` text path (`@/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_fused.py:949-974`) uses static hard-coded templates only. The text branch *cannot learn*. Result: when visual heads drift toward MVTec during training, nothing drifts correspondingly on the text side to stabilize the fused decision. The text prior is frozen at CLIP's zero-shot capability.

### 3.2 Alignment score used as direct, parameter-free output

AdaptCLIP exposes the raw cosine residual (`align_scores`) as a final anomaly-map component:

```python
align_score, min_idx = torch.min(1.0 - torch.bmm(query, prompt.T), dim=-1)
align_score = F.interpolate(align_score, size=(img_size, img_size))
align_scores.append(align_score)  # → directly part of final output
```

We retain this as `inctrl_patch_map` in `fused_patch_map` (`@/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_fused.py:919-936`) with a learnable softmax mixing weight:

```python
fused_patch_map = (
    patch_map_fusion_weights[0] * inctrl_patch_map
    + patch_map_fusion_weights[1] * pqa_patch_map
)
```

During MVTec training, the learnable `patch_map_fusion_logits` drift toward favoring `pqa_patch_map` (which fits MVTec) over the parameter-free `inctrl_patch_map` (which generalizes better). The very mechanism that should let the robust signal dominate gets trained away.

### 3.3 Multi-domain training (not architectural)

AdaptCLIP paper results use multi-source training (MVTec + VisA + others in various experiments). Our Step 1-3 runs are single-source MVTec training. This is a setup difference, not an algorithm difference — but it is *the* reason VisA fails: there is no VisA signal in the training set to counteract MVTec-specific drift.

**Implication**: We should not expect to match AdaptCLIP's cross-domain numbers under strict single-source training. Our realistic ceiling is something like "close to zero-shot on VisA, meaningful lift on MVTec-like domains (ELPV)."

---

## 4. Phase 2 Options (Ranked)

Ordered by expected-benefit / implementation-cost ratio. Each is a hypothesis with an expected failure mode.

### Option A — Replace BN with LN in `PQAdapter.sharebn` (1 line)

Change `@/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_fused.py:170-172`:

```python
# from:
self.sharebn = nn.ModuleList(nn.BatchNorm2d(feature_dim) for _ in range(num_layers))
# to:
self.sharebn = nn.ModuleList(
    nn.GroupNorm(1, feature_dim) for _ in range(num_layers)  # GN(1) == LN over (C,H,W)
)
```

LayerNorm-like normalization computes per-sample statistics → immune to the MVTec→VisA running-stat shift described in §2.1.

- **Predicted VisA 4-shot uplift**: +0.02 ~ +0.05
- **Risk**: MVTec in-distribution accuracy may drop 0.005–0.01 because BN's population statistics can be slightly more informative than per-sample normalization on a homogeneous source
- **Testing cost**: 1 line change + 1 unit test + one training run

### Option B — Freeze learnable `local_heads` + `global_heads`, keep only BN + fusion weights + `decision_head` trainable

Modify `@/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_fused.py:845-860` `get_trainable_parameters()`:

```python
# exclude:
# - self.prompt_query_adapter.local_heads
# - self.prompt_query_adapter.global_heads
# - self.image_head
# include only:
# - self.prompt_query_adapter.sharebn (re-trains BN stats, but no conv specialization)
# - patch_map_fusion_logits
# - pqa_layer_weights / patch_layer_weights
# - decision_head.branch_logits + bias
```

This reduces trainable count from ~3M to ~20, preventing MVTec anomaly-morphology specialization (§2.3). The architecture effectively becomes "frozen CLIP + frozen heads + tiny learnable mixer."

- **Predicted VisA 4-shot uplift**: +0.03 ~ +0.08
- **Risk**: severe drop on in-domain metrics if we ever train on a target domain — but we don't, our protocol is strict cross-domain only
- **Testing cost**: 10 lines in `get_trainable_parameters` + `--freeze-pqa-heads` CLI flag + test

### Option C — A + B combined

LN + frozen heads = "closest thing we can cheaply do to match AdaptCLIP's frozen-backbone-thin-adapter spirit while keeping our decision-head infrastructure intact."

- **Predicted VisA 4-shot uplift**: +0.05 ~ +0.10
- **Risk**: low — both changes are batch-independent / reduce overfitting
- **Testing cost**: two flags, one run

### Option D — Learnable textual prompt tokens (high effort)

Add ~100 LOC mirroring AdaptCLIP's `TextualAdapter`. Requires new loss terms to prevent collapse (e.g., orthogonality between normal/anomaly context tokens). Only worth pursuing if A+B+C fails and we need to close the gap to paper numbers.

- **Predicted VisA 4-shot uplift**: +0.02 ~ +0.05 *on top of A+B*
- **Risk**: high — loss design is non-trivial; easy to make it worse
- **Testing cost**: 1-2 days of engineering + multiple ablation runs

### Option E — Abandon single-source protocol (out of scope)

Train on MVTec + VisA jointly. This would trivially fix VisA but violates the leave-one-out cross-domain protocol that the rest of the experimental setup (`TEST_DATASETS_BY_TRAIN`) enforces. Not a viable option under current constraints.

---

## 5. Decision Log

| When | Decision | Rationale |
| --- | --- | --- |
| 2026-04-21 | Run Step 4 (prior=0.3) *before* any Phase 2 code change | Low cost (one CLI flag), rules out the "prior strength insufficient" hypothesis cleanly |
| If Step 4 passes VisA ≥ 0.80 | Close Phase 1, document numbers | Exit criteria met, no more work needed for Phase 1 |
| If Step 4 fails VisA | Implement Option A (BN→LN) first | Cheapest fix targeting the highest-confidence failure mode |
| If A alone fails | Implement Option C (A+B combined) | Second-cheapest, addresses two failure modes simultaneously |
| If C fails | Re-evaluate — possibly Option D or accept "zero-shot ceiling on VisA" | Algorithmic limits may be reached |

---

## 6. Open Questions for Future Investigation

1. **Is BN the primary culprit, or is conv-head specialization equally important?**
   Diagnostic: run Option A alone vs Option B alone, compare VisA deltas. If A >> B, BN wins. If similar, both mechanisms contribute.

2. **Does `patch_map_fusion_logits` actually drift toward PQA during training?**
   Check `aux["patch_map_fusion_weights"]` across a training run. If it stays close to `[0.5, 0.5]`, §3.2 hypothesis is wrong. If it drifts hard toward PQA (e.g., `[0.1, 0.9]`), this confirms the "fusion head trains away the robust signal" claim.

3. **What fraction of the PQA failure on VisA is algorithmic (§2.2) vs trainable (§2.1, 2.3)?**
   Diagnostic: compare `max_cosine` distributions on normal vs anomalous VisA patches at Step 0 (pre-training). If distributions already overlap heavily, §2.2 is the bottleneck and no amount of training tricks will help.

These questions are worth tooling up *if* Option C fails to clear VisA. Otherwise they are noise.

---

## 7. References

- `@/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_fused.py` — current `InCTRLPQA` model
- `@/Users/xinye/Desktop/InCTRL/open_clip/adaptclip.py` — AdaptCLIP reference implementation
- `@/Users/xinye/Desktop/InCTRL/reports/original_inctrl_baseline.md` — baseline AUROC numbers
- `@/Users/xinye/Desktop/InCTRL/train_local.py` — `_PUBLISHED_BASELINE_AUROC`, `_PHASE1_EXIT_THRESHOLDS_4SHOT`
- Step 3 summary JSON: `/Users/xinye/Downloads/cross_shot_train_shot_4_summary (3).json` (archived in this session)
