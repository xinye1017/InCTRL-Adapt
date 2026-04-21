# Phase 1 — Cross-Domain Regression Analysis: InCTRLPQA vs Original InCTRL

**Date**: 2026-04-21 (rev 2)
**Branch**: `feat/inctrl-pqa-fused-restructure`
**Scope**: Diagnose why `InCTRLPQA` loses ~10 AUROC points relative to the original `InCTRL` on MVTec→VisA **cross-domain** evaluation (same protocol). Identify specific architectural regression sources introduced by the PQA refactor.
**Status**: Diagnostic — drives Phase 2 ablations. Option A (mask_loss=0) is the highest-priority experiment.

---

## 1. Numeric Evidence

All runs use identical data / seed / model class (`InCTRLPQA` in `open_clip/inctrl_pqa_fused.py`). They differ only in training duration and `--prior-loss-weight`.

| Setting | AITEX-2 | **AITEX-4** | AITEX-8 | ELPV-2 | **ELPV-4** | ELPV-8 | VisA-2 | **VisA-4** | VisA-8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step 1 (40 steps, prior=0) | 0.681 | 0.696 | 0.755 | 0.854 | 0.843 | 0.846 | 0.830 | **0.832** | 0.828 |
| Step 2 (1000 steps, prior=0) | 0.654 | **0.676** | 0.742 | 0.887 | **0.886** | 0.885 | 0.788 | **0.783** | 0.781 |
| Step 3 (1000 steps, prior=0.1) | 0.699 | **0.702** | 0.757 | 0.884 | **0.884** | 0.883 | 0.771 | **0.779** | 0.774 |
| **Original InCTRL (cross-domain)** | 0.761 | **0.790** | 0.806 | 0.839 | **0.846** | 0.872 | 0.858 | **0.877** | 0.887 |
| **Gap vs original (Step 3)** | -0.062 | **-0.088** | -0.049 | +0.045 | **+0.038** | +0.011 | -0.087 | **-0.098** | -0.113 |
| **Sanity floor (4-shot)** | — | **0.73** | — | — | **0.82** | — | — | **0.80** | — |

(Baselines sourced from `reports/original_inctrl_baseline.md`. **All baseline numbers are cross-domain**: original InCTRL trained on MVTec, tested on X.)

**Critical reframing**: The gap is NOT a "cross-domain penalty" — original InCTRL achieves these numbers under the exact same protocol. The gap is a **regression introduced by the InCTRLPQA refactor**. The sanity floors (formerly "Phase 1 exit") are minimum thresholds; the true target is matching the original numbers.

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

## 2. Why PQA Is Not a Stateless Feature Processor

A common intuition: "PQA just processes image features, so adding it can only help." This is true for *stateless* modules (e.g., cosine similarity). It is false for our `PQAAdapter`, which carries **~3.4M trainable parameters** that memorize MVTec statistics:

| State | Location | Cross-domain behaviour |
| --- | --- | --- |
| `BatchNorm2d` running mean/var | `inctrl_pqa_fused.py:170-172` | Uses MVTec-learned channel statistics to normalize VisA inputs → wrong normalization |
| `PQAConvLocalHead` conv filters × 3 | `inctrl_pqa_fused.py:173-175` | Trained to detect MVTec anomaly shapes → fires on wrong patterns in VisA |
| `PQAGlobalHead.mlp_adapter` × 3 | `inctrl_pqa_fused.py:176-178` | Maps MVTec context-map statistics → anomaly score; wrong on VisA distributions |
| `decision_head.branch_logits` | `inctrl_pqa_fused.py:135` | Learns "trust PQA, ignore text" on MVTec; applies verbatim to VisA |
| `patch_map_fusion_logits` | `inctrl_pqa_fused.py` | Learns to over-weight PQA patch map vs robust cosine residual |

Even when the *prompt data* at inference is VisA, the *processing pipeline* was shaped by MVTec. The module gives **confident wrong predictions**, not "neutral extra information." In the decision head's convex combination, a confident wrong logit (e.g., `pqa_logit = +3.0`) overwhelms a correct small-magnitude text logit (e.g., `text_logit = -0.5`), producing a worse final answer than text alone.

---

## 3. Architectural Regression Sources (Code Diff: Original InCTRL vs InCTRLPQA)

The original `InCTRL` class lives in `open_clip/model.py:443-592`. The current `InCTRLPQA` is in `open_clip/inctrl_pqa_fused.py:310+`. Direct comparison reveals six regression sources, ranked by expected contribution to the VisA drop.

### 3.0 Parameter Count Explosion (1000×)

| Module | Original InCTRL | InCTRLPQA |
| --- | ---: | ---: |
| Global-feature `Adapter(640, 4)` | ~2.5K | — |
| `ImageResidualHead` | — | ~100K |
| `diff_head: Linear(225, 1)` | ~225 | — |
| `diff_head_ref: Linear(640, 1)` | ~640 | — |
| `PQAConvLocalHead` × 3 | — | **~3M** |
| `PQAGlobalHead` × 3 | — | ~300K |
| `BatchNorm2d` × 3 | — | ~6K |
| `ScalarFusionHead` | — | 5 |
| Layer/fusion weights | — | ~8 |
| **Total trainable** | **~3.5K** | **~3.4M** |

Same training data, same training steps, 1000× more parameters → overfitting to the single training source is inevitable. The original's minimal parameter budget is a strong implicit regularizer.

### 3.1 H1 — Pixel-level Mask Loss (highest confidence, likely primary cause)

Original `InCTRL.forward` returns only two scalars: `final_score, img_ref_score` (`model.py:592`). **There is no pixel-level output. The original training therefore cannot apply dense mask supervision.**

Our `InCTRLPQA` exposes `pqa_local_logits` (pixel-level 2-class map) and applies (`inctrl_pqa_losses.py:46-55`):

```python
target_mask = F.interpolate(masks, size=logits.shape[-2:], mode="nearest").clamp(0.0, 1.0)
target_labels = target_mask.squeeze(1).long()
focal_loss = multiclass_focal_loss(logits, target_labels)
anomaly_dice = binary_dice_loss(probabilities[:, 1], target_mask.squeeze(1))
normal_dice = binary_dice_loss(probabilities[:, 0], 1.0 - target_mask.squeeze(1))
```

With `mask_loss_weight = 1.0` (the global maximum in our loss schedule), this is the strongest gradient signal in training. It drives conv filters to **precisely match MVTec anomaly masks** — category-specific defect morphologies that don't transfer to VisA.

> **This is a wholly new inductive bias we introduced that the original InCTRL does not have.** Removing it is the cheapest test of the regression hypothesis.

### 3.2 H2 — Decision Head: Fixed Averaging → Learnable Convex Combination (high)

Original (`model.py:583-588`):

```python
holistic_map = text_score + img_ref_score + patch_ref_map   # three paths added 1:1:1
hl_score = self.diff_head.forward(holistic_map)              # Linear(225, 1) fixed
final_score = (hl_score + fg_score) / 2                       # fixed 50:50 with parameter-free fg
```

- Three signals (text, image residual, patch residual) add with **fixed unit weights**
- `fg_score = Fp_map.max(dim=0).values` is a **parameter-free patch-max** worth 50% of the final score

Our `InCTRLPQA` (`inctrl_pqa_fused.py:976-978`):

```python
branch_logits = torch.stack([patch_logit, pqa_logit, image_logit, text_logit], dim=-1)
final_logit = self.decision_head(branch_logits)  # softmax-weighted, learnable
```

The original **cannot** "learn to silence the text branch on MVTec" — the weights are frozen at 1:1:1. Ours learns this bias, then misapplies it on VisA.

### 3.3 H3 — Layer Fusion: Uniform Mean → Learnable Softmax (high)

Original (`model.py:553`):

```python
Fp_map = torch.mean(Fp_map.squeeze(2), dim=0)  # 3 layers, plain mean
```

Ours (`inctrl_pqa_fused.py:919-924`):

```python
inctrl_patch_map = sum(w * r for w, r in zip(patch_layer_weights, pq_outputs["residual_maps"]))
pqa_patch_map    = sum(w * p for w, p in zip(pqa_layer_weights,   pq_outputs["pqa_patch_maps"]))
```

Both `patch_layer_weights` and `pqa_layer_weights` are learnable softmax. Training drifts them toward "the layer that fits MVTec best," losing the multi-layer ensemble robustness.

### 3.4 H4 — Added Conv Local Heads (high — enables H1)

Original has NO conv-based local heads. Patch signal is pure cosine distance (`model.py:546-548`):

```python
s = (0.5 * (1 - (tmp @ tmp_n.T))).min(dim=1).values
```

This is **parameter-free** → trivially cross-domain stable. Our 3 × `PQAConvLocalHead` (~1M conv params each) replaces this stable path with a learned filter bank, and the mask loss (H1) pushes those filters to specialize on MVTec shapes.

### 3.5 H5 — `fg_score` (parameter-free patch-max) Removed (medium)

Original: `fg_score = Fp_map.max(dim=0).values` is an independent 50% contributor to the final score, entirely parameter-free.

Ours: the residual map is merged into `fused_patch_map` via learnable `patch_map_fusion_logits`, then reduced to a single `patch_logit` that competes with other branches inside the learnable decision head. The parameter-free channel is gone.

### 3.6 H6 — BatchNorm Introduction (medium, downstream of H4)

Original has no `BatchNorm2d` in the residual/patch path — no conv means no BN. We added BN for the conv heads; its running statistics memorize MVTec distribution, compounding H4. Removing H4 removes H6 automatically.

---

## 3A. AdaptCLIP Contrast (Secondary Reference)

AdaptCLIP (`open_clip/adaptclip.py:715-800`) uses a structurally similar `PQAdapter`. Three system-level differences:

1. **Learnable textual adapter** (`TextualAdapter` with `n_ctx_pos`, `n_ctx_neg`) adapts text prompts across domains; our text branch is static
2. **Alignment score as direct, parameter-free output** — raw cosine residual goes directly to final output; ours mixes it with PQA via learnable `patch_map_fusion_logits`
3. **Multi-domain training** — AdaptCLIP paper uses multi-source training; our protocol is single-source

Note: comparing with AdaptCLIP is a secondary concern. The primary gap to close is vs **original InCTRL** which uses the same single-source cross-domain protocol and achieves 0.877 on VisA.

---

## 4. Phase 2 Options (Ranked as Regression Ablations)

Ordered by diagnostic signal per training run. Each option tests a specific hypothesis from §3.

### Option A — Zero out `mask_loss_weight` (tests H1 — no code change)

```bash
python train_local.py --train-datasets mvtec --train-shot 4 \
  --epochs 10 --steps-per-epoch 100 --num-workers 8 \
  --mask-loss-weight 0.0 --prior-loss-weight 0.3
```

- **Tests**: H1 (mask loss is the primary regression source)
- **Predicted VisA 4-shot**: 0.83+ (rise from 0.779)
- **Risk**: loses pixel-level localization on MVTec (irrelevant for cross-domain AUROC protocol)
- **Cost**: zero code change, one training run

### Option B — Replace layer fusion with uniform mean (tests H3 — small code change)

Force `_get_layer_weights` to return `torch.ones(N) / N` and disable gradient. ~5 lines.

- **Tests**: H3 (learnable layer weights drift toward MVTec-optimal layer)
- **Predicted VisA 4-shot uplift**: +0.01 ~ +0.03 on top of A
- **Cost**: trivial code change + test

### Option C — Replace `ScalarFusionHead` with fixed equal weighting (tests H2 — small code change)

Return `torch.mean(branch_logits, dim=-1)` instead of softmax-weighted sum. ~10 lines.

- **Tests**: H2 (decision head learns "trust PQA, ignore text" on MVTec)
- **Predicted VisA 4-shot uplift**: +0.01 ~ +0.03 on top of A
- **Cost**: small code change + test

### Option D — Freeze conv local heads + global heads (tests H4 — moderate)

Exclude `local_heads` and `global_heads` from optimizer via `get_trainable_parameters()`. Add `--freeze-pqa-heads` flag.

- **Tests**: H4 (3M conv params specialize on MVTec morphology)
- **Predicted VisA 4-shot uplift**: +0.03 ~ +0.08
- **Cost**: modify `get_trainable_parameters` + flag + test. Only run if A+B+C insufficient.

### Option E — Full revert to original InCTRL + re-apply Phase 1 surgical fixes (last resort)

If A–D cannot close the gap, the original `InCTRL` class (`model.py:443-592`) is the known-good baseline. Apply our decision-head / text-logit-scale / prior-loss improvements there instead.

- **Cost**: substantial rework of train_local / loss / model export
- **Only if** all above options fail to approach original InCTRL numbers

---

## 5. Decision Log

| When | Decision | Rationale |
| --- | --- | --- |
| 2026-04-21 rev 2 | **Run Option A (`--mask-loss-weight 0.0`) as top priority** | Zero code change; tests H1 (highest-confidence regression source); original InCTRL has no mask loss at all |
| If Option A lifts VisA ≥ 0.85 | Commit to "H1 is primary," stack B+C for remaining gap | High-confidence evidence |
| If Option A alone reaches ≥ 0.88 | Close Phase 2, original parity achieved | Exit condition |
| If Option A lifts VisA but doesn't close gap | Stack Option B + C (fixed fusion weights) | Gradually restore original structure |
| If Option A does NOT move VisA | Reconsider — move to Option D (freeze heads) then E (revert) | H1 falsified |

---

## 6. Open Questions

1. **How much of the VisA gap does H1 (mask loss) account for?** Option A answers this directly. If VisA rises from 0.779 to 0.85+, H1 explains most of it. If it only rises to 0.81, other hypotheses (H2–H4) share the blame.
2. **Does `patch_map_fusion_logits` actually drift toward PQA during training?** Check `aux["patch_map_fusion_weights"]` across a training run. If `[0.5, 0.5]`, H5 is wrong. If `[0.1, 0.9]`, confirmed.
3. **Does ELPV drop when mask loss is removed?** If yes, mask loss helps MVTec-like domains at VisA's expense — a trade-off. If no, mask loss was unambiguously harmful.
4. **Is there any cross-domain scenario where PQA dense supervision helps?** Possibly not under single-source training. Confirmed if Option A + D matches or exceeds original InCTRL.

---

## 7. References

- `open_clip/model.py:443-592` — original `InCTRL` class (the cross-domain baseline that achieves 0.877 on VisA)
- `open_clip/inctrl_pqa_fused.py:310+` — current `InCTRLPQA` class
- `open_clip/inctrl_pqa_losses.py:23-55` — `compute_pqa_mask_loss` (H1 regression source)
- `open_clip/adaptclip.py:715-800` — AdaptCLIP reference
- `reports/original_inctrl_baseline.md` — cross-domain baseline AUROC numbers
- `train_local.py:955` — `--mask-loss-weight` CLI flag (Option A)
- Step 1-3 summary JSONs archived in download history

---

## 8. Changelog

- **2026-04-21 rev 2** — Corrected baseline interpretation: all published numbers are **cross-domain** (MVTec→X), not in-domain. Reframed the entire analysis as a regression investigation. Added §2 (PQA as stateful module), §3 (6 architectural regression sources from code diff), reordered Phase 2 options with Option A (mask_loss=0) as top priority. Removed speculation about "inherent single-source gap" — original InCTRL proves the gap is closeable.
- **2026-04-21 rev 1** — Initial diagnosis (BatchNorm memorization, cosine collapse, conv head specialization). Partially correct mechanisms but wrong framing (assumed in-domain baseline).
