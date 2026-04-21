# InCTRL PQA Fused Restructure Design

**Date:** 2026-04-19

**Goal**

Restructure `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_fused.py` so it keeps the InCTRL-style multi-evidence decision framework while using AdaptCLIP-style PQA as the single patch-analysis branch.

The design should:
- preserve the InCTRL shape of the model at a high level: image evidence, text evidence, patch evidence, final decision fusion,
- replace the current overlapping patch pathways with one clean patch branch,
- keep PQA responsible for local mask supervision and image-level patch evidence,
- remove duplicated residual computations and repeated fusion stages,
- allow interface cleanup instead of preserving current intermediate output keys.

**Non-Goals**

- Turning the model into a full AdaptCLIP clone.
- Rewriting the CLIP visual or text backbone.
- Preserving backward compatibility for all current intermediate outputs in `forward()`.
- Changing prompt cache and text prototype cache semantics unless required for consistency.

## Current State

The current fused implementation in `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_fused.py` combines:
- InCTRL-style image evidence through `image_head`,
- InCTRL-style text evidence through static normal/anomaly text prototypes,
- a PQA-inspired `PQAdapter` with local and global heads,
- a patch-map fusion stage,
- a `holistic_head`,
- a final `decision_head`.

This gets the broad idea right, but the structure is carrying unnecessary overlap.

### Main structural problems

1. **Residual evidence is computed twice**
   - `raw_residual_outputs` from `_compute_patch_residuals()` produces one patch-residual path.
   - `PQAdapter.forward()` recomputes a closely related residual path again.
   - This creates `raw_base_patch_map`, `base_patch_map`, and related duplicate diagnostics that describe nearly the same evidence.

2. **Patch evidence is fused too many times**
   - The model first fuses residual and PQA patch evidence into a patch map.
   - Then it mixes patch evidence with scalar evidence through `holistic_head`.
   - Then it mixes scalar evidence again through `decision_head`.
   - This makes the scoring stack harder to reason about and weakens the meaning of each intermediate output.

3. **Primary outputs and debug outputs are mixed together**
   - The current result dictionary exposes many overlapping tensors such as `raw_base_patch_map`, `base_patch_map`, `hybrid_patch_map`, `raw_max_patch_logit`, `max_base_patch_logit`, `max_hybrid_patch_logit`.
   - Many of these are useful for diagnosis but not part of the stable model contract.

4. **The loss surface is less direct than it should be**
   - `final_logit` and `pqa_logit` are supervised explicitly.
   - `pqa_local_logits` are supervised through mask loss.
   - But the fused patch map influences training only indirectly through later heads.

## Design Principle

Keep InCTRL's outer decision structure, but let PQA own the patch branch.

That means:
- **InCTRL stays responsible for multi-source decision fusion**.
- **PQA stays responsible for prompt-query patch alignment, local anomaly map prediction, and patch-derived global anomaly evidence**.
- The model should have only one patch branch visible at the top level.

## Proposed Architecture

### 1. Retain three top-level evidence branches

The model will keep three explicit evidence sources:

1. **Image branch**
   - query global feature vs prompt global prototype
   - produces `image_logit`

2. **Text branch**
   - query global feature vs normal/anomaly text prototypes
   - produces `text_logit`

3. **Patch branch**
   - prompt-query patch alignment plus PQA local/global heads
   - produces:
     - `inctrl_patch_map`
     - `pqa_patch_map`
     - `pqa_logit`
     - `pqa_local_logits`
   - patch evidence is fused once to produce:
     - `fused_patch_map`
     - `patch_logit`

Final decision uses only the scalar outputs of these branches.

### 2. Remove `holistic_head`

`holistic_head` will be removed from `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_fused.py`.

Reason:
- it acts like an extra scalar-fusion stage after patch evidence has already been fused,
- it reintroduces image/text/PQA information before the final decision layer,
- it makes the model harder to interpret because evidence is partially fused twice.

The final fusion path becomes:
- `patch_logit`
- `pqa_logit`
- `image_logit`
- `text_logit`
- `decision_head([patch_logit, pqa_logit, image_logit, text_logit]) -> final_logit`

### 3. Make `PQAdapter` the only patch branch owner

`PQAdapter` will be simplified so it returns a single coherent patch-branch payload.

It will own:
- prompt-query patch matching,
- aligned prompt feature gathering,
- residual patch evidence,
- PQA local anomaly map prediction,
- PQA global anomaly logit prediction.

It will no longer return multiple overlapping forms of the same evidence under different names.

### 4. Keep patch fusion explicit and shallow

Patch fusion remains important because the model needs:
- a classical InCTRL-style alignment signal,
- a learned PQA-style local anomaly signal.

The fused patch map is:
- `fused_patch_map = alpha * inctrl_patch_map + beta * pqa_patch_map`
- `alpha` and `beta` remain learned through a 2-way softmax parameter.

After that, patch evidence should be compressed once into `patch_logit`.

## Detailed Module Changes

### `PQAdapter`

`PQAdapter.forward()` should return a dictionary with these top-level keys:
- `inctrl_patch_maps`: list of per-layer patch alignment maps
- `pqa_patch_maps`: list of per-layer PQA patch anomaly maps
- `pqa_global_logits`: list of per-layer image-level PQA logits
- `pqa_local_logits`: list of per-layer local segmentation logits
- `aligned_indices`: list of prompt-match indices
- `aligned_prompt_features`: list of aligned prompt features
- `layer_weights`: normalized layer weights

The caller will aggregate the per-layer tensors into one top-level patch branch result.

The adapter should not also expose separate duplicate residual outputs that are recomputed elsewhere.

### `InCTRLPQA.__init__()`

Keep:
- `patch_projection`
- `prompt_query_adapter`
- `image_head`
- `decision_head`
- `patch_map_fusion_logits`

Remove:
- `holistic_head`

Change:
- `decision_head` input dimension from 5 to 4
- final decision vector becomes `[patch_logit, pqa_logit, image_logit, text_logit]`

### New helper for patch-to-logit reduction

Add a focused helper that converts `fused_patch_map` into `patch_logit`.

Recommended reduction:
- `patch_logit = 0.5 * max_logit + 0.5 * topk_mean_logit`

This keeps the InCTRL instinct of preserving strong anomaly peaks while reducing sensitivity to a single noisy patch.

Implementation detail:
- operate on patch scores or patch-map values directly,
- clamp before logit conversion if needed,
- keep the helper local to `InCTRLPQA` unless later reuse justifies extraction.

## Forward Pipeline

### New forward flow

1. Encode query image with the frozen visual tower.
2. Build query global feature and query patch levels.
3. Load prompt features from cache or encode prompt images.
4. Build image branch:
   - average prompt global features into a prompt global prototype,
   - compare against normalized query global feature,
   - produce `image_logit` and `image_score`.
5. Build text branch:
   - construct normal/anomaly text prototypes,
   - compare against normalized query global feature,
   - produce `text_logit` and `text_score`.
6. Build patch branch through `PQAdapter`:
   - compute aligned prompt features per patch,
   - compute per-layer InCTRL residual patch maps,
   - compute per-layer PQA local logits,
   - compute per-layer PQA patch maps,
   - compute per-layer PQA global logits.
7. Aggregate per-layer patch outputs with normalized layer weights.
8. Fuse patch maps:
   - `fused_patch_map = alpha * inctrl_patch_map + beta * pqa_patch_map`
9. Reduce `fused_patch_map` into `patch_logit`.
10. Build final decision input:
   - `[patch_logit, pqa_logit, image_logit, text_logit]`
11. Produce `final_logit` and `final_score` through `decision_head`.

### Removed forward behavior

The forward path will no longer produce or depend on:
- `raw_residual_outputs`
- `raw_base_patch_map`
- `base_patch_map`
- `hybrid_patch_map`
- `holistic_logit`
- `base_logit`
- `max_base_patch_logit`
- `max_hybrid_patch_logit`
- any duplicate scalar derived from slightly different names for the same patch evidence

## Output Interface

### Stable primary outputs

The main result dictionary should contain:
- `final_logit`
- `final_score`
- `patch_logit`
- `patch_score`
- `pqa_logit`
- `pqa_score`
- `image_logit`
- `image_score`
- `text_logit`
- `text_score`
- `fused_patch_map`
- `pqa_local_logits`

### Auxiliary outputs

When `return_aux=True`, `aux` should contain diagnostic data only:
- `inctrl_patch_map`
- `pqa_patch_map`
- `aligned_indices`
- `aligned_prompt_features`
- `patch_fusion_weights`
- `decision_input`
- `text_prototypes`
- optional per-layer patch maps and logits if they are useful for experiments

This separates the stable model contract from analysis-only tensors.

### Cache interfaces to preserve

Keep these interfaces intact unless implementation forces a clear improvement:
- `build_prompt_feature_cache()`
- `build_text_prototype_cache()`

These are external integration points for evaluation and should not be changed casually.

## Loss Design

Modify `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_losses.py` so the training objective is aligned with the simplified architecture.

### Required losses

1. **Final decision loss**
   - `final_loss = loss_fn(final_logit, labels)`

2. **Patch-branch image-level auxiliary loss**
   - `pqa_loss = loss_fn(pqa_logit, labels)`

3. **Patch local mask loss**
   - keep the current `compute_pqa_mask_loss()` structure,
   - continue supervising `pqa_local_logits`.

### Optional auxiliary loss

4. **Image branch loss**
   - `image_loss = loss_fn(image_logit, labels)`
   - keep it optional and default-disabled.

### Removed loss dependencies

The loss module should no longer expect:
- `base_logit`
- `holistic_logit`
- any old patch-map scalar whose only purpose was supporting the removed fusion path

### Recommended `compute_training_loss()` outputs

Return metrics with these names:
- `final_loss`
- `pqa_loss`
- `pqa_mask_loss`
- `image_loss`
- `total_loss`

If `patch_logit` is later given direct supervision, it can be added as an explicit auxiliary term, but it is not required in this design.

## File-Level Changes

### `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_fused.py`

Modify to:
- simplify `PQAdapter.forward()` outputs,
- remove duplicate residual computation pathways,
- remove `HolisticScoringHead` usage,
- shrink final decision input from 5 channels to 4,
- add a helper for `fused_patch_map -> patch_logit`,
- rewrite `forward()` around the simplified three-branch design,
- clean up the returned output dictionary.

### `/Users/xinye/Desktop/InCTRL/open_clip/inctrl_pqa_losses.py`

Modify to:
- keep `compute_pqa_mask_loss()` centered on `pqa_local_logits`,
- simplify `compute_training_loss()` to match the new forward outputs,
- remove dependence on deleted intermediate logits.

### `/Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_fused.py`

Modify to:
- assert the new primary outputs,
- remove assertions tied to deleted intermediate outputs,
- preserve gradient checks for final decision, PQA global head, and PQA local head,
- add a gradient check that `patch_logit` influences patch-fusion parameters.

## Testing Strategy

Required tests:
- `forward()` returns the new stable output keys,
- `decision_input` has shape `[B, 4]` when returned through `aux`,
- `final_logit.backward()` updates `decision_head`,
- `pqa_logit.backward()` updates PQA global head parameters,
- `compute_pqa_mask_loss()` updates PQA local head parameters,
- `patch_logit.backward()` updates `patch_map_fusion_logits`,
- prompt feature cache and text prototype cache still work.

Verification command:
- `python -m pytest /Users/xinye/Desktop/InCTRL/tests/test_inctrl_pqa_fused.py -q`

If loss tests are extended separately, also run the focused loss test file.

## Risks and Mitigations

### Risk: removing `holistic_head` lowers accuracy

Mitigation:
- keep patch fusion learnable,
- use a slightly richer `patch_logit` reduction than pure max pooling,
- compare before/after on the current evaluation slice before broader rollout.

### Risk: patch evidence becomes too compressed

Mitigation:
- retain both alignment-based and learned patch maps before scalar reduction,
- keep `fused_patch_map` in outputs for visualization and ablations.

### Risk: interface cleanup breaks downstream code

Mitigation:
- preserve cache builders,
- update tests first,
- update training loss together with forward interface,
- treat deleted outputs as an intentional contract change, not a silent regression.

### Risk: per-layer debug visibility is lost

Mitigation:
- keep per-layer values in `aux` instead of the main output contract.

## Acceptance Criteria

The change is complete when:
- the model still has explicit image, text, and patch evidence branches,
- patch evidence is owned by one coherent PQA-backed branch,
- duplicate residual computation paths are removed,
- `holistic_head` is removed,
- final decision uses only `[patch_logit, pqa_logit, image_logit, text_logit]`,
- the main output dictionary is reduced to stable training and inference values,
- loss computation matches the simplified interface,
- tests cover the new output contract and gradient flow,
- prompt and text prototype cache interfaces still work.
