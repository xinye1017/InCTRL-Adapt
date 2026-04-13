# InCTRL AdaptCLIP Alignment Design

**Date:** 2026-04-08

**Goal**

Align InCTRL with the AdaptCLIP paper/README design by:
- replacing the current textual post-MLP with a prompt learner,
- routing the visual adapter through both global and multi-layer patch branches,
- removing independent TA/VA feature switches,
- training TA and VA with true alternating optimization.

**Non-Goals**

- Replacing InCTRL's holistic fusion framework with AdaptCLIP's full scoring stack.
- Rewriting the CLIP backbone or changing dataset/task definitions.
- Preserving backward compatibility with the current adapter toggle interface.

## Current State

The current InCTRL implementation in `/Users/xinye/Desktop/InCTRL/open_clip/model.py` uses:
- a residual `Adapter` on global image tokens only,
- a residual `text_adapter` on already-encoded static text features,
- fixed simultaneous use of textual and visual adapter paths in forward,
- joint optimization of all trainable modules in `/Users/xinye/Desktop/InCTRL/engine_IC.py`.

This diverges from the AdaptCLIP paper design in three important ways:
- textual adaptation happens after text encoding instead of in prompt space,
- visual adaptation does not enter the patch-local branch,
- textual and visual adaptation are not trained alternately.

## Proposed Architecture

### 1. Textual Adapter becomes Prompt Learner

Replace the current post-encoding text adapter with a learnable prompt module.

The new textual path will:
- keep the existing anomaly text/template semantics already used by InCTRL,
- build tokenized prompt shells with prefix and suffix buffers,
- introduce learnable context parameters for positive and negative prompts,
- compose prompt embeddings as `prefix + learned_ctx + suffix`,
- encode the composed prompt embeddings with the CLIP text transformer,
- produce normal/anomaly text features directly from prompted text encoding.

The prompt learner will expose:
- prompted feature generation for the current batch object types,
- static template feature preparation for reusable normal/anomaly prompt shells,
- a forward path that mirrors AdaptCLIP's prompt-learning behavior while fitting InCTRL's object-type driven text flow.

The old `text_adapter` MLP path will be removed.

### 2. Visual Adapter enters both global and patch branches

Keep the lightweight residual bottleneck adapter style, but expand its use.

The new visual path will:
- apply the visual adapter to query global tokens,
- apply the visual adapter to reference global tokens,
- apply the visual adapter to all three patch-token levels used by InCTRL,
- compute global residuals from adapted global tokens,
- compute patch residual maps from adapted patch tokens at all three levels,
- aggregate multi-level patch residuals after adaptation.

This preserves InCTRL's multi-layer residual-learning structure while ensuring that visual adaptation modifies the local branch instead of only the global branch.

### 3. Holistic fusion remains the task head

InCTRL's final anomaly score remains based on:
- text score,
- image-reference score,
- patch-reference map,
- holistic head fusion.

The change is only in the feature sources:
- text score will come from prompted text features,
- image-reference score will use VA-adapted global features,
- patch-reference map will use VA-adapted patch features from all three layers.

This keeps the change set focused and avoids turning InCTRL into a different model family.

## Forward Pipeline

### Training and inference pipeline

1. Encode query image and reference normal images with the frozen CLIP visual tower.
2. Extract:
- query global token,
- reference global token(s),
- three levels of query patch tokens,
- three levels of reference patch tokens.
3. Run the visual adapter on:
- query global token,
- reference global token(s),
- each of the three query patch-token levels,
- each of the three reference patch-token levels.
4. Build prompted normal/anomaly text features with the prompt learner and text transformer.
5. Compute text similarity from adapted global image features and prompted text features.
6. Compute image-level residual score from adapted query/reference global features.
7. Compute multi-level patch residual maps from adapted query/reference patch features.
8. Fuse the three score sources through the existing holistic scoring path.

## Alternating Optimization

### Phase schedule

Use epoch-level alternating optimization:
- even-numbered epochs: `ta` phase,
- odd-numbered epochs: `va` phase.

### Trainable modules by phase

`ta` phase:
- trainable: prompt learner only,
- frozen: visual adapter, `diff_head`, `diff_head_ref`, CLIP visual tower, CLIP text tower.

`va` phase:
- trainable: visual adapter, `diff_head`, `diff_head_ref`,
- frozen: prompt learner, CLIP visual tower, CLIP text tower.

### Forward behavior during alternating

Forward always runs the full graph so the score definition stays identical across phases.
Gradient flow changes by phase through `requires_grad` control:
- in `ta` phase, only prompted text parameters update,
- in `va` phase, only visual adaptation and scoring heads update.

### Removed interface

The following behavior will be removed:
- `use_textual_adapter`,
- `use_visual_adapter`,
- any code path that skips TA or VA entirely.

The model always instantiates both components. Training phase, not feature flags, decides what updates.

## File-Level Changes

### `/Users/xinye/Desktop/InCTRL/open_clip/model.py`

Modify to:
- replace `text_adapter` with a prompt learner module,
- add prompted text encoding helpers,
- remove textual/visual adapter toggles,
- route visual adapter through global and all three patch levels,
- update text feature bank construction,
- update forward to use prompted text features and adapted patch features,
- add helper(s) for phase-based parameter freezing if kept in model.

### `/Users/xinye/Desktop/InCTRL/engine_IC.py`

Modify to:
- introduce alternating phase selection by epoch,
- apply phase-specific freezing/unfreezing before training each epoch,
- rebuild or update optimizer from currently trainable parameters,
- log current phase and trainable parameter names for verification.

### `/Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py`

Extend tests to cover:
- prompted text path is used instead of post-text MLP,
- visual adapter affects patch branches,
- alternating phase correctly freezes and unfreezes parameter groups,
- text feature bank shape and holistic map shape remain stable.

## Testing Strategy

Use test-first updates for each behavior change.

Required tests:
- prompt learner based text feature bank returns `[B, 2, D]` and uses prompted encoding,
- adapted patch path runs through all three patch levels,
- alternating `ta` phase enables only prompt learner parameters,
- alternating `va` phase enables only visual adapter and score heads,
- existing holistic fusion shape contract remains valid.

Verification commands:
- `python -m pytest /Users/xinye/Desktop/InCTRL/test_holistic_map_shapes.py -q`
- if needed, a focused smoke test for trainable parameter logging in `engine_IC.py`.

## Risks and Mitigations

### Risk: prompt learner shape mismatches

Mitigation:
- keep prompt learner tensor layout close to AdaptCLIP's prompt assembly,
- add focused unit tests for prompt assembly and text feature bank output.

### Risk: patch adapter on three levels changes residual tensor shapes

Mitigation:
- adapt features level-by-level without changing existing score aggregation shape,
- preserve `[B, 225]` patch map contract before holistic fusion.

### Risk: optimizer continues to hold stale parameter groups across phases

Mitigation:
- rebuild optimizer per epoch from currently trainable parameters,
- test trainable parameter names explicitly.

## Acceptance Criteria

The change is complete when:
- textual adaptation is implemented as prompt learning, not post-text MLP,
- visual adaptation is applied to global features and all three patch levels,
- TA/VA independent switches are gone,
- training alternates between TA and VA phases by epoch,
- `ta` phase updates only prompt learner parameters,
- `va` phase updates only visual adapter and scoring heads,
- unit tests covering the new feature flow pass.
