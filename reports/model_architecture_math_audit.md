# InCTRL / AdaptCLIP Model Architecture Math Audit

Date: 2026-04-18

## Scope

This audit checks the current `open_clip/inctrl_three_adapters.py` and `train_local.py` design against:

- InCTRL: in-context residual learning with patch residuals, image residual learning, text prior, holistic scoring, and `LIRL + Lh`.
- AdaptCLIP: visual adapter, textual adapter, prompt-query adapter, alternating learning, and adapter-output averaging.

The goal is not novelty. The goal is to ensure any model change has mathematical support and does not silently break the original InCTRL scoring assumptions.

## Root Cause

The current model is not a faithful implementation of either paper. It mixes InCTRL-style residual scoring with AdaptCLIP-style adapters, but several mathematically important parts are changed at the same time:

- The patch residual map used in final scoring is no longer the pure InCTRL residual map `Mx`; it is a hybrid of nearest-neighbor residual and an uncalibrated learned context score.
- The final score no longer follows the original implementation's stable fusion rule `(holistic_score + max_patch_score) / 2`; it uses `clamp(holistic_score + alpha * max_patch_score)`.
- The image residual branch is computed, but the training loss no longer includes the separate `LIRL` term required by InCTRL.
- The textual adapter is descriptor-conditioned CoOp-style prompting, while AdaptCLIP describes two learnable binary prompts without hand-written templates.
- The prompt-query adapter does not implement AdaptCLIP's full segmentation/global-head objective. It uses a learned patch score, but training only provides image labels through the final score.
- Alternating learning is implemented as parameter freezing by epoch, but the forward path always uses both visual and textual branches. This is weaker than AdaptCLIP's formula-level separation of fixed text vs fixed visual representations during adapter learning.

Given the observed VisA mean AUROC around 0.66 after MVTec training, this degradation is consistent with the architecture: the model has introduced trainable heads that can dominate the original residual evidence without preserving the original loss decomposition.

## Paper Formula Check

### InCTRL Patch-Level Residual

Paper:

```text
M_lx(i,j) = 1 - <T_lx(i,j), h(T_lx(i,j)|P')>
Mx = (1 / n) * sum_l M_lx
sp(x) = max(Mx)
```

Original code:

```text
sim = 0.5 * (1 - cosine(query_patch, prompt_patch))
Fp_map = mean_l min_prompt_patch(sim)
max_diff_score = max(Fp_map)
```

Current three-adapter code:

```text
residual = 0.5 * (1 - max_cosine)
context_score = patch_head(query + beta * abs(query - aligned_prompt))
patch_evidence = gamma_r * residual + gamma_c * context_score
hybrid_patch_map = weighted_sum_l(patch_evidence_l)
max_patch_score = max(hybrid_patch_map)
```

Assessment: partially compliant, but high-risk. The nearest-neighbor residual is still present. However, the final `sp(x)` is no longer `max(Mx)`; it is `max(hybrid_patch_map)`. That changes the InCTRL meaning of the max-patch compensation term.

### InCTRL Image-Level Residual

Paper:

```text
Ip = mean_k psi(fv(x'_k))
Ix = psi(fv(x))
Fx = Ix - Ip
si(x) = eta(Fx)
LIRL = Lb(si(x), y)
```

Original code uses `token_n - token_ad`, not `token_ad - token_n`, but this is the implementation that produced the original checkpoint behavior.

Current three-adapter code:

```text
prompt_global_proto = mean(adapted_prompt_global)
image_residual = prompt_global_proto - adapted_query_global
image_score = image_head(image_residual)
```

Assessment: structurally close to the original implementation, but incomplete in training. `image_score` is not supervised with a separate `LIRL` term. In the original training loop, the model returned both final score and image residual score, and optimized both with focal loss.

### InCTRL Text Prior

Paper:

```text
Fn = mean text_encoder(normal_prompts)
Fa = mean text_encoder(anomaly_prompts)
sa(x) = exp(Fa^T fv(x)) / (exp(Fn^T fv(x)) + exp(Fa^T fv(x)))
```

Original code:

```text
logits = 100 * [cos(query, normal_proto), cos(query, anomaly_proto)]
text_score = softmax(logits)[anomaly]
```

Current three-adapter code:

```text
normal_proto, anomaly_proto = textual_adapter(...)
text_score = sigmoid(scale * (cos(query, anomaly_proto) - cos(query, normal_proto)))
```

Assessment: the sigmoid of logit difference is mathematically equivalent to two-class softmax if prototypes are fixed and the scale is shared. But the current learned descriptor-conditioned textual adapter changes the paper assumption. It is no longer the InCTRL fixed WinCLIP-style text prior, and it is also not AdaptCLIP's pure binary prompt learning.

### InCTRL Fusion and Final Score

Paper:

```text
M+x = Mx + si(x) + sa(x)
s(x) = phi(M+x) + alpha * sp(x)
LInCTRL = LIRL + Lh
```

Original code:

```text
holistic_map = text_score + img_ref_score + Fp_map
hl_score = diff_head(holistic_map)
final_score = (hl_score + max_diff_score) / 2
loss = focal(final_score, y) + focal(img_ref_score, y)
```

Current three-adapter code:

```text
holistic_input = hybrid_patch_map + lambda_g * image_score + lambda_t * text_score
holistic_score = holistic_head(holistic_input)
final_score = clamp(holistic_score + alpha * max_patch_score, 0, 1)
loss = focal(final_score, y)
```

Assessment: non-compliant with the original implementation and incomplete relative to the paper. The current loss omits `LIRL`, and the final fusion changed from calibrated averaging to additive clamp. This can easily saturate scores or make random heads dominate residual evidence.

### AdaptCLIP Visual Adapter

Paper:

```text
F'q_i = Fq_i + MLP(Fq_i; theta_l_v)
f'q = fq + MLP(fq; theta_g_v)
```

Current code:

```text
VisualAdapter.adapt_global(...)
VisualAdapter.adapt_local(...)
```

Assessment: mostly compliant at the module level. Residual MLP and zero initialization are reasonable. The issue is not the visual adapter itself; the issue is how its outputs are fused and supervised.

### AdaptCLIP Textual Adapter

Paper:

```text
theta_a, theta_n in R^(r x d)
w'a = T(theta_a)
w'n = T(theta_n)
```

Current code:

```text
prompt_text = "X X ... X {descriptor}"
learned_ctx replaces the X tokens
prototype_adapter(mean(encoded_descriptor_prompts))
```

Assessment: not faithful to AdaptCLIP. This is closer to descriptor-conditioned CoOp/CoCoOp-style prompting. It may be useful as an ablation, but it should not be described as directly implementing AdaptCLIP's textual adapter.

### AdaptCLIP Prompt-Query Adapter

Paper:

```text
Fp'_i = nearest_neighbor(Fq_i, {Fp_j})
Fbar_i = Fq_i + |Fq_i - Fp'_i|
Yp = G(Fbar)
yp = MLP((AvgPool(Fbar) + MaxPool(Fbar)) / 2)
```

Current code:

```text
aligned_prompt = nearest by cosine
context_feat = query + beta * abs(query - aligned_prompt)
context_score = patch_head(context_feat)
patch_evidence = gamma_r * residual + gamma_c * context_score
```

Assessment: partially compliant in alignment and joint context-residual feature construction. Not compliant in the objective and output structure. The paper trains local segmentation with focal/dice losses and a global PQA head from pooled joint features. Current code trains only the final image-level anomaly score.

### Alternating Learning

Paper:

```text
When training VA: fix binary text embeddings.
When training TA: fix visual tokens.
Joint visual-text learning overfits and hurts cross-domain generalization.
```

Current code:

```text
epoch % 2 == 0 -> visual params require_grad=True
epoch % 2 == 1 -> text params require_grad=True
forward always computes visual adapter, textual adapter, PQA, image head, holistic head
```

Assessment: directionally aligned but not formula-faithful. Gradients are alternated, but branch usage is not. Also, PQA/image/holistic heads are bundled with the visual phase, so the visual phase updates more than just the AdaptCLIP visual adapter.

## Recommended Direction

The safest next step is not to continue training the current three-adapter model. We should first restore an InCTRL-faithful scoring backbone, then add one isolated adapter ablation at a time.

Recommended default architecture:

```text
base residual map:
  Mx = mean_l min_prompt_patch(0.5 * (1 - cosine(query_patch_l, prompt_patch_l)))

image residual:
  si = image_head(prompt_global_proto - query_global)

text prior:
  sa = fixed or cached WinCLIP/InCTRL text prior

fusion:
  Mplus = Mx + si + sa
  final = (phi(Mplus) + max(Mx)) / 2

loss:
  L = focal(final, y) + focal(si, y)
```

Then add adapter ablations under explicit config switches:

- Visual adapter only: residual MLP on query and prompt visual features, zero-init, original InCTRL fusion unchanged.
- Textual adapter only: either keep fixed InCTRL text prior, or add a separate pure AdaptCLIP-style two-prompt branch as an ablation.
- PQA only: if no pixel masks are used, do not replace `Mx` with a learned patch head by default. Keep `Mx` as the final `sp(x)` source, and log PQA evidence separately until validated.

## Files To Modify First

- `open_clip/inctrl_three_adapters.py`: separate `base_residual_map` from `hybrid_patch_map`; make final scoring optionally InCTRL-faithful; expose branch outputs clearly.
- `train_local.py`: restore `LIRL + Lh` by adding focal loss on `image_score`; keep phase alternation but avoid training unrelated heads in the wrong phase.
- `open_clip/config/defaults.py`: add explicit ablation knobs, such as `FUSION_MODE`, `USE_FIXED_TEXT_PRIOR`, `USE_PQA_IN_FINAL_MAP`, and `USE_IMAGE_RESIDUAL_LOSS`.
- `experiments/`: record the exact ablation protocol before launching cloud runs.
- `tools/`: add a small forward-score audit script comparing original InCTRL score branches vs adapter score branches on the same mini-batch.

## Validation Plan

1. Import/compile check:

```text
python -m py_compile train_local.py open_clip/inctrl_three_adapters.py
```

2. Forward-path check on one mini-batch:

```text
verify output keys:
  final_score
  image_score
  text_score
  base_patch_map
  hybrid_patch_map
  max_base_patch_score
```

3. Zero-init sanity check:

```text
With adapters enabled but zero-init and PQA excluded from final map,
adapter final score should be close to the original InCTRL-style score.
```

4. Loss check:

```text
Confirm train loss includes:
  focal(final_score, label)
  focal(image_score, label)
```

5. Minimal evaluation:

```text
MVTec-trained checkpoint -> VisA candle 4-shot smoke test.
Do not proceed to full cloud training unless smoke AUROC is at least close to original InCTRL behavior.
```

6. Full cross-shot evaluation:

```text
Train one 4-shot model per source dataset.
Evaluate 2-shot, 4-shot, and 8-shot target prompts.
Compare against original InCTRL baseline report.
```

## Expected Risks

- Full AdaptCLIP reproduction needs pixel-level supervision for the PQA segmentation head. Without masks, training the PQA local head only through image-level final loss is weak and may overfit.
- Descriptor-capped textual prompts reduce memory but change the text prior relative to the original InCTRL baseline.
- If the final map uses learned PQA evidence instead of pure residual `Mx`, the max-patch compensation term loses its original interpretation.
- Training VA, PQA, image head, and holistic head together in the visual phase makes ablation attribution difficult.
- The current low VisA AUROC is likely architecture-induced, so longer training on cloud is unlikely to rescue it without formula-level corrections.

## Decision

Do not treat the current three-adapter model as a valid improvement over InCTRL. Treat it as a failed exploratory ablation.

The next implementation should use InCTRL-faithful fusion and loss as the default, then introduce adapters behind isolated config switches. This gives every code path a clear mathematical role and gives every experiment a fair comparison against the original baseline.

## 2026-04-18 Phase 2 Implementation Note

The follow-up hybrid implementation now targets the intended "InCTRL backbone + AdaptCLIP adapters" design:

- patch residual uses `1 - max_cosine`
- image residual uses `query - prompt_proto`
- VA adapts query and prompt visual features before residual comparison
- TA defaults to static InCTRL text prototypes plus a zero-init AdaptCLIP-style binary prompt residual
- PQA always participates when enabled and contributes both local patch evidence and a global comparison logit
- final training branches are logits-first, while evaluation keeps sigmoid scores
- additive InCTRL-style base fusion is restored as the default score backbone

See `experiments/inctrl_adaptclip_hybrid_phase2.md` for validation commands and smoke-run output.
