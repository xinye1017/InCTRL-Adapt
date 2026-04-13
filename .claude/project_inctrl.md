---
name: project_inctrl
description: InCTRL GAD model project context, AdaptCLIP alignment plan
type: project
---

# InCTRL Project Context

**What it is:** InCTRL is a Generalist Anomaly Detection (GAD) model from CVPR 2024. It learns in-context residual relationships between query images and few-shot normal sample prompts to detect anomalies across diverse domains.

**Key directories:**
- `open_clip/model.py` — core InCTRL model with Adapter, text_adapter, holistic fusion
- `engine_IC.py` — training engine with joint TA/VA optimization
- `datasets/preprocess/` — dataset converters and JSON generators for AD datasets
- `test_holistic_map_shapes.py` — shape contract tests

**Current active work (2026-04-08 plan):** Align InCTRL with AdaptCLIP paper design:
1. Replace post-text MLP (`text_adapter`) with a `PromptLearner` that learns prompts in token-embedding space
2. Route `visual_adapter` through global tokens AND all three patch-token levels (currently only global)
3. Remove `use_textual_adapter` / `use_visual_adapter` feature toggles
4. Implement epoch-level alternating optimization: even epochs = TA phase, odd epochs = VA phase

**Why it matters:** The current InCTRL implementation diverges from AdaptCLIP in three ways that limit textual adaptation quality, patch-level visual adaptation, and training dynamics.

**How to apply:** When modifying `open_clip/model.py`, expect `PromptLearner`, `_build_prompted_text_features`, `_adapt_patch_levels`, `_compute_patch_residual_map`, and `set_alternating_phase`. When modifying `engine_IC.py`, expect `get_training_phase` and `build_phase_optimizer` to be introduced.

**Dataset context:** Works on MVTec AD format. Uses 9 AD datasets (ELPV, SDD, AITEX, VisA, MVTec AD, BrainMRI, HeadCT, MNIST, CIFAR-10). Single RTX 3090 training.
