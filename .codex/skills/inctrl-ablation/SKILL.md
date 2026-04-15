---
name: inctrl-ablation
description: Use this when the task involves modifying or comparing InCTRL architecture components, visual adapters, auxiliary heads, memory modules, prompt tuning, feature fusion, or scoring modules. Do not use for dataset-only tasks.
---

# Goal
Design and implement scoped ablations for InCTRL-based industrial anomaly detection.

# Procedure
1. Define the exact hypothesis.
2. Compare the current baseline with the proposed architectural change.
3. Keep the change isolated so the ablation is attributable.
4. Add or reuse config knobs instead of hard-coding experiment behavior.
5. Update code path, config usage, and experiment notes together.
6. Add a minimal validation or smoke test if code changes.

# Required output
- hypothesis
- files changed
- config knobs
- expected upside
- risks and confounders
- validation command

# Guardrails
- Avoid mixing multiple architecture ideas into one ablation.
- Keep defaults backward-compatible unless the user explicitly asks to replace the baseline.
- When touching the visual adapter or open_clip modules, verify imports and at least one small forward-path check if feasible.
