---
name: inctrl-ablation
description: Use this when the task involves modifying or comparing InCTRL architecture components, visual adapters, auxiliary heads, memory modules, prompt tuning, feature fusion, or scoring modules. Do not use for dataset-only tasks.
---

# Goal
Design and implement scoped ablations for InCTRL-based industrial anomaly detection, following the project's experiment conventions.

# Context
- Train source: MVTec AD (normal + anomalous).
- Test targets: VisA, AITEX, ELPV (cross-domain few-shot).
- Shot settings: 2, 4, 8 (controlled by `--shot`).
- Baseline reference: `reports/original_inctrl_baseline.md` — all results must be compared against the matching dataset + shot row in that file.
- Active checklist: `reports/va_ablation_checklist.md` — update this file when running VA-related ablations; create a new `reports/<topic>_ablation_checklist.md` for other ablation topics.
- Results directory: `results/` with naming convention `<experiment_name>_<shot>shot_<epochs>ep`.

# Procedure
1. Define the exact hypothesis: what component is being added/removed/changed, and what effect is expected.
2. Read the current baseline from `reports/original_inctrl_baseline.md` for the matching dataset + shot configuration.
3. Keep the change isolated — one variable per ablation. If multiple components must change together, document the coupling explicitly.
4. Add or reuse config knobs in the training command instead of hard-coding experiment behavior. Common knobs:
   - `FUSION.IMAGE_WEIGHT`, `FUSION.PATCH_WEIGHT`, `FUSION.PQA_WEIGHT`, `FUSION.TEXT_WEIGHT`, `FUSION.VISUAL_WEIGHT`
   - `LOSS.IMAGE_WEIGHT`, `LOSS.PQA_WEIGHT`, `LOSS.MASK_WEIGHT`, `LOSS.TEXT_WEIGHT`, `LOSS.VISUAL_WEIGHT`, `LOSS.VISUAL_MASK_WEIGHT`
5. Update code path, config usage, and experiment notes together.
6. Run a smoke test (1 epoch, small batch) to verify the code path does not crash before launching the full run.

# Run protocol
1. Launch with `python train_local.py --train_dataset mvtec --test_dataset <target> --shot <N> --max_epoch 15 --steps_per_epoch 100 --output_dir results/<name> <CONFIG_OVERRIDES>`.
2. Use multi-seed evaluation: run at least 3 seeds (default seeds: 0, 1, 2) per configuration.
3. Record mean and standard deviation across seeds for each metric.
4. For 2-shot: run all three target datasets (VisA, AITEX, ELPV).
5. If 2-shot shows promise (mean AUROC within 0.005 of or above No-VA baseline), expand to 4-shot and 8-shot.
6. If 2-shot shows clear regression (>0.01 below baseline on any dataset), stop and record as negative result.

# Metrics
Always report:
- **Image-level**: AUROC (primary), AUPR.
- **Pixel-level**: AUROC, PRO (when available).
- **Error rates**: FPR, FNR (when available).
- **Per-category breakdown** for VisA (12 categories: candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum).

Comparison columns:
- vs No-VA final (current best known config).
- vs InCTRL original baseline (from `reports/original_inctrl_baseline.md`).

# Decision criteria
| Condition | Action |
| --- | --- |
| AUROC >= No-VA baseline AND no dataset regresses >0.005 | Positive result. Expand to 4/8-shot and multi-seed. |
| AUROC within 0.005 of No-VA on average, minor trade-offs | Marginal. Document trade-off, decide based on paper narrative. |
| AUROC < No-VA by >0.005 on any primary dataset | Negative result. Record conclusion, do not expand unless resolving a specific paper reviewer concern. |
| Per-category instability (std > 0.02 across seeds) | Investigate few-shot sampling sensitivity before drawing conclusions. |

# Required output
When completing an ablation, produce or update:
1. **Hypothesis**: one sentence, testable.
2. **Config**: the exact `train_local.py` command with all overrides.
3. **Results table**: matching the format in `va_ablation_checklist.md` — columns for AUROC, AUPR, FPR, delta vs baselines, status.
4. **Per-category results** (VisA): table with AUROC and AUPR per category.
5. **Conclusion**: one of {positive, marginal, negative} with justification.
6. **Next step**: expand, stop, or investigate further.

# Guardrails
- Avoid mixing multiple architecture ideas into one ablation.
- Keep defaults backward-compatible unless the user explicitly asks to replace the baseline.
- When touching the visual adapter or open_clip modules, verify imports and at least one small forward-path check if feasible.
- Never report a single-seed result as final — always note if multi-seed has not been completed.
- Always compare against `reports/original_inctrl_baseline.md`, not against memory or prior conversation.
