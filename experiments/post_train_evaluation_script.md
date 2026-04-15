# Post-Training Evaluation Script

Date: 2026-04-15

## Hypothesis

The visual-adapter ablation should be selected with one reproducible post-training evaluation path that discovers checkpoints, evaluates the same test datasets, aggregates image-level and optional pixel-level metrics, visualizes comparisons, and writes a final model recommendation.

## Files Changed

- `tools/post_train_evaluation.py`

## Config Knobs

- `--checkpoint-roots`: directories scanned recursively for completed checkpoints.
- `--datasets`: test datasets to evaluate, default `aitex elpv visa`.
- `--eval-shots`: override few-shot prompt counts; by default each checkpoint uses its training shot.
- `--selection-priority`: `highest_auroc`, `lowest_fpr`, or `balanced`.
- `--threshold-policy`: `max_f1` or `fixed_0.5`.
- `--skip-figures`: run metric/report aggregation without plotting.
- `train_va.py --skip-post-eval`: skip the automatic post-training evaluation hook.
- `train_va.py --post-eval-selection-priority`: choose the default model selection policy used by the hook.

## Outputs

- `reports/post_train_evaluation/experiment_results.csv`
- `reports/post_train_evaluation/experiment_ranking.csv`
- `reports/post_train_evaluation/summary.json`
- `reports/post_train_evaluation/top_failure_candidates.csv`
- `reports/figures/*.png`
- `reports/final_model_selection_report.md`

## Validation

Minimal syntax validation:

```powershell
python -m py_compile tools\post_train_evaluation.py
```

Result: passed.

Full evaluation command after training:

```powershell
python tools\post_train_evaluation.py --selection-priority balanced
```

## Risks

- Current InCTRL forward returns image-level scores only, so pixel metrics and PRO are `N/A` unless a future dataloader/model exposes masks and anomaly maps.
- Top-k failure output is a score table; representative image panels require sample paths from the dataset batch.
