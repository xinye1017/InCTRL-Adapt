# Post-Training Evaluation Report Scaffold

This repository now includes `tools/post_train_evaluation.py` for automatic checkpoint discovery, test evaluation, visualization, ablation comparison, and final model selection after the VA training matrix completes.

Run:

```powershell
python tools\post_train_evaluation.py --selection-priority balanced
```

`train_va.py` also invokes this script automatically after all requested training runs finish. Use `--skip-post-eval` when you want training only.

Main outputs:

- `reports/post_train_evaluation/experiment_results.csv`
- `reports/post_train_evaluation/experiment_ranking.csv`
- `reports/post_train_evaluation/summary.json`
- `reports/post_train_evaluation/top_failure_candidates.csv`
- `reports/figures/*.png`
- `reports/final_model_selection_report.md`

Pixel-level metrics and PRO are emitted when masks and anomaly maps are available; with the current image-level InCTRL forward path they are reported as `N/A`.

