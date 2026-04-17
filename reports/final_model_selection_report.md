# Final Model Selection Report

Generated: 2026-04-15T14:11:52
Selection priority: `balanced`

## Metric Ranking

| Rank | Experiment | Mode | Train Shot | AUROC | AUPR | F1 | FPR | FNR | Balanced |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | InCTRL_trained_on_MVTec_VA_ablation/global_only/2-shot | global_only | 2 | 0.7655 | 0.7361 | 0.6971 | 0.4781 | 0.1760 | 0.7260 |
| 2 | InCTRL_trained_on_MVTec_VA_ablation/global_only/4-shot | global_only | 4 | 0.7593 | 0.7335 | 0.6940 | 0.5148 | 0.1808 | 0.7184 |
| 3 | InCTRL_trained_on_MVTec_VA_ablation/global_only/8-shot | global_only | 8 | 0.7607 | 0.7285 | 0.6944 | 0.5360 | 0.1757 | 0.7161 |
| 4 | InCTRL_trained_on_MVTec_VA_ablation/global_local/2-shot | global_local | 2 | 0.7212 | 0.7418 | 0.6743 | 0.6769 | 0.1702 | 0.6880 |
| 5 | InCTRL_trained_on_MVTec_VA_ablation/global_local/8-shot | global_local | 8 | 0.7131 | 0.7307 | 0.6692 | 0.6221 | 0.2192 | 0.6820 |
| 6 | InCTRL_trained_on_MVTec_VA_ablation/local_only/2-shot | local_only | 2 | 0.6936 | 0.7233 | 0.6759 | 0.6533 | 0.1975 | 0.6737 |
| 7 | InCTRL_trained_on_MVTec_VA_ablation/local_only/8-shot | local_only | 8 | 0.6953 | 0.7194 | 0.6696 | 0.6512 | 0.2007 | 0.6719 |
| 8 | InCTRL_trained_on_MVTec_VA_ablation/global_local/4-shot | global_local | 4 | 0.6942 | 0.7119 | 0.6612 | 0.6543 | 0.2069 | 0.6671 |
| 9 | InCTRL_trained_on_MVTec_VA_ablation/local_only/4-shot | local_only | 4 | 0.6850 | 0.7138 | 0.6655 | 0.6741 | 0.1914 | 0.6647 |

## Ablation Conclusions

| Mode | N | Mean AUROC | Mean AUPR | Mean F1 | Mean FPR | Mean FNR |
| --- | --- | --- | --- | --- | --- | --- |
| global_local | 3 | 0.7095 | 0.7281 | 0.6682 | 0.6511 | 0.1987 |
| global_only | 3 | 0.7618 | 0.7327 | 0.6952 | 0.5096 | 0.1775 |
| local_only | 3 | 0.6913 | 0.7188 | 0.6703 | 0.6595 | 0.1965 |

- Best adapter mode by mean AUROC: `global_only`.
- Visual adapter improves over baseline: `None`.
- Complexity judgment: `prefer_simpler_global_only_until_more_evidence`.

## Visualization Summary

- `/root/InCTRL/reports/figures/ablation_auroc_bar.png`
- `/root/InCTRL/reports/figures/metric_f1_bar.png`
- `/root/InCTRL/reports/figures/false_positive_rate_bar.png`
- `/root/InCTRL/reports/figures/false_negative_rate_bar.png`
- `/root/InCTRL/reports/figures/metric_comparison_line.png`
- `/root/InCTRL/reports/figures/confusion_summary_stacked.png`

## Recommended Final Training Strategy

Use `global_only` with train shot `2`.

## Recommended Production Model Checkpoint

`/root/InCTRL/checkpoints/InCTRL_trained_on_MVTec_VA_ablation/global_only/2/checkpoint`

## Notes

- Image-level metrics are computed from model anomaly scores.
- Pixel-level metrics and PRO are `N/A` unless the dataloader/model exposes masks and anomaly maps.
- Top-k failure candidates are saved as score rows; raw image visualization requires sample paths from the dataset.