# Original InCTRL Baseline Results

Source PDF: `D:\Data\Downloads\InCTRL.pdf`

Date recorded: 2026-04-17

## Purpose

This file is the canonical baseline reference for future InCTRL modifications in this repository. New model variants should be compared against these original-paper baseline results before being considered an improvement.

## Baseline Table

| Setting | ELPV | AITEX | VisA | MVTec AD |
| --- | ---: | ---: | ---: | ---: |
| 0-shot | 0.733 +/- 0.000 | 0.733 +/- 0.000 | 0.781 +/- 0.000 | 0.912 +/- 0.000 |
| 2-shot | 0.839 +/- 0.003 | 0.761 +/- 0.029 | 0.858 +/- 0.022 | 0.940 +/- 0.015 |
| 4-shot | 0.846 +/- 0.011 | 0.790 +/- 0.018 | 0.877 +/- 0.019 | 0.945 +/- 0.018 |
| 8-shot | 0.872 +/- 0.013 | 0.806 +/- 0.036 | 0.887 +/- 0.021 | 0.953 +/- 0.013 |

## Comparison Rule

- Treat these numbers as the original InCTRL baseline.
- Compare any new architecture, training policy, scoring module, or adapter ablation against the matching dataset and shot setting.
- Report both absolute score and delta versus this baseline.
- For cross-shot evaluation, compare each evaluation shot against the corresponding row in this table.

