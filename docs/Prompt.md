# Project Prompt

## Mission
Improve an InCTRL-based industrial anomaly detection system for real production-like data.

The main optimization target is fewer false positives on real industrial images while preserving benchmark performance and localization quality.

## Non-goals
- Do not chase novelty without a measurable experiment.
- Do not make large architecture changes without a scoped ablation.
- Do not mutate raw datasets in place.

## Persistent Constraints
- Keep experiments reproducible.
- Keep compute reasonable.
- Prefer small, attributable changes.
- Save reports in `reports/`.
- Save experiment notes in `experiments/`.
- Save temporary analysis scripts in `tools/`.

## Default Team Pattern
For broad diagnosis or experiment design, act as lead and explicitly spawn bounded subagents:
- data
- model
- train
- eval

The lead should merge findings into a prioritized plan before code changes.
