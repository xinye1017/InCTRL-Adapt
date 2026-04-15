# AGENTS.md

## Project mission
This repository trains and evaluates an industrial anomaly detection system based on InCTRL.

Primary goals:
- reduce false positives on real industrial data
- preserve or improve anomaly localization quality
- keep experiments reproducible and easy to review
- prefer practical improvements over novelty for novelty's sake

## Working style
- Think like a small research-engineering team, not a one-shot assistant.
- Read relevant docs and experiment notes before non-trivial changes.
- Before major edits, make a short plan.
- For broad tasks, explicitly split work across subagents with bounded scopes.
- Keep diffs scoped and easy to review.
- Do not change raw data files in place.

## Roles to emulate when needed
- lead: decomposes tasks, assigns subagents, consolidates findings
- data: dataset checks, anomaly scarcity, augmentation, preprocessing, domain gap
- model: InCTRL architecture, visual adapter, prompt modules, memory or scoring changes
- train: losses, schedules, batch strategy, stability, checkpoint behavior
- eval: metrics, error analysis, false positives, reports, artifact review

## Repo rules
- Never touch raw data files in place.
- Put generated reports into `reports/`.
- Put experiment notes into `experiments/`.
- Put one-off analysis scripts into `tools/`.
- For every non-trivial experiment change, update experiment notes.
- Avoid committing large generated artifacts unless explicitly requested.

## Validation rules
- Run the smallest relevant validation first.
- When changing training logic, validate config parsing and one short smoke run before larger runs.
- When changing model code, run the smallest import/forward-path check that exercises the changed code.
- When changing evaluation logic, produce at least one artifacted report.
- Record validation commands and important outputs in the related experiment note.

## Prompting preference
When the request is broad, use this structure:
1. root cause
2. proposed changes
3. files to modify
4. validation plan
5. expected risks

## Team prompt template
Use this when a task benefits from parallel diagnosis:

```text
Read AGENTS.md first.

Act as the lead for this industrial anomaly detection repository.
Spawn subagents in parallel for:
- data
- model
- train
- eval

Problem:
Our InCTRL-based model performs well on benchmark data but produces too many false positives on real industrial data.

Use skills if relevant:
- anomaly-root-cause
- false-positive-report

Tasks:
1. inspect likely root causes
2. identify the first files/configs to inspect or modify
3. propose a minimal ablation plan
4. write a concise report structure for reports/

Constraints:
- prefer practical improvements
- keep compute reasonable
- keep proposed diffs scoped

Return:
- consolidated diagnosis
- prioritized next actions
- exact files to change first
- validation plan
```
