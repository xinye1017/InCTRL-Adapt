# Project Status

## Baseline
Unknown. Fill in the current best checkpoint, config, metrics, and command after the next verified run.

## What Has Been Tried
- Codex project-team scaffolding has been added:
  - `AGENTS.md`
  - `.codex/config.toml`
  - reusable Codex skills under `.codex/skills/`
  - durable project memory docs under `docs/`
- Agent onboarding经验文档已补充：`docs/agent_onboarding_inctrl_adaptclip_pqa_fused.md`

## Current Risks
- Benchmark performance may not predict real industrial false-positive behavior.
- Thresholding or preprocessing mismatches may look like model failure.
- Multiple simultaneous code changes can make ablations hard to attribute.

## Next Recommended Action
Run a lead-style diagnosis with data, model, train, and eval subagents to inspect:
- dataset pipeline and domain gap
- InCTRL model/scoring path
- training loss and batch strategy
- evaluation metrics and false-positive reporting

Save the resulting diagnosis in `reports/` and update this file with the selected next ablation.
