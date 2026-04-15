# ROUTER

## Purpose
This router defines the logic for assigning tasks to specific agents in the InCTRL industrial anomaly detection team.

## Rules

| Problem/Task Domain | Target Agent | Trigger Keywords |
|---------------------|--------------|------------------|
| **Task Breakdown & Routing** | `orchestrator` | coordinate, assign, plan, breakdown, pipeline |
| **Data Issues** | `data_engineer` | dataset, augmentation, MVTec, VisA, imbalance, preprocessing, synthetic anomaly |
| **Model Architecture** | `model_engineer` | architecture, layers, attention, prompt tuning, visual adapter, checkpoint |
| **Training Problems** | `training_engineer` | loss, epoch, learning rate, contrastive, reconstruction, batch, scheduler |
| **Evaluation / Metrics** | `evaluation_engineer` | AUROC, PRO, F1, false positive, evaluation, metrics, visualization |
| **New Ideas / Papers** | `research_analyst` | paper, SOTA, literature, idea, transferable, CVPR, ICCV |
| **Experiment Tracking** | `experiment_manager` | logs, tracking, config, wandb, tensorboard, reproducible |

## Workflow Example
1. User provides a high-level goal: "Improve VisA dataset performance"
2. `orchestrator` breaks this down:
   - Data generation -> `data_engineer`
   - Prompt modification -> `model_engineer`
   - Evaluation -> `evaluation_engineer`
3. Each agent executes their specific tasks according to their SKILL.md
4. `experiment_manager` tracks the configuration and results of the full pipeline.
