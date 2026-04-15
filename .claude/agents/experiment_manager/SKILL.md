# Experiment Manager Skills

## 1. Configuration Management
When setting up an experiment:
1. Generate a comprehensive YAML schema covering Data, Model, Training, and Eval parameters.
2. Ensure every parameter has a default value.
3. Enforce seed fixing (`torch.manual_seed`, `np.random.seed`) in the config boilerplate.

## 2. Logging Integration
When adding tracking to a script:
1. Provide boilerplate for initializing wandb/TensorBoard.
2. Define exactly what metrics should be logged at which step (e.g., loss every 10 steps, AUROC at epoch end).
3. Include code to save the best model checkpoint based on a target metric.

## 3. Run Comparison
When analyzing past experiments:
1. Extract the delta (what changed) between two configurations.
2. Correlate the configuration change with the metric change.
3. Generate a summary conclusion (e.g., "Increasing batch size from 16 to 32 reduced PRO by 2%").
