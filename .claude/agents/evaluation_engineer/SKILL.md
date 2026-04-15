# Evaluation Engineer Skills

## 1. Metric Calculation
When evaluating predictions:
1. Ensure metrics handle severe class imbalance (prefer AP over AUROC for highly skewed test sets).
2. Implement PRO metric carefully, ensuring connected components are correctly identified in the ground truth.
3. Output vectorized PyTorch/NumPy implementations of metrics to avoid CPU bottlenecks.

## 2. Thresholding Strategy
When converting continuous anomaly scores to binary predictions:
1. Implement F1-Max thresholding (finding the threshold that maximizes the F1 score).
2. Provide alternative thresholding based on acceptable False Positive Rates (e.g., threshold at 1% FPR).

## 3. Failure Case Analysis
When analyzing model errors:
1. Sort test samples by loss/score discrepancy.
2. Group False Positives by visual characteristics (e.g., lighting artifacts, misalignment).
3. Generate a Markdown report detailing *why* the model failed, passing these insights back to the Orchestrator for data/model adjustments.
