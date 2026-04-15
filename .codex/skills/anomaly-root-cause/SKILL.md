---
name: anomaly-root-cause
description: Use this when the task is to diagnose poor anomaly detection performance, domain gap, unstable results, missed defects, or false positives in industrial anomaly detection experiments. Do not use for pure code refactors unrelated to model behavior.
---

# Goal
Diagnose why the current industrial anomaly detection system underperforms.

# Procedure
1. Read `AGENTS.md` and relevant experiment notes.
2. Identify the exact symptom:
   - high false positives
   - missed anomalies
   - unstable training
   - benchmark good, real data bad
   - image-level good, pixel-level weak
3. Inspect the relevant surfaces:
   - dataset loading and label mapping
   - augmentation and preprocessing
   - model config and scoring path
   - training loss, scheduler, batch composition, checkpoint selection
   - evaluation metrics and thresholding
4. Produce findings under:
   - likely root causes
   - strongest evidence
   - quickest fixes
   - highest-upside deeper changes
5. Recommend a validation sequence from cheapest to most expensive.

# Output format
1. Symptom summary
2. Root cause hypotheses
3. Evidence needed or already present
4. Recommended fixes
5. Validation plan

# Project notes
- Treat false positives on real industrial data as a first-class failure mode.
- Separate domain-gap hypotheses from code-path bugs.
- Prefer fixes that can be validated with a short smoke run or a small artifacted report before full training.
