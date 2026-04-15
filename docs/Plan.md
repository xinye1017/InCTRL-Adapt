# Project Plan

## Current Phase
Establish a reproducible improvement loop for InCTRL on industrial anomaly detection, with special attention to false positives on real data.

## Near-Term Milestones
1. Baseline inventory
   - identify current training, evaluation, and checkpoint scripts
   - record the best known benchmark and real-data behavior
   - document current false-positive symptoms
2. False-positive diagnosis
   - create an error taxonomy
   - separate data/domain issues from model/scoring issues
   - identify the cheapest validation run
3. Minimal ablations
   - change one factor at a time
   - record config knobs and expected effects
   - compare against the same baseline split and metrics
4. Report and iterate
   - save concise reports under `reports/`
   - update `docs/Status.md` after each meaningful run

## Default Validation Ladder
1. Static inspection or config parse check.
2. Import and smoke test for the changed path.
3. One tiny run on a small class/subset.
4. Full benchmark comparison.
5. Real industrial false-positive review.

## Open Questions
- Which real industrial dataset or folder is the primary optimization target?
- Which metric best represents the business cost of false positives?
- Are thresholds fixed, calibrated per class, or selected globally?
- Which current checkpoint is considered the baseline?
