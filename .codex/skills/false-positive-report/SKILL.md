---
name: false-positive-report
description: Use this when the task is to analyze false positives, produce evaluation artifacts, summarize failure cases, compare benchmark vs real-world industrial performance, or write a report from prediction outputs.
---

# Goal
Generate a concrete false-positive analysis report.

# Procedure
1. Gather prediction artifacts, labels, configs, and experiment notes if available.
2. Separate image-level and pixel-level errors.
3. Group false positives by pattern:
   - texture confusion
   - illumination or reflection
   - repetitive background
   - annotation ambiguity
   - domain shift
   - threshold or calibration issue
   - preprocessing mismatch
4. Save findings to `reports/` with concise tables and representative examples when available.
5. End with targeted remediation suggestions linked to each error pattern.

# Output format
1. Error taxonomy
2. Representative cases
3. Likely source of each error
4. Suggested data/model/training fixes
5. Next validation run

# Report naming
Use `reports/false_positive_report_YYYY-MM-DD.md` unless the user gives a specific name.
