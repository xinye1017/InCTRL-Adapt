---
name: feedback_workstyle
description: Collaboration preferences and work style guidance
type: feedback
---

# Collaboration Preferences

**Terse responses preferred.** Do not summarize what was just done — the user can read the diff. Skip filler words and unnecessary transitions. Lead with the answer or action.

**Why:** Previous experience showed the user values efficiency over verbose explanations.

**How to apply:** When completing a task, output only essential information. No trailing summaries like "I've done X" unless action is required from the user.

**Rule: prefer one bundled PR over many small ones for refactors in this area.** When working on InCTRL model changes, prefer a single coherent PR rather than many tiny commits, to reduce churn.

**Why:** Confirmed after choosing a bundled approach for a prior refactor — user preferred the consolidated workflow.

**How to apply:** When implementing the AdaptCLIP alignment plan (4 tasks), aim for a single PR covering all changes rather than splitting into separate PRs per task.

**Rule: test-first approach for model changes.** Write the failing test first, verify it fails, then implement.

**Why:** The project uses test-driven development for model modifications, per the implementation plan in `docs/superpowers/plans/2026-04-08-inctrl-adaptclip-alignment.md`.

**How to apply:** Before modifying `open_clip/model.py`, add/update tests in `test_holistic_map_shapes.py` first.
