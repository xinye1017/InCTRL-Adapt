# NB Optimizer Structure Refactor Design

**Goal:** Make the `nb-optimizer` skill capable of deeply understanding a Jupyter Notebook's real dependency graph and analytical intent, then restructuring it into clearer, more modular, and more maintainable notebook sections without changing the user's core workflow or results.

## Core Intent

The skill is not primarily a deletion tool.

Its main job is:

- reconstruct the notebook's logical pipeline
- identify implicit module boundaries
- reduce cross-cell entanglement
- improve readability, maintainability, and top-to-bottom executability

Deletion is secondary. It should happen only when a cell is clearly redundant after the notebook structure has been understood.

## Scope

This design covers two parts:

- update the `nb-optimizer` skill instructions so the workflow centers on structural understanding and modular refactoring
- add a lightweight helper script that supports notebook inspection and safe structural analysis, rather than aggressive code rewriting

This design does not cover:

- automatic semantic rewriting of arbitrary analysis logic
- automatic extraction into external Python modules by default
- agent-free "one-click perfect optimization"
- aggressive code deletion based on shallow heuristics

## User Intent

When a user asks to review or optimize a notebook, they want the agent to understand:

- what each cell is doing
- how cells depend on each other
- where the notebook's true stages begin and end
- where structure, naming, or state flow hurts maintainability

The desired result is a notebook that feels intentionally organized, easier to read, easier to rerun, and easier to extend.

## Design Summary

The skill should operate in three layers:

### 1. Structural understanding

The agent first audits the notebook as a pipeline, not as isolated cells.

It should infer:

- setup and environment cells
- data loading cells
- preprocessing or feature engineering cells
- exploration and visualization cells
- modeling or evaluation cells
- result or export cells
- hidden state dependencies across these stages

### 2. Refactor planning

Before rewriting, the agent should decide:

- which cells should stay as-is
- which cells should be merged because they belong to one logical unit
- which cells should be split because they mix multiple responsibilities
- which cells should move to a different position because the narrative or dependency flow is broken
- which cells are redundant only after the full structure is understood

### 3. Refactor execution

The agent then produces:

- a clear audit report
- a proposed new notebook structure
- a rewritten notebook or cell-by-cell replacement in execution order
- explicit reasons for each major structural change

## Optional Helper Script

The bundled script should support understanding, not replace it.

### Proposed script

- `scripts/inspect_notebook.py`

### Script purpose

Parse a notebook with `nbformat` and emit structured information that helps the agent reason about refactoring, such as:

- cell index and type
- line count and source preview
- import statements
- assigned variable names
- referenced variable names
- markdown headings
- likely stage classification hints
- output presence

### Script output modes

- human-readable summary
- JSON report for downstream reasoning

### Script boundaries

The script should not:

- rewrite code cells
- reorder cells
- delete cells
- decide final structure on its own

It is an inspection aid for the agent, not an optimizer by itself.

## Refactor Principles

The skill should optimize for maintainability through structure:

- one cell, one main responsibility when practical
- minimal hidden state
- imports centralized unless local imports are justified
- setup separated from analysis
- preprocessing separated from visualization and modeling
- markdown narrative aligned with the code that follows
- reproducibility preserved through explicit seeds, paths, and assumptions

When restructuring, prefer:

- merging fragmented cells that form one coherent step
- splitting oversized mixed-purpose cells
- reordering cells to match dependency flow
- renaming variables for consistency when safe

Only after these decisions should the agent remove redundant cells, debug prints, or stale outputs.

## Output Contract

The skill should keep the same 4-step response format:

### Step 1: 审计报告

Prioritize structural and dependency issues first, then redundancy and cleanup issues.

### Step 2: 优化建议

Recommend a clearer module layout and cell ordering, including merge, split, move, and delete decisions with reasons.

### Step 3: 重构后的 Notebook

Provide the rewritten notebook with explicit structural improvements and explain each major change.

### Step 4: 额外推荐

Suggest optional extraction into `.py` modules, validation cells, or parameterized config only when it materially improves maintainability.

## Validation

Validation should focus on whether the skill improves structure without drifting behavior.

For the helper script:

- confirm it can parse notebooks reliably
- confirm it emits stable structural summaries for representative notebooks

For the skill behavior:

- verify the audit identifies module boundaries and hidden dependencies
- verify refactor suggestions emphasize structure before subtraction
- verify the rewritten notebook remains top-to-bottom executable under the same environment assumptions

## Files

- Create: `~/.codex/skills/nb-optimizer/scripts/inspect_notebook.py`
- Modify: `~/.codex/skills/nb-optimizer/SKILL.md`

## Rationale

This design keeps the notebook intelligence in the agent, where contextual reasoning belongs, while using a helper script to make notebook structure easier to inspect consistently. That matches the actual purpose of the skill: not aggressive deletion, but higher-quality notebook architecture.
