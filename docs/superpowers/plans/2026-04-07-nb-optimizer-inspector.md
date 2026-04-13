# NB Optimizer Inspector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a notebook inspection helper and update the `nb-optimizer` skill so it performs structure-first notebook optimization based on dependency and module-boundary understanding.

**Architecture:** Keep notebook intelligence in the agent and add one small helper script that extracts stable structural signals from `.ipynb` files. Implement the script as a standalone Python CLI powered by `nbformat` and `ast`, then update the skill instructions so the agent uses the script as an inspection aid instead of treating cleanup as the primary objective.

**Tech Stack:** Python 3, `nbformat`, `ast`, `argparse`, `json`, `pytest`

---

### Task 1: Create Failing Tests For Notebook Inspection

**Files:**
- Create: `/Users/xinye/.codex/skills/nb-optimizer/tests/test_inspect_notebook.py`
- Test: `/Users/xinye/.codex/skills/nb-optimizer/tests/test_inspect_notebook.py`

- [ ] **Step 1: Write the failing tests**

```python
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import nbformat


SCRIPT_PATH = Path("/Users/xinye/.codex/skills/nb-optimizer/scripts/inspect_notebook.py")


def load_module():
    spec = importlib.util.spec_from_file_location("inspect_notebook", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_sample_notebook(tmp_path: Path) -> Path:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Setup\nImport dependencies."),
            nbformat.v4.new_code_cell("import pandas as pd\nfrom pathlib import Path"),
            nbformat.v4.new_markdown_cell("## Data Loading"),
            nbformat.v4.new_code_cell(
                "data_path = Path('data.csv')\n"
                "df = pd.read_csv(data_path)\n"
                "features = df[['a', 'b']]\n"
            ),
            nbformat.v4.new_markdown_cell("## Modeling"),
            nbformat.v4.new_code_cell(
                "score = features.mean().sum()\n"
                "print(score)\n",
                execution_count=7,
                outputs=[nbformat.v4.new_output("stream", text="3.0\n")],
            ),
        ]
    )
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.11"}

    notebook_path = tmp_path / "sample.ipynb"
    nbformat.write(notebook, notebook_path)
    return notebook_path


def test_build_report_extracts_structure_and_symbol_hints(tmp_path: Path):
    notebook_path = build_sample_notebook(tmp_path)
    module = load_module()

    report = module.build_report(notebook_path)

    assert report["notebook_path"] == str(notebook_path)
    assert report["cell_count"] == 6
    assert report["stage_counts"]["setup"] >= 1
    assert report["stage_counts"]["data_loading"] >= 1
    assert report["stage_counts"]["modeling"] >= 1

    code_cells = [cell for cell in report["cells"] if cell["cell_type"] == "code"]
    assert code_cells[0]["imports"] == ["import pandas as pd", "from pathlib import Path"]
    assert code_cells[1]["assigned_names"] == ["data_path", "df", "features"]
    assert "pd" in code_cells[1]["referenced_names"]
    assert code_cells[2]["has_output"] is True
    assert code_cells[2]["stage_hint"] == "modeling"


def test_cli_json_output_emits_stable_report(tmp_path: Path):
    notebook_path = build_sample_notebook(tmp_path)

    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(notebook_path), "--format", "json"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["notebook_path"] == str(notebook_path)
    assert payload["cell_count"] == 6
    assert payload["cells"][0]["markdown_headings"] == ["# Setup"]
    assert payload["cells"][3]["stage_hint"] == "data_loading"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest /Users/xinye/.codex/skills/nb-optimizer/tests/test_inspect_notebook.py -q`

Expected: FAIL because `/Users/xinye/.codex/skills/nb-optimizer/scripts/inspect_notebook.py` does not exist yet.

- [ ] **Step 3: Commit the failing tests**

```bash
git add /Users/xinye/.codex/skills/nb-optimizer/tests/test_inspect_notebook.py
git commit -m "test: add failing notebook inspector tests"
```

### Task 2: Implement The Notebook Inspection CLI

**Files:**
- Create: `/Users/xinye/.codex/skills/nb-optimizer/scripts/inspect_notebook.py`
- Modify: `/Users/xinye/.codex/skills/nb-optimizer/tests/test_inspect_notebook.py`
- Test: `/Users/xinye/.codex/skills/nb-optimizer/tests/test_inspect_notebook.py`

- [ ] **Step 1: Write the minimal implementation**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

import nbformat


STAGE_KEYWORDS = {
    "setup": ("setup", "environment", "import", "config"),
    "data_loading": ("load", "read_", "dataset", "data", "csv", "parquet"),
    "preprocessing": ("clean", "transform", "feature", "encode", "preprocess"),
    "eda": ("plot", "visual", "distribution", "explore", "eda"),
    "modeling": ("train", "fit", "predict", "model", "score", "evaluate"),
    "results": ("result", "summary", "export", "save", "submission"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a Jupyter notebook and emit structure-oriented analysis hints."
    )
    parser.add_argument("input_path", type=Path, help="Path to the .ipynb notebook")
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format for the report",
    )
    return parser.parse_args()


def source_preview(source: str, limit: int = 80) -> str:
    compact = " ".join(line.strip() for line in source.splitlines() if line.strip())
    return compact[:limit]


def extract_markdown_headings(source: str) -> list[str]:
    return [line.strip() for line in source.splitlines() if line.lstrip().startswith("#")]


def collect_imports(tree: ast.AST) -> list[str]:
    imports: list[str] = []
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    imports.append(f"import {alias.name} as {alias.asname}")
                else:
                    imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            parts = []
            for alias in node.names:
                if alias.asname:
                    parts.append(f"{alias.name} as {alias.asname}")
                else:
                    parts.append(alias.name)
            imports.append(f"from {module} import {', '.join(parts)}")
    return imports


def collect_assigned_names(tree: ast.AST) -> list[str]:
    assigned: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                assigned.update(iter_target_names(target))
        elif isinstance(node, ast.AnnAssign):
            assigned.update(iter_target_names(node.target))
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            assigned.update(iter_target_names(node.target))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assigned.add(node.name)
    return sorted(assigned)


def iter_target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in node.elts:
            names.update(iter_target_names(item))
        return names
    return set()


def collect_referenced_names(tree: ast.AST) -> list[str]:
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    return sorted(names)


def infer_stage(
    cell_type: str,
    source: str,
    markdown_headings: list[str],
    imports: list[str],
    assigned_names: list[str],
    referenced_names: list[str],
) -> str:
    lowered = "\n".join(markdown_headings + [source]).lower()
    if cell_type == "markdown":
        for stage, keywords in STAGE_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                return stage
        return "narrative"
    if imports and not assigned_names and not referenced_names:
        return "setup"
    for stage, keywords in STAGE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return stage
    return "analysis"


def build_cell_report(cell: dict[str, Any], index: int) -> dict[str, Any]:
    source = cell.get("source", "")
    if cell["cell_type"] == "markdown":
        headings = extract_markdown_headings(source)
        return {
            "index": index,
            "cell_type": "markdown",
            "line_count": len(source.splitlines()),
            "source_preview": source_preview(source),
            "markdown_headings": headings,
            "imports": [],
            "assigned_names": [],
            "referenced_names": [],
            "has_output": False,
            "stage_hint": infer_stage("markdown", source, headings, [], [], []),
        }

    headings: list[str] = []
    imports: list[str] = []
    assigned_names: list[str] = []
    referenced_names: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    if tree is not None:
        imports = collect_imports(tree)
        assigned_names = collect_assigned_names(tree)
        referenced_names = collect_referenced_names(tree)

    return {
        "index": index,
        "cell_type": "code",
        "line_count": len(source.splitlines()),
        "source_preview": source_preview(source),
        "markdown_headings": headings,
        "imports": imports,
        "assigned_names": assigned_names,
        "referenced_names": referenced_names,
        "has_output": bool(cell.get("outputs")),
        "stage_hint": infer_stage(
            "code", source, headings, imports, assigned_names, referenced_names
        ),
    }


def build_report(notebook_path: Path) -> dict[str, Any]:
    notebook = nbformat.read(notebook_path, as_version=4)
    cells = [build_cell_report(cell, index) for index, cell in enumerate(notebook.cells)]
    stage_counts = Counter(cell["stage_hint"] for cell in cells)
    return {
        "notebook_path": str(notebook_path),
        "cell_count": len(cells),
        "stage_counts": dict(stage_counts),
        "cells": cells,
    }


def render_text_report(report: dict[str, Any]) -> str:
    lines = [
        f"Notebook: {report['notebook_path']}",
        f"Cells: {report['cell_count']}",
        "Stage counts:",
    ]
    for stage, count in sorted(report["stage_counts"].items()):
        lines.append(f"  - {stage}: {count}")
    lines.append("Cells:")
    for cell in report["cells"]:
        lines.append(
            f"  - [{cell['index']}] {cell['cell_type']} "
            f"stage={cell['stage_hint']} lines={cell['line_count']} "
            f"preview={cell['source_preview']!r}"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    report = build_report(args.input_path)
    if args.format == "json":
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(render_text_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the tests to verify they pass**

Run: `pytest /Users/xinye/.codex/skills/nb-optimizer/tests/test_inspect_notebook.py -q`

Expected: PASS with `2 passed`.

- [ ] **Step 3: Smoke-test the CLI help output**

Run: `python3 /Users/xinye/.codex/skills/nb-optimizer/scripts/inspect_notebook.py --help`

Expected: PASS and usage text containing `input_path` and `--format`.

- [ ] **Step 4: Commit the script implementation**

```bash
git add /Users/xinye/.codex/skills/nb-optimizer/scripts/inspect_notebook.py /Users/xinye/.codex/skills/nb-optimizer/tests/test_inspect_notebook.py
git commit -m "feat: add notebook structure inspector"
```

### Task 3: Update The Skill Instructions And Validate The Skill

**Files:**
- Modify: `/Users/xinye/.codex/skills/nb-optimizer/SKILL.md`
- Test: `/Users/xinye/.codex/skills/nb-optimizer/SKILL.md`

- [ ] **Step 1: Rewrite the skill instructions around structure-first optimization**

Replace `/Users/xinye/.codex/skills/nb-optimizer/SKILL.md` with:

```md
---
name: nb-optimizer
description: Audit, restructure, and optimize Jupyter Notebook (.ipynb) files by understanding their dependency flow, module boundaries, and analytical intent before refactoring. Use when Codex needs to review, modularize, clean up, or productionize a notebook while preserving the user's core workflow and outputs. Also use when the user says things like "审查这个 notebook", "优化这个 notebook", "refactor this notebook", or asks for notebook-level structure, maintainability, and consistency improvements.
---

# NB Optimizer

Optimize notebooks through structural understanding first and subtraction second.
Preserve the original analytical goal and core behavior while making the notebook easier to read, rerun, maintain, and extend.

## Input Contract

- Accept one of:
  - a `.ipynb` file path
  - full notebook JSON
  - selected cells pasted inline
  - a direct request to review or optimize a notebook already present in the workspace
- Reconstruct missing details from the notebook itself whenever possible.
- If the notebook is large, audit first and refactor in phases.

## Output Contract

Respond in exactly 4 steps using Markdown:

### Step 1: 审计报告
- Summarize issues by `高 / 中 / 低` priority.
- Prioritize module-boundary problems, hidden dependencies, state leakage, narrative breaks, and maintainability risks before cleanup-only issues.

### Step 2: 优化建议
- Propose the target notebook structure and recommended cell order.
- Say which cells to keep, merge, split, move, or delete, and explain why.
- Quantify expected gains when possible, such as fewer cross-cell dependencies, clearer stages, or less duplication.

### Step 3: 重构后的 Notebook
- Provide the optimized notebook as either:
  - full `.ipynb` JSON when practical, or
  - cell-by-cell Markdown and code replacements in execution order
- After each major change, add a short reason in parentheses.
- Ensure the notebook is cleanly runnable from top to bottom under the same environment assumptions.

### Step 4: 额外推荐
- Add only pragmatic follow-ups such as extracting helpers into `.py` modules, adding validation cells, parameterizing paths, or introducing lightweight config.
- Do not propose unrelated new features.

## Workflow

1. Inspect the full notebook structure before proposing edits.
2. Reconstruct the logical pipeline from imports to final outputs.
3. Infer module boundaries and notebook stages, including:
   - setup and environment
   - data loading
   - preprocessing or feature engineering
   - exploration or visualization
   - modeling or evaluation
   - results or export
4. Identify structural problems in this order:
   - broken dependency flow
   - hidden state or out-of-order execution dependence
   - cells that mix multiple responsibilities
   - fragmented stages that should be merged
   - cells that belong in a different position
   - redundancy that becomes obvious after structure is understood
5. Prefer structural refactoring:
   - merge cells that form one coherent unit
   - split cells that do too many jobs
   - reorder cells to match dependency flow and narrative clarity
   - rename variables for consistency when safe
6. Use deletion only after the notebook structure is understood and only for clearly redundant cells.
7. Keep Markdown aligned with the code that follows.
8. Preserve analytical meaning and outputs the user clearly relies on.

## Inspection Aid

Use `/Users/xinye/.codex/skills/nb-optimizer/scripts/inspect_notebook.py` when a notebook is large enough that a structural summary would improve confidence.

The script is an inspection helper, not an optimizer. Use it to gather:

- cell types and indexes
- source previews and line counts
- markdown headings
- import statements
- assigned and referenced names
- output presence
- stage hints

Use the script like this:

```bash
python3 /Users/xinye/.codex/skills/nb-optimizer/scripts/inspect_notebook.py /path/to/notebook.ipynb
python3 /Users/xinye/.codex/skills/nb-optimizer/scripts/inspect_notebook.py /path/to/notebook.ipynb --format json
```

## Hard Rules

- Preserve the user's core functionality and intent.
- Do not add new features unless the user explicitly requests them.
- Optimize for maintainability through structure, not shallow cleanup.
- Avoid notebook-only hidden state; the final version must execute in order.
- Use a professional, objective tone and explain the reason for every major change.
```

- [ ] **Step 2: Validate the skill structure**

Run: `python3 /Users/xinye/.codex/skills/.system/skill-creator/scripts/quick_validate.py /Users/xinye/.codex/skills/nb-optimizer`

Expected: PASS with `Skill is valid!`

- [ ] **Step 3: Re-run the notebook inspector tests**

Run: `pytest /Users/xinye/.codex/skills/nb-optimizer/tests/test_inspect_notebook.py -q`

Expected: PASS with `2 passed`.

- [ ] **Step 4: Commit the documentation update**

```bash
git add /Users/xinye/.codex/skills/nb-optimizer/SKILL.md
git commit -m "docs: refocus nb optimizer on structure-first refactoring"
```
