"""Agent workspace materialization.

The workspace is what an agent under test sees: the task specification and a
brief describing the deliverable contract. It never contains the reference
solution, and by default it lives outside the repository so that a shell command
cannot reach `tasks/*/reference/`.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

from aedl.spec import TaskSpec

SUBMISSION_NAME = "submission.npz"


def task_digest(spec: TaskSpec) -> str:
    """SHA-256 of the task file, so a run record pins the exact spec scored."""
    assert spec.path is not None, "TaskSpec must come from a file to be hashed"
    return hashlib.sha256(spec.path.read_bytes()).hexdigest()


def _requirements_table(spec: TaskSpec) -> str:
    rows = ["| requirement | metric | limit |", "|---|---|---|"]
    rows += [f"| {r.id} | `{r.metric}` | {r.limit} |" for r in spec.requirements]
    return "\n".join(rows)


def render_brief(spec: TaskSpec) -> str:
    """The task statement handed to the agent.

    Includes the requirements and their thresholds — a real design spec states
    what it must meet. Excludes any hint of the reference technique.
    """
    assert spec.path is not None
    context_yaml = _extract_block(spec.path.read_text(), "context:")
    deliverable = spec.deliverable.get("description", "").strip()
    fmt = spec.deliverable.get("format", "npz")

    return f"""# {spec.title}

Task id: `{spec.id}`  (tier {spec.tier})

{spec.summary}

## Design context

```yaml
{context_yaml}
```

## What to submit

Write your design to **`{SUBMISSION_NAME}`** in this directory, format `{fmt}`.

{deliverable}

## Requirements

Your submission must satisfy every requirement below. Each is scored by
deterministic code that recomputes the physics from your submitted file.

{_requirements_table(spec)}

## How scoring works

- The evaluator applies the hardware constraints and any element failures listed
  in the design context itself. Do not pre-apply failures to your weights;
  submit the weights you would program into working hardware.
- Metrics are computed from your file alone. Nothing you write in prose is scored.
- You may use any method. `numpy` and the `phased_array` package are installed.

Full machine-readable spec: `task.yaml` in this directory.
"""


def _extract_block(text: str, header: str) -> str:
    """Return the indented YAML block following `header`, header included."""
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.startswith(header))
    except StopIteration:
        return ""
    out = [lines[start]]
    for ln in lines[start + 1 :]:
        if ln and not ln[0].isspace():
            break
        out.append(ln)
    return "\n".join(out).rstrip()


def materialize(spec: TaskSpec, isolation: str = "tmpdir", parent: Path | None = None) -> Path:
    """Create the agent workspace and return its path.

    isolation:
      "tmpdir"  - a fresh temp directory outside the repo (default)
      "inplace" - `parent/workspace`, for debugging; readable from the repo tree
    """
    if isolation == "tmpdir":
        workspace = Path(tempfile.mkdtemp(prefix="aedl-ws-"))
    elif isolation == "inplace":
        if parent is None:
            raise ValueError("isolation='inplace' requires a parent directory")
        workspace = parent / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError(f"unknown isolation mode {isolation!r}")

    assert spec.path is not None
    shutil.copy2(spec.path, workspace / "task.yaml")
    (workspace / "BRIEF.md").write_text(render_brief(spec))
    return workspace


def collect(workspace: Path, destination: Path) -> None:
    """Copy the finished workspace into the run bundle."""
    if workspace.resolve() == destination.resolve():
        return
    shutil.copytree(workspace, destination, dirs_exist_ok=True)
