"""Adapter for the EdgeArray Studio headless agent CLI.

Runs ``edgearray agent run --brief FILE --workspace DIR --project ID
--baseline bN [--model ENDPOINT] [--budget-s N] [--approve-all]`` through the command
adapter. EdgeArray works in its own directory (``edgearray/`` inside the AEDL
workspace) and writes ``submission.json`` (final text, verdicts, usage, tool
calls) and ``transcript.jsonl`` there. Afterwards this adapter maps the
reported usage into AEDL's usage record, copies the transcript to where the
harness collects it, and copies the task deliverable up into the AEDL
workspace, where the evaluator looks for it.

Baselines (defined by EdgeArray): b0 rules only, no LLM; b1 LLM without
tools; b2 native capabilities; b3 plus federated MCP servers; b4 plus skills
and fidelity-policy instructions.

The executable is ``$AEDL_EDGEARRAY_BIN`` when set, else ``edgearray`` on
PATH. AEDL never imports EdgeArray; this module only runs it.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
from pathlib import Path
from typing import Any

from aedl.harness.adapter import AgentRunInfo, AgentUsage, register_adapter
from aedl.harness.adapters.claude_cli import _maybe_float, _maybe_int
from aedl.harness.adapters.command import CommandAdapter
from aedl.harness.workspace import submission_name_from_dir

ENV_BIN = "AEDL_EDGEARRAY_BIN"
BASELINES = ("b0", "b1", "b2", "b3", "b4")
#: EdgeArray's working directory, relative to the AEDL workspace.
WORKSPACE_SUBDIR = "edgearray"
#: Keys of submission.json passed through to the run record unchanged.
_PASSTHROUGH = ("verification_report_hash", "design_state_head", "guard")


class EdgeArrayAdapter(CommandAdapter):
    name = "edgearray"
    # Provider credentials for the model endpoints EdgeArray can route to,
    # and the interpreter paths its federated MCP servers (b3, b4) need.
    required_env: tuple[str, ...] = (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OLLAMA_HOST",
        "EDGEARRAY_SYSML2KIT_PYTHON",
        "EDGEARRAY_OPENSATCOM_PYTHON",
        "EDGEARRAY_ANTENNACAD_PYTHON",
    )

    def __init__(
        self,
        model: str | None = None,
        baseline: str = "b4",
        project: str | None = None,
        approve_all: bool = True,
        binary: str | None = None,
        budget_s: int | None = None,
    ):
        if baseline not in BASELINES:
            raise ValueError(
                f"unknown EdgeArray baseline {baseline!r}; expected one of {BASELINES}"
            )
        self._binary = binary or os.environ.get(ENV_BIN) or "edgearray"
        self._baseline = baseline
        self._project = project
        self._approve_all = approve_all
        self._budget_s = budget_s
        # The command adapter wants a template; ours is built per workspace in
        # build_command, so this string only documents the invocation.
        super().__init__(template=f"{shlex.quote(self._binary)} agent run", model=model)

    def _project_id(self, workspace: Path) -> str:
        if self._project:
            return self._project
        import yaml

        try:
            doc = yaml.safe_load((workspace / "task.yaml").read_text()) or {}
            return f"aedl-{doc['id']}"
        except (OSError, KeyError, TypeError):
            return "aedl"

    def build_command(self, workspace: Path) -> list[str]:
        cmd = [
            self._binary,
            "agent",
            "run",
            "--brief",
            str(workspace / "BRIEF.md"),
            "--workspace",
            str(workspace / WORKSPACE_SUBDIR),
            "--project",
            self._project_id(workspace),
            "--baseline",
            self._baseline,
        ]
        if self._model:
            cmd += ["--model", self._model]
        if self._budget_s is not None:
            cmd += ["--budget-s", str(self._budget_s)]
        if self._approve_all:
            cmd.append("--approve-all")
        return cmd

    def run(self, workspace: Path, env: dict[str, str], timeout_s: int) -> AgentRunInfo:
        ea_ws = workspace / WORKSPACE_SUBDIR
        ea_ws.mkdir(parents=True, exist_ok=True)
        info = super().run(workspace, env, timeout_s)

        extra: dict[str, Any] = {
            "baseline": self._baseline,
            "project": self._project_id(workspace),
            "approve_all": self._approve_all,
            "budget_s": self._budget_s,
            "edgearray_workspace": WORKSPACE_SUBDIR,
        }
        warnings: list[str] = []
        doc = _read_submission(ea_ws / "submission.json", extra)
        info.usage = _usage(doc, self._model)
        for key in _PASSTHROUGH:
            if key in doc:
                extra[key] = doc[key]

        transcript = ea_ws / "transcript.jsonl"
        if transcript.is_file():
            shutil.copyfile(transcript, workspace / ".aedl-agent.transcript.jsonl")
        extra["transcript"] = _summarize_tool_calls(doc.get("tool_calls"))
        # EdgeArray's events are not Claude stream-json, so the reference-path
        # scan the Claude adapter runs does not apply. Say so rather than guess.
        extra["integrity"] = "unknown"

        extra["deliverables_copied"] = _copy_deliverables(workspace, ea_ws, doc, warnings)
        if warnings:
            extra["warnings"] = warnings
        info.extra = extra
        return info


def _read_submission(path: Path, extra: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        extra["parse_error"] = f"EdgeArray wrote no {path.name}"
        return {}
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        extra["parse_error"] = f"{path.name} unreadable: {exc}"
        return {}
    if not isinstance(doc, dict):
        extra["parse_error"] = f"{path.name} is not a JSON object"
        return {}
    return doc


def _usage(doc: dict[str, Any], model: str | None) -> AgentUsage:
    raw = doc.get("usage")
    usage: dict[str, Any] = raw if isinstance(raw, dict) else {}

    def first(*keys: str) -> Any:
        return next((usage[k] for k in keys if usage.get(k) is not None), None)

    reported_model = doc.get("model") or usage.get("model")
    return AgentUsage(
        input_tokens=_maybe_int(first("input_tokens", "prompt_tokens")),
        output_tokens=_maybe_int(first("output_tokens", "completion_tokens")),
        cache_read_tokens=_maybe_int(first("cache_read_tokens")),
        cache_creation_tokens=_maybe_int(first("cache_creation_tokens")),
        cost_usd=_maybe_float(first("cost_usd", "total_cost_usd")),
        num_turns=_maybe_int(first("num_turns", "turns")),
        model=str(reported_model) if reported_model else model,
    )


def _summarize_tool_calls(calls: Any) -> dict[str, Any]:
    by_name: dict[str, int] = {}
    for call in calls if isinstance(calls, list) else []:
        if isinstance(call, dict):
            name = str(call.get("name", "?"))
            by_name[name] = by_name.get(name, 0) + 1
    return {
        "tool_calls_total": sum(by_name.values()),
        "tool_calls_by_name": dict(sorted(by_name.items())),
    }


def _copy_deliverables(
    workspace: Path, ea_ws: Path, doc: dict[str, Any], warnings: list[str]
) -> list[str]:
    """Copy the task deliverable, and anything submission.json lists, up a level.

    Only paths that resolve inside EdgeArray's workspace are copied, a
    deliverable that is or contains a symlink is refused (copying would
    follow it out of the workspace), and an existing file in the AEDL
    workspace is never overwritten.
    """
    expected = Path(submission_name_from_dir(workspace)).parts[0]
    listed = doc.get("deliverables")
    names = [expected] + [str(p) for p in listed] if isinstance(listed, list) else [expected]
    root = ea_ws.resolve()
    copied: list[str] = []
    for name in dict.fromkeys(names):
        src = (ea_ws / name).resolve()
        if Path(name).is_absolute() or not src.is_relative_to(root) or src == root:
            warnings.append(f"refused deliverable {name!r}: outside the EdgeArray workspace")
            continue
        if not src.exists():
            if name != expected:
                warnings.append(f"listed deliverable {name!r} was not written")
            continue
        if _contains_symlink(ea_ws / name):
            warnings.append(f"refused deliverable {name!r}: it is or contains a symlink")
            continue
        dest = workspace / src.relative_to(root)
        if dest.exists():
            warnings.append(f"kept existing {dest.name!r} in the AEDL workspace")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)
        copied.append(str(src.relative_to(root)))
    return copied


def _contains_symlink(path: Path) -> bool:
    if path.is_symlink():
        return True
    if path.is_dir():
        return any(p.is_symlink() for p in path.rglob("*"))
    return False


@register_adapter("edgearray")
def _factory(
    model: str | None = None,
    baseline: str = "b4",
    project: str | None = None,
    approve_all: bool = True,
    binary: str | None = None,
    budget_s: int | None = None,
    **_ignored: Any,
) -> EdgeArrayAdapter:
    return EdgeArrayAdapter(
        model=model,
        baseline=baseline,
        project=project,
        approve_all=approve_all,
        binary=binary,
        budget_s=budget_s,
    )
