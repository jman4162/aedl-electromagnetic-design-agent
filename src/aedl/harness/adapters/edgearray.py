"""Adapter for the EdgeArray Studio headless agent CLI.

Runs ``edgearray agent run --brief FILE --workspace DIR --project ID
--data-dir DIR --baseline bN [--model ENDPOINT] [--budget-s N] [--approve-all]``
through the command adapter. EdgeArray works in its own directory
(``edgearray/`` inside the AEDL workspace) and writes ``submission.json``
(final text, verdicts, usage, tool calls) and ``transcript.jsonl`` there.

Every run gets its own data directory (``edgearray/data-<token>``) and project
id (``aedl-<task>-<token>``), so a run starts with an empty design state and
no run can see another's artifacts. That is what keeps the baselines of a
ladder from contaminating each other.

The EdgeArray agent cannot write files into the AEDL workspace. Afterwards
this adapter maps the reported usage into AEDL's usage record, copies the
transcript to where the harness collects it, and builds the task deliverable
from the run's data directory: t4-001 through
``scripts/bundle_from_edgearray.py``, t1-001 from the head ``element.design``
artifact. A task without a mapping is recorded as an error, not skipped.

Baselines (defined by EdgeArray): b0 rules only, no LLM; b1 LLM without
tools; b2 native capabilities; b3 plus federated MCP servers; b4 plus skills
and fidelity-policy instructions.

The executable is ``$AEDL_EDGEARRAY_BIN`` when set, else ``edgearray`` on
PATH. AEDL never imports EdgeArray; this module only runs it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
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
_CONVERTER = Path(__file__).resolve().parents[4] / "scripts" / "bundle_from_edgearray.py"


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
        self._run_token: str | None = None
        # The command adapter wants a template; ours is built per workspace in
        # build_command, so this string only documents the invocation.
        super().__init__(template=f"{shlex.quote(self._binary)} agent run", model=model)

    def _token(self) -> str:
        """A token unique to this run; ``run`` replaces it for every call."""
        if self._run_token is None:
            self._run_token = uuid.uuid4().hex[:8]
        return self._run_token

    def _task_id(self, workspace: Path) -> str | None:
        import yaml

        try:
            doc = yaml.safe_load((workspace / "task.yaml").read_text()) or {}
            return str(doc["id"])
        except (OSError, KeyError, TypeError):
            return None

    def _project_id(self, workspace: Path) -> str:
        if self._project:
            return self._project
        task_id = self._task_id(workspace)
        return f"aedl-{task_id}-{self._token()}" if task_id else f"aedl-{self._token()}"

    def _data_dir(self, workspace: Path) -> Path:
        return workspace / WORKSPACE_SUBDIR / f"data-{self._token()}"

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
            "--data-dir",
            str(self._data_dir(workspace)),
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
        self._run_token = uuid.uuid4().hex[:8]
        ea_ws = workspace / WORKSPACE_SUBDIR
        ea_ws.mkdir(parents=True, exist_ok=True)
        info = super().run(workspace, env, timeout_s)

        data_dir = self._data_dir(workspace)
        project = self._project_id(workspace)
        extra: dict[str, Any] = {
            "baseline": self._baseline,
            "project": project,
            "data_dir": str(data_dir.relative_to(workspace)),
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

        # A timed-out run is not scored, so building from its half-written
        # design state would only suggest a deliverable that does not count.
        if not info.timed_out:
            _build_deliverable(
                self._task_id(workspace), workspace, data_dir, project, extra, warnings
            )
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
        cache_creation_tokens=_maybe_int(first("cache_write_tokens", "cache_creation_tokens")),
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


class DeliverableError(RuntimeError):
    """The run's data directory cannot yield the task deliverable."""


def _converter() -> ModuleType:
    """scripts/bundle_from_edgearray.py, loaded once. It imports nothing from EdgeArray."""
    name = "aedl_bundle_from_edgearray"
    if name in sys.modules:
        return sys.modules[name]
    if not _CONVERTER.is_file():
        raise DeliverableError(
            f"{_CONVERTER} not found; deliverables are built from an AEDL source checkout"
        )
    spec = importlib.util.spec_from_file_location(name, _CONVERTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _build_t4_001(data_dir: Path, project: str, workspace: Path) -> dict[str, Any]:
    """submission/ (bundle.json, element, board, conversion.json) from the design head."""
    conv = _converter()
    final = workspace / "submission"
    staging = workspace / ".submission.partial"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        conv.convert(data_dir, project, staging)
        record = json.loads((staging / "conversion.json").read_text())
        os.replace(staging, final)
    except conv.ConversionError as exc:
        raise DeliverableError(str(exc)) from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    files = sorted(
        f"submission/{p.relative_to(final).as_posix()}" for p in final.rglob("*") if p.is_file()
    )
    return {
        "files": files,
        "source": "design head",
        "source_revision": record["source_revision"],
        "filled_in": record["filled_in"],
        "assumed_verdicts": record["assumed_verdicts"],
        "_warnings": [f"filled in: {note}" for note in record["filled_in"]],
    }


def _build_t1_001(data_dir: Path, project: str, workspace: Path) -> dict[str, Any]:
    """design.yaml from the head ``element.design`` artifact, else the most recent one."""
    conv = _converter()
    warnings: list[str] = []
    payload: dict[str, Any] | None = None
    source = "design head"
    try:
        proj = conv.project_dir(data_dir, project)
        if (proj / "design" / "head.json").is_file():
            revision = conv.head_revision(proj)
            if "element" in revision.get("nodes", {}):
                _, _, _, payload = conv.node_artifact(proj, revision, "element", "element.design")
            else:
                warnings.append("design head has no element node; using the most recent artifact")
        else:
            warnings.append("no design head; using the most recent element.design artifact")
        if payload is None:
            payload = _latest_element_payload(proj)
            source = "most recent element.design artifact"
        text = conv.element_yaml(payload)
    except conv.ConversionError as exc:
        raise DeliverableError(str(exc)) from exc
    (workspace / "design.yaml").write_text(text)
    return {"files": ["design.yaml"], "source": source, "_warnings": warnings}


def _latest_element_payload(proj: Path) -> dict[str, Any]:
    """Payload of the element.design artifact with the latest manifest ``created_at``."""
    candidates: list[tuple[str, str, Path]] = []
    for manifest_path in (proj / "artifacts").glob("*/*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(manifest, dict) and manifest.get("kind") == "element.design":
            candidates.append(
                (
                    str(manifest.get("created_at", "")),
                    manifest_path.parent.name,
                    manifest_path.parent,
                )
            )
    if not candidates:
        raise DeliverableError(f"no element.design artifact under {proj / 'artifacts'}")
    _, _, folder = max(candidates)
    try:
        payload = json.loads((folder / "payload.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DeliverableError(f"cannot read {folder / 'payload.json'}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DeliverableError(f"{folder / 'payload.json'} is not a JSON object")
    return payload


#: Task id -> builder of the task deliverable from the run's data directory.
DELIVERABLE_BUILDERS: dict[str, Callable[[Path, str, Path], dict[str, Any]]] = {
    "t1-001": _build_t1_001,
    "t4-001": _build_t4_001,
}


def _build_deliverable(
    task_id: str | None,
    workspace: Path,
    data_dir: Path,
    project: str,
    extra: dict[str, Any],
    warnings: list[str],
) -> None:
    """Build the task deliverable into the AEDL workspace, or record why not.

    The agent cannot write into the AEDL workspace, so the evaluator's input
    comes from the run's data directory. An existing file is never overwritten.
    Any failure goes to ``extra["deliverable_error"]`` and ``warnings``; the
    harness then reports it as the reason the run has no submission.
    """

    def fail(message: str) -> None:
        extra["deliverable_error"] = message
        warnings.append(message)

    if task_id is None:
        fail("cannot read the task id from task.yaml; no deliverable built")
        return
    builder = DELIVERABLE_BUILDERS.get(task_id)
    if builder is None:
        fail(
            f"no deliverable mapping for task {task_id}; the EdgeArray adapter builds "
            f"deliverables for {', '.join(sorted(DELIVERABLE_BUILDERS))} only"
        )
        return
    expected = Path(submission_name_from_dir(workspace)).parts[0]
    if (workspace / expected).exists():
        warnings.append(f"kept existing {expected!r} in the AEDL workspace; nothing built")
        return
    try:
        built = builder(data_dir, project, workspace)
    except DeliverableError as exc:
        fail(f"could not build the {task_id} deliverable from {data_dir.name}: {exc}")
        return
    except (KeyError, TypeError, ValueError, OSError) as exc:
        # A payload shaped differently from what the converter reads is the
        # agent's result, not a crash of the harness.
        fail(
            f"could not build the {task_id} deliverable from {data_dir.name}: "
            f"malformed artifact ({type(exc).__name__}: {exc})"
        )
        return
    warnings.extend(built.pop("_warnings", []))
    extra["deliverable"] = {"task": task_id, **built}


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
