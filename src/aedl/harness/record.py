"""Run-bundle provenance.

Every run writes a manifest, even if the agent crashed or timed out, so that a
failed attempt is still a scorable data point rather than a hole in the record.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

#: Packages whose versions change benchmark results and so must be pinned in the record.
TRACKED_PACKAGES = (
    "aedl",
    "phased-array-modeling",
    "phased-array-systems",
    "opensatcom",
    "numpy",
    "scipy",
    "edgefem",
)


def dependency_versions() -> dict[str, str]:
    out = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def harness_environment() -> dict[str, Any]:
    """The interpreter and library versions that score a run."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": dependency_versions(),
    }


#: Import name -> distribution name, for comparing the agent's probe against the
#: harness environment. Only ``phased_array`` is non-mechanical: the
#: ``phased-array-modeling`` distribution installs a package called
#: ``phased_array``. The probe cannot see ``aedl``, ``scipy`` or ``edgefem``, so
#: they have no counterpart and are left out.
PROBE_TO_DISTRIBUTION = {
    "numpy": "numpy",
    "phased_array": "phased-array-modeling",
    "phased_array_systems": "phased-array-systems",
    "opensatcom": "opensatcom",
}


def _absent(value: Any) -> bool:
    """Both spellings of "this package is not here"."""
    return value is None or value == "not installed"


def environment_skew(
    environment: dict[str, Any], agent_interpreter: dict[str, Any]
) -> list[dict[str, str]]:
    """Packages the harness scored with at a version the agent did not design with.

    A version recorded in the wrong process is not provenance. The harness venv
    is not on the agent's PATH, so the two environments drift, and a run whose
    result was scored against different library versions than the agent used is
    a different experiment than it appears to be.

    Entries where either side is absent, or where the probe failed outright, are
    omitted: that is unknown rather than skew.
    """
    if not agent_interpreter or "error" in agent_interpreter:
        return []
    deps = environment.get("dependencies") or {}
    pairs = [("python", environment.get("python"), agent_interpreter.get("python"))]
    pairs += [
        (dist, deps.get(dist), agent_interpreter.get(probe))
        for probe, dist in PROBE_TO_DISTRIBUTION.items()
    ]
    return [
        {"package": name, "harness": str(harness), "agent": str(agent)}
        for name, harness, agent in pairs
        if not _absent(harness) and not _absent(agent) and harness != agent
    ]


def _git(*args: str, cwd: Path) -> str | None:
    try:
        proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _evaluator_source_sha256(name: str) -> str | None:
    import inspect

    from aedl.registry import get_evaluator

    try:
        source = inspect.getsourcefile(get_evaluator(name))
    except KeyError:
        return None
    if not source:
        return None
    try:
        return hashlib.sha256(Path(source).read_bytes()).hexdigest()
    except OSError:
        return None


def code_provenance(evaluator: str | None = None) -> dict[str, Any]:
    """The AEDL revision and evaluator source that scored a run.

    ``task_sha256`` pins what was asked; this pins what did the asking. The
    installed-metadata version cannot do that job: it records the packaging, not
    the code, and stays fixed across a whole run history while the source moves
    under it. Every bundle written before this field existed reports
    ``dependencies.aedl == "0.0.1"``, a stale editable install, for exactly that
    reason.

    Returns ``{}`` outside a git checkout rather than raising, so an installed
    copy still produces a manifest.
    """
    out: dict[str, Any] = {}
    root = Path(__file__).resolve().parents[3]
    # `git rev-parse` searches upwards, so confirm the checkout it found is this
    # one before trusting the sha. An installed copy inside an unrelated repo
    # would otherwise be recorded under that repo's revision.
    top = _git("rev-parse", "--show-toplevel", cwd=root)
    if top and Path(top).resolve() == root:
        sha = _git("rev-parse", "HEAD", cwd=root)
        if sha:
            out["aedl_git_sha"] = sha
            out["aedl_git_dirty"] = bool(_git("status", "--porcelain", cwd=root))
    if evaluator:
        out["evaluator"] = evaluator
        digest = _evaluator_source_sha256(evaluator)
        if digest:
            out["evaluator_source_sha256"] = digest
    return out


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class RunRecord:
    run_id: str
    task_id: str
    task_sha256: str
    agent: str
    status: str  # pass | fail | no_submission | agent_error | evaluator_error
    started_utc: str
    agent_command: list[str] = field(default_factory=list)
    model: str | None = None
    isolation: str = "tmpdir"
    instrumented: bool = True
    agent_returncode: int | None = None
    timed_out: bool = False
    agent_wall_time_s: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    calls: dict[str, Any] = field(default_factory=dict)
    requirements: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    #: The interpreter and library versions the agent itself reaches for, which
    #: need not match the harness environment that scores the result.
    agent_interpreter: dict[str, Any] = field(default_factory=dict)
    #: Packages where `environment` and `agent_interpreter` disagree. Recording
    #: both halves is only useful if something reads them against each other.
    environment_skew: list[dict[str, str]] = field(default_factory=list)
    #: The AEDL revision and evaluator source that scored this run.
    code: dict[str, Any] = field(default_factory=dict)
    #: Root W3C trace id handed to the MCP servers, so the spans they write to
    #: server-trace.jsonl join back to this bundle.
    trace_id: str | None = None
    #: How the agent authenticated. Subscription runs are not billed per token,
    #: so their reported cost is an API-equivalent estimate, not money spent.
    auth: dict[str, Any] = field(default_factory=dict)
    #: Tool-call counts from the agent's stream-json transcript
    #: (transcript.jsonl in the bundle holds the raw events).
    transcript: dict[str, Any] = field(default_factory=dict)
    #: clean | suspect | unknown — "suspect" when the transcript shows a
    #: filesystem tool referencing a tasks/*/reference/ path. A flag for
    #: review, not a verdict.
    integrity: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["environment"] = d["environment"] or harness_environment()
        if not d["environment_skew"]:
            d["environment_skew"] = environment_skew(d["environment"], d["agent_interpreter"])
        return d

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))


def load_record(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data
