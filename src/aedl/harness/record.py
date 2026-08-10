"""Run-bundle provenance.

Every run writes a manifest, even if the agent crashed or timed out, so that a
failed attempt is still a scorable data point rather than a hole in the record.
"""

from __future__ import annotations

import json
import platform
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
    #: How the agent authenticated. Subscription runs are not billed per token,
    #: so their reported cost is an API-equivalent estimate, not money spent.
    auth: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["environment"] = d["environment"] or {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "dependencies": dependency_versions(),
        }
        return d

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))


def load_record(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data
