"""Agent adapter interface.

An adapter runs some agent in a prepared workspace and reports what it cost.
Adapters do not score anything — the evaluator does that from the artifact the
agent leaves behind.
"""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class AgentUsage:
    """What the agent consumed. Fields are None when an adapter cannot report them."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    cost_usd: float | None = None
    num_turns: int | None = None
    model: str | None = None


@dataclass
class AgentRunInfo:
    returncode: int
    wall_time_s: float
    usage: AgentUsage = field(default_factory=AgentUsage)
    command: list[str] = field(default_factory=list)
    timed_out: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class AgentAdapter(Protocol):
    name: str
    #: Host environment variables this adapter needs passed through, beyond
    #: the harness base allowlist. Keep it to credentials the agent genuinely
    #: requires; everything listed here is visible to the agent.
    required_env: tuple[str, ...]

    def run(self, workspace: Path, env: dict[str, str], timeout_s: int) -> AgentRunInfo: ...


def run_subprocess(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout_s: int,
) -> tuple[int, str, str, bool]:
    """Run an agent command, killing the whole process tree on timeout.

    `subprocess.run(timeout=...)` kills only the direct child, so anything the
    agent backgrounded keeps running: burning CPU, still writing to the
    workspace, and able to read the reference once permissions are restored.
    Starting a new session puts the child and its descendants in one process
    group that can be signalled as a unit.

    Returns (returncode, stdout, stderr, timed_out).
    """
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        return proc.returncode, stdout, stderr, False
    except subprocess.TimeoutExpired:
        _terminate_group(proc)
        stdout, stderr = proc.communicate()
        return 124, stdout or "", (stderr or "") + f"\n[aedl] timed out after {timeout_s}s", True


def _terminate_group(proc: subprocess.Popen[str]) -> None:
    """SIGTERM the process group, then SIGKILL anything that ignores it."""
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        proc.kill()
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def as_text(value: object) -> str:
    """Coerce subprocess output to str; TimeoutExpired may hand back bytes."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


_ADAPTERS: dict[str, Callable[..., AgentAdapter]] = {}


def register_adapter(
    name: str,
) -> Callable[[Callable[..., AgentAdapter]], Callable[..., AgentAdapter]]:
    def deco(factory: Callable[..., AgentAdapter]) -> Callable[..., AgentAdapter]:
        if name in _ADAPTERS:
            raise ValueError(f"adapter {name!r} already registered")
        _ADAPTERS[name] = factory
        return factory

    return deco


def get_adapter(name: str, **kwargs: Any) -> AgentAdapter:
    try:
        factory = _ADAPTERS[name]
    except KeyError:
        raise KeyError(f"unknown agent adapter {name!r}; registered: {sorted(_ADAPTERS)}") from None
    return factory(**kwargs)


def adapter_names() -> list[str]:
    return sorted(_ADAPTERS)
