"""Agent adapter interface.

An adapter runs some agent in a prepared workspace and reports what it cost.
Adapters do not score anything — the evaluator does that from the artifact the
agent leaves behind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol


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
    extra: dict = field(default_factory=dict)


class AgentAdapter(Protocol):
    name: str

    def run(self, workspace: Path, env: dict[str, str], timeout_s: int) -> AgentRunInfo: ...


_ADAPTERS: dict[str, Callable[..., AgentAdapter]] = {}


def register_adapter(name: str) -> Callable:
    def deco(factory):
        if name in _ADAPTERS:
            raise ValueError(f"adapter {name!r} already registered")
        _ADAPTERS[name] = factory
        return factory

    return deco


def get_adapter(name: str, **kwargs) -> AgentAdapter:
    try:
        factory = _ADAPTERS[name]
    except KeyError:
        raise KeyError(
            f"unknown agent adapter {name!r}; registered: {sorted(_ADAPTERS)}"
        ) from None
    return factory(**kwargs)


def adapter_names() -> list[str]:
    return sorted(_ADAPTERS)
