"""Agent harness: run an agent against a task, score the artifact, record the cost."""

from aedl.harness.adapter import (
    AgentAdapter,
    AgentRunInfo,
    AgentUsage,
    adapter_names,
    get_adapter,
    register_adapter,
)
from aedl.harness.record import RunRecord
from aedl.harness.run import run_task

# Importing the package registers the built-in adapters.
from aedl.harness import adapters as _adapters  # noqa: F401

__all__ = [
    "AgentAdapter",
    "AgentRunInfo",
    "AgentUsage",
    "RunRecord",
    "adapter_names",
    "get_adapter",
    "register_adapter",
    "run_task",
]
