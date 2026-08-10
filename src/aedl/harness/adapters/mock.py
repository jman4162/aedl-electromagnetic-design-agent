"""Scripted adapter used by the tests.

Takes a callable that is handed the workspace directory and may write a
submission (or not, or raise). Nothing here touches a network or a model.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aedl.harness.adapter import AgentRunInfo, AgentUsage, register_adapter


class MockAdapter:
    name = "mock"

    def __init__(self, behavior: Callable[[Path], None] | None = None, returncode: int = 0):
        self._behavior = behavior
        self._returncode = returncode

    def run(self, workspace: Path, env: dict[str, str], timeout_s: int) -> AgentRunInfo:
        start = time.perf_counter()
        if self._behavior is not None:
            self._behavior(workspace)
        return AgentRunInfo(
            returncode=self._returncode,
            wall_time_s=time.perf_counter() - start,
            usage=AgentUsage(model="mock"),
            command=["<mock>"],
        )


@register_adapter("mock")
def _factory(
    behavior: Callable[[Path], None] | None = None,
    returncode: int = 0,
    **_ignored: Any,
) -> MockAdapter:
    return MockAdapter(behavior=behavior, returncode=returncode)
