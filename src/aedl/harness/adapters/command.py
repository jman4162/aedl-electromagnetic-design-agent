"""Generic subprocess adapter: run any agent CLI in the workspace.

This is the extension point. The command template may reference `{brief}` (path
to BRIEF.md), `{task}` (path to task.yaml), and `{submission}` (the expected
output path). Usage and cost are not reported — no portable way to get them.
"""

from __future__ import annotations

import shlex
import time
from pathlib import Path
from typing import Any

from aedl.harness.adapter import (
    AgentRunInfo,
    AgentUsage,
    register_adapter,
    run_subprocess,
)
from aedl.harness.workspace import submission_name_from_dir


class CommandAdapter:
    name = "command"
    # Unknown third-party agent: pass nothing by default. Operators who need a
    # credential should construct the adapter with it declared explicitly.
    required_env: tuple[str, ...] = ()

    def __init__(self, template: str, model: str | None = None):
        if not template:
            raise ValueError("the 'command' adapter requires --agent-command")
        self._template = template
        self._model = model

    def build_command(self, workspace: Path) -> list[str]:
        rendered = self._template.format(
            brief=str(workspace / "BRIEF.md"),
            task=str(workspace / "task.yaml"),
            submission=str(workspace / submission_name_from_dir(workspace)),
        )
        return shlex.split(rendered)

    def run(self, workspace: Path, env: dict[str, str], timeout_s: int) -> AgentRunInfo:
        cmd = self.build_command(workspace)
        start = time.perf_counter()
        returncode, stdout, stderr, timed_out = run_subprocess(cmd, workspace, env, timeout_s)

        (workspace / ".aedl-agent.stdout").write_text(stdout)
        (workspace / ".aedl-agent.stderr").write_text(stderr)
        return AgentRunInfo(
            returncode=returncode,
            wall_time_s=time.perf_counter() - start,
            usage=AgentUsage(model=self._model),
            command=cmd,
            timed_out=timed_out,
        )


@register_adapter("command")
def _factory(template: str = "", model: str | None = None, **_ignored: Any) -> CommandAdapter:
    return CommandAdapter(template=template, model=model)
