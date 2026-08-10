"""Generic subprocess adapter: run any agent CLI in the workspace.

This is the extension point. The command template may reference `{brief}` (path
to BRIEF.md), `{task}` (path to task.yaml), and `{submission}` (the expected
output path). Usage and cost are not reported — no portable way to get them.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path

from aedl.harness.adapter import AgentRunInfo, AgentUsage, register_adapter
from aedl.harness.workspace import SUBMISSION_NAME


class CommandAdapter:
    name = "command"

    def __init__(self, template: str, model: str | None = None):
        if not template:
            raise ValueError("the 'command' adapter requires --agent-command")
        self._template = template
        self._model = model

    def build_command(self, workspace: Path) -> list[str]:
        rendered = self._template.format(
            brief=str(workspace / "BRIEF.md"),
            task=str(workspace / "task.yaml"),
            submission=str(workspace / SUBMISSION_NAME),
        )
        return shlex.split(rendered)

    def run(self, workspace: Path, env: dict[str, str], timeout_s: int) -> AgentRunInfo:
        cmd = self.build_command(workspace)
        start = time.perf_counter()
        timed_out = False
        try:
            proc = subprocess.run(
                cmd, cwd=workspace, env=env, timeout=timeout_s,
                capture_output=True, text=True,
            )
            returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = 124
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + f"\n[aedl] timed out after {timeout_s}s"
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")

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
def _factory(template: str = "", model: str | None = None, **_ignored) -> CommandAdapter:
    return CommandAdapter(template=template, model=model)
