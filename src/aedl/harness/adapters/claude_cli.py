"""Adapter for the Claude Code CLI in headless mode.

Config isolation matters here: a benchmark run that inherits the operator's
personal settings, skills, and MCP servers does not reproduce for anyone else.
The flags below pin the tool surface and the setting sources, and the run
records exactly which were used.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from aedl.harness.adapter import AgentRunInfo, AgentUsage, register_adapter

DEFAULT_TOOLS = "Bash,Read,Write,Edit,Glob,Grep"
PROMPT = (
    "Read BRIEF.md in the current directory and produce the submission file it "
    "asks for. Work in this directory only. When you are done, verify the file "
    "exists and has the required contents."
)


class ClaudeCliAdapter:
    name = "claude"

    def __init__(
        self,
        model: str | None = None,
        tools: str = DEFAULT_TOOLS,
        setting_sources: str = "project",
        max_budget_usd: float | None = 5.0,
        binary: str = "claude",
    ):
        self._model = model
        self._tools = tools
        self._setting_sources = setting_sources
        self._max_budget_usd = max_budget_usd
        self._binary = binary

    def build_command(self) -> list[str]:
        cmd = [
            self._binary, "-p", PROMPT,
            "--output-format", "json",
            "--permission-mode", "bypassPermissions",
            "--tools", self._tools,
            "--setting-sources", self._setting_sources,
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--exclude-dynamic-system-prompt-sections",
            "--no-session-persistence",
        ]
        if self._model:
            cmd += ["--model", self._model]
        if self._max_budget_usd is not None:
            cmd += ["--max-budget-usd", str(self._max_budget_usd)]
        return cmd

    def run(self, workspace: Path, env: dict[str, str], timeout_s: int) -> AgentRunInfo:
        cmd = self.build_command()
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
            stdout = _as_text(exc.stdout)
            stderr = _as_text(exc.stderr) + f"\n[aedl] timed out after {timeout_s}s"

        (workspace / ".aedl-agent.stdout").write_text(stdout)
        (workspace / ".aedl-agent.stderr").write_text(stderr)

        usage, extra = _parse_result_json(stdout)
        if usage.model is None:
            usage.model = self._model
        return AgentRunInfo(
            returncode=returncode,
            wall_time_s=time.perf_counter() - start,
            usage=usage,
            command=cmd,
            timed_out=timed_out,
            extra=extra,
        )


def _as_text(value) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _parse_result_json(stdout: str) -> tuple[AgentUsage, dict]:
    """Pull usage out of `--output-format json`. Never raises on odd output."""
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return AgentUsage(), {"parse_error": "stdout was not JSON"}
    if isinstance(payload, list):  # defensive: some versions emit a list
        payload = next((p for p in reversed(payload) if isinstance(p, dict)), {})
    if not isinstance(payload, dict):
        return AgentUsage(), {"parse_error": "unexpected JSON shape"}

    raw_usage = payload.get("usage") or {}
    usage = AgentUsage(
        input_tokens=_maybe_int(raw_usage.get("input_tokens")),
        output_tokens=_maybe_int(raw_usage.get("output_tokens")),
        cache_read_tokens=_maybe_int(raw_usage.get("cache_read_input_tokens")),
        cache_creation_tokens=_maybe_int(raw_usage.get("cache_creation_input_tokens")),
        cost_usd=_maybe_float(payload.get("total_cost_usd")),
        num_turns=_maybe_int(payload.get("num_turns")),
        model=payload.get("model"),
    )
    extra = {
        k: payload[k]
        for k in ("session_id", "duration_ms", "duration_api_ms", "is_error",
                  "subtype", "stop_reason", "permission_denials")
        if k in payload
    }
    model_usage = payload.get("modelUsage")
    if isinstance(model_usage, dict) and model_usage:
        extra["model_usage"] = model_usage
        if usage.model is None:
            usage.model = primary_model(model_usage)
    return usage, extra


def primary_model(model_usage: dict) -> str | None:
    """The model that did the work.

    Claude Code delegates cheap internal operations to a small model, so
    `modelUsage` routinely lists two entries and the first one is not the model
    under test. Attribute the run to whichever produced the most output tokens.
    """
    if not model_usage:
        return None
    return max(
        model_usage,
        key=lambda name: (model_usage[name] or {}).get("outputTokens", 0),
    )


def _maybe_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _maybe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@register_adapter("claude")
def _factory(
    model: str | None = None,
    tools: str = DEFAULT_TOOLS,
    setting_sources: str = "project",
    max_budget_usd: float | None = 5.0,
    binary: str = "claude",
    **_ignored,
) -> ClaudeCliAdapter:
    return ClaudeCliAdapter(
        model=model, tools=tools, setting_sources=setting_sources,
        max_budget_usd=max_budget_usd, binary=binary,
    )
