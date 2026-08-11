"""Adapter for the Claude Code CLI in headless mode.

Config isolation matters here: a benchmark run that inherits the operator's
personal settings, skills, and MCP servers does not reproduce for anyone else.
The flags below pin the tool surface and the setting sources, and the run
records exactly which were used.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from aedl.harness.adapter import (
    AgentRunInfo,
    AgentUsage,
    register_adapter,
    run_subprocess,
)

DEFAULT_TOOLS = "Bash,Read,Write,Edit,Glob,Grep"
PROMPT = (
    "Read BRIEF.md in the current directory and produce the submission file it "
    "asks for. Work in this directory only. When you are done, verify the file "
    "exists and has the required contents."
)


class ClaudeCliAdapter:
    name = "claude"
    # Subscription auth resolves through HOME (keychain by uid on macOS, a file
    # under $HOME on Linux), so no credential variable is needed. API-key auth
    # is opt-in and passed through only when the operator has set it.
    required_env: tuple[str, ...] = (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
    )

    def __init__(
        self,
        model: str | None = None,
        tools: str = DEFAULT_TOOLS,
        setting_sources: str = "project",
        max_budget_usd: float | None = 5.0,
        binary: str = "claude",
        mcp_config: Path | str | None = None,
        mcp_tools: str = "",
    ):
        self._model = model
        self._tools = tools
        self._setting_sources = setting_sources
        self._max_budget_usd = max_budget_usd
        self._binary = binary
        # An explicit MCP config composes with --strict-mcp-config: the pair
        # means "exactly these servers and nothing inherited from the
        # operator", which is what makes an MCP-attached run reproducible.
        self._mcp_config = Path(mcp_config) if mcp_config is not None else None
        self._mcp_tools = mcp_tools

    def build_command(self) -> list[str]:
        cmd = [
            self._binary,
            "-p",
            PROMPT,
            "--output-format",
            "json",
            "--permission-mode",
            "bypassPermissions",
            "--tools",
            self._tools + ("," + self._mcp_tools if self._mcp_tools else ""),
            "--setting-sources",
            self._setting_sources,
        ]
        if self._mcp_config is not None:
            cmd += ["--mcp-config", str(self._mcp_config)]
        cmd += [
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
        returncode, stdout, stderr, timed_out = run_subprocess(cmd, workspace, env, timeout_s)

        (workspace / ".aedl-agent.stdout").write_text(stdout)
        (workspace / ".aedl-agent.stderr").write_text(stderr)

        usage, extra = _parse_result_json(stdout)
        if usage.model is None:
            usage.model = self._model
        if self._mcp_config is not None:
            import hashlib

            extra["mcp_config_sha256"] = hashlib.sha256(self._mcp_config.read_bytes()).hexdigest()
            extra["mcp_tools"] = self._mcp_tools
        return AgentRunInfo(
            returncode=returncode,
            wall_time_s=time.perf_counter() - start,
            usage=usage,
            command=cmd,
            timed_out=timed_out,
            extra=extra,
        )


def _parse_result_json(stdout: str) -> tuple[AgentUsage, dict[str, Any]]:
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
        for k in (
            "session_id",
            "duration_ms",
            "duration_api_ms",
            "is_error",
            "subtype",
            "stop_reason",
            "permission_denials",
        )
        if k in payload
    }
    model_usage = payload.get("modelUsage")
    if isinstance(model_usage, dict) and model_usage:
        extra["model_usage"] = model_usage
        if usage.model is None:
            usage.model = primary_model(model_usage)
    return usage, extra


def primary_model(model_usage: dict[str, Any]) -> str | None:
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


def _maybe_int(v: object) -> int | None:
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _maybe_float(v: object) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        return None
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
    mcp_config: Path | str | None = None,
    mcp_tools: str = "",
    **_ignored: Any,
) -> ClaudeCliAdapter:
    return ClaudeCliAdapter(
        model=model,
        tools=tools,
        setting_sources=setting_sources,
        max_budget_usd=max_budget_usd,
        binary=binary,
        mcp_config=mcp_config,
        mcp_tools=mcp_tools,
    )
