"""Parsing of the Claude CLI result JSON."""

import json

from aedl.harness.adapters.claude_cli import (
    ClaudeCliAdapter,
    _parse_result_json,
    primary_model,
)

# Shape observed from `claude -p --output-format json` (2.1.226): no top-level
# "model" key, and modelUsage lists the small internal model alongside the one
# actually under test.
SAMPLE = {
    "num_turns": 42,
    "total_cost_usd": 1.8329703,
    "usage": {
        "input_tokens": 84,
        "output_tokens": 45925,
        "cache_read_input_tokens": 2220851,
        "cache_creation_input_tokens": 79598,
    },
    "modelUsage": {
        "claude-haiku-4-5-20251001": {"outputTokens": 16, "costUSD": 0.000635},
        "claude-sonnet-5": {"outputTokens": 45925, "costUSD": 1.83297},
    },
    "session_id": "abc123",
    "is_error": False,
}


def test_primary_model_is_the_one_doing_the_work():
    assert primary_model(SAMPLE["modelUsage"]) == "claude-sonnet-5"


def test_primary_model_handles_empty():
    assert primary_model({}) is None


def test_parse_extracts_usage_including_cache_tokens():
    usage, extra = _parse_result_json(json.dumps(SAMPLE))
    assert usage.model == "claude-sonnet-5"
    assert usage.output_tokens == 45925
    assert usage.cache_read_tokens == 2220851
    assert usage.cache_creation_tokens == 79598
    assert usage.num_turns == 42
    assert usage.cost_usd == 1.8329703
    assert extra["session_id"] == "abc123"


def test_parse_survives_non_json_output():
    usage, extra = _parse_result_json("claude: command failed\n")
    assert usage.model is None and usage.cost_usd is None
    assert "parse_error" in extra


def test_command_pins_config_isolation_and_model():
    cmd = ClaudeCliAdapter(model="sonnet").build_command()
    assert cmd[:2] == ["claude", "-p"]
    for flag in (
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--tools",
        "--setting-sources",
        "--output-format",
    ):
        assert flag in cmd, f"{flag} missing: a run would inherit operator config"
    assert cmd[cmd.index("--model") + 1] == "sonnet"
    assert cmd[cmd.index("--permission-mode") + 1] == "bypassPermissions"


class TestMcpConfig:
    def test_unset_is_byte_identical_to_default(self):
        """No MCP config means the exact command shipped today."""
        assert (
            ClaudeCliAdapter().build_command()
            == ClaudeCliAdapter(mcp_config=None, mcp_tools="").build_command()
        )

    def test_mcp_config_composes_with_strict(self, tmp_path):
        config = tmp_path / "servers.json"
        config.write_text('{"mcpServers": {}}')
        cmd = ClaudeCliAdapter(
            mcp_config=config, mcp_tools="mcp__opensatcom__link_snapshot"
        ).build_command()

        assert "--mcp-config" in cmd
        assert cmd[cmd.index("--mcp-config") + 1] == str(config)
        # --strict-mcp-config stays: explicit config only, nothing inherited.
        assert "--strict-mcp-config" in cmd
        assert cmd.index("--mcp-config") < cmd.index("--strict-mcp-config")
        tools = cmd[cmd.index("--tools") + 1]
        assert tools.endswith(",mcp__opensatcom__link_snapshot")

    def test_config_hash_recorded(self, tmp_path, monkeypatch):
        import hashlib

        config = tmp_path / "servers.json"
        config.write_text('{"mcpServers": {"opensatcom": {"command": "opensatcom"}}}')
        adapter = ClaudeCliAdapter(mcp_config=config, mcp_tools="mcp__opensatcom__link_snapshot")

        def fake_run(cmd, workspace, env, timeout_s):
            return 0, '{"result": "done", "usage": {}}', "", False

        monkeypatch.setattr("aedl.harness.adapters.claude_cli.run_subprocess", fake_run)
        info = adapter.run(tmp_path, {}, 60)
        assert info.extra["mcp_config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
        assert info.extra["mcp_tools"] == "mcp__opensatcom__link_snapshot"
