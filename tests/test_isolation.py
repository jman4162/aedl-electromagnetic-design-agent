"""Security properties of the run environment.

These guard the two findings an external review rated most serious: host
credentials reaching the agent, and descendants surviving a timeout.
"""

import os
import subprocess
import sys
import time

import pytest

from aedl.harness import get_adapter
from aedl.harness.adapter import run_subprocess
from aedl.harness.run import BASE_ENV_ALLOWLIST, build_child_env

SECRETS = [
    "SSH_AUTH_SOCK",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "OPENAI_API_KEY",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "NPM_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
]


@pytest.fixture
def planted(monkeypatch):
    for name in SECRETS:
        monkeypatch.setenv(name, f"synthetic-{name}")
    return SECRETS


def test_no_host_secret_reaches_the_agent(planted):
    env = build_child_env(get_adapter("command", template="true"))
    leaked = [n for n in planted if n in env]
    assert leaked == [], f"these reached the agent: {leaked}"


def test_ssh_agent_socket_is_never_forwarded(planted):
    """A live SSH_AUTH_SOCK would let the agent authenticate as the operator."""
    for name in ("claude", "command", "mock"):
        adapter = get_adapter(name, template="true")
        assert "SSH_AUTH_SOCK" not in build_child_env(adapter)


def test_base_allowlist_is_enough_to_run_a_process(planted):
    env = build_child_env(get_adapter("mock"))
    assert "PATH" in env and "HOME" in env
    proc = subprocess.run(
        [sys.executable, "-c", "print('ok')"], env=env, capture_output=True, text=True
    )
    assert proc.stdout.strip() == "ok"


def test_adapter_declared_credentials_pass_through(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-key")
    assert "ANTHROPIC_API_KEY" in build_child_env(get_adapter("claude"))
    # ...but only for the adapter that declared it.
    assert "ANTHROPIC_API_KEY" not in build_child_env(get_adapter("command", template="true"))


def test_allowlist_carries_no_credential_shaped_names():
    for name in BASE_ENV_ALLOWLIST:
        assert not any(t in name for t in ("TOKEN", "KEY", "SECRET", "PASSWORD", "AUTH"))


def test_timeout_kills_the_whole_process_tree(tmp_path):
    """subprocess.run(timeout=) leaves grandchildren alive; run_subprocess must not."""
    marker = tmp_path / "grandchild.pid"
    script = tmp_path / "spawn.py"
    script.write_text(
        "import subprocess, sys, time, pathlib\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(child.pid))\n"
        "time.sleep(120)\n"
    )
    env = {"PATH": os.environ.get("PATH", "")}
    rc, _, stderr, timed_out = run_subprocess(
        [sys.executable, str(script)], tmp_path, env, timeout_s=3
    )
    assert timed_out and rc == 124 and "timed out" in stderr

    pid = int(marker.read_text())
    for _ in range(50):  # give the signal a moment to land
        if not _alive(pid):
            break
        time.sleep(0.1)
    assert not _alive(pid), f"grandchild {pid} survived the timeout"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
