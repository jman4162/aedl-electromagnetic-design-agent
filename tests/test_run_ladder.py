"""scripts/run_ladder.py: baselines x repeats through the EdgeArray adapter."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from aedl.harness.adapters.edgearray import ENV_BIN
from test_edgearray_adapter import T1_REFERENCE, make_fake, make_project

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_ladder.py"


@pytest.fixture(scope="module")
def ladder():
    return runpy.run_path(str(SCRIPT))


@pytest.fixture()
def fake_bin(tmp_path, monkeypatch):
    project = make_project(tmp_path / "project", [(T1_REFERENCE, "2026-10-01T00:00:00Z")], head=0)
    monkeypatch.setenv(ENV_BIN, str(make_fake(tmp_path, project)))


def _argv(tmp_path, *extra):
    return [
        "--task",
        "t1-001",
        "--model",
        "ollama/qwen3.5:9b",
        "--baselines",
        "b0,b2",
        "--repeats",
        "2",
        "--tasks-dir",
        str(REPO / "tasks"),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--leaderboard",
        str(tmp_path / "leaderboard.md"),
        "--no-instrument",
        *extra,
    ]


def test_runs_every_baseline_and_repeat(ladder, fake_bin, tmp_path, capsys):
    assert ladder["main"](_argv(tmp_path)) == 0
    assert len(list((tmp_path / "runs").glob("*/manifest.json"))) == 4
    out = capsys.readouterr().out
    assert "| t1-001 | b0 | - | 2 | 100% | analytic-screening | 0 | 0.0000 |" in out
    assert (
        "| t1-001 | b2 | ollama/qwen3.5:9b | 2 | 100% | analytic-screening | 1,540 | 0.0125 |"
        in out
    )
    assert not (tmp_path / "leaderboard.md").exists()


def test_write_leaderboard_appends(ladder, fake_bin, tmp_path):
    board = tmp_path / "leaderboard.md"
    board.write_text("# existing\n")
    assert ladder["main"](_argv(tmp_path, "--write-leaderboard")) == 0
    text = board.read_text()
    assert text.startswith("# existing\n")
    assert "EdgeArray baseline ladder" in text
    assert (
        "| task | baseline | model | runs | pass rate | tier | mean tokens | mean cost USD |"
        in text
    )
    assert "Approvals: all rejected" in text


def test_table_handles_runs_without_usage(ladder):
    rows = [
        {
            "task": "t1-001",
            "baseline": "b1",
            "model": "m",
            "status": "fail",
            "tokens": None,
            "cost_usd": None,
        },
        {
            "task": "t1-001",
            "baseline": "b1",
            "model": "m",
            "status": "pass",
            "tokens": 10,
            "cost_usd": None,
        },
    ]
    table = ladder["ladder_table"](rows)
    assert "| t1-001 | b1 | m | 2 | 50% | 10 | — |" in table


def test_unknown_baseline_is_an_argument_error(ladder, tmp_path):
    with pytest.raises(SystemExit):
        ladder["main"](["--task", "t1-001", "--model", "m", "--baselines", "b7"])


def test_approval_and_budget_reach_every_baseline(ladder, fake_bin, tmp_path, capsys):
    import json

    argv = _argv(tmp_path, "--approve-all", "--budget-s", "300")
    assert ladder["main"](argv) == 0
    assert "Approvals: all approved (--approve-all). EdgeArray budget per attempt: 300 s." in (
        capsys.readouterr().out
    )
    seen = [
        json.loads(p.read_text())
        for p in (tmp_path / "runs").glob("*/workspace/edgearray/argv.json")
    ]
    assert len(seen) == 4
    for args in seen:
        assert "--approve-all" in args
        assert args[args.index("--budget-s") + 1] == "300"
        # b0 runs EdgeArray's rules without a model.
        assert ("--model" in args) == (args[args.index("--baseline") + 1] != "b0")


def test_every_attempt_has_its_own_data_dir_and_project(ladder, fake_bin, tmp_path):
    """Baselines and repeats never share EdgeArray state, so one attempt's
    design history cannot reach the next."""
    import json

    assert ladder["main"](_argv(tmp_path)) == 0
    seen = [
        json.loads(p.read_text())
        for p in (tmp_path / "runs").glob("*/workspace/edgearray/argv.json")
    ]
    assert len(seen) == 4
    assert len({args[args.index("--data-dir") + 1] for args in seen}) == 4
    assert len({args[args.index("--project") + 1] for args in seen}) == 4
