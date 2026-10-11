"""A screening-tier verdict stays labelled wherever verdicts are read.

t1-001's analytic tier cannot certify band coverage. The label lives in the
evaluator's info block, so the run record, the leaderboard, and the ladder
table must carry it forward or a reader sees a bare "pass".
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from aedl.harness import get_adapter, run_task
from aedl.harness.record import screening_tier
from aedl.harness.report import render
from aedl.spec import find_task

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"


def test_screening_tier_from_info():
    assert screening_tier({"certifies_band_coverage": False}) == "analytic-screening"
    assert screening_tier({"certifies_band_coverage": True}) is None
    assert screening_tier({}) is None


def test_t1_run_record_carries_the_tier(tmp_path):
    pytest.importorskip("antenna_cad", reason="requires the 'element' extra (antenna-cad)")
    spec = find_task(TASKS, "t1-001")
    reference = (TASKS / "t1-001-patch-28ghz-ro4350b" / "reference" / "design.yaml").read_text()

    def behave(workspace: Path) -> None:
        (workspace / "design.yaml").write_text(reference)

    record, bundle = run_task(
        spec,
        get_adapter("mock", behavior=behave),
        runs_dir=tmp_path / "runs",
        timeout_s=60,
        instrumented=False,
    )
    assert record.status == "pass"
    assert record.tier == "analytic-screening"
    assert json.loads((bundle / "manifest.json").read_text())["tier"] == "analytic-screening"
    shutil.rmtree(tmp_path / "runs")


def _rec(task: str, status: str, tier: str | None) -> dict:
    return {
        "run_id": f"r-{task}-{status}",
        "task_id": task,
        "agent": "mock",
        "model": "m",
        "status": status,
        "requirements": [],
        "usage": {},
        "tier": tier,
    }


def test_report_shows_a_tier_column_only_when_a_run_has_one():
    plain = render([_rec("t2-001", "pass", None)])
    assert "tier" not in plain.split("## Runs")[1].splitlines()[2]

    mixed = render([_rec("t2-001", "pass", None), _rec("t1-001", "pass", "analytic-screening")])
    summary, runs = mixed.split("## Runs")
    assert "| tier |" in summary and "| tier |" in runs
    assert "analytic-screening" in summary and "analytic-screening" in runs


def test_ladder_table_shows_tier_and_recomputed_passes():
    import runpy

    ladder = runpy.run_path(str(REPO / "scripts" / "run_ladder.py"))
    rows = [
        {
            "task": "t1-001",
            "baseline": "b2",
            "model": "m",
            "status": "pass",
            "tokens": 5,
            "cost_usd": 0.1,
            "tier": "analytic-screening",
            "recomputed_passes": None,
        },
        {
            "task": "t4-001",
            "baseline": "b2",
            "model": "m",
            "status": "pass",
            "tokens": 5,
            "cost_usd": 0.1,
            "tier": None,
            "recomputed_passes": 9,
        },
    ]
    table = ladder["ladder_table"](rows)
    header = table.splitlines()[0]
    assert "tier" in header and "recomputed passes" in header
    assert "analytic-screening" in table
    assert "| 9.0 |" in table
