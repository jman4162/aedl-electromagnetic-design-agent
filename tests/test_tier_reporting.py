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


def _based(task: str, status: str, baseline: str | None, *, where: str = "extra") -> dict:
    rec = _rec(task, status, None)
    if baseline is not None and where == "extra":
        rec["usage"] = {"extra": {"baseline": baseline}}
    elif baseline is not None:
        rec["baseline"] = baseline
    return rec


def test_summary_groups_by_baseline_as_well_as_task_and_model():
    """EdgeArray's b0..b4 on the same task and model are different conditions;
    pooling them into one pass rate hides the ladder."""
    from aedl.harness.report import summary_table

    recs = [
        _based("t1-001", "pass", "b0"),
        _based("t1-001", "fail", "b2"),
        _based("t1-001", "fail", "b2"),
    ]
    lines = summary_table(recs).splitlines()
    assert "baseline" in lines[0]
    body = lines[2:]
    assert len(body) == 2
    b0 = next(ln for ln in body if "| b0 |" in ln)
    b2 = next(ln for ln in body if "| b2 |" in ln)
    assert "| 1 | 1 | 100% |" in b0
    assert "| 2 | 0 | 0% |" in b2


def test_runs_table_has_a_baseline_column():
    from aedl.harness.report import runs_table

    table = runs_table([_based("t1-001", "pass", "b3"), _based("t1-001", "pass", None)])
    lines = table.splitlines()
    assert "baseline" in lines[0]
    assert lines[1].count("|") == lines[0].count("|")
    assert "| b3 |" in lines[2]
    assert "| — |" in lines[3]


def test_baseline_is_read_from_the_record_when_usage_has_none():
    from aedl.harness.report import baseline_of

    assert baseline_of(_based("t", "pass", "b1", where="record")) == "b1"
    assert baseline_of(_based("t", "pass", "b4")) == "b4"
    assert baseline_of(_based("t", "pass", None)) is None


def test_report_without_baselines_keeps_its_columns():
    from aedl.harness.report import runs_table, summary_table

    recs = [_rec("t2-001", "pass", None)]
    assert "baseline" not in summary_table(recs)
    assert "baseline" not in runs_table(recs)
