"""Harness behaviour: workspace hygiene, the four run outcomes, cost accounting."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aedl.harness import get_adapter, run_task
from aedl.harness import instrument, workspace as ws
from aedl.harness.report import load_runs, render
from aedl.spec import find_task

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"
SOLVE = TASKS / "t2-001-steered-beam-2bit" / "reference" / "solve.py"


@pytest.fixture(scope="module")
def spec():
    return find_task(TASKS, "t2-001")


def _reference_weights():
    return runpy.run_path(str(SOLVE))["solve"]()


def _writer(weights):
    def behave(workspace: Path):
        np.savez(workspace / ws.SUBMISSION_NAME, weights=weights)

    return behave


def _run(spec, tmp_path, behavior=None, returncode=0, instrumented=False):
    adapter = get_adapter("mock", behavior=behavior, returncode=returncode)
    return run_task(
        spec, adapter, runs_dir=tmp_path / "runs",
        isolation="tmpdir", timeout_s=60, instrumented=instrumented,
    )


# --- workspace -------------------------------------------------------------

def test_workspace_has_brief_and_task_but_no_reference(spec, tmp_path):
    work = ws.materialize(spec, isolation="inplace", parent=tmp_path)
    names = {p.name for p in work.iterdir()}
    assert names == {"task.yaml", "BRIEF.md"}
    assert not (work / "reference").exists()


def test_brief_states_requirements_but_not_the_reference_technique(spec, tmp_path):
    brief = ws.render_brief(spec)
    for req in spec.requirements:
        assert req.id in brief
        assert req.metric in brief
    assert "submission.npz" in brief
    assert "dither" not in brief.lower()


# --- run outcomes ----------------------------------------------------------

def test_reference_submission_passes(spec, tmp_path):
    record, bundle = _run(spec, tmp_path, _writer(_reference_weights()))
    assert record.status == "pass"
    assert (bundle / "result.json").exists()
    assert all(r["passed"] for r in record.requirements)


def test_wrong_submission_fails_on_steering(spec, tmp_path):
    record, _ = _run(spec, tmp_path, _writer(np.ones(256, dtype=complex)))
    assert record.status == "fail"
    failed = {r["requirement_id"] for r in record.requirements if not r["passed"]}
    assert "steering" in failed


def test_missing_submission_is_recorded_not_raised(spec, tmp_path):
    record, bundle = _run(spec, tmp_path, behavior=None)
    assert record.status == "no_submission"
    assert "submission.npz" in record.error
    assert (bundle / "manifest.json").exists()


def test_malformed_submission_is_evaluator_error(spec, tmp_path):
    record, _ = _run(spec, tmp_path, _writer(np.ones(7, dtype=complex)))
    assert record.status == "evaluator_error"
    assert "shape" in record.error


def test_agent_exception_still_writes_bundle(spec, tmp_path):
    def explode(workspace):
        raise RuntimeError("adapter blew up")

    record, bundle = _run(spec, tmp_path, explode)
    assert record.status == "agent_error"
    assert "adapter blew up" in record.error
    assert (bundle / "manifest.json").exists()


def test_manifest_pins_task_hash_and_dependencies(spec, tmp_path):
    record, bundle = _run(spec, tmp_path, _writer(_reference_weights()))
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["task_sha256"] == ws.task_digest(spec)
    assert len(manifest["task_sha256"]) == 64
    deps = manifest["environment"]["dependencies"]
    assert deps["phased-array-modeling"].startswith("1.")
    assert "numpy" in deps


# --- cost accounting -------------------------------------------------------

def test_instrumentation_counts_calls_by_tier(tmp_path):
    shim = instrument.write_payload(tmp_path / "shim")
    log = tmp_path / "calls.jsonl"
    script = tmp_path / "probe.py"
    script.write_text(
        "import phased_array as pa\n"
        "wl = 299792458.0 / 1e10\n"
        "k = pa.wavelength_to_k(wl)\n"
        "g = pa.create_rectangular_array(4, 4, dx=0.5, dy=0.5, wavelength=wl)\n"
        "w = pa.steering_vector(k, g.x, g.y, 0.0, 0.0)\n"
        "pa.compute_full_pattern(g.x, g.y, w, k, n_theta=9, n_phi=9)\n"
        "pa.compute_full_pattern(g.x, g.y, w, k, n_theta=9, n_phi=9)\n"
    )
    env = instrument.build_env({"PATH": "/usr/bin:/bin"}, shim, log)
    subprocess.run([sys.executable, str(script)], env=env, check=True, capture_output=True)

    summary = instrument.summarize(log)
    assert summary["calls_by_function"]["phased_array:compute_full_pattern"] == 2
    assert summary["calls_by_tier"]["aperture_model"] == 2
    # array_factor_vectorized is called inside compute_full_pattern; it must not
    # inflate the headline count.
    assert summary["nested_calls"] > 0
    assert summary["total_calls"] == 2


def test_summarize_handles_missing_log(tmp_path):
    assert instrument.summarize(tmp_path / "absent.jsonl")["total_calls"] == 0


# --- reporting -------------------------------------------------------------

def test_report_aggregates_multiple_runs(spec, tmp_path):
    runs = tmp_path / "runs"
    _run(spec, tmp_path, _writer(_reference_weights()))
    _run(spec, tmp_path, _writer(np.ones(256, dtype=complex)))
    records = load_runs(runs)
    assert len(records) == 2

    text = render(records)
    assert "t2-001" in text
    assert "| 2 | 1 | 50% |" in text  # attempts, passed, pass rate
    assert "steering" in text  # the failing requirement is named


def test_report_with_no_runs(tmp_path):
    assert render(load_runs(tmp_path)) == "No runs found."


def test_agent_interpreter_is_probed_and_recorded(spec, tmp_path):
    """The agent's python need not be the harness venv; record what it actually gets."""
    record, _ = _run(spec, tmp_path, _writer(_reference_weights()))
    probe = record.agent_interpreter
    assert "error" not in probe, probe
    assert probe["executable"]
    assert probe["numpy"], "agent interpreter must have numpy; BRIEF.md promises it"
    assert probe["phased_array"], "BRIEF.md promises phased_array is importable"


def test_reference_solutions_are_hidden_during_a_run_and_restored(spec, tmp_path):
    """The answer key must not be readable while an agent is working."""
    from aedl.harness.run import hidden_references

    reference = SOLVE.parent
    before = reference.stat().st_mode & 0o777
    seen = {}
    # Solve before the run: solve.py lives in the directory about to be hidden.
    weights = _reference_weights()

    def peek(workspace: Path):
        seen["readable"] = os_access_ok(reference)
        np.savez(workspace / ws.SUBMISSION_NAME, weights=weights)

    record, _ = _run(spec, tmp_path, peek)
    assert seen["readable"] is False, "reference stayed readable during the run"
    assert reference.stat().st_mode & 0o777 == before, "mode not restored"
    assert record.status == "pass"


def os_access_ok(path: Path) -> bool:
    import os

    return os.access(path, os.R_OK | os.X_OK)
