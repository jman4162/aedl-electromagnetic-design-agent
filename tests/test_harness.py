"""Harness behaviour: workspace hygiene, the four run outcomes, cost accounting."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aedl.harness import get_adapter, instrument, run_task
from aedl.harness import workspace as ws
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
        spec,
        adapter,
        runs_dir=tmp_path / "runs",
        isolation="tmpdir",
        timeout_s=60,
        instrumented=instrumented,
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
    # The brief must not name the technique the reference solution uses.
    for tell in ("coordinate descent", "dither"):
        assert tell not in brief.lower()


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
    _, bundle = _run(spec, tmp_path, _writer(_reference_weights()))
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


# --- provenance of the scoring side ----------------------------------------


def test_manifest_pins_the_code_that_scored_the_run(spec, tmp_path):
    """A task hash says what was asked; this says what did the asking."""
    _, bundle = _run(spec, tmp_path, _writer(_reference_weights()))
    code = json.loads((bundle / "manifest.json").read_text())["code"]
    assert len(code["aedl_git_sha"]) == 40
    assert isinstance(code["aedl_git_dirty"], bool)
    assert code["evaluator"] == spec.evaluator
    assert len(code["evaluator_source_sha256"]) == 64


def test_code_provenance_degrades_outside_a_checkout(tmp_path, monkeypatch):
    """An installed copy still writes a manifest, without inventing a revision."""
    from aedl.harness import record as rec

    monkeypatch.setattr(rec, "_git", lambda *a, **k: None)
    out = rec.code_provenance("array_pattern")
    assert "aedl_git_sha" not in out
    assert out["evaluator"] == "array_pattern"


def test_code_provenance_ignores_an_unrelated_enclosing_repo(monkeypatch):
    """`git rev-parse` searches upwards; a foreign toplevel must not be adopted."""
    from aedl.harness import record as rec

    monkeypatch.setattr(rec, "_git", lambda *a, **k: "/somewhere/else")
    assert "aedl_git_sha" not in rec.code_provenance()


# --- provenance of the environment -----------------------------------------


def test_environment_skew_is_empty_when_both_sides_agree(spec, tmp_path):
    """The probe runs against this machine, so a normal run has nothing to report."""
    record, _ = _run(spec, tmp_path, _writer(_reference_weights()))
    for skew in record.environment_skew:
        assert skew["harness"] != skew["agent"]


def test_environment_skew_maps_import_names_to_distributions():
    """`phased_array` is installed by `phased-array-modeling`; a naive diff misses it."""
    from aedl.harness.record import environment_skew

    env = {
        "python": "3.12.11",
        "dependencies": {"phased-array-modeling": "1.4.0", "numpy": "2.5.2"},
    }
    probe = {"python": "3.12.11", "phased_array": "1.5.0", "numpy": "2.5.2"}
    assert environment_skew(env, probe) == [
        {"package": "phased-array-modeling", "harness": "1.4.0", "agent": "1.5.0"}
    ]


def test_environment_skew_treats_a_missing_package_as_unknown():
    """CI installs neither side of some packages; absent is not skew."""
    from aedl.harness.record import environment_skew

    env = {"python": "3.12.11", "dependencies": {"opensatcom": "not installed"}}
    assert environment_skew(env, {"python": "3.12.11", "opensatcom": None}) == []
    assert environment_skew(env, {"python": "3.12.11", "opensatcom": "0.6.0"}) == []


def test_environment_skew_is_silent_when_the_probe_failed():
    """A failed probe knows nothing about the agent's versions, including that they match."""
    from aedl.harness.record import environment_skew

    env = {"python": "3.12.11", "dependencies": {"numpy": "2.5.2"}}
    assert environment_skew(env, {"error": "TimeoutExpired: ..."}) == []
    assert environment_skew(env, {}) == []


def test_report_shows_skew_only_when_there_is_some():
    from aedl.harness.report import render, skew_table

    clean = [{"run_id": "a", "task_id": "t2-001", "status": "pass"}]
    assert skew_table(clean) == ""
    assert "Environment skew" not in render(clean)

    skewed = [
        {
            "run_id": "b",
            "task_id": "t2-001",
            "status": "pass",
            "environment_skew": [{"package": "numpy", "harness": "2.5.2", "agent": "2.5.1"}],
        }
    ]
    assert "Environment skew" in render(skewed)
    assert "2.5.1" in skew_table(skewed)


def test_report_derives_skew_for_bundles_written_before_the_field_existed():
    """Both halves were always recorded; manifests are not rewritten, so derive at read time."""
    from aedl.harness.report import skew_of

    old = {
        "run_id": "c",
        "environment": {"python": "3.12.11", "dependencies": {"numpy": "2.5.2"}},
        "agent_interpreter": {"python": "3.12.11", "numpy": "2.5.1"},
    }
    assert skew_of(old) == [{"package": "numpy", "harness": "2.5.2", "agent": "2.5.1"}]
    # A recorded empty list is an answer, not a gap: do not re-derive over it.
    assert skew_of({**old, "environment_skew": []}) == []


# --- provenance of the tool spans ------------------------------------------


def test_run_mints_one_traceparent_and_records_its_trace_id(spec, tmp_path):
    """Every MCP server in a run gets the same root, so its spans share one trace."""
    record, bundle = _run(spec, tmp_path, _writer(_reference_weights()))
    assert record.trace_id is not None
    assert len(record.trace_id) == 32
    assert json.loads((bundle / "manifest.json").read_text())["trace_id"] == record.trace_id


def test_traceparent_is_a_valid_w3c_header():
    from aedl.harness.run import new_traceparent

    header, trace_id = new_traceparent()
    version, got_trace, span_id, flags = header.split("-")
    assert (version, flags) == ("00", "01")
    assert got_trace == trace_id
    assert len(trace_id) == 32 and int(trace_id, 16) != 0
    assert len(span_id) == 16 and int(span_id, 16) != 0
    assert new_traceparent()[1] != trace_id


def test_traceparent_reaches_every_mcp_server(tmp_path):
    """APAB opens a fresh root trace per call unless the caller hands one over."""
    from aedl.harness.adapters.claude_cli import inject_server_env

    config = {"mcpServers": {"apab": {"command": "apab"}, "opensatcom": {"command": "osc"}}}
    env = {
        "AEDL_CALL_LOG": str(tmp_path / "calls.jsonl"),
        "TRACEPARENT": "00-" + "a" * 32 + "-" + "b" * 16 + "-01",
    }
    servers = inject_server_env(config, env)["mcpServers"]
    for server in servers.values():
        assert server["env"]["TRACEPARENT"] == env["TRACEPARENT"]
        assert server["env"]["APAB_TRACE_JSONL"].endswith("server-trace.jsonl")


def test_reference_solutions_are_hidden_during_a_run_and_restored(spec, tmp_path):
    """The answer key must not be readable while an agent is working."""

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


# --- Stage 0 regressions ---------------------------------------------------


def test_timeout_is_not_scored_even_if_a_submission_exists(spec, tmp_path):
    """A timed-out agent may leave a half-written or stale file. Scoring it
    would silently admit an incomplete design as a pass."""
    from aedl.harness.adapter import AgentRunInfo, AgentUsage

    weights = _reference_weights()

    class TimedOut:
        name = "mock"

        def run(self, workspace, env, timeout_s):
            np.savez(workspace / ws.SUBMISSION_NAME, weights=weights)
            return AgentRunInfo(
                returncode=124,
                wall_time_s=1.0,
                usage=AgentUsage(model="mock"),
                command=["<mock>"],
                timed_out=True,
            )

    record, _ = run_task(
        spec,
        TimedOut(),
        runs_dir=tmp_path / "runs",
        isolation="tmpdir",
        timeout_s=60,
        instrumented=False,
    )
    assert record.status == "timeout"
    assert record.requirements == []
    assert "exceeded" in record.error


def test_concurrent_runs_do_not_lock_the_reference_directory(spec, tmp_path):
    """A second run must not record 0o000 as the original mode."""
    from aedl.harness.run import hidden_references

    tasks_dir = TASKS
    reference = SOLVE.parent
    original = reference.stat().st_mode & 0o777

    with hidden_references(tasks_dir):
        assert not os_access_ok(reference)
        with hidden_references(tasks_dir):  # concurrent run starts and finishes
            pass
        # The inner context must not have restored anything.
        assert not os_access_ok(reference)
    assert reference.stat().st_mode & 0o777 == original
    assert not (reference.parent / hidden_references.SENTINEL).exists()


def test_recover_restores_a_directory_left_hidden(spec, tmp_path):
    from aedl.harness.run import hidden_references

    reference = SOLVE.parent
    original = reference.stat().st_mode & 0o777
    ctx = hidden_references(TASKS)
    ctx.__enter__()
    ctx._held.clear()  # simulate the process dying before cleanup
    ctx.__exit__()
    assert not os_access_ok(reference)

    restored = hidden_references.recover(TASKS)
    assert reference in restored
    assert reference.stat().st_mode & 0o777 == original


def test_report_survives_a_partial_manifest():
    """Bundles are written in a finally block and can be truncated."""
    from aedl.harness.report import render, runs_table, summary_table

    partial = [{"run_id": "x", "task_id": "t2-001"}]  # no status, usage, agent
    text = render(partial)
    assert "t2-001" in text
    assert "unknown" in runs_table(partial)
    assert "0%" in summary_table(partial)


def test_report_tolerates_a_float_turn_count():
    from aedl.harness.report import runs_table

    rec = {
        "run_id": "x",
        "task_id": "t",
        "agent": "a",
        "status": "pass",
        "usage": {"num_turns": 2.0},
    }
    assert "| 2 |" in runs_table([rec]) or "2.0" in runs_table([rec])


def test_summary_table_of_empty_result():
    from aedl.result import EvaluationResult

    out = EvaluationResult(task_id="t", passed=True, requirements=()).summary_table()
    assert "no requirements" in out


def test_median_cost_is_the_true_median_for_even_samples():
    from aedl.harness.report import summary_table

    recs = [
        {"task_id": "t", "model": "m", "status": "pass", "usage": {"cost_usd": c}}
        for c in (1.0, 3.0)
    ]
    # The upper-middle value (3.000) would be wrong; the median is 2.000.
    assert "| 2.000 |" in summary_table(recs)


def test_cost_column_is_labelled_as_an_estimate():
    """Subscription runs are not billed per token; the figure is an estimate."""
    from aedl.harness.report import runs_table, summary_table

    assert "est. cost USD" in runs_table([])
    assert "median est. cost USD" in summary_table([])


class TestBriefTemplating:
    """The scoring-notes override and filename threading must not disturb t2-001."""

    def test_t2_brief_uses_default_notes_and_filename(self, tmp_path):
        spec = find_task(TASKS, "t2-001")
        brief = ws.render_brief(spec)
        assert "`submission.npz`" in brief
        # The default array notes render verbatim.
        assert "Do not pre-apply failures to your weights" in brief
        assert "`numpy` and the `phased_array` package are installed" in brief

    def test_scoring_notes_override(self, tmp_path):
        import dataclasses

        spec = find_task(TASKS, "t2-001")
        spec = dataclasses.replace(
            spec,
            deliverable={
                **spec.deliverable,
                "filename": "architecture.yaml",
                "scoring_notes": ["Margins are scored worst-case.", "Nothing in prose is scored."],
            },
        )
        brief = ws.render_brief(spec)
        assert "`architecture.yaml`" in brief
        assert "- Margins are scored worst-case." in brief
        assert "Do not pre-apply failures to your weights" not in brief

    def test_submission_name_from_dir(self, tmp_path):
        (tmp_path / "task.yaml").write_text("id: x\ndeliverable:\n  filename: architecture.yaml\n")
        assert ws.submission_name_from_dir(tmp_path) == "architecture.yaml"
        assert ws.submission_name_from_dir(tmp_path / "missing") == "submission.npz"


def test_instrument_counts_dotted_class_method(tmp_path):
    """A "module:Class.method" tier target patches the class object."""
    target_dir = tmp_path / "libs"
    target_dir.mkdir()
    (target_dir / "fakelink.py").write_text(
        "class Engine:\n    def evaluate(self, x):\n        return 2 * x\n"
    )

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    shim_dir = instrument.write_payload(bundle)
    env = instrument.build_env(
        {"PATH": "/usr/bin:/bin"},
        shim_dir,
        bundle / "calls.jsonl",
        tiers={"reduced_order": ["fakelink:Engine.evaluate"]},
    )
    env["PYTHONPATH"] = f"{shim_dir}:{target_dir}"

    code = "from fakelink import Engine\nassert Engine().evaluate(3) == 6\n"
    subprocess.run([sys.executable, "-c", code], env=env, check=True)

    summary = instrument.summarize(bundle / "calls.jsonl")
    assert summary["total_calls"] == 1
    assert summary["calls_by_tier"] == {"reduced_order": 1}
    assert "fakelink:Engine.evaluate" in summary["calls_by_function"]


class TestShimEnvPaths:
    def test_env_paths_are_absolute(self, tmp_path, monkeypatch):
        """Relative shim paths silently resolve nowhere in the agent process
        (cwd = workspace) and in MCP server subprocesses (cwd = launcher's
        choice) — the reason every early bundle recorded zero calls."""
        from aedl.harness import instrument

        monkeypatch.chdir(tmp_path)
        shim = instrument.write_payload(Path("bundle/.shim"))
        env = instrument.build_env({}, shim, Path("bundle/calls.jsonl"))
        assert Path(env["AEDL_CALL_LOG"]).is_absolute()
        assert Path(env["PYTHONPATH"].split(":")[0]).is_absolute()
