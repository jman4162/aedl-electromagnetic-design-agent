"""Evaluator sanity for t3-001: the reference passes; naive approaches fail
for the right reasons; the held-out envelope binds."""

from __future__ import annotations

import dataclasses
import runpy
from pathlib import Path

import pytest
import yaml

from aedl import get_evaluator
from aedl.spec import find_task

pytest.importorskip("phased_array_systems", reason="requires the 'systems' extra")
pytest.importorskip("opensatcom", reason="requires the 'systems' extra")

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"
SOLVE = TASKS / "t3-001-satcom-terminal-28ghz" / "reference" / "solve.py"


@pytest.fixture(scope="module")
def spec():
    return find_task(TASKS, "t3-001")


@pytest.fixture(scope="module")
def solve_mod():
    # Load before any harness run hides the reference directory.
    return runpy.run_path(str(SOLVE))


@pytest.fixture(scope="module")
def reference_arch(solve_mod):
    return solve_mod["solve"]()


@pytest.fixture(scope="module")
def evaluate(spec):
    evaluator = get_evaluator(spec.evaluator)

    def run(arch_dict, tmpdir, task_spec=None):
        path = Path(tmpdir) / "architecture.yaml"
        path.write_text(yaml.safe_dump(arch_dict, sort_keys=False))
        return evaluator(task_spec if task_spec is not None else spec, path)

    return run


@pytest.fixture(scope="module")
def reference_result(evaluate, reference_arch, tmp_path_factory):
    return evaluate(reference_arch, tmp_path_factory.mktemp("ref"))


def _failures(result):
    return {r.requirement_id for r in result.requirements if not r.passed}


def _metric(result, name):
    return next(r.value for r in result.requirements if r.requirement_id == name)


def _tweak(arch, **overrides):
    doc = {k: dict(v) for k, v in arch.items()}
    for dotted, value in overrides.items():
        section, key = dotted.split("__")
        doc[section][key] = value
    return doc


def test_reference_passes(reference_result):
    assert reference_result.passed, reference_result.summary_table()


def test_reference_crosscheck_agreement(reference_result):
    """Two independent codebases agree on the same design well inside tolerance."""
    metrics = reference_result.info["metrics"]
    assert metrics["crosscheck_gain_disagreement_db"] < 0.3
    assert metrics["crosscheck_clearsky_margin_disagreement_db"] < 0.6


def test_envelope_binds_for_reference(reference_result):
    """The held-out envelope, not the nominal scenario, sets the margin."""
    nominal = reference_result.info["nominal_margin_db"]
    worst = _metric(reference_result, "link-margin")
    assert worst < nominal - 3.0, (worst, nominal)
    assert reference_result.info["binding"]["point"] is not None


def test_determinism(evaluate, reference_arch, tmp_path_factory):
    r1 = evaluate(reference_arch, tmp_path_factory.mktemp("d1"))
    r2 = evaluate(reference_arch, tmp_path_factory.mktemp("d2"))
    assert {r.requirement_id: r.value for r in r1.requirements} == {
        r.requirement_id: r.value for r in r2.requirements
    }


def test_seed_order_invariance(evaluate, reference_arch, spec, tmp_path_factory):
    params = dict(spec.evaluator_params)
    params["envelope"] = dict(params["envelope"])
    params["envelope"]["failure_seeds"] = list(reversed(params["envelope"]["failure_seeds"]))
    reversed_spec = dataclasses.replace(spec, evaluator_params=params)

    a = evaluate(reference_arch, tmp_path_factory.mktemp("s1"))
    b = evaluate(reference_arch, tmp_path_factory.mktemp("s2"), task_spec=reversed_spec)
    assert {r.requirement_id: r.value for r in a.requirements} == {
        r.requirement_id: r.value for r in b.requirements
    }


def test_brute_force_aperture_fails_ceilings_and_sidelobes(evaluate, tmp_path_factory):
    """Max aperture, uniform taper, cheap parts: the classic brute force."""
    brute = {
        "array": {
            "nx": 32,
            "ny": 32,
            "dx_lambda": 0.5,
            "dy_lambda": 0.5,
            "taper_type": "uniform",
            "taper_sll_db": -13.0,
            "phase_bits": 4,
        },
        "rf": {"tx_power_w_per_elem": 0.05, "pa_class": "A"},
        "digital": {
            "digitization_level": "analog",
            "adc_enob": 8.0,
            "subarray_nx": 8,
            "subarray_ny": 8,
        },
    }
    result = evaluate(brute, tmp_path_factory.mktemp("brute"))
    assert not result.passed
    failures = _failures(result)
    assert "unit-cost" in failures, failures
    assert "sidelobes" in failures, failures


def test_brute_force_power_fails_prime_power(evaluate, reference_arch, tmp_path_factory):
    """Small aperture at maximum per-element power blows the power ceiling."""
    hot = _tweak(reference_arch, rf__tx_power_w_per_elem=2.0)
    result = evaluate(hot, tmp_path_factory.mktemp("hot"))
    assert not result.passed
    assert "prime-power" in _failures(result)


def test_nominal_only_design_fails_the_envelope(
    evaluate, solve_mod, reference_arch, spec, tmp_path_factory
):
    """A design power-tuned to the nominal scenario passes nominal, fails held-out."""
    task = solve_mod["_load_task"]()
    nominal = task["context"]["nominal_scenario"]
    nominal_point = {
        "scan_deg": nominal["scan_angle_deg"],
        "elevation_deg": nominal["elevation_deg"],
        "range_km": nominal["slant_range_km"],
        "rain_mmh": nominal["rain_rate_mmh"],
        "sky_temp_k": nominal["sky_noise_temp_k"],
    }
    design = {
        "n": reference_arch["array"]["nx"],
        "taper_type": reference_arch["array"]["taper_type"],
        "taper_sll_db": reference_arch["array"]["taper_sll_db"],
        "phase_bits": reference_arch["array"]["phase_bits"],
        "pa_class": reference_arch["rf"]["pa_class"],
        "digitization_level": reference_arch["digital"]["digitization_level"],
        "adc_enob": reference_arch["digital"]["adc_enob"],
    }
    lo, hi = 1e-3, 2.0
    for _ in range(16):
        mid = (lo + hi) / 2
        arch = solve_mod["_build_arch"](task, design, mid)
        if solve_mod["_margin_at"](task, arch, nominal_point, None) >= 0.3:
            hi = mid
        else:
            lo = mid

    doc = _tweak(reference_arch, rf__tx_power_w_per_elem=round(hi, 4))
    result = evaluate(doc, tmp_path_factory.mktemp("nom"))
    assert result.info["nominal_margin_db"] > 0.0
    assert not result.passed
    assert "link-margin" in _failures(result)


def test_wide_spacing_fails_grating(evaluate, reference_arch, tmp_path_factory):
    wide = _tweak(reference_arch, array__dx_lambda=0.7, array__dy_lambda=0.7)
    result = evaluate(wide, tmp_path_factory.mktemp("wide"))
    assert not result.passed
    assert "grating" in _failures(result)


def test_pinned_fields_rejected(evaluate, reference_arch, tmp_path_factory):
    for section, key in (("rf", "pa_efficiency"), ("rf", "noise_figure_db"), ("array", "cost")):
        doc = {k: dict(v) for k, v in reference_arch.items()}
        doc[section][key] = 0.99
        with pytest.raises(ValueError, match="parts table"):
            evaluate(doc, tmp_path_factory.mktemp(f"pin-{key}"))


def test_unknown_and_missing_keys_rejected(evaluate, reference_arch, tmp_path_factory):
    doc = {k: dict(v) for k, v in reference_arch.items()}
    doc["array"]["mystery"] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        evaluate(doc, tmp_path_factory.mktemp("unknown"))

    doc = {k: dict(v) for k, v in reference_arch.items()}
    del doc["rf"]["pa_class"]
    with pytest.raises(ValueError, match="missing keys"):
        evaluate(doc, tmp_path_factory.mktemp("missing"))


def test_bad_subarray_tiling_rejected(evaluate, reference_arch, tmp_path_factory):
    doc = _tweak(reference_arch, digital__subarray_nx=5, digital__subarray_ny=5)
    with pytest.raises(ValueError):
        evaluate(doc, tmp_path_factory.mktemp("tiling"))


def test_brief_declares_bounds_but_not_the_envelope(spec):
    from aedl.harness import workspace as ws

    brief = ws.render_brief(spec)
    assert "`architecture.yaml`" in brief
    assert "worst-case" in brief
    assert "opensatcom" in brief
    # The held-out points and seeds must not leak.
    for seed in spec.evaluator_params["envelope"]["failure_seeds"]:
        assert str(seed) not in brief
    assert "failure_seeds" not in brief
    assert "points:" not in brief
    # And neither must the reference's technique.
    assert "bisect" not in brief.lower()
    assert "coordinate descent" not in brief.lower()


def test_harness_end_to_end_with_mock(spec, reference_arch, tmp_path):
    from aedl.harness import get_adapter, run_task

    def writes_architecture(workspace: Path) -> None:
        (workspace / "architecture.yaml").write_text(
            yaml.safe_dump(reference_arch, sort_keys=False)
        )

    record, _ = run_task(
        spec,
        get_adapter("mock", behavior=writes_architecture),
        runs_dir=tmp_path / "runs",
        isolation="tmpdir",
        timeout_s=600,
        instrumented=False,
    )
    assert record.status == "pass", record.error


def test_harness_enforces_the_task_filename(spec, tmp_path):
    """Writing the t2 filename instead of architecture.yaml is no submission."""
    from aedl.harness import get_adapter, run_task

    def writes_wrong_file(workspace: Path) -> None:
        (workspace / "submission.npz").write_text("not even npz")

    record, _ = run_task(
        spec,
        get_adapter("mock", behavior=writes_wrong_file),
        runs_dir=tmp_path / "runs",
        isolation="tmpdir",
        timeout_s=600,
        instrumented=False,
    )
    assert record.status == "no_submission"
    assert "architecture.yaml" in (record.error or "")
