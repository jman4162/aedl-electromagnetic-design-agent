"""Evaluator sanity for t3-002: the reference passes; naive approaches fail
for the right reasons; the held-out envelope binds."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import pytest
import yaml

from aedl import get_evaluator
from aedl.spec import find_task

pytest.importorskip("phased_array_systems", reason="requires the 'systems' extra")

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"
REFERENCE = TASKS / "t3-002-xband-search-radar" / "reference" / "architecture.yaml"


@pytest.fixture(scope="module")
def spec():
    return find_task(TASKS, "t3-002")


@pytest.fixture(scope="module")
def reference_arch():
    # The committed architecture (written by reference/solve.py). Loaded
    # before any harness run hides the reference directory.
    return yaml.safe_load(REFERENCE.read_text())


@pytest.fixture(scope="module")
def evaluate(spec):
    evaluator = get_evaluator(spec.evaluator)

    def run(arch_dict, tmpdir, task_spec=None):
        path = Path(tmpdir) / "architecture.yaml"
        path.write_text(yaml.safe_dump(arch_dict))
        return evaluator(task_spec or spec, path)

    return run


@pytest.fixture(scope="module")
def reference_result(evaluate, reference_arch, tmp_path_factory):
    return evaluate(reference_arch, tmp_path_factory.mktemp("ref"))


def _failures(result):
    return [r.requirement_id for r in result.requirements if not r.passed]


def _metric(result, name):
    return result.info["metrics"][name]


def _tweak(arch, **overrides):
    doc = copy.deepcopy(arch)
    for dotted, value in overrides.items():
        section, key = dotted.split("__")
        doc[section][key] = value
    return doc


def test_reference_passes(reference_result):
    assert reference_result.passed, _failures(reference_result)


def test_reference_crosschecks_agree(reference_result):
    assert _metric(reference_result, "crosscheck_gain_disagreement_db") <= 0.5
    assert _metric(reference_result, "crosscheck_pd_disagreement") <= 0.12


def test_envelope_binds_for_reference(reference_result):
    """The worst-case Pd must come from the held-out envelope, strictly
    below what the nominal point alone would suggest — otherwise the
    envelope is decoration."""
    per_point = reference_result.info["per_point"]
    pds = [row["pd"] for row in per_point]
    assert min(pds) < max(pds) - 0.02, "envelope points are indistinguishable"
    assert reference_result.info["binding"]["point"] is not None


def test_determinism(evaluate, reference_arch, tmp_path_factory):
    a = evaluate(reference_arch, tmp_path_factory.mktemp("d1"))
    b = evaluate(reference_arch, tmp_path_factory.mktemp("d2"))
    assert a.info["metrics"] == b.info["metrics"]


def test_seed_order_invariance(evaluate, reference_arch, spec, tmp_path_factory):
    shuffled = dataclasses.replace(
        spec,
        evaluator_params={
            **spec.evaluator_params,
            "envelope": {
                **spec.evaluator_params["envelope"],
                "failure_seeds": list(reversed(spec.evaluator_params["envelope"]["failure_seeds"])),
            },
        },
    )
    a = evaluate(reference_arch, tmp_path_factory.mktemp("s1"))
    b = evaluate(reference_arch, tmp_path_factory.mktemp("s2"), task_spec=shuffled)
    assert a.info["metrics"] == b.info["metrics"]


def test_long_dwell_fails_frame_time(evaluate, reference_arch, tmp_path_factory):
    """Maximum integration at minimum PRF blows the search frame budget:
    the dwell-time / beam-count product is the binding axis. (The geometry
    stays at the reference size: PAM's full-pattern recompute needs
    ~16 bytes x n_elements x 260k grid points, so a giant-aperture variant
    of this test would be OOM-killed, not failed.)"""
    doc = copy.deepcopy(reference_arch)
    doc["waveform"]["n_pulses"] = 64
    doc["waveform"]["prf_hz"] = 500.0
    result = evaluate(doc, tmp_path_factory.mktemp("dwell"))
    assert not result.passed
    assert "frame-time" in _failures(result)


def test_hot_cheap_pa_fails_availability(evaluate, reference_arch, tmp_path_factory):
    """Class-A PA at max power runs the junction hot; Arrhenius derating
    erodes availability below the floor."""
    doc = _tweak(reference_arch, rf__pa_class="A", rf__tx_power_w_per_elem=12.0)
    result = evaluate(doc, tmp_path_factory.mktemp("hot"))
    assert not result.passed
    assert "availability" in _failures(result)


def test_nominal_only_design_fails_the_envelope(evaluate, reference_arch, spec, tmp_path_factory):
    """A design tuned to pass only the nominal point fails the held-out
    envelope: starve the reference's power until the nominal still closes
    but the worst point does not."""
    nominal_only = None
    for scale in (0.7, 0.55, 0.4, 0.3):
        doc = copy.deepcopy(reference_arch)
        doc["rf"]["tx_power_w_per_elem"] = round(
            reference_arch["rf"]["tx_power_w_per_elem"] * scale, 2
        )
        result = evaluate(doc, tmp_path_factory.mktemp(f"nom{int(scale * 100)}"))
        pds = [row["pd"] for row in result.info["per_point"]]
        floor = 0.7
        if max(pds) >= floor and _metric(result, "worst_case_pd") < floor:
            nominal_only = result
            break
    assert nominal_only is not None, "could not construct a nominal-only design"
    assert "detection" in _failures(nominal_only)


def test_wide_spacing_fails_grating(evaluate, reference_arch, tmp_path_factory):
    doc = _tweak(reference_arch, array__dx_lambda=0.9, array__dy_lambda=0.9)
    result = evaluate(doc, tmp_path_factory.mktemp("grate"))
    assert not result.passed
    assert "grating" in _failures(result)


def test_pinned_fields_rejected(evaluate, reference_arch, tmp_path_factory):
    for section, key, value in (
        ("rf", "pa_efficiency", 0.9),
        ("rf", "duty_cycle", 0.01),
        ("waveform", "swerling", 0),
        ("waveform", "pfa", 1e-2),
        ("digital", "cfar_type", "none"),
    ):
        doc = copy.deepcopy(reference_arch)
        doc[section][key] = value
        with pytest.raises(ValueError, match=r"pinned|parts table|must not set"):
            evaluate(doc, tmp_path_factory.mktemp(f"pin_{key}"))


def test_unknown_and_missing_keys_rejected(evaluate, reference_arch, tmp_path_factory):
    doc = copy.deepcopy(reference_arch)
    doc["array"]["mystery"] = 1.0
    with pytest.raises(ValueError, match="unknown keys"):
        evaluate(doc, tmp_path_factory.mktemp("unk"))

    doc = copy.deepcopy(reference_arch)
    del doc["waveform"]
    with pytest.raises(ValueError, match="missing keys"):
        evaluate(doc, tmp_path_factory.mktemp("mis"))


def test_waveform_bounds_rejected(evaluate, reference_arch, tmp_path_factory):
    doc = _tweak(reference_arch, waveform__prf_hz=10_000.0)
    with pytest.raises(ValueError, match="prf_hz"):
        evaluate(doc, tmp_path_factory.mktemp("prf"))

    doc = _tweak(reference_arch, waveform__n_pulses=1000)
    with pytest.raises(ValueError, match="n_pulses"):
        evaluate(doc, tmp_path_factory.mktemp("np"))


def test_brief_declares_bounds_not_envelope(spec):
    """The rendered brief must show the declared bounds and never the
    held-out points or seeds."""
    from aedl.harness.workspace import render_brief

    brief = render_brief(spec)
    assert "evaluation_envelope_bounds" in brief or "clutter_range_km_max" in brief
    assert "failure_seeds" not in brief
    assert "points:" not in brief
    for seed in spec.evaluator_params["envelope"]["failure_seeds"]:
        assert str(seed) not in brief


def test_duty_cycle_is_derived(reference_result, spec):
    tau_s = float(spec.context["parts"]["tau_us"]) * 1e-6
    prf = float(reference_result.info["architecture"]["waveform"]["prf_hz"])
    assert reference_result.info["duty_cycle"] == pytest.approx(tau_s * prf)
