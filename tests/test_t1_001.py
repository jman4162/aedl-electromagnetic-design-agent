"""Evaluator sanity for t1-001: the reference passes, and each requirement
catches a specific wrong design.

The traps, each verified numerically before the thresholds were frozen
(docs/tasks/t1-001.md):

- the textbook design (the 0.254 mm default laminate, center frequency
  28 GHz, 50 ohm) misses the band twice: dispersion pulls the resonance
  below 27.5 GHz, and the laminate is too thin for the bandwidth;
- re-centring on the thin laminate fixes the resonance but not the bandwidth;
- a thick laminate fixes the bandwidth and breaks the footprint;
- the right laminate without dispersion pre-compensation lands below the band.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
from pathlib import Path

import pytest
import yaml

from aedl import get_evaluator
from aedl.spec import find_task

pytest.importorskip("antenna_cad", reason="requires the 'element' extra (antenna-cad)")

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"
REFERENCE = TASKS / "t1-001-patch-28ghz-ro4350b" / "reference" / "design.yaml"


@pytest.fixture(scope="module")
def spec():
    return find_task(TASKS, "t1-001")


@pytest.fixture(scope="module")
def reference_doc():
    # Loaded before any harness run hides the reference directory.
    return yaml.safe_load(REFERENCE.read_text())


@pytest.fixture(scope="module")
def evaluate(spec):
    evaluator = get_evaluator(spec.evaluator)

    def run(doc, tmpdir, task_spec=None):
        path = Path(tmpdir) / "design.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False))
        return evaluator(task_spec or spec, path)

    return run


@pytest.fixture(scope="module")
def reference_result(evaluate, reference_doc, tmp_path_factory):
    return evaluate(reference_doc, tmp_path_factory.mktemp("ref"))


def _design(fc_ghz: float, z_ohm: float = 50.0, h_mm: float = 0.254) -> dict:
    return {
        "design": {"name": "probe"},
        "requirements": {
            "center_frequency": f"{fc_ghz} GHz",
            "impedance": f"{z_ohm} ohm",
            "substrate": "RO4350B",
            "substrate_height": f"{h_mm} mm",
            "polarization": "linear",
        },
        "element": {"type": "rectangular_patch"},
    }


def _failures(result):
    return {r.requirement_id for r in result.requirements if not r.passed}


def _metric(result, name):
    return result.info["metrics"][name]


def test_task_loads_through_the_registry(spec):
    assert spec.tier == 1
    assert spec.evaluator == "patch_element"
    assert get_evaluator(spec.evaluator) is not None
    assert {r.id for r in spec.requirements} == {"s11-in-band", "footprint", "f-res-lo", "f-res-hi"}


def test_reference_passes(reference_result):
    assert reference_result.passed, _failures(reference_result)


def test_evaluator_produces_every_required_metric(spec, reference_result):
    assert {r.metric for r in spec.requirements} <= set(reference_result.info["metrics"])


def test_reference_is_labelled_analytic_tier(reference_result):
    """An analytic pass is a screening verdict, and the result says so."""
    info = reference_result.info
    assert info["fidelity"] == "analytic"
    assert info["certifies_band_coverage"] is False
    assert info["uncertainty"]["f_res_fraction"] == pytest.approx(0.03)
    shifted = info["uncertainty"]["s11_max_in_band_db_with_f_res_shift"]
    assert set(shifted) == {"-3%", "-1%", "+1%", "+3%"}
    # The +/-3% band is wider than the slack the reference has, which is
    # the point of reporting it.
    assert max(shifted.values()) > -10.0


def test_determinism(evaluate, reference_doc, tmp_path_factory):
    a = evaluate(reference_doc, tmp_path_factory.mktemp("d1"))
    b = evaluate(reference_doc, tmp_path_factory.mktemp("d2"))
    assert a.info["metrics"] == b.info["metrics"]
    assert a.info["uncertainty"] == b.info["uncertainty"]


def test_f_res_is_computed_from_geometry_not_claimed(reference_result, reference_doc):
    """The resonance comes from the synthesized dimensions; the claimed
    center frequency never enters the cavity model."""
    claimed_ghz = float(reference_doc["requirements"]["center_frequency"].split()[0])
    f_res = _metric(reference_result, "f_res_ghz")
    assert abs(f_res - claimed_ghz) / claimed_ghz > 0.01
    assert 27.5 <= f_res <= 28.5


def test_textbook_design_fails(evaluate, tmp_path):
    result = evaluate(_design(28.0, 50.0, 0.254), tmp_path)
    assert not result.passed
    assert {"s11-in-band", "f-res-lo"} <= _failures(result)


def test_thin_laminate_fails_bandwidth_even_when_centred(evaluate, tmp_path):
    result = evaluate(_design(28.55, 70.0, 0.254), tmp_path)
    assert _failures(result) == {"s11-in-band"}


def test_thick_laminate_fails_footprint(evaluate, tmp_path):
    result = evaluate(_design(29.4, 50.0, 0.762), tmp_path)
    assert _failures(result) == {"footprint"}
    assert _metric(result, "s11_max_in_band_db") < -10.0


def test_uncompensated_frequency_lands_below_the_band(evaluate, tmp_path):
    result = evaluate(_design(28.0, 60.0, 0.508), tmp_path)
    assert "f-res-lo" in _failures(result)


def test_high_resonance_fails_f_res_hi(evaluate, tmp_path):
    result = evaluate(_design(30.5, 60.0, 0.508), tmp_path)
    assert "f-res-hi" in _failures(result)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d["requirements"].__setitem__("substrate", "FR4"), "RO4350B"),
        (lambda d: d["requirements"].__setitem__("substrate_height", "0.3 mm"), "stock"),
        (lambda d: d["requirements"].__setitem__("patch_width", "3 mm"), "unknown"),
        (lambda d: d.__setitem__("geometry", {"width": 3.0}), "unknown"),
        (lambda d: d.__setitem__("array", {"nx": 2, "ny": 2}), "single element"),
        (lambda d: d["requirements"].__setitem__("polarization", "circular"), "linear"),
    ],
    ids=["substrate", "height", "geometry-in-requirements", "geometry-section", "array", "pol"],
)
def test_invalid_specs_rejected(evaluate, reference_doc, tmp_path, mutate, match):
    doc = copy.deepcopy(reference_doc)
    mutate(doc)
    with pytest.raises(ValueError, match=match):
        evaluate(doc, tmp_path)


def test_unknown_fidelity_rejected(evaluate, reference_doc, spec, tmp_path):
    bad = dataclasses.replace(spec, evaluator_params={**spec.evaluator_params, "fidelity": "hfss"})
    with pytest.raises(ValueError, match="fidelity"):
        evaluate(reference_doc, tmp_path, task_spec=bad)


def _openems_available() -> bool:
    if importlib.util.find_spec("openEMS") and importlib.util.find_spec("CSXCAD"):
        return True
    try:
        from antenna_cad.solvers.openems.solver import _docker_available

        return bool(_docker_available())
    except Exception:
        return False


@pytest.mark.skipif(_openems_available(), reason="openEMS is installed here")
def test_openems_tier_without_openems_says_why(evaluate, reference_doc, spec, tmp_path):
    full = dataclasses.replace(
        spec, evaluator_params={**spec.evaluator_params, "fidelity": "openems"}
    )
    with pytest.raises(RuntimeError, match="openEMS"):
        evaluate(reference_doc, tmp_path, task_spec=full)


@pytest.mark.requires_openems
@pytest.mark.skipif(not _openems_available(), reason="openEMS not installed (nightly tier)")
def test_openems_tier_scores_the_reference(evaluate, reference_doc, spec, tmp_path):
    full = dataclasses.replace(
        spec, evaluator_params={**spec.evaluator_params, "fidelity": "openems"}
    )
    result = evaluate(reference_doc, tmp_path, task_spec=full)
    assert result.info["fidelity"] == "openems"
    assert {r.metric for r in spec.requirements} <= set(result.info["metrics"])


def test_brief_does_not_carry_evaluator_params(spec):
    from aedl.harness.workspace import render_brief

    brief = render_brief(spec)
    assert "n_band_points" not in brief
    assert "27.5" in brief and "28.5" in brief
