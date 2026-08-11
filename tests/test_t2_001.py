"""Evaluator sanity for t2-001: the reference passes; wrong solutions fail
for the right reasons."""

import runpy
from pathlib import Path

import numpy as np
import phased_array as pa
import pytest

from aedl import get_evaluator
from aedl.spec import find_task

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"
SOLVE = TASKS / "t2-001-steered-beam-2bit" / "reference" / "solve.py"

C0 = 299_792_458.0
THETA, PHI = 27.0, 10.0
STEP = np.pi / 2


@pytest.fixture(scope="module")
def spec():
    return find_task(TASKS, "t2-001")


@pytest.fixture(scope="module")
def evaluate(spec):
    ev = get_evaluator(spec.evaluator)

    def run(weights, tmpdir):
        path = Path(tmpdir) / "submission.npz"
        np.savez(path, weights=weights)
        return ev(spec, path)

    return run


@pytest.fixture(scope="module")
def geometry():
    wavelength = C0 / 10e9
    geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5, wavelength=wavelength)
    return geom, pa.wavelength_to_k(wavelength)


def _reference_weights():
    return runpy.run_path(str(SOLVE))["solve"]()


def _ideal_phase(geometry):
    geom, k = geometry
    return np.angle(pa.steering_vector(k, geom.x, geom.y, THETA, PHI))


def _failures(result):
    return {r.requirement_id for r in result.requirements if not r.passed}


def _metric(result, name):
    return next(r.value for r in result.requirements if r.requirement_id == name)


def test_reference_passes(evaluate, tmp_path_factory):
    result = evaluate(_reference_weights(), tmp_path_factory.mktemp("ref"))
    assert result.passed, result.summary_table()


def test_naive_quantized_steering_fails_sidelobes(evaluate, geometry, tmp_path_factory):
    """Rounding onto the 2-bit grid steers correctly but leaves the sidelobes high."""
    quantized = np.exp(1j * np.round(_ideal_phase(geometry) / STEP) * STEP)
    result = evaluate(quantized, tmp_path_factory.mktemp("naive"))
    assert not result.passed
    assert _failures(result) == {"sidelobes"}
    assert _metric(result, "steering") < 1.5


def _best_global_rotation(ideal_phase):
    """Offset minimizing the worst-case distance from the grid, and that distance."""
    offsets = np.linspace(0.0, STEP, 721)
    residual = np.array(
        [np.max(np.abs(((ideal_phase + o + STEP / 2) % STEP) - STEP / 2)) for o in offsets]
    )
    i = int(np.argmin(residual))
    return offsets[i], residual[i]


def test_global_phase_rotation_cannot_make_phases_exactly_representable(geometry):
    """A constant phase offset costs nothing physically. At some steering angles it
    makes every ideal phase land exactly on the quantization grid, erasing the
    quantization error and collapsing the task — that is what killed the original
    (30, 0) geometry, where this residual was 0."""
    _, residual = _best_global_rotation(_ideal_phase(geometry))
    assert np.degrees(residual) > 20.0, (
        f"a global rotation leaves only {np.degrees(residual):.1f} deg of "
        "quantization error; the task is degenerate at this steering angle"
    )


def test_best_global_rotation_still_fails_sidelobes(evaluate, geometry, tmp_path_factory):
    """End-to-end check at the single most favourable offset."""
    ideal = _ideal_phase(geometry)
    offset, _ = _best_global_rotation(ideal)
    weights = np.exp(1j * (np.round((ideal + offset) / STEP) * STEP - offset))
    result = evaluate(weights, tmp_path_factory.mktemp("rot"))
    assert not result.passed
    assert "sidelobes" in _failures(result)


def test_unquantized_steering_fails_phase_grid(evaluate, geometry, tmp_path_factory):
    geom, k = geometry
    w = pa.steering_vector(k, geom.x, geom.y, THETA, PHI)
    result = evaluate(w, tmp_path_factory.mktemp("unquant"))
    assert not result.passed
    assert "phase-quantization" in _failures(result)


def test_broadside_fails_steering(evaluate, geometry, tmp_path_factory):
    geom, _ = geometry
    result = evaluate(np.ones(geom.n_elements, dtype=complex), tmp_path_factory.mktemp("bs"))
    assert not result.passed
    assert "steering" in _failures(result)


def test_amplitude_taper_fails_phase_only(evaluate, tmp_path_factory):
    w = _reference_weights() * np.linspace(0.5, 1.0, 256)
    result = evaluate(w, tmp_path_factory.mktemp("taper"))
    assert not result.passed
    assert "phase-only-control" in _failures(result)


def test_wrong_shape_rejected(evaluate, tmp_path_factory):
    with pytest.raises(ValueError, match="shape"):
        evaluate(np.ones(64, dtype=complex), tmp_path_factory.mktemp("shape"))


def test_determinism(evaluate, tmp_path_factory):
    w = _reference_weights()
    r1 = evaluate(w, tmp_path_factory.mktemp("d1"))
    r2 = evaluate(w, tmp_path_factory.mktemp("d2"))
    assert {r.requirement_id: r.value for r in r1.requirements} == {
        r.requirement_id: r.value for r in r2.requirements
    }


def _fixed_radius_sidelobe(weights, spec, radius_deg):
    """What the metric reported before: max outside a fixed radius of the target."""
    from aedl.evaluators.array_pattern import _angular_separation_deg

    ctx = spec.context
    wavelength = C0 / 10e9
    geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5, wavelength=wavelength)
    w = np.asarray(weights, dtype=complex).copy()
    w[np.asarray(ctx["failed_elements"], dtype=int)] = 0.0
    theta, phi, pattern_db = pa.compute_full_pattern(
        geom.x,
        geom.y,
        w,
        pa.wavelength_to_k(wavelength),
        n_theta=int(spec.evaluator_params["n_theta"]),
        n_phi=int(spec.evaluator_params["n_phi"]),
        theta_range=(0.0, np.pi),
        element_pattern_func=pa.element_pattern,
        cos_exp_theta=1.0,
    )
    tg, pg = np.meshgrid(theta, phi, indexing="ij")
    sep = _angular_separation_deg(tg, pg, np.radians(THETA), np.radians(PHI))
    return float(np.max(pattern_db[sep > radius_deg]) - np.max(pattern_db))


def test_sidelobe_metric_is_a_local_maximum_not_the_main_lobe_skirt(
    evaluate, spec, tmp_path_factory
):
    """The reported sidelobe must be a real lobe, not a point on the main beam.

    The task used to exclude a fixed 8 degree radius around the target. For a
    16x16 array steered to 27 degrees that radius sits inside the main lobe, so
    once a design pushed its true sidelobes below the skirt level at 8 degrees
    the metric reported the skirt instead and stopped responding to the
    sidelobes. The reference hit exactly that: -16.72 dB reported against
    -16.98 dB actually achieved.
    """
    weights = _reference_weights()
    reported = _metric(evaluate(weights, tmp_path_factory.mktemp("sll")), "sidelobes")
    skirt = _fixed_radius_sidelobe(weights, spec, 8.0)

    # The old fixed radius clipped the skirt in, and the skirt is the higher of
    # the two, so it understated the design by a quarter of a decibel.
    assert skirt > reported + 0.2
    assert reported == pytest.approx(-16.98, abs=0.02)


def test_sidelobe_metric_tracks_continuous_refinement(evaluate, tmp_path_factory):
    """Reading the metric off the scoring grid must not flatter any design.

    Independent measurement, refining each local maximum continuously off the
    grid, gives these values. The grid reading has to match them, or the task
    scores optimized designs on sampling luck.
    """
    expected = {"naive": -10.67, "reference": -16.98}
    naive = np.exp(1j * np.round(_ideal_phase(_geometry_pair()) / STEP) * STEP)

    for name, weights in (("naive", naive), ("reference", _reference_weights())):
        result = evaluate(weights, tmp_path_factory.mktemp(name))
        assert _metric(result, "sidelobes") == pytest.approx(expected[name], abs=0.02)


def _geometry_pair():
    wavelength = C0 / 10e9
    geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5, wavelength=wavelength)
    return geom, pa.wavelength_to_k(wavelength)


def test_sidelobe_metric_handles_a_beam_steered_to_phi_zero(evaluate, geometry, tmp_path_factory):
    """A peak at phi = 0 must not be counted twice.

    compute_full_pattern spans a full turn with the endpoint duplicated, so
    column 0 and the last column hold the same samples. A beam steered to
    phi = 0 therefore has its peak in both, and the second copy used to look
    like a fresh lobe, making the metric report 0 dB.
    """
    geom, k = geometry
    ideal = np.angle(pa.steering_vector(k, geom.x, geom.y, 30.0, 0.0))
    result = evaluate(np.exp(1j * ideal), tmp_path_factory.mktemp("phi0"))

    sll = _metric(result, "sidelobes")
    assert sll < -5.0, f"metric collapsed on a phi=0 beam: {sll}"
    assert sll == pytest.approx(-12.32, abs=0.05)
