"""Deterministic evaluator for planar-array pattern-synthesis tasks.

The submission is an .npz file containing a complex ``weights`` array, one entry
per element, ordered to match ``phased_array.create_rectangular_array``. That
function meshgrids with ``indexing='ij'`` and ravels, so **y varies fastest and
x slowest**: index ``i`` is element ``(ix, iy) = divmod(i, ny)``. The evaluator
enforces the hardware
constraints declared in the task context (phase-only control, phase-shifter bit
depth), zeroes the failed elements listed in the spec, computes the far-field
pattern with ``phased-array-modeling``, and scores every requirement.

Metrics produced (requirements reference these by name):

- ``amplitude_error``: max | |w| - 1 | over active elements (phase-only control)
- ``phase_grid_error_deg``: max distance of any active element's phase from the
  allowed phase-shifter grid
- ``peak_direction_error_deg``: angle between the pattern peak and the target
- ``peak_sidelobe_level_db``: highest pattern value (relative to peak, dB)
  outside ``exclusion_radius_deg`` of the target direction
- ``directivity_dbi``: full-sphere directivity including the element pattern
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import phased_array as pa

from aedl.registry import register_evaluator
from aedl.result import EvaluationResult, RequirementResult
from aedl.spec import TaskSpec

C0 = 299_792_458.0


def _angular_separation_deg(
    theta1: npt.NDArray[np.float64] | float,
    phi1: npt.NDArray[np.float64] | float,
    theta2: npt.NDArray[np.float64] | float,
    phi2: npt.NDArray[np.float64] | float,
) -> npt.NDArray[np.float64]:
    """Great-circle angle between directions given in radians, result in degrees."""
    dot = np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2) + np.cos(theta1) * np.cos(theta2)
    separation: npt.NDArray[np.float64] = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    return separation


def _peak_sidelobe_db(pattern_db: npt.NDArray[np.float64]) -> float:
    """Peak sidelobe relative to the main beam, by local-maximum detection.

    A sidelobe is a local maximum of the pattern other than the main beam, so
    the peak sidelobe is the second-highest local maximum. Walking the samples
    in descending order, the first one with no already-visited neighbour starts
    a new lobe; the first such sample after the global peak is that second
    maximum.

    This replaces a fixed angular exclusion radius around the target. A hand-set
    radius has to be wider than the main lobe, and the main lobe widens with
    scan angle and with any taper, so a radius that is safe for one design sits
    inside the main lobe for another. When that happens the reported "peak
    sidelobe" is a sample on the main-lobe skirt, and the metric stops
    responding to the sidelobes it is supposed to measure.

    Grid conventions: rows are theta, columns are phi over a full turn with the
    endpoint duplicated, so column wrap skips the duplicate.
    """
    n_theta, n_phi = pattern_db.shape
    span = n_phi - 1 if n_phi > 1 else 1  # last column repeats the first

    order = np.argsort(pattern_db, axis=None)[::-1]
    rows, cols = np.unravel_index(order, pattern_db.shape)
    peak_db = float(pattern_db[rows[0], cols[0]])

    visited = np.zeros(pattern_db.shape, dtype=bool)
    for n, (i, j) in enumerate(zip(rows, cols, strict=True)):
        has_higher_neighbour = False
        for di in (-1, 0, 1):
            ii = i + di
            if not 0 <= ii < n_theta:
                continue
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                if visited[ii, (j + dj) % span]:
                    has_higher_neighbour = True
                    break
            if has_higher_neighbour:
                break
        if not has_higher_neighbour and n > 0:
            return float(pattern_db[i, j]) - peak_db
        visited[i, j] = True
        if n_phi > 1 and j in (0, span):
            # Keep the duplicated endpoint columns consistent with each other.
            visited[i, 0] = visited[i, span] = True

    return float("-inf")


def _directivity_dbi(
    theta_g: npt.NDArray[np.float64],
    phi_g: npt.NDArray[np.float64],
    pattern_db: npt.NDArray[np.float64],
) -> float:
    """Full-sphere directivity from a pattern in dB on a regular theta/phi grid.

    Local implementation: phased_array.compute_directivity calls np.trapz,
    which NumPy 2 removed.
    """
    power = 10.0 ** (pattern_db / 10.0)
    integrand = power * np.sin(theta_g)
    total = np.trapezoid(np.trapezoid(integrand, phi_g[0, :], axis=1), theta_g[:, 0], axis=0)
    return float(10.0 * np.log10(4.0 * np.pi * np.max(power) / total))


@register_evaluator("array_pattern")
def evaluate(spec: TaskSpec, submission: Path) -> EvaluationResult:
    t_start = time.perf_counter()
    ctx = spec.context
    arr = ctx["array"]

    wavelength = C0 / (float(arr["frequency_ghz"]) * 1e9)
    geom = pa.create_rectangular_array(
        int(arr["nx"]),
        int(arr["ny"]),
        dx=float(arr["dx_wl"]),
        dy=float(arr["dy_wl"]),
        wavelength=wavelength,
    )
    n = geom.n_elements
    k = pa.wavelength_to_k(wavelength)

    with np.load(submission) as data:
        if "weights" not in data:
            raise ValueError("submission .npz must contain a 'weights' array")
        weights = np.asarray(data["weights"], dtype=complex).ravel()
    if weights.shape != (n,):
        raise ValueError(f"weights shape {weights.shape} != ({n},)")

    failed = np.asarray(ctx.get("failed_elements", []), dtype=int)
    active = np.setdiff1d(np.arange(n), failed)
    w_active = weights[active]

    metrics: dict[str, float] = {}

    # Hardware-constraint compliance (checked before failures are applied).
    metrics["amplitude_error"] = float(np.max(np.abs(np.abs(w_active) - 1.0)))
    phase_bits = int(ctx["phase_bits"])
    step = 360.0 / (2**phase_bits)
    phase_deg = np.degrees(np.angle(w_active)) % 360.0
    dist = np.abs(phase_deg - step * np.round(phase_deg / step))
    metrics["phase_grid_error_deg"] = float(np.max(np.minimum(dist, 360.0 - dist)))

    # The evaluator, not the agent, applies element failures.
    w = weights.copy()
    if failed.size:
        w[failed] = 0.0

    target_theta = np.radians(float(ctx["target"]["theta_deg"]))
    target_phi = np.radians(float(ctx["target"]["phi_deg"]))

    element = ctx.get("element", {})
    element_kwargs = {}
    element_func = None
    if element.get("model") == "cos_q":
        element_func = pa.element_pattern
        element_kwargs = {"cos_exp_theta": float(element.get("q", 1.0))}
    elif element:
        raise ValueError(f"unknown element model {element.get('model')!r}")

    theta, phi, pattern_db = pa.compute_full_pattern(
        geom.x,
        geom.y,
        w,
        k,
        n_theta=int(spec.evaluator_params.get("n_theta", 361)),
        n_phi=int(spec.evaluator_params.get("n_phi", 721)),
        theta_range=(0.0, np.pi),
        element_pattern_func=element_func,
        **element_kwargs,
    )
    theta_g, phi_g = np.meshgrid(theta, phi, indexing="ij")

    peak_idx = np.unravel_index(np.argmax(pattern_db), pattern_db.shape)
    peak_db = pattern_db[peak_idx]
    metrics["peak_direction_error_deg"] = float(
        _angular_separation_deg(theta_g[peak_idx], phi_g[peak_idx], target_theta, target_phi)
    )

    exclusion = ctx.get("exclusion_radius_deg")
    if exclusion is None:
        metrics["peak_sidelobe_level_db"] = _peak_sidelobe_db(pattern_db)
    else:
        # Fixed-radius exclusion, kept for tasks that pin one deliberately.
        sep = _angular_separation_deg(theta_g, phi_g, target_theta, target_phi)
        sidelobe_region = sep > float(exclusion)
        metrics["peak_sidelobe_level_db"] = float(np.max(pattern_db[sidelobe_region]) - peak_db)

    metrics["directivity_dbi"] = _directivity_dbi(theta_g, phi_g, pattern_db)

    req_results = []
    for req in spec.requirements:
        if req.metric not in metrics:
            raise KeyError(
                f"task {spec.id}: requirement {req.id!r} references unknown "
                f"metric {req.metric!r}; available: {sorted(metrics)}"
            )
        value = metrics[req.metric]
        req_results.append(
            RequirementResult(
                requirement_id=req.id,
                metric=req.metric,
                value=value,
                limit=req.limit,
                passed=req.check(value),
            )
        )

    return EvaluationResult(
        task_id=spec.id,
        passed=all(r.passed for r in req_results),
        requirements=tuple(req_results),
        info={"metrics": metrics, "n_elements": n, "n_failed": int(failed.size)},
        cost={"evaluation_wall_time_s": round(time.perf_counter() - t_start, 3)},
    )
