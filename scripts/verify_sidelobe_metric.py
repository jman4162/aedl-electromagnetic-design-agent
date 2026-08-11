"""Check t2-001's peak-sidelobe metric against continuous refinement.

The evaluator reads the pattern on a 361 x 721 grid and reports the
second-highest local maximum as the peak sidelobe. This script re-measures the
same designs independently: it seeds from local maxima of a three-times denser
grid, refines each one continuously with Nelder-Mead, discards any that climbs
into the main beam, and reports the best survivor. Agreement between the two
columns is the evidence that the grid reading is not flattering any design.

It also reports what a fixed angular exclusion around the target would have
said, which is what the task did before 2026-08-10. For a 16x16 array steered
to 27 degrees an 8 degree radius sits inside the main lobe, so once a design
pushes its true sidelobes below its own skirt level at 8 degrees, the metric
reports the skirt and stops responding to the sidelobes.

Usage:
    python scripts/verify_sidelobe_metric.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import phased_array as pa
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aedl.evaluators.array_pattern import _angular_separation_deg  # noqa: E402
from aedl.registry import get_evaluator  # noqa: E402
from aedl.spec import find_task  # noqa: E402

C0 = 299_792_458.0
SEARCH_REFINE = 3  # search-grid density relative to the scoring grid
N_SEEDS = 400  # strongest distinct local maxima to refine
OLD_EXCLUSION_DEG = 8.0  # the fixed radius the task used before the fix


def _geometry(ctx: dict) -> tuple[object, float, float]:
    arr = ctx["array"]
    wavelength = C0 / (float(arr["frequency_ghz"]) * 1e9)
    geom = pa.create_rectangular_array(
        int(arr["nx"]),
        int(arr["ny"]),
        dx=float(arr["dx_wl"]),
        dy=float(arr["dy_wl"]),
        wavelength=wavelength,
    )
    q = float(ctx.get("element", {}).get("q", 1.0))
    return geom, pa.wavelength_to_k(wavelength), q


def _apply_failures(weights: npt.NDArray[np.complex128], ctx: dict) -> npt.NDArray[np.complex128]:
    w = np.asarray(weights, dtype=complex).copy()
    failed = np.asarray(ctx.get("failed_elements", []), dtype=int)
    if failed.size:
        w[failed] = 0.0
    return w


def _pattern_db_at(theta: float, phi: float, geom, w, k: float, q: float) -> float:
    """Pattern in dB at one direction, same physics as the evaluator."""
    if np.cos(theta) <= 0.0:
        return -400.0
    u = np.sin(theta) * np.cos(phi)
    v = np.sin(theta) * np.sin(phi)
    af = np.sum(w * np.exp(1j * k * (geom.x * u + geom.y * v)))
    return float(20.0 * np.log10(np.abs(af) * np.cos(theta) ** q + 1e-300))


def refined_peak_sidelobe(weights, ctx: dict, n_theta: int, n_phi: int) -> float:
    """Peak sidelobe by continuous refinement, relative to the refined main peak."""
    geom, k, q = _geometry(ctx)
    w = _apply_failures(weights, ctx)
    target = (
        np.radians(float(ctx["target"]["theta_deg"])),
        np.radians(float(ctx["target"]["phi_deg"])),
    )

    theta, phi, dense = pa.compute_full_pattern(
        geom.x,
        geom.y,
        w,
        k,
        n_theta=(n_theta - 1) * SEARCH_REFINE + 1,
        n_phi=(n_phi - 1) * SEARCH_REFINE + 1,
        theta_range=(0.0, np.pi),
        element_pattern_func=pa.element_pattern,
        cos_exp_theta=q,
    )
    tg, pg = np.meshgrid(theta, phi, indexing="ij")

    is_max = np.zeros_like(dense, dtype=bool)
    is_max[1:-1, 1:-1] = True
    for dt, dp in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        is_max &= dense >= np.roll(np.roll(dense, dt, axis=0), dp, axis=1)

    neg = lambda p: -_pattern_db_at(float(p[0]), float(p[1]), geom, w, k, q)  # noqa: E731
    opts = {"xatol": 1e-12, "fatol": 1e-12}

    main = -float(minimize(neg, np.array(target), method="Nelder-Mead", options=opts).fun)

    order = np.argsort(np.where(is_max, dense, -np.inf), axis=None)[::-1][:N_SEEDS]
    best = -np.inf
    for a, b in zip(*np.unravel_index(order, dense.shape), strict=True):
        if not np.isfinite(dense[a, b]):
            continue
        res = minimize(neg, np.array([tg[a, b], pg[a, b]]), method="Nelder-Mead", options=opts)
        # A seed on the main lobe climbs to the main peak; that is not a sidelobe.
        if -float(res.fun) >= main - 1e-6:
            continue
        best = max(best, -float(res.fun))
    return best - main


def fixed_radius_sidelobe(weights, ctx: dict, n_theta: int, n_phi: int, radius_deg: float) -> float:
    """What a fixed angular exclusion around the target reports."""
    geom, k, q = _geometry(ctx)
    w = _apply_failures(weights, ctx)
    theta, phi, pattern_db = pa.compute_full_pattern(
        geom.x,
        geom.y,
        w,
        k,
        n_theta=n_theta,
        n_phi=n_phi,
        theta_range=(0.0, np.pi),
        element_pattern_func=pa.element_pattern,
        cos_exp_theta=q,
    )
    tg, pg = np.meshgrid(theta, phi, indexing="ij")
    sep = _angular_separation_deg(
        tg,
        pg,
        np.radians(float(ctx["target"]["theta_deg"])),
        np.radians(float(ctx["target"]["phi_deg"])),
    )
    return float(np.max(pattern_db[sep > radius_deg]) - np.max(pattern_db))


def main() -> None:
    spec = find_task(REPO / "tasks", "t2-001")
    ctx = spec.context
    n_theta = int(spec.evaluator_params.get("n_theta", 361))
    n_phi = int(spec.evaluator_params.get("n_phi", 721))
    evaluate = get_evaluator(spec.evaluator)

    geom, k, _ = _geometry(ctx)
    th = np.radians(float(ctx["target"]["theta_deg"]))
    ph = np.radians(float(ctx["target"]["phi_deg"]))
    ideal = -k * (geom.x * np.sin(th) * np.cos(ph) + geom.y * np.sin(th) * np.sin(ph))
    step = 2 * np.pi / (2 ** int(ctx["phase_bits"]))

    sys.path.insert(0, str(REPO / "tasks" / "t2-001-steered-beam-2bit" / "reference"))
    import solve as reference

    designs = [
        ("direct 2-bit rounding", np.exp(1j * (np.round(ideal / step) * step))),
        ("reference solution", reference.solve()),
    ]
    for bundle in sorted((REPO / "runs").glob("*/workspace/submission.npz")):
        with np.load(bundle) as data:
            designs.append((f"agent {bundle.parts[-3][-8:]}", data["weights"]))

    tmp = REPO / ".verify_submission.npz"
    print(f"scoring grid {n_theta} x {n_phi}, search grid {SEARCH_REFINE}x denser\n")
    print(f"{'design':<24}{'evaluator':>11}{'refined':>10}{'gap':>7}{'8deg radius':>13}")
    try:
        for name, weights in designs:
            np.savez(tmp, weights=np.asarray(weights, dtype=complex))
            scored = next(
                r.value
                for r in evaluate(spec, tmp).requirements
                if r.metric.startswith("peak_side")
            )
            refined = refined_peak_sidelobe(weights, ctx, n_theta, n_phi)
            old = fixed_radius_sidelobe(weights, ctx, n_theta, n_phi, OLD_EXCLUSION_DEG)
            print(f"{name:<24}{scored:11.2f}{refined:10.2f}{scored - refined:7.2f}{old:13.2f}")
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
