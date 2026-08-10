"""Regenerate the README figures from the task and the reference solution.

Every figure here is an argument a reader can check, not decoration. Each one
is computed the same way the evaluator computes, so a figure cannot disagree
with a reported score: `verify_against_evaluator()` asserts that before
anything is drawn.

SVG rather than PNG, following the house pattern: it is text, so a diff shows
what changed. Output is byte-deterministic given a fixed matplotlib version,
which is what lets CI fail on a stale figure.

Usage:
    python scripts/generate_readme_figures.py                    # fast, ~10 s
    python scripts/generate_readme_figures.py --recompute-grid-bias
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "aedl"

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import phased_array as pa  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "docs" / "_static"
SOLVE = REPO / "tasks" / "t2-001-steered-beam-2bit" / "reference" / "solve.py"
GRID_BIAS_DATA = TARGET / "grid-bias.json"

C0 = 299_792_458.0
FREQ_HZ = 10e9
NX = NY = 16
FAILED = np.array([7, 23, 42, 58, 77, 96, 113, 128, 150, 171, 199, 220, 245])
EXCLUSION_DEG = 8.0
STEP = np.pi / 2  # 2-bit phase shifters

# The geometry t2-001 originally used, where the task turned out to be
# degenerate, and the one it was retargeted to.
ORIGINAL = (30.0, 0.0)
CURRENT = (27.0, 10.0)
ORIGINAL_REQUIREMENT_DB = -9.0
CURRENT_REQUIREMENT_DB = -14.0

SCORED_GRID = (361, 721)

# Wong colourblind-safe palette, as used in metasurfaces-py. Colour follows the
# entity: the unoptimized family is always orange and the optimized one always
# blue, across all three figures.
NAIVE = "#D55E00"
OPTIMIZED = "#0072B2"
INK = "#333333"
MUTED = "#666666"
GRID_LINE = "#e3e3e3"
SURFACE = "#ffffff"

STYLE = {
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.8,
    "grid.color": GRID_LINE,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.4,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "svg.fonttype": "path",
}


# --- physics, matching src/aedl/evaluators/array_pattern.py -----------------


def geometry() -> tuple[object, float]:
    wavelength = C0 / FREQ_HZ
    geom = pa.create_rectangular_array(NX, NY, dx=0.5, dy=0.5, wavelength=wavelength)
    return geom, pa.wavelength_to_k(wavelength)


def separation_deg(theta, phi, t0_deg: float, p0_deg: float):
    t0, p0 = np.radians(t0_deg), np.radians(p0_deg)
    dot = np.sin(theta) * np.sin(t0) * np.cos(phi - p0) + np.cos(theta) * np.cos(t0)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def pattern(weights, n_theta: int, n_phi: int):
    """The evaluator's own call, so figures and scores share a computation."""
    geom, k = geometry()
    w = np.asarray(weights, dtype=complex).copy()
    w[FAILED] = 0.0
    theta, phi, pattern_db = pa.compute_full_pattern(
        geom.x,
        geom.y,
        w,
        k,
        n_theta=n_theta,
        n_phi=n_phi,
        theta_range=(0.0, np.pi),
        element_pattern_func=pa.element_pattern,
        cos_exp_theta=1.0,
    )
    return theta, phi, pattern_db


def peak_sidelobe_db(weights, target, grid=SCORED_GRID) -> float:
    theta, phi, pattern_db = pattern(weights, *grid)
    tg, pg = np.meshgrid(theta, phi, indexing="ij")
    sep = separation_deg(tg, pg, *target)
    return float(np.max(pattern_db[sep > EXCLUSION_DEG]) - np.max(pattern_db))


def _raw_db(weights, theta, phi):
    """Un-normalized pattern in dB at arbitrary directions.

    `compute_full_pattern` normalizes to its own peak, which is fine on a grid
    but useless for refinement: both the main beam and the sidelobe have to be
    refined in the same raw units before taking their ratio.
    """
    geom, k = geometry()
    w = np.asarray(weights, dtype=complex).copy()
    w[FAILED] = 0.0
    u = (np.sin(theta) * np.cos(phi)).ravel()
    v = (np.sin(theta) * np.sin(phi)).ravel()
    element = np.clip(np.cos(theta).ravel(), 0.0, None)
    field = np.exp(1j * k * (np.outer(u, geom.x) + np.outer(v, geom.y))) @ w
    return 20.0 * np.log10(np.maximum(np.abs(field) * element, 1e-30))


def _refine_peak(weights, t0, p0, span_t, span_p, target, outside, rounds=7):
    """Zoom in on a peak until the sampling step stops mattering."""
    t, p = t0, p0
    best_val = -np.inf
    for _ in range(rounds):
        ts = np.linspace(t - span_t, t + span_t, 11)
        ps = np.linspace(p - span_p, p + span_p, 11)
        tt, pp = np.meshgrid(ts, ps, indexing="ij")
        vals = _raw_db(weights, tt, pp)
        if outside:
            keep = separation_deg(tt, pp, *target).ravel() > EXCLUSION_DEG
            vals = np.where(keep, vals, -np.inf)
        i = int(np.argmax(vals))
        best_val = float(vals[i])
        t, p = tt.ravel()[i], pp.ravel()[i]
        span_t /= 5.0
        span_p /= 5.0
    return best_val


def refined_sidelobe_db(weights, target, grid=(721, 1441)) -> float:
    """Sidelobe peak found by local refinement rather than read off a grid.

    A sculpted null leaves a narrow residual peak that a fixed grid can step
    over, so the grid reading is optimistic. Zooming in on both the main beam
    and the sidelobe recovers the value the grid missed.
    """
    theta, phi, pattern_db = pattern(weights, *grid)
    tg, pg = np.meshgrid(theta, phi, indexing="ij")
    sep = separation_deg(tg, pg, *target)
    span_t = float(theta[1] - theta[0])
    span_p = float(phi[1] - phi[0])

    i, j = np.unravel_index(np.argmax(pattern_db), pattern_db.shape)
    main = _refine_peak(weights, theta[i], phi[j], span_t, span_p, target, outside=False)

    # Refine several candidates, not just the grid's argmax: a neighbouring
    # sidelobe that reads slightly lower on the grid can be the higher one once
    # the sampling error is removed, which is the whole point of the exercise.
    masked = np.where(sep > EXCLUSION_DEG, pattern_db, -np.inf)
    flat = masked.ravel()
    candidates = np.argpartition(flat, -400)[-400:]
    seen: list[tuple[float, float]] = []
    side = -np.inf
    for idx in candidates[np.argsort(flat[candidates])[::-1]]:
        i, j = np.unravel_index(idx, masked.shape)
        t, p = float(theta[i]), float(phi[j])
        # One refinement per distinct lobe; nearby samples share a peak.
        if any(separation_deg(t, p, np.degrees(t0), np.degrees(p0)) < 1.0 for t0, p0 in seen):
            continue
        seen.append((t, p))
        side = max(side, _refine_peak(weights, t, p, span_t, span_p, target, outside=True))
        if len(seen) >= 12:
            break

    return side - main


def principal_cut(weights, grid=SCORED_GRID, phi_cut_deg: float = 0.0):
    """A cut through phi_cut_deg, spanning -90..90 degrees of theta."""
    theta, phi, pattern_db = pattern(weights, *grid)
    phi_deg, theta_deg = np.degrees(phi), np.degrees(theta)
    i0 = int(np.argmin(np.abs(phi_deg - phi_cut_deg)))
    i1 = int(np.argmin(np.abs(phi_deg - (phi_cut_deg + 180.0))))
    angles = np.concatenate([-theta_deg[:0:-1], theta_deg])
    cut = np.concatenate([pattern_db[:0:-1, i1], pattern_db[:, i0]])
    return angles, cut


# --- the designs the figures compare ---------------------------------------


def ideal_phase(target) -> np.ndarray:
    geom, k = geometry()
    return np.angle(pa.steering_vector(k, geom.x, geom.y, target[0], target[1]))


def direct_rounding(target) -> np.ndarray:
    return np.exp(1j * (np.round(ideal_phase(target) / STEP) * STEP))


def rotated_then_rounded(target, offset_rad: float) -> np.ndarray:
    """Shift the phase reference, then round. The shift itself is free."""
    return np.exp(1j * (np.round((ideal_phase(target) + offset_rad) / STEP) * STEP))


def best_global_rotation(target) -> tuple[np.ndarray, float]:
    """The offset leaving the least worst-case distance from the phase grid."""
    ideal = ideal_phase(target)
    offsets = np.linspace(0.0, STEP, 721)
    residual = np.array(
        [np.max(np.abs(((ideal + o + STEP / 2) % STEP) - STEP / 2)) for o in offsets]
    )
    best = offsets[int(np.argmin(residual))]
    return rotated_then_rounded(target, best), float(np.degrees(residual.min()))


def coordinate_descent() -> np.ndarray:
    return runpy.run_path(str(SOLVE))["solve"]()


def verify_against_evaluator() -> None:
    """The figures must agree with what `aedl evaluate` reports."""
    sys.path.insert(0, str(REPO / "src"))
    from aedl import get_evaluator
    from aedl.spec import find_task

    spec = find_task(REPO / "tasks", "t2-001")
    weights = coordinate_descent()
    mine = peak_sidelobe_db(weights, CURRENT)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "submission.npz"
        np.savez(path, weights=weights)
        result = get_evaluator(spec.evaluator)(spec, path)
    theirs = next(r.value for r in result.requirements if r.requirement_id == "sidelobes")
    if abs(mine - theirs) > 1e-6:
        raise SystemExit(f"figure physics disagrees with the evaluator: {mine:.6f} vs {theirs:.6f}")


# --- figures ----------------------------------------------------------------


def save(fig, name: str) -> Path:
    TARGET.mkdir(parents=True, exist_ok=True)
    path = TARGET / name
    fig.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        facecolor=SURFACE,
        metadata={"Date": None},
    )
    plt.close(fig)
    return path


def figure_free_lunch() -> Path:
    """The degeneracy that made the original task collapse."""
    naive = direct_rounding(ORIGINAL)
    rotated = rotated_then_rounded(ORIGINAL, STEP / 2)  # +45 degrees

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for weights, colour, label in (
        (naive, NAIVE, "round onto the 2-bit grid"),
        (rotated, OPTIMIZED, "add 45° to every element, then round"),
    ):
        angles, cut = principal_cut(weights)
        sll = peak_sidelobe_db(weights, ORIGINAL)
        ax.plot(angles, cut, color=colour, label=f"{label}  ({sll:.1f} dB)")

    ax.axhline(ORIGINAL_REQUIREMENT_DB, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.annotate(
        f"requirement {ORIGINAL_REQUIREMENT_DB:.0f} dB",
        xy=(88, ORIGINAL_REQUIREMENT_DB),
        xytext=(88, ORIGINAL_REQUIREMENT_DB + 1.6),
        color=MUTED,
        fontsize=8,
        ha="right",
    )
    ax.set_xlim(-90, 90)
    ax.set_ylim(-40, 3)
    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_xlabel("θ (degrees), φ = 0 cut")
    ax.set_ylabel("gain relative to peak (dB)")
    ax.set_title(
        "A constant phase shift is free, and at (30°, 0°) it removed the task",
        loc="left",
        pad=10,
    )
    ax.grid(True, axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False, borderaxespad=0.6)
    return save(fig, "free-lunch.svg")


def figure_recalibration() -> Path:
    """After retargeting, the shortcut buys nothing and the bar means something."""
    rotated, residual_deg = best_global_rotation(CURRENT)
    rows = [
        ("direct rounding onto the 2-bit grid", direct_rounding(CURRENT), NAIVE),
        (f"best global rotation ({residual_deg:.0f}° left off-grid)", rotated, NAIVE),
        ("greedy coordinate descent", coordinate_descent(), OPTIMIZED),
    ]
    values = [peak_sidelobe_db(w, CURRENT) for _, w, _ in rows]

    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    ypos = np.arange(len(rows))[::-1]
    for y, (_label, _, colour), value in zip(ypos, rows, values, strict=True):
        ax.plot([value, CURRENT_REQUIREMENT_DB], [y, y], color=GRID_LINE, lw=1.2, zorder=1)
        ax.plot(value, y, "o", color=colour, ms=9, zorder=2)
        verdict = "passes" if value <= CURRENT_REQUIREMENT_DB else "fails"
        ax.annotate(
            f"{value:.2f} dB, {verdict}",
            xy=(value, y),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK,
        )

    ax.axvline(CURRENT_REQUIREMENT_DB, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.annotate(
        f"requirement {CURRENT_REQUIREMENT_DB:.0f} dB",
        xy=(CURRENT_REQUIREMENT_DB, -0.62),
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels([label for label, _, _ in rows])
    ax.set_ylim(-0.9, len(rows) - 0.35)
    ax.set_xlim(-18.2, -9.4)
    ax.set_xlabel("peak sidelobe (dB), lower is better")
    ax.set_title(
        "Retargeted to (27°, 10°): the shortcut is closed, the threshold binds",
        loc="left",
        pad=10,
    )
    ax.grid(True, axis="x", alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return save(fig, "recalibration.svg")


def compute_grid_bias() -> dict:
    """Slow: several full-sphere patterns per design. Writes the cached JSON."""
    grids = [(361, 721), (541, 1081), (721, 1441), (1081, 2161)]
    designs = {
        "direct rounding": direct_rounding(CURRENT),
        "coordinate descent": coordinate_descent(),
    }
    data: dict = {"grids": [n_theta for n_theta, _ in grids], "series": {}}
    for name, weights in designs.items():
        print(f"  {name}: ", end="", flush=True)
        scored = []
        for grid in grids:
            scored.append(peak_sidelobe_db(weights, CURRENT, grid=grid))
            print(f"{grid[0]} ", end="", flush=True)
        continuous = refined_sidelobe_db(weights, CURRENT)
        print(f"-> continuous {continuous:.3f} dB")
        data["series"][name] = {"scored": scored, "continuous": continuous}
    GRID_BIAS_DATA.parent.mkdir(parents=True, exist_ok=True)
    GRID_BIAS_DATA.write_text(json.dumps(data, indent=2) + "\n")
    return data


def figure_grid_bias() -> Path:
    """The scored grid flatters optimized designs and leaves naive ones alone."""
    if not GRID_BIAS_DATA.exists():
        raise SystemExit(
            f"no cached measurement at {GRID_BIAS_DATA.relative_to(REPO)}. Build it:\n"
            "  python scripts/generate_readme_figures.py --recompute-grid-bias"
        )
    data = json.loads(GRID_BIAS_DATA.read_text())
    grids = data["grids"]
    x = np.arange(len(grids))

    # Plot the error itself, not the absolute level. On an absolute axis
    # spanning -10 to -17 dB a fifth of a decibel is invisible, and the
    # differential error is the entire claim. Positive means the grid reported
    # a better sidelobe than the design actually achieves.
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for name, colour in (("direct rounding", NAIVE), ("coordinate descent", OPTIMIZED)):
        series = data["series"][name]
        flattery = [series["continuous"] - s for s in series["scored"]]
        ax.plot(x, flattery, "o-", color=colour, ms=7, label=name, zorder=3)
        ax.annotate(
            f"{flattery[0]:+.3f} dB at the scored grid",
            xy=(0, flattery[0]),
            xytext=(12, -24 if flattery[0] > 0.05 else -16),
            textcoords="offset points",
            fontsize=8,
            color=colour,
            fontweight="bold",
        )

    ax.axhline(0.0, color=INK, lw=0.9)
    ax.annotate(
        "0 = the grid told the truth",
        xy=(len(grids) - 1, 0),
        xytext=(0, 7),
        textcoords="offset points",
        fontsize=8,
        color=MUTED,
        ha="right",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}×{2 * n - 1}" for n in grids])
    ax.set_xlim(-0.3, len(grids) - 0.7)
    ax.margins(y=0.18)
    ax.set_xlabel("evaluation grid (n_theta × n_phi)")
    ax.set_ylabel("dB by which the grid flatters the design")
    ax.set_title(
        "The scoring grid flatters optimized designs, not unoptimized ones",
        loc="left",
        pad=10,
    )
    ax.grid(True, axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    return save(fig, "grid-bias.svg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recompute-grid-bias",
        action="store_true",
        help="rebuild the cached grid-bias measurement (slow, minutes)",
    )
    args = parser.parse_args()

    with plt.rc_context(STYLE):
        verify_against_evaluator()
        if args.recompute_grid_bias:
            print("measuring grid bias:")
            compute_grid_bias()
        written = [figure_free_lunch(), figure_recalibration(), figure_grid_bias()]

    for path in written:
        print(f"wrote {path.relative_to(REPO)} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
