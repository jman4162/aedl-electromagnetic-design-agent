"""Reference solution for t2-001: coordinate descent over the 2-bit phase states.

Direct rounding of the ideal steering phases onto the four available states
points the beam correctly but leaves a roughly -10.7 dB sidelobe floor. The
residual quantization error is what radiates, and with only four phase states
per element the way to suppress it is to choose the states jointly rather than
independently.

This does one pass of greedy coordinate descent: repeatedly, for each working
element, try all four states and keep the one that most reduces the peak
sidelobe. The field is updated incrementally (rank-one per element), so a sweep
costs one matrix-vector product per candidate rather than a full pattern
recomputation. It converges in three or four sweeps to about -16.7 dB.

Note the failed elements are excluded from optimization but included as zeros in
the field, which is what the hardware actually does.

Usage: python solve.py [output.npz]
"""

import sys
from pathlib import Path

import numpy as np
import phased_array as pa

C0 = 299_792_458.0
TARGET_THETA_DEG = 27.0
TARGET_PHI_DEG = 10.0
EXCLUSION_DEG = 8.0
FAILED = np.array([7, 23, 42, 58, 77, 96, 113, 128, 150, 171, 199, 220, 245])


def solve(max_sweeps: int = 8, seed: int = 0) -> np.ndarray:
    wavelength = C0 / 10e9
    geom = pa.create_rectangular_array(16, 16, dx=0.5, dy=0.5, wavelength=wavelength)
    k = pa.wavelength_to_k(wavelength)
    n = geom.n_elements

    step = np.pi / 2
    states = np.exp(1j * np.arange(4) * step)
    active = np.setdiff1d(np.arange(n), FAILED)

    t0 = np.radians(TARGET_THETA_DEG)
    p0 = np.radians(TARGET_PHI_DEG)

    # Sidelobe sample directions over the forward hemisphere. Coarser than the
    # evaluator's grid on purpose: fine enough to steer the optimization, cheap
    # enough to sweep repeatedly.
    theta = np.linspace(0.0, np.pi / 2, 151)
    phi = np.linspace(0.0, 2 * np.pi, 201)
    tg, pg = np.meshgrid(theta, phi, indexing="ij")
    sep = np.degrees(
        np.arccos(
            np.clip(
                np.sin(tg) * np.sin(t0) * np.cos(pg - p0) + np.cos(tg) * np.cos(t0),
                -1.0, 1.0,
            )
        )
    )
    outside = (sep > EXCLUSION_DEG).ravel()
    u = (np.sin(tg) * np.cos(pg)).ravel()
    v = (np.sin(tg) * np.sin(pg)).ravel()
    element = np.cos(tg).ravel()  # cos^1 element pattern
    steering_matrix = (
        np.exp(1j * k * (np.outer(u, geom.x) + np.outer(v, geom.y))) * element[:, None]
    )[outside]
    boresight = np.exp(
        1j * k * (np.sin(t0) * np.cos(p0) * geom.x + np.sin(t0) * np.sin(p0) * geom.y)
    ) * np.cos(t0)

    ideal_phase = np.angle(
        pa.steering_vector(k, geom.x, geom.y, TARGET_THETA_DEG, TARGET_PHI_DEG)
    )
    weights = states[np.round(ideal_phase / step).astype(int) % 4]
    weights[FAILED] = 0.0

    field = steering_matrix @ weights
    main = boresight @ weights

    def sidelobe_db(f, m):
        return 20.0 * np.log10(np.abs(f).max() / abs(m))

    best = sidelobe_db(field, main)
    rng = np.random.default_rng(seed)
    for _ in range(max_sweeps):
        changed = 0
        for e in rng.permutation(active):
            current = weights[e]
            chosen, chosen_score = current, best
            for state in states:
                if state == current:
                    continue
                delta = state - current
                score = sidelobe_db(field + steering_matrix[:, e] * delta,
                                    main + boresight[e] * delta)
                if score < chosen_score - 1e-9:
                    chosen, chosen_score = state, score
            if chosen is not current and chosen != current:
                delta = chosen - current
                field += steering_matrix[:, e] * delta
                main += boresight[e] * delta
                weights[e] = chosen
                best = chosen_score
                changed += 1
        if changed == 0:
            break

    # Submit the weights that would be programmed into working hardware; the
    # evaluator applies the failures itself.
    weights[FAILED] = 1.0 + 0j
    return weights


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reference.npz")
    np.savez(out, weights=solve())
    print(f"wrote {out}")
