"""Reference solution for t3-002: priced DOE against the declared corners.

Uses only what the BRIEF declares — the envelope *bounds*, never the
held-out points. The declared structure gives two designable worst
corners:

- clutter corner: clutter_range_km_max, rcs_dbsm_min_clutter,
  scan_deg_max, sea_state_max
- clear corner:   range_km_max, rcs_dbsm_min, 30 deg scan, sea state 0

Strategy: unit cost is independent of per-element power, so candidates
are walked cheapest-first; the first whose corners close at maximum
power gets its power bisected down to the Pd floor plus a robustness
buffer (absorbing failure-seed spread, measured in
scripts/calibrate_t3_002.py), then occupancy, availability, power and
sidelobe requirements are checked directly. The first fully compliant
design wins.

Usage: python solve.py [architecture.yaml]
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

TASK_DIR = Path(__file__).resolve().parents[1]

#: Buffer above the Pd floor at the corner, absorbing failure-seed spread
#: (seed-sensitivity in scripts/calibrate_t3_002.py measured ~0.02 across
#: seeds at the binding corner).
PD_ROBUSTNESS_BUFFER = 0.03

#: Local seed for a failure-draw sanity check; NOT one of the held-out seeds.
LOCAL_SEED = 999


def _load_spec() -> Any:
    from aedl.spec import load_task

    return load_task(TASK_DIR / "task.yaml")


def _corners(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    b = ctx["evaluation_envelope_bounds"]
    return [
        {
            "range_km": float(b["clutter_range_km_max"]),
            "rcs_dbsm": float(b["rcs_dbsm_min_clutter"]),
            "scan_deg": float(b["scan_deg_max"]),
            "sea_state": int(b["sea_state_max"]),
            "target_height_m": 5.0,
        },
        {
            "range_km": float(b["range_km_max"]),
            "rcs_dbsm": float(b["rcs_dbsm_min"]),
            "scan_deg": 30.0,
            "sea_state": 0,
            "target_height_m": 5.0,
        },
    ]


def _design_doc(
    nx: int, taper_sll: float, bits: int, pa: str, prf: float, n_p: int, tx_w: float
) -> dict[str, Any]:
    return {
        "array": {
            "nx": nx,
            "ny": nx,
            "dx_lambda": 0.5,
            "dy_lambda": 0.5,
            "taper_type": "taylor",
            "taper_sll_db": taper_sll,
            "phase_bits": bits,
        },
        "rf": {"tx_power_w_per_elem": tx_w, "pa_class": pa},
        "waveform": {"prf_hz": prf, "n_pulses": n_p},
        "digital": {
            "digitization_level": "subarray",
            "adc_enob": 10.0,
            "subarray_nx": 8,
            "subarray_ny": 8,
        },
    }


def _arch(spec: Any, doc: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    import tempfile

    from aedl.evaluators.radar_search import _load_architecture

    with tempfile.TemporaryDirectory() as td:
        sub = Path(td) / "architecture.yaml"
        sub.write_text(yaml.safe_dump(doc))
        arch, doc2, _ = _load_architecture(spec, sub)
    return arch, doc2


def _elem_cost(ctx: dict[str, Any], pa: str, bits: int) -> float:
    parts = ctx["parts"]
    return (
        float(parts["element_base_cost_usd"])
        + float(parts["pa_classes"][pa]["cost_adder_usd_per_elem"])
        + float({int(k): float(v) for k, v in parts["phase_shifter"].items()}[bits])
    )


def _requirement(spec: Any, req_id: str, bound: str) -> float:
    req = next(r for r in spec.requirements if r.id == req_id)
    value = req.min if bound == "min" else req.max
    assert value is not None, f"requirement {req_id} has no {bound} bound"
    return float(value)


def solve(verbose: bool = False) -> dict[str, Any]:
    from phased_array_systems.evaluate import evaluate_case

    from aedl.evaluators.radar_search import (
        _pam_pattern,
        _pas_point,
        _scenario,
        _timeline_occupancy,
    )

    spec = _load_spec()
    ctx = spec.context
    corners = _corners(ctx)
    tx_max = float(ctx["parts"]["tx_power_w_per_elem_max"])
    n_channels_cost = float(ctx["parts"]["adc_cost_usd_per_channel"])

    pd_floor = _requirement(spec, "detection", "min")
    target_pd = pd_floor + PD_ROBUSTNESS_BUFFER
    occ_ceiling = _requirement(spec, "frame-time", "max")
    avail_floor = _requirement(spec, "availability", "min")
    power_ceiling = _requirement(spec, "prime-power", "max")
    cost_ceiling = _requirement(spec, "unit-cost", "max")
    sll_ceiling = _requirement(spec, "sidelobes", "max")

    candidates = []
    for nx, taper_sll, bits, pa, prf, n_p in itertools.product(
        (40, 48, 56),
        (-28.0, -32.0),
        (4, 5),
        ("B", "C"),
        (2000.0, 2500.0, 3000.0),
        (16, 24, 32),
    ):
        # Cost is power-independent: elements x per-element price + ADCs.
        n_elem = nx * nx
        n_sub = (nx // 8) ** 2
        cost = _elem_cost(ctx, pa, bits) * n_elem + n_channels_cost * n_sub
        candidates.append((cost, nx, taper_sll, bits, pa, prf, n_p))
    candidates.sort()

    for cost, nx, taper_sll, bits, pa, prf, n_p in candidates:
        if cost > cost_ceiling:
            continue

        def pd_worst(
            tx_w: float,
            nx: int = nx,
            taper_sll: float = taper_sll,
            bits: int = bits,
            pa: str = pa,
            prf: float = prf,
            n_p: int = n_p,
        ) -> float:
            arch, doc = _arch(spec, _design_doc(nx, taper_sll, bits, pa, prf, n_p, tx_w))
            return min(
                _pas_point(arch, spec, corner, doc, None)["pd_achieved"] for corner in corners
            )

        if pd_worst(tx_max) < target_pd:
            continue  # geometry cannot close even at max power

        lo, hi = 0.5, tx_max
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            if pd_worst(mid) >= target_pd:
                hi = mid
            else:
                lo = mid
        tx_w = round(hi + 0.05, 2)  # small margin over the bisection cell

        doc = _design_doc(nx, taper_sll, bits, pa, prf, n_p, tx_w)
        arch, doc2 = _arch(spec, doc)

        # Remaining requirements, computed directly.
        nominal = _scenario(spec, dict(ctx["nominal_scenario"]), doc2)
        case = evaluate_case(arch, nominal)
        corner_metrics = _pas_point(arch, spec, corners[0], doc2, None)
        occupancy = _timeline_occupancy(spec, doc2, corner_metrics)
        prime_power = float(case["prime_power_w"])
        availability = float(case["array_availability"])
        sll = _pam_pattern(arch, spec, float(corners[0]["scan_deg"]), LOCAL_SEED)[1]
        pd_seeded = _pas_point(arch, spec, corners[0], doc2, LOCAL_SEED)["pd_achieved"]

        checks = {
            "occupancy": occupancy <= occ_ceiling,
            "availability": availability >= avail_floor,
            "prime_power": prime_power <= power_ceiling,
            "sll": sll <= sll_ceiling,
            "pd_seeded": pd_seeded >= pd_floor,
        }
        if verbose:
            print(
                f"nx={nx} sll={taper_sll} bits={bits} pa={pa} prf={prf:.0f} "
                f"np={n_p} tx={tx_w} -> cost={cost:.0f} occ={occupancy:.2f} "
                f"avail={availability:.4f} P={prime_power:.0f} sll={sll:.1f} "
                f"pd999={pd_seeded:.3f} {checks}"
            )
        if all(checks.values()):
            return doc

    raise SystemExit("no candidate satisfies every declared requirement")


if __name__ == "__main__":
    design = solve(verbose=True)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else TASK_DIR / "reference" / "architecture.yaml"
    out.write_text(yaml.safe_dump(design, sort_keys=False))
    print(f"\nwrote {out}")
    print(yaml.safe_dump(design, sort_keys=False))
