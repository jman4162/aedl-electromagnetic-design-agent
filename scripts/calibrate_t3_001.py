"""Calibration measurements for t3-001. Every frozen threshold reproduces from here.

Steps (run in order; each prints the numbers the task freezes):

1. rain-divergence: PAS vs opensatcom rain models over the envelope's ranges.
   Decides the envelope's maximum rain rate and confirms the clear-sky-only
   agreement requirement shape.
2. frontier: the reference DOE without ceilings, plus the naive families
   (max-aperture uniform brute force; nominal-only tuning). Decides the
   prime-power and unit-cost ceilings and the sidelobe threshold, frozen
   strictly between the compliant and naive frontiers.
3. crosscheck: PAS vs PAM gain and PAS vs opensatcom clear-sky margin for a
   spread of candidate architectures. Decides the two cross-check tolerances.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/calibrate_t3_001.py [step]
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

SOLVE = REPO / "tasks" / "t3-001-satcom-terminal-28ghz" / "reference" / "solve.py"
_solve_mod = runpy.run_path(str(SOLVE))


def step_rain() -> None:
    from opensatcom.core.models import PropagationConditions
    from opensatcom.propagation.rain import RainAttenuationP618
    from phased_array_systems.models.comms.propagation import compute_rain_loss

    f = 28e9
    print("rain mm/h | PAS 1390 km | PAS 815 km | osc el20 1390 km | osc el45 815 km")
    for rr in (0.5, 1.0, 2.0, 2.5, 4.0, 8.0):
        cond = PropagationConditions(rain_rate_mm_per_hr=rr)
        print(
            f"{rr:9.1f} | {compute_rain_loss(f, 1390e3, rr):11.2f} | "
            f"{compute_rain_loss(f, 815e3, rr):10.2f} | "
            f"{RainAttenuationP618(0.995).total_path_loss_db(f, 20.0, 1390e3, cond):16.2f} | "
            f"{RainAttenuationP618(0.995).total_path_loss_db(f, 45.0, 815e3, cond):15.2f}"
        )
    print(
        "\nThe models diverge by an order of magnitude; rain agreement cannot be a\n"
        "requirement. Envelope rain is capped at 2.5 mm/h (PAS charge ~9 dB at the\n"
        "far edge, heavy but designable); agreement is required clear-sky only."
    )


def step_frontier() -> None:
    task = _solve_mod["_load_task"]()
    corner = _solve_mod["_worst_corner"](task)
    tx_max = float(task["context"]["parts"]["tx_power_w_per_elem_max"])

    import itertools

    print("== compliant DOE (margin-feasible at the worst corner, no ceilings) ==")
    rows = []
    axes = itertools.product(
        (16, 24, 32),
        (("taylor", -22.0), ("taylor", -28.0), ("taylor", -34.0), ("chebyshev", -28.0)),
        (4, 5, 6),
        ("A", "B", "C"),
        (("subarray", 10.0), ("element", 8.0)),
    )
    for n, (taper_type, taper_sll), bits, pa_class, (digitization, enob) in axes:
        design = {
            "n": n,
            "taper_type": taper_type,
            "taper_sll_db": taper_sll,
            "phase_bits": bits,
            "pa_class": pa_class,
            "digitization_level": digitization,
            "adc_enob": enob,
        }
        arch = _solve_mod["_build_arch"](task, design, tx_max)
        if _solve_mod["_margin_at"](task, arch, corner, 999) < 0.4:
            continue
        lo, hi = 1e-3, tx_max
        for _ in range(16):
            mid = (lo + hi) / 2
            arch = _solve_mod["_build_arch"](task, design, mid)
            if _solve_mod["_margin_at"](task, arch, corner, 999) >= 0.4:
                hi = mid
            else:
                lo = mid
        arch = _solve_mod["_build_arch"](task, design, hi)
        prime_w, cost_usd = _solve_mod["_swapc"](task, arch, corner)
        rows.append((cost_usd, prime_w, design, hi))

    rows.sort(key=lambda r: r[0])
    for cost_usd, prime_w, design, tx in rows[:8]:
        print(
            f"  ${cost_usd:8.0f}  {prime_w:7.1f} W  n={design['n']} "
            f"{design['taper_type']}{design['taper_sll_db']} bits={design['phase_bits']} "
            f"pa={design['pa_class']} {design['digitization_level']} tx={tx:.3f} W"
        )
    cheapest = rows[0]
    print(f"\ncheapest margin-feasible: ${cheapest[0]:.0f}, {cheapest[1]:.1f} W")

    # SLL of the best few tapered designs at the worst corner, and of the
    # naive uniform families, to place the sidelobe threshold between them.
    print("\n== sidelobes at the worst corner (full-pattern, seeded failures) ==")
    for cost_usd, _prime_w, design, tx in rows[:4]:
        arch = _solve_mod["_build_arch"](task, design, tx)
        sll = _solve_mod["_sll_at"](task, arch, corner["scan_deg"], 999)
        print(f"  tapered  ${cost_usd:8.0f}: sll {sll:7.2f} dB")

    print("\n== naive family: max aperture, uniform, coarse bits, cheap PA ==")
    for n, bits in ((32, 4), (32, 5), (40, 4)):
        design = {
            "n": n,
            "taper_type": "uniform",
            "taper_sll_db": -13.0,
            "phase_bits": bits,
            "pa_class": "A",
            "digitization_level": "analog",
            "adc_enob": 8.0,
        }
        arch = _solve_mod["_build_arch"](task, design, tx_max)
        margin = _solve_mod["_margin_at"](task, arch, corner, 999)
        if margin < 0.4:
            print(f"  n={n} bits={bits}: cannot close the worst corner even at max power")
            continue
        lo, hi = 1e-3, tx_max
        for _ in range(16):
            mid = (lo + hi) / 2
            arch = _solve_mod["_build_arch"](task, design, mid)
            if _solve_mod["_margin_at"](task, arch, corner, 999) >= 0.4:
                hi = mid
            else:
                lo = mid
        arch = _solve_mod["_build_arch"](task, design, hi)
        prime_w, cost_usd = _solve_mod["_swapc"](task, arch, corner)
        sll = _solve_mod["_sll_at"](task, arch, corner["scan_deg"], 999)
        print(
            f"  n={n} bits={bits}: ${cost_usd:8.0f}, {prime_w:7.1f} W, "
            f"sll {sll:7.2f} dB, tx={hi:.3f} W"
        )


def step_crosscheck() -> None:
    import aedl.evaluators  # noqa: F401
    from aedl.registry import get_evaluator
    from aedl.spec import find_task

    spec = find_task(REPO / "tasks", "t3-001")
    evaluate = get_evaluator("satcom_terminal")

    import tempfile

    import yaml

    designs = {
        "n24-taylor28-b5-B": {
            "array": {
                "nx": 24,
                "ny": 24,
                "dx_lambda": 0.5,
                "dy_lambda": 0.5,
                "taper_type": "taylor",
                "taper_sll_db": -28.0,
                "phase_bits": 5,
            },
            "rf": {"tx_power_w_per_elem": 0.8, "pa_class": "B"},
            "digital": {
                "digitization_level": "subarray",
                "adc_enob": 10.0,
                "subarray_nx": 8,
                "subarray_ny": 8,
            },
        },
        "n32-cheb28-b6-C": {
            "array": {
                "nx": 32,
                "ny": 32,
                "dx_lambda": 0.5,
                "dy_lambda": 0.5,
                "taper_type": "chebyshev",
                "taper_sll_db": -28.0,
                "phase_bits": 6,
            },
            "rf": {"tx_power_w_per_elem": 0.35, "pa_class": "C"},
            "digital": {
                "digitization_level": "element",
                "adc_enob": 8.0,
                "subarray_nx": 8,
                "subarray_ny": 8,
            },
        },
        "n16-taylor22-b4-A": {
            "array": {
                "nx": 16,
                "ny": 16,
                "dx_lambda": 0.5,
                "dy_lambda": 0.5,
                "taper_type": "taylor",
                "taper_sll_db": -22.0,
                "phase_bits": 4,
            },
            "rf": {"tx_power_w_per_elem": 1.8, "pa_class": "A"},
            "digital": {
                "digitization_level": "analog",
                "adc_enob": 8.0,
                "subarray_nx": 8,
                "subarray_ny": 8,
            },
        },
    }
    print(f"{'design':<22}{'gain-crosscheck':>16}{'clearsky-agree':>15}{'sll(worst)':>12}")
    gains, agrees = [], []
    for name, doc in designs.items():
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(doc, f)
            path = Path(f.name)
        result = evaluate(spec, path)
        m = result.info["metrics"]
        gains.append(m["crosscheck_gain_disagreement_db"])
        agrees.append(m["crosscheck_clearsky_margin_disagreement_db"])
        print(
            f"{name:<22}{m['crosscheck_gain_disagreement_db']:>16.3f}"
            f"{m['crosscheck_clearsky_margin_disagreement_db']:>15.3f}"
            f"{m['worst_case_pattern_sll_db']:>12.2f}"
        )
        path.unlink()
    print(
        f"\nmax gain disagreement {max(gains):.3f} dB; max clear-sky disagreement "
        f"{max(agrees):.3f} dB. Freeze tolerances with ~3x headroom."
    )


STEPS = {"rain": step_rain, "frontier": step_frontier, "crosscheck": step_crosscheck}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else None
    for name, fn in STEPS.items():
        if which in (None, name):
            print(f"\n===== {name} =====")
            fn()
