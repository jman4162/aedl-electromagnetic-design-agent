"""Calibration for t3-002 (X-band maritime search radar).

Every threshold in the task is frozen from a measurement this script
prints. Steps:

1. clutter    — PAS sea sigma-0 vs published X-band ranges over the
                envelope's grazing angles and sea states, plus the SCNR
                landscape that decides where clutter-limited detection is
                feasible at all (PAS models no MTI/Doppler, so the
                envelope must live where noncoherent detection works).
2. mc         — Monte Carlo vs analytic Pd agreement across the operating
                region; freezes the pd-crosscheck tolerance.
3. reliability— behavior of prob_meeting_spec / availability across PA
                classes and power levels; decides the availability metric
                and floor.
4. frontier   — reference DOE without ceilings + naive families; freezes
                the power/cost ceilings and requirement thresholds
                strictly between the compliant and naive frontiers.

Run: PYTHONPATH=src python scripts/calibrate_t3_002.py [step]
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

TASK = Path(__file__).resolve().parent.parent / "tasks" / "t3-002-xband-search-radar"


def _spec() -> Any:
    from aedl.spec import load_task

    return load_task(TASK / "task.yaml")


def _evaluate(design: dict[str, Any]) -> Any:
    import tempfile

    import aedl.evaluators  # noqa: F401
    from aedl.registry import get_evaluator

    spec = _spec()
    with tempfile.TemporaryDirectory() as td:
        sub = Path(td) / "architecture.yaml"
        sub.write_text(yaml.safe_dump(design))
        return get_evaluator(spec.evaluator)(spec, sub)


def step_clutter() -> None:
    """PAS sea sigma-0 vs published X-band values; SCNR feasibility map."""
    from phased_array_systems.models.radar.clutter import sea_clutter_sigma0

    print("== sea sigma-0 (dB) at 9.5 GHz, HH ==")
    print("Published X-band comparisons (Nathanson tables / GIT model")
    print("summaries; see calibration notes in the task README): at ~0.1-1 deg")
    print("grazing, sigma0 spans roughly -55..-35 dB from SS2 to SS5.")
    print(f"{'grazing':>8} " + " ".join(f"SS{s:<4d}" for s in (2, 3, 4, 5)))
    for grazing in (0.1, 0.3, 1.0, 3.0, 10.0):
        row = [sea_clutter_sigma0(s, grazing, 9.5e9, polarization="HH") for s in (2, 3, 4, 5)]
        print(f"{grazing:8.2f} " + " ".join(f"{v:6.1f}" for v in row))

    print("\n== SCNR landscape (32x32 taylor -30, 10 W/elem, 24 pulses, surface target 5 m) ==")
    from phased_array_systems.models.antenna import PhasedArrayAdapter
    from phased_array_systems.models.radar import RadarModel

    from aedl.evaluators.radar_search import _load_architecture, _scenario

    design = {
        "array": {
            "nx": 32,
            "ny": 32,
            "dx_lambda": 0.5,
            "dy_lambda": 0.5,
            "taper_type": "taylor",
            "taper_sll_db": -30.0,
            "phase_bits": 5,
        },
        "rf": {"tx_power_w_per_elem": 10.0, "pa_class": "B"},
        "waveform": {"prf_hz": 2500.0, "n_pulses": 24},
        "digital": {
            "digitization_level": "subarray",
            "adc_enob": 10.0,
            "subarray_nx": 8,
            "subarray_ny": 8,
        },
    }
    import tempfile

    spec = _spec()
    with tempfile.TemporaryDirectory() as td:
        sub = Path(td) / "architecture.yaml"
        sub.write_text(yaml.safe_dump(design))
        arch, doc, _ = _load_architecture(spec, sub)

    print(f"{'range':>6} {'SS':>3} {'scnr_db':>8} {'snr_db':>7} {'scr_db':>7} {'pd':>7}")
    for range_km in (10.0, 14.0, 18.0, 22.0, 26.0):
        for ss in (0, 2, 3, 4):
            point = {
                "range_km": range_km,
                "rcs_dbsm": 0.0,
                "scan_deg": 0.0,
                "sea_state": ss,
                "target_height_m": 5.0,
            }
            scenario = _scenario(spec, point, doc)
            antenna = PhasedArrayAdapter(use_analytical_fallback=False).evaluate(arch, scenario, {})
            radar = RadarModel().evaluate(arch, scenario, dict(antenna))
            print(
                f"{range_km:6.0f} {ss:3d} {radar.get('scnr_db', float('nan')):8.1f} "
                f"{radar['snr_single_pulse_db']:7.1f} "
                f"{radar.get('scr_db', float('nan')):7.1f} {radar['pd_achieved']:7.3f}"
            )


def step_mc() -> None:
    """MC vs analytic Pd across the operating region."""
    from phased_array_systems.models.radar.cfar import cfar_loss_db
    from phased_array_systems.models.radar.detection import compute_pd_from_snr

    from aedl.evaluators.radar_mc import simulate_pd_swerling1_cfar

    loss = cfar_loss_db("CA", 16, pfa=1e-6)
    print(f"analytic CA-CFAR loss (16 cells): {loss:.3f} dB")
    print(f"{'scnr':>5} {'n_p':>4} {'mc_pd':>7} {'pas_pd':>7} {'diff':>7}")
    worst_high_pd = 0.0
    for scnr in (10.0, 14.0, 18.0, 22.0, 26.0):
        for n_p in (4, 8, 16, 32, 64):
            mc = simulate_pd_swerling1_cfar(scnr, 1e-6, n_p, 16)
            pas = compute_pd_from_snr(scnr - loss, 1e-6, swerling=1, n_pulses=n_p)
            diff = abs(mc - pas)
            marker = ""
            if pas >= 0.85:
                worst_high_pd = max(worst_high_pd, diff)
                marker = " *"
            print(f"{scnr:5.0f} {n_p:4d} {mc:7.4f} {pas:7.4f} {diff:7.4f}{marker}")
    print(f"\nworst |mc - analytic| where analytic Pd >= 0.85: {worst_high_pd:.4f}")
    print("(PAS is one-sidedly conservative at multi-pulse: its CFAR loss is")
    print("calibrated at n=1 and does not shrink with integrated reference")
    print("cells. Freeze the tolerance above this measured worst case.)")


def step_reliability() -> None:
    """prob_meeting_spec / availability across PA classes and power."""
    print(f"{'pa':>3} {'tx_w':>5} {'Tj_C':>6} {'trm_mtbf':>9} {'avail':>7} {'p_spec':>7}")
    for pa_class in ("A", "B", "C"):
        for tx_w in (4.0, 8.0, 12.0):
            design = {
                "array": {
                    "nx": 32,
                    "ny": 32,
                    "dx_lambda": 0.5,
                    "dy_lambda": 0.5,
                    "taper_type": "taylor",
                    "taper_sll_db": -30.0,
                    "phase_bits": 5,
                },
                "rf": {"tx_power_w_per_elem": tx_w, "pa_class": pa_class},
                "waveform": {"prf_hz": 2000.0, "n_pulses": 16},
                "digital": {
                    "digitization_level": "subarray",
                    "adc_enob": 10.0,
                    "subarray_nx": 8,
                    "subarray_ny": 8,
                },
            }
            import tempfile

            from phased_array_systems.evaluate import evaluate_case

            from aedl.evaluators.radar_search import _load_architecture, _scenario

            spec = _spec()
            with tempfile.TemporaryDirectory() as td:
                sub = Path(td) / "architecture.yaml"
                sub.write_text(yaml.safe_dump(design))
                arch, doc, _ = _load_architecture(spec, sub)
            nominal = _scenario(spec, dict(spec.context["nominal_scenario"]), doc)
            case = evaluate_case(arch, nominal)
            print(
                f"{pa_class:>3} {tx_w:5.1f} {case.get('junction_temp_c', float('nan')):6.1f} "
                f"{case['trm_mtbf_hours']:9.0f} {case['array_availability']:7.4f} "
                f"{case['prob_meeting_spec']:7.4f}"
            )


def step_frontier() -> None:
    """Reference DOE + naive families -> ceilings and thresholds."""
    designs: list[tuple[str, dict[str, Any]]] = []
    for nx, taper_sll, bits, pa_class, prf, n_p, tx_w in itertools.product(
        (32, 40, 48, 56),
        (-30.0,),
        (5,),
        ("B", "C"),
        (2500.0,),
        (16, 24, 32),
        (8.0, 12.0),
    ):
        designs.append(
            (
                f"n{nx}_sll{taper_sll:.0f}_b{bits}_{pa_class}_prf{prf:.0f}_np{n_p}_tx{tx_w:.0f}",
                {
                    "array": {
                        "nx": nx,
                        "ny": nx,
                        "dx_lambda": 0.5,
                        "dy_lambda": 0.5,
                        "taper_type": "taylor",
                        "taper_sll_db": taper_sll,
                        "phase_bits": bits,
                    },
                    "rf": {"tx_power_w_per_elem": tx_w, "pa_class": pa_class},
                    "waveform": {"prf_hz": prf, "n_pulses": n_p},
                    "digital": {
                        "digitization_level": "subarray",
                        "adc_enob": 10.0,
                        "subarray_nx": 8,
                        "subarray_ny": 8,
                    },
                },
            )
        )

    print(f"scoring {len(designs)} designs against the draft envelope ...")
    rows = []
    for name, design in designs:
        result = _evaluate(design)
        m = result.info["metrics"]
        rows.append(
            (
                name,
                m["worst_case_pd"],
                m["worst_case_timeline_occupancy"],
                m["array_availability"],
                m["prime_power_w"],
                m["unit_cost_usd"],
                m["worst_case_pattern_sll_db"],
            )
        )
        print(
            f"{name:44s} pd={rows[-1][1]:.3f} occ={rows[-1][2]:.2f} "
            f"avail={rows[-1][3]:.4f} P={rows[-1][4]:6.0f} $={rows[-1][5]:8.0f} "
            f"sll={rows[-1][6]:6.1f}"
        )

    feasible = [r for r in rows if r[1] >= 0.7 and r[2] <= 1.0 and r[3] >= 0.995]
    print(f"\n{len(feasible)}/{len(rows)} meet pd>=0.7, occupancy<=1, avail>=0.995")
    if feasible:
        print("cheapest by cost:")
        for r in sorted(feasible, key=lambda r: r[5])[:5]:
            print(f"  {r[0]:44s} P={r[4]:6.0f} $={r[5]:8.0f} avail={r[3]:.4f}")


if __name__ == "__main__":
    steps = {
        "clutter": step_clutter,
        "mc": step_mc,
        "reliability": step_reliability,
        "frontier": step_frontier,
    }
    wanted = sys.argv[1:] or list(steps)
    for name in wanted:
        print(f"\n================ {name} ================")
        steps[name]()
