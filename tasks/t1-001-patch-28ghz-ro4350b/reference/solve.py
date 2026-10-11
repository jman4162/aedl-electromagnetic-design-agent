"""Reference solution for t1-001: grid search over laminate, feed impedance,
and synthesis target, scored by the evaluator's own analytic tier.

Uses only what the BRIEF declares: the stock thicknesses, the band, the
footprint limit, and the documented model. For each stock thickness and
feed impedance it scans the synthesis target frequency (the frequency
antenna-cad sizes the patch for, which is not where the patch ends up
resonating once dispersion is counted), keeps designs inside the footprint,
and returns the one with the lowest worst-case in-band |S11|.

Usage: python solve.py [design.yaml]
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

TASK_DIR = Path(__file__).resolve().parents[1]

IMPEDANCES_OHM = (40.0, 50.0, 60.0, 70.0, 80.0)
#: Coarse synthesis-target grid, then a fine pass around the best point.
COARSE_GHZ = [27.5 + 0.1 * i for i in range(31)]
FINE_STEP_GHZ = 0.01


def design_doc(fc_ghz: float, z_ohm: float, h_mm: float) -> dict[str, Any]:
    return {
        "design": {"name": "t1_001_reference"},
        "requirements": {
            "center_frequency": f"{fc_ghz:.2f} GHz",
            "impedance": f"{z_ohm:g} ohm",
            "substrate": "RO4350B",
            "substrate_height": f"{h_mm:g} mm",
            "polarization": "linear",
        },
        "element": {"type": "rectangular_patch"},
    }


def score(spec: Any, doc: dict[str, Any]) -> dict[str, float]:
    from aedl.evaluators.patch_element import element_metrics

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "design.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False))
        metrics, _ = element_metrics(spec, path)
    return metrics


def solve(verbose: bool = False) -> dict[str, Any]:
    from aedl.spec import load_task

    spec = load_task(TASK_DIR / "task.yaml")
    ctx = spec.context
    limit = float(ctx["footprint_mm_max"])
    lo, hi = (float(x) for x in ctx["band_ghz"])

    def feasible_worst(doc: dict[str, Any]) -> float:
        m = score(spec, doc)
        if m["footprint_mm"] > limit or not lo <= m["f_res_ghz"] <= hi:
            return float("inf")
        return m["s11_max_in_band_db"]

    best: tuple[float, dict[str, Any]] | None = None
    for h_mm in ctx["substrate"]["stock_heights_mm"]:
        for z in IMPEDANCES_OHM:
            coarse = [(feasible_worst(design_doc(fc, z, h_mm)), fc) for fc in COARSE_GHZ]
            worst, fc0 = min(coarse)
            if worst == float("inf"):
                continue
            fine = [fc0 + FINE_STEP_GHZ * k for k in range(-10, 11)]
            worst, fc = min((feasible_worst(design_doc(f, z, h_mm)), f) for f in fine)
            if verbose:
                print(f"h={h_mm} mm z={z:g} ohm fc={fc:.2f} GHz -> worst in-band {worst:.2f} dB")
            if best is None or worst < best[0]:
                best = (worst, design_doc(fc, z, h_mm))
    if best is None:
        raise SystemExit("no stock laminate meets the footprint with f_res in band")
    return best[1]


if __name__ == "__main__":
    design = solve(verbose=True)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else TASK_DIR / "reference" / "design.yaml"
    out.write_text(yaml.safe_dump(design, sort_keys=False))
    print(f"\nwrote {out}")
    print(yaml.safe_dump(design, sort_keys=False))
