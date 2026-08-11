"""Reference solution for t3-001: priced DOE + power bisection at the worst corner.

The search uses only what the brief declares: the parts table, the ceilings and
requirement thresholds in task.yaml, and the declared envelope *bounds* (it
never reads the held-out envelope points or seeds). Designing from the brief is
part of the claim the task makes: it is solvable from the information an agent
receives.

Stage 1 enumerates the discrete axes (aperture size, taper, phase bits, PA
class, digitization) and, for each combination, bisects per-element transmit
power to a small margin buffer at the declared worst corner (max scan, min
elevation, max range, max rain, max sky temperature, the stated failure rate
with a local seed that is deliberately not an envelope seed). Stage 2 filters
by every declared requirement, checking the sidelobe requirement with a
full-pattern recompute for the surviving candidates only, and returns the
cheapest compliant design, tie-broken by prime power.

Plain itertools over ~200 combinations; deliberately not NSGA-II. The search
is transparent, and its technique stays out of the brief.

Usage: python solve.py [architecture.yaml]
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

TASK = Path(__file__).resolve().parents[1] / "task.yaml"

MARGIN_BUFFER_DB = 0.4  # absorbs held-out points being off-corner + seed draws
SLL_ROBUSTNESS_DB = 1.5  # failure-seed draws move worst-corner SLL by ~1.1 dB
LOCAL_SEED = 999  # not one of the envelope seeds, by construction


def _load_task() -> dict:
    return yaml.safe_load(TASK.read_text())


def _worst_corner(task: dict) -> dict:
    bounds = task["context"]["evaluation_envelope_bounds"]
    return {
        "scan_deg": float(bounds["scan_angle_deg_max"]),
        "elevation_deg": float(bounds["elevation_deg_min"]),
        "range_km": float(bounds["slant_range_km_max"]),
        "rain_mmh": float(bounds["rain_rate_mmh_max"]),
        "sky_temp_k": float(bounds["sky_temp_k_max"])
        if "sky_temp_k_max" in bounds
        else float(bounds["sky_noise_temp_k_max"]),
    }


def _build_arch(task: dict, design: dict, tx_power: float) -> Any:
    from phased_array_systems.architecture import (
        Architecture,
        ArrayConfig,
        CostConfig,
        DigitalConfig,
        RFChainConfig,
    )

    ctx = task["context"]
    parts = ctx["parts"]
    pa_class = parts["pa_classes"][design["pa_class"]]
    cost_per_elem = (
        float(parts["element_base_cost_usd"])
        + float(pa_class["cost_adder_usd_per_elem"])
        + float(parts["phase_shifter"][design["phase_bits"]])
    )
    return Architecture(
        array=ArrayConfig(
            nx=design["n"],
            ny=design["n"],
            dx_lambda=0.5,
            dy_lambda=0.5,
            taper_type=design["taper_type"],
            taper_sll_db=design["taper_sll_db"],
            phase_bits=design["phase_bits"],
            scan_limit_deg=float(ctx["evaluation_envelope_bounds"]["scan_angle_deg_max"]),
            element_cos_exp=float(ctx["element"]["q"]),
            max_subarray_nx=8,
            max_subarray_ny=8,
        ),
        rf=RFChainConfig(
            tx_power_w_per_elem=tx_power,
            pa_efficiency=float(pa_class["pa_efficiency"]),
            noise_figure_db=float(ctx["link"]["satellite_noise_figure_db"]),
            feed_loss_db=float(parts["feed_loss_db"]),
            rx_power_w_per_elem=float(parts["rx_power_w_per_elem"]),
        ),
        cost=CostConfig(cost_per_elem_usd=cost_per_elem, nre_usd=0.0),
        digital=DigitalConfig(
            digitization_level=design["digitization_level"],
            adc_enob=design["adc_enob"],
        ),
    )


def _margin_at(task: dict, arch: Any, point: dict, seed: int | None) -> float:
    from phased_array_systems.models.antenna import PhasedArrayAdapter
    from phased_array_systems.models.comms import CommsLinkModel
    from phased_array_systems.scenarios import CommsLinkScenario

    link = task["context"]["link"]
    scenario = CommsLinkScenario(
        freq_hz=float(link["frequency_ghz"]) * 1e9,
        bandwidth_hz=float(link["bandwidth_mhz"]) * 1e6,
        range_m=float(point["range_km"]) * 1e3,
        required_snr_db=float(link["required_snr_db"]),
        scan_angle_deg=float(point["scan_deg"]),
        rx_antenna_gain_db=float(link["satellite_rx_gain_db"]),
        rx_noise_temp_k=float(point["sky_temp_k"]),
        rain_rate_mmh=float(point["rain_mmh"]),
        elevation_deg=float(point["elevation_deg"]),
    )
    ctx: dict[str, Any] = {}
    if seed is not None:
        ctx["failure_rate"] = float(
            task["context"]["evaluation_envelope_bounds"]["element_failure_rate"]
        )
        ctx["meta.seed"] = seed
    antenna = PhasedArrayAdapter(use_analytical_fallback=False).evaluate(arch, scenario, ctx)
    return float(CommsLinkModel().evaluate(arch, scenario, dict(antenna))["link_margin_db"])


def _swapc(task: dict, arch: Any, point: dict) -> tuple[float, float]:
    from phased_array_systems.models.antenna import PhasedArrayAdapter
    from phased_array_systems.models.swapc import CostModel, PowerModel
    from phased_array_systems.scenarios import CommsLinkScenario

    link = task["context"]["link"]
    scenario = CommsLinkScenario(
        freq_hz=float(link["frequency_ghz"]) * 1e9,
        bandwidth_hz=float(link["bandwidth_mhz"]) * 1e6,
        range_m=float(point["range_km"]) * 1e3,
        required_snr_db=float(link["required_snr_db"]),
        scan_angle_deg=float(point["scan_deg"]),
    )
    antenna = PhasedArrayAdapter(use_analytical_fallback=False).evaluate(arch, scenario, {})
    power = PowerModel().evaluate(arch, scenario, dict(antenna))
    cost = CostModel().evaluate(arch, scenario, dict(antenna))
    parts = task["context"]["parts"]
    n_channels = int(getattr(arch, "n_digital_channels", arch.array.nx * arch.array.ny))
    unit_cost = float(cost["total_cost_usd"]) + float(parts["adc_cost_usd_per_channel"]) * float(
        n_channels
    )
    return float(power["prime_power_w"]), unit_cost


def _sll_at(task: dict, arch: Any, scan_deg: float, seed: int | None) -> float:
    """Full-pattern sidelobe recompute, mirroring what the evaluator checks."""
    import phased_array as pa

    from aedl.evaluators.array_pattern import _peak_sidelobe_db

    ctx = task["context"]
    freq_hz = float(ctx["link"]["frequency_ghz"]) * 1e9
    wavelength = 299_792_458.0 / freq_hz
    array = arch.array
    geom = pa.create_rectangular_array(
        array.nx, array.ny, dx=array.dx_lambda, dy=array.dy_lambda, wavelength=wavelength
    )
    k = pa.wavelength_to_k(wavelength)
    if array.taper_type == "taylor":
        taper = pa.taylor_taper_2d(array.nx, array.ny, sidelobe_dB=array.taper_sll_db)
    elif array.taper_type == "chebyshev":
        taper = pa.chebyshev_taper_2d(array.nx, array.ny, sidelobe_dB=array.taper_sll_db)
    else:
        taper = np.ones((array.nx, array.ny))
    weights = taper.ravel().astype(complex)
    weights *= pa.steering_vector(k, geom.x, geom.y, scan_deg, 0.0)
    weights = pa.quantize_phase(weights, n_bits=array.phase_bits)
    if seed is not None:
        rate = float(ctx["evaluation_envelope_bounds"]["element_failure_rate"])
        weights, _ = pa.simulate_element_failures(weights, rate, seed=seed)
    _, _, pattern_db = pa.compute_full_pattern(
        geom.x,
        geom.y,
        weights,
        k,
        n_theta=361,
        n_phi=721,
        theta_range=(0.0, np.pi),
        element_pattern_func=pa.element_pattern,
        cos_exp_theta=float(ctx["element"]["q"]),
    )
    return float(_peak_sidelobe_db(pattern_db))


def _requirement(task: dict, req_id: str) -> float:
    for req in task["requirements"]:
        if req["id"] == req_id:
            return float(req.get("max", req.get("min")))
    raise KeyError(req_id)


def solve(verbose: bool = False) -> dict:
    task = _load_task()
    corner = _worst_corner(task)
    parts = task["context"]["parts"]
    tx_max = float(parts["tx_power_w_per_elem_max"])

    power_ceiling = _requirement(task, "prime-power")
    cost_ceiling = _requirement(task, "unit-cost")
    sll_ceiling = _requirement(task, "sidelobes")

    designs = []
    axes = itertools.product(
        (16, 24, 32),  # n
        (("taylor", -22.0), ("taylor", -28.0), ("taylor", -34.0), ("chebyshev", -28.0)),
        (4, 5, 6),  # phase bits
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
        # Feasibility probe at max power, then bisect down to the buffer.
        arch = _build_arch(task, design, tx_max)
        if _margin_at(task, arch, corner, LOCAL_SEED) < MARGIN_BUFFER_DB:
            continue
        lo, hi = 1e-3, tx_max
        for _ in range(18):
            mid = (lo + hi) / 2
            arch = _build_arch(task, design, mid)
            if _margin_at(task, arch, corner, LOCAL_SEED) >= MARGIN_BUFFER_DB:
                hi = mid
            else:
                lo = mid
        design["tx_power_w_per_elem"] = round(hi, 4)
        arch = _build_arch(task, design, hi)
        prime_w, cost_usd = _swapc(task, arch, corner)
        if prime_w > power_ceiling or cost_usd > cost_ceiling:
            continue
        designs.append((cost_usd, prime_w, design))
        if verbose:
            print(
                f"feasible: n={n} {taper_type}{taper_sll} bits={bits} pa={pa_class} "
                f"{digitization} tx={design['tx_power_w_per_elem']}W "
                f"-> {prime_w:.0f} W, ${cost_usd:.0f}"
            )

    if not designs:
        raise RuntimeError("no design satisfies the declared requirements")

    # Cheapest-first, then check the expensive sidelobe requirement.
    for cost_usd, _prime_w, design in sorted(designs, key=lambda d: (d[0], d[1])):
        arch = _build_arch(task, design, design["tx_power_w_per_elem"])
        sll = _sll_at(task, arch, corner["scan_deg"], LOCAL_SEED)
        # The evaluator scores the max over its held-out failure seeds, which
        # sits above any single draw; carry a buffer so one unlucky draw does
        # not flip the requirement.
        needed = sll_ceiling - SLL_ROBUSTNESS_DB
        if verbose:
            print(
                f"sll check: ${cost_usd:.0f} {design['taper_type']}{design['taper_sll_db']} "
                f"bits={design['phase_bits']} -> {sll:.2f} dB (need <= {needed})"
            )
        if sll <= needed:
            return {
                "array": {
                    "nx": design["n"],
                    "ny": design["n"],
                    "dx_lambda": 0.5,
                    "dy_lambda": 0.5,
                    "taper_type": design["taper_type"],
                    "taper_sll_db": design["taper_sll_db"],
                    "phase_bits": design["phase_bits"],
                },
                "rf": {
                    "tx_power_w_per_elem": design["tx_power_w_per_elem"],
                    "pa_class": design["pa_class"],
                },
                "digital": {
                    "digitization_level": design["digitization_level"],
                    "adc_enob": design["adc_enob"],
                    "subarray_nx": 8,
                    "subarray_ny": 8,
                },
            }
    raise RuntimeError("no ceiling-feasible design meets the sidelobe requirement")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("architecture.yaml")
    architecture = solve(verbose=True)
    out.write_text(yaml.safe_dump(architecture, sort_keys=False))
    print(f"wrote {out}")
