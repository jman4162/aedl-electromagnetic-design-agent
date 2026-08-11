"""Deterministic evaluator for tier-3 SATCOM terminal architecture tasks.

The submission is an ``architecture.yaml`` naming discrete and continuous
architecture choices. The evaluator prices the design from the task's parts
table (efficiency, noise figure, and costs are never accepted from the
submission), evaluates the link worst-case over a held-out envelope of
scenario points and element-failure seeds, and cross-checks the two claims
that matter with independent code:

- pattern gain and sidelobes are recomputed by full pattern integration with
  ``phased-array-modeling`` (quantization and seeded failures applied), using
  the local-maximum sidelobe extraction from :mod:`aedl.evaluators.array_pattern`;
- the link margin is recomputed with ``opensatcom``, a link-budget codebase
  the design model shares no code with. Its rain model differs from
  phased-array-systems' by construction (P.618 slant path vs a terrestrial
  effective-path model), so rain-path agreement is deliberately not a
  requirement: agreement is required clear-sky, and both models must
  independently close the link.

Metrics produced (requirements reference these by name):

- ``worst_case_link_margin_db``: min PAS margin over envelope x failure seeds
- ``worst_case_pattern_sll_db``: max full-pattern SLL over seeds at the
  binding point's scan angle
- ``opensatcom_worst_case_margin_db``: min opensatcom margin over the envelope
- ``crosscheck_clearsky_margin_disagreement_db``: |PAS - opensatcom| at the
  nominal clear-sky scenario
- ``crosscheck_gain_disagreement_db``: |PAS g_peak - PAM full-pattern
  directivity| at the nominal scan
- ``prime_power_w``, ``unit_cost_usd``: priced from the parts table
- ``grating_margin_lambda``: max safe spacing minus the design's spacing
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from aedl.registry import register_evaluator
from aedl.result import EvaluationResult, RequirementResult
from aedl.spec import TaskSpec

# Keys the submission must not contain anywhere: these are priced or pinned by
# the task, and accepting them would reopen the free-lunch axes.
_FORBIDDEN_KEYS = {
    "pa_efficiency",
    "noise_figure_db",
    "feed_loss_db",
    "cost",
    "cost_per_elem_usd",
    "cost_usd",
    "rx_power_w_per_elem",
}

_ARRAY_KEYS = {"nx", "ny", "dx_lambda", "dy_lambda", "taper_type", "taper_sll_db", "phase_bits"}


def _num(metrics: dict[str, Any], key: str, default: float | None = None) -> float:
    """A metric that must be numeric; TypeError names the offender otherwise."""
    value = metrics.get(key, default)
    if not isinstance(value, (int, float)):
        raise TypeError(f"metric {key!r} is {type(value).__name__}, expected a number")
    return float(value)


_RF_KEYS = {"tx_power_w_per_elem", "pa_class"}
_DIGITAL_KEYS = {"digitization_level", "adc_enob", "subarray_nx", "subarray_ny"}


def _reject_forbidden(doc: Any, path: str = "") -> None:
    if isinstance(doc, dict):
        for key, value in doc.items():
            where = f"{path}.{key}" if path else str(key)
            if str(key) in _FORBIDDEN_KEYS:
                raise ValueError(
                    f"architecture must not set {where!r}: efficiency, noise figure, "
                    "losses and costs come from the task's parts table"
                )
            _reject_forbidden(value, where)
    elif isinstance(doc, list):
        for i, item in enumerate(doc):
            _reject_forbidden(item, f"{path}[{i}]")


def _require_keys(section: dict[str, Any], expected: set[str], name: str) -> None:
    missing = expected - set(section)
    extra = set(section) - expected
    if missing:
        raise ValueError(f"{name} is missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{name} has unknown keys: {sorted(extra)}")


def _load_architecture(spec: TaskSpec, submission: Path) -> tuple[Any, dict[str, Any]]:
    """Parse and price the submission; returns (Architecture, submission dict)."""
    from phased_array_systems.architecture import (
        Architecture,
        ArrayConfig,
        CostConfig,
        DigitalConfig,
        RFChainConfig,
    )
    from pydantic import ValidationError

    doc = yaml.safe_load(submission.read_text())
    if not isinstance(doc, dict):
        raise ValueError("architecture.yaml must be a mapping")
    _reject_forbidden(doc)
    _require_keys(doc, {"array", "rf", "digital"}, "architecture.yaml")
    _require_keys(doc["array"], _ARRAY_KEYS, "array")
    _require_keys(doc["rf"], _RF_KEYS, "rf")
    _require_keys(doc["digital"], _DIGITAL_KEYS, "digital")

    ctx = spec.context
    parts = ctx["parts"]
    link = ctx["link"]

    pa_class = str(doc["rf"]["pa_class"])
    if pa_class not in parts["pa_classes"]:
        raise ValueError(f"pa_class {pa_class!r} not in {sorted(parts['pa_classes'])}")
    pa = parts["pa_classes"][pa_class]

    phase_bits = int(doc["array"]["phase_bits"])
    shifter_prices = {int(k): float(v) for k, v in parts["phase_shifter"].items()}
    if phase_bits not in shifter_prices:
        raise ValueError(f"phase_bits {phase_bits} not in priced set {sorted(shifter_prices)}")

    tx_power = float(doc["rf"]["tx_power_w_per_elem"])
    tx_max = float(parts["tx_power_w_per_elem_max"])
    if not 0.0 < tx_power <= tx_max:
        raise ValueError(f"tx_power_w_per_elem must be in (0, {tx_max}]")

    cost_per_elem = (
        float(parts["element_base_cost_usd"])
        + float(pa["cost_adder_usd_per_elem"])
        + shifter_prices[phase_bits]
    )

    try:
        arch = Architecture(
            array=ArrayConfig(
                nx=int(doc["array"]["nx"]),
                ny=int(doc["array"]["ny"]),
                dx_lambda=float(doc["array"]["dx_lambda"]),
                dy_lambda=float(doc["array"]["dy_lambda"]),
                taper_type=doc["array"]["taper_type"],
                taper_sll_db=float(doc["array"]["taper_sll_db"]),
                phase_bits=phase_bits,
                scan_limit_deg=float(ctx["evaluation_envelope_bounds"]["scan_angle_deg_max"]),
                element_cos_exp=float(ctx["element"]["q"]),
                max_subarray_nx=int(doc["digital"]["subarray_nx"]),
                max_subarray_ny=int(doc["digital"]["subarray_ny"]),
            ),
            rf=RFChainConfig(
                tx_power_w_per_elem=tx_power,
                pa_efficiency=float(pa["pa_efficiency"]),
                noise_figure_db=float(link["satellite_noise_figure_db"]),
                feed_loss_db=float(parts["feed_loss_db"]),
                rx_power_w_per_elem=float(parts["rx_power_w_per_elem"]),
            ),
            cost=CostConfig(cost_per_elem_usd=cost_per_elem, nre_usd=0.0),
            digital=DigitalConfig(
                digitization_level=doc["digital"]["digitization_level"],
                adc_enob=float(doc["digital"]["adc_enob"]),
            ),
        )
    except ValidationError as exc:
        raise ValueError(f"architecture rejected by the system model: {exc}") from exc
    return arch, doc


def _scenario(spec: TaskSpec, point: dict[str, Any]) -> Any:
    from phased_array_systems.scenarios import CommsLinkScenario

    link = spec.context["link"]
    return CommsLinkScenario(
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


def _pas_point(
    arch: Any, spec: TaskSpec, point: dict[str, Any], seed: int | None
) -> dict[str, float]:
    from phased_array_systems.models.antenna import PhasedArrayAdapter
    from phased_array_systems.models.comms import CommsLinkModel

    scenario = _scenario(spec, point)
    ctx: dict[str, Any] = {}
    if seed is not None:
        ctx["failure_rate"] = float(
            spec.context["evaluation_envelope_bounds"]["element_failure_rate"]
        )
        ctx["meta.seed"] = seed
    antenna = PhasedArrayAdapter(use_analytical_fallback=False).evaluate(arch, scenario, ctx)
    link = CommsLinkModel().evaluate(arch, scenario, dict(antenna))
    return {
        "link_margin_db": _num(link, "link_margin_db"),
        "g_peak_db": _num(antenna, "g_peak_db"),
        "n_failed_elements": _num(antenna, "n_failed_elements", 0.0),
    }


def _pam_pattern(
    arch: Any, spec: TaskSpec, scan_deg: float, seed: int | None
) -> tuple[float, float]:
    """(directivity_dbi, peak_sidelobe_db) from a full-pattern recompute."""
    import phased_array as pa

    from aedl.evaluators.array_pattern import _directivity_dbi, _peak_sidelobe_db

    params = spec.evaluator_params.get("crosscheck", {})
    n_theta = int(params.get("pam_n_theta", 721))
    n_phi = int(params.get("pam_n_phi", 1441))

    freq_hz = float(spec.context["link"]["frequency_ghz"]) * 1e9
    wavelength = 299_792_458.0 / freq_hz
    array = arch.array
    geom = pa.create_rectangular_array(
        array.nx, array.ny, dx=array.dx_lambda, dy=array.dy_lambda, wavelength=wavelength
    )
    k = pa.wavelength_to_k(wavelength)

    tapers = {
        "taylor": lambda: pa.taylor_taper_2d(array.nx, array.ny, sidelobe_dB=array.taper_sll_db),
        "chebyshev": lambda: pa.chebyshev_taper_2d(
            array.nx, array.ny, sidelobe_dB=array.taper_sll_db
        ),
        "uniform": lambda: np.ones((array.nx, array.ny)),
    }
    if array.taper_type not in tapers:
        raise ValueError(f"unsupported taper_type {array.taper_type!r}")
    weights = np.asarray(tapers[array.taper_type]()).ravel().astype(complex)  # type: ignore[no-untyped-call]
    weights *= pa.steering_vector(k, geom.x, geom.y, scan_deg, 0.0)
    if array.phase_bits is not None:
        weights = pa.quantize_phase(weights, n_bits=array.phase_bits)
    if seed is not None:
        rate = float(spec.context["evaluation_envelope_bounds"]["element_failure_rate"])
        weights, _ = pa.simulate_element_failures(weights, rate, seed=seed)

    theta, phi, pattern_db = pa.compute_full_pattern(
        geom.x,
        geom.y,
        weights,
        k,
        n_theta=n_theta,
        n_phi=n_phi,
        theta_range=(0.0, np.pi),
        element_pattern_func=pa.element_pattern,
        cos_exp_theta=float(spec.context["element"]["q"]),
    )
    theta_g, phi_g = np.meshgrid(theta, phi, indexing="ij")
    return _directivity_dbi(theta_g, phi_g, pattern_db), _peak_sidelobe_db(pattern_db)


def _opensatcom_margin(gain_dbi: float, arch: Any, spec: TaskSpec, point: dict[str, Any]) -> float:
    """Independent link recompute with the PAM-verified gain injected."""
    from opensatcom.antenna.parametric import ParametricAntenna
    from opensatcom.core.models import (
        LinkInputs,
        PropagationConditions,
        RFChainModel,
        Scenario,
        Terminal,
    )
    from opensatcom.link.engine import DefaultLinkEngine
    from opensatcom.propagation.composite import CompositePropagation
    from opensatcom.propagation.fspl import FreeSpacePropagation
    from opensatcom.propagation.gas import GaseousAbsorptionP676
    from opensatcom.propagation.rain import RainAttenuationP618

    link = spec.context["link"]
    params = spec.evaluator_params.get("opensatcom", {})
    freq_hz = float(link["frequency_ghz"]) * 1e9
    bandwidth_hz = float(link["bandwidth_mhz"]) * 1e6

    # Mirror the PAS noise convention: T_sys = T_ant + 290 (F - 1).
    nf_lin = 10.0 ** (float(link["satellite_noise_figure_db"]) / 10.0)
    rx_temp_k = float(point["sky_temp_k"]) + 290.0 * (nf_lin - 1.0)

    scenario = Scenario(
        name="t3-crosscheck",
        direction="uplink",
        freq_hz=freq_hz,
        bandwidth_hz=bandwidth_hz,
        polarization="RHCP",
        required_metric="ebn0_db",
        required_value=float(link["required_snr_db"]),
    )
    tx = Terminal("terminal", 0.0, 0.0, 0.0)
    rx = Terminal("satellite", 0.0, 0.0, 550e3, system_noise_temp_k=rx_temp_k)
    total_tx_w = float(arch.rf.tx_power_w_per_elem) * int(arch.array.nx) * int(arch.array.ny)
    rf = RFChainModel(
        tx_power_w=total_tx_w,
        tx_losses_db=float(spec.context["parts"]["feed_loss_db"]),
        rx_noise_temp_k=rx_temp_k,
    )
    propagation = CompositePropagation(
        [
            FreeSpacePropagation(),
            GaseousAbsorptionP676(),
            RainAttenuationP618(
                availability_target=float(params.get("availability_target", 0.995))
            ),
        ]
    )
    inputs = LinkInputs(
        tx_terminal=tx,
        rx_terminal=rx,
        scenario=scenario,
        tx_antenna=ParametricAntenna(gain_dbi=gain_dbi),
        rx_antenna=ParametricAntenna(gain_dbi=float(link["satellite_rx_gain_db"])),
        propagation=propagation,
        rf_chain=rf,
    )
    cond = PropagationConditions(rain_rate_mm_per_hr=float(point["rain_mmh"]) or None)
    out = DefaultLinkEngine().evaluate_snapshot(
        elev_deg=float(point["elevation_deg"]),
        az_deg=0.0,
        range_m=float(point["range_km"]) * 1e3,
        inputs=inputs,
        cond=cond,
    )
    # ebn0 over the full bandwidth equals SNR in that bandwidth, matching the
    # PAS margin definition (snr in bandwidth minus required snr).
    return float(out.ebn0_db) - float(link["required_snr_db"])


@register_evaluator("satcom_terminal")
def evaluate(spec: TaskSpec, submission: Path) -> EvaluationResult:
    t_start = time.perf_counter()
    ctx = spec.context
    parts = ctx["parts"]
    envelope = spec.evaluator_params["envelope"]
    points: list[dict[str, Any]] = list(envelope["points"])
    seeds: list[int] = sorted(int(s) for s in envelope["failure_seeds"])

    arch, submitted = _load_architecture(spec, submission)

    from phased_array_systems.models.antenna import PhasedArrayAdapter
    from phased_array_systems.models.swapc import CostModel, PowerModel

    nominal = dict(ctx["nominal_scenario"])
    nominal_point = {
        "scan_deg": nominal["scan_angle_deg"],
        "elevation_deg": nominal["elevation_deg"],
        "range_km": nominal["slant_range_km"],
        "rain_mmh": nominal["rain_rate_mmh"],
        "sky_temp_k": nominal["sky_noise_temp_k"],
    }
    nominal_scenario = _scenario(spec, nominal_point)
    antenna_nominal = PhasedArrayAdapter(use_analytical_fallback=False).evaluate(
        arch, nominal_scenario, {}
    )

    metrics: dict[str, float] = {}

    # --- SWaP-C, scenario-independent at duty 1.0 ---
    power = PowerModel().evaluate(arch, nominal_scenario, dict(antenna_nominal))
    cost = CostModel().evaluate(arch, nominal_scenario, dict(antenna_nominal))
    n_channels = int(getattr(arch, "n_digital_channels", arch.array.nx * arch.array.ny))
    metrics["prime_power_w"] = _num(power, "prime_power_w")
    metrics["unit_cost_usd"] = _num(cost, "total_cost_usd") + float(
        parts["adc_cost_usd_per_channel"]
    ) * float(n_channels)
    metrics["grating_margin_lambda"] = _num(antenna_nominal, "max_safe_spacing_lambda") - max(
        float(arch.array.dx_lambda), float(arch.array.dy_lambda)
    )

    # --- Worst case over the held-out envelope x failure seeds ---
    per_point: list[dict[str, Any]] = []
    worst: tuple[float, int | None, int | None] = (float("inf"), None, None)
    n_evals = 0
    for idx, point in enumerate(points):
        for seed in seeds:
            result = _pas_point(arch, spec, point, seed)
            n_evals += 1
            per_point.append({"point": idx, "seed": seed, **result})
            if result["link_margin_db"] < worst[0]:
                worst = (result["link_margin_db"], idx, seed)
    metrics["worst_case_link_margin_db"] = worst[0]

    # --- PAM cross-check: gain at nominal, sidelobes at the binding scan ---
    pam_gain_nominal, _ = _pam_pattern(arch, spec, float(nominal_point["scan_deg"]), None)
    metrics["crosscheck_gain_disagreement_db"] = abs(
        _num(antenna_nominal, "g_peak_db") - pam_gain_nominal
    )
    binding_scan = float(points[worst[1]]["scan_deg"]) if worst[1] is not None else 0.0
    metrics["worst_case_pattern_sll_db"] = max(
        _pam_pattern(arch, spec, binding_scan, seed)[1] for seed in seeds
    )

    # --- opensatcom independent link check ---
    pas_nominal = _pas_point(arch, spec, nominal_point, None)
    osc_nominal = _opensatcom_margin(pam_gain_nominal, arch, spec, nominal_point)
    metrics["crosscheck_clearsky_margin_disagreement_db"] = abs(
        pas_nominal["link_margin_db"] - osc_nominal
    )
    osc_worst = float("inf")
    pam_gain_by_scan: dict[float, float] = {float(nominal_point["scan_deg"]): pam_gain_nominal}
    for point in points:
        scan = float(point["scan_deg"])
        if scan not in pam_gain_by_scan:
            pam_gain_by_scan[scan] = _pam_pattern(arch, spec, scan, None)[0]
        osc_worst = min(osc_worst, _opensatcom_margin(pam_gain_by_scan[scan], arch, spec, point))
    metrics["opensatcom_worst_case_margin_db"] = osc_worst

    # --- Requirements ---
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

    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str | None] = {}
    for dist in ("phased-array-systems", "phased-array-modeling", "opensatcom"):
        try:
            versions[dist] = version(dist)
        except PackageNotFoundError:
            versions[dist] = None

    return EvaluationResult(
        task_id=spec.id,
        passed=all(r.passed for r in req_results),
        requirements=tuple(req_results),
        info={
            "metrics": metrics,
            "architecture": submitted,
            "per_point": per_point,
            "binding": {"point": worst[1], "seed": worst[2]},
            "nominal_margin_db": pas_nominal["link_margin_db"],
            "versions": versions,
        },
        cost={
            "evaluation_wall_time_s": round(time.perf_counter() - t_start, 3),
            "n_model_evals": n_evals,
        },
    )
