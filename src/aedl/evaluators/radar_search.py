"""Deterministic evaluator for the tier-3 X-band search radar task.

The submission is an ``architecture.yaml`` naming array, RF, waveform, and
digital choices. The evaluator prices the design from the task's parts
table (efficiency, noise figure, MTBFs, costs, pulse width, and every
detection-statistics setting are pinned by the task), evaluates detection
worst-case over a held-out envelope of scenario points and element-failure
seeds, and cross-checks the claims that matter:

- pattern gain and sidelobes are recomputed by full pattern integration
  with ``phased-array-modeling`` (quantization and seeded failures
  applied), exactly as in the satcom evaluator;
- the detection probability at the binding envelope point is recomputed by
  Monte Carlo simulation of a Swerling-1 target through an actual
  cell-averaging CFAR (:mod:`aedl.evaluators.radar_mc`), sharing no code
  with phased-array-systems' analytic chain. The analytic chain models
  CFAR as an SNR loss calibrated at n_pulses=1, which is measurably
  conservative for multi-pulse integration; the frozen tolerance covers
  that known one-sided disagreement.

The scored trade: a bigger aperture narrows the beam, which multiplies
search beam positions and blows the frame-time budget; fewer pulses per
dwell lose integration gain, forcing per-element power up, which heats the
junction and erodes end-of-mission availability through Arrhenius
derating. Duty cycle is computed from the pinned pulse width and the
submitted PRF — submitting it is an error.

Honest scope note: the sea-clutter *level* (sigma-0) has no independent
recompute; the Monte Carlo validates detection statistics given SCNR. The
clutter model's provenance is established in the calibration script
against published X-band ranges, not per-run.

Metrics produced (requirements reference these by name):

- ``worst_case_pd``: min Pd over envelope x failure seeds (PAS chain)
- ``worst_case_timeline_occupancy``: max occupancy over envelope points
- ``array_availability``: steady-state availability from the pinned MTBFs
  and MTTR, with junction temperature fed forward from dissipated power
- ``worst_case_pattern_sll_db``: max full-pattern SLL over seeds at the
  binding point's scan angle (PAM recompute)
- ``crosscheck_pd_disagreement``: |MC - analytic| Pd at the binding point
- ``crosscheck_gain_disagreement_db``: |PAS g_peak - PAM directivity|
- ``prime_power_w``, ``unit_cost_usd``: priced from the parts table
- ``grating_margin_lambda``: max safe spacing minus the design's spacing
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from aedl.registry import register_evaluator
from aedl.result import EvaluationResult, RequirementResult
from aedl.spec import TaskSpec

# Keys the submission must not contain anywhere: these are priced or pinned
# by the task, and accepting them would reopen the free-lunch axes.
_FORBIDDEN_KEYS = {
    "pa_efficiency",
    "noise_figure_db",
    "feed_loss_db",
    "cost",
    "cost_per_elem_usd",
    "cost_usd",
    "rx_power_w_per_elem",
    "duty_cycle",
    "tau_us",
    "pulse_width_us",
    "swerling",
    "pfa",
    "pd_required",
    "cfar_type",
    "cfar_ref_cells",
    "cfar_guard_cells",
    "component_mtbfs",
    "mtbf",
    "mttr_hours",
    "mission_hours",
    "thermal_resistance_c_per_w",
    "ambient_temp_c",
    "operating_temp_c",
}

_ARRAY_KEYS = {"nx", "ny", "dx_lambda", "dy_lambda", "taper_type", "taper_sll_db", "phase_bits"}
_RF_KEYS = {"tx_power_w_per_elem", "pa_class"}
_WAVEFORM_KEYS = {"prf_hz", "n_pulses"}
_DIGITAL_KEYS = {"digitization_level", "adc_enob", "subarray_nx", "subarray_ny"}


def _num(metrics: dict[str, Any], key: str, default: float | None = None) -> float:
    value = metrics.get(key, default)
    if not isinstance(value, (int, float)):
        raise TypeError(f"metric {key!r} is {type(value).__name__}, expected a number")
    return float(value)


def _reject_forbidden(doc: Any, path: str = "") -> None:
    if isinstance(doc, dict):
        for key, value in doc.items():
            where = f"{path}.{key}" if path else str(key)
            if str(key) in _FORBIDDEN_KEYS:
                raise ValueError(
                    f"architecture must not set {where!r}: detection statistics, "
                    "duty cycle, reliability, losses and costs are pinned by the task"
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


def _load_architecture(spec: TaskSpec, submission: Path) -> tuple[Any, dict[str, Any], float]:
    """Parse and price the submission; returns (Architecture, dict, duty_cycle)."""
    from phased_array_systems.architecture import (
        Architecture,
        ArrayConfig,
        CostConfig,
        DigitalConfig,
        ReliabilityConfig,
        RFChainConfig,
    )
    from pydantic import ValidationError

    doc = yaml.safe_load(submission.read_text())
    if not isinstance(doc, dict):
        raise ValueError("architecture.yaml must be a mapping")
    _reject_forbidden(doc)
    _require_keys(doc, {"array", "rf", "waveform", "digital"}, "architecture.yaml")
    _require_keys(doc["array"], _ARRAY_KEYS, "array")
    _require_keys(doc["rf"], _RF_KEYS, "rf")
    _require_keys(doc["waveform"], _WAVEFORM_KEYS, "waveform")
    _require_keys(doc["digital"], _DIGITAL_KEYS, "digital")

    ctx = spec.context
    parts = ctx["parts"]
    radar = ctx["radar"]
    rel = ctx["reliability"]
    wf_bounds = ctx["waveform_bounds"]

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

    prf_hz = float(doc["waveform"]["prf_hz"])
    n_pulses = int(doc["waveform"]["n_pulses"])
    if not float(wf_bounds["prf_hz_min"]) <= prf_hz <= float(wf_bounds["prf_hz_max"]):
        raise ValueError(
            f"prf_hz must be in [{wf_bounds['prf_hz_min']}, {wf_bounds['prf_hz_max']}]"
        )
    if not int(wf_bounds["n_pulses_min"]) <= n_pulses <= int(wf_bounds["n_pulses_max"]):
        raise ValueError(
            f"n_pulses must be in [{wf_bounds['n_pulses_min']}, {wf_bounds['n_pulses_max']}]"
        )

    # Duty cycle is derived, never submitted: the thermal free-lunch axis.
    tau_s = float(parts["tau_us"]) * 1e-6
    duty_cycle = tau_s * prf_hz
    if duty_cycle >= 1.0:
        raise ValueError(
            f"pinned pulse width {parts['tau_us']} us at prf {prf_hz} Hz gives "
            f"duty {duty_cycle:.2f} >= 1"
        )

    cost_per_elem = (
        float(parts["element_base_cost_usd"])
        + float(pa["cost_adder_usd_per_elem"])
        + shifter_prices[phase_bits]
    )

    # PA class selects efficiency AND its MTBF: cheap PAs run hot and die
    # young, making the class a genuine efficiency/cost/reliability trade.
    mtbfs = {str(k): float(v) for k, v in rel["component_mtbfs"].items()}
    mtbfs["pa"] = float(pa["pa_mtbf_hours"])

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
                scan_limit_deg=float(ctx["evaluation_envelope_bounds"]["scan_deg_max"]),
                element_cos_exp=float(ctx["element"]["q"]),
                max_subarray_nx=int(doc["digital"]["subarray_nx"]),
                max_subarray_ny=int(doc["digital"]["subarray_ny"]),
            ),
            rf=RFChainConfig(
                tx_power_w_per_elem=tx_power,
                pa_efficiency=float(pa["pa_efficiency"]),
                noise_figure_db=float(radar["noise_figure_db"]),
                feed_loss_db=float(parts["feed_loss_db"]),
                rx_power_w_per_elem=float(parts["rx_power_w_per_elem"]),
            ),
            cost=CostConfig(cost_per_elem_usd=cost_per_elem, nre_usd=0.0),
            reliability=ReliabilityConfig(
                component_mtbfs=mtbfs,
                thermal_resistance_c_per_w=float(rel["thermal_resistance_c_per_w"]),
                ambient_temp_c=float(rel["ambient_temp_c"]),
                mttr_hours=float(rel["mttr_hours"]),
                mission_hours=float(rel["mission_hours"]),
            ),
            digital=DigitalConfig(
                digitization_level=doc["digital"]["digitization_level"],
                adc_enob=float(doc["digital"]["adc_enob"]),
            ),
        )
    except ValidationError as exc:
        raise ValueError(f"architecture rejected by the system model: {exc}") from exc
    return arch, doc, duty_cycle


def _scenario(spec: TaskSpec, point: dict[str, Any], doc: dict[str, Any]) -> Any:
    from phased_array_systems.scenarios import RadarDetectionScenario

    ctx = spec.context
    radar = ctx["radar"]
    parts = ctx["parts"]
    search = ctx["search"]
    sea_state = int(point.get("sea_state", 0))
    return RadarDetectionScenario(
        freq_hz=float(radar["frequency_ghz"]) * 1e9,
        bandwidth_hz=float(radar["bandwidth_mhz"]) * 1e6,
        range_m=float(point["range_km"]) * 1e3,
        target_rcs_dbsm=float(point["rcs_dbsm"]),
        scan_angle_deg=float(point["scan_deg"]),
        pfa=float(radar["pfa"]),
        pd_required=float(radar["pd_required"]),
        n_pulses=int(doc["waveform"]["n_pulses"]),
        swerling=1,
        integration_type="noncoherent",
        duty_cycle=float(parts["tau_us"]) * 1e-6 * float(doc["waveform"]["prf_hz"]),
        clutter_type="sea" if sea_state > 0 else "none",
        sea_state=max(sea_state, 1),
        antenna_height_m=float(radar["antenna_height_m"]),
        target_height_m=float(point["target_height_m"]),
        cfar_type=cast(Any, str(radar["cfar_type"])),
        cfar_ref_cells=int(radar["cfar_ref_cells"]),
        cfar_guard_cells=int(radar["cfar_guard_cells"]),
        prf_hz=float(doc["waveform"]["prf_hz"]),
        search_az_extent_deg=float(search["az_extent_deg"]),
        search_el_extent_deg=float(search["el_extent_deg"]),
        beam_overhead_us=float(search["beam_overhead_us"]),
        search_frame_time_ms=float(search["frame_time_ms"]),
    )


def _pas_point(
    arch: Any, spec: TaskSpec, point: dict[str, Any], doc: dict[str, Any], seed: int | None
) -> dict[str, float]:
    """Radar chain at one envelope point: adapter (seeded failures) -> RadarModel."""
    from phased_array_systems.models.antenna import PhasedArrayAdapter
    from phased_array_systems.models.radar import RadarModel

    scenario = _scenario(spec, point, doc)
    ctx: dict[str, Any] = {}
    if seed is not None:
        ctx["failure_rate"] = float(
            spec.context["evaluation_envelope_bounds"]["element_failure_rate"]
        )
        ctx["meta.seed"] = seed
    antenna = PhasedArrayAdapter(use_analytical_fallback=False).evaluate(arch, scenario, ctx)
    radar = RadarModel().evaluate(arch, scenario, dict(antenna))
    return {
        "pd_achieved": _num(radar, "pd_achieved"),
        "scnr_db": _num(radar, "scnr_db", _num(radar, "snr_single_pulse_db")),
        "cfar_loss_db": _num(radar, "cfar_loss_db", 0.0),
        "g_peak_db": _num(antenna, "g_peak_db"),
        "beamwidth_az_deg": _num(antenna, "beamwidth_az_deg", 5.0),
        "beamwidth_el_deg": _num(antenna, "beamwidth_el_deg", 5.0),
    }


def _timeline_occupancy(
    spec: TaskSpec, doc: dict[str, Any], point_metrics: dict[str, float]
) -> float:
    """Search occupancy at one point, via PAS's own scheduling model."""
    from phased_array_systems.models.digital.scheduling import max_update_rate

    search = spec.context["search"]
    n_pulses = int(doc["waveform"]["n_pulses"])
    prf_hz = float(doc["waveform"]["prf_hz"])
    dwell_time_s = n_pulses / prf_hz
    bw_az = point_metrics["beamwidth_az_deg"]
    bw_el = point_metrics["beamwidth_el_deg"]
    # Defense in depth: a submission must never crash the evaluator on a
    # degenerate pattern cut. Non-finite widths fall back to the other axis.
    if not math.isfinite(bw_el):
        bw_el = bw_az
    if not math.isfinite(bw_az):
        bw_az = bw_el
    if not math.isfinite(bw_az):
        raise ValueError("both principal-plane beamwidths are undefined")
    beam_sr = math.radians(bw_az) * math.radians(bw_el)
    volume_sr = math.radians(float(search["az_extent_deg"])) * math.radians(
        float(search["el_extent_deg"])
    )
    timeline = max_update_rate(
        scan_volume_sr=volume_sr,
        beam_solid_angle_sr=beam_sr,
        dwell_time_us=dwell_time_s * 1e6,
        overhead_us=float(search["beam_overhead_us"]),
    )
    return float(timeline["frame_time_ms"]) / float(search["frame_time_ms"])


def _pam_pattern(
    arch: Any, spec: TaskSpec, scan_deg: float, seed: int | None
) -> tuple[float, float]:
    """(directivity_dbi, peak_sidelobe_db) from a full-pattern recompute."""
    import phased_array as pa

    from aedl.evaluators.array_pattern import _directivity_dbi, _peak_sidelobe_db

    params = spec.evaluator_params.get("crosscheck", {})
    n_theta = int(params.get("pam_n_theta", 361))
    n_phi = int(params.get("pam_n_phi", 721))

    freq_hz = float(spec.context["radar"]["frequency_ghz"]) * 1e9
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


@register_evaluator("radar_search")
def evaluate(spec: TaskSpec, submission: Path) -> EvaluationResult:
    t_start = time.perf_counter()
    ctx = spec.context
    parts = ctx["parts"]
    envelope = spec.evaluator_params["envelope"]
    points: list[dict[str, Any]] = list(envelope["points"])
    seeds: list[int] = sorted(int(s) for s in envelope["failure_seeds"])

    arch, submitted, duty_cycle = _load_architecture(spec, submission)

    from phased_array_systems.evaluate import evaluate_case
    from phased_array_systems.models.antenna import PhasedArrayAdapter

    nominal_point = dict(ctx["nominal_scenario"])
    nominal_scenario = _scenario(spec, nominal_point, submitted)
    antenna_nominal = PhasedArrayAdapter(use_analytical_fallback=False).evaluate(
        arch, nominal_scenario, {}
    )

    metrics: dict[str, float] = {}

    # --- SWaP-C + reliability, from the shipped top-level path (duty-aware
    # PA power, junction-temp feed-forward, Arrhenius derating) ---
    case = evaluate_case(arch, nominal_scenario)
    metrics["prime_power_w"] = _num(case, "prime_power_w")
    metrics["array_availability"] = _num(case, "array_availability")
    n_channels = int(getattr(arch, "n_digital_channels", arch.array.nx * arch.array.ny))
    metrics["unit_cost_usd"] = _num(case, "total_cost_usd") + float(
        parts["adc_cost_usd_per_channel"]
    ) * float(n_channels)
    metrics["grating_margin_lambda"] = _num(antenna_nominal, "max_safe_spacing_lambda") - max(
        float(arch.array.dx_lambda), float(arch.array.dy_lambda)
    )

    # --- Worst case over the held-out envelope x failure seeds ---
    per_point: list[dict[str, Any]] = []
    worst: tuple[float, int | None, int | None] = (float("inf"), None, None)
    worst_metrics: dict[str, float] | None = None
    occupancy_worst = 0.0
    n_evals = 0
    for idx, point in enumerate(points):
        for seed in seeds:
            result = _pas_point(arch, spec, point, submitted, seed)
            n_evals += 1
            per_point.append(
                {
                    "point": idx,
                    "seed": seed,
                    "pd": result["pd_achieved"],
                    "scnr_db": result["scnr_db"],
                }
            )
            if result["pd_achieved"] < worst[0]:
                worst = (result["pd_achieved"], idx, seed)
                worst_metrics = result
            occupancy_worst = max(occupancy_worst, _timeline_occupancy(spec, submitted, result))
    metrics["worst_case_pd"] = worst[0]
    metrics["worst_case_timeline_occupancy"] = occupancy_worst

    # --- PAM cross-check: gain at nominal, sidelobes at the binding scan ---
    pam_gain_nominal, _ = _pam_pattern(arch, spec, float(nominal_point["scan_deg"]), None)
    metrics["crosscheck_gain_disagreement_db"] = abs(
        _num(antenna_nominal, "g_peak_db") - pam_gain_nominal
    )
    binding_scan = float(points[worst[1]]["scan_deg"]) if worst[1] is not None else 0.0
    metrics["worst_case_pattern_sll_db"] = max(
        _pam_pattern(arch, spec, binding_scan, seed)[1] for seed in seeds
    )

    # --- Monte Carlo detection cross-check at the binding point ---
    from aedl.evaluators.radar_mc import simulate_pd_swerling1_cfar

    assert worst_metrics is not None
    mc_params = spec.evaluator_params.get("crosscheck", {})
    mc_pd = simulate_pd_swerling1_cfar(
        scnr_db=worst_metrics["scnr_db"],
        pfa=float(ctx["radar"]["pfa"]),
        n_pulses=int(submitted["waveform"]["n_pulses"]),
        n_ref=int(ctx["radar"]["cfar_ref_cells"]),
        n_trials=int(mc_params.get("mc_trials", 200_000)),
        seed=int(mc_params.get("mc_seed", 20260812)),
    )
    metrics["crosscheck_pd_disagreement"] = abs(mc_pd - worst[0])

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
    for dist in ("phased-array-systems", "phased-array-modeling"):
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
            "duty_cycle": duty_cycle,
            "per_point": per_point,
            "binding": {"point": worst[1], "seed": worst[2]},
            "mc_pd_at_binding": mc_pd,
            "versions": versions,
        },
        cost={
            "evaluation_wall_time_s": round(time.perf_counter() - t_start, 3),
            "n_model_evals": n_evals,
        },
    )
