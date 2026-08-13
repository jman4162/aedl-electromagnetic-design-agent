import sys, math
sys.path.insert(0, '/Users/johnhodge/Documents/code/phased-array-systems/src')
import numpy as np
from phased_array_systems.architecture import Architecture, ArrayConfig, RFChainConfig, DigitalConfig, ReliabilityConfig, CostConfig
from phased_array_systems.scenarios import RadarDetectionScenario
from phased_array_systems.models.antenna.adapter import PhasedArrayAdapter
from phased_array_systems.models.radar.equation import RadarModel
from phased_array_systems.models.swapc.power import PowerModel
from phased_array_systems.models.rf.reliability import TRMReliabilitySpec, analyze_array_reliability
from phased_array_systems.models.digital.scheduling import max_update_rate

C_LIGHT = 299792458.0

CTX = dict(
    freq_hz=9.5e9, bandwidth_hz=5.0e6, pfa=1e-6, pd_required=0.7, noise_figure_db=3.5,
    antenna_height_m=30.0, cfar_type="CA", cfar_ref_cells=16, cfar_guard_cells=2,
    az_extent_deg=60.0, el_extent_deg=15.0, frame_time_ms=3000.0, beam_overhead_us=50.0,
    tau_us=33.0,
)

PA_CLASSES = {
    "A": dict(pa_efficiency=0.20, pa_mtbf_hours=150000.0, cost_adder=40.0),
    "B": dict(pa_efficiency=0.30, pa_mtbf_hours=300000.0, cost_adder=120.0),
    "C": dict(pa_efficiency=0.40, pa_mtbf_hours=500000.0, cost_adder=260.0),
}
PS_COST = {3: 5.0, 4: 12.0, 5: 25.0, 6: 50.0}
ELEMENT_BASE_COST = 90.0
ADC_COST_PER_CH = 40.0
RX_POWER_W_PER_ELEM = 0.25
FEED_LOSS_DB = 1.5

COMPONENT_MTBFS_BASE = dict(
    lna=500000.0, phase_shifter=2000000.0, attenuator=3000000.0,
    switch=1000000.0, control_asic=1000000.0,
)

RELIAB = dict(thermal_resistance_c_per_w=40.0, ambient_temp_c=40.0, mttr_hours=24.0, mission_hours=4380.0)

# evaluation envelope points (visible ones from task.yaml; treat as representative)
ENV_POINTS = [
    dict(range_km=12.0, rcs_dbsm=6.0, scan_deg=10.0, sea_state=3, target_height_m=5.0),
    dict(range_km=14.0, rcs_dbsm=6.0, scan_deg=40.0, sea_state=3, target_height_m=5.0),
    dict(range_km=13.0, rcs_dbsm=6.0, scan_deg=25.0, sea_state=3, target_height_m=5.0),
    dict(range_km=18.0, rcs_dbsm=5.0, scan_deg=33.0, sea_state=2, target_height_m=5.0),
    dict(range_km=22.0, rcs_dbsm=4.0, scan_deg=20.0, sea_state=2, target_height_m=5.0),
    dict(range_km=24.0, rcs_dbsm=3.0, scan_deg=30.0, sea_state=0, target_height_m=5.0),
    dict(range_km=26.0, rcs_dbsm=2.0, scan_deg=15.0, sea_state=0, target_height_m=5.0),
    # extra corner points near bounds for margin
    dict(range_km=14.0, rcs_dbsm=6.0, scan_deg=0.0, sea_state=3, target_height_m=5.0),
    dict(range_km=26.0, rcs_dbsm=2.0, scan_deg=40.0, sea_state=0, target_height_m=5.0),
    dict(range_km=13.0, rcs_dbsm=6.0, scan_deg=40.0, sea_state=3, target_height_m=5.0),
]

FAILURE_SEEDS = [101, 211, 307, 401, 503]
FAILURE_RATE = 0.02


def build_arch(nx, ny, dx, dy, taper_type, taper_sll_db, phase_bits,
                tx_power_w_per_elem, pa_class, digitization_level, adc_enob,
                subarray_nx, subarray_ny, max_subarray_nx=8, max_subarray_ny=8):
    array = ArrayConfig(
        nx=nx, ny=ny, dx_lambda=dx, dy_lambda=dy,
        taper_type=taper_type, taper_sll_db=taper_sll_db, phase_bits=phase_bits,
        scan_limit_deg=40.0,
        max_subarray_nx=max_subarray_nx, max_subarray_ny=max_subarray_ny,
        element_cos_exp=1.3,
    )
    pa = PA_CLASSES[pa_class]
    rf = RFChainConfig(
        tx_power_w_per_elem=tx_power_w_per_elem,
        rx_power_w_per_elem=RX_POWER_W_PER_ELEM,
        pa_efficiency=pa["pa_efficiency"],
        noise_figure_db=CTX["noise_figure_db"],
        feed_loss_db=FEED_LOSS_DB,
    )
    comp_mtbfs = dict(COMPONENT_MTBFS_BASE)
    comp_mtbfs["pa"] = pa["pa_mtbf_hours"]
    reliability = ReliabilityConfig(
        component_mtbfs=comp_mtbfs,
        thermal_resistance_c_per_w=RELIAB["thermal_resistance_c_per_w"],
        ambient_temp_c=RELIAB["ambient_temp_c"],
        mttr_hours=RELIAB["mttr_hours"],
        mission_hours=RELIAB["mission_hours"],
    )
    digital = DigitalConfig(
        digitization_level=digitization_level, adc_enob=adc_enob,
    )
    arch = Architecture(array=array, rf=rf, reliability=reliability, digital=digital,
                         cost=CostConfig())
    return arch, subarray_nx, subarray_ny


def n_digital_channels(arch, subarray_nx, subarray_ny):
    level = arch.digital.digitization_level
    if level == "element":
        return arch.array.n_elements
    if level == "subarray":
        nsx = math.ceil(arch.array.nx / subarray_nx)
        nsy = math.ceil(arch.array.ny / subarray_ny)
        return nsx * nsy
    return 1


def unit_cost(arch, pa_class, phase_bits, subarray_nx, subarray_ny):
    n = arch.array.n_elements
    cost = n * ELEMENT_BASE_COST
    cost += n * PA_CLASSES[pa_class]["cost_adder"]
    cost += n * PS_COST[phase_bits]
    cost += n_digital_channels(arch, subarray_nx, subarray_ny) * ADC_COST_PER_CH
    return cost


def evaluate_design(nx, ny, dx, dy, taper_type, taper_sll_db, phase_bits,
                     tx_power_w_per_elem, pa_class, prf_hz, n_pulses,
                     digitization_level, adc_enob, subarray_nx, subarray_ny,
                     verbose=False):
    arch, snx, sny = build_arch(nx, ny, dx, dy, taper_type, taper_sll_db, phase_bits,
                                 tx_power_w_per_elem, pa_class, digitization_level, adc_enob,
                                 subarray_nx, subarray_ny)

    duty = (CTX["tau_us"] * 1e-6) * prf_hz

    antenna_model = PhasedArrayAdapter(use_analytical_fallback=True)
    radar_model = RadarModel()
    power_model = PowerModel()

    results = {}

    # ---- nominal pattern metrics (no failures) at scan=0 for gain/SLL baseline ----
    class DummyScenario:
        pass

    def make_scenario(scan_deg, range_km, rcs_dbsm, sea_state, target_height_m):
        clutter_type = "sea" if sea_state > 0 else "none"
        return RadarDetectionScenario(
            freq_hz=CTX["freq_hz"], bandwidth_hz=CTX["bandwidth_hz"],
            range_m=range_km * 1e3, target_rcs_dbsm=rcs_dbsm,
            pfa=CTX["pfa"], pd_required=CTX["pd_required"],
            n_pulses=n_pulses, swerling=1, duty_cycle=duty,
            scan_angle_deg=scan_deg, integration_type="noncoherent",
            clutter_type=clutter_type, sea_state=sea_state,
            antenna_height_m=CTX["antenna_height_m"], target_height_m=target_height_m,
            cfar_type=CTX["cfar_type"], cfar_ref_cells=CTX["cfar_ref_cells"],
            cfar_guard_cells=CTX["cfar_guard_cells"],
            prf_hz=prf_hz, search_az_extent_deg=CTX["az_extent_deg"],
            search_el_extent_deg=CTX["el_extent_deg"], beam_overhead_us=CTX["beam_overhead_us"],
            search_frame_time_ms=CTX["frame_time_ms"],
        )

    # ---- worst-case Pd across envelope points and failure seeds ----
    worst_pd = 1.0
    worst_pd_point = None
    worst_sll = -999.0
    worst_gain = None
    worst_timeline_occ = 0.0
    nan_hit = False

    for pt in ENV_POINTS:
        scenario = make_scenario(pt["scan_deg"], pt["range_km"], pt["rcs_dbsm"], pt["sea_state"], pt["target_height_m"])
        # no-failure baseline for timeline / gain crosscheck bookkeeping
        ctx0 = {}
        ant0 = antenna_model.evaluate(arch, scenario, ctx0)
        # timeline occupancy at this scan (worst over points)
        bw_az = ant0["beamwidth_az_deg"]; bw_el = ant0["beamwidth_el_deg"]
        if math.isnan(bw_az) or math.isnan(bw_el):
            nan_hit = True
            worst_timeline_occ = float("inf")
        else:
            beam_sr = math.radians(bw_az) * math.radians(bw_el)
            scan_vol_sr = math.radians(CTX["az_extent_deg"]) * math.radians(CTX["el_extent_deg"])
            dwell_time_us = n_pulses / prf_hz * 1e6
            timeline = max_update_rate(scan_vol_sr, beam_sr, dwell_time_us, CTX["beam_overhead_us"])
            occ = timeline["frame_time_ms"] / CTX["frame_time_ms"]
            worst_timeline_occ = max(worst_timeline_occ, occ)

        for seed in FAILURE_SEEDS:
            ctxf = {"failure_rate": FAILURE_RATE, "meta.seed": seed}
            antf = antenna_model.evaluate(arch, scenario, ctxf)
            if math.isnan(antf["beamwidth_az_deg"]) or math.isnan(antf["beamwidth_el_deg"]):
                nan_hit = True
                continue
            radar_ctx = dict(antf)
            rad = radar_model.evaluate(arch, scenario, radar_ctx)
            pd = rad["pd_achieved"]
            if math.isnan(pd):
                nan_hit = True
                continue
            if pd < worst_pd:
                worst_pd = pd
                worst_pd_point = (pt, seed)
            if antf["sll_db"] > worst_sll:
                worst_sll = antf["sll_db"]

    # gain crosscheck: with pattern integration (already used), g_ant_db already from full pattern -> crosscheck disagreement ~0
    # grating margin at worst scan (40 deg, matches scan_limit_deg used)
    from phased_array_systems.models.antenna.grating import check_grating_lobes
    grating_info = check_grating_lobes(dx, dy, 40.0)
    grating_margin = grating_info["max_safe_spacing_lambda"] - max(dx, dy)

    # power & cost at nominal scenario (duty cycle based)
    nominal_scenario = make_scenario(25.0, 13.0, 6.0, 3, 5.0)
    ctx_nom = antenna_model.evaluate(arch, nominal_scenario, {})
    power_metrics = power_model.evaluate(arch, nominal_scenario, ctx_nom)
    prime_power_w = power_metrics["prime_power_w"]

    cost_usd = unit_cost(arch, pa_class, phase_bits, subarray_nx, subarray_ny)

    # reliability: thermal feed-forward
    dc_w = power_metrics["dc_power_w"]; rf_avg_w = power_metrics["rf_avg_power_w"]
    heat_per_elem_w = max(0.0, dc_w - rf_avg_w) / arch.array.n_elements
    op_temp_c = RELIAB["ambient_temp_c"] + RELIAB["thermal_resistance_c_per_w"] * heat_per_elem_w
    comp_mtbfs = dict(COMPONENT_MTBFS_BASE); comp_mtbfs["pa"] = PA_CLASSES[pa_class]["pa_mtbf_hours"]
    spec = TRMReliabilitySpec(component_mtbfs=comp_mtbfs, operating_temp_c=op_temp_c,
                               mttr_hours=RELIAB["mttr_hours"], mission_hours=RELIAB["mission_hours"])
    rel = analyze_array_reliability(arch.array.n_elements, spec, original_sll_db=worst_sll)

    out = dict(
        n_elements=arch.array.n_elements,
        worst_case_pd=worst_pd, worst_pd_point=worst_pd_point,
        worst_case_pattern_sll_db=worst_sll,
        worst_case_timeline_occupancy=worst_timeline_occ,
        grating_margin_lambda=grating_margin,
        prime_power_w=prime_power_w,
        unit_cost_usd=cost_usd,
        array_availability=rel.availability,
        operating_temp_c=op_temp_c,
        duty_cycle=duty,
        nan_hit=nan_hit,
    )
    if verbose:
        for k, v in out.items():
            print(f"{k}: {v}")
    return out


if __name__ == "__main__":
    r = evaluate_design(
        nx=64, ny=32, dx=0.5, dy=0.5, taper_type="taylor", taper_sll_db=-30.0, phase_bits=4,
        tx_power_w_per_elem=8.0, pa_class="B", prf_hz=3000.0, n_pulses=8,
        digitization_level="subarray", adc_enob=12.0, subarray_nx=8, subarray_ny=8,
        verbose=True,
    )
