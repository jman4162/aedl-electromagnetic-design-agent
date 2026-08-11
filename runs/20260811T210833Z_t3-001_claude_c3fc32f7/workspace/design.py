import sys
sys.path.insert(0, '/Users/johnhodge/code/Phased-Array-Antenna-Model')
import math
import numpy as np
import phased_array as pa

from phased_array_systems.models.comms.propagation import (
    compute_fspl, compute_atmospheric_loss, compute_rain_loss,
)
from phased_array_systems.models.antenna.errors import (
    phase_quantization_loss_db, phase_quantization_rms_rad, rms_sidelobe_floor_db,
)
from phased_array_systems.models.antenna.grating import check_grating_lobes
from phased_array_systems.constants import K_B, W_TO_DBW

# ---- Fixed context from BRIEF/task.yaml ----
FREQ_HZ = 28.0e9
BW_HZ = 50.0e6
REQ_SNR_DB = 6.0
SAT_RX_GAIN_DB = 38.0
SAT_NF_DB = 2.5
DUTY = 1.0

PARTS = dict(
    element_base_cost_usd=55.0,
    pa_classes={
        "A": dict(pa_efficiency=0.15, cost_adder_usd_per_elem=15.0),
        "B": dict(pa_efficiency=0.25, cost_adder_usd_per_elem=45.0),
        "C": dict(pa_efficiency=0.35, cost_adder_usd_per_elem=110.0),
    },
    phase_shifter={3: 5.0, 4: 12.0, 5: 25.0, 6: 50.0},
    adc_cost_usd_per_channel=40.0,
    rx_power_w_per_elem=0.15,
    feed_loss_db=1.5,
    tx_power_w_per_elem_max=2.0,
)

ENVELOPE = [
    dict(scan_deg=0.0,  elevation_deg=88.0, range_km=603.0,  rain_mmh=0.0, sky_temp_k=152.0),
    dict(scan_deg=17.0, elevation_deg=71.0, range_km=632.0,  rain_mmh=0.0, sky_temp_k=160.0),
    dict(scan_deg=33.5, elevation_deg=52.0, range_km=741.0,  rain_mmh=0.8, sky_temp_k=205.0),
    dict(scan_deg=41.0, elevation_deg=41.0, range_km=856.0,  rain_mmh=1.2, sky_temp_k=228.0),
    dict(scan_deg=48.5, elevation_deg=33.0, range_km=1004.0, rain_mmh=1.7, sky_temp_k=244.0),
    dict(scan_deg=54.0, elevation_deg=27.5, range_km=1153.0, rain_mmh=2.0, sky_temp_k=259.0),
    dict(scan_deg=58.0, elevation_deg=22.5, range_km=1311.0, rain_mmh=2.3, sky_temp_k=269.0),
    dict(scan_deg=60.0, elevation_deg=20.0, range_km=1390.0, rain_mmh=2.5, sky_temp_k=275.0),
    dict(scan_deg=60.0, elevation_deg=20.0, range_km=1390.0, rain_mmh=0.0, sky_temp_k=190.0),
]
FAILURE_SEEDS = [101, 211, 307, 401, 503]
FAILURE_RATE = 0.02
SCAN_LIMIT_DEG = 60.0  # evaluation_envelope_bounds.scan_angle_deg_max

WAVELENGTH = 299_792_458.0 / FREQ_HZ
K = 2 * np.pi / WAVELENGTH


def build_taper(taper_type, nx, ny, sll_db):
    if taper_type == "uniform":
        return np.ones(nx * ny)
    elif taper_type == "taylor":
        return np.asarray(pa.taylor_taper_2d(nx, ny, sidelobe_dB=sll_db)).ravel()
    elif taper_type == "chebyshev":
        return np.asarray(pa.chebyshev_taper_2d(nx, ny, sidelobe_dB=sll_db)).ravel()
    raise ValueError(taper_type)


def gain_and_sll_for_point(nx, ny, dx, dy, taper_type, taper_sll_db, phase_bits,
                            scan_deg, failure_seed=None):
    geom = pa.create_rectangular_array(nx, ny, dx, dy, wavelength=WAVELENGTH)
    taper_w = build_taper(taper_type, nx, ny, taper_sll_db)
    taper_eff = pa.compute_taper_efficiency(taper_w)
    taper_loss_db = -10 * np.log10(taper_eff) if taper_eff > 0 else 0.0

    sv = pa.steering_vector(K, geom.x, geom.y, scan_deg, 0.0)
    weights = taper_w * sv

    if phase_bits is not None:
        weights = pa.quantize_phase(weights, n_bits=phase_bits)

    n_failed = 0
    if failure_seed is not None:
        weights, fail_mask = pa.simulate_element_failures(weights, FAILURE_RATE, seed=failure_seed)
        n_failed = int(np.sum(fail_mask))

    theta_deg = np.linspace(-90, 90, 721)
    theta_rad = np.radians(theta_deg)
    phi_az = np.zeros_like(theta_rad)
    tp_az = pa.total_pattern(theta_rad, phi_az, geom.x, geom.y, weights, K,
                              element_pattern_func=pa.element_pattern, cos_exp_theta=1.3)
    tp_az_db = 20 * np.log10(np.abs(tp_az) + 1e-12)
    tp_az_db = tp_az_db - np.max(tp_az_db)

    from phased_array_systems.models.antenna.metrics import compute_sidelobe_level, compute_directivity_rectangular, compute_scan_loss
    sll = compute_sidelobe_level(tp_az_db, theta_deg)

    directivity_db = compute_directivity_rectangular(nx, ny, dx, dy)
    scan_loss_db = compute_scan_loss(scan_deg)
    quant_loss_db = phase_quantization_loss_db(phase_bits) if phase_bits else 0.0
    g_peak_db = directivity_db - scan_loss_db - taper_loss_db - quant_loss_db

    return dict(g_peak_db=g_peak_db, sll_db=sll, taper_eff=taper_eff,
                taper_loss_db=taper_loss_db, quant_loss_db=quant_loss_db,
                scan_loss_db=scan_loss_db, directivity_db=directivity_db, n_failed=n_failed)


def link_margin(g_tx_db, feed_loss_db, tx_power_total_w, range_km, elevation_deg, rain_mmh, sky_temp_k):
    tx_power_total_dbw = W_TO_DBW(tx_power_total_w)
    eirp_dbw = tx_power_total_dbw + g_tx_db - feed_loss_db

    range_m = range_km * 1000.0
    fspl_db = compute_fspl(FREQ_HZ, range_m)
    atmo_db = compute_atmospheric_loss(FREQ_HZ, range_m, elevation_deg=elevation_deg)
    rain_db = compute_rain_loss(FREQ_HZ, range_m, rain_rate_mmh=rain_mmh)
    total_path_loss_db = fspl_db + atmo_db + rain_db

    rx_power_dbw = eirp_dbw - total_path_loss_db + SAT_RX_GAIN_DB

    noise_factor = 10.0 ** (SAT_NF_DB / 10.0)
    t_sys_k = sky_temp_k + 290.0 * (noise_factor - 1.0)
    noise_power_dbw = W_TO_DBW(K_B * t_sys_k * BW_HZ)

    snr_db = rx_power_dbw - noise_power_dbw
    margin_db = snr_db - REQ_SNR_DB
    return margin_db, dict(fspl_db=fspl_db, atmo_db=atmo_db, rain_db=rain_db,
                            eirp_dbw=eirp_dbw, rx_power_dbw=rx_power_dbw,
                            noise_power_dbw=noise_power_dbw, snr_db=snr_db, t_sys_k=t_sys_k)


def evaluate_design(nx, ny, dx, dy, taper_type, taper_sll_db, phase_bits,
                     tx_power_w_per_elem, pa_class, digitization_level,
                     subarray_nx, subarray_ny, adc_enob, verbose=False):
    n_elements = nx * ny
    pa_eff = PARTS["pa_classes"][pa_class]["pa_efficiency"]
    pa_adder = PARTS["pa_classes"][pa_class]["cost_adder_usd_per_elem"]
    ps_cost = PARTS["phase_shifter"][phase_bits]

    # cost
    n_subarrays = math.ceil(nx / subarray_nx) * math.ceil(ny / subarray_ny)
    if digitization_level == "element":
        n_chan = n_elements
    elif digitization_level == "subarray":
        n_chan = n_subarrays
    else:
        n_chan = 1
    unit_cost_usd = n_elements * (PARTS["element_base_cost_usd"] + pa_adder + ps_cost) \
        + n_chan * PARTS["adc_cost_usd_per_channel"]

    # power
    rf_total_w = n_elements * tx_power_w_per_elem * DUTY
    pa_dc_w = rf_total_w / pa_eff
    rx_dc_w = n_elements * PARTS["rx_power_w_per_elem"]
    prime_power_w = pa_dc_w + rx_dc_w

    # grating
    grating_info = check_grating_lobes(dx, dy, SCAN_LIMIT_DEG)
    grating_margin_lambda = grating_info["max_safe_spacing_lambda"] - max(dx, dy)

    # worst-case over envelope x failure seeds (+ no-failure baseline)
    worst_margin = math.inf
    worst_sll = -math.inf
    worst_case_detail = None
    seeds_to_try = FAILURE_SEEDS + [None]
    for pt in ENVELOPE:
        for seed in seeds_to_try:
            g = gain_and_sll_for_point(nx, ny, dx, dy, taper_type, taper_sll_db, phase_bits,
                                        pt["scan_deg"], failure_seed=seed)
            tx_power_total_w = n_elements * tx_power_w_per_elem
            m, detail = link_margin(g["g_peak_db"], PARTS["feed_loss_db"], tx_power_total_w,
                                     pt["range_km"], pt["elevation_deg"], pt["rain_mmh"], pt["sky_temp_k"])
            if m < worst_margin:
                worst_margin = m
                worst_case_detail = (pt, seed, g, detail)
            if g["sll_db"] > worst_sll:
                worst_sll = g["sll_db"]

    result = dict(
        n_elements=n_elements,
        unit_cost_usd=unit_cost_usd,
        prime_power_w=prime_power_w,
        grating_margin_lambda=grating_margin_lambda,
        worst_case_link_margin_db=worst_margin,
        worst_case_pattern_sll_db=worst_sll,
    )
    if verbose:
        pt, seed, g, detail = worst_case_detail
        print("Worst-case point:", pt, "seed=", seed)
        print("  g_peak_db=%.3f sll_db=%.3f taper_eff=%.3f n_failed=%d" % (g["g_peak_db"], g["sll_db"], g["taper_eff"], g["n_failed"]))
        print("  ", detail)
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=32)
    p.add_argument("--ny", type=int, default=32)
    p.add_argument("--dx", type=float, default=0.5)
    p.add_argument("--dy", type=float, default=0.5)
    p.add_argument("--taper", default="taylor")
    p.add_argument("--sll", type=float, default=-22.0)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--txp", type=float, default=1.0)
    p.add_argument("--pa", default="B")
    p.add_argument("--dig", default="analog")
    p.add_argument("--subnx", type=int, default=8)
    p.add_argument("--subny", type=int, default=8)
    p.add_argument("--enob", type=float, default=8.0)
    p.add_argument("-v", action="store_true")
    args = p.parse_args()

    res = evaluate_design(args.nx, args.ny, args.dx, args.dy, args.taper, args.sll, args.bits,
                           args.txp, args.pa, args.dig, args.subnx, args.subny, args.enob, verbose=args.v)
    for k, vv in res.items():
        print(f"{k}: {vv}")
