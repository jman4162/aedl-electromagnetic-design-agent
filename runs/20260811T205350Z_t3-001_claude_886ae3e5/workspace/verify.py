import math
import numpy as np
import phased_array as pa
from phased_array_systems.models.comms import compute_atmospheric_loss, compute_rain_loss, compute_fspl
from phased_array_systems.models.antenna.metrics import _first_null_index

C = 299792458.0
FREQ = 28e9
LAM = C / FREQ
K = 2 * math.pi / LAM
BW = 50e6
REQ_SNR = 6.0
SAT_G = 38.0
SAT_NF = 2.5
FEED_LOSS = 1.5
KB = 1.380649e-23

PARTS = {
    "element_base_cost_usd": 55.0,
    "pa_classes": {
        "A": {"pa_efficiency": 0.15, "cost_adder_usd_per_elem": 15.0},
        "B": {"pa_efficiency": 0.25, "cost_adder_usd_per_elem": 45.0},
        "C": {"pa_efficiency": 0.35, "cost_adder_usd_per_elem": 110.0},
    },
    "phase_shifter": {3: 5.0, 4: 12.0, 5: 25.0, 6: 50.0},
    "adc_cost_usd_per_channel": 40.0,
    "rx_power_w_per_elem": 0.15,
}

ENVELOPE = [
    dict(scan=0.0, elevation=88.0, range_km=603.0, rain=0.0, sky=152.0),
    dict(scan=17.0, elevation=71.0, range_km=632.0, rain=0.0, sky=160.0),
    dict(scan=33.5, elevation=52.0, range_km=741.0, rain=0.8, sky=205.0),
    dict(scan=41.0, elevation=41.0, range_km=856.0, rain=1.2, sky=228.0),
    dict(scan=48.5, elevation=33.0, range_km=1004.0, rain=1.7, sky=244.0),
    dict(scan=54.0, elevation=27.5, range_km=1153.0, rain=2.0, sky=259.0),
    dict(scan=58.0, elevation=22.5, range_km=1311.0, rain=2.3, sky=269.0),
    dict(scan=60.0, elevation=20.0, range_km=1390.0, rain=2.5, sky=275.0),
    dict(scan=60.0, elevation=20.0, range_km=1390.0, rain=0.0, sky=190.0),
]
FAILURE_SEEDS = [101, 211, 307, 401, 503]
FAILURE_RATE = 0.02
SCAN_LIMIT_DEG = 60.0
Q = 1.3


def taper_weights(taper_type, nx, ny, sll_db):
    if taper_type == "uniform":
        return np.ones(nx * ny, dtype=complex)
    elif taper_type == "taylor":
        return pa.taylor_taper_2d(nx, ny, sidelobe_dB=sll_db).ravel().astype(complex)
    elif taper_type == "chebyshev":
        return pa.chebyshev_taper_2d(nx, ny, sidelobe_dB=sll_db).ravel().astype(complex)
    else:
        raise ValueError(taper_type)


def build_weights(arch, scan_deg, seed):
    nx, ny = arch["nx"], arch["ny"]
    geom = pa.create_rectangular_array(nx, ny, arch["dx_lambda"], arch["dy_lambda"], wavelength=LAM)
    tw = taper_weights(arch["taper_type"], nx, ny, arch["taper_sll_db"])
    # taper efficiency (linear), for reference
    taper_eff = (np.abs(np.sum(tw)) ** 2 / len(tw)) / np.sum(np.abs(tw) ** 2)
    sv = pa.steering_vector(K, geom.x, geom.y, scan_deg, 0.0)
    weights = tw * sv
    weights = pa.quantize_phase(weights, n_bits=arch["phase_bits"])
    if seed is not None:
        weights, fail_mask = pa.simulate_element_failures(weights, FAILURE_RATE, seed=seed)
        n_failed = int(np.sum(fail_mask))
    else:
        n_failed = 0
    return geom, weights, taper_eff, n_failed


def compute_directivity_local(theta, phi, pattern):
    # Same algorithm as phased_array.compute_directivity, but using
    # np.trapezoid since np.trapz was removed in numpy 2.x.
    power = np.abs(pattern) ** 2
    peak_power = np.max(power)
    d_theta = theta[1, 0] - theta[0, 0] if theta.shape[0] > 1 else np.pi
    d_phi = phi[0, 1] - phi[0, 0] if phi.shape[1] > 1 else 2 * np.pi
    integrand = power * np.sin(theta)
    total_power = np.trapezoid(np.trapezoid(integrand, dx=d_phi, axis=1), dx=d_theta)
    if total_power > 0:
        return 4 * np.pi * peak_power / total_power
    return 1.0


def full_pattern_metrics(geom, weights, scan_deg, n_theta=361, n_phi=721):
    theta_1d, phi_1d, theta_grid, phi_grid = pa.create_theta_phi_grid(
        (0, math.pi / 2), (0, 2 * math.pi), n_theta, n_phi
    )
    cplx = pa.total_pattern(
        theta_grid, phi_grid, geom.x, geom.y, weights, K,
        element_pattern_func=pa.element_pattern, cos_exp_theta=Q,
    )
    directivity_lin = compute_directivity_local(theta_grid, phi_grid, cplx)
    directivity_db = 10 * math.log10(directivity_lin)

    power_db = 20 * np.log10(np.abs(cplx) + 1e-30)
    power_db -= np.max(power_db)

    peak_idx = np.unravel_index(np.argmax(power_db), power_db.shape)
    it, ip = peak_idx

    # theta cut through peak phi
    theta_cut = power_db[:, ip]
    lo = _first_null_index(theta_cut, it, -1)
    hi = _first_null_index(theta_cut, it, +1)
    lo = 0 if lo is None else lo
    hi = len(theta_cut) - 1 if hi is None else hi
    theta_lo, theta_hi = theta_1d[lo], theta_1d[hi]

    # phi cut through peak theta; phi_1d[0]==0 and phi_1d[-1]==2*pi are the
    # same physical angle, so roll the peak to mid-array before searching
    # for nulls to avoid a spurious boundary at the wrap point.
    phi_cut = power_db[it, :]
    phi_len = len(phi_cut)
    dphi_step = phi_1d[1] - phi_1d[0]
    shift = phi_len // 2 - ip
    phi_cut_rolled = np.roll(phi_cut, shift)
    ip_rolled = (ip + shift) % phi_len
    plo_r = _first_null_index(phi_cut_rolled, ip_rolled, -1)
    phigo_hi_r = _first_null_index(phi_cut_rolled, ip_rolled, +1)
    plo_r = 0 if plo_r is None else plo_r
    phigo_hi_r = phi_len - 1 if phigo_hi_r is None else phigo_hi_r
    phi_half = max(ip_rolled - plo_r, phigo_hi_r - ip_rolled) * 1.2 * dphi_step

    # exclusion window widened by 20% margin beyond first nulls
    theta_half = max(theta_1d[it] - theta_lo, theta_hi - theta_1d[it]) * 1.2

    dphi = np.abs(((phi_grid - phi_1d[ip] + math.pi) % (2 * math.pi)) - math.pi)
    mask = (np.abs(theta_grid - theta_1d[it]) <= theta_half) & (dphi <= phi_half)
    sidelobe_region = power_db[~mask]
    sll_db = float(np.max(sidelobe_region)) if sidelobe_region.size else -math.inf

    return directivity_db, sll_db


def pattern_table(arch, seeds=(None,) + tuple(FAILURE_SEEDS)):
    scans = sorted({e["scan"] for e in ENVELOPE})
    table = {}
    for scan in scans:
        for seed in seeds:
            geom, weights, taper_eff, n_failed = build_weights(arch, scan, seed)
            g_tx_db, sll_db = full_pattern_metrics(geom, weights, scan)
            table[(scan, seed)] = (g_tx_db, sll_db)
    return table


def link_margin_from_gain(arch, pa_class, tx_power_w_per_elem, env, g_tx_db):
    n_elements = arch["nx"] * arch["ny"]
    tx_power_total_dbw = 10 * math.log10(tx_power_w_per_elem * n_elements)
    eirp_dbw = tx_power_total_dbw + g_tx_db - FEED_LOSS

    range_m = env["range_km"] * 1000.0
    fspl = compute_fspl(FREQ, range_m)
    atmo = compute_atmospheric_loss(FREQ, range_m, elevation_deg=env["elevation"])
    rain = compute_rain_loss(FREQ, range_m, rain_rate_mmh=env["rain"])
    total_loss = fspl + atmo + rain

    rx_power_dbw = eirp_dbw - total_loss + SAT_G

    noise_factor = 10 ** (SAT_NF / 10)
    t_sys = env["sky"] + 290 * (noise_factor - 1)
    n_dbw = 10 * math.log10(KB * t_sys * BW)

    snr_db = rx_power_dbw - n_dbw
    margin_db = snr_db - REQ_SNR
    return margin_db


def grating_margin(dx_lambda, dy_lambda, scan_limit_deg=SCAN_LIMIT_DEG):
    sin_max = math.sin(math.radians(scan_limit_deg))
    max_safe = 1.0 / (1.0 + sin_max)
    return max_safe - max(dx_lambda, dy_lambda)


def cost_and_power(arch, pa_class, tx_power_w_per_elem, digitization_level, subarray_nx, subarray_ny):
    n_elements = arch["nx"] * arch["ny"]
    pa_eff = PARTS["pa_classes"][pa_class]["pa_efficiency"]
    pa_cost = PARTS["pa_classes"][pa_class]["cost_adder_usd_per_elem"]
    ps_cost = PARTS["phase_shifter"][arch["phase_bits"]]

    if digitization_level == "element":
        n_channels = n_elements
    elif digitization_level == "subarray":
        n_channels = (arch["nx"] // subarray_nx) * (arch["ny"] // subarray_ny)
    elif digitization_level == "analog":
        n_channels = 1
    else:
        raise ValueError(digitization_level)

    unit_cost = n_elements * (PARTS["element_base_cost_usd"] + pa_cost + ps_cost) + n_channels * PARTS["adc_cost_usd_per_channel"]
    prime_power = n_elements * (tx_power_w_per_elem / pa_eff) + n_elements * PARTS["rx_power_w_per_elem"]
    return unit_cost, prime_power, n_channels


def evaluate_design(arch, pa_class, tx_power_w_per_elem, digitization_level, subarray_nx, subarray_ny,
                     verbose=False, table=None):
    if table is None:
        table = pattern_table(arch)

    worst_margin = math.inf
    worst_sll = -math.inf
    worst_env = None
    worst_seed = None
    for env in ENVELOPE:
        for seed in (None,) + tuple(FAILURE_SEEDS):
            g_tx_db, sll_db = table[(env["scan"], seed)]
            margin = link_margin_from_gain(arch, pa_class, tx_power_w_per_elem, env, g_tx_db)
            if margin < worst_margin:
                worst_margin = margin
                worst_env = env
                worst_seed = seed
            if sll_db > worst_sll:
                worst_sll = sll_db

    unit_cost, prime_power, n_channels = cost_and_power(
        arch, pa_class, tx_power_w_per_elem, digitization_level, subarray_nx, subarray_ny
    )
    gm = grating_margin(arch["dx_lambda"], arch["dy_lambda"])

    if verbose:
        print(f"worst_margin_db={worst_margin:.3f} at env={worst_env} seed={worst_seed}")
        print(f"worst_sll_db={worst_sll:.3f}")
        print(f"unit_cost_usd={unit_cost:.1f}  prime_power_w={prime_power:.1f}  n_channels={n_channels}")
        print(f"grating_margin_lambda={gm:.4f}")

    return dict(
        worst_margin=worst_margin, worst_sll=worst_sll, unit_cost=unit_cost,
        prime_power=prime_power, grating_margin=gm, n_channels=n_channels,
    )
