import numpy as np
import phased_array as pa
import time

Nx, Ny = 16, 16
dx_wl, dy_wl = 0.5, 0.5
freq_hz = 10.0e9
c = 299792458.0
wavelength = c / freq_hz
k = 2 * np.pi / wavelength

theta0_deg, phi0_deg = 27.0, 10.0
n_bits = 2
n_states = 2 ** n_bits
phase_states = np.arange(n_states) * (2 * np.pi / n_states)  # 0, 90, 180, 270 deg

failed_elements = np.array([7, 23, 42, 58, 77, 96, 113, 128, 150, 171, 199, 220, 245])
exclusion_radius_deg = 8.0

geom = pa.create_rectangular_array(Nx, Ny, dx_wl, dy_wl, wavelength=wavelength)
x, y = geom.x, geom.y
n_el = geom.n_elements

active_mask = np.ones(n_el, dtype=bool)
active_mask[failed_elements] = False
active_idx = np.where(active_mask)[0]


def element_pattern_func(theta, phi):
    return pa.element_pattern(theta, phi, cos_exp_theta=1.0, max_gain_dBi=0.0)


def angular_sep_deg(theta1, phi1, theta2, phi2):
    c_ang = (np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2)
             + np.cos(theta1) * np.cos(theta2))
    c_ang = np.clip(c_ang, -1.0, 1.0)
    return np.degrees(np.arccos(c_ang))


def compute_directivity_compat(theta, phi, pattern):
    power = np.abs(pattern) ** 2
    peak_power = np.max(power)
    d_theta = theta[1, 0] - theta[0, 0] if theta.shape[0] > 1 else np.pi
    d_phi = phi[0, 1] - phi[0, 0] if phi.shape[1] > 1 else 2 * np.pi
    integrand = power * np.sin(theta)
    total_power = np.trapezoid(np.trapezoid(integrand, dx=d_phi, axis=1), dx=d_theta)
    return 4 * np.pi * peak_power / total_power if total_power > 0 else 1.0


def eval_full(weights, n_theta=361, n_phi=721):
    t1, p1, theta_grid, phi_grid = pa.create_theta_phi_grid(n_theta=n_theta, n_phi=n_phi)
    pattern = pa.total_pattern(theta_grid, phi_grid, x, y, weights, k,
                                element_pattern_func=element_pattern_func)
    power = np.abs(pattern) ** 2

    idx = np.unravel_index(np.argmax(power), power.shape)
    peak_theta, peak_phi = theta_grid[idx], phi_grid[idx]
    peak_dir_err = angular_sep_deg(peak_theta, peak_phi,
                                    np.radians(theta0_deg), np.radians(phi0_deg))

    sep = angular_sep_deg(theta_grid, phi_grid,
                           np.radians(theta0_deg), np.radians(phi0_deg))
    outside = sep > exclusion_radius_deg
    peak_power = power[idx]
    sidelobe_power = np.max(power[outside])
    psll_db = 10 * np.log10(sidelobe_power / peak_power)

    directivity = compute_directivity_compat(theta_grid, phi_grid, pattern)
    directivity_dbi = 10 * np.log10(directivity)

    return {
        'peak_direction_error_deg': float(peak_dir_err),
        'peak_sidelobe_level_db': float(psll_db),
        'directivity_dbi': float(directivity_dbi),
    }


# ---------------- optimization on a moderate-resolution grid ----------------
N_THETA_OPT, N_PHI_OPT = 181, 361  # 1 deg steps -> 65341 points
t1, p1, theta_grid, phi_grid = pa.create_theta_phi_grid(n_theta=N_THETA_OPT, n_phi=N_PHI_OPT)
theta_f = theta_grid.ravel()
phi_f = phi_grid.ravel()
n_ang = theta_f.size

EP = element_pattern_func(theta_grid, phi_grid).ravel()  # (n_ang,)

sep = angular_sep_deg(theta_f, phi_f, np.radians(theta0_deg), np.radians(phi0_deg))
mainlobe_mask = sep <= exclusion_radius_deg
sidelobe_mask = ~mainlobe_mask

u = np.sin(theta_f) * np.cos(phi_f)
v = np.sin(theta_f) * np.sin(phi_f)

# phase terms for each active element across all angles: shape (n_active, n_ang)
phase_terms = np.exp(1j * k * (np.outer(x[active_idx], u) + np.outer(y[active_idx], v)))  # complex128
# fold element pattern in here so AF*EP is what we track directly
contrib = phase_terms * EP[None, :]  # (n_active, n_ang)

n_active = active_idx.size

# initial guess: direct quantized steering phases
sv = pa.steering_vector(k, x, y, theta0_deg, phi0_deg)
w0 = pa.quantize_phase(sv, n_bits)
phase_idx = np.round(np.angle(w0[active_idx]) % (2 * np.pi) / (2 * np.pi / n_states)).astype(int) % n_states

# current total field (EP*AF) at each angle
field = np.sum(np.exp(1j * phase_states[phase_idx])[:, None] * contrib, axis=0)

# precompute per-state contribution vectors: state_contrib[s, e, :] = exp(j*state_s)*contrib[e,:]
state_phasors = np.exp(1j * phase_states)  # (n_states,)


def objective(field):
    power = np.abs(field) ** 2
    main_peak = np.max(power[mainlobe_mask])
    side_peak = np.max(power[sidelobe_mask])
    return side_peak / main_peak, side_peak, main_peak


ratio, side_peak, main_peak = objective(field)
print("init (opt grid):", 10 * np.log10(ratio))

rng = np.random.default_rng(0)
order_base = np.arange(n_active)

t_start = time.time()
n_sweeps = 15
best_ratio = ratio
for sweep in range(n_sweeps):
    order = order_base.copy()
    rng.shuffle(order)
    improved = False
    for e in order:
        old_state = phase_idx[e]
        old_val = state_phasors[old_state] * contrib[e]
        base_field = field - old_val
        best_local_ratio = None
        best_state = old_state
        best_field_candidate = None
        for s in range(n_states):
            trial_field = base_field + state_phasors[s] * contrib[e]
            power = np.abs(trial_field) ** 2
            main_peak_t = np.max(power[mainlobe_mask])
            side_peak_t = np.max(power[sidelobe_mask])
            r = side_peak_t / main_peak_t
            if best_local_ratio is None or r < best_local_ratio:
                best_local_ratio = r
                best_state = s
                best_field_candidate = trial_field
        if best_state != old_state:
            improved = True
        phase_idx[e] = best_state
        field = best_field_candidate
    ratio, side_peak, main_peak = objective(field)
    print(f"sweep {sweep}: psll={10*np.log10(ratio):.3f} dB, elapsed={time.time()-t_start:.1f}s")
    if not improved:
        print("converged, no changes this sweep")
        break

# build full weight vector: unit amplitude, on-grid phase for ALL elements
# (including the dead ones) -- this is what would actually be programmed
# into hardware; the evaluator applies the failures itself.
weights = np.ones(n_el, dtype=complex)
weights[active_idx] = np.exp(1j * phase_states[phase_idx])
dead_idx = np.where(~active_mask)[0]
dead_phase = np.angle(pa.quantize_phase(sv[dead_idx], n_bits))
weights[dead_idx] = np.exp(1j * dead_phase)

print("optimized (opt grid) psll_db:", 10 * np.log10(ratio))

# sanity: evaluate with failures actually zeroed (mimics evaluator physics)
weights_eval = weights.copy()
weights_eval[dead_idx] = 0.0
print("full-grid eval (with failures applied):", eval_full(weights_eval))

amp_err = np.max(np.abs(np.abs(weights) - 1.0))
phase_step = 2 * np.pi / n_states
ph = np.angle(weights)
grid_err = np.degrees(np.min(np.abs(((ph[:, None] - phase_states[None, :] + np.pi) % (2*np.pi)) - np.pi), axis=1))
print("amplitude_error:", amp_err, "max phase_grid_error_deg:", np.max(grid_err))

np.savez('submission.npz', weights=weights.astype(complex))
