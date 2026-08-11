import numpy as np
import phased_array as pa

# ---- context ----
Nx, Ny = 16, 16
dx_wl, dy_wl = 0.5, 0.5
freq_hz = 10.0e9
c = 299792458.0
wavelength = c / freq_hz
k = 2 * np.pi / wavelength

theta0_deg, phi0_deg = 27.0, 10.0
n_bits = 2
failed_elements = np.array([7, 23, 42, 58, 77, 96, 113, 128, 150, 171, 199, 220, 245])
exclusion_radius_deg = 8.0

geom = pa.create_rectangular_array(Nx, Ny, dx_wl, dy_wl, wavelength=wavelength)
x, y = geom.x, geom.y
n_el = geom.n_elements
assert n_el == 256

active_mask = np.ones(n_el, dtype=bool)
active_mask[failed_elements] = False

def element_pattern_func(theta, phi):
    return pa.element_pattern(theta, phi, cos_exp_theta=1.0, max_gain_dBi=0.0)


def angular_sep_deg(theta1, phi1, theta2, phi2):
    # angle between two (theta,phi) directions in radians -> degrees
    c_ang = (np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2)
             + np.cos(theta1) * np.cos(theta2))
    c_ang = np.clip(c_ang, -1.0, 1.0)
    return np.degrees(np.arccos(c_ang))


def compute_directivity_compat(theta, phi, pattern):
    # same formula as pa.compute_directivity, but using np.trapezoid
    # (this numpy version dropped np.trapz)
    power = np.abs(pattern) ** 2
    peak_power = np.max(power)
    d_theta = theta[1, 0] - theta[0, 0] if theta.shape[0] > 1 else np.pi
    d_phi = phi[0, 1] - phi[0, 0] if phi.shape[1] > 1 else 2 * np.pi
    integrand = power * np.sin(theta)
    total_power = np.trapezoid(np.trapezoid(integrand, dx=d_phi, axis=1), dx=d_theta)
    return 4 * np.pi * peak_power / total_power if total_power > 0 else 1.0


def eval_full(weights):
    """Evaluate on the full evaluator grid (n_theta=361, n_phi=721)."""
    t1, p1, theta_grid, phi_grid = pa.create_theta_phi_grid(n_theta=361, n_phi=721)
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
        'peak_direction_error_deg': peak_dir_err,
        'peak_sidelobe_level_db': psll_db,
        'directivity_dbi': directivity_dbi,
    }


# ---- baseline: direct quantization of steering vector ----
sv = pa.steering_vector(k, x, y, theta0_deg, phi0_deg)
w_direct = pa.quantize_phase(sv, n_bits)
w_direct_eval = w_direct.copy()
w_direct_eval[~active_mask] = 0.0
print("baseline direct quantization:", eval_full(w_direct_eval))
