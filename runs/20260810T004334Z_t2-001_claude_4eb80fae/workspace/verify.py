import numpy as np
import phased_array as pa
from phased_array.core import element_pattern, array_factor_vectorized, compute_directivity
from phased_array.utils import create_theta_phi_grid

NX, NY = 16, 16
DX_WL, DY_WL = 0.5, 0.5
FREQ_HZ = 10.0e9
C = 3e8
WAVELENGTH = C / FREQ_HZ
THETA0, PHI0 = 27.0, 10.0
FAILED = np.array([7, 23, 42, 58, 77, 96, 113, 128, 150, 171, 199, 220, 245])
EXCL_DEG = 8.0
N_BITS = 2
STEP = 2 * np.pi / (2 ** N_BITS)

geom = pa.create_rectangular_array(NX, NY, DX_WL, DY_WL, wavelength=WAVELENGTH, center=True)
k = pa.wavelength_to_k(WAVELENGTH)
x, y = geom.x, geom.y
N = x.size
alive = np.ones(N, dtype=bool)
alive[FAILED] = False

data = np.load("submission.npz")
weights = data["weights"]
assert weights.shape == (256,), weights.shape
assert weights.dtype == np.complex128 or np.iscomplexobj(weights)

# amplitude error (working elements)
amp_err = np.max(np.abs(np.abs(weights[alive]) - 1.0))
print("amplitude_error (working elems):", amp_err)

# phase grid error (deg), working elements
ph = np.angle(weights[alive])
q = np.round(ph / STEP) * STEP
err_deg = np.rad2deg(np.max(np.abs(((ph - q + np.pi) % (2*np.pi)) - np.pi)))
print("phase_grid_error_deg (working elems):", err_deg)
ph_all = np.angle(weights)
q_all = np.round(ph_all / STEP) * STEP
err_deg_all = np.rad2deg(np.max(np.abs(((ph_all - q_all + np.pi) % (2*np.pi)) - np.pi)))
print("phase_grid_error_deg (all elems):", err_deg_all)

# apply evaluator-style failures: zero dead elements
w_eval = weights.copy()
w_eval[~alive] = 0.0

# fine grid matching evaluator params
theta_1d, phi_1d, theta_g, phi_g = create_theta_phi_grid(
    theta_range=(0, np.pi), phi_range=(0, 2*np.pi), n_theta=361, n_phi=721
)

AF = array_factor_vectorized(theta_g, phi_g, x, y, w_eval, k)
EP = element_pattern(theta_g, phi_g, cos_exp_theta=1.0, max_gain_dBi=0.0)
P = EP * AF
mag2 = np.abs(P) ** 2

# peak direction
idx = np.unravel_index(np.argmax(mag2), mag2.shape)
peak_theta = theta_g[idx]
peak_phi = phi_g[idx]
peak_val = mag2[idx]

theta0_r, phi0_r = np.deg2rad(THETA0), np.deg2rad(PHI0)
cos_ang = (np.cos(peak_theta) * np.cos(theta0_r) +
           np.sin(peak_theta) * np.sin(theta0_r) * np.cos(peak_phi - phi0_r))
peak_dir_err_deg = np.rad2deg(np.arccos(np.clip(cos_ang, -1, 1)))
print("peak_direction_error_deg:", peak_dir_err_deg)
print("peak at theta,phi (deg):", np.rad2deg(peak_theta), np.rad2deg(peak_phi))

# sidelobe: exclude cone of EXCL_DEG around target
cos_ang_grid = (np.cos(theta_g) * np.cos(theta0_r) +
                np.sin(theta_g) * np.sin(theta0_r) * np.cos(phi_g - phi0_r))
cos_ang_grid = np.clip(cos_ang_grid, -1, 1)
ang_from_target = np.rad2deg(np.arccos(cos_ang_grid))
sl_mask = ang_from_target > EXCL_DEG
psl_db = 10 * np.log10(mag2[sl_mask].max() / peak_val)
print("peak_sidelobe_level_db:", psl_db)

# directivity (manual trapz since this numpy build lacks np.trapz)
d_theta = theta_1d[1] - theta_1d[0]
d_phi = phi_1d[1] - phi_1d[0]
integrand = mag2 * np.sin(theta_g)
total_power = np.trapezoid(np.trapezoid(integrand, dx=d_phi, axis=1), dx=d_theta)
directivity = 4 * np.pi * peak_val / total_power
directivity_dbi = 10 * np.log10(directivity)
print("directivity_dbi:", directivity_dbi)

print()
print("=== requirement check ===")
print("steering <=1.5:", peak_dir_err_deg <= 1.5)
print("sidelobes <=-14.0:", psl_db <= -14.0)
print("directivity >=25.5:", directivity_dbi >= 25.5)
print("phase-quantization <=0.1:", err_deg_all <= 0.1)
print("phase-only-control <=0.001:", amp_err <= 0.001)
