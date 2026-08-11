"""
Design a 2-bit phase-only steering weight set for a 16x16 half-wave array,
robust to 13 dead elements, meeting sidelobe/directivity/pointing specs.
"""
import numpy as np
import phased_array as pa

rng = np.random.default_rng(0)

# ---- Context ----
NX, NY = 16, 16
DX_WL, DY_WL = 0.5, 0.5
FREQ_HZ = 10.0e9
C = 3e8
WAVELENGTH = C / FREQ_HZ
THETA0, PHI0 = 27.0, 10.0
FAILED = np.array([7, 23, 42, 58, 77, 96, 113, 128, 150, 171, 199, 220, 245])
EXCL_DEG = 8.0
N_BITS = 2
N_STATES = 2 ** N_BITS
STEP = 2 * np.pi / N_STATES  # 90 deg

geom = pa.create_rectangular_array(NX, NY, DX_WL, DY_WL, wavelength=WAVELENGTH, center=True)
k = pa.wavelength_to_k(WAVELENGTH)
x, y = geom.x, geom.y
N = x.size

alive = np.ones(N, dtype=bool)
alive[FAILED] = False

ideal = pa.steering_vector(k, x, y, THETA0, PHI0)
ideal_phase = np.angle(ideal)

# ---- angular grid for optimization (forward hemisphere only, near eval res) ----
theta_deg_c = np.linspace(0, 90, 181)   # 0.5 deg
phi_deg_c = np.linspace(0, 360, 360, endpoint=False)  # 1 deg
TH, PH = np.meshgrid(np.deg2rad(theta_deg_c), np.deg2rad(phi_deg_c), indexing='ij')
th_flat = TH.ravel()
ph_flat = PH.ravel()

u = np.sin(th_flat) * np.cos(ph_flat)
v = np.sin(th_flat) * np.sin(ph_flat)
EXP = np.exp(1j * k * (np.outer(u, x) + np.outer(v, y)))  # (n_angles, N)
EP = np.cos(th_flat)  # cos_q, q=1, max_gain 0dBi -> linear 1
EP = np.where(np.cos(th_flat) > 0, EP, 0.0)

theta0_r, phi0_r = np.deg2rad(THETA0), np.deg2rad(PHI0)
cos_ang = (np.cos(th_flat) * np.cos(theta0_r) +
           np.sin(th_flat) * np.sin(theta0_r) * np.cos(ph_flat - phi0_r))
cos_ang = np.clip(cos_ang, -1, 1)
ang_from_target = np.rad2deg(np.arccos(cos_ang))
sidelobe_mask = ang_from_target > EXCL_DEG


def quantize(phases):
    return np.round(phases / STEP) * STEP


def weights_from_phase(phase):
    w = np.exp(1j * phase)
    w[~alive] = 0.0
    return w


def psl_db(phase):
    w = weights_from_phase(phase)
    AF = EXP @ w
    P = EP * AF
    mag2 = np.abs(P) ** 2
    peak = mag2.max()
    sl = mag2[sidelobe_mask].max()
    return 10 * np.log10(sl / peak)


# ---- init: quantized ideal steering ----
phase = quantize(ideal_phase)
phase[~alive] = quantize(ideal_phase)[~alive]

w = weights_from_phase(phase)
AF = EXP @ w
print("baseline PSL (dB):", psl_db(phase))

# ---- greedy coordinate descent over active elements, 4-state search ----
states = (np.arange(N_STATES) * STEP)  # 0,90,180,270 in rad
active_idx = np.where(alive)[0]

AF_cur = EXP @ weights_from_phase(phase)


def psl_from_AF(AF_arr):
    P = EP * AF_arr
    mag2 = np.abs(P) ** 2
    peak = mag2.max()
    sl = mag2[sidelobe_mask].max()
    return 10 * np.log10(sl / peak), peak


best_psl, _ = psl_from_AF(AF_cur)
print("start PSL:", best_psl)

n_passes = 12
for p in range(n_passes):
    order = active_idx.copy()
    rng.shuffle(order)
    improved = False
    for n in order:
        w_old = np.exp(1j * phase[n])
        best_state = phase[n]
        best_local_psl = None
        for s in states:
            w_new = np.exp(1j * s)
            AF_trial = AF_cur + (w_new - w_old) * EXP[:, n]
            trial_psl, _ = psl_from_AF(AF_trial)
            if best_local_psl is None or trial_psl < best_local_psl:
                best_local_psl = trial_psl
                best_state = s
                best_AF = AF_trial
        if best_state != phase[n]:
            phase[n] = best_state
            AF_cur = best_AF
            improved = True
    cur_psl, _ = psl_from_AF(AF_cur)
    print(f"pass {p}: PSL = {cur_psl:.3f} dB")
    if cur_psl < -18.0:
        break
    if not improved:
        break

final_psl, _ = psl_from_AF(AF_cur)
print("final coarse-grid PSL (dB):", final_psl)

# ---- build submission weights ----
final_phase = quantize(phase)  # ensure exactly on grid (should already be)
weights = np.exp(1j * final_phase)  # unit amplitude everywhere, incl. dead (per instructions)

# sanity: phase grid error
grid_err = np.max(np.abs(((final_phase - quantize(final_phase) + np.pi) % (2*np.pi)) - np.pi))
print("max phase grid error (rad):", grid_err)

np.savez("submission.npz", weights=weights.astype(np.complex128))
print("saved submission.npz, weights shape:", weights.shape)
