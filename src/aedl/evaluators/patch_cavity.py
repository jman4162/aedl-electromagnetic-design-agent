"""Analytic cavity model of an inset-fed rectangular microstrip patch.

Used by the ``patch_element`` evaluator (task t1-001) at its ``analytic``
fidelity tier. Pure NumPy; nothing here imports antenna-cad or any tool
under evaluation. The model takes physical dimensions only, so the
resonance it reports is computed from geometry and never from the
frequency a submission claims to have designed for.

Formulas (SI units throughout):

- Static effective permittivity and characteristic impedance of a
  microstrip of width-to-height ratio ``u``: Hammerstad and Jensen,
  "Accurate models for microstrip computer-aided design", IEEE MTT-S 1980.
- Frequency dispersion of the effective permittivity: Kirschning and
  Jansen, Electronics Letters 18(6), 1982. Valid for 0.1 <= u <= 100,
  1 <= eps_r <= 20, f*h <= 25 GHz*mm.
- Open-end length extension: Hammerstad (Balanis, Antenna Theory, 4th ed.,
  eq. 14-2).
- TM010 resonance: ``f = c / (2 (L + 2 dL) sqrt(eps_eff(f)))``, solved by
  fixed-point iteration because ``eps_eff`` depends on ``f``, with
  ``eps_eff`` the dispersive value of a line as wide as the patch.
- Radiating-slot self and mutual conductances: Balanis eqs. 14-12 and
  14-18a, by fixed-grid quadrature.
- Quality factors (Balanis section 14.2.2): radiation
  ``Q_rad = w eps0 eps_r W L / (4 h (G1 + G12))`` from the cavity's stored
  energy over the two-slot radiated power; conductor
  ``Q_c = h sqrt(pi f mu0 sigma)``; dielectric ``Q_d = 1 / tan_delta``;
  ``1/Q_t = 1/Q_rad + 1/Q_c + 1/Q_d``.
- Input resistance at the inset: ``R_edge cos^4(pi y0 / L)`` with
  ``R_edge = (Q_t / Q_rad) / (2 (G1 + G12))``. The cos^4 taper is the
  notched-inset empirical correction (Basilio et al., IEEE TAP 49(1),
  2001); a probe feed would follow cos^2.
- Input impedance: a parallel RLC, ``Z = R_in / (1 + j Q_t (f/f_r - f_r/f))``,
  transformed to the port along the feed line as a lossless line of
  Hammerstad-Jensen impedance and dispersive phase constant.

Neglected, and the direction each pushes the answer:

- surface-wave loss and feed-line radiation (Q too high, so bandwidth is
  underestimated; at 28 GHz on RO4350B the Jackson-Alexopoulos estimate puts
  the surface-wave share at about 15 % of the power for h = 0.254 mm and
  26 % for h = 0.508 mm);
- inset-notch and feed reactance (shifts the match locus off the real axis);
- feed-line loss (below 0.2 dB round trip here; would improve |S11| slightly).

Accuracy band, stated in docs/tasks/t1-001.md: resonance +/-3 % (the static
and dispersive treatments of eps_eff alone differ by 3.3 % for the t1-001
reference patch on 0.508 mm), bandwidth low by up to about 25 % (the
surface-wave omission alone). A pass at this tier is a screening verdict,
not a certification of band coverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import j0

C0 = 299_792_458.0
MU0 = 4e-7 * math.pi
EPS0 = 1.0 / (MU0 * C0**2)
ETA0 = MU0 * C0

#: Quadrature grid for the slot-conductance integrals.
_N_THETA = 4001
#: Fixed-point iterations for the dispersive resonance; converges in < 10.
_RES_ITERATIONS = 60


def eps_eff_static(u: float, eps_r: float) -> float:
    """Quasi-static effective permittivity (Hammerstad-Jensen 1980)."""
    if u <= 0:
        raise ValueError(f"width/height must be positive, got {u}")
    a = (
        1.0
        + math.log((u**4 + (u / 52.0) ** 2) / (u**4 + 0.432)) / 49.0
        + math.log(1.0 + (u / 18.1) ** 3) / 18.7
    )
    b = 0.564 * ((eps_r - 0.9) / (eps_r + 3.0)) ** 0.053
    return float((eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 10.0 / u) ** (-a * b))


def eps_eff_dispersive(u: float, eps_r: float, f_hz: float, h_m: float) -> float:
    """Effective permittivity at ``f_hz`` (Kirschning-Jansen 1982)."""
    static = eps_eff_static(u, eps_r)
    fn = f_hz / 1e9 * h_m * 1e3  # GHz * mm
    p1 = (
        0.27488
        + (0.6315 + 0.525 / (1.0 + 0.0157 * fn) ** 20) * u
        - 0.065683 * math.exp(-8.7513 * u)
    )
    p2 = 0.33622 * (1.0 - math.exp(-0.03442 * eps_r))
    p3 = 0.0363 * math.exp(-4.6 * u) * (1.0 - math.exp(-((fn / 38.7) ** 4.97)))
    p4 = 1.0 + 2.751 * (1.0 - math.exp(-((eps_r / 15.916) ** 8)))
    p = p1 * p2 * ((0.1844 + p3 * p4) * fn) ** 1.5763
    return float(eps_r - (eps_r - static) / (1.0 + p))


def z0_static(u: float, eps_r: float) -> float:
    """Quasi-static characteristic impedance in ohms (Hammerstad-Jensen 1980)."""
    f_u = 6.0 + (2.0 * math.pi - 6.0) * math.exp(-((30.666 / u) ** 0.7528))
    z01 = ETA0 / (2.0 * math.pi) * math.log(f_u / u + math.sqrt(1.0 + (2.0 / u) ** 2))
    return z01 / math.sqrt(eps_eff_static(u, eps_r))


def length_extension(width_m: float, height_m: float, eps_eff: float) -> float:
    """Open-end length extension dL in meters (Hammerstad)."""
    u = width_m / height_m
    return 0.412 * height_m * (eps_eff + 0.3) * (u + 0.264) / ((eps_eff - 0.258) * (u + 0.8))


def resonant_frequency(
    width_m: float,
    length_m: float,
    height_m: float,
    eps_r: float,
    dispersive: bool = True,
) -> float:
    """TM010 resonance in hertz from the patch dimensions alone."""
    u = width_m / height_m
    static = eps_eff_static(u, eps_r)
    # Start from the static answer; dispersion only lowers it.
    f = C0 / (
        2.0 * (length_m + 2.0 * length_extension(width_m, height_m, static)) * math.sqrt(static)
    )
    if not dispersive:
        return f
    for _ in range(_RES_ITERATIONS):
        eps = eps_eff_dispersive(u, eps_r, f, height_m)
        f_next = C0 / (
            2.0 * (length_m + 2.0 * length_extension(width_m, height_m, eps)) * math.sqrt(eps)
        )
        if abs(f_next - f) < 1.0:  # hertz
            return f_next
        f = f_next
    return f


def slot_conductances(f_hz: float, width_m: float, length_m: float) -> tuple[float, float]:
    """(G1, G12) in siemens for the two radiating slots (Balanis 14-12, 14-18a)."""
    k0 = 2.0 * math.pi * f_hz / C0
    theta = np.linspace(1e-6, math.pi - 1e-6, _N_THETA)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    slot = (np.sin(k0 * width_m / 2.0 * cos_t) / cos_t) ** 2 * sin_t**3
    norm = 120.0 * math.pi**2
    g1 = float(np.trapezoid(slot, theta)) / norm
    g12 = float(np.trapezoid(slot * j0(k0 * length_m * sin_t), theta)) / norm
    return g1, g12


@dataclass(frozen=True)
class PatchGeometry:
    """Physical dimensions of an inset-fed patch and its laminate (SI units).

    ``feed_length_m`` runs from the port to the inset point.
    """

    width_m: float
    length_m: float
    inset_m: float
    feed_width_m: float
    feed_length_m: float
    height_m: float
    eps_r: float
    tan_delta: float
    sigma_s_per_m: float


@dataclass(frozen=True)
class CavityModel:
    """The resonator's parameters, all derived from a :class:`PatchGeometry`."""

    f_res_hz: float
    eps_eff_res: float
    q_rad: float
    q_cond: float
    q_diel: float
    q_total: float
    r_edge_ohm: float
    r_in_ohm: float
    z_feed_ohm: float


def cavity_model(geom: PatchGeometry) -> CavityModel:
    """Resonance, quality factors, and inset resistance of the patch."""
    w, length, h = geom.width_m, geom.length_m, geom.height_m
    f_res = resonant_frequency(w, length, h, geom.eps_r, dispersive=True)
    g1, g12 = slot_conductances(f_res, w, length)
    omega = 2.0 * math.pi * f_res
    q_rad = omega * EPS0 * geom.eps_r * w * length / (4.0 * h * (g1 + g12))
    q_cond = h * math.sqrt(math.pi * f_res * MU0 * geom.sigma_s_per_m)
    q_diel = 1.0 / geom.tan_delta if geom.tan_delta > 0 else math.inf
    q_total = 1.0 / (1.0 / q_rad + 1.0 / q_cond + 1.0 / q_diel)
    r_edge = q_total / q_rad / (2.0 * (g1 + g12))
    r_in = r_edge * math.cos(math.pi * geom.inset_m / length) ** 4
    return CavityModel(
        f_res_hz=f_res,
        eps_eff_res=eps_eff_dispersive(w / h, geom.eps_r, f_res, h),
        q_rad=q_rad,
        q_cond=q_cond,
        q_diel=q_diel,
        q_total=q_total,
        r_edge_ohm=r_edge,
        r_in_ohm=r_in,
        z_feed_ohm=z0_static(geom.feed_width_m / h, geom.eps_r),
    )


def port_s11(
    model: CavityModel,
    geom: PatchGeometry,
    freqs_hz: NDArray[np.float64],
    z_ref_ohm: float,
    f_res_scale: float = 1.0,
) -> NDArray[np.complex128]:
    """Complex S11 at the feed port, referenced to ``z_ref_ohm``.

    ``f_res_scale`` moves the resonance by a fraction (1.03 is +3 %) with
    everything else held, which is how the evaluator reports what the
    model's resonance uncertainty does to the band.
    """
    f = np.asarray(freqs_hz, dtype=float)
    f_r = model.f_res_hz * f_res_scale
    z_patch = model.r_in_ohm / (1.0 + 1j * model.q_total * (f / f_r - f_r / f))
    u_feed = geom.feed_width_m / geom.height_m
    eps_line = np.array([eps_eff_dispersive(u_feed, geom.eps_r, fi, geom.height_m) for fi in f])
    tan_bl = np.tan(2.0 * math.pi * f * np.sqrt(eps_line) / C0 * geom.feed_length_m)
    zc = model.z_feed_ohm
    z_port = zc * (z_patch + 1j * zc * tan_bl) / (zc + 1j * z_patch * tan_bl)
    s11: NDArray[np.complex128] = np.asarray(
        (z_port - z_ref_ohm) / (z_port + z_ref_ohm), dtype=np.complex128
    )
    return s11
