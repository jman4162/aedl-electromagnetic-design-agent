"""The t1-001 cavity model against textbook worked examples.

Every closed form in :mod:`aedl.evaluators.patch_cavity` is checked against a
published number before it is trusted to score anything: Balanis's worked
patch (Antenna Theory, 4th ed., examples 14.1 and 14.3) and Pozar's 50-ohm
line (Microwave Engineering, 4th ed., example 3.7).
"""

from __future__ import annotations

import numpy as np
import pytest

from aedl.evaluators import patch_cavity as pc

# Balanis example 14.1/14.3: RT/duroid 5880, 10 GHz.
BALANIS_W = 1.186e-2
BALANIS_L = 0.906e-2
BALANIS_H = 0.1588e-2
BALANIS_EPS = 2.2


def test_slot_conductances_match_balanis_example_14_3():
    g1, g12 = pc.slot_conductances(10e9, BALANIS_W, BALANIS_L)
    assert g1 == pytest.approx(1.57e-3, rel=0.01)
    # Balanis: R_in = 1 / (2 (G1 + G12)) = 228.3 ohm at the radiating edge.
    assert 1.0 / (2.0 * (g1 + g12)) == pytest.approx(228.3, rel=0.005)


def test_length_extension_matches_balanis_example_14_1():
    eps_eff = pc.eps_eff_static(BALANIS_W / BALANIS_H, BALANIS_EPS)
    assert pc.length_extension(BALANIS_W, BALANIS_H, eps_eff) == pytest.approx(0.081e-2, rel=0.01)


def test_static_resonance_of_balanis_patch_is_10_ghz():
    f_res = pc.resonant_frequency(BALANIS_W, BALANIS_L, BALANIS_H, BALANIS_EPS, dispersive=False)
    assert f_res == pytest.approx(10e9, rel=0.01)


def test_hammerstad_jensen_line_matches_pozar_example():
    # Pozar ex. 3.7: 50 ohm on eps_r 2.2, d = 0.159 cm -> W = 0.49 cm (W/d = 3.081).
    u = 3.081
    assert pc.z0_static(u, 2.2) == pytest.approx(50.0, rel=0.01)
    assert pc.eps_eff_static(u, 2.2) == pytest.approx(1.87, rel=0.01)


class TestDispersion:
    def test_low_frequency_limit_is_static(self):
        u, eps_r, h = 13.8, 3.66, 0.254e-3
        static = pc.eps_eff_static(u, eps_r)
        assert pc.eps_eff_dispersive(u, eps_r, 1e6, h) == pytest.approx(static, rel=1e-4)

    def test_rises_with_frequency_toward_eps_r(self):
        u, eps_r, h = 13.8, 3.66, 0.254e-3
        values = [pc.eps_eff_dispersive(u, eps_r, f, h) for f in (1e9, 10e9, 28e9, 60e9)]
        assert values == sorted(values)
        assert values[-1] < eps_r


def _geometry(r_target: float = 50.0) -> pc.PatchGeometry:
    """Balanis patch fed at the inset where the cos^4 taper gives r_target."""
    return pc.PatchGeometry(
        width_m=BALANIS_W,
        length_m=BALANIS_L,
        inset_m=0.0,
        feed_width_m=0.49e-2,
        feed_length_m=1.0e-2,
        height_m=BALANIS_H,
        eps_r=BALANIS_EPS,
        tan_delta=0.0009,
        sigma_s_per_m=5.8e7,
    )


def test_quality_factors_combine_in_parallel():
    model = pc.cavity_model(_geometry())
    assert model.q_total < min(model.q_rad, model.q_cond, model.q_diel)
    expected = 1.0 / (1.0 / model.q_rad + 1.0 / model.q_cond + 1.0 / model.q_diel)
    assert model.q_total == pytest.approx(expected)


def test_conductor_q_scales_with_substrate_height():
    thin = pc.cavity_model(_geometry())
    thick_geom = pc.PatchGeometry(**{**_geometry().__dict__, "height_m": 2 * BALANIS_H})
    thick = pc.cavity_model(thick_geom)
    # Q_c = h sqrt(pi f mu0 sigma); the thicker patch also resonates lower.
    expected = 2.0 * (thick.f_res_hz / thin.f_res_hz) ** 0.5
    assert thick.q_cond / thin.q_cond == pytest.approx(expected, rel=1e-9)


def test_losses_lower_the_edge_resistance():
    model = pc.cavity_model(_geometry())
    g1, g12 = pc.slot_conductances(model.f_res_hz, BALANIS_W, BALANIS_L)
    lossless = 1.0 / (2.0 * (g1 + g12))
    assert model.r_edge_ohm == pytest.approx(lossless * model.q_total / model.q_rad)
    assert model.r_edge_ohm < lossless


def test_matched_resonator_on_matched_line_reflects_nothing():
    """At resonance a 50-ohm patch on a 50-ohm line is a 50-ohm load."""
    geom = _geometry()
    model = pc.cavity_model(geom)
    matched = pc.CavityModel(**{**model.__dict__, "r_in_ohm": model.z_feed_ohm})
    s11 = pc.port_s11(matched, geom, np.array([model.f_res_hz]), z_ref_ohm=model.z_feed_ohm)
    assert 20 * np.log10(abs(s11[0])) < -60.0


def test_resonance_shift_moves_the_match():
    geom = _geometry()
    model = pc.cavity_model(geom)
    f = np.array([model.f_res_hz])
    centred = abs(pc.port_s11(model, geom, f, z_ref_ohm=50.0)[0])
    shifted = abs(pc.port_s11(model, geom, f, z_ref_ohm=50.0, f_res_scale=1.03)[0])
    assert shifted > centred
