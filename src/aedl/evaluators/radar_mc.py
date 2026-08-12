"""Monte Carlo radar detection cross-check.

An independent second opinion for the tier-3 radar evaluator. There is no
second radar codebase in the ecosystem (opensatcom plays that role for
links), so the cross-check simulates detection directly: Swerling-1
target draws, complex Gaussian noise per pulse, square-law noncoherent
integration, and an actual cell-averaging CFAR whose threshold comes from
the exact closed-form false-alarm statistics — not from
phased-array-systems' loss approximations, which this module deliberately
imports nothing from.

What it validates: the detection-statistics composition
``pd = f(per-pulse SCNR, Pfa, Swerling, n_pulses, CA-CFAR)`` that
``RadarModel`` computes as ``compute_pd_from_snr(scnr - cfar_loss_db, ...)``.
What it does not validate: the clutter *level* (sea sigma-0 models); the
simulation takes SCNR as given and treats clutter-plus-noise as complex
Gaussian, the same distributional assumption the analytic chain makes.

CA-CFAR threshold: with square-law noncoherent integration of N pulses,
the cell under test is Gamma(N, 1)-distributed under H0 and the sum of
n_ref reference cells is Gamma(n_ref*N, 1). For threshold T = (alpha/n_ref)*S,
P(X > cS) with X ~ Gamma(a), S ~ Gamma(b) equals
1 - I_{c/(1+c)}(a, b) (regularized incomplete beta), which is solved for
alpha by bisection. See Richards, *Fundamentals of Radar Signal
Processing*, ch. 7 (CFAR) for the setup.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, special


def ca_cfar_threshold_factor(pfa: float, n_ref: int, n_pulses: int = 1) -> float:
    """Exact CA-CFAR threshold multiplier alpha for square-law N-pulse cells.

    The detector declares a target when CUT > (alpha / n_ref) * sum(ref).
    """
    if not 0.0 < pfa < 1.0:
        raise ValueError("pfa must be in (0, 1)")
    if n_ref < 1 or n_pulses < 1:
        raise ValueError("n_ref and n_pulses must be >= 1")
    a = float(n_pulses)
    b = float(n_ref * n_pulses)

    def pfa_of(alpha: float) -> float:
        c = alpha / n_ref
        return 1.0 - float(special.betainc(a, b, c / (1.0 + c)))

    return float(optimize.brentq(lambda al: pfa_of(al) - pfa, 1e-6, 1e6))


def simulate_pd_swerling1_cfar(
    scnr_db: float,
    pfa: float,
    n_pulses: int,
    n_ref: int,
    n_trials: int = 200_000,
    seed: int = 20260812,
) -> float:
    """Monte Carlo Pd: Swerling-1 target through an actual CA-CFAR.

    One Rayleigh-amplitude draw per trial (scan-to-scan fluctuation,
    constant over the dwell's pulses), complex Gaussian unit-power noise
    per pulse, square-law sum over pulses, CA-CFAR threshold from
    noise-only reference cells. Deterministic under the fixed seed;
    standard error at 2e5 trials is ~0.001.
    """
    rng = np.random.default_rng(seed)
    alpha = ca_cfar_threshold_factor(pfa, n_ref, n_pulses)
    mean_snr = 10.0 ** (scnr_db / 10.0)

    # Swerling 1: exponential power draw per scan, constant across pulses.
    target_power = rng.exponential(mean_snr, size=(n_trials, 1))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(n_trials, n_pulses))
    signal = np.sqrt(target_power) * np.exp(1j * phases)
    noise = (
        rng.standard_normal((n_trials, n_pulses)) + 1j * rng.standard_normal((n_trials, n_pulses))
    ) / np.sqrt(2.0)
    cut = np.abs(signal + noise) ** 2
    cut_stat = cut.sum(axis=1)

    # Reference cells are noise-only, each noncoherently integrated the
    # same way. Chi-square sums drawn directly as Gamma(N) to keep memory
    # bounded (statistically identical to summing per-pulse squares).
    ref_sum = rng.gamma(shape=float(n_ref * n_pulses), scale=1.0, size=n_trials)
    threshold = (alpha / n_ref) * ref_sum
    return float(np.mean(cut_stat > threshold))
