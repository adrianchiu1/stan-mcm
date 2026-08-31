"""Numeric-pinning unit tests for the LW-SV numerical conventions
(lw-sv-spec.md §1.1, "Data and units conventions").

**Status: live, worked-example tests, wired to production code where it
exists.** Each convention keeps a small, local reference implementation
(the arithmetic pinned independently of any code path) PLUS a
`test_production_*` twin calling the real function with the same worked
numbers: convention 1 against `macrotoolkit.smoother.build_lw_matrices`
(S2), convention 2 against `macrotoolkit.smoother.sv_rw_noncentered` /
`sv_diag_variance_path` (S3). Convention 3's production transform
(deriving inflation from a price level in `macrotoolkit.data`) still does
not exist -- the toolkit takes pre-constructed inflation as input in v1 --
so only the reference implementation pins it; rewire it the same way if
that transform is ever added.

The three conventions pinned here:

1. **`g` is annualized; the state transition uses `g/4`.** Per lw-sv-spec.md
   §1.3: `y*_t = y*_{t-1} + g_{t-1}/4 + eps_y*,t`. `g` itself is reported/
   used unscaled everywhere else (priors, outputs); only the quarterly
   transition step divides by 4.
2. **`h` is a log-variance; the corresponding standard deviation is
   `exp(h/2)`.** Per lw-sv-spec.md §1.5: shock sd at time t is
   `exp(h_s,t / 2)`, explicitly *not* `exp(h_s,t)` (which would be the
   variance, not the sd) or `h_s,t` used directly.
3. **Inflation is `400 * dlog(P)`.** Per lw-sv-spec.md §1.1: annualized q/q
   inflation from a price level `P` is `400 * (log(P_t) - log(P_{t-1}))` --
   the `400` scales a quarterly log difference (`100*` for percent, `4x` for
   annualization) to an annualized percent.
"""
from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# Convention 1: g annualized, transition uses g/4
# ---------------------------------------------------------------------------


def _quarterly_increment(g_annual: float) -> float:
    """Reference implementation of the arithmetic S2's
    `y*_t = y*_{t-1} + g_{t-1}/4 + eps_y*,t` transition (spec §1.3) must
    apply to `g`: divide the annualized rate by 4 to get the quarterly
    increment."""
    return g_annual / 4.0


def test_g_is_annualized_transition_uses_g_over_4() -> None:
    assert _quarterly_increment(12.0) == pytest.approx(3.0)
    assert _quarterly_increment(2.8) == pytest.approx(0.7)

    # Negative check: g used unscaled (the classic omitted-/4 bug) would
    # give a very different, wrong number for a plausible annualized g.
    g_annual = 2.8
    wrong_unscaled = g_annual
    assert _quarterly_increment(g_annual) != pytest.approx(wrong_unscaled)


def test_production_transition_matrix_applies_g_over_4() -> None:
    """S2 rewiring: the same convention checked against the real
    `macrotoolkit.smoother.build_lw_matrices`. The transition applied to a
    clean state (y* = 100, g = 2.8 annualized, z = 0.4, no shocks) must move
    y* by exactly g/4 = 0.7 -- and the g and z states themselves are carried
    unscaled (g stays annualized everywhere; only the y* step divides by 4).
    """
    import numpy as np

    from macrotoolkit.smoother import build_lw_matrices

    params = {
        "a1": 1.2, "a2": -0.4, "a_r": -0.1, "b_pi": 0.8, "b_y": 0.15,
        "sigma_ystar": 0.4, "sigma_g": 0.12, "sigma_z": 0.08,
        "sigma_is": 0.5, "sigma_pc": 0.8,
    }
    F, Q, A, Z, R = build_lw_matrices(params, c=1.0)

    g_annual = 2.8
    xi = np.array([100.0, 99.3, 98.6, g_annual, g_annual, 0.4, 0.4])
    xi_next = F @ xi

    assert xi_next[0] == pytest.approx(100.0 + _quarterly_increment(g_annual))  # y* += g/4
    assert xi_next[0] != pytest.approx(100.0 + g_annual)  # the omitted-/4 bug
    assert xi_next[3] == pytest.approx(g_annual)  # g itself carried unscaled
    assert xi_next[5] == pytest.approx(0.4)  # z carried unscaled

    # r* = c*g + z with g annualized needs NO 4x factor (spec §1.1): the
    # IS-curve loading on the lagged g states is -(c*a_r/2), same magnitude
    # as on the z states -- a 4x mismatch between them is the bug this pins.
    assert Z[0, 3] == pytest.approx(-params["a_r"] / 2.0)
    assert Z[0, 5] == pytest.approx(-params["a_r"] / 2.0)

    # And the Q cross-term carries the /4 consistently: the y* shock is
    # eps_ystar + eps_g/4, so cov(y* shock, g shock) = sigma_g^2 / 4 and
    # var(y* shock) picks up sigma_g^2 / 16.
    assert Q[0, 3] == pytest.approx(params["sigma_g"] ** 2 / 4.0)
    assert Q[0, 0] == pytest.approx(params["sigma_ystar"] ** 2 + params["sigma_g"] ** 2 / 16.0)


# ---------------------------------------------------------------------------
# Convention 2: h is log-variance; sd = exp(h/2)
# ---------------------------------------------------------------------------


def _sd_from_log_variance(h: float) -> float:
    """Reference implementation of spec §1.5's convention: `h` is a
    log-variance, so the corresponding standard deviation is `exp(h/2)`
    (equivalently, `exp(h)` is the variance)."""
    return math.exp(h / 2.0)


def test_h_is_log_variance_sd_is_exp_h_over_2() -> None:
    assert _sd_from_log_variance(0.0) == pytest.approx(1.0)

    h = 2.0 * math.log(2.0)  # exp(h/2) = exp(ln 2) = 2
    assert _sd_from_log_variance(h) == pytest.approx(2.0)

    h2 = 2.0 * math.log(0.5)  # exp(h/2) = exp(ln 0.5) = 0.5
    assert _sd_from_log_variance(h2) == pytest.approx(0.5)

    # Explicit negative check: the classic factor-of-2 bug spec §1.5 warns
    # about is using exp(h) directly as the sd (that's actually the
    # variance). Assert the buggy formula gives a *different*, wrong answer
    # for the same h -- this is exactly the bug class this test exists to
    # catch once it's rewired against real S2 code.
    buggy_sd = math.exp(h)  # wrong: this is exp(h), not exp(h/2)
    correct_sd = _sd_from_log_variance(h)
    assert buggy_sd == pytest.approx(4.0)
    assert correct_sd == pytest.approx(2.0)
    assert buggy_sd != pytest.approx(correct_sd)


def test_production_sv_helpers_treat_h_as_log_variance() -> None:
    """S3 rewiring: the same convention checked against the real
    `macrotoolkit.smoother` SV helpers (mirrors of
    stan/functions/sv_rw_noncentered.stan; the Stan side is held to the
    Python side by the G1 harness's loglik_sv comparison).

    - `sv_diag_variance_path` puts exp(h) -- the VARIANCE -- on the
      measurement covariance diagonal, so at h = 2*ln(2) the R entry is
      4.0 and the implied sd is sqrt(4.0) = 2.0 = exp(h/2), NOT
      exp(h) = 4.0 (the factor-of-2 bug this convention exists to stop).
    - `sv_rw_noncentered` is the plain non-centered random walk
      h_t = h_0 + sigma_h * cumsum(nu), with h_0 NOT itself an
      observation-period value (observation t uses h[t], t >= 1).
    """
    import numpy as np

    from macrotoolkit.smoother import sv_diag_variance_path, sv_rw_noncentered

    h = 2.0 * math.log(2.0)
    R = sv_diag_variance_path(np.array([h]), np.array([0.0]))
    assert R.shape == (1, 2, 2)
    assert R[0, 0, 0] == pytest.approx(4.0)  # variance = exp(h)
    assert math.sqrt(R[0, 0, 0]) == pytest.approx(_sd_from_log_variance(h))  # sd = exp(h/2)
    assert math.sqrt(R[0, 0, 0]) != pytest.approx(math.exp(h))  # NOT exp(h)
    assert R[0, 1, 1] == pytest.approx(1.0)  # exp(0) = 1
    assert R[0, 0, 1] == 0.0 and R[0, 1, 0] == 0.0  # diagonal by construction

    path = sv_rw_noncentered(1.5, 0.5, np.array([1.0, -2.0, 3.0]))
    np.testing.assert_allclose(path, [2.0, 1.0, 2.5])
    # sigma_h = 0: the path is flat at h_0 (the funnel's degenerate corner).
    np.testing.assert_allclose(
        sv_rw_noncentered(-0.7, 0.0, np.array([1.0, 1.0])), [-0.7, -0.7]
    )


# ---------------------------------------------------------------------------
# Convention 3: inflation = 400 * dlog(P)
# ---------------------------------------------------------------------------


def _annualized_inflation(p_t: float, p_prev: float) -> float:
    """Reference implementation of spec §1.1's convention: annualized q/q
    inflation from a price level `P` is `400 * (log(P_t) - log(P_{t-1}))`."""
    return 400.0 * math.log(p_t / p_prev)


def test_inflation_is_400_times_dlog_price_level() -> None:
    # A small, realistic q/q move: price index 100 -> 100.5 (a 0.5% q/q
    # rise) annualizes to approximately 1.995%.
    assert _annualized_inflation(100.5, 100.0) == pytest.approx(1.995, abs=1e-3)

    # An exact clean-number case: choose p_t/p_prev = exp(1/400) so the
    # annualized result is exactly 1.0.
    p_prev = 100.0
    p_t = p_prev * math.exp(1.0 / 400.0)
    assert _annualized_inflation(p_t, p_prev) == pytest.approx(1.0)

    # Negative checks: the two documented wrong scalings (spec-adjacent
    # bugs this convention guards against) must NOT match the correct
    # convention for the same clean-number case.
    quarterly_not_annualized = 100.0 * math.log(p_t / p_prev)  # 100*dlog(P), missing the 4x
    annualized_not_percent = 4.0 * math.log(p_t / p_prev)  # 4*dlog(P), missing the 100x
    assert quarterly_not_annualized != pytest.approx(1.0)
    assert annualized_not_percent != pytest.approx(1.0)
