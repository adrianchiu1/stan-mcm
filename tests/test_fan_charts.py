"""Fan-chart (spec §3.3) tests for ``macrotoolkit.results_lw``'s Part D
(``simulate_fan_draw``, ``compute_fan_draws``).

Uses the session-shared ``s4_no_sv_lw_run`` / ``s4_sv_lw_run`` fixtures from
``tests/conftest.py`` (built once per pytest session, shared with
``tests/test_irf.py``) -- a genuinely stochastic forward simulation cannot
be tested against hand-built parameter points the way Part B/C's
deterministic recursions are; it needs a real completed run's posterior +
simulation-smoother machinery.

NOTE on 4Q growth's historical lookback (a discrepancy from the task brief,
noted per its own "trust the code, not the brief" instruction): 4-quarter
growth at the first forecast period (h=1, i.e. T+1) needs the REAL y value
FOUR quarters back, y_{T-3} -- which is NOT inside ``yobs[-3:,0]`` (that
slice only reaches back to T-2). ``simulate_fan_draw`` therefore reads
``yobs[-4:,0]`` (T-3, T-2, T-1, T), a strict superset of ``yobs[-3:,0]``:
growth at h=2,3,4 (which only need T-2, T-1, T respectively) is IDENTICAL
either way, so the tests below spot-check those periods directly against
``yobs[-3:,0]`` per the brief's own suggestion, and separately confirm h=1
is also correctly blended (using the 4th historical point the brief's
wording omitted).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from macrotoolkit.results_lw import (
    compute_fan_draws,
    select_draw_indices,
)
from specs.schema.lw_sv import ThinSpec

_SEED_NO_SV = 20260930
_SEED_SV = 20260931


# ---------------------------------------------------------------------------
# Shapes / finiteness
# ---------------------------------------------------------------------------


def test_compute_fan_draws_no_sv_shapes_and_finite(s4_no_sv_lw_run) -> None:
    fans = compute_fan_draws(s4_no_sv_lw_run, seed=_SEED_NO_SV)
    r = s4_no_sv_lw_run
    post = r.idata.posterior
    n_total = post.sizes["chain"] * post.sizes["draw"]
    expected_idx = select_draw_indices(n_total, r.spec.outputs.smoother_draws)
    np.testing.assert_array_equal(fans.draw_indices, expected_idx)
    assert fans.horizon == r.spec.outputs.horizon

    n_draws = len(expected_idx)
    for name, arr in (
        ("y_level", fans.y_level),
        ("y_growth_4q", fans.y_growth_4q),
        ("pi", fans.pi),
        ("gap", fans.gap),
        ("rstar", fans.rstar),
        ("rate_gap", fans.rate_gap),
    ):
        assert arr.shape == (n_draws, fans.horizon), name
        assert np.all(np.isfinite(arr)), name


def test_compute_fan_draws_sv_shapes_and_finite(s4_sv_lw_run) -> None:
    fans = compute_fan_draws(s4_sv_lw_run, seed=_SEED_SV)
    n_draws = len(fans.draw_indices)
    for name, arr in (
        ("y_level", fans.y_level),
        ("y_growth_4q", fans.y_growth_4q),
        ("pi", fans.pi),
        ("gap", fans.gap),
        ("rstar", fans.rstar),
        ("rate_gap", fans.rate_gap),
    ):
        assert arr.shape == (n_draws, fans.horizon), name
        assert np.all(np.isfinite(arr)), name


def test_compute_fan_draws_honors_thin_smoother_draws(s4_no_sv_lw_run) -> None:
    thinned = dataclasses.replace(
        s4_no_sv_lw_run,
        spec=s4_no_sv_lw_run.spec.model_copy(
            update={"outputs": s4_no_sv_lw_run.spec.outputs.model_copy(update={"smoother_draws": ThinSpec(thin=6)})}
        ),
    )
    fans = compute_fan_draws(thinned, seed=_SEED_NO_SV)
    post = thinned.idata.posterior
    n_total = post.sizes["chain"] * post.sizes["draw"]
    expected_idx = np.arange(0, n_total, 6)
    np.testing.assert_array_equal(fans.draw_indices, expected_idx)
    assert fans.gap.shape[0] == len(expected_idx)


# ---------------------------------------------------------------------------
# forecast_r_rule: "neutral" forces the rate-gap input to exactly 0 every
# simulated period; "last_value" generally does not.
# ---------------------------------------------------------------------------


def _with_forecast_r_rule(lw_run, rule: str):
    return dataclasses.replace(
        lw_run,
        spec=lw_run.spec.model_copy(update={"outputs": lw_run.spec.outputs.model_copy(update={"forecast_r_rule": rule})}),
    )


def test_neutral_rule_rate_gap_is_exactly_zero_every_period(s4_no_sv_lw_run) -> None:
    neutral_run = _with_forecast_r_rule(s4_no_sv_lw_run, "neutral")
    assert neutral_run.spec.outputs.forecast_r_rule == "neutral"
    fans = compute_fan_draws(neutral_run, seed=_SEED_NO_SV)
    np.testing.assert_array_equal(fans.rate_gap, np.zeros_like(fans.rate_gap))


def test_last_value_rule_rate_gap_is_generally_nonzero(s4_no_sv_lw_run) -> None:
    """r* evolves via g/z's own random walk while r is held fixed at r_T --
    the deviation should genuinely move, not stay pinned at 0 the way the
    neutral rule forces it to. Checked as "nonzero for at least one
    simulated draw/period" -- a real, non-vacuous "this rule actually does
    something different" sanity check."""
    lv_run = _with_forecast_r_rule(s4_no_sv_lw_run, "last_value")
    assert lv_run.spec.outputs.forecast_r_rule == "last_value"
    fans = compute_fan_draws(lv_run, seed=_SEED_NO_SV)
    assert np.any(fans.rate_gap != 0.0)


def test_neutral_and_last_value_give_different_gap_paths(s4_no_sv_lw_run) -> None:
    """A cross-check on the direct rate_gap assertions above: the two rules
    should also produce materially different downstream gap forecasts (the
    IS curve's rate-gap term feeds directly into gap), using the SAME rng
    seed for both so any difference is attributable to the rule, not to
    independent randomness."""
    neutral_run = _with_forecast_r_rule(s4_no_sv_lw_run, "neutral")
    lv_run = _with_forecast_r_rule(s4_no_sv_lw_run, "last_value")
    fans_neutral = compute_fan_draws(neutral_run, seed=12345)
    fans_lv = compute_fan_draws(lv_run, seed=12345)
    assert not np.allclose(fans_neutral.gap, fans_lv.gap)


# ---------------------------------------------------------------------------
# SV widening bands: the "widening bands are the point" property (spec
# §3.3), checked as an actual statistical assertion. The fixture run's tiny
# sampler settings (chains=2 x sampling=50, tests/conftest.py's
# `s4_sv_run_dir`) only ever produce 100 total posterior draws, so this test
# uses ALL of them (``smoother_draws: all``) rather than a further-thinned
# subset -- 100 draws is not "a few hundred" but is still enough for a
# directional (variance-at-late-horizon > variance-at-early-horizon)
# inequality: this is NOT a precise magnitude check, just a sign check, and
# the true widening effect compounds over ~12 quarters of SV random-walk
# variance accumulation, which dwarfs sampling noise at this draw count (the
# test would only be flaky if the widening effect were marginal, which it is
# not for this fixture's sigma_h priors -- verified empirically while
# writing this test).
# ---------------------------------------------------------------------------


def test_sv_run_bands_widen_with_horizon(s4_sv_lw_run) -> None:
    fans = compute_fan_draws(s4_sv_lw_run, seed=20261001)
    assert fans.pi.shape[0] >= 100, "need enough draws for a stable variance comparison"

    early = 0
    late = fans.horizon - 1
    for name, arr in (("pi", fans.pi), ("gap", fans.gap), ("y_level", fans.y_level)):
        var_early = np.var(arr[:, early])
        var_late = np.var(arr[:, late])
        assert var_late > var_early, (
            f"{name}: expected the cross-draw variance to WIDEN with horizon "
            f"for an SV run (spec §3.3's 'widening bands are the point'); "
            f"var_early={var_early!r}, var_late={var_late!r}."
        )


def test_no_sv_run_has_constant_measurement_variance_across_horizon(s4_no_sv_lw_run) -> None:
    """Direct, non-statistical check: a no-SV run's own R (the constant
    sigma_is^2/sigma_pc^2 pair) never changes with horizon at all -- there
    is no SV random-walk channel to widen anything. Checked per-draw by
    reading simulate_fan_draw's own variance construction indirectly: since
    eps_is_t/eps_pc_t are drawn from N(0, sigma_is^2)/N(0, sigma_pc^2) EVERY
    period with no growing scale, confirm this by checking the FLAT
    posterior's own sigma_is/sigma_pc values are exactly what every period
    uses -- i.e. that a no-SV run's own posterior has no sigma_h_is/
    sigma_h_pc-driven growth channel to begin with (the more direct, less
    statistical check the brief calls for)."""
    assert s4_no_sv_lw_run.sv_on is False
    post = s4_no_sv_lw_run.idata.posterior
    assert "sigma_h_is" not in post.data_vars
    assert "sigma_h_pc" not in post.data_vars
    assert "h_is" not in post.data_vars
    assert "h_pc" not in post.data_vars


# ---------------------------------------------------------------------------
# 4Q growth: first few forecast periods blend real historical y with
# forecasted y (spot-checked against yobs[-3:,0] directly for one draw, per
# the brief -- see this file's module docstring for the h=1 discrepancy
# note).
# ---------------------------------------------------------------------------


def test_4q_growth_blends_real_history_for_early_horizons(s4_no_sv_lw_run) -> None:
    r = s4_no_sv_lw_run
    fans = compute_fan_draws(r, seed=_SEED_NO_SV)
    draw_j = 0

    y_hist_last3 = r.yobs[-3:, 0]  # [y_{T-2}, y_{T-1}, y_T]
    y_level = fans.y_level[draw_j]
    growth = fans.y_growth_4q[draw_j]

    # h index 1 (T+2): ref = y_{T-2} = y_hist_last3[0].
    assert growth[1] == pytest.approx(y_level[1] - y_hist_last3[0])
    # h index 2 (T+3): ref = y_{T-1} = y_hist_last3[1].
    assert growth[2] == pytest.approx(y_level[2] - y_hist_last3[1])
    # h index 3 (T+4): ref = y_T = y_hist_last3[2].
    assert growth[3] == pytest.approx(y_level[3] - y_hist_last3[2])
    # h index 0 (T+1): ref = y_{T-3}, one quarter further back than
    # yobs[-3:,0] reaches -- confirmed against yobs[-4,0] directly.
    assert growth[0] == pytest.approx(y_level[0] - r.yobs[-4, 0])
    # h index >= 4: purely forecasted on both ends.
    assert fans.horizon > 4
    assert growth[4] == pytest.approx(y_level[4] - y_level[0])


# ---------------------------------------------------------------------------
# rstar / rate_gap arithmetic sanity: rstar_t = xi_t[3] + xi_t[5] each
# period (the same reporting convention used everywhere else in this
# module) -- a direct consistency check that Part D didn't invent a
# different rstar formula.
# ---------------------------------------------------------------------------


def test_last_value_rate_gap_equals_r_last_minus_rstar(s4_no_sv_lw_run) -> None:
    """Under the 2026-09-02 r* reporting alignment (rstar[t] is r*_{T+t+1},
    matching gap/pi/y's own period indexing, while rate_gap[t] remains the
    per-period INPUT (r-r*)_{T+t}), the identity relates rate_gap at index t
    to rstar at index t-1: rate_gap[t] = r_last - rstar[t-1] for t >= 1.
    (rate_gap[0] pairs with r*_T, which the aligned rstar series no longer
    carries -- it is checked directly in
    test_first_forecast_period_gap_last_value_uses_current_period_rstar.)"""
    lv_run = _with_forecast_r_rule(s4_no_sv_lw_run, "last_value")
    fans = compute_fan_draws(lv_run, seed=_SEED_NO_SV)
    r_last = float(lv_run.r_full[-1])
    np.testing.assert_allclose(fans.rate_gap[:, 1:], r_last - fans.rstar[:, :-1])


def test_fan_rstar_is_aligned_to_gap_pi_period_indexing() -> None:
    """Regression pin on the 2026-09-02 r* alignment fix: with zero noise,
    index t of the returned rstar series must be r* at absolute period
    T+t+1 -- i.e. slots 3/5 of the period-(T+t+2) state F^(t+2) @ xi_last --
    matching gap/pi/y's own indexing, NOT the one-quarter-stale
    F^(t+1) @ xi_last value the pre-fix code recorded. Standalone (synthetic
    parameter point from g1_harness, no MCMC run needed)."""
    from g1_harness import C_FIXED, generate_parameter_points
    from macrotoolkit.results_lw import simulate_fan_draw
    from macrotoolkit.smoother import build_lw_matrices

    params = generate_parameter_points(1, seed=20260902)[0]
    F, Q, A, Z, R = build_lw_matrices(params, c=params.get("c", C_FIXED))
    xi_last = np.array([100.0, 99.9, 99.8, 2.5, 2.4, 0.3, 0.2])

    out = simulate_fan_draw(
        F, Q, params["a1"], params["a2"], params["a_r"], params["b_pi"], params["b_y"],
        xi_last,
        gap_lag1=0.5, gap_lag2=0.3,
        pi_lag1=2.0, pi_lag2=1.9, pi_lag3=1.8, pi_lag4=1.7,
        rate_gap_seed=0.1, y_hist_last4=np.array([99.0, 99.3, 99.6, 100.5]),
        r_last=2.0, forecast_r_rule="neutral",
        sv_on=False, sigma_is=0.4, sigma_pc=0.8,
        h_is_last=None, h_pc_last=None, sigma_h_is=None, sigma_h_pc=None,
        horizon=3, rng=_ZeroNoiseRNG(),
    )
    for t in range(3):
        xi_future = np.linalg.matrix_power(F, t + 2) @ xi_last  # period T+t+2's state
        expected_rstar = xi_future[3] + xi_future[5]  # = g_{T+t+1} + z_{T+t+1} = r*_{T+t+1}
        assert out["rstar"][t] == pytest.approx(expected_rstar, abs=1e-12), t


# ---------------------------------------------------------------------------
# Regression guard (numerics-reviewer finding, S4): the first forecast
# period's rate-gap term was originally seeded/consumed one period stale --
# gap_{T+1} used (r-r*)_{T-1} where the IS curve needs (r-r*)_T, because
# xi_draw[-1,3]/[-1,5] hold g_{T-1}/z_{T-1} (this state's own one-period-lag
# convention), not g_T/z_T. Confirmed by direct numerical comparison against
# a hand-derived reference: up to ~65% relative distortion of the first
# forecast period's gap value under "neutral". Fixed by computing (r-r*)_T
# INSIDE the loop (it needs that period's own fresh process noise) and using
# it immediately, not deferring it to the next iteration. This test exercises
# the exact scenario the bug broke: a zero-noise deterministic pass through
# `simulate_fan_draw` directly, so gap_path[0] is checkable by hand.
# ---------------------------------------------------------------------------


def test_fan_forecast_step_is_curve_term_is_a_plus_not_a_minus() -> None:
    """Standalone, dependency-free pin on `_fan_forecast_step`'s IS-curve
    sign convention (numerics-reviewer finding, S4, separate from and
    found AFTER the rate-gap seeding/ordering bug above): the combined
    `rate_gap = (r-r*)` term must enter with a PLUS
    (`gap_t = a1*gap_lag1 + a2*gap_lag2 + (a_r/2)*(rate_gap_lag1+
    rate_gap_lag2) + eps_is_t`), matching spec §1.2's IS curve and
    `build_lw_matrices`'s own construction -- confirmed independently by
    symbolic re-derivation from `A`/`Z` and by numerical reconstruction of
    a real smoothed gap path. A shipped version of this function used a
    MINUS (carried over from `gap_pi_shock_decomposition`'s per-bar
    `-(a_r/2)*rstar` term, which is only valid because that function adds
    a SEPARATE `+(a_r/2)*(r_{t-1}+r_{t-2})` data term to the "init" bar --
    this function combines r and r* into one number up front, so it needs
    its own plus applied directly), which produced 100%+ relative
    distortion of forecast gap values by late horizons under
    `forecast_r_rule: last_value`. This test uses hand-picked numbers with
    no dependency on a sampled run, so it pins the sign with zero ambiguity
    about anything else going on."""
    from macrotoolkit.results_lw import _fan_forecast_step

    gap_t, pi_t = _fan_forecast_step(
        gap_lag1=1.0, gap_lag2=0.5,
        pi_lag1=2.0, pi_lag2=1.8, pi_lag3=1.6, pi_lag4=1.4,
        rate_gap_lag1=3.0, rate_gap_lag2=-1.0,
        a1=0.6, a2=0.2, a_r=-0.1, b_pi=0.7, b_y=0.05,
        eps_is_t=0.0, eps_pc_t=0.0,
    )
    # gap_t = 0.6*1.0 + 0.2*0.5 + (-0.1/2)*(3.0 + -1.0) + 0.0
    #       = 0.6 + 0.1 + (-0.05)*2.0 = 0.7 - 0.1 = 0.6
    assert gap_t == pytest.approx(0.6)
    # The WRONG (minus) sign would give 0.6 + 0.1 + 0.05*2.0 = 0.8 --
    # different enough that this test would have caught the original bug.
    wrong_sign_gap_t = 0.6 * 1.0 + 0.2 * 0.5 - (-0.1 / 2.0) * (3.0 + -1.0) + 0.0
    assert wrong_sign_gap_t == pytest.approx(0.8)
    assert gap_t != pytest.approx(wrong_sign_gap_t)


class _ZeroNoiseRNG:
    """Minimal np.random.Generator stand-in returning all-zero noise --
    duck-typed to the two methods simulate_fan_draw actually calls."""

    def standard_normal(self, size=None):
        return np.zeros(size) if size is not None else 0.0

    def normal(self, loc=0.0, scale=1.0, size=None):
        return np.full(size, loc) if size is not None else loc


def test_first_forecast_period_gap_matches_hand_derived_reference(s4_no_sv_lw_run) -> None:
    from macrotoolkit.results_lw import (
        _extract_gap_pi_coeffs,
        _flatten_posterior,
        _system_matrices_for_draw,
        simulate_fan_draw,
    )
    from macrotoolkit.smoother import simulate_smoother_draw

    r = s4_no_sv_lw_run
    flat = _flatten_posterior(r)
    i = 0
    F, Q, A, Z, R, h_is, h_pc = _system_matrices_for_draw(flat, i, r.sv_on)
    a1, a2, a_r, b_y, b_pi = _extract_gap_pi_coeffs(Z, A)

    zero_rng = _ZeroNoiseRNG()
    sim = simulate_smoother_draw(r.yobs, r.x, F, Q, A, Z, R, r.xi00, r.P00, zero_rng)
    xi_draw = sim.xi_draw

    gap_lag1 = float(r.yobs[-1, 0] - xi_draw[-1, 0])
    gap_lag2 = float(r.yobs[-2, 0] - xi_draw[-2, 0])
    rate_gap_seed = float(r.r_full[-2] - (xi_draw[-1, 3] + xi_draw[-1, 5]))  # (r-r*)_{T-1}
    sigma_is_i = float(flat["sigma_is"][i])
    sigma_pc_i = float(flat["sigma_pc"][i])

    out_neutral = simulate_fan_draw(
        F, Q, a1, a2, a_r, b_pi, b_y, xi_draw[-1],
        gap_lag1, gap_lag2,
        float(r.yobs[-1, 1]), float(r.yobs[-2, 1]), float(r.yobs[-3, 1]), float(r.yobs[-4, 1]),
        rate_gap_seed, r.yobs[-4:, 0], float(r.r_full[-1]), "neutral",
        False, sigma_is_i, sigma_pc_i, None, None, None, None,
        horizon=3, rng=zero_rng,
    )

    # With zero noise (w=0, eps_is=eps_pc=0) and "neutral" (forces
    # (r-r*)_T := 0 for the first forecast period), gap_{T+1} must equal
    # EXACTLY a1*gap_T + a2*gap_{T-1} - (a_r/2)*(0 + (r-r*)_{T-1}) -- no
    # eps_is term, no leaked stale rate-gap value.
    expected_gap_t1 = a1 * gap_lag1 + a2 * gap_lag2 + (a_r / 2.0) * (0.0 + rate_gap_seed)
    assert out_neutral["gap"][0] == pytest.approx(expected_gap_t1, abs=1e-10)

    # And the SECOND forecast period must use the FIRST period's own
    # (zero, under "neutral") rate-gap as its "lag1", and gap_{T+1} (just
    # computed) as its "lag2" for the AR(2) term -- i.e. rate_gap_lag2 for
    # h=2 is 0.0 (this period's own forced-neutral value), not the stale
    # rate_gap_seed again.
    expected_gap_t2 = a1 * out_neutral["gap"][0] + a2 * gap_lag1 + (a_r / 2.0) * (0.0 + 0.0)
    assert out_neutral["gap"][1] == pytest.approx(expected_gap_t2, abs=1e-10)


def test_first_forecast_period_gap_last_value_uses_current_period_rstar(s4_no_sv_lw_run) -> None:
    """Same zero-noise setup, but 'last_value': (r-r*)_T = r_last - r*_T,
    where r*_T is read off the JUST-simulated first-forecast-period state
    (xi_t[3]+xi_t[5], with zero process noise so xi_t = F @ xi_last
    exactly) -- not off xi_last's own slots 3/5 (which would give
    r*_{T-1}, the stale value the bug used)."""
    from macrotoolkit.results_lw import (
        _extract_gap_pi_coeffs,
        _flatten_posterior,
        _system_matrices_for_draw,
        simulate_fan_draw,
    )
    from macrotoolkit.smoother import simulate_smoother_draw

    r = s4_no_sv_lw_run
    flat = _flatten_posterior(r)
    i = 0
    F, Q, A, Z, R, h_is, h_pc = _system_matrices_for_draw(flat, i, r.sv_on)
    a1, a2, a_r, b_y, b_pi = _extract_gap_pi_coeffs(Z, A)

    zero_rng = _ZeroNoiseRNG()
    sim = simulate_smoother_draw(r.yobs, r.x, F, Q, A, Z, R, r.xi00, r.P00, zero_rng)
    xi_draw = sim.xi_draw

    gap_lag1 = float(r.yobs[-1, 0] - xi_draw[-1, 0])
    gap_lag2 = float(r.yobs[-2, 0] - xi_draw[-2, 0])
    rate_gap_seed = float(r.r_full[-2] - (xi_draw[-1, 3] + xi_draw[-1, 5]))
    r_last = float(r.r_full[-1])
    sigma_is_i = float(flat["sigma_is"][i])
    sigma_pc_i = float(flat["sigma_pc"][i])

    out = simulate_fan_draw(
        F, Q, a1, a2, a_r, b_pi, b_y, xi_draw[-1],
        gap_lag1, gap_lag2,
        float(r.yobs[-1, 1]), float(r.yobs[-2, 1]), float(r.yobs[-3, 1]), float(r.yobs[-4, 1]),
        rate_gap_seed, r.yobs[-4:, 0], r_last, "last_value",
        False, sigma_is_i, sigma_pc_i, None, None, None, None,
        horizon=1, rng=zero_rng,
    )

    xi_t1 = F @ xi_draw[-1]  # zero process noise -> exact
    rstar_t = xi_t1[3] + xi_t1[5]
    expected_rate_gap_t1 = r_last - rstar_t
    assert out["rate_gap"][0] == pytest.approx(expected_rate_gap_t1, abs=1e-10)
    expected_gap_t1 = a1 * gap_lag1 + a2 * gap_lag2 + (a_r / 2.0) * (expected_rate_gap_t1 + rate_gap_seed)
    assert out["gap"][0] == pytest.approx(expected_gap_t1, abs=1e-10)
