"""S4 gate: Durbin-Koopman simulation-smoother validation
(lw-sv-spec.md §2.4, plans/S4-plan.md's resolved "open question 1").

There is no Stan-side mirror to compare against here (§2.4's simulation
smoother is Python-only), so this file substitutes two checks in place of a
G1-style Stan comparison:

1. **Monte Carlo mean/variance convergence** -- draw many DK samples at a
   fixed parameter point and assert the empirical mean/variance of
   ``xi_draw`` across draws converges to ``kalman_smoother``'s
   ``xi_smooth``/``diag(P_smooth)``, within a tolerance derived from the
   analytic Monte Carlo standard error at that draw count (a documented
   safety margin keeps this non-flaky without hiding a real bug).
2. **Deterministic zero-plus-noise identity check** -- with the plus path's
   noise forced to exactly zero, the DK combination step must reduce
   ``xi_draw`` to EXACTLY ``xi_smooth`` (near machine precision), a
   real code-level check of the combination step (not a re-derivation of
   the same formula), via the ``zero_noise=True`` seam on
   ``simulate_smoother_draw``.

A third block validates the algebraic structural-shock recovery (spec §2.4
point 2): reconstructing state transitions from the recovered shocks
reproduces ``xi_draw`` exactly by construction (a regression guard against a
sign/index slip), and the deterministic-lag-copy rows of the recovered
process noise are ~0 for t = 1..T-1 (NOT at t = 0 -- see
``macrotoolkit.smoother._recover_structural_shocks``'s boundary note, which
this file also pins as its own explicit test).
"""
from __future__ import annotations

import numpy as np

from g1_harness import (
    C_FIXED,
    SYNTHETIC_DATA,
    generate_parameter_points,
)
from macrotoolkit.smoother import (
    build_lw_matrices,
    build_lw_regressors,
    default_initial_state,
    kalman_smoother,
    simulate_smoother_draw,
)

# ---------------------------------------------------------------------------
# Fixed parameter point + dataset shared by every test in this file
# ---------------------------------------------------------------------------

#: A single, fixed, realistic no-SV parameter point drawn from the same
#: prior generator G1 uses (specs.schema.lw_sv's priors table) -- not
#: hand-invented, per the task brief's "do not invent model equations,
#: prior values" instruction. Distinct seed from g1_harness's own
#: PARAM_SEED so this file's draw is independently reproducible.
_PARAM_SEED = 20260902
_PARAMS = generate_parameter_points(1, seed=_PARAM_SEED)[0]

#: g1_harness's fixed synthetic dataset (T=50 raw quarters -> T=46
#: estimation-sample rows after the 4 pre-sample lags) -- plausible
#: magnitudes only, not an economic fixture (see g1_harness.py docstring).
_YOBS, _X = build_lw_regressors(SYNTHETIC_DATA.y, SYNTHETIC_DATA.pi, SYNTHETIC_DATA.r)
_F, _Q, _A, _Z, _R = build_lw_matrices(_PARAMS, c=_PARAMS.get("c", C_FIXED))
_XI00, _P00 = default_initial_state(float(SYNTHETIC_DATA.y[4]))

#: Monte Carlo draw count and safety margins (documented rationale below).
N_DRAWS = 5000
#: Mean convergence: the sampling distribution of a sample mean of n iid
#: draws is (asymptotically/exactly, for a Gaussian-derived quantity)
#: N(true_mean, se^2) with se = sqrt(true_var / n) -- a 5x margin on that se
#: keeps a per-(t, state) two-sided z-test's false-positive rate on the
#: order of 1e-6 per cell (P(|Z|>5) ~ 6e-7), safely below the ~320-cell
#: (46 periods x 7 states) family this test checks, without hiding a real
#: mean bug (which would show up at 10-100x se, not 5x).
MEAN_SE_MARGIN = 5.0
#: Variance convergence: for n iid Gaussian draws, Var(sample variance) =
#: 2*sigma^4/(n-1) exactly (a standard result), so
#: se(sample_var) = true_var * sqrt(2/(n-1)). The sampling distribution of a
#: sample variance is a scaled chi-squared, which is right-skewed (heavier
#: right tail than a normal at the same nominal "sigma" multiple) -- a
#: slightly larger margin than the mean check compensates for that skew
#: while still catching a real variance bug (e.g. a missing/duplicated
#: noise term, which would show up at a very different scale, not 1.2x).
VAR_SE_MARGIN = 6.0


def _real_smooth() -> dict:
    return kalman_smoother(_YOBS, _X, _F, _Q, _A, _Z, _R, _XI00, _P00)


# ---------------------------------------------------------------------------
# 1. Monte Carlo mean/variance convergence to the RTS smoother
# ---------------------------------------------------------------------------


def test_dk_draws_mean_and_variance_converge_to_rts_smoother() -> None:
    """`N_DRAWS` DK draws at a fixed parameter point: the empirical mean and
    per-period variance of `xi_draw` must converge to `kalman_smoother`'s
    `xi_smooth`/`diag(P_smooth)` within the analytic Monte Carlo standard
    error (times the documented margins above). Asserted per-(period,
    state-slot) cell, not just a summary norm, so a failure names exactly
    which slot and period are off.
    """
    real = _real_smooth()
    xi_smooth = real["xi_smooth"]
    P_smooth = real["P_smooth"]
    T, n = xi_smooth.shape

    rng = np.random.default_rng(20260903)
    draws = np.empty((N_DRAWS, T, n))
    for i in range(N_DRAWS):
        d = simulate_smoother_draw(
            _YOBS, _X, _F, _Q, _A, _Z, _R, _XI00, _P00, rng, xi_smooth=xi_smooth
        )
        draws[i] = d.xi_draw

    empirical_mean = draws.mean(axis=0)
    empirical_var = draws.var(axis=0, ddof=1)
    true_var = np.diagonal(P_smooth, axis1=1, axis2=2)  # (T, n)

    se_mean = np.sqrt(true_var / N_DRAWS)
    se_var = true_var * np.sqrt(2.0 / (N_DRAWS - 1))

    mean_diff = np.abs(empirical_mean - xi_smooth)
    mean_z = mean_diff / se_mean
    worst = np.unravel_index(np.argmax(mean_z), mean_z.shape)
    assert np.all(mean_z <= MEAN_SE_MARGIN), (
        f"DK draw empirical mean fails to converge to RTS xi_smooth: worst "
        f"cell (period={worst[0]}, state_slot={worst[1]}): "
        f"empirical_mean={empirical_mean[worst]!r}, "
        f"xi_smooth={xi_smooth[worst]!r}, diff={mean_diff[worst]:.3e}, "
        f"se={se_mean[worst]:.3e}, z={mean_z[worst]:.2f} > margin={MEAN_SE_MARGIN}."
    )

    var_diff = np.abs(empirical_var - true_var)
    var_z = var_diff / se_var
    worst_v = np.unravel_index(np.argmax(var_z), var_z.shape)
    assert np.all(var_z <= VAR_SE_MARGIN), (
        f"DK draw empirical variance fails to converge to RTS diag(P_smooth): "
        f"worst cell (period={worst_v[0]}, state_slot={worst_v[1]}): "
        f"empirical_var={empirical_var[worst_v]!r}, "
        f"true_var={true_var[worst_v]!r}, diff={var_diff[worst_v]:.3e}, "
        f"se={se_var[worst_v]:.3e}, z={var_z[worst_v]:.2f} > margin={VAR_SE_MARGIN}."
    )


# ---------------------------------------------------------------------------
# 2. Deterministic zero-plus-noise identity check ("G1-style mirror")
# ---------------------------------------------------------------------------


def test_zero_plus_noise_dk_draw_equals_rts_smoother_exactly() -> None:
    """With the plus-path noise forced to exactly zero (`zero_noise=True`:
    `xi+_0 = xi00` exactly, `xi+_t = F @ xi+_{t-1}` deterministically, no
    `w+`/`e+` at any t), the DK identity `xi_draw = xi_smooth - xi+_smooth +
    xi+` must reduce to EXACTLY `xi_smooth`, near machine precision -- both
    smooths of a zero-noise "plus" system coincide with the deterministic
    path (a real mathematical identity, not a tautology: `kalman_smoother`
    still uses the real, nonzero Q/R in its own recursion; the plus path's
    OWN noise is what's forced to zero), so this exercises the real
    combination code path (step 4 of plans/S4-plan.md's algorithm) rather
    than re-deriving the same formula in the test.
    """
    real = _real_smooth()
    xi_smooth = real["xi_smooth"]

    rng = np.random.default_rng(20260904)  # unused when zero_noise=True, but required
    draw = simulate_smoother_draw(
        _YOBS, _X, _F, _Q, _A, _Z, _R, _XI00, _P00, rng,
        xi_smooth=xi_smooth, zero_noise=True,
    )

    diff = np.abs(draw.xi_draw - xi_smooth)
    assert diff.max() < 1e-9, (
        f"zero-plus-noise DK draw does not reduce to xi_smooth: max abs "
        f"diff={diff.max():.3e} at index {np.unravel_index(np.argmax(diff), diff.shape)}."
    )


def test_zero_plus_noise_is_actually_a_nontrivial_check() -> None:
    """Guard against the identity test above passing vacuously: a
    stochastic (`zero_noise=False`) draw at the same seed/system must
    actually DIFFER from `xi_smooth` (otherwise the RNG or the plus-path
    noise injection could be silently broken and both tests above would
    pass for the wrong reason)."""
    real = _real_smooth()
    xi_smooth = real["xi_smooth"]
    rng = np.random.default_rng(20260905)
    draw = simulate_smoother_draw(
        _YOBS, _X, _F, _Q, _A, _Z, _R, _XI00, _P00, rng, xi_smooth=xi_smooth
    )
    diff = np.abs(draw.xi_draw - xi_smooth).max()
    assert diff > 1e-3, (
        f"a stochastic DK draw was suspiciously close to xi_smooth "
        f"(max abs diff={diff:.3e}) -- the plus-path noise injection may be "
        f"broken (e.g. zero_noise accidentally always True)."
    )


# ---------------------------------------------------------------------------
# 3. Structural-shock recovery: reconstruction identity + lag-copy sanity
# ---------------------------------------------------------------------------


def test_recovered_shocks_reconstruct_state_transitions_exactly() -> None:
    """Plugging the recovered shocks back through F/Q's structure must
    reproduce `xi_draw` exactly at rows 0, 3, 5 for EVERY t (the three rows
    the 5-shock parametrization actually covers: y*, g, z), and at rows 1,
    2, 4, 6 (the deterministic lag-copy states) for t = 1..T-1 -- a
    regression guard against a sign/index slip in
    `_recover_structural_shocks`, per the task brief.

    Rows 1, 2, 4, 6 are DELIBERATELY excluded from the reconstruction check
    at t=0: the 5-shock parametrization only ever reconstructs those rows as
    exact zero (`w[:, [1,2,4,6]] = 0`), which is correct for t=1..T-1 but,
    per `_recover_structural_shocks`'s documented boundary note, w_0's rows
    1,2,4,6 are genuinely nonzero (xi00 is a raw prior mean, not a smoothed
    estimate) -- reconstructing xi_draw[0] from the 5 named shocks ALONE
    necessarily misses that t=0-only residual. `test_lag_copy_rows_are_not_
    zero_at_t0_boundary` pins that residual explicitly.
    """
    rng = np.random.default_rng(20260906)
    real = _real_smooth()
    draw = simulate_smoother_draw(
        _YOBS, _X, _F, _Q, _A, _Z, _R, _XI00, _P00, rng, xi_smooth=real["xi_smooth"]
    )

    T = draw.xi_draw.shape[0]
    w = np.zeros((T, 7))
    w[:, 0] = draw.eps_ystar + draw.eps_g / 4.0
    w[:, 3] = draw.eps_g
    w[:, 5] = draw.eps_z

    xi_prev = np.vstack([_XI00[None, :], draw.xi_draw[:-1]])
    reconstructed = xi_prev @ _F.T + w

    diff_core = np.abs(reconstructed[:, [0, 3, 5]] - draw.xi_draw[:, [0, 3, 5]])
    assert diff_core.max() < 1e-9, (
        f"reconstructing xi_draw's y*/g/z rows from recovered shocks failed: "
        f"max abs diff={diff_core.max():.3e} at "
        f"{np.unravel_index(np.argmax(diff_core), diff_core.shape)}."
    )

    diff_lag = np.abs(reconstructed[1:, [1, 2, 4, 6]] - draw.xi_draw[1:, [1, 2, 4, 6]])
    assert diff_lag.max() < 1e-9, (
        f"reconstructing xi_draw's lag-copy rows (1,2,4,6) for t=1..T-1 "
        f"failed: max abs diff={diff_lag.max():.3e} at "
        f"{np.unravel_index(np.argmax(diff_lag), diff_lag.shape)}."
    )
    assert np.abs(w[1:, [1, 2, 4, 6]]).max() < 1e-9, (
        f"recovered process noise in the deterministic lag-copy rows "
        f"(1, 2, 4, 6) is not ~0 for t=1..T-1: max abs "
        f"value={np.abs(w[1:, [1, 2, 4, 6]]).max():.3e}."
    )


def test_lag_copy_rows_are_not_zero_at_t0_boundary() -> None:
    """Pins the documented boundary behavior at t=0 (see
    `_recover_structural_shocks`'s docstring): unlike t=1..T-1, w_0's rows
    1, 2, 4, 6 are NOT forced to ~0, because xi00 is the raw PRIOR mean of
    the pre-sample state (not itself a smoothed/updated estimate) -- so the
    full-sample smoothed belief about the lagged states at t=0 can and does
    differ from that prior mean. This is a positive assertion (a nonzero
    residual is expected) precisely so a future change that "fixes" this by
    silently zeroing it out (masking a real discrepancy) gets caught."""
    rng = np.random.default_rng(20260907)
    real = _real_smooth()
    draw = simulate_smoother_draw(
        _YOBS, _X, _F, _Q, _A, _Z, _R, _XI00, _P00, rng, xi_smooth=real["xi_smooth"]
    )
    w0 = draw.xi_draw[0] - _F @ _XI00
    assert np.abs(w0[[1, 2, 4, 6]]).max() > 1e-3, (
        "expected a nonzero t=0 lag-copy residual (xi00 is a raw prior "
        "mean, not a smoothed estimate) but got one close to zero -- "
        f"w0[[1,2,4,6]]={w0[[1, 2, 4, 6]]!r}; if this is now genuinely ~0, "
        "the boundary-note docstring in _recover_structural_shocks needs "
        "revisiting, not this test's tolerance."
    )


def test_recovered_measurement_shocks_match_residual_by_construction() -> None:
    """`e_t` computed from the recovered shocks (`eps_is`, `eps_pc`) matches
    `yobs[t] - A'x[t] - Z@xi_draw[t]` exactly -- a regression guard against a
    sign/index slip (this should be true by construction, per the task
    brief)."""
    rng = np.random.default_rng(20260908)
    real = _real_smooth()
    draw = simulate_smoother_draw(
        _YOBS, _X, _F, _Q, _A, _Z, _R, _XI00, _P00, rng, xi_smooth=real["xi_smooth"]
    )
    e_expected = _YOBS - _X @ _A - draw.xi_draw @ _Z.T
    e_recovered = np.column_stack([draw.eps_is, draw.eps_pc])
    diff = np.abs(e_expected - e_recovered)
    assert diff.max() < 1e-12, (
        f"recovered measurement shocks (eps_is, eps_pc) do not match the "
        f"observation residual: max abs diff={diff.max():.3e}."
    )


# ---------------------------------------------------------------------------
# Basic API/shape sanity (not a numerical gate, but cheap and worth pinning)
# ---------------------------------------------------------------------------


def test_simulate_smoother_draw_computes_xi_smooth_internally_if_not_given() -> None:
    """`xi_smooth` is optional -- when omitted, `simulate_smoother_draw`
    must compute it internally via `kalman_smoother` and produce the same
    result as passing it explicitly (same rng state sequence, so the two
    calls must be bit-identical given fresh, identically-seeded rngs)."""
    real = _real_smooth()
    draw_explicit = simulate_smoother_draw(
        _YOBS, _X, _F, _Q, _A, _Z, _R, _XI00, _P00,
        np.random.default_rng(42), xi_smooth=real["xi_smooth"],
    )
    draw_implicit = simulate_smoother_draw(
        _YOBS, _X, _F, _Q, _A, _Z, _R, _XI00, _P00, np.random.default_rng(42),
    )
    np.testing.assert_array_equal(draw_explicit.xi_draw, draw_implicit.xi_draw)


def test_simulate_smoother_draw_shapes() -> None:
    real = _real_smooth()
    rng = np.random.default_rng(7)
    draw = simulate_smoother_draw(
        _YOBS, _X, _F, _Q, _A, _Z, _R, _XI00, _P00, rng, xi_smooth=real["xi_smooth"]
    )
    T = _YOBS.shape[0]
    assert draw.xi_draw.shape == (T, 7)
    for name in ("eps_ystar", "eps_g", "eps_z", "eps_is", "eps_pc"):
        arr = getattr(draw, name)
        assert arr.shape == (T,), f"{name} shape {arr.shape} != ({T},)"
        assert np.all(np.isfinite(arr)), f"{name} has non-finite values"


def test_simulate_smoother_draw_accepts_R_as_constant_or_time_varying_path() -> None:
    """`R` accepts the same constant-(m,m)-or-(T,m,m)-path forms as
    `kalman_loglik`/`kalman_smoother` (both routed through `_as_R_path`)."""
    from macrotoolkit.smoother import _as_R_path

    T = _YOBS.shape[0]
    R_path = _as_R_path(_R, T)  # tile the constant R to a (T, 2, 2) path
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    draw_const = simulate_smoother_draw(_YOBS, _X, _F, _Q, _A, _Z, _R, _XI00, _P00, rng1)
    draw_path = simulate_smoother_draw(_YOBS, _X, _F, _Q, _A, _Z, R_path, _XI00, _P00, rng2)
    np.testing.assert_array_equal(draw_const.xi_draw, draw_path.xi_draw)


# ---------------------------------------------------------------------------
# S6: generic structural-shock recovery from declared loadings
# ---------------------------------------------------------------------------


def test_generic_shock_recovery_is_bit_identical_to_the_lw_sv_recovery() -> None:
    """S6 WP2 generalized the lw_sv-specific shock recovery (slots 0/3/5,
    the /4) into ``recover_shocks`` driven by ``StateSpaceMeta.recovery_
    order``. For lw_sv the two must agree BIT FOR BIT (0.25*x == x/4 in
    IEEE double; /1.0 is exact) -- the behavior-preservation pin, and
    ``simulate_smoother_draw(meta=...)`` must fill both the legacy
    attributes and the named dicts identically."""
    from macrotoolkit.families.lw_sv import LW_STATE_META
    from macrotoolkit.smoother import _recover_structural_shocks, recover_shocks

    F, Q, A, Z, R = _F, _Q, _A, _Z, _R
    rng = np.random.default_rng(20260904)
    draw_legacy = simulate_smoother_draw(_YOBS, _X, F, Q, A, Z, R, _XI00, _P00, rng)
    rng = np.random.default_rng(20260904)
    draw_meta = simulate_smoother_draw(_YOBS, _X, F, Q, A, Z, R, _XI00, _P00, rng, meta=LW_STATE_META)
    np.testing.assert_array_equal(draw_legacy.xi_draw, draw_meta.xi_draw)
    for legacy_name, shock in (("eps_ystar", "ystar"), ("eps_g", "g"), ("eps_z", "z")):
        np.testing.assert_array_equal(getattr(draw_legacy, legacy_name), draw_meta.state_shocks[shock])
        np.testing.assert_array_equal(getattr(draw_meta, legacy_name), draw_meta.state_shocks[shock])
        np.testing.assert_array_equal(draw_legacy.state_shocks[shock], draw_meta.state_shocks[shock])
    for legacy_name, shock in (("eps_is", "is"), ("eps_pc", "pc")):
        np.testing.assert_array_equal(getattr(draw_legacy, legacy_name), draw_meta.meas_shocks[shock])

    # Direct comparison of the two recovery functions on the same path.
    xi = draw_legacy.xi_draw
    legacy = _recover_structural_shocks(xi, _XI00, _YOBS, _X, A, Z, F)
    state, meas = recover_shocks(xi, _XI00, _YOBS, _X, A, Z, F, LW_STATE_META)
    np.testing.assert_array_equal(legacy[0], state["ystar"])
    np.testing.assert_array_equal(legacy[1], state["g"])
    np.testing.assert_array_equal(legacy[2], state["z"])
    np.testing.assert_array_equal(legacy[3], meas["is"])
    np.testing.assert_array_equal(legacy[4], meas["pc"])


def test_recovery_order_is_triangular_for_lw_sv_and_rejects_non_triangular() -> None:
    import pytest

    from macrotoolkit.families.base import StateSpaceMeta
    from macrotoolkit.families.lw_sv import LW_STATE_META

    order = LW_STATE_META.recovery_order()
    names = [o[0] for o in order]
    # ystar must come AFTER g (its slot is shared with the g loading).
    assert names.index("ystar") > names.index("g")
    ystar_step = next(o for o in order if o[0] == "ystar")
    assert ystar_step[1] == ("ystar", 0) and ystar_step[2] == 1.0 and ystar_step[3] == (("g", 0.25),)

    bad = StateSpaceMeta(
        state_labels=(("a", 0),),
        state_shocks=("u", "v"),
        shock_loadings={"u": {("a", 0): 1.0}, "v": {("a", 0): 2.0}},
        measurement_shocks=("e",),
        obs_names=("y",),
    )
    with pytest.raises(ValueError, match="not triangular"):
        bad.recovery_order()
