"""Generic engine (S5-decisions item 1) pins: the feedback map matches
``build_lw_regressors`` column-for-column, and the engine's observation
arithmetic reproduces the measurement equation with the declared feedback.

The heavy behavioral coverage lives where it always did -- gate G6
(``tests/test_g6_hd_identity.py``), the IRF structural tests
(``tests/test_irf.py``), the fan-chart pins (``tests/test_fan_charts.py``)
-- all of which now exercise the engine through ``results_lw``'s public
functions. This file pins the engine-specific contracts those suites
assume.
"""
from __future__ import annotations

import numpy as np
import pytest

from g1_harness import SYNTHETIC_DATA
from macrotoolkit.engine import (
    ConstantExogRule,
    StateLinearExogRule,
    observable_recursion,
    propagate_state_shock,
)
from macrotoolkit.families.base import ExogLag, ObsLag, ObsLagMean
from macrotoolkit.families.lw_sv import LW_STATE_META
from macrotoolkit.smoother import build_lw_matrices, build_lw_regressors

_PARAMS = {
    "a1": 1.1,
    "a2": -0.3,
    "a_r": -0.08,
    "b_pi": 0.7,
    "b_y": 0.1,
    "sigma_ystar": 0.5,
    "sigma_g": 0.03,
    "sigma_z": 0.08,
    "sigma_is": 0.4,
    "sigma_pc": 0.8,
}


def test_feedback_map_matches_build_lw_regressors_column_for_column() -> None:
    """The feedback map IS build_lw_regressors' x-column layout, as data:
    rebuilding x from the declaration + the raw full-length series must
    reproduce build_lw_regressors' output bit-for-bit."""
    y = np.asarray(SYNTHETIC_DATA.y, dtype=np.float64)
    pi = np.asarray(SYNTHETIC_DATA.pi, dtype=np.float64)
    r = np.asarray(SYNTHETIC_DATA.r, dtype=np.float64)
    yobs, x = build_lw_regressors(y, pi, r)
    T = x.shape[0]
    full = {"y": y, "pi": pi, "r": r}

    assert len(LW_STATE_META.feedback_map) == x.shape[1]
    rebuilt = np.zeros_like(x)
    for j, term in enumerate(LW_STATE_META.feedback_map):
        for i in range(T):
            t_full = i + 4  # row i = full-series index i+4 (4 pre-sample lags)
            if isinstance(term, ObsLag):
                rebuilt[i, j] = full[term.name][t_full - term.lag]
            elif isinstance(term, ObsLagMean):
                total = 0.0
                for lag in term.lags:
                    total += full[term.name][t_full - lag]
                rebuilt[i, j] = total / len(term.lags)
            elif isinstance(term, ExogLag):
                rebuilt[i, j] = full[term.name][t_full - term.lag]
    np.testing.assert_array_equal(rebuilt, x)


def test_observable_recursion_is_the_measurement_equation_with_feedback() -> None:
    """Hand-checkable single-bar case: zero state path, one eps_is impulse,
    zero exog. Period 0: y = eps (nothing else nonzero); period 1:
    y = a1 * eps (own-lag feedback through the map), pi = b_y * eps
    (Phillips picks up y_{t-1}); rate-gap and pi channels into y stay
    exactly zero (zero rows/entries of A)."""
    F, Q, A, Z, R = build_lw_matrices(_PARAMS, c=1.0)
    T = 3
    state = np.zeros((T, LW_STATE_META.n_state))
    meas = np.zeros((T, 2))
    eps = 0.37
    meas[0, 0] = eps

    obs = observable_recursion(A, Z, LW_STATE_META, state, meas)
    assert obs[0, 0] == eps  # exactly: all other terms are exact zeros
    assert obs[0, 1] == 0.0
    assert obs[1, 0] == pytest.approx(_PARAMS["a1"] * eps, abs=1e-15)
    assert obs[1, 1] == pytest.approx(_PARAMS["b_y"] * eps, abs=1e-15)
    assert obs[2, 0] == pytest.approx(
        _PARAMS["a1"] * obs[1, 0] + _PARAMS["a2"] * eps, abs=1e-15
    )


def test_observable_recursion_rate_term_sign_is_spec_1_2_plus() -> None:
    """The IS-curve rate term must enter y with +(a_r/2)*(r_{t-1}+r_{t-2})
    from the DATA side and -(a_r/2)*(r*_{t-1}+r*_{t-2}) from the STATE side
    (spec §1.2's +(a_r/2)*[(r-r*)_{t-1}+(r-r*)_{t-2}] once combined) --
    the sign whose hand-application produced S4's second fan-chart bug.
    Here the state carries g_{t-1} = 2.0 (r*_{t-1} = 2.0) and the data
    carries r_{t-1} = 3.0, so period 0's y must move by
    (a_r/2)*(3.0 - 2.0) = a_r/2 < 0 net of the y* level terms."""
    F, Q, A, Z, R = build_lw_matrices(_PARAMS, c=1.0)
    T = 1
    state = np.zeros((T, LW_STATE_META.n_state))
    state[0, LW_STATE_META.slot("g", -1)] = 2.0
    exog = np.zeros((T, len(LW_STATE_META.feedback_map)))
    exog[0, 2] = 3.0  # the ExogLag("r", 1) column
    meas = np.zeros((T, 2))

    obs = observable_recursion(A, Z, LW_STATE_META, state, meas, exog_x=exog)
    a_r = _PARAMS["a_r"]
    assert obs[0, 0] == pytest.approx((a_r / 2.0) * (3.0 - 2.0), abs=1e-15)
    # The wrong (flipped) combination would give -(a_r/2)*(3-2) = +0.04.
    assert obs[0, 0] != pytest.approx(-(a_r / 2.0) * (3.0 - 2.0))


def test_observable_recursion_seed_validation() -> None:
    F, Q, A, Z, R = build_lw_matrices(_PARAMS, c=1.0)
    state = np.zeros((2, LW_STATE_META.n_state))
    meas = np.zeros((2, 2))
    with pytest.raises(ValueError, match="outside the feedback map's depth"):
        observable_recursion(A, Z, LW_STATE_META, state, meas, obs_seeds={"y": {3: 1.0}})


def test_propagate_state_shock_matches_manual_recursion() -> None:
    F, Q, A, Z, R = build_lw_matrices(_PARAMS, c=1.0)
    b = LW_STATE_META.injection_vector("g")
    eps = np.array([0.5, 0.0, -0.2])
    out = propagate_state_shock(F, b, eps)
    prev = np.zeros(7)
    for t in range(3):
        prev = F @ prev + b * eps[t]
        np.testing.assert_array_equal(out[t], prev)


def test_simulate_forward_is_family_agnostic_on_a_toy_meta() -> None:
    """The engine's genericity claim, checked away from lw_sv's shapes: a
    toy 2-state/1-observable family (level random walk observed with its
    own first lag as a regressor) run through simulate_forward end to end,
    zero-noise, against a closed-form hand recursion."""
    from macrotoolkit.engine import ConstantMeasurementNoise, simulate_forward
    from macrotoolkit.families.base import StateSpaceMeta

    meta = StateSpaceMeta(
        state_labels=(("lvl", 0), ("lvl", -1)),
        state_shocks=("lvl",),
        shock_loadings={"lvl": {("lvl", 0): 1.0}},
        measurement_shocks=("e",),
        obs_names=("v",),
        exog_names=("u",),
        feedback_map=(ObsLag("v", 1), ExogLag("u", 1)),
    )
    F = np.array([[1.0, 0.0], [1.0, 0.0]])  # lvl RW; (lvl,-1) lag copy
    Q = np.zeros((2, 2))
    A = np.array([[0.5], [0.25]])  # v_t = 0.5*v_{t-1} + 0.25*u_{t-1} + lvl_t
    Z = np.array([[1.0, 0.0]])

    class _ZeroRng:
        def standard_normal(self, size=None):
            return np.zeros(size) if size is not None else 0.0

        def normal(self, loc=0.0, scale=1.0, size=None):
            return loc

    out = simulate_forward(
        F, Q, A, Z, meta,
        xi_last=np.array([2.0, 1.9]),
        obs_seeds={"v": {1: 3.0}},
        exog_seeds={},
        exog_rules={"u": ConstantExogRule(4.0)},
        meas_noise=ConstantMeasurementNoise((0.7,)),
        horizon=3,
        rng=_ZeroRng(),
    )
    obs = out["obs"][:, 0]
    # lvl stays 2.0 (zero noise); v_t = 0.5*v_{t-1} + 0.25*4.0 + 2.0.
    v = 3.0
    for t in range(3):
        v = 0.5 * v + 0.25 * 4.0 + 2.0
        assert obs[t] == pytest.approx(v, abs=1e-15), t
    assert out["states"].shape == (4, 2)
    np.testing.assert_allclose(out["states"][:, 0], 2.0)
    np.testing.assert_array_equal(out["exog_resolved"]["u"], np.full(3, 4.0))


def test_exog_rules() -> None:
    rule_const = ConstantExogRule(2.5)
    assert rule_const.resolve(np.arange(7.0)) == 2.5
    rule_lin = StateLinearExogRule(LW_STATE_META, {("g", -1): 1.0, ("z", -1): 1.0})
    xi = np.zeros(7)
    xi[LW_STATE_META.slot("g", -1)] = 1.25
    xi[LW_STATE_META.slot("z", -1)] = -0.5
    assert rule_lin.resolve(xi) == pytest.approx(0.75)
    with pytest.raises(KeyError):
        StateLinearExogRule(LW_STATE_META, {("g", 0): 1.0})


def test_constant_state_noise_reproduces_the_pre_s6_state_draw() -> None:
    """S6: ``simulate_forward``'s default state noise is
    ``ConstantStateNoise(Q)``, whose draw is exactly the pre-S6
    ``_psd_sqrt(Q) @ rng.standard_normal(n)`` -- same factor, same RNG
    consumption -- so seeded fan streams are unchanged. And the SV state
    noise model builds innovations from the declared loadings with the
    log-variance random walk continued per shock (h is log-VARIANCE)."""
    from macrotoolkit.engine import ConstantStateNoise, RandomWalkLogVarianceStateNoise
    from macrotoolkit.smoother import _psd_sqrt

    p = _PARAMS
    F, Q, A, Z, R = build_lw_matrices(p, c=1.0)
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    noise = ConstantStateNoise(Q)
    sqrt_Q = _psd_sqrt(Q)
    for _ in range(5):
        np.testing.assert_array_equal(noise.step(rng_a), sqrt_Q @ rng_b.standard_normal(Q.shape[0]))

    sv = RandomWalkLogVarianceStateNoise(
        LW_STATE_META, {"z": 2.0 * np.log(0.3)}, {"z": 0.1}, constant_sds={"ystar": 0.5, "g": 0.05}
    )
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(11)
    h = 2.0 * np.log(0.3)
    w = sv.step(rng_a)
    expected = np.zeros(7)
    expected += LW_STATE_META.injection_vector("ystar") * rng_b.normal(0.0, 0.5)
    expected += LW_STATE_META.injection_vector("g") * rng_b.normal(0.0, 0.05)
    h = h + 0.1 * rng_b.standard_normal()
    expected += LW_STATE_META.injection_vector("z") * rng_b.normal(0.0, np.exp(h / 2.0))
    np.testing.assert_array_equal(w, expected)
    with pytest.raises(ValueError, match="neither an SV path nor a constant sd"):
        RandomWalkLogVarianceStateNoise(LW_STATE_META, {"z": 0.0}, {"z": 0.1})
