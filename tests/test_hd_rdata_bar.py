"""Pins on the "rdata" historical-decomposition bar (pre-S5 review decision,
2026-09-02, DECISIONS.md): the exogenous real-rate data injection
``+(a_r/2)(r_{t-1}+r_{t-2})`` is its OWN labeled bar in gap's B2
decomposition, no longer folded into "init" -- folding the entire cumulative
policy contribution into a line labeled "Initial condition" was judged
misleading. The split re-attributes between the two non-structural bars
only; every G6 sum identity is unchanged (still covered by
``tests/test_g6_hd_identity.py``).

Standalone synthetic setup (g1_harness parameter point + dataset, DK draws,
no MCMC) -- the same pattern as the G6 file's own cases.
"""
from __future__ import annotations

import numpy as np
import pytest

from g1_harness import C_FIXED, SYNTHETIC_DATA, generate_parameter_points
from macrotoolkit.results_lw import (
    GAP_BARS,
    PI_BARS,
    gap_pi_shock_decomposition,
    state_shock_decomposition,
)
from macrotoolkit.smoother import (
    build_lw_matrices,
    build_lw_regressors,
    default_initial_state,
    kalman_smoother,
    simulate_smoother_draw,
)

_Y_FULL = np.asarray(SYNTHETIC_DATA.y, dtype=np.float64)
_PI_FULL = np.asarray(SYNTHETIC_DATA.pi, dtype=np.float64)
_R_FULL = np.asarray(SYNTHETIC_DATA.r, dtype=np.float64)
_YOBS, _X = build_lw_regressors(_Y_FULL, _PI_FULL, _R_FULL)
_XI00, _P00 = default_initial_state(float(_Y_FULL[4]))


def _one_draw_decomposition():
    params = generate_parameter_points(1, seed=20260902)[0]
    F, Q, A, Z, R = build_lw_matrices(params, c=params.get("c", C_FIXED))
    real = kalman_smoother(_YOBS, _X, F, Q, A, Z, R, _XI00, _P00)
    rng = np.random.default_rng(20260903)
    draw = simulate_smoother_draw(
        _YOBS, _X, F, Q, A, Z, R, _XI00, _P00, rng, xi_smooth=real["xi_smooth"]
    )
    comp = state_shock_decomposition(F, draw.xi_draw, draw.eps_ystar, draw.eps_g, draw.eps_z)
    gp = gap_pi_shock_decomposition(
        F, A, Z, _X, _Y_FULL, _PI_FULL, draw.eps_is, draw.eps_pc, comp
    )
    return params, gp


def test_rdata_is_a_gap_and_pi_bar() -> None:
    assert "rdata" in GAP_BARS
    assert "rdata" in PI_BARS


def test_rdata_bar_satisfies_its_own_recursion_with_the_r_injection() -> None:
    """gap_rdata must satisfy EXACTLY gap_rdata[t] = a1*lag1 + a2*lag2 +
    (a_r/2)*(r_{t-1}+r_{t-2}), from zero pre-sample seeds -- the pure
    AR-propagated real-rate data contribution, with no r* term (the r data
    never enters the state) and no shock term."""
    params, gp = _one_draw_decomposition()
    a1, a2, a_r = params["a1"], params["a2"], params["a_r"]
    rdata = gp["gap"]["rdata"]
    lag1 = lag2 = 0.0
    for t in range(len(rdata)):
        expected = a1 * lag1 + a2 * lag2 + (a_r / 2.0) * (_X[t, 2] + _X[t, 3])
        assert rdata[t] == pytest.approx(expected, abs=1e-12), t
        lag2, lag1 = lag1, rdata[t]


def test_init_bar_no_longer_carries_the_r_data_injection() -> None:
    """The 'init' gap bar must be invariant to the real-rate DATA (x's
    columns 2/3): recomputing the decomposition with those columns zeroed
    changes 'rdata' but leaves 'init' bit-identical. (Before the split,
    'init' absorbed the whole r-data path and this test would fail.)"""
    params = generate_parameter_points(1, seed=20260902)[0]
    F, Q, A, Z, R = build_lw_matrices(params, c=params.get("c", C_FIXED))
    real = kalman_smoother(_YOBS, _X, F, Q, A, Z, R, _XI00, _P00)
    rng = np.random.default_rng(20260903)
    draw = simulate_smoother_draw(
        _YOBS, _X, F, Q, A, Z, R, _XI00, _P00, rng, xi_smooth=real["xi_smooth"]
    )
    comp = state_shock_decomposition(F, draw.xi_draw, draw.eps_ystar, draw.eps_g, draw.eps_z)

    gp = gap_pi_shock_decomposition(F, A, Z, _X, _Y_FULL, _PI_FULL, draw.eps_is, draw.eps_pc, comp)
    x_no_r = _X.copy()
    x_no_r[:, 2] = 0.0
    x_no_r[:, 3] = 0.0
    gp_no_r = gap_pi_shock_decomposition(
        F, A, Z, x_no_r, _Y_FULL, _PI_FULL, draw.eps_is, draw.eps_pc, comp
    )

    np.testing.assert_array_equal(gp["gap"]["init"], gp_no_r["gap"]["init"])
    assert not np.allclose(gp["gap"]["rdata"], gp_no_r["gap"]["rdata"])
    # With the r columns zeroed, the rdata bar has nothing to propagate.
    np.testing.assert_array_equal(gp_no_r["gap"]["rdata"], np.zeros_like(gp["gap"]["rdata"]))
