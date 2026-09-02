"""G6 gate (lw-sv-spec.md §5, G6 row): "HD reconstruction identity | exact to
1e-6 per period per draw."

This file IS gate G6. It exercises ``macrotoolkit.results_lw``'s
historical-decomposition machinery (spec §3.4) on a handful of simulation-
smoother draws -- at both a no-SV and an SV-style parameter point, per the
task brief's "cover at least one SV case" instruction -- and checks, PER
PERIOD, PER DRAW (not just a summary norm, so a failure names the exact
period):

1. B1 (state-level linear decomposition) reconstructs ``xi_draw`` exactly
   (to ~1e-9, a pure linear-algebra identity -- tighter than G6's own 1e-6
   bar, checked first as a cheap sanity gate before the harder B2/B3 checks).
2. B2 (gap's own AR(2) IS-curve shock-bar recursion) reconstructs
   ``yobs[:,0] - xi_draw[:,0]`` to G6's 1e-6 tolerance.
3. B3 (pi's own Phillips-curve shock-bar recursion) reconstructs
   ``yobs[:,1]`` to 1e-6.
4. B4 (y-level bars) reconstructs ``yobs[:,0]`` to 1e-6.

This is a reconstruction-IDENTITY test, not a statistical one: it either
holds exactly by construction or it doesn't, so a small number of draws (no
MCMC needed) is deliberately used -- correctness here does not average out
over many draws.
"""
from __future__ import annotations

import numpy as np
import pytest

from g1_harness import (
    C_FIXED,
    SYNTHETIC_DATA,
    generate_h_paths,
    generate_parameter_points,
)
from macrotoolkit.results_lw import (
    four_quarter_growth,
    gap_pi_shock_decomposition,
    state_shock_decomposition,
    y_level_decomposition,
)
from macrotoolkit.smoother import (
    build_lw_matrices,
    build_lw_regressors,
    default_initial_state,
    kalman_smoother,
    simulate_smoother_draw,
    sv_diag_variance_path,
)

# ---------------------------------------------------------------------------
# Shared fixtures: g1_harness's fixed synthetic dataset (see its own
# docstring -- plausible magnitudes only, not an economic fixture; that is
# appropriate here since G6 checks an algebraic identity, not economic
# plausibility).
# ---------------------------------------------------------------------------

_Y_FULL = np.asarray(SYNTHETIC_DATA.y, dtype=np.float64)
_PI_FULL = np.asarray(SYNTHETIC_DATA.pi, dtype=np.float64)
_R_FULL = np.asarray(SYNTHETIC_DATA.r, dtype=np.float64)
_YOBS, _X = build_lw_regressors(_Y_FULL, _PI_FULL, _R_FULL)
_XI00, _P00 = default_initial_state(float(_Y_FULL[4]))
_T = _YOBS.shape[0]

_N_DRAWS = 15
_PARAM_SEED = 20260909
_H_PATH_SEED = 20260910

_STATE_TOL = 1e-9
_G6_TOL = 1e-6  # spec §5's own G6 tolerance


def _draws_for_point(F, Q, A, Z, R, n_draws: int, seed: int):
    """Real-data smoothed states + n_draws DK simulation-smoother draws at
    one fixed system-matrix point."""
    real = kalman_smoother(_YOBS, _X, F, Q, A, Z, R, _XI00, _P00)
    rng = np.random.default_rng(seed)
    draws = [
        simulate_smoother_draw(_YOBS, _X, F, Q, A, Z, R, _XI00, _P00, rng, xi_smooth=real["xi_smooth"])
        for _ in range(n_draws)
    ]
    return draws


def _no_sv_case():
    params = generate_parameter_points(1, seed=_PARAM_SEED)[0]
    F, Q, A, Z, R = build_lw_matrices(params, c=params.get("c", C_FIXED))
    draws = _draws_for_point(F, Q, A, Z, R, _N_DRAWS, seed=_PARAM_SEED + 1)
    return F, Q, A, Z, R, draws


def _sv_case():
    params = generate_parameter_points(1, seed=_PARAM_SEED + 100)[0]
    F, Q, A, Z, _R_const = build_lw_matrices(params, c=params.get("c", C_FIXED))
    h_paths = generate_h_paths([params], _T, seed=_H_PATH_SEED)
    h = h_paths[0]  # (T, 2): column 0 = IS, column 1 = PC
    R = sv_diag_variance_path(h[:, 0], h[:, 1])
    draws = _draws_for_point(F, Q, A, Z, R, _N_DRAWS, seed=_PARAM_SEED + 101)
    return F, Q, A, Z, R, draws


_CASES = {"no_sv": _no_sv_case, "sv": _sv_case}


# ---------------------------------------------------------------------------
# 1. B1: state-level decomposition identity (cheap sanity gate)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", sorted(_CASES))
def test_g6_state_decomposition_reconstructs_xi_draw(case_name: str) -> None:
    """The B1 sum identity holds by construction (``"init"`` is defined as
    the residual -- see ``state_shock_decomposition``'s docstring), so this
    is a cheap regression guard on the plumbing, not a real correctness
    test of the decomposition by itself."""
    F, Q, A, Z, R, draws = _CASES[case_name]()
    for d_idx, draw in enumerate(draws):
        comp = state_shock_decomposition(F, draw.xi_draw, draw.eps_ystar, draw.eps_g, draw.eps_z)
        reconstructed = comp["init"] + comp["ystar"] + comp["g"] + comp["z"]
        diff = np.abs(reconstructed - draw.xi_draw)
        assert diff.max() < _STATE_TOL, (
            f"[{case_name}] B1 state decomposition fails to reconstruct "
            f"xi_draw for MC draw {d_idx}: max abs diff={diff.max():.3e} at "
            f"{np.unravel_index(np.argmax(diff), diff.shape)} (tol={_STATE_TOL:.1e})."
        )


@pytest.mark.parametrize("case_name", sorted(_CASES))
def test_g6_init_bar_satisfies_its_own_homogeneous_transition_for_t_ge_1(case_name: str) -> None:
    """A genuine (non-tautological) correctness check on the "init" bar,
    per its docstring: for t = 1..T-1 (excluding the documented t=0
    boundary artifact -- see ``state_shock_decomposition``'s docstring and
    ``tests/test_smoother_sim.py``'s own t=0 exclusion), the residual
    "init" bar must independently satisfy ``xi_init[t] == F @
    xi_init[t-1]`` -- this is NOT true by construction of the residual
    definition alone; it holds only because eps_ystar/eps_g/eps_z correctly
    account for every OTHER source of state-transition noise from t=1
    onward."""
    F, Q, A, Z, R, draws = _CASES[case_name]()
    for d_idx, draw in enumerate(draws):
        comp = state_shock_decomposition(F, draw.xi_draw, draw.eps_ystar, draw.eps_g, draw.eps_z)
        xi_init = comp["init"]
        predicted = xi_init[:-1] @ F.T  # row t-1 -> predicted row t, for t=1..T-1
        diff = np.abs(predicted - xi_init[1:])
        assert diff.max() < _STATE_TOL, (
            f"[{case_name}] init bar fails its own homogeneous F-transition "
            f"for MC draw {d_idx} at t>=1: max abs diff={diff.max():.3e} at "
            f"row {int(np.unravel_index(np.argmax(diff), diff.shape)[0]) + 1} "
            f"(tol={_STATE_TOL:.1e})."
        )


# ---------------------------------------------------------------------------
# 2. B2: gap's own shock-bar recursion -- G6 proper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", sorted(_CASES))
def test_g6_gap_hd_reconstructs_smoothed_series(case_name: str) -> None:
    F, Q, A, Z, R, draws = _CASES[case_name]()
    for d_idx, draw in enumerate(draws):
        comp = state_shock_decomposition(F, draw.xi_draw, draw.eps_ystar, draw.eps_g, draw.eps_z)
        gp = gap_pi_shock_decomposition(
            F, A, Z, _X, _Y_FULL, _PI_FULL, draw.eps_is, draw.eps_pc, comp
        )
        gap_sum = sum(gp["gap"].values())
        expected = _YOBS[:, 0] - draw.xi_draw[:, 0]
        diff = np.abs(gap_sum - expected)
        worst = int(np.argmax(diff))
        assert diff[worst] < _G6_TOL, (
            f"[{case_name}] G6 gap HD reconstruction fails for MC draw "
            f"{d_idx} at period index {worst}: bar_sum={gap_sum[worst]!r}, "
            f"expected={expected[worst]!r}, diff={diff[worst]:.3e} >= "
            f"{_G6_TOL:.1e}."
        )
        # Per-period assertion, not just the worst cell: a real bug should
        # fail at (close to) every period, not just the argmax.
        assert np.all(diff < _G6_TOL), (
            f"[{case_name}] G6 gap HD reconstruction fails at {int((diff >= _G6_TOL).sum())} "
            f"of {len(diff)} periods for MC draw {d_idx} (worst={diff.max():.3e})."
        )


# ---------------------------------------------------------------------------
# 3. B3: pi's own shock-bar recursion -- G6 proper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", sorted(_CASES))
def test_g6_pi_hd_reconstructs_smoothed_series(case_name: str) -> None:
    F, Q, A, Z, R, draws = _CASES[case_name]()
    for d_idx, draw in enumerate(draws):
        comp = state_shock_decomposition(F, draw.xi_draw, draw.eps_ystar, draw.eps_g, draw.eps_z)
        gp = gap_pi_shock_decomposition(
            F, A, Z, _X, _Y_FULL, _PI_FULL, draw.eps_is, draw.eps_pc, comp
        )
        pi_sum = sum(gp["pi"].values())
        expected = _YOBS[:, 1]
        diff = np.abs(pi_sum - expected)
        assert np.all(diff < _G6_TOL), (
            f"[{case_name}] G6 pi HD reconstruction fails for MC draw "
            f"{d_idx}: {int((diff >= _G6_TOL).sum())} of {len(diff)} periods "
            f"exceed tol={_G6_TOL:.1e} (worst={diff.max():.3e} at period "
            f"{int(np.argmax(diff))})."
        )


# ---------------------------------------------------------------------------
# 4. B4: y-level bars -- G6 proper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", sorted(_CASES))
def test_g6_y_level_hd_reconstructs_observed_series(case_name: str) -> None:
    F, Q, A, Z, R, draws = _CASES[case_name]()
    for d_idx, draw in enumerate(draws):
        comp = state_shock_decomposition(F, draw.xi_draw, draw.eps_ystar, draw.eps_g, draw.eps_z)
        gp = gap_pi_shock_decomposition(
            F, A, Z, _X, _Y_FULL, _PI_FULL, draw.eps_is, draw.eps_pc, comp
        )
        y_bars = y_level_decomposition(gp["gap"], comp)
        y_sum = sum(y_bars.values())
        expected = _YOBS[:, 0]
        diff = np.abs(y_sum - expected)
        assert np.all(diff < _G6_TOL), (
            f"[{case_name}] G6 y-level HD reconstruction fails for MC draw "
            f"{d_idx}: {int((diff >= _G6_TOL).sum())} of {len(diff)} periods "
            f"exceed tol={_G6_TOL:.1e} (worst={diff.max():.3e} at period "
            f"{int(np.argmax(diff))})."
        )

        # And the 4-quarter growth bars sum to the observed 4Q growth
        # wherever both are defined (rows 4..T-1).
        growth_bars = four_quarter_growth(y_bars)
        growth_sum = sum(growth_bars.values())
        expected_growth = expected[4:] - expected[:-4]
        diff_growth = np.abs(growth_sum[4:] - expected_growth)
        assert np.all(diff_growth < _G6_TOL), (
            f"[{case_name}] G6 4Q growth HD reconstruction fails for MC "
            f"draw {d_idx}: worst={diff_growth.max():.3e} >= {_G6_TOL:.1e}."
        )
        assert np.all(np.isnan(growth_sum[:4])), (
            f"[{case_name}] expected the first 4 4Q-growth rows to be NaN "
            f"(undefined -- would reach into the pre-sample window)."
        )


# ---------------------------------------------------------------------------
# Units-convention pin: h is log-VARIANCE, so a volatility path is
# exp(h/2) -- the arithmetic results_lw.compute_trend_cycle_draws applies
# directly to a draw's saved h_is/h_pc (lw-sv-spec.md §1.5, HANDOFF.md's S4
# warning). Pinned here as a small, local, worked-example check, matching
# tests/test_units_conventions.py's own pattern for this convention.
# ---------------------------------------------------------------------------


def test_volatility_path_convention_is_exp_h_over_2() -> None:
    h = np.array([0.0, np.log(4.0), -2.0 * np.log(2.0)])
    vol = np.exp(h / 2.0)
    np.testing.assert_allclose(vol, np.array([1.0, 2.0, 0.5]))
    # Negative check: exp(h) (the variance, not the sd) would give a very
    # different, wrong number for the same h -- guards against the classic
    # log-variance-vs-log-sd mixup.
    assert not np.allclose(np.exp(h), vol)
