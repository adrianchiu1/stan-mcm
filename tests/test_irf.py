"""IRF matrix (spec §3.2) sanity tests for ``macrotoolkit.results_lw``'s
Part C (``impulse_response_for_shock``, ``irf_shock_size``,
``compute_irf_draws``).

Two kinds of coverage, mirroring ``tests/test_g6_hd_identity.py``'s own
split for Part B:

1. STRUCTURAL checks at hand-built parameter points (no MCMC needed --
   these are checkable, non-tautological facts about the model's own
   equations, not statistical properties that need averaging over draws),
   using ``g1_harness``'s parameter-point generator for a no-SV case and a
   fixed synthetic h-path for an SV case (the same pattern
   ``tests/test_g6_hd_identity.py`` uses).
2. PLUMBING checks (shapes/finiteness/``outputs.smoother_draws`` respect)
   against a real, tiny, completed lw_sv run -- the ``s4_no_sv_lw_run`` /
   ``s4_sv_lw_run`` fixtures from ``tests/conftest.py``, session-shared with
   ``tests/test_fan_charts.py`` so the underlying MCMC sampling only ever
   runs once per pytest session.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from g1_harness import C_FIXED, generate_h_paths, generate_parameter_points
from macrotoolkit.results_lw import (
    IRF_RESPONSES,
    IRF_SHOCKS,
    compute_irf_draws,
    impulse_response_for_shock,
    irf_shock_size,
    select_draw_indices,
)
from macrotoolkit.smoother import build_lw_matrices
from specs.schema.lw_sv import ThinSpec

_PARAM_SEED = 20260920
_H_PATH_SEED = 20260921
_HORIZON = 20


def _no_sv_point(seed_offset: int = 0) -> dict:
    return generate_parameter_points(1, seed=_PARAM_SEED + seed_offset)[0]


def _matrices_for(params: dict) -> tuple:
    return build_lw_matrices(params, c=params.get("c", C_FIXED))


def _sv_h_paths(params: dict, T: int, seed_offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    h = generate_h_paths([params], T, seed=_H_PATH_SEED + seed_offset)[0]
    return h[:, 0], h[:, 1]


# ---------------------------------------------------------------------------
# 1a. eps_ystar impulse: structural checks
# ---------------------------------------------------------------------------


def test_eps_ystar_gap_and_rstar_identically_zero() -> None:
    """A pure potential-level shock never enters g/z's state slots or the
    IS curve's rate-gap term -- checkable structural fact, not a
    tautology. rstar stays EXACTLY zero (a zero state path is zero with no
    arithmetic); gap is zero up to float cancellation only, since the
    engine (S5-decisions item 1) computes y via the measurement equation
    and subtracts y* -- the a1*y_{t-1} term and Z's -a1*y*_{t-1} entry
    cancel algebraically but round independently (~1 ulp of the response
    scale; DECISIONS.md 2026-09-02)."""
    params = _no_sv_point()
    F, Q, A, Z, R = _matrices_for(params)
    resp = impulse_response_for_shock(F, A, Z, "ystar", 0.42, _HORIZON)
    np.testing.assert_allclose(resp["gap"], np.zeros(_HORIZON), atol=1e-12)
    np.testing.assert_array_equal(resp["rstar"], np.zeros(_HORIZON))


def test_eps_ystar_y_response_equals_own_state_path() -> None:
    """y*_t's own state slot (0) is a random walk with no mean reversion and
    no other shock feeding it here, so after a single impulse at t=0 it
    stays at EXACTLY the shock size for every subsequent period (F[0,0]=1,
    no decay) -- a closed-form, independently-derivable fact, not merely a
    restatement of the implementation: since gap is identically 0 for this
    shock (checked above), y = gap + y* = y* = shock_size at every horizon."""
    params = _no_sv_point()
    F, Q, A, Z, R = _matrices_for(params)
    size = 0.37
    resp = impulse_response_for_shock(F, A, Z, "ystar", size, _HORIZON)
    np.testing.assert_allclose(resp["y"], np.full(_HORIZON, size))
    np.testing.assert_allclose(resp["g"], np.zeros(_HORIZON))


# ---------------------------------------------------------------------------
# 1b. eps_g / eps_z impulses: permanent (non-decaying) rstar response
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shock", ["g", "z"])
def test_trend_shock_produces_permanent_rstar_response(shock: str) -> None:
    """g and z are literal random walks (F[3,3] = F[5,5] = 1, no mean
    reversion) -- a unit impulse to either persists in r* = g + z EXACTLY,
    not merely "doesn't decay much": the response is a flat constant at
    every horizon, distinguishing a permanent trend shock from a transitory
    one (eps_is/eps_pc, checked below)."""
    params = _no_sv_point(seed_offset=1)
    F, Q, A, Z, R = _matrices_for(params)
    size = 0.11
    resp = impulse_response_for_shock(F, A, Z, shock, size, _HORIZON)
    np.testing.assert_allclose(resp["rstar"], np.full(_HORIZON, size))
    # And it stays at (not merely "close to") its period-1 value.
    np.testing.assert_allclose(resp["rstar"], np.full(_HORIZON, resp["rstar"][0]))


def test_eps_g_own_g_response_is_also_permanent() -> None:
    params = _no_sv_point(seed_offset=2)
    F, Q, A, Z, R = _matrices_for(params)
    size = 0.05
    resp = impulse_response_for_shock(F, A, Z, "g", size, _HORIZON)
    np.testing.assert_allclose(resp["g"], np.full(_HORIZON, size))


# ---------------------------------------------------------------------------
# 2. eps_is impulse: gap responds at horizon 1, pi responds only from
#    horizon 2 (the Phillips curve's own gap_{t-1} lag).
# ---------------------------------------------------------------------------


def test_eps_is_gap_nonzero_at_first_horizon_and_bounded() -> None:
    params = _no_sv_point(seed_offset=3)
    F, Q, A, Z, R = _matrices_for(params)
    size = 0.25
    resp = impulse_response_for_shock(F, A, Z, "is", size, _HORIZON)
    assert resp["gap"][0] == pytest.approx(size)
    assert np.all(np.isfinite(resp["gap"]))
    # (a1, a2) is drawn from the AR(2) stationarity triangle (g1_harness's
    # rejection sampler) -- the impulse response of a stationary AR(2) is
    # bounded; check it does not explode (a generous bound, not a tight
    # decay-rate assertion, to avoid flakiness near the stationarity
    # boundary).
    assert np.max(np.abs(resp["gap"])) < 100.0 * abs(size)


def test_eps_is_pi_response_delayed_by_one_period() -> None:
    """pi's own equation only ever picks up gap_{t-1} (spec §1.2) -- a
    ONE-period-delayed channel. At horizon index 0 (the shock's own period)
    pi's response must be IDENTICALLY zero (gap_{t-1} at that point is still
    the pre-impulse value, 0); from horizon index 1 onward it must be
    nonzero (gap's own horizon-0 response has fed into pi's lag by then)."""
    params = _no_sv_point(seed_offset=3)
    F, Q, A, Z, R = _matrices_for(params)
    size = 0.25
    resp = impulse_response_for_shock(F, A, Z, "is", size, _HORIZON)
    assert resp["pi"][0] == 0.0
    assert resp["pi"][1] != 0.0


# ---------------------------------------------------------------------------
# 3. eps_pc impulse: no real-side channel at all; pi decays via its own AR.
# ---------------------------------------------------------------------------


def test_eps_pc_real_side_identically_zero_every_horizon() -> None:
    params = _no_sv_point(seed_offset=4)
    F, Q, A, Z, R = _matrices_for(params)
    resp = impulse_response_for_shock(F, A, Z, "pc", 0.3, _HORIZON)
    np.testing.assert_array_equal(resp["gap"], np.zeros(_HORIZON))
    np.testing.assert_array_equal(resp["rstar"], np.zeros(_HORIZON))
    np.testing.assert_array_equal(resp["y"], np.zeros(_HORIZON))
    np.testing.assert_array_equal(resp["g"], np.zeros(_HORIZON))


def test_eps_pc_pi_response_nonzero_then_ar_decaying() -> None:
    params = _no_sv_point(seed_offset=4)
    F, Q, A, Z, R = _matrices_for(params)
    size = 0.3
    resp = impulse_response_for_shock(F, A, Z, "pc", size, _HORIZON)
    assert resp["pi"][0] == pytest.approx(size)
    assert np.all(np.isfinite(resp["pi"]))
    # b_pi in (0, 1) (Beta(8,2) prior) -- pi's own AR feedback is a
    # contraction, so the response should be materially smaller by the end
    # of a 20-period horizon than at its own initial impulse.
    assert abs(resp["pi"][-1]) < abs(resp["pi"][0])


# ---------------------------------------------------------------------------
# 4. Linearity in shock size (regression guard).
# ---------------------------------------------------------------------------


def test_halving_shock_size_exactly_halves_response() -> None:
    params = _no_sv_point(seed_offset=5)
    F, Q, A, Z, R = _matrices_for(params)
    full = impulse_response_for_shock(F, A, Z, "g", 0.08, _HORIZON)
    half = impulse_response_for_shock(F, A, Z, "g", 0.04, _HORIZON)
    for name in ("gap", "pi", "rstar", "y", "g"):
        np.testing.assert_allclose(half[name], full[name] / 2.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. irf_shock_size
# ---------------------------------------------------------------------------


def test_irf_shock_size_no_sv_reads_constant_sigma() -> None:
    flat = {
        "sigma_ystar": np.array([0.5]),
        "sigma_g": np.array([0.02]),
        "sigma_z": np.array([0.05]),
        "sigma_is": np.array([0.3]),
        "sigma_pc": np.array([0.4]),
    }
    for shock in IRF_SHOCKS:
        size = irf_shock_size(shock, flat, 0, sv_on=False, irf_vol_reference="end_of_sample", h_is=None, h_pc=None)
        assert size == pytest.approx(float(flat[f"sigma_{shock}"][0]))


def test_irf_shock_size_sv_trend_shocks_ignore_h() -> None:
    """ystar/g/z are never SV shocks -- their size comes from the constant
    sigma_* posterior regardless of sv_on/h paths."""
    flat = {"sigma_ystar": np.array([0.5]), "sigma_g": np.array([0.02]), "sigma_z": np.array([0.05])}
    h_is = np.array([1.0, 2.0, 3.0])
    h_pc = np.array([-1.0, -2.0, -3.0])
    for shock in ("ystar", "g", "z"):
        size = irf_shock_size(shock, flat, 0, sv_on=True, irf_vol_reference="end_of_sample", h_is=h_is, h_pc=h_pc)
        assert size == pytest.approx(float(flat[f"sigma_{shock}"][0]))


def test_irf_shock_size_sv_end_of_sample_vs_sample_mean_differ() -> None:
    flat: dict = {}
    h_is = np.array([0.0, 1.0, 2.0])  # log-VARIANCE path, rising
    h_pc = np.array([0.0, -1.0, -2.0])  # falling
    size_eos_is = irf_shock_size("is", flat, 0, sv_on=True, irf_vol_reference="end_of_sample", h_is=h_is, h_pc=h_pc)
    size_mean_is = irf_shock_size("is", flat, 0, sv_on=True, irf_vol_reference="sample_mean", h_is=h_is, h_pc=h_pc)
    assert size_eos_is == pytest.approx(np.exp(h_is[-1] / 2.0))
    assert size_mean_is == pytest.approx(np.exp(h_is.mean() / 2.0))
    assert size_eos_is != pytest.approx(size_mean_is)

    size_eos_pc = irf_shock_size("pc", flat, 0, sv_on=True, irf_vol_reference="end_of_sample", h_is=h_is, h_pc=h_pc)
    assert size_eos_pc == pytest.approx(np.exp(h_pc[-1] / 2.0))
    # Negative check: end_of_sample must NOT equal mean(exp(h/2)) (the OTHER
    # defensible-but-not-chosen convention) -- guards against silently
    # picking the wrong one of the two.
    assert size_mean_is != pytest.approx(float(np.mean(np.exp(h_is / 2.0))))


def test_irf_shock_size_rejects_unknown_shock() -> None:
    with pytest.raises(ValueError, match="shock"):
        irf_shock_size("bogus", {}, 0, sv_on=False, irf_vol_reference="end_of_sample", h_is=None, h_pc=None)


def test_irf_shock_size_rejects_bad_vol_reference() -> None:
    h = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="irf_vol_reference"):
        irf_shock_size("is", {}, 0, sv_on=True, irf_vol_reference="bogus", h_is=h, h_pc=h)


def test_impulse_response_for_shock_rejects_unknown_shock() -> None:
    params = _no_sv_point(seed_offset=6)
    F, Q, A, Z, R = _matrices_for(params)
    with pytest.raises(ValueError, match="shock"):
        impulse_response_for_shock(F, A, Z, "bogus", 1.0, _HORIZON)


# ---------------------------------------------------------------------------
# 6. compute_irf_draws: plumbing against a real, tiny, completed run.
# ---------------------------------------------------------------------------


def test_compute_irf_draws_no_sv_shapes_and_finite(s4_no_sv_lw_run) -> None:
    irf = compute_irf_draws(s4_no_sv_lw_run)
    post = s4_no_sv_lw_run.idata.posterior
    n_total = post.sizes["chain"] * post.sizes["draw"]
    expected_idx = select_draw_indices(n_total, s4_no_sv_lw_run.spec.outputs.smoother_draws)
    np.testing.assert_array_equal(irf.draw_indices, expected_idx)
    assert irf.horizon == s4_no_sv_lw_run.spec.outputs.irf_horizon

    n_draws = len(expected_idx)
    assert set(irf.responses) == set(IRF_SHOCKS)
    for shock in IRF_SHOCKS:
        assert set(irf.responses[shock]) == set(IRF_RESPONSES)
        for name, arr in irf.responses[shock].items():
            assert arr.shape == (n_draws, irf.horizon), (shock, name)
            assert np.all(np.isfinite(arr)), (shock, name)


def test_compute_irf_draws_sv_shapes_and_finite(s4_sv_lw_run) -> None:
    irf = compute_irf_draws(s4_sv_lw_run)
    n_draws = len(irf.draw_indices)
    for shock in IRF_SHOCKS:
        for name, arr in irf.responses[shock].items():
            assert arr.shape == (n_draws, irf.horizon), (shock, name)
            assert np.all(np.isfinite(arr)), (shock, name)
    # eps_ystar's gap/rstar columns are zero at EVERY draw too (not just
    # the hand-built structural test above) -- a real-run end-to-end
    # regression guard. gap to float cancellation only (the engine's
    # measurement-equation grouping -- see
    # test_eps_ystar_gap_and_rstar_identically_zero); rstar exactly.
    np.testing.assert_allclose(irf.responses["ystar"]["gap"], np.zeros_like(irf.responses["ystar"]["gap"]), atol=1e-12)
    np.testing.assert_array_equal(irf.responses["ystar"]["rstar"], np.zeros_like(irf.responses["ystar"]["rstar"]))


def test_compute_irf_draws_honors_thin_smoother_draws(s4_no_sv_lw_run) -> None:
    thinned = dataclasses.replace(
        s4_no_sv_lw_run,
        spec=s4_no_sv_lw_run.spec.model_copy(
            update={"outputs": s4_no_sv_lw_run.spec.outputs.model_copy(update={"smoother_draws": ThinSpec(thin=6)})}
        ),
    )
    irf = compute_irf_draws(thinned)
    post = thinned.idata.posterior
    n_total = post.sizes["chain"] * post.sizes["draw"]
    expected_idx = np.arange(0, n_total, 6)
    np.testing.assert_array_equal(irf.draw_indices, expected_idx)
    assert irf.responses["is"]["gap"].shape[0] == len(expected_idx)
