"""G1 gate (lw-sv-spec.md §5, G1 row): "Python KF vs Stan KF log-likelihood,
50 random parameter points | max abs diff < 1e-8."

LIVE as of S2: `macrotoolkit.smoother` (the Python KF mirror) and the Stan
KF (`stan/functions/kalman_loglik_tv.stan` + `ssm_matrices_lw.stan`,
exposed through the test-only harness program
`stan/templates/g1_loglik_harness.stan.j2`) both exist, and the comparison
test below runs them against each other on the harness's fixed synthetic
dataset. The Stan side is evaluated in ONE compile + one fixed_param run for
all 50 points (`g1_harness.stan_kf_loglik_batch`); per-point agreement is
then asserted through `g1_harness.compare_loglik`, whose failure message
names the offending parameter point.

Requires a working CmdStan (pinned 2.36.0; see DECISIONS.md) -- the same
requirement the S1 acceptance tests already have.
"""
from __future__ import annotations

import numpy as np

from g1_harness import (
    C_FIXED,
    LOGLIK_TOL,
    N_PARAM_POINTS,
    PARAM_NAMES,
    PARAM_SEED,
    SYNTHETIC_DATA,
    compare_loglik,
    generate_h_paths,
    generate_parameter_points,
    python_kf_loglik,
    python_kf_loglik_tv,
    stan_kf_loglik_batch,
)


def test_generate_parameter_points_returns_requested_count() -> None:
    points = generate_parameter_points(N_PARAM_POINTS, seed=PARAM_SEED)
    assert len(points) == N_PARAM_POINTS


def test_generate_parameter_points_is_reproducible_given_same_seed() -> None:
    # G1 failures must be reproducible: same (n, seed) -> byte-identical
    # points, every time.
    points_a = generate_parameter_points(N_PARAM_POINTS, seed=PARAM_SEED)
    points_b = generate_parameter_points(N_PARAM_POINTS, seed=PARAM_SEED)
    assert points_a == points_b


def test_generated_parameter_points_satisfy_all_prior_constraints() -> None:
    """Every one of the 50 generated points satisfies every constraint the
    priors (spec §1.6 + the confirmed sigma_is/sigma_pc entry, see
    specs/schema/lw_sv.py) and structural equations (§1.2) impose."""
    points = generate_parameter_points(N_PARAM_POINTS, seed=PARAM_SEED)
    assert len(points) == N_PARAM_POINTS

    for p in points:
        assert set(p.keys()) == set(PARAM_NAMES) | {"c"}

        a1, a2 = p["a1"], p["a2"]

        # AR(2) stationarity triangle (spec §1.2's gap equation, §1.6's
        # "joint stationarity check via prior predictive").
        assert -1.0 < a2 < 1.0, p
        assert a1 + a2 < 1.0, p
        assert a2 - a1 < 1.0, p

        # Sign constraints enforced by prior support (spec §1.2, §1.6):
        # a_r < 0 (IS slope sign identification), b_y > 0 (Phillips slope
        # sign identification), b_pi on [0, 1] (Beta(8,2) support).
        assert p["a_r"] < 0.0, p
        assert p["b_y"] > 0.0, p
        assert 0.0 <= p["b_pi"] <= 1.0, p

        # Half-Normal scale parameters: strictly positive (P(X=0) = 0 for a
        # continuous half-normal; this also guards against an accidental
        # sign-flip bug turning "abs()" into a no-op).
        for scale in ("sigma_ystar", "sigma_g", "sigma_z", "sigma_is", "sigma_pc"):
            assert p[scale] > 0.0, (scale, p)

        # c fixed at 1.0, not sampled (spec default estimate_c: false).
        assert p["c"] == C_FIXED


def test_time_varying_R_shape_mismatch_raises() -> None:
    """The (T, m, m) R-path validation must fail loudly on a wrong length
    or rank (numerics-review suggestion 2026-08-31: the guard itself needs
    a negative test so a refactor can't silently drop it). The Stan-side
    twin is the reject() in kalman_loglik_tv.stan; it is exercised only at
    the reject site's expense of a compile, so only the Python side is
    pinned here."""
    import pytest

    from macrotoolkit.smoother import _as_R_path

    R_ok = np.eye(2)
    assert _as_R_path(R_ok, 5).shape == (5, 2, 2)
    with pytest.raises(ValueError, match="7 entries but yobs has T=5"):
        _as_R_path(np.zeros((7, 2, 2)), 5)
    with pytest.raises(ValueError, match="must be"):
        _as_R_path(np.zeros(3), 5)


def test_g1_python_kf_matches_stan_kf_loglik() -> None:
    """The G1 gate proper, covering BOTH filter paths of the generalized
    KF (S3): the constant-R overload and the time-varying R_t = diag(exp(h))
    array form, each Python-vs-Stan at all 50 points. The Stan side runs
    once for all points and both paths; each value is then compared against
    the Python mirror through `compare_loglik`, so a mismatch fails loudly
    naming the point.

    Seeds are `PARAM_SEED`/`H_PATH_SEED` (recorded in `tests/g1_harness.py`);
    if this ever fails for real, re-run `generate_parameter_points(
    N_PARAM_POINTS, seed=PARAM_SEED)` (+ `generate_h_paths`) to reproduce
    the exact failing inputs.
    """
    points = generate_parameter_points(N_PARAM_POINTS, seed=PARAM_SEED)
    h_paths = generate_h_paths(points, SYNTHETIC_DATA.T - 4)
    stan_ll, stan_ll_tv = stan_kf_loglik_batch(points, SYNTHETIC_DATA, h_paths)
    assert stan_ll.shape == (N_PARAM_POINTS,)
    assert stan_ll_tv.shape == (N_PARAM_POINTS,)
    assert np.all(np.isfinite(stan_ll))
    assert np.all(np.isfinite(stan_ll_tv))
    # The tv path must actually differ from the constant path (a wiring bug
    # returning the constant loglik twice would otherwise pass trivially).
    assert np.max(np.abs(stan_ll - stan_ll_tv)) > 1.0

    diffs = [
        compare_loglik(
            p,
            SYNTHETIC_DATA,
            python_kf_loglik,
            lambda _p, _d, i=i: float(stan_ll[i]),  # precomputed Stan value
            tol=LOGLIK_TOL,
        )
        for i, p in enumerate(points)
    ]
    assert max(diffs) < LOGLIK_TOL

    diffs_tv = [
        compare_loglik(
            p,
            SYNTHETIC_DATA,
            lambda _p, _d, i=i: python_kf_loglik_tv(_p, h_paths[i], _d),
            lambda _p, _d, i=i: float(stan_ll_tv[i]),
            tol=LOGLIK_TOL,
        )
        for i, p in enumerate(points)
    ]
    assert max(diffs_tv) < LOGLIK_TOL
