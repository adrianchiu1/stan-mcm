"""G1 gate (lw-sv-spec.md §5, G1 row): "Python KF vs Stan KF log-likelihood,
50 random parameter points | max abs diff < 1e-8."

Neither the Python KF mirror (`macrotoolkit.smoother`, S2 scope) nor the
Stan KF function (`stan/functions/kalman_loglik_tv.stan`, S2 scope) exists
yet -- both are explicitly out of scope for this pass (see
`plans/S2-plan.md`). What *is* testable now, and pinned below as a real
passing test, is that the parameter-point generator itself
(`tests/g1_harness.py::generate_parameter_points`) actually produces points
that satisfy every constraint the spec's priors impose. The G1 comparison
test proper is structurally complete (it already calls
`tests.g1_harness.compare_loglik` against the two stub functions) but marked
`skip` until S2 lands the real functions -- at that point S2 only needs to:

1. remove the `@pytest.mark.skip` decorator, and
2. replace `python_kf_loglik_stub` / `stan_kf_loglik_stub` below with the
   real KF log-likelihood functions

to turn this into the live G1 gate.
"""
from __future__ import annotations

import pytest

from g1_harness import (
    C_FIXED,
    LOGLIK_TOL,
    N_PARAM_POINTS,
    PARAM_SEED,
    SYNTHETIC_DATA,
    compare_loglik,
    generate_parameter_points,
    python_kf_loglik_stub,
    stan_kf_loglik_stub,
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
    """Real, passing test: pins down that every one of the 50 generated
    points actually satisfies every constraint the spec's priors (§1.6) and
    structural equations (§1.2) impose -- this is genuinely checkable now,
    independent of whether the KF mirrors exist."""
    points = generate_parameter_points(N_PARAM_POINTS, seed=PARAM_SEED)
    assert len(points) == N_PARAM_POINTS

    for p in points:
        assert set(p.keys()) == {
            "a1",
            "a2",
            "a_r",
            "b_pi",
            "b_y",
            "sigma_ystar",
            "sigma_g",
            "sigma_z",
            "c",
        }

        a1, a2 = p["a1"], p["a2"]
        a_r, b_pi, b_y = p["a_r"], p["b_pi"], p["b_y"]
        sigma_ystar, sigma_g, sigma_z = p["sigma_ystar"], p["sigma_g"], p["sigma_z"]

        # AR(2) stationarity triangle (spec §1.2's gap equation, §1.6's
        # "joint stationarity check via prior predictive").
        assert -1.0 < a2 < 1.0, p
        assert a1 + a2 < 1.0, p
        assert a2 - a1 < 1.0, p

        # Sign constraints enforced by prior support (spec §1.2, §1.6):
        # a_r < 0 (IS slope sign identification), b_y > 0 (Phillips slope
        # sign identification), b_pi on [0, 1] (Beta(8,2) support).
        assert a_r < 0.0, p
        assert b_y > 0.0, p
        assert 0.0 <= b_pi <= 1.0, p

        # Half-Normal scale parameters: strictly positive (P(X=0) = 0 for a
        # continuous half-normal; this also guards against an accidental
        # sign-flip bug turning "abs()" into a no-op).
        assert sigma_ystar > 0.0, p
        assert sigma_g > 0.0, p
        assert sigma_z > 0.0, p

        # c fixed at 1.0, not sampled (spec default estimate_c: false).
        assert p["c"] == C_FIXED


@pytest.mark.skip(
    reason="awaiting macrotoolkit.smoother + stan/functions/kalman_loglik_tv.stan (S2)"
)
def test_g1_python_kf_matches_stan_kf_loglik() -> None:
    """The actual G1 gate. Structurally complete already -- this is exactly
    what S2 runs once the skip is removed and the two stub references below
    are swapped for the real Python and Stan KF log-likelihood functions.

    Seed is `PARAM_SEED` (recorded in `tests/g1_harness.py`); if this ever
    fails for real, re-run `generate_parameter_points(N_PARAM_POINTS,
    seed=PARAM_SEED)` to reproduce the exact failing parameter point set.
    """
    points = generate_parameter_points(N_PARAM_POINTS, seed=PARAM_SEED)
    diffs = [
        compare_loglik(
            p,
            SYNTHETIC_DATA,
            python_kf_loglik_stub,
            stan_kf_loglik_stub,
            tol=LOGLIK_TOL,
        )
        for p in points
    ]
    assert max(diffs) < LOGLIK_TOL
