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
    PRE_QT_FIXTURE,
    PRE_QT_TOL,
    PRE_ZT_FIXTURE,
    SYNTHETIC_DATA,
    compare_loglik,
    generate_h_paths,
    generate_parameter_points,
    generate_q_paths,
    generate_q_sv_inputs,
    generate_sv_inputs,
    generate_z_scales,
    python_kf_loglik,
    python_kf_loglik_sv,
    python_kf_loglik_svq,
    python_kf_loglik_tv,
    python_kf_loglik_tvq,
    python_kf_loglik_tvz,
    python_kf_loglik_tvzqr,
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


def test_q_path_normalization_and_constant_q_bit_identity() -> None:
    """S6 Q_t generalization, Python side: ``_as_Q_path`` tiles a constant
    Q / validates a path (the ``_as_R_path`` twin), and a (T, n, n) path
    whose every entry IS the constant Q yields a BIT-IDENTICAL log-
    likelihood to the constant call -- the arithmetic ``F P F' + Q[t]``
    is the same operation on the same values."""
    import pytest

    from macrotoolkit.smoother import (
        _as_Q_path,
        build_lw_matrices,
        build_lw_regressors,
        default_initial_state,
        kalman_loglik,
    )

    assert _as_Q_path(np.eye(7), 5).shape == (5, 7, 7)
    with pytest.raises(ValueError, match="7 entries but yobs has T=5"):
        _as_Q_path(np.zeros((7, 7, 7)), 5)
    with pytest.raises(ValueError, match="must be"):
        _as_Q_path(np.zeros(3), 5)

    p = generate_parameter_points(3, seed=PARAM_SEED)[0]
    yobs, x = build_lw_regressors(SYNTHETIC_DATA.y, SYNTHETIC_DATA.pi, SYNTHETIC_DATA.r)
    F, Q, A, Z, R = build_lw_matrices(p, c=C_FIXED)
    xi00, P00 = default_initial_state(float(SYNTHETIC_DATA.y[4]))
    ll_const = kalman_loglik(yobs, x, F, Q, A, Z, R, xi00, P00)
    ll_path = kalman_loglik(yobs, x, F, np.repeat(Q[None], yobs.shape[0], axis=0), A, Z, R, xi00, P00)
    assert ll_const == ll_path


def test_constant_q_stan_loglik_reproduces_pre_generalization_fixture() -> None:
    """The constant-Q REGRESSION PIN (plans/S6-plan.md conflict item 1):
    the generalized Stan filter's constant-Q values -- all three pre-S6
    paths (constant R, time-varying R, production SV composition) -- must
    reproduce the log-likelihoods captured from the UNTOUCHED S5 program
    at the same 50 points (tests/fixtures/g1/pre_qt_stan_loglik.csv,
    captured 2026-09-04 before any stan/ edit) to CmdStan's 18-sig-fig
    output resolution. This is what "the Q_t generalization does not change
    constant-Q numerics" means, on the Stan side, independently of the
    Python mirror."""
    import csv
    from pathlib import Path

    points = generate_parameter_points(N_PARAM_POINTS, seed=PARAM_SEED)
    stan_ll, stan_ll_tv, stan_ll_sv, _, _, _, _ = stan_kf_loglik_batch(points, SYNTHETIC_DATA)
    rows = list(csv.DictReader((Path(__file__).parent / PRE_QT_FIXTURE).open()))
    assert len(rows) == N_PARAM_POINTS
    for i, row in enumerate(rows):
        assert int(row["point"]) == i
        for got, key in ((stan_ll[i], "loglik"), (stan_ll_tv[i], "loglik_tv"), (stan_ll_sv[i], "loglik_sv")):
            want = float(row[key])
            assert abs(float(got) - want) < PRE_QT_TOL, (
                f"constant-Q regression pin failed at point {i}, path {key}: "
                f"got {got!r}, pre-generalization fixture {want!r}"
            )


def test_constant_z_stan_loglik_reproduces_pre_zt_fixture_exactly() -> None:
    """The constant-Z REGRESSION PIN (plans/S9-plan.md decision 1): the
    Z_t-generalized Stan filter's constant-Z values on ALL FIVE S8 paths
    (constant, time-varying R, the R_t SV composition, time-varying Q, the
    Q_t SV composition) must reproduce the log-likelihoods captured from
    the UNTOUCHED S8 program at the same 50 points
    (tests/fixtures/g1/pre_zt_stan_loglik.csv, captured 2026-09-05 before
    any stan/ edit) with difference EXACTLY 0.0: ``Z[t] P Z[t]'`` with
    every ``Z[t]`` the same matrix is the same arithmetic as ``Z P Z'``."""
    import csv
    from pathlib import Path

    points = generate_parameter_points(N_PARAM_POINTS, seed=PARAM_SEED)
    got = stan_kf_loglik_batch(points, SYNTHETIC_DATA)[:5]
    rows = list(csv.DictReader((Path(__file__).parent / PRE_ZT_FIXTURE).open()))
    assert len(rows) == N_PARAM_POINTS
    worst = 0.0
    for i, row in enumerate(rows):
        assert int(row["point"]) == i
        for arr, key in zip(got, ("loglik", "loglik_tv", "loglik_sv", "loglik_tvq", "loglik_svq")):
            want = float(row[key])
            worst = max(worst, abs(float(arr[i]) - want))
            assert float(arr[i]) == want, (
                f"constant-Z regression pin failed at point {i}, path {key}: got {float(arr[i])!r}, pre-S9 fixture {want!r}"
            )
    assert worst == 0.0


def test_g1_python_kf_matches_stan_kf_loglik() -> None:
    """The G1 gate proper, covering all FIVE filter paths of the
    generalized KF: the constant-Q/constant-R overload, the time-varying
    R_t = diag(exp(h)) array form, the production R_t SV-helper composition
    (sv_rw_noncentered -> sv_diag_variance_path -> KF) -- S3 -- and, S6,
    the time-varying Q_t path (the g shock's scale through lw_Q per
    period) and the production Q_t SV composition (sv_rw_noncentered ->
    sv_scalar_variance_path on Q's z slot), each Python-vs-Stan at all 50
    points. The Stan side runs once for all points and paths; each value is
    then compared against the Python mirror through `compare_loglik`, so a
    mismatch fails loudly naming the point.

    Seeds are `PARAM_SEED`/`H_PATH_SEED`/`SV_INPUT_SEED`/`Q_PATH_SEED`/
    `Q_SV_INPUT_SEED` (recorded in `tests/g1_harness.py`); if this ever
    fails for real, re-run the generators at those seeds to reproduce the
    exact failing inputs.
    """
    points = generate_parameter_points(N_PARAM_POINTS, seed=PARAM_SEED)
    h_paths = generate_h_paths(points, SYNTHETIC_DATA.T - 4)
    sv_inputs = generate_sv_inputs(points, SYNTHETIC_DATA.T - 4)
    q_paths = generate_q_paths(points, SYNTHETIC_DATA.T - 4)
    q_sv_inputs = generate_q_sv_inputs(points, SYNTHETIC_DATA.T - 4)
    z_scales = generate_z_scales(points, SYNTHETIC_DATA.T - 4)
    stan_ll, stan_ll_tv, stan_ll_sv, stan_ll_tvq, stan_ll_svq, stan_ll_tvz, stan_ll_tvzqr = stan_kf_loglik_batch(
        points, SYNTHETIC_DATA, h_paths, sv_inputs, q_paths, q_sv_inputs, z_scales
    )
    for arr in (stan_ll, stan_ll_tv, stan_ll_sv, stan_ll_tvq, stan_ll_svq, stan_ll_tvz, stan_ll_tvzqr):
        assert arr.shape == (N_PARAM_POINTS,)
        assert np.all(np.isfinite(arr))
    # The tv/sv paths must actually differ from the constant path (a wiring
    # bug returning the constant loglik would otherwise pass trivially).
    assert np.max(np.abs(stan_ll - stan_ll_tv)) > 1.0
    assert np.max(np.abs(stan_ll - stan_ll_sv)) > 1.0
    assert np.max(np.abs(stan_ll - stan_ll_tvq)) > 1e-3
    assert np.max(np.abs(stan_ll - stan_ll_svq)) > 1e-3
    assert np.max(np.abs(stan_ll - stan_ll_tvz)) > 1e-3
    assert np.max(np.abs(stan_ll_tv - stan_ll_tvzqr)) > 1e-3

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

    diffs_sv = [
        compare_loglik(
            p,
            SYNTHETIC_DATA,
            lambda _p, _d, i=i: python_kf_loglik_sv(_p, sv_inputs, i, _d),
            lambda _p, _d, i=i: float(stan_ll_sv[i]),
            tol=LOGLIK_TOL,
        )
        for i, p in enumerate(points)
    ]
    assert max(diffs_sv) < LOGLIK_TOL

    diffs_tvq = [
        compare_loglik(
            p,
            SYNTHETIC_DATA,
            lambda _p, _d, i=i: python_kf_loglik_tvq(_p, q_paths[i], _d),
            lambda _p, _d, i=i: float(stan_ll_tvq[i]),
            tol=LOGLIK_TOL,
        )
        for i, p in enumerate(points)
    ]
    assert max(diffs_tvq) < LOGLIK_TOL

    diffs_svq = [
        compare_loglik(
            p,
            SYNTHETIC_DATA,
            lambda _p, _d, i=i: python_kf_loglik_svq(_p, q_sv_inputs, i, _d),
            lambda _p, _d, i=i: float(stan_ll_svq[i]),
            tol=LOGLIK_TOL,
        )
        for i, p in enumerate(points)
    ]
    assert max(diffs_svq) < LOGLIK_TOL

    # S9: the Z_t path (constant Q, R) and the full time-varying core.
    diffs_tvz = [
        compare_loglik(
            p,
            SYNTHETIC_DATA,
            lambda _p, _d, i=i: python_kf_loglik_tvz(_p, z_scales[i], _d),
            lambda _p, _d, i=i: float(stan_ll_tvz[i]),
            tol=LOGLIK_TOL,
        )
        for i, p in enumerate(points)
    ]
    assert max(diffs_tvz) < LOGLIK_TOL

    diffs_tvzqr = [
        compare_loglik(
            p,
            SYNTHETIC_DATA,
            lambda _p, _d, i=i: python_kf_loglik_tvzqr(_p, z_scales[i], q_paths[i], h_paths[i], _d),
            lambda _p, _d, i=i: float(stan_ll_tvzqr[i]),
            tol=LOGLIK_TOL,
        )
        for i, p in enumerate(points)
    ]
    assert max(diffs_tvzqr) < LOGLIK_TOL
    print(f"G1 max |Stan - Python|: const {max(diffs):.2e} tvR {max(diffs_tv):.2e} svR {max(diffs_sv):.2e} "
          f"tvQ {max(diffs_tvq):.2e} svQ {max(diffs_svq):.2e} tvZ {max(diffs_tvz):.2e} tvZQR {max(diffs_tvzqr):.2e}")
