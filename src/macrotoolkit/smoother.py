"""Python mirror of the Stan Kalman-filter machinery for the LW model.

Scope (S2 + the S3 KF generalization, plans/S3-plan.md, + S4's simulation
smoother, plans/S4-plan.md, + the S6 Q_t generalization, plans/S6-plan.md):
the KF log-likelihood (gate G1) with constant OR time-varying measurement
covariance R_t AND constant OR time-varying state-innovation covariance
Q_t (lw_sv's SV enters through R_t only -- in the marginalized LW form the
SV shocks are measurement errors; UCSV's trend-shock SV enters through
Q_t, the one real KF extension family #2 needed, done by the exact R_t
playbook: array-of-matrices core + constant-case delegation), the RTS
fixed-interval smoother (gate G5a), and the Durbin-Koopman *simulation*
smoother of spec §2.4 (Python-only -- no `stan/` mirror; validated by
tests/test_smoother_sim.py instead of a G1-style Stan comparison).

Four layers, the first three mirroring the Stan side exactly:

1. ``build_lw_matrices``      <-> ``stan/functions/ssm_matrices_lw.stan``
2. ``kalman_loglik``           <-> ``stan/functions/kalman_loglik_tv.stan``
3. ``kalman_smoother``         (Python only; validated against the HLW
                                oracle in G5a)
4. ``simulate_smoother_draw``  (Python only, S4; per-draw joint state +
                                structural-shock path draws feeding the
                                trend-cycle objects, historical
                                decomposition, and fan-chart seeds -- spec
                                §2.4)

State-space form (Hamilton/HLW notation; all constants stamped, no options):

    state (n=7):  xi_t = [y*_t, y*_{t-1}, y*_{t-2},
                          g_{t-1}, g_{t-2}, z_{t-1}, z_{t-2}]
    obs   (m=2):  yobs_t = [y_t, pi_t]
    exog  (k=6):  x_t = [y_{t-1}, y_{t-2}, r_{t-1}, r_{t-2},
                         pi_{t-1}, (pi_{t-2}+pi_{t-3}+pi_{t-4})/3]

    xi_t    = F xi_{t-1} + w_t,        w_t ~ N(0, Q)
    yobs_t  = A' x_t + Z xi_t + e_t,   e_t ~ N(0, R_t)

Units conventions (lw-sv-spec.md §1.1/§1.3, enforced by
tests/test_units_conventions.py): ``g`` is ANNUALIZED everywhere in this
module -- priors, states, outputs. Only the potential-output transition
divides by 4 (``y*_t = y*_{t-1} + g_{t-1}/4 + eps``), which is where the
F[0,3] = 0.25 and the /16, /4 factors in Q come from. r* = c*g + z needs no
4x factor precisely because g is annualized. (HLW's own code keeps g
quarterly and multiplies by 4 at reporting time; the two parameterizations
are exact unit transforms of each other -- see
tests/test_g5a_hlw_replication.py for the S = diag(1,1,1,4,4,1,1)
conversion of their initial conditions.)

Timing note: the state carries y* contemporaneously but g and z lagged one
quarter (HLW's own layout). In terms of xi_{t-1} (whose 4th element is
g_{t-2}), the y* transition is y*_t = y*_{t-1} + g_{t-2}/4 + (eps_g,{t-1}/4
+ eps_y*,t) -- i.e. the period's g innovation feeds both the g state and,
divided by 4, the y* state, which is exactly the off-diagonal Q[0,3] term.

Numerics (spec §2.2): the KF uses Cholesky factorization of the innovation
covariance (never a raw inverse/determinant) and explicitly re-symmetrizes
every covariance after each update, in the same order as the Stan function,
so the two mirrors agree to ~1e-8 (gate G1) rather than merely "closely".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from specs.schema.lw_sv import INITIAL_STATE_PRIOR

N_STATE = 7
N_OBS = 2
N_EXOG = 6


# ---------------------------------------------------------------------------
# 1. State-space matrix construction (mirror of ssm_matrices_lw.stan)
# ---------------------------------------------------------------------------


def build_lw_matrices(params: dict, c: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble (F, Q, A, Z, R) for the no-SV LW model at one parameter
    point.

    ``params`` needs keys: a1, a2, a_r, b_pi, b_y, sigma_ystar, sigma_g,
    sigma_z, sigma_is, sigma_pc (all scale parameters in sd units, g-related
    ones annualized). ``c`` is the r* = c*g + z loading (spec §1.3; fixed
    1.0 in S2).

    Returns:
        F: (7,7) transition;  Q: (7,7) state innovation covariance;
        A: (6,2) exogenous loading (obs = A' x + ...);  Z: (2,7) state
        loading;  R: (2,2) measurement covariance.
    """
    a1, a2, a_r = params["a1"], params["a2"], params["a_r"]
    b_pi, b_y = params["b_pi"], params["b_y"]
    s_ystar, s_g, s_z = params["sigma_ystar"], params["sigma_g"], params["sigma_z"]
    s_is, s_pc = params["sigma_is"], params["sigma_pc"]

    F = np.zeros((N_STATE, N_STATE))
    F[0, 0] = 1.0
    F[0, 3] = 0.25  # g annualized; quarterly increment is g/4 (spec §1.3)
    F[1, 0] = 1.0
    F[2, 1] = 1.0
    F[3, 3] = 1.0
    F[4, 3] = 1.0
    F[5, 5] = 1.0
    F[6, 5] = 1.0

    Q = np.zeros((N_STATE, N_STATE))
    # y* shock = eps_ystar + eps_g/4 (annualized g innovation enters the
    # quarterly y* step divided by 4); g shock = eps_g; z shock = eps_z.
    Q[0, 0] = s_ystar**2 + s_g**2 / 16.0
    Q[0, 3] = Q[3, 0] = s_g**2 / 4.0
    Q[3, 3] = s_g**2
    Q[5, 5] = s_z**2

    A = np.zeros((N_EXOG, N_OBS))
    A[0, 0] = a1
    A[1, 0] = a2
    A[2, 0] = a_r / 2.0
    A[3, 0] = a_r / 2.0
    A[0, 1] = b_y
    A[4, 1] = b_pi
    A[5, 1] = 1.0 - b_pi

    Z = np.zeros((N_OBS, N_STATE))
    Z[0, 0] = 1.0
    Z[0, 1] = -a1
    Z[0, 2] = -a2
    # -(a_r/2) * r*_{t-j} with r* = c*g + z, g annualized: no 4x factor.
    Z[0, 3] = -c * a_r / 2.0
    Z[0, 4] = -c * a_r / 2.0
    Z[0, 5] = -a_r / 2.0
    Z[0, 6] = -a_r / 2.0
    Z[1, 1] = -b_y

    R = np.diag([s_is**2, s_pc**2])

    return F, Q, A, Z, R


def build_lw_regressors(y: np.ndarray, pi: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the observation matrix and exogenous-regressor matrix from raw
    series that INCLUDE the four pre-sample lag quarters.

    Inputs are full-length arrays of T+4 quarters (spec/HLW convention: data
    must start 4 quarters before the estimation sample, for pi's t-4 lag).
    Returns (yobs, x): yobs is (T,2) = [y_t, pi_t]; x is (T,6) in the
    ordering documented at module top. Row 0 corresponds to the 5th input
    quarter (the first estimation-sample quarter).
    """
    y = np.asarray(y, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    if not (len(y) == len(pi) == len(r)):
        raise ValueError(
            f"y, pi, r must have equal length; got {len(y)}, {len(pi)}, {len(r)}."
        )
    if len(y) < 5:
        raise ValueError(
            f"Need at least 5 quarters (4 pre-sample lags + 1 observation), got {len(y)}."
        )
    T = len(y) - 4
    s = slice(4, 4 + T)  # estimation-sample rows in the full arrays

    yobs = np.column_stack([y[s], pi[s]])
    x = np.column_stack(
        [
            y[3 : 3 + T],
            y[2 : 2 + T],
            r[3 : 3 + T],
            r[2 : 2 + T],
            pi[3 : 3 + T],
            (pi[2 : 2 + T] + pi[1 : 1 + T] + pi[0:T]) / 3.0,
        ]
    )
    return yobs, x


def default_initial_state(y_first: float) -> tuple[np.ndarray, np.ndarray]:
    """The spec §1.6 initial-state prior as an explicit (xi_00, P_00) pair
    (spec §2.2: pass mean/cov explicitly, no ad-hoc diffuse hacks).

    y*_0 ~ N(y_first, 2^2) "diffuse-ish"; g_0 ~ N(3, 1^2) annualized;
    z_0 ~ N(0, 1^2). The state's lag slots (y*_{-1}, y*_{-2}, g_{-1},
    z_{-1}) get the same marginal prior, treated as a priori independent --
    a pragmatic, documented simplification (the KF forgets the initial
    cross-correlations quickly; G5a, where initialization precision
    matters, uses HLW's own exact xi.00/P.00 instead of this default).
    """
    p = INITIAL_STATE_PRIOR
    xi00 = np.array(
        [y_first, y_first, y_first, p["g_mean"], p["g_mean"], p["z_mean"], p["z_mean"]]
    )
    P00 = np.diag(
        [
            p["ystar_sd"] ** 2,
            p["ystar_sd"] ** 2,
            p["ystar_sd"] ** 2,
            p["g_sd"] ** 2,
            p["g_sd"] ** 2,
            p["z_sd"] ** 2,
            p["z_sd"] ** 2,
        ]
    ).astype(np.float64)
    return xi00, P00


# ---------------------------------------------------------------------------
# 2. Kalman filter log-likelihood (mirror of kalman_loglik_tv.stan)
# ---------------------------------------------------------------------------

LOG_2PI = np.log(2.0 * np.pi)


def _as_R_path(R: np.ndarray, T: int) -> np.ndarray:
    """Normalize a measurement covariance argument to the (T, m, m) path
    ``_kf_core`` consumes: a single (m, m) matrix (the constant case) is
    tiled to T copies -- the exact analogue of the Stan side's rep_array
    overload -- and a (T, m, m) path passes through with its length
    validated."""
    R = np.asarray(R, dtype=np.float64)
    if R.ndim == 2:
        return np.ascontiguousarray(np.broadcast_to(R, (T, R.shape[0], R.shape[1])))
    if R.ndim == 3:
        if R.shape[0] != T:
            raise ValueError(
                f"Time-varying R has {R.shape[0]} entries but yobs has T={T} rows."
            )
        return np.ascontiguousarray(R)
    raise ValueError(f"R must be (m, m) or (T, m, m); got shape {R.shape}.")


def _as_Q_path(Q: np.ndarray, T: int) -> np.ndarray:
    """Normalize a state-innovation covariance argument to the (T, n, n)
    path ``_kf_core`` consumes (S6 Q_t generalization, the exact analogue
    of :func:`_as_R_path`): a single (n, n) matrix (the constant case) is
    tiled to T copies -- mirroring the Stan side's ``rep_array`` overload
    -- and a (T, n, n) path passes through with its length validated."""
    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim == 2:
        return np.ascontiguousarray(np.broadcast_to(Q, (T, Q.shape[0], Q.shape[1])))
    if Q.ndim == 3:
        if Q.shape[0] != T:
            raise ValueError(
                f"Time-varying Q has {Q.shape[0]} entries but yobs has T={T} rows."
            )
        return np.ascontiguousarray(Q)
    raise ValueError(f"Q must be (n, n) or (T, n, n); got shape {Q.shape}.")


@njit(cache=True)
def _kf_core(yobs, x, F, Q, A, Z, R, xi00, P00, want_states):  # pragma: no cover -- numba
    """Shared filter recursion. Returns (loglik, xi_pred, P_pred, xi_filt,
    P_filt); the state arrays are only populated when ``want_states`` is
    True (the loglik-only path skips the copies, not the math, so both
    paths produce bit-identical loglik values).

    ``R`` is the (T, m, m) measurement-covariance PATH (S3 generalization,
    DECISIONS.md 2026-08-31) and ``Q`` the (T, n, n) state-innovation
    covariance PATH (S6 generalization, the same playbook); the constant
    cases arrive as T copies, built by the public wrappers, so the
    constant-Q arithmetic (``F P F' + Q[t]`` with every ``Q[t]`` the same
    matrix) is bit-identical to the pre-S6 ``F P F' + Q`` -- pinned by
    tests/test_g1_mirror.py. Same operation order as the Stan mirror and
    HLW's kalman.log.likelihood.R: predict from (xi_{0|0}, P_{0|0}), then
    update, for t = 1..T. Cholesky of the innovation covariance for both
    the quadratic form and the log-determinant; every covariance explicitly
    re-symmetrized.
    """
    T = yobs.shape[0]
    m = yobs.shape[1]
    n = F.shape[0]

    xi_pred = np.zeros((T, n))
    P_pred = np.zeros((T, n, n))
    xi_filt = np.zeros((T, n))
    P_filt = np.zeros((T, n, n))

    xi_tt = xi00.copy()
    P_tt = P00.copy()
    loglik = 0.0

    for t in range(T):
        xi_tp = F @ xi_tt
        P_tp = F @ P_tt @ F.T + Q[t]
        P_tp = 0.5 * (P_tp + P_tp.T)

        err = yobs[t] - A.T @ x[t] - Z @ xi_tp
        S = Z @ P_tp @ Z.T + R[t]
        S = 0.5 * (S + S.T)
        L = np.linalg.cholesky(S)

        # log det S = 2 * sum(log(diag(L)));  quadratic form via L^-1 err.
        logdet = 0.0
        for i in range(m):
            logdet += np.log(L[i, i])
        logdet *= 2.0
        u = np.linalg.solve(L, err)  # lower-triangular solve
        quad = u @ u
        loglik += -0.5 * (m * LOG_2PI + logdet + quad)

        # Gain K = P Z' S^-1 via the Cholesky factor.
        ZP = Z @ P_tp  # (m, n)
        K = np.linalg.solve(L.T, np.linalg.solve(L, ZP)).T  # (n, m)
        xi_tt = xi_tp + K @ err
        P_tt = P_tp - K @ ZP
        P_tt = 0.5 * (P_tt + P_tt.T)

        if want_states:
            xi_pred[t] = xi_tp
            P_pred[t] = P_tp
            xi_filt[t] = xi_tt
            P_filt[t] = P_tt

    return loglik, xi_pred, P_pred, xi_filt, P_filt


def kalman_loglik(
    yobs: np.ndarray,
    x: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    A: np.ndarray,
    Z: np.ndarray,
    R: np.ndarray,
    xi00: np.ndarray,
    P00: np.ndarray,
) -> float:
    """KF log-likelihood of ``yobs`` (T,m) given exogenous ``x`` (T,k) and
    the state-space (F, Q, A, Z, R) with explicit initial state
    (xi00 = xi_{0|0}, P00 = P_{0|0}).

    ``R`` may be a single (m, m) matrix (constant measurement covariance)
    or a (T, m, m) path (the S3 time-varying generalization, DECISIONS.md
    2026-08-31 -- for lw_sv with SV, R_t = diag(exp(h_IS,t), exp(h_PC,t))).
    ``Q`` likewise may be a single (n, n) matrix or a (T, n, n) path (the
    S6 generalization -- for UCSV, Q_t = [exp(h_eta,t)], the trend-shock
    SV). lw_sv passes a constant Q: in the marginalized LW form its SV
    shocks are measurement errors, so time variation enters through R_t.
    """
    yobs = np.ascontiguousarray(yobs, dtype=np.float64)
    ll, *_ = _kf_core(
        yobs,
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(F, dtype=np.float64),
        _as_Q_path(Q, yobs.shape[0]),
        np.ascontiguousarray(A, dtype=np.float64),
        np.ascontiguousarray(Z, dtype=np.float64),
        _as_R_path(R, yobs.shape[0]),
        np.ascontiguousarray(xi00, dtype=np.float64),
        np.ascontiguousarray(P00, dtype=np.float64),
        False,
    )
    return float(ll)


def lw_kalman_loglik(params: dict, y: np.ndarray, pi: np.ndarray, r: np.ndarray,
                     xi00: np.ndarray | None = None, P00: np.ndarray | None = None,
                     c: float = 1.0) -> float:
    """Convenience wrapper: LW matrices + regressors + KF log-likelihood in
    one call. ``y``/``pi``/``r`` include the 4 pre-sample lag quarters.
    Default initialization is ``default_initial_state(y[4])`` (the first
    estimation-sample observation anchors y*_0)."""
    yobs, x = build_lw_regressors(y, pi, r)
    F, Q, A, Z, R = build_lw_matrices(params, c=c)
    if xi00 is None or P00 is None:
        xi00_d, P00_d = default_initial_state(float(y[4]))
        xi00 = xi00_d if xi00 is None else xi00
        P00 = P00_d if P00 is None else P00
    return kalman_loglik(yobs, x, F, Q, A, Z, R, xi00, P00)


# ---------------------------------------------------------------------------
# 3. Stochastic-volatility helpers (mirror of
#    stan/functions/sv_rw_noncentered.stan; held together by the G1
#    harness's SV-path comparison). UNITS (spec §1.5, pinned by
#    tests/test_units_conventions.py): h is log-VARIANCE -- exp(h) is the
#    variance entering the measurement covariance, exp(h/2) is the sd.
# ---------------------------------------------------------------------------


def sv_rw_noncentered(h0: float, sigma_h: float, nu: np.ndarray) -> np.ndarray:
    """Non-centered SV random-walk path: h_t = h_{t-1} + sigma_h * nu_t for
    t = 1..T from the realized initial log-variance ``h0`` (the caller
    builds it as mu_h0 + sd * h0_raw, per spec §1.5's h_0 ~ N(mu_h0, 1))
    and standard-normal innovations ``nu``. Observation t uses h[t]; h_0
    itself is the pre-sample initial condition."""
    return h0 + sigma_h * np.cumsum(np.asarray(nu, dtype=np.float64))


def sv_diag_variance_path(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Time-varying diagonal measurement covariance from two log-variance
    paths: R_t = diag(exp(h1_t), exp(h2_t)) -- exp(h) because h is
    log-VARIANCE. For lw_sv, h1 = IS, h2 = PC, matching the observation
    order [y, pi]. Returns (T, 2, 2), ready for :func:`kalman_loglik`."""
    h1 = np.asarray(h1, dtype=np.float64)
    h2 = np.asarray(h2, dtype=np.float64)
    if h1.shape != h2.shape or h1.ndim != 1:
        raise ValueError(
            f"h1 and h2 must be equal-length 1-d arrays; got {h1.shape} and {h2.shape}."
        )
    R = np.zeros((h1.shape[0], 2, 2))
    R[:, 0, 0] = np.exp(h1)
    R[:, 1, 1] = np.exp(h2)
    return R


def sv_scalar_variance_path(h: np.ndarray) -> np.ndarray:
    """Time-varying 1x1 covariance path from ONE log-variance path:
    C_t = [exp(h_t)] -- exp(h) because h is log-VARIANCE (spec §1.5's
    convention, shared by every family). The (T, 1, 1) result feeds either
    argument of :func:`kalman_loglik` (UCSV: Q_t from h_eta, R_t from
    h_eps). Mirrors stan/functions/sv_scalar_variance_path.stan."""
    h = np.asarray(h, dtype=np.float64)
    if h.ndim != 1:
        raise ValueError(f"h must be a 1-d log-variance path; got shape {h.shape}.")
    C = np.zeros((h.shape[0], 1, 1))
    C[:, 0, 0] = np.exp(h)
    return C


# ---------------------------------------------------------------------------
# 4. Fixed-interval (RTS) smoother -- G5a's engine. NOT the DK simulation
#    smoother (S4 scope).
# ---------------------------------------------------------------------------


@njit(cache=True)
def _rts_smooth(xi_pred, P_pred, xi_filt, P_filt, F):  # pragma: no cover -- numba
    T, n = xi_filt.shape
    xi_sm = np.zeros((T, n))
    P_sm = np.zeros((T, n, n))
    xi_sm[T - 1] = xi_filt[T - 1]
    P_sm[T - 1] = P_filt[T - 1]
    for t in range(T - 2, -1, -1):
        # J_t = P_{t|t} F' P_{t+1|t}^-1 via the Cholesky factor of the
        # (symmetric PD) predicted covariance -- same numerics doctrine as
        # the filter: factor once, triangular-shaped solves, no raw inverse.
        L = np.linalg.cholesky(P_pred[t + 1])
        J = np.linalg.solve(L.T, np.linalg.solve(L, F @ P_filt[t])).T
        xi_sm[t] = xi_filt[t] + J @ (xi_sm[t + 1] - xi_pred[t + 1])
        P_sm[t] = P_filt[t] + J @ (P_sm[t + 1] - P_pred[t + 1]) @ J.T
        P_sm[t] = 0.5 * (P_sm[t] + P_sm[t].T)
    return xi_sm, P_sm


def kalman_smoother(
    yobs: np.ndarray,
    x: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    A: np.ndarray,
    Z: np.ndarray,
    R: np.ndarray,
    xi00: np.ndarray,
    P00: np.ndarray,
) -> dict:
    """Run the filter + RTS smoother. Returns a dict with 'loglik',
    'xi_filt' (T,n), 'P_filt', 'xi_pred', 'P_pred', 'xi_smooth' (T,n),
    'P_smooth' -- the filtered/one-sided and smoothed/two-sided state
    paths G5a compares against the HLW oracle. ``R`` is (m, m) constant or
    a (T, m, m) path, and ``Q`` (n, n) constant or a (T, n, n) path, as in
    :func:`kalman_loglik`."""
    yobs = np.ascontiguousarray(yobs, dtype=np.float64)
    T = yobs.shape[0]
    x_c = np.ascontiguousarray(x, dtype=np.float64)
    F_c = np.ascontiguousarray(F, dtype=np.float64)
    A_c = np.ascontiguousarray(A, dtype=np.float64)
    Z_c = np.ascontiguousarray(Z, dtype=np.float64)
    tail = [
        np.ascontiguousarray(a, dtype=np.float64)
        for a in (xi00, P00)
    ]
    Q_path = _as_Q_path(Q, T)
    R_path = _as_R_path(R, T)
    ll, xi_pred, P_pred, xi_filt, P_filt = _kf_core(
        yobs, x_c, F_c, Q_path, A_c, Z_c, R_path, *tail, True
    )
    xi_sm, P_sm = _rts_smooth(xi_pred, P_pred, xi_filt, P_filt, F_c)
    return {
        "loglik": float(ll),
        "xi_pred": xi_pred,
        "P_pred": P_pred,
        "xi_filt": xi_filt,
        "P_filt": P_filt,
        "xi_smooth": xi_sm,
        "P_smooth": P_sm,
    }


# ---------------------------------------------------------------------------
# 5. Durbin-Koopman simulation smoother (spec §2.4, plans/S4-plan.md "open
#    question 1", user-confirmed 2026-08-31: the literal two-pass DK
#    algorithm, not FFBS). Per posterior draw, at that draw's own (F, Q, A,
#    Z, R_t) and the REAL yobs/x, produces a joint draw of the full state
#    path from its exact smoothing distribution p(xi_1:T | yobs), plus the
#    five structural shocks it implies (spec §2.4 point 2) -- feeding the
#    trend-cycle objects, historical decomposition, and fan-chart seeds
#    downstream (results_lw.py, S4 next step).
#
#    GATE: this section's correctness is gated by tests/test_smoother_sim.py
#    (Monte Carlo mean/variance convergence to kalman_smoother's xi_smooth/
#    P_smooth, plus a deterministic zero-plus-noise identity check standing
#    in for a Stan-side G1 mirror, since §2.4 is Python-only) -- do not
#    build results_lw.py against this section's output until that file is
#    green.
#
#    HANDOFF.md's S4 warning applies here as much as to results_lw.py: for
#    an SV draw, build R as sv_diag_variance_path(h_is, h_pc) from that
#    draw's OWN saved h_is/h_pc transformed parameters -- never re-derive h
#    from nu.
# ---------------------------------------------------------------------------


def _psd_sqrt(M: np.ndarray) -> np.ndarray:
    """A symmetric matrix square root of a symmetric positive-semidefinite
    matrix M (or a batch of them: any leading shape, last two dims square --
    ``np.linalg.eigh`` batches automatically), robust to exact rank
    deficiency.

    Needed because Q is NOT full rank: per ``build_lw_matrices``'s
    construction, only Q[0,0], Q[0,3]/Q[3,0], Q[3,3], Q[5,5] are nonzero --
    rows/cols 1, 2, 4, 6 (the deterministic lag-copy states, HANDOFF.md's
    state-slot warning) are identically zero, making Q rank 3 of 7. A plain
    Cholesky factor (``np.linalg.cholesky``, used everywhere else in this
    module) raises ``LinAlgError: Matrix is not positive definite`` on a
    rank-deficient input even though Q is a perfectly good covariance to
    simulate from -- eigendecomposition sidesteps that: ``M = V diag(w)
    V'``, any tiny/negative eigenvalues (floating-point noise around exact
    zeros) clipped to 0 before the sqrt, so ``L @ L.T == M`` (up to
    floating-point round-off) for any PSD input, full-rank or not.

    Re-symmetrizes ``M`` first (``0.5 * (M + M.T)``, batched over any
    leading dims) -- the same explicit-symmetrization discipline
    ``_kf_core``/``_rts_smooth`` apply to every propagated covariance
    elsewhere in this module (numerics-reviewer, S4). ``np.linalg.eigh``
    would otherwise silently read only ``M``'s lower triangle, masking a
    genuinely asymmetric input rather than catching or averaging it.
    """
    M = 0.5 * (M + np.swapaxes(M, -1, -2))
    w, v = np.linalg.eigh(M)
    w = np.clip(w, 0.0, None)
    sqrt_w = np.sqrt(w)
    return v * sqrt_w[..., None, :]  # V @ diag(sqrt(w)), batched


def _simulate_plus_path(
    F: np.ndarray,
    Q: np.ndarray,
    A: np.ndarray,
    Z: np.ndarray,
    R_path: np.ndarray,
    x: np.ndarray,
    xi00: np.ndarray,
    P00: np.ndarray,
    rng: np.random.Generator,
    zero_noise: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one draw from the *unconditional* linear-Gaussian model at
    the given system matrices (Durbin & Koopman 2002's "plus" path, step 1
    of plans/S4-plan.md's algorithm): ``xi+_0 ~ N(xi00, P00)``,
    ``xi+_t = F @ xi+_{t-1} + w+_t`` (``w+_t ~ N(0, Q_t)``),
    ``y+_t = A'x_t + Z @ xi+_t + e+_t`` (``e+_t ~ N(0, R_t)``) for
    t = 1..T. ``Q`` is a constant (n, n) matrix or a (T, n, n) path (S6).
    Reuses the real ``x``/``A``/``Z``/``R_path``; only the noise
    draws and the resulting state/obs path are simulated. Returns
    ``(xi_plus, y_plus)``, both length T, indexed exactly like
    ``kalman_smoother``'s ``xi_pred``/``xi_filt`` (row t = period t+1 in
    the ``xi+_1, ..., xi+_T`` notation above).

    ``zero_noise=True`` short-circuits every noise draw to exactly zero
    (``xi+_0 = xi00`` exactly, no ``w+``/``e+`` at any t) -- the
    deterministic "plus" system used by the zero-plus-noise identity check
    in tests/test_smoother_sim.py. It exercises the same code path as the
    real stochastic draw (same matrix multiplies, same loop), just with the
    RNG draws replaced by zeros, so that test is a real check of the DK
    combination step downstream, not a re-derivation of the same formula.

    Ordinary numpy-random Python function, not ``@njit``: numba's RNG story
    is awkward for multivariate-normal draws (plans/S4-plan.md's resolved
    open question 1), and the per-draw cost here is dominated by the two
    ``kalman_smoother`` passes this feeds, not this O(T) simulation loop --
    matching numba style would fight the RNG for no real benefit.
    """
    T = x.shape[0]
    n = F.shape[0]
    m = A.shape[1]

    xi_plus = np.zeros((T, n))
    y_plus = np.zeros((T, m))

    if zero_noise:
        xi_prev = xi00.copy()
    else:
        sqrt_P00 = _psd_sqrt(P00)
        xi_prev = xi00 + sqrt_P00 @ rng.standard_normal(n)

    # Q is (n, n) constant -- one factorization, the pre-S6 code path
    # exactly -- or a (T, n, n) path (S6 Q_t generalization: per-period
    # factors, batched by eigh like R_path below).
    if zero_noise:
        sqrt_Q = sqrt_Q_path = None
    elif Q.ndim == 2:
        sqrt_Q, sqrt_Q_path = _psd_sqrt(Q), None
    else:
        sqrt_Q, sqrt_Q_path = None, _psd_sqrt(Q)
    # R_path may be time-varying (SV); eigh batches over the leading T
    # dimension in one call rather than T separate decompositions.
    sqrt_R_path = None if zero_noise else _psd_sqrt(R_path)

    for t in range(T):
        if zero_noise:
            w_plus = np.zeros(n)
        elif sqrt_Q_path is None:
            w_plus = sqrt_Q @ rng.standard_normal(n)
        else:
            w_plus = sqrt_Q_path[t] @ rng.standard_normal(n)
        xi_t = F @ xi_prev + w_plus
        e_plus = np.zeros(m) if zero_noise else sqrt_R_path[t] @ rng.standard_normal(m)
        y_plus[t] = A.T @ x[t] + Z @ xi_t + e_plus
        xi_plus[t] = xi_t
        xi_prev = xi_t

    return xi_plus, y_plus


def recover_shocks(
    xi_draw: np.ndarray,
    xi00: np.ndarray,
    yobs: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    Z: np.ndarray,
    F: np.ndarray,
    meta,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """FAMILY-GENERIC structural-shock recovery (S6 WP2): invert a drawn
    state path into the named STATE shocks via the family's declared
    loadings (``StateSpaceMeta.recovery_order`` -- an exact triangular
    solve of ``w = B eps``, no least squares) and the named MEASUREMENT
    shocks from the per-period residual (diagonal R: the residual's row j
    IS measurement shock j, in ``meta.measurement_shocks`` order). Same
    process-noise/residual algebra as :func:`_recover_structural_shocks`
    (whose lw_sv-specific slot arithmetic this generalizes; for lw_sv the
    two are bit-identical, pinned by tests/test_smoother_sim.py) -- see
    that function's boundary note about w_0's lag-copy rows.
    Returns ``({state_shock: (T,)}, {measurement_shock: (T,)})``."""
    xi_prev = np.vstack([xi00[None, :], xi_draw[:-1]])
    w = xi_draw - xi_prev @ F.T
    eps: dict[str, np.ndarray] = {}
    for shock, label, coef, others in meta.recovery_order():
        acc = w[:, meta.slot(*label)]
        for other, other_coef in others:
            acc = acc - other_coef * eps[other]
        eps[shock] = acc / coef
    state_shocks = {s: eps[s] for s in meta.state_shocks}
    e = yobs - x @ A - xi_draw @ Z.T
    meas_shocks = {name: e[:, j] for j, name in enumerate(meta.measurement_shocks)}
    return state_shocks, meas_shocks


@dataclass
class SimSmootherDraw:
    """One Durbin-Koopman simulation-smoother draw: the joint state path
    ``xi_draw`` (T, 7), drawn from its exact smoothing distribution
    p(xi_1:T | yobs) at one posterior draw's system matrices, plus the five
    structural shocks it implies (spec §2.4 point 2), recovered
    algebraically from consecutive drawn states (process noise) and the
    per-period measurement residual (measurement error) -- no extra
    randomness beyond the plus-path draw itself.

    Shock naming matches spec §1.4's structural list: ``eps_ystar``
    (potential-output level shock), ``eps_g`` (trend-growth shock, ANNUALIZED
    per this module's units convention), ``eps_z`` (other r*/headwinds
    shock), ``eps_is`` (IS/demand measurement shock), ``eps_pc``
    (Phillips-curve/supply measurement shock). Each is length T, aligned
    with ``xi_draw``'s rows (period t's shock realizes going INTO state t,
    i.e. the same period-t indexing ``kalman_smoother`` uses throughout).
    """

    xi_draw: np.ndarray
    eps_ystar: np.ndarray | None = None
    eps_g: np.ndarray | None = None
    eps_z: np.ndarray | None = None
    eps_is: np.ndarray | None = None
    eps_pc: np.ndarray | None = None
    #: S6: the same shocks keyed by the family's declared names
    #: (``state_shocks[name]`` / ``meas_shocks[name]``), populated for every
    #: family; the five lw_sv attributes above stay for existing callers.
    state_shocks: dict[str, np.ndarray] | None = None
    meas_shocks: dict[str, np.ndarray] | None = None


def _recover_structural_shocks(
    xi_draw: np.ndarray,
    xi00: np.ndarray,
    yobs: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    Z: np.ndarray,
    F: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Algebraically invert a drawn state path ``xi_draw`` (T, 7) into the
    five structural shocks, given ``build_lw_matrices``'s EXACT Q
    construction (Q[0,0] = s_ystar^2 + s_g^2/16, Q[0,3] = Q[3,0] = s_g^2/4,
    Q[3,3] = s_g^2, Q[5,5] = s_z^2, all else zero -- so the y* process noise
    row is exactly ``eps_ystar,t + eps_g,t/4`` and rows 1, 2, 4, 6 are the
    deterministic lag-copy states with zero process noise, per this module's
    top-of-file state-space docstring and HANDOFF.md's state-slot warning).

    Process noise ``w_t = xi_draw[t] - F @ xi_draw[t-1]`` for t = 1..T-1 and
    ``w_0 = xi_draw[0] - F @ xi00`` for t = 0 (xi00 is the real period-0
    state, NOT the plus-path draw -- the drawn path is conditioned on the
    real initial state). Measurement error
    ``e_t = yobs[t] - A'x[t] - Z @ xi_draw[t]`` for every t; R is diagonal
    (no IS/PC correlation in this model), so ``e_t`` decomposes directly
    into ``(eps_is,t, eps_pc,t)`` with no further work.

    Vectorized over t (no explicit T-loop): ``xi_prev @ F.T`` row t equals
    ``F @ xi_prev[t]`` for every row (a standard vec-identity), so the
    process-noise recovery is one matrix multiply, not a per-period loop.

    BOUNDARY NOTE (measured, tests/test_smoother_sim.py pins it): rows 1, 2,
    4, 6 of ``w`` (the deterministic lag-copy states, e.g. w[t][1] should
    equal ``xi_draw[t][1] - xi_draw[t-1][0]``, which the model forces to be
    EXACTLY zero for t = 1..T-1, confirmed to machine precision) are NOT
    close to zero at t = 0 specifically. That is expected, not a bug: t=0's
    "previous state" is the raw prior ``xi00`` (its mean, per
    :func:`default_initial_state`'s pragmatic independent-lag-slots
    simplification), not a smoothed/updated estimate -- ``xi_draw[0][1]`` is
    the FULL-SAMPLE smoothed belief about y*_{-1}, which can and does differ
    from its PRE-DATA prior mean ``xi00[0]``. Downstream code should not
    treat w_0's rows 1, 2, 4, 6 as diagnostic of a bug.
    """
    xi_prev = np.vstack([xi00[None, :], xi_draw[:-1]])
    w = xi_draw - xi_prev @ F.T  # (T, 7)

    eps_g = w[:, 3]
    eps_z = w[:, 5]
    eps_ystar = w[:, 0] - eps_g / 4.0

    e = yobs - x @ A - xi_draw @ Z.T  # (T, 2); A'x_t == x_t @ A row-wise
    eps_is = e[:, 0]
    eps_pc = e[:, 1]

    return eps_ystar, eps_g, eps_z, eps_is, eps_pc


def simulate_smoother_draw(
    yobs: np.ndarray,
    x: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    A: np.ndarray,
    Z: np.ndarray,
    R: np.ndarray,
    xi00: np.ndarray,
    P00: np.ndarray,
    rng: np.random.Generator,
    xi_smooth: np.ndarray | None = None,
    *,
    zero_noise: bool = False,
    meta=None,
) -> SimSmootherDraw:
    """One Durbin & Koopman (2002) simulation-smoother draw
    (plans/S4-plan.md's resolved open question 1, the literal two-pass DK
    algorithm): a joint draw of the full state path from its exact
    smoothing distribution p(xi_1:T | yobs), at this posterior draw's own
    system matrices ``(F, Q, A, Z, R)`` and the REAL data ``(yobs, x)``,
    plus the five structural shocks it implies.

    Algorithm (spec §2.4, plans/S4-plan.md):
      1. Simulate an unconditional "plus" path ``(xi+, y+)`` at the same
         system matrices (:func:`_simulate_plus_path`).
      2. Smooth the REAL data -> ``xi_smooth`` (via :func:`kalman_smoother`,
         unless the caller already has it -- see ``xi_smooth`` below).
      3. Smooth the SIMULATED ``y+`` -> ``xi+_smooth`` (another
         :func:`kalman_smoother` call, same system matrices).
      4. ``xi_draw = xi_smooth - xi+_smooth + xi+`` -- the DK identity: the
         smoothing error ``xi_smooth - xi+_smooth`` has exactly the
         conditional distribution needed to add back onto the known plus
         path, since both smooths share the same linear smoothing operator
         and their Gaussian innovations cancel it exactly.

    ``R`` accepts the same constant-(m,m)-or-(T,m,m)-path forms as
    :func:`kalman_loglik`/:func:`kalman_smoother` (normalized via
    :func:`_as_R_path`), and ``Q`` a constant (n,n) matrix or a (T,n,n)
    path (S6: a family with SV on a STATE shock, e.g. UCSV's trend shock,
    passes that draw's own exp(h) path) -- for an SV draw, pass
    ``sv_diag_variance_path(h_is, h_pc)`` built from THAT draw's own saved
    h_is/h_pc transformed parameters (HANDOFF.md's S4 warning: never
    re-derive h from nu).

    ``rng`` must be an ``np.random.Generator`` (e.g.
    ``np.random.default_rng(seed)``) -- never the legacy global numpy
    random state, so a caller drawing many samples at the same or different
    parameter points controls reproducibility explicitly.

    ``xi_smooth``: if the caller already has the real-data smooth (e.g.
    doing many draws at the same parameter point), pass it in to skip the
    redundant recompute; otherwise it is computed internally via
    :func:`kalman_smoother`. (Only ``xi_smooth`` is needed for step 4 above
    -- not ``P_smooth`` -- so there is no ``P_smooth`` parameter.)

    ``zero_noise``: testing seam only (see :func:`_simulate_plus_path`) --
    forces the plus path's noise to exactly zero, which by the DK identity
    collapses ``xi_draw`` to exactly ``xi_smooth`` (tests/test_smoother_sim.py's
    zero-plus-noise identity check, the "G1-style mirror" spec §2.4 calls
    for since there is no Stan-side smoother to mirror against).

    ``meta`` (S6): the family's :class:`StateSpaceMeta`; when given, the
    structural shocks are recovered generically from its declared loadings
    (:func:`recover_shocks`) and returned in ``state_shocks``/
    ``meas_shocks``; when ``None`` (the lw_sv default, every pre-S6 call
    site) the historical lw_sv recovery runs and ALSO fills the named
    dicts, so both interfaces are always available.

    Returns a :class:`SimSmootherDraw` with the drawn state path and the
    recovered structural shocks (spec §2.4 point 2), each length T.
    """
    yobs = np.ascontiguousarray(yobs, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.float64)
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    if Q.ndim == 3:
        Q = _as_Q_path(Q, yobs.shape[0])  # validates the path length
    A = np.ascontiguousarray(A, dtype=np.float64)
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    xi00 = np.ascontiguousarray(xi00, dtype=np.float64)
    P00 = np.ascontiguousarray(P00, dtype=np.float64)
    R_path = _as_R_path(R, yobs.shape[0])

    if xi_smooth is None:
        xi_smooth = kalman_smoother(yobs, x, F, Q, A, Z, R_path, xi00, P00)["xi_smooth"]
    xi_smooth = np.ascontiguousarray(xi_smooth, dtype=np.float64)

    xi_plus, y_plus = _simulate_plus_path(
        F, Q, A, Z, R_path, x, xi00, P00, rng, zero_noise=zero_noise
    )
    xi_plus_smooth = kalman_smoother(y_plus, x, F, Q, A, Z, R_path, xi00, P00)["xi_smooth"]

    xi_draw = xi_smooth - xi_plus_smooth + xi_plus

    if meta is not None:
        state_shocks, meas_shocks = recover_shocks(xi_draw, xi00, yobs, x, A, Z, F, meta)
        return SimSmootherDraw(
            xi_draw=xi_draw,
            eps_ystar=state_shocks.get("ystar"),
            eps_g=state_shocks.get("g"),
            eps_z=state_shocks.get("z"),
            eps_is=meas_shocks.get("is"),
            eps_pc=meas_shocks.get("pc"),
            state_shocks=state_shocks,
            meas_shocks=meas_shocks,
        )

    eps_ystar, eps_g, eps_z, eps_is, eps_pc = _recover_structural_shocks(
        xi_draw, xi00, yobs, x, A, Z, F
    )

    return SimSmootherDraw(
        xi_draw=xi_draw,
        eps_ystar=eps_ystar,
        eps_g=eps_g,
        eps_z=eps_z,
        eps_is=eps_is,
        eps_pc=eps_pc,
        state_shocks={"ystar": eps_ystar, "g": eps_g, "z": eps_z},
        meas_shocks={"is": eps_is, "pc": eps_pc},
    )
