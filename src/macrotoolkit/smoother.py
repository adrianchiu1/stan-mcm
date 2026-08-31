"""Python mirror of the Stan Kalman-filter machinery for the LW model.

Scope (S2 + the S3 KF generalization, plans/S3-plan.md): the KF
log-likelihood (gate G1) with constant OR time-varying measurement
covariance R_t (in the marginalized LW form the SV shocks are measurement
errors, so SV time variation enters through R_t; Q stays constant -- see
HANDOFF.md's S3 warning), and the RTS fixed-interval smoother (gate G5a).
The Durbin-Koopman *simulation* smoother of spec §2.4 is S4 scope -- do
not add it here until S4.

Three layers, mirroring the Stan side exactly:

1. ``build_lw_matrices``      <-> ``stan/functions/ssm_matrices_lw.stan``
2. ``kalman_loglik``           <-> ``stan/functions/kalman_loglik_tv.stan``
3. ``kalman_smoother``         (Python only until S4's simulation smoother;
                                validated against the HLW oracle in G5a)

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


@njit(cache=True)
def _kf_core(yobs, x, F, Q, A, Z, R, xi00, P00, want_states):  # pragma: no cover -- numba
    """Shared filter recursion. Returns (loglik, xi_pred, P_pred, xi_filt,
    P_filt); the state arrays are only populated when ``want_states`` is
    True (the loglik-only path skips the copies, not the math, so both
    paths produce bit-identical loglik values).

    ``R`` is the (T, m, m) measurement-covariance PATH (S3 generalization,
    DECISIONS.md 2026-08-31; the constant case arrives as T copies, built
    by the public wrappers). Same operation order as the Stan mirror and
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
        P_tp = F @ P_tt @ F.T + Q
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
    Q stays constant: in the marginalized LW form the SV shocks are
    measurement errors, so time variation enters through R_t only.
    """
    yobs = np.ascontiguousarray(yobs, dtype=np.float64)
    ll, *_ = _kf_core(
        yobs,
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(F, dtype=np.float64),
        np.ascontiguousarray(Q, dtype=np.float64),
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
# 3. Fixed-interval (RTS) smoother -- G5a's engine. NOT the DK simulation
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
    a (T, m, m) path, as in :func:`kalman_loglik`."""
    yobs = np.ascontiguousarray(yobs, dtype=np.float64)
    args = [
        np.ascontiguousarray(a, dtype=np.float64)
        for a in (x, F, Q, A, Z)
    ]
    tail = [
        np.ascontiguousarray(a, dtype=np.float64)
        for a in (xi00, P00)
    ]
    R_path = _as_R_path(R, yobs.shape[0])
    ll, xi_pred, P_pred, xi_filt, P_filt = _kf_core(
        yobs, *args, R_path, *tail, True
    )
    xi_sm, P_sm = _rts_smooth(xi_pred, P_pred, xi_filt, P_filt, args[1])
    return {
        "loglik": float(ll),
        "xi_pred": xi_pred,
        "P_pred": P_pred,
        "xi_filt": xi_filt,
        "P_filt": P_filt,
        "xi_smooth": xi_sm,
        "P_smooth": P_sm,
    }
