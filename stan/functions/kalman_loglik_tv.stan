// kalman_loglik_tv.stan -- shared Kalman-filter log-likelihood
// (lw-sv-spec.md §2.2). Included into rendered programs via a Jinja
// include directive inside a functions block; contains function
// definitions only, no block wrapper.
//
// SCOPE (S3 generalization of R_t, decisions 2026-08-31; S6 generalization
// of Q_t, plans/S6-plan.md, DECISIONS.md 2026-09-04): the filter core takes
// a T-array of state-innovation covariances Q_t AND a T-array of
// measurement covariances R_t -- the full time-varying form spec §2.2
// contracts. lw_sv routes its SV through R_t (in the marginalized LW form
// the SV shocks are MEASUREMENT errors; Q is constant); UCSV's trend-shock
// SV enters the STATE innovation, which is what Q_t is for. Thin overloads
// below delegate the constant cases with rep_array so call sites read
// naturally -- they are wrappers, not second filter implementations. The
// array form is deliberately family-agnostic: callers build Q_t / R_t
// however their family defines them.
//
// Model (Hamilton/HLW notation, mirrored exactly by
// macrotoolkit.smoother._kf_core -- gate G1 holds the two to <1e-8):
//
//   xi_t   = F xi_{t-1} + w_t,        w_t ~ N(0, Q_t)     (state, dim n)
//   yobs_t = A' x_t + Z xi_t + e_t,   e_t ~ N(0, R_t)     (obs m, exog k)
//
// Initial state: explicit (xi00 = xi_{0|0}, P00 = P_{0|0}) -- spec §2.2
// forbids ad-hoc diffuse hacks; the caller supplies the prior mean/cov.
//
// Numerics (spec §2.2): Cholesky factorization of the innovation covariance
// for both the log-determinant and every solve (no inverse, no det); each
// covariance explicitly re-symmetrized after construction/update. Operation
// order matches the Python mirror step for step. The constant-Q overload's
// arithmetic (F P F' + Q[t] with every Q[t] the same matrix) is identical
// to the pre-S6 constant-Q filter's, so constant-Q likelihoods are
// unchanged to output resolution (pinned by tests/test_g1_mirror.py).

/**
 * Kalman-filter log-likelihood, time-varying state-innovation AND
 * measurement covariances.
 *
 * @param yobs T x m observations (row t = observation vector at t)
 * @param x    T x k exogenous regressors entering the measurement equation
 * @param F    n x n state transition
 * @param Q    array of T n x n state innovation covariances (Q[t] used at t)
 * @param A    k x m exogenous loading (measurement mean adds A' x_t)
 * @param Z    m x n state loading
 * @param R    array of T m x m measurement covariances (R[t] used at t)
 * @param xi00 n-vector initial state mean xi_{0|0}
 * @param P00  n x n initial state covariance P_{0|0}
 * @return total log-likelihood sum_t log p(yobs_t | yobs_{1:t-1})
 */
real kalman_loglik(matrix yobs, matrix x,
                   matrix F, array[] matrix Q, matrix A, matrix Z, array[] matrix R,
                   vector xi00, matrix P00) {
  int T = rows(yobs);
  int m = cols(yobs);
  int n = rows(F);
  vector[n] xi_tt = xi00;
  matrix[n, n] P_tt = P00;
  real ll = 0;

  if (size(R) != T) {
    reject("kalman_loglik: size(R) = ", size(R), " must equal T = ", T);
  }
  if (size(Q) != T) {
    reject("kalman_loglik: size(Q) = ", size(Q), " must equal T = ", T);
  }

  for (t in 1:T) {
    vector[n] xi_tp = F * xi_tt;
    matrix[n, n] P_tp = F * P_tt * F' + Q[t];
    P_tp = 0.5 * (P_tp + P_tp');

    vector[m] err = yobs[t]' - A' * x[t]' - Z * xi_tp;
    matrix[m, m] S = Z * P_tp * Z' + R[t];
    S = 0.5 * (S + S');
    matrix[m, m] L = cholesky_decompose(S);

    vector[m] u = mdivide_left_tri_low(L, err);
    ll += -0.5 * (m * log(2 * pi()) + 2 * sum(log(diagonal(L))) + dot_self(u));

    matrix[m, n] ZP = Z * P_tp;
    // K = P Z' S^-1 = (L^-1 ZP)' L^-1, using only triangular solves.
    matrix[n, m] K = mdivide_right_tri_low(mdivide_left_tri_low(L, ZP)', L);
    xi_tt = xi_tp + K * err;
    P_tt = P_tp - K * ZP;
    P_tt = 0.5 * (P_tt + P_tt');
  }
  return ll;
}

/**
 * Constant-Q, time-varying-R overload (lw_sv's SV variant): delegates to
 * the full filter with T copies of Q. A wrapper, not a second
 * implementation.
 */
real kalman_loglik(matrix yobs, matrix x,
                   matrix F, matrix Q, matrix A, matrix Z, array[] matrix R,
                   vector xi00, matrix P00) {
  return kalman_loglik(yobs, x, F, rep_array(Q, rows(yobs)), A, Z, R,
                       xi00, P00);
}

/**
 * Time-varying-Q, constant-R overload: delegates with T copies of R.
 */
real kalman_loglik(matrix yobs, matrix x,
                   matrix F, array[] matrix Q, matrix A, matrix Z, matrix R,
                   vector xi00, matrix P00) {
  return kalman_loglik(yobs, x, F, Q, A, Z, rep_array(R, rows(yobs)),
                       xi00, P00);
}

/**
 * Constant-covariance overload (lw_sv's no-SV variant): delegates to the
 * full filter with T copies of Q and of R. Exists so constant call sites
 * keep reading naturally; NOT a separate filter implementation.
 */
real kalman_loglik(matrix yobs, matrix x,
                   matrix F, matrix Q, matrix A, matrix Z, matrix R,
                   vector xi00, matrix P00) {
  return kalman_loglik(yobs, x, F, rep_array(Q, rows(yobs)), A, Z,
                       rep_array(R, rows(yobs)), xi00, P00);
}
