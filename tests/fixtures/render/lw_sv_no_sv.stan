// lw_sv.stan -- Laubach-Williams model, rendered from lw_sv.stan.j2.
//
// S2 RENDER: the no-SV variant (model.options.sv_shocks == []). Constant
// IS/PC shock scales sigma_is/sigma_pc stand in for the SV paths; S3 adds
// the non-centered SV block to THIS template (spec §7's staged build: S2 ->
// S3 grow one file, not two divergent ones).
//
// Rao-Blackwellized likelihood (spec §2.1): the linear states (y*, g, z and
// lags) are marginalized by the Kalman filter in the model block; Stan
// samples only the static parameters.
//
// All family-specific structure is stamped at render time (spec §2.3): the
// priors below come from specs/schema/lw_sv.py DEFAULT_PRIORS merged with
// any per-run overrides in the spec's `priors:` block; c is stamped as a
// constant (1.0) because estimate_c is false in S2.
//
// The initial state (xi00, P00) arrives as data, built by
// macrotoolkit.smoother.default_initial_state from spec §1.6's
// initial-state priors (y*_0 anchored at the first observation) -- spec
// §2.2: the KF takes the initial mean/cov explicitly.

functions {
// ssm_matrices_lw.stan -- state-space matrix constructors for the LW form
// (lw-sv-spec.md §1.2-§1.4, §2.2). Included into rendered programs via a
// Jinja include directive inside a functions block; function definitions
// only, no block wrapper.
//
// Mirrored exactly by macrotoolkit.smoother.build_lw_matrices (gate G1 and
// the G5a matrix unit test hold the two sides together). See that module's
// docstring for the full derivation; the layout in brief:
//
//   state (n=7):  xi_t = [y*_t, y*_{t-1}, y*_{t-2},
//                         g_{t-1}, g_{t-2}, z_{t-1}, z_{t-2}]
//   obs   (m=2):  yobs_t = [y_t, pi_t]
//   exog  (k=6):  x_t = [y_{t-1}, y_{t-2}, r_{t-1}, r_{t-2},
//                        pi_{t-1}, (pi_{t-2}+pi_{t-3}+pi_{t-4})/3]
//
// UNITS (spec §1.1/§1.3, the conventions tests/test_units_conventions.py
// pins): g is ANNUALIZED everywhere. The potential-output transition uses
// g/4 (hence F[1,4] = 0.25 and the /16, /4 factors in Q); r* = c*g + z
// needs no 4x factor (hence the plain c * a_r/2 loadings in Z).

/** Transition matrix F (constant; no free parameters). */
matrix lw_F() {
  matrix[7, 7] F = rep_matrix(0, 7, 7);
  F[1, 1] = 1;
  F[1, 4] = 0.25;  // y*_t = y*_{t-1} + g_{t-1}/4 + shock (g annualized)
  F[2, 1] = 1;
  F[3, 2] = 1;
  F[4, 4] = 1;
  F[5, 4] = 1;
  F[6, 6] = 1;
  F[7, 6] = 1;
  return F;
}

/**
 * State innovation covariance Q. The y* transition's shock is
 * eps_ystar + eps_g/4 (the period's annualized g innovation enters the
 * quarterly y* step divided by 4), giving the off-diagonal term with the
 * g shock.
 */
matrix lw_Q(real sigma_ystar, real sigma_g, real sigma_z) {
  matrix[7, 7] Q = rep_matrix(0, 7, 7);
  Q[1, 1] = square(sigma_ystar) + square(sigma_g) / 16.0;
  Q[1, 4] = square(sigma_g) / 4.0;
  Q[4, 1] = Q[1, 4];
  Q[4, 4] = square(sigma_g);
  Q[6, 6] = square(sigma_z);
  return Q;
}

/** Exogenous loading A (k=6 x m=2): measurement mean adds A' x_t. */
matrix lw_A(real a1, real a2, real a_r, real b_pi, real b_y) {
  matrix[6, 2] A = rep_matrix(0, 6, 2);
  A[1, 1] = a1;        // y_{t-1} in the IS curve
  A[2, 1] = a2;        // y_{t-2}
  A[3, 1] = a_r / 2;   // r_{t-1}
  A[4, 1] = a_r / 2;   // r_{t-2}
  A[1, 2] = b_y;       // y_{t-1} in the Phillips curve
  A[5, 2] = b_pi;      // pi_{t-1}
  A[6, 2] = 1 - b_pi;  // mean of pi_{t-2..t-4} (sum-to-one restriction)
  return A;
}

/**
 * State loading Z (m=2 x n=7). Row 1 (output): gap_t = y_t - y*_t, with the
 * IS curve's lagged gaps and real-rate gaps expanded onto the lag states;
 * -(a_r/2) * r*_{t-j} with r* = c*g + z. Row 2 (inflation): -b_y * y*_{t-1}
 * completes b_y * gap_{t-1}.
 */
matrix lw_Z(real a1, real a2, real a_r, real b_y, real c) {
  matrix[2, 7] Z = rep_matrix(0, 2, 7);
  Z[1, 1] = 1;
  Z[1, 2] = -a1;
  Z[1, 3] = -a2;
  Z[1, 4] = -c * a_r / 2;
  Z[1, 5] = -c * a_r / 2;
  Z[1, 6] = -a_r / 2;
  Z[1, 7] = -a_r / 2;
  Z[2, 2] = -b_y;
  return Z;
}

/** Measurement covariance R: constant IS/PC shock scales (no-SV variant;
 *  S3's SV block replaces this with a time-varying R_t). */
matrix lw_R(real sigma_is, real sigma_pc) {
  matrix[2, 2] R = rep_matrix(0, 2, 2);
  R[1, 1] = square(sigma_is);
  R[2, 2] = square(sigma_pc);
  return R;
}// kalman_loglik_tv.stan -- shared Kalman-filter log-likelihood
// (lw-sv-spec.md §2.2). Included into rendered programs via a Jinja
// include directive inside a functions block; contains function
// definitions only, no block wrapper.
//
// SCOPE (S3 generalization, decisions 2026-08-31, DECISIONS.md): the filter
// takes a T-array of measurement covariances R_t -- the time-varying form
// spec §2.2 contracts, landed where it actually bites for lw_sv (SV makes
// the IS/PC shock scales, i.e. the MEASUREMENT covariance here, time-
// varying; Q stays constant). A thin constant-R overload below delegates
// with rep_array so no-SV call sites read naturally -- it is a wrapper,
// not a second filter implementation. The array form is deliberately
// family-agnostic: callers build R_t however their family defines it.
//
// Model (Hamilton/HLW notation, mirrored exactly by
// macrotoolkit.smoother._kf_core -- gate G1 holds the two to <1e-8):
//
//   xi_t   = F xi_{t-1} + w_t,        w_t ~ N(0, Q)       (state, dim n)
//   yobs_t = A' x_t + Z xi_t + e_t,   e_t ~ N(0, R_t)     (obs m, exog k)
//
// Initial state: explicit (xi00 = xi_{0|0}, P00 = P_{0|0}) -- spec §2.2
// forbids ad-hoc diffuse hacks; the caller supplies the prior mean/cov.
//
// Numerics (spec §2.2): Cholesky factorization of the innovation covariance
// for both the log-determinant and every solve (no inverse, no det); each
// covariance explicitly re-symmetrized after construction/update. Operation
// order matches the Python mirror step for step.

/**
 * Kalman-filter log-likelihood, time-varying measurement covariance.
 *
 * @param yobs T x m observations (row t = observation vector at t)
 * @param x    T x k exogenous regressors entering the measurement equation
 * @param F    n x n state transition
 * @param Q    n x n state innovation covariance
 * @param A    k x m exogenous loading (measurement mean adds A' x_t)
 * @param Z    m x n state loading
 * @param R    array of T m x m measurement covariances (R[t] used at t)
 * @param xi00 n-vector initial state mean xi_{0|0}
 * @param P00  n x n initial state covariance P_{0|0}
 * @return total log-likelihood sum_t log p(yobs_t | yobs_{1:t-1})
 */
real kalman_loglik(matrix yobs, matrix x,
                   matrix F, matrix Q, matrix A, matrix Z, array[] matrix R,
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

  for (t in 1:T) {
    vector[n] xi_tp = F * xi_tt;
    matrix[n, n] P_tp = F * P_tt * F' + Q;
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
 * Constant-measurement-covariance overload: delegates to the time-varying
 * filter with T copies of R. Exists so no-SV call sites keep reading
 * naturally; NOT a separate filter implementation.
 */
real kalman_loglik(matrix yobs, matrix x,
                   matrix F, matrix Q, matrix A, matrix Z, matrix R,
                   vector xi00, matrix P00) {
  return kalman_loglik(yobs, x, F, Q, A, Z, rep_array(R, rows(yobs)),
                       xi00, P00);
}}

data {
  int<lower=1> T;
  matrix[T, 2] yobs;        // [y_t (100*ln GDP), pi_t (annualized q/q %)]
  matrix[T, 6] x;           // [y_{t-1}, y_{t-2}, r_{t-1}, r_{t-2}, pi_{t-1}, mean(pi_{t-2..t-4})]
  vector[7] xi00;           // initial state mean xi_{0|0}
  matrix[7, 7] P00;         // initial state covariance P_{0|0}
}

transformed data {
  matrix[7, 7] F = lw_F();  // parameter-free; build once
  real c = 1.0;         // r* = c*g + z loading (spec §1.3; fixed, estimate_c: false)
}

parameters {
  real a1;                          // gap AR(1) coefficient
  real a2;                          // gap AR(2) coefficient
  real<upper=0> a_r;                // IS slope; sign identification via support
  real<lower=0, upper=1> b_pi;      // weight on first inflation lag
  real<lower=0> b_y;                // Phillips slope; sign identification via support
  real<lower=0> sigma_ystar;        // potential level shock scale
  real<lower=0> sigma_g;            // trend growth shock scale (annualized) -- pile-up control prior
  real<lower=0> sigma_z;            // z shock scale -- pile-up control prior
  real<lower=0> sigma_is;           // constant IS shock scale (no-SV variant)
  real<lower=0> sigma_pc;           // constant PC shock scale (no-SV variant)
}

model {
  // Priors (stamped from specs/schema/lw_sv.py defaults + per-run overrides).
  // Truncations are carried by the parameter constraints above; Stan
  // renormalizes automatically for fixed truncation bounds.
  a1 ~ normal(1.2, 0.3);
  a2 ~ normal(-0.4, 0.3);
  a_r ~ normal(-0.1, 0.05);
  b_pi ~ beta(8.0, 2.0);
  b_y ~ normal(0.15, 0.1);
  sigma_ystar ~ normal(0, 0.4);  // Half-N via <lower=0>
  sigma_g ~ normal(0, 0.03);          // Half-N via <lower=0>
  sigma_z ~ normal(0, 0.08);          // Half-N via <lower=0>
  sigma_is ~ normal(0, 1.0);        // Half-N via <lower=0>
  sigma_pc ~ normal(0, 1.0);        // Half-N via <lower=0>

  // Marginal likelihood via the Kalman filter.
  target += kalman_loglik(yobs, x,
                          F,
                          lw_Q(sigma_ystar, sigma_g, sigma_z),
                          lw_A(a1, a2, a_r, b_pi, b_y),
                          lw_Z(a1, a2, a_r, b_y, c),
                          lw_R(sigma_is, sigma_pc),
                          xi00, P00);
}