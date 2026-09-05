// lw_sv.stan -- Laubach-Williams model, rendered from lw_sv.stan.j2.
//
// One template, two variants (spec §7's staged build: S2 -> S3 grew one
// file, not two divergent ones), selected by model.options.sv_shocks:
//
// - sv_shocks == []: the no-SV variant. Constant IS/PC shock scales
//   sigma_is/sigma_pc. This render is pinned byte-for-byte
//   (tests/test_render.py::test_no_sv_render_is_byte_stable).
// - sv_shocks == [is, pc] (the one validated non-empty combination):
//   non-centered stochastic volatility on both measurement shocks (spec
//   §1.5). sigma_is/sigma_pc are REPLACED by log-variance random walks
//   h_s,t = h_s,t-1 + sigma_h_s * nu_s,t with h_s,0 ~ N(mu_h0_s, sd^2),
//   mu_h0_s the data-side OLS anchor. h is log-VARIANCE (sd = exp(h/2));
//   the KF sees R_t = diag(exp(h_IS,t), exp(h_PC,t)) -- in the
//   marginalized LW form the SV shocks are MEASUREMENT errors, so time
//   variation enters R_t, not Q (HANDOFF.md's S3 warning).
//
// Rao-Blackwellized likelihood (spec §2.1): the linear states (y*, g, z and
// lags) are marginalized by the Kalman filter; Stan samples only the
// static parameters. Since S6 the filter value is the transformed
// parameter kf_loglik (added to target in the model block) so the exact
// rendered program exposes its KF log-likelihood to the automatic
// fit-time mirror check (S6 WP3) and saves it with every draw.
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
// SCOPE (S3 generalization of R_t, decisions 2026-08-31; S6 generalization
// of Q_t, plans/S6-plan.md, DECISIONS.md 2026-09-04; S9 generalization of
// Z_t, plans/S9-plan.md): the filter core takes a T-array of state-
// innovation covariances Q_t, a T-array of measurement loadings Z_t AND a
// T-array of measurement covariances R_t -- the full time-varying form
// spec §2.2 contracts. lw_sv routes its SV through R_t (in the
// marginalized LW form the SV shocks are MEASUREMENT errors; Q is
// constant); UCSV's trend-shock SV enters the STATE innovation, which is
// what Q_t is for; an authored model with a DATA-DEPENDENT loading (a
// state multiplied by a lagged observable or an exogenous series -- the
// TVP regression / TVP-AR / TVP-VAR, S9 E4) routes the data through Z_t.
// Thin overloads below delegate the constant cases with rep_array so call
// sites read naturally -- they are wrappers, not second filter
// implementations. The array form is deliberately family-agnostic:
// callers build Q_t / Z_t / R_t however their family defines them.
//
// Model (Hamilton/HLW notation, mirrored exactly by
// macrotoolkit.smoother._kf_core -- gate G1 holds the two to <1e-8):
//
//   xi_t   = F xi_{t-1} + w_t,          w_t ~ N(0, Q_t)     (state, dim n)
//   yobs_t = A' x_t + Z_t xi_t + e_t,   e_t ~ N(0, R_t)     (obs m, exog k)
//
// Initial state: explicit (xi00 = xi_{0|0}, P00 = P_{0|0}) -- spec §2.2
// forbids ad-hoc diffuse hacks; the caller supplies the prior mean/cov.
//
// Numerics (spec §2.2): Cholesky factorization of the innovation covariance
// for both the log-determinant and every solve (no inverse, no det); each
// covariance explicitly re-symmetrized after construction/update. Operation
// order matches the Python mirror step for step. The constant-Q overload's
// arithmetic (F P F' + Q[t] with every Q[t] the same matrix) is identical
// to the pre-S6 constant-Q filter's, and the constant-Z overloads'
// arithmetic (Z[t] P Z[t]' with every Z[t] the same matrix) is identical
// to the pre-S9 constant-Z filter's, so constant-case likelihoods are
// unchanged to output resolution (pinned by tests/test_g1_mirror.py:
// tests/fixtures/g1/pre_qt_stan_loglik.csv, pre_zt_stan_loglik.csv).

/**
 * Kalman-filter log-likelihood, time-varying state-innovation covariance,
 * measurement loadings AND measurement covariance (the core).
 *
 * @param yobs T x m observations (row t = observation vector at t)
 * @param x    T x k exogenous regressors entering the measurement equation
 * @param F    n x n state transition
 * @param Q    array of T n x n state innovation covariances (Q[t] used at t)
 * @param A    k x m exogenous loading (measurement mean adds A' x_t)
 * @param Z    array of T m x n state loadings (Z[t] used at t)
 * @param R    array of T m x m measurement covariances (R[t] used at t)
 * @param xi00 n-vector initial state mean xi_{0|0}
 * @param P00  n x n initial state covariance P_{0|0}
 * @return total log-likelihood sum_t log p(yobs_t | yobs_{1:t-1})
 */
real kalman_loglik(matrix yobs, matrix x,
                   matrix F, array[] matrix Q, matrix A, array[] matrix Z, array[] matrix R,
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
  if (size(Z) != T) {
    reject("kalman_loglik: size(Z) = ", size(Z), " must equal T = ", T);
  }

  for (t in 1:T) {
    vector[n] xi_tp = F * xi_tt;
    matrix[n, n] P_tp = F * P_tt * F' + Q[t];
    P_tp = 0.5 * (P_tp + P_tp');

    vector[m] err = yobs[t]' - A' * x[t]' - Z[t] * xi_tp;
    matrix[m, m] S = Z[t] * P_tp * Z[t]' + R[t];
    S = 0.5 * (S + S');
    matrix[m, m] L = cholesky_decompose(S);

    vector[m] u = mdivide_left_tri_low(L, err);
    ll += -0.5 * (m * log(2 * pi()) + 2 * sum(log(diagonal(L))) + dot_self(u));

    matrix[m, n] ZP = Z[t] * P_tp;
    // K = P Z' S^-1 = (L^-1 ZP)' L^-1, using only triangular solves.
    matrix[n, m] K = mdivide_right_tri_low(mdivide_left_tri_low(L, ZP)', L);
    xi_tt = xi_tp + K * err;
    P_tt = P_tp - K * ZP;
    P_tt = 0.5 * (P_tt + P_tt');
  }
  return ll;
}

// ---------------------------------------------------------------------------
// Constant-Z overloads (every pre-S9 call site): delegate to the core with
// T copies of Z. Wrappers, not second implementations.
// ---------------------------------------------------------------------------

/**
 * Time-varying Q and R, constant Z (the S6 core's signature).
 */
real kalman_loglik(matrix yobs, matrix x,
                   matrix F, array[] matrix Q, matrix A, matrix Z, array[] matrix R,
                   vector xi00, matrix P00) {
  return kalman_loglik(yobs, x, F, Q, A, rep_array(Z, rows(yobs)), R,
                       xi00, P00);
}

/**
 * Constant-Q, time-varying-R overload (lw_sv's SV variant): delegates to
 * the full filter with T copies of Q and of Z.
 */
real kalman_loglik(matrix yobs, matrix x,
                   matrix F, matrix Q, matrix A, matrix Z, array[] matrix R,
                   vector xi00, matrix P00) {
  return kalman_loglik(yobs, x, F, rep_array(Q, rows(yobs)), A,
                       rep_array(Z, rows(yobs)), R, xi00, P00);
}

/**
 * Time-varying-Q, constant-R overload: delegates with T copies of R and of Z.
 */
real kalman_loglik(matrix yobs, matrix x,
                   matrix F, array[] matrix Q, matrix A, matrix Z, matrix R,
                   vector xi00, matrix P00) {
  return kalman_loglik(yobs, x, F, Q, A, rep_array(Z, rows(yobs)),
                       rep_array(R, rows(yobs)), xi00, P00);
}

/**
 * Constant-covariance overload (lw_sv's no-SV variant): delegates to the
 * full filter with T copies of Q, of Z and of R. Exists so constant call
 * sites keep reading naturally; NOT a separate filter implementation.
 */
real kalman_loglik(matrix yobs, matrix x,
                   matrix F, matrix Q, matrix A, matrix Z, matrix R,
                   vector xi00, matrix P00) {
  return kalman_loglik(yobs, x, F, rep_array(Q, rows(yobs)), A,
                       rep_array(Z, rows(yobs)), rep_array(R, rows(yobs)),
                       xi00, P00);
}

// ---------------------------------------------------------------------------
// Time-varying-Z overloads (S9 E4 call sites with constant Q and/or R).
// ---------------------------------------------------------------------------

/**
 * Time-varying Z, constant Q and R (the TVP regression / TVP-VAR with
 * constant shock scales): delegates with T copies of Q and of R.
 */
real kalman_loglik(matrix yobs, matrix x,
                   matrix F, matrix Q, matrix A, array[] matrix Z, matrix R,
                   vector xi00, matrix P00) {
  return kalman_loglik(yobs, x, F, rep_array(Q, rows(yobs)), A, Z,
                       rep_array(R, rows(yobs)), xi00, P00);
}

/**
 * Time-varying Z and R, constant Q (the TVP-AR with measurement SV):
 * delegates with T copies of Q.
 */
real kalman_loglik(matrix yobs, matrix x,
                   matrix F, matrix Q, matrix A, array[] matrix Z, array[] matrix R,
                   vector xi00, matrix P00) {
  return kalman_loglik(yobs, x, F, rep_array(Q, rows(yobs)), A, Z, R,
                       xi00, P00);
}

/**
 * Time-varying Z and Q, constant R (a coefficient state with SV):
 * delegates with T copies of R.
 */
real kalman_loglik(matrix yobs, matrix x,
                   matrix F, array[] matrix Q, matrix A, array[] matrix Z, matrix R,
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

transformed parameters {
  // The Kalman-filter marginal log-likelihood (spec §2.1), kept as a
  // transformed parameter (S6 WP3): evaluated once per log-density
  // evaluation (no duplicate filter pass), added to target below, saved
  // with every draw, and read by the automatic fit-time Stan-vs-Python
  // mirror check at prior draws (fixed_param evaluation of this program).
  real kf_loglik = kalman_loglik(yobs, x,
                                 F,
                                 lw_Q(sigma_ystar, sigma_g, sigma_z),
                                 lw_A(a1, a2, a_r, b_pi, b_y),
                                 lw_Z(a1, a2, a_r, b_y, c),
                                 lw_R(sigma_is, sigma_pc),
                                 xi00, P00);
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

  // Marginal likelihood via the Kalman filter (the transformed parameter).
  target += kf_loglik;
}