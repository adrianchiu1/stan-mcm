// sv_rw_noncentered.stan -- stochastic-volatility path construction
// (lw-sv-spec.md §1.5, §2.2). Included into rendered programs via a Jinja
// include directive inside a functions block; function definitions only,
// no block wrapper.
//
// This file is included ONLY by SV-variant renders (non-empty
// `sv_shocks`): the no-SV render is pinned byte-for-byte
// (tests/test_render.py::test_no_sv_render_is_byte_stable), so SV helpers
// must live here, never in the includes the no-SV render shares.
//
// UNITS (spec §1.5, pinned by tests/test_units_conventions.py): h is
// log-VARIANCE. exp(h) is a variance -- what goes on the measurement
// covariance diagonal -- and exp(h/2) is the corresponding sd. State this
// once, enforce everywhere.
//
// Mirrored exactly by macrotoolkit.smoother.sv_rw_noncentered /
// sv_diag_variance_path (held together by the G1 harness's SV-path
// comparison).

/**
 * Non-centered SV random-walk path: h_t = h_{t-1} + sigma_h * nu_t for
 * t = 1..T, from the realized initial log-variance h_0 (the caller builds
 * it non-centered as mu_h0 + h0_raw with h0_raw ~ std_normal(), per spec
 * §1.5's h_0 ~ N(mu_h0, 1)) and the standard-normal innovation vector nu.
 * Observation t uses h[t]; h_0 itself is the pre-sample initial condition,
 * not an observation-period value.
 *
 * @param h0      realized initial log-variance h_0
 * @param sigma_h log-variance random-walk scale (>= 0)
 * @param nu      T standard-normal innovations (non-centered: sampled, not h)
 * @return vector[T] of log-variances h_1..h_T
 */
vector sv_rw_noncentered(real h0, real sigma_h, vector nu) {
  return h0 + sigma_h * cumulative_sum(nu);
}

/**
 * Time-varying diagonal measurement covariance from two log-variance
 * paths: R_t = diag(exp(h1_t), exp(h2_t)) -- exp(h) because h is
 * log-VARIANCE (spec §1.5). For lw_sv, h1 = IS, h2 = PC, matching the
 * observation order [y, pi].
 *
 * @param h1 T log-variances of the first measurement shock
 * @param h2 T log-variances of the second measurement shock
 * @return array of T 2x2 diagonal covariance matrices
 */
array[] matrix sv_diag_variance_path(vector h1, vector h2) {
  int T = num_elements(h1);
  array[T] matrix[2, 2] R;
  for (t in 1:T) {
    R[t] = diag_matrix([exp(h1[t]), exp(h2[t])]');
  }
  return R;
}
