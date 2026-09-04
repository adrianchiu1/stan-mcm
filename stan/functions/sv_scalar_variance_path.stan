// sv_scalar_variance_path.stan -- ONE log-variance path -> a T-array of
// 1x1 covariance matrices (S6, family #2 UCSV). Included into rendered
// programs via a Jinja include directive inside a functions block;
// function definitions only, no block wrapper. Deliberately a separate
// file from sv_rw_noncentered.stan (which lw_sv's SV renders include) so
// the lw_sv renders do not change for a helper only UCSV uses.
//
// UNITS (spec §1.5, shared by every family): h is log-VARIANCE. exp(h) is
// the variance -- what goes on a covariance diagonal -- and exp(h/2) is the
// corresponding sd.
//
// Mirrored exactly by macrotoolkit.smoother.sv_scalar_variance_path (held
// together by the G1 harness's Q_t SV-composition comparison, loglik_svq).

/**
 * Time-varying 1x1 covariance path from one log-variance path:
 * C_t = [exp(h_t)]. Feeds either argument of kalman_loglik -- UCSV uses
 * it for Q_t (trend shock, h_eta) and R_t (transitory shock, h_eps).
 *
 * @param h T log-variances
 * @return array of T 1x1 covariance matrices
 */
array[] matrix sv_scalar_variance_path(vector h) {
  int T = num_elements(h);
  array[T] matrix[1, 1] C;
  for (t in 1:T) {
    C[t] = [[exp(h[t])]];
  }
  return C;
}
