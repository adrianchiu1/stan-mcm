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
}
