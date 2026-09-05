# Chapter 2, example 3: the steady-state (Villani 2009) VAR

Handbook: the VAR in deviations from long-run means `mu` with a prior
`mu ~ N((1, 1), 0.001 I)`, Minnesota priors on the lag coefficients (no
constant), a Gibbs block for `mu` (Villani's Appendix A), a 10-year
forecast.

Here: the long-run means are shock-free CONSTANT STATES `mu_y`, `mu_pi`
(`au.var(..., intercept="steady_state")`, S8 E0) whose initial condition
IS the handbook's prior (`init(1.0, sqrt(0.001))`); the intercept of each
recursive equation is the Villani identity written with parameter-
expression coefficients on the states, and the filter integrates `mu`
exactly (tests/test_s8_var.py: the KF likelihood equals the closed-form
marginal over `mu`). No new machinery. Smoothed `mu` paths are flat
lines (a constant state) at the posterior of the long-run means.

## Smoke run record

- run `0257900173a5` (2 chains x 300/300, 89s): fit-time mirror check max |Stan - Python| = 1.82e-12 (relative 1.9e-16) over 5 prior draws, diagnostics verdict WARN (max R-hat 1.0125 > 1.01; min bulk/tail ESS (213/282) < 400).
- fast validation tier: PASS -- mirror PASS (max |Stan - Python| KF loglik = 1.182e-11 (max relative 5.480e-15) over 25 prior draws (gate max(1e-08, 1e-11 * |loglik|))); hd_identity PASS (historical-decomposition reconstruction: max |error| = 1.31e-14 over 5 simulation-smoother draws (gate 1e-06))
- long-run means (constant states, posterior median of the smoothed path): mu_y 1.006, mu_pi 0.999.
