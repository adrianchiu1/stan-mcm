# Chapter 2, example 1: a bivariate VAR(2) with a Minnesota prior (+ a 3-year forecast)

Handbook: US GDP growth and inflation 1948Q1-2010Q4, `lambda1..4 = 1`,
prior mean 1 on the own first lag, the diagonal `H` from the AR(1)
residual sds `s1`, `s2`; `Sigma ~ IW(I, N + 1)`; a 12-quarter forecast.

Here: `au.var(..., priors=au.minnesota_priors(...))` -- the recursive
(Cholesky-ordered `y`, `pi`) form with the INDEPENDENT-NORMAL Minnesota
prior on the recursive-form lag coefficients and constants, exactly the
handbook's `H` arithmetic (tests/test_s8_var.py pins it against example
1's `H`); the contemporaneous coefficient `a0_pi_y ~ N(0, 10)`, the
orthogonal shock scales `half_normal(5)` in place of the inverse-Wishart
(`macrotoolkit/authoring/var.py` states the correspondence). Forecast
fan = the engine's forward simulation.

## Smoke run record

- run `4d8fc15de6a1` (2 chains x 300/300, 76s): fit-time mirror check max |Stan - Python| = 1.82e-12 (relative 2.0e-16) over 5 prior draws, diagnostics verdict WARN (max R-hat 1.0197 > 1.01; min bulk/tail ESS (296/305) < 400).
- fast validation tier: PASS -- mirror PASS (max |Stan - Python| KF loglik = 1.397e-09 (max relative 1.323e-15) over 25 prior draws (gate max(1e-08, 1e-11 * |loglik|))); hd_identity PASS (historical-decomposition reconstruction: max |error| = 1.72e-13 over 5 simulation-smoother draws (gate 1e-06))
