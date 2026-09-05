# Chapter 1, example 3: the AR(2) with AR(1) disturbances, in its exact substituted form

Handbook: `Y_t = c + b1 Y_{t-1} + b2 Y_{t-2} + v_t`, `v_t = rho v_{t-1} + e_t`,
estimated by the Cochrane-Orcutt-style Gibbs blocks (quasi-differenced
regression for `B`, a regression of the residual on its lag for `rho`).

Here: the exactly equivalent quasi-differenced equation
`infl = c(1 - rho) + (b1 + rho) infl[-1] + (b2 - rho b1) infl[-2] - rho b2 infl[-3] + e`
-- coefficient EXPRESSIONS over the parameters, which the grammar accepts
(the model stays linear in the series) -- with the handbook's `N(0, 1)`
priors on `c`, `b1`, `b2` and `rho` (`rho` truncated to (-1, 1), the
stationarity the handbook imposes by rejection). One lag more of
pre-sample data is consumed (`lag_depth = 3`).

## Smoke run record

- run `7a486a179c3c` (2 chains x 300/300, reused): fit-time mirror check max |Stan - Python| = 3.64e-12 (relative 2.1e-16) over 5 prior draws, diagnostics verdict FAIL (max R-hat 1.0142 > 1.01; min bulk/tail ESS (146/78) < 100).
- fast validation tier: PASS -- mirror PASS (max |Stan - Python| KF loglik = 1.164e-10 (max relative 6.731e-16) over 25 prior draws (gate max(1e-08, 1e-11 * |loglik|))); hd_identity PASS (historical-decomposition reconstruction: max |error| = 3.64e-14 over 5 simulation-smoother draws (gate 1e-06))
