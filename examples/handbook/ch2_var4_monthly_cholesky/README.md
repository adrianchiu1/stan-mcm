# Chapter 2, example 2: a 4-variable monthly VAR(2) with Cholesky-identified IRFs

Handbook: FFR, 10-year yield, unemployment, inflation (monthly, 46 rows
ending 2010m12), `lambda1 = 0.1`, `lambda3 = 0.05`, `lambda4 = 1`, own
first-lag mean 0.95, the FFR equation's cross-lag coefficients shrunk to
zero (prior variance 1e-9); IRFs to a government-bond-yield shock through
`A0 = chol(Sigma)`.

Here: the same Minnesota arithmetic through `au.minnesota_priors` (the
1e-9 entries applied by name), the recursive form in the handbook's
ordering, and the engine's structural IRFs -- under the ordering they ARE
the Cholesky IRFs (`irf_horizon: 36`); the `e_bond10y` shock is the
handbook's yield shock. The sample is 46 months, so the smoke chains are
short and the posterior is prior-dominated exactly as in the handbook.

## Smoke run record

- run `74381f101aa1` (2 chains x 300/300, reused): fit-time mirror check max |Stan - Python| = 7.04e-08 (relative 5.3e-12) over 5 prior draws, diagnostics verdict WARN (592 iteration(s) hit max_treedepth=10; max R-hat 1.0210 > 1.01; min bulk/tail ESS (215/265) < 400).
- fast validation tier: PASS -- mirror PASS (max |Stan - Python| KF loglik = 7.040e-08 (max relative 5.282e-12) over 25 prior draws (gate max(1e-08, 1e-11 * |loglik|))); hd_identity PASS (historical-decomposition reconstruction: max |error| = 9.21e-13 over 5 simulation-smoother draws (gate 1e-06))
- Cholesky IRFs to the bond-yield shock (example 2), impact medians: ffr +0.000, bond10y +0.310, unemployment +0.045, inflation +0.074.
