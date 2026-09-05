# Chapter 3, example 3: a TVP-VAR for US GDP growth, CPI inflation and the federal funds rate

Handbook: a VAR(2) with a constant whose 21 coefficients are random
walks (`beta_t = beta_(t-1) + v_t`, `Q ~ IW(Q0, T0)`), constant `Sigma ~
IW`, Carter-Kohn for the coefficient paths with a stability rejection,
110,000 sweeps; priors and initial conditions from a 40-quarter
pre-sample (`P00 = V0 = kron(sigma0, inv(x0'x0))`, `beta0 = vec(b0)`,
`Q0 = V0 T0 3.5e-4`); time-varying IRFs to a policy shock identified by
sign restrictions.

Here (S9 E4 + S8 E5): the recursive (Cholesky-ordered
gdp_growth -> cpi_inflation -> ffr) form -- constant `Sigma` through the
`a0` contemporaneous coefficients and constant shock scales -- with EVERY
coefficient (intercepts and lags, 21 states) a random walk multiplying
the lagged observables (the data-dependent loading `Z_t`; the E5
substitution composes the loadings with `a0`). Initial conditions from
the handbook's pre-sample OLS, stamped by `build_specs.py` (in per cent;
`sqrt(diag(V0))` per coefficient; e.g. the gdp_growth intercept
`2.854 +- 2.194`). The inverse-Wishart `Q0 = V0 T0
3.5e-4` implies a per-coefficient random-walk sd of `0.118 sqrt(V0_ii)`
-- about 0.01 for the lag coefficients here -- and is replaced by one
`half_normal(0.01)` random-walk scale per equation (`sq_*`; a first
smoke at `0.05` let the coefficient drift absorb the innovations,
DECISIONS.md 2026-09-05); the IW on `Sigma` by `half_normal(2)` shock
scales and `N(0, 1)` on `a0`.
Estimation sample 1964Q3-2010Q2 (184 rows; `data.sample.start` is the
second lag row). Outputs: IRFs to the three orthogonalized shocks
CONDITIONAL on the coefficient state at 1975Q1, 1995Q1 and 2008Q4
(`outputs.irf_dates`; the handbook's sign-restricted policy shock is a
post-processor over these, not run here), the 21 smoothed coefficient
paths, the HD with the coefficient shocks omitted (reason stated), the
fan. The stability rejection is not imposed (the KF marginalizes the
paths; explosive posterior mass is reported by the fast tier's
stationarity filter, not truncated). The smoke run is short:
21 states, `m = 3`, 27 sampled parameters; its record reports the scale
posteriors because a per-equation drift scale shared by the intercept
and the lag coefficients lets a random-walk intercept stand in for a
persistent series' innovations (the ffr equation does this) -- stated,
not tuned away, in a smoke run.

## Smoke run record

- run `2ee4bc9e9b98` (2 chains x 200/200, reused): fit-time mirror check max |Stan - Python| = 3.31e-10 (relative 2.0e-14) over 5 prior draws, diagnostics verdict FAIL (max R-hat 1.0261 > 1.01; min bulk/tail ESS (122/52) < 100).
- fast validation tier: PASS -- mirror PASS (max |Stan - Python| KF loglik = 5.864e-09 (max relative 8.731e-14) over 25 prior draws (gate max(1e-08, 1e-11 * |loglik|))); hd_identity PASS (historical-decomposition reconstruction: max |error| = 2.56e-12 over 5 simulation-smoother draws (gate 1e-06))
- IRFs to the ffr shock at 1975-01-01 (Cholesky ordering gdp_growth -> cpi_inflation -> ffr; medians at h = 1 / 4 / 8): gdp_growth +0.000 / -0.044 / -0.002, cpi_inflation +0.000 / +0.029 / +0.015, ffr +0.079 / -0.002 / +0.009.
- IRFs to the ffr shock at 1995-01-01 (Cholesky ordering gdp_growth -> cpi_inflation -> ffr; medians at h = 1 / 4 / 8): gdp_growth +0.000 / +0.000 / -0.011, cpi_inflation +0.000 / +0.017 / +0.012, ffr +0.079 / +0.042 / +0.013.
- IRFs to the ffr shock at 2008-10-01 (Cholesky ordering gdp_growth -> cpi_inflation -> ffr; medians at h = 1 / 4 / 8): gdp_growth +0.000 / -0.026 / -0.002, cpi_inflation +0.000 / +0.031 / -0.003, ffr +0.079 / +0.010 / +0.001.
- 18 coefficient shocks omitted from the IRF/HD (no response from rest); the coefficient paths are in the states figure.
- posterior medians of the scales: innovations s_gdp_growth 2.930, s_cpi_inflation 0.508, s_ffr 0.091; coefficient drift sq_gdp_growth 0.012, sq_cpi_inflation 0.010, sq_ffr 0.052 (prior half_normal(0.01)). Where a drift scale is pulled far above its prior the equation's random-walk intercept is absorbing the series' persistence in place of the innovation -- the handbook's tight IW Q0 and stability rejection prevent this; a per-coefficient scale or a fixed drift scale is the modeling response.
