# Chapter 5, example 5: a TVP-AR(1) with stochastic volatility for UK inflation

Handbook: `pi_t = c_t + b_t pi_(t-1) + e_t`, `var(e_t) = h_t` a
log-random-walk volatility (`ln h_t = ln h_(t-1) + g^(1/2) u_t`),
`(c_t, b_t)` random walks with covariance `Q ~ IW(Q0, T0)`; a
date-by-date independence Metropolis step for `h` (Jacquier-Polson-Rossi),
Carter-Kohn for the coefficients, 50,000 sweeps. Data: the UK price level
1914Q1-2011Q1 (`inflation.xlsx`), annual inflation `100 (ln P_t - ln
P_(t-4))`, a 10-observation training sample for the initial conditions.

Here (S9 E4 + the established SV block): the same equation with `c`,
`b` random-walk STATES (`b*pi[-1]` is the data-dependent loading) and
`e` under the non-centered random-walk log-variance SV. The handbook's
training-sample OLS gives the initial conditions, stamped by
`build_specs.py`: `B0 = (0.2558, 13.9856)` (slope, constant),
`sqrt(diag(VV0)) = (0.7463, 13.5133)`, `mu_h0 = ln(std(E0)^2) =
1.125` with `h0_sd = sqrt(10)` (the handbook's `sigmabar`). The
handbook's `Q0 = VV0 T0 1e-4` inverse-Wishart prior is replaced by
`half_normal(0.1)` / `half_normal(0.05)` scales on the two random walks
and `g ~ IG(1, 0.01)` by `half_normal(0.3)` on `sigma_h`. The estimation
sample is the handbook's (1917Q4-2011Q1, 374 rows; `data.sample.start`
is the lag row). Outputs: the smoothed `c_t`, `b_t` and `exp(h_t/2)`
paths (the handbook's four panels; the long-run mean `c_t/(1 - b_t)` is
in the smoke record), IRFs conditional on the coefficient state at three
dates (`outputs.irf_dates`; the coefficient shocks are omitted with the
reason stated), the fan chart.

## Smoke run record

- run `56e3014fa57a` (2 chains x 300/300, reused): fit-time mirror check max |Stan - Python| = 4.55e-13 (relative 2.7e-16) over 5 prior draws, diagnostics verdict WARN (max R-hat 1.0307 > 1.01; min bulk/tail ESS (309/168) < 400).
- fast validation tier: PASS -- mirror PASS (max |Stan - Python| KF loglik = 1.983e-10 (max relative 1.306e-14) over 25 prior draws (gate max(1e-08, 1e-11 * |loglik|))); hd_identity PASS (historical-decomposition reconstruction: max |error| = 3.08e-06 over 5 simulation-smoother draws (gate 1e-06, or 1e-12 x the draw's largest bar where that is larger: the largest error is draw 4's, whose bars reach 2.3e+09 so its gate is 2.25e-03))
- the handbook's four panels at 1930Q1 / 1975Q1 / 2008Q4 (posterior medians): b_t 0.83, 0.97, 0.67; c_t -0.60, 2.46, 0.67; long-run mean c/(1-b) -1.7, 31.3, 1.9; volatility exp(h/2) 1.52, 0.35, 1.09.
- IRF of the e shock on pi at h = 4, conditional on the coefficient state at 1930-01-01: +0.52, 1975-01-01: +0.98, 2008-10-01: +0.30 (impact = 1 s.d. at the end-of-sample volatility; omitted: ['eta_b']).
