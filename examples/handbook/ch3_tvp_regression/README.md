# Chapter 3, examples 1-2: the TVP regression on the handbook's artificial DGP

Handbook: `Y_t = beta_t X_t + e_t`, `beta_t = beta_{t-1} + v_t`, with
`R = var(e) = 0.01`, `Q = var(v) = 0.001` FIXED, `beta_0 = 0`, `P_0 = 1`,
`T = 500`, `X ~ N(0, 1)` -- example 1 runs the Kalman filter, example 2
adds the Carter-Kohn backward draw. The script simulates fresh data on
every run; `make_data.py` simulates it ONCE at seed 20260905 into
`data/ch3_tvp_example1_sim.csv` (the true `beta_t` is written alongside
as `beta_true`; the spec does not read it).

Here (S9 E4): `Y = beta*X + e` with `beta` a random-walk STATE and `X`
an exogenous series at lag 0 (E2) -- the product compiles to the
data-dependent loading `Z_t = X_t`; the two variances are ESTIMATED
(`half_normal` scales) instead of fixed, `init(0, 1)` is the handbook's
`(beta_0, P_0)`. The smoothed `beta` band (the states figure) is the
object of interest; the test suite checks the truth lies inside the 90%
band and that the handbook's own filter loop reproduces the KF's
filtered path to 1e-10 (`tests/test_s9_grammar.py`,
`tests/test_s9_stan.py`).

## Smoke run record

- run `f48f15521246` (2 chains x 300/300, 58s): fit-time mirror check max |Stan - Python| = 8.53e-14 (relative 4.2e-16) over 5 prior draws, diagnostics verdict WARN (min bulk/tail ESS (286/207) < 400).
- fast validation tier: PASS -- mirror PASS (max |Stan - Python| KF loglik = 1.137e-12 (max relative 7.856e-16) over 25 prior draws (gate max(1e-08, 1e-11 * |loglik|))); hd_identity PASS (historical-decomposition reconstruction: max |error| = 1.11e-16 over 5 simulation-smoother draws (gate 1e-06))
- beta_t recovery (examples 1-2): the true path lies inside the 90% smoothed band at 91.0% of periods; RMSE of the smoothed median vs the truth 0.0371 (the handbook's fixed Q = 0.001, R = 0.01 are estimated here).
