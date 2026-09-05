# Chapter 1, examples 1-2: an AR(2) for US inflation with a constant (+ a 3-year forecast)

Handbook: `Y_t = c + b1 Y_{t-1} + b2 Y_{t-2} + v_t`, Gibbs with `B ~ N(0, I)`
and `sigma^2 ~ IG(T0 = 1, D0 = 0.1)`, stability enforced by rejection;
example 2 adds the 12-quarter forecast fan.

Here (S8 E0/E1): the constant is a PARAMETER with the handbook's `N(0, 1)`
prior (`c`), the lag coefficients likewise; the inverse-gamma on `sigma^2`
is replaced by `half_normal(2)` on `sigma` (the authored prior menu);
stationarity is not imposed (the posterior mass on explosive roots is
negligible for this series). No state: the KF runs with `n = 0` and is
exactly the Gaussian regression likelihood. The fan chart is the
engine's forward simulation (`outputs.horizon: 12`). The alternative
route -- the constant as a shock-free state `c = c[-1]` with `init(0, 1)`
-- gives the same marginal likelihood (the E0 oracle) with `c` integrated
by the filter instead of sampled.

## Smoke run record

- run `d2db589230ea` (2 chains x 300/300, reused): fit-time mirror check max |Stan - Python| = 3.64e-12 (relative 1.3e-16) over 5 prior draws, diagnostics verdict WARN (max R-hat 1.0228 > 1.01; min bulk/tail ESS (245/207) < 400).
- fast validation tier: PASS -- mirror PASS (max |Stan - Python| KF loglik = 3.725e-08 (max relative 6.788e-16) over 25 prior draws (gate max(1e-08, 1e-11 * |loglik|))); hd_identity PASS (historical-decomposition reconstruction: max |error| = 2.13e-14 over 5 simulation-smoother draws (gate 1e-06))
