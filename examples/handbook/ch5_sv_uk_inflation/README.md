# Chapter 5, example 4: the stochastic-volatility model for UK inflation, in the handbook's own form

Handbook: `y_t = e_t`, `e_t ~ N(0, h_t)`, `ln h_t = ln h_{t-1} + sqrt(g) u_t`, `g ~ IG(0.01, 1)`,
`ln h_0 ~ N(mubar, 10)` with `mubar` the log variance of a 10-quarter training sample; the
volatility path drawn date by date by the Jacquier-Polson-Rossi independence Metropolis step.
Data: the UK price level 1914Q1-2011Q1, annual inflation `100 (ln P_t - ln P_{t-4})`.

Here: a constant mean `c` (E1) and the SV block on the measurement shock -- no state at all
(E0). `g ~ IG` is replaced by `half_normal(0.3)` on the log-variance random walk's standard
deviation `sigma_h`; `mubar` by the `log_var_diff` anchor (no training sample discarded) with
`h0_sd = 3`. `h` is the log VARIANCE in both toolkits; the shock's standard deviation is
`exp(h/2)`. The log-variance path is sampled by NUTS in the non-centered parameterization.
