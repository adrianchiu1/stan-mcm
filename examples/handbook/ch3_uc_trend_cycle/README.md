# Chapter 3, §2 (equations 2.6-2.7): the unobserved-components trend-cycle model

Handbook: `Y_t = C_t + tau_t` EXACTLY (no measurement error), `tau_t =
tau_{t-1} + e2_t`, `C_t = c + a1 C_{t-1} + a2 C_{t-2} + e1_t`, in the
state-space form with state `(C_t, tau_t, C_{t-1})`, the constant in the
transition intercept vector, a singular `R = 0` and a `Q` with a possible
`e1`-`e2` covariance; a generic example (no script).

Here (S8 E1 + E3): the shock-free measurement row (`R = 0`, allowed
because the row loads stochastic states -- the PD proof in
plans/S8-plan.md), the cycle's constant as a transition DRIFT (the
implicit unit state `_const`), orthogonal shocks (the handbook's `Q`
off-diagonal is not expressible: a shared/correlated state shock is
outside the grammar; stated). Applied to US inflation (trend inflation +
an AR(2) cycle -- the handbook gives no dataset for §2). The compiled
matrices equal the hand-built (2.6)-(2.7) construction and the DK draws
reproduce `Y = C + tau` per period (tests/test_s8_grammar.py).

## Smoke run record

- run `e8c65ebfc66b` (4 chains x 500/500, 51s): fit-time mirror check max |Stan - Python| = 2.27e-13 (relative 2.9e-16) over 5 prior draws, diagnostics verdict PASS (no divergences, treedepth hits, low E-BFMI, high R-hat, or low ESS detected).
- fast validation tier: PASS -- mirror PASS (max |Stan - Python| KF loglik = 4.547e-12 (max relative 9.607e-16) over 25 prior draws (gate max(1e-08, 1e-11 * |loglik|))); hd_identity PASS (historical-decomposition reconstruction: max |error| = 8.44e-15 over 5 simulation-smoother draws (gate 1e-06))
