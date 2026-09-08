# Chapter 6 — Bayesian estimation of linear DSGE models (gap chapter)

*Handbook Chapter 6: a three-equation New Keynesian model (an IS curve, a
Phillips curve, a policy rule) written in Sims' canonical form
`Γ₀ y_t = Γ₁ y_{t-1} + C + Ψ z_t + Π η_t`, solved by `gensys`
(`model_solve.m`, `example1.m`), its likelihood evaluated by the Kalman
filter on the solved state space (`likelihood.m`, `example2.m`), the
parameters estimated by random-walk Metropolis–Hastings with bounds
(`example3.m`), and the model compared with a restricted version by the
Gelfand–Dey marginal likelihood (`example4.m`).*

## The model and the estimator

The handbook's estimation problem has the shape the toolkit is built for:
conditional on the structural parameters `θ`, the solved model *is* a
linear-Gaussian state-space system `y_t = H β_t`, `β_t = F(θ) β_{t-1} +
G(θ) ε_t`, and the likelihood is the Kalman filter's. The random-walk MH in
`example3.m` samples `θ` over exactly that marginal likelihood; NUTS
would do the same job with less tuning. So the obstacle is not the
sampler or the filter.

## Why this is outside macrotoolkit's class

The obstacle is the map from `θ` to `(F, G)`. In the handbook it is a
rational-expectations *solution* — `gensys` finds the stable solution of
the canonical system, and whether a stable, unique solution exists
depends on `θ` (the Taylor-principle region). In macrotoolkit the system
matrices are declared *directly* by the equations: the compiler evaluates
coefficient expressions in the parameters entry by entry, and the Stan
program performs the same operations. A generalised eigenvalue
decomposition with a determinacy check inside the likelihood is a
different kind of object: it is not expressible as coefficient
expressions, it has regions of the parameter space where the likelihood
is undefined (the handbook handles this by bounds and rejection), and its
derivatives — which NUTS needs — pass through the eigen-decomposition.

`VISION.md` lists "DSGE solution steps inside the estimation loop" as a
non-goal. The companion therefore does not estimate the Chapter 6 model.

## What would be needed, and what other tools do

- **Solving outside and estimating inside.** A DSGE model solved at a
  *fixed* `θ` is an ordinary state-space model, and the toolkit estimates
  ordinary state-space models: the shock scales and measurement errors of
  a solved model can be estimated today by declaring the solved `F` and
  `G` as numeric matrices in transition equations. That is calibration-
  plus-estimation of the shocks, not DSGE estimation.
- **Estimating the reduced form the DSGE implies.** The handbook's
  three-equation model has a VAR(1) reduced form in `(x, π, i)`. Chapter 2
  of this companion estimates such VARs with Minnesota or steady-state
  priors and identifies shocks by ordering or by sign restrictions —
  the standard empirical counterpart to a small DSGE, without the
  cross-equation restrictions.
- **Purpose-built tools.** Dynare's Bayesian estimation, Stan programs
  with a hand-written `gensys` in the transformed parameters block, or
  the `gEcon`/`DSGE.jl` ecosystems are the right tools for the Chapter 6
  problem. Their likelihood is the same Kalman-filter likelihood the
  toolkit validates — so a solved-model oracle at fixed `θ` (the toolkit's
  filter vs Dynare's at the same matrices) would be the one useful bridge
  test if a comparison were ever needed.

## Status

Outside the class by design. No planned extension. The Chapter 6 model's
*reduced form* is estimable with Chapter 2's machinery, and the marginal-
likelihood comparison of `example4.m` waits on E8 (model comparison) in
any case.
