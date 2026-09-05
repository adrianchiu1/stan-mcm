# Chapter 2, example 8: conditional forecasts (Waggoner-Zha) from a bivariate VAR(2)

Handbook: OLS/flat-prior Gibbs for the VAR (the conditional forecast is
appended to the data each sweep), inflation constrained to `(1, 1, 1)`
over 3 quarters, the restricted structural shocks `N(R'(RR')^+ r, I -
R'(RR')^+ R)` with `R` from the Cholesky IRFs.

Here: the recursive VAR(2) with the example-1 Minnesota priors (the
handbook's flat prior puts no mass on stationary draws, so the fast
tier's HD-identity gate -- which needs stationary prior points -- cannot
run under it; stated) and no data augmentation ( the posterior conditions on the observed sample only --
the handbook's augmentation step feeds the conditional path back into
the VAR posterior, a Gibbs device this KF-marginal likelihood does not
replicate; stated) -- and `macrotoolkit.postprocess.conditional_forecast_for_run`
with `{("pi", 0): 1, ("pi", 1): 1, ("pi", 2): 1}` -- the conditioned
series reproduces the path exactly, GDP growth carries the implied
distribution (tests/test_s8_postprocess.py). The specification is
example 1's model under a different name (the post-processor is the
example), so its posterior record coincides with `ch2_bivar_minnesota`'s.

## Smoke run record

- run `b6efa20845d4` (2 chains x 300/300, 79s): fit-time mirror check max |Stan - Python| = 1.82e-12 (relative 2.0e-16) over 5 prior draws, diagnostics verdict WARN (max R-hat 1.0197 > 1.01; min bulk/tail ESS (296/305) < 400).
- fast validation tier: PASS -- mirror PASS (max |Stan - Python| KF loglik = 1.397e-09 (max relative 1.323e-15) over 25 prior draws (gate max(1e-08, 1e-11 * |loglik|))); hd_identity PASS (historical-decomposition reconstruction: max |error| = 1.72e-13 over 5 simulation-smoother draws (gate 1e-06))
- conditional forecast (example 8): inflation held at (1, 1, 1) -> median GDP growth path [2.878, 3.135, 3.478] (unconditional [2.843, 3.187, 3.572]); the pi path reproduces (1, 1, 1) to 1.5e-07.
