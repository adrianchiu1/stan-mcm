# S3 plan — add stochastic volatility to the LW model

**Status: plan of record, drafted 2026-08-31 at S2 close. Implementation
not started.** Spec §7's S3 row: "Add non-centered SV block; retune
(adapt_delta, priors) | G4-precursor: clean sampling (0 or explainable
divergences) on US data; G3 pass."

## Why now (empirical motivation, not just roadmap)

The S2 COVID stress test (`STRESS-TESTS.md` §3) showed exactly the failure
mode SV removes: estimated on the full vintage through 2026Q1 with
constant shock scales, the model absorbs 2020 into σ_y* (US) or σ_IS +
a collapsed gap AR structure (EA), contaminating every parameter. SV lets
the IS/PC shock variances spike endogenously in 2020 — the Bayesian
counterpart of HLW's hand-set κ variance scalings — and should also make
the Great Moderation visible (spec §3.1: the exp(h/2) volatility-path
plot "sells the SV feature").

## Scope

Per spec §1.5, §2.1–2.3: non-centered SV on the IS and PC shocks only
(`sv_shocks: [is, pc]`, the one validated combination in v1). Constant
σ_y*, σ_g, σ_z stay as in S2. Rao-Blackwellization unchanged: Stan
samples ~10 statics + 2×T standard-normal SV innovations; the KF
marginalizes the linear states, now with time-varying innovation
covariance.

**Critical structural fact** (HANDOFF.md warning, discovered in S2): in
the marginalized LW form the IS/PC shocks are the MEASUREMENT-equation
errors — SV makes **R_t** time-varying, not Q_t. Spec §2.2's
"time-varying Q_t (and optionally H_t)" wording maps onto this form as:
Q stays constant, R_t = diag(exp(h_IS,t), exp(h_PC,t)).

## Files touched

- `stan/functions/kalman_loglik_tv.stan` — generalize to time-varying
  measurement covariance (accept a T-array of R_t, or the two h paths
  directly). **Mandatory regression check** (DECISIONS.md 2026-08-31):
  after the rewrite, the constant case must still reproduce S2's G1
  (<1e-8) and G5a (~1e-12) results — wire the existing tests to the new
  signature, do not fork the function.
- `stan/functions/sv_rw_noncentered.stan` (new, spec §2.2) — builds the h
  path from innovations + scale + initial: h_t = h_{t-1} + σ_h·ν_t,
  h_0 ~ N(μ_h0, 1). **Convention (units test pins this): h is
  log-VARIANCE; sd = exp(h/2).**
- `stan/templates/lw_sv.stan.j2` — the same file grows the SV block
  behind Jinja conditionals on `sv_shocks` (S2's no-SV render must stay
  byte-stable for empty `sv_shocks`, so S2 run hashes are unchanged).
- `specs/schema/lw_sv.py` — drop the `sv_shocks == []` validator pin;
  validate `[is, pc]` as the only non-empty combination; add the SV
  priors (σ_h,IS, σ_h,PC ~ Half-N(0, 0.2²); μ_h0,s ~ N(2·ln σ̂_OLS,s, 1))
  to `DEFAULT_PRIORS`.
- `src/macrotoolkit/run.py` — `build_stan_data` gains the rough OLS pass
  that anchors μ_h0 (spec §1.6): OLS residual sds of the IS/PC curves on
  the loaded data, passed to Stan as data.
- `src/macrotoolkit/smoother.py` — KF accepts time-varying R_t (same
  regression obligation as the Stan side); needed so G1's mirror
  comparison can cover the SV variant (extend `tests/g1_harness.py` to
  draw h paths at parameter points).
- `tests/test_units_conventions.py` — rewire the exp(h)-vs-exp(h/2) test
  to the real `sv_rw_noncentered`/Python helper once they exist.
- `tests/test_g3_sbc.py` (new) — SBC for the **no-SV** variant (G3 per
  spec §5): uniform rank statistics, visual + χ² check, `slow` marker.

## Gates and acceptance (spec §7 S3 row)

1. **G3**: SBC on the no-SV variant passes (uniform ranks). Doable
   independently of the SV block — consider landing it first as its own
   PR, since it validates S2's sampler end-to-end.
2. **G4-precursor**: full SV model samples cleanly on US data (0 or
   explainable/documented divergences) after retuning adapt_delta and, if
   needed, priors. Expect funnel geometry from σ_h near zero — the
   non-centered parameterization is required, not optional.
3. **Regression**: S2's G1/G5a still green through the KF generalization.
4. **The COVID demonstration** (stretch, not gated): re-run the
   full-vintage US spec with SV on — exp(h/2) should spike in 2020 and
   the structural parameters should stop being contaminated relative to
   the pre-COVID run. This is the stress test's payoff exhibit.

## Open questions for S3 implementation

1. **KF signature for time-varying R**: pass an array of matrices, or the
   two h vectors + a builder inside the KF? (Array of matrices is more
   general for future families; h vectors are cheaper. Lean array-of-R_t,
   constructed by a helper, so `kalman_loglik` stays family-agnostic.)
2. **SBC scale for G3**: number of replications (spec doesn't fix it;
   99 or 199 rank bins are conventional). Budget: each replication is a
   full NUTS fit — G2 cost ~2 min/fit suggests 100 fits ≈ 3–4 h, fine as
   a `slow`-marked overnight gate, but decide before building.
3. **μ_h0 OLS anchor definition**: spec §1.6 says "rough OLS pass" —
   pin down the exact regression (the IS/PC curves at HLW's own OLS-lag
   construction, as in `rstar.stage3.R`'s initialization) so run hashes
   are reproducible.

## Ground rules

Stan geometry work (the SV block, divergence hunting) is
`stan-engineer`-tier work: per this repo's rules it runs only in a
session a human is driving or directly supervising. This plan is the
checkpoint for that decision.
