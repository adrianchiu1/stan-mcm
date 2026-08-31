# Decisions log

One dated line per judgment call not fixed by the spec, with rationale.
Newest first.

- **2026-08-13 — `pytest` is installed as a `uv tool` with the project's
  full dependency set + an editable `macrotoolkit` install, rather than
  relying on a project-local `.venv` someone must remember to activate.**
  `scripts/gate-check.sh`'s `SubagentStop` hook invokes a bare `pytest -q
  ...` with no shell setup of its own — it needs `pytest` to import
  `macrotoolkit`/`numpy`/`pandas`/`cmdstanpy`/etc. out of the box. A
  project `.venv` (what S1/S1-test-suite work used) satisfies this only
  within a shell where it's been sourced; deleting it between turns (as
  this session did, for tidiness) breaks the hook for whoever runs it
  next. `uv tool install pytest --with cmdstanpy --with numba --with arviz
  --with pydantic --with jinja2 --with matplotlib --with pyyaml --with
  pandas --with click --with h5netcdf --with-editable .` (receipt at
  `~/.local/share/uv/tools/pytest/uv-receipt.toml`) makes the globally-on-
  `PATH` `pytest` binary self-sufficient. Container/environment-level only
  — touches nothing under version control. Revisit if `pyproject.toml`'s
  dependency list changes (the tool install needs re-running to pick up
  new/changed deps; it does not auto-track `pyproject.toml`).

- **2026-08-13 — S1 toy model shows real (if mild) divergences; left as-is
  rather than hand-tuned to a clean PASS.** The non-centered random-walk
  local-level model (`stan/templates/local_level.stan.j2`) produces ~0.5%
  divergent transitions on the example run (`diagnostics.json` verdict:
  FAIL). Tried `adapt_delta` 0.9–0.97 and several warmup lengths; the
  divergence rate stayed roughly constant (a mild funnel characteristic of
  this model class near `sigma_level -> 0`, not a plumbing bug) and
  `adapt_delta=0.99` traded divergences for 78% max-treedepth hits instead.
  Spec §7's S1 acceptance test is "produces immutable run dir with draws +
  diagnostics" — not "achieves a clean PASS verdict" (that's S3+ territory
  for the real model). A FAIL verdict with a correct, specific reason is
  the diagnostics harness doing its job, not a defect; not spending further
  effort chasing a cosmetic PASS on a throwaway infra-proving toy.
- **2026-08-13 — `data.snapshot.csv` is a byte-identical copy of the whole
  source CSV, not the mapped/trimmed subset.** Keeps it in lockstep with
  the run-identity hash, which is computed over the full file's raw bytes
  (not the mapped/trimmed data) — the snapshot should reproduce exactly
  what was hashed.
- **2026-08-13 — `runs/<hash>/spec.yaml` is the canonicalized spec
  (validated, sorted-key YAML via `RunSpec.to_canonical_yaml()`), not a
  copy of the original source file.** A faithful audit record regardless
  of the original file's key order/comments/formatting.
- **2026-08-13 — Compile cache lives at `.mtk_cache/stan/<hash>_<version>/`,
  gitignored, not under `stan/templates/`.** Keeps the templates directory
  free of build artifacts; cache is always reproducible from template +
  context + CmdStan version.
- **2026-08-13 — `build_stan_data` (DataFrame -> Stan `data` block) is a
  small explicit dispatch function in `run.py`, not yet generalized onto
  the family registry.** Only `local_level` exists; guessing `lw_sv`'s
  data-block shape now would be premature. Flagged in its docstring for S2
  to generalize once that shape is actually known.

- **2026-08-13 — G5a oracle is a self-derived reproduction, not literally
  HLW's own published numbers, and that's treated as sufficient.** Ran
  `HLW_2017_Code/` (the genuine 2017 code) ourselves on data we already
  had, since neither supplied workbook publishes 2017-vintage MLE
  parameters or a smoothed series for any vintage. Sanity-checked the
  reproduction's one-sided output against a real published vintage
  (`real_time_estimates.xlsx` sheet `2019Q2`, same sample end by
  construction): 0.06pp mean / 0.31pp max deviation across 234 quarters
  on output gap/g/z/r*. Treating this as close enough to trust as the
  G5a oracle — the residual gap is attributable to named, expected causes
  (data revisions since 2019, independent optimizer path), not a
  methodology error. Output in
  `tests/fixtures/hlw/derived/us_2017_reproduction/`, full detail in its
  `README.md` and in `FIXTURES.md`. `run.se=FALSE` (skipped the
  5000-iteration Monte Carlo standard-error procedure) since G5a compares
  point/smoothed state paths, not confidence intervals.

- **2026-08-13 — Pin CmdStan to version 2.36.0, never call `install_cmdstan`
  without an explicit `version=`.** `api.github.com` (used by cmdstanpy's
  default "resolve latest release" path) is blocked by this environment's
  egress policy; direct `github.com/.../releases/download/...` asset URLs
  are not. Pinning sidesteps the blocked lookup and also gives reproducible
  builds. Verified 2.36.0 downloads, builds, and samples correctly in this
  environment (see `PREFLIGHT.md` §2). Revisit if a newer CmdStan is needed
  for a specific feature.

- **2026-08-13 — HLW fixture data: use the user-supplied workbooks, not a
  fetch.** `www.newyorkfed.org` is blocked by org egress policy (confirmed
  403 at CONNECT). Per instructions, did not reconstruct data from memory
  or substitute another source. Asked the user; the user supplied
  `Holston_Laubach_Williams_current_estimates.xlsx` and
  `Holston_Laubach_Williams_real_time_estimates.xlsx` directly, now in
  `tests/fixtures/hlw/data/`. Treated as authoritative NY Fed output based
  on their internal `info`-sheet headers and citations, not independently
  re-verified against the live site (that site is unreachable from this
  session).

- **2026-08-13 — HLW model-variant target for G5a: 2017, not 2023.**
  Recommended to the user with reasoning (G5a validates the shared KF
  library the *production* model uses, and production's functional form is
  the 2017 equations — SV sits on top of that, it doesn't replace HLW's
  COVID-adjustment machinery; G5b's own pass criterion already restricts
  its window to 2000–2019, independently confirming COVID-era agreement
  isn't needed for launch). User then supplied the genuine 2017 code
  (`HLW_2017_Code/`, verified by direct term-by-term comparison of its
  Stage 3 matrices against spec §1.2–§1.3, and by grep showing zero
  COVID/κ/φ references) — treating this as confirmation. See `FIXTURES.md`.

- **2026-08-13 — Install R via apt (`--no-install-recommends`) +
  build `tis` from `github.com/cran/tis` rather than CRAN.** Needed to
  eventually run `HLW_2017_Code/` ourselves and derive a self-consistent
  G5a oracle (published parameter values for any 2017-vintage aren't
  available in either supplied workbook). `cran.r-project.org` /
  `cloud.r-project.org` are blocked the same way as `newyorkfed.org`
  (403 at CONNECT); `r-base-core`, `r-cran-nloptr`, `r-cran-mfilter` are
  in Ubuntu's apt repos and install directly. `tis` isn't packaged for
  Ubuntu, but its CRAN mirror is a plain readable git repo on GitHub,
  which is reachable — `git clone` + `R CMD INSTALL` builds it from
  source with no CRAN access needed. Verified all three load correctly.
  A first `apt-get install r-base-core ...` attempt (with recommends)
  pulled in ~886MB of unrelated GUI/media-codec packages and failed on
  stale-mirror 404s for a few of them; `--no-install-recommends` avoided
  that entirely and installed cleanly.

- **2026-08-31 — S2 open question 2 resolved: build the constant-covariance
  KF first, generalize in S3.** User decision. `kalman_loglik_tv.stan` for
  S2 is the simpler constant-`Q` form (all S2 needs, and all G1/G2/G5a
  exercise); match HLW via G5a first, then generalize the function to
  time-varying `Q_t` when S3 adds SV. Spec §2.2's "must handle time-varying
  state innovation covariance" contract is deferred to S3, not dropped —
  S3's first task includes that generalization plus a regression check that
  the constant case still reproduces S2's G1/G5a results.

- **2026-08-31 — S2 open question 3 signed off: G5a tolerance is against
  our own reproduction, near machine precision.** User confirmed the
  clarification already in `plans/S2-plan.md`: the oracle is
  `tests/fixtures/hlw/derived/us_2017_reproduction/output/us_2017_smoothed.csv`,
  and our KF/smoother evaluated at `us_2017_parameters.csv` must match it
  to floating-point-level tolerance. The 0.06pp/0.31pp gap vs. the
  published 2019Q2 vintage is a separate, documented discrepancy (data
  revisions, optimizer path) and is not slack our code may consume.

- **2026-08-31 — S2 open question 0 (constant IS/PC shock-scale priors):
  HLW's own convention established; recommendation is option (a) with
  Half-N(0, 1²), awaiting user confirmation.** What HLW actually does
  (from `tests/fixtures/hlw/HLW_2017_Code/rstar.stage3.R` +
  `unpack.parameters.stage3.R`): frequentist MLE, no priors of any kind.
  `σ_ỹ` and `σ_π` are elements 6–7 of the stage-3 parameter vector, freely
  estimated by L-BFGS with no bounds (the only bounded parameters are
  `a_3 ≤ −0.0025` and `b_2 ≥ 0.025`), initialized at the OLS residual
  standard errors of the IS and Phillips curves. The pile-up machinery
  (median-unbiased `λ_g`, `λ_z`) constrains only `σ_g` and `σ_z`, which
  are *derived* (`σ_g = λ_g·σ_y*`, `σ_z = λ_z·σ_ỹ/a_r`), never free.
  So option (a)'s Bayesian analogue: `σ_IS`, `σ_PC` as free constant
  scales with a weakly-informative prior doing no identification work.
  Recommended concrete prior: Half-N(0, 1²) for both — matches the spec's
  Half-Normal convention for scale parameters, comfortably covers the
  US MLE values (σ_ỹ≈0.34, σ_π≈0.80) without centering on them (keeping
  the prior independent of the G5a oracle, unlike option (b)), and these
  parameters are not pile-up-prone so looseness costs nothing. Not yet
  confirmed by the user — do not write `specs/schema/lw_sv.py`'s priors
  until it is.

- **2026-08-31 — S2 open question 0 closed: user confirmed Half-N(0, 1²)
  priors on `σ_IS`, `σ_PC`.** Option (a) as recommended in the entry
  above: free constant scales, weakly-informative Half-N(0, 1²), no
  relation to the SV parameters, no identification work. This was the
  last blocking input — S2 implementation can start. All three of
  `HANDOFF.md`'s open questions are now resolved.

- **2026-08-31 — G5a initialization: dump HLW's exact `xi.00`/`P.00` from
  the R run rather than re-deriving them in Python.** Reading
  `calculate.covariance.R` revealed `P.00` is the output of a *full inner
  nloptr L-BFGS optimization* (starting from `0.2*I`, with numerical
  gradients) — reproducing that trajectory bit-for-bit outside R is not
  realistic, and spec §2.2 requires the KF to take the initial mean/cov
  explicitly anyway. `run_us_2017.R` extended to write
  `output/us_2017_xi00.csv` / `us_2017_P00.csv`; R re-installed in this
  container per the documented apt + github.com/cran/tis procedure, and the
  regeneration reproduced every previously committed fixture CSV
  **byte-identically** (determinism confirmed). The G5a tests convert these
  quarterly-g initial conditions to the spec's annualized-g state units with
  `S = diag(1,1,1,4,4,1,1)`.

- **2026-08-31 — CmdStan output precision: any test comparing Stan-computed
  reals against Python must pass `sig_figs=18` to `model.sample`.** CmdStan
  writes draws to CSV with 6 significant figures by default; on G1's
  log-likelihoods (some |ll| ~ 5e4) that alone produced ~3e-2 apparent
  "mismatch" — three orders of magnitude over the 1e-8 gate — while the
  actual Stan-vs-Python agreement is ~4e-12. Recorded because the symptom
  (G1 "fails" with a diff that scales with |loglik|) looks exactly like a
  real numerics bug and invites a wild-goose chase through the KF algebra.

- **2026-08-31 — S2 gates implemented and green: G1 (max diff 3.6e-12 vs
  1e-8 gate over 50 prior draws), G5a (loglik ~6e-12, all four
  filtered+smoothed series ~2e-12 vs the oracle, asserted at 1e-8), plus a
  term-for-term matrix cross-check against `unpack.parameters.stage3.R`.**
  The lw_sv state keeps g annualized (spec convention) — verified an exact
  unit transform of HLW's quarterly-g system, so no tolerance is consumed
  by the convention difference. G2's "~90% coverage / no systematic bias"
  is operationalized in `tests/test_g2_parameter_recovery.py` (pooled
  coverage in [0.80, 0.97] over 200 cells, per-parameter floor 0.6,
  3-sigma t-test on sigma_g/sigma_z posterior-median errors), with the
  rationale in its docstring.

- **2026-08-31 — S2 COMPLETE: all gates green.** G2 passed (20 simulated
  datasets, pooled coverage + per-parameter floor + 3-sigma no-bias on
  sigma_g/sigma_z, ~38 min under the `slow` marker). First real US run
  (`examples/us_lw_sv/`, run 24b6288dddad): diagnostics PASS, structural
  coefficients bracket the HLW MLE, smoothed r*/gap correlate 0.98 with
  HLW's smoothed series; sigma_g/sigma_z posteriors sit below HLW's
  MUE-implied values by design (pile-up priors replace MUE — sensitivity
  sweep remains S5 scope). numerics-reviewer pass found no correctness
  defects; its one hygiene fix (Cholesky-based RTS gain) is applied. S2's
  acceptance row in spec §7 — "G1, G2, G5a pass; posterior sensible on US
  data" — is met in full. S3 warnings recorded in HANDOFF.md.

- **2026-08-31 — S2 external benchmarking + COVID stress test recorded in
  `STRESS-TESTS.md`; S3 plan drafted (`plans/S3-plan.md`).** Pre-COVID
  reference runs track HLW's published current estimates at ρ ≈ 0.94–0.99
  (documented σ_z-prior level gap). Full-latest-vintage re-estimations
  (through 2026Q1, no COVID machinery) ran as deliberate stress tests:
  sampler geometry held (US 0 divergences; EA 2/6000), but COVID is
  absorbed into constant shock scales through different channels per
  economy (US: σ_y*; EA: σ_IS + gap-AR collapse) — the empirical
  motivation for S3's stochastic volatility, now written into the S3 plan.
  Pre-COVID runs remain the reference results; full-vintage specs are
  marked experiments.

- **2026-08-31 — S3 plan's three open questions resolved (user decisions,
  start of S3 implementation).** (1) **KF signature for time-varying R**:
  the generalized `kalman_loglik` takes an array of T measurement-covariance
  matrices (`array[] matrix R` / `(T,m,m)` ndarray), built by a small
  helper from the h paths — the KF stays family-agnostic; the h-vectors-
  inside-KF alternative was rejected as baking lw_sv structure into the
  shared function. (2) **G3 SBC scale**: 200 replications (~6–7 h under the
  `slow` marker), ranks from posterior draws thinned to 99 (100 rank
  values), χ² over 20 bins → 10 expected per bin; 100 was rejected as
  underpowered (5/bin), 500 as overkill for a gate G4 will repeat.
  (3) **μ_h0 OLS anchor**: mirror HLW's own stage-3 initialization
  (`rstar.stage3.R` lines 22–48) on our trimmed data (which, like HLW's,
  includes the 4 pre-sample lag quarters): gap⁰ = residual of OLS of y on
  [const, linear trend] over the FULL trimmed sample; IS: OLS of gap⁰_t on
  [gap⁰_{t-1}, gap⁰_{t-2}, (r_{t-1}+r_{t-2})/2, const] over estimation
  rows, σ̂_IS = √(RSS/(n−4)); PC: OLS of π_t on [π_{t-1},
  (π_{t-2}+π_{t-3}+π_{t-4})/3, gap⁰_{t-1}] with NO intercept,
  σ̂_PC = √(RSS/(n−3)); μ_h0,s = 2·ln(σ̂_s). Deterministic given the
  trimmed data, so run hashes stay reproducible.

- **2026-08-31 — G3 SBC runs with the a1/a2 priors overridden to
  N(0.8, 0.1²)/N(−0.25, 0.05²) via the production `priors:` override path;
  everything else production-default.** User decision, forced by a measured
  numerical fact: the production a1/a2 defaults put ~32% of prior mass on
  non-stationary gap dynamics (a1+a2 ~ N(0.8, 0.42²)), and SBC must sample
  the exact fitted prior (no stationarity rejection — that's G1/G2's
  machinery, invalid here). A 3-rep smoke showed one such draw simulating
  |y| ≈ 2e13 at T=120, at which point the KF covariance update (P entries
  ~1e26) loses everything to float64 cancellation (`cholesky_decompose`
  not-PD; all 10 ranks at the extremes) — garbage ranks for ~a third of
  replications, poisoning χ² regardless of pipeline correctness, and
  shortening T doesn't fix the tail. Under the override the stationarity
  boundary sits ~4σ out (non-stationary mass ~3e-5), so the exact prior is
  simulable with no rejection anywhere and SBC exactness holds. Rejected
  alternatives: an SBC-only stationarity-truncated (a1,a2) parameterization
  (changes the sampled geometry away from the production program and adds
  a template branch only a test uses); sim-side rejection with the model
  prior unchanged (breaks SBC's prior-equality requirement in a region
  holding real prior mass — uninterpretable marginal failures). G3 thus
  validates the production program/geometry/override path exactly, at two
  shifted hyperparameter values; SBC of the production a1/a2 values
  themselves remains impossible in float64 with a plain (non-square-root)
  KF, recorded here as a known limit.

- **2026-08-31 — S2 run-hash stability sacrificed for a single KF
  implementation (user decision resolving a conflict in plans/S3-plan.md).**
  The plan demanded both "generalize `kalman_loglik_tv.stan`, don't fork"
  and "the empty-`sv_shocks` render stays byte-stable so S2 run hashes
  don't change" — jointly impossible, because the run hash covers the
  rendered source, which INLINES the included function files: any edit to
  the KF text changes every no-SV render's hash. Chosen: one generalized
  filter (array-of-R_t core + a thin constant-R overload delegating via
  rep_array; Python mirror likewise via `_as_R_path`), accepting that
  re-running an S2 spec now produces a new run hash (old run dirs remain
  valid immutable records; G1 at 5.5e-12 on both filter paths and G5a at
  ~1e-12 prove the constant case is numerically unchanged). The
  byte-stability requirement is re-scoped to what it can mean and what
  actually matters: from the S3 baseline onward, the no-SV render is
  pinned byte-for-byte against `tests/fixtures/render/lw_sv_no_sv.stan`
  (`test_no_sv_render_is_byte_stable`), so the SV conditionals — and any
  future edit — can never leak into the no-SV render unnoticed; changing
  that fixture requires a recorded decision. A numerics-reviewer pass over
  the generalization found no defects (one noted non-issue: the constant-R
  overload allocates a T-array per likelihood evaluation in no-SV models —
  accepted cost of the delegation design).

- **2026-08-31 — S3 SV stage landed: template conditionals, schema, OLS
  anchor; no-SV render pin regenerated once (comment-only header change).**
  Design points: (1) the template treats `sv_shocks` as a single boolean
  conditional — the schema admits only `[]` and the canonical `[is, pc]`
  (normalized from any order so run identity is order-independent;
  single-shock SV rejected as unvalidated in v1). (2) With SV on, the
  constant `sigma_is`/`sigma_pc` parameters are REPLACED, not shadowed;
  h paths are built non-centered in `transformed parameters` (h_0 = mu_h0
  + sd·h0_raw, h0_raw ~ std_normal; observation t uses h_t = h_0 +
  σ_h·Σν) so the draws carry the authoritative log-variance paths for S4's
  outputs. (3) `build_stan_data` computes the HLW-exact mu_h0 OLS anchors
  unconditionally for lw_sv (CmdStan ignores unused data; the anchors are
  deterministic functions of the trimmed data so run identity is
  untouched); on the US 1960–2019 window they imply OLS residual sds 0.75
  (IS) / 0.82 (PC) — PC essentially on HLW's MLE σ_π ≈ 0.80, IS above the
  MLE σ_ỹ ≈ 0.34 exactly as HLW's own linear-detrend initialization
  behaves. (4) The no-SV render fixture was regenerated ONCE in this
  stage: the template header comment now documents both variants
  (a comment-only change — verified 0 non-comment diff lines against the
  prior pin). The pin is expected to stay stable from here.

- **2026-08-31 — G3 PASSED: SBC, no-SV variant, 200 replications.**
  `pytest -m slow tests/test_g3_sbc.py` — 3h37m wall. Per-parameter χ²
  uniformity (20 bins, 10 expected/bin): p-values 0.073 (a2) to 0.735
  (σ_z), all ten parameters comfortably above the 0.001 floor with a
  healthy spread (no clustering at either extreme). Sampler health: 5
  divergent transitions in 200 × 3,000 = 600,000 post-warmup draws
  (ceiling 600). Rank histograms + per-rep CSV archived from
  `tests/artifacts/g3_sbc/` (gitignored; regenerable — seeds fixed at
  G3_SEED_BASE = 20260901). Configuration per the recorded decisions:
  production template + default priors with the a1/a2 override
  N(0.8, 0.1²)/N(−0.25, 0.05²) through the production override path;
  ranks from 1,500 pooled draws thinned to 99. The prior-to-posterior
  pipeline (template, KF likelihood, priors-as-stamped, NUTS) is
  calibrated end-to-end for the no-SV variant.

- **2026-08-31 — S3 acceptance (G4-precursor) PASSED on the first attempt:
  full SV model on US data, run `70ad47166eaf`.**
  `examples/us_lw_sv/spec_sv.yaml` (sv_shocks: [is, pc], 1961Q1–2019Q2,
  4 chains × 1500/1500, DEFAULT adapt_delta 0.95 — no retuning needed):
  verdict **PASS** — 0 divergences, 0 treedepth hits, E-BFMI 0.88–1.01,
  max R-hat 1.004, min bulk/tail ESS 2628/1620. The expected σ_h funnel
  never materialized as a sampling problem: the non-centered
  parameterization plus data-supported volatility variation (σ_h,IS
  median 0.25 [0.14, 0.42]; σ_h,PC 0.22 [0.14, 0.32] — 5th percentiles
  well off zero) keeps the mass away from the funnel neck. Volatility
  paths are economically right on cue (spec §3.1): exp(h_PC/2) peaks
  ≈1.5 at the 1974 oil shock; exp(h_IS/2) decays from ≈1.1 (late 1970s)
  to 0.22 by 2019 — the Great Moderation. Notable posterior shifts vs
  the S2 no-SV reference (24b6288dddad): σ_y* 0.24 (vs 0.54), a_r −0.047
  (vs −0.070), b_y 0.036 (vs 0.073) — time-varying measurement variances
  reallocate what the constant-scale model forced elsewhere; recorded in
  `examples/us_lw_sv/README.md`.

- **2026-08-31 — S4 plan's three open questions resolved (user decisions,
  start of S4 implementation, full rationale in `plans/S4-plan.md`).**
  (1) **Simulation smoother algorithm**: the literal two-pass
  Durbin-Koopman smoother (spec §2.4), not the FFBS alternative that was
  proposed with rationale (FFBS reuses `_rts_smooth`'s `J_t` directly at
  half the per-draw cost) — user chose to implement the spec's named
  algorithm as written. Per draw: simulate a "plus" state+observation path
  from the unconditional model at that draw's own (F, Q, A, Z, R_t),
  filter+RTS-smooth both the real data and the plus path (reusing the
  existing `kalman_smoother`), combine via `xi_draw = xi_smooth -
  xi+_smooth + xi+`. Structural shocks for HD still fall out algebraically
  from consecutive drawn states (no separate shock-smoother needed).
  Validation: Monte Carlo mean/variance convergence to `kalman_smoother`'s
  output, plus a deterministic zero-plus-noise check that the DK
  combination step reduces exactly to `xi_smooth` (a code-level mirror
  independent of RNG, standing in for G1's Stan-vs-Python mirror since
  §2.4 is Python-only). (2) **Smoother-draw thinning**: benchmark first —
  default `outputs.smoother_draws: all`, measure wall-clock on the
  regenerated `spec_sv.yaml` run (6,000 draws) before writing any plotting
  code against it; only switch the example specs' default to a thinned
  value (proposed `thin: 10` if needed) if the full pass measurably
  exceeds ~2 minutes, recorded here with the actual number once measured.
  The DK choice in (1) roughly doubles the per-draw cost versus FFBS,
  making this benchmark more likely to bind. (3) **IRF 5×5 grid
  columns**: `gap, π, r*, y, g` (rows/shocks are already fixed by spec
  §1.4's five named shocks) — chosen so every shock has at least one
  column showing its own structural role (ε_IS→gap, ε_PC→π, ε_g→g,
  ε_z→r*, ε_y*→y); `z`'s own path was dropped in favor of `r*` (their sum)
  since `z` alone is visually near-identical to `r* − g` and `r*` is the
  object the report/README already center on.

- **2026-08-31 — S3 COVID payoff exhibit delivered: run `9d10bcf32a40`
  (full vintage through 2026Q1, SV on, no hand-set COVID machinery).**
  Diagnostics PASS (0 divergences, 0 treedepth hits, max R-hat 1.005, min
  bulk/tail ESS 1779/917, E-BFMI 0.88–0.99). The four largest exp(h_IS/2)
  posterior medians are exactly 2020Q1–Q4 (peak 3.77 in 2020Q3; 1.38 at
  2019Q4, back to 0.75 by 2022Q1) — the endogenous Bayesian counterpart
  of HLW's hand-set κ variance scaling, discovered from the data rather
  than imposed. Structural de-contamination vs STRESS-TESTS.md §3's no-SV
  absorption channel: σ_y* 0.33 [0.19, 0.42] (no-SV full-vintage: 0.94),
  gap AR a1/a2 = 1.26/−0.29 ≈ the SV pre-COVID values (no-SV had
  collapsed to 1.15). σ_h,IS rises to 0.60 [0.44, 0.78] on this window
  (0.25 pre-COVID) — the RW scale carries the 2020 jump. Full record in
  `examples/us_lw_sv/README.md`. Figures regenerable from the run store
  via the session's sv_run_report script (exp(h/2) medians + 68/90%
  bands).

