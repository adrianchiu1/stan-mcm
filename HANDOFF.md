# Handoff

Status as of 2026-09-05 (end of the S8 session). **S8 WP1-WP3 are
complete and green on branch `claude/s8-grammar-var-extensions-o193mw`
(pushed; not merged, no PR opened, per the brief); WP4 is IN PROGRESS --
see the section right below.** The binding scope record is
`plans/S8-plan.md` (its five brief-vs-doctrine conflicts, the E3 proof,
and the progress record at its end); the measured gate results and the
judgment calls are in DECISIONS.md 2026-09-05.

## S8 IN PROGRESS -- resume here

Done and recorded (DECISIONS.md 2026-09-05; `tests/test_s8_*.py`, marker
`authored`): **WP1** the grammar bundle (E0 zero state shocks / zero
states, E1 intercepts and drifts, E2 contemporaneous exogenous
regressors, E3 shock-free rows with the PD rule, E5 contemporaneous
observables -> recursive VARs with non-diagonal R through measurement
loadings), every extension gated against a hand-built oracle, the
compiled Stan programs against the Python mirror at 50 prior draws, and
the E5 fit oracle against OLS on the handbook data; **WP2** `au.var`,
`au.minnesota_priors`, the Villani steady-state form (closed-form
oracle), FEVD in the generic layer (+ the `fevd` output module);
**WP3** `macrotoolkit.postprocess` (sign restrictions, Waggoner-Zha
conditional forecasts) with tests on synthetic draws and adapters over a
real run. One numerics-reviewer pass over WP1 (no must-fix; three
should-fix applied). **WP4** `examples/handbook/`: `make_data.py`
(all Chapter 1-3 files -> CSV, dates reconstructed and recorded),
`build_specs.py` (nine specs generated through the authoring API),
`run_smoke.py` (fit + fast tier + post-processors -> README records).
Smoke records exist for `ch1_ar2`, `ch1_ar2_ar1err`, `ch2_bivar_minnesota`,
`ch2_steady_state`, `ch2_conditional`, `ch3_uc_trend_cycle`,
`ch2_var4_monthly_cholesky` (see each README's "Smoke run record").

**Not done -- resume here:** (1) the two large smoke runs,
`ch2_signs_11var` (319 parameters; the sign-restriction post-processor
call is in `run_smoke.py`) and `ch3_dfm_uk_panel` (m = 40, n = 6; the
40x40 innovation Cholesky per period makes NUTS slow) -- run
`python examples/handbook/run_smoke.py ch2_signs_11var ch3_dfm_uk_panel`
when a machine-hour is available and check the records in; if the DFM's
mirror check or diagnostics disappoint, that is the finding to record
(short chains are stated as smoke, not evidence). (2) The FAVAR's rate
block (the policy rate as an observable inside the factor VAR) needs
the E5 substitution on the STATE side, which the grammar rejects
("shock loadings must be numeric") -- an S9 item next to E4. (3) The
handbook-example notebook (none was built; the specs + READMEs are the
walkthrough). (4) `examples/handbook/README.md`'s table is the index;
the run ids recorded there are smoke runs at short chains -- every
record says so.

## Warnings for whoever builds S9

- **E4 (data-dependent measurement loadings, the TVP regression of
  handbook §3.2's first example and the TVP-VAR of example 3) is next**
  and is NOT a grammar tweak: `Z_t` varies with data, so the KF core
  (`kalman_loglik_tv.stan` + `_kf_core`) needs a `Z_t` path exactly as
  `R_t`/`Q_t` were added (the S3/S6 playbook: array-of-matrices core,
  constant overloads, G1 mirror, the run-hash consequence for every
  family whose render inlines the filter -- see DECISIONS 2026-09-04
  WP2a for how that was taken last time). Budget the hash re-identification.
- **The mirror gate is now `|diff| < max(1e-8, 1e-11 |loglik|)`**
  (`qc.mirror_rtol`). Flat-prior draws reach |loglik| ~ 1e7-1e9 where an
  absolute 1e-8 is below float64 resolution; the relative term is inert
  for every hand family (|loglik| ~ 1e2-1e3). Do not loosen it further:
  a wide prior on a recursive VAR's contemporaneous coefficients (`a0`)
  makes the innovation Cholesky lose ~4 digits per recursion level --
  `au.var`'s `a0_sd` default is 1.0 for exactly that reason (the
  4-variable example failed the gate at N(0, 10)). If a model fails the
  relative gate, look at the conditioning of R = M D M' before touching
  the tolerance.
- **The `_const` unit state** (a transition drift) has exactly zero
  variance: the RTS smoother's masked-Cholesky branch exists for it
  (`smoother._rts_smooth`; condition = an exact 0.0 predicted diagonal).
  Do not give it a tiny variance "to be safe" -- the branch is exact and
  the pre-S8 arithmetic is untouched only because the condition is exact.
- **The HD `init` bar carries the drift** (labelled "incl. drift"); the
  `const` bar is the MEASUREMENT intercept's column. A shock-free row has
  no measurement bar; its identity is `Y = C + tau` per period.
- **`measurement_loadings` is `None` for every hand family and every S7
  model**; every consumer branches on `M is None` to the pre-S8 path.
  Keep it that way -- the full suite's bit-identity for lw_sv/ucsv
  depends on it.
- **A flat prior has no stationary draws**: the fast tier's HD-identity
  gate raises (FAIL, not skip) for `N(0, 10)` VAR coefficients. Use the
  Minnesota prior (or any prior with stationary mass) for anything the
  fast tier must pass; `ch2_conditional` was moved for this reason.
- **Prior-predictive figures exclude explosive paths and COUNT them in
  the title** (a flat-prior VAR's prior mass is mostly explosive); the
  rule (non-finite or > 1e3 x the data scale) is on the figure.
- Term order is still meaning; the Const column is x column 0.
- All S7 warnings below still apply; the S7 stage record follows
  unchanged.

## What S7 delivered (all green)

**Equation-level model authoring** (`macrotoolkit.authoring`, canonical
import `from macrotoolkit import authoring as au`; the spec-side grammar
and structural derivation in `specs/schema/equations.py`,
`specs/schema/authored_structure.py`, `specs/schema/authored.py`):

- **The authored system is DATA.** `model.family: authored` with
  `model.options` = the model definition (observables, exogenous series,
  measurement + transition equation STRINGS, the prior table, shock
  declarations -- constant scale or the established non-centered SV block
  -- explicit initial conditions, optional per-exogenous forecast rules).
  Equations are parsed, linearized and CANONICALIZED at spec-parse time
  (formatting never reaches the hash; term order does, deliberately --
  plan open question 2), and the whole definition enters the run-identity
  hash through `to_estimation_yaml`. `au.Model(...)` and the small
  helpers (`normal`, `half_normal`, `beta`, `shock`, `sv`, `first_obs`,
  `init`, `forecast`, `log_var_diff`) build the SAME dict a YAML spec
  carries; `mtk.spec("authored", options=model, ...)` accepts the Model.
- **Compiled to the EXISTING contract.** `CompiledModel` (cached per
  canonical definition) exposes the `StateSpaceMeta` (slots with time
  offsets derived mechanically: a lagged LHS `g[-1] = g[-2] + eta_g`
  carries a state lagged, HLW's timing; head-slot substitution for
  contemporaneous state references, in topological order; lag-copy rows;
  feedback map in first-appearance order), numeric `(F, Q, A, Z, R)`
  builders that perform the operations the template performs in the
  same order (`Q = Σ var_s b_s b_sᵀ` accumulated in shock order; a
  measurement shock's SV through `R_t`, a state shock's through the S6
  `Q_t`), regressors with the feedback map's depth as pre-sample rows,
  explicit `(xi00, P00)`, the `log_var_diff` anchor, prior resolution
  with the hand families' override discipline, the prior sampler, and a
  stationarity filter (spectral radius of `F` and of the feedback
  companion ≤ 1: unit roots pass, explosive roots fail).
- **ONE generic template** `stan/templates/authored.stan.j2`, fully
  stamped (constant matrices in `transformed data`, parameter-dependent
  ones in `transformed parameters`, `kf_loglik` as a transformed
  parameter per S6's mechanism; no runtime branching). **No shared Stan
  text and no existing family template or spec was edited**; every
  existing example spec's rendered-source and estimation-identity hashes
  are pinned to a fixture captured on the untouched baseline
  (`tests/test_existing_family_hashes_pinned.py`).
- **The oracles gate (the stage's G1-equivalent).** The three hand
  families expressed as equations (`tests/authored_oracles.py`): compiled
  meta == hand meta EXACTLY for ucsv and local_level and, after the
  declared shock-label map, for lw_sv (plan conflict item 2); matrices at
  50 prior points to <1e-15 (F bit-identical; Q/R paths bit-identical);
  regressors/initial state/anchors bit-identical; the compiled program's
  `kf_loglik` vs the HAND template's at 50 prior draws: **max |diff| =
  0.0** for lw_sv no-SV and SV and ucsv no-SV and SV (identical
  floating-point operations), and vs the Python mirror 9.1e-13 /
  2.3e-12 / 2.1e-12 / 1.1e-13; local_level (no hand KF; plan conflict
  item 1) vs the mirror 1.8e-12.
- **Pipeline integration.** One registered family `authored` (plan
  conflict item 3) with two additive registry changes:
  `FamilyEntry.dynamic_required_mapping` (required `data.mapping` keys =
  the model's observables + exogenous series) and `run.build_stan_data`
  passing the spec to builders that take it (signature-inspected; hand
  families untouched). `mtk.fit` runs the fit-time mirror check on the
  authored program like any other; `Run.param_table`, `report`, `sweep`
  (`prior_sd_table`/`headline_series` capabilities) all generic.
- **Generic output layer** (`authoring/results.py`, `plots.py`,
  `outputs.py` over `results_core` + the engine; no recursion written):
  `prior_predictive`, `states` (every state's head slot with its offset
  stamped, `exp(h/2)` panel for SV shocks), `irf` (every shock → every
  observable and state), `hd` (per observable; the G6 identity holds at
  ~1e-13 incl. the `exog` data bar), `fan` -- declared with the new
  `OutputModule.available` predicate: omitted, with the reason stated in
  the report and raised by the API, unless every lagged exogenous series
  has a forecast rule (`last_value` | `constant` | `state_linear`).
- **Auto-validation.** `mtk validate <spec.yaml> --tier fast` /
  `mtk.validate(spec, data=df)` build the fast tier from the definition
  (`authoring/validation.py::suite_for`: `mirror` at 25 prior draws of
  the production render + `hd_identity` at stationary prior points);
  `recovery_design` / `sbc_design` are one-call constructors (generic
  structural simulator through the engine at fixed anchors) -- exposed,
  NOT registered or run (per-model work). `run_validation_suite` runs an
  in-memory suite; `mtk validate authored` (the family name) tells you
  to pass the spec.
- **The payoff notebook** `examples/notebook_api/authored_uc_gap.ipynb`
  (built by `build_authored_notebook.py`, committed executed): a
  bivariate UC output-gap model with an AR(2) gap STATE carrying SV on
  the demand shock and an accelerationist Phillips curve -- a model that
  does not exist in the codebase -- authored, estimated on the in-repo US
  data, swept over `sigma_g`, with states/IRF/HD/fan figures, the live G6
  check, and its fast validation tier. Run record in the notebook and in
  DECISIONS.md.

## Warnings from S7 (still binding)

- **Term order in an authored equation is meaning.** It fixes the
  regressor (feedback-map) column order and therefore the rendered `A`
  and the run identity; reordering terms gives an equivalent model with a
  different hash. Formatting does not (canonical re-emission). Do not
  "fix" this by sorting terms -- lw_sv's hand feedback map is only
  reproduced because first-appearance order is preserved.
- **Shock names cannot coincide with series names** (a bare name would
  be ambiguous); lw_sv's hand labels (`"ystar"` for the ystar shock) are
  therefore mapped in the oracle test, not reproduced. Structure is
  compared exactly; labels are user choices.
- **The stationarity filter allows unit roots on both sides** (random
  walks; lw_sv's Phillips curve has lag coefficients summing to one) and
  rejects only explosive roots. An HD identity that "fails" on an
  authored model with an explosive prior draw is the S6 float64 lesson,
  not a bug -- check `CompiledModel.is_stationary` first.
- **`mu_h0` for an authored SV shock is a number or the `log_var_diff`
  rule.** lw_sv's HLW-regression OLS anchor is family-specific and NOT
  expressible; the lw_sv oracle gate supplies it as data. An authored
  model wanting a data anchor uses `au.log_var_diff(series, fraction)`.
- **The recovery/SBC constructors fix anchors** (initial state, `mu_h0`)
  the G3/G4/UCSV way; the generic simulator supplies pre-sample rows as
  zeros and needs exogenous paths supplied. Pre-register the constants
  in DECISIONS.md before running anything (ENGINEERING.md rung 3).
- All S6 warnings still apply: never set `TQDM_DISABLE=1` around a fit;
  a notebook must use `macrotoolkit.run.REPO_ROOT` (never walk parent
  directories); every lw_sv spec re-identified in S6 (see
  `examples/us_lw_sv/README.md`); `results_lw.py` is still the validated
  lw_sv instantiation, not a declaration over `results_core.py`; the
  mirror check needs every parameter of the rendered program in the
  inits (`authoring/family.py::stan_inits_and_h` builds them from the
  compiled parameter list, so a template change must update both).
- Any change to `engine.py`, `smoother.py`, `results_core.py`, the KF
  Stan files, a family's metadata, or now `authoring/compile.py` /
  `authored_structure.py` / `authored.stan.j2` requires a fresh
  numerics-reviewer pass (one ran over the S7 change set; findings and
  disposition in DECISIONS.md).

## Environment (fresh container recipe)

- `uv tool install pytest --with cmdstanpy --with numba --with arviz
  --with pydantic --with jinja2 --with matplotlib --with pyyaml --with
  pandas --with click --with h5netcdf --with openpyxl --with scipy
  --with nbformat --with nbclient --with ipykernel --with-editable .`
  (add `~/.local/share/uv/tools/pytest/bin` to `PATH`; use that venv's
  `python` for scripts).
- CmdStan **pinned 2.36.0** at `~/.cmdstan` (`python -m
  cmdstanpy.install_cmdstan --dir ~/.cmdstan --version 2.36.0`; never
  without `--version` -- `api.github.com` is blocked; ~5-25 min build).
- Fast suite `pytest -m "not slow"`: **382 passed, 0 skipped** at
  stage end (~5-6 min; the S7 gates compile five extra programs
  and run two tiny authored fits). Marker `authored` selects the S7
  tests. Slow: `-m "slow and ucsv"` (G2 ~20 min + SBC ~1-1.5 h,
  resumable), `-m "slow and lw_sv"` (G2 ~38 min, G3 ~3.5-7 h, G4 ~3-5 h
  resumable), the sampling notebooks under `-m slow`.
- Reference runs: `runs-archive/` (S5 fixtures) for the output layer;
  `mtk run examples/us_ucsv/spec.yaml` is a few minutes.

## S1-S6 record (condensed, still green)

lw_sv: G1 (5 paths), G2, G3, G4 (pre-registered SBC), G5a (~1e-12 vs the
HLW oracle), G6, the DK smoother, output modules, the HTML report, `mtk
sweep`, G5b. ucsv: G1, production mirror, G2 PASSED, SBC PASSED at the
pre-registered design, G6. S6: the notebook API, the fit-time mirror
check, `mtk validate`. Details in README.md's ladder table,
`STRESS-TESTS.md`, the example READMEs and DECISIONS.md's dated entries.

## What's next after S8 (the S7 list, still open)

1. **Per-authored-model SBC registration + the promotion/production
   layer.** `authoring/validation.py` exposes `recovery_design` /
   `sbc_design`; the missing pieces are (a) a way to REGISTER a
   pre-registered design for an authored model (a `validation:` block in
   the spec, or a sidecar file keyed by the canonical model id) so `mtk
   validate <spec> --tier sbc` runs it, and (b) the promotion record S6
   already named: which validated program + design a production run may
   use, with the mirror check and gate verdicts attached, scheduled
   re-validation, and the `runs-archive` → published-run workflow. The
   notebook model is the obvious first design to pre-register.
2. **`results_lw.py` migration onto `results_core.py`** (S6/S7 backlog;
   the authored results module shows what a declaration over the core
   looks like for a model with an exogenous block and lagged-carried
   states -- the same shape lw_sv needs).
3. Backlog carried forward: PACF stationarity parameterization for AR
   blocks (now relevant to authored AR states too); intercepts/drifts as
   a constant state (a small DSL extension once a model needs it);
   `estimate_c`; missing observations / exact diffuse init when DFM
   arrives (capability-matrix doctrine).
