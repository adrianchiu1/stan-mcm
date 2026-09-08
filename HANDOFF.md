# Handoff

Status as of 2026-09-05 (end of the S9 session). **S9 WP1-WP5 are
complete and green on branch `claude/s9-tvp-loadings-ewmngd` (pushed; no
PR opened, per the brief).** The binding scope record is
`plans/S9-plan.md` (its seven design decisions and the progress record
at its end); the measured gate results, the hash decision and the
numerics-reviewer disposition are in DECISIONS.md 2026-09-05 (the two S9
entries). The S8 leftovers (the two large handbook smoke runs, the FAVAR
rate block, a handbook notebook) are still open -- see "What's next".

## What S9 delivered (all green)

**Time-varying parameters via data-dependent measurement loadings (E4).**
`Y = c_t + B_t*x_t + e` and `pi = c_t + b_t*pi[-1] + e` -- a STATE
multiplied by a lagged observable, a `mean()` of one observable, or an
exogenous series (lag 0 included) -- compile and estimate:

- **Grammar** (`specs/schema/equations.py::ProductTerm`, `linearize`
  with kind-aware classification; `authored_structure.py`): the product
  is a data-dependent loading `Z_t = Z0 + sum_j x_t[j] Zx_j` on the
  feedback-map column the data factor already declares
  (`ModelStructure.Zx`, `StateSpaceMeta.data_loadings`); E5 composes
  `Zx`; every other product keeps the S7 rejection message (extended by
  one sentence naming the E4 shape); a product in a transition equation
  is rejected as E6. `ModelStructure.coefficient_shocks` /
  `StateSpaceMeta.coefficient_shocks`: state shocks whose loaded slots
  reach no additively-loaded slot through F's pattern.
- **Shared KF `Z_t` path** (`stan/functions/kalman_loglik_tv.stan`:
  `array[] matrix Z` core + seven `rep_array` overloads; `smoother.py`:
  `_as_Z_path`, `_kf_core` with `Z[t]`, the RTS inputs, the DK plus path,
  `recover_shocks` through `Z_t`). G1 over SEVEN paths (<= 7.3e-12); the
  constant-Z regression pin EXACT (0.0) on all five S8 paths
  (`tests/fixtures/g1/pre_zt_stan_loglik.csv`); the render pins
  regenerated once with the recorded decision; every lw_sv/ucsv spec
  re-identified (lineage in the example READMEs).
- **Compiler / template** (`authoring/compile.py::build_Zx`,
  `build_loadings`, `build_Z_path`, `build_matrices(x=)`,
  `is_stationary(xi_ref=)`; `authoring/stan.py::_zt_entries`;
  `authored.stan.j2` `Zt` in transformed data when numeric, else in
  transformed parameters): Stan text and Python path perform identical
  operations entry by entry; the three E4 programs mirror the Python KF
  at 50 prior draws (4.6e-13 / 3.6e-12 / 6.6e-11).
- **Engine** (`engine.py::DataLoadings`, `observable_recursion(loadings,
  coef_path)`, `simulate_forward(loadings)`; `results_core.py`
  `DrawMatrices.loadings`, `observable_bars`, `impulse_response(loadings,
  coef_state)`, `hd_bar_names` minus the coefficient shocks;
  `authoring/results.py` dated IRFs via `outputs.irf_dates`
  (`IRFDraws.by_date`, `reference_dates`, `omitted_shocks`), the HD at
  the drawn coefficient path, the fan / prior predictive / structural
  simulator on the full bilinear system): the G6 identity exact; the
  fast tier applies unchanged.
- **Oracles** (`tests/test_s9_grammar.py`, `tests/test_s9_stan.py`):
  (a) the TVP regression at Q = 0, P00 = 0 equals the constant-coefficient
  regression EXACTLY (0.0) and, with P00 > 0, the closed-form marginal
  likelihood; (b) the handbook's `example1.m` filter loop reproduces the
  KF's filtered path to 1e-10 and the fitted DGP recovers beta_t (inside
  the 90% band at 91% of periods; smoothed RMSE 0.037 vs the handbook
  filter's 0.052 at the true parameters); (c) the TVP-AR(1)-SV recovery
  design (pre-registered constants in DECISIONS.md; a 2-dataset smoke in
  the fast suite; the 20-dataset gate under `slow` RAN and PASSED:
  pooled coverage 0.875, per-parameter minimum 0.70 on the SV scale).
- **Examples** (`examples/handbook/`): `ch3_tvp_regression` (the example
  1/2 DGP simulated once by `make_data.py`), `ch5_tvp_ar1_sv` (UK
  inflation from `inflation.xlsx`, training-sample initial conditions),
  `ch3_tvp_var` (US GDP/CPI/R, 21 random-walk coefficients, constant
  Sigma via E5, IRFs at three dates); smoke records in each README.

## Warnings for whoever builds S10

- **E6 (time-varying transition paths / Primiceri) is next** and is the
  `Z_t` playbook applied to `F`: `array[] matrix F` core + overloads +
  `_as_F_path` + a G1 path + a constant-F fixture pin captured BEFORE the
  `stan/` edit (`tests/fixtures/g1/pre_ft_stan_loglik.csv`, all seven
  paths) + the render pins regenerated once + the identity lineage.
  Budget the hash re-identification again. The engine's
  `propagate_state_shock` and `state_components` assume a constant F --
  with `F_t` the state components are propagated through the path; the
  HD identity still holds by the same induction. Grammar: a product in a
  transition equation is today rejected with a message naming E6
  (`authored_structure.py`, the `ProductTerm` branch of the transition
  loop) -- that is the hook. Primiceri's model also needs a STOCHASTIC
  volatility on the COEFFICIENT shocks' covariance (a full `Q_t` -- S6
  gives the diagonal) and the A_t (contemporaneous) drift; scope it.
- **E7 (missing observations)** is a row-selection filter step in both
  mirrors (`kalman_loglik_tv.stan` + `_kf_core`): per period, keep the
  observed rows of `yobs`, `Z_t`, `R_t` (`A'x_t` too); the DK plus path
  and `recover_shocks` must skip the missing rows; the data loader must
  stop rejecting NaN for a family that declares missing-data support;
  the capability matrix says how (the DFM / mixed-frequency row). G1
  needs a masked path.
- **Coefficient states are STRUCTURE in the outputs** (plan decision 6):
  the HD holds them at the drawn path, IRFs condition on
  `outputs.irf_dates`, and coefficient-only shocks are OMITTED from the
  bars and the IRFs with the reason in the caption. Do not "fix" the
  omission by adding zero bars, and do not decompose the coefficient
  path into bars in observable space (bilinear -- the identity breaks).
  If a user wants "the contribution of the coefficient drift", that is a
  counterfactual (re-simulate at a frozen coefficient), a new output.
- **`build_matrices` needs `x` for a model with data loadings** and
  returns the (T, m, n) path; `DrawMatrices.Z` is the CONSTANT part
  `Z0` and `DrawMatrices.loadings` carries the rest; `smoother_draws`
  builds the KF path from the real `x`. Every consumer branches on
  `loadings is None` to the pre-S9 path -- keep it that way (the hand
  families' bit-identity depends on it, as with `M is None` in S8).
- **`is_stationary(params, xi_ref=)`**: with coefficient states the
  feedback companion depends on the coefficient values; the identity
  gate passes `xi00`. A model whose coefficient prior is centred on
  explosive dynamics is filtered like an explosive draw.
- **The E3 rank rule with E4** is generic (generic parameter AND
  regressor values); a shock-free row loaded only through a data-
  dependent coefficient is singular wherever the regressor is exactly
  zero -- the Cholesky reports it at fit time; nothing patches it.
- **`outputs.irf_dates` are report options** (outside the estimation
  identity, the S5 split); the S8 handbook specs gained the empty field
  on rebuild.
- **The TVP-VAR smoke is slow** (21 states, 27 sampled parameters; see
  its README record for the time) -- budget an hour for a re-run.
- All S8 warnings below still apply (the `_const` unit state's exact
  zero variance; the HD `init` bar carrying the drift; the mirror gate's
  relative term; flat priors and the identity gate). The S7 warnings and
  the S7 stage record follow unchanged.

## Warnings from S8 (still binding)

- **The mirror gate is `|diff| < max(1e-8, 1e-11 |loglik|)`**
  (`qc.mirror_rtol`); do not loosen it -- a wide prior on a recursive
  VAR's `a0` makes the innovation Cholesky lose digits (`au.var`'s
  `a0_sd` default is 1.0 for that reason).
- **The `_const` unit state** has exactly zero variance; the RTS
  smoother's masked-Cholesky branch exists for it (condition = an exact
  0.0 predicted diagonal). Do not give it a tiny variance.
- **The HD `init` bar carries the drift**; the `const` bar is the
  MEASUREMENT intercept's column.
- **`measurement_loadings` is `None` for every hand family and every S7
  model**; every consumer branches on `M is None` to the pre-S8 path.
- **A flat prior has no stationary draws**: the fast tier's HD-identity
  gate raises (FAIL, not skip) for `N(0, 10)` VAR coefficients.
- Term order is still meaning; the Const column is x column 0.

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
  --with nbformat --with nbclient --with ipykernel --with xlrd --with
  openpyxl --with-editable .` (S9: `xlrd`/`openpyxl` for the handbook's
  `.xls`/`.xlsx` data files -- only `make_data.py` needs them)
  (add `~/.local/share/uv/tools/pytest/bin` to `PATH`; use that venv's
  `python` for scripts).
- CmdStan **pinned 2.36.0** at `~/.cmdstan` (`python -m
  cmdstanpy.install_cmdstan --dir ~/.cmdstan --version 2.36.0`; never
  without `--version` -- `api.github.com` is blocked; ~5-25 min build).
- Fast suite `pytest -m "not slow"`: **450 passed, 8 deselected** at
  S9 stage end (~15 min on a contended 4-core container, ~6 min on a
  quiet one; the S8/S9 gates compile a dozen extra programs and run
  several short authored fits, incl. the example-1 DGP fit, the dated-IRF
  TVP-AR(1) fit and the 2-dataset recovery smoke). The slow S9 recovery
  gate: `pytest -m "slow" tests/test_s9_stan.py` (~20 x 2-chain fits). Marker `authored` selects the S7
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

## What's next after S9

1. **E6 -- time-varying transition paths (Primiceri 2005)** and **E7 --
   missing observations / mixed frequency** (the DFM row of the
   capability matrix; Chapter 3 example 5 is the mixed-frequency VAR):
   see the warnings above for the playbook each follows.
2. **The S8 leftovers**: the two large handbook smoke runs
   (`ch2_signs_11var`, `ch3_dfm_uk_panel` -- HANDOFF's S8 section said
   how; budget an hour or more each), the FAVAR's rate block (the E5
   substitution on the STATE side), a handbook-example notebook.
3. **An SBC design for the TVP-AR(1)-SV** (the recovery gate PASSED at
   the pre-registered design, DECISIONS.md 2026-09-05; SBC is the next
   rung -- `authoring.validation.sbc_design` at the same anchors, the
   G3/G4 constants pre-registered first).
4. The S7 list, still open: per-authored-model SBC registration + the
   promotion/production layer; `results_lw.py` migration onto
   `results_core.py`; PACF stationarity parameterization for AR blocks;
   `estimate_c`; exact diffuse init when DFM arrives.
