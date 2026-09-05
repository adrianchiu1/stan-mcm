# S9 plan — time-varying-parameter models via data-dependent measurement loadings (E4)

**Status: plan of record, drafted 2026-09-05 at S9 start, on branch
`claude/s9-tvp-loadings-ewmngd` from `origin/main` (S8 merged as PR #9).**
Baseline at stage start: fresh container per the HANDOFF recipe (CmdStan
2.36.0 at `~/.cmdstan`; `xlrd`/`openpyxl` added for the Chapter 5 data),
the fast suite run on the untouched tree before any change (count recorded
in DECISIONS.md at the first commit), and the constant-Z regression fixture
`tests/fixtures/g1/pre_zt_stan_loglik.csv` captured from the UNTOUCHED S8
program (all five G1 paths at the 50 points, 18 significant figures) before
any `stan/` edit -- its three S3 columns reproduce
`pre_qt_stan_loglik.csv` value for value, which is the check that the
capture is faithful. Progress is recorded at the end of this file as each
milestone lands.

Binding conventions inherited unchanged: immutable hash-identified runs,
the validation ladder (ENGINEERING.md), named state metadata, the G1
Stan-vs-Python mirror discipline, a numerics-reviewer pass over every KF /
smoother / engine / matrix-construction change before it is committed, the
dated DECISIONS.md log. **This stage edits shared Stan text** (the filter
core) -- the recorded S3/S6 precedent, taken once (below).

## Goal in one sentence

`Y = c_t + B_t*x_t + e` and `pi = c_t + b_t*pi[-1] + e` -- a STATE
multiplied by a lagged observable or an exogenous series -- compile and
estimate through the existing KF / mirror / smoother / engine / report
machinery: the handbook's Chapter 3 TVP regression (example 1/2 DGP), the
Chapter 5 TVP-AR(1) with stochastic volatility (UK inflation) and the
Chapter 3 TVP-VAR (example 3, US GDP/CPI/R) as authored specs, each
extension landing behind the S7/S8 gates.

## The one design fact everything follows from

A product term `coef * s[-k] * d` (`s` a state, `d` a data column of the
feedback map) puts the DATA in the loading: the measurement equation
becomes `y_t = A' x_t + Z_t xi_t + e_t` with

    Z_t = Z0 + sum_j x_t[j] * Zx_j                                  (E4)

where `Z0` is the constant (possibly parameter-dependent) part every S7/S8
model already has, `x_t[j]` is column `j` of the regressor matrix the
feedback map ALREADY declares (an `ObsLag`, `ObsLagMean` or `ExogLag`
column, lag 0 allowed for an exogenous series via E2), and `Zx_j` is the
sparse coefficient matrix of the products on that column (entry `(row,
slot)` = the coefficient expression). Nothing new is read from the data:
the product's data factor IS a feedback-map column, so the KF, the engine
and the hash see exactly the regressor matrix they already see.

Consequences, in the order the stage builds them:

1. **The shared KF needs a `Z_t` PATH** (the capability matrix's
   "Time-varying `Z_t`: not built" row) -- the S3/S6 playbook: an
   `array[] matrix Z` core, constant overloads delegating via
   `rep_array`, the Python mirror `_as_Z_path` in the filter, the RTS
   smoother's inputs, the DK plus path and the shock recovery.
2. **The grammar** classifies a product of two symbol-carrying factors
   instead of rejecting it, ONLY for the one shape E4 allows; every other
   product keeps its S7 rejection message.
3. **The engine** builds `Z_t` from the path it is simulating (the
   feedback closure now carries time-varying coefficients).
4. **Oracles, examples, docs.**

## Where the brief and the repo's own doctrine meet (decisions)

1. **The run-hash consequence of editing `kalman_loglik_tv.stan`.** The
   rendered source of every family that inlines the filter (lw_sv, ucsv,
   the S7/S8 authored programs) changes text, so every existing spec's
   run identity moves. The repo's record (DECISIONS 2026-08-31, 2026-09-04
   WP2a) takes this ONCE per generalization with (a) a regression pin
   proving constant-case numerics are unchanged on the Stan side
   (`pre_zt_stan_loglik.csv`, diff EXACTLY 0.0 on all five S8 paths),
   (b) the render pins regenerated once with a recorded decision
   (`tests/fixtures/render/s7_existing_family_hashes.json`,
   `tests/fixtures/render/lw_sv_no_sv.stan`), (c) the lineage recorded
   in the example READMEs (old identity -> new identity, computed without
   sampling). The archived reference runs stay valid immutable records
   under their old identities. The S8 handbook smoke records keep their
   (S8-program) run ids with a note; re-running them buys nothing.
2. **Which products are allowed.** Exactly one factor is a state (any lag
   `s` or `s[-k]`, resolved to a slot exactly as a plain state reference
   is), the other is a lagged observable (`y[-k]`, k >= 1), a
   `mean(y[-k], ...)`, or an exogenous series at any lag including 0
   (E2); a coefficient expression over parameters/numbers may multiply
   the pair. Rejected, with the S7 message ("multiplies two series/shock
   terms ...") extended by one sentence naming the E4 shape: state x
   state, shock x anything, data x data, a triple product, and state x
   CONTEMPORANEOUS observable (an E5 substitution of the other row would
   put a state x state product on the row -- nonlinear). A product in a
   TRANSITION equation is rejected as a time-varying transition (E6, the
   next stage). A product cannot be the data factor of a `mean()`.
3. **Where the coefficient of a product lives.** A state's contribution
   through a product is a LOADING, not a regressor: `A` (the constant
   regressor coefficients) is unchanged; `Zx_j[row, slot]` carries the
   coefficient expression, evaluated per draw exactly as `Z0`'s entries
   are (Python `evaluate` of the same tree the template prints). `Z_t`
   is built in `transformed data` when every `Z0`/`Zx` entry is numeric
   (the TVP regression, the TVP-AR), else in `transformed parameters`
   (the TVP-VAR's E5-composed rows carry `a0` coefficients). Operation
   order per entry is fixed on both sides: `z0 + (c1) * x[t, j1] + (c2)
   * x[t, j2] + ...` with columns ascending, a unit coefficient printed
   as the bare column; the Python path accumulates `Z0 + x[:, j] * Zx_j`
   in the same column order, which is the same floating-point operations
   entry by entry (`+ 0.0` and `* 1.0` are exact), so the two sides agree
   at G1's gate, not approximately.
4. **E5 composition.** A row substituted into another (E5) composes its
   `Zx` rows exactly as its `Z0`/`A`/`M` rows compose (coefficient x
   entry, added); the TVP-VAR's recursive form is therefore expressible
   with the constant-Sigma identification E5 already gives.
5. **E3 with E4.** The rank rule `rank([M | Z B]) = m` is evaluated at a
   generic parameter point AND generic regressor values (the products'
   columns get generic values); a shock-free row whose only stochastic
   loading is data-dependent passes generically and the fit-time mirror
   check verifies the Cholesky on the actual data (a row `Y = b*x[-1]`
   with no shock is singular at every `x_{t-1} = 0` -- the model's
   problem, reported by the Cholesky, not silently patched).
6. **The engine: coefficient states are STRUCTURE, additive states are
   COMPONENTS.** For the HD and the IRFs the bilinear term `x_t[j] *
   (Zx_j xi_t)` is linear in the observable path given the coefficient
   path and linear in the coefficient path given the observable path,
   but not in both. The decomposition that keeps the G6 identity exact
   AND closes the feedback loop with time-varying coefficients treats the
   coefficient path as GIVEN (the drawn state path `xi_draw`, the same
   for every bar) and lets every bar feed its OWN observables back
   through it: bar k is `y^k_t = A' x^k_t + sum_j x^k_t[j] (Zx_j xi_t)
   + Z0 comp^k_t + e^k_t`, and `sum_k y^k_t = A' x_t + Z_t xi_t + e_t =
   y_t` by induction on the (linear, time-varying) feedback. The states'
   own decomposition (init + each shock) is unchanged. A state shock
   that can only ever move coefficient slots (its loaded slots reach no
   `Z0` slot through F's pattern -- structural, generic) has no additive
   observable bar and no IRF from rest (a coefficient deviation from a
   zero path multiplies zero); the HD and IRF modules OMIT those shocks
   and state the reason in the caption (the HD identity gate applies
   unchanged, over the bars that exist). IRFs of a model with data
   loadings are conditional on a coefficient state: `outputs.irf_dates`
   (a list of dates; default: the last estimation row) selects the
   reference dates and the response at date tau uses the DK-drawn state
   at tau (per posterior draw), the handbook's example 3 practice
   (`irfsim(btemp)` at each date). The forward simulation (fan, prior
   predictive, the structural simulator behind the recovery/SBC designs)
   is the full bilinear system: `Z_t` from the simulated regressors, the
   coefficient states continuing as their random walks. The fan's
   omitted-with-reason rule is unchanged (every referenced exogenous
   series needs a forecast rule).
7. **The stationarity filter of the HD identity gate** (`is_stationary`)
   takes the feedback companion at `A + Zx_j . xi_ref` with `xi_ref` the
   initial-state mean (the gate passes `xi00`); a model whose coefficient
   prior is centred on explosive dynamics is filtered like an explosive
   constant-coefficient prior draw (the S6 lesson).

## WP1 — the KF `Z_t` path (Stan core + Python mirror + G1 + pins)

- `stan/functions/kalman_loglik_tv.stan`: core signature `(yobs, x, F,
  array[] matrix Q, A, array[] matrix Z, array[] matrix R, xi00, P00)`
  using `Z[t]` at step t; the four S6 signatures (constant Z) become
  wrappers delegating with `rep_array(Z, T)`; three new wrappers for
  array Z with constant Q and/or R. One filter implementation, seven
  one-line wrappers.
- `smoother.py`: `_as_Z_path` (the `_as_R_path`/`_as_Q_path` twin);
  `_kf_core` takes the (T, m, n) path; `kalman_loglik`, `kalman_smoother`,
  `_simulate_plus_path` (y+_t = A'x_t + Z_t xi+_t + e+_t),
  `recover_shocks` (the residual through Z_t), `simulate_smoother_draw`
  accept `Z` as (m, n) or (T, m, n).
- G1 (`tests/g1_harness.py`, the harness template, `tests/test_g1_mirror.py`):
  two new paths -- `loglik_tvz` (a per-point data-scaled `Z_t` path:
  entry (1, 1) of `lw_Z` multiplied by a random-walk factor, constant Q
  and R) and `loglik_tvzqr` (array Z, array Q and array R together --
  the full core signature) -- each Stan-vs-Python at the 50 points; the
  constant-Z regression pin: all five S8 paths reproduce
  `pre_zt_stan_loglik.csv` with diff EXACTLY 0.0.
- Pins regenerated once (`s7_existing_family_hashes.json`,
  `lw_sv_no_sv.stan`), the lineage recorded in `examples/us_lw_sv/README.md`
  and `examples/us_ucsv/README.md`.
- Numerics-reviewer pass over the core + mirror before the commit.

## WP2 — grammar, structure, compiler, template

- `equations.py`: `ProductTerm(state: SeriesTerm, data: SeriesTerm |
  MeanTerm)`; `linearize` takes a finer `kind_of` (``"state"`` /
  ``"observable"`` / ``"exogenous"`` / ``"shock"`` / ``"param"``; the
  legacy ``"symbol"`` still means "a series that may not be multiplied");
  the `Mul` case distributes `(terms_L + c_L)(terms_R + c_R)` and
  classifies each `term x term` pair per decision 2.
- `authored_structure.py`: a product in a measurement row resolves its
  state factor through `need_meas` (a lag beyond the carried head is an
  error, as today) and its data factor through the feedback map (first
  appearance, `lag_depth`); rows carry `Zx` alongside `Z`; E5 composes
  `Zx`; `ModelStructure.Zx: dict[(row, slot), ((column, Expr), ...)]`,
  `coefficient_shocks` (decision 6), `matrix_is_numeric("Z")` covers
  `Zx`; parameter usage covers `Zx`; a product in a transition equation
  is rejected (E6).
- `families/base.py`: `StateSpaceMeta.data_loadings` (optional; the
  sparsity pattern `((obs, label, column), ...)` -- declarative like
  `measurement_loadings`) and `coefficient_shocks`; `None`/empty for
  every hand family and every S7/S8 model, so every consumer's pre-S9
  path is untouched.
- `compile.py`: `build_Zx(params) -> {column: (m, n)} | None`,
  `build_Z_path(params, x) -> (T, m, n)`, `build_matrices(..., x=)`
  returning the path when the model has data loadings (an omitted `x`
  is a hard error then); `is_stationary(params, xi_ref=None)`.
- `authoring/stan.py` + `authored.stan.j2`: `Zt` (decision 3) in
  `transformed data` or `transformed parameters`; the `kalman_loglik`
  call uses `Zt`.
- `authoring/family.py::python_kf_loglik` passes `x`.
- Gates (`tests/test_s9_grammar.py`, `tests/test_s9_stan.py`): the
  classification table (accepted shapes, every rejection with its
  message); the compiled `Zx` pattern and `Z_t` path vs a hand-built
  construction at prior draws; E5 composition of `Zx`; the Stan
  `kf_loglik` vs the mirror at 50 prior draws for the TVP regression
  (numeric `Zt` in transformed data), the TVP-AR(1) with SV (`Zt` +
  `Rt`) and a small recursive TVP-VAR (parameter-dependent `Zt` in
  transformed parameters); the hash round trip.

## WP3 — engine, results, outputs

- `engine.py`: `DataLoadings(Z0, Zx)` with `at(x_t)` and `path(x)`;
  `observable_recursion(..., loadings=, coef_path=)`,
  `simulate_forward(..., loadings=)`; `results_core.py`:
  `DrawMatrices.Zx`, `smoother_draws` builds the KF's `Z_t` path from the
  draw's loadings and the real `x`, `observable_bars(..., loadings=,
  coef_path=)`, `impulse_response(..., loadings=, coef_state=)`,
  `hd_bar_names` minus the coefficient shocks; `authoring/results.py`:
  `matrices_for_draw` carries `Zx`, the HD passes `sim.xi_draw` as the
  coefficient path, IRFs at `outputs.irf_dates` through the DK draws
  (`IRFDraws.reference_dates`, `by_date`), the fan / prior predictive /
  structural simulator pass the loadings; `plots.py` overlays the dated
  IRFs; `specs/schema/authored.py::AuthoredOutputs.irf_dates`.
- Gates: the HD identity (G6's shape) on the TVP-AR(1) at stationary
  prior points and on a DK draw of a fitted run; the zero-noise seam
  (the forward simulation with zero noise reproduces `A'x_t + Z_t xi_t`
  row by row with `Z_t` from the SIMULATED regressors); the dated IRF of
  a measurement shock in the TVP-AR equals the hand recursion
  `y_h = b_tau y_{h-1}` at the drawn `b_tau`.
- Numerics-reviewer pass over the engine/results changes before the
  commit.

## WP4 — oracles

- (a) **TVP regression with Q = 0 vs the constant-coefficient
  regression.** `Y = beta*X + e`, `beta = beta[-1] + eta`: at `sigma_eta
  = 0` and `P00 = 0` the KF log-likelihood equals the E2 constant-
  coefficient program's (`Y = beta*X + e`, beta a parameter) at `beta =
  xi00` EXACTLY (diff 0.0 in the mirror: identical operations); at
  `sigma_eta = 0`, `P00 > 0` it equals the closed-form marginal
  likelihood of the regression with `beta ~ N(xi00, P00)` integrated
  out (matrix determinant lemma, ~1e-10 relative) -- the E0 oracle with a
  regressor, which S8 could not express.
- (b) **Handbook Chapter 3 example 1/2.** The artificial DGP (`Y =
  beta_t X + e`, `beta` a random walk, Q = 0.001, R = 0.01, `beta_0 = 0`,
  `P00 = 1`) simulated at a fixed seed: (i) the handbook's own Kalman
  filter loop (`example1.m`, transcribed) at the true (Q, R) reproduces
  `kalman_smoother`'s filtered `beta_{t|t}` to ~1e-10 -- an independent
  oracle of the filter WITH a `Z_t` path; (ii) the authored model fitted
  by NUTS (short chains) recovers `beta_t` -- the true path inside the
  90% smoothed band at >= 80% of the periods and the smoothed median's
  RMSE against the truth below the filter's own.
- (c) **Parameter-recovery gate (G2's shape) on the TVP-AR(1) with SV**
  through `authoring.validation.recovery_design` (the generic structural
  simulator now runs the bilinear system): a fast-suite smoke at a small
  design plus the registered slow design with its constants recorded in
  DECISIONS.md before the run.

## WP5 — examples (`examples/handbook/`)

- `ch3_tvp_regression`: the example 1/2 DGP as data (`make_data.py`
  simulates it at a recorded seed into `ch3_tvp_example1_sim.csv`, the
  true `beta_t` included as a column the spec does not read), `Y =
  beta*X + e` (X exogenous, lag 0 -- E2 + E4), `beta = beta[-1] + eta`,
  `init(0, 1)`; states figure vs the truth in the README.
- `ch5_tvp_ar1_sv`: Chapter 5 example 5 -- UK annual CPI inflation
  (`inflation.xlsx`, 389 quarterly price-level rows 1914Q1-2011Q1 per the
  script's `TT = 1917.75:0.25:2011` after the 4 lags, the 1 regression
  lag and the 10-observation training sample), `pi = c + b*pi[-1] + e`,
  `c = c[-1] + eta_c`, `b = b[-1] + eta_b`, `e` with SV; initial
  conditions from the handbook's training-sample OLS (stamped by
  `build_specs.py`, recorded).
- `ch3_tvp_var`: Chapter 3 example 3 -- US GDP growth / CPI inflation /
  FFR (`ch3_usdata_tvp.csv`), VAR(2) with a constant in recursive form
  (E5, constant Sigma through `a0` parameters and constant shock scales),
  every coefficient a random-walk state (21 states; one shared
  random-walk scale per equation), `irf_dates` at three dates
  (1975Q1, 1995Q1, 2008Q4), the Cholesky IRFs to the rate shock at each.
- Smoke runs through `run_smoke.py` (short chains, hashes recorded in
  each README); the examples index table extended.

## WP6 — stage end

DECISIONS.md entries (the hash decision, the design decisions above,
gate results, the reviewer passes), the capability matrix row, README's
authoring section, HANDOFF rewritten for S10 (E6 time-varying transition
paths / Primiceri; E7 missing observations), the S9 progress record here.

## Sequencing and commits

WP1 (incl. pins) -> WP2 -> WP3+WP4 -> WP5 -> WP6; each a commit or a few,
pushed; the fast suite green at every commit; a numerics-reviewer pass
before the WP1 commit and before the WP3 commit.

## Open questions, resolved

1. **Does a product change the hash of a model without one?** No: the
   canonical equation text is the same, `Zx` is empty, the template's
   `Zt` blocks are skipped, and `build_matrices` returns the constant
   `Z`; only the shared filter text moves every hash, once (decision 1).
2. **Is `b*x` with `b` a PARAMETER and `x` data still E2?** Yes (a
   coefficient times a data term is linear); the product classification
   only triggers when BOTH factors carry symbols.
3. **Can a coefficient state also enter additively?** Yes (`y = b + b*x
   [-1]` is legal); its additive part is decomposed into bars, its
   coefficient part is held at the drawn path (decision 6); the shock is
   not a coefficient-only shock then.
4. **What does the states figure show for a coefficient state?** Its
   smoothed path with bands, as for any state -- for the TVP regression
   that IS the object of interest (the README overlays the truth).

## Progress record (2026-09-05)

- **WP1 executed** -- the `Z_t` core + seven overloads, the Python
  mirror, G1 over seven paths (max |Stan - Python| 7.3e-12), the
  constant-Z pin EXACT (0.0) on all five S8 paths, the render pins
  regenerated once, the lineage recorded. Decision 1 taken as written.
- **WP2 executed** -- `ProductTerm`, the kind-aware classification with
  the S7 message kept, `Zx` with E5 composition, the generic-regressor E3
  rank check, `coefficient_shocks`, `build_Z_path`/`build_matrices(x=)`,
  `Zt` in the template; the three E4 programs mirror the Python KF at 50
  prior draws (4.6e-13 / 3.6e-12 / 6.6e-11).
- **WP3 executed** -- `DataLoadings`, the bilinear forward simulation,
  the HD at the drawn coefficient path (identity exact), dated IRFs
  (`outputs.irf_dates`), coefficient-only shocks omitted with the reason;
  one gate change (the HD identity gate's relative term, DECISIONS (5)).
- **WP4 executed** -- (a) exact (0.0) and closed-form (1e-9); (b, i) the
  handbook filter loop at 1e-10; (b, ii) the fitted DGP recovers beta_t
  (91% inside the 90% band; smoothed RMSE 0.037 < the filter's 0.052);
  (c) the recovery design pre-registered (DECISIONS) with a 2-dataset
  smoke in the fast suite and the 20-dataset gate under `slow` (not run
  this stage).
- **WP5 executed** -- the three examples with smoke records (each README);
  the TVP-VAR's drift prior tightened to the handbook's magnitude after
  the first smoke (DECISIONS (6)).
- **WP6** -- docs (this record, DECISIONS, HANDOFF for S10, the
  capability matrix, the READMEs).
