# S8 plan — grammar and VAR extensions for the CCBS handbook (Chapters 1–3)

**Status: plan of record, drafted 2026-09-05 at S8 start, on branch
`claude/s8-grammar-var-extensions-o193mw` from `origin/main` (S7 merged as
PR #8; `src/macrotoolkit/authoring/` present).** Baseline at stage start:
fresh container per the HANDOFF recipe (CmdStan 2.36.0 at `~/.cmdstan`),
fast suite run before any change (count recorded in DECISIONS.md at the
first commit). Progress is recorded at the end of this file as each work
package lands.

Binding conventions inherited unchanged: immutable hash-identified runs,
the validation ladder (ENGINEERING.md), named state metadata, the G1
Stan-vs-Python mirror discipline, a numerics-reviewer pass over every KF /
smoother / engine / matrix-construction change, the dated DECISIONS.md
log, **no edit to any shared Stan text or existing family template**
(`tests/test_existing_family_hashes_pinned.py` stays green; the ONLY
Stan file this stage edits is the authored family's own
`stan/templates/authored.stan.j2`, which no existing example spec
renders).

## Goal in one sentence

Extend the S7 equation grammar so the Blake & Mumtaz (2017) handbook
examples of Chapters 1–3 (AR regressions with a constant, recursive
Bayesian VARs with Minnesota / steady-state priors, sign-restricted and
conditional VAR forecasts, the UC trend-cycle model, a DFM) are authored
as equations and estimated through the EXISTING KF / mirror / smoother /
engine / report machinery, each extension landing behind the S7 gates.

## Where the brief and the repo's own doctrine conflict (repo wins)

1. **"E5 makes a recursive VAR expressible" — but S7 requires at least
   one transition equation, and a VAR in observables has NO state.**
   The KF core is dimension-agnostic: with `n = 0` the numba mirror
   reproduces the direct Gaussian regression likelihood EXACTLY
   (checked at stage start: max |diff| 0.0 over 20 rows; the RTS
   smoother and the DK plus-path run with a `(T, 0)` state), and Stan
   accepts zero-size matrices/vectors (`matrix[0, 0]`, `vector[0]`) the
   same way the S6 `matrix[T, 0] x` does. Resolution: the "at least one
   transition equation" rule is dropped (`n_state = 0` is the E0 limit
   of "zero stochastic state shocks"); no dummy state is injected, the
   filter is not special-cased. A model with no state AND no
   measurement shock is rejected (nothing stochastic).
2. **"E3: R diagonal may carry zeros" and "E5: orthogonal shocks with
   contemporaneous observables" together mean R is no longer diagonal.**
   Substituting `y1`'s equation into `y2 = a*y1 + ... + e2` puts
   `a*e1 + e2` on row 2: the structural measurement shocks stay
   orthogonal, but the reduced-form `R = M D M'` (M unit lower
   triangular in substitution order, D = diag of shock variances) is
   full. `recover_shocks` today reads "residual row j IS measurement
   shock j" — an lw_sv/ucsv truth that E5 breaks. Resolution: the
   measurement side gets the SAME declaration the state side has had
   since S5 — a per-shock LOADING (`StateSpaceMeta.measurement_loadings`,
   default = the unit self-loading every hand family has, so
   `LW_STATE_META` / `UCSV_STATE_META` are unchanged and their pinned
   behaviour is bit-identical) — and the measurement-shock recovery
   becomes the exact triangular solve the state side already uses
   (`recovery_order`'s discipline). Loadings on the measurement side are
   PARAMETER-DEPENDENT by nature (the contemporaneous coefficient IS
   the parameter), unlike state loadings (numeric, S7's rule): they are
   carried as coefficient expressions in the structure and evaluated
   per draw; the meta declares the sparsity pattern and the own-row.
3. **"Native intercepts: a transition constant becomes a drift" — the
   shared KF carries no drift vector, and the shared Stan text may not
   change.** Resolution: a transition drift is compiled as a loading on
   an IMPLICIT deterministic unit state `_const` (slot `("_const", 0)`,
   `F[_const,_const] = 1`, no shock, `xi00 = 1`, `P00 = 0`), the textbook
   augmentation; `F[s, _const] = drift expression`. The name cannot
   collide with a user state (identifiers must start with a letter).
   The user never declares it; `initial_state` still lists exactly the
   user's states. The HD's `init` bar therefore carries the drift's
   deterministic contribution — stated in the bar's caption.
4. **"E2: extend the engine's exogenous supply to lag 0" — the engine's
   forecast-rule contract resolves a series' LAG-1 value for the period
   being simulated (the S4 timing lesson, structural in
   `simulate_forward`).** Resolution: the contract is kept EXACTLY for
   every series without a contemporaneous reference (lw_sv's `neutral`
   rule and every pinned fan chart are unchanged, RNG order included);
   a series with a lag-0 term in the feedback map is resolved for the
   CURRENT period instead (its lag-1..depth values come from a register
   seeded with real data), so `last_value`/`constant` mean what they say
   and `state_linear` reads the freshly drawn state row (offset −1 slots
   = the previous period; the user declares it, the figure states it).
5. **Term order vs the constant column.** S7's "term order is meaning"
   fixes x-column order by first appearance; a constant has no
   appearance order across equations (one column of ones is shared).
   Resolution: the constant column, when any measurement equation has
   an intercept, is column 0 of x (the handbook's `X = [1, lags]`
   layout); everything else keeps first-appearance order.

## WP1 — the grammar bundle (each item: structural rule + tests + oracle)

Grammar changes (all in `specs/schema/authored_structure.py`; the
tokenizer/parser/linearizer are unchanged — every construct below was
already PARSED in S7 and rejected structurally):

| | Construct | S7 | S8 rule |
|---|---|---|---|
| E0 | no stochastic state shock at all (`c = c[-1]` only) / no transition equation | crashed in `loading_matrix` / rejected | first-class: `B` is `(n, 0)`, `Q = 0` in `transformed data`, no state-shock block; `n = 0` allowed |
| E1 | constant term in a measurement equation (parameter or number) | rejected ("intercept") | a `Const` feedback column (x column 0 = 1); `A[0, row] = expr` |
| E1 | constant term in a transition equation | rejected ("drift") | loading on the implicit `_const` unit state (conflict 3) |
| E2 | `beta*x` with `x` exogenous, contemporaneous | rejected ("strictly lagged") | `ExogLag(x, 0)` in the feedback map; regressors/engine/HD/fan supply the current value (conflict 4) |
| E3 | measurement row with NO shock | rejected ("exactly one") | allowed iff the innovation covariance stays PD: structural check `rank([M | Z B]) = m` at a generic parameter point (proof below; the implemented form, which the numerics reviewer re-derived as the cleaner general condition); `R` carries a zero diagonal entry |
| E5 | contemporaneous OBSERVABLE on a RHS | rejected ("simultaneous") | substituted acyclically in dependency order (the state-side head-substitution rule applied to measurement rows): A/Z/const rows compose; the other equation's shock enters with the substitution coefficient (conflict 2); a cycle is an error naming the observables |

Unchanged fences (still rejected with a message): products of series,
series in a denominator, lagged shocks, `mean()` over anything but lags
of one observable, a state referenced more currently than carried,
shared shocks between the state and measurement sides, a shock written
twice, parameter-dependent STATE loadings, nonlinearities, regime
switching, missing data, mixed frequency, exact-diffuse init,
data-dependent measurement loadings (E4 — S9).

**E3 proof (the KF innovation covariance stays PD).** `S_t = Z P_{t|t-1}
Z' + R_t` with `P_{t|t-1} = F P_{t-1|t-1} F' + Q_t ⪰ Q_t = B D_t B'` and
`R_t = M E_t M'` (`M` the measurement loadings, unit on each own row;
`D_t`, `E_t` the diagonal shock variances, all `> 0`: constant scales
are `<lower=0>` half-normals, SV variances are `exp(h)`). Hence
`S_t ⪰ W diag(E_t, D_t) W'` with `W = [M | Z B]`, and `v' S_t v = 0`
forces `W' v = 0`; so `rank(W) = m` ⇒ `S_t ≻ 0` for every `t ≥ 1` (the
first step already adds `Q_1`; `P00` may be singular) and every SV
realization, because the condition is on coefficient RANK, not variance
magnitude. When every row has its own shock `M` has full row rank and
the condition is automatic (the pre-S8 case); with shock-free rows it
says the rows no measurement shock explains must be explained by
independent stochastic-state combinations. The compiler checks the rank
at a random generic parameter point (rank is generic in the coefficient
expressions; a special point could only lower it). E0 + E3 (no state
shock and a shock-free row) is therefore rejected — nothing stochastic
could explain the row. The mirror gate and the fit-time check then
verify the Cholesky succeeds at prior draws.

**Downstream made first-class for empty `B`, zero `R` rows, non-diagonal
`R`, `n = 0`, the `_const` state and lag-0 exogenous columns:** the
template (`Q` in `transformed data` when no state shock; `R` entries as
`Σ_j coef_ij coef_kj var_j` term lists, mirrored exactly by the Python
`build_R`; zero-size declarations), `compile.py` (`build_R` from
measurement loadings; `initial_state` for `_const`; regressors for lag 0
and `Const`), `families/base.py` (`measurement_loadings`, `Const`
feedback term, lag-0 `ExogLag`, `loading_matrix` for zero shocks, a
measurement `recovery_order`), `smoother.recover_shocks` (measurement
shocks by exact triangular solve through the loadings), `engine.py`
(`Const` column; contemporaneous exogenous resolution; measurement
loading applied to the shock draws), `results_core.py` (`observable_bars`
and `impulse_response` inject a measurement shock through its loading
column; `const` HD bar), `authoring/results.py` (noise models per SHOCK,
seeds for lag 0), `authoring/validation.py` (design constructors), the
prior sampler (unchanged), HD (no state-shock bars when there are none).
Every one of these is a numerics-reviewer item.

**Oracles (WP1 gates, `tests/test_s8_grammar.py` + oracle helpers):**
- E0: the AR(2)-with-constant-state regression's KF log-likelihood equals
  the closed-form marginal Gaussian regression likelihood with the
  constant integrated out under `c ~ N(mean, sd²)` (matrix determinant
  lemma: `y − Xb ~ N(mean·1, σ²I + sd²·11')`) at 50 prior draws, ~1e-10;
  compiled Stan `kf_loglik` == the mirror at those draws.
- E1: (measurement) the same regression with `c` a PARAMETER equals the
  conditional regression likelihood exactly; (transition) an AR(1) state
  with drift `mu` equals the hand-augmented matrices `[c, 1]` and the
  mirror; the two intercept routes are documented ("constant state =
  the intercept is integrated by the KF under its normal prior and
  appears as a smoothed state; parameter = it is sampled, sweepable,
  and has a posterior in the parameter table — use the parameter form
  unless the intercept is meant to be time-varying later or you want it
  out of the sampler").
- E2: `y = b*x + ... + e` regressors equal the hand-built columns; the
  engine's HD/fan/prior-predictive supply the current value; hash and
  mirror gates.
- E3: the handbook §3.2 UC trend-cycle model (`Y = C + tau` exactly,
  `C = c0 + a1 C[-1] + a2 C[-2] + e1`, `tau = tau[-1] + e2`) compiled
  meta/matrices == a hand-built matrix construction following (2.6)–(2.7)
  with the `_const` augmentation; compiled Stan `kf_loglik` == the mirror
  at 50 prior draws; DK draws with `R = 0` rows reproduce `Y` exactly
  (`C + tau = Y` per period, the measurement identity); mirror + HD
  identity gates.
- E5: a bivariate VAR(2) on handbook data (`Code2017/CHAPTER2/DATA/
  DATAIN.XLS`: US GDP growth, inflation, 1948Q1–2010Q4) in recursive
  form with flat-ish priors: posterior means of the reduced-form
  coefficients within Monte Carlo error of OLS and the reduced-form
  `Σ = M D M'` matching the OLS residual covariance; the structural
  impact matrix equals the Cholesky factor of `Σ`; the E5 substitution
  reproduces a hand-substituted equation's A/Z/R exactly.
- Existing S7 oracles (lw_sv, ucsv, local_level: meta, matrices, the
  exact `kf_loglik` gate) unchanged; existing family hashes pinned.

## WP2 — VAR support layer

- `au.var(name, observables, p, *, priors=..., intercept=True,
  ordering=...)` → an `au.Model` of recursive measurement equations
  (`y_i = c_i + Σ_{j<i} a0_i_j*y_j + Σ_k Σ_j b_i_j_k*y_j[-k] + e_i`) with
  the parameter declarations and half-normal shock scales; the ordering
  is the Cholesky ordering (the engine's structural IRFs under it are
  the Cholesky IRFs, per the brief).
- `au.minnesota_priors(data, observables, p, lambda1, lambda2, lambda3,
  lambda4, own_mean=1.0)` → per-coefficient normal priors: own lag k
  `N(own_mean·[k==1], (λ1/k^λ3)²)`, cross lag `N(0, (s_i λ1 λ2 /(s_j
  k^λ3))²)`, constant `N(0, (s_i λ4)²)`, with `s_i` the residual sd of an
  AR(1)-with-constant OLS regression of series i — the handbook's
  example 1 arithmetic (Chapter 2 §2). Documented as the INDEPENDENT-
  NORMAL Minnesota prior (Litterman's diagonal `H` on `vec(B)` with the
  shock scales estimated separately), NOT the natural-conjugate /
  inverse-Wishart prior (§3) nor the dummy-observation prior (§5); the
  correspondence: conditional on Σ, the handbook's Gibbs step draws
  `vec(B) ~ N(M*, V*)` with exactly this `H`; here the same normal
  priors sit on the RECURSIVE-form coefficients (the reduced-form
  coefficient of eq i is the recursive one plus `a0` combinations),
  `a0` gets a flat-ish normal, and Σ's IW prior is replaced by half-normal
  scales on the orthogonal shocks. Stated in the docstring and README.
- FEVD in the generic layer (`results_core.fevd` from structural IRF
  arrays; `authoring.results.compute_fevd_draws`; a `fevd` output module
  for authored models).
- Steady-state (Villani) form as a worked oracle: long-run means as
  constant states (`mu_y = mu_y[-1]` with `init(mean, sd)`) entering the
  measurement equation with coefficient `(1 − Σ_k b_ii_k)` and
  `−Σ_k b_ij_k` — no new machinery; the oracle is the E0 closed form
  with a vector constant.

## WP3 — two post-processors (`macrotoolkit/postprocess/`)

- `sign_restrictions.py`: Rubio-Ramírez/Waggoner/Zha — per posterior
  draw, Haar-distributed `Q` via QR with positive diagonal (the
  handbook's `getqr`), candidate structural IRFs `irf_h Q`, restrictions
  as `(variable, shock, horizons, sign)`, column sign flips allowed,
  rejection sampling with a try cap; retained rotations → IRF bands;
  the handbook's "closest to median" variant (example 7) optional.
  Tests on synthetic draws: retained rotations satisfy every restriction,
  `Q'Q = I`, impact·impact' is invariant (= Σ), the median-closest
  variant returns one retained rotation.
- `conditional_forecast.py`: Waggoner–Zha hard conditions — `R` from the
  structural IRFs, `r = path − unconditional`, restricted shocks
  `N(R'(RR')⁺r, I − R'(RR')⁺R)`, draws through the IRFs added to the
  unconditional forecast. Tests: the conditioned series reproduces the
  conditioning path exactly; with no conditions the mean/covariance
  equal the engine fan chart's (and given the SAME shock draws the path
  equals the engine's deterministic superposition — the zero-noise seam
  doctrine).

## WP4 — handbook examples as specs (`examples/handbook/`)

`make_data.py` (xlrd for the old `.xls`, openpyxl for `.xlsx`) writing
CSVs with an ISO `date` column; one directory per example with
`spec.yaml`, a README recording the run hash, the fit-time mirror check
and the fast validation tier; short chains (smoke). In order of what
WP1–WP3 unlock: `ch1_ar2` (E0/E1), `ch1_ar2_ar1err` (exact substituted
form: `y_t = c(1−ρ) + (b1+ρ) y_{t−1} + (b2 − ρ b1) y_{t−2} − ρ b2 y_{t−3} + e`),
`ch2_bivar_minnesota` (E5 + Minnesota + forecast), `ch2_var4_monthly_
cholesky` (E5, Cholesky IRFs), `ch2_steady_state` (Villani),
`ch2_signs_11var` (E5 + sign restrictions; sampler reduced for smoke),
`ch2_conditional` (E5 + WZ), `ch3_uc_trend_cycle` (E1 + E3),
`ch3_dfm_uk_panel` (the DFM part of example 4 on the 40-series panel;
the FAVAR VAR block if time allows).

## Sequencing and commits

WP1 → WP2 → WP3 → WP4 → stage end; each WP a commit (or a few), pushed;
fast suite green at every commit; numerics-reviewer pass before the WP1
commit (and again if WP2/WP3 touch the engine); the pinned-hash test
lives throughout. At ~2h of work the remainder is assessed and the stage
stops at a WP boundary with an "S8 IN PROGRESS — resume here" HANDOFF
section if needed.

## Open questions, resolved

1. **Where does an intercept's prior live?** A parameter intercept is a
   normal entry of the prior table (sweepable, overridable). A
   constant-state intercept's `(mean, sd)` is its initial condition (in
   the hash through `initial_state`), not a prior-table entry — it is
   integrated by the filter, not sampled.
2. **Is a numeric intercept (`y = 2.0 + ...`) allowed?** Yes: a
   parameter-free constant column entry (lands in `transformed data`).
3. **Do E5 substitutions change the hash of models that never used
   them?** No: nothing in the canonical form or the render context of an
   S7 model changes; the S7 oracle gates and the pinned fixture prove it.
4. **`n_obs` when rows have no shock.** `obs_names` is the row list;
   `measurement_shocks` is the shock list (≤ rows) with each shock's
   own row declared by its loading; `StateSpaceMeta.n_obs` reads
   `obs_names` when given (hand families give both, equal length).

## Progress record (2026-09-05)

- **WP1 executed** -- every grammar row above landed with its oracle;
  the Stan gates pass at 50 prior draws for E0 (constant state; n = 0),
  E1 (drift), E2, E3 (singular R), E5 (full R); the E5 fit oracle passes
  (VAR(2) on the handbook data vs OLS; Sigma; the Cholesky impact). One
  gate rule changed and is recorded (DECISIONS 2026-09-05): the mirror
  criterion is `|diff| < max(1e-8, 1e-11 |loglik|)`. One smoother change
  (the masked-Cholesky RTS branch for exactly-deterministic slots).
  Numerics-reviewer pass: no must-fix; three should-fix applied.
- **WP2 executed** -- `au.var`, `au.var_parts`, `au.minnesota_priors`
  (example 1's H pinned), the steady-state form (closed-form oracle;
  caught a wrong recursive-form intercept at 8% on first run), FEVD.
- **WP3 executed** -- `macrotoolkit.postprocess` (sign restrictions,
  conditional forecasts) with synthetic-draw tests and run adapters.
- **WP4 in progress** -- data converter, spec builder, smoke runner and
  seven of nine smoke records done; `ch2_signs_11var` and
  `ch3_dfm_uk_panel` not yet run (HANDOFF.md says how); the FAVAR rate
  block and a handbook notebook are open.
- Conflict 2's "conflict" resolution held: `LW_STATE_META` /
  `UCSV_STATE_META` unchanged, every hand-family hash pinned, the full
  fast suite green with the change set.
