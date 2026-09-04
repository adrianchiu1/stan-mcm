# S7 plan — equation-level model authoring

**Status: plan of record, drafted 2026-09-04 at S7 start, on branch
`claude/s7-equation-dsl-og1xvc` from `origin/main` (S1–S6 complete, PR #7
merged; `src/macrotoolkit/api.py` present, so the branch starts from main
as the brief prefers).** Baseline at stage start: fresh container per the
HANDOFF recipe (CmdStan 2.36.0 built at `~/.cmdstan`), fast suite green
(count recorded in DECISIONS.md at the first commit).

Binding conventions inherited unchanged: immutable hash-identified runs,
the validation ladder (ENGINEERING.md), named state metadata, the G1
Stan-vs-Python mirror discipline, a numerics-reviewer pass over every
KF/derivation/matrix-construction change, the dated DECISIONS.md log, and
**no edit to any shared Stan text or existing family template** (existing
family hashes must not change this stage — pinned by a test that renders
every example spec and compares the source hash to a fixture captured at
stage start).

## Goal in one sentence

An economist writes a linear-Gaussian state-space model as equations
(named series, lags, shocks, priors, initial conditions) in a notebook or
YAML; the framework compiles it into exactly the declaration surface a
hand-written family provides (`StateSpaceMeta`, shock loadings, feedback
map, matrix builders, prior table, prior sampler, mirror declaration,
rendered Stan program, output modules, validation suite) and the model
flows through the ENTIRE existing machinery unchanged.

## Where the brief and the repo's own doctrine conflict (repo wins)

1. **"kf_loglik equivalence vs the HAND template for all three families"
   — local_level has no `kf_loglik`.** The S1 toy template direct-samples
   the state path (`mu[1..T]` as parameters, `mu_1 ~ N(y_1, 10^2)`); it
   never calls the shared filter, and its initial condition is on `mu_1`
   (not on a pre-sample `mu_0`), so no explicit `(xi00, P00)` reproduces
   its marginal likelihood exactly at every parameter point. Resolution:
   for local_level the oracle is the Python mirror on hand-built matrices
   (the ladder's rung 1, ENGINEERING.md): compiled meta == a hand-written
   local-level meta (exact), compiled matrices == a hand builder (exact),
   compiled Stan `kf_loglik` == `smoother.kalman_loglik` at 50 prior points
   (1e-8). The lw_sv and ucsv gates compare against the hand templates'
   own `kf_loglik` as the brief asks.
2. **"compiled StateSpaceMeta == the hand-written one (exact)" — shock
   NAMES.** lw_sv names its state shocks after the states they drive
   (`state_shocks == ("ystar", "g", "z")`); the DSL forbids a name being
   both a series and a shock (a bare `g` in an equation would be
   ambiguous). Resolution: the equivalence tests assert exact equality of
   every structural field (state labels/offsets/order, shock ORDER,
   loading coefficients, measurement-shock row alignment, obs/exog names,
   the feedback map) after mapping the authored shock labels to the hand
   family's labels through a declared correspondence; shock names are
   user labels, everything else is structure. Recorded here and in the
   test.
3. **"a registry FamilyEntry constructed at load".** `FAMILY_REGISTRY` is
   keyed by `model.family` and consulted by `ModelSpec`'s validator at
   parse time; capabilities are dotted paths. Resolution: ONE registered
   family, `authored`, whose capabilities are generic functions that
   compile the spec's model definition on demand (cached by the model's
   canonical form). The per-model bundle (`CompiledModel`: meta, matrix
   builders, prior table, render context, prior sampler, mirror
   declaration, forecast rules) IS the "entry constructed at load"; the
   registry stays if/elif-free and static. Two registry additions, both
   additive: `dynamic_required_mapping` (the required `data.mapping` keys
   are the authored model's observables + exogenous series, so the
   registry's static tuple cannot express them) and `build_stan_data`
   receiving the spec (`run.build_stan_data(family, df, spec=...)`;
   hand families ignore it — signature-inspected, no behaviour change).
4. **`mtk validate <authored>`** takes a family NAME today. Resolution:
   `api.validate` / `mtk validate` accept a family name OR a spec path
   (or a `RunSpec`) whose family is `authored`; the suite is then
   auto-instantiated from the spec (fast tier registered; recovery/SBC
   exposed as one-call constructors, not registered — S8+ work per
   model).

## The authoring surface (design + argument)

**Equations are strings; the parsed IR is the data.** Argued: strings are
what an economist writes on a whiteboard, they serialize into YAML with no
machinery, they round-trip, and they hash. A symbolic-object surface
(sympy-like) would need its own serializer AND a parser for the YAML form
anyway, and adds a dependency. So: one grammar, parsed once by a
pure-Python parser living spec-side (`specs/schema/equations.py`, no
numpy: `specs.schema` must stay importable without it), validated inside
the family's Pydantic options model, and **canonicalized by re-emission
from the IR** (fixed spacing, fixed numeric formatting, TERM ORDER
PRESERVED — see open question 2). The canonical strings are what
`to_estimation_yaml` serializes, so the equations enter the run-identity
hash and any change to an equation, prior or SV flag changes the hash.

Grammar (linear in series and shocks; coefficients are expressions over
declared parameters and numbers):

```
equation   := lhs "=" expr
lhs        := NAME | NAME "[" "-" INT "]"        (a lagged LHS is allowed for TRANSITION equations only:
                                                 it declares the state is CARRIED lagged — HLW's g/z trick)
expr       := linear combination with + - * / ( ) and unary minus of:
              NAME            a state / shock (contemporaneous), or a parameter (coefficient)
              NAME[-k]        a state, observable or exogenous series lagged k >= 1
              mean(NAME[-k1], NAME[-k2], ...)   the plain mean of lags of ONE observable
                                                 (exactly ObsLagMean; one regressor column)
              NUMBER
```

Products of two series/shock terms, a series in a denominator, a
constant/intercept term, a contemporaneous observable on a RHS, a
contemporaneous exogenous series, a shock with a lag, `mean()` over
anything but lags of one observable, a state referenced "more current"
than it is carried — every one of these is a hard error naming the
construct and the limitation.

Options (`model.family: authored`, `model.options`):

```yaml
name: bivariate_trend_cycle          # a label (in the hash, shown in reports)
observables: [y, pi]                 # observation-row order
exogenous: [r]                       # optional; each may carry a forecast rule (fan charts)
equations:
  measurement:
    - "y = ystar + c + e_y"
    - "pi = b_pi*pi[-1] + (1 - b_pi)*mean(pi[-2], pi[-3], pi[-4]) + b_y*c[-1] + e_pi"
  transition:
    - "ystar = ystar[-1] + 0.25*g[-1] + eta_ystar"
    - "g[-1] = g[-2] + eta_g"        # carried lagged (HLW): slots (g,-1), (g,-2)
    - "c = a1*c[-1] + a2*c[-2] + eta_c"
parameters:                          # the prior table (the menu the templates already stamp)
  a1: {dist: normal, mu: 1.2, sd: 0.3}
  b_y: {dist: normal, mu: 0.15, sd: 0.1, lower: 0}      # truncation = Stan constraint
  b_pi: {dist: beta, a: 8, b: 2}
  sigma_ystar: {dist: half_normal, sd: 0.4}
shocks:
  eta_ystar: {sd: sigma_ystar}       # constant scale: a declared half_normal parameter
  e_y: {sv: {sigma_h: {dist: half_normal, sd: 0.2}, h0_sd: 1.0,
             mu_h0: {anchor: log_var_diff, series: y, fraction: 0.25}}}   # or a number
initial_state:                       # explicit (xi00, P00): every state, mean + sd
  ystar: {mean: {first_obs: y}, sd: 2.0}
  g: {mean: 3.0, sd: 1.0}
forecast_rules:                      # optional, per exogenous series -- fan charts need all of them
  r: {rule: last_value}              # | {rule: constant, value: 2.0} | {rule: state_linear, terms: {"g[-1]": 1, "z[-1]": 1}}
```

`RunSpec.priors` overrides prior FIELDS per name exactly as for the hand
families (unknown name / unknown field = hard error; SV auto-entries
`sigma_h_<shock>` and `mu_h0_<shock>` are overridable so `mtk sweep`
works unchanged). SV per shock adds the established non-centered block:
parameters `sigma_h_<s>`, `h0_<s>_raw`, `nu_<s>`; transformed
`h_<s> = sv_rw_noncentered(mu_h0_<s> + h0_sd*h0_<s>_raw, sigma_h_<s>, nu_<s>)`;
a measurement shock's SV routes through `R_t`, a state shock's through the
S6 `Q_t` machinery. `mu_h0` is a number or the one data rule the hand
families' anchors reduce to for an authored model (`log_var_diff`:
`ln(fraction * Var(Δ series))`, exactly UCSV's equal split); lw_sv's
HLW-regression anchor is family-specific and is supplied as data in the
equivalence test, not expressed in the DSL.

Python surface: `macrotoolkit.authoring` (`mtk.authoring`) — small
helpers (`normal`, `half_normal`, `beta`, `shock`, `sv`, `first_obs`,
`init`, `Model(...)`) that build the SAME options dict; `mtk.spec(
"authored", options=model, data=...)`. No parallel spec: the options
model is the Pydantic class the registry validates.

## Compilation (M1) — the mechanical derivation

State layout: states in transition-equation order; each state `s` with
head offset `d_s` (0, or the LHS lag) carries slots `(s, -d_s), (s,
-d_s-1), ..., (s, -d_s-(n_s-1))` where `n_s` is the smallest number
making every reference resolvable: a transition reference `s[-k]` needs
`k - 1 - d_s < n_s`; a measurement reference needs `k - d_s < n_s`. Lag
slots are deterministic copies (`F[(s,-j),(s,-(j-1))] = 1`, no shock).

Reference resolution in a TRANSITION equation for row t: `s'[-k]` with
`k == d_{s'}` is the head slot of the CURRENT row → substitute its own
equation (linear composition of its F row and shock loadings; the
dependency graph must be acyclic); `k > d_{s'}` → slot `(s', -(k-1))` of
`xi_{t-1}`; `k < d_{s'}` → error. In a MEASUREMENT equation `s'[-k]` →
slot `(s', -k)` of `xi_t`; observable/exogenous lags → x columns in
first-appearance order (`ObsLag`/`ObsLagMean`/`ExogLag` inferred), the
column order being the feedback map. This reproduces lw_sv's HLW layout
`[y*_t, y*_{t-1}, y*_{t-2}, g_{t-1}, g_{t-2}, z_{t-1}, z_{t-2}]`, its
loadings (`g` → 1.0 into `(g,-1)`, 0.25 into `(ystar,0)`), its `Z`/`A`
entries and its six-column feedback map EXACTLY (the M1 gate).

Matrices: `F`, `A`, `Z` entries are coefficient expressions (a tiny AST:
numbers, parameters, `+ - * /`, unary minus) evaluated in Python for the
mirror and emitted verbatim as Stan expressions; `Q = Σ_s var_s b_s b_sᵀ`
accumulated in shock order (constant or per period); `R = diag(var)` in
observation-row order. Constant matrices (no parameter in any entry) are
classified at compile time and land in Stan's `transformed data`.

Pre-sample rows: the max observable/exogenous lag depth `L` (lw_sv: 4,
ucsv: 0) — the first `L` trimmed rows seed the regressors; `first_obs`
anchors read the first ESTIMATION row (row `L`), exactly `build_stan_data`'s
conventions in both hand families.

Stationarity for the identity gate (the S6 lesson): a prior point is
"stationary" iff the spectral radius of `F` is ≤ 1 (random walks allowed,
explosive roots not) AND the endogenous feedback (the ObsLag/ObsLagMean
columns of `A` as a companion system) has spectral radius < 1.

## Stan generation (M2)

One template, `stan/templates/authored.stan.j2`, fully stamped by the
compiled structure (no runtime branching): functions (the shared filter
always; the SV helpers iff any SV shock), data (`T`, `yobs`, `x` if k>0,
`xi00`, `P00`, `mu_h0_<s>` per SV shock), transformed data (constant
matrices; the empty `matrix[T, 0] x` exactly as ucsv), parameters (the
prior table with constraints; the SV block), transformed parameters
(`h_<s>`, parameter-dependent matrices, the `Q_t`/`R_t` arrays,
`kf_loglik` as a transformed parameter — S6's mechanism), model (priors
stamped; `target += kf_loglik`). Generated identifiers are validated
against Stan's lexical rules and the reserved names the template uses.
The M2 gate: at 50 prior draws on the same data, the compiled program's
`kf_loglik` equals the hand template's (`lw_sv` no-SV and SV, `ucsv`
no-SV and SV) to 1e-8 — via one fixed_param evaluation per program (the
S6 `qc.stan_kf_loglik_at_points` mechanism) at inits mapped between the
two parameterizations — and equals the Python mirror.

## Pipeline integration (M3), auto-validation + outputs (M4)

`specs/schema/authored.py` (`AuthoredOptions`, `AuthoredOutputs`), the
`authored` registry entry, `macrotoolkit/authoring/` (`compile.py`,
`family.py` capabilities, `results.py` + `plots.py` + `outputs.py` over
`results_core.py` and the generic engine, `validation.py`). Identity
tests: same equations → same hash; whitespace/format changes → same hash;
any equation/prior/SV change → new hash; YAML round trip. `mtk.fit` end to
end on a small authored model with the fit-time mirror check recorded;
report renders; sweep over an authored prior; `mtk validate <spec.yaml>
--tier fast`. Output modules (generic): `prior_predictive`, `states`
(every state's head slot with bands, + `exp(h/2)` volatility panel for SV
shocks), `irf` (every shock → every observable and state), `hd` (per
observable: init, each state shock, each measurement shock, exog), `fan`
(only when every exogenous series has a forecast rule; else omitted with
a stated reason — `OutputModule.available` added, default `None` =
always). Family ordering doctrine: `results_core` + engine, no recursion
written here.

## Demo (M5) and stage end (M6)

A committed, executed notebook authoring a model that does not exist in
the codebase — a bivariate UC output-gap model (y = trend + cycle with an
AR(2) cycle as a STATE, a Phillips curve in inflation with lagged cycle,
SV on the demand shock) on the in-repo US data — estimating it, sweeping
a prior, showing state/HD/IRF plots and running its fast validation tier.
Then README (authoring section + ladder note), HANDOFF rewritten for S8,
DECISIONS entries, this plan marked executed.

## Open questions, resolved

1. **Strings vs symbolic objects** → strings + IR (above).
2. **Does term order matter for the hash?** Yes, deliberately. Term order
   in a measurement equation determines the regressor (x) column order,
   which is the feedback-map order and the rendered `A`'s row order;
   equation order determines observation rows and state slots. The
   canonical form therefore preserves term order (it normalizes spacing
   and number formatting only). Reordering terms yields an equivalent
   model whose rendered program differs, and a different identity is the
   truthful outcome (the Stan source hash would differ anyway).
3. **Where do authored priors live?** Declared in `options.parameters`
   (the prior table, in the hash); overridden per run through
   `RunSpec.priors` like every family, so sweeps need nothing new.
4. **Local-level exactness** → conflict item 1.
5. **Shock names** → conflict item 2.
6. **How generic is the fan chart?** Exactly as generic as the engine:
   forecast rules per exogenous series (`constant`, `last_value`,
   `state_linear`); observable/exogenous registers seeded from the run's
   own last rows; SV continuation through the S6 noise models. Absent
   rules ⇒ module omitted with a reason.
7. **Nonlinearities, regime switching, missing data, mixed frequency,
   time-varying loadings, exact-diffuse init, intercepts** → out of scope,
   each rejected with a message naming the limitation; the KF is not
   extended this stage.

## Milestones and commits

M1 IR + compiler + equivalence tests (metas/matrices vs local_level,
ucsv, lw_sv) · M2 template + kf_loglik gate (numerics-reviewer pass before
commit) · M3 schema/registry/hash/fit/report · M4 validation tier,
outputs, sweep · M5 notebook · M6 docs. Fast suite green at every commit;
branch pushed after each milestone; the existing-hash pin test lives from
M1 on.
