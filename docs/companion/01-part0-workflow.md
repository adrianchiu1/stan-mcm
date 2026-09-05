# Part 0 — From Gibbs to NUTS: the workflow that replaces the sampler chapters

*Replaces the handbook's algorithmic material: Chapter 1 §3.1–3.2 and
§3.7–3.8 (the Gibbs sampler and its convergence), Chapter 3 §3–5 (the
Gibbs algorithm for state-space models, the Kalman filter and the Carter–
Kohn algorithm in MATLAB), Chapter 5 §2–4 and §6 (the Metropolis–Hastings
algorithm and its convergence).*

## 1. What the handbook's samplers do, and what happens to them here

Every model in the handbook is estimated by writing down conditional
posteriors and cycling through them. The recipe has three recurring
pieces:

- **Gibbs blocks for regression coefficients and variances** (Chapter 1
  §3; every VAR block in Chapter 2). Conditional on the variance, the
  coefficients are normal; conditional on the coefficients, the variance is
  inverse-gamma or inverse-Wishart. The sampler alternates.
- **Carter–Kohn for unobserved states** (Chapter 3 §3, §5). Conditional on
  the parameters, the states are drawn as a block from the Kalman filter's
  output by a backward recursion. Every state-space model in Chapters 3, 5
  and 7 has such a block.
- **Metropolis–Hastings where no conditional is available** (Chapter 5):
  a random-walk or independence proposal, an acceptance probability, and
  tuning of the proposal scale by hand until the acceptance rate is
  reasonable — used for the nonlinear regression, for the stochastic
  volatility paths (the Jacquier–Polson–Rossi date-by-date step), for the
  threshold in a threshold VAR, and for the DSGE parameters in Chapter 6.

macrotoolkit replaces all three with one sampler and one structural fact.
The fact: **every model in the toolkit's class is linear-Gaussian
conditional on a small set of parameters and (where present) the
stochastic-volatility paths.** For such a model the Kalman filter computes
the *marginal* likelihood of the parameters with the states integrated out
exactly — the handbook uses this same likelihood in Chapter 5 §3 (the TVP
regression estimated "by maximising the Kalman filter likelihood" with MH)
and in Chapter 6. The sampler is the No-U-Turn Sampler (NUTS) over that
marginal posterior: no conditionals to derive, no proposal scales to
tune, no burn-in judged by eye.

The three handbook pieces map as follows.

| Handbook | Companion |
|---|---|
| Gibbs blocks for coefficients and variances | NUTS over the joint posterior of the static parameters; the priors are declared per parameter (§2 below) |
| Carter–Kohn draws of the states | the states are integrated out during sampling; **after** sampling, the Durbin–Koopman simulation smoother draws the whole state path once per posterior draw (§3) — the same object Carter–Kohn produces, obtained from the marginal posterior rather than inside the Gibbs cycle |
| MH for the stochastic-volatility paths (JPR) | the log-variance random walks are sampled by NUTS jointly with everything else, in the non-centred parameterisation; no mixture approximation (Kim–Shephard–Chib) and no date-by-date proposal |
| MH for nonlinear or regime parameters | outside the class: the toolkit does not estimate models that are nonlinear in the parameters through the data (Chapter 5's `Y = b₁X^{b₂}`), that switch regimes (Chapter 4, the threshold VARs), or that need a model solution inside the likelihood (Chapter 6). Those chapters say so |
| convergence checks by recursive means, autocorrelations, Geweke (Chapter 1 §3.7–3.8, Chapter 5 §6) | the diagnostics verdict on every run: divergences, tree-depth saturation, E-BFMI, R-hat, effective sample sizes (§4) |

The price of the replacement is that the model must be *declared* rather
than *coded*: the declaration is what the compiler turns into the
state-space matrices the filter needs. The gain is that the declaration is
data — hashed, versioned, reproducible — and that a single validated
filter, smoother and output engine serve every model.

## 2. Writing the model down

A model is a system of equations in named series. The handbook's first
model (Chapter 1 §3.3, `example1.m`), the AR(2) for US inflation,

    infl_t = c + b1 infl_{t-1} + b2 infl_{t-2} + e_t,   e_t ~ N(0, σ²)

is written as

```python
from macrotoolkit import api as mtk
from macrotoolkit import authoring as au

model = au.Model(
    "ch1_ar2",
    observables=["infl"],
    measurement=["infl = c + b1*infl[-1] + b2*infl[-2] + e"],
    parameters={"c": au.normal(0.0, 1.0), "b1": au.normal(0.0, 1.0),
                "b2": au.normal(0.0, 1.0), "sigma": au.half_normal(2.0)},
    shocks={"e": au.shock("sigma")},
)
```

The grammar (`specs/schema/equations.py`) allows names, numbers, `+ - * /`,
lags written `x[-k]`, `mean(x[-2], x[-3], x[-4])` for an average of lags
of one observable, and coefficient *expressions* in the parameters
(`(b2 - rho*b1)*infl[-2]` is legal — Chapter 1 §4 uses it). It rejects,
with a message naming the construct, anything nonlinear in the series
(`ystar*gap`), leads, a shock appearing twice, or a shock shared between a
measurement and a transition equation.

The declaration has five parts, and each maps to a handbook object:

| Part | What it declares | Handbook counterpart |
|---|---|---|
| `observables`, `exogenous` | the data series by name; the CSV columns are mapped in the spec's `data:` block | the `Y`, `X` matrices loaded on line 5 of every script |
| `measurement`, `transition` | the equations; every transition equation's left-hand side is a state | the `H`, `F`, `μ` matrices set up by hand in Chapter 3 |
| `parameters` | one prior per static parameter: `normal` (optionally truncated), `half_normal` for scales, `beta` on [0, 1] | `B0`, `Σ0`, `T0`, `D0` — the prior means, variances, degrees of freedom and scales |
| `shocks` | each shock's scale: a half-normal parameter, or `au.sv(...)` for a log-variance random walk | `σ²` and its inverse-gamma prior; the SV block of Chapter 5 §4 |
| `initial_state` | mean and standard deviation of every state at time 0 | `β0|0`, `P0|0` in the Kalman filter set-up (Chapter 3 §4, lines 21–23) |

The compiler (`authoring/compile.py`) derives the state vector, including
the extra lag slots a reference like `gap[-2]` needs, the shock loadings,
the regressor columns implied by lagged observables (the *feedback map*,
which tells the forecasting and decomposition engine how to close the
loop), the system matrices `(F, Q, A, Z, R)` as functions of the
parameters, and a Stan program. `model.describe()` prints the derived
layout. For the AR(2) there are no states at all: the filter runs with
`n = 0` and the likelihood it computes *is* the Gaussian regression
likelihood the handbook writes in equation (2.2).

### What differs from the handbook, always stated

The toolkit's prior menu is normal, half-normal and beta. The handbook's
inverse-gamma prior on a variance is replaced by a half-normal prior on the
standard deviation; its inverse-Wishart prior on a VAR covariance by
half-normal priors on orthogonal shock scales plus normal priors on the
contemporaneous coefficients of a recursive form (Chapter 2 §1 explains
the correspondence); its dummy-observation priors by the independent-
normal Minnesota prior they mostly encode. Each chapter has a *What differs*
box with the exact substitution. They are substitutions of the prior, not
of the model: the likelihood is the handbook's.

## 3. What the toolkit does with the model

**The spec is the unit of work.** `mtk.spec("authored", options=model,
data=..., sampler=..., outputs=...)` builds a validated specification; the
same content in YAML is the same spec. Its *estimation identity* is the
hash of the model, the data bytes, the rendered Stan program and the
toolchain version — so a run has a name (`d2db589230ea`) that means "these
equations, these priors, this data, this program". Report options (which
figures, how many smoother draws, the forecast horizon) and QC settings
sit outside the identity. Re-running the same spec returns the existing run
rather than sampling again.

**The prior-predictive check.** Before looking at the posterior the
report simulates paths from the priors alone (`outputs.figure("prior_predictive")`).
This is the counterpart of the handbook's advice (Chapter 1 §2) to look
at what the prior implies; here it is a figure on every report, and it is
honest — a prior that admits explosive autoregressions produces visibly
explosive paths.

**The fit-time mirror check.** Every `mtk.fit` evaluates the compiled
Stan program's Kalman-filter log-likelihood at five prior draws *before
sampling* and compares it with an independent Python implementation of
the same filter on the same data; the run fails if they differ by more
than `max(1e-8, 1e-11·|loglik|)`. For an authored model this is the check
that the compiler produced the program you think it did. It is recorded
in `diagnostics.json` and printed in the report header.

**Sampling.** NUTS via CmdStan (pinned 2.36.0), 4 chains by default. The
handbook's 25,000-sweep Gibbs runs with 24,000 discarded (Chapter 1 §3.7)
have as their counterpart 500–1,500 warm-up iterations per chain and an
equal number of retained draws; the retained draws are far less
autocorrelated than Gibbs draws, which is why the effective sample sizes
in §4 are the numbers to read, not the raw counts.

**The Kalman filter and the smoother (Chapter 3 §4–5).** The handbook
codes the filter (`example1.m`) and Carter–Kohn (`example2.m`) by hand on
an artificial TVP regression. The toolkit's filter is `smoother.py`
(mirrored in Stan; the two agree to ~1e-12 on every filter path, the
validation ladder's rung 1). The companion's test suite runs the
handbook's own loop from `example1.m` against the toolkit's filter on the
same simulated data: the filtered paths agree to 1e-10
(`tests/test_s9_grammar.py`). The Durbin–Koopman simulation smoother
plays Carter–Kohn's role: for each posterior draw of the parameters it
simulates an unconditional "plus" path, smooths both the real and the
simulated data, and combines them into an exact joint draw of the whole
state path. Every band on a state, every impulse response, every
decomposition bar downstream is a quantile across those draws, so they
carry parameter *and* state uncertainty.

## 4. Reading a run: the diagnostics verdict

The handbook checks convergence by plotting recursive means and
autocorrelations (Chapter 1 §3.7, `example4.m`) and by Geweke's statistic
(§3.8, `example5.m`). Every toolkit run instead carries a verdict —
`PASS`, `WARN` or `FAIL` — with its reasons:

| Diagnostic | What it detects | Threshold |
|---|---|---|
| divergent transitions | the sampler could not follow the posterior's curvature (a funnel, a boundary); the draws near it are biased | any divergence is reported; the reference runs have none |
| tree-depth saturation | the sampler hit its trajectory-length cap — a badly scaled or strongly correlated posterior | reported when it occurs |
| E-BFMI | poor energy transitions between iterations, the signature of a heavy-tailed or funnel-shaped posterior | below 0.2 |
| R-hat | between-chain vs within-chain variance — the modern form of "have the chains converged to the same distribution" | above 1.01 → WARN |
| bulk and tail ESS | the number of independent draws the autocorrelated chain is worth, separately for the centre and the tails | below 400 → WARN, below 100 → FAIL |

The handbook's Chapter 1 §3.7 shows 500 Gibbs sweeps failing these
checks and 25,000 passing; the companion's smoke runs, at 2 chains × 300
draws, mostly carry WARN for the same reason, and say so in each README.
A WARN is not a bug: it is the instruction to run longer, exactly as the
handbook's Figure 14 is. The chapters report the verdicts as measured.

## 5. Outputs

Everything the handbook computes by hand from the retained draws
(forecast distributions by simulation, Chapter 1 §3.4; impulse responses,
Chapter 2 §6; time-varying responses, Chapter 3 §6) is a module of the
output engine, computed per posterior draw:

- `states` — smoothed states (and SV paths) with 68%/90% bands, the
  handbook's plots of `βt` and `ht`;
- `irf` — structural impulse responses to every shock; for a recursive
  VAR under its ordering these are the Cholesky responses; for a model
  with time-varying coefficients, responses conditional on the coefficient
  state at chosen dates;
- `fevd` — forecast-error variance decompositions from the same responses;
- `hd` — the historical decomposition into every structural shock plus the
  initial condition, with a reconstruction identity checked per period
  and per draw;
- `fan` — forecast distributions by stochastic simulation from each draw's
  terminal state; exogenous series follow declared forecast rules or the
  fan is omitted with a stated reason;
- `prior_predictive` — the prior's implied paths.

Two operations that are post-processing in the handbook are post-
processors here as well: sign restrictions (`postprocess.sign_restrictions`,
Chapter 2 §6) rotate the impact matrix per draw and keep the rotations that
satisfy the restrictions; conditional forecasts
(`postprocess.conditional_forecast`, Chapter 2 §7) compute the Waggoner–
Zha restricted-shock distribution from the responses and the unconditional
forecast.

`mtk.sweep` runs the same spec under halved and doubled prior scales as
ordinary immutable runs and reports the prior-to-posterior contraction of
each swept parameter — the tool the companion uses whenever a prior does
identification work.

## 6. Validation: what "the model is right" means here

The handbook's evidence that a sampler works is that it recovers the
parameters of artificial data (Chapter 3 §4–5, Chapter 4 §6, Chapter 5 §3
all start from simulated series). The toolkit makes that a graded
procedure, `mtk.validate(spec, tier=...)`:

- **fast** — the mirror check at 25 prior draws, and the decomposition
  identity on simulation-smoother draws; seconds; run on every companion
  example;
- **recovery** — 20 simulated datasets from the prior, fitted; the true
  values must lie inside the 90% bands about 90% of the time with no bias
  on the weakly identified scales; the handbook's "the algorithm recovers
  the true parameters" as a statistic;
- **sbc** — simulation-based calibration: 100 replications of draw-from-
  the-prior, simulate, fit, rank the truth among the posterior draws; the
  ranks must be uniform (χ² test per quantity). The design is registered
  in `DECISIONS.md` before the first replication and never adjusted to
  pass. This is the certificate a model earns before it is relied on; the
  toolkit's two hand-written families and the S9 TVP-AR(1)-SV model carry
  one.

## 7. The report

`run.report()` writes one self-contained HTML file: the verdict and its
reasons, the mirror check, the prior-predictive figure, every output
module's figures, the parameter table (posterior median, 90% interval,
R-hat, ESS). It is the artefact to circulate; it references nothing
outside itself.

## 8. Reading the rest of this companion

Each chapter follows the handbook's sections. For every example: the
handbook's model and estimator in two lines; the companion's equations;
the *What differs* box; the run — its identity, verdict, mirror check and
the figures that correspond to the handbook's; and the interpretation.
Examples the toolkit cannot run are stated as gaps with the extension
they need (the inventory defines E6–E8) or as outside the class.
