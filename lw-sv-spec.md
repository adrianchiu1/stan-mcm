# LW-SV Vertical Slice — Specification v1.0

**Project:** Open-source Bayesian state-space macro toolkit (Python + Stan)
**Slice:** Laubach-Williams with stochastic volatility, end-to-end (data → Stan → outputs)
**Status:** Locked decisions incorporated (2026-08-13). Ready for Claude Code kickoff.

---

## 0. Locked decisions

1. Architecture: **Option C** — shared Stan functions library + per-family Jinja templates, driven by a declarative YAML spec. The spec schema is the spine of the whole toolkit.
2. Inflation measure: **core PCE** (quarterly, annualized q/q log difference).
3. Real rate: **pre-constructed by the user** and supplied in the flat file (nominal policy rate minus expected inflation, annualized). The toolkit does not construct it in v1.
4. SV placement default: **IS and Phillips shocks only.** Constant innovation variances on y*, g, z (pile-up controlled by priors, not MUE). SV on trend shocks is a future flag, not v1.
5. Historical decomposition target: **the three observables plus the output gap**, decomposed into the five structural shocks.
6. Out of scope for the toolkit generally: DSGE solution steps, particle filters, regime switching, ALFRED-style vintages.
7. Data ingestion: user-supplied flat file (CSV). No fetchers in v1.

---

## 1. Model definition

### 1.1 Data and units conventions

Quarterly data, T ≈ 200–280 observations typical (US: 1961Q1–present).

| Symbol | Series | Units |
|---|---|---|
| y_t | 100 × ln(real GDP) | log-points |
| π_t | Core PCE inflation | annualized q/q %, i.e. 400 × Δln(P) |
| r_t | Ex-ante real short rate | annualized % |

Convention: **trend growth g_t is expressed in annualized terms** so that r* = c·g + z needs no 4× factor. The potential-output transition therefore uses g/4.

### 1.2 Measurement / structural equations

**IS curve (output gap):**

```
gap_t = a1·gap_{t-1} + a2·gap_{t-2}
        + (a_r/2)·[(r_{t-1} − r*_{t-1}) + (r_{t-2} − r*_{t-2})]
        + ε_IS,t
```
with gap_t ≡ y_t − y*_t and a_r < 0 (enforced by prior support).

**Phillips curve:**

```
π_t = b_π·π_{t-1} + (1 − b_π)·π̄_{t-2:4} + b_y·gap_{t-1} + ε_PC,t
```
where π̄_{t-2:4} = (π_{t-2} + π_{t-3} + π_{t-4})/3, coefficients on lagged inflation sum to one (HLW restriction; implicit unit root in inflation), and b_y > 0 (enforced by prior support).

**Real-rate identity:** the r_t series enters the IS curve directly as data; r*_t is a state-defined quantity.

### 1.3 State transitions

```
y*_t = y*_{t-1} + g_{t-1}/4 + ε_y*,t
g_t  = g_{t-1} + ε_g,t
z_t  = z_{t-1} + ε_z,t
r*_t = c·g_t + z_t
```
`c` fixed at 1.0 by default; spec flag `estimate_c: true` promotes it to a parameter with prior N(1, 0.25²) truncated positive.

### 1.4 Structural shocks (all model-identified; no rotation machinery)

| Shock | Interpretation | Variance |
|---|---|---|
| ε_IS | Demand | SV: exp(h_IS,t) |
| ε_PC | Supply / cost-push | SV: exp(h_PC,t) |
| ε_y* | Potential level | constant σ_y*² |
| ε_g | Trend growth | constant σ_g² |
| ε_z | Other r* (headwinds) | constant σ_z² |

### 1.5 Stochastic volatility (non-centered)

For s ∈ {IS, PC}:

```
h_s,t = h_s,t-1 + σ_h,s · ν_s,t,   ν_s,t ~ N(0,1)   (non-centered: sample ν, build h)
h_s,0 ~ N(μ_h0,s, 1)
```
Shock standard deviation at t is exp(h_s,t / 2) if h is log-variance — **convention: h is log-variance; sd = exp(h/2).** State this once and enforce everywhere (a classic source of factor-of-2 bugs).

### 1.6 Priors (the ones that matter)

| Parameter | Prior | Rationale |
|---|---|---|
| a1, a2 | a1 ~ N(1.2, 0.3²); a2 ~ N(−0.4, 0.3²); joint stationarity check via prior predictive | Hump-shaped gap persistence |
| a_r | N(−0.1, 0.05²) truncated < 0 | IS slope; sign identification |
| b_π | Beta(8, 2) on [0,1] | Weight on first inflation lag |
| b_y | N(0.15, 0.1²) truncated > 0 | Phillips slope; sign identification |
| σ_y* | Half-N(0, 0.4²) | Potential level shock scale |
| σ_g | Half-N(0, 0.03²) | **Pile-up control.** Tight prior keeps g slow-moving; replaces the MUE λ_g machinery. Document prominently. |
| σ_z | Half-N(0, 0.08²) | **Pile-up control**, analogous to λ_z. |
| σ_h,IS, σ_h,PC | Half-N(0, 0.2²) | Log-vol random-walk scales |
| μ_h0,s | N(2·ln(σ̂_OLS,s), 1) | Initialized from a rough OLS pass on data |
| Initial states | y*_0 diffuse-ish N(y_0, 2²); g_0 ~ N(3, 1²) annualized; z_0 ~ N(0, 1²) | Weakly data-anchored |

These are **defaults exposed in the spec schema**; all overridable per run. Prior-sensitivity of σ_g and σ_z must be documented with a small sweep in the example notebook — these priors do identification work.

---

## 2. Estimation architecture

### 2.1 Stan strategy

**Rao-Blackwellized sampler.** Conditional on the SV paths {h_IS,t, h_PC,t}, the model is linear-Gaussian. Stan samples:

- ~10 static parameters,
- 2 × T standard-normal SV innovations (non-centered),

and evaluates the likelihood by a **Kalman filter with time-varying innovation covariance** inside the model block. The linear states (y*, g, z, gap lags) are marginalized out. State dimension ≈ 6; per-gradient KF cost is trivial at T ≈ 260.

**Do not** sample the linear state innovations in Stan for this model. (The toolkit will support that variant later for large DFMs; not here.)

### 2.2 Shared functions library (first entries)

```
stan/functions/
  kalman_loglik_tv.stan     # KF log-likelihood, time-varying Q_t (and optionally H_t)
  sv_rw_noncentered.stan    # builds h path from innovations + scale + initial
  ssm_matrices_lw.stan      # assembles Z, T, R, c, d for the LW form (template-stamped)
```
`kalman_loglik_tv` must handle: time-varying state innovation covariance, exact treatment of the initial state prior (no ad-hoc diffuse hacks — pass mean/cov explicitly), and numerically symmetric covariance updates (Joseph form or explicit symmetrization each step).

### 2.3 Template + spec

Even with one model, the Jinja scaffold ships now. The template stamps: which shocks carry SV, whether c is estimated, lag structure constants. No runtime branching in Stan.

Spec schema (Pydantic, mirrored in YAML) — draft:

```yaml
model:
  family: lw_sv
  options:
    sv_shocks: [is, pc]        # v1: only this combination validated
    estimate_c: false
data:
  file: data/us_quarterly.csv
  date_column: date
  mapping: { y: lgdp100, pi: core_pce_ann, r: real_rate }
  sample: { start: 1961Q1, end: null }   # null = last available
priors: { }                    # empty = defaults from §1.6; any key overrides
sampler:
  chains: 4
  warmup: 1500
  sampling: 1500
  adapt_delta: 0.95
  max_treedepth: 12
  seed: 20260813
outputs:
  horizon: 12                  # fan-chart quarters
  irf_horizon: 20
  irf_vol_reference: end_of_sample   # or sample_mean
  smoother_draws: all          # or thin: k
```

Run identity = SHA-256 over (canonicalized spec, data file hash, Stan source hash, Stan/CmdStan version). Run store: `runs/<hash12>/` containing `spec.yaml`, `data.snapshot.csv`, `draws.nc` (ArviZ InferenceData), `diagnostics.json`, `report.html`, `log.txt`. Immutable once written.

### 2.4 Python-side simulation smoother

One numba-accelerated **Durbin-Koopman simulation smoother (FFBS acceptable alternative)** operating per posterior draw, given that draw's static parameters and SV path. It produces joint draws of the full state path *and* the structural shock path. This single routine feeds:

1. smoothed trend-cycle objects (y*, gap, g, z, r*),
2. smoothed structural shocks → historical decomposition,
3. terminal state draws → fan-chart simulation seeds.

**Validation requirement:** the Python KF log-likelihood must match the Stan KF log-likelihood to ~1e-8 on a grid of test parameter values before anything downstream is trusted. This is the single most important internal consistency check in the slice.

---

## 3. Outputs

### 3.1 Trend-cycle plots

- y with y* band; gap panel beneath with zero line.
- r with r* band.
- g (annualized) with band.
- z with band.
- **Volatility paths:** exp(h_IS,t/2) and exp(h_PC,t/2) with bands. (Great Moderation and COVID should be visible; this plot sells the SV feature.)

Bands: 68% and 90% pointwise credible intervals from smoother draws.

### 3.2 IRF matrix

Grid: 5 shocks × 5 responses (y, π, r-relevant objects: gap, r*, plus y and π levels as appropriate per shock — final row/column selection at implementation, target the economically meaningful 5×5).

Computed analytically from posterior draws of (Z, T, R) by iterating the state transition. **Convention:** one-standard-deviation shock at the reference volatility (`irf_vol_reference`), stamped in the subplot titles. Median + 68/90 bands, `irf_horizon` quarters.

### 3.3 Fan charts

Per draw: simulate forward from a terminal-state draw, propagating (a) linear states, (b) **the SV random walks** (widening bands are the point), (c) measurement construction of y, π, r-relevant observables. r_t is exogenous data in-sample; for forecasting, hold the real-rate *gap* input at a documented convention — v1 default: r_{T+h} = r*_{T+h} (neutral policy) with the convention printed on the chart. Expose `forecast_r_rule: neutral | last_value | user_path` in the schema, implement `neutral` and `last_value` in v1.

Fans: 10/20/…/90 percentile bands, H = `outputs.horizon`, for y (level and 4-quarter growth), π, gap, r*.

### 3.4 Historical decomposition

Per draw, from smoothed shocks: propagate each shock stream through the system in isolation from the initial state; residual line = initial-condition contribution. Stacked bars (posterior-median contributions) for: gap, π, and 4-quarter GDP growth; optional y level. Sum-check: contributions + initial-condition path must reconstruct the smoothed series to numerical tolerance every period (automated test).

### 3.5 HTML report

One self-contained `report.html` per run: header (run hash, spec summary, data span), diagnostics verdict, then §3.1–3.4 figures, then a parameter table (posterior median, 90% CI, R-hat, ESS). This is the v1 "front-end."

---

## 4. Diagnostics (per-run, automated)

- Divergences (count + parallel-coordinates plot if > 0), E-BFMI per chain, max treedepth hits.
- R-hat / bulk-ESS / tail-ESS **grouped**: statics table in full; SV innovation blocks summarized (worst-k display).
- Prior-predictive check figure (gap and inflation paths simulated from the prior).
- A one-line PASS / WARN / FAIL verdict with reasons, written to `diagnostics.json` and the report header.

---

## 5. Validation gates (hard, ordered)

| Gate | Test | Pass criterion |
|---|---|---|
| G1 | Python KF vs Stan KF log-likelihood, 50 random parameter points | max abs diff < 1e-8 |
| G2 | Parameter recovery on simulated data (no SV), 20 datasets | true values inside 90% CI ≈ 90% of the time; no systematic bias in σ_g, σ_z |
| G3 | SBC, no-SV variant | uniform rank statistics (visual + χ² check) |
| G4 | SBC, full SV variant | same |
| G5a | **Exact HLW replication (fixed parameters):** fix system matrices at HLW's published MLE parameter values, run our KF + smoother on their published US input data | smoothed r*, g, gap match HLW's published one-sided and smoothed series to numerical tolerance (< 1bp after accounting for any documented spec detail differences). Any residual discrepancy must be traced to a specific, documented specification difference — never waved away |
| G5b | **Bayesian tracking:** SV off, HLW-like (loose) priors, their published data | posterior median r* and gap track published HLW within ±50bp / ±0.5pp over 2000–2019 (documented figure) |
| G6 | HD reconstruction identity | exact to 1e-6 per period per draw |

**HLW fixtures policy:** the NY Fed's published HLW R code, input data, and output series are checked into `tests/fixtures/hlw/` (with source URLs and retrieval dates) and used as regression fixtures wherever a component can be tested against them — the KF likelihood at their parameter values, the smoother output (G5a), and the measurement/transition matrix construction (unit test: our `ssm_matrices_lw` output at their parameters equals matrices reverse-engineered from their code). HLW is the external oracle for every linear-Gaussian component of this slice.

G5 is the external credibility exhibit; its figure goes in the README.

---

## 6. Repository layout

```
macrotoolkit/
  specs/schema/           # Pydantic models; lw_sv fragment
  stan/functions/         # shared library (§2.2)
  stan/templates/lw_sv.stan.j2
  src/macrotoolkit/
    data.py               # CSV load, mapping, transforms, sample trim, hash
    render.py             # Jinja → .stan, source-hash compile cache
    run.py                # CmdStanPy execution, run store
    smoother.py           # numba DK simulation smoother + KF (mirror of Stan)
    results_lw.py         # LWResults: trends, irfs, fans, hd
    plots.py              # plotting grammar
    report.py             # HTML assembly
    cli.py                # `mtk run spec.yaml`, `mtk report <hash>`
  tests/                  # gates G1–G6 as pytest suites (SBC gated behind a slow marker)
  tests/fixtures/hlw/     # NY Fed HLW code, data, published output series (the external oracle)
  examples/us_lw_sv/      # spec + data schema doc + worked notebook incl. σ_g/σ_z prior sweep
```

---

## 7. Build stages and acceptance tests

| Stage | Content | Acceptance |
|---|---|---|
| S1 | Spec schema, run store, render/compile/run harness; trivial local-level model end-to-end | `mtk run` produces immutable run dir with draws + diagnostics for the toy model |
| S2 | LW without SV: template, KF functions, priors; Python KF mirror; HLW fixtures wired in | G1, G2, **G5a** pass; posterior sensible on US data |
| S3 | Add non-centered SV block; retune (adapt_delta, priors) | G4-precursor: clean sampling (0 or explainable divergences) on US data; G3 pass |
| S4 | DK smoother; four output modules; HTML report | G6 pass; report renders all figures from a real run |
| S5 | Full SBC (G3, G4), Bayesian HLW tracking (G5b), prior-sweep notebook, docs | All gates green; README with G5a/G5b figures |

Stages are strictly sequential; each gate blocks the next stage. Claude Code model allocation per your established pattern: Opus/Fable for S1 schema design review and all Stan geometry debugging; Sonnet for implementation passes.

---

## 8. Effort estimate (unchanged from design session)

16–24 focused dev-days (~4–6 weeks part-time). Risk concentrates in: (i) SV geometry / divergence hunting in S3; (ii) σ_g, σ_z prior judgment (research time, not code time); (iii) the DK smoother correctness (mitigated by G1's KF mirror requirement).

---

## 9. Explicitly deferred (recorded so scope stays honest)

SV on trend shocks; Student-t measurement errors / outlier-robust SV (Antolín-Díaz-style fat tails — natural S6); time-varying Phillips slope; estimate_c default-on; FastAPI service + React front-end; second model family (DFM or VAR-SV — the schema and run store are built to receive them).
