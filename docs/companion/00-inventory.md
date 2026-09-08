# Inventory: every handbook example, its verdict, and where it lives

Source: Blake & Mumtaz (2017), *Applied Bayesian Econometrics for Central
Bankers*, CCBS Technical Handbook No. 4, updated 2017
(`docs/applied-bayesian-econometrics-for-central-bankers-updated-2017.pdf`)
with the authors' code and data (`docs/applied-bayesian-economics-code-files.zip`,
`Code2017/CHAPTER1..7`, `APPENDIX`). Verdicts were first derived against
the S7 equation grammar and then *realised* by stages S8 (grammar bundle
E0/E1/E2/E3/E5, the VAR layer, two post-processors) and S9 (E4,
data-dependent measurement loadings). What remains is marked with the
extension it needs.

**Verdict codes.** **A** runs today (a committed spec and a recorded run
under `examples/handbook/`) · **C** needs an extension not yet built
(E6–E8) · **D** outside the linear-Gaussian + stochastic-volatility class
by design (VISION.md non-goals) · **X** algorithm content (Gibbs / MH
mechanics) with no model to estimate — replaced by Part 0.

**Extensions.** Built: E0 zero stochastic state shocks (pure regressions),
E1 native intercepts / transition drifts, E2 contemporaneous exogenous
regressors, E3 shock-free measurement rows (singular R), E5 contemporaneous
observables on the right-hand side (recursive VARs) with the Minnesota
helper and FEVD, the sign-restriction and Waggoner–Zha post-processors,
E4 data-dependent measurement loadings (TVP models). Not built: **E6**
time-varying transition matrices and Primiceri's drifting contemporaneous
relations (TVP-VAR-SV, Chapter 7), **E7** missing observations (mixed
frequency; conditional forecasting by smoothing), **E8** model comparison
(marginal likelihood by bridge sampling, LOO).

## Chapter 1 — Gibbs sampling for linear regression models

Data: `inflation.xls` (US CPI inflation, 1948Q1–2010Q4) →
`examples/handbook/data/ch1_inflation.csv`.

| Script | Model / content | Verdict | Where |
|---|---|---|---|
| example1.m | AR(2) for US inflation with a constant; `B ~ N(0, I)`, `σ² ~ IG(1, 0.1)` | **A** | `ch1_ar2` (E0/E1: a regression with no state; the KF runs with `n = 0`) |
| example2.m | the same model, the 12-quarter forecast distribution | **A** | `ch1_ar2` (`outputs.horizon: 12`, the engine's fan) |
| example3.m | AR(2) with AR(1) disturbances (Chib 1993) | **A** | `ch1_ar2_ar1err` (the exact quasi-differenced equation with coefficient expressions in the parameters) |
| example4.m | convergence: recursive means, autocorrelations | **X** | Part 0 §4 (the diagnostics verdict) |
| example5.m | the Geweke convergence statistic | **X** | Part 0 §4 |
| example6.m, Appendix §5 | marginal likelihood by Chib's method | **C** (E8) | Chapter 1 §5 states the gap |

## Chapter 2 — Gibbs sampling for VARs

Data: `datain.xls` (US GDP growth, inflation, 1948Q1–2010Q4) →
`ch2_datain.csv`; `dataUS.xls` (monthly FFR, 10-year yield, unemployment,
inflation, 2007m1–2010m12) → `ch2_dataus_monthly.csv`; `usdata1.xls`
(11 series, 1971Q1–2010Q4) → `ch2_usdata_11var.csv`.

| Script | Model / content | Verdict | Where |
|---|---|---|---|
| example1.m | bivariate VAR(2), Minnesota prior, independent normal–inverse-Wishart Gibbs, 3-year forecast | **A** | `ch2_bivar_minnesota` (E5 recursive form; `au.minnesota_priors`; *IW replaced*, stated) |
| example2.m | 4-variable monthly VAR(2), tight prior, Cholesky IRFs to a bond-yield shock | **A** | `ch2_var4_monthly_cholesky` |
| example3.m | steady-state prior (Villani 2009) | **A** | `ch2_steady_state` (long-run means as constant states) |
| example4.m | dummy-observation priors (Banbura et al.): Minnesota, sum-of-coefficients, common trend | **A (approximation)** | the Minnesota part exactly; the sum-of-coefficients and co-persistence dummies are not independent-normal priors — stated in Chapter 2 §5 |
| example5–7.m | 11-variable VAR(2), sign restrictions for a monetary policy shock (three algorithm variants) | **A** | `ch2_signs_11var` + `postprocess.sign_restrictions` (Minnesota normals in place of the dummy-observation prior; the 11-variable smoke run is a publication-length run session) |
| example8.m | conditional forecasting (Waggoner–Zha) | **A** | `ch2_conditional` + `postprocess.conditional_forecast` (no data augmentation of the conditional path, stated) |
| example9.m, Appendix §9 | marginal likelihood for a VAR | **C** (E8) | Chapter 2 §8 |

## Chapter 3 — Gibbs sampling for state-space models

Data: `usdata.xls` (US GDP growth, CPI inflation, FFR) → `ch3_usdata_tvp.csv`;
the UK panel (`datain.xls`, `names.xls`, `index.xls`, `baserate.xls`) →
`ch3_uk_panel_levels.csv`, `ch3_uk_panel_index.csv`, `ch3_uk_panel_dfm.csv`;
example 1's artificial DGP simulated once → `ch3_tvp_example1_sim.csv`.

| Script / section | Model / content | Verdict | Where |
|---|---|---|---|
| §2 (2.4)–(2.5) | TVP regression `Y = c_t + B_t X_t + e` | **A** | `ch3_tvp_regression` (E4) |
| §2 (2.6)–(2.7) | unobserved-components trend–cycle, `Y = C + τ` exactly, AR(2) cycle with a constant | **A** | `ch3_uc_trend_cycle` (E1 drift + E3 singular R; orthogonal shocks — the handbook's `Q` off-diagonal is not expressible, stated) |
| §2 (2.8)–(2.9) | dynamic factor model, static loadings, AR(2) factor | **A** | the DFM block of `ch3_dfm_uk_panel` |
| example1.m | the Kalman filter on artificial TVP data | **X + A** | Part 0 §3; the filter is `smoother.py`; the handbook's own loop reproduces the toolkit's filtered path to 1e-10 (`tests/test_s9_grammar.py`) |
| example2.m | Carter–Kohn on the same | **X + A** | Part 0 §3; the Durbin–Koopman simulation smoother |
| example3.m | TVP-VAR(2), constant Σ, time-varying IRFs | **A** | `ch3_tvp_var` (E4 + E5; 21 coefficient states; IRFs conditional on the coefficient state at three dates; the handbook's sign-restricted policy shock is a post-processor over these) |
| example4.m | FAVAR on the 40-series UK panel + Bank Rate | **A (DFM block)** | `ch3_dfm_uk_panel`; the (factor, rate) VAR block with the rate as an observable inside the factor VAR is a follow-up (E5 on the state side) |
| example5.m | mixed-frequency VAR (temporal aggregation) | **C** (E7) | Chapter 3 §7 |

## Chapter 4 — Gibbs sampling for Markov-switching models

| Script | Model | Verdict |
|---|---|---|
| example1–2.m | the Hamilton filter; the backward recursion for the regimes | **D / X** |
| example3.m | Markov-switching regression, two regimes | **D** |
| example4.m | Markov-switching VAR with dummy priors | **D** |
| example5.m | two independent chains (four regimes) | **D** |
| example6.m | time-varying transition probabilities | **D** |
| example7.m | two chains, extension | **D** |

Chapter 4 of the companion states why and what other tools do.

## Chapter 5 — The Metropolis–Hastings algorithm

Data: `inflation.xlsx` (UK price level, 1914Q1–2011Q1) → `ch5_uk_inflation.csv`
(annual inflation, `100(ln P_t − ln P_{t−4})`); `usdata.xls`; `tvardata*.xlsx`.

| Script | Model / content | Verdict | Where |
|---|---|---|---|
| example1–2.m | nonlinear regression `Y = b₁ X^{b₂} + e` by random-walk MH (artificial) | **D / X** | Part 0 §1 (nonlinear in a parameter exponent on data) |
| example3.m | TVP regression by MH on the Kalman-filter likelihood | **A** | `ch3_tvp_regression` (the same model; NUTS over the same marginal likelihood) |
| example4.m | stochastic-volatility model for UK inflation (JPR independence MH) | **A** | Chapter 5 §2: the mean-only form `infl = c + e` under SV (E0) and the UC-SV form (`infl = τ + e`, run `03d419822418`) |
| example5.m | TVP-AR(1) with stochastic volatility | **A** | `ch5_tvp_ar1_sv` (E4 + the SV block; dated IRFs) |
| example6.m, §5 | TVP-VAR with stochastic volatility (Primiceri) | **C** (E6) | Chapter 5 §4 |
| example7.m, Appendix §8 | Gelfand–Dey marginal likelihood | **C** (E8) | Chapter 5 §6 |
| SVAR.m | structural VAR, `A₀` by MH (over-identified) | **A (recursive form)** | Chapter 5 §5 (the recursive form is exactly identified; over-identifying zero restrictions are fixed coefficients) |
| starVAR.m | smooth-transition (LSTAR) VAR | **D** | Chapter 5 §5 |
| thresholdvar.m, thresholdvarNFCI.m | threshold VAR with an MH step for the threshold | **D** | Chapter 5 §5 |

## Chapter 6 — Bayesian estimation of linear DSGE models

| Script | Content | Verdict |
|---|---|---|
| example1.m | the 3-equation New Keynesian model solved by gensys; artificial data; IRFs | **D** |
| example2.m | the likelihood through the Kalman filter on the solved state space | **D** |
| example3–4.m | random-walk MH; marginal likelihood by Gelfand–Dey; the restricted model | **D** |

## Chapter 7 — State-space models with time-varying parameters

| Script | Model | Verdict | Where |
|---|---|---|---|
| example1.m, generate_data.m | DFM with world and country factors, time-varying AR coefficients, stochastic volatility on every shock (Mumtaz & Surico) | **C** (E6) | Chapter 7; the constant-coefficient DFM with SV on the factor and idiosyncratic shocks is the bridge model |

## Appendix — Introduction to MATLAB

Verdict **X**: the OLS, AR and maximum-likelihood exercises are covered by
the companion's Part 0 as the toolkit's Python API.

## Empirical record of the verdicts

- S8 smoke runs (2 × 300/300 unless stated) with the fit-time mirror
  check and the fast validation tier: `ch1_ar2` `d2db589230ea`,
  `ch1_ar2_ar1err` `7a486a179c3c`, `ch2_bivar_minnesota` `4d8fc15de6a1`,
  `ch2_var4_monthly_cholesky` `74381f101aa1`, `ch2_steady_state`
  `0257900173a5`, `ch2_conditional` `b6efa20845d4`, `ch3_uc_trend_cycle`
  `e8c65ebfc66b` (4 × 500/500, PASS), `ch3_tvp_regression` `f48f15521246`,
  `ch5_tvp_ar1_sv` `56e3014fa57a`, `ch3_tvp_var` `2ee4bc9e9b98`
  (2 × 200/200). Every fast tier PASS. The records are in each example's
  README.
- Before S8, the companion author verified the two boundary cases by
  hand on the S7 branch: the Chapter 1 AR(2) with the constant as a
  deterministic state crashed the compiler (zero state shocks — the E0
  finding that S8 fixed), and the Chapter 5 SV model on the handbook's UK
  inflation ran end to end in the UC-SV form (run `03d419822418`,
  4 × 500/500, mirror check 0.0).
