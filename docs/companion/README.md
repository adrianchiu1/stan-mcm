# Applied Bayesian Econometrics for Central Bankers — the macrotoolkit companion

A companion to Blake & Mumtaz, *Applied Bayesian Econometrics for Central
Bankers* (CCBS Technical Handbook No. 4, updated 2017; the PDF and the
authors' MATLAB code and data are in `docs/`). The handbook teaches the
models *and* the samplers — Gibbs conditionals, Carter–Kohn, Metropolis–
Hastings — with MATLAB code for every example. This companion keeps the
handbook's chapter structure and its examples, on the handbook's own data,
and replaces the sampler code with **macrotoolkit**: each model is written
as a system of equations, compiled to a state-space program, estimated by
NUTS with the linear-Gaussian states integrated out by the Kalman filter,
and validated by the toolkit's gates. Where the handbook derives a
conditional posterior, the companion shows the equation that declares the
model; where it plots Gibbs draws, the companion reads the diagnostics
verdict.

**Every number and figure in the companion comes from a committed spec
under `examples/handbook/` and a recorded run.** Where a prior family the
handbook uses is not in the toolkit's menu (inverse-Wishart, dummy
observations), the substitution is stated in a *What differs from the
handbook* box, never glossed. Where a model is outside the toolkit's
class, the chapter says so and says what would be needed.

## How to read

| File | Handbook | Content |
|---|---|---|
| [00-inventory.md](00-inventory.md) | all | every example script mapped to a verdict: runs / gap (with the extension it needs) / outside the class |
| [01-part0-workflow.md](01-part0-workflow.md) | Ch. 1 §3.7–3.8, Ch. 3 §4–5, Ch. 5 §2–4, §6 | **Part 0 — From Gibbs to NUTS**: the workflow that replaces the sampler chapters |
| [02-ch1-regression.md](02-ch1-regression.md) | Ch. 1 | linear regression: the AR(2) for US inflation, forecasting, AR(1) errors, diagnostics, marginal likelihood |
| [03-ch2-vars.md](03-ch2-vars.md) | Ch. 2 | VARs: Minnesota, Cholesky IRFs, steady-state prior, dummy observations, sign restrictions, conditional forecasts |
| [04-ch3-state-space.md](04-ch3-state-space.md) | Ch. 3 | state-space models: the filter and the smoother, unobserved components, the TVP regression, the TVP-VAR, the FAVAR, mixed frequency |
| [05-ch4-markov-switching.md](05-ch4-markov-switching.md) | Ch. 4 | *gap chapter*: Markov-switching models |
| [06-ch5-mh-and-sv.md](06-ch5-mh-and-sv.md) | Ch. 5 | stochastic volatility, the TVP-AR with SV, the TVP-VAR with SV, structural and threshold VARs |
| [07-ch6-dsge.md](07-ch6-dsge.md) | Ch. 6 | *gap chapter*: linear DSGE estimation |
| [08-ch7-tvp-dfm.md](08-ch7-tvp-dfm.md) | Ch. 7 | the time-varying dynamic factor model with SV, and the constant-coefficient bridge that runs today |

The executable counterparts are the notebooks under
`examples/handbook/notebooks/` (one per chapter, built by
`build_notebooks.py`, committed executed) and the specs under
`examples/handbook/<example>/spec.yaml`, each with a README recording the
handbook's construction, the toolkit's, the differences, and the run.

## Running the examples

```bash
# environment: HANDOFF.md's recipe (CmdStan 2.36.0 pinned; xlrd + openpyxl for the handbook's .xls files)
python examples/handbook/make_data.py       # the handbook's .xls/.xlsx -> examples/handbook/data/*.csv
python examples/handbook/build_specs.py     # the specs (VAR specs are generated through au.var)
python examples/handbook/run_smoke.py       # short-chain fits + fast validation tier + post-processors, records appended to each README
mtk run examples/handbook/ch1_ar2/spec.yaml # any single example; mtk report <hash> for the self-contained HTML
```

The smoke runs in the example READMEs use short chains (2 × 300/300) and
are diagnostics-honest: several carry a WARN or FAIL verdict at that
length, and the companion says so. Publication-length runs are the run
sessions described in `09-run-sessions.md`.

## Status

| Chapter | Runs today | Gap (extension) | Outside the class |
|---|---|---|---|
| 1 Regression | ex. 1–3 | ex. 6 marginal likelihood (E8) | — |
| 2 VARs | ex. 1–3, 5–8 (Minnesota / recursive form; sign restrictions and conditional forecasts as post-processors) | ex. 4 dummy-observation priors (stated approximation), ex. 9 marginal likelihood (E8) | — |
| 3 State space | §2 UC, §2 DFM, ex. 1–4 (TVP regression, TVP-VAR, DFM part of the FAVAR) | FAVAR rate block (follow-up), ex. 5 mixed frequency (E7) | — |
| 4 Markov switching | — | — | all (regime switching is a stated non-goal) |
| 5 MH | ex. 3–5 (TVP via the filter, SV, TVP-AR-SV), SVAR (recursive form) | ex. 6 TVP-VAR-SV (E6), ex. 7 marginal likelihood (E8) | ex. 1–2 nonlinear regression, threshold and STAR VARs |
| 6 DSGE | — | — | all (DSGE solution in the loop is a stated non-goal) |
| 7 TVP-DFM-SV | the constant-coefficient DFM-SV bridge | time-varying AR coefficients (E6) | — |

Extension ids (E0–E8) are defined in the inventory.
