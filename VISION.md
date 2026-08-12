# VISION — macrotoolkit

## What this is

An open-source Python + Stan toolkit for estimating flexible Bayesian state-space macro models — dynamic factor models, structural VARs, unobserved-components/star-variable models, and their stochastic-volatility variants — driven end-to-end by declarative model specifications, and eventually operable from a front-end by economists who never touch Stan.

## The spine

**A YAML/JSON model spec is the single source of truth.** Everything else — Stan code generation, estimation, the run store, results objects, plots, reports, and ultimately the GUI — is machinery that consumes or produces specs and run artifacts. A run is identified by the hash of (spec, data, Stan source, toolchain version) and is immutable and reproducible forever.

## Architecture in one paragraph

A shared Stan functions library (Kalman recursions with time-varying covariances, non-centered SV blocks, state-space matrix constructors) is composed into per-model-family Jinja templates. Templates are stamped into concrete, fully-specialized Stan programs at spec-render time — no runtime branching inside Stan, no universal mega-model. Estimation is full-Bayes NUTS via CmdStanPy, with model-appropriate parameterization per family (Rao-Blackwellized/marginalized states where the geometry favors it). Python owns everything around the sampler: data wrangling, simulation smoothing, structural identification (e.g. sign restrictions as post-processing over reduced-form posteriors), forecasting, decompositions, diagnostics, and reporting.

## What "done well" means

1. **Trustworthy.** Every model family ships with simulation-based calibration results, parameter-recovery tests, and — wherever one exists — replication against a published external oracle. Diagnostics are automatic and honest: every run gets a PASS/WARN/FAIL verdict.
2. **Easy.** An economist writes (or clicks together) a spec, points at a CSV, and gets an HTML report with trend-cycle plots, IRFs, fan charts, and historical decompositions. Quirks of individual models live in their family modules, never in the user's way.
3. **Extensible.** Adding a model family = a template + a spec-schema fragment + a results module + a validation suite. The front-end and infrastructure pick it up automatically.

## Current slice (v1)

Laubach-Williams with stochastic volatility on the demand and supply shocks: the full vertical — CSV in, Stan NUTS estimation, simulation smoother, four output families (trend-cycle, IRF matrix, fan charts, historical decomposition), HTML report, CLI. The HLW published code, data, and estimates serve as the external oracle for every linear-Gaussian component. Specification: `lw-sv-spec.md` (authoritative).

## Roadmap after v1 (indicative, not committed)

DFM family (spec-driven loading restrictions, mixed frequency, fat tails à la Antolín-Díaz et al.) · VAR-SV reduced form + identification strategies (sign, zero+sign, narrative) as plug-in post-processors · UC/star-variable generalizations (UCSV trend inflation, output gap, u-star) · conditional forecasting / scenario analysis · model comparison and stacking over the run store · FastAPI service + job queue + React front-end with per-family spec editors.

## Non-goals

DSGE solution steps inside the estimation loop; particle filters and non-linear/regime-switching filtering; ALFRED-style real-time vintage management; very-large-N (>100 variable) DFMs where full-Bayes NUTS is the wrong tool; proprietary data connectors in the open-source core.

## Engineering values

Validation gates are hard blockers, sequenced, and never skipped. Numerical conventions (units, log-variance vs log-sd, annualization) are stated once in the spec and enforced by tests. Python mirrors of Stan computations must agree to near machine precision before anything downstream is trusted. Discrepancies with external oracles are traced to documented causes, never tolerated as "close enough" without explanation.
