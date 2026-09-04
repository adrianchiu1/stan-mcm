# macrotoolkit

An open-source Python + Stan toolkit for estimating flexible Bayesian
state-space macro models with NUTS — driven end-to-end by declarative YAML
specs: CSV in → rendered Stan program → CmdStanPy estimation → immutable,
hash-identified run store → simulation smoother → trend-cycle plots, IRFs,
fan charts, historical decompositions → one self-contained HTML report.
The long-run vision (dynamic factor models, VAR-SV, UC/star-variable
families, a front-end for economists who never touch Stan) is in
[VISION.md](VISION.md); the engineering doctrine that binds every family is
in [ENGINEERING.md](ENGINEERING.md).

**Status**: through stage S6 the toolkit has **two validated families** —
the v1 vertical slice, Laubach–Williams with stochastic volatility
(`lw_sv`, spec in [lw-sv-spec.md](lw-sv-spec.md)), and family #2,
Stock–Watson trend inflation with SV on both shocks (`ucsv`, scoped by
[docs/kf-capability-matrix.md](docs/kf-capability-matrix.md)) — a
**notebook-first Python API** (`from macrotoolkit import api as mtk`;
`mtk.fit(spec, dataframe)` → run handle → inline figures → report →
sweeps, the CLI being a thin shell over it), and **automatic per-model
QC**: every fit cross-checks the exact rendered Stan program's Kalman-
filter log-likelihood against the Python mirror before sampling, and
`mtk validate <family>` runs a family's registered gate suite into one
validation report. lw_sv is deliberately a *dry run of the general
framework*: exact HLW replication is not a goal (see the G5b exhibit
below); UCSV was the test that the shared machinery is genuinely shared
(it needed ONE real KF extension, time-varying `Q_t`, and otherwise only
declarations).

## Validation ladder (all gates green)

Every family climbs the same ladder (ENGINEERING.md), and since S6 every
fit runs the first rung automatically (the fit-time mirror check, G1's
1e-8 gate at prior draws of the exact rendered program, recorded in the
run's diagnostics and report header) while `mtk validate <family> --tier
fast|recovery|sbc|all` runs the registered rungs on demand.

`lw_sv`'s results:

| Gate | What it proves | Result |
|---|---|---|
| G1 | Stan KF ≡ Python KF mirror, 50 prior points, 3 filter paths | max diff ~5.5e-12 (gate 1e-8) |
| G2 | Parameter recovery, 20 simulated datasets | pooled 90% coverage in band, no σ_g/σ_z bias |
| G3 | SBC, no-SV variant, 200 pre-registered replications | χ² p ∈ [0.073, 0.735] on all 10 params |
| G4 | SBC, full-SV variant, 100 pre-registered replications | χ² p ∈ [0.067, 0.978] on all 12 ranked quantities; 8/150,000 divergences |
| G5a | Exact HLW replication at fixed parameters | ~1e-12 vs the HLW (2017) oracle |
| G6 | Historical-decomposition reconstruction identity | exact to 1e-6 per period per draw |
| G5b | *(informational exhibit, not a gate)* filtered vs published one-sided HLW | differences measured + attributed (below) |

`ucsv`'s results (S6; there is **no external oracle** for UCSV — nothing
like HLW's published code — so the family's credibility rests on the
mirror, recovery and SBC rungs, stated explicitly):

| Gate | What it proves | Result |
|---|---|---|
| G1 | Stan KF ≡ Python KF mirror over 5 filter paths incl. time-varying Q_t (shared filter, 50 points) | max diff 7.3e-12 (gate 1e-8); constant-Q values reproduce the pre-S6 program exactly |
| G1 (production render) | fit-time mirror at prior draws of the UCSV program | 1.8e-12 |
| G2 | Parameter recovery, 20 pre-registered simulated datasets | PASS: pooled 90% coverage 0.85 (band [0.80, 0.97]); per quantity 0.85/0.85/0.75/0.95; no σ_h bias |
| SBC | 100 pre-registered replications, 4 ranked quantities | χ² p = 0.596 (sigma_h_eps), 0.911 (sigma_h_eta), 0.760 (h0_eps), 0.052 (h0_eta) — all above the 0.001 floor; 4/150,000 divergences |
| G6 | HD reconstruction identity (bars sum to inflation) | ~1e-13 per period per draw (gate 1e-6) |

## Exhibits

### G5a — the external credibility exhibit

Our KF + RTS smoother, fixed at the HLW (2017) MLE parameters and exact
initial conditions, against HLW's own code's smoothed output: the two
lines coincide to machine precision (max |diff| ~1e-12, stamped per
panel).

![G5a oracle replication](docs/exhibits/g5a_oracle_replication.png)

### G5b — filtered vs published one-sided HLW (informational)

HLW publish one-sided estimates only ("All estimates are one-sided"), so
the like-for-like comparison is our FILTERED posterior series against
their published series. **No pass/fail** — a gate you'd pass by tuning
priors toward a target validates nothing. Measured on the reference run:
r* correlation 0.941; the final-period r* difference (+1.44) decomposes
via r* = g + z into **+1.51 from z and −0.07 from g** — the gap sits
exactly where the deliberate σ_z pile-up identification prior acts, and
trend growth g essentially matches. Full attribution (five documented
causes) in
[docs/exhibits/g5b_filtered_vs_published.md](docs/exhibits/g5b_filtered_vs_published.md).

![G5b filtered vs published](docs/exhibits/g5b_filtered_vs_published.png)

### Stochastic volatility absorbs COVID endogenously

The full-vintage run (through 2026Q1) carries **no hand-set COVID
machinery**; the SV block discovers it from the data — the four largest
IS-shock volatility medians land in exactly 2020Q1–Q4 (peak 3.77), the
Bayesian counterpart of HLW's hand-set κ variance scaling:

![COVID SV volatility](docs/exhibits/covid_sv_volatility.png)

### Prior sensitivity: the mandated σ_g/σ_z sweep

The two pile-up priors do deliberate identification work (they replace
HLW's median-unbiased-estimator machinery), so their sensitivity is
documented by a reusable tool, not a notebook:
`mtk sweep examples/us_lw_sv/sweep_sigma_g_z.yaml` runs
halved/doubled prior scales as ordinary immutable runs and writes one
comparison report with posterior tables, **prior→posterior contraction**
readouts, and the headline r*/gap series overlaid across cells.

Results (5 cells, all 0 divergences; full record in DECISIONS.md
2026-09-03): the σ_g/σ_z posteriors scale near-proportionally with their
prior scales (σ_g median 0.036/0.068/0.109 under prior sd
0.015/0.03/0.06; σ_z median 0.027/0.054/0.115 under 0.04/0.08/0.16) with
contraction of only ~0.04–0.14 — the data contribute little information
about these scales, which is precisely the pile-up problem the priors
exist to resolve, now measured rather than asserted. And the G5b
attribution cross-checks monotonically: the filtered final-period r* gap
to the published series is +1.54 / +1.44 / +1.16 under the tightened /
default / doubled σ_z prior (final z: +0.02 / −0.07 / −0.35).

## Quickstart

```bash
# environment (fresh container recipe -- see HANDOFF.md for details)
uv tool install pytest --with cmdstanpy --with numba --with arviz \
  --with pydantic --with jinja2 --with matplotlib --with pyyaml \
  --with pandas --with click --with h5netcdf --with openpyxl \
  --with-editable .
python -m cmdstanpy.install_cmdstan --dir ~/.cmdstan --version 2.36.0

# estimate, report, sweep, validate (the CLI is a thin shell over the API)
mtk run examples/us_lw_sv/spec_sv.yaml        # ~40-80 min, immutable runs/<hash12>/
mtk run examples/us_ucsv/spec.yaml            # family #2, a few minutes
mtk report <hash12>                           # self-contained report.html
mtk sweep examples/us_lw_sv/sweep_sigma_g_z.yaml
mtk validate ucsv --tier fast                 # one validation report per family/tier

# tests (fast suite; slow gates run with -m slow; one-liners in scripts/gate-check.sh)
pytest -m "not slow"
scripts/gate-check.sh validate lw_sv fast
```

From a notebook (S6): no YAML, no CLI —

```python
from macrotoolkit import api as mtk
spec = mtk.spec("ucsv", data={"file": "dataframe.csv", "date_column": "date",
                              "mapping": {"pi": "core_pce_ann"}})
run = mtk.fit(spec, df)                 # pandas DataFrame in; immutable hashed run out
run.verdict, run.mirror_check           # diagnostics + the automatic KF mirror check
run.outputs().figure("trend_cycle")     # matplotlib Figures, never files
run.param_table(); run.report()         # DataFrame; self-contained report.html
```

Executed example notebooks: [examples/notebook_api/](examples/notebook_api/README.md)
(the lw_sv output layer on the archived reference run without sampling;
UCSV end to end).

A tracked, draw-thinned copy of both reference runs lives in
[runs-archive/](runs-archive/README.md) so a fresh checkout can exercise
the whole output layer without sampling first (development fixtures —
regenerate for publication numbers).

## Adding a model family

Mechanically true since S4.5 and exercised by UCSV in S6 (registry
contract, `specs/schema/FAMILY_REGISTRY`): a family = Stan template +
spec-schema fragment + a `macrotoolkit/families/<family>.py` numerics
module (named state metadata with explicit time offsets, the
endogenous-lag feedback map, prior sampler, builders, the mirror-check
declaration) + a results/plots/outputs declaration over the generic
results core and engine (`results_core.py`, `engine.py`) + one registry
entry + a `families/<family>_validation.py` declaring the gate designs
(the generic SBC engine, recovery arithmetic, mirror and identity gates
live in `macrotoolkit/validation/`). The run store, report, notebook
API, sweep tool, prior-predictive check, fit-time mirror check and `mtk
validate` pick the family up automatically. The KF's capability
boundary (what is built, what waits for the family that needs it) is
[docs/kf-capability-matrix.md](docs/kf-capability-matrix.md).

## Repository map

- `specs/schema/` — the spec spine: Pydantic schema (`RunSpec` with
  `qc:`/`outputs:` outside the identity hash) + the family registry
- `stan/` — shared Stan functions library (the Q_t/R_t-generalized
  Kalman filter, SV helpers) + per-family Jinja templates
- `src/macrotoolkit/` — `api` (the notebook-first public API) · data →
  render → run store (`run`) → `qc` (fit-time mirror check) → `smoother`
  → `engine` → `results_core` / `results_lw` / `results_ucsv` → plots →
  `outputs*` (declared figure modules) → `report` → `cli` (+ `sweep`,
  `validation/` (SBC engine, recovery, gate suites), `families/`
  (per-family declarations + validation designs), G5b exhibit)
- `tests/` — gates G1–G6 as pytest suites; markers `lw_sv` / `ucsv` /
  `validation` / `slow`
- `examples/us_lw_sv/`, `examples/us_ucsv/` — the worked examples
  (specs, data derivation, run records); `examples/notebook_api/` — the
  executed API notebooks
- `runs-archive/` — tracked draw-thinned reference runs (development
  fixtures)
- `DECISIONS.md` — every judgment call, dated, newest first
- `HANDOFF.md` — the current stage-end handoff
