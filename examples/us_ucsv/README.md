# US UCSV example (S6, family #2)

Stock–Watson unobserved-components trend inflation with stochastic
volatility on both shocks:

    pi_t  = tau_t + eps_t,       eps_t ~ N(0, exp(h_eps,t))
    tau_t = tau_{t-1} + eta_t,   eta_t ~ N(0, exp(h_eta,t))

Data: `data/us_core_pce.csv` — US quarterly core PCE inflation, annualized
q/q percent (400·dlog P), derived by `make_data.py` from the NY Fed HLW
workbook fixture already in the repo (the same `inflation` column
`examples/us_lw_sv` uses as `core_pce_ann`; see FIXTURES.md). 1960Q1–2026Q1;
`spec.yaml` estimates the pre-COVID window through 2019Q4,
`spec_full_vintage.yaml` the whole file (COVID quarters included, no
hand-set COVID machinery — the SV absorbs them). UCSV has no pre-sample
lag convention: every row in the window is estimated.

Two equivalent ways to run it:

```bash
mtk run examples/us_ucsv/spec.yaml          # -> runs/f2b48ebc98a4/
mtk report f2b48ebc98a4
```

or the notebook `examples/notebook_api/ucsv_us_inflation.ipynb`, which
builds the same spec in Python, fits through `mtk.fit`, and lands on the
same hash (the notebook is committed executed).

Every fit records the automatic Stan-vs-Python Kalman-filter mirror check
(S6 WP3) in `diagnostics.json` and the report header.

**S9 identity note.** The shared filter gained a time-varying loading
`Z_t` in S9 (plans/S9-plan.md decision 1; DECISIONS.md 2026-09-05), and
the filter text is inlined into the rendered program, so `mtk run
examples/us_ucsv/spec.yaml` now produces run `6f4d5d043436`
(`spec_full_vintage.yaml`: `bd55f3a3f061`) -- numerics unchanged (the
constant-Z regression pin is exact; the mirror check and G6 are green).
The run record below (`f2b48ebc98a4`) is the S6-program run and remains
a valid immutable record under that identity.

## Run record (2026-09-04, run `f2b48ebc98a4`, 1960Q1–2019Q4, T = 240)

Estimated through the notebook API first (`mtk.fit`), then `mtk run
examples/us_ucsv/spec.yaml` as the CLI cross-check: **same hash,
idempotent no-op**, `mtk report f2b48ebc98a4` renders 6 embedded figures.
4 chains × 1000/1000, adapt_delta 0.95. Diagnostics **PASS**: 0
divergences, 0 treedepth hits, max R-hat 1.006, min bulk/tail ESS
1294/1056, E-BFMI 0.92–0.97. Fit-time mirror check: max |Stan − Python|
KF log-likelihood 1.1e-13 over 5 prior draws (gate 1e-8). ~14 minutes of
sampling (contended host).

| Parameter | Posterior median [90% CI] |
|---|---|
| sigma_h_eps (transitory log-variance RW scale) | 0.21 [0.11, 0.33] |
| sigma_h_eta (trend log-variance RW scale) | 0.29 [0.18, 0.45] |
| h0_eps_raw / h0_eta_raw (non-centered initial log-variances) | −0.81 [−1.90, 0.23] / −0.88 [−2.11, 0.27] |

Posterior-median trend inflation τ: 1.0 (1961Q1) → 7.7 (1975Q1) → 9.2
(1980Q1) → 4.1 (1990Q1) → 1.7 (2000Q1) → 1.6 (2010Q1) → 1.6 (2019Q4) —
the Great Inflation and the Volcker disinflation carried by the trend,
the last two decades flat at ~1.6. The volatility paths (exp(h/2),
standard deviations) tell the Stock–Watson story: the **trend**-shock
sd peaks in 1974Q2–Q4 (1.34 → 1.27) and ends at 0.06 (an essentially
anchored trend by 2019), while the **transitory**-shock sd peaks around
2008Q4/2009Q1 (0.75) and 1983Q3 (0.74) and ends at 0.35. The HD identity
(bars: initial condition, trend shocks, transitory shocks) reconstructs
inflation to 4e-16 per period per draw.

## Validation (no external oracle exists for UCSV)

There is no published UCSV reference code to replicate the way HLW's
code anchored lw_sv's G5a, so the family's external credibility rests on
the ladder's other rungs, run through `mtk validate ucsv --tier <tier>`
(registered designs in `macrotoolkit/families/ucsv_validation.py`):

| Gate | Tier | Result |
|---|---|---|
| G1 mirror (shared filter, 5 paths incl. Q_t) | `tests/test_g1_mirror.py` | max diff 7.3e-12 (gate 1e-8) |
| Production-render mirror at 25 prior draws | fast | 1.8e-12 |
| G6 HD identity | fast | 2e-16 (gate 1e-6) |
| G2 recovery (20 datasets, pre-registered) | recovery | PASS: pooled coverage 0.85, per quantity 0.85/0.85/0.75/0.95, no σ_h bias |
| SBC (100 reps, pre-registered) | sbc | PASS: χ² p 0.596/0.911/0.760/0.052, 4 divergences |

Both slow tiers ran once at exactly their pre-registered designs on
2026-09-04 (DECISIONS.md); the SBC record is crash-resumable
(`tests/artifacts/ucsv_sbc/ranks.csv`, gitignored, regenerable from the
fixed seeds via `scripts/run_ucsv_sbc.py`).
