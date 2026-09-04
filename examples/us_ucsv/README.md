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
| G2 recovery (20 datasets, pre-registered) | recovery | see below |
| SBC (100 reps, pre-registered) | sbc | see below |

Results of the slow tiers are recorded in DECISIONS.md (2026-09-04) and
summarized below once run.
