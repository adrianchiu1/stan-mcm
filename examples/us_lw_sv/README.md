# US Laubach-Williams example (S2: no-SV variant)

Spec: `spec.yaml` (family `lw_sv`, `sv_shocks: []`). Data:
`data/us_quarterly.csv`, derived by `make_data.py` from the HLW fixture
workbook (1960Q1–2019Q2; the first four quarters are pre-sample lags, so
estimation covers 1961Q1–2019Q2 — the same window as the G5a oracle).

Run from the repo root:

```
mtk run examples/us_lw_sv/spec.yaml
```

## S2 qualitative check (spec §7: "posterior sensible on US data")

First real-data run, 2026-08-31 (run `24b6288dddad`, seed 20260813,
4 chains × 1500/1500): diagnostics verdict **PASS** — 0 divergences, 0
treedepth hits, max R-hat 1.007, min bulk/tail ESS 1043/605.

Posterior vs. the HLW (2017) MLE point (the G5a fixture; sigma_g
annualized ×4 for comparability):

| Parameter | Posterior median [90% CI] | HLW MLE |
|---|---|---|
| a1 | +1.44 [+1.20, +1.64] | +1.51 |
| a2 | −0.50 [−0.70, −0.25] | −0.57 |
| a_r | −0.070 [−0.109, −0.039] | −0.074 |
| b_pi | +0.70 [+0.59, +0.79] | +0.67 |
| b_y | +0.073 [+0.028, +0.143] | +0.074 |
| sigma_ystar | +0.54 [+0.26, +0.65] | +0.57 |
| sigma_g | +0.063 [+0.037, +0.095] | +0.154 |
| sigma_z | +0.058 [+0.006, +0.164] | +0.176 |
| sigma_is | +0.42 [+0.27, +0.66] | +0.34 |
| sigma_pc | +0.80 [+0.74, +0.87] | +0.80 |

Every structural coefficient brackets the HLW MLE comfortably. The two
deliberate exceptions are `sigma_g` and `sigma_z`: their posteriors sit
well below HLW's MUE-implied values because spec §1.6's tight Half-Normal
priors (sd 0.03 / 0.08) *replace* the median-unbiased-estimator machinery
as the pile-up control — these priors do identification work by design,
and spec §1.6 mandates a prior-sensitivity sweep (S5 scope) for exactly
this reason.

Smoothed states at the posterior median track HLW's smoothed series with
correlation 0.98 (r*) / 0.98 (gap). r* declines from ≈3.9% (1961) to
≈1.6% (2019) vs. HLW's 4.3% → 0.7%: the flatter late-sample decline is
the direct, expected consequence of the smaller `sigma_z` (z moves less,
so less of the r* decline is attributed to headwinds). Documented cause,
not a discrepancy to fix — the exact-replication claim lives in G5a
(`tests/test_g5a_hlw_replication.py`), where our KF/smoother at HLW's own
parameters matches their output to ~1e-12.

## Full latest-vintage run, 2026-08-31 (run `18f32ac9793f`, experiment)

`spec_full_vintage.yaml`: 1961Q1–2026Q1, COVID quarters included, **no COVID
machinery** (HLW's 2023 model scales 2020Q2–Q4 shock variances ×7.6 and adds
a dummy; the S2 model cannot). Diagnostics **WARN** (tail ESS 396 vs 400
threshold; 0 divergences, max R-hat 1.002).

COVID is absorbed through the potential-level shock: σ_y* ≈ 0.94 (vs 0.54
pre-COVID) and gap persistence flattens (a₁ 1.15 vs 1.44). Tracking vs the
published current one-sided estimates over 1961Q1–2026Q1: r* corr 0.914
(mean |diff| 0.75pp), g 0.871, gap 0.845, z 0.871. At 2026Q1 our filtered
r* is +2.4 vs HLW's +1.1 — the pile-up-prior σ_z effect plus COVID
absorption. Use the pre-COVID run above as the reference result.
