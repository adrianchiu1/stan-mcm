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

## S3: SV variant (`spec_sv.yaml`), 2026-08-31 — acceptance run `70ad47166eaf`

`sv_shocks: [is, pc]`: non-centered SV on the IS/PC measurement shocks,
mu_h0 anchored by the HLW-exact OLS pass (DECISIONS.md 2026-08-31). Same
window as the S2 reference. Diagnostics **PASS at the DEFAULT
adapt_delta 0.95 — no retuning needed**: 0 divergences, 0 treedepth hits,
E-BFMI 0.88–1.01, max R-hat 1.004, min bulk/tail ESS 2628/1620. The
anticipated sigma_h funnel never bit: the posteriors put sigma_h,IS at
0.25 [0.14, 0.42] and sigma_h,PC at 0.22 [0.14, 0.32] (5th percentiles
well off zero), so the non-centered geometry stays benign.

The exp(h/2) volatility paths (spec §3.1's exhibit): PC volatility peaks
≈1.5 around 1974 (oil shock) and IS volatility decays from ≈1.1 in the
late 1970s to 0.22 by 2019 — the Great Moderation, visible with no
COVID-era data in sight.

Statics shift where time-varying measurement variance reallocates what
constant scales forced elsewhere: sigma_y* 0.24 (vs 0.54 no-SV), a_r
−0.047 (vs −0.070), b_y 0.036 (vs 0.073), a1/a2 1.28/−0.31 (vs
1.44/−0.50). sigma_g/sigma_z are unchanged (pile-up priors still bind).

## S3: COVID payoff exhibit (`spec_sv_full_vintage.yaml`), run `9d10bcf32a40`

The stress test's answer (STRESS-TESTS.md §3): full vintage through
2026Q1, COVID quarters included, SV on, **no hand-set COVID machinery**.
Diagnostics **PASS** (0 divergences, 0 treedepth hits, max R-hat 1.005,
min bulk/tail ESS 1779/917; contrast the no-SV EA run's marginal FAIL).

- **The 2020 spike is endogenous and exactly placed:** the four largest
  exp(h_IS/2) medians are 2020Q3 (3.77), 2020Q2 (3.52), 2020Q4 (2.48),
  2020Q1 (2.19), decaying to 0.75 by 2022Q1 — the Bayesian counterpart
  of HLW's hand-set 2020Q2–Q4 kappa variance scaling, discovered from
  the data. sigma_h,IS rises to 0.60 [0.44, 0.78] to carry the jump.
- **Structural parameters de-contaminate:** sigma_y* 0.33 [0.19, 0.42]
  vs the no-SV full-vintage run's 0.94 (pre-COVID: 0.54); gap AR
  a1/a2 = 1.26/−0.29, essentially the SV pre-COVID values 1.28/−0.31,
  instead of the no-SV collapse to 1.15. The absorption channel
  documented in STRESS-TESTS.md §3 is closed.

## S4: DK simulation smoother, output modules, HTML report

Spec §7's S4 row: "DK smoother; four output modules; HTML report | G6
pass; report renders all figures from a real run." Both S3 runs above
were regenerated container-locally against the S4 branch (same specs,
same seeds) to build and validate the output modules against; both
reproduced their S3 diagnostics exactly:

| Run | Spec | Verdict | Divergences | Max R-hat | Min bulk/tail ESS |
|---|---|---|---|---|---|
| `70ad47166eaf` | `spec_sv.yaml` | PASS | 0 | 1.0045 | 2628 / 1620 |
| `930459224ca0` | `spec_sv_full_vintage.yaml` | PASS | 0 | 1.0046 | 1779 / 917 |

(New hashes vs S3's `70ad47166eaf`→same, `9d10bcf32a40`→`930459224ca0`:
the full-vintage run's hash changed because S4 added a typed
`outputs:` schema for `lw_sv` — DECISIONS.md's 2026-08-31 "typed outputs
schema" entry — which changes what an empty `outputs: {}` canonicalizes
to; `spec_sv.yaml` happened to already carry the same hash since it was
regenerated before that schema change landed. Numerics are unchanged;
this is a schema-precision improvement, not a re-estimation.)

Gate G6 (HD reconstruction identity, exact to 1e-6 per period per draw)
passes on both no-SV and SV synthetic parameter points
(`tests/test_g6_hd_identity.py`). The S4 acceptance test
(`tests/test_report.py`) runs `mtk report` against the real
`70ad47166eaf` run end to end: `report.html` renders all four output
modules (trend-cycle incl. the exp(h/2) volatility panel, the 5×5 IRF
grid, 5 fan charts, 4 historical-decomposition charts) as one
self-contained file (11 embedded images, zero external references) plus
the diagnostics verdict and a parameter table. Generate it yourself with
`mtk report <hash>`; the two run directories above each now have their
own `runs/<hash>/report.html` (gitignored, regenerate via `mtk run` +
`mtk report`).

## S5: regeneration at the split run identity, prior-predictive, sweep

Both reference runs were regenerated once more in S5 against the
estimation-vs-report identity split (S5-decisions item 3; the one final
hash migration — DECISIONS.md 2026-09-02), again reproducing their
recorded diagnostics exactly:

| Run | Spec | Verdict | Divergences | Max R-hat | Min bulk/tail ESS |
|---|---|---|---|---|---|
| `a00958509083` | `spec_sv.yaml` | PASS | 0 | 1.0045 | 2627 / 1620 |
| `eb73e644be0b` | `spec_sv_full_vintage.yaml` | PASS | 0 | 1.0046 | 1778 / 917 |

(Hash lineage: `70ad47166eaf` → `a00958509083`; `9d10bcf32a40` →
`930459224ca0` → `eb73e644be0b`. Every migration is a spec/identity
schema change, never a numerics change — the diagnostics tables prove
it.) Draw-thinned (×10) tracked copies of both live in `runs-archive/`
as development fixtures (see its README).

S5's report additions render for both runs: the spec §4 prior-predictive
check figure (12 embedded images per report now) joins the diagnostics
block. The mandated σ_g/σ_z prior-sensitivity sweep runs from
`sweep_sigma_g_z.yaml` in this directory
(`mtk sweep examples/us_lw_sv/sweep_sigma_g_z.yaml`); its comparison
report lands under `sweeps/sigma_g_z/` with prior→posterior contraction
readouts, and the repo-level exhibits built from these runs (G5b
filtered-vs-published with its measured z-attribution; the COVID SV
volatility figure) are in `docs/exhibits/`.
