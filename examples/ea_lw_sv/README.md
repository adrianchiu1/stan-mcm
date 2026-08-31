# Euro Area Laubach-Williams example (S2: no-SV variant)

Spec: `spec.yaml`. Data: `data/ea_quarterly.csv`, derived by `make_data.py`
from the NY Fed current-estimates workbook's "EA input data" sheet
(1971Q1–2019Q4; estimation 1972Q1–2019Q4 — HLW's own EA start, ending
before COVID since the S2 model has no COVID-adjustment machinery).

Run from the repo root:

```
mtk run examples/ea_lw_sv/spec.yaml
```

## First run, 2026-08-31 (run `b6a26e973787`)

Diagnostics **PASS** — 0 divergences, max R-hat 1.002, min bulk ESS 1361.

Tracking vs. HLW's published one-sided EA series (2019Q2 real-time vintage,
2017 model, common window 1972Q1–2019Q2), using our filtered states
like-for-like:

| Series | Correlation | Mean \|diff\| |
|---|---|---|
| r* | 0.955 | 0.40pp |
| g | 0.984 | 0.12pp |
| output gap | 0.988 | 0.30pp |

Posterior medians: a1 +1.49, a2 −0.55, a_r −0.043, b_pi +0.69, b_y +0.078,
sigma_ystar 0.31, sigma_g (ann.) 0.064, sigma_z 0.052, sigma_is 0.41,
sigma_pc 1.00. The published 2023-model Table-1 EA parameters (estimated on
1972–2026 with COVID machinery and estimated c≈0.97) are indicative
context, not a like-for-like target; the same sigma_g/sigma_z
pile-up-prior caveat as the US example applies.

Context: HLW publish no Germany model — this EA aggregate (Germany is its
largest economy) is the closest published benchmark for Germany.

## Full latest-vintage run, 2026-08-31 (run `ab363d5026a1`, experiment)

`spec_full_vintage.yaml`: 1972Q1–2026Q1, COVID included, no COVID machinery
(HLW scale EA 2020Q2–Q4 shock variances ×19.8). Diagnostics **FAIL** on the
strict zero-divergence rule: 2 divergences in 6,000 draws (max R-hat 1.007,
ESS > 1000) — marginal, flagged.

The EA absorbs COVID differently from the US: σ_IS ≈ 0.99 (vs 0.41
pre-COVID) and the gap's AR structure collapses (a₁,a₂ = 0.77,+0.01 vs
1.49,−0.55); σ_y* also roughly doubles. Tracking vs published current
estimates over 1972Q1–2026Q1: r* corr 0.929, g 0.961, gap 0.861, z 0.711.
Use the pre-COVID run above as the reference result.
