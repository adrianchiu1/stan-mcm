# HLW fixtures — inventory

Tracks what's checked into `tests/fixtures/hlw/` against what spec §5's
"HLW fixtures policy" requires (NY Fed published R code, input data, and
output series, with source URLs and retrieval dates). Written during S0
preflight; update whenever fixture contents change.

## What's present

### Code

```
tests/fixtures/hlw/HLW_2023_Replication_Code/HLW_2023_Replication_Code/
  HLW_Replication_Code_Guide.pdf   # documentation note, dated 2023-08-31
  run.hlw.{us,ca,ea}.R, run.hlw.estimation.R
  rstar.stage{1,2,3}.R, unpack.parameters.stage{1,2,3}.R
  kalman.states.R, kalman.states.wrapper.R
  kalman.log.likelihood.R, log.likelihood.wrapper.R
  kalman.standard.errors.R
  median.unbiased.estimator.stage{1,2}.R
  calculate.covariance.R, format.output.R, utilities.R
```

Already in the repo at the initial commit; no source-URL/retrieval-date note
accompanied it. Likely source (per the guide's own citation, footnote 2):
`https://www.newyorkfed.org/research/policy/rstar`.

### Data (added this session — user-supplied upload, since `www.newyorkfed.org`
is blocked by this environment's egress policy; see `PREFLIGHT.md` §3)

```
tests/fixtures/hlw/data/
  Holston_Laubach_Williams_current_estimates.xlsx   # 178 KB
  Holston_Laubach_Williams_real_time_estimates.xlsx  # 1.77 MB
```

Source: NY Fed r* page (`https://www.newyorkfed.org/research/policy/rstar`),
per the files' own `info` sheets. Retrieved and supplied by the user
2026-08-13 (upload timestamp; original download date by the user unknown —
ask if it matters, e.g. for pinning a data vintage).

**`current_estimates.xlsx`** — sheets: `info`, `HLW Estimates`, `Parameters`,
`US input data`, `CA input data`, `EA input data`.

- `US input data`: columns `date, gdp.log, inflation, inflation.expectations,
  interest, covid.ind`, quarterly 1960Q1–2026Q1 (265 rows). This is exactly
  the `inputData/Holston_Laubach_Williams_estimates.xlsx` /
  `sheet="US input data"` the R code (`run.hlw.us.R:166`) expects — the R
  code will run unmodified once pointed at this file.
- `Parameters`: published MLE point estimates for the **current (2023
  COVID-adjusted) model**, full sample 1961Q1–2026Q1: `lambda_g, lambda_z,
  a_y1, a_y2, a_r, b_pi, b_y, c, phi, kappa_2020Q2-Q4, kappa_2021, kappa_2022,
  kappa_2023, sigma_y~, sigma_pi, sigma_y*, sigma_g, sigma_z, sigma_r*`, plus
  standard errors. Confirms: `c` is **freely estimated** here (US ≈ 1.116),
  `phi` and `kappa_*` are nonzero/≠1 — i.e. this is unambiguously the 2023
  COVID-adjusted parameterization, not the plain 2017 one.
- `HLW Estimates`: published **one-sided only** (sheet header says so
  explicitly) `g, z, r*, output gap` for US/CA/EA, 1961Q1–2026Q1. No
  smoothed/two-sided series — consistent with the guide's note that
  `run.hlw.XX.R` doesn't write two-sided estimates to CSV (`format.output.R`
  only formats one-sided output; two-sided is returned in-memory as
  `two.sided.est.XX` and discarded by the stock scripts).

**`real_time_estimates.xlsx`** — one sheet per vintage, `2015Q4` … `2026Q1`
(with a gap 2020Q3–2022Q3), each a one-sided-estimates snapshot as of that
vintage's data cutoff. Its `info` sheet states explicitly: **"Estimates
prior to 2022:Q4 come from the model described in ... Journal of
International Economics, 2017."** i.e. vintage sheets `2015Q4`–`2019Q4`
(21 columns: Output Gap / Trend Growth / z / r* × {US, CA, EA, UK}, no
COVID terms) are genuine **HLW (2017)** one-sided output. `2020Q1`–`2020Q2`
(26 columns, adds an "Adjusted output gap" block) are transitional. `2022Q4`
onward (17 columns) match the 2023 COVID-adjusted `current_estimates.xlsx`
layout.

## What's still missing, and why it matters for G5a

We now have, for the **2023 COVID-adjusted model**: matching input data +
published MLE parameters + published one-sided estimates, all mutually
consistent (same vintage, same model). That's a complete, usable oracle
**if** G5a targets the 2023 model. What's still absent: the two-sided
(smoothed) series for that model — not published anywhere in these files,
per the guide's own admission; getting it would mean running the R code
ourselves (needs R + its packages, not evaluated in this session) or
asking NY Fed directly.

For the **plain HLW (2017) model that `lw-sv-spec.md` actually specifies**
(spec §1.2–§1.3; no COVID dummy, no φ, no κ_t, `c` fixed at 1 by default):
we have genuine 2017-model **one-sided output** (the pre-2020 vintage
sheets in `real_time_estimates.xlsx`), but **not** the parameter values
that produced them — each vintage was estimated by MLE on its own sample,
and those per-vintage parameter values aren't published in either workbook
(`Parameters` only has the current, full-sample, 2023-model MLE values).
Without matching parameters, we can't reproduce a specific vintage's
one-sided path exactly — a KF run at the wrong parameter values isn't a
meaningful G5a oracle regardless of how correct the KF code is.

**Net: neither model variant currently has a fully self-consistent
(data + exact parameters + published output) fixture set for the plain
2017 equations. The 2023-model fixture set is complete except for the
smoothed series.** This is the live version of the model-vintage question
raised to the user in S0 (still open — see `HANDOFF.md`); options, now
sharpened by these concrete facts:

1. **Target the 2023 model for G5a**, extending the KF/smoother fixture
   harness (test-only, not the v1 Stan model) with the COVID dummy + φ +
   κ_t terms so it matches `current_estimates.xlsx` exactly, and either
   accept one-sided-only comparison or run the R code once (if R becomes
   available) to get the smoothed series too. Closest to spec's intent
   ("HLW is the external oracle... every residual discrepancy traced to
   a documented difference") while working entirely from what we now have.
2. **Target the 2017 model** using a pre-2020 vintage's one-sided output
   as a *qualitative* check only (no exact-parameter reproduction possible
   without more data), or pursue getting the actual 2017-vintage parameter
   values from the original 2017 JIE paper / its own replication package
   (a further fetch, same blocked domain, or a further user upload).
3. Some combination — e.g. G5a proper targets 2023-model machinery
   (matrices literally match published parameters), G5b's "loose-prior
   Bayesian tracking" check (which only needs to track published series
   within a band, not hit machine precision) uses the 2017-vintage
   one-sided series as its comparison band instead.

This doesn't block S1. It blocks starting S2's G5a test design until
decided — flagged in `HANDOFF.md`.

## Bottom line for S1

S1's acceptance test (`mtk run examples/toy/spec.yaml` on a synthetic
local-level model) needs none of this and proceeds unblocked.
