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

```
tests/fixtures/hlw/HLW_2017_Code/
  HLW_Code_Guide.pdf               # documentation note, dated 2017-02-06
  run.hlw.R, run.hlw.estimation.R
  prepare.rstar.data.{us,ca,ea,uk}.R
  rstar.stage{1,2,3}.R, unpack.parameters.stage{1,2,3}.R
  kalman.states.R, kalman.states.wrapper.R
  kalman.log.likelihood.R, log.likelihood.wrapper.R
  kalman.standard.errors.R, hpfilter.R
  median.unbiased.estimator.stage{1,2}.R
  calculate.covariance.R, format.output.R, utilities.R
```

Added this session — user-supplied upload (`HLW_Code.zip`), since fetching
from NY Fed is blocked (see `PREFLIGHT.md` §3). **This is the genuine
original HLW (2017) code** (guide dated 2017-02-06; original estimation
sample 1961Q1–2016Q3, code file timestamps through 2020 suggest minor later
maintenance, e.g. `run.hlw.R`'s default `sample.end <- c(2019,2)`). Verified
by direct comparison against `lw-sv-spec.md`: the Stage 3 matrices in the
guide (H′, A′, F, Q) match spec §1.2–§1.3 term for term — no COVID dummy,
no φ, no κ_t, `r*_t = g_t + z_t` (no `c`). Also confirmed by exhaustive
grep: zero occurrences of `covid`/`kappa`/`phi` anywhere in the code.
**This is the code the toolkit's G5a should target.**

Unlike the 2023 archive, this one does not bundle input data — it expects
raw FRED series in a `rawData/` folder, auto-fetched via `wget` in
`prepare.rstar.data.us.R` (not run in this session; FRED reachability not
yet tested). It also has no bundled parameter-estimates table.

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

## Decision: G5a targets the 2017 model (resolved 2026-08-13)

Recommended and adopted — see `DECISIONS.md`. Reasoning in brief: G5a
validates the shared KF/smoother library that the *production* model uses
(`kalman_loglik_tv.stan`/`ssm_matrices_lw.stan`), and the production model's
functional form is the 2017 equations (SV is layered on top, not a
replacement for the COVID-adjustment machinery). Targeting 2023 would mean
building and validating a KF variant (COVID dummy + φ + κ_t) the shipped
model doesn't have, weakening rather than strengthening the G5 credibility
exhibit. G5b's own pass criterion (spec §5) already restricts its
comparison window to 2000–2019, independently confirming COVID-era
agreement isn't needed for launch.

## What's still needed, and the path to get it

Neither workbook publishes the 2017-vintage MLE parameter values or a
smoothed/two-sided series for any 2017-model vintage — `real_time_estimates.xlsx`
has genuine 2017-model **one-sided** output only (vintage sheets
`2015Q4`–`2019Q4`), and `current_estimates.xlsx`'s `Parameters` sheet is
the current 2023-model MLE, not 2017's.

**Resolution path: run `HLW_2017_Code/` ourselves.** Now verified viable —
R 4.3.3 + `nloptr`/`mFilter` (via apt, `--no-install-recommends`) + `tis`
(built from source via `git clone https://github.com/cran/tis.git`, the
CRAN read-only mirror — direct CRAN access is blocked the same way
`newyorkfed.org` is, but this git clone isn't) all install and load
cleanly in this environment. Plan: feed the existing `US input data` sheet
(already covers 1960Q1 onward with the right columns) into
`rstar.stage{1,2,3}.R` / `run.hlw.estimation.R` via a small driver script
(bypassing `prepare.rstar.data.us.R`'s FRED auto-fetch, not `wget`-tested
here), trimmed to a pre-COVID sample — `sample.end <- c(2019,2)` matches
the code's own default and lines up with the `2019Q2` vintage sheet
already in `real_time_estimates.xlsx`, giving a free sanity check: our
run's one-sided output should track that published vintage closely (not
byte-exact, since today's GDP/PCE series have been revised since 2019).
This also solves the "no smoothed series published" gap for free — the
code computes `two.sided.est.*` internally, `format.output.R` just doesn't
write it to CSV; a few added lines fix that.

This produces a fully self-consistent 2017-model oracle: input data +
independently-reproduced MLE parameters + one-sided output (cross-checked
against the real 2019Q2 vintage) + smoothed output (not otherwise
available anywhere). Not yet executed — next concrete step, pending
go-ahead.

## Bottom line for S1

S1's acceptance test (`mtk run examples/toy/spec.yaml` on a synthetic
local-level model) needs none of this and proceeds unblocked.
