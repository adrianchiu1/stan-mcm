# US HLW (2017) reproduction — derived G5a oracle

**Not an official HLW publication.** This is our own reproduction, generated
by running the genuine HLW (2017) code (`tests/fixtures/hlw/HLW_2017_Code/`,
verified against `lw-sv-spec.md` — see `../../../../FIXTURES.md`) on data
from `tests/fixtures/hlw/data/Holston_Laubach_Williams_current_estimates.xlsx`.
Generated because neither supplied workbook publishes 2017-vintage MLE
parameter values or a smoothed/two-sided series for any vintage — see
`FIXTURES.md` for the full picture. Regenerate with:

```
Rscript run_us_2017.R
```

from this directory (needs R with `tis`, `mFilter`, `nloptr` — see
`PREFLIGHT.md` §3 for how those were installed here).

## What this is

- Sample: US, 1961Q1–2019Q2 (`sample.end <- c(2019,2)`, the code's own
  default — chosen deliberately so it lines up with a real published
  vintage for a sanity check, see below).
- Input data: `inputData/rstar.data.us.csv`, the `US input data` sheet from
  `current_estimates.xlsx` trimmed to 1960Q1–2019Q2 (four quarters before
  `sample.start`, as the code requires) and reduced to the four columns
  `run.hlw.R` expects (`gdp.log, inflation, inflation.expectations,
  interest`). Note this is *today's* (2026) revised vintage of GDP/PCE
  data, not the real-time-as-of-2019 vintage HLW actually used — a
  documented, expected source of small discrepancy (see below).
- `prepare.rstar.data.us.R`'s FRED auto-fetch was bypassed entirely (not
  exercised or tested in this session) — we already had comparable data.
- `run.se = FALSE`: the 5000-iteration Monte Carlo standard-error
  procedure (`kalman.standard.errors.R`) was skipped. Not needed for a
  point-estimate oracle; G5a compares smoothed state paths, not CIs.

## Sanity check against genuine published output

`tests/fixtures/hlw/data/Holston_Laubach_Williams_real_time_estimates.xlsx`
sheet `2019Q2` is HLW's own genuinely-published one-sided US output for
this exact sample end (its `info` sheet confirms it's 2017-model output).
Comparing our one-sided series against it, full sample (234 quarters):

| Series | mean diff | max abs diff |
|---|---|---|
| output gap | 0.060 | 0.314 |
| g (annualized) | -0.040 | 0.250 |
| z | 0.015 | 0.105 |
| r* | -0.025 | 0.319 |

Sub-percentage-point agreement across the board, with no systematic drift.
The residual gap is consistent with (a) GDP/PCE data revisions in the ~7
years since the original 2019Q2 run, and (b) our own independent `nloptr`
MLE optimization landing at a very slightly different point than HLW's
original run (different starting values, possibly different local
optimum). Both are expected, documented sources of discrepancy — not bugs.
This does **not** replace exact-match testing: it's a methodology sanity
check, not itself the G5a pass criterion.

## Output files

- `output/us_2017_parameters.csv` — the full MLE parameter vector plus
  `lambda_g`, `lambda_z`, and the derived `sigma_g = lambda_g * sigma_y*`,
  `sigma_z = |lambda_z * sigma_ytilde / a_r|` (the `unpack.parameters.stage3.R`
  `Q` matrix squares this term, so only its magnitude is meaningful — the
  sign is an artifact of `a_r`'s sign).
- `output/us_2017_one_sided.csv` — filtered (real-time) `rstar, g, z,
  output_gap`, one row per quarter, 1961Q1–2019Q2.
- `output/us_2017_smoothed.csv` — smoothed (two-sided) `rstar, g, z,
  output_gap` for the same span. **This is the piece that's otherwise
  unavailable anywhere** — HLW's own `format.output.R` never writes it to
  CSV; `run_us_2017.R` extracts it directly from `out.stage3$*.smoothed`
  (already computed internally by `kalman.states.wrapper.R`, just
  discarded by the stock scripts).

## Using this as the G5a oracle

Fix `stan/functions/ssm_matrices_lw.stan` / the Python KF mirror at the
parameter values in `us_2017_parameters.csv` (with `c` implicitly 1, no
COVID/κ terms — matches spec §1.2–§1.3 exactly), run on the same input
data window, and compare smoothed `rstar, g, z, output_gap` against
`us_2017_smoothed.csv`. Any discrepancy beyond numerical tolerance needs
tracing to a specific cause per spec §5's G5a rule — this reproduction's
own known discrepancy sources (data vintage, optimizer path) are already
named above so they aren't rediscovered from scratch.
