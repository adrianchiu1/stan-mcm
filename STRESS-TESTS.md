# Stress tests & external-benchmark record — S2 (LW without SV)

Consolidated record of every check the S2 model has been put through
beyond the automated gates, as of 2026-08-31. The automated gates
themselves (G1/G2/G5a — all green) are documented in `HANDOFF.md` and the
test suite; per-run detail lives in the example READMEs
(`examples/us_lw_sv/`, `examples/ea_lw_sv/`). The interactive exhibit for
everything below is the "r* Replication Scorecard" artifact (published
from the 2026-08-31 session; regenerate from the run store + the
workbooks if needed).

## 1. The three comparison methods, ranked

1. **Fixed-parameter exact replication** (strongest — automated as G5a):
   freeze the system matrices at HLW's own MLE parameters and their exact
   initial conditions, run our filter + smoother on their input data, diff
   every series. Same linear-Gaussian computation ⇒ any gap is a bug.
   Result: max |Δ| ≈ 1e-12 across loglik and all filtered + smoothed
   series (`tests/test_g5a_hlw_replication.py`).
2. **Parameter table**: posterior medians/90% CIs vs HLW's published MLE
   point. Fast sanity check; makes the deliberate prior-driven deviations
   (σ_g, σ_z) visible rather than hidden.
3. **State-path overlay**: our filtered (one-sided) states vs published
   one-sided series, like for like, with correlations and mean |level|
   gaps. Bundles Bayesian-vs-MLE, priors, model vintage, and data
   revisions — informative, not a bug detector.

## 2. Pre-COVID reference runs (the S2 acceptance results)

Estimated on pre-2020 windows, compared against the published one-sided
series (correlation / mean |diff|, filtered):

| | window | r* | g | gap | z | diagnostics |
|---|---|---|---|---|---|---|
| US `24b6288dddad` | 1961Q1–2019Q2 | 0.955 / 0.40pp | 0.938 / 0.26pp | 0.967 / 0.22pp | 0.957 / 0.42pp | PASS, 0 div |
| EA `b6a26e973787` | 1972Q1–2019Q4 | 0.953 / 0.28pp | 0.991 / 0.12pp | 0.978 / 0.42pp | 0.769 / 0.24pp | PASS, 0 div |

(Overlay target: the NY Fed current-estimates workbook, 2023 COVID model —
the same file as the published `newyorkfed.org` current_estimates.xlsx,
from the copy committed in `tests/fixtures/hlw/data/`; the site itself is
blocked from the dev environment.) Known, documented level difference: the
spec's σ_g/σ_z pile-up priors (replacing HLW's MUE machinery) produce a
smaller σ_z, hence a flatter late-sample r* decline (~+0.5–0.9pp vs HLW by
2019). EA z correlates lower (0.77) because the 2023 model's estimated
c≈0.97 splits r* between g and z differently than our fixed c=1.

## 3. COVID stress test: full latest-vintage re-estimation (2026-08-31)

**Design.** Re-estimate both economies on the full current vintage through
2026Q1, COVID quarters included, with NO COVID machinery — HLW's 2023
model scales 2020Q2–Q4 shock variances by κ ≈ 7.6 (US) / 19.8 (EA) and
adds a dummy; the S2 model deliberately has neither. Specs:
`examples/*/spec_full_vintage.yaml` (marked experiments; the pre-COVID
runs remain the reference results).

**Findings.**

- **Sampler robustness:** geometry survived the outliers essentially
  intact. US `18f32ac9793f`: 0 divergences, max R-hat 1.002 (WARN only
  for tail ESS 396 vs the 400 threshold). EA `ab363d5026a1`: 2 divergences
  in 6,000 draws (strict FAIL flag; R-hat 1.007, ESS > 1000) — marginal.
- **Absorption channels differ by economy.** With no COVID variance
  scaling available, the US model pushes 2020 into the potential-level
  shock (σ_y* 0.54 → 0.94; gap AR flattens, a₁ 1.44 → 1.15). The EA
  pushes it into the demand shock (σ_IS 0.41 → 0.99) and the gap's AR
  structure collapses (a₁,a₂ = 1.49,−0.55 → 0.77,+0.01), with σ_y* also
  roughly doubling.
- **Tracking degrades but does not break:** US r*/g/gap/z correlations
  0.914/0.871/0.845/0.871 (vs 0.94–0.97 pre-COVID); EA
  0.929/0.961/0.861/0.711. At 2026Q1 our filtered US r* is +2.4 vs HLW's
  +1.1 — the pile-up-prior σ_z effect, amplified by COVID absorption.
- **Conclusion:** the contaminated-parameter mechanism is exactly the
  failure mode S3's stochastic volatility is designed to remove — SV lets
  the IS/PC shock variances spike endogenously in 2020, the Bayesian
  counterpart of HLW's hand-set κ's. This stress test is the empirical
  motivation for S3.

## 4. Determinism / fixture regeneration check

Re-running the HLW (2017) R reproduction end-to-end in a fresh container
(R reinstalled from the documented recipe) reproduced every committed
fixture CSV **byte-identically**, while adding the exact `xi.00`/`P.00`
initial conditions G5a consumes. See
`tests/fixtures/hlw/derived/us_2017_reproduction/README.md`.

## 5. Coverage limits (what has NOT been stress-tested)

- **UK:** published 2017-model estimates are in the fixtures (2019Q2
  real-time vintage — the last vintage carrying the UK), but no UK input
  data exists in the repo (the NY Fed dropped UK input sheets; HLW's 2017
  code builds them from local ONS/BOE raw files). A UK run is turnkey
  once four input series are supplied in the `rstar.data.us.csv` shape.
- **Germany:** HLW publish no Germany model; the EA aggregate is the
  closest published benchmark. A Germany-only run needs user-supplied
  data and a non-HLW benchmark (Bundesbank/ECB r* studies).
- **Canada** is available in the workbook (input + published estimates)
  and simply hasn't been run.
- G3/G4 (SBC), G5b (Bayesian tracking band 2000–2019, ±50bp/±0.5pp
  formal criterion), and G6 are later-stage gates, not yet implemented.
