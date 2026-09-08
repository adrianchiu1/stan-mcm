# Run sessions: publication-length runs and the heavy examples

The companion's numbers come from smoke runs (2 chains × 300/300 for most
examples; the diagnostics verdicts say WARN at that length). The run
sessions below replace them with publication-length runs and execute
the examples too heavy for the writing session. They are execution
tasks: no code changes, no prior changes, every result reported as
measured.

Common prompt header for every session:

```
You are running the CCBS-handbook companion examples for macrotoolkit
(repo: adrianchiu1/stan-mcm). This is an EXECUTION task: no changes to
src/, stan/ or specs/; no edits to priors or samplers except the chain
lengths named below. Environment per HANDOFF.md (CmdStan 2.36.0 pinned;
xlrd and openpyxl for the handbook data). Confirm pytest -m "not slow"
is green first. Work on the companion branch (or a branch from it named
claude/companion-runs-<slug>). Report every run as measured: hash,
verdict and reasons, the fit-time mirror check, the fast validation
tier. Do not re-seed, shorten or retune a run that WARNs or FAILs;
record it and say why it might. Commit and push after each example.
```

## Session A — Chapters 1, 2 (light), 3 (light), 5: publication length

For each of `ch1_ar2`, `ch1_ar2_ar1err`, `ch2_bivar_minnesota`,
`ch2_var4_monthly_cholesky`, `ch2_steady_state`, `ch2_conditional`,
`ch3_uc_trend_cycle`, `ch3_tvp_regression`, `ch5_sv_uk_inflation`,
`ch5_sv_ucsv_form`, `ch5_tvp_ar1_sv`:

1. Copy the spec to a sibling directory `<name>_pub/spec.yaml` with the
   sampler set to 4 chains × 1000 warm-up / 1000 sampling (keep every
   other field; the run identity changes with the sampler, so this is a
   new run, not a rerun).
2. `mtk run <name>_pub/spec.yaml`, then `mtk validate <name>_pub/spec.yaml
   --tier fast`, then `mtk report <hash>`.
3. Append a "Publication run record" section to the example's README
   with the same fields as the smoke record, and the parameter table's
   medians and 90% intervals.
4. For `ch1_ar2_ar1err` report the per-chain posterior medians of `rho`
   and `b1`: the companion states this posterior is bimodal; the chains
   should show which mode(s) they occupy.
5. Re-execute the chapter notebooks against the `_pub` specs by pointing
   `build_notebooks.py`'s `fit()` at the `_pub` directories (a one-line
   change in the notebook builder's `HB / name` path is acceptable for
   this session) with `--execute`; commit the executed notebooks.

Expected wall time: under two hours in total.

## Session B — the 11-variable sign-restriction VAR (Chapter 2 §6)

`ch2_signs_11var` has 319 sampled parameters and timed out at the S8
session length. Run it as 2 chains × 400/400 first (record the wall time
per iteration), then, if under 90 minutes, as 4 chains × 800/800. After
the fit run the post-processor exactly as `examples/handbook/run_smoke.py`
does for this example (seven impact restrictions, `max_tries=2000`) and
the `closest_to_median=100` variant; record the acceptance counts and the
median impact of the monetary-policy shock on each of the eleven
variables at horizons 0, 4, 8, 12. Append the record to the README and
execute the Chapter 2 notebook with `HANDBOOK_HEAVY=1`.

Expected wall time: two to four hours.

## Session C — the two factor models and the TVP-VAR (Chapters 3 and 7)

- `ch3_dfm_uk_panel` (40 series, 6 states): 2 chains × 400/400; record
  the smoothed factors' correlation with the first three principal
  components of the standardised panel (the handbook's starting values)
  and the loadings' posterior medians for the ten largest.
- `ch3_tvp_var` (21 coefficient states): 4 chains × 800/800; record the
  dated IRFs to the funds-rate shock at 1975Q1, 1995Q1, 2008Q4 at
  horizons 1, 4, 8 and the six scale posteriors, and state whether the
  funds-rate equation's drift scale still sits far above its 0.01 prior
  (the companion's Chapter 3 §5 discusses this).
- `ch7_dfm_sv` (40 series, 3 factors with SV): 2 chains × 400/400; record
  the three factor-volatility paths' medians at the panel's start,
  middle and end, and the factor AR coefficients.

Execute the Chapter 3 and Chapter 7 notebooks with `HANDBOOK_HEAVY=1`.
Expected wall time: four to eight hours; each example is a separate
commit so a container loss costs one example.

## Session D — certification of one companion model (optional)

Register and run an SBC design for `ch3_uc_trend_cycle` (the cleanest
PASS in the companion) following the pre-registration entry pattern in
`DECISIONS.md` (the UCSV and TVP-AR(1)-SV entries): 100 replications,
T = 100, the five static parameters ranked, fixed anchors, seeds
recorded before the first replication. Report as measured; do not adjust
the design afterwards.

## What comes back into the companion

Each session's records replace the smoke numbers in the corresponding
chapter tables (the chapters cite run hashes, so the replacement is a
visible edit), the executed notebooks are committed in place, and the
figures under `docs/companion/figures/` are regenerated from the
publication runs. The companion's text does not otherwise change unless
a publication run contradicts a smoke-run statement — in which case the
text changes and says so.
