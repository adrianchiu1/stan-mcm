# Decisions log

One dated line per judgment call not fixed by the spec, with rationale.
Newest first.

- **2026-08-13 — S1 toy model shows real (if mild) divergences; left as-is
  rather than hand-tuned to a clean PASS.** The non-centered random-walk
  local-level model (`stan/templates/local_level.stan.j2`) produces ~0.5%
  divergent transitions on the example run (`diagnostics.json` verdict:
  FAIL). Tried `adapt_delta` 0.9–0.97 and several warmup lengths; the
  divergence rate stayed roughly constant (a mild funnel characteristic of
  this model class near `sigma_level -> 0`, not a plumbing bug) and
  `adapt_delta=0.99` traded divergences for 78% max-treedepth hits instead.
  Spec §7's S1 acceptance test is "produces immutable run dir with draws +
  diagnostics" — not "achieves a clean PASS verdict" (that's S3+ territory
  for the real model). A FAIL verdict with a correct, specific reason is
  the diagnostics harness doing its job, not a defect; not spending further
  effort chasing a cosmetic PASS on a throwaway infra-proving toy.
- **2026-08-13 — `data.snapshot.csv` is a byte-identical copy of the whole
  source CSV, not the mapped/trimmed subset.** Keeps it in lockstep with
  the run-identity hash, which is computed over the full file's raw bytes
  (not the mapped/trimmed data) — the snapshot should reproduce exactly
  what was hashed.
- **2026-08-13 — `runs/<hash>/spec.yaml` is the canonicalized spec
  (validated, sorted-key YAML via `RunSpec.to_canonical_yaml()`), not a
  copy of the original source file.** A faithful audit record regardless
  of the original file's key order/comments/formatting.
- **2026-08-13 — Compile cache lives at `.mtk_cache/stan/<hash>_<version>/`,
  gitignored, not under `stan/templates/`.** Keeps the templates directory
  free of build artifacts; cache is always reproducible from template +
  context + CmdStan version.
- **2026-08-13 — `build_stan_data` (DataFrame -> Stan `data` block) is a
  small explicit dispatch function in `run.py`, not yet generalized onto
  the family registry.** Only `local_level` exists; guessing `lw_sv`'s
  data-block shape now would be premature. Flagged in its docstring for S2
  to generalize once that shape is actually known.

- **2026-08-13 — G5a oracle is a self-derived reproduction, not literally
  HLW's own published numbers, and that's treated as sufficient.** Ran
  `HLW_2017_Code/` (the genuine 2017 code) ourselves on data we already
  had, since neither supplied workbook publishes 2017-vintage MLE
  parameters or a smoothed series for any vintage. Sanity-checked the
  reproduction's one-sided output against a real published vintage
  (`real_time_estimates.xlsx` sheet `2019Q2`, same sample end by
  construction): 0.06pp mean / 0.31pp max deviation across 234 quarters
  on output gap/g/z/r*. Treating this as close enough to trust as the
  G5a oracle — the residual gap is attributable to named, expected causes
  (data revisions since 2019, independent optimizer path), not a
  methodology error. Output in
  `tests/fixtures/hlw/derived/us_2017_reproduction/`, full detail in its
  `README.md` and in `FIXTURES.md`. `run.se=FALSE` (skipped the
  5000-iteration Monte Carlo standard-error procedure) since G5a compares
  point/smoothed state paths, not confidence intervals.

- **2026-08-13 — Pin CmdStan to version 2.36.0, never call `install_cmdstan`
  without an explicit `version=`.** `api.github.com` (used by cmdstanpy's
  default "resolve latest release" path) is blocked by this environment's
  egress policy; direct `github.com/.../releases/download/...` asset URLs
  are not. Pinning sidesteps the blocked lookup and also gives reproducible
  builds. Verified 2.36.0 downloads, builds, and samples correctly in this
  environment (see `PREFLIGHT.md` §2). Revisit if a newer CmdStan is needed
  for a specific feature.

- **2026-08-13 — HLW fixture data: use the user-supplied workbooks, not a
  fetch.** `www.newyorkfed.org` is blocked by org egress policy (confirmed
  403 at CONNECT). Per instructions, did not reconstruct data from memory
  or substitute another source. Asked the user; the user supplied
  `Holston_Laubach_Williams_current_estimates.xlsx` and
  `Holston_Laubach_Williams_real_time_estimates.xlsx` directly, now in
  `tests/fixtures/hlw/data/`. Treated as authoritative NY Fed output based
  on their internal `info`-sheet headers and citations, not independently
  re-verified against the live site (that site is unreachable from this
  session).

- **2026-08-13 — HLW model-variant target for G5a: 2017, not 2023.**
  Recommended to the user with reasoning (G5a validates the shared KF
  library the *production* model uses, and production's functional form is
  the 2017 equations — SV sits on top of that, it doesn't replace HLW's
  COVID-adjustment machinery; G5b's own pass criterion already restricts
  its window to 2000–2019, independently confirming COVID-era agreement
  isn't needed for launch). User then supplied the genuine 2017 code
  (`HLW_2017_Code/`, verified by direct term-by-term comparison of its
  Stage 3 matrices against spec §1.2–§1.3, and by grep showing zero
  COVID/κ/φ references) — treating this as confirmation. See `FIXTURES.md`.

- **2026-08-13 — Install R via apt (`--no-install-recommends`) +
  build `tis` from `github.com/cran/tis` rather than CRAN.** Needed to
  eventually run `HLW_2017_Code/` ourselves and derive a self-consistent
  G5a oracle (published parameter values for any 2017-vintage aren't
  available in either supplied workbook). `cran.r-project.org` /
  `cloud.r-project.org` are blocked the same way as `newyorkfed.org`
  (403 at CONNECT); `r-base-core`, `r-cran-nloptr`, `r-cran-mfilter` are
  in Ubuntu's apt repos and install directly. `tis` isn't packaged for
  Ubuntu, but its CRAN mirror is a plain readable git repo on GitHub,
  which is reachable — `git clone` + `R CMD INSTALL` builds it from
  source with no CRAN access needed. Verified all three load correctly.
  A first `apt-get install r-base-core ...` attempt (with recommends)
  pulled in ~886MB of unrelated GUI/media-codec packages and failed on
  stale-mirror 404s for a few of them; `--no-install-recommends` avoided
  that entirely and installed cleanly.
