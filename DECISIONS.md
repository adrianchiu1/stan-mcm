# Decisions log

One dated line per judgment call not fixed by the spec, with rationale.
Newest first.

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
