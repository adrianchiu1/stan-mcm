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

- **2026-08-13 — HLW model-variant target for G5a: left open, not decided
  unilaterally.** The checked-in R code and the newly supplied
  `current_estimates.xlsx` are HLW's 2023 COVID-adjusted model; `lw-sv-spec.md`
  specifies the plain 2017 model. This is a modeling-scope decision (whether
  to extend the G5a fixture harness with COVID/κ_t terms, or pursue exact
  2017-vintage parameters some other way), not a fixture-plumbing detail —
  left for the human to decide before S2 begins designing G5a. See
  `FIXTURES.md` for the full option set. Does not block S1.
