# Handoff

Status as of 2026-08-31 (end of session). **S2 is complete, green, merged
to main, and externally benchmarked.** G1, G2, G5a pass; US and EA runs
track HLW's published estimates (see `STRESS-TESTS.md` for the full
benchmarking + COVID stress-test record, including the full-latest-vintage
experiments). **The next stage is S3 — `plans/S3-plan.md` is the drafted
plan of record**, including the empirical motivation the stress test
supplied and the mandatory KF regression check.

## What passed

**S0 (preflight).** PyPI and CmdStan both work in this environment (CmdStan
needs `version=` pinned — `api.github.com` is blocked; see `PREFLIGHT.md`).
HLW fixtures assembled: the genuine HLW (2017) replication code (matches
`lw-sv-spec.md` term for term), real published data/parameters/estimates,
and a self-derived, sanity-checked G5a oracle built by running HLW's own
code ourselves (`tests/fixtures/hlw/derived/us_2017_reproduction/` — full
detail in `FIXTURES.md`).

**S1 (spec schema, run store, render/compile/run harness, toy model).**
`mtk run` produces an immutable, content-addressed `runs/<hash12>/`;
idempotent re-run; hash-sensitive; refuses to overwrite a crashed partial
run. All still green.

**S2 (LW without SV) — implemented 2026-08-31, all gates green:**

- **G1** (`tests/test_g1_mirror.py`, live): Python KF
  (`macrotoolkit.smoother`, numba) vs Stan KF
  (`stan/functions/kalman_loglik_tv.stan` via the test harness template)
  agree to **3.6e-12** over 50 prior draws — gate is 1e-8.
- **G5a** (`tests/test_g5a_hlw_replication.py`): our KF + RTS smoother at
  HLW's MLE parameters, with HLW's exact dumped `xi.00`/`P.00`, reproduce
  the oracle's log-likelihood and all four filtered *and* smoothed series
  (r*, g, z, gap) to **~1e-12** (gate 1e-8) — the near-machine-precision
  criterion signed off on 2026-08-31. Plus a term-for-term matrix
  cross-check against `unpack.parameters.stage3.R`.
- **G2** (`tests/test_g2_parameter_recovery.py`, `slow` marker, ~38 min):
  20 simulated no-SV datasets, full NUTS each; pooled 90%-CI coverage in
  [0.80, 0.97], per-parameter floor, 3-sigma no-bias test on
  sigma_g/sigma_z — passed 2026-08-31.
- **Qualitative check**: first real US run (`examples/us_lw_sv/`, run
  `24b6288dddad`) — diagnostics **PASS** (0 divergences, max R-hat 1.007),
  every structural coefficient brackets the HLW MLE, smoothed r*/gap track
  HLW's at 0.98 correlation. `sigma_g`/`sigma_z` posteriors sit below
  HLW's MUE values *by design* (the spec's pile-up priors replace MUE);
  consequences documented in `examples/us_lw_sv/README.md`.

Key S2 decisions and traps are in `DECISIONS.md` (2026-08-31 entries):
the confirmed Half-N(0,1²) priors on `sigma_is`/`sigma_pc`; the
constant-covariance-first KF; the dumped-from-R initial conditions (HLW's
`P.00` is an inner MLE product — never re-derive it); and the
`sig_figs=18` requirement for any Stan-vs-Python numeric comparison
(CmdStan's 6-sig-fig CSV default mimics a numerics bug).

A `numerics-reviewer` pass over the S2 KF/state-space code found no
correctness defects (one hygiene fix applied: Cholesky-based RTS gain).

## What's staged, not yet built (S3)

Per spec §7: add the non-centered SV block to `stan/templates/lw_sv.stan.j2`
(the same file — don't fork it), generalize
`stan/functions/kalman_loglik_tv.stan` to time-varying covariance (per the
2026-08-31 decision this MUST include a regression check that the constant
case still reproduces S2's G1/G5a results), retune sampling
(adapt_delta, priors), and pass G3 (SBC, no-SV) + clean sampling on US
data. `LwSvOptions.sv_shocks` and the schema plumbing are already in place
— S3 removes the validator that pins it to `[]`.

Per this repo's ground rules, `stan-engineer`-type work (the SV block's
geometry) should be driven or directly supervised by a human.

## Warnings for whoever builds S3/S4

- **State timing convention** (flagged by the numerics review): state
  slots 4/6 (1-indexed) hold `g_{t-1}`/`z_{t-1}`, NOT `g_t`/`z_t`; the
  reporting mapping (g = slot 4, z = slot 6, r* = g+z, gap = y − slot 1)
  matches HLW's own convention and is validated against the oracle. S4's
  `results_lw.py` must inherit this exact mapping (copy
  `tests/test_g5a_hlw_replication.py::_series_from_states`), not re-derive
  it from the spec's plain-English equations — doing so would introduce a
  silent one-quarter shift.
- In the LW form, SV on the IS/PC shocks makes the **measurement**
  covariance R_t time-varying (those shocks are measurement errors in the
  marginalized state-space), not the state Q_t — spec §2.2's wording
  notwithstanding, that's where S3's generalization actually bites.
- The Durbin-Koopman simulation smoother is S4 scope and deliberately does
  not exist yet; `smoother.py` currently ends at the RTS smoother.
- `default_initial_state` treats state lag slots as a priori independent
  (documented simplification). G5a bypasses it (uses HLW's exact init);
  G1/G2 use it consistently on both sides. Revisit only with a reason.

## Everything else worth knowing

- `PREFLIGHT.md`, `FIXTURES.md`, `DECISIONS.md` are the running record —
  read them before re-deriving anything.
- CmdStan 2.36.0 at `~/.cmdstan` (container-local); `pytest` is a `uv
  tool` install with the project deps + editable `macrotoolkit` (re-run
  the `uv tool install` in `DECISIONS.md` if `pyproject.toml` changes).
- R (+ `tis` from github.com/cran/tis, `nloptr`, `mFilter` via apt) is
  needed only to regenerate the G5a fixture; the install recipe is in
  `DECISIONS.md`/`PREFLIGHT.md` §3. The 2026-08-31 regeneration was
  byte-identical to the committed CSVs.
- The G2 gate is behind `pytest -m slow` (~38 min); the fast suite
  (`pytest -m "not slow"`) is ~20 s and covers G1 + G5a.
