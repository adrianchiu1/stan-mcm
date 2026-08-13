# S2 plan — LW without SV: template, KF functions, priors, Python mirror, HLW fixtures wired in

**Status: plan only.** Per the project's ground rules, this stage's actual
implementation (anything under `stan/`, `specs/schema/lw_sv.py`,
`src/macrotoolkit/smoother.py`) is not started by this pass — only
non-gated prep (this plan, G1 test scaffolding against a stub, units-
convention unit tests). `stan-engineer` (the agent that would author
`stan/functions/kalman_loglik_tv.stan`, `stan/functions/ssm_matrices_lw.stan`,
and `stan/templates/lw_sv.stan.j2`) is explicitly not to be invoked in an
unsupervised session per this repo's agent roster (`.claude/agents/`) — a
human should drive or directly supervise that work. This document is what
it should do.

Spec §7 acceptance test (verbatim): *"G1, G2, G5a pass; posterior sensible
on US data."* Restated concretely per spec §5:

- **G1**: Python KF log-likelihood matches Stan KF log-likelihood to
  `max abs diff < 1e-8` across 50 random parameter points (drawn from/near
  the priors in spec §1.6, no-SV variant — constant `σ_y*, σ_g, σ_z`, no
  `h_IS,t`/`h_PC,t`).
- **G2**: parameter recovery on 20 simulated (no-SV) datasets — true
  values inside each parameter's 90% posterior CI ≈90% of the time; no
  systematic bias in `σ_g`, `σ_z` specifically (spec flags these as the
  parameters pile-up priors are meant to discipline).
- **G5a**: our KF/smoother at HLW's published (here: reproduced, see
  below) parameter values reproduces their smoothed `r*, g, gap` to
  numerical tolerance, any residual traced to a named cause.
- Plus a qualitative check: posterior on real US data (1961–2019, using
  the same window as the G5a fixtures) looks sensible — no formal pass/fail
  gate, but a real run + eyeball check, presumably backing the worked
  example that becomes `examples/us_lw_sv/`.

## G5a oracle: already resolved, not an open question

`tests/fixtures/hlw/derived/us_2017_reproduction/` (built during S0/S1 prep)
is the G5a oracle: `output/us_2017_parameters.csv` (the parameter point to
fix our system matrices at) and `output/us_2017_smoothed.csv` (the target
smoothed `r*, g, z, output_gap` series, 1961Q1–2019Q2). **Important
clarification for whoever builds G5a**: the tolerance here should be near
machine precision (matching G1's spirit), *not* the ~0.06pp mean / 0.31pp
max deviation documented in that fixture's `README.md`. That deviation is
between our reproduction and the genuinely-published 2019Q2 vintage (a
different, real discrepancy: data revisions since 2019, independent
optimizer path) — it is not a tolerance our KF/smoother needs to
accommodate. Our KF/smoother should match `us_2017_smoothed.csv` almost
exactly, because both are literally evaluating the same linear-Gaussian
model at the same parameter values — any gap beyond floating-point-level
disagreement is a real bug in our KF, full stop. Don't conflate the two
discrepancy sources.

## Files to create (S2 implementation proper — not this pass)

```
specs/schema/lw_sv.py            # LwSvOptions: sv_shocks, estimate_c (spec §2.3 draft)
                                  # register in specs/schema/__init__.py FAMILY_REGISTRY

stan/functions/kalman_loglik_tv.stan   # KF log-lik, time-varying Q_t support (unused by
                                        # S2's constant-variance case, but built to spec
                                        # §2.2's general contract so S3 doesn't need a rewrite)
stan/functions/ssm_matrices_lw.stan    # assembles Z, T, R, c, d for the LW form

stan/templates/lw_sv.stan.j2     # S2 renders WITHOUT the SV blocks (no h_IS,t/h_PC,t,
                                  # constant sigma_y*/sigma_g/sigma_z) -- S3 adds the
                                  # non-centered SV block to this same template, per
                                  # spec §7's staged build (S2 -> S3 grow one file, not
                                  # two divergent ones)

src/macrotoolkit/smoother.py     # S2 scope: numba-accelerated KF log-likelihood mirror
                                  # ONLY (needed for G1). The Durbin-Koopman simulation
                                  # smoother (spec §2.4) is S4 scope -- don't build it
                                  # early just because the file name suggests the whole
                                  # thing; grow this file in S4, don't front-load it.

examples/us_lw_sv/data/us_quarterly.csv  # can be derived directly from
                                  # tests/fixtures/hlw/data/*current_estimates.xlsx's
                                  # "US input data" sheet: gdp.log -> y (as 100*ln, per
                                  # spec §1.1's "100 x ln(real GDP)" -- note the HLW raw
                                  # column is un-scaled ln(GDP), needs the x100), inflation
                                  # column already matches spec's "annualized q/q %" core
                                  # PCE convention directly, interest - inflation.expectations
                                  # = the real rate spec wants pre-constructed (locked
                                  # decision #3). Reuses data already in hand -- no new
                                  # fetch needed.
examples/us_lw_sv/spec.yaml      # family: lw_sv, sv_shocks: [] for S2 (S3 flips this on)

tests/test_g1_mirror.py          # promote from this pass's stub (see below) to a real,
                                  # passing test once smoother.py + the Stan functions exist
tests/test_g2_parameter_recovery.py
tests/test_g5a_hlw_replication.py   # loads tests/fixtures/hlw/derived/us_2017_reproduction/
```

## This pass's actual deliverables (non-gated prep, done now)

1. This plan.
2. `tests/g1_scaffold.py` (or similar) — the mirror-comparison harness and
   parameter-point generator described in the unsupervised-scope brief,
   built against stubs for the not-yet-existing `smoother.py` KF and the
   Stan KF function, so G1's actual test in S2 just wires two real
   functions into an already-correct, already-tested harness rather than
   being designed from scratch under gate pressure.
3. `tests/test_units_conventions.py` promoted from documented skips to
   real, passing numeric-pinning tests of the three conventions themselves
   (not of production code, which doesn't exist yet) — see that file for
   the worked examples. When S2 lands real conversion code, extend/rewire
   these to import and check it directly.
4. This session's `FIXTURES.md` already covers the HLW fixtures
   cataloguing item from the brief — nothing further needed there.

## Open questions for S2 implementation (not blocking this prep pass)

1. **`estimate_c` for the S2/G5a run**: HLW (2017)'s stage-3 model fixes
   `c` implicitly at 1 (`r* = g + z`, no `c` term at all — see
   `tests/fixtures/hlw/HLW_2017_Code/HLW_Code_Guide.pdf` §7.5). Spec's
   default is also `c` fixed at 1.0. No conflict — just confirming G5a
   should run with `estimate_c: false` (the default), not the alternative.
2. **`kalman_loglik_tv.stan`'s generality vs. S2's actual need**: spec
   §2.2 says this function "must handle... time-varying state innovation
   covariance" — S2's no-SV model doesn't need that yet (S3 does). Building
   the general time-varying form now (unused until S3) vs. a simpler
   constant-covariance version now and generalizing in S3 is a real
   engineering trade-off for `stan-engineer` to make when this stage
   starts — flagging so it's a conscious choice, not an oversight either
   way.
3. **G2's synthetic-data generator**: needs its own parameter-point +
   state-path simulator (distinct from G1's parameter-point generator,
   which doesn't simulate data, just samples parameter values). Not
   built in this prep pass — no fixture dependency, purely synthetic, so
   no reason to front-load it; flagging only so it isn't forgotten as a
   separate piece of work from G1's scaffold.
