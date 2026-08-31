# S2 plan — LW without SV: template, KF functions, priors, Python mirror, HLW fixtures wired in

> **STATUS 2026-08-31: S2 IMPLEMENTED AND COMPLETE.** All gates green — G1
> at 3.6e-12, G5a at ~1e-12 (both vs 1e-8 gates), G2 passed (20 datasets,
> ~38 min, `slow` marker), US posterior PASS and sensible
> (`examples/us_lw_sv/README.md`). Every file in the "Files to create"
> list below exists except `tests/test_g2_parameter_recovery.py`'s
> simulator, which landed as `tests/g2_harness.py`. See `HANDOFF.md` for
> the current state and S3 warnings; the text below is kept as the
> original plan of record.

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
2. `tests/g1_harness.py` + `tests/test_g1_mirror.py` — the mirror-comparison
   harness and parameter-point generator described in the unsupervised-scope
   brief, built against stubs for the not-yet-existing `smoother.py` KF and
   the Stan KF function, so G1's actual test in S2 just wires two real
   functions into an already-correct, already-tested harness rather than
   being designed from scratch under gate pressure. 3 real passing tests
   (point count, reproducibility, every generated point actually satisfies
   the spec's prior constraints) + 1 structurally-complete-but-skipped test
   for the real G1 comparison.
3. `tests/test_units_conventions.py` promoted from documented skips to
   real, passing numeric-pinning tests of the three conventions themselves
   (not of production code, which doesn't exist yet) — see that file for
   the worked examples. When S2 lands real conversion code, extend/rewire
   these to import and check it directly.
4. This session's `FIXTURES.md` already covers the HLW fixtures
   cataloguing item from the brief — nothing further needed there.

## Open questions for S2 implementation (not blocking this prep pass)

> **Status update 2026-08-31** (see `DECISIONS.md` entries of the same
> date): question 2 is **resolved** — build the constant-covariance KF
> for S2, generalize to time-varying `Q_t` in S3. The G5a-tolerance
> clarification below ("near machine precision against our own
> reproduction") is **signed off** by the user. Question 0 is
> **closed** — HLW's own convention is documented (MLE, no priors,
> `σ_ỹ`/`σ_π` free unbounded constants) and the user confirmed
> option (a): **Half-N(0, 1²) priors on `σ_IS`, `σ_PC`** — this is now
> the spec for `specs/schema/lw_sv.py`'s priors and the Stan template.
> Questions 1 and 3 remain open as written (question 1 is a
> confirm-the-default, question 3 is a build item, neither blocks the
> start of S2).

0. **No prior exists anywhere in the spec for the no-SV variant's constant
   IS/Phillips shock scales.** Surfaced while building the G1 parameter-
   point generator (`tests/g1_harness.py`), worth a closer look than the
   three below since it's a genuine spec gap, not just an implementation
   trade-off. Spec §1.4's shocks table says `ε_IS`/`ε_PC` variance is *always*
   `exp(h_IS,t)`/`exp(h_PC,t)` — SV, full stop, no constant-variance case is
   ever described for them (unlike `σ_y*, σ_g, σ_z`, which are constant in
   every variant). Spec §1.6's priors table accordingly has no entry for a
   constant analogue (call it `σ_IS`/`σ_PC`, matching HLW's own `σ_ỹ, σ_π`
   naming) — every prior it lists is either a structural coefficient, one
   of the three constant-innovation shocks, or an SV-path parameter. But
   spec §7's S2 row is explicitly "LW **without** SV" as a real, separate
   build stage before S3 "adds" the SV block — which only makes sense if
   the no-SV variant has *some* constant value standing in for
   `exp(h_IS,t)`/`exp(h_PC,t)` in the meantime. G1's parameter-point
   generator (this pass) sidesteps this by only sampling the 8 parameters
   the priors table actually specifies — it does not need `σ_IS`/`σ_PC`
   itself, since G1 only checks that two log-likelihood implementations
   agree at a given parameter point, not that the point is "complete" by
   some model-specification standard. But S2's actual Stan/Python KF code
   does need a concrete answer. Candidate options for whoever picks up S2
   (not resolved here): (a) reuse HLW's own convention directly — treat
   `σ_IS`/`σ_PC` exactly like HLW's `σ_ỹ`/`σ_π` (constant, estimated by
   the sampler with some weakly-informative prior, no relation to the SV
   parameters at all) since that's literally what "LW without SV" should
   mean structurally; (b) center a prior on the values already in
   `tests/fixtures/hlw/derived/us_2017_reproduction/output/us_2017_parameters.csv`
   (`sigma_ytilde≈0.34`, `sigma_pi≈0.80` for the US) as an empirically-
   grounded weakly-informative prior. (a) seems like the more obviously
   correct reading of "without SV" and doesn't require deciding on a made-
   up prior scale, but flagging both since this needs a real decision, not
   an assumption, before S2's Stan template or Pydantic priors schema can
   be written.

1. **`estimate_c` for the S2/G5a run**: HLW (2017)'s stage-3 model fixes
   `c` implicitly at 1 (`r* = g + z`, no `c` term at all — see
   `tests/fixtures/hlw/HLW_2017_Code/HLW_Code_Guide.pdf` §7.5). Spec's
   default is also `c` fixed at 1.0. No conflict — just confirming G5a
   should run with `estimate_c: false` (the default), not the alternative.
2. **`kalman_loglik_tv.stan`'s generality vs. S2's actual need** —
   **RESOLVED 2026-08-31 (user decision)**: build the simpler
   constant-covariance version for S2, match HLW via G5a, then generalize
   to the time-varying-`Q_t` form when S3 adds SV. Spec §2.2's
   time-varying contract is deferred, not dropped: S3's first task is that
   generalization plus a regression check that the constant case still
   reproduces S2's G1/G5a results after the rewrite.
3. **G2's synthetic-data generator**: needs its own parameter-point +
   state-path simulator (distinct from G1's parameter-point generator,
   which doesn't simulate data, just samples parameter values). Not
   built in this prep pass — no fixture dependency, purely synthetic, so
   no reason to front-load it; flagging only so it isn't forgotten as a
   separate piece of work from G1's scaffold.
