# Handoff

Status as of 2026-08-13. S1 is complete and green. S2 is planned and
non-gated prep is done; S2 implementation itself has not started.

## What passed

**S0 (preflight).** PyPI and CmdStan both work in this environment (CmdStan
needs `version=` pinned — `api.github.com` is blocked; see `PREFLIGHT.md`).
HLW fixtures assembled: the genuine HLW (2017) replication code (matches
`lw-sv-spec.md` term for term), real published data/parameters/estimates
(both the 2023 COVID-adjusted vintage and genuine pre-2020 2017-vintage
one-sided output), and a self-derived, sanity-checked G5a oracle built by
running HLW's own code ourselves (`tests/fixtures/hlw/derived/us_2017_reproduction/`
— full detail in `FIXTURES.md`).

**S1 (spec schema, run store, render/compile/run harness, toy model).**
`mtk run examples/toy/spec.yaml` produces an immutable, content-addressed
`runs/<hash12>/` with `spec.yaml`, `data.snapshot.csv`, `draws.nc`,
`diagnostics.json`, `log.txt`; idempotent re-run; hash-sensitive to any
spec/data change; refuses to silently overwrite a crashed partial run.
84 implementation tests + a fully independent verification pass (I ran the
acceptance test, idempotency check, and hash-sensitivity check by hand,
not just trusting the building agent's report). Everything committed and
pushed to `claude/stan-mcm-preflight-hn60cj`.

**S2 prep (non-gated).** `plans/S2-plan.md` written (files S2 will create,
G1/G2/G5a restated concretely, the G5a-tolerance clarification below).
`tests/g1_harness.py` + `tests/test_g1_mirror.py`: the G1 mirror-comparison
harness and parameter-point generator, built against stubs for the KF
functions S2 hasn't written yet — 3 real passing tests (point count,
reproducibility, every point satisfies the spec's prior constraints) + 1
structurally-complete-but-skipped test that only needs the stubs swapped
for real functions once S2 lands. `tests/test_units_conventions.py`
promoted from skipped placeholders to 3 real, numeric-pinning tests of the
three unit conventions (with explicit negative checks for the exact bug
classes the spec warns about — the `exp(h)` vs `exp(h/2)` factor-of-two,
the `g` vs `g/4` mixup, the `100*` vs `400*` scaling slip). Full suite:
**90 passed, 1 skipped.**

## What's staged, not yet built

S2 implementation proper: `specs/schema/lw_sv.py`, `stan/functions/kalman_loglik_tv.stan`,
`stan/functions/ssm_matrices_lw.stan`, `stan/templates/lw_sv.stan.j2`,
`src/macrotoolkit/smoother.py` (KF log-likelihood only — the Durbin-Koopman
simulation smoother is S4 scope, don't front-load it), `examples/us_lw_sv/`.
None of this exists yet. Per this repo's ground rules, `stan-engineer`
(the agent that would author the Stan files) is not invoked without a
human directly driving/supervising that pass — this handoff is the
checkpoint for that decision, among the others below.

## Open questions — need your input before S2 implementation starts

> **Update 2026-08-31** — user reviewed these; see the three `DECISIONS.md`
> entries of that date for the full record. Question 2 is **resolved**
> (constant-covariance KF for S2, generalize in S3 with a regression check).
> Question 3 is **signed off** (G5a tolerance = near machine precision
> against our own reproduction). Question 1 is answered but not closed:
> HLW's own convention is straight MLE with no priors — `σ_ỹ`/`σ_π` are
> free, unbounded constants initialized at IS/PC OLS residual SDs, and the
> pile-up machinery (`λ_g`, `λ_z`) never touches them. Standing
> recommendation: option (a) with Half-N(0, 1²) on `σ_IS`, `σ_PC` —
> **awaiting user confirmation** before writing `specs/schema/lw_sv.py`'s
> priors. That confirmation is the last input S2 implementation is
> blocked on.

1. **No prior exists anywhere in the spec for the no-SV variant's constant
   IS/Phillips shock scales.** Real spec gap, surfaced while building the
   G1 harness — not a nit. Spec's shocks table (§1.4) has `ε_IS`/`ε_PC`
   variance as *always* `exp(h_IS,t)`/`exp(h_PC,t)` (SV, no constant-
   variance case ever described), so §1.6's priors table has no entry for
   a constant stand-in. But S2 is explicitly "LW **without** SV" as its
   own build stage before S3 adds SV — which needs *some* constant value
   in the meantime. Two candidate resolutions in `plans/S2-plan.md`'s
   open-questions section: (a) treat it exactly like HLW's own `σ_ỹ, σ_π`
   (freely estimated, weakly-informative prior, no relation to SV
   parameters — the more obviously "correct" reading of "without SV"), or
   (b) center a prior on the values already in
   `tests/fixtures/hlw/derived/us_2017_reproduction/output/us_2017_parameters.csv`
   (`σ_ỹ≈0.34, σ_π≈0.80` for the US). Needs a real decision, not an
   assumption.
2. **`kalman_loglik_tv.stan`'s generality vs. S2's actual need**: build
   the full time-varying-covariance form now (spec §2.2's stated contract,
   unused until S3) or a simpler constant-covariance version now,
   generalized in S3? Real engineering trade-off, not resolved here —
   `plans/S2-plan.md` open question 2.
3. Carried over from S0/S1, resolved for practical purposes but worth
   your explicit sign-off before it's load-bearing: the G5a oracle is a
   *self-derived reproduction* of HLW's 2017 code (run by us on data we
   had, since neither NY Fed workbook you supplied publishes 2017-vintage
   parameters or a smoothed series for any vintage), not literally HLW's
   own published numbers. Sanity-checked to 0.06pp mean / 0.31pp max
   deviation against a genuinely-published vintage — see `DECISIONS.md`
   and `tests/fixtures/hlw/derived/us_2017_reproduction/README.md`.
   `plans/S2-plan.md` clarifies G5a's actual tolerance should be against
   *this* reproduction (near machine precision expected, since it's the
   same computation), not against that 0.06/0.31pp gap.

## Everything else worth knowing

- `PREFLIGHT.md`, `FIXTURES.md`, `DECISIONS.md` are the running record of
  every environment fact and judgment call this session made — read those
  before re-deriving something that's already answered there.
- CmdStan lives at `~/.cmdstan/cmdstan-2.36.0` (container-local, not
  committed). `pytest` is a `uv tool` with the project's deps + an
  editable `macrotoolkit` install (also container-local) — see
  `DECISIONS.md`'s last entry if `scripts/gate-check.sh` ever mysteriously
  fails to import `macrotoolkit` again; re-run the `uv tool install`
  command there if `pyproject.toml`'s deps change.
- Nothing under `stan/` or `specs/schema/lw_sv.py` exists yet — S2 starts
  from a clean slate there, guided by `plans/S2-plan.md`.
