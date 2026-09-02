# S4 plan — DK/FFBS simulation smoother, four output modules, HTML report

**Status: plan of record, drafted 2026-08-31 at S3 close, open questions
resolved with the user the same day (see "Resolved decisions" below).
Implementation starting now.** Spec §7's S4 row: "DK smoother; four output
modules; HTML report | G6 pass; report renders all figures from a real
run."

## Scope

Per spec §2.4, §3.1–3.5, §5 (G6 row): a per-draw joint simulation smoother
in Python (`smoother.py`, extending — not replacing — S2/S3's KF/RTS
machinery), the reporting layer that turns smoother draws into the four
output modules, and the HTML report that assembles them. No `stan/`
changes: the smoother consumes each posterior draw's saved parameters
(statics + `h_is`/`h_pc` transformed parameters for SV runs) exactly as
HANDOFF.md's warnings specify — it does not touch the Stan program at all.

## Files touched

- `src/macrotoolkit/smoother.py` — add the simulation smoother (algorithm
  choice below) as new functions alongside the existing filter/RTS code;
  add a per-draw entry point that takes a draw's static params + h paths
  (via `sv_diag_variance_path`, never re-derived from `nu`) and returns a
  joint state-path draw plus the structural shock path it implies.
- `src/macrotoolkit/results_lw.py` (new) — `LWResults`: loads a run's
  `draws.nc`, runs the simulation smoother across the configured draw
  subset, and exposes trend-cycle series, IRFs, fan-chart paths, and
  historical-decomposition contributions. The state→series reporting
  mapping is copied verbatim from
  `tests/test_g5a_hlw_replication.py::_series_from_states` (HANDOFF.md
  warning: state slots 4/6 hold `g_{t-1}`/`z_{t-1}`; a spec-prose
  re-derivation silently shifts everything one quarter).
- `src/macrotoolkit/plots.py` (new) — plotting grammar for §3.1–3.4:
  trend-cycle panels (y/y*, gap, r/r*, g, z) with 68/90% bands, the
  exp(h/2) volatility-path exhibit, the IRF 5×5 grid, fan charts, HD
  stacked bars.
- `src/macrotoolkit/report.py` (new) — assembles one self-contained
  `report.html` per run (header, diagnostics verdict, §3.1–3.4 figures,
  parameter table) per spec §3.5.
- `src/macrotoolkit/cli.py` — wire `mtk report <hash>` to `report.py`
  (currently a stub `NotImplementedError`).
- `specs/schema/lw_sv.py` (or a new `outputs` fragment alongside it) — a
  typed `outputs` schema for `lw_sv` (currently `RunSpec.outputs` is a
  free-form dict, per `specs/schema/base.py`): `horizon`, `irf_horizon`,
  `irf_vol_reference: end_of_sample | sample_mean`, `smoother_draws: all |
  {thin: k}`, `forecast_r_rule: neutral | last_value | user_path` (v1
  implements `neutral` and `last_value`; `user_path` is schema-only,
  rejected until a later stage). Defaults taken from spec §2.3's example
  YAML (`horizon: 12`, `irf_horizon: 20`, `irf_vol_reference:
  end_of_sample`, `smoother_draws: all`).
- `tests/test_smoother_sim.py` (new) — simulation-smoother validation:
  per-draw smoothed means agree with the existing RTS smoother, plus the
  moment-matching mirror check described under G6/open-question 1 below.
- `tests/test_g6_hd_identity.py` (new) — the G6 gate.
- `tests/test_results_lw.py` (new) — `_series_from_states`-equivalence
  test (byte-for-byte same mapping as G5a's), IRF/fan/HD shape and
  convention tests (units, horizons, band ordering).
- `tests/test_report.py` (new) — report renders end-to-end from a real
  (small/synthetic) run and contains every §3.1–3.4 figure; a byte-stable
  fixture is NOT pinned here (report content depends on live run draws,
  unlike the `stan/` render pin).
- `examples/us_lw_sv/README.md` — record the S4 acceptance run and attach
  or link the rendered report.
- `HANDOFF.md`, `DECISIONS.md` — updated at stage end per the open
  questions' resolutions and any implementation findings (e.g., a measured
  smoother-draws timing number).

**Not touched:** anything under `stan/` (per the task brief: stop and ask
before any Stan change — S4 shouldn't need one), `specs/schema/base.py`'s
`RunSpec` structure beyond typing the `outputs` dict for `lw_sv`.

## Gates and acceptance (spec §7 S4 row)

1. **Simulation-smoother validation** (spec §2.4, a precondition on
   everything downstream): per-draw smoothed means agree with the existing
   RTS smoother; a G1-style mirror check (see open question 1).
2. **G6**: HD reconstruction identity, exact to 1e-6 per period per draw
   (see open question 3 for how "per draw" is operationalized for a gate
   that must run in finite time).
3. **Report acceptance**: `mtk report <hash>` on the regenerated
   `spec_sv.yaml` run produces a `report.html` containing every §3.1–3.4
   figure (trend-cycle incl. volatility paths, IRF matrix, fan charts, HD
   bars), the diagnostics verdict, and the parameter table — rendered from
   that real run's `draws.nc`, not a stub/placeholder figure.
4. **Regression**: `tests/test_render.py::test_no_sv_render_is_byte_stable`
   stays green throughout (no `stan/` edits are expected to be needed).

## Open questions for S4 implementation (resolved, user confirmation 2026-08-31)

### 1. Simulation smoother algorithm: two-pass Durbin-Koopman (literal §2.4 reading)

**Decision: implement the literal two-pass DK simulation smoother, not
FFBS.** FFBS was proposed with rationale (reuses `_rts_smooth`'s `J_t`
directly, half the per-draw cost) but the user chose DK — implementing
spec §2.4's named algorithm as written rather than substituting the
allowed alternative.

**Algorithm** (Durbin & Koopman 2002, applied per posterior draw at that
draw's own (F, Q, A, Z, R_t) and real data `yobs`):

1. Simulate a "plus" draw from the *unconditional* model at the same
   system matrices: `xi+_0 ~ N(xi00, P00)`, `xi+_t = F xi+_{t-1} + w+_t`
   (`w+_t ~ N(0, Q)`), `y+_t = A'x_t + Z xi+_t + e+_t` (`e+_t ~ N(0,
   R_t)`) — reuses `x_t`/`A`/`Z`/`R_t` from the real data/draw (only the
   noise and resulting state/obs path are simulated).
2. Run the existing filter + RTS smoother (`kalman_smoother`) on the
   **real** `yobs` to get `xi_smooth` (already implemented, S2/S3).
3. Run the same filter + RTS smoother on the **simulated** `y+` to get
   `xi+_smooth`.
4. The simulation-smoother draw is `xi_draw = xi_smooth - xi+_smooth +
   xi+` (the DK identity: the smoothing error `xi_smooth - xi+_smooth` has
   exactly the right conditional distribution to add back onto the known
   plus-path `xi+`, since both smooths share the same linear smoothing
   operator and Gaussian innovations cancel it exactly).

This costs two filter+RTS passes per draw (one already needed for the
point-smoothed series; add one more for the plus path) plus one forward
simulation — straightforward given `_kf_core`/`_rts_smooth` already exist
and are reusable as-is for both the real and plus passes.

Structural shocks for HD (spec §2.4 point 2) still fall out algebraically
from the drawn state path with no extra randomness: consecutive states
give the process noise (`eps_g,t = xi_draw[t][3] - xi_draw[t-1][3]`,
`eps_z,t` from slot 5, `eps_ystar,t` from slot 0 minus `eps_g,t/4`,
matching `Q`'s exact construction in `build_lw_matrices`); the measurement
error at `t` is `yobs[t] - A'x[t] - Z@xi_draw[t]`. No separate
"shock smoother" is needed once the state simulation smoother exists.

**Validation plan** (satisfies "validate against RTS" + "G1-style
mirror", no Stan-side mirror since §2.4 is Python-only): (a) a Monte Carlo
test draws many DK samples at a fixed parameter point and asserts their
empirical mean converges to `kalman_smoother`'s `xi_smooth` and empirical
per-period variance converges to `diag(P_smooth)`, within Monte Carlo
error at a documented draw count (proposed: 5,000 draws, tolerance from
the analytic MC standard error at that count plus a margin); (b) the
"G1-style mirror" is a structural check specific to the DK identity: with
the plus-path noise *fixed at zero* (`w+_t = 0`, `e+_t = 0` for all `t`,
so `xi+ = xi00` propagated deterministically through `F` and `y+` its
noiseless observation), step 4 must reduce to `xi_draw == xi_smooth`
exactly (both smooths of a zero-noise "plus" system coincide with the
deterministic path, cancelling in the identity) — a deterministic,
code-level check that the DK combination step is wired correctly,
independent of RNG.

### 2. Smoother-draw thinning for the output modules (confirmed: benchmark first)

Spec §2.3's draft schema exposes `outputs.smoother_draws: all | thin: k`
with `all` as the example default. Every draw's smoother pass is a fresh
Python/numba per-draw DK pass (two filter+RTS-smooth passes plus a
forward simulation, per open question 1's resolution; no cross-draw
sharing possible) — for a typical run (4 chains × 1500 = 6,000 draws,
T≈230), this is 6,000 independent passes each doing ~2x the per-draw KF
work S2/S3 already numba-compiled (`@njit(cache=True)`), but it has not
been benchmarked yet at this stage's scale — S2/S3 only ever ran the
filter once per draw during MCMC sampling *inside Stan*, never twice per
draw, 6,000 times, in a tight Python loop after the fact. The DK choice
over FFBS (open question 1) makes this benchmark more likely to matter,
not less.

**Confirmed (user, 2026-08-31): benchmark first.** Default
`smoother_draws: all` (matches the spec's own example default), implement
`{thin: k}` as specified, and benchmark actual wall-clock cost on the
regenerated `spec_sv.yaml` run (6,000 draws) as the first thing built in
`results_lw.py` — before writing any plotting code against it. If `all`
takes materially longer than the report is worth waiting for (proposed
threshold: **more than ~2 minutes** for the smoother pass alone on that
run), record the measured number in `DECISIONS.md` and switch the example
specs' default report generation to a thinned `smoother_draws` (proposed
fallback: `thin: 10`, i.e. 600 effective draws — enough for stable
90%/68% band estimates without materially widening them versus the full
6,000). The schema default stays `all` either way (an explicit choice
belongs in the spec, not a silent implementation cap); only the example
specs' `outputs:` block would set `thin` if benchmarking forces it.

### 3. IRF 5×5 grid: shocks (fixed by spec) and response selection (confirmed)

**Shocks (rows): fixed by spec §1.4** — the five structural shocks are
already an exhaustive, named list: ε_IS (demand), ε_PC (supply/cost-push),
ε_y* (potential level), ε_g (trend growth), ε_z (other r*/headwinds). No
selection needed.

**Responses (columns): proposing gap, π, r\*, y (level), g** — covering
every state/observable a reader would want after a shock:

- `gap` and `π` — the two measurement-equation observables (besides `y`
  itself), and the natural home for reading policy-relevant persistence.
- `r*` — the trend real-rate aggregate (`g + z`, c=1), directly shows how
  ε_g/ε_z propagate into the object the whole model exists to estimate.
- `y` (level, cumulative impulse) — distinguishes shocks that move output
  permanently (ε_y*, ε_g via the trend) from those that only move the gap
  transiently (ε_IS, ε_PC), which `gap` alone can't show.
- `g` — trend growth's own path, the natural "own-response" column for
  ε_g and otherwise flat for every other shock (a useful visual contrast
  in the grid).

This gives every shock at least one column where its own structural role
is visible (ε_IS → gap, ε_PC → π, ε_g → g, ε_z → r\*, ε_y* → y), which is
the "target the economically meaningful 5×5" the spec asks for. `z` alone
(headwinds) was considered and dropped in favor of `r*` (its sum with
`g`), since `r*` is the object the report and README already discuss and
`z`'s own path is visually near-identical to `r* − g`.

**Confirmed (user, 2026-08-31): gap, π, r\*, y, g** — as proposed above.

## Ground rules

- Run `numerics-reviewer` after any smoother/state-space/decomposition
  change, before committing (per the task brief).
- `sig_figs=18` for any Stan-vs-Python numeric comparison (none expected
  in S4 proper, since §2.4 is Python-only, but the existing G1 harness
  stays green as a regression check).
- No `stan/` edits without stopping to ask first (per the task brief);
  `tests/test_render.py::test_no_sv_render_is_byte_stable` must stay
  green throughout — S4 shouldn't touch anything that could break it.
- Record decisions in `DECISIONS.md` as they're made (algorithm choice,
  the smoother-draws timing benchmark result, IRF grid confirmation);
  update `HANDOFF.md` at stage end.
