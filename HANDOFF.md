# Handoff

Status as of 2026-08-31 (end of S4 session). **S4 is complete and green on
branch `claude/s4-dk-simulation-output-poztdt`** (not yet merged to main;
the user decides when/how to PR). Spec §7's S4 row — "DK smoother; four
output modules; HTML report | G6 pass; report renders all figures from a
real run" — is met in full. S2/S3's records are condensed below; per-run
detail lives in `examples/us_lw_sv/README.md`, decisions and traps in
`DECISIONS.md` (2026-08-31 entries, especially the two numerics bugs the
mandatory reviewer pass caught in the fan-chart module — read those before
touching `results_lw.py`'s Part D).

## What S4 delivered (all green)

- **DK simulation smoother** — `smoother.py` §5
  (`simulate_smoother_draw`): the literal two-pass Durbin-Koopman algorithm
  (user's explicit choice over the cheaper FFBS alternative, `plans/
  S4-plan.md`'s resolved open question 1), per posterior draw: simulate an
  unconditional "plus" path at that draw's own system matrices, smooth both
  the real and plus data, combine via the DK identity. Also recovers the 5
  structural shocks algebraically from consecutive drawn states. Validated
  by Monte Carlo mean/variance convergence to the existing RTS smoother
  plus a deterministic zero-plus-noise identity check (no Stan-side mirror
  exists since §2.4 is Python-only).
- **`results_lw.py`** — four parts, all consuming a completed run's own
  stored artifacts (never re-deriving from the original spec file's
  location):
  - Part A: trend-cycle series (§3.1) — reporting mapping copied VERBATIM
    from `tests/test_g5a_hlw_replication.py::_series_from_states`.
  - Part B: historical decomposition (§3.4) + **gate G6**. Two subtleties
    discovered and fixed while building this (both documented at length in
    the code and in DECISIONS.md — read before modifying): (1) the
    "initial condition" bar must be the RESIDUAL `xi_draw - (sum of the 3
    named trend-shock contributions)`, not a forward-propagated `F@xi00`
    path — `xi00`/`P00` is only a prior, and the RTS smoother genuinely
    revises it using the whole sample; a naive `F@xi00` bar misses that
    revision at exactly the state slots gap's own AR recursion reads. (2)
    the two measurement shocks (eps_IS, eps_PC) need their OWN persistent
    AR-propagated contribution (gap's IS-curve feedback, π's Phillips-curve
    feedback) — spec §0/§1.4 name them as 2 of the 5 structural shocks the
    HD must attribute, so a same-period-only residual is spec-non-compliant
    as well as economically wrong.
  - Part C: IRF matrix (§3.2) — a theoretical impulse response reuses Part
    B's own machinery (zero real data, zero "init" bar, a synthetic
    one-off impulse). 5 shocks × 5 responses (`gap, pi, rstar, y, g` —
    `plans/S4-plan.md`'s confirmed open question 3).
  - Part D: fan charts (§3.3) — genuinely new stochastic forward
    simulation from each draw's terminal smoother state, continuing the SV
    log-variance random walks forward (SV runs) so bands genuinely widen
    with horizon. **The numerics-reviewer pass caught two real bugs here
    before commit** (a rate-gap seeding/ordering bug, and a separate
    IS-curve sign error found in a follow-up pass on the same function —
    full account in DECISIONS.md's 2026-08-31 "two numerics bugs" entry).
    Read that entry before touching `_fan_forecast_step`/`simulate_fan_draw`.
- **`plots.py`** — matplotlib figures for all four modules (`Figure`
  objects, not encoded strings — `report.py` owns embedding).
- **`report.py` + `mtk report <hash>`** — one self-contained `report.html`
  per run (base64-embedded PNGs, zero external references): header,
  diagnostics verdict, §3.1–3.4 figures, parameter table (scalar
  parameters only — vector SV path parameters are in the volatility panel
  instead).
- **G6 passes** (`tests/test_g6_hd_identity.py`) at both no-SV and SV
  synthetic parameter points, per-period, to spec's 1e-6 tolerance.
- **S4 acceptance test** (`tests/test_report.py`) runs `mtk report` end to
  end against the real, regenerated `70ad47166eaf` SV run: 11 embedded
  images (1 trend-cycle + 1 IRF grid + 5 fan charts + 4 HD charts), zero
  external references, recognizable parameter rows.
- **outputs schema** — `LwSvOutputs` (`specs/schema/lw_sv.py`) types
  `RunSpec.outputs` for `lw_sv` (`horizon`, `irf_horizon`,
  `irf_vol_reference`, `smoother_draws`, `forecast_r_rule`) — changes the
  `lw_sv` run-identity hash again (same tradeoff as S3's KF generalization;
  old run dirs remain valid records).

## Warnings for whoever builds S5

- **`smoother_draws: all` is fine, measured**: the full DK smoother pass
  over 6,000 draws takes ~40s on the regenerated `spec_sv.yaml` run — no
  thinning default was needed (`plans/S4-plan.md`'s open question 2,
  resolved). Re-benchmark if a much larger run (more chains/draws, longer
  T) makes report generation noticeably slow.
- **The fan-chart module (Part D) is the highest-risk code in this
  stage** — two real bugs were caught there by review, both subtle
  period-alignment/sign issues arising from the state's slot-3/5
  one-period-lag convention interacting with a genuinely NEW piece of
  machinery (stochastic forward simulation, not just reading already-
  smoothed rows). Any future change to `simulate_fan_draw`/
  `_fan_forecast_step`/`compute_fan_draws` needs a fresh, careful
  numerics-reviewer pass — do not assume "it already passed review" covers
  a modified version.
- **`c` is hard-assumed 1.0** in `results_lw.py`'s `gap_pi_shock_
  decomposition` and the fan-chart module (both have an explicit
  `NotImplementedError` guard, not a silent wrong-answer) — if `S5` or
  later ever promotes `estimate_c` to `true`, both need the `g`/`z`
  weighting generalized (currently `rstar = g + z` unweighted; the true
  form is `c*g + z`), not just the guard removed.
- **G2/G3/G4 status unchanged from S3**: G2/G3 fit the no-SV variant only;
  G4 (full SV SBC) and G5b (Bayesian HLW tracking) are S5 scope per spec
  §7, still not built.
- Everything in S3's own warnings (state-slot timing, h timing, prior
  overrides variant-checked) still applies unchanged.

## Environment (fresh container recipe)

- `uv tool install pytest --with cmdstanpy --with numba --with arviz
  --with pydantic --with jinja2 --with matplotlib --with pyyaml --with
  pandas --with click --with h5netcdf --with openpyxl --with-editable .`
  (a `uv.lock` side-product is gitignored — never commit it). Add
  `~/.local/share/uv/tools/pytest/bin` to `PATH` to get `pytest`/`mtk`.
- CmdStan **pinned 2.36.0** at `~/.cmdstan` (never call `install_cmdstan`
  without `version=` — `api.github.com` is blocked; PREFLIGHT.md §2).
- Fast suite `pytest -m "not slow"`: **217 passed** (~1-4.5 min depending
  on Stan-compile cache state). Slow gates: G2 ~38 min, G3 ~3.5–7 h
  (`pytest -m slow tests/test_g3_sbc.py`).
- The two SV runs live in the (gitignored) run store `runs/70ad47166eaf`,
  `runs/930459224ca0` (S4 regenerated both; the full-vintage run's hash
  changed from S3's `9d10bcf32a40` because of the `outputs` schema
  addition — numerics unchanged, see DECISIONS.md) — container-local;
  regenerate via `mtk run examples/us_lw_sv/spec_sv.yaml` /
  `spec_sv_full_vintage.yaml` (~40–80 min each), then `mtk report <hash>`
  (~3 min each, dominated by the DK smoother's 6,000-draw pass across four
  output modules).

## S2/S3 record (condensed, still green)

G1 (KF vs Stan, ~5.5e-12 across three filter paths — constant, time-
varying, production SV composition), G2 (20-dataset parameter recovery),
G3 (200-rep SBC, no-SV, χ² p ∈ [0.073, 0.735]), G5a (~1e-12 vs the
self-derived HLW oracle), the non-centered SV block (`sv_shocks: [is,
pc]`, `h` is log-variance, `sd = exp(h/2)`), and both SV acceptance runs
(0 divergences, PASS) — see `STRESS-TESTS.md` and `examples/us_lw_sv/
README.md` for full detail, including the COVID payoff exhibit (SV
absorbs the 2020 shock endogenously; structural parameters de-contaminate
vs. the no-SV stress test).

## What's staged next (S5, per spec §7)

"Full SBC (G3, G4), Bayesian HLW tracking (G5b), prior-sweep notebook,
docs | All gates green; README with G5a/G5b figures." Concretely: G4 (SBC
on the full SV variant — G3's machinery generalizes, but expect the same
kind of prior-override care G3 needed for a1/a2's non-stationary tail);
G5b (SV off, HLW-like loose priors, tracking their published data within
±50bp/±0.5pp over 2000–2019); the σ_g/σ_z prior-sensitivity sweep spec
§1.6 mandates (a notebook, `examples/us_lw_sv/`); final docs pass.
