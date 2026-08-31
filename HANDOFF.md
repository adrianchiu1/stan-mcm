# Handoff

Status as of 2026-08-31 (end of S3 session). **S3 is complete and green on
branch `claude/s3-stochastic-volatility-lw-z0vvmj`** (not yet merged to
main; the user decides when/how to PR — the G3 commits lead the branch so
they can be split into their own PR if wanted). Spec §7's S3 row — "Add
non-centered SV block; retune (adapt_delta, priors) | G4-precursor: clean
sampling on US data; G3 pass" — is met in full, with **no retuning
needed** (default adapt_delta 0.95 sampled cleanly). S2's record is one
section down; per-run detail lives in `examples/us_lw_sv/README.md`,
decisions and traps in `DECISIONS.md` (2026-08-31 entries).

## What S3 delivered (all green)

- **G3 (SBC, no-SV)** — `tests/test_g3_sbc.py`, `slow`, 200 replications,
  3h37m: per-parameter χ² uniformity p ∈ [0.073, 0.735] over all 10
  parameters; 5/600,000 divergent draws. Ran with the a1/a2 prior
  overridden to N(0.8, 0.1²)/N(−0.25, 0.05²) through the PRODUCTION
  override path — the production defaults put ~1/3 prior mass on
  non-stationary gap dynamics whose simulated data destroys the KF in
  float64 (measured; see the DECISIONS entry). SBC of the production
  a1/a2 values themselves is impossible with a plain KF; recorded limit.
- **KF generalized to time-varying R_t** — `kalman_loglik` takes an array
  of T measurement covariances (constant-R thin overload delegates via
  rep_array; Python mirror `_as_R_path`). G1 now covers three paths
  (constant, data-R_t, and the production SV composition), all at
  ~5.5e-12; G5a untouched at ~1e-12. ONE filter implementation — the
  price was S2 run-hash stability (user decision; old run dirs remain
  valid records).
- **SV block** — `stan/templates/lw_sv.stan.j2` grew the non-centered SV
  variant behind a single `{% if sv_shocks %}` conditional (schema
  accepts only `[]` / canonical `["is","pc"]`). h paths live in
  `transformed parameters` (authoritative for S4's outputs — do NOT
  re-derive h from nu). mu_h0 anchors = HLW-exact OLS pass
  (`lw_mu_h0_anchors` in run.py, mirroring rstar.stage3.R lines 22–48).
- **Acceptance (G4-precursor)** — run `70ad47166eaf`
  (`examples/us_lw_sv/spec_sv.yaml`, 1961Q1–2019Q2): PASS, 0 divergences,
  0 treedepth hits, E-BFMI 0.88–1.01, max R-hat 1.004. σ_h posteriors
  well off zero (5th pct ≈ 0.14) — the funnel neck holds no mass, which
  is WHY sampling is clean; a dataset with weak volatility variation may
  still need adapt_delta ≥ 0.98 (untested territory).
- **COVID payoff exhibit** — run `9d10bcf32a40`
  (`spec_sv_full_vintage.yaml`, through 2026Q1, no COVID machinery):
  PASS; the four largest exp(h_IS/2) medians are exactly 2020Q1–Q4 (peak
  3.77), and the STRESS-TESTS.md §3 contamination reverses (σ_y* 0.94 →
  0.33; gap AR restored to pre-COVID values). The stress test's motivating
  failure mode is closed.

## Byte-stability pin (new invariant — read before touching stan/)

`tests/test_render.py::test_no_sv_render_is_byte_stable` pins the no-SV
render byte-for-byte against `tests/fixtures/render/lw_sv_no_sv.stan`.
Any edit to the template or ANY included functions file that reaches the
no-SV render fails it. That's the point: changing that fixture is a
deliberate act requiring a DECISIONS.md entry (it moves every future
no-SV run hash). SV-only helpers belong in includes the no-SV render
doesn't pull (`sv_rw_noncentered.stan` pattern).

## Warnings for whoever builds S4

- **State timing convention** (unchanged from S2): state slots 4/6
  (1-indexed) hold `g_{t-1}`/`z_{t-1}`; copy
  `tests/test_g5a_hlw_replication.py::_series_from_states` for reporting,
  never re-derive from the spec's prose.
- **h timing**: observation t uses `h[t] = h_0 + σ_h·Σ_{s≤t} ν_s`; `h_0`
  is pre-sample (prior N(mu_h0_data, 1), non-centered via `h0_*_raw`).
  Read h from the saved `h_is`/`h_pc` transformed parameters.
- **The DK simulation smoother (S4) must consume the per-draw R_t path**:
  build it as `sv_diag_variance_path(h_is, h_pc)` from that draw's h —
  the Python mirror already accepts `(T, m, m)` R everywhere
  (`kalman_loglik`, `kalman_smoother`). For no-SV draws pass the constant
  R; `_as_R_path` normalizes both.
- **Prior overrides are variant-checked**: `priors: {sigma_is: ...}` on
  an SV spec (or `sigma_h_*`/`mu_h0_*` on a no-SV spec) is a hard error,
  not a silent no-op (numerics-review fix, tested).
- The G2/G3 gates fit the no-SV variant; nothing yet SBC-tests the SV
  variant itself — that's G4 (S5 scope, per spec §7).

## Environment (fresh container recipe)

- `uv tool install pytest --with cmdstanpy --with numba --with arviz
  --with pydantic --with jinja2 --with matplotlib --with pyyaml --with
  pandas --with click --with h5netcdf --with openpyxl --with-editable .`
  (a `uv.lock` side-product is gitignored — never commit it).
- CmdStan **pinned 2.36.0** at `~/.cmdstan` (never call `install_cmdstan`
  without `version=` — `api.github.com` is blocked; PREFLIGHT.md §2).
- Fast suite `pytest -m "not slow"`: **112 passed** (~1 min warm cache;
  first run pays Stan compiles). Slow gates: G2 ~38 min, G3 ~3.5–7 h
  (`pytest -m slow tests/test_g3_sbc.py`).
- The two S3 runs live in the (gitignored) run store `runs/70ad47166eaf`,
  `runs/9d10bcf32a40` — container-local; regenerate via `mtk run
  examples/us_lw_sv/spec_sv.yaml` / `spec_sv_full_vintage.yaml`
  (~40–80 min each). Volatility-path figures regenerate from the run
  store (per-draw `h_is`/`h_pc` are in `draws.nc`).

## S2 record (unchanged, still green)

G1 (3.6e-12 → now 5.5e-12 across three KF paths), G2 (20-dataset
recovery), G5a (~1e-12 vs the self-derived HLW oracle,
`tests/fixtures/hlw/derived/us_2017_reproduction/`), US/EA benchmark runs
tracking HLW at ρ ≈ 0.94–0.99 (`STRESS-TESTS.md`). The pile-up priors
deliberately replace HLW's MUE machinery; σ_g/σ_z sensitivity sweep
remains S5 scope.

## What's staged next (S4, per spec §7)

DK simulation smoother (`smoother.py` currently ends at the RTS
smoother); the four output modules (trend-cycle plots incl. the exp(h/2)
exhibit, IRFs, fans, historical decomposition); HTML report; G6
(HD reconstruction identity, 1e-6). `results_lw.py` must inherit the G5a
reporting mapping verbatim.
