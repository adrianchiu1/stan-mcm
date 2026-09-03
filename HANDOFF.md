# Handoff

Status as of 2026-09-03 (end of S5 session). **S5 is complete and green on
branch `claude/s5-macrotoolkit-sh26bc`** (not merged; the user decides
when/how to PR). Every gate in spec §5 — as amended by the 2026-09-02
pre-S5 review (`plans/S5-decisions.md`, still the binding scope record) —
is green: G1–G4, G5a, G6 pass; G5b is delivered as the informational
exhibit it was demoted to. The lw_sv dry run of the general framework is
finished; **the next stage is family #2: UCSV.**

## What S5 delivered (all green)

**The S4.5 refactor block** (S5-decisions items 1–4, one commit each, each
behavior-preserving with the preservation demonstrated):

- **Named state/coefficient metadata** (item 2): `macrotoolkit/families/`
  — `StateSpaceMeta` declares state-slot labels with explicit time offsets
  (`("g", -1)` IS the slot-3-lag convention), per-shock state loadings
  (`Q == B diag(σ²) Bᵀ` pinned against `build_lw_matrices` by
  `tests/test_state_metadata.py`), and measurement-shock order;
  `structural_coefficients(Z, A)` is the one place matrix positions are
  read. No slot-peeking survives downstream.
- **Feedback map + ONE generic engine** (item 1): families declare, as
  data, which x columns are which lags of which observables
  (`StateSpaceMeta.feedback_map`, pinned column-for-column against
  `build_lw_regressors`); `macrotoolkit/engine.py` runs the measurement
  equation as written for HD bars, IRFs, and stochastic fan simulation,
  replacing `results_lw.py`'s three hand-derived gap-space recursions
  (where both S4 bugs lived). Old-vs-new outputs agree to ~1.8e-14; both
  S4 fan-chart fixes are now structural (rule resolution timing; signs
  from A/Z, nothing hand-applied).
- **Run identity split** (item 3): the hash covers the spec MINUS
  `outputs:` (`to_estimation_yaml`); run dirs carry `spec.yaml`
  (estimation, immutable) + `outputs.yaml` (report options, refreshable);
  `load_run_spec` reassembles and still reads pre-split dirs. Report
  options and report-schema changes can never orphan an MCMC run again.
  This was the one final hash migration.
- **FamilyEntry contract complete** (item 4): every capability on
  `specs/schema/FAMILY_REGISTRY` as lazily-resolved `"module:attr"` paths
  (build_stan_data, build_render_context, state_meta, results_loader,
  report_writer, prior_sd_table, headline_series); no if/elif family
  dispatch anywhere (test-pinned); `mtk report` dispatches via the
  registry. Adding a family = template + schema fragment + families
  module + one registry entry, mechanically.

**S5 features:**

- **Prior-predictive check** (item 7, spec §4): every report carries a
  figure of gap/inflation paths simulated from the run's own RESOLVED
  priors through the same matrices + generic engine
  (`compute_prior_predictive_draws`; `families.lw_sv.sample_prior_params`
  mirrors the template's distributions and truncations). New outputs
  field `prior_predictive_draws` (default 200).
- **`mtk sweep`** (item 9): base spec + prior-override grid → ordinary
  immutable runs (idempotent cell-by-cell) + one comparison report with
  posterior tables, prior→posterior contraction (per cell against that
  cell's own resolved prior), and headline-series overlays. The mandated
  σ_g/σ_z sweep ran (5 cells, 0 divergences anywhere): posteriors scale
  near-proportionally with the prior scales, contraction only ~0.04–0.14
  — the pile-up priors measurably do the identification work — and the
  G5b attribution cross-checks monotonically (filtered 2019Q2 r* gap
  +1.54/+1.44/+1.16 under tightened/default/doubled σ_z). Full record:
  DECISIONS.md 2026-09-03; report under `sweeps/sigma_g_z/` (gitignored,
  regenerable).
- **Generic SBC engine + G4** (items 11/6/10): `tests/sbc_harness.py`
  takes a design (family, options, prior config, prior sampler,
  simulator, data builder, ranked quantities, constants) and returns rank
  statistics — G3's engine generalized, with G3's recorded gate retained
  verbatim and a fast byte-equivalence pin. The a1/a2 stationarity
  override is the documented family-level SBC prior config
  (`SBC_STATIONARITY_PRIOR_CONFIG`, item 6). **G4 PASSED at exactly its
  pre-registered design** (100 reps × T=80; χ² p ∈ [0.067, 0.978] on all
  12 ranked quantities incl. σ_h and h0; 8/150,000 divergences; see
  DECISIONS.md 2026-09-03). The engine has crash-resume from ranks.csv
  (byte-identical by per-rep seeding) — see the environment warning below
  for why.
- **G5b informational exhibit** (item 8): `docs/exhibits/` — our FILTERED
  series vs the published one-sided HLW series (their workbook: "All
  estimates are one-sided", so this is the only like-for-like). r* corr
  0.941; the +1.44 final-period gap decomposes via r* = g + z into +1.51
  from z and −0.07 from g — the σ_z pile-up prior's channel, exactly as
  attributed, now cross-checked by the sweep. Five documented causes on
  the exhibit; NO pass/fail.
- **Item 16**: both reference SV runs regenerated at the final hash
  (`a00958509083` pre-COVID, `eb73e644be0b` full vintage), reproducing
  their recorded diagnostics to the digit; draw-thinned ×10 TRACKED
  fixtures in `runs-archive/` (4–4.5MB each) with the report acceptance
  test falling back to them — fresh containers run output-layer
  acceptance coverage without sampling.
- **Docs**: root `README.md` (validation ladder, G5a/G5b/COVID exhibit
  figures, sweep results, quickstart, family contract);
  `docs/kf-capability-matrix.md` (item 5 — see "UCSV" below);
  `examples/us_lw_sv/README.md` run-record updates.

## Warnings for whoever builds the UCSV stage

- **This environment's container restarts unpredictably** (~2.5h apart
  observed on 2026-09-02/03; per-iteration speed also varies ~3× across
  boots). Detached AND harness-tracked processes both die. Standing
  mitigations, keep using them: run long compute through resumable units
  (the run store's per-cell idempotency; the SBC engine's ranks.csv
  crash-resume), chain ~25-min `send_later` self-check-ins plus an hourly
  watchdog trigger that clean partial run dirs (`runs/<hash>` without
  `_SUCCESS` — inspect, then remove) and relaunch. Never assume a
  multi-hour monolith will survive.
- **The engine is the only recursion implementation now.** Any change to
  `engine.py`, `families/*`'s metadata, or the smoother still requires a
  fresh numerics-reviewer pass per ENGINEERING.md (three passes ran clean
  this stage; the process previously caught two real S4 bugs).
- **Float-regrouping tolerance doctrine** (DECISIONS.md 2026-09-02, item
  1 entry): the engine's measurement-equation grouping matches the old
  hand-rolled recursions to ~1e-14, not bit-for-bit; structural exact
  zeros that follow from genuine sparsity remain exact and are pinned
  exactly. Don't "fix" a ±1e-15 by re-deriving a gap-space recursion.
- **`c` is still hard-assumed 1.0** in the r* = g + z reporting and the
  neutral fan rule; `require_c_is_one` guards every summer (now including
  the IRF). Promoting `estimate_c` needs the c-weighted generalization,
  not guard removal.
- **G3's gate file is frozen history**: its recorded 2026-08-31 pass
  corresponds to that exact code; the engine's byte-equivalence pin
  (`tests/test_g4_sbc.py`) is the bridge. Re-running G3 should go through
  a G3 SbcDesign, not by editing the legacy file.
- No `stan/` file was touched anywhere in S5 (as the brief anticipated);
  `test_no_sv_render_is_byte_stable` is untouched.

## Environment (fresh container recipe)

- `uv tool install pytest --with cmdstanpy --with numba --with arviz
  --with pydantic --with jinja2 --with matplotlib --with pyyaml --with
  pandas --with click --with h5netcdf --with openpyxl --with-editable .`
  (add `~/.local/share/uv/tools/pytest/bin` to `PATH`; `uv.lock`
  side-product stays gitignored).
- CmdStan **pinned 2.36.0** at `~/.cmdstan` (never call `install_cmdstan`
  without `version=` — `api.github.com` is blocked; PREFLIGHT.md §2).
- Fast suite `pytest -m "not slow"`: **284 passed, 0 skipped** at stage
  end (the tracked `runs-archive/` fixtures mean the run-dependent
  acceptance tests never skip on a fresh checkout). Slow gates: G2 ~38 min, G3 ~3.5–7h, G4 ~3–5h
  (`pytest -m slow tests/test_g4_sbc.py` resumes from
  `tests/artifacts/g4_sbc/ranks.csv` if present; `scripts/run_g4.py` is
  the resumable driver).
- Reference runs: regenerate via `mtk run examples/us_lw_sv/spec_sv.yaml`
  / `spec_sv_full_vintage.yaml` (~60–95 min each on a good boot); or use
  `runs-archive/` for development. Sweep: `mtk sweep
  examples/us_lw_sv/sweep_sigma_g_z.yaml` (idempotent against an existing
  store).

## S1–S4 record (condensed, still green)

G1 (KF mirror ~5.5e-12 over three filter paths), G2 (20-dataset
recovery), G3 (200-rep no-SV SBC, χ² p ∈ [0.073, 0.735]), G5a (~1e-12 vs
the self-derived HLW 2017 oracle), G6 (HD identity 1e-6), the DK
simulation smoother, four output modules, and the self-contained HTML
report — details in `STRESS-TESTS.md`, `examples/us_lw_sv/README.md`, and
DECISIONS.md's dated entries.

## What's next: UCSV (family #2)

Scoping lives in `docs/kf-capability-matrix.md` (the item-5 deliverable).
Headlines:

1. **The one real KF change is time-varying `Q_t`** (trend-shock SV — the
   deferred spec §0.4 flag): generalize `kalman_loglik_tv.stan` + the
   Python mirror exactly the way S3 generalized `R_t` (array-of-matrices
   core + constant overload), gated by a G1 mirror extension and a
   constant-Q regression pin. Expect the render/hash consequences of the
   S3 precedent.
2. **Everything else is declarations**: template + `specs/schema/ucsv.py`
   + `macrotoolkit/families/ucsv.py` (state labels like `(("tau", 0),)`,
   EMPTY feedback map — the degenerate case the engine already supports
   and tests — prior sampler, builders) + one `FAMILY_REGISTRY` entry +
   gates instantiated from the generic harnesses.
3. **Harness audit conclusions** (item 11's audit, recorded 2026-09-03):
   the SBC engine is already generic (use an `SbcDesign`). G1's
   mirror-gate shape (N prior points, Stan-vs-Python loglik at 1e-8) and
   G2's coverage/bias gate arithmetic are the next lift-outs — small,
   done when UCSV needs them, not speculatively; the necessarily
   family-authored pieces are each family's Stan loglik harness template
   and its structural-equation simulator. Prior samplers and data/render
   builders are already registry capabilities.
4. Deliberately NOT built (per item 5's doctrine): missing observations,
   exact diffuse init, time-varying Z/A, rank-deficient-R filtering,
   large-n scaling — each waits for the family that needs it.
