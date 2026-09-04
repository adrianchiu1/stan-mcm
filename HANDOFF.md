# Handoff

Status as of 2026-09-04 (end of S6 session). **S6 is complete and green on
branch `claude/s6-macrotoolkit-jpguf5`** (pushed; PR #7,
https://github.com/adrianchiu1/stan-mcm/pull/7, opened by the user from
the Claude Code UI; the stage-per-PR cadence continues). The binding scope record is `plans/S6-plan.md`
(including the two places the brief and the repo's own docs conflicted
and how they were resolved). Three work packages shipped in order, each
at a clean boundary: **WP1** the notebook-first Python API, **WP2**
family #2 (UCSV) with the one real KF extension (time-varying `Q_t`) and
its full validation ladder, **WP3** automatic per-model QC (fit-time
mirror check + `mtk validate`). **The natural S7 candidates are the
equation-authoring DSL and the promotion/production layer** (see "What's
next").

## What S6 delivered (all green)

**WP1 — notebook-first Python API** (`macrotoolkit.api`; canonical
import `from macrotoolkit import api as mtk`):

- The Pydantic spec models ARE the spec API (`mtk.spec(...)` returns a
  validated `RunSpec`; a Python-built spec hashes identically to its YAML
  equivalent — pinned against `examples/us_lw_sv/spec_sv.yaml`).
- `mtk.fit(spec, data)` with `data` a CSV path or a pandas DataFrame
  (staged to a canonical CSV: only the spec's columns, ISO dates, shortest
  round-trip floats, fixed name `dataframe.csv`, so identity is a function
  of content). `run.py`'s `run()` is now a wrapper over the spec-taking
  core `run_spec(spec, base_dir=...)`.
- `Run` handle: `hash`, `spec`, `diagnostics`, `verdict`, `mirror_check`,
  lazy `idata`, `results()`, `outputs()` (the family's declared output
  modules as live matplotlib Figures), `param_table()` (DataFrame),
  `report()`; `mtk.load_run(hash_or_dir)`; `mtk.sweep(...)` returning a
  programmatic `SweepComparison` (`.table()`, `.headline`) next to the
  HTML; `mtk.validate(family, tier)`.
- The report is family-generic over `FamilyEntry.output_modules`
  (`OutputModule(name, heading, figure_title, compute, plot, caption)`;
  lw_sv's declaration in `outputs_lw.py` reproduces the S4/S5 report
  exactly). The CLI (`cli.py`) is a thin shell: every command delegates
  to the API (test-pinned; no pipeline logic in it).
- `examples/notebook_api/lw_sv_from_archive.ipynb` (committed executed:
  the whole lw_sv output layer on `runs-archive/a00958509083`, no
  sampling) and `ucsv_us_inflation.ipynb` (committed executed: UCSV end
  to end, samples); `tests/test_notebooks.py` re-executes them.

**WP2 — family #2, UCSV** (`pi_t = tau_t + eps_t`, `tau_t = tau_{t-1} +
eta_t`, SV on both shocks; `docs/kf-capability-matrix.md` was the binding
scope):

- **The ONE real KF extension, `Q_t`**, by the S3 `R_t` playbook: the
  Stan core takes `array[] matrix Q` and `array[] matrix R` with three
  delegating constant overloads; the Python mirror takes a constant or a
  `(T, n, n)` path everywhere (`_as_Q_path`, `_kf_core` with `Q[t]`,
  smoother, DK plus path). G1 extended to FIVE paths (constant, tv-R,
  production R-SV, tv-Q, production Q-SV): max |Stan − Python| 5.5e-12 /
  5.5e-12 / 1.8e-12 / 7.3e-12 / 7.3e-12. **Constant-Q regression pin**:
  the Stan constant-Q values at G1's 50 points reproduce a fixture
  captured from the untouched S5 program (`tests/fixtures/g1/pre_qt_stan_
  loglik.csv`) with difference exactly 0.0; a `(T,n,n)` path of identical
  Q is bit-identical to the constant Python call. **Run-hash consequence
  taken deliberately** (DECISIONS.md 2026-09-04, plan conflict item 1):
  every lw_sv spec re-identifies once (the rendered source inlines the
  filter; the S3 precedent); the no-SV render pin was regenerated once
  with the recorded decision; the archived reference runs stay valid
  under their old names and `examples/us_lw_sv/README.md` records the
  lineage (`spec_sv.yaml` → `8ba1420a4145`, full vintage →
  `ec87f45d0a43`). The reference runs were NOT regenerated.
- **Everything else is declarations**: `stan/templates/ucsv.stan.j2`
  (+ the ucsv-only `sv_scalar_variance_path.stan`), `specs/schema/ucsv.py`,
  `families/ucsv.py` (`UCSV_STATE_META` = `(("tau", 0),)`, EMPTY feedback
  map, `ln(Var(Δpi)/2)` mu_h0 anchors, prior sampler, mirror declaration),
  one registry entry, `results_ucsv.py` / `plots_ucsv.py` /
  `outputs_ucsv.py` over the new family-generic `results_core.py`.
  **Generic-layer residue fixed** (the full list in DECISIONS.md
  2026-09-04 WP2b): the smoother's shock recovery (now
  `recover_shocks` via `StateSpaceMeta.recovery_order`, bit-identical for
  lw_sv), the engine's state noise (`StateNoise` protocol; constant case
  = the exact pre-S6 draw), the report/param-table/figure plumbing
  (WP1), and the loader/draw-loop/HD/IRF pieces that only existed inside
  `results_lw.py`.
- **Validation ladder for ucsv** (no external oracle — stated in README):
  G1 as above; the fast tier's production-render mirror (1.8e-12) and HD
  identity (2e-16); **G2 recovery** PASSED (pooled 90%-CI coverage 0.85, per quantity 0.85/0.85/0.75/0.95, no σ_h bias, 4 divergences over 20 fits); **SBC at the
  pre-registered design** (100 reps × T=100, 4 ranked quantities, fixed
  anchors, seeds 20260920+i; DECISIONS.md 2026-09-04) PASSED at exactly the registered design: χ² p = 0.596 / 0.911 / 0.760 / 0.052 on sigma_h_eps / sigma_h_eta / h0_eps / h0_eta (floor 0.001), 4 divergences over 150,000 post-warmup draws, ~1.7 h of compute (63 s/rep).
  Slow gates: `pytest -m "slow and ucsv"` (~1.5 h total; the SBC resumes
  from `tests/artifacts/ucsv_sbc/ranks.csv`; `scripts/run_ucsv_sbc.py`
  is the resumable driver).
- **Worked example** `examples/us_ucsv/` (US core PCE from the in-repo
  HLW workbook fixture, `make_data.py`), through the notebook API with
  `mtk run`/`mtk report` as the CLI cross-check (same hash
  `f2b48ebc98a4`); run record in its README.

**WP3 — automatic per-model QC**:

- **Auto-G1 at fit time** (`macrotoolkit/qc.py`): every `fit()` / `mtk
  run` evaluates the EXACT rendered program's `kf_loglik` (now a
  transformed parameter in every KF family template, `target +=
  kf_loglik`) at `qc.mirror_points` (default 5) prior draws via one
  fixed_param CmdStan call, against the Python mirror on the same Stan
  data, BEFORE sampling; records `diagnostics.json["mirror_check"]` and
  the report header row; raises `MirrorCheckError` past 1e-8 and removes
  the run dir. Spec-level toggle: the new top-level `qc:` block, outside
  the identity hash like `outputs:` (`qc.yaml` in the run dir).
- **`mtk validate <family> --tier fast|recovery|sbc|all`** /
  `api.validate`: `FamilyEntry.validation_suite` → `families/<family>_
  validation.py` declaring `Gate`s per tier; generic builders
  (`mirror_gate`, `hd_identity_gate`, `recovery_gate`, `sbc_gate`) in
  `macrotoolkit/validation/suite.py`; the SBC engine moved into the
  package (`validation/sbc.py`; `tests/sbc_harness.py` is a shim) with
  lw_sv's G4 design moved verbatim (`tests/g4_harness.py` is a shim) and
  the G2 arithmetic lifted (`validation/recovery.py`). One report per
  invocation: `validation/<family>/report.html` + `summary.json`. Both
  families wired.
- `scripts/gate-check.sh fast | family <fam> | validate <fam> [tier] |
  slow <fam>`; markers `lw_sv`, `ucsv`, `validation`.

## Warnings for whoever builds S7

- **Never set `TQDM_DISABLE=1` around a fit.** It makes cmdstanpy's
  progress handling raise inside `model.sample` (chains report retcode
  -1 and the run fails with a misleading "Error during sampling"). Two
  full-suite runs were lost to it this stage before the cause was
  isolated (DECISIONS.md 2026-09-04 WP1).
- **A notebook executed from a tmp directory must not walk parent
  directories looking for the repo** (`Path.parent` of `/` is `/`: an
  infinite loop that pinned a kernel at 100% CPU). Use
  `macrotoolkit.run.REPO_ROOT`.
- **Every lw_sv spec re-identified in S6** (Q_t + `kf_loglik`); see
  `examples/us_lw_sv/README.md`'s lineage table. Do not "fix" a hash
  mismatch against the archived names; regenerate when publication
  numbers are needed.
- **`results_lw.py` is still the validated lw_sv instantiation, not a
  declaration over `results_core.py`.** It delegates where identical and
  otherwise stays; migrating it onto the core is S7 backlog and needs a
  behavior-preservation pass (bit-level pins exist for the smoother
  recovery and the engine's state noise to start from).
- **The identity gates evaluate at STATIONARY prior points** (lw_sv's
  registered `hd_identity_gate` filters (a1, a2)); an explosive AR(2)
  draw measures float64 cancellation over 230 quarters, not the
  identity. Found on the gate's first run; documented in the gate.
- **The mirror check's fixed_param evaluation needs every parameter of
  the rendered program in the inits** (a family's `mirror_points` must
  stay in sync with its template's parameter block); a missing name
  makes CmdStan initialize it randomly and the check fail loudly (the
  right failure, but confusing). `adapt_engaged=False` is required with
  `iter_warmup=0`.
- Container restarts (S5's warning) still apply; every slow gate this
  stage is resumable (SBC ranks.csv; the recovery gate re-runs — it is
  ~20 minutes).
- Any change to `engine.py`, `smoother.py`, `results_core.py`, the KF
  Stan files or a family's metadata still requires a fresh
  numerics-reviewer pass (one ran clean over the S6 change set, findings
  and disposition in DECISIONS.md).

## Environment (fresh container recipe)

- `uv tool install pytest --with cmdstanpy --with numba --with arviz
  --with pydantic --with jinja2 --with matplotlib --with pyyaml --with
  pandas --with click --with h5netcdf --with openpyxl --with scipy
  --with nbformat --with nbclient --with ipykernel --with-editable .`
  (add `~/.local/share/uv/tools/pytest/bin` to `PATH`; the notebook
  packages are needed by `tests/test_notebooks.py`).
- CmdStan **pinned 2.36.0** at `~/.cmdstan` (`python -m
  cmdstanpy.install_cmdstan --dir ~/.cmdstan --version 2.36.0`; never
  without `--version` — `api.github.com` is blocked; ~25 min build).
- Fast suite `pytest -m "not slow"`: **325 passed, 0 skipped** at
  stage end (~5–7 min). Slow: `-m "slow and ucsv"` (G2 ~20 min + SBC
  ~1–1.5 h, resumable), `-m "slow and lw_sv"` (G2 ~38 min, G3 ~3.5–7 h,
  G4 ~3–5 h resumable).
- Reference runs: `runs-archive/` (S5 fixtures) for the output layer;
  `mtk run examples/us_ucsv/spec.yaml` is a few minutes.

## S1–S5 record (condensed, still green)

lw_sv: G1 (now 5 paths), G2, G3 (200-rep no-SV SBC), G4 (100-rep full-SV
SBC at the pre-registered design), G5a (~1e-12 vs the HLW oracle), G6, the
DK smoother, four output modules, the HTML report, `mtk sweep` with the
mandated σ_g/σ_z sweep, the G5b informational exhibit — details in
README.md's ladder table, `STRESS-TESTS.md`, `examples/us_lw_sv/README.md`
and DECISIONS.md's dated entries.

## What's next: S7 candidates

1. **Equation-authoring DSL**: economists write the measurement/state
   equations (named series, lags, shocks) and the framework derives the
   `StateSpaceMeta`, matrices, template and mirror declaration — UCSV
   showed the declaration surface is small and regular (state labels,
   shock loadings, feedback map, prior table, mu_h0 anchors), which is
   exactly what a DSL should emit.
2. **Promotion / production layer**: `mtk validate` gives a per-family
   verdict; the missing piece is a promotion record (which validated
   program+design a production run may use, with the mirror check and
   gate verdicts attached), scheduled re-validation, and the
   `runs-archive` → published-run workflow.
3. Backlog carried forward: migrate `results_lw.py` onto
   `results_core.py`; PACF stationarity parameterization for AR blocks
   (S5 item 6); `estimate_c` (needs the c-weighted r* generalization);
   missing observations / exact diffuse init when DFM arrives
   (capability matrix doctrine: build with the family that needs it).
