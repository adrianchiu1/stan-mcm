# S6 plan — notebook API, family #2 (UCSV), automatic per-model QC

**Status: plan of record, drafted 2026-09-04 at S6 start, on branch
`claude/s6-macrotoolkit-jpguf5` from `origin/main` (S1–S5 complete, PR #6
merged); EXECUTED IN FULL the same day -- all three work packages shipped
at their boundaries (see HANDOFF.md and DECISIONS.md 2026-09-04 for what
landed and the measured gate results). Three work packages, shipped in
this order, each independently shippable at its boundary.** Binding conventions inherited unchanged:
immutable hash-identified runs, the validation ladder (ENGINEERING.md),
named state metadata (never raw slot indices), the G1 Stan-vs-Python
mirror discipline, the pre-registration rule for SBC designs, a
numerics-reviewer pass over every KF/smoother/matrix/engine change, and
the dated DECISIONS.md log.

Baseline at stage start (fresh container, HANDOFF recipe): CmdStan 2.36.0
built at `~/.cmdstan`; fast suite `pytest -m "not slow"` green (count
recorded in DECISIONS.md at the first commit).

## Where the brief and the repo's docs conflict (repo docs win)

1. **"Do NOT change existing lw_sv run-identity hashes" vs the S3
   precedent.** The run hash covers the rendered Stan source, which
   INLINES `stan/functions/kalman_loglik_tv.stan`; any edit to the KF text
   changes every lw_sv render's hash. DECISIONS.md 2026-08-31 ("S2 run-hash
   stability sacrificed for a single KF implementation", a user decision)
   chose ONE generalized filter over hash stability, and both HANDOFF.md
   ("expect the render/hash consequences of the S3 precedent") and
   `docs/kf-capability-matrix.md` anticipate the same for `Q_t`.
   Resolution: generalize `kalman_loglik_tv.stan` in place (array-of-`Q_t`
   core + constant overloads, exactly the R_t playbook); accept that every
   lw_sv spec re-identifies once more; regenerate the no-SV render pin
   fixture ONCE with a recorded decision (the fixture's own docstring
   allows exactly this); and satisfy the brief's *intent* — constant-Q
   numerics unchanged — with three pins: (a) the Python `_kf_core` loglik
   on the constant-Q path is BIT-IDENTICAL before/after (test), (b) the
   Stan constant-Q loglik at G1's 50 points equals a fixture captured from
   the pre-change program (`tests/fixtures/g1/pre_qt_stan_loglik.csv`,
   captured on the untouched baseline before any `stan/` edit; tolerance
   1e-10), (c) G1/G5a/G6 stay green. Archived reference runs remain valid
   immutable records; the example README records the hash lineage (the
   new identity of each example spec is computed without sampling).
2. **"Auto-G1 at a handful of prior draws"** — implemented literally
   (fixed_param evaluation of the exact rendered program at prior draws,
   BEFORE sampling), which requires the rendered program to EXPOSE its KF
   log-likelihood. Chosen mechanism: `kf_loglik` becomes a transformed
   parameter in every family template (`target += kf_loglik`), so no
   duplicate KF call and every posterior draw also carries the Stan-side
   KF loglik. This is a template edit for lw_sv (hash consequence already
   accepted under item 1 — the two edits land in the same identity move).
3. **The `qc:` toggle must be spec-level but must not enter run
   identity** (a QC setting never changes the estimation). It is a new
   top-level `RunSpec.qc` block excluded from `to_estimation_yaml` the
   way `outputs` is (S5-decisions item 3's split, extended), stored in
   the run dir as `qc.yaml` alongside `outputs.yaml`.

## WP1 — notebook-first Python API (wrap, don't rewrite)

New `src/macrotoolkit/api.py`, re-exported from `macrotoolkit/__init__`:

- **Spec construction**: the Pydantic models ARE the API (`RunSpec`,
  `ModelSpec`, `DataSpec`, `SampleSpec`, `SamplerSpec`, the family options/
  outputs models) plus one convenience `mtk.spec(family, data=..., options=
  ..., priors=..., sampler=..., outputs=...)` returning a validated
  `RunSpec`. No parallel spec: a Python-built `RunSpec` with the same
  content as a YAML file yields the same `to_estimation_yaml()` and the
  same run hash — pinned by a test against `examples/us_lw_sv/spec_sv.yaml`.
- **`fit(spec, data=None, *, base_dir=None, runs_root=None) -> Run`**.
  `spec` may be a `RunSpec` or a YAML path (then `base_dir` = its
  directory, exactly `mtk run`). `data` may be a CSV path (overrides
  `spec.data.file`; hash parity with the YAML holds iff the path string
  and base_dir match the YAML's) or a pandas DataFrame — serialized to a
  canonical CSV (`float_format="%.17g"`, ISO dates, `\n`) under a private
  staging dir with `data.file = "dataframe.csv"`, so the identity is a
  deterministic function of the frame's content (test: same frame → same
  hash; perturbed value → different hash). The DataFrame's columns are
  interpreted through `spec.data.date_column`/`mapping` exactly like a CSV.
- **`Run` handle** (wraps a completed immutable run dir): `hash`,
  `run_dir`, `spec`, `family`, `is_new`, `diagnostics` (dict), `verdict`,
  `mirror_check` (WP3), `idata` (lazy), `results()` (family results object
  via the registry), `outputs()` (see below), `param_table()` →
  DataFrame, `report()` → Path. `load_run(hash_or_dir, runs_root=None)`
  opens an existing run (the notebook uses `runs-archive/`).
- **Inline figures**: a new registry capability `output_modules` — each
  family declares an ordered tuple of `OutputModule(name, title,
  compute(results) -> data object, plot(data, results) -> Figure |
  dict[str, Figure], caption)`. `run.outputs()` returns an `Outputs`
  object: `.names`, `.compute(name)`, `.figure(name)` (matplotlib Figures,
  never files). `report.py` becomes generic over the same declaration
  (header, diagnostics, modules in order, parameter table), so lw_sv's
  report keeps the same 12 embedded figures and section titles.
- **`sweep(...)` from Python**: `run_sweep` is split into a
  `SweepSpec`-taking core plus the file wrapper; the comparison data
  (per-cell posterior stats, prior sds, contraction, headline series) is
  computed by one function returning a `SweepComparison` dataclass with
  `.table()` (long-form DataFrame) that both the HTML report and the API
  consume.
- **CLI**: `cli.py` calls `api.fit`, `api.load_run(...).report()`,
  `api.sweep`; output text unchanged (tests pin it).
- **Notebook**: `examples/notebook_api/lw_sv_from_archive.ipynb`, executed
  and committed with outputs, running the whole output layer on
  `runs-archive/a00958509083` (no sampling), plus a spec-building and
  hash-parity cell; a fast test executes it via nbclient.

## WP2 — family #2: UCSV

Model (`docs/kf-capability-matrix.md`, binding): `pi_t = tau_t + eps_t`,
`tau_t = tau_{t-1} + eta_t`, `eps_t ~ N(0, exp(h_eps,t))`, `eta_t ~ N(0,
exp(h_eta,t))`, both `h` non-centered random walks (`h_0 ~ N(mu_h0, sd^2)`,
`mu_h0` a data-derived anchor as in lw_sv), `tau_0 ~ N(pi_1, sd^2)`
explicit. State n=1, obs m=1, NO exogenous block (k=0: the engine's
declared degenerate case).

**WP2a — the one real KF extension (`Q_t`)**, R_t playbook exactly:
`kalman_loglik(yobs, x, F, array[] matrix Q, A, Z, array[] matrix R, ...)`
core; overloads for constant Q with array R, constant Q with constant R
(existing signatures, now delegating via `rep_array`), and array Q with
constant R. Python: `_kf_core` takes a `(T, n, n)` Q path; `_as_Q_path`
mirrors `_as_R_path`; `kalman_loglik`/`kalman_smoother`/
`simulate_smoother_draw` accept constant-or-path Q (the DK plus-path
draws `sqrt(Q_t)` per period). G1 harness extension: `loglik_tvq` (Q_t
path, plain inline diag build) and `loglik_svq` (production composition:
`sv_rw_noncentered` → scalar variance path) alongside the three existing
paths, at all 50 points, gate 1e-8. Constant-Q pins per conflict item 1.
Numerics-reviewer pass before the commit.

**WP2b — declarations**: `stan/templates/ucsv.stan.j2` (+ a ucsv-only
`stan/functions/sv_scalar_variance_path.stan` helper building the 1x1
array paths), `specs/schema/ucsv.py` (`UcsvOptions`: `sv_shocks` = `[]`
or canonical `[eps, eta]` (default), same discipline as lw_sv; `UcsvOutputs`
with the same field set as lw_sv's minus `forecast_r_rule`;
`DEFAULT_PRIORS`), `macrotoolkit/families/ucsv.py` (`UCSV_STATE_META` with
`(("tau", 0),)`, shock `eta` loading 1.0 into `("tau", 0)`, measurement
shock `eps`, obs `pi`, EMPTY feedback map; matrix builder; data/render
builders; `mu_h0` anchors from `Var(Δpi)`; prior sampler; prior sds;
headline series), one `FAMILY_REGISTRY` entry, and
`macrotoolkit/results_ucsv.py` as a thin declaration over a new generic
results core. Generic-layer fixes expected (each recorded in DECISIONS):
the simulation smoother's structural-shock recovery is lw_sv-hardcoded
(slots 0/3/5, `/4`) → generic recovery from `StateSpaceMeta.shock_loadings`
(triangular, bit-identical for lw_sv — pinned); the engine's forward
simulation draws constant-Q process noise → a `StateNoise` protocol with
the constant case as default (RNG order preserved) and an SV state-noise
model for UCSV's fan; the report/plots/param-table lw_sv assumptions
(header rows, `LWRun`, `load_lw_run`) → registry-driven. `results_lw.py`
is NOT forked and NOT rewritten: it delegates to generic pieces where the
delegation is behavior-preserving and stays as the validated lw_sv
instantiation otherwise.

**WP2c — validation ladder for ucsv** (no external oracle exists for
UCSV — stated explicitly in the README/HANDOFF; SBC + recovery carry the
weight): G1 (WP2a's extension, on the shared filter), G2 recovery (20
simulated datasets, coverage/bias arithmetic lifted from `test_g2` into a
generic helper), SBC through `sbc_harness.SbcDesign` at a PRE-REGISTERED
design recorded in DECISIONS.md before the run (see below), G6 HD
identity per period per draw. Expensive tests `-m slow`; the slow ucsv
suite is run once this stage and its results recorded.

Pre-registered UCSV SBC design (to be copied verbatim into DECISIONS.md
before the run starts): 100 replications, T=100, 2 chains x 750/750,
adapt_delta 0.95, max_treedepth 12, ranks from 99 evenly thinned pooled
draws, 10 chi^2 bins, p-floor 0.001 per quantity, divergence ceiling 150;
ranked quantities: `sigma_h_eps`, `sigma_h_eta`, `h0_eps`, `h0_eta`
(recovered from the non-centered `h0_*_raw` via the template's own line);
fixed anchors `mu_h0_eps = mu_h0_eta = 2*ln(0.5)` and fixed initial-state
anchor `tau_0 ~ N(2.0, sd_tau0^2)` passed identically to simulator and
fit; seeds `UCSV_SBC_SEED_BASE = 20260920` (+ i). A <=3-rep smoke may run
first solely to measure per-rep cost; the design does not shrink.

**Worked example**: `examples/us_ucsv/` — US quarterly core PCE inflation,
built by `make_data.py` from the HLW fixture workbook already in the repo
(the same `inflation` column `examples/us_lw_sv` uses, documented), spec +
README; demonstrated end-to-end through the WP1 notebook API in
`examples/notebook_api/ucsv_us_inflation.ipynb` (this one samples — UCSV
is cheap) with `mtk run` + `mtk report` as the CLI cross-check.

## WP3 — automatic per-model QC

- **Auto-G1 at fit time** (`qc.mirror_check`, default ON, `qc.mirror_points`
  default 5): before sampling, `fit`/`mtk run` draws K parameter points
  from the family's prior sampler, converts each to Stan inits via a new
  family capability `mirror_inits(params, stan_data) -> dict`, runs the
  EXACT rendered program with `fixed_param=True, iter_sampling=1` at
  those inits (reads `kf_loglik`), evaluates the Python mirror at the
  same point (family capability `mirror_loglik(params, stan_data) ->
  float`), records `{n_points, max_abs_diff, tolerance, passed}` in
  `diagnostics.json["mirror_check"]`, and raises (no partial run dir left
  behind) past 1e-8. Report header shows the result. Cost: K process
  spawns, sub-second each.
- **`mtk validate <family> [--tier fast|recovery|sbc|all]` / `api.validate`**:
  the registry's `validation_suite` capability points at a
  `macrotoolkit/families/<family>_validation.py` module declaring
  `ValidationSuite(fast=[...], recovery=[...], sbc=[...])`, each gate a
  `Gate(name, tier, run() -> GateResult(verdict PASS/WARN/FAIL, metrics,
  notes))`. The SBC engine moves into the package
  (`macrotoolkit/validation/sbc.py`; `tests/sbc_harness.py` becomes a
  re-export shim so G3/G4 tests are untouched) together with G4's design
  and simulator; lw_sv and ucsv both wired. One validation report
  (`validation/<family>/report.html`, reusing the report machinery)
  summarizes PASS/WARN/FAIL per gate.
- **CI-style one-liner**: `scripts/gate-check.sh [fast|validate <family>
  <tier>]` and pytest markers `ucsv`, `lw_sv`, `validation`.

## Sequencing and commits

1. Baseline + plan (this file) + pre-change G1 fixture capture.
2. WP1: run-core refactor + api + tests; output_modules registry + generic
   report; sweep core; CLI thin; notebook. (2–3 commits.)
3. WP2a: Q_t generalization + G1 extension + pins + reviewer pass. (1 commit.)
4. WP2b: generic results core + UCSV declarations + fast tests + reviewer
   pass over the smoother/engine generalizations. (1–2 commits.)
5. WP2c: gates + SBC pre-registration entry → run → record; example +
   notebook. (2 commits.)
6. WP3: qc block + mirror check + validate + gate-check. (1–2 commits.)
7. Stage-end docs.

If WP3 is at risk, WP1+WP2 ship complete and WP3 is scoped down in
HANDOFF; if WP2c's SBC cannot complete (container restarts), the
crash-resume engine is used and the partial record is stated honestly —
never a shrunk design.

## Acceptance criteria (from the brief, verbatim intent)

- WP1: committed executed notebook under `examples/notebook_api/` running
  the archived runs' output layer without sampling; unit tests for
  spec-hash parity and DataFrame input; CLI behavior unchanged.
- WP2: Q_t in Stan + Python mirror gated by a G1 extension over
  constant-Q, time-varying-Q, and production SV composition; constant-Q
  regression pin; UCSV family registered with named state metadata, EMPTY
  feedback map, prior sampler, builders, thin results module; G1/G2/SBC
  (pre-registered)/G6 for ucsv with the slow suite run once; worked
  example through the notebook API with CLI cross-check; every lw_sv
  residue generalized is listed in DECISIONS.
- WP3: mirror check ON by default, recorded in the run record and the
  report header, failing loudly past 1e-8; `mtk validate <family>` with
  tiers and one validation report; lw_sv and ucsv wired; one-line
  CI-style invocation.
- Stage end: README (ladder + repo map), HANDOFF rewrite for S7, DECISIONS,
  example READMEs with hash lineage; fast suite green with its count
  stated; slow gates recorded; branch pushed, no PR.
