# S1 plan — spec schema, run store, render/compile/run harness, toy model

Spec §7 acceptance test (verbatim): *"`mtk run` produces immutable run dir
with draws + diagnostics for the toy model."* Concretely, restated:

> `mtk run examples/toy/spec.yaml` exits 0, and produces a new directory
> `runs/<hash12>/` containing at least `spec.yaml`, `data.snapshot.csv`,
> `draws.nc` (a valid ArviZ `InferenceData` with a `posterior` group and
> nonzero draws for every declared parameter), `diagnostics.json` (valid
> JSON with a divergence count, R-hat/ESS summary, and a PASS/WARN/FAIL
> verdict string), and `log.txt`. Running the identical spec+data again
> produces the *same* `<hash12>` and does not rewrite the existing
> directory (immutability). Changing one byte of the spec or data produces
> a *different* hash and a new directory.

S1 does not touch `stan/functions/kalman_loglik_tv.stan` or anything
LW-specific — that's S2. It proves the plumbing with a model simple enough
to have no real geometry risk.

## Toy model: local-level

Not specified verbatim in `lw-sv-spec.md` (S1's toy is deliberately outside
the LW-SV spec's scope) — designing it now, logging the choice in
`DECISIONS.md` once approved. Standard two-parameter local-level model:

```
y_t   = mu_t + eps_t,      eps_t ~ N(0, sigma_obs^2)
mu_t  = mu_{t-1} + eta_t,  eta_t ~ N(0, sigma_level^2)
mu_1  ~ N(y_1, 10)                          # weakly-informative init
sigma_obs, sigma_level ~ Half-N(0, 1)
```

**Implementation choice:** sample `mu[1..T]` directly as Stan parameters
(non-centered: `mu[t] = mu[t-1] + sigma_level * eta_raw[t]`), rather than
marginalizing with a Kalman filter. Spec's shared KF library
(`kalman_loglik_tv.stan`, Rao-Blackwellized marginalization) is explicitly
S2 scope, built for the LW model's needs (time-varying `Q_t`). Building a
second, throwaway KF for a toy model in S1 would duplicate that work
sloppily. Direct-sampling a length-~100 random walk is simple, has no
geometry risk worth mentioning, and still exercises every plumbing stage
spec cares about for S1 (schema → render → compile → sample → store).

Synthetic data: generate a ~100-quarter series in Python from fixed seed
20260813 at known `sigma_obs`, `sigma_level`, save as
`examples/toy/data.csv` (checked in, not regenerated at run time — runs
must be reproducible from a static file per the run-identity hash).

## Files to create

```
pyproject.toml                          # deps + `mtk` console-script entry point
.gitignore                              # runs/, __pycache__/, .venv/, .cmdstan/, *.egg-info

specs/schema/__init__.py                # family registry: name -> (pydantic model, template path)
specs/schema/base.py                    # RunSpec: data/priors/sampler/outputs blocks common to all families
specs/schema/local_level.py             # local_level family options fragment (toy)

stan/templates/local_level.stan.j2      # toy model template

src/macrotoolkit/__init__.py
src/macrotoolkit/data.py                # CSV load, column mapping, sample trim, content hash
src/macrotoolkit/render.py              # Jinja render -> .stan; source-hash compile cache via CmdStanModel
src/macrotoolkit/run.py                 # run identity hash, CmdStanPy execution, immutable run store
src/macrotoolkit/cli.py                 # `mtk run <spec.yaml>` (click)

examples/toy/spec.yaml
examples/toy/data.csv

tests/conftest.py
tests/test_units_conventions.py         # placeholders for g/4, exp(h/2), 400*dlog(P) — see note below
tests/test_spec_schema.py               # validation errors name the field + say what's valid
tests/test_data.py                      # load/mapping/hash
tests/test_render.py                    # jinja render + compile cache behavior
tests/test_run_store.py                 # hashing, immutability, directory contents
tests/test_s1_acceptance.py             # the acceptance test above, end to end via `mtk run`
```

**Not created in S1:** anything under `src/macrotoolkit/smoother.py`,
`results_lw.py`, `plots.py`, `report.py`, or `stan/functions/`. Those are
S2–S4. `mtk report <hash>` is stubbed as a `NotImplementedError` CLI path
at most, not built out.

## Design notes / how S1 pieces map to spec §2.3

- **Run identity** = SHA-256 over (canonicalized spec YAML, data file hash,
  rendered Stan source hash, CmdStan version) — matches spec exactly.
  "Canonicalized" = parsed through the Pydantic model and re-serialized
  with sorted keys, so key order / comments / whitespace in the YAML don't
  change the hash.
- **Run store immutability**: `run.py` refuses to write into an existing
  `runs/<hash12>/` — if the directory exists with a `_SUCCESS` marker
  (written last), it's a no-op success (idempotent re-run); if it exists
  without one (a previous crashed run), that's an error asking the human
  to remove it manually rather than silently overwriting.
- **Family registry**: `specs/schema/__init__.py` maps `model.family` →
  Pydantic options model + Jinja template path, so `lw_sv` can be added in
  S2 by registering a new entry, not by touching S1's dispatch code.
  `RunSpec.model.options` is validated via a Pydantic discriminated union
  on `family`; S1 only registers `local_level`.
- **CmdStan version pin**: uses the `2.36.0` pin from `PREFLIGHT.md` /
  `DECISIONS.md`. `render.py`'s compile cache is keyed on
  `(stan_source_hash, cmdstan_version)`.
- **Diagnostics scope for S1**: divergence count, max-treedepth hit count,
  E-BFMI per chain, R-hat/bulk-ESS/tail-ESS per parameter (via ArviZ), and
  a PASS/WARN/FAIL verdict with the reasons that produced it. This is the
  foundation spec §4 elaborates in S4 (prior-predictive figure, HTML
  embedding) — not built now.
- **Units-convention tests**: spec's `g` annualized / `h` log-variance /
  `400*dlog(P)` conventions don't apply to the toy local-level model at
  all (it has no such quantities) — S2 introduces them alongside the LW
  equations. `tests/test_units_conventions.py` is created in S1 as an
  empty-but-documented placeholder (per the unsupervised-run scope note
  asking for "units-convention unit tests" to be staged even before S2),
  stating the three conventions in its module docstring and `pytest.skip`
  until S2 provides real functions to test against — not asserting
  anything vacuous.

## Dependencies beyond the S0-verified set

S0 verified the install-risk set (`cmdstanpy`, `numba`, `arviz`, `pydantic`,
`jinja2`, `pytest`, `matplotlib`). S1 additionally needs, all pure-Python
and low-risk given PyPI access is confirmed working: `pyyaml` (spec I/O),
`pandas` (CSV/data handling), `click` (CLI), `h5netcdf` or `netCDF4`
(ArviZ's `to_netcdf`, needed for `draws.nc`). Flagging so this isn't a
silent scope add — will note in `DECISIONS.md` once confirmed installable
(expected trivial; not re-verifying each individually unless one fails).

## `.claude/active-gate`

Not created in S1. Per `.claude/README.md`, `gate-check.sh` runs the fast
suite (`pytest -q -m "not slow"`) whenever `.claude/active-gate` is absent,
and `tests/` already exists — so the hook is live for S1 without needing a
named gate (there is no G-numbered gate until G1 in S2). S2 will create
`.claude/active-gate` containing `G1` when it starts. Flagging as a plan
decision rather than silently deciding it, since the README's phrasing
("Stage S1 should create it") could be read either way.

## Open questions

1. **CLI framework**: proposing `click` (small, stable, no extra
   transitive deps). No strong reason to prefer `typer`/`argparse` instead;
   say if you'd rather one of those.
2. **`draws.nc` backend**: ArviZ needs either `h5netcdf` or `netCDF4` to
   write NetCDF. Proposing `h5netcdf` (lighter pure-wheel install, no
   system HDF5/netCDF C library dependency risk). Flag if you have a
   preference.
3. **Toy model design** (above) — direct-sample the state path rather than
   building a throwaway KF. Confirm this reads as in-scope for S1 and not
   as under-delivering on "prove the estimation harness."
4. Everything in `FIXTURES.md`'s open item (HLW model-vintage target for
   G5a) is **not** an S1 blocker — restating here only so it isn't lost;
   it needs an answer before S2's G5a test is designed, not before S1
   starts.

Waiting for go-ahead before implementing, per your instructions.
