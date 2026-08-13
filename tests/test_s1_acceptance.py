"""S1 acceptance test (lw-sv-spec.md §7 / plans/S1-plan.md §"Files to
create"):

    `mtk run` produces immutable run dir with draws + diagnostics for the
    toy model.

Restated concretely in plans/S1-plan.md: `mtk run examples/toy/spec.yaml`
exits 0 and produces `runs/<hash12>/` containing `spec.yaml`,
`data.snapshot.csv`, `draws.nc` (valid ArviZ InferenceData, nonzero draws
for every declared parameter), `diagnostics.json` (PASS/WARN/FAIL verdict +
reasons), and `log.txt`.

Interface choice -- `macrotoolkit.run.run()` directly, not
`click.testing.CliRunner` / subprocess on `mtk`: `src/macrotoolkit/cli.py`'s
`run` command has no `--runs-root` override; it always writes to
`REPO_ROOT / "runs"` (the real, checked-in repo path) via
`macrotoolkit.run.run()`'s default. Driving this test through the CLI as-is
would write real run artifacts into the developer's actual `runs/`
directory as a side effect of running the test suite -- exactly what
`tests/conftest.py`'s `runs_root` fixture exists to avoid. Calling
`macrotoolkit.run.run()` directly with an explicit `runs_root=tmp_path`
exercises the identical pipeline (`cli.py`'s `run` command is a thin
wrapper: parse the path, call this same function, format the result as
text) while keeping the test hermetic. This is specifically the one test
in the suite that uses the real, shipped `examples/toy/spec.yaml` +
`examples/toy/data.csv` (not a tiny override) -- appropriate since it is
testing the shipped example itself.

Per plans/S1-plan.md's `DECISIONS.md` entry (2026-08-13, "S1 toy model
shows real divergences"), this toy model has known mild divergences and is
expected to produce verdict FAIL -- but the *pipeline* (schema -> render ->
compile -> sample -> store -> diagnostics) must complete and produce a
well-formed diagnostics artifact regardless. We assert that, not a clean
PASS.
"""
from __future__ import annotations

from pathlib import Path

import arviz as az
import numpy as np
import pytest

from macrotoolkit.run import run

REQUIRED_ARTIFACTS = ("spec.yaml", "data.snapshot.csv", "draws.nc", "diagnostics.json", "log.txt", "_SUCCESS")


@pytest.fixture(scope="module")
def toy_run_result(tmp_path_factory: pytest.TempPathFactory):
    """One real sampling run of the shipped toy example, shared across the
    assertions in this module (chains=4, warmup=1000, sampling=1000, per
    examples/toy/spec.yaml -- not overridden, since this test is
    specifically about the shipped example)."""
    repo_root = Path(__file__).resolve().parents[1]
    spec_path = repo_root / "examples" / "toy" / "spec.yaml"
    runs_root = tmp_path_factory.mktemp("s1_acceptance_runs")
    return run(spec_path, runs_root=runs_root)


def test_run_exits_successfully_and_is_new(toy_run_result) -> None:
    assert toy_run_result.is_new is True
    assert toy_run_result.run_dir.is_dir()
    assert len(toy_run_result.run_id) == 12


def test_run_dir_contains_all_required_artifacts(toy_run_result) -> None:
    for name in REQUIRED_ARTIFACTS:
        path = toy_run_result.run_dir / name
        assert path.exists(), f"missing required artifact: {name}"


def test_data_snapshot_is_byte_identical_to_source_csv(toy_run_result) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_csv = repo_root / "examples" / "toy" / "data.csv"
    snapshot = toy_run_result.run_dir / "data.snapshot.csv"
    assert snapshot.read_bytes() == source_csv.read_bytes()


def test_draws_nc_is_valid_inference_data_with_nonzero_variance_draws(toy_run_result) -> None:
    draws_path = toy_run_result.run_dir / "draws.nc"
    idata = az.from_netcdf(str(draws_path))

    assert hasattr(idata, "posterior")
    posterior = idata.posterior

    for var in ("sigma_obs", "sigma_level"):
        assert var in posterior.data_vars, f"posterior missing declared parameter {var!r}"
        values = posterior[var].values
        assert values.size > 0
        assert np.isfinite(values).all(), f"{var} draws contain non-finite values"
        assert np.std(values) > 0, f"{var} draws have zero variance -- sampler did not move"

    # `mu` (the state path) is also a declared parameter of the toy model.
    assert "mu" in posterior.data_vars
    assert np.std(posterior["mu"].values) > 0


def test_diagnostics_json_is_well_formed(toy_run_result) -> None:
    import json

    diagnostics_path = toy_run_result.run_dir / "diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())

    assert diagnostics["verdict"] in {"PASS", "WARN", "FAIL"}
    assert isinstance(diagnostics["reasons"], list)
    assert len(diagnostics["reasons"]) > 0
    assert all(isinstance(r, str) and r for r in diagnostics["reasons"])

    assert isinstance(diagnostics["divergences"], int)
    assert isinstance(diagnostics["max_treedepth_hits"], int)
    assert "rhat_max" in diagnostics
    assert "ess_bulk_min" in diagnostics
    assert "ess_tail_min" in diagnostics
    assert "e_bfmi_per_chain" in diagnostics

    # This test deliberately does NOT assert verdict == "PASS": the toy
    # model has known, documented mild divergences (DECISIONS.md,
    # 2026-08-13). It asserts the diagnostics artifact is well-formed and
    # that run.verdict (returned by run()) matches the file on disk.
    assert toy_run_result.verdict == diagnostics["verdict"]


def test_run_result_verdict_is_a_known_value(toy_run_result) -> None:
    # Documents current expectation without hard-freezing it: if this ever
    # flips to PASS because the model/priors improved, that's fine; if it's
    # something outside {PASS, WARN, FAIL}, that's a real bug.
    assert toy_run_result.verdict in {"PASS", "WARN", "FAIL"}


def test_rerun_of_toy_spec_is_idempotent(toy_run_result) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    spec_path = repo_root / "examples" / "toy" / "spec.yaml"
    result2 = run(spec_path, runs_root=toy_run_result.run_dir.parent)
    assert result2.is_new is False
    assert result2.run_id == toy_run_result.run_id
    assert result2.run_dir == toy_run_result.run_dir
