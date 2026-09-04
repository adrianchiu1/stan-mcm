"""``macrotoolkit.api`` (S6 WP1): the notebook-first Python API.

Pins the two acceptance properties the S6 brief names -- (1) a spec built
in Python produces the IDENTICAL run identity as its YAML equivalent, and
(2) a pandas DataFrame is accepted as data with a deterministic identity
-- plus the Run handle / outputs / sweep surface, and the CLI's
thin-shell property (its commands call the API).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import macrotoolkit
from macrotoolkit import api
from macrotoolkit import api as mtk
from macrotoolkit.data import load_data
from macrotoolkit.render import get_cmdstan_version, render_stan_source, stan_source_hash
from macrotoolkit.run import build_render_context, compute_run_id
from specs.schema import get_family
from specs.schema.base import RunSpec, load_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SPEC = REPO_ROOT / "examples" / "us_lw_sv" / "spec_sv.yaml"
ARCHIVED_RUN_DIR = REPO_ROOT / "runs-archive" / "a00958509083"


def _identity(spec: RunSpec, base_dir: Path) -> str:
    """The run hash exactly as run_spec() computes it (no sampling)."""
    _, raw_hash, _ = load_data(spec, base_dir=base_dir)
    source = render_stan_source(get_family(spec.model.family).template, build_render_context(spec))
    return compute_run_id(spec, raw_hash, stan_source_hash(source), get_cmdstan_version())[1]


# ---------------------------------------------------------------------------
# (1) spec-hash parity: Python-built spec == YAML spec
# ---------------------------------------------------------------------------


def test_python_built_spec_has_identical_run_identity_to_yaml_equivalent() -> None:
    yaml_spec = load_spec(str(EXAMPLE_SPEC))
    py_spec = mtk.spec(
        "lw_sv",
        data={
            "file": "data/us_quarterly.csv",
            "date_column": "date",
            "mapping": {"y": "lgdp100", "pi": "core_pce_ann", "r": "real_rate"},
        },
        options={"sv_shocks": ["is", "pc"], "estimate_c": False},
        sampler={"chains": 4, "warmup": 1500, "sampling": 1500, "adapt_delta": 0.95, "max_treedepth": 12, "seed": 20260813},
    )
    assert py_spec.to_estimation_yaml() == yaml_spec.to_estimation_yaml()
    assert _identity(py_spec, EXAMPLE_SPEC.parent) == _identity(yaml_spec, EXAMPLE_SPEC.parent)


def test_python_built_spec_validates_like_yaml() -> None:
    with pytest.raises(ValueError, match="Unknown model.family"):
        mtk.spec("no_such_family", data={"file": "x.csv", "date_column": "date", "mapping": {"y": "y"}})
    with pytest.raises(ValueError, match="missing required key"):
        mtk.spec("lw_sv", data={"file": "x.csv", "date_column": "date", "mapping": {"y": "y"}})
    with pytest.raises(ValueError, match="sv_shocks"):
        mtk.spec(
            "lw_sv",
            data={"file": "x.csv", "date_column": "date", "mapping": {"y": "y", "pi": "pi", "r": "r"}},
            options={"sv_shocks": ["is"]},
        )


def test_spec_options_and_outputs_are_typed_models() -> None:
    s = mtk.spec(
        "lw_sv",
        data={"file": "x.csv", "date_column": "date", "mapping": {"y": "y", "pi": "pi", "r": "r"}},
        outputs={"horizon": 8, "smoother_draws": {"thin": 3}},
        qc={"mirror_points": 2},
    )
    assert s.model.options.sv_shocks == []
    assert s.outputs.horizon == 8 and s.outputs.smoother_draws.thin == 3
    assert s.qc.mirror_points == 2 and s.qc.mirror_check is True


# ---------------------------------------------------------------------------
# (2) DataFrame input
# ---------------------------------------------------------------------------


def _frame(n: int = 12, seed: int = 20260904) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="QS")
    return pd.DataFrame({"date": dates, "obs": rng.normal(size=n), "unused": np.arange(n)})


def _ll_spec() -> RunSpec:
    return mtk.spec(
        "local_level",
        data={"file": "dataframe.csv", "date_column": "date", "mapping": {"y": "obs"}},
        sampler={"chains": 2, "warmup": 50, "sampling": 50, "seed": 20260813},
    )


def test_dataframe_staging_is_canonical_and_content_determined(tmp_path: Path) -> None:
    spec = _ll_spec()
    df = _frame()
    b1 = api.dataframe_csv_bytes(df, spec)
    # Same content, shuffled columns, a different index, dates as strings.
    df2 = df[["unused", "obs", "date"]].copy()
    df2.index = df2.index + 100
    df2["date"] = df2["date"].dt.strftime("%Y-%m-%d")
    b2 = api.dataframe_csv_bytes(df2, spec)
    assert b1 == b2
    # Only the spec's columns are serialized (the unused column is dropped).
    header = b1.decode().splitlines()[0]
    assert header == "date,obs"
    # A perturbed value changes the bytes (and therefore the identity).
    df3 = df.copy()
    df3.loc[3, "obs"] += 1e-9
    assert api.dataframe_csv_bytes(df3, spec) != b1
    # Missing required column fails loudly, naming it.
    with pytest.raises(ValueError, match="missing column"):
        api.dataframe_csv_bytes(df.drop(columns=["obs"]), spec)

    staged = api.stage_dataframe(df, spec, tmp_path / "stage")
    assert staged.data.file == api.DATAFRAME_FILE_NAME
    assert (tmp_path / "stage" / api.DATAFRAME_FILE_NAME).read_bytes() == b1


def test_fit_accepts_dataframe_and_is_idempotent(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    spec = _ll_spec()
    run = mtk.fit(spec, _frame(), runs_root=runs_root)
    assert run.is_new
    assert (run.run_dir / "_SUCCESS").is_file()
    assert (run.run_dir / "qc.yaml").is_file()
    assert run.family == "local_level"
    assert run.verdict in ("PASS", "WARN", "FAIL")
    assert run.spec.data.file == api.DATAFRAME_FILE_NAME
    # The snapshot is the canonical serialization -- the run dir is self-contained.
    assert (run.run_dir / "data.snapshot.csv").read_bytes() == api.dataframe_csv_bytes(_frame(), spec)

    again = mtk.fit(spec, _frame(), runs_root=runs_root)
    assert not again.is_new and again.hash == run.hash

    other = mtk.fit(spec, _frame(seed=1), runs_root=runs_root)
    assert other.hash != run.hash

    loaded = mtk.load_run(run.hash, runs_root=runs_root)
    assert loaded.run_dir == run.run_dir
    assert mtk.load_run(run.run_dir).hash == run.hash
    assert "local_level" in repr(loaded)
    assert loaded.idata.posterior.sizes["chain"] == 2


def test_fit_accepts_csv_path_and_matches_yaml_identity(tmp_path: Path) -> None:
    """A CSV path given to fit() + a Python spec reproduce the identity of
    the equivalent YAML spec pointing at the same file (same relative path
    string, same base dir)."""
    df = _frame()
    csv_path = tmp_path / "obs.csv"
    df[["date", "obs"]].to_csv(csv_path, index=False, date_format="%Y-%m-%d")
    spec_yaml = {
        "model": {"family": "local_level", "options": {}},
        "data": {"file": "obs.csv", "date_column": "date", "mapping": {"y": "obs"}},
        "sampler": {"chains": 2, "warmup": 50, "sampling": 50, "seed": 20260813},
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_yaml))
    runs_root = tmp_path / "runs"

    py_spec = mtk.spec("local_level", data={"file": "placeholder.csv", "date_column": "date", "mapping": {"y": "obs"}},
                       sampler={"chains": 2, "warmup": 50, "sampling": 50, "seed": 20260813})
    run_py = mtk.fit(py_spec, "obs.csv", base_dir=tmp_path, runs_root=runs_root)
    run_yaml = mtk.fit(spec_path, runs_root=runs_root)
    assert run_py.is_new and not run_yaml.is_new
    assert run_py.hash == run_yaml.hash


def test_package_level_shortcuts_do_not_shadow_submodules() -> None:
    """`macrotoolkit.fit` is the API function, but `macrotoolkit.sweep` /
    `.report` are the submodules -- the canonical notebook import is
    `from macrotoolkit import api as mtk`."""
    import types

    assert macrotoolkit.fit is api.fit and macrotoolkit.RunSpec is RunSpec
    import macrotoolkit.sweep as sweep_module

    assert isinstance(macrotoolkit.sweep, types.ModuleType) and macrotoolkit.sweep is sweep_module
    assert callable(api.sweep) and callable(api.report)


def test_load_run_unknown_hash_errors_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        mtk.load_run("000000000000", runs_root=tmp_path)


# ---------------------------------------------------------------------------
# Run handle: results / outputs / param table / report (archived lw_sv run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def archived_run(tmp_path_factory) -> api.Run:
    """A tmp copy of the tracked draw-thinned reference run, thinned
    further for speed (outputs.yaml is outside the identity, so the copy's
    smoother_draws thinning is a legitimate report-option refresh)."""
    if not ARCHIVED_RUN_DIR.is_dir():
        pytest.skip(f"{ARCHIVED_RUN_DIR} not present")
    dest = tmp_path_factory.mktemp("api_archive") / ARCHIVED_RUN_DIR.name
    shutil.copytree(ARCHIVED_RUN_DIR, dest)
    outputs = yaml.safe_load((dest / "outputs.yaml").read_text())
    outputs["smoother_draws"] = {"thin": 20}
    outputs["prior_predictive_draws"] = 20
    (dest / "outputs.yaml").write_text(yaml.safe_dump(outputs, sort_keys=True))
    return mtk.load_run(dest)


def test_run_handle_exposes_results_outputs_and_figures(archived_run: api.Run) -> None:
    from matplotlib.figure import Figure

    run = archived_run
    assert run.hash == "a00958509083"
    assert run.family == "lw_sv"
    assert run.verdict == "PASS"
    assert run.mirror_check is None  # pre-S6 run record
    res = run.results()
    assert res is run.results()  # cached
    assert res.sv_on

    outs = run.outputs()
    assert outs.names == ("prior_predictive", "trend_cycle", "irf", "fan", "hd")
    tcd = outs.compute("trend_cycle")
    assert tcd is outs.compute("trend_cycle")
    assert tcd.rstar.shape[1] == len(res.dates)
    fig = outs.figure("trend_cycle")
    assert isinstance(fig, Figure)
    fans = outs.figure("fan")
    assert isinstance(fans, dict) and list(fans) == ["y_level", "y_growth_4q", "pi", "gap", "rstar"]
    assert all(isinstance(f, Figure) for f in fans.values())
    with pytest.raises(KeyError, match="No output module"):
        outs.figure("nope")
    import matplotlib.pyplot as plt

    plt.close("all")

    table = run.param_table()
    assert {"median", "ci_lo", "ci_hi", "rhat", "ess_bulk", "ess_tail"} <= set(table.columns)
    assert "sigma_g" in table.index and "a1" in table.index


def test_run_report_writes_self_contained_html(archived_run: api.Run) -> None:
    path = archived_run.report()
    assert path == archived_run.run_dir / "report.html"
    text = path.read_text()
    assert "a00958509083" in text and "PASS" in text
    assert text.count("<img ") == 12
    assert "http://" not in text and "https://" not in text
    # The generic header reports the family's display name and options.
    assert "LW-SV report" in text and "sv_shocks" in text
    assert "KF mirror check" in text  # WP3's row is present even for pre-S6 runs


# ---------------------------------------------------------------------------
# sweep from Python
# ---------------------------------------------------------------------------


def test_sweep_from_python_returns_programmatic_comparison(s4_lw_sv_data_csv: Path, tmp_path: Path) -> None:
    base = mtk.spec(
        "lw_sv",
        data={"file": s4_lw_sv_data_csv.name, "date_column": "date",
              "mapping": {"y": "gdp_log100", "pi": "core_infl", "r": "real_short_rate"}},
        options={"sv_shocks": []},
        sampler={"chains": 2, "warmup": 50, "sampling": 50, "seed": 20260813},
    )
    result = mtk.sweep(
        {"name": "api_sweep", "cells": [{"label": "baseline", "priors": {}},
                                        {"label": "sigma_g_tight", "priors": {"sigma_g": {"sd": 0.015}}}]},
        base_spec=base,
        base_dir=s4_lw_sv_data_csv.parent,
        runs_root=tmp_path / "runs",
        sweeps_root=tmp_path / "sweeps",
    )
    assert [c.label for c in result.cells] == ["baseline", "sigma_g_tight"]
    cmp_ = result.comparison
    assert cmp_.swept_params == ["sigma_g"]
    table = cmp_.table()
    assert set(table["cell"]) == {"baseline", "sigma_g_tight"}
    assert table["parameter"].iloc[0] == "sigma_g"  # swept parameters listed first
    row = table[(table.parameter == "sigma_g") & (table.cell == "sigma_g_tight")].iloc[0]
    assert row["swept"] and np.isfinite(row["contraction"]) and row["prior_sd"] > 0
    assert cmp_.headline is not None and set(cmp_.headline["baseline"]) == {"rstar", "gap"}
    assert result.report_path.is_file()
    # A YAML-driven sweep through the same core is idempotent against the store.
    again = mtk.sweep(
        {"name": "api_sweep", "cells": [{"label": "baseline", "priors": {}}]},
        base_spec=base, base_dir=s4_lw_sv_data_csv.parent,
        runs_root=tmp_path / "runs", sweeps_root=tmp_path / "sweeps",
    )
    assert not again.cells[0].run_result.is_new


# ---------------------------------------------------------------------------
# CLI is a thin shell over the API
# ---------------------------------------------------------------------------


def test_cli_commands_delegate_to_api(monkeypatch, tmp_path: Path) -> None:
    from click.testing import CliRunner

    from macrotoolkit.cli import main

    calls: list[tuple] = []

    class _FakeRun:
        hash = "abc123abc123"
        verdict = "PASS"
        is_new = True
        run_dir = tmp_path / "abc123abc123"

    monkeypatch.setattr(api, "fit", lambda spec_path, *a, **k: calls.append(("fit", spec_path)) or _FakeRun())
    spec_path = tmp_path / "s.yaml"
    spec_path.write_text("model: {family: local_level}\n")
    out = CliRunner().invoke(main, ["run", str(spec_path)])
    assert out.exit_code == 0, out.output
    assert calls == [("fit", str(spec_path))]
    assert "abc123abc123" in out.output and "PASS" in out.output


def test_cli_run_source_contains_no_pipeline_logic() -> None:
    import inspect

    from macrotoolkit import cli

    src = inspect.getsource(cli)
    for forbidden in ("compile_model", "render_stan_source", "load_data", "model.sample", "compute_run_id"):
        assert forbidden not in src, forbidden
    assert "api.fit(" in src and "api.sweep(" in src and "api.load_run(" in src
