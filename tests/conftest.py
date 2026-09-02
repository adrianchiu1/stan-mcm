"""Shared fixtures for the macrotoolkit test suite.

Key isolation rule: no test writes into the repo's real `runs/` directory.
Any test that calls `macrotoolkit.run.run()` must pass `runs_root=<tmp_path
based dir>` explicitly (the `runs_root` fixture below), so tests never
collide with each other or with a developer's local `runs/`.

`CMDSTAN` is expected to already be set in the environment per the task's
env-setup instructions (pinned to `~/.cmdstan/cmdstan-2.36.0`). The
`_ensure_cmdstan` autouse fixture below just calls
`macrotoolkit.render.ensure_cmdstan_path()` once per session so a
misconfigured environment fails fast, at collection time, with the clear
error message that function already provides -- rather than deep inside a
render/compile call in some unrelated test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def _ensure_cmdstan() -> None:
    from macrotoolkit.render import ensure_cmdstan_path

    ensure_cmdstan_path()


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    """A fresh, empty `runs/`-like directory, private to this test."""
    root = tmp_path / "runs"
    root.mkdir()
    return root


def _tiny_local_level_rows(n: int = 12, seed: int = 20260813) -> pd.DataFrame:
    """A small, fast, deterministic local-level dataset -- distinct from
    `examples/toy/data.csv` (which is reserved for the S1 acceptance test).
    Twelve quarters is enough to exercise the model's plumbing without the
    cost of a ~100-row sample."""
    rng = np.random.default_rng(seed)
    sigma_obs, sigma_level, mu0 = 0.4, 0.3, 2.0
    eta = rng.normal(0.0, sigma_level, size=n)
    mu = np.empty(n)
    mu[0] = mu0
    for t in range(1, n):
        mu[t] = mu[t - 1] + eta[t]
    eps = rng.normal(0.0, sigma_obs, size=n)
    y = mu + eps
    dates = pd.date_range(start="2000-01-01", periods=n, freq="QS")
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "obs": y})


@pytest.fixture
def tiny_data_csv(tmp_path: Path) -> Path:
    """Writes a small local-level CSV (`date`, `obs` columns) and returns
    its path."""
    path = tmp_path / "tiny_data.csv"
    _tiny_local_level_rows().to_csv(path, index=False)
    return path


def tiny_spec_dict(data_file_name: str = "tiny_data.csv") -> dict:
    """A minimal, valid RunSpec-shaped dict for the local_level family,
    with tiny sampler settings (fast: a couple hundred total iterations)
    for tests that need *a* completed run, not sampling quality."""
    return {
        "model": {"family": "local_level", "options": {}},
        "data": {
            "file": data_file_name,
            "date_column": "date",
            "mapping": {"y": "obs"},
            "sample": {"start": None, "end": None},
        },
        "priors": {},
        "sampler": {
            "chains": 2,
            "warmup": 50,
            "sampling": 50,
            "adapt_delta": 0.8,
            "max_treedepth": 10,
            "seed": 20260813,
        },
        "outputs": {},
    }


@pytest.fixture
def tiny_spec_path(tmp_path: Path, tiny_data_csv: Path) -> Path:
    """Writes a minimal, valid spec YAML next to `tiny_data_csv` (spec
    references the data file by its name, resolved relative to the spec's
    own directory) and returns the spec's path."""
    spec = tiny_spec_dict(data_file_name=tiny_data_csv.name)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return spec_path


@pytest.fixture
def toy_spec_path(repo_root: Path) -> Path:
    """The real, checked-in `examples/toy/spec.yaml` -- for the S1
    acceptance test only."""
    return repo_root / "examples" / "toy" / "spec.yaml"


# ---------------------------------------------------------------------------
# S4 IRF/fan-chart shared fixtures: one tiny, real no-SV lw_sv run and one
# tiny, real SV lw_sv run, SESSION-scoped so the underlying MCMC sampling
# only ever runs ONCE per pytest session, shared by every test FILE that
# needs a real completed lw_sv run for macrotoolkit.results_lw's Part C
# (tests/test_irf.py) / Part D (tests/test_fan_charts.py) coverage.
# Deliberately independent of tests/test_results_lw.py's own (module-scoped,
# not cross-file-shared) fixtures of the same shape -- that file is not
# touched by this addition.
# ---------------------------------------------------------------------------


def _s4_lw_sv_spec_dict(data_file_name: str, sv_shocks: list[str]) -> dict:
    return {
        "model": {"family": "lw_sv", "options": {"sv_shocks": sv_shocks, "estimate_c": False}},
        "data": {
            "file": data_file_name,
            "date_column": "date",
            "mapping": {"y": "gdp_log100", "pi": "core_infl", "r": "real_short_rate"},
            "sample": {"start": None, "end": None},
        },
        "priors": {},
        "sampler": {
            "chains": 2,
            "warmup": 50,
            "sampling": 50,
            "adapt_delta": 0.8,
            "max_treedepth": 10,
            "seed": 20260813,
        },
        "outputs": {},
    }


def _s4_make_lw_sv_run(
    data_csv: Path, sv_shocks: list[str], tmp_path_factory: pytest.TempPathFactory, label: str
) -> Path:
    from macrotoolkit.run import run

    spec_dict = _s4_lw_sv_spec_dict(data_csv.name, sv_shocks)
    spec_path = data_csv.parent / f"spec_{label}.yaml"
    spec_path.write_text(yaml.safe_dump(spec_dict, sort_keys=False))
    runs_root = tmp_path_factory.mktemp(f"s4_lw_sv_runs_{label}")
    result = run(spec_path, runs_root=runs_root)
    return result.run_dir


@pytest.fixture(scope="session")
def s4_lw_sv_data_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small (T=50 raw quarters -> 46 estimation rows), fast, deterministic
    lw_sv-shaped CSV, built from `g1_harness.SYNTHETIC_DATA` -- shared by
    every S4 IRF/fan-chart test file (session-scoped, so it is built once
    per pytest session)."""
    from g1_harness import SYNTHETIC_DATA

    d = tmp_path_factory.mktemp("s4_lw_sv_data")
    path = d / "lw_sv_data.csv"
    df = pd.DataFrame(
        {
            "date": SYNTHETIC_DATA.dates.strftime("%Y-%m-%d"),
            "gdp_log100": SYNTHETIC_DATA.y,
            "core_infl": SYNTHETIC_DATA.pi,
            "real_short_rate": SYNTHETIC_DATA.r,
        }
    )
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def s4_no_sv_run_dir(s4_lw_sv_data_csv: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny, real, completed no-SV (`sv_shocks: []`) lw_sv run directory,
    built once per session and shared by tests/test_irf.py and
    tests/test_fan_charts.py."""
    return _s4_make_lw_sv_run(s4_lw_sv_data_csv, [], tmp_path_factory, "s4_no_sv")


@pytest.fixture(scope="session")
def s4_sv_run_dir(s4_lw_sv_data_csv: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The SV (`sv_shocks: [is, pc]`) counterpart of `s4_no_sv_run_dir`."""
    return _s4_make_lw_sv_run(s4_lw_sv_data_csv, ["is", "pc"], tmp_path_factory, "s4_sv")


@pytest.fixture(scope="session")
def s4_no_sv_lw_run(s4_no_sv_run_dir: Path):
    from macrotoolkit.results_lw import load_lw_run

    return load_lw_run(s4_no_sv_run_dir)


@pytest.fixture(scope="session")
def s4_sv_lw_run(s4_sv_run_dir: Path):
    from macrotoolkit.results_lw import load_lw_run

    return load_lw_run(s4_sv_run_dir)
