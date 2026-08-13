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
