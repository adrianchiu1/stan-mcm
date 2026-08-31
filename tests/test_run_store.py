"""`macrotoolkit.run` tests: run-identity hashing (`compute_run_id`) and the
immutable run store (`run()`).

Hashing-sensitivity tests use `compute_run_id` directly with fake hash
strings (no real sampling needed). The immutability contract is tested with
actual `run()` calls against tiny sampler settings (`tests/conftest.py`'s
`tiny_spec_path` fixture: chains=2, warmup=50, sampling=50) so each test
still exercises the real render -> compile -> sample -> store pipeline
without paying for a full-size run.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from macrotoolkit.render import get_cmdstan_version
from macrotoolkit.run import compute_run_id, run
from specs.schema.base import RunSpec

SEED = 20260813  # fixed seed for reproducibility of any randomized inputs below


def _base_spec() -> RunSpec:
    d = {
        "model": {"family": "local_level", "options": {}},
        "data": {"file": "data.csv", "date_column": "date", "mapping": {"y": "obs"}},
    }
    return RunSpec.model_validate(d)


# --- compute_run_id: determinism + sensitivity -----------------------------


def test_compute_run_id_deterministic_for_identical_inputs() -> None:
    spec = _base_spec()
    full1, h1 = compute_run_id(spec, "data_hash_a", "stan_hash_a", "2.36.0")
    full2, h2 = compute_run_id(spec, "data_hash_a", "stan_hash_a", "2.36.0")
    assert full1 == full2
    assert h1 == h2
    assert len(h1) == 12
    assert len(full1) == 64


def test_compute_run_id_changes_with_spec() -> None:
    spec_a = _base_spec()
    d_b = {
        "model": {"family": "local_level", "options": {}},
        "data": {"file": "data.csv", "date_column": "date", "mapping": {"y": "obs"}},
        "sampler": {"seed": 1},
    }
    spec_b = RunSpec.model_validate(d_b)
    full_a, _ = compute_run_id(spec_a, "data_hash", "stan_hash", "2.36.0")
    full_b, _ = compute_run_id(spec_b, "data_hash", "stan_hash", "2.36.0")
    assert full_a != full_b


def test_compute_run_id_changes_with_data_hash() -> None:
    spec = _base_spec()
    full_a, _ = compute_run_id(spec, "data_hash_a", "stan_hash", "2.36.0")
    full_b, _ = compute_run_id(spec, "data_hash_b", "stan_hash", "2.36.0")
    assert full_a != full_b


def test_compute_run_id_changes_with_stan_hash() -> None:
    spec = _base_spec()
    full_a, _ = compute_run_id(spec, "data_hash", "stan_hash_a", "2.36.0")
    full_b, _ = compute_run_id(spec, "data_hash", "stan_hash_b", "2.36.0")
    assert full_a != full_b


def test_compute_run_id_changes_with_cmdstan_version() -> None:
    spec = _base_spec()
    full_a, _ = compute_run_id(spec, "data_hash", "stan_hash", "2.36.0")
    full_b, _ = compute_run_id(spec, "data_hash", "stan_hash", "2.35.0")
    assert full_a != full_b


def test_run_id_hash12_is_prefix_of_full_digest() -> None:
    spec = _base_spec()
    full, h12 = compute_run_id(spec, "data_hash", "stan_hash", "2.36.0")
    assert full.startswith(h12)


# --- immutability contract, via real run() calls ---------------------------


REQUIRED_ARTIFACTS = ("spec.yaml", "data.snapshot.csv", "draws.nc", "diagnostics.json", "log.txt", "_SUCCESS")


def test_fresh_run_creates_success_and_all_artifacts(tiny_spec_path: Path, runs_root: Path) -> None:
    result = run(tiny_spec_path, runs_root=runs_root)
    assert result.is_new is True
    assert result.run_dir == runs_root / result.run_id
    assert result.run_dir.is_dir()
    for name in REQUIRED_ARTIFACTS:
        assert (result.run_dir / name).exists(), f"missing artifact: {name}"
    assert result.verdict in {"PASS", "WARN", "FAIL"}


def test_rerun_identical_spec_is_idempotent_noop(tiny_spec_path: Path, runs_root: Path) -> None:
    result1 = run(tiny_spec_path, runs_root=runs_root)
    assert result1.is_new is True

    # Snapshot mtimes and content of every artifact before the second run.
    before = {
        name: ((result1.run_dir / name).stat().st_mtime, (result1.run_dir / name).read_bytes())
        for name in REQUIRED_ARTIFACTS
    }

    time.sleep(0.05)  # so an accidental rewrite would be detectable via mtime
    result2 = run(tiny_spec_path, runs_root=runs_root)

    assert result2.is_new is False
    assert result2.run_id == result1.run_id
    assert result2.run_dir == result1.run_dir
    # `run()` returns no verdict on the idempotent-no-op path (it didn't
    # recompute diagnostics); this documents that behavior explicitly.
    assert result2.verdict is None

    for name in REQUIRED_ARTIFACTS:
        mtime_before, content_before = before[name]
        path = result1.run_dir / name
        assert path.stat().st_mtime == mtime_before, f"{name} mtime changed on idempotent re-run"
        assert path.read_bytes() == content_before, f"{name} content changed on idempotent re-run"


def test_changing_spec_produces_different_run_dir(tiny_spec_path: Path, runs_root: Path, tmp_path: Path) -> None:
    result1 = run(tiny_spec_path, runs_root=runs_root)

    # Change one byte of the spec: a different sampler seed.
    text = tiny_spec_path.read_text()
    assert "seed: 20260813" in text
    other_spec_path = tmp_path / "spec_changed_seed.yaml"
    other_spec_path.write_text(text.replace("seed: 20260813", "seed: 20260814"))

    result2 = run(other_spec_path, runs_root=runs_root)
    assert result2.is_new is True
    assert result2.run_id != result1.run_id
    assert result2.run_dir != result1.run_dir
    assert result1.run_dir.is_dir()  # original run untouched


def test_changing_data_byte_produces_different_run_dir(
    tmp_path: Path, tiny_spec_path: Path, runs_root: Path
) -> None:
    result1 = run(tiny_spec_path, runs_root=runs_root)

    # tiny_spec_path's data file lives alongside it; find and mutate it by
    # one byte via the spec's data.file field.
    import yaml

    spec_dict = yaml.safe_load(tiny_spec_path.read_text())
    data_path = tiny_spec_path.parent / spec_dict["data"]["file"]
    original_bytes = data_path.read_bytes()

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_spec_path = other_dir / "spec.yaml"
    other_data_path = other_dir / spec_dict["data"]["file"]
    other_spec_path.write_text(tiny_spec_path.read_text())
    # Flip a byte in the CSV (append a distinguishing trailing newline).
    other_data_path.write_bytes(original_bytes + b"\n")

    result2 = run(other_spec_path, runs_root=runs_root)
    assert result2.is_new is True
    assert result2.run_id != result1.run_id


def test_partial_run_dir_without_success_marker_raises(tiny_spec_path: Path, runs_root: Path) -> None:
    """Simulate a crashed/interrupted prior run: a run directory exists at
    the expected hash but has no `_SUCCESS` marker. `run()` must raise
    rather than silently overwrite it."""
    spec = RunSpec.model_validate(__import__("yaml").safe_load(tiny_spec_path.read_text()))
    from macrotoolkit.data import load_data
    from macrotoolkit.render import render_stan_source, stan_source_hash
    from specs.schema import get_family

    family = get_family(spec.model.family)
    _, raw_hash, _ = load_data(spec, tiny_spec_path)
    source = render_stan_source(family.template, {})
    src_hash = stan_source_hash(source)
    version = get_cmdstan_version()
    _, run_id = compute_run_id(spec, raw_hash, src_hash, version)

    partial_dir = runs_root / run_id
    partial_dir.mkdir(parents=True)
    (partial_dir / "log.txt").write_text("crashed mid-run\n")
    # deliberately no _SUCCESS marker

    with pytest.raises(RuntimeError, match="_SUCCESS"):
        run(tiny_spec_path, runs_root=runs_root)

    # The partial directory must not have been silently completed/overwritten.
    assert not (partial_dir / "_SUCCESS").exists()
    assert (partial_dir / "log.txt").read_text() == "crashed mid-run\n"
