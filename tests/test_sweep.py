"""`mtk sweep` (S5-decisions item 9): sweep-spec validation, the
cell-spec derivation merge semantics, the contraction readout, and one
tiny real end-to-end sweep (2 lw_sv cells through the ordinary run store +
the comparison report).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from macrotoolkit.sweep import contraction, derive_cell_spec, run_sweep
from specs.schema.sweep import SweepSpec, load_sweep_spec

# ---------------------------------------------------------------------------
# Sweep-spec validation
# ---------------------------------------------------------------------------


def _valid_sweep_dict() -> dict:
    return {
        "name": "s",
        "base_spec": "spec.yaml",
        "cells": [
            {"label": "baseline", "priors": {}},
            {"label": "tight", "priors": {"sigma_g": {"sd": 0.015}}},
        ],
    }


def test_sweep_spec_valid() -> None:
    sweep = SweepSpec.model_validate(_valid_sweep_dict())
    assert [c.label for c in sweep.cells] == ["baseline", "tight"]


def test_sweep_spec_rejects_duplicate_labels() -> None:
    d = _valid_sweep_dict()
    d["cells"].append({"label": "tight", "priors": {}})
    with pytest.raises(ValueError, match="unique"):
        SweepSpec.model_validate(d)


def test_sweep_spec_rejects_empty_cells_and_bad_labels() -> None:
    d = _valid_sweep_dict()
    d["cells"] = []
    with pytest.raises(ValueError, match="at least one"):
        SweepSpec.model_validate(d)
    d2 = _valid_sweep_dict()
    d2["cells"][0]["label"] = "a/b"
    with pytest.raises(ValueError, match="path separators"):
        SweepSpec.model_validate(d2)


def test_load_sweep_spec_rejects_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "sweep.yaml"
    p.write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_sweep_spec(str(p))


def test_checked_in_sigma_g_z_sweep_file_is_valid(repo_root: Path) -> None:
    """The mandated sigma_g/sigma_z sweep (spec §1.6) ships as a valid
    sweep file: baseline + one-at-a-time halved/doubled pile-up scales."""
    sweep = load_sweep_spec(str(repo_root / "examples" / "us_lw_sv" / "sweep_sigma_g_z.yaml"))
    assert sweep.name == "sigma_g_z"
    labels = [c.label for c in sweep.cells]
    assert labels == ["baseline", "sigma_g_tight", "sigma_g_loose", "sigma_z_tight", "sigma_z_loose"]
    by_label = {c.label: c.priors for c in sweep.cells}
    assert by_label["baseline"] == {}
    assert by_label["sigma_g_tight"] == {"sigma_g": {"sd": 0.015}}
    assert by_label["sigma_z_loose"] == {"sigma_z": {"sd": 0.16}}


# ---------------------------------------------------------------------------
# Cell-spec derivation
# ---------------------------------------------------------------------------


def _lw_base_raw() -> dict:
    return {
        "model": {"family": "lw_sv", "options": {"sv_shocks": [], "estimate_c": False}},
        "data": {"file": "d.csv", "date_column": "date", "mapping": {"y": "y", "pi": "pi", "r": "r"}},
        "priors": {"sigma_z": {"sd": 0.05}},
        "outputs": {},
    }


def test_derive_cell_spec_merges_cell_priors_over_base() -> None:
    spec = derive_cell_spec(_lw_base_raw(), {"sigma_g": {"sd": 0.015}})
    assert spec.priors == {"sigma_z": {"sd": 0.05}, "sigma_g": {"sd": 0.015}}


def test_derive_cell_spec_cell_entry_replaces_base_entry_wholesale() -> None:
    spec = derive_cell_spec(_lw_base_raw(), {"sigma_z": {"sd": 0.2}})
    assert spec.priors == {"sigma_z": {"sd": 0.2}}


def test_derive_cell_spec_does_not_mutate_base_raw() -> None:
    base = _lw_base_raw()
    derive_cell_spec(base, {"sigma_g": {"sd": 0.015}})
    assert base["priors"] == {"sigma_z": {"sd": 0.05}}


def test_derive_cell_spec_outputs_do_not_change_identity() -> None:
    """Cells differ only in priors, so cells with identical priors map to
    the SAME estimation identity (item 3's split) -- documented behavior:
    such cells are idempotent aliases of one run."""
    a = derive_cell_spec(_lw_base_raw(), {})
    b = derive_cell_spec(_lw_base_raw(), {})
    assert a.to_estimation_yaml() == b.to_estimation_yaml()


# ---------------------------------------------------------------------------
# Contraction
# ---------------------------------------------------------------------------


def test_contraction_readout() -> None:
    assert contraction(1.0, 0.0) == pytest.approx(1.0)
    assert contraction(1.0, 1.0) == pytest.approx(0.0)
    assert contraction(1.0, 2.0) == pytest.approx(-3.0)
    assert np.isnan(contraction(0.0, 1.0))


# ---------------------------------------------------------------------------
# End-to-end: a tiny 2-cell lw_sv sweep through the real pipeline
# ---------------------------------------------------------------------------


def test_run_sweep_end_to_end(s4_lw_sv_data_csv: Path, tmp_path: Path) -> None:
    base = {
        "model": {"family": "lw_sv", "options": {"sv_shocks": [], "estimate_c": False}},
        "data": {
            "file": s4_lw_sv_data_csv.name,
            "date_column": "date",
            "mapping": {"y": "gdp_log100", "pi": "core_infl", "r": "real_short_rate"},
            "sample": {"start": None, "end": None},
        },
        "priors": {},
        "sampler": {"chains": 2, "warmup": 50, "sampling": 50, "adapt_delta": 0.8, "max_treedepth": 10, "seed": 20260813},
        "outputs": {},
    }
    base_path = s4_lw_sv_data_csv.parent / "sweep_base_spec.yaml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False))
    sweep_yaml = {
        "name": "tiny_test_sweep",
        "base_spec": str(base_path),
        "cells": [
            {"label": "baseline", "priors": {}},
            {"label": "sigma_g_tight", "priors": {"sigma_g": {"sd": 0.015}}},
        ],
    }
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(yaml.safe_dump(sweep_yaml, sort_keys=False))

    runs_root = tmp_path / "runs"
    sweeps_root = tmp_path / "sweeps"
    result = run_sweep(sweep_path, runs_root=runs_root, sweeps_root=sweeps_root)

    # Two distinct cells -> two distinct immutable runs in the ordinary store.
    assert len(result.cells) == 2
    ids = {c.run_result.run_id for c in result.cells}
    assert len(ids) == 2
    for c in result.cells:
        assert (c.run_result.run_dir / "_SUCCESS").is_file()

    # cells.json traceability record.
    cells_json = json.loads((result.out_dir / "cells.json").read_text())
    assert set(cells_json) == {"baseline", "sigma_g_tight"}
    assert cells_json["sigma_g_tight"]["priors_override"] == {"sigma_g": {"sd": 0.015}}

    # Comparison report: self-contained, mentions the swept parameter and
    # both cells, carries the contraction column and the headline overlay.
    html = result.report_path.read_text()
    assert "sigma_g" in html
    assert "baseline" in html and "sigma_g_tight" in html
    assert "Contraction" in html
    assert "data:image/png;base64," in html
    assert "http://" not in html and "https://" not in html

    # Idempotent re-invocation: both cells are run()-level no-ops.
    result2 = run_sweep(sweep_path, runs_root=runs_root, sweeps_root=sweeps_root)
    assert all(not c.run_result.is_new for c in result2.cells)
