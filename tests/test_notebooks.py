"""The committed, executed example notebooks (S6 WP1 acceptance: the
archived reference run's output layer driven through the Python API with
no sampling) must execute cleanly from a fresh kernel. Each notebook is
copied to a tmp directory before execution so the committed file (with
its stored outputs) is never rewritten by the test."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "examples" / "notebook_api"

#: Notebooks that run WITHOUT sampling (fast-suite members). Notebooks that
#: sample (the UCSV worked example) are executed under the slow marker.
FAST_NOTEBOOKS = ["lw_sv_from_archive.ipynb"]


def _execute(path: Path, tmp_path: Path):
    import nbformat
    from nbclient import NotebookClient

    work = tmp_path / path.name
    shutil.copy(path, work)
    nb = nbformat.read(work, as_version=4)
    client = NotebookClient(
        nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}
    )
    client.execute()
    return nb


@pytest.mark.parametrize("name", FAST_NOTEBOOKS)
def test_committed_notebook_carries_outputs_and_figures(name: str) -> None:
    nb = json.loads((NOTEBOOK_DIR / name).read_text())
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, name
    executed = [c for c in code_cells if c.get("execution_count") is not None]
    assert len(executed) == len(code_cells), "the committed notebook must be fully executed"
    pngs = sum(
        1 for c in code_cells for o in c.get("outputs", []) if "image/png" in o.get("data", {})
    )
    assert pngs >= 5, "inline figures must be stored in the committed notebook"
    for c in code_cells:
        for o in c.get("outputs", []):
            assert o.get("output_type") != "error", o


@pytest.mark.parametrize("name", FAST_NOTEBOOKS)
def test_notebook_executes_from_a_fresh_kernel(name: str, tmp_path: Path) -> None:
    """Re-executes the notebook (the archived run is copied to a tmp dir
    by the notebook itself, so nothing under runs-archive/ is touched)."""
    nb = _execute(NOTEBOOK_DIR / name, tmp_path)
    for cell in nb.cells:
        if cell.cell_type == "code":
            for out in cell.get("outputs", []):
                assert out.get("output_type") != "error", out
    # Nothing was written into the tracked archive.
    assert not (REPO_ROOT / "runs-archive" / "a00958509083" / "report.html").exists()
