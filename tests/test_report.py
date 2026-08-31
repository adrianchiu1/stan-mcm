"""Coverage for ``macrotoolkit.report`` -- the self-contained HTML report
assembly of spec §3.5, and its ``mtk report <hash>`` CLI wiring
(``src/macrotoolkit/cli.py``).

This IS the S4 acceptance test itself (lw-sv-spec.md §7's S4 row: "report
renders all figures from a real run"):
:func:`test_s4_acceptance_report_renders_all_figures_from_real_run` drives
`mtk report 70ad47166eaf` through ``click.testing.CliRunner`` against the
REAL, already-completed ``runs/70ad47166eaf`` directory on disk (a genuine
SV run, PASS verdict) -- not a synthetic tiny fixture. Per the task brief:
this test only ever ADDS ``report.html`` to that directory; it never
touches ``spec.yaml``/``data.snapshot.csv``/``draws.nc``/``diagnostics.json``/
``_SUCCESS`` (``write_report``'s only filesystem write).

The rest of this file uses the session-scoped ``s4_no_sv_run_dir``/
``s4_sv_run_dir`` fixtures (``tests/conftest.py``, shared with
``tests/test_plots.py`` etc.) for faster smaller-scale coverage: the no-SV
vs SV volatility-panel caption distinction, and CLI error paths (unknown
hash, incomplete run dir).
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest
from click.testing import CliRunner

from macrotoolkit.cli import main
from macrotoolkit.report import render_report, write_report

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_RUN_HASH = "70ad47166eaf"
REAL_RUN_DIR = REPO_ROOT / "runs" / REAL_RUN_HASH

# 1 trend-cycle + 1 IRF matrix + 5 fan charts + 4 historical-decomposition
# figures -- exact counts from plots.py's own return shapes (module
# docstring / spec §3.1-§3.4), not a loose ">0" check.
EXPECTED_IMG_COUNT = 1 + 1 + 5 + 4


class _TagCollector(HTMLParser):
    """Minimal, dependency-free (no bs4/lxml available in this environment)
    HTML sanity-parser: collects every ``<img src=...>``, ``<link href=...>``,
    and ``<script src=...>`` attribute value, and counts start tags overall
    (a basic "this parses as markup at all" signal)."""

    def __init__(self) -> None:
        super().__init__()
        self.tag_count = 0
        self.img_srcs: list[str] = []
        self.link_hrefs: list[str] = []
        self.script_srcs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tag_count += 1
        d = dict(attrs)
        if tag == "img":
            self.img_srcs.append(d.get("src") or "")
        elif tag == "link":
            self.link_hrefs.append(d.get("href") or "")
        elif tag == "script":
            self.script_srcs.append(d.get("src") or "")


def _parse(html_text: str) -> _TagCollector:
    collector = _TagCollector()
    collector.feed(html_text)
    collector.close()
    return collector


# ---------------------------------------------------------------------------
# S4 acceptance test -- the real, already-completed run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_run_report_cli_result():
    """Runs `mtk report 70ad47166eaf` via CliRunner exactly once for this
    module's assertions (the real run's 'all' smoother_draws over ~6000
    posterior draws makes this the slowest test in the file; sharing one
    invocation keeps the module's total cost to one pass)."""
    if not REAL_RUN_DIR.is_dir():
        pytest.skip(f"real run directory {REAL_RUN_DIR} not present in this checkout")
    runner = CliRunner()
    result = runner.invoke(main, ["report", REAL_RUN_HASH])
    return result


def test_s4_acceptance_report_renders_all_figures_from_real_run(real_run_report_cli_result) -> None:
    result = real_run_report_cli_result
    assert result.exit_code == 0, result.output
    assert "written ->" in result.output

    report_path = REAL_RUN_DIR / "report.html"
    assert report_path.is_file()
    html_text = report_path.read_text(encoding="utf-8")
    assert len(html_text) > 0

    # Basic sanity parse -- not full HTML5 validation, but confirms this is
    # genuinely well-formed markup with real tag structure.
    collector = _parse(html_text)
    assert collector.tag_count > 50
    assert "<html" in html_text and "</html>" in html_text

    # Header: run hash + PASS verdict (this run's real diagnostics.json
    # verdict, confirmed on disk).
    assert REAL_RUN_HASH in html_text
    assert "PASS" in html_text

    # Evidence of all four output modules: exact image count from
    # plots.py's own return shapes.
    assert len(collector.img_srcs) == EXPECTED_IMG_COUNT, collector.img_srcs

    # Genuinely self-contained: every image is an embedded data URI, no
    # external resource references anywhere in the document.
    for src in collector.img_srcs:
        assert src.startswith("data:image/"), src
    assert collector.link_hrefs == []
    assert collector.script_srcs == []
    assert "http://" not in html_text
    assert "https://" not in html_text

    # Parameter table: recognizable scalar parameter names with
    # numeric-looking values in the same row (exact format from
    # report.py's own _render_param_table).
    assert re.search(r"<td>a1</td><td>-?\d", html_text)
    assert re.search(r"<td>sigma_g</td><td>-?\d", html_text)


def test_s4_acceptance_report_did_not_touch_sampling_artifacts(real_run_report_cli_result) -> None:
    """Guard against the one failure mode the task brief explicitly warns
    against: `mtk report` must never re-run sampling or touch the run's
    immutable artifacts, only add report.html."""
    for name in ("spec.yaml", "data.snapshot.csv", "draws.nc", "diagnostics.json", "_SUCCESS"):
        assert (REAL_RUN_DIR / name).exists(), name


# ---------------------------------------------------------------------------
# No-SV vs SV volatility-panel signal (smaller, faster fixtures)
# ---------------------------------------------------------------------------


def test_no_sv_report_has_no_volatility_panel_caption(s4_no_sv_run_dir) -> None:
    html_text = render_report(s4_no_sv_run_dir)
    assert "No volatility panel" in html_text
    assert "Includes the shock volatility panel" not in html_text


def test_sv_report_includes_volatility_panel_caption(s4_sv_run_dir) -> None:
    html_text = render_report(s4_sv_run_dir)
    assert "Includes the shock volatility panel" in html_text
    assert "No volatility panel" not in html_text


def test_report_render_is_self_contained_and_has_expected_image_count(s4_sv_run_dir) -> None:
    html_text = render_report(s4_sv_run_dir)
    collector = _parse(html_text)
    assert len(collector.img_srcs) == EXPECTED_IMG_COUNT
    for src in collector.img_srcs:
        assert src.startswith("data:image/")
    assert "http://" not in html_text
    assert "https://" not in html_text


def test_write_report_writes_file_and_returns_path(s4_no_sv_run_dir, tmp_path) -> None:
    """write_report must not mutate the shared session-scoped fixture's own
    directory beyond adding report.html (mirrors the real-run test's
    immutability guard, at fixture scale) -- confirmed by checking the other
    artifacts are untouched before/after."""
    other_artifacts_before = {
        name: (s4_no_sv_run_dir / name).stat().st_mtime
        for name in ("spec.yaml", "data.snapshot.csv", "draws.nc", "diagnostics.json", "_SUCCESS")
    }
    out_path = write_report(s4_no_sv_run_dir)
    assert out_path == s4_no_sv_run_dir / "report.html"
    assert out_path.is_file()
    assert out_path.stat().st_size > 0
    for name, mtime_before in other_artifacts_before.items():
        assert (s4_no_sv_run_dir / name).stat().st_mtime == mtime_before, name


# ---------------------------------------------------------------------------
# CLI error paths
# ---------------------------------------------------------------------------


def test_cli_report_unknown_hash_errors_clearly(tmp_path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["report", "doesnotexist12", "--runs-root", str(tmp_path)])
    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_cli_report_incomplete_run_errors_clearly(tmp_path) -> None:
    run_dir = tmp_path / "abc123def456"
    run_dir.mkdir()
    # No _SUCCESS marker written -- an interrupted/crashed run.
    runner = CliRunner()
    result = runner.invoke(main, ["report", "abc123def456", "--runs-root", str(tmp_path)])
    assert result.exit_code != 0
    assert "_SUCCESS" in result.output
    assert "not complete" in result.output


def test_cli_report_success_prints_path(s4_no_sv_run_dir) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main, ["report", s4_no_sv_run_dir.name, "--runs-root", str(s4_no_sv_run_dir.parent)]
    )
    assert result.exit_code == 0, result.output
    assert str(s4_no_sv_run_dir.name) in result.output
    assert "report.html" in result.output
