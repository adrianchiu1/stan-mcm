"""Coverage for ``macrotoolkit.plots`` -- the matplotlib plotting grammar
for spec §3.1-§3.4's four output modules.

This is visualization code, not model numerics (no G-gate governs it), so
tests focus on: (1) each function runs without error against a real, tiny,
completed ``lw_sv`` run's output objects, (2) return types/shapes are
right (``Figure`` objects, correct subplot-grid dimensions), (3) edge cases
(a no-SV run must SKIP the volatility panel rather than erroring on
``None``), and (4) at least one rendered PNG per function is saved under
``tests/artifacts/plots/`` (gitignored, regenerable -- see
``tests/conftest.py``'s isolation-rule docstring and the repo's
``.gitignore`` entry for ``tests/artifacts/``) so a human can eyeball the
output.

Reuses the session-scoped ``s4_no_sv_lw_run``/``s4_sv_lw_run`` fixtures
(``tests/conftest.py``) -- shared with ``tests/test_irf.py``/
``tests/test_fan_charts.py``, so the underlying MCMC sampling only ever
runs once per pytest session. To keep this file's own smoother-rerun cost
low (every ``compute_*_draws`` call re-runs the DK simulation smoother once
per selected posterior draw), every call below thins to a small, fixed
number of draws via ``ThinSpec`` -- correctness of the aggregation itself
is already pinned by ``tests/test_results_lw.py``/``tests/test_irf.py``/
``tests/test_fan_charts.py``; this file only needs ENOUGH draws to exercise
percentile/band code paths without degenerate (single-draw) percentiles.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # headless test environment -- no display needed

from matplotlib.figure import Figure

from macrotoolkit.plots import (
    plot_fan_charts,
    plot_historical_decomposition,
    plot_irf_matrix,
    plot_trend_cycle,
)
from macrotoolkit.results_lw import (
    IRF_RESPONSES,
    IRF_SHOCKS,
    compute_fan_draws,
    compute_historical_decomposition_draws,
    compute_irf_draws,
    compute_trend_cycle_draws,
)
from specs.schema.lw_sv import ThinSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "tests" / "artifacts" / "plots"

_THIN = 10  # every 10th posterior draw -- fast but > 1 draw for percentiles


def _thin_run(lw_run, thin: int = _THIN):
    """Return a shallow copy of ``lw_run`` with ``outputs.smoother_draws``
    overridden to ``ThinSpec(thin=thin)`` -- mirrors
    ``tests/test_results_lw.py``'s / ``tests/test_irf.py``'s own thinning
    pattern for keeping smoother-rerun cost low in a plumbing test."""
    return dataclasses.replace(
        lw_run,
        spec=lw_run.spec.model_copy(
            update={"outputs": lw_run.spec.outputs.model_copy(update={"smoother_draws": ThinSpec(thin=thin)})}
        ),
    )


def _save(fig: Figure, name: str) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / name
    fig.savefig(path, format="png", dpi=80)
    return path


# ---------------------------------------------------------------------------
# plot_trend_cycle
# ---------------------------------------------------------------------------


def test_plot_trend_cycle_no_sv_returns_figure_and_skips_vol_panel(s4_no_sv_lw_run) -> None:
    r = _thin_run(s4_no_sv_lw_run)
    tcd = compute_trend_cycle_draws(r, seed=1)
    fig = plot_trend_cycle(tcd, r)
    try:
        assert isinstance(fig, Figure)
        # 5 panels (y, gap, r, g, z) -- no volatility panel for a no-SV run.
        assert len(fig.axes) == 5
        path = _save(fig, "trend_cycle_no_sv.png")
        assert path.is_file()
        assert path.stat().st_size > 0
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_plot_trend_cycle_sv_includes_vol_panel(s4_sv_lw_run) -> None:
    r = _thin_run(s4_sv_lw_run)
    tcd = compute_trend_cycle_draws(r, seed=2)
    assert tcd.vol_is is not None and tcd.vol_pc is not None
    fig = plot_trend_cycle(tcd, r)
    try:
        assert isinstance(fig, Figure)
        # 6 panels (y, gap, r, g, z, volatility) for an SV run.
        assert len(fig.axes) == 6
        path = _save(fig, "trend_cycle_sv.png")
        assert path.is_file() and path.stat().st_size > 0
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_plot_trend_cycle_mismatched_lw_run_raises(s4_no_sv_lw_run, s4_sv_lw_run) -> None:
    """tcd and lw_run must come from the same loaded run -- a mismatched
    pairing (different data length) must raise, not silently misalign."""
    r_no_sv = _thin_run(s4_no_sv_lw_run)
    tcd = compute_trend_cycle_draws(r_no_sv, seed=3)
    # Fabricate a length mismatch by truncating r_full on a copy.
    bad_run = dataclasses.replace(r_no_sv, r_full=r_no_sv.r_full[:-1])
    with pytest.raises(ValueError, match="length"):
        plot_trend_cycle(tcd, bad_run)


# ---------------------------------------------------------------------------
# plot_irf_matrix
# ---------------------------------------------------------------------------


def test_plot_irf_matrix_no_sv_5x5_grid(s4_no_sv_lw_run) -> None:
    r = _thin_run(s4_no_sv_lw_run)
    irf = compute_irf_draws(r)
    fig = plot_irf_matrix(irf, r.spec.outputs.irf_vol_reference)
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == len(IRF_SHOCKS) * len(IRF_RESPONSES) == 25
        path = _save(fig, "irf_matrix_no_sv.png")
        assert path.is_file() and path.stat().st_size > 0
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_plot_irf_matrix_sv_5x5_grid(s4_sv_lw_run) -> None:
    r = _thin_run(s4_sv_lw_run)
    irf = compute_irf_draws(r)
    fig = plot_irf_matrix(irf, r.spec.outputs.irf_vol_reference)
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 25
        path = _save(fig, "irf_matrix_sv.png")
        assert path.is_file() and path.stat().st_size > 0
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_fan_charts
# ---------------------------------------------------------------------------


def test_plot_fan_charts_no_sv_returns_five_figures(s4_no_sv_lw_run) -> None:
    r = _thin_run(s4_no_sv_lw_run)
    fans = compute_fan_draws(r, seed=4)
    figs = plot_fan_charts(fans, r.spec.outputs.forecast_r_rule)
    try:
        assert set(figs) == {"y_level", "y_growth_4q", "pi", "gap", "rstar"}
        for name, fig in figs.items():
            assert isinstance(fig, Figure), name
            assert len(fig.axes) == 1, name
        path = _save(figs["gap"], "fan_gap_no_sv.png")
        assert path.is_file() and path.stat().st_size > 0
    finally:
        import matplotlib.pyplot as plt

        for fig in figs.values():
            plt.close(fig)


def test_plot_fan_charts_sv_widens_with_sv_random_walk(s4_sv_lw_run) -> None:
    """Not a numerics gate, but a basic sanity check that the SV run's own
    fan-chart output (SV random walk continuing forward, spec §3.3) still
    renders correctly."""
    r = _thin_run(s4_sv_lw_run)
    fans = compute_fan_draws(r, seed=5)
    figs = plot_fan_charts(fans, r.spec.outputs.forecast_r_rule)
    try:
        assert set(figs) == {"y_level", "y_growth_4q", "pi", "gap", "rstar"}
        path = _save(figs["pi"], "fan_pi_sv.png")
        assert path.is_file() and path.stat().st_size > 0
    finally:
        import matplotlib.pyplot as plt

        for fig in figs.values():
            plt.close(fig)


def test_plot_fan_charts_prints_forecast_r_rule_convention(s4_no_sv_lw_run) -> None:
    r = _thin_run(s4_no_sv_lw_run)
    fans = compute_fan_draws(r, seed=6)
    figs = plot_fan_charts(fans, "last_value")
    try:
        texts = [t.get_text() for t in figs["gap"].texts]
        assert any("last_value" in t for t in texts)
    finally:
        import matplotlib.pyplot as plt

        for fig in figs.values():
            plt.close(fig)


# ---------------------------------------------------------------------------
# plot_historical_decomposition
# ---------------------------------------------------------------------------


def test_plot_historical_decomposition_no_sv_returns_four_figures(s4_no_sv_lw_run) -> None:
    r = _thin_run(s4_no_sv_lw_run)
    hdd = compute_historical_decomposition_draws(r, seed=7)
    figs = plot_historical_decomposition(hdd)
    try:
        assert set(figs) == {"gap", "pi", "y_growth_4q", "y_level"}
        for name, fig in figs.items():
            assert isinstance(fig, Figure), name
            assert len(fig.axes) == 1, name
        path = _save(figs["gap"], "hd_gap_no_sv.png")
        assert path.is_file() and path.stat().st_size > 0
    finally:
        import matplotlib.pyplot as plt

        for fig in figs.values():
            plt.close(fig)


def test_plot_historical_decomposition_sv_returns_four_figures(s4_sv_lw_run) -> None:
    r = _thin_run(s4_sv_lw_run)
    hdd = compute_historical_decomposition_draws(r, seed=8)
    figs = plot_historical_decomposition(hdd)
    try:
        assert set(figs) == {"gap", "pi", "y_growth_4q", "y_level"}
        path = _save(figs["pi"], "hd_pi_sv.png")
        assert path.is_file() and path.stat().st_size > 0
    finally:
        import matplotlib.pyplot as plt

        for fig in figs.values():
            plt.close(fig)


def test_plot_historical_decomposition_init_bar_is_a_line_not_stacked(s4_no_sv_lw_run) -> None:
    """The 'init' bar must be visually distinguishable (spec §3.4: 'residual
    line = initial-condition contribution') -- confirm it is rendered as a
    Line2D (dashed), not folded into the bar containers, and that its label
    is present in the legend."""
    r = _thin_run(s4_no_sv_lw_run)
    hdd = compute_historical_decomposition_draws(r, seed=9)
    figs = plot_historical_decomposition(hdd)
    try:
        ax = figs["gap"].axes[0]
        labels = [line.get_label() for line in ax.get_lines()]
        assert "Initial condition" in labels
        # bar containers should NOT include an "init"-labeled one.
        bar_labels = [c.get_label() for c in ax.containers]
        assert "Initial condition" not in bar_labels
    finally:
        import matplotlib.pyplot as plt

        for fig in figs.values():
            plt.close(fig)
