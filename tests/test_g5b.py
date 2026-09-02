"""G5b informational exhibit (S5-decisions item 8) plumbing tests: the
published-series loader, the per-draw FILTERED series computation, the
date alignment/statistics, and the exhibit outputs. The exhibit itself is
informational -- deliberately NO tolerance gate on the differences (the
whole point of the demotion) -- so these tests pin mechanics, not values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from macrotoolkit.g5b import (
    build_exhibit,
    compute_filtered_series_draws,
    exhibit_notes_markdown,
    load_published_one_sided,
    plot_exhibit,
)


def test_published_loader_2019q2_sheet() -> None:
    pub = load_published_one_sided("2019Q2")
    assert list(pub.columns) == ["date", "output_gap", "g", "z", "rstar"]
    assert pub["date"].iloc[0] == pd.Timestamp("1961-01-01")
    assert pub["date"].iloc[-1] >= pd.Timestamp("2019-01-01")
    # r* = g + z is HLW's own identity (c estimated only in the 2023
    # model); the published one-sided columns satisfy it closely.
    resid = pub["rstar"] - (pub["g"] + pub["z"])
    assert float(np.max(np.abs(resid))) < 0.35  # published rounding + c!=1 residue
    # Magnitudes are the familiar published ones (r* within a plausible band).
    assert -1.0 < pub["rstar"].iloc[-1] < 2.5


def test_filtered_series_draws_shapes_and_are_not_the_smoothed_series(s4_sv_lw_run) -> None:
    fsd = compute_filtered_series_draws(s4_sv_lw_run, thin=10)
    T = len(s4_sv_lw_run.dates)
    n = len(fsd.draw_indices)
    assert n >= 5
    for name in ("output_gap", "g", "z", "rstar"):
        arr = getattr(fsd, name)
        assert arr.shape == (n, T), name
        assert np.all(np.isfinite(arr)), name
    # rstar = g + z by the reporting identity, exactly.
    np.testing.assert_allclose(fsd.rstar, fsd.g + fsd.z, atol=1e-12)
    # Filtered (one-sided) != smoothed (two-sided): compare against the
    # smoother-based trend-cycle machinery at the same draws -- early-
    # sample values must genuinely differ (the smoother sees the future).
    from macrotoolkit.results_lw import compute_trend_cycle_draws

    tcd = compute_trend_cycle_draws(s4_sv_lw_run, seed=1)
    assert not np.allclose(fsd.rstar[0], tcd.rstar[0], atol=1e-3)


def test_build_exhibit_aligns_and_summarizes(s4_sv_lw_run) -> None:
    ex = build_exhibit(s4_sv_lw_run, vintage="2019Q2", thin=10)
    assert len(ex.dates) >= 40  # synthetic 2001-2012 window intersects the sheet
    for name in ("rstar", "output_gap", "g", "z"):
        assert ex.ours_median[name].shape == (len(ex.dates),)
        assert ex.published[name].shape == (len(ex.dates),)
        s = ex.stats[name]
        for key in ("mean_abs_diff", "max_abs_diff", "mean_abs_diff_2000plus", "corr", "final_diff"):
            assert np.isfinite(s[key]), (name, key)
        assert s["max_abs_diff"] >= s["mean_abs_diff"] >= 0.0


def test_exhibit_notes_and_figure(s4_sv_lw_run) -> None:
    ex = build_exhibit(s4_sv_lw_run, vintage="2019Q2", thin=10)
    md = exhibit_notes_markdown(ex, "abc123def456")
    assert "no" in md.lower() and "pass/fail" in md
    for cause in ("pile-up", "MLE", "Stochastic volatility", "vintage", "Initialization"):
        assert cause in md, cause
    assert "abc123def456" in md

    import matplotlib.pyplot as plt

    fig = plot_exhibit(ex)
    assert len(fig.axes) == 4
    plt.close(fig)
