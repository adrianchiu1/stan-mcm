"""Matplotlib figures for the ``ucsv`` family's output modules (S6 WP2),
reusing ``plots.py``'s house conventions (68/90% bands, the 9-percentile
fan, signed stacked HD bars) -- figures returned, never saved. h is
log-VARIANCE: the volatility panel plots exp(h/2), the standard deviation."""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from macrotoolkit.plots import _DATA_COLOR, _ZERO_COLOR, _fan_chart_ax, _plot_band, _stacked_signed_bar

if TYPE_CHECKING:
    from macrotoolkit.results_ucsv import FanDraws, HDDraws, IRFDraws, PriorPredictiveDraws, TrendCycleDraws, UcsvRun

_BAR_COLORS = {"init": "0.55", "eta": "tab:orange", "eps": "tab:red"}
_BAR_LABELS = {"init": "Initial condition", "eta": "eta (trend shock)", "eps": "eps (transitory shock)"}


def plot_trend_cycle(tcd: "TrendCycleDraws", run: "UcsvRun") -> Figure:
    """Inflation with the trend band; the transitory component with a zero
    line; (SV) the two shock-sd paths exp(h/2)."""
    n_panels = 3 if tcd.sv_on else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(10.0, 2.6 * n_panels), sharex=True)
    d = tcd.dates
    ax = axes[0]
    ax.plot(d, tcd.pi, color=_DATA_COLOR, linewidth=1.0, label="inflation (data)")
    _plot_band(ax, d, tcd.series["tau"], "tab:blue", "trend tau")
    ax.set_title("Inflation and trend inflation (tau)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax = axes[1]
    _plot_band(ax, d, tcd.series["transitory"], "tab:orange", "pi - tau")
    ax.axhline(0.0, color=_ZERO_COLOR, linewidth=1.0, linestyle="--")
    ax.set_title("Transitory component (pi - tau)")
    ax.legend(loc="upper left", fontsize=8)
    if tcd.sv_on:
        ax = axes[2]
        _plot_band(ax, d, tcd.series["vol_eps"], "tab:red", "exp(h_eps/2)")
        _plot_band(ax, d, tcd.series["vol_eta"], "tab:purple", "exp(h_eta/2)")
        ax.set_title("Shock volatility paths (standard deviation, exp(h/2))")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
    axes[-1].set_xlabel("Date")
    fig.suptitle("UCSV trend-cycle decomposition", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


def plot_irf_matrix(irf: "IRFDraws", irf_vol_reference: str) -> Figure:
    from macrotoolkit.results_ucsv import IRF_RESPONSES, IRF_SHOCKS

    h = np.arange(1, irf.horizon + 1)
    fig, axes = plt.subplots(len(IRF_SHOCKS), len(IRF_RESPONSES), figsize=(9.0, 6.0), sharex=True, squeeze=False)
    for i, shock in enumerate(IRF_SHOCKS):
        for j, resp in enumerate(IRF_RESPONSES):
            ax = axes[i][j]
            _plot_band(ax, h, irf.responses[shock][resp], "tab:blue", resp)
            ax.axhline(0.0, color=_ZERO_COLOR, linewidth=0.8, linestyle="--")
            if i == 0:
                ax.set_title(resp, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"shock: {_BAR_LABELS.get(shock, shock)}", fontsize=9)
            if i == len(IRF_SHOCKS) - 1:
                ax.set_xlabel("Quarters ahead", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(
        f"UCSV IRFs -- 1 s.d. shock at irf_vol_reference={irf_vol_reference!r} (median + 68%/90% bands)",
        fontsize=11,
    )
    fig.text(0.5, 0.94, "Convention: a trend (eta) shock moves pi permanently one-for-one; a transitory (eps) shock lasts one period.",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return fig


def plot_fan_charts(fans: "FanDraws") -> dict[str, Figure]:
    h = np.arange(1, fans.horizon + 1)
    figs: dict[str, Figure] = {}
    for key, arr, color, title in (
        ("pi", fans.pi, "tab:purple", "Inflation (pi), forecast"),
        ("tau", fans.tau, "tab:blue", "Trend inflation (tau), forecast"),
    ):
        fig, ax = plt.subplots(figsize=(9.0, 4.5))
        _fan_chart_ax(ax, h, arr, color, title)
        ax.set_xlabel("Quarters ahead")
        ax.legend(loc="upper left", fontsize=8)
        fig.text(0.01, 0.01, "Convention: both SV log-variance random walks continue forward (widening bands are the point).",
                 fontsize=8, color="0.3")
        fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
        figs[key] = fig
    return figs


def plot_historical_decomposition(hdd: "HDDraws") -> dict[str, Figure]:
    from macrotoolkit.results_ucsv import PI_BARS

    x = mdates.date2num(hdd.dates.to_pydatetime())
    medians = {k: np.median(hdd.pi[k], axis=0) for k in PI_BARS}
    fig, ax = plt.subplots(figsize=(11.0, 4.5))
    order = [k for k in PI_BARS if k != "init"]
    # Reuse plots.py's signed stacker with this family's colors/labels.
    import macrotoolkit.plots as _p

    saved_colors, saved_labels = _p._BAR_COLORS, _p._BAR_LABELS
    try:
        _p._BAR_COLORS, _p._BAR_LABELS = {**saved_colors, **_BAR_COLORS}, {**saved_labels, **_BAR_LABELS}
        _stacked_signed_bar(ax, x, medians, order, 60.0)
    finally:
        _p._BAR_COLORS, _p._BAR_LABELS = saved_colors, saved_labels
    ax.plot(x, medians["init"], color="black", linewidth=1.4, linestyle="--", label=_BAR_LABELS["init"])
    total = sum(medians[k] for k in medians)
    ax.plot(x, total, color="black", linewidth=1.0, alpha=0.6, label="reconstructed total (= pi)")
    ax.axhline(0.0, color=_ZERO_COLOR, linewidth=0.8, linestyle=":")
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("Inflation: historical decomposition (posterior median)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=7, ncol=3)
    fig.tight_layout()
    return {"pi": fig}


def plot_prior_predictive(ppd: "PriorPredictiveDraws") -> Figure:
    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    _plot_band(ax, ppd.dates, ppd.pi, "tab:purple", "prior inflation")
    for j in range(min(8, ppd.n_draws)):
        ax.plot(ppd.dates, ppd.pi[j], color="tab:purple", alpha=0.25, linewidth=0.6)
    ax.plot(ppd.dates, ppd.pi_actual, color=_DATA_COLOR, linewidth=1.2, label="inflation (data)")
    ax.set_title(f"Prior-predictive check: {ppd.n_draws} inflation paths from the run's resolved priors (data overlaid)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_xlabel("Date")
    fig.text(0.01, 0.005, "Convention: parameters ~ resolved priors; tau_0 ~ N(pi_1, tau0_sd^2); both SV random walks from their data-anchored h_0.",
             fontsize=6.5, color="0.35")
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 1.0))
    return fig
