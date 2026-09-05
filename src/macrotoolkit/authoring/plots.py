"""Matplotlib figures for authored models' output modules (S7), reusing
``plots.py``'s house conventions (68/90% bands, the 9-percentile fan,
signed stacked HD bars) -- figures returned, never saved. Every figure
states its conventions on the figure (a lagged-carried state is labelled
with its offset; h is log-VARIANCE so volatility panels plot exp(h/2))."""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from macrotoolkit.plots import _DATA_COLOR, _ZERO_COLOR, _fan_chart_ax, _plot_band, _stacked_signed_bar

if TYPE_CHECKING:
    from macrotoolkit.authoring.results import AuthoredRun, FanDraws, HDDraws, IRFDraws, PriorPredictiveDraws, StateDraws

_PALETTE = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]


def _colors(names) -> dict[str, str]:
    out = {"init": "0.55", "exog": "tab:gray"}
    k = 0
    for n in names:
        if n not in out:
            out[n] = _PALETTE[k % len(_PALETTE)]
            k += 1
    return out


def plot_states(sd: "StateDraws", run: "AuthoredRun") -> Figure:
    """One panel per observable (data + the states that load on it are
    NOT inferred -- the observables are shown as data), one per state
    (band), and one volatility panel per SV shock."""
    name = run.compiled.name
    panels = len(sd.observables) + len(sd.states) + (1 if sd.vol else 0)
    fig, axes = plt.subplots(panels, 1, figsize=(10.0, 2.5 * panels), sharex=True, squeeze=False)
    axes = axes[:, 0]
    d = sd.dates
    i = 0
    for o, y in sd.observables.items():
        ax = axes[i]
        ax.plot(d, y, color=_DATA_COLOR, linewidth=1.0, label=f"{o} (data)")
        ax.set_title(f"Observable {o}")
        ax.legend(loc="upper left", fontsize=8)
        i += 1
    colors = _colors(sd.states)
    for s, arr in sd.states.items():
        ax = axes[i]
        _plot_band(ax, d, arr, colors[s], sd.labels[s])
        ax.axhline(0.0, color=_ZERO_COLOR, linewidth=0.8, linestyle=":")
        title = f"State {s}"
        if sd.labels[s] != s:
            title += f"  (carried as {sd.labels[s]}: the value at row t is dated t{sd.labels[s][len(s):]})"
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)
        i += 1
    if sd.vol:
        ax = axes[i]
        vcolors = _colors(sd.vol)
        for s, arr in sd.vol.items():
            _plot_band(ax, d, arr, vcolors[s], f"exp(h_{s}/2)")
        ax.set_title("Shock volatility paths (standard deviation, exp(h/2); h is log-variance)")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
    axes[-1].set_xlabel("Date")
    fig.suptitle(f"{name}: smoothed states (median, 68%/90% bands)", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


def plot_irf_matrix(irf: "IRFDraws", run: "AuthoredRun") -> Figure:
    h = np.arange(1, irf.horizon + 1)
    fig, axes = plt.subplots(len(irf.shocks), len(irf.targets), figsize=(3.0 * len(irf.targets) + 2, 2.4 * len(irf.shocks) + 1.5), sharex=True, squeeze=False)
    for i, shock in enumerate(irf.shocks):
        for j, target in enumerate(irf.targets):
            ax = axes[i][j]
            _plot_band(ax, h, irf.responses[shock][target], "tab:blue", target)
            ax.axhline(0.0, color=_ZERO_COLOR, linewidth=0.8, linestyle="--")
            if i == 0:
                ax.set_title(target, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"shock: {shock}", fontsize=9)
            if i == len(irf.shocks) - 1:
                ax.set_xlabel("Periods ahead", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(
        f"{run.compiled.name}: IRFs -- 1 s.d. shock at irf_vol_reference={run.spec.outputs.irf_vol_reference!r} (median + 68%/90% bands)",
        fontsize=11,
    )
    fig.text(0.5, 0.94, "Convention: responses from rest through the measurement equation as written (feedback via the declared lags); states shown at their carried slot.",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return fig


def plot_fan_charts(fans: "FanDraws", run: "AuthoredRun") -> dict[str, Figure]:
    h = np.arange(1, fans.horizon + 1)
    figs: dict[str, Figure] = {}
    rules = ", ".join(f"{e}: {r.rule}" for e, r in run.compiled.options.forecast_rules.items()) or "no exogenous series"
    for key, arr in list(fans.obs.items()) + list(fans.states.items()):
        fig, ax = plt.subplots(figsize=(9.0, 4.5))
        _fan_chart_ax(ax, h, arr, "tab:purple", f"{key}: forecast")
        ax.set_xlabel("Periods ahead")
        ax.legend(loc="upper left", fontsize=8)
        fig.text(0.01, 0.01, f"Convention: SV log-variance random walks continue forward; exogenous forecast rules -- {rules}.", fontsize=8, color="0.3")
        fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
        figs[key] = fig
    return figs


def plot_historical_decomposition(hdd: "HDDraws", run: "AuthoredRun") -> dict[str, Figure]:
    import macrotoolkit.plots as _p

    x = mdates.date2num(hdd.dates.to_pydatetime())
    colors = _colors(hdd.bars)
    labels = {b: {"init": "Initial condition", "exog": "Exogenous series (data)"}.get(b, f"{b} shock") for b in hdd.bars}
    figs: dict[str, Figure] = {}
    for o, bars in hdd.obs.items():
        medians = {b: np.median(bars[b], axis=0) for b in hdd.bars}
        fig, ax = plt.subplots(figsize=(11.0, 4.5))
        order = [b for b in hdd.bars if b != "init"]
        saved_colors, saved_labels = _p._BAR_COLORS, _p._BAR_LABELS
        try:
            _p._BAR_COLORS, _p._BAR_LABELS = {**saved_colors, **colors}, {**saved_labels, **labels}
            _stacked_signed_bar(ax, x, medians, order, 60.0)
        finally:
            _p._BAR_COLORS, _p._BAR_LABELS = saved_colors, saved_labels
        ax.plot(x, medians["init"], color="black", linewidth=1.4, linestyle="--", label=labels["init"])
        total = sum(medians[b] for b in medians)
        ax.plot(x, total, color="black", linewidth=1.0, alpha=0.6, label=f"reconstructed total (= {o})")
        ax.axhline(0.0, color=_ZERO_COLOR, linewidth=0.8, linestyle=":")
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_title(f"{o}: historical decomposition (posterior median; bars sum to the data -- the G6 identity)")
        ax.set_xlabel("Date")
        ax.legend(loc="upper left", fontsize=7, ncol=3)
        fig.tight_layout()
        figs[o] = fig
    return figs


def plot_prior_predictive(ppd: "PriorPredictiveDraws", run: "AuthoredRun") -> Figure:
    n = len(ppd.obs)
    fig, axes = plt.subplots(n, 1, figsize=(10.0, 3.5 * n), sharex=True, squeeze=False)
    for ax, (o, arr) in zip(axes[:, 0], ppd.obs.items()):
        _plot_band(ax, ppd.dates, arr, "tab:purple", f"prior {o}")
        for j in range(min(8, ppd.n_draws)):
            ax.plot(ppd.dates, arr[j], color="tab:purple", alpha=0.25, linewidth=0.6)
        ax.plot(ppd.dates, ppd.obs_actual[o], color=_DATA_COLOR, linewidth=1.2, label=f"{o} (data)")
        ax.set_title(f"Prior-predictive check: {ppd.n_draws} {o} paths from the run's resolved priors (data overlaid)")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
    axes[-1, 0].set_xlabel("Date")
    fig.text(0.01, 0.005, "Convention: parameters ~ resolved priors; xi_0 ~ N(xi00, P00); SV random walks from their anchored h_0; exogenous series at their real values.",
             fontsize=6.5, color="0.35")
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 1.0))
    return fig
