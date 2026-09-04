"""Matplotlib plotting grammar for the ``lw_sv`` family's four output
modules (lw-sv-spec.md §3.1-§3.4). Consumes ``macrotoolkit.results_lw``'s
output objects (``TrendCycleDraws``, ``IRFDraws``, ``FanDraws``,
``HDDrawsAggregated``, ``LWRun``) and returns matplotlib ``Figure`` objects
-- this module's job ends at producing figures; it never calls
``savefig``/``show`` itself, and it never touches ``report.py`` (a later S4
step, not yet written) or the HTML-embedding question.

Numerical conventions this module inherits from ``macrotoolkit.results_lw``
/ ``macrotoolkit.smoother`` (lw-sv-spec.md §1.1/§1.3/§1.5) -- re-touched
directly here only insofar as plotted values must not silently mislabel
them:
``g`` is ANNUALIZED (plotted as-is, no further scaling); ``pi`` is
``400 * dlog(P)`` (also plotted as-is -- it arrives already converted, by
``results_lw``'s own reporting mapping); the SV volatility paths plotted
here are ``exp(h/2)`` (the STANDARD DEVIATION, computed by
``results_lw.compute_trend_cycle_draws`` -- this module trusts, never
recomputes, that value) -- NEVER ``exp(h)`` (the variance).

Design choices (read this before touching ``report.py``, which depends on
some of these):

- **Return type: matplotlib ``Figure`` objects, not encoded strings.**
  Figures are strictly more useful to a caller (can be ``savefig``'d to any
  format, inspected in a notebook, or base64-PNG-encoded by ``report.py``
  itself with two lines of ``io.BytesIO`` + ``fig.savefig(..., format="png")``
  when it actually builds the HTML report) and this keeps this module a
  pure "spec output -> Figure" function library with no opinion about the
  eventual embedding format. ``report.py`` is expected to call
  ``fig.savefig(buf, format="png", dpi=...)`` (or ``format="svg"``) itself.
- **One function per output module** (``plot_trend_cycle``,
  ``plot_irf_matrix``, ``plot_fan_charts``, ``plot_historical_decomposition``).
  The trend-cycle and IRF-matrix functions each return ONE composite
  ``Figure`` (a natural single panel-grid / 5x5 grid respectively); the
  fan-chart and historical-decomposition functions each return a
  ``dict[str, Figure]`` keyed by series name (``"y_level"``,
  ``"y_growth_4q"``, ``"pi"``, ``"gap"``, ``"rstar"`` / ``"gap"``, ``"pi"``,
  ``"y_growth_4q"``, ``"y_level"``) -- these are 4-5 genuinely separate
  charts (different units, different x-axis meaning is the same but the
  series themselves don't share a natural panel grid the way trend-cycle's
  y/gap/r/g/z do), so one ``Figure`` each is more useful to a report that
  wants to place them independently.
- **Color palette**: matplotlib's default ``tab10`` cycle for generic
  series; a small fixed name -> color dict (``_BAR_COLORS`` below) for the
  historical-decomposition/IRF bars, kept CONSISTENT across the
  HD and (implicitly, by shock identity) any future cross-referencing so a
  reader learns "orange = trend growth shock" once.
- **Figure sizing**: module-level constants below (``_FIGSIZE_*``); nothing
  clever, just large enough to read comfortably in an HTML report.
- **Bands**: 68% credible interval = ``np.percentile`` at ``[16, 84]``,
  90% = ``[5, 95]`` (standard normal-ish "1-sigma" / "1.645-sigma" pointwise
  interval convention) -- drawn with ``fill_between``, 90% at a lighter
  alpha (0.15) underneath, 68% at a darker alpha (0.35) on top, both in the
  series' own color, so the two bands read as one shaded "cloud" instead of
  two colors.
- **Fan charts** (spec §3.3's own, different convention -- "10/20/.../90
  percentile bands", 9 percentile levels, not a generic 68/90 band): drawn
  as 8 ADJACENT ``fill_between`` bands between each pair of consecutive
  percentiles (10-20, 20-30, ..., 80-90), shaded with alpha increasing
  toward the center (darkest immediately around the median), PLUS the
  median (50th percentile) itself as a solid line -- 8 shaded bands + 1
  median line = 9 rendered elements, directly realizing the spec's "9
  percentile levels" as 9 distinct visual layers and producing a genuine
  gradient "fan" (like a Bank-of-England-style fan chart) rather than a
  single flat two-tone band.
- **Multi-panel layout**: ``plt.subplots`` grids within a single ``Figure``
  for trend-cycle (stacked rows, shared x-axis) and the IRF matrix (5x5
  grid); fan charts and historical decomposition use one ``Figure`` (one
  ``Axes``) per series, returned in a dict (see above).
- **No file I/O, no ``plt.show()``** anywhere in this module -- every
  function returns ``Figure`` objects with open, unclosed figures; the
  CALLER (a report-builder, or a test that wants an eyeball artifact) is
  responsible for saving/closing them. (Tests in this repo close their own
  figures after saving, to avoid unbounded figure accumulation across a
  session.)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from macrotoolkit.results_lw import GAP_BARS, IRF_RESPONSES, IRF_SHOCKS, PI_BARS

if TYPE_CHECKING:
    from macrotoolkit.results_lw import (
        FanDraws,
        HDDrawsAggregated,
        IRFDraws,
        LWRun,
        PriorPredictiveDraws,
        TrendCycleDraws,
    )

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

#: 68%/90% pointwise credible-interval percentile pairs (see module
#: docstring's "Bands" note). Used by trend-cycle and IRF-matrix plots.
_CI_68 = (16.0, 84.0)
_CI_90 = (5.0, 95.0)
_ALPHA_68 = 0.35
_ALPHA_90 = 0.15

#: Fan-chart percentile levels (spec §3.3: "10/20/.../90 percentile
#: bands" -- 9 levels, adjacent pairs shaded, see module docstring).
_FAN_PERCENTILES = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0)

#: Named colors for the 7 historical-decomposition/IRF bars -- kept
#: consistent everywhere a bar/shock name appears (module docstring).
_BAR_COLORS: dict[str, str] = {
    "init": "0.55",  # neutral grey -- not a structural shock
    "rdata": "tab:brown",  # exogenous real-rate data -- also not a structural shock
    "ystar": "tab:blue",
    "g": "tab:orange",
    "z": "tab:green",
    "is": "tab:red",
    "pc": "tab:purple",
}

#: Human-readable bar/shock labels for legends and IRF row titles.
_BAR_LABELS: dict[str, str] = {
    "init": "Initial condition",
    "rdata": "Real rate (data)",
    "ystar": "eps_y* (potential level)",
    "g": "eps_g (trend growth)",
    "z": "eps_z (other r*)",
    "is": "eps_IS (demand)",
    "pc": "eps_PC (supply)",
}

_FIGSIZE_TC_PANEL = (10.0, 2.4)  # (width, height-per-panel) for plot_trend_cycle
_FIGSIZE_IRF = (20.0, 20.0)
_FIGSIZE_FAN = (9.0, 4.5)
_FIGSIZE_HD = (11.0, 4.5)

_MEDIAN_COLOR = "tab:blue"
_DATA_COLOR = "black"
_ZERO_COLOR = "0.4"


def _median_and_bands(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``arr`` is ``(n_draws, T)``. Returns ``(median, lo68, hi68, lo90,
    hi90)``, each ``(T,)``, via ``np.percentile`` at ``_CI_68``/``_CI_90``
    (see module docstring's "Bands" note -- NOT reversed: ``lo`` is always
    the smaller percentile)."""
    median = np.percentile(arr, 50.0, axis=0)
    lo68, hi68 = np.percentile(arr, _CI_68, axis=0)
    lo90, hi90 = np.percentile(arr, _CI_90, axis=0)
    return median, lo68, hi68, lo90, hi90


def _plot_band(ax, x, arr: np.ndarray, color: str, label: str) -> None:
    """Plot ``arr``'s (n_draws, T) median line + 68%/90% bands on ``ax`` at
    x-positions ``x``, in ``color``."""
    median, lo68, hi68, lo90, hi90 = _median_and_bands(arr)
    ax.fill_between(x, lo90, hi90, color=color, alpha=_ALPHA_90, linewidth=0, label=f"{label} 90% CI")
    ax.fill_between(x, lo68, hi68, color=color, alpha=_ALPHA_68, linewidth=0, label=f"{label} 68% CI")
    ax.plot(x, median, color=color, linewidth=1.5, label=f"{label} median")


# ---------------------------------------------------------------------------
# spec §3.1 -- Trend-cycle plots
# ---------------------------------------------------------------------------


def plot_trend_cycle(tcd: "TrendCycleDraws", lw_run: "LWRun") -> Figure:
    """One composite figure, spec §3.1: y with y* band; gap panel beneath
    with a zero line; r with r* band; g (annualized) with band; z with
    band; (SV runs only) the ``exp(h_IS,t/2)``/``exp(h_PC,t/2)`` volatility
    paths with bands -- spec's own words, "this plot sells the SV feature".

    ``lw_run`` supplies the real ``r`` data series (``TrendCycleDraws``
    itself only carries ``y``/``pi``, per its own docstring) -- read as
    ``lw_run.r_full[4:]``, the length-T estimation-sample slice aligned to
    ``tcd.dates`` (``LWRun``'s own docstring: ``y_full``/``pi_full``/
    ``r_full`` are the FULL length-(T+4) series including the 4 pre-sample
    lag quarters; row i of ``tcd.dates`` = period i+1 = ``r_full[i+4]``).
    """
    dates = tcd.dates
    r = np.asarray(lw_run.r_full[4:], dtype=np.float64)
    if len(r) != len(dates):
        raise ValueError(
            f"plot_trend_cycle: lw_run.r_full[4:] has length {len(r)} but "
            f"tcd.dates has length {len(dates)} -- lw_run and tcd must come "
            f"from the SAME loaded run (mismatched objects passed in)."
        )

    sv_on = tcd.vol_is is not None and tcd.vol_pc is not None
    n_panels = 6 if sv_on else 5

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(_FIGSIZE_TC_PANEL[0], _FIGSIZE_TC_PANEL[1] * n_panels), sharex=True
    )

    ax = axes[0]
    ax.plot(dates, tcd.y, color=_DATA_COLOR, linewidth=1.2, label="y (data)")
    _plot_band(ax, dates, tcd.ystar, "tab:blue", "y*")
    ax.set_title("Output (y) and potential output (y*)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    ax = axes[1]
    _plot_band(ax, dates, tcd.output_gap, "tab:orange", "gap")
    ax.axhline(0.0, color=_ZERO_COLOR, linewidth=1.0, linestyle="--")
    ax.set_title("Output gap")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    ax.plot(dates, r, color=_DATA_COLOR, linewidth=1.2, label="r (data)")
    _plot_band(ax, dates, tcd.rstar, "tab:green", "r*")
    ax.set_title("Real rate (r) and natural rate (r*)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    ax = axes[3]
    _plot_band(ax, dates, tcd.g, "tab:red", "g")
    ax.set_title("Trend growth g (annualized)")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[4]
    _plot_band(ax, dates, tcd.z, "tab:purple", "z")
    ax.set_title("Other r* component z")
    ax.legend(loc="upper left", fontsize=8)

    if sv_on:
        ax = axes[5]
        _plot_band(ax, dates, tcd.vol_is, "tab:red", "exp(h_IS/2)")
        _plot_band(ax, dates, tcd.vol_pc, "tab:purple", "exp(h_PC/2)")
        ax.set_title("Shock volatility paths (standard deviation, exp(h/2))")
        ax.legend(loc="upper left", fontsize=8, ncol=2)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Trend-cycle decomposition (lw-sv-spec.md §3.1)", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


# ---------------------------------------------------------------------------
# spec §3.2 -- IRF matrix
# ---------------------------------------------------------------------------


def plot_irf_matrix(irf: "IRFDraws", irf_vol_reference: str) -> Figure:
    """The 5x5 IRF grid (spec §3.2): ``IRF_SHOCKS`` (rows) x
    ``IRF_RESPONSES`` (columns), each subplot showing the median + 68%/90%
    bands over ``irf.horizon`` quarters. ``irf_vol_reference`` (spec
    §3.2's "convention: one-standard-deviation shock at the reference
    volatility ... stamped in the subplot titles" -- stamped once, in the
    figure suptitle, plus named in each row's leftmost subplot label, since
    repeating the full sentence in all 25 subplot titles would be
    unreadable) is ``lw_run.spec.outputs.irf_vol_reference`` -- not stored
    on ``IRFDraws`` itself, so the caller must pass it explicitly.
    """
    h = np.arange(1, irf.horizon + 1)
    fig, axes = plt.subplots(len(IRF_SHOCKS), len(IRF_RESPONSES), figsize=_FIGSIZE_IRF, sharex=True)

    for i, shock in enumerate(IRF_SHOCKS):
        for j, response in enumerate(IRF_RESPONSES):
            ax = axes[i, j]
            arr = irf.responses[shock][response]
            _plot_band(ax, h, arr, _MEDIAN_COLOR, response)
            ax.axhline(0.0, color=_ZERO_COLOR, linewidth=0.8, linestyle="--")
            if i == 0:
                ax.set_title(response, fontsize=10)
            if j == 0:
                shock_label = _BAR_LABELS.get(shock, shock)
                ax.set_ylabel(f"shock: {shock_label}", fontsize=9)
            if i == len(IRF_SHOCKS) - 1:
                ax.set_xlabel("Quarters ahead", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        f"IRF matrix -- 1 s.d. shock at irf_vol_reference={irf_vol_reference!r} "
        f"(median + 68%/90% credible bands, lw-sv-spec.md §3.2)",
        fontsize=13,
    )
    # Convention stamp (pre-S5 review decision, 2026-09-02, DECISIONS.md):
    # these IRFs hold the real rate r FIXED (no policy response), so shocks
    # that move r* (eps_g, eps_z) open a permanent (r - r*) gap whose gap/pi
    # effects persist -- amplified by the near-unit-root gap AR(2) -- rather
    # than decaying to zero. That is a deliberate convention, not a bug, and
    # must be stated ON the figure so a reader is not left inferring it.
    fig.text(
        0.5, 0.955,
        "Convention: the real rate r is held FIXED (no policy response), so eps_g/eps_z shocks "
        "open a permanent (r - r*) gap -- their gap/pi responses persist by design.",
        ha="center", fontsize=10, color="0.35",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.945))
    return fig


# ---------------------------------------------------------------------------
# spec §3.3 -- Fan charts
# ---------------------------------------------------------------------------


def _fan_chart_ax(ax, h: np.ndarray, arr: np.ndarray, color: str, title: str) -> None:
    """Draw ONE fan chart (spec §3.3's 9-percentile-level convention -- see
    module docstring's "Fan charts" note) on ``ax``: 8 adjacent shaded bands
    between consecutive percentiles in ``_FAN_PERCENTILES``, shading darkest
    around the median, plus the median line itself."""
    pcts = np.percentile(arr, _FAN_PERCENTILES, axis=0)  # (9, T)
    n_bands = len(_FAN_PERCENTILES) - 1  # 8
    for k in range(n_bands):
        # Alpha peaks for the innermost band (around the median) and
        # tapers toward the outermost band -- the "fan" gradient look.
        dist_from_center = abs(k - (n_bands - 1) / 2.0)
        alpha = 0.55 - 0.32 * (dist_from_center / ((n_bands - 1) / 2.0))
        ax.fill_between(h, pcts[k], pcts[k + 1], color=color, alpha=max(alpha, 0.08), linewidth=0)
    median = pcts[_FAN_PERCENTILES.index(50.0)]
    ax.plot(h, median, color=color, linewidth=1.6, label="median")
    ax.set_title(title)


def plot_fan_charts(fans: "FanDraws", forecast_r_rule: str) -> dict[str, Figure]:
    """Spec §3.3's fan charts: 10/20/.../90 percentile bands, ``fans.horizon``
    quarters ahead, for y (level and 4-quarter growth), pi, gap, r* --
    5 separate ``Figure`` objects, keyed ``"y_level"``, ``"y_growth_4q"``,
    ``"pi"``, ``"gap"``, ``"rstar"``. ``forecast_r_rule`` (spec §3.3: "with
    the convention printed on the chart") is printed as a subtitle/
    annotation on every figure -- not stored on ``FanDraws`` itself, so the
    caller must pass it explicitly (matches ``plot_irf_matrix``'s
    ``irf_vol_reference`` parameter for the same reason).
    """
    h = np.arange(1, fans.horizon + 1)
    series = {
        "y_level": (fans.y_level, "tab:blue", "Output level (y), forecast"),
        "y_growth_4q": (fans.y_growth_4q, "tab:cyan", "4-quarter output growth, forecast"),
        "pi": (fans.pi, "tab:purple", "Inflation (pi), forecast"),
        "gap": (fans.gap, "tab:orange", "Output gap, forecast"),
        "rstar": (fans.rstar, "tab:green", "Natural rate (r*), forecast"),
    }

    figs: dict[str, Figure] = {}
    for key, (arr, color, title) in series.items():
        fig, ax = plt.subplots(figsize=_FIGSIZE_FAN)
        _fan_chart_ax(ax, h, arr, color, title)
        ax.set_xlabel("Quarters ahead")
        ax.legend(loc="upper left", fontsize=8)
        fig.text(
            0.01, 0.01, f"forecast_r_rule = {forecast_r_rule!r} (real-rate-gap convention, spec §3.3)",
            fontsize=8, color="0.3",
        )
        fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
        figs[key] = fig
    return figs


# ---------------------------------------------------------------------------
# spec §3.4 -- Historical decomposition
# ---------------------------------------------------------------------------


def _stacked_signed_bar(ax, x: np.ndarray, series: dict[str, np.ndarray], order: list[str], width: float) -> None:
    """Stacked bar chart with SIGNED stacking: positive contributions stack
    upward from a running positive baseline, negative contributions stack
    downward from a running negative baseline (so a bar with mixed-sign
    contributions across periods never overlaps itself)."""
    T = len(x)
    bottom_pos = np.zeros(T)
    bottom_neg = np.zeros(T)
    for k in order:
        vals = series[k]
        pos = np.where(vals > 0.0, vals, 0.0)
        neg = np.where(vals < 0.0, vals, 0.0)
        color = _BAR_COLORS.get(k, None)
        label = _BAR_LABELS.get(k, k)
        ax.bar(x, pos, bottom=bottom_pos, width=width, color=color, label=label)
        ax.bar(x, neg, bottom=bottom_neg, width=width, color=color)
        bottom_pos += pos
        bottom_neg += neg


def _hd_chart(dates, bar_medians: dict[str, np.ndarray], bar_order: list[str], title: str) -> Figure:
    """One historical-decomposition stacked-bar chart: posterior-median
    per-bar contributions, stacked (signed), for ``bar_order`` (excludes
    ``"init"`` -- spec §3.4: "residual line = initial-condition
    contribution" -- so ``"init"`` is drawn as its own overlaid dashed line,
    not folded into the stack), plus a solid black "reconstructed" line
    (the sum of every bar, including init -- equal to the actual smoothed/
    observed series to numerical tolerance, gate G6) so a reader can
    visually confirm the bars + init line add up.
    """
    x = mdates.date2num(dates.to_pydatetime())
    width = 60.0  # ~1 quarter in days; leaves a small visible gap between bars

    fig, ax = plt.subplots(figsize=_FIGSIZE_HD)
    _stacked_signed_bar(ax, x, bar_medians, bar_order, width)

    if "init" in bar_medians:
        ax.plot(
            x, bar_medians["init"], color="black", linewidth=1.4, linestyle="--",
            label=_BAR_LABELS["init"],
        )

    total = sum(bar_medians[k] for k in bar_medians)
    ax.plot(x, total, color="black", linewidth=1.0, linestyle="-", alpha=0.6, label="reconstructed total")

    ax.axhline(0.0, color=_ZERO_COLOR, linewidth=0.8, linestyle=":")
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=7, ncol=3)
    fig.tight_layout()
    return fig


def plot_historical_decomposition(hdd: "HDDrawsAggregated") -> dict[str, Figure]:
    """Spec §3.4's historical decomposition: stacked bar charts of
    posterior-MEDIAN per-bar contributions (``np.median`` across
    ``hdd``'s ``n_draws`` axis), for gap, pi, 4-quarter GDP growth, and
    (spec: "optional") y level -- 4 ``Figure`` objects, keyed ``"gap"``,
    ``"pi"``, ``"y_growth_4q"``, ``"y_level"``. The ``"init"`` bar is drawn
    as its own overlaid dashed line (spec §3.4: "residual line =
    initial-condition contribution"), not stacked with the structural-
    shock bars -- see :func:`_hd_chart`.
    """
    dates = hdd.dates
    figs: dict[str, Figure] = {}

    gap_medians = {k: np.median(hdd.gap[k], axis=0) for k in GAP_BARS}
    gap_order = [k for k in GAP_BARS if k != "init"]
    figs["gap"] = _hd_chart(dates, gap_medians, gap_order, "Output gap: historical decomposition (posterior median)")

    pi_medians = {k: np.median(hdd.pi[k], axis=0) for k in PI_BARS}
    pi_order = [k for k in PI_BARS if k != "init"]
    figs["pi"] = _hd_chart(dates, pi_medians, pi_order, "Inflation: historical decomposition (posterior median)")

    growth_medians = {k: np.median(hdd.y_growth_4q[k], axis=0) for k in PI_BARS}
    growth_order = [k for k in PI_BARS if k != "init"]
    # First 4 periods are NaN (four_quarter_growth's own convention,
    # results_lw.py) -- trim them from this chart rather than plotting NaN
    # bars/lines.
    valid = ~np.isnan(growth_medians[PI_BARS[0]])
    growth_dates = dates[valid]
    growth_medians_trimmed = {k: v[valid] for k, v in growth_medians.items()}
    figs["y_growth_4q"] = _hd_chart(
        growth_dates, growth_medians_trimmed, growth_order,
        "4-quarter GDP growth: historical decomposition (posterior median)",
    )

    y_medians = {k: np.median(hdd.y[k], axis=0) for k in PI_BARS}
    y_order = [k for k in PI_BARS if k != "init"]
    figs["y_level"] = _hd_chart(dates, y_medians, y_order, "Output level (y): historical decomposition (posterior median)")

    return figs


# ---------------------------------------------------------------------------
# spec §4 -- Prior-predictive check (S5-decisions item 7)
# ---------------------------------------------------------------------------

_FIGSIZE_PRIOR_PRED = (10.0, 6.5)
_N_SPAGHETTI = 8  # individual prior paths drawn faintly over the bands


def plot_prior_predictive(ppd: "PriorPredictiveDraws") -> Figure:
    """Spec §4's prior-predictive check figure: gap and inflation paths
    simulated from the run's own RESOLVED priors (defaults + overrides --
    the exact config the template stamped), with the real inflation series
    overlaid for scale. In a framework whose priors deliberately do
    identification work (the sigma_g/sigma_z pile-up controls), "what do
    my priors imply about observable paths" is core functionality, not a
    nicety (plans/S5-decisions.md item 7).

    Median + 68/90% pointwise bands across the prior draws (house
    convention), plus a few faint individual paths so the reader sees what
    single prior draws look like, not just the envelope. The figure states
    its own conventions (ENGINEERING.md: "figures state their conventions
    on the figure").
    """
    fig, axes = plt.subplots(2, 1, figsize=_FIGSIZE_PRIOR_PRED, sharex=True)
    dates = ppd.dates
    n_spag = min(_N_SPAGHETTI, ppd.n_draws)

    ax = axes[0]
    _plot_band(ax, dates, ppd.gap, "tab:orange", "prior gap")
    for j in range(n_spag):
        ax.plot(dates, ppd.gap[j], color="tab:orange", alpha=0.25, linewidth=0.6)
    ax.axhline(0.0, color=_ZERO_COLOR, linewidth=1.0, linestyle="--")
    ax.set_title("Output gap paths implied by the prior")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    _plot_band(ax, dates, ppd.pi, "tab:purple", "prior inflation")
    for j in range(n_spag):
        ax.plot(dates, ppd.pi[j], color="tab:purple", alpha=0.25, linewidth=0.6)
    ax.plot(dates, ppd.pi_actual, color=_DATA_COLOR, linewidth=1.2, label="inflation (data)")
    ax.set_title("Inflation paths implied by the prior (real data overlaid)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_xlabel("Date")

    fig.suptitle(
        f"Prior-predictive check (spec §4): {ppd.n_draws} paths from the run's resolved priors",
        fontsize=13,
    )
    fig.text(
        0.01,
        0.005,
        "Convention: parameters ~ the run's own resolved priors (defaults + spec overrides, template "
        "truncations respected); initial states ~ the run's (xi00, P00) prior; real r series supplied "
        "as the exogenous input; endogenous y/pi feedback closed through the declared feedback map.",
        fontsize=6.5,
        color="0.35",
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
    return fig
