"""The ``lw_sv`` family's OUTPUT MODULE declaration (S6 WP1) -- the one
place that says which figures an lw_sv report / notebook shows, in what
order, with what headings and captions. Registered on
``specs.schema.FAMILY_REGISTRY`` as ``output_modules``; consumed by the
generic report assembler (:mod:`macrotoolkit.report`) and by
:class:`macrotoolkit.api.Outputs`.

Pure wiring: every ``compute`` is one of ``results_lw``'s validated
``compute_*_draws`` entry points and every ``plot`` one of ``plots.py``'s
figure functions. The headings, figure titles, captions and figure order
below reproduce the S4/S5 report layout exactly (spec §3.5: diagnostics
-- with the spec §4 prior-predictive figure grouped under it -- then
§3.1-§3.4, then the parameter table), so the lw_sv report is unchanged by
the generalization (``tests/test_report.py`` pins its image count and the
volatility-panel captions).
"""
from __future__ import annotations

from macrotoolkit.outputs import OutputModule


def _prior_predictive_compute(lw_run):
    from macrotoolkit.results_lw import compute_prior_predictive_draws

    return compute_prior_predictive_draws(lw_run)


def _prior_predictive_plot(ppd, lw_run):
    from macrotoolkit.plots import plot_prior_predictive

    return plot_prior_predictive(ppd)


def _prior_predictive_caption(ppd, lw_run) -> str:
    return (
        f"{ppd.n_draws} full observable paths simulated from the run's own "
        f"resolved priors (defaults + spec overrides) through the same "
        f"matrices/engine the run used -- 'what do my priors imply about "
        f"observable paths', S5-decisions item 7. Needs no posterior draws."
    )


def _trend_cycle_compute(lw_run):
    from macrotoolkit.results_lw import compute_trend_cycle_draws

    return compute_trend_cycle_draws(lw_run)


def _trend_cycle_plot(tcd, lw_run):
    from macrotoolkit.plots import plot_trend_cycle

    return plot_trend_cycle(tcd, lw_run)


def _trend_cycle_caption(tcd, lw_run) -> str:
    return (
        "Includes the shock volatility panel (exp(h/2)) -- this is an SV run (sv_shocks: [is, pc])."
        if lw_run.sv_on
        else "No volatility panel -- this is a no-SV run (sv_shocks: []); constant IS/PC shock variances."
    )


def _irf_compute(lw_run):
    from macrotoolkit.results_lw import compute_irf_draws

    return compute_irf_draws(lw_run)


def _irf_plot(irf, lw_run):
    from macrotoolkit.plots import plot_irf_matrix

    return plot_irf_matrix(irf, lw_run.spec.outputs.irf_vol_reference)


def _irf_caption(irf, lw_run) -> str:
    return (
        f"5 shocks x 5 responses, horizon={irf.horizon} quarters, "
        f"irf_vol_reference={lw_run.spec.outputs.irf_vol_reference!r}."
    )


def _fan_compute(lw_run):
    from macrotoolkit.results_lw import compute_fan_draws

    return compute_fan_draws(lw_run)


def _fan_plot(fans, lw_run):
    from macrotoolkit.plots import plot_fan_charts

    return plot_fan_charts(fans, lw_run.spec.outputs.forecast_r_rule)


def _fan_caption(fans, lw_run) -> str:
    return (
        f"Horizon={fans.horizon} quarters, forecast_r_rule={lw_run.spec.outputs.forecast_r_rule!r} "
        f"(also printed on each chart)."
    )


def _hd_compute(lw_run):
    from macrotoolkit.results_lw import compute_historical_decomposition_draws

    return compute_historical_decomposition_draws(lw_run)


def _hd_plot(hdd, lw_run):
    from macrotoolkit.plots import plot_historical_decomposition

    return plot_historical_decomposition(hdd)


OUTPUT_MODULES: tuple[OutputModule, ...] = (
    OutputModule(
        name="prior_predictive",
        heading="Diagnostics: prior-predictive check (spec §4)",
        figure_title="Prior-predictive check (spec §4)",
        compute=_prior_predictive_compute,
        plot=_prior_predictive_plot,
        caption=_prior_predictive_caption,
    ),
    OutputModule(
        name="trend_cycle",
        heading="3.1 Trend-cycle plots",
        figure_title="Trend-cycle decomposition (spec §3.1)",
        compute=_trend_cycle_compute,
        plot=_trend_cycle_plot,
        caption=_trend_cycle_caption,
    ),
    OutputModule(
        name="irf",
        heading="3.2 IRF matrix",
        figure_title="IRF matrix (spec §3.2)",
        compute=_irf_compute,
        plot=_irf_plot,
        caption=_irf_caption,
        dpi=90,
    ),
    OutputModule(
        name="fan",
        heading="3.3 Fan charts",
        figure_title="Fan chart",
        compute=_fan_compute,
        plot=_fan_plot,
        caption=_fan_caption,
        figure_order=("y_level", "y_growth_4q", "pi", "gap", "rstar"),
    ),
    OutputModule(
        name="hd",
        heading="3.4 Historical decomposition",
        figure_title="Historical decomposition",
        compute=_hd_compute,
        plot=_hd_plot,
        figure_order=("gap", "pi", "y_growth_4q", "y_level"),
    ),
)
