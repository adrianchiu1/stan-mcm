"""The ``ucsv`` family's OUTPUT MODULE declaration (S6 WP2): pure wiring
of ``results_ucsv``'s compute entry points to ``plots_ucsv``'s figures,
consumed by the generic report and the Python API exactly like lw_sv's."""
from __future__ import annotations

from macrotoolkit.outputs import OutputModule


def _pp_compute(run):
    from macrotoolkit.results_ucsv import compute_prior_predictive_draws

    return compute_prior_predictive_draws(run)


def _pp_plot(ppd, run):
    from macrotoolkit.plots_ucsv import plot_prior_predictive

    return plot_prior_predictive(ppd)


def _tc_compute(run):
    from macrotoolkit.results_ucsv import compute_trend_cycle_draws

    return compute_trend_cycle_draws(run)


def _tc_plot(tcd, run):
    from macrotoolkit.plots_ucsv import plot_trend_cycle

    return plot_trend_cycle(tcd, run)


def _tc_caption(tcd, run) -> str:
    return (
        "Includes the shock volatility panel (exp(h/2)) -- this is an SV run (sv_shocks: [eps, eta])."
        if run.sv_on
        else "No volatility panel -- this is a no-SV run (sv_shocks: []); constant shock variances."
    )


def _irf_compute(run):
    from macrotoolkit.results_ucsv import compute_irf_draws

    return compute_irf_draws(run)


def _irf_plot(irf, run):
    from macrotoolkit.plots_ucsv import plot_irf_matrix

    return plot_irf_matrix(irf, run.spec.outputs.irf_vol_reference)


def _fan_compute(run):
    from macrotoolkit.results_ucsv import compute_fan_draws

    return compute_fan_draws(run)


def _fan_plot(fans, run):
    from macrotoolkit.plots_ucsv import plot_fan_charts

    return plot_fan_charts(fans)


def _hd_compute(run):
    from macrotoolkit.results_ucsv import compute_historical_decomposition_draws

    return compute_historical_decomposition_draws(run)


def _hd_plot(hdd, run):
    from macrotoolkit.plots_ucsv import plot_historical_decomposition

    return plot_historical_decomposition(hdd)


OUTPUT_MODULES: tuple[OutputModule, ...] = (
    OutputModule(
        name="prior_predictive",
        heading="Diagnostics: prior-predictive check (spec §4)",
        figure_title="Prior-predictive check",
        compute=_pp_compute,
        plot=_pp_plot,
        caption=lambda ppd, run: f"{ppd.n_draws} inflation paths simulated from the run's own resolved priors through the same matrices/engine the run used.",
    ),
    OutputModule(
        name="trend_cycle",
        heading="Trend-cycle decomposition",
        figure_title="Trend inflation, transitory component, volatility paths",
        compute=_tc_compute,
        plot=_tc_plot,
        caption=_tc_caption,
    ),
    OutputModule(
        name="irf",
        heading="Impulse responses",
        figure_title="IRFs (eta, eps -> pi, tau)",
        compute=_irf_compute,
        plot=_irf_plot,
        caption=lambda irf, run: f"2 shocks x 2 responses, horizon={irf.horizon} quarters, irf_vol_reference={run.spec.outputs.irf_vol_reference!r}.",
    ),
    OutputModule(
        name="fan",
        heading="Fan charts",
        figure_title="Fan chart",
        compute=_fan_compute,
        plot=_fan_plot,
        caption=lambda fans, run: f"Horizon={fans.horizon} quarters; both SV log-variance random walks continue forward.",
        figure_order=("pi", "tau"),
    ),
    OutputModule(
        name="hd",
        heading="Historical decomposition",
        figure_title="Historical decomposition",
        compute=_hd_compute,
        plot=_hd_plot,
        figure_order=("pi",),
    ),
)
