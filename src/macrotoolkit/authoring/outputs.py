"""The authored family's OUTPUT MODULE declaration (S7): generic modules
wiring ``authoring.results``'s compute entry points to
``authoring.plots``'s figures, consumed by the generic report and the
Python API exactly like the hand families'. The fan module declares an
``available`` predicate: it is omitted -- with the stated reason in the
report -- unless every exogenous series has a forecast rule."""
from __future__ import annotations

from macrotoolkit.outputs import OutputModule


def _pp_compute(run):
    from macrotoolkit.authoring.results import compute_prior_predictive_draws

    return compute_prior_predictive_draws(run)


def _pp_plot(ppd, run):
    from macrotoolkit.authoring.plots import plot_prior_predictive

    return plot_prior_predictive(ppd, run)


def _states_compute(run):
    from macrotoolkit.authoring.results import compute_state_draws

    return compute_state_draws(run)


def _states_plot(sd, run):
    from macrotoolkit.authoring.plots import plot_states

    return plot_states(sd, run)


def _states_caption(sd, run) -> str:
    st = run.compiled.structure
    slots = ", ".join(f"{n}[{o}]" if o else n for n, o in st.state_labels)
    sv = f"; SV shocks {list(st.sv_shocks)} (volatility panel plots exp(h/2), h is log-variance)" if st.sv_shocks else "; no SV shocks"
    return f"State vector [{slots}]; each state shown at its carried slot{sv}."


def _irf_compute(run):
    from macrotoolkit.authoring.results import compute_irf_draws

    return compute_irf_draws(run)


def _irf_plot(irf, run):
    from macrotoolkit.authoring.plots import plot_irf_matrix

    return plot_irf_matrix(irf, run)


def _fan_compute(run):
    from macrotoolkit.authoring.results import compute_fan_draws

    return compute_fan_draws(run)


def _fan_plot(fans, run):
    from macrotoolkit.authoring.plots import plot_fan_charts

    return plot_fan_charts(fans, run)


def _fan_available(run):
    from macrotoolkit.authoring.results import fan_unavailable_reason

    return fan_unavailable_reason(run.compiled)


def _hd_compute(run):
    from macrotoolkit.authoring.results import compute_historical_decomposition_draws

    return compute_historical_decomposition_draws(run)


def _hd_plot(hdd, run):
    from macrotoolkit.authoring.plots import plot_historical_decomposition

    return plot_historical_decomposition(hdd, run)


OUTPUT_MODULES: tuple[OutputModule, ...] = (
    OutputModule(
        name="prior_predictive",
        heading="Diagnostics: prior-predictive check",
        figure_title="Prior-predictive check",
        compute=_pp_compute,
        plot=_pp_plot,
        caption=lambda ppd, run: f"{ppd.n_draws} paths of every observable simulated from the run's own resolved priors through the same matrices/engine the run used.",
    ),
    OutputModule(
        name="states",
        heading="Smoothed states",
        figure_title="Smoothed state paths",
        compute=_states_compute,
        plot=_states_plot,
        caption=_states_caption,
    ),
    OutputModule(
        name="irf",
        heading="Impulse responses",
        figure_title="IRF matrix (every shock -> every observable and state)",
        compute=_irf_compute,
        plot=_irf_plot,
        caption=lambda irf, run: f"{len(irf.shocks)} shocks x {len(irf.targets)} responses, horizon={irf.horizon}, irf_vol_reference={run.spec.outputs.irf_vol_reference!r}.",
    ),
    OutputModule(
        name="fan",
        heading="Fan charts",
        figure_title="Fan chart",
        compute=_fan_compute,
        plot=_fan_plot,
        caption=lambda fans, run: f"Horizon={fans.horizon}; SV log-variance random walks continue forward; exogenous series follow their declared forecast rules.",
        available=_fan_available,
    ),
    OutputModule(
        name="hd",
        heading="Historical decomposition",
        figure_title="Historical decomposition",
        compute=_hd_compute,
        plot=_hd_plot,
        caption=lambda hdd, run: f"Bars {list(hdd.bars)} sum to each observable per period per draw (the G6 identity).",
    ),
)
