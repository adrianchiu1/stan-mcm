"""Per-run outputs for the ``ucsv`` family (S6 WP2) -- a thin DECLARATION
over the generic results core (:mod:`macrotoolkit.results_core`), the
Q_t-generalized DK simulation smoother, and the ONE generic simulate/IRF/HD
engine. Nothing here derives a recursion: the family contributes only

- how to load its run (which posterior variables, how the system matrices
  are built per draw -- ``sv_scalar_variance_path`` on both covariances
  for an SV draw);
- its REPORTING MAPPING in names: ``tau = state("tau", 0)``,
  ``transitory = pi - tau``, ``vol_eps = exp(h_eps/2)``,
  ``vol_eta = exp(h_eta/2)`` (h is log-VARIANCE, sd = exp(h/2));
- the IRF shock sizes (constant scales, or exp(h_ref/2) at the reference
  point for an SV draw -- lw_sv's convention, incl. the "average h, then
  convert once" rule for ``sample_mean``);
- the fan chart's noise models: BOTH log-variance random walks continue
  forward (the trend shock's through the engine's S6 ``StateNoise``
  protocol, the transitory shock's through ``MeasurementNoise``).

Output objects mirror ``results_lw``'s shapes (per-draw ``(n_draws, T)``
arrays, no banding -- ``plots_ucsv.py`` does the banding) so the same
output-module/report plumbing consumes them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from macrotoolkit.engine import (
    ConstantMeasurementNoise,
    ConstantStateNoise,
    RandomWalkLogVarianceNoise,
    RandomWalkLogVarianceStateNoise,
    simulate_forward,
)
from macrotoolkit.families.ucsv import (
    UCSV_STATE_META,
    build_ucsv_matrices,
    build_ucsv_regressors,
    default_initial_state,
    ucsv_mu_h0_anchors,
)
from macrotoolkit.results_core import (
    DrawMatrices,
    flatten_posterior,
    hd_bar_names,
    impulse_response,
    load_run_dataframe,
    observable_bars,
    select_draw_indices,
    smoother_draws,
    state_components,
)
from specs.schema.base import RunSpec

META = UCSV_STATE_META
_S_TAU = META.slot("tau", 0)
_PI_ROW = META.obs_index("pi")

#: HD bars in observable (pi) space: init, eta (trend), eps (transitory).
PI_BARS: tuple[str, ...] = hd_bar_names(META)
IRF_SHOCKS: tuple[str, ...] = META.state_shocks + META.measurement_shocks  # ("eta", "eps")
IRF_RESPONSES: tuple[str, ...] = ("pi", "tau")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class UcsvRun:
    """A loaded, completed ``ucsv`` run (the ``results_loader`` capability's
    return value): system inputs + the raw ArviZ posterior. Exposes
    ``spec``/``dates``/``idata`` (what the generic report/API read)."""

    run_dir: Path
    spec: RunSpec
    idata: object
    sv_on: bool
    yobs: np.ndarray  # (T, 1)
    x: np.ndarray  # (T, 0)
    xi00: np.ndarray
    P00: np.ndarray
    pi: np.ndarray  # (T,)
    dates: pd.DatetimeIndex


def load_ucsv_run(run_dir: str | Path) -> UcsvRun:
    import arviz as az

    from macrotoolkit.run import load_run_spec

    run_dir = Path(run_dir)
    for name in ("spec.yaml", "draws.nc"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"{run_dir / name} not found -- {run_dir} does not look like a completed run directory.")
    spec = load_run_spec(run_dir)
    if spec.model.family != "ucsv":
        raise ValueError(f"results_ucsv.load_ucsv_run only supports model.family 'ucsv'; run {run_dir} has family {spec.model.family!r}.")
    df = load_run_dataframe(run_dir, spec)
    pi = df["pi"].to_numpy(dtype=np.float64)
    yobs, x = build_ucsv_regressors(pi)
    xi00, P00 = default_initial_state(float(pi[0]))
    return UcsvRun(
        run_dir=run_dir,
        spec=spec,
        idata=az.from_netcdf(str(run_dir / "draws.nc")),
        sv_on=bool(spec.model.options.sv_shocks),
        yobs=yobs,
        x=x,
        xi00=xi00,
        P00=P00,
        pi=pi,
        dates=pd.DatetimeIndex(df["date"].to_numpy()),
    )


def posterior_variable_names(sv_on: bool) -> list[str]:
    return ["h_eps", "h_eta", "sigma_h_eps", "sigma_h_eta"] if sv_on else ["sigma_eps", "sigma_eta"]


def _flatten(run: UcsvRun) -> dict[str, np.ndarray]:
    return flatten_posterior(run.idata, posterior_variable_names(run.sv_on), run.run_dir)


def matrices_for_draw(flat, i: int, sv_on: bool) -> DrawMatrices:
    """One draw's ``(F, Q, A, Z, R)``: constant scales for a no-SV draw;
    for an SV draw the (T, 1, 1) variance paths from that draw's OWN saved
    ``h_eta`` (Q_t) and ``h_eps`` (R_t) -- never re-derived from nu."""
    from macrotoolkit.smoother import sv_scalar_variance_path

    if sv_on:
        F, _, A, Z, _ = build_ucsv_matrices({"sigma_eta": 1.0, "sigma_eps": 1.0})
        h_eps = np.asarray(flat["h_eps"][i], dtype=np.float64)
        h_eta = np.asarray(flat["h_eta"][i], dtype=np.float64)
        return DrawMatrices(
            F=F, Q=sv_scalar_variance_path(h_eta), A=A, Z=Z, R=sv_scalar_variance_path(h_eps),
            extras={"h_eps": h_eps, "h_eta": h_eta,
                    "sigma_h_eps": float(flat["sigma_h_eps"][i]), "sigma_h_eta": float(flat["sigma_h_eta"][i])},
        )
    params = {"sigma_eta": float(flat["sigma_eta"][i]), "sigma_eps": float(flat["sigma_eps"][i])}
    F, Q, A, Z, R = build_ucsv_matrices(params)
    return DrawMatrices(F=F, Q=Q, A=A, Z=Z, R=R, extras=params)


def _draw_loop(run: UcsvRun, seed: int | None):
    flat = _flatten(run)
    n_total = next(iter(flat.values())).shape[0]
    idx = select_draw_indices(n_total, run.spec.outputs.smoother_draws)
    rng = np.random.default_rng(seed if seed is not None else run.spec.sampler.seed)
    it = smoother_draws(run.yobs, run.x, run.xi00, run.P00, META, flat, idx,
                        lambda f, i: matrices_for_draw(f, i, run.sv_on), rng)
    return flat, idx, rng, it


# ---------------------------------------------------------------------------
# Trend-cycle (the §3.1 analogue)
# ---------------------------------------------------------------------------


@dataclass
class TrendCycleDraws:
    """``series[name]`` is ``(n_draws, T)`` for ``tau`` (trend inflation),
    ``transitory`` (pi - tau) and, for SV runs, ``vol_eps``/``vol_eta``
    (exp(h/2), the shock STANDARD DEVIATIONS); ``pi`` the data."""

    draw_indices: np.ndarray
    dates: pd.DatetimeIndex
    pi: np.ndarray
    series: dict[str, np.ndarray]
    sv_on: bool


def compute_trend_cycle_draws(run: UcsvRun, *, seed: int | None = None) -> TrendCycleDraws:
    _, idx, _, it = _draw_loop(run, seed)
    T = run.yobs.shape[0]
    n = len(idx)
    tau = np.empty((n, T))
    vol_eps = np.empty((n, T)) if run.sv_on else None
    vol_eta = np.empty((n, T)) if run.sv_on else None
    for j, (_, dm, sim) in enumerate(it):
        tau[j] = sim.xi_draw[:, _S_TAU]
        if run.sv_on:
            vol_eps[j] = np.exp(dm.extras["h_eps"] / 2.0)  # h is log-VARIANCE
            vol_eta[j] = np.exp(dm.extras["h_eta"] / 2.0)
    series = {"tau": tau, "transitory": run.pi[None, :] - tau}
    if run.sv_on:
        series["vol_eps"] = vol_eps
        series["vol_eta"] = vol_eta
    return TrendCycleDraws(draw_indices=idx, dates=run.dates, pi=run.pi.copy(), series=series, sv_on=run.sv_on)


# ---------------------------------------------------------------------------
# Historical decomposition (§3.4 analogue) -- G6 identity per period
# ---------------------------------------------------------------------------


@dataclass
class HDDraws:
    """``pi[bar]`` is ``(n_draws, T)`` for ``bar in PI_BARS`` (init, eta,
    eps) and sums to the observed pi per period per draw (G6);
    ``tau[bar]`` the state components (init, eta) summing to the drawn
    tau path."""

    draw_indices: np.ndarray
    dates: pd.DatetimeIndex
    pi: dict[str, np.ndarray]
    tau: dict[str, np.ndarray]


def historical_decomposition_draw(dm: DrawMatrices, xi_draw: np.ndarray, state_shocks, meas_shocks) -> tuple[dict, dict]:
    comps = state_components(dm.F, META, xi_draw, state_shocks)
    bars = observable_bars(dm.A, dm.Z, META, comps, meas_shocks)
    pi_bars = {k: bars[k][:, _PI_ROW] for k in PI_BARS}
    tau_bars = {k: comps[k][:, _S_TAU] for k in ("init", *META.state_shocks)}
    return pi_bars, tau_bars


def compute_historical_decomposition_draws(run: UcsvRun, *, seed: int | None = None) -> HDDraws:
    _, idx, _, it = _draw_loop(run, seed)
    T = run.yobs.shape[0]
    n = len(idx)
    pi = {k: np.empty((n, T)) for k in PI_BARS}
    tau = {k: np.empty((n, T)) for k in ("init", *META.state_shocks)}
    for j, (_, dm, sim) in enumerate(it):
        pb, tb = historical_decomposition_draw(dm, sim.xi_draw, sim.state_shocks, sim.meas_shocks)
        for k in PI_BARS:
            pi[k][j] = pb[k]
        for k in tau:
            tau[k][j] = tb[k]
    return HDDraws(draw_indices=idx, dates=run.dates, pi=pi, tau=tau)


# ---------------------------------------------------------------------------
# IRFs (§3.2 analogue)
# ---------------------------------------------------------------------------


@dataclass
class IRFDraws:
    draw_indices: np.ndarray
    horizon: int
    responses: dict[str, dict[str, np.ndarray]]  # shock -> response -> (n_draws, H)


def irf_shock_size(shock: str, dm: DrawMatrices, sv_on: bool, irf_vol_reference: str) -> float:
    """One-sd shock at the reference volatility: the draw's constant scale
    for a no-SV draw; exp(h_ref/2) for an SV draw with h_ref = h[-1]
    (end_of_sample) or mean(h) (sample_mean: average the log-variance,
    then convert once -- results_lw's documented convention)."""
    if not sv_on:
        return float(dm.extras[f"sigma_{shock}"])
    h = dm.extras[f"h_{shock}"]
    if irf_vol_reference == "end_of_sample":
        h_ref = float(h[-1])
    elif irf_vol_reference == "sample_mean":
        h_ref = float(np.mean(h))
    else:
        raise ValueError(f"outputs.irf_vol_reference must be 'end_of_sample' or 'sample_mean'; got {irf_vol_reference!r}.")
    return float(np.exp(h_ref / 2.0))


def compute_irf_draws(run: UcsvRun) -> IRFDraws:
    flat = _flatten(run)
    n_total = next(iter(flat.values())).shape[0]
    idx = select_draw_indices(n_total, run.spec.outputs.smoother_draws)
    H = run.spec.outputs.irf_horizon
    vol_ref = run.spec.outputs.irf_vol_reference
    responses = {s: {r: np.empty((len(idx), H)) for r in IRF_RESPONSES} for s in IRF_SHOCKS}
    for j, i in enumerate(idx):
        dm = matrices_for_draw(flat, int(i), run.sv_on)
        for shock in IRF_SHOCKS:
            comp, obs = impulse_response(dm.F, dm.A, dm.Z, META, shock, irf_shock_size(shock, dm, run.sv_on, vol_ref), H)
            responses[shock]["pi"][j] = obs[:, _PI_ROW]
            responses[shock]["tau"][j] = comp[:, _S_TAU]
    return IRFDraws(draw_indices=idx, horizon=H, responses=responses)


# ---------------------------------------------------------------------------
# Fan charts (§3.3 analogue): both SV random walks continue forward
# ---------------------------------------------------------------------------


@dataclass
class FanDraws:
    draw_indices: np.ndarray
    horizon: int
    pi: np.ndarray  # (n_draws, H)
    tau: np.ndarray  # (n_draws, H)


def _noise_models(dm: DrawMatrices, sv_on: bool, h_last: dict | None = None):
    """The forward-simulation noise models for one draw: SV continuation
    of BOTH log-variance random walks from the draw's terminal h (state
    side via RandomWalkLogVarianceStateNoise, measurement side via
    RandomWalkLogVarianceNoise), or constant scales."""
    if sv_on:
        h_eps_last = float(dm.extras["h_eps"][-1]) if h_last is None else h_last["eps"]
        h_eta_last = float(dm.extras["h_eta"][-1]) if h_last is None else h_last["eta"]
        state_noise = RandomWalkLogVarianceStateNoise(META, {"eta": h_eta_last}, {"eta": dm.extras["sigma_h_eta"]})
        meas_noise = RandomWalkLogVarianceNoise((h_eps_last,), (dm.extras["sigma_h_eps"],))
    else:
        state_noise = ConstantStateNoise(dm.Q)
        meas_noise = ConstantMeasurementNoise((float(dm.extras["sigma_eps"]),))
    return state_noise, meas_noise


def compute_fan_draws(run: UcsvRun, *, seed: int | None = None) -> FanDraws:
    _, idx, rng, it = _draw_loop(run, seed)
    H = run.spec.outputs.horizon
    n = len(idx)
    pi = np.empty((n, H))
    tau = np.empty((n, H))
    for j, (_, dm, sim) in enumerate(it):
        state_noise, meas_noise = _noise_models(dm, run.sv_on)
        Q_const = dm.Q if dm.Q.ndim == 2 else dm.Q[-1]
        out = simulate_forward(dm.F, Q_const, dm.A, dm.Z, META, sim.xi_draw[-1], {}, {}, {}, meas_noise, H, rng,
                               state_noise=state_noise)
        pi[j] = out["obs"][:, _PI_ROW]
        tau[j] = out["states"][:H, _S_TAU]
    return FanDraws(draw_indices=idx, horizon=H, pi=pi, tau=tau)


# ---------------------------------------------------------------------------
# Prior-predictive check (spec §4)
# ---------------------------------------------------------------------------


@dataclass
class PriorPredictiveDraws:
    n_draws: int
    dates: pd.DatetimeIndex
    pi: np.ndarray  # (n_draws, T)
    tau: np.ndarray  # (n_draws, T)
    pi_actual: np.ndarray


def compute_prior_predictive_draws(run: UcsvRun, *, seed: int | None = None) -> PriorPredictiveDraws:
    """Full pi paths from the run's own RESOLVED priors through the same
    matrices/engine the run used: parameters via the family prior sampler
    (data-anchored h_0 for SV), tau_0 ~ N(xi00, P00), both SV random walks
    from h_0 through the engine's noise models."""
    from macrotoolkit.families.ucsv import sample_prior_params
    from macrotoolkit.run import build_render_context
    from macrotoolkit.smoother import _psd_sqrt

    T = run.yobs.shape[0]
    n_draws = run.spec.outputs.prior_predictive_draws
    priors = build_render_context(run.spec)["priors"]
    mu_eps = mu_eta = None
    if run.sv_on:
        mu_eps, mu_eta = ucsv_mu_h0_anchors(run.pi)
    sqrt_P00 = _psd_sqrt(run.P00)
    rng = np.random.default_rng(seed if seed is not None else run.spec.sampler.seed)
    pi = np.empty((n_draws, T))
    tau = np.empty((n_draws, T))
    for j in range(n_draws):
        params = sample_prior_params(priors, run.sv_on, rng, mu_h0_eps=mu_eps, mu_h0_eta=mu_eta)
        if run.sv_on:
            F, Q, A, Z, R = build_ucsv_matrices({"sigma_eta": 1.0, "sigma_eps": 1.0})
            dm = DrawMatrices(F=F, Q=Q, A=A, Z=Z, R=R,
                              extras={"sigma_h_eps": params["sigma_h_eps"], "sigma_h_eta": params["sigma_h_eta"]})
            state_noise, meas_noise = _noise_models(dm, True, {"eps": params["h0_eps"], "eta": params["h0_eta"]})
        else:
            F, Q, A, Z, R = build_ucsv_matrices(params)
            dm = DrawMatrices(F=F, Q=Q, A=A, Z=Z, R=R, extras=dict(params))
            state_noise, meas_noise = _noise_models(dm, False)
        xi_init = run.xi00 + sqrt_P00 @ rng.standard_normal(len(run.xi00))
        out = simulate_forward(F, Q, A, Z, META, xi_init, {}, {}, {}, meas_noise, T, rng, state_noise=state_noise)
        pi[j] = out["obs"][:, _PI_ROW]
        tau[j] = out["states"][:T, _S_TAU]
    return PriorPredictiveDraws(n_draws=n_draws, dates=run.dates, pi=pi, tau=tau, pi_actual=run.pi.copy())
