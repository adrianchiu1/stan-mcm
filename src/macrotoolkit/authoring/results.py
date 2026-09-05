"""Per-run outputs for AUTHORED models (S7 M3/M4) -- a generic DECLARATION
over the results core (:mod:`macrotoolkit.results_core`), the DK
simulation smoother and the ONE generic simulate/IRF/HD engine, driven
entirely by the compiled model's :class:`StateSpaceMeta`. No recursion is
written here; the family contributes only

- how to load a run (which posterior variables: the prior table's scalar
  parameters and, per SV shock, the saved ``h_<shock>`` path -- never
  re-derived from ``nu``), and how to build one draw's ``(F, Q, A, Z, R)``
  through :meth:`CompiledModel.build_matrices`;
- the reporting mapping in NAMES: every state's head slot as its series,
  ``exp(h/2)`` volatility paths for SV shocks (h is log-VARIANCE);
- the IRF shock sizes (the draw's constant scale, or ``exp(h_ref/2)`` at
  the reference point -- the established "average h, then convert once"
  rule for ``sample_mean``);
- the fan chart's noise models (SV shocks continue their log-variance
  random walks through the S6 ``StateNoise``/``MeasurementNoise``
  protocols) and the exogenous forecast rules declared in the model
  definition -- fan charts exist only when EVERY exogenous series the
  feedback map lags has a rule (:func:`fan_unavailable_reason`);
- the prior-predictive check from the run's own resolved priors.

Output objects mirror ``results_ucsv``'s shapes (per-draw ``(n_draws, T)``
arrays keyed by NAME, no banding) so the same output-module/report
plumbing consumes them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from macrotoolkit.authoring.compile import CompiledModel, compiled_for_spec
from macrotoolkit.engine import (
    ConstantExogRule,
    DataPathExogRule,
    RandomWalkLogVarianceStateNoise,
    StateLinearExogRule,
    simulate_forward,
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
from specs.schema.authored import parse_slot_key
from specs.schema.base import RunSpec


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class AuthoredRun:
    """A loaded, completed authored run (the ``results_loader`` capability's
    return value). Exposes ``spec``/``dates``/``idata`` (what the generic
    report/API read) plus the compiled model and the system inputs."""

    run_dir: Path
    spec: RunSpec
    idata: object
    compiled: CompiledModel
    series: dict[str, np.ndarray]  # every observable/exogenous series, ALL trimmed rows
    yobs: np.ndarray  # (T, m)
    x: np.ndarray  # (T, k)
    xi00: np.ndarray
    P00: np.ndarray
    anchors: dict[str, float]
    dates: pd.DatetimeIndex  # the T estimation rows

    @property
    def sv_on(self) -> bool:
        return self.compiled.sv_on

    @property
    def meta(self):
        return self.compiled.meta

    @property
    def T(self) -> int:
        return int(self.yobs.shape[0])


def load_authored_run(run_dir: str | Path) -> AuthoredRun:
    import arviz as az

    from macrotoolkit.run import load_run_spec

    run_dir = Path(run_dir)
    for name in ("spec.yaml", "draws.nc"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"{run_dir / name} not found -- {run_dir} does not look like a completed run directory.")
    spec = load_run_spec(run_dir)
    if spec.model.family != "authored":
        raise ValueError(f"load_authored_run only supports model.family 'authored'; run {run_dir} has family {spec.model.family!r}.")
    compiled = compiled_for_spec(spec)
    df = load_run_dataframe(run_dir, spec)
    series = compiled.series_arrays(df)
    yobs, x = compiled.regressors(series)
    xi00, P00 = compiled.initial_state(series)
    L = compiled.lag_depth
    return AuthoredRun(
        run_dir=run_dir,
        spec=spec,
        idata=az.from_netcdf(str(run_dir / "draws.nc")),
        compiled=compiled,
        series=series,
        yobs=yobs,
        x=x,
        xi00=xi00,
        P00=P00,
        anchors=compiled.anchors(series),
        dates=pd.DatetimeIndex(df["date"].to_numpy()[L:]),
    )


def posterior_variable_names(compiled: CompiledModel) -> list[str]:
    return list(compiled.param_names) + [f"h_{s}" for s in compiled.sv_shocks]


def _flatten(run: AuthoredRun) -> dict[str, np.ndarray]:
    return flatten_posterior(run.idata, posterior_variable_names(run.compiled), run.run_dir)


def matrices_for_draw(compiled: CompiledModel, flat: Mapping[str, np.ndarray], i: int, T: int) -> DrawMatrices:
    """One draw's ``(F, Q, A, Z, R)`` from its OWN saved scalar parameters
    and ``h_<shock>`` paths."""
    params = {name: float(flat[name][i]) for name in compiled.param_names}
    h = {s: np.asarray(flat[f"h_{s}"][i], dtype=np.float64) for s in compiled.sv_shocks}
    F, Q, A, Z, R = compiled.build_matrices(params, h=h or None, T=T)
    extras: dict[str, Any] = dict(params)
    for s, path in h.items():
        extras[f"h_{s}"] = path
    return DrawMatrices(F=F, Q=Q, A=A, Z=Z, R=R, extras=extras)


def _draw_loop(run: AuthoredRun, seed: int | None):
    flat = _flatten(run)
    n_total = next(iter(flat.values())).shape[0]
    idx = select_draw_indices(n_total, run.spec.outputs.smoother_draws)
    rng = np.random.default_rng(seed if seed is not None else run.spec.sampler.seed)
    it = smoother_draws(run.yobs, run.x, run.xi00, run.P00, run.meta, flat, idx,
                        lambda f, i: matrices_for_draw(run.compiled, f, i, run.T), rng)
    return flat, idx, rng, it


def state_series_names(compiled: CompiledModel) -> list[str]:
    return list(compiled.structure.state_names)


def _head_slot(compiled: CompiledModel, name: str) -> int:
    return compiled.structure.head_slot(name)


def state_label(compiled: CompiledModel, name: str) -> str:
    """``g`` carried lagged reads ``g[-1]`` -- the figure stamps the offset."""
    d = compiled.structure.head_offset[name]
    return name if d == 0 else f"{name}[-{d}]"


# ---------------------------------------------------------------------------
# State paths (the trend-cycle analogue)
# ---------------------------------------------------------------------------


@dataclass
class StateDraws:
    """``states[name]`` is ``(n_draws, T)`` for every state's head slot;
    ``vol[shock]`` = ``exp(h/2)`` (the shock STANDARD DEVIATION) for SV
    shocks; ``observables[name]`` the data (T,)."""

    draw_indices: np.ndarray
    dates: pd.DatetimeIndex
    observables: dict[str, np.ndarray]
    states: dict[str, np.ndarray]
    vol: dict[str, np.ndarray]
    labels: dict[str, str]


def compute_state_draws(run: AuthoredRun, *, seed: int | None = None) -> StateDraws:
    c = run.compiled
    _, idx, _, it = _draw_loop(run, seed)
    n, T = len(idx), run.T
    states = {name: np.empty((n, T)) for name in state_series_names(c)}
    vol = {s: np.empty((n, T)) for s in c.sv_shocks}
    slots = {name: _head_slot(c, name) for name in states}
    for j, (_, dm, sim) in enumerate(it):
        for name, slot in slots.items():
            states[name][j] = sim.xi_draw[:, slot]
        for s in c.sv_shocks:
            vol[s][j] = np.exp(dm.extras[f"h_{s}"] / 2.0)  # h is log-VARIANCE
    observables = {name: run.yobs[:, i].copy() for i, name in enumerate(c.meta.obs_names)}
    return StateDraws(draw_indices=idx, dates=run.dates, observables=observables, states=states, vol=vol,
                      labels={name: state_label(c, name) for name in states})


# ---------------------------------------------------------------------------
# Historical decomposition -- the G6 identity per observable
# ---------------------------------------------------------------------------


@dataclass
class HDDraws:
    """``obs[name][bar]`` is ``(n_draws, T)`` and sums over bars to the
    observed series per period per draw (G6); ``states[name][bar]`` the
    state components (init + each state shock) summing to the drawn path."""

    draw_indices: np.ndarray
    dates: pd.DatetimeIndex
    bars: tuple[str, ...]
    obs: dict[str, dict[str, np.ndarray]]
    states: dict[str, dict[str, np.ndarray]]


def init_obs_seeds(run_or_compiled, series: Mapping[str, np.ndarray]) -> dict[str, dict[int, float]]:
    """The REAL pre-sample observable values feeding the ``init`` bar's
    feedback registers: lag k of observable ``o`` is row ``L - k``."""
    c = run_or_compiled.compiled if isinstance(run_or_compiled, AuthoredRun) else run_or_compiled
    L = c.lag_depth
    seeds: dict[str, dict[int, float]] = {}
    for o in c.meta.obs_names:
        depth = c.meta.obs_lag_depth(o)
        if depth:
            seeds[o] = {k: float(series[o][L - k]) for k in range(1, depth + 1)}
    return seeds


def historical_decomposition_draw(compiled: CompiledModel, dm: DrawMatrices, xi_draw, state_shocks, meas_shocks, x, seeds):
    comps = state_components(dm.F, compiled.meta, xi_draw, state_shocks)
    bars = observable_bars(dm.A, dm.Z, compiled.meta, comps, meas_shocks, x, seeds)
    return comps, bars


def compute_historical_decomposition_draws(run: AuthoredRun, *, seed: int | None = None) -> HDDraws:
    c = run.compiled
    meta = c.meta
    _, idx, _, it = _draw_loop(run, seed)
    n, T = len(idx), run.T
    bar_names = hd_bar_names(meta)
    obs = {o: {b: np.empty((n, T)) for b in bar_names} for o in meta.obs_names}
    state_bars = ("init", *meta.state_shocks)
    states = {s: {b: np.empty((n, T)) for b in state_bars} for s in state_series_names(c)}
    seeds = init_obs_seeds(run, run.series)
    for j, (_, dm, sim) in enumerate(it):
        comps, bars = historical_decomposition_draw(c, dm, sim.xi_draw, sim.state_shocks, sim.meas_shocks, run.x, seeds)
        for i, o in enumerate(meta.obs_names):
            for b in bar_names:
                obs[o][b][j] = bars[b][:, i]
        for s in states:
            slot = _head_slot(c, s)
            for b in state_bars:
                states[s][b][j] = comps[b][:, slot]
    return HDDraws(draw_indices=idx, dates=run.dates, bars=bar_names, obs=obs, states=states)


# ---------------------------------------------------------------------------
# IRFs
# ---------------------------------------------------------------------------


@dataclass
class IRFDraws:
    draw_indices: np.ndarray
    horizon: int
    shocks: tuple[str, ...]
    targets: tuple[str, ...]  # observables then states
    responses: dict[str, dict[str, np.ndarray]]  # shock -> target -> (n_draws, H)


def irf_shock_size(compiled: CompiledModel, shock: str, dm: DrawMatrices, irf_vol_reference: str) -> float:
    """One-sd shock: the draw's constant scale, or ``exp(h_ref/2)`` with
    ``h_ref`` = h[-1] (end_of_sample) or mean(h) (sample_mean)."""
    if shock not in compiled.sv_shocks:
        return float(dm.extras[compiled.structure.shock_scale_param[shock]])
    h = dm.extras[f"h_{shock}"]
    if irf_vol_reference == "end_of_sample":
        h_ref = float(h[-1])
    elif irf_vol_reference == "sample_mean":
        h_ref = float(np.mean(h))
    else:
        raise ValueError(f"outputs.irf_vol_reference must be 'end_of_sample' or 'sample_mean'; got {irf_vol_reference!r}.")
    return float(np.exp(h_ref / 2.0))


def compute_irf_draws(run: AuthoredRun) -> IRFDraws:
    c = run.compiled
    meta = c.meta
    flat = _flatten(run)
    n_total = next(iter(flat.values())).shape[0]
    idx = select_draw_indices(n_total, run.spec.outputs.smoother_draws)
    H = run.spec.outputs.irf_horizon
    vol_ref = run.spec.outputs.irf_vol_reference
    shocks = tuple(meta.state_shocks) + tuple(meta.measurement_shocks)
    targets = tuple(meta.obs_names) + tuple(state_series_names(c))
    responses = {s: {t: np.empty((len(idx), H)) for t in targets} for s in shocks}
    for j, i in enumerate(idx):
        dm = matrices_for_draw(c, flat, int(i), run.T)
        for s in shocks:
            comp, obs = impulse_response(dm.F, dm.A, dm.Z, meta, s, irf_shock_size(c, s, dm, vol_ref), H)
            for oi, o in enumerate(meta.obs_names):
                responses[s][o][j] = obs[:, oi]
            for st in state_series_names(c):
                responses[s][st][j] = comp[:, _head_slot(c, st)]
    return IRFDraws(draw_indices=idx, horizon=H, shocks=shocks, targets=targets, responses=responses)


# ---------------------------------------------------------------------------
# Fan charts (only with a forecast rule for every lagged exogenous series)
# ---------------------------------------------------------------------------


class MixedMeasurementNoise:
    """Per-row measurement shocks: SV rows continue their log-variance
    random walk (all h innovations first in row order, then all eps draws
    in row order -- ``RandomWalkLogVarianceNoise``'s RNG order), constant
    rows draw at a fixed sd. h is log-VARIANCE (sd = exp(h/2))."""

    def __init__(self, sds: Mapping[int, float], h_last: Mapping[int, float], sigma_h: Mapping[int, float], m: int) -> None:
        self._m = m
        self._sds = dict(sds)
        self._h = dict(h_last)
        self._sigma_h = dict(sigma_h)

    def step(self, rng: np.random.Generator) -> np.ndarray:
        for j in sorted(self._h):
            self._h[j] = self._h[j] + self._sigma_h[j] * rng.standard_normal()
        eps = np.empty(self._m)
        for j in range(self._m):
            # sd = sqrt(exp(h)) exactly as RandomWalkLogVarianceNoise writes it
            # (h is log-VARIANCE); a constant row draws at its fixed sd.
            sd = float(np.sqrt(np.exp(self._h[j]))) if j in self._h else self._sds[j]
            eps[j] = rng.normal(0.0, sd)
        return eps


def fan_unavailable_reason(compiled: CompiledModel) -> str | None:
    """``None`` when every exogenous series the feedback map lags has a
    declared forecast rule; else the reason the fan chart is omitted."""
    missing = [e for e in compiled.meta.exog_names if compiled.meta.exog_lag_depth(e) > 0 and e not in compiled.options.forecast_rules]
    if missing:
        return (
            f"fan charts need a forecast rule for every exogenous series entering the model; none declared for "
            f"{missing} (add model.options.forecast_rules[<series>] = {{rule: last_value | constant | state_linear}})."
        )
    return None


def _noise_models(compiled: CompiledModel, dm: DrawMatrices, h_last: Mapping[str, float] | None = None):
    """The forward-simulation noise models for one draw: SV shocks continue
    from the draw's terminal h (or the supplied ``h_last``), constant
    shocks at the draw's scale."""
    meta = compiled.meta
    st = compiled.structure

    def h_of(s: str) -> float:
        return float(h_last[s]) if h_last is not None else float(dm.extras[f"h_{s}"][-1])

    sv_state = {s: h_of(s) for s in meta.state_shocks if s in compiled.sv_shocks}
    sigma_h_state = {s: float(dm.extras[f"sigma_h_{s}"]) for s in sv_state}
    const_state = {s: float(dm.extras[st.shock_scale_param[s]]) for s in meta.state_shocks if s not in compiled.sv_shocks}
    state_noise = RandomWalkLogVarianceStateNoise(meta, sv_state, sigma_h_state, const_state)
    sds, hl, sh = {}, {}, {}
    for j, s in enumerate(meta.measurement_shocks):
        if s in compiled.sv_shocks:
            hl[j] = h_of(s)
            sh[j] = float(dm.extras[f"sigma_h_{s}"])
        else:
            sds[j] = float(dm.extras[st.shock_scale_param[s]])
    meas_noise = MixedMeasurementNoise(sds, hl, sh, meta.n_obs)
    return state_noise, meas_noise


def exog_rules_for(compiled: CompiledModel, series: Mapping[str, np.ndarray], t_last: int):
    """The engine's rule objects per lagged exogenous series from the
    model's ``forecast_rules``; ``last_value`` holds the series at row
    ``t_last`` (the last trimmed row)."""
    rules = {}
    for e in compiled.meta.exog_names:
        if compiled.meta.exog_lag_depth(e) == 0:
            continue
        decl = compiled.options.forecast_rules[e]
        if decl.rule == "last_value":
            rules[e] = ConstantExogRule(float(series[e][t_last]))
        elif decl.rule == "constant":
            rules[e] = ConstantExogRule(float(decl.value))
        else:
            rules[e] = StateLinearExogRule(compiled.meta, {parse_slot_key(k): float(v) for k, v in decl.terms.items()})
    return rules


def _terminal_seeds(compiled: CompiledModel, series: Mapping[str, np.ndarray]):
    """Registers seeded from the run's own last rows, with the engine's
    natural lag semantics relative to the FIRST forecast period (row
    ``n_rows``): lag k of any series is row ``n_rows - k``. Observables
    seed lags 1..depth; exogenous series seed lags 2..depth only -- the
    lag-1 value is always the forecast rule's resolution inside the loop
    (the S4 timing lesson, structural in ``simulate_forward``)."""
    meta = compiled.meta
    n_rows = len(next(iter(series.values())))
    obs_seeds = {o: {k: float(series[o][n_rows - k]) for k in range(1, meta.obs_lag_depth(o) + 1)} for o in meta.obs_names if meta.obs_lag_depth(o)}
    exog_seeds = {}
    for e in meta.exog_names:
        depth = meta.exog_lag_depth(e)
        if depth >= 2:
            exog_seeds[e] = {k: float(series[e][n_rows - k]) for k in range(2, depth + 1)}
    return obs_seeds, exog_seeds


@dataclass
class FanDraws:
    draw_indices: np.ndarray
    horizon: int
    obs: dict[str, np.ndarray]  # (n_draws, H)
    states: dict[str, np.ndarray]


def compute_fan_draws(run: AuthoredRun, *, seed: int | None = None) -> FanDraws:
    c = run.compiled
    reason = fan_unavailable_reason(c)
    if reason is not None:
        raise ValueError(reason)
    meta = c.meta
    _, idx, rng, it = _draw_loop(run, seed)
    H = run.spec.outputs.horizon
    n = len(idx)
    obs = {o: np.empty((n, H)) for o in meta.obs_names}
    states = {s: np.empty((n, H)) for s in state_series_names(c)}
    obs_seeds, exog_seeds = _terminal_seeds(c, run.series)
    n_rows = len(next(iter(run.series.values())))
    for j, (_, dm, sim) in enumerate(it):
        state_noise, meas_noise = _noise_models(c, dm)
        # The engine's Q argument is only consulted when no state_noise is
        # given; every authored simulation supplies one (SV continuation +
        # constant shocks by name), so a placeholder is passed explicitly.
        Q_const = dm.Q if dm.Q.ndim == 2 else dm.Q[-1]
        out = simulate_forward(dm.F, Q_const, dm.A, dm.Z, meta, sim.xi_draw[-1], obs_seeds, exog_seeds,
                               exog_rules_for(c, run.series, n_rows - 1), meas_noise, H, rng, state_noise=state_noise)
        for oi, o in enumerate(meta.obs_names):
            obs[o][j] = out["obs"][:, oi]
        for s in states:
            states[s][j] = out["states"][:H, _head_slot(c, s)]
    return FanDraws(draw_indices=idx, horizon=H, obs=obs, states=states)


# ---------------------------------------------------------------------------
# Prior-predictive check
# ---------------------------------------------------------------------------


@dataclass
class PriorPredictiveDraws:
    n_draws: int
    dates: pd.DatetimeIndex
    obs: dict[str, np.ndarray]  # (n_draws, T)
    obs_actual: dict[str, np.ndarray]


def compute_prior_predictive_draws(run: AuthoredRun, *, seed: int | None = None) -> PriorPredictiveDraws:
    """Observable paths from the run's own RESOLVED priors through the
    same matrices/engine the run used: parameters via the prior sampler
    (h_0 at the data anchors), ``xi_0 ~ N(xi00, P00)``, SV random walks
    from h_0 through the noise models, exogenous series at their REAL
    in-sample values (``DataPathExogRule``) with the pre-sample rows
    seeding the registers."""
    from macrotoolkit.smoother import _psd_sqrt

    c = run.compiled
    meta = c.meta
    L = c.lag_depth
    T = run.T
    n_draws = run.spec.outputs.prior_predictive_draws
    priors = c.resolve_priors(run.spec.priors)
    sqrt_P00 = _psd_sqrt(run.P00)
    rng = np.random.default_rng(seed if seed is not None else run.spec.sampler.seed)
    obs = {o: np.empty((n_draws, T)) for o in meta.obs_names}
    obs_seeds = {o: {k: float(run.series[o][L - k]) for k in range(1, meta.obs_lag_depth(o) + 1)} for o in meta.obs_names if meta.obs_lag_depth(o)}
    exog_seeds = {}
    for e in meta.exog_names:
        depth = meta.exog_lag_depth(e)
        if depth >= 2:
            exog_seeds[e] = {k: float(run.series[e][L - k]) for k in range(2, depth + 1)}
    for j in range(n_draws):
        params = c.sample_prior_params(priors, rng, run.anchors)
        h0 = {s: params[f"h0_{s}"] for s in c.sv_shocks}
        dm_params = {**params, **{f"h_{s}": np.array([h0[s]]) for s in c.sv_shocks}}
        F, Q, A, Z, R = c.build_matrices(params, h={s: np.array([h0[s]]) for s in c.sv_shocks} or None, T=1)
        dm = DrawMatrices(F=F, Q=Q if Q.ndim == 2 else Q[0], A=A, Z=Z, R=R if R.ndim == 2 else R[0], extras=dm_params)
        state_noise, meas_noise = _noise_models(c, dm, h_last=h0)
        rules = {e: DataPathExogRule(run.series[e][L - 1 : L - 1 + T]) for e in meta.exog_names if meta.exog_lag_depth(e)}
        xi_init = run.xi00 + sqrt_P00 @ rng.standard_normal(len(run.xi00))
        out = simulate_forward(F, dm.Q, A, Z, meta, xi_init, obs_seeds, exog_seeds, rules, meas_noise, T, rng, state_noise=state_noise)
        for oi, o in enumerate(meta.obs_names):
            obs[o][j] = out["obs"][:, oi]
    actual = {o: run.yobs[:, i].copy() for i, o in enumerate(meta.obs_names)}
    return PriorPredictiveDraws(n_draws=n_draws, dates=run.dates, obs=obs, obs_actual=actual)


# ---------------------------------------------------------------------------
# The G6 identity at a PRIOR point (the fast validation tier's gate)
# ---------------------------------------------------------------------------


def hd_reconstruction_error(spec: RunSpec, df, draw_index: int, *, seed_base: int = 20260923, max_attempts: int = 200) -> float:
    """G6's shape at a STATIONARY prior point (``CompiledModel.is_stationary``,
    the S6 lesson): one simulation-smoother draw on the real data ->
    generic bars -> max |sum of bars - observable| over observables and
    periods."""
    from macrotoolkit.smoother import simulate_smoother_draw, sv_rw_noncentered

    c = compiled_for_spec(spec)
    series = c.series_arrays(df)
    yobs, x = c.regressors(series)
    xi00, P00 = c.initial_state(series)
    anchors = c.anchors(series)
    priors = c.resolve_priors(spec.priors)
    T = yobs.shape[0]
    rng = np.random.default_rng(seed_base + draw_index)
    for _ in range(max_attempts):
        params = c.sample_prior_params(priors, rng, anchors)
        if c.is_stationary(params):
            break
    else:
        raise RuntimeError(f"No stationary prior draw in {max_attempts} attempts for authored model {c.name!r}.")
    h = {s: sv_rw_noncentered(params[f"h0_{s}"], params[f"sigma_h_{s}"], rng.standard_normal(T)) for s in c.sv_shocks}
    F, Q, A, Z, R = c.build_matrices(params, h=h or None, T=T)
    sim = simulate_smoother_draw(yobs, x, F, Q, A, Z, R, xi00, P00, rng, meta=c.meta)
    comps = state_components(F, c.meta, sim.xi_draw, sim.state_shocks)
    bars = observable_bars(A, Z, c.meta, comps, sim.meas_shocks, x, init_obs_seeds(c, series))
    total = sum(bars.values())
    return float(np.max(np.abs(total - yobs)))
