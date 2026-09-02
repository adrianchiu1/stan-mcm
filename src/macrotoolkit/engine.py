"""The ONE generic simulate/IRF/HD engine (S5-decisions item 1).

``results_lw.py`` used to hand-derive the IS/Phillips observation
recursions three separate times (historical decomposition, IRF, fan
charts) in "gap space" -- algebraic regroupings of the measurement
equation whose slot/timing conventions were where both S4 bugs lived.
This module replaces all of them with the measurement equation AS WRITTEN,
driven entirely by a family's declared metadata
(:class:`macrotoolkit.families.base.StateSpaceMeta` -- named state slots,
shock loadings, and the item-1 FEEDBACK MAP declaring which x columns are
which lags of which observables):

    xi_t = F @ xi_{t-1} + w_t                     (state recursion)
    x_t  = built from lagged observables / exogenous series
           per the feedback map                    (the closed feedback loop)
    y_t  = A' x_t + Z @ xi_t + e_t                (measurement equation)

Three public entry points, all running the SAME x-building + measurement
arithmetic:

- :func:`propagate_state_shock` -- one named state shock's own stream
  propagated through F (the state side of an HD bar or IRF row).
- :func:`observable_recursion` -- the deterministic observation recursion
  for ONE additive component (an HD bar, an IRF response): given that
  component's state path, measurement-shock injections, exogenous inputs,
  and pre-sample observable seeds, produce its observable path. Linearity
  of the whole system is what makes per-component runs sum to the total --
  the same property gate G6 checks.
- :func:`simulate_forward` -- the stochastic forward simulation (fan
  charts): fresh state/measurement noise per period, exogenous series
  resolved by a declared forecast rule, observables fed back through the
  same feedback map.

Numerical note (recorded in DECISIONS.md, 2026-09-02): computing
observables from the measurement equation directly regroups float
operations relative to the old gap-space recursions, so outputs agree to
~1e-12 (amplified rounding), not bit-for-bit; every existing tolerance
(G6's 1e-6, the fan-chart pins' 1e-10/1e-12) has orders of magnitude of
headroom. Structural exact zeros that follow from genuine sparsity (a
zero row of A, a zero state path, zero injections) remain EXACTLY zero.

Timing convention for :func:`simulate_forward` (the S4 fan-chart lessons,
now structural): the state row drawn at step t carries offset ``-1`` slots
describing period t-1, so an exogenous forecast rule that needs the
current state (e.g. lw_sv's "neutral" r := r*) is resolved AFTER that
step's state draw and used in the SAME iteration -- the rule supplies the
series' lag-1 value; deeper lags come from the register of previously
resolved values, seeded from real data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from macrotoolkit.families.base import (
    ExogLag,
    ObsLag,
    ObsLagMean,
    StateLabel,
    StateSpaceMeta,
)


def propagate_state_shock(F: np.ndarray, b: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Propagate ONE shock stream's own process noise through the state
    transition matrix ``F``, starting from the zero state vector at t=-1::

        xi[t] = F @ xi[t-1] + b * eps[t]

    ``b`` is the shock's declared (n,) state-loading vector
    (``meta.injection_vector(shock)``). Returns ``(len(eps), n)``.
    (Moved verbatim from ``results_lw._propagate_shock_through_F`` --
    identical float operations.)
    """
    T = len(eps)
    n = b.shape[0]
    xi = np.zeros((T, n))
    prev = np.zeros(n)
    for t in range(T):
        cur = F @ prev + b * eps[t]
        xi[t] = cur
        prev = cur
    return xi


class _Registers:
    """Per-series lag registers for the feedback loop: ``value(lag)`` reads
    the value ``lag`` periods back; ``push`` shifts a new current value in.
    Seeded per series with pre-sample values (default 0.0 -- the "from
    rest" convention every non-init HD bar and every IRF uses)."""

    def __init__(self, names: tuple[str, ...], depth_of, seeds: Mapping[str, Mapping[int, float]] | None) -> None:
        seeds = seeds or {}
        self._hist: dict[str, list[float]] = {}
        for name in names:
            depth = depth_of(name)
            series_seeds = seeds.get(name, {})
            unknown = set(series_seeds) - set(range(1, depth + 1))
            if unknown:
                raise ValueError(
                    f"Seeds for series {name!r} reference lag(s) {sorted(unknown)} "
                    f"outside the feedback map's depth {depth}."
                )
            self._hist[name] = [float(series_seeds.get(lag, 0.0)) for lag in range(1, depth + 1)]

    def value(self, name: str, lag: int) -> float:
        return self._hist[name][lag - 1]

    def push(self, name: str, value: float) -> None:
        hist = self._hist[name]
        if hist:
            self._hist[name] = [value] + hist[:-1]


def _build_x_row(
    meta: StateSpaceMeta,
    obs_reg: _Registers,
    exog_row: np.ndarray | None,
) -> np.ndarray:
    """One period's x vector from the feedback map: endogenous columns from
    the observable registers, exogenous columns copied from ``exog_row``
    (the caller supplies real or zero data per column; ``None`` = all
    zero)."""
    x_t = np.zeros(len(meta.feedback_map))
    for j, term in enumerate(meta.feedback_map):
        if isinstance(term, ObsLag):
            x_t[j] = obs_reg.value(term.name, term.lag)
        elif isinstance(term, ObsLagMean):
            total = 0.0
            for lag in term.lags:
                total += obs_reg.value(term.name, lag)
            x_t[j] = total / len(term.lags)
        elif exog_row is not None:  # ExogLag
            x_t[j] = exog_row[j]
    return x_t


def observable_recursion(
    A: np.ndarray,
    Z: np.ndarray,
    meta: StateSpaceMeta,
    state_path: np.ndarray,
    meas_inject: np.ndarray,
    exog_x: np.ndarray | None = None,
    obs_seeds: Mapping[str, Mapping[int, float]] | None = None,
) -> np.ndarray:
    """The deterministic observation recursion for ONE additive component:

        y_t = A' x_t + Z @ state_path[t] + meas_inject[t]

    with x_t's endogenous columns fed back from this component's OWN past
    observables (per the feedback map, seeded by ``obs_seeds`` -- pre-
    sample values, default 0.0) and its exogenous columns read from
    ``exog_x`` (a (T, k) matrix in x's own column layout; only the ExogLag
    columns are consulted; ``None`` means all-zero exogenous input -- the
    convention for every component except a data-injection bar).

    Returns the (T, m) observable path. By linearity, components computed
    this way sum to the full model's observables -- the identity gate G6
    checks per period.
    """
    T = state_path.shape[0]
    if meas_inject.shape != (T, meta.n_obs):
        raise ValueError(
            f"meas_inject must be (T, {meta.n_obs}) = ({T}, {meta.n_obs}); "
            f"got {meas_inject.shape}."
        )
    if exog_x is not None and exog_x.shape[0] != T:
        raise ValueError(
            f"exog_x has {exog_x.shape[0]} rows but state_path has T={T}."
        )
    obs_reg = _Registers(meta.obs_names, meta.obs_lag_depth, obs_seeds)

    obs = np.zeros((T, meta.n_obs))
    for t in range(T):
        x_t = _build_x_row(meta, obs_reg, exog_x[t] if exog_x is not None else None)
        y_t = A.T @ x_t + Z @ state_path[t] + meas_inject[t]
        obs[t] = y_t
        for name in meta.obs_names:
            obs_reg.push(name, float(y_t[meta.obs_index(name)]))
    return obs


# ---------------------------------------------------------------------------
# Stochastic forward simulation (fan charts)
# ---------------------------------------------------------------------------


class ExogForecastRule(Protocol):
    """Resolves an exogenous series' LAG-1 value for the period being
    simulated, given the state row drawn for that period (whose offset -1
    slots describe the previous period -- see the module docstring's
    timing convention)."""

    def resolve(self, state_row: np.ndarray) -> float: ...


@dataclass(frozen=True)
class ConstantExogRule:
    """Hold the series at a fixed value (lw_sv's ``last_value`` r rule)."""

    value: float

    def resolve(self, state_row: np.ndarray) -> float:
        return self.value


class DataPathExogRule:
    """Resolve the series from a KNOWN data path (e.g. the real in-sample
    r series for a prior-predictive simulation): stateful, returning the
    next value on each call. Valid because :func:`simulate_forward` calls
    each rule's ``resolve`` exactly once per step, in step order (part of
    the engine's contract); construct a fresh instance per simulation."""

    def __init__(self, values) -> None:
        self._values = np.asarray(values, dtype=np.float64)
        self._i = 0

    def resolve(self, state_row: np.ndarray) -> float:
        if self._i >= len(self._values):
            raise ValueError(
                f"DataPathExogRule exhausted after {len(self._values)} "
                f"values -- the simulation horizon exceeds the supplied "
                f"data path."
            )
        v = float(self._values[self._i])
        self._i += 1
        return v


class StateLinearExogRule:
    """Resolve the series' previous-period value as a linear combination of
    named state slots -- e.g. lw_sv's ``neutral`` rule r := r* with
    ``terms = {("g", -1): 1.0, ("z", -1): 1.0}`` (the offset -1 labels are
    exactly what makes this the PREVIOUS period's r*, resolvable only from
    the current step's freshly drawn state -- the S4 timing lesson)."""

    def __init__(self, meta: StateSpaceMeta, terms: Mapping[StateLabel, float]) -> None:
        self._slots = [(meta.slot(*label), coef) for label, coef in terms.items()]

    def resolve(self, state_row: np.ndarray) -> float:
        total = 0.0
        for slot, coef in self._slots:
            total += coef * state_row[slot]
        return total


class MeasurementNoise(Protocol):
    """Per-period measurement-shock draw for the forward simulation. RNG
    consumption order is part of the contract (it preserves seeded
    reproducibility with the pre-engine implementation)."""

    def step(self, rng: np.random.Generator) -> np.ndarray: ...


class ConstantMeasurementNoise:
    """Constant scales: eps_j ~ N(0, sd_j^2) each period, drawn in
    observation-row order (matches the pre-engine
    ``rng.normal(0, sqrt(sd**2))`` per shock, in the same order)."""

    def __init__(self, sds: tuple[float, ...]) -> None:
        self._sds = sds

    def step(self, rng: np.random.Generator) -> np.ndarray:
        eps = np.empty(len(self._sds))
        for j, sd in enumerate(self._sds):
            var = sd**2
            eps[j] = rng.normal(0.0, np.sqrt(var))
        return eps


class RandomWalkLogVarianceNoise:
    """SV continuation (spec §3.3's widening-bands channel): each shock's
    log-VARIANCE h continues its random walk (h += sigma_h * nu, nu fresh
    every period) from the draw's own terminal h, and eps_j ~ N(0, exp(h_j))
    -- h is log-variance, sd = exp(h/2) (spec §1.5). Draw order per period:
    all h innovations first (row order), then all eps draws (row order) --
    the pre-engine ``simulate_fan_draw``'s exact RNG consumption order.
    """

    def __init__(self, h_last: tuple[float, ...], sigma_h: tuple[float, ...]) -> None:
        if len(h_last) != len(sigma_h):
            raise ValueError(
                f"h_last and sigma_h must align per measurement shock; got "
                f"lengths {len(h_last)}, {len(sigma_h)}."
            )
        self._h = list(h_last)
        self._sigma_h = sigma_h

    def step(self, rng: np.random.Generator) -> np.ndarray:
        for j in range(len(self._h)):
            self._h[j] = self._h[j] + self._sigma_h[j] * rng.standard_normal()
        eps = np.empty(len(self._h))
        for j in range(len(self._h)):
            var = float(np.exp(self._h[j]))  # h is log-VARIANCE (spec §1.5)
            eps[j] = rng.normal(0.0, np.sqrt(var))
        return eps


def simulate_forward(
    F: np.ndarray,
    Q: np.ndarray,
    A: np.ndarray,
    Z: np.ndarray,
    meta: StateSpaceMeta,
    xi_last: np.ndarray,
    obs_seeds: Mapping[str, Mapping[int, float]],
    exog_seeds: Mapping[str, Mapping[int, float]],
    exog_rules: Mapping[str, ExogForecastRule],
    meas_noise: MeasurementNoise,
    horizon: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    """ONE stochastic forward realization, ``horizon`` periods ahead of the
    terminal state ``xi_last``. Per period, in this exact order (RNG
    consumption preserved from the pre-engine implementation):

    1. draw process noise, step the state: ``xi_s = F @ xi_prev + w``;
    2. resolve each exogenous series' lag-1 value from ``exog_rules`` given
       the fresh state row (deeper lags come from previously resolved
       values, seeded by ``exog_seeds`` = real data) -- each rule's
       ``resolve`` is called EXACTLY once per step, in step order (a
       contract stateful rules like :class:`DataPathExogRule` rely on);
    3. draw measurement shocks via ``meas_noise.step``;
    4. build x from the feedback registers, apply the measurement equation,
       push the new observables (and the resolved exogenous values) into
       their registers.

    After the loop, ONE extra noise draw + F step produces a final state
    row, so callers can align reported series defined on offset ``-1``
    slots (e.g. lw_sv's r*) to the same period indexing as the observables
    -- the 2026-09-02 alignment fix, now structural.

    Returns ``{"obs": (horizon, m), "states": (horizon + 1, n),
    "exog_resolved": {name: (horizon,)}}`` -- ``states[t]`` is the state
    drawn for forecast step t; ``states[horizon]`` is the extra post-loop
    row; ``exog_resolved[name][t]`` is the rule-resolved lag-1 value used
    at step t (lw_sv's rate-gap diagnostic derives from it).
    """
    for name in meta.exog_names:
        if meta.exog_lag_depth(name) > 0 and name not in exog_rules:
            raise ValueError(
                f"simulate_forward needs an exog_rules entry for series "
                f"{name!r} (the feedback map references its lags)."
            )

    n = F.shape[0]
    from macrotoolkit.smoother import _psd_sqrt

    sqrt_Q = _psd_sqrt(Q)
    obs_reg = _Registers(meta.obs_names, meta.obs_lag_depth, obs_seeds)
    # Exogenous registers hold PREVIOUSLY RESOLVED values only (the lag-1
    # value is always this step's fresh resolution), so entry i covers lag
    # i+2. Seeds use natural lag semantics relative to the FIRST simulated
    # period: {"r": {2: r_{T-1}}} seeds the lag-2 slot for step 0.
    exog_hist: dict[str, list[float]] = {}
    for name in exog_rules:
        depth = meta.exog_lag_depth(name)
        series_seeds = dict(exog_seeds.get(name, {}))
        unknown = set(series_seeds) - set(range(2, depth + 1))
        if unknown:
            raise ValueError(
                f"Exog seeds for series {name!r} reference lag(s) "
                f"{sorted(unknown)}; only lags 2..{depth} are seedable (the "
                f"lag-1 value is always resolved by the forecast rule)."
            )
        exog_hist[name] = [float(series_seeds.get(lag, 0.0)) for lag in range(2, depth + 1)]

    obs = np.empty((horizon, meta.n_obs))
    states = np.empty((horizon + 1, n))
    exog_resolved: dict[str, np.ndarray] = {name: np.empty(horizon) for name in exog_rules}

    xi_prev = np.asarray(xi_last, dtype=np.float64).copy()
    for t in range(horizon):
        w = sqrt_Q @ rng.standard_normal(n)
        xi_t = F @ xi_prev + w
        states[t] = xi_t

        resolved = {name: rule.resolve(xi_t) for name, rule in exog_rules.items()}
        for name, value in resolved.items():
            exog_resolved[name][t] = value

        eps = meas_noise.step(rng)

        # x's exogenous lag-1 columns read this step's freshly resolved
        # value; deeper lags read the register (prior resolutions/seeds).
        x_t = np.zeros(len(meta.feedback_map))
        for j, term in enumerate(meta.feedback_map):
            if isinstance(term, ObsLag):
                x_t[j] = obs_reg.value(term.name, term.lag)
            elif isinstance(term, ObsLagMean):
                total = 0.0
                for lag in term.lags:
                    total += obs_reg.value(term.name, lag)
                x_t[j] = total / len(term.lags)
            else:  # ExogLag
                x_t[j] = resolved[term.name] if term.lag == 1 else exog_hist[term.name][term.lag - 2]

        y_t = A.T @ x_t + Z @ xi_t + eps
        obs[t] = y_t

        for name in meta.obs_names:
            obs_reg.push(name, float(y_t[meta.obs_index(name)]))
        for name, value in resolved.items():
            hist = exog_hist[name]
            if hist:
                exog_hist[name] = [value] + hist[:-1]
        xi_prev = xi_t

    # The post-loop alignment row -- drawn AFTER every in-loop draw so the
    # in-loop noise stream (and therefore all observables at a given seed)
    # is independent of this row's existence.
    w_final = sqrt_Q @ rng.standard_normal(n)
    states[horizon] = F @ xi_prev + w_final

    return {"obs": obs, "states": states, "exog_resolved": exog_resolved}
