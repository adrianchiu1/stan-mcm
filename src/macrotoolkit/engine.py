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

S9 E4 (data-dependent loadings, plans/S9-plan.md decision 6): a
measurement loading may carry the regressors -- ``Z_t = Z0 + sum_j x_t[j]
Zx_j`` (:class:`DataLoadings`) -- so the measurement equation is BILINEAR
in the observable path and the coefficient states. The forward simulation
runs the full bilinear system (``Z_t`` from the SIMULATED regressors, the
coefficient states drawn like any state). The additive-component runs
(HD bars, IRFs) keep the G6 identity exact by treating the coefficient
path as GIVEN -- the same drawn path for every component, ``coef_path``
-- while every component feeds its OWN observables back through it:
``y^k_t = A' x^k_t + sum_j x^k_t[j] (Zx_j xi_t) + Z0 comp^k_t + e^k_t``,
which sums to ``y_t`` by induction on the (linear, time-varying)
feedback.

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
    Const,
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


@dataclass(frozen=True)
class DataLoadings:
    """S9 E4: ``Z_t = Z0 + sum_j x_t[j] * Zx[j]`` -- the constant loading
    ``Z0`` (m, n) plus one coefficient matrix per regressor column that
    carries a state x data product, in ASCENDING column order (the
    accumulation order is part of the contract: it is the order the Stan
    template prints, so :meth:`path` mirrors ``Zt`` bit for bit)."""

    Z0: np.ndarray
    Zx: tuple[tuple[int, np.ndarray], ...]

    def data_part(self, x_t: np.ndarray) -> np.ndarray:
        """``sum_j x_t[j] * Zx_j`` (m, n) -- the data-dependent part alone."""
        out = None
        for j, M in self.Zx:
            term = x_t[j] * M
            out = term if out is None else out + term
        return np.zeros_like(self.Z0) if out is None else out

    def at(self, x_t: np.ndarray) -> np.ndarray:
        """``Z_t`` for one period's regressor row."""
        Z = self.Z0
        for j, M in self.Zx:
            Z = Z + x_t[j] * M
        return Z

    def path(self, x: np.ndarray) -> np.ndarray:
        """The (T, m, n) path for a (T, k) regressor matrix, accumulated in
        column order (``Z0 + x[:, j1] Zx_j1 + x[:, j2] Zx_j2 + ...``)."""
        T = x.shape[0]
        Z = np.repeat(self.Z0[None, :, :], T, axis=0)
        for j, M in self.Zx:
            Z = Z + x[:, j, None, None] * M[None, :, :]
        return Z


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
    const_on: bool = False,
) -> np.ndarray:
    """One period's x vector from the feedback map: endogenous columns from
    the observable registers, exogenous columns copied from ``exog_row``
    (the caller supplies real or zero data per column; ``None`` = all
    zero), the ``Const`` column (S8 E1) 1 iff ``const_on`` -- never read
    from ``exog_row``, so a data-injection bar cannot double count it."""
    x_t = np.zeros(len(meta.feedback_map))
    for j, term in enumerate(meta.feedback_map):
        if isinstance(term, ObsLag):
            x_t[j] = obs_reg.value(term.name, term.lag)
        elif isinstance(term, ObsLagMean):
            total = 0.0
            for lag in term.lags:
                total += obs_reg.value(term.name, lag)
            x_t[j] = total / len(term.lags)
        elif isinstance(term, Const):
            if const_on:
                x_t[j] = 1.0
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
    const_on: bool = False,
    loadings: "DataLoadings | None" = None,
    coef_path: np.ndarray | None = None,
) -> np.ndarray:
    """The deterministic observation recursion for ONE additive component:

        y_t = A' x_t + Z @ state_path[t] + meas_inject[t]
             [+ (sum_j x_t[j] Zx_j) @ coef_path[t]         (S9 E4)]

    with x_t's endogenous columns fed back from this component's OWN past
    observables (per the feedback map, seeded by ``obs_seeds`` -- pre-
    sample values, default 0.0) and its exogenous columns read from
    ``exog_x`` (a (T, k) matrix in x's own column layout; only the ExogLag
    columns are consulted; ``None`` means all-zero exogenous input -- the
    convention for every component except a data-injection bar); the
    ``Const`` column is 1 iff ``const_on`` (S8 E1: a component is a
    DEVIATION path unless it is the intercept's own bar). With
    ``loadings`` (S9 E4) the data-dependent part of the loading multiplies
    the GIVEN coefficient path ``coef_path`` (T, n) -- the full drawn
    state path, the same for every component (module docstring) -- while
    ``Z`` is the constant part ``Z0`` applied to this component's own
    state path.

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
    if loadings is not None and (coef_path is None or coef_path.shape != (T, meta.n_state)):
        raise ValueError(
            f"observable_recursion: data-dependent loadings need the coefficient path coef_path of shape "
            f"({T}, {meta.n_state}); got {None if coef_path is None else coef_path.shape}."
        )
    obs_reg = _Registers(meta.obs_names, meta.obs_lag_depth, obs_seeds)

    obs = np.zeros((T, meta.n_obs))
    for t in range(T):
        x_t = _build_x_row(meta, obs_reg, exog_x[t] if exog_x is not None else None, const_on)
        y_t = A.T @ x_t + Z @ state_path[t] + meas_inject[t]
        if loadings is not None:
            y_t = y_t + loadings.data_part(x_t) @ coef_path[t]
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
    timing convention). S8 E2: for a series the feedback map references
    at LAG 0 (``meta.exog_contemporaneous``) the rule must return the
    CURRENT period's value instead -- ``simulate_forward`` uses the
    resolution as that period's lag-0 regressor and shifts it into the
    lag-1.. register afterwards (``last_value``/``constant`` mean the
    same thing either way; a ``state_linear`` rule still reads the
    freshly drawn state row)."""

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


class StateNoise(Protocol):
    """Per-period STATE-innovation draw for the forward simulation (S6:
    families with SV on a state shock -- UCSV's trend shock -- continue
    that random walk forward, spec §3.3's widening bands on the trend
    side). RNG consumption order is part of the contract."""

    def step(self, rng: np.random.Generator) -> np.ndarray: ...


class ConstantStateNoise:
    """Constant state innovation covariance: ``w = sqrt(Q) @ z`` with
    ``z ~ N(0, I)`` -- exactly the pre-S6 draw (same matrix square root,
    same RNG consumption), the default when no state-noise model is
    declared."""

    def __init__(self, Q: np.ndarray) -> None:
        from macrotoolkit.smoother import _psd_sqrt

        self._sqrt_Q = _psd_sqrt(np.asarray(Q, dtype=np.float64))
        self._n = self._sqrt_Q.shape[0]

    def step(self, rng: np.random.Generator) -> np.ndarray:
        return self._sqrt_Q @ rng.standard_normal(self._n)


class RandomWalkLogVarianceStateNoise:
    """State innovations built shock by shock from the family's declared
    loadings, with SV shocks continuing their log-variance random walks
    (``h += sigma_h * nu``; the shock's VARIANCE is exp(h), i.e. it is
    drawn with sd = exp(h/2) -- h is log-VARIANCE) and constant shocks at
    fixed sds. Per period, in
    ``meta.state_shocks`` order: an SV shock draws its h innovation then
    its realization; a constant shock draws its realization only.
    ``w = sum_s b_s * eps_s`` (``b_s = meta.injection_vector(s)``)."""

    def __init__(
        self,
        meta: StateSpaceMeta,
        sv_h_last: Mapping[str, float],
        sv_sigma_h: Mapping[str, float],
        constant_sds: Mapping[str, float] | None = None,
    ) -> None:
        constant_sds = dict(constant_sds or {})
        unknown = set(sv_h_last) | set(sv_sigma_h) | set(constant_sds)
        unknown -= set(meta.state_shocks)
        if unknown:
            raise ValueError(f"Unknown state shock(s) {sorted(unknown)}; declared: {list(meta.state_shocks)}.")
        if set(sv_h_last) != set(sv_sigma_h):
            raise ValueError("sv_h_last and sv_sigma_h must name the same SV state shocks.")
        missing = [s for s in meta.state_shocks if s not in sv_h_last and s not in constant_sds]
        if missing:
            raise ValueError(f"State shock(s) {missing} have neither an SV path nor a constant sd.")
        self._shocks = tuple(meta.state_shocks)
        self._b = {s: meta.injection_vector(s) for s in self._shocks}
        self._h = {s: float(v) for s, v in sv_h_last.items()}
        self._sigma_h = {s: float(v) for s, v in sv_sigma_h.items()}
        self._const = {s: float(v) for s, v in constant_sds.items()}
        self._n = meta.n_state

    def step(self, rng: np.random.Generator) -> np.ndarray:
        w = np.zeros(self._n)
        for s in self._shocks:
            if s in self._h:
                self._h[s] = self._h[s] + self._sigma_h[s] * rng.standard_normal()
                sd = float(np.exp(self._h[s] / 2.0))  # h is log-VARIANCE
            else:
                sd = self._const[s]
            w += self._b[s] * rng.normal(0.0, sd)
        return w


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
    state_noise: StateNoise | None = None,
    meas_loading: np.ndarray | None = None,
    loadings: "DataLoadings | None" = None,
) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    """ONE stochastic forward realization, ``horizon`` periods ahead of the
    terminal state ``xi_last``. Per period, in this exact order (RNG
    consumption preserved from the pre-engine implementation):

    1. draw process noise, step the state: ``xi_s = F @ xi_prev + w`` --
       from ``state_noise`` (S6: an SV state shock continues its random
       walk), default :class:`ConstantStateNoise` over ``Q`` (the exact
       pre-S6 draw);
    2. resolve each exogenous series' lag-1 value from ``exog_rules`` given
       the fresh state row (deeper lags come from previously resolved
       values, seeded by ``exog_seeds`` = real data) -- each rule's
       ``resolve`` is called EXACTLY once per step, in step order (a
       contract stateful rules like :class:`DataPathExogRule` rely on).
       S8 E2: a series the feedback map references at LAG 0
       (``meta.exog_contemporaneous``) is resolved for the CURRENT period
       instead; its lags 1..depth then come from the register, which
       ``exog_seeds`` seeds at lags 1..depth (plans/S8-plan.md conflict 4;
       every series without a lag-0 reference keeps the S4 contract
       exactly);
    3. draw measurement shocks via ``meas_noise.step`` -- one per declared
       measurement SHOCK; with ``meas_loading`` (S8 E5, the (m, n_shocks)
       matrix ``M``) the row-space error is ``M @ eps``, else ``eps`` is
       already in row space (the pre-S8 identity);
    4. build x from the feedback registers (the ``Const`` column is 1: a
       forward path is a LEVEL path), apply the measurement equation --
       with ``loadings`` (S9 E4) at ``Z_t = loadings.at(x_t)``, the
       time-varying coefficients drawn as states and the regressors from
       the simulated path (the full bilinear system) -- and push the new
       observables (and the resolved exogenous values) into their registers.

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
        if meta.exog_is_referenced(name) and name not in exog_rules:
            raise ValueError(
                f"simulate_forward needs an exog_rules entry for series "
                f"{name!r} (the feedback map references it)."
            )

    n = F.shape[0]
    if state_noise is None:
        state_noise = ConstantStateNoise(Q)
    obs_reg = _Registers(meta.obs_names, meta.obs_lag_depth, obs_seeds)
    # Exogenous registers hold PREVIOUSLY RESOLVED values only (the lag-1
    # value is always this step's fresh resolution), so entry i covers lag
    # i+2. Seeds use natural lag semantics relative to the FIRST simulated
    # period: {"r": {2: r_{T-1}}} seeds the lag-2 slot for step 0.
    # A series referenced at lag 0 (S8 E2) is resolved for the CURRENT
    # period, so its register starts at lag 1 (entry i covers lag i+1) and
    # seeds cover lags 1..depth.
    exog_hist: dict[str, list[float]] = {}
    first_lag: dict[str, int] = {}
    for name in exog_rules:
        depth = meta.exog_lag_depth(name)
        lo = 1 if meta.exog_contemporaneous(name) else 2
        first_lag[name] = lo
        series_seeds = dict(exog_seeds.get(name, {}))
        unknown = set(series_seeds) - set(range(lo, depth + 1))
        if unknown:
            raise ValueError(
                f"Exog seeds for series {name!r} reference lag(s) "
                f"{sorted(unknown)}; only lags {lo}..{depth} are seedable (the "
                f"lag-{lo - 1} value is always resolved by the forecast rule)."
            )
        exog_hist[name] = [float(series_seeds.get(lag, 0.0)) for lag in range(lo, depth + 1)]

    obs = np.empty((horizon, meta.n_obs))
    states = np.empty((horizon + 1, n))
    exog_resolved: dict[str, np.ndarray] = {name: np.empty(horizon) for name in exog_rules}

    xi_prev = np.asarray(xi_last, dtype=np.float64).copy()
    for t in range(horizon):
        w = state_noise.step(rng)
        xi_t = F @ xi_prev + w
        states[t] = xi_t

        resolved = {name: rule.resolve(xi_t) for name, rule in exog_rules.items()}
        for name, value in resolved.items():
            exog_resolved[name][t] = value

        eps = meas_noise.step(rng)
        e_row = eps if meas_loading is None else meas_loading @ eps

        # x's exogenous lag-1 columns read this step's freshly resolved
        # value (lag-0 columns for a contemporaneously referenced series);
        # deeper lags read the register (prior resolutions/seeds).
        x_t = np.zeros(len(meta.feedback_map))
        for j, term in enumerate(meta.feedback_map):
            if isinstance(term, ObsLag):
                x_t[j] = obs_reg.value(term.name, term.lag)
            elif isinstance(term, ObsLagMean):
                total = 0.0
                for lag in term.lags:
                    total += obs_reg.value(term.name, lag)
                x_t[j] = total / len(term.lags)
            elif isinstance(term, Const):
                x_t[j] = 1.0
            else:  # ExogLag
                lo = first_lag[term.name]
                x_t[j] = resolved[term.name] if term.lag == lo - 1 else exog_hist[term.name][term.lag - lo]

        Z_t = Z if loadings is None else loadings.at(x_t)
        y_t = A.T @ x_t + Z_t @ xi_t + e_row
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
    w_final = state_noise.step(rng)
    states[horizon] = F @ xi_prev + w_final

    return {"obs": obs, "states": states, "exog_resolved": exog_resolved}
