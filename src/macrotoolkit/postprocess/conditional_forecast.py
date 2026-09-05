"""Waggoner-Zha (1999) conditional forecasts with HARD conditions (Blake
& Mumtaz 2017 Chapter 2 §7, example 8) as a POST-PROCESSOR over a run's
unconditional forecast and structural IRFs.

In a linear model the ``H``-step forecast path is

    y_{T+h} = ybar_{T+h} + sum_{s=0..h} irf[h - s] e_{T+s},   e ~ N(0, I_k)

with ``ybar`` the unconditional (zero-shock) forecast and ``irf[j]`` the
``(m, k)`` response at horizon ``j`` to the unit-variance structural
shocks. Hard conditions ``y_{T+h, v} = c`` are linear restrictions on the
stacked shock vector ``e = (e_T, ..., e_{T+H-1})``: ``R e = r`` with
``R[c, s*k + j] = irf[h - s][v, j]`` and ``r = c - ybar_{T+h, v}``. The
restricted shocks are Gaussian with mean ``R' (R R')^+ r`` and covariance
``I - R' (R R')^+ R`` (the handbook's ``MBAR``/``VBAR`` with ``pinv``;
computed here through the SVD of ``R``, the same quantities better
conditioned);
draws through the IRFs added to ``ybar`` are the conditional forecast.
With no conditions the mean is zero and the covariance the identity: the
draws have exactly the fan chart's distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass
class ConditionalForecast:
    """``mean`` ``(H, m)`` the conditional mean path; ``draws`` ``(n_draws,
    H, m)``; ``shock_mean`` ``(H, k)`` / ``shock_cov`` ``(H k, H k)`` the
    restricted shocks' distribution; ``targets`` / ``shocks`` the labels."""

    mean: np.ndarray
    draws: np.ndarray
    shock_mean: np.ndarray
    shock_cov: np.ndarray
    unconditional: np.ndarray
    targets: tuple[str, ...]
    shocks: tuple[str, ...]


def _stack_R(irf: np.ndarray, conditions: Mapping[tuple[int, int], float], ybar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H, m, k = irf.shape
    rows, r = [], []
    for (v, h), value in conditions.items():
        row = np.zeros(H * k)
        for s in range(h + 1):
            row[s * k : (s + 1) * k] = irf[h - s, v, :]
        rows.append(row)
        r.append(float(value) - ybar[h, v])
    return np.array(rows).reshape(len(rows), H * k), np.array(r)


def _paths(irf: np.ndarray, ybar: np.ndarray, e: np.ndarray) -> np.ndarray:
    """``e`` is ``(n, H, k)``; returns ``(n, H, m)`` paths."""
    n, H, k = e.shape
    out = np.repeat(ybar[None, :, :], n, axis=0)
    for h in range(H):
        for s in range(h + 1):
            out[:, h, :] += e[:, s, :] @ irf[h - s].T
    return out


def conditional_forecast(
    unconditional: np.ndarray,
    irf: np.ndarray,
    conditions: Mapping[tuple[str | int, int], float],
    rng: np.random.Generator,
    n_draws: int,
    *,
    targets: Sequence[str] | None = None,
    shocks: Sequence[str] | None = None,
    shock_draws: np.ndarray | None = None,
) -> ConditionalForecast:
    """Conditional forecast for ONE (posterior draw's) model: ``unconditional``
    ``(H, m)``, ``irf`` ``(H, m, k)`` at unit-variance shock sizes,
    ``conditions`` ``{(variable, horizon): value}`` (variable by name or
    row index, horizon 0-based). ``shock_draws`` ``(n_draws, H, k)`` of
    standard normals replaces the internal draw (the zero-noise seam: the
    same standard normals through the engine give the same paths)."""
    ybar = np.asarray(unconditional, dtype=np.float64)
    irf = np.asarray(irf, dtype=np.float64)
    if ybar.ndim != 2 or irf.ndim != 3 or irf.shape[0] != ybar.shape[0] or irf.shape[1] != ybar.shape[1]:
        raise ValueError(f"unconditional must be (H, m) and irf (H, m, k); got {ybar.shape} and {irf.shape}.")
    H, m, k = irf.shape
    targets = tuple(targets) if targets is not None else tuple(f"y{i}" for i in range(m))
    shocks = tuple(shocks) if shocks is not None else tuple(f"e{j}" for j in range(k))
    conds: dict[tuple[int, int], float] = {}
    for (v, h), value in conditions.items():
        vi = targets.index(v) if isinstance(v, str) else int(v)
        if not 0 <= h < H:
            raise ValueError(f"condition horizon {h} outside 0..{H - 1}.")
        conds[(vi, int(h))] = float(value)
    if conds:
        R, r = _stack_R(irf, conds, ybar)
        # R'(RR')^+ r and I - R'(RR')^+ R through the SVD of R itself (not of
        # RR', whose conditioning is squared): the minimum-norm solution and
        # the projector off R's row space, exactly the handbook's pinv
        # formulas in exact arithmetic, with the conditions reproduced to
        # ~1e-12 even at badly scaled posterior draws.
        U, sv, Vt = np.linalg.svd(R, full_matrices=False)
        keep = sv > sv[0] * 1e-13 if sv.size else np.zeros(0, dtype=bool)
        Vr = Vt[keep]
        mean_vec = Vr.T @ ((U[:, keep].T @ r) / sv[keep])
        cov = np.eye(H * k) - Vr.T @ Vr
    else:
        mean_vec = np.zeros(H * k)
        cov = np.eye(H * k)
    from macrotoolkit.smoother import _psd_sqrt

    if shock_draws is None:
        z = rng.standard_normal((n_draws, H * k))
    else:
        z = np.asarray(shock_draws, dtype=np.float64).reshape(n_draws, H * k)
    e = (mean_vec[None, :] + z @ _psd_sqrt(cov).T).reshape(n_draws, H, k)
    mean_path = _paths(irf, ybar, mean_vec.reshape(1, H, k))[0]
    return ConditionalForecast(mean=mean_path, draws=_paths(irf, ybar, e), shock_mean=mean_vec.reshape(H, k), shock_cov=cov,
                               unconditional=ybar, targets=targets, shocks=shocks)


def _authored(run):
    """Accept an ``api.Run`` (whose ``results()`` loads the family's run
    object) or the loaded ``AuthoredRun`` itself."""
    if hasattr(run, "compiled"):
        return run
    if hasattr(run, "results"):
        return run.results()
    raise TypeError(f"expected an api.Run or an AuthoredRun; got {type(run).__name__}.")


def unconditional_and_irfs_for_run(run, horizon: int, *, seed: int | None = None):
    """Per selected posterior draw of an AUTHORED run: the zero-shock
    forward path ``(H, m)`` from the draw's terminal smoothed state (the
    fan chart's own seeds and forecast rules) and the structural IRF array
    ``(H, m, k)`` over every shock at 1-sd sizes. Yields ``(draw_index,
    ybar, irf)``."""
    from macrotoolkit.authoring.results import _draw_loop, _terminal_seeds, exog_rules_for, irf_shock_size
    from macrotoolkit.engine import simulate_forward
    from macrotoolkit.results_core import impulse_response

    run = _authored(run)
    c = run.compiled
    meta = c.meta
    shocks = tuple(meta.state_shocks) + tuple(meta.measurement_shocks)
    obs_seeds, exog_seeds = _terminal_seeds(c, run.series)
    n_rows = len(next(iter(run.series.values())))

    class _Zero:
        def __init__(self, n):
            self.n = n

        def step(self, rng):
            return np.zeros(self.n)

    _, idx, rng, it = _draw_loop(run, seed)
    for i, dm, sim in it:
        Q_const = dm.Q if dm.Q.ndim == 2 else dm.Q[-1]
        out = simulate_forward(dm.F, Q_const, dm.A, dm.Z, meta, sim.xi_draw[-1], obs_seeds, exog_seeds, exog_rules_for(c, run.series, n_rows - 1),
                               _Zero(meta.n_meas_shocks), horizon, rng, state_noise=_Zero(meta.n_state), meas_loading=dm.M)
        irf = np.empty((horizon, meta.n_obs, len(shocks)))
        for j, s in enumerate(shocks):
            irf[:, :, j] = impulse_response(dm.F, dm.A, dm.Z, meta, s, irf_shock_size(c, s, dm, run.spec.outputs.irf_vol_reference), horizon, M=dm.M)[1]
        yield i, out["obs"], irf, shocks


def conditional_forecast_for_run(run, conditions: Mapping[tuple[str, int], float], horizon: int, *, draws_per_posterior_draw: int = 1, seed: int | None = None) -> ConditionalForecast:
    """The conditional forecast of an authored run: per selected posterior
    draw the zero-shock path and the structural IRFs, the WZ restricted
    shocks drawn ``draws_per_posterior_draw`` times, pooled over draws.
    ``conditions`` are keyed by observable name and 0-based horizon."""
    run = _authored(run)
    rng = np.random.default_rng(seed if seed is not None else run.spec.sampler.seed + 1)
    means, draws, ybars = [], [], []
    labels = None
    for _, ybar, irf, shocks in unconditional_and_irfs_for_run(run, horizon, seed=seed):
        cf = conditional_forecast(ybar, irf, conditions, rng, draws_per_posterior_draw, targets=run.meta.obs_names, shocks=shocks)
        means.append(cf.mean)
        draws.append(cf.draws)
        ybars.append(ybar)
        labels = (cf.targets, cf.shocks)
    if labels is None:
        raise ValueError("the run has no selected posterior draws.")
    return ConditionalForecast(mean=np.mean(means, axis=0), draws=np.concatenate(draws, axis=0), shock_mean=np.zeros(0), shock_cov=np.zeros((0, 0)),
                               unconditional=np.mean(ybars, axis=0), targets=labels[0], shocks=labels[1])
