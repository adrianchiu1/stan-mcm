"""FAMILY-GENERIC results core (S6 WP2): the pieces of a family's output
layer that contain no family knowledge at all, lifted out of the lw_sv
results module so family #2 (UCSV, ``results_ucsv.py``) is a thin
DECLARATION over them -- and any later family likewise.

What lives here (every piece parameterized by the family's declared
metadata, never by names or slot indices):

- run loading: the model-ready DataFrame from a run's OWN
  ``data.snapshot.csv`` (:func:`load_run_dataframe`), the posterior
  flattened over (chain, draw) (:func:`flatten_posterior`), and the
  draw-subset selection (:func:`select_draw_indices`);
- the per-draw simulation-smoother loop (:func:`smoother_draws`): the DK
  smoother at each selected draw's own system matrices, returning state
  paths and the named structural shocks recovered through the family's
  declared loadings;
- the historical-decomposition arithmetic: state components per declared
  state shock plus the initial-condition RESIDUAL
  (:func:`state_components`), and the observable bars via the ONE generic
  engine (:func:`observable_bars`) -- one bar per declared state shock,
  one per declared measurement shock, an ``init`` bar, and an ``exog``
  bar when the feedback map carries genuinely exogenous columns. By
  linearity the bars sum to the observables -- the G6 identity;
- impulse responses (:func:`impulse_response`): one shock's synthetic
  impulse through the same two engine calls, from rest.

lw_sv's ``results_lw.py`` remains the validated lw_sv instantiation
(its reporting mapping, gap-space bars, c==1 guards and fan seeds are
family-specific); where its code IS generic it delegates here
(``select_draw_indices``, ``_load_run_dataframe``). The residue that had
to be generalized for UCSV is listed in DECISIONS.md (2026-09-04).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import pandas as pd

from macrotoolkit.engine import observable_recursion, propagate_state_shock
from macrotoolkit.families.base import ExogLag, StateSpaceMeta
from specs.schema.base import RunSpec


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_run_dataframe(run_dir: Path, spec: RunSpec) -> pd.DataFrame:
    """Reconstruct the model-ready DataFrame (date + mapped columns,
    trimmed to ``spec.data.sample``) from a completed run's OWN
    ``data.snapshot.csv`` (the byte-identical copy of the source CSV
    hashed at run time) -- a run directory is self-contained. Mirrors
    ``macrotoolkit.data.load_data``'s mapping/date-parsing/sample-trim
    steps exactly; only the file-resolution step differs."""
    from macrotoolkit.data import _parse_sample_bound

    csv_path = run_dir / "data.snapshot.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{csv_path} not found -- {run_dir} does not look like a "
            f"completed run directory (expected data.snapshot.csv alongside "
            f"spec.yaml and draws.nc)."
        )
    raw_df = pd.read_csv(csv_path)

    if spec.data.date_column not in raw_df.columns:
        raise ValueError(
            f"data.date_column {spec.data.date_column!r} (from {run_dir / 'spec.yaml'}) "
            f"not found in {csv_path}. Available columns: {list(raw_df.columns)}."
        )
    dates = pd.to_datetime(raw_df[spec.data.date_column])
    df = pd.DataFrame({"date": dates})
    for model_var, csv_col in spec.data.mapping.items():
        if csv_col not in raw_df.columns:
            raise ValueError(
                f"data.mapping[{model_var!r}] = {csv_col!r} not found in "
                f"{csv_path}. Available columns: {list(raw_df.columns)}."
            )
        series = pd.to_numeric(raw_df[csv_col], errors="coerce")
        if series.isna().any():
            raise ValueError(
                f"data.mapping[{model_var!r}] = {csv_col!r} in {csv_path} "
                f"has missing/non-numeric values -- this should be "
                f"impossible for a run's own snapshot; the snapshot file "
                f"may be corrupted or was hand-edited."
            )
        df[model_var] = series
    df = df.sort_values("date").reset_index(drop=True)

    if spec.data.sample.start is not None:
        start_ts = _parse_sample_bound(spec.data.sample.start, "data.sample.start")
        df = df[df["date"] >= start_ts]
    if spec.data.sample.end is not None:
        end_ts = _parse_sample_bound(spec.data.sample.end, "data.sample.end", end=True)
        df = df[df["date"] <= end_ts]
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"data.sample (start={spec.data.sample.start!r}, "
            f"end={spec.data.sample.end!r}) trims {csv_path} to zero rows "
            f"for run {run_dir} -- the spec's sample bounds do not match "
            f"this run's own data snapshot."
        )
    return df


def select_draw_indices(n_total: int, smoother_draws: object) -> np.ndarray:
    """Map ``spec.outputs.smoother_draws`` (``"all"`` or a ``ThinSpec``
    with ``thin=k``) to indices into the flattened (chain*draw) posterior,
    in flattening order (chain-major)."""
    if smoother_draws == "all":
        return np.arange(n_total)
    thin = getattr(smoother_draws, "thin", None)
    if thin is None:
        raise ValueError(
            f"outputs.smoother_draws must be 'all' or a ThinSpec(thin=k); "
            f"got {smoother_draws!r}."
        )
    return np.arange(0, n_total, thin)


def flatten_posterior(idata, names: list[str], run_dir: Path | None = None) -> dict[str, np.ndarray]:
    """The named posterior variables as plain numpy arrays flattened over
    (chain, draw) into one leading sample axis -- done once up front so
    per-draw loops are plain indexing. A missing variable fails loudly."""
    post = idata.posterior
    n_total = post.sizes["chain"] * post.sizes["draw"]
    flat: dict[str, np.ndarray] = {}
    for name in names:
        if name not in post.data_vars:
            raise ValueError(
                f"Posterior{f' in {run_dir}' if run_dir else ''} has no variable "
                f"{name!r}. Available variables: {sorted(post.data_vars)}."
            )
        arr = np.asarray(post[name].values)
        flat[name] = arr.reshape((n_total,) + arr.shape[2:])
    return flat


# ---------------------------------------------------------------------------
# Per-draw system matrices + the simulation-smoother loop
# ---------------------------------------------------------------------------


@dataclass
class DrawMatrices:
    """One posterior draw's system matrices: ``Q``/``R`` constant
    ``(n,n)``/``(m,m)`` or time-varying ``(T,n,n)``/``(T,m,m)`` paths, plus
    whatever per-draw extras the family's outputs need (h paths, scalar
    sds) keyed by name."""

    F: np.ndarray
    Q: np.ndarray
    A: np.ndarray
    Z: np.ndarray
    R: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)


MatricesForDraw = Callable[[Mapping[str, np.ndarray], int], DrawMatrices]


def smoother_draws(
    yobs: np.ndarray,
    x: np.ndarray,
    xi00: np.ndarray,
    P00: np.ndarray,
    meta: StateSpaceMeta,
    flat: Mapping[str, np.ndarray],
    idx: np.ndarray,
    matrices_for_draw: MatricesForDraw,
    rng: np.random.Generator,
) -> Iterator[tuple[int, DrawMatrices, Any]]:
    """Yield ``(draw_index, DrawMatrices, SimSmootherDraw)`` for each
    selected draw: the DK simulation smoother at that draw's own matrices
    and the REAL data, structural shocks recovered generically
    (``simulate_smoother_draw(..., meta=meta)``). One ``rng`` advanced
    sequentially across draws (the established convention)."""
    from macrotoolkit.smoother import simulate_smoother_draw

    for i in idx:
        dm = matrices_for_draw(flat, int(i))
        sim = simulate_smoother_draw(yobs, x, dm.F, dm.Q, dm.A, dm.Z, dm.R, xi00, P00, rng, meta=meta)
        yield int(i), dm, sim


# ---------------------------------------------------------------------------
# Historical decomposition (generic)
# ---------------------------------------------------------------------------


def state_components(F: np.ndarray, meta: StateSpaceMeta, xi_draw: np.ndarray, state_shocks: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Linear decomposition of a drawn state path into one component per
    declared state shock (its own stream propagated through ``F`` at the
    declared loading, from the zero state) plus ``"init"`` = the RESIDUAL
    -- the full-sample-revised initial-condition contribution (see
    ``results_lw.state_shock_decomposition`` for why the residual, not a
    propagated prior). Sums to ``xi_draw`` exactly by construction."""
    comps: dict[str, np.ndarray] = {}
    total = np.zeros_like(xi_draw)
    for shock in meta.state_shocks:
        comps[shock] = propagate_state_shock(F, meta.injection_vector(shock), state_shocks[shock])
        total = total + comps[shock]
    comps["init"] = xi_draw - total
    return comps


def hd_bar_names(meta: StateSpaceMeta) -> tuple[str, ...]:
    """The observable-space bars, in display order: init, each state
    shock, each measurement shock, and ``exog`` iff the feedback map has
    genuinely exogenous columns."""
    bars = ["init", *meta.state_shocks, *meta.measurement_shocks]
    if any(isinstance(t, ExogLag) for t in meta.feedback_map):
        bars.append("exog")
    return tuple(bars)


def observable_bars(
    A: np.ndarray,
    Z: np.ndarray,
    meta: StateSpaceMeta,
    components: Mapping[str, np.ndarray],
    meas_shocks: Mapping[str, np.ndarray],
    x: np.ndarray | None = None,
    init_obs_seeds: Mapping[str, Mapping[int, float]] | None = None,
) -> dict[str, np.ndarray]:
    """Each bar's own OBSERVABLE path (T, m) via the generic engine's
    observation recursion: state bars carry their state component, a
    measurement bar injects its recovered shock into its observation row,
    the ``exog`` bar carries the real exogenous regressor columns, the
    ``init`` bar its state component plus the real pre-sample observable
    seeds; everything else runs from rest. By linearity the bars sum to
    the observables -- the G6 identity."""
    T = next(iter(components.values())).shape[0]
    zeros_state = np.zeros((T, meta.n_state))
    bars: dict[str, np.ndarray] = {}
    for k in hd_bar_names(meta):
        state_path = components[k] if k in components else zeros_state
        meas = np.zeros((T, meta.n_obs))
        if k in meta.measurement_shocks:
            meas[:, meta.measurement_shocks.index(k)] = meas_shocks[k]
        exog = x if k == "exog" else None
        seeds = init_obs_seeds if k == "init" else None
        bars[k] = observable_recursion(A, Z, meta, state_path, meas, exog, seeds)
    return bars


# ---------------------------------------------------------------------------
# Impulse responses (generic)
# ---------------------------------------------------------------------------


def impulse_response(
    F: np.ndarray, A: np.ndarray, Z: np.ndarray, meta: StateSpaceMeta, shock: str, size: float, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """One shock's theoretical impulse response from rest: returns the
    state path ``(horizon, n)`` and the observable path ``(horizon, m)``
    -- exactly one HD bar with a synthetic one-off impulse."""
    impulse = np.zeros(horizon)
    impulse[0] = size
    meas = np.zeros((horizon, meta.n_obs))
    if shock in meta.state_shocks:
        comp = propagate_state_shock(F, meta.injection_vector(shock), impulse)
    elif shock in meta.measurement_shocks:
        comp = np.zeros((horizon, meta.n_state))
        meas[:, meta.measurement_shocks.index(shock)] = impulse
    else:
        raise ValueError(
            f"Unknown shock {shock!r}; declared state shocks {list(meta.state_shocks)}, "
            f"measurement shocks {list(meta.measurement_shocks)}."
        )
    obs = observable_recursion(A, Z, meta, comp, meas)
    return comp, obs
