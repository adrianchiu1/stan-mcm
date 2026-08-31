"""Per-run, per-posterior-draw outputs for the ``lw_sv`` family: trend-cycle
series (spec §3.1) and historical decomposition (spec §3.4). Everything here
consumes a *completed* run directory (``runs/<hash12>/``) and re-runs the
Durbin-Koopman simulation smoother (``macrotoolkit.smoother.
simulate_smoother_draw``) per selected posterior draw -- gated by G1 (this
module must not be trusted until the smoother's own gate, spec §2.4, is
green) and validated for the historical-decomposition arithmetic specifically
by gate G6 (``tests/test_g6_hd_identity.py``).

Four pieces, kept separable in this file:

- **Part A -- trend-cycle loading/aggregation** (``load_lw_run``,
  ``compute_trend_cycle_draws``): reconstructs a run's data/system inputs
  from its own stored artifacts (``spec.yaml``, ``data.snapshot.csv``,
  ``draws.nc``) and produces per-draw (y*, gap, g, z, r*, and -- SV only --
  the volatility paths) series, stacked over the posterior draws selected by
  ``outputs.smoother_draws``. Does NOT compute credible-interval bands --
  that is ``plots.py``'s job (a later S4 step).
- **Part B -- historical decomposition** (``state_shock_decomposition``,
  ``gap_pi_shock_decomposition``, ``y_level_decomposition``,
  ``historical_decomposition_draw``): per single simulation-smoother draw,
  propagates each of the 5 structural shocks (eps_ystar, eps_g, eps_z,
  eps_is, eps_pc) through the model's OWN linear recursions (the state
  transition for the 3 process shocks; gap's own AR(2) IS-curve feedback and
  pi's own Phillips-curve feedback for eps_is/eps_pc, spec §3.4) so that
  every one of the 5 shocks -- including the two measurement shocks -- gets
  an economically sensible, persistent contribution rather than a same-
  period-only residual. Aggregation across draws (e.g. posterior-median
  contributions for the stacked-bar chart) is the caller's job.
- **Part C -- IRF matrix** (``impulse_response_for_shock``,
  ``irf_shock_size``, ``compute_irf_draws``, spec §3.2): a theoretical
  impulse response is a historical decomposition with no real data, no
  "init" contribution (there is no prior/smoothed belief to carry -- gap/pi
  start entirely at rest), and a synthetic one-off impulse instead of a
  recovered shock path -- so this reuses Part B's own machinery (the
  ``_propagate_shock_through_F`` state-propagation helper for the 3 trend
  shocks, ``gap_pi_shock_decomposition``'s AR-recursion arithmetic for
  gap/pi) rather than reimplementing it.
- **Part D -- fan charts** (``simulate_fan_draw``, ``compute_fan_draws``,
  spec §3.3): genuinely new STOCHASTIC forward simulation (fresh process/
  measurement noise every period, including -- for an SV run -- continuing
  the SV log-variance random walks forward, spec §3.3's "widening bands are
  the point"), seeded from a draw's own terminal smoother state and the
  REAL last few periods' gap/pi/rate-gap lag registers (not zero -- unlike
  Part C's from-rest IRF). Reuses the same per-period IS-curve/Phillips-
  curve arithmetic as Part B/C (factored into ``_fan_forecast_step``), just
  combined into one stochastic realization per draw rather than a
  shock-by-shock decomposition.

Numerical conventions this module inherits from ``macrotoolkit.smoother``
(lw-sv-spec.md §1.1/§1.3/§1.5, enforced there by
``tests/test_units_conventions.py``; re-touched directly in this file only
by the SV volatility-path computation below):
``g`` is ANNUALIZED everywhere (state slot 3/5, 0-indexed, i.e. HANDOFF.md's
"state slots 4/6 (1-indexed) hold g_{t-1}/z_{t-1}"); the potential-output
transition alone divides by 4. ``h`` is log-VARIANCE, so the reported
volatility path is ``exp(h/2)`` (the STANDARD DEVIATION), never ``exp(h)``
(the variance) or ``h`` itself -- computed here directly from that draw's
own saved ``h_is``/``h_pc`` transformed parameters, per HANDOFF.md's S4
warning ("never re-derive h from nu").

Reporting mapping (spec §3.1's y*/gap/g/z/r* series) is copied VERBATIM from
``tests/test_g5a_hlw_replication.py::_series_from_states``, per HANDOFF.md's
explicit instruction -- never re-derived from spec prose:
``rstar = xi[:,3] + xi[:,5]``, ``g = xi[:,3]``, ``z = xi[:,5]``,
``output_gap = yobs[:,0] - xi[:,0]``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from specs.schema.base import RunSpec, load_spec

# ---------------------------------------------------------------------------
# Part A: loading a completed run + trend-cycle series (spec §3.1)
# ---------------------------------------------------------------------------


def _load_run_dataframe(run_dir: Path, spec: RunSpec) -> pd.DataFrame:
    """Reconstruct the model-ready DataFrame (date + mapped y/pi/r columns,
    trimmed to ``spec.data.sample``) from a completed run's OWN artifacts --
    ``data.snapshot.csv`` (the byte-identical copy of the source CSV
    ``macrotoolkit.data.load_data`` hashed at run time) -- rather than
    re-resolving ``spec.data.file`` against the original spec file's
    directory (which may not exist, or may have changed, in whatever
    environment this function is later called from; a run directory is
    meant to be self-contained). Mirrors ``macrotoolkit.data.load_data``'s
    mapping/date-parsing/sample-trim steps exactly; only the file-resolution
    step differs.
    """
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


@dataclass
class LWRun:
    """A loaded, completed ``lw_sv`` run: the system inputs (data, initial
    state) plus the raw ArviZ posterior needed to re-run the DK simulation
    smoother per draw (spec §2.4) and compute this module's §3.1/§3.4
    output series. Immutable snapshot of ``run_dir``'s own artifacts.

    ``yobs``/``x``/``xi00``/``P00`` are built by
    ``macrotoolkit.smoother.build_lw_regressors``/``default_initial_state``
    from the same trimmed data ``macrotoolkit.run.build_stan_data`` used at
    run time (reused, not re-derived) -- ``yobs`` is (T,2), ``x`` is (T,6).
    ``y_full``/``pi_full``/``r_full`` are the FULL length-(T+4) series
    (including the 4 pre-sample lag quarters), needed by Part B's historical
    decomposition for its own pre-sample-anchored recursions. ``dates`` is
    the length-T estimation-sample date index (row i = period i+1).
    """

    run_dir: Path
    spec: RunSpec
    idata: object  # arviz.InferenceData -- typed as object to avoid an arviz import at module scope
    sv_on: bool
    yobs: np.ndarray
    x: np.ndarray
    xi00: np.ndarray
    P00: np.ndarray
    y_full: np.ndarray
    pi_full: np.ndarray
    r_full: np.ndarray
    dates: pd.DatetimeIndex


def load_lw_run(run_dir: str | Path) -> LWRun:
    """Load a completed ``lw_sv`` run directory into an :class:`LWRun`."""
    import arviz as az

    from macrotoolkit.smoother import build_lw_regressors, default_initial_state

    run_dir = Path(run_dir)
    spec_path = run_dir / "spec.yaml"
    draws_path = run_dir / "draws.nc"
    if not spec_path.is_file():
        raise FileNotFoundError(
            f"{spec_path} not found -- {run_dir} does not look like a "
            f"completed run directory."
        )
    if not draws_path.is_file():
        raise FileNotFoundError(
            f"{draws_path} not found -- {run_dir} does not look like a "
            f"completed run directory."
        )

    spec = load_spec(str(spec_path))
    if spec.model.family != "lw_sv":
        raise ValueError(
            f"results_lw.load_lw_run only supports model.family 'lw_sv'; "
            f"run {run_dir} has family {spec.model.family!r}."
        )

    df = _load_run_dataframe(run_dir, spec)
    if len(df) < 5:
        raise ValueError(
            f"Run {run_dir}'s data has only {len(df)} row(s) after "
            f"trimming; lw_sv needs at least 5 (4 pre-sample lag quarters + "
            f"1 estimation quarter)."
        )

    y_full = df["y"].to_numpy(dtype=np.float64)
    pi_full = df["pi"].to_numpy(dtype=np.float64)
    r_full = df["r"].to_numpy(dtype=np.float64)

    yobs, x = build_lw_regressors(y_full, pi_full, r_full)
    xi00, P00 = default_initial_state(float(y_full[4]))

    idata = az.from_netcdf(str(draws_path))
    sv_on = bool(spec.model.options.sv_shocks)
    dates = pd.DatetimeIndex(df["date"].to_numpy()[4:])

    return LWRun(
        run_dir=run_dir,
        spec=spec,
        idata=idata,
        sv_on=sv_on,
        yobs=yobs,
        x=x,
        xi00=xi00,
        P00=P00,
        y_full=y_full,
        pi_full=pi_full,
        r_full=r_full,
        dates=dates,
    )


def select_draw_indices(n_total: int, smoother_draws: object) -> np.ndarray:
    """Map ``spec.outputs.smoother_draws`` (``"all"`` or
    ``specs.schema.lw_sv.ThinSpec(thin=k)``) to an array of indices into the
    flattened (chain*draw) posterior, in flattening order (chain-major, i.e.
    ``arr.reshape((n_chain*n_draw,) + arr.shape[2:])``'s own row order)."""
    if smoother_draws == "all":
        return np.arange(n_total)
    thin = getattr(smoother_draws, "thin", None)
    if thin is None:
        raise ValueError(
            f"outputs.smoother_draws must be 'all' or a ThinSpec(thin=k); "
            f"got {smoother_draws!r}."
        )
    return np.arange(0, n_total, thin)


#: Static (time-invariant) parameters common to both the no-SV and SV
#: variants -- everything build_lw_matrices needs except sigma_is/sigma_pc
#: (no-SV only) / h_is/h_pc (SV only).
_STATIC_PARAM_NAMES = ("a1", "a2", "a_r", "b_pi", "b_y", "sigma_ystar", "sigma_g", "sigma_z")


def _flatten_posterior(lw_run: LWRun) -> dict[str, np.ndarray]:
    """Extract the posterior variables this module needs as plain numpy
    arrays, flattened over (chain, draw) into one leading "sample" axis --
    done ONCE up front (not per-draw inside a hot loop) so the per-draw
    smoother loop below is plain numpy indexing, not repeated xarray
    overhead."""
    post = lw_run.idata.posterior
    n_total = post.sizes["chain"] * post.sizes["draw"]
    names = list(_STATIC_PARAM_NAMES) + (["h_is", "h_pc"] if lw_run.sv_on else ["sigma_is", "sigma_pc"])
    flat: dict[str, np.ndarray] = {}
    for name in names:
        if name not in post.data_vars:
            raise ValueError(
                f"Posterior in {lw_run.run_dir} has no variable {name!r} -- "
                f"expected for a {'SV' if lw_run.sv_on else 'no-SV'} lw_sv "
                f"run (sv_shocks={lw_run.spec.model.options.sv_shocks!r}). "
                f"Available variables: {sorted(post.data_vars)}."
            )
        arr = np.asarray(post[name].values)
        flat[name] = arr.reshape((n_total,) + arr.shape[2:])
    return flat


def _system_matrices_for_draw(
    flat: dict[str, np.ndarray], i: int, sv_on: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Build one posterior draw's ``(F, Q, A, Z, R)`` system matrices, plus
    that draw's own ``h_is``/``h_pc`` paths (``None``/``None`` for a no-SV
    draw). ``c`` is fixed at 1.0 (spec §1.3's default; ``estimate_c`` is
    validated ``False`` everywhere in the current scope, per
    ``specs.schema.lw_sv.LwSvOptions``)."""
    from macrotoolkit.smoother import build_lw_matrices, sv_diag_variance_path

    params = {name: float(flat[name][i]) for name in _STATIC_PARAM_NAMES}
    if sv_on:
        # sigma_is/sigma_pc: build_lw_matrices always needs SOME values to
        # construct its (F, Q, A, Z, R) tuple, but for an SV draw the
        # measurement covariance it would build from them is discarded
        # (`_` below) and replaced by sv_diag_variance_path(h_is, h_pc) --
        # that draw's OWN saved h_is/h_pc transformed parameters
        # (HANDOFF.md's S4 warning: never re-derive h from nu). Any
        # positive placeholder works; F/Q/A/Z do not depend on
        # sigma_is/sigma_pc at all (see build_lw_matrices).
        params["sigma_is"] = 1.0
        params["sigma_pc"] = 1.0
        F, Q, A, Z, _ = build_lw_matrices(params, c=1.0)
        h_is = flat["h_is"][i]
        h_pc = flat["h_pc"][i]
        R = sv_diag_variance_path(h_is, h_pc)
        return F, Q, A, Z, R, h_is, h_pc

    params["sigma_is"] = float(flat["sigma_is"][i])
    params["sigma_pc"] = float(flat["sigma_pc"][i])
    F, Q, A, Z, R = build_lw_matrices(params, c=1.0)
    return F, Q, A, Z, R, None, None


@dataclass
class TrendCycleDraws:
    """Per-draw trend-cycle series (spec §3.1), stacked over the posterior
    draws selected by ``outputs.smoother_draws``. Reporting mapping copied
    VERBATIM from
    ``tests/test_g5a_hlw_replication.py::_series_from_states``.

    ``dates``/``y``/``pi`` are the real observed data, length T, identical
    across draws. Every other field is (n_draws, T): one smoothed
    trend-cycle path per selected posterior draw. ``vol_is``/``vol_pc`` are
    ``None`` for a no-SV run (spec §3.1's volatility-path panel is SV-only).
    No credible-interval banding here -- that is ``plots.py``'s job.
    """

    draw_indices: np.ndarray  # (n_draws,) -- indices into the flattened posterior this batch used
    dates: pd.DatetimeIndex  # (T,)
    y: np.ndarray  # (T,)
    pi: np.ndarray  # (T,)
    ystar: np.ndarray  # (n_draws, T)
    output_gap: np.ndarray  # (n_draws, T)
    g: np.ndarray  # (n_draws, T) -- annualized
    z: np.ndarray  # (n_draws, T)
    rstar: np.ndarray  # (n_draws, T)
    vol_is: np.ndarray | None  # (n_draws, T) or None
    vol_pc: np.ndarray | None  # (n_draws, T) or None


def compute_trend_cycle_draws(lw_run: LWRun, *, seed: int | None = None) -> TrendCycleDraws:
    """Run the DK simulation smoother once per selected posterior draw
    (``lw_run.spec.outputs.smoother_draws``) and compute spec §3.1's
    trend-cycle series from each draw's state path. ``seed`` defaults to
    ``lw_run.spec.sampler.seed`` (documented, reproducible default); a
    single ``np.random.Generator`` is advanced sequentially across all
    selected draws (not re-seeded per draw), matching
    ``tests/test_smoother_sim.py``'s own usage pattern.
    """
    from macrotoolkit.smoother import simulate_smoother_draw

    flat = _flatten_posterior(lw_run)
    n_total = flat["a1"].shape[0]
    idx = select_draw_indices(n_total, lw_run.spec.outputs.smoother_draws)

    T = lw_run.yobs.shape[0]
    n = len(idx)
    ystar = np.empty((n, T))
    output_gap = np.empty((n, T))
    g_arr = np.empty((n, T))
    z_arr = np.empty((n, T))
    rstar = np.empty((n, T))
    vol_is = np.empty((n, T)) if lw_run.sv_on else None
    vol_pc = np.empty((n, T)) if lw_run.sv_on else None

    rng = np.random.default_rng(seed if seed is not None else lw_run.spec.sampler.seed)

    for j, i in enumerate(idx):
        F, Q, A, Z, R, h_is, h_pc = _system_matrices_for_draw(flat, int(i), lw_run.sv_on)
        sim = simulate_smoother_draw(lw_run.yobs, lw_run.x, F, Q, A, Z, R, lw_run.xi00, lw_run.P00, rng)
        xi = sim.xi_draw
        ystar[j] = xi[:, 0]
        output_gap[j] = lw_run.yobs[:, 0] - xi[:, 0]
        g_arr[j] = xi[:, 3]
        z_arr[j] = xi[:, 5]
        rstar[j] = xi[:, 3] + xi[:, 5]
        if lw_run.sv_on:
            # h is log-VARIANCE (spec §1.5); exp(h/2) is the standard
            # deviation -- exp(h) would be the variance, a units bug.
            vol_is[j] = np.exp(h_is / 2.0)
            vol_pc[j] = np.exp(h_pc / 2.0)

    return TrendCycleDraws(
        draw_indices=idx,
        dates=lw_run.dates,
        y=lw_run.yobs[:, 0],
        pi=lw_run.yobs[:, 1],
        ystar=ystar,
        output_gap=output_gap,
        g=g_arr,
        z=z_arr,
        rstar=rstar,
        vol_is=vol_is,
        vol_pc=vol_pc,
    )


# ---------------------------------------------------------------------------
# Part B: historical decomposition (spec §3.4) -- see this module's
# docstring for why a naive "gap = data - state" split is wrong, and
# lw-sv-spec.md §3.4 / the S4 task brief for the derivation this
# implements.
# ---------------------------------------------------------------------------


def _w_ystar_injection(eps_t: float, n: int) -> np.ndarray:
    """The y*-shock's own per-period state-innovation vector -- eps_ystar
    only ever enters state slot 0 (``build_lw_matrices``' exact ``Q``
    construction). Shared by :func:`state_shock_decomposition` (B1, fed a
    real recovered shock path) and :func:`impulse_response_for_shock` (Part
    C, fed a synthetic impulse array) via :func:`_propagate_shock_through_F`."""
    w = np.zeros(n)
    w[0] = eps_t
    return w


def _w_g_injection(eps_t: float, n: int) -> np.ndarray:
    """The g-shock's own per-period state-innovation vector -- the
    ANNUALIZED g innovation enters both slot 3 (g itself) and, divided by 4,
    slot 0 (y*'s quarterly increment g/4, lw-sv-spec.md §1.3's g/4
    convention). See :func:`_w_ystar_injection` for the sharing rationale."""
    w = np.zeros(n)
    w[0] = eps_t / 4.0
    w[3] = eps_t
    return w


def _w_z_injection(eps_t: float, n: int) -> np.ndarray:
    """The z-shock's own per-period state-innovation vector -- eps_z only
    ever enters state slot 5. See :func:`_w_ystar_injection` for the
    sharing rationale."""
    w = np.zeros(n)
    w[5] = eps_t
    return w


def _propagate_shock_through_F(F: np.ndarray, n: int, eps: np.ndarray, inject) -> np.ndarray:
    """Propagate ONE shock stream's own process noise through the state
    transition matrix ``F``, starting from the zero state vector at t=-1::

        xi[t] = F @ xi[t-1] + inject(eps[t], n)

    ``inject`` maps this period's scalar shock realization (plus the state
    dimension ``n``) to that period's (n,) state-innovation vector (one of
    :func:`_w_ystar_injection` / :func:`_w_g_injection` / :func:`_w_z_injection`).

    Factored out of :func:`state_shock_decomposition`'s inner loop, which
    used to triplicate this exact loop body once per shock (numerics-
    reviewer finding, S4) -- extracting it lets :func:`impulse_response_for_shock`
    (Part C, spec §3.2) reuse the IDENTICAL arithmetic on a synthetic impulse
    array instead of a real recovered shock path, rather than a fourth
    copy-pasted loop. Behavior-preserving: bit-for-bit identical float
    operations, in the same order, as the pre-refactor inline loop --
    confirmed by re-running ``tests/test_g6_hd_identity.py`` /
    ``tests/test_results_lw.py`` unchanged after this extraction.

    Returns an ``(len(eps), n)`` array.
    """
    T = len(eps)
    xi = np.zeros((T, n))
    prev = np.zeros(n)
    for t in range(T):
        cur = F @ prev + inject(eps[t], n)
        xi[t] = cur
        prev = cur
    return xi


def state_shock_decomposition(
    F: np.ndarray,
    xi_draw: np.ndarray,
    eps_ystar: np.ndarray,
    eps_g: np.ndarray,
    eps_z: np.ndarray,
) -> dict[str, np.ndarray]:
    """B1: linear decomposition of a drawn state path into 4 additive
    "bars": each of the 3 process shocks' own contribution, propagated
    through the SAME transition matrix ``F`` used to build the state
    itself, plus an initial-condition "init" bar defined as the RESIDUAL
    (spec §3.4's own words: "residual line = initial-condition
    contribution") --

        xi_ystar[t] = F @ xi_ystar[t-1] + [eps_ystar_t,0,0,0,0,0,0]
        xi_g[t]     = F @ xi_g[t-1]     + [eps_g_t/4,0,0,eps_g_t,0,0,0]
        xi_z[t]     = F @ xi_z[t-1]     + [0,0,0,0,0,eps_z_t,0]
        xi_init[t]  = xi_draw[t] - xi_ystar[t] - xi_g[t] - xi_z[t]

    (xi_ystar/xi_g/xi_z start from the zero vector at t=-1.)

    WHY a residual, not a separately-propagated ``F @ xi00`` forward path
    (as a first draft of this function did, and as a naive reading of "the
    initial state propagated forward" might suggest -- which is also why
    this function no longer even takes ``xi00`` as a parameter) --
    discovered while building gate G6 (``tests/test_g6_hd_identity.py``):
    ``xi00``/``P00`` is only a PRIOR for period 0's state, and the RTS
    smoother genuinely revises that prior using the WHOLE sample. Row 0
    (period 1)'s deterministic lag-copy state slots -- 1, 2, 4, 6, i.e.
    ``y*_0``, ``y*_{-1}``, ``g_{-1}``, ``z_{-1}`` -- are exactly this
    revised, full-sample-informed belief (see
    ``macrotoolkit.smoother._recover_structural_shocks``'s boundary note
    and ``tests/test_smoother_sim.py::test_lag_copy_rows_are_not_zero_at_t0_
    boundary``), and genuinely differ from ``(F @ xi00)``'s corresponding
    entries (the raw, un-revised prior). ``eps_ystar``/``eps_g``/``eps_z``
    cannot capture that revision -- by ``build_lw_matrices``'s own ``Q``
    construction they only ever inject into slots 0, 3, 5. A
    ``F @ xi00``-propagated "init" bar therefore MISSES this t=0 revision
    entirely, and -- because gap's own IS-curve recursion (B2 below) reads
    ``rstar2`` off slots 4/6, which ARE among the affected slots -- that
    single-period miss propagates through the AR(2)/AR(4) feedback and
    fails to decay away within a typical sample length, breaking G6's
    1e-6-per-period bar for the entire series (empirically confirmed while
    developing this module; not a hypothetical).

    Defining "init" as the residual instead sidesteps the whole issue: the
    B1 sum identity holds EXACTLY (to floating-point precision) at every t
    BY CONSTRUCTION -- and, as a genuine (non-tautological) correctness
    check, for t = 1..T-1 (where the boundary artifact does not apply) this
    residual independently satisfies ``xi_init[t] == F @ xi_init[t-1]`` to
    ~1e-9, pinned by ``tests/test_g6_hd_identity.py``.

    Returns a dict with keys ``"init"``, ``"ystar"``, ``"g"``, ``"z"``, each
    an array of shape (T, 7) -- the same state-slot layout as ``xi_draw``.
    """
    T, n = xi_draw.shape
    xi_ystar = _propagate_shock_through_F(F, n, eps_ystar, _w_ystar_injection)
    xi_g = _propagate_shock_through_F(F, n, eps_g, _w_g_injection)
    xi_z = _propagate_shock_through_F(F, n, eps_z, _w_z_injection)

    xi_init = xi_draw - xi_ystar - xi_g - xi_z
    return {"init": xi_init, "ystar": xi_ystar, "g": xi_g, "z": xi_z}


#: The 5 shock-bars gap's own IS-curve recursion decomposes into (spec §3.4).
GAP_BARS: tuple[str, ...] = ("init", "ystar", "g", "z", "is")
#: pi's own Phillips-curve recursion adds one more bar: "pc" (eps_pc never
#: touches gap, so it is not in GAP_BARS -- its gap contribution is
#: identically zero every period, by construction of the recursion below).
PI_BARS: tuple[str, ...] = GAP_BARS + ("pc",)


def gap_pi_shock_decomposition(
    F: np.ndarray,
    A: np.ndarray,
    Z: np.ndarray,
    x: np.ndarray,
    y_full: np.ndarray,
    pi_full: np.ndarray,
    eps_is: np.ndarray,
    eps_pc: np.ndarray,
    state_components: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """B2/B3: propagate each structural shock through gap's OWN AR(2)
    IS-curve recursion and pi's OWN Phillips-curve recursion (not just the
    state), so eps_is/eps_pc get an economically sensible, persistent
    contribution via the model's own AR feedback (a1/a2 for gap, b_pi for
    pi) rather than a same-period-only measurement residual. See this
    module's docstring for why the naive "gap = data - state" split omits
    this feedback entirely.

    Structural coefficients (a1, a2, a_r, b_y, b_pi) are read directly off
    ``Z``/``A`` (the exact entries ``build_lw_matrices`` stamps: ``Z[0,1] =
    -a1``, ``Z[0,2] = -a2``, ``Z[0,3] = -c*a_r/2`` with ``c=1`` fixed,
    ``Z[1,1] = -b_y``, ``A[4,1] = b_pi``) rather than a separate params
    dict, so this function only needs a draw's system matrices -- the same
    inputs the caller already has for ``simulate_smoother_draw``.

    ``state_components`` is B1's output (:func:`state_shock_decomposition`):
    the "init"/"ystar"/"g"/"z" bars' own r* = g+z contributions feed gap's
    recursion via ``rstar1 = comp[i,3] + comp[i,5]`` (g_{t-1}+z_{t-1}),
    ``rstar2 = comp[i,4] + comp[i,6]`` (g_{t-2}+z_{t-2}) -- the "is" bar has
    no state component (eps_is never enters the state), so its rstar terms
    are 0.

    ``y_full``/``pi_full`` are the FULL length-(T+4) data series (including
    the 4 pre-sample lag quarters, same convention as
    ``build_lw_regressors``' inputs) -- needed to seed the "init" bar's
    pre-sample registers. Gap's own pre-sample seed (period 0's/period -1's
    "init"-bar gap) is ``y_full[3] - state_components["init"][0, 1]`` /
    ``y_full[2] - state_components["init"][0, 2]`` -- the SMOOTHED belief
    about y*_0/y*_{-1} embedded in row 0's own state slots 1/2 (NOT the raw
    prior mean ``xi00[0]``/``xi00[1]``; see
    :func:`state_shock_decomposition`'s docstring for why that distinction
    matters here). pi's own pre-sample seed needs no such correction --
    inflation is raw exogenous data at every lag, prior or not -- so it
    reads directly off ``pi_full``. ``x`` is the (T,6) exogenous-regressor
    matrix (``build_lw_regressors``' output); columns 2/3 are
    r_{t-1}/r_{t-2}, used only by the "init" bar's data-injection term (the
    real-rate data enters the gap equation exogenously, so it belongs
    entirely to the initial-condition bar, not to any of the 5 structural
    shocks).

    Returns ``{"gap": {bar: (T,) array for bar in GAP_BARS}, "pi": {bar:
    (T,) array for bar in PI_BARS}}``. Sum-checked to 1e-6 per period by
    gate G6 (``tests/test_g6_hd_identity.py``): ``sum(gap.values()) ==
    yobs[:,0] - xi_draw[:,0]`` and ``sum(pi.values()) == yobs[:,1]``.
    """
    T = x.shape[0]
    if len(y_full) != T + 4 or len(pi_full) != T + 4:
        raise ValueError(
            f"y_full/pi_full must be length T+4 = {T + 4} (the 4 pre-sample "
            f"lag quarters plus the T estimation-sample quarters, same "
            f"convention as build_lw_regressors' inputs); got lengths "
            f"{len(y_full)}, {len(pi_full)}."
        )

    a1 = -Z[0, 1]
    a2 = -Z[0, 2]
    # a_r read off the z-lag entries (Z[0,5]/Z[0,6] = -a_r/2, no c factor --
    # build_lw_matrices only multiplies the G-lag entries Z[0,3]/Z[0,4] by
    # c), so this stays correct regardless of c; the c==1.0 assertion below
    # is what actually needs c to be 1, not this extraction.
    a_r = -2.0 * Z[0, 5]
    b_y = -Z[1, 1]
    b_pi = A[4, 1]
    # rstar1/rstar2 below are computed as the UNWEIGHTED sum g+z (c=1's
    # r*=g+z), matching build_lw_matrices' c=1.0 fixed default everywhere in
    # current scope (estimate_c is hard-validated False --
    # specs/schema/lw_sv.py). Numerics-reviewer finding (S4): if
    # estimate_c ever becomes true, Z[0,3]/Z[0,4] (the g lags) would carry a
    # c != 1 factor the z lags don't, and a naive unweighted g+z sum would
    # silently corrupt the historical decomposition (verified numerically:
    # c=1.5 produces a gap-HD reconstruction error of ~0.36 against G6's
    # 1e-6 tolerance) -- fail loudly instead of silently.
    c_check = Z[0, 3] / Z[0, 5] if Z[0, 5] != 0.0 else 1.0
    if not np.isclose(c_check, 1.0, atol=1e-9):
        raise NotImplementedError(
            f"gap_pi_shock_decomposition assumes c == 1.0 (spec §1.3's "
            f"default; estimate_c is not implemented anywhere in current "
            f"scope), but the system matrices imply c = {c_check!r}. The "
            f"rstar1/rstar2 computation below sums g+z unweighted, which is "
            f"only correct for c == 1.0 -- generalize it (weight the g "
            f"terms by c) before removing this guard."
        )

    gap = {k: np.zeros(T) for k in GAP_BARS}
    pi = {k: np.zeros(T) for k in PI_BARS}

    gap_lag1 = {k: 0.0 for k in GAP_BARS}
    gap_lag2 = {k: 0.0 for k in GAP_BARS}
    # Period 0's / period -1's gap, seeding the "init" bar's own recursion.
    # Uses state_components["init"]'s row 0, slots 1/2 -- the SMOOTHED
    # belief about y*_0/y*_{-1} (row 0 = period 1's state; slot 1 = y*_{t-1}
    # = y*_0, slot 2 = y*_{t-2} = y*_{-1}) -- NOT the raw prior mean
    # xi00[0]/xi00[1]. This matters: xi00/P00 is only a PRIOR for period 0's
    # state, and the RTS smoother genuinely revises that prior using the
    # whole sample (the same reason state_shock_decomposition's "init" bar
    # is a residual, not an F@xi00 forward path -- see its docstring). Since
    # eps_ystar/eps_g/eps_z contribute exactly 0 to state_components["init"]
    # row 0's slots 1/2 (verified: build_lw_matrices' Q only ever injects
    # into slots 0, 3, 5), state_components["init"][0, 1]/[0, 2] IS the
    # smoothed y*_0/y*_{-1} belief, entirely attributable to the init bar.
    # Using xi00 directly here (a first draft's choice, matching a literal
    # reading of "gap_lag1['init'] = y[3] - xi00[0]") breaks the B2 sum
    # identity for the ENTIRE sample, not just period 0/1: the resulting
    # one-period seed error propagates forward through gap's own AR(2)
    # feedback and does not decay away within a typical sample length
    # (empirically confirmed while developing this module -- verify by
    # reverting to xi00 and re-running tests/test_g6_hd_identity.py).
    gap_lag1["init"] = float(y_full[3] - state_components["init"][0, 1])
    gap_lag2["init"] = float(y_full[2] - state_components["init"][0, 2])

    pi_lag1 = {k: 0.0 for k in PI_BARS}
    pi_lag2 = {k: 0.0 for k in PI_BARS}
    pi_lag3 = {k: 0.0 for k in PI_BARS}
    pi_lag4 = {k: 0.0 for k in PI_BARS}
    pi_lag1["init"] = float(pi_full[3])
    pi_lag2["init"] = float(pi_full[2])
    pi_lag3["init"] = float(pi_full[1])
    pi_lag4["init"] = float(pi_full[0])

    for i in range(T):
        # --- B2: this period's gap, for each of the 5 gap bars ---
        gap_t: dict[str, float] = {}
        for k in GAP_BARS:
            if k == "is":
                rstar1 = rstar2 = 0.0
            else:
                comp = state_components[k]
                rstar1 = comp[i, 3] + comp[i, 5]
                rstar2 = comp[i, 4] + comp[i, 6]
            val = a1 * gap_lag1[k] + a2 * gap_lag2[k] - (a_r / 2.0) * (rstar1 + rstar2)
            if k == "is":
                val += eps_is[i]
            if k == "init":
                val += (a_r / 2.0) * (x[i, 2] + x[i, 3])
            gap_t[k] = val

        # Snapshot gap_lag1 BEFORE this iteration's B2 register shift: pi_t's
        # equation uses gap_{t-1}, which is exactly what gap_lag1 holds right
        # now (walking into this iteration), not this iteration's freshly
        # computed gap_t.
        gap_lag1_pre = dict(gap_lag1)

        # --- B3: this period's pi, for each of the 6 pi bars ---
        pi_t: dict[str, float] = {}
        for k in PI_BARS:
            pibar = (pi_lag2[k] + pi_lag3[k] + pi_lag4[k]) / 3.0
            gap_term = gap_lag1_pre[k] if k in gap_lag1_pre else 0.0
            val = b_pi * pi_lag1[k] + (1.0 - b_pi) * pibar + b_y * gap_term
            if k == "pc":
                val += eps_pc[i]
            pi_t[k] = val

        # --- record + shift registers ---
        for k in GAP_BARS:
            gap[k][i] = gap_t[k]
            gap_lag2[k] = gap_lag1[k]
            gap_lag1[k] = gap_t[k]

        for k in PI_BARS:
            pi[k][i] = pi_t[k]
            pi_lag4[k] = pi_lag3[k]
            pi_lag3[k] = pi_lag2[k]
            pi_lag2[k] = pi_lag1[k]
            pi_lag1[k] = pi_t[k]

    return {"gap": gap, "pi": pi}


def y_level_decomposition(gap: dict[str, np.ndarray], state_components: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """B4: y-level bars, ``y = gap + y*`` decomposed additively across the 6
    bars (``PI_BARS``' full set, since y needs a "pc" bar too, always
    identically zero -- eps_pc never touches y):

        y_k = gap_k + state_components[k][:, 0]   for k in {init, ystar, g, z}
        y_is = gap_is                              # eps_is's only channel into y is via gap
        y_pc = 0

    Sum-check (exact, no tolerance needed beyond floating point): ``sum(
    y.values()) == yobs[:,0]`` -- this follows immediately from B1's and
    B2's own (already sum-checked) identities, so it needs no separate
    numerical argument.
    """
    y: dict[str, np.ndarray] = {}
    for k in ("init", "ystar", "g", "z"):
        y[k] = gap[k] + state_components[k][:, 0]
    y["is"] = gap["is"].copy()
    y["pc"] = np.zeros_like(gap["is"])
    return y


def four_quarter_growth(y_bars: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """B4: 4-quarter growth per bar, ``growth_t = y_t - y_{t-4}`` for row
    index ``i >= 4`` (period t >= 5); the first 4 rows are ``NaN`` (growth
    needs 4 quarters of history that would otherwise reach outside the
    estimation sample -- a normal, documented convention, not extended into
    the pre-sample window)."""
    growth: dict[str, np.ndarray] = {}
    for k, arr in y_bars.items():
        g = np.full(arr.shape[0], np.nan)
        g[4:] = arr[4:] - arr[:-4]
        growth[k] = g
    return growth


@dataclass
class HDDraw:
    """One draw's full historical-decomposition bundle (spec §3.4):

    - ``state_components``: B1's 4-key dict (``"init"``/``"ystar"``/``"g"``/
      ``"z"``), each (T, 7) -- sums to ``xi_draw``.
    - ``gap``: B2's 5-key dict (``GAP_BARS``), each (T,) -- sums to
      ``yobs[:,0] - xi_draw[:,0]``.
    - ``pi``: B3's 6-key dict (``PI_BARS``), each (T,) -- sums to
      ``yobs[:,1]``.
    - ``y``: B4's 6-key dict (``PI_BARS``), each (T,) -- sums to
      ``yobs[:,0]``.
    - ``y_growth_4q``: B4's 6-key dict, each (T,) with the first 4 entries
      ``NaN`` -- sums (where defined) to ``yobs[4:,0] - yobs[:-4,0]``.
    """

    state_components: dict[str, np.ndarray]
    gap: dict[str, np.ndarray]
    pi: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    y_growth_4q: dict[str, np.ndarray]


def historical_decomposition_draw(
    F: np.ndarray,
    A: np.ndarray,
    Z: np.ndarray,
    x: np.ndarray,
    y_full: np.ndarray,
    pi_full: np.ndarray,
    xi_draw: np.ndarray,
    eps_ystar: np.ndarray,
    eps_g: np.ndarray,
    eps_z: np.ndarray,
    eps_is: np.ndarray,
    eps_pc: np.ndarray,
) -> HDDraw:
    """Run B1-B4 for one simulation-smoother draw (this draw's own recovered
    state path + shocks + system matrices) and package the result as an
    :class:`HDDraw`. See :func:`state_shock_decomposition`,
    :func:`gap_pi_shock_decomposition`, :func:`y_level_decomposition`,
    :func:`four_quarter_growth` for the individual steps. No ``xi00``
    parameter -- neither B1 nor B2/B3 needs it (see their own docstrings
    for why the initial-condition "bar" is instead a residual against the
    already-smoothed ``xi_draw``)."""
    state_components = state_shock_decomposition(F, xi_draw, eps_ystar, eps_g, eps_z)
    gp = gap_pi_shock_decomposition(F, A, Z, x, y_full, pi_full, eps_is, eps_pc, state_components)
    y_bars = y_level_decomposition(gp["gap"], state_components)
    growth = four_quarter_growth(y_bars)
    return HDDraw(state_components=state_components, gap=gp["gap"], pi=gp["pi"], y=y_bars, y_growth_4q=growth)


@dataclass
class HDDrawsAggregated:
    """Per-draw historical-decomposition bars (spec §3.4), stacked over the
    posterior draws selected by ``outputs.smoother_draws`` -- same shape
    convention as :class:`TrendCycleDraws` (one ``(n_draws, T)`` array per
    bar, rather than :class:`HDDraw`'s one-draw-at-a-time bundle).

    ``gap``/``pi``/``y``/``y_growth_4q`` mirror :class:`HDDraw`'s own
    per-bar dicts (``GAP_BARS`` for ``gap``, ``PI_BARS`` for the other
    three -- see that class's docstring), with each bar's value now an
    ``(n_draws, T)`` array instead of a ``(T,)`` one.  ``state_components``
    likewise mirrors :class:`HDDraw`'s 4-key dict (``"init"``/``"ystar"``/
    ``"g"``/``"z"``), each an ``(n_draws, T, 7)`` array -- kept for
    completeness/debugging even though the report (spec §3.4) only needs
    gap/pi/y/y_growth_4q.

    No aggregation/percentile/median math here -- that is ``plots.py``'s
    job (spec §3.4: "Stacked bars (posterior-median contributions)").
    """

    draw_indices: np.ndarray  # (n_draws,) -- indices into the flattened posterior this batch used
    dates: pd.DatetimeIndex  # (T,)
    gap: dict[str, np.ndarray]  # bar -> (n_draws, T), GAP_BARS
    pi: dict[str, np.ndarray]  # bar -> (n_draws, T), PI_BARS
    y: dict[str, np.ndarray]  # bar -> (n_draws, T), PI_BARS
    y_growth_4q: dict[str, np.ndarray]  # bar -> (n_draws, T), PI_BARS (first 4 cols NaN)
    state_components: dict[str, np.ndarray]  # "init"/"ystar"/"g"/"z" -> (n_draws, T, 7)


def compute_historical_decomposition_draws(lw_run: LWRun, *, seed: int | None = None) -> HDDrawsAggregated:
    """Run :func:`historical_decomposition_for_draw` once per selected
    posterior draw (``lw_run.spec.outputs.smoother_draws``) and stack each
    bar's per-draw series into ``(n_draws, T)`` arrays -- the same
    loop-and-stack pattern :func:`compute_trend_cycle_draws` /
    :func:`compute_irf_draws` / :func:`compute_fan_draws` each already use.
    ``seed`` defaults to ``lw_run.spec.sampler.seed``; a single
    ``np.random.Generator`` is advanced sequentially across all selected
    draws (matching those functions' own usage pattern), so this batch's
    simulation-smoother draws are NOT identical to a same-seeded single-draw
    call to :func:`historical_decomposition_for_draw` beyond the first draw
    in the sequence.
    """
    flat = _flatten_posterior(lw_run)
    n_total = flat["a1"].shape[0]
    idx = select_draw_indices(n_total, lw_run.spec.outputs.smoother_draws)

    T = lw_run.yobs.shape[0]
    n = len(idx)

    gap = {k: np.empty((n, T)) for k in GAP_BARS}
    pi = {k: np.empty((n, T)) for k in PI_BARS}
    y = {k: np.empty((n, T)) for k in PI_BARS}
    y_growth_4q = {k: np.empty((n, T)) for k in PI_BARS}
    state_components = {k: np.empty((n, T, 7)) for k in ("init", "ystar", "g", "z")}

    rng = np.random.default_rng(seed if seed is not None else lw_run.spec.sampler.seed)

    for j, i in enumerate(idx):
        hd = historical_decomposition_for_draw(lw_run, flat, int(i), rng)
        for k in GAP_BARS:
            gap[k][j] = hd.gap[k]
        for k in PI_BARS:
            pi[k][j] = hd.pi[k]
            y[k][j] = hd.y[k]
            y_growth_4q[k][j] = hd.y_growth_4q[k]
        for k in ("init", "ystar", "g", "z"):
            state_components[k][j] = hd.state_components[k]

    return HDDrawsAggregated(
        draw_indices=idx,
        dates=lw_run.dates,
        gap=gap,
        pi=pi,
        y=y,
        y_growth_4q=y_growth_4q,
        state_components=state_components,
    )


def historical_decomposition_for_draw(
    lw_run: LWRun,
    flat: dict[str, np.ndarray],
    draw_index: int,
    rng: np.random.Generator,
) -> HDDraw:
    """Glue: run the DK simulation smoother for one posterior draw of a
    loaded run (``lw_run``/``flat`` from :func:`load_lw_run`/
    :func:`_flatten_posterior`) and feed its recovered state + shocks into
    :func:`historical_decomposition_draw`. ``flat`` is passed in (not
    recomputed) so a caller looping over many draws for the HD's own
    posterior-median aggregation pays the ``_flatten_posterior`` cost once."""
    from macrotoolkit.smoother import simulate_smoother_draw

    F, Q, A, Z, R, _h_is, _h_pc = _system_matrices_for_draw(flat, draw_index, lw_run.sv_on)
    sim = simulate_smoother_draw(lw_run.yobs, lw_run.x, F, Q, A, Z, R, lw_run.xi00, lw_run.P00, rng)
    return historical_decomposition_draw(
        F, A, Z, lw_run.x, lw_run.y_full, lw_run.pi_full, sim.xi_draw,
        sim.eps_ystar, sim.eps_g, sim.eps_z, sim.eps_is, sim.eps_pc,
    )


# ---------------------------------------------------------------------------
# Part C: IRF matrix (spec §3.2)
# ---------------------------------------------------------------------------

#: The 5 structural shocks (rows of the IRF grid) -- spec §1.4's exhaustive,
#: named list. No selection needed (unlike the responses below): every shock
#: gets a row.
IRF_SHOCKS: tuple[str, ...] = ("ystar", "g", "z", "is", "pc")

#: The 5 named responses (columns of the IRF grid) -- plans/S4-plan.md's
#: "open question 3", user-confirmed 2026-08-31: gap, pi (the two
#: measurement-equation observables besides y itself), r* (=g+z, c=1, the
#: trend real-rate aggregate), y (level, cumulative impulse -- distinguishes
#: permanent from transitory shocks), g (trend growth's own path, the
#: natural "own-response" column for the g shock).
IRF_RESPONSES: tuple[str, ...] = ("gap", "pi", "rstar", "y", "g")


def irf_shock_size(
    shock: str,
    flat: dict[str, np.ndarray],
    i: int,
    sv_on: bool,
    irf_vol_reference: str,
    h_is: np.ndarray | None,
    h_pc: np.ndarray | None,
) -> float:
    """One posterior draw's "one-standard-deviation shock at the reference
    volatility" size (spec §3.2) for one of :data:`IRF_SHOCKS`.

    - ``shock in {"ystar", "g", "z"}``: that draw's own constant posterior
      scale, ``sigma_ystar``/``sigma_g``/``sigma_z`` -- these 3 shocks are
      NEVER SV shocks in this model (spec §1.4's table restricts SV to
      IS/PC only), so there is no "reference volatility" question for them.
    - ``shock in {"is", "pc"}``, no-SV run: that draw's own constant
      ``sigma_is``/``sigma_pc`` (SV is off, so there is nothing to
      reference a point on).
    - ``shock in {"is", "pc"}``, SV run: ``exp(h_ref / 2)`` (h is
      log-VARIANCE, spec §1.5 -- sd is ``exp(h/2)``, never ``exp(h)``),
      where ``h_ref`` is that draw's own ``h_is``/``h_pc`` path evaluated
      at ``irf_vol_reference``: ``"end_of_sample"`` -> ``h[-1]``,
      ``"sample_mean"`` -> ``h.mean()``. NOTE the ``"sample_mean"`` case
      takes the mean of the log-variance path ITSELF and only then converts
      to a standard deviation -- i.e. ``exp(mean(h) / 2)``, deliberately
      NOT ``mean(exp(h / 2))`` (the mean of the already-converted sd path).
      Both are defensible "reference volatility" conventions and only one
      can be picked; this module picks "average the log-variance, then
      convert once" because it reads most naturally as ONE reference h
      (spec §3.2's own phrase), converted to a shock size a single time --
      not an average of T already-converted numbers.
    """
    if shock in ("ystar", "g", "z"):
        return float(flat[f"sigma_{shock}"][i])
    if shock not in ("is", "pc"):
        raise ValueError(
            f"irf_shock_size: shock must be one of {IRF_SHOCKS!r}; got {shock!r}."
        )
    if not sv_on:
        return float(flat[f"sigma_{shock}"][i])
    h = h_is if shock == "is" else h_pc
    if h is None:
        raise ValueError(
            f"irf_shock_size: sv_on=True but h_{shock} is None -- the caller "
            f"must pass this draw's own saved h_{shock} path (from "
            f"_system_matrices_for_draw), not None."
        )
    if irf_vol_reference == "end_of_sample":
        h_ref = float(h[-1])
    elif irf_vol_reference == "sample_mean":
        h_ref = float(np.mean(h))
    else:
        raise ValueError(
            f"irf_shock_size: outputs.irf_vol_reference must be "
            f"'end_of_sample' or 'sample_mean'; got {irf_vol_reference!r}."
        )
    return float(np.exp(h_ref / 2.0))


def impulse_response_for_shock(
    F: np.ndarray,
    A: np.ndarray,
    Z: np.ndarray,
    shock: str,
    shock_size: float,
    horizon: int,
) -> dict[str, np.ndarray]:
    """The theoretical impulse response of ONE structural shock (spec §3.2),
    at ONE posterior draw's system matrices, for ``horizon`` periods.

    Structurally, an impulse response is exactly a historical decomposition
    with (a) no real data, (b) a single synthetic impulse instead of a real
    recovered shock path, and (c) no "init" bar at all -- gap/pi start
    entirely at rest (all lag registers zero) and there is no prior/smoothed
    belief about a pre-sample state to carry forward (Part B's "init" bar
    only ever exists because a REAL run has a genuine smoothed initial
    condition; a synthetic impulse response has none by construction). This
    function builds exactly that: a zero "init" state-component bar, the 3
    trend shocks' state contributions built the SAME WAY B1's
    :func:`state_shock_decomposition` does (via the shared
    :func:`_propagate_shock_through_F` helper, fed a synthetic impulse array
    -- all zero except ``shock_size`` at index 0 -- instead of a recovered
    shock path), and gap/pi's response read directly off
    :func:`gap_pi_shock_decomposition`'s own AR-recursion arithmetic, called
    on all-zero real data (``x``, ``y_full``, ``pi_full``) so its "init" bar
    is identically zero throughout (the SAME B1/B2 sum-identity logic gate
    G6 already validates: with zero real data and a zero "init"
    state-component, ``gap_lag1["init"] = y_full[3] - 0 = 0``, and it stays
    zero every period) plus a synthetic ``eps_is``/``eps_pc`` impulse array
    (measurement shocks that never enter the state at all -- their only
    channel is gap/pi's own AR feedback).

    ``shock`` must be one of :data:`IRF_SHOCKS`; ``shock_size`` is the
    one-off impulse applied at period 0 (spec §3.2's "one-standard-deviation
    shock at the reference volatility" -- computed by :func:`irf_shock_size`,
    a caller concern, not this function's).

    Returns :data:`IRF_RESPONSES` (``{"gap", "pi", "rstar", "y", "g"}``),
    each a length-``horizon`` array -- the EXACT SAME reporting formulas
    used elsewhere in this module (``compute_trend_cycle_draws``):
    ``rstar = state[:, 3] + state[:, 5]``, ``y = gap + state[:, 0]``,
    ``g = state[:, 3]``.
    """
    if shock not in IRF_SHOCKS:
        raise ValueError(
            f"impulse_response_for_shock: shock must be one of {IRF_SHOCKS!r}; got {shock!r}."
        )
    n = F.shape[0]

    eps_ystar = np.zeros(horizon)
    eps_g = np.zeros(horizon)
    eps_z = np.zeros(horizon)
    eps_is = np.zeros(horizon)
    eps_pc = np.zeros(horizon)
    {"ystar": eps_ystar, "g": eps_g, "z": eps_z, "is": eps_is, "pc": eps_pc}[shock][0] = shock_size

    state_components = {
        "init": np.zeros((horizon, n)),
        "ystar": _propagate_shock_through_F(F, n, eps_ystar, _w_ystar_injection),
        "g": _propagate_shock_through_F(F, n, eps_g, _w_g_injection),
        "z": _propagate_shock_through_F(F, n, eps_z, _w_z_injection),
    }

    x_zero = np.zeros((horizon, 6))
    y_full_zero = np.zeros(horizon + 4)
    pi_full_zero = np.zeros(horizon + 4)
    gp = gap_pi_shock_decomposition(
        F, A, Z, x_zero, y_full_zero, pi_full_zero, eps_is, eps_pc, state_components
    )

    state_total = (
        state_components["init"] + state_components["ystar"] + state_components["g"] + state_components["z"]
    )
    gap_response = gp["gap"][shock] if shock in GAP_BARS else np.zeros(horizon)
    pi_response = gp["pi"][shock]
    rstar_response = state_total[:, 3] + state_total[:, 5]
    g_response = state_total[:, 3]
    y_response = gap_response + state_total[:, 0]

    return {
        "gap": gap_response,
        "pi": pi_response,
        "rstar": rstar_response,
        "y": y_response,
        "g": g_response,
    }


@dataclass
class IRFDraws:
    """The full 5x5 IRF grid (spec §3.2), stacked over the posterior draws
    selected by ``outputs.smoother_draws``.

    ``responses[shock][response]`` is an ``(n_draws, irf_horizon)`` array of
    per-draw impulse-response paths, for ``shock in IRF_SHOCKS`` and
    ``response in IRF_RESPONSES``. No median/68%/90% banding here -- that is
    ``plots.py``'s job (same division of labor as ``compute_trend_cycle_draws``).
    """

    draw_indices: np.ndarray  # (n_draws,) -- indices into the flattened posterior this batch used
    horizon: int  # = outputs.irf_horizon
    responses: dict[str, dict[str, np.ndarray]]


def compute_irf_draws(lw_run: LWRun) -> IRFDraws:
    """Build spec §3.2's 5x5 IRF grid across the posterior draws selected by
    ``lw_run.spec.outputs.smoother_draws``, at horizon
    ``outputs.irf_horizon`` and reference volatility
    ``outputs.irf_vol_reference``.

    Purely a function of each selected draw's own system matrices
    (``F``, ``A``, ``Z``) and its own ``sigma_*``/``h_*`` posterior values --
    unlike Part A/B/D, this needs no simulation smoother pass and no RNG (an
    impulse response is deterministic given a shock size, spec §3.2:
    "computed analytically from posterior draws of (Z, T, R)").
    """
    flat = _flatten_posterior(lw_run)
    n_total = flat["a1"].shape[0]
    idx = select_draw_indices(n_total, lw_run.spec.outputs.smoother_draws)
    horizon = lw_run.spec.outputs.irf_horizon
    vol_ref = lw_run.spec.outputs.irf_vol_reference

    n = len(idx)
    responses: dict[str, dict[str, np.ndarray]] = {
        shock: {resp: np.empty((n, horizon)) for resp in IRF_RESPONSES} for shock in IRF_SHOCKS
    }

    for j, i in enumerate(idx):
        i = int(i)
        F, Q, A, Z, R, h_is, h_pc = _system_matrices_for_draw(flat, i, lw_run.sv_on)
        for shock in IRF_SHOCKS:
            size = irf_shock_size(shock, flat, i, lw_run.sv_on, vol_ref, h_is, h_pc)
            resp = impulse_response_for_shock(F, A, Z, shock, size, horizon)
            for name in IRF_RESPONSES:
                responses[shock][name][j] = resp[name]

    return IRFDraws(draw_indices=idx, horizon=horizon, responses=responses)


# ---------------------------------------------------------------------------
# Part D: fan charts (spec §3.3)
# ---------------------------------------------------------------------------


def _extract_gap_pi_coeffs(Z: np.ndarray, A: np.ndarray) -> tuple[float, float, float, float, float]:
    """The 5 structural coefficients (a1, a2, a_r, b_y, b_pi) read directly
    off a draw's ``Z``/``A`` system matrices -- the SAME extraction
    :func:`gap_pi_shock_decomposition` documents and performs internally
    (``Z[0,1] = -a1``, ``Z[0,2] = -a2``, ``Z[0,5] = -a_r/2`` -- no ``c``
    factor, since ``a_r`` is read off the z-lag entries, matching that
    function's own comment -- ``Z[1,1] = -b_y``, ``A[4,1] = b_pi``).

    Deliberately duplicated here rather than imported out of
    ``gap_pi_shock_decomposition`` (which is not touched by this module's S4
    fan-chart addition at all, per the task brief's "do not modify
    gap_pi_shock_decomposition ... behavior" constraint): the fan-chart
    forward simulation below needs the SAME 5 numbers for its own one-step
    AR recursion, but is not itself calling that already-G6-validated
    function's shock-decomposition machinery, so re-deriving the 5 numbers
    from ``Z``/``A`` directly (rather than threading a private helper
    through an unrelated, validated function) keeps the two call sites
    independent.
    """
    a1 = -Z[0, 1]
    a2 = -Z[0, 2]
    a_r = -2.0 * Z[0, 5]
    b_y = -Z[1, 1]
    b_pi = A[4, 1]
    return a1, a2, a_r, b_y, b_pi


def _fan_forecast_step(
    gap_lag1: float,
    gap_lag2: float,
    pi_lag1: float,
    pi_lag2: float,
    pi_lag3: float,
    pi_lag4: float,
    rate_gap_lag1: float,
    rate_gap_lag2: float,
    a1: float,
    a2: float,
    a_r: float,
    b_pi: float,
    b_y: float,
    eps_is_t: float,
    eps_pc_t: float,
) -> tuple[float, float]:
    """One forward forecast period's gap/pi AR recursion (spec §3.3) -- the
    SAME underlying IS-curve/Phillips-curve arithmetic
    :func:`gap_pi_shock_decomposition` applies per bar (spec §1.2), just
    COMBINED rather than decomposed bar-by-bar (a fan-chart draw is a single
    stochastic realization, not a shock-by-shock attribution)::

        gap_t = a1*gap_lag1 + a2*gap_lag2
                + (a_r/2)*(rate_gap_lag1 + rate_gap_lag2) + eps_is_t
        pi_t  = b_pi*pi_lag1 + (1-b_pi)*mean(pi_lag2, pi_lag3, pi_lag4)
                + b_y*gap_lag1 + eps_pc_t

    ``rate_gap_lag1``/``rate_gap_lag2`` are ``(r - r*)`` at the two prior
    periods (whatever those periods' own convention: real data if still
    in-sample, or ``forecast_r_rule``'s convention if already a forecast
    period -- the caller's concern, not this function's).

    **The IS-curve term is a PLUS, not a minus** (numerics-reviewer finding,
    S4, confirmed independently against ``build_lw_matrices``'s own
    construction): ``gap_pi_shock_decomposition``'s per-bar recursion uses
    ``-(a_r/2)*(rstar1+rstar2)`` for EVERY bar plus a SEPARATE
    ``+(a_r/2)*(r_{t-1}+r_{t-2})`` term added ONLY to the "init" bar (the
    real-rate DATA injection) -- summed across all bars, those two pieces
    combine into ``+(a_r/2)*[(r_{t-1}-r*_{t-1})+(r_{t-2}-r*_{t-2})]``, a
    PLUS overall. This function combines ``r`` and ``r*`` into ONE
    ``rate_gap`` number up front (no separate data-injection term to
    provide the offsetting plus), so it needs its OWN plus sign applied
    directly -- an earlier version of this function kept the minus that
    was only ever valid for the r*-only HALF of that split, which
  produced up to a ~100%+ relative distortion of forecast gap values
    under ``forecast_r_rule: last_value`` by the end of a 12-period
    horizon (confirmed numerically before this fix).

    ``pi_t`` uses ``gap_lag1`` (gap ONE period before pi_t's own period,
    spec §1.2's Phillips curve), NOT the just-computed ``gap_t`` -- matching
    ``gap_pi_shock_decomposition``'s own "snapshot gap_lag1 BEFORE this
    iteration's register shift" note.

    Returns ``(gap_t, pi_t)``.
    """
    gap_t = a1 * gap_lag1 + a2 * gap_lag2 + (a_r / 2.0) * (rate_gap_lag1 + rate_gap_lag2) + eps_is_t
    pibar_t = (pi_lag2 + pi_lag3 + pi_lag4) / 3.0
    pi_t = b_pi * pi_lag1 + (1.0 - b_pi) * pibar_t + b_y * gap_lag1 + eps_pc_t
    return gap_t, pi_t


def simulate_fan_draw(
    F: np.ndarray,
    Q: np.ndarray,
    a1: float,
    a2: float,
    a_r: float,
    b_pi: float,
    b_y: float,
    xi_last: np.ndarray,
    gap_lag1: float,
    gap_lag2: float,
    pi_lag1: float,
    pi_lag2: float,
    pi_lag3: float,
    pi_lag4: float,
    rate_gap_seed: float,
    y_hist_last4: np.ndarray,
    r_last: float,
    forecast_r_rule: str,
    sv_on: bool,
    sigma_is: float | None,
    sigma_pc: float | None,
    h_is_last: float | None,
    h_pc_last: float | None,
    sigma_h_is: float | None,
    sigma_h_pc: float | None,
    horizon: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """One posterior draw's ONE stochastic fan-chart forward simulation
    (spec §3.3), ``horizon`` periods ahead of that draw's own terminal
    simulation-smoother state ``xi_last`` (= ``sim.xi_draw[-1]``, absolute
    period T), seeded by that SAME draw's ACTUAL last few periods' real
    gap/pi lag registers -- ``gap_lag1``/``gap_lag2`` =
    ``yobs[-1,0]-xi_draw[-1,0]`` / ``yobs[-2,0]-xi_draw[-2,0]``,
    ``pi_lag1..4`` = ``yobs[-1:-5:-1,1]``.

    ``rate_gap_seed`` is ``(r-r*)_{T-1}`` -- the ONLY ``(r-r*)`` value
    derivable BEFORE this function's own forward loop starts (numerics-
    reviewer finding, S4: an earlier draft tried to also seed
    ``(r-r*)_T`` from ``r_full[-1] - (xi_draw[-1,3]+xi_draw[-1,5])``, but
    ``xi_draw[-1,3]``/``[-1,5]`` are ``g_{T-1}``/``z_{T-1}`` -- the state's
    slot-3/5 one-period lag (this module's/``smoother.py``'s own
    convention) means row T's OWN state slots never show ``g_T``/``z_T``;
    those only become visible one ``F``-step later, which requires THIS
    period's own fresh process-noise draw. So ``(r-r*)_T`` cannot be known
    ahead of time -- it is computed INSIDE the loop below, in the SAME
    iteration that needs it as ``rate_gap_lag1``, not deferred to the next
    iteration as an earlier version of this function did (confirmed wrong
    by direct numerical comparison against a hand-derived reference: up to
    an ~65% relative distortion of the first forecast period's gap value
    under the "neutral" rule, where the bug caused a nonzero historical
    rate-gap value to leak into what should have been a zero term)).
    ``(r-r*)_{T-1}`` has no such problem: it is read directly off row T's
    OWN state slots (``xi_draw[-1,3]+xi_draw[-1,5]`` = ``g_{T-1}+z_{T-1}``
    = ``r*_{T-1}`` exactly, by the same slot convention), paired with the
    real ``r_full[-2]`` (= ``r_{T-1}``) -- both genuinely in-sample,
    already-realized quantities, so ``forecast_r_rule`` does not apply to
    this ONE seed (the rule only governs ``(r-r*)`` from period T onward,
    computed inside the loop -- see below).

    Per period, FRESH stochastic noise is drawn: this draw's own posterior
    ``Q`` for the state (``xi_{T+h} = F @ xi_{T+h-1} + w``, ``w ~ N(0, Q)``,
    via ``macrotoolkit.smoother._psd_sqrt`` -- Q is rank-deficient, see that
    function's own docstring), and either constant ``sigma_is``/``sigma_pc``
    (no-SV) or the CONTINUED SV log-variance random walk (SV --
    ``h_{T+h} = h_{T+h-1} + sigma_h * nu``, ``nu ~ N(0,1)`` fresh every
    period, starting from this draw's own last saved ``h_is``/``h_pc``) for
    the measurement shocks. This per-period-fresh-noise draw is what makes
    an SV run's bands genuinely WIDEN with horizon (spec §3.3: "widening
    bands are the point") and what makes this a stochastic SIMULATION,
    unlike :func:`gap_pi_shock_decomposition` / :func:`impulse_response_for_shock`'s
    deterministic recursions.

    ``forecast_r_rule`` governs ``(r-r*)`` from absolute period T onward
    (every value the loop itself computes -- i.e. starting with the FIRST
    forecast period's own "t-1" input, not just periods strictly after the
    sample end; only ``(r-r*)_{T-1}``, the ``rate_gap_seed`` above, predates
    the rule): ``"neutral"`` forces ``(r_t - r*_t) := 0`` for every such t
    -- r is defined to exactly track r*; ``"last_value"`` holds
    ``r_t := r_last`` (the run's actual last observed r) while
    ``r*_t = xi_t[3] + xi_t[5]`` keeps evolving through the state's own
    random walk, so the deviation genuinely moves. Only these 2 values are
    implemented -- ``user_path`` is already rejected at the schema level
    (``specs.schema.lw_sv.LwSvOutputs``).

    ``sv_on=False``: pass ``sigma_is``/``sigma_pc`` (that draw's own
    constant posterior scales); leave ``h_is_last``/``h_pc_last``/
    ``sigma_h_is``/``sigma_h_pc`` as ``None`` -- the measurement covariance
    stays constant every period, so this channel contributes NO growing
    variance with horizon (the no-SV/SV contrast spec §3.3 calls out).
    ``sv_on=True``: the reverse (``sigma_is``/``sigma_pc`` ``None``, the 4 SV
    args real).

    Returns ``{"y_level", "y_growth_4q", "pi", "gap", "rstar", "rate_gap"}``,
    each length ``horizon``. ``rate_gap`` (the per-period ``(r-r*)`` INPUT
    actually used inside the loop -- not a reporting series in its own
    right, but a useful diagnostic) is exposed mainly so
    ``outputs.forecast_r_rule``'s convention can be checked directly rather
    than only indirectly through gap's own downstream path.
    """
    if forecast_r_rule not in ("neutral", "last_value"):
        raise ValueError(
            f"simulate_fan_draw: forecast_r_rule must be 'neutral' or "
            f"'last_value' (user_path is not implemented -- rejected at the "
            f"schema level, specs.schema.lw_sv.LwSvOutputs); got {forecast_r_rule!r}."
        )
    if sv_on:
        if h_is_last is None or h_pc_last is None or sigma_h_is is None or sigma_h_pc is None:
            raise ValueError(
                "simulate_fan_draw: sv_on=True requires h_is_last/h_pc_last/"
                "sigma_h_is/sigma_h_pc (that draw's own saved SV state) to "
                "all be real numbers, not None."
            )
    else:
        if sigma_is is None or sigma_pc is None:
            raise ValueError(
                "simulate_fan_draw: sv_on=False requires sigma_is/sigma_pc "
                "(that draw's own constant posterior scales) to be real "
                "numbers, not None."
            )

    from macrotoolkit.smoother import _psd_sqrt

    n = F.shape[0]
    sqrt_Q = _psd_sqrt(Q)

    xi_prev = np.asarray(xi_last, dtype=np.float64).copy()
    h_is_prev = h_is_last
    h_pc_prev = h_pc_last
    # (r-r*)_{T-1} -- the one seed value predating forecast_r_rule (see this
    # function's docstring). Rolled forward into rate_gap_lag1 each
    # iteration AFTER that iteration's _fan_forecast_step call, exactly
    # mirroring gap_lag1/pi_lag1's own end-of-iteration shift below.
    rate_gap_prev = rate_gap_seed

    y_level = np.empty(horizon)
    pi_path = np.empty(horizon)
    gap_path = np.empty(horizon)
    rstar_path = np.empty(horizon)
    rate_gap_path = np.empty(horizon)

    for t in range(horizon):
        w = sqrt_Q @ rng.standard_normal(n)
        xi_t = F @ xi_prev + w
        # xi_t's slots 3/5 hold g/z ONE PERIOD BEHIND xi_t's own period
        # (this module's/smoother.py's state-slot convention) -- so
        # rstar_t here is r* at the period BEFORE the one xi_t nominally
        # represents, i.e. exactly the period this loop iteration is
        # computing gap/pi FOR (t=0 -> absolute period T, feeding forecast
        # period T+1's gap equation as its "t-1" term). This is the value
        # that requires xi_t's fresh process noise to become knowable --
        # it could not have been seeded before the loop started (numerics-
        # reviewer finding, S4 -- see rate_gap_seed's docstring above).
        rstar_t = xi_t[3] + xi_t[5]

        if forecast_r_rule == "neutral":
            rate_gap_t = 0.0
        else:  # "last_value"
            rate_gap_t = r_last - rstar_t

        if sv_on:
            h_is_prev = h_is_prev + sigma_h_is * rng.standard_normal()
            h_pc_prev = h_pc_prev + sigma_h_pc * rng.standard_normal()
            var_is = float(np.exp(h_is_prev))  # h is log-VARIANCE (spec §1.5)
            var_pc = float(np.exp(h_pc_prev))
        else:
            var_is = sigma_is**2
            var_pc = sigma_pc**2

        eps_is_t = rng.normal(0.0, np.sqrt(var_is))
        eps_pc_t = rng.normal(0.0, np.sqrt(var_pc))

        # rate_gap_t (just computed above, for THIS iteration's own period)
        # is used immediately as "lag1" -- not deferred to the next
        # iteration, which was the confirmed bug this fix corrects.
        gap_t, pi_t = _fan_forecast_step(
            gap_lag1, gap_lag2, pi_lag1, pi_lag2, pi_lag3, pi_lag4,
            rate_gap_t, rate_gap_prev, a1, a2, a_r, b_pi, b_y, eps_is_t, eps_pc_t,
        )

        y_level[t] = gap_t + xi_t[0]
        pi_path[t] = pi_t
        gap_path[t] = gap_t
        rstar_path[t] = rstar_t
        rate_gap_path[t] = rate_gap_t

        xi_prev = xi_t
        gap_lag2, gap_lag1 = gap_lag1, gap_t
        pi_lag4, pi_lag3, pi_lag2, pi_lag1 = pi_lag3, pi_lag2, pi_lag1, pi_t
        rate_gap_prev = rate_gap_t

    y_hist_last4 = np.asarray(y_hist_last4, dtype=np.float64)
    y_growth_4q = np.empty(horizon)
    for t in range(horizon):
        # growth at forecast period T+(t+1) needs y at T+(t+1)-4 = T+(t-3):
        # for t <= 3 that period is still in-sample (t=0 -> T-3, ...,
        # t=3 -> T), read from the real y_hist_last4 seed; for t >= 4 it is
        # itself an earlier forecast period, already computed in y_level.
        ref = y_hist_last4[t] if t <= 3 else y_level[t - 4]
        y_growth_4q[t] = y_level[t] - ref

    return {
        "y_level": y_level,
        "y_growth_4q": y_growth_4q,
        "pi": pi_path,
        "gap": gap_path,
        "rstar": rstar_path,
        "rate_gap": rate_gap_path,
    }


@dataclass
class FanDraws:
    """Per-draw stochastic fan-chart forward simulations (spec §3.3), each
    ``horizon`` = ``outputs.horizon`` quarters ahead of the sample end,
    stacked over the posterior draws selected by ``outputs.smoother_draws``.

    ONE combined stochastic realization per draw (not a shock-by-shock
    decomposition, unlike Part B/C) -- aggregating draws into the
    10/20/.../90 percentile fan bands (spec §3.3) is ``plots.py``'s job, not
    this module's (same division of labor as ``compute_trend_cycle_draws``).
    ``rate_gap`` is exposed as a diagnostic (see :func:`simulate_fan_draw`).
    """

    draw_indices: np.ndarray  # (n_draws,)
    horizon: int  # = outputs.horizon
    y_level: np.ndarray  # (n_draws, horizon)
    y_growth_4q: np.ndarray  # (n_draws, horizon)
    pi: np.ndarray  # (n_draws, horizon)
    gap: np.ndarray  # (n_draws, horizon)
    rstar: np.ndarray  # (n_draws, horizon)
    rate_gap: np.ndarray  # (n_draws, horizon) -- diagnostic, see simulate_fan_draw


def compute_fan_draws(lw_run: LWRun, *, seed: int | None = None) -> FanDraws:
    """Run the DK simulation smoother once per selected posterior draw
    (``lw_run.spec.outputs.smoother_draws``) to get that draw's terminal
    state + real terminal lag registers, then :func:`simulate_fan_draw` for
    ``outputs.horizon`` periods forward at ``outputs.forecast_r_rule``'s
    convention (spec §3.3). ``seed`` defaults to
    ``lw_run.spec.sampler.seed``; a single ``np.random.Generator`` is
    advanced sequentially across every selected draw's smoother pass AND its
    forward simulation (matching :func:`compute_trend_cycle_draws`'s own
    usage pattern).
    """
    from macrotoolkit.smoother import simulate_smoother_draw

    T = lw_run.yobs.shape[0]
    if T < 4:
        raise ValueError(
            f"compute_fan_draws needs at least 4 estimation-sample quarters "
            f"to seed pi's 4-lag register and y's 4Q-growth history; got "
            f"T={T} for run {lw_run.run_dir}."
        )

    flat = _flatten_posterior(lw_run)
    n_total = flat["a1"].shape[0]
    idx = select_draw_indices(n_total, lw_run.spec.outputs.smoother_draws)
    horizon = lw_run.spec.outputs.horizon
    rule = lw_run.spec.outputs.forecast_r_rule

    if lw_run.sv_on:
        post = lw_run.idata.posterior
        n_post_total = post.sizes["chain"] * post.sizes["draw"]
        for name in ("sigma_h_is", "sigma_h_pc"):
            if name not in post.data_vars:
                raise ValueError(
                    f"Posterior in {lw_run.run_dir} has no variable {name!r} "
                    f"-- expected for an SV lw_sv run's fan-chart SV "
                    f"random-walk continuation (spec §3.3). Available "
                    f"variables: {sorted(post.data_vars)}."
                )
        sigma_h_is_flat = np.asarray(post["sigma_h_is"].values).reshape(n_post_total)
        sigma_h_pc_flat = np.asarray(post["sigma_h_pc"].values).reshape(n_post_total)

    n = len(idx)
    y_level = np.empty((n, horizon))
    y_growth_4q = np.empty((n, horizon))
    pi_arr = np.empty((n, horizon))
    gap_arr = np.empty((n, horizon))
    rstar_arr = np.empty((n, horizon))
    rate_gap_arr = np.empty((n, horizon))

    rng = np.random.default_rng(seed if seed is not None else lw_run.spec.sampler.seed)

    for j, i in enumerate(idx):
        i = int(i)
        F, Q, A, Z, R, h_is, h_pc = _system_matrices_for_draw(flat, i, lw_run.sv_on)
        a1, a2, a_r, b_y, b_pi = _extract_gap_pi_coeffs(Z, A)
        # rstar_t below (both here and inside simulate_fan_draw's loop) sums
        # g+z unweighted, correct only for c==1.0 -- same guard
        # gap_pi_shock_decomposition applies to itself (numerics-reviewer,
        # S4); dormant today (estimate_c is hard-validated False everywhere,
        # specs/schema/lw_sv.py) but fails loudly instead of silently
        # corrupting the fan chart if that ever changes.
        c_check = Z[0, 3] / Z[0, 5] if Z[0, 5] != 0.0 else 1.0
        if not np.isclose(c_check, 1.0, atol=1e-9):
            raise NotImplementedError(
                f"compute_fan_draws assumes c == 1.0 (spec §1.3's default; "
                f"estimate_c is not implemented anywhere in current scope), "
                f"but the system matrices imply c = {c_check!r}. rstar_t "
                f"sums g+z unweighted, which is only correct for c == 1.0."
            )
        sim = simulate_smoother_draw(lw_run.yobs, lw_run.x, F, Q, A, Z, R, lw_run.xi00, lw_run.P00, rng)
        xi_draw = sim.xi_draw

        gap_lag1 = float(lw_run.yobs[-1, 0] - xi_draw[-1, 0])
        gap_lag2 = float(lw_run.yobs[-2, 0] - xi_draw[-2, 0])
        pi_lag1 = float(lw_run.yobs[-1, 1])
        pi_lag2 = float(lw_run.yobs[-2, 1])
        pi_lag3 = float(lw_run.yobs[-3, 1])
        pi_lag4 = float(lw_run.yobs[-4, 1])
        # (r-r*)_{T-1} ONLY -- the sole rate-gap value derivable before
        # simulate_fan_draw's own loop starts; see that function's
        # rate_gap_seed docstring for why (r-r*)_T cannot be seeded here.
        rate_gap_seed = float(lw_run.r_full[-2] - (xi_draw[-1, 3] + xi_draw[-1, 5]))
        y_hist_last4 = lw_run.yobs[-4:, 0]
        r_last = float(lw_run.r_full[-1])

        if lw_run.sv_on:
            sigma_is_i: float | None = None
            sigma_pc_i: float | None = None
            h_is_last = float(h_is[-1])
            h_pc_last = float(h_pc[-1])
            sigma_h_is_i = float(sigma_h_is_flat[i])
            sigma_h_pc_i = float(sigma_h_pc_flat[i])
        else:
            sigma_is_i = float(flat["sigma_is"][i])
            sigma_pc_i = float(flat["sigma_pc"][i])
            h_is_last = h_pc_last = sigma_h_is_i = sigma_h_pc_i = None

        out = simulate_fan_draw(
            F, Q, a1, a2, a_r, b_pi, b_y,
            xi_draw[-1], gap_lag1, gap_lag2, pi_lag1, pi_lag2, pi_lag3, pi_lag4,
            rate_gap_seed, y_hist_last4, r_last, rule,
            lw_run.sv_on, sigma_is_i, sigma_pc_i, h_is_last, h_pc_last,
            sigma_h_is_i, sigma_h_pc_i, horizon, rng,
        )

        y_level[j] = out["y_level"]
        y_growth_4q[j] = out["y_growth_4q"]
        pi_arr[j] = out["pi"]
        gap_arr[j] = out["gap"]
        rstar_arr[j] = out["rstar"]
        rate_gap_arr[j] = out["rate_gap"]

    return FanDraws(
        draw_indices=idx,
        horizon=horizon,
        y_level=y_level,
        y_growth_4q=y_growth_4q,
        pi=pi_arr,
        gap=gap_arr,
        rstar=rstar_arr,
        rate_gap=rate_gap_arr,
    )
