"""Per-run, per-posterior-draw outputs for the ``lw_sv`` family: trend-cycle
series (spec §3.1) and historical decomposition (spec §3.4). Everything here
consumes a *completed* run directory (``runs/<hash12>/``) and re-runs the
Durbin-Koopman simulation smoother (``macrotoolkit.smoother.
simulate_smoother_draw``) per selected posterior draw -- gated by G1 (this
module must not be trusted until the smoother's own gate, spec §2.4, is
green) and validated for the historical-decomposition arithmetic specifically
by gate G6 (``tests/test_g6_hd_identity.py``).

Two independent pieces, kept separable in this file:

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
    xi_ystar = np.zeros((T, n))
    xi_g = np.zeros((T, n))
    xi_z = np.zeros((T, n))

    prev_ystar = np.zeros(n)
    prev_g = np.zeros(n)
    prev_z = np.zeros(n)

    for t in range(T):
        w_ystar = np.zeros(n)
        w_ystar[0] = eps_ystar[t]
        cur_ystar = F @ prev_ystar + w_ystar

        w_g = np.zeros(n)
        w_g[0] = eps_g[t] / 4.0
        w_g[3] = eps_g[t]
        cur_g = F @ prev_g + w_g

        w_z = np.zeros(n)
        w_z[5] = eps_z[t]
        cur_z = F @ prev_z + w_z

        xi_ystar[t] = cur_ystar
        xi_g[t] = cur_g
        xi_z[t] = cur_z

        prev_ystar, prev_g, prev_z = cur_ystar, cur_g, cur_z

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
