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
  eps_is, eps_pc) through the model's OWN linear recursions -- since
  S5-decisions item 1, via the ONE generic engine
  (``macrotoolkit.engine``): the state transition for the 3 process shocks
  (``propagate_state_shock`` at the family's declared loadings), and the
  measurement equation with feedback-map-driven endogenous lags for every
  bar's observable path (``observable_recursion``) -- so every one of the
  5 shocks, including the two measurement shocks, gets an economically
  sensible, persistent contribution rather than a same-period-only
  residual (spec §3.4). Aggregation across draws (e.g. posterior-median
  contributions for the stacked-bar chart) is the caller's job.
- **Part C -- IRF matrix** (``impulse_response_for_shock``,
  ``irf_shock_size``, ``compute_irf_draws``, spec §3.2): a theoretical
  impulse response is exactly one Part-B bar with no real data and a
  synthetic one-off impulse instead of a recovered shock path -- the same
  two engine calls, from rest.
- **Part D -- fan charts** (``simulate_fan_draw``, ``compute_fan_draws``,
  spec §3.3): STOCHASTIC forward simulation (fresh process/measurement
  noise every period, including -- for an SV run -- continuing the SV
  log-variance random walks forward, spec §3.3's "widening bands are the
  point"), seeded from a draw's own terminal smoother state and the REAL
  last few periods' lag registers (not zero -- unlike Part C's from-rest
  IRF). Since item 1 this is the engine's ``simulate_forward`` (same
  measurement-equation arithmetic, plus per-period noise and a declared
  exogenous forecast rule for r), with ``simulate_fan_draw`` reduced to a
  seed-translating wrapper.

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

from macrotoolkit.engine import (
    ConstantExogRule,
    ConstantMeasurementNoise,
    DataPathExogRule,
    RandomWalkLogVarianceNoise,
    StateLinearExogRule,
    observable_recursion,
    propagate_state_shock,
    simulate_forward,
)
from macrotoolkit.families.lw_sv import (
    LW_STATE_META,
    NEUTRAL_R_RULE_TERMS,
    require_c_is_one,
    structural_coefficients,
)
from specs.schema.base import RunSpec

# Alias kept because the state-side propagation moved verbatim into the
# generic engine (S5-decisions item 1) -- identical float operations.
_propagate_shock_through_F = propagate_state_shock

# Named state-slot indices (S5-decisions item 2): the ONE source of the
# lw_sv slot layout is macrotoolkit.families.lw_sv.LW_STATE_META (pinned
# against build_lw_matrices by tests/test_state_metadata.py); this module
# only ever addresses the state through these named lookups. The (name,
# offset) labels carry the state's own timing convention explicitly --
# e.g. ("g", -1) IS the "slot 3 holds g lagged one period" fact both S4
# fan-chart bugs tripped over when it lived only in comments.
_S_YSTAR = LW_STATE_META.slot("ystar", 0)
_S_YSTAR_M1 = LW_STATE_META.slot("ystar", -1)
_S_YSTAR_M2 = LW_STATE_META.slot("ystar", -2)
_S_G_M1 = LW_STATE_META.slot("g", -1)
_S_G_M2 = LW_STATE_META.slot("g", -2)
_S_Z_M1 = LW_STATE_META.slot("z", -1)
_S_Z_M2 = LW_STATE_META.slot("z", -2)

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

    from macrotoolkit.run import load_run_spec

    spec = load_run_spec(run_dir)
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
        # Reporting convention (verbatim from the G5a oracle mapping, per
        # this module's docstring): row t reports the (g, -1)/(z, -1) slots
        # -- HLW's own layout, validated against their output at ~1e-12.
        ystar[j] = xi[:, _S_YSTAR]
        output_gap[j] = lw_run.yobs[:, 0] - xi[:, _S_YSTAR]
        g_arr[j] = xi[:, _S_G_M1]
        z_arr[j] = xi[:, _S_Z_M1]
        rstar[j] = xi[:, _S_G_M1] + xi[:, _S_Z_M1]
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
    xi_ystar = _propagate_shock_through_F(F, LW_STATE_META.injection_vector("ystar"), eps_ystar)
    xi_g = _propagate_shock_through_F(F, LW_STATE_META.injection_vector("g"), eps_g)
    xi_z = _propagate_shock_through_F(F, LW_STATE_META.injection_vector("z"), eps_z)

    xi_init = xi_draw - xi_ystar - xi_g - xi_z
    return {"init": xi_init, "ystar": xi_ystar, "g": xi_g, "z": xi_z}


#: The 6 bars gap's own IS-curve recursion decomposes into (spec §3.4's 5
#: structural shocks' bars plus "rdata"): "rdata" is the exogenous REAL-RATE
#: DATA contribution -- the ``+(a_r/2)*(r_{t-1}+r_{t-2})`` injection AR-
#: propagated through gap's own recursion. It used to be folded into "init"
#: (pre-S5 review decision, 2026-09-02, DECISIONS.md): that made the
#: "Initial condition" line silently carry the entire cumulative
#: monetary-policy contribution, dominating the chart forever under a
#: misleading label. Splitting it out changes NO sums (G6's per-period
#: reconstruction identity is bar-count-independent by construction);
#: it only re-attributes between the two non-structural bars.
GAP_BARS: tuple[str, ...] = ("init", "rdata", "ystar", "g", "z", "is")
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
    """B2/B3: each bar's own observable path via the generic engine
    (:func:`macrotoolkit.engine.observable_recursion`, S5-decisions item 1)
    -- the measurement equation AS WRITTEN, with x's endogenous lag columns
    fed back from each bar's OWN simulated past observables per the
    family's declared feedback map. This replaces the hand-derived
    gap-space IS-curve/Phillips-curve recursions that used to live here
    (they were algebraic regroupings of the same equation); eps_is/eps_pc
    still get an economically sensible, persistent contribution via the
    model's own AR feedback (through the feedback map's y/pi lag columns)
    rather than a same-period-only measurement residual -- see this
    module's docstring for why the naive "gap = data - state" split omits
    that feedback entirely.

    Per bar, the engine inputs are:

    - state path: B1's component for "init"/"ystar"/"g"/"z" (the r* = c*g+z
      feedback enters through ``Z``'s own lag-slot entries); identically
      zero for "is"/"pc"/"rdata" (neither the measurement shocks nor the
      real-rate data ever enters the state).
    - measurement injection: ``eps_is`` into the y row for the "is" bar,
      ``eps_pc`` into the pi row for "pc"; zero otherwise.
    - exogenous input: the real ``x`` matrix's ExogLag columns
      (r_{t-1}/r_{t-2}) for the "rdata" bar ONLY (the 2026-09-02 bar split,
      DECISIONS.md: the cumulative policy contribution is its own labeled
      bar, no longer folded into "Initial condition"); zero for every other
      bar -- which also makes "init" invariant to the r data by
      construction.
    - observable seeds: the REAL pre-sample data (``y_full[3]``/
      ``y_full[2]``, ``pi_full[3..0]``) for the "init" bar; zero ("from
      rest") for every other bar. In observable space the init seed is just
      the raw data -- the old gap-space subtlety (seed gap with
      ``y_full[3] - state_components["init"][0, (ystar,-1)]``, the SMOOTHED
      y*_0 belief, never the raw prior mean ``xi00[0]``) is now implicit:
      the engine's Z-term reads the init bar's own state path, whose row 0
      lag slots carry exactly that smoothed belief. The old form's
      empirical lesson stands: an ``xi00``-based seed breaks the B2 sum
      identity for the entire sample (see
      :func:`state_shock_decomposition`'s docstring).

    Bar k's gap is then ``obs_y_k - state_path_k[:, (ystar, 0)]`` and its
    pi is ``obs_pi_k`` directly. By linearity these bars sum to the totals
    -- the per-period identity gate G6 checks (``sum(gap.values()) ==
    yobs[:,0] - xi_draw[:,0]`` and ``sum(pi.values()) == yobs[:,1]``, to
    1e-6, ``tests/test_g6_hd_identity.py``). Float caveat (DECISIONS.md
    2026-09-02): the engine's grouping is the measurement equation's, not
    the old gap-space recursion's, so bar values agree with the
    pre-engine implementation to ~1e-9 (dominated by the smoother's own
    lag-copy rounding), not bit-for-bit -- far inside every consuming
    tolerance.

    Returns ``{"gap": {bar: (T,) for bar in GAP_BARS}, "pi": {bar: (T,)
    for bar in PI_BARS}}``.
    """
    T = x.shape[0]
    if len(y_full) != T + 4 or len(pi_full) != T + 4:
        raise ValueError(
            f"y_full/pi_full must be length T+4 = {T + 4} (the 4 pre-sample "
            f"lag quarters plus the T estimation-sample quarters, same "
            f"convention as build_lw_regressors' inputs); got lengths "
            f"{len(y_full)}, {len(pi_full)}."
        )

    # The engine itself is c-agnostic (Z carries c's weighting exactly),
    # but the guard is kept: the surrounding HD reporting (r* = g + z in
    # state_shock_decomposition's consumers, plots) still assumes c == 1,
    # and silently returning bars a c != 1 caller cannot report correctly
    # would reintroduce the failure mode the guard exists for.
    coeffs = structural_coefficients(Z, A)
    require_c_is_one(coeffs["c"], "gap_pi_shock_decomposition")

    y_row = LW_STATE_META.obs_index("y")
    pi_row = LW_STATE_META.obs_index("pi")
    zeros_state = np.zeros((T, LW_STATE_META.n_state))

    gap: dict[str, np.ndarray] = {}
    pi: dict[str, np.ndarray] = {}
    for k in PI_BARS:
        state_path = state_components[k] if k in state_components else zeros_state
        meas = np.zeros((T, LW_STATE_META.n_obs))
        if k == "is":
            meas[:, y_row] = eps_is
        elif k == "pc":
            meas[:, pi_row] = eps_pc
        exog = x if k == "rdata" else None
        seeds = None
        if k == "init":
            seeds = {
                "y": {1: float(y_full[3]), 2: float(y_full[2])},
                "pi": {
                    1: float(pi_full[3]),
                    2: float(pi_full[2]),
                    3: float(pi_full[1]),
                    4: float(pi_full[0]),
                },
            }
        obs = observable_recursion(A, Z, LW_STATE_META, state_path, meas, exog, seeds)
        if k in GAP_BARS:
            gap[k] = obs[:, y_row] - state_path[:, _S_YSTAR]
        pi[k] = obs[:, pi_row]

    return {"gap": gap, "pi": pi}


def y_level_decomposition(gap: dict[str, np.ndarray], state_components: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """B4: y-level bars, ``y = gap + y*`` decomposed additively across the 7
    bars (``PI_BARS``' full set, since y needs a "pc" bar too, always
    identically zero -- eps_pc never touches y):

        y_k = gap_k + state_components[k][:, 0]   for k in {init, ystar, g, z}
        y_is = gap_is                              # eps_is's only channel into y is via gap
        y_rdata = gap_rdata                        # likewise: r data never enters the state
        y_pc = 0

    Sum-check (exact, no tolerance needed beyond floating point): ``sum(
    y.values()) == yobs[:,0]`` -- this follows immediately from B1's and
    B2's own (already sum-checked) identities, so it needs no separate
    numerical argument.
    """
    y: dict[str, np.ndarray] = {}
    for k in ("init", "ystar", "g", "z"):
        y[k] = gap[k] + state_components[k][:, _S_YSTAR]
    # "is" and "rdata" have no state (y*) component: their only channel into
    # y is via gap (eps_is by definition; the real-rate data because r never
    # enters the state either -- same reasoning as their zero rstar terms in
    # gap_pi_shock_decomposition).
    y["is"] = gap["is"].copy()
    y["rdata"] = gap["rdata"].copy()
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
    - ``gap``: B2's 6-key dict (``GAP_BARS``), each (T,) -- sums to
      ``yobs[:,0] - xi_draw[:,0]``.
    - ``pi``: B3's 7-key dict (``PI_BARS``), each (T,) -- sums to
      ``yobs[:,1]``.
    - ``y``: B4's 7-key dict (``PI_BARS``), each (T,) -- sums to
      ``yobs[:,0]``.
    - ``y_growth_4q``: B4's 7-key dict, each (T,) with the first 4 entries
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

    Structurally, an impulse response is exactly ONE historical-
    decomposition bar with no real data: a single synthetic impulse (all
    zero except ``shock_size`` at index 0) instead of a recovered shock
    path, everything starting from rest (zero lag registers, no
    prior/smoothed pre-sample belief to carry -- Part B's "init" bar only
    ever exists because a REAL run has one). So this runs exactly the same
    generic engine calls Part B's bars run (S5-decisions item 1): the 3
    trend shocks' state contribution via
    :func:`macrotoolkit.engine.propagate_state_shock` at the family's own
    declared loading, the measurement shocks as an injection into their
    declared observation row (they never enter the state at all -- their
    only channel is the observation recursion's own feedback), and the
    observable path via :func:`macrotoolkit.engine.observable_recursion`
    with zero exogenous input and zero seeds.

    ``shock`` must be one of :data:`IRF_SHOCKS`; ``shock_size`` is the
    one-off impulse applied at period 0 (spec §3.2's "one-standard-deviation
    shock at the reference volatility" -- computed by :func:`irf_shock_size`,
    a caller concern, not this function's).

    Returns :data:`IRF_RESPONSES` (``{"gap", "pi", "rstar", "y", "g"}``),
    each a length-``horizon`` array -- the same NAMED reporting mapping
    used elsewhere in this module: ``rstar = state(g,-1) + state(z,-1)``,
    ``gap = y - state(ystar,0)``, ``g = state(g,-1)``.
    """
    if shock not in IRF_SHOCKS:
        raise ValueError(
            f"impulse_response_for_shock: shock must be one of {IRF_SHOCKS!r}; got {shock!r}."
        )
    # The rstar response below sums g+z unweighted (c == 1) -- the same
    # guard Parts B/D apply (numerics-reviewer suggestion, S4.5: close the
    # one place the c == 1 assumption was unguarded).
    require_c_is_one(structural_coefficients(Z, A)["c"], "impulse_response_for_shock")

    impulse = np.zeros(horizon)
    impulse[0] = shock_size

    meas = np.zeros((horizon, LW_STATE_META.n_obs))
    if shock in LW_STATE_META.state_shocks:
        comp = propagate_state_shock(F, LW_STATE_META.injection_vector(shock), impulse)
    else:
        # A measurement shock never enters the state; its only channel is
        # the observation recursion's own feedback (spec §1.4).
        comp = np.zeros((horizon, LW_STATE_META.n_state))
        meas[:, LW_STATE_META.measurement_shocks.index(shock)] = impulse

    obs = observable_recursion(A, Z, LW_STATE_META, comp, meas)

    y_row = LW_STATE_META.obs_index("y")
    pi_row = LW_STATE_META.obs_index("pi")
    return {
        "gap": obs[:, y_row] - comp[:, _S_YSTAR],
        "pi": obs[:, pi_row],
        "rstar": comp[:, _S_G_M1] + comp[:, _S_Z_M1],
        "y": obs[:, y_row],
        "g": comp[:, _S_G_M1],
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
    """The 5 structural coefficients (a1, a2, a_r, b_y, b_pi) as a tuple --
    a thin unpacking of :func:`macrotoolkit.families.lw_sv.
    structural_coefficients` (the ONE place the matrix positions live,
    S5-decisions item 2; this replaced an earlier deliberate duplication of
    the extraction). Kept because the fan-chart tests exercise it directly;
    new code should consume the named dict instead."""
    coeffs = structural_coefficients(Z, A)
    return coeffs["a1"], coeffs["a2"], coeffs["a_r"], coeffs["b_y"], coeffs["b_pi"]


def simulate_fan_draw(
    F: np.ndarray,
    Q: np.ndarray,
    A: np.ndarray,
    Z: np.ndarray,
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
    period T) -- now a thin lw_sv-facing wrapper over the generic engine's
    :func:`macrotoolkit.engine.simulate_forward` (S5-decisions item 1),
    which runs the measurement equation with the family's declared feedback
    map instead of the hand-derived gap-space recursion that used to live
    here.

    The wrapper's job is seed translation, preserving this function's
    established gap-register interface (all real-data-derived, from the
    SAME draw's own last periods): the engine wants pre-sample OBSERVABLE
    values, so ``gap_lag1``/``gap_lag2`` (= ``yobs[-1,0]-xi_draw[-1,
    (ystar,0)]`` / ``yobs[-2,0]-xi_draw[-2,(ystar,0)]``) are converted back
    to y levels by adding ``xi_last``'s own ``(ystar,0)``/``(ystar,-1)``
    slots, and ``rate_gap_seed`` (= ``(r-r*)_{T-1}``, the ONE rate-gap
    value derivable before the forward loop starts -- the S4 numerics-
    reviewer lesson, now structural in the engine's timing convention) is
    converted back to ``r_{T-1}`` by adding ``r*_{T-1}`` read off
    ``xi_last``'s own ``(g,-1)``/``(z,-1)`` slots. Because the engine's
    first step reads those exact same values back out of the (exactly
    F-copied) next state row, the conversions cancel to within a couple of
    ulps -- verified against the pre-engine implementation by the existing
    zero-noise hand-derived-reference tests (tests/test_fan_charts.py).

    Both S4 fan-chart bug fixes are now structural engine properties rather
    than hand-maintained conventions: (1) ``(r-r*)_T`` is resolved by the
    forecast rule INSIDE the loop from that period's own freshly drawn
    state and used in the same iteration (the rate-gap seeding/ordering
    fix); (2) the real-rate term enters through ``A``/``Z``'s own signed
    entries, so there is no hand-applied sign to get wrong (the IS-curve
    sign fix -- pinned at the engine level by tests/test_engine.py).

    ``forecast_r_rule`` governs ``(r-r*)`` from absolute period T onward:
    ``"neutral"`` resolves r := r* off the state's own named slots (the
    family-declared ``NEUTRAL_R_RULE_TERMS``), forcing the rate-gap input
    to exactly 0.0; ``"last_value"`` holds r at ``r_last`` while r* keeps
    evolving. ``sv_on`` selects the measurement-noise model: constant
    ``sigma_is``/``sigma_pc``, or the CONTINUED SV log-variance random
    walks from ``h_is_last``/``h_pc_last`` at scales ``sigma_h_is``/
    ``sigma_h_pc`` (spec §3.3's "widening bands are the point"). RNG
    consumption order is preserved exactly from the pre-engine
    implementation (state noise, then SV innovations, then measurement
    shocks, each period; one extra state draw after the loop for the r*
    alignment).

    Returns ``{"y_level", "y_growth_4q", "pi", "gap", "rstar",
    "rate_gap"}``, each length ``horizon``. Index ``t`` of ``y_level``/
    ``pi``/``gap``/``rstar`` is absolute period T+t+1 for ALL FOUR series
    (the 2026-09-02 r* alignment fix: r* at a period only becomes visible
    in the NEXT period's state row, so the engine's extra post-loop state
    row supplies the final point). ``rate_gap`` keeps its input-diagnostic
    timing: index t is the ``(r-r*)`` INPUT used at forecast step t.
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
        meas_noise = RandomWalkLogVarianceNoise((h_is_last, h_pc_last), (sigma_h_is, sigma_h_pc))
    else:
        if sigma_is is None or sigma_pc is None:
            raise ValueError(
                "simulate_fan_draw: sv_on=False requires sigma_is/sigma_pc "
                "(that draw's own constant posterior scales) to be real "
                "numbers, not None."
            )
        meas_noise = ConstantMeasurementNoise((sigma_is, sigma_pc))

    xi_last = np.asarray(xi_last, dtype=np.float64)
    # Seed translation (see docstring): gap registers -> y levels; the
    # rate-gap seed -> r_{T-1}.
    rstar_prev = xi_last[_S_G_M1] + xi_last[_S_Z_M1]  # r*_{T-1}
    obs_seeds = {
        "y": {
            1: float(gap_lag1 + xi_last[_S_YSTAR]),
            2: float(gap_lag2 + xi_last[_S_YSTAR_M1]),
        },
        "pi": {1: float(pi_lag1), 2: float(pi_lag2), 3: float(pi_lag3), 4: float(pi_lag4)},
    }
    exog_seeds = {"r": {2: float(rate_gap_seed + rstar_prev)}}

    if forecast_r_rule == "neutral":
        r_rule = StateLinearExogRule(LW_STATE_META, NEUTRAL_R_RULE_TERMS)
    else:
        r_rule = ConstantExogRule(float(r_last))

    out = simulate_forward(
        F, Q, A, Z, LW_STATE_META,
        xi_last, obs_seeds, exog_seeds, {"r": r_rule}, meas_noise, horizon, rng,
    )
    obs = out["obs"]
    states = out["states"]
    r_resolved = out["exog_resolved"]["r"]

    y_row = LW_STATE_META.obs_index("y")
    pi_row = LW_STATE_META.obs_index("pi")
    y_level = obs[:, y_row]
    pi_path = obs[:, pi_row]
    # Named reporting mapping: gap_t = y_t - y*_t; r* aligned to the same
    # period indexing via the NEXT state row's (g,-1)/(z,-1) slots (the
    # engine's post-loop extra row supplies the final point); rate_gap is
    # the per-step (r-r*) INPUT -- exactly 0.0 under "neutral" because the
    # rule and this diagnostic read the same slots the same way.
    gap_path = y_level - states[:horizon, _S_YSTAR]
    rstar_path = states[1:, _S_G_M1] + states[1:, _S_Z_M1]
    rate_gap_path = r_resolved - (states[:horizon, _S_G_M1] + states[:horizon, _S_Z_M1])

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
        # The c == 1 guard: the fan's r* reporting (and the neutral rule's
        # r := g + z resolution) sums g+z unweighted -- numerics-reviewer,
        # S4; dormant today but fails loudly instead of silently
        # corrupting the fan chart.
        require_c_is_one(structural_coefficients(Z, A)["c"], "compute_fan_draws")
        sim = simulate_smoother_draw(lw_run.yobs, lw_run.x, F, Q, A, Z, R, lw_run.xi00, lw_run.P00, rng)
        xi_draw = sim.xi_draw

        gap_lag1 = float(lw_run.yobs[-1, 0] - xi_draw[-1, _S_YSTAR])
        gap_lag2 = float(lw_run.yobs[-2, 0] - xi_draw[-2, _S_YSTAR])
        pi_lag1 = float(lw_run.yobs[-1, 1])
        pi_lag2 = float(lw_run.yobs[-2, 1])
        pi_lag3 = float(lw_run.yobs[-3, 1])
        pi_lag4 = float(lw_run.yobs[-4, 1])
        # (r-r*)_{T-1} ONLY -- the sole rate-gap value derivable before
        # simulate_fan_draw's own loop starts; see that function's
        # rate_gap_seed docstring for why (r-r*)_T cannot be seeded here.
        rate_gap_seed = float(lw_run.r_full[-2] - (xi_draw[-1, _S_G_M1] + xi_draw[-1, _S_Z_M1]))
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
            F, Q, A, Z,
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


# ---------------------------------------------------------------------------
# Part E: prior-predictive check (spec §4, S5-decisions item 7)
# ---------------------------------------------------------------------------


@dataclass
class PriorPredictiveDraws:
    """Prior-predictive observable paths (spec §4's "what do my priors
    imply about observable paths" check, S5-decisions item 7): ``n_draws``
    full generative simulations over the estimation sample, each at a
    FRESH parameter point drawn from the run's own RESOLVED priors (the
    defaults plus the spec's overrides -- the exact config the template
    stamped, so priors doing deliberate identification work, like the
    sigma_g/sigma_z pile-up controls, show up exactly as the sampler sees
    them).

    ``gap``/``pi``/``y`` are ``(n_draws, T)`` simulated paths;
    ``pi_actual``/``y_actual`` the real data for overlay. Aggregation to
    percentile bands is ``plots.py``'s job, as everywhere else.
    """

    n_draws: int
    dates: pd.DatetimeIndex  # (T,)
    gap: np.ndarray  # (n_draws, T)
    pi: np.ndarray  # (n_draws, T)
    y: np.ndarray  # (n_draws, T)
    pi_actual: np.ndarray  # (T,)
    y_actual: np.ndarray  # (T,)


def compute_prior_predictive_draws(lw_run: LWRun, *, seed: int | None = None) -> PriorPredictiveDraws:
    """Simulate ``lw_run.spec.outputs.prior_predictive_draws`` full
    observable paths from the prior, through the SAME machinery the run
    itself used (S5-decisions item 7): parameters from the run's resolved
    priors (:func:`macrotoolkit.families.lw_sv.sample_prior_params` --
    including the template's own truncation constraints and, for an SV
    run, the data-anchored mu_h0 initial log-variances), matrices from
    ``build_lw_matrices``, and paths from the generic engine's
    :func:`macrotoolkit.engine.simulate_forward` -- initial state drawn
    from the run's own (xi00, P00) prior, endogenous y/pi feedback closed
    through the family's declared feedback map, the REAL r series supplied
    as the exogenous input (:class:`macrotoolkit.engine.DataPathExogRule`),
    and the real 4 pre-sample lag quarters seeding the observable
    registers. Needs no posterior draws at all -- only the run's spec and
    data snapshot.

    ``seed`` defaults to ``lw_run.spec.sampler.seed`` (documented,
    reproducible default, same convention as the other compute_* entry
    points).
    """
    from macrotoolkit.families.lw_sv import lw_mu_h0_anchors, sample_prior_params
    from macrotoolkit.run import build_render_context
    from macrotoolkit.smoother import _psd_sqrt, build_lw_matrices

    T = lw_run.yobs.shape[0]
    n_draws = lw_run.spec.outputs.prior_predictive_draws
    priors = build_render_context(lw_run.spec)["priors"]
    sv_on = lw_run.sv_on

    mu_h0_is = mu_h0_pc = None
    if sv_on:
        mu_h0_is, mu_h0_pc = lw_mu_h0_anchors(lw_run.y_full, lw_run.pi_full, lw_run.r_full)

    y_row = LW_STATE_META.obs_index("y")
    pi_row = LW_STATE_META.obs_index("pi")
    # Observable seeds: the real 4 pre-sample lag quarters (full-array
    # indices 0..3; estimation row 0 = full index 4).
    obs_seeds = {
        "y": {1: float(lw_run.y_full[3]), 2: float(lw_run.y_full[2])},
        "pi": {
            1: float(lw_run.pi_full[3]),
            2: float(lw_run.pi_full[2]),
            3: float(lw_run.pi_full[1]),
            4: float(lw_run.pi_full[0]),
        },
    }
    # Real r as the exogenous input: step t (estimation row t, period t+1)
    # needs r_{t} = full index t+3 as its lag-1 value; the lag-2 seed is
    # r at full index 2.
    r_lag1_path = lw_run.r_full[3 : 3 + T]
    exog_seeds = {"r": {2: float(lw_run.r_full[2])}}

    sqrt_P00 = _psd_sqrt(lw_run.P00)

    gap = np.empty((n_draws, T))
    pi = np.empty((n_draws, T))
    y = np.empty((n_draws, T))

    rng = np.random.default_rng(seed if seed is not None else lw_run.spec.sampler.seed)

    for j in range(n_draws):
        params = sample_prior_params(priors, sv_on, rng, mu_h0_is=mu_h0_is, mu_h0_pc=mu_h0_pc)
        if sv_on:
            # build_lw_matrices needs SOME sigma_is/sigma_pc to shape R,
            # which the engine's SV noise model then replaces entirely --
            # same placeholder convention as _system_matrices_for_draw.
            mat_params = {**params, "sigma_is": 1.0, "sigma_pc": 1.0}
            meas_noise = RandomWalkLogVarianceNoise(
                (params["h0_is"], params["h0_pc"]),
                (params["sigma_h_is"], params["sigma_h_pc"]),
            )
        else:
            mat_params = params
            meas_noise = ConstantMeasurementNoise((params["sigma_is"], params["sigma_pc"]))
        F, Q, A, Z, _ = build_lw_matrices(mat_params, c=1.0)

        # Initial state from the run's own explicit prior (spec §2.2: no
        # ad-hoc diffuse hacks) -- one fresh draw per parameter point.
        xi_init = lw_run.xi00 + sqrt_P00 @ rng.standard_normal(len(lw_run.xi00))

        out = simulate_forward(
            F, Q, A, Z, LW_STATE_META,
            xi_init, obs_seeds, exog_seeds,
            {"r": DataPathExogRule(r_lag1_path)}, meas_noise, T, rng,
        )
        obs = out["obs"]
        states = out["states"]
        y[j] = obs[:, y_row]
        pi[j] = obs[:, pi_row]
        gap[j] = obs[:, y_row] - states[:T, _S_YSTAR]

    return PriorPredictiveDraws(
        n_draws=n_draws,
        dates=lw_run.dates,
        gap=gap,
        pi=pi,
        y=y,
        pi_actual=lw_run.yobs[:, 1].copy(),
        y_actual=lw_run.yobs[:, 0].copy(),
    )
