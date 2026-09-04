"""lw_sv family numerics metadata (S5-decisions item 2).

THE one place the lw_sv state-slot layout and structural-coefficient matrix
positions are written down outside the validated matrix constructor itself
(``macrotoolkit.smoother.build_lw_matrices``, which this module's metadata
is pinned against by ``tests/test_state_metadata.py``). Everything
downstream -- ``results_lw.py``'s output modules, the generic engines --
consumes the NAMES exported here, never raw indices.

State layout (``smoother.py``'s module docstring, HLW's own):

    xi_t = [y*_t, y*_{t-1}, y*_{t-2}, g_{t-1}, g_{t-2}, z_{t-1}, z_{t-2}]

expressed below as ``(name, offset)`` labels -- note g and z are carried
LAGGED one period relative to the state row's own period (the "slot-3/5
one-period-lag convention" both S4 fan-chart bugs tripped over; with named
offsets the convention is in the label, not in the reader's memory).

Shock timing: a period-t structural shock realizes going INTO row t's
state (``smoother.SimSmootherDraw``'s convention), so e.g. the g shock
lands in the slot labeled ``("g", -1)`` -- the row-t slot that carries the
most recent g value.
"""
from __future__ import annotations

import numpy as np

from macrotoolkit.families.base import ExogLag, ObsLag, ObsLagMean, StateSpaceMeta

#: Named state metadata for lw_sv. The g shock's 0.25 loading into
#: ("ystar", 0) is the quarterly g/4 increment (g is ANNUALIZED everywhere;
#: only the potential-output transition divides by 4 -- spec §1.3, enforced
#: by tests/test_units_conventions.py); 0.25 * eps is bit-identical to
#: eps / 4.0 in IEEE double, so this loading reproduces the historical
#: hand-rolled injection exactly.
LW_STATE_META = StateSpaceMeta(
    state_labels=(
        ("ystar", 0),
        ("ystar", -1),
        ("ystar", -2),
        ("g", -1),
        ("g", -2),
        ("z", -1),
        ("z", -2),
    ),
    state_shocks=("ystar", "g", "z"),
    shock_loadings={
        "ystar": {("ystar", 0): 1.0},
        "g": {("g", -1): 1.0, ("ystar", 0): 0.25},
        "z": {("z", -1): 1.0},
    },
    measurement_shocks=("is", "pc"),
    obs_names=("y", "pi"),
    exog_names=("r",),
    # The feedback map (S5-decisions item 1): what each x column IS, in
    # build_lw_regressors' exact column order -- x_t = [y_{t-1}, y_{t-2},
    # r_{t-1}, r_{t-2}, pi_{t-1}, (pi_{t-2}+pi_{t-3}+pi_{t-4})/3]. Pinned
    # column-for-column against build_lw_regressors by
    # tests/test_engine.py.
    feedback_map=(
        ObsLag("y", 1),
        ObsLag("y", 2),
        ExogLag("r", 1),
        ExogLag("r", 2),
        ObsLag("pi", 1),
        ObsLagMean("pi", (2, 3, 4)),
    ),
)

#: Observation-row order (measurement equation rows of yobs/Z/R).
OBS_NAMES = LW_STATE_META.obs_names

#: The documented, reusable SBC PRIOR CONFIG (S5-decisions item 6 -- the
#: G3-style prior overrides "promoted from ad hoc to a documented, reusable
#: mechanism, used by G4"; ENGINEERING.md "Stationarity for SBC"). SBC must
#: sample EXACTLY the fitted prior -- no stationarity rejection, which is
#: G1/G2 machinery and invalid for calibration -- but the production a1/a2
#: defaults put ~1/3 of prior mass on non-stationary gap dynamics whose
#: simulated data is numerically un-filterable in float64 (measured
#: 2026-08-31, DECISIONS.md: |y| ~ 2e13 at T=120, KF covariance update
#: loses everything to cancellation, ranks come out garbage). Under THIS
#: override the stationarity boundary sits ~4 sigma out (non-stationary
#: mass ~3e-5), so the exact prior is simulable with no rejection anywhere.
#: Applied through the PRODUCTION `priors:` override path on the fit side
#: and to the prior sampler on the simulation side, so the two stay exactly
#: equal -- every SBC design (G3's no-SV gate, G4's full-SV gate, any
#: future family analogue) consumes this one declaration. A stationarity-
#: enforcing PACF parameterization for AR blocks is framework backlog, not
#: S5 scope (S5-decisions item 6).
SBC_STATIONARITY_PRIOR_CONFIG: dict[str, dict] = {
    "a1": {"mu": 0.8, "sd": 0.1},
    "a2": {"mu": -0.25, "sd": 0.05},
}

#: The "neutral" forecast_r_rule's declaration (spec §3.3: r_{T+h} :=
#: r*_{T+h}, neutral policy): r's previous-period value resolves as
#: r* = g + z read off the named offset -1 slots -- valid for c == 1 only
#: (the same require_c_is_one guard applies at the call site). Consumed by
#: macrotoolkit.engine.StateLinearExogRule.
NEUTRAL_R_RULE_TERMS = {("g", -1): 1.0, ("z", -1): 1.0}


def structural_coefficients(Z: np.ndarray, A: np.ndarray) -> dict[str, float]:
    """The named structural-coefficient dict read off one draw's ``Z``/``A``
    system matrices -- the exact entries ``build_lw_matrices`` stamps,
    inverted here in the ONE place that knows where they live:

    - ``a1 = -Z[y, (ystar,-1)]``, ``a2 = -Z[y, (ystar,-2)]`` (gap-lag
      conversion terms of the IS curve);
    - ``a_r = -2 * Z[y, (z,-1)]`` -- read off the z-lag entry, which
      carries NO ``c`` factor (``build_lw_matrices`` multiplies only the
      g-lag entries by ``c``), so the extraction stays correct regardless
      of ``c``;
    - ``b_y = -Z[pi, (ystar,-1)]`` (Phillips curve's gap-lag conversion);
    - ``b_pi = A[4, 1]`` -- exogenous row 4 is pi_{t-1}, observation
      column 1 is pi (``build_lw_regressors``' x-column order; becomes a
      named feedback-map lookup in S5-decisions item 1);
    - ``c``: the r* = c*g + z loading, recovered as the ratio of the g-lag
      to the z-lag Z entries (1.0 when the z entry is zero, i.e. a_r = 0,
      where the ratio is undefined and c is unidentified in Z anyway).

    Downstream code MUST consume this dict (or a draw's posterior values
    directly) rather than re-peeking matrix entries.
    """
    y_row = OBS_NAMES.index("y")
    pi_row = OBS_NAMES.index("pi")
    s_ystar_m1 = LW_STATE_META.slot("ystar", -1)
    s_ystar_m2 = LW_STATE_META.slot("ystar", -2)
    s_g_m1 = LW_STATE_META.slot("g", -1)
    s_z_m1 = LW_STATE_META.slot("z", -1)

    z_lag_entry = Z[y_row, s_z_m1]
    c = Z[y_row, s_g_m1] / z_lag_entry if z_lag_entry != 0.0 else 1.0
    return {
        "a1": -Z[y_row, s_ystar_m1],
        "a2": -Z[y_row, s_ystar_m2],
        "a_r": -2.0 * z_lag_entry,
        "b_y": -Z[pi_row, s_ystar_m1],
        "b_pi": A[4, 1],
        "c": float(c),
    }


# ---------------------------------------------------------------------------
# Spec-driven builders (S5-decisions item 4: moved here from run.py's
# if/elif chains; dispatched via specs.schema.FAMILY_REGISTRY's dotted
# paths so adding a family never edits run.py again).
# ---------------------------------------------------------------------------


def lw_mu_h0_anchors(y: np.ndarray, pi: np.ndarray, r: np.ndarray) -> tuple[float, float]:
    """The mu_h0 OLS anchors for the SV variant: mu_h0_s = 2*ln(sigma_hat_
    OLS,s) (h is log-variance, so exp(mu_h0/2) = sigma_hat), where
    sigma_hat_OLS is the residual sd of a rough OLS pass mirroring HLW's
    own stage-3 initialization EXACTLY (rstar.stage3.R lines 22-48;
    user-confirmed definition, DECISIONS.md 2026-08-31):

    - gap proxy: residual of OLS of y on [const, linear trend] over the
      FULL trimmed sample (lag quarters included). y is already 100*ln(GDP)
      (spec §1.1), so no extra x100 -- same units as HLW's `output.gap`.
    - IS: OLS of gap_t on [gap_{t-1}, gap_{t-2}, (r_{t-1}+r_{t-2})/2, 1]
      over the T estimation rows; sigma_hat = sqrt(RSS / (T - 4)).
    - PC: OLS of pi_t on [pi_{t-1}, (pi_{t-2}+pi_{t-3}+pi_{t-4})/3,
      gap_{t-1}], NO intercept; sigma_hat = sqrt(RSS / (T - 3)).

    Deterministic given the trimmed data, so run identity stays
    reproducible. Inputs are the full trimmed arrays INCLUDING the 4
    pre-sample lag quarters (build_lw_regressors' convention).
    """
    y = np.asarray(y, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    n = len(y)
    t_est = n - 4
    if t_est <= 4:
        raise ValueError(
            f"mu_h0 OLS anchor needs at least 9 data rows (4 lag quarters + "
            f"5 estimation quarters, for a positive-dof IS regression); the "
            f"trimmed data has {n}."
        )

    trend = np.column_stack([np.ones(n), np.arange(1, n + 1, dtype=np.float64)])
    gap = y - trend @ np.linalg.lstsq(trend, y, rcond=None)[0]

    y_is = gap[4:n]
    x_is = np.column_stack(
        [
            gap[3 : 3 + t_est],
            gap[2 : 2 + t_est],
            (r[3 : 3 + t_est] + r[2 : 2 + t_est]) / 2.0,
            np.ones(t_est),
        ]
    )
    resid_is = y_is - x_is @ np.linalg.lstsq(x_is, y_is, rcond=None)[0]
    sigma_is = float(np.sqrt(resid_is @ resid_is / (t_est - x_is.shape[1])))

    y_pc = pi[4:n]
    x_pc = np.column_stack(
        [
            pi[3 : 3 + t_est],
            (pi[2 : 2 + t_est] + pi[1 : 1 + t_est] + pi[0:t_est]) / 3.0,
            gap[3 : 3 + t_est],
        ]
    )
    resid_pc = y_pc - x_pc @ np.linalg.lstsq(x_pc, y_pc, rcond=None)[0]
    sigma_pc = float(np.sqrt(resid_pc @ resid_pc / (t_est - x_pc.shape[1])))

    if not (np.isfinite(sigma_is) and np.isfinite(sigma_pc) and sigma_is > 0 and sigma_pc > 0):
        raise ValueError(
            f"mu_h0 OLS anchor produced a non-positive/non-finite residual "
            f"sd (IS {sigma_is!r}, PC {sigma_pc!r}) -- degenerate input "
            f"data? The anchor needs genuine residual variation."
        )
    return 2.0 * float(np.log(sigma_is)), 2.0 * float(np.log(sigma_pc))


def build_stan_data(df) -> dict:
    """Map the loaded/mapped DataFrame to lw_sv's Stan `data` block.

    Convention (HLW's own, spec §1.2's lag structure): the loaded data
    must INCLUDE four pre-sample lag quarters -- the first estimation
    quarter is row 5 of the trimmed data, because the Phillips curve needs
    pi_{t-4}. `data.sample.start` in the spec therefore points at the
    first LAG quarter, four quarters before the first estimated one.
    """
    from macrotoolkit.smoother import build_lw_regressors, default_initial_state

    if len(df) < 5:
        raise ValueError(
            f"lw_sv needs at least 5 data rows (4 pre-sample lag "
            f"quarters + 1 estimation quarter); the trimmed data has "
            f"{len(df)}. Note data.sample.start must include the 4 lag "
            f"quarters before the first estimation quarter."
        )
    yobs, x = build_lw_regressors(
        df["y"].to_numpy(), df["pi"].to_numpy(), df["r"].to_numpy()
    )
    xi00, P00 = default_initial_state(float(df["y"].to_numpy()[4]))
    # The mu_h0 OLS anchors are computed unconditionally: the SV render
    # declares them as data; a no-SV render simply doesn't (CmdStan
    # ignores unused input entries), and they don't enter run identity
    # (the hash covers the raw data file, not this dict).
    mu_h0_is, mu_h0_pc = lw_mu_h0_anchors(
        df["y"].to_numpy(), df["pi"].to_numpy(), df["r"].to_numpy()
    )
    return {
        "T": int(yobs.shape[0]),
        "yobs": yobs,
        "x": x,
        "xi00": xi00,
        "P00": P00,
        "mu_h0_is": mu_h0_is,
        "mu_h0_pc": mu_h0_pc,
    }


def build_render_context(spec) -> dict:
    """Template context for lw_sv's Jinja render: stamps c and the resolved
    priors (specs/schema/lw_sv.py defaults, overridden per key by the
    spec's `priors:` block -- unknown prior names are a hard error, and
    overriding a prior that is INACTIVE for the run's sv_shocks variant is
    a hard error too, so a typo'd override never silently does nothing)."""
    from specs.schema.lw_sv import (
        DEFAULT_PRIORS,
        NO_SV_ONLY_PRIOR_NAMES,
        SV_ONLY_PRIOR_NAMES,
    )

    sv_on = bool(spec.model.options.sv_shocks)
    inactive = NO_SV_ONLY_PRIOR_NAMES if sv_on else SV_ONLY_PRIOR_NAMES
    priors = {name: dict(entry) for name, entry in DEFAULT_PRIORS.items()}
    for name, override in spec.priors.items():
        if name not in priors:
            raise ValueError(
                f"priors[{name!r}] is not a parameter of the lw_sv "
                f"family. Valid names: {sorted(priors)}."
            )
        if name in inactive:
            variant = "sv_shocks: [is, pc]" if sv_on else "sv_shocks: []"
            raise ValueError(
                f"priors[{name!r}] does not exist in the variant this "
                f"spec selects ({variant}) -- the override would "
                f"silently do nothing. These prior names belong only to "
                f"the other variant: {sorted(inactive)}."
            )
        if not isinstance(override, dict):
            raise ValueError(
                f"priors[{name!r}] must be a mapping of prior fields to "
                f"override (e.g. {{sd: 0.5}}), got {override!r}."
            )
        merged = {**priors[name], **override}
        unknown = set(merged) - set(priors[name])
        if unknown:
            raise ValueError(
                f"priors[{name!r}] has unknown field(s) {sorted(unknown)}; "
                f"the default entry's fields are {sorted(priors[name])}."
            )
        priors[name] = merged
    # estimate_c is validated false (specs/schema/lw_sv.py), so c is always
    # the spec §1.3 default here. sv_shocks is [] or the canonical
    # ["is", "pc"] (schema-normalized); the template's SV blocks render iff
    # it is non-empty.
    return {
        "c": 1.0,
        "priors": priors,
        "sv_shocks": list(spec.model.options.sv_shocks),
    }


def sample_prior_params(
    priors: dict[str, dict],
    sv_on: bool,
    rng: np.random.Generator,
    mu_h0_is: float | None = None,
    mu_h0_pc: float | None = None,
) -> dict[str, float]:
    """Draw ONE parameter point from the lw_sv prior (S5-decisions item 7:
    the prior-predictive check simulates from the RUN'S OWN resolved
    priors -- ``build_render_context(spec)["priors"]``, i.e. the defaults
    plus the spec's overrides, the exact config the template stamped).

    Distribution semantics mirror the template's own declarations
    (``stan/templates/lw_sv.stan.j2``): ``normal(mu, sd)``;
    ``beta(a, b)``; ``half_normal(sd)`` = |N(0, sd^2)|; and the two
    TRUNCATED normals expressed in Stan as parameter CONSTRAINTS rather
    than in the prior density -- ``a_r`` has ``<upper=0>`` and ``b_y``
    ``<lower=0>`` -- sampled here by rejection so the draws live on the
    same support the sampler explores.

    ``sv_on=False`` draws include ``sigma_is``/``sigma_pc``;
    ``sv_on=True`` draws instead include ``sigma_h_is``/``sigma_h_pc``
    and the initial log-variances ``h0_is``/``h0_pc`` ~ N(mu_h0, sd)
    (spec §1.5) -- the caller supplies the data-derived mu_h0 anchors
    (:func:`lw_mu_h0_anchors`).
    """

    def draw(name: str, *, lower: float | None = None, upper: float | None = None, mean_override: float | None = None) -> float:
        entry = priors[name]
        dist = entry["dist"]
        if dist == "normal":
            mu = mean_override if mean_override is not None else entry["mu"]
            # Rejection from the untruncated normal == Stan's renormalized
            # truncated prior. Capped so a pathological override (mean far
            # beyond the truncation bound) fails loudly instead of
            # spinning (numerics-reviewer suggestion; same 10k cap as
            # g3_harness's historical sampler).
            for _ in range(10_000):
                v = rng.normal(mu, entry["sd"])
                if (lower is None or v > lower) and (upper is None or v < upper):
                    return float(v)
            raise RuntimeError(
                f"sample_prior_params: failed to draw {name!r} inside its "
                f"truncation bound within 10,000 attempts -- the resolved "
                f"prior (mu={mu}, sd={entry['sd']}, lower={lower}, "
                f"upper={upper}) puts essentially no mass on the "
                f"parameter's support."
            )
        if dist == "beta":
            return float(rng.beta(entry["a"], entry["b"]))
        if dist == "half_normal":
            return float(abs(rng.normal(0.0, entry["sd"])))
        raise ValueError(
            f"sample_prior_params: unknown prior dist {dist!r} for "
            f"{name!r} -- the template stamps only normal/beta/half_normal."
        )

    params: dict[str, float] = {
        "a1": draw("a1"),
        "a2": draw("a2"),
        "a_r": draw("a_r", upper=0.0),  # template constraint <upper=0>
        "b_pi": draw("b_pi"),
        "b_y": draw("b_y", lower=0.0),  # template constraint <lower=0>
        "sigma_ystar": draw("sigma_ystar"),
        "sigma_g": draw("sigma_g"),
        "sigma_z": draw("sigma_z"),
    }
    if sv_on:
        if mu_h0_is is None or mu_h0_pc is None:
            raise ValueError(
                "sample_prior_params: sv_on=True needs the data-derived "
                "mu_h0_is/mu_h0_pc anchors (lw_mu_h0_anchors)."
            )
        params["sigma_h_is"] = draw("sigma_h_is")
        params["sigma_h_pc"] = draw("sigma_h_pc")
        params["h0_is"] = draw("mu_h0_is", mean_override=mu_h0_is)
        params["h0_pc"] = draw("mu_h0_pc", mean_override=mu_h0_pc)
    else:
        params["sigma_is"] = draw("sigma_is")
        params["sigma_pc"] = draw("sigma_pc")
    return params


def require_c_is_one(c: float, where: str) -> None:
    """Fail loudly if the system matrices imply ``c != 1.0`` (spec §1.3's
    fixed default; ``estimate_c`` is hard-validated ``False`` everywhere in
    current scope, ``specs/schema/lw_sv.py``).

    Numerics-reviewer finding (S4): every downstream computation that sums
    ``g + z`` UNWEIGHTED (r* = g + z) is correct only for c == 1; with
    c != 1 the g terms would need a ``c`` weight, and a naive unweighted
    sum silently corrupts the result (verified numerically for the HD:
    c = 1.5 produces a gap-HD reconstruction error of ~0.36 against G6's
    1e-6 tolerance). ``where`` names the calling computation so the error
    points at the code that must be generalized before the guard is
    removed.
    """
    if not np.isclose(c, 1.0, atol=1e-9):
        raise NotImplementedError(
            f"{where} assumes c == 1.0 (spec §1.3's default; estimate_c is "
            f"not implemented anywhere in current scope), but the system "
            f"matrices imply c = {c!r}. The r* = g + z computation sums g "
            f"and z unweighted, which is only correct for c == 1.0 -- "
            f"generalize it (weight the g terms by c) before removing this "
            f"guard."
        )


def prior_scalar_sds(spec, df, n_draws: int = 10_000, seed: int = 20260902) -> dict[str, float]:
    """Monte-Carlo prior standard deviations of the family's scalar
    parameters under a spec's RESOLVED priors (defaults + overrides) --
    the prior side of the sweep report's prior→posterior contraction
    readout (S5-decisions item 9). Monte Carlo through
    :func:`sample_prior_params` rather than closed forms so ANY prior the
    family can stamp (truncated normals included) is covered by the same
    code path the prior-predictive check uses. ``df`` is the trimmed
    model-ready DataFrame (needed for the SV variant's data-derived mu_h0
    anchors). Deterministic given ``seed``.
    """
    resolved = build_render_context(spec)["priors"]
    sv_on = bool(spec.model.options.sv_shocks)
    mu_h0_is = mu_h0_pc = None
    if sv_on:
        mu_h0_is, mu_h0_pc = lw_mu_h0_anchors(
            df["y"].to_numpy(), df["pi"].to_numpy(), df["r"].to_numpy()
        )
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(n_draws):
        p = sample_prior_params(resolved, sv_on, rng, mu_h0_is=mu_h0_is, mu_h0_pc=mu_h0_pc)
        for k, v in p.items():
            draws.setdefault(k, []).append(v)
    return {k: float(np.std(np.asarray(v))) for k, v in draws.items()}


def headline_series(run_dir, thin: int = 5, seed: int | None = None) -> dict[str, tuple]:
    """The family's headline smoothed series for cross-run comparison
    overlays (the sweep report, S5-decisions item 9): posterior-median r*
    and output gap from the trend-cycle machinery, with smoother draws
    thinned x``thin`` (report-side only -- stored runs are untouched; the
    sweep overlay needs medians, not full band resolution). Returns
    ``{name: (dates, median_path)}``.
    """
    import dataclasses

    from macrotoolkit.results_lw import compute_trend_cycle_draws, load_lw_run
    from specs.schema.lw_sv import ThinSpec

    lw_run = load_lw_run(run_dir)
    if thin > 1:
        lw_run = dataclasses.replace(
            lw_run,
            spec=lw_run.spec.model_copy(
                update={"outputs": lw_run.spec.outputs.model_copy(update={"smoother_draws": ThinSpec(thin=thin)})}
            ),
        )
    tcd = compute_trend_cycle_draws(lw_run, seed=seed)
    return {
        "rstar": (tcd.dates, np.median(tcd.rstar, axis=0)),
        "gap": (tcd.dates, np.median(tcd.output_gap, axis=0)),
    }


# ---------------------------------------------------------------------------
# Automatic fit-time mirror check (S6 WP3): prior draw -> Stan inits +
# the Python KF mirror at the same point on the same Stan data.
# ---------------------------------------------------------------------------


def mirror_points(spec, stan_data: dict, n: int, rng: np.random.Generator):
    """``n`` prior draws of the rendered lw_sv program's parameters as Stan
    inits (SV variant: the non-centered ``h0_*_raw`` and fresh ``nu_*``
    innovation vectors) with the Python KF log-likelihood at each --
    matrices via ``build_lw_matrices``, the SV variant's R_t through the
    same ``sv_rw_noncentered`` -> ``sv_diag_variance_path`` composition
    the template stamps."""
    from macrotoolkit.qc import MirrorPoint
    from macrotoolkit.smoother import (
        build_lw_matrices,
        kalman_loglik,
        sv_diag_variance_path,
        sv_rw_noncentered,
    )

    priors = build_render_context(spec)["priors"]
    sv_on = bool(spec.model.options.sv_shocks)
    T = int(stan_data["T"])
    yobs, x, xi00, P00 = stan_data["yobs"], stan_data["x"], stan_data["xi00"], stan_data["P00"]
    points = []
    for _ in range(n):
        params = sample_prior_params(
            priors, sv_on, rng,
            mu_h0_is=stan_data.get("mu_h0_is"), mu_h0_pc=stan_data.get("mu_h0_pc"),
        )
        inits = {k: params[k] for k in ("a1", "a2", "a_r", "b_pi", "b_y", "sigma_ystar", "sigma_g", "sigma_z")}
        if sv_on:
            nu_is = rng.standard_normal(T)
            nu_pc = rng.standard_normal(T)
            inits.update(
                sigma_h_is=params["sigma_h_is"],
                sigma_h_pc=params["sigma_h_pc"],
                h0_is_raw=(params["h0_is"] - stan_data["mu_h0_is"]) / priors["mu_h0_is"]["sd"],
                h0_pc_raw=(params["h0_pc"] - stan_data["mu_h0_pc"]) / priors["mu_h0_pc"]["sd"],
                nu_is=nu_is,
                nu_pc=nu_pc,
            )
            # The Python side rebuilds h_0 exactly as the template does
            # (mu_h0 + sd * raw) so both sides see the identical point.
            h0_is = stan_data["mu_h0_is"] + priors["mu_h0_is"]["sd"] * inits["h0_is_raw"]
            h0_pc = stan_data["mu_h0_pc"] + priors["mu_h0_pc"]["sd"] * inits["h0_pc_raw"]
            F, Q, A, Z, _ = build_lw_matrices({**params, "sigma_is": 1.0, "sigma_pc": 1.0}, c=1.0)
            R = sv_diag_variance_path(
                sv_rw_noncentered(h0_is, params["sigma_h_is"], nu_is),
                sv_rw_noncentered(h0_pc, params["sigma_h_pc"], nu_pc),
            )
        else:
            inits.update(sigma_is=params["sigma_is"], sigma_pc=params["sigma_pc"])
            F, Q, A, Z, R = build_lw_matrices(params, c=1.0)
        points.append(MirrorPoint(inits=inits, loglik_python=kalman_loglik(yobs, x, F, Q, A, Z, R, xi00, P00)))
    return points


def _mirror_decl():
    from macrotoolkit.qc import MirrorDecl

    return MirrorDecl(draw_points=mirror_points)


#: The registry's ``mirror`` capability (resolved lazily; see qc.py).
MIRROR = _mirror_decl()
