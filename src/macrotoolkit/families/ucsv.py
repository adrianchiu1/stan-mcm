"""ucsv family numerics module (S6 WP2, family #2): named state metadata,
matrix builders, spec-driven data/render builders, the prior sampler, and
the sweep/report capabilities -- ALL declarations over the shared
machinery (the Q_t-generalized KF, the DK simulation smoother, the generic
simulate/IRF/HD engine). No UCSV-specific recursion exists anywhere.

Model (specs/schema/ucsv.py):

    pi_t  = tau_t + eps_t,        eps_t ~ N(0, exp(h_eps,t))
    tau_t = tau_{t-1} + eta_t,    eta_t ~ N(0, exp(h_eta,t))

State-space form: state n=1 ``xi_t = [tau_t]``, obs m=1 ``yobs_t = [pi_t]``,
NO exogenous block (k=0: ``x`` is (T, 0), ``A`` is (0, 1), the engine's
declared degenerate case -- the feedback map is EMPTY)::

    F = [1],  Q_t = [exp(h_eta,t)],  Z = [1],  R_t = [exp(h_eps,t)]

Both time-varying covariances go through ``macrotoolkit.smoother``'s
(T, 1, 1) paths (``sv_scalar_variance_path``), mirroring the Stan
template's ``sv_scalar_variance_path`` -- h is log-VARIANCE (sd =
exp(h/2)), the convention shared by every family.

Units: inflation as supplied (the worked example uses annualized q/q
percent, 400*dlog P). No pre-sample lag quarters: the estimation sample is
every trimmed row (unlike lw_sv's 4-lag convention).
"""
from __future__ import annotations

import numpy as np

from macrotoolkit.families.base import StateSpaceMeta

#: Named state metadata for ucsv: one contemporaneous state slot, one
#: state shock loading 1.0 into it, one measurement shock, one observable,
#: no exogenous series, EMPTY feedback map.
UCSV_STATE_META = StateSpaceMeta(
    state_labels=(("tau", 0),),
    state_shocks=("eta",),
    shock_loadings={"eta": {("tau", 0): 1.0}},
    measurement_shocks=("eps",),
    obs_names=("pi",),
    exog_names=(),
    feedback_map=(),
)

OBS_NAMES = UCSV_STATE_META.obs_names

#: The family's SBC prior config (ENGINEERING.md "Stationarity for SBC"):
#: UCSV has no AR block, so nothing needs a stationarity override -- the
#: production priors ARE the SBC priors. Declared explicitly (empty) so
#: the validation designs consume one mechanism across families.
SBC_PRIOR_CONFIG: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Matrices (mirror of the ucsv template's inline constructors)
# ---------------------------------------------------------------------------


def build_ucsv_matrices(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the CONSTANT-variance (no-SV) ``(F, Q, A, Z, R)`` at one
    parameter point: ``params`` needs ``sigma_eta`` and ``sigma_eps`` (sd
    units). For an SV draw the caller replaces Q/R by the (T, 1, 1)
    variance paths ``sv_scalar_variance_path(h_eta)`` /
    ``sv_scalar_variance_path(h_eps)``; F/A/Z are parameter-free."""
    F = np.array([[1.0]])
    Q = np.array([[float(params["sigma_eta"]) ** 2]])
    A = np.zeros((0, 1))
    Z = np.array([[1.0]])
    R = np.array([[float(params["sigma_eps"]) ** 2]])
    return F, Q, A, Z, R


def build_ucsv_regressors(pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(yobs, x)``: yobs is (T, 1) = [pi_t]; x is the EMPTY (T, 0)
    exogenous matrix (no regressors)."""
    pi = np.asarray(pi, dtype=np.float64)
    if pi.ndim != 1 or pi.shape[0] < 2:
        raise ValueError(f"pi must be a 1-d series of at least 2 observations; got shape {pi.shape}.")
    return pi[:, None].copy(), np.zeros((pi.shape[0], 0))


def default_initial_state(pi_first: float) -> tuple[np.ndarray, np.ndarray]:
    """``tau_0 ~ N(pi_first, tau0_sd^2)`` as an explicit ``(xi00, P00)``
    (spec §2.2: explicit initial state, no diffuse hack)."""
    from specs.schema.ucsv import INITIAL_STATE_PRIOR

    sd = float(INITIAL_STATE_PRIOR["tau0_sd"])
    return np.array([float(pi_first)]), np.array([[sd**2]])


def ucsv_mu_h0_anchors(pi: np.ndarray) -> tuple[float, float]:
    """The h_0 prior MEANS, from the data, by an EQUAL SPLIT of the
    first-difference variance ``v = Var(Delta pi)`` between the two
    shocks: ``Delta pi_t = eta_t + eps_t - eps_{t-1}`` has variance
    ``sigma_eta^2 + 2 sigma_eps^2``, so crediting each term with ``v/2``
    gives ``sigma_eta^2 = v/2`` and ``sigma_eps^2 = v/4`` --
    ``mu_h0_eta = ln(v/2)``, ``mu_h0_eps = ln(v/4)`` (log-VARIANCE; the
    numerics-reviewer pass of 2026-09-04 caught the original ``ln(v/2)``
    for eps disagreeing with this derivation). An agnostic starting point
    only: the h_0 prior sd is 1.0 in log-variance (specs/schema/ucsv.py).
    Deterministic given the trimmed data (run identity is unaffected: the
    hash covers the raw file, not this dict). Returns
    ``(mu_h0_eps, mu_h0_eta)``."""
    pi = np.asarray(pi, dtype=np.float64)
    if pi.shape[0] < 3:
        raise ValueError(f"mu_h0 anchor needs at least 3 observations; got {pi.shape[0]}.")
    v = float(np.var(np.diff(pi), ddof=1))
    if not (np.isfinite(v) and v > 0.0):
        raise ValueError(
            f"mu_h0 anchor: Var(Delta pi) = {v!r} is not positive/finite -- "
            f"degenerate input data? The anchor needs genuine variation."
        )
    return float(np.log(v / 4.0)), float(np.log(v / 2.0))


# ---------------------------------------------------------------------------
# Spec-driven builders (registry capabilities)
# ---------------------------------------------------------------------------


def build_stan_data(df) -> dict:
    """The ucsv template's ``data`` block from the trimmed model-ready
    DataFrame (column ``pi``). Every trimmed row is an estimation
    observation (no pre-sample lags)."""
    pi = df["pi"].to_numpy(dtype=np.float64)
    if pi.shape[0] < 3:
        raise ValueError(
            f"ucsv needs at least 3 data rows; the trimmed data has {pi.shape[0]}."
        )
    yobs, x = build_ucsv_regressors(pi)
    xi00, P00 = default_initial_state(float(pi[0]))
    mu_h0_eps, mu_h0_eta = ucsv_mu_h0_anchors(pi)
    return {
        "T": int(yobs.shape[0]),
        "yobs": yobs,
        "xi00": xi00,
        "P00": P00,
        "mu_h0_eps": mu_h0_eps,
        "mu_h0_eta": mu_h0_eta,
    }


def build_render_context(spec) -> dict:
    """Template context: resolved priors (defaults + variant-checked
    overrides, lw_sv's exact discipline) and the sv_shocks flag."""
    from specs.schema.ucsv import (
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
                f"priors[{name!r}] is not a parameter of the ucsv family. "
                f"Valid names: {sorted(priors)}."
            )
        if name in inactive:
            variant = "sv_shocks: [eps, eta]" if sv_on else "sv_shocks: []"
            raise ValueError(
                f"priors[{name!r}] does not exist in the variant this spec "
                f"selects ({variant}) -- the override would silently do "
                f"nothing. These prior names belong only to the other "
                f"variant: {sorted(inactive)}."
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
    return {"priors": priors, "sv_shocks": list(spec.model.options.sv_shocks)}


def sample_prior_params(
    priors: dict[str, dict],
    sv_on: bool,
    rng: np.random.Generator,
    mu_h0_eps: float | None = None,
    mu_h0_eta: float | None = None,
) -> dict[str, float]:
    """ONE parameter point from the ucsv prior, mirroring the template's
    declarations exactly (half_normal = |N(0, sd^2)|; h_0 ~ N(mu_h0, sd)
    at the supplied data anchors for the SV variant)."""

    def half_normal(name: str) -> float:
        entry = priors[name]
        if entry["dist"] != "half_normal":
            raise ValueError(f"{name!r} is expected to be half_normal in the ucsv template; got {entry!r}.")
        return float(abs(rng.normal(0.0, entry["sd"])))

    if sv_on:
        if mu_h0_eps is None or mu_h0_eta is None:
            raise ValueError("sample_prior_params: sv_on=True needs the data-derived mu_h0_eps/mu_h0_eta anchors.")
        return {
            "sigma_h_eps": half_normal("sigma_h_eps"),
            "sigma_h_eta": half_normal("sigma_h_eta"),
            "h0_eps": float(rng.normal(mu_h0_eps, priors["mu_h0_eps"]["sd"])),
            "h0_eta": float(rng.normal(mu_h0_eta, priors["mu_h0_eta"]["sd"])),
        }
    return {"sigma_eps": half_normal("sigma_eps"), "sigma_eta": half_normal("sigma_eta")}


def prior_scalar_sds(spec, df, n_draws: int = 10_000, seed: int = 20260902) -> dict[str, float]:
    """Monte-Carlo prior sds of the scalar parameters under the spec's
    resolved priors (the sweep report's contraction readout)."""
    resolved = build_render_context(spec)["priors"]
    sv_on = bool(spec.model.options.sv_shocks)
    mu_eps = mu_eta = None
    if sv_on:
        mu_eps, mu_eta = ucsv_mu_h0_anchors(df["pi"].to_numpy(dtype=np.float64))
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(n_draws):
        p = sample_prior_params(resolved, sv_on, rng, mu_h0_eps=mu_eps, mu_h0_eta=mu_eta)
        for k, v in p.items():
            draws.setdefault(k, []).append(v)
    return {k: float(np.std(np.asarray(v))) for k, v in draws.items()}


def headline_series(run_dir, thin: int = 5, seed: int | None = None) -> dict[str, tuple]:
    """Posterior-median trend inflation tau for cross-run overlays (the
    sweep report), smoother draws thinned x``thin`` report-side."""
    import dataclasses

    from macrotoolkit.results_ucsv import compute_trend_cycle_draws, load_ucsv_run
    from specs.schema.ucsv import ThinSpec

    ucsv_run = load_ucsv_run(run_dir)
    if thin > 1:
        ucsv_run = dataclasses.replace(
            ucsv_run,
            spec=ucsv_run.spec.model_copy(
                update={"outputs": ucsv_run.spec.outputs.model_copy(update={"smoother_draws": ThinSpec(thin=thin)})}
            ),
        )
    tcd = compute_trend_cycle_draws(ucsv_run, seed=seed)
    return {"tau": (tcd.dates, np.median(tcd.series["tau"], axis=0))}


# ---------------------------------------------------------------------------
# Automatic fit-time mirror check (S6 WP3)
# ---------------------------------------------------------------------------


def mirror_points(spec, stan_data: dict, n: int, rng: np.random.Generator):
    """``n`` prior draws as Stan inits + the Python KF mirror at each: for
    the SV variant Q_t/R_t from ``sv_rw_noncentered`` ->
    ``sv_scalar_variance_path`` on the trend/transitory log-variance paths
    (exactly the template's composition); constant scales otherwise."""
    from macrotoolkit.qc import MirrorPoint
    from macrotoolkit.smoother import kalman_loglik, sv_rw_noncentered, sv_scalar_variance_path

    priors = build_render_context(spec)["priors"]
    sv_on = bool(spec.model.options.sv_shocks)
    T = int(stan_data["T"])
    yobs, xi00, P00 = stan_data["yobs"], stan_data["xi00"], stan_data["P00"]
    x = np.zeros((T, 0))
    points = []
    for _ in range(n):
        params = sample_prior_params(
            priors, sv_on, rng, mu_h0_eps=stan_data.get("mu_h0_eps"), mu_h0_eta=stan_data.get("mu_h0_eta")
        )
        if sv_on:
            nu_eps = rng.standard_normal(T)
            nu_eta = rng.standard_normal(T)
            h0_eps_raw = (params["h0_eps"] - stan_data["mu_h0_eps"]) / priors["mu_h0_eps"]["sd"]
            h0_eta_raw = (params["h0_eta"] - stan_data["mu_h0_eta"]) / priors["mu_h0_eta"]["sd"]
            inits = {
                "sigma_h_eps": params["sigma_h_eps"], "sigma_h_eta": params["sigma_h_eta"],
                "h0_eps_raw": h0_eps_raw, "h0_eta_raw": h0_eta_raw, "nu_eps": nu_eps, "nu_eta": nu_eta,
            }
            h0_eps = stan_data["mu_h0_eps"] + priors["mu_h0_eps"]["sd"] * h0_eps_raw
            h0_eta = stan_data["mu_h0_eta"] + priors["mu_h0_eta"]["sd"] * h0_eta_raw
            F, _, A, Z, _ = build_ucsv_matrices({"sigma_eta": 1.0, "sigma_eps": 1.0})
            Q = sv_scalar_variance_path(sv_rw_noncentered(h0_eta, params["sigma_h_eta"], nu_eta))
            R = sv_scalar_variance_path(sv_rw_noncentered(h0_eps, params["sigma_h_eps"], nu_eps))
        else:
            inits = dict(params)
            F, Q, A, Z, R = build_ucsv_matrices(params)
        points.append(MirrorPoint(inits=inits, loglik_python=kalman_loglik(yobs, x, F, Q, A, Z, R, xi00, P00)))
    return points


def _mirror_decl():
    from macrotoolkit.qc import MirrorDecl

    return MirrorDecl(draw_points=mirror_points)


MIRROR = _mirror_decl()
