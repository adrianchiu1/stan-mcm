"""VAR support for authored models (S8 WP2): a VAR(p) in named
observables expanded into the RECURSIVE (Cholesky-ordered) measurement
equations the S8 grammar accepts (E5: contemporaneous observables
substituted acyclically, orthogonal structural shocks), and the
INDEPENDENT-NORMAL Minnesota prior of Blake & Mumtaz (2017) Chapter 2 §2
as per-coefficient normal priors.

The recursive form. With the ordering ``y_1, ..., y_N`` the equations are

    y_i = c_i + sum_{j<i} a0_i_j * y_j + sum_{k=1..p} sum_j b_i_j_k * y_j[-k] + e_i

with orthogonal shocks ``e_i ~ N(0, sigma_i^2)``. The compiler substitutes
each ``y_j`` (j < i) recursively, so the reduced-form equation i carries
``sum_{j<i} a0_i_j e_j + e_i`` and the reduced-form covariance is
``Sigma = M diag(sigma^2) M'`` with ``M`` unit lower triangular -- the
Cholesky factor of ``Sigma`` IS the structural impact matrix, and the
engine's structural IRFs under the ordering are the Cholesky IRFs (the
handbook's ``A0 = chol(Sigma)``, Chapter 2 example 2). The reduced-form
coefficient of equation i is the recursive one plus the ``a0`` combinations
of the earlier equations' coefficients (``b^{rf}_i = b_i + sum_{j<i} a0_i_j
b^{rf}_j``).

The Minnesota prior (``minnesota_priors``). Litterman's prior as the
handbook's example 1 writes it -- a DIAGONAL prior covariance ``H`` on
``vec(B)``:

    own lag k        ~ N(own_mean * [k == 1],  (lambda1 / k^lambda3)^2)
    cross lag k (j)  ~ N(0,  (s_i * lambda1 * lambda2 / (s_j * k^lambda3))^2)
    constant         ~ N(0,  (s_i * lambda4)^2)

with ``s_i`` the residual standard error of an AR(1)-with-constant OLS
regression of series i (the handbook's ``s1``/``s2``: ``sqrt(RSS/(T-2))``
over the estimation rows). This is the INDEPENDENT-NORMAL Minnesota
prior: it is NOT the natural-conjugate / normal-inverse-Wishart prior of
§3 (whose ``H`` is Kronecker-structured with Sigma) nor the dummy-
observation implementation of §5 (Banbura et al.). Correspondence:
conditional on Sigma, the handbook's Gibbs step draws ``vec(B) ~
N(M*, V*)`` with ``V* = (H^-1 + Sigma^-1 (x) X'X)^-1`` -- exactly the
posterior these normal priors imply for the reduced-form coefficients.
Two deliberate differences here: (1) the priors are placed on the
RECURSIVE-form coefficients ``b_i_j_k`` (equation i's own coefficients
before substitution), while the handbook's sit on the reduced form; the
two coincide for the first equation and differ for later ones by the
``a0`` combinations, whose own prior (``a0_sd``) is flat-ish; (2) Sigma's
inverse-Wishart prior is replaced by half-normal priors on the orthogonal
shock scales ``sigma_i`` and normal priors on ``a0``. State this when
reporting a Minnesota-prior VAR estimated here.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from macrotoolkit.authoring.dsl import Model, half_normal, init, normal


def var_names(observables: Sequence[str], p: int, *, intercept: bool = True) -> dict[str, list[str]]:
    """The parameter names ``var`` declares, by role: ``constants``
    (``c_<y>``), ``a0`` (``a0_<yi>_<yj>``, j before i in the ordering),
    ``lags`` (``b_<yi>_<yj>_<k>``), ``scales`` (``sigma_<y>``)."""
    obs = list(observables)
    out = {"constants": [f"c_{y}" for y in obs] if intercept else [], "a0": [], "lags": [], "scales": [f"sigma_{y}" for y in obs]}
    for i, yi in enumerate(obs):
        for yj in obs[:i]:
            out["a0"].append(f"a0_{yi}_{yj}")
        for k in range(1, p + 1):
            for yj in obs:
                out["lags"].append(f"b_{yi}_{yj}_{k}")
    return out


def var_parts(
    observables: Sequence[str],
    p: int,
    *,
    intercept: str | bool = "parameter",
    priors: Mapping[str, dict] | None = None,
    coef_sd: float = 10.0,
    a0_sd: float = 1.0,
    const_sd: float = 10.0,
    sigma_sd: float = 5.0,
    steady_state_init: Mapping[str, dict] | None = None,
    extra_terms: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The equation strings, parameter priors, shock declarations (and,
    for the steady-state form, the transition equations + initial
    conditions) of a recursive VAR(p) -- the pieces ``var`` assembles,
    exposed so a VAR block can be combined with states (a FAVAR's factor
    block, an exogenous block through ``extra_terms``).

    ``intercept``: ``"parameter"`` (the default; ``c_<y>`` parameters),
    ``"steady_state"`` (Villani's form: long-run means as shock-free
    CONSTANT STATES ``mu_<y>`` with ``steady_state_init[y] = init(mean,
    sd)`` -- the intercept of equation i is ``mu_i - sum_k sum_j b_i_j_k
    mu_j`` written as state terms with parameter-expression
    coefficients, E0 + E1), or ``False``/``None`` (no intercept).
    ``priors`` overrides the default flat-ish normals per parameter name
    (e.g. the output of ``minnesota_priors``). ``a0_sd`` defaults to 1:
    the contemporaneous coefficients are correlations-scale objects (the
    handbook's ``Sigma ~ IW(I, N+1)`` puts them at O(1)), and a wide prior
    on them makes the innovation Cholesky ill-conditioned at prior draws
    (``R22 - L21^2`` cancels ~4 digits per level of the recursion) -- with
    ``N(0, 10)`` the fit-time mirror check fails at the 1e-11 relative
    gate for a 4-variable VAR, a float64 fact, not a model error. ``extra_terms[y]`` appends
    a string to equation ``y`` (e.g. ``"+ g_y*x"`` for an exogenous
    regressor declared on the Model).
    """
    obs = list(observables)
    if p < 1:
        raise ValueError("A VAR needs p >= 1 lags.")
    if len(set(obs)) != len(obs):
        raise ValueError(f"observables must be distinct: {obs}.")
    if intercept is True:
        intercept = "parameter"
    if intercept not in ("parameter", "steady_state", False, None):
        raise ValueError(f"intercept must be 'parameter', 'steady_state' or False; got {intercept!r}.")
    priors = dict(priors or {})
    params: dict[str, dict] = {}
    shocks: dict[str, dict] = {}
    measurement: list[str] = []
    transition: list[str] = []
    initial_state: dict[str, dict] = {}
    if intercept == "steady_state":
        if not steady_state_init:
            raise ValueError("intercept='steady_state' needs steady_state_init = {observable: init(mean, sd)} for every observable.")
        for y in obs:
            if y not in steady_state_init:
                raise ValueError(f"steady_state_init lacks {y!r}.")
            transition.append(f"mu_{y} = mu_{y}[-1]")
            initial_state[f"mu_{y}"] = dict(steady_state_init[y])
    for i, yi in enumerate(obs):
        terms: list[str] = []
        if intercept == "parameter":
            terms.append(f"c_{yi}")
            params[f"c_{yi}"] = priors.get(f"c_{yi}", normal(0.0, const_sd))
        for yj in obs[:i]:
            terms.append(f"a0_{yi}_{yj}*{yj}")
            params[f"a0_{yi}_{yj}"] = priors.get(f"a0_{yi}_{yj}", normal(0.0, a0_sd))
        if intercept == "steady_state":
            # The RECURSIVE equation in deviations from the long-run means:
            #   y_i - mu_i = sum_{j<i} a0_i_j (y_j - mu_j) + sum_k sum_j b_i_j_k (y_j[-k] - mu_j) + e_i
            # so its intercept is (1 - sum_k b_i_i_k) mu_i - sum_{j != i} (a0_i_j [j < i] + sum_k b_i_j_k) mu_j.
            for j, yj in enumerate(obs):
                lag_sum = " + ".join(f"b_{yi}_{yj}_{k}" for k in range(1, p + 1))
                if yj == yi:
                    terms.append(f"(1 - {lag_sum.replace(' + ', ' - ')})*mu_{yj}")
                elif j < i:
                    terms.append(f"-(a0_{yi}_{yj} + {lag_sum})*mu_{yj}")
                else:
                    terms.append(f"-({lag_sum})*mu_{yj}")
        for k in range(1, p + 1):
            for yj in obs:
                nm = f"b_{yi}_{yj}_{k}"
                terms.append(f"{nm}*{yj}[-{k}]")
                params[nm] = priors.get(nm, normal(0.0, coef_sd))
        terms.append(f"e_{yi}")
        eq = f"{yi} = " + " + ".join(terms)
        if extra_terms and yi in extra_terms:
            eq += " " + extra_terms[yi].strip()
        measurement.append(eq)
        params[f"sigma_{yi}"] = priors.get(f"sigma_{yi}", half_normal(sigma_sd))
        shocks[f"e_{yi}"] = {"sd": f"sigma_{yi}"}
    unknown = sorted(set(priors) - set(params))
    if unknown:
        raise ValueError(f"priors name parameter(s) the VAR does not declare: {unknown}; declared: {sorted(params)}.")
    return {"measurement": measurement, "transition": transition, "parameters": params, "shocks": shocks, "initial_state": initial_state}


def var(
    name: str,
    observables: Sequence[str],
    p: int,
    *,
    intercept: str | bool = "parameter",
    priors: Mapping[str, dict] | None = None,
    coef_sd: float = 10.0,
    a0_sd: float = 1.0,
    const_sd: float = 10.0,
    sigma_sd: float = 5.0,
    steady_state_init: Mapping[str, dict] | None = None,
    exogenous: Sequence[str] | None = None,
    extra_terms: Mapping[str, str] | None = None,
    extra_parameters: Mapping[str, dict] | None = None,
    forecast_rules: Mapping[str, dict] | None = None,
) -> Model:
    """An authored recursive VAR(p) in ``observables`` (the Cholesky
    ordering = the list order). See :func:`var_parts` for the options;
    ``exogenous`` + ``extra_terms`` + ``extra_parameters`` add exogenous
    regressors (``extra_terms={"y": "+ g_y*x"}``, ``extra_parameters=
    {"g_y": normal(0, 1)}``)."""
    parts = var_parts(observables, p, intercept=intercept, priors=priors, coef_sd=coef_sd, a0_sd=a0_sd, const_sd=const_sd,
                      sigma_sd=sigma_sd, steady_state_init=steady_state_init, extra_terms=extra_terms)
    params = {**parts["parameters"], **dict(extra_parameters or {})}
    return Model(name, observables=list(observables), measurement=parts["measurement"], transition=parts["transition"] or None,
                 parameters=params, shocks=parts["shocks"], initial_state=parts["initial_state"] or None,
                 exogenous=list(exogenous or []), forecast_rules=forecast_rules)


def ar1_residual_sds(data, observables: Sequence[str], p: int) -> dict[str, float]:
    """``s_i``: the residual standard error of ``y_i = a + b y_i[-1] + u``
    by OLS over the ESTIMATION rows (the first ``p`` rows dropped, as the
    handbook drops them before its AR(1) pre-regressions), ``sqrt(RSS /
    (T - 2))`` -- example 1's ``s1``, ``s2``."""
    out = {}
    for y in observables:
        s = np.asarray(data[y], dtype=np.float64)[p:]
        if s.shape[0] < 4:
            raise ValueError(f"ar1_residual_sds: series {y!r} has {s.shape[0]} estimation rows; need at least 4.")
        yy, x = s[1:], np.column_stack([np.ones(s.shape[0] - 1), s[:-1]])
        b = np.linalg.lstsq(x, yy, rcond=None)[0]
        r = yy - x @ b
        out[y] = float(np.sqrt(r @ r / (yy.shape[0] - 2)))
    return out


def minnesota_priors(
    data,
    observables: Sequence[str],
    p: int,
    *,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda3: float = 1.0,
    lambda4: float = 1.0,
    own_mean: float | Mapping[str, float] = 1.0,
    residual_sds: Mapping[str, float] | None = None,
) -> dict[str, dict]:
    """Per-coefficient normal priors for :func:`var` in the handbook's
    Chapter 2 §2 arithmetic (module docstring): own lag 1 mean
    ``own_mean`` (a number or per observable; 1 for levels, 0 or 0.95 for
    growth rates as the handbook's examples choose), own lag k sd
    ``lambda1 / k^lambda3``, cross-lag sd ``s_i lambda1 lambda2 / (s_j
    k^lambda3)``, constant sd ``s_i lambda4``. ``residual_sds`` overrides
    the AR(1) pre-regression sds (else :func:`ar1_residual_sds` on
    ``data``). Only ``c_*`` and ``b_*`` entries are returned; ``a0_*`` and
    ``sigma_*`` keep :func:`var`'s defaults."""
    obs = list(observables)
    s = dict(residual_sds) if residual_sds is not None else ar1_residual_sds(data, obs, p)
    means = {y: float(own_mean[y]) for y in obs} if isinstance(own_mean, Mapping) else {y: float(own_mean) for y in obs}
    priors: dict[str, dict] = {}
    for yi in obs:
        priors[f"c_{yi}"] = normal(0.0, s[yi] * lambda4)
        for k in range(1, p + 1):
            for yj in obs:
                if yj == yi:
                    priors[f"b_{yi}_{yj}_{k}"] = normal(means[yi] if k == 1 else 0.0, lambda1 / k**lambda3)
                else:
                    priors[f"b_{yi}_{yj}_{k}"] = normal(0.0, s[yi] * lambda1 * lambda2 / (s[yj] * k**lambda3))
    return priors


def reduced_form(compiled, params: Mapping[str, float]) -> dict[str, np.ndarray]:
    """The reduced-form objects of a recursive VAR at a parameter point:
    ``B`` (k x N, the ``A`` matrix -- rows in the feedback-map column
    order, incl. the constant row when present), ``Sigma = M D M'`` and
    the Cholesky impact ``M diag(sigma)``."""
    A = compiled.build_A(params)
    R = compiled.build_R(params)
    M = compiled.build_M(params)
    if M is None:
        M = np.eye(compiled.meta.n_obs)
    sds = np.array([float(params[compiled.structure.shock_scale_param[s]]) for s in compiled.meta.measurement_shocks])
    return {"B": A, "Sigma": R, "impact": M * sds[None, :]}
