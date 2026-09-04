"""Options + priors fragment for the ``ucsv`` family (family #2, S6 WP2;
scoped by docs/kf-capability-matrix.md): Stock-Watson (2007) unobserved-
components trend inflation with stochastic volatility on BOTH shocks.

Model::

    pi_t  = tau_t + eps_t,        eps_t ~ N(0, exp(h_eps,t))   (transitory)
    tau_t = tau_{t-1} + eta_t,    eta_t ~ N(0, exp(h_eta,t))   (trend)
    h_s,t = h_s,t-1 + sigma_h_s * nu_s,t,   h_s,0 ~ N(mu_h0_s, sd^2)
    tau_0 ~ N(pi_1, sd_tau0^2)    (explicit initial state, spec §2.2)

Univariate, complete data, NO exogenous regressors (the engine's empty-
feedback-map degenerate case). h is log-VARIANCE everywhere (sd =
exp(h/2)); inflation is whatever the user's column is (the worked example
uses annualized q/q core PCE, 400*dlog P, as lw_sv does).

``sv_shocks`` follows lw_sv's discipline: ``[eps, eta]`` (the production
variant -- the DEFAULT here, since "SV on both shocks" IS the family) or
``[]`` (constant variances; the toy no-SV variant, exercised by the
constant-Q G1 path). Single-shock combinations are rejected as
unvalidated. The list is normalized to canonical order so run identity
never depends on how a spec ordered it.

Priors (``DEFAULT_PRIORS``, the single source of truth consumed by the
render context, the prior sampler, and the validation designs):

- ``sigma_h_eps``, ``sigma_h_eta`` ~ Half-N(0, 0.2^2): the log-variance
  random-walk scales, the same magnitude lw_sv's SV block uses (spec
  §1.6) -- boundary-prone scale parameters whose prior deliberately does
  identification work (ENGINEERING.md), hence swept by the family's
  registered validation designs.
- ``mu_h0_eps``, ``mu_h0_eta``: h_0 ~ N(mu_h0, 1^2) with the MEAN a data
  anchor (``macrotoolkit.families.ucsv.ucsv_mu_h0_anchors``: the equal
  split of Var(Delta pi) = sigma_eta^2 + 2 sigma_eps^2 between the two
  terms, i.e. mu_h0_eta = ln(v/2), mu_h0_eps = ln(v/4) -- a deliberately
  agnostic split, weak at sd 1 in log-variance); only the sd is a stamped,
  overridable prior field, as in lw_sv.
- No-SV variant only: ``sigma_eps``, ``sigma_eta`` ~ Half-N(0, 1^2)
  (weakly informative constant scales, lw_sv's sigma_is/sigma_pc
  convention).
- Initial state: ``INITIAL_STATE_PRIOR["tau0_sd"] = 5.0`` percentage
  points around the first observation -- "diffuse-ish", the established
  explicit-prior pattern (no exact-diffuse recursion, per the capability
  matrix's doctrine).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_PRIORS: dict[str, dict] = {
    # SV variant (the production family).
    "sigma_h_eps": {"dist": "half_normal", "sd": 0.2},
    "sigma_h_eta": {"dist": "half_normal", "sd": 0.2},
    "mu_h0_eps": {"dist": "normal", "sd": 1.0},  # mean = data anchor
    "mu_h0_eta": {"dist": "normal", "sd": 1.0},  # mean = data anchor
    # No-SV variant: constant transitory/trend shock scales.
    "sigma_eps": {"dist": "half_normal", "sd": 1.0},
    "sigma_eta": {"dist": "half_normal", "sd": 1.0},
}

NO_SV_ONLY_PRIOR_NAMES = frozenset({"sigma_eps", "sigma_eta"})
SV_ONLY_PRIOR_NAMES = frozenset({"sigma_h_eps", "sigma_h_eta", "mu_h0_eps", "mu_h0_eta"})

#: tau_0 ~ N(pi_first, tau0_sd^2) -- explicit initial state.
INITIAL_STATE_PRIOR = {"tau0_sd": 5.0}


class ThinSpec(BaseModel):
    """``outputs.smoother_draws: {thin: k}`` (as lw_sv's)."""

    model_config = ConfigDict(extra="forbid")

    thin: int = Field(gt=0, description="Use every k-th posterior draw; must be >= 1.")


class UcsvOutputs(BaseModel):
    """``outputs:`` for ucsv -- the lw_sv field set minus ``forecast_r_rule``
    (UCSV has no exogenous input to forecast)."""

    model_config = ConfigDict(extra="forbid")

    horizon: int = Field(default=12, ge=1, description="Fan-chart forecast horizon in quarters.")
    irf_horizon: int = Field(default=20, ge=1, description="IRF horizon in quarters.")
    irf_vol_reference: Literal["end_of_sample", "sample_mean"] = Field(
        default="end_of_sample",
        description="Reference point for the 'one-sd shock' IRF convention.",
    )
    smoother_draws: Literal["all"] | ThinSpec = Field(
        default="all",
        description="'all' (every posterior draw) or {thin: k} (every k-th draw).",
    )
    prior_predictive_draws: int = Field(default=200, ge=1, description="Prior-predictive paths simulated for the report figure.")


class UcsvOptions(BaseModel):
    """Options for ``model.family: ucsv``."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["ucsv"] = "ucsv"
    sv_shocks: list[Literal["eps", "eta"]] = Field(default_factory=lambda: ["eps", "eta"])

    @field_validator("sv_shocks")
    @classmethod
    def _validated_combinations_only(cls, v: list[str]) -> list[str]:
        unique = set(v)
        if len(unique) != len(v):
            raise ValueError(f"model.options.sv_shocks has duplicate entries: {v!r}.")
        if unique and unique != {"eps", "eta"}:
            raise ValueError(
                f"model.options.sv_shocks must be [eps, eta] (SV on both shocks "
                f"-- the UCSV family proper, the default) or [] (constant "
                f"variances). Single-shock combinations are unvalidated. Got {v!r}."
            )
        return ["eps", "eta"] if unique else []
