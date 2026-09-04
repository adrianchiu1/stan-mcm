"""Options + priors fragment for the ``lw_sv`` family (lw-sv-spec.md §2.3's
draft schema; S2 registered the no-SV variant, S3 adds SV).

Current scope (spec §7, S3):

- ``sv_shocks`` is ``[]`` (the no-SV variant) or ``[is, pc]`` -- spec
  §1.5/§2.3: SV on the IS and Phillips measurement shocks jointly is the
  ONE validated non-empty combination in v1. Single-shock combinations are
  rejected with a clear error (they would render but are unvalidated).
  The list is normalized to the canonical order ``["is", "pc"]`` so run
  identity never depends on how a spec happened to order it.
- ``estimate_c`` must be ``False`` -- spec §1.3's default. Promoting ``c``
  to a parameter (prior N(1, 0.25²) truncated positive) is deferred; the
  flag is validated, not silently ignored.

Priors: the §1.6 defaults live here as ``DEFAULT_PRIORS`` -- a single
source of truth consumed by the Stan template's render context
(``macrotoolkit.families.lw_sv.build_render_context``), the G1/G2 parameter-point
generators (``tests/g1_harness.py``), and the Python KF mirror's tests.
``RunSpec.priors`` keys override these per run.

``sigma_is`` / ``sigma_pc`` -- the no-SV variant's constant IS/Phillips
shock scales -- are NOT in spec §1.6's table (the spec only ever describes
those two shocks with SV). Their prior here, Half-N(0, 1²), is the
2026-08-31 user-confirmed decision recorded in DECISIONS.md: treat them
exactly like HLW's own ``sigma_ytilde``/``sigma_pi`` (freely estimated
constants, weakly-informative prior doing no identification work,
deliberately not centered on the G5a oracle's MLE values).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Spec §1.6 default priors (plus the confirmed sigma_is/sigma_pc entry).
#: Each value is (distribution, params) in the parameterization the Stan
#: template stamps directly: normal(mu, sd), beta(a, b), half_normal(sd).
#: "truncated" normals are expressed by the Stan parameter constraint
#: (<upper=0> etc.), not by the prior density itself -- matching how the
#: template declares a_r and b_y.
DEFAULT_PRIORS: dict[str, dict] = {
    "a1": {"dist": "normal", "mu": 1.2, "sd": 0.3},
    "a2": {"dist": "normal", "mu": -0.4, "sd": 0.3},
    "a_r": {"dist": "normal", "mu": -0.1, "sd": 0.05},  # <upper=0> constraint
    "b_pi": {"dist": "beta", "a": 8.0, "b": 2.0},
    "b_y": {"dist": "normal", "mu": 0.15, "sd": 0.1},  # <lower=0> constraint
    "sigma_ystar": {"dist": "half_normal", "sd": 0.4},
    "sigma_g": {"dist": "half_normal", "sd": 0.03},  # pile-up control (spec §1.6)
    "sigma_z": {"dist": "half_normal", "sd": 0.08},  # pile-up control (spec §1.6)
    # No-SV constant IS/PC shock scales -- DECISIONS.md 2026-08-31, not spec §1.6.
    "sigma_is": {"dist": "half_normal", "sd": 1.0},
    "sigma_pc": {"dist": "half_normal", "sd": 1.0},
    # SV variant (spec §1.5-§1.6; only in renders with sv_shocks == [is, pc]).
    # sigma_h_*: log-variance random-walk scales.
    "sigma_h_is": {"dist": "half_normal", "sd": 0.2},
    "sigma_h_pc": {"dist": "half_normal", "sd": 0.2},
    # mu_h0_*: the h_0 ~ N(mu_h0, sd^2) initial-log-variance prior. The MEAN
    # is data -- the 2*ln(sigma_hat_OLS) anchor build_stan_data computes
    # (HLW-exact OLS pass, DECISIONS.md 2026-08-31) -- so only the sd is a
    # stamped, overridable prior field here.
    "mu_h0_is": {"dist": "normal", "sd": 1.0},
    "mu_h0_pc": {"dist": "normal", "sd": 1.0},
}

#: Prior names that only exist in one variant (the other variant's render
#: never references them). Used by build_render_context to reject overrides
#: of priors that are INACTIVE for the run's sv_shocks -- otherwise a
#: typo'd variant override would silently do nothing (numerics-review
#: finding, 2026-08-31).
NO_SV_ONLY_PRIOR_NAMES = frozenset({"sigma_is", "sigma_pc"})
SV_ONLY_PRIOR_NAMES = frozenset({"sigma_h_is", "sigma_h_pc", "mu_h0_is", "mu_h0_pc"})

#: Initial-state prior (spec §1.6 "Initial states" row). y*_0 is anchored at
#: the first observation of y at run time; the numbers here are the sds/means
#: that don't depend on data. See macrotoolkit.smoother.default_initial_state
#: for how these become the 7-dim (xi_00, P_00) pair.
INITIAL_STATE_PRIOR = {
    "ystar_sd": 2.0,   # y*_0 ~ N(y_first, 2^2), "diffuse-ish"
    "g_mean": 3.0,     # g_0 ~ N(3, 1^2), annualized
    "g_sd": 1.0,
    "z_mean": 0.0,     # z_0 ~ N(0, 1^2)
    "z_sd": 1.0,
}


class ThinSpec(BaseModel):
    """``outputs.smoother_draws: {thin: k}`` -- use every ``k``-th posterior
    draw for the simulation-smoother-derived output modules (trend-cycle
    bands, fans, HD) instead of every draw. ``k`` must be a positive
    integer (``thin: 1`` is equivalent to, but not the same spelling as,
    ``smoother_draws: all``)."""

    model_config = ConfigDict(extra="forbid")

    thin: int = Field(gt=0, description="Use every k-th posterior draw; must be >= 1.")


class LwSvOutputs(BaseModel):
    """Options for ``lw_sv``'s ``outputs:`` block (spec §2.3 draft schema,
    §3.1-3.5's output modules). Defaults match the spec §2.3 example YAML.

    ``smoother_draws``: the Durbin-Koopman simulation smoother
    (``macrotoolkit.smoother``) re-runs once per posterior draw used here --
    a real per-draw cost, benchmarked and documented in DECISIONS.md
    (2026-08-31 S4 open question 2). ``"all"`` is the spec's own example
    default; ``{thin: k}`` is the documented fallback if the full pass is
    too slow for interactive report generation.

    ``forecast_r_rule``: spec §3.3 exposes three values in the schema but
    only mandates implementing two in v1 (``neutral``: hold the real-rate
    *gap* input at r_{T+h} = r*_{T+h}; ``last_value``: hold r_{T+h} at its
    last observed value). ``user_path`` is schema-only for now -- accepted
    by the type so the schema documents the full spec vocabulary, but
    rejected by a validator with a clear "not implemented yet" message
    (same pattern as ``LwSvOptions.estimate_c`` below), not silently
    treated as ``neutral``.
    """

    model_config = ConfigDict(extra="forbid")

    horizon: int = Field(default=12, ge=1, description="Fan-chart forecast horizon in quarters.")
    irf_horizon: int = Field(default=20, ge=1, description="IRF horizon in quarters.")
    irf_vol_reference: Literal["end_of_sample", "sample_mean"] = Field(
        default="end_of_sample",
        description="Reference point for the 'one-sd shock' IRF convention (spec §3.2).",
    )
    smoother_draws: Literal["all"] | ThinSpec = Field(
        default="all",
        description="'all' (every posterior draw) or {thin: k} (every k-th draw).",
    )
    forecast_r_rule: Literal["neutral", "last_value", "user_path"] = Field(
        default="neutral",
        description="Real-rate-gap convention for fan-chart forecasting (spec §3.3).",
    )
    # Since the run-identity split (S5-decisions item 3) outputs sits
    # OUTSIDE the hash, so adding report options like this one no longer
    # orphans existing MCMC runs.
    prior_predictive_draws: int = Field(
        default=200,
        ge=1,
        description=(
            "Number of full observable paths simulated from the run's own "
            "resolved priors for the prior-predictive check figure "
            "(spec §4, S5-decisions item 7)."
        ),
    )

    @field_validator("forecast_r_rule")
    @classmethod
    def _user_path_not_implemented(cls, v: str) -> str:
        if v == "user_path":
            raise ValueError(
                "outputs.forecast_r_rule: user_path is not implemented yet "
                "-- spec §3.3 mandates only 'neutral' and 'last_value' in "
                "v1. Use one of those, or supply a user path via a later "
                "stage once user_path exists."
            )
        return v


class LwSvOptions(BaseModel):
    """Options for ``model.family: lw_sv`` (spec §2.3 draft schema)."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["lw_sv"] = "lw_sv"
    sv_shocks: list[Literal["is", "pc"]] = Field(default_factory=list)
    estimate_c: bool = False

    @field_validator("sv_shocks")
    @classmethod
    def _validated_combinations_only(cls, v: list[str]) -> list[str]:
        unique = set(v)
        if len(unique) != len(v):
            raise ValueError(
                f"model.options.sv_shocks has duplicate entries: {v!r}."
            )
        if unique and unique != {"is", "pc"}:
            raise ValueError(
                f"model.options.sv_shocks must be [] (no SV) or [is, pc] "
                f"(SV on both measurement shocks) -- the one combination "
                f"validated in v1 (lw-sv-spec.md §1.5, §2.3). Got {v!r}."
            )
        # Canonical order: run identity must not depend on spec list order.
        return ["is", "pc"] if unique else []

    @field_validator("estimate_c")
    @classmethod
    def _c_fixed_in_s2(cls, v: bool) -> bool:
        if v:
            raise ValueError(
                "model.options.estimate_c: true is not implemented yet -- S2 "
                "fixes c at 1.0 (spec §1.3's default; matches HLW 2017, whose "
                "stage-3 model has no c term, i.e. c == 1 -- see "
                "plans/S2-plan.md open question 1). Use estimate_c: false."
            )
        return v
