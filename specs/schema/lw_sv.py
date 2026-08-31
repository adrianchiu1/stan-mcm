"""Options + priors fragment for the ``lw_sv`` family (lw-sv-spec.md §2.3's
draft schema; S2 registers the no-SV variant).

S2 scope (spec §7): "LW without SV". Concretely:

- ``sv_shocks`` must be ``[]`` -- the SV block is S3 scope. The field exists
  now (with the spec's draft shape) so S3 flips it on without a schema
  migration, but any non-empty value is rejected with a clear error until
  the S3 template lands.
- ``estimate_c`` must be ``False`` -- spec §1.3's default. Promoting ``c``
  to a parameter (prior N(1, 0.25²) truncated positive) is deferred; the
  flag is validated, not silently ignored.

Priors: the §1.6 defaults live here as ``DEFAULT_PRIORS`` -- a single
source of truth consumed by the Stan template's render context
(``macrotoolkit.run.build_render_context``), the G1/G2 parameter-point
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
}

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


class LwSvOptions(BaseModel):
    """Options for ``model.family: lw_sv`` (spec §2.3 draft schema)."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["lw_sv"] = "lw_sv"
    sv_shocks: list[Literal["is", "pc"]] = Field(default_factory=list)
    estimate_c: bool = False

    @field_validator("sv_shocks")
    @classmethod
    def _no_sv_in_s2(cls, v: list[str]) -> list[str]:
        if v:
            raise ValueError(
                "model.options.sv_shocks must be [] for now: the SV block is "
                "S3 scope (lw-sv-spec.md §7) and its template does not exist "
                "yet. S2 estimates the LW model without SV -- constant "
                "sigma_is/sigma_pc shock scales."
            )
        return v

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
