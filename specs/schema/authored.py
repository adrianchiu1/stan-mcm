"""Options + outputs fragment for the ``authored`` family (S7): a model
written as EQUATIONS in the spec itself. The options model IS the model
definition -- observables, exogenous series, measurement and transition
equations (strings in the DSL of ``specs/schema/equations.py``), the prior
table, the shock declarations (constant scale or SV), explicit initial
conditions, and optional per-exogenous-series forecast rules -- so an
authored model is a declarative spec exactly like a hand-written family's
(VISION.md's spine) and every one of these fields enters the run-identity
hash through ``RunSpec.to_estimation_yaml``.

Validation happens HERE, at spec-parse time: every equation is parsed,
linearized and canonicalized (the canonical string replaces the user's
text, so formatting never changes the hash), and the full structural
compile (``specs/schema/authored_structure.py``) runs so any construct
outside the DSL's scope fails loudly with a message naming the limitation
before anything is rendered or sampled.

Scope fence (plans/S7-plan.md, widened by plans/S8-plan.md WP1):
linear-Gaussian state-space models only -- measurement equations linear
in states, lagged AND contemporaneous observables (substituted
recursively, S8 E5), exogenous series at any lag including 0 (E2), an
intercept (E1) and at most one iid Gaussian measurement shock of the
row's own (a shock-free row is allowed when a stochastic state explains
it, E3); transition equations linear in states and state shocks with an
optional drift (E1); no transition equation at all (a pure regression /
VAR, E0); non-centered random-walk log-variance SV on any shock; explicit
initial conditions; priors from the menu the existing templates stamp
(normal with optional truncation bounds, half_normal, beta).
A STATE multiplied by a lagged observable, a ``mean()`` of one
observable or an exogenous series (S9 E4: a time-varying coefficient,
compiled to a data-dependent loading ``Z_t``) is allowed in a measurement
equation. Nonlinearities beyond that one bilinear shape, regime
switching, missing data, mixed frequency, time-varying transitions and
exact-diffuse initialization are rejected with a message.
"""
from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from specs.schema.authored_structure import ModelStructure, ModelStructureError, compile_structure
from specs.schema.equations import EquationSyntaxError, LinearityError, canonical_equation


# ---------------------------------------------------------------------------
# Priors (the menu the existing templates already stamp)
# ---------------------------------------------------------------------------


class NormalPrior(BaseModel):
    """``normal(mu, sd)``; ``lower``/``upper`` truncate through the Stan
    parameter constraint (lw_sv's ``a_r <upper=0>`` / ``b_y <lower=0>``
    pattern -- the density is renormalized by Stan, sampled by rejection
    on the Python side)."""

    model_config = ConfigDict(extra="forbid")

    dist: Literal["normal"]
    mu: float
    sd: float = Field(gt=0)
    lower: float | None = None
    upper: float | None = None

    @model_validator(mode="after")
    def _bounds(self) -> "NormalPrior":
        if self.lower is not None and self.upper is not None and not self.lower < self.upper:
            raise ValueError(f"normal prior bounds need lower < upper; got lower={self.lower}, upper={self.upper}.")
        return self


class HalfNormalPrior(BaseModel):
    """``half_normal(sd)`` = |N(0, sd^2)|, declared ``<lower=0>``."""

    model_config = ConfigDict(extra="forbid")

    dist: Literal["half_normal"]
    sd: float = Field(gt=0)


class BetaPrior(BaseModel):
    """``beta(a, b)`` on (0, 1)."""

    model_config = ConfigDict(extra="forbid")

    dist: Literal["beta"]
    a: float = Field(gt=0)
    b: float = Field(gt=0)


Prior = Annotated[Union[NormalPrior, HalfNormalPrior, BetaPrior], Field(discriminator="dist")]


# ---------------------------------------------------------------------------
# Shocks, initial state, forecast rules
# ---------------------------------------------------------------------------


class LogVarDiffAnchor(BaseModel):
    """``mu_h0 = ln(fraction * Var(Delta series))`` -- the one data rule the
    hand families' anchors reduce to for an authored model (UCSV's equal
    split: fraction 1/4 for the transitory shock, 1/2 for the trend shock).
    Deterministic given the trimmed data; run identity covers the raw
    file, not the derived number (the established convention)."""

    model_config = ConfigDict(extra="forbid")

    anchor: Literal["log_var_diff"]
    series: str
    fraction: float = Field(default=1.0, gt=0)


class SvSpec(BaseModel):
    """Non-centered random-walk log-variance SV on one shock (the
    established block): ``h_t = h_{t-1} + sigma_h * nu_t``, ``h_0 ~
    N(mu_h0, h0_sd^2)``; h is log-VARIANCE (sd = exp(h/2)). Adds the
    prior-table entries ``sigma_h_<shock>`` (half_normal) and
    ``mu_h0_<shock>`` (normal with the given sd; mean = ``mu_h0``)."""

    model_config = ConfigDict(extra="forbid")

    sigma_h: HalfNormalPrior = Field(default_factory=lambda: HalfNormalPrior(dist="half_normal", sd=0.2))
    h0_sd: float = Field(default=1.0, gt=0)
    mu_h0: float | LogVarDiffAnchor = 0.0


class ShockSpec(BaseModel):
    """Exactly one of ``sd`` (the name of a declared half_normal parameter
    carrying the constant scale) or ``sv``."""

    model_config = ConfigDict(extra="forbid")

    sd: str | None = None
    sv: SvSpec | None = None

    @model_validator(mode="after")
    def _one_of(self) -> "ShockSpec":
        if (self.sd is None) == (self.sv is None):
            raise ValueError("a shock declares exactly one of 'sd' (a half_normal parameter name) or 'sv' (an SV block).")
        return self


class FirstObs(BaseModel):
    """Initial-state mean anchored at the first ESTIMATION-sample value
    of an observable (lw_sv's ``y*_0 ~ N(y_first, .)`` / UCSV's ``tau_0 ~
    N(pi_1, .)``)."""

    model_config = ConfigDict(extra="forbid")

    first_obs: str


class InitialStateSpec(BaseModel):
    """``s_0 ~ N(mean, sd^2)``, applied to every slot of the state (the
    lag slots get the same marginal, a priori independent -- lw_sv's
    documented simplification, ``smoother.default_initial_state``)."""

    model_config = ConfigDict(extra="forbid")

    mean: float | FirstObs
    sd: float = Field(gt=0)


class ForecastRule(BaseModel):
    """How an exogenous series continues out of sample (fan charts):
    ``last_value`` (hold), ``constant`` (``value``), or ``state_linear``
    (a linear combination of state slots, keys like ``"g[-1]"``) -- the
    engine's three rule classes."""

    model_config = ConfigDict(extra="forbid")

    rule: Literal["last_value", "constant", "state_linear"]
    value: float | None = None
    terms: dict[str, float] | None = None

    @model_validator(mode="after")
    def _fields(self) -> "ForecastRule":
        if self.rule == "constant" and self.value is None:
            raise ValueError("forecast rule 'constant' needs 'value'.")
        if self.rule == "state_linear" and not self.terms:
            raise ValueError("forecast rule 'state_linear' needs non-empty 'terms' ({'g[-1]': 1.0, ...}).")
        if self.rule == "last_value" and (self.value is not None or self.terms is not None):
            raise ValueError("forecast rule 'last_value' takes no 'value'/'terms'.")
        return self


_SLOT_KEY_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)(?:\[-(\d+)\])?$")


def parse_slot_key(key: str) -> tuple[str, int]:
    """``"g[-1]"`` -> ``("g", -1)``; ``"tau"`` -> ``("tau", 0)``."""
    m = _SLOT_KEY_RE.match(key.strip())
    if not m:
        raise ValueError(f"{key!r} is not a state-slot key; write 'name' or 'name[-k]'.")
    return m.group(1), -int(m.group(2) or 0)


class Equations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    measurement: list[str] = Field(min_length=1)
    transition: list[str] = Field(default_factory=list)  # may be empty (S8 E0: a pure regression / VAR has no state)


# ---------------------------------------------------------------------------
# The options model = the model definition
# ---------------------------------------------------------------------------


class AuthoredOptions(BaseModel):
    """``model.options`` for ``model.family: authored`` -- see the module
    docstring. After validation ``equations`` hold the CANONICAL strings
    and :attr:`structure` (excluded from serialization) the compiled
    layout."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["authored"] = "authored"
    name: str
    observables: list[str] = Field(min_length=1)
    exogenous: list[str] = Field(default_factory=list)
    equations: Equations
    parameters: dict[str, Prior] = Field(default_factory=dict)
    shocks: dict[str, ShockSpec]
    initial_state: dict[str, InitialStateSpec]
    forecast_rules: dict[str, ForecastRule] = Field(default_factory=dict)

    _structure: ModelStructure | None = PrivateAttr(default=None)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model.options.name must be a non-empty label.")
        if any(ch in v for ch in "/\\"):
            raise ValueError(f"model.options.name {v!r} must not contain path separators.")
        return v

    @model_validator(mode="after")
    def _compile(self) -> "AuthoredOptions":
        try:
            structure = compile_structure(
                name=self.name,
                observables=list(self.observables),
                exogenous=list(self.exogenous),
                measurement=list(self.equations.measurement),
                transition=list(self.equations.transition),
                parameters=list(self.parameters),
                shocks={k: ({"sv": True} if v.sv is not None else {"sd": v.sd}) for k, v in self.shocks.items()},
            )
        except (EquationSyntaxError, LinearityError, ModelStructureError) as exc:
            raise ValueError(f"model.options (authored model {self.name!r}): {exc}") from None
        # Canonicalize the equations in place (formatting never reaches the hash).
        self.equations.measurement = [canonical_equation(e) for e in self.equations.measurement]
        self.equations.transition = [canonical_equation(e) for e in self.equations.transition]
        # Shock scales must be declared half_normal parameters.
        for sh, decl in self.shocks.items():
            if decl.sd is not None:
                prior = self.parameters.get(decl.sd)
                if prior is None:
                    raise ValueError(
                        f"model.options.shocks[{sh!r}].sd = {decl.sd!r} is not a declared parameter; declare it under "
                        f"parameters as {{dist: half_normal, sd: <scale>}}."
                    )
                if not isinstance(prior, HalfNormalPrior):
                    raise ValueError(
                        f"model.options.shocks[{sh!r}].sd = {decl.sd!r} must be a half_normal parameter (a positive "
                        f"scale); got dist={prior.dist!r}."
                    )
            else:
                anchor = decl.sv.mu_h0
                if isinstance(anchor, LogVarDiffAnchor) and anchor.series not in self.observables:
                    raise ValueError(
                        f"model.options.shocks[{sh!r}].sv.mu_h0.series = {anchor.series!r} is not an observable "
                        f"(the log_var_diff anchor is computed from an observed series)."
                    )
        # Initial state: every state, nothing else.
        states = set(structure.state_names)
        missing = [s for s in structure.state_names if s not in self.initial_state]
        extra = [s for s in self.initial_state if s not in states]
        if missing or extra:
            raise ValueError(
                f"model.options.initial_state must give mean/sd for every state and nothing else: missing {missing}, "
                f"unknown {extra} (states: {list(structure.state_names)}). The KF takes an explicit (xi00, P00); there is no diffuse hack."
            )
        for s, init in self.initial_state.items():
            if isinstance(init.mean, FirstObs) and init.mean.first_obs not in self.observables:
                raise ValueError(f"model.options.initial_state[{s!r}].mean.first_obs = {init.mean.first_obs!r} is not an observable.")
        # Forecast rules: exogenous series only; state_linear terms must be real slots.
        for name, rule in self.forecast_rules.items():
            if name not in self.exogenous:
                raise ValueError(f"model.options.forecast_rules[{name!r}]: not an exogenous series ({list(self.exogenous)}).")
            if rule.rule == "state_linear":
                for key in rule.terms or {}:
                    try:
                        label = parse_slot_key(key)
                    except ValueError as exc:
                        raise ValueError(f"model.options.forecast_rules[{name!r}].terms: {exc}") from None
                    if label not in structure.state_labels:
                        raise ValueError(
                            f"model.options.forecast_rules[{name!r}].terms key {key!r} is not a state slot; "
                            f"slots: {[f'{n}[{o}]' if o else n for n, o in structure.state_labels]}."
                        )
        self._structure = structure
        return self

    @property
    def structure(self) -> ModelStructure:
        """The compiled structural layout (not serialized)."""
        assert self._structure is not None
        return self._structure

    # --- the prior table -----------------------------------------------
    def prior_table(self) -> dict[str, dict[str, Any]]:
        """The family-style DEFAULT prior table: ``{name: {dist, ...}}``
        for every declared parameter plus, per SV shock, ``sigma_h_<s>``
        and ``mu_h0_<s>`` (mean = the anchor, so only ``sd`` is a stamped
        field -- lw_sv/ucsv's convention). Truncation bounds are structure
        (see :meth:`bounds`), not prior fields."""
        table: dict[str, dict[str, Any]] = {}
        for name, prior in self.parameters.items():
            entry = prior.model_dump()
            entry.pop("lower", None)
            entry.pop("upper", None)
            table[name] = entry
        for sh, decl in self.shocks.items():
            if decl.sv is not None:
                table[f"sigma_h_{sh}"] = decl.sv.sigma_h.model_dump()
                table[f"mu_h0_{sh}"] = {"dist": "normal", "sd": decl.sv.h0_sd}
        return table

    def bounds(self) -> dict[str, tuple[float | None, float | None]]:
        """Per parameter ``(lower, upper)`` constraint (None = unbounded);
        half_normal is ``(0, None)``, beta ``(0, 1)``."""
        out: dict[str, tuple[float | None, float | None]] = {}
        for name, prior in self.parameters.items():
            if isinstance(prior, NormalPrior):
                out[name] = (prior.lower, prior.upper)
            elif isinstance(prior, HalfNormalPrior):
                out[name] = (0.0, None)
            else:
                out[name] = (0.0, 1.0)
        for sh, decl in self.shocks.items():
            if decl.sv is not None:
                out[f"sigma_h_{sh}"] = (0.0, None)
        return out


class ThinSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thin: int = Field(gt=0, description="Use every k-th posterior draw; must be >= 1.")


class AuthoredOutputs(BaseModel):
    """``outputs:`` for authored models -- the ucsv field set (no
    family-specific forecast option: exogenous forecast rules are part of
    the model definition)."""

    model_config = ConfigDict(extra="forbid")

    horizon: int = Field(default=12, ge=1)
    irf_horizon: int = Field(default=20, ge=1)
    irf_vol_reference: Literal["end_of_sample", "sample_mean"] = "end_of_sample"
    #: S9 E4: for a model with data-dependent loadings the IRFs are
    #: conditional on the coefficient state at a reference date; these ISO
    #: dates (estimation rows) select them -- empty = the last estimation
    #: row. Ignored for a model without products.
    irf_dates: list[str] = Field(default_factory=list)
    smoother_draws: Literal["all"] | ThinSpec = "all"
    prior_predictive_draws: int = Field(default=200, ge=1)


def required_mapping(options: AuthoredOptions) -> tuple[str, ...]:
    """The ``data.mapping`` keys an authored model needs: its observables
    and exogenous series (the registry's ``dynamic_required_mapping``)."""
    return tuple(options.observables) + tuple(options.exogenous)
