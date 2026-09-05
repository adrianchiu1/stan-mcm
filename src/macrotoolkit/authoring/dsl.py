"""Notebook helpers for authoring (S7): tiny constructors that build the
SAME plain dicts the YAML form carries (``specs/schema/authored.py``), so
a model authored in Python and one loaded from YAML validate to the same
``AuthoredOptions`` and therefore the same run identity. Nothing here is a
second spec: :class:`Model` is a convenience over the options dict, and
``mtk.spec("authored", options=model, data=...)`` accepts it directly.

    from macrotoolkit import authoring as au
    model = au.Model(
        "uc_gap",
        observables=["y", "pi"],
        measurement=["y = ystar + c + e_y", "pi = b_pi*pi[-1] + (1 - b_pi)*mean(pi[-2], pi[-3], pi[-4]) + b_y*c[-1] + e_pi"],
        transition=["ystar = ystar[-1] + 0.25*g[-1] + eta_ystar", "g[-1] = g[-2] + eta_g", "c = a1*c[-1] + a2*c[-2] + eta_c"],
        parameters={"a1": au.normal(1.2, 0.3), "b_y": au.normal(0.15, 0.1, lower=0), "sigma_c": au.half_normal(0.5), ...},
        shocks={"eta_c": au.shock("sigma_c"), "e_y": au.sv(sigma_h=0.2, mu_h0=au.log_var_diff("y", 0.25))},
        initial_state={"ystar": au.init(au.first_obs("y"), 2.0), "g": au.init(3.0, 1.0), "c": au.init(0.0, 2.0)},
    )
"""
from __future__ import annotations

from typing import Any, Mapping

from specs.schema.authored import AuthoredOptions


def normal(mu: float, sd: float, *, lower: float | None = None, upper: float | None = None) -> dict:
    out: dict[str, Any] = {"dist": "normal", "mu": float(mu), "sd": float(sd)}
    if lower is not None:
        out["lower"] = float(lower)
    if upper is not None:
        out["upper"] = float(upper)
    return out


def half_normal(sd: float) -> dict:
    return {"dist": "half_normal", "sd": float(sd)}


def beta(a: float, b: float) -> dict:
    return {"dist": "beta", "a": float(a), "b": float(b)}


def shock(sd: str) -> dict:
    """A constant-scale shock: ``sd`` names a declared half_normal parameter."""
    return {"sd": sd}


def log_var_diff(series: str, fraction: float = 1.0) -> dict:
    """``mu_h0 = ln(fraction * Var(Delta series))`` (UCSV's equal-split anchor)."""
    return {"anchor": "log_var_diff", "series": series, "fraction": float(fraction)}


def sv(*, sigma_h: float = 0.2, h0_sd: float = 1.0, mu_h0: float | dict = 0.0) -> dict:
    """SV on a shock: ``sigma_h`` is the half_normal scale of the
    log-variance random walk, ``h0_sd`` the sd of ``h_0 ~ N(mu_h0, .)``,
    ``mu_h0`` a number or :func:`log_var_diff`."""
    return {"sv": {"sigma_h": half_normal(sigma_h), "h0_sd": float(h0_sd), "mu_h0": mu_h0}}


def first_obs(observable: str) -> dict:
    return {"first_obs": observable}


def init(mean: float | dict, sd: float) -> dict:
    return {"mean": mean, "sd": float(sd)}


def forecast(rule: str, *, value: float | None = None, terms: Mapping[str, float] | None = None) -> dict:
    out: dict[str, Any] = {"rule": rule}
    if value is not None:
        out["value"] = float(value)
    if terms is not None:
        out["terms"] = dict(terms)
    return out


class Model:
    """An authored model definition; ``.options`` is the validated
    :class:`AuthoredOptions` (pass it, or the ``Model`` itself, to
    ``mtk.spec("authored", options=...)``); ``.to_dict()`` the plain
    canonical dict (what the YAML would say)."""

    def __init__(
        self,
        name: str,
        *,
        observables: list[str],
        measurement: list[str],
        transition: list[str] | None = None,
        parameters: Mapping[str, dict],
        shocks: Mapping[str, dict],
        initial_state: Mapping[str, dict] | None = None,
        exogenous: list[str] | None = None,
        forecast_rules: Mapping[str, dict] | None = None,
    ) -> None:
        raw = {
            "name": name,
            "observables": list(observables),
            "exogenous": list(exogenous or []),
            "equations": {"measurement": list(measurement), "transition": list(transition or [])},
            "parameters": {k: dict(v) for k, v in parameters.items()},
            "shocks": {k: dict(v) for k, v in shocks.items()},
            "initial_state": {k: dict(v) for k, v in (initial_state or {}).items()},
            "forecast_rules": {k: dict(v) for k, v in (forecast_rules or {}).items()},
        }
        self.options = AuthoredOptions.model_validate(raw)

    def to_dict(self) -> dict:
        return self.options.model_dump(mode="json")

    @property
    def structure(self):
        return self.options.structure

    def compiled(self):
        from macrotoolkit.authoring.compile import compile_model

        return compile_model(self.options)

    def describe(self) -> str:
        """A readable summary of the compiled layout (states/slots,
        shocks, regressors) for the notebook."""
        st = self.options.structure
        lines = [f"authored model {st.name!r}"]
        lines.append(f"  observables: {list(st.obs_names)}   exogenous: {list(st.exog_names)}")
        lines.append("  state vector: " + ", ".join(f"{n}[{o}]" if o else n for n, o in st.state_labels))
        lines.append(f"  state shocks: {list(st.state_shocks)}   measurement shocks: {list(st.measurement_shocks)}")
        lines.append(f"  SV shocks: {list(st.sv_shocks)}")
        fb = []
        for t in st.feedback_map:
            if t[0] == "const":
                fb.append("1")
            elif t[0] == "obs_lag_mean":
                fb.append("mean(" + ", ".join(f"{t[1]}[-{k}]" for k in t[2]) + ")")
            else:
                fb.append(f"{t[1]}[-{t[2]}]" if t[2] else t[1])
        lines.append(f"  regressor columns (feedback map): {fb}   pre-sample lag rows: {st.lag_depth}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.describe()
