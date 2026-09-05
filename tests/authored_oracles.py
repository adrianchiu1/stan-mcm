"""The three hand-written families EXPRESSED AS EQUATIONS (S7 acceptance
support module; not a test file): the authored definitions the M1/M2
equivalence gates compare against ``local_level``, ``ucsv`` and ``lw_sv``.

Shock-name correspondence (plans/S7-plan.md conflict item 2): lw_sv names
its state shocks after the states they drive (``"ystar"``, ``"g"``,
``"z"``), which the DSL forbids (a name is a series OR a shock), so the
authored lw_sv uses ``eta_*`` / ``e_*`` labels and the gates map them
through ``LW_SHOCK_MAP``. ucsv's and local_level's labels coincide with
the hand ones.
"""
from __future__ import annotations

from macrotoolkit import authoring as au

# --- lw_sv (lw-sv-spec.md §1.2-1.4; HLW's own state layout) ---------------

LW_SHOCK_MAP = {"eta_ystar": "ystar", "eta_g": "g", "eta_z": "z", "e_is": "is", "e_pc": "pc"}


def lw_sv_model(sv: bool) -> au.Model:
    from specs.schema.lw_sv import DEFAULT_PRIORS as P

    params = {
        "a1": au.normal(P["a1"]["mu"], P["a1"]["sd"]),
        "a2": au.normal(P["a2"]["mu"], P["a2"]["sd"]),
        "a_r": au.normal(P["a_r"]["mu"], P["a_r"]["sd"], upper=0.0),
        "b_pi": au.beta(P["b_pi"]["a"], P["b_pi"]["b"]),
        "b_y": au.normal(P["b_y"]["mu"], P["b_y"]["sd"], lower=0.0),
        "sigma_ystar": au.half_normal(P["sigma_ystar"]["sd"]),
        "sigma_g": au.half_normal(P["sigma_g"]["sd"]),
        "sigma_z": au.half_normal(P["sigma_z"]["sd"]),
    }
    if sv:
        shocks_meas = {
            "e_is": au.sv(sigma_h=P["sigma_h_is"]["sd"], h0_sd=P["mu_h0_is"]["sd"], mu_h0=0.0),
            "e_pc": au.sv(sigma_h=P["sigma_h_pc"]["sd"], h0_sd=P["mu_h0_pc"]["sd"], mu_h0=0.0),
        }
    else:
        params["sigma_is"] = au.half_normal(P["sigma_is"]["sd"])
        params["sigma_pc"] = au.half_normal(P["sigma_pc"]["sd"])
        shocks_meas = {"e_is": au.shock("sigma_is"), "e_pc": au.shock("sigma_pc")}
    return au.Model(
        "lw_sv_authored",
        observables=["y", "pi"],
        exogenous=["r"],
        measurement=[
            "y = ystar + a1*(y[-1] - ystar[-1]) + a2*(y[-2] - ystar[-2])"
            " + (a_r/2)*(r[-1] - g[-1] - z[-1]) + (a_r/2)*(r[-2] - g[-2] - z[-2]) + e_is",
            "pi = b_pi*pi[-1] + (1 - b_pi)*mean(pi[-2], pi[-3], pi[-4]) + b_y*(y[-1] - ystar[-1]) + e_pc",
        ],
        transition=[
            "ystar = ystar[-1] + 0.25*g[-1] + eta_ystar",  # g annualized: quarterly increment g/4
            "g[-1] = g[-2] + eta_g",  # carried lagged (HLW's timing)
            "z[-1] = z[-2] + eta_z",
        ],
        parameters=params,
        shocks={"eta_ystar": au.shock("sigma_ystar"), "eta_g": au.shock("sigma_g"), "eta_z": au.shock("sigma_z"), **shocks_meas},
        initial_state={
            "ystar": au.init(au.first_obs("y"), 2.0),
            "g": au.init(3.0, 1.0),
            "z": au.init(0.0, 1.0),
        },
        forecast_rules={"r": au.forecast("state_linear", terms={"g[-1]": 1.0, "z[-1]": 1.0})},
    )


# --- ucsv (specs/schema/ucsv.py) ------------------------------------------


def ucsv_model(sv: bool) -> au.Model:
    from specs.schema.ucsv import DEFAULT_PRIORS as P, INITIAL_STATE_PRIOR

    if sv:
        params: dict = {}
        shocks = {
            "eps": au.sv(sigma_h=P["sigma_h_eps"]["sd"], h0_sd=P["mu_h0_eps"]["sd"], mu_h0=au.log_var_diff("pi", 0.25)),
            "eta": au.sv(sigma_h=P["sigma_h_eta"]["sd"], h0_sd=P["mu_h0_eta"]["sd"], mu_h0=au.log_var_diff("pi", 0.5)),
        }
    else:
        params = {"sigma_eps": au.half_normal(P["sigma_eps"]["sd"]), "sigma_eta": au.half_normal(P["sigma_eta"]["sd"])}
        shocks = {"eps": au.shock("sigma_eps"), "eta": au.shock("sigma_eta")}
    return au.Model(
        "ucsv_authored",
        observables=["pi"],
        measurement=["pi = tau + eps"],
        transition=["tau = tau[-1] + eta"],
        parameters=params,
        shocks=shocks,
        initial_state={"tau": au.init(au.first_obs("pi"), INITIAL_STATE_PRIOR["tau0_sd"])},
    )


# --- local_level (stan/templates/local_level.stan.j2) ----------------------


def local_level_model() -> au.Model:
    return au.Model(
        "local_level_authored",
        observables=["y"],
        measurement=["y = mu + eps"],
        transition=["mu = mu[-1] + eta"],
        parameters={"sigma_obs": au.half_normal(1.0), "sigma_level": au.half_normal(1.0)},
        shocks={"eps": au.shock("sigma_obs"), "eta": au.shock("sigma_level")},
        initial_state={"mu": au.init(au.first_obs("y"), 10.0)},
    )
