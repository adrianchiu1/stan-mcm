"""The S9 oracle models (plans/S9-plan.md): authored definitions shared by
``tests/test_s9_grammar.py``, ``tests/test_s9_stan.py`` and the handbook
example builder. A support module, not a test file."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macrotoolkit import authoring as au

REPO = __file__.rsplit("/tests/", 1)[0]


def tvp_regression(*, x_lag: int = 0, sigma_eta_sd: float = 0.1) -> au.Model:
    """Handbook Chapter 3 example 1/2: ``Y = beta_t X + e``, ``beta`` a
    random walk (E4 with an exogenous regressor at lag ``x_lag``; lag 0 is
    E2 + E4)."""
    xref = "X" if x_lag == 0 else f"X[-{x_lag}]"
    return au.Model(
        "tvp_regression", observables=["Y"], exogenous=["X"],
        measurement=[f"Y = beta*{xref} + e"], transition=["beta = beta[-1] + eta"],
        parameters={"s_e": au.half_normal(0.5), "s_eta": au.half_normal(sigma_eta_sd)},
        shocks={"e": au.shock("s_e"), "eta": au.shock("s_eta")},
        initial_state={"beta": au.init(0.0, 1.0)},
        forecast_rules={"X": au.forecast("last_value")},
    )


def constant_regression() -> au.Model:
    """The same regression with ``beta`` a PARAMETER (E2, constant Z, n = 0)."""
    return au.Model(
        "constant_regression", observables=["Y"], exogenous=["X"],
        measurement=["Y = beta*X + e"],
        parameters={"beta": au.normal(0.0, 1.0), "s_e": au.half_normal(0.5)},
        shocks={"e": au.shock("s_e")},
    )


def tvp_ar1(*, sv: bool = True) -> au.Model:
    """Handbook Chapter 5 example 5: ``pi = c_t + b_t pi[-1] + e``, both
    coefficients random walks, ``e`` with SV (the TVP-AR(1)-SV)."""
    return au.Model(
        "tvp_ar1_sv" if sv else "tvp_ar1", observables=["pi"],
        measurement=["pi = c + b*pi[-1] + e"], transition=["c = c[-1] + eta_c", "b = b[-1] + eta_b"],
        parameters={"s_c": au.half_normal(0.1), "s_b": au.half_normal(0.05), **({} if sv else {"s_e": au.half_normal(1.0)})},
        shocks={"eta_c": au.shock("s_c"), "eta_b": au.shock("s_b"), "e": au.sv(sigma_h=0.2, mu_h0=au.log_var_diff("pi", 0.5)) if sv else au.shock("s_e")},
        initial_state={"c": au.init(0.0, 1.0), "b": au.init(0.5, 0.5)},
    )


def tvp_var(observables=("y", "pi"), p: int = 1, *, name: str = "tvp_var") -> au.Model:
    """A recursive (Cholesky-ordered, E5) VAR(p) with a constant whose
    coefficients are ALL random-walk states (E4), constant Sigma through
    the ``a0`` contemporaneous coefficients and constant shock scales --
    Chapter 3 example 3's TVP-VAR in the grammar (one shared random-walk
    scale ``s_q``)."""
    obs = list(observables)
    measurement, transition, params, shocks, init = [], [], {"s_q": au.half_normal(0.05)}, {}, {}
    for i, yi in enumerate(obs):
        terms = [f"c_{yi}"]
        states = [f"c_{yi}"]
        for j, yj in enumerate(obs[:i]):
            terms.append(f"a0_{yi}_{yj}*{yj}")
            params[f"a0_{yi}_{yj}"] = au.normal(0.0, 1.0)
        for k in range(1, p + 1):
            for yj in obs:
                terms.append(f"b_{yi}_{yj}_{k}*{yj}[-{k}]")
                states.append(f"b_{yi}_{yj}_{k}")
        measurement.append(f"{yi} = " + " + ".join(terms) + f" + e_{yi}")
        params[f"s_{yi}"] = au.half_normal(1.0)
        shocks[f"e_{yi}"] = au.shock(f"s_{yi}")
        for s in states:
            transition.append(f"{s} = {s}[-1] + eta_{s}")
            shocks[f"eta_{s}"] = au.shock("s_q")
            init[s] = au.init(0.0, 0.5)
    return au.Model(name, observables=obs, measurement=measurement, transition=transition, parameters=params, shocks=shocks, initial_state=init)


def example1_data() -> pd.DataFrame:
    """The committed example 1/2 DGP simulation (``make_data.py``)."""
    return pd.read_csv(f"{REPO}/examples/handbook/data/ch3_tvp_example1_sim.csv", parse_dates=["date"])


def uk_inflation() -> pd.DataFrame:
    return pd.read_csv(f"{REPO}/examples/handbook/data/ch5_uk_inflation.csv", parse_dates=["date"])


def handbook_kalman_filter(Y: np.ndarray, X: np.ndarray, beta0: float, p00: float, F: float, Q: float, R: float, mu: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A transcription of ``Code2017/CHAPTER3/example1.m``'s Kalman filter
    loop (scalar state, ``inv(feta)`` as written): returns the filtered
    ``beta_{t|t}`` and its variance ``p_{t|t}`` -- the independent oracle of
    the filter with a data-dependent loading."""
    t = len(Y)
    beta_tt = np.empty(t)
    ptt = np.empty(t)
    beta11, p11 = beta0, p00
    for i in range(t):
        x = X[i]
        beta10 = mu + beta11 * F
        p10 = F * p11 * F + Q
        yhat = x * beta10
        eta = Y[i] - yhat
        feta = x * p10 * x + R
        K = (p10 * x) * (1.0 / feta)
        beta11 = beta10 + K * eta
        p11 = p10 - K * (x * p10)
        beta_tt[i] = beta11
        ptt[i] = p11
    return beta_tt, ptt
