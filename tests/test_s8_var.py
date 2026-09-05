"""S8 WP2: the VAR support layer -- ``au.var`` expands to the recursive
equations; ``au.minnesota_priors`` reproduces the handbook's Chapter 2
example 1 arithmetic exactly; the steady-state (Villani) form is a
worked oracle (its KF likelihood equals the closed-form marginal with
the vector of long-run means integrated out); the generic FEVD sums to
one and reproduces a hand computation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macrotoolkit import authoring as au
from macrotoolkit.smoother import kalman_loglik

pytestmark = pytest.mark.authored

REPO = Path(__file__).resolve().parents[1]


def _handbook_df() -> pd.DataFrame:
    return pd.read_csv(REPO / "examples/handbook/data/ch2_datain.csv").rename(columns={"gdp_growth": "y", "inflation": "pi"})


def test_var_expands_to_the_recursive_equations_and_declares_every_parameter() -> None:
    m = au.var("v", ["y", "pi", "r"], 2)
    eqs = m.options.equations.measurement
    assert eqs[0].startswith("y = c_y + b_y_y_1*y[-1]") and " + e_y" in eqs[0]
    assert "a0_pi_y*y" in eqs[1] and "a0_r_y*y + a0_r_pi*pi" in eqs[2]
    st = m.structure
    assert st.n_state == 0 and st.measurement_shocks == ("e_y", "e_pi", "e_r")
    assert st.meas_loadings == {"e_y": ("y", "pi", "r"), "e_pi": ("pi", "r"), "e_r": ("r",)}
    names = au.var_parts(["y", "pi", "r"], 2)["parameters"]
    assert len(names) == 3 + 3 + 3 * 2 * 3 + 3  # constants, a0, lags, scales
    with pytest.raises(ValueError, match="does not declare"):
        au.var("bad", ["y"], 1, priors={"nope": au.normal(0, 1)})
    # No intercept; an exogenous block through extra_terms.
    m2 = au.var("v2", ["y", "pi"], 1, intercept=False, exogenous=["x"], extra_terms={"y": "+ g*x"}, extra_parameters={"g": au.normal(0, 1)},
                forecast_rules={"x": au.forecast("last_value")})
    assert "c_y" not in m2.compiled().prior_table and m2.structure.feedback_map[0] == ("obs_lag", "y", 1)
    assert ("exog_lag", "x", 0) in m2.structure.feedback_map


def test_minnesota_priors_reproduce_the_handbook_example1_H_matrix() -> None:
    """Chapter 2 example 1: lambda1..4 = 1, B0 own first lag 1, H diagonal
    from s1/s2 (AR(1) residual sds over rows 3:end)."""
    df = _handbook_df()
    Y = df[["y", "pi"]].to_numpy(dtype=float)
    # The handbook's s1, s2 (X = [1, lag1] over rows 3:end, i.e. the estimation rows of a VAR(2)).
    def s_of(col):
        y = Y[2:, col]
        x = np.column_stack([np.ones(y.shape[0] - 1), y[:-1]])
        b = np.linalg.solve(x.T @ x, x.T @ y[1:])
        r = y[1:] - x @ b
        return np.sqrt(r @ r / (y.shape[0] - 1 - 2))
    s1, s2 = s_of(0), s_of(1)
    pr = au.minnesota_priors(df, ["y", "pi"], 2)
    assert pr["c_y"] == au.normal(0.0, s1 * 1.0) and pr["c_pi"] == au.normal(0.0, s2 * 1.0)
    assert pr["b_y_y_1"] == au.normal(1.0, 1.0) and pr["b_y_pi_1"]["sd"] == pytest.approx(s1 / s2)
    assert pr["b_y_y_2"] == au.normal(0.0, 0.5) and pr["b_y_pi_2"]["sd"] == pytest.approx(s1 / (s2 * 2.0))
    assert pr["b_pi_y_1"]["sd"] == pytest.approx(s2 / s1) and pr["b_pi_pi_2"] == au.normal(0.0, 0.5)
    # Hyperparameters scale as documented; the priors feed var() by name.
    tight = au.minnesota_priors(df, ["y", "pi"], 2, lambda1=0.1, lambda3=0.05, lambda4=1.0, own_mean=0.95)
    assert tight["b_y_y_1"] == au.normal(0.95, 0.1) and tight["b_y_y_2"]["sd"] == pytest.approx(0.1 / 2**0.05)
    m = au.var("mn", ["y", "pi"], 2, priors=tight)
    assert m.compiled().prior_table["b_y_y_1"] == {"dist": "normal", "mu": 0.95, "sd": 0.1}


def _closed_form_villani(Y: np.ndarray, B: np.ndarray, Sigma: np.ndarray, m0: np.ndarray, V0: np.ndarray) -> float:
    """``r_t = y_t - B y_{t-1} = (I - B) mu + u_t``, ``mu ~ N(m0, V0)``,
    ``u ~ N(0, Sigma)``: the stacked ``r`` is Gaussian with mean
    ``1 (x) (I-B) m0`` and covariance ``I (x) Sigma + 11' (x) (I-B) V0 (I-B)'``."""
    N = Y.shape[1]
    r = (Y[1:] - Y[:-1] @ B.T).reshape(-1)
    T = Y.shape[0] - 1
    G = np.eye(N) - B
    mean = np.tile(G @ m0, T)
    cov = np.kron(np.eye(T), Sigma) + np.kron(np.ones((T, T)), G @ V0 @ G.T)
    L = np.linalg.cholesky(cov)
    u = np.linalg.solve(L, r - mean)
    return float(-0.5 * (r.shape[0] * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(L))) + u @ u))


def test_steady_state_form_is_the_closed_form_marginal_over_the_long_run_means() -> None:
    df = _handbook_df().iloc[:61]
    m = au.var("ss", ["y", "pi"], 1, intercept="steady_state", steady_state_init={"y": au.init(1.0, 0.5), "pi": au.init(1.0, 0.5)})
    c = m.compiled()
    assert c.meta.state_labels == (("mu_y", 0), ("mu_pi", 0)) and c.meta.state_shocks == ()
    series = {"y": df["y"].to_numpy(dtype=float), "pi": df["pi"].to_numpy(dtype=float)}
    yobs, x = c.regressors(series)
    xi00, P00 = c.initial_state(series)
    Y = np.column_stack([series["y"], series["pi"]])
    rng = np.random.default_rng(20260915)
    priors = c.resolve_priors({})
    worst = 0.0
    for _ in range(30):
        p = c.sample_prior_params(priors, rng, {})
        p = {**p, "b_y_y_1": rng.uniform(-0.6, 0.6), "b_y_pi_1": rng.uniform(-0.3, 0.3), "b_pi_y_1": rng.uniform(-0.3, 0.3), "b_pi_pi_1": rng.uniform(-0.6, 0.6),
             "a0_pi_y": rng.uniform(-1, 1), "sigma_y": rng.uniform(0.5, 3), "sigma_pi": rng.uniform(0.5, 3)}
        F, Q, A, Z, R = c.build_matrices(p)
        ll = kalman_loglik(yobs, x, F, Q, A, Z, R, xi00, P00)
        a0 = p["a0_pi_y"]
        B = np.array([[p["b_y_y_1"], p["b_y_pi_1"]], [p["b_pi_y_1"] + a0 * p["b_y_y_1"], p["b_pi_pi_1"] + a0 * p["b_y_pi_1"]]])
        M = np.array([[1.0, 0.0], [a0, 1.0]])
        Sigma = M @ np.diag([p["sigma_y"] ** 2, p["sigma_pi"] ** 2]) @ M.T
        ref = _closed_form_villani(Y, B, Sigma, np.array([1.0, 1.0]), np.diag([0.25, 0.25]))
        worst = max(worst, abs(ll - ref) / abs(ref))
    assert worst < 1e-9, worst


def test_fevd_sums_to_one_and_matches_a_hand_computation() -> None:
    from macrotoolkit.results_core import fevd

    r1 = np.array([1.0, 0.5, 0.25])
    r2 = np.array([0.0, 1.0, 0.0])
    out = fevd({"a": r1, "b": r2})
    np.testing.assert_allclose(out["a"] + out["b"], 1.0)
    np.testing.assert_allclose(out["a"], [1.0, 1.25 / 2.25, 1.3125 / 2.3125])
    zero = fevd({"a": np.zeros(2), "b": np.zeros(2)})
    assert np.all(np.isnan(zero["a"]))
