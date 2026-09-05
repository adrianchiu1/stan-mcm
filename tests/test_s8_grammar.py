"""S8 WP1 gates (plans/S8-plan.md): the five grammar extensions, each
against a hand-built oracle -- structure, matrices, regressors, the KF
mirror at prior draws, the DK smoother and the engine. No Stan here; the
compiled programs' ``kf_loglik`` is gated in ``tests/test_s8_stan.py``.

E0 zero stochastic state shocks (and zero states); E1 intercepts
(measurement: a parameter or a number through the Const column;
transition: a drift through the implicit ``_const`` unit state); E2
contemporaneous exogenous regressors; E3 shock-free measurement rows
(singular R, the handbook §3.2 UC trend-cycle model); E5 contemporaneous
observables substituted recursively (a Cholesky-ordered VAR).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macrotoolkit import authoring as au
from macrotoolkit.families.base import Const, ExogLag, ObsLag, StateSpaceMeta
from macrotoolkit.smoother import kalman_loglik, simulate_smoother_draw
from specs.schema.authored import AuthoredOptions

pytestmark = pytest.mark.authored

N_POINTS = 50
SEED = 20260912
REPO_DATA = pd.read_csv(__file__.rsplit("/tests/", 1)[0] + "/examples/handbook/data/ch1_inflation.csv")


# ---------------------------------------------------------------------------
# Oracle models
# ---------------------------------------------------------------------------


def ar2_constant_state(mean: float = 0.0, sd: float = 1.0) -> au.Model:
    """The handbook's Chapter 1 AR(2) regression with the constant as a
    shock-free STATE pinned by its initial condition (E0)."""
    return au.Model(
        "ch1_ar2_const_state", observables=["infl"],
        measurement=["infl = c + b1*infl[-1] + b2*infl[-2] + e"], transition=["c = c[-1]"],
        parameters={"b1": au.normal(0.0, 1.0), "b2": au.normal(0.0, 1.0), "sigma": au.half_normal(1.0)},
        shocks={"e": au.shock("sigma")}, initial_state={"c": au.init(mean, sd)},
    )


def ar2_constant_param() -> au.Model:
    """The same regression with the constant a PARAMETER (E1, measurement)
    -- no state at all (E0's n = 0 limit)."""
    return au.Model(
        "ch1_ar2_const_param", observables=["infl"],
        measurement=["infl = c + b1*infl[-1] + b2*infl[-2] + e"],
        parameters={"c": au.normal(0.0, 1.0), "b1": au.normal(0.0, 1.0), "b2": au.normal(0.0, 1.0), "sigma": au.half_normal(1.0)},
        shocks={"e": au.shock("sigma")},
    )


def drift_ar1_state() -> au.Model:
    """An AR(1) state with a drift parameter (E1, transition)."""
    return au.Model(
        "drift_ar1", observables=["y"], measurement=["y = tau + e"], transition=["tau = mu + rho*tau[-1] + eta"],
        parameters={"mu": au.normal(0.0, 1.0), "rho": au.normal(0.5, 0.3), "s_e": au.half_normal(1.0), "s_eta": au.half_normal(1.0)},
        shocks={"e": au.shock("s_e"), "eta": au.shock("s_eta")}, initial_state={"tau": au.init(0.0, 2.0)},
    )


def uc_trend_cycle() -> au.Model:
    """Handbook §3.2 (2.6)-(2.7): Y = C + tau EXACTLY (no measurement
    shock, E3), C an AR(2) cycle with a constant (E1 drift), tau a random
    walk; orthogonal shocks (the handbook allows a Q off-diagonal)."""
    return au.Model(
        "ch3_uc_trend_cycle", observables=["Y"], measurement=["Y = C + tau"],
        transition=["C = c0 + a1*C[-1] + a2*C[-2] + e1", "tau = tau[-1] + e2"],
        parameters={"c0": au.normal(0.0, 0.5), "a1": au.normal(1.0, 0.3), "a2": au.normal(-0.3, 0.3), "s1": au.half_normal(1.0), "s2": au.half_normal(0.5)},
        shocks={"e1": au.shock("s1"), "e2": au.shock("s2")},
        initial_state={"C": au.init(0.0, 2.0), "tau": au.init(au.first_obs("Y"), 5.0)},
    )


def bivar_recursive(p: int = 1) -> au.Model:
    """A bivariate VAR(p) in recursive (Cholesky-ordered) form (E5)."""
    lags = lambda y: " + ".join(f"b_{y}_{v}_{k}*{v}[-{k}]" for k in range(1, p + 1) for v in ("y", "pi"))
    params = {"c_y": au.normal(0, 10), "c_pi": au.normal(0, 10), "a0": au.normal(0, 10), "s_y": au.half_normal(5), "s_pi": au.half_normal(5)}
    for y in ("y", "pi"):
        for k in range(1, p + 1):
            for v in ("y", "pi"):
                params[f"b_{y}_{v}_{k}"] = au.normal(0, 10)
    return au.Model(
        f"bivar_var{p}", observables=["y", "pi"],
        measurement=[f"y = c_y + {lags('y')} + e_y", f"pi = a0*y + c_pi + {lags('pi')} + e_pi"],
        parameters=params, shocks={"e_y": au.shock("s_y"), "e_pi": au.shock("s_pi")},
    )


def _draws(model: au.Model, n: int, seed: int):
    c = model.compiled()
    priors = c.resolve_priors({})
    rng = np.random.default_rng(seed)
    return c, [c.sample_prior_params(priors, rng, {}) for _ in range(n)]


# ---------------------------------------------------------------------------
# E0 -- zero stochastic state shocks: the KF integrates the constant exactly
# ---------------------------------------------------------------------------


def _closed_form_marginal(y, y1, y2, b1, b2, sigma, mean, sd) -> float:
    """``r = y - b1 y_{-1} - b2 y_{-2} ~ N(mean 1, sigma^2 I + sd^2 11')``
    by the matrix determinant lemma / Sherman-Morrison."""
    r = y - b1 * y1 - b2 * y2 - mean
    T = r.shape[0]
    v, w = sigma**2, sd**2
    logdet = T * np.log(v) + np.log(1.0 + T * w / v)
    quad = (r @ r) / v - (w / (v * (v + T * w))) * (r.sum() ** 2)
    return float(-0.5 * (T * np.log(2 * np.pi) + logdet + quad))


def test_e0_constant_state_regression_equals_closed_form_marginal_likelihood() -> None:
    mean, sd = 0.3, 0.8
    m = ar2_constant_state(mean, sd)
    c, pts = _draws(m, N_POINTS, SEED)
    assert c.meta.state_shocks == () and c.meta.loading_matrix().shape == (1, 0) and c.meta.state_labels == (("c", 0),)
    series = {"infl": REPO_DATA["infl" + "ation"].to_numpy(dtype=float)}
    yobs, x = c.regressors(series)
    xi00, P00 = c.initial_state(series)
    np.testing.assert_array_equal(xi00, [mean]) and np.testing.assert_array_equal(P00, [[sd * sd]])
    y = series["infl"]
    worst = 0.0
    for p in pts:
        F, Q, A, Z, R = c.build_matrices(p)
        np.testing.assert_array_equal(Q, [[0.0]])
        ll = kalman_loglik(yobs, x, F, Q, A, Z, R, xi00, P00)
        ref = _closed_form_marginal(y[2:], y[1:-1], y[:-2], p["b1"], p["b2"], p["sigma"], mean, sd)
        worst = max(worst, abs(ll - ref) / max(1.0, abs(ref)))
    assert worst < 1e-10, worst
    # Downstream: the DK smoother recovers no state shock; the HD has no state-shock bar.
    from macrotoolkit.results_core import hd_bar_names, observable_bars, state_components

    rng = np.random.default_rng(1)
    stationary = next(p for p in pts if c.is_stationary(p))  # the identity at a STATIONARY point (the S6 lesson)
    F, Q, A, Z, R = c.build_matrices(stationary)
    sim = simulate_smoother_draw(yobs, x, F, Q, A, Z, R, xi00, P00, rng, meta=c.meta)
    assert sim.state_shocks == {} and set(sim.meas_shocks) == {"e"}
    assert np.allclose(sim.xi_draw, sim.xi_draw[0])  # a deterministic state is constant along the draw
    comps = state_components(F, c.meta, sim.xi_draw, sim.state_shocks)
    assert list(comps) == ["init"]
    assert hd_bar_names(c.meta) == ("init", "e")
    from macrotoolkit.authoring.results import init_obs_seeds

    bars = observable_bars(A, Z, c.meta, comps, sim.meas_shocks, x, init_obs_seeds(c, series))
    assert np.max(np.abs(sum(bars.values()) - yobs)) < 1e-9


def test_e0_prior_sampler_designs_and_engine_handle_no_state_shock() -> None:
    from macrotoolkit.authoring.validation import simulate_dataset
    from macrotoolkit.engine import RandomWalkLogVarianceStateNoise

    m = ar2_constant_state()
    c, pts = _draws(m, 3, SEED)
    noise = RandomWalkLogVarianceStateNoise(c.meta, {}, {}, {})
    assert noise.step(np.random.default_rng(0)).shape == (1,) and noise.step(np.random.default_rng(0))[0] == 0.0
    data = simulate_dataset(c, pts[0], np.random.default_rng(2), 30, xi00=np.array([0.5]), P00=np.array([[0.0]]))
    assert data["infl"].shape == (30,) and np.all(np.isfinite(data["infl"]))
    # Zero states (a pure regression): the KF core takes n = 0 and the smoother a (T, 0) path.
    c2, pts2 = _draws(ar2_constant_param(), 2, SEED)
    assert c2.meta.n_state == 0 and c2.meta.feedback_map == (Const(), ObsLag("infl", 1), ObsLag("infl", 2))
    series = {"infl": REPO_DATA["inflation"].to_numpy(dtype=float)}
    yobs, x = c2.regressors(series)
    np.testing.assert_array_equal(x[:, 0], 1.0)
    xi00, P00 = c2.initial_state(series)
    assert xi00.shape == (0,) and P00.shape == (0, 0)
    F, Q, A, Z, R = c2.build_matrices(pts2[0])
    sim = simulate_smoother_draw(yobs, x, F, Q, A, Z, R, xi00, P00, np.random.default_rng(0), meta=c2.meta)
    assert sim.xi_draw.shape == (yobs.shape[0], 0)
    np.testing.assert_allclose(sim.meas_shocks["e"], yobs[:, 0] - x @ A[:, 0], atol=1e-12)


# ---------------------------------------------------------------------------
# E1 -- intercepts
# ---------------------------------------------------------------------------


def test_e1_measurement_intercept_parameter_equals_conditional_regression_likelihood() -> None:
    c, pts = _draws(ar2_constant_param(), N_POINTS, SEED)
    series = {"infl": REPO_DATA["inflation"].to_numpy(dtype=float)}
    yobs, x = c.regressors(series)
    xi00, P00 = c.initial_state(series)
    y = series["infl"]
    for p in pts:
        F, Q, A, Z, R = c.build_matrices(p)
        ll = kalman_loglik(yobs, x, F, Q, A, Z, R, xi00, P00)
        resid = y[2:] - p["c"] - p["b1"] * y[1:-1] - p["b2"] * y[:-2]
        ref = float(np.sum(-0.5 * (np.log(2 * np.pi * p["sigma"] ** 2) + resid**2 / p["sigma"] ** 2)))
        assert abs(ll - ref) < 1e-9 * max(1.0, abs(ref))
    # A NUMERIC intercept is a parameter-free Const entry (lands in transformed data).
    m = au.Model("num_int", observables=["y"], measurement=["y = 2.5 + 0.5*y[-1] + e"], parameters={"s": au.half_normal(1)}, shocks={"e": au.shock("s")})
    assert m.compiled().structure.matrix_is_numeric("A") and m.compiled().build_A({})[0, 0] == 2.5


def test_e1_transition_drift_is_the_unit_state_augmentation() -> None:
    c, pts = _draws(drift_ar1_state(), N_POINTS, SEED)
    assert c.meta.state_labels == (("tau", 0), ("_const", 0)) and c.structure.state_names == ("tau",)
    series = {"y": np.array([0.3, 0.1, -0.2, 0.4, 0.0, 0.2])}
    xi00, P00 = c.initial_state(series)
    np.testing.assert_array_equal(xi00, [0.0, 1.0])
    np.testing.assert_array_equal(P00, [[4.0, 0.0], [0.0, 0.0]])
    yobs, x = c.regressors(series)
    for p in pts:
        F, Q, A, Z, R = c.build_matrices(p)
        Fh = np.array([[p["rho"], p["mu"]], [0.0, 1.0]])
        Qh = np.array([[p["s_eta"] ** 2, 0.0], [0.0, 0.0]])
        np.testing.assert_array_equal(F, Fh)
        np.testing.assert_array_equal(Q, Qh)
        np.testing.assert_array_equal(Z, [[1.0, 0.0]])
        assert kalman_loglik(yobs, x, F, Q, A, Z, R, xi00, P00) == kalman_loglik(yobs, x, Fh, Qh, A, Z, R, xi00, P00)
    # The drift keeps the unit state exactly 1 along a DK draw; is_stationary allows its unit root.
    F, Q, A, Z, R = c.build_matrices(pts[0])
    sim = simulate_smoother_draw(yobs, x, F, Q, A, Z, R, xi00, P00, np.random.default_rng(3), meta=c.meta)
    np.testing.assert_allclose(sim.xi_draw[:, 1], 1.0, atol=1e-12)
    assert c.is_stationary({**pts[0], "rho": 0.5})
    # The two intercept routes: the constant-state route needs no parameter; the parameter route has a prior-table entry.
    assert "c" not in ar2_constant_state().compiled().prior_table and "c" in ar2_constant_param().compiled().prior_table


# ---------------------------------------------------------------------------
# E2 -- contemporaneous exogenous regressors
# ---------------------------------------------------------------------------


def exog_model() -> au.Model:
    return au.Model(
        "exog0", observables=["y"], exogenous=["x"], measurement=["y = c + b0*x + b1*x[-1] + b2*x[-2] + 0.5*y[-1] + e"],
        parameters={"c": au.normal(0, 1), "b0": au.normal(0, 1), "b1": au.normal(0, 1), "b2": au.normal(0, 1), "s": au.half_normal(1)},
        shocks={"e": au.shock("s")}, forecast_rules={"x": au.forecast("last_value")},
    )


def test_e2_contemporaneous_exogenous_regressor_columns_and_engine_timing() -> None:
    from macrotoolkit.authoring.results import _terminal_seeds, data_path_rules, init_obs_seeds, presample_exog_seeds
    from macrotoolkit.engine import ConstantExogRule, simulate_forward

    c = exog_model().compiled()
    assert c.meta.feedback_map == (Const(), ExogLag("x", 0), ExogLag("x", 1), ExogLag("x", 2), ObsLag("y", 1))
    assert c.meta.exog_contemporaneous("x") and c.meta.exog_is_referenced("x") and c.meta.exog_lag_depth("x") == 2 and c.lag_depth == 2
    n = 12
    series = {"y": np.arange(n, dtype=float), "x": 100.0 + np.arange(n)}
    yobs, x = c.regressors(series)
    np.testing.assert_array_equal(x[:, 1], series["x"][2:])  # lag 0: the estimation rows themselves
    np.testing.assert_array_equal(x[:, 2], series["x"][1:-1])
    np.testing.assert_array_equal(x[:, 3], series["x"][:-2])
    assert presample_exog_seeds(c, series) == {"x": {1: 101.0, 2: 100.0}}
    assert _terminal_seeds(c, series)[1] == {"x": {1: 111.0, 2: 110.0}}
    np.testing.assert_array_equal(data_path_rules(c, series, 10)["x"]._values, series["x"][2:])
    # Zero-noise seam: the forward simulation with the data-path rule reproduces A'x row by row.
    p = {"c": 0.2, "b0": 0.3, "b1": -0.1, "b2": 0.05, "s": 1.0}
    F, Q, A, Z, R = c.build_matrices(p)

    class Zero:
        def step(self, rng):
            return np.zeros(1)

    out = simulate_forward(F, Q, A, Z, c.meta, np.zeros(0), init_obs_seeds(c, series), presample_exog_seeds(c, series),
                           data_path_rules(c, series, 10), Zero(), 10, np.random.default_rng(0), state_noise=Zero())
    # Expected: y_t = c + b0 x_t + b1 x_{t-1} + b2 x_{t-2} + 0.5 y_{t-1} with y fed back from the simulated path.
    y_prev, expect = series["y"][1], []
    for t in range(10):
        xt = series["x"]
        val = p["c"] + p["b0"] * xt[2 + t] + p["b1"] * xt[1 + t] + p["b2"] * xt[t] + 0.5 * y_prev
        expect.append(val)
        y_prev = val
    np.testing.assert_allclose(out["obs"][:, 0], expect, atol=1e-12)
    # The last_value rule holds x at its last value for every lag once the register rolls over.
    out2 = simulate_forward(F, Q, A, Z, c.meta, np.zeros(0), {"y": {1: 0.0}}, {"x": {1: 5.0, 2: 4.0}}, {"x": ConstantExogRule(7.0)}, Zero(), 3, np.random.default_rng(0), state_noise=Zero())
    y0 = 0.2 + 0.3 * 7 - 0.1 * 5 + 0.05 * 4
    y1 = 0.2 + 0.3 * 7 - 0.1 * 7 + 0.05 * 5 + 0.5 * y0
    y2 = 0.2 + 0.3 * 7 - 0.1 * 7 + 0.05 * 7 + 0.5 * y1
    np.testing.assert_allclose(out2["obs"][:, 0], [y0, y1, y2], atol=1e-12)


# ---------------------------------------------------------------------------
# E3 -- shock-free measurement rows: the handbook §3.2 UC trend-cycle model
# ---------------------------------------------------------------------------


def test_e3_uc_trend_cycle_matches_hand_built_matrices_and_the_measurement_identity() -> None:
    c, pts = _draws(uc_trend_cycle(), N_POINTS, SEED)
    st = c.structure
    assert st.state_labels == (("C", 0), ("C", -1), ("tau", 0), ("_const", 0))
    assert st.measurement_shocks == () and st.shock_free_rows() == ("Y",) and c.meta.measurement_loadings == {}
    assert c.meta.n_obs == 1 and c.meta.n_meas_shocks == 0
    Y = REPO_DATA["inflation"].to_numpy(dtype=float)[:80]
    series = {"Y": Y}
    yobs, x = c.regressors(series)
    xi00, P00 = c.initial_state(series)
    np.testing.assert_array_equal(xi00, [0.0, 0.0, Y[0], 1.0])
    np.testing.assert_array_equal(np.diag(P00), [4.0, 4.0, 25.0, 0.0])
    for p in pts:
        F, Q, A, Z, R = c.build_matrices(p)
        Fh = np.array([[p["a1"], p["a2"], 0.0, p["c0"]], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        Qh = np.diag([p["s1"] ** 2, 0.0, p["s2"] ** 2, 0.0])
        np.testing.assert_array_equal(F, Fh)
        np.testing.assert_array_equal(Q, Qh)
        np.testing.assert_array_equal(Z, [[1.0, 0.0, 1.0, 0.0]])
        np.testing.assert_array_equal(R, [[0.0]])
        assert A.shape == (0, 1)
        ll = kalman_loglik(yobs, x, F, Q, A, Z, R, xi00, P00)
        assert np.isfinite(ll) and ll == kalman_loglik(yobs, x, Fh, Qh, A, Z, R, xi00, P00)
    # DK draws with R = 0: C + tau reproduces Y exactly (the measurement identity), shocks recovered.
    F, Q, A, Z, R = c.build_matrices(pts[0])
    rng = np.random.default_rng(5)
    for _ in range(3):
        sim = simulate_smoother_draw(yobs, x, F, Q, A, Z, R, xi00, P00, rng, meta=c.meta)
        np.testing.assert_allclose(sim.xi_draw @ Z.T, yobs, atol=1e-7 * np.max(np.abs(yobs)))
        assert set(sim.state_shocks) == {"e1", "e2"} and sim.meas_shocks == {}
    from macrotoolkit.authoring.results import hd_reconstruction_error

    from macrotoolkit import api as mtk

    spec = mtk.spec("authored", options=uc_trend_cycle(), data={"file": "x.csv", "date_column": "date", "mapping": {"Y": "inflation"}})
    df = REPO_DATA.iloc[:80].rename(columns={"inflation": "Y"})
    df["date"] = pd.to_datetime(df["date"])
    assert hd_reconstruction_error(spec, df, 0) < 1e-6


def test_e3_rank_rule_rejects_an_unexplained_shock_free_row_and_accepts_an_explained_one() -> None:
    base = uc_trend_cycle().to_dict()
    # Two shock-free rows loading the SAME stochastic combination: singular S.
    bad = dict(base)
    bad["observables"] = ["Y", "W"]
    bad["equations"] = {"measurement": ["Y = C + tau", "W = 2*C + 2*tau"], "transition": base["equations"]["transition"]}
    with pytest.raises(ValueError, match="singular"):
        AuthoredOptions.model_validate(bad)
    # Two shock-free rows loading DIFFERENT shocks: fine (rank 2).
    ok = dict(bad)
    ok["equations"] = {"measurement": ["Y = C + tau", "W = tau"], "transition": base["equations"]["transition"]}
    st = AuthoredOptions.model_validate(ok).structure
    assert st.shock_free_rows() == ("Y", "W")


# ---------------------------------------------------------------------------
# E5 -- contemporaneous observables: a recursive VAR
# ---------------------------------------------------------------------------


def test_e5_substitution_reproduces_the_hand_substituted_rows_and_the_cholesky_impact() -> None:
    from macrotoolkit.results_core import impulse_response

    c, pts = _draws(bivar_recursive(1), N_POINTS, SEED)
    st = c.structure
    assert st.obs_substitution_order == ("y", "pi") and st.meas_loadings == {"e_y": ("y", "pi"), "e_pi": ("pi",)}
    assert c.meta.measurement_loadings == {"e_y": ("y", "pi"), "e_pi": ("pi",)}
    assert c.meta.meas_recovery_order() == (("e_y", 0, ()), ("e_pi", 1, ("e_y",)))
    assert c.meta.feedback_map == (Const(), ObsLag("y", 1), ObsLag("pi", 1)) and c.meta.n_state == 0
    for p in pts:
        F, Q, A, Z, R = c.build_matrices(p)
        M = c.build_M(p)
        a0 = p["a0"]
        # Hand substitution: pi = (c_pi + a0 c_y) + (b_pi_y_1 + a0 b_y_y_1) y[-1] + (b_pi_pi_1 + a0 b_y_pi_1) pi[-1] + a0 e_y + e_pi
        Ah = np.array([[p["c_y"], p["c_pi"] + a0 * p["c_y"]], [p["b_y_y_1"], p["b_pi_y_1"] + a0 * p["b_y_y_1"]], [p["b_y_pi_1"], p["b_pi_pi_1"] + a0 * p["b_y_pi_1"]]])
        np.testing.assert_array_equal(A, Ah)
        np.testing.assert_array_equal(M, [[1.0, 0.0], [a0, 1.0]])
        D = np.diag([p["s_y"] ** 2, p["s_pi"] ** 2])
        np.testing.assert_allclose(R, M @ D @ M.T, rtol=1e-14)
        # The structural impact matrix (one-sd shocks through their loadings) IS the Cholesky factor of R.
        impact = np.column_stack([impulse_response(F, A, Z, c.meta, s, np.sqrt(D[j, j]), 1, M=M)[1][0] for j, s in enumerate(c.meta.measurement_shocks)])
        np.testing.assert_allclose(impact, np.linalg.cholesky(R), rtol=1e-8)  # flat-prior draws are badly scaled (|a0| ~ 10)
    # A shock in a substituted equation is recovered exactly through the triangular loading.
    data = pd.read_csv(__file__.rsplit("/tests/", 1)[0] + "/examples/handbook/data/ch2_datain.csv")
    series = {"y": data["gdp_growth"].to_numpy(dtype=float), "pi": data["inflation"].to_numpy(dtype=float)}
    yobs, x = c.regressors(series)
    xi00, P00 = c.initial_state(series)
    point = {"c_y": 1.0, "c_pi": 0.5, "a0": 0.3, "s_y": 3.0, "s_pi": 1.5, "b_y_y_1": 0.3, "b_y_pi_1": -0.1, "b_pi_y_1": 0.1, "b_pi_pi_1": 0.7}  # stationary (the HD identity needs it)
    assert c.is_stationary(point)
    F, Q, A, Z, R = c.build_matrices(point)
    M = c.build_M(point)
    sim = simulate_smoother_draw(yobs, x, F, Q, A, Z, R, xi00, P00, np.random.default_rng(0), meta=c.meta, M=M)
    resid = yobs - x @ A
    np.testing.assert_allclose(sim.meas_shocks["e_y"], resid[:, 0], atol=1e-12)
    np.testing.assert_allclose(sim.meas_shocks["e_pi"], resid[:, 1] - point["a0"] * resid[:, 0], atol=1e-12)
    # HD bars (with the const bar) reconstruct both observables; the const bar carries the intercepts.
    from macrotoolkit.authoring.results import init_obs_seeds
    from macrotoolkit.results_core import hd_bar_names, observable_bars, state_components

    comps = state_components(F, c.meta, sim.xi_draw, sim.state_shocks)
    bars = observable_bars(A, Z, c.meta, comps, sim.meas_shocks, x, init_obs_seeds(c, series), M=M)
    assert hd_bar_names(c.meta) == ("init", "e_y", "e_pi", "const")
    assert np.max(np.abs(sum(bars.values()) - yobs)) < 1e-9
    assert np.all(bars["e_y"][:, 1] != 0.0)  # e_y reaches pi contemporaneously through a0


def test_e5_engine_forward_simulation_applies_the_measurement_loading() -> None:
    from macrotoolkit.engine import simulate_forward

    c = bivar_recursive(1).compiled()
    p = {"c_y": 0.1, "c_pi": 0.2, "a0": 0.7, "s_y": 1.0, "s_pi": 1.0, "b_y_y_1": 0.3, "b_y_pi_1": 0.0, "b_pi_y_1": 0.0, "b_pi_pi_1": 0.4}
    F, Q, A, Z, R = c.build_matrices(p)
    M = c.build_M(p)

    class Unit:
        def __init__(self):
            self.t = 0

        def step(self, rng):
            self.t += 1
            return np.array([1.0, 0.0]) if self.t == 1 else np.zeros(2)

    class ZeroState:
        def step(self, rng):
            return np.zeros(0)

    out = simulate_forward(F, Q, A, Z, c.meta, np.zeros(0), {"y": {1: 0.0}, "pi": {1: 0.0}}, {}, {}, Unit(), 2, np.random.default_rng(0),
                           state_noise=ZeroState(), meas_loading=M)
    np.testing.assert_allclose(out["obs"][0], [0.1 + 1.0, 0.2 + 0.7 * 0.1 + 0.7 * 1.0], atol=1e-12)


def test_e5_ordering_is_meaning_and_a_var_with_states_composes_z() -> None:
    """The Cholesky ordering is the equation order; a substituted equation
    also carries its states into the referencing row (Z composes)."""
    m = au.Model(
        "var_with_trend", observables=["y", "pi"], measurement=["y = tau + e_y", "pi = a0*y + 0.5*pi[-1] + e_pi"],
        transition=["tau = tau[-1] + eta"],
        parameters={"a0": au.normal(0, 1), "s_y": au.half_normal(1), "s_pi": au.half_normal(1), "s_eta": au.half_normal(1)},
        shocks={"e_y": au.shock("s_y"), "e_pi": au.shock("s_pi"), "eta": au.shock("s_eta")}, initial_state={"tau": au.init(0.0, 1.0)},
    ).compiled()
    Z = m.build_Z({"a0": 0.6})
    np.testing.assert_array_equal(Z, [[1.0], [0.6]])
    reordered = au.Model(
        "var_with_trend2", observables=["pi", "y"], measurement=["pi = 0.5*pi[-1] + e_pi", "y = a0*pi + tau + e_y"],
        transition=["tau = tau[-1] + eta"],
        parameters={"a0": au.normal(0, 1), "s_y": au.half_normal(1), "s_pi": au.half_normal(1), "s_eta": au.half_normal(1)},
        shocks={"e_y": au.shock("s_y"), "e_pi": au.shock("s_pi"), "eta": au.shock("s_eta")}, initial_state={"tau": au.init(0.0, 1.0)},
    ).compiled()
    assert reordered.meta.measurement_loadings == {"e_pi": ("pi", "y"), "e_y": ("y",)}


# ---------------------------------------------------------------------------
# The declaration layer: measurement loadings on StateSpaceMeta
# ---------------------------------------------------------------------------


def test_state_space_meta_measurement_loadings_validation_and_pre_s8_defaults() -> None:
    from macrotoolkit.families.lw_sv import LW_STATE_META
    from macrotoolkit.families.ucsv import UCSV_STATE_META

    for meta in (LW_STATE_META, UCSV_STATE_META):
        assert meta.measurement_loadings is None
        assert meta.meas_recovery_order() == tuple((s, j, ()) for j, s in enumerate(meta.measurement_shocks))
        assert [meta.meas_shock_row(s) for s in meta.measurement_shocks] == list(range(meta.n_obs))
    meta = StateSpaceMeta(state_labels=(), state_shocks=(), shock_loadings={}, measurement_shocks=("a", "b"), obs_names=("y1", "y2", "y3"),
                          measurement_loadings={"a": ("y1", "y2"), "b": ("y2",)})
    assert meta.n_obs == 3 and meta.n_meas_shocks == 2 and meta.meas_shock_row("b") == 1
    assert meta.meas_recovery_order() == (("a", 0, ()), ("b", 1, ("a",)))
    with pytest.raises(ValueError, match="share an own row"):
        StateSpaceMeta(state_labels=(), state_shocks=(), shock_loadings={}, measurement_shocks=("a", "b"), obs_names=("y1", "y2"),
                       measurement_loadings={"a": ("y1",), "b": ("y1", "y2")})
    with pytest.raises(ValueError, match="exactly the measurement_shocks"):
        StateSpaceMeta(state_labels=(), state_shocks=(), shock_loadings={}, measurement_shocks=("a",), obs_names=("y1",), measurement_loadings={"z": ("y1",)})
    with pytest.raises(ValueError, match="strictly lagged"):
        StateSpaceMeta(state_labels=(), state_shocks=(), shock_loadings={}, measurement_shocks=("a",), obs_names=("y1",), feedback_map=(ObsLag("y1", 0),))
    StateSpaceMeta(state_labels=(), state_shocks=(), shock_loadings={}, measurement_shocks=("a",), obs_names=("y1",), exog_names=("x",), feedback_map=(ExogLag("x", 0), Const()))


def test_s8_constructs_change_the_hash_and_round_trip_through_yaml() -> None:
    import yaml

    for model in (ar2_constant_state(), ar2_constant_param(), drift_ar1_state(), uc_trend_cycle(), bivar_recursive(2), exog_model()):
        d = model.to_dict()
        again = AuthoredOptions.model_validate(yaml.safe_load(yaml.safe_dump(d)))
        assert again.model_dump() == model.options.model_dump()
        assert au.compile_model(again) is model.compiled()
