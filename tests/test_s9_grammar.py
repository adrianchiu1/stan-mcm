"""S9 gates (plans/S9-plan.md), Python side: the E4 product grammar
(what is accepted, what keeps its S7 rejection), the compiled structure
(``Zx``, the feedback column, E5 composition, coefficient-only shocks),
the numeric ``Z_t`` path against a hand construction, the Stan text the
emitter prints, the KF/DK smoother with a ``Z_t`` path, the engine's
bilinear closure (HD identity, the zero-noise seam, dated IRFs), the
oracles (a) TVP regression with Q = 0 vs the constant-coefficient
regression and (b, i) the handbook's own Chapter 3 example 1 filter, and
the hash round trip. The compiled programs' ``kf_loglik`` and the fits
are gated in ``tests/test_s9_stan.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macrotoolkit import api as mtk
from macrotoolkit import authoring as au
from macrotoolkit.authoring.stan import TEMPLATE_NAME, render_context
from macrotoolkit.engine import ConstantMeasurementNoise, DataLoadings, simulate_forward
from macrotoolkit.families.base import ExogLag, ObsLag, StateSpaceMeta
from macrotoolkit.render import render_stan_source
from macrotoolkit.results_core import hd_bar_names, impulse_response, observable_bars, state_components
from macrotoolkit.smoother import _as_Z_path, kalman_loglik, kalman_smoother, simulate_smoother_draw
from s9_models import constant_regression, example1_data, handbook_kalman_filter, tvp_ar1, tvp_regression, tvp_var
from specs.schema.authored import AuthoredOptions
from specs.schema.equations import LinearityError, ProductTerm, SeriesTerm, linearize, parse_equation

pytestmark = pytest.mark.authored

N_POINTS = 50
SEED = 20260915


def _draws(model: au.Model, n: int, seed: int):
    c = model.compiled()
    priors = c.resolve_priors({})
    rng = np.random.default_rng(seed)
    return c, [c.sample_prior_params(priors, rng, {}) for _ in range(n)]


def _model(measurement, transition=(), exogenous=(), name="t"):
    text = " ".join(measurement) + " " + " ".join(transition)
    params = {"s_e": au.half_normal(1), "s_eta": au.half_normal(1)}
    if "a*" in text or "*a" in text:
        params["a"] = au.normal(0, 1)
    shocks = {"e": au.shock("s_e"), "e2": au.shock("s_e")}
    if transition:
        shocks["eta"] = au.shock("s_eta")
    else:
        params.pop("s_eta")
    return au.Model(name, observables=["y", "z"], exogenous=list(exogenous), measurement=list(measurement), transition=list(transition),
                    parameters=params, shocks=shocks, initial_state={"b": au.init(0.0, 1.0)} if transition else {})


# ---------------------------------------------------------------------------
# The E4 classification: what compiles, what keeps its S7 rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rhs", ["b*y[-1] + e", "y[-1]*b + e", "a*b*y[-2] + e", "2.0*(b[-1]*u) + e", "b*mean(y[-1], y[-2]) + e", "(b + 1.0)*u[-1] + e", "b*u + e"])
def test_e4_accepted_products_compile_to_a_data_loading(rhs: str) -> None:
    m = _model([f"y = {rhs}", "z = e2"], ["b = b[-1] + eta"], exogenous=["u"])
    st = m.structure
    assert st.has_data_loadings and len(st.Zx) == 1
    (row, slot), terms = next(iter(st.Zx.items()))
    assert row == 0 and len(terms) == 1
    assert st.coefficient_shocks == ("eta",)


def test_e4_distribution_keeps_the_linear_parts() -> None:
    """``(b + 1)*u[-1]`` is ``b*u[-1] + u[-1]``: one product AND one plain
    regressor column, sharing the column."""
    m = _model(["y = (b + 1.0)*u[-1] + e", "z = e2"], ["b = b[-1] + eta"], exogenous=["u"])
    st = m.structure
    assert st.feedback_map == (("exog_lag", "u", 1),)
    assert st.Zx == {(0, 0): ((0, st.Zx[(0, 0)][0][1]),)} and st.Zx[(0, 0)][0][1].emit() == "1.0"
    assert st.A[(0, 0)].emit() == "1.0"


@pytest.mark.parametrize(
    "rhs, transition, fragment",
    [
        ("b*b + e", ["b = b[-1] + eta"], "multiplies two series/shock terms"),
        ("b*e + e2", ["b = b[-1] + eta"], "multiplies two series/shock terms"),
        ("y[-1]*z[-1] + e", [], "multiplies two series/shock terms"),
        ("b*z + e", ["b = b[-1] + eta"], "multiplies two series/shock terms"),  # a contemporaneous observable is not data
        ("b*y[-1]*u + e", ["b = b[-1] + eta"], "multiplies two series/shock terms"),  # a triple product
        ("y[-1]/b + e", ["b = b[-1] + eta"], "divides by a series/shock term"),
    ],
)
def test_e4_rejections_keep_the_s7_messages(rhs: str, transition, fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        _model([f"y = {rhs}", "z = e2"], transition, exogenous=["u"])


def test_e4_rejection_message_names_the_allowed_shape() -> None:
    with pytest.raises(ValueError, match="a STATE times a lagged observable"):
        _model(["y = b*b + e", "z = e2"], ["b = b[-1] + eta"])


def test_e4_product_in_a_transition_is_e6_and_rejected() -> None:
    with pytest.raises(ValueError, match="time-varying TRANSITION"):
        _model(["y = b + e", "z = e2"], ["b = b[-1]*y[-1] + eta"])


def test_e4_coarse_symbol_kind_still_rejects_every_product() -> None:
    """A caller classifying every series as the S7 ``"symbol"`` gets the
    S7 behaviour: no product compiles."""
    eq = parse_equation("y = b*x[-1]")
    with pytest.raises(LinearityError, match="multiplies two series/shock terms"):
        linearize(eq.rhs, lambda n: "symbol" if n in ("b", "x") else None)
    lf = linearize(eq.rhs, lambda n: {"b": "state", "x": "observable"}.get(n))
    assert list(lf.terms) == [ProductTerm(SeriesTerm("b", 0), SeriesTerm("x", 1))]


def test_e4_state_lag_beyond_the_carried_head_is_an_error() -> None:
    with pytest.raises(ValueError, match="carried lagged"):
        au.Model("t", observables=["y"], measurement=["y = b*y[-1] + e"], transition=["b[-1] = b[-2] + eta"],
                 parameters={"s_e": au.half_normal(1), "s_eta": au.half_normal(1)}, shocks={"e": au.shock("s_e"), "eta": au.shock("s_eta")},
                 initial_state={"b": au.init(0, 1)})


# ---------------------------------------------------------------------------
# Structure: the oracle models
# ---------------------------------------------------------------------------


def test_tvp_regression_structure_and_meta() -> None:
    m = tvp_regression()
    st = m.structure
    assert st.feedback_map == (("exog_lag", "X", 0),) and st.lag_depth == 0
    assert st.Z == {} and st.Zx == {(0, 0): ((0, st.Zx[(0, 0)][0][1]),)} and st.Zx[(0, 0)][0][1].emit() == "1.0"
    assert st.A == {}  # X enters ONLY through the loading
    assert st.coefficient_shocks == ("eta",) and st.matrix_is_numeric("Z")
    meta = m.compiled().meta
    assert meta.data_loadings == (("Y", ("beta", 0), 0),) and meta.coefficient_shocks == ("eta",)
    assert meta.additive_state_shocks() == () and meta.feedback_map == (ExogLag("X", 0),)
    assert hd_bar_names(meta) == ("init", "e", "exog")


def test_tvp_ar1_structure() -> None:
    st = tvp_ar1(sv=False).structure
    assert st.state_labels == (("c", 0), ("b", 0))
    assert st.feedback_map == (("obs_lag", "pi", 1),)
    assert st.Z == {(0, 0): st.Z[(0, 0)]} and st.Z[(0, 0)].emit() == "1.0"  # c enters additively
    assert st.Zx == {(0, 1): ((0, st.Zx[(0, 1)][0][1]),)}  # b multiplies pi[-1]
    assert st.coefficient_shocks == ("eta_b",)
    meta = tvp_ar1(sv=False).compiled().meta
    assert meta.additive_state_shocks() == ("eta_c",) and hd_bar_names(meta) == ("init", "eta_c", "e")


def test_tvp_var_e5_composition_composes_zx_with_the_a0_coefficient() -> None:
    m = tvp_var(p=1)
    st = m.structure
    labels = list(st.state_labels)
    c_y, b_yy, b_ypi, c_pi, b_piy, b_pipi = (labels.index((s, 0)) for s in ("c_y", "b_y_y_1", "b_y_pi_1", "c_pi", "b_pi_y_1", "b_pi_pi_1"))
    col_y, col_pi = 0, 1
    assert st.feedback_map == (("obs_lag", "y", 1), ("obs_lag", "pi", 1))
    # Own rows: unit products on the own coefficient states.
    assert st.Zx[(0, b_yy)][0][0] == col_y and st.Zx[(0, b_ypi)][0][0] == col_pi
    # The pi row (= a0*y + own): y's Zx and Z entries composed with a0.
    assert st.Zx[(1, b_yy)] == ((col_y, st.Zx[(1, b_yy)][0][1]),) and st.Zx[(1, b_yy)][0][1].emit() == "a0_pi_y"
    assert st.Zx[(1, b_ypi)][0][1].emit() == "a0_pi_y"
    assert st.Z[(1, c_y)].emit() == "a0_pi_y" and st.Z[(1, c_pi)].emit() == "1.0"
    assert st.Zx[(1, b_piy)][0][1].emit() == "1.0"
    assert not st.matrix_is_numeric("Z")  # a0 is a parameter -> Zt in transformed parameters
    assert set(st.coefficient_shocks) == {"eta_b_y_y_1", "eta_b_y_pi_1", "eta_b_pi_y_1", "eta_b_pi_pi_1"}
    assert st.meas_loadings["e_y"] == ("y", "pi")


def test_numeric_z_path_equals_the_hand_construction_at_prior_draws() -> None:
    m = tvp_var(p=1)
    c, points = _draws(m, 10, SEED)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((7, 2))
    st = c.structure
    for p in points:
        Z_path = c.build_Z_path(p, x)
        Z0 = c.build_Z(p)
        assert Z_path.shape == (7, 2, st.n_state)
        for t in range(7):
            hand = Z0.copy()
            for (i, k), terms in st.Zx.items():
                for j, e in terms:
                    from specs.schema.equations import evaluate

                    hand[i, k] += x[t, j] * evaluate(e, p)
            np.testing.assert_allclose(Z_path[t], hand, rtol=0, atol=1e-15)
        # build_matrices returns the path; without x it refuses.
        F, Q, A, Z, R = c.build_matrices(p, x=x)
        assert Z.shape == Z_path.shape and np.array_equal(Z, Z_path)
        with pytest.raises(ValueError, match="needs the regressor matrix"):
            c.build_matrices(p)
    # DataLoadings.at/path agree entry for entry.
    dl = c.build_loadings(points[0])
    assert isinstance(dl, DataLoadings)
    np.testing.assert_array_equal(np.stack([dl.at(x[t]) for t in range(7)]), dl.path(x))


def test_meta_validation_of_data_loadings() -> None:
    base = dict(state_labels=(("b", 0),), state_shocks=("eta",), shock_loadings={"eta": {("b", 0): 1.0}}, measurement_shocks=("e",), obs_names=("y",), feedback_map=(ObsLag("y", 1),))
    StateSpaceMeta(**base, data_loadings=(("y", ("b", 0), 0),), coefficient_shocks=("eta",))
    with pytest.raises(ValueError, match="unknown observable"):
        StateSpaceMeta(**base, data_loadings=(("q", ("b", 0), 0),))
    with pytest.raises(ValueError, match="unknown state slot"):
        StateSpaceMeta(**base, data_loadings=(("y", ("b", -1), 0),))
    with pytest.raises(ValueError, match="not a data column"):
        StateSpaceMeta(**base, data_loadings=(("y", ("b", 0), 3),))
    with pytest.raises(ValueError, match="are not state shocks"):
        StateSpaceMeta(**base, coefficient_shocks=("e",))


# ---------------------------------------------------------------------------
# The Stan text
# ---------------------------------------------------------------------------


def test_emitter_prints_zt_in_the_right_block_with_the_python_operation_order() -> None:
    c = tvp_regression().compiled()
    src = render_stan_source(TEMPLATE_NAME, render_context(c, c.resolve_priors({})))
    td = src.split("transformed data")[1].split("parameters {")[0]
    assert "array[T] matrix[1, 1] Zt;" in td and "Zt[t][1, 1] = x[t, 1];" in td
    assert "kalman_loglik(yobs, x, F, Q, A, Zt, R, xi00, P00)" in src
    c = tvp_var(p=1).compiled()
    src = render_stan_source(TEMPLATE_NAME, render_context(c, c.resolve_priors({})))
    tp = src.split("transformed parameters")[1]
    assert "array[T] matrix[2, 6] Zt;" in tp and "Zt[t][2, 1] = a0_pi_y;" in tp
    assert "Zt[t][2, 2] = (a0_pi_y) * x[t, 1];" in tp and "Zt[t][1, 2] = x[t, 1];" in tp
    assert "matrix[2, 6] Z " not in src  # no constant Z when Zt exists


# ---------------------------------------------------------------------------
# The KF / DK smoother with a Z_t path
# ---------------------------------------------------------------------------


def test_as_z_path_and_the_constant_z_bit_identity() -> None:
    assert _as_Z_path(np.eye(2), 5).shape == (5, 2, 2)
    with pytest.raises(ValueError, match="7 entries but yobs has T=5"):
        _as_Z_path(np.zeros((7, 2, 2)), 5)
    with pytest.raises(ValueError, match="must be"):
        _as_Z_path(np.zeros(3), 5)
    c, points = _draws(tvp_ar1(sv=False), 3, SEED)
    df = example1_data()
    y = df["Y"].to_numpy()[:80] + 0.3
    series = {"pi": y}
    yobs, x = c.regressors(series)
    xi00, P00 = c.initial_state(series)
    F, Q, A, Z, R = c.build_matrices(points[0], x=x)
    ll_path = kalman_loglik(yobs, x, F, Q, A, Z, R, xi00, P00)
    # A constant Z tiled into a path gives the bit-identical value.
    Zc = np.array([[1.0, 0.7]])
    ll_c = kalman_loglik(yobs, x, F, Q, A, Zc, R, xi00, P00)
    ll_c_path = kalman_loglik(yobs, x, F, Q, A, np.repeat(Zc[None], yobs.shape[0], axis=0), R, xi00, P00)
    assert ll_c == ll_c_path and np.isfinite(ll_path) and ll_path != ll_c


def test_dk_smoother_with_a_z_path_zero_noise_identity_and_exact_residuals() -> None:
    c, points = _draws(tvp_ar1(sv=False), 2, SEED + 1)
    y = example1_data()["Y"].to_numpy()[:120] * 3.0 + 1.0
    series = {"pi": y}
    yobs, x = c.regressors(series)
    xi00, P00 = c.initial_state(series)
    for p in points:
        F, Q, A, Z, R = c.build_matrices(p, x=x)
        sm = kalman_smoother(yobs, x, F, Q, A, Z, R, xi00, P00)
        d0 = simulate_smoother_draw(yobs, x, F, Q, A, Z, R, xi00, P00, np.random.default_rng(0), meta=c.meta, zero_noise=True)
        np.testing.assert_allclose(d0.xi_draw, sm["xi_smooth"], rtol=0, atol=1e-12)
        d = simulate_smoother_draw(yobs, x, F, Q, A, Z, R, xi00, P00, np.random.default_rng(1), meta=c.meta)
        # The measurement shock recovered through Z_t reproduces the data exactly.
        recon = np.einsum("tmn,tn->tm", Z, d.xi_draw)[:, 0] + d.meas_shocks["e"]
        np.testing.assert_allclose(recon, yobs[:, 0], rtol=0, atol=1e-10)
        # ... and the state shocks reconstruct the transitions.
        w = d.xi_draw[1:] - d.xi_draw[:-1] @ F.T
        np.testing.assert_allclose(w[:, 0], d.state_shocks["eta_c"][1:], atol=1e-12)
        np.testing.assert_allclose(w[:, 1], d.state_shocks["eta_b"][1:], atol=1e-12)


# ---------------------------------------------------------------------------
# Oracle (a): TVP regression with Q = 0 vs the constant-coefficient regression
# ---------------------------------------------------------------------------


def test_oracle_a_tvp_regression_with_q0_equals_the_constant_regression_exactly() -> None:
    df = example1_data().iloc[:200]
    series = {"Y": df["Y"].to_numpy(), "X": df["X"].to_numpy()}
    c_tvp = tvp_regression().compiled()
    c_const = constant_regression().compiled()
    yobs, x = c_tvp.regressors(series)
    yobs2, x2 = c_const.regressors(series)
    np.testing.assert_array_equal(x, x2)
    n0 = c_const.meta.n_state
    rng = np.random.default_rng(SEED)
    worst = 0.0
    for _ in range(N_POINTS):
        beta, s_e = rng.normal(0.0, 1.0), abs(rng.normal(0.0, 0.5))
        F, Q, A, Z, R = c_tvp.build_matrices({"s_e": s_e, "s_eta": 0.0}, x=x)
        ll_tvp = kalman_loglik(yobs, x, F, Q, A, Z, R, np.array([beta]), np.zeros((1, 1)))
        F2, Q2, A2, Z2, R2 = c_const.build_matrices({"beta": beta, "s_e": s_e})
        ll_const = kalman_loglik(yobs2, x2, F2, Q2, A2, Z2, R2, np.zeros(n0), np.zeros((n0, n0)))
        worst = max(worst, abs(ll_tvp - ll_const))
        assert ll_tvp == ll_const, (beta, s_e, ll_tvp, ll_const)
    assert worst == 0.0


def test_oracle_a_tvp_regression_with_q0_and_diffuse_beta_equals_the_marginal_closed_form() -> None:
    """``beta ~ N(m0, P0)`` integrated: ``Y ~ N(m0 X, s^2 I + P0 X X')`` by
    the matrix determinant lemma / Sherman-Morrison (the E0 oracle with a
    regressor, which S8 could not express)."""
    df = example1_data().iloc[:150]
    series = {"Y": df["Y"].to_numpy(), "X": df["X"].to_numpy()}
    c = tvp_regression().compiled()
    yobs, x = c.regressors(series)
    X, Y = x[:, 0], yobs[:, 0]
    rng = np.random.default_rng(SEED + 2)
    T = len(Y)
    for _ in range(N_POINTS):
        m0, P0, s_e = rng.normal(0.0, 1.0), abs(rng.normal(0.0, 1.0)) + 0.05, abs(rng.normal(0.0, 0.5)) + 0.01
        F, Q, A, Z, R = c.build_matrices({"s_e": s_e, "s_eta": 0.0}, x=x)
        ll = kalman_loglik(yobs, x, F, Q, A, Z, R, np.array([m0]), np.array([[P0]]))
        r = Y - m0 * X
        s2 = s_e**2
        xx = float(X @ X)
        logdet = T * np.log(s2) + np.log(1.0 + P0 * xx / s2)
        quad = float(r @ r) / s2 - P0 * float(X @ r) ** 2 / (s2 * (s2 + P0 * xx))
        closed = -0.5 * (T * np.log(2 * np.pi) + logdet + quad)
        assert abs(ll - closed) < 1e-9 * max(1.0, abs(closed)), (ll, closed)


# ---------------------------------------------------------------------------
# Oracle (b, i): the handbook's own Chapter 3 example 1 filter
# ---------------------------------------------------------------------------


def test_oracle_b_handbook_example1_filter_reproduces_the_kf_filtered_beta() -> None:
    df = example1_data()
    series = {"Y": df["Y"].to_numpy(), "X": df["X"].to_numpy()}
    c = tvp_regression().compiled()
    yobs, x = c.regressors(series)
    Q_true, R_true = 0.001, 0.01
    F, Q, A, Z, R = c.build_matrices({"s_e": np.sqrt(R_true), "s_eta": np.sqrt(Q_true)}, x=x)
    assert Q[0, 0] == pytest.approx(Q_true) and R[0, 0] == pytest.approx(R_true)
    sm = kalman_smoother(yobs, x, F, Q, A, Z, R, np.array([0.0]), np.array([[1.0]]))
    beta_tt, ptt = handbook_kalman_filter(yobs[:, 0], x[:, 0], beta0=0.0, p00=1.0, F=1.0, Q=Q_true, R=R_true)
    np.testing.assert_allclose(sm["xi_filt"][:, 0], beta_tt, rtol=0, atol=1e-10)
    np.testing.assert_allclose(sm["P_filt"][:, 0, 0], ptt, rtol=0, atol=1e-12)
    # The filtered path tracks the truth (the handbook's figure): RMSE well below the beta scale.
    truth = df["beta_true"].to_numpy()
    assert np.sqrt(np.mean((beta_tt - truth) ** 2)) < 0.3 * np.std(truth)  # a sanity bound (the filter RMSE is ~0.05 at R = 0.01), not the oracle


# ---------------------------------------------------------------------------
# The engine: bilinear closure
# ---------------------------------------------------------------------------


class _ZeroRng:
    def standard_normal(self, size=None):
        return np.zeros(size) if size is not None else 0.0

    def normal(self, loc=0.0, scale=1.0, size=None):
        return loc


def test_engine_forward_simulation_builds_zt_from_the_simulated_path() -> None:
    """Zero noise: ``pi_t = c + b pi_{t-1}`` with the coefficients held at
    the terminal state -- the hand recursion, Z_t from the simulated
    regressors (the feedback closure with time-varying coefficients)."""
    c = tvp_ar1(sv=False).compiled()
    p = {"s_c": 0.1, "s_b": 0.1, "s_e": 0.5}
    F, Q, A, Z, R = c.build_F(p), c.build_Q(p), c.build_A(p), c.build_Z(p), c.build_R(p)
    dl = c.build_loadings(p)
    xi_last = np.array([0.4, 0.8])
    out = simulate_forward(F, np.zeros((2, 2)), A, Z, c.meta, xi_last, {"pi": {1: 2.0}}, {}, {}, ConstantMeasurementNoise((0.5,)), 5, _ZeroRng(), loadings=dl)
    pi = 2.0
    for t in range(5):
        pi = 0.4 + 0.8 * pi
        assert out["obs"][t, 0] == pytest.approx(pi, abs=1e-14)
    # Without the loadings the same call ignores the product (Z0 only): the pre-S9 arithmetic.
    out0 = simulate_forward(F, np.zeros((2, 2)), A, Z, c.meta, xi_last, {"pi": {1: 2.0}}, {}, {}, ConstantMeasurementNoise((0.5,)), 5, _ZeroRng())
    np.testing.assert_allclose(out0["obs"][:, 0], 0.4)


def test_engine_hd_identity_holds_exactly_with_coefficients_held_at_the_drawn_path() -> None:
    c, points = _draws(tvp_ar1(sv=False), 3, SEED + 3)
    y = example1_data()["Y"].to_numpy()[:150] * 2.0 + 0.5
    series = {"pi": y}
    yobs, x = c.regressors(series)
    xi00, P00 = c.initial_state(series)
    seeds = {"pi": {1: float(series["pi"][0])}}
    for p in points:
        F, Q, A, Z, R = c.build_matrices(p, x=x)
        dl = c.build_loadings(p)
        sim = simulate_smoother_draw(yobs, x, F, Q, A, Z, R, xi00, P00, np.random.default_rng(7), meta=c.meta)
        comps = state_components(F, c.meta, sim.xi_draw, sim.state_shocks)
        bars = observable_bars(A, dl.Z0, c.meta, comps, sim.meas_shocks, x, seeds, loadings=dl, coef_path=sim.xi_draw)
        assert set(bars) == {"init", "eta_c", "e"}
        total = sum(bars.values())
        assert np.max(np.abs(total[:, 0] - yobs[:, 0])) < 1e-9 * max(1.0, np.max(np.abs(yobs)))
        with pytest.raises(ValueError, match="coefficient path"):
            observable_bars(A, dl.Z0, c.meta, comps, sim.meas_shocks, x, seeds, loadings=dl)


def test_engine_dated_irf_is_the_hand_recursion_at_the_reference_coefficient() -> None:
    c = tvp_ar1(sv=False).compiled()
    p = {"s_c": 0.1, "s_b": 0.1, "s_e": 0.5}
    F, A, Z = c.build_F(p), c.build_A(p), c.build_Z(p)
    dl = c.build_loadings(p)
    coef = np.array([0.3, 0.6])  # (c, b) at the reference date
    comp, obs = impulse_response(F, A, Z, c.meta, "e", 2.0, 6, loadings=dl, coef_state=coef)
    np.testing.assert_allclose(obs[:, 0], 2.0 * 0.6 ** np.arange(6), rtol=1e-13)
    comp, obs = impulse_response(F, A, Z, c.meta, "eta_c", 1.0, 6, loadings=dl, coef_state=coef)
    # A permanent intercept shift propagating through b: y_h = 1 + 0.6 y_{h-1}.
    y, expect = 0.0, []
    for _ in range(6):
        y = 1.0 + 0.6 * y
        expect.append(y)
    np.testing.assert_allclose(obs[:, 0], expect, rtol=1e-13)
    with pytest.raises(ValueError, match="moves only time-varying coefficients"):
        impulse_response(F, A, Z, c.meta, "eta_b", 1.0, 6, loadings=dl, coef_state=coef)
    with pytest.raises(ValueError, match="reference coefficient state"):
        impulse_response(F, A, Z, c.meta, "e", 1.0, 6, loadings=dl)


def test_hd_identity_gate_at_prior_points_for_the_tvp_ar1() -> None:
    from macrotoolkit.authoring.results import hd_reconstruction_error

    df = example1_data().iloc[:160].copy()
    df["pi"] = df["Y"] * 2.0 + 0.5
    spec = mtk.spec("authored", options=tvp_ar1(sv=False), data={"file": "dataframe.csv", "date_column": "date", "mapping": {"pi": "pi"}})
    for k in range(3):
        err, scale = hd_reconstruction_error(spec, df[["date", "pi"]], k)
        assert err < max(1e-6, 1e-12 * scale)


def test_is_stationary_uses_the_coefficient_state_at_the_reference() -> None:
    c = tvp_ar1(sv=False).compiled()
    p = {"s_c": 0.1, "s_b": 0.1, "s_e": 0.5}
    assert c.is_stationary(p)  # coefficients at zero
    assert c.is_stationary(p, xi_ref=np.array([0.0, 0.9]))
    assert not c.is_stationary(p, xi_ref=np.array([0.0, 1.2]))


# ---------------------------------------------------------------------------
# Hash round trip
# ---------------------------------------------------------------------------


def test_e4_round_trips_through_yaml_and_changes_the_hash() -> None:
    m = tvp_ar1(sv=True)
    data = {"file": "dataframe.csv", "date_column": "date", "mapping": {"pi": "infl"}}
    spec = mtk.spec("authored", options=m, data=data, outputs={"irf_dates": ["1990-01-01"]})
    assert spec.outputs.irf_dates == ["1990-01-01"]
    text = spec.to_estimation_yaml()
    assert "pi = c + b*pi[-1] + e" in text and "irf_dates" not in text  # outputs are report options, not identity (the S5 split)
    import yaml

    opts = AuthoredOptions.model_validate(yaml.safe_load(text)["model"]["options"])
    assert opts.structure.Zx.keys() == m.structure.Zx.keys() and opts.structure.coefficient_shocks == ("eta_b",)
    const = au.Model("tvp_ar1_sv", observables=["pi"], measurement=["pi = c + b + e"], transition=["c = c[-1] + eta_c", "b = b[-1] + eta_b"],
                     parameters={"s_c": au.half_normal(0.1), "s_b": au.half_normal(0.05)},
                     shocks={"eta_c": au.shock("s_c"), "eta_b": au.shock("s_b"), "e": au.sv(sigma_h=0.2, mu_h0=au.log_var_diff("pi", 0.5))},
                     initial_state={"c": au.init(0.0, 1.0), "b": au.init(0.5, 0.5)})
    assert mtk.spec("authored", options=const, data=data).to_estimation_yaml() != mtk.spec("authored", options=m, data=data).to_estimation_yaml()
