"""S8 WP1 Stan-side gates: the generic template renders, compiles and
its ``kf_loglik`` equals the Python mirror at prior draws for every
extension (E0 zero/no states, E1 intercepts and drifts, E2 lag-0
exogenous regressors, E3 a shock-free row, E5 a recursive VAR with a
non-diagonal R), through the S6 fixed_param mechanism; and the E5 fit
oracle -- a bivariate VAR(2) on the handbook's Chapter 2 data in recursive
form with flat-ish priors gives posterior means within Monte Carlo error
of OLS, with the reduced-form Sigma reconstructed from the recursive form
matching the OLS residual covariance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macrotoolkit import api as mtk
from macrotoolkit import authoring as au
from macrotoolkit.authoring.family import python_kf_loglik, stan_inits_and_h
from macrotoolkit.authoring.stan import TEMPLATE_NAME, render_context
from macrotoolkit.qc import MirrorPoint, stan_kf_loglik_at_points
from macrotoolkit.render import compile_model, render_stan_source
from test_s8_grammar import ar2_constant_param, ar2_constant_state, bivar_recursive, drift_ar1_state, exog_model, uc_trend_cycle

pytestmark = pytest.mark.authored

REPO = Path(__file__).resolve().parents[1]
N_POINTS = 50
TOL = 1e-8  # G1's absolute gate ...
RTOL = 1e-11  # ... with the S8 relative slack for badly scaled (flat-prior) points: |diff| < max(TOL, RTOL * |loglik|)
SEED = 20260913


def _program(model: au.Model):
    compiled = model.compiled()
    priors = compiled.resolve_priors({})
    source = render_stan_source(TEMPLATE_NAME, render_context(compiled, priors))
    stan_model, _ = compile_model(source)
    return compiled, priors, stan_model, source


def _gate(model: au.Model, df: pd.DataFrame, n_points: int = N_POINTS) -> str:
    compiled, priors, stan_model, source = _program(model)
    data = compiled.stan_data(df)
    anchors = {k: v for k, v in data.items() if k.startswith("mu_h0_")}
    rng = np.random.default_rng(SEED)
    inits, py = [], []
    for _ in range(n_points):
        params = compiled.sample_prior_params(priors, rng, anchors)
        ia, h = stan_inits_and_h(compiled, params, priors, anchors, rng, data["T"])
        inits.append(ia)
        py.append(python_kf_loglik(compiled, params, data, h))
    ll = stan_kf_loglik_at_points(stan_model, data, [MirrorPoint(inits=i, loglik_python=float("nan")) for i in inits], seed=SEED)
    assert np.all(np.isfinite(ll))
    py = np.asarray(py)
    diffs = np.abs(ll - py)
    gates = np.maximum(TOL, RTOL * np.abs(py))
    d_rel = float(np.max(diffs / np.maximum(1.0, np.abs(py))))
    assert np.all(diffs < gates), f"{model.options.name}: Stan vs Python max |diff| = {np.max(diffs):.3e}, max relative {d_rel:.3e}"
    return source


@pytest.fixture(scope="module")
def infl() -> pd.DataFrame:
    df = pd.read_csv(REPO / "examples/handbook/data/ch1_inflation.csv", parse_dates=["date"])
    return df.rename(columns={"inflation": "infl"})


def test_e0_constant_state_and_zero_state_programs_mirror(infl: pd.DataFrame) -> None:
    source = _gate(ar2_constant_state(), infl)
    assert "matrix[1, 1] Q = rep_matrix(0, 1, 1);   // Q = 0: no state shock" in source.split("transformed data")[1].split("parameters")[0]
    source = _gate(ar2_constant_param(), infl)
    assert "vector[0] xi00 = rep_vector(0, 0);" in source and "matrix[0, 0] F" in source and "matrix[1, 0] Z" in source


def test_e1_drift_program_mirrors(infl: pd.DataFrame) -> None:
    df = infl.rename(columns={"infl": "y"}).iloc[:100]
    source = _gate(drift_ar1_state(), df)
    assert "F[1, 2] = mu;" in source and "F[2, 2] = 1.0;" in source


def test_e2_contemporaneous_exogenous_program_mirrors(infl: pd.DataFrame) -> None:
    rng = np.random.default_rng(4)
    df = infl.rename(columns={"infl": "y"}).iloc[:120].copy()
    df["x"] = rng.standard_normal(len(df)).cumsum()
    _gate(exog_model(), df)


def test_e3_uc_trend_cycle_program_mirrors_with_singular_R(infl: pd.DataFrame) -> None:
    df = infl.rename(columns={"infl": "Y"})
    source = _gate(uc_trend_cycle(), df)
    tp = source.split("transformed parameters")[1]
    assert "matrix[1, 1] R = rep_matrix(0, 1, 1);" in tp and "R[1, 1] =" not in tp  # the shock-free row has no R entry


def test_e5_recursive_var_program_mirrors_with_full_R() -> None:
    df = pd.read_csv(REPO / "examples/handbook/data/ch2_datain.csv", parse_dates=["date"]).rename(columns={"gdp_growth": "y", "inflation": "pi"})
    source = _gate(bivar_recursive(1), df)
    tp = source.split("transformed parameters")[1]
    assert "R[1, 2] = (a0) * square(s_y);" in tp and "R[2, 2] = (a0*a0) * square(s_y) + square(s_pi);" in tp


# ---------------------------------------------------------------------------
# The E5 fit oracle: recursive VAR(2) vs OLS on the handbook data
# ---------------------------------------------------------------------------


def _ols_var2(Y: np.ndarray):
    T = Y.shape[0] - 2
    X = np.column_stack([np.ones(T), Y[1:-1, 0], Y[1:-1, 1], Y[:-2, 0], Y[:-2, 1]])
    B = np.linalg.solve(X.T @ X, X.T @ Y[2:])
    U = Y[2:] - X @ B
    return B, U.T @ U / (T - X.shape[1])


def test_e5_var2_posterior_means_match_ols_and_reduced_form_sigma(tmp_path: Path) -> None:
    df = pd.read_csv(REPO / "examples/handbook/data/ch2_datain.csv", parse_dates=["date"])
    Y = df[["gdp_growth", "inflation"]].to_numpy(dtype=float)
    B_ols, S_ols = _ols_var2(Y)
    spec = mtk.spec("authored", options=bivar_recursive(2), data={"file": "dataframe.csv", "date_column": "date", "mapping": {"y": "gdp_growth", "pi": "inflation"}},
                    sampler={"chains": 4, "warmup": 400, "sampling": 400, "seed": 20260914}, outputs={"smoother_draws": {"thin": 20}, "prior_predictive_draws": 5})
    run = mtk.fit(spec, df, runs_root=tmp_path / "runs")
    assert run.mirror_check["passed"]
    post = run.idata.posterior
    d = {k: np.asarray(post[k].values).reshape(-1) for k in post.data_vars if k != "kf_loglik"}
    # Reduced-form coefficients per draw (eq pi: recursive + a0 * eq y), in the OLS X layout [1, y1, pi1, y2, pi2].
    cols = ["c", "y_1", "pi_1", "y_2", "pi_2"]
    eq_y = np.column_stack([d["c_y"], d["b_y_y_1"], d["b_y_pi_1"], d["b_y_y_2"], d["b_y_pi_2"]])
    eq_pi_rec = np.column_stack([d["c_pi"], d["b_pi_y_1"], d["b_pi_pi_1"], d["b_pi_y_2"], d["b_pi_pi_2"]])
    eq_pi = eq_pi_rec + d["a0"][:, None] * eq_y
    for j, (draws, ols) in enumerate(((eq_y, B_ols[:, 0]), (eq_pi, B_ols[:, 1]))):
        mean, sd = draws.mean(axis=0), draws.std(axis=0)
        ok = np.abs(mean - ols) < 0.25 * sd
        assert np.all(ok), f"equation {j}: posterior means {mean} vs OLS {ols} (sd {sd})"
    # Reduced-form Sigma = M D M' per draw vs the OLS residual covariance.
    a0, sy, spi = d["a0"], d["s_y"], d["s_pi"]
    S11, S12, S22 = (sy**2).mean(), (a0 * sy**2).mean(), (a0**2 * sy**2 + spi**2).mean()
    scale = np.sqrt(S_ols[0, 0] * S_ols[1, 1])  # the off-diagonal is near zero on this data: compare on Sigma's own scale
    np.testing.assert_allclose([S11, S12, S22], [S_ols[0, 0], S_ols[0, 1], S_ols[1, 1]], atol=0.05 * scale)
    # The engine's structural IRFs under the ordering are the Cholesky IRFs: impact = chol(Sigma).
    irf = run.outputs().compute("irf")
    impact = np.array([[irf.responses[s][o][:, 0].mean() for s in ("e_y", "e_pi")] for o in ("y", "pi")])
    # The off-diagonal is near zero on this data (a 15% relative miss is 0.003 in absolute terms over a
    # thinned draw subset): compare on chol(Sigma)'s own scale, as the Sigma check above does.
    np.testing.assert_allclose(impact, np.linalg.cholesky(np.array([[S11, S12], [S12, S22]])), rtol=0.1, atol=0.05 * np.sqrt(scale))
    assert impact[0, 1] == 0.0  # e_pi never hits y on impact: the ordering
    outs = run.outputs()
    assert "fan" in outs.names and outs.unavailable_reason("fan") is None
    fans = outs.compute("fan")
    assert np.all(np.isfinite(fans.obs["pi"]))
    fv = outs.compute("fevd")
    np.testing.assert_allclose(fv.shares["y"]["e_y"][:, 0], 1.0)  # on impact y is its own shock under the ordering
    # The post-processors over this run: sign restrictions on the Cholesky IRF array, a WZ conditional forecast.
    from macrotoolkit.postprocess import SignRestriction, conditional_forecast_for_run, irf_array_from_draws, sign_restricted_irfs

    arr, targets, shocks = irf_array_from_draws(irf, targets=("y", "pi"))
    assert arr.shape[1:] == (run.spec.outputs.irf_horizon, 2, 2) and shocks == ("e_y", "e_pi")
    sr = sign_restricted_irfs(arr, [SignRestriction("supply", "y", -1, (0,)), SignRestriction("supply", "pi", +1, (0,))], targets, np.random.default_rng(1), max_tries=500)
    assert sr.irf.shape[0] > 0 and np.all(sr.irf[:, 0, 0, 0] < 0) and np.all(sr.irf[:, 0, 1, 0] > 0)
    cf = conditional_forecast_for_run(run, {("pi", 0): 1.0, ("pi", 1): 1.0, ("pi", 2): 1.0}, horizon=8, draws_per_posterior_draw=2)
    # The restricted shocks come from a pseudo-inverse solve whose conditioning is draw-dependent (a near-unit-root
    # posterior draw has huge IRFs): 1e-6 absolute on a path of ones (one element reached 1.1e-7 on one container).
    np.testing.assert_allclose(cf.draws[:, :3, 1], 1.0, atol=1e-6)
    assert cf.draws.shape == (2 * len(irf.draw_indices), 8, 2) and np.std(cf.draws[:, 5, 1]) > 0
    html = run.report().read_text()
    assert "PASS:" in html
