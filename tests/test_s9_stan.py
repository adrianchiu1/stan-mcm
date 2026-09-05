"""S9 Stan-side gates (plans/S9-plan.md): the generic template renders,
compiles and its ``kf_loglik`` equals the Python mirror at 50 prior draws
for the three E4 shapes (a numeric ``Zt`` in transformed data for the TVP
regression; ``Zt`` with ``Rt`` for the TVP-AR(1) with SV; a parameter-
dependent ``Zt`` in transformed parameters for the recursive TVP-VAR);
oracle (b, ii) -- the handbook Chapter 3 example 1/2 DGP fitted by NUTS
recovers ``beta_t`` within bands; the fitted-run outputs (HD identity,
dated IRFs with the coefficient-only shocks omitted, the fan); and oracle
(c) -- the parameter-recovery gate (G2's shape) on the TVP-AR(1) with SV
(a fast smoke at a tiny design; the registered design under ``slow``).
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
from s9_models import example1_data, handbook_kalman_filter, tvp_ar1, tvp_regression, tvp_var

pytestmark = pytest.mark.authored

N_POINTS = 50
TOL = 1e-8
RTOL = 1e-11
SEED = 20260916


def _program(model: au.Model):
    compiled = model.compiled()
    priors = compiled.resolve_priors({})
    source = render_stan_source(TEMPLATE_NAME, render_context(compiled, priors))
    stan_model, _ = compile_model(source)
    return compiled, priors, stan_model, source


def _gate(model: au.Model, df: pd.DataFrame, n_points: int = N_POINTS) -> tuple[str, float]:
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
    assert np.all(diffs < gates), f"{model.options.name}: Stan vs Python max |diff| = {np.max(diffs):.3e}"
    return source, float(np.max(diffs))


def test_tvp_regression_program_mirrors_with_numeric_zt_in_transformed_data() -> None:
    source, worst = _gate(tvp_regression(), example1_data())
    td = source.split("transformed data")[1].split("parameters {")[0]
    assert "array[T] matrix[1, 1] Zt;" in td and "Zt[t][1, 1] = x[t, 1];" in td
    print(f"tvp_regression mirror max |diff| {worst:.2e}")


def test_tvp_ar1_sv_program_mirrors_with_zt_and_rt() -> None:
    df = example1_data().iloc[:160].copy()
    df["pi"] = df["Y"] * 3.0 + 1.0
    source, worst = _gate(tvp_ar1(sv=True), df[["date", "pi"]])
    assert "array[T] matrix[1, 2] Zt;" in source and "kalman_loglik(yobs, x, F, Q, A, Zt, Rt, xi00, P00)" in source
    print(f"tvp_ar1_sv mirror max |diff| {worst:.2e}")


def test_tvp_var_program_mirrors_with_parameter_dependent_zt_in_transformed_parameters() -> None:
    from macrotoolkit.run import REPO_ROOT

    df = pd.read_csv(REPO_ROOT / "examples/handbook/data/ch2_datain.csv", parse_dates=["date"]).rename(columns={"gdp_growth": "y", "inflation": "pi"}).iloc[:120]
    source, worst = _gate(tvp_var(p=1), df)
    tp = source.split("transformed parameters")[1]
    assert "array[T] matrix[2, 6] Zt;" in tp and "Zt[t][2, 2] = (a0_pi_y) * x[t, 1];" in tp
    print(f"tvp_var mirror max |diff| {worst:.2e}")


# ---------------------------------------------------------------------------
# Oracle (b, ii): the example 1/2 DGP fitted -- beta_t recovered within bands
# ---------------------------------------------------------------------------


def test_oracle_b_example1_fit_recovers_beta_t_within_bands(tmp_path: Path) -> None:
    df = example1_data()
    spec = mtk.spec("authored", options=tvp_regression(), data={"file": "dataframe.csv", "date_column": "date", "mapping": {"Y": "Y", "X": "X"}},
                    sampler={"chains": 2, "warmup": 300, "sampling": 300, "seed": 20260917}, outputs={"smoother_draws": {"thin": 10}, "prior_predictive_draws": 5, "horizon": 8})
    run = mtk.fit(spec, df[["date", "Y", "X"]], runs_root=tmp_path / "runs")
    assert run.mirror_check["passed"]
    post = run.idata.posterior
    s_e, s_eta = np.asarray(post["s_e"].values).reshape(-1), np.asarray(post["s_eta"].values).reshape(-1)
    # The DGP's scales sqrt(0.01) = 0.1 and sqrt(0.001) = 0.0316 inside the 90% intervals.
    assert np.quantile(s_e, 0.05) < 0.1 < np.quantile(s_e, 0.95)
    assert np.quantile(s_eta, 0.05) < np.sqrt(0.001) < np.quantile(s_eta, 0.95)
    sd = run.outputs().compute("states")
    beta = sd.states["beta"]  # (n_draws, T)
    truth = df["beta_true"].to_numpy()
    lo, hi = np.quantile(beta, 0.05, axis=0), np.quantile(beta, 0.95, axis=0)
    inside = np.mean((truth >= lo) & (truth <= hi))
    assert inside >= 0.80, inside
    med = np.median(beta, axis=0)
    beta_tt, _ = handbook_kalman_filter(df["Y"].to_numpy(), df["X"].to_numpy(), 0.0, 1.0, 1.0, 0.001, 0.01)
    rmse_smooth, rmse_filter = np.sqrt(np.mean((med - truth) ** 2)), np.sqrt(np.mean((beta_tt - truth) ** 2))
    assert rmse_smooth < rmse_filter, (rmse_smooth, rmse_filter)
    print(f"example1 fit: truth inside the 90% band at {inside:.1%} of periods; RMSE smoothed {rmse_smooth:.4f} vs the handbook filter at the true parameters {rmse_filter:.4f}")
    # The outputs over a TVP run: the HD identity, the coefficient-only shock omitted from the bars and the IRFs, the fan.
    outs = run.outputs()
    hd = outs.compute("hd")
    assert hd.bars == ("init", "e", "exog") and "eta" in hd.states["beta"]
    total = sum(hd.obs["Y"][b] for b in hd.bars)
    assert np.max(np.abs(total - df["Y"].to_numpy()[None, :])) < 1e-8
    irf = outs.compute("irf")
    assert irf.shocks == ("e",) and irf.omitted_shocks == ("eta",) and "no impulse response from rest" in irf.omitted_reason
    assert irf.reference_dates == (str(run.results().dates[-1].date()),)
    fans = outs.compute("fan")
    assert np.all(np.isfinite(fans.obs["Y"])) and fans.obs["Y"].shape[1] == 8
    html = run.report().read_text()
    assert "PASS:" in html


# ---------------------------------------------------------------------------
# The dated IRFs of the TVP-AR(1) on a fitted run
# ---------------------------------------------------------------------------


def test_tvp_ar1_dated_irfs_and_hd_identity_on_a_fitted_run(tmp_path: Path) -> None:
    from macrotoolkit.run import REPO_ROOT

    df = pd.read_csv(REPO_ROOT / "examples/handbook/data/ch1_inflation.csv", parse_dates=["date"]).iloc[:120]
    spec = mtk.spec("authored", options=tvp_ar1(sv=False), data={"file": "dataframe.csv", "date_column": "date", "mapping": {"pi": "inflation"}},
                    sampler={"chains": 2, "warmup": 250, "sampling": 250, "seed": 20260918},
                    outputs={"smoother_draws": {"thin": 25}, "prior_predictive_draws": 3, "irf_horizon": 8, "irf_dates": ["1960-01-01", "1975-01-01"]})
    run = mtk.fit(spec, df, runs_root=tmp_path / "runs")
    assert run.mirror_check["passed"]
    outs = run.outputs()
    irf = outs.compute("irf")
    assert irf.reference_dates == ("1960-01-01", "1975-01-01") and set(irf.by_date) == {"1960-01-01", "1975-01-01"}
    assert irf.shocks == ("eta_c", "e") and irf.omitted_shocks == ("eta_b",)
    # A one-sd e shock at date tau propagates as b_tau^h: the impact is the shock size, the decay the drawn b.
    r = irf.by_date["1975-01-01"]["e"]["pi"]
    assert np.all(r[:, 0] > 0) and np.all(np.abs(r[:, 1] / r[:, 0]) < 1.5)
    hd = outs.compute("hd")
    assert hd.bars == ("init", "eta_c", "e")
    total = sum(hd.obs["pi"][b] for b in hd.bars)
    assert np.max(np.abs(total - run.results().yobs[:, 0][None, :])) < 1e-8
    with pytest.raises(ValueError, match="not an estimation row"):
        bad = spec.model_copy(update={"outputs": spec.outputs.model_copy(update={"irf_dates": ["1800-01-01"]})})
        from macrotoolkit.authoring.results import compute_irf_draws, load_authored_run
        import dataclasses

        compute_irf_draws(dataclasses.replace(load_authored_run(run.run_dir), spec=bad))
    fig = outs.figure("irf")
    assert fig is not None


# ---------------------------------------------------------------------------
# Oracle (c): parameter recovery (G2's shape) on the TVP-AR(1) with SV
# ---------------------------------------------------------------------------


def _tvp_ar1_sv_recovery_design(n_datasets: int, **sampler):
    from macrotoolkit.authoring.validation import recovery_design

    spec = mtk.spec("authored", options=tvp_ar1(sv=True), data={"file": "dataframe.csv", "date_column": "date", "mapping": {"pi": "infl"}})
    # Fixed anchors (pre-registered, DECISIONS.md 2026-09-05): xi00 = (c, b) = (0.5, 0.5), P00 = diag(0.25, 0.04),
    # mu_h0_e = ln(0.25); T = 150 after one lag row.
    return recovery_design(spec, name="tvp_ar1_sv_recovery", T=150, xi00=np.array([0.5, 0.5]), P00=np.diag([0.25, 0.04]),
                           n_datasets=n_datasets, seed_base=20260919, anchors={"mu_h0_e": float(np.log(0.25))}, **sampler)


def test_oracle_c_recovery_design_runs_on_the_bilinear_simulator_smoke() -> None:
    """The fast smoke: two datasets through the generic structural simulator
    (Z_t from the simulated regressors) and the production render; the
    registered gate below is the evidence."""
    from macrotoolkit.validation.recovery import run_recovery

    design = _tvp_ar1_sv_recovery_design(2, chains=2, iter_warmup=200, iter_sampling=200)
    rng = np.random.default_rng(0)
    truth = design.draw_truth(rng)
    ds = design.simulate(truth, rng)
    assert ds["pi"].shape == (150,) and np.all(np.isfinite(ds["pi"]))
    data = design.stan_data(ds)
    assert data["T"] == 150 and data["x"].shape == (150, 1) and data["x"][1, 0] == ds["pi"][0]  # the lag column is the simulated path
    res = run_recovery(design)
    assert res.inside.shape == (2, len(design.param_labels)) and len(res.divergences) == 2


@pytest.mark.slow
def test_oracle_c_recovery_gate_tvp_ar1_sv(tmp_path: Path) -> None:
    """The registered design (constants in DECISIONS.md 2026-09-05): 20
    datasets, 2 chains x 500/500, pooled 90%-CI coverage in [0.80, 0.97],
    per-parameter floor 0.6, no systematic bias in the scales."""
    from macrotoolkit.validation.recovery import evaluate_recovery, run_recovery, write_coverage_csv

    design = _tvp_ar1_sv_recovery_design(20, chains=2, iter_warmup=500, iter_sampling=500)
    res = run_recovery(design)
    verdict, reasons, metrics = evaluate_recovery(design, res)
    write_coverage_csv(tmp_path / "coverage.csv", design, res)
    print(f"tvp_ar1_sv recovery: {verdict} {metrics} {reasons}")
    assert verdict == "PASS", reasons
