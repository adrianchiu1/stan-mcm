"""Family #2 (``ucsv``, S6 WP2): the fast-suite coverage of the family
declarations and their consistency pins, one tiny REAL fit through the
Python API (with the automatic mirror check, the report, and the output
modules), and the G6 historical-decomposition identity for ucsv.

The validation LADDER for ucsv lives in tests/test_ucsv_gates.py (G1 is
part of tests/test_g1_mirror.py's shared-filter gate; G2 recovery and the
pre-registered SBC are slow; the G6 identity is here, fast).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macrotoolkit import api as mtk
from macrotoolkit.families.base import StateSpaceMeta
from macrotoolkit.families.ucsv import (
    UCSV_STATE_META,
    build_render_context,
    build_stan_data,
    build_ucsv_matrices,
    build_ucsv_regressors,
    default_initial_state,
    sample_prior_params,
    ucsv_mu_h0_anchors,
)
from specs.schema import FAMILY_REGISTRY, get_family
from specs.schema.base import RunSpec
from specs.schema.ucsv import DEFAULT_PRIORS, INITIAL_STATE_PRIOR

pytestmark = pytest.mark.ucsv


def _spec(sv_shocks, priors=None, **kw) -> RunSpec:
    return mtk.spec(
        "ucsv",
        data={"file": "infl.csv", "date_column": "date", "mapping": {"pi": "infl"}},
        options={"sv_shocks": sv_shocks},
        priors=priors or {},
        **kw,
    )


def _simulate_ucsv(T: int = 80, seed: int = 20260906) -> pd.DataFrame:
    """A small UCSV-shaped series: random-walk trend + noise, both with
    mild SV -- plausible inflation magnitudes."""
    rng = np.random.default_rng(seed)
    tau = 2.0
    h_eps, h_eta = 2 * np.log(0.8), 2 * np.log(0.3)
    pi = np.empty(T)
    for t in range(T):
        h_eta += 0.1 * rng.standard_normal()
        h_eps += 0.1 * rng.standard_normal()
        tau += rng.normal(0.0, np.exp(h_eta / 2))
        pi[t] = tau + rng.normal(0.0, np.exp(h_eps / 2))
    return pd.DataFrame({"date": pd.date_range("1990-01-01", periods=T, freq="QS"), "infl": pi})


# ---------------------------------------------------------------------------
# Schema + registry
# ---------------------------------------------------------------------------


def test_registry_entry_declares_the_full_output_contract() -> None:
    entry = get_family("ucsv")
    assert entry.template == "ucsv.stan.j2" and entry.required_mapping == ("pi",)
    for cap in ("build_stan_data", "build_render_context", "state_meta", "results_loader",
                "report_writer", "prior_sd_table", "headline_series", "output_modules"):
        assert entry.resolve(cap) is not None, cap
    assert isinstance(entry.resolve("state_meta"), StateSpaceMeta)


def test_options_default_to_both_sv_shocks_and_reject_single_shock() -> None:
    s = RunSpec.model_validate({"model": {"family": "ucsv", "options": {}},
                                "data": {"file": "x.csv", "date_column": "date", "mapping": {"pi": "p"}}})
    assert s.model.options.sv_shocks == ["eps", "eta"]
    assert _spec(["eta", "eps"]).model.options.sv_shocks == ["eps", "eta"]  # canonical order
    assert _spec([]).model.options.sv_shocks == []
    with pytest.raises(ValueError, match="sv_shocks"):
        _spec(["eps"])
    with pytest.raises(ValueError, match="missing required key"):
        mtk.spec("ucsv", data={"file": "x.csv", "date_column": "date", "mapping": {"y": "p"}})
    assert _spec([]).outputs.horizon == 12 and _spec([]).outputs.prior_predictive_draws == 200
    with pytest.raises(ValueError):
        _spec([], outputs={"forecast_r_rule": "neutral"})  # not a ucsv output option


def test_prior_overrides_are_variant_checked() -> None:
    ctx = build_render_context(_spec(["eps", "eta"], {"sigma_h_eta": {"sd": 0.5}}))
    assert ctx["priors"]["sigma_h_eta"]["sd"] == 0.5 and ctx["sv_shocks"] == ["eps", "eta"]
    with pytest.raises(ValueError, match="does not exist in the variant"):
        build_render_context(_spec(["eps", "eta"], {"sigma_eps": {"sd": 2.0}}))
    with pytest.raises(ValueError, match="does not exist in the variant"):
        build_render_context(_spec([], {"sigma_h_eps": {"sd": 0.5}}))
    with pytest.raises(ValueError, match="not a parameter of the ucsv family"):
        build_render_context(_spec([], {"sigma_typo": {"sd": 1.0}}))


# ---------------------------------------------------------------------------
# Metadata + matrices pins
# ---------------------------------------------------------------------------


def test_state_metadata_matches_matrix_builder_and_feedback_map_is_empty() -> None:
    meta = UCSV_STATE_META
    assert meta.state_labels == (("tau", 0),) and meta.feedback_map == ()
    assert meta.state_shocks == ("eta",) and meta.measurement_shocks == ("eps",)
    F, Q, A, Z, R = build_ucsv_matrices({"sigma_eta": 0.3, "sigma_eps": 0.8})
    B = meta.loading_matrix()
    np.testing.assert_array_equal(B @ np.diag([0.3**2]) @ B.T, Q)
    assert F.shape == (1, 1) and Z.shape == (1, 1) and A.shape == (0, 1)
    np.testing.assert_array_equal(R, [[0.8**2]])
    assert meta.recovery_order()[0][:3] == ("eta", ("tau", 0), 1.0)


def test_regressors_initial_state_and_anchors() -> None:
    pi = np.array([1.0, 2.0, 4.0, 3.0, 5.0])
    yobs, x = build_ucsv_regressors(pi)
    assert yobs.shape == (5, 1) and x.shape == (5, 0)
    xi00, P00 = default_initial_state(float(pi[0]))
    np.testing.assert_array_equal(xi00, [1.0])
    np.testing.assert_array_equal(P00, [[INITIAL_STATE_PRIOR["tau0_sd"] ** 2]])
    mu_eps, mu_eta = ucsv_mu_h0_anchors(pi)
    v = np.var(np.diff(pi), ddof=1)
    # Equal split of Var(Delta pi) = sigma_eta^2 + 2 sigma_eps^2.
    assert mu_eta == pytest.approx(np.log(v / 2.0)) and mu_eps == pytest.approx(np.log(v / 4.0))
    data = build_stan_data(pd.DataFrame({"date": pd.date_range("2000-01-01", periods=5, freq="QS"), "pi": pi}))
    assert data["T"] == 5 and data["yobs"].shape == (5, 1) and "x" not in data
    with pytest.raises(ValueError, match="at least 3"):
        build_stan_data(pd.DataFrame({"pi": pi[:2]}))


def test_prior_sampler_mirrors_template_declarations() -> None:
    priors = build_render_context(_spec(["eps", "eta"]))["priors"]
    rng = np.random.default_rng(3)
    draws = [sample_prior_params(priors, True, rng, mu_h0_eps=-1.0, mu_h0_eta=-2.0) for _ in range(4000)]
    assert set(draws[0]) == {"sigma_h_eps", "sigma_h_eta", "h0_eps", "h0_eta"}
    assert all(d["sigma_h_eps"] >= 0 and d["sigma_h_eta"] >= 0 for d in draws)
    h0 = np.array([d["h0_eta"] for d in draws])
    assert abs(h0.mean() + 2.0) < 0.06 and abs(h0.std() - DEFAULT_PRIORS["mu_h0_eta"]["sd"]) < 0.06
    sh = np.array([d["sigma_h_eps"] for d in draws])
    assert abs(sh.mean() - DEFAULT_PRIORS["sigma_h_eps"]["sd"] * np.sqrt(2 / np.pi)) < 0.01
    no_sv = sample_prior_params(build_render_context(_spec([]))["priors"], False, rng)
    assert set(no_sv) == {"sigma_eps", "sigma_eta"}
    with pytest.raises(ValueError, match="anchors"):
        sample_prior_params(priors, True, rng)


# ---------------------------------------------------------------------------
# A tiny real fit through the API: mirror check, results, outputs, report, G6
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ucsv_run(tmp_path_factory) -> mtk.Run:
    df = _simulate_ucsv()
    spec = _spec(["eps", "eta"], sampler={"chains": 2, "warmup": 100, "sampling": 100, "seed": 20260813},
                 outputs={"smoother_draws": {"thin": 4}, "prior_predictive_draws": 20, "horizon": 6})
    return mtk.fit(spec, df, runs_root=tmp_path_factory.mktemp("ucsv_runs"))


def test_fit_records_a_passing_mirror_check_and_saved_kf_loglik(ucsv_run: mtk.Run) -> None:
    mc = ucsv_run.mirror_check
    assert mc is not None and mc["passed"] and mc["n_points"] == 5
    assert mc["max_abs_diff"] < mc["tolerance"] == 1e-8
    assert "kf_loglik" in ucsv_run.idata.posterior.data_vars
    assert (ucsv_run.run_dir / "qc.yaml").is_file()


def test_results_outputs_and_report(ucsv_run: mtk.Run) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    res = ucsv_run.results()
    assert res.sv_on and res.x.shape == (80, 0) and len(res.dates) == 80
    outs = ucsv_run.outputs()
    assert outs.names == ("prior_predictive", "trend_cycle", "irf", "fan", "hd")
    tcd = outs.compute("trend_cycle")
    assert set(tcd.series) == {"tau", "transitory", "vol_eps", "vol_eta"}
    np.testing.assert_allclose(tcd.series["tau"] + tcd.series["transitory"], np.broadcast_to(tcd.pi, tcd.series["tau"].shape), atol=1e-12)
    assert np.all(tcd.series["vol_eps"] > 0)
    irf = outs.compute("irf")
    # A trend shock moves pi one-for-one permanently; a transitory shock lasts one period.
    for j in range(irf.responses["eta"]["pi"].shape[0]):
        np.testing.assert_allclose(irf.responses["eta"]["pi"][j], irf.responses["eta"]["pi"][j][0], atol=1e-12)
        assert np.all(irf.responses["eps"]["pi"][j][1:] == 0.0) and irf.responses["eps"]["tau"][j].max() == 0.0
    fans = outs.compute("fan")
    assert fans.pi.shape == fans.tau.shape == (len(tcd.draw_indices), 6)
    ppd = outs.compute("prior_predictive")
    assert ppd.pi.shape == (20, 80) and np.all(np.isfinite(ppd.pi))
    for name in outs.names:
        f = outs.figure(name)
        assert isinstance(f, Figure) or all(isinstance(v, Figure) for v in f.values())
    plt.close("all")
    table = ucsv_run.param_table()
    assert {"sigma_h_eps", "sigma_h_eta", "kf_loglik"} <= set(table.index)

    html = ucsv_run.report().read_text()
    assert "UCSV report" in html and "KF mirror check" in html and "PASS:" in html
    assert html.count("<img ") == 1 + 1 + 1 + 2 + 1
    assert "http://" not in html and "https://" not in html


def test_g6_hd_identity_for_ucsv(ucsv_run: mtk.Run) -> None:
    """G6 for ucsv: per period, per draw, the observable bars reconstruct
    pi exactly (1e-6, the ladder's identity tolerance; here ~1e-13) and
    the state components reconstruct the drawn tau path."""
    from macrotoolkit.results_ucsv import PI_BARS, compute_historical_decomposition_draws

    hdd = compute_historical_decomposition_draws(ucsv_run.results())
    assert PI_BARS == ("init", "eta", "eps")
    total = sum(hdd.pi[k] for k in PI_BARS)
    pi = ucsv_run.results().pi
    assert np.max(np.abs(total - pi[None, :])) < 1e-6
    tau_total = hdd.tau["init"] + hdd.tau["eta"]
    # tau = pi - eps bar exactly (pi = tau + eps): init+eta bars in pi space equal tau.
    np.testing.assert_allclose(hdd.pi["init"] + hdd.pi["eta"], tau_total, atol=1e-6)


def test_no_sv_variant_fits_and_reports(tmp_path: Path) -> None:
    df = _simulate_ucsv(T=40, seed=5)
    spec = _spec([], sampler={"chains": 2, "warmup": 60, "sampling": 60, "seed": 20260813},
                 outputs={"prior_predictive_draws": 10, "horizon": 4})
    run = mtk.fit(spec, df, runs_root=tmp_path / "runs")
    assert run.mirror_check["passed"]
    assert set(run.param_table().index) >= {"sigma_eps", "sigma_eta"}
    html = run.report().read_text()
    assert "No volatility panel" in html
