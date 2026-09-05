"""S7 M3/M4: an authored model through the ENTIRE existing machinery --
run identity (same equations -> same hash; any equation/prior/SV change ->
new hash; YAML round trip), ``mtk.fit`` with the fit-time mirror check
recorded, the generic output modules and report (fan omitted with a
stated reason when an exogenous series has no forecast rule), the G6
identity, ``mtk validate <spec.yaml> --tier fast``, and ``mtk.sweep``
over an authored prior.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from macrotoolkit import api as mtk
from macrotoolkit import authoring as au
from macrotoolkit.data import load_data
from macrotoolkit.render import get_cmdstan_version, render_stan_source, stan_source_hash
from macrotoolkit.run import build_render_context, compute_run_id
from specs.schema import get_family
from specs.schema.base import RunSpec, load_spec

pytestmark = pytest.mark.authored


def trend_cycle_model(sv_cycle: bool = True, exogenous: bool = False, forecast_rule: bool = False, cycle_sd: float = 0.5) -> au.Model:
    """A level + AR(2) cycle model (a UC output-gap skeleton) with SV on the
    cycle shock; optionally an exogenous series r entering lagged."""
    meas = "y = lvl + c + e" + (" + b_r*r[-1]" if exogenous else "")
    params = {"a1": au.normal(1.0, 0.3), "a2": au.normal(-0.3, 0.2), "s_lvl": au.half_normal(0.3), "s_e": au.half_normal(0.5)}
    shocks = {"eta_lvl": au.shock("s_lvl"), "e": au.shock("s_e")}
    if sv_cycle:
        shocks["eta_c"] = au.sv(sigma_h=0.2, mu_h0=au.log_var_diff("y", 0.5))
    else:
        params["s_c"] = au.half_normal(cycle_sd)
        shocks["eta_c"] = au.shock("s_c")
    if exogenous:
        params["b_r"] = au.normal(0.0, 0.5)
    return au.Model(
        "trend_cycle_toy",
        observables=["y"],
        exogenous=["r"] if exogenous else None,
        measurement=[meas],
        transition=["lvl = lvl[-1] + eta_lvl", "c = a1*c[-1] + a2*c[-2] + eta_c"],
        parameters=params,
        shocks=shocks,
        initial_state={"lvl": au.init(au.first_obs("y"), 2.0), "c": au.init(0.0, 1.0)},
        forecast_rules={"r": au.forecast("last_value")} if (exogenous and forecast_rule) else None,
    )


def simulated_frame(T: int = 70, seed: int = 20260910, with_r: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lvl, c1, c2 = 100.0, 0.0, 0.0
    y = np.empty(T)
    r = np.empty(T)
    h = 2 * np.log(0.5)
    for t in range(T):
        lvl += rng.normal(0.0, 0.3)
        h += 0.1 * rng.standard_normal()
        c = 1.0 * c1 - 0.3 * c2 + rng.normal(0.0, np.exp(h / 2))
        c2, c1 = c1, c
        r[t] = 2.0 + 0.5 * rng.standard_normal()
        y[t] = lvl + c + rng.normal(0.0, 0.4) + (0.2 * r[t - 1] if with_r and t > 0 else 0.0)
    out = pd.DataFrame({"date": pd.date_range("1995-01-01", periods=T, freq="QS"), "gdp": y})
    if with_r:
        out["rate"] = r
    return out


def _spec(model: au.Model, mapping: dict | None = None, **kw) -> RunSpec:
    return mtk.spec(
        "authored",
        data={"file": "dataframe.csv", "date_column": "date", "mapping": mapping or {"y": "gdp"}},
        options=model,
        **kw,
    )


def _identity(spec: RunSpec, df: pd.DataFrame, tmp: Path) -> str:
    staged = mtk.stage_dataframe(df, spec, tmp)
    _, raw_hash, _ = load_data(staged, base_dir=tmp)
    source = render_stan_source(get_family("authored").template, build_render_context(staged))
    return compute_run_id(staged, raw_hash, stan_source_hash(source), get_cmdstan_version())[1]


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------


def test_equations_enter_the_run_identity(tmp_path: Path) -> None:
    df = simulated_frame()
    base = _spec(trend_cycle_model())
    h0 = _identity(base, df, tmp_path / "a")
    # Same equations, sloppier formatting: same canonical form, same hash.
    loose = base.model.options.model_dump(mode="json")
    loose["equations"]["transition"] = ["lvl=lvl[-1]+eta_lvl", "c = a1 * c[-1] + a2*c[-2]   + eta_c"]
    assert _identity(mtk.spec("authored", data=base.data, options=loose), df, tmp_path / "b") == h0
    # An equation change, a prior change, an SV flag change: new hashes, all distinct.
    m_eq = trend_cycle_model()
    eq = m_eq.to_dict()
    eq["equations"]["transition"][1] = "c = a1*c[-1] + eta_c"
    eq["parameters"].pop("a2")
    h_eq = _identity(mtk.spec("authored", data=base.data, options=eq), df, tmp_path / "c")
    h_prior = _identity(_spec(trend_cycle_model(), priors={"a1": {"sd": 0.5}}), df, tmp_path / "d")
    h_sv = _identity(_spec(trend_cycle_model(sv_cycle=False)), df, tmp_path / "e")
    assert len({h0, h_eq, h_prior, h_sv}) == 4
    # Term order is meaning (regressor/feedback order), so it changes the identity too.
    reordered = base.model.options.model_dump(mode="json")
    reordered["equations"]["measurement"] = ["y = c + lvl + e"]
    assert _identity(mtk.spec("authored", data=base.data, options=reordered), df, tmp_path / "f") != h0


def test_authored_spec_round_trips_through_yaml(tmp_path: Path) -> None:
    spec = _spec(trend_cycle_model(exogenous=True, forecast_rule=True), mapping={"y": "gdp", "r": "rate"}, priors={"a1": {"sd": 0.4}})
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))
    again = load_spec(str(path))
    assert again.to_estimation_yaml() == spec.to_estimation_yaml()
    assert again.model.options.structure.state_labels == spec.model.options.structure.state_labels
    # The required data mapping is the model's own observables + exogenous series.
    with pytest.raises(ValueError, match="missing required key"):
        _spec(trend_cycle_model(exogenous=True), mapping={"y": "gdp"})
    with pytest.raises(ValueError, match="model.options"):
        mtk.spec("authored", data=spec.data, options={**spec.model.options.model_dump(mode="json"), "equations": {"measurement": ["y = lvl*c + e"], "transition": ["lvl = lvl[-1] + eta_lvl"]}})


# ---------------------------------------------------------------------------
# A real fit through the API: mirror check, outputs, report, G6
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def authored_run(tmp_path_factory) -> mtk.Run:
    spec = _spec(
        trend_cycle_model(),
        sampler={"chains": 2, "warmup": 100, "sampling": 100, "adapt_delta": 0.9, "seed": 20260813},
        outputs={"smoother_draws": {"thin": 4}, "prior_predictive_draws": 15, "horizon": 6, "irf_horizon": 8},
    )
    return mtk.fit(spec, simulated_frame(), runs_root=tmp_path_factory.mktemp("authored_runs"))


def test_fit_records_the_mirror_check_and_saves_kf_loglik(authored_run: mtk.Run) -> None:
    mc = authored_run.mirror_check
    assert mc is not None and mc["passed"] and mc["n_points"] == 5 and mc["max_abs_diff"] < 1e-8
    assert "kf_loglik" in authored_run.idata.posterior.data_vars
    assert authored_run.family == "authored" and authored_run.spec.model.options.name == "trend_cycle_toy"
    assert {"a1", "a2", "s_lvl", "s_e", "sigma_h_eta_c", "h0_eta_c_raw"} <= set(authored_run.param_table().index)


def test_outputs_report_and_g6_identity(authored_run: mtk.Run) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    res = authored_run.results()
    outs = authored_run.outputs()
    assert outs.names == ("prior_predictive", "states", "irf", "fevd", "fan", "hd")
    assert all(outs.unavailable_reason(n) is None for n in outs.names)  # no exogenous series: fan applies
    sd = outs.compute("states")
    assert set(sd.states) == {"lvl", "c"} and set(sd.vol) == {"eta_c"} and sd.labels == {"lvl": "lvl", "c": "c"}
    hdd = outs.compute("hd")
    assert hdd.bars == ("init", "eta_lvl", "eta_c", "e")
    total = sum(hdd.obs["y"][b] for b in hdd.bars)
    assert np.max(np.abs(total - res.yobs[:, 0][None, :])) < 1e-6  # G6
    np.testing.assert_allclose(hdd.states["c"]["init"] + hdd.states["c"]["eta_lvl"] + hdd.states["c"]["eta_c"], sd.states["c"], atol=1e-8)
    irf = outs.compute("irf")
    assert irf.shocks == ("eta_lvl", "eta_c", "e") and irf.targets == ("y", "lvl", "c")
    for j in range(irf.responses["eta_lvl"]["y"].shape[0]):
        np.testing.assert_allclose(irf.responses["eta_lvl"]["y"][j], irf.responses["eta_lvl"]["y"][j][0], atol=1e-12)  # level shock: permanent
        assert np.all(irf.responses["e"]["y"][j][1:] == 0.0)
    fans = outs.compute("fan")
    assert fans.obs["y"].shape == fans.states["c"].shape == (len(sd.draw_indices), 6)
    ppd = outs.compute("prior_predictive")
    assert ppd.obs["y"].shape == (15, 70) and np.all(np.isfinite(ppd.obs["y"]))
    for name in outs.names:
        f = outs.figure(name)
        assert isinstance(f, Figure) or all(isinstance(v, Figure) for v in f.values())
    plt.close("all")
    html = authored_run.report().read_text()
    assert "Authored model report" in html and "KF mirror check" in html and "PASS:" in html
    fv = outs.compute("fevd")
    np.testing.assert_allclose(sum(fv.shares["y"][s] for s in fv.shocks), 1.0, atol=1e-12)  # shares sum to one (S8 FEVD)
    assert html.count("<img ") == 1 + 1 + 1 + 1 + 3 + 1  # pp, states, irf, fevd, fan (y, lvl, c), hd (y)
    assert "http://" not in html and "https://" not in html


def test_fan_is_omitted_with_a_reason_without_forecast_rules(tmp_path: Path) -> None:
    df = simulated_frame(T=50, seed=7, with_r=True)
    spec = _spec(trend_cycle_model(sv_cycle=False, exogenous=True), mapping={"y": "gdp", "r": "rate"},
                 sampler={"chains": 2, "warmup": 60, "sampling": 60, "seed": 20260813},
                 outputs={"prior_predictive_draws": 5, "horizon": 4, "smoother_draws": {"thin": 10}})
    run = mtk.fit(spec, df, runs_root=tmp_path / "runs")
    assert run.mirror_check["passed"]
    outs = run.outputs()
    reason = outs.unavailable_reason("fan")
    assert reason is not None and "forecast rule" in reason and "['r']" in reason
    with pytest.raises(ValueError, match="forecast rule"):
        outs.compute("fan")
    html = run.report().read_text()
    assert "Omitted:" in html and "forecast rule" in html
    assert "fan" not in outs.figures()
    # HD carries an exog bar and still reconstructs y (G6 with a data-injection bar).
    hdd = outs.compute("hd")
    assert hdd.bars[-1] == "exog"
    assert np.max(np.abs(sum(hdd.obs["y"][b] for b in hdd.bars) - run.results().yobs[:, 0][None, :])) < 1e-6
    # With a forecast rule the same model's fan applies (compute only; identity differs: the rule is in the definition).
    spec2 = _spec(trend_cycle_model(sv_cycle=False, exogenous=True, forecast_rule=True), mapping={"y": "gdp", "r": "rate"},
                  sampler={"chains": 2, "warmup": 60, "sampling": 60, "seed": 20260813}, outputs={"horizon": 4, "smoother_draws": {"thin": 10}})
    run2 = mtk.fit(spec2, df, runs_root=tmp_path / "runs")
    assert run2.hash != run.hash
    fans = run2.outputs().compute("fan")
    assert fans.obs["y"].shape[1] == 4 and np.all(np.isfinite(fans.obs["y"]))


# ---------------------------------------------------------------------------
# mtk validate <spec.yaml> and mtk sweep over an authored prior
# ---------------------------------------------------------------------------


def test_validate_fast_tier_from_a_spec_path_and_from_python(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from macrotoolkit.cli import main

    df = simulated_frame(T=60, seed=11)
    df.to_csv(tmp_path / "gdp.csv", index=False)
    spec = mtk.spec("authored", data={"file": "gdp.csv", "date_column": "date", "mapping": {"y": "gdp"}}, options=trend_cycle_model())
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))
    result = mtk.validate(tmp_path / "spec.yaml", tier="fast", out_root=tmp_path / "validation")
    assert [g.name for g in result.gates] == ["mirror", "hd_identity"]
    assert result.verdict == "PASS", [g.summary for g in result.gates]
    assert result.gates[0].metrics["max_abs_diff"] < 1e-8 and result.gates[0].metrics["n_points"] == 25
    assert result.gates[1].metrics["max_abs_error"] < 1e-6
    assert result.out_dir.name == "authored_trend_cycle_toy" and "family authored:trend_cycle_toy" in result.report_path.read_text()
    # From Python with a DataFrame standing in for the file.
    result2 = mtk.validate(spec, tier="fast", out_root=tmp_path / "validation2", data=df)
    assert result2.verdict == "PASS"
    # The CLI accepts the spec path; a family name still routes to the registered suite.
    cli = CliRunner().invoke(main, ["validate", str(tmp_path / "spec.yaml"), "--tier", "fast", "--out-root", str(tmp_path / "v3")])
    assert cli.exit_code == 0, cli.output
    assert "mirror [fast]: PASS" in cli.output and "hd_identity [fast]: PASS" in cli.output
    with pytest.raises(ValueError, match="pass the spec path"):
        mtk.validate("authored", tier="fast", out_root=tmp_path / "v4")


def test_sweep_over_an_authored_prior(tmp_path: Path) -> None:
    df = simulated_frame(T=50, seed=13)
    base = _spec(trend_cycle_model(sv_cycle=False), sampler={"chains": 2, "warmup": 60, "sampling": 60, "seed": 20260813},
                 outputs={"smoother_draws": {"thin": 10}, "prior_predictive_draws": 3})
    res = mtk.sweep(
        {"name": "authored_s_c_sweep", "cells": [{"label": "base", "priors": {}}, {"label": "tight", "priors": {"s_c": {"sd": 0.1}}}]},
        base_spec=base, data=df, runs_root=tmp_path / "runs", sweeps_root=tmp_path / "sweeps",
    )
    table = res.comparison.table()
    assert set(table["cell"]) == {"base", "tight"} and "s_c" in set(table["parameter"])
    assert res.comparison.swept_params == ["s_c"]
    row = table[(table["parameter"] == "s_c") & (table["cell"] == "tight")].iloc[0]
    assert np.isfinite(row["prior_sd"]) and np.isfinite(row["contraction"])
    assert set(res.comparison.headline["base"]) == {"lvl", "c"}
    assert res.report_path.is_file()
    with pytest.raises(ValueError, match="not a parameter of authored model"):
        mtk.sweep({"name": "bad", "cells": [{"label": "x", "priors": {"nope": {"sd": 1.0}}}]}, base_spec=base, data=df,
                  runs_root=tmp_path / "runs", sweeps_root=tmp_path / "sweeps")


def test_terminal_and_presample_register_seeds_use_natural_lag_rows() -> None:
    """The fan chart's registers: lag k relative to the first forecast
    period is row n_rows - k (observables lags 1.., exogenous lags 2..;
    the exogenous lag-1 value is the rule's resolution)."""
    from macrotoolkit.authoring.results import _terminal_seeds, init_obs_seeds

    m = au.Model(
        "seeds", observables=["y"], exogenous=["r"], measurement=["y = lvl + 0.5*y[-1] + 0.2*y[-2] + b1*r[-1] + b2*r[-2] + b3*r[-3] + e"],
        transition=["lvl = lvl[-1] + eta"],
        parameters={"b1": au.normal(0, 1), "b2": au.normal(0, 1), "b3": au.normal(0, 1), "s": au.half_normal(1), "se": au.half_normal(1)},
        shocks={"eta": au.shock("s"), "e": au.shock("se")}, initial_state={"lvl": au.init(0.0, 1.0)},
        forecast_rules={"r": au.forecast("last_value")},
    ).compiled()
    series = {"y": np.arange(10.0), "r": 100.0 + np.arange(10.0)}
    obs_seeds, exog_seeds = _terminal_seeds(m, series)
    assert obs_seeds == {"y": {1: 9.0, 2: 8.0}}
    assert exog_seeds == {"r": {2: 108.0, 3: 107.0}}
    # Pre-sample seeds for the init HD bar: lag k of the first estimation row (row L=3) is row L - k.
    assert init_obs_seeds(m, series) == {"y": {1: 2.0, 2: 1.0}}


def test_mixed_measurement_noise_pins_rng_order_and_exp_half_h() -> None:
    """All-SV rows reproduce ``RandomWalkLogVarianceNoise`` draw for draw
    (its documented RNG order: all h innovations first, then eps in row
    order; sd = exp(h/2)); a constant row draws N(0, sd) in row order."""
    from macrotoolkit.authoring.results import MixedMeasurementNoise
    from macrotoolkit.engine import RandomWalkLogVarianceNoise

    ref = RandomWalkLogVarianceNoise((-1.0, -2.0), (0.2, 0.3))
    mine = MixedMeasurementNoise({}, {0: -1.0, 1: -2.0}, {0: 0.2, 1: 0.3}, 2)
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    for _ in range(20):
        np.testing.assert_array_equal(mine.step(r2), ref.step(r1))
    mixed = MixedMeasurementNoise({1: 0.7}, {0: -1.0}, {0: 0.2}, 2)
    rng = np.random.default_rng(9)
    eps = mixed.step(rng)
    check = np.random.default_rng(9)
    h = -1.0 + 0.2 * check.standard_normal()
    e0 = check.normal(0.0, np.sqrt(np.exp(h)))
    e1 = check.normal(0.0, 0.7)
    np.testing.assert_array_equal(eps, [e0, e1])


def test_design_constructors_simulate_models_with_exogenous_blocks_and_measurement_sv() -> None:
    """The recovery/SBC constructors' generic simulator handles an
    exogenous block (supplied paths, pre-sample rows included) and SV on a
    MEASUREMENT shock (the R_t side through the mixed noise model), and
    its Stan data block is the production program's."""
    from macrotoolkit.authoring.validation import recovery_design, sbc_design, simulate_dataset

    m = au.Model(
        "exog_meas_sv", observables=["y"], exogenous=["r"],
        measurement=["y = lvl + 0.3*y[-1] + b1*r[-1] + b2*r[-2] + e"], transition=["lvl = lvl[-1] + eta"],
        parameters={"b1": au.normal(0.2, 0.1), "b2": au.normal(0.0, 0.1), "s": au.half_normal(0.3)},
        shocks={"eta": au.shock("s"), "e": au.sv(sigma_h=0.2, mu_h0=-1.0)}, initial_state={"lvl": au.init(0.0, 1.0)},
    )
    spec = mtk.spec("authored", options=m, data={"file": "x.csv", "date_column": "date", "mapping": {"y": "y", "r": "r"}})
    T, L = 30, m.compiled().lag_depth
    assert L == 2
    r_path = 1.0 + 0.1 * np.arange(L + T)
    design = recovery_design(spec, name="toy", T=T, xi00=[0.0], P00=[[1.0]], n_datasets=2, seed_base=1, anchors={"mu_h0_e": -1.0}, exog_paths={"r": r_path})
    rng = np.random.default_rng(3)
    truth = design.draw_truth(rng)
    assert {"b1", "b2", "s", "sigma_h_e", "h0_e"} <= set(truth)
    dataset = design.simulate(truth, rng)
    assert set(dataset) == {"y", "r"} and dataset["y"].shape == (T,) and np.all(np.isfinite(dataset["y"]))
    np.testing.assert_array_equal(dataset["r"], r_path)
    data = design.stan_data(dataset)
    assert data["T"] == T and data["x"].shape == (T, 3) and data["mu_h0_e"] == -1.0
    # x's exogenous columns are the lagged supplied path; the observable lag column is zero-seeded then simulated.
    np.testing.assert_array_equal(data["x"][:, 1], r_path[L - 1 : L - 1 + T])
    np.testing.assert_array_equal(data["x"][:, 2], r_path[L - 2 : L - 2 + T])
    assert data["x"][0, 0] == 0.0 and data["x"][1, 0] == dataset["y"][0]
    with pytest.raises(ValueError, match="length L \\+ T"):
        simulate_dataset(m.compiled(), truth, rng, T, xi00=np.array([0.0]), P00=np.array([[1.0]]))
    sbc = sbc_design(spec, name="toy_sbc", T=T, xi00=[0.0], P00=[[1.0]], n_replications=2, rank_draws=9, rank_bins=3, seed_base=1,
                     anchors={"mu_h0_e": -1.0}, exog_paths={"r": r_path})
    assert sbc.param_labels == ("b1", "b2", "s", "sigma_h_e", "h0_e") and sbc.expected_priors == m.compiled().resolve_priors({})
