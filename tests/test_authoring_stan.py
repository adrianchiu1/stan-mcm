"""S7 M2 gate (the stage's G1-equivalent, a HARD gate): the generic
authored template's ``kf_loglik`` equals the HAND template's ``kf_loglik``
at 50 prior draws on the same data -- lw_sv (no-SV and SV: R_t) and ucsv
(no-SV and SV: Q_t AND R_t) -- to 1e-8 (observed ~1e-11), and equals the
Python mirror; local_level (whose hand template has no KF, plans/S7-plan.md
conflict item 1) is gated against the Python mirror. Both programs are
evaluated by the S6 fixed_param mechanism (``qc.stan_kf_loglik_at_points``)
at inits mapped between the two parameterizations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from authored_oracles import local_level_model, lw_sv_model, ucsv_model
from macrotoolkit.authoring.family import python_kf_loglik, stan_inits_and_h
from macrotoolkit.authoring.stan import TEMPLATE_NAME, render_context
from macrotoolkit.qc import MirrorPoint, stan_kf_loglik_at_points
from macrotoolkit.render import compile_model, render_stan_source
from specs.schema.base import RunSpec

pytestmark = pytest.mark.authored

N_POINTS = 50
TOL = 1e-8
SEED = 20260908


def _authored_program(model):
    compiled = model.compiled()
    priors = compiled.resolve_priors({})
    stan_model, _ = compile_model(render_stan_source(TEMPLATE_NAME, render_context(compiled, priors)))
    return compiled, priors, stan_model


def _hand_program(family: str, options: dict, mapping: dict):
    from macrotoolkit.run import build_render_context
    from specs.schema import get_family

    spec = RunSpec.model_validate({"model": {"family": family, "options": options}, "data": {"file": "x.csv", "date_column": "date", "mapping": mapping}})
    stan_model, _ = compile_model(render_stan_source(get_family(family).template, build_render_context(spec)))
    return stan_model


def _evaluate(stan_model, stan_data: dict, inits: list[dict]) -> np.ndarray:
    points = [MirrorPoint(inits=i, loglik_python=float("nan")) for i in inits]
    return stan_kf_loglik_at_points(stan_model, stan_data, points, seed=SEED)


def _lw_df() -> pd.DataFrame:
    from g1_harness import SYNTHETIC_DATA

    return pd.DataFrame({"date": SYNTHETIC_DATA.dates, "y": SYNTHETIC_DATA.y, "pi": SYNTHETIC_DATA.pi, "r": SYNTHETIC_DATA.r})


def _ucsv_df(T: int = 60, seed: int = 20260909) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tau, pi = 2.0, np.empty(T)
    for t in range(T):
        tau += rng.normal(0.0, 0.3)
        pi[t] = tau + rng.normal(0.0, 0.8)
    return pd.DataFrame({"date": pd.date_range("1990-01-01", periods=T, freq="QS"), "pi": pi})


# ---------------------------------------------------------------------------
# lw_sv
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sv", [False, True], ids=["no_sv", "sv"])
def test_lw_sv_kf_loglik_equals_hand_template_and_python_mirror(sv: bool) -> None:
    from macrotoolkit.families.lw_sv import build_stan_data as hand_stan_data

    df = _lw_df()
    hand = _hand_program("lw_sv", {"sv_shocks": ["is", "pc"] if sv else []}, {"y": "y", "pi": "pi", "r": "r"})
    hand_data = hand_stan_data(df)
    compiled, priors, authored = _authored_program(lw_sv_model(sv))
    data = compiled.stan_data(df)
    if sv:
        # lw_sv's mu_h0 anchors are its HLW-regression OLS pass (family-specific);
        # supply the hand values as data so both programs see the same h_0.
        data["mu_h0_e_is"], data["mu_h0_e_pc"] = hand_data["mu_h0_is"], hand_data["mu_h0_pc"]
    for key in ("T", "yobs", "x", "xi00", "P00"):
        np.testing.assert_array_equal(np.asarray(data[key]), np.asarray(hand_data[key]))
    anchors = {k: v for k, v in data.items() if k.startswith("mu_h0_")}
    rng = np.random.default_rng(SEED)
    inits_a, inits_h, py = [], [], []
    for _ in range(N_POINTS):
        params = compiled.sample_prior_params(priors, rng, anchors)
        ia, h = stan_inits_and_h(compiled, params, priors, anchors, rng, data["T"])
        ih = {k: ia[k] for k in ("a1", "a2", "a_r", "b_pi", "b_y", "sigma_ystar", "sigma_g", "sigma_z")}
        if sv:
            for mine, hand_name in (("e_is", "is"), ("e_pc", "pc")):
                ih[f"sigma_h_{hand_name}"] = ia[f"sigma_h_{mine}"]
                ih[f"h0_{hand_name}_raw"] = ia[f"h0_{mine}_raw"]
                ih[f"nu_{hand_name}"] = ia[f"nu_{mine}"]
        else:
            ih["sigma_is"], ih["sigma_pc"] = ia["sigma_is"], ia["sigma_pc"]
        inits_a.append(ia)
        inits_h.append(ih)
        py.append(python_kf_loglik(compiled, params, data, h))
    ll_authored = _evaluate(authored, data, inits_a)
    ll_hand = _evaluate(hand, hand_data, inits_h)
    py = np.asarray(py)
    assert np.all(np.isfinite(ll_authored)) and np.all(np.isfinite(ll_hand))
    d_hand = float(np.max(np.abs(ll_authored - ll_hand)))
    d_py = float(np.max(np.abs(ll_authored - py)))
    assert d_hand < TOL, f"authored vs hand lw_sv template: max |diff| = {d_hand:.3e}"
    assert d_py < TOL, f"authored Stan vs Python mirror: max |diff| = {d_py:.3e}"


# ---------------------------------------------------------------------------
# ucsv (SV on a STATE shock: the Q_t side; and on the measurement shock: R_t)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sv", [False, True], ids=["no_sv", "sv"])
def test_ucsv_kf_loglik_equals_hand_template_and_python_mirror(sv: bool) -> None:
    from macrotoolkit.families.ucsv import build_stan_data as hand_stan_data

    df = _ucsv_df()
    hand = _hand_program("ucsv", {"sv_shocks": ["eps", "eta"] if sv else []}, {"pi": "pi"})
    hand_data = hand_stan_data(df)
    compiled, priors, authored = _authored_program(ucsv_model(sv))
    data = compiled.stan_data(df)
    for key in ("T", "yobs", "xi00", "P00"):
        np.testing.assert_array_equal(np.asarray(data[key]), np.asarray(hand_data[key]))
    if sv:
        assert data["mu_h0_eps"] == hand_data["mu_h0_eps"] and data["mu_h0_eta"] == hand_data["mu_h0_eta"]
    anchors = {k: v for k, v in data.items() if k.startswith("mu_h0_")}
    rng = np.random.default_rng(SEED)
    inits, py = [], []
    for _ in range(N_POINTS):
        params = compiled.sample_prior_params(priors, rng, anchors)
        ia, h = stan_inits_and_h(compiled, params, priors, anchors, rng, data["T"])
        inits.append(ia)  # the hand ucsv program uses the SAME parameter names
        py.append(python_kf_loglik(compiled, params, data, h))
    ll_authored = _evaluate(authored, data, inits)
    ll_hand = _evaluate(hand, hand_data, inits)
    d_hand = float(np.max(np.abs(ll_authored - ll_hand)))
    d_py = float(np.max(np.abs(ll_authored - np.asarray(py))))
    assert np.all(np.isfinite(ll_authored))
    assert d_hand < TOL, f"authored vs hand ucsv template: max |diff| = {d_hand:.3e}"
    assert d_py < TOL, f"authored Stan vs Python mirror: max |diff| = {d_py:.3e}"


# ---------------------------------------------------------------------------
# local_level: no hand KF exists -> the Python mirror is the oracle
# ---------------------------------------------------------------------------


def test_local_level_kf_loglik_equals_python_mirror() -> None:
    from conftest import _tiny_local_level_rows

    df = _tiny_local_level_rows(n=40).rename(columns={"obs": "y"})
    compiled, priors, authored = _authored_program(local_level_model())
    data = compiled.stan_data(df)
    rng = np.random.default_rng(SEED)
    inits, py = [], []
    for _ in range(N_POINTS):
        params = compiled.sample_prior_params(priors, rng, {})
        ia, h = stan_inits_and_h(compiled, params, priors, {}, rng, data["T"])
        inits.append(ia)
        py.append(python_kf_loglik(compiled, params, data, h))
    ll = _evaluate(authored, data, inits)
    d = float(np.max(np.abs(ll - np.asarray(py))))
    assert np.all(np.isfinite(ll)) and d < TOL, f"authored local level Stan vs Python: {d:.3e}"


# ---------------------------------------------------------------------------
# A parameter-dependent F with SV on the STATE shock renders and compiles
# (matrices land in transformed parameters; Q_t path)
# ---------------------------------------------------------------------------


def test_ar_state_with_state_sv_renders_compiles_and_mirrors() -> None:
    from macrotoolkit import authoring as au

    m = au.Model(
        "ar_cycle_sv",
        observables=["y"],
        measurement=["y = lvl + c + e"],
        transition=["lvl = lvl[-1] + eta_lvl", "c = a1*c[-1] + a2*c[-2] + eta_c"],
        parameters={"a1": au.normal(1.0, 0.3), "a2": au.normal(-0.3, 0.3), "s_lvl": au.half_normal(0.5), "s_e": au.half_normal(0.5)},
        shocks={"eta_lvl": au.shock("s_lvl"), "e": au.shock("s_e"), "eta_c": au.sv(sigma_h=0.2, mu_h0=au.log_var_diff("y", 0.5))},
        initial_state={"lvl": au.init(au.first_obs("y"), 2.0), "c": au.init(0.0, 1.0)},
    )
    compiled, priors, authored = _authored_program(m)
    source = render_stan_source(TEMPLATE_NAME, render_context(compiled, priors))
    assert "array[T] matrix[3, 3] Qt" in source and "matrix[3, 3] F = rep_matrix" in source.split("transformed parameters")[1]
    assert "F[2, 2] = a1;" in source and "F[2, 3] = a2;" in source and "F[3, 2] = 1.0;" in source
    df = _ucsv_df().rename(columns={"pi": "y"})
    data = compiled.stan_data(df)
    anchors = {k: v for k, v in data.items() if k.startswith("mu_h0_")}
    rng = np.random.default_rng(SEED)
    inits, py = [], []
    for _ in range(10):
        params = compiled.sample_prior_params(priors, rng, anchors)
        ia, h = stan_inits_and_h(compiled, params, priors, anchors, rng, data["T"])
        inits.append(ia)
        py.append(python_kf_loglik(compiled, params, data, h))
    ll = _evaluate(authored, data, inits)
    assert float(np.max(np.abs(ll - np.asarray(py)))) < TOL
