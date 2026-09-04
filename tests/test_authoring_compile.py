"""S7 M1 gate: the equation compiler reproduces the hand-written
families' declaration surface EXACTLY -- ``StateSpaceMeta`` (structure),
the system matrices at 50 prior draws (near machine precision), the
regressors/initial state/anchors -- for local_level, ucsv and lw_sv
(``tests/authored_oracles.py``), plus the scope fence and canonicalization.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from authored_oracles import LW_SHOCK_MAP, local_level_model, lw_sv_model, ucsv_model
from macrotoolkit import authoring as au
from macrotoolkit.families.base import StateSpaceMeta
from specs.schema.authored import AuthoredOptions
from specs.schema.equations import canonical_equation

pytestmark = pytest.mark.authored

N_POINTS = 50
SEED = 20260907


def _renamed(meta: StateSpaceMeta, shock_map: dict[str, str]) -> StateSpaceMeta:
    """The compiled meta with its shock labels mapped to the hand family's
    (plans/S7-plan.md conflict item 2: labels are user choices, everything
    else is structure and compared exactly)."""
    return StateSpaceMeta(
        state_labels=meta.state_labels,
        state_shocks=tuple(shock_map[s] for s in meta.state_shocks),
        shock_loadings={shock_map[s]: dict(v) for s, v in meta.shock_loadings.items()},
        measurement_shocks=tuple(shock_map[s] for s in meta.measurement_shocks),
        obs_names=meta.obs_names,
        exog_names=meta.exog_names,
        feedback_map=meta.feedback_map,
    )


# ---------------------------------------------------------------------------
# lw_sv
# ---------------------------------------------------------------------------


def test_lw_sv_meta_is_reproduced_exactly() -> None:
    from macrotoolkit.families.lw_sv import LW_STATE_META

    compiled = lw_sv_model(sv=False).compiled()
    assert _renamed(compiled.meta, LW_SHOCK_MAP) == LW_STATE_META
    assert compiled.lag_depth == 4
    # The SV variant has the same structure (SV changes R_t, not the layout).
    assert _renamed(lw_sv_model(sv=True).compiled().meta, LW_SHOCK_MAP) == LW_STATE_META


def test_lw_sv_matrices_match_hand_builder_at_prior_points() -> None:
    from g1_harness import generate_parameter_points
    from macrotoolkit.smoother import build_lw_matrices

    compiled = lw_sv_model(sv=False).compiled()
    worst = 0.0
    for p in generate_parameter_points(N_POINTS, SEED):
        F, Q, A, Z, R = compiled.build_matrices(p)
        Fh, Qh, Ah, Zh, Rh = build_lw_matrices(p, c=1.0)
        for mine, hand in ((F, Fh), (Q, Qh), (A, Ah), (Z, Zh), (R, Rh)):
            assert mine.shape == hand.shape
            worst = max(worst, float(np.max(np.abs(mine - hand))))
        np.testing.assert_array_equal(F, Fh)  # parameter-free: bit-identical
    assert worst < 1e-15, worst


def test_lw_sv_sv_variant_R_path_matches_hand_composition() -> None:
    from g1_harness import generate_parameter_points, generate_sv_inputs
    from macrotoolkit.smoother import build_lw_matrices, sv_diag_variance_path, sv_rw_noncentered

    compiled = lw_sv_model(sv=True).compiled()
    assert compiled.sv_shocks == ("e_is", "e_pc")
    T = 40
    points = generate_parameter_points(5, SEED)
    sv = generate_sv_inputs(points, T)
    for i, p in enumerate(points):
        h_is = sv_rw_noncentered(float(sv["h0_is"][i]), float(sv["sigma_h_is"][i]), sv["nu"][i, :, 0])
        h_pc = sv_rw_noncentered(float(sv["h0_pc"][i]), float(sv["sigma_h_pc"][i]), sv["nu"][i, :, 1])
        F, Q, A, Z, R = compiled.build_matrices(p, h={"e_is": h_is, "e_pc": h_pc})
        Fh, Qh, Ah, Zh, _ = build_lw_matrices({**p, "sigma_is": 1.0, "sigma_pc": 1.0}, c=1.0)
        np.testing.assert_array_equal(R, sv_diag_variance_path(h_is, h_pc))
        assert Q.ndim == 2 and np.max(np.abs(Q - Qh)) < 1e-15


def test_lw_sv_regressors_and_initial_state_match_hand_builders() -> None:
    from g1_harness import SYNTHETIC_DATA
    from macrotoolkit.smoother import build_lw_regressors, default_initial_state

    compiled = lw_sv_model(sv=False).compiled()
    series = {"y": SYNTHETIC_DATA.y, "pi": SYNTHETIC_DATA.pi, "r": SYNTHETIC_DATA.r}
    yobs, x = compiled.regressors(series)
    yobs_h, x_h = build_lw_regressors(SYNTHETIC_DATA.y, SYNTHETIC_DATA.pi, SYNTHETIC_DATA.r)
    np.testing.assert_array_equal(yobs, yobs_h)
    np.testing.assert_array_equal(x, x_h)
    xi00, P00 = compiled.initial_state(series)
    xi00_h, P00_h = default_initial_state(float(SYNTHETIC_DATA.y[4]))
    np.testing.assert_array_equal(xi00, xi00_h)
    np.testing.assert_array_equal(P00, P00_h)
    df = pd.DataFrame({"date": SYNTHETIC_DATA.dates, **series})
    data = compiled.stan_data(df)
    assert data["T"] == SYNTHETIC_DATA.T - 4 and data["x"].shape == (SYNTHETIC_DATA.T - 4, 6)


# ---------------------------------------------------------------------------
# ucsv
# ---------------------------------------------------------------------------


def test_ucsv_meta_matrices_anchors_match_hand_family() -> None:
    from macrotoolkit.families.ucsv import (
        UCSV_STATE_META,
        build_ucsv_matrices,
        build_ucsv_regressors,
        default_initial_state,
        ucsv_mu_h0_anchors,
    )
    from macrotoolkit.smoother import sv_rw_noncentered, sv_scalar_variance_path

    no_sv = ucsv_model(sv=False).compiled()
    with_sv = ucsv_model(sv=True).compiled()
    assert no_sv.meta == UCSV_STATE_META and with_sv.meta == UCSV_STATE_META
    assert no_sv.lag_depth == 0 and with_sv.sv_shocks == ("eps", "eta")
    rng = np.random.default_rng(SEED)
    for _ in range(N_POINTS):
        p = {"sigma_eps": abs(rng.normal()), "sigma_eta": abs(rng.normal())}
        for mine, hand in zip(no_sv.build_matrices(p), build_ucsv_matrices(p)):
            np.testing.assert_array_equal(mine, hand)
    T = 30
    h_eps, h_eta = sv_rw_noncentered(-1.0, 0.2, rng.standard_normal(T)), sv_rw_noncentered(-2.0, 0.3, rng.standard_normal(T))
    F, Q, A, Z, R = with_sv.build_matrices({}, h={"eps": h_eps, "eta": h_eta})
    np.testing.assert_array_equal(Q, sv_scalar_variance_path(h_eta))
    np.testing.assert_array_equal(R, sv_scalar_variance_path(h_eps))
    pi = np.array([1.0, 2.0, 4.0, 3.0, 5.0, 4.5])
    series = {"pi": pi}
    yobs, x = with_sv.regressors(series)
    yobs_h, x_h = build_ucsv_regressors(pi)
    np.testing.assert_array_equal(yobs, yobs_h)
    assert x.shape == x_h.shape == (6, 0)
    xi00, P00 = with_sv.initial_state(series)
    xi00_h, P00_h = default_initial_state(float(pi[0]))
    np.testing.assert_array_equal(xi00, xi00_h)
    np.testing.assert_array_equal(P00, P00_h)
    mu_eps, mu_eta = ucsv_mu_h0_anchors(pi)
    anchors = with_sv.anchors(series)
    assert anchors["mu_h0_eps"] == mu_eps and anchors["mu_h0_eta"] == mu_eta
    assert set(with_sv.prior_table) == {"sigma_h_eps", "sigma_h_eta", "mu_h0_eps", "mu_h0_eta"}


# ---------------------------------------------------------------------------
# local_level (no hand meta/builder exists: the oracle is hand-written here)
# ---------------------------------------------------------------------------


def test_local_level_meta_and_matrices() -> None:
    compiled = local_level_model().compiled()
    assert compiled.meta == StateSpaceMeta(
        state_labels=(("mu", 0),),
        state_shocks=("eta",),
        shock_loadings={"eta": {("mu", 0): 1.0}},
        measurement_shocks=("eps",),
        obs_names=("y",),
    )
    F, Q, A, Z, R = compiled.build_matrices({"sigma_obs": 0.4, "sigma_level": 0.3})
    np.testing.assert_array_equal(F, [[1.0]])
    np.testing.assert_array_equal(Q, [[0.3 * 0.3]])
    np.testing.assert_array_equal(Z, [[1.0]])
    np.testing.assert_array_equal(R, [[0.4 * 0.4]])
    assert A.shape == (0, 1)
    y = np.array([1.0, 1.5, 0.5])
    xi00, P00 = compiled.initial_state({"y": y})
    np.testing.assert_array_equal(xi00, [1.0])
    np.testing.assert_array_equal(P00, [[100.0]])


# ---------------------------------------------------------------------------
# Priors, sampler, stationarity
# ---------------------------------------------------------------------------


def test_prior_resolution_and_sampler_follow_the_family_discipline() -> None:
    compiled = lw_sv_model(sv=True).compiled()
    table = compiled.prior_table
    assert {"sigma_h_e_is", "mu_h0_e_is", "sigma_h_e_pc", "mu_h0_e_pc"} <= set(table)
    resolved = compiled.resolve_priors({"a1": {"sd": 0.5}, "sigma_h_e_is": {"sd": 0.1}})
    assert resolved["a1"] == {"dist": "normal", "mu": 1.2, "sd": 0.5} and resolved["sigma_h_e_is"]["sd"] == 0.1
    with pytest.raises(ValueError, match="not a parameter of authored model"):
        compiled.resolve_priors({"sigma_is": {"sd": 1.0}})  # no-SV-only name does not exist in this model
    with pytest.raises(ValueError, match="unknown field"):
        compiled.resolve_priors({"a1": {"scale": 0.5}})
    with pytest.raises(ValueError, match="structure"):
        compiled.resolve_priors({"a1": {"dist": "beta"}})
    rng = np.random.default_rng(3)
    anchors = {"mu_h0_e_is": -1.0, "mu_h0_e_pc": -2.0}
    draws = [compiled.sample_prior_params(resolved, rng, anchors) for _ in range(2000)]
    assert set(draws[0]) == set(compiled.param_names) | {"h0_e_is", "h0_e_pc"}
    assert all(d["a_r"] < 0 and d["b_y"] > 0 and 0 < d["b_pi"] < 1 and d["sigma_g"] >= 0 for d in draws)
    h0 = np.array([d["h0_e_pc"] for d in draws])
    assert abs(h0.mean() + 2.0) < 0.1 and abs(h0.std() - 1.0) < 0.1
    with pytest.raises(ValueError, match="anchor"):
        compiled.sample_prior_params(resolved, rng)


def test_stationarity_filter_uses_F_roots_and_the_feedback_companion() -> None:
    lw = lw_sv_model(sv=False).compiled()
    base = {"a1": 1.2, "a2": -0.4, "a_r": -0.1, "b_pi": 0.8, "b_y": 0.15, "sigma_ystar": 0.4, "sigma_g": 0.03, "sigma_z": 0.08, "sigma_is": 1.0, "sigma_pc": 1.0}
    assert lw.is_stationary(base)  # RW roots in F and the Phillips curve's unit root are allowed; the gap AR(2) is stationary
    assert not lw.is_stationary({**base, "a1": 1.5, "a2": -0.4})  # a1 + a2 >= 1: explosive feedback
    # A state-side AR(2) cycle: explosive roots in F are rejected.
    m = au.Model(
        "ar_state", observables=["y"], measurement=["y = c + e"], transition=["c = a1*c[-1] + a2*c[-2] + eta"],
        parameters={"a1": au.normal(1.0, 0.3), "a2": au.normal(-0.3, 0.3), "s": au.half_normal(1.0), "se": au.half_normal(1.0)},
        shocks={"eta": au.shock("s"), "e": au.shock("se")}, initial_state={"c": au.init(0.0, 1.0)},
    ).compiled()
    assert m.meta.state_labels == (("c", 0), ("c", -1))
    assert m.is_stationary({"a1": 1.0, "a2": -0.3, "s": 1.0, "se": 1.0})
    assert not m.is_stationary({"a1": 1.5, "a2": 0.2, "s": 1.0, "se": 1.0})


# ---------------------------------------------------------------------------
# Canonical form
# ---------------------------------------------------------------------------


def test_canonical_form_normalizes_formatting_but_keeps_term_order() -> None:
    a = canonical_equation("y=ystar+a1 * ( y[-1]-ystar[-1] )+e")
    b = canonical_equation("y = ystar + a1*(y[-1] - ystar[-1]) + e")
    assert a == b == "y = ystar + a1*(y[-1] - ystar[-1]) + e"
    assert canonical_equation("y = e + ystar") != canonical_equation("y = ystar + e")
    opts = lw_sv_model(sv=False).to_dict()
    loose = dict(opts)
    loose["equations"] = {"measurement": [" y=ystar+a1*(y[-1]-ystar[-1])+a2*(y[-2]-ystar[-2])+(a_r/2)*(r[-1]-g[-1]-z[-1])+(a_r/2)*(r[-2]-g[-2]-z[-2])+e_is",
                                          opts["equations"]["measurement"][1]],
                          "transition": opts["equations"]["transition"]}
    assert AuthoredOptions.model_validate(loose).model_dump() == AuthoredOptions.model_validate(opts).model_dump()


# ---------------------------------------------------------------------------
# The scope fence: every construct outside the DSL fails naming the limitation
# ---------------------------------------------------------------------------


def _ll(**overrides):
    base = local_level_model().to_dict()
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"equations": {"measurement": ["y = mu*mu + eps"], "transition": ["mu = mu[-1] + eta"]}}, "LINEAR"),
        ({"equations": {"measurement": ["y = mu + 1 + eps"], "transition": ["mu = mu[-1] + eta"]}}, "intercept"),
        ({"equations": {"measurement": ["y = mu + eps"], "transition": ["mu = mu[-1] + 0.1 + eta"]}}, "drift"),
        ({"equations": {"measurement": ["y = mu + eps"], "transition": ["mu = mu[-1] + y[-1] + eta"]}}, "transition equations must be linear in states"),
        ({"equations": {"measurement": ["y = mu + eps + eta"], "transition": ["mu = mu[-1] + eta"]}}, "transition equation AND a measurement equation"),
        ({"equations": {"measurement": ["y = mu"], "transition": ["mu = mu[-1] + eta + eps"]}}, "exactly one measurement shock"),
        ({"equations": {"measurement": ["y = mu + 2*eps"], "transition": ["mu = mu[-1] + eta"]}}, "unit coefficient"),
        ({"equations": {"measurement": ["y = mu + eps"], "transition": ["mu = mu[-1] + eta[-1]"]}}, "MA term"),
        ({"equations": {"measurement": ["y = mu + eps"], "transition": ["mu = mu[-1] + eta + eta"]}}, "once per equation"),
        ({"equations": {"measurement": ["y = mu + eps"], "transition": ["mu = sigma_level*mu[-1] + eta"]}}, "scale a shock AND appear as a coefficient"),
        ({"equations": {"measurement": ["y = mu + eps"], "transition": ["mu = mu[-1] + sigma_obs*eta"]}}, "numeric constants"),
        ({"equations": {"measurement": ["y = mu[+1] + eps"], "transition": ["mu = mu[-1] + eta"]}}, "LEAD"),
        ({"equations": {"measurement": ["y = mu + eps"], "transition": ["mu[-1] = mu[-2] + eta"]}}, "carried lagged"),
        ({"equations": {"measurement": ["y = mu + eps"], "transition": ["mu = nu[-1] + eta"]}}, "Unknown name"),
        ({"equations": {"measurement": ["y = mu + eps"], "transition": ["mu = mu[-1] + eta", "mu = eta"]}}, "declared twice"),
        ({"parameters": {"sigma_obs": {"dist": "half_normal", "sd": 1.0}, "sigma_level": {"dist": "half_normal", "sd": 1.0}, "unused": {"dist": "normal", "mu": 0, "sd": 1}}}, "appear in no equation"),
        ({"initial_state": {}}, "initial_state"),
        ({"parameters": {"sigma_obs": {"dist": "half_normal", "sd": 1.0}}, "shocks": {"eps": {"sd": "sigma_obs"}, "eta": {"sd": "nope"}}}, "not a declared parameter"),
        ({"parameters": {"sigma_obs": {"dist": "half_normal", "sd": 1.0}, "sigma_level": {"dist": "normal", "mu": 0, "sd": 1}}}, "must be a half_normal parameter"),
        ({"forecast_rules": {"y": {"rule": "last_value"}}}, "not an exogenous series"),
    ],
)
def test_scope_fence_rejects_with_a_message_naming_the_limitation(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        AuthoredOptions.model_validate(_ll(**overrides))


def test_contemporaneous_observable_and_exogenous_are_rejected() -> None:
    base = local_level_model().to_dict()
    base["observables"] = ["y", "w"]
    base["equations"] = {"measurement": ["y = mu + w + eps", "w = mu + e2"], "transition": ["mu = mu[-1] + eta"]}
    base["shocks"]["e2"] = {"sd": "sigma_obs"}
    with pytest.raises(ValueError, match="simultaneous observables"):
        AuthoredOptions.model_validate(base)
    base = local_level_model().to_dict()
    base["exogenous"] = ["r"]
    base["equations"] = {"measurement": ["y = mu + b*r + eps"], "transition": ["mu = mu[-1] + eta"]}
    base["parameters"]["b"] = {"dist": "normal", "mu": 0.0, "sd": 1.0}
    with pytest.raises(ValueError, match="strictly lagged"):
        AuthoredOptions.model_validate(base)


def test_simultaneous_state_cycle_and_head_substitution() -> None:
    with pytest.raises(ValueError, match="cycle"):
        au.Model(
            "cyc", observables=["y"], measurement=["y = a + e"], transition=["a = 0.5*b + ea", "b = 0.5*a + eb"],
            parameters={"s": au.half_normal(1.0)}, shocks={"e": au.shock("s"), "ea": au.shock("s"), "eb": au.shock("s")},
            initial_state={"a": au.init(0.0, 1.0), "b": au.init(0.0, 1.0)},
        )
    # A contemporaneous head reference substitutes the other state's equation (F row + loadings).
    m = au.Model(
        "sub", observables=["y"], measurement=["y = lvl + e"], transition=["lvl = lvl[-1] + 0.5*trend + el", "trend = trend[-1] + et"],
        parameters={"s": au.half_normal(1.0)}, shocks={"e": au.shock("s"), "el": au.shock("s"), "et": au.shock("s")},
        initial_state={"lvl": au.init(0.0, 1.0), "trend": au.init(0.0, 1.0)},
    ).compiled()
    F = m.build_F({})
    np.testing.assert_array_equal(F, [[1.0, 0.5], [0.0, 1.0]])
    assert m.meta.shock_loadings == {"el": {("lvl", 0): 1.0}, "et": {("trend", 0): 1.0, ("lvl", 0): 0.5}}


def test_model_helpers_round_trip_through_yaml_and_describe() -> None:
    import yaml

    m = lw_sv_model(sv=True)
    d = m.to_dict()
    again = AuthoredOptions.model_validate(yaml.safe_load(yaml.safe_dump(d)))
    assert again.model_dump() == m.options.model_dump()
    assert "state vector: ystar, ystar[-1], ystar[-2], g[-1], g[-2], z[-1], z[-2]" in m.describe()
    assert m.compiled() is au.compile_model(again)  # cache hit on the canonical definition


def test_coefficient_printer_round_trips_bit_exactly_under_random_nesting() -> None:
    """The Python/Stan bit-identity invariant: evaluating a coefficient
    tree equals re-parsing its emitted text and evaluating THAT (Stan
    re-parses the printed text) -- for random nestings of + - * / ( ) and
    unary minus (the numerics-reviewer must-fix: a same-precedence right
    child must be parenthesized, a*(b/c) != (a*b)/c in floating point)."""
    import random

    from specs.schema.equations import Add, Div, Mul, Name, Neg, Num, Sub, evaluate, parse_equation

    rng = random.Random(20260911)
    params = {"a": 0.1, "b": 0.7, "c": 0.3, "d": -1.9, "e2": 2.5}

    def tree(depth: int):
        if depth == 0 or rng.random() < 0.25:
            return Name(rng.choice(list(params))) if rng.random() < 0.7 else Num(rng.choice([0.25, 2.0, 3.0, 0.1]))
        op = rng.choice([Add, Sub, Mul, Div, Neg])
        if op is Neg:
            return Neg(tree(depth - 1))
        return op(tree(depth - 1), tree(depth - 1))

    for _ in range(3000):
        t = tree(4)
        text = t.emit()
        # Re-parse as a coefficient of a symbol: '(text)*s' then read the coefficient back.
        eq = parse_equation(f"y = ({text})*s")
        from specs.schema.equations import linearize

        lf = linearize(eq.rhs, lambda n: "symbol" if n == "s" else ("param" if n in params else None))
        coef = next(iter(lf.terms.values()))
        try:
            direct = evaluate(t, params)
        except ZeroDivisionError:
            continue
        assert evaluate(coef, params) == direct, (text, coef.emit())
