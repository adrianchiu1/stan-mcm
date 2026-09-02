"""Prior-predictive check (spec §4, S5-decisions item 7):
``sample_prior_params`` respects the template's own distributions and
truncation constraints, and ``compute_prior_predictive_draws`` simulates
full observable paths from the run's own resolved priors through the
generic engine -- overrides included, no posterior draws needed.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from macrotoolkit.families.lw_sv import sample_prior_params
from macrotoolkit.results_lw import compute_prior_predictive_draws
from specs.schema.lw_sv import DEFAULT_PRIORS

_SEED = 20260902


def _resolved_defaults() -> dict:
    return {name: dict(entry) for name, entry in DEFAULT_PRIORS.items()}


# ---------------------------------------------------------------------------
# sample_prior_params
# ---------------------------------------------------------------------------


def test_sampled_params_respect_template_constraints_no_sv() -> None:
    rng = np.random.default_rng(_SEED)
    priors = _resolved_defaults()
    for _ in range(200):
        p = sample_prior_params(priors, sv_on=False, rng=rng)
        assert p["a_r"] < 0.0  # template <upper=0>
        assert p["b_y"] > 0.0  # template <lower=0>
        assert 0.0 < p["b_pi"] < 1.0  # beta support
        for s in ("sigma_ystar", "sigma_g", "sigma_z", "sigma_is", "sigma_pc"):
            assert p[s] >= 0.0  # half-normal support
        assert "sigma_h_is" not in p and "h0_is" not in p


def test_sampled_params_sv_variant_has_sv_parts_and_anchored_h0() -> None:
    rng = np.random.default_rng(_SEED)
    priors = _resolved_defaults()
    anchor_is, anchor_pc = -2.0, 0.5
    draws = [
        sample_prior_params(priors, sv_on=True, rng=rng, mu_h0_is=anchor_is, mu_h0_pc=anchor_pc)
        for _ in range(500)
    ]
    for p in draws:
        assert p["sigma_h_is"] >= 0.0 and p["sigma_h_pc"] >= 0.0
        assert "sigma_is" not in p and "sigma_pc" not in p
    # h0 ~ N(anchor, 1) (spec §1.5) -- the sample mean must sit near the
    # data anchor, not near 0 (a wrong-anchor bug would shift it).
    h0_is = np.array([p["h0_is"] for p in draws])
    h0_pc = np.array([p["h0_pc"] for p in draws])
    assert abs(h0_is.mean() - anchor_is) < 0.15
    assert abs(h0_pc.mean() - anchor_pc) < 0.15


def test_sampled_params_sv_without_anchors_raises() -> None:
    rng = np.random.default_rng(_SEED)
    with pytest.raises(ValueError, match="mu_h0"):
        sample_prior_params(_resolved_defaults(), sv_on=True, rng=rng)


def test_prior_override_shifts_the_sampled_distribution() -> None:
    """The prior-predictive must consume the run's RESOLVED priors: an
    override must move the sampled draws (here: a nearly-degenerate a1)."""
    rng = np.random.default_rng(_SEED)
    priors = _resolved_defaults()
    priors["a1"] = {"dist": "normal", "mu": 0.5, "sd": 1e-8}
    draws = np.array([sample_prior_params(priors, sv_on=False, rng=rng)["a1"] for _ in range(50)])
    np.testing.assert_allclose(draws, 0.5, atol=1e-6)


# ---------------------------------------------------------------------------
# compute_prior_predictive_draws (real tiny run fixtures; no posterior use)
# ---------------------------------------------------------------------------


def _with_n_draws(lw_run, n: int):
    return dataclasses.replace(
        lw_run,
        spec=lw_run.spec.model_copy(
            update={"outputs": lw_run.spec.outputs.model_copy(update={"prior_predictive_draws": n})}
        ),
    )


def test_prior_predictive_shapes_and_finiteness_no_sv(s4_no_sv_lw_run) -> None:
    r = _with_n_draws(s4_no_sv_lw_run, 25)
    ppd = compute_prior_predictive_draws(r, seed=_SEED)
    T = len(r.dates)
    assert ppd.n_draws == 25
    for name, arr in (("gap", ppd.gap), ("pi", ppd.pi), ("y", ppd.y)):
        assert arr.shape == (25, T), name
        assert np.all(np.isfinite(arr)), name
    np.testing.assert_array_equal(ppd.pi_actual, r.yobs[:, 1])
    np.testing.assert_array_equal(ppd.y_actual, r.yobs[:, 0])


def test_prior_predictive_runs_for_sv_variant(s4_sv_lw_run) -> None:
    r = _with_n_draws(s4_sv_lw_run, 15)
    ppd = compute_prior_predictive_draws(r, seed=_SEED)
    assert ppd.gap.shape[0] == 15
    assert np.all(np.isfinite(ppd.pi))


def test_prior_predictive_is_seed_reproducible_and_seed_sensitive(s4_no_sv_lw_run) -> None:
    r = _with_n_draws(s4_no_sv_lw_run, 10)
    a = compute_prior_predictive_draws(r, seed=123)
    b = compute_prior_predictive_draws(r, seed=123)
    c = compute_prior_predictive_draws(r, seed=124)
    np.testing.assert_array_equal(a.gap, b.gap)
    assert not np.array_equal(a.gap, c.gap)


def test_prior_predictive_paths_vary_across_draws(s4_no_sv_lw_run) -> None:
    """Different prior draws must produce genuinely different paths (a
    frozen-parameter bug would collapse the spread)."""
    r = _with_n_draws(s4_no_sv_lw_run, 10)
    ppd = compute_prior_predictive_draws(r, seed=_SEED)
    assert np.std(ppd.gap[:, -1]) > 0.0
    assert np.std(ppd.pi[:, -1]) > 0.0


def test_prior_predictive_seeds_match_build_lw_regressors_convention(
    s4_no_sv_lw_run, monkeypatch
) -> None:
    """Pin the seed/exog-path literals inside compute_prior_predictive_draws
    against build_lw_regressors' OWN x construction (numerics-reviewer
    recommendation: an off-by-one in the seed dicts would pass the shape/
    finiteness tests silently). Spies on the engine call to capture the
    exact seeds and the DataPathExogRule's data path, then checks them
    against the real x matrix: row 0's endogenous-lag columns are pure
    pre-sample seeds, the ObsLagMean column is the mean of the declared pi
    seeds, and the rule's path is exactly x's r-lag1 column."""
    import macrotoolkit.results_lw as rl
    from macrotoolkit.smoother import build_lw_regressors

    captured: list[tuple] = []
    real_sf = rl.simulate_forward

    def spy(F, Q, A, Z, meta, xi_last, obs_seeds, exog_seeds, exog_rules, mn, horizon, rng):
        captured.append((obs_seeds, exog_seeds, exog_rules["r"], horizon))
        return real_sf(F, Q, A, Z, meta, xi_last, obs_seeds, exog_seeds, exog_rules, mn, horizon, rng)

    monkeypatch.setattr(rl, "simulate_forward", spy)
    r = _with_n_draws(s4_no_sv_lw_run, 2)
    compute_prior_predictive_draws(r, seed=_SEED)

    obs_seeds, exog_seeds, rule, horizon = captured[0]
    yobs, x = build_lw_regressors(r.y_full, r.pi_full, r.r_full)
    T = x.shape[0]
    assert horizon == T

    # Row 0 of the real x is entirely pre-sample data -- the seeds must
    # reproduce it column for column (feedback-map order: y1, y2, r1, r2,
    # pi1, mean(pi2..4)).
    assert obs_seeds["y"][1] == x[0, 0]
    assert obs_seeds["y"][2] == x[0, 1]
    assert exog_seeds["r"] == {2: x[0, 3]}
    assert obs_seeds["pi"][1] == x[0, 4]
    pibar_seed = (obs_seeds["pi"][2] + obs_seeds["pi"][3] + obs_seeds["pi"][4]) / 3.0
    assert pibar_seed == pytest.approx(x[0, 5], abs=1e-12)
    # The rule's data path is exactly x's r-lag1 column, all T rows.
    np.testing.assert_array_equal(rule._values, x[:, 2])


def test_plot_prior_predictive_returns_two_panel_figure(s4_no_sv_lw_run) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from macrotoolkit.plots import plot_prior_predictive

    r = _with_n_draws(s4_no_sv_lw_run, 10)
    ppd = compute_prior_predictive_draws(r, seed=_SEED)
    fig = plot_prior_predictive(ppd)
    assert len(fig.axes) == 2
    import matplotlib.pyplot as plt

    plt.close(fig)
