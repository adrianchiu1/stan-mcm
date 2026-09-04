"""G3 gate (lw-sv-spec.md §5): "SBC, no-SV variant | uniform rank
statistics (visual + chi^2 check)".

Simulation-based calibration (Talts et al. 2018): for each replication,
draw a parameter point from EXACTLY the Stan program's prior, simulate a
dataset from exactly the generative model the program's KF likelihood
defines, fit the production-rendered lw_sv model, and rank the prior draw
among thinned posterior draws. Under a correct prior-to-posterior pipeline
(model code + sampler), each parameter's ranks are uniform on
{0, ..., RANK_DRAWS}. See tests/g3_harness.py for the two exactness
requirements that keep this file from reusing the G1/G2 machinery
(no stationarity rejection on (a1, a2); state-space simulation from the
explicit xi_0 ~ N(xi00, P00) prior with fixed conditioning data).

Prior configuration (user decision 2026-08-31, DECISIONS.md): the fit uses
the PRODUCTION template + default priors EXCEPT a1/a2, overridden to
g3_harness.SBC_PRIOR_OVERRIDES through the production `priors:` override
path (the same mechanism a user's spec.yaml takes) -- the production
defaults put ~1/3 of prior mass on non-stationary gap dynamics whose
simulated data is numerically un-filterable in float64. So G3 validates
the production program, geometry, and override code path exactly, at two
shifted hyperparameter values.

Scale (user-confirmed 2026-08-31, DECISIONS.md): N_REPLICATIONS = 200 full
NUTS fits under the `slow` marker (~6-7 h; run explicitly with
`pytest -m slow tests/test_g3_sbc.py`).

Accept/reject rule ("uniform rank statistics" needs a concrete gate):

- Per parameter, ranks are binned into RANK_BINS = 20 equal bins (10
  expected per bin at 200 replications) and tested chi^2 against uniform
  (dof 19). The gate fails if any parameter's p-value drops below
  CHI2_P_FLOOR = 0.001. With 10 parameters that puts the false-failure
  rate near 1% -- and because every seed is fixed, a false failure would
  be *permanent*, which is why the floor is 0.001 rather than the naive
  0.05/10: real miscalibration (a prior/simulator mismatch, a factor-of-2
  bug, a broken sampler) drives p-values many orders of magnitude below
  any of these thresholds, while a marginal p is overwhelmingly a rank
  fluctuation. The visual check covers the marginal zone.

- The visual half of the gate: rank histograms for all 10 parameters are
  written to tests/artifacts/g3_sbc/rank_histograms.png (with the per-rep
  ranks + fit diagnostics in ranks.csv alongside, so the figure can be
  re-rendered without re-fitting). The artifacts directory is gitignored;
  the record of a passing run belongs in DECISIONS.md/HANDOFF.md, not the
  tree.

- Sampler health is recorded, not gated per-rep: total divergences across
  all replications must stay below DIVERGENT_TOTAL_LIMIT (divergent
  transitions bias the posterior sample SBC ranks against, so a broadly
  divergent run invalidates the calibration evidence itself -- but the
  no-SV model has sampled cleanly everywhere so far, and a handful of
  divergences across 200 x 3000 draws is noise, not bias worth failing a
  7-hour gate over).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import g1_harness
from g3_harness import (
    CONDITIONING,
    G3_SEED_BASE,
    N_REPLICATIONS,
    PARAM_NAMES,
    RANK_BINS,
    RANK_DRAWS,
    SBC_PRIOR_OVERRIDES,
    SBC_PRIORS,
    SIM_T,
    draw_exact_prior,
    rank_statistic,
    simulate_from_state_space,
    thin_evenly,
)

from macrotoolkit.render import compile_model, render_stan_source
from macrotoolkit.run import build_render_context
from macrotoolkit.smoother import build_lw_regressors
from specs.schema.base import RunSpec

pytestmark = pytest.mark.lw_sv

CHI2_P_FLOOR = 0.001
DIVERGENT_TOTAL_LIMIT = 600  # 0.1% of 200 reps x 3000 post-warmup draws

ARTIFACT_DIR = Path(__file__).parent / "artifacts" / "g3_sbc"


# ---------------------------------------------------------------------------
# Fast unit guards (run in the `not slow` suite): pin the harness properties
# the gate's validity depends on, so a well-meaning refactor can't silently
# break SBC exactness between overnight runs.
# ---------------------------------------------------------------------------


def test_param_names_match_g1() -> None:
    """The two harnesses must agree on the parameter set and order (the
    fit-side extraction loop indexes by this tuple)."""
    assert PARAM_NAMES == g1_harness.PARAM_NAMES


def test_sbc_priors_are_defaults_plus_a1_a2_override() -> None:
    """The G3 prior must be the production defaults EXCEPT the confirmed
    a1/a2 override (g3_harness docstring point 1) -- and the sampler must
    actually draw from it. A drive-by 'sync' of SBC_PRIORS back to
    DEFAULT_PRIORS (whose non-stationary mass is numerically fatal to SBC)
    fails here."""
    from specs.schema.lw_sv import DEFAULT_PRIORS

    assert set(SBC_PRIOR_OVERRIDES) == {"a1", "a2"}
    for name, entry in SBC_PRIORS.items():
        if name in SBC_PRIOR_OVERRIDES:
            assert entry == {**DEFAULT_PRIORS[name], **SBC_PRIOR_OVERRIDES[name]}
            assert entry != DEFAULT_PRIORS[name]
        else:
            assert entry == DEFAULT_PRIORS[name]

    rng = np.random.default_rng(1234)
    draws = [draw_exact_prior(rng) for _ in range(2000)]
    a1 = np.array([d["a1"] for d in draws])
    a2 = np.array([d["a2"] for d in draws])
    # Loose moment checks that the generator reads the OVERRIDE, not the
    # defaults (whose means 1.2/-0.4 sit ~4/3 sigma away from these bands).
    assert abs(a1.mean() - SBC_PRIOR_OVERRIDES["a1"]["mu"]) < 0.02
    assert abs(a2.mean() - SBC_PRIOR_OVERRIDES["a2"]["mu"]) < 0.01
    assert abs(a1.std() - SBC_PRIOR_OVERRIDES["a1"]["sd"]) < 0.02
    # Under the override the stationarity boundary is ~4 sigma out: draws
    # are (essentially) all stationary WITHOUT any rejection filtering.
    nonstationary = ~((-1.0 < a2) & (a2 < 1.0) & (a1 + a2 < 1.0) & (a2 - a1 < 1.0))
    assert nonstationary.sum() == 0
    # And the constrained parameters must still respect the Stan supports.
    assert all(d["a_r"] < 0.0 and d["b_y"] > 0.0 and 0.0 <= d["b_pi"] <= 1.0 for d in draws)
    assert all(d[k] >= 0.0 for d in draws for k in PARAM_NAMES[5:])


def test_simulator_is_deterministic_and_conditioning_is_fixed() -> None:
    rng_a = np.random.default_rng(G3_SEED_BASE)
    params_a = draw_exact_prior(rng_a)
    y_a, pi_a, r_a = simulate_from_state_space(params_a, CONDITIONING, rng_a)

    rng_b = np.random.default_rng(G3_SEED_BASE)
    params_b = draw_exact_prior(rng_b)
    y_b, pi_b, r_b = simulate_from_state_space(params_b, CONDITIONING, rng_b)

    assert params_a == params_b
    np.testing.assert_array_equal(y_a, y_b)
    np.testing.assert_array_equal(pi_a, pi_b)
    np.testing.assert_array_equal(r_a, r_b)

    assert y_a.shape == (SIM_T + 4,)
    # The conditioning rows pass through untouched (they are data, not sim).
    np.testing.assert_array_equal(y_a[:4], CONDITIONING.y_pre)
    np.testing.assert_array_equal(pi_a[:4], CONDITIONING.pi_pre)
    np.testing.assert_array_equal(r_a, CONDITIONING.r)


def test_rank_statistic_and_thinning() -> None:
    draws = np.arange(100, dtype=float)  # 0..99
    assert rank_statistic(-1.0, draws) == 0
    assert rank_statistic(1000.0, draws) == 100
    assert rank_statistic(49.5, draws) == 50

    thinned = thin_evenly(np.arange(1500, dtype=float), RANK_DRAWS)
    assert thinned.shape == (RANK_DRAWS,)
    assert thinned[0] == 0.0 and thinned[-1] == 1499.0
    assert np.all(np.diff(thinned) > 0)

    with pytest.raises(ValueError):
        thin_evenly(np.arange(10, dtype=float), RANK_DRAWS)


# ---------------------------------------------------------------------------
# The G3 gate
# ---------------------------------------------------------------------------


def _g3_render_context() -> dict:
    """The render context `mtk run` would build for an lw_sv spec whose
    `priors:` block carries the G3 a1/a2 override -- the production override
    path end-to-end (mirrors test_g2_parameter_recovery.py's shape, plus
    the override)."""
    spec = RunSpec.model_validate(
        {
            "model": {"family": "lw_sv", "options": {}},
            "data": {
                "file": "unused.csv",
                "date_column": "date",
                "mapping": {"y": "y", "pi": "pi", "r": "r"},
            },
            "priors": SBC_PRIOR_OVERRIDES,
        }
    )
    context = build_render_context(spec)
    # The rendered prior must be exactly the prior the simulator samples --
    # SBC's validity is this equality, so assert it rather than assume it.
    assert context["priors"] == SBC_PRIORS
    return context


def _write_ranks_csv(path: Path, ranks: np.ndarray, divergences: list[int], n_done: int) -> None:
    """Persist per-replication ranks incrementally (a ~7 h run should not
    lose everything to a crash at replication 190)."""
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["replication", *PARAM_NAMES, "divergences"])
        for i in range(n_done):
            writer.writerow([i, *ranks[i].tolist(), divergences[i]])


def _plot_rank_histograms(path: Path, ranks: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rep = ranks.shape[0]
    expected = n_rep / RANK_BINS
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=True)
    edges = np.linspace(0, RANK_DRAWS + 1, RANK_BINS + 1)
    for j, (ax, name) in enumerate(zip(axes.ravel(), PARAM_NAMES)):
        ax.hist(ranks[:, j], bins=edges, color="steelblue", edgecolor="white")
        ax.axhline(expected, color="firebrick", linestyle="--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("rank")
    fig.suptitle(
        f"G3 SBC rank histograms -- no-SV lw_sv, {n_rep} replications "
        f"(dashed line = uniform expectation {expected:.0f}/bin)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@pytest.mark.slow
def test_g3_sbc_no_sv() -> None:
    from scipy import stats  # arviz dependency, present in the test env

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    source = render_stan_source("lw_sv.stan.j2", _g3_render_context())
    model, _ = compile_model(source)

    ranks = np.zeros((N_REPLICATIONS, len(PARAM_NAMES)), dtype=int)
    divergences: list[int] = []

    for i in range(N_REPLICATIONS):
        rng = np.random.default_rng(G3_SEED_BASE + i)
        truth = draw_exact_prior(rng)
        y, pi, r = simulate_from_state_space(truth, CONDITIONING, rng)
        yobs, x = build_lw_regressors(y, pi, r)
        fit = model.sample(
            data={
                "T": yobs.shape[0],
                "yobs": yobs,
                "x": x,
                "xi00": CONDITIONING.xi00,
                "P00": CONDITIONING.P00,
            },
            chains=2,
            parallel_chains=2,
            iter_warmup=750,
            iter_sampling=750,
            adapt_delta=0.95,
            max_treedepth=12,
            seed=G3_SEED_BASE + i,
            show_progress=False,
        )
        for j, name in enumerate(PARAM_NAMES):
            draws = fit.stan_variable(name)
            ranks[i, j] = rank_statistic(truth[name], thin_evenly(draws, RANK_DRAWS))
        divergences.append(int(np.sum(fit.method_variables()["divergent__"])))

        if (i + 1) % 10 == 0 or i == N_REPLICATIONS - 1:
            _write_ranks_csv(ARTIFACT_DIR / "ranks.csv", ranks, divergences, i + 1)

    _plot_rank_histograms(ARTIFACT_DIR / "rank_histograms.png", ranks)

    # chi^2 uniformity per parameter.
    edges = np.linspace(0, RANK_DRAWS + 1, RANK_BINS + 1)
    pvals = {}
    for j, name in enumerate(PARAM_NAMES):
        observed, _ = np.histogram(ranks[:, j], bins=edges)
        expected = np.full(RANK_BINS, N_REPLICATIONS / RANK_BINS)
        chi2 = float(((observed - expected) ** 2 / expected).sum())
        pvals[name] = float(stats.chi2.sf(chi2, df=RANK_BINS - 1))

    total_div = int(sum(divergences))
    assert total_div < DIVERGENT_TOTAL_LIMIT, (
        f"G3 sampler health: {total_div} divergent transitions across "
        f"{N_REPLICATIONS} replications (limit {DIVERGENT_TOTAL_LIMIT}) -- "
        f"a broadly divergent run biases the very posterior draws SBC ranks, "
        f"so this invalidates the calibration evidence. Per-rep counts in "
        f"{ARTIFACT_DIR / 'ranks.csv'}."
    )

    failing = {n: p for n, p in pvals.items() if p < CHI2_P_FLOOR}
    assert not failing, (
        f"G3 SBC rank uniformity rejected for {sorted(failing)} "
        f"(chi^2 p-values {failing}; floor {CHI2_P_FLOOR}). All p-values: "
        f"{pvals}. Inspect {ARTIFACT_DIR / 'rank_histograms.png'} and "
        f"{ARTIFACT_DIR / 'ranks.csv'}."
    )
