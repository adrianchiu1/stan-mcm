"""G2 gate (lw-sv-spec.md §5): "Parameter recovery on simulated data (no
SV), 20 datasets | true values inside 90% CI ~90% of the time; no
systematic bias in sigma_g, sigma_z."

Marked `slow`: 20 full NUTS estimations. Run explicitly with
`pytest -m slow tests/test_g2_parameter_recovery.py`.

Operationalization (documented here because "~90%" needs a concrete
accept/reject rule):

- True parameter points: drawn from the priors by G1's generator (seeded,
  reproducible); dataset i simulated at point i with seed SIM_SEED_BASE+i.
- Coverage: pooled over 10 parameters x 20 datasets = 200 cells, each cell
  1 if the true value lies inside the equal-tailed 90% posterior CI. With
  within-dataset correlation the effective sample is < 200, so the accept
  band is deliberately loose: pooled coverage in [0.80, 0.97]. A
  per-parameter floor (>= 12/20 = 0.6) catches a single pathologically
  miscalibrated parameter hiding in an acceptable pool.
- sigma_g/sigma_z bias: the mean signed error of the posterior median
  across datasets must be within 3 standard errors of zero -- "no
  systematic bias" as a 3-sigma t-test, the parameters spec §1.6's pile-up
  priors are meant to discipline.

The estimations reuse the production render/compile path (the same
lw_sv.stan.j2 render `mtk run` uses, with default priors), 2 chains x 750
draws each -- enough for 90% CI endpoints, chosen to keep the gate's total
runtime tolerable.
"""
from __future__ import annotations

import numpy as np
import pytest

from g1_harness import PARAM_NAMES, generate_parameter_points
from g2_harness import N_DATASETS, SIM_SEED_BASE, SIM_T, simulate_lw_dataset

from macrotoolkit.render import compile_model, render_stan_source
from macrotoolkit.run import build_render_context
from macrotoolkit.smoother import build_lw_regressors, default_initial_state
from specs.schema.base import RunSpec

#: Seed for the 20 true parameter points (distinct from G1's PARAM_SEED so
#: the two gates don't share draws).
G2_PARAM_SEED = 20260830

COVERAGE_LOW = 0.80
COVERAGE_HIGH = 0.97
PER_PARAM_FLOOR = 0.6
BIAS_T_LIMIT = 3.0


def _default_lw_render_context() -> dict:
    """The same render context `mtk run` builds for an lw_sv spec with no
    prior overrides."""
    spec = RunSpec.model_validate(
        {
            "model": {"family": "lw_sv", "options": {}},
            "data": {
                "file": "unused.csv",
                "date_column": "date",
                "mapping": {"y": "y", "pi": "pi", "r": "r"},
            },
        }
    )
    return build_render_context(spec)


@pytest.mark.slow
def test_g2_parameter_recovery_no_sv() -> None:
    truths = generate_parameter_points(N_DATASETS, seed=G2_PARAM_SEED)

    source = render_stan_source("lw_sv.stan.j2", _default_lw_render_context())
    model, _ = compile_model(source)

    inside = np.zeros((N_DATASETS, len(PARAM_NAMES)), dtype=bool)
    median_err = {"sigma_g": [], "sigma_z": []}

    for i, truth in enumerate(truths):
        sim = simulate_lw_dataset(truth, t=SIM_T, seed=SIM_SEED_BASE + i)
        yobs, x = build_lw_regressors(sim.y, sim.pi, sim.r)
        xi00, P00 = default_initial_state(float(sim.y[4]))
        fit = model.sample(
            data={
                "T": yobs.shape[0],
                "yobs": yobs,
                "x": x,
                "xi00": xi00,
                "P00": P00,
            },
            chains=2,
            parallel_chains=2,
            iter_warmup=750,
            iter_sampling=750,
            adapt_delta=0.95,
            max_treedepth=12,
            seed=SIM_SEED_BASE + i,
            show_progress=False,
        )
        for j, name in enumerate(PARAM_NAMES):
            draws = fit.stan_variable(name)
            lo, hi = np.quantile(draws, [0.05, 0.95])
            inside[i, j] = lo <= truth[name] <= hi
            if name in median_err:
                median_err[name].append(float(np.median(draws)) - truth[name])

    pooled = inside.mean()
    per_param = {name: inside[:, j].mean() for j, name in enumerate(PARAM_NAMES)}

    assert COVERAGE_LOW <= pooled <= COVERAGE_HIGH, (
        f"G2 pooled 90%-CI coverage {pooled:.3f} outside "
        f"[{COVERAGE_LOW}, {COVERAGE_HIGH}]. Per-parameter: {per_param}"
    )
    for name, cov in per_param.items():
        assert cov >= PER_PARAM_FLOOR, (
            f"G2 coverage for {name} is {cov:.2f} < {PER_PARAM_FLOOR} "
            f"(per-parameter floor). Full table: {per_param}"
        )

    for name, errs in median_err.items():
        errs_arr = np.array(errs)
        se = errs_arr.std(ddof=1) / np.sqrt(len(errs_arr))
        tstat = errs_arr.mean() / se if se > 0 else 0.0
        assert abs(tstat) < BIAS_T_LIMIT, (
            f"G2 systematic bias in {name}: mean posterior-median error "
            f"{errs_arr.mean():+.5f} (se {se:.5f}, t={tstat:+.2f}) exceeds "
            f"the {BIAS_T_LIMIT}-sigma no-bias criterion."
        )
