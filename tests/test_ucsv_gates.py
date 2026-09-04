"""The ``ucsv`` validation LADDER (S6 WP2c), instantiated from the generic
harnesses and driven by the family's REGISTERED designs
(macrotoolkit/families/ucsv_validation.py -- the same declarations
``mtk validate ucsv`` runs):

- G1: the shared filter's mirror gate, tests/test_g1_mirror.py (the Q_t
  paths), plus the family's own production-render mirror gate in the
  fast tier (tests/test_validation.py).
- G2: parameter recovery, ``UCSV_G2_DESIGN`` -- slow.
- G3/G4-style SBC: ``UCSV_SBC_DESIGN``, PRE-REGISTERED in DECISIONS.md
  (2026-09-04) BEFORE the run; the constants are pinned literally below
  so the design cannot drift; crash-resumable through ranks.csv -- slow.
- G6: the HD identity, tests/test_ucsv_family.py (fast) + the fast tier.

There is no external oracle for UCSV (no G5a analogue); the ladder's
external credibility rests on recovery + SBC, stated in README/HANDOFF.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from macrotoolkit.families import ucsv_validation as uv

pytestmark = pytest.mark.ucsv

ARTIFACT_DIR = Path(__file__).parent / "artifacts" / "ucsv_sbc"
G2_ARTIFACT_DIR = Path(__file__).parent / "artifacts" / "ucsv_g2"


def test_sbc_design_constants_are_the_preregistered_ones() -> None:
    """Pins the design recorded in DECISIONS.md 2026-09-04 literally."""
    d = uv.UCSV_SBC_DESIGN
    assert (uv.SBC_N_REPLICATIONS, uv.SBC_SIM_T) == (100, 100)
    assert (d.n_replications, d.rank_draws, d.rank_bins, d.seed_base) == (100, 99, 10, 20260920)
    assert (d.chains, d.iter_warmup, d.iter_sampling, d.adapt_delta, d.max_treedepth) == (2, 750, 750, 0.95, 12)
    assert (uv.SBC_DIVERGENT_TOTAL_LIMIT, uv.SBC_CHI2_P_FLOOR) == (150, 0.001)
    assert d.param_labels == ("sigma_h_eps", "sigma_h_eta", "h0_eps", "h0_eta")
    assert uv.MU_H0_ANCHOR == 2.0 * np.log(0.5) and uv.TAU0_ANCHOR == 2.0 and uv.TAU0_SD == 5.0
    assert d.model_options == {"sv_shocks": ["eps", "eta"]} and d.prior_overrides == {}
    assert d.expected_priors == uv.UCSV_PRIORS
    g2 = uv.UCSV_G2_DESIGN
    assert (g2.n_datasets, uv.G2_SIM_T, g2.seed_base) == (20, 100, 20260921)
    assert g2.param_labels == d.param_labels and g2.bias_params == ("sigma_h_eps", "sigma_h_eta")


def test_prior_sampler_and_simulator_are_exact_and_deterministic() -> None:
    rng_a = np.random.default_rng(uv.SBC_SEED_BASE)
    p_a = uv.draw_exact_prior(rng_a)
    pi_a = uv.simulate_ucsv_dataset(p_a, rng_a)
    rng_b = np.random.default_rng(uv.SBC_SEED_BASE)
    p_b = uv.draw_exact_prior(rng_b)
    pi_b = uv.simulate_ucsv_dataset(p_b, rng_b)
    assert p_a == p_b and pi_a.shape == (uv.SBC_SIM_T,)
    np.testing.assert_array_equal(pi_a, pi_b)
    # Sampled prior == rendered prior (the render step asserts it too).
    assert set(p_a) == {"sigma_h_eps", "sigma_h_eta", "h0_eps", "h0_eta"}
    draws = [uv.draw_exact_prior(np.random.default_rng(i)) for i in range(2000)]
    h0 = np.array([d["h0_eps"] for d in draws])
    assert abs(h0.mean() - uv.MU_H0_ANCHOR) < 0.08 and abs(h0.std() - uv.H0_PRIOR_SD) < 0.06
    data = uv.stan_data_fixed_anchors(pi_a)
    assert data["mu_h0_eps"] == uv.MU_H0_ANCHOR and float(data["xi00"][0]) == uv.TAU0_ANCHOR
    assert data["yobs"].shape == (uv.SBC_SIM_T, 1)


def test_render_of_the_design_matches_its_expected_prior() -> None:
    from macrotoolkit.validation.sbc import render_design_model

    model = render_design_model(uv.UCSV_SBC_DESIGN)  # asserts expected_priors internally
    assert model.exe_file is not None


# ---------------------------------------------------------------------------
# The slow gates
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_g2_parameter_recovery_ucsv() -> None:
    from macrotoolkit.validation.recovery import evaluate_recovery, run_recovery, write_coverage_csv

    G2_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    result = run_recovery(uv.UCSV_G2_DESIGN)
    write_coverage_csv(G2_ARTIFACT_DIR / "coverage.csv", uv.UCSV_G2_DESIGN, result)
    verdict, reasons, metrics = evaluate_recovery(uv.UCSV_G2_DESIGN, result)
    assert verdict == "PASS", (reasons, metrics)


@pytest.mark.slow
def test_sbc_ucsv_full_sv() -> None:
    """The pre-registered UCSV SBC gate: resumes from ARTIFACT_DIR/ranks.csv
    when present (a completed run reloads without re-fitting) and applies
    the pre-registered accept/reject rule."""
    from macrotoolkit.validation.sbc import chi2_pvalues, run_sbc

    res = run_sbc(uv.UCSV_SBC_DESIGN, ARTIFACT_DIR)
    assert res.ranks.shape == (uv.SBC_N_REPLICATIONS, 4)
    pvals = chi2_pvalues(uv.UCSV_SBC_DESIGN, res.ranks)
    total_div = int(sum(res.divergences))
    for name, p in pvals.items():
        assert p >= uv.SBC_CHI2_P_FLOOR, f"UCSV SBC: rank chi^2 p = {p:.4f} < {uv.SBC_CHI2_P_FLOOR} for {name} (all: {pvals})"
    assert total_div <= uv.SBC_DIVERGENT_TOTAL_LIMIT, f"{total_div} divergences > {uv.SBC_DIVERGENT_TOTAL_LIMIT}"
