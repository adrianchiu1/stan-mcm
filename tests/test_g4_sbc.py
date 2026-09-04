"""G4 gate (lw-sv-spec.md §5): "SBC, full SV variant | uniform rank
statistics (visual + chi^2 check)" -- run as a PRE-REGISTERED reduced
design (S5-decisions item 10; the exact constants live in
tests/g4_harness.py and are recorded in DECISIONS.md BEFORE the run, never
adjusted afterward to pass), through the family-parameterized SBC engine
(tests/sbc_harness.py, S5-decisions item 11 -- built by generalizing G3's
engine, so G4 is an instantiation, not a reconstruction).

Fast unit guards below pin the properties the gate's validity depends on
(exact-prior equality, simulator determinism, the pre-registered constants
themselves, and the engine's equivalence to G3's recorded generation
path); the gate itself runs under the `slow` marker:
`pytest -m slow tests/test_g4_sbc.py` (~a day of compute).

Accept/reject rule (same shape as G3's, at the reduced design's scale):
per parameter, ranks over RANK_BINS = 10 equal bins (10 expected per bin
at 100 replications), chi^2 against uniform (dof 9), fail below
CHI2_P_FLOOR = 0.001 -- with 12 ranked parameters the false-failure rate
stays ~1%, and every seed is fixed so a false failure would be permanent
(same rationale as G3's floor). Total divergences across all replications
must stay under DIVERGENT_TOTAL_LIMIT = 150 (0.1% of pooled post-warmup
draws): divergent transitions bias the very posterior draws SBC ranks
against, so a broadly divergent run invalidates the calibration evidence
itself.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import g3_harness
import g4_harness
from g4_harness import (
    CHI2_P_FLOOR,
    DIVERGENT_TOTAL_LIMIT,
    G4_DESIGN,
    G4_PRIORS,
    MU_H0_IS_ANCHOR,
    MU_H0_PC_ANCHOR,
    PARAM_LABELS,
    draw_exact_prior_sv,
    simulate_from_state_space_sv,
)
from sbc_harness import SbcDesign, run_sbc

pytestmark = pytest.mark.lw_sv

ARTIFACT_DIR = Path(__file__).parent / "artifacts" / "g4_sbc"


# ---------------------------------------------------------------------------
# Fast unit guards
# ---------------------------------------------------------------------------


def test_g4_priors_are_sv_defaults_plus_the_documented_sbc_config() -> None:
    from macrotoolkit.families.lw_sv import SBC_STATIONARITY_PRIOR_CONFIG
    from specs.schema.lw_sv import DEFAULT_PRIORS

    assert set(SBC_STATIONARITY_PRIOR_CONFIG) == {"a1", "a2"}
    for name, entry in G4_PRIORS.items():
        if name in SBC_STATIONARITY_PRIOR_CONFIG:
            assert entry == {**DEFAULT_PRIORS[name], **SBC_STATIONARITY_PRIOR_CONFIG[name]}
            assert entry != DEFAULT_PRIORS[name]
        else:
            assert entry == DEFAULT_PRIORS[name]
    # The design's render-time exactness assertion is armed.
    assert G4_DESIGN.expected_priors == G4_PRIORS
    # And G3/G4 share the ONE documented SBC prior config (item 6).
    assert g3_harness.SBC_PRIOR_OVERRIDES is SBC_STATIONARITY_PRIOR_CONFIG


def test_g4_prior_sampler_supports_and_anchored_h0() -> None:
    rng = np.random.default_rng(1234)
    draws = [draw_exact_prior_sv(rng) for _ in range(2000)]
    for d in draws:
        assert d["a_r"] < 0.0 and d["b_y"] > 0.0 and 0.0 <= d["b_pi"] <= 1.0
        for k in ("sigma_ystar", "sigma_g", "sigma_z", "sigma_h_is", "sigma_h_pc"):
            assert d[k] >= 0.0
        assert "sigma_is" not in d and "sigma_pc" not in d
    a1 = np.array([d["a1"] for d in draws])
    a2 = np.array([d["a2"] for d in draws])
    # The SBC config is in force (production defaults sit ~4/3 sigma away).
    assert abs(a1.mean() - 0.8) < 0.02 and abs(a2.mean() + 0.25) < 0.01
    nonstationary = ~((-1.0 < a2) & (a2 < 1.0) & (a1 + a2 < 1.0) & (a2 - a1 < 1.0))
    assert nonstationary.sum() == 0
    # h0 draws center on the FIXED anchors (module docstring point 2).
    h0_is = np.array([d["h0_is"] for d in draws])
    h0_pc = np.array([d["h0_pc"] for d in draws])
    assert abs(h0_is.mean() - MU_H0_IS_ANCHOR) < 0.06
    assert abs(h0_pc.mean() - MU_H0_PC_ANCHOR) < 0.06


def test_g4_simulator_deterministic_and_conditioning_fixed() -> None:
    rng_a = np.random.default_rng(g4_harness.G4_SEED_BASE)
    truth_a = draw_exact_prior_sv(rng_a)
    y_a, pi_a, r_a = simulate_from_state_space_sv(truth_a, rng_a)

    rng_b = np.random.default_rng(g4_harness.G4_SEED_BASE)
    truth_b = draw_exact_prior_sv(rng_b)
    y_b, pi_b, r_b = simulate_from_state_space_sv(truth_b, rng_b)

    assert truth_a == truth_b
    np.testing.assert_array_equal(y_a, y_b)
    np.testing.assert_array_equal(pi_a, pi_b)
    np.testing.assert_array_equal(r_a, r_b)
    assert y_a.shape == (g4_harness.SIM_T + 4,)
    np.testing.assert_array_equal(y_a[:4], g4_harness.Y_PRE)
    np.testing.assert_array_equal(pi_a[:4], g4_harness.PI_PRE)
    np.testing.assert_array_equal(r_a, g4_harness.R_PATH)


def test_g4_design_constants_are_the_preregistered_ones() -> None:
    """A literal pin of the pre-registered design (DECISIONS.md,
    2026-09-02): the design cannot quietly shrink -- or grow -- between
    registration and the run."""
    assert g4_harness.N_REPLICATIONS == 100
    assert g4_harness.SIM_T == 80
    assert g4_harness.RANK_DRAWS == 99
    assert g4_harness.RANK_BINS == 10
    assert g4_harness.G4_SEED_BASE == 20260910
    assert (g4_harness.CHAINS, g4_harness.ITER_WARMUP, g4_harness.ITER_SAMPLING) == (2, 750, 750)
    assert (g4_harness.ADAPT_DELTA, g4_harness.MAX_TREEDEPTH) == (0.95, 12)
    assert DIVERGENT_TOTAL_LIMIT == 150 and CHI2_P_FLOOR == 0.001
    assert MU_H0_IS_ANCHOR == pytest.approx(2.0 * np.log(0.75))
    assert MU_H0_PC_ANCHOR == pytest.approx(2.0 * np.log(0.80))
    assert PARAM_LABELS == (
        "a1", "a2", "a_r", "b_pi", "b_y",
        "sigma_ystar", "sigma_g", "sigma_z",
        "sigma_h_is", "sigma_h_pc", "h0_is", "h0_pc",
    )
    assert G4_DESIGN.param_labels == PARAM_LABELS
    assert G4_DESIGN.n_replications == 100


def test_engine_reproduces_g3s_recorded_generation_path() -> None:
    """Item 11's "instantiation, not reconstruction" pin: a G3 design fed
    g3_harness's exact historical callables makes the ENGINE's replication
    loop draw the identical (truth, dataset) pair the recorded 2026-08-31
    G3 run drew at replication 0 -- same seeds, same RNG consumption. (The
    recorded G3 gate file itself is retained verbatim; this proves the
    generic engine is the same generator, without re-running 7 hours.)"""
    g3_design = SbcDesign(
        name="G3 no-SV (engine equivalence pin)",
        family="lw_sv",
        model_options={"sv_shocks": []},
        prior_overrides=g3_harness.SBC_PRIOR_OVERRIDES,
        draw_prior=g3_harness.draw_exact_prior,
        simulate=lambda params, rng: g3_harness.simulate_from_state_space(
            params, g3_harness.CONDITIONING, rng
        ),
        stan_data=lambda ds: {},
        ranked_params=g3_harness.PARAM_NAMES,
        n_replications=g3_harness.N_REPLICATIONS,
        rank_draws=g3_harness.RANK_DRAWS,
        rank_bins=g3_harness.RANK_BINS,
        seed_base=g3_harness.G3_SEED_BASE,
    )
    # The engine's per-rep sequence for rep i: default_rng(seed_base + i),
    # draw_prior, simulate -- byte-identical to test_g3_sbc's own loop.
    i = 0
    rng_engine = np.random.default_rng(g3_design.seed_base + i)
    truth_engine = g3_design.draw_prior(rng_engine)
    y_e, pi_e, r_e = g3_design.simulate(truth_engine, rng_engine)

    rng_legacy = np.random.default_rng(g3_harness.G3_SEED_BASE + i)
    truth_legacy = g3_harness.draw_exact_prior(rng_legacy)
    y_l, pi_l, r_l = g3_harness.simulate_from_state_space(
        truth_legacy, g3_harness.CONDITIONING, rng_legacy
    )

    assert truth_engine == truth_legacy
    np.testing.assert_array_equal(y_e, y_l)
    np.testing.assert_array_equal(pi_e, pi_l)
    np.testing.assert_array_equal(r_e, r_l)

    # And the engine's rank/thin/chi2 primitives are the same functions'
    # arithmetic as the recorded gate's own.
    import sbc_harness

    draws = np.arange(1500, dtype=float)
    np.testing.assert_array_equal(
        sbc_harness.thin_evenly(draws, 99), g3_harness.thin_evenly(draws, 99)
    )
    assert sbc_harness.rank_statistic(700.0, draws) == g3_harness.rank_statistic(700.0, draws)


def test_run_sbc_crash_resume_is_byte_identical(tmp_path) -> None:
    """The engine's crash-resume (added after this environment's container
    restarts killed multi-hour compute twice on 2026-09-02): a run
    interrupted after k replications and resumed must produce EXACTLY the
    ranks of an uninterrupted run -- each rep is fully determined by
    seed_base + i. Uses a stub model so this stays in the fast suite."""
    from sbc_harness import _load_ranks_csv, run_sbc

    class _FakeFit:
        def __init__(self, seed: int) -> None:
            self._rng = np.random.default_rng(seed + 777)

        def stan_variable(self, name):
            return self._rng.normal(size=200)

        def method_variables(self):
            return {"divergent__": np.zeros((10, 2))}

    class _FakeModel:
        def sample(self, **kwargs):
            return _FakeFit(kwargs["seed"])

    design = SbcDesign(
        name="resume pin",
        family="lw_sv",
        model_options={},
        prior_overrides={},
        draw_prior=lambda rng: {"p": float(rng.normal())},
        simulate=lambda params, rng: None,
        stan_data=lambda ds: {},
        ranked_params=("p",),
        n_replications=6,
        rank_draws=99,
        rank_bins=5,
        seed_base=424242,
    )

    full = run_sbc(design, tmp_path / "a", model=_FakeModel(), resume=False, progress_every=1)
    part = run_sbc(design, tmp_path / "b", model=_FakeModel(), n_replications=3, progress_every=1)
    assert part.ranks.shape[0] == 3
    resumed = run_sbc(design, tmp_path / "b", model=_FakeModel(), progress_every=1)
    assert resumed.resumed_from == 3
    np.testing.assert_array_equal(resumed.ranks, full.ranks)
    assert resumed.divergences == full.divergences

    # A ranks.csv from a DIFFERENT design (other labels) must be refused.
    other = tmp_path / "b" / "ranks.csv"
    text = other.read_text().replace("p", "q")
    other.write_text(text)
    with pytest.raises(ValueError, match="refusing to resume"):
        _load_ranks_csv(other, design.param_labels)


# ---------------------------------------------------------------------------
# The G4 gate (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_g4_sbc_full_sv() -> None:
    from sbc_harness import chi2_pvalues

    result = run_sbc(G4_DESIGN, ARTIFACT_DIR)

    total_div = int(sum(result.divergences))
    assert total_div < DIVERGENT_TOTAL_LIMIT, (
        f"G4 sampler health: {total_div} divergent transitions across "
        f"{G4_DESIGN.n_replications} replications (limit "
        f"{DIVERGENT_TOTAL_LIMIT}) -- a broadly divergent run biases the "
        f"very posterior draws SBC ranks, invalidating the calibration "
        f"evidence. Per-rep counts in {ARTIFACT_DIR / 'ranks.csv'}."
    )

    pvals = chi2_pvalues(G4_DESIGN, result.ranks)
    failing = {n: p for n, p in pvals.items() if p < CHI2_P_FLOOR}
    assert not failing, (
        f"G4 SBC rank uniformity rejected for {sorted(failing)} "
        f"(chi^2 p-values {failing}; floor {CHI2_P_FLOOR}). All p-values: "
        f"{pvals}. Inspect {ARTIFACT_DIR / 'rank_histograms.png'} and "
        f"{ARTIFACT_DIR / 'ranks.csv'}."
    )
