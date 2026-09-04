"""The ``ucsv`` family's REGISTERED VALIDATION DESIGNS (S6 WP2c/WP3):
what ``mtk validate ucsv`` runs per tier, instantiated from the generic
harnesses -- nothing here is a new gate mechanism.

- fast: the Stan-vs-Python KF mirror at prior draws of the production
  render on the worked example's data (G1's shape), and the historical-
  decomposition reconstruction identity (G6's shape).
- recovery: :data:`UCSV_G2_DESIGN` -- parameter recovery on simulated
  data (G2's shape).
- sbc: :data:`UCSV_SBC_DESIGN` -- the PRE-REGISTERED simulation-based-
  calibration design (G3/G4's shape). The constants below are recorded in
  DECISIONS.md (2026-09-04) BEFORE the run and are never adjusted
  afterward to pass; ``tests/test_ucsv_gates.py`` pins them literally.

There is NO external oracle for UCSV (nothing like HLW's published code
for lw_sv's G5a), stated explicitly in README/HANDOFF: the family's
external credibility rests on the mirror gate, recovery, and SBC.

SBC exactness for UCSV (the two family-authored pieces, per the S5 audit):

1. **Prior sampler** = :func:`macrotoolkit.families.ucsv.sample_prior_params`
   on the resolved production priors (``SBC_PRIOR_CONFIG`` is empty: UCSV
   has no AR block, nothing needs a stationarity override), at FIXED
   h_0 anchors -- production derives mu_h0 from the data, but an SBC
   simulator draws h_0 before any data exists, so both the simulator and
   the fit use ``MU_H0_ANCHOR = 2*ln(0.5)`` (the G4 pattern). Likewise the
   initial state: ``tau_0 ~ N(TAU0_ANCHOR, tau0_sd^2)`` in the simulator
   and ``xi00 = TAU0_ANCHOR`` in the fit (G3's fixed Y_ANCHOR pattern),
   rather than the production ``xi00 = pi_1`` data anchor. The Stan
   program takes both as plain data, so the production program is
   validated at fixed rather than data-chosen anchors.
2. **Structural simulator** :func:`simulate_ucsv_dataset`: the model's
   own equations (tau random walk, pi = tau + eps) with both log-variance
   random walks continued from the drawn h_0 -- observation t uses h_t =
   h_{t-1} + sigma_h * nu_t (the template's ``sv_rw_noncentered`` timing).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from macrotoolkit.families.ucsv import (
    SBC_PRIOR_CONFIG,
    build_render_context,
    sample_prior_params,
)
from macrotoolkit.validation.recovery import RecoveryDesign
from macrotoolkit.validation.sbc import SbcDesign
from specs.schema.base import RunSpec
from specs.schema.ucsv import INITIAL_STATE_PRIOR

# --- PRE-REGISTERED SBC DESIGN CONSTANTS (recorded in DECISIONS.md -------
# --- 2026-09-04 BEFORE the run; never adjusted afterward to pass) --------

SBC_N_REPLICATIONS = 100
SBC_SIM_T = 100
SBC_RANK_DRAWS = 99
SBC_RANK_BINS = 10
SBC_SEED_BASE = 20260920
SBC_CHAINS = 2
SBC_ITER_WARMUP = 750
SBC_ITER_SAMPLING = 750
SBC_ADAPT_DELTA = 0.95
SBC_MAX_TREEDEPTH = 12
SBC_DIVERGENT_TOTAL_LIMIT = 150  # 0.1% of 100 x 1500 post-warmup draws
SBC_CHI2_P_FLOOR = 0.001

#: Fixed anchors (module docstring point 1).
MU_H0_ANCHOR = 2.0 * float(np.log(0.5))   # both shocks: sd 0.5 pp -> log-variance
TAU0_ANCHOR = 2.0                          # tau_0 prior mean (pp), P00 = tau0_sd^2
TAU0_SD = float(INITIAL_STATE_PRIOR["tau0_sd"])

SBC_PARAM_LABELS = ("sigma_h_eps", "sigma_h_eta", "h0_eps", "h0_eta")

# --- G2 recovery design constants -----------------------------------------
G2_N_DATASETS = 20
G2_SIM_T = 100
G2_SEED_BASE = 20260921


def _resolved_priors() -> dict:
    spec = RunSpec.model_validate(
        {
            "model": {"family": "ucsv", "options": {"sv_shocks": ["eps", "eta"]}},
            "data": {"file": "unused.csv", "date_column": "date", "mapping": {"pi": "pi"}},
            "priors": SBC_PRIOR_CONFIG,
        }
    )
    return build_render_context(spec)["priors"]


UCSV_PRIORS = _resolved_priors()
H0_PRIOR_SD = float(UCSV_PRIORS["mu_h0_eps"]["sd"])


def draw_exact_prior(rng: np.random.Generator) -> dict:
    """One draw from EXACTLY the prior the render fits, at the fixed
    anchors."""
    return sample_prior_params(UCSV_PRIORS, True, rng, mu_h0_eps=MU_H0_ANCHOR, mu_h0_eta=MU_H0_ANCHOR)


def simulate_ucsv_dataset(params: dict, rng: np.random.Generator, t: int = SBC_SIM_T) -> np.ndarray:
    """One dataset from exactly the generative model the Stan program's
    KF likelihood defines (module docstring point 2). Returns pi (t,)."""
    tau = TAU0_ANCHOR + TAU0_SD * rng.standard_normal()
    h_eps = params["h0_eps"]
    h_eta = params["h0_eta"]
    pi = np.empty(t)
    for i in range(t):
        # Observation i uses h_i = h_{i-1} + sigma_h * nu_i (both shocks).
        h_eta = h_eta + params["sigma_h_eta"] * rng.standard_normal()
        h_eps = h_eps + params["sigma_h_eps"] * rng.standard_normal()
        tau = tau + rng.normal(0.0, np.exp(h_eta / 2.0))  # h is log-VARIANCE
        pi[i] = tau + rng.normal(0.0, np.exp(h_eps / 2.0))
    if not np.all(np.isfinite(pi)):
        raise FloatingPointError(f"Simulated UCSV dataset is non-finite (params={params!r}).")
    return pi


def stan_data_fixed_anchors(pi: np.ndarray) -> dict:
    """The fit's data block at the FIXED anchors (not the production
    data-derived ones)."""
    pi = np.asarray(pi, dtype=np.float64)
    return {
        "T": int(pi.shape[0]),
        "yobs": pi[:, None],
        "xi00": np.array([TAU0_ANCHOR]),
        "P00": np.array([[TAU0_SD**2]]),
        "mu_h0_eps": MU_H0_ANCHOR,
        "mu_h0_eta": MU_H0_ANCHOR,
    }


def _h0_extractor(raw_name: str):
    def extract(fit) -> np.ndarray:
        return MU_H0_ANCHOR + H0_PRIOR_SD * np.asarray(fit.stan_variable(raw_name))

    return extract


UCSV_SBC_DESIGN = SbcDesign(
    name="UCSV SBC (full SV)",
    family="ucsv",
    model_options={"sv_shocks": ["eps", "eta"]},
    prior_overrides=SBC_PRIOR_CONFIG,
    draw_prior=draw_exact_prior,
    simulate=lambda params, rng: simulate_ucsv_dataset(params, rng),
    stan_data=stan_data_fixed_anchors,
    ranked_params=(
        "sigma_h_eps", "sigma_h_eta",
        ("h0_eps", _h0_extractor("h0_eps_raw")),
        ("h0_eta", _h0_extractor("h0_eta_raw")),
    ),
    n_replications=SBC_N_REPLICATIONS,
    rank_draws=SBC_RANK_DRAWS,
    rank_bins=SBC_RANK_BINS,
    seed_base=SBC_SEED_BASE,
    chains=SBC_CHAINS,
    iter_warmup=SBC_ITER_WARMUP,
    iter_sampling=SBC_ITER_SAMPLING,
    adapt_delta=SBC_ADAPT_DELTA,
    max_treedepth=SBC_MAX_TREEDEPTH,
    expected_priors=UCSV_PRIORS,
)

UCSV_G2_DESIGN = RecoveryDesign(
    name="UCSV parameter recovery (full SV)",
    family="ucsv",
    model_options={"sv_shocks": ["eps", "eta"]},
    prior_overrides=SBC_PRIOR_CONFIG,
    draw_truth=draw_exact_prior,
    simulate=lambda params, rng: simulate_ucsv_dataset(params, rng, t=G2_SIM_T),
    stan_data=stan_data_fixed_anchors,
    params=(
        "sigma_h_eps", "sigma_h_eta",
        ("h0_eps", _h0_extractor("h0_eps_raw")),
        ("h0_eta", _h0_extractor("h0_eta_raw")),
    ),
    bias_params=("sigma_h_eps", "sigma_h_eta"),
    n_datasets=G2_N_DATASETS,
    seed_base=G2_SEED_BASE,
    chains=SBC_CHAINS,
    iter_warmup=SBC_ITER_WARMUP,
    iter_sampling=SBC_ITER_SAMPLING,
    adapt_delta=SBC_ADAPT_DELTA,
    max_treedepth=SBC_MAX_TREEDEPTH,
    expected_priors=UCSV_PRIORS,
)


# ---------------------------------------------------------------------------
# The registered suite
# ---------------------------------------------------------------------------

_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "us_ucsv"


def example_spec() -> RunSpec:
    from specs.schema.base import load_spec

    return load_spec(str(_EXAMPLE_DIR / "spec.yaml"))


def example_df():
    from macrotoolkit.data import load_data

    df, _, _ = load_data(example_spec(), base_dir=_EXAMPLE_DIR)
    return df


def hd_reconstruction_error(spec: RunSpec, df, draw_index: int) -> float:
    """G6's shape for ucsv at a prior point: simulation-smoother draw ->
    generic observable bars -> max |sum of bars - pi| over the sample."""
    from macrotoolkit.families.ucsv import (
        UCSV_STATE_META,
        build_ucsv_matrices,
        build_ucsv_regressors,
        default_initial_state,
        ucsv_mu_h0_anchors,
    )
    from macrotoolkit.results_core import observable_bars, state_components
    from macrotoolkit.smoother import simulate_smoother_draw, sv_rw_noncentered, sv_scalar_variance_path

    pi = df["pi"].to_numpy(dtype=np.float64)
    yobs, x = build_ucsv_regressors(pi)
    xi00, P00 = default_initial_state(float(pi[0]))
    mu_eps, mu_eta = ucsv_mu_h0_anchors(pi)
    rng = np.random.default_rng(20260923 + draw_index)
    priors = build_render_context(spec)["priors"]
    params = sample_prior_params(priors, True, rng, mu_h0_eps=mu_eps, mu_h0_eta=mu_eta)
    T = pi.shape[0]
    Q = sv_scalar_variance_path(sv_rw_noncentered(params["h0_eta"], params["sigma_h_eta"], rng.standard_normal(T)))
    R = sv_scalar_variance_path(sv_rw_noncentered(params["h0_eps"], params["sigma_h_eps"], rng.standard_normal(T)))
    F, _, A, Z, _ = build_ucsv_matrices({"sigma_eta": 1.0, "sigma_eps": 1.0})
    sim = simulate_smoother_draw(yobs, x, F, Q, A, Z, R, xi00, P00, rng, meta=UCSV_STATE_META)
    comps = state_components(F, UCSV_STATE_META, sim.xi_draw, sim.state_shocks)
    bars = observable_bars(A, Z, UCSV_STATE_META, comps, sim.meas_shocks)
    total = sum(b[:, 0] for b in bars.values())
    return float(np.max(np.abs(total - pi)))


def _suite():
    from macrotoolkit.validation.suite import (
        ValidationSuite,
        hd_identity_gate,
        mirror_gate,
        recovery_gate,
        sbc_gate,
    )

    return ValidationSuite(
        family="ucsv",
        gates=(
            mirror_gate("ucsv", example_spec, example_df),
            hd_identity_gate("ucsv", example_spec, example_df, hd_reconstruction_error),
            recovery_gate(UCSV_G2_DESIGN),
            sbc_gate(UCSV_SBC_DESIGN, p_floor=SBC_CHI2_P_FLOOR, divergence_limit=SBC_DIVERGENT_TOTAL_LIMIT),
        ),
    )


VALIDATION_SUITE = _suite()
