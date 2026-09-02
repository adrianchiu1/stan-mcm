"""G4 support: the FULL-SV SBC design (lw-sv-spec.md §5, G4 row: "SBC,
full SV variant | uniform rank statistics"), expressed as an
:class:`sbc_harness.SbcDesign` for the family-parameterized engine
(S5-decisions items 10-11). Support module -- the gate lives in
tests/test_g4_sbc.py; the PRE-REGISTERED design constants below are
recorded in DECISIONS.md BEFORE the run starts and never adjusted
afterward to pass (ENGINEERING.md ladder rung 3).

SBC exactness, SV edition (extending g3_harness's two no-SV points):

1. **Prior config**: the family's documented SBC stationarity prior config
   (``SBC_STATIONARITY_PRIOR_CONFIG``, S5-decisions item 6 -- the same
   a1/a2 override G3 ran and passed with) applied through the production
   ``priors:`` override path; everything else production-default,
   including the SV block's sigma_h Half-N(0, 0.2^2) and h_0 ~ N(mu_h0, 1).
   The prior sampler IS the family's own
   :func:`macrotoolkit.families.lw_sv.sample_prior_params` -- the same
   function the prior-predictive check uses -- called on the resolved
   (defaults + SBC config) priors, so fit prior == sampled prior by
   construction, asserted at render time via ``expected_priors``.

2. **Fixed mu_h0 anchors.** Production derives mu_h0 from the run's own
   data (the OLS anchor); an SBC simulator must draw h_0 BEFORE any data
   exists, so -- exactly like G3's fixed Y_ANCHOR for xi00 -- G4 fixes
   mu_h0 at plausible constants (2*ln(0.75) IS / 2*ln(0.80) PC, the
   magnitudes the real US window produces, DECISIONS.md 2026-08-31) and
   passes the SAME constants to the simulator's h_0 draw and the fit's
   data. The Stan program takes mu_h0 as plain data, so this validates
   the same program the runtime runs, at a fixed rather than data-chosen
   anchor.

3. **SV simulation**: per period, the measurement sds are exp(h_t/2) with
   h_t = h_{t-1} + sigma_h * nu_t continuing from the drawn h_0 --
   observation t uses h_t (spec §1.5's timing, matching the template's
   ``sv_rw_noncentered`` construction: h_1 = h_0 + sigma_h*nu_1 governs
   the first observation).

4. **Ranked quantities**: the 10 scalar statics plus the two initial
   log-variances h0_is/h0_pc (recovered from the posterior's non-centered
   ``h0_*_raw`` via h0 = mu_h0 + sd*raw, the template's own line) -- 12
   ranks per replication.
"""
from __future__ import annotations

import numpy as np

from g3_harness import Y_ANCHOR
from macrotoolkit.families.lw_sv import (
    LW_STATE_META,
    SBC_STATIONARITY_PRIOR_CONFIG,
    build_render_context,
    sample_prior_params,
)
from macrotoolkit.smoother import build_lw_matrices, build_lw_regressors, default_initial_state
from sbc_harness import SbcDesign
from specs.schema.base import RunSpec

# --- PRE-REGISTERED DESIGN CONSTANTS (S5-decisions item 10: fixed and ---
# --- recorded in DECISIONS.md BEFORE the run; never adjusted to pass) ---

N_REPLICATIONS = 100  # ~1 day of compute at the measured per-rep cost
SIM_T = 80            # shorter synthetic T than G3's 120 (the reduced design)
RANK_DRAWS = 99       # ranks on {0..99}, as G3
RANK_BINS = 10        # 10 expected per bin at 100 reps (G3: 20 bins at 200)
G4_SEED_BASE = 20260910
CONDITIONING_SEED = 20260909
CHAINS = 2
ITER_WARMUP = 750
ITER_SAMPLING = 750   # pooled 1500 post-warmup draws per rep, as G3
ADAPT_DELTA = 0.95
MAX_TREEDEPTH = 12
DIVERGENT_TOTAL_LIMIT = 150  # 0.1% of 100 reps x 1500 post-warmup draws
CHI2_P_FLOOR = 0.001

#: Fixed mu_h0 anchors (module docstring point 2): 2*ln(sigma) at the
#: plausible OLS residual sds the real US window produces (0.75 IS /
#: 0.80 PC, DECISIONS.md 2026-08-31).
MU_H0_IS_ANCHOR = 2.0 * float(np.log(0.75))
MU_H0_PC_ANCHOR = 2.0 * float(np.log(0.80))

PARAM_LABELS = (
    "a1", "a2", "a_r", "b_pi", "b_y",
    "sigma_ystar", "sigma_g", "sigma_z",
    "sigma_h_is", "sigma_h_pc", "h0_is", "h0_pc",
)


def _resolved_sv_priors() -> dict:
    """The resolved prior dict the G4 render stamps: production defaults +
    the documented SBC stationarity config, through the production
    override path (same assertion pattern as G3's _g3_render_context)."""
    spec = RunSpec.model_validate(
        {
            "model": {"family": "lw_sv", "options": {"sv_shocks": ["is", "pc"]}},
            "data": {
                "file": "unused.csv",
                "date_column": "date",
                "mapping": {"y": "y", "pi": "pi", "r": "r"},
            },
            "priors": SBC_STATIONARITY_PRIOR_CONFIG,
        }
    )
    return build_render_context(spec)["priors"]


G4_PRIORS = _resolved_sv_priors()

#: The template's stamped h_0 non-centering scale (h_0 = mu_h0 + sd*raw).
H0_PRIOR_SD = float(G4_PRIORS["mu_h0_is"]["sd"])


def draw_exact_prior_sv(rng: np.random.Generator) -> dict:
    """One draw from EXACTLY the prior the G4 render fits: the family's
    own prior sampler on the resolved (defaults + SBC config) priors, at
    the FIXED mu_h0 anchors."""
    return sample_prior_params(
        G4_PRIORS, sv_on=True, rng=rng, mu_h0_is=MU_H0_IS_ANCHOR, mu_h0_pc=MU_H0_PC_ANCHOR
    )


# --- fixed conditioning data (G3's pattern, its own seed) -------------------


def _build_conditioning(t: int = SIM_T, seed: int = CONDITIONING_SEED):
    """Same construction as g3_harness._build_conditioning (plausible
    magnitudes, spec §1.1 units), at G4's own seed and T."""
    rng = np.random.default_rng(seed)
    y_pre = Y_ANCHOR - 0.75 * np.arange(4, 0, -1)
    pi_pre = 2.0 + rng.normal(0.0, 0.3, size=4)
    r = np.empty(t + 4)
    dev = 0.0
    for i in range(t + 4):
        dev = 0.8 * dev + rng.normal(0.0, 0.6)
        r[i] = 3.0 + dev
    xi00, P00 = default_initial_state(Y_ANCHOR)
    return y_pre, pi_pre, r, xi00, P00


Y_PRE, PI_PRE, R_PATH, XI00, P00 = _build_conditioning()

_B_YSTAR = LW_STATE_META.injection_vector("ystar")
_B_G = LW_STATE_META.injection_vector("g")
_B_Z = LW_STATE_META.injection_vector("z")


def simulate_from_state_space_sv(params: dict, rng: np.random.Generator, t: int = SIM_T):
    """Simulate one dataset from exactly the generative model the G4 Stan
    program's KF likelihood defines: xi_0 ~ N(xi00, P00); state shocks
    injected at the family's declared loadings; per-period SV measurement
    sds exp(h_t/2) with h continuing its random walk from the drawn h_0
    (module docstring point 3). Returns (y, pi, r) full (t+4,)-length
    arrays, ready for build_lw_regressors."""
    F, _, A, Z, _ = build_lw_matrices({**params, "sigma_is": 1.0, "sigma_pc": 1.0}, c=1.0)

    y = np.empty(t + 4)
    pi = np.empty(t + 4)
    y[:4] = Y_PRE
    pi[:4] = PI_PRE
    r = R_PATH.copy()

    xi = rng.multivariate_normal(XI00, P00)
    h_is = params["h0_is"]
    h_pc = params["h0_pc"]

    for i in range(4, t + 4):
        w = (
            _B_YSTAR * rng.normal(0.0, params["sigma_ystar"])
            + _B_G * rng.normal(0.0, params["sigma_g"])
            + _B_Z * rng.normal(0.0, params["sigma_z"])
        )
        xi = F @ xi + w

        # Observation i uses h_i = h_{i-1} + sigma_h * nu_i (spec §1.5).
        h_is = h_is + params["sigma_h_is"] * rng.normal()
        h_pc = h_pc + params["sigma_h_pc"] * rng.normal()

        x_t = np.array(
            [
                y[i - 1],
                y[i - 2],
                r[i - 1],
                r[i - 2],
                pi[i - 1],
                (pi[i - 2] + pi[i - 3] + pi[i - 4]) / 3.0,
            ]
        )
        # h is log-VARIANCE: sd = exp(h/2), never exp(h).
        e = np.array(
            [rng.normal(0.0, np.exp(h_is / 2.0)), rng.normal(0.0, np.exp(h_pc / 2.0))]
        )
        obs = A.T @ x_t + Z @ xi + e
        y[i] = obs[0]
        pi[i] = obs[1]

    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(pi))):
        raise FloatingPointError(
            f"Simulated SV dataset is non-finite (params={params!r}) -- "
            f"investigate rather than skip."
        )
    return y, pi, r


def stan_data_sv(dataset) -> dict:
    y, pi, r = dataset
    yobs, x = build_lw_regressors(y, pi, r)
    return {
        "T": yobs.shape[0],
        "yobs": yobs,
        "x": x,
        "xi00": XI00,
        "P00": P00,
        "mu_h0_is": MU_H0_IS_ANCHOR,
        "mu_h0_pc": MU_H0_PC_ANCHOR,
    }


def _h0_extractor(anchor: float, raw_name: str):
    def extract(fit) -> np.ndarray:
        return anchor + H0_PRIOR_SD * np.asarray(fit.stan_variable(raw_name))

    return extract


G4_DESIGN = SbcDesign(
    name="G4 full-SV",
    family="lw_sv",
    model_options={"sv_shocks": ["is", "pc"]},
    prior_overrides=SBC_STATIONARITY_PRIOR_CONFIG,
    draw_prior=draw_exact_prior_sv,
    simulate=lambda params, rng: simulate_from_state_space_sv(params, rng),
    stan_data=stan_data_sv,
    ranked_params=(
        "a1", "a2", "a_r", "b_pi", "b_y",
        "sigma_ystar", "sigma_g", "sigma_z", "sigma_h_is", "sigma_h_pc",
        ("h0_is", _h0_extractor(MU_H0_IS_ANCHOR, "h0_is_raw")),
        ("h0_pc", _h0_extractor(MU_H0_PC_ANCHOR, "h0_pc_raw")),
    ),
    n_replications=N_REPLICATIONS,
    rank_draws=RANK_DRAWS,
    rank_bins=RANK_BINS,
    seed_base=G4_SEED_BASE,
    chains=CHAINS,
    iter_warmup=ITER_WARMUP,
    iter_sampling=ITER_SAMPLING,
    adapt_delta=ADAPT_DELTA,
    max_treedepth=MAX_TREEDEPTH,
    expected_priors=G4_PRIORS,
)
