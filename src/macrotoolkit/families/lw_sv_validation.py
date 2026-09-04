"""The ``lw_sv`` family's REGISTERED VALIDATION DESIGNS (S6 WP3): what
``mtk validate lw_sv`` runs per tier.

- fast: the Stan-vs-Python KF mirror at prior draws of the production SV
  render on the US example data (G1's shape) and the historical-
  decomposition reconstruction identity (G6's shape).
- recovery: :data:`LW_G2_DESIGN` -- parameter recovery on the no-SV
  variant (the shape of the recorded G2 gate: stationarity-filtered
  prior truths, the structural-equation simulator, 20 datasets).
- sbc: :data:`G4_DESIGN` -- the PRE-REGISTERED full-SV SBC design,
  moved here VERBATIM from tests/g4_harness.py (which now re-exports it)
  so the G4 gate stays an instantiation of one declaration; its
  constants are those DECISIONS.md recorded on 2026-09-02, unchanged.

The G4 design's original docstring follows, kept intact:

G4 support: the FULL-SV SBC design (lw-sv-spec.md §5, G4 row: "SBC,
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

from macrotoolkit.families.lw_sv import (
    LW_STATE_META,
    SBC_STATIONARITY_PRIOR_CONFIG,
    build_render_context,
    sample_prior_params,
)
from macrotoolkit.smoother import build_lw_matrices, build_lw_regressors, default_initial_state
from macrotoolkit.validation.sbc import SbcDesign
from specs.schema.base import RunSpec

#: G3's fixed initial-observation anchor (tests/g3_harness.py), shared by
#: the G4 conditioning data.
Y_ANCHOR = 900.0

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


# ---------------------------------------------------------------------------
# G2 recovery design (the shape of tests/test_g2_parameter_recovery.py)
# ---------------------------------------------------------------------------

from macrotoolkit.validation.recovery import RecoveryDesign  # noqa: E402

G2_N_DATASETS = 20
G2_SIM_T = 120
G2_BURN_IN = 40
G2_SEED_BASE = 20260831
G2_PARAM_SEED = 20260830
G2_PARAM_NAMES = ("a1", "a2", "a_r", "b_pi", "b_y", "sigma_ystar", "sigma_g", "sigma_z", "sigma_is", "sigma_pc")


def _is_stationary_ar2(a1: float, a2: float) -> bool:
    return (-1.0 < a2 < 1.0) and (a1 + a2 < 1.0) and (a2 - a1 < 1.0)


def draw_stationary_truth(rng: np.random.Generator) -> dict:
    """A no-SV truth from the production prior with the (a1, a2)
    stationarity filter (G1/G2's documented rejection -- recovery is not
    calibration, so filtering is legitimate here)."""
    from macrotoolkit.families.lw_sv import build_render_context as _ctx
    from macrotoolkit.families.lw_sv import sample_prior_params as _sample

    spec = RunSpec.model_validate(
        {"model": {"family": "lw_sv", "options": {}},
         "data": {"file": "unused.csv", "date_column": "date", "mapping": {"y": "y", "pi": "pi", "r": "r"}}}
    )
    priors = _ctx(spec)["priors"]
    for _ in range(10_000):
        p = _sample(priors, False, rng)
        if _is_stationary_ar2(p["a1"], p["a2"]):
            p["c"] = 1.0
            return p
    raise RuntimeError("could not draw a stationary (a1, a2) pair in 10,000 attempts")


def simulate_lw_dataset(params: dict, t: int = G2_SIM_T, seed: int | None = None, rng: np.random.Generator | None = None):
    """The no-SV LW structural-equation simulator (moved verbatim from
    tests/g2_harness.py, which re-exports it): latent random walks, an
    AR(1) real-rate deviation around r*, the IS and Phillips curves as
    written. Returns ``(y, pi, r)`` full (t+4,)-length arrays."""
    if rng is None:
        rng = np.random.default_rng(seed)
    c = params.get("c", 1.0)
    a1, a2, a_r = params["a1"], params["a2"], params["a_r"]
    b_pi, b_y = params["b_pi"], params["b_y"]
    n = G2_BURN_IN + 4 + t
    g = np.empty(n)
    z = np.empty(n)
    ystar = np.empty(n)
    g[0] = 3.0
    z[0] = 0.0
    ystar[0] = 900.0
    eps_g = rng.normal(0.0, params["sigma_g"], size=n)
    eps_z = rng.normal(0.0, params["sigma_z"], size=n)
    eps_ystar = rng.normal(0.0, params["sigma_ystar"], size=n)
    for i in range(1, n):
        g[i] = g[i - 1] + eps_g[i]
        z[i] = z[i - 1] + eps_z[i]
        ystar[i] = ystar[i - 1] + g[i - 1] / 4.0 + eps_ystar[i]
    rstar = c * g + z
    r = np.empty(n)
    dev = 0.0
    for i in range(n):
        dev = 0.8 * dev + rng.normal(0.0, 0.8)
        r[i] = rstar[i] + dev
    gap = np.zeros(n)
    eps_is = rng.normal(0.0, params["sigma_is"], size=n)
    for i in range(2, n):
        rate_gap_term = (a_r / 2.0) * ((r[i - 1] - rstar[i - 1]) + (r[i - 2] - rstar[i - 2]))
        gap[i] = a1 * gap[i - 1] + a2 * gap[i - 2] + rate_gap_term + eps_is[i]
    pi = np.full(n, 2.0)
    eps_pc = rng.normal(0.0, params["sigma_pc"], size=n)
    for i in range(4, n):
        pibar = (pi[i - 2] + pi[i - 3] + pi[i - 4]) / 3.0
        pi[i] = b_pi * pi[i - 1] + (1.0 - b_pi) * pibar + b_y * gap[i - 1] + eps_pc[i]
    y = ystar + gap
    keep = slice(G2_BURN_IN, n)
    return y[keep], pi[keep], r[keep]


def stan_data_no_sv(dataset) -> dict:
    y, pi, r = dataset
    yobs, x = build_lw_regressors(y, pi, r)
    xi00, P00 = default_initial_state(float(y[4]))
    return {"T": yobs.shape[0], "yobs": yobs, "x": x, "xi00": xi00, "P00": P00}


LW_G2_DESIGN = RecoveryDesign(
    name="lw_sv parameter recovery (no SV)",
    family="lw_sv",
    model_options={"sv_shocks": []},
    prior_overrides={},
    draw_truth=draw_stationary_truth,
    simulate=lambda params, rng: simulate_lw_dataset(params, t=G2_SIM_T, rng=rng),
    stan_data=stan_data_no_sv,
    params=G2_PARAM_NAMES,
    bias_params=("sigma_g", "sigma_z"),
    n_datasets=G2_N_DATASETS,
    seed_base=G2_SEED_BASE,
)


# ---------------------------------------------------------------------------
# The registered suite
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "us_lw_sv"


def example_spec() -> RunSpec:
    from specs.schema.base import load_spec

    return load_spec(str(_EXAMPLE_DIR / "spec_sv.yaml"))


def example_df():
    from macrotoolkit.data import load_data

    df, _, _ = load_data(example_spec(), base_dir=_EXAMPLE_DIR)
    return df


def hd_reconstruction_error(spec: RunSpec, df, draw_index: int) -> float:
    """G6's shape for lw_sv at a prior point of the production SV render:
    a simulation-smoother draw -> results_lw's validated bars -> max
    |sum of gap bars - (y - y*)| and |sum of pi bars - pi|."""
    from macrotoolkit.families.lw_sv import build_render_context as _ctx
    from macrotoolkit.families.lw_sv import lw_mu_h0_anchors
    from macrotoolkit.families.lw_sv import sample_prior_params as _sample
    from macrotoolkit.results_lw import gap_pi_shock_decomposition, state_shock_decomposition
    from macrotoolkit.smoother import simulate_smoother_draw, sv_diag_variance_path, sv_rw_noncentered

    y, pi, r = (df[k].to_numpy(dtype=np.float64) for k in ("y", "pi", "r"))
    yobs, x = build_lw_regressors(y, pi, r)
    xi00, P00 = default_initial_state(float(y[4]))
    mu_is, mu_pc = lw_mu_h0_anchors(y, pi, r)
    rng = np.random.default_rng(20260924 + draw_index)
    # Stationary (a1, a2) only, as G6's own fixture points are: the
    # reconstruction is a linear-algebra identity, and evaluating it at an
    # explosive AR(2) draw (a third of the production prior's mass --
    # DECISIONS.md 2026-08-31) only measures float64 cancellation over
    # ~230 quarters, not the identity.
    priors = _ctx(spec)["priors"]
    for _ in range(10_000):
        params = _sample(priors, True, rng, mu_h0_is=mu_is, mu_h0_pc=mu_pc)
        if _is_stationary_ar2(params["a1"], params["a2"]):
            break
    else:
        raise RuntimeError("could not draw a stationary (a1, a2) pair in 10,000 attempts")
    T = yobs.shape[0]
    F, Q, A, Z, _ = build_lw_matrices({**params, "sigma_is": 1.0, "sigma_pc": 1.0}, c=1.0)
    R = sv_diag_variance_path(
        sv_rw_noncentered(params["h0_is"], params["sigma_h_is"], rng.standard_normal(T)),
        sv_rw_noncentered(params["h0_pc"], params["sigma_h_pc"], rng.standard_normal(T)),
    )
    sim = simulate_smoother_draw(yobs, x, F, Q, A, Z, R, xi00, P00, rng)
    comps = state_shock_decomposition(F, sim.xi_draw, sim.eps_ystar, sim.eps_g, sim.eps_z)
    gp = gap_pi_shock_decomposition(F, A, Z, x, y, pi, sim.eps_is, sim.eps_pc, comps)
    gap_err = np.max(np.abs(sum(gp["gap"].values()) - (yobs[:, 0] - sim.xi_draw[:, LW_STATE_META.slot("ystar", 0)])))
    pi_err = np.max(np.abs(sum(gp["pi"].values()) - yobs[:, 1]))
    return float(max(gap_err, pi_err))


def _suite():
    from macrotoolkit.validation.suite import (
        ValidationSuite,
        hd_identity_gate,
        mirror_gate,
        recovery_gate,
        sbc_gate,
    )

    return ValidationSuite(
        family="lw_sv",
        gates=(
            mirror_gate("lw_sv", example_spec, example_df),
            hd_identity_gate("lw_sv", example_spec, example_df, hd_reconstruction_error),
            recovery_gate(LW_G2_DESIGN),
            sbc_gate(G4_DESIGN, p_floor=CHI2_P_FLOOR, divergence_limit=DIVERGENT_TOTAL_LIMIT),
        ),
    )


VALIDATION_SUITE = _suite()
