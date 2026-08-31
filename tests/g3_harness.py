"""G3 support: exact-prior sampler + state-space simulator for SBC on the
no-SV LW variant (lw-sv-spec.md §5, G3 row: "SBC, no-SV variant | uniform
rank statistics (visual + chi^2 check)"). Support module -- no `test_*`
functions here; the gate lives in tests/test_g3_sbc.py.

Why this module exists separately from g1/g2_harness: SBC is a *calibration*
test, so the simulator must draw from EXACTLY the generative model the Stan
program defines -- prior included. Two deliberate differences from the G1/G2
machinery follow (both would corrupt SBC ranks if reused as-is):

1. **No stationarity rejection on (a1, a2) -- and SBC-specific (a1, a2)
   prior values.** `g1_harness.generate_parameter_points` resamples
   non-stationary (a1, a2) pairs, but the Stan program's prior is a plain
   unconstrained normal on both, so SBC must not filter draws -- and under
   the PRODUCTION defaults N(1.2, 0.3^2) x N(-0.4, 0.3^2) about a third of
   prior mass is non-stationary, which is numerically fatal: explosive
   draws simulate data up to |y| ~ 1e13 at T = 120, where the KF's
   covariance update (P entries ~ 1e26) loses everything to float64
   cancellation and ranks come out garbage (measured, 2026-08-31 smoke:
   every rank at 0/99, `cholesky_decompose` not-PD). Per the user decision
   of 2026-08-31 (DECISIONS.md), G3 therefore runs with
   SBC_PRIOR_OVERRIDES on a1/a2 -- applied through the PRODUCTION prior-
   override mechanism on the fit side and to `draw_exact_prior` here, so
   the two priors stay exactly equal and no rejection sampling is needed
   anywhere for a1/a2 (residual non-stationary mass ~3e-5). `a_r`/`b_y`
   rejection sampling stays: those Stan parameters ARE constrained
   (<upper=0>/<lower=0>), and rejection from the untruncated normal equals
   Stan's renormalized truncated prior.

2. **Simulation from the state-space form with xi_0 ~ N(xi00, P00), fixed
   conditioning data.** The Stan model is the KF marginal likelihood given
   (xi00, P00) and exogenous rows; its generative counterpart draws the
   initial state from that explicit prior and iterates (F, Q, A, Z, R) --
   NOT g2_harness's structural-equation simulator, whose burn-in +
   point-initialization scheme is a different (if similar) initial-state
   distribution. The recursive x_t (lagged y/pi) is built exactly as
   `build_lw_regressors` will rebuild it at fit time; r is exogenous.

   The production runtime anchors xi00 at the first estimation observation
   (`default_initial_state(y[4])`) -- a data-dependent convenience the
   simulator cannot reproduce (xi00 is needed before y[4] exists). Since the
   Stan program takes (xi00, P00) as plain data, SBC fixes the anchor at
   Y_ANCHOR for BOTH simulation and fit; this validates the same program the
   runtime runs, at a fixed rather than data-chosen anchor.

Conditioning data (pre-sample y/pi rows, the full exogenous r path) is fixed
once, module-level, from its own seed -- calibration must hold for any fixed
conditioning data, and fixing it keeps replication i's randomness fully
described by G3_SEED_BASE + i.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from macrotoolkit.smoother import build_lw_matrices, default_initial_state
from specs.schema.lw_sv import DEFAULT_PRIORS

#: Replication count -- user-confirmed 2026-08-31 (DECISIONS.md): 200 full
#: NUTS fits (~7 h under the `slow` marker), giving 10 expected counts per
#: chi^2 bin with RANK_BINS = 20.
N_REPLICATIONS = 200

#: Posterior draws are thinned to this many (evenly spaced) before ranking,
#: so ranks take values 0..RANK_DRAWS (RANK_DRAWS + 1 = 100 values) and
#: thinning strips most autocorrelation (SBC's uniformity result assumes
#: approximately independent draws).
RANK_DRAWS = 99

#: Histogram/chi^2 bins over the 100 rank values: 20 bins x 5 values each.
RANK_BINS = 20

#: Estimation-sample length (plus the 4 lag quarters); matches G2's SIM_T
#: so the fit cost and identification are the known quantities.
SIM_T = 120

#: Seed base: replication i draws its parameter point, initial state, shock
#: and measurement noise from default_rng(G3_SEED_BASE + i) and fits with
#: sampler seed G3_SEED_BASE + i, so any single replication reproduces in
#: isolation.
G3_SEED_BASE = 20260901

#: Seed for the fixed conditioning data (distinct so it is not correlated
#: with any replication's draws).
CONDITIONING_SEED = 20260900

#: Fixed y*_0 anchor for xi00 (see module docstring; 100*ln(GDP) scale,
#: same magnitude g2_harness uses).
Y_ANCHOR = 900.0

#: The 10 static parameters, in g1_harness.PARAM_NAMES order (kept as our
#: own tuple so the two modules can drift only loudly, via the test
#: asserting equality).
PARAM_NAMES = (
    "a1", "a2", "a_r", "b_pi", "b_y",
    "sigma_ystar", "sigma_g", "sigma_z", "sigma_is", "sigma_pc",
)

#: G3's (a1, a2) prior override (user decision 2026-08-31, DECISIONS.md; see
#: module docstring point 1): keeps the gap persistent (a1 + a2 ~ N(0.55,
#: 0.112^2)) while putting the stationarity boundary 4 sigma out, so the
#: exact prior is simulable in float64 with NO rejection filtering. Applied
#: identically to the fit (via the production `priors:` override path) and
#: to `draw_exact_prior` below.
SBC_PRIOR_OVERRIDES: dict[str, dict] = {
    "a1": {"mu": 0.8, "sd": 0.1},
    "a2": {"mu": -0.25, "sd": 0.05},
}

#: The full prior G3 samples and fits: production defaults + the override.
SBC_PRIORS: dict[str, dict] = {
    name: {**entry, **SBC_PRIOR_OVERRIDES.get(name, {})}
    for name, entry in DEFAULT_PRIORS.items()
}


def draw_exact_prior(rng: np.random.Generator) -> dict:
    """One draw from EXACTLY the prior the G3 Stan render fits (SBC_PRIORS =
    production defaults + the a1/a2 override, plus the template's parameter
    constraints). See module docstring for why (a1, a2) is neither
    stationarity-filtered here nor at its production default values."""
    max_attempts = 10_000

    a_r = None
    for _ in range(max_attempts):
        cand = rng.normal(SBC_PRIORS["a_r"]["mu"], SBC_PRIORS["a_r"]["sd"])
        if cand < 0.0:
            a_r = cand
            break
    if a_r is None:
        raise RuntimeError("Failed to draw a_r < 0 within 10_000 attempts.")

    b_y = None
    for _ in range(max_attempts):
        cand = rng.normal(SBC_PRIORS["b_y"]["mu"], SBC_PRIORS["b_y"]["sd"])
        if cand > 0.0:
            b_y = cand
            break
    if b_y is None:
        raise RuntimeError("Failed to draw b_y > 0 within 10_000 attempts.")

    return {
        "a1": float(rng.normal(SBC_PRIORS["a1"]["mu"], SBC_PRIORS["a1"]["sd"])),
        "a2": float(rng.normal(SBC_PRIORS["a2"]["mu"], SBC_PRIORS["a2"]["sd"])),
        "a_r": float(a_r),
        "b_pi": float(rng.beta(SBC_PRIORS["b_pi"]["a"], SBC_PRIORS["b_pi"]["b"])),
        "b_y": float(b_y),
        "sigma_ystar": abs(float(rng.normal(0.0, SBC_PRIORS["sigma_ystar"]["sd"]))),
        "sigma_g": abs(float(rng.normal(0.0, SBC_PRIORS["sigma_g"]["sd"]))),
        "sigma_z": abs(float(rng.normal(0.0, SBC_PRIORS["sigma_z"]["sd"]))),
        "sigma_is": abs(float(rng.normal(0.0, SBC_PRIORS["sigma_is"]["sd"]))),
        "sigma_pc": abs(float(rng.normal(0.0, SBC_PRIORS["sigma_pc"]["sd"]))),
    }


@dataclass(frozen=True)
class SbcConditioning:
    """The fixed, replication-independent data every SBC fit conditions on:
    4 pre-sample rows of (y, pi), the full exogenous r path, and the
    explicit initial-state prior (xi00, P00)."""

    y_pre: np.ndarray   # (4,)
    pi_pre: np.ndarray  # (4,)
    r: np.ndarray       # (SIM_T + 4,)
    xi00: np.ndarray    # (7,)
    P00: np.ndarray     # (7, 7)


def _build_conditioning(t: int = SIM_T, seed: int = CONDITIONING_SEED) -> SbcConditioning:
    """Plausible-magnitude fixed conditioning data (spec §1.1 units): y on
    the 100*ln(GDP) scale growing ~3% annualized into the Y_ANCHOR, pi
    around 2, r a persistent AR(1) around 3 (the initial-state prior's
    g_0 + z_0 mean, so early rate gaps are moderate)."""
    rng = np.random.default_rng(seed)
    y_pre = Y_ANCHOR - 0.75 * np.arange(4, 0, -1)  # ~3% annualized growth
    pi_pre = 2.0 + rng.normal(0.0, 0.3, size=4)
    r = np.empty(t + 4)
    dev = 0.0
    for i in range(t + 4):
        dev = 0.8 * dev + rng.normal(0.0, 0.6)
        r[i] = 3.0 + dev
    xi00, P00 = default_initial_state(Y_ANCHOR)
    return SbcConditioning(y_pre=y_pre, pi_pre=pi_pre, r=r, xi00=xi00, P00=P00)


#: Module-level constant: byte-identical across runs and replications.
CONDITIONING = _build_conditioning()


def simulate_from_state_space(
    params: dict, cond: SbcConditioning, rng: np.random.Generator, t: int = SIM_T
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate one dataset from the exact model the Stan program evaluates:
    xi_0 ~ N(xi00, P00); xi_t = F xi_{t-1} + w_t with w_t assembled from the
    three structural state shocks (reproducing Q's cross term exactly);
    yobs_t = A' x_t + Z xi_t + e_t, e_t ~ N(0, R), with x_t built from the
    fixed pre-sample rows + the recursion's own lagged output + the fixed r
    path -- the same construction build_lw_regressors performs at fit time.

    Returns (y, pi, r) as full (t+4,)-length arrays (4 lag rows included),
    ready for build_lw_regressors.
    """
    c = params.get("c", 1.0)
    F, Q, A, Z, R = build_lw_matrices(params, c=c)

    y = np.empty(t + 4)
    pi = np.empty(t + 4)
    y[:4] = cond.y_pre
    pi[:4] = cond.pi_pre
    r = cond.r.copy()

    xi = rng.multivariate_normal(cond.xi00, cond.P00)
    s_ystar, s_g, s_z = params["sigma_ystar"], params["sigma_g"], params["sigma_z"]
    s_is, s_pc = params["sigma_is"], params["sigma_pc"]

    for i in range(4, t + 4):
        eps_ystar = rng.normal(0.0, s_ystar)
        eps_g = rng.normal(0.0, s_g)
        eps_z = rng.normal(0.0, s_z)
        w = np.zeros(7)
        w[0] = eps_ystar + eps_g / 4.0  # y* shock includes eps_g/4 (Q[0,0], Q[0,3])
        w[3] = eps_g
        w[5] = eps_z
        xi = F @ xi + w

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
        e = np.array([rng.normal(0.0, s_is), rng.normal(0.0, s_pc)])
        obs = A.T @ x_t + Z @ xi + e
        y[i] = obs[0]
        pi[i] = obs[1]

    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(pi))):
        raise FloatingPointError(
            f"Simulated dataset is non-finite (params={params!r}) -- an "
            f"explosive (a1, a2) draw beyond float64 range, which should be "
            f"~impossible under the prior. Investigate rather than skip."
        )
    return y, pi, r


def rank_statistic(theta_true: float, draws_thinned: np.ndarray) -> int:
    """SBC rank: the number of thinned posterior draws strictly below the
    prior draw. Uniform on {0, ..., len(draws_thinned)} under calibration."""
    return int(np.sum(draws_thinned < theta_true))


def thin_evenly(draws: np.ndarray, k: int = RANK_DRAWS) -> np.ndarray:
    """k evenly spaced draws from the pooled posterior sample (strips the
    bulk of the autocorrelation SBC's uniformity result assumes away)."""
    n = draws.shape[0]
    if n < k:
        raise ValueError(f"Need at least {k} posterior draws, got {n}.")
    idx = np.linspace(0, n - 1, k).round().astype(int)
    return draws[idx]
