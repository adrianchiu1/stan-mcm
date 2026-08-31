"""G2 support: synthetic-data simulator for the no-SV LW model
(lw-sv-spec.md §5, G2 row: "Parameter recovery on simulated data (no SV),
20 datasets"). Support module -- no `test_*` functions here; the gate lives
in tests/test_g2_parameter_recovery.py.

The simulator draws state and shock paths directly from the structural
equations (spec §1.2-§1.3) at a given parameter point -- deliberately NOT
through the state-space matrices, so G2 exercises an independent encoding
of the model from the one the KF likelihood uses (a transcription error in
either shows up as failed recovery, rather than cancelling out).

Distinct from G1's machinery on purpose (plans/S2-plan.md open question 3):
G1's generator samples parameter points only; this module simulates data
given a point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Number of simulated datasets (spec §5's G2 row).
N_DATASETS = 20

#: Estimation-sample length per dataset (plus the 4 lag quarters the data
#: convention requires). Shorter than the US sample to keep 20 estimations
#: affordable, long enough for the trends to be identified.
SIM_T = 120

#: Pre-sample burn-in quarters discarded so initial conditions wash out of
#: the stationary components (gap, inflation deviations).
BURN_IN = 40

#: Seed base: dataset i is simulated with SIM_SEED_BASE + i, so any failing
#: dataset is reproducible in isolation.
SIM_SEED_BASE = 20260831


@dataclass(frozen=True)
class SimulatedLwData:
    """One simulated no-SV LW dataset: observable series INCLUDING the 4
    pre-sample lag quarters (same convention as build_lw_regressors), plus
    the true latent paths for reference/debugging."""

    y: np.ndarray       # (T+4,) 100*ln(GDP)-scale observable
    pi: np.ndarray      # (T+4,) annualized q/q % inflation
    r: np.ndarray       # (T+4,) annualized % ex-ante real rate (exogenous)
    ystar: np.ndarray   # (T+4,) true potential
    g: np.ndarray       # (T+4,) true trend growth, annualized
    z: np.ndarray       # (T+4,) true z
    gap: np.ndarray     # (T+4,) true output gap


def simulate_lw_dataset(params: dict, t: int = SIM_T, seed: int = SIM_SEED_BASE) -> SimulatedLwData:
    """Simulate `t` estimation quarters (+4 lag quarters) of the no-SV LW
    model at `params` (keys as in g1_harness.PARAM_NAMES, plus optional
    "c", default 1.0).

    The exogenous real rate follows a persistent AR(1) around the r* path
    (r_t = r*_t + stationary deviation), so the IS curve's real-rate gap
    regressor has realistic variation without exploding.
    """
    rng = np.random.default_rng(seed)
    c = params.get("c", 1.0)
    a1, a2, a_r = params["a1"], params["a2"], params["a_r"]
    b_pi, b_y = params["b_pi"], params["b_y"]

    n = BURN_IN + 4 + t

    # Latent trends: random walks (g, z annualized; y* steps by g/4).
    g = np.empty(n)
    z = np.empty(n)
    ystar = np.empty(n)
    g[0] = 3.0
    z[0] = 0.0
    ystar[0] = 900.0  # 100*ln(GDP) scale
    eps_g = rng.normal(0.0, params["sigma_g"], size=n)
    eps_z = rng.normal(0.0, params["sigma_z"], size=n)
    eps_ystar = rng.normal(0.0, params["sigma_ystar"], size=n)
    for i in range(1, n):
        g[i] = g[i - 1] + eps_g[i]
        z[i] = z[i - 1] + eps_z[i]
        ystar[i] = ystar[i - 1] + g[i - 1] / 4.0 + eps_ystar[i]

    rstar = c * g + z

    # Exogenous real rate: r* plus a persistent stationary deviation.
    r = np.empty(n)
    dev = 0.0
    for i in range(n):
        dev = 0.8 * dev + rng.normal(0.0, 0.8)
        r[i] = rstar[i] + dev

    # Output gap: AR(2) + real-rate-gap term (spec §1.2's IS curve).
    gap = np.zeros(n)
    eps_is = rng.normal(0.0, params["sigma_is"], size=n)
    for i in range(2, n):
        rate_gap_term = (a_r / 2.0) * ((r[i - 1] - rstar[i - 1]) + (r[i - 2] - rstar[i - 2]))
        gap[i] = a1 * gap[i - 1] + a2 * gap[i - 2] + rate_gap_term + eps_is[i]

    # Inflation: sum-to-one lag structure + lagged gap (spec §1.2's PC).
    pi = np.full(n, 2.0)
    eps_pc = rng.normal(0.0, params["sigma_pc"], size=n)
    for i in range(4, n):
        pibar = (pi[i - 2] + pi[i - 3] + pi[i - 4]) / 3.0
        pi[i] = b_pi * pi[i - 1] + (1.0 - b_pi) * pibar + b_y * gap[i - 1] + eps_pc[i]

    y = ystar + gap

    keep = slice(BURN_IN, n)
    return SimulatedLwData(
        y=y[keep], pi=pi[keep], r=r[keep],
        ystar=ystar[keep], g=g[keep], z=z[keep], gap=gap[keep],
    )
