"""G1 mirror-comparison harness (lw-sv-spec.md §5, G1 row): "Python KF vs
Stan KF log-likelihood, 50 random parameter points | max abs diff < 1e-8."

This is a **support module**, not itself a test file pytest collects tests
from (no `test_*` functions live here). Built as S2 prep against stubs; S2
(2026-08-31) replaced the stubs with the real wiring: `python_kf_loglik`
calls `macrotoolkit.smoother`'s KF mirror, and `stan_kf_loglik_batch`
compiles + runs the G1 Stan harness program
(`stan/templates/g1_loglik_harness.stan.j2`) once for all points.

--------------------------------------------------------------------------
Parameter-point generator
--------------------------------------------------------------------------

Samples the no-SV LW model's static parameters from lw-sv-spec.md §1.6's
priors table, restricted to the no-SV variant: constant `sigma_y*, sigma_g,
sigma_z`; no SV/h-path parameters (`sigma_h,IS`, `sigma_h,PC`, `mu_h0,s`) at
all, since those don't exist outside the SV variant. `c` is fixed at 1.0
(spec's default `estimate_c: false`), not sampled.

The no-SV variant's constant IS/PC shock scales (`sigma_is`, `sigma_pc`)
are not in spec §1.6's table; their prior -- Half-N(0, 1^2), sampled here
alongside the 8 table parameters -- is the 2026-08-31 user-confirmed
decision recorded in DECISIONS.md and encoded in
`specs.schema.lw_sv.DEFAULT_PRIORS`, which this generator reads so the two
never drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from specs.schema.lw_sv import DEFAULT_PRIORS

# ---------------------------------------------------------------------------
# Constants shared by the harness and by tests/test_g1_mirror.py
# ---------------------------------------------------------------------------

#: The 10 no-SV static parameters this harness samples, in the column order
#: the G1 Stan harness (stan/templates/g1_loglik_harness.stan.j2) expects.
PARAM_NAMES = (
    "a1", "a2", "a_r", "b_pi", "b_y",
    "sigma_ystar", "sigma_g", "sigma_z", "sigma_is", "sigma_pc",
)

#: `c` (spec §1.3) fixed at 1.0 -- spec default `estimate_c: false`.
C_FIXED = 1.0

#: G1's own gate parameters (lw-sv-spec.md §5): 50 points, tolerance 1e-8.
N_PARAM_POINTS = 50
LOGLIK_TOL = 1e-8

#: Fixed seed for the parameter-point generator, recorded here so any G1
#: failure is reproducible by re-running `generate_parameter_points(50,
#: seed=PARAM_SEED)` exactly. Matches the repo-wide convention (spec §2.3's
#: example `sampler.seed: 20260813`) rather than an arbitrary choice.
PARAM_SEED = 20260813

#: Fixed seed for the synthetic dataset (see `_generate_synthetic_data`
#: below). Deliberately distinct from PARAM_SEED so the two axes of
#: randomness (parameter points vs. data) are independently reproducible.
DATA_SEED = 20260814

#: Length (in quarters) of the synthetic dataset.
SYNTHETIC_T = 50


def _is_stationary_ar2(a1: float, a2: float) -> bool:
    """AR(2) stationarity triangle for `gap_t = a1*gap_{t-1} + a2*gap_{t-2}
    + ...` (spec §1.2): `a2 in (-1, 1)`, `a1 + a2 < 1`, `a2 - a1 < 1`."""
    return (-1.0 < a2 < 1.0) and (a1 + a2 < 1.0) and (a2 - a1 < 1.0)


def generate_parameter_points(n: int, seed: int) -> list[dict]:
    """Draw `n` parameter points from lw-sv-spec.md §1.6's priors (no-SV
    variant). Each point is a `dict` with keys `PARAM_NAMES` plus `"c"`
    (always `C_FIXED`).

    (a1, a2) pairs that fail the AR(2) stationarity triangle are rejected
    and resampled (both jointly redrawn), per the spec's "joint stationarity
    check via prior predictive" note. `a_r` and `b_y` are likewise drawn by
    rejection sampling from their (untruncated) normal priors, restricted to
    `a_r < 0` and `b_y > 0` respectively, matching the spec's "truncated"
    prior wording literally (as opposed to e.g. a folded/reflected normal).

    Uses `np.random.default_rng(seed)` (not the legacy global numpy RNG) so
    a given `(n, seed)` pair is exactly reproducible -- record the seed used
    alongside any reported G1 failure.
    """
    rng = np.random.default_rng(seed)
    max_attempts = 10_000
    points: list[dict] = []

    while len(points) < n:
        a1 = a2 = None
        for _ in range(max_attempts):
            cand_a1 = rng.normal(1.2, 0.3)
            cand_a2 = rng.normal(-0.4, 0.3)
            if _is_stationary_ar2(cand_a1, cand_a2):
                a1, a2 = cand_a1, cand_a2
                break
        if a1 is None:
            raise RuntimeError(
                f"Failed to draw a stationary (a1, a2) pair from "
                f"N(1.2, 0.3^2) x N(-0.4, 0.3^2) within {max_attempts} "
                f"attempts -- check the priors or the rejection region."
            )

        a_r = None
        for _ in range(max_attempts):
            cand = rng.normal(-0.1, 0.05)
            if cand < 0.0:
                a_r = cand
                break
        if a_r is None:
            raise RuntimeError(
                f"Failed to draw a_r < 0 from N(-0.1, 0.05^2) within "
                f"{max_attempts} attempts."
            )

        b_pi = float(rng.beta(8.0, 2.0))

        b_y = None
        for _ in range(max_attempts):
            cand = rng.normal(0.15, 0.1)
            if cand > 0.0:
                b_y = cand
                break
        if b_y is None:
            raise RuntimeError(
                f"Failed to draw b_y > 0 from N(0.15, 0.1^2) within "
                f"{max_attempts} attempts."
            )

        # Half-Normal(0, scale^2) == |N(0, scale^2)|. Scales come from
        # specs.schema.lw_sv.DEFAULT_PRIORS (the single source of truth for
        # the §1.6 defaults + the confirmed sigma_is/sigma_pc entry).
        sigma_ystar = abs(float(rng.normal(0.0, DEFAULT_PRIORS["sigma_ystar"]["sd"])))
        sigma_g = abs(float(rng.normal(0.0, DEFAULT_PRIORS["sigma_g"]["sd"])))
        sigma_z = abs(float(rng.normal(0.0, DEFAULT_PRIORS["sigma_z"]["sd"])))
        sigma_is = abs(float(rng.normal(0.0, DEFAULT_PRIORS["sigma_is"]["sd"])))
        sigma_pc = abs(float(rng.normal(0.0, DEFAULT_PRIORS["sigma_pc"]["sd"])))

        points.append(
            {
                "a1": float(a1),
                "a2": float(a2),
                "a_r": float(a_r),
                "b_pi": b_pi,
                "b_y": float(b_y),
                "sigma_ystar": sigma_ystar,
                "sigma_g": sigma_g,
                "sigma_z": sigma_z,
                "sigma_is": sigma_is,
                "sigma_pc": sigma_pc,
                "c": C_FIXED,
            }
        )

    return points


# ---------------------------------------------------------------------------
# Fixed synthetic dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticKFData:
    """A small, fixed synthetic quarterly panel for exercising KF algebra
    only. **Not an economic fixture** -- no structural relationship is
    assumed or guaranteed to hold between `y`, `pi`, and `r` (unlike
    `tests/fixtures/hlw/`, which *is* real published data and is the G5a
    oracle). Only plausible magnitudes matter here, so that G1's KF
    comparison exercises realistic-scale numbers rather than degenerate
    ones."""

    dates: pd.DatetimeIndex
    y: np.ndarray
    pi: np.ndarray
    r: np.ndarray

    @property
    def T(self) -> int:  # noqa: N802 -- matches spec's own "T" notation
        return len(self.y)


def _generate_synthetic_data(t: int = SYNTHETIC_T, seed: int = DATA_SEED) -> SyntheticKFData:
    """Build the fixed synthetic dataset once, deterministically. Magnitudes
    per this task's brief:
    - `y`: log-GDP-like, ~8-10 with a mild trend + noise (spec §1.1: `y_t =
      100*ln(real GDP)`, so values in the 8-10 range are the right order of
      magnitude for a log-GDP-like series in log-points)
    - `pi`: ~0-4 (spec §1.1: annualized q/q %)
    - `r`: ~0-6 (spec §1.1: annualized %)
    """
    rng = np.random.default_rng(seed)
    trend = np.linspace(8.0, 9.6, t)
    y = trend + rng.normal(0.0, 0.05, size=t)
    pi = np.clip(2.0 + rng.normal(0.0, 0.8, size=t), 0.0, 4.0)
    r = np.clip(3.0 + rng.normal(0.0, 1.2, size=t), 0.0, 6.0)
    dates = pd.date_range("2000-01-01", periods=t, freq="QS")
    return SyntheticKFData(dates=dates, y=y, pi=pi, r=r)


#: Module-level constant: generated once, at import time, with a fixed seed,
#: so every test run (and every future S2 wiring of the real KF functions)
#: sees byte-identical data.
SYNTHETIC_DATA = _generate_synthetic_data()


# ---------------------------------------------------------------------------
# Real log-likelihood functions (the S2 wiring that replaced this file's
# original pre-S2 stubs)
# ---------------------------------------------------------------------------


def python_kf_loglik(params: dict, data: SyntheticKFData) -> float:
    """The Python KF log-likelihood mirror (`macrotoolkit.smoother`) at one
    parameter point on the synthetic dataset, with the family's default
    initial state (anchored at the first estimation-sample observation)."""
    from macrotoolkit.smoother import (
        build_lw_matrices,
        build_lw_regressors,
        default_initial_state,
        kalman_loglik,
    )

    yobs, x = build_lw_regressors(data.y, data.pi, data.r)
    F, Q, A, Z, R = build_lw_matrices(params, c=params.get("c", C_FIXED))
    xi00, P00 = default_initial_state(float(data.y[4]))
    return kalman_loglik(yobs, x, F, Q, A, Z, R, xi00, P00)


def stan_kf_loglik_batch(points: list[dict], data: SyntheticKFData) -> np.ndarray:
    """Evaluate the Stan-side KF log-likelihood at every parameter point in
    one compile + one fixed_param run of the G1 harness program
    (stan/templates/g1_loglik_harness.stan.j2), which builds the system
    matrices with the same shared `stan/functions/` library the lw_sv
    template uses. Returns the vector of log-likelihoods in `points` order.

    `sig_figs=18`: CmdStan writes draws as CSV with 6 significant figures by
    default, which alone would exceed G1's 1e-8 tolerance for any
    |loglik| > ~1e-2 -- full precision output is load-bearing here, not an
    optimization.
    """
    from macrotoolkit.render import compile_model, render_stan_source
    from macrotoolkit.smoother import build_lw_regressors, default_initial_state

    yobs, x = build_lw_regressors(data.y, data.pi, data.r)
    xi00, P00 = default_initial_state(float(data.y[4]))
    pmat = np.array([[p[k] for k in PARAM_NAMES] for p in points])

    source = render_stan_source("g1_loglik_harness.stan.j2", {})
    model, _ = compile_model(source)
    fit = model.sample(
        data={
            "T": yobs.shape[0],
            "yobs": yobs,
            "x": x,
            "xi00": xi00,
            "P00": P00,
            "c": C_FIXED,
            "n_points": len(points),
            "params": pmat,
        },
        fixed_param=True,
        chains=1,
        iter_sampling=1,
        seed=PARAM_SEED,
        sig_figs=18,
        show_progress=False,
    )
    return fit.stan_variable("loglik")[0]


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------


def compare_loglik(
    params: dict,
    data: SyntheticKFData,
    python_fn: Callable[[dict, SyntheticKFData], float],
    stan_fn: Callable[[dict, SyntheticKFData], float],
    tol: float = LOGLIK_TOL,
) -> float:
    """Evaluate `python_fn` and `stan_fn` at the same `(params, data)` and
    assert they agree to within `tol` (G1's own tolerance, spec §5).

    Dependency-injected on purpose: `python_fn`/`stan_fn` are passed in
    rather than imported here, so S2 can call this with the real
    `macrotoolkit.smoother` KF log-likelihood and the real Stan KF
    log-likelihood without touching this file at all.

    Returns the absolute difference on success. Raises `AssertionError`
    (naming both values, the diff, the tolerance, and the offending
    `params`) on failure -- this is a gate, not a report, so a mismatch must
    fail loudly rather than be logged and passed through.
    """
    ll_python = python_fn(params, data)
    ll_stan = stan_fn(params, data)
    diff = abs(ll_python - ll_stan)
    assert diff < tol, (
        f"G1 gate failed: |python_loglik - stan_loglik| = {diff:.3e} >= "
        f"tol={tol:.1e} at params={params!r}. "
        f"python_loglik={ll_python!r}, stan_loglik={ll_stan!r}."
    )
    return diff
