"""G1 mirror-comparison harness (lw-sv-spec.md §5, G1 row): "Python KF vs
Stan KF log-likelihood, 50 random parameter points | max abs diff < 1e-8."

This is a **support module**, not itself a test file pytest collects tests
from (no `test_*` functions live here). It exists so that when S2 lands
`macrotoolkit.smoother`'s KF log-likelihood mirror and a render+expose
harness for `stan/functions/kalman_loglik_tv.stan`, the actual G1 test
(`tests/test_g1_mirror.py`) only has to wire in two real functions -- the
parameter-point generator, synthetic dataset, and comparison utility are
already built and already tested here.

Scope note: per this task's brief, this file is prep for S2, not S2 itself.
It must not import or depend on `macrotoolkit.smoother` or any Stan function
that doesn't exist yet -- see `python_kf_loglik_stub` / `stan_kf_loglik_stub`
below, which raise `NotImplementedError` naming exactly what S2 needs to
build.

--------------------------------------------------------------------------
Parameter-point generator
--------------------------------------------------------------------------

Samples the no-SV LW model's static parameters from lw-sv-spec.md §1.6's
priors table, restricted to the no-SV variant: constant `sigma_y*, sigma_g,
sigma_z`; no SV/h-path parameters (`sigma_h,IS`, `sigma_h,PC`, `mu_h0,s`) at
all, since those don't exist outside the SV variant. `c` is fixed at 1.0
(spec's default `estimate_c: false`), not sampled.

Note on scope: spec §1.6's prior table has no entry for a constant IS/PC
shock scale -- in the full SV model those are entirely determined by the SV
path (`exp(h/2)`); the no-SV variant's equivalent constant-variance
parameters for the IS and Phillips-curve shocks aren't specified anywhere in
the spec's priors table. This harness therefore samples exactly the 8
parameters the spec priors table gives values for (`a1, a2, a_r, b_pi, b_y,
sigma_ystar, sigma_g, sigma_z`), per this task's explicit brief. If S2's real
KF functions need additional constant IS/PC shock-scale parameters to fully
specify Q_t for the no-SV variant, that is a decision for whoever wires the
real functions in here -- flagged, not resolved, by this prep pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants shared by the harness and by tests/test_g1_mirror.py
# ---------------------------------------------------------------------------

#: The 8 no-SV static parameters this harness samples (see module docstring
#: for why the list stops here and doesn't include IS/PC shock scales).
PARAM_NAMES = ("a1", "a2", "a_r", "b_pi", "b_y", "sigma_ystar", "sigma_g", "sigma_z")

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

        # Half-Normal(0, scale^2) == |N(0, scale^2)|.
        sigma_ystar = abs(float(rng.normal(0.0, 0.4)))
        sigma_g = abs(float(rng.normal(0.0, 0.03)))
        sigma_z = abs(float(rng.normal(0.0, 0.08)))

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
# Stub log-likelihood functions (S2 replaces these with the real thing)
# ---------------------------------------------------------------------------


def python_kf_loglik_stub(params: dict, data: SyntheticKFData) -> float:
    """Stand-in for the not-yet-written Python KF log-likelihood mirror.

    S2 must implement this as (or wire this call site directly to) a KF
    log-likelihood function inside `macrotoolkit.smoother` (per
    lw-sv-spec.md §2.4's "the Python KF log-likelihood must match the Stan
    KF log-likelihood" requirement, and plans/S2-plan.md's file list:
    `src/macrotoolkit/smoother.py`, S2 scope limited to the KF log-likelihood
    only -- the Durbin-Koopman simulation smoother itself is S4 scope, do
    not front-load it here).
    """
    raise NotImplementedError(
        "python_kf_loglik_stub: awaiting macrotoolkit.smoother's KF "
        "log-likelihood function (src/macrotoolkit/smoother.py, S2 scope; "
        "not yet written). Replace this stub with the real function (or "
        "call it directly from tests/test_g1_mirror.py) once it exists."
    )


def stan_kf_loglik_stub(params: dict, data: SyntheticKFData) -> float:
    """Stand-in for the not-yet-written Stan KF log-likelihood, exposed to
    Python.

    S2 must implement `stan/functions/kalman_loglik_tv.stan` (lw-sv-spec.md
    §2.2) and a thin render+expose harness around it -- e.g. a small `.stan`
    program that `#include`s the function and exposes it via CmdStan's
    standalone-function interface (or an equivalent mechanism) -- so it can
    be called from Python with the same `(params, data) -> float` signature
    as the Python mirror.
    """
    raise NotImplementedError(
        "stan_kf_loglik_stub: awaiting stan/functions/kalman_loglik_tv.stan "
        "(S2 scope; not yet written) plus a render+expose harness that "
        "calls it from Python. Replace this stub with the real function (or "
        "call it directly from tests/test_g1_mirror.py) once both exist."
    )


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
