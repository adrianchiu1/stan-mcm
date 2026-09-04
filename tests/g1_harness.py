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

#: Seed + scale for the per-point log-variance paths exercising the
#: time-varying-R_t KF (S3 generalization). The scale matches the SV
#: prior's sigma_h ~ Half-N(0, 0.2^2) magnitude so paths are the size the
#: production model produces.
H_PATH_SEED = 20260901
H_PATH_RW_SD = 0.2

#: Seed for the per-point (h0, sigma_h, nu) inputs exercising the
#: PRODUCTION SV helpers (sv_rw_noncentered / sv_diag_variance_path) on
#: both sides of the mirror. Distinct from H_PATH_SEED so the two
#: time-varying comparisons don't share randomness.
SV_INPUT_SEED = 20260902

#: Seeds for the S6 Q_t paths: per-point log-variance paths for the
#: TIME-VARYING STATE-innovation covariance (`loglik_tvq`: Q_t built
#: inline from a per-period scale on the g shock) and the production
#: composition (`loglik_svq`: sv_rw_noncentered -> a scalar variance path
#: scaling the z shock). Distinct seeds, same discipline as above.
Q_PATH_SEED = 20260903
Q_SV_INPUT_SEED = 20260905

#: Fixture of the PRE-S6 Stan harness's log-likelihoods at the 50 points
#: (captured on the untouched S5 baseline, 2026-09-04, before any stan/
#: edit): the constant-Q regression pin -- the generalized filter must
#: reproduce these to ~1e-10 (CmdStan's 18-sig-fig CSV output resolution).
PRE_QT_FIXTURE = "fixtures/g1/pre_qt_stan_loglik.csv"
PRE_QT_TOL = 1e-10


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


def generate_h_paths(points: list[dict], t: int, seed: int = H_PATH_SEED) -> np.ndarray:
    """Per-point log-VARIANCE paths for the time-varying-R_t G1 comparison:
    for each point, h_{s,1} = 2*ln(sigma_s) (so the path starts at the
    point's own constant-variant scale -- spec §1.5's h-is-log-variance
    convention) followed by random-walk increments of sd H_PATH_RW_SD.
    Returns (n_points, t, 2) with column 0 = IS, 1 = PC."""
    rng = np.random.default_rng(seed)
    h = np.empty((len(points), t, 2))
    for i, p in enumerate(points):
        start = np.array([2.0 * np.log(p["sigma_is"]), 2.0 * np.log(p["sigma_pc"])])
        increments = rng.normal(0.0, H_PATH_RW_SD, size=(t, 2))
        increments[0] = 0.0
        h[i] = start + np.cumsum(increments, axis=0)
    return h


def generate_sv_inputs(points: list[dict], t: int, seed: int = SV_INPUT_SEED) -> dict:
    """Per-point inputs for the SV-helper mirror comparison: realized
    initial log-variances h0 (anchored at each point's own scales, spec
    §1.5's h_0 ~ N(2*ln(sigma), 1)), RW scales sigma_h ~ |N(0, 0.2^2)| (the
    production prior's magnitude), and standard-normal innovations nu with
    shape (n_points, t, 2), column 0 = IS, 1 = PC."""
    rng = np.random.default_rng(seed)
    n = len(points)
    h0_is = np.array([2.0 * np.log(p["sigma_is"]) for p in points]) + rng.normal(0.0, 1.0, n)
    h0_pc = np.array([2.0 * np.log(p["sigma_pc"]) for p in points]) + rng.normal(0.0, 1.0, n)
    return {
        "h0_is": h0_is,
        "h0_pc": h0_pc,
        "sigma_h_is": np.abs(rng.normal(0.0, 0.2, n)),
        "sigma_h_pc": np.abs(rng.normal(0.0, 0.2, n)),
        "nu": rng.normal(0.0, 1.0, size=(n, t, 2)),
    }


def python_kf_loglik_sv(params: dict, sv: dict, i: int, data: SyntheticKFData) -> float:
    """Python mirror of the SV composition the production SV render uses:
    h = sv_rw_noncentered(h0, sigma_h, nu), R_t = sv_diag_variance_path(h_is,
    h_pc), then the time-varying KF."""
    from macrotoolkit.smoother import (
        build_lw_matrices,
        build_lw_regressors,
        default_initial_state,
        kalman_loglik,
        sv_diag_variance_path,
        sv_rw_noncentered,
    )

    yobs, x = build_lw_regressors(data.y, data.pi, data.r)
    F, Q, A, Z, _ = build_lw_matrices(params, c=params.get("c", C_FIXED))
    xi00, P00 = default_initial_state(float(data.y[4]))
    h_is = sv_rw_noncentered(float(sv["h0_is"][i]), float(sv["sigma_h_is"][i]), sv["nu"][i, :, 0])
    h_pc = sv_rw_noncentered(float(sv["h0_pc"][i]), float(sv["sigma_h_pc"][i]), sv["nu"][i, :, 1])
    return kalman_loglik(yobs, x, F, Q, A, Z, sv_diag_variance_path(h_is, h_pc), xi00, P00)


def generate_q_paths(points: list[dict], t: int, seed: int = Q_PATH_SEED) -> np.ndarray:
    """Per-point log-variance paths for the TIME-VARYING Q_t comparison
    (S6): for each point, hq_1 = 2*ln(sigma_g) followed by random-walk
    increments of sd H_PATH_RW_SD. The harness turns them into
    Q_t = lw_Q(sigma_ystar, exp(hq_t/2), sigma_z) -- the g shock's scale
    varying over time through the family's own Q constructor. Returns
    (n_points, t)."""
    rng = np.random.default_rng(seed)
    hq = np.empty((len(points), t))
    for i, p in enumerate(points):
        increments = rng.normal(0.0, H_PATH_RW_SD, size=t)
        increments[0] = 0.0
        hq[i] = 2.0 * np.log(p["sigma_g"]) + np.cumsum(increments)
    return hq


def generate_q_sv_inputs(points: list[dict], t: int, seed: int = Q_SV_INPUT_SEED) -> dict:
    """Per-point (h0, sigma_h, nu) inputs for the production Q_t SV
    composition (S6): the z shock's log-variance follows
    sv_rw_noncentered(h0_z, sigma_h_z, nu_z), anchored at each point's own
    sigma_z (h_0 ~ N(2*ln(sigma_z), 1)), turned into a (T, 1, 1) variance
    path by sv_scalar_variance_path and placed on the z slot of Q_t."""
    rng = np.random.default_rng(seed)
    n = len(points)
    return {
        "h0_z": np.array([2.0 * np.log(p["sigma_z"]) for p in points]) + rng.normal(0.0, 1.0, n),
        "sigma_h_z": np.abs(rng.normal(0.0, 0.2, n)),
        "nu_z": rng.normal(0.0, 1.0, size=(n, t)),
    }


def _lw_Q_path_from_g_logvar(params: dict, hq: np.ndarray) -> np.ndarray:
    """(T, 7, 7) Q path with the g shock scale exp(hq_t/2) at each t
    (h is log-VARIANCE: sd = exp(h/2)), every other entry constant --
    built through build_lw_matrices so the mirror uses the family's own
    constructor, exactly as the Stan harness calls lw_Q per period."""
    from macrotoolkit.smoother import build_lw_matrices

    T = hq.shape[0]
    Q_path = np.empty((T, 7, 7))
    for t in range(T):
        p_t = {**params, "sigma_g": float(np.exp(hq[t] / 2.0))}
        Q_path[t] = build_lw_matrices(p_t, c=params.get("c", C_FIXED))[1]
    return Q_path


def python_kf_loglik_tvq(params: dict, hq: np.ndarray, data: SyntheticKFData) -> float:
    """Python mirror of the TIME-VARYING STATE-innovation-covariance KF
    (S6 Q_t generalization) at one parameter point: Q_t from the point's
    g-shock log-variance path, constant R from the matrix builder."""
    from macrotoolkit.smoother import (
        build_lw_matrices,
        build_lw_regressors,
        default_initial_state,
        kalman_loglik,
    )

    yobs, x = build_lw_regressors(data.y, data.pi, data.r)
    F, _, A, Z, R = build_lw_matrices(params, c=params.get("c", C_FIXED))
    xi00, P00 = default_initial_state(float(data.y[4]))
    return kalman_loglik(yobs, x, F, _lw_Q_path_from_g_logvar(params, hq), A, Z, R, xi00, P00)


def python_kf_loglik_svq(params: dict, qsv: dict, i: int, data: SyntheticKFData) -> float:
    """Python mirror of the production Q_t SV composition (S6): the z
    shock's variance path exp(h_z,t) from sv_rw_noncentered ->
    sv_scalar_variance_path, placed on Q's z slot per period; constant R."""
    from macrotoolkit.families.lw_sv import LW_STATE_META
    from macrotoolkit.smoother import (
        build_lw_matrices,
        build_lw_regressors,
        default_initial_state,
        kalman_loglik,
        sv_rw_noncentered,
        sv_scalar_variance_path,
    )

    yobs, x = build_lw_regressors(data.y, data.pi, data.r)
    F, Q, A, Z, R = build_lw_matrices(params, c=params.get("c", C_FIXED))
    xi00, P00 = default_initial_state(float(data.y[4]))
    h_z = sv_rw_noncentered(float(qsv["h0_z"][i]), float(qsv["sigma_h_z"][i]), qsv["nu_z"][i])
    var_z = sv_scalar_variance_path(h_z)[:, 0, 0]
    z_slot = LW_STATE_META.slot("z", -1)
    T = yobs.shape[0]
    Q_path = np.repeat(Q[None, :, :], T, axis=0)
    Q_path[:, z_slot, z_slot] = var_z
    return kalman_loglik(yobs, x, F, Q_path, A, Z, R, xi00, P00)


def python_kf_loglik_tv(params: dict, h: np.ndarray, data: SyntheticKFData) -> float:
    """Python mirror of the TIME-VARYING measurement-covariance KF at one
    parameter point: R_t = diag(exp(h_t)) (h is log-variance) replacing the
    constant R from the matrix builder."""
    from macrotoolkit.smoother import (
        build_lw_matrices,
        build_lw_regressors,
        default_initial_state,
        kalman_loglik,
    )

    yobs, x = build_lw_regressors(data.y, data.pi, data.r)
    F, Q, A, Z, _ = build_lw_matrices(params, c=params.get("c", C_FIXED))
    xi00, P00 = default_initial_state(float(data.y[4]))
    T = yobs.shape[0]
    R_path = np.zeros((T, 2, 2))
    R_path[:, 0, 0] = np.exp(h[:, 0])
    R_path[:, 1, 1] = np.exp(h[:, 1])
    return kalman_loglik(yobs, x, F, Q, A, Z, R_path, xi00, P00)


def stan_kf_loglik_batch(
    points: list[dict],
    data: SyntheticKFData,
    h_paths: np.ndarray | None = None,
    sv_inputs: dict | None = None,
    q_paths: np.ndarray | None = None,
    q_sv_inputs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the Stan-side KF log-likelihoods at every parameter point in
    one compile + one fixed_param run of the G1 harness program
    (stan/templates/g1_loglik_harness.stan.j2), which builds the system
    matrices with the same shared `stan/functions/` library the lw_sv
    template uses. Returns `(loglik, loglik_tv, loglik_sv, loglik_tvq,
    loglik_svq)` in `points` order: the constant-Q/constant-R filter path,
    the time-varying-R_t path at `h_paths` (default:
    `generate_h_paths(points, T)`), the production-SV-helper composition at
    `sv_inputs` (default: `generate_sv_inputs(points, T)`), and -- S6 --
    the time-varying-Q_t path at `q_paths` (default `generate_q_paths`)
    and the production Q_t SV composition at `q_sv_inputs` (default
    `generate_q_sv_inputs`).

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
    if h_paths is None:
        h_paths = generate_h_paths(points, yobs.shape[0])
    if sv_inputs is None:
        sv_inputs = generate_sv_inputs(points, yobs.shape[0])
    if q_paths is None:
        q_paths = generate_q_paths(points, yobs.shape[0])
    if q_sv_inputs is None:
        q_sv_inputs = generate_q_sv_inputs(points, yobs.shape[0])

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
            "h": h_paths,
            **sv_inputs,
            "hq": q_paths,
            **q_sv_inputs,
        },
        fixed_param=True,
        chains=1,
        iter_sampling=1,
        seed=PARAM_SEED,
        sig_figs=18,
        show_progress=False,
    )
    return (
        fit.stan_variable("loglik")[0],
        fit.stan_variable("loglik_tv")[0],
        fit.stan_variable("loglik_sv")[0],
        fit.stan_variable("loglik_tvq")[0],
        fit.stan_variable("loglik_svq")[0],
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
