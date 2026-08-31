"""G5a gate (lw-sv-spec.md §5): exact HLW replication at fixed parameters.

Oracle: `tests/fixtures/hlw/derived/us_2017_reproduction/` -- our own
verified run of the genuine HLW (2017) R code (see that fixture's README
and FIXTURES.md). Per the 2026-08-31 sign-off (DECISIONS.md), the pass
criterion is NEAR MACHINE PRECISION against this reproduction: our KF and
smoother, fixed at the fixture's MLE parameter values with the fixture's
exact initial conditions, are evaluating literally the same linear-Gaussian
model as HLW's own `kalman.states.R`, so any gap beyond floating-point
noise is a real bug -- the ~0.06pp/0.31pp deviation documented against the
published 2019Q2 vintage is a *different* discrepancy (data revisions,
optimizer path) and is not slack these tests may consume.

Units bridge (the one deliberate difference from HLW's code): our state
keeps g ANNUALIZED (spec §1.1/§1.3); HLW's keeps it quarterly and
multiplies by 4 at reporting time. The two are exact unit transforms under
S = diag(1,1,1,4,4,1,1):

    xi_ann = S xi_hlw,   P_ann = S P_hlw S',
    sigma_g_ann = 4 * sigma_g_hlw,
    F_ann = S F_hlw S^-1,  Q_ann = S Q_hlw S',  Z_ann = Z_hlw S^-1.

The fixture's xi.00/P.00 (dumped verbatim from the R run -- P.00 is the
output of calculate.covariance.R's inner MLE pass, which no mirror should
re-derive independently) are converted with S before entering our filter.

Tolerances: observed agreement is ~1e-12 (loglik) / ~2e-12 (states); the
asserts use 1e-8 -- three orders of magnitude of headroom over float noise,
eight below any economically meaningful discrepancy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macrotoolkit.smoother import (
    build_lw_matrices,
    build_lw_regressors,
    kalman_loglik,
    kalman_smoother,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "hlw" / "derived" / "us_2017_reproduction"

LOGLIK_TOL = 1e-8
STATE_TOL = 1e-8

#: Quarterly-g -> annualized-g state unit conversion (see module docstring).
S_UNITS = np.diag([1.0, 1.0, 1.0, 4.0, 4.0, 1.0, 1.0])


@pytest.fixture(scope="module")
def oracle() -> dict:
    params = (
        pd.read_csv(FIXTURE_DIR / "output" / "us_2017_parameters.csv")
        .set_index("parameter")["value"]
    )
    data = pd.read_csv(FIXTURE_DIR / "inputData" / "rstar.data.us.csv")
    return {
        "params": params,
        "y": 100.0 * data["gdp.log"].to_numpy(),  # spec §1.1: y = 100 * ln(GDP)
        "pi": data["inflation"].to_numpy(),
        "r": (data["interest"] - data["inflation.expectations"]).to_numpy(),
        "xi00_q": pd.read_csv(FIXTURE_DIR / "output" / "us_2017_xi00.csv")["xi00"].to_numpy(),
        "P00_q": pd.read_csv(FIXTURE_DIR / "output" / "us_2017_P00.csv").to_numpy(),
        "smoothed": pd.read_csv(FIXTURE_DIR / "output" / "us_2017_smoothed.csv"),
        "one_sided": pd.read_csv(FIXTURE_DIR / "output" / "us_2017_one_sided.csv"),
    }


def _our_params(hlw: pd.Series) -> dict:
    """Map the fixture's HLW parameter names/units to ours: sigma_ytilde/
    sigma_pi are the constant IS/PC shock scales (measurement noise), and
    sigma_g is annualized (x4) per the spec's g convention."""
    return {
        "a1": hlw["a_y1"],
        "a2": hlw["a_y2"],
        "a_r": hlw["a_r"],
        "b_pi": hlw["b_pi"],
        "b_y": hlw["b_y"],
        "sigma_ystar": hlw["sigma_ystar"],
        "sigma_g": 4.0 * hlw["sigma_g"],
        "sigma_z": hlw["sigma_z"],
        "sigma_is": hlw["sigma_ytilde"],
        "sigma_pc": hlw["sigma_pi"],
    }


def _run_smoother(o: dict) -> tuple[dict, np.ndarray]:
    yobs, x = build_lw_regressors(o["y"], o["pi"], o["r"])
    F, Q, A, Z, R = build_lw_matrices(_our_params(o["params"]), c=1.0)
    xi00 = S_UNITS @ o["xi00_q"]
    P00 = S_UNITS @ o["P00_q"] @ S_UNITS.T
    return kalman_smoother(yobs, x, F, Q, A, Z, R, xi00, P00), yobs


def _series_from_states(xi: np.ndarray, yobs: np.ndarray) -> pd.DataFrame:
    """HLW's four reported series from a (T,7) state path in our annualized
    units: g = state 4 (already annualized), z = state 6, r* = g + z (c=1),
    gap = y_t - y*_t."""
    return pd.DataFrame(
        {
            "rstar": xi[:, 3] + xi[:, 5],
            "g": xi[:, 3],
            "z": xi[:, 5],
            "output_gap": yobs[:, 0] - xi[:, 0],
        }
    )


def test_g5a_loglik_matches_hlw_mle_loglik(oracle: dict) -> None:
    yobs, x = build_lw_regressors(oracle["y"], oracle["pi"], oracle["r"])
    F, Q, A, Z, R = build_lw_matrices(_our_params(oracle["params"]), c=1.0)
    xi00 = S_UNITS @ oracle["xi00_q"]
    P00 = S_UNITS @ oracle["P00_q"] @ S_UNITS.T
    ll = kalman_loglik(yobs, x, F, Q, A, Z, R, xi00, P00)
    diff = abs(ll - oracle["params"]["log_likelihood"])
    assert diff < LOGLIK_TOL, (
        f"G5a loglik mismatch: ours={ll!r}, HLW={oracle['params']['log_likelihood']!r}, "
        f"|diff|={diff:.3e} >= {LOGLIK_TOL:.1e}"
    )


def test_g5a_smoothed_states_match_hlw(oracle: dict) -> None:
    res, yobs = _run_smoother(oracle)
    ours = _series_from_states(res["xi_smooth"], yobs)
    for col in ("rstar", "g", "z", "output_gap"):
        diff = np.abs(ours[col].to_numpy() - oracle["smoothed"][col].to_numpy())
        assert diff.max() < STATE_TOL, (
            f"G5a smoothed {col} mismatch: max abs diff {diff.max():.3e} >= "
            f"{STATE_TOL:.1e} at quarter index {int(diff.argmax())}"
        )


def test_g5a_filtered_states_match_hlw(oracle: dict) -> None:
    res, yobs = _run_smoother(oracle)
    ours = _series_from_states(res["xi_filt"], yobs)
    for col in ("rstar", "g", "z", "output_gap"):
        diff = np.abs(ours[col].to_numpy() - oracle["one_sided"][col].to_numpy())
        assert diff.max() < STATE_TOL, (
            f"G5a filtered {col} mismatch: max abs diff {diff.max():.3e} >= "
            f"{STATE_TOL:.1e} at quarter index {int(diff.argmax())}"
        )


def test_ssm_matrices_match_hlw_stage3_construction(oracle: dict) -> None:
    """Spec §5's HLW-fixtures-policy unit test: our matrix construction,
    unit-transformed back to HLW's quarterly-g convention, equals the
    matrices `unpack.parameters.stage3.R` builds at the fixture's theta
    (reverse-engineered here term for term from that file)."""
    hlw = oracle["params"]
    F, Q, A, Z, R = build_lw_matrices(_our_params(hlw), c=1.0)

    Sinv = np.linalg.inv(S_UNITS)
    F_q = Sinv @ F @ S_UNITS
    Q_q = Sinv @ Q @ Sinv.T
    Z_q = Z @ S_UNITS

    a_y1, a_y2, a_r = hlw["a_y1"], hlw["a_y2"], hlw["a_r"]
    b_pi, b_y = hlw["b_pi"], hlw["b_y"]
    s_yt, s_pi, s_ys = hlw["sigma_ytilde"], hlw["sigma_pi"], hlw["sigma_ystar"]
    lam_g, lam_z = hlw["lambda_g"], hlw["lambda_z"]

    # unpack.parameters.stage3.R, transcribed (their H is our Z transposed;
    # their A is transposed of the (2,6) form -- ours is already (6,2)).
    F_hlw = np.zeros((7, 7))
    for i, j in [(0, 0), (0, 3), (1, 0), (2, 1), (3, 3), (4, 3), (5, 5), (6, 5)]:
        F_hlw[i, j] = 1.0
    Q_hlw = np.zeros((7, 7))
    Q_hlw[0, 0] = (1 + lam_g**2) * s_ys**2
    Q_hlw[0, 3] = Q_hlw[3, 0] = Q_hlw[3, 3] = (lam_g * s_ys) ** 2
    Q_hlw[5, 5] = (lam_z * s_yt / a_r) ** 2
    A_hlw = np.zeros((6, 2))
    A_hlw[0, 0], A_hlw[1, 0] = a_y1, a_y2
    A_hlw[2, 0] = A_hlw[3, 0] = a_r / 2
    A_hlw[0, 1], A_hlw[4, 1], A_hlw[5, 1] = b_y, b_pi, 1 - b_pi
    Z_hlw = np.zeros((2, 7))
    Z_hlw[0] = [1, -a_y1, -a_y2, -a_r * 2, -a_r * 2, -a_r / 2, -a_r / 2]
    Z_hlw[1, 1] = -b_y
    R_hlw = np.diag([s_yt**2, s_pi**2])

    np.testing.assert_allclose(F_q, F_hlw, atol=1e-14)
    # The fixture stores sigma_g = lambda_g * sigma_ystar and sigma_z =
    # |lambda_z * sigma_ytilde / a_r| rounded through CSV; matrix entries
    # square them, so compare at 1e-12 relative rather than exact.
    np.testing.assert_allclose(Q_q, Q_hlw, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(A, A_hlw, atol=1e-14)
    np.testing.assert_allclose(Z_q, Z_hlw, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(R, R_hlw, rtol=1e-14)
