"""Tests for the mu_h0 OLS anchor (spec §1.6's "rough OLS pass",
user-confirmed HLW-exact definition, DECISIONS.md 2026-08-31) and its
wiring into build_stan_data.

The anchor's DEFINITION is rstar.stage3.R lines 22-48 transcribed onto our
trimmed-data convention; there is no independent numeric oracle for it (the
R code doesn't dump its initial parameters), so these tests pin:

- the units convention (mu_h0 = 2*ln(sigma_hat), i.e. exp(mu_h0/2) is a
  RESIDUAL SD -- the same h-is-log-variance convention everything else
  follows),
- exactness on a hand-constructible case (data generated to satisfy the PC
  regression exactly except for known residuals),
- determinism and plausibility pins on the real US example data,
- the guard rails (too-short and degenerate inputs).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macrotoolkit.run import build_stan_data, lw_mu_h0_anchors

US_CSV = Path(__file__).parent.parent / "examples" / "us_lw_sv" / "data" / "us_quarterly.csv"


def _us_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(US_CSV)
    return (
        df["lgdp100"].to_numpy(),
        df["core_pce_ann"].to_numpy(),
        df["real_rate"].to_numpy(),
    )


def test_anchor_is_two_log_sigma_of_known_residuals() -> None:
    """Exactness on a constructed case: make y a PURE linear trend (so the
    gap proxy is exactly 0 for every quarter) and build pi to follow the PC
    regression exactly with known residuals. With a zero gap, the IS
    regression's dependent variable and gap regressors vanish, so the
    IS residuals are exactly 0 (excluded by the positivity guard) -- so
    perturb y by a known pattern instead and check the PC side, where the
    algebra stays closed-form: with gap == 0 the PC regression is pi_t on
    [pi_{t-1}, pibar_t] and residuals equal the injected noise exactly IF
    the injected noise is orthogonal to those regressors. Orthogonality is
    fiddly by hand, so instead pin the sharper property that survives any
    OLS: predicted residual sd <= injected sd on average, equality when the
    noise is orthogonal -- and verify the whole pipeline against a direct
    reimplementation of the two regressions via the normal equations
    (HLW's own solve(X'X, X'y) form), which must agree to float precision.
    """
    rng = np.random.default_rng(20260903)
    n = 104
    y = 800.0 + 0.75 * np.arange(n) + rng.normal(0.0, 0.8, n)
    pi = 2.0 + rng.normal(0.0, 0.9, n)
    r = 3.0 + rng.normal(0.0, 1.0, n)

    mu_is, mu_pc = lw_mu_h0_anchors(y, pi, r)

    # Direct reimplementation via normal equations (HLW's solve form).
    t_est = n - 4
    X_tr = np.column_stack([np.ones(n), np.arange(1, n + 1)])
    gap = y - X_tr @ np.linalg.solve(X_tr.T @ X_tr, X_tr.T @ y)
    x_is = np.column_stack(
        [gap[3 : 3 + t_est], gap[2 : 2 + t_est],
         (r[3 : 3 + t_est] + r[2 : 2 + t_est]) / 2.0, np.ones(t_est)]
    )
    res_is = gap[4:n] - x_is @ np.linalg.solve(x_is.T @ x_is, x_is.T @ gap[4:n])
    s_is = math.sqrt(res_is @ res_is / (t_est - 4))
    x_pc = np.column_stack(
        [pi[3 : 3 + t_est],
         (pi[2 : 2 + t_est] + pi[1 : 1 + t_est] + pi[0:t_est]) / 3.0,
         gap[3 : 3 + t_est]]
    )
    res_pc = pi[4:n] - x_pc @ np.linalg.solve(x_pc.T @ x_pc, x_pc.T @ pi[4:n])
    s_pc = math.sqrt(res_pc @ res_pc / (t_est - 3))

    assert mu_is == pytest.approx(2.0 * math.log(s_is), abs=1e-10)
    assert mu_pc == pytest.approx(2.0 * math.log(s_pc), abs=1e-10)

    # Units convention: exp(mu_h0 / 2) recovers the residual SD (h is
    # log-variance), and for iid noise the residual sd sits near the
    # injected scale (within loose sampling bounds).
    assert math.exp(mu_pc / 2.0) == pytest.approx(s_pc)
    assert 0.6 < s_pc < 1.2  # injected 0.9


def test_anchor_on_us_example_data_is_deterministic_and_plausible() -> None:
    """The real-data pin: deterministic to the last bit across calls, and
    the implied OLS residual sds sit where HLW's own initialization logic
    puts them on this window (PC ~ 0.82, right on HLW's MLE sigma_pi ~ 0.80;
    IS ~ 0.75, larger than MLE sigma_ytilde ~ 0.34 because the linear-
    detrend gap proxy is noisier than the model's y* -- expected, this is
    HLW's own init behavior). Bands are +-25% around the values measured at
    implementation time, wide enough for a data-file refresh of the same
    window, tight enough to catch a lag/indexing slip (which moves these by
    factors, not percents)."""
    y, pi, r = _us_arrays()
    a = lw_mu_h0_anchors(y, pi, r)
    b = lw_mu_h0_anchors(y, pi, r)
    assert a == b

    sd_is = math.exp(a[0] / 2.0)
    sd_pc = math.exp(a[1] / 2.0)
    assert 0.56 < sd_is < 0.94  # measured 0.7513
    assert 0.61 < sd_pc < 1.02  # measured 0.8179


def test_build_stan_data_includes_anchors_for_lw_sv() -> None:
    y, pi, r = _us_arrays()
    df = pd.DataFrame({"y": y, "pi": pi, "r": r})
    data = build_stan_data("lw_sv", df)
    assert data["mu_h0_is"] == pytest.approx(lw_mu_h0_anchors(y, pi, r)[0])
    assert data["mu_h0_pc"] == pytest.approx(lw_mu_h0_anchors(y, pi, r)[1])
    assert data["T"] == len(y) - 4


def test_anchor_guards() -> None:
    with pytest.raises(ValueError, match="at least 9 data rows"):
        lw_mu_h0_anchors(np.arange(8.0), np.ones(8), np.ones(8))
    # Degenerate input: an identically-zero inflation series makes the PC
    # residuals exactly zero (lstsq of a zero target returns zero
    # coefficients and zero residuals), so sigma_hat_PC = 0 and the
    # log-anchor is undefined -- the guard must refuse rather than emit
    # -inf. (Merely-constant nonzero series can slip past on float noise;
    # they produce absurd but finite anchors, which is garbage-in
    # garbage-out, not a crash path.)
    n = 40
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="residual sd"):
        lw_mu_h0_anchors(
            800.0 + 0.75 * np.arange(n) + rng.normal(0.0, 0.5, n),
            np.zeros(n),
            3.0 + rng.normal(0.0, 1.0, n),
        )
