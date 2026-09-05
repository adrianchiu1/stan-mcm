"""S8 WP3: the two post-processors on SYNTHETIC draws -- sign restrictions
(Haar rotations by QR with a positive R diagonal, rejection sampling with
column flips and the identified shock moved first; the "closest to
median" variant) and Waggoner-Zha conditional forecasts (hard conditions
reproduced exactly; with no conditions the draws are the fan chart's:
given the same standard normals the path equals the engine's forward
simulation -- the zero-noise-seam doctrine)."""
from __future__ import annotations

import numpy as np
import pytest

from macrotoolkit import authoring as au
from macrotoolkit.postprocess import SignRestriction, conditional_forecast, haar_rotation, sign_restricted_irfs

pytestmark = pytest.mark.authored


def _synthetic_irfs(n_draws: int, H: int, rng: np.random.Generator):
    """VAR(1) IRFs ``B^h chol(Sigma)`` over 3 variables (r, y, pi) with a
    little posterior jitter per draw."""
    B0 = np.array([[0.7, 0.1, 0.2], [-0.3, 0.6, 0.0], [-0.1, 0.2, 0.5]])
    S0 = np.array([[0.5, 0.1, 0.05], [0.1, 1.0, 0.2], [0.05, 0.2, 0.8]])
    irf = np.empty((n_draws, H, 3, 3))
    sigmas = []
    for d in range(n_draws):
        B = B0 + 0.02 * rng.standard_normal((3, 3))
        S = S0 + 0.02 * rng.standard_normal((3, 3))
        S = S @ S.T / 2 + 0.3 * np.eye(3)
        L = np.linalg.cholesky(S)
        P = np.eye(3)
        for h in range(H):
            irf[d, h] = P @ L
            P = B @ P
        sigmas.append(S)
    return irf, sigmas


def test_haar_rotation_is_orthonormal_with_the_getqr_sign_convention() -> None:
    rng = np.random.default_rng(1)
    for n in (2, 3, 6):
        Q = haar_rotation(n, rng)
        np.testing.assert_allclose(Q.T @ Q, np.eye(n), atol=1e-12)
    # The convention: the QR of the same normal matrix with R's diagonal made positive.
    rng = np.random.default_rng(7)
    K = np.random.default_rng(7).standard_normal((3, 3))
    q, r = np.linalg.qr(K)
    for i in range(3):
        if r[i, i] < 0:
            q[:, i] = -q[:, i]
    np.testing.assert_allclose(haar_rotation(3, rng), q, atol=1e-14)


def test_sign_restrictions_retain_only_satisfying_rotations_and_keep_sigma() -> None:
    rng = np.random.default_rng(20260916)
    irf, sigmas = _synthetic_irfs(12, 8, rng)
    targets = ("r", "y", "pi")
    restr = [SignRestriction("mp", "r", +1, (0,)), SignRestriction("mp", "y", -1, (0, 1, 2)), SignRestriction("mp", "pi", -1, (0, 1))]
    out = sign_restricted_irfs(irf, restr, targets, rng, max_tries=2000)
    assert out.shock_order[0] == "mp" and len(out.shock_order) == 3 and out.irf.shape[1:] == (8, 3, 3)
    assert len(out.rejected_draws) == 0 and out.irf.shape[0] == 12
    for i, d in enumerate(out.draw_index):
        Q = out.rotations[i]
        np.testing.assert_allclose(Q.T @ Q, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(out.irf[i], irf[d] @ Q, atol=1e-12)  # the rotation actually applied
        imp = out.irf[i, 0]
        np.testing.assert_allclose(imp @ imp.T, sigmas[d], atol=1e-10)  # Sigma is invariant under rotation
        assert imp[0, 0] > 0 and np.all(out.irf[i, :3, 1, 0] < 0) and np.all(out.irf[i, :2, 2, 0] < 0)
    assert np.all(out.n_tries >= 1)
    b = out.bands("mp", "y")
    assert b.shape == (3, 8) and np.all(b[0] <= b[1]) and np.all(b[1] <= b[2])
    # Two restricted shocks are matched to DISTINCT columns; contradictory restrictions reject everything.
    two = restr + [SignRestriction("demand", "y", +1, (0,)), SignRestriction("demand", "pi", +1, (0,)), SignRestriction("demand", "r", +1, (0,))]
    out2 = sign_restricted_irfs(irf, two, targets, np.random.default_rng(3), max_tries=3000)
    assert out2.shock_order[:2] == ("mp", "demand") and out2.irf.shape[0] > 0
    for i in range(out2.irf.shape[0]):
        assert out2.irf[i, 0, 1, 1] > 0 and out2.irf[i, 0, 1, 0] < 0
    bad = [SignRestriction("x", "y", +1, (0,)), SignRestriction("x", "y", -1, (0,))]
    out3 = sign_restricted_irfs(irf, bad, targets, np.random.default_rng(3), max_tries=20)
    assert out3.irf.shape[0] == 0 and len(out3.rejected_draws) == 12 and np.all(out3.n_tries == 20)
    # Closest-to-median: exactly one rotation per draw, chosen among the accepted candidates.
    out4 = sign_restricted_irfs(irf, restr, targets, np.random.default_rng(5), max_tries=5000, closest_to_median=25)
    assert out4.irf.shape[0] == 12 and list(out4.draw_index) == list(range(12))
    with pytest.raises(ValueError, match="unknown variable"):
        sign_restricted_irfs(irf, [SignRestriction("mp", "z", 1)], targets, rng)
    with pytest.raises(ValueError):
        SignRestriction("mp", "r", 2)


def _bivar_var1():
    m = au.Model(
        "cf_var", observables=["y", "pi"],
        measurement=["y = c_y + b_yy*y[-1] + b_yp*pi[-1] + e_y", "pi = a0*y + c_p + b_py*y[-1] + b_pp*pi[-1] + e_p"],
        parameters={"c_y": au.normal(0, 1), "c_p": au.normal(0, 1), "a0": au.normal(0, 1), "b_yy": au.normal(0, 1), "b_yp": au.normal(0, 1),
                    "b_py": au.normal(0, 1), "b_pp": au.normal(0, 1), "s_y": au.half_normal(1), "s_p": au.half_normal(1)},
        shocks={"e_y": au.shock("s_y"), "e_p": au.shock("s_p")},
    ).compiled()
    p = {"c_y": 0.5, "c_p": 0.2, "a0": 0.4, "b_yy": 0.5, "b_yp": -0.2, "b_py": 0.1, "b_pp": 0.6, "s_y": 1.5, "s_p": 0.8}
    return m, p


def _uncond_and_irf(c, p, seeds, H):
    from macrotoolkit.engine import simulate_forward
    from macrotoolkit.results_core import impulse_response

    F, Q, A, Z, R = c.build_matrices(p)
    M = c.build_M(p)

    class Zero:
        def __init__(self, n):
            self.n = n

        def step(self, rng):
            return np.zeros(self.n)

    ybar = simulate_forward(F, Q, A, Z, c.meta, np.zeros(0), seeds, {}, {}, Zero(2), H, np.random.default_rng(0), state_noise=Zero(0), meas_loading=M)["obs"]
    irf = np.stack([impulse_response(F, A, Z, c.meta, s, p[c.structure.shock_scale_param[s]], H, M=M)[1] for s in c.meta.measurement_shocks], axis=2)
    return ybar, irf, (F, Q, A, Z, R, M)


def test_conditional_forecast_reproduces_hard_conditions_exactly() -> None:
    c, p = _bivar_var1()
    H = 6
    seeds = {"y": {1: 1.0}, "pi": {1: 2.0}}
    ybar, irf, _ = _uncond_and_irf(c, p, seeds, H)
    conds = {("pi", 0): 1.0, ("pi", 1): 1.0, ("pi", 2): 1.0}  # the handbook's example 8 path
    cf = conditional_forecast(ybar, irf, conds, np.random.default_rng(2), 300, targets=("y", "pi"), shocks=("e_y", "e_p"))
    for h in range(3):
        np.testing.assert_allclose(cf.draws[:, h, 1], 1.0, atol=1e-10)
        assert abs(cf.mean[h, 1] - 1.0) < 1e-10
    assert np.std(cf.draws[:, 3, 1]) > 0.0 and np.std(cf.draws[:, 0, 0]) > 0.0  # y and later pi stay random
    # The restricted-shock mean/variance are the WZ formulas (pinv form).
    from macrotoolkit.postprocess.conditional_forecast import _stack_R

    R, r = _stack_R(irf, {(1, h): 1.0 for h in range(3)}, ybar)
    np.testing.assert_allclose(cf.shock_mean.reshape(-1), R.T @ np.linalg.pinv(R @ R.T) @ r, atol=1e-12)
    np.testing.assert_allclose(cf.shock_cov, np.eye(H * 2) - R.T @ np.linalg.pinv(R @ R.T) @ R, atol=1e-12)
    np.testing.assert_allclose(R @ cf.shock_cov, 0.0, atol=1e-10)  # conditioned directions carry no variance


def test_conditional_forecast_without_conditions_is_the_fan_chart() -> None:
    from macrotoolkit.engine import simulate_forward

    c, p = _bivar_var1()
    H = 5
    seeds = {"y": {1: 1.0}, "pi": {1: 2.0}}
    ybar, irf, (F, Q, A, Z, R, M) = _uncond_and_irf(c, p, seeds, H)
    z = np.random.default_rng(9).standard_normal((4, H, 2))
    cf = conditional_forecast(ybar, irf, {}, np.random.default_rng(0), 4, shock_draws=z)
    np.testing.assert_allclose(cf.mean, ybar)
    sds = np.array([p["s_y"], p["s_p"]])

    class Given:
        def __init__(self, seq):
            self.seq, self.t = seq, 0

        def step(self, rng):
            v = self.seq[self.t] * sds
            self.t += 1
            return v

    class ZeroState:
        def step(self, rng):
            return np.zeros(0)

    for d in range(4):
        out = simulate_forward(F, Q, A, Z, c.meta, np.zeros(0), seeds, {}, {}, Given(z[d]), H, np.random.default_rng(0), state_noise=ZeroState(), meas_loading=M)
        np.testing.assert_allclose(cf.draws[d], out["obs"], atol=1e-10)  # same standard normals -> the same path
    # In distribution: the draw mean converges to the unconditional path.
    big = conditional_forecast(ybar, irf, {}, np.random.default_rng(11), 20000)
    assert np.max(np.abs(big.draws.mean(axis=0) - ybar)) < 0.1
