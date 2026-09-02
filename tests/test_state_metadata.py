"""Named state/coefficient metadata (S5-decisions item 2).

``macrotoolkit.families.lw_sv.LW_STATE_META`` is the named mirror of the
slot layout ``macrotoolkit.smoother.build_lw_matrices`` defines; these
tests pin the two against each other so the metadata can never silently
drift from the validated matrix constructor:

1. Slot lookups resolve to the documented indices (and unknown labels fail
   loudly).
2. The declared shock loadings reproduce ``build_lw_matrices``' Q exactly:
   ``Q == B @ diag(sigma^2) @ B.T`` for the family's 3 state shocks.
3. Deterministic lag-copy slots (every label ``(name, k)`` whose ``(name,
   k+1)`` is also a label) are exactly that in F -- a single 1.0 copying
   the one-period-fresher slot -- and carry zero process noise in Q.
4. ``structural_coefficients`` recovers the exact parameter values (and
   c) that ``build_lw_matrices`` stamped into Z/A.
5. ``require_c_is_one`` passes at c == 1 and raises (naming the caller)
   otherwise.
"""
from __future__ import annotations

import numpy as np
import pytest

from macrotoolkit.families.base import StateSpaceMeta
from macrotoolkit.families.lw_sv import (
    LW_STATE_META,
    require_c_is_one,
    structural_coefficients,
)
from macrotoolkit.smoother import N_STATE, build_lw_matrices

_PARAMS = {
    "a1": 1.17,
    "a2": -0.33,
    "a_r": -0.084,
    "b_pi": 0.71,
    "b_y": 0.093,
    "sigma_ystar": 0.51,
    "sigma_g": 0.027,
    "sigma_z": 0.062,
    "sigma_is": 0.41,
    "sigma_pc": 0.77,
}


def test_state_labels_cover_the_state_dimension() -> None:
    assert LW_STATE_META.n_state == N_STATE
    assert len(set(LW_STATE_META.state_labels)) == N_STATE


def test_slot_lookups_match_documented_layout() -> None:
    # smoother.py's module docstring: xi = [y*_t, y*_{t-1}, y*_{t-2},
    # g_{t-1}, g_{t-2}, z_{t-1}, z_{t-2}].
    assert LW_STATE_META.slot("ystar", 0) == 0
    assert LW_STATE_META.slot("ystar", -1) == 1
    assert LW_STATE_META.slot("ystar", -2) == 2
    assert LW_STATE_META.slot("g", -1) == 3
    assert LW_STATE_META.slot("g", -2) == 4
    assert LW_STATE_META.slot("z", -1) == 5
    assert LW_STATE_META.slot("z", -2) == 6


def test_unknown_slot_fails_loudly_naming_valid_labels() -> None:
    with pytest.raises(KeyError, match=r"\('g', 0\)"):
        LW_STATE_META.slot("g", 0)  # g is carried lagged; (g, 0) must not resolve
    with pytest.raises(KeyError, match="valid labels"):
        LW_STATE_META.slot("nope")


def test_unknown_shock_fails_loudly() -> None:
    with pytest.raises(KeyError, match="valid state shocks"):
        LW_STATE_META.injection_vector("is")  # a MEASUREMENT shock, not a state shock


def test_loading_matrix_reproduces_Q_exactly() -> None:
    _, Q, _, _, _ = build_lw_matrices(_PARAMS, c=1.0)
    B = LW_STATE_META.loading_matrix()
    sigmas = np.array([_PARAMS["sigma_ystar"], _PARAMS["sigma_g"], _PARAMS["sigma_z"]])
    Q_from_meta = B @ np.diag(sigmas**2) @ B.T
    np.testing.assert_allclose(Q_from_meta, Q, rtol=0.0, atol=1e-16)


def test_injection_vectors_match_historical_hand_rolled_construction() -> None:
    eps = 0.7391
    w_ystar = LW_STATE_META.injection_vector("ystar") * eps
    w_g = LW_STATE_META.injection_vector("g") * eps
    w_z = LW_STATE_META.injection_vector("z") * eps

    expected_ystar = np.zeros(N_STATE)
    expected_ystar[0] = eps
    expected_g = np.zeros(N_STATE)
    expected_g[0] = eps / 4.0  # bit-identical to 0.25 * eps
    expected_g[3] = eps
    expected_z = np.zeros(N_STATE)
    expected_z[5] = eps

    np.testing.assert_array_equal(w_ystar, expected_ystar)
    np.testing.assert_array_equal(w_g, expected_g)
    np.testing.assert_array_equal(w_z, expected_z)


def test_lag_copy_slots_are_deterministic_copies_in_F_and_noiseless_in_Q() -> None:
    F, Q, _, _, _ = build_lw_matrices(_PARAMS, c=1.0)
    lag_copy_rows = []
    for name, k in LW_STATE_META.state_labels:
        if (name, k + 1) in LW_STATE_META.state_labels:
            i = LW_STATE_META.slot(name, k)
            j = LW_STATE_META.slot(name, k + 1)
            lag_copy_rows.append(i)
            expected_row = np.zeros(N_STATE)
            expected_row[j] = 1.0
            np.testing.assert_array_equal(F[i], expected_row)
            assert np.all(Q[i] == 0.0) and np.all(Q[:, i] == 0.0)
    # Exactly the 4 documented deterministic lag-copy slots (HANDOFF.md's
    # state-slot warning): y*_{t-1}, y*_{t-2}, g_{t-2}, z_{t-2}.
    assert sorted(lag_copy_rows) == [1, 2, 4, 6]


@pytest.mark.parametrize("c", [1.0, 1.3])
def test_structural_coefficients_recover_stamped_values(c: float) -> None:
    _, _, A, Z, _ = build_lw_matrices(_PARAMS, c=c)
    coeffs = structural_coefficients(Z, A)
    assert coeffs["a1"] == pytest.approx(_PARAMS["a1"], abs=1e-15)
    assert coeffs["a2"] == pytest.approx(_PARAMS["a2"], abs=1e-15)
    assert coeffs["a_r"] == pytest.approx(_PARAMS["a_r"], abs=1e-15)
    assert coeffs["b_y"] == pytest.approx(_PARAMS["b_y"], abs=1e-15)
    assert coeffs["b_pi"] == pytest.approx(_PARAMS["b_pi"], abs=1e-15)
    assert coeffs["c"] == pytest.approx(c, abs=1e-12)


def test_structural_coefficients_c_defaults_to_one_when_a_r_is_zero() -> None:
    params = dict(_PARAMS, a_r=0.0)
    _, _, A, Z, _ = build_lw_matrices(params, c=1.0)
    assert structural_coefficients(Z, A)["c"] == 1.0


def test_require_c_is_one_guard() -> None:
    require_c_is_one(1.0, "test")  # no raise
    require_c_is_one(1.0 + 1e-12, "test")  # inside the 1e-9 tolerance
    with pytest.raises(NotImplementedError, match="my_computation"):
        require_c_is_one(1.5, "my_computation")


def test_meta_validation_rejects_inconsistent_declarations() -> None:
    with pytest.raises(ValueError, match="unique"):
        StateSpaceMeta(
            state_labels=(("a", 0), ("a", 0)),
            state_shocks=(),
            shock_loadings={},
            measurement_shocks=(),
        )
    with pytest.raises(ValueError, match="no entry in shock_loadings"):
        StateSpaceMeta(
            state_labels=(("a", 0),),
            state_shocks=("a",),
            shock_loadings={},
            measurement_shocks=(),
        )
    with pytest.raises(ValueError, match="unknown state label"):
        StateSpaceMeta(
            state_labels=(("a", 0),),
            state_shocks=("a",),
            shock_loadings={"a": {("b", 0): 1.0}},
            measurement_shocks=(),
        )
