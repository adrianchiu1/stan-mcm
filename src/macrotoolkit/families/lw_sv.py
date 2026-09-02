"""lw_sv family numerics metadata (S5-decisions item 2).

THE one place the lw_sv state-slot layout and structural-coefficient matrix
positions are written down outside the validated matrix constructor itself
(``macrotoolkit.smoother.build_lw_matrices``, which this module's metadata
is pinned against by ``tests/test_state_metadata.py``). Everything
downstream -- ``results_lw.py``'s output modules, the generic engines --
consumes the NAMES exported here, never raw indices.

State layout (``smoother.py``'s module docstring, HLW's own):

    xi_t = [y*_t, y*_{t-1}, y*_{t-2}, g_{t-1}, g_{t-2}, z_{t-1}, z_{t-2}]

expressed below as ``(name, offset)`` labels -- note g and z are carried
LAGGED one period relative to the state row's own period (the "slot-3/5
one-period-lag convention" both S4 fan-chart bugs tripped over; with named
offsets the convention is in the label, not in the reader's memory).

Shock timing: a period-t structural shock realizes going INTO row t's
state (``smoother.SimSmootherDraw``'s convention), so e.g. the g shock
lands in the slot labeled ``("g", -1)`` -- the row-t slot that carries the
most recent g value.
"""
from __future__ import annotations

import numpy as np

from macrotoolkit.families.base import StateSpaceMeta

#: Named state metadata for lw_sv. The g shock's 0.25 loading into
#: ("ystar", 0) is the quarterly g/4 increment (g is ANNUALIZED everywhere;
#: only the potential-output transition divides by 4 -- spec §1.3, enforced
#: by tests/test_units_conventions.py); 0.25 * eps is bit-identical to
#: eps / 4.0 in IEEE double, so this loading reproduces the historical
#: hand-rolled injection exactly.
LW_STATE_META = StateSpaceMeta(
    state_labels=(
        ("ystar", 0),
        ("ystar", -1),
        ("ystar", -2),
        ("g", -1),
        ("g", -2),
        ("z", -1),
        ("z", -2),
    ),
    state_shocks=("ystar", "g", "z"),
    shock_loadings={
        "ystar": {("ystar", 0): 1.0},
        "g": {("g", -1): 1.0, ("ystar", 0): 0.25},
        "z": {("z", -1): 1.0},
    },
    measurement_shocks=("is", "pc"),
)

#: Observation-row order (measurement equation rows of yobs/Z/R).
OBS_NAMES = ("y", "pi")


def structural_coefficients(Z: np.ndarray, A: np.ndarray) -> dict[str, float]:
    """The named structural-coefficient dict read off one draw's ``Z``/``A``
    system matrices -- the exact entries ``build_lw_matrices`` stamps,
    inverted here in the ONE place that knows where they live:

    - ``a1 = -Z[y, (ystar,-1)]``, ``a2 = -Z[y, (ystar,-2)]`` (gap-lag
      conversion terms of the IS curve);
    - ``a_r = -2 * Z[y, (z,-1)]`` -- read off the z-lag entry, which
      carries NO ``c`` factor (``build_lw_matrices`` multiplies only the
      g-lag entries by ``c``), so the extraction stays correct regardless
      of ``c``;
    - ``b_y = -Z[pi, (ystar,-1)]`` (Phillips curve's gap-lag conversion);
    - ``b_pi = A[4, 1]`` -- exogenous row 4 is pi_{t-1}, observation
      column 1 is pi (``build_lw_regressors``' x-column order; becomes a
      named feedback-map lookup in S5-decisions item 1);
    - ``c``: the r* = c*g + z loading, recovered as the ratio of the g-lag
      to the z-lag Z entries (1.0 when the z entry is zero, i.e. a_r = 0,
      where the ratio is undefined and c is unidentified in Z anyway).

    Downstream code MUST consume this dict (or a draw's posterior values
    directly) rather than re-peeking matrix entries.
    """
    y_row = OBS_NAMES.index("y")
    pi_row = OBS_NAMES.index("pi")
    s_ystar_m1 = LW_STATE_META.slot("ystar", -1)
    s_ystar_m2 = LW_STATE_META.slot("ystar", -2)
    s_g_m1 = LW_STATE_META.slot("g", -1)
    s_z_m1 = LW_STATE_META.slot("z", -1)

    z_lag_entry = Z[y_row, s_z_m1]
    c = Z[y_row, s_g_m1] / z_lag_entry if z_lag_entry != 0.0 else 1.0
    return {
        "a1": -Z[y_row, s_ystar_m1],
        "a2": -Z[y_row, s_ystar_m2],
        "a_r": -2.0 * z_lag_entry,
        "b_y": -Z[pi_row, s_ystar_m1],
        "b_pi": A[4, 1],
        "c": float(c),
    }


def require_c_is_one(c: float, where: str) -> None:
    """Fail loudly if the system matrices imply ``c != 1.0`` (spec §1.3's
    fixed default; ``estimate_c`` is hard-validated ``False`` everywhere in
    current scope, ``specs/schema/lw_sv.py``).

    Numerics-reviewer finding (S4): every downstream computation that sums
    ``g + z`` UNWEIGHTED (r* = g + z) is correct only for c == 1; with
    c != 1 the g terms would need a ``c`` weight, and a naive unweighted
    sum silently corrupts the result (verified numerically for the HD:
    c = 1.5 produces a gap-HD reconstruction error of ~0.36 against G6's
    1e-6 tolerance). ``where`` names the calling computation so the error
    points at the code that must be generalized before the guard is
    removed.
    """
    if not np.isclose(c, 1.0, atol=1e-9):
        raise NotImplementedError(
            f"{where} assumes c == 1.0 (spec §1.3's default; estimate_c is "
            f"not implemented anywhere in current scope), but the system "
            f"matrices imply c = {c!r}. The r* = g + z computation sums g "
            f"and z unweighted, which is only correct for c == 1.0 -- "
            f"generalize it (weight the g terms by c) before removing this "
            f"guard."
        )
