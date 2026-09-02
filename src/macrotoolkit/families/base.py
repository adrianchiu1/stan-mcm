"""Generic named state-metadata machinery (S5-decisions item 2).

Every family exports one :class:`StateSpaceMeta` instance declaring, as
data:

- the state vector's slot labels, each a ``(name, time_offset)`` pair --
  e.g. ``("g", -1)`` for a slot that carries trend growth lagged one period
  relative to the state row's own period (the lw_sv state's own layout,
  ``macrotoolkit.smoother``'s module docstring);
- the named structural STATE shocks and how each one loads into the state
  vector (the ``B`` in ``xi_t = F xi_{t-1} + B eps_t``, declared entry by
  entry against slot LABELS, never raw indices);
- the named measurement shocks, in observation-row order.

Downstream code (results modules, output engines) looks slots up by name
via :meth:`StateSpaceMeta.slot` and builds shock-injection vectors via
:meth:`StateSpaceMeta.injection_vector` -- it never hard-codes ``xi[:, 3]``
or "slot 3 is g lagged" again (the bug class behind both S4 fan-chart bugs,
per ``plans/S5-decisions.md`` item 2 and ENGINEERING.md).

This module is deliberately family-agnostic and numpy-light; each family's
concrete instance (e.g. ``macrotoolkit.families.lw_sv.LW_STATE_META``) is
the ONE place that family's slot layout is written down, and carries a
consistency test against the family's own matrix constructor
(``tests/test_state_metadata.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

#: A state-slot label: (series name, time offset relative to the state row's
#: own period). Offset 0 = contemporaneous, -1 = lagged one period, etc.
StateLabel = tuple[str, int]


@dataclass(frozen=True)
class StateSpaceMeta:
    """Named state metadata for one model family.

    - ``state_labels``: one :data:`StateLabel` per state slot, in slot
      order. Unique; length = the family's state dimension.
    - ``state_shocks``: the named structural shocks that enter the STATE
      transition, in declared order (this order defines
      :meth:`loading_matrix`'s columns).
    - ``shock_loadings``: per state shock, a mapping ``{label: coefficient}``
      of where (and scaled by what) a unit shock realization lands in the
      state-innovation vector. E.g. lw_sv's annualized g shock loads 1.0
      into the ``("g", -1)`` slot and 0.25 into ``("ystar", 0)`` (the
      quarterly g/4 increment, spec §1.3).
    - ``measurement_shocks``: the named shocks entering the MEASUREMENT
      equation, in observation-row order (lw_sv: ``("is", "pc")`` for
      observation rows ``[y, pi]``).
    """

    state_labels: tuple[StateLabel, ...]
    state_shocks: tuple[str, ...]
    shock_loadings: Mapping[str, Mapping[StateLabel, float]]
    measurement_shocks: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.state_labels)) != len(self.state_labels):
            raise ValueError(
                f"state_labels must be unique; got {self.state_labels!r}."
            )
        missing = [s for s in self.state_shocks if s not in self.shock_loadings]
        if missing:
            raise ValueError(
                f"state_shocks {missing} have no entry in shock_loadings."
            )
        extra = [s for s in self.shock_loadings if s not in self.state_shocks]
        if extra:
            raise ValueError(
                f"shock_loadings has entries {extra} not named in state_shocks."
            )
        for shock, loading in self.shock_loadings.items():
            unknown = [lab for lab in loading if lab not in self.state_labels]
            if unknown:
                raise ValueError(
                    f"shock_loadings[{shock!r}] references unknown state "
                    f"label(s) {unknown}; valid labels: {list(self.state_labels)}."
                )

    @property
    def n_state(self) -> int:
        return len(self.state_labels)

    def slot(self, name: str, offset: int = 0) -> int:
        """The state-vector index of the slot labeled ``(name, offset)``.
        Raises ``KeyError`` naming the valid labels if absent -- a typo'd
        name/offset must fail loudly, never fall back to an index guess."""
        label = (name, offset)
        try:
            return self.state_labels.index(label)
        except ValueError:
            raise KeyError(
                f"No state slot labeled {label!r}; valid labels: "
                f"{list(self.state_labels)}."
            ) from None

    def injection_vector(self, shock: str) -> np.ndarray:
        """The (n_state,) loading vector ``b`` for one named state shock:
        a period's state innovation from a realization ``eps_t`` of this
        shock is ``b * eps_t``. Raises ``KeyError`` for an unknown shock."""
        if shock not in self.shock_loadings:
            raise KeyError(
                f"No state shock named {shock!r}; valid state shocks: "
                f"{list(self.state_shocks)}."
            )
        b = np.zeros(self.n_state)
        for label, coef in self.shock_loadings[shock].items():
            b[self.slot(*label)] = coef
        return b

    def loading_matrix(self) -> np.ndarray:
        """The (n_state, n_state_shocks) structural loading matrix ``B``
        (columns in ``state_shocks`` order), so the state-innovation
        covariance is ``Q = B @ diag(sigma^2) @ B.T`` for per-shock
        variances ``sigma^2`` -- pinned against the family's own matrix
        constructor by ``tests/test_state_metadata.py``."""
        return np.column_stack([self.injection_vector(s) for s in self.state_shocks])
