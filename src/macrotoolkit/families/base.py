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


# ---------------------------------------------------------------------------
# Feedback map (S5-decisions item 1): the family's declaration of what each
# exogenous-regressor (x) column IS. lw_sv makes itself conditionally linear
# by hiding lagged endogenous observables (y, pi) in the "exogenous" x
# matrix -- valid for the likelihood, but it means (F, Q, A, Z, R) alone is
# not the full generative model. The feedback map closes that gap AS DATA:
# each x column is declared to be either a lag of a named observable (the
# endogenous feedback the generic engine must supply from its own simulated
# path), a mean of several such lags, or a lag of a genuinely exogenous
# named series (supplied by data or by a forecast rule). The ONE generic
# simulate/IRF/HD engine (macrotoolkit.engine) consumes this declaration;
# no downstream code hand-derives the observation recursions again.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObsLag:
    """x column = observable ``name`` lagged ``lag`` >= 1 periods."""

    name: str
    lag: int


@dataclass(frozen=True)
class ObsLagMean:
    """x column = the plain mean of observable ``name`` at the given lags
    (each >= 1), summed in declared order then divided once -- e.g. lw_sv's
    pi-bar column ``(pi_{t-2} + pi_{t-3} + pi_{t-4}) / 3``."""

    name: str
    lags: tuple[int, ...]


@dataclass(frozen=True)
class ExogLag:
    """x column = genuinely exogenous series ``name`` lagged ``lag`` >= 0
    periods -- real data in-sample; a declared forecast rule out of
    sample. ``lag == 0`` (S8 E2) is a CONTEMPORANEOUS exogenous regressor:
    the engine resolves the series for the period being simulated
    instead of the previous one (see ``simulate_forward``)."""

    name: str
    lag: int


@dataclass(frozen=True)
class Const:
    """x column = 1 in every period (S8 E1): a measurement-equation
    intercept enters ``A'x_t`` through this column. The engine switches
    it on for level paths (forward simulation, the HD ``const`` bar) and
    off for deviation paths (IRFs, every other HD bar)."""


FeedbackTerm = ObsLag | ObsLagMean | ExogLag | Const


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
    - ``obs_names``: the named observables, in observation-row order
      (matches ``measurement_shocks`` element-for-element when both are
      declared).
    - ``exog_names``: genuinely exogenous input series (lw_sv: ``("r",)``).
    - ``feedback_map``: one :data:`FeedbackTerm` per x column, in column
      order (S5-decisions item 1) -- empty for a family whose measurement
      equation has no exogenous-regressor matrix.
    - ``measurement_loadings`` (S8 E5, optional): per measurement shock,
      the observation ROWS it loads into, its OWN row first -- the
      measurement-side analogue of ``shock_loadings``, declared as a
      sparsity pattern because the coefficients are parameter-dependent
      (a recursive VAR's contemporaneous coefficient IS the parameter)
      and arrive per draw as the numeric ``M`` matrix. ``None`` (every
      hand family) means each shock loads its own row only, by position,
      and every consumer takes the pre-S8 path bit for bit. With it,
      ``measurement_shocks`` may be SHORTER than ``obs_names`` (a
      shock-free row, S8 E3) and ``n_obs`` is the row count.
    """

    state_labels: tuple[StateLabel, ...]
    state_shocks: tuple[str, ...]
    shock_loadings: Mapping[str, Mapping[StateLabel, float]]
    measurement_shocks: tuple[str, ...]
    obs_names: tuple[str, ...] = ()
    exog_names: tuple[str, ...] = ()
    feedback_map: tuple[FeedbackTerm, ...] = ()
    measurement_loadings: Mapping[str, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        if self.measurement_loadings is None:
            if self.obs_names and len(self.obs_names) != len(self.measurement_shocks):
                raise ValueError(
                    f"obs_names {self.obs_names} and measurement_shocks "
                    f"{self.measurement_shocks} must align row-for-row."
                )
        else:
            if not self.obs_names:
                raise ValueError("measurement_loadings needs obs_names (the observation rows).")
            if set(self.measurement_loadings) != set(self.measurement_shocks):
                raise ValueError(
                    f"measurement_loadings must name exactly the measurement_shocks "
                    f"{self.measurement_shocks}; got {sorted(self.measurement_loadings)}."
                )
            own_rows = []
            for shock, rows in self.measurement_loadings.items():
                if not rows:
                    raise ValueError(f"measurement_loadings[{shock!r}] is empty; a shock loads at least its own row.")
                unknown = [r for r in rows if r not in self.obs_names]
                if unknown:
                    raise ValueError(f"measurement_loadings[{shock!r}] references unknown observable(s) {unknown}; obs_names: {self.obs_names}.")
                own_rows.append(rows[0])
            if len(set(own_rows)) != len(own_rows):
                raise ValueError(f"measurement_loadings: two shocks share an own row ({own_rows}); one own shock per row.")
        for term in self.feedback_map:
            if isinstance(term, (ObsLag, ObsLagMean)):
                if term.name not in self.obs_names:
                    raise ValueError(
                        f"feedback_map term {term!r} references unknown "
                        f"observable; obs_names: {self.obs_names}."
                    )
            elif isinstance(term, ExogLag):
                if term.name not in self.exog_names:
                    raise ValueError(
                        f"feedback_map term {term!r} references unknown "
                        f"exogenous series; exog_names: {self.exog_names}."
                    )
                if term.lag < 0:
                    raise ValueError(f"feedback_map term {term!r} has a negative lag.")
                continue
            elif isinstance(term, Const):
                continue
            else:
                raise ValueError(f"feedback_map term {term!r} is not a FeedbackTerm.")
            lags = term.lags if isinstance(term, ObsLagMean) else (term.lag,)
            if any(l < 1 for l in lags):
                raise ValueError(
                    f"feedback_map term {term!r} has a lag < 1; endogenous x columns "
                    f"must be strictly lagged (a contemporaneous endogenous "
                    f"regressor would not be a valid feedback declaration)."
                )
        if sum(isinstance(t, Const) for t in self.feedback_map) > 1:
            raise ValueError("feedback_map declares more than one Const column; one column of ones suffices.")
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

    @property
    def n_obs(self) -> int:
        return len(self.obs_names) if self.obs_names else len(self.measurement_shocks)

    @property
    def n_meas_shocks(self) -> int:
        return len(self.measurement_shocks)

    def meas_shock_row(self, shock: str) -> int:
        """The observation row a measurement shock OWNS (the row of the
        equation it was written in): its position when no loadings are
        declared, else its declared own row."""
        if self.measurement_loadings is None:
            return self.measurement_shocks.index(shock)
        return self.obs_index(self.measurement_loadings[shock][0])

    def meas_recovery_order(self) -> tuple[tuple[str, int, tuple[str, ...]], ...]:
        """How to invert a measurement residual ``e = M eps`` back into the
        named measurement shocks EXACTLY: resolve each shock off its OWN
        row once every other shock loading into that row is resolved
        (``eps_s = (e[row] - sum_r M[row, r] eps_r) / M[row, s]``) -- the
        measurement-side twin of :meth:`recovery_order`. Returns
        ``(shock, own_row_index, other_shocks_on_that_row)`` in resolution
        order; without declared loadings each shock is simply its own
        row's residual (the pre-S8 read)."""
        if self.measurement_loadings is None:
            return tuple((s, j, ()) for j, s in enumerate(self.measurement_shocks))
        loaders: dict[str, list[str]] = {}
        for shock, rows in self.measurement_loadings.items():
            for r in rows:
                loaders.setdefault(r, []).append(shock)
        resolved: list[str] = []
        order: list[tuple[str, int, tuple[str, ...]]] = []
        pending = list(self.measurement_shocks)
        while pending:
            progress = False
            for shock in list(pending):
                own = self.measurement_loadings[shock][0]
                others = tuple(s for s in loaders[own] if s != shock)
                if all(s in resolved for s in others):
                    order.append((shock, self.obs_index(own), others))
                    resolved.append(shock)
                    pending.remove(shock)
                    progress = True
                    break
            if not progress:
                raise ValueError(
                    f"measurement_loadings are not triangular: cannot recover shocks {pending} "
                    f"from their own rows (resolved so far: {resolved})."
                )
        return tuple(order)

    def has_const_column(self) -> bool:
        return any(isinstance(t, Const) for t in self.feedback_map)

    def exog_is_referenced(self, name: str) -> bool:
        """True iff some x column is a lag (any lag, 0 included) of the
        exogenous series."""
        return any(isinstance(t, ExogLag) and t.name == name for t in self.feedback_map)

    def exog_contemporaneous(self, name: str) -> bool:
        """True iff the feedback map references the exogenous series at
        lag 0 (S8 E2) -- the engine then resolves it for the CURRENT
        period."""
        return any(isinstance(t, ExogLag) and t.name == name and t.lag == 0 for t in self.feedback_map)

    def obs_index(self, name: str) -> int:
        """Observation-row index of the named observable."""
        try:
            return self.obs_names.index(name)
        except ValueError:
            raise KeyError(
                f"No observable named {name!r}; obs_names: {list(self.obs_names)}."
            ) from None

    def obs_lag_depth(self, name: str) -> int:
        """How many past periods of observable ``name`` the feedback map
        reaches back (0 if never referenced)."""
        depth = 0
        for term in self.feedback_map:
            if isinstance(term, ObsLag) and term.name == name:
                depth = max(depth, term.lag)
            elif isinstance(term, ObsLagMean) and term.name == name:
                depth = max(depth, max(term.lags))
        return depth

    def exog_lag_depth(self, name: str) -> int:
        """How many past periods of exogenous series ``name`` the feedback
        map reaches back (0 if never referenced)."""
        depth = 0
        for term in self.feedback_map:
            if isinstance(term, ExogLag) and term.name == name:
                depth = max(depth, term.lag)
        return depth

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

    def recovery_order(self) -> tuple[tuple[str, StateLabel, float, tuple[tuple[str, float], ...]], ...]:
        """How to invert a state innovation ``w = B eps`` back into the
        named shocks EXACTLY (no least squares): resolve shocks in an
        order where each shock is read off ONE of its slots whose other
        loaders are already resolved -- ``eps_s = (w[slot] - sum_r coef_r
        eps_r) / coef_s``. Returns a tuple of ``(shock, slot_label,
        coef, ((other_shock, other_coef), ...))`` in resolution order.
        Raises if the loadings are not triangularizable this way (a
        family whose shocks all load only onto shared slots would need a
        different recovery -- fail loudly rather than approximate).

        For lw_sv this reproduces the historical hand-written recovery
        bit for bit: g from ("g",-1), z from ("z",-1), then ystar =
        (w[ystar,0] - 0.25*eps_g)/1.0 (0.25*x == x/4 exactly in IEEE
        double; /1.0 is exact) -- pinned by tests/test_smoother_sim.py.
        """
        loaders: dict[StateLabel, list[tuple[str, float]]] = {}
        for shock in self.state_shocks:
            for label, coef in self.shock_loadings[shock].items():
                if coef != 0.0:
                    loaders.setdefault(label, []).append((shock, coef))
        resolved: list[str] = []
        order: list[tuple[str, StateLabel, float, tuple[tuple[str, float], ...]]] = []
        pending = list(self.state_shocks)
        while pending:
            progress = False
            for shock in list(pending):
                for label, coef in self.shock_loadings[shock].items():
                    if coef == 0.0:
                        continue
                    others = [(s, c) for s, c in loaders[label] if s != shock]
                    if all(s in resolved for s, _ in others):
                        order.append((shock, label, float(coef), tuple(others)))
                        resolved.append(shock)
                        pending.remove(shock)
                        progress = True
                        break
                if progress:
                    break
            if not progress:
                raise ValueError(
                    f"shock_loadings are not triangular: cannot recover shocks "
                    f"{pending} from any slot whose other loaders are already "
                    f"resolved (resolved so far: {resolved})."
                )
        return tuple(order)

    def loading_matrix(self) -> np.ndarray:
        """The (n_state, n_state_shocks) structural loading matrix ``B``
        (columns in ``state_shocks`` order), so the state-innovation
        covariance is ``Q = B @ diag(sigma^2) @ B.T`` for per-shock
        variances ``sigma^2`` -- pinned against the family's own matrix
        constructor by ``tests/test_state_metadata.py``."""
        if not self.state_shocks:  # S8 E0: no stochastic state shock -> B is (n, 0), Q = 0
            return np.zeros((self.n_state, 0))
        return np.column_stack([self.injection_vector(s) for s in self.state_shocks])
