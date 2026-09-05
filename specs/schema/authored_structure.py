"""STRUCTURAL compilation of an authored model (S7, M1): from the parsed,
linearized equations to the state-space layout and the SYMBOLIC system
matrices -- pure Python, no numpy, run at spec-validation time so every
structural error (a state referenced more currently than it is carried, a
simultaneous-state cycle, an intercept, a shock in two equations, ...)
fails where every other spec error fails, naming the construct and the
limitation.

The derivation (plans/S7-plan.md "Compilation"):

- **States** are the transition equations' LHS names, in equation order;
  a state's HEAD OFFSET ``d_s`` is its LHS lag (``g[-1] = g[-2] + eta_g``
  declares that ``g`` is CARRIED lagged -- HLW's timing trick, which is
  what makes lw_sv's hand layout reproducible exactly). State ``s`` gets
  slots ``(s, -d_s), (s, -d_s-1), ..., (s, -d_s-(n_s-1))`` with ``n_s`` the
  smallest count that makes every reference resolvable.
- **Transition references** ``s'[-k]`` (row t): ``k == d_{s'}`` is the head
  slot of the CURRENT row -> substitute its own equation (an acyclic
  dependency order is required); ``k > d_{s'}`` -> slot ``(s', -(k-1))``
  of ``xi_{t-1}``; ``k < d_{s'}`` -> error.
- **Measurement references** ``s'[-k]`` -> slot ``(s', -k)`` of ``xi_t``;
  observable / exogenous lags -> regressor (x) columns in first-appearance
  order, which IS the feedback map (ObsLag / ObsLagMean / ExogLag).
- Lag slots are deterministic copies: ``F[(s,-j), (s,-(j-1))] = 1``.
- Shock loadings must be numeric (``StateSpaceMeta`` declares them as
  data); a measurement equation carries AT MOST one measurement shock of
  its own, at unit coefficient.

S8 extensions (plans/S8-plan.md, WP1) -- the same derivation, widened:

- **E0** no stochastic state shock at all, and no transition equation at
  all (``n = 0``): first-class (``B`` is ``(n, 0)``, ``Q = 0``).
- **E1** a constant term in a measurement equation becomes the
  ``("const",)`` feedback column (x column 0 = 1) with ``A[0, row]`` the
  constant expression; a constant term in a transition equation (a
  drift) becomes a loading on an IMPLICIT deterministic unit state
  ``_const`` (last slot; ``F[_const, _const] = 1``, ``xi00 = 1``,
  ``P00 = 0``; the name cannot collide with a user state).
- **E2** a contemporaneous exogenous regressor is ``("exog_lag", x, 0)``.
- **E3** a measurement row may carry no shock (singular ``R``) iff the
  innovation covariance stays positive definite:
  ``rank([M | Z B]) = n_obs`` (``M`` the measurement loadings, ``B`` the
  state loadings), checked at a generic parameter point -- the proof is
  in plans/S8-plan.md.
- **E5** a contemporaneous OBSERVABLE on a right-hand side is substituted
  (its own measurement row -- A, Z, constant, AND its shock -- composed
  into the referencing row with the coefficient) in dependency order; a
  cycle is an error. The composed measurement-shock loadings ``M`` are
  therefore parameter-dependent coefficient expressions (unit on the own
  row) and ``R = M diag(var) M'`` is no longer diagonal in general.

S9 extension (plans/S9-plan.md, E4 -- time-varying coefficients):

- **E4** a measurement row may multiply a STATE (any lag) by a DATA
  series (a lagged observable, a ``mean()`` of one observable, or an
  exogenous series at any lag): the product is a data-dependent
  measurement loading, ``Z_t = Z0 + sum_j x_t[j] * Zx_j`` with ``j`` the
  data factor's feedback-map column (declared exactly as a plain data
  term's column is) and ``Zx_j[row, slot]`` the coefficient expression.
  E5 composes ``Zx`` rows as it composes ``Z``/``A``/``M`` rows. A product
  in a transition equation is a time-varying transition (E6, the next
  stage) and is rejected. A state shock whose loaded slots can reach no
  additively-loaded (``Z0``) slot through ``F``'s pattern moves only
  coefficients (``coefficient_shocks``): it has no additive observable
  bar and no impulse response from rest.

Everything here is a plain dataclass of plain values (labels as tuples,
matrix entries as :class:`specs.schema.equations.Expr` trees keyed by
index) so the numpy-side compiler (``macrotoolkit.authoring.compile``)
only has to evaluate, never re-derive.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from specs.schema.equations import (
    Expr,
    evaluate,
    LinearForm,
    MeanTerm,
    Num,
    ProductTerm,
    SeriesTerm,
    _add_coef,
    _simplify_mul,
    is_numeric,
    linearize,
    name_occurrences,
    numeric_value,
    parameters_in,
    parse_equation,
)


class ModelStructureError(ValueError):
    """The declared model is outside the DSL's scope or internally
    inconsistent; the message names the construct and the limitation."""


_IDENT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

#: Names a PARAMETER may not use (parameters become Stan identifiers):
#: Stan keywords/types/blocks, the names the generic template declares, and
#: library functions the template calls. Series and shock names never
#: appear in the Stan program as identifiers (observables are rows of
#: ``yobs``, states are slots of ``xi``, SV shocks get prefixed names), so
#: they only have to be valid grammar identifiers.
RESERVED_NAMES = frozenset(
    """
    for in while repeat until if then else true false int real vector simplex ordered
    positive_ordered row_vector matrix cholesky_factor_corr cholesky_factor_cov corr_matrix
    cov_matrix array data parameters model functions generated quantities transformed
    return break continue print reject fatal_error target lower upper offset multiplier
    void tuple complex profile increment_log_prob get_lp lp__ T yobs x xi00 P00 F Q A Z R
    Qt Rt Zt kf_loglik mean pi e sqrt exp log square rep_matrix rep_array diag_matrix cumulative_sum
    kalman_loglik sv_rw_noncentered sv_scalar_variance_path
    """.split()
)

#: Names no series/shock may use (the grammar's own function).
GRAMMAR_RESERVED = frozenset({"mean"})

#: Feedback-map terms as plain tuples (converted to ObsLag / ObsLagMean /
#: ExogLag / Const by the numpy side).
FeedbackTuple = tuple

#: The implicit deterministic unit state carrying transition drifts (E1).
CONST_STATE = "_const"


@dataclass
class ModelStructure:
    """The compiled structure of one authored model (see module doc)."""

    name: str
    obs_names: tuple[str, ...]
    exog_names: tuple[str, ...]
    state_names: tuple[str, ...]  # the USER's states (the implicit _const state is not among them)
    head_offset: dict[str, int]
    n_slots: dict[str, int]
    state_labels: tuple[tuple[str, int], ...]  # includes ("_const", 0) last when a drift exists
    state_shocks: tuple[str, ...]
    measurement_shocks: tuple[str, ...]  # own-row order (<= n_obs of them)
    shock_loadings: dict[str, dict[tuple[str, int], float]]
    feedback_map: tuple[FeedbackTuple, ...]
    #: Sparse symbolic matrices: missing entries are exact zeros.
    F: dict[tuple[int, int], Expr]
    A: dict[tuple[int, int], Expr]  # (x column, observation row)
    Z: dict[tuple[int, int], Expr]  # (observation row, state slot)
    #: Measurement-shock loadings (observation row, shock index) -> coefficient (unit on the own row).
    M: dict[tuple[int, int], Expr]
    #: Per measurement shock, the observation rows it loads into, own row first.
    meas_loadings: dict[str, tuple[str, ...]]
    params: tuple[str, ...]
    #: Constant-scale shocks -> the half_normal parameter carrying their sd.
    shock_scale_param: dict[str, str]
    sv_shocks: tuple[str, ...]
    lag_depth: int
    has_const_state: bool = False
    obs_substitution_order: tuple[str, ...] = field(default_factory=tuple)
    canonical_measurement: tuple[str, ...] = field(default_factory=tuple)
    canonical_transition: tuple[str, ...] = field(default_factory=tuple)
    #: S9 E4: data-dependent loadings ``(observation row, state slot) ->
    #: ((x column, coefficient), ...)`` in ascending column order, so
    #: ``Z_t[row, slot] = Z[row, slot] + sum_j x_t[j] * coef_j``.
    Zx: dict[tuple[int, int], tuple[tuple[int, Expr], ...]] = field(default_factory=dict)
    #: S9 E4: state shocks that move only time-varying coefficients.
    coefficient_shocks: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_state(self) -> int:
        return len(self.state_labels)

    @property
    def n_obs(self) -> int:
        return len(self.obs_names)

    @property
    def n_exog_cols(self) -> int:
        return len(self.feedback_map)

    def slot(self, name: str, offset: int) -> int:
        return self.state_labels.index((name, offset))

    def head_slot(self, name: str) -> int:
        return self.slot(name, -self.head_offset[name])

    def matrix_is_numeric(self, which: str) -> bool:
        entries = {"F": self.F, "A": self.A, "Z": self.Z, "M": self.M}[which]
        ok = all(is_numeric(e) for e in entries.values())
        if which == "Z":
            ok = ok and all(is_numeric(e) for terms in self.Zx.values() for _, e in terms)
        return ok

    @property
    def has_data_loadings(self) -> bool:
        """True iff some measurement row multiplies a state by data (E4)."""
        return bool(self.Zx)

    def data_loading_pattern(self) -> tuple[tuple[str, tuple[str, int], int], ...]:
        """The sparsity pattern of ``Zx`` as ``(observable, state label,
        x column)`` triples (the declarative form ``StateSpaceMeta``
        carries)."""
        out = []
        for (i, k), terms in sorted(self.Zx.items()):
            for j, _ in terms:
                out.append((self.obs_names[i], self.state_labels[k], j))
        return tuple(out)

    def meas_loading_is_identity(self) -> bool:
        """True iff every measurement shock loads its own row only, at
        unit coefficient, and every row has a shock -- the pre-S8 case
        (every consumer then takes the pre-S8 code path bit for bit)."""
        if len(self.measurement_shocks) != self.n_obs or len(self.M) != self.n_obs:
            return False
        return all(i == j and isinstance(e, Num) and e.value == 1.0 for (i, j), e in self.M.items())

    def shock_free_rows(self) -> tuple[str, ...]:
        loaded = {i for (i, _) in self.M}
        return tuple(o for i, o in enumerate(self.obs_names) if i not in loaded)

    def sv_state_shocks(self) -> tuple[str, ...]:
        return tuple(s for s in self.state_shocks if s in self.sv_shocks)

    def sv_measurement_shocks(self) -> tuple[str, ...]:
        return tuple(s for s in self.measurement_shocks if s in self.sv_shocks)


def _check_identifier(name: str, role: str) -> None:
    if not isinstance(name, str) or not _IDENT_RE.match(name):
        raise ModelStructureError(
            f"{role} name {name!r} is not a valid identifier (letters, digits and underscores, starting with a letter)."
        )
    if name.endswith("__"):
        raise ModelStructureError(f"{role} name {name!r} must not end in '__' (reserved by Stan).")
    if name in GRAMMAR_RESERVED:
        raise ModelStructureError(f"{role} name {name!r} is the grammar's own function name; choose another.")
    if role == "parameter" and name in RESERVED_NAMES:
        raise ModelStructureError(
            f"{role} name {name!r} is reserved (a Stan keyword, or a name the generated program uses); choose another."
        )


def _generic_point(params: tuple[str, ...]) -> dict[str, float]:
    """A fixed pseudo-random parameter point for GENERIC rank checks
    (rank is generic in the coefficient expressions; a special point --
    e.g. two parameters equal -- could only lower it)."""
    import random

    rng = random.Random(20260905)
    return {p: rng.uniform(0.37, 1.91) * rng.choice((-1.0, 1.0)) for p in params}


def _rank(rows: list[list[float]], tol: float = 1e-9) -> int:
    """Rank of a small dense matrix by Gaussian elimination with partial
    pivoting (pure Python: specs.schema stays numpy-free)."""
    A = [list(r) for r in rows]
    if not A or not A[0]:
        return 0
    n_rows, n_cols = len(A), len(A[0])
    scale = max((abs(v) for r in A for v in r), default=0.0) or 1.0
    rank = 0
    col = 0
    while rank < n_rows and col < n_cols:
        piv = max(range(rank, n_rows), key=lambda i: abs(A[i][col]))
        if abs(A[piv][col]) <= tol * scale:
            col += 1
            continue
        A[rank], A[piv] = A[piv], A[rank]
        for i in range(rank + 1, n_rows):
            f = A[i][col] / A[rank][col]
            if f != 0.0:
                for j in range(col, n_cols):
                    A[i][j] -= f * A[rank][j]
        rank += 1
        col += 1
    return rank


def compile_structure(
    *,
    name: str,
    observables: list[str],
    exogenous: list[str],
    measurement: list[str],
    transition: list[str],
    parameters: list[str],
    shocks: dict[str, dict],
) -> ModelStructure:
    """Derive the :class:`ModelStructure` (see module doc). ``shocks`` maps
    shock name -> ``{"sd": <param>}`` or ``{"sv": True}``."""
    # --- names ---------------------------------------------------------
    for role, names in (("observable", observables), ("exogenous series", exogenous), ("parameter", parameters), ("shock", list(shocks))):
        for n in names:
            _check_identifier(n, role)
    if not observables:
        raise ModelStructureError("An authored model needs at least one observable.")
    parsed_meas = [parse_equation(e) for e in measurement]
    parsed_trans = [parse_equation(e) for e in transition]
    states = [p.lhs_name for p in parsed_trans]
    for s in states:
        _check_identifier(s, "state")
    all_names: list[tuple[str, str]] = (
        [(n, "observable") for n in observables] + [(n, "exogenous series") for n in exogenous]
        + [(n, "state") for n in states] + [(n, "shock") for n in shocks] + [(n, "parameter") for n in parameters]
    )
    seen: dict[str, str] = {}
    for n, role in all_names:
        if n in seen:
            raise ModelStructureError(
                f"Name {n!r} is declared twice ({seen[n]} and {role}); observables, exogenous series, "
                f"states, shocks and parameters must all have distinct names."
            )
        seen[n] = role

    if len(parsed_meas) != len(observables):
        raise ModelStructureError(
            f"Expected one measurement equation per observable ({len(observables)}: {observables}); got {len(parsed_meas)}."
        )
    for i, (p, obs) in enumerate(zip(parsed_meas, observables)):
        if p.lhs_name != obs or p.lhs_lag != 0:
            raise ModelStructureError(
                f"Measurement equation {i} must define observable {obs!r} (declared order; a bare LHS), got LHS {p.emit().split(' = ')[0]!r}."
            )
    kinds = {**{n: "observable" for n in observables}, **{n: "exogenous" for n in exogenous}, **{n: "state" for n in states}, **{n: "shock" for n in shocks}, **{n: "param" for n in parameters}}
    kind_of = kinds.get

    meas_forms: list[LinearForm] = [linearize(p.rhs, kind_of, equation=text) for text, p in zip(measurement, parsed_meas)]
    trans_forms: list[LinearForm] = [linearize(p.rhs, kind_of, equation=text) for text, p in zip(transition, parsed_trans)]

    head_offset = {p.lhs_name: p.lhs_lag for p in parsed_trans}
    obs_set, exog_set, state_set, shock_set = set(observables), set(exogenous), set(states), set(shocks)
    obs_index = {o: i for i, o in enumerate(observables)}

    # --- shock classification -----------------------------------------
    in_trans: dict[str, list[str]] = {}
    in_meas: dict[str, list[int]] = {}
    for text, lf, p in zip(transition, trans_forms, parsed_trans):
        for t in lf.terms:
            if isinstance(t, SeriesTerm) and t.name in shock_set:
                if t.lag != 0:
                    raise ModelStructureError(
                        f"Equation {text!r}: shock {t.name!r} appears lagged ({t.name}[-{t.lag}]) -- shocks are contemporaneous; "
                        f"an MA term needs an auxiliary state (write the lagged shock as a state whose equation is 'aux = {t.name}')."
                    )
                in_trans.setdefault(t.name, []).append(p.lhs_name)
    for i, (text, lf) in enumerate(zip(measurement, meas_forms)):
        for t in lf.terms:
            if isinstance(t, SeriesTerm) and t.name in shock_set:
                if t.lag != 0:
                    raise ModelStructureError(f"Equation {text!r}: shock {t.name!r} appears lagged -- shocks are contemporaneous.")
                in_meas.setdefault(t.name, []).append(i)
    for text, p in zip(measurement + transition, parsed_meas + parsed_trans):
        for nm, cnt in name_occurrences(p.rhs).items():
            if nm in shock_set and cnt > 1:
                raise ModelStructureError(
                    f"Equation {text!r} references shock {nm!r} {cnt} times -- a shock appears once per equation "
                    f"(writing it twice would silently scale its variance; use a distinct shock or a coefficient)."
                )
    for sh in shocks:
        if sh in in_trans and sh in in_meas:
            raise ModelStructureError(
                f"Shock {sh!r} appears in a transition equation AND a measurement equation -- a shock is either a state "
                f"shock or a measurement shock (a shared shock would correlate Q and R, outside this stage)."
            )
        if sh not in in_trans and sh not in in_meas:
            raise ModelStructureError(f"Shock {sh!r} is declared but appears in no equation.")
    own_shock: dict[int, str] = {}  # observation row -> its own measurement shock
    for i, (text, lf) in enumerate(zip(measurement, meas_forms)):
        here = [s for s in shocks if i in in_meas.get(s, [])]
        if len(here) > 1:
            raise ModelStructureError(
                f"Measurement equation {text!r} carries {len(here)} measurement shocks {here}; a row has at most ONE shock of its "
                f"own (a second orthogonal shock reaching the row must come through another observable's equation, S8 E5)."
            )
        if here:
            coef = lf.terms[SeriesTerm(here[0], 0)]
            if not (is_numeric(coef) and numeric_value(coef) == 1.0):
                raise ModelStructureError(
                    f"Measurement shock {here[0]!r} in {text!r} must enter with unit coefficient (found {coef.emit()}); "
                    f"scale it through its sd parameter instead."
                )
            own_shock[i] = here[0]
    for s, rows in in_meas.items():
        if len(rows) != 1:
            raise ModelStructureError(f"Measurement shock {s!r} appears in {len(rows)} measurement equations; one row per shock.")
    meas_shocks: list[str] = [own_shock[i] for i in range(len(observables)) if i in own_shock]
    # State shocks in order of first appearance across transition equations.
    state_shocks: list[str] = []
    for lf in trans_forms:
        for t in lf.terms:
            if isinstance(t, SeriesTerm) and t.name in in_trans and t.name not in state_shocks:
                state_shocks.append(t.name)
    if not state_shocks and not meas_shocks:
        raise ModelStructureError(
            "The model has no state shock and no measurement shock -- nothing is stochastic; declare a shock."
        )

    # --- slot counts ---------------------------------------------------
    n_slots = {s: 1 for s in states}

    def need_meas(s: str, k: int, text: str) -> None:
        d = head_offset[s]
        if k < d:
            raise ModelStructureError(
                f"Equation {text!r} references {s}[-{k}] but state {s!r} is carried lagged (its transition equation is "
                f"written for {s}[-{d}]), so {s}[-{k}] is not in the state vector. Reference {s}[-{d}] or later, or write "
                f"the transition equation for {s!r} itself."
            )
        n_slots[s] = max(n_slots[s], k - d + 1)

    def need_trans(s: str, k: int, text: str) -> str:
        d = head_offset[s]
        if k < d:
            raise ModelStructureError(
                f"Equation {text!r} references {s}[-{k}] but state {s!r} is carried lagged (written for {s}[-{d}]); "
                f"{s}[-{k}] is not in the state vector."
            )
        if k == d:
            return "head"
        n_slots[s] = max(n_slots[s], k - d)
        return "lag"

    # --- feedback map (x columns): the Const column first when any
    # measurement equation has an intercept (plans/S8-plan.md conflict 5),
    # then first-appearance order ---------------------------------------
    feedback: list[FeedbackTuple] = []
    if any(lf.const is not None for lf in meas_forms):
        feedback.append(("const",))
    lag_depth = 0
    obs_deps: dict[int, dict[int, Expr]] = {i: {} for i in range(len(observables))}  # row -> {row of the contemporaneous observable: coef}

    def data_column(t, text: str) -> FeedbackTuple:
        """Register the feedback column of a DATA term (a mean(), a lagged
        observable, or an exogenous series at any lag) and return it."""
        nonlocal lag_depth
        if isinstance(t, MeanTerm):
            if t.name not in obs_set:
                raise ModelStructureError(
                    f"Equation {text!r}: mean() is only supported over lags of an OBSERVABLE (it becomes one "
                    f"ObsLagMean regressor column); {t.name!r} is not an observable."
                )
            fb = ("obs_lag_mean", t.name, tuple(t.lags))
            lag_depth = max(lag_depth, max(t.lags))
        elif t.name in obs_set:
            fb = ("obs_lag", t.name, t.lag)
            lag_depth = max(lag_depth, t.lag)
        else:
            fb = ("exog_lag", t.name, t.lag)  # lag 0 = contemporaneous (E2)
            lag_depth = max(lag_depth, t.lag)
        if fb not in feedback:
            feedback.append(fb)
        return fb

    for i, (text, lf) in enumerate(zip(measurement, meas_forms)):
        for t, coef in lf.terms.items():
            if isinstance(t, ProductTerm):  # S9 E4: state x data -> a data-dependent loading
                need_meas(t.state.name, t.state.lag, text)
                data_column(t.data, text)
            elif isinstance(t, MeanTerm):
                data_column(t, text)
            elif t.name in state_set:
                need_meas(t.name, t.lag, text)
            elif t.name in obs_set:
                if t.lag == 0:
                    if t.name == observables[i]:
                        raise ModelStructureError(f"Equation {text!r}: observable {t.name!r} references itself contemporaneously.")
                    obs_deps[i][obs_index[t.name]] = coef  # E5: substituted below
                    continue
                data_column(t, text)
            elif t.name in exog_set:
                data_column(t, text)
            elif t.name in shock_set:
                pass
            else:  # pragma: no cover -- linearize rejects unknown names
                raise ModelStructureError(f"Unknown name {t.name!r} in {text!r}.")

    ref_kind: list[dict[SeriesTerm, str]] = []
    for text, lf, p in zip(transition, trans_forms, parsed_trans):
        kinds_here: dict[SeriesTerm, str] = {}
        for t in lf.terms:
            if isinstance(t, ProductTerm):
                raise ModelStructureError(
                    f"Equation {text!r}: {t.state.name}'s transition multiplies a state by data ({t.data.name}) -- a "
                    f"time-varying TRANSITION (a data-dependent F_t) is outside this stage's scope (S10, E6); a "
                    f"time-varying coefficient (S9, E4) is a state multiplied by data in a MEASUREMENT equation."
                )
            if isinstance(t, MeanTerm) or t.name in obs_set or t.name in exog_set:
                what = "mean()" if isinstance(t, MeanTerm) else ("observable" if t.name in obs_set else "exogenous series")
                raise ModelStructureError(
                    f"Equation {text!r}: transition equations must be linear in states and state shocks only; "
                    f"{t.name!r} is an {what} (the KF's state equation xi_t = F xi_(t-1) + w_t carries no regressor "
                    f"block in this stage -- move the observable's effect into a measurement equation or model it as a state)."
                )
            if t.name in state_set:
                kinds_here[t] = need_trans(t.name, t.lag, text)
            else:
                kinds_here[t] = "shock"
        ref_kind.append(kinds_here)

    # --- slot layout (the implicit _const state last, E1 drifts) ------
    has_const_state = any(lf.const is not None for lf in trans_forms)
    labels: list[tuple[str, int]] = []
    for s in states:
        d = head_offset[s]
        for j in range(n_slots[s]):
            labels.append((s, -(d + j)))
    if has_const_state:
        labels.append((CONST_STATE, 0))
    slot_index = {lab: i for i, lab in enumerate(labels)}

    # --- F rows + loadings (topological substitution order) ------------
    deps: dict[str, set[str]] = {s: set() for s in states}
    for p, kinds_here in zip(parsed_trans, ref_kind):
        for t, kind in kinds_here.items():
            if kind == "head":
                deps[p.lhs_name].add(t.name)
    order: list[str] = []
    resolved: set[str] = set()
    pending = list(states)
    while pending:
        progress = False
        for s in list(pending):
            if deps[s] <= resolved:
                order.append(s)
                resolved.add(s)
                pending.remove(s)
                progress = True
        if not progress:
            raise ModelStructureError(
                f"Transition equations reference each other contemporaneously in a cycle among states {sorted(pending)} "
                f"(each references the other's current value); a simultaneous system must be solved by hand before authoring."
            )

    F: dict[tuple[int, int], Expr] = {}
    loadings: dict[str, dict[tuple[str, int], Expr]] = {sh: {} for sh in state_shocks}
    row_of = {s: slot_index[(s, -head_offset[s])] for s in states}
    text_of = {p.lhs_name: text for p, text in zip(parsed_trans, transition)}
    form_of = {p.lhs_name: (lf, kinds_here) for p, lf, kinds_here in zip(parsed_trans, trans_forms, ref_kind)}

    def add_entry(store: dict, key, value: Expr) -> None:
        store[key] = _add_coef(store[key], value) if key in store else value

    for s in order:
        lf, kinds_here = form_of[s]
        r = row_of[s]
        if lf.const is not None:  # E1 drift: a loading on the unit state
            add_entry(F, (r, slot_index[(CONST_STATE, 0)]), lf.const)
        for t, kind in kinds_here.items():
            coef = lf.terms[t]
            if kind == "lag":
                add_entry(F, (r, slot_index[(t.name, -(t.lag - 1))]), coef)
            elif kind == "head":
                r2 = row_of[t.name]
                for (rr, cc), val in list(F.items()):
                    if rr == r2:
                        add_entry(F, (r, cc), _simplify_mul(coef, val))
                for sh in state_shocks:
                    if (t.name, -head_offset[t.name]) in loadings[sh]:
                        if not is_numeric(coef):
                            raise ModelStructureError(
                                f"Equation {text_of[s]!r} references the current value of state {t.name!r} with coefficient "
                                f"{coef.emit()}, which would make shock {sh!r}'s loading into {s!r} parameter-dependent; "
                                f"shock loadings must be numeric constants in this stage (StateSpaceMeta declares them as data)."
                            )
                        add_entry(loadings[sh], (s, -head_offset[s]), _simplify_mul(coef, loadings[sh][(t.name, -head_offset[t.name])]))
            else:  # shock
                if not is_numeric(coef):
                    raise ModelStructureError(
                        f"Equation {text_of[s]!r}: shock {t.name!r} enters with coefficient {coef.emit()} -- shock loadings "
                        f"must be numeric constants in this stage (put the scale in the shock's sd parameter)."
                    )
                add_entry(loadings[t.name], (s, -head_offset[s]), coef)
    # Lag-copy rows.
    for s in states:
        d = head_offset[s]
        for j in range(1, n_slots[s]):
            F[(slot_index[(s, -(d + j))], slot_index[(s, -(d + j - 1))])] = Num(1.0)
    if has_const_state:
        c = slot_index[(CONST_STATE, 0)]
        F[(c, c)] = Num(1.0)

    numeric_loadings: dict[str, dict[tuple[str, int], float]] = {}
    for sh in state_shocks:
        entries = {lab: numeric_value(c) for lab, c in loadings[sh].items()}
        entries = {lab: v for lab, v in entries.items() if v != 0.0}
        if not entries:
            raise ModelStructureError(f"State shock {sh!r} has an all-zero loading into the state vector.")
        numeric_loadings[sh] = entries

    # --- Z, A, M: own rows, then E5 substitution in dependency order ----
    col_index = {fb: j for j, fb in enumerate(feedback)}
    shock_col = {s: j for j, s in enumerate(meas_shocks)}
    own_Z: list[dict[int, Expr]] = [{} for _ in observables]
    own_A: list[dict[int, Expr]] = [{} for _ in observables]
    own_M: list[dict[int, Expr]] = [{} for _ in observables]
    own_Zx: list[dict[tuple[int, int], Expr]] = [{} for _ in observables]  # (x column, slot) -> coef (S9 E4)

    def fb_of(t) -> FeedbackTuple:
        if isinstance(t, MeanTerm):
            return ("obs_lag_mean", t.name, tuple(t.lags))
        if t.name in obs_set:
            return ("obs_lag", t.name, t.lag)
        return ("exog_lag", t.name, t.lag)

    for i, lf in enumerate(meas_forms):
        if lf.const is not None:
            add_entry(own_A[i], col_index[("const",)], lf.const)
        if i in own_shock:
            own_M[i][shock_col[own_shock[i]]] = Num(1.0)
        for t, coef in lf.terms.items():
            if isinstance(t, ProductTerm):
                add_entry(own_Zx[i], (col_index[fb_of(t.data)], slot_index[(t.state.name, -t.state.lag)]), coef)
            elif isinstance(t, MeanTerm):
                add_entry(own_A[i], col_index[("obs_lag_mean", t.name, tuple(t.lags))], coef)
            elif t.name in state_set:
                add_entry(own_Z[i], slot_index[(t.name, -t.lag)], coef)
            elif t.name in obs_set:
                if t.lag > 0:
                    add_entry(own_A[i], col_index[("obs_lag", t.name, t.lag)], coef)
            elif t.name in exog_set:
                add_entry(own_A[i], col_index[("exog_lag", t.name, t.lag)], coef)
    # Topological order over observables (a cycle is an error naming them).
    sub_order: list[int] = []
    resolved_obs: set[int] = set()
    pending_obs = list(range(len(observables)))
    while pending_obs:
        progress = False
        for i in list(pending_obs):
            if set(obs_deps[i]) <= resolved_obs:
                sub_order.append(i)
                resolved_obs.add(i)
                pending_obs.remove(i)
                progress = True
        if not progress:
            raise ModelStructureError(
                f"Measurement equations reference each other contemporaneously in a cycle among observables "
                f"{[observables[i] for i in pending_obs]}; a recursive (Cholesky-ordered) system is acyclic -- each observable "
                f"may depend contemporaneously only on observables earlier in the ordering."
            )
    rows_Z: list[dict[int, Expr]] = [dict(r) for r in own_Z]
    rows_A: list[dict[int, Expr]] = [dict(r) for r in own_A]
    rows_M: list[dict[int, Expr]] = [dict(r) for r in own_M]
    rows_Zx: list[dict[tuple[int, int], Expr]] = [dict(r) for r in own_Zx]
    for i in sub_order:
        for j, coef in obs_deps[i].items():  # rows_*[j] are already fully composed
            for store_i, store_j in ((rows_Z[i], rows_Z[j]), (rows_A[i], rows_A[j]), (rows_M[i], rows_M[j]), (rows_Zx[i], rows_Zx[j])):
                for k, val in store_j.items():
                    add_entry(store_i, k, _simplify_mul(coef, val))
    Z: dict[tuple[int, int], Expr] = {(i, k): v for i, row in enumerate(rows_Z) for k, v in row.items()}
    Zx: dict[tuple[int, int], tuple[tuple[int, Expr], ...]] = {}
    for i, row in enumerate(rows_Zx):
        by_slot: dict[int, list[tuple[int, Expr]]] = {}
        for (j, k), v in row.items():
            by_slot.setdefault(k, []).append((j, v))
        for k, terms in by_slot.items():
            Zx[(i, k)] = tuple(sorted(terms, key=lambda jt: jt[0]))
    A: dict[tuple[int, int], Expr] = {(k, i): v for i, row in enumerate(rows_A) for k, v in row.items()}
    M: dict[tuple[int, int], Expr] = {(i, j): v for i, row in enumerate(rows_M) for j, v in row.items()}
    meas_loadings: dict[str, tuple[str, ...]] = {}
    row_of_shock = {sh: i for i, sh in own_shock.items()}
    for s in meas_shocks:
        j = shock_col[s]
        own_row = observables[row_of_shock[s]]
        others = [observables[i] for i in range(len(observables)) if (i, j) in M and observables[i] != own_row]
        meas_loadings[s] = (own_row, *others)

    # --- shock scales --------------------------------------------------
    scale_param: dict[str, str] = {}
    sv: list[str] = []
    for sh, decl in shocks.items():
        if decl.get("sv"):
            sv.append(sh)
        else:
            scale_param[sh] = decl["sd"]

    # --- E3: shock-free rows keep the innovation covariance PD ---------
    free_rows = [i for i in range(len(observables)) if i not in {r for (r, _) in M}]
    if free_rows:
        point = _generic_point(tuple(parameters) + tuple(f"__x{j}" for j in range(len(feedback))))
        xg = [point[f"__x{j}"] for j in range(len(feedback))]  # generic regressor values (S9 E4)
        n_state = len(labels)
        B_cols = [[float(numeric_loadings[sh].get(lab, 0.0)) for lab in labels] for sh in state_shocks]  # per shock: n-vector
        rows = []
        for i in range(len(observables)):
            m_part = [evaluate(M[(i, j)], point) if (i, j) in M else 0.0 for j in range(len(meas_shocks))]
            z_row = [evaluate(Z[(i, k)], point) if (i, k) in Z else 0.0 for k in range(n_state)]
            for k in range(n_state):
                for j, e in Zx.get((i, k), ()):
                    z_row[k] += xg[j] * evaluate(e, point)
            zb_part = [sum(z_row[k] * b[k] for k in range(n_state)) for b in B_cols]
            rows.append(m_part + zb_part)
        if _rank(rows) < len(observables):
            names_free = [observables[i] for i in free_rows]
            raise ModelStructureError(
                f"Measurement row(s) {names_free} carry no shock (singular R), and the innovation covariance "
                f"S_t = Z P Z' + R would be singular: rank([M | Z B]) < {len(observables)} -- a shock-free row must load a "
                f"state driven by a state shock that no other shock-free row already explains (plans/S8-plan.md, E3)."
            )

    # --- parameter usage -----------------------------------------------
    used: set[str] = set()
    for store in (F, A, Z, M):
        for e in store.values():
            used |= parameters_in(e)
    for terms in Zx.values():
        for _, e in terms:
            used |= parameters_in(e)
    doubled = sorted(used & set(scale_param.values()))
    if doubled:
        raise ModelStructureError(
            f"Parameter(s) {doubled} scale a shock AND appear as a coefficient in an equation -- a shock's sd parameter "
            f"is its own scale (declare a separate parameter for the coefficient)."
        )
    used |= set(scale_param.values())
    unused = [p for p in parameters if p not in used]
    if unused:
        raise ModelStructureError(
            f"Parameter(s) {unused} are declared but appear in no equation and scale no shock -- an unused parameter "
            f"would be sampled from its prior alone; remove it or use it."
        )

    # --- S9 E4: coefficient-only shocks (structural reachability) --------
    coefficient_shocks: list[str] = []
    if Zx:
        additive_slots = {k for (_, k) in Z}
        coef_slots = {k for (_, k) in Zx}
        succ: dict[int, set[int]] = {}
        for (r, c) in F:  # slot c at t-1 feeds slot r at t
            succ.setdefault(c, set()).add(r)
        for sh in state_shocks:
            start = {slot_index[lab] for lab in numeric_loadings[sh]}
            reach = set(start)
            frontier = list(start)
            while frontier:
                s = frontier.pop()
                for r in succ.get(s, ()):
                    if r not in reach:
                        reach.add(r)
                        frontier.append(r)
            if not (reach & additive_slots) and (reach & coef_slots):
                coefficient_shocks.append(sh)

    return ModelStructure(
        name=name,
        obs_names=tuple(observables),
        exog_names=tuple(exogenous),
        state_names=tuple(states),
        head_offset=head_offset,
        n_slots=n_slots,
        state_labels=tuple(labels),
        state_shocks=tuple(state_shocks),
        measurement_shocks=tuple(meas_shocks),
        shock_loadings=numeric_loadings,
        feedback_map=tuple(feedback),
        F=F,
        A=A,
        Z=Z,
        M=M,
        meas_loadings=meas_loadings,
        params=tuple(parameters),
        shock_scale_param=scale_param,
        sv_shocks=tuple(sv),
        lag_depth=lag_depth,
        has_const_state=has_const_state,
        obs_substitution_order=tuple(observables[i] for i in sub_order),
        canonical_measurement=tuple(p.emit() for p in parsed_meas),
        canonical_transition=tuple(p.emit() for p in parsed_trans),
        Zx=Zx,
        coefficient_shocks=tuple(coefficient_shocks),
    )
