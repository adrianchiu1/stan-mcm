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
  data); a measurement equation carries exactly one measurement shock at
  unit coefficient.

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
    LinearForm,
    MeanTerm,
    Num,
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
    Qt Rt kf_loglik mean pi e sqrt exp log square rep_matrix rep_array diag_matrix cumulative_sum
    kalman_loglik sv_rw_noncentered sv_scalar_variance_path
    """.split()
)

#: Names no series/shock may use (the grammar's own function).
GRAMMAR_RESERVED = frozenset({"mean"})

#: Feedback-map terms as plain tuples (converted to ObsLag / ObsLagMean /
#: ExogLag by the numpy side).
FeedbackTuple = tuple


@dataclass
class ModelStructure:
    """The compiled structure of one authored model (see module doc)."""

    name: str
    obs_names: tuple[str, ...]
    exog_names: tuple[str, ...]
    state_names: tuple[str, ...]
    head_offset: dict[str, int]
    n_slots: dict[str, int]
    state_labels: tuple[tuple[str, int], ...]
    state_shocks: tuple[str, ...]
    measurement_shocks: tuple[str, ...]
    shock_loadings: dict[str, dict[tuple[str, int], float]]
    feedback_map: tuple[FeedbackTuple, ...]
    #: Sparse symbolic matrices: missing entries are exact zeros.
    F: dict[tuple[int, int], Expr]
    A: dict[tuple[int, int], Expr]  # (x column, observation row)
    Z: dict[tuple[int, int], Expr]  # (observation row, state slot)
    params: tuple[str, ...]
    #: Constant-scale shocks -> the half_normal parameter carrying their sd.
    shock_scale_param: dict[str, str]
    sv_shocks: tuple[str, ...]
    lag_depth: int
    canonical_measurement: tuple[str, ...] = field(default_factory=tuple)
    canonical_transition: tuple[str, ...] = field(default_factory=tuple)

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
        entries = {"F": self.F, "A": self.A, "Z": self.Z}[which]
        return all(is_numeric(e) for e in entries.values())

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
    if not transition:
        raise ModelStructureError("An authored model needs at least one transition equation (a state).")
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
    kinds = {**{n: "symbol" for n in observables}, **{n: "symbol" for n in exogenous}, **{n: "symbol" for n in states}, **{n: "symbol" for n in shocks}, **{n: "param" for n in parameters}}
    kind_of = kinds.get

    meas_forms: list[LinearForm] = []
    for text, p in zip(measurement, parsed_meas):
        lf = linearize(p.rhs, kind_of, equation=text)
        if lf.const is not None:
            raise ModelStructureError(
                f"Equation {text!r} has a constant term ({lf.const.emit()}) -- intercepts are outside this stage "
                f"(the KF measurement equation carries no constant; add a constant state or demean the series)."
            )
        meas_forms.append(lf)
    trans_forms: list[LinearForm] = []
    for text, p in zip(transition, parsed_trans):
        lf = linearize(p.rhs, kind_of, equation=text)
        if lf.const is not None:
            raise ModelStructureError(
                f"Equation {text!r} has a constant term ({lf.const.emit()}) -- intercepts/drifts are outside this stage "
                f"(model a drift as a state, e.g. a random-walk growth state)."
            )
        trans_forms.append(lf)

    head_offset = {p.lhs_name: p.lhs_lag for p in parsed_trans}
    obs_set, exog_set, state_set, shock_set = set(observables), set(exogenous), set(states), set(shocks)

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
    meas_shocks: list[str] = []
    for i, (text, lf) in enumerate(zip(measurement, meas_forms)):
        here = [s for s in shocks if i in in_meas.get(s, [])]
        if len(here) != 1:
            raise ModelStructureError(
                f"Measurement equation {text!r} must carry exactly one measurement shock (found {here}); a shock-free "
                f"measurement row (singular R) or several shocks on one row are outside this stage."
            )
        coef = lf.terms[SeriesTerm(here[0], 0)]
        if not (is_numeric(coef) and numeric_value(coef) == 1.0):
            raise ModelStructureError(
                f"Measurement shock {here[0]!r} in {text!r} must enter with unit coefficient (found {coef.emit()}); "
                f"scale it through its sd parameter instead."
            )
        meas_shocks.append(here[0])
    for s in meas_shocks:
        if len(in_meas[s]) != 1:
            raise ModelStructureError(f"Measurement shock {s!r} appears in {len(in_meas[s])} measurement equations; one row per shock.")
    # State shocks in order of first appearance across transition equations.
    state_shocks: list[str] = []
    for lf in trans_forms:
        for t in lf.terms:
            if isinstance(t, SeriesTerm) and t.name in in_trans and t.name not in state_shocks:
                state_shocks.append(t.name)

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

    feedback: list[FeedbackTuple] = []
    lag_depth = 0
    for text, lf in zip(measurement, meas_forms):
        for t in lf.terms:
            if isinstance(t, MeanTerm):
                if t.name not in obs_set:
                    raise ModelStructureError(
                        f"Equation {text!r}: mean() is only supported over lags of an OBSERVABLE (it becomes one "
                        f"ObsLagMean regressor column); {t.name!r} is not an observable."
                    )
                fb = ("obs_lag_mean", t.name, tuple(t.lags))
                if fb not in feedback:
                    feedback.append(fb)
                lag_depth = max(lag_depth, max(t.lags))
            elif t.name in state_set:
                need_meas(t.name, t.lag, text)
            elif t.name in obs_set:
                if t.lag == 0:
                    raise ModelStructureError(
                        f"Equation {text!r}: observable {t.name!r} appears contemporaneously on the right-hand side -- "
                        f"simultaneous observables are outside the DSL (the measurement equation must be y_t = A'x_t + Z xi_t + e_t; "
                        f"substitute the other equation or lag the reference)."
                    )
                fb = ("obs_lag", t.name, t.lag)
                if fb not in feedback:
                    feedback.append(fb)
                lag_depth = max(lag_depth, t.lag)
            elif t.name in exog_set:
                if t.lag == 0:
                    raise ModelStructureError(
                        f"Equation {text!r}: exogenous series {t.name!r} appears contemporaneously -- the feedback map "
                        f"requires strictly lagged regressors (ExogLag with lag >= 1); write {t.name}[-1] (and date the "
                        f"series accordingly)."
                    )
                fb = ("exog_lag", t.name, t.lag)
                if fb not in feedback:
                    feedback.append(fb)
                lag_depth = max(lag_depth, t.lag)
            elif t.name in shock_set:
                pass
            else:  # pragma: no cover -- linearize rejects unknown names
                raise ModelStructureError(f"Unknown name {t.name!r} in {text!r}.")

    ref_kind: list[dict[SeriesTerm, str]] = []
    for text, lf, p in zip(transition, trans_forms, parsed_trans):
        kinds_here: dict[SeriesTerm, str] = {}
        for t in lf.terms:
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

    # --- slot layout ---------------------------------------------------
    labels: list[tuple[str, int]] = []
    for s in states:
        d = head_offset[s]
        for j in range(n_slots[s]):
            labels.append((s, -(d + j)))
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

    numeric_loadings: dict[str, dict[tuple[str, int], float]] = {}
    for sh in state_shocks:
        entries = {lab: numeric_value(c) for lab, c in loadings[sh].items()}
        entries = {lab: v for lab, v in entries.items() if v != 0.0}
        if not entries:
            raise ModelStructureError(f"State shock {sh!r} has an all-zero loading into the state vector.")
        numeric_loadings[sh] = entries

    # --- Z, A ----------------------------------------------------------
    Z: dict[tuple[int, int], Expr] = {}
    A: dict[tuple[int, int], Expr] = {}
    col_index = {fb: j for j, fb in enumerate(feedback)}
    for i, lf in enumerate(meas_forms):
        for t, coef in lf.terms.items():
            if isinstance(t, MeanTerm):
                add_entry(A, (col_index[("obs_lag_mean", t.name, tuple(t.lags))], i), coef)
            elif t.name in state_set:
                add_entry(Z, (i, slot_index[(t.name, -t.lag)]), coef)
            elif t.name in obs_set:
                add_entry(A, (col_index[("obs_lag", t.name, t.lag)], i), coef)
            elif t.name in exog_set:
                add_entry(A, (col_index[("exog_lag", t.name, t.lag)], i), coef)

    # --- shock scales --------------------------------------------------
    scale_param: dict[str, str] = {}
    sv: list[str] = []
    for sh, decl in shocks.items():
        if decl.get("sv"):
            sv.append(sh)
        else:
            scale_param[sh] = decl["sd"]

    # --- parameter usage -----------------------------------------------
    used: set[str] = set()
    for store in (F, A, Z):
        for e in store.values():
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
        params=tuple(parameters),
        shock_scale_param=scale_param,
        sv_shocks=tuple(sv),
        lag_depth=lag_depth,
        canonical_measurement=tuple(p.emit() for p in parsed_meas),
        canonical_transition=tuple(p.emit() for p in parsed_trans),
    )
