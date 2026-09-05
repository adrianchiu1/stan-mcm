"""The equation grammar of the authored-model DSL (S7): tokenizer, parser,
expression tree, canonical re-emission, and the LINEARIZATION that turns a
parsed right-hand side into a linear form over series/shock terms with
coefficient expressions over parameters and numbers.

Pure Python (no numpy) and spec-side on purpose: ``specs.schema`` must stay
importable without the numerics stack, and the authored family's Pydantic
options model (``specs/schema/authored.py``) parses and canonicalizes every
equation at validation time so a malformed equation fails where every
other spec error fails -- naming the equation and the construct.

Grammar (plans/S7-plan.md):

    equation := lhs "=" expr
    lhs      := NAME | NAME "[" "-" INT "]"
    expr     := linear combination with + - * / ( ) and unary minus of
                NAME | NAME "[" "-" INT "]" | mean(NAME[-k], ...) | NUMBER

Which NAMEs are series/shocks ("symbols") and which are parameters is
NOT a lexical property -- :func:`linearize` takes the classification from
the declaration (the options model knows the declared observables,
exogenous series, states, shocks and parameters). The parser is agnostic.

S9 (E4, plans/S9-plan.md): the linear form may carry PRODUCT terms -- a
state (any lag) multiplied by a data series (a lagged observable, a
``mean()`` of lags of one observable, or an exogenous series at any lag)
-- the one bilinear shape that compiles to a data-dependent measurement
loading ``Z_t``. :func:`linearize` classifies such a product when the
caller's ``kind_of`` distinguishes ``"state"`` / ``"observable"`` /
``"exogenous"`` / ``"shock"``; every other product of symbol-carrying
factors keeps the S7 rejection.

Canonical form (:meth:`Expr.emit`): re-emission from the tree with fixed
spacing and float formatting, TERM ORDER PRESERVED (plans/S7-plan.md open
question 2: term order determines regressor-column order, so it is
meaning, not formatting). The same printer produces the Stan expression
text for coefficient entries (numbers as ``repr(float)``, which Stan
accepts, parameters by name), so the Python evaluation and the Stan
program perform the SAME floating-point operations in the same order.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?|(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<op>[-+*/()\[\],=]))"
)
_NUM_RE = re.compile(r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")


class EquationSyntaxError(ValueError):
    """A malformed equation; the message names the equation text and the
    offending position/construct."""


@dataclass(frozen=True)
class _Tok:
    kind: str  # "num" | "name" | "op" | "end"
    text: str
    pos: int


def _tokenize(text: str) -> list[_Tok]:
    toks: list[_Tok] = []
    pos = 0
    n = len(text)
    while pos < n:
        if text[pos].isspace():
            pos += 1
            continue
        m = _NUM_RE.match(text, pos)
        if m:
            toks.append(_Tok("num", m.group(0), pos))
            pos = m.end()
            continue
        m = re.compile(r"[A-Za-z_][A-Za-z0-9_]*").match(text, pos)
        if m:
            toks.append(_Tok("name", m.group(0), pos))
            pos = m.end()
            continue
        ch = text[pos]
        if ch in "-+*/()[],=":
            toks.append(_Tok("op", ch, pos))
            pos += 1
            continue
        raise EquationSyntaxError(
            f"Equation {text!r}: unexpected character {ch!r} at position {pos}. "
            f"The grammar allows names, numbers, + - * / ( ) [ ] , and =."
        )
    toks.append(_Tok("end", "", n))
    return toks


# ---------------------------------------------------------------------------
# Expression tree
# ---------------------------------------------------------------------------


class Expr:
    """Base class of the parsed expression tree. Every node re-emits itself
    canonically via :meth:`emit`."""

    precedence = 0

    def emit(self) -> str:  # pragma: no cover -- abstract
        raise NotImplementedError

    def __str__(self) -> str:
        return self.emit()

    def _child(self, child: "Expr", *, right: bool = False) -> str:
        """Emit a child with parentheses whenever re-parsing could regroup
        it: a lower-precedence child always, and a SAME-precedence RIGHT
        child always (``a*(b/c)`` must not print as ``a*b/c``, which Stan
        and this parser read as ``(a*b)/c`` -- a different floating-point
        result; numerics-reviewer must-fix, S7). Parentheses are never
        wrong, only occasionally verbose."""
        text = child.emit()
        if child.precedence < self.precedence or (right and child.precedence == self.precedence):
            return f"({text})"
        return text


def _fmt_num(value: float) -> str:
    return repr(float(value))


@dataclass(frozen=True)
class Num(Expr):
    value: float
    precedence = 4

    def emit(self) -> str:
        return _fmt_num(self.value)


@dataclass(frozen=True)
class Name(Expr):
    """A bare name: a parameter, or a contemporaneous state/shock -- the
    classification is applied by :func:`linearize`."""

    name: str
    precedence = 4

    def emit(self) -> str:
        return self.name


@dataclass(frozen=True)
class LagRef(Expr):
    """``name[-lag]`` with ``lag >= 1``."""

    name: str
    lag: int
    precedence = 4

    def emit(self) -> str:
        return f"{self.name}[-{self.lag}]"


@dataclass(frozen=True)
class MeanRef(Expr):
    """``mean(name[-k1], name[-k2], ...)`` -- the plain mean of lags of ONE
    series (compiles to one ``ObsLagMean`` regressor column)."""

    name: str
    lags: tuple[int, ...]
    precedence = 4

    def emit(self) -> str:
        return "mean(" + ", ".join(f"{self.name}[-{k}]" for k in self.lags) + ")"


@dataclass(frozen=True)
class Neg(Expr):
    operand: Expr
    precedence = 3

    def emit(self) -> str:
        inner = self.operand.emit()
        if self.operand.precedence < 3:
            inner = f"({inner})"
        return f"-{inner}"


@dataclass(frozen=True)
class Add(Expr):
    left: Expr
    right: Expr
    precedence = 1

    def emit(self) -> str:
        return f"{self._child(self.left)} + {self._child(self.right, right=True)}"


@dataclass(frozen=True)
class Sub(Expr):
    left: Expr
    right: Expr
    precedence = 1

    def emit(self) -> str:
        return f"{self._child(self.left)} - {self._child(self.right, right=True)}"


@dataclass(frozen=True)
class Mul(Expr):
    left: Expr
    right: Expr
    precedence = 2

    def emit(self) -> str:
        return f"{self._child(self.left)}*{self._child(self.right, right=True)}"


@dataclass(frozen=True)
class Div(Expr):
    left: Expr
    right: Expr
    precedence = 2

    def emit(self) -> str:
        return f"{self._child(self.left)}/{self._child(self.right, right=True)}"


# ---------------------------------------------------------------------------
# Parser (recursive descent; standard precedence, left associative)
# ---------------------------------------------------------------------------


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.toks = _tokenize(text)
        self.i = 0

    def peek(self) -> _Tok:
        return self.toks[self.i]

    def take(self) -> _Tok:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def expect_op(self, op: str) -> _Tok:
        tok = self.peek()
        if tok.kind != "op" or tok.text != op:
            found = repr(tok.text) if tok.kind != "end" else "end of equation"
            raise EquationSyntaxError(
                f"Equation {self.text!r}: expected {op!r} at position {tok.pos}, found {found}."
            )
        return self.take()

    def parse_expr(self) -> Expr:
        node = self.parse_term()
        while self.peek().kind == "op" and self.peek().text in "+-":
            op = self.take().text
            rhs = self.parse_term()
            node = Add(node, rhs) if op == "+" else Sub(node, rhs)
        return node

    def parse_term(self) -> Expr:
        node = self.parse_unary()
        while self.peek().kind == "op" and self.peek().text in "*/":
            op = self.take().text
            rhs = self.parse_unary()
            node = Mul(node, rhs) if op == "*" else Div(node, rhs)
        return node

    def parse_unary(self) -> Expr:
        tok = self.peek()
        if tok.kind == "op" and tok.text == "-":
            self.take()
            return Neg(self.parse_unary())
        if tok.kind == "op" and tok.text == "+":
            self.take()
            return self.parse_unary()
        return self.parse_atom()

    def parse_lag_suffix(self, name: str) -> int | None:
        """``[-k]`` after a name, or None when absent."""
        tok = self.peek()
        if not (tok.kind == "op" and tok.text == "["):
            return None
        self.take()
        sign = self.peek()
        if sign.kind == "op" and sign.text == "-":
            self.take()
            num = self.peek()
            if num.kind != "num" or not num.text.isdigit():
                raise EquationSyntaxError(
                    f"Equation {self.text!r}: lag of {name!r} at position {num.pos} must be a "
                    f"positive integer written as {name}[-k]."
                )
            self.take()
            lag = int(num.text)
            if lag < 1:
                raise EquationSyntaxError(
                    f"Equation {self.text!r}: {name}[-0] is not a lag -- write the bare name "
                    f"{name!r} for the contemporaneous value."
                )
            self.expect_op("]")
            return lag
        if sign.kind == "num":
            raise EquationSyntaxError(
                f"Equation {self.text!r}: {name}[{sign.text}] at position {sign.pos} -- only "
                f"lags are supported, written {name}[-k] with k >= 1 (leads and "
                f"{name}[0] are outside the DSL; write the bare name for the "
                f"contemporaneous value)."
            )
        if sign.kind == "op" and sign.text == "+":
            raise EquationSyntaxError(
                f"Equation {self.text!r}: {name}[+k] at position {sign.pos} is a LEAD; the "
                f"DSL supports lags only ({name}[-k], k >= 1)."
            )
        raise EquationSyntaxError(
            f"Equation {self.text!r}: malformed lag suffix after {name!r} at position {sign.pos}; "
            f"expected {name}[-k]."
        )

    def parse_atom(self) -> Expr:
        tok = self.peek()
        if tok.kind == "num":
            self.take()
            return Num(float(tok.text))
        if tok.kind == "name":
            self.take()
            if tok.text == "mean" and self.peek().kind == "op" and self.peek().text == "(":
                return self.parse_mean()
            lag = self.parse_lag_suffix(tok.text)
            return Name(tok.text) if lag is None else LagRef(tok.text, lag)
        if tok.kind == "op" and tok.text == "(":
            self.take()
            node = self.parse_expr()
            self.expect_op(")")
            return node
        if tok.kind == "end":
            raise EquationSyntaxError(f"Equation {self.text!r}: unexpected end of equation (missing operand).")
        raise EquationSyntaxError(
            f"Equation {self.text!r}: unexpected {tok.text!r} at position {tok.pos}."
        )

    def parse_mean(self) -> Expr:
        self.expect_op("(")
        name: str | None = None
        lags: list[int] = []
        while True:
            tok = self.peek()
            if tok.kind != "name":
                raise EquationSyntaxError(
                    f"Equation {self.text!r}: mean(...) arguments must be lags of ONE series, "
                    f"e.g. mean(pi[-2], pi[-3], pi[-4]); found {tok.text!r} at position {tok.pos}."
                )
            self.take()
            lag = self.parse_lag_suffix(tok.text)
            if lag is None:
                raise EquationSyntaxError(
                    f"Equation {self.text!r}: mean(...) arguments must be LAGGED ({tok.text}[-k], k >= 1); "
                    f"a contemporaneous value inside mean() is outside the DSL (it would be a "
                    f"simultaneous regressor, not a feedback-map column)."
                )
            if name is None:
                name = tok.text
            elif tok.text != name:
                raise EquationSyntaxError(
                    f"Equation {self.text!r}: mean(...) must average lags of ONE series (it compiles "
                    f"to one ObsLagMean regressor column); found {name!r} and {tok.text!r}."
                )
            lags.append(lag)
            nxt = self.peek()
            if nxt.kind == "op" and nxt.text == ",":
                self.take()
                continue
            self.expect_op(")")
            break
        if len(lags) != len(set(lags)):
            raise EquationSyntaxError(f"Equation {self.text!r}: mean(...) lists a repeated lag {lags}.")
        return MeanRef(name, tuple(lags))  # type: ignore[arg-type]


@dataclass(frozen=True)
class ParsedEquation:
    """``lhs_name[-lhs_lag] = rhs``; ``lhs_lag`` is 0 for a bare LHS."""

    lhs_name: str
    lhs_lag: int
    rhs: Expr

    def emit(self) -> str:
        lhs = self.lhs_name if self.lhs_lag == 0 else f"{self.lhs_name}[-{self.lhs_lag}]"
        return f"{lhs} = {self.rhs.emit()}"


def parse_equation(text: str) -> ParsedEquation:
    """Parse one ``lhs = rhs`` equation string into its tree."""
    if not isinstance(text, str) or not text.strip():
        raise EquationSyntaxError(f"An equation must be a non-empty string of the form 'lhs = rhs'; got {text!r}.")
    toks = _tokenize(text)
    eq_positions = [i for i, t in enumerate(toks) if t.kind == "op" and t.text == "="]
    if len(eq_positions) != 1:
        raise EquationSyntaxError(
            f"Equation {text!r} must contain exactly one '=' (lhs = rhs); found {len(eq_positions)}."
        )
    split = eq_positions[0]
    lhs_toks = toks[:split] + [_Tok("end", "", toks[split].pos)]
    rhs_toks = toks[split + 1 :]

    p = _Parser.__new__(_Parser)
    p.text, p.toks, p.i = text, lhs_toks, 0
    tok = p.peek()
    if tok.kind != "name":
        raise EquationSyntaxError(
            f"Equation {text!r}: the left-hand side must be a series name, optionally lagged "
            f"(e.g. 'ystar' or 'g[-1]')."
        )
    p.take()
    lag = p.parse_lag_suffix(tok.text)
    if p.peek().kind != "end":
        raise EquationSyntaxError(
            f"Equation {text!r}: the left-hand side must be a single series name (optionally lagged)."
        )
    rp = _Parser.__new__(_Parser)
    rp.text, rp.toks, rp.i = text, rhs_toks, 0
    if rp.peek().kind == "end":
        raise EquationSyntaxError(f"Equation {text!r}: empty right-hand side.")
    rhs = rp.parse_expr()
    if rp.peek().kind != "end":
        bad = rp.peek()
        raise EquationSyntaxError(
            f"Equation {text!r}: unexpected {bad.text!r} at position {bad.pos}."
        )
    return ParsedEquation(lhs_name=tok.text, lhs_lag=lag or 0, rhs=rhs)


def canonical_equation(text: str) -> str:
    """The canonical re-emission of an equation string (the form that
    enters the spec's serialization and therefore the run-identity hash)."""
    return parse_equation(text).emit()


# ---------------------------------------------------------------------------
# Linearization: rhs tree -> linear form over symbol terms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeriesTerm:
    """A symbol reference: ``name`` at ``lag`` (0 = contemporaneous)."""

    name: str
    lag: int


@dataclass(frozen=True)
class MeanTerm:
    name: str
    lags: tuple[int, ...]


@dataclass(frozen=True)
class ProductTerm:
    """S9 E4: ``state * data`` -- ``state`` a :class:`SeriesTerm` naming a
    state at lag >= 0, ``data`` a :class:`SeriesTerm` (a lagged observable,
    lag >= 1, or an exogenous series, lag >= 0) or a :class:`MeanTerm`.
    Compiles to an entry of the data-dependent loading ``Zx_j`` on the
    data factor's feedback column ``j``."""

    state: SeriesTerm
    data: "SeriesTerm | MeanTerm"


Term = SeriesTerm | MeanTerm | ProductTerm

#: The ``kind_of`` answers :func:`linearize` treats as a series/shock (a
#: symbol-carrying factor); ``"symbol"`` is the S7 coarse answer, which may
#: not be multiplied by another symbol.
_SYMBOL_KINDS = frozenset({"symbol", "state", "observable", "exogenous", "shock"})


class LinearityError(ValueError):
    """The right-hand side is not linear in the declared series/shocks, or
    references an undeclared name."""


@dataclass
class LinearForm:
    """``sum_j coef_j * term_j + const`` with coefficient expressions over
    parameters and numbers. ``const`` is ``None`` when there is no constant
    part (the common case; an intercept is rejected downstream)."""

    terms: dict[Term, Expr]
    const: Expr | None = None

    @property
    def has_terms(self) -> bool:
        return bool(self.terms)


def _simplify_mul(a: Expr, b: Expr) -> Expr:
    """``a * b`` with the unit/negated-unit shortcuts that keep coefficient
    text readable and the arithmetic exact (1.0*x == x and (-1.0)*x == -x
    are exact in IEEE double, so the shortcut changes nothing numerically)."""
    if isinstance(a, Num) and a.value == 1.0:
        return b
    if isinstance(b, Num) and b.value == 1.0:
        return a
    if isinstance(a, Neg) and isinstance(a.operand, Num) and a.operand.value == 1.0:
        return _simplify_neg(b)
    if isinstance(b, Neg) and isinstance(b.operand, Num) and b.operand.value == 1.0:
        return _simplify_neg(a)
    return Mul(a, b)


def _simplify_neg(a: Expr) -> Expr:
    if isinstance(a, Neg):
        return a.operand
    return Neg(a)


def _add_coef(a: Expr, b: Expr) -> Expr:
    """``a + b``, written ``a - c`` when ``b`` is ``-c`` (a + (-c) and a - c
    are the same IEEE operation, so this changes only the text)."""
    if isinstance(b, Neg):
        return Sub(a, b.operand)
    return Add(a, b)


def _add_opt(a: Expr | None, b: Expr | None) -> Expr | None:
    if a is None:
        return b
    if b is None:
        return a
    return _add_coef(a, b)


def _product_term(a: Term, b: Term, kind_of: Callable[[str], str | None]) -> ProductTerm | None:
    """The E4 classification of ``a * b``: exactly one factor a STATE
    (any lag) and the other a data series -- a lagged observable
    (lag >= 1), a ``mean()`` of one observable, or an exogenous series at
    any lag (0 included, E2). ``None`` for every other pair."""
    def is_state(t: Term) -> bool:
        return isinstance(t, SeriesTerm) and kind_of(t.name) == "state"

    def is_data(t: Term) -> bool:
        if isinstance(t, MeanTerm):
            return kind_of(t.name) == "observable"
        if isinstance(t, SeriesTerm):
            k = kind_of(t.name)
            return (k == "observable" and t.lag >= 1) or k == "exogenous"
        return False

    if is_state(a) and is_data(b):
        return ProductTerm(a, b)
    if is_state(b) and is_data(a):
        return ProductTerm(b, a)
    return None


def _term_text(t: Term) -> str:
    if isinstance(t, ProductTerm):
        return f"{_term_text(t.state)}*{_term_text(t.data)}"
    if isinstance(t, MeanTerm):
        return "mean(" + ", ".join(f"{t.name}[-{k}]" for k in t.lags) + ")"
    return t.name if t.lag == 0 else f"{t.name}[-{t.lag}]"


def linearize(expr: Expr, kind_of: Callable[[str], str | None], *, equation: str = "") -> LinearForm:
    """Distribute ``expr`` into a :class:`LinearForm`. ``kind_of(name)``
    returns ``"symbol"`` for a series/shock (or, finer, ``"state"`` /
    ``"observable"`` / ``"exogenous"`` / ``"shock"`` -- S9), ``"param"``
    for a parameter, or ``None`` for an undeclared name (a hard error
    naming it).

    Rejected, each with a message naming the limitation: a product of two
    symbol-carrying factors (nonlinear) OTHER than the E4 shape (S9: a
    state times a lagged observable / mean() / exogenous series, which
    becomes a :class:`ProductTerm`; only when ``kind_of`` distinguishes
    the kinds), a symbol in a denominator, a lagged or mean()'d name that
    is a parameter.
    """
    where = f" in equation {equation!r}" if equation else ""

    def rec(node: Expr) -> LinearForm:
        if isinstance(node, Num):
            return LinearForm({}, node)
        if isinstance(node, Name):
            kind = kind_of(node.name)
            if kind in _SYMBOL_KINDS:
                return LinearForm({SeriesTerm(node.name, 0): Num(1.0)})
            if kind == "param":
                return LinearForm({}, node)
            raise LinearityError(f"Unknown name {node.name!r}{where}: it is not a declared observable, exogenous series, state, shock or parameter.")
        if isinstance(node, LagRef):
            kind = kind_of(node.name)
            if kind in _SYMBOL_KINDS:
                return LinearForm({SeriesTerm(node.name, node.lag): Num(1.0)})
            if kind == "param":
                raise LinearityError(f"{node.emit()}{where}: {node.name!r} is a parameter; parameters cannot be lagged.")
            raise LinearityError(f"Unknown name {node.name!r}{where} (referenced as {node.emit()}).")
        if isinstance(node, MeanRef):
            kind = kind_of(node.name)
            if kind in _SYMBOL_KINDS:
                return LinearForm({MeanTerm(node.name, node.lags): Num(1.0)})
            if kind == "param":
                raise LinearityError(f"{node.emit()}{where}: {node.name!r} is a parameter; mean() takes lags of a series.")
            raise LinearityError(f"Unknown name {node.name!r}{where} (referenced as {node.emit()}).")
        if isinstance(node, Neg):
            inner = rec(node.operand)
            return LinearForm({t: _simplify_neg(c) for t, c in inner.terms.items()}, None if inner.const is None else _simplify_neg(inner.const))
        if isinstance(node, (Add, Sub)):
            left = rec(node.left)
            right = rec(node.right)
            if isinstance(node, Sub):
                right = LinearForm({t: _simplify_neg(c) for t, c in right.terms.items()}, None if right.const is None else _simplify_neg(right.const))
            terms = dict(left.terms)
            for t, c in right.terms.items():
                terms[t] = _add_coef(terms[t], c) if t in terms else c
            return LinearForm(terms, _add_opt(left.const, right.const))
        if isinstance(node, Mul):
            left = rec(node.left)
            right = rec(node.right)
            if left.has_terms and right.has_terms:
                # S9 E4: (terms_L + c_L) * (terms_R + c_R) distributes; every
                # term x term pair must be the one allowed bilinear shape.
                terms: dict[Term, Expr] = {}
                for ta, ca in left.terms.items():
                    for tb, cb in right.terms.items():
                        prod = _product_term(ta, tb, kind_of)
                        if prod is None:
                            raise LinearityError(
                                f"{node.emit()}{where} multiplies two series/shock terms -- the model must be "
                                f"LINEAR in states, observables, exogenous series and shocks (nonlinear "
                                f"state-space models are outside this stage's scope). The one product allowed "
                                f"(S9 E4, a time-varying coefficient) is a STATE times a lagged observable, a "
                                f"mean() of one observable, or an exogenous series -- {_term_text(ta)}*{_term_text(tb)} "
                                f"is not of that shape."
                            )
                        c = _simplify_mul(ca, cb)
                        terms[prod] = _add_coef(terms[prod], c) if prod in terms else c
                if right.const is not None:
                    for t, c in left.terms.items():
                        v = _simplify_mul(c, right.const)
                        terms[t] = _add_coef(terms[t], v) if t in terms else v
                if left.const is not None:
                    for t, c in right.terms.items():
                        v = _simplify_mul(left.const, c)
                        terms[t] = _add_coef(terms[t], v) if t in terms else v
                const = None if left.const is None or right.const is None else _simplify_mul(left.const, right.const)
                return LinearForm(terms, const)
            if right.has_terms:
                left, right = right, left
            if left.has_terms:
                if right.const is None:
                    return LinearForm({})
                k = right.const
                return LinearForm({t: _simplify_mul(c, k) for t, c in left.terms.items()})
            if left.const is None or right.const is None:
                return LinearForm({})
            return LinearForm({}, _simplify_mul(left.const, right.const))
        if isinstance(node, Div):
            left = rec(node.left)
            right = rec(node.right)
            if right.has_terms:
                raise LinearityError(
                    f"{node.emit()}{where} divides by a series/shock term -- not linear (outside scope)."
                )
            if right.const is None:
                raise LinearityError(f"{node.emit()}{where}: division by a zero constant.")
            k = right.const
            if left.has_terms:
                return LinearForm({t: Div(c, k) for t, c in left.terms.items()})
            if left.const is None:
                return LinearForm({})
            return LinearForm({}, Div(left.const, k))
        raise LinearityError(f"Unsupported expression node {type(node).__name__}{where}.")  # pragma: no cover

    return rec(expr)


# ---------------------------------------------------------------------------
# Coefficient evaluation (Python side of the Stan-emitted expression)
# ---------------------------------------------------------------------------


def evaluate(expr: Expr, params: dict) -> float:
    """Evaluate a coefficient expression at a parameter point, performing
    exactly the operations :meth:`Expr.emit` writes into the Stan program,
    in the same order."""
    if isinstance(expr, Num):
        return float(expr.value)
    if isinstance(expr, Name):
        try:
            return float(params[expr.name])
        except KeyError:
            raise KeyError(f"Coefficient expression references parameter {expr.name!r}, which the parameter point lacks; have {sorted(params)}.") from None
    if isinstance(expr, Neg):
        return -evaluate(expr.operand, params)
    if isinstance(expr, Add):
        return evaluate(expr.left, params) + evaluate(expr.right, params)
    if isinstance(expr, Sub):
        return evaluate(expr.left, params) - evaluate(expr.right, params)
    if isinstance(expr, Mul):
        return evaluate(expr.left, params) * evaluate(expr.right, params)
    if isinstance(expr, Div):
        return evaluate(expr.left, params) / evaluate(expr.right, params)
    raise TypeError(f"Cannot evaluate {type(expr).__name__} as a coefficient (series references are not coefficients).")


def term_names(term: Term) -> tuple[str, ...]:
    """The series names a term references (both factors of a product)."""
    if isinstance(term, ProductTerm):
        return (term.state.name, term.data.name)
    return (term.name,)


def name_occurrences(expr: Expr) -> dict[str, int]:
    """How many times each bare or lagged NAME occurs in a parsed tree
    (before linearization merges them) -- used to reject a shock written
    twice in one equation."""
    counts: dict[str, int] = {}

    def rec(node: Expr) -> None:
        if isinstance(node, (Name, LagRef, MeanRef)):
            counts[node.name] = counts.get(node.name, 0) + 1
        elif isinstance(node, Neg):
            rec(node.operand)
        elif isinstance(node, (Add, Sub, Mul, Div)):
            rec(node.left)
            rec(node.right)

    rec(expr)
    return counts


def parameters_in(expr: Expr) -> set[str]:
    """The parameter names a coefficient expression references."""
    if isinstance(expr, Name):
        return {expr.name}
    if isinstance(expr, Neg):
        return parameters_in(expr.operand)
    if isinstance(expr, (Add, Sub, Mul, Div)):
        return parameters_in(expr.left) | parameters_in(expr.right)
    return set()


def is_numeric(expr: Expr) -> bool:
    """True iff the coefficient expression contains no parameter."""
    return not parameters_in(expr)


def numeric_value(expr: Expr) -> float:
    """The value of a parameter-free coefficient expression."""
    return evaluate(expr, {})
