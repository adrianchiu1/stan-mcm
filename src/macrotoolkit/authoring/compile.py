"""The numpy side of authored-model compilation (S7 M1): from the
validated options (whose ``structure`` already holds the layout and the
SYMBOLIC matrices, ``specs/schema/authored_structure.py``) to the numeric
declaration surface a hand-written family provides:

- :attr:`CompiledModel.meta` -- the family's :class:`StateSpaceMeta`
  (named slots with time offsets, shock loadings, feedback map);
- :meth:`CompiledModel.build_matrices` -- ``(F, Q, A, Z, R)`` at a
  parameter point, ``Q``/``R`` constant or ``(T, ., .)`` paths for SV
  shocks (a measurement shock's SV through ``R_t``, a state shock's
  through the S6 ``Q_t`` machinery), performing EXACTLY the operations the
  generic Stan template performs in the same order (see
  :mod:`macrotoolkit.authoring.stan`), so the two sides agree to G1's
  tolerance rather than approximately;
- :meth:`CompiledModel.regressors` / :meth:`initial_state` /
  :meth:`anchors` / :meth:`stan_data` -- the data side (pre-sample lag rows
  = the feedback map's depth; ``first_obs`` anchors on the first
  estimation row; explicit ``(xi00, P00)``);
- :meth:`CompiledModel.resolve_priors` (defaults + variant-checked
  overrides, the hand families' exact discipline) and
  :meth:`sample_prior_params` (the template's own distribution
  semantics: truncated normals by rejection, half_normal = |N|, beta);
- :meth:`CompiledModel.is_stationary` -- the identity gate's stationary-
  prior-point filter (the S6 lesson) as a structural check: spectral
  radius of ``F`` <= 1 (random walks allowed) and of the endogenous
  feedback's companion matrix < 1.

Nothing here re-derives structure: the symbolic entries are evaluated with
:func:`specs.schema.equations.evaluate`, the same tree the Stan emitter
prints.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from macrotoolkit.families.base import Const, ExogLag, ObsLag, ObsLagMean, StateSpaceMeta
from specs.schema.authored import AuthoredOptions, FirstObs, LogVarDiffAnchor
from specs.schema.authored_structure import CONST_STATE, ModelStructure
from specs.schema.equations import Expr, Num, evaluate


def feedback_terms(structure: ModelStructure):
    """The structure's plain feedback tuples as the engine's typed terms."""
    out = []
    for term in structure.feedback_map:
        kind = term[0]
        if kind == "const":
            out.append(Const())
        elif kind == "obs_lag":
            out.append(ObsLag(term[1], term[2]))
        elif kind == "obs_lag_mean":
            out.append(ObsLagMean(term[1], tuple(term[2])))
        else:
            out.append(ExogLag(term[1], term[2]))
    return tuple(out)


def build_meta(structure: ModelStructure) -> StateSpaceMeta:
    """The compiled :class:`StateSpaceMeta`; its constructor validates the
    declaration and :meth:`recovery_order` is evaluated once so a
    non-triangular loading pattern fails at compile time, not inside the
    smoother."""
    meta = StateSpaceMeta(
        state_labels=tuple(structure.state_labels),
        state_shocks=tuple(structure.state_shocks),
        shock_loadings={s: dict(loads) for s, loads in structure.shock_loadings.items()},
        measurement_shocks=tuple(structure.measurement_shocks),
        obs_names=tuple(structure.obs_names),
        exog_names=tuple(structure.exog_names),
        feedback_map=feedback_terms(structure),
        # S8 E3/E5: declared only when the measurement side is NOT the
        # pre-S8 "one own shock per row" identity, so an S7 model's meta
        # (and every consumer's code path) is unchanged.
        measurement_loadings=None if structure.meas_loading_is_identity() else {s: tuple(rows) for s, rows in structure.meas_loadings.items()},
    )
    meta.recovery_order()
    meta.meas_recovery_order()
    return meta


@dataclass(frozen=True)
class _QTerm:
    """One ``coef * var_shock`` term of a Q entry (``coef = B[i,s]*B[j,s]``)."""

    coef: float
    shock: str


def _q_terms(meta: StateSpaceMeta) -> dict[tuple[int, int], tuple[_QTerm, ...]]:
    """``Q = sum_s var_s b_s b_s'`` entry by entry, shocks in declared
    order, zero products dropped; the SAME term lists the Stan emitter
    prints, so both sides accumulate identical operations in identical
    order (lower triangle mirrored)."""
    n = meta.n_state
    B = meta.loading_matrix()
    out: dict[tuple[int, int], tuple[_QTerm, ...]] = {}
    for i in range(n):
        for j in range(i, n):
            terms = []
            for k, s in enumerate(meta.state_shocks):
                c = B[i, k] * B[j, k]
                if c != 0.0:
                    terms.append(_QTerm(float(c), s))
            if terms:
                out[(i, j)] = tuple(terms)
    return out


@dataclass(frozen=True)
class _RTerm:
    """One ``coef * var_shock`` term of an R entry (S8 E5): ``coef`` is the
    coefficient TREE ``M[i,s]*M[k,s]`` (unit factors dropped), evaluated
    in Python and printed into Stan from the same tree."""

    coef: Expr
    shock: str


def _r_terms(structure: ModelStructure) -> dict[tuple[int, int], tuple[_RTerm, ...]]:
    """``R = sum_s var_s m_s m_s'`` entry by entry (upper triangle), shocks
    in declared order -- the measurement-side twin of :func:`_q_terms`.
    For the pre-S8 identity loading every entry is the diagonal
    ``var_s`` alone (coefficient ``1.0``, printed as the bare variance,
    so an S7 program's text is unchanged)."""
    from specs.schema.equations import _simplify_mul

    m = structure.n_obs
    out: dict[tuple[int, int], tuple[_RTerm, ...]] = {}
    for i in range(m):
        for k in range(i, m):
            terms = []
            for j, s in enumerate(structure.measurement_shocks):
                a, b = structure.M.get((i, j)), structure.M.get((k, j))
                if a is None or b is None:
                    continue
                terms.append(_RTerm(_simplify_mul(a, b), s))
            if terms:
                out[(i, k)] = tuple(terms)
    return out


@dataclass
class CompiledModel:
    """The compiled bundle for one authored model (see module doc)."""

    options: AuthoredOptions
    structure: ModelStructure
    meta: StateSpaceMeta
    prior_table: dict[str, dict[str, Any]]
    bounds: dict[str, tuple[float | None, float | None]]
    q_terms: dict[tuple[int, int], tuple[_QTerm, ...]]
    canonical_id: str
    r_terms: dict[tuple[int, int], tuple[_RTerm, ...]] = None  # type: ignore[assignment]

    # --- convenience --------------------------------------------------
    @property
    def name(self) -> str:
        return self.options.name

    @property
    def sv_shocks(self) -> tuple[str, ...]:
        return self.structure.sv_shocks

    @property
    def sv_on(self) -> bool:
        return bool(self.structure.sv_shocks)

    @property
    def lag_depth(self) -> int:
        return self.structure.lag_depth

    @property
    def param_names(self) -> tuple[str, ...]:
        """Every scalar parameter of the rendered program's ``parameters``
        block that the prior sampler draws, in prior-table order: the
        declared parameters plus ``sigma_h_<s>`` per SV shock (``h0_<s>``
        is drawn too but enters the program non-centered as
        ``h0_<s>_raw``)."""
        return tuple(self.structure.params) + tuple(f"sigma_h_{s}" for s in self.sv_shocks)

    def shock_variance_expr(self, shock: str) -> str:
        """How the shock's VARIANCE is written in Stan: ``square(sigma)``
        for a constant scale, ``exp(h_<shock>[t])`` for SV (h is
        log-VARIANCE)."""
        if shock in self.sv_shocks:
            return f"exp(h_{shock}[t])"
        return f"square({self.structure.shock_scale_param[shock]})"

    # --- numeric matrices ---------------------------------------------
    def _dense(self, entries: Mapping[tuple[int, int], Expr], shape: tuple[int, int], params: Mapping[str, float]) -> np.ndarray:
        M = np.zeros(shape)
        for (i, j), expr in entries.items():
            M[i, j] = evaluate(expr, params)
        return M

    def build_F(self, params: Mapping[str, float]) -> np.ndarray:
        n = self.meta.n_state
        return self._dense(self.structure.F, (n, n), params)

    def build_A(self, params: Mapping[str, float]) -> np.ndarray:
        return self._dense(self.structure.A, (self.structure.n_exog_cols, self.meta.n_obs), params)

    def build_Z(self, params: Mapping[str, float]) -> np.ndarray:
        return self._dense(self.structure.Z, (self.meta.n_obs, self.meta.n_state), params)

    def build_M(self, params: Mapping[str, float]) -> np.ndarray | None:
        """The measurement-shock loading matrix ``(m, n_meas_shocks)``
        (S8 E5) at a parameter point, or ``None`` when it is the pre-S8
        identity (every consumer then keeps its bit-identical path)."""
        if self.structure.meas_loading_is_identity():
            return None
        return self._dense(self.structure.M, (self.meta.n_obs, self.meta.n_meas_shocks), params)

    def _shock_variance(self, shock: str, params: Mapping[str, float], h: Mapping[str, np.ndarray] | None):
        if shock in self.sv_shocks:
            if h is None or shock not in h:
                raise ValueError(
                    f"build_matrices: shock {shock!r} has SV; pass its log-variance path in h={{'{shock}': (T,)}}."
                )
            return np.exp(np.asarray(h[shock], dtype=np.float64))  # h is log-VARIANCE
        sd = float(params[self.structure.shock_scale_param[shock]])
        return sd * sd  # Stan's square()

    def build_Q(self, params: Mapping[str, float], h: Mapping[str, np.ndarray] | None = None, T: int | None = None) -> np.ndarray:
        """``Q`` (n, n) when no state shock has SV, else the (T, n, n)
        path; each entry accumulates ``coef * var_s`` in shock order."""
        n = self.meta.n_state
        tv = any(s in self.sv_shocks for s in self.meta.state_shocks)
        if tv:
            if T is None:
                T = len(next(iter(np.asarray(h[s]) for s in self.meta.state_shocks if s in self.sv_shocks)))
            Q = np.zeros((T, n, n))
        else:
            Q = np.zeros((n, n))
        for (i, j), terms in self.q_terms.items():
            acc = None
            for term in terms:
                contrib = term.coef * self._shock_variance(term.shock, params, h)
                acc = contrib if acc is None else acc + contrib
            if tv:
                Q[:, i, j] = acc
                Q[:, j, i] = acc
            else:
                Q[i, j] = acc
                Q[j, i] = acc
        return Q

    def build_R(self, params: Mapping[str, float], h: Mapping[str, np.ndarray] | None = None, T: int | None = None) -> np.ndarray:
        """``R = sum_s var_s m_s m_s'`` (m, m), or the (T, m, m) path when a
        measurement shock has SV; each entry accumulates ``coef * var_s``
        in shock order exactly as the template prints it (a unit
        coefficient is not multiplied -- ``1.0 * v == v`` exactly, and the
        template prints the bare variance). Rows without a shock (S8 E3)
        stay exactly zero."""
        m = self.meta.n_obs
        tv = any(s in self.sv_shocks for s in self.meta.measurement_shocks)
        if tv:
            if T is None:
                T = len(next(iter(np.asarray(h[s]) for s in self.meta.measurement_shocks if s in self.sv_shocks)))
            R = np.zeros((T, m, m))
        else:
            R = np.zeros((m, m))
        for (i, k), terms in self.r_terms.items():
            acc = None
            for term in terms:
                var = self._shock_variance(term.shock, params, h)
                if isinstance(term.coef, Num) and term.coef.value == 1.0:
                    contrib = var
                else:
                    contrib = evaluate(term.coef, params) * var
                acc = contrib if acc is None else acc + contrib
            if tv:
                R[:, i, k] = acc
                R[:, k, i] = acc
            else:
                R[i, k] = acc
                R[k, i] = acc
        return R

    def build_matrices(self, params: Mapping[str, float], h: Mapping[str, np.ndarray] | None = None, T: int | None = None):
        """``(F, Q, A, Z, R)`` at ``params`` (scales in sd units); ``h``
        supplies the log-variance path of every SV shock."""
        return self.build_F(params), self.build_Q(params, h, T), self.build_A(params), self.build_Z(params), self.build_R(params, h, T)

    # --- data side ----------------------------------------------------
    def series_arrays(self, df) -> dict[str, np.ndarray]:
        out = {}
        for name in tuple(self.meta.obs_names) + tuple(self.meta.exog_names):
            if name not in df.columns:
                raise ValueError(f"The model-ready data has no column {name!r}; columns: {list(df.columns)}.")
            out[name] = np.asarray(df[name].to_numpy(), dtype=np.float64)
        return out

    def regressors(self, series: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """``(yobs, x)``: rows ``L..`` of the observables (``L`` = the
        feedback map's lag depth, the pre-sample rows) and the regressor
        matrix column by column per the feedback map -- lag ``k`` of a
        series is its rows ``L-k .. L-k+T`` (``k = 0``, S8 E2, the
        estimation rows themselves); the ``Const`` column is ones; a mean
        column sums its lags in the listed order and divides once."""
        L = self.lag_depth
        n_rows = len(next(iter(series.values())))
        T = n_rows - L
        if T < 1:
            raise ValueError(
                f"authored model {self.name!r} needs at least {L + 1} data rows ({L} pre-sample lag rows for its "
                f"feedback map + 1 estimation row); the trimmed data has {n_rows}. data.sample.start must include the lag rows."
            )
        yobs = np.column_stack([series[name][L : L + T] for name in self.meta.obs_names])
        cols = []
        for term in self.meta.feedback_map:
            if isinstance(term, Const):
                cols.append(np.ones(T))
            elif isinstance(term, ObsLagMean):
                total = None
                for k in term.lags:
                    sl = series[term.name][L - k : L - k + T]
                    total = sl.copy() if total is None else total + sl
                cols.append(total / float(len(term.lags)))
            else:
                cols.append(series[term.name][L - term.lag : L - term.lag + T].copy())
        x = np.column_stack(cols) if cols else np.zeros((T, 0))
        return yobs, x

    def initial_state(self, series: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Explicit ``(xi00, P00)``: every slot of a state gets its declared
        marginal (``first_obs`` = the observable's first ESTIMATION-row
        value), a priori independent -- lw_sv's documented simplification."""
        L = self.lag_depth
        xi00 = np.zeros(self.meta.n_state)
        P00 = np.zeros((self.meta.n_state, self.meta.n_state))
        for i, (name, _) in enumerate(self.meta.state_labels):
            if name == CONST_STATE:  # S8 E1: the deterministic unit state carrying drifts
                xi00[i] = 1.0
                continue
            init = self.options.initial_state[name]
            mean = float(series[init.mean.first_obs][L]) if isinstance(init.mean, FirstObs) else float(init.mean)
            xi00[i] = mean
            P00[i, i] = float(init.sd) ** 2
        return xi00, P00

    def anchors(self, series: Mapping[str, np.ndarray]) -> dict[str, float]:
        """``{mu_h0_<shock>: value}`` per SV shock: the declared constant, or
        ``ln(fraction * Var(Delta series))`` over the estimation rows."""
        L = self.lag_depth
        out: dict[str, float] = {}
        for shock in self.sv_shocks:
            anchor = self.options.shocks[shock].sv.mu_h0
            if isinstance(anchor, LogVarDiffAnchor):
                s = series[anchor.series][L:]
                if s.shape[0] < 3:
                    raise ValueError(f"mu_h0 anchor for shock {shock!r} needs at least 3 estimation rows of {anchor.series!r}; got {s.shape[0]}.")
                v = float(np.var(np.diff(s), ddof=1))
                if not (np.isfinite(v) and v > 0.0):
                    raise ValueError(f"mu_h0 anchor for shock {shock!r}: Var(Delta {anchor.series}) = {v!r} is not positive/finite -- degenerate data?")
                out[f"mu_h0_{shock}"] = float(np.log(anchor.fraction * v))
            else:
                out[f"mu_h0_{shock}"] = float(anchor)
        return out

    def stan_data(self, df) -> dict:
        """The generic template's ``data`` block from the model-ready
        DataFrame (``T``, ``yobs``, ``x`` when k > 0, ``xi00``, ``P00``,
        the SV anchors)."""
        series = self.series_arrays(df)
        yobs, x = self.regressors(series)
        xi00, P00 = self.initial_state(series)
        data = {"T": int(yobs.shape[0]), "yobs": yobs}
        if self.meta.n_state > 0:  # n == 0 (S8 E0): the template builds the zero-size objects in transformed data
            data["xi00"], data["P00"] = xi00, P00
        if x.shape[1] > 0:
            data["x"] = x
        data.update(self.anchors(series))
        return data

    # --- priors -------------------------------------------------------
    def resolve_priors(self, overrides: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """Defaults + per-run overrides with the hand families' exact
        discipline: unknown prior name or unknown field = hard error."""
        priors = {name: dict(entry) for name, entry in self.prior_table.items()}
        for name, override in overrides.items():
            if name not in priors:
                raise ValueError(
                    f"priors[{name!r}] is not a parameter of authored model {self.name!r}. Valid names: {sorted(priors)}."
                )
            if not isinstance(override, Mapping):
                raise ValueError(f"priors[{name!r}] must be a mapping of prior fields to override (e.g. {{sd: 0.5}}), got {override!r}.")
            merged = {**priors[name], **dict(override)}
            unknown = set(merged) - set(priors[name])
            if unknown:
                raise ValueError(
                    f"priors[{name!r}] has unknown field(s) {sorted(unknown)}; the default entry's fields are {sorted(priors[name])}."
                )
            if merged.get("dist") != priors[name]["dist"]:
                raise ValueError(f"priors[{name!r}]: the distribution family ({priors[name]['dist']!r}) is structure, not an overridable field.")
            priors[name] = merged
        return priors

    def sample_prior_params(self, priors: Mapping[str, Mapping[str, Any]], rng: np.random.Generator, anchors: Mapping[str, float] | None = None) -> dict[str, float]:
        """ONE parameter point from the resolved prior, with the template's
        own semantics (truncation by rejection on the parameter's support;
        half_normal = |N(0, sd^2)|; beta). SV shocks add ``sigma_h_<s>`` and
        the realized ``h0_<s>`` ~ N(mu_h0_<s>, h0_sd^2) at the supplied
        anchors."""
        out: dict[str, float] = {}
        for name in self.param_names:
            entry = priors[name]
            lower, upper = self.bounds.get(name, (None, None))
            dist = entry["dist"]
            if dist == "normal":
                for _ in range(10_000):
                    v = rng.normal(entry["mu"], entry["sd"])
                    if (lower is None or v > lower) and (upper is None or v < upper):
                        out[name] = float(v)
                        break
                else:
                    raise RuntimeError(
                        f"sample_prior_params: failed to draw {name!r} inside ({lower}, {upper}) within 10,000 attempts -- "
                        f"the resolved prior (mu={entry['mu']}, sd={entry['sd']}) puts essentially no mass on the support."
                    )
            elif dist == "half_normal":
                out[name] = float(abs(rng.normal(0.0, entry["sd"])))
            elif dist == "beta":
                out[name] = float(rng.beta(entry["a"], entry["b"]))
            else:  # pragma: no cover -- the schema's Prior union
                raise ValueError(f"Unknown prior dist {dist!r} for {name!r}.")
        for shock in self.sv_shocks:
            key = f"mu_h0_{shock}"
            if anchors is None or key not in anchors:
                raise ValueError(f"sample_prior_params: SV shock {shock!r} needs the data anchor {key!r} (CompiledModel.anchors).")
            out[f"h0_{shock}"] = float(rng.normal(anchors[key], priors[key]["sd"]))
        return out

    # --- stationarity (the identity gate's prior-point filter) ---------
    def is_stationary(self, params: Mapping[str, float], tol: float = 1e-9) -> bool:
        """Spectral radius of ``F`` <= 1 AND of the endogenous feedback's
        companion matrix <= 1: unit roots are the norm (random-walk trends;
        lw_sv's accelerationist Phillips curve, whose inflation-lag
        coefficients sum to one) and pass; EXPLOSIVE roots fail (an
        explosive draw measures float64 cancellation over the sample, not
        an identity -- the S6 lesson behind the stationary-prior-points
        gate)."""
        F = self.build_F(params)
        if F.size and np.max(np.abs(np.linalg.eigvals(F))) > 1.0 + tol:
            return False
        A = self.build_A(params)
        m = self.meta.n_obs
        depth = max((self.meta.obs_lag_depth(o) for o in self.meta.obs_names), default=0)
        if depth == 0:
            return True
        Phi = np.zeros((depth, m, m))  # Phi[k-1][i, j]: coefficient of obs j at lag k in row i
        for col, term in enumerate(self.meta.feedback_map):
            if isinstance(term, ObsLag):
                Phi[term.lag - 1, :, self.meta.obs_index(term.name)] += A[col, :]
            elif isinstance(term, ObsLagMean):
                for k in term.lags:
                    Phi[k - 1, :, self.meta.obs_index(term.name)] += A[col, :] / len(term.lags)
        comp = np.zeros((m * depth, m * depth))
        comp[:m, :] = np.concatenate([Phi[k] for k in range(depth)], axis=1)
        if depth > 1:
            comp[m:, : m * (depth - 1)] = np.eye(m * (depth - 1))
        return bool(np.max(np.abs(np.linalg.eigvals(comp))) <= 1.0 + tol)

    # --- symbolic inspection helpers (used by the Stan emitter) --------
    def numeric_matrices(self) -> dict[str, bool]:
        return {k: self.structure.matrix_is_numeric(k) for k in ("F", "A", "Z", "M")}


def canonical_model_id(options: AuthoredOptions) -> str:
    """SHA-256 of the canonical options serialization (sorted keys) -- the
    compile cache key; NOT the run identity (that is ``run.compute_run_id``
    over the whole estimation spec + data + rendered source)."""
    payload = json.dumps(options.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_CACHE: dict[str, CompiledModel] = {}


def compile_model(options: AuthoredOptions | Mapping[str, Any]) -> CompiledModel:
    """Compile (cached by canonical definition) an authored model."""
    if not isinstance(options, AuthoredOptions):
        options = AuthoredOptions.model_validate(dict(options))
    key = canonical_model_id(options)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    structure = options.structure
    meta = build_meta(structure)
    compiled = CompiledModel(
        options=options,
        structure=structure,
        meta=meta,
        prior_table=options.prior_table(),
        bounds=options.bounds(),
        q_terms=_q_terms(meta),
        canonical_id=key,
        r_terms=_r_terms(structure),
    )
    _CACHE[key] = compiled
    return compiled


def compiled_for_spec(spec) -> CompiledModel:
    """The compiled bundle for a ``RunSpec`` whose family is ``authored``."""
    if spec.model.family != "authored":
        raise ValueError(f"compiled_for_spec: model.family is {spec.model.family!r}, not 'authored'.")
    return compile_model(spec.model.options)
