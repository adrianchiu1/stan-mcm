"""The render context of the ONE generic authored-model template
(``stan/templates/authored.stan.j2``, S7 M2): everything family-specific
is stamped from the compiled structure at render time -- no runtime
branching in Stan (ENGINEERING.md "The spec is the spine").

What the template receives, and how it mirrors
:class:`macrotoolkit.authoring.compile.CompiledModel`:

- ``F``/``A``/``Z``: sparse entry lists ``(i, j, expr_text)`` (1-based)
  whose ``expr_text`` is the SAME coefficient tree the Python builder
  evaluates, printed by :meth:`Expr.emit` -- identical floating-point
  operations in identical order. A matrix with no parameter in any entry
  is built once in ``transformed data``.
- ``Q``: per entry the ordered ``coef * var_s`` term list
  (``CompiledModel.q_terms``); a variance is ``square(sigma)`` for a
  constant-scale shock or ``exp(h_<s>[t])`` for an SV shock (h is
  log-VARIANCE). Constant when no state shock has SV, else an
  ``array[T] matrix`` path ``Qt`` (the S6 Q_t machinery); ``R`` likewise
  (``Rt`` for measurement SV).
- ``params``: one declaration + prior statement per prior-table entry
  (normal with truncation carried by the parameter constraint;
  half_normal via ``<lower=0>``; beta), then the SV block per SV shock
  (``sigma_h_<s>``, non-centered ``h0_<s>_raw``, ``nu_<s>``, and
  ``h_<s> = sv_rw_noncentered(mu_h0_<s> + h0_sd * h0_<s>_raw, sigma_h_<s>,
  nu_<s>)`` -- exactly the hand templates' line).
- ``kf_loglik`` as a transformed parameter (S6 WP3's mechanism), the
  quantity the fit-time mirror check and the M2 equivalence gate read.
"""
from __future__ import annotations

from typing import Any

from macrotoolkit.authoring.compile import CompiledModel
from specs.schema.equations import _fmt_num

TEMPLATE_NAME = "authored.stan.j2"


def _entries(sparse: dict[tuple[int, int], Any]) -> list[tuple[int, int, str]]:
    return [(i + 1, j + 1, expr.emit()) for (i, j), expr in sorted(sparse.items())]


def _var_text(compiled: CompiledModel, shock: str) -> str:
    return compiled.shock_variance_expr(shock)


def _q_entry_text(compiled: CompiledModel, terms) -> str:
    parts = []
    for term in terms:
        var = _var_text(compiled, term.shock)
        parts.append(var if term.coef == 1.0 else f"{_fmt_num(term.coef)} * {var}")
    return " + ".join(parts)


def _param_decl(name: str, entry: dict, bounds: tuple[float | None, float | None]) -> str:
    lower, upper = bounds
    constraint = []
    if lower is not None:
        constraint.append(f"lower={_fmt_num(lower)}")
    if upper is not None:
        constraint.append(f"upper={_fmt_num(upper)}")
    c = f"<{', '.join(constraint)}>" if constraint else ""
    return f"real{c} {name};"


def _prior_stmt(name: str, entry: dict) -> str:
    dist = entry["dist"]
    if dist == "normal":
        return f"{name} ~ normal({_fmt_num(entry['mu'])}, {_fmt_num(entry['sd'])});"
    if dist == "half_normal":
        return f"{name} ~ normal(0, {_fmt_num(entry['sd'])});  // Half-N via <lower=0>"
    if dist == "beta":
        return f"{name} ~ beta({_fmt_num(entry['a'])}, {_fmt_num(entry['b'])});"
    raise ValueError(f"Unknown prior dist {dist!r} for {name!r}.")  # pragma: no cover


def render_context(compiled: CompiledModel, priors: dict[str, dict]) -> dict[str, Any]:
    """The Jinja context for ``authored.stan.j2`` at the RESOLVED priors
    (defaults + the spec's overrides)."""
    st = compiled.structure
    meta = compiled.meta
    n, m, k = meta.n_state, meta.n_obs, st.n_exog_cols
    numeric = compiled.numeric_matrices()
    sv_state = any(s in compiled.sv_shocks for s in meta.state_shocks)
    sv_meas = any(s in compiled.sv_shocks for s in meta.measurement_shocks)

    params = []
    for name in st.params:
        params.append({"name": name, "decl": _param_decl(name, priors[name], compiled.bounds[name]), "prior": _prior_stmt(name, priors[name])})
    sv_blocks = []
    for s in compiled.sv_shocks:
        sv_blocks.append(
            {
                "shock": s,
                "sigma_h_prior": _prior_stmt(f"sigma_h_{s}", priors[f"sigma_h_{s}"]),
                "h0_sd": _fmt_num(priors[f"mu_h0_{s}"]["sd"]),
                "kind": "state" if s in meta.state_shocks else "measurement",
            }
        )
    q_entries = [(i + 1, j + 1, _q_entry_text(compiled, terms)) for (i, j), terms in sorted(compiled.q_terms.items())]
    r_entries = [(i + 1, _var_text(compiled, s)) for i, s in enumerate(meta.measurement_shocks)]

    return {
        "name": compiled.name,
        "n": n,
        "m": m,
        "k": k,
        "state_labels": [f"{nm}[{o}]" if o else nm for nm, o in meta.state_labels],
        "obs_names": list(meta.obs_names),
        "exog_cols": [_fb_text(t) for t in st.feedback_map],
        "equations_measurement": list(st.canonical_measurement),
        "equations_transition": list(st.canonical_transition),
        "F": {"numeric": numeric["F"], "entries": _entries(st.F)},
        "A": {"numeric": numeric["A"], "entries": _entries(st.A)},
        "Z": {"numeric": numeric["Z"], "entries": _entries(st.Z)},
        "Q_entries": q_entries,
        "R_entries": r_entries,
        "sv": bool(compiled.sv_shocks),
        "sv_shocks": list(compiled.sv_shocks),
        "sv_state": sv_state,
        "sv_meas": sv_meas,
        "params": params,
        "sv_blocks": sv_blocks,
    }


def _fb_text(term) -> str:
    if term[0] == "obs_lag_mean":
        return "mean(" + ", ".join(f"{term[1]}[-{k}]" for k in term[2]) + ")"
    return f"{term[1]}[-{term[2]}]"
