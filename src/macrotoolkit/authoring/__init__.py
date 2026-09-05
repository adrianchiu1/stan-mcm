"""Equation-level model authoring (S7): compile a model written as
equations into the declaration surface a hand-written family provides --
``StateSpaceMeta``, matrix builders, prior table + sampler, mirror
declaration, a rendered Stan program, output modules and a validation
suite -- so an authored model flows through the ENTIRE existing machinery
(``mtk.fit`` with the fit-time mirror check, the immutable run store, the
DK smoother, the generic simulate/IRF/HD engine, the report, ``mtk sweep``
and ``mtk validate``) unchanged.

Layers:

- ``specs/schema/equations.py`` (grammar, IR, canonical form) and
  ``specs/schema/authored_structure.py`` (the structural derivation) are
  pure Python and spec-side; ``specs/schema/authored.py`` is the Pydantic
  options model that validates through them at spec-parse time.
- :mod:`macrotoolkit.authoring.compile` -- the numpy side: the
  :class:`CompiledModel` bundle (meta, numeric matrix builders, data
  builders, prior resolution + sampler, stationarity check), cached per
  canonical model definition.
- :mod:`macrotoolkit.authoring.stan` -- the render context for the ONE
  generic template ``stan/templates/authored.stan.j2``.
- :mod:`macrotoolkit.authoring.family` -- the registry capabilities
  (``build_stan_data``, ``build_render_context``, ``mirror``, ...).
- :mod:`macrotoolkit.authoring.results` / ``plots`` / ``outputs`` -- the
  generic output layer over ``results_core`` and the engine.
- :mod:`macrotoolkit.authoring.validation` -- the auto-instantiated fast
  tier and the one-call recovery/SBC constructors.
- :mod:`macrotoolkit.authoring.dsl` -- the small notebook helpers
  (``Model``, ``normal``, ``half_normal``, ``beta``, ``shock``, ``sv``,
  ``first_obs``, ``init``).
- :mod:`macrotoolkit.authoring.var` (S8) -- ``var`` (a recursive VAR(p)
  as measurement equations), ``minnesota_priors`` (the independent-normal
  Minnesota prior), the steady-state (Villani) form.
"""
from __future__ import annotations

from macrotoolkit.authoring.compile import CompiledModel, compile_model, compiled_for_spec  # noqa: F401
from macrotoolkit.authoring.dsl import (  # noqa: F401
    Model,
    beta,
    first_obs,
    forecast,
    half_normal,
    init,
    log_var_diff,
    normal,
    shock,
    sv,
)
from macrotoolkit.authoring.var import minnesota_priors, var, var_parts  # noqa: F401

__all__ = [
    "var",
    "var_parts",
    "minnesota_priors",
    "CompiledModel",
    "compile_model",
    "compiled_for_spec",
    "Model",
    "normal",
    "half_normal",
    "beta",
    "shock",
    "sv",
    "first_obs",
    "init",
    "forecast",
    "log_var_diff",
]
