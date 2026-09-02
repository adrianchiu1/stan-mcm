"""Family registry: the ONE place a model family's capabilities are
declared (S5-decisions item 4 completed the contract -- no if/elif family
dispatch anywhere else).

To add a new model family: write a ``specs/schema/<family>.py`` options
fragment, write ``stan/templates/<family>.stan.j2``, write a
``src/macrotoolkit/families/<family>.py`` numerics module (state metadata,
``build_stan_data``, ``build_render_context``, and -- once the family has
output modules -- a results loader/report writer), and add ONE entry to
``FAMILY_REGISTRY`` below. Nothing else -- ``RunSpec`` in ``base.py``,
``render.py``, ``run.py``, the CLI -- needs to change; they all consult
this registry by ``model.family`` string.

Numerics-side capabilities are declared as DOTTED PATHS
(``"module:attribute"``) resolved lazily on first use
(:meth:`FamilyEntry.resolve`): this package must stay importable without
numpy/pandas (``macrotoolkit`` imports it at module scope), so the registry
cannot import the numerics modules eagerly without creating an import
cycle.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Type

from pydantic import BaseModel

from specs.schema.local_level import LocalLevelOptions
from specs.schema.lw_sv import LwSvOptions, LwSvOutputs


@dataclass(frozen=True)
class FamilyEntry:
    """One registered model family -- the complete capability contract.

    Spec-side (eagerly imported, Pydantic-only):

    - ``options_model``: Pydantic model validating ``model.options``.
    - ``template``: filename under ``stan/templates/`` rendered for this
      family.
    - ``required_mapping``: keys that must be present in ``data.mapping``
      for this family (e.g. ``("y",)`` for local_level; lw_sv needs
      ``("y", "pi", "r")``).
    - ``outputs_model``: Pydantic model validating ``outputs`` (report/
      output-module config, OUTSIDE the run-identity hash -- S5-decisions
      item 3), or ``None`` if the family has no typed output config yet
      (``outputs`` then stays a free-form dict).

    Numerics-side (lazy ``"module:attr"`` dotted paths; ``None`` = the
    family does not provide that capability):

    - ``build_stan_data``: ``(df) -> dict`` -- the Stan ``data`` block.
    - ``build_render_context``: ``(spec) -> dict`` -- the Jinja context.
    - ``state_meta``: the family's ``StateSpaceMeta`` declaration (named
      state slots, shock loadings, feedback map -- S5-decisions items 1-2).
    - ``results_loader``: ``(run_dir) -> results object`` for the output
      modules.
    - ``report_writer``: ``(run_dir) -> Path`` -- writes the family's
      self-contained HTML report into the run dir.
    - ``prior_sd_table``: ``(spec, df) -> {param: prior sd}`` -- the prior
      side of the sweep report's prior→posterior contraction readout
      (S5-decisions item 9).
    - ``headline_series``: ``(run_dir) -> {name: (dates, median_path)}``
      -- the family's headline smoothed series for cross-run comparison
      overlays (the sweep report).
    """

    options_model: Type[BaseModel]
    template: str
    required_mapping: tuple[str, ...]
    outputs_model: Type[BaseModel] | None = None
    build_stan_data: str | None = None
    build_render_context: str | None = None
    state_meta: str | None = None
    results_loader: str | None = None
    report_writer: str | None = None
    prior_sd_table: str | None = None
    headline_series: str | None = None

    def resolve(self, capability: str) -> Any:
        """Resolve one of the dotted-path capability fields to the actual
        object, or ``None`` if the family does not declare it. Raises a
        descriptive error for an unknown capability name or a path that
        fails to import/resolve (a registry typo must fail loudly)."""
        if capability not in (
            "build_stan_data",
            "build_render_context",
            "state_meta",
            "results_loader",
            "report_writer",
            "prior_sd_table",
            "headline_series",
        ):
            raise ValueError(
                f"Unknown family capability {capability!r} -- see "
                f"FamilyEntry's docstring for the declared capability set."
            )
        path = getattr(self, capability)
        if path is None:
            return None
        module_name, sep, attr = path.partition(":")
        if not sep or not attr:
            raise ValueError(
                f"FamilyEntry.{capability} = {path!r} is not a valid "
                f"'module:attribute' dotted path."
            )
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"FamilyEntry.{capability} = {path!r}: module "
                f"{module_name!r} failed to import: {exc}"
            ) from exc
        try:
            return getattr(module, attr)
        except AttributeError:
            raise AttributeError(
                f"FamilyEntry.{capability} = {path!r}: module "
                f"{module_name!r} has no attribute {attr!r}."
            ) from None


FAMILY_REGISTRY: dict[str, FamilyEntry] = {
    "local_level": FamilyEntry(
        options_model=LocalLevelOptions,
        template="local_level.stan.j2",
        required_mapping=("y",),
        build_stan_data="macrotoolkit.families.local_level:build_stan_data",
        build_render_context="macrotoolkit.families.local_level:build_render_context",
        # No state metadata, results module, or report assembler: the S1
        # toy family has no output modules (spec §7's S1 scope).
    ),
    "lw_sv": FamilyEntry(
        options_model=LwSvOptions,
        template="lw_sv.stan.j2",
        required_mapping=("y", "pi", "r"),
        outputs_model=LwSvOutputs,
        build_stan_data="macrotoolkit.families.lw_sv:build_stan_data",
        build_render_context="macrotoolkit.families.lw_sv:build_render_context",
        state_meta="macrotoolkit.families.lw_sv:LW_STATE_META",
        results_loader="macrotoolkit.results_lw:load_lw_run",
        report_writer="macrotoolkit.report:write_report",
        prior_sd_table="macrotoolkit.families.lw_sv:prior_scalar_sds",
        headline_series="macrotoolkit.families.lw_sv:headline_series",
    ),
}


def get_family(name: str) -> FamilyEntry:
    """Look up a registered family by name, or raise a descriptive error."""
    if name not in FAMILY_REGISTRY:
        known = ", ".join(sorted(FAMILY_REGISTRY)) or "(none registered)"
        raise ValueError(
            f"Unknown model.family {name!r}. Valid values are: {known}. "
            f"Example: model: {{family: local_level, options: {{}}}}"
        )
    return FAMILY_REGISTRY[name]
