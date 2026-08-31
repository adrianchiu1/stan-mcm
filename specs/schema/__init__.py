"""Family registry: maps ``model.family`` -> (Pydantic options model, Jinja
template path, required data-mapping keys).

To add a new model family (e.g. ``lw_sv`` in S2): write a
``specs/schema/<family>.py`` options fragment, write
``stan/templates/<family>.stan.j2``, and add one entry to
``FAMILY_REGISTRY`` below. No other dispatch code -- ``RunSpec`` in
``base.py``, ``render.py``, ``run.py`` -- needs to change; they all consult
this registry by ``model.family`` string.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from pydantic import BaseModel

from specs.schema.local_level import LocalLevelOptions
from specs.schema.lw_sv import LwSvOptions, LwSvOutputs


@dataclass(frozen=True)
class FamilyEntry:
    """One registered model family.

    - ``options_model``: Pydantic model validating ``model.options``.
    - ``template``: filename under ``stan/templates/`` rendered for this
      family.
    - ``required_mapping``: keys that must be present in ``data.mapping``
      for this family (e.g. ``("y",)`` for local_level; LW-SV needs
      ``("y", "pi", "r")``).
    - ``outputs_model``: Pydantic model validating ``outputs`` (S4,
      lw-sv-spec.md §2.3/§3), or ``None`` if the family has no typed
      output-module config yet (``outputs`` then stays the free-form dict
      ``RunSpec`` declares -- ``local_level`` has no output modules to
      configure).
    """

    options_model: Type[BaseModel]
    template: str
    required_mapping: tuple[str, ...]
    outputs_model: Type[BaseModel] | None = None


FAMILY_REGISTRY: dict[str, FamilyEntry] = {
    "local_level": FamilyEntry(
        options_model=LocalLevelOptions,
        template="local_level.stan.j2",
        required_mapping=("y",),
    ),
    "lw_sv": FamilyEntry(
        options_model=LwSvOptions,
        template="lw_sv.stan.j2",
        required_mapping=("y", "pi", "r"),
        outputs_model=LwSvOutputs,
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
