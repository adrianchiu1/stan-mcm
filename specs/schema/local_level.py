"""Options fragment for the ``local_level`` toy model family.

Model (fixed for S1 -- not yet configurable via spec options):

    y_t   = mu_t + eps_t,      eps_t ~ N(0, sigma_obs^2)
    mu_t  = mu_{t-1} + eta_t,  eta_t ~ N(0, sigma_level^2)
    mu_1  ~ N(y_1, 10)
    sigma_obs, sigma_level ~ Half-N(0, 1)

``local_level`` has no observed-series conventions in common with the LW-SV
model (no ``g``/``h``/inflation units) -- see ``lw-sv-spec.md``'s numerical
conventions for those; this toy family predates and is outside that scope.

No family-specific knobs are exposed yet. This fragment exists so the
family-registry / per-family-options plumbing (``specs/schema/__init__.py``)
has a second, concrete shape to dispatch on, ready for richer families
(e.g. ``lw_sv`` in S2) to be registered alongside it without touching
dispatch code.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class LocalLevelOptions(BaseModel):
    """Options for ``model.family: local_level``. Currently no fields beyond
    the family discriminator; any key here is rejected with a clear error
    naming the offending key, since ``extra="forbid"``."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["local_level"] = "local_level"
