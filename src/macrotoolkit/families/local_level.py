"""local_level family numerics-side callables (S5-decisions item 4).

The S1 toy family: one observable, no exogenous regressors, no options.
Registered on ``specs.schema.FAMILY_REGISTRY`` via dotted paths; see
``macrotoolkit.families`` for the spec-side/numerics-side split rationale.
"""
from __future__ import annotations

import pandas as pd

from specs.schema.base import RunSpec


def build_stan_data(df: pd.DataFrame) -> dict:
    """Stan ``data`` block for the local-level template: just T and y."""
    return {"T": int(len(df)), "y": df["y"].tolist()}


def build_render_context(spec: RunSpec) -> dict:
    """The local-level template takes an empty context (no options, priors
    fixed in the template) -- still routed through Jinja so the pipeline is
    exercised uniformly across families."""
    return {}
