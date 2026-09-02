"""Sweep spec (S5-decisions item 9): a base run spec + a grid of prior
overrides, run as a batch through the ordinary run store and summarized in
ONE comparison report.

Design (recorded in DECISIONS.md, 2026-09-02): a sweep adds NO new storage
concept -- every cell is a normal, hash-identified, immutable run
(``runs/<hash12>/``), so re-running a sweep is idempotent cell-by-cell for
free, and any cell's full report remains available via ``mtk report``.
The sweep layer only orchestrates the batch and assembles the comparison.

YAML shape::

    name: sigma_g_z_sweep
    base_spec: spec_sv.yaml        # resolved relative to the sweep file
    cells:
      - label: baseline
        priors: {}                 # merged over the base spec's priors
      - label: sigma_g_tight
        priors: {sigma_g: {sd: 0.015}}

Merge semantics: a cell's ``priors`` mapping is merged over the BASE
spec's own ``priors`` block key-by-key (a cell entry replaces the base
entry for that parameter wholesale); the merged block then goes through
the family's ordinary resolution against its defaults, including the
variant-checked override validation (ENGINEERING.md: overriding an
inactive prior is a hard error).
"""
from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SweepCell(BaseModel):
    """One grid cell: a label (used in the comparison report and the cell
    listing) plus the prior overrides distinguishing it from the base."""

    model_config = ConfigDict(extra="forbid")

    label: str
    priors: dict[str, Any] = Field(default_factory=dict)

    @field_validator("label")
    @classmethod
    def _label_nonempty_and_pathsafe(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("cells[].label must be a non-empty string.")
        if any(ch in v for ch in "/\\"):
            raise ValueError(
                f"cells[].label {v!r} must not contain path separators "
                f"(labels name report rows and artifact files)."
            )
        return v


class SweepSpec(BaseModel):
    """Top-level sweep file: name, base spec path, and the cell grid."""

    model_config = ConfigDict(extra="forbid")

    name: str
    base_spec: str
    cells: list[SweepCell]

    @field_validator("name")
    @classmethod
    def _name_nonempty_and_pathsafe(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must be a non-empty string.")
        if any(ch in v for ch in "/\\"):
            raise ValueError(f"name {v!r} must not contain path separators.")
        return v

    @field_validator("cells")
    @classmethod
    def _cells_nonempty_unique(cls, v: list[SweepCell]) -> list[SweepCell]:
        if not v:
            raise ValueError("cells must contain at least one cell.")
        labels = [c.label for c in v]
        dupes = sorted({l for l in labels if labels.count(l) > 1})
        if dupes:
            raise ValueError(f"cells[].label values must be unique; duplicated: {dupes}.")
        return v


def load_sweep_spec(path: str) -> SweepSpec:
    """Load and validate a sweep YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Sweep file {path!r} must contain a YAML mapping with keys "
            f"name, base_spec, cells; got {type(raw).__name__}."
        )
    return SweepSpec.model_validate(raw)
