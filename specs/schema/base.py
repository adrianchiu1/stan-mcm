"""``RunSpec`` -- the Pydantic v2 model mirrored exactly by the run-spec YAML
(``lw-sv-spec.md`` §2.3). This is the single source of truth: every stage of
the pipeline (data loading, rendering, compiling, sampling, run-identity
hashing) consumes a validated ``RunSpec``, never raw YAML.

Validation contract (hard requirement, not polish): every validation error
must name the offending field (Pydantic does this via the error's ``loc``)
and state what a valid value looks like. Where Pydantic's default message
is not self-explanatory (unknown family, missing required data-mapping
keys), we raise a custom message that spells out a concrete example.

Every model here uses ``extra="forbid"`` so a typo'd or unsupported key is
rejected loudly, naming the unexpected key, rather than silently ignored.
"""
from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from specs.schema import get_family


class SampleSpec(BaseModel):
    """Date-range trim applied to the loaded data.

    ``start``/``end`` are pandas-parseable date strings (e.g. ``"1961-01-01"``
    or ``"1961Q1"``) or ``null``/omitted, meaning "from the first row" /
    "through the last row" respectively. Actual parsing against the data's
    date column happens in ``macrotoolkit.data`` (it can then report the
    data's real available range in any error), not here.
    """

    model_config = ConfigDict(extra="forbid")

    start: str | None = None
    end: str | None = None

    @model_validator(mode="after")
    def _non_empty_strings(self) -> "SampleSpec":
        for field_name in ("start", "end"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(
                    f"data.sample.{field_name} must be null or a non-empty "
                    f"date string (e.g. \"1961Q1\" or \"1961-01-01\"), got "
                    f"an empty string."
                )
        return self


class DataSpec(BaseModel):
    """Where the data lives and how CSV columns map to model variables.

    ``file`` is resolved relative to the directory containing the spec YAML
    file (not relative to the process's current working directory) -- see
    ``macrotoolkit.data.load_data``.
    """

    model_config = ConfigDict(extra="forbid")

    file: str
    date_column: str
    mapping: dict[str, str]
    sample: SampleSpec = Field(default_factory=SampleSpec)

    @model_validator(mode="after")
    def _validate_fields(self) -> "DataSpec":
        if not self.file.strip():
            raise ValueError(
                "data.file must be a non-empty path to a CSV file, e.g. "
                "\"examples/toy/data.csv\" (resolved relative to the spec "
                "file's directory)."
            )
        if not self.date_column.strip():
            raise ValueError(
                "data.date_column must be a non-empty CSV column name "
                "containing the observation date, e.g. \"date\"."
            )
        if not self.mapping:
            raise ValueError(
                "data.mapping must be a non-empty dict mapping model "
                "variable names to CSV column names, e.g. "
                "{y: gdp_gap} for the local_level family."
            )
        for model_var, csv_col in self.mapping.items():
            if not isinstance(csv_col, str) or not csv_col.strip():
                raise ValueError(
                    f"data.mapping[{model_var!r}] must be a non-empty CSV "
                    f"column name string, got {csv_col!r}."
                )
        return self


class SamplerSpec(BaseModel):
    """CmdStanPy NUTS sampler configuration."""

    model_config = ConfigDict(extra="forbid")

    chains: int = Field(default=4, ge=1, description="Number of MCMC chains; must be >= 1.")
    warmup: int = Field(default=1000, ge=0, description="Warmup iterations per chain; must be >= 0.")
    sampling: int = Field(default=1000, ge=1, description="Post-warmup sampling iterations per chain; must be >= 1.")
    adapt_delta: float = Field(default=0.8, gt=0, lt=1, description="Target NUTS acceptance rate; must be in (0, 1).")
    max_treedepth: int = Field(default=10, ge=1, description="Maximum NUTS tree depth; must be >= 1.")
    seed: int = Field(default=20260813, description="RNG seed; any integer.")


class QcSpec(BaseModel):
    """Automatic per-run quality control (S6 WP3). OUTSIDE the run-identity
    hash -- a QC setting never changes the estimation -- and stored in the
    run dir as ``qc.yaml`` (refreshable, like ``outputs.yaml``).

    ``mirror_check``: before sampling, evaluate the exact rendered Stan
    program's Kalman-filter log-likelihood at ``mirror_points`` draws from
    the family's prior and compare each against the Python KF mirror
    (``macrotoolkit.smoother``); the max abs difference is recorded in the
    run's ``diagnostics.json`` and the run FAILS LOUDLY (no run directory
    is left behind) past ``mirror_tolerance`` -- G1's own 1e-8 gate,
    applied automatically to every fit. ON by default.
    """

    model_config = ConfigDict(extra="forbid")

    mirror_check: bool = True
    mirror_points: int = Field(default=5, ge=1, description="Prior draws evaluated on both sides of the mirror.")
    mirror_tolerance: float = Field(default=1e-8, gt=0, description="Max abs loglik difference tolerated (G1's gate).")
    mirror_rtol: float = Field(
        default=1e-11, ge=0,
        description="Relative slack per point: a point passes when |Stan - Python| < max(mirror_tolerance, mirror_rtol * |loglik|) (S8: flat-prior draws reach |loglik| ~ 1e7-1e9, where identical float64 arithmetic differs by ~1e-14 relative, and a recursive VAR at |a0| ~ 10 loses ~4 digits in the innovation Cholesky's cancellation).",
    )
    mirror_seed: int = Field(default=20260904, description="Seed for the prior draws the mirror check evaluates.")


class ModelSpec(BaseModel):
    """``model.family`` selects a registered family (``specs/schema``
    ``FAMILY_REGISTRY``); ``model.options`` is validated against that
    family's own Pydantic options model. This is a manually-dispatched
    discriminated union keyed on the ``family`` string, rather than a
    static ``typing.Union`` of every family's options type, so new families
    can be registered (``specs/schema/__init__.py``) without editing this
    class.
    """

    model_config = ConfigDict(extra="forbid")

    family: str
    options: Any = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_options(self) -> "ModelSpec":
        entry = get_family(self.family)  # raises a descriptive error if unknown
        if isinstance(self.options, entry.options_model):
            return self
        raw = self.options if self.options is not None else {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"model.options must be a mapping (dict) of option keys for "
                f"family {self.family!r}, got {type(raw).__name__}."
            )
        # Let the family's own model raise field-named errors on bad keys.
        self.options = entry.options_model.model_validate(raw)
        return self


class RunSpec(BaseModel):
    """Top-level run spec: mirrors the YAML in ``lw-sv-spec.md`` §2.3.

    ``priors`` is a free-form dict (empty = family defaults; any key
    overrides a default). S1's ``local_level`` family does not consume it
    (its priors are fixed in the Stan template -- see
    ``specs/schema/local_level.py``); later families validate specific
    prior keys in their own options/priors handling.

    ``outputs`` is family-specific: validated against the family registry's
    ``outputs_model`` when one is registered (S4 added ``lw_sv``'s --
    ``specs/schema/lw_sv.py``'s ``LwSvOutputs``), the same manually-dispatched
    pattern ``ModelSpec._validate_options`` uses for ``model.options``.
    ``local_level`` has no ``outputs_model`` registered, so its ``outputs``
    stays the free-form dict this field declares.

    ``qc`` (S6 WP3) configures the automatic per-run quality control
    (:class:`QcSpec`); like ``outputs`` it sits OUTSIDE the estimation
    identity (:meth:`to_estimation_yaml` drops both).
    """

    model_config = ConfigDict(extra="forbid")

    model: ModelSpec
    data: DataSpec
    priors: dict[str, Any] = Field(default_factory=dict)
    sampler: SamplerSpec = Field(default_factory=SamplerSpec)
    outputs: Any = Field(default_factory=dict)
    qc: QcSpec = Field(default_factory=QcSpec)

    @model_validator(mode="after")
    def _validate_required_mapping(self) -> "RunSpec":
        entry = get_family(self.model.family)
        required = entry.required_mapping_for(self.model.options)
        missing = [k for k in required if k not in self.data.mapping]
        if missing:
            example = ", ".join(f"{k}: <csv_column_name>" for k in required)
            raise ValueError(
                f"data.mapping is missing required key(s) {missing} for "
                f"model.family {self.model.family!r}. It must include: "
                f"{{{example}}}."
            )
        return self

    @model_validator(mode="after")
    def _validate_outputs(self) -> "RunSpec":
        entry = get_family(self.model.family)
        if entry.outputs_model is None:
            if not isinstance(self.outputs, dict):
                raise ValueError(
                    f"outputs must be a mapping (dict); got "
                    f"{type(self.outputs).__name__}."
                )
            return self
        if isinstance(self.outputs, entry.outputs_model):
            return self
        raw = self.outputs if self.outputs is not None else {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"outputs must be a mapping (dict) of output keys for "
                f"model.family {self.model.family!r}, got "
                f"{type(raw).__name__}."
            )
        self.outputs = entry.outputs_model.model_validate(raw)
        return self

    def to_canonical_yaml(self) -> str:
        """Serialize the FULL spec deterministically: parsed-and-revalidated
        field values, dumped with sorted keys, so YAML formatting, comments,
        or key order in the source file never change the serialization --
        only the *validated meaning* of the spec does.

        NOTE: since the S4.5 run-identity split (S5-decisions item 3), the
        run-identity hash consumes :meth:`to_estimation_yaml` (which
        EXCLUDES ``outputs``), not this full form.
        """
        payload = self.model_dump(mode="json")
        return yaml.safe_dump(payload, sort_keys=True, default_flow_style=False)

    def to_estimation_yaml(self) -> str:
        """The ESTIMATION identity serialization (S5-decisions item 3):
        the canonical spec WITHOUT the ``outputs:`` block. ``outputs``
        configures report/output modules only -- it never reaches the
        sampler, the rendered Stan program, or the data pipeline -- so two
        specs differing only in ``outputs`` describe the SAME estimation
        and must map to the same run identity (report options live in the
        run dir, outside the hash; changing the report-option schema no
        longer orphans MCMC runs). Same canonicalization discipline as
        :meth:`to_canonical_yaml` (validated values, sorted keys)."""
        payload = self.model_dump(mode="json")
        payload.pop("outputs", None)
        # qc (S6 WP3) is likewise QC configuration, not estimation content.
        payload.pop("qc", None)
        return yaml.safe_dump(payload, sort_keys=True, default_flow_style=False)

    def qc_to_canonical_yaml(self) -> str:
        """Canonical serialization of the ``qc:`` block alone (S6 WP3) --
        written to a run dir's ``qc.yaml``, outside the identity hash like
        ``outputs.yaml``."""
        payload = self.model_dump(mode="json")["qc"]
        return yaml.safe_dump(payload, sort_keys=True, default_flow_style=False)

    def outputs_to_canonical_yaml(self) -> str:
        """Canonical serialization of the ``outputs:`` block alone -- the
        report/output-module config written to a run dir's ``outputs.yaml``
        (outside the run-identity hash; see :meth:`to_estimation_yaml`)."""
        payload = self.model_dump(mode="json")["outputs"]
        return yaml.safe_dump(payload, sort_keys=True, default_flow_style=False)


def load_spec(path: str) -> RunSpec:
    """Load and validate a spec YAML file from ``path``."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"Spec file {path!r} is empty; expected a YAML mapping with top-level keys: model, data, sampler.")
    if not isinstance(raw, dict):
        raise ValueError(f"Spec file {path!r} must contain a YAML mapping at the top level, got {type(raw).__name__}.")
    return RunSpec.model_validate(raw)
