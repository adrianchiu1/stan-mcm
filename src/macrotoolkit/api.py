"""The notebook-first public API (S6 WP1): the full estimate → results →
figures → report → sweep workflow from Python, with no YAML files and no
CLI. ``mtk`` (``macrotoolkit.cli``) is a thin shell over this module.

Design rules (plans/S6-plan.md):

- **The Pydantic spec models ARE the spec API.** ``RunSpec`` and its
  parts are re-exported here; :func:`spec` is a convenience builder that
  returns a validated ``RunSpec``. There is no parallel "Python spec": a
  spec built here with the same content as a YAML file serializes to the
  same ``to_estimation_yaml()`` and therefore the same run identity
  (pinned by ``tests/test_api.py``).
- **Runs stay immutable and hash-identified.** :func:`fit` is the same
  run-store pipeline ``mtk run`` uses (``macrotoolkit.run.run_spec``);
  :class:`Run` is a read-only handle over a completed run directory.
- **Data may be a CSV path or a pandas DataFrame.** A DataFrame is
  serialized to a canonical CSV (see :func:`stage_dataframe`) so its run
  identity is a deterministic function of the frame's content, through
  exactly the same hashing/snapshot path a CSV file takes.
- **Figures, not files.** :meth:`Run.outputs` exposes the family's
  declared output modules (:mod:`macrotoolkit.outputs`) as live matplotlib
  ``Figure`` objects; :meth:`Run.report` still writes the self-contained
  ``report.html``.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from specs.schema import FAMILY_REGISTRY, get_family
from specs.schema.base import (  # noqa: F401  (re-exported spec API)
    DataSpec,
    ModelSpec,
    QcSpec,
    RunSpec,
    SampleSpec,
    SamplerSpec,
    load_spec,
)

from macrotoolkit.run import REPO_ROOT, RunResult, load_run_spec
from macrotoolkit.run import run_spec as _run_spec

__all__ = [
    "DataSpec",
    "ModelSpec",
    "QcSpec",
    "RunSpec",
    "SampleSpec",
    "SamplerSpec",
    "Run",
    "Outputs",
    "spec",
    "fit",
    "load_run",
    "report",
    "results",
    "sweep",
    "validate",
    "stage_dataframe",
    "load_spec",
    "FAMILY_REGISTRY",
]

#: The file name a staged DataFrame takes (it enters the run identity as
#: ``data.file``, so it is a fixed constant, not a temp-dir path).
DATAFRAME_FILE_NAME = "dataframe.csv"


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------


def spec(
    family: str,
    *,
    data: DataSpec | Mapping[str, Any],
    options: Mapping[str, Any] | Any | None = None,
    priors: Mapping[str, Any] | None = None,
    sampler: SamplerSpec | Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | Any | None = None,
    qc: QcSpec | Mapping[str, Any] | None = None,
) -> RunSpec:
    """Build a validated :class:`RunSpec` from Python objects -- the same
    validation a YAML spec gets (unknown families, missing mapping keys,
    unknown options and prior names all fail loudly, naming the field).

    ``data`` is a :class:`DataSpec` or a dict with ``file``,
    ``date_column``, ``mapping`` (and optionally ``sample``). For a
    DataFrame workflow, ``file`` may be anything -- :func:`fit` replaces it
    with :data:`DATAFRAME_FILE_NAME` when a frame is supplied -- but the
    conventional value is ``"dataframe.csv"`` so the spec reads the same
    before and after staging.
    """
    raw: dict[str, Any] = {
        "model": {"family": family, "options": dict(options) if isinstance(options, Mapping) else (options if options is not None else {})},
        "data": data if isinstance(data, DataSpec) else dict(data),
        "priors": dict(priors) if priors else {},
    }
    if sampler is not None:
        raw["sampler"] = sampler if isinstance(sampler, SamplerSpec) else dict(sampler)
    if outputs is not None:
        raw["outputs"] = dict(outputs) if isinstance(outputs, Mapping) else outputs
    if qc is not None:
        raw["qc"] = qc if isinstance(qc, QcSpec) else dict(qc)
    return RunSpec.model_validate(raw)


# ---------------------------------------------------------------------------
# DataFrame staging
# ---------------------------------------------------------------------------


def dataframe_csv_bytes(df: pd.DataFrame, run_spec: RunSpec) -> bytes:
    """The canonical CSV serialization of ``df`` for ``run_spec``: only the
    columns the spec consumes (``data.date_column`` first, then the mapped
    columns in ``data.mapping``'s key order), index dropped, dates as ISO
    ``YYYY-MM-DD`` (datetime-typed columns) or as given (string columns),
    floats in pandas' shortest round-trip representation, ``\\n`` line
    endings. Deterministic in the frame's CONTENT, so two frames with the
    same values (in any column order, any index) stage to identical bytes
    and therefore the same run identity.

    Columns the spec does not reference are deliberately excluded: a
    notebook frame often carries many unrelated series, and the run's
    ``data.snapshot.csv`` should record exactly what the estimation used.
    (A CSV FILE is hashed byte-for-byte as given -- extra columns in a
    file do change its identity. The two paths are documented as
    different; ``tests/test_api.py`` pins both.)
    """
    ds = run_spec.data
    needed = [ds.date_column] + [c for c in ds.mapping.values() if c != ds.date_column]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing column(s) {missing} required by the spec "
            f"(date_column={ds.date_column!r}, mapping={dict(ds.mapping)!r}). "
            f"Available columns: {list(df.columns)}."
        )
    out = df.loc[:, needed].copy()
    if pd.api.types.is_datetime64_any_dtype(out[ds.date_column]):
        out[ds.date_column] = pd.to_datetime(out[ds.date_column]).dt.strftime("%Y-%m-%d")
    return out.to_csv(index=False, lineterminator="\n").encode("utf-8")


def stage_dataframe(df: pd.DataFrame, run_spec: RunSpec, staging_dir: Path) -> RunSpec:
    """Write ``df``'s canonical CSV (:func:`dataframe_csv_bytes`) to
    ``staging_dir / DATAFRAME_FILE_NAME`` and return ``run_spec`` with
    ``data.file`` pointed at it. The staged file is the ordinary
    ``data.file`` input from here on (hashed and snapshotted by the run
    store exactly like a user's CSV)."""
    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / DATAFRAME_FILE_NAME).write_bytes(dataframe_csv_bytes(df, run_spec))
    return run_spec.model_copy(
        update={"data": run_spec.data.model_copy(update={"file": DATAFRAME_FILE_NAME})}
    )


# ---------------------------------------------------------------------------
# Run handle
# ---------------------------------------------------------------------------


class Outputs:
    """A run's family-declared output modules as live objects: ``names``,
    :meth:`compute` (the module's data object, cached), :meth:`figure`
    (its matplotlib ``Figure`` -- or ``{key: Figure}`` for multi-figure
    modules), :meth:`figures` (everything). Figures are returned open;
    the caller shows/saves/closes them."""

    def __init__(self, run: "Run") -> None:
        self._run = run
        entry = get_family(run.family)
        modules = entry.resolve("output_modules")
        if modules is None:
            raise NotImplementedError(
                f"Family {run.family!r} declares no output_modules capability "
                f"in FAMILY_REGISTRY (specs/schema) -- it has no output layer yet."
            )
        self._modules = {m.name: m for m in modules}
        self._data: dict[str, Any] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._modules)

    def module(self, name: str):
        if name not in self._modules:
            raise KeyError(
                f"No output module named {name!r} for family {self._run.family!r}; "
                f"available: {list(self._modules)}."
            )
        return self._modules[name]

    def compute(self, name: str) -> Any:
        """The module's data object (per-draw arrays etc.), computed once
        and cached on this handle."""
        if name not in self._data:
            self._data[name] = self.module(name).compute(self._run.results())
        return self._data[name]

    def figure(self, name: str):
        """One module's figure(s): a single ``Figure`` for single-figure
        modules, else an ordered ``{key: Figure}`` dict."""
        mod = self.module(name)
        figs = mod.figures(self.compute(name), self._run.results())
        if list(figs) == [name]:
            return figs[name]
        return figs

    def figures(self) -> dict[str, dict[str, Any]]:
        """Every module's figures: ``{module: {label: Figure}}``."""
        return {
            name: mod.figures(self.compute(name), self._run.results())
            for name, mod in self._modules.items()
        }


@dataclass
class Run:
    """A read-only handle over one completed, immutable run directory.

    Attributes/properties: ``hash`` (the 12-char run id, = the directory
    name), ``run_dir``, ``is_new`` (whether :func:`fit` sampled it or found
    it already complete), ``spec`` (the reassembled full ``RunSpec``),
    ``family``, ``diagnostics`` (the stored ``diagnostics.json`` dict),
    ``verdict`` (PASS/WARN/FAIL), ``mirror_check`` (the automatic
    Stan-vs-Python KF cross-check record, S6 WP3), ``idata`` (ArviZ
    InferenceData, loaded lazily). Methods: :meth:`results` (the family's
    results object, cached), :meth:`outputs` (figures), :meth:`param_table`
    (posterior summary DataFrame), :meth:`report` (writes report.html).
    """

    run_dir: Path
    is_new: bool = False
    _spec: RunSpec | None = field(default=None, repr=False)
    _results: Any = field(default=None, repr=False)
    _idata: Any = field(default=None, repr=False)
    _outputs: Outputs | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        if not (self.run_dir / "_SUCCESS").is_file():
            raise FileNotFoundError(
                f"{self.run_dir} is not a completed run directory (no _SUCCESS "
                f"marker). If a previous run crashed there, inspect log.txt and "
                f"remove the directory before re-running."
            )

    # --- identity -----------------------------------------------------------
    @property
    def hash(self) -> str:
        return self.run_dir.name

    @property
    def spec(self) -> RunSpec:
        if self._spec is None:
            self._spec = load_run_spec(self.run_dir)
        return self._spec

    @property
    def family(self) -> str:
        return self.spec.model.family

    # --- diagnostics ---------------------------------------------------------
    @property
    def diagnostics(self) -> dict:
        path = self.run_dir / "diagnostics.json"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found in run directory {self.run_dir}.")
        return json.loads(path.read_text())

    @property
    def verdict(self) -> str:
        return self.diagnostics["verdict"]

    @property
    def mirror_check(self) -> dict | None:
        """The automatic Stan-vs-Python KF mirror check record (S6 WP3), or
        ``None`` for runs that predate it / had it disabled."""
        return self.diagnostics.get("mirror_check")

    # --- posterior / results -------------------------------------------------
    @property
    def idata(self):
        if self._idata is None:
            import arviz as az

            self._idata = az.from_netcdf(str(self.run_dir / "draws.nc"))
        return self._idata

    def results(self) -> Any:
        """The family's loaded results object (the registry's
        ``results_loader`` capability), cached on this handle."""
        if self._results is None:
            loader = get_family(self.family).resolve("results_loader")
            if loader is None:
                raise NotImplementedError(
                    f"Family {self.family!r} declares no results_loader capability "
                    f"in FAMILY_REGISTRY (specs/schema)."
                )
            self._results = loader(self.run_dir)
        return self._results

    def outputs(self) -> Outputs:
        if self._outputs is None:
            self._outputs = Outputs(self)
        return self._outputs

    def param_table(self) -> pd.DataFrame:
        """Posterior median, 90% CI, R-hat, bulk/tail ESS for every scalar
        parameter -- the report's parameter table as a DataFrame."""
        from macrotoolkit.report import param_table_rows

        return pd.DataFrame(param_table_rows(self.idata)).set_index("name")

    def report(self) -> Path:
        """Write (or rewrite) ``report.html`` into the run directory via the
        family's registered report writer; returns the path."""
        writer = get_family(self.family).resolve("report_writer")
        if writer is None:
            raise NotImplementedError(
                f"Family {self.family!r} declares no report_writer capability in "
                f"FAMILY_REGISTRY (specs/schema) -- this family has no report "
                f"support yet."
            )
        return Path(writer(self.run_dir))

    def __repr__(self) -> str:
        return f"Run({self.hash}, family={self.family!r}, verdict={self.verdict!r}, dir={str(self.run_dir)!r})"


def _from_result(result: RunResult) -> Run:
    return Run(run_dir=result.run_dir, is_new=result.is_new)


# ---------------------------------------------------------------------------
# fit / load_run / report / results
# ---------------------------------------------------------------------------


def fit(
    run_spec: RunSpec | str | Path,
    data: pd.DataFrame | str | Path | None = None,
    *,
    base_dir: str | Path | None = None,
    runs_root: str | Path | None = None,
) -> Run:
    """Estimate ``run_spec`` (or idempotently reuse the identical completed
    run) and return a :class:`Run` handle.

    - ``run_spec``: a :class:`RunSpec`, or a path to a spec YAML (then
      ``base_dir`` defaults to the file's directory -- exactly ``mtk run``).
    - ``data``: ``None`` (use ``spec.data.file``, resolved against
      ``base_dir``, default the current working directory); a CSV path
      (replaces ``spec.data.file`` with the path AS GIVEN, resolved against
      ``base_dir`` -- hash parity with a YAML spec holds when the relative
      path string and base directory match the YAML's); or a pandas
      DataFrame (staged to a canonical CSV, see :func:`stage_dataframe`).
    - ``runs_root``: the run store root (default ``<repo>/runs``).
    """
    if isinstance(run_spec, (str, Path)):
        spec_path = Path(run_spec).resolve()
        run_spec_obj = load_spec(str(spec_path))
        base = Path(base_dir).resolve() if base_dir is not None else spec_path.parent
    else:
        run_spec_obj = run_spec
        base = Path(base_dir).resolve() if base_dir is not None else Path.cwd()

    if isinstance(data, pd.DataFrame):
        with tempfile.TemporaryDirectory(prefix="mtk_dataframe_") as tmp:
            staged = stage_dataframe(data, run_spec_obj, Path(tmp))
            result = _run_spec(staged, base_dir=Path(tmp), runs_root=runs_root)
        return _from_result(result)
    if data is not None:
        run_spec_obj = run_spec_obj.model_copy(
            update={"data": run_spec_obj.data.model_copy(update={"file": str(data)})}
        )
    result = _run_spec(run_spec_obj, base_dir=base, runs_root=runs_root)
    return _from_result(result)


def load_run(run: str | Path | Run, runs_root: str | Path | None = None) -> Run:
    """Open an existing completed run: by 12-char hash (looked up under
    ``runs_root``, default ``<repo>/runs``) or by directory path (e.g. a
    ``runs-archive/<hash>`` fixture)."""
    if isinstance(run, Run):
        return run
    candidate = Path(run)
    if candidate.is_dir():
        return Run(run_dir=candidate.resolve())
    root = Path(runs_root).resolve() if runs_root is not None else REPO_ROOT / "runs"
    run_dir = root / str(run)
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Run {run!r} not found: neither a directory nor {run_dir} exists. "
            f"Pass a run hash (a 12-character runs/<hash12>/ name), a run "
            f"directory path, or the correct runs_root."
        )
    return Run(run_dir=run_dir)


def report(run: str | Path | Run, runs_root: str | Path | None = None) -> Path:
    """``mtk report``: write the self-contained report.html for ``run``."""
    return load_run(run, runs_root).report()


def results(run: str | Path | Run, runs_root: str | Path | None = None) -> Any:
    """The family results object for ``run`` (see :meth:`Run.results`)."""
    return load_run(run, runs_root).results()


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


def sweep(
    sweep_spec: "str | Path | Any",
    *,
    base_spec: RunSpec | str | Path | None = None,
    data: pd.DataFrame | None = None,
    base_dir: str | Path | None = None,
    runs_root: str | Path | None = None,
    sweeps_root: str | Path | None = None,
):
    """Run a prior sweep (S5-decisions item 9) from Python and return a
    :class:`macrotoolkit.sweep.SweepResult` whose ``comparison`` attribute
    carries the programmatic comparison (``comparison.table()`` is the
    long-form DataFrame of per-cell posterior summaries and prior→posterior
    contraction; ``comparison.headline`` the overlaid headline series) --
    the same data the HTML report renders.

    ``sweep_spec``: a sweep YAML path (``mtk sweep``'s input; its
    ``base_spec`` resolves relative to the file) or a
    :class:`specs.schema.sweep.SweepSpec` / dict (``name``, ``cells``; with
    ``base_spec`` then given here as a :class:`RunSpec` or spec path).
    ``data``: an optional DataFrame standing in for the base spec's
    ``data.file`` (staged once, shared by every cell).
    """
    from specs.schema.sweep import SweepSpec, load_sweep_spec

    from macrotoolkit.sweep import run_sweep_spec

    if isinstance(sweep_spec, (str, Path)):
        path = Path(sweep_spec).resolve()
        sw = load_sweep_spec(str(path))
        if base_spec is None:
            base_spec = path.parent / sw.base_spec
        if base_dir is None and isinstance(base_spec, (str, Path)):
            base_dir = Path(base_spec).resolve().parent
    elif isinstance(sweep_spec, SweepSpec):
        sw = sweep_spec
    else:
        raw = dict(sweep_spec)
        raw.setdefault("base_spec", "<python>")
        sw = SweepSpec.model_validate(raw)

    if base_spec is None:
        raise ValueError("sweep() needs a base spec: a sweep YAML with base_spec, or base_spec=RunSpec/path.")
    if isinstance(base_spec, (str, Path)):
        bpath = Path(base_spec).resolve()
        base = load_spec(str(bpath))
        if base_dir is None:
            base_dir = bpath.parent
        base_label = str(bpath)
    else:
        base = base_spec
        base_label = "<RunSpec built in Python>"

    if data is not None:
        with tempfile.TemporaryDirectory(prefix="mtk_sweep_dataframe_") as tmp:
            staged = stage_dataframe(data, base, Path(tmp))
            return run_sweep_spec(
                sw, staged, base_dir=Path(tmp), base_label=base_label,
                runs_root=runs_root, sweeps_root=sweeps_root,
            )
    base_dir = Path(base_dir).resolve() if base_dir is not None else Path.cwd()
    return run_sweep_spec(
        sw, base, base_dir=base_dir, base_label=base_label,
        runs_root=runs_root, sweeps_root=sweeps_root,
    )


# ---------------------------------------------------------------------------
# validate (S6 WP3)
# ---------------------------------------------------------------------------


def validate(family: str, tier: str = "fast", out_root: str | Path | None = None, *, progress=None):
    """``mtk validate <family> --tier <tier>``: run the family's registered
    validation gates (``fast`` | ``recovery`` | ``sbc`` | ``all``) and
    write ``validation/<family>/report.html``. Returns a
    :class:`macrotoolkit.validation.suite.ValidationResult` (``verdict``,
    per-gate ``gates`` with verdict/reasons/metrics, ``report_path``)."""
    from macrotoolkit.validation.suite import run_validation

    return run_validation(family, tier=tier, out_root=out_root, progress=progress)
