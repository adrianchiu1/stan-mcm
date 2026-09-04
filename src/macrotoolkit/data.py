"""CSV data loading for macrotoolkit run specs.

Responsibilities:
- resolve `data.file` relative to the spec file's directory,
- hash the raw CSV bytes (SHA-256) for the run-identity hash -- this hash
  covers the entire source file's bytes, independent of `date_column` /
  `mapping` / `sample` (those are part of the canonicalized spec and are
  therefore already covered by the spec hash component of run identity),
- load, rename columns per `data.mapping`, parse `data.date_column`, and
  trim to `data.sample`,
- return a tidy, model-ready DataFrame.

Numerical conventions: this module is generic over model families and does
not itself apply any of lw-sv-spec.md's unit conventions (`g` annualized,
`h` log-variance, `400*dlog(P)` inflation) -- those belong to family-
specific code once the LW-SV family exists (S2). The `local_level` toy
family used in S1 has no such quantities; see
`tests/test_units_conventions.py` for where those conventions get tested
once S2 introduces them.

No silent fallbacks: missing files, missing columns, unparseable dates, and
missing/non-numeric values in mapped columns are all hard errors that name
the offending field/column and say what to do about it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from specs.schema.base import RunSpec


def file_sha256(path: Path) -> str:
    """SHA-256 hex digest of a file's raw bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_data_path(spec: RunSpec, spec_path: Path | None = None, *, base_dir: Path | None = None) -> Path:
    """Resolve `spec.data.file`, relative to the directory containing the
    spec YAML file (not the process's current working directory) -- or,
    for a spec built in Python with no file (S6 WP1's API), relative to an
    explicit ``base_dir``. Exactly one of the two anchors is given."""
    if (spec_path is None) == (base_dir is None):
        raise ValueError("resolve_data_path needs exactly one of spec_path or base_dir.")
    spec_dir = Path(base_dir).resolve() if base_dir is not None else Path(spec_path).resolve().parent
    data_path = (spec_dir / spec.data.file).resolve()
    if not data_path.is_file():
        where = f"in {spec_path}" if spec_path is not None else "in the RunSpec"
        raise FileNotFoundError(
            f"data.file {spec.data.file!r} (resolved to {data_path}) does "
            f"not exist. Fix data.file {where} to point at a CSV file, "
            f"relative to the base directory ({spec_dir})."
        )
    return data_path


def _looks_like_year_quarter(s: str) -> bool:
    return "Q" in s.upper() and any(c.isdigit() for c in s)


def _parse_sample_bound(value: str, field_name: str, end: bool = False) -> pd.Timestamp:
    try:
        if _looks_like_year_quarter(value):
            period = pd.Period(value, freq="Q")
            return period.to_timestamp(how="end" if end else "start")
        return pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(
            f"{field_name} {value!r} could not be parsed as a date. Use an "
            f"ISO date (\"1961-01-01\") or a year-quarter (\"1961Q1\")."
        ) from exc


def load_data(spec: RunSpec, spec_path: Path | None = None, *, base_dir: Path | None = None) -> tuple[pd.DataFrame, str, Path]:
    """Load, map, and trim the data referenced by `spec`.

    Returns `(mapped_df, raw_file_hash, resolved_data_path)`. `mapped_df`
    has a `date` column (parsed datetime) plus one numeric column per
    `spec.data.mapping` key (e.g. `y`), sorted by date, trimmed to
    `spec.data.sample`. ``data.file`` resolves against the spec file's
    directory (``spec_path``) or an explicit ``base_dir`` (see
    :func:`resolve_data_path`).
    """
    data_path = resolve_data_path(spec, spec_path, base_dir=base_dir)
    raw_hash = file_sha256(data_path)

    try:
        raw_df = pd.read_csv(data_path)
    except Exception as exc:
        raise ValueError(f"Failed to parse {data_path} as CSV: {exc}") from exc

    if spec.data.date_column not in raw_df.columns:
        raise ValueError(
            f"data.date_column {spec.data.date_column!r} not found in "
            f"{data_path}. Available columns: {list(raw_df.columns)}."
        )
    for model_var, csv_col in spec.data.mapping.items():
        if csv_col not in raw_df.columns:
            raise ValueError(
                f"data.mapping[{model_var!r}] = {csv_col!r} not found in "
                f"{data_path}. Available columns: {list(raw_df.columns)}."
            )

    try:
        dates = pd.to_datetime(raw_df[spec.data.date_column])
    except Exception as exc:
        raise ValueError(
            f"data.date_column {spec.data.date_column!r} in {data_path} "
            f"could not be parsed as dates: {exc}"
        ) from exc

    df = pd.DataFrame({"date": dates})
    for model_var, csv_col in spec.data.mapping.items():
        series = pd.to_numeric(raw_df[csv_col], errors="coerce")
        n_bad = int(series.isna().sum())
        if n_bad:
            raise ValueError(
                f"data.mapping[{model_var!r}] = {csv_col!r} in {data_path} "
                f"has {n_bad} value(s) that are missing or not numeric. "
                f"Fill or drop them before running -- macrotoolkit does not "
                f"silently impute missing data."
            )
        df[model_var] = series

    df = df.sort_values("date").reset_index(drop=True)

    if spec.data.sample.start is not None:
        start_ts = _parse_sample_bound(spec.data.sample.start, "data.sample.start")
        df = df[df["date"] >= start_ts]
    if spec.data.sample.end is not None:
        end_ts = _parse_sample_bound(spec.data.sample.end, "data.sample.end", end=True)
        df = df[df["date"] <= end_ts]

    df = df.reset_index(drop=True)
    if df.empty:
        available = f"{dates.min()} to {dates.max()}" if len(dates) else "(no rows)"
        raise ValueError(
            f"data.sample (start={spec.data.sample.start!r}, "
            f"end={spec.data.sample.end!r}) trims {data_path} to zero rows. "
            f"Data available range is {available}; widen the sample or fix "
            f"the date format."
        )

    return df, raw_hash, data_path
