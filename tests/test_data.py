"""`macrotoolkit.data.load_data` / `file_sha256` tests.

Covers: column mapping/renaming, date parsing (ISO and "YYYYQn"), sample
trim (start/end/both/neither), missing-file error, missing-column error,
non-numeric-value error, empty-after-trim error, and `file_sha256`
correctness against a manually computed digest.
"""
from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from specs.schema.base import RunSpec
from macrotoolkit.data import file_sha256, load_data, resolve_data_path


def _spec(tmp_path, csv_name="data.csv", date_column="date", mapping=None, sample=None):
    d = {
        "model": {"family": "local_level", "options": {}},
        "data": {
            "file": csv_name,
            "date_column": date_column,
            "mapping": mapping or {"y": "obs"},
        },
    }
    if sample is not None:
        d["data"]["sample"] = sample
    return RunSpec.model_validate(d)


def _write_csv(tmp_path, name, rows):
    path = tmp_path / name
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


# --- basic mapping / renaming ---------------------------------------------


def test_load_data_maps_and_renames_columns(tmp_path):
    _write_csv(
        tmp_path,
        "data.csv",
        {"date": ["2000-01-01", "2000-04-01", "2000-07-01"], "obs": [1.0, 2.0, 3.0]},
    )
    spec = _spec(tmp_path)
    df, raw_hash, data_path = load_data(spec, tmp_path / "spec.yaml")
    assert list(df.columns) == ["date", "y"]
    assert df["y"].tolist() == [1.0, 2.0, 3.0]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert data_path == (tmp_path / "data.csv").resolve()


def test_load_data_maps_multiple_columns(tmp_path):
    _write_csv(
        tmp_path,
        "data.csv",
        {"date": ["2000-01-01", "2000-04-01"], "gdp": [1.0, 2.0], "cpi": [0.5, 0.6]},
    )
    spec = _spec(tmp_path, mapping={"y": "gdp", "z": "cpi"})
    df, _, _ = load_data(spec, tmp_path / "spec.yaml")
    assert set(df.columns) == {"date", "y", "z"}
    assert df["y"].tolist() == [1.0, 2.0]
    assert df["z"].tolist() == [0.5, 0.6]


def test_load_data_sorts_by_date(tmp_path):
    _write_csv(
        tmp_path,
        "data.csv",
        {"date": ["2000-07-01", "2000-01-01", "2000-04-01"], "obs": [3.0, 1.0, 2.0]},
    )
    spec = _spec(tmp_path)
    df, _, _ = load_data(spec, tmp_path / "spec.yaml")
    assert df["y"].tolist() == [1.0, 2.0, 3.0]
    assert list(df["date"]) == sorted(df["date"])


# --- date parsing ----------------------------------------------------------


def test_load_data_parses_iso_dates(tmp_path):
    _write_csv(
        tmp_path,
        "data.csv",
        {"date": ["1961-01-01", "1961-04-01"], "obs": [1.0, 2.0]},
    )
    spec = _spec(tmp_path)
    df, _, _ = load_data(spec, tmp_path / "spec.yaml")
    assert df["date"].iloc[0] == pd.Timestamp("1961-01-01")
    assert df["date"].iloc[1] == pd.Timestamp("1961-04-01")


def test_load_data_parses_year_quarter_sample_bounds(tmp_path):
    """The date *column* itself is parsed by pandas (ISO-ish strings);
    "1961Q1"-style strings are exercised via the `sample.start`/`end`
    bounds, which is where `_parse_sample_bound` actually handles them."""
    _write_csv(
        tmp_path,
        "data.csv",
        {
            "date": ["1960-10-01", "1961-01-01", "1961-04-01", "1961-07-01"],
            "obs": [1.0, 2.0, 3.0, 4.0],
        },
    )
    spec = _spec(tmp_path, sample={"start": "1961Q1", "end": "1961Q1"})
    df, _, _ = load_data(spec, tmp_path / "spec.yaml")
    assert df["y"].tolist() == [2.0]
    assert df["date"].iloc[0] == pd.Timestamp("1961-01-01")


# --- sample trim -----------------------------------------------------------


def _quarterly_csv(tmp_path):
    dates = pd.date_range("2000-01-01", periods=8, freq="QS").strftime("%Y-%m-%d").tolist()
    return _write_csv(tmp_path, "data.csv", {"date": dates, "obs": list(range(1, 9))})


def test_sample_trim_start_only(tmp_path):
    _quarterly_csv(tmp_path)
    spec = _spec(tmp_path, sample={"start": "2000-10-01", "end": None})
    df, _, _ = load_data(spec, tmp_path / "spec.yaml")
    assert df["y"].tolist() == [4, 5, 6, 7, 8]


def test_sample_trim_end_only(tmp_path):
    _quarterly_csv(tmp_path)
    spec = _spec(tmp_path, sample={"start": None, "end": "2000-10-01"})
    df, _, _ = load_data(spec, tmp_path / "spec.yaml")
    assert df["y"].tolist() == [1, 2, 3, 4]


def test_sample_trim_start_and_end(tmp_path):
    _quarterly_csv(tmp_path)
    spec = _spec(tmp_path, sample={"start": "2000-04-01", "end": "2000-10-01"})
    df, _, _ = load_data(spec, tmp_path / "spec.yaml")
    assert df["y"].tolist() == [2, 3, 4]


def test_sample_trim_neither(tmp_path):
    _quarterly_csv(tmp_path)
    spec = _spec(tmp_path, sample={"start": None, "end": None})
    df, _, _ = load_data(spec, tmp_path / "spec.yaml")
    assert df["y"].tolist() == list(range(1, 9))


def test_sample_trim_to_empty_raises(tmp_path):
    _quarterly_csv(tmp_path)
    spec = _spec(tmp_path, sample={"start": "2099-01-01", "end": None})
    with pytest.raises(ValueError, match="zero rows"):
        load_data(spec, tmp_path / "spec.yaml")


# --- error cases -------------------------------------------------------


def test_missing_file_raises(tmp_path):
    spec = _spec(tmp_path, csv_name="nope.csv")
    with pytest.raises(FileNotFoundError, match="nope.csv"):
        load_data(spec, tmp_path / "spec.yaml")


def test_missing_date_column_raises(tmp_path):
    _write_csv(tmp_path, "data.csv", {"not_date": ["2000-01-01"], "obs": [1.0]})
    spec = _spec(tmp_path)
    with pytest.raises(ValueError, match="date_column"):
        load_data(spec, tmp_path / "spec.yaml")


def test_missing_mapped_column_raises(tmp_path):
    _write_csv(tmp_path, "data.csv", {"date": ["2000-01-01"], "not_obs": [1.0]})
    spec = _spec(tmp_path)
    with pytest.raises(ValueError, match=r"mapping\['y'\]"):
        load_data(spec, tmp_path / "spec.yaml")


def test_non_numeric_value_raises(tmp_path):
    _write_csv(
        tmp_path,
        "data.csv",
        {"date": ["2000-01-01", "2000-04-01"], "obs": ["1.0", "not_a_number"]},
    )
    spec = _spec(tmp_path)
    with pytest.raises(ValueError, match="missing or not numeric"):
        load_data(spec, tmp_path / "spec.yaml")


def test_missing_value_raises(tmp_path):
    _write_csv(
        tmp_path,
        "data.csv",
        {"date": ["2000-01-01", "2000-04-01"], "obs": [1.0, None]},
    )
    spec = _spec(tmp_path)
    with pytest.raises(ValueError, match="missing or not numeric"):
        load_data(spec, tmp_path / "spec.yaml")


def test_unparseable_date_raises(tmp_path):
    _write_csv(
        tmp_path,
        "data.csv",
        {"date": ["not-a-date", "also-not-a-date"], "obs": [1.0, 2.0]},
    )
    spec = _spec(tmp_path)
    with pytest.raises(ValueError):
        load_data(spec, tmp_path / "spec.yaml")


def test_data_file_resolved_relative_to_spec_dir(tmp_path):
    subdir = tmp_path / "sub"
    subdir.mkdir()
    _write_csv(subdir, "data.csv", {"date": ["2000-01-01"], "obs": [1.0]})
    spec = _spec(subdir)
    # spec_path lives in `subdir`, so data.file="data.csv" resolves there
    # even though the process cwd is elsewhere.
    df, _, data_path = load_data(spec, subdir / "spec.yaml")
    assert data_path == (subdir / "data.csv").resolve()
    assert len(df) == 1


# --- file_sha256 -----------------------------------------------------------


def test_file_sha256_matches_manual_digest(tmp_path):
    path = tmp_path / "known.txt"
    content = b"macrotoolkit S1 test fixture\n"
    path.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert file_sha256(path) == expected


def test_file_sha256_pinned_value(tmp_path):
    path = tmp_path / "known.txt"
    path.write_bytes(b"hello world\n")
    # sha256("hello world\n") is a well-known, independently verifiable value.
    assert file_sha256(path) == "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447"


def test_file_sha256_changes_with_content(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_bytes(b"content A")
    b.write_bytes(b"content B")
    assert file_sha256(a) != file_sha256(b)


def test_file_sha256_stable_for_same_content(tmp_path):
    a = tmp_path / "a.txt"
    a.write_bytes(b"same content")
    h1 = file_sha256(a)
    h2 = file_sha256(a)
    assert h1 == h2


def test_load_data_raw_hash_matches_file_sha256(tmp_path):
    _write_csv(tmp_path, "data.csv", {"date": ["2000-01-01"], "obs": [1.0]})
    spec = _spec(tmp_path)
    _, raw_hash, data_path = load_data(spec, tmp_path / "spec.yaml")
    assert raw_hash == file_sha256(data_path)


def test_resolve_data_path_missing_file_message(tmp_path):
    spec = _spec(tmp_path, csv_name="missing.csv")
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        resolve_data_path(spec, tmp_path / "spec.yaml")
