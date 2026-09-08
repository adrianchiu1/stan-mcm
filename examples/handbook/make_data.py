"""Convert the Blake & Mumtaz (2017) CCBS handbook data files (Chapters
1-3 of ``docs/applied-bayesian-economics-code-files.zip``, ``Code2017/``)
to the CSVs the ``examples/handbook/*`` specs read.

    python examples/handbook/make_data.py [--zip docs/applied-bayesian-economics-code-files.zip]

The handbook's ``.xls`` files carry no dates except ``inflation.xls``
(quarter labels); every other file's date column is reconstructed from
the sample the handbook's own scripts state in their comments (recorded
per file below) -- ISO first-of-period dates. ``xlrd`` reads the old
``.xls`` format. The outputs are committed so the specs and tests never
touch the zip.
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "data"


def _xls(zf: zipfile.ZipFile, member: str) -> list[list]:
    import xlrd

    wb = xlrd.open_workbook(file_contents=zf.read(member))
    sh = wb.sheets()[0]
    return [sh.row_values(r) for r in range(sh.nrows)]


def _dates(start: str, n: int, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq=freq)


def convert(zip_path: Path) -> None:
    OUT.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        # Chapter 1: US CPI inflation, quarterly, 1948Q1-2010Q4 (labels in the file).
        rows = _xls(zf, "Code2017/CHAPTER1/DATA/inflation.xls")[1:]
        labels = [r[0] for r in rows]
        dates = pd.PeriodIndex(labels, freq="Q").to_timestamp()
        pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "inflation": [float(r[1]) for r in rows]}).to_csv(OUT / "ch1_inflation.csv", index=False)

        # Chapter 2 examples 1/3/8: US GDP growth and inflation, 1948Q1-2010Q4
        # (example1.m: "data for US GDP growth and inflation 1948q1 2010q4").
        rows = _xls(zf, "Code2017/CHAPTER2/DATA/DATAIN.XLS")
        d = _dates("1948-01-01", len(rows), "QS")
        pd.DataFrame({"date": d.strftime("%Y-%m-%d"), "gdp_growth": [float(r[0]) for r in rows], "inflation": [float(r[1]) for r in rows]}).to_csv(OUT / "ch2_datain.csv", index=False)

        # Chapter 2 example 2: FFR, 10y yield, unemployment, inflation, monthly
        # ending 2010m12 (example2.m: "2007m1 2010m12"; the file has 46 rows,
        # so the reconstructed sample is 2007m3-2010m12).
        rows = _xls(zf, "Code2017/CHAPTER2/DATA/dataUS.xls")
        d = pd.date_range(end="2010-12-01", periods=len(rows), freq="MS")
        pd.DataFrame({"date": d.strftime("%Y-%m-%d"), "ffr": [float(r[0]) for r in rows], "bond10y": [float(r[1]) for r in rows],
                      "unemployment": [float(r[2]) for r in rows], "inflation": [float(r[3]) for r in rows]}).to_csv(OUT / "ch2_dataus_monthly.csv", index=False)

        # Chapter 2 examples 5-7: the 11-variable quarterly US panel (column
        # order as the handbook's sign-restriction code reads it), 160 rows
        # ending 2010Q4 -> 1971Q1-2010Q4.
        rows = _xls(zf, "Code2017/CHAPTER2/DATA/USDATA1.XLS")
        names = ["ffr", "gdp_growth", "cpi_inflation", "pce_growth", "unemployment", "investment", "net_exports", "m2", "bond10y", "stock_growth", "yen_dollar"]
        d = pd.date_range(end="2010-10-01", periods=len(rows), freq="QS")
        frame = {"date": d.strftime("%Y-%m-%d")}
        for j, nm in enumerate(names):
            frame[nm] = [float(r[j]) for r in rows]
        pd.DataFrame(frame).to_csv(OUT / "ch2_usdata_11var.csv", index=False)

        # Chapter 3 example 3: dlog(GDP), dlog(CPI), FFR quarterly (example3.m
        # plots from 1964.75 after a 40-quarter pre-sample + 2 lags; 226 rows
        # ending 2010Q2 -> 1954Q1-2010Q2), in per cent (the script divides by 100).
        rows = _xls(zf, "Code2017/CHAPTER3/DATA/USDATA.XLS")
        d = pd.date_range(end="2010-04-01", periods=len(rows), freq="QS")
        pd.DataFrame({"date": d.strftime("%Y-%m-%d"), "gdp_growth": [float(r[0]) for r in rows], "cpi_inflation": [float(r[1]) for r in rows], "ffr": [float(r[2]) for r in rows]}).to_csv(OUT / "ch3_usdata_tvp.csv", index=False)

        # Chapter 3 example 4: the 40-series UK panel (levels; names.xls,
        # index.xls give the transformation and fast/slow flags) and the
        # policy rate, 145 rows. The handbook gives no dates; a quarterly
        # index ending 2006Q1 is used (period index, NOT a documented sample).
        rows = _xls(zf, "Code2017/CHAPTER3/DATA/DATAIN.XLS")
        names = [str(r[0]).strip() for r in _xls(zf, "Code2017/CHAPTER3/DATA/NAMES.XLS")]
        index = _xls(zf, "Code2017/CHAPTER3/DATA/INDEX.XLS")
        rate = _xls(zf, "Code2017/CHAPTER3/DATA/BASERATE.XLS")
        d = pd.date_range(end="2006-01-01", periods=len(rows), freq="QS")
        slug = []
        for k, nm in enumerate(names):
            s = "".join(ch if ch.isalnum() else "_" for ch in nm.lower()).strip("_")
            while "__" in s:
                s = s.replace("__", "_")
            slug.append(f"s{k + 1:02d}_{s}")
        frame = {"date": d.strftime("%Y-%m-%d")}
        for j, nm in enumerate(slug):
            frame[nm] = [float(r[j]) for r in rows]
        frame["base_rate"] = [float(r[0]) for r in rate]
        pd.DataFrame(frame).to_csv(OUT / "ch3_uk_panel_levels.csv", index=False)
        pd.DataFrame({"series": slug, "name": names, "transform": [int(r[0]) for r in index], "fast": [int(r[1]) for r in index]}).to_csv(OUT / "ch3_uk_panel_index.csv", index=False)

        # Chapter 5 example 5 (S9): the UK price level, quarterly, 389 rows.
        # example5.m builds annual inflation 100*(ln P_t - ln P_{t-4}), drops
        # the first 4 rows, one regression lag and a 10-observation training
        # sample, and plots the remaining 374 rows on TT = 1917.75:0.25:2011
        # (1917Q4-2011Q1), so the price level runs 1914Q1-2011Q1. The CSV
        # carries the level AND the annual inflation from 1915Q1 (the spec
        # reads `infl`; the first four rows have no annual difference and are
        # omitted rather than written as missing values).
        import openpyxl

        wb = openpyxl.load_workbook(io.BytesIO(zf.read("Code2017/CHAPTER5/DATA/inflation.xlsx")), read_only=True)
        level = [float(r[0]) for r in wb.worksheets[0].iter_rows(values_only=True) if r and r[0] is not None]
        d = pd.date_range("1914-01-01", periods=len(level), freq="QS")
        lv = pd.Series(level, dtype=float)
        infl = 100.0 * (np.log(lv) - np.log(lv.shift(4)))
        pd.DataFrame({"date": d[4:].strftime("%Y-%m-%d"), "cpi_level": lv.to_numpy()[4:], "infl": infl.to_numpy()[4:]}).to_csv(OUT / "ch5_uk_inflation.csv", index=False)

    # Chapter 3 examples 1/2 (S9): the ARTIFICIAL TVP-regression DGP of
    # example1.m -- Y_t = beta_t X_t + e1_t, beta_t = beta_{t-1} + e2_t,
    # var(e1) = R = 0.01, var(e2) = Q = 0.001, beta_1 = 0, X ~ N(0, 1) --
    # simulated once at a recorded seed (the handbook's own script draws
    # fresh data on every run). T = 500 rows as in the script; the true
    # beta_t is written alongside so the README can overlay it (the spec
    # does not read it). Dates are a quarterly index (no calendar meaning).
    # The loop below is the script's `for j=2:t` verbatim: row 0 is the
    # handbook's zero initial row (beta_1 = 0 AND Y_1 = 0 -- its e1(1) is
    # drawn and discarded), kept as the handbook has it (numerics-reviewer
    # note, DECISIONS.md 2026-09-05) rather than "corrected".
    T, Q, R = 500, 0.001, 0.01
    rng = np.random.default_rng(20260905)
    e1 = rng.standard_normal(T) * np.sqrt(R)
    e2 = rng.standard_normal(T) * np.sqrt(Q)
    X = rng.standard_normal(T)
    beta = np.zeros(T)
    Y = np.zeros(T)
    for j in range(1, T):
        beta[j] = beta[j - 1] + e2[j]
        Y[j] = X[j] * beta[j] + e1[j]
    d = pd.date_range("1900-01-01", periods=T, freq="QS")
    pd.DataFrame({"date": d.strftime("%Y-%m-%d"), "Y": Y, "X": X, "beta_true": beta}).to_csv(OUT / "ch3_tvp_example1_sim.csv", index=False)
    print(f"wrote CSVs to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(REPO / "docs" / "applied-bayesian-economics-code-files.zip"))
    convert(Path(ap.parse_args().zip))
