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
    print(f"wrote CSVs to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(REPO / "docs" / "applied-bayesian-economics-code-files.zip"))
    convert(Path(ap.parse_args().zip))
