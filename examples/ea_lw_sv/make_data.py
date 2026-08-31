"""Derive examples/ea_lw_sv/data/ea_quarterly.csv from the NY Fed
current-estimates workbook's "EA input data" sheet
(tests/fixtures/hlw/data/Holston_Laubach_Williams_current_estimates.xlsx).

Same transforms as the US example (lw-sv-spec.md §1.1, locked decision 3):
lgdp100 = 100*gdp.log; core infl already annualized q/q %; real_rate =
interest - inflation.expectations.

Window: 1971Q1-2019Q4. The start gives the 4 pre-sample lag quarters
before HLW's own EA estimation start (1972Q1). The end deliberately stops
before 2020: the S2 model is the 2017 (pre-COVID) form with no
covid.ind/kappa machinery, so COVID-era quarters would be specification
error, not signal. (The workbook's data itself is today's vintage --
current data revisions are a known, documented source of small
discrepancy vs published estimates, same as the US fixture.)

Requires openpyxl (pandas Excel engine). Run from the repo root:
    python examples/ea_lw_sv/make_data.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = REPO / "tests" / "fixtures" / "hlw" / "data" / "Holston_Laubach_Williams_current_estimates.xlsx"


def main() -> None:
    raw = pd.read_excel(SOURCE, sheet_name="EA input data", header=0)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[raw["date"] <= "2019-12-31"].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "date": raw["date"].dt.strftime("%Y-%m-%d"),
            "lgdp100": 100.0 * raw["gdp.log"],
            "core_infl_ann": raw["inflation"],
            "real_rate": raw["interest"] - raw["inflation.expectations"],
        }
    )
    target = HERE / "data" / "ea_quarterly.csv"
    target.parent.mkdir(exist_ok=True)
    out.to_csv(target, index=False)
    print(f"Wrote {target} ({len(out)} rows, {out['date'].iloc[0]}..{out['date'].iloc[-1]})")


if __name__ == "__main__":
    main()
