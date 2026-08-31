"""Derive examples/us_lw_sv/data/us_quarterly.csv from the HLW fixture data
already in the repo (plans/S2-plan.md: "reuses data already in hand -- no
new fetch needed").

Source: tests/fixtures/hlw/derived/us_2017_reproduction/inputData/
rstar.data.us.csv -- itself the "US input data" sheet of the NY Fed's
current-estimates workbook (see that fixture's README), trimmed to
1960Q1-2019Q2. Transforms applied here, per lw-sv-spec.md §1.1 and locked
decision 3:

- lgdp100      = 100 * gdp.log   (the raw column is un-scaled ln(GDP);
                                  spec wants y = 100 x ln(real GDP))
- core_pce_ann = inflation       (already annualized q/q %, spec-conform)
- real_rate    = interest - inflation.expectations
                                 (the pre-constructed ex-ante real rate the
                                  spec requires as input; the toolkit never
                                  builds it itself in v1)

The window deliberately starts 1960Q1 -- four lag quarters before the
1961Q1 estimation start (see build_stan_data's lw_sv convention).

Run from the repo root:  python examples/us_lw_sv/make_data.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = REPO / "tests" / "fixtures" / "hlw" / "derived" / "us_2017_reproduction" / "inputData" / "rstar.data.us.csv"


WORKBOOK = REPO / "tests" / "fixtures" / "hlw" / "data" / "Holston_Laubach_Williams_current_estimates.xlsx"


def main() -> None:
    raw = pd.read_csv(SOURCE)
    dates = pd.period_range("1960Q1", periods=len(raw), freq="Q")
    out = pd.DataFrame(
        {
            "date": dates.to_timestamp(how="start").strftime("%Y-%m-%d"),
            "lgdp100": 100.0 * raw["gdp.log"],
            "core_pce_ann": raw["inflation"],
            "real_rate": raw["interest"] - raw["inflation.expectations"],
        }
    )
    target = HERE / "data" / "us_quarterly.csv"
    target.parent.mkdir(exist_ok=True)
    out.to_csv(target, index=False)
    print(f"Wrote {target} ({len(out)} rows, {out['date'].iloc[0]}..{out['date'].iloc[-1]})")

    # Full latest-vintage file (through 2026Q1, COVID quarters INCLUDED) from
    # the workbook's "US input data" sheet directly. Used by
    # spec_full_vintage.yaml -- an experiment, not the reference example: the
    # S2 model has none of HLW's 2023 COVID machinery (covid.ind, kappa
    # variance scaling), so estimates over this window absorb 2020 into the
    # constant shock scales. Requires openpyxl.
    full = pd.read_excel(WORKBOOK, sheet_name="US input data", header=0)
    full["date"] = pd.to_datetime(full["date"])
    out_full = pd.DataFrame(
        {
            "date": full["date"].dt.strftime("%Y-%m-%d"),
            "lgdp100": 100.0 * full["gdp.log"],
            "core_pce_ann": full["inflation"],
            "real_rate": full["interest"] - full["inflation.expectations"],
        }
    )
    target_full = HERE / "data" / "us_quarterly_full.csv"
    out_full.to_csv(target_full, index=False)
    print(f"Wrote {target_full} ({len(out_full)} rows, {out_full['date'].iloc[0]}..{out_full['date'].iloc[-1]})")


if __name__ == "__main__":
    main()
