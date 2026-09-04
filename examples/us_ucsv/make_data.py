"""Derive examples/us_ucsv/data/us_core_pce.csv -- US quarterly core PCE
inflation, annualized q/q percent (400 * dlog P, lw-sv-spec.md §1.1's
inflation convention) -- from data already in the repo: the NY Fed
HLW current-estimates workbook's "US input data" sheet
(tests/fixtures/hlw/data/Holston_Laubach_Williams_current_estimates.xlsx,
see FIXTURES.md), whose `inflation` column is exactly the core PCE
series examples/us_lw_sv uses as `core_pce_ann`. No new fetch.

The UCSV family (S6 WP2) has no pre-sample lag convention, so the file
starts at the workbook's first quarter (1960Q1) and runs through its
last (2026Q1, COVID quarters included -- SV on both shocks is the point).
`spec.yaml` estimates 1960Q1-2019Q4 (the pre-COVID reference window);
`spec_full_vintage.yaml` the full file.

Run from the repo root:  python examples/us_ucsv/make_data.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKBOOK = REPO / "tests" / "fixtures" / "hlw" / "data" / "Holston_Laubach_Williams_current_estimates.xlsx"


def main() -> None:
    full = pd.read_excel(WORKBOOK, sheet_name="US input data", header=0)
    full["date"] = pd.to_datetime(full["date"])
    out = pd.DataFrame(
        {
            "date": full["date"].dt.strftime("%Y-%m-%d"),
            "core_pce_ann": full["inflation"],
        }
    )
    target = HERE / "data" / "us_core_pce.csv"
    target.parent.mkdir(exist_ok=True)
    out.to_csv(target, index=False)
    print(f"Wrote {target} ({len(out)} rows, {out['date'].iloc[0]}..{out['date'].iloc[-1]})")


if __name__ == "__main__":
    main()
