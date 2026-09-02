#!/usr/bin/env python
"""Generate the G5a README exhibit figure (spec §5: "G5 is the external
credibility exhibit; its figure goes in the README"): our KF/RTS smoother,
fixed at the HLW (2017) oracle's MLE parameters and exact initial
conditions, overlaid on the oracle's own smoothed series -- with the
measured max |diff| stamped per panel (~1e-12, i.e. the two lines are one
line). Writes docs/exhibits/g5a_oracle_replication.png.

Reuses tests/test_g5a_hlw_replication.py's own fixture-loading and mapping
code verbatim (never a re-derivation), per the repo's reporting-mapping
doctrine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from test_g5a_hlw_replication import (  # noqa: E402
    FIXTURE_DIR,
    _our_params,
    _run_smoother,
    _series_from_states,
)


def _load_oracle() -> dict:
    params = (
        pd.read_csv(FIXTURE_DIR / "output" / "us_2017_parameters.csv")
        .set_index("parameter")["value"]
    )
    data = pd.read_csv(FIXTURE_DIR / "inputData" / "rstar.data.us.csv")
    return {
        "params": params,
        "y": 100.0 * data["gdp.log"].to_numpy(),
        "pi": data["inflation"].to_numpy(),
        "r": (data["interest"] - data["inflation.expectations"]).to_numpy(),
        "xi00_q": pd.read_csv(FIXTURE_DIR / "output" / "us_2017_xi00.csv")["xi00"].to_numpy(),
        "P00_q": pd.read_csv(FIXTURE_DIR / "output" / "us_2017_P00.csv").to_numpy(),
        "smoothed": pd.read_csv(FIXTURE_DIR / "output" / "us_2017_smoothed.csv"),
        "dates": pd.to_datetime(data["date"]) if "date" in data.columns else None,
    }


def main() -> None:
    o = _load_oracle()
    res, yobs = _run_smoother(o)
    ours = _series_from_states(res["xi_smooth"], yobs)
    oracle_s = o["smoothed"]
    T = len(ours)
    x = (
        o["dates"].to_numpy()[4 : 4 + T]
        if o["dates"] is not None
        else np.arange(T)
    )

    titles = {"rstar": "r*", "g": "g (annualized)", "z": "z", "output_gap": "output gap"}
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for ax, col in zip(axes, titles):
        diff = float(np.max(np.abs(ours[col].to_numpy() - oracle_s[col].to_numpy())))
        ax.plot(x, oracle_s[col], color="black", linewidth=2.2, label="HLW (2017) code, smoothed")
        ax.plot(x, ours[col], color="tab:orange", linewidth=0.9, linestyle="--",
                label="macrotoolkit KF+RTS, same parameters")
        ax.set_title(f"{titles[col]} -- max |diff| = {diff:.2e}")
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Date" if o["dates"] is not None else "Quarter")
    fig.suptitle(
        "G5a: exact HLW replication at fixed parameters (gate: <1e-8; observed ~1e-12)\n"
        "Two lines per panel -- they coincide to machine precision.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = REPO_ROOT / "docs" / "exhibits" / "g5a_oracle_replication.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
