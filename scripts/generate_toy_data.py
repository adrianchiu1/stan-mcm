"""One-off generator for `examples/toy/data.csv`.

Not part of the runtime path -- run manually once, and the resulting CSV is
committed as a static file. Runs must be reproducible from that static
file (the run-identity hash is over the data file's bytes), so data
generation is deliberately kept out of `mtk run`'s code path.

Simulates a local-level model:
    y_t   = mu_t + eps_t,      eps_t ~ N(0, sigma_obs^2)
    mu_t  = mu_{t-1} + eta_t,  eta_t ~ N(0, sigma_level^2)
at known sigma_obs / sigma_level, ~100 quarters, fixed seed 20260813, so the
toy model in examples/toy/spec.yaml has a clean, well-identified
parameter-recovery problem.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260813
N_QUARTERS = 100
SIGMA_OBS = 0.4
SIGMA_LEVEL = 0.3
MU0 = 2.0  # arbitrary starting level
START_DATE = "1998-01-01"

OUT_PATH = Path(__file__).resolve().parents[1] / "examples" / "toy" / "data.csv"


def main() -> None:
    rng = np.random.default_rng(SEED)

    eta = rng.normal(0.0, SIGMA_LEVEL, size=N_QUARTERS)
    mu = np.empty(N_QUARTERS)
    mu[0] = MU0
    for t in range(1, N_QUARTERS):
        mu[t] = mu[t - 1] + eta[t]

    eps = rng.normal(0.0, SIGMA_OBS, size=N_QUARTERS)
    y = mu + eps

    dates = pd.date_range(start=START_DATE, periods=N_QUARTERS, freq="QS")

    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "obs": y})
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")
    print(f"True sigma_obs={SIGMA_OBS}, sigma_level={SIGMA_LEVEL}, seed={SEED}")


if __name__ == "__main__":
    main()
