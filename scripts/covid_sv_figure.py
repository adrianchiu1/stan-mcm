#!/usr/bin/env python
"""Generate the COVID/SV payoff exhibit for the README (the S3 result the
examples README records: run the FULL vintage through 2026Q1 with NO
hand-set COVID machinery, and the SV block absorbs 2020 endogenously --
exp(h_IS/2) spikes in exactly 2020's quarters and decays back, the
Bayesian counterpart of HLW's hand-set kappa variance scaling).

    python scripts/covid_sv_figure.py [run_hash]

Reads the volatility paths directly off the run's saved h_is/h_pc
transformed parameters (never re-derived from nu -- HANDOFF's S4 warning;
h is log-VARIANCE, sd = exp(h/2)). Writes
docs/exhibits/covid_sv_volatility.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HASH = "eb73e644be0b"  # the regenerated full-vintage reference run


def main() -> None:
    from macrotoolkit.results_lw import load_lw_run

    run_hash = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HASH
    run_dir = REPO_ROOT / "runs" / run_hash
    if not run_dir.is_dir():
        run_dir = REPO_ROOT / "runs-archive" / run_hash
    lw_run = load_lw_run(run_dir)
    if not lw_run.sv_on:
        raise SystemExit(f"error: run {run_hash} is not an SV run.")

    post = lw_run.idata.posterior
    vol = {}
    for name in ("h_is", "h_pc"):
        h = np.asarray(post[name].values)
        h = h.reshape((-1, h.shape[-1]))  # (n_draws, T)
        # h is log-VARIANCE: sd = exp(h/2), never exp(h).
        sd = np.exp(h / 2.0)
        vol[name] = {
            "median": np.median(sd, axis=0),
            "lo68": np.percentile(sd, 16.0, axis=0),
            "hi68": np.percentile(sd, 84.0, axis=0),
            "lo90": np.percentile(sd, 5.0, axis=0),
            "hi90": np.percentile(sd, 95.0, axis=0),
        }

    dates = lw_run.dates
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for ax, (name, label, color) in zip(
        axes,
        (("h_is", "IS/demand shock sd exp(h_IS/2)", "tab:red"),
         ("h_pc", "Phillips/supply shock sd exp(h_PC/2)", "tab:purple")),
    ):
        v = vol[name]
        ax.fill_between(dates, v["lo90"], v["hi90"], color=color, alpha=0.15, linewidth=0, label="90% CI")
        ax.fill_between(dates, v["lo68"], v["hi68"], color=color, alpha=0.30, linewidth=0, label="68% CI")
        ax.plot(dates, v["median"], color=color, linewidth=1.6, label="posterior median")
        ax.set_title(label)
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Date")
    fig.suptitle(
        f"Stochastic volatility absorbs COVID endogenously (run {run_hash}, full vintage,\n"
        "NO hand-set COVID machinery: the 2020 spike is discovered from the data, not imposed)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = REPO_ROOT / "docs" / "exhibits" / "covid_sv_volatility.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
