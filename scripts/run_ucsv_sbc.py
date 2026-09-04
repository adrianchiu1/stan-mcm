#!/usr/bin/env python
"""Resumable driver for the PRE-REGISTERED UCSV SBC design (S6 WP2c,
DECISIONS.md 2026-09-04) -- the same declaration `mtk validate ucsv
--tier sbc` runs, with crash-resume through
tests/artifacts/ucsv_sbc/ranks.csv (each replication is fully determined
by seed_base + i, so a resumed run is byte-identical to an uninterrupted
one; this environment's container restarts unpredictably).

    python scripts/run_ucsv_sbc.py            # run/resume all 100 replications
    python scripts/run_ucsv_sbc.py --smoke 3  # timing smoke: the first 3 reps only

The gate assertions live in tests/test_ucsv_gates.py's slow test, which
reloads a completed ranks.csv without re-fitting.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from macrotoolkit.families.ucsv_validation import (  # noqa: E402
    SBC_CHI2_P_FLOOR,
    SBC_DIVERGENT_TOTAL_LIMIT,
    UCSV_SBC_DESIGN,
)
from macrotoolkit.validation.sbc import chi2_pvalues, run_sbc  # noqa: E402

ARTIFACT_DIR = REPO_ROOT / "tests" / "artifacts" / "ucsv_sbc"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=int, default=None, metavar="N", help="run only the first N replications")
    args = parser.parse_args()
    t0 = time.time()
    res = run_sbc(UCSV_SBC_DESIGN, ARTIFACT_DIR, n_replications=args.smoke, progress_every=2)
    n = res.ranks.shape[0]
    elapsed = time.time() - t0
    done_now = n - res.resumed_from
    print(f"{n} replications banked ({res.resumed_from} resumed); {elapsed:.0f}s for {done_now} new "
          f"({elapsed / max(done_now, 1):.1f}s/rep); divergences total {sum(res.divergences)} "
          f"(ceiling {SBC_DIVERGENT_TOTAL_LIMIT})")
    if args.smoke is None:
        pvals = chi2_pvalues(UCSV_SBC_DESIGN, res.ranks)
        for k, v in pvals.items():
            print(f"  chi2 p[{k}] = {v:.4f}{'  <-- below floor' if v < SBC_CHI2_P_FLOOR else ''}")


if __name__ == "__main__":
    main()
