#!/usr/bin/env python
"""Resumable driver for the PRE-REGISTERED uc_gap_sv SBC design (S8 UC-gap
SBC execution, DECISIONS.md 2026-09-04) -- cloned from
``scripts/run_ucsv_sbc.py``, the same generic engine, with crash-resume
through ``tests/artifacts/uc_gap_sbc/ranks.csv`` (each replication is
fully determined by seed_base + i, so a resumed run is byte-identical to
an uninterrupted one; this environment's container restarts
unpredictably).

    python scripts/run_uc_gap_sbc.py            # run/resume all 100 replications
    python scripts/run_uc_gap_sbc.py --smoke 3  # timing smoke: the first 3 reps only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples" / "notebook_api"))

from uc_gap_sv_validation import (  # noqa: E402
    ARTIFACT_DIR,
    SBC_CHI2_P_FLOOR,
    SBC_DIVERGENT_TOTAL_LIMIT,
    UC_GAP_SV_SBC_DESIGN,
    render_uc_gap_sv_model,
)
from macrotoolkit.validation.sbc import chi2_pvalues, run_sbc  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=int, default=None, metavar="N", help="run only the first N replications")
    args = parser.parse_args()
    t0 = time.time()
    # model=... bypasses render_design_model (the generic engine's compile
    # step, which does not fit the authored family's render-context shape
    # -- see render_uc_gap_sv_model's docstring); the SAME "rendered prior
    # == sampled prior" assertion still runs, just against the compiled
    # model's own resolved priors.
    model = render_uc_gap_sv_model()
    res = run_sbc(UC_GAP_SV_SBC_DESIGN, ARTIFACT_DIR, model=model, n_replications=args.smoke, progress_every=2)
    n = res.ranks.shape[0]
    elapsed = time.time() - t0
    done_now = n - res.resumed_from
    print(f"{n} replications banked ({res.resumed_from} resumed); {elapsed:.0f}s for {done_now} new "
          f"({elapsed / max(done_now, 1):.1f}s/rep); divergences total {sum(res.divergences)} "
          f"(ceiling {SBC_DIVERGENT_TOTAL_LIMIT})")
    if args.smoke is None:
        pvals = chi2_pvalues(UC_GAP_SV_SBC_DESIGN, res.ranks)
        for k, v in pvals.items():
            print(f"  chi2 p[{k}] = {v:.4f}{'  <-- below floor' if v < SBC_CHI2_P_FLOOR else ''}")


if __name__ == "__main__":
    main()
