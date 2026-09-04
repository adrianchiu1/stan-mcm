#!/usr/bin/env python
"""Resumable G4 driver (the pre-registered full-SV SBC gate's execution
vehicle -- DECISIONS.md 2026-09-02 pre-registration).

    python scripts/run_g4.py            # run/resume the full 100 replications
    python scripts/run_g4.py --smoke 3  # timing smoke: just the first 3 reps

Runs `tests/g4_harness.G4_DESIGN` through the generic engine with
crash-resume: ranks stream to tests/artifacts/g4_sbc/ranks.csv every 2
replications, and a relaunch continues from the first missing replication
(each rep is fully determined by seed_base + i, so a resumed run is
byte-identical to an uninterrupted one -- needed because this
environment's container restarts unpredictably). The smoke's reps ARE the
design's own first reps, so the full run resumes past them at no cost.

After the final replication this prints the chi^2 p-values and divergence
total; the actual GATE assertions live in tests/test_g4_sbc.py's slow
test, which (thanks to resume) reloads the completed ranks.csv without
re-fitting and applies the pre-registered accept/reject rule.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from g4_harness import G4_DESIGN  # noqa: E402
from sbc_harness import chi2_pvalues, run_sbc  # noqa: E402

ARTIFACT_DIR = REPO_ROOT / "tests" / "artifacts" / "g4_sbc"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=int, default=None, metavar="N",
                        help="run only the first N replications (timing smoke)")
    args = parser.parse_args()

    t0 = time.time()
    result = run_sbc(
        G4_DESIGN,
        ARTIFACT_DIR,
        n_replications=args.smoke,
        progress_every=1 if args.smoke else 2,
        resume=True,
    )
    elapsed = time.time() - t0
    n_done = result.ranks.shape[0]
    n_new = n_done - result.resumed_from
    print(f"replications complete: {n_done} (resumed {result.resumed_from}, ran {n_new} now)")
    if n_new > 0:
        per_rep = elapsed / n_new
        print(f"wall: {elapsed:.0f}s for {n_new} reps = {per_rep:.0f}s/rep "
              f"-> extrapolated full {G4_DESIGN.n_replications}-rep cost "
              f"{per_rep * G4_DESIGN.n_replications / 3600:.1f}h")
    print(f"total divergences so far: {sum(result.divergences)}")
    if args.smoke is None and n_done == G4_DESIGN.n_replications:
        pvals = chi2_pvalues(G4_DESIGN, result.ranks)
        print("chi^2 p-values:", {k: round(v, 4) for k, v in pvals.items()})


if __name__ == "__main__":
    main()
