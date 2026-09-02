#!/usr/bin/env python
"""Archive a completed run as a TRACKED, draw-thinned development fixture
(S5-decisions item 16): ``runs-archive/<hash12>/`` mirrors the run dir's
artifacts with the posterior draws thinned ~x10, so a fresh container gets
working output-layer fixtures from plain git instead of paying 40-80
minutes of sampling first.

    python scripts/archive_run.py <hash12> [--thin 10] [--runs-root runs]

DEVELOPMENT FIXTURES ONLY -- regenerate the full runs for publication
numbers (see runs-archive/README.md). The archive keeps the run's ORIGINAL
hash directory name and its full-run diagnostics.json (the diagnostics of
the 6,000-draw run, not of the thinned subset -- clearly the more useful
record, and labeled as such in ARCHIVE_NOTE.md); only draws.nc is altered
(every k-th draw per chain, keeping the chain structure so ArviZ/report
machinery work unchanged).
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

COPIED_ARTIFACTS = ("spec.yaml", "outputs.yaml", "data.snapshot.csv", "diagnostics.json", "_SUCCESS")


def archive_run(run_hash: str, thin: int = 10, runs_root: Path | None = None) -> Path:
    import arviz as az

    runs_root = runs_root or REPO_ROOT / "runs"
    run_dir = runs_root / run_hash
    if not (run_dir / "_SUCCESS").is_file():
        raise SystemExit(f"error: {run_dir} is not a completed run directory (_SUCCESS missing).")

    out_dir = REPO_ROOT / "runs-archive" / run_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in COPIED_ARTIFACTS:
        src = run_dir / name
        if src.is_file():
            shutil.copy2(src, out_dir / name)

    idata = az.from_netcdf(str(run_dir / "draws.nc"))
    thinned = idata.isel(draw=slice(0, None, thin))
    thinned.to_netcdf(str(out_dir / "draws.nc"))

    n_full = idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
    n_thin = thinned.posterior.sizes["chain"] * thinned.posterior.sizes["draw"]
    (out_dir / "ARCHIVE_NOTE.md").write_text(
        f"# DEVELOPMENT FIXTURE -- draws thinned x{thin}\n\n"
        f"Archived {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from "
        f"`runs/{run_hash}` (S5-decisions item 16).\n\n"
        f"- `draws.nc` keeps every {thin}th posterior draw per chain: "
        f"{n_full} -> {n_thin} total draws. Posterior summaries and figures\n"
        f"  from this archive are development-quality ONLY -- **regenerate the\n"
        f"  full run (`mtk run` on the spec below) for publication numbers.**\n"
        f"- `diagnostics.json` is the FULL run's diagnostics record (verdict,\n"
        f"  R-hat/ESS, divergences of the un-thinned run), copied verbatim.\n"
        f"- `spec.yaml`/`outputs.yaml`/`data.snapshot.csv` are byte-identical\n"
        f"  copies; re-running the spec reproduces the full run at this same\n"
        f"  hash (estimation identity is unchanged by archiving).\n"
    )
    print(f"archived {run_dir} -> {out_dir} ({n_full} -> {n_thin} draws)")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_hash")
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--runs-root", type=Path, default=None)
    args = parser.parse_args()
    archive_run(args.run_hash, thin=args.thin, runs_root=args.runs_root)


if __name__ == "__main__":
    main()
