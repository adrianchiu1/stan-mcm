"""`mtk` command-line interface (click)."""
from __future__ import annotations

import sys
from pathlib import Path

import click

from macrotoolkit import report as report_module
from macrotoolkit.run import REPO_ROOT
from macrotoolkit.run import run as execute_run


@click.group()
def main() -> None:
    """macrotoolkit CLI."""


@main.command("run")
@click.argument("spec_path", type=click.Path(exists=True, dir_okay=False))
def run_cmd(spec_path: str) -> None:
    """Run the model spec at SPEC_PATH, writing (or reusing) an immutable
    run directory under runs/<hash12>/."""
    try:
        result = execute_run(spec_path)
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)
    if result.is_new:
        click.echo(f"Run {result.run_id} complete (verdict: {result.verdict}) -> {result.run_dir}")
    else:
        click.echo(f"Run {result.run_id} already exists and is complete (idempotent no-op) -> {result.run_dir}")


@main.command("report")
@click.argument("run_hash")
@click.option(
    "--runs-root",
    type=click.Path(file_okay=False),
    default=None,
    help="Override the runs/ root directory (defaults to REPO_ROOT/runs, matching `mtk run`'s own convention).",
)
def report_cmd(run_hash: str, runs_root: str | None) -> None:
    """Render report.html for the completed run RUN_HASH, writing it into
    that run's own runs/<hash>/ directory (lw-sv-spec.md §3.5)."""
    roots = Path(runs_root).resolve() if runs_root is not None else REPO_ROOT / "runs"
    run_dir = roots / run_hash
    if not run_dir.is_dir():
        click.echo(
            f"error: run directory {run_dir} does not exist -- run `mtk run "
            f"<spec.yaml>` first, or pass the correct run hash (a 12-character "
            f"runs/<hash12>/ directory name) / --runs-root.",
            err=True,
        )
        sys.exit(1)
    if not (run_dir / "_SUCCESS").is_file():
        click.echo(
            f"error: run {run_hash} at {run_dir} is not complete (no "
            f"_SUCCESS marker) -- it may have crashed or been interrupted "
            f"partway through. Inspect {run_dir / 'log.txt'} and re-run "
            f"`mtk run` if needed; macrotoolkit never reports on an "
            f"incomplete run.",
            err=True,
        )
        sys.exit(1)

    try:
        out_path = report_module.write_report(run_dir)
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)
    click.echo(f"Report for run {run_hash} written -> {out_path}")


if __name__ == "__main__":
    main()
