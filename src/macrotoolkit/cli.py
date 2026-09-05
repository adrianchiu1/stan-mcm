"""`mtk` command-line interface (click) -- a THIN SHELL over
:mod:`macrotoolkit.api` (S6 WP1): every command parses its arguments,
calls the corresponding API function, and formats the result as text.
No pipeline logic lives here."""
from __future__ import annotations

import sys
from pathlib import Path

import click

from macrotoolkit.run import REPO_ROOT


@click.group()
def main() -> None:
    """macrotoolkit CLI."""


@main.command("run")
@click.argument("spec_path", type=click.Path(exists=True, dir_okay=False))
def run_cmd(spec_path: str) -> None:
    """Run the model spec at SPEC_PATH, writing (or reusing) an immutable
    run directory under runs/<hash12>/."""
    from macrotoolkit import api

    try:
        result = api.fit(spec_path)
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)
    if result.is_new:
        click.echo(f"Run {result.hash} complete (verdict: {result.verdict}) -> {result.run_dir}")
    else:
        click.echo(f"Run {result.hash} already exists and is complete (idempotent no-op) -> {result.run_dir}")


@main.command("sweep")
@click.argument("sweep_path", type=click.Path(exists=True, dir_okay=False))
def sweep_cmd(sweep_path: str) -> None:
    """Run the prior sweep described by SWEEP_PATH (base spec + prior
    override grid, S5-decisions item 9): every cell is an ordinary
    immutable run in runs/ (idempotent cell-by-cell), and one comparison
    report -- posteriors, prior-to-posterior contraction, headline series
    across the grid -- is written under sweeps/<name>/."""
    from macrotoolkit import api

    try:
        result = api.sweep(sweep_path)
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)
    for cell in result.cells:
        state = "new" if cell.run_result.is_new else "existing"
        click.echo(f"  cell {cell.label}: run {cell.run_result.run_id} ({state}, verdict: {cell.verdict})")
    click.echo(f"Sweep {result.name} complete -> {result.report_path}")


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

    from macrotoolkit import api

    try:
        out_path = api.load_run(run_dir).report()
    except NotImplementedError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)
    click.echo(f"Report for run {run_hash} written -> {out_path}")


@main.command("validate")
@click.argument("family")
@click.option(
    "--tier",
    type=click.Choice(["fast", "recovery", "sbc", "all"]),
    default="fast",
    show_default=True,
    help="Which registered gate tier(s) to run.",
)
@click.option(
    "--out-root",
    type=click.Path(file_okay=False),
    default=None,
    help="Where to write validation/<family>/ (defaults to REPO_ROOT/validation).",
)
def validate_cmd(family: str, tier: str, out_root: str | None) -> None:
    """Run FAMILY's registered validation gates (S6 WP3) and write one
    validation report summarizing PASS/WARN/FAIL per gate. FAMILY may
    also be the path of an AUTHORED model's spec YAML (S7), whose fast
    tier is auto-instantiated from the equations."""
    from macrotoolkit import api

    try:
        result = api.validate(family, tier=tier, out_root=out_root)
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)
    for g in result.gates:
        click.echo(f"  {g.name} [{g.tier}]: {g.verdict} -- {g.summary}")
    click.echo(f"Validation of {family} ({tier}): {result.verdict} -> {result.report_path}")
    if result.verdict == "FAIL":
        sys.exit(2)


if __name__ == "__main__":
    main()
