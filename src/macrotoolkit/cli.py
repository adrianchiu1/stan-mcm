"""`mtk` command-line interface (click)."""
from __future__ import annotations

import sys

import click

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
def report_cmd(run_hash: str) -> None:
    """Render an HTML report for a completed run. Not implemented until S4
    (lw-sv-spec.md §3.5, §7)."""
    raise NotImplementedError(
        "`mtk report` is S4 scope (HTML report assembly, lw-sv-spec.md §3.5) "
        "and is not implemented yet."
    )


if __name__ == "__main__":
    main()
