"""`mtk sweep` -- the reusable prior-sensitivity sweep tool (S5-decisions
item 9): base spec + a grid of prior overrides -> batched runs in the
ORDINARY run store + one comparison report.

A sweep adds no new storage concept: every cell is a normal,
hash-identified, immutable run (each cell's derived spec = the base spec
with the cell's prior overrides merged over its ``priors:`` block), so
re-invoking a sweep is idempotent cell-by-cell for free (completed cells
are `run()`-level no-ops), any cell's full per-run report stays available
via `mtk report <hash>`, and interrupting a sweep loses nothing. The sweep
layer only orchestrates the batch and assembles the comparison report:

- a cell table (label, run hash, diagnostics verdict);
- per-parameter posterior summaries across cells (median, 90% CI,
  posterior sd), swept parameters listed first;
- the prior→posterior CONTRACTION readout, ``1 - (posterior sd / prior
  sd)^2`` per parameter per cell (prior sds Monte-Carlo'd from each
  cell's own resolved priors via the family's ``prior_sd_table``
  capability) -- the direct measure of how much identification work each
  prior is doing, the point of the mandated sigma_g/sigma_z sweep
  (spec §1.6);
- the family's headline smoothed series overlaid across cells (the
  ``headline_series`` registry capability; e.g. lw_sv's posterior-median
  r* and gap).

Output: ``sweeps/<name>/report.html`` + ``cells.json`` (label -> run hash
mapping, for traceability), gitignored like ``runs/``.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from specs.schema import get_family
from specs.schema.base import RunSpec
from specs.schema.sweep import SweepSpec, load_sweep_spec

from macrotoolkit.run import REPO_ROOT, RunResult, run

_CI_LO, _CI_HI = 5.0, 95.0


@dataclass
class SweepCellResult:
    label: str
    spec: RunSpec
    run_result: RunResult
    verdict: str


@dataclass
class SweepResult:
    name: str
    out_dir: Path
    report_path: Path
    cells: list[SweepCellResult]


def derive_cell_spec(base_raw: dict, cell_priors: dict) -> RunSpec:
    """The cell's derived RunSpec: the base spec's raw dict with the
    cell's prior overrides merged over its ``priors:`` block (cell entry
    replaces the base entry for that parameter wholesale; the merged block
    then goes through the family's ordinary default-resolution and
    variant-checked validation at render time)."""
    raw = copy.deepcopy(base_raw)
    base_priors = raw.get("priors") or {}
    raw["priors"] = {**base_priors, **copy.deepcopy(cell_priors)}
    return RunSpec.model_validate(raw)


def _posterior_scalar_stats(run_dir: Path) -> dict[str, dict[str, float]]:
    """Per-scalar-parameter posterior summaries from a completed run's
    draws.nc: median, 5/95 percentiles, sd -- flattened over chains."""
    import arviz as az

    idata = az.from_netcdf(str(run_dir / "draws.nc"))
    post = idata.posterior
    stats: dict[str, dict[str, float]] = {}
    for name in post.data_vars:
        da = post[name]
        if set(da.dims) != {"chain", "draw"}:
            continue  # vector parameters (SV paths) are not sweep-table rows
        flat = np.asarray(da.values).reshape(-1)
        stats[name] = {
            "median": float(np.median(flat)),
            "lo": float(np.percentile(flat, _CI_LO)),
            "hi": float(np.percentile(flat, _CI_HI)),
            "sd": float(np.std(flat)),
        }
    return stats


def contraction(prior_sd: float, posterior_sd: float) -> float:
    """Prior→posterior contraction, ``1 - (posterior sd / prior sd)^2``
    (Schad et al.'s posterior-contraction convention): 1 = the data
    determined the parameter completely; 0 = the posterior is as wide as
    the prior (the prior did all the work); negative = the posterior is
    WIDER than the prior."""
    if prior_sd <= 0.0:
        return float("nan")
    return 1.0 - (posterior_sd / prior_sd) ** 2


def run_sweep(
    sweep_path: str | Path,
    runs_root: str | Path | None = None,
    sweeps_root: str | Path | None = None,
) -> SweepResult:
    """Run every cell of the sweep at ``sweep_path`` (idempotently, via the
    ordinary run store) and write the comparison report. Returns a
    :class:`SweepResult`."""
    sweep_path = Path(sweep_path).resolve()
    sweep = load_sweep_spec(str(sweep_path))
    base_path = (sweep_path.parent / sweep.base_spec).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(
            f"Sweep base_spec {sweep.base_spec!r} -> {base_path} not found "
            f"(resolved relative to the sweep file's directory)."
        )
    base_raw = yaml.safe_load(base_path.read_text())
    if not isinstance(base_raw, dict):
        raise ValueError(f"Base spec {base_path} must contain a YAML mapping.")

    cells: list[SweepCellResult] = []
    for cell in sweep.cells:
        spec = derive_cell_spec(base_raw, cell.priors)
        result = run(base_path, runs_root=runs_root, spec_override=spec)
        diagnostics = json.loads((result.run_dir / "diagnostics.json").read_text())
        cells.append(
            SweepCellResult(
                label=cell.label,
                spec=spec,
                run_result=result,
                verdict=diagnostics["verdict"],
            )
        )

    out_root = Path(sweeps_root).resolve() if sweeps_root is not None else REPO_ROOT / "sweeps"
    out_dir = out_root / sweep.name
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "cells.json").write_text(
        json.dumps(
            {
                c.label: {
                    "run_id": c.run_result.run_id,
                    "run_dir": str(c.run_result.run_dir),
                    "verdict": c.verdict,
                    "priors_override": next(
                        sc.priors for sc in sweep.cells if sc.label == c.label
                    ),
                }
                for c in cells
            },
            indent=2,
            sort_keys=True,
        )
    )

    html = render_sweep_report(sweep, base_path, cells)
    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")

    return SweepResult(name=sweep.name, out_dir=out_dir, report_path=report_path, cells=cells)


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1100px; color: #222; }
table { border-collapse: collapse; margin: 0.8rem 0 1.6rem; font-size: 0.85rem; }
th, td { border: 1px solid #ccc; padding: 0.3rem 0.55rem; text-align: right; }
th { background: #f0f0f0; }
td.lbl, th.lbl { text-align: left; }
.swept { background: #fff7e0; }
.caption { color: #555; font-size: 0.85rem; }
img { max-width: 100%; }
h2 { margin-top: 2rem; }
"""


def _esc(value: object) -> str:
    import html

    return html.escape(str(value))


def _fig_to_data_uri(fig) -> str:
    import base64
    import io

    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _headline_overlay_figure(family_headline, cells: list[SweepCellResult]):
    """One overlay figure: each headline series as a panel, one line per
    cell. ``family_headline`` is the registry capability (or None)."""
    if family_headline is None:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_cell = {c.label: family_headline(c.run_result.run_dir) for c in cells}
    series_names = list(next(iter(per_cell.values())).keys())
    fig, axes = plt.subplots(
        len(series_names), 1, figsize=(10.0, 3.0 * len(series_names)), sharex=True, squeeze=False
    )
    for i, name in enumerate(series_names):
        ax = axes[i][0]
        for label, series in per_cell.items():
            dates, median = series[name]
            ax.plot(dates, median, linewidth=1.4, label=label)
        ax.set_title(f"Posterior-median {name} across sweep cells")
        ax.legend(loc="best", fontsize=8)
    axes[-1][0].set_xlabel("Date")
    fig.tight_layout()
    return fig


def render_sweep_report(sweep: SweepSpec, base_path: Path, cells: list[SweepCellResult]) -> str:
    """Assemble the self-contained comparison HTML (one file, embedded
    figure, zero external references -- same doctrine as the per-run
    report)."""
    family_name = cells[0].spec.model.family
    entry = get_family(family_name)
    prior_sd_fn = entry.resolve("prior_sd_table")
    headline_fn = entry.resolve("headline_series")

    swept_params: list[str] = []
    for cell in sweep.cells:
        for k in cell.priors:
            if k not in swept_params:
                swept_params.append(k)

    # Per-cell posterior stats + (per-cell!) prior sds -- each cell's
    # contraction uses that cell's own resolved priors, since the override
    # changes the prior it is measured against.
    post_stats = {c.label: _posterior_scalar_stats(c.run_result.run_dir) for c in cells}
    prior_sds: dict[str, dict[str, float]] = {}
    if prior_sd_fn is not None:
        from macrotoolkit.data import load_data

        for c in cells:
            df, _, _ = load_data(c.spec, base_path)
            prior_sds[c.label] = prior_sd_fn(c.spec, df)

    param_names = sorted({p for stats in post_stats.values() for p in stats})
    ordered = [p for p in swept_params if p in param_names] + [
        p for p in param_names if p not in swept_params
    ]

    # --- cell table ---
    rows = []
    for c in cells:
        overrides = next(sc.priors for sc in sweep.cells if sc.label == c.label)
        rows.append(
            f"<tr><td class='lbl'>{_esc(c.label)}</td>"
            f"<td class='lbl'><code>{_esc(c.run_result.run_id)}</code></td>"
            f"<td>{_esc(c.verdict)}</td>"
            f"<td class='lbl'><code>{_esc(json.dumps(overrides, sort_keys=True))}</code></td></tr>"
        )
    cell_table = (
        "<table><tr><th class='lbl'>Cell</th><th class='lbl'>Run</th>"
        "<th>Verdict</th><th class='lbl'>Prior overrides</th></tr>" + "".join(rows) + "</table>"
    )

    # --- posterior + contraction table ---
    header = "<tr><th class='lbl'>Parameter</th><th class='lbl'>Cell</th><th>Median</th><th>5%</th><th>95%</th><th>Posterior sd</th>"
    if prior_sds:
        header += "<th>Prior sd</th><th>Contraction</th>"
    header += "</tr>"
    prows = []
    for p in ordered:
        for c in cells:
            s = post_stats[c.label].get(p)
            if s is None:
                continue
            klass = " class='swept'" if p in swept_params else ""
            row = (
                f"<tr{klass}><td class='lbl'>{_esc(p)}</td><td class='lbl'>{_esc(c.label)}</td>"
                f"<td>{s['median']:.4g}</td><td>{s['lo']:.4g}</td><td>{s['hi']:.4g}</td>"
                f"<td>{s['sd']:.4g}</td>"
            )
            if prior_sds:
                prior_sd = prior_sds.get(c.label, {}).get(p)
                if prior_sd is None:
                    row += "<td>--</td><td>--</td>"
                else:
                    row += f"<td>{prior_sd:.4g}</td><td>{contraction(prior_sd, s['sd']):.3f}</td>"
            row += "</tr>"
            prows.append(row)
    param_table = f"<table>{header}{''.join(prows)}</table>"

    fig = _headline_overlay_figure(headline_fn, cells)
    overlay_html = (
        f"<img src='{_fig_to_data_uri(fig)}' alt='headline series overlay'>"
        if fig is not None
        else "<p class='caption'>(family declares no headline_series capability -- no overlay)</p>"
    )

    contraction_note = (
        "Contraction = 1 - (posterior sd / prior sd)^2, per cell against that cell's OWN resolved "
        "prior (prior sds Monte-Carlo'd through the family's prior sampler): 1 = data-determined, "
        "0 = prior-determined, negative = posterior wider than prior. Priors doing deliberate "
        "identification work (the sigma_g/sigma_z pile-up controls) show LOW contraction by design "
        "-- that is what this readout makes visible."
    )

    return (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n<meta charset='utf-8'>\n"
        f"<title>Sweep report -- {_esc(sweep.name)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n"
        f"<h1>Prior sweep: {_esc(sweep.name)}</h1>\n"
        f"<p class='caption'>Base spec: <code>{_esc(base_path)}</code> "
        f"(family <code>{_esc(family_name)}</code>); {len(cells)} cells, each an ordinary "
        f"immutable run -- per-cell full reports via <code>mtk report &lt;hash&gt;</code>.</p>\n"
        f"<h2>Cells</h2>\n{cell_table}\n"
        f"<h2>Posteriors and prior→posterior contraction</h2>\n"
        f"<p class='caption'>{_esc(contraction_note)}</p>\n{param_table}\n"
        f"<h2>Headline series across cells</h2>\n{overlay_html}\n"
        "</body>\n</html>\n"
    )
