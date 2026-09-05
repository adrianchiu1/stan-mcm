"""Self-contained HTML report assembly for a completed run
(lw-sv-spec.md §3.5): "One self-contained ``report.html`` per run: header
(run hash, spec summary, data span), diagnostics verdict, then §3.1-§3.4
figures, then a parameter table (posterior median, 90% CI, R-hat, ESS)."

FAMILY-GENERIC since S6 WP1: this module owns NO model numerics and no
family knowledge. It loads the run through the family registry
(``results_loader``) and renders the family's declared OUTPUT MODULES
(``output_modules``, :mod:`macrotoolkit.outputs`) in declared order, each
module supplying its own compute/plot callables, heading, figure title and
caption. lw_sv's declaration (:mod:`macrotoolkit.outputs_lw`) reproduces
the S4/S5 report layout exactly; a new family gets a report by declaring
its modules, never by editing this file.

Every ``Figure`` is turned into an embeddable ``data:image/png;base64,...``
URI (``plots.py``'s own docstring: figures, not encoded strings, are its
business; embedding is this module's). Numerical conventions (g
annualized, h log-variance, inflation 400*dlog P) are enforced upstream by
the families' own results/plots modules and their tests; this module only
displays.

Two public entry points:

- :func:`render_report` -- ``(run_dir) -> str``, pure (no filesystem write).
- :func:`write_report` -- ``(run_dir) -> Path``, writes
  ``run_dir / "report.html"``. ``mtk report <hash>`` / ``Run.report()``
  call this one.

Writing ``report.html`` into an already-``_SUCCESS``-marked run directory
is NOT a violation of ``run.py``'s "immutable once written" doctrine: that
doctrine covers the *sampling artifacts*, which this module only ever
READS. ``report.html`` is a derived, regenerable view.

The results object a family's ``results_loader`` returns must expose
``spec`` (the run's RunSpec), ``dates`` (the estimation-sample
DatetimeIndex) and ``idata`` (the ArviZ posterior) -- the three things the
generic header and parameter table read.
"""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from specs.schema import get_family

from macrotoolkit.outputs import OutputModule
from macrotoolkit.run import load_run_spec

#: Verdict -> (background color, border color) -- plain CSS, no framework,
#: matching this codebase's "no silent fallbacks / scannable" conventions
#: (PASS green, WARN amber, FAIL red).
_VERDICT_COLORS: dict[str, tuple[str, str]] = {
    "PASS": ("#e6f4ea", "#1a7f37"),
    "WARN": ("#fff8e6", "#9a6700"),
    "FAIL": ("#fde8e8", "#cf222e"),
}

_CSS = """
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 2rem auto; max-width: 1400px; color: #1f2328; line-height: 1.45; }
h1 { border-bottom: 2px solid #d0d7de; padding-bottom: 0.4rem; }
h2 { margin-top: 2.5rem; border-bottom: 1px solid #d0d7de; padding-bottom: 0.3rem; }
h3 { margin-top: 1.5rem; }
table { border-collapse: collapse; margin: 0.75rem 0 1.5rem 0; }
th, td { border: 1px solid #d0d7de; padding: 0.35rem 0.75rem; text-align: left; font-variant-numeric: tabular-nums; }
th { background: #f6f8fa; }
.verdict-box { border: 2px solid; border-radius: 6px; padding: 1rem 1.25rem; margin: 0.75rem 0 1.5rem 0; }
.verdict-label { font-size: 1.4rem; font-weight: 700; }
.reasons { margin-top: 0.5rem; }
.caption { color: #57606a; font-size: 0.9rem; margin: 0.25rem 0 1.25rem 0; }
figure { margin: 0 0 2rem 0; }
figure img { max-width: 100%; height: auto; border: 1px solid #d0d7de; border-radius: 4px; }
.meta-table td:first-child, .meta-table th:first-child { font-weight: 600; white-space: nowrap; }
.fig-grid { display: flex; flex-wrap: wrap; gap: 1.5rem; }
.fig-grid figure { flex: 1 1 560px; max-width: 700px; }
"""


def _esc(value: object) -> str:
    """HTML-escape any value's ``str()`` (defensive -- spec fields / file
    paths could in principle contain HTML-meaningful characters)."""
    return html.escape(str(value))


def _fig_to_data_uri(fig, *, dpi: int = 100) -> str:
    """Encode a matplotlib ``Figure`` as a self-contained
    ``data:image/png;base64,...`` URI (spec §3.5: "self-contained" means no
    external file references) and close the figure (this module's
    responsibility per ``plots.py``'s own docstring), so repeated report
    generation in one process does not accumulate open figures."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _figure_block(title: str, uri: str, caption: str | None = None) -> str:
    cap_html = f'<figcaption class="caption">{_esc(caption)}</figcaption>' if caption else ""
    return (
        f'<figure><figcaption><strong>{_esc(title)}</strong></figcaption>'
        f'<img src="{uri}" alt="{_esc(title)}">{cap_html}</figure>'
    )


def verdict_box_html(verdict: str, reasons: list[str], extra_rows: list[tuple[str, object]] | None = None) -> str:
    """A PASS/WARN/FAIL verdict box (shared with the validation report,
    S6 WP3, so gate verdicts read exactly like run diagnostics)."""
    bg, border = _VERDICT_COLORS.get(verdict, ("#f6f8fa", "#57606a"))
    reasons_html = "".join(f"<li>{_esc(r)}</li>" for r in reasons)
    rows_html = ""
    if extra_rows:
        rows_html = "<table class=\"meta-table\">" + "\n".join(
            f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in extra_rows
        ) + "</table>\n"
    return (
        f'<div class="verdict-box" style="background:{bg};border-color:{border};">\n'
        f'<span class="verdict-label" style="color:{border};">Verdict: {_esc(verdict)}</span>\n'
        f'<div class="reasons"><ul>{reasons_html}</ul></div>\n'
        f"{rows_html}"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Header + diagnostics
# ---------------------------------------------------------------------------


def _family_title(family_name: str) -> str:
    entry = get_family(family_name)
    return entry.display_name or family_name


def _render_header(results: Any, run_hash: str) -> str:
    spec = results.spec
    options = spec.model.options
    sampler = spec.sampler
    dates = results.dates
    start_q = str(dates[0].to_period("Q"))
    end_q = str(dates[-1].to_period("Q"))
    mapping_str = ", ".join(f"{k} = {v}" for k, v in spec.data.mapping.items())

    rows: list[tuple[str, object]] = [
        ("Run hash", run_hash),
        ("Model family", spec.model.family),
    ]
    option_items = options.model_dump() if hasattr(options, "model_dump") else dict(options or {})
    for key, value in option_items.items():
        if key == "family":
            continue
        shown = value if value not in ([], None, {}) else f"{value!r} (default/none)"
        rows.append((key, shown))
    rows += [
        ("Data file", spec.data.file),
        ("Data mapping", mapping_str),
        ("Data span (estimation sample)", f"{start_q} -- {end_q} ({len(dates)} quarters)"),
        ("Sampler: chains", sampler.chains),
        ("Sampler: warmup", sampler.warmup),
        ("Sampler: sampling", sampler.sampling),
        ("Sampler: adapt_delta", sampler.adapt_delta),
        ("Sampler: max_treedepth", sampler.max_treedepth),
        ("Sampler: seed", sampler.seed),
    ]
    body = "\n".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in rows)
    return (
        f"<h1>{_esc(_family_title(spec.model.family))} report -- run {_esc(run_hash)}</h1>\n"
        f'<table class="meta-table">{body}</table>'
    )


def _mirror_check_rows(diagnostics: dict) -> list[tuple[str, object]]:
    """The automatic Stan-vs-Python KF mirror check (S6 WP3) as header
    rows -- shown whenever the run record carries one."""
    mc = diagnostics.get("mirror_check")
    if not mc:
        return [("KF mirror check (Stan vs Python)", "not recorded (pre-S6 run or qc.mirror_check: false)")]
    status = "PASS" if mc.get("passed") else "FAIL"
    return [
        (
            "KF mirror check (Stan vs Python)",
            f"{status}: max |diff| = {mc.get('max_abs_diff'):.3e} over {mc.get('n_points')} prior "
            f"draw(s), gate {mc.get('tolerance'):.1e}",
        )
    ]


def _render_diagnostics(diagnostics: dict) -> str:
    ebfmi = ", ".join(f"{x:.3f}" for x in diagnostics["e_bfmi_per_chain"])
    rows: list[tuple[str, object]] = [
        ("Divergences", diagnostics["divergences"]),
        (
            "Max-treedepth hits",
            f"{diagnostics['max_treedepth_hits']} (config max_treedepth={diagnostics['max_treedepth_config']})",
        ),
        ("E-BFMI per chain", ebfmi),
        ("Max R-hat (across parameters)", f"{diagnostics['rhat_max']:.4f}"),
        ("Min bulk ESS (across parameters)", f"{diagnostics['ess_bulk_min']:.0f}"),
        ("Min tail ESS (across parameters)", f"{diagnostics['ess_tail_min']:.0f}"),
    ] + _mirror_check_rows(diagnostics)
    return "<h2>Diagnostics</h2>\n" + verdict_box_html(
        diagnostics["verdict"], diagnostics["reasons"], rows
    )


# ---------------------------------------------------------------------------
# Output modules
# ---------------------------------------------------------------------------


def _render_module(module: OutputModule, results: Any) -> str:
    reason = module.unavailable_reason(results)
    if reason is not None:
        return (
            f"<h2>{_esc(module.heading)}</h2>\n"
            f'<p class="caption"><strong>Omitted:</strong> {_esc(reason)}</p>'
        )
    data = module.compute(results)
    figs = module.figures(data, results)
    caption = module.caption(data, results) if module.caption is not None else None
    if list(figs) == [module.name]:
        uri = _fig_to_data_uri(figs[module.name], dpi=module.dpi)
        block = _figure_block(module.figure_title, uri, caption)
        return f"<h2>{_esc(module.heading)}</h2>\n{block}"
    blocks = [
        _figure_block(module.figure_label(key), _fig_to_data_uri(fig, dpi=module.dpi))
        for key, fig in figs.items()
    ]
    grid = f'<div class="fig-grid">{"".join(blocks)}</div>'
    cap_html = f'<p class="caption">{_esc(caption)}</p>\n' if caption else ""
    return f"<h2>{_esc(module.heading)}</h2>\n{cap_html}{grid}"


# ---------------------------------------------------------------------------
# Parameter table
# ---------------------------------------------------------------------------


def _scalar_param_names(posterior) -> list[str]:
    """The TRUE scalar (time-invariant) parameters of this run's posterior --
    every ``data_vars`` entry whose only dims are ``('chain', 'draw')``,
    i.e. no extra per-index dimension. Determined by inspecting the actual
    posterior's own dims (never guessed/hardcoded): for an SV run this
    excludes the length-T vector parameters (h paths, nu innovations) and
    includes the scalar non-centered SV parameters; for a no-SV run every
    parameter is scalar. A v1 parameter table reports scalars only --
    per-element rows for a length-T vector parameter would make the table
    useless as a "read this at a glance" artifact.
    """
    names = []
    for name in sorted(posterior.data_vars):
        if set(posterior[name].dims) == {"chain", "draw"}:
            names.append(name)
    return names


def param_table_rows(idata) -> list[dict]:
    """Posterior median, 90% CI (5th/95th percentile), R-hat, bulk/tail ESS
    for every scalar parameter (:func:`_scalar_param_names`) -- computed
    directly from the ArviZ posterior via ``numpy``/``arviz``, not read off
    ``diagnostics.json`` (which stores a worst-case reduction over any
    vector dims). Shared by the report and ``Run.param_table()``."""
    import arviz as az

    posterior = idata.posterior
    names = _scalar_param_names(posterior)
    if not names:
        raise ValueError(
            f"Posterior has no scalar (chain,draw)-only variables -- cannot "
            f"build a parameter table. Available variables: {sorted(posterior.data_vars)}."
        )
    rhat_ds = az.rhat(idata, var_names=names)
    ess_bulk_ds = az.ess(idata, var_names=names, method="bulk")
    ess_tail_ds = az.ess(idata, var_names=names, method="tail")

    rows = []
    for name in names:
        arr = np.asarray(posterior[name].values, dtype=np.float64).reshape(-1)
        median = float(np.percentile(arr, 50.0))
        lo, hi = (float(v) for v in np.percentile(arr, [5.0, 95.0]))
        rows.append(
            {
                "name": name,
                "median": median,
                "ci_lo": lo,
                "ci_hi": hi,
                "rhat": float(np.asarray(rhat_ds[name].values)),
                "ess_bulk": float(np.asarray(ess_bulk_ds[name].values)),
                "ess_tail": float(np.asarray(ess_tail_ds[name].values)),
            }
        )
    return rows


def _param_table_rows(results: Any) -> list[dict]:
    """Backward-compatible alias over the results object."""
    return param_table_rows(results.idata)


def _render_param_table(results: Any) -> str:
    rows = param_table_rows(results.idata)
    header = (
        "<tr><th>Parameter</th><th>Median</th><th>90% CI low (5%)</th>"
        "<th>90% CI high (95%)</th><th>R-hat</th><th>ESS bulk</th><th>ESS tail</th></tr>"
    )
    body = "\n".join(
        f"<tr><td>{_esc(r['name'])}</td><td>{r['median']:.4g}</td>"
        f"<td>{r['ci_lo']:.4g}</td><td>{r['ci_hi']:.4g}</td>"
        f"<td>{r['rhat']:.4f}</td><td>{r['ess_bulk']:.0f}</td><td>{r['ess_tail']:.0f}</td></tr>"
        for r in rows
    )
    caption = (
        "Scalar (time-invariant) parameters only -- vector-valued path "
        "parameters (e.g. an SV run's h and nu vectors) are omitted from "
        "this table; see the trend-cycle volatility panel above for those."
    )
    return (
        "<h2>Parameter table</h2>\n"
        f'<p class="caption">{_esc(caption)}</p>\n'
        f"<table>{header}\n{body}</table>"
    )


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


def render_report(run_dir: str | Path) -> str:
    """Render a completed run's full ``report.html`` document (spec §3.5)
    as a string, for any family declaring ``results_loader`` +
    ``output_modules``. Pure with respect to ``run_dir``: reads its
    already-written artifacts but never writes anything -- see
    :func:`write_report` for the filesystem-writing wrapper."""
    run_dir = Path(run_dir)
    diagnostics_path = run_dir / "diagnostics.json"
    if not diagnostics_path.is_file():
        raise FileNotFoundError(
            f"{diagnostics_path} not found -- {run_dir} does not look like a "
            f"completed run directory (expected diagnostics.json alongside "
            f"spec.yaml/draws.nc). Run `mtk run <spec.yaml>` first."
        )
    diagnostics = json.loads(diagnostics_path.read_text())

    spec = load_run_spec(run_dir)
    entry = get_family(spec.model.family)
    loader = entry.resolve("results_loader")
    modules = entry.resolve("output_modules")
    if loader is None or modules is None:
        raise NotImplementedError(
            f"Family {spec.model.family!r} declares no results_loader/"
            f"output_modules capabilities in FAMILY_REGISTRY (specs/schema) "
            f"-- it has no report support yet."
        )
    results = loader(run_dir)
    run_hash = run_dir.name

    sections = [
        _render_header(results, run_hash),
        _render_diagnostics(diagnostics),
        *[_render_module(m, results) for m in modules],
        _render_param_table(results),
    ]
    body = "\n".join(sections)
    title = f"{_family_title(spec.model.family)} report -- {run_hash}"

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{_esc(title)}</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def write_report(run_dir: str | Path) -> Path:
    """Render (:func:`render_report`) and write ``run_dir / "report.html"``.
    Returns the written path. Safe to call on an already-``_SUCCESS``-marked
    run directory (see module docstring) -- overwrites any previous
    ``report.html`` in place, since a report is not itself part of run
    identity."""
    run_dir = Path(run_dir)
    html_text = render_report(run_dir)
    out_path = run_dir / "report.html"
    out_path.write_text(html_text, encoding="utf-8")
    return out_path
