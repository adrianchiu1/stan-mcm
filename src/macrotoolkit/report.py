"""Self-contained HTML report assembly for a completed ``lw_sv`` run
(lw-sv-spec.md §3.5): "One self-contained ``report.html`` per run: header
(run hash, spec summary, data span), diagnostics verdict, then §3.1-§3.4
figures, then a parameter table (posterior median, 90% CI, R-hat, ESS)."

This module owns NO model numerics of its own -- it only calls
``macrotoolkit.results_lw``'s four already-validated ``compute_*_draws``
entry points and ``macrotoolkit.plots``'s four plotting functions, then
turns their ``Figure`` outputs into embeddable ``data:image/png;base64,...``
URIs (``plots.py``'s own module docstring: figures, not encoded strings, are
its business; embedding is this module's). Nothing here re-derives or
re-checks the numerical conventions those upstream modules already enforce
and unit-test:

- ``g`` is ANNUALIZED everywhere (lw-sv-spec.md §1.1); the potential-output
  transition alone divides by 4.
- ``h`` is log-VARIANCE; the plotted/labeled volatility path is
  ``exp(h/2)`` (the standard deviation), never ``exp(h)``.
- Inflation is ``400 * dlog(P)``.

This module only DISPLAYS values already computed under those conventions
by ``macrotoolkit.results_lw``/``macrotoolkit.smoother``/``macrotoolkit.plots``
(each already covered by ``tests/test_units_conventions.py`` and their own
gate tests) -- so it needs no separate units unit test of its own; see
``tests/test_report.py`` for this module's own (presentation-level) coverage
instead.

Two public entry points:

- :func:`render_report` -- ``(run_dir) -> str``, pure (no filesystem write):
  loads the run + diagnostics, computes the four output modules, renders
  every figure to an embedded PNG, and returns the full HTML document as a
  string. Kept side-effect-free (besides reading ``run_dir``'s own already-
  written artifacts) so it is directly testable without needing to inspect
  a written file.
- :func:`write_report` -- ``(run_dir) -> Path``, the thin wrapper that calls
  :func:`render_report` and writes the result to ``run_dir / "report.html"``.
  ``cli.py``'s ``mtk report <hash>`` command calls this one.

Writing ``report.html`` into an already-``_SUCCESS``-marked run directory is
NOT a violation of ``run.py``'s "immutable once written" doctrine: that
doctrine covers the *sampling artifacts* (``spec.yaml``/``data.snapshot.csv``/
``draws.nc``/``diagnostics.json``), which this module only ever READS, never
writes or mutates. ``report.html`` is a derived, regenerable view, explicitly
listed as part of the run store's intended contents (lw-sv-spec.md §2.3) and
written by THIS command, never by ``mtk run``.
"""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import numpy as np

from macrotoolkit.results_lw import LWRun, load_lw_run

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
    responsibility per ``plots.py``'s own docstring: "the CALLER ... is
    responsible for saving/closing them"), so repeated report generation in
    one process does not accumulate open figures."""
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


# ---------------------------------------------------------------------------
# Header + diagnostics
# ---------------------------------------------------------------------------


def _render_header(lw_run: LWRun, run_hash: str) -> str:
    spec = lw_run.spec
    options = spec.model.options
    sampler = spec.sampler
    start_q = str(lw_run.dates[0].to_period("Q"))
    end_q = str(lw_run.dates[-1].to_period("Q"))
    mapping_str = ", ".join(f"{k} = {v}" for k, v in spec.data.mapping.items())

    rows = [
        ("Run hash", run_hash),
        ("Model family", spec.model.family),
        ("sv_shocks", options.sv_shocks or "[] (no SV)"),
        ("estimate_c", options.estimate_c),
        ("Data file", spec.data.file),
        ("Data mapping", mapping_str),
        ("Data span (estimation sample)", f"{start_q} -- {end_q} ({len(lw_run.dates)} quarters)"),
        ("Sampler: chains", sampler.chains),
        ("Sampler: warmup", sampler.warmup),
        ("Sampler: sampling", sampler.sampling),
        ("Sampler: adapt_delta", sampler.adapt_delta),
        ("Sampler: max_treedepth", sampler.max_treedepth),
        ("Sampler: seed", sampler.seed),
    ]
    body = "\n".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in rows)
    return (
        f"<h1>LW-SV report -- run {_esc(run_hash)}</h1>\n"
        f'<table class="meta-table">{body}</table>'
    )


def _render_diagnostics(diagnostics: dict) -> str:
    verdict = diagnostics["verdict"]
    bg, border = _VERDICT_COLORS.get(verdict, ("#f6f8fa", "#57606a"))
    reasons_html = "".join(f"<li>{_esc(r)}</li>" for r in diagnostics["reasons"])
    ebfmi = ", ".join(f"{x:.3f}" for x in diagnostics["e_bfmi_per_chain"])

    rows = [
        ("Divergences", diagnostics["divergences"]),
        (
            "Max-treedepth hits",
            f"{diagnostics['max_treedepth_hits']} (config max_treedepth={diagnostics['max_treedepth_config']})",
        ),
        ("E-BFMI per chain", ebfmi),
        ("Max R-hat (across parameters)", f"{diagnostics['rhat_max']:.4f}"),
        ("Min bulk ESS (across parameters)", f"{diagnostics['ess_bulk_min']:.0f}"),
        ("Min tail ESS (across parameters)", f"{diagnostics['ess_tail_min']:.0f}"),
    ]
    rows_html = "\n".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in rows)

    return (
        "<h2>Diagnostics</h2>\n"
        f'<div class="verdict-box" style="background:{bg};border-color:{border};">\n'
        f'<span class="verdict-label" style="color:{border};">Verdict: {_esc(verdict)}</span>\n'
        f'<div class="reasons"><ul>{reasons_html}</ul></div>\n'
        f'<table class="meta-table">{rows_html}</table>\n'
        "</div>"
    )


# ---------------------------------------------------------------------------
# §3.1-3.4 figures
# ---------------------------------------------------------------------------


def _render_trend_cycle_section(lw_run: LWRun) -> str:
    from macrotoolkit.plots import plot_trend_cycle
    from macrotoolkit.results_lw import compute_trend_cycle_draws

    tcd = compute_trend_cycle_draws(lw_run)
    fig = plot_trend_cycle(tcd, lw_run)
    uri = _fig_to_data_uri(fig, dpi=100)
    vol_caption = (
        "Includes the shock volatility panel (exp(h/2)) -- this is an SV run (sv_shocks: [is, pc])."
        if lw_run.sv_on
        else "No volatility panel -- this is a no-SV run (sv_shocks: []); constant IS/PC shock variances."
    )
    block = _figure_block("Trend-cycle decomposition (spec §3.1)", uri, vol_caption)
    return f"<h2>3.1 Trend-cycle plots</h2>\n{block}"


def _render_irf_section(lw_run: LWRun) -> str:
    from macrotoolkit.plots import plot_irf_matrix
    from macrotoolkit.results_lw import compute_irf_draws

    irf = compute_irf_draws(lw_run)
    fig = plot_irf_matrix(irf, lw_run.spec.outputs.irf_vol_reference)
    uri = _fig_to_data_uri(fig, dpi=90)
    block = _figure_block(
        "IRF matrix (spec §3.2)",
        uri,
        f"5 shocks x 5 responses, horizon={irf.horizon} quarters, "
        f"irf_vol_reference={lw_run.spec.outputs.irf_vol_reference!r}.",
    )
    return f"<h2>3.2 IRF matrix</h2>\n{block}"


def _render_fan_section(lw_run: LWRun) -> str:
    from macrotoolkit.plots import plot_fan_charts
    from macrotoolkit.results_lw import compute_fan_draws

    fans = compute_fan_draws(lw_run)
    figs = plot_fan_charts(fans, lw_run.spec.outputs.forecast_r_rule)
    order = ["y_level", "y_growth_4q", "pi", "gap", "rstar"]
    blocks = []
    for key in order:
        uri = _fig_to_data_uri(figs[key], dpi=100)
        blocks.append(_figure_block(f"Fan chart: {key}", uri))
    grid = f'<div class="fig-grid">{"".join(blocks)}</div>'
    caption = (
        f"Horizon={fans.horizon} quarters, forecast_r_rule={lw_run.spec.outputs.forecast_r_rule!r} "
        f"(also printed on each chart)."
    )
    return f"<h2>3.3 Fan charts</h2>\n<p class=\"caption\">{_esc(caption)}</p>\n{grid}"


def _render_prior_predictive_section(lw_run: LWRun) -> str:
    from macrotoolkit.plots import plot_prior_predictive
    from macrotoolkit.results_lw import compute_prior_predictive_draws

    ppd = compute_prior_predictive_draws(lw_run)
    fig = plot_prior_predictive(ppd)
    uri = _fig_to_data_uri(fig, dpi=100)
    block = _figure_block(
        "Prior-predictive check (spec §4)",
        uri,
        f"{ppd.n_draws} full observable paths simulated from the run's own "
        f"resolved priors (defaults + spec overrides) through the same "
        f"matrices/engine the run used -- 'what do my priors imply about "
        f"observable paths', S5-decisions item 7. Needs no posterior draws.",
    )
    # Grouped with the diagnostics block (spec §4 lists the prior-
    # predictive figure among the per-run diagnostics, and §3.5's report
    # layout numbers only §3.1-3.4 as output sections -- numerics-reviewer
    # ordering fix, 2026-09-02).
    return f"<h2>Diagnostics: prior-predictive check (spec §4)</h2>\n{block}"


def _render_hd_section(lw_run: LWRun) -> str:
    from macrotoolkit.plots import plot_historical_decomposition
    from macrotoolkit.results_lw import compute_historical_decomposition_draws

    hdd = compute_historical_decomposition_draws(lw_run)
    figs = plot_historical_decomposition(hdd)
    order = ["gap", "pi", "y_growth_4q", "y_level"]
    blocks = []
    for key in order:
        uri = _fig_to_data_uri(figs[key], dpi=100)
        blocks.append(_figure_block(f"Historical decomposition: {key}", uri))
    grid = f'<div class="fig-grid">{"".join(blocks)}</div>'
    return f"<h2>3.4 Historical decomposition</h2>\n{grid}"


# ---------------------------------------------------------------------------
# Parameter table
# ---------------------------------------------------------------------------


def _scalar_param_names(posterior) -> list[str]:
    """The TRUE scalar (time-invariant) parameters of this run's posterior --
    every ``data_vars`` entry whose only dims are ``('chain', 'draw')``,
    i.e. no extra per-index dimension. Determined by inspecting the actual
    posterior's own dims (never guessed/hardcoded): for an SV run this
    excludes the length-T vector parameters ``h_is``/``h_pc``/``nu_is``/
    ``nu_pc`` (and includes the scalar non-centered SV parameters
    ``h0_is_raw``/``h0_pc_raw``/``sigma_h_is``/``sigma_h_pc``); for a no-SV
    run every parameter (``a1, a2, a_r, b_pi, b_y, sigma_ystar, sigma_g,
    sigma_z, sigma_is, sigma_pc``) is scalar. A v1 parameter table reports
    scalars only -- per-element rows for a length-T vector parameter would
    make the table useless as a "read this at a glance" artifact; that is
    what the trend-cycle/volatility plot is for.
    """
    names = []
    for name in sorted(posterior.data_vars):
        if set(posterior[name].dims) == {"chain", "draw"}:
            names.append(name)
    return names


def _param_table_rows(lw_run: LWRun) -> list[dict]:
    """Posterior median, 90% CI (5th/95th percentile), R-hat, bulk/tail ESS
    for every scalar parameter (:func:`_scalar_param_names`) -- computed
    directly from ``lw_run.idata.posterior`` (ArviZ) via ``numpy``/``arviz``,
    not read off ``diagnostics.json`` (which stores a worst-case reduction
    over any vector dims; scalar-only here, computed fresh)."""
    import arviz as az

    posterior = lw_run.idata.posterior
    names = _scalar_param_names(posterior)
    if not names:
        raise ValueError(
            f"Run {lw_run.run_dir} has no scalar (chain,draw)-only posterior "
            f"variables -- cannot build a parameter table. Available "
            f"variables: {sorted(posterior.data_vars)}."
        )
    rhat_ds = az.rhat(lw_run.idata, var_names=names)
    ess_bulk_ds = az.ess(lw_run.idata, var_names=names, method="bulk")
    ess_tail_ds = az.ess(lw_run.idata, var_names=names, method="tail")

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


def _render_param_table(lw_run: LWRun) -> str:
    rows = _param_table_rows(lw_run)
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
        "Scalar (time-invariant) parameters only -- vector-valued SV path "
        "parameters (h_is, h_pc, nu_is, nu_pc for an SV run) are omitted "
        "from this table; see the trend-cycle volatility panel above for "
        "those."
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
    """Render a completed ``lw_sv`` run's full ``report.html`` document
    (spec §3.5) as a string. Pure with respect to ``run_dir``: reads
    ``run_dir``'s own already-written artifacts (``spec.yaml``,
    ``data.snapshot.csv``, ``draws.nc`` via :func:`macrotoolkit.results_lw.
    load_lw_run`, plus ``diagnostics.json`` directly) but never writes
    anything -- see :func:`write_report` for the filesystem-writing wrapper.
    """
    run_dir = Path(run_dir)
    diagnostics_path = run_dir / "diagnostics.json"
    if not diagnostics_path.is_file():
        raise FileNotFoundError(
            f"{diagnostics_path} not found -- {run_dir} does not look like a "
            f"completed run directory (expected diagnostics.json alongside "
            f"spec.yaml/draws.nc). Run `mtk run <spec.yaml>` first."
        )
    diagnostics = json.loads(diagnostics_path.read_text())

    lw_run = load_lw_run(run_dir)
    run_hash = run_dir.name

    sections = [
        _render_header(lw_run, run_hash),
        _render_diagnostics(diagnostics),
        _render_prior_predictive_section(lw_run),
        _render_trend_cycle_section(lw_run),
        _render_irf_section(lw_run),
        _render_fan_section(lw_run),
        _render_hd_section(lw_run),
        _render_param_table(lw_run),
    ]
    body = "\n".join(sections)

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>LW-SV report -- {_esc(run_hash)}</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def write_report(run_dir: str | Path) -> Path:
    """Render (:func:`render_report`) and write ``run_dir / "report.html"``.
    Returns the written path. Safe to call on an already-``_SUCCESS``-marked
    run directory (see module docstring: ``report.html`` is a derived,
    regenerable view, not a sampling artifact covered by ``run.py``'s
    immutability doctrine) -- overwrites any previous ``report.html`` in
    place, since a report is not itself part of run identity."""
    run_dir = Path(run_dir)
    html_text = render_report(run_dir)
    out_path = run_dir / "report.html"
    out_path.write_text(html_text, encoding="utf-8")
    return out_path
