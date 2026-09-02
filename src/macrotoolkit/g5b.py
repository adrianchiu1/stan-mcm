"""G5b -- the INFORMATIONAL exhibit (S5-decisions item 8; spec §5's G5b row
as amended 2026-09-02): our model's FILTERED (one-sided) estimates versus
the NY Fed's published one-sided HLW series, with every difference
explained and attributed. NO pass/fail criterion: the original ±50bp gate
was demoted by user decision -- "a gate you'd pass by tuning priors toward
a target validates nothing" -- and exact HLW replication is explicitly not
a goal (the KF/smoother core's ~1e-12 G5a validation is already banked).

Like-for-like notes (why FILTERED vs one-sided is the right comparison):
HLW publish NO smoothed estimates at all -- their workbook's own header
states "All estimates are one-sided" -- so comparing our smoothed series
against their published numbers would be comparing different estimands.
Our filtered series uses the identical state-to-series mapping G5a
validated against HLW's own output at ~1e-12 (rstar = state(g,-1) +
state(z,-1) etc., HLW's own reporting convention), evaluated per posterior
draw and summarized by the posterior median with 68/90% bands.

Documented, attributed differences (each stated on the exhibit):
1. **sigma_z pile-up prior vs MUE lambda_z** -- the deliberate Half-N(0,
   0.08^2) identification prior (spec §1.6) holds sigma_z below HLW's
   MUE-implied value, damping |z| and pulling r* toward g; the dominant,
   deliberate cause of the late-sample r* level gap (STRESS-TESTS.md's
   documented sigma_z-prior level gap; quantified per-date in the exhibit
   table, and testable via the sigma_z_loose cell of the mandated
   `mtk sweep` sigma_g_z sweep).
2. **Bayesian posterior median vs frequentist MLE plug-in** -- HLW filter
   at a single L-BFGS point estimate; we integrate over the posterior.
3. **Stochastic volatility** -- our reference run carries SV on the IS/PC
   shocks; HLW's model has constant variances (reallocates what constant
   scales force elsewhere; DECISIONS.md 2026-08-31 records the posterior
   shifts).
4. **Data vintage** -- our example CSV vs the vintage underlying the
   published sheet (G5a's fixture work measured this class of difference
   at ~0.06pp mean / 0.31pp max on the states for a same-model
   reproduction).
5. **Initialization** -- our explicit (xi00, P00) prior anchored at the
   first observation vs HLW's inner-optimization P.00 procedure.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from macrotoolkit.families.lw_sv import LW_STATE_META

REPO_ROOT = Path(__file__).resolve().parents[2]
_PUBLISHED_XLSX = (
    REPO_ROOT / "tests" / "fixtures" / "hlw" / "data"
    / "Holston_Laubach_Williams_real_time_estimates.xlsx"
)

#: US column indices in the real-time workbook's vintage sheets (header
#: rows 4-5: blocks Output Gap / Trend Growth (g, annualized) / Other
#: Determinants (z) / Natural Rate (r*), US sub-column of each).
_US_COLS = {"output_gap": 2, "g": 7, "z": 12, "rstar": 17}
_DATE_COL = 0
_FIRST_DATA_ROW = 6


@dataclass
class FilteredSeriesDraws:
    """Per-posterior-draw FILTERED (one-sided) series, same reporting
    mapping and shapes as ``results_lw.TrendCycleDraws`` but from the KF's
    ``xi_filt`` instead of smoother draws (and deterministic given the
    posterior draw -- no simulation-smoother RNG involved)."""

    draw_indices: np.ndarray
    dates: pd.DatetimeIndex
    output_gap: np.ndarray  # (n_draws, T)
    g: np.ndarray
    z: np.ndarray
    rstar: np.ndarray


def compute_filtered_series_draws(lw_run, *, thin: int = 1) -> FilteredSeriesDraws:
    """Run the (deterministic) Kalman filter once per selected posterior
    draw and map filtered states to the G5a-validated reporting series.
    ``thin`` further thins the run's own ``outputs.smoother_draws``
    selection (an exhibit needs medians, not full band resolution)."""
    from macrotoolkit.results_lw import (
        _flatten_posterior,
        _system_matrices_for_draw,
        select_draw_indices,
    )
    from macrotoolkit.smoother import kalman_smoother

    s_ystar = LW_STATE_META.slot("ystar", 0)
    s_g = LW_STATE_META.slot("g", -1)
    s_z = LW_STATE_META.slot("z", -1)

    flat = _flatten_posterior(lw_run)
    n_total = flat["a1"].shape[0]
    idx = select_draw_indices(n_total, lw_run.spec.outputs.smoother_draws)[::thin]

    T = lw_run.yobs.shape[0]
    n = len(idx)
    gap = np.empty((n, T))
    g_arr = np.empty((n, T))
    z_arr = np.empty((n, T))
    rstar = np.empty((n, T))

    for j, i in enumerate(idx):
        F, Q, A, Z, R, _, _ = _system_matrices_for_draw(flat, int(i), lw_run.sv_on)
        out = kalman_smoother(lw_run.yobs, lw_run.x, F, Q, A, Z, R, lw_run.xi00, lw_run.P00)
        xi_filt = out["xi_filt"]
        gap[j] = lw_run.yobs[:, 0] - xi_filt[:, s_ystar]
        g_arr[j] = xi_filt[:, s_g]
        z_arr[j] = xi_filt[:, s_z]
        rstar[j] = xi_filt[:, s_g] + xi_filt[:, s_z]

    return FilteredSeriesDraws(
        draw_indices=idx, dates=lw_run.dates, output_gap=gap, g=g_arr, z=z_arr, rstar=rstar
    )


def load_published_one_sided(vintage: str = "2019Q2") -> pd.DataFrame:
    """The NY Fed's published one-sided US series from the real-time
    workbook's ``vintage`` sheet (user-supplied fixture, FIXTURES.md):
    columns date, output_gap, g, z, rstar."""
    raw = pd.read_excel(_PUBLISHED_XLSX, sheet_name=vintage, header=None)
    data = raw.iloc[_FIRST_DATA_ROW:]
    out = pd.DataFrame({"date": pd.to_datetime(data.iloc[:, _DATE_COL])})
    for name, col in _US_COLS.items():
        out[name] = pd.to_numeric(data.iloc[:, col].to_numpy(), errors="coerce")
    out = out.dropna(subset=["date", "rstar"]).reset_index(drop=True)
    if out.empty:
        raise ValueError(
            f"No usable US rows in sheet {vintage!r} of {_PUBLISHED_XLSX} -- "
            f"column layout changed? Expected US blocks at {_US_COLS}."
        )
    return out


@dataclass
class G5bExhibit:
    vintage: str
    dates: pd.DatetimeIndex  # aligned intersection
    ours_median: dict[str, np.ndarray]  # series -> (T,)
    ours_lo90: dict[str, np.ndarray]
    ours_hi90: dict[str, np.ndarray]
    published: dict[str, np.ndarray]
    stats: dict[str, dict[str, float]]  # series -> summary stats


_SERIES = ("rstar", "output_gap", "g", "z")


def build_exhibit(lw_run, *, vintage: str = "2019Q2", thin: int = 5) -> G5bExhibit:
    """Align our filtered posterior series with the published one-sided
    series on their common dates and compute the exhibit's summary
    statistics (mean/max absolute difference over the common sample and
    over 2000+, correlation, and the final-period difference -- the
    numbers the attribution table cites)."""
    ours = compute_filtered_series_draws(lw_run, thin=thin)
    pub = load_published_one_sided(vintage)

    common = ours.dates.intersection(pd.DatetimeIndex(pub["date"]))
    if len(common) < 40:
        raise ValueError(
            f"Only {len(common)} common quarters between the run's sample "
            f"and the published {vintage} sheet -- wrong run/vintage pairing?"
        )
    ours_pos = ours.dates.get_indexer(common)
    pub_indexed = pub.set_index("date").loc[common]

    ours_arrays = {"rstar": ours.rstar, "output_gap": ours.output_gap, "g": ours.g, "z": ours.z}
    ours_median: dict[str, np.ndarray] = {}
    ours_lo90: dict[str, np.ndarray] = {}
    ours_hi90: dict[str, np.ndarray] = {}
    published: dict[str, np.ndarray] = {}
    stats: dict[str, dict[str, float]] = {}
    late = common >= pd.Timestamp("2000-01-01")

    for name in _SERIES:
        med = np.median(ours_arrays[name], axis=0)[ours_pos]
        lo = np.percentile(ours_arrays[name], 5.0, axis=0)[ours_pos]
        hi = np.percentile(ours_arrays[name], 95.0, axis=0)[ours_pos]
        pub_s = pub_indexed[name].to_numpy(dtype=np.float64)
        diff = med - pub_s
        ours_median[name] = med
        ours_lo90[name] = lo
        ours_hi90[name] = hi
        published[name] = pub_s
        stats[name] = {
            "mean_abs_diff": float(np.mean(np.abs(diff))),
            "max_abs_diff": float(np.max(np.abs(diff))),
            "mean_abs_diff_2000plus": float(np.mean(np.abs(diff[late]))),
            "corr": float(np.corrcoef(med, pub_s)[0, 1]),
            "final_diff": float(diff[-1]),
        }

    return G5bExhibit(
        vintage=vintage,
        dates=common,
        ours_median=ours_median,
        ours_lo90=ours_lo90,
        ours_hi90=ours_hi90,
        published=published,
        stats=stats,
    )


def plot_exhibit(ex: G5bExhibit):
    """The exhibit figure: our filtered posterior median + 90% band vs the
    published one-sided line, one panel per series, conventions stated on
    the figure (ENGINEERING.md). Returns the matplotlib Figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {
        "rstar": "Natural rate r* (one-sided)",
        "output_gap": "Output gap (one-sided)",
        "g": "Trend growth g, annualized (one-sided)",
        "z": "Other determinants z (one-sided)",
    }
    fig, axes = plt.subplots(len(_SERIES), 1, figsize=(10.0, 3.0 * len(_SERIES)), sharex=True)
    for ax, name in zip(axes, _SERIES):
        ax.fill_between(
            ex.dates, ex.ours_lo90[name], ex.ours_hi90[name],
            color="tab:blue", alpha=0.15, linewidth=0, label="ours 90% band",
        )
        ax.plot(ex.dates, ex.ours_median[name], color="tab:blue", linewidth=1.5,
                label="ours: filtered posterior median (lw_sv, SV on)")
        ax.plot(ex.dates, ex.published[name], color="black", linewidth=1.2, linestyle="--",
                label=f"published HLW one-sided ({ex.vintage} vintage)")
        s = ex.stats[name]
        ax.set_title(
            f"{titles[name]} -- mean |diff| {s['mean_abs_diff']:.2f}, "
            f"2000+ {s['mean_abs_diff_2000plus']:.2f}, corr {s['corr']:.3f}"
        )
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Date")
    fig.suptitle(
        "G5b informational exhibit: our FILTERED series vs the published one-sided HLW series\n"
        "(no pass/fail -- differences documented and attributed; see the exhibit notes)",
        fontsize=12,
    )
    fig.text(
        0.01, 0.005,
        "Like-for-like: HLW publish one-sided estimates only ('All estimates are one-sided'); ours are "
        "the KF-filtered series per posterior draw, G5a-validated state-to-series mapping, posterior "
        "median +/- 90% band. Differences attributed in the exhibit notes: sigma_z pile-up prior (the "
        "deliberate identification choice), Bayesian median vs MLE plug-in, SV vs constant variances, "
        "data vintage, initialization.",
        fontsize=6.5, color="0.35",
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    return fig


def exhibit_notes_markdown(ex: G5bExhibit, run_hash: str) -> str:
    """The written half of the exhibit: the summary-statistics table plus
    the attribution of every difference (module docstring's five causes),
    as markdown for docs/README embedding."""
    lines = [
        "# G5b informational exhibit: filtered vs published one-sided HLW",
        "",
        f"Run `{run_hash}` (reference SV run) vs the NY Fed real-time workbook's "
        f"`{ex.vintage}` vintage sheet (one-sided; their header: \"All estimates are "
        f"one-sided\" -- HLW publish no smoothed estimates, so the one-sided/FILTERED "
        f"comparison is the only like-for-like one). **Informational only -- no "
        f"pass/fail** (S5-decisions item 8: a gate passed by tuning priors toward a "
        f"target validates nothing).",
        "",
        "| Series | mean abs diff | mean abs diff (2000+) | max abs diff | corr | final-period diff |",
        "|---|---|---|---|---|---|",
    ]
    for name in _SERIES:
        s = ex.stats[name]
        lines.append(
            f"| {name} | {s['mean_abs_diff']:.2f} | {s['mean_abs_diff_2000plus']:.2f} "
            f"| {s['max_abs_diff']:.2f} | {s['corr']:.3f} | {s['final_diff']:+.2f} |"
        )
    lines += [
        "",
        "Attributed causes of the differences (each deliberate or documented, none a",
        "numerics discrepancy -- the KF/smoother core matches HLW's own machinery to",
        "~1e-12 at fixed parameters, gate G5a):",
        "",
        "1. **sigma_z pile-up prior vs MUE lambda_z** -- the deliberate Half-N(0, 0.08^2)",
        "   identification prior (spec §1.6) holds sigma_z below HLW's MUE-implied value,",
        "   damping |z| and pulling r* toward g. This is the dominant cause of the",
        "   late-sample r* level gap, and it is testable: the mandated `mtk sweep`",
        "   sigma_g_z sweep's `sigma_z_loose` cell (prior sd doubled) moves r* toward the",
        "   published series.",
        "2. **Bayesian posterior median vs frequentist MLE plug-in** -- HLW filter at a",
        "   single L-BFGS point estimate; we integrate over the posterior.",
        "3. **Stochastic volatility** -- our reference run has SV on the IS/PC shocks;",
        "   HLW's model has constant variances (DECISIONS.md 2026-08-31 records the",
        "   induced posterior shifts, e.g. sigma_y* 0.24 vs 0.54).",
        "4. **Data vintage** -- our example CSV vs the vintage underlying the published",
        "   sheet (~0.06pp mean / 0.31pp max on the states for a same-model reproduction,",
        "   G5a fixture work).",
        "5. **Initialization** -- our explicit (xi00, P00) prior anchored at the first",
        "   observation vs HLW's inner-optimization P.00 procedure (G5a used their exact",
        "   values; the production run does not).",
    ]
    return "\n".join(lines) + "\n"


def write_exhibit(run_dir, out_dir, *, vintage: str = "2019Q2", thin: int = 5) -> tuple[Path, Path]:
    """Build and write the exhibit PNG + markdown notes for a completed
    reference run. Returns (png_path, md_path)."""
    import matplotlib.pyplot as plt

    from macrotoolkit.results_lw import load_lw_run

    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lw_run = load_lw_run(run_dir)
    ex = build_exhibit(lw_run, vintage=vintage, thin=thin)
    fig = plot_exhibit(ex)
    png_path = out_dir / "g5b_filtered_vs_published.png"
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    md_path = out_dir / "g5b_filtered_vs_published.md"
    md_path.write_text(exhibit_notes_markdown(ex, run_dir.name), encoding="utf-8")
    return png_path, md_path
