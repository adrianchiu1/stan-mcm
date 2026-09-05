"""Build (and optionally execute) the companion notebooks, one per handbook chapter, from the
committed specs under examples/handbook/. Each notebook is the executable counterpart of the
matching docs/companion chapter: it fits the chapter's examples (reusing runs by hash), prints
the run identity, the diagnostics verdict and the fit-time mirror check, shows the parameter
table and the output figures, applies the post-processors the handbook's examples call for, and
runs the fast validation tier.

    python examples/handbook/notebooks/build_notebooks.py                 # write all notebooks
    python examples/handbook/notebooks/build_notebooks.py --execute ch1   # write + execute in place
    python examples/handbook/notebooks/build_notebooks.py --execute --skip-heavy ch2 ch3

"Heavy" examples (the 11-variable sign-restriction VAR, the 40-series DFMs, the TVP-VAR) take
an hour or more each; --skip-heavy leaves their cells in the notebook but marks them for a run
session (the cell is a no-op unless HANDBOOK_HEAVY=1 is set in the environment).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
HEAVY = {"ch2_signs_11var", "ch3_dfm_uk_panel", "ch3_tvp_var", "ch7_dfm_sv"}

PREAMBLE = '''from pathlib import Path
import os
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from macrotoolkit import api as mtk
from macrotoolkit.run import REPO_ROOT as REPO   # never walk parent directories (HANDOFF warning)

HB = REPO / "examples" / "handbook"
HEAVY = os.environ.get("HANDBOOK_HEAVY") == "1"

def fit(name):
    """Fit an example spec (reused by hash if the run exists) and print its record."""
    run = mtk.fit(str(HB / name / "spec.yaml"))
    mc = run.mirror_check
    print(f"{name}: run {run.hash} | verdict {run.verdict} | {'; '.join(run.diagnostics['reasons'])}")
    print(f"  fit-time mirror check: max |Stan - Python| KF loglik = {mc['max_abs_diff']:.2e} over {mc['n_points']} prior draws")
    return run

def heavy(name):
    if not HEAVY:
        print(f"{name}: heavy example -- run with HANDBOOK_HEAVY=1 (a run session); skipped here.")
        return None
    return fit(name)
'''


def _fit_cell(name: str, heavy: bool = False) -> str:
    return f'run = {"heavy" if heavy else "fit"}("{name}")\n' + ("run.param_table().round(4) if run is not None else None" if heavy else "run.param_table().round(4)")


def _fig_cell(mod: str, key: str | None = None, guard: bool = False) -> str:
    call = f'run.outputs().figure("{mod}")' + (f'["{key}"]' if key else "")
    return (f"{call} if run is not None else None") if guard else call


def _validate_cell(name: str) -> str:
    return f'''result = mtk.validate(HB / "{name}" / "spec.yaml", tier="fast")
for g in result.gates:
    print(f"{{g.name:12s}} {{g.verdict}}  {{g.summary}}")'''


def ch1() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell("# Chapter 1 — Linear regression models\n\nThe executable counterpart of `docs/companion/02-ch1-regression.md`: the handbook's AR(2) for US inflation (examples 1–2) and the AR(2) with AR(1) errors (example 3), on the handbook's `inflation.xls`."),
        new_code_cell(PREAMBLE),
        new_markdown_cell("## The AR(2) with a constant (example 1) — compare with the handbook's Table 1: α 0.2494, B₁ 1.3867, B₂ −0.4600"),
        new_code_cell(_fit_cell("ch1_ar2")),
        new_markdown_cell("The prior-predictive check: under N(0, 1) autoregressive priors a large share of prior paths is explosive — the handbook enforces stability by rejection; the toolkit shows the prior's implication."),
        new_code_cell(_fig_cell("prior_predictive")),
        new_markdown_cell("## The forecast distribution (example 2): the `fan` module simulates forward from every posterior draw"),
        new_code_cell(_fig_cell("fan", "infl")),
        new_markdown_cell("## Serially correlated errors (example 3): the exact quasi-differenced equation\n\nThe verdict at this chain length is FAIL on tail ESS — the posterior is bimodal (an autoregressive root can sit in the regression or in the error), which the handbook's Figure 14 shows as chains jumping between two configurations."),
        new_code_cell(_fit_cell("ch1_ar2_ar1err")),
        new_code_cell(_fig_cell("fan", "infl")),
        new_markdown_cell("## The fast validation tier on the AR(2)"),
        new_code_cell(_validate_cell("ch1_ar2")),
        new_code_cell('plt.close("all")'),
    ]
    return new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})


def ch2() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell("# Chapter 2 — Vector autoregressions\n\nThe executable counterpart of `docs/companion/03-ch2-vars.md`: the Minnesota bivariate VAR (example 1), the monthly Cholesky VAR (example 2), the steady-state prior (example 3), sign restrictions (examples 5–7) and the conditional forecast (example 8)."),
        new_code_cell(PREAMBLE),
        new_markdown_cell("## Example 1: the bivariate VAR(2) with a Minnesota prior, in recursive form"),
        new_code_cell(_fit_cell("ch2_bivar_minnesota")),
        new_code_cell(_fig_cell("irf")),
        new_code_cell(_fig_cell("fan", "y")),
        new_markdown_cell("## Example 2: the 4-variable monthly VAR — the engine's structural IRFs under the ordering are the Cholesky IRFs"),
        new_code_cell(_fit_cell("ch2_var4_monthly_cholesky")),
        new_code_cell('irf = run.outputs().compute("irf")\nr = irf.responses["e_bond10y"]\nprint("impact medians of the bond-yield shock:", {o: round(float(np.median(r[o][:, 0])), 3) for o in r})\nrun.outputs().figure("irf")'),
        new_markdown_cell("## Example 3: the steady-state (Villani) prior — long-run means as constant states"),
        new_code_cell(_fit_cell("ch2_steady_state")),
        new_code_cell('sd = run.outputs().compute("states")\nprint({k: round(float(np.median(v)), 3) for k, v in sd.states.items()})\nrun.outputs().figure("fan")["y"]'),
        new_markdown_cell("## Example 8: the Waggoner–Zha conditional forecast — inflation held at 1% for three quarters"),
        new_code_cell(_fit_cell("ch2_conditional")),
        new_code_cell('''from macrotoolkit.postprocess import conditional_forecast_for_run
cf = conditional_forecast_for_run(run, {("pi", 0): 1.0, ("pi", 1): 1.0, ("pi", 2): 1.0}, horizon=3, draws_per_posterior_draw=2)
med = np.median(cf.draws, axis=0)
print("conditional median GDP-growth path:", np.round(med[:, 0], 3).tolist(), "| unconditional:", np.round(cf.unconditional[:, 0], 3).tolist())
print("the inflation path reproduces (1, 1, 1) to", f"{np.max(np.abs(cf.draws[:, :, 1] - 1.0)):.1e}")
fig, ax = plt.subplots(figsize=(7, 3))
for q, ls in ((0.05, ":"), (0.5, "-"), (0.95, ":")):
    ax.plot(range(1, 4), np.quantile(cf.draws[:, :, 0], q, axis=0), color="C0", linestyle=ls)
ax.plot(range(1, 4), cf.unconditional[:, 0], color="0.4", label="unconditional median")
ax.set_title("GDP growth: conditional (blue, 5/50/95%) vs unconditional"); ax.legend(); ax'''),
        new_markdown_cell("## Examples 5–7: the 11-variable VAR with sign restrictions (heavy — a run session; set HANDBOOK_HEAVY=1)"),
        new_code_cell(_fit_cell("ch2_signs_11var", heavy=True)),
        new_code_cell('''if run is not None:
    from macrotoolkit.postprocess import SignRestriction, irf_array_from_draws, sign_restricted_irfs
    obs = tuple(run.results().meta.obs_names)
    arr, targets, shocks = irf_array_from_draws(run.outputs().compute("irf"), targets=obs)
    restr = [SignRestriction("mp", v, s, (0,)) for v, s in (("ffr", +1), ("gdp_growth", -1), ("cpi_inflation", -1), ("pce_growth", -1), ("unemployment", +1), ("investment", -1), ("m2", -1))]
    sr = sign_restricted_irfs(arr, restr, targets, np.random.default_rng(20260905), max_tries=2000)
    print(f"{sr.irf.shape[0]} of {arr.shape[0]} posterior draws accepted; median impact on gdp_growth {sr.bands('mp', 'gdp_growth')[1][0]:+.3f}")'''),
        new_code_cell('plt.close("all")'),
    ]
    return new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})


def ch3() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell("# Chapter 3 — State-space models\n\nThe executable counterpart of `docs/companion/04-ch3-state-space.md`: the TVP regression on the handbook's artificial DGP (examples 1–2), the unobserved-components trend–cycle model (§2), the TVP-VAR (example 3) and the DFM block of the FAVAR (example 4)."),
        new_code_cell(PREAMBLE),
        new_markdown_cell("## Examples 1–2: the TVP regression — the true `√R = 0.1`, `√Q ≈ 0.032` are recovered and the smoothed β_t band contains the truth"),
        new_code_cell(_fit_cell("ch3_tvp_regression")),
        new_code_cell('''truth = pd.read_csv(HB / "data" / "ch3_tvp_example1_sim.csv")["beta_true"].to_numpy()
sd = run.outputs().compute("states"); beta = sd.states["beta"]
lo, hi, med = np.quantile(beta, 0.05, axis=0), np.quantile(beta, 0.95, axis=0), np.median(beta, axis=0)
print(f"truth inside the 90% band at {np.mean((truth >= lo) & (truth <= hi)):.1%} of periods; RMSE of the smoothed median {np.sqrt(np.mean((med - truth) ** 2)):.4f}")
run.outputs().figure("states")'''),
        new_markdown_cell("## §2: the unobserved-components trend–cycle model on US inflation (no measurement error: a singular R)"),
        new_code_cell(_fit_cell("ch3_uc_trend_cycle")),
        new_code_cell(_fig_cell("states")),
        new_code_cell(_fig_cell("hd", "Y")),
        new_markdown_cell("## Example 3: the TVP-VAR — IRFs conditional on the coefficient state at three dates (heavy)"),
        new_code_cell(_fit_cell("ch3_tvp_var", heavy=True)),
        new_code_cell('''if run is not None:
    irf = run.outputs().compute("irf")
    for lab, resp in irf.by_date.items():
        print(lab, {o: round(float(np.median(resp["e_ffr"][o][:, 3])), 3) for o in resp["e_ffr"]})
    run.outputs().figure("states")'''),
        new_markdown_cell("## Example 4: the DFM block of the FAVAR on the 40-series UK panel (heavy)"),
        new_code_cell(_fit_cell("ch3_dfm_uk_panel", heavy=True)),
        new_code_cell(_fig_cell("states", guard=True)),
        new_code_cell('plt.close("all")'),
    ]
    return new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})


def ch5() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell("# Chapter 5 — Stochastic volatility and time-varying parameters\n\nThe executable counterpart of `docs/companion/06-ch5-mh-and-sv.md`: the SV model for UK inflation (example 4) in the handbook's mean-only form and in unobserved-components form, and the TVP-AR(1) with SV (example 5)."),
        new_code_cell(PREAMBLE),
        new_markdown_cell("## Example 4: the SV model in the handbook's own form — a constant mean, no state, the SV block on the shock"),
        new_code_cell(_fit_cell("ch5_sv_uk_inflation")),
        new_code_cell(_fig_cell("states")),
        new_markdown_cell("## The same SV shock around a random-walk level (the form first run on the S7 branch)"),
        new_code_cell(_fit_cell("ch5_sv_ucsv_form")),
        new_code_cell(_fig_cell("states")),
        new_markdown_cell("## Example 5: the TVP-AR(1) with SV — the handbook's four panels at three dates, and IRFs conditional on the coefficient state"),
        new_code_cell(_fit_cell("ch5_tvp_ar1_sv")),
        new_code_cell('''sd = run.outputs().compute("states"); d = sd.dates
picks = [int(np.argmin(np.abs(d - np.datetime64(x)))) for x in ("1930-01-01", "1975-01-01", "2008-10-01")]
c, b = sd.states["c"], sd.states["b"]
print("b_t:", [round(float(np.median(b[:, i])), 2) for i in picks], "c_t:", [round(float(np.median(c[:, i])), 2) for i in picks],
      "long-run mean:", [round(float(np.median((c / (1 - b))[:, i])), 1) for i in picks], "exp(h/2):", [round(float(np.median(sd.vol["e"][:, i])), 2) for i in picks])
run.outputs().figure("states")'''),
        new_code_cell('irf = run.outputs().compute("irf")\nprint({k: round(float(np.median(v["e"]["pi"][:, 3])), 2) for k, v in irf.by_date.items()}, "| omitted:", list(irf.omitted_shocks))\nrun.outputs().figure("irf")'),
        new_code_cell('plt.close("all")'),
    ]
    return new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})


def ch7() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell("# Chapter 7 — The dynamic factor model with stochastic volatility (the constant-coefficient bridge)\n\nThe executable counterpart of `docs/companion/08-ch7-tvp-dfm.md`: world and country factors with SV on the factor shocks on the handbook's generated panel (`dataxx01.mat`, the first two countries); the handbook's time-varying AR coefficients are extension E6."),
        new_code_cell(PREAMBLE),
        new_code_cell(_fit_cell("ch7_dfm_sv", heavy=True)),
        new_code_cell(_fig_cell("states", guard=True)),
        new_code_cell('plt.close("all")'),
    ]
    return new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})


BUILDERS = {"ch1": ch1, "ch2": ch2, "ch3": ch3, "ch5": ch5, "ch7": ch7}


def strip_progress_noise(nb: nbformat.NotebookNode) -> None:
    """Drop cmdstanpy/tqdm progress lines from stream outputs so the committed notebook is readable."""
    pat = re.compile(r"^(chain \d+:|\s*\d+%\||.*cmdstanpy - INFO).*$", re.M)
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                text = out.get("text", "").replace("\r", "\n")
                out["text"] = "\n".join(l for l in text.split("\n") if l.strip() and not pat.match(l)) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chapters", nargs="*", default=list(BUILDERS))
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--skip-heavy", action="store_true", help="execute with HANDBOOK_HEAVY unset (heavy cells print a notice)")
    ap.add_argument("--timeout", type=int, default=6 * 3600)
    args = ap.parse_args()
    for ch in args.chapters:
        nb = BUILDERS[ch]()
        path = HERE / f"{ch}.ipynb"
        if args.execute:
            import os
            from nbclient import NotebookClient
            if not args.skip_heavy:
                os.environ["HANDBOOK_HEAVY"] = "1"
            client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
            client.execute()
            strip_progress_noise(nb)
        nbformat.write(nb, path)
        print(f"wrote {path}{' (executed)' if args.execute else ''}")


if __name__ == "__main__":
    main()
