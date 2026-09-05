"""Smoke-run the handbook examples (S8 WP4): fit each spec with its short
chains, run the fast validation tier, apply the post-processors where the
example calls for them, and append a run record to the example's README.

    python examples/handbook/run_smoke.py [name ...]     # default: all, in order

Records: run id, the fit-time mirror check, the diagnostics verdict, the
fast tier (mirror + HD identity), and per example the post-processor
readout (sign-restricted IRF bands / the conditional path).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from macrotoolkit import api as mtk
from macrotoolkit.run import REPO_ROOT

HERE = Path(__file__).resolve().parent
ORDER = ["ch1_ar2", "ch1_ar2_ar1err", "ch2_bivar_minnesota", "ch2_var4_monthly_cholesky", "ch2_steady_state", "ch2_conditional",
         "ch3_uc_trend_cycle", "ch2_signs_11var", "ch3_dfm_uk_panel"]


def _postprocess(name: str, run) -> list[str]:
    lines = []
    obs_names = tuple(run.results().meta.obs_names)
    if name == "ch2_signs_11var":
        from macrotoolkit.postprocess import SignRestriction, irf_array_from_draws, sign_restricted_irfs

        irf = run.outputs().compute("irf")
        arr, targets, shocks = irf_array_from_draws(irf, targets=obs_names)
        restr = [SignRestriction("mp", v, s, (0,)) for v, s in (("ffr", +1), ("gdp_growth", -1), ("cpi_inflation", -1), ("pce_growth", -1), ("unemployment", +1), ("investment", -1), ("m2", -1))]
        t0 = time.time()
        sr = sign_restricted_irfs(arr, restr, targets, np.random.default_rng(20260905), max_tries=2000)
        lines.append(f"- sign restrictions (examples 5/6): {sr.irf.shape[0]} of {arr.shape[0]} posterior draws accepted within 2000 tries "
                     f"({len(sr.rejected_draws)} rejected; mean tries {sr.n_tries.mean():.0f}; {time.time() - t0:.0f}s); "
                     f"median impact of the MP shock on gdp_growth {sr.bands('mp', 'gdp_growth')[1][0]:+.3f}, on ffr {sr.bands('mp', 'ffr')[1][0]:+.3f}.")
    if name == "ch2_conditional":
        from macrotoolkit.postprocess import conditional_forecast_for_run

        cf = conditional_forecast_for_run(run, {("pi", 0): 1.0, ("pi", 1): 1.0, ("pi", 2): 1.0}, horizon=3, draws_per_posterior_draw=2)
        med = np.median(cf.draws, axis=0)
        lines.append(f"- conditional forecast (example 8): inflation held at (1, 1, 1) -> median GDP growth path {np.round(med[:, 0], 3).tolist()} "
                     f"(unconditional {np.round(cf.unconditional[:, 0], 3).tolist()}); the pi path reproduces (1, 1, 1) to {np.max(np.abs(cf.draws[:, :, 1] - 1.0)):.1e}.")
    if name == "ch2_var4_monthly_cholesky":
        irf = run.outputs().compute("irf")
        r = irf.responses["e_bond10y"]
        lines.append("- Cholesky IRFs to the bond-yield shock (example 2), impact medians: " + ", ".join(f"{o} {np.median(r[o][:, 0]):+.3f}" for o in obs_names) + ".")
    if name == "ch2_steady_state":
        sd = run.outputs().compute("states")
        lines.append("- long-run means (constant states, posterior median of the smoothed path): " + ", ".join(f"{k} {np.median(v):.3f}" for k, v in sd.states.items()) + ".")
    return lines


def main(names: list[str]) -> None:
    for name in names:
        spec_path = HERE / name / "spec.yaml"
        t0 = time.time()
        run = mtk.fit(str(spec_path))
        elapsed = time.time() - t0
        mc = run.mirror_check
        diag = json.loads((run.run_dir / "diagnostics.json").read_text())
        val = mtk.validate(spec_path, tier="fast", out_root=REPO_ROOT / "validation")
        lines = [
            "",
            "## Smoke run record",
            "",
            f"- run `{run.hash}` ({run.spec.sampler.chains} chains x {run.spec.sampler.warmup}/{run.spec.sampler.sampling}, {f'{elapsed:.0f}s' if elapsed > 2 else 'reused'}): "
            f"fit-time mirror check max |Stan - Python| = {mc['max_abs_diff']:.2e} (relative {mc.get('max_rel_diff', float('nan')):.1e}) over {mc['n_points']} prior draws, "
            f"diagnostics verdict {diag.get('verdict')}" + (f" ({'; '.join(diag['reasons'])})" if diag.get("reasons") else "") + ".",
            f"- fast validation tier: {val.verdict} -- " + "; ".join(f"{g.name} {g.verdict} ({g.summary})" for g in val.gates),
        ]
        lines += _postprocess(name, run)
        readme = HERE / name / "README.md"
        text = readme.read_text().split("\n## Smoke run record")[0].rstrip()
        readme.write_text(text + "\n" + "\n".join(lines) + "\n")
        print(f"{name}: run {run.hash} mirror {mc['max_abs_diff']:.2e} validation {val.verdict} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1:] or ORDER)
