"""Companion-only specs (beyond S8/S9's build_specs.py): the Chapter 5 stochastic-volatility
model for UK inflation in the handbook's own mean-only form (E0: no state) and in the
unobserved-components form the companion first ran on the S7 branch.
    python examples/handbook/build_companion_specs.py
"""
from __future__ import annotations
from pathlib import Path
import yaml
from macrotoolkit import api as mtk
from macrotoolkit import authoring as au

HERE = Path(__file__).resolve().parent
SMOKE = {"chains": 4, "warmup": 500, "sampling": 500, "seed": 20260905, "adapt_delta": 0.9}
DATA = {"file": "../data/ch5_uk_inflation.csv", "date_column": "date", "mapping": {"infl": "infl"}}


def _write(name: str, spec, readme: str) -> None:
    d = HERE / name; d.mkdir(exist_ok=True)
    (d / "spec.yaml").write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))
    (d / "README.md").write_text(readme.strip() + "\n")
    print(f"wrote {d / 'spec.yaml'}")


def ch5_sv_uk_inflation() -> None:
    m = au.Model(
        "ch5_sv_uk_inflation", observables=["infl"],
        measurement=["infl = c + e"],
        parameters={"c": au.normal(0.0, 5.0)},
        shocks={"e": au.sv(sigma_h=0.3, h0_sd=3.0, mu_h0=au.log_var_diff("infl", 1.0))},
    )
    spec = mtk.spec("authored", options=m, data=DATA, sampler=SMOKE, outputs={"horizon": 12, "irf_horizon": 12})
    _write("ch5_sv_uk_inflation", spec, """
# Chapter 5, example 4: the stochastic-volatility model for UK inflation, in the handbook's own form

Handbook: `y_t = e_t`, `e_t ~ N(0, h_t)`, `ln h_t = ln h_{t-1} + sqrt(g) u_t`, `g ~ IG(0.01, 1)`,
`ln h_0 ~ N(mubar, 10)` with `mubar` the log variance of a 10-quarter training sample; the
volatility path drawn date by date by the Jacquier-Polson-Rossi independence Metropolis step.
Data: the UK price level 1914Q1-2011Q1, annual inflation `100 (ln P_t - ln P_{t-4})`.

Here: a constant mean `c` (E1) and the SV block on the measurement shock -- no state at all
(E0). `g ~ IG` is replaced by `half_normal(0.3)` on the log-variance random walk's standard
deviation `sigma_h`; `mubar` by the `log_var_diff` anchor (no training sample discarded) with
`h0_sd = 3`. `h` is the log VARIANCE in both toolkits; the shock's standard deviation is
`exp(h/2)`. The log-variance path is sampled by NUTS in the non-centered parameterization.
""")


def ch5_sv_ucsv_form() -> None:
    m = au.Model(
        "ch5_sv_ucsv_form", observables=["infl"],
        measurement=["infl = tau + e"], transition=["tau = tau[-1] + eta"],
        parameters={"sigma_eta": au.half_normal(0.5)},
        shocks={"eta": au.shock("sigma_eta"), "e": au.sv(sigma_h=0.3, h0_sd=3.0, mu_h0=au.log_var_diff("infl", 1.0))},
        initial_state={"tau": au.init(au.first_obs("infl"), 5.0)},
    )
    spec = mtk.spec("authored", options=m, data=DATA, sampler=SMOKE, outputs={"horizon": 12, "irf_horizon": 12})
    _write("ch5_sv_ucsv_form", spec, """
# Chapter 5, example 4 in unobserved-components form: a random-walk level with an SV transitory shock

The companion's first check of the handbook's SV model (run on the S7 branch as `03d419822418`
before the zero-state extension E0 existed): the level is a random walk `tau`, the transitory
shock `e` carries the log-variance random walk. The same SV block and anchors as
`ch5_sv_uk_inflation`; `sigma_eta ~ half_normal(0.5)`; `tau_0 ~ N(infl_1, 5^2)`.
""")


if __name__ == "__main__":
    ch5_sv_uk_inflation()
    ch5_sv_ucsv_form()
