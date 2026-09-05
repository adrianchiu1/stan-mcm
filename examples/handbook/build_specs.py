"""Write the handbook example specs (S8 WP4): one directory per example
under ``examples/handbook/`` with a ``spec.yaml`` built through the
authoring API (``au.var``, ``au.minnesota_priors``, plain ``au.Model``) so
the VAR specs -- dozens of parameters -- are generated, reproducible and
identical to what a notebook would author. Run after ``make_data.py``:

    python examples/handbook/build_specs.py

Each example's README records what the handbook does, what is expressed
here and what is NOT (the prior families the handbook uses that the
authored grammar replaces), and -- once ``run_smoke.py`` has run -- the
run identity, the fit-time mirror check and the fast validation tier.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from macrotoolkit import api as mtk
from macrotoolkit import authoring as au

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

#: Smoke-run sampler settings (short chains; the handbook's runs are
#: 5,000-40,000 Gibbs sweeps, ours are NUTS with a marginal likelihood).
SMOKE = {"chains": 2, "warmup": 300, "sampling": 300, "seed": 20260905}
SMOKE_BIG = {"chains": 2, "warmup": 200, "sampling": 200, "seed": 20260905}


def _write(name: str, spec, readme: str) -> None:
    """Write the spec and the README's narrative; a README's appended
    "Smoke run record" section (written by ``run_smoke.py``) is kept."""
    d = HERE / name
    d.mkdir(exist_ok=True)
    (d / "spec.yaml").write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))
    record = ""
    path = d / "README.md"
    if path.is_file() and "\n## Smoke run record" in path.read_text():
        record = "\n## Smoke run record" + path.read_text().split("\n## Smoke run record", 1)[1]
    path.write_text(readme.strip() + "\n" + record)
    print(f"wrote {d / 'spec.yaml'}")


def _data(file: str, mapping: dict, **kw) -> dict:
    return {"file": f"../data/{file}", "date_column": "date", "mapping": mapping, **kw}


# ---------------------------------------------------------------------------
# Chapter 1
# ---------------------------------------------------------------------------


def ch1_ar2() -> None:
    m = au.Model(
        "ch1_ar2", observables=["infl"],
        measurement=["infl = c + b1*infl[-1] + b2*infl[-2] + e"],
        parameters={"c": au.normal(0.0, 1.0), "b1": au.normal(0.0, 1.0), "b2": au.normal(0.0, 1.0), "sigma": au.half_normal(2.0)},
        shocks={"e": au.shock("sigma")},
    )
    spec = mtk.spec("authored", options=m, data=_data("ch1_inflation.csv", {"infl": "inflation"}), sampler=SMOKE, outputs={"horizon": 12, "irf_horizon": 12})
    _write("ch1_ar2", spec, """
# Chapter 1, examples 1-2: an AR(2) for US inflation with a constant (+ a 3-year forecast)

Handbook: `Y_t = c + b1 Y_{t-1} + b2 Y_{t-2} + v_t`, Gibbs with `B ~ N(0, I)`
and `sigma^2 ~ IG(T0 = 1, D0 = 0.1)`, stability enforced by rejection;
example 2 adds the 12-quarter forecast fan.

Here (S8 E0/E1): the constant is a PARAMETER with the handbook's `N(0, 1)`
prior (`c`), the lag coefficients likewise; the inverse-gamma on `sigma^2`
is replaced by `half_normal(2)` on `sigma` (the authored prior menu);
stationarity is not imposed (the posterior mass on explosive roots is
negligible for this series). No state: the KF runs with `n = 0` and is
exactly the Gaussian regression likelihood. The fan chart is the
engine's forward simulation (`outputs.horizon: 12`). The alternative
route -- the constant as a shock-free state `c = c[-1]` with `init(0, 1)`
-- gives the same marginal likelihood (the E0 oracle) with `c` integrated
by the filter instead of sampled.
""")


def ch1_ar2_ar1err() -> None:
    # Y_t = c + b1 Y_{t-1} + b2 Y_{t-2} + v_t, v_t = rho v_{t-1} + e_t  ->  quasi-differenced exact form:
    # Y_t = c (1 - rho) + (b1 + rho) Y_{t-1} + (b2 - rho b1) Y_{t-2} - rho b2 Y_{t-3} + e_t
    m = au.Model(
        "ch1_ar2_ar1err", observables=["infl"],
        measurement=["infl = c*(1 - rho) + (b1 + rho)*infl[-1] + (b2 - rho*b1)*infl[-2] - rho*b2*infl[-3] + e"],
        parameters={"c": au.normal(0.0, 1.0), "b1": au.normal(0.0, 1.0), "b2": au.normal(0.0, 1.0), "rho": au.normal(0.0, 1.0, lower=-1.0, upper=1.0), "sigma": au.half_normal(2.0)},
        shocks={"e": au.shock("sigma")},
    )
    spec = mtk.spec("authored", options=m, data=_data("ch1_inflation.csv", {"infl": "inflation"}), sampler=SMOKE, outputs={"horizon": 12, "irf_horizon": 12})
    _write("ch1_ar2_ar1err", spec, """
# Chapter 1, example 3: the AR(2) with AR(1) disturbances, in its exact substituted form

Handbook: `Y_t = c + b1 Y_{t-1} + b2 Y_{t-2} + v_t`, `v_t = rho v_{t-1} + e_t`,
estimated by the Cochrane-Orcutt-style Gibbs blocks (quasi-differenced
regression for `B`, a regression of the residual on its lag for `rho`).

Here: the exactly equivalent quasi-differenced equation
`infl = c(1 - rho) + (b1 + rho) infl[-1] + (b2 - rho b1) infl[-2] - rho b2 infl[-3] + e`
-- coefficient EXPRESSIONS over the parameters, which the grammar accepts
(the model stays linear in the series) -- with the handbook's `N(0, 1)`
priors on `c`, `b1`, `b2` and `rho` (`rho` truncated to (-1, 1), the
stationarity the handbook imposes by rejection). One lag more of
pre-sample data is consumed (`lag_depth = 3`).
""")


# ---------------------------------------------------------------------------
# Chapter 2
# ---------------------------------------------------------------------------


def ch2_bivar_minnesota() -> None:
    df = pd.read_csv(DATA / "ch2_datain.csv").rename(columns={"gdp_growth": "y", "inflation": "pi"})
    priors = au.minnesota_priors(df, ["y", "pi"], 2, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0, own_mean=1.0)
    m = au.var("ch2_bivar_minnesota", ["y", "pi"], 2, priors=priors)
    spec = mtk.spec("authored", options=m, data=_data("ch2_datain.csv", {"y": "gdp_growth", "pi": "inflation"}), sampler=SMOKE, outputs={"horizon": 12, "irf_horizon": 12})
    _write("ch2_bivar_minnesota", spec, """
# Chapter 2, example 1: a bivariate VAR(2) with a Minnesota prior (+ a 3-year forecast)

Handbook: US GDP growth and inflation 1948Q1-2010Q4, `lambda1..4 = 1`,
prior mean 1 on the own first lag, the diagonal `H` from the AR(1)
residual sds `s1`, `s2`; `Sigma ~ IW(I, N + 1)`; a 12-quarter forecast.

Here: `au.var(..., priors=au.minnesota_priors(...))` -- the recursive
(Cholesky-ordered `y`, `pi`) form with the INDEPENDENT-NORMAL Minnesota
prior on the recursive-form lag coefficients and constants, exactly the
handbook's `H` arithmetic (tests/test_s8_var.py pins it against example
1's `H`); the contemporaneous coefficient `a0_pi_y ~ N(0, 10)`, the
orthogonal shock scales `half_normal(5)` in place of the inverse-Wishart
(`macrotoolkit/authoring/var.py` states the correspondence). Forecast
fan = the engine's forward simulation.
""")


def ch2_var4_monthly_cholesky() -> None:
    obs = ["ffr", "bond10y", "unemployment", "inflation"]
    df = pd.read_csv(DATA / "ch2_dataus_monthly.csv")
    priors = au.minnesota_priors(df, obs, 2, lambda1=0.1, lambda2=1.0, lambda3=0.05, lambda4=1.0, own_mean=0.95)
    # example2.m: the FFR equation's cross-lag coefficients get prior variance 1e-9 ("close to zero").
    for k in (1, 2):
        for other in obs[1:]:
            priors[f"b_ffr_{other}_{k}"] = au.normal(0.0, 1e-9**0.5)
    m = au.var("ch2_var4_monthly_cholesky", obs, 2, priors=priors)
    spec = mtk.spec("authored", options=m, data=_data("ch2_dataus_monthly.csv", {o: o for o in obs}), sampler=SMOKE, outputs={"horizon": 12, "irf_horizon": 36})
    _write("ch2_var4_monthly_cholesky", spec, """
# Chapter 2, example 2: a 4-variable monthly VAR(2) with Cholesky-identified IRFs

Handbook: FFR, 10-year yield, unemployment, inflation (monthly, 46 rows
ending 2010m12), `lambda1 = 0.1`, `lambda3 = 0.05`, `lambda4 = 1`, own
first-lag mean 0.95, the FFR equation's cross-lag coefficients shrunk to
zero (prior variance 1e-9); IRFs to a government-bond-yield shock through
`A0 = chol(Sigma)`.

Here: the same Minnesota arithmetic through `au.minnesota_priors` (the
1e-9 entries applied by name), the recursive form in the handbook's
ordering, and the engine's structural IRFs -- under the ordering they ARE
the Cholesky IRFs (`irf_horizon: 36`); the `e_bond10y` shock is the
handbook's yield shock. The sample is 46 months, so the smoke chains are
short and the posterior is prior-dominated exactly as in the handbook.
""")


def ch2_steady_state() -> None:
    df = pd.read_csv(DATA / "ch2_datain.csv").rename(columns={"gdp_growth": "y", "inflation": "pi"})
    priors = {k: v for k, v in au.minnesota_priors(df, ["y", "pi"], 2, own_mean=1.0).items() if not k.startswith("c_")}
    m = au.var("ch2_steady_state", ["y", "pi"], 2, intercept="steady_state", priors=priors,
               steady_state_init={"y": au.init(1.0, 0.001**0.5), "pi": au.init(1.0, 0.001**0.5)})
    spec = mtk.spec("authored", options=m, data=_data("ch2_datain.csv", {"y": "gdp_growth", "pi": "inflation"}), sampler=SMOKE, outputs={"horizon": 40, "irf_horizon": 12})
    _write("ch2_steady_state", spec, """
# Chapter 2, example 3: the steady-state (Villani 2009) VAR

Handbook: the VAR in deviations from long-run means `mu` with a prior
`mu ~ N((1, 1), 0.001 I)`, Minnesota priors on the lag coefficients (no
constant), a Gibbs block for `mu` (Villani's Appendix A), a 10-year
forecast.

Here: the long-run means are shock-free CONSTANT STATES `mu_y`, `mu_pi`
(`au.var(..., intercept="steady_state")`, S8 E0) whose initial condition
IS the handbook's prior (`init(1.0, sqrt(0.001))`); the intercept of each
recursive equation is the Villani identity written with parameter-
expression coefficients on the states, and the filter integrates `mu`
exactly (tests/test_s8_var.py: the KF likelihood equals the closed-form
marginal over `mu`). No new machinery. Smoothed `mu` paths are flat
lines (a constant state) at the posterior of the long-run means.
""")


def ch2_signs_11var() -> None:
    obs = ["ffr", "gdp_growth", "cpi_inflation", "pce_growth", "unemployment", "investment", "net_exports", "m2", "bond10y", "stock_growth", "yen_dollar"]
    df = pd.read_csv(DATA / "ch2_usdata_11var.csv")
    priors = au.minnesota_priors(df, obs, 2, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0, own_mean=1.0)
    m = au.var("ch2_signs_11var", obs, 2, priors=priors)
    spec = mtk.spec("authored", options=m, data=_data("ch2_usdata_11var.csv", {o: o for o in obs}), sampler=SMOKE_BIG,
                    outputs={"horizon": 8, "irf_horizon": 36, "smoother_draws": {"thin": 4}, "prior_predictive_draws": 20})
    _write("ch2_signs_11var", spec, """
# Chapter 2, examples 5-7: the 11-variable VAR(2) with sign restrictions

Handbook: 11 US quarterly series (160 rows ending 2010Q4), the
dummy-observation (Banbura et al.) prior with `lambda = 1`, `tau = 10
lambda`, `epsilon = 1`; a monetary-policy shock identified by sign
restrictions on impact (R up; GDP, inflation, consumption, investment,
money down; unemployment up) via `Q = getqr(randn)` rotations of
`chol(Sigma)`; example 6 scans the rows of `Q A0` for the pattern,
example 7 keeps the rotation closest to the median of 100 accepted ones.

Here: the recursive VAR(2) with INDEPENDENT-NORMAL Minnesota priors at
`lambda1..4 = 1` (the dummy-observation prior -- its sum-of-coefficients
and co-persistence dummies included -- is NOT expressible in the authored
grammar; this is the closest prior the menu offers and is stated as such),
sampler reduced for the smoke run (319 parameters), and the sign
restrictions as a POST-PROCESSOR over the engine's Cholesky IRF array:
`macrotoolkit.postprocess.sign_restricted_irfs` with the handbook's
seven impact restrictions, column flips allowed, the identified shock
first (examples 5/6), or `closest_to_median=100` (example 7). See
`run_smoke.py` for the call.
""")


def ch2_conditional() -> None:
    df = pd.read_csv(DATA / "ch2_datain.csv").rename(columns={"gdp_growth": "y", "inflation": "pi"})
    m = au.var("ch2_conditional", ["y", "pi"], 2, priors=au.minnesota_priors(df, ["y", "pi"], 2, own_mean=1.0))
    spec = mtk.spec("authored", options=m, data=_data("ch2_datain.csv", {"y": "gdp_growth", "pi": "inflation"}), sampler=SMOKE, outputs={"horizon": 3, "irf_horizon": 12})
    _write("ch2_conditional", spec, """
# Chapter 2, example 8: conditional forecasts (Waggoner-Zha) from a bivariate VAR(2)

Handbook: OLS/flat-prior Gibbs for the VAR (the conditional forecast is
appended to the data each sweep), inflation constrained to `(1, 1, 1)`
over 3 quarters, the restricted structural shocks `N(R'(RR')^+ r, I -
R'(RR')^+ R)` with `R` from the Cholesky IRFs.

Here: the recursive VAR(2) with the example-1 Minnesota priors (the
handbook's flat prior puts no mass on stationary draws, so the fast
tier's HD-identity gate -- which needs stationary prior points -- cannot
run under it; stated) and no data augmentation ( the posterior conditions on the observed sample only --
the handbook's augmentation step feeds the conditional path back into
the VAR posterior, a Gibbs device this KF-marginal likelihood does not
replicate; stated) -- and `macrotoolkit.postprocess.conditional_forecast_for_run`
with `{("pi", 0): 1, ("pi", 1): 1, ("pi", 2): 1}` -- the conditioned
series reproduces the path exactly, GDP growth carries the implied
distribution (tests/test_s8_postprocess.py). The specification is
example 1's model under a different name (the post-processor is the
example), so its posterior record coincides with `ch2_bivar_minnesota`'s.
""")


# ---------------------------------------------------------------------------
# Chapter 3
# ---------------------------------------------------------------------------


def ch3_uc_trend_cycle() -> None:
    m = au.Model(
        "ch3_uc_trend_cycle", observables=["Y"], measurement=["Y = C + tau"],
        transition=["C = c0 + a1*C[-1] + a2*C[-2] + e1", "tau = tau[-1] + e2"],
        parameters={"c0": au.normal(0.0, 0.5), "a1": au.normal(1.0, 0.3), "a2": au.normal(-0.3, 0.3), "s1": au.half_normal(1.0), "s2": au.half_normal(0.5)},
        shocks={"e1": au.shock("s1"), "e2": au.shock("s2")},
        initial_state={"C": au.init(0.0, 2.0), "tau": au.init(au.first_obs("Y"), 5.0)},
    )
    spec = mtk.spec("authored", options=m, data=_data("ch1_inflation.csv", {"Y": "inflation"}), sampler={**SMOKE, "chains": 4, "warmup": 500, "sampling": 500, "adapt_delta": 0.9}, outputs={"horizon": 12, "irf_horizon": 20})
    _write("ch3_uc_trend_cycle", spec, """
# Chapter 3, §2 (equations 2.6-2.7): the unobserved-components trend-cycle model

Handbook: `Y_t = C_t + tau_t` EXACTLY (no measurement error), `tau_t =
tau_{t-1} + e2_t`, `C_t = c + a1 C_{t-1} + a2 C_{t-2} + e1_t`, in the
state-space form with state `(C_t, tau_t, C_{t-1})`, the constant in the
transition intercept vector, a singular `R = 0` and a `Q` with a possible
`e1`-`e2` covariance; a generic example (no script).

Here (S8 E1 + E3): the shock-free measurement row (`R = 0`, allowed
because the row loads stochastic states -- the PD proof in
plans/S8-plan.md), the cycle's constant as a transition DRIFT (the
implicit unit state `_const`), orthogonal shocks (the handbook's `Q`
off-diagonal is not expressible: a shared/correlated state shock is
outside the grammar; stated). Applied to US inflation (trend inflation +
an AR(2) cycle -- the handbook gives no dataset for §2). The compiled
matrices equal the hand-built (2.6)-(2.7) construction and the DK draws
reproduce `Y = C + tau` per period (tests/test_s8_grammar.py).
""")


def ch3_dfm_uk_panel() -> None:
    levels = pd.read_csv(DATA / "ch3_uk_panel_levels.csv")
    index = pd.read_csv(DATA / "ch3_uk_panel_index.csv")
    # example4.m: dindex 1 -> 100 * diff(log); 3 -> diff; else levels (row 2 on); then standardise.
    out = pd.DataFrame({"date": levels["date"].iloc[1:].to_numpy()})
    for _, row in index.iterrows():
        s = levels[row["series"]].to_numpy(dtype=float)
        if row["transform"] == 1:
            d = 100.0 * np.diff(np.log(s))
        elif row["transform"] == 3:
            d = np.diff(s)
        else:
            d = s[1:]
        out[row["series"]] = (d - d.mean()) / d.std(ddof=1)
    out.to_csv(DATA / "ch3_uk_panel_dfm.csv", index=False)
    series = list(index["series"])
    factors = ["f1", "f2", "f3"]
    measurement, params, shocks = [], {}, {}
    for i, s in enumerate(series):
        if i < 3:  # identification: the top 3x3 loading block is the identity
            measurement.append(f"{s} = {factors[i]} + e_{s}")
        else:
            terms = " + ".join(f"l_{s}_{f}*{f}" for f in factors)
            measurement.append(f"{s} = {terms} + e_{s}")
            for f in factors:
                params[f"l_{s}_{f}"] = au.normal(0.0, 1.0)
        params[f"sd_{s}"] = au.half_normal(1.0)
        shocks[f"e_{s}"] = au.shock(f"sd_{s}")
    transition = []
    for f in factors:
        terms = " + ".join(f"b_{f}_{g}_{k}*{g}[-{k}]" for k in (1, 2) for g in factors)
        transition.append(f"{f} = {terms} + eta_{f}")
        for k in (1, 2):
            for g in factors:
                params[f"b_{f}_{g}_{k}"] = au.normal(0.5 if (g == f and k == 1) else 0.0, 0.5)
        params[f"s_{f}"] = au.half_normal(1.0)
        shocks[f"eta_{f}"] = au.shock(f"s_{f}")
    m = au.Model("ch3_dfm_uk_panel", observables=series, measurement=measurement, transition=transition, parameters=params, shocks=shocks,
                 initial_state={f: au.init(0.0, 1.0) for f in factors})
    spec = mtk.spec("authored", options=m, data=_data("ch3_uk_panel_dfm.csv", {s: s for s in series}), sampler={**SMOKE_BIG, "warmup": 150, "sampling": 150},
                    outputs={"horizon": 4, "irf_horizon": 16, "smoother_draws": {"thin": 5}, "prior_predictive_draws": 10})
    _write("ch3_dfm_uk_panel", spec, """
# Chapter 3, example 4 (the DFM block): three factors from the 40-series UK panel

Handbook: a FAVAR -- 40 UK series (log-differenced / differenced per
`index.xls`, standardised) load on 3 factors (plus the policy rate for
the "fast" series); the factors and the rate follow a VAR(2); Gibbs with
flat priors, factors drawn by Carter-Kohn, identification by fixing the
top 3x3 loading block to the identity.

Here: the DFM part -- every series `= l_i1 f1 + l_i2 f2 + l_i3 f3 + e_i`
(the first three series load their own factor with unit coefficient, the
identity block), the factors a VAR(2) in the STATE with orthogonal shocks
(the handbook's full `Sigma` on the factor VAR is a correlated state
shock, outside the grammar; the FAVAR's rate block -- the rate as an
observable inside the factor VAR -- would need the E5 substitution on
the state side and is left for a follow-up), `N(0, 1)` loadings,
`half_normal(1)` idiosyncratic sds. `m = 40`, `n = 6`: the smoke run's
chains are short; the KF's 40x40 innovation Cholesky per period is the
cost. Dates: the handbook gives none; `make_data.py` uses a quarterly
period index ending 2006Q1 (a label, not a documented sample).
""")


if __name__ == "__main__":
    ch1_ar2()
    ch1_ar2_ar1err()
    ch2_bivar_minnesota()
    ch2_var4_monthly_cholesky()
    ch2_steady_state()
    ch2_signs_11var()
    ch2_conditional()
    ch3_uc_trend_cycle()
    ch3_dfm_uk_panel()
