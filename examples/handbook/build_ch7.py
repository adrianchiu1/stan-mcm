"""Chapter 7 bridge model (companion): the Mumtaz-Surico dynamic factor model with world and country
factors on the handbook's generated data (Code2017/CHAPTER7/data/dataxx01.mat), with CONSTANT factor
AR coefficients and stochastic volatility on the factor shocks -- the part of the Chapter 7 model the
toolkit's class covers today (time-varying AR coefficients are extension E6). Writes
data/ch7_dfm_sim.csv (the standardised panel, first NCOUNTRIES countries) and ch7_dfm_sv/spec.yaml.
    python examples/handbook/build_ch7.py [--countries 2]
"""
from __future__ import annotations
import argparse, io, zipfile
from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy.io import loadmat
from macrotoolkit import api as mtk
from macrotoolkit import authoring as au

HERE = Path(__file__).resolve().parent
ZIP = HERE.parents[1] / "docs" / "applied-bayesian-economics-code-files.zip"

def main(ncountries: int) -> None:
    with zipfile.ZipFile(ZIP) as z:
        raw = z.read("Code2017/CHAPTER7/data/dataxx01.mat")
    m = loadmat(io.BytesIO(raw))
    dataS, index = m["dataS"], m["index"].ravel().astype(int)
    keep = np.isin(index, np.unique(index)[:ncountries])
    X = dataS[:, keep]; idx = index[keep]
    X = (X - X.mean(0)) / X.std(0, ddof=1)                      # example1.m: standardise(dataS)
    T, N = X.shape
    names = [f"c{c}_s{j}" for c in np.unique(idx) for j in range(1, int((idx == c).sum()) + 1)]
    dates = pd.period_range("1960Q1", periods=T, freq="Q").to_timestamp()   # a label: the handbook's data are simulated
    df = pd.DataFrame(X, columns=names); df.insert(0, "date", dates)
    (HERE / "data").mkdir(exist_ok=True); df.to_csv(HERE / "data" / "ch7_dfm_sim.csv", index=False)

    countries = list(np.unique(idx))
    factors = ["fw"] + [f"fc{c}" for c in countries]
    measurement, params, shocks, init = [], {}, {}, {}
    first_of_country = {c: True for c in countries}
    for i, s in enumerate(names):
        c = idx[i]
        if i == 0:                                   # identification (generate_data.m line 99): unit loading blocks
            eq = f"{s} = fw + e_{s}"
        elif first_of_country[c]:
            eq = f"{s} = l_{s}_fw*fw + fc{c} + e_{s}"; params[f"l_{s}_fw"] = au.normal(0.0, 1.0)
        else:
            eq = f"{s} = l_{s}_fw*fw + l_{s}_fc*fc{c} + e_{s}"
            params[f"l_{s}_fw"] = au.normal(0.0, 1.0); params[f"l_{s}_fc"] = au.normal(0.0, 1.0)
        first_of_country[c] = False
        measurement.append(eq)
        params[f"se_{s}"] = au.half_normal(1.0); shocks[f"e_{s}"] = au.shock(f"se_{s}")
    transition = []
    for f in factors:                                # AR(2) factors, constant coefficients, SV on the factor shock
        transition.append(f"{f} = a1_{f}*{f}[-1] + a2_{f}*{f}[-2] + eta_{f}")
        params[f"a1_{f}"] = au.normal(0.5, 0.3); params[f"a2_{f}"] = au.normal(0.0, 0.3)
        shocks[f"eta_{f}"] = au.sv(sigma_h=0.2, h0_sd=1.0, mu_h0=0.0)
        init[f] = au.init(0.0, 1.0)
    model = au.Model("ch7_dfm_sv", observables=names, measurement=measurement, transition=transition,
                     parameters=params, shocks=shocks, initial_state=init)
    print(model.describe())
    spec = mtk.spec("authored", options=model,
                    data={"file": "../data/ch7_dfm_sim.csv", "date_column": "date", "mapping": {s: s for s in names}},
                    sampler={"chains": 2, "warmup": 200, "sampling": 200, "seed": 20260905, "adapt_delta": 0.9},
                    outputs={"horizon": 8, "irf_horizon": 12, "smoother_draws": {"thin": 4}, "prior_predictive_draws": 20})
    d = HERE / "ch7_dfm_sv"; d.mkdir(exist_ok=True)
    (d / "spec.yaml").write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))
    print(f"wrote {d/'spec.yaml'}: N = {N} series, {len(factors)} factors, T = {T}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--countries", type=int, default=2)
    main(ap.parse_args().countries)
