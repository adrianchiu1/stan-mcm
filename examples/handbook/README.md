# The CCBS handbook examples (Blake & Mumtaz 2017, Chapters 1-3) as authored specs

S8 (plans/S8-plan.md, WP4). Every example is a plain `spec.yaml` built
through the authoring API by `build_specs.py` (the VAR specs are
generated, not hand-written) on the CSVs `make_data.py` converts from
the handbook's code zip (`docs/applied-bayesian-economics-code-files.zip`,
`Code2017/`); `run_smoke.py` fits each with short chains, runs the fast
validation tier and the post-processors, and appends the run record to
the example's README.

```
python examples/handbook/make_data.py      # .xls -> data/*.csv (xlrd)
python examples/handbook/build_specs.py    # specs (+ the DFM panel transform)
python examples/handbook/run_smoke.py      # smoke runs + records (or name the examples)
```

| Example | Handbook | Grammar / layer | Prior / identification differences (stated in each README) |
|---|---|---|---|
| `ch1_ar2` | Ch. 1 ex. 1-2 | E0/E1: a regression with a constant, no state (`n = 0`), forecast fan | half-normal scale for the inverse-gamma on sigma^2 |
| `ch1_ar2_ar1err` | Ch. 1 ex. 3 | the exact quasi-differenced form with coefficient expressions | rho truncated to (-1, 1) instead of rejection |
| `ch2_bivar_minnesota` | Ch. 2 ex. 1 | E5 recursive VAR(2), `au.minnesota_priors`, forecast | independent-normal Minnesota on the recursive form; half-normal scales for the IW |
| `ch2_var4_monthly_cholesky` | Ch. 2 ex. 2 | E5, Cholesky IRFs = the engine's structural IRFs under the ordering | the 1e-9 cross-lag shrinkage applied by name |
| `ch2_steady_state` | Ch. 2 ex. 3 | E0: long-run means as constant states (Villani) | the mu prior is the states' initial condition |
| `ch2_signs_11var` | Ch. 2 ex. 5-7 | E5 + `postprocess.sign_restrictions` | Minnesota normals in place of the dummy-observation prior; sampler reduced |
| `ch2_conditional` | Ch. 2 ex. 8 | E5 + `postprocess.conditional_forecast` | no data augmentation of the conditional path |
| `ch3_uc_trend_cycle` | Ch. 3 §2 (2.6)-(2.7) | E1 drift + E3 shock-free row (`R = 0`) | orthogonal shocks (no Q off-diagonal); applied to US inflation |
| `ch3_dfm_uk_panel` | Ch. 3 ex. 4 (DFM part) | 40 measurement equations on 3 VAR(2) factor states | orthogonal factor shocks; the FAVAR rate block left for a follow-up |

The data files carry reconstructed dates where the handbook gives none
(`make_data.py` records the assumption per file). `runs/` and
`validation/` are gitignored; the records in each README are the
evidence.
