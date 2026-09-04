# Notebook API examples (S6 WP1)

The full macrotoolkit workflow from a Jupyter notebook -- no YAML files, no
CLI -- through `from macrotoolkit import api as mtk`.

| Notebook | What it shows | Samples? |
|---|---|---|
| `lw_sv_from_archive.ipynb` | Spec construction (hash parity with the YAML spec), the `Run` handle over the archived reference run `runs-archive/a00958509083`, results objects, inline trend-cycle / IRF / fan / HD / prior-predictive figures, the parameter table, `run.report()`, and the `mtk.sweep` call shape | No (archived run) |
| `ucsv_us_inflation.ipynb` | Family #2 (UCSV, S6 WP2) end to end on US quarterly core PCE inflation: DataFrame in, `mtk.fit`, mirror-check record, trend / volatility / fan / HD figures, report | Yes (~1-2 min) |
| `authored_uc_gap.ipynb` | **S7's end-state walkthrough**: a bivariate UC output-gap model (AR(2) gap STATE with SV on the demand shock, accelerationist Phillips curve) that does not exist in the codebase, written as equations with `macrotoolkit.authoring`, compiled, estimated on the in-repo US data, swept over `sigma_g`, with states/IRF/HD/fan figures, the live G6 check and its auto-instantiated fast validation tier. Built by `build_authored_notebook.py` (`--execute` re-executes in place) | Yes (3 fits, ~1-2 h) |

All notebooks are committed **executed** (outputs stored) and re-executed
by `tests/test_notebooks.py` (the sampling notebooks under `-m slow`).
Regenerate after an API change with:

```bash
cd examples/notebook_api
python -c "import nbformat; from nbclient import NotebookClient; \
  nb = nbformat.read('lw_sv_from_archive.ipynb', as_version=4); \
  NotebookClient(nb, kernel_name='python3').execute(); \
  nbformat.write(nb, 'lw_sv_from_archive.ipynb')"
```

Authoring surface (`macrotoolkit.authoring`, S7): `Model`, `normal`,
`half_normal`, `beta`, `shock`, `sv`, `log_var_diff`, `first_obs`, `init`,
`forecast`; `mtk.spec("authored", options=model, ...)`.

API surface (all in `macrotoolkit.api`): `spec`, `fit`, `load_run`, `Run`
(`hash`, `spec`, `diagnostics`, `verdict`, `mirror_check`, `idata`,
`results()`, `outputs()`, `param_table()`, `report()`), `sweep`,
`validate`, and the re-exported Pydantic spec models (`RunSpec`,
`DataSpec`, `SamplerSpec`, ...).
