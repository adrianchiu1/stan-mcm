"""macrotoolkit -- Bayesian state-space macroeconometrics toolkit.

Numerical conventions enforced across this package once model families
beyond the S1 `local_level` toy exist (see lw-sv-spec.md §1.1, §1.5):

- `g` (trend growth) is **annualized**; the potential-output transition
  uses `g/4`.
- `h` is **log-variance**; the corresponding shock standard deviation is
  `exp(h/2)`.
- Inflation is `400 * dlog(P)` (annualized q/q log difference, in percent).

`local_level`, S1's toy family, has none of these quantities -- see
`tests/test_units_conventions.py` for where they get real tests once S2
introduces the LW-SV family.
"""

__version__ = "0.1.0"

# The notebook-first public API (S6 WP1). The canonical notebook import is
#     from macrotoolkit import api as mtk
# (`mtk.fit`, `mtk.sweep`, `mtk.report`, ...). A few non-clashing names are
# also reachable lazily on the package itself (`macrotoolkit.fit`); the API
# functions `sweep`/`report`/`results` are NOT, because submodules of the
# same names (`macrotoolkit.sweep`, `macrotoolkit.report`) own those
# attributes once imported.


def __getattr__(name: str):
    import importlib

    if name == "api" or name in _API_NAMES:
        api = importlib.import_module("macrotoolkit.api")
        return api if name == "api" else getattr(api, name)
    raise AttributeError(f"module 'macrotoolkit' has no attribute {name!r}")


_API_NAMES = frozenset(
    {
        "DataSpec", "ModelSpec", "QcSpec", "RunSpec", "SampleSpec", "SamplerSpec",
        "Run", "Outputs", "spec", "fit", "load_run", "validate", "stage_dataframe",
        "load_spec", "FAMILY_REGISTRY",
    }
)
