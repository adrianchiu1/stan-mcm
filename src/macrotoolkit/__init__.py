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
