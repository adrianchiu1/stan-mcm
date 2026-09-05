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
