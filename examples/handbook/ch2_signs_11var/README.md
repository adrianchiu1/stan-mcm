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
