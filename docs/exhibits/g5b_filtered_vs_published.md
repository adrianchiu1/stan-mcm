# G5b informational exhibit: filtered vs published one-sided HLW

Run `a00958509083` (reference SV run) vs the NY Fed real-time workbook's `2019Q2` vintage sheet (one-sided; their header: "All estimates are one-sided" -- HLW publish no smoothed estimates, so the one-sided/FILTERED comparison is the only like-for-like one). **Informational only -- no pass/fail** (S5-decisions item 8: a gate passed by tuning priors toward a target validates nothing).

| Series | mean abs diff | mean abs diff (2000+) | max abs diff | corr | final-period diff |
|---|---|---|---|---|---|
| rstar | 0.65 | 0.96 | 2.16 | 0.941 | +1.44 |
| output_gap | 1.34 | 1.17 | 3.94 | 0.829 | +0.43 |
| g | 0.25 | 0.24 | 2.19 | 0.921 | -0.07 |
| z | 0.48 | 0.80 | 1.60 | 0.800 | +1.51 |

Measured attribution via the r* = g + z identity: the final-period r*
difference of +1.44 decomposes into +1.51 from z and -0.07 from g --
i.e. essentially the entire late-sample r* gap sits in z, exactly where the
deliberate sigma_z pile-up prior acts (cause 1 below); trend growth g is
close to the published series.

Attributed causes of the differences (each deliberate or documented, none a
numerics discrepancy -- the KF/smoother core matches HLW's own machinery to
~1e-12 at fixed parameters, gate G5a):

1. **sigma_z pile-up prior vs MUE lambda_z** -- the deliberate Half-N(0, 0.08^2)
   identification prior (spec §1.6) holds sigma_z below HLW's MUE-implied value,
   damping |z| and pulling r* toward g. This is the dominant cause of the
   late-sample r* level gap, and it is testable: the mandated `mtk sweep`
   sigma_g_z sweep's `sigma_z_loose` cell (prior sd doubled) moves r* toward the
   published series.
2. **Bayesian posterior median vs frequentist MLE plug-in** -- HLW filter at a
   single L-BFGS point estimate; we integrate over the posterior.
3. **Stochastic volatility** -- our reference run has SV on the IS/PC shocks;
   HLW's model has constant variances (DECISIONS.md 2026-08-31 records the
   induced posterior shifts, e.g. sigma_y* 0.24 vs 0.54).
4. **Data vintage** -- our example CSV vs the vintage underlying the published
   sheet (~0.06pp mean / 0.31pp max on the states for a same-model reproduction,
   G5a fixture work).
5. **Initialization** -- our explicit (xi00, P00) prior anchored at the first
   observation vs HLW's inner-optimization P.00 procedure (G5a used their exact
   values; the production run does not).
