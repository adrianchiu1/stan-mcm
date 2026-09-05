# Chapter 5, example 4 in unobserved-components form: a random-walk level with an SV transitory shock

The companion's first check of the handbook's SV model (run on the S7 branch as `03d419822418`
before the zero-state extension E0 existed): the level is a random walk `tau`, the transitory
shock `e` carries the log-variance random walk. The same SV block and anchors as
`ch5_sv_uk_inflation`; `sigma_eta ~ half_normal(0.5)`; `tau_0 ~ N(infl_1, 5^2)`.
