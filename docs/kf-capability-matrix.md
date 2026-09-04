# KF/smoother capability matrix

The item-5 deliverable of the S5 docs pass (`plans/S5-decisions.md`):
what the shared Kalman machinery (`stan/functions/kalman_loglik_tv.stan` +
its ~1e-12 Python mirror `macrotoolkit/smoother.py`, gates G1/G5a) can and
cannot do today, scoped against **UCSV as family #2** and the further
roadmap families (DFM, VAR-SV). Doctrine (S5-decisions item 5): **nothing
here is built speculatively** — a capability is added when the family that
needs it arrives, behind the same G1-mirror gate.

UCSV = Stock–Watson unobserved-components trend inflation with SV on both
the trend and transitory shocks: `pi_t = tau_t + eps_t`,
`tau_t = tau_{t-1} + eta_t`, `eps_t ~ N(0, exp(h_eps,t))`,
`eta_t ~ N(0, exp(h_eta,t))` — univariate, complete data, no exogenous
regressors, both SV flags exercised.

| Capability | Status today | UCSV needs it? | First family that does | Notes |
|---|---|---|---|---|
| Time-varying measurement covariance `R_t` | **Built + validated** (S3 generalization; G1 at ~5.5e-12 on constant, time-varying, and production SV composition paths) | **Yes** (transitory-shock SV) | lw_sv (done) | Array-of-matrices interface; constant case delegates via `rep_array`/`_as_R_path`. |
| Time-varying state innovation covariance `Q_t` | **Built + validated (S6)**: `array[] matrix Q` core with three delegating constant overloads, Python mirror `_as_Q_path`; G1 at ≤7.3e-12 over five paths (constant, tv-R, R-SV composition, tv-Q, Q-SV composition); the constant-Q Stan values reproduce the pre-S6 program EXACTLY (fixture pin) | **Yes** (trend-shock SV enters the STATE innovation) | **UCSV** (done) | Done by the `R_t` playbook; the run-hash consequence (every lw_sv spec re-identified once) taken deliberately per the 2026-08-31 precedent — DECISIONS.md 2026-09-04. |
| Explicit informative initial state `(xi00, P00)` | **Built + validated** (spec §2.2: mean/cov passed explicitly; G5a used HLW's exact values) | Yes (a diffuse-ish explicit prior on tau_0, same pattern as y*_0) | lw_sv (done) | |
| Exact diffuse initialization (Koopman exact-diffuse recursions) | **Not built** (explicit large-variance priors stand in) | No — an explicit `tau_0 ~ N(pi_1, big)` prior is the established pattern here | DFM, possibly (large cross-sections make init precision matter more) | Build only with a mirror gate; exact-diffuse changes the first-step algebra, not just inputs. |
| Missing observations | **Not built** (KF assumes complete `yobs`) | No (quarterly headline/core inflation is complete) | **DFM** (mixed frequency = systematically missing rows) | Standard row-selection KF step; touches both Stan and Python mirrors. |
| Time-varying `Z_t` / `A_t` (measurement loadings/exog maps) | **Not built** (constant, template-stamped) | No | DFM (time-varying loadings variant) / TV-Phillips-slope backlog | |
| Rank-deficient / singular `R` | **Not needed in the filter** (R diagonal, positive); simulation side handles rank-deficient covariances via `_psd_sqrt` (eigendecomposition, S4) | No | VAR-SV with identities in the measurement block, if ever | Filter would need Cholesky→eigen fallback or measurement collapsing. |
| Large-n scaling (univariate/sequential filtering, steady-state KF) | **Not built** (dense `n=7` filter is trivial at T≈260) | No (UCSV is n=1: strictly cheaper than lw_sv) | Large DFMs (explicitly near the VISION non-goal boundary at >100 series) | |
| Exogenous regressor block `A'x_t` + endogenous-lag feedback map | **Built, generic** (S4.5 items 1–2: declared metadata + one engine) | Trivially (UCSV has NO x — empty feedback map, the degenerate case the engine already supports) | lw_sv (done) | |
| RTS smoother / DK simulation smoother / generic simulate-IRF-HD engine | **Built + validated** (G5a ~1e-12; G6 1e-6; S4.5 engine) | Yes — all reusable as-is once matrices exist (dimension-agnostic) | lw_sv (done) | UCSV's results module is a thin metadata + reporting-mapping declaration. |

## What adding UCSV actually required (S6 outcome)

1. **`Q_t` generalization** of `kalman_loglik_tv.stan` + the Python mirror
   — done exactly as scoped (G1 over five paths; constant-Q fixture pin
   exact; G5a/G6 unchanged).
2. Family declarations everywhere else — true, with four places the
   generic layer had lw_sv residue that had to be generalized first
   (recorded in DECISIONS.md 2026-09-04 WP2b): the smoother's
   structural-shock recovery, the engine's state noise, the report/
   figure plumbing, and the loader/HD/IRF pieces that only existed inside
   `results_lw.py` (now `results_core.py`). UCSV's own results module is
   a ~330-line declaration; the family's validation ladder is
   instantiated from `macrotoolkit/validation/` (SBC engine, recovery
   arithmetic, mirror/identity gates).
3. No missing-data, diffuse-init, TV-loading, or scaling work was done —
   per the item-5 doctrine those wait for the family that needs them
   (DFM next).
