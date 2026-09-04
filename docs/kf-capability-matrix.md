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
| Time-varying state innovation covariance `Q_t` | **Not built** — `Q` constant (S3 deliberately routed lw_sv's SV through `R_t`; the marginalized LW form has SV only on measurement shocks) | **YES — the one real KF extension UCSV requires** (trend-shock SV enters the STATE innovation) | **UCSV** | Same generalization pattern as `R_t` (array-of-`Q_t` core + constant overload); spec §2.2 always named this in the contract. Requires a fresh G1 mirror extension + the no-SV render byte-stability discipline (DECISIONS.md 2026-08-31 precedent for the run-hash consequences). This is also the "deferred SV-on-trend-shocks flag" (spec §0.4). |
| Explicit informative initial state `(xi00, P00)` | **Built + validated** (spec §2.2: mean/cov passed explicitly; G5a used HLW's exact values) | Yes (a diffuse-ish explicit prior on tau_0, same pattern as y*_0) | lw_sv (done) | |
| Exact diffuse initialization (Koopman exact-diffuse recursions) | **Not built** (explicit large-variance priors stand in) | No — an explicit `tau_0 ~ N(pi_1, big)` prior is the established pattern here | DFM, possibly (large cross-sections make init precision matter more) | Build only with a mirror gate; exact-diffuse changes the first-step algebra, not just inputs. |
| Missing observations | **Not built** (KF assumes complete `yobs`) | No (quarterly headline/core inflation is complete) | **DFM** (mixed frequency = systematically missing rows) | Standard row-selection KF step; touches both Stan and Python mirrors. |
| Time-varying `Z_t` / `A_t` (measurement loadings/exog maps) | **Not built** (constant, template-stamped) | No | DFM (time-varying loadings variant) / TV-Phillips-slope backlog | |
| Rank-deficient / singular `R` | **Not needed in the filter** (R diagonal, positive); simulation side handles rank-deficient covariances via `_psd_sqrt` (eigendecomposition, S4) | No | VAR-SV with identities in the measurement block, if ever | Filter would need Cholesky→eigen fallback or measurement collapsing. |
| Large-n scaling (univariate/sequential filtering, steady-state KF) | **Not built** (dense `n=7` filter is trivial at T≈260) | No (UCSV is n=1: strictly cheaper than lw_sv) | Large DFMs (explicitly near the VISION non-goal boundary at >100 series) | |
| Exogenous regressor block `A'x_t` + endogenous-lag feedback map | **Built, generic** (S4.5 items 1–2: declared metadata + one engine) | Trivially (UCSV has NO x — empty feedback map, the degenerate case the engine already supports) | lw_sv (done) | |
| RTS smoother / DK simulation smoother / generic simulate-IRF-HD engine | **Built + validated** (G5a ~1e-12; G6 1e-6; S4.5 engine) | Yes — all reusable as-is once matrices exist (dimension-agnostic) | lw_sv (done) | UCSV's results module is a thin metadata + reporting-mapping declaration. |

## What adding UCSV actually requires (scoping summary)

1. **`Q_t` generalization** of `kalman_loglik_tv.stan` + the Python mirror
   (the one real KF change), gated by a G1 extension over all Q-paths, with
   a regression pin that the constant-Q case reproduces current G1/G5a
   results — the exact playbook of S3's `R_t` generalization.
2. Family declarations only, everywhere else: template + schema fragment +
   `families/ucsv.py` (state metadata `(("tau", 0),)`-style, empty feedback
   map, prior sampler) + registry entry + validation suite instantiated
   from the generic harnesses (SBC via `tests/sbc_harness.py`; G1/G2
   patterns audited for family-parameterization at that point).
3. No missing-data, diffuse-init, TV-loading, or scaling work — per the
   item-5 doctrine those wait for the family that needs them.
