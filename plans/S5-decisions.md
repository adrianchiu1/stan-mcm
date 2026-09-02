# Pre-S5 decision menu (2026-09-02)

Decisions taken with the user in the 2026-09-02 review session, which
re-examined S1–S4 **from the framework's perspective**: the project's real
product is a general tool for estimating state-space models with NUTS
(VISION.md); HLW/lw_sv is the dry-run test article. Exact HLW replication
is explicitly NOT a goal — its value (validating the KF/smoother core to
~1e-12, G5a) is already banked. Each item below is a recorded user
decision; rationale in one breath each. The S5 session should treat this
file as binding scope, alongside `plans/S4-plan.md`'s style of execution.

## A. Architecture refactors — do these FIRST, as an "S4.5" block, before any S5 feature work

1. **Endogenous lags become a first-class "feedback map."** lw_sv makes
   itself conditionally linear by hiding lagged endogenous observables
   (y, π) in the "exogenous" x matrix — valid for the likelihood, but it
   means (F,Q,A,Z,R) is not the full generative model, which is why
   `results_lw.py` hand-derives the IS/Phillips recursions across ~1,500
   lines (where both S4 bugs lived). Decision: each family declares, as
   data, which x-columns are which lags of which observables; ONE generic
   simulate/IRF/HD engine consumes the declaration. The validated KF core
   and template are NOT touched. (Companion-form "lags in the state" was
   considered and rejected for v1: it would rewrite validated code and
   re-run every gate; VAR/DFM families are fully generative anyway.)
2. **Named state/coefficient metadata replaces slot-peeking.** Families
   export a state-label vector with explicit time offsets (e.g.
   `("g", -1)`) and a named structural-coefficient dict; downstream code
   consumes names, never indices like `-Z[0,1]` or "slot 3 is g lagged."
   Kills the bug class behind both S4 fan-chart bugs. Non-numeric refactor;
   gates stay green.
3. **Run identity splits in two.** Estimation identity = hash of
   (spec minus `outputs:`, data, rendered Stan, toolchain); report/output
   options live in the run dir but OUTSIDE the hash, so reports regenerate
   freely and report-option schema changes stop orphaning MCMC runs. One
   final hash migration, then stability.
4. **Complete the FamilyEntry contract.** Move `build_stan_data` and
   `build_render_context`'s if/elif chains (run.py) plus the results
   module, plot set, and report assembler onto the family registry, so
   VISION's "adding a family = template + schema fragment + results module
   + validation suite" is mechanically true.

## B. Family roadmap

5. **Family #2 is UCSV** (Stock–Watson unobserved-components trend
   inflation with SV): the smallest true generalization step — no new KF
   features required, exercises the deferred SV-on-trend-shocks flag, and
   cheaply stress-tests decisions A1–A4. DFM (missing data, mixed
   frequency, scaling) and VAR-SV (identification post-processors) come
   after. A KF capability matrix (missing obs, time-varying Z/A, exact
   diffuse init, rank-deficient R, large-n scaling) should be written down
   as part of S5's docs pass, but none of those capabilities are built
   speculatively.

## C. Sampler-side machinery

6. **Stationarity: doctrine now, PACF later.** The G3-style prior
   overrides that keep SBC out of the explosive AR region are promoted
   from ad hoc to a documented, reusable "SBC prior config" mechanism
   (used by G4). A stationarity-enforcing partial-autocorrelation
   parameterization for AR blocks goes on the framework backlog as an
   opt-in feature — not built in S5.
7. **Prior-predictive check gets built in S5, generically.** Spec §4
   promises it and it was never implemented; in a framework whose priors
   deliberately do identification work (σ_g/σ_z pile-up control), "what do
   my priors imply about observable paths" is core functionality. Simulate
   from the prior through the same template/matrices; one figure per run
   report.

## D. S5 scope, reshaped

8. **G5b is demoted from hard gate to informational exhibit** (spec
   amendment, user-approved). Produce the like-for-like comparison figure
   (our FILTERED series vs the published one-sided series — HLW publish no
   smoothed estimates at all; their workbook states "All estimates are
   one-sided") plus a documented explanation of every difference and its
   cause (the tight σ_z pile-up prior accounts for ~90bp of r* at 2019).
   No ±50bp pass/fail: a gate you'd pass by tuning priors toward a target
   validates nothing.
9. **The prior sweep is built as a reusable tool** (`mtk sweep`): base
   spec + a grid of prior overrides → batched runs in the run store + one
   comparison report (key posteriors, prior→posterior contraction,
   headline series across the grid). The mandated σ_g/σ_z sweep is its
   first use.
10. **G4 (full-SV SBC) runs as a pre-registered reduced design** sized to
    roughly one day of compute (on the order of ~100 reps, shorter
    synthetic T, fewer post-warmup draws — exact numbers to be fixed and
    recorded in DECISIONS.md BEFORE the run starts, so the design cannot
    quietly shrink to pass). G4 is the highest-value remaining validation:
    the end-to-end calibration proof of the Rao-Blackwellized NUTS pattern.
11. **Validation harnesses go generic during G4.** Build G4 BY
    generalizing G3's SBC engine (family + prior config in, rank
    statistics out); audit the G1/G2 harnesses for family-parameterization
    at the same time, so UCSV's gates are instantiation, not
    reconstruction.

## E. Output-layer fixes (landed on this branch, 2026-09-02 — see DECISIONS.md)

12. IRF figures now STAMP the constant-r convention (r held fixed, so
    eps_g/eps_z shocks have persistent gap/π effects by design).
13. The gap HD's real-rate data contribution is its own "rdata" bar,
    no longer hidden inside the "Initial condition" line. G6 identities
    unchanged.
14. The fan-chart r* series is aligned to the same period indexing as the
    gap/π/y fans (it was one quarter behind its axis label).

## F. Process / ops

15. **ENGINEERING.md** records the framework doctrine (validation ladder,
    numerical conventions, review requirements) — written 2026-09-02.
16. **Thinned run archive**: the next regeneration of the two reference SV
    runs writes a small tracked `runs-archive/` (draws thinned ~×10, a few
    MB, plain git) labeled "development fixtures — regenerate for
    publication numbers", so fresh containers stop paying 40–80 min before
    output-layer work.
17. **Stage-per-PR cadence continues**: the S4 branch merges to main via
    its own PR before S5 begins; this review branch's changes are a second,
    small PR.
