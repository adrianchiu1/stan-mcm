# Engineering doctrine — macrotoolkit

Framework-level rules distilled from the S1–S4 dry run (the lw_sv vertical
slice). These bind every future model family; they are not LW-specific
lessons. Origin and evidence for each live in DECISIONS.md; scope decisions
for the next stage in `plans/S5-decisions.md`.

## The validation ladder

Every family climbs the same ladder, in order, each rung blocking the next:

1. **Mirror gate.** Any computation that exists in both Stan and Python
   (likelihood, matrix construction) must agree to near machine precision
   (~1e-8 asserted; ~1e-12 observed) on a grid of random parameter points
   before anything downstream is trusted. This is the single most important
   internal check in the toolkit.
2. **Parameter recovery** on simulated data (true values inside the 90% CI
   ~90% of the time; no systematic bias in the weakly identified scales).
3. **Simulation-based calibration** (SBC), with a pre-registered design —
   replication count, synthetic T, draws per rep fixed and recorded BEFORE
   the run, never adjusted afterward to pass.
4. **External oracle replication where one exists** (fixed parameters,
   near machine precision — e.g. G5a vs the HLW code). Discrepancies are
   traced to specific documented causes, never waved through as "close".
5. **Output identities.** Anything claiming to decompose or reconstruct a
   series carries an exact per-period, per-draw reconstruction test
   (G6-style, 1e-6).

## Numerical conventions

- **Units are stated once and enforced by tests** (a
  `test_units_conventions.py`-style file per convention set). For lw_sv:
  g is annualized everywhere; only the potential-output transition divides
  by 4.
- **h is log-VARIANCE**: sd = exp(h/2), never exp(h). Stated once,
  enforced everywhere; the classic factor-of-2 bug class.
- **Kalman numerics doctrine**: Cholesky factorization for every solve and
  log-determinant (no raw inverse or determinant); every propagated
  covariance explicitly re-symmetrized; Python mirror performs the same
  operations in the same order as Stan, so agreement is exact-ish, not
  approximate.
- **Initial states are explicit** (mean/cov passed in), never ad-hoc
  diffuse hacks.
- **Rank-deficient covariances** are simulated via eigendecomposition
  square roots (`_psd_sqrt`), never plain Cholesky.

## Parameterization and priors

- **Non-centered parameterization** for random-walk/scale hierarchies (SV
  blocks) is the default; it kept the σ_h funnel benign at stock
  adapt_delta.
- **Boundary-prone scale parameters** (random-walk innovation sds near
  zero — the pile-up problem) get deliberately informative priors that are
  DOCUMENTED as doing identification work, and every such prior carries a
  mandated sensitivity sweep.
- **Prior overrides are variant-checked**: overriding a prior that is
  inactive for the run's configuration is a hard error, never a silent
  no-op.
- **Stationarity for SBC**: use the documented SBC prior-config mechanism
  (see `plans/S5-decisions.md` item 6); a PACF-based stationary
  parameterization is backlog.

## Code structure

- **The spec is the spine**; templates are stamped, with no runtime
  branching in Stan.
- **Downstream code consumes named metadata, never matrix slots or
  implicit timing conventions** (S5-decisions item 2). Both S4 bugs were
  slot/timing mistakes.
- **Endogenous lags are declared via the family's feedback map**
  (S5-decisions item 1), so simulation/IRF/HD engines stay generic.
- **The run store is immutable and hash-identified**; estimation identity
  excludes report options (S5-decisions item 3).
- **Family capabilities live in the registry** (FamilyEntry), not in
  if/elif chains.

## Process

- **Mandatory numerics review** for any change touching the filter,
  smoother, state-space construction, forecasting, or decompositions —
  a fresh pass per change, even to previously reviewed code. This process
  caught two real pre-commit bugs in S4; treat it as load-bearing.
- **Stochastic code ships a zero-noise seam**: a deterministic path
  through the same code (not a re-derivation) that collapses to a known
  identity, so the combination logic is testable without RNG.
- **Figures state their conventions on the figure** (e.g. the IRF
  constant-r stamp): a reader must never have to infer a modeling
  convention from a surprising shape.
- **Stage-per-PR**: each stage merges to main via its own PR before the
  next begins; run records and decisions are committed with the stage.
