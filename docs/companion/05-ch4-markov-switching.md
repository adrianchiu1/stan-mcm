# Chapter 4 — Gibbs sampling for Markov-switching models (gap chapter)

*Handbook Chapter 4: switching regressions, the Hamilton filter, the
backward recursion for the regime path, the Gibbs sampler for a two-regime
regression (`example3.m`), a Markov-switching VAR (`example4.m`), models
with two independent chains (`example5.m`, `example7.m`) and time-varying
transition probabilities (`example6.m`).*

## The models

The handbook's Chapter 4 model is

    Y_t = c_{S_t} + B_{S_t} X_t + v_t,   v_t ~ N(0, σ²_{S_t}),

with `S_t ∈ {1, 2}` a first-order Markov chain with transition matrix `P`.
Its Gibbs sampler (§3) draws the regime path by the Hamilton filter and a
backward recursion (§4–5), the transition probabilities from a Dirichlet
(§6, step 2), and the regime-specific coefficients and variances by the
Chapter 1 blocks on the observations assigned to each regime. The
extensions (§7) apply the same structure to a VAR with dummy-observation
priors, to two independent chains (four regimes), and to transition
probabilities driven by a probit index.

## Why this is outside macrotoolkit's class

The toolkit estimates models that are **linear-Gaussian conditional on a
set of continuous parameters and paths** — the class in which the Kalman
filter computes the marginal likelihood exactly. A Markov-switching model
is linear-Gaussian conditional on the *discrete* regime path `S_1..S_T`,
and the filter that integrates that path out is the Hamilton filter, not
the Kalman filter. The two recursions differ in kind: Hamilton's carries a
probability vector over regimes and mixes conditional densities; the
Kalman filter carries a Gaussian mean and covariance. Combining them for
models that are state-space *and* regime-switching (Kim's approximate
filter) is a further approximation.

`VISION.md` lists regime switching and particle filtering as non-goals for
the toolkit's core, for a reason that is also a design principle: every
capability of the shared filter is validated by the mirror gate and the
ladder above it, and a second filter of a different mathematical kind
would need its own ladder. The companion therefore does not estimate the
Chapter 4 models.

## What would be needed, and what other tools do

- **Marginalising the regimes inside Stan.** The Hamilton filter *is*
  expressible in Stan: the forward algorithm for a hidden Markov model
  gives the log-likelihood with the regime path summed out, and NUTS can
  then sample the continuous parameters (`c`, `B`, `σ²` per regime and
  the transition probabilities, with the usual label-switching care via
  an ordering constraint). The regime path is recovered afterwards by the
  forward–backward smoother. This is a well-trodden Stan pattern for the
  *regression* models of §3 and the MS-VAR of §7; it is not a small
  extension of macrotoolkit because it replaces the likelihood, the
  smoother and the output engine rather than adding a matrix path to
  them. It is an item for a different family with its own validation
  ladder, not for the linear-Gaussian one.
- **Markov-switching state-space models** (the handbook's "further
  reading" direction, Kim & Nelson 1999) need Kim's collapsing
  approximation or a particle filter — both explicitly outside the
  toolkit's doctrine of exact likelihoods with a machine-precision mirror.
- **The models an economist actually reaches for** in this chapter —
  regime-dependent means and variances of inflation, of GDP growth, of a
  policy rule — often have a linear-Gaussian relative that the toolkit
  does estimate: stochastic volatility (Chapter 5 §2) replaces a
  variance switch with a smooth log-variance path; a time-varying
  coefficient (Chapter 5 §3) replaces a coefficient switch with a random
  walk. The companion's UK-inflation examples show both. Whether a
  discrete break or a drift is the right description is an economic
  question; the toolkit's answer to it is one of the two, not a
  mixture.

## Status

Outside the class by design. No planned extension. A Stan HMM family
would be a separate project with its own ladder; if it is wanted, the
handbook's `example3.m` (the two-regime regression on artificial data)
is the natural first oracle.
