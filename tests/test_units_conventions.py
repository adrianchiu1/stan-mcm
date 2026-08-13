"""Numeric-pinning unit tests for the LW-SV numerical conventions
(lw-sv-spec.md §1.1, "Data and units conventions").

**Status: live, worked-example tests, not placeholders.** S2's real
conversion code (`macrotoolkit.smoother`'s state-transition mirror, any SV
log-variance-to-sd helper, and the data-prep step deriving inflation from a
price level) does not exist yet -- this pass is explicitly scoped to *not*
create it (see plans/S2-plan.md). So instead of `pytest.mark.skip` stubs,
this module defines small, local reference implementations of each
convention directly here (deliberately *not* under `src/macrotoolkit/` or
`specs/` -- that would be S2 implementation, out of scope for this pass) and
pins them down with concrete numeric assertions. This is worth having now
because the conventions themselves -- the arithmetic, not any particular
code path -- are exactly what's easy to get subtly wrong (a `g` vs `g/4`
mixup, an `exp(h)` vs `exp(h/2)` factor-of-two bug, a `100*` vs `400*`
scaling slip) and cheap to pin permanently.

**When S2 lands real code:** extend or rewire each test below to import and
call the real function (`macrotoolkit.smoother`'s transition step, its SV
sd helper, `macrotoolkit.data`'s inflation transform) in place of, or in
addition to, the local reference implementation, so the same worked numbers
end up validating production code instead of just this file's own arithmetic.

The three conventions pinned here:

1. **`g` is annualized; the state transition uses `g/4`.** Per lw-sv-spec.md
   §1.3: `y*_t = y*_{t-1} + g_{t-1}/4 + eps_y*,t`. `g` itself is reported/
   used unscaled everywhere else (priors, outputs); only the quarterly
   transition step divides by 4.
2. **`h` is a log-variance; the corresponding standard deviation is
   `exp(h/2)`.** Per lw-sv-spec.md §1.5: shock sd at time t is
   `exp(h_s,t / 2)`, explicitly *not* `exp(h_s,t)` (which would be the
   variance, not the sd) or `h_s,t` used directly.
3. **Inflation is `400 * dlog(P)`.** Per lw-sv-spec.md §1.1: annualized q/q
   inflation from a price level `P` is `400 * (log(P_t) - log(P_{t-1}))` --
   the `400` scales a quarterly log difference (`100*` for percent, `4x` for
   annualization) to an annualized percent.
"""
from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# Convention 1: g annualized, transition uses g/4
# ---------------------------------------------------------------------------


def _quarterly_increment(g_annual: float) -> float:
    """Reference implementation of the arithmetic S2's
    `y*_t = y*_{t-1} + g_{t-1}/4 + eps_y*,t` transition (spec §1.3) must
    apply to `g`: divide the annualized rate by 4 to get the quarterly
    increment."""
    return g_annual / 4.0


def test_g_is_annualized_transition_uses_g_over_4() -> None:
    assert _quarterly_increment(12.0) == pytest.approx(3.0)
    assert _quarterly_increment(2.8) == pytest.approx(0.7)

    # Negative check: g used unscaled (the classic omitted-/4 bug) would
    # give a very different, wrong number for a plausible annualized g.
    g_annual = 2.8
    wrong_unscaled = g_annual
    assert _quarterly_increment(g_annual) != pytest.approx(wrong_unscaled)


# ---------------------------------------------------------------------------
# Convention 2: h is log-variance; sd = exp(h/2)
# ---------------------------------------------------------------------------


def _sd_from_log_variance(h: float) -> float:
    """Reference implementation of spec §1.5's convention: `h` is a
    log-variance, so the corresponding standard deviation is `exp(h/2)`
    (equivalently, `exp(h)` is the variance)."""
    return math.exp(h / 2.0)


def test_h_is_log_variance_sd_is_exp_h_over_2() -> None:
    assert _sd_from_log_variance(0.0) == pytest.approx(1.0)

    h = 2.0 * math.log(2.0)  # exp(h/2) = exp(ln 2) = 2
    assert _sd_from_log_variance(h) == pytest.approx(2.0)

    h2 = 2.0 * math.log(0.5)  # exp(h/2) = exp(ln 0.5) = 0.5
    assert _sd_from_log_variance(h2) == pytest.approx(0.5)

    # Explicit negative check: the classic factor-of-2 bug spec §1.5 warns
    # about is using exp(h) directly as the sd (that's actually the
    # variance). Assert the buggy formula gives a *different*, wrong answer
    # for the same h -- this is exactly the bug class this test exists to
    # catch once it's rewired against real S2 code.
    buggy_sd = math.exp(h)  # wrong: this is exp(h), not exp(h/2)
    correct_sd = _sd_from_log_variance(h)
    assert buggy_sd == pytest.approx(4.0)
    assert correct_sd == pytest.approx(2.0)
    assert buggy_sd != pytest.approx(correct_sd)


# ---------------------------------------------------------------------------
# Convention 3: inflation = 400 * dlog(P)
# ---------------------------------------------------------------------------


def _annualized_inflation(p_t: float, p_prev: float) -> float:
    """Reference implementation of spec §1.1's convention: annualized q/q
    inflation from a price level `P` is `400 * (log(P_t) - log(P_{t-1}))`."""
    return 400.0 * math.log(p_t / p_prev)


def test_inflation_is_400_times_dlog_price_level() -> None:
    # A small, realistic q/q move: price index 100 -> 100.5 (a 0.5% q/q
    # rise) annualizes to approximately 1.995%.
    assert _annualized_inflation(100.5, 100.0) == pytest.approx(1.995, abs=1e-3)

    # An exact clean-number case: choose p_t/p_prev = exp(1/400) so the
    # annualized result is exactly 1.0.
    p_prev = 100.0
    p_t = p_prev * math.exp(1.0 / 400.0)
    assert _annualized_inflation(p_t, p_prev) == pytest.approx(1.0)

    # Negative checks: the two documented wrong scalings (spec-adjacent
    # bugs this convention guards against) must NOT match the correct
    # convention for the same clean-number case.
    quarterly_not_annualized = 100.0 * math.log(p_t / p_prev)  # 100*dlog(P), missing the 4x
    annualized_not_percent = 4.0 * math.log(p_t / p_prev)  # 4*dlog(P), missing the 100x
    assert quarterly_not_annualized != pytest.approx(1.0)
    assert annualized_not_percent != pytest.approx(1.0)
