"""Documented placeholder for the LW-SV numerical-conventions unit tests
(lw-sv-spec.md §1.1, "Data and units conventions").

S1's toy model (`local_level`) has none of these quantities -- it is
deliberately outside the LW-SV spec's scope (see plans/S1-plan.md, "Toy
model: local-level"). These conventions are introduced in S2 alongside the
`lw_sv` family's equations (`specs/schema/lw_sv.py`,
`stan/templates/lw_sv.stan.j2`, and whatever Python-side units-conversion
helpers S2 adds, e.g. under `src/macrotoolkit/`).

The three conventions this module pins down (to be asserted against real
S2 code once it exists -- not invented/stubbed now):

1. **`g` is annualized; the state transition uses `g/4`.** `g` (trend
   growth) is expressed as an annualized rate throughout outputs and
   priors, but the quarterly state-transition equation for potential
   output divides it by 4 (`y*_t = y*_{t-1} + g_{t-1}/4 + ...`), per
   lw-sv-spec.md §1.3. A units-conversion test must confirm any Python-side
   mirror of this transition (`macrotoolkit.smoother`, once it exists)
   applies `g/4`, not `g`, in the state equation, while reporting `g`
   itself unscaled.

2. **`h` is a log-variance; the corresponding standard deviation is
   `exp(h/2)`.** The non-centered stochastic-volatility states (`h`, per
   lw-sv-spec.md §1.5) are log-variances, not log-standard-deviations or
   variances directly. A units-conversion test must confirm any code that
   turns `h` into a usable shock scale computes `sd = exp(h / 2)`
   (equivalently `exp(h) = variance`), not `exp(h)` used directly as an sd
   or `h` used directly as a variance.

3. **Inflation is `400 * dlog(P)`.** Per lw-sv-spec.md §1.1, the observed
   inflation series is constructed from the price level `P` as
   `400 * (log(P_t) - log(P_{t-1}))` -- an annualized quarterly log
   difference in percentage points (the `400` scales a quarterly log
   difference to an annualized percent, i.e. `100 * 4`). A units-conversion
   test must confirm the data-prep step that derives inflation from a price
   level uses exactly this transform, not `100 * dlog(P)` (quarterly, not
   annualized) or `4 * dlog(P)` (annualized, not percent) or raw
   differences instead of log differences.

Each convention below is stubbed as a single `pytest.mark.skip`-decorated
test rather than a vacuous `assert True`: the test names the specific S2
function it will call once that function exists, so it fails loudly (via
`ImportError`/`AttributeError`, were the skip removed) rather than passing
by accident if pointed at the wrong thing.
"""
from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason=(
        "S2 scope: no lw_sv state-transition / Python KF mirror exists yet "
        "(src/macrotoolkit/smoother.py is not created until S2, per "
        "plans/S1-plan.md's 'Not created in S1' list). Once it exists, this "
        "test should construct a known g and a known previous state, call "
        "the transition function, and assert the potential-output "
        "increment used g/4 (not g) while any reported/returned g itself "
        "is unscaled."
    )
)
def test_g_is_annualized_transition_uses_g_over_4() -> None:
    raise NotImplementedError


@pytest.mark.skip(
    reason=(
        "S2 scope: no stochastic-volatility state (h) or its consuming "
        "code exists yet -- SV is introduced in S3 per lw-sv-spec.md §1.5 "
        "and plans/S1-plan.md's stage table (S1 has no SV at all). Once "
        "the relevant conversion helper exists, this test should feed it a "
        "known log-variance h and assert the returned standard deviation "
        "equals exp(h / 2), not exp(h) or h itself."
    )
)
def test_h_is_log_variance_sd_is_exp_h_over_2() -> None:
    raise NotImplementedError


@pytest.mark.skip(
    reason=(
        "S2 scope: no LW-SV data-prep step deriving inflation from a price "
        "level exists yet (macrotoolkit.data is generic/family-agnostic in "
        "S1; see its module docstring). Once S2 adds that transform, this "
        "test should feed it a known price-level series and assert the "
        "output equals 400 * (log(P_t) - log(P_{t-1})), not 100*dlog(P), "
        "4*dlog(P), or a non-log difference."
    )
)
def test_inflation_is_400_times_dlog_price_level() -> None:
    raise NotImplementedError
