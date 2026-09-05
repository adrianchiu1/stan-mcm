"""Post-processors over run outputs (S8 WP3): identification and
conditioning steps that consume a run's structural IRFs and unconditional
forecasts and never touch the sampler -- sign restrictions
(:mod:`.sign_restrictions`) and Waggoner-Zha conditional forecasts
(:mod:`.conditional_forecast`). Both operate on plain arrays (tested on
synthetic draws) with thin adapters from an authored run's ``IRFDraws``.
"""
from __future__ import annotations

from macrotoolkit.postprocess.conditional_forecast import ConditionalForecast, conditional_forecast, conditional_forecast_for_run  # noqa: F401
from macrotoolkit.postprocess.sign_restrictions import (  # noqa: F401
    SignRestrictedIRFs,
    SignRestriction,
    haar_rotation,
    irf_array_from_draws,
    sign_restricted_irfs,
)
