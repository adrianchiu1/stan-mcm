"""Re-export shim (S6 WP3): the G4 full-SV SBC design moved VERBATIM into
the package (``macrotoolkit.families.lw_sv_validation``) so ``mtk validate
lw_sv --tier sbc`` and the G4 gate run ONE declaration. Every name the G4
gate file and scripts/run_g4.py import is re-exported here unchanged; the
pre-registered constants are the 2026-09-02 ones."""
from macrotoolkit.families.lw_sv_validation import *  # noqa: F401,F403
from macrotoolkit.families.lw_sv_validation import (  # noqa: F401
    ADAPT_DELTA,
    CHAINS,
    CHI2_P_FLOOR,
    CONDITIONING_SEED,
    DIVERGENT_TOTAL_LIMIT,
    G4_DESIGN,
    G4_PRIORS,
    G4_SEED_BASE,
    H0_PRIOR_SD,
    ITER_SAMPLING,
    ITER_WARMUP,
    MAX_TREEDEPTH,
    MU_H0_IS_ANCHOR,
    MU_H0_PC_ANCHOR,
    N_REPLICATIONS,
    P00,
    PARAM_LABELS,
    PI_PRE,
    R_PATH,
    RANK_BINS,
    RANK_DRAWS,
    SIM_T,
    XI00,
    Y_PRE,
    _build_conditioning,
    _h0_extractor,
    _resolved_sv_priors,
    draw_exact_prior_sv,
    simulate_from_state_space_sv,
    stan_data_sv,
)
