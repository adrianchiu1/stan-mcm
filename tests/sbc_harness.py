"""Re-export shim (S6 WP3): the generic SBC engine now lives in the
package as ``macrotoolkit.validation.sbc`` so ``mtk validate`` can drive
it; every name is re-exported here unchanged so the G3/G4 gate files and
scripts/run_g4.py keep importing ``sbc_harness``."""
from macrotoolkit.validation.sbc import *  # noqa: F401,F403
from macrotoolkit.validation.sbc import (  # noqa: F401
    RankedParam,
    SbcDesign,
    SbcRunResult,
    _load_ranks_csv,
    _pooled_draws,
    chi2_pvalues,
    plot_rank_histograms,
    rank_statistic,
    render_design_model,
    run_sbc,
    thin_evenly,
    write_ranks_csv,
)
