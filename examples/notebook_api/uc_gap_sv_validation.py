"""The PRE-REGISTERED SBC design for the authored ``uc_gap_sv`` model
(``examples/notebook_api/authored_uc_gap.ipynb``; S8 UC-gap SBC
execution; DECISIONS.md 2026-09-04 "UC-gap SBC PRE-REGISTERED"). This is
per-model work, not framework work (S7's `plans/S7-plan.md`: the authored
system exposes `macrotoolkit.authoring.validation.sbc_design` as a
one-call constructor, NOT registered by default -- registering and
running a design for a specific authored model is S8+ work). Nothing here
is a new gate mechanism: this module builds ONE design for uc_gap_sv by
calling the generic constructor with the model's own equations, reusing
the UCSV/G4 fixed-anchor precedent for the two data-derived pieces the
production spec has and an SBC simulator cannot.

The model definition below is copied VERBATIM from
``authored_uc_gap.ipynb`` cell 3 -- "reuse that exact definition" (the
task brief) -- and must never drift from the notebook's.

SBC exactness for uc_gap_sv (the two model-specific pieces the generic
constructor needs; ``macrotoolkit.authoring.validation.simulate_dataset``
supplies the rest -- the compiled model's own equations through the
engine's ``simulate_forward``, exactly the generative model the rendered
program's KF likelihood defines):

1. **Prior sampler** = the compiled model's own
   ``sample_prior_params`` on the PRODUCTION priors -- no override (the
   UCSV precedent: this model's stationarity is not a prior-sampler
   concern here either, matching every other family's SBC design; the
   AR(2) gap coefficients a1/a2 keep their declared N(1.2, 0.3) /
   N(-0.4, 0.3) priors, same as production) -- at FIXED anchors in place
   of the two data-derived pieces of the production spec:

   - **ystar's initial mean.** Production uses
     ``au.init(au.first_obs("y"), 2.0)``: xi00[ystar] = the data's first
     observation. An SBC simulator draws the initial state before any
     data exists, so both the simulator and the fit use
     ``YSTAR_ANCHOR = 900.0`` -- literally ``Y_ANCHOR`` from
     ``macrotoolkit/families/lw_sv_validation.py`` (G3's fixed-xi00
     pattern), reused unchanged because "y" here maps to the EXACT SAME
     "lgdp100" series as lw_sv's own y observable (same file, same
     column, same units) -- not a new choice, the established one.
     g's and gap's xi00 means (3.0, 0.0) are already fixed by the
     model's own ``initial_state`` declaration, so only ystar needs an
     override; g's, gap's and gap[-1]'s marginal sd (1.0, 2.0, 2.0) are
     read straight off the model's own declaration, unchanged.
   - **eta_gap's mu_h0 anchor.** Production uses
     ``au.log_var_diff("y", 0.5)``: mu_h0 = ln(0.5 * Var(Delta y)) over
     the estimation sample. Fixed instead at
     ``MU_H0_ETA_GAP_ANCHOR = 2*ln(0.5)`` -- the UCSV/G4 fixed-mu_h0
     formula (module docstring of ``ucsv_validation.py``: "the
     log-variance of a plausible half-unit shock"), reused verbatim
     rather than invented fresh. Evaluated once on the notebook's own
     worked-example data (1960Q1-2019Q4 lgdp100), the production anchor
     is ln(0.5 * Var(Delta y)) = -1.11 -- the same order of magnitude as
     2*ln(0.5) = -1.386, confirming plausibility without the design
     being data-derived.

2. **Structural simulator**: ``simulate_dataset`` needs no exogenous
   paths (``compiled.meta.exog_names == ()`` for this model -- no
   exogenous series in the definition).

Execution rule (G4's/UCSV's): a <=3-replication smoke may run first
solely to measure per-replication wall cost; its reps are the design's
own first reps. The design does not shrink silently.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from macrotoolkit import api as mtk
from macrotoolkit import authoring as au
from macrotoolkit.authoring.compile import compiled_for_spec
from macrotoolkit.authoring.validation import sbc_design

# --- PRE-REGISTERED SBC DESIGN CONSTANTS (recorded in DECISIONS.md -------
# --- 2026-09-04 BEFORE the run; never adjusted afterward to pass) --------

SBC_N_REPLICATIONS = 100
SBC_SIM_T = 100
SBC_RANK_DRAWS = 99
SBC_RANK_BINS = 10
SBC_SEED_BASE = 20260904  # fresh: distinct from UCSV (20260920/21), G4 (20260910), G2 (20260831)
SBC_CHAINS = 2
SBC_ITER_WARMUP = 750
SBC_ITER_SAMPLING = 750
SBC_ADAPT_DELTA = 0.95
SBC_MAX_TREEDEPTH = 12
SBC_DIVERGENT_TOTAL_LIMIT = 150  # 0.1% of 100 x 1500 post-warmup draws
SBC_CHI2_P_FLOOR = 0.001

#: Fixed anchors (module docstring points 1a/1b).
YSTAR_ANCHOR = 900.0                          # == lw_sv_validation.Y_ANCHOR (same "y" = lgdp100 series)
MU_H0_ETA_GAP_ANCHOR = 2.0 * float(np.log(0.5))  # == the UCSV/G4 fixed-mu_h0 formula


def uc_gap_sv_model() -> au.Model:
    """The exact ``uc_gap_sv`` definition from
    ``authored_uc_gap.ipynb`` cell 3 -- copied verbatim, never re-derived."""
    return au.Model(
        "uc_gap_sv",
        observables=["y", "pi"],
        measurement=[
            "y = ystar + gap + e_y",
            "pi = b_pi*pi[-1] + (1 - b_pi)*mean(pi[-2], pi[-3], pi[-4]) + b_y*gap[-1] + e_pi",
        ],
        transition=[
            "ystar = ystar[-1] + 0.25*g + eta_ystar",   # g annualized: quarterly increment g/4
            "g = g[-1] + eta_g",
            "gap = a1*gap[-1] + a2*gap[-2] + eta_gap",
        ],
        parameters={
            "a1": au.normal(1.2, 0.3),
            "a2": au.normal(-0.4, 0.3),
            "b_pi": au.beta(8.0, 2.0),
            "b_y": au.normal(0.15, 0.1, lower=0.0),
            "sigma_ystar": au.half_normal(0.3),
            "sigma_g": au.half_normal(0.03),        # pile-up control, as in lw_sv (documented identification work)
            "sigma_y": au.half_normal(0.2),         # measurement noise on output
            "sigma_pi": au.half_normal(1.0),
        },
        shocks={
            "eta_ystar": au.shock("sigma_ystar"),
            "eta_g": au.shock("sigma_g"),
            "eta_gap": au.sv(sigma_h=0.2, h0_sd=1.0, mu_h0=au.log_var_diff("y", 0.5)),   # the demand shock, with SV
            "e_y": au.shock("sigma_y"),
            "e_pi": au.shock("sigma_pi"),
        },
        initial_state={
            "ystar": au.init(au.first_obs("y"), 2.0),
            "g": au.init(3.0, 1.0),
            "gap": au.init(0.0, 2.0),
        },
    )


def uc_gap_sv_spec():
    """A ``RunSpec`` for the model, production priors (no override) --
    ``data`` is a placeholder never loaded (the design supplies simulated
    data and fixed anchors directly)."""
    return mtk.spec(
        "authored",
        options=uc_gap_sv_model(),
        data={"file": "unused.csv", "date_column": "date", "mapping": {"y": "y", "pi": "pi"}},
    )


def _fixed_xi00_p00(compiled, mean_overrides):
    """``(xi00, P00)`` at the model's own declared marginal sd's, with
    ``mean_overrides`` substituted for any state whose declared mean is
    data-derived (``FirstObs``) -- the rest read straight off the
    model's own ``initial_state`` declaration (module docstring point 1)."""
    n = compiled.meta.n_state
    xi00 = np.zeros(n)
    p00 = np.zeros((n, n))
    for i, (name, _) in enumerate(compiled.meta.state_labels):
        init = compiled.options.initial_state[name]
        mean = mean_overrides[name] if name in mean_overrides else float(init.mean)
        xi00[i] = mean
        p00[i, i] = float(init.sd) ** 2
    return xi00, p00


def build_uc_gap_sv_sbc_design():
    """The pre-registered :class:`SbcDesign` -- an instantiation of
    :func:`macrotoolkit.authoring.validation.sbc_design`, nothing new."""
    spec = uc_gap_sv_spec()
    compiled = compiled_for_spec(spec)
    xi00, p00 = _fixed_xi00_p00(compiled, {"ystar": YSTAR_ANCHOR})
    return sbc_design(
        spec,
        name="uc_gap_sv SBC",
        T=SBC_SIM_T,
        xi00=xi00,
        P00=p00,
        n_replications=SBC_N_REPLICATIONS,
        rank_draws=SBC_RANK_DRAWS,
        rank_bins=SBC_RANK_BINS,
        seed_base=SBC_SEED_BASE,
        anchors={"mu_h0_eta_gap": MU_H0_ETA_GAP_ANCHOR},
        chains=SBC_CHAINS,
        iter_warmup=SBC_ITER_WARMUP,
        iter_sampling=SBC_ITER_SAMPLING,
        adapt_delta=SBC_ADAPT_DELTA,
        max_treedepth=SBC_MAX_TREEDEPTH,
    )


UC_GAP_SV_SBC_DESIGN = build_uc_gap_sv_sbc_design()

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "tests" / "artifacts" / "uc_gap_sbc"


def render_uc_gap_sv_model():
    """Render + compile the PRODUCTION ``authored.stan.j2`` template for
    uc_gap_sv, through the exact same path a user's spec takes
    (``authoring.family.build_render_context`` -> ``render_stan_source``
    -> ``compile_model``) -- NOT
    ``macrotoolkit.validation.sbc.render_design_model`` (the generic SBC
    engine's compile step), which asserts ``context["priors"] ==
    expected_priors``: a hand-family render context has a top-level
    ``priors`` dict, but the authored family's (``authoring/stan.py::
    render_context``) has none -- priors are stamped straight into each
    parameter's declaration/prior statement instead, so that assertion
    raises ``KeyError`` for every authored design, not just this one, and
    is therefore a framework gap outside this task's scope
    (``src/macrotoolkit/validation/sbc.py`` is under ``src/macrotoolkit/``
    -- not touched here). The SAME exactness invariant the generic engine
    checks is asserted below, directly against the compiled model's own
    resolved priors, before ``run_sbc`` ever gets an already-compiled
    model (bypassing ``render_design_model`` via its ``model=`` kwarg)."""
    from macrotoolkit.authoring.family import build_render_context
    from macrotoolkit.render import compile_model, render_stan_source

    spec = uc_gap_sv_spec()
    compiled = compiled_for_spec(spec)
    resolved = compiled.resolve_priors(spec.priors)
    assert resolved == UC_GAP_SV_SBC_DESIGN.expected_priors, (
        "uc_gap_sv SBC: the resolved production prior differs from the design's "
        "expected prior -- SBC exactness is exactly this equality, so the gate must not run."
    )
    context = build_render_context(spec)
    source = render_stan_source("authored.stan.j2", context)
    model, _ = compile_model(source)
    return model
