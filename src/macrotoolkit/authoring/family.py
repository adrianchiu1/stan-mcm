"""The ``authored`` family's registry capabilities (S7 M2/M3): generic
functions over the compiled model of the spec they receive -- the
"FamilyEntry constructed at load" of plans/S7-plan.md conflict item 3 is
the cached :class:`CompiledModel` these resolve for each spec.

- :func:`build_stan_data` ``(df, spec)`` and :func:`build_render_context`
  ``(spec)`` -- the two builders every family provides (``run.py``
  dispatches through the registry; ``build_stan_data`` receives the spec
  because the data layout depends on the model definition).
- :func:`mirror_points` -- the fit-time mirror declaration: prior draws
  as Stan inits (every parameter of the rendered program, SV shocks
  non-centered) plus the Python KF mirror at the same point through the
  SAME composition the template stamps.
- :func:`prior_scalar_sds`, :func:`headline_series` -- the sweep
  capabilities.
"""
from __future__ import annotations

import numpy as np

from macrotoolkit.authoring.compile import CompiledModel, compiled_for_spec


def build_stan_data(df, spec=None) -> dict:
    if spec is None:
        raise ValueError("authored.build_stan_data needs the spec (the data layout is the model definition's).")
    return compiled_for_spec(spec).stan_data(df)


def resolved_priors(spec) -> dict[str, dict]:
    return compiled_for_spec(spec).resolve_priors(spec.priors)


def build_render_context(spec) -> dict:
    from macrotoolkit.authoring.stan import render_context

    compiled = compiled_for_spec(spec)
    return render_context(compiled, compiled.resolve_priors(spec.priors))


# ---------------------------------------------------------------------------
# Prior points -> Stan inits + the Python mirror
# ---------------------------------------------------------------------------


def stan_inits_and_h(compiled: CompiledModel, params: dict, priors: dict, anchors: dict, rng: np.random.Generator, T: int):
    """Turn one prior draw into the rendered program's inits (declared
    parameters, ``sigma_h_<s>``, non-centered ``h0_<s>_raw`` = (h0 -
    mu_h0)/sd, fresh ``nu_<s>``) and the log-variance paths the Python
    side rebuilds EXACTLY as the template does (``mu_h0 + sd * raw`` then
    ``sv_rw_noncentered``)."""
    from macrotoolkit.smoother import sv_rw_noncentered

    inits = {name: float(params[name]) for name in compiled.param_names}
    h: dict[str, np.ndarray] = {}
    for s in compiled.sv_shocks:
        sd = float(priors[f"mu_h0_{s}"]["sd"])
        mu = float(anchors[f"mu_h0_{s}"])
        raw = (float(params[f"h0_{s}"]) - mu) / sd
        nu = rng.standard_normal(T)
        inits[f"h0_{s}_raw"] = raw
        inits[f"nu_{s}"] = nu
        h[s] = sv_rw_noncentered(mu + sd * raw, float(params[f"sigma_h_{s}"]), nu)
    return inits, h


def python_kf_loglik(compiled: CompiledModel, params: dict, stan_data: dict, h: dict[str, np.ndarray] | None) -> float:
    from macrotoolkit.smoother import kalman_loglik

    T = int(stan_data["T"])
    x = stan_data["x"] if "x" in stan_data else np.zeros((T, 0))
    n = compiled.meta.n_state
    xi00 = stan_data["xi00"] if "xi00" in stan_data else np.zeros(n)
    P00 = stan_data["P00"] if "P00" in stan_data else np.zeros((n, n))
    F, Q, A, Z, R = compiled.build_matrices(params, h=h or None, T=T)
    return kalman_loglik(stan_data["yobs"], x, F, Q, A, Z, R, xi00, P00)


def mirror_points(spec, stan_data: dict, n: int, rng: np.random.Generator):
    from macrotoolkit.qc import MirrorPoint

    compiled = compiled_for_spec(spec)
    priors = compiled.resolve_priors(spec.priors)
    T = int(stan_data["T"])
    anchors = {k: v for k, v in stan_data.items() if k.startswith("mu_h0_")}
    points = []
    for _ in range(n):
        params = compiled.sample_prior_params(priors, rng, anchors)
        inits, h = stan_inits_and_h(compiled, params, priors, anchors, rng, T)
        points.append(MirrorPoint(inits=inits, loglik_python=python_kf_loglik(compiled, params, stan_data, h)))
    return points


def _mirror_decl():
    from macrotoolkit.qc import MirrorDecl

    return MirrorDecl(draw_points=mirror_points)


MIRROR = _mirror_decl()


# ---------------------------------------------------------------------------
# Sweep capabilities
# ---------------------------------------------------------------------------


def prior_scalar_sds(spec, df, n_draws: int = 10_000, seed: int = 20260902) -> dict[str, float]:
    """Monte-Carlo prior sds of the scalar parameters under the spec's
    resolved priors (the sweep report's contraction readout)."""
    compiled = compiled_for_spec(spec)
    priors = compiled.resolve_priors(spec.priors)
    anchors = compiled.anchors(compiled.series_arrays(df))
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(n_draws):
        p = compiled.sample_prior_params(priors, rng, anchors)
        for k, v in p.items():
            draws.setdefault(k, []).append(v)
    return {k: float(np.std(np.asarray(v))) for k, v in draws.items()}


def headline_series(run_dir, thin: int = 5, seed: int | None = None) -> dict[str, tuple]:
    """Posterior-median path of every state's head slot (the sweep
    report's cross-run overlay)."""
    import dataclasses

    from macrotoolkit.authoring.results import compute_state_draws, load_authored_run
    from specs.schema.authored import ThinSpec

    run = load_authored_run(run_dir)
    if thin > 1:
        run = dataclasses.replace(
            run,
            spec=run.spec.model_copy(update={"outputs": run.spec.outputs.model_copy(update={"smoother_draws": ThinSpec(thin=thin)})}),
        )
    sd = compute_state_draws(run, seed=seed)
    return {name: (sd.dates, np.median(sd.states[name], axis=0)) for name in sd.states}
