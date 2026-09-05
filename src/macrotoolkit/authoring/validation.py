"""Auto-instantiated validation for AUTHORED models (S7 M4): the fast
tier (the Stan-vs-Python mirror gate at prior draws of the production
render + the historical-decomposition identity at STATIONARY prior points)
is built for every authored model from its spec by :func:`suite_for`;
the recovery and SBC tiers are exposed as ONE-CALL constructors taking an
authored design (:func:`recovery_design`, :func:`sbc_design`) but are NOT
registered by default -- pre-registering and running an SBC design is
per-model work (S8+), not framework work (plans/S7-plan.md).

The structural simulator every recovery/SBC design needs is generic here:
the compiled model's own equations through the engine's
``simulate_forward`` from ``xi_0 ~ N(xi00, P00)`` with the SV random walks
continued from the drawn ``h_0`` -- exactly the generative model the
rendered program's KF likelihood defines (the SBC exactness requirement).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from macrotoolkit.authoring.compile import CompiledModel, compiled_for_spec
from specs.schema.base import RunSpec, load_spec


def _as_spec(spec_or_path) -> RunSpec:
    if isinstance(spec_or_path, RunSpec):
        return spec_or_path
    return load_spec(str(spec_or_path))


def _df_for(spec: RunSpec, base_dir: Path | None, data=None):
    from macrotoolkit.data import load_data

    if data is not None:
        import tempfile

        from macrotoolkit.api import stage_dataframe

        with tempfile.TemporaryDirectory(prefix="mtk_validate_") as tmp:
            staged = stage_dataframe(data, spec, Path(tmp))
            df, _, _ = load_data(staged, base_dir=Path(tmp))
        return df
    df, _, _ = load_data(spec, base_dir=base_dir if base_dir is not None else Path.cwd())
    return df


def suite_for(spec_or_path, *, base_dir: str | Path | None = None, data=None, n_mirror_points: int = 25):
    """The auto-instantiated :class:`ValidationSuite` for an authored spec
    (a ``RunSpec`` or a spec YAML path; ``data`` may be a DataFrame
    standing in for ``data.file``): the fast tier -- ``mirror`` (G1's shape,
    ``n_mirror_points`` prior draws of the production render) and
    ``hd_identity`` (G6's shape at stationary prior points)."""
    from macrotoolkit.validation.suite import ValidationSuite, hd_identity_gate, mirror_gate

    spec = _as_spec(spec_or_path)
    if spec.model.family != "authored":
        raise ValueError(f"suite_for: model.family is {spec.model.family!r}; for a hand-written family use mtk validate <family>.")
    if base_dir is None and not isinstance(spec_or_path, RunSpec):
        base_dir = Path(spec_or_path).resolve().parent
    base = Path(base_dir).resolve() if base_dir is not None else None
    compiled = compiled_for_spec(spec)

    def spec_builder() -> RunSpec:
        return spec

    def df_builder():
        return _df_for(spec, base, data)

    from macrotoolkit.authoring.results import hd_reconstruction_error

    return ValidationSuite(
        family=f"authored:{compiled.name}",
        gates=(
            mirror_gate("authored", spec_builder, df_builder, n_points=n_mirror_points),
            hd_identity_gate("authored", spec_builder, df_builder, hd_reconstruction_error),
        ),
    )


# ---------------------------------------------------------------------------
# Generic structural simulator + design constructors (not registered)
# ---------------------------------------------------------------------------


def simulate_dataset(compiled: CompiledModel, params: Mapping[str, float], rng: np.random.Generator, T: int, *,
                     xi00: np.ndarray, P00: np.ndarray, exog_paths: Mapping[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """One dataset from the model's own equations via the engine --
    exactly the generative model the rendered program's KF likelihood
    defines: ``xi_0 ~ N(xi00, P00)``, SV shocks continue from
    ``params['h0_<s>']``, the observable feedback registers start at ZERO
    (the ``L`` pre-sample rows of the simulated dataset are zeros, which is
    what :func:`_fixed_stan_data` prepends). Returns ``T`` rows of every
    observable and, for every exogenous series, its FULL path of ``L + T``
    rows (pre-sample rows included) -- supplied by the caller through
    ``exog_paths[e]`` (length ``L + T``, aligned with the dataset rows: lag
    k of estimation step i is row ``L + i - k``)."""
    from macrotoolkit.authoring.results import DrawMatrices, _noise_models, data_path_rules, presample_exog_seeds
    from macrotoolkit.engine import simulate_forward
    from macrotoolkit.smoother import _psd_sqrt

    meta = compiled.meta
    L = compiled.lag_depth
    h0 = {s: float(params[f"h0_{s}"]) for s in compiled.sv_shocks}
    F, Q, A, Z, R = compiled.build_matrices(params, h={s: np.array([h0[s]]) for s in compiled.sv_shocks} or None, T=1)
    extras = {**{k: float(v) for k, v in params.items() if not k.startswith("h0_")}, **{f"h_{s}": np.array([h0[s]]) for s in compiled.sv_shocks}}
    dm = DrawMatrices(F=F, Q=Q if Q.ndim == 2 else Q[0], A=A, Z=Z, R=R if R.ndim == 2 else R[0], extras=extras, M=compiled.build_M(params))
    state_noise, meas_noise = _noise_models(compiled, dm, h_last=h0)
    out_exog: dict[str, np.ndarray] = {}
    for e in meta.exog_names:
        if exog_paths is None or e not in exog_paths:
            raise ValueError(
                f"simulate_dataset: exogenous series {e!r} needs a supplied path exog_paths[{e!r}] of length L + T = {L + T} "
                f"(pre-sample rows included)."
            )
        path = np.asarray(exog_paths[e], dtype=np.float64)
        if path.shape != (L + T,):
            raise ValueError(f"simulate_dataset: exog_paths[{e!r}] must have shape ({L + T},) = (L + T,); got {path.shape}.")
        out_exog[e] = path.copy()
    rules = data_path_rules(compiled, out_exog, T)
    exog_seeds = presample_exog_seeds(compiled, out_exog)
    xi_init = xi00 + _psd_sqrt(P00) @ rng.standard_normal(len(xi00))
    out = simulate_forward(F, dm.Q, A, Z, meta, xi_init, {}, exog_seeds, rules, meas_noise, T, rng, state_noise=state_noise, meas_loading=dm.M)
    return {**{o: out["obs"][:, i].copy() for i, o in enumerate(meta.obs_names)}, **out_exog}


def _fixed_stan_data(compiled: CompiledModel, xi00: np.ndarray, P00: np.ndarray, anchors: Mapping[str, float]):
    """The design's Stan data at FIXED anchors: ``L`` zero pre-sample rows
    for the observables (the simulator's zero-seeded registers), the
    exogenous series' full simulated/supplied paths."""

    def stan_data(dataset: Mapping[str, np.ndarray]) -> dict:
        L = compiled.lag_depth
        series = {o: np.concatenate([np.zeros(L), np.asarray(dataset[o], dtype=np.float64)]) for o in compiled.meta.obs_names}
        for e in compiled.meta.exog_names:
            series[e] = np.asarray(dataset[e], dtype=np.float64)
        yobs, x = compiled.regressors(series)
        data = {"T": int(yobs.shape[0]), "yobs": yobs, "xi00": np.asarray(xi00, dtype=np.float64), "P00": np.asarray(P00, dtype=np.float64)}
        if x.shape[1] > 0:
            data["x"] = x
        data.update({k: float(v) for k, v in anchors.items()})
        return data

    return stan_data


def _ranked(compiled: CompiledModel, anchors: Mapping[str, float], priors: Mapping[str, Mapping[str, Any]]) -> list:
    """The ranked/scored quantities: every scalar parameter plus, per SV
    shock, ``h0_<s>`` recovered from the non-centered raw (the template's
    own line, G4's/UCSV's pattern)."""
    out: list = list(compiled.param_names)
    for s in compiled.sv_shocks:
        mu, sd = float(anchors[f"mu_h0_{s}"]), float(priors[f"mu_h0_{s}"]["sd"])

        def extract(fit, raw=f"h0_{s}_raw", mu=mu, sd=sd):
            return mu + sd * np.asarray(fit.stan_variable(raw))

        out.append((f"h0_{s}", extract))
    return out


def _design_pieces(spec: RunSpec, T: int, xi00, P00, anchors: Mapping[str, float] | None, exog_paths):
    compiled = compiled_for_spec(spec)
    priors = compiled.resolve_priors(spec.priors)
    anchors = dict(anchors or {})
    for s in compiled.sv_shocks:
        anchors.setdefault(f"mu_h0_{s}", 0.0)
    xi00 = np.asarray(xi00, dtype=np.float64)
    P00 = np.asarray(P00, dtype=np.float64)

    def draw_prior(rng: np.random.Generator) -> dict:
        return compiled.sample_prior_params(priors, rng, anchors)

    def simulate(params: dict, rng: np.random.Generator):
        return simulate_dataset(compiled, params, rng, T, xi00=xi00, P00=P00, exog_paths=exog_paths)

    return compiled, priors, anchors, draw_prior, simulate, _fixed_stan_data(compiled, xi00, P00, anchors)


def recovery_design(spec: RunSpec, *, name: str, T: int, xi00, P00, n_datasets: int, seed_base: int,
                    anchors: Mapping[str, float] | None = None, exog_paths=None, bias_params: Sequence[str] | None = None, **sampler):
    """A :class:`RecoveryDesign` (G2's shape) for an authored spec at FIXED
    initial-state / h_0 anchors (the G3/G4/UCSV fixed-anchor pattern: the
    simulator draws before any data exists). Not run here."""
    from macrotoolkit.validation.recovery import RecoveryDesign

    compiled, priors, anchors, draw_prior, simulate, stan_data = _design_pieces(spec, T, xi00, P00, anchors, exog_paths)
    ranked = _ranked(compiled, anchors, priors)
    return RecoveryDesign(
        name=name, family="authored", model_options=spec.model.options, prior_overrides=dict(spec.priors),
        draw_truth=draw_prior, simulate=simulate, stan_data=stan_data, params=ranked,
        bias_params=tuple(bias_params) if bias_params is not None else tuple(p for p in compiled.param_names if compiled.bounds.get(p, (None, None))[0] == 0.0),
        n_datasets=n_datasets, seed_base=seed_base, expected_priors=priors, **sampler,
    )


def sbc_design(spec: RunSpec, *, name: str, T: int, xi00, P00, n_replications: int, rank_draws: int, rank_bins: int, seed_base: int,
               anchors: Mapping[str, float] | None = None, exog_paths=None, **sampler):
    """An :class:`SbcDesign` (G3/G4's shape) for an authored spec at fixed
    anchors. Constructing it is NOT pre-registering it: record the
    constants in DECISIONS.md before any run (ENGINEERING.md rung 3)."""
    from macrotoolkit.validation.sbc import SbcDesign

    compiled, priors, anchors, draw_prior, simulate, stan_data = _design_pieces(spec, T, xi00, P00, anchors, exog_paths)
    return SbcDesign(
        name=name, family="authored", model_options=spec.model.options, prior_overrides=dict(spec.priors),
        draw_prior=draw_prior, simulate=simulate, stan_data=stan_data, ranked_params=_ranked(compiled, anchors, priors),
        n_replications=n_replications, rank_draws=rank_draws, rank_bins=rank_bins, seed_base=seed_base, expected_priors=priors, **sampler,
    )


def recovery_gate_for(design):
    from macrotoolkit.validation.suite import recovery_gate

    return recovery_gate(design)


def sbc_gate_for(design, **kw):
    from macrotoolkit.validation.suite import sbc_gate

    return sbc_gate(design, **kw)
