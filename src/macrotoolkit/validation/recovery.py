"""Generic PARAMETER-RECOVERY gate arithmetic (ENGINEERING.md ladder rung
2; the G2 shape lifted out of tests/test_g2_parameter_recovery.py in S6
WP3 so any family instantiates it from a design): N simulated datasets at
prior-drawn truths, one production-render fit each, pooled 90%-CI
coverage in an accept band, a per-parameter coverage floor, and a
"no systematic bias" t-test on named boundary-prone scales.

A :class:`RecoveryDesign` supplies the family-authored pieces (the SBC
engine needs the same two: a prior sampler and a structural simulator)
plus the accept/reject constants; :func:`run_recovery` returns the
per-dataset coverage table and :func:`evaluate_recovery` applies the rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from macrotoolkit.validation.sbc import RankedParam, _pooled_draws, render_design_model


@dataclass(frozen=True)
class RecoveryDesign:
    """One family's parameter-recovery gate design.

    - ``family`` / ``model_options`` / ``prior_overrides``: the production
      render (through the ordinary spec + override path).
    - ``draw_truth(rng)``: a parameter point from the family's prior (the
      family may apply a documented stationarity filter here -- G1/G2
      machinery, NOT SBC's exact-prior rule).
    - ``simulate(truth, rng)``: a dataset from the structural equations
      at that truth; ``stan_data(dataset)``: its Stan data block.
    - ``params``: the quantities to score -- posterior variable names, or
      ``(label, extractor)`` pairs for transformed quantities (the SBC
      engine's convention); ``bias_params``: the labels whose
      posterior-median bias is t-tested.
    - constants: ``n_datasets``, sampler settings, ``coverage_low/high``
      (pooled 90%-CI coverage band), ``per_param_floor``, ``bias_t_limit``.
    """

    name: str
    family: str
    model_options: Mapping[str, Any]
    prior_overrides: Mapping[str, Any]
    draw_truth: Callable[[np.random.Generator], dict]
    simulate: Callable[[dict, np.random.Generator], Any]
    stan_data: Callable[[Any], dict]
    params: Sequence[RankedParam]
    bias_params: Sequence[str]
    n_datasets: int
    seed_base: int
    chains: int = 2
    iter_warmup: int = 750
    iter_sampling: int = 750
    adapt_delta: float = 0.95
    max_treedepth: int = 12
    coverage_low: float = 0.80
    coverage_high: float = 0.97
    per_param_floor: float = 0.6
    bias_t_limit: float = 3.0
    expected_priors: Mapping[str, Any] | None = None

    @property
    def param_labels(self) -> tuple[str, ...]:
        return tuple(p if isinstance(p, str) else p[0] for p in self.params)


@dataclass
class RecoveryResult:
    inside: np.ndarray  # (n_datasets, n_params) bool
    median_error: dict[str, list[float]] = field(default_factory=dict)
    divergences: list[int] = field(default_factory=list)

    @property
    def pooled_coverage(self) -> float:
        return float(self.inside.mean())


def run_recovery(design: RecoveryDesign, *, model=None, n_datasets: int | None = None, progress=None) -> RecoveryResult:
    """Fit every simulated dataset (dataset i at ``default_rng(seed_base +
    i)`` for truth and simulation, sampler seed ``seed_base + i``) and
    score 90%-CI coverage per parameter plus posterior-median errors for
    the bias parameters. ``n_datasets`` overrides the design's count for
    smoke runs only."""
    n = design.n_datasets if n_datasets is None else n_datasets
    if model is None:
        model = render_design_model(_as_sbc_like(design))
    labels = design.param_labels
    inside = np.zeros((n, len(labels)), dtype=bool)
    errs: dict[str, list[float]] = {p: [] for p in design.bias_params}
    divs: list[int] = []
    for i in range(n):
        rng = np.random.default_rng(design.seed_base + i)
        truth = design.draw_truth(rng)
        dataset = design.simulate(truth, rng)
        fit = model.sample(
            data=design.stan_data(dataset),
            chains=design.chains,
            parallel_chains=design.chains,
            iter_warmup=design.iter_warmup,
            iter_sampling=design.iter_sampling,
            adapt_delta=design.adapt_delta,
            max_treedepth=design.max_treedepth,
            seed=design.seed_base + i,
            show_progress=False,
        )
        for j, (name, param) in enumerate(zip(labels, design.params)):
            draws = _pooled_draws(fit, param)
            lo, hi = np.quantile(draws, [0.05, 0.95])
            inside[i, j] = lo <= truth[name] <= hi
            if name in errs:
                errs[name].append(float(np.median(draws)) - truth[name])
        divs.append(int(np.sum(fit.method_variables()["divergent__"])))
        if progress is not None:
            progress(i + 1, n)
    return RecoveryResult(inside=inside, median_error=errs, divergences=divs)


def evaluate_recovery(design: RecoveryDesign, result: RecoveryResult) -> tuple[str, list[str], dict]:
    """Apply the accept/reject rule: PASS when the pooled coverage is in
    the band, every parameter clears the floor, and no bias t-stat
    exceeds the limit; FAIL otherwise. Returns ``(verdict, reasons,
    metrics)``."""
    reasons: list[str] = []
    pooled = result.pooled_coverage
    per_param = {name: float(result.inside[:, j].mean()) for j, name in enumerate(design.param_labels)}
    metrics: dict[str, Any] = {"pooled_coverage": pooled, "per_param_coverage": per_param, "divergences_total": int(sum(result.divergences))}
    verdict = "PASS"
    if not (design.coverage_low <= pooled <= design.coverage_high):
        verdict = "FAIL"
        reasons.append(f"pooled 90%-CI coverage {pooled:.3f} outside [{design.coverage_low}, {design.coverage_high}]")
    for name, cov in per_param.items():
        if cov < design.per_param_floor:
            verdict = "FAIL"
            reasons.append(f"coverage for {name} is {cov:.2f} < floor {design.per_param_floor}")
    tstats: dict[str, float] = {}
    for name, errs in result.median_error.items():
        arr = np.asarray(errs, dtype=np.float64)
        se = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        t = float(arr.mean() / se) if se > 0 else 0.0
        tstats[name] = t
        if abs(t) >= design.bias_t_limit:
            verdict = "FAIL"
            reasons.append(f"systematic bias in {name}: mean median error {arr.mean():+.5f} (t={t:+.2f}) >= {design.bias_t_limit}-sigma")
    metrics["bias_t"] = tstats
    if not reasons:
        reasons.append(
            f"pooled coverage {pooled:.3f} in band; per-parameter min "
            f"{min(per_param.values()):.2f}; bias |t| max {max(abs(v) for v in tstats.values()) if tstats else 0.0:.2f}"
        )
    return verdict, reasons, metrics


def _as_sbc_like(design: RecoveryDesign):
    """The render step is shared with the SBC engine (same production
    spec + override path + expected-prior assertion)."""
    from macrotoolkit.validation.sbc import SbcDesign

    return SbcDesign(
        name=design.name, family=design.family, model_options=design.model_options,
        prior_overrides=design.prior_overrides, draw_prior=design.draw_truth, simulate=design.simulate,
        stan_data=design.stan_data, ranked_params=tuple(design.params), n_replications=design.n_datasets,
        rank_draws=99, rank_bins=10, seed_base=design.seed_base, expected_priors=design.expected_priors,
    )


def write_coverage_csv(path: Path, design: RecoveryDesign, result: RecoveryResult) -> None:
    import csv

    with Path(path).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", *design.param_labels, "divergences"])
        for i in range(result.inside.shape[0]):
            w.writerow([i, *[int(v) for v in result.inside[i]], result.divergences[i]])
