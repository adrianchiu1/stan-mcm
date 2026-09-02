"""The FAMILY-PARAMETERIZED SBC engine (S5-decisions item 11): "family +
prior config in, rank statistics out". Built by generalizing G3's engine --
the render/compile step, the replication loop (seeded prior draw ->
simulate -> fit -> rank), incremental artifact persistence, the rank
histograms, and the chi^2 uniformity gate are all family-agnostic here;
what varies per gate is captured in an :class:`SbcDesign`:

- the model configuration (family, options, prior overrides -- applied
  through the PRODUCTION override path, using the family's documented SBC
  prior config where stationarity demands it, S5-decisions item 6);
- the prior sampler and dataset simulator (callables: SBC exactness lives
  in these two being the EXACT generative model the rendered program
  fits, so they are design-authored and design-audited, not guessed
  generically);
- the Stan data builder (dataset -> data dict);
- the ranked parameter list (each entry either a posterior variable name,
  or a (name, extractor) pair for ranked quantities that need a transform
  of posterior draws);
- the replication/thinning/binning/seed constants.

G3's gate (tests/test_g3_sbc.py) runs THROUGH this engine with its exact
historical callables and seeds -- pinned equivalent to the pre-engine
implementation by fast guards -- and G4's full-SV gate
(tests/test_g4_sbc.py) is an instantiation, not a reconstruction. So is
any future family's SBC gate (UCSV: new design, same engine).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from macrotoolkit.render import compile_model, render_stan_source
from macrotoolkit.run import build_render_context
from specs.schema import get_family
from specs.schema.base import RunSpec


def rank_statistic(theta_true: float, draws_thinned: np.ndarray) -> int:
    """SBC rank: the number of thinned posterior draws strictly below the
    prior draw. Uniform on {0, ..., len(draws_thinned)} under calibration."""
    return int(np.sum(draws_thinned < theta_true))


def thin_evenly(draws: np.ndarray, k: int) -> np.ndarray:
    """k evenly spaced draws from the pooled posterior sample (strips the
    bulk of the autocorrelation SBC's uniformity result assumes away)."""
    n = draws.shape[0]
    if n < k:
        raise ValueError(f"Need at least {k} posterior draws, got {n}.")
    idx = np.linspace(0, n - 1, k).round().astype(int)
    return draws[idx]


#: A ranked quantity: either a posterior variable name (ranked directly),
#: or (label, extractor) where extractor(fit) returns the pooled draws --
#: for quantities needing a transform (e.g. G4's h0 = mu_h0 + sd*h0_raw).
RankedParam = str | tuple[str, Callable[[Any], np.ndarray]]


@dataclass(frozen=True)
class SbcDesign:
    """One SBC gate's full pre-registered design (ENGINEERING.md ladder
    rung 3: fixed and recorded BEFORE the run, never adjusted afterward to
    pass)."""

    name: str
    family: str
    model_options: Mapping[str, Any]
    prior_overrides: Mapping[str, Any]
    draw_prior: Callable[[np.random.Generator], dict]
    simulate: Callable[[dict, np.random.Generator], Any]
    stan_data: Callable[[Any], dict]
    ranked_params: Sequence[RankedParam]
    n_replications: int
    rank_draws: int
    rank_bins: int
    seed_base: int
    chains: int = 2
    iter_warmup: int = 750
    iter_sampling: int = 750
    adapt_delta: float = 0.95
    max_treedepth: int = 12
    #: If set, render_design_model asserts the resolved render-context
    #: priors equal this dict -- the "rendered prior == sampled prior"
    #: exactness assertion SBC's validity rests on.
    expected_priors: Mapping[str, Any] | None = None

    @property
    def param_labels(self) -> tuple[str, ...]:
        return tuple(p if isinstance(p, str) else p[0] for p in self.ranked_params)


def render_design_model(design: SbcDesign):
    """Render + compile the PRODUCTION template for the design's family/
    options/prior-overrides, through the production spec + override path
    (the same mechanism a user's spec.yaml takes)."""
    entry = get_family(design.family)
    spec = RunSpec.model_validate(
        {
            "model": {"family": design.family, "options": dict(design.model_options)},
            "data": {
                "file": "unused.csv",
                "date_column": "date",
                "mapping": {k: k for k in entry.required_mapping},
            },
            "priors": dict(design.prior_overrides),
        }
    )
    context = build_render_context(spec)
    if design.expected_priors is not None:
        assert context["priors"] == design.expected_priors, (
            f"SBC design {design.name!r}: the rendered prior differs from "
            f"the design's expected prior -- SBC exactness is exactly this "
            f"equality, so the gate must not run."
        )
    source = render_stan_source(entry.template, context)
    model, _ = compile_model(source)
    return model


def _pooled_draws(fit, param: RankedParam) -> np.ndarray:
    if isinstance(param, str):
        return np.asarray(fit.stan_variable(param))
    _, extractor = param
    return np.asarray(extractor(fit))


def write_ranks_csv(path: Path, labels: Sequence[str], ranks: np.ndarray, divergences: list[int], n_done: int) -> None:
    """Persist per-replication ranks incrementally (a multi-hour run must
    not lose everything to a crash near the end)."""
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["replication", *labels, "divergences"])
        for i in range(n_done):
            writer.writerow([i, *ranks[i].tolist(), divergences[i]])


def plot_rank_histograms(path: Path, design: SbcDesign, ranks: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = design.param_labels
    n_rep = ranks.shape[0]
    expected = n_rep / design.rank_bins
    ncol = 5
    nrow = int(np.ceil(len(labels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.5 * nrow), sharey=True, squeeze=False)
    edges = np.linspace(0, design.rank_draws + 1, design.rank_bins + 1)
    for j, name in enumerate(labels):
        ax = axes[j // ncol][j % ncol]
        ax.hist(ranks[:, j], bins=edges, color="steelblue", edgecolor="white")
        ax.axhline(expected, color="firebrick", linestyle="--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("rank")
    for j in range(len(labels), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(
        f"{design.name} SBC rank histograms -- {n_rep} replications "
        f"(dashed line = uniform expectation {expected:.0f}/bin)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def chi2_pvalues(design: SbcDesign, ranks: np.ndarray) -> dict[str, float]:
    from scipy import stats

    edges = np.linspace(0, design.rank_draws + 1, design.rank_bins + 1)
    n_rep = ranks.shape[0]
    pvals: dict[str, float] = {}
    for j, name in enumerate(design.param_labels):
        observed, _ = np.histogram(ranks[:, j], bins=edges)
        expected = np.full(design.rank_bins, n_rep / design.rank_bins)
        chi2 = float(((observed - expected) ** 2 / expected).sum())
        pvals[name] = float(stats.chi2.sf(chi2, df=design.rank_bins - 1))
    return pvals


@dataclass
class SbcRunResult:
    ranks: np.ndarray  # (n_replications, n_params)
    divergences: list[int] = field(default_factory=list)


def run_sbc(
    design: SbcDesign,
    artifact_dir: Path,
    *,
    model=None,
    n_replications: int | None = None,
    progress_every: int = 10,
) -> SbcRunResult:
    """The replication loop, generically: for i in 0..N-1, seed
    ``default_rng(seed_base + i)``, draw truth from the design's prior
    sampler, simulate the dataset, fit the rendered production model at
    sampler seed ``seed_base + i``, rank each ranked parameter's truth
    among ``rank_draws`` evenly thinned pooled posterior draws. Ranks and
    per-rep divergence counts stream to ``artifact_dir/ranks.csv`` every
    ``progress_every`` reps. ``n_replications`` may override the design's
    count for smoke/timing runs ONLY (a gate run uses the pre-registered
    count -- passing a smaller value does not change what is registered)."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if model is None:
        model = render_design_model(design)
    n_rep = design.n_replications if n_replications is None else n_replications
    labels = design.param_labels
    ranks = np.zeros((n_rep, len(labels)), dtype=int)
    divergences: list[int] = []

    for i in range(n_rep):
        rng = np.random.default_rng(design.seed_base + i)
        truth = design.draw_prior(rng)
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
        for j, param in enumerate(design.ranked_params):
            draws = _pooled_draws(fit, param)
            ranks[i, j] = rank_statistic(truth[labels[j]], thin_evenly(draws, design.rank_draws))
        divergences.append(int(np.sum(fit.method_variables()["divergent__"])))

        if (i + 1) % progress_every == 0 or i == n_rep - 1:
            write_ranks_csv(artifact_dir / "ranks.csv", labels, ranks, divergences, i + 1)

    plot_rank_histograms(artifact_dir / "rank_histograms.png", design, ranks)
    return SbcRunResult(ranks=ranks, divergences=divergences)
