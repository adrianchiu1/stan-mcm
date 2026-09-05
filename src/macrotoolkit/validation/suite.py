"""One-command validation (S6 WP3): ``mtk validate <family> [--tier ...]``
/ ``api.validate``. A family registers a :class:`ValidationSuite` -- its
gates per tier -- via ``FamilyEntry.validation_suite``; the runner
executes the requested tiers and writes ONE validation report
(``validation/<family>/report.html``, the run-report machinery reused:
same verdict boxes, same self-contained HTML) summarizing PASS/WARN/FAIL
per gate with the same honest verdict style the run diagnostics use.

Tiers (the ladder, ENGINEERING.md):

- ``fast``: identities and mirror gates -- the Stan-vs-Python KF
  cross-check at prior draws of the production render (G1's shape) and
  the historical-decomposition reconstruction identity (G6's shape);
  seconds.
- ``recovery``: parameter recovery on simulated data (G2's shape),
  ~tens of minutes.
- ``sbc``: simulation-based calibration at the family's PRE-REGISTERED
  design (G3/G4's shape; hours) -- the design constants are recorded in
  DECISIONS.md before any run and never adjusted afterward.

Every gate returns a :class:`GateResult` with verdict, reasons and
metrics; the suite's verdict is the worst gate verdict.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

TIERS = ("fast", "recovery", "sbc")
_RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}


@dataclass
class GateResult:
    name: str
    tier: str
    verdict: str  # PASS / WARN / FAIL
    reasons: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        return "; ".join(self.reasons)


@dataclass(frozen=True)
class Gate:
    """One registered gate: ``run(artifact_dir) -> GateResult``."""

    name: str
    tier: str
    description: str
    run: Callable[[Path], GateResult]

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"Gate {self.name!r}: tier must be one of {TIERS}, got {self.tier!r}.")


@dataclass(frozen=True)
class ValidationSuite:
    """A family's registered gates, per tier."""

    family: str
    gates: tuple[Gate, ...]

    def for_tier(self, tier: str) -> tuple[Gate, ...]:
        if tier == "all":
            return self.gates
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS + ('all',)}, got {tier!r}.")
        return tuple(g for g in self.gates if g.tier == tier)


@dataclass
class ValidationResult:
    family: str
    tier: str
    gates: list[GateResult]
    out_dir: Path
    report_path: Path

    @property
    def verdict(self) -> str:
        if not self.gates:
            return "PASS"
        return max((g.verdict for g in self.gates), key=lambda v: _RANK.get(v, 2))


def worst(verdicts) -> str:
    return max(verdicts, key=lambda v: _RANK.get(v, 2), default="PASS")


def run_validation(family: str, tier: str = "fast", out_root: str | Path | None = None, *, progress=None) -> ValidationResult:
    """Run ``family``'s registered gates for ``tier`` (``fast`` |
    ``recovery`` | ``sbc`` | ``all``) and write the validation report."""
    from specs.schema import get_family


    entry = get_family(family)
    suite = entry.resolve("validation_suite")
    if suite is None:
        raise NotImplementedError(
            f"Family {family!r} declares no validation_suite capability in "
            f"FAMILY_REGISTRY (specs/schema) -- register its gate designs in "
            f"macrotoolkit/families/{family}_validation.py first."
        )
    if not isinstance(suite, ValidationSuite):
        raise ValueError(
            f"Family {family!r}'s validation suite is built per MODEL DEFINITION (S7): pass the spec path "
            f"instead -- mtk validate <spec.yaml> --tier {tier} / api.validate(spec)."
        )
    return run_validation_suite(suite, tier=tier, out_root=out_root, progress=progress)


def run_validation_suite(suite: "ValidationSuite", tier: str = "fast", out_root: str | Path | None = None, *, progress=None) -> ValidationResult:
    """Run an in-memory suite's ``tier`` and write its report under
    ``validation/<suite.family>/`` (a ``:`` in the name becomes ``_``)."""
    from macrotoolkit.run import REPO_ROOT

    family = suite.family
    gates = suite.for_tier(tier)
    if not gates:
        raise ValueError(f"Family {family!r} registers no gates in tier {tier!r}.")
    out_dir = (Path(out_root).resolve() if out_root is not None else REPO_ROOT / "validation") / family.replace(":", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[GateResult] = []
    for gate in gates:
        artifact_dir = out_dir / gate.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            res = gate.run(artifact_dir)
        except Exception as exc:  # a crashing gate is a FAIL, never a skip
            res = GateResult(name=gate.name, tier=gate.tier, verdict="FAIL", reasons=[f"gate raised {type(exc).__name__}: {exc}"])
        res.seconds = time.time() - t0
        results.append(res)
        if progress is not None:
            progress(res)
        (artifact_dir / "result.json").write_text(json.dumps(_jsonable(res), indent=2, sort_keys=True))

    report_path = out_dir / "report.html"
    report_path.write_text(render_validation_report(family, tier, gates, results), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps({"family": family, "tier": tier, "verdict": worst(r.verdict for r in results),
                    "gates": [_jsonable(r) for r in results]}, indent=2, sort_keys=True)
    )
    return ValidationResult(family=family, tier=tier, gates=results, out_dir=out_dir, report_path=report_path)


def _jsonable(res: GateResult) -> dict:
    def conv(v):
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if hasattr(v, "item"):
            return v.item()
        return v

    return {"name": res.name, "tier": res.tier, "verdict": res.verdict, "reasons": list(res.reasons),
            "metrics": conv(res.metrics), "seconds": res.seconds, "artifacts": dict(res.artifacts)}


def render_validation_report(family: str, tier: str, gates, results: list[GateResult]) -> str:
    """The self-contained validation report, reusing the run report's CSS
    and verdict box so gate verdicts read exactly like run diagnostics."""
    from macrotoolkit.report import _CSS, _esc, verdict_box_html

    overall = worst(r.verdict for r in results)
    rows = "".join(
        f"<tr><td>{_esc(r.name)}</td><td>{_esc(r.tier)}</td><td>{_esc(r.verdict)}</td>"
        f"<td>{_esc(r.summary)}</td><td>{r.seconds:.1f}s</td></tr>"
        for r in results
    )
    table = ("<table><tr><th>Gate</th><th>Tier</th><th>Verdict</th><th>Summary</th><th>Time</th></tr>" + rows + "</table>")
    sections = []
    for gate, r in zip(gates, results):
        metrics_rows = [(k, _fmt(v)) for k, v in r.metrics.items()]
        sections.append(
            f"<h2>{_esc(r.name)} <small>[{_esc(r.tier)}]</small></h2>\n"
            f"<p class=\"caption\">{_esc(gate.description)}</p>\n"
            + verdict_box_html(r.verdict, r.reasons, metrics_rows)
            + "".join(f"<p class=\"caption\">artifact: <code>{_esc(k)}</code> = {_esc(v)}</p>" for k, v in r.artifacts.items())
        )
    body = (
        f"<h1>Validation report -- family {_esc(family)} (tier: {_esc(tier)})</h1>\n"
        + verdict_box_html(overall, [f"{len(results)} gate(s) run; overall verdict is the worst gate verdict."])
        + table
        + "\n".join(sections)
    )
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>Validation -- {_esc(family)} ({_esc(tier)})</title>\n<style>{_CSS}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, dict):
        return ", ".join(f"{k}={_fmt(x)}" for k, x in v.items())
    return str(v)


# ---------------------------------------------------------------------------
# Generic gate builders (families instantiate these with their declarations)
# ---------------------------------------------------------------------------


def mirror_gate(family: str, spec_builder: Callable[[], Any], df_builder: Callable[[], Any], n_points: int = 25, tolerance: float = 1e-8) -> Gate:
    """G1's shape as a registered fast gate: render + compile the
    production program for ``spec_builder()``, build its Stan data from
    ``df_builder()``, and cross-check ``kf_loglik`` against the Python
    mirror at ``n_points`` prior draws (the family's ``mirror``
    capability)."""

    def _run(artifact_dir: Path) -> GateResult:
        from specs.schema import get_family

        from macrotoolkit.qc import MirrorCheckError, run_mirror_check
        from macrotoolkit.render import compile_model, render_stan_source
        from macrotoolkit.run import build_render_context, build_stan_data

        entry = get_family(family)
        spec = spec_builder().model_copy(update={"qc": spec_builder().qc.model_copy(update={"mirror_points": n_points, "mirror_tolerance": tolerance})})
        model, _ = compile_model(render_stan_source(entry.template, build_render_context(spec)))
        stan_data = build_stan_data(family, df_builder(), spec=spec)
        try:
            rec = run_mirror_check(spec, stan_data, model, entry.resolve("mirror"))
        except MirrorCheckError as exc:
            return GateResult("mirror", "fast", "FAIL", [str(exc)])
        (artifact_dir / "mirror.json").write_text(json.dumps(rec, indent=2))
        return GateResult(
            "mirror", "fast", "PASS",
            [f"max |Stan - Python| KF loglik = {rec['max_abs_diff']:.3e} (max relative {rec['max_rel_diff']:.3e}) over {rec['n_points']} prior draws (gate max({tolerance:.0e}, {rec['rtol']:.0e} * |loglik|))"],
            {"max_abs_diff": rec["max_abs_diff"], "max_rel_diff": rec["max_rel_diff"], "n_points": rec["n_points"], "tolerance": tolerance},
        )

    return Gate("mirror", "fast", f"Stan-vs-Python Kalman-filter log-likelihood at {n_points} prior draws of the production render (G1's shape).", _run)


def recovery_gate(design) -> Gate:
    from macrotoolkit.validation.recovery import evaluate_recovery, run_recovery, write_coverage_csv

    def _run(artifact_dir: Path) -> GateResult:
        result = run_recovery(design)
        write_coverage_csv(artifact_dir / "coverage.csv", design, result)
        verdict, reasons, metrics = evaluate_recovery(design, result)
        return GateResult("recovery", "recovery", verdict, reasons, metrics, artifacts={"coverage": str(artifact_dir / "coverage.csv")})

    return Gate(
        "recovery", "recovery",
        f"{design.name}: {design.n_datasets} simulated datasets at prior-drawn truths; pooled 90%-CI coverage in "
        f"[{design.coverage_low}, {design.coverage_high}], per-parameter floor {design.per_param_floor}, "
        f"bias |t| < {design.bias_t_limit} on {list(design.bias_params)} (G2's shape).",
        _run,
    )


def sbc_gate(design, *, p_floor: float = 0.001, divergence_limit: int | None = None, artifact_root: Path | None = None) -> Gate:
    """A family's PRE-REGISTERED SBC design as a registered gate (crash-
    resumable through ranks.csv in ``artifact_root``, default the gate's
    artifact dir)."""

    def _run(artifact_dir: Path) -> GateResult:
        from macrotoolkit.validation.sbc import chi2_pvalues, run_sbc

        root = Path(artifact_root) if artifact_root is not None else artifact_dir
        res = run_sbc(design, root)
        pvals = chi2_pvalues(design, res.ranks)
        total_div = int(sum(res.divergences))
        reasons: list[str] = []
        verdict = "PASS"
        for name, p in pvals.items():
            if p < p_floor:
                verdict = "FAIL"
                reasons.append(f"rank chi^2 p = {p:.4f} < {p_floor} for {name}")
        if divergence_limit is not None and total_div > divergence_limit:
            verdict = "FAIL"
            reasons.append(f"{total_div} divergent transitions > ceiling {divergence_limit}")
        if not reasons:
            reasons.append(
                f"chi^2 p in [{min(pvals.values()):.3f}, {max(pvals.values()):.3f}] on {len(pvals)} ranked quantities; "
                f"{total_div} divergences over {design.n_replications} replications"
            )
        return GateResult(
            "sbc", "sbc", verdict, reasons,
            {"chi2_p": pvals, "divergences_total": total_div, "n_replications": design.n_replications, "resumed_from": res.resumed_from},
            artifacts={"ranks": str(root / "ranks.csv"), "histograms": str(root / "rank_histograms.png")},
        )

    return Gate(
        "sbc", "sbc",
        f"{design.name}: {design.n_replications} pre-registered SBC replications, ranks from {design.rank_draws} thinned draws, "
        f"{design.rank_bins} chi^2 bins, p-floor {p_floor} (G3/G4's shape).",
        _run,
    )


def hd_identity_gate(family: str, spec_builder: Callable[[], Any], df_builder: Callable[[], Any], reconstruct: Callable[[Any, int], float], n_draws: int = 5, tolerance: float = 1e-6) -> Gate:
    """G6's shape: ``reconstruct(results_like, draw_index) -> max abs
    reconstruction error`` supplied by the family over a prior-point
    results object; PASS when every draw's error is below ``tolerance``."""

    def _run(artifact_dir: Path) -> GateResult:
        errs = [float(reconstruct(spec_builder(), df_builder(), i)) for i in range(n_draws)]
        worst_err = max(errs)
        ok = worst_err < tolerance
        return GateResult(
            "hd_identity", "fast", "PASS" if ok else "FAIL",
            [f"historical-decomposition reconstruction: max |error| = {worst_err:.2e} over {n_draws} simulation-smoother draws (gate {tolerance:.0e})"],
            {"max_abs_error": worst_err, "n_draws": n_draws, "tolerance": tolerance},
        )

    return Gate("hd_identity", "fast", f"Per-period historical-decomposition reconstruction identity over {n_draws} simulation-smoother draws (G6's shape).", _run)
