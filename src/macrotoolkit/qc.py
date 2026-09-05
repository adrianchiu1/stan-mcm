"""Automatic per-run quality control (S6 WP3): the fit-time
Stan-vs-Python Kalman-filter MIRROR CHECK -- gate G1's discipline
(ENGINEERING.md ladder rung 1) applied automatically to every fit, for
the EXACT rendered program the run samples.

Mechanism: before sampling, draw ``qc.mirror_points`` parameter points
from the family's prior (the family's ``mirror`` capability turns each
draw into Stan inits and evaluates the Python KF mirror on the same
Stan data), run the compiled program with ``fixed_param=True`` for one
iteration per point (one chain per point, each chain initialized at its
point), read the program's ``kf_loglik`` transformed parameter, and
compare. The max abs difference is recorded in ``diagnostics.json``
(``"mirror_check"``) and shown in the report header; past
``qc.mirror_tolerance`` (1e-8, G1's gate) the run FAILS LOUDLY before any
sampling happens (:class:`MirrorCheckError`; the run directory is removed
so no partial dir is left behind).

Cost: one fixed_param CmdStan invocation (K one-iteration chains) plus K
Python filter passes -- well under a second on lw_sv-sized problems,
trivial against NUTS.

A family that has no Kalman-filter likelihood (the local_level toy, which
samples its states directly) declares no ``mirror`` capability; its
record says so (``applicable: false``) rather than silently passing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

#: A family's mirror declaration: ``draw_points(spec, stan_data, n, rng)``
#: returns ``n`` :class:`MirrorPoint`s -- Stan inits (every parameter of
#: the rendered program, vectors as arrays) plus the Python mirror's KF
#: log-likelihood at exactly that point on exactly that data.


@dataclass(frozen=True)
class MirrorPoint:
    inits: dict[str, Any]
    loglik_python: float


@dataclass(frozen=True)
class MirrorDecl:
    draw_points: Callable[[Any, dict, int, np.random.Generator], Sequence[MirrorPoint]]
    #: Name of the Stan quantity carrying the KF log-likelihood.
    stan_variable: str = "kf_loglik"


class MirrorCheckError(RuntimeError):
    """The rendered program's KF log-likelihood disagrees with the Python
    mirror beyond the gate -- the model is not trusted; nothing was
    sampled."""


def _jsonable(inits: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in inits.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def stan_kf_loglik_at_points(model, stan_data: dict, points: Sequence[MirrorPoint], *, seed: int, variable: str = "kf_loglik") -> np.ndarray:
    """Evaluate ``variable`` of the compiled program at each point's inits:
    ONE fixed_param run with one single-iteration chain per point (chain i
    initialized at point i). ``sig_figs=18``: CmdStan's default 6-figure
    CSV output would by itself exceed a 1e-8 gate."""
    fit = model.sample(
        data=stan_data,
        chains=len(points),
        parallel_chains=1,
        iter_warmup=0,
        iter_sampling=1,
        fixed_param=True,
        adapt_engaged=False,
        inits=[_jsonable(p.inits) for p in points],
        seed=seed,
        sig_figs=18,
        show_progress=False,
    )
    values = np.asarray(fit.stan_variable(variable), dtype=np.float64).reshape(-1)
    if values.shape[0] != len(points):
        raise RuntimeError(
            f"mirror check: expected {len(points)} values of {variable!r}, got shape {values.shape}."
        )
    return values


def run_mirror_check(spec, stan_data: dict, model, mirror: MirrorDecl | None) -> dict:
    """The check itself; returns the record stored under
    ``diagnostics.json["mirror_check"]``. Raises :class:`MirrorCheckError`
    past the tolerance."""
    qc = spec.qc
    if not qc.mirror_check:
        return {"enabled": False, "applicable": mirror is not None, "passed": None}
    if mirror is None:
        return {
            "enabled": True,
            "applicable": False,
            "passed": None,
            "note": (
                f"family {spec.model.family!r} declares no mirror capability "
                f"(no Kalman-filter likelihood to cross-check)."
            ),
        }
    rng = np.random.default_rng(qc.mirror_seed)
    points = list(mirror.draw_points(spec, stan_data, qc.mirror_points, rng))
    stan_vals = stan_kf_loglik_at_points(model, stan_data, points, seed=qc.mirror_seed, variable=mirror.stan_variable)
    py_vals = np.array([p.loglik_python for p in points], dtype=np.float64)
    diffs = np.abs(stan_vals - py_vals)
    finite = bool(np.all(np.isfinite(stan_vals)) and np.all(np.isfinite(py_vals)))
    max_diff = float(np.max(diffs)) if finite else float("inf")
    # Per point: |diff| < max(atol, rtol * |loglik|). The relative term
    # (S8; rtol 1e-11 against float64's ~1e-16, a T-step accumulation and an ill-conditioned innovation Cholesky at extreme flat-prior draws)
    # only matters at badly scaled points -- for every hand family and
    # every S7 gate |loglik| ~ 1e2-1e3, where atol = 1e-8 is the binding
    # (and unchanged) rule.
    gates = np.maximum(qc.mirror_tolerance, qc.mirror_rtol * np.abs(py_vals)) if finite else np.zeros_like(diffs)
    max_rel_diff = float(np.max(diffs / np.maximum(1.0, np.abs(py_vals)))) if finite else float("inf")
    passed = finite and bool(np.all(diffs < gates))
    record = {
        "enabled": True,
        "applicable": True,
        "n_points": len(points),
        "tolerance": qc.mirror_tolerance,
        "rtol": qc.mirror_rtol,
        "seed": qc.mirror_seed,
        "max_abs_diff": max_diff,
        "max_rel_diff": max_rel_diff,
        "passed": bool(passed),
        "points": [
            {"stan": float(s), "python": float(p), "abs_diff": float(d)}
            for s, p, d in zip(stan_vals, py_vals, diffs)
        ],
    }
    if not passed:
        worst = int(np.argmax(diffs - gates)) if finite else 0
        raise MirrorCheckError(
            f"KF mirror check FAILED for family {spec.model.family!r}: max |Stan - "
            f"Python| = {max_diff:.3e} over {len(points)} prior draws exceeds the "
            f"gate max({qc.mirror_tolerance:.1e}, {qc.mirror_rtol:.0e} * |loglik|) (worst point {worst}: Stan "
            f"{stan_vals[worst]!r}, Python {py_vals[worst]!r}). The rendered program "
            f"and macrotoolkit.smoother disagree -- nothing was sampled. Fix the "
            f"discrepancy (or, to investigate a run anyway, set qc.mirror_check: "
            f"false in the spec, which is recorded in the run's qc.yaml)."
        )
    return record
