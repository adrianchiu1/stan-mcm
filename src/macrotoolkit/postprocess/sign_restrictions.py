"""Sign restrictions by rotation (Rubio-Ramirez, Waggoner & Zha 2010;
Blake & Mumtaz 2017 Chapter 2 §6, examples 5-7) as a POST-PROCESSOR over
a run's structural impulse responses.

Given per-draw structural IRFs ``irf[d, h]`` (an ``(m, k)`` matrix: the
response of the ``m`` targets at horizon ``h`` to each of the ``k``
orthogonal unit-variance structural shocks -- for a recursive VAR the
Cholesky IRFs the engine already produces, ``irf[d, 0] = chol(Sigma)``),
any orthonormal ``Q`` (``Q'Q = I``) gives another observationally
equivalent structural model with IRFs ``irf[d, h] Q``. A candidate ``Q``
is Haar-distributed via the QR decomposition of a standard-normal matrix
with the diagonal of ``R`` made positive (the handbook's ``getqr``);
candidates are accepted when the restricted shocks' responses carry the
required signs at the required horizons (a column may be flipped -- the
handbook's "check signs but reverse them"), rejected otherwise, up to a
try cap per draw. Restricted shocks are matched to candidate columns by
search (the handbook scans the rows of ``Q A0`` for the pattern and moves
the identified shock first), so the retained IRFs list the restricted
shocks FIRST, in the order given, then the unrestricted columns.

The "closest to median" variant (example 7): per draw, ``closest_to_median``
accepted rotations are collected and the one whose impact matrix is
nearest (squared distance over its entries) to their element-wise median
is kept -- one rotation per posterior draw, an ad-hoc choice the handbook
makes and Fry & Pagan (2011) criticize; off by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SignRestriction:
    """``sign`` (+1 / -1) of the response of ``variable`` to the shock
    labelled ``shock`` at every horizon in ``horizons`` (0 = impact)."""

    shock: str
    variable: str
    sign: int
    horizons: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if self.sign not in (1, -1):
            raise ValueError(f"sign must be +1 or -1; got {self.sign!r}.")
        if not self.horizons or any(h < 0 for h in self.horizons):
            raise ValueError(f"horizons must be non-empty and >= 0; got {self.horizons!r}.")


def haar_rotation(n: int, rng: np.random.Generator) -> np.ndarray:
    """A Haar-distributed orthonormal ``(n, n)`` matrix: ``Q`` from the QR
    decomposition of a standard-normal matrix with each column signed so
    the corresponding diagonal of ``R`` is positive (``getqr``)."""
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs[None, :]


@dataclass
class SignRestrictedIRFs:
    """``irf`` is ``(n_kept, H, m, k)`` in the identified basis (restricted
    shocks first, in ``shock_order``); ``draw_index[i]`` is the posterior
    draw the i-th kept rotation came from; ``rotations[i]`` the ``(k, k)``
    orthonormal matrix (columns permuted/flipped as applied);
    ``n_tries[i]`` the candidates drawn for that draw; ``rejected_draws``
    the draws for which no candidate was accepted within ``max_tries``."""

    irf: np.ndarray
    draw_index: np.ndarray
    rotations: np.ndarray
    n_tries: np.ndarray
    rejected_draws: np.ndarray
    targets: tuple[str, ...]
    shock_order: tuple[str, ...]

    def response(self, shock: str, variable: str) -> np.ndarray:
        """``(n_kept, H)`` responses of ``variable`` to ``shock``."""
        return self.irf[:, :, self.targets.index(variable), self.shock_order.index(shock)]

    def bands(self, shock: str, variable: str, quantiles: Sequence[float] = (0.16, 0.5, 0.84)) -> np.ndarray:
        """``(len(quantiles), H)`` pointwise quantiles of :meth:`response`."""
        return np.quantile(self.response(shock, variable), quantiles, axis=0)


def _satisfies(col: np.ndarray, checks: Sequence[tuple[int, int, int]]) -> bool:
    """``col`` is ``(H, m)``; ``checks`` are ``(horizon, variable_index, sign)``."""
    return all(sign * col[h, v] > 0.0 for h, v, sign in checks)


def _match(candidate: np.ndarray, groups: Sequence[Sequence[tuple[int, int, int]]], allow_flip: bool) -> list[tuple[int, float]] | None:
    """Assign each restricted-shock group to a distinct column of the
    rotated IRF ``candidate`` ``(H, m, k)`` (with a sign flip when
    allowed). Depth-first over columns; returns ``[(column, flip)]`` per
    group or ``None``."""
    k = candidate.shape[2]

    def rec(g: int, used: set[int]) -> list[tuple[int, float]] | None:
        if g == len(groups):
            return []
        for j in range(k):
            if j in used:
                continue
            col = candidate[:, :, j]
            for flip in ((1.0, -1.0) if allow_flip else (1.0,)):
                if _satisfies(flip * col, groups[g]):
                    rest = rec(g + 1, used | {j})
                    if rest is not None:
                        return [(j, flip)] + rest
        return None

    return rec(0, set())


def sign_restricted_irfs(
    irf: np.ndarray,
    restrictions: Sequence[SignRestriction],
    targets: Sequence[str],
    rng: np.random.Generator,
    *,
    max_tries: int = 1000,
    allow_flip: bool = True,
    closest_to_median: int | None = None,
    shock_names: Sequence[str] | None = None,
) -> SignRestrictedIRFs:
    """Rejection sampling over Haar rotations of ``irf`` ``(n_draws, H, m,
    k)`` (see the module doc). ``shock_names`` labels the UNRESTRICTED
    columns in the output (default ``shock_<j>``)."""
    irf = np.asarray(irf, dtype=np.float64)
    if irf.ndim != 4:
        raise ValueError(f"irf must be (n_draws, H, m, k); got shape {irf.shape}.")
    n_draws, H, m, k = irf.shape
    targets = tuple(targets)
    if len(targets) != m:
        raise ValueError(f"targets ({len(targets)}) must match the IRF's variable dimension m = {m}.")
    if not restrictions:
        raise ValueError("at least one SignRestriction is needed.")
    labels: list[str] = []
    for r in restrictions:
        if r.shock not in labels:
            labels.append(r.shock)
        if r.variable not in targets:
            raise ValueError(f"restriction on unknown variable {r.variable!r}; targets: {targets}.")
        if max(r.horizons) >= H:
            raise ValueError(f"restriction horizon {max(r.horizons)} beyond the IRF horizon H = {H}.")
    if len(labels) > k:
        raise ValueError(f"{len(labels)} restricted shocks but only k = {k} structural shocks.")
    groups = [[(h, targets.index(r.variable), r.sign) for r in restrictions if r.shock == lab for h in r.horizons] for lab in labels]
    n_free = k - len(labels)
    free_names = tuple(shock_names) if shock_names is not None else tuple(f"shock_{j}" for j in range(n_free))
    if len(free_names) != n_free:
        raise ValueError(f"shock_names must label the {n_free} unrestricted columns; got {len(free_names)}.")
    kept, draw_index, rotations, n_tries, rejected = [], [], [], [], []
    for d in range(n_draws):
        accepted: list[tuple[np.ndarray, np.ndarray]] = []
        tries = 0
        want = closest_to_median or 1
        while tries < max_tries and len(accepted) < want:
            tries += 1
            Q = haar_rotation(k, rng)
            cand = irf[d] @ Q  # (H, m, k)
            match = _match(cand, groups, allow_flip)
            if match is None:
                continue
            cols = [j for j, _ in match] + [j for j in range(k) if j not in {jj for jj, _ in match}]
            flips = np.array([f for _, f in match] + [1.0] * n_free)
            P = np.zeros((k, k))
            for new, old in enumerate(cols):
                P[old, new] = flips[new]
            accepted.append((cand @ P, Q @ P))
        n_tries.append(tries)
        if not accepted:
            rejected.append(d)
            continue
        if closest_to_median and len(accepted) > 1:
            impacts = np.stack([a[0][0] for a in accepted]).reshape(len(accepted), -1)
            med = np.median(impacts, axis=0)
            best = int(np.argmin(np.sum((impacts - med[None, :]) ** 2, axis=1)))
            accepted = [accepted[best]]
        for irf_rot, Q_rot in accepted:
            kept.append(irf_rot)
            rotations.append(Q_rot)
            draw_index.append(d)
    return SignRestrictedIRFs(
        irf=np.stack(kept) if kept else np.zeros((0, H, m, k)),
        draw_index=np.asarray(draw_index, dtype=int),
        rotations=np.stack(rotations) if rotations else np.zeros((0, k, k)),
        n_tries=np.asarray(n_tries, dtype=int),
        rejected_draws=np.asarray(rejected, dtype=int),
        targets=targets,
        shock_order=tuple(labels) + free_names,
    )


def irf_array_from_draws(irf_draws, shocks: Sequence[str] | None = None, targets: Sequence[str] | None = None) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """An authored run's ``IRFDraws`` (``responses[shock][target]`` =
    ``(n_draws, H)`` at 1-sd sizes) as the ``(n_draws, H, m, k)`` array the
    post-processors take, over the given shocks (default: all) and targets
    (default: the observables). For a recursive VAR with all measurement
    shocks this is the Cholesky IRF array."""
    shocks = tuple(shocks) if shocks is not None else tuple(irf_draws.shocks)
    targets = tuple(targets) if targets is not None else tuple(t for t in irf_draws.targets if t not in getattr(irf_draws, "state_targets", ()))
    first = irf_draws.responses[shocks[0]][targets[0]]
    n, H = first.shape
    out = np.empty((n, H, len(targets), len(shocks)))
    for j, s in enumerate(shocks):
        for i, t in enumerate(targets):
            out[:, :, i, j] = irf_draws.responses[s][t]
    return out, targets, shocks
