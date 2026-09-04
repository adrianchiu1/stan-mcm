"""Family-declared OUTPUT MODULES (S6 WP1): the one place a family says
what its report/notebook figures are.

Each family registers, via ``FamilyEntry.output_modules`` (a dotted path
to a tuple of :class:`OutputModule`), an ordered list of output modules --
"trend_cycle", "irf", "fan", "hd", "prior_predictive" for lw_sv. Every
module is a pair of callables over the family's loaded results object
(the ``results_loader`` capability's return value): ``compute(results)``
produces the module's data object (per-draw arrays, no banding) and
``plot(data, results)`` turns it into matplotlib ``Figure`` objects --
either one ``Figure`` or a ``dict[str, Figure]`` keyed by series name.

Two consumers share the declaration so they can never drift apart:

- :mod:`macrotoolkit.report` renders every module, in declared order,
  into the self-contained ``report.html`` (headings, figure titles and
  captions come from the module fields below);
- :class:`macrotoolkit.api.Outputs` exposes the same modules to a
  notebook (``run.outputs().figure("trend_cycle")`` returns the live
  ``Figure``, never a file).

Nothing here computes numerics: the callables are the family's own
already-validated ``compute_*_draws`` / ``plot_*`` functions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class OutputModule:
    """One family output module.

    - ``name``: short identifier (``"trend_cycle"``), the key notebooks use.
    - ``heading``: the report section heading (``"3.1 Trend-cycle plots"``).
    - ``figure_title``: the figure title for a single-``Figure`` module, or
      the per-figure title PREFIX for a dict-of-figures module (rendered
      as ``"<prefix>: <key>"``).
    - ``compute``: ``results -> data`` (the module's per-draw data object).
    - ``plot``: ``(data, results) -> Figure | dict[str, Figure]``.
    - ``caption``: optional ``(data, results) -> str``; rendered as the
      figure caption (single figure) or as a paragraph above the figure
      grid (dict of figures).
    - ``dpi``: PNG raster resolution when embedded in the report.
    - ``figure_order``: for dict-of-figures modules, the keys in display
      order (``None`` = the dict's own order).
    """

    name: str
    heading: str
    figure_title: str
    compute: Callable[[Any], Any]
    plot: Callable[[Any, Any], Any]
    caption: Callable[[Any, Any], str] | None = None
    dpi: int = 100
    figure_order: tuple[str, ...] | None = None

    def figures(self, data: Any, results: Any) -> dict[str, Any]:
        """Run ``plot`` and normalize the result to an ordered
        ``{label: Figure}`` dict (a single ``Figure`` maps to
        ``{name: fig}``)."""
        out = self.plot(data, results)
        if isinstance(out, dict):
            keys = list(self.figure_order) if self.figure_order is not None else list(out)
            missing = [k for k in keys if k not in out]
            if missing:
                raise KeyError(
                    f"Output module {self.name!r}: plot returned no figure for "
                    f"{missing}; returned keys {sorted(out)}."
                )
            return {k: out[k] for k in keys}
        return {self.name: out}

    def figure_label(self, key: str) -> str:
        """Display title of one figure of this module."""
        if key == self.name:
            return self.figure_title
        return f"{self.figure_title}: {key}"
