"""Per-family NUMERICS metadata and machinery (S5-decisions items 1-2, 4).

``specs/schema/`` owns each family's *spec-side* fragment (options model,
priors, outputs model, registry entry); this package owns the *numerics-side*
declarations the output engines consume -- named state metadata, structural-
coefficient extraction, and (item 1) the endogenous-lag feedback map. The
split avoids an import cycle: ``specs.schema`` must stay importable without
numpy-heavy machinery, while everything here may import numpy freely.

Doctrine (ENGINEERING.md "Code structure"): downstream code consumes NAMED
metadata, never matrix slots or implicit timing conventions. The modules in
this package are the ONE place per family where the raw slot layout is
written down; everything else looks indices up by ``(name, time_offset)``.
"""
