# DEVELOPMENT FIXTURE -- draws thinned x10

Archived 2026-09-02 17:29 UTC from `runs/a00958509083` (S5-decisions item 16).

- `draws.nc` keeps every 10th posterior draw per chain: 6000 -> 600 total draws. Posterior summaries and figures
  from this archive are development-quality ONLY -- **regenerate the
  full run (`mtk run` on the spec below) for publication numbers.**
- `diagnostics.json` is the FULL run's diagnostics record (verdict,
  R-hat/ESS, divergences of the un-thinned run), copied verbatim.
- `spec.yaml`/`outputs.yaml`/`data.snapshot.csv` are byte-identical
  copies; re-running the spec reproduces the full run at this same
  hash (estimation identity is unchanged by archiving).
