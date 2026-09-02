# runs-archive/ — DEVELOPMENT FIXTURES (draw-thinned), not publication runs

Tracked, draw-thinned (~×10) copies of the two reference SV runs
(S5-decisions item 16), so a fresh container has working output-layer
fixtures from plain git instead of paying 40–80 minutes of sampling per run
before any report/results work.

**Regenerate the full runs for publication numbers**: each archived
directory's `spec.yaml` + `data.snapshot.csv` reproduce the full run at the
SAME hash via `mtk run examples/us_lw_sv/<spec>.yaml` (archiving does not
touch estimation identity). Every directory carries an `ARCHIVE_NOTE.md`
stating its thinning and provenance; `diagnostics.json` is always the FULL
run's record. Created by `scripts/archive_run.py <hash12>`.

Posterior medians from the thinned draws are fine for development and test
fixtures; credible-interval edges and tail quantities are noisier than the
full run's — do not quote them.
