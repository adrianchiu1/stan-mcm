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

**S6 note (2026-09-04):** the S6 Kalman-filter generalization (time-varying
`Q_t`, plus `kf_loglik` as a transformed parameter) changed the rendered
Stan source and therefore every lw_sv identity: `spec_sv.yaml` now
identifies as `8ba1420a4145` and `spec_sv_full_vintage.yaml` as
`ec87f45d0a43`. The two archived directories remain valid immutable
records of the S5 program (numerically identical on the constant-Q path,
proven by the G1 pre-change fixture pin); a regeneration lands under the
new names. `mtk report` / the API read them unchanged.
