# S0 Preflight Report

Date: 2026-08-12. Environment: Claude Code remote execution container (ephemeral),
outbound HTTPS via a policy-enforcing egress proxy (`/root/.ccr/README.md`). This
document records ground truth about what this environment can and cannot reach,
so later stages don't re-derive it or silently assume network access that isn't
there.

## 1. PyPI dependency install

**Result: PASS.**

`pypi.org` and `files.pythonhosted.org` are on the proxy's always-allowed list
(confirmed via `/root/.ccr/README.md` noProxy config, independent of
`.claude/settings.json`). Installed into a scratch venv
(`/tmp/.../scratchpad/preflight-venv`, not committed):

```
cmdstanpy numba arviz pydantic jinja2 pytest matplotlib
```

All installed and imported cleanly (`cmdstanpy 1.3.0`). No blockers here — S1 can
declare these as real dependencies (`pyproject.toml`).

## 2. CmdStan install + compile

**Result: PASS, with one required workaround.**

`cmdstanpy.install_cmdstan()`'s default path calls `api.github.com` to resolve
the "latest release" version. `api.github.com` is **not** on the egress
allowlist and returns a hard 403 at the proxy (confirmed via
`http://127.0.0.1:37519/__agentproxy/status` → `recentRelayFailures`, and via a
direct `curl` returning 403). This is an organization policy denial, not a
transient failure — do not retry it.

Direct release-asset downloads from `github.com/stan-dev/cmdstan/releases/download/...`
**do work** (curl confirmed 200). So the fix is: always pass an explicit
`--version` to `install_cmdstan` (or the `version=` kwarg to `cmdstanpy.install_cmdstan()`)
so it skips the `api.github.com` "latest" lookup entirely.

Verified end to end in the scratch venv:

```
python -m cmdstanpy.install_cmdstan --dir <scratch>/.cmdstan --version 2.36.0 --progress
```

This downloaded the CmdStan 2.36.0 source tarball, built it with the system
toolchain (g++ 13.3.0 / GNU Make 4.3, both present, 4 cores / 15 GiB RAM
available; build took ~25 minutes wall time), and produced working
`bin/stanc`, `bin/stansummary`, `bin/diagnose`, etc.

Then compiled and sampled a trivial two-parameter Stan program end to end
via `CmdStanPy`:

```stan
parameters { real y; }
model { y ~ normal(0, 1); }
```

```
m = CmdStanModel(stan_file="smoke.stan")
fit = m.sample(chains=2, iter_warmup=200, iter_sampling=200, seed=1)
```

Result: compiled cleanly, both chains sampled, posterior recovered
`y ≈ N(0,1)` as expected (mean 0.066, sd 1.09, R-hat 1.008). **Full CmdStan
toolchain confirmed working end to end**, not just installed.

**Action for S1 / `render.py` / CI setup:** never call `install_cmdstan`
without pinning `version=`. Pick a version and record it (candidate:
`2.36.0`, the version verified here); document the pin in
`pyproject.toml`/`DECISIONS.md` once S1 lands, since "latest" is unreachable
in this environment class.

## 3. HLW fixtures

**Result: fetch from `www.newyorkfed.org` confirmed blocked; user supplied the data directly instead. Fixtures now present but with a documented model-variant gap — see below and `FIXTURES.md`.**

`tests/fixtures/hlw/HLW_2023_Replication_Code/HLW_2023_Replication_Code/` is
already checked into the repo (added in the initial commit, before this
session). It contains:

- All R source files for the HLW (2023) "COVID-adjusted" three-stage
  estimation (`rstar.stage{1,2,3}.R`, `kalman.*.R`, `run.hlw.{us,ca,ea}.R`,
  `utilities.R`, etc.)
- `HLW_Replication_Code_Guide.pdf` — the documentation note, which is
  genuinely useful: it prints the exact system matrices (H, A, F, Q, R) for
  both the 2023 COVID-adjusted model *and*, for reference in §5.4, the
  original HLW (2017) model that `lw-sv-spec.md` follows.

**Missing at first check:** the `inputData/Holston_Laubach_Williams_estimates.xlsx`
file that every `run.hlw.XX.R` script reads (`read.xlsx("inputData/...")`),
and the published one-sided/smoothed estimates that G5a needs as the
oracle. Neither was in the fixtures directory or anywhere else in the repo
at session start.

**Fetch attempt:** the source is `https://www.newyorkfed.org/research/policy/rstar`
(and the underlying media-library XLSX link on `www.newyorkfed.org`). This
domain **is** listed in `.claude/settings.json`'s `sandbox.allowedDomains`,
but that setting only governs the local Claude Code sandbox — it does not
extend the org-level egress proxy used in this remote execution environment
(this is exactly the caveat `.claude/README.md` already flags). Direct
confirmation:

```
$ curl -sS -m 15 https://www.newyorkfed.org/research/policy/rstar
curl: (56) CONNECT tunnel failed, response 403

$ curl -s http://127.0.0.1:37519/__agentproxy/status | jq .recentRelayFailures
[{"kind":"connect_rejected","detail":"gateway answered 403 to CONNECT (policy denial or upstream failure)","host":"www.newyorkfed.org:443"} , ...]
```

This is an organization policy denial (403 at CONNECT), not a transient
network issue — per the proxy README, it should be reported, not retried.

**Per instructions, I did not improvise a substitute, reconstruct the data
from memory, or skip ahead to Bayesian estimation.** I recorded the
blockage and asked the user how to proceed (`AskUserQuestion`, unanswered —
apparently an unattended run) — but the user then supplied the two NY Fed
workbooks directly mid-session (`Holston_Laubach_Williams_current_estimates.xlsx`,
`Holston_Laubach_Williams_real_time_estimates.xlsx`), now checked into
`tests/fixtures/hlw/data/`. **Data is resolved.** See `FIXTURES.md` for full
inventory.

### A second, independent finding: model-variant mismatch (partially resolved by the new data, decision still open)

The checked-in R code is the **HLW (2023) "COVID-adjusted" variant**
(state vector includes a COVID dummy block `d_t`, a φ parameter, and
time-varying measurement-error scaling `κ_t`; see guide §5.3), which is
architecturally different from the **HLW (2017) model** that
`lw-sv-spec.md` specifies (no COVID terms, no κ_t — see guide §5.4, included
in the same PDF for reference). The newly-supplied `current_estimates.xlsx`
confirms this concretely: its published `Parameters` sheet has `c` freely
estimated (≈1.116 for the US) and nonzero `phi`/`kappa_*` — unambiguously
the 2023 parameterization, not ours.

The good news: `real_time_estimates.xlsx` also contains genuine **HLW
(2017)** one-sided output (vintage sheets `2015Q4`–`2019Q4`; its `info`
sheet says so explicitly: *"Estimates prior to 2022:Q4 come from the model
described in ... Journal of International Economics, 2017"*). The bad
news: those vintages' own MLE parameter values aren't published anywhere in
either workbook, so we can't yet reproduce them exactly — only the current
(2023-model, full-sample) parameters are published. **Net: we have a fully
self-consistent oracle (data + params + output) for the 2023 model, and a
partial (output only, no matching params) one for the 2017 model.** Full
detail and options in `FIXTURES.md`. This still needs a human decision
before S2's G5a test is designed — it does not block S1.

**Update, same session:** resolved. Recommended targeting the 2017 model
for G5a (reasoning in `DECISIONS.md`); the user then supplied the genuine
HLW (2017) replication code (`HLW_2017_Code/`), confirmed by direct
term-by-term comparison against spec §1.2–§1.3 and by grep (zero
COVID/κ/φ references). Neither workbook has 2017-vintage published
parameters or a smoothed series, so the plan is to run this code
ourselves — and **R is now confirmed viable in this environment**: R
4.3.3 + `nloptr`/`mFilter` install via apt (`--no-install-recommends`
needed — a plain `apt-get install` pulls ~886MB of unrelated
GUI/media-codec packages, some of which 404 on a stale mirror index and
abort the whole transaction); `tis` (not packaged for Ubuntu, and direct
CRAN access is blocked the same way `newyorkfed.org` is) builds from
source off `github.com/cran/tis`, the CRAN read-only git mirror, which
*is* reachable. All three load correctly. Not yet run end-to-end — see
`FIXTURES.md` for the concrete plan.

## Summary

| Check | Status |
|---|---|
| PyPI deps installable | PASS |
| CmdStan installs, builds, compiles + samples a trivial model | PASS (requires pinning `version=`, `api.github.com` is blocked) |
| HLW fixtures — code | Present (checked in prior to this session) |
| HLW fixtures — input data / published estimates | **Present** — user-supplied, since the NY Fed fetch is blocked (see `FIXTURES.md`) |
| HLW fixtures — model-variant mismatch (2023 COVID-adjusted vs. 2017 spec model) | **Sharpened by the new data; still needs a decision before S2 G5a work** (see `FIXTURES.md`) |

S0 is otherwise clear to proceed into S1.
