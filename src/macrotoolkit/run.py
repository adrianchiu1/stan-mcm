"""Run identity hashing, CmdStanPy execution, and the immutable run store.

ESTIMATION identity (S5-decisions item 3) = SHA-256 over (canonicalized
spec YAML MINUS the `outputs:` block, data file hash, rendered Stan source
hash, CmdStan version) -- lw-sv-spec.md §2.3 as amended 2026-09-02.
"Canonicalized" means the spec parsed through the Pydantic `RunSpec` model
and re-serialized with sorted keys (`RunSpec.to_estimation_yaml`), so YAML
formatting, comments, or key order never change a run's hash -- only the
validated *estimation-relevant meaning* of the spec does. Report/output
options (`outputs:`) live in the run dir but OUTSIDE the hash, so reports
regenerate freely and report-option schema changes never orphan MCMC runs.

Run store layout: `runs/<hash12>/` containing `spec.yaml` (the canonical
ESTIMATION spec -- the immutable identity record), `outputs.yaml` (the
report/output options -- the one deliberately NON-immutable artifact,
refreshed by an idempotent re-run whose spec carries different options),
`data.snapshot.csv` (byte-identical copy of the source CSV that was
hashed), `draws.nc` (ArviZ InferenceData), `diagnostics.json`, `log.txt`,
and a `_SUCCESS` marker written last, once every artifact above has been
written successfully. `load_run_spec()` reassembles the full RunSpec from
the split (and still loads pre-split run dirs whose spec.yaml carries an
inline outputs block).

Immutability: run directories' ESTIMATION artifacts are never mutated once
written (`outputs.yaml` and `report.html` are the documented exceptions).
  - If `runs/<hash12>/` already exists *with* `_SUCCESS`: this is an
    idempotent no-op for the MCMC -- the identical estimation was already
    run successfully; only `outputs.yaml` may be refreshed (content-
    compared first, so a truly identical re-run rewrites nothing).
  - If it exists *without* `_SUCCESS`: a previous run crashed or was
    interrupted partway through. We raise, telling the human to remove the
    directory manually after inspecting `log.txt` -- we never silently
    overwrite a partial run.

Diagnostics scope (S1 placeholder -- S4 formalizes this per lw-sv-spec.md
§4): divergence count, max-treedepth-hit count, E-BFMI per chain, and
R-hat/bulk-ESS/tail-ESS per parameter (worst-case across each parameter's
elements, since a vector-valued Stan parameter like `mu[1..T]` would
otherwise blow up the JSON with per-index detail). A PASS/WARN/FAIL verdict
is derived from fixed thresholds documented at `compute_diagnostics`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from specs.schema import get_family
from specs.schema.base import RunSpec, SamplerSpec, load_spec

from macrotoolkit.data import load_data
from macrotoolkit.render import compile_model, get_cmdstan_version, render_stan_source, stan_source_hash

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- diagnostics thresholds (S1 placeholder; document rationale, don't hide it) ---
# R-hat: Vehtari et al. (2021) rank-normalized R-hat convention (warn > 1.01);
#   > 1.05 is a much larger red flag, used here as the FAIL line.
# ESS: same paper's rule of thumb that bulk/tail ESS >= 400 is needed for
#   reliable posterior summaries; < 100 is treated as an outright FAIL.
# E-BFMI: Stan's own diagnostic warning threshold is 0.3; < 0.2 is used here
#   as the FAIL line (a materially worse energy transition).
RHAT_WARN = 1.01
RHAT_FAIL = 1.05
ESS_WARN = 400
ESS_FAIL = 100
EBFMI_WARN = 0.3
EBFMI_FAIL = 0.2


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    is_new: bool
    verdict: str | None = None


def compute_run_id(spec: RunSpec, data_hash: str, stan_hash: str, cmdstan_version: str) -> tuple[str, str]:
    """Returns (full_hex_digest, hash12) for the ESTIMATION identity
    (S5-decisions item 3): the hash covers (spec minus ``outputs:``, data,
    rendered Stan source, toolchain version). ``outputs`` is report/output
    configuration only -- it never reaches the sampler -- so it lives in
    the run dir (``outputs.yaml``) OUTSIDE the hash, and report-option
    changes or report-schema evolution never orphan an MCMC run. This was
    the one final hash migration (the payload key is renamed to
    ``estimation_spec`` so the migration is self-describing); the
    estimation identity is stable from here.
    """
    payload = {
        "estimation_spec": spec.to_estimation_yaml(),
        "data_sha256": data_hash,
        "stan_source_sha256": stan_hash,
        "cmdstan_version": cmdstan_version,
    }
    full = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return full, full[:12]


def load_run_spec(run_dir: str | Path) -> RunSpec:
    """Reassemble a completed run directory's full :class:`RunSpec` from
    its split artifacts (S5-decisions item 3): ``spec.yaml`` carries the
    canonical ESTIMATION spec (the immutable identity record) and
    ``outputs.yaml`` the report/output options (outside the identity hash,
    refreshable). Pre-split run dirs -- whose ``spec.yaml`` still carries
    an inline ``outputs:`` block and have no ``outputs.yaml`` -- load
    unchanged (old run dirs remain valid records)."""
    import yaml

    run_dir = Path(run_dir)
    spec_path = run_dir / "spec.yaml"
    if not spec_path.is_file():
        raise FileNotFoundError(
            f"{spec_path} not found -- {run_dir} does not look like a "
            f"completed run directory."
        )
    raw = yaml.safe_load(spec_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{spec_path} must contain a YAML mapping, got {type(raw).__name__}.")
    outputs_path = run_dir / "outputs.yaml"
    if outputs_path.is_file():
        raw["outputs"] = yaml.safe_load(outputs_path.read_text())
    return RunSpec.model_validate(raw)


def lw_mu_h0_anchors(y: np.ndarray, pi: np.ndarray, r: np.ndarray) -> tuple[float, float]:
    """The mu_h0 OLS anchors for the SV variant: mu_h0_s = 2*ln(sigma_hat_
    OLS,s) (h is log-variance, so exp(mu_h0/2) = sigma_hat), where
    sigma_hat_OLS is the residual sd of a rough OLS pass mirroring HLW's
    own stage-3 initialization EXACTLY (rstar.stage3.R lines 22-48;
    user-confirmed definition, DECISIONS.md 2026-08-31):

    - gap proxy: residual of OLS of y on [const, linear trend] over the
      FULL trimmed sample (lag quarters included). y is already 100*ln(GDP)
      (spec §1.1), so no extra x100 -- same units as HLW's `output.gap`.
    - IS: OLS of gap_t on [gap_{t-1}, gap_{t-2}, (r_{t-1}+r_{t-2})/2, 1]
      over the T estimation rows; sigma_hat = sqrt(RSS / (T - 4)).
    - PC: OLS of pi_t on [pi_{t-1}, (pi_{t-2}+pi_{t-3}+pi_{t-4})/3,
      gap_{t-1}], NO intercept; sigma_hat = sqrt(RSS / (T - 3)).

    Deterministic given the trimmed data, so run identity stays
    reproducible. Inputs are the full trimmed arrays INCLUDING the 4
    pre-sample lag quarters (build_lw_regressors' convention).
    """
    y = np.asarray(y, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    n = len(y)
    t_est = n - 4
    if t_est <= 4:
        raise ValueError(
            f"mu_h0 OLS anchor needs at least 9 data rows (4 lag quarters + "
            f"5 estimation quarters, for a positive-dof IS regression); the "
            f"trimmed data has {n}."
        )

    trend = np.column_stack([np.ones(n), np.arange(1, n + 1, dtype=np.float64)])
    gap = y - trend @ np.linalg.lstsq(trend, y, rcond=None)[0]

    y_is = gap[4:n]
    x_is = np.column_stack(
        [
            gap[3 : 3 + t_est],
            gap[2 : 2 + t_est],
            (r[3 : 3 + t_est] + r[2 : 2 + t_est]) / 2.0,
            np.ones(t_est),
        ]
    )
    resid_is = y_is - x_is @ np.linalg.lstsq(x_is, y_is, rcond=None)[0]
    sigma_is = float(np.sqrt(resid_is @ resid_is / (t_est - x_is.shape[1])))

    y_pc = pi[4:n]
    x_pc = np.column_stack(
        [
            pi[3 : 3 + t_est],
            (pi[2 : 2 + t_est] + pi[1 : 1 + t_est] + pi[0:t_est]) / 3.0,
            gap[3 : 3 + t_est],
        ]
    )
    resid_pc = y_pc - x_pc @ np.linalg.lstsq(x_pc, y_pc, rcond=None)[0]
    sigma_pc = float(np.sqrt(resid_pc @ resid_pc / (t_est - x_pc.shape[1])))

    if not (np.isfinite(sigma_is) and np.isfinite(sigma_pc) and sigma_is > 0 and sigma_pc > 0):
        raise ValueError(
            f"mu_h0 OLS anchor produced a non-positive/non-finite residual "
            f"sd (IS {sigma_is!r}, PC {sigma_pc!r}) -- degenerate input "
            f"data? The anchor needs genuine residual variation."
        )
    return 2.0 * float(np.log(sigma_is)), 2.0 * float(np.log(sigma_pc))


def build_stan_data(family_name: str, df: pd.DataFrame) -> dict:
    """Map the loaded/mapped DataFrame to the Stan `data` block for a given
    family.

    `lw_sv` convention (HLW's own, spec §1.2's lag structure): the loaded
    data must INCLUDE four pre-sample lag quarters -- the first estimation
    quarter is row 5 of the trimmed data, because the Phillips curve needs
    pi_{t-4}. `data.sample.start` in the spec therefore points at the first
    LAG quarter, four quarters before the first estimated one.
    """
    if family_name == "local_level":
        return {"T": int(len(df)), "y": df["y"].tolist()}
    if family_name == "lw_sv":
        from macrotoolkit.smoother import build_lw_regressors, default_initial_state

        if len(df) < 5:
            raise ValueError(
                f"lw_sv needs at least 5 data rows (4 pre-sample lag "
                f"quarters + 1 estimation quarter); the trimmed data has "
                f"{len(df)}. Note data.sample.start must include the 4 lag "
                f"quarters before the first estimation quarter."
            )
        yobs, x = build_lw_regressors(
            df["y"].to_numpy(), df["pi"].to_numpy(), df["r"].to_numpy()
        )
        xi00, P00 = default_initial_state(float(df["y"].to_numpy()[4]))
        # The mu_h0 OLS anchors are computed unconditionally: the SV render
        # declares them as data; a no-SV render simply doesn't (CmdStan
        # ignores unused input entries), and they don't enter run identity
        # (the hash covers the raw data file, not this dict).
        mu_h0_is, mu_h0_pc = lw_mu_h0_anchors(
            df["y"].to_numpy(), df["pi"].to_numpy(), df["r"].to_numpy()
        )
        return {
            "T": int(yobs.shape[0]),
            "yobs": yobs,
            "x": x,
            "xi00": xi00,
            "P00": P00,
            "mu_h0_is": mu_h0_is,
            "mu_h0_pc": mu_h0_pc,
        }
    raise NotImplementedError(
        f"build_stan_data has no mapping for model.family {family_name!r}. "
        f"Implemented: 'local_level', 'lw_sv'."
    )


def build_render_context(spec: RunSpec) -> dict:
    """Template context for a family's Jinja render: family options plus
    spec-derived constants. `lw_sv` stamps c and the resolved priors
    (specs/schema/lw_sv.py defaults, overridden per key by the spec's
    `priors:` block -- unknown prior names are a hard error, so a typo'd
    override never silently falls back to the default)."""
    if spec.model.family == "local_level":
        return {}
    if spec.model.family == "lw_sv":
        from specs.schema.lw_sv import (
            DEFAULT_PRIORS,
            NO_SV_ONLY_PRIOR_NAMES,
            SV_ONLY_PRIOR_NAMES,
        )

        sv_on = bool(spec.model.options.sv_shocks)
        inactive = NO_SV_ONLY_PRIOR_NAMES if sv_on else SV_ONLY_PRIOR_NAMES
        priors = {name: dict(entry) for name, entry in DEFAULT_PRIORS.items()}
        for name, override in spec.priors.items():
            if name not in priors:
                raise ValueError(
                    f"priors[{name!r}] is not a parameter of the lw_sv "
                    f"family. Valid names: {sorted(priors)}."
                )
            if name in inactive:
                variant = "sv_shocks: [is, pc]" if sv_on else "sv_shocks: []"
                raise ValueError(
                    f"priors[{name!r}] does not exist in the variant this "
                    f"spec selects ({variant}) -- the override would "
                    f"silently do nothing. These prior names belong only to "
                    f"the other variant: {sorted(inactive)}."
                )
            if not isinstance(override, dict):
                raise ValueError(
                    f"priors[{name!r}] must be a mapping of prior fields to "
                    f"override (e.g. {{sd: 0.5}}), got {override!r}."
                )
            merged = {**priors[name], **override}
            unknown = set(merged) - set(priors[name])
            if unknown:
                raise ValueError(
                    f"priors[{name!r}] has unknown field(s) {sorted(unknown)}; "
                    f"the default entry's fields are {sorted(priors[name])}."
                )
            priors[name] = merged
        # estimate_c is validated false in S2 (specs/schema/lw_sv.py), so c
        # is always the spec §1.3 default here. sv_shocks is [] or the
        # canonical ["is", "pc"] (schema-normalized); the template's SV
        # blocks render iff it is non-empty.
        return {
            "c": 1.0,
            "priors": priors,
            "sv_shocks": list(spec.model.options.sv_shocks),
        }
    raise NotImplementedError(
        f"build_render_context has no context for model.family "
        f"{spec.model.family!r}. Implemented: 'local_level', 'lw_sv'."
    )


def compute_diagnostics(idata: az.InferenceData, sampler_spec: SamplerSpec) -> dict:
    """Compute the diagnostics summary + PASS/WARN/FAIL verdict for a
    completed run. See module docstring for the threshold rationale."""
    ss = idata.sample_stats
    n_divergent = int(ss["diverging"].sum().item())
    treedepth_hits = int((ss["tree_depth"] >= sampler_spec.max_treedepth).sum().item())

    ebfmi_per_chain = [float(x) for x in az.bfmi(idata)]
    ebfmi_min = min(ebfmi_per_chain) if ebfmi_per_chain else float("nan")

    rhat_ds = az.rhat(idata)
    ess_bulk_ds = az.ess(idata, method="bulk")
    ess_tail_ds = az.ess(idata, method="tail")

    params: dict[str, dict] = {}
    for var in rhat_ds.data_vars:
        params[var] = {
            "rhat_max": float(rhat_ds[var].max().values),
            "ess_bulk_min": float(ess_bulk_ds[var].min().values),
            "ess_tail_min": float(ess_tail_ds[var].min().values),
        }

    rhat_max_overall = max((p["rhat_max"] for p in params.values()), default=float("nan"))
    ess_bulk_min_overall = min((p["ess_bulk_min"] for p in params.values()), default=float("nan"))
    ess_tail_min_overall = min((p["ess_tail_min"] for p in params.values()), default=float("nan"))

    reasons: list[str] = []
    verdict = "PASS"

    def escalate(level: str, reason: str) -> None:
        nonlocal verdict
        reasons.append(reason)
        if level == "FAIL":
            verdict = "FAIL"
        elif level == "WARN" and verdict != "FAIL":
            verdict = "WARN"

    if n_divergent > 0:
        escalate("FAIL", f"{n_divergent} divergent transition(s)")
    if treedepth_hits > 0:
        escalate("WARN", f"{treedepth_hits} iteration(s) hit max_treedepth={sampler_spec.max_treedepth}")
    if ebfmi_min < EBFMI_FAIL:
        escalate("FAIL", f"E-BFMI {ebfmi_min:.3f} < {EBFMI_FAIL} on at least one chain (per-chain: {ebfmi_per_chain})")
    elif ebfmi_min < EBFMI_WARN:
        escalate("WARN", f"E-BFMI {ebfmi_min:.3f} < {EBFMI_WARN} on at least one chain (per-chain: {ebfmi_per_chain})")
    if rhat_max_overall > RHAT_FAIL:
        escalate("FAIL", f"max R-hat {rhat_max_overall:.4f} > {RHAT_FAIL}")
    elif rhat_max_overall > RHAT_WARN:
        escalate("WARN", f"max R-hat {rhat_max_overall:.4f} > {RHAT_WARN}")
    if ess_bulk_min_overall < ESS_FAIL or ess_tail_min_overall < ESS_FAIL:
        escalate("FAIL", f"min bulk/tail ESS ({ess_bulk_min_overall:.0f}/{ess_tail_min_overall:.0f}) < {ESS_FAIL}")
    elif ess_bulk_min_overall < ESS_WARN or ess_tail_min_overall < ESS_WARN:
        escalate("WARN", f"min bulk/tail ESS ({ess_bulk_min_overall:.0f}/{ess_tail_min_overall:.0f}) < {ESS_WARN}")

    if not reasons:
        reasons.append("no divergences, treedepth hits, low E-BFMI, high R-hat, or low ESS detected")

    return {
        "verdict": verdict,
        "reasons": reasons,
        "divergences": n_divergent,
        "max_treedepth_hits": treedepth_hits,
        "max_treedepth_config": sampler_spec.max_treedepth,
        "e_bfmi_per_chain": ebfmi_per_chain,
        "rhat_max": rhat_max_overall,
        "ess_bulk_min": ess_bulk_min_overall,
        "ess_tail_min": ess_tail_min_overall,
        "params": params,
        "thresholds": {
            "rhat_warn": RHAT_WARN,
            "rhat_fail": RHAT_FAIL,
            "ess_warn": ESS_WARN,
            "ess_fail": ESS_FAIL,
            "ebfmi_warn": EBFMI_WARN,
            "ebfmi_fail": EBFMI_FAIL,
            "note": (
                "S1 placeholder thresholds (Vehtari et al. 2021 "
                "rank-normalized R-hat/ESS conventions; Stan's own 0.3 "
                "E-BFMI warning heuristic). S4 formalizes diagnostics per "
                "lw-sv-spec.md §4."
            ),
        },
    }


def run(spec_path: str | Path, runs_root: str | Path | None = None) -> RunResult:
    """Execute (or idempotently no-op) the run described by the spec at
    `spec_path`. Returns a `RunResult`."""
    spec_path = Path(spec_path).resolve()
    runs_root_path = Path(runs_root).resolve() if runs_root is not None else REPO_ROOT / "runs"

    spec = load_spec(str(spec_path))
    family = get_family(spec.model.family)

    df, raw_hash, data_path = load_data(spec, spec_path)

    context = build_render_context(spec)
    source = render_stan_source(family.template, context)
    src_hash = stan_source_hash(source)
    cmdstan_version = get_cmdstan_version()

    run_id_full, run_id = compute_run_id(spec, raw_hash, src_hash, cmdstan_version)
    run_dir = runs_root_path / run_id
    success_marker = run_dir / "_SUCCESS"

    if run_dir.exists():
        if success_marker.exists():
            # Idempotent estimation no-op -- but outputs.yaml sits OUTSIDE
            # the identity (S5-decisions item 3), so a re-run whose spec
            # carries different report options refreshes it (content-
            # compared first: an identical re-run leaves every artifact
            # byte- and mtime-untouched).
            outputs_yaml = spec.outputs_to_canonical_yaml()
            outputs_path = run_dir / "outputs.yaml"
            if not outputs_path.is_file() or outputs_path.read_text() != outputs_yaml:
                outputs_path.write_text(outputs_yaml)
            return RunResult(run_id=run_id, run_dir=run_dir, is_new=False)
        raise RuntimeError(
            f"Run directory {run_dir} already exists but has no _SUCCESS "
            f"marker -- a previous run of this identical spec+data likely "
            f"crashed or was interrupted partway through. Inspect "
            f"{run_dir / 'log.txt'} if present, then remove {run_dir} "
            f"manually before re-running. macrotoolkit never overwrites an "
            f"existing run directory automatically."
        )

    run_dir.mkdir(parents=True)
    log_path = run_dir / "log.txt"

    logger = logging.getLogger(f"macrotoolkit.run.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(file_handler)
    cmdstanpy_logger = logging.getLogger("cmdstanpy")
    cmdstanpy_logger.addHandler(file_handler)

    try:
        logger.info("Run id: %s (full digest: %s)", run_id, run_id_full)
        logger.info("Spec file: %s", spec_path)
        logger.info("Model family: %s", spec.model.family)
        logger.info("Data file: %s (sha256=%s)", data_path, raw_hash)
        logger.info("Rendered Stan source sha256: %s", src_hash)
        logger.info("CmdStan version: %s", cmdstan_version)

        # spec.yaml = the canonical ESTIMATION spec (the immutable identity
        # record); outputs.yaml = the report/output options, outside the
        # hash and refreshable on later idempotent re-runs (S5-decisions
        # item 3). load_run_spec() reassembles the full RunSpec.
        (run_dir / "spec.yaml").write_text(spec.to_estimation_yaml())
        (run_dir / "outputs.yaml").write_text(spec.outputs_to_canonical_yaml())
        shutil.copy2(data_path, run_dir / "data.snapshot.csv")

        model, _ = compile_model(source, cmdstan_version)
        logger.info("Compiled model executable: %s", model.exe_file)

        stan_data = build_stan_data(spec.model.family, df)
        logger.info(
            "Sampling: chains=%d warmup=%d sampling=%d adapt_delta=%s max_treedepth=%d seed=%d",
            spec.sampler.chains,
            spec.sampler.warmup,
            spec.sampler.sampling,
            spec.sampler.adapt_delta,
            spec.sampler.max_treedepth,
            spec.sampler.seed,
        )

        fit = model.sample(
            data=stan_data,
            chains=spec.sampler.chains,
            iter_warmup=spec.sampler.warmup,
            iter_sampling=spec.sampler.sampling,
            adapt_delta=spec.sampler.adapt_delta,
            max_treedepth=spec.sampler.max_treedepth,
            seed=spec.sampler.seed,
        )

        observed_key = "yobs" if "yobs" in stan_data else "y"
        idata = az.from_cmdstanpy(posterior=fit, observed_data={observed_key: stan_data[observed_key]})
        draws_path = run_dir / "draws.nc"
        idata.to_netcdf(str(draws_path))
        logger.info("Wrote %s", draws_path)

        diagnostics = compute_diagnostics(idata, spec.sampler)
        (run_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
        logger.info("Diagnostics verdict: %s (reasons: %s)", diagnostics["verdict"], diagnostics["reasons"])

        success_marker.write_text(datetime.now(timezone.utc).isoformat() + "\n")
        logger.info("Run complete: %s", run_dir)

        return RunResult(run_id=run_id, run_dir=run_dir, is_new=True, verdict=diagnostics["verdict"])
    except Exception:
        logger.exception("Run failed")
        raise
    finally:
        logger.removeHandler(file_handler)
        cmdstanpy_logger.removeHandler(file_handler)
        file_handler.close()
