"""Coverage for ``macrotoolkit.results_lw`` Part A (``load_lw_run``,
``select_draw_indices``, ``_flatten_posterior``, ``_system_matrices_for_draw``,
``compute_trend_cycle_draws``) and the Part-A/B glue
(``historical_decomposition_for_draw``) -- the plumbing that
``tests/test_g6_hd_identity.py`` (G6 proper, hand-built synthetic states)
never exercises, because it never loads a real run directory.

Two REAL, tiny, completed ``lw_sv`` runs (one no-SV, one SV) drive every test
below -- built once, module-scoped (see ``no_sv_run_dir``/``sv_run_dir``
below), via ``macrotoolkit.run.run()`` with minimal sampler settings
(chains=2, warmup=50, sampling=50 -- ``tests/test_run_store.py``'s own
tiny-settings convention), on ``g1_harness.SYNTHETIC_DATA`` (T=50 raw
quarters -> 46 estimation rows; see that module's own docstring: plausible
magnitudes only, not an economic fixture -- appropriate here since this
file's job is exercising the LOADING/PLUMBING seam, not economic
plausibility -- that is G5a/G5b's job). Both runs write into an isolated
``tmp_path_factory`` directory, per ``tests/conftest.py``'s documented
isolation rule (never the real ``runs/``).

Seeds: sampler seed 20260813 (matches the repo-wide convention, e.g.
``tests/g1_harness.py``'s ``PARAM_SEED``); per-test smoother/HD seeds are
named at each call site below so a failure is reproducible.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from g1_harness import SYNTHETIC_DATA
from macrotoolkit.results_lw import (
    GAP_BARS,
    PI_BARS,
    LWRun,
    _flatten_posterior,
    _system_matrices_for_draw,
    compute_historical_decomposition_draws,
    compute_trend_cycle_draws,
    historical_decomposition_for_draw,
    load_lw_run,
    select_draw_indices,
)
from macrotoolkit.run import run
from specs.schema.base import RunSpec
from specs.schema.lw_sv import ThinSpec

# ---------------------------------------------------------------------------
# Shared fixtures: one small synthetic CSV, two tiny real lw_sv runs (no-SV,
# SV) built from it, and LWRun objects loaded from each.
# ---------------------------------------------------------------------------

_TINY_SAMPLER = {
    "chains": 2,
    "warmup": 50,
    "sampling": 50,
    "adapt_delta": 0.8,
    "max_treedepth": 10,
    "seed": 20260813,
}


def _lw_sv_spec_dict(data_file_name: str, sv_shocks: list[str]) -> dict:
    """A minimal, valid RunSpec-shaped dict for the lw_sv family, with tiny
    sampler settings -- mirrors ``tests/conftest.py``'s
    ``tiny_spec_dict``/``tiny_spec_path`` pattern for local_level, adapted
    for lw_sv's required (y, pi, r) mapping and model.options.sv_shocks
    branch. Column names in ``mapping`` are deliberately distinct from the
    model's own y/pi/r keys, so a passing test genuinely exercises
    ``data.mapping`` wiring rather than coincidentally matching names."""
    return {
        "model": {"family": "lw_sv", "options": {"sv_shocks": sv_shocks, "estimate_c": False}},
        "data": {
            "file": data_file_name,
            "date_column": "date",
            "mapping": {"y": "gdp_log100", "pi": "core_infl", "r": "real_short_rate"},
            "sample": {"start": None, "end": None},
        },
        "priors": {},
        "sampler": dict(_TINY_SAMPLER),
        "outputs": {},
    }


@pytest.fixture(scope="module")
def small_lw_sv_data_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small (T=50 raw quarters -> 46 estimation rows), fast, deterministic
    lw_sv-shaped CSV, built from ``g1_harness.SYNTHETIC_DATA``."""
    d = tmp_path_factory.mktemp("results_lw_data")
    path = d / "lw_sv_data.csv"
    df = pd.DataFrame(
        {
            "date": SYNTHETIC_DATA.dates.strftime("%Y-%m-%d"),
            "gdp_log100": SYNTHETIC_DATA.y,
            "core_infl": SYNTHETIC_DATA.pi,
            "real_short_rate": SYNTHETIC_DATA.r,
        }
    )
    df.to_csv(path, index=False)
    return path


def _make_lw_sv_run(
    data_csv: Path, sv_shocks: list[str], tmp_path_factory: pytest.TempPathFactory, label: str
) -> Path:
    spec_dict = _lw_sv_spec_dict(data_csv.name, sv_shocks)
    spec_path = data_csv.parent / f"spec_{label}.yaml"
    spec_path.write_text(yaml.safe_dump(spec_dict, sort_keys=False))
    runs_root = tmp_path_factory.mktemp(f"results_lw_runs_{label}")
    result = run(spec_path, runs_root=runs_root)
    return result.run_dir


@pytest.fixture(scope="module")
def no_sv_run_dir(small_lw_sv_data_csv: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny, real, completed no-SV (``sv_shocks: []``) lw_sv run directory,
    built once and shared by every test in this file that needs it."""
    return _make_lw_sv_run(small_lw_sv_data_csv, [], tmp_path_factory, "no_sv")


@pytest.fixture(scope="module")
def sv_run_dir(small_lw_sv_data_csv: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny, real, completed SV (``sv_shocks: [is, pc]``) lw_sv run
    directory, built once and shared by every test in this file that needs
    it."""
    return _make_lw_sv_run(small_lw_sv_data_csv, ["is", "pc"], tmp_path_factory, "sv")


@pytest.fixture(scope="module")
def no_sv_lw_run(no_sv_run_dir: Path) -> LWRun:
    return load_lw_run(no_sv_run_dir)


@pytest.fixture(scope="module")
def sv_lw_run(sv_run_dir: Path) -> LWRun:
    return load_lw_run(sv_run_dir)


# ---------------------------------------------------------------------------
# load_lw_run
# ---------------------------------------------------------------------------


def test_load_lw_run_no_sv_shapes_and_flags(no_sv_lw_run: LWRun) -> None:
    r = no_sv_lw_run
    assert r.sv_on is False
    T_full = len(SYNTHETIC_DATA.y)
    T = T_full - 4
    assert r.yobs.shape == (T, 2)
    assert r.x.shape == (T, 6)
    assert r.xi00.shape == (7,)
    assert r.P00.shape == (7, 7)
    assert r.y_full.shape == (T_full,)
    assert r.pi_full.shape == (T_full,)
    assert r.r_full.shape == (T_full,)
    assert len(r.dates) == T
    np.testing.assert_allclose(r.y_full, SYNTHETIC_DATA.y)
    np.testing.assert_allclose(r.pi_full, SYNTHETIC_DATA.pi)
    np.testing.assert_allclose(r.r_full, SYNTHETIC_DATA.r)
    # yobs/x are exactly build_lw_regressors' own output on this run's data.
    from macrotoolkit.smoother import build_lw_regressors

    expected_yobs, expected_x = build_lw_regressors(SYNTHETIC_DATA.y, SYNTHETIC_DATA.pi, SYNTHETIC_DATA.r)
    np.testing.assert_allclose(r.yobs, expected_yobs)
    np.testing.assert_allclose(r.x, expected_x)


def test_load_lw_run_sv_on_true_for_sv_run_same_shapes(sv_lw_run: LWRun, no_sv_lw_run: LWRun) -> None:
    assert sv_lw_run.sv_on is True
    # Same underlying data for both runs -- shapes must match exactly.
    assert sv_lw_run.yobs.shape == no_sv_lw_run.yobs.shape
    assert sv_lw_run.x.shape == no_sv_lw_run.x.shape
    np.testing.assert_allclose(sv_lw_run.yobs, no_sv_lw_run.yobs)
    np.testing.assert_allclose(sv_lw_run.x, no_sv_lw_run.x)


def test_load_lw_run_raises_on_missing_spec_yaml(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="spec.yaml"):
        load_lw_run(tmp_path)


def test_load_lw_run_raises_on_missing_draws_nc(tmp_path: Path) -> None:
    spec = RunSpec.model_validate(
        {
            "model": {"family": "local_level", "options": {}},
            "data": {"file": "data.csv", "date_column": "date", "mapping": {"y": "obs"}},
        }
    )
    (tmp_path / "spec.yaml").write_text(spec.to_canonical_yaml())
    with pytest.raises(FileNotFoundError, match="draws.nc"):
        load_lw_run(tmp_path)


def test_load_lw_run_raises_on_non_lw_sv_family(tmp_path: Path) -> None:
    """A local_level run directory (spec.yaml + a draws.nc placeholder --
    the family check happens before draws.nc is ever parsed as netCDF, so an
    empty placeholder file is enough) must be rejected with a clear,
    family-naming error."""
    spec = RunSpec.model_validate(
        {
            "model": {"family": "local_level", "options": {}},
            "data": {"file": "data.csv", "date_column": "date", "mapping": {"y": "obs"}},
        }
    )
    (tmp_path / "spec.yaml").write_text(spec.to_canonical_yaml())
    (tmp_path / "draws.nc").write_bytes(b"")
    with pytest.raises(ValueError, match="lw_sv"):
        load_lw_run(tmp_path)


# ---------------------------------------------------------------------------
# select_draw_indices
# ---------------------------------------------------------------------------


def test_select_draw_indices_all_returns_arange() -> None:
    idx = select_draw_indices(37, "all")
    np.testing.assert_array_equal(idx, np.arange(37))


def test_select_draw_indices_thin_returns_every_kth() -> None:
    idx = select_draw_indices(37, ThinSpec(thin=5))
    np.testing.assert_array_equal(idx, np.arange(0, 37, 5))


def test_select_draw_indices_thin_one_equals_all() -> None:
    idx_thin1 = select_draw_indices(20, ThinSpec(thin=1))
    idx_all = select_draw_indices(20, "all")
    np.testing.assert_array_equal(idx_thin1, idx_all)


def test_select_draw_indices_malformed_value_raises() -> None:
    with pytest.raises(ValueError, match="smoother_draws"):
        select_draw_indices(10, "bogus")
    with pytest.raises(ValueError, match="smoother_draws"):
        select_draw_indices(10, None)
    with pytest.raises(ValueError, match="smoother_draws"):
        select_draw_indices(10, 42)


# ---------------------------------------------------------------------------
# _flatten_posterior
# ---------------------------------------------------------------------------

_STATIC = {"a1", "a2", "a_r", "b_pi", "b_y", "sigma_ystar", "sigma_g", "sigma_z"}


def test_flatten_posterior_no_sv_shape_and_variable_set(no_sv_lw_run: LWRun) -> None:
    flat = _flatten_posterior(no_sv_lw_run)
    post = no_sv_lw_run.idata.posterior
    n_total = post.sizes["chain"] * post.sizes["draw"]
    assert set(flat) == _STATIC | {"sigma_is", "sigma_pc"}
    for name, arr in flat.items():
        assert arr.shape[0] == n_total, name
        assert np.all(np.isfinite(arr)), name


def test_flatten_posterior_sv_shape_and_variable_set(sv_lw_run: LWRun) -> None:
    flat = _flatten_posterior(sv_lw_run)
    post = sv_lw_run.idata.posterior
    n_total = post.sizes["chain"] * post.sizes["draw"]
    assert set(flat) == _STATIC | {"h_is", "h_pc"}
    T = sv_lw_run.yobs.shape[0]
    for name in _STATIC:
        assert flat[name].shape == (n_total,), name
    assert flat["h_is"].shape == (n_total, T)
    assert flat["h_pc"].shape == (n_total, T)
    assert np.all(np.isfinite(flat["h_is"]))
    assert np.all(np.isfinite(flat["h_pc"]))


def test_flatten_posterior_raises_naming_missing_variable_on_sv_mismatch(no_sv_lw_run: LWRun) -> None:
    """Forcing ``sv_on=True`` against a genuinely no-SV run's posterior (which
    has no ``h_is``/``h_pc`` variables) must raise, naming the missing
    variable -- not silently return whatever partial dict it managed to
    build."""
    mismatched = dataclasses.replace(no_sv_lw_run, sv_on=True)
    with pytest.raises(ValueError, match="h_is"):
        _flatten_posterior(mismatched)


def test_flatten_posterior_raises_naming_missing_variable_on_no_sv_mismatch(sv_lw_run: LWRun) -> None:
    """The mirror-image mismatch: forcing ``sv_on=False`` against a genuinely
    SV run's posterior (no ``sigma_is``/``sigma_pc`` variables -- SV runs
    only save ``h_is``/``h_pc``) must likewise raise, naming the missing
    variable."""
    mismatched = dataclasses.replace(sv_lw_run, sv_on=False)
    with pytest.raises(ValueError, match="sigma_is"):
        _flatten_posterior(mismatched)


# ---------------------------------------------------------------------------
# _system_matrices_for_draw
# ---------------------------------------------------------------------------


def test_system_matrices_for_draw_no_sv_has_no_h_paths(no_sv_lw_run: LWRun) -> None:
    flat = _flatten_posterior(no_sv_lw_run)
    i = 0
    F, Q, A, Z, R, h_is, h_pc = _system_matrices_for_draw(flat, i, sv_on=False)
    assert h_is is None
    assert h_pc is None
    assert F.shape == (7, 7)
    assert Q.shape == (7, 7)
    assert A.shape == (6, 2)
    assert Z.shape == (2, 7)
    assert R.shape == (2, 2)
    expected_R = np.diag([flat["sigma_is"][i] ** 2, flat["sigma_pc"][i] ** 2])
    np.testing.assert_allclose(R, expected_R)


def test_system_matrices_for_draw_sv_has_real_h_paths_and_R_reflects_them(sv_lw_run: LWRun) -> None:
    flat = _flatten_posterior(sv_lw_run)
    T = sv_lw_run.yobs.shape[0]
    i = 3
    F, Q, A, Z, R, h_is, h_pc = _system_matrices_for_draw(flat, i, sv_on=True)
    assert h_is is not None and h_pc is not None
    assert h_is.shape == (T,)
    assert h_pc.shape == (T,)
    np.testing.assert_allclose(h_is, flat["h_is"][i])
    np.testing.assert_allclose(h_pc, flat["h_pc"][i])
    assert R.shape == (T, 2, 2)
    # R's diagonal at a few periods equals exp(h_is[t])/exp(h_pc[t]) for
    # THIS draw's own h paths -- confirms the sigma_is=sigma_pc=1.0
    # placeholder used to build F/Q/A/Z never leaks into R (numerics-
    # reviewer's specific concern: a real bug here would silently return
    # diag(1.0, 1.0) instead of the SV time-varying path).
    for t in (0, T // 2, T - 1):
        assert R[t, 0, 0] == pytest.approx(np.exp(h_is[t]))
        assert R[t, 1, 1] == pytest.approx(np.exp(h_pc[t]))
        assert R[t, 0, 1] == 0.0
        assert R[t, 1, 0] == 0.0


def test_system_matrices_for_draw_F_Q_A_Z_independent_of_sv_placeholder(sv_lw_run: LWRun, no_sv_lw_run: LWRun) -> None:
    """F/Q/A/Z do not depend on sigma_is/sigma_pc at all (per
    ``_system_matrices_for_draw``'s own docstring) -- so an SV draw built
    from a static-parameter point that happens to match a no-SV draw's own
    static parameters should produce identical F/Q/A/Z regardless of the
    sigma_is=sigma_pc=1.0 placeholder. Cheap regression guard on that
    documented independence."""
    flat_no_sv = _flatten_posterior(no_sv_lw_run)
    flat_sv = dict(flat_no_sv)
    # Reuse the no-SV draw's static params but bolt on fake h paths so the
    # SV branch of _system_matrices_for_draw is exercised with a KNOWN
    # static-parameter point.
    T = no_sv_lw_run.yobs.shape[0]
    flat_sv["h_is"] = np.zeros((flat_no_sv["a1"].shape[0], T))
    flat_sv["h_pc"] = np.zeros((flat_no_sv["a1"].shape[0], T))
    i = 0
    F0, Q0, A0, Z0, _, _, _ = _system_matrices_for_draw(flat_no_sv, i, sv_on=False)
    F1, Q1, A1, Z1, _, _, _ = _system_matrices_for_draw(flat_sv, i, sv_on=True)
    np.testing.assert_allclose(F0, F1)
    np.testing.assert_allclose(Q0, Q1)
    np.testing.assert_allclose(A0, A1)
    np.testing.assert_allclose(Z0, Z1)


# ---------------------------------------------------------------------------
# compute_trend_cycle_draws
# ---------------------------------------------------------------------------


def test_compute_trend_cycle_draws_no_sv_shapes_and_no_vol(no_sv_lw_run: LWRun) -> None:
    r = no_sv_lw_run
    tcd = compute_trend_cycle_draws(r, seed=20260901)

    post = r.idata.posterior
    n_total = post.sizes["chain"] * post.sizes["draw"]
    expected_idx = select_draw_indices(n_total, r.spec.outputs.smoother_draws)
    np.testing.assert_array_equal(tcd.draw_indices, expected_idx)

    T = r.yobs.shape[0]
    n_draws = len(expected_idx)
    for name, arr in (
        ("ystar", tcd.ystar),
        ("output_gap", tcd.output_gap),
        ("g", tcd.g),
        ("z", tcd.z),
        ("rstar", tcd.rstar),
    ):
        assert arr.shape == (n_draws, T), name
        assert np.all(np.isfinite(arr)), name

    assert tcd.vol_is is None
    assert tcd.vol_pc is None

    assert tcd.y.shape == (T,)
    assert tcd.pi.shape == (T,)
    np.testing.assert_allclose(tcd.y, r.yobs[:, 0])
    np.testing.assert_allclose(tcd.pi, r.yobs[:, 1])

    # Reporting-mapping pins (copied verbatim from
    # tests/test_g5a_hlw_replication.py::_series_from_states, per this
    # module's own docstring): rstar = g + z, output_gap = y - ystar.
    np.testing.assert_allclose(tcd.rstar, tcd.g + tcd.z)
    np.testing.assert_allclose(tcd.output_gap, tcd.y[None, :] - tcd.ystar)


def test_compute_trend_cycle_draws_sv_populates_vol_paths_exp_h_over_2(sv_lw_run: LWRun) -> None:
    r = sv_lw_run
    tcd = compute_trend_cycle_draws(r, seed=20260902)

    T = r.yobs.shape[0]
    n_draws = len(tcd.draw_indices)
    assert tcd.vol_is is not None
    assert tcd.vol_pc is not None
    assert tcd.vol_is.shape == (n_draws, T)
    assert tcd.vol_pc.shape == (n_draws, T)
    assert np.all(np.isfinite(tcd.vol_is))
    assert np.all(np.isfinite(tcd.vol_pc))
    # h is log-VARIANCE (spec §1.5): the reported volatility path is
    # exp(h/2), never exp(h) -- checked here through the REAL code path
    # (compute_trend_cycle_draws), against that draw's own saved h_is/h_pc,
    # not a reimplementation of the arithmetic (test_g6_hd_identity.py's
    # test_volatility_path_convention_is_exp_h_over_2 already pins the
    # standalone arithmetic; this is its end-to-end counterpart).
    flat = _flatten_posterior(r)
    for j, i in enumerate(tcd.draw_indices):
        np.testing.assert_allclose(tcd.vol_is[j], np.exp(flat["h_is"][i] / 2.0))
        np.testing.assert_allclose(tcd.vol_pc[j], np.exp(flat["h_pc"][i] / 2.0))
        # Negative check: exp(h) (the variance) would give a materially
        # different, wrong number almost everywhere -- guards against the
        # classic log-variance-vs-log-sd mixup leaking into this call site.
        assert not np.allclose(tcd.vol_is[j], np.exp(flat["h_is"][i]))

    np.testing.assert_allclose(tcd.y, r.yobs[:, 0])
    np.testing.assert_allclose(tcd.pi, r.yobs[:, 1])


def test_compute_trend_cycle_draws_honors_thin_smoother_draws(no_sv_lw_run: LWRun) -> None:
    """``outputs.smoother_draws`` drives how many draws get processed --
    swap in a ``ThinSpec`` (a real spec field this module reads, not a
    monkeypatch of its own logic) and confirm the draw count/indices follow."""
    thinned = dataclasses.replace(
        no_sv_lw_run, spec=no_sv_lw_run.spec.model_copy(update={"outputs": no_sv_lw_run.spec.outputs.model_copy(update={"smoother_draws": ThinSpec(thin=7)})})
    )
    tcd = compute_trend_cycle_draws(thinned, seed=20260903)
    post = thinned.idata.posterior
    n_total = post.sizes["chain"] * post.sizes["draw"]
    expected_idx = np.arange(0, n_total, 7)
    np.testing.assert_array_equal(tcd.draw_indices, expected_idx)
    assert tcd.ystar.shape[0] == len(expected_idx)


# ---------------------------------------------------------------------------
# historical_decomposition_for_draw -- the Part A/B seam.
# ---------------------------------------------------------------------------


def _check_hd_draw_reconstructs_g6_identity(lw_run: LWRun, draw_index: int, seed: int) -> None:
    """Runs the real Part-A-to-Part-B glue for one posterior draw of a
    LOADED, REAL run and checks G6's own 1e-6-per-period sum identities
    against that SAME draw's own state (recovered from the B1 residual
    identity, which test_g6_hd_identity.py separately pins to ~1e-9) and
    ``lw_run.yobs`` (loaded by Part A from the run's own data snapshot) --
    the seam the hand-built-data G6 test never exercises."""
    flat = _flatten_posterior(lw_run)
    rng = np.random.default_rng(seed)
    hd = historical_decomposition_for_draw(lw_run, flat, draw_index, rng)

    xi_draw = (
        hd.state_components["init"]
        + hd.state_components["ystar"]
        + hd.state_components["g"]
        + hd.state_components["z"]
    )

    gap_sum = sum(hd.gap.values())
    expected_gap = lw_run.yobs[:, 0] - xi_draw[:, 0]
    diff_gap = np.abs(gap_sum - expected_gap)
    assert np.all(diff_gap < 1e-6), f"gap HD reconstruction fails: worst={diff_gap.max():.3e} at {int(np.argmax(diff_gap))}"

    pi_sum = sum(hd.pi.values())
    diff_pi = np.abs(pi_sum - lw_run.yobs[:, 1])
    assert np.all(diff_pi < 1e-6), f"pi HD reconstruction fails: worst={diff_pi.max():.3e} at {int(np.argmax(diff_pi))}"

    y_sum = sum(hd.y.values())
    diff_y = np.abs(y_sum - lw_run.yobs[:, 0])
    assert np.all(diff_y < 1e-6), f"y-level HD reconstruction fails: worst={diff_y.max():.3e} at {int(np.argmax(diff_y))}"


def test_historical_decomposition_for_draw_runs_and_reconstructs_no_sv(no_sv_lw_run: LWRun) -> None:
    _check_hd_draw_reconstructs_g6_identity(no_sv_lw_run, draw_index=0, seed=20260904)


def test_historical_decomposition_for_draw_runs_and_reconstructs_sv(sv_lw_run: LWRun) -> None:
    _check_hd_draw_reconstructs_g6_identity(sv_lw_run, draw_index=0, seed=20260905)


def test_historical_decomposition_for_draw_reconstructs_at_a_later_draw_index_too(sv_lw_run: LWRun) -> None:
    """Not just draw 0 -- a mid-batch draw index, so this isn't accidentally
    only exercising the first posterior sample."""
    post = sv_lw_run.idata.posterior
    n_total = post.sizes["chain"] * post.sizes["draw"]
    mid = n_total // 2
    _check_hd_draw_reconstructs_g6_identity(sv_lw_run, draw_index=mid, seed=20260906)


# ---------------------------------------------------------------------------
# compute_historical_decomposition_draws -- the S4-step-4 loop-and-stack
# addition, exercising the SAME G6 sum identity as
# _check_hd_draw_reconstructs_g6_identity above, but across a small BATCH of
# draws (via the aggregation loop) instead of one hand-picked draw index --
# a bug that only shows up from stacking into the wrong row/column would
# slip past the single-draw checks above but not this one.
# ---------------------------------------------------------------------------


def _check_hd_draws_aggregated_shapes_and_identity(lw_run: LWRun, seed: int) -> None:
    hdd = compute_historical_decomposition_draws(lw_run, seed=seed)

    post = lw_run.idata.posterior
    n_total = post.sizes["chain"] * post.sizes["draw"]
    expected_idx = select_draw_indices(n_total, lw_run.spec.outputs.smoother_draws)
    np.testing.assert_array_equal(hdd.draw_indices, expected_idx)

    T = lw_run.yobs.shape[0]
    n_draws = len(expected_idx)
    assert len(hdd.dates) == T
    pd.testing.assert_index_equal(hdd.dates, lw_run.dates)

    assert set(hdd.gap) == set(GAP_BARS)
    assert set(hdd.pi) == set(PI_BARS)
    assert set(hdd.y) == set(PI_BARS)
    assert set(hdd.y_growth_4q) == set(PI_BARS)
    assert set(hdd.state_components) == {"init", "ystar", "g", "z"}

    for k, arr in hdd.gap.items():
        assert arr.shape == (n_draws, T), k
        assert np.all(np.isfinite(arr)), k
    for k, arr in hdd.pi.items():
        assert arr.shape == (n_draws, T), k
        assert np.all(np.isfinite(arr)), k
    for k, arr in hdd.y.items():
        assert arr.shape == (n_draws, T), k
        assert np.all(np.isfinite(arr)), k
    for k, arr in hdd.y_growth_4q.items():
        assert arr.shape == (n_draws, T), k
        # First 4 columns are NaN by four_quarter_growth's own convention;
        # the rest must be finite.
        assert np.all(np.isnan(arr[:, :4])), k
        assert np.all(np.isfinite(arr[:, 4:])), k
    for k, arr in hdd.state_components.items():
        assert arr.shape == (n_draws, T, 7), k
        assert np.all(np.isfinite(arr)), k

    # Same G6 sum identity as _check_hd_draw_reconstructs_g6_identity, now
    # checked for EVERY draw in the batch at once (a stacking-into-the-
    # wrong-row bug would break this even though each draw's own HDDraw,
    # pre-stacking, individually satisfies it).
    xi_draw = (
        hdd.state_components["init"]
        + hdd.state_components["ystar"]
        + hdd.state_components["g"]
        + hdd.state_components["z"]
    )  # (n_draws, T, 7)

    gap_sum = sum(hdd.gap.values())  # (n_draws, T)
    expected_gap = lw_run.yobs[None, :, 0] - xi_draw[:, :, 0]
    diff_gap = np.abs(gap_sum - expected_gap)
    assert np.all(diff_gap < 1e-6), f"gap HD batch reconstruction fails: worst={diff_gap.max():.3e}"

    pi_sum = sum(hdd.pi.values())
    diff_pi = np.abs(pi_sum - lw_run.yobs[None, :, 1])
    assert np.all(diff_pi < 1e-6), f"pi HD batch reconstruction fails: worst={diff_pi.max():.3e}"

    y_sum = sum(hdd.y.values())
    diff_y = np.abs(y_sum - lw_run.yobs[None, :, 0])
    assert np.all(diff_y < 1e-6), f"y-level HD batch reconstruction fails: worst={diff_y.max():.3e}"


def test_compute_historical_decomposition_draws_no_sv(no_sv_lw_run: LWRun) -> None:
    _check_hd_draws_aggregated_shapes_and_identity(no_sv_lw_run, seed=20260910)


def test_compute_historical_decomposition_draws_sv(sv_lw_run: LWRun) -> None:
    _check_hd_draws_aggregated_shapes_and_identity(sv_lw_run, seed=20260911)


def test_compute_historical_decomposition_draws_honors_thin_smoother_draws(no_sv_lw_run: LWRun) -> None:
    """Same ``outputs.smoother_draws`` respect check
    ``test_compute_trend_cycle_draws_honors_thin_smoother_draws`` runs for
    ``compute_trend_cycle_draws``, mirrored here for the new aggregation
    function."""
    thinned = dataclasses.replace(
        no_sv_lw_run,
        spec=no_sv_lw_run.spec.model_copy(
            update={"outputs": no_sv_lw_run.spec.outputs.model_copy(update={"smoother_draws": ThinSpec(thin=7)})}
        ),
    )
    hdd = compute_historical_decomposition_draws(thinned, seed=20260912)
    post = thinned.idata.posterior
    n_total = post.sizes["chain"] * post.sizes["draw"]
    expected_idx = np.arange(0, n_total, 7)
    np.testing.assert_array_equal(hdd.draw_indices, expected_idx)
    assert hdd.gap["init"].shape[0] == len(expected_idx)
