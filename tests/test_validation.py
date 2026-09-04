"""``mtk validate`` / ``api.validate`` (S6 WP3): the registered gate
suites, the tier runner, the validation report, and the CLI wiring. The
fast tiers run for real here (seconds); the recovery/sbc tiers are
exercised through tiny stand-in designs so the runner's accept/reject
plumbing is covered without hours of sampling (the real designs run under
``-m slow`` in tests/test_ucsv_gates.py / tests/test_g4_sbc.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from macrotoolkit import api
from macrotoolkit.cli import main
from macrotoolkit.validation.recovery import RecoveryDesign, RecoveryResult, evaluate_recovery
from macrotoolkit.validation.suite import Gate, GateResult, ValidationSuite, run_validation, worst
from specs.schema import get_family

pytestmark = pytest.mark.validation


@pytest.mark.parametrize("family", ["ucsv", "lw_sv"])
def test_fast_tier_runs_and_reports(family: str, tmp_path: Path) -> None:
    result = api.validate(family, tier="fast", out_root=tmp_path)
    assert [g.name for g in result.gates] == ["mirror", "hd_identity"]
    assert result.verdict == "PASS", [g.summary for g in result.gates]
    mirror = result.gates[0]
    assert mirror.metrics["max_abs_diff"] < 1e-8 and mirror.metrics["n_points"] == 25
    assert result.gates[1].metrics["max_abs_error"] < 1e-6
    html = result.report_path.read_text()
    assert f"family {family}" in html and "Verdict: PASS" in html
    assert (result.out_dir / "summary.json").is_file()
    summary = json.loads((result.out_dir / "summary.json").read_text())
    assert summary["verdict"] == "PASS" and len(summary["gates"]) == 2
    assert (result.out_dir / "mirror" / "mirror.json").is_file()


def test_registered_suites_cover_every_tier() -> None:
    for family in ("ucsv", "lw_sv"):
        suite = get_family(family).resolve("validation_suite")
        assert isinstance(suite, ValidationSuite)
        assert {g.tier for g in suite.gates} == {"fast", "recovery", "sbc"}
        assert [g.name for g in suite.for_tier("sbc")] == ["sbc"]
    with pytest.raises(NotImplementedError, match="validation_suite"):
        api.validate("local_level", tier="fast")
    with pytest.raises(ValueError, match="tier"):
        get_family("ucsv").resolve("validation_suite").for_tier("nope")


def test_verdict_aggregation_and_crashing_gate_is_a_fail(tmp_path: Path, monkeypatch) -> None:
    assert worst(["PASS", "WARN"]) == "WARN" and worst(["FAIL", "PASS"]) == "FAIL" and worst([]) == "PASS"

    def ok(_: Path) -> GateResult:
        return GateResult("ok", "fast", "PASS", ["fine"], {"x": 1.0})

    def boom(_: Path) -> GateResult:
        raise RuntimeError("kaboom")

    suite = ValidationSuite("ucsv", (Gate("ok", "fast", "ok gate", ok), Gate("boom", "fast", "crashes", boom)))
    import specs.schema as schema_mod

    entry = get_family("ucsv")
    monkeypatch.setattr(schema_mod.FamilyEntry, "resolve", lambda self, cap, _e=entry.resolve: suite if cap == "validation_suite" else _e(self, cap))
    result = run_validation("ucsv", tier="fast", out_root=tmp_path)
    assert [g.verdict for g in result.gates] == ["PASS", "FAIL"]
    assert "kaboom" in result.gates[1].summary and result.verdict == "FAIL"
    with pytest.raises(ValueError, match="tier must be"):
        Gate("x", "nope", "", ok)


def test_recovery_rule_arithmetic() -> None:
    design = RecoveryDesign(
        name="stub", family="ucsv", model_options={}, prior_overrides={}, draw_truth=lambda r: {},
        simulate=lambda p, r: None, stan_data=lambda d: {}, params=("a", "b"), bias_params=("a",),
        n_datasets=20, seed_base=1,
    )
    inside = np.ones((20, 2), dtype=bool)
    inside[:2, 0] = False
    res = RecoveryResult(inside=inside, median_error={"a": list(np.random.default_rng(0).normal(0, 1, 20))}, divergences=[0] * 20)
    verdict, reasons, metrics = evaluate_recovery(design, res)
    assert verdict == "PASS" and metrics["pooled_coverage"] == 0.95
    inside[:, 1] = False  # pooled 0.45 -> FAIL, b below the floor
    verdict, reasons, _ = evaluate_recovery(design, RecoveryResult(inside=inside, median_error={"a": [0.5] * 20}, divergences=[0] * 20))
    assert verdict == "FAIL" and any("coverage for b" in r for r in reasons) and any("pooled" in r for r in reasons)
    # A constant positive median error is a systematic bias (se 0 -> t 0 is
    # the degenerate no-spread case; use a tight spread instead).
    errs = list(0.5 + 0.01 * np.random.default_rng(1).normal(size=20))
    verdict, reasons, metrics = evaluate_recovery(design, RecoveryResult(inside=np.ones((20, 2), bool), median_error={"a": errs}, divergences=[0] * 20))
    assert verdict == "FAIL" and abs(metrics["bias_t"]["a"]) > 3


def test_cli_validate_delegates_and_prints_gate_lines(monkeypatch, tmp_path: Path) -> None:
    from macrotoolkit.validation.suite import ValidationResult

    fake = ValidationResult(
        family="ucsv", tier="fast",
        gates=[GateResult("mirror", "fast", "PASS", ["ok"]), GateResult("hd_identity", "fast", "WARN", ["meh"])],
        out_dir=tmp_path, report_path=tmp_path / "report.html",
    )
    calls = []
    monkeypatch.setattr(api, "validate", lambda family, tier="fast", out_root=None: calls.append((family, tier)) or fake)
    out = CliRunner().invoke(main, ["validate", "ucsv", "--tier", "fast"])
    assert out.exit_code == 0, out.output
    assert calls == [("ucsv", "fast")]
    assert "mirror [fast]: PASS" in out.output and "Validation of ucsv (fast): WARN" in out.output
    fake.gates[1].verdict = "FAIL"
    out = CliRunner().invoke(main, ["validate", "ucsv"])
    assert out.exit_code == 2
