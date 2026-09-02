"""RunSpec validation tests (lw-sv-spec.md §2.3, plans/S1-plan.md).

Contract under test (from specs/schema/base.py's module docstring): every
validation error must name the offending field and state what a valid value
looks like; every model uses `extra="forbid"` so unknown keys are rejected
loudly.
"""
from __future__ import annotations

import copy

import pydantic
import pytest
import yaml

from specs.schema.base import RunSpec, load_spec


def _valid_spec_dict() -> dict:
    return {
        "model": {"family": "local_level", "options": {}},
        "data": {
            "file": "data.csv",
            "date_column": "date",
            "mapping": {"y": "obs"},
        },
    }


def test_valid_spec_loads() -> None:
    spec = RunSpec.model_validate(_valid_spec_dict())
    assert spec.model.family == "local_level"
    assert spec.data.file == "data.csv"
    assert spec.data.date_column == "date"
    assert spec.data.mapping == {"y": "obs"}
    # Defaults filled in.
    assert spec.sampler.chains == 4
    assert spec.sampler.warmup == 1000
    assert spec.sampler.sampling == 1000
    assert spec.sampler.adapt_delta == 0.8
    assert spec.sampler.max_treedepth == 10
    assert spec.data.sample.start is None
    assert spec.data.sample.end is None
    assert spec.priors == {}
    assert spec.outputs == {}


def test_unknown_family_names_field_and_gives_example() -> None:
    d = _valid_spec_dict()
    d["model"]["family"] = "bogus_family"
    with pytest.raises(pydantic.ValidationError) as exc_info:
        RunSpec.model_validate(d)
    msg = str(exc_info.value)
    assert "bogus_family" in msg
    assert "local_level" in msg  # names the one valid family currently registered
    assert "model.options" in msg or "family:" in msg  # gives a concrete example


def test_missing_required_mapping_key_for_family() -> None:
    d = _valid_spec_dict()
    d["data"]["mapping"] = {"not_y": "obs"}  # local_level requires "y"
    with pytest.raises(pydantic.ValidationError) as exc_info:
        RunSpec.model_validate(d)
    msg = str(exc_info.value)
    assert "data.mapping" in msg
    assert "'y'" in msg or '"y"' in msg
    assert "local_level" in msg


@pytest.mark.parametrize(
    "overrides,expected_loc_substring",
    [
        ({"chains": 0}, "sampler.chains"),
        ({"warmup": -1}, "sampler.warmup"),
        ({"sampling": 0}, "sampler.sampling"),
        ({"adapt_delta": 0.0}, "sampler.adapt_delta"),
        ({"adapt_delta": 1.0}, "sampler.adapt_delta"),
        ({"adapt_delta": -0.1}, "sampler.adapt_delta"),
        ({"max_treedepth": 0}, "sampler.max_treedepth"),
    ],
)
def test_sampler_bounds_rejected(overrides: dict, expected_loc_substring: str) -> None:
    d = _valid_spec_dict()
    d["sampler"] = overrides
    with pytest.raises(pydantic.ValidationError) as exc_info:
        RunSpec.model_validate(d)
    locs = [".".join(str(p) for p in err["loc"]) for err in exc_info.value.errors()]
    assert any(loc.startswith(expected_loc_substring) for loc in locs), locs


def test_empty_data_file_rejected() -> None:
    d = _valid_spec_dict()
    d["data"]["file"] = ""
    with pytest.raises(pydantic.ValidationError) as exc_info:
        RunSpec.model_validate(d)
    msg = str(exc_info.value)
    assert "data.file" in msg
    assert "examples/toy/data.csv" in msg  # gives a concrete valid example


def test_empty_data_file_whitespace_only_rejected() -> None:
    d = _valid_spec_dict()
    d["data"]["file"] = "   "
    with pytest.raises(pydantic.ValidationError) as exc_info:
        RunSpec.model_validate(d)
    assert "data.file" in str(exc_info.value)


def test_empty_date_column_rejected() -> None:
    d = _valid_spec_dict()
    d["data"]["date_column"] = ""
    with pytest.raises(pydantic.ValidationError) as exc_info:
        RunSpec.model_validate(d)
    msg = str(exc_info.value)
    assert "data.date_column" in msg
    assert '"date"' in msg or "'date'" in msg  # gives a concrete valid example


def test_empty_mapping_rejected() -> None:
    d = _valid_spec_dict()
    d["data"]["mapping"] = {}
    with pytest.raises(pydantic.ValidationError) as exc_info:
        RunSpec.model_validate(d)
    msg = str(exc_info.value)
    assert "data.mapping" in msg
    assert "non-empty" in msg


def test_mapping_value_must_be_nonempty_string() -> None:
    d = _valid_spec_dict()
    d["data"]["mapping"] = {"y": ""}
    with pytest.raises(pydantic.ValidationError) as exc_info:
        RunSpec.model_validate(d)
    msg = str(exc_info.value)
    assert "data.mapping" in msg
    assert "'y'" in msg


@pytest.mark.parametrize("field", ["start", "end"])
def test_sample_start_end_empty_string_rejected(field: str) -> None:
    d = _valid_spec_dict()
    d["data"]["sample"] = {field: ""}
    with pytest.raises(pydantic.ValidationError) as exc_info:
        RunSpec.model_validate(d)
    msg = str(exc_info.value)
    assert f"data.sample.{field}" in msg
    assert "non-empty" in msg


def test_sample_start_end_null_is_valid() -> None:
    d = _valid_spec_dict()
    d["data"]["sample"] = {"start": None, "end": None}
    spec = RunSpec.model_validate(d)
    assert spec.data.sample.start is None
    assert spec.data.sample.end is None


@pytest.mark.parametrize(
    "path,bad_key",
    [
        ((), "unexpected_top_level_key"),
        (("model",), "unexpected_model_key"),
        (("data",), "unexpected_data_key"),
        (("sampler",), "unexpected_sampler_key"),
        (("data", "sample"), "unexpected_sample_key"),
    ],
)
def test_extra_forbid_rejects_unknown_keys(path: tuple, bad_key: str) -> None:
    d = _valid_spec_dict()
    target = d
    for p in path:
        target = target.setdefault(p, {})
    target[bad_key] = "surprise"
    with pytest.raises(pydantic.ValidationError) as exc_info:
        RunSpec.model_validate(d)
    msg = str(exc_info.value)
    assert bad_key in msg
    assert "extra" in msg.lower() or "permitted" in msg.lower()


def test_local_level_options_extra_forbid_rejects_unknown_key() -> None:
    d = _valid_spec_dict()
    d["model"]["options"] = {"unsupported_option": 1}
    with pytest.raises(pydantic.ValidationError) as exc_info:
        RunSpec.model_validate(d)
    msg = str(exc_info.value)
    assert "unsupported_option" in msg


def test_model_options_non_dict_rejected() -> None:
    d = _valid_spec_dict()
    d["model"]["options"] = "not-a-dict"
    with pytest.raises(pydantic.ValidationError) as exc_info:
        RunSpec.model_validate(d)
    msg = str(exc_info.value)
    assert "model.options" in msg


def test_load_spec_missing_file(tmp_path) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_spec(str(missing))


def test_load_spec_empty_file_rejected(tmp_path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_spec(str(p))


def test_load_spec_non_mapping_top_level_rejected(tmp_path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="mapping"):
        load_spec(str(p))


def test_load_spec_valid_file(tmp_path) -> None:
    d = _valid_spec_dict()
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(d))
    spec = load_spec(str(p))
    assert spec.model.family == "local_level"


# --- to_canonical_yaml determinism ---------------------------------------


def test_canonical_yaml_is_deterministic_for_identical_content() -> None:
    d = _valid_spec_dict()
    spec1 = RunSpec.model_validate(copy.deepcopy(d))
    spec2 = RunSpec.model_validate(copy.deepcopy(d))
    assert spec1.to_canonical_yaml() == spec2.to_canonical_yaml()


def test_canonical_yaml_stable_across_repeated_calls() -> None:
    spec = RunSpec.model_validate(_valid_spec_dict())
    assert spec.to_canonical_yaml() == spec.to_canonical_yaml()


def test_canonical_yaml_independent_of_source_key_order() -> None:
    ordered_a = (
        "model:\n"
        "  family: local_level\n"
        "  options: {}\n"
        "data:\n"
        "  file: data.csv\n"
        "  date_column: date\n"
        "  mapping: {y: obs}\n"
    )
    ordered_b = (
        "data:\n"
        "  mapping: {y: obs}\n"
        "  date_column: date\n"
        "  file: data.csv\n"
        "model:\n"
        "  options: {}\n"
        "  family: local_level\n"
    )
    spec_a = RunSpec.model_validate(yaml.safe_load(ordered_a))
    spec_b = RunSpec.model_validate(yaml.safe_load(ordered_b))
    assert spec_a.to_canonical_yaml() == spec_b.to_canonical_yaml()


def test_canonical_yaml_differs_when_meaning_differs() -> None:
    d1 = _valid_spec_dict()
    d2 = _valid_spec_dict()
    d2["sampler"] = {"seed": 1}
    spec1 = RunSpec.model_validate(d1)
    spec2 = RunSpec.model_validate(d2)
    assert spec1.to_canonical_yaml() != spec2.to_canonical_yaml()


def test_canonical_yaml_round_trips_through_yaml_parser() -> None:
    spec = RunSpec.model_validate(_valid_spec_dict())
    canonical = spec.to_canonical_yaml()
    reparsed = yaml.safe_load(canonical)
    assert reparsed["model"]["family"] == "local_level"
    assert reparsed["data"]["mapping"] == {"y": "obs"}


# ---------------------------------------------------------------------------
# lw_sv options: sv_shocks combinations (S3)
# ---------------------------------------------------------------------------


def _lw_sv_spec_dict(sv_shocks: list) -> dict:
    return {
        "model": {"family": "lw_sv", "options": {"sv_shocks": sv_shocks}},
        "data": {
            "file": "data.csv",
            "date_column": "date",
            "mapping": {"y": "y", "pi": "pi", "r": "r"},
        },
    }


def test_lw_sv_sv_shocks_accepts_empty_and_both() -> None:
    assert RunSpec.model_validate(_lw_sv_spec_dict([])).model.options.sv_shocks == []
    assert RunSpec.model_validate(_lw_sv_spec_dict(["is", "pc"])).model.options.sv_shocks == ["is", "pc"]


def test_lw_sv_sv_shocks_normalized_to_canonical_order() -> None:
    """Run identity must not depend on how a spec ordered the list: [pc, is]
    canonicalizes to [is, pc], so the canonical YAML (hashed into run
    identity) is order-independent."""
    a = RunSpec.model_validate(_lw_sv_spec_dict(["pc", "is"]))
    b = RunSpec.model_validate(_lw_sv_spec_dict(["is", "pc"]))
    assert a.model.options.sv_shocks == ["is", "pc"]
    assert a.to_canonical_yaml() == b.to_canonical_yaml()


def test_lw_sv_sv_shocks_rejects_single_shock_and_duplicates() -> None:
    """[is, pc] is the ONE validated non-empty combination in v1 (spec
    §1.5/§2.3); single-shock SV would render but is unvalidated, so it must
    be rejected loudly, naming the field."""
    with pytest.raises(pydantic.ValidationError, match="sv_shocks"):
        RunSpec.model_validate(_lw_sv_spec_dict(["is"]))
    with pytest.raises(pydantic.ValidationError, match="sv_shocks"):
        RunSpec.model_validate(_lw_sv_spec_dict(["pc"]))
    with pytest.raises(pydantic.ValidationError, match="duplicate"):
        RunSpec.model_validate(_lw_sv_spec_dict(["is", "is"]))
    with pytest.raises(pydantic.ValidationError):
        RunSpec.model_validate(_lw_sv_spec_dict(["is", "pc", "extra"]))


# ---------------------------------------------------------------------------
# lw_sv outputs schema (S4, lw-sv-spec.md §2.3/§3; specs/schema/lw_sv.py
# LwSvOutputs) -- typed per-family validation via FamilyEntry.outputs_model,
# the same manually-dispatched pattern model.options already uses.
# ---------------------------------------------------------------------------


def test_lw_sv_outputs_defaults_match_spec_example() -> None:
    spec = RunSpec.model_validate(_lw_sv_spec_dict(["is", "pc"]))
    out = spec.outputs
    assert out.horizon == 12
    assert out.irf_horizon == 20
    assert out.irf_vol_reference == "end_of_sample"
    assert out.smoother_draws == "all"
    assert out.forecast_r_rule == "neutral"


def test_lw_sv_outputs_accepts_thin_spec() -> None:
    d = _lw_sv_spec_dict(["is", "pc"])
    d["outputs"] = {"smoother_draws": {"thin": 10}}
    spec = RunSpec.model_validate(d)
    assert spec.outputs.smoother_draws.thin == 10


def test_lw_sv_outputs_thin_must_be_positive() -> None:
    d = _lw_sv_spec_dict(["is", "pc"])
    d["outputs"] = {"smoother_draws": {"thin": 0}}
    with pytest.raises(pydantic.ValidationError, match="thin"):
        RunSpec.model_validate(d)


def test_lw_sv_outputs_accepts_last_value_forecast_rule() -> None:
    d = _lw_sv_spec_dict(["is", "pc"])
    d["outputs"] = {"forecast_r_rule": "last_value"}
    spec = RunSpec.model_validate(d)
    assert spec.outputs.forecast_r_rule == "last_value"


def test_lw_sv_outputs_rejects_user_path_as_not_implemented() -> None:
    """spec §3.3 exposes user_path in the schema vocabulary but v1 only
    implements neutral/last_value -- a spec asking for user_path must fail
    loudly, not silently fall back to neutral."""
    d = _lw_sv_spec_dict(["is", "pc"])
    d["outputs"] = {"forecast_r_rule": "user_path"}
    with pytest.raises(pydantic.ValidationError, match="user_path"):
        RunSpec.model_validate(d)


def test_lw_sv_outputs_extra_forbid_rejects_unknown_key() -> None:
    d = _lw_sv_spec_dict(["is", "pc"])
    d["outputs"] = {"not_a_real_key": 1}
    with pytest.raises(pydantic.ValidationError, match="not_a_real_key"):
        RunSpec.model_validate(d)


def test_lw_sv_outputs_non_dict_rejected() -> None:
    d = _lw_sv_spec_dict(["is", "pc"])
    d["outputs"] = "all"
    with pytest.raises(pydantic.ValidationError, match="outputs must be a mapping"):
        RunSpec.model_validate(d)


def test_local_level_outputs_stays_free_form_dict() -> None:
    """local_level has no registered outputs_model -- outputs stays the
    free-form dict RunSpec declares, unaffected by lw_sv's typed schema."""
    d = _valid_spec_dict()
    d["outputs"] = {"anything": "goes"}
    spec = RunSpec.model_validate(d)
    assert spec.outputs == {"anything": "goes"}
