"""FamilyEntry contract completion (S5-decisions item 4): every capability
a family provides is declared on the registry as a lazily-resolved dotted
path, and there is no if/elif family dispatch left in run.py/cli.py.
"""
from __future__ import annotations

import pytest

from specs.schema import FAMILY_REGISTRY, FamilyEntry, get_family
from specs.schema.local_level import LocalLevelOptions


def test_every_registered_capability_path_resolves() -> None:
    """A registry typo must fail here, at test time -- not at first use in
    production. Resolves EVERY declared dotted path for EVERY family."""
    for name, entry in FAMILY_REGISTRY.items():
        for capability in (
            "build_stan_data",
            "build_render_context",
            "state_meta",
            "results_loader",
            "report_writer",
        ):
            if getattr(entry, capability) is not None:
                obj = entry.resolve(capability)
                assert obj is not None, (name, capability)


def test_lw_sv_declares_the_full_contract() -> None:
    entry = get_family("lw_sv")
    assert callable(entry.resolve("build_stan_data"))
    assert callable(entry.resolve("build_render_context"))
    assert callable(entry.resolve("results_loader"))
    assert callable(entry.resolve("report_writer"))
    from macrotoolkit.families.lw_sv import LW_STATE_META

    assert entry.resolve("state_meta") is LW_STATE_META


def test_local_level_has_no_output_capabilities() -> None:
    entry = get_family("local_level")
    assert entry.resolve("results_loader") is None
    assert entry.resolve("report_writer") is None
    assert entry.resolve("state_meta") is None
    assert callable(entry.resolve("build_stan_data"))


def test_resolve_rejects_unknown_capability_and_bad_paths() -> None:
    entry = get_family("lw_sv")
    with pytest.raises(ValueError, match="Unknown family capability"):
        entry.resolve("nonsense")
    bad = FamilyEntry(
        options_model=LocalLevelOptions,
        template="local_level.stan.j2",
        required_mapping=("y",),
        build_stan_data="macrotoolkit.families.local_level:no_such_attr",
        build_render_context="not-a-dotted-path",
    )
    with pytest.raises(AttributeError, match="no_such_attr"):
        bad.resolve("build_stan_data")
    with pytest.raises(ValueError, match="dotted path"):
        bad.resolve("build_render_context")


def test_run_module_dispatchers_delegate_to_the_registry() -> None:
    """run.build_stan_data/build_render_context are thin registry
    dispatchers -- same behavior as the family modules called directly."""
    import pandas as pd

    from macrotoolkit.families.local_level import build_stan_data as ll_build
    from macrotoolkit.run import build_stan_data

    df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    assert build_stan_data("local_level", df) == ll_build(df)


def test_no_family_ifelif_dispatch_left_in_run_py() -> None:
    """The mechanical pin on item 4's 'no if/elif family dispatch' claim:
    run.py must not branch on family-name string literals anymore."""
    from pathlib import Path

    import macrotoolkit.run as run_module

    source = Path(run_module.__file__).read_text()
    assert 'family == "lw_sv"' not in source
    assert 'family == "local_level"' not in source
    assert 'family_name == "lw_sv"' not in source
    assert 'family_name == "local_level"' not in source
