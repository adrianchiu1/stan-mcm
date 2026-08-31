"""`macrotoolkit.render` tests: Jinja rendering, source hashing, and the
on-disk compile cache.

We don't test CmdStan's own recompilation-skipping logic (whether a stale
executable is rebuilt) -- only macrotoolkit's cache-key/directory logic:
same source -> same cache dir, second compile call doesn't rewrite the
cached .stan file or trigger a second `CmdStanModel` build (checked via
mtime of the compiled executable, which CmdStan only touches on an actual
build).
"""
from __future__ import annotations

import time

import pytest

from macrotoolkit.render import (
    CACHE_ROOT,
    compile_model,
    get_cmdstan_version,
    render_stan_source,
    stan_source_hash,
)


def test_render_local_level_template_no_error() -> None:
    source = render_stan_source("local_level.stan.j2", {})
    assert "parameters" in source
    assert "sigma_obs" in source
    assert "sigma_level" in source


def test_render_unknown_template_errors_clearly() -> None:
    with pytest.raises(FileNotFoundError, match="not_a_real_template.stan.j2"):
        render_stan_source("not_a_real_template.stan.j2", {})


def test_source_hash_same_text_same_hash() -> None:
    source = render_stan_source("local_level.stan.j2", {})
    h1 = stan_source_hash(source)
    h2 = stan_source_hash(source)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex digest


def test_source_hash_differs_for_different_text() -> None:
    h1 = stan_source_hash("model {}\n")
    h2 = stan_source_hash("model {}\n// trailing comment\n")
    assert h1 != h2


def test_render_is_deterministic_across_calls() -> None:
    a = render_stan_source("local_level.stan.j2", {})
    b = render_stan_source("local_level.stan.j2", {})
    assert a == b


def test_no_sv_render_is_byte_stable() -> None:
    """The rendered no-SV lw_sv program is pinned byte-for-byte against
    tests/fixtures/render/lw_sv_no_sv.stan (decision 2026-08-31,
    DECISIONS.md): S3's SV additions must sit entirely behind `sv_shocks`
    conditionals, so the empty-`sv_shocks` render -- and with it every
    no-SV run hash -- cannot drift by accident. If this fails, either a
    template/functions edit leaked into the no-SV render (fix the leak) or
    the change is a DELIBERATE, reviewed no-SV change (regenerate the
    fixture and record the decision + hash consequences in DECISIONS.md).
    """
    from pathlib import Path

    from specs.schema.base import RunSpec
    from macrotoolkit.run import build_render_context

    spec = RunSpec.model_validate(
        {
            "model": {"family": "lw_sv", "options": {}},
            "data": {
                "file": "unused.csv",
                "date_column": "date",
                "mapping": {"y": "y", "pi": "pi", "r": "r"},
            },
        }
    )
    rendered = render_stan_source("lw_sv.stan.j2", build_render_context(spec))
    pinned = (Path(__file__).parent / "fixtures" / "render" / "lw_sv_no_sv.stan").read_text()
    assert rendered == pinned


def test_sv_render_has_sv_blocks_and_compiles() -> None:
    """The sv_shocks: [is, pc] render carries the full SV machinery -- the
    sv function include, the mu_h0 data entries, the non-centered
    parameters, the h transformed parameters, the SV priors, and the
    time-varying KF call -- and none of the no-SV-only pieces; and it
    compiles (cached after the first run)."""
    from specs.schema.base import RunSpec
    from macrotoolkit.run import build_render_context

    spec = RunSpec.model_validate(
        {
            "model": {"family": "lw_sv", "options": {"sv_shocks": ["is", "pc"]}},
            "data": {
                "file": "unused.csv",
                "date_column": "date",
                "mapping": {"y": "y", "pi": "pi", "r": "r"},
            },
        }
    )
    source = render_stan_source("lw_sv.stan.j2", build_render_context(spec))

    for needle in (
        "vector sv_rw_noncentered(",
        "real mu_h0_is;",
        "real<lower=0> sigma_h_is;",
        "vector[T] nu_pc;",
        "transformed parameters {",
        "sv_diag_variance_path(h_is, h_pc)",
        "nu_is ~ std_normal();",
    ):
        assert needle in source, f"SV render is missing {needle!r}"
    # The constant-scale machinery must be fully replaced, not coexist
    # (the lw_R DEFINITION still rides in with the shared include; what
    # must be gone is the constant parameters, their priors, and the call).
    for absent in (
        "real<lower=0> sigma_is;",
        "real<lower=0> sigma_pc;",
        "sigma_is ~",
        "sigma_pc ~",
        "lw_R(sigma_is",
    ):
        assert absent not in source, f"SV render still contains {absent!r}"

    model, _ = compile_model(source)
    assert model.exe_file is not None


def test_compile_model_local_level_succeeds() -> None:
    source = render_stan_source("local_level.stan.j2", {})
    model, src_hash = compile_model(source)
    assert src_hash == stan_source_hash(source)
    assert model.exe_file is not None
    import os

    assert os.path.isfile(model.exe_file)


def test_compile_cache_directory_keyed_on_source_hash_and_version() -> None:
    source = render_stan_source("local_level.stan.j2", {})
    version = get_cmdstan_version()
    src_hash = stan_source_hash(source)
    _, _ = compile_model(source, cmdstan_version=version)
    expected_key = f"{src_hash[:16]}_{version}"
    cache_dir = CACHE_ROOT / expected_key
    assert cache_dir.is_dir()
    assert (cache_dir / "model.stan").read_text() == source


def test_compile_model_second_call_does_not_recompile(tmp_path) -> None:
    """Second call with byte-identical source must not rebuild the
    executable -- verified via the compiled executable's mtime being
    unchanged (CmdStanModel only touches it on an actual build)."""
    source = render_stan_source("local_level.stan.j2", {})
    version = get_cmdstan_version()

    model1, _ = compile_model(source, cmdstan_version=version)
    exe_path = model1.exe_file
    assert exe_path is not None
    mtime_1 = _mtime(exe_path)

    # Ensure any timestamp change would be detectable.
    time.sleep(0.05)

    model2, _ = compile_model(source, cmdstan_version=version)
    mtime_2 = _mtime(model2.exe_file)

    assert model1.exe_file == model2.exe_file
    assert mtime_1 == mtime_2, (
        "second compile_model() call rewrote/rebuilt the cached executable "
        "for byte-identical source; expected the cache to be reused"
    )


def test_compile_cache_reused_for_semantically_identical_rerender() -> None:
    """Rendering the same template+context twice and compiling both
    produces the exact same cache directory (not two different ones)."""
    source_a = render_stan_source("local_level.stan.j2", {})
    source_b = render_stan_source("local_level.stan.j2", {})
    version = get_cmdstan_version()
    model_a, hash_a = compile_model(source_a, cmdstan_version=version)
    model_b, hash_b = compile_model(source_b, cmdstan_version=version)
    assert hash_a == hash_b
    assert model_a.exe_file == model_b.exe_file


def _mtime(path: str) -> float:
    import os

    return os.stat(path).st_mtime
