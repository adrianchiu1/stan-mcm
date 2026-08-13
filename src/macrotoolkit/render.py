"""Jinja rendering of Stan templates + a compile cache.

Render: `stan/templates/<family>.stan.j2` + a template context (family
options plus any spec-derived constants) -> a concrete .stan source string.
No runtime branching inside Stan -- everything family-specific is stamped
in at render time (lw-sv-spec.md §2.3). `local_level`'s template currently
takes an empty context (it has no options), but is still routed through
Jinja so the render/compile pipeline is exercised uniformly across
families.

Compile cache: rendered source is written once per
(stan_source_hash, cmdstan_version) under `.mtk_cache/stan/<key>/`, and
`CmdStanModel` compiles it there. Re-rendering byte-identical source (same
family, same template, same context, same CmdStan) never triggers a second
compile -- `CmdStanModel` also skips recompilation if a matching, up-to-date
executable is already present in that directory.

CmdStan version pin: 2.36.0 (see DECISIONS.md and PREFLIGHT.md). Never call
`cmdstanpy.install_cmdstan()` without an explicit `version=` -- the default
"latest" lookup hits `api.github.com`, which this environment's egress
policy blocks.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import cmdstanpy
from cmdstanpy import CmdStanModel
from jinja2 import Environment, FileSystemLoader, StrictUndefined

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "stan" / "templates"
CACHE_ROOT = REPO_ROOT / ".mtk_cache" / "stan"

PINNED_CMDSTAN_VERSION = "2.36.0"


def ensure_cmdstan_path() -> None:
    """Point cmdstanpy at a CmdStan install without ever falling back to the
    "resolve latest release" path (blocked by egress policy here). Honors
    the `CMDSTAN` env var if set; otherwise defaults to the pinned-version
    install location documented in DECISIONS.md."""
    cmdstan_env = os.environ.get("CMDSTAN")
    if cmdstan_env:
        cmdstanpy.set_cmdstan_path(cmdstan_env)
        return
    default_path = Path.home() / ".cmdstan" / f"cmdstan-{PINNED_CMDSTAN_VERSION}"
    if not default_path.is_dir():
        raise RuntimeError(
            f"CmdStan not found. Set the CMDSTAN environment variable to a "
            f"CmdStan install directory (pinned version "
            f"{PINNED_CMDSTAN_VERSION} per DECISIONS.md), or install one at "
            f"{default_path}. Do not call cmdstanpy.install_cmdstan() "
            f"without an explicit version= -- the default 'latest' lookup "
            f"hits api.github.com, which is blocked in this environment."
        )
    cmdstanpy.set_cmdstan_path(str(default_path))


def get_cmdstan_version() -> str:
    """CmdStan version string, e.g. '2.36.0', derived from the active
    CmdStan install directory name."""
    ensure_cmdstan_path()
    name = Path(cmdstanpy.cmdstan_path()).name
    prefix = "cmdstan-"
    if not name.startswith(prefix):
        raise RuntimeError(
            f"Could not determine CmdStan version from install directory "
            f"name {name!r}; expected it to start with {prefix!r}."
        )
    return name[len(prefix):]


def _jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        undefined=StrictUndefined,  # a missing context var is a hard error, not blank output
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_stan_source(template_name: str, context: dict) -> str:
    """Render `stan/templates/<template_name>` with `context`, returning the
    Stan source as a string."""
    env = _jinja_env()
    try:
        template = env.get_template(template_name)
    except Exception as exc:
        raise FileNotFoundError(
            f"Stan template {template_name!r} not found under "
            f"{TEMPLATES_DIR}. Check specs/schema FAMILY_REGISTRY's "
            f"'template' entry for this family. ({exc})"
        ) from exc
    try:
        return template.render(**context)
    except Exception as exc:
        raise ValueError(
            f"Failed to render Stan template {template_name!r} with context "
            f"keys {sorted(context)}: {exc}"
        ) from exc


def stan_source_hash(source: str) -> str:
    """SHA-256 hex digest of rendered Stan source text."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def compile_model(source: str, cmdstan_version: str | None = None) -> tuple[CmdStanModel, str]:
    """Compile rendered Stan `source`, using the on-disk cache keyed on
    (source hash, cmdstan version). Returns (CmdStanModel, source_hash)."""
    ensure_cmdstan_path()
    if cmdstan_version is None:
        cmdstan_version = get_cmdstan_version()
    src_hash = stan_source_hash(source)
    key = f"{src_hash[:16]}_{cmdstan_version}"
    cache_dir = CACHE_ROOT / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    stan_file = cache_dir / "model.stan"

    if stan_file.exists():
        existing = stan_file.read_text()
        if existing != source:
            # Should be unreachable -- the cache key is derived from the
            # source hash -- unless the cache directory was hand-edited.
            raise RuntimeError(
                f"Compile cache corruption at {stan_file}: cached source "
                f"does not match the hash-derived source. Remove "
                f"{cache_dir} manually and retry."
            )
    else:
        stan_file.write_text(source)

    # CmdStanModel itself skips recompilation if stan_file's compiled
    # executable is already present and newer than the source -- combined
    # with the source-hash-keyed cache directory, this means identical
    # rendered source is never recompiled.
    model = CmdStanModel(stan_file=str(stan_file))
    return model, src_hash
