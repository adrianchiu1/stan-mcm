"""S7 guard: adding the authored family must not change any EXISTING
family's rendered program or estimation identity. Every example spec's
rendered-source SHA-256 and canonical estimation YAML are pinned to a
fixture captured at S7 start (before any S7 change), so a shared-Stan
edit or a spec-schema change that leaks into an existing family fails
here, loudly."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from macrotoolkit.render import render_stan_source, stan_source_hash
from macrotoolkit.run import build_render_context
from specs.schema import get_family
from specs.schema.base import load_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "render" / "s7_existing_family_hashes.json"
SPECS = [
    "examples/toy/spec.yaml",
    "examples/us_lw_sv/spec.yaml",
    "examples/us_lw_sv/spec_sv.yaml",
    "examples/us_lw_sv/spec_full_vintage.yaml",
    "examples/us_lw_sv/spec_sv_full_vintage.yaml",
    "examples/us_ucsv/spec.yaml",
    "examples/us_ucsv/spec_full_vintage.yaml",
]


def current_hashes() -> dict[str, dict[str, str]]:
    out = {}
    for rel in SPECS:
        spec = load_spec(str(REPO_ROOT / rel))
        source = render_stan_source(get_family(spec.model.family).template, build_render_context(spec))
        out[rel] = {
            "stan_source_sha256": stan_source_hash(source),
            "estimation_yaml_sha256": hashlib.sha256(spec.to_estimation_yaml().encode("utf-8")).hexdigest(),
        }
    return out


@pytest.mark.parametrize("rel", SPECS)
def test_existing_family_render_and_estimation_identity_unchanged(rel: str) -> None:
    pinned = json.loads(FIXTURE.read_text())
    assert current_hashes()[rel] == pinned[rel], (
        f"{rel}: the rendered Stan source or the canonical estimation spec changed since S7 start -- "
        f"an existing family's run identity would move. S7 adds, it never edits, shared Stan text or family specs."
    )


if __name__ == "__main__":  # capture the fixture (run ONCE on the untouched baseline)
    FIXTURE.write_text(json.dumps(current_hashes(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {FIXTURE}")
