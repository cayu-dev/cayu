from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_readme_preserves_the_product_overview_and_routes_to_the_cayu_map() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "## Why Cayu" in readme
    assert "## What Cayu provides" in readme
    assert "## Quickstart" in readme
    assert "This compact example shows the core API" in readme
    assert "src/cayu/guides/authoring.md#cayu-map" in readme
    assert "https://github.com/cayu-dev/cayu/blob/main/examples/README.md" in readme
    assert "one safe example tool" not in readme

    repository_prefix = "https://github.com/cayu-dev/cayu/blob/main/"
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", readme):
        if not target.startswith(repository_prefix):
            continue
        relative = target.removeprefix(repository_prefix).split("#", 1)[0]
        assert (ROOT / relative).exists(), f"missing README target: {target}"


def test_authoring_guide_is_the_canonical_cayu_map() -> None:
    guide = (ROOT / "src" / "cayu" / "guides" / "authoring.md").read_text(encoding="utf-8")

    assert "## Cayu Map" in guide
    assert "Use only the concepts your agent needs." in guide
    assert "Reviewed or retrievable knowledge" in guide
    assert "Usage limits or cost control" in guide
    assert "https://github.com/cayu-dev/cayu/blob/main/examples/README.md" in guide


def test_examples_index_routes_from_simple_to_advanced_examples() -> None:
    index_path = ROOT / "examples" / "README.md"
    index = index_path.read_text(encoding="utf-8")

    assert "## Start here" in index
    assert "## Tools and providers" in index
    assert "## Execution environments" in index
    assert "## Durable orchestration" in index
    assert "## Operations and advanced strategies" in index
    assert "ADVANCED_RUNTIME_EXAMPLES.md" in index

    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", index):
        if "://" in target or target.startswith("#"):
            continue
        resolved = (index_path.parent / target.split("#", 1)[0]).resolve()
        assert resolved.exists(), f"missing examples index target: {target}"
