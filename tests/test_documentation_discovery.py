from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_readme_preserves_the_product_overview_and_routes_to_the_cayu_map() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "## Why Cayu" in readme
    assert "## What Cayu provides" in readme
    assert "## Quickstart" in readme
    assert "This compact example shows the core API" in readme
    assert "src/cayu/guides/authoring.md#cayu-map" in readme
    assert "cayu guide durable-operations" in readme
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


def test_readme_routes_provider_searches_to_the_canonical_compatibility_guide() -> None:
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())

    assert "Compatible Chat Completions" in readme
    assert "cayu guide providers#openrouter" in readme
    assert "src/cayu/guides/providers.md#openrouter" in readme
    for service in ("OpenRouter", "Fireworks", "Baseten", "OpenCode Go"):
        assert service in readme


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


def test_readme_and_overview_docs_open_with_the_cayu_identity() -> None:
    from cayu.cli._identity import CAYU_CAPABILITIES, CAYU_IDENTITY, CAYU_IS_NOT

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    first_screen = " ".join(readme.split("\n## ", 1)[0].split())
    assert CAYU_IDENTITY in first_screen
    for capability, parts in CAYU_CAPABILITIES:
        assert f"| {capability} | {parts} |" in first_screen
    assert CAYU_IS_NOT in first_screen

    for relative in ("docs/architecture.md", "docs/public-concepts.md"):
        overview = " ".join((ROOT / relative).read_text(encoding="utf-8").split())
        assert CAYU_IDENTITY in overview, relative


# Exact `cayu==<version>` pins go stale and get copied into new projects. Prose should
# say `pip install cayu` or `uvx cayu ...`. Generated projects pin their dependency in
# pyproject.toml, which is not prose and is not scanned here.
_CAYU_VERSION_PIN = re.compile(r"\bcayu(?:\[[^\]]*\])?\s*==\s*\d")
_INTENTIONAL_CAYU_PINS = {
    "docs/release-notes.md": "upgrade notes name the exact release they describe",
    "docs/cayu-cloud.md": "shows how `cayu cloud init` rewrites an existing pinned requirement",
}


def test_prose_does_not_pin_cayu_versions() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    offenders = []
    for relative in filter(None, tracked.split("\0")):
        if relative.startswith("tests/") or relative in _INTENTIONAL_CAYU_PINS:
            continue
        for number, line in enumerate(
            (ROOT / relative).read_text(encoding="utf-8").splitlines(), 1
        ):
            if _CAYU_VERSION_PIN.search(line):
                offenders.append(f"{relative}:{number}: {line.strip()}")
    assert not offenders, "use `pip install cayu` or `uvx cayu` instead:\n" + "\n".join(offenders)


def test_cayu_version_pin_pattern_matches_install_commands() -> None:
    for pinned in ("uvx --from cayu==0.9.1 cayu new", 'pip install "cayu[postgres]==0.11.3"'):
        assert _CAYU_VERSION_PIN.search(pinned), pinned
    for unpinned in ("pip install cayu", "uvx cayu new", "install `cayu==X.Y.Z`", "mycayu==1"):
        assert _CAYU_VERSION_PIN.search(unpinned) is None, unpinned


def test_guides_keep_their_declared_discovery_names_in_short_entry_points() -> None:
    """Guides own the names readers must see before opening the full guide.

    Add a cayu-guide-discovery JSON comment to any guide, naming its README row,
    authoring-map row, reference section and essential public API names. Keep the
    declaration with the guide so new names extend this check without test edits.
    """
    import json

    guides = ROOT / "src/cayu/guides"
    readme = (ROOT / "README.md").read_text()
    authoring = (guides / "authoring.md").read_text()
    references = (guides / "references.md").read_text()
    declared = []
    for path in sorted(guides.glob("*.md")):
        source = path.read_text()
        markers = re.findall(r"<!-- cayu-guide-discovery: (.*?) -->", source)
        assert source.count("<!-- cayu-guide-discovery:") == len(markers), path
        for marker in markers:
            contract = json.loads(marker)
            assert set(contract) == {"names", "readme_row", "authoring_row", "reference"}, path
            names = contract["names"]
            assert names and len(set(names)) == len(names), path
            entry_points = {}
            for label, text, row in (
                ("README", readme, contract["readme_row"]),
                ("authoring map", authoring, contract["authoring_row"]),
            ):
                rows = [line for line in text.splitlines() if line.startswith(f"| {row} |")]
                assert len(rows) == 1, (path, label, row)
                entry_points[label] = rows[0]
            heading = f"## {contract['reference']}\n"
            assert references.count(heading) == 1, (path, heading)
            entry_points["references"] = references.split(heading, 1)[1].split("\n## ", 1)[0]
            guide_body = re.sub(r"<!--.*?-->", "", source, flags=re.S)
            for name in names:
                assert re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name), (path, name)
                assert f"`{name}`" in guide_body, (path, name)
                for label, text in entry_points.items():
                    assert f"`{name}`" in text, f"{path.name}: {label} omits {name}"
            declared.append(path.stem)
    assert "external-agents" in declared
