"""`cayu new --preset b2b` writes what the b2b reference project says it should.

The reference project (`manifest.json`'s `reference`) is a b2b project changed by hand until
it is what the generator should write. CI can't read it, so `tests/fixtures/b2b_reference/`
keeps the generated files that work covers, and the manifest says for each whether it is the
reference's file byte for byte (with the reference file's SHA-256) or why it differs.
`scripts/b2b_reference_fixtures.py --reference PATH` checks those claims against a checkout
of the reference and records new ones.

`web/static` isn't compared: it is the build of the `web/` source compared here, which
`scripts/build_b2b_web.py --check` and the project's own `npm run check:build` verify.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cayu.cli.scaffold import project_files

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "b2b_reference"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
SUFFIX = ".fixture"


def _in_scope(path: str) -> bool:
    if path.startswith(tuple(MANIFEST["excluded_prefixes"])):
        return False
    return path in MANIFEST["paths"] or path.startswith(tuple(MANIFEST["prefixes"]))


@pytest.fixture(scope="module")
def generated() -> dict[str, str]:
    files = project_files(MANIFEST["project"], preset="b2b")
    return {path: text for path, text in files.items() if _in_scope(path)}


def _fixtures() -> dict[str, bytes]:
    root = FIXTURES / "files"
    return {
        path.relative_to(root).as_posix().removesuffix(SUFFIX): path.read_bytes()
        for path in root.rglob(f"*{SUFFIX}")
    }


def test_the_generator_writes_the_recorded_files(generated: dict[str, str]) -> None:
    fixtures = _fixtures()

    assert set(generated) == set(fixtures) == set(MANIFEST["files"]), (
        "The generated files and the recorded ones differ; run "
        "`python scripts/b2b_reference_fixtures.py --reference PATH --write`."
    )
    changed = sorted(
        path for path, text in generated.items() if text.encode("utf-8") != fixtures[path]
    )
    assert changed == [], (
        "These generated files no longer match the reference fixtures. If the change is "
        "meant, say why in each file's `deviation` in tests/fixtures/b2b_reference/"
        "manifest.json and record it with `python scripts/b2b_reference_fixtures.py "
        "--reference PATH --write`."
    )


def test_each_file_is_the_reference_s_or_says_why_it_differs() -> None:
    fixtures = _fixtures()
    for path, entry in MANIFEST["files"].items():
        digest = hashlib.sha256(fixtures[path]).hexdigest()
        if entry.get("same"):
            assert entry["reference_sha256"] == digest, path
            assert "deviation" not in entry, path
        else:
            assert entry["deviation"].strip(), path
            assert entry["reference_sha256"] != digest, f"{path} is the reference's; mark it same"


def test_reference_files_the_generator_leaves_out_say_why(generated: dict[str, str]) -> None:
    left_out = MANIFEST["not_generated"]

    assert all(reason.strip() for reason in left_out.values())
    assert set(left_out).isdisjoint(generated)
    assert all(_in_scope(path) for path in left_out)


def test_no_fixture_names_a_private_package() -> None:
    text = "\n".join(data.decode("utf-8") for data in _fixtures().values())
    for private in ("@cayu-tech", "cayu-design", "@cayu/design", "npm.pkg.github.com"):
        assert private not in text
