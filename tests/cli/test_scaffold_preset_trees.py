"""Pin each preset's generated file tree without a web app.

`tests/fixtures/preset_trees/<preset>.json` holds the SHA-256 of each file `cayu new --preset
PRESET --without web-app` writes: no `web/`, Node or `/app` routes. The service preset has no
web app, so its tree is its default project. The b2b app is recorded separately in
`tests/fixtures/b2b_tree.json` (`test_scaffold_app_web.py`).

To record an intentional change to these projects, run

    uv run --no-sync python tests/cli/test_scaffold_preset_trees.py --write
"""

from __future__ import annotations

import hashlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import pytest

from cayu.cli import scaffold
from cayu.cli.scaffold import project_files

REPOSITORY = Path(__file__).resolve().parents[2]
TREES = REPOSITORY / "tests/fixtures/preset_trees"
# The project each tree is recorded for, made by a Cayu release of this version, so that
# neither a release nor the checkout the tests run from changes it.
PROJECT = "acme-desk"
VERSION = "0.0.0"
PRESETS = ("agent", "service", "coding", "long-running")


def tree(preset: str) -> dict[str, str]:
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(scaffold, "_cayu_source_checkout", lambda: None))
        stack.enter_context(mock.patch.object(scaffold, "_installed_cayu_version", lambda: VERSION))
        files = project_files(PROJECT, preset=preset, without_capabilities=("web-app",))
    return {path: hashlib.sha256(text.encode("utf-8")).hexdigest() for path, text in files.items()}


@pytest.mark.parametrize("preset", PRESETS)
def test_a_preset_without_the_web_app_is_the_recorded_tree(preset: str) -> None:
    recorded = json.loads((TREES / f"{preset}.json").read_text(encoding="utf-8"))
    files = tree(preset)

    assert recorded["project"] == PROJECT and recorded["cayu_version"] == VERSION
    differences = sorted(
        [f"new: {path}" for path in files.keys() - recorded["files"].keys()]
        + [f"gone: {path}" for path in recorded["files"].keys() - files.keys()]
        + [
            f"changed: {path}"
            for path in files.keys() & recorded["files"].keys()
            if files[path] != recorded["files"][path]
        ]
    )
    assert differences == [], (
        f"A {preset} project without the web app is no longer the recorded tree. If the "
        "change is meant, record it with "
        "`uv run --no-sync python tests/cli/test_scaffold_preset_trees.py --write`."
    )
    assert not any(path.startswith(("web/", "integrations/http/")) for path in files)


def _write() -> None:
    TREES.mkdir(parents=True, exist_ok=True)
    for preset in PRESETS:
        files = tree(preset)
        record = {"project": PROJECT, "cayu_version": VERSION, "files": files}
        (TREES / f"{preset}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        raise SystemExit(f"usage: {sys.argv[0]} --write")
    _write()
