"""The web app `cayu new` writes into `web/`: the generic app and the layers on top of it.

`cayu new --preset b2b` gets the generic app (`templates/app_web/`) with the organizations
layer (`templates/app_web_organizations/`). `tests/fixtures/b2b_tree.json` holds the SHA-256 of
each file of a b2b project, and the generator keeps writing exactly that tree. When a change to
what a b2b project gets is meant, record the new tree with

    uv run --no-sync python tests/cli/test_scaffold_app_web.py --write

That the generic app builds on its own, and the b2b app's committed build is the build of its
source, is `scripts/build_app_web.py --check`'s to verify: it needs Node.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import pytest

from cayu.cli import scaffold
from cayu.cli.scaffold import project_files
from cayu.cli.scaffold_app_web import APP, OPERATORS, ORGANIZATIONS, layer_sources, web_app_files

REPOSITORY = Path(__file__).resolve().parents[2]
TEMPLATES = REPOSITORY / "src/cayu/cli/templates"
TREE = REPOSITORY / "tests/fixtures/b2b_tree.json"
# The project the tree is recorded for, made by a Cayu release of this version, so that
# neither a release nor the checkout the tests run from changes it.
PROJECT = "acme-desk"
VERSION = "0.0.0"


def _b2b_tree() -> dict[str, str]:
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(scaffold, "_cayu_source_checkout", lambda: None))
        stack.enter_context(mock.patch.object(scaffold, "_installed_cayu_version", lambda: VERSION))
        files = project_files(PROJECT, preset="b2b")
    return {path: hashlib.sha256(text.encode("utf-8")).hexdigest() for path, text in files.items()}


def test_a_b2b_project_is_the_recorded_tree() -> None:
    recorded = json.loads(TREE.read_text(encoding="utf-8"))
    tree = _b2b_tree()

    assert recorded["project"] == PROJECT and recorded["cayu_version"] == VERSION
    differences = sorted(
        [f"new: {path}" for path in tree.keys() - recorded["files"].keys()]
        + [f"gone: {path}" for path in recorded["files"].keys() - tree.keys()]
        + [
            f"changed: {path}"
            for path in tree.keys() & recorded["files"].keys()
            if tree[path] != recorded["files"][path]
        ]
    )
    assert differences == [], (
        "A b2b project is no longer the recorded tree. If the change is meant, record it with "
        "`uv run --no-sync python tests/cli/test_scaffold_app_web.py --write`."
    )


def test_the_generic_app_names_no_organizations_members_or_departments() -> None:
    app = TEMPLATES / APP
    found = []
    for path in sorted(app.rglob("*")):
        if not path.is_file() or "node_modules" in path.parts:
            continue
        # The role `GET /app/whoami` reports (`admin`, `developer` or `member`) is the generic
        # routes' own, not a membership of anything.
        text = path.read_text(encoding="utf-8").replace('"member"', "")
        for match in re.finditer(r"organi[sz]ation|department|\bmember|b2b", text, re.IGNORECASE):
            found.append(f"{path.relative_to(app)}: {match.group(0)}")
    assert found == []
    # It lists only the generated areas, and adds nothing to the shell.
    registry = (app / "src/agent/index.tsx").read_text(encoding="utf-8")
    assert "export const areas: AppArea[] = [overviewArea(" in registry
    assert "settingsArea]" in registry
    assert "export const shell: ShellParts = {};" in registry
    # It ships no build of its own: each composition's is its last layer's.
    assert not (app / "static").exists()


def test_the_operator_app_is_the_generic_app_with_the_operators_layer() -> None:
    app = layer_sources(TEMPLATES)
    composed = layer_sources(TEMPLATES, (OPERATORS,))
    layer = TEMPLATES / OPERATORS

    from_layer = {
        path for path, source in composed.items() if Path(str(source)).is_relative_to(layer)
    }
    # The layer replaces only the area registry, and adds Task runs, the smoke test and the build.
    assert from_layer & app.keys() == {"src/agent/index.tsx"}
    assert {
        "src/agent/task-runs.tsx",
        "e2e/package.json",
        "e2e/package-lock.json",
        "e2e/playwright.config.ts",
        "e2e/smoke.spec.ts",
        "static/index.html",
    } <= from_layer
    registry = (layer / "src/agent/index.tsx").read_text(encoding="utf-8")
    assert "export const shell: ShellParts = {};" in registry
    assert "taskRunsArea," in registry
    # Its people are the Agent's operators: it names no organizations, members or departments.
    for path in sorted(from_layer - {"e2e/package-lock.json"}):
        if not path.startswith("static/"):
            text = (layer / path).read_text(encoding="utf-8")
            assert not re.search(r"organi[sz]ation|department|\bmember|b2b", text, re.I), path
    # Every preset with the `web-app` capability gets exactly this app.
    files = project_files(PROJECT)
    assert {f"web/{path}" for path in composed} == {
        path for path in files if path.startswith("web/")
    }
    assert '"name": "web",' in files["web/package.json"]


def test_the_b2b_app_is_the_generic_app_with_the_organizations_layer() -> None:
    app = layer_sources(TEMPLATES)
    composed = layer_sources(TEMPLATES, (ORGANIZATIONS,))
    layer = TEMPLATES / ORGANIZATIONS

    from_layer = {
        path for path, source in composed.items() if Path(str(source)).is_relative_to(layer)
    }
    # The layer replaces only the area registry; everything else of the app is the generic one.
    assert from_layer & app.keys() == {"src/agent/index.tsx"}
    assert {path: composed[path] for path in composed.keys() - from_layer} == {
        path: source for path, source in app.items() if path != "src/agent/index.tsx"
    }
    assert {
        "src/agent/members.tsx",
        "src/agent/organization.tsx",
        "src/agent/sign-in.tsx",
        "static/index.html",
    } <= from_layer
    files = project_files(PROJECT, preset="b2b")
    assert {f"web/{path}" for path in composed} == {
        path for path in files if path.startswith("web/")
    }
    assert '"name": "b2b-web",' in files["web/package.json"]
    assert files["web/package-lock.json"].count('"name": "b2b-web",') == 2


def test_a_layer_adds_files_and_replaces_only_the_agents_own(tmp_path: Path) -> None:
    _write(
        tmp_path / APP,
        {
            "src/app/area.ts": "/** GENERATED. The area pattern. */\n",
            "src/agent/index.tsx": "/** THE AGENT'S OWN. The areas. */\n",
            "static/assets/index-app.js": "app\n",
            "static/index.html": "<title>Agent</title>\n",
        },
    )
    _write(
        tmp_path / "areas",
        {
            "src/agent/index.tsx": "/** THE AGENT'S OWN. More areas. */\n",
            "src/agent/more.tsx": "/** THE AGENT'S OWN. Another area. */\n",
            "static/assets/index-areas.js": "areas\n",
            "static/index.html": "<title>Agent</title>\n",
        },
    )
    _write(tmp_path / "shell", {"src/app/area.ts": "/** GENERATED. Another pattern. */\n"})

    composed = layer_sources(tmp_path, ("areas",))
    assert {path: source.read_text(encoding="utf-8") for path, source in composed.items()} == {
        "src/app/area.ts": "/** GENERATED. The area pattern. */\n",
        "src/agent/index.tsx": "/** THE AGENT'S OWN. More areas. */\n",
        "src/agent/more.tsx": "/** THE AGENT'S OWN. Another area. */\n",
        # The layer's build replaces the app's whole: its files have other names.
        "static/assets/index-areas.js": "areas\n",
        "static/index.html": "<title>Agent</title>\n",
    }
    with pytest.raises(ValueError, match="replaces src/app/area.ts"):
        layer_sources(tmp_path, ("shell",))


def test_the_pages_are_titled_with_the_project_and_the_package_named() -> None:
    files = web_app_files("Acme & Co", layers=(ORGANIZATIONS,), package_name="acme-web")

    for page in ("web/index.html", "web/static/index.html"):
        assert "<title>Acme &amp; Co</title>" in files[page]
        assert "<title>Agent</title>" not in files[page]
    assert json.loads(files["web/package.json"])["name"] == "acme-web"
    lock = json.loads(files["web/package-lock.json"])
    assert lock["name"] == lock["packages"][""]["name"] == "acme-web"


def _write(root: Path, files: dict[str, str]) -> None:
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        raise SystemExit(f"usage: {sys.argv[0]} --write")
    record = {"project": PROJECT, "cayu_version": VERSION, "files": _b2b_tree()}
    TREE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
