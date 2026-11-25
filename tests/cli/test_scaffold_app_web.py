"""The web app `cayu new` writes into `web/`: the generic app and the layers on top of it.

`cayu new --preset b2b` gets the generic app (`templates/app_web/`) with the organizations
layer (`templates/app_web_organizations/`), and an Agent with knowledge gets the knowledge layer
(`templates/app_web_knowledge/`) on top of its app. Each composition's committed build is in
`templates/app_web_builds/`. `tests/fixtures/b2b_tree.json` holds the SHA-256 of
each file of a b2b project, and the generator keeps writing exactly that tree. When a change to
what a b2b project gets is meant, record the new tree with

    uv run --no-sync python tests/cli/test_scaffold_app_web.py --write

That the generic app builds on its own, and each committed build is the build of its
composition's source, is `scripts/build_app_web.py --check`'s to verify: it needs Node.
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
from cayu.cli.scaffold_app_web import (
    APP,
    BUILDS,
    CONVERSATIONS,
    KNOWLEDGE,
    OPERATORS,
    ORGANIZATIONS,
    app_build,
    app_sources,
    build_name,
    layer_sources,
    web_app_files,
    with_knowledge_area,
)

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
    # It ships no build of its own, and neither does a layer: each composition's is in
    # app_web_builds/.
    for name in (APP, OPERATORS, CONVERSATIONS, ORGANIZATIONS, KNOWLEDGE):
        assert not (TEMPLATES / name / "static").exists(), name


def test_the_web_app_and_the_dashboard_pin_one_vite_plus() -> None:
    # Vite+ is a release candidate, pinned exactly. `vite` is its core under Vite's name, the
    # same version, also for any dependency that asks for `vite`; a project that resolved
    # another `vite` would build with a different bundler than the one it lints and tests with.
    for root in (TEMPLATES / APP, REPOSITORY / "dashboard"):
        manifest = json.loads((root / "package.json").read_text(encoding="utf-8"))
        lock = json.loads((root / "package-lock.json").read_text(encoding="utf-8"))
        packages = lock["packages"]
        version = packages["node_modules/vite-plus"]["version"]
        core = f"npm:@voidzero-dev/vite-plus-core@{version}"

        assert re.fullmatch(r"\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?", version), root
        assert manifest["devDependencies"]["vite-plus"] == version, root
        assert manifest["devDependencies"]["vite"] == core, root
        assert manifest["overrides"]["vite"] == core, root
        assert packages[""]["devDependencies"] == manifest["devDependencies"], root
        assert packages["node_modules/vite"]["name"] == "@voidzero-dev/vite-plus-core", root
        assert packages["node_modules/vite"]["version"] == version, root
        assert not [path for path in packages if "biome" in path], root
        assert not (root / "biome.json").exists(), root
    template = json.loads((TEMPLATES / APP / "package.json").read_text(encoding="utf-8"))
    dashboard = json.loads((REPOSITORY / "dashboard/package.json").read_text(encoding="utf-8"))
    assert template["devDependencies"]["vite-plus"] == dashboard["devDependencies"]["vite-plus"]


def test_the_operator_app_is_the_generic_app_with_the_operators_layer() -> None:
    app = layer_sources(TEMPLATES)
    composed = layer_sources(TEMPLATES, (OPERATORS,))
    layer = TEMPLATES / OPERATORS

    from_layer = {
        path for path, source in composed.items() if Path(str(source)).is_relative_to(layer)
    }
    # The layer replaces only the area registry, and adds Task runs and its smoke test. The
    # smoke test's package and Playwright configuration are the generic app's.
    assert from_layer & app.keys() == {"src/agent/index.tsx"}
    assert {"src/agent/task-runs.tsx", "e2e/smoke.spec.ts"} <= from_layer
    assert {
        "e2e/package.json",
        "e2e/package-lock.json",
        "e2e/playwright.config.ts",
    } <= app.keys()
    assert "static/index.html" in app_build(TEMPLATES, (OPERATORS,))
    registry = (layer / "src/agent/index.tsx").read_text(encoding="utf-8")
    assert "export const shell: ShellParts = {};" in registry
    assert "taskRunsArea," in registry
    # Its people are the Agent's operators: it names no organizations, members or departments.
    for path in sorted(from_layer - {"e2e/package-lock.json"}):
        text = (layer / path).read_text(encoding="utf-8")
        assert not re.search(r"organi[sz]ation|department|\bmember|b2b", text, re.I), path
    # The long-running preset gets exactly this app, and the coding preset, which has
    # knowledge, this app with the Knowledge area.
    for preset, layers in (("long-running", (OPERATORS,)), ("coding", (OPERATORS, KNOWLEDGE))):
        files = project_files(PROJECT, preset=preset)
        assert {path for path in web_app_files(PROJECT, layers=layers)} == {
            path for path in files if path.startswith("web/")
        }
        assert '"name": "web",' in files["web/package.json"]


def test_the_agent_presets_app_adds_conversations_through_cayus_api() -> None:
    operators = layer_sources(TEMPLATES, (OPERATORS,))
    composed = layer_sources(TEMPLATES, (OPERATORS, CONVERSATIONS))
    layer = TEMPLATES / CONVERSATIONS

    from_layer = {
        path for path, source in composed.items() if Path(str(source)).is_relative_to(layer)
    }
    # It replaces the area registry, and adds the Conversations area.
    assert from_layer & operators.keys() == {"src/agent/index.tsx"}
    assert {
        "src/agent/conversations.tsx",
        "src/agent/cayu.ts",
        "src/agent/markdown.tsx",
    } <= from_layer
    assert "static/index.html" in app_build(TEMPLATES, (OPERATORS, CONVERSATIONS))
    registry = (layer / "src/agent/index.tsx").read_text(encoding="utf-8")
    assert "conversationsArea,\n  taskRunsArea," in registry
    # Conversations go through Cayu's own API, which applies its role rules, not /app.
    client = (layer / "src/agent/cayu.ts").read_text(encoding="utf-8")
    assert 'new URL("/api/", window.location.origin)' in client
    assert "fetch(`/api/${path}`" in client
    # A conversation answered in Cayu's inbox shows without a reload: it's read again on
    # coming back and watched while unsettled, but never while this page streams its reply.
    page = (layer / "src/agent/conversations.tsx").read_text(encoding="utf-8")
    assert "refetchOnWindowFocus: !live," in page
    assert '"interrupted"' in page and "watchUnsettled(query)" in page
    assert "refetchInterval: live ? false :" in page
    assert page.index("setStreaming(id, true)") < page.index('stream("resume"')
    assert "} finally {\n      setStreaming(id, false);" in page
    # The agent preset has knowledge, so its app has the Knowledge area on top.
    files = project_files(PROJECT)
    assert set(web_app_files(PROJECT, layers=(OPERATORS, CONVERSATIONS, KNOWLEDGE))) == {
        path for path in files if path.startswith("web/")
    }


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
        # Its smoke test signs a demonstration person in first.
        "e2e/smoke.spec.ts",
    } <= from_layer
    # The b2b preset has no knowledge, so no Knowledge area.
    files = project_files(PROJECT, preset="b2b")
    assert set(web_app_files(PROJECT, layers=(ORGANIZATIONS,))) == {
        path for path in files if path.startswith("web/")
    }
    assert not [path for path in files if "knowledge" in path and path.startswith("web/")]
    assert '"name": "b2b-web",' in files["web/package.json"]
    assert files["web/package-lock.json"].count('"name": "b2b-web",') == 2


def test_a_layer_adds_files_and_replaces_only_the_agents_own(tmp_path: Path) -> None:
    _write(
        tmp_path / APP,
        {
            "src/app/area.ts": "/** GENERATED. The area pattern. */\n",
            "src/agent/index.tsx": "/** THE AGENT'S OWN. The areas. */\n",
        },
    )
    _write(
        tmp_path / "areas",
        {
            "src/agent/index.tsx": "/** THE AGENT'S OWN. More areas. */\n",
            "src/agent/more.tsx": "/** THE AGENT'S OWN. Another area. */\n",
        },
    )
    _write(tmp_path / "shell", {"src/app/area.ts": "/** GENERATED. Another pattern. */\n"})
    _write(tmp_path / BUILDS / "areas", {"index.html": "<title>Agent</title>\n"})

    composed = layer_sources(tmp_path, ("areas",))
    assert {path: source.read_text(encoding="utf-8") for path, source in composed.items()} == {
        "src/app/area.ts": "/** GENERATED. The area pattern. */\n",
        "src/agent/index.tsx": "/** THE AGENT'S OWN. More areas. */\n",
        "src/agent/more.tsx": "/** THE AGENT'S OWN. Another area. */\n",
    }
    with pytest.raises(ValueError, match="replaces src/app/area.ts"):
        layer_sources(tmp_path, ("shell",))
    # The composition's build is its own, whole; the generic app on its own has none.
    assert set(app_build(tmp_path, ("areas",))) == {"static/index.html"}
    assert app_build(tmp_path) == {}
    with pytest.raises(RuntimeError, match="No committed build of shell"):
        app_build(tmp_path, ("shell",))


def test_every_composition_cayu_new_writes_has_its_build_and_no_other_build_is_kept() -> None:
    compositions = {
        (OPERATORS,),
        (OPERATORS, CONVERSATIONS),
        (ORGANIZATIONS,),
        (OPERATORS, KNOWLEDGE),
        (OPERATORS, CONVERSATIONS, KNOWLEDGE),
        (ORGANIZATIONS, KNOWLEDGE),
    }
    assert build_name((OPERATORS, CONVERSATIONS, KNOWLEDGE)) == "operators+conversations+knowledge"
    for layers in compositions:
        build = app_build(TEMPLATES, layers)
        page = build["static/index.html"].read_text(encoding="utf-8")
        # The page names the built files it loads, and the build holds them.
        for asset in re.findall(r'(?:src|href)="/(assets/[^"]+)"', page):
            assert f"static/{asset}" in build, (layers, asset)
    kept = {path.name for path in (TEMPLATES / BUILDS).iterdir() if path.is_dir()}
    assert kept == {build_name(layers) for layers in compositions}


def test_the_knowledge_layer_adds_the_area_on_top_of_any_app() -> None:
    layer = TEMPLATES / KNOWLEDGE
    for base in ((OPERATORS,), (OPERATORS, CONVERSATIONS), (ORGANIZATIONS,)):
        below = app_sources(TEMPLATES, base)
        composed = app_sources(TEMPLATES, (*base, KNOWLEDGE))
        added = composed.keys() - below.keys()
        # It adds the area, its client and its smoke test, and replaces no file...
        assert added == {
            "src/generated/knowledge/index.tsx",
            "src/api/knowledge.ts",
            "e2e/knowledge.spec.ts",
        }
        assert all(composed[path] == below[path] for path in below if path != "src/agent/index.tsx")
        # ...but lists the area in the app's registry, just before Settings.
        registry = composed["src/agent/index.tsx"]
        assert registry == with_knowledge_area(below["src/agent/index.tsx"])
        assert 'import { knowledgeArea } from "../generated/knowledge";' in registry
        assert re.search(r"knowledgeArea,\s+settingsArea(?:,|\])", registry)
    # The page is generated, and so is its client: they are the generator's to rewrite.
    for path in ("src/generated/knowledge/index.tsx", "src/api/knowledge.ts"):
        assert "GENERATED." in (layer / path).read_text(encoding="utf-8")
    # It names no organizations: who reads and changes what is the server's to say.
    page = (layer / "src/generated/knowledge/index.tsx").read_text(encoding="utf-8")
    assert not re.search(r"organi[sz]ation|\bmember|b2b", page, re.I)
    # A one-line list, as the generic app has, takes it too; a registry without Settings doesn't.
    one_line = (
        'import { settingsArea } from "../generated/settings";\n'
        "export const areas: AppArea[] = [overviewArea(), settingsArea];\n"
    )
    assert "[overviewArea(), knowledgeArea, settingsArea]" in with_knowledge_area(one_line)
    with pytest.raises(RuntimeError, match="doesn't list the generated Settings area once"):
        with_knowledge_area("export const areas: AppArea[] = [overviewArea()];\n")


def test_an_agent_with_knowledge_gets_the_knowledge_area_and_one_without_it_doesnt() -> None:
    for choices in (
        {},
        {"preset": "coding"},
        {"with_capabilities": ("organizations",)},
    ):
        files = project_files(PROJECT, **choices)
        assert "web/src/generated/knowledge/index.tsx" in files, choices
        assert "integrations/http/generated/knowledge.py" in files, choices
        assert "tests/test_knowledge_area.py" in files, choices
        assert "knowledgeArea," in files["web/src/agent/index.tsx"]
    for choices in (
        {"without_capabilities": ("knowledge", "memory")},
        {"preset": "long-running"},
        {"preset": "b2b"},
    ):
        files = project_files(PROJECT, **choices)
        assert not [
            path
            for path in files
            if "knowledge" in path.split("/")[-1]
            and (
                path.startswith(("web/", "integrations/")) or path == "tests/test_knowledge_area.py"
            )
        ], choices
        assert "knowledge" not in files["integrations/http/__init__.py"]


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
