"""The web app `cayu new` writes into `web/`: a generic app, and layers on top of it.

`templates/app_web/` is the app every layer builds on, with nothing about who its people are
beyond the role `GET /app/whoami` reports: the shell and its `@kit` layer, the generated
Overview and Settings, the TaskRuns block, the status pages, the data layer, the build
scripts, and the smoke test's package and Playwright configuration in `e2e/`, which serves the
app with `cayu serve --dev`. Its `src/agent/index.tsx` is the area registry: the Agent's areas in navigation order
and what it adds to the shell (its sign-in page, items in the profile menu), which the
generated shell reads.

A layer is a folder next to it whose files go on top, in order. It adds its own areas, pages
and API calls, and replaces only files the app marks `THE AGENT'S OWN.`, above all that
registry; never a `GENERATED.` file, a building block or the build tooling.

`templates/app_web_operators/` is the layer of the `web-app` capability, for an Agent whose
people are its operators: its Task runs area, and its smoke test (`e2e/smoke.spec.ts`), which
opens every area of the app.

`templates/app_web_conversations/` goes on top of the operators layer for an Agent people chat
with (the agent preset): its Conversations area, through Cayu's own API at /api.

`templates/app_web_organizations/` is the layer of the `organizations` capability (the b2b
preset), on the generic app alone: member sign-in for `cayu serve --dev`, the Members area,
the organization switcher, conversations scoped to an organization, their calls to the
organizations' own routes, and a smoke test that signs a demonstration person in and opens
every area.

`templates/app_web_knowledge/` goes last, on top of any of those apps, when the Agent has the
`knowledge` capability: the generated Knowledge area (`src/generated/knowledge/`, reading
`/app/knowledge`) and a smoke test for it. It replaces no file: composing it lists the area in
the app's registry, just before Settings.

`templates/app_web_builds/<name>/` is the committed build of each composition `cayu new`
writes, named by its layers (`build_name`), which becomes the project's `web/static`. A build's
files have hashed names, so each composition has a whole build of its own.
`scripts/build_app_web.py` builds and checks each composition.

This module imports nothing from Cayu, so the build script loads it from its file.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from importlib.resources import files as package_files
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

APP = "app_web"
OPERATORS = "app_web_operators"
CONVERSATIONS = "app_web_conversations"
ORGANIZATIONS = "app_web_organizations"
KNOWLEDGE = "app_web_knowledge"
# The committed builds, one folder for each composition (`build_name`).
BUILDS = "app_web_builds"

# Where the composed app keeps its committed build, which `cayu serve` serves
# ([tool.cayu.serve] ui), relative to the app.
BUILT = "static"
# The marker of a file the Agent owns, which a layer may replace.
AGENT_OWN = "THE AGENT'S OWN."
# The pages' title in the templates and their builds: the app shows it as the Agent's name.
TITLE = "<title>Agent</title>"
PAGES = ("index.html", f"{BUILT}/index.html")
# The app's npm package name, in package.json and twice in package-lock.json.
PACKAGE_NAME = '"name": "web",'
_PACKAGE_NAMES = {"package.json": 1, "package-lock.json": 2}
# What composing the knowledge layer changes in the app's area registry.
REGISTRY = "src/agent/index.tsx"
_SETTINGS_IMPORT = 'import { settingsArea } from "../generated/settings";\n'
_KNOWLEDGE_IMPORT = 'import { knowledgeArea } from "../generated/knowledge";\n'
_GENERATED_AREAS = "the generated Overview and Settings"


def templates() -> Traversable:
    return package_files("cayu.cli").joinpath("templates")


def layer_sources(root: Traversable, layers: Sequence[str] = ()) -> dict[str, Traversable]:
    """Which template file each of the composed app's source files comes from, by its path in
    the app. Its build is `app_build`'s.

    Raises ValueError when a layer would replace a file the app doesn't mark as the Agent's
    own."""

    sources = _tree(root.joinpath(APP))
    for name in layers:
        for path, item in _tree(root.joinpath(name)).items():
            replaced = sources.get(path)
            if replaced is not None and AGENT_OWN not in replaced.read_text(encoding="utf-8"):
                raise ValueError(
                    f"The {name} layer replaces {path}, which isn't the Agent's own; a layer "
                    "adds files and replaces only those marked THE AGENT'S OWN."
                )
            sources[path] = item
    return sources


def app_sources(root: Traversable, layers: Sequence[str] = ()) -> dict[str, str]:
    """The composed app's source files and their text: the layers' files on the generic app,
    with the knowledge layer's area listed in the registry."""

    files = {
        path: item.read_text(encoding="utf-8") for path, item in layer_sources(root, layers).items()
    }
    if KNOWLEDGE in layers:
        files[REGISTRY] = with_knowledge_area(files[REGISTRY])
    return files


def with_knowledge_area(registry: str) -> str:
    """``registry`` with the generated Knowledge area listed just before Settings.

    Raises RuntimeError when it doesn't import and list Settings once each."""

    start = registry.find("export const areas: AppArea[] = [")
    listed = registry.find("settingsArea", start)
    if (
        registry.count(_SETTINGS_IMPORT) != 1
        or start < 0
        or listed < 0
        or registry.find("settingsArea", listed + 1) >= 0
    ):
        raise RuntimeError(f"{REGISTRY} doesn't list the generated Settings area once.")
    # On a line of its own, as the layers list their areas, or in a one-line list.
    indent = registry[registry.rfind("\n", 0, listed) + 1 : listed]
    entry = f"knowledgeArea,\n{indent}" if not indent.strip() else "knowledgeArea, "
    registry = registry[:listed] + entry + registry[listed:]
    registry = registry.replace(_SETTINGS_IMPORT, _KNOWLEDGE_IMPORT + _SETTINGS_IMPORT)
    return registry.replace(_GENERATED_AREAS, "the generated Overview, Knowledge and Settings", 1)


def build_name(layers: Sequence[str]) -> str:
    """The name of a composition's build in `app_web_builds/`: its layers without the
    `app_web_` prefix, joined by `+`, as `operators+conversations+knowledge`."""

    return "+".join(name.removeprefix(f"{APP}_") for name in layers)


def app_build(root: Traversable, layers: Sequence[str] = ()) -> dict[str, Traversable]:
    """The composed app's committed build, by its path in the app (`static/...`): none for the
    generic app on its own.

    Raises RuntimeError when the composition has no committed build."""

    if not layers:
        return {}
    folder = root.joinpath(BUILDS).joinpath(build_name(layers))
    if not folder.is_dir():
        raise RuntimeError(
            f"No committed build of {build_name(layers)}; run `python scripts/build_app_web.py`."
        )
    return {f"{BUILT}/{path}": item for path, item in _tree(folder).items()}


def web_app_files(
    title: str,
    *,
    layers: Sequence[str] = (),
    package_name: str = "web",
    root: Traversable | None = None,
) -> dict[str, str]:
    """The files for `web/`: the app's source, which a project rebuilds with
    `npm --prefix web run build`, and the built files in `web/static` it serves.

    Both pages are titled `title`. Vite copies the title as it is, and the page isn't named by
    its content, so a fresh build still matches `web/static`. The npm package is named
    `package_name`, which no build reads."""

    root = root if root is not None else templates()
    files = app_sources(root, layers)
    files.update(
        {path: item.read_text(encoding="utf-8") for path, item in app_build(root, layers).items()}
    )
    for page in PAGES:
        if files.get(page, "").count(TITLE) != 1:
            raise RuntimeError(f"{page} in the web app has no {TITLE}")
        files[page] = files[page].replace(TITLE, f"<title>{html.escape(title)}</title>")
    for path, count in _PACKAGE_NAMES.items():
        if files[path].count(PACKAGE_NAME) != count:
            raise RuntimeError(f"{path} in the web app doesn't name its package {PACKAGE_NAME}")
        files[path] = files[path].replace(PACKAGE_NAME, f'"name": {json.dumps(package_name)},')
    # In the order of a walk of the folders, each sorted by name.
    return {f"web/{path}": files[path] for path in sorted(files, key=lambda path: path.split("/"))}


def _tree(node: Traversable, prefix: str = "") -> dict[str, Traversable]:
    found: dict[str, Traversable] = {}
    for child in sorted(node.iterdir(), key=lambda item: item.name):
        if child.is_dir():
            found.update(_tree(child, f"{prefix}{child.name}/"))
        else:
            found[f"{prefix}{child.name}"] = child
    return found
