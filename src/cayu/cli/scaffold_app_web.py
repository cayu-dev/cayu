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
registry; never a `GENERATED.` file, a building block or the build tooling. A layer's
`static/` is the committed build of the app with the layers up to it, and replaces the app's
whole: build output has hashed names, so another build's files can't stay beside it.
`scripts/build_app_web.py` builds and checks each composition.

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

# The committed build `cayu serve` serves ([tool.cayu.serve] ui), relative to the app.
BUILT = "static"
# The marker of a file the Agent owns, which a layer may replace.
AGENT_OWN = "THE AGENT'S OWN."
# The pages' title in the templates and their builds: the app shows it as the Agent's name.
TITLE = "<title>Agent</title>"
PAGES = ("index.html", f"{BUILT}/index.html")
# The app's npm package name, in package.json and twice in package-lock.json.
PACKAGE_NAME = '"name": "web",'
_PACKAGE_NAMES = {"package.json": 1, "package-lock.json": 2}


def templates() -> Traversable:
    return package_files("cayu.cli").joinpath("templates")


def layer_sources(root: Traversable, layers: Sequence[str] = ()) -> dict[str, Traversable]:
    """Which template file each of the composed app's files comes from, by its path in the app.

    Raises ValueError when a layer would replace a file the app doesn't mark as the Agent's
    own."""

    sources = _tree(root.joinpath(APP))
    for name in layers:
        layer = _tree(root.joinpath(name))
        if any(path.startswith(f"{BUILT}/") for path in layer):
            sources = {
                path: item for path, item in sources.items() if not path.startswith(f"{BUILT}/")
            }
        for path, item in layer.items():
            replaced = sources.get(path)
            if replaced is not None and AGENT_OWN not in replaced.read_text(encoding="utf-8"):
                raise ValueError(
                    f"The {name} layer replaces {path}, which isn't the Agent's own; a layer "
                    "adds files and replaces only those marked THE AGENT'S OWN."
                )
            sources[path] = item
    return sources


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

    sources = layer_sources(root if root is not None else templates(), layers)
    files = {path: item.read_text(encoding="utf-8") for path, item in sources.items()}
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
