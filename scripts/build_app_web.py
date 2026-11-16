"""Build, or check, the web apps `cayu new` writes into `web/`.

Their source lives in the package so `cayu new` can copy it: the generic app in
`src/cayu/cli/templates/app_web/` and the layers on top of it, composed as
`src/cayu/cli/scaffold_app_web.py` says. Installed packages must never land there, so each app
is composed and built in a temporary folder:

- the generic app on its own, which ships no build;
- the generic app with the operators layer, the `web-app` capability's;
- that app with the conversations layer, the agent preset's;
- the generic app with the organizations layer, the b2b preset's;
- each of those three with the knowledge layer on top, for an Agent with knowledge.

Each composition but the generic app has a committed build in
`src/cayu/cli/templates/app_web_builds/<name>/`, which `cayu new` copies to `web/static`.

Without `--check` each app is built, and a committed build is replaced with the fresh one.
With `--check` each is linted, format-checked and typechecked; one with a committed build then runs its own
`npm run check:build`, the check a generated project runs, which fails when the committed build
differs from a fresh build of the source, and one without is built.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "src" / "cayu" / "cli" / "templates"


def _composition():
    # Loaded from its file: importing the `cayu` package needs its dependencies, and this
    # script runs with any Python.
    path = ROOT / "src" / "cayu" / "cli" / "scaffold_app_web.py"
    spec = importlib.util.spec_from_file_location("scaffold_app_web", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="lint, check formatting, typecheck and fail on a stale build",
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="NAME",
        help="only this composition, by its build's name (operators+knowledge); repeatable",
    )
    args = parser.parse_args(argv)
    web = _composition()
    apps = (
        ("generic app", ()),
        ("operator app", (web.OPERATORS,)),
        ("operator app with conversations", (web.OPERATORS, web.CONVERSATIONS)),
        ("b2b app", (web.ORGANIZATIONS,)),
        ("operator app with knowledge", (web.OPERATORS, web.KNOWLEDGE)),
        (
            "operator app with conversations and knowledge",
            (web.OPERATORS, web.CONVERSATIONS, web.KNOWLEDGE),
        ),
        ("organizations app with knowledge", (web.ORGANIZATIONS, web.KNOWLEDGE)),
    )
    for label, layers in apps:
        if args.only and web.build_name(layers) not in args.only:
            continue
        print(f"== The {label} ({' + '.join((web.APP, *layers))})", flush=True)
        if not _build(web, layers, check=args.check):
            return 1
    return 0


def _build(web, layers: tuple[str, ...], *, check: bool) -> bool:
    sources = web.app_sources(TEMPLATES, layers)
    # The folder that holds this app's committed build. The generic app on its own has none.
    owner = TEMPLATES / web.BUILDS / web.build_name(layers) if layers else None
    with tempfile.TemporaryDirectory(prefix="cayu-app-web-") as directory:
        work = Path(directory) / "web"
        for path, text in sources.items():
            (work / path).parent.mkdir(parents=True, exist_ok=True)
            (work / path).write_bytes(text.encode("utf-8"))
        if check and owner is not None:
            # The committed build, for `check:build` to compare.
            for path, source in web.app_build(TEMPLATES, layers).items():
                (work / path).parent.mkdir(parents=True, exist_ok=True)
                (work / path).write_bytes(source.read_bytes())
        steps: list[list[str]] = [["ci", "--no-audit", "--no-fund"]]
        if check:
            steps += [["run", "lint"], ["run", "format:check"], ["run", "typecheck"]]
            # A fresh build compared with the committed one, else just a build.
            steps.append(["run", "check:build"] if owner is not None else ["run", "build"])
        else:
            # The build typechecks too.
            steps.append(["run", "build"])
        for step in steps:
            if subprocess.run(["npm", *step], cwd=work, check=False).returncode != 0:
                if step == ["run", "check:build"]:
                    print(
                        f"The committed build in {owner} differs from its source; run "
                        "`python scripts/build_app_web.py` and commit the result.",
                        file=sys.stderr,
                    )
                return False
        if not check and owner is not None:
            shutil.rmtree(owner, ignore_errors=True)
            shutil.copytree(work / web.BUILT, owner)
    return True


if __name__ == "__main__":
    raise SystemExit(main())
