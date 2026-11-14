"""Build, or check, the web apps `cayu new` writes into `web/`.

Their source lives in the package so `cayu new` can copy it: the generic app in
`src/cayu/cli/templates/app_web/` and the layers on top of it, composed as
`src/cayu/cli/scaffold_app_web.py` says. Installed packages must never land there, so each app
is composed and built in a temporary folder:

- the generic app on its own, which ships no build yet;
- the generic app with the organizations layer, the b2b preset's, whose committed build is
  the layer's `static/`.

Without `--check` each app is built, and a committed build is replaced with the fresh one.
With `--check` each is linted and typechecked; one with a committed build then runs its own
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
        "--check", action="store_true", help="lint, typecheck and fail on a stale build"
    )
    args = parser.parse_args(argv)
    web = _composition()
    apps = (("generic app", ()), ("b2b app", (web.ORGANIZATIONS,)))
    for label, layers in apps:
        print(f"== The {label} ({' + '.join((web.APP, *layers))})", flush=True)
        if not _build(web, layers, check=args.check):
            return 1
    return 0


def _build(web, layers: tuple[str, ...], *, check: bool) -> bool:
    sources = web.layer_sources(TEMPLATES, layers)
    built = f"{web.BUILT}/"
    # The folder whose static/ is this app's committed build, if it has one.
    page = sources.get(f"{built}index.html")
    owner = Path(str(page)).parents[1] if page is not None else None
    with tempfile.TemporaryDirectory(prefix="cayu-app-web-") as directory:
        work = Path(directory) / "web"
        for path, source in sources.items():
            # The previous build only for `check:build` to compare.
            if path.startswith(built) and not check:
                continue
            (work / path).parent.mkdir(parents=True, exist_ok=True)
            (work / path).write_bytes(source.read_bytes())
        steps: list[list[str]] = [["ci", "--no-audit", "--no-fund"]]
        if check:
            steps += [["run", "lint"], ["run", "typecheck"]]
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
            shutil.rmtree(owner / web.BUILT, ignore_errors=True)
            shutil.copytree(work / web.BUILT, owner / web.BUILT)
    return True


if __name__ == "__main__":
    raise SystemExit(main())
