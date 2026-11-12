"""Build, or check, the B2B preset's packaged web interface.

The interface's source lives in the package so `cayu new --preset b2b` can copy it, which means
its installed packages must never land there. This builds in a temporary copy and writes back
only `static/`. With `--check` it lints and typechecks instead, and runs the interface's own
`npm run check:build`, the check a generated project runs, which fails when the committed build
differs from a fresh build of the source.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "src" / "cayu" / "cli" / "templates" / "b2b_web"
BUILT = "static"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="lint, typecheck and fail on a stale build"
    )
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="cayu-b2b-web-") as directory:
        work = Path(directory) / "web"
        shutil.copytree(
            TEMPLATE,
            work,
            # Installed packages anywhere; the previous build only for `check:build` to compare.
            ignore=lambda directory, names: {
                name
                for name in names
                if name == "node_modules"
                or (not args.check and Path(directory) == TEMPLATE and name == BUILT)
            },
        )
        steps: list[list[str]] = [["ci", "--no-audit", "--no-fund"]]
        if args.check:
            steps += [["run", "lint"], ["run", "typecheck"], ["run", "check:build"]]
        else:
            # The build typechecks too.
            steps.append(["run", "build"])
        for step in steps:
            if subprocess.run(["npm", *step], cwd=work, check=False).returncode != 0:
                if step == ["run", "check:build"]:
                    print(
                        "The committed B2B web build differs from its source; run "
                        "`python scripts/build_b2b_web.py` and commit the result.",
                        file=sys.stderr,
                    )
                return 1
        if args.check:
            return 0
        shutil.rmtree(TEMPLATE / BUILT, ignore_errors=True)
        shutil.copytree(work / BUILT, TEMPLATE / BUILT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
