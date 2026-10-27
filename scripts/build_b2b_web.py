"""Build, or check, the B2B preset's packaged web interface.

The interface's source lives in the package so `cayu new --preset b2b` can copy it, which means
its installed packages must never land there. This builds in a temporary copy and writes back
only `static/`. With `--check` it also lints and typechecks, and fails when the committed build
differs from the source.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "src" / "cayu" / "cli" / "templates" / "b2b_web"
BUILT = "static"


def _differences(left: Path, right: Path, prefix: Path) -> list[str]:
    comparison = filecmp.dircmp(left, right)
    names = (
        comparison.left_only
        + comparison.right_only
        + comparison.common_funny
        + comparison.funny_files
    )
    _, mismatched, errors = filecmp.cmpfiles(left, right, comparison.common_files, shallow=False)
    found = [str(prefix / name) for name in names + mismatched + errors]
    for name in comparison.common_dirs:
        found += _differences(left / name, right / name, prefix / name)
    return found


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
            # Installed packages anywhere, and the previous build at the top.
            ignore=lambda directory, names: {
                name
                for name in names
                if name == "node_modules" or (Path(directory) == TEMPLATE and name == BUILT)
            },
        )
        steps: list[list[str]] = [["ci", "--no-audit", "--no-fund"]]
        if args.check:
            # The build typechecks too.
            steps.append(["run", "lint"])
        steps.append(["run", "build"])
        for step in steps:
            subprocess.run(["npm", *step], cwd=work, check=True)
        if args.check:
            stale = _differences(TEMPLATE / BUILT, work / BUILT, Path(BUILT))
            if stale:
                print(
                    "The committed B2B web build differs from its source; run "
                    "`python scripts/build_b2b_web.py` and commit the result:\n  "
                    + "\n  ".join(stale),
                    file=sys.stderr,
                )
                return 1
            return 0
        shutil.rmtree(TEMPLATE / BUILT, ignore_errors=True)
        shutil.copytree(work / BUILT, TEMPLATE / BUILT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
