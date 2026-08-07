"""Compare an ejected dashboard build with the assets in the installed Cayu wheel."""

from __future__ import annotations

import argparse
import sys
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dashboard_build", type=Path)
    args = parser.parse_args(argv)

    expected = _resource_contents(files("cayu").joinpath("server", "dashboard"))
    actual = _directory_contents(args.dashboard_build)
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    changed = sorted(
        path for path in expected.keys() & actual.keys() if expected[path] != actual[path]
    )
    if missing or unexpected or changed:
        if missing:
            print(f"missing built dashboard files: {', '.join(missing)}", file=sys.stderr)
        if unexpected:
            print(f"unexpected built dashboard files: {', '.join(unexpected)}", file=sys.stderr)
        if changed:
            print(f"changed built dashboard files: {', '.join(changed)}", file=sys.stderr)
        return 1
    print(f"validated {len(actual)} ejected dashboard build files against installed Cayu assets")
    return 0


def _resource_contents(root: Traversable) -> dict[str, bytes]:
    if not root.is_dir():
        raise ValueError("installed Cayu package omits compiled dashboard assets")
    contents: dict[str, bytes] = {}

    def visit(directory: Traversable, prefix: PurePosixPath) -> None:
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = prefix / child.name
            if child.is_dir():
                visit(child, relative)
            elif child.is_file():
                contents[relative.as_posix()] = child.read_bytes()
            else:
                raise ValueError(f"installed dashboard contains unsupported entry: {relative}")

    visit(root, PurePosixPath())
    return contents


def _directory_contents(root: Path) -> dict[str, bytes]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"dashboard build must be an ordinary directory: {root}")
    contents: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"dashboard build must not contain links: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"dashboard build contains unsupported entry: {relative}")
        contents[relative] = path.read_bytes()
    return contents


if __name__ == "__main__":
    raise SystemExit(main())
