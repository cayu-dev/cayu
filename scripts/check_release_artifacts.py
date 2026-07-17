"""Validate Cayu source and wheel archives before publication."""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

_SDIST_REQUIRED = {
    "LICENSE",
    "NOTICE",
    "PKG-INFO",
    "README.md",
    "pyproject.toml",
    "src/cayu/__init__.py",
    "src/cayu/data/__init__.py",
    "src/cayu/data/default_model_catalog.json",
    "src/cayu/data/default_price_book.json",
    "src/cayu/server/dashboard/THIRD_PARTY_LICENSES.md",
}
_SDIST_ALLOWED_ROOTS = {
    ".gitignore",
    "LICENSE",
    "NOTICE",
    "PKG-INFO",
    "README.md",
    "pyproject.toml",
}
_SDIST_ALLOWED_TREES = {"src"}
_WHEEL_REQUIRED = {
    "cayu/__init__.py",
    "cayu/cli/_targets.py",
    "cayu/cli/__init__.py",
    "cayu/cli/console.py",
    "cayu/data/__init__.py",
    "cayu/data/default_model_catalog.json",
    "cayu/data/default_price_book.json",
    "cayu/guides/application-anatomy.md",
    "cayu/guides/authoring.md",
    "cayu/guides/diagnostics.md",
    "cayu/guides/tool-effects.md",
    "cayu/server/dashboard/THIRD_PARTY_LICENSES.md",
    "cayu/server/dashboard/index.html",
}
_THIRD_PARTY_LICENSE_MARKERS = {
    "## @base-ui/react -",
    "## class-variance-authority -",
    "## lucide-react -",
    "## react -",
    "## shadcn/ui registry source (MIT)",
    "## tailwindcss -",
    "## tw-animate-css -",
}
_FORBIDDEN_PARTS = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "dist"}


def _fail(message: str) -> None:
    raise ValueError(message)


def _validate_safe_path(name: str, *, archive: Path) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        _fail(f"{archive}: unsafe archive path: {name}")
    if _FORBIDDEN_PARTS.intersection(path.parts):
        _fail(f"{archive}: forbidden build-local path: {name}")
    if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
        _fail(f"{archive}: generated Python cache included: {name}")
    return path


def _validate_third_party_licenses(contents: bytes, *, archive: Path) -> None:
    try:
        notice = contents.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{archive}: third-party license inventory is not UTF-8") from exc

    missing = sorted(marker for marker in _THIRD_PARTY_LICENSE_MARKERS if marker not in notice)
    if missing:
        _fail(f"{archive}: third-party license inventory is incomplete: {', '.join(missing)}")


def validate_sdist(archive: Path) -> None:
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
    if not members:
        _fail(f"{archive}: source distribution is empty")
    if any(member.issym() or member.islnk() for member in members):
        _fail(f"{archive}: source distribution must not contain links")

    paths = [_validate_safe_path(member.name, archive=archive) for member in members]
    roots = {path.parts[0] for path in paths if path.parts}
    if len(roots) != 1:
        _fail(f"{archive}: expected one source-distribution root, found {sorted(roots)}")
    root = next(iter(roots))
    relative_names: set[str] = set()
    for path in paths:
        relative = path.relative_to(root)
        if not relative.parts:
            continue
        relative_names.add(str(relative))
        top = relative.parts[0]
        if top not in _SDIST_ALLOWED_ROOTS and top not in _SDIST_ALLOWED_TREES:
            _fail(f"{archive}: unexpected source-distribution path: {relative}")

    missing = sorted(_SDIST_REQUIRED - relative_names)
    if missing:
        _fail(f"{archive}: missing required source files: {', '.join(missing)}")

    notice_name = f"{root}/src/cayu/server/dashboard/THIRD_PARTY_LICENSES.md"
    with tarfile.open(archive, "r:gz") as source:
        extracted = source.extractfile(notice_name)
        if extracted is None:
            raise ValueError(f"{archive}: could not read third-party license inventory")
        notice_contents = extracted.read()
    _validate_third_party_licenses(notice_contents, archive=archive)


def validate_wheel(archive: Path) -> None:
    with zipfile.ZipFile(archive) as wheel:
        names = wheel.namelist()
    if not names:
        _fail(f"{archive}: wheel is empty")
    paths = [_validate_safe_path(name, archive=archive) for name in names]
    name_set = {str(path) for path in paths}

    missing = sorted(_WHEEL_REQUIRED - name_set)
    if missing:
        _fail(f"{archive}: missing required wheel files: {', '.join(missing)}")
    with zipfile.ZipFile(archive) as wheel:
        notice_contents = wheel.read("cayu/server/dashboard/THIRD_PARTY_LICENSES.md")
    _validate_third_party_licenses(notice_contents, archive=archive)
    if not any(
        name.startswith("cayu/server/dashboard/assets/") and name.endswith(".js")
        for name in name_set
    ):
        _fail(f"{archive}: packaged dashboard JavaScript is missing")
    if not any(
        name.startswith("cayu/server/dashboard/assets/") and name.endswith(".css")
        for name in name_set
    ):
        _fail(f"{archive}: packaged dashboard CSS is missing")

    dist_info = {path.parts[0] for path in paths if path.parts[0].endswith(".dist-info")}
    if len(dist_info) != 1:
        _fail(f"{archive}: expected one .dist-info directory, found {sorted(dist_info)}")
    metadata_root = next(iter(dist_info))
    top_level = {path.parts[0] for path in paths if path.parts}
    unexpected = sorted(top_level - {"cayu", metadata_root})
    if unexpected:
        _fail(f"{archive}: unexpected wheel top-level paths: {', '.join(unexpected)}")
    metadata_required = {
        f"{metadata_root}/METADATA",
        f"{metadata_root}/RECORD",
        f"{metadata_root}/WHEEL",
        f"{metadata_root}/entry_points.txt",
        f"{metadata_root}/licenses/LICENSE",
        f"{metadata_root}/licenses/NOTICE",
    }
    missing_metadata = sorted(metadata_required - name_set)
    if missing_metadata:
        _fail(f"{archive}: missing wheel metadata: {', '.join(missing_metadata)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args(argv)

    sdists = [path for path in args.archives if path.name.endswith(".tar.gz")]
    wheels = [path for path in args.archives if path.suffix == ".whl"]
    if len(sdists) != 1 or len(wheels) != 1:
        parser.error("pass exactly one .tar.gz source distribution and one .whl wheel")
    validate_sdist(sdists[0])
    validate_wheel(wheels[0])
    print(f"validated {sdists[0]} and {wheels[0]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
