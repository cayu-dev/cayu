"""Validate Cayu source and wheel archives before publication."""

from __future__ import annotations

import argparse
import stat
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from cayu.cli.lambda_microvm import _SidecarArtifactError, _validate_artifact_contents

_SIDECAR_MANIFEST = "cayu-lambda-microvm-sidecar-manifest.json"
_SDIST_SIDECAR_PREFIX = "examples/aws/lambda_microvm_sidecar"
_WHEEL_SIDECAR_PREFIX = "cayu/data/lambda_microvm_sidecar"
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
    f"{_SDIST_SIDECAR_PREFIX}/{_SIDECAR_MANIFEST}",
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
    f"{_WHEEL_SIDECAR_PREFIX}/{_SIDECAR_MANIFEST}",
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
_NON_PUBLIC_IDENTIFIERS = (
    b"vertex" + b"kg",
    b"lane" + b"-" + b"agent",
)


def _fail(message: str) -> None:
    raise ValueError(message)


def _validate_safe_path(name: str, *, archive: Path) -> PurePosixPath:
    try:
        normalized_name = name.casefold().encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{archive}: archive path is not valid UTF-8: {name!r}") from exc
    if any(identifier in normalized_name for identifier in _NON_PUBLIC_IDENTIFIERS):
        _fail(f"{archive}: non-public identifier included in archive path: {name}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        _fail(f"{archive}: unsafe archive path: {name}")
    if _FORBIDDEN_PARTS.intersection(path.parts):
        _fail(f"{archive}: forbidden build-local path: {name}")
    if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
        _fail(f"{archive}: generated Python cache included: {name}")
    return path


def _validate_unique_paths(paths: list[PurePosixPath], *, archive: Path) -> None:
    normalized = [str(path) for path in paths]
    if len(normalized) != len(set(normalized)):
        _fail(f"{archive}: archive contains duplicate member paths")
    casefolded = [name.casefold() for name in normalized]
    if len(casefolded) != len(set(casefolded)):
        _fail(f"{archive}: archive contains case-colliding member paths")


def _validate_third_party_licenses(contents: bytes, *, archive: Path) -> None:
    try:
        notice = contents.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{archive}: third-party license inventory is not UTF-8") from exc
    missing = sorted(marker for marker in _THIRD_PARTY_LICENSE_MARKERS if marker not in notice)
    if missing:
        _fail(f"{archive}: third-party license inventory is incomplete: {', '.join(missing)}")


def _validate_publication_contents(
    contents: bytes,
    *,
    archive: Path,
    member_name: str,
) -> None:
    normalized = contents.lower()
    if any(identifier in normalized for identifier in _NON_PUBLIC_IDENTIFIERS):
        _fail(f"{archive}: non-public identifier included in {member_name}")


def _metadata_version(contents: bytes, *, archive: Path, member_name: str) -> str:
    version = BytesParser().parsebytes(contents).get("Version")
    if version is None or not version.strip():
        _fail(f"{archive}: {member_name} has no package version")
    return version.strip()


def _validate_sidecar(
    contents: dict[str, bytes],
    *,
    archive: Path,
    package_version: str,
) -> dict[str, bytes]:
    try:
        artifact = _validate_artifact_contents(
            contents,
            expected_cayu_version=package_version,
        )
    except _SidecarArtifactError as exc:
        raise ValueError(f"{archive}: invalid sidecar artifact: {exc}") from exc
    return artifact.contents


def validate_sdist(archive: Path) -> dict[str, bytes]:
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        if not members:
            _fail(f"{archive}: source distribution is empty")
        if any(member.issym() or member.islnk() for member in members):
            _fail(f"{archive}: source distribution must not contain links")

        paths = [_validate_safe_path(member.name, archive=archive) for member in members]
        _validate_unique_paths(paths, archive=archive)
        roots = {path.parts[0] for path in paths if path.parts}
        if len(roots) != 1:
            _fail(f"{archive}: expected one source-distribution root, found {sorted(roots)}")
        root = next(iter(roots))
        relative_names: set[str] = set()
        contents_by_relative_name: dict[str, bytes] = {}
        for member, path in zip(members, paths, strict=True):
            relative = path.relative_to(root)
            if not relative.parts:
                continue
            relative_name = str(relative)
            relative_names.add(relative_name)
            top = relative.parts[0]
            if top == "examples":
                if relative.parts[:3] != ("examples", "aws", "lambda_microvm_sidecar"):
                    _fail(f"{archive}: unexpected source-distribution path: {relative}")
            elif top not in _SDIST_ALLOWED_ROOTS and top not in _SDIST_ALLOWED_TREES:
                _fail(f"{archive}: unexpected source-distribution path: {relative}")
            if not member.isfile():
                continue
            extracted = source.extractfile(member)
            if extracted is None:
                _fail(f"{archive}: could not read archive member {member.name}")
            content = extracted.read()
            contents_by_relative_name[relative_name] = content
            _validate_publication_contents(content, archive=archive, member_name=member.name)

    missing = sorted(_SDIST_REQUIRED - relative_names)
    if missing:
        _fail(f"{archive}: missing required source files: {', '.join(missing)}")
    notice_name = "src/cayu/server/dashboard/THIRD_PARTY_LICENSES.md"
    _validate_third_party_licenses(contents_by_relative_name[notice_name], archive=archive)
    package_version = _metadata_version(
        contents_by_relative_name["PKG-INFO"], archive=archive, member_name="PKG-INFO"
    )
    prefix = f"{_SDIST_SIDECAR_PREFIX}/"
    sidecar_contents = {
        name.removeprefix(prefix): content
        for name, content in contents_by_relative_name.items()
        if name.startswith(prefix)
    }
    return _validate_sidecar(
        sidecar_contents,
        archive=archive,
        package_version=package_version,
    )


def validate_wheel(archive: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(archive) as wheel:
        members = wheel.infolist()
        if not members:
            _fail(f"{archive}: wheel is empty")
        links = [
            member.filename
            for member in members
            if member.create_system == 3 and stat.S_ISLNK((member.external_attr >> 16) & 0xFFFF)
        ]
        if links:
            _fail(f"{archive}: wheel must not contain links: {', '.join(sorted(links))}")
        paths = [_validate_safe_path(member.filename, archive=archive) for member in members]
        _validate_unique_paths(paths, archive=archive)
        name_set = {str(path) for path in paths}
        file_contents: dict[str, bytes] = {}
        for member in members:
            if member.is_dir():
                continue
            content = wheel.read(member)
            file_contents[member.filename] = content
            _validate_publication_contents(
                content,
                archive=archive,
                member_name=member.filename,
            )

    missing = sorted(_WHEEL_REQUIRED - name_set)
    if missing:
        _fail(f"{archive}: missing required wheel files: {', '.join(missing)}")
    _validate_third_party_licenses(
        file_contents["cayu/server/dashboard/THIRD_PARTY_LICENSES.md"], archive=archive
    )
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

    metadata_name = f"{metadata_root}/METADATA"
    package_version = _metadata_version(
        file_contents[metadata_name], archive=archive, member_name=metadata_name
    )
    prefix = f"{_WHEEL_SIDECAR_PREFIX}/"
    sidecar_contents = {
        name.removeprefix(prefix): content
        for name, content in file_contents.items()
        if name.startswith(prefix)
    }
    return _validate_sidecar(
        sidecar_contents,
        archive=archive,
        package_version=package_version,
    )


def validate_sidecar_equivalence(
    sdist: Path,
    wheel: Path,
    *,
    sdist_contents: dict[str, bytes] | None = None,
    wheel_contents: dict[str, bytes] | None = None,
) -> None:
    source_contents = sdist_contents if sdist_contents is not None else validate_sdist(sdist)
    package_contents = wheel_contents if wheel_contents is not None else validate_wheel(wheel)
    if source_contents.keys() != package_contents.keys():
        _fail(f"{wheel}: packaged sidecar inventory differs from the sdist source")
    mismatched = sorted(
        name for name in source_contents if source_contents[name] != package_contents[name]
    )
    if mismatched:
        _fail(f"{wheel}: packaged sidecar differs from the sdist source: {', '.join(mismatched)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args(argv)
    sdists = [path for path in args.archives if path.name.endswith(".tar.gz")]
    wheels = [path for path in args.archives if path.suffix == ".whl"]
    if len(sdists) != 1 or len(wheels) != 1:
        parser.error("pass exactly one .tar.gz source distribution and one .whl wheel")
    sdist_contents = validate_sdist(sdists[0])
    wheel_contents = validate_wheel(wheels[0])
    validate_sidecar_equivalence(
        sdists[0],
        wheels[0],
        sdist_contents=sdist_contents,
        wheel_contents=wheel_contents,
    )
    print(f"validated {sdists[0]} and {wheels[0]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
