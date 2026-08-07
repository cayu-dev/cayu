"""Editable dashboard source commands."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import re
import stat
import sys
import uuid
import zipfile
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast

from cayu._server_contract_version import SERVER_CONTRACT_VERSION

_BUNDLE_DIRECTORY = "dashboard_source"
_BUNDLE_NAME_PATTERN = re.compile(r"cayu-dashboard-source-(?P<version>[^/]+)\.zip\Z")
_MANIFEST_NAME = "cayu-dashboard-source.json"
_MANIFEST_KEYS = {
    "artifact_version",
    "cayu_version",
    "compiled_dashboard_digest",
    "files",
    "generated_api_digest",
    "schema_version",
    "server_contract_version",
    "source_digest",
}
_FILE_KEYS = {"path", "sha256", "size"}
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_WINDOWS_REPARSE_TAG_NAME_SURROGATE = 0x20000000
_WINDOWS_INVALID_FILENAME_CHARACTERS = frozenset('<>:"|?*')
_WINDOWS_PRIVATE_DIRECTORY_SDDL = "D:P(A;OICI;FA;;;OW)(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)"
_WINDOWS_RESERVED_FILENAME_STEMS = frozenset(
    {
        "aux",
        "clock$",
        "con",
        "conin$",
        "conout$",
        "nul",
        "prn",
        *(f"com{suffix}" for suffix in "123456789¹²³"),
        *(f"lpt{suffix}" for suffix in "123456789¹²³"),
    }
)
_MAX_ARCHIVE_FILES = 4096
_MAX_ARCHIVE_FILE_BYTES = 16 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
_REQUIRED_SOURCE_FILES = {
    "LICENSE",
    "NOTICE",
    "README.md",
    "REDISTRIBUTION.md",
    "THIRD_PARTY_LICENSES.md",
    "biome.json",
    "package-lock.json",
    "package.json",
    "scripts/check-third-party-licenses.mjs",
    "scripts/finalize-third-party-licenses.mjs",
    "scripts/generate-api-types.py",
    "scripts/run-python.mjs",
    "server-openapi.json",
    "src/lib/generated/server-api/index.ts",
    "src/lib/generated/server-api/types.gen.ts",
    "src/lib/release-metadata.ts",
    "src/main.tsx",
    "tests/dashboard-capabilities.test.mjs",
    "tests/run-python.test.mjs",
    "third_party/shadcn-ui.LICENSE",
    "tsconfig.json",
    "vite.config.ts",
}


class DashboardSourceError(RuntimeError):
    """The packaged dashboard source is invalid or cannot be extracted safely."""


@dataclass(frozen=True)
class _DestinationParentGuard:
    path: Path
    identity: os.stat_result

    @classmethod
    def capture(cls, path: Path) -> _DestinationParentGuard:
        _reject_link_components(path)
        try:
            identity = path.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise DashboardSourceError(f"destination parent must already exist: {path}") from exc
        except OSError as exc:
            raise DashboardSourceError(
                f"could not inspect destination parent {path}: {exc}"
            ) from exc
        if not stat.S_ISDIR(identity.st_mode):
            raise DashboardSourceError(f"destination parent must be a directory: {path}")
        return cls(path=path, identity=identity)

    def assert_unchanged(self) -> None:
        try:
            _reject_link_components(self.path)
            current = self.path.stat(follow_symlinks=False)
        except (DashboardSourceError, OSError) as exc:
            raise DashboardSourceError(
                f"destination parent changed during extraction: {self.path}"
            ) from exc
        if not stat.S_ISDIR(current.st_mode) or not os.path.samestat(self.identity, current):
            raise DashboardSourceError(f"destination parent changed during extraction: {self.path}")


@dataclass(frozen=True)
class _StagingGuard:
    path: Path
    identity: os.stat_result

    @classmethod
    def capture(cls, path: Path) -> _StagingGuard:
        try:
            _reject_link_components(path)
            identity = path.stat(follow_symlinks=False)
        except (DashboardSourceError, OSError) as exc:
            raise DashboardSourceError(
                f"could not capture staging directory ownership: {path}"
            ) from exc
        if not stat.S_ISDIR(identity.st_mode):
            raise DashboardSourceError(f"staging path must be a directory: {path}")
        return cls(path=path, identity=identity)

    def assert_unchanged(self, path: Path | None = None) -> None:
        candidate = self.path if path is None else path
        try:
            _reject_link_components(candidate)
            current = candidate.stat(follow_symlinks=False)
        except (DashboardSourceError, OSError) as exc:
            raise DashboardSourceError(
                f"staging directory changed during extraction: {candidate}"
            ) from exc
        if not stat.S_ISDIR(current.st_mode) or not os.path.samestat(self.identity, current):
            raise DashboardSourceError(f"staging directory changed during extraction: {candidate}")


@dataclass(frozen=True)
class DashboardSourceFile:
    path: str
    size: int
    sha256: str

    def as_manifest_value(self) -> dict[str, str | int]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True)
class DashboardSourceManifest:
    schema_version: int
    artifact_version: int
    cayu_version: str
    server_contract_version: str
    source_digest: str
    generated_api_digest: str
    compiled_dashboard_digest: str
    files: tuple[DashboardSourceFile, ...]


@dataclass(frozen=True)
class ValidatedDashboardSource:
    manifest: DashboardSourceManifest
    contents: dict[str, bytes]


@dataclass(frozen=True)
class DashboardEjectResult:
    destination: Path
    manifest: DashboardSourceManifest


@dataclass(frozen=True)
class _BundleResource:
    resource: Traversable
    cayu_version: str


def add_dashboard_parser(subparsers: Any) -> None:
    """Register the ``dashboard eject`` command group."""
    dashboard = subparsers.add_parser(
        "dashboard",
        help="Manage the bundled Cayu control-plane source.",
        description=(
            "Manage the bundled Cayu control-plane source. "
            "Use `cayu dashboard eject DESTINATION` to create an editable project."
        ),
    )
    dashboard_commands = dashboard.add_subparsers(dest="dashboard_command", required=True)
    eject = dashboard_commands.add_parser(
        "eject",
        help="Extract the version-matched dashboard source.",
        description=(
            "Extract the version-matched dashboard source into an empty destination, "
            "then install its Node dependencies and customize it locally."
        ),
    )
    eject.add_argument("destination", type=Path, metavar="DESTINATION")


def run_dashboard(args: argparse.Namespace) -> int:
    """Dispatch a parsed ``dashboard`` invocation."""
    try:
        result = eject_dashboard_source(args.destination)
    except Exception as exc:
        _print_cli_error(exc)
        return 1

    manifest = result.manifest
    print(
        f"ejected editable Cayu dashboard source to {result.destination}\n"
        f"installed Cayu version: {manifest.cayu_version}\n"
        f"dashboard source version: {manifest.cayu_version}\n"
        f"dashboard server contract: v{manifest.server_contract_version}\n\n"
        "Next steps:\n"
        f"  Project directory: {result.destination}\n"
        "  In that directory, run:\n"
        "    npm ci\n"
        "    npm run dev\n"
        "    npm run build\n\n"
        "Serve the production build with one of:\n"
        '  DashboardConfig(directory=Path("dist"))\n'
        '  mount_cayu(app, cayu_app, dashboard_dir=Path("dist"), ...)\n'
        '  mount_dashboard(app, dashboard_dir=Path("dist"), ...)'
    )
    return 0


def eject_dashboard_source(
    destination: Path,
    *,
    bundle_bytes: bytes | None = None,
    expected_cayu_version: str | None = None,
    expected_server_contract_version: str = SERVER_CONTRACT_VERSION,
) -> DashboardEjectResult:
    """Validate and atomically materialize the packaged editable dashboard source."""
    using_packaged_bundle = bundle_bytes is None
    if bundle_bytes is None:
        resource = _dashboard_source_resource()
        try:
            bundle_bytes = resource.resource.read_bytes()
        except OSError as exc:
            raise DashboardSourceError(
                f"could not read the dashboard source bundle: {exc}"
            ) from exc
        expected_cayu_version = resource.cayu_version
    elif expected_cayu_version is None:
        from cayu.cli import _version

        expected_cayu_version = _version()

    artifact = validate_dashboard_source_bundle(
        bundle_bytes,
        expected_cayu_version=expected_cayu_version,
        expected_server_contract_version=expected_server_contract_version,
    )
    if using_packaged_bundle:
        compiled_digest = _packaged_compiled_dashboard_digest()
        if compiled_digest != artifact.manifest.compiled_dashboard_digest:
            raise DashboardSourceError(
                "packaged dashboard source does not match the compiled dashboard: "
                f"expected {artifact.manifest.compiled_dashboard_digest}, "
                f"found {compiled_digest}"
            )
    destination = _validate_destination(destination)
    _reject_link_components(destination.parent)
    parent_guard = _DestinationParentGuard.capture(destination.parent)
    staging, staging_guard = _create_staging_directory(
        destination,
        parent_guard=parent_guard,
    )
    try:
        parent_guard.assert_unchanged()
        _write_staging_tree(
            staging,
            artifact.contents,
            staging_guard=staging_guard,
        )
        staging_guard.assert_unchanged()
        _publish_staged_tree(
            staging,
            destination,
            parent_guard=parent_guard,
            staging_guard=staging_guard,
        )
    except BaseException as exc:
        try:
            parent_guard.assert_unchanged()
        except DashboardSourceError as cleanup_error:
            exc.add_note(f"could not safely remove staging directory {staging}: {cleanup_error}")
        else:
            _remove_staging_tree_if_owned(staging_guard, error=exc)
        raise
    return DashboardEjectResult(destination=destination, manifest=artifact.manifest)


def validate_dashboard_source_bundle(
    bundle_bytes: bytes,
    *,
    expected_cayu_version: str,
    expected_server_contract_version: str,
) -> ValidatedDashboardSource:
    """Validate one complete dashboard source ZIP without extracting it."""
    if len(bundle_bytes) > _MAX_ARCHIVE_BYTES:
        raise DashboardSourceError("dashboard source bundle exceeds its size limit")
    try:
        archive = zipfile.ZipFile(io.BytesIO(bundle_bytes))
    except (OSError, zipfile.BadZipFile) as exc:
        raise DashboardSourceError("dashboard source bundle is not a valid ZIP archive") from exc

    with archive:
        members = archive.infolist()
        if not members:
            raise DashboardSourceError("dashboard source bundle is empty")
        if len(members) > _MAX_ARCHIVE_FILES:
            raise DashboardSourceError("dashboard source bundle contains too many files")

        normalized_paths: list[str] = []
        seen_paths: set[str] = set()
        seen_casefolded: set[str] = set()
        total_size = 0
        for member in members:
            path = _validate_archive_path(member.filename)
            casefolded = path.casefold()
            if path in seen_paths or casefolded in seen_casefolded:
                raise DashboardSourceError(f"duplicate dashboard source archive path: {path}")
            seen_paths.add(path)
            seen_casefolded.add(casefolded)
            normalized_paths.append(path)
            _validate_archive_member_type(member, path=path)
            if member.file_size > _MAX_ARCHIVE_FILE_BYTES:
                raise DashboardSourceError(f"dashboard source archive file is too large: {path}")
            total_size += member.file_size
            if total_size > _MAX_ARCHIVE_BYTES:
                raise DashboardSourceError("dashboard source bundle expands beyond its size limit")

        _validate_path_topology(normalized_paths, label="archive")

        contents: dict[str, bytes] = {}
        for member, path in zip(members, normalized_paths, strict=True):
            try:
                content = archive.read(member)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise DashboardSourceError(
                    f"could not read dashboard source archive file: {path}"
                ) from exc
            if len(content) != member.file_size:
                raise DashboardSourceError(f"dashboard source archive size mismatch: {path}")
            contents[path] = content

    raw_manifest = contents.pop(_MANIFEST_NAME, None)
    if raw_manifest is None:
        raise DashboardSourceError(f"dashboard source manifest is missing: {_MANIFEST_NAME}")
    manifest = _parse_manifest(raw_manifest)
    if manifest.cayu_version != expected_cayu_version:
        raise DashboardSourceError(
            "dashboard source Cayu version mismatch: "
            f"expected {expected_cayu_version}, found {manifest.cayu_version}"
        )
    if manifest.server_contract_version != expected_server_contract_version:
        raise DashboardSourceError(
            "dashboard source server contract mismatch: "
            f"expected {expected_server_contract_version}, "
            f"found {manifest.server_contract_version}"
        )

    expected_paths = {item.path for item in manifest.files}
    actual_paths = set(contents)
    missing = sorted(expected_paths - actual_paths)
    unexpected = sorted(actual_paths - expected_paths)
    if missing:
        raise DashboardSourceError(
            f"dashboard source bundle is missing manifest files: {', '.join(missing)}"
        )
    if unexpected:
        raise DashboardSourceError(
            f"dashboard source bundle contains unexpected files: {', '.join(unexpected)}"
        )
    required_missing = sorted(_REQUIRED_SOURCE_FILES - actual_paths)
    if required_missing:
        raise DashboardSourceError(
            f"dashboard source bundle is incomplete: {', '.join(required_missing)}"
        )
    for item in manifest.files:
        content = contents[item.path]
        if len(content) != item.size:
            raise DashboardSourceError(
                f"dashboard source size mismatch for {item.path}: "
                f"expected {item.size}, found {len(content)}"
            )
        digest = _sha256(content)
        if digest != item.sha256:
            raise DashboardSourceError(
                f"dashboard source digest mismatch for {item.path}: "
                f"expected {item.sha256}, found {digest}"
            )

    source_digest = _contents_digest(contents)
    if source_digest != manifest.source_digest:
        raise DashboardSourceError(
            "dashboard source aggregate digest mismatch: "
            f"expected {manifest.source_digest}, found {source_digest}"
        )
    generated_api = _generated_api_contents(contents)
    generated_api_digest = _contents_digest(generated_api)
    if generated_api_digest != manifest.generated_api_digest:
        raise DashboardSourceError(
            "dashboard generated API digest mismatch: "
            f"expected {manifest.generated_api_digest}, found {generated_api_digest}"
        )
    _validate_release_metadata(contents, manifest=manifest)
    contents[_MANIFEST_NAME] = raw_manifest
    return ValidatedDashboardSource(manifest=manifest, contents=contents)


def render_dashboard_source_manifest(
    contents: dict[str, bytes],
    *,
    cayu_version: str,
    server_contract_version: str,
    compiled_dashboard_digest: str,
) -> bytes:
    """Render canonical metadata for a deterministic dashboard source bundle."""
    source_files = [
        DashboardSourceFile(path=path, size=len(contents[path]), sha256=_sha256(contents[path]))
        for path in sorted(contents)
    ]
    generated_api = _generated_api_contents(contents)
    manifest = {
        "artifact_version": 1,
        "cayu_version": cayu_version,
        "compiled_dashboard_digest": compiled_dashboard_digest,
        "files": [item.as_manifest_value() for item in source_files],
        "generated_api_digest": _contents_digest(generated_api),
        "schema_version": 1,
        "server_contract_version": server_contract_version,
        "source_digest": _contents_digest(contents),
    }
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()


def contents_digest(contents: dict[str, bytes]) -> str:
    """Return the canonical digest for a relative-path byte tree."""
    return _contents_digest(contents)


def _generated_api_contents(contents: dict[str, bytes]) -> dict[str, bytes]:
    return {
        path: content
        for path, content in contents.items()
        if path == "server-openapi.json" or path.startswith("src/lib/generated/server-api/")
    }


def _dashboard_source_resource() -> _BundleResource:
    root = files("cayu.data").joinpath(_BUNDLE_DIRECTORY)
    if _is_symlink(root) or not root.is_dir():
        raise DashboardSourceError("the installed Cayu distribution omits dashboard source")
    try:
        candidates = sorted(
            (
                child
                for child in root.iterdir()
                if child.is_file() and _BUNDLE_NAME_PATTERN.fullmatch(child.name) is not None
            ),
            key=lambda child: child.name,
        )
    except OSError as exc:
        raise DashboardSourceError(f"could not enumerate packaged dashboard source: {exc}") from exc
    if len(candidates) != 1:
        raise DashboardSourceError(
            "the installed Cayu distribution must contain exactly one dashboard source bundle"
        )
    if _is_symlink(candidates[0]):
        raise DashboardSourceError("packaged dashboard source must not contain symbolic links")
    try:
        installed_version = importlib.metadata.version("cayu")
    except importlib.metadata.PackageNotFoundError as exc:
        raise DashboardSourceError(
            "the packaged dashboard source cannot be verified without Cayu distribution metadata"
        ) from exc
    expected_name = f"cayu-dashboard-source-{installed_version}.zip"
    if candidates[0].name != expected_name:
        raise DashboardSourceError(
            f"dashboard source bundle filename mismatch: expected {expected_name}, "
            f"found {candidates[0].name}"
        )
    return _BundleResource(resource=candidates[0], cayu_version=installed_version)


def _parse_manifest(raw: bytes) -> DashboardSourceManifest:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DashboardSourceError("dashboard source manifest must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise DashboardSourceError("dashboard source manifest must be a JSON object")
    if set(value) != _MANIFEST_KEYS:
        raise DashboardSourceError("dashboard source manifest fields are invalid")
    schema_version = _require_positive_int(value, "schema_version")
    artifact_version = _require_positive_int(value, "artifact_version")
    if schema_version != 1:
        raise DashboardSourceError(
            f"unsupported dashboard source manifest schema: {schema_version}"
        )
    if artifact_version != 1:
        raise DashboardSourceError(f"unsupported dashboard source artifact: {artifact_version}")
    cayu_version = _require_nonblank_string(value, "cayu_version")
    server_contract_version = _require_nonblank_string(value, "server_contract_version")
    source_digest = _require_sha256(value, "source_digest")
    generated_api_digest = _require_sha256(value, "generated_api_digest")
    compiled_dashboard_digest = _require_sha256(value, "compiled_dashboard_digest")

    raw_files = value.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise DashboardSourceError("dashboard source manifest files must be a non-empty array")
    parsed_files: list[DashboardSourceFile] = []
    seen_paths: set[str] = set()
    seen_casefolded: set[str] = set()
    for index, raw_file in enumerate(raw_files):
        if not isinstance(raw_file, dict) or set(raw_file) != _FILE_KEYS:
            raise DashboardSourceError(f"dashboard source manifest file entry {index} is invalid")
        file_value = cast("dict[str, Any]", raw_file)
        path = _validate_manifest_path(file_value.get("path"), index=index)
        if path in seen_paths or path.casefold() in seen_casefolded:
            raise DashboardSourceError(f"duplicate dashboard source manifest path: {path}")
        seen_paths.add(path)
        seen_casefolded.add(path.casefold())
        size = file_value.get("size")
        if type(size) is not int or size < 0 or size > _MAX_ARCHIVE_FILE_BYTES:
            raise DashboardSourceError(f"invalid dashboard source file size: {path}")
        digest = file_value.get("sha256")
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise DashboardSourceError(f"invalid dashboard source file digest: {path}")
        parsed_files.append(DashboardSourceFile(path=path, size=size, sha256=digest))
    paths = [item.path for item in parsed_files]
    _validate_path_topology(paths, label="manifest")
    if paths != sorted(paths):
        raise DashboardSourceError("dashboard source manifest files must be sorted by path")
    return DashboardSourceManifest(
        schema_version=schema_version,
        artifact_version=artifact_version,
        cayu_version=cayu_version,
        server_contract_version=server_contract_version,
        source_digest=source_digest,
        generated_api_digest=generated_api_digest,
        compiled_dashboard_digest=compiled_dashboard_digest,
        files=tuple(parsed_files),
    )


def _validate_archive_path(value: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        raise DashboardSourceError(f"unsafe dashboard source archive path: {value!r}")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise DashboardSourceError("dashboard source archive paths must be valid UTF-8") from exc
    path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or windows_path.drive
        or windows_path.root
        or ":" in value
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(_is_unsafe_windows_archive_component(part) for part in path.parts)
    ):
        raise DashboardSourceError(f"unsafe dashboard source archive path: {value!r}")
    return value


def _validate_path_topology(paths: list[str], *, label: str) -> None:
    by_components = {
        tuple(part.casefold() for part in PurePosixPath(path).parts): path for path in paths
    }
    for components, path in by_components.items():
        for depth in range(1, len(components)):
            ancestor = by_components.get(components[:depth])
            if ancestor is not None:
                raise DashboardSourceError(
                    f"conflicting dashboard source {label} paths: {ancestor!r} and {path!r}"
                )


def _is_unsafe_windows_archive_component(component: str) -> bool:
    if component.endswith((" ", ".")) or component.strip(" ") in {".", ".."}:
        return True
    if any(
        ord(character) < 32 or character in _WINDOWS_INVALID_FILENAME_CHARACTERS
        for character in component
    ):
        return True
    stem = component.partition(".")[0].rstrip(" ").casefold()
    return stem in _WINDOWS_RESERVED_FILENAME_STEMS


def _validate_manifest_path(value: Any, *, index: int) -> str:
    if not isinstance(value, str):
        raise DashboardSourceError(f"invalid dashboard source manifest path at index {index}")
    path = _validate_archive_path(value)
    if path == _MANIFEST_NAME:
        raise DashboardSourceError("dashboard source manifest must not list itself")
    return path


def _validate_archive_member_type(member: zipfile.ZipInfo, *, path: str) -> None:
    if member.flag_bits & 0x1:
        raise DashboardSourceError(f"encrypted dashboard source archive entry: {path}")
    if member.compress_type != zipfile.ZIP_STORED:
        raise DashboardSourceError(f"unsupported dashboard source compression: {path}")
    mode = (member.external_attr >> 16) & 0xFFFF
    if member.create_system != 3 or not stat.S_ISREG(mode):
        raise DashboardSourceError(f"unsupported dashboard source archive entry type: {path}")


def _validate_release_metadata(
    contents: dict[str, bytes],
    *,
    manifest: DashboardSourceManifest,
) -> None:
    path = "src/lib/release-metadata.ts"
    content = contents.get(path)
    if content is None:
        raise DashboardSourceError(f"dashboard release metadata is missing: {path}")
    expected = (
        f'export const DASHBOARD_SOURCE_CAYU_VERSION = "{manifest.cayu_version}"\n'
        f'export const SUPPORTED_SERVER_CONTRACT_VERSION = "{manifest.server_contract_version}"\n'
    ).encode()
    if content != expected:
        raise DashboardSourceError("dashboard release metadata does not match its bundle manifest")


def _require_positive_int(value: dict[str, Any], field: str) -> int:
    candidate = value.get(field)
    if type(candidate) is not int or candidate < 1:
        raise DashboardSourceError(f"dashboard source manifest {field} must be positive")
    return candidate


def _require_nonblank_string(value: dict[str, Any], field: str) -> str:
    candidate = value.get(field)
    if not isinstance(candidate, str) or not candidate.strip() or candidate != candidate.strip():
        raise DashboardSourceError(f"dashboard source manifest {field} must be non-blank")
    return candidate


def _require_sha256(value: dict[str, Any], field: str) -> str:
    candidate = value.get(field)
    if not isinstance(candidate, str) or _SHA256_PATTERN.fullmatch(candidate) is None:
        raise DashboardSourceError(f"dashboard source manifest {field} must be a SHA-256 digest")
    return candidate


def _contents_digest(contents: dict[str, bytes]) -> str:
    files = [
        DashboardSourceFile(path=path, size=len(contents[path]), sha256=_sha256(contents[path]))
        for path in sorted(contents)
    ]
    canonical = [item.as_manifest_value() for item in files]
    return _sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def _sha256(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _validate_destination(destination: Path) -> Path:
    try:
        os.fspath(destination).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise DashboardSourceError("destination path must be valid UTF-8") from exc
    try:
        destination = Path(os.path.normpath(destination.expanduser().absolute()))
    except (OSError, RuntimeError) as exc:
        raise DashboardSourceError(f"could not resolve destination {destination}: {exc}") from exc
    _reject_link_components(destination)
    if destination.parent == destination:
        raise DashboardSourceError("destination must not be a filesystem root")
    current_directory = Path.cwd().resolve()
    resolved = destination.resolve(strict=False)
    if resolved == current_directory or resolved in current_directory.parents:
        raise DashboardSourceError(
            "destination must not be the current working directory or one of its ancestors"
        )
    home_directory = Path.home().resolve()
    if resolved == home_directory or resolved in home_directory.parents:
        raise DashboardSourceError(
            "destination must not be the home directory or one of its ancestors"
        )
    if destination.exists() and not destination.is_dir():
        raise DashboardSourceError(f"destination must be a directory: {destination}")
    if destination.is_dir() and next(destination.iterdir(), None) is not None:
        raise DashboardSourceError(f"destination must be empty: {destination}")
    return destination


def _packaged_compiled_dashboard_digest() -> str:
    root = files("cayu").joinpath("server", "dashboard")
    if _is_symlink(root) or not root.is_dir():
        raise DashboardSourceError(
            "the installed Cayu distribution omits compiled dashboard assets"
        )
    return _contents_digest(_collect_resource_files(root, label="compiled dashboard"))


def _collect_resource_files(root: Traversable, *, label: str) -> dict[str, bytes]:
    contents: dict[str, bytes] = {}

    def visit(directory: Traversable, prefix: PurePosixPath) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda child: child.name)
        except OSError as exc:
            raise DashboardSourceError(f"could not enumerate packaged {label}: {exc}") from exc
        for child in children:
            relative = prefix / child.name
            path = relative.as_posix()
            if _is_symlink(child):
                raise DashboardSourceError(f"packaged {label} must not contain links: {path}")
            if child.is_dir():
                visit(child, relative)
            elif child.is_file():
                try:
                    contents[path] = child.read_bytes()
                except OSError as exc:
                    raise DashboardSourceError(f"could not read packaged {label}: {path}") from exc
            else:
                raise DashboardSourceError(f"packaged {label} contains unsupported entry: {path}")

    visit(root, PurePosixPath())
    return contents


def _is_symlink(resource: Traversable) -> bool:
    return isinstance(resource, Path) and resource.is_symlink()


def _reject_link_components(path: Path) -> None:
    for component in reversed((path, *path.parents)):
        try:
            identity = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise DashboardSourceError(
                f"could not inspect destination path component {component}: {exc}"
            ) from exc
        if stat.S_ISLNK(identity.st_mode) or _is_windows_name_surrogate(identity):
            raise DashboardSourceError(
                f"destination must not traverse a symbolic link or junction: {component}"
            )


def _is_windows_name_surrogate(value: os.stat_result) -> bool:
    file_attributes = getattr(value, "st_file_attributes", 0)
    reparse_tag = getattr(value, "st_reparse_tag", None)
    if reparse_tag is not None and reparse_tag & _WINDOWS_REPARSE_TAG_NAME_SURROGATE:
        return True
    return bool(file_attributes & _WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT) and not reparse_tag


def _create_staging_directory(
    destination: Path,
    *,
    parent_guard: _DestinationParentGuard,
) -> tuple[Path, _StagingGuard]:
    if os.name != "nt":
        return _create_private_posix_staging_directory(
            destination,
            parent_guard=parent_guard,
        )
    with _windows_directory_namespace_fence(parent_guard.path):
        parent_guard.assert_unchanged()
        staging_guard = _create_private_windows_staging_directory(destination)
        staging = staging_guard.path
        try:
            staging_guard.assert_unchanged()
            parent_guard.assert_unchanged()
        except BaseException as exc:
            _remove_new_empty_staging_after_creation_failure(staging_guard, error=exc)
            raise
    return staging, staging_guard


def _create_private_posix_staging_directory(
    destination: Path,
    *,
    parent_guard: _DestinationParentGuard,
) -> tuple[Path, _StagingGuard]:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        parent_descriptor = os.open(parent_guard.path, directory_flags)
    except OSError as exc:
        raise DashboardSourceError(
            f"destination parent changed during extraction: {parent_guard.path}"
        ) from exc

    created_name: str | None = None
    created_identity: os.stat_result | None = None
    creation_error: BaseException | None = None
    try:
        opened_parent_identity = os.fstat(parent_descriptor)
        if not stat.S_ISDIR(opened_parent_identity.st_mode) or not os.path.samestat(
            parent_guard.identity, opened_parent_identity
        ):
            raise DashboardSourceError(
                f"destination parent changed during extraction: {parent_guard.path}"
            )
        for _attempt in range(100):
            candidate_name = f".{destination.name}.cayu-dashboard-{uuid.uuid4().hex}"
            try:
                os.mkdir(candidate_name, mode=0o700, dir_fd=parent_descriptor)
            except FileExistsError:
                continue
            created_name = candidate_name
            break
        if created_name is None:
            raise DashboardSourceError("could not allocate a private staging directory")

        created_identity = os.stat(
            created_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        child_descriptor = os.open(
            created_name,
            directory_flags,
            dir_fd=parent_descriptor,
        )
        child_error: BaseException | None = None
        try:
            opened_child_identity = os.fstat(child_descriptor)
            if (
                not stat.S_ISDIR(created_identity.st_mode)
                or not stat.S_ISDIR(opened_child_identity.st_mode)
                or not os.path.samestat(created_identity, opened_child_identity)
            ):
                raise DashboardSourceError(
                    f"staging directory changed during extraction: "
                    f"{parent_guard.path / created_name}"
                )
        except BaseException as exc:
            child_error = exc
            raise
        finally:
            _close_descriptor(child_descriptor, error=child_error)

        staging = parent_guard.path / created_name
        staging_guard = _StagingGuard(path=staging, identity=created_identity)
        parent_guard.assert_unchanged()
        staging_guard.assert_unchanged()
        return staging, staging_guard
    except BaseException as exc:
        creation_error = exc
        if created_name is not None:
            _remove_new_empty_posix_staging_after_creation_failure(
                parent_descriptor,
                created_name,
                identity=created_identity,
                error=exc,
            )
        raise
    finally:
        _close_descriptor(parent_descriptor, error=creation_error)


def _remove_new_empty_posix_staging_after_creation_failure(
    parent_descriptor: int,
    name: str,
    *,
    identity: os.stat_result | None,
    error: BaseException,
) -> None:
    if identity is None:
        error.add_note(
            f"could not safely remove newly created staging directory {name}: "
            "ownership was not captured"
        )
        return
    try:
        current = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(current.st_mode) or not os.path.samestat(identity, current):
            raise DashboardSourceError("staging directory ownership changed")
        os.rmdir(name, dir_fd=parent_descriptor)
    except (DashboardSourceError, OSError) as cleanup_error:
        error.add_note(
            f"could not safely remove newly created staging directory {name}: {cleanup_error}"
        )


def _create_private_windows_staging_directory(destination: Path) -> _StagingGuard:
    import ctypes
    from ctypes import wintypes

    class _SecurityAttributes(ctypes.Structure):
        _fields_ = (
            ("nLength", wintypes.DWORD),
            ("lpSecurityDescriptor", wintypes.LPVOID),
            ("bInheritHandle", wintypes.BOOL),
        )

    windows_ctypes: Any = ctypes
    advapi32 = windows_ctypes.WinDLL("advapi32", use_last_error=True)
    convert_security_descriptor = advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert_security_descriptor.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.DWORD),
    )
    convert_security_descriptor.restype = wintypes.BOOL
    kernel32 = windows_ctypes.WinDLL("kernel32", use_last_error=True)
    create_directory = kernel32.CreateDirectoryW
    create_directory.argtypes = (
        wintypes.LPCWSTR,
        ctypes.POINTER(_SecurityAttributes),
    )
    create_directory.restype = wintypes.BOOL
    local_free = kernel32.LocalFree
    local_free.argtypes = (wintypes.LPVOID,)
    local_free.restype = wintypes.LPVOID

    security_descriptor = wintypes.LPVOID()
    if not convert_security_descriptor(
        _WINDOWS_PRIVATE_DIRECTORY_SDDL,
        1,
        ctypes.byref(security_descriptor),
        None,
    ):
        error_code = windows_ctypes.get_last_error()
        raise OSError(
            error_code,
            "could not construct a private staging-directory DACL: "
            f"{windows_ctypes.FormatError(error_code)}",
        )
    attributes = _SecurityAttributes(
        ctypes.sizeof(_SecurityAttributes),
        security_descriptor,
        False,
    )
    created: Path | None = None
    created_guard: _StagingGuard | None = None
    creation_error: BaseException | None = None
    try:
        for _attempt in range(100):
            candidate = destination.parent / (
                f".{destination.name}.cayu-dashboard-{uuid.uuid4().hex}"
            )
            if create_directory(str(candidate), ctypes.byref(attributes)):
                created = candidate
                break
            error_code = windows_ctypes.get_last_error()
            if error_code not in {80, 183}:
                raise OSError(
                    error_code,
                    f"could not create private staging directory {candidate}: "
                    f"{windows_ctypes.FormatError(error_code)}",
                )
        if created is None:
            raise DashboardSourceError("could not allocate a private staging directory")
        created_guard = _StagingGuard.capture(created)
    except BaseException as exc:
        creation_error = exc
        if created is not None:
            if created_guard is None:
                exc.add_note(
                    f"could not safely remove newly created staging directory {created}: "
                    "ownership was not captured"
                )
            else:
                _remove_new_empty_staging_after_creation_failure(created_guard, error=exc)
        raise
    finally:
        if local_free(security_descriptor):
            free_error = OSError("could not release the staging-directory security descriptor")
            if creation_error is not None:
                creation_error.add_note(str(free_error))
            else:
                if created_guard is not None:
                    _remove_new_empty_staging_after_creation_failure(
                        created_guard,
                        error=free_error,
                    )
                raise free_error

    if created_guard is None:
        raise DashboardSourceError("could not capture staging directory ownership")
    try:
        _assert_windows_directory_dacl_is_protected(created_guard.path)
    except BaseException as exc:
        _remove_new_empty_staging_after_creation_failure(created_guard, error=exc)
        raise
    return created_guard


def _assert_windows_directory_dacl_is_protected(path: Path) -> None:
    dacl_present, dacl_protected = _windows_directory_dacl_state(path)
    if not dacl_present or not dacl_protected:
        raise DashboardSourceError(
            f"staging directory does not have a protected private DACL: {path}"
        )


def _assert_windows_directory_dacl_is_inherited(path: Path) -> None:
    dacl_present, dacl_protected = _windows_directory_dacl_state(path)
    if not dacl_present or dacl_protected:
        raise DashboardSourceError(
            f"published dashboard directory did not inherit parent permissions: {path}"
        )


def _windows_directory_dacl_state(path: Path) -> tuple[bool, bool]:
    import ctypes
    from ctypes import wintypes

    windows_ctypes: Any = ctypes
    advapi32 = windows_ctypes.WinDLL("advapi32", use_last_error=True)
    get_named_security_info = advapi32.GetNamedSecurityInfoW
    get_named_security_info.argtypes = (
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
    )
    get_named_security_info.restype = wintypes.DWORD
    get_security_descriptor_control = advapi32.GetSecurityDescriptorControl
    get_security_descriptor_control.argtypes = (
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.WORD),
        ctypes.POINTER(wintypes.DWORD),
    )
    get_security_descriptor_control.restype = wintypes.BOOL
    kernel32 = windows_ctypes.WinDLL("kernel32", use_last_error=True)
    local_free = kernel32.LocalFree
    local_free.argtypes = (wintypes.LPVOID,)
    local_free.restype = wintypes.LPVOID

    dacl = wintypes.LPVOID()
    security_descriptor = wintypes.LPVOID()
    error_code = get_named_security_info(
        ctypes.create_unicode_buffer(str(path)),
        1,
        0x4,
        None,
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(security_descriptor),
    )
    if error_code:
        raise OSError(
            error_code,
            f"could not inspect staging-directory DACL for {path}: "
            f"{windows_ctypes.FormatError(error_code)}",
        )
    inspection_error: BaseException | None = None
    try:
        control = wintypes.WORD()
        revision = wintypes.DWORD()
        if not get_security_descriptor_control(
            security_descriptor,
            ctypes.byref(control),
            ctypes.byref(revision),
        ):
            error_code = windows_ctypes.get_last_error()
            raise OSError(
                error_code,
                f"could not inspect staging-directory DACL control for {path}: "
                f"{windows_ctypes.FormatError(error_code)}",
            )
        return bool(dacl), bool(control.value & 0x1000)
    except BaseException as exc:
        inspection_error = exc
        raise
    finally:
        if local_free(security_descriptor):
            free_error = OSError(f"could not release staging-directory DACL metadata for {path}")
            if inspection_error is not None:
                inspection_error.add_note(str(free_error))
            else:
                raise free_error


def _restore_windows_directory_inheritance(path: Path) -> None:
    import ctypes
    from ctypes import wintypes

    class _Acl(ctypes.Structure):
        _fields_ = (
            ("AclRevision", wintypes.BYTE),
            ("Sbz1", wintypes.BYTE),
            ("AclSize", wintypes.WORD),
            ("AceCount", wintypes.WORD),
            ("Sbz2", wintypes.WORD),
        )

    windows_ctypes: Any = ctypes
    advapi32 = windows_ctypes.WinDLL("advapi32", use_last_error=True)
    initialize_acl = advapi32.InitializeAcl
    initialize_acl.argtypes = (
        ctypes.POINTER(_Acl),
        wintypes.DWORD,
        wintypes.DWORD,
    )
    initialize_acl.restype = wintypes.BOOL
    set_named_security_info = advapi32.SetNamedSecurityInfoW
    set_named_security_info.argtypes = (
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.LPVOID,
    )
    set_named_security_info.restype = wintypes.DWORD

    empty_dacl = _Acl()
    if not initialize_acl(ctypes.byref(empty_dacl), ctypes.sizeof(empty_dacl), 2):
        error_code = windows_ctypes.get_last_error()
        raise OSError(
            error_code,
            f"could not initialize the published dashboard DACL for {path}: "
            f"{windows_ctypes.FormatError(error_code)}",
        )
    error_code = set_named_security_info(
        ctypes.create_unicode_buffer(str(path)),
        1,
        0x4 | 0x20000000,
        None,
        None,
        ctypes.byref(empty_dacl),
        None,
    )
    if error_code:
        raise OSError(
            error_code,
            f"could not restore inherited permissions on {path}: "
            f"{windows_ctypes.FormatError(error_code)}",
        )
    _assert_windows_directory_dacl_is_inherited(path)


def _restore_published_directory_permissions(path: Path) -> None:
    if os.name == "nt":
        _restore_windows_directory_inheritance(path)


def _remove_new_empty_staging_after_creation_failure(
    staging_guard: _StagingGuard,
    *,
    error: BaseException,
) -> None:
    cleanup_path = staging_guard.path.parent / (
        f".{staging_guard.path.name}.cayu-dashboard-cleanup-{uuid.uuid4().hex}"
    )
    try:
        staging_guard.assert_unchanged()
        staging_guard.path.rename(cleanup_path)
        staging_guard.assert_unchanged(cleanup_path)
        cleanup_path.rmdir()
    except (DashboardSourceError, OSError) as cleanup_error:
        error.add_note(
            "could not safely remove newly created staging directory "
            f"{staging_guard.path}: {cleanup_error}"
        )


def _write_staging_tree(
    staging: Path,
    contents: dict[str, bytes],
    *,
    staging_guard: _StagingGuard,
) -> None:
    if os.name == "nt":
        _write_staging_tree_on_windows(
            staging,
            contents,
            staging_guard=staging_guard,
        )
        return
    _write_staging_tree_from_fd(
        staging,
        contents,
        staging_guard=staging_guard,
    )


def _write_staging_tree_from_fd(
    staging: Path,
    contents: dict[str, bytes],
    *,
    staging_guard: _StagingGuard,
) -> None:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(staging, directory_flags)
    except OSError as exc:
        raise DashboardSourceError(
            f"staging directory changed during extraction: {staging}"
        ) from exc
    write_error: BaseException | None = None
    try:
        identity = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(identity.st_mode)
            or _is_windows_name_surrogate(identity)
            or not os.path.samestat(staging_guard.identity, identity)
        ):
            raise DashboardSourceError(f"staging directory changed during extraction: {staging}")
        directory_identities: dict[tuple[str, ...], os.stat_result] = {(): identity}
        for relative, content in sorted(contents.items()):
            parts = PurePosixPath(relative).parts
            _write_staging_file_from_fd(
                descriptor,
                staging=staging,
                parts=parts,
                content=content,
                directory_flags=directory_flags,
                directory_identities=directory_identities,
            )
        os.fchmod(descriptor, 0o755)
    except BaseException as exc:
        write_error = exc
        raise
    finally:
        _close_descriptor(descriptor, error=write_error)


def _write_staging_file_from_fd(
    root_descriptor: int,
    *,
    staging: Path,
    parts: tuple[str, ...],
    content: bytes,
    directory_flags: int,
    directory_identities: dict[tuple[str, ...], os.stat_result],
) -> None:
    descriptor = os.dup(root_descriptor)
    prefix: tuple[str, ...] = ()
    write_error: BaseException | None = None
    try:
        for component in parts[:-1]:
            prefix = (*prefix, component)
            expected_identity = directory_identities.get(prefix)
            if expected_identity is None:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                except FileExistsError as exc:
                    raise DashboardSourceError(
                        "staging directory acquired an unexpected entry during extraction: "
                        f"{staging.joinpath(*prefix)}"
                    ) from exc
            child_descriptor = os.open(component, directory_flags, dir_fd=descriptor)
            child_error: BaseException | None = None
            try:
                child_identity = os.fstat(child_descriptor)
                if (
                    not stat.S_ISDIR(child_identity.st_mode)
                    or _is_windows_name_surrogate(child_identity)
                    or (
                        expected_identity is not None
                        and not os.path.samestat(expected_identity, child_identity)
                    )
                ):
                    raise DashboardSourceError(
                        f"staging directory changed during extraction: {staging.joinpath(*prefix)}"
                    )
                if expected_identity is None:
                    directory_identities[prefix] = child_identity
                os.fchmod(child_descriptor, 0o755)
            except BaseException as exc:
                child_error = exc
                raise
            finally:
                if child_error is not None:
                    _close_descriptor(child_descriptor, error=child_error)
            previous_descriptor = descriptor
            descriptor = child_descriptor
            _close_descriptor(previous_descriptor)

        file_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            file_descriptor = os.open(parts[-1], file_flags, 0o600, dir_fd=descriptor)
        except FileExistsError as exc:
            raise DashboardSourceError(
                "staging directory acquired an unexpected entry during extraction: "
                f"{staging.joinpath(*parts)}"
            ) from exc
        file_error: BaseException | None = None
        try:
            remaining = memoryview(content)
            while remaining:
                written = os.write(file_descriptor, remaining)
                if written == 0:
                    raise OSError(
                        f"could not write dashboard source file: {staging.joinpath(*parts)}"
                    )
                remaining = remaining[written:]
            os.fchmod(file_descriptor, 0o644)
        except BaseException as exc:
            file_error = exc
            raise
        finally:
            _close_descriptor(file_descriptor, error=file_error)
    except BaseException as exc:
        write_error = exc
        raise
    finally:
        _close_descriptor(descriptor, error=write_error)


def _close_descriptor(descriptor: int, *, error: BaseException | None = None) -> None:
    try:
        os.close(descriptor)
    except OSError as close_error:
        if error is None:
            raise
        error.add_note(f"could not close staging descriptor: {close_error}")


def _write_staging_tree_on_windows(
    staging: Path,
    contents: dict[str, bytes],
    *,
    staging_guard: _StagingGuard,
) -> None:
    with ExitStack() as directory_fences:
        directory_fences.enter_context(_windows_directory_namespace_fence(staging))
        staging_guard.assert_unchanged(staging)
        directory_identities: dict[tuple[str, ...], os.stat_result] = {(): staging_guard.identity}
        for relative, content in sorted(contents.items()):
            parts = PurePosixPath(relative).parts
            current = staging
            prefix: tuple[str, ...] = ()
            for component in parts[:-1]:
                prefix = (*prefix, component)
                current /= component
                expected_identity = directory_identities.get(prefix)
                if expected_identity is None:
                    try:
                        current.mkdir()
                    except FileExistsError as exc:
                        raise DashboardSourceError(
                            "staging directory acquired an unexpected entry during extraction: "
                            f"{current}"
                        ) from exc
                    directory_fences.enter_context(_windows_directory_namespace_fence(current))
                    identity = current.stat(follow_symlinks=False)
                    if not stat.S_ISDIR(identity.st_mode) or _is_windows_name_surrogate(identity):
                        raise DashboardSourceError(
                            f"staging directory changed during extraction: {current}"
                        )
                    directory_identities[prefix] = identity
                else:
                    identity = current.stat(follow_symlinks=False)
                    if (
                        not stat.S_ISDIR(identity.st_mode)
                        or _is_windows_name_surrogate(identity)
                        or not os.path.samestat(expected_identity, identity)
                    ):
                        raise DashboardSourceError(
                            f"staging directory changed during extraction: {current}"
                        )
            path = current / parts[-1]
            try:
                with path.open("xb") as output:
                    output.write(content)
            except FileExistsError as exc:
                raise DashboardSourceError(
                    f"staging directory acquired an unexpected entry during extraction: {path}"
                ) from exc
        staging_guard.assert_unchanged(staging)


def _remove_staging_tree_if_owned(
    staging_guard: _StagingGuard,
    *,
    error: BaseException,
) -> None:
    try:
        staging_guard.path.lstat()
    except FileNotFoundError:
        return
    except OSError as cleanup_error:
        error.add_note(
            f"could not safely inspect staging directory {staging_guard.path}: {cleanup_error}"
        )
        return
    try:
        staging_guard.assert_unchanged()
    except DashboardSourceError as cleanup_error:
        error.add_note(
            f"could not safely remove staging directory {staging_guard.path}: {cleanup_error}"
        )
        return
    cleanup_path = staging_guard.path.parent / (
        f".{staging_guard.path.name}.cayu-dashboard-cleanup-{uuid.uuid4().hex}"
    )
    try:
        staging_guard.path.rename(cleanup_path)
    except FileNotFoundError:
        return
    except OSError as cleanup_error:
        error.add_note(
            f"could not safely isolate staging directory {staging_guard.path}: {cleanup_error}"
        )
        return
    try:
        _remove_owned_staging_directory(cleanup_path, staging_guard=staging_guard)
    except (DashboardSourceError, OSError) as cleanup_error:
        error.add_note(f"could not safely remove staging directory {cleanup_path}: {cleanup_error}")


def _remove_owned_staging_directory(
    path: Path,
    *,
    staging_guard: _StagingGuard,
) -> None:
    if os.name == "nt":
        _delete_windows_entry_by_handle(
            path,
            expected_identity=staging_guard.identity,
        )
        return
    _remove_directory_contents_by_fd(path, staging_guard=staging_guard)
    staging_guard.assert_unchanged(path)
    path.rmdir()


def _remove_directory_contents_by_fd(
    path: Path,
    *,
    staging_guard: _StagingGuard,
) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    cleanup_error: BaseException | None = None
    try:
        identity = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(identity.st_mode)
            or _is_windows_name_surrogate(identity)
            or not os.path.samestat(staging_guard.identity, identity)
        ):
            raise DashboardSourceError(f"staging directory changed during extraction: {path}")
        _remove_directory_contents_from_fd(descriptor, path=path, flags=flags)
    except BaseException as exc:
        cleanup_error = exc
        raise
    finally:
        _close_descriptor(descriptor, error=cleanup_error)


def _remove_directory_contents_from_fd(
    descriptor: int,
    *,
    path: Path,
    flags: int,
) -> None:
    with os.scandir(descriptor) as entries:
        names = [entry.name for entry in entries]
    for name in names:
        child_path = path / name
        try:
            identity = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(identity.st_mode) or _is_windows_name_surrogate(identity):
            raise DashboardSourceError(
                f"staging directory acquired an unsafe link during cleanup: {child_path}"
            )
        if stat.S_ISDIR(identity.st_mode):
            child_descriptor = os.open(name, flags, dir_fd=descriptor)
            child_error: BaseException | None = None
            try:
                opened_identity = os.fstat(child_descriptor)
                if (
                    not stat.S_ISDIR(opened_identity.st_mode)
                    or _is_windows_name_surrogate(opened_identity)
                    or not os.path.samestat(identity, opened_identity)
                ):
                    raise DashboardSourceError(
                        f"staging directory changed during cleanup: {child_path}"
                    )
                _remove_directory_contents_from_fd(
                    child_descriptor,
                    path=child_path,
                    flags=flags,
                )
            except BaseException as exc:
                child_error = exc
                raise
            finally:
                _close_descriptor(child_descriptor, error=child_error)
            current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if not os.path.samestat(opened_identity, current):
                raise DashboardSourceError(
                    f"staging directory changed during cleanup: {child_path}"
                )
            os.rmdir(name, dir_fd=descriptor)
            continue
        if not stat.S_ISREG(identity.st_mode):
            raise DashboardSourceError(
                f"staging directory acquired an unsupported entry during cleanup: {child_path}"
            )
        current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if not os.path.samestat(identity, current):
            raise DashboardSourceError(f"staging file changed during cleanup: {child_path}")
        os.unlink(name, dir_fd=descriptor)


def _delete_windows_entry_by_handle(
    path: Path,
    *,
    expected_identity: os.stat_result,
) -> None:
    try:
        identity = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DashboardSourceError(f"staging entry changed during cleanup: {path}") from exc
    if not os.path.samestat(expected_identity, identity):
        raise DashboardSourceError(f"staging entry changed during cleanup: {path}")
    if stat.S_ISLNK(identity.st_mode) or _is_windows_name_surrogate(identity):
        raise DashboardSourceError(
            f"staging directory acquired an unsafe link during cleanup: {path}"
        )

    with _windows_deletion_handle(path) as mark_for_deletion:
        try:
            opened_identity = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise DashboardSourceError(f"staging entry changed during cleanup: {path}") from exc
        if (
            not os.path.samestat(identity, opened_identity)
            or stat.S_ISLNK(opened_identity.st_mode)
            or _is_windows_name_surrogate(opened_identity)
        ):
            raise DashboardSourceError(f"staging entry changed during cleanup: {path}")

        if stat.S_ISDIR(opened_identity.st_mode):
            try:
                children = list(path.iterdir())
            except OSError as exc:
                raise DashboardSourceError(
                    f"could not inspect staging directory during cleanup: {path}"
                ) from exc
            for child in children:
                try:
                    child_identity = child.stat(follow_symlinks=False)
                except OSError as exc:
                    raise DashboardSourceError(
                        f"staging entry changed during cleanup: {child}"
                    ) from exc
                _delete_windows_entry_by_handle(
                    child,
                    expected_identity=child_identity,
                )
        elif not stat.S_ISREG(opened_identity.st_mode):
            raise DashboardSourceError(
                f"staging directory acquired an unsupported entry during cleanup: {path}"
            )

        try:
            current = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise DashboardSourceError(f"staging entry changed during cleanup: {path}") from exc
        if (
            not os.path.samestat(opened_identity, current)
            or stat.S_IFMT(opened_identity.st_mode) != stat.S_IFMT(current.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or _is_windows_name_surrogate(current)
        ):
            raise DashboardSourceError(f"staging entry changed during cleanup: {path}")
        mark_for_deletion()


@contextmanager
def _windows_deletion_handle(path: Path) -> Iterator[Callable[[], None]]:
    if os.name != "nt":
        raise DashboardSourceError("Windows cleanup handles require Windows")

    import ctypes
    from ctypes import wintypes

    class _FileDispositionInfo(ctypes.Structure):
        _fields_ = (("delete_file", wintypes.BOOLEAN),)

    windows_ctypes: Any = ctypes
    kernel32 = windows_ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    set_file_information = kernel32.SetFileInformationByHandle
    set_file_information.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    set_file_information.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    handle = create_file(
        str(path),
        0x00010000 | 0x80,
        0x1,
        None,
        0x3,
        0x00200000 | 0x02000000,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        error_code = windows_ctypes.get_last_error()
        raise OSError(error_code, f"could not pin staging entry during cleanup: {path}")

    def mark_for_deletion() -> None:
        disposition = _FileDispositionInfo(delete_file=True)
        if not set_file_information(
            handle,
            4,
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            error_code = windows_ctypes.get_last_error()
            raise OSError(error_code, f"could not remove staging entry: {path}")

    cleanup_error: BaseException | None = None
    try:
        yield mark_for_deletion
    except BaseException as exc:
        cleanup_error = exc
        raise
    finally:
        if not close_handle(handle):
            error_code = windows_ctypes.get_last_error()
            close_error = OSError(
                error_code,
                f"could not release staging cleanup handle {path}",
            )
            if cleanup_error is not None:
                cleanup_error.add_note(str(close_error))
            else:
                raise close_error


@contextmanager
def _windows_directory_namespace_fence(path: Path) -> Iterator[None]:
    if os.name != "nt":
        yield
        return

    import ctypes
    from ctypes import wintypes

    windows_ctypes: Any = ctypes
    kernel32 = windows_ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    handle = create_file(
        str(path),
        0x80,
        0x1 | 0x2,
        None,
        0x3,
        0x00200000 | 0x02000000,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        error_code = windows_ctypes.get_last_error()
        raise OSError(error_code, f"could not fence staging directory {path}")
    fence_error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        fence_error = exc
        raise
    finally:
        if not close_handle(handle):
            error_code = windows_ctypes.get_last_error()
            close_error = OSError(
                error_code,
                f"could not release staging directory fence {path}",
            )
            if fence_error is not None:
                fence_error.add_note(str(close_error))
            else:
                raise close_error


def _publish_staged_tree(
    staging: Path,
    destination: Path,
    *,
    parent_guard: _DestinationParentGuard,
    staging_guard: _StagingGuard,
) -> None:
    parent_guard.assert_unchanged()
    staging_guard.assert_unchanged()
    _validate_destination(destination)
    parent_guard.assert_unchanged()
    staging_guard.assert_unchanged()
    with _publication_parent_namespace(parent_guard) as parent_descriptor:
        _publish_staged_tree_in_parent(
            staging,
            destination,
            parent_guard=parent_guard,
            staging_guard=staging_guard,
            parent_descriptor=parent_descriptor,
        )


@contextmanager
def _publication_parent_namespace(
    parent_guard: _DestinationParentGuard,
) -> Iterator[int | None]:
    if os.name == "nt":
        with _windows_directory_namespace_fence(parent_guard.path):
            parent_guard.assert_unchanged()
            yield None
            parent_guard.assert_unchanged()
        return

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        parent_descriptor = os.open(parent_guard.path, directory_flags)
    except OSError as exc:
        raise DashboardSourceError(
            f"destination parent changed during extraction: {parent_guard.path}"
        ) from exc
    publication_error: BaseException | None = None
    try:
        opened_parent_identity = os.fstat(parent_descriptor)
        if not stat.S_ISDIR(opened_parent_identity.st_mode) or not os.path.samestat(
            parent_guard.identity, opened_parent_identity
        ):
            raise DashboardSourceError(
                f"destination parent changed during extraction: {parent_guard.path}"
            )
        yield parent_descriptor
        parent_guard.assert_unchanged()
    except BaseException as exc:
        publication_error = exc
        raise
    finally:
        _close_descriptor(parent_descriptor, error=publication_error)


def _publish_staged_tree_in_parent(
    staging: Path,
    destination: Path,
    *,
    parent_guard: _DestinationParentGuard,
    staging_guard: _StagingGuard,
    parent_descriptor: int | None,
) -> None:
    original_empty: Path | None = None
    original_empty_identity: os.stat_result | None = None
    destination_identity = _publication_entry_identity(
        destination,
        parent_descriptor=parent_descriptor,
    )
    if destination_identity is not None:
        _assert_safe_publication_directory(
            destination_identity,
            destination=destination,
        )
        original_empty = destination.parent / (
            f".{destination.name}.cayu-dashboard-empty-{uuid.uuid4().hex}"
        )
        parent_guard.assert_unchanged()
        _rename_publication_entry(
            destination,
            original_empty,
            parent_descriptor=parent_descriptor,
        )
        try:
            parent_guard.assert_unchanged()
            _assert_empty_publication_directory(
                original_empty,
                destination=destination,
                expected_identity=destination_identity,
                parent_descriptor=parent_descriptor,
            )
            original_empty_identity = destination_identity
        except BaseException as exc:
            _restore_original_destination(
                original_empty,
                destination,
                parent_descriptor=parent_descriptor,
                error=exc,
            )
            raise
    try:
        parent_guard.assert_unchanged()
        staging_guard.assert_unchanged()
        _rename_publication_entry(
            staging,
            destination,
            parent_descriptor=parent_descriptor,
        )
    except BaseException as exc:
        if original_empty is not None:
            _restore_original_destination(
                original_empty,
                destination,
                parent_descriptor=parent_descriptor,
                error=exc,
            )
        raise
    try:
        staging_guard.assert_unchanged(destination)
        parent_guard.assert_unchanged()
    except BaseException as exc:
        _preserve_conflicting_destination(
            destination,
            parent_guard=parent_guard,
            parent_descriptor=parent_descriptor,
            error=exc,
        )
        if original_empty is not None:
            _restore_original_destination(
                original_empty,
                destination,
                parent_descriptor=parent_descriptor,
                error=exc,
            )
        raise
    if os.name == "nt":
        try:
            with _windows_directory_namespace_fence(destination):
                staging_guard.assert_unchanged(destination)
                _restore_published_directory_permissions(destination)
                staging_guard.assert_unchanged(destination)
            parent_guard.assert_unchanged()
        except BaseException as exc:
            _preserve_conflicting_destination(
                destination,
                parent_guard=parent_guard,
                parent_descriptor=parent_descriptor,
                error=exc,
            )
            if original_empty is not None:
                _restore_original_destination(
                    original_empty,
                    destination,
                    parent_descriptor=parent_descriptor,
                    error=exc,
                )
            raise
    if original_empty is None:
        return
    if original_empty_identity is None:
        raise DashboardSourceError("original destination ownership was not captured")
    try:
        _remove_publication_directory(
            original_empty,
            expected_identity=original_empty_identity,
            parent_descriptor=parent_descriptor,
        )
    except (DashboardSourceError, OSError) as exc:
        raise DashboardSourceError(
            "original destination cleanup conflicted after publication; "
            f"the dashboard source remains at {destination}, and the original destination "
            f"remains at {original_empty}: {exc}"
        ) from exc


def _publication_entry_identity(
    path: Path,
    *,
    parent_descriptor: int | None,
) -> os.stat_result | None:
    try:
        if parent_descriptor is None:
            return path.stat(follow_symlinks=False)
        return os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise DashboardSourceError(
            f"could not inspect destination during publication: {path}"
        ) from exc


def _assert_safe_publication_directory(
    identity: os.stat_result,
    *,
    destination: Path,
) -> None:
    if stat.S_ISLNK(identity.st_mode) or _is_windows_name_surrogate(identity):
        raise DashboardSourceError(
            f"destination must not traverse a symbolic link or junction: {destination}"
        )
    if not stat.S_ISDIR(identity.st_mode):
        raise DashboardSourceError(f"destination must be a directory: {destination}")


def _assert_empty_publication_directory(
    path: Path,
    *,
    destination: Path,
    expected_identity: os.stat_result,
    parent_descriptor: int | None,
) -> None:
    if parent_descriptor is None:
        with _windows_directory_namespace_fence(path):
            current = _publication_entry_identity(path, parent_descriptor=None)
            if current is None or not os.path.samestat(expected_identity, current):
                raise DashboardSourceError(f"destination changed during publication: {destination}")
            _assert_safe_publication_directory(current, destination=destination)
            with os.scandir(path) as entries:
                if next(entries, None) is not None:
                    raise DashboardSourceError(f"destination must be empty: {destination}")
        return

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path.name, directory_flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise DashboardSourceError(
            f"destination changed during publication: {destination}"
        ) from exc
    inspection_error: BaseException | None = None
    try:
        current = os.fstat(descriptor)
        if not os.path.samestat(expected_identity, current):
            raise DashboardSourceError(f"destination changed during publication: {destination}")
        _assert_safe_publication_directory(current, destination=destination)
        with os.scandir(descriptor) as entries:
            if next(entries, None) is not None:
                raise DashboardSourceError(f"destination must be empty: {destination}")
    except BaseException as exc:
        inspection_error = exc
        raise
    finally:
        _close_descriptor(descriptor, error=inspection_error)


def _rename_publication_entry(
    source: Path,
    destination: Path,
    *,
    parent_descriptor: int | None,
) -> None:
    if parent_descriptor is None:
        source.rename(destination)
        return
    os.rename(
        source.name,
        destination.name,
        src_dir_fd=parent_descriptor,
        dst_dir_fd=parent_descriptor,
    )


def _remove_publication_directory(
    path: Path,
    *,
    expected_identity: os.stat_result,
    parent_descriptor: int | None,
) -> None:
    current = _publication_entry_identity(path, parent_descriptor=parent_descriptor)
    if current is None or not os.path.samestat(expected_identity, current):
        raise DashboardSourceError(f"original destination changed during cleanup: {path}")
    _assert_safe_publication_directory(current, destination=path)
    if parent_descriptor is None:
        path.rmdir()
        return
    os.rmdir(path.name, dir_fd=parent_descriptor)


def _preserve_conflicting_destination(
    destination: Path,
    *,
    parent_guard: _DestinationParentGuard,
    parent_descriptor: int | None,
    error: BaseException,
) -> None:
    preserved = destination.parent / (
        f".{destination.name}.cayu-dashboard-conflict-{uuid.uuid4().hex}"
    )
    try:
        if parent_descriptor is None:
            parent_guard.assert_unchanged()
        _rename_publication_entry(
            destination,
            preserved,
            parent_descriptor=parent_descriptor,
        )
    except (DashboardSourceError, OSError) as preserve_error:
        error.add_note(
            f"could not preserve the conflicting destination at {destination}: {preserve_error}"
        )
        return
    error.add_note(f"preserved the conflicting destination at {preserved}")


def _restore_original_destination(
    original: Path,
    destination: Path,
    *,
    parent_descriptor: int | None,
    error: BaseException,
) -> None:
    try:
        _rename_publication_entry(
            original,
            destination,
            parent_descriptor=parent_descriptor,
        )
    except OSError as restore_error:
        error.add_note(
            f"could not restore the original destination at {destination}: {restore_error}"
        )


def _print_cli_error(exc: BaseException) -> None:
    print(f"error: {exc}", file=sys.stderr)
    for note in getattr(exc, "__notes__", ()):
        print(f"note: {note}", file=sys.stderr)
