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
import zipfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast

from cayu._server_contract_version import SERVER_CONTRACT_VERSION
from cayu.cli import _guarded_tree_publication as _tree_publication
from cayu.cli._guarded_tree_publication import (
    DestinationPolicy,
    GuardedTreePublicationError,
    GuardedTreeStage,
    publish_guarded_tree,
    validate_guarded_tree_files,
)

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
class _StagingGuard:
    path: Path
    identity: os.stat_result
    stable_identity: _tree_publication._Identity | None

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
        # Lambda sidecar export continues to use its pre-extraction identity
        # contract until that publisher migrates to the guarded-tree owner.
        return cls(path=path, identity=identity, stable_identity=None)

    @classmethod
    def from_publication(cls, stage: GuardedTreeStage) -> _StagingGuard:
        path = stage._specialized_path()
        try:
            identity = stage.capture_owned_identity()
        except GuardedTreePublicationError as exc:
            raise DashboardSourceError(
                f"staging directory changed during extraction: {path}"
            ) from exc
        return cls(
            path=path,
            identity=identity,
            stable_identity=stage._publication_identity,
        )

    def assert_unchanged(self, path: Path | None = None) -> None:
        candidate = self.path if path is None else path
        try:
            _reject_link_components(candidate)
            current = candidate.stat(follow_symlinks=False)
        except (DashboardSourceError, OSError) as exc:
            raise DashboardSourceError(
                f"staging directory changed during extraction: {candidate}"
            ) from exc
        if not stat.S_ISDIR(current.st_mode):
            raise DashboardSourceError(f"staging directory changed during extraction: {candidate}")
        if self.stable_identity is None:
            if not os.path.samestat(self.identity, current):
                raise DashboardSourceError(
                    f"staging directory changed during extraction: {candidate}"
                )
            return
        try:
            stable_identity = _tree_publication._capture_stable_identity(
                current,
                path=candidate,
            )
        except GuardedTreePublicationError as exc:
            raise DashboardSourceError(
                f"staging directory changed during extraction: {candidate}"
            ) from exc
        if stable_identity != self.stable_identity:
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
    try:
        validate_guarded_tree_files(artifact.contents)
    except GuardedTreePublicationError as exc:
        raise DashboardSourceError(str(exc)) from exc
    destination = _validate_destination(destination)
    _require_existing_destination_parent(destination.parent)

    def populate(staging: GuardedTreeStage) -> None:
        staging_guard = _StagingGuard.from_publication(staging)
        _write_staging_tree(
            staging_guard.path,
            artifact.contents,
            staging_guard=staging_guard,
        )

    try:
        publish_guarded_tree(
            destination,
            consumer="dashboard_source",
            request_digest=_dashboard_publication_request_digest(artifact.contents),
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=populate,
        )
    except GuardedTreePublicationError as exc:
        message = (
            f"destination must be empty: {destination}"
            if exc.code == "destination_not_empty"
            else str(exc)
        )
        if exc.paths:
            message += "; affected paths: " + ", ".join(repr(path) for path in exc.paths)
        translated = DashboardSourceError(message)
        for note in getattr(exc, "__notes__", ()):
            translated.add_note(note)
        raise translated from exc
    return DashboardEjectResult(destination=destination, manifest=artifact.manifest)


def _dashboard_publication_request_digest(contents: dict[str, bytes]) -> str:
    return _contents_digest(contents)


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
    return _tree_publication._is_unsafe_windows_component(component)


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
    return destination


def _require_existing_destination_parent(parent: Path) -> None:
    _reject_link_components(parent)
    try:
        identity = parent.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise DashboardSourceError(f"destination parent must already exist: {parent}") from exc
    except OSError as exc:
        raise DashboardSourceError(f"could not inspect destination parent {parent}: {exc}") from exc
    if not stat.S_ISDIR(identity.st_mode):
        raise DashboardSourceError(f"destination parent must be a directory: {parent}")


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
        if stat.S_ISLNK(identity.st_mode) or _is_windows_reparse_point(identity):
            raise DashboardSourceError(
                f"destination must not traverse a symbolic link or junction: {component}"
            )


def _is_windows_reparse_point(value: os.stat_result) -> bool:
    file_attributes = getattr(value, "st_file_attributes", 0)
    return bool(file_attributes & _WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT)


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
            or _is_windows_reparse_point(identity)
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
                    or _is_windows_reparse_point(child_identity)
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
                    if not stat.S_ISDIR(identity.st_mode) or _is_windows_reparse_point(identity):
                        raise DashboardSourceError(
                            f"staging directory changed during extraction: {current}"
                        )
                    directory_identities[prefix] = identity
                else:
                    identity = current.stat(follow_symlinks=False)
                    if (
                        not stat.S_ISDIR(identity.st_mode)
                        or _is_windows_reparse_point(identity)
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


def _remove_directory_contents_from_fd(
    descriptor: int,
    *,
    path: Path,
    flags: int,
) -> None:
    """Compatibility seam for Lambda until its PR-2 publisher migration."""

    with os.scandir(descriptor) as entries:
        names = [entry.name for entry in entries]
    for name in names:
        child_path = path / name
        try:
            identity = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(identity.st_mode) or _tree_publication._is_windows_reparse_point(identity):
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
                    or _tree_publication._is_windows_reparse_point(opened_identity)
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


def _remove_owned_staging_directory(
    path: Path,
    *,
    staging_guard: _StagingGuard,
) -> None:
    """Compatibility seam for Lambda until its PR-2 publisher migration."""

    expected = staging_guard.stable_identity
    if expected is None:
        if os.name == "nt":
            _remove_windows_entry_with_legacy_identity(
                path,
                expected_identity=staging_guard.identity,
            )
            return
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        cleanup_error: BaseException | None = None
        try:
            opened = os.fstat(descriptor)
            if not os.path.samestat(staging_guard.identity, opened):
                raise DashboardSourceError(f"staging directory changed during cleanup: {path}")
            _remove_directory_contents_from_fd(descriptor, path=path, flags=flags)
        except BaseException as exc:
            cleanup_error = exc
            raise
        finally:
            _close_descriptor(descriptor, error=cleanup_error)
        staging_guard.assert_unchanged(path)
        path.rmdir()
        return
    if os.name == "nt":
        try:
            _tree_publication._delete_windows_entry_by_handle(path, expected=expected)
        except GuardedTreePublicationError as exc:
            raise DashboardSourceError(str(exc)) from exc
        return
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    cleanup_error: BaseException | None = None
    try:
        opened = os.fstat(descriptor)
        if _tree_publication._capture_stable_identity(opened, descriptor=descriptor) != expected:
            raise DashboardSourceError(f"staging directory changed during cleanup: {path}")
        _remove_directory_contents_from_fd(descriptor, path=path, flags=flags)
    except BaseException as exc:
        cleanup_error = exc
        raise
    finally:
        _close_descriptor(descriptor, error=cleanup_error)
    staging_guard.assert_unchanged(path)
    path.rmdir()


def _remove_windows_entry_with_legacy_identity(
    path: Path,
    *,
    expected_identity: os.stat_result,
) -> None:
    """Retain Lambda's current Windows cleanup contract until its migration."""

    try:
        identity = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DashboardSourceError(f"staging entry changed during cleanup: {path}") from exc
    if not os.path.samestat(expected_identity, identity):
        raise DashboardSourceError(f"staging entry changed during cleanup: {path}")
    if stat.S_ISLNK(identity.st_mode) or _tree_publication._is_windows_reparse_point(identity):
        raise DashboardSourceError(
            f"staging directory acquired an unsafe link during cleanup: {path}"
        )

    with _tree_publication._windows_deletion_handle(path) as (_handle, mark_for_deletion):
        try:
            opened_identity = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise DashboardSourceError(f"staging entry changed during cleanup: {path}") from exc
        if (
            not os.path.samestat(identity, opened_identity)
            or stat.S_ISLNK(opened_identity.st_mode)
            or _tree_publication._is_windows_reparse_point(opened_identity)
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
                _remove_windows_entry_with_legacy_identity(
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
            or _tree_publication._is_windows_reparse_point(current)
        ):
            raise DashboardSourceError(f"staging entry changed during cleanup: {path}")
        mark_for_deletion()


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


def _print_cli_error(exc: BaseException) -> None:
    print(f"error: {exc}", file=sys.stderr)
    for note in getattr(exc, "__notes__", ()):
        print(f"note: {note}", file=sys.stderr)
