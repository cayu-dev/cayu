"""Local Lambda MicroVM sidecar artifact commands."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import stat
import sys
import tempfile
import tomllib
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath
from typing import Any, cast

from cayu.cli.dashboard import (
    DashboardSourceError,
    _remove_directory_contents_from_fd,
    _remove_owned_staging_directory,
    _StagingGuard,
    _windows_directory_namespace_fence,
)

_MANIFEST_NAME = "cayu-lambda-microvm-sidecar-manifest.json"
_PACKAGE_RESOURCE_DIRECTORY = "lambda_microvm_sidecar"
_MANIFEST_KEYS = {
    "artifact_version",
    "cayu_version",
    "content_digest",
    "files",
    "protocol_version",
    "schema_version",
}
_FILE_KEYS = {"path", "sha256", "size"}
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


class _SidecarArtifactError(RuntimeError):
    """The packaged sidecar artifact is invalid or could not be exported."""


@dataclass(frozen=True)
class _SidecarFile:
    path: str
    size: int
    sha256: str

    def as_manifest_value(self) -> dict[str, str | int]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True)
class _SidecarManifest:
    schema_version: int
    artifact_version: int
    cayu_version: str
    protocol_version: str
    content_digest: str
    files: tuple[_SidecarFile, ...]


@dataclass(frozen=True)
class _ValidatedSidecarArtifact:
    manifest: _SidecarManifest
    contents: dict[str, bytes]


@dataclass(frozen=True)
class _SidecarExportResult:
    destination: Path
    content_digest: str


@dataclass(frozen=True)
class _SidecarResource:
    root: Traversable
    cayu_version: str


@dataclass(frozen=True)
class _DirectoryGuard:
    path: Path
    identity: os.stat_result

    @classmethod
    def capture(cls, path: Path, *, label: str) -> _DirectoryGuard:
        _reject_link_components(path)
        try:
            identity = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise _SidecarArtifactError(f"could not capture {label} identity: {path}") from exc
        if not stat.S_ISDIR(identity.st_mode):
            raise _SidecarArtifactError(f"{label} must be a directory: {path}")
        return cls(path=path, identity=identity)

    def assert_unchanged(self, *, label: str, path: Path | None = None) -> None:
        candidate = self.path if path is None else path
        try:
            _reject_link_components(candidate)
            current = candidate.stat(follow_symlinks=False)
        except (OSError, _SidecarArtifactError) as exc:
            raise _SidecarArtifactError(
                f"{label} changed during sidecar export: {candidate}"
            ) from exc
        if not stat.S_ISDIR(current.st_mode) or not os.path.samestat(self.identity, current):
            raise _SidecarArtifactError(f"{label} changed during sidecar export: {candidate}")


def add_lambda_microvm_parser(subparsers: Any) -> None:
    """Register the ``lambda-microvm sidecar export`` command group."""
    lambda_microvm = subparsers.add_parser(
        "lambda-microvm",
        help="Manage AWS Lambda MicroVM support artifacts.",
        description=(
            "Manage AWS Lambda MicroVM support artifacts. "
            "Use the sidecar export command to materialize a build context."
        ),
    )
    lambda_commands = lambda_microvm.add_subparsers(
        dest="lambda_microvm_command",
        required=True,
    )
    sidecar = lambda_commands.add_parser(
        "sidecar",
        help="Manage the first-party Lambda MicroVM command sidecar.",
        description=(
            "Manage the first-party Lambda MicroVM command sidecar. "
            "Use `cayu lambda-microvm sidecar export DESTINATION` next."
        ),
    )
    sidecar_commands = sidecar.add_subparsers(dest="sidecar_command", required=True)
    export = sidecar_commands.add_parser(
        "export",
        help="Export the versioned sidecar image build context.",
        description=(
            "Export the versioned sidecar image build context. "
            "Build the emitted context with the container tooling documented inside it."
        ),
    )
    export.add_argument("destination", type=Path, metavar="DESTINATION")
    export.add_argument(
        "--replace",
        action="store_true",
        help="Delete and replace all contents of an existing destination directory.",
    )


def run_lambda_microvm(args: argparse.Namespace) -> int:
    """Dispatch a parsed ``lambda-microvm`` invocation."""
    try:
        result = _export_sidecar(args.destination, replace=args.replace)
        print(
            f"exported Lambda MicroVM sidecar to {result.destination}\n"
            f"content digest: {result.content_digest}"
        )
    except Exception as exc:
        _print_cli_error(exc)
        return 1
    return 0


def _export_sidecar(
    destination: Path,
    *,
    replace: bool,
    resource_root: Traversable | None = None,
    expected_cayu_version: str | None = None,
) -> _SidecarExportResult:
    if resource_root is None:
        artifact = _load_default_validated_artifact()
    else:
        if expected_cayu_version is None:
            from cayu.cli import _version

            expected_cayu_version = _version()
        artifact = _load_validated_artifact(
            resource_root,
            expected_cayu_version=expected_cayu_version,
        )
    destination = _validate_destination(destination, replace=replace)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_link_components(destination.parent)
    parent_guard = _DirectoryGuard.capture(destination.parent, label="destination parent")
    with _publication_parent_namespace(parent_guard) as parent_descriptor:
        destination_identity = _publication_entry_identity(
            destination,
            parent_descriptor=parent_descriptor,
        )
        staging, staging_guard = _create_staging_directory(
            destination,
            parent_guard=parent_guard,
            parent_descriptor=parent_descriptor,
        )
        try:
            parent_guard.assert_unchanged(label="destination parent")
            staging_guard.assert_unchanged(label="staging directory")
            _write_staging_tree(staging, artifact)
            parent_guard.assert_unchanged(label="destination parent")
            staging_guard.assert_unchanged(label="staging directory")
            _publish_staged_tree(
                staging,
                destination,
                replace=replace,
                parent_guard=parent_guard,
                staging_guard=staging_guard,
                destination_identity=destination_identity,
                parent_descriptor=parent_descriptor,
            )
            if parent_descriptor is None:
                try:
                    parent_guard.assert_unchanged(label="destination parent")
                except BaseException as exc:
                    _preserve_owned_publication_after_parent_change(
                        destination,
                        staging_guard=staging_guard,
                        error=exc,
                    )
                    raise
        except BaseException as exc:
            _remove_owned_tree(
                staging_guard,
                error=exc,
                label="staging directory",
                parent_descriptor=parent_descriptor,
            )
            raise
    return _SidecarExportResult(
        destination=destination,
        content_digest=artifact.manifest.content_digest,
    )


def _print_cli_error(exc: BaseException) -> None:
    print(f"error: {exc}", file=sys.stderr)
    for note in getattr(exc, "__notes__", ()):
        print(f"note: {note}", file=sys.stderr)


def _load_default_validated_artifact() -> _ValidatedSidecarArtifact:
    resource = _sidecar_resource()
    return _load_validated_artifact(
        resource.root,
        expected_cayu_version=resource.cayu_version,
    )


def _sidecar_resource() -> _SidecarResource:
    packaged = files("cayu.data").joinpath(_PACKAGE_RESOURCE_DIRECTORY)
    if packaged.is_dir():
        try:
            installed_version = importlib.metadata.version("cayu")
        except importlib.metadata.PackageNotFoundError as exc:
            raise _SidecarArtifactError(
                "the packaged sidecar cannot be verified without Cayu distribution metadata"
            ) from exc
        return _SidecarResource(root=packaged, cayu_version=installed_version)

    # A normal wheel/sdist installation must never consult a checkout. This
    # fallback exists only when this module itself is executing from src/cayu.
    module_path = Path(__file__).resolve()
    project_root = module_path.parents[3]
    expected_module = project_root / "src" / "cayu" / "cli" / "lambda_microvm.py"
    source = project_root / "examples" / "aws" / "lambda_microvm_sidecar"
    if expected_module.resolve() == module_path and source.is_dir():
        try:
            project = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
            source_version = project["project"]["version"]
        except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
            raise _SidecarArtifactError(
                f"could not read the source-checkout Cayu version: {exc}"
            ) from exc
        if not isinstance(source_version, str) or not source_version.strip():
            raise _SidecarArtifactError("source-checkout Cayu version is invalid")
        return _SidecarResource(root=source, cayu_version=source_version)
    raise _SidecarArtifactError("the installed Cayu distribution omits the sidecar artifact")


def _load_validated_artifact(
    root: Traversable,
    *,
    expected_cayu_version: str,
) -> _ValidatedSidecarArtifact:
    if _is_symlink(root) or not root.is_dir():
        raise _SidecarArtifactError(
            "sidecar resource root must be an ordinary directory, not a link"
        )
    manifest_resource = root.joinpath(_MANIFEST_NAME)
    if _is_symlink(manifest_resource) or not manifest_resource.is_file():
        raise _SidecarArtifactError(f"sidecar manifest is missing: {_MANIFEST_NAME}")
    try:
        raw_manifest = manifest_resource.read_bytes()
    except OSError as exc:
        raise _SidecarArtifactError(f"could not read sidecar manifest: {exc}") from exc
    contents = _collect_resource_files(root)
    manifest_bytes = contents.pop(_MANIFEST_NAME, None)
    if manifest_bytes is None:
        raise _SidecarArtifactError(f"sidecar manifest is missing: {_MANIFEST_NAME}")
    if manifest_bytes != raw_manifest:
        raise _SidecarArtifactError("sidecar manifest changed while it was being validated")

    return _validate_artifact_contents(
        {**contents, _MANIFEST_NAME: manifest_bytes},
        expected_cayu_version=expected_cayu_version,
    )


def _validate_artifact_contents(
    artifact_contents: dict[str, bytes],
    *,
    expected_cayu_version: str,
) -> _ValidatedSidecarArtifact:
    """Validate one complete manifest-governed sidecar tree."""
    contents = dict(artifact_contents)
    manifest_bytes = contents.pop(_MANIFEST_NAME, None)
    if manifest_bytes is None:
        raise _SidecarArtifactError(f"sidecar manifest is missing: {_MANIFEST_NAME}")
    manifest = _parse_manifest(manifest_bytes)

    expected_paths = {item.path for item in manifest.files}
    actual_paths = set(contents)
    missing = sorted(expected_paths - actual_paths)
    unexpected = sorted(actual_paths - expected_paths)
    if missing:
        raise _SidecarArtifactError(
            f"sidecar artifact is missing manifest files: {', '.join(missing)}"
        )
    if unexpected:
        raise _SidecarArtifactError(
            f"sidecar artifact contains unexpected files: {', '.join(unexpected)}"
        )

    for item in manifest.files:
        content = contents[item.path]
        if len(content) != item.size:
            raise _SidecarArtifactError(
                f"sidecar resource size mismatch for {item.path}: "
                f"expected {item.size}, found {len(content)}"
            )
        digest = _sha256(content)
        if digest != item.sha256:
            raise _SidecarArtifactError(
                f"sidecar resource digest mismatch for {item.path}: "
                f"expected {item.sha256}, found {digest}"
            )

    canonical_files = [item.as_manifest_value() for item in manifest.files]
    content_digest = _sha256(
        json.dumps(
            canonical_files,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if content_digest != manifest.content_digest:
        raise _SidecarArtifactError(
            "sidecar aggregate digest mismatch: "
            f"expected {manifest.content_digest}, found {content_digest}"
        )

    from cayu.runners.aws_lambda_microvm import LAMBDA_MICROVM_PROTOCOL_VERSION

    if manifest.cayu_version != expected_cayu_version:
        raise _SidecarArtifactError(
            "sidecar Cayu version mismatch: "
            f"expected {expected_cayu_version}, found {manifest.cayu_version}"
        )
    if manifest.protocol_version != LAMBDA_MICROVM_PROTOCOL_VERSION:
        raise _SidecarArtifactError(
            "sidecar protocol version mismatch: "
            f"expected {LAMBDA_MICROVM_PROTOCOL_VERSION}, "
            f"found {manifest.protocol_version}"
        )

    return _ValidatedSidecarArtifact(
        manifest=manifest,
        contents={**contents, _MANIFEST_NAME: manifest_bytes},
    )


def _render_manifest(
    contents: dict[str, bytes],
    *,
    cayu_version: str,
    protocol_version: str,
) -> bytes:
    """Render the canonical manifest for a sidecar tree without its manifest."""
    files = [
        _SidecarFile(path=path, size=len(contents[path]), sha256=_sha256(contents[path]))
        for path in sorted(contents)
    ]
    canonical_files = [item.as_manifest_value() for item in files]
    manifest = {
        "artifact_version": 1,
        "cayu_version": cayu_version,
        "content_digest": _sha256(
            json.dumps(
                canonical_files,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ),
        "files": canonical_files,
        "protocol_version": protocol_version,
        "schema_version": 1,
    }
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _parse_manifest(raw: bytes) -> _SidecarManifest:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _SidecarArtifactError("sidecar manifest must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise _SidecarArtifactError("sidecar manifest must be a JSON object")
    keys = set(value)
    if keys != _MANIFEST_KEYS:
        missing = sorted(_MANIFEST_KEYS - keys)
        unexpected = sorted(keys - _MANIFEST_KEYS)
        detail: list[str] = []
        if missing:
            detail.append(f"missing {', '.join(missing)}")
        if unexpected:
            detail.append(f"unexpected {', '.join(unexpected)}")
        raise _SidecarArtifactError(f"sidecar manifest fields are invalid: {'; '.join(detail)}")

    schema_version = _require_positive_int(value, "schema_version")
    artifact_version = _require_positive_int(value, "artifact_version")
    if schema_version != 1:
        raise _SidecarArtifactError(
            f"unsupported sidecar manifest schema version: {schema_version}"
        )
    if artifact_version != 1:
        raise _SidecarArtifactError(f"unsupported sidecar artifact version: {artifact_version}")
    cayu_version = _require_nonblank_string(value, "cayu_version")
    protocol_version = _require_nonblank_string(value, "protocol_version")
    content_digest = _require_sha256(value, "content_digest")

    raw_files = value.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise _SidecarArtifactError("sidecar manifest files must be a non-empty array")
    parsed_files: list[_SidecarFile] = []
    seen_paths: set[str] = set()
    seen_casefolded_paths: set[str] = set()
    for index, raw_file in enumerate(raw_files):
        if not isinstance(raw_file, dict) or set(raw_file) != _FILE_KEYS:
            raise _SidecarArtifactError(
                f"sidecar manifest file entry {index} must contain path, sha256, and size"
            )
        file_value = cast("dict[str, Any]", raw_file)
        path = _validate_manifest_path(file_value.get("path"), index=index)
        if path in seen_paths or path.casefold() in seen_casefolded_paths:
            raise _SidecarArtifactError(f"duplicate sidecar manifest path: {path}")
        seen_paths.add(path)
        seen_casefolded_paths.add(path.casefold())
        size = file_value.get("size")
        if type(size) is not int or size < 0:
            raise _SidecarArtifactError(
                f"sidecar manifest file size must be a non-negative integer: {path}"
            )
        digest = file_value.get("sha256")
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise _SidecarArtifactError(f"invalid sidecar manifest SHA-256 for {path}")
        parsed_files.append(_SidecarFile(path=path, size=size, sha256=digest))

    paths = [item.path for item in parsed_files]
    if paths != sorted(paths):
        raise _SidecarArtifactError("sidecar manifest files must be sorted by path")
    return _SidecarManifest(
        schema_version=schema_version,
        artifact_version=artifact_version,
        cayu_version=cayu_version,
        protocol_version=protocol_version,
        content_digest=content_digest,
        files=tuple(parsed_files),
    )


def _require_positive_int(value: dict[str, Any], field: str) -> int:
    candidate = value.get(field)
    if type(candidate) is not int or candidate < 1:
        raise _SidecarArtifactError(f"sidecar manifest {field} must be a positive integer")
    return candidate


def _require_nonblank_string(value: dict[str, Any], field: str) -> str:
    candidate = value.get(field)
    if not isinstance(candidate, str) or not candidate.strip() or candidate != candidate.strip():
        raise _SidecarArtifactError(f"sidecar manifest {field} must be a clean non-blank string")
    return candidate


def _require_sha256(value: dict[str, Any], field: str) -> str:
    candidate = value.get(field)
    if not isinstance(candidate, str) or _SHA256_PATTERN.fullmatch(candidate) is None:
        raise _SidecarArtifactError(f"sidecar manifest {field} must be a SHA-256 digest")
    return candidate


def _validate_manifest_path(value: Any, *, index: int) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise _SidecarArtifactError(f"invalid sidecar manifest path at index {index}")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise _SidecarArtifactError(
            f"sidecar manifest path at index {index} must be valid UTF-8"
        ) from exc
    if "\x00" in value:
        raise _SidecarArtifactError(f"unsafe sidecar manifest path at index {index}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or value == _MANIFEST_NAME
    ):
        raise _SidecarArtifactError(f"unsafe sidecar manifest path: {value}")
    return value


def _collect_resource_files(root: Traversable) -> dict[str, bytes]:
    collected: dict[str, bytes] = {}
    collected_casefolded_paths: set[str] = set()

    def visit(directory: Traversable, prefix: PurePosixPath) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda child: child.name)
        except OSError as exc:
            raise _SidecarArtifactError(f"could not enumerate sidecar resources: {exc}") from exc
        for child in children:
            if child.name == "__pycache__" or child.name.endswith((".pyc", ".pyo")):
                continue
            try:
                child.name.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise _SidecarArtifactError("sidecar resource names must be valid UTF-8") from exc
            if child.name in {"", ".", ".."} or "/" in child.name or "\\" in child.name:
                raise _SidecarArtifactError(f"unsafe sidecar resource name: {child.name}")
            relative = prefix / child.name
            path = relative.as_posix()
            if _is_symlink(child):
                raise _SidecarArtifactError(f"sidecar resources must not contain links: {path}")
            if child.is_dir():
                visit(child, relative)
            elif child.is_file():
                casefolded_path = path.casefold()
                if casefolded_path in collected_casefolded_paths:
                    raise _SidecarArtifactError(f"case-colliding sidecar resource path: {path}")
                try:
                    collected[path] = child.read_bytes()
                except OSError as exc:
                    raise _SidecarArtifactError(
                        f"could not read sidecar resource {path}: {exc}"
                    ) from exc
                collected_casefolded_paths.add(casefolded_path)
            else:
                raise _SidecarArtifactError(f"unsupported sidecar resource type: {path}")

    visit(root, PurePosixPath())
    return collected


def _is_symlink(resource: Traversable) -> bool:
    return isinstance(resource, Path) and resource.is_symlink()


def _sha256(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _validate_destination(destination: Path, *, replace: bool) -> Path:
    try:
        os.fspath(destination).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise _SidecarArtifactError("destination path must be valid UTF-8") from exc
    try:
        destination = destination.expanduser()
    except RuntimeError as exc:
        raise _SidecarArtifactError(f"could not expand destination {destination}: {exc}") from exc
    destination = Path(os.path.normpath(destination.absolute()))
    _reject_link_components(destination)
    try:
        resolved = destination.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise _SidecarArtifactError(f"could not resolve destination {destination}: {exc}") from exc
    if resolved.parent == resolved:
        raise _SidecarArtifactError("destination must not be a filesystem root")
    current_directory = Path.cwd().resolve()
    if resolved == current_directory or resolved in current_directory.parents:
        raise _SidecarArtifactError(
            "destination must not be the current working directory or one of its ancestors"
        )
    home_directory = Path.home().resolve()
    if resolved == home_directory or resolved in home_directory.parents:
        raise _SidecarArtifactError(
            "destination must not be the home directory or one of its ancestors"
        )
    if resolved.exists() and not resolved.is_dir():
        raise _SidecarArtifactError(f"destination must be a directory: {resolved}")
    if resolved.is_dir() and not replace and next(resolved.iterdir(), None) is not None:
        raise _SidecarArtifactError(
            f"destination is not empty; pass --replace to replace it: {resolved}"
        )
    return resolved


def _reject_link_components(path: Path) -> None:
    for component in reversed((path, *path.parents)):
        try:
            identity = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise _SidecarArtifactError(
                f"could not inspect destination path component {component}: {exc}"
            ) from exc
        if stat.S_ISLNK(identity.st_mode) or getattr(identity, "st_reparse_tag", 0):
            raise _SidecarArtifactError(
                f"destination must not be a symlink or junction: {component}"
            )


@contextmanager
def _publication_parent_namespace(
    parent_guard: _DirectoryGuard,
) -> Iterator[int | None]:
    if os.name == "nt":
        with _windows_directory_namespace_fence(parent_guard.path):
            parent_guard.assert_unchanged(label="destination parent")
            yield None
            parent_guard.assert_unchanged(label="destination parent")
        return

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(parent_guard.path, flags)
    except OSError as exc:
        raise _SidecarArtifactError(
            f"destination parent changed during sidecar export: {parent_guard.path}"
        ) from exc
    namespace_error: BaseException | None = None
    try:
        identity = os.fstat(descriptor)
        if not stat.S_ISDIR(identity.st_mode) or not os.path.samestat(
            parent_guard.identity, identity
        ):
            raise _SidecarArtifactError(
                f"destination parent changed during sidecar export: {parent_guard.path}"
            )
        yield descriptor
        parent_guard.assert_unchanged(label="destination parent")
    except BaseException as exc:
        namespace_error = exc
        raise
    finally:
        _close_descriptor(descriptor, error=namespace_error)


def _create_staging_directory(
    destination: Path,
    *,
    parent_guard: _DirectoryGuard,
    parent_descriptor: int | None,
) -> tuple[Path, _DirectoryGuard]:
    if parent_descriptor is None:
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.cayu-sidecar-",
                dir=destination.parent,
            )
        )
        guard = _DirectoryGuard.capture(staging, label="staging directory")
        parent_guard.assert_unchanged(label="destination parent")
        return staging, guard

    created_name: str | None = None
    try:
        for _attempt in range(100):
            candidate = f".{destination.name}.cayu-sidecar-{uuid.uuid4().hex}"
            try:
                os.mkdir(candidate, mode=0o700, dir_fd=parent_descriptor)
            except FileExistsError:
                continue
            created_name = candidate
            break
        if created_name is None:
            raise _SidecarArtifactError("could not allocate a private staging directory")
        identity = os.stat(
            created_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        _assert_safe_directory_identity(
            identity,
            path=destination.parent / created_name,
            label="staging directory",
        )
        return destination.parent / created_name, _DirectoryGuard(
            path=destination.parent / created_name,
            identity=identity,
        )
    except BaseException as exc:
        if created_name is not None:
            try:
                os.rmdir(created_name, dir_fd=parent_descriptor)
            except OSError as cleanup_error:
                exc.add_note(
                    "could not remove newly created staging directory "
                    f"{destination.parent / created_name}: {cleanup_error}"
                )
        raise


def _publication_entry_identity(
    path: Path,
    *,
    parent_descriptor: int | None,
) -> os.stat_result | None:
    try:
        if parent_descriptor is None:
            identity = path.stat(follow_symlinks=False)
        else:
            identity = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _SidecarArtifactError(f"could not inspect destination: {path}") from exc
    _assert_safe_directory_identity(identity, path=path, label="destination")
    return identity


def _assert_safe_directory_identity(
    identity: os.stat_result,
    *,
    path: Path,
    label: str,
) -> None:
    if stat.S_ISLNK(identity.st_mode) or getattr(identity, "st_reparse_tag", 0):
        raise _SidecarArtifactError(f"{label} must not be a symlink or junction: {path}")
    if not stat.S_ISDIR(identity.st_mode):
        raise _SidecarArtifactError(f"{label} must be a directory: {path}")


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


def _close_descriptor(descriptor: int, *, error: BaseException | None = None) -> None:
    try:
        os.close(descriptor)
    except OSError as close_error:
        if error is None:
            raise
        error.add_note(f"could not close sidecar directory descriptor: {close_error}")


def _write_staging_tree(
    staging: Path,
    artifact: _ValidatedSidecarArtifact,
) -> None:
    if os.name != "nt":
        os.chmod(staging, 0o755)
    for relative, content in sorted(artifact.contents.items()):
        path = staging.joinpath(*PurePosixPath(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            output.write(content)
    if os.name != "nt":
        for path in sorted(staging.rglob("*")):
            os.chmod(path, 0o755 if path.is_dir() else 0o644)


def _publish_staged_tree(
    staging: Path,
    destination: Path,
    *,
    replace: bool,
    parent_guard: _DirectoryGuard,
    staging_guard: _DirectoryGuard,
    destination_identity: os.stat_result | None,
    parent_descriptor: int | None,
) -> None:
    parent_guard.assert_unchanged(label="destination parent")
    staging_guard.assert_unchanged(label="staging directory")
    current_destination = _publication_entry_identity(
        destination,
        parent_descriptor=parent_descriptor,
    )
    if not _same_optional_identity(destination_identity, current_destination):
        raise _SidecarArtifactError(f"destination changed during sidecar export: {destination}")
    backup: Path | None = None
    backup_guard: _DirectoryGuard | None = None
    if destination_identity is not None:
        backup = destination.parent / (
            f".{destination.name}.cayu-sidecar-backup-{uuid.uuid4().hex}"
        )
        _rename_publication_entry(
            destination,
            backup,
            parent_descriptor=parent_descriptor,
        )
        backup_guard = _DirectoryGuard(path=backup, identity=destination_identity)
        moved_identity = _publication_entry_identity(
            backup,
            parent_descriptor=parent_descriptor,
        )
        if moved_identity is None or not os.path.samestat(destination_identity, moved_identity):
            error = _SidecarArtifactError(
                f"destination changed during backup publication: {destination}"
            )
            _preserve_conflicting_entry(
                backup,
                parent_descriptor=parent_descriptor,
                error=error,
            )
            raise error
    try:
        parent_guard.assert_unchanged(label="destination parent")
        staging_guard.assert_unchanged(label="staging directory")
        _rename_publication_entry(
            staging,
            destination,
            parent_descriptor=parent_descriptor,
        )
        published_identity = _publication_entry_identity(
            destination,
            parent_descriptor=parent_descriptor,
        )
        if published_identity is None or not os.path.samestat(
            staging_guard.identity, published_identity
        ):
            raise _SidecarArtifactError(
                f"staging directory changed during publication: {destination}"
            )
    except BaseException as exc:
        current_destination = _publication_entry_identity(
            destination,
            parent_descriptor=parent_descriptor,
        )
        if current_destination is not None and not os.path.samestat(
            staging_guard.identity, current_destination
        ):
            _preserve_conflicting_entry(
                destination,
                parent_descriptor=parent_descriptor,
                error=exc,
            )
        if backup is not None:
            exc.add_note(f"the original destination remains at {backup}")
        raise

    if backup_guard is not None:
        cleanup_error = _SidecarArtifactError(
            f"export completed at {destination}, but old destination cleanup failed at {backup}"
        )
        removed = _remove_owned_tree(
            backup_guard,
            error=cleanup_error,
            label="old destination",
            require_empty=not replace,
            parent_descriptor=parent_descriptor,
        )
        if not removed:
            raise cleanup_error


def _same_optional_identity(
    expected: os.stat_result | None,
    current: os.stat_result | None,
) -> bool:
    if expected is None or current is None:
        return expected is current
    return os.path.samestat(expected, current)


def _preserve_conflicting_entry(
    path: Path,
    *,
    parent_descriptor: int | None,
    error: BaseException,
) -> None:
    preserved = path.parent / f".{path.name}.cayu-sidecar-conflict-{uuid.uuid4().hex}"
    try:
        _rename_publication_entry(
            path,
            preserved,
            parent_descriptor=parent_descriptor,
        )
    except OSError as preserve_error:
        error.add_note(f"could not preserve conflicting directory at {path}: {preserve_error}")
        return
    error.add_note(f"preserved conflicting directory at {preserved}")


def _preserve_owned_publication_after_parent_change(
    destination: Path,
    *,
    staging_guard: _DirectoryGuard,
    error: BaseException,
) -> None:
    try:
        current = _publication_entry_identity(
            destination,
            parent_descriptor=None,
        )
    except _SidecarArtifactError as inspection_error:
        error.add_note(
            f"could not inspect replacement destination {destination}: {inspection_error}"
        )
        return
    if current is None or not os.path.samestat(staging_guard.identity, current):
        return
    _preserve_conflicting_entry(
        destination,
        parent_descriptor=None,
        error=error,
    )


def _remove_owned_tree(
    guard: _DirectoryGuard,
    *,
    error: BaseException,
    label: str,
    parent_descriptor: int | None,
    require_empty: bool = False,
) -> bool:
    isolated: Path | None = None
    try:
        current = _publication_entry_identity(
            guard.path,
            parent_descriptor=parent_descriptor,
        )
        if current is None:
            return True
        if not os.path.samestat(guard.identity, current):
            raise _SidecarArtifactError(f"{label} changed during sidecar export: {guard.path}")
        if require_empty and not _publication_directory_is_empty(
            guard.path,
            expected_identity=guard.identity,
            parent_descriptor=parent_descriptor,
        ):
            raise _SidecarArtifactError(f"{label} is no longer empty: {guard.path}")
        isolated = guard.path.parent / f".{guard.path.name}.cayu-sidecar-cleanup-{uuid.uuid4().hex}"
        _rename_publication_entry(
            guard.path,
            isolated,
            parent_descriptor=parent_descriptor,
        )
        isolated_identity = _publication_entry_identity(
            isolated,
            parent_descriptor=parent_descriptor,
        )
        if isolated_identity is None or not os.path.samestat(guard.identity, isolated_identity):
            raise _SidecarArtifactError(f"{label} changed during cleanup: {isolated}")
        _remove_owned_publication_directory(
            isolated,
            expected_identity=guard.identity,
            parent_descriptor=parent_descriptor,
        )
        return True
    except (DashboardSourceError, OSError, _SidecarArtifactError) as cleanup_error:
        if isolated is not None:
            try:
                isolated_identity = _publication_entry_identity(
                    isolated,
                    parent_descriptor=parent_descriptor,
                )
                original_identity = _publication_entry_identity(
                    guard.path,
                    parent_descriptor=parent_descriptor,
                )
                if (
                    isolated_identity is not None
                    and os.path.samestat(guard.identity, isolated_identity)
                    and original_identity is None
                ):
                    _rename_publication_entry(
                        isolated,
                        guard.path,
                        parent_descriptor=parent_descriptor,
                    )
            except (OSError, _SidecarArtifactError) as restore_error:
                error.add_note(f"could not restore preserved {label} {guard.path}: {restore_error}")
        error.add_note(f"could not safely remove {label} {guard.path}: {cleanup_error}")
        return False


def _publication_directory_is_empty(
    path: Path,
    *,
    expected_identity: os.stat_result,
    parent_descriptor: int | None,
) -> bool:
    if parent_descriptor is None:
        with _windows_directory_namespace_fence(path):
            current = path.stat(follow_symlinks=False)
            if not os.path.samestat(expected_identity, current):
                raise _SidecarArtifactError(f"directory changed during sidecar export: {path}")
            return next(path.iterdir(), None) is None

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
    inspection_error: BaseException | None = None
    try:
        current = os.fstat(descriptor)
        if not os.path.samestat(expected_identity, current):
            raise _SidecarArtifactError(f"directory changed during sidecar export: {path}")
        with os.scandir(descriptor) as entries:
            return next(entries, None) is None
    except BaseException as exc:
        inspection_error = exc
        raise
    finally:
        _close_descriptor(descriptor, error=inspection_error)


def _remove_owned_publication_directory(
    path: Path,
    *,
    expected_identity: os.stat_result,
    parent_descriptor: int | None,
) -> None:
    if parent_descriptor is None:
        _remove_owned_staging_directory(
            path,
            staging_guard=_StagingGuard(path=path, identity=expected_identity),
        )
        return

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
    cleanup_error: BaseException | None = None
    try:
        opened_identity = os.fstat(descriptor)
        if not os.path.samestat(expected_identity, opened_identity):
            raise _SidecarArtifactError(f"directory changed during cleanup: {path}")
        _remove_directory_contents_from_fd(descriptor, path=path, flags=flags)
        current = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not os.path.samestat(opened_identity, current):
            raise _SidecarArtifactError(f"directory changed during cleanup: {path}")
        os.rmdir(path.name, dir_fd=parent_descriptor)
    except BaseException as exc:
        cleanup_error = exc
        raise
    finally:
        _close_descriptor(descriptor, error=cleanup_error)
