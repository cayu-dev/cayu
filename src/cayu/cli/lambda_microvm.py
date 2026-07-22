"""Local Lambda MicroVM sidecar artifact commands."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
import tempfile
import tomllib
import uuid
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath
from typing import Any, cast

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
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.cayu-sidecar-",
            dir=destination.parent,
        )
    )
    try:
        _write_staging_tree(staging, artifact)
        _publish_staged_tree(staging, destination, replace=replace)
    except BaseException as exc:
        if staging.exists():
            try:
                shutil.rmtree(staging)
            except OSError as cleanup_error:
                exc.add_note(f"could not remove staging directory {staging}: {cleanup_error}")
        raise
    if staging.exists():
        try:
            shutil.rmtree(staging)
        except OSError as exc:
            raise _SidecarArtifactError(
                f"export completed but staging cleanup failed at {staging}: {exc}"
            ) from exc
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
    if destination.is_symlink():
        raise _SidecarArtifactError(f"destination must not be a symlink: {destination}")
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


def _publish_staged_tree(staging: Path, destination: Path, *, replace: bool) -> None:
    backup: Path | None = None
    if destination.exists():
        backup = destination.parent / (
            f".{destination.name}.cayu-sidecar-backup-{uuid.uuid4().hex}"
        )
        destination.rename(backup)
    try:
        staging.rename(destination)
    except BaseException as exc:
        if backup is not None:
            exc.add_note(f"the original destination remains at {backup}")
        raise

    if backup is not None:
        try:
            if replace:
                shutil.rmtree(backup)
            else:
                backup.rmdir()
        except OSError as cleanup_error:
            raise _SidecarArtifactError(
                f"export completed at {destination}, but old destination cleanup "
                f"failed at {backup}: {cleanup_error}"
            ) from cleanup_error
