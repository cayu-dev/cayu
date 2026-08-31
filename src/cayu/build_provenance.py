"""Immutable Cayu runtime build provenance.

The strong paths consume packaging/deployment evidence.  Editable source trees
use a deterministic, explicitly weak content identity and never claim wheel or
container provenance.
"""

from __future__ import annotations

import json
import os
from enum import StrEnum
from functools import lru_cache
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank

RUNTIME_BUILD_PROVENANCE_SCHEMA_VERSION = 1
RUNTIME_BUILD_PROVENANCE_RECIPE = "cayu.runtime-build-provenance.v1"
RUNTIME_BUILD_PROVENANCE_ENV = "CAYU_RUNTIME_BUILD_PROVENANCE"
RUNTIME_BUILD_PROVENANCE_STRICT_ENV = "CAYU_REQUIRE_STRONG_RUNTIME_BUILD_PROVENANCE"
RUNTIME_BUILD_PROVENANCE_MAX_MANIFEST_BYTES = 4096
RUNTIME_BUILD_PROVENANCE_MAX_SOURCE_REVISION_BYTES = 256
RUNTIME_BUILD_PROVENANCE_MAX_DETAIL_CODE_BYTES = 64
_LOWER_SHA256_CHARACTERS = frozenset("0123456789abcdef")


class RuntimeBuildProvenanceOrigin(StrEnum):
    """Boundary that supplied one runtime build identity."""

    WHEEL_RECORD = "wheel_record"
    OCI_IMAGE_DIGEST = "oci_image_digest"
    DEVELOPMENT_SOURCE_TREE = "development_source_tree"
    EXPLICIT_MANIFEST = "explicit_manifest"
    LEGACY_RECORD = "legacy_record"
    UNAVAILABLE = "unavailable"


class RuntimeBuildArtifactKind(StrEnum):
    """Digest domain whose bytes or manifest identify the build."""

    WHEEL = "wheel"
    OCI_IMAGE = "oci_image"
    SOURCE_TREE = "source_tree"
    OTHER = "other"


class RuntimeBuildProvenanceAvailability(StrEnum):
    """Whether exact build identity is available for admission."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class RuntimeBuildProvenanceStrength(StrEnum):
    """Execution-profile strength vocabulary for runtime build identity."""

    STRUCTURAL = "structural"
    APPLICATION_VERSIONED = "application_versioned"
    UNAVAILABLE = "unavailable"


def _validate_sha256(value: str, field_name: str) -> str:
    if len(value) != 64 or any(character not in _LOWER_SHA256_CHARACTERS for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def _validate_bounded_text(value: str, field_name: str, max_bytes: int) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{field_name} exceeds {max_bytes} UTF-8 bytes.")
    return value


def _runtime_build_fingerprint(
    *,
    origin: RuntimeBuildProvenanceOrigin,
    artifact_kind: RuntimeBuildArtifactKind,
    artifact_digest: str,
    strength: RuntimeBuildProvenanceStrength,
) -> str:
    material = {
        "record_type": RUNTIME_BUILD_PROVENANCE_RECIPE,
        "schema_version": RUNTIME_BUILD_PROVENANCE_SCHEMA_VERSION,
        "origin": origin.value,
        "artifact_kind": artifact_kind.value,
        "artifact_digest": artifact_digest,
        "strength": strength.value,
    }
    return sha256(canonical_durable_json_bytes(material, "runtime_build_provenance")).hexdigest()


class RuntimeBuildProvenance(BaseModel):
    """Versioned, bounded identity for the exact Cayu runtime build."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = RUNTIME_BUILD_PROVENANCE_SCHEMA_VERSION
    recipe: Literal["cayu.runtime-build-provenance.v1"] = RUNTIME_BUILD_PROVENANCE_RECIPE
    origin: RuntimeBuildProvenanceOrigin
    availability: RuntimeBuildProvenanceAvailability
    strength: RuntimeBuildProvenanceStrength
    fingerprint: str | None = None
    artifact_kind: RuntimeBuildArtifactKind | None = None
    artifact_digest: str | None = None
    source_revision: str | None = None
    detail_code: str | None = None

    @field_validator("fingerprint", "artifact_digest")
    @classmethod
    def validate_optional_sha256(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _validate_sha256(value, info.field_name)

    @field_validator("source_revision")
    @classmethod
    def validate_source_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_bounded_text(
            value,
            "source_revision",
            RUNTIME_BUILD_PROVENANCE_MAX_SOURCE_REVISION_BYTES,
        )

    @field_validator("detail_code")
    @classmethod
    def validate_detail_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = _validate_bounded_text(
            value,
            "detail_code",
            RUNTIME_BUILD_PROVENANCE_MAX_DETAIL_CODE_BYTES,
        )
        if not value.isascii() or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in value
        ):
            raise ValueError("detail_code must be a lowercase ASCII identifier.")
        return value

    @model_validator(mode="after")
    def validate_state(self) -> RuntimeBuildProvenance:
        unavailable = self.availability is RuntimeBuildProvenanceAvailability.UNAVAILABLE
        if unavailable != (self.fingerprint is None):
            raise ValueError("Unavailable build provenance must omit its fingerprint.")
        if unavailable != (self.strength is RuntimeBuildProvenanceStrength.UNAVAILABLE):
            raise ValueError("Unavailable build provenance must use unavailable strength.")
        if unavailable:
            if self.artifact_kind is not None or self.artifact_digest is not None:
                raise ValueError("Unavailable build provenance cannot claim an artifact digest.")
            if self.detail_code is None:
                raise ValueError("Unavailable build provenance requires a bounded detail code.")
            if self.origin not in {
                RuntimeBuildProvenanceOrigin.LEGACY_RECORD,
                RuntimeBuildProvenanceOrigin.UNAVAILABLE,
            }:
                raise ValueError("Unavailable build provenance uses an invalid origin.")
        else:
            if self.detail_code is not None:
                raise ValueError("Available build provenance cannot carry an unavailable detail.")
            if self.origin in {
                RuntimeBuildProvenanceOrigin.LEGACY_RECORD,
                RuntimeBuildProvenanceOrigin.UNAVAILABLE,
            }:
                raise ValueError("Available build provenance uses an invalid origin.")
            if self.strength is RuntimeBuildProvenanceStrength.STRUCTURAL and (
                self.artifact_kind is None or self.artifact_digest is None
            ):
                raise ValueError("Structural build provenance requires a typed artifact digest.")
            if (self.artifact_kind is None) != (self.artifact_digest is None):
                raise ValueError("Artifact kind and digest must be supplied together.")
            if self.artifact_kind is not None and self.artifact_digest is not None:
                expected_fingerprint = _runtime_build_fingerprint(
                    origin=self.origin,
                    artifact_kind=self.artifact_kind,
                    artifact_digest=self.artifact_digest,
                    strength=self.strength,
                )
                if self.fingerprint != expected_fingerprint:
                    raise ValueError("Runtime build fingerprint conflicts with its manifest.")
        if self.origin is RuntimeBuildProvenanceOrigin.OCI_IMAGE_DIGEST and (
            self.artifact_kind is not RuntimeBuildArtifactKind.OCI_IMAGE
        ):
            raise ValueError("OCI image provenance must retain the OCI image digest domain.")
        if self.origin is RuntimeBuildProvenanceOrigin.WHEEL_RECORD and (
            self.artifact_kind is not RuntimeBuildArtifactKind.WHEEL
        ):
            raise ValueError("Wheel provenance must retain the wheel digest domain.")
        if self.origin is RuntimeBuildProvenanceOrigin.DEVELOPMENT_SOURCE_TREE and (
            self.strength is not RuntimeBuildProvenanceStrength.APPLICATION_VERSIONED
            or self.artifact_kind is not RuntimeBuildArtifactKind.SOURCE_TREE
        ):
            raise ValueError("Development source identity must remain explicitly weak.")
        return self

    @classmethod
    def unavailable(
        cls,
        detail_code: str,
        *,
        legacy: bool = False,
    ) -> RuntimeBuildProvenance:
        """Return explicit unavailable or legacy provenance."""

        return cls(
            origin=(
                RuntimeBuildProvenanceOrigin.LEGACY_RECORD
                if legacy
                else RuntimeBuildProvenanceOrigin.UNAVAILABLE
            ),
            availability=RuntimeBuildProvenanceAvailability.UNAVAILABLE,
            strength=RuntimeBuildProvenanceStrength.UNAVAILABLE,
            detail_code=detail_code,
        )

    @classmethod
    def from_artifact_digest(
        cls,
        *,
        origin: RuntimeBuildProvenanceOrigin,
        artifact_kind: RuntimeBuildArtifactKind,
        artifact_digest: str,
        source_revision: str | None = None,
        strength: RuntimeBuildProvenanceStrength = RuntimeBuildProvenanceStrength.STRUCTURAL,
    ) -> RuntimeBuildProvenance:
        """Derive a domain-separated runtime fingerprint from build evidence."""

        artifact_digest = _validate_sha256(artifact_digest, "artifact_digest")
        fingerprint = _runtime_build_fingerprint(
            origin=origin,
            artifact_kind=artifact_kind,
            artifact_digest=artifact_digest,
            strength=strength,
        )
        return cls(
            origin=origin,
            availability=RuntimeBuildProvenanceAvailability.AVAILABLE,
            strength=strength,
            fingerprint=fingerprint,
            artifact_kind=artifact_kind,
            artifact_digest=artifact_digest,
            source_revision=source_revision,
        )


def copy_runtime_build_provenance(value: RuntimeBuildProvenance) -> RuntimeBuildProvenance:
    """Detach and revalidate runtime build provenance."""

    if type(value) is not RuntimeBuildProvenance:
        raise TypeError("runtime_build_provenance must be RuntimeBuildProvenance.")
    return RuntimeBuildProvenance.model_validate(value.model_dump(mode="json"))


def runtime_build_provenance_identity(
    value: RuntimeBuildProvenance,
) -> RuntimeBuildProvenance:
    """Return authoritative build evidence without diagnostic source revision."""

    copied = copy_runtime_build_provenance(value).model_dump(mode="json")
    copied["source_revision"] = None
    return RuntimeBuildProvenance.model_validate(copied)


def legacy_runtime_build_provenance() -> RuntimeBuildProvenance:
    """Identity for session records created before build provenance existed."""

    return RuntimeBuildProvenance.unavailable("legacy_record", legacy=True)


def _distribution_is_editable(distribution: metadata.Distribution) -> bool:
    try:
        raw = distribution.read_text("direct_url.json")
        if raw is None:
            return False
        value = json.loads(raw)
    except (OSError, TypeError, ValueError):
        return False
    return bool(
        isinstance(value, dict)
        and isinstance(value.get("dir_info"), dict)
        and value["dir_info"].get("editable") is True
    )


def _distribution_owns_runtime_module(
    distribution: metadata.Distribution,
    runtime_module: Path,
) -> bool:
    """Return whether a hashed wheel record owns the imported runtime module."""

    try:
        imported_path = runtime_module.resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    for item in distribution.files or ():
        file_hash = item.hash
        if file_hash is None or file_hash.mode != "sha256":
            continue
        try:
            installed_path = Path(str(distribution.locate_file(item))).resolve(strict=True)
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        if installed_path == imported_path:
            return True
    return False


def _wheel_record_entry_may_omit_hash(path: str) -> bool:
    """Return whether the wheel standard permits this non-behavioral entry to be unhashed."""

    return path.endswith(
        (
            ".dist-info/RECORD",
            ".dist-info/RECORD.jws",
            ".dist-info/RECORD.p7s",
        )
    )


def _cached_bytecode_source_path(path: str) -> str | None:
    """Return the covered source path for an installer-generated bytecode entry."""

    if not path.endswith((".pyc", ".pyo")):
        return None
    cache_marker = "/__pycache__/"
    if cache_marker not in path:
        return f"{path[:-1]}"
    package_path, cached_name = path.rsplit(cache_marker, maxsplit=1)
    source_name, separator, _cache_tag = cached_name.partition(".")
    if not separator or not source_name:
        return None
    return f"{package_path}/{source_name}.py"


def _wheel_record_provenance(
    distribution: metadata.Distribution,
) -> RuntimeBuildProvenance | None:
    files = tuple(distribution.files or ())
    hashed_paths = {
        str(item).replace("\\", "/")
        for item in files
        if item.hash is not None and item.hash.mode == "sha256"
    }
    records: list[dict[str, Any]] = []
    for item in files:
        path = str(item).replace("\\", "/")
        file_hash = item.hash
        if file_hash is None:
            bytecode_source = _cached_bytecode_source_path(path)
            if bytecode_source is not None:
                if bytecode_source not in hashed_paths:
                    return None
                continue
            if not _wheel_record_entry_may_omit_hash(path):
                return None
            continue
        if file_hash.mode != "sha256":
            return None
        records.append(
            {
                "path": path,
                "sha256_urlsafe_base64": file_hash.value,
                "size": item.size,
            }
        )
    if not records:
        return None
    record_digest = sha256(
        canonical_durable_json_bytes(
            {
                "record_type": "cayu.wheel-record-projection.v1",
                "files": sorted(records, key=lambda item: item["path"]),
            },
            "wheel_record",
        )
    ).hexdigest()
    return RuntimeBuildProvenance.from_artifact_digest(
        origin=RuntimeBuildProvenanceOrigin.WHEEL_RECORD,
        artifact_kind=RuntimeBuildArtifactKind.WHEEL,
        artifact_digest=record_digest,
    )


def _development_source_tree_provenance(package_root: Path) -> RuntimeBuildProvenance:
    digest = sha256()
    digest.update(b"cayu.development-source-tree.v1\0")
    try:
        paths = sorted(
            path
            for path in package_root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
        if not paths:
            return RuntimeBuildProvenance.unavailable("source_tree_empty")
        for path in paths:
            if path.is_symlink():
                return RuntimeBuildProvenance.unavailable("source_tree_symlink")
            relative = path.relative_to(package_root).as_posix().encode("utf-8")
            content_digest = sha256()
            size = 0
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    size += len(chunk)
                    content_digest.update(chunk)
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(size.to_bytes(8, "big"))
            digest.update(content_digest.digest())
    except (OSError, UnicodeError, ValueError):
        return RuntimeBuildProvenance.unavailable("source_tree_unreadable")
    return RuntimeBuildProvenance.from_artifact_digest(
        origin=RuntimeBuildProvenanceOrigin.DEVELOPMENT_SOURCE_TREE,
        artifact_kind=RuntimeBuildArtifactKind.SOURCE_TREE,
        artifact_digest=digest.hexdigest(),
        strength=RuntimeBuildProvenanceStrength.APPLICATION_VERSIONED,
    )


def _explicit_manifest_from_environment() -> RuntimeBuildProvenance | None:
    raw = os.environ.get(RUNTIME_BUILD_PROVENANCE_ENV)
    if raw is None:
        return None
    encoded = raw.encode("utf-8")
    if len(encoded) > RUNTIME_BUILD_PROVENANCE_MAX_MANIFEST_BYTES:
        raise RuntimeError("Configured runtime build-provenance manifest exceeds 4096 bytes.")
    try:
        value = json.loads(raw)
        provenance = RuntimeBuildProvenance.model_validate(value)
    except Exception as exc:
        raise RuntimeError("Configured runtime build-provenance manifest is invalid.") from exc
    if provenance.origin in {
        RuntimeBuildProvenanceOrigin.DEVELOPMENT_SOURCE_TREE,
        RuntimeBuildProvenanceOrigin.LEGACY_RECORD,
        RuntimeBuildProvenanceOrigin.UNAVAILABLE,
        RuntimeBuildProvenanceOrigin.WHEEL_RECORD,
    }:
        raise RuntimeError(
            "Configured runtime build provenance must use explicit_manifest or "
            "oci_image_digest origin."
        )
    return provenance


def _strict_build_provenance_required() -> bool:
    value = os.environ.get(RUNTIME_BUILD_PROVENANCE_STRICT_ENV)
    if value is None or value == "0":
        return False
    if value == "1":
        return True
    raise RuntimeError(f"{RUNTIME_BUILD_PROVENANCE_STRICT_ENV} must be '0' or '1'.")


@lru_cache(maxsize=1)
def current_runtime_build_provenance() -> RuntimeBuildProvenance:
    """Resolve and cache one immutable build identity for this process."""

    provenance = _explicit_manifest_from_environment()
    if provenance is None:
        try:
            distribution = metadata.distribution("cayu")
        except metadata.PackageNotFoundError:
            distribution = None
        if (
            distribution is not None
            and not _distribution_is_editable(distribution)
            and _distribution_owns_runtime_module(distribution, Path(__file__))
        ):
            provenance = _wheel_record_provenance(distribution)
        if provenance is None:
            provenance = _development_source_tree_provenance(Path(__file__).resolve().parents[1])
    if _strict_build_provenance_required() and (
        provenance.availability is not RuntimeBuildProvenanceAvailability.AVAILABLE
        or provenance.strength is not RuntimeBuildProvenanceStrength.STRUCTURAL
    ):
        raise RuntimeError(
            "Strong runtime build provenance is required but this Cayu build has only "
            f"{provenance.availability.value}/{provenance.strength.value} evidence."
        )
    return copy_runtime_build_provenance(provenance)


__all__ = [
    "RUNTIME_BUILD_PROVENANCE_ENV",
    "RUNTIME_BUILD_PROVENANCE_MAX_MANIFEST_BYTES",
    "RUNTIME_BUILD_PROVENANCE_RECIPE",
    "RUNTIME_BUILD_PROVENANCE_SCHEMA_VERSION",
    "RUNTIME_BUILD_PROVENANCE_STRICT_ENV",
    "RuntimeBuildArtifactKind",
    "RuntimeBuildProvenance",
    "RuntimeBuildProvenanceAvailability",
    "RuntimeBuildProvenanceOrigin",
    "RuntimeBuildProvenanceStrength",
    "copy_runtime_build_provenance",
    "current_runtime_build_provenance",
    "legacy_runtime_build_provenance",
    "runtime_build_provenance_identity",
]
