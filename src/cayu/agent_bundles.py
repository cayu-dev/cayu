"""Portable, content-addressed bundles for complete Cayu agent snapshots.

``AgentSnapshot`` remains the sole agent-state manifest.  This module transports
the authenticated snapshot-node closure plus the provider-owned component
objects named by that manifest.  It deliberately does not introduce another
state root, logical agent identity, activation authority, or secret container.
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
import shutil
import stat
import threading
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager, suppress
from enum import StrEnum
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._filesystem_lock import cooperative_path_lock
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_durable_clean_nonblank,
)
from cayu.agent_snapshots import (
    AGENT_SNAPSHOT_MAX_BYTES,
    AgentSnapshot,
    AgentSnapshotAccess,
    AgentSnapshotCaptureRequest,
    AgentSnapshotCompleteness,
    AgentSnapshotComponentCapture,
    AgentSnapshotComponentKind,
    AgentSnapshotComponentProvider,
    AgentSnapshotComponentRef,
    AgentSnapshotComponentSelector,
    AgentSnapshotConsistency,
    AgentSnapshotCoordinator,
    AgentSnapshotExecutionProfileRef,
    AgentSnapshotIdentityBinding,
    AgentSnapshotLogicalRef,
    AgentSnapshotMaterialization,
    AgentSnapshotMaterializationCapability,
    AgentSnapshotMaterializationOperation,
    AgentSnapshotMaterializationRequest,
    AgentSnapshotMaterializedComponent,
    AgentSnapshotNode,
    AgentSnapshotOverlayKind,
    AgentSnapshotOverlayRef,
    AgentSnapshotPinReceipt,
    AgentSnapshotPinRequest,
    AgentSnapshotProtection,
    AgentSnapshotProtectionKind,
    AgentSnapshotRedaction,
    AgentSnapshotRef,
    AgentSnapshotResultBinding,
    AgentSnapshotRetentionClass,
    AgentSnapshotStore,
    AgentSnapshotSubject,
    AgentSnapshotTerminalDisposition,
    AgentSnapshotTrialBinding,
    AgentSnapshotTrialStateMode,
    agent_snapshot_from_json,
)
from cayu.vaults.redaction import SecretRedactor

AGENT_BUNDLE_RECORD_TYPE = "cayu.agent-bundle"
AGENT_BUNDLE_SCHEMA_VERSION = 1
AGENT_BUNDLE_INDEX_FILENAME = "index.json"
AGENT_BUNDLE_MAX_INDEX_BYTES = 16 * 1024 * 1024
AGENT_BUNDLE_MAX_OBJECTS = 1_000_000
AGENT_BUNDLE_MAX_OBJECT_BYTES = 64 * 1024 * 1024 * 1024
AGENT_BUNDLE_MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024 * 1024
AGENT_BUNDLE_OBJECT_DIRECTORY = "objects"

_SHA256_CHARS = frozenset("0123456789abcdef")
_SAFE_IDENTIFIER_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:._-"
)
_FORBIDDEN_PORTABLE_PAYLOAD_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "evaluator_truth",
        "expected_answer",
        "hidden_case",
        "judge_prompt",
        "password",
        "private_key",
        "provider_continuation",
        "secret",
        "token",
    }
)


def _clean(value: str, field_name: str, *, max_chars: int = 512) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value) > max_chars:
        raise ValueError(f"{field_name} must be at most {max_chars} characters.")
    return value


def _digest(value: bytes) -> str:
    return sha256(value).hexdigest()


def _file_digest_and_size(path: Path, *, max_bytes: int) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise AgentBundleError("object_path_not_regular")
    declared_size = path.stat().st_size
    if declared_size > max_bytes:
        raise AgentBundleError("object_size_limit_exceeded")
    hasher = sha256()
    byte_count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            byte_count += len(chunk)
            if byte_count > max_bytes:
                raise AgentBundleError("object_size_limit_exceeded")
            hasher.update(chunk)
    if byte_count != declared_size:
        raise AgentBundleError("object_changed_during_read")
    return hasher.hexdigest(), byte_count


def _file_contains_secret(path: Path, redactor: SecretRedactor) -> bool:
    overlap = max(0, redactor.max_secret_utf8_bytes - 1)
    tail = b""
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            window = tail + chunk
            if redactor.contains_secret_bytes(window):
                return True
            tail = window[-overlap:] if overlap else b""
    return False


def _sha256_hex(value: str, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def _content_digest(value: object, field_name: str) -> str:
    return _digest(canonical_durable_json_bytes(value, field_name))


def _ordered_unique_text(
    value: object,
    field_name: str,
    *,
    max_items: int = 256,
) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{field_name} must be an ordered array.")
    copied = tuple(
        _clean(item, f"{field_name}[{index}]", max_chars=256)
        if type(item) is str
        else (_raise_string(field_name, index))
        for index, item in enumerate(value)
    )
    if len(copied) > max_items:
        raise ValueError(f"{field_name} exceeds its item limit.")
    if copied != tuple(sorted(set(copied))):
        raise ValueError(f"{field_name} must be unique and sorted.")
    return copied


def _raise_string(field_name: str, index: int) -> str:
    raise ValueError(f"{field_name}[{index}] must be a string.")


def _relative_path(value: str, field_name: str) -> str:
    value = _clean(value, field_name, max_chars=1024)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value.startswith(("/", "\\"))
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field_name} must be a canonical relative POSIX path.")
    normalized = path.as_posix()
    if normalized != value:
        raise ValueError(f"{field_name} must be a canonical relative POSIX path.")
    return normalized


def _canonical_json(value: BaseModel, field_name: str) -> bytes:
    return canonical_durable_json_bytes(value.model_dump(mode="json"), field_name)


def _reject_private_payload_fields(value: object, path: str = "payload") -> None:
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{path} keys must be strings.")
            normalized = key.casefold().replace("-", "_")
            if normalized in _FORBIDDEN_PORTABLE_PAYLOAD_KEYS:
                raise ValueError(
                    f"{path}.{key} is external/private state and cannot enter a component package."
                )
            _reject_private_payload_fields(item, f"{path}.{key}")
    elif type(value) is list:
        for index, item in enumerate(value):
            _reject_private_payload_fields(item, f"{path}[{index}]")


def _safe_object_path(root: Path, digest: str) -> Path:
    _sha256_hex(digest, "digest")
    return root / AGENT_BUNDLE_OBJECT_DIRECTORY / digest[:2] / digest[2:]


_CAS_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_CAS_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_CAS_PATH_ERROR_NUMBERS = frozenset(
    number
    for number in (
        getattr(errno, "ELOOP", None),
        getattr(errno, "ENOTDIR", None),
    )
    if number is not None
)


def _require_cas_descriptor_guards() -> None:
    if (
        not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or os.open not in os.supports_dir_fd
        or os.mkdir not in os.supports_dir_fd
        or os.rename not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.unlink not in os.supports_dir_fd
    ):
        raise AgentBundleError("object_store_descriptor_guard_unavailable")


@contextmanager
def _open_object_shard(
    root: Path,
    digest: str,
    *,
    create: bool,
) -> Iterator[tuple[int, str] | None]:
    """Pin a CAS shard directory without following a substituted symlink."""

    _sha256_hex(digest, "digest")
    _require_cas_descriptor_guards()
    base = root / AGENT_BUNDLE_OBJECT_DIRECTORY
    try:
        base_fd = os.open(base, _CAS_DIRECTORY_FLAGS)
    except OSError as error:
        if error.errno in _CAS_PATH_ERROR_NUMBERS:
            raise AgentBundleError("object_path_not_regular") from error
        raise
    shard_fd: int | None = None
    try:
        try:
            shard_fd = os.open(digest[:2], _CAS_DIRECTORY_FLAGS, dir_fd=base_fd)
        except FileNotFoundError:
            if not create:
                yield None
                return
            with suppress(FileExistsError):
                os.mkdir(digest[:2], mode=0o700, dir_fd=base_fd)
            try:
                shard_fd = os.open(digest[:2], _CAS_DIRECTORY_FLAGS, dir_fd=base_fd)
            except OSError as error:
                if error.errno in _CAS_PATH_ERROR_NUMBERS:
                    raise AgentBundleError("object_path_not_regular") from error
                raise
        except OSError as error:
            if error.errno in _CAS_PATH_ERROR_NUMBERS:
                raise AgentBundleError("object_path_not_regular") from error
            raise
        yield shard_fd, digest[2:]
    finally:
        if shard_fd is not None:
            os.close(shard_fd)
        os.close(base_fd)


@contextmanager
def _open_regular_object(
    root: Path,
    digest: str,
) -> Iterator[tuple[BinaryIO, int] | None]:
    with _open_object_shard(root, digest, create=False) as opened:
        if opened is None:
            yield None
            return
        shard_fd, name = opened
        try:
            descriptor = os.open(name, _CAS_FILE_FLAGS, dir_fd=shard_fd)
        except FileNotFoundError:
            yield None
            return
        except OSError as error:
            if error.errno in _CAS_PATH_ERROR_NUMBERS:
                raise AgentBundleError("object_path_not_regular") from error
            raise
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            os.close(descriptor)
            raise AgentBundleError("object_path_not_regular")
        with os.fdopen(descriptor, "rb") as stream:
            yield stream, info.st_size


def _read_regular_at(
    directory_fd: int,
    name: str,
    *,
    max_bytes: int,
) -> tuple[bytes, int]:
    try:
        descriptor = os.open(name, _CAS_FILE_FLAGS, dir_fd=directory_fd)
    except OSError as error:
        if error.errno in _CAS_PATH_ERROR_NUMBERS:
            raise AgentBundleError("object_path_not_regular") from error
        raise
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise AgentBundleError("object_path_not_regular")
    if info.st_size > max_bytes:
        os.close(descriptor)
        raise AgentBundleError("object_size_limit_exceeded")
    with os.fdopen(descriptor, "rb") as stream:
        content = stream.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise AgentBundleError("object_size_limit_exceeded")
    if len(content) != info.st_size:
        raise AgentBundleError("object_changed_during_read")
    return content, info.st_size


@asynccontextmanager
async def _async_path_lock(
    root: Path,
    relative_path: str,
    *,
    lock_directory_name: str,
) -> AsyncIterator[None]:
    manager = cooperative_path_lock(
        root,
        relative_path,
        lock_directory_name=lock_directory_name,
    )
    acquire = asyncio.create_task(asyncio.to_thread(manager.__enter__))
    try:
        await asyncio.shield(acquire)
    except asyncio.CancelledError:
        await acquire
        release = asyncio.create_task(asyncio.to_thread(manager.__exit__, None, None, None))
        await asyncio.shield(release)
        raise
    try:
        yield
    except BaseException as error:
        release = asyncio.create_task(
            asyncio.to_thread(manager.__exit__, type(error), error, error.__traceback__)
        )
        try:
            await asyncio.shield(release)
        except asyncio.CancelledError:
            await release
            raise
        raise
    else:
        release = asyncio.create_task(asyncio.to_thread(manager.__exit__, None, None, None))
        try:
            await asyncio.shield(release)
        except asyncio.CancelledError:
            await release
            raise


class _BundleModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        validate_default=True,
    )


class AgentSnapshotProfile(StrEnum):
    REUSABLE_AGENT = "reusable_agent"
    CONTINUING_AGENT = "continuing_agent"
    EVALUATION_CANDIDATE = "evaluation_candidate"


class AgentSnapshotSessionDisposition(StrEnum):
    FRESH_ON_MATERIALIZE = "fresh_on_materialize"
    SAFE_FRONTIER = "safe_frontier"
    NON_RESTORABLE = "non_restorable"


class AgentBundleMode(StrEnum):
    FULL = "full"
    THIN = "thin"


class AgentSnapshotMaterializationMode(StrEnum):
    MATERIALIZE = "materialize"
    FORK_AS_SEED = "fork_as_seed"
    RESTORE = "restore"


class AgentBundleObjectKind(StrEnum):
    SNAPSHOT_DOCUMENT = "snapshot_document"
    SNAPSHOT_NODE = "snapshot_node"
    COMPONENT_MANIFEST = "component_manifest"
    COMPONENT_BLOB = "component_blob"
    OPERATION_RECEIPT = "operation_receipt"


class AgentExternalBindingKind(StrEnum):
    CREDENTIAL = "credential"
    EVALUATOR = "evaluator"
    BUDGET = "budget"
    LEASE = "lease"
    EXTERNAL_SERVICE = "external_service"
    HOSTED_PROVIDER = "hosted_provider"


class AgentExternalBindingRequirement(_BundleModel):
    kind: AgentExternalBindingKind
    name: StrictStr = Field(max_length=256)
    requirement_fingerprint: StrictStr
    required: StrictBool = True

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str, info) -> str:
        value = _clean(value, info.field_name, max_chars=256)
        if any(character not in _SAFE_IDENTIFIER_CHARS for character in value):
            raise ValueError("External binding names must be opaque identifiers.")
        return value

    @field_validator("requirement_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    def identity_material(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class AgentSnapshotComponentFile(_BundleModel):
    path: StrictStr = Field(max_length=1024)
    digest: StrictStr
    byte_count: StrictInt = Field(ge=0, le=AGENT_BUNDLE_MAX_OBJECT_BYTES)
    content_type: StrictStr = Field(max_length=256)
    executable: StrictBool = False

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str, info) -> str:
        return _relative_path(value, info.field_name)

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, value: str, info) -> str:
        value = _clean(value, info.field_name, max_chars=256)
        if "/" not in value or any(character.isspace() for character in value):
            raise ValueError("content_type must be a bounded media type.")
        return value


class AgentSnapshotComponentPackage(_BundleModel):
    """Provider-owned portable content named by a snapshot component ref."""

    record_type: Literal["cayu.agent-snapshot-component"] = "cayu.agent-snapshot-component"
    schema_version: Literal[1] = 1
    kind: AgentSnapshotComponentKind
    provider_id: StrictStr = Field(max_length=256)
    component_schema: StrictStr = Field(max_length=256)
    profile: AgentSnapshotProfile
    session_disposition: AgentSnapshotSessionDisposition | None = None
    payload: dict[str, object]
    files: tuple[AgentSnapshotComponentFile, ...] = ()
    external_bindings: tuple[AgentExternalBindingRequirement, ...] = ()

    @field_validator("provider_id", "component_schema")
    @classmethod
    def validate_identifiers(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @field_validator("payload", mode="before")
    @classmethod
    def validate_payload(cls, value: object) -> dict[str, object]:
        return copy_durable_json_object(value, "agent_snapshot_component.payload")

    @field_validator("files", "external_bindings", mode="before")
    @classmethod
    def validate_arrays(cls, value: object, info) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError(f"{info.field_name} must be an ordered array.")
        return value

    @model_validator(mode="after")
    def validate_package(self) -> AgentSnapshotComponentPackage:
        _reject_private_payload_fields(self.payload)
        paths = tuple(file.path for file in self.files)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("Component files must be unique and sorted by path.")
        binding_keys = tuple(
            (binding.kind.value, binding.name, binding.requirement_fingerprint)
            for binding in self.external_bindings
        )
        if binding_keys != tuple(sorted(set(binding_keys))):
            raise ValueError("External bindings must be unique and canonically ordered.")
        if self.kind is AgentSnapshotComponentKind.SESSION:
            if self.session_disposition is None:
                raise ValueError("Session packages require an explicit disposition.")
            if (
                self.profile is AgentSnapshotProfile.REUSABLE_AGENT
                and self.session_disposition
                is not AgentSnapshotSessionDisposition.FRESH_ON_MATERIALIZE
            ):
                raise ValueError("Reusable agents must start with a fresh session.")
            if (
                self.profile is AgentSnapshotProfile.CONTINUING_AGENT
                and self.session_disposition is not AgentSnapshotSessionDisposition.SAFE_FRONTIER
            ):
                raise ValueError("Continuing agents require an exact safe frontier.")
        elif self.session_disposition is not None:
            raise ValueError("Only session packages declare a session disposition.")
        return self

    def identity_material(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    @property
    def digest(self) -> str:
        return _content_digest(self.identity_material(), "agent_snapshot_component")

    @property
    def document(self) -> bytes:
        return _canonical_json(self, "agent_snapshot_component")


class AgentBundleObjectRef(_BundleModel):
    digest: StrictStr
    kind: AgentBundleObjectKind
    schema_id: StrictStr = Field(max_length=256)
    byte_count: StrictInt = Field(ge=0, le=AGENT_BUNDLE_MAX_OBJECT_BYTES)

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("schema_id")
    @classmethod
    def validate_schema(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    def identity_material(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    @model_validator(mode="after")
    def validate_kind_size(self) -> AgentBundleObjectRef:
        if self.byte_count == 0 and self.kind is not AgentBundleObjectKind.COMPONENT_BLOB:
            raise ValueError("Portable bundle objects cannot be empty.")
        if self.kind is AgentBundleObjectKind.SNAPSHOT_DOCUMENT and (
            self.byte_count > AGENT_SNAPSHOT_MAX_BYTES
        ):
            raise ValueError("Snapshot document exceeds its portable byte limit.")
        if (
            self.kind
            in {
                AgentBundleObjectKind.SNAPSHOT_NODE,
                AgentBundleObjectKind.COMPONENT_MANIFEST,
                AgentBundleObjectKind.OPERATION_RECEIPT,
            }
            and self.byte_count > AGENT_BUNDLE_MAX_INDEX_BYTES
        ):
            raise ValueError("Structured bundle object exceeds its byte limit.")
        return self


class AgentBundleSizeReport(_BundleModel):
    root_manifest_bytes: StrictInt = Field(ge=1)
    logical_closure_bytes: StrictInt = Field(ge=1)
    unique_stored_bytes: StrictInt = Field(ge=0)
    shared_stored_bytes: StrictInt = Field(ge=0)
    incremental_transfer_bytes: StrictInt = Field(ge=0)
    materialized_disk_bytes: StrictInt = Field(ge=0)
    unresolved_external_bindings: tuple[StrictStr, ...] = ()

    @field_validator("unresolved_external_bindings", mode="before")
    @classmethod
    def validate_unresolved(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, list | tuple):
            raise ValueError("unresolved_external_bindings must be an ordered array.")
        copied = tuple(
            _clean(item, f"unresolved_external_bindings[{index}]", max_chars=512)
            if type(item) is str
            else _raise_string("unresolved_external_bindings", index)
            for index, item in enumerate(value)
        )
        if copied != tuple(sorted(set(copied))):
            raise ValueError("unresolved_external_bindings must be unique and sorted.")
        return copied


class AgentBundle(_BundleModel):
    record_type: Literal["cayu.agent-bundle"] = AGENT_BUNDLE_RECORD_TYPE
    schema_version: Literal[1] = AGENT_BUNDLE_SCHEMA_VERSION
    bundle_id: StrictStr
    snapshot_ref: AgentSnapshotRef
    export_binding_id: StrictStr
    export_authority_scope_fingerprint: StrictStr
    destination_inventory_fingerprint: StrictStr
    profile: AgentSnapshotProfile
    mode: AgentBundleMode
    snapshot_document: AgentBundleObjectRef
    closure: tuple[AgentBundleObjectRef, ...]
    transferred_digests: tuple[StrictStr, ...]
    external_bindings: tuple[AgentExternalBindingRequirement, ...]
    size_report: AgentBundleSizeReport

    @field_validator(
        "bundle_id",
        "export_binding_id",
        "export_authority_scope_fingerprint",
        "destination_inventory_fingerprint",
    )
    @classmethod
    def validate_bundle_id(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("closure", mode="before")
    @classmethod
    def validate_closure_array(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError("closure must be an ordered array.")
        return value

    @field_validator("transferred_digests", mode="before")
    @classmethod
    def validate_transferred(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, list | tuple):
            raise ValueError("transferred_digests must be an ordered array.")
        copied = tuple(
            _sha256_hex(item, "transferred_digests")
            if type(item) is str
            else _raise_string("transferred_digests", index)
            for index, item in enumerate(value)
        )
        if copied != tuple(sorted(set(copied))):
            raise ValueError("transferred_digests must be unique and sorted.")
        return copied

    @field_validator("external_bindings", mode="before")
    @classmethod
    def validate_bindings_array(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError("external_bindings must be an ordered array.")
        return value

    @model_validator(mode="after")
    def validate_bundle(self) -> AgentBundle:
        keys = tuple((item.digest, item.kind.value, item.schema_id) for item in self.closure)
        if not self.closure or keys != tuple(sorted(set(keys))):
            raise ValueError("Bundle closure must be nonempty, unique, and canonical.")
        if len(self.closure) > AGENT_BUNDLE_MAX_OBJECTS:
            raise ValueError("Bundle closure exceeds its object-count limit.")
        by_digest: dict[str, AgentBundleObjectRef] = {}
        for item in self.closure:
            existing = by_digest.get(item.digest)
            if existing is not None and existing != item:
                raise ValueError("One digest cannot name conflicting bundle object metadata.")
            by_digest[item.digest] = item
        if by_digest.get(self.snapshot_document.digest) != self.snapshot_document:
            raise ValueError("Snapshot document must be present in the closure.")
        if self.snapshot_document.kind is not AgentBundleObjectKind.SNAPSHOT_DOCUMENT:
            raise ValueError("snapshot_document must name the snapshot-document object kind.")
        if sum(item.kind is AgentBundleObjectKind.SNAPSHOT_DOCUMENT for item in self.closure) != 1:
            raise ValueError("A bundle must contain exactly one snapshot document.")
        root_nodes = tuple(
            item
            for item in self.closure
            if item.kind is AgentBundleObjectKind.SNAPSHOT_NODE
            and item.digest == self.snapshot_ref.snapshot_root
            and item.schema_id == "cayu.agent-snapshot.manifest.v3"
        )
        if len(root_nodes) != 1:
            raise ValueError("A bundle must contain its exact root manifest node once.")
        if any(item.kind is AgentBundleObjectKind.OPERATION_RECEIPT for item in self.closure):
            raise ValueError("Operation receipts are not part of portable agent state.")
        if not set(self.transferred_digests).issubset(by_digest):
            raise ValueError("Transferred objects must belong to the declared closure.")
        if self.snapshot_document.digest not in self.transferred_digests:
            raise ValueError("Every bundle must physically transfer its snapshot document.")
        if self.mode is AgentBundleMode.FULL and set(self.transferred_digests) != set(by_digest):
            raise ValueError("A full bundle must transfer its complete closure.")
        binding_keys = tuple(
            (binding.kind.value, binding.name, binding.requirement_fingerprint)
            for binding in self.external_bindings
        )
        if binding_keys != tuple(sorted(set(binding_keys))):
            raise ValueError("Bundle external bindings must be unique and canonical.")
        logical_bytes = sum(item.byte_count for item in self.closure)
        transferred_bytes = sum(by_digest[digest].byte_count for digest in self.transferred_digests)
        if logical_bytes > AGENT_BUNDLE_MAX_TOTAL_BYTES:
            raise ValueError("Bundle logical closure exceeds its total-byte limit.")
        if (
            self.size_report.root_manifest_bytes != root_nodes[0].byte_count
            or self.size_report.logical_closure_bytes != logical_bytes
            or self.size_report.incremental_transfer_bytes != transferred_bytes
            or self.size_report.unique_stored_bytes != transferred_bytes
            or self.size_report.shared_stored_bytes != logical_bytes - transferred_bytes
        ):
            raise ValueError("Bundle size report contradicts its object inventory.")
        expected_unresolved = tuple(
            sorted(
                f"{item.kind.value}:{item.name}:{item.requirement_fingerprint}"
                for item in self.external_bindings
                if item.required
            )
        )
        if self.size_report.unresolved_external_bindings != expected_unresolved:
            raise ValueError("Bundle size report contradicts its external bindings.")
        if self.bundle_id != _content_digest(self.identity_material(), "agent_bundle"):
            raise ValueError("AgentBundle bundle_id does not match its canonical index.")
        return self

    def identity_material(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"bundle_id"})

    @classmethod
    def create(
        cls,
        *,
        snapshot_ref: AgentSnapshotRef,
        export_binding_id: str,
        export_authority_scope_fingerprint: str,
        destination_inventory_fingerprint: str,
        profile: AgentSnapshotProfile,
        mode: AgentBundleMode,
        snapshot_document: AgentBundleObjectRef,
        closure: Iterable[AgentBundleObjectRef],
        transferred_digests: Iterable[str],
        external_bindings: Iterable[AgentExternalBindingRequirement],
        size_report: AgentBundleSizeReport,
    ) -> AgentBundle:
        values: dict[str, Any] = {
            "snapshot_ref": snapshot_ref,
            "export_binding_id": export_binding_id,
            "export_authority_scope_fingerprint": export_authority_scope_fingerprint,
            "destination_inventory_fingerprint": destination_inventory_fingerprint,
            "profile": profile,
            "mode": mode,
            "snapshot_document": snapshot_document,
            "closure": tuple(
                sorted(closure, key=lambda item: (item.digest, item.kind.value, item.schema_id))
            ),
            "transferred_digests": tuple(sorted(set(transferred_digests))),
            "external_bindings": tuple(
                sorted(
                    external_bindings,
                    key=lambda item: (
                        item.kind.value,
                        item.name,
                        item.requirement_fingerprint,
                    ),
                )
            ),
            "size_report": size_report,
        }
        provisional = cls.model_construct(bundle_id="0" * 64, **values)
        return cls(
            bundle_id=_content_digest(provisional.identity_material(), "agent_bundle"),
            **values,
        )


class AgentBundleInventory(_BundleModel):
    object_digests: tuple[StrictStr, ...] = ()

    @field_validator("object_digests", mode="before")
    @classmethod
    def validate_digests(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, list | tuple):
            raise ValueError("object_digests must be an ordered array.")
        copied = tuple(
            _sha256_hex(item, "object_digests")
            if type(item) is str
            else _raise_string("object_digests", index)
            for index, item in enumerate(value)
        )
        if copied != tuple(sorted(set(copied))):
            raise ValueError("object_digests must be unique and sorted.")
        return copied

    @property
    def fingerprint(self) -> str:
        return _content_digest(
            {"object_digests": list(self.object_digests)},
            "agent_bundle_inventory",
        )


class AgentBundleError(RuntimeError):
    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = _clean(code, "code", max_chars=256)
        self.detail = None if detail is None else _clean(detail, "detail", max_chars=512)
        super().__init__(self.code if self.detail is None else f"{self.code}: {self.detail}")


class AgentSnapshotObjectStore(ABC):
    """Content-addressed provider-object storage composed with AgentSnapshotStore."""

    @abstractmethod
    async def put(self, reference: AgentBundleObjectRef, content: bytes) -> None:
        pass

    @abstractmethod
    async def get(self, reference: AgentBundleObjectRef) -> bytes | None:
        pass

    @abstractmethod
    async def has_digest(self, digest: str) -> bool:
        pass

    @abstractmethod
    async def get_digest(self, digest: str, *, max_bytes: int) -> bytes | None:
        """Load one bounded content-addressed object when metadata is discovered later."""

    @abstractmethod
    async def verify(self, reference: AgentBundleObjectRef) -> bool:
        pass

    @abstractmethod
    async def put_file(self, reference: AgentBundleObjectRef, source: str | Path) -> None:
        """Stream one regular file into immutable object storage."""

    @abstractmethod
    async def copy_to(self, reference: AgentBundleObjectRef, destination: str | Path) -> None:
        """Stream one verified immutable object to a new regular file."""

    @abstractmethod
    async def contains_secret(
        self,
        reference: AgentBundleObjectRef,
        redactor: SecretRedactor,
    ) -> bool:
        pass

    @abstractmethod
    async def inventory(self) -> AgentBundleInventory:
        pass


class FileSystemAgentSnapshotObjectStore(AgentSnapshotObjectStore):
    """Ordinary-filesystem CAS with atomic immutable-object publication."""

    def __init__(self, root: str | Path) -> None:
        path = Path(root)
        if not path.is_absolute():
            raise ValueError("Agent snapshot object-store root must be absolute.")
        self.root = path
        self._lock = threading.RLock()
        (self.root / AGENT_BUNDLE_OBJECT_DIRECTORY).mkdir(parents=True, exist_ok=True)

    async def put(self, reference: AgentBundleObjectRef, content: bytes) -> None:
        if type(reference) is not AgentBundleObjectRef:
            raise TypeError("reference must be an AgentBundleObjectRef.")
        if type(content) is not bytes:
            raise TypeError("content must be bytes.")
        await asyncio.to_thread(self._put_sync, reference, content)

    def _put_sync(self, reference: AgentBundleObjectRef, content: bytes) -> None:
        validated = AgentBundleObjectRef.model_validate(reference.model_dump(mode="json"))
        if len(content) != validated.byte_count or _digest(content) != validated.digest:
            raise AgentBundleError("object_integrity_mismatch")
        with (
            self._lock,
            cooperative_path_lock(
                self.root,
                validated.digest,
                lock_directory_name="cayu-agent-snapshot-object-locks",
            ),
            _open_object_shard(self.root, validated.digest, create=True) as opened,
        ):
            assert opened is not None
            shard_fd, name = opened
            try:
                existing, _ = _read_regular_at(
                    shard_fd,
                    name,
                    max_bytes=validated.byte_count,
                )
            except FileNotFoundError:
                pass
            else:
                if existing != content:
                    raise AgentBundleError("object_collision")
                return
            temporary = f".{name}.{os.getpid()}.tmp"
            try:
                temporary_info = os.stat(
                    temporary,
                    dir_fd=shard_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISREG(temporary_info.st_mode):
                    raise AgentBundleError("object_staging_conflict")
                os.unlink(temporary, dir_fd=shard_fd)
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(temporary, flags, 0o600, dir_fd=shard_fd)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.rename(
                    temporary,
                    name,
                    src_dir_fd=shard_fd,
                    dst_dir_fd=shard_fd,
                )
            except BaseException:
                with suppress(FileNotFoundError):
                    os.unlink(temporary, dir_fd=shard_fd)
                raise

    async def get(self, reference: AgentBundleObjectRef) -> bytes | None:
        if type(reference) is not AgentBundleObjectRef:
            raise TypeError("reference must be an AgentBundleObjectRef.")
        return await asyncio.to_thread(self._get_sync, reference)

    def _get_sync(self, reference: AgentBundleObjectRef) -> bytes | None:
        validated = AgentBundleObjectRef.model_validate(reference.model_dump(mode="json"))
        with _open_regular_object(self.root, validated.digest) as opened:
            if opened is None:
                return None
            stream, size = opened
            if size > validated.byte_count:
                raise AgentBundleError("stored_object_integrity_mismatch")
            content = stream.read(validated.byte_count + 1)
        if len(content) != validated.byte_count or _digest(content) != validated.digest:
            raise AgentBundleError("stored_object_integrity_mismatch")
        return content

    async def has_digest(self, digest: str) -> bool:
        _sha256_hex(digest, "digest")
        return await asyncio.to_thread(self._has_sync, digest)

    async def get_digest(self, digest: str, *, max_bytes: int) -> bytes | None:
        _sha256_hex(digest, "digest")
        if type(max_bytes) is not int or isinstance(max_bytes, bool) or max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer.")
        return await asyncio.to_thread(self._get_digest_sync, digest, max_bytes)

    def _get_digest_sync(self, digest: str, max_bytes: int) -> bytes | None:
        with _open_regular_object(self.root, digest) as opened:
            if opened is None:
                return None
            stream, size = opened
            if size > max_bytes:
                raise AgentBundleError("object_size_limit_exceeded")
            content = stream.read(max_bytes + 1)
        if len(content) != size or _digest(content) != digest:
            raise AgentBundleError("stored_object_integrity_mismatch")
        return content

    async def verify(self, reference: AgentBundleObjectRef) -> bool:
        if type(reference) is not AgentBundleObjectRef:
            raise TypeError("reference must be an AgentBundleObjectRef.")
        return await asyncio.to_thread(self._verify_sync, reference)

    def _verify_sync(self, reference: AgentBundleObjectRef) -> bool:
        with _open_regular_object(self.root, reference.digest) as opened:
            if opened is None:
                return False
            stream, size = opened
            if size > reference.byte_count:
                return False
            hasher = sha256()
            byte_count = 0
            while chunk := stream.read(1024 * 1024):
                byte_count += len(chunk)
                if byte_count > reference.byte_count:
                    return False
                hasher.update(chunk)
        return hasher.hexdigest() == reference.digest and byte_count == reference.byte_count

    async def put_file(self, reference: AgentBundleObjectRef, source: str | Path) -> None:
        if type(reference) is not AgentBundleObjectRef:
            raise TypeError("reference must be an AgentBundleObjectRef.")
        source_path = Path(source)
        if not source_path.is_absolute():
            raise ValueError("Object source path must be absolute.")
        await asyncio.to_thread(self._put_file_sync, reference, source_path)

    def _put_file_sync(self, reference: AgentBundleObjectRef, source: Path) -> None:
        source_digest, source_size = _file_digest_and_size(
            source,
            max_bytes=reference.byte_count,
        )
        if source_digest != reference.digest or source_size != reference.byte_count:
            raise AgentBundleError("object_integrity_mismatch")
        with (
            self._lock,
            cooperative_path_lock(
                self.root,
                reference.digest,
                lock_directory_name="cayu-agent-snapshot-object-locks",
            ),
            _open_object_shard(self.root, reference.digest, create=True) as opened,
        ):
            assert opened is not None
            shard_fd, name = opened
            try:
                existing, _ = _read_regular_at(
                    shard_fd,
                    name,
                    max_bytes=reference.byte_count,
                )
            except FileNotFoundError:
                pass
            else:
                if len(existing) != reference.byte_count or _digest(existing) != reference.digest:
                    raise AgentBundleError("object_collision")
                return
            temporary = f".{name}.{os.getpid()}.tmp"
            try:
                temporary_info = os.stat(
                    temporary,
                    dir_fd=shard_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISREG(temporary_info.st_mode):
                    raise AgentBundleError("object_staging_conflict")
                os.unlink(temporary, dir_fd=shard_fd)
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(temporary, flags, 0o600, dir_fd=shard_fd)
            try:
                with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output:
                    while chunk := input_stream.read(1024 * 1024):
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                copied, copied_size = _read_regular_at(
                    shard_fd,
                    temporary,
                    max_bytes=reference.byte_count,
                )
                if _digest(copied) != reference.digest or copied_size != reference.byte_count:
                    raise AgentBundleError("object_changed_during_copy")
                os.rename(
                    temporary,
                    name,
                    src_dir_fd=shard_fd,
                    dst_dir_fd=shard_fd,
                )
            except BaseException:
                with suppress(FileNotFoundError):
                    os.unlink(temporary, dir_fd=shard_fd)
                raise

    async def copy_to(
        self,
        reference: AgentBundleObjectRef,
        destination: str | Path,
    ) -> None:
        if type(reference) is not AgentBundleObjectRef:
            raise TypeError("reference must be an AgentBundleObjectRef.")
        destination_path = Path(destination)
        if not destination_path.is_absolute():
            raise ValueError("Object destination path must be absolute.")
        await asyncio.to_thread(self._copy_to_sync, reference, destination_path)

    def _copy_to_sync(self, reference: AgentBundleObjectRef, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _open_regular_object(self.root, reference.digest) as opened:
            if opened is None:
                raise AgentBundleError("stored_object_integrity_mismatch")
            input_stream, source_size = opened
            if source_size != reference.byte_count:
                raise AgentBundleError("stored_object_integrity_mismatch")
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                hasher = sha256()
                byte_count = 0
                with os.fdopen(descriptor, "wb") as output:
                    while chunk := input_stream.read(1024 * 1024):
                        byte_count += len(chunk)
                        if byte_count > reference.byte_count:
                            raise AgentBundleError("stored_object_integrity_mismatch")
                        hasher.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if hasher.hexdigest() != reference.digest or byte_count != reference.byte_count:
                    raise AgentBundleError("stored_object_integrity_mismatch")
            except BaseException:
                destination.unlink(missing_ok=True)
                raise

    async def contains_secret(
        self,
        reference: AgentBundleObjectRef,
        redactor: SecretRedactor,
    ) -> bool:
        if type(reference) is not AgentBundleObjectRef:
            raise TypeError("reference must be an AgentBundleObjectRef.")
        if not isinstance(redactor, SecretRedactor):
            raise TypeError("redactor must be a SecretRedactor.")
        return await asyncio.to_thread(self._contains_secret_sync, reference, redactor)

    def _contains_secret_sync(
        self,
        reference: AgentBundleObjectRef,
        redactor: SecretRedactor,
    ) -> bool:
        with _open_regular_object(self.root, reference.digest) as opened:
            if opened is None:
                raise AgentBundleError("stored_object_integrity_mismatch")
            stream, size = opened
            if size != reference.byte_count:
                raise AgentBundleError("stored_object_integrity_mismatch")
            overlap = max(0, redactor.max_secret_utf8_bytes - 1)
            tail = b""
            contains = False
            hasher = sha256()
            byte_count = 0
            while chunk := stream.read(1024 * 1024):
                byte_count += len(chunk)
                if byte_count > reference.byte_count:
                    raise AgentBundleError("object_changed_during_read")
                hasher.update(chunk)
                window = tail + chunk
                contains = contains or redactor.contains_secret_bytes(window)
                tail = window[-overlap:] if overlap else b""
        if hasher.hexdigest() != reference.digest or byte_count != reference.byte_count:
            raise AgentBundleError("object_changed_during_read")
        return contains

    def _has_sync(self, digest: str) -> bool:
        with _open_object_shard(self.root, digest, create=False) as opened:
            if opened is None:
                return False
            shard_fd, name = opened
            try:
                info = os.stat(name, dir_fd=shard_fd, follow_symlinks=False)
            except FileNotFoundError:
                return False
            if stat.S_ISLNK(info.st_mode):
                raise AgentBundleError("object_path_not_regular")
            return stat.S_ISREG(info.st_mode)

    async def inventory(self) -> AgentBundleInventory:
        return await asyncio.to_thread(self._inventory_sync)

    def _inventory_sync(self) -> AgentBundleInventory:
        base = self.root / AGENT_BUNDLE_OBJECT_DIRECTORY
        digests: list[str] = []
        for prefix in sorted(base.iterdir() if base.exists() else ()):
            if prefix.is_symlink() or not prefix.is_dir() or len(prefix.name) != 2:
                continue
            for entry in sorted(prefix.iterdir()):
                digest = prefix.name + entry.name
                if (
                    not entry.is_symlink()
                    and entry.is_file()
                    and len(entry.name) == 62
                    and len(digest) == 64
                    and all(character in _SHA256_CHARS for character in digest)
                ):
                    digests.append(digest)
        return AgentBundleInventory(object_digests=tuple(sorted(set(digests))))


def agent_snapshot_component_package(
    *,
    kind: AgentSnapshotComponentKind,
    provider_id: str,
    component_schema: str,
    profile: AgentSnapshotProfile,
    payload: dict[str, object],
    files: Mapping[str, bytes] | None = None,
    file_content_types: Mapping[str, str] | None = None,
    executable_paths: Iterable[str] = (),
    external_bindings: Iterable[AgentExternalBindingRequirement] = (),
    session_disposition: AgentSnapshotSessionDisposition | None = None,
) -> tuple[AgentSnapshotComponentPackage, dict[str, bytes]]:
    """Build a strict portable component and its deduplicated blob objects."""

    file_values = {} if files is None else dict(files)
    content_types = {} if file_content_types is None else dict(file_content_types)
    executables = {_relative_path(path, "executable_paths") for path in executable_paths}
    if not executables.issubset(file_values):
        raise ValueError("Every executable path must name a component file.")
    blobs: dict[str, bytes] = {}
    refs: list[AgentSnapshotComponentFile] = []
    for path, content in sorted(file_values.items()):
        normalized = _relative_path(path, "files.path")
        if type(content) is not bytes:
            raise TypeError("Component file contents must be bytes.")
        digest = _digest(content)
        existing = blobs.get(digest)
        if existing is not None and existing != content:
            raise AgentBundleError("blob_digest_collision")
        blobs[digest] = content
        refs.append(
            AgentSnapshotComponentFile(
                path=normalized,
                digest=digest,
                byte_count=len(content),
                content_type=content_types.get(normalized, "application/octet-stream"),
                executable=normalized in executables,
            )
        )
    package = AgentSnapshotComponentPackage(
        kind=kind,
        provider_id=provider_id,
        component_schema=component_schema,
        profile=profile,
        session_disposition=session_disposition,
        payload=payload,
        files=tuple(refs),
        external_bindings=tuple(
            sorted(
                external_bindings,
                key=lambda item: (item.kind.value, item.name, item.requirement_fingerprint),
            )
        ),
    )
    return package, blobs


async def store_agent_snapshot_component_package(
    *,
    kind: AgentSnapshotComponentKind,
    provider_id: str,
    component_schema: str,
    profile: AgentSnapshotProfile,
    payload: dict[str, object],
    file_paths: Mapping[str, str | Path],
    object_store: AgentSnapshotObjectStore,
    file_content_types: Mapping[str, str] | None = None,
    executable_paths: Iterable[str] = (),
    external_bindings: Iterable[AgentExternalBindingRequirement] = (),
    session_disposition: AgentSnapshotSessionDisposition | None = None,
    secret_redactor: SecretRedactor | None = None,
) -> AgentSnapshotComponentPackage:
    """Stream regular files into CAS and return their portable component manifest."""

    content_types = {} if file_content_types is None else dict(file_content_types)
    executables = {_relative_path(path, "executable_paths") for path in executable_paths}
    normalized_sources: dict[str, Path] = {}
    refs: list[AgentSnapshotComponentFile] = []
    for logical_path, source in sorted(file_paths.items()):
        normalized = _relative_path(logical_path, "file_paths.path")
        source_path = Path(source)
        if not source_path.is_absolute():
            raise ValueError("Component source paths must be absolute.")
        digest, byte_count = await asyncio.to_thread(
            _file_digest_and_size,
            source_path,
            max_bytes=AGENT_BUNDLE_MAX_OBJECT_BYTES,
        )
        if secret_redactor is not None and await asyncio.to_thread(
            _file_contains_secret,
            source_path,
            secret_redactor,
        ):
            raise AgentBundleError("component_contains_secret")
        normalized_sources[normalized] = source_path
        refs.append(
            AgentSnapshotComponentFile(
                path=normalized,
                digest=digest,
                byte_count=byte_count,
                content_type=content_types.get(normalized, "application/octet-stream"),
                executable=normalized in executables,
            )
        )
    if not executables.issubset(normalized_sources):
        raise ValueError("Every executable path must name a component file.")
    package = AgentSnapshotComponentPackage(
        kind=kind,
        provider_id=provider_id,
        component_schema=component_schema,
        profile=profile,
        session_disposition=session_disposition,
        payload=payload,
        files=tuple(refs),
        external_bindings=tuple(
            sorted(
                external_bindings,
                key=lambda item: (item.kind.value, item.name, item.requirement_fingerprint),
            )
        ),
    )
    if secret_redactor is not None and secret_redactor.contains_secret_bytes(package.document):
        raise AgentBundleError("component_contains_secret")
    for file in package.files:
        reference = AgentBundleObjectRef(
            digest=file.digest,
            kind=AgentBundleObjectKind.COMPONENT_BLOB,
            schema_id="cayu.agent-snapshot.component-blob.v1",
            byte_count=file.byte_count,
        )
        await object_store.put_file(reference, normalized_sources[file.path])
    await object_store.put(
        _object_ref(
            package.document,
            kind=AgentBundleObjectKind.COMPONENT_MANIFEST,
            schema_id=component_schema,
        ),
        package.document,
    )
    return package


def _object_ref(
    content: bytes,
    *,
    kind: AgentBundleObjectKind,
    schema_id: str,
) -> AgentBundleObjectRef:
    return AgentBundleObjectRef(
        digest=_digest(content),
        kind=kind,
        schema_id=schema_id,
        byte_count=len(content),
    )


class PortableAgentSnapshotComponentProvider(AgentSnapshotComponentProvider):
    """Production adapter for immutable provider-owned component packages.

    Owning subsystems construct the package from an exact safe frontier.  This
    adapter supplies durable content storage, verification and idempotent
    filesystem materialization; it does not infer which application data is
    authorized to inherit.
    """

    def __init__(
        self,
        package: AgentSnapshotComponentPackage,
        blobs: Mapping[str, bytes],
        *,
        object_store: AgentSnapshotObjectStore,
        materialization_root: str | Path,
        execution_profile: AgentSnapshotExecutionProfileRef | None = None,
        consistency: AgentSnapshotConsistency = AgentSnapshotConsistency.FRONTIER_CONSISTENT,
        redaction: AgentSnapshotRedaction = AgentSnapshotRedaction.BOUNDED_PROJECTION,
        materialization: AgentSnapshotMaterializationCapability = (
            AgentSnapshotMaterializationCapability.RESTORABLE
        ),
        revision: str | None = None,
        frontier: str | None = None,
        secret_redactor: SecretRedactor | None = None,
    ) -> None:
        if type(package) is not AgentSnapshotComponentPackage:
            raise TypeError("package must be an AgentSnapshotComponentPackage.")
        if package.kind is AgentSnapshotComponentKind.EXECUTION_PROFILE and (
            execution_profile is None
        ):
            raise ValueError("Execution-profile packages require their typed identity.")
        if package.kind is not AgentSnapshotComponentKind.EXECUTION_PROFILE and (
            execution_profile is not None
        ):
            raise ValueError("Only execution-profile packages carry their typed identity.")
        root = Path(materialization_root)
        if not root.is_absolute():
            raise ValueError("materialization_root must be absolute.")
        self.package = package
        self.blobs = dict(blobs)
        self.object_store = object_store
        self.materialization_root = root
        self.execution_profile = execution_profile
        self.consistency = consistency
        self.redaction = redaction
        self.materialization_capability = materialization
        self.revision = revision or f"component:{package.digest}"
        self.frontier = frontier
        if secret_redactor is not None and not isinstance(secret_redactor, SecretRedactor):
            raise TypeError("secret_redactor must be a SecretRedactor.")
        self.secret_redactor = secret_redactor
        self.kind = package.kind
        self.provider_id = package.provider_id
        self._locks: dict[str, asyncio.Lock] = {}

    async def _ensure_stored(self) -> None:
        if self.secret_redactor is not None and self.secret_redactor.contains_secret_bytes(
            self.package.document
        ):
            raise AgentBundleError("component_contains_secret")
        package_ref = _object_ref(
            self.package.document,
            kind=AgentBundleObjectKind.COMPONENT_MANIFEST,
            schema_id=self.package.component_schema,
        )
        if package_ref.digest != self.package.digest:
            raise AgentBundleError("component_identity_mismatch")
        await self.object_store.put(package_ref, self.package.document)
        declared = {file.digest: file for file in self.package.files}
        if self.blobs and set(declared) != set(self.blobs):
            raise AgentBundleError("component_blob_set_mismatch")
        for digest, file in sorted(declared.items()):
            reference = AgentBundleObjectRef(
                digest=digest,
                kind=AgentBundleObjectKind.COMPONENT_BLOB,
                schema_id="cayu.agent-snapshot.component-blob.v1",
                byte_count=file.byte_count,
            )
            content = self.blobs.get(digest)
            if content is None:
                if not await self.object_store.verify(reference):
                    raise AgentBundleError("component_blob_missing")
                if self.secret_redactor is not None and await self.object_store.contains_secret(
                    reference,
                    self.secret_redactor,
                ):
                    raise AgentBundleError("component_contains_secret")
                continue
            if self.secret_redactor is not None and self.secret_redactor.contains_secret_bytes(
                content
            ):
                raise AgentBundleError("component_contains_secret")
            await self.object_store.put(reference, content)

    async def capture(
        self,
        request: AgentSnapshotCaptureRequest,
        selector: AgentSnapshotComponentSelector,
    ) -> AgentSnapshotComponentCapture:
        if selector.kind is not self.kind:
            raise AgentBundleError("component_kind_mismatch")
        await self._ensure_stored()
        component = AgentSnapshotComponentRef(
            kind=self.kind,
            provider_id=self.provider_id,
            logical=AgentSnapshotLogicalRef(
                fingerprint=self.package.digest,
                revision=self.revision,
                frontier=self.frontier,
                scope_fingerprint=request.authority_scope_fingerprint,
                source_ref=f"cayu-ref:component:{self.package.digest}",
            ),
            consistency=self.consistency,
            completeness=AgentSnapshotCompleteness.COMPLETE,
            redaction=self.redaction,
            materialization=self.materialization_capability,
            required=selector.required,
        )
        return AgentSnapshotComponentCapture(
            component=component,
            execution_profile=self.execution_profile,
        )

    async def verify(
        self,
        snapshot: AgentSnapshot,
        component: AgentSnapshotComponentRef,
    ) -> bool:
        if (
            component.kind is not self.kind
            or component.provider_id != self.provider_id
            or component.logical.fingerprint != self.package.digest
        ):
            return False
        package_ref = _object_ref(
            self.package.document,
            kind=AgentBundleObjectKind.COMPONENT_MANIFEST,
            schema_id=self.package.component_schema,
        )
        stored = await self.object_store.get(package_ref)
        if stored != self.package.document:
            return False
        for file in self.package.files:
            reference = AgentBundleObjectRef(
                digest=file.digest,
                kind=AgentBundleObjectKind.COMPONENT_BLOB,
                schema_id="cayu.agent-snapshot.component-blob.v1",
                byte_count=file.byte_count,
            )
            if not await self.object_store.verify(reference):
                return False
        return True

    async def materialize(
        self,
        snapshot: AgentSnapshot,
        component: AgentSnapshotComponentRef,
        request: AgentSnapshotMaterializationRequest,
        operation: AgentSnapshotMaterializationOperation,
    ) -> AgentSnapshotMaterializedComponent:
        return await self._materialize_once(component, request, operation)

    async def recover_materialization_operation(
        self,
        snapshot: AgentSnapshot,
        component: AgentSnapshotComponentRef,
        request: AgentSnapshotMaterializationRequest,
        operation: AgentSnapshotMaterializationOperation,
    ) -> AgentSnapshotMaterializedComponent:
        return await self._materialize_once(component, request, operation)

    async def recover(
        self,
        snapshot: AgentSnapshot,
        component: AgentSnapshotComponentRef,
        materialized: AgentSnapshotMaterializedComponent,
        materialization: AgentSnapshotMaterialization,
    ) -> AgentSnapshotMaterializedComponent:
        if (
            materialization.snapshot_fingerprint != snapshot.snapshot_root
            or materialized.kind is not component.kind
            or materialized.baseline_fingerprint != component.logical.fingerprint
            or materialized.capability is not component.materialization
        ):
            raise AgentBundleError("materialization_recovery_identity_mismatch")
        operation_material: dict[str, object] = {
            "record_type": "cayu.agent-snapshot-materialization-operation",
            "schema_version": 1,
            "snapshot_fingerprint": materialization.snapshot_fingerprint,
            "candidate_id": materialization.candidate_id,
            "state_scope_id": materialization.state_scope_id,
            "state_mode": materialization.state_mode.value,
            "component_kind": component.kind.value,
            "provider_id": component.provider_id,
            "baseline_fingerprint": component.logical.fingerprint,
            "capability": component.materialization.value,
        }
        if materialization.state_partition_fingerprint is not None:
            operation_material["state_partition_fingerprint"] = (
                materialization.state_partition_fingerprint
            )
        operation_id = _content_digest(
            operation_material,
            "snapshot_materialization_operation",
        )
        await asyncio.to_thread(
            self._recover_sync,
            component,
            materialization.state_scope_id,
            operation_id,
        )
        return materialized

    def _recover_sync(
        self,
        component: AgentSnapshotComponentRef,
        state_scope_id: str,
        operation_id: str,
    ) -> None:
        if self.materialization_root.is_symlink() or not self.materialization_root.is_dir():
            raise AgentBundleError("materialization_recovery_destination_missing")
        with cooperative_path_lock(
            self.materialization_root,
            f"{state_scope_id}/{self.kind.value}",
            lock_directory_name="cayu-agent-snapshot-materialization-locks",
        ):
            destination = self.materialization_root / state_scope_id / self.kind.value
            self._verify_materialized_destination(
                destination,
                component=component,
                state_scope_id=state_scope_id,
                operation_id=operation_id,
            )

    async def _materialize_once(
        self,
        component: AgentSnapshotComponentRef,
        request: AgentSnapshotMaterializationRequest,
        operation: AgentSnapshotMaterializationOperation,
    ) -> AgentSnapshotMaterializedComponent:
        lock = self._locks.setdefault(operation.operation_id, asyncio.Lock())
        async with lock:
            return await asyncio.to_thread(
                self._materialize_sync,
                component,
                request,
                operation,
            )

    def _materialize_sync(
        self,
        component: AgentSnapshotComponentRef,
        request: AgentSnapshotMaterializationRequest,
        operation: AgentSnapshotMaterializationOperation,
    ) -> AgentSnapshotMaterializedComponent:
        self.materialization_root.mkdir(parents=True, exist_ok=True)
        with cooperative_path_lock(
            self.materialization_root,
            f"{request.state_scope_id}/{self.kind.value}",
            lock_directory_name="cayu-agent-snapshot-materialization-locks",
        ):
            return self._materialize_locked(component, request, operation)

    def _materialize_locked(
        self,
        component: AgentSnapshotComponentRef,
        request: AgentSnapshotMaterializationRequest,
        operation: AgentSnapshotMaterializationOperation,
    ) -> AgentSnapshotMaterializedComponent:
        destination = self.materialization_root / request.state_scope_id / self.kind.value
        receipt_path = destination / ".cayu-materialization.json"
        expected = {
            "operation_id": operation.operation_id,
            "component_digest": component.logical.fingerprint,
            "state_scope_id": request.state_scope_id,
        }
        if receipt_path.exists():
            self._verify_materialized_destination(
                destination,
                component=component,
                state_scope_id=request.state_scope_id,
                operation_id=operation.operation_id,
            )
            return self._materialized_component(component, request, destination)
        temporary = destination.parent / f".{destination.name}.{operation.operation_id}.tmp"
        if destination.exists():
            raise AgentBundleError("materialization_outcome_unknown")
        if temporary.exists():
            if temporary.is_symlink() or not temporary.is_dir():
                raise AgentBundleError("materialization_staging_conflict")
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        try:
            for file in self.package.files:
                target = temporary.joinpath(*PurePosixPath(file.path).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                content = self.blobs.get(file.digest)
                if content is not None:
                    descriptor = os.open(
                        target,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o700 if file.executable else 0o600,
                    )
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(content)
                        stream.flush()
                        os.fsync(stream.fileno())
                else:
                    reference = AgentBundleObjectRef(
                        digest=file.digest,
                        kind=AgentBundleObjectKind.COMPONENT_BLOB,
                        schema_id="cayu.agent-snapshot.component-blob.v1",
                        byte_count=file.byte_count,
                    )
                    asyncio.run(self.object_store.copy_to(reference, target))
                    target.chmod(0o700 if file.executable else 0o600)
            receipt = temporary / receipt_path.name
            receipt.write_bytes(canonical_durable_json_bytes(expected, "materialization_receipt"))
            os.replace(temporary, destination)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return self._materialized_component(component, request, destination)

    def _verify_materialized_destination(
        self,
        destination: Path,
        *,
        component: AgentSnapshotComponentRef,
        state_scope_id: str,
        operation_id: str,
    ) -> None:
        if destination.is_symlink() or not destination.is_dir():
            raise AgentBundleError("materialization_recovery_destination_missing")
        receipt_name = ".cayu-materialization.json"
        expected_paths = {file.path for file in self.package.files} | {receipt_name}
        observed_paths: set[str] = set()
        for current, directories, filenames in os.walk(destination, followlinks=False):
            current_path = Path(current)
            for directory in directories:
                child = current_path / directory
                if child.is_symlink() or not child.is_dir():
                    raise AgentBundleError("materialization_path_not_regular")
            for filename in filenames:
                child = current_path / filename
                if child.is_symlink() or not child.is_file():
                    raise AgentBundleError("materialization_path_not_regular")
                observed_paths.add(child.relative_to(destination).as_posix())
        if observed_paths != expected_paths:
            raise AgentBundleError("materialization_closure_mismatch")
        expected_receipt = {
            "operation_id": operation_id,
            "component_digest": component.logical.fingerprint,
            "state_scope_id": state_scope_id,
        }
        receipt_path = destination / receipt_name
        try:
            durable = json.loads(receipt_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise AgentBundleError("materialization_receipt_invalid") from error
        if durable != expected_receipt:
            raise AgentBundleError("materialization_operation_conflict")
        for file in self.package.files:
            path = destination.joinpath(*PurePosixPath(file.path).parts)
            digest, byte_count = _file_digest_and_size(path, max_bytes=file.byte_count)
            if digest != file.digest or byte_count != file.byte_count:
                raise AgentBundleError("materialization_file_integrity_mismatch")
            is_executable = bool(path.stat().st_mode & stat.S_IXUSR)
            if is_executable != file.executable:
                raise AgentBundleError("materialization_file_mode_mismatch")

    def _materialized_component(
        self,
        component: AgentSnapshotComponentRef,
        request: AgentSnapshotMaterializationRequest,
        destination: Path,
    ) -> AgentSnapshotMaterializedComponent:
        overlay = None
        if self.kind in {
            AgentSnapshotComponentKind.MEMORY,
            AgentSnapshotComponentKind.WORKSPACE,
        }:
            overlay_kind = (
                AgentSnapshotOverlayKind.MEMORY
                if self.kind is AgentSnapshotComponentKind.MEMORY
                else AgentSnapshotOverlayKind.WORKSPACE
            )
            overlay = AgentSnapshotOverlayRef.create(
                kind=overlay_kind,
                overlay_id=f"{self.kind.value}-{request.state_scope_id[:16]}",
                baseline_fingerprint=component.logical.fingerprint,
                candidate_id=request.candidate_id,
                state_scope_id=request.state_scope_id,
                source_ref=f"cayu-ref:overlay:{request.state_scope_id}",
            )
        return AgentSnapshotMaterializedComponent(
            kind=self.kind,
            baseline_fingerprint=component.logical.fingerprint,
            capability=component.materialization,
            materialization_ref=(
                f"cayu-ref:materialized:{self.kind.value}:{request.state_scope_id}"
            ),
            overlay=overlay,
        )


async def load_portable_agent_snapshot_component_providers(
    snapshot: AgentSnapshot,
    *,
    object_store: AgentSnapshotObjectStore,
    materialization_root: str | Path,
    secret_redactor: SecretRedactor | None = None,
) -> tuple[PortableAgentSnapshotComponentProvider, ...]:
    """Reconstruct portable providers from an imported snapshot and object CAS."""

    if type(snapshot) is not AgentSnapshot:
        raise TypeError("snapshot must be an AgentSnapshot.")
    providers: list[PortableAgentSnapshotComponentProvider] = []
    for component in snapshot.components:
        package_bytes = await object_store.get_digest(
            component.logical.fingerprint,
            max_bytes=AGENT_BUNDLE_MAX_INDEX_BYTES,
        )
        if package_bytes is None:
            raise AgentBundleError("component_manifest_missing", component.kind.value)
        package = _component_package_from_bytes(package_bytes)
        if (
            package.kind is not component.kind
            or package.provider_id != component.provider_id
            or package.digest != component.logical.fingerprint
        ):
            raise AgentBundleError("component_manifest_identity_mismatch", component.kind.value)
        blobs: dict[str, bytes] = {}
        for file in package.files:
            reference = AgentBundleObjectRef(
                digest=file.digest,
                kind=AgentBundleObjectKind.COMPONENT_BLOB,
                schema_id="cayu.agent-snapshot.component-blob.v1",
                byte_count=file.byte_count,
            )
            if not await object_store.verify(reference):
                raise AgentBundleError("component_blob_missing", component.kind.value)
        providers.append(
            PortableAgentSnapshotComponentProvider(
                package,
                blobs,
                object_store=object_store,
                materialization_root=materialization_root,
                execution_profile=(
                    snapshot.execution_profile
                    if component.kind is AgentSnapshotComponentKind.EXECUTION_PROFILE
                    else None
                ),
                consistency=component.consistency,
                redaction=component.redaction,
                materialization=component.materialization,
                revision=component.logical.revision,
                frontier=component.logical.frontier,
                secret_redactor=secret_redactor,
            )
        )
    return tuple(providers)


class AgentBundleExportReceipt(_BundleModel):
    record_type: Literal["cayu.agent-bundle-export-receipt"] = "cayu.agent-bundle-export-receipt"
    schema_version: Literal[1] = 1
    receipt_id: StrictStr
    operation_id: StrictStr = Field(max_length=256)
    bundle: AgentBundle
    destination_fingerprint: StrictStr

    @field_validator("receipt_id", "destination_fingerprint")
    @classmethod
    def validate_digests(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("operation_id")
    @classmethod
    def validate_operation_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @model_validator(mode="after")
    def validate_receipt(self) -> AgentBundleExportReceipt:
        if self.receipt_id != _content_digest(self.identity_material(), "bundle_export_receipt"):
            raise ValueError("Bundle export receipt identity does not match its contents.")
        return self

    def identity_material(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"receipt_id"})

    @classmethod
    def create(
        cls,
        *,
        operation_id: str,
        bundle: AgentBundle,
        destination_fingerprint: str,
    ) -> AgentBundleExportReceipt:
        values: dict[str, Any] = {
            "operation_id": operation_id,
            "bundle": bundle,
            "destination_fingerprint": destination_fingerprint,
        }
        provisional = cls.model_construct(receipt_id="0" * 64, **values)
        return cls(
            receipt_id=_content_digest(provisional.identity_material(), "bundle_export_receipt"),
            **values,
        )


class AgentBundleImportReceipt(_BundleModel):
    record_type: Literal["cayu.agent-bundle-import-receipt"] = "cayu.agent-bundle-import-receipt"
    schema_version: Literal[1] = 1
    receipt_id: StrictStr
    operation_id: StrictStr = Field(max_length=256)
    bundle_id: StrictStr
    snapshot_ref: AgentSnapshotRef
    binding_id: StrictStr
    pin: AgentSnapshotPinReceipt
    imported_digests: tuple[StrictStr, ...]
    reused_digests: tuple[StrictStr, ...]

    @field_validator("receipt_id", "bundle_id", "binding_id")
    @classmethod
    def validate_digests(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("operation_id")
    @classmethod
    def validate_operation_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @field_validator("imported_digests", "reused_digests", mode="before")
    @classmethod
    def validate_digest_arrays(cls, value: object, info) -> tuple[str, ...]:
        if not isinstance(value, list | tuple):
            raise ValueError(f"{info.field_name} must be an ordered array.")
        copied = tuple(
            _sha256_hex(item, info.field_name)
            if type(item) is str
            else _raise_string(info.field_name, index)
            for index, item in enumerate(value)
        )
        if copied != tuple(sorted(set(copied))):
            raise ValueError(f"{info.field_name} must be unique and sorted.")
        return copied

    @model_validator(mode="after")
    def validate_receipt(self) -> AgentBundleImportReceipt:
        if set(self.imported_digests) & set(self.reused_digests):
            raise ValueError("Imported and reused object sets must be disjoint.")
        if self.pin.snapshot != self.snapshot_ref or self.pin.binding_id != self.binding_id:
            raise ValueError("Import pin must protect the imported binding and root.")
        if self.receipt_id != _content_digest(self.identity_material(), "bundle_import_receipt"):
            raise ValueError("Bundle import receipt identity does not match its contents.")
        return self

    def identity_material(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"receipt_id"})

    @classmethod
    def create(
        cls,
        *,
        operation_id: str,
        bundle: AgentBundle,
        binding_id: str,
        pin: AgentSnapshotPinReceipt,
        imported_digests: Iterable[str],
        reused_digests: Iterable[str],
    ) -> AgentBundleImportReceipt:
        values: dict[str, Any] = {
            "operation_id": operation_id,
            "bundle_id": bundle.bundle_id,
            "snapshot_ref": bundle.snapshot_ref,
            "binding_id": binding_id,
            "pin": pin,
            "imported_digests": tuple(sorted(set(imported_digests))),
            "reused_digests": tuple(sorted(set(reused_digests))),
        }
        provisional = cls.model_construct(receipt_id="0" * 64, **values)
        return cls(
            receipt_id=_content_digest(provisional.identity_material(), "bundle_import_receipt"),
            **values,
        )


class AgentExternalBindingResolution(_BundleModel):
    requirement: AgentExternalBindingRequirement
    resolution_fingerprint: StrictStr
    authority_fingerprint: StrictStr

    @field_validator("resolution_fingerprint", "authority_fingerprint")
    @classmethod
    def validate_digests(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    def identity_material(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class AgentMaterializationFreshIdentities(_BundleModel):
    runtime_identity: StrictStr
    session_identity: StrictStr
    operation_identity: StrictStr
    budget_identity: StrictStr
    lease_identity: StrictStr
    scratch_identity: StrictStr
    evaluator_identity: StrictStr | None = None
    discovery_grant_ids: tuple[StrictStr, ...] = ()

    @field_validator(
        "runtime_identity",
        "session_identity",
        "operation_identity",
        "budget_identity",
        "lease_identity",
        "scratch_identity",
        "evaluator_identity",
    )
    @classmethod
    def validate_digests(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_hex(value, info.field_name)

    @field_validator("discovery_grant_ids", mode="before")
    @classmethod
    def validate_discovery_grants(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, list | tuple):
            raise ValueError("discovery_grant_ids must be an ordered array.")
        copied = tuple(
            _sha256_hex(item, "discovery_grant_ids")
            if type(item) is str
            else _raise_string("discovery_grant_ids", index)
            for index, item in enumerate(value)
        )
        if copied != tuple(sorted(set(copied))):
            raise ValueError("discovery_grant_ids must be unique and sorted.")
        return copied

    @model_validator(mode="after")
    def require_empty_discovery_grants(self) -> AgentMaterializationFreshIdentities:
        if self.discovery_grant_ids:
            raise ValueError("Fresh materializations must begin with no discovery grants.")
        identities = tuple(
            value
            for value in (
                self.runtime_identity,
                self.session_identity,
                self.operation_identity,
                self.budget_identity,
                self.lease_identity,
                self.scratch_identity,
                self.evaluator_identity,
            )
            if value is not None
        )
        if len(identities) != len(set(identities)):
            raise ValueError("Fresh materialization identities must be distinct.")
        return self

    def identity_material(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class AgentBundleMaterializationRequest(_BundleModel):
    operation_id: StrictStr = Field(max_length=256)
    bundle: AgentBundle
    access: AgentSnapshotAccess
    mode: AgentSnapshotMaterializationMode
    candidate_id: StrictStr = Field(max_length=256)
    trial_id: StrictStr = Field(max_length=256)
    state_mode: AgentSnapshotTrialStateMode

    @field_validator("operation_id", "candidate_id", "trial_id")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @model_validator(mode="after")
    def validate_request(self) -> AgentBundleMaterializationRequest:
        if self.bundle.snapshot_ref != self.access.snapshot:
            raise ValueError("Materialization access must authorize the bundle root.")
        if (
            self.bundle.profile is AgentSnapshotProfile.REUSABLE_AGENT
            and self.mode is AgentSnapshotMaterializationMode.RESTORE
        ):
            raise ValueError("A fresh-session reusable agent cannot restore a prior session.")
        return self

    @property
    def request_fingerprint(self) -> str:
        return _content_digest(
            self.model_dump(mode="json"),
            "agent_bundle_materialization_request",
        )


class AgentBundleMaterializationAuthorization(_BundleModel):
    """Application-owned allocation and external-binding authorization receipt."""

    record_type: Literal["cayu.agent-bundle-materialization-authorization"] = (
        "cayu.agent-bundle-materialization-authorization"
    )
    schema_version: Literal[1] = 1
    authorization_id: StrictStr
    request_fingerprint: StrictStr
    authority_fingerprint: StrictStr
    fresh_identities: AgentMaterializationFreshIdentities
    external_bindings: tuple[AgentExternalBindingResolution, ...] = ()

    @field_validator("authorization_id", "request_fingerprint", "authority_fingerprint")
    @classmethod
    def validate_digests(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("external_bindings", mode="before")
    @classmethod
    def validate_bindings_array(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError("external_bindings must be an ordered array.")
        return value

    @model_validator(mode="after")
    def validate_authorization(self) -> AgentBundleMaterializationAuthorization:
        resolution_keys = tuple(
            (
                item.requirement.kind.value,
                item.requirement.name,
                item.requirement.requirement_fingerprint,
            )
            for item in self.external_bindings
        )
        if resolution_keys != tuple(sorted(set(resolution_keys))):
            raise ValueError("External binding resolutions must be unique and canonical.")
        if self.authorization_id != _content_digest(
            self.identity_material(), "agent_bundle_materialization_authorization"
        ):
            raise ValueError("Bundle materialization authorization identity is invalid.")
        return self

    def identity_material(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"authorization_id"})

    @property
    def state_partition_fingerprint(self) -> str:
        return _content_digest(
            {
                "authorization_id": self.authorization_id,
                "fresh_identities": self.fresh_identities.identity_material(),
                "external_bindings": [item.identity_material() for item in self.external_bindings],
            },
            "agent_bundle_materialization_partition",
        )

    @classmethod
    def create(
        cls,
        *,
        request: AgentBundleMaterializationRequest,
        authority_fingerprint: str,
        fresh_identities: AgentMaterializationFreshIdentities,
        external_bindings: Iterable[AgentExternalBindingResolution] = (),
    ) -> AgentBundleMaterializationAuthorization:
        values: dict[str, Any] = {
            "request_fingerprint": request.request_fingerprint,
            "authority_fingerprint": authority_fingerprint,
            "fresh_identities": fresh_identities,
            "external_bindings": tuple(external_bindings),
        }
        provisional = cls.model_construct(authorization_id="0" * 64, **values)
        return cls(
            authorization_id=_content_digest(
                provisional.identity_material(),
                "agent_bundle_materialization_authorization",
            ),
            **values,
        )


class AgentBundleMaterializationAuthority(ABC):
    """Application authority that allocates and activates fresh runtime bindings."""

    @abstractmethod
    async def authorize_materialization(
        self,
        request: AgentBundleMaterializationRequest,
    ) -> AgentBundleMaterializationAuthorization:
        """Return a source-owned receipt for this exact request."""


def _validate_materialization_authorization(
    request: AgentBundleMaterializationRequest,
    authorization: AgentBundleMaterializationAuthorization,
) -> None:
    if authorization.request_fingerprint != request.request_fingerprint:
        raise AgentBundleError("materialization_authorization_request_mismatch")
    resolution_keys = {
        (
            item.requirement.kind.value,
            item.requirement.name,
            item.requirement.requirement_fingerprint,
        )
        for item in authorization.external_bindings
    }
    required = {
        (item.kind.value, item.name, item.requirement_fingerprint)
        for item in request.bundle.external_bindings
        if item.required
    }
    declared = {
        (item.kind.value, item.name, item.requirement_fingerprint)
        for item in request.bundle.external_bindings
    }
    if not required.issubset(resolution_keys) or not resolution_keys.issubset(declared):
        raise AgentBundleError("materialization_external_bindings_unauthorized")
    if (
        request.bundle.profile is AgentSnapshotProfile.EVALUATION_CANDIDATE
        and authorization.fresh_identities.evaluator_identity is None
    ):
        raise AgentBundleError("materialization_evaluator_unauthorized")


class AgentBundleMaterializationReceipt(_BundleModel):
    record_type: Literal["cayu.agent-bundle-materialization-receipt"] = (
        "cayu.agent-bundle-materialization-receipt"
    )
    schema_version: Literal[1] = 1
    receipt_id: StrictStr
    operation_id: StrictStr = Field(max_length=256)
    bundle_id: StrictStr
    materialization_request_fingerprint: StrictStr
    snapshot_ref: AgentSnapshotRef
    profile: AgentSnapshotProfile
    mode: AgentSnapshotMaterializationMode
    materialization: AgentSnapshotMaterialization
    materialization_fingerprint: StrictStr
    state_scope_id: StrictStr
    authorization: AgentBundleMaterializationAuthorization
    fresh_identities: AgentMaterializationFreshIdentities
    external_bindings: tuple[AgentExternalBindingResolution, ...]

    @field_validator(
        "receipt_id",
        "bundle_id",
        "materialization_request_fingerprint",
        "materialization_fingerprint",
        "state_scope_id",
    )
    @classmethod
    def validate_digests(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("operation_id")
    @classmethod
    def validate_operation_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @field_validator("external_bindings", mode="before")
    @classmethod
    def validate_bindings_array(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError("external_bindings must be an ordered array.")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> AgentBundleMaterializationReceipt:
        if (
            self.materialization.fingerprint != self.materialization_fingerprint
            or self.materialization.snapshot_fingerprint != self.snapshot_ref.snapshot_root
            or self.materialization.state_scope_id != self.state_scope_id
            or self.authorization.fresh_identities != self.fresh_identities
            or self.authorization.external_bindings != self.external_bindings
            or self.authorization.request_fingerprint != self.materialization_request_fingerprint
        ):
            raise ValueError("Bundle receipt contradicts its component materialization.")
        if self.receipt_id != _content_digest(
            self.identity_material(), "bundle_materialization_receipt"
        ):
            raise ValueError("Bundle materialization receipt identity is invalid.")
        return self

    def identity_material(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"receipt_id"})

    @classmethod
    def create(
        cls,
        *,
        request: AgentBundleMaterializationRequest,
        authorization: AgentBundleMaterializationAuthorization,
        materialization: AgentSnapshotMaterialization,
    ) -> AgentBundleMaterializationReceipt:
        values: dict[str, Any] = {
            "operation_id": request.operation_id,
            "bundle_id": request.bundle.bundle_id,
            "materialization_request_fingerprint": request.request_fingerprint,
            "snapshot_ref": request.bundle.snapshot_ref,
            "profile": request.bundle.profile,
            "mode": request.mode,
            "materialization": materialization,
            "materialization_fingerprint": materialization.fingerprint,
            "state_scope_id": materialization.state_scope_id,
            "authorization": authorization,
            "fresh_identities": authorization.fresh_identities,
            "external_bindings": authorization.external_bindings,
        }
        provisional = cls.model_construct(receipt_id="0" * 64, **values)
        return cls(
            receipt_id=_content_digest(
                provisional.identity_material(), "bundle_materialization_receipt"
            ),
            **values,
        )


class AgentSnapshotTerminalCaptureRequest(_BundleModel):
    operation_id: StrictStr = Field(max_length=256)
    capture: AgentSnapshotCaptureRequest
    parent_materialization_fingerprint: StrictStr
    parent_result_fingerprint: StrictStr
    profile: AgentSnapshotProfile
    finalization_policy: StrictStr = Field(max_length=256)

    @field_validator("operation_id", "finalization_policy")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @field_validator("parent_materialization_fingerprint", "parent_result_fingerprint")
    @classmethod
    def validate_digests(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @model_validator(mode="after")
    def validate_safe_terminal(self) -> AgentSnapshotTerminalCaptureRequest:
        if self.capture.parent_snapshot_fingerprint is None:
            raise ValueError("Terminal capture must name its parent snapshot root.")
        if self.capture.evaluator is not None or self.capture.promotion_authority is not None:
            raise ValueError("Terminal capture cannot inherit evaluator or promotion authority.")
        return self

    @property
    def request_fingerprint(self) -> str:
        return _content_digest(
            self.model_dump(mode="json"),
            "agent_snapshot_terminal_capture_request",
        )


class AgentSnapshotTerminalAuthorization(_BundleModel):
    """Application-owned verification of one durable runtime result and frontier."""

    record_type: Literal["cayu.agent-snapshot-terminal-authorization"] = (
        "cayu.agent-snapshot-terminal-authorization"
    )
    schema_version: Literal[1] = 1
    authorization_id: StrictStr
    request_fingerprint: StrictStr
    result_fingerprint: StrictStr
    runtime_evidence_fingerprint: StrictStr
    authority_fingerprint: StrictStr

    @field_validator(
        "authorization_id",
        "request_fingerprint",
        "result_fingerprint",
        "runtime_evidence_fingerprint",
        "authority_fingerprint",
    )
    @classmethod
    def validate_digests(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @model_validator(mode="after")
    def validate_authorization(self) -> AgentSnapshotTerminalAuthorization:
        if self.authorization_id != _content_digest(
            self.identity_material(), "agent_snapshot_terminal_authorization"
        ):
            raise ValueError("Terminal authorization identity is invalid.")
        return self

    def identity_material(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"authorization_id"})

    @classmethod
    def create(
        cls,
        *,
        request: AgentSnapshotTerminalCaptureRequest,
        result: AgentSnapshotResultBinding,
        authority_fingerprint: str,
    ) -> AgentSnapshotTerminalAuthorization:
        values: dict[str, Any] = {
            "request_fingerprint": request.request_fingerprint,
            "result_fingerprint": result.fingerprint,
            "runtime_evidence_fingerprint": result.runtime_evidence_fingerprint,
            "authority_fingerprint": authority_fingerprint,
        }
        provisional = cls.model_construct(authorization_id="0" * 64, **values)
        return cls(
            authorization_id=_content_digest(
                provisional.identity_material(),
                "agent_snapshot_terminal_authorization",
            ),
            **values,
        )


class AgentSnapshotTerminalAuthority(ABC):
    """Application authority that verifies durable runtime/session terminal evidence."""

    @abstractmethod
    async def authorize_terminal_capture(
        self,
        *,
        request: AgentSnapshotTerminalCaptureRequest,
        materialization: AgentSnapshotMaterialization,
        trial: AgentSnapshotTrialBinding,
        result: AgentSnapshotResultBinding,
    ) -> AgentSnapshotTerminalAuthorization:
        """Verify source-owned runtime evidence and authorize descendant capture."""


class AgentSnapshotTerminalCaptureReceipt(_BundleModel):
    record_type: Literal["cayu.agent-snapshot-terminal-capture-receipt"] = (
        "cayu.agent-snapshot-terminal-capture-receipt"
    )
    schema_version: Literal[1] = 1
    receipt_id: StrictStr
    operation_id: StrictStr = Field(max_length=256)
    terminal_request_fingerprint: StrictStr
    parent_materialization_fingerprint: StrictStr
    parent_result_fingerprint: StrictStr
    parent_snapshot_ref: AgentSnapshotRef
    descendant_snapshot_ref: AgentSnapshotRef
    profile: AgentSnapshotProfile
    terminal_disposition: AgentSnapshotTerminalDisposition
    finalization_policy: StrictStr = Field(max_length=256)
    safe_frontier_fingerprint: StrictStr | None = None
    terminal_authorization: AgentSnapshotTerminalAuthorization

    @field_validator(
        "receipt_id",
        "terminal_request_fingerprint",
        "parent_materialization_fingerprint",
        "parent_result_fingerprint",
        "safe_frontier_fingerprint",
    )
    @classmethod
    def validate_digests(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_hex(value, info.field_name)

    @field_validator("operation_id", "finalization_policy")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @model_validator(mode="after")
    def validate_receipt(self) -> AgentSnapshotTerminalCaptureReceipt:
        if (
            self.terminal_authorization.result_fingerprint != self.parent_result_fingerprint
            or self.terminal_authorization.request_fingerprint != self.terminal_request_fingerprint
        ):
            raise ValueError("Terminal capture receipt contradicts its authorization.")
        if self.receipt_id != _content_digest(self.identity_material(), "terminal_capture_receipt"):
            raise ValueError("Terminal capture receipt identity is invalid.")
        return self

    def identity_material(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"receipt_id"})

    @classmethod
    def create(
        cls,
        *,
        request: AgentSnapshotTerminalCaptureRequest,
        result: AgentSnapshotResultBinding,
        authorization: AgentSnapshotTerminalAuthorization,
        descendant_snapshot_ref: AgentSnapshotRef,
    ) -> AgentSnapshotTerminalCaptureReceipt:
        parent_root = request.capture.parent_snapshot_fingerprint
        assert parent_root is not None
        values: dict[str, Any] = {
            "operation_id": request.operation_id,
            "terminal_request_fingerprint": request.request_fingerprint,
            "parent_materialization_fingerprint": request.parent_materialization_fingerprint,
            "parent_result_fingerprint": result.fingerprint,
            "parent_snapshot_ref": AgentSnapshotRef(snapshot_root=parent_root),
            "descendant_snapshot_ref": descendant_snapshot_ref,
            "profile": request.profile,
            "terminal_disposition": result.terminal_disposition,
            "finalization_policy": request.finalization_policy,
            "safe_frontier_fingerprint": result.safe_frontier_fingerprint,
            "terminal_authorization": authorization,
        }
        provisional = cls.model_construct(receipt_id="0" * 64, **values)
        return cls(
            receipt_id=_content_digest(provisional.identity_material(), "terminal_capture_receipt"),
            **values,
        )


def _bundle_path_fingerprint(path: Path) -> str:
    return _content_digest(
        {"absolute_path": str(path.resolve(strict=False))},
        "agent_bundle_destination",
    )


def _component_package_from_bytes(content: bytes) -> AgentSnapshotComponentPackage:
    if len(content) > AGENT_BUNDLE_MAX_INDEX_BYTES:
        raise AgentBundleError("component_manifest_size_limit_exceeded")
    try:
        package = AgentSnapshotComponentPackage.model_validate_json(content)
    except Exception as error:
        raise AgentBundleError("component_manifest_invalid") from error
    if package.document != content or package.digest != _digest(content):
        raise AgentBundleError("component_manifest_not_canonical")
    return package


def _rebind_logical_ref(
    reference: AgentSnapshotLogicalRef,
    scope_fingerprint: str,
) -> AgentSnapshotLogicalRef:
    if reference.scope_fingerprint is None:
        return reference
    return reference.model_copy(update={"scope_fingerprint": scope_fingerprint})


def _rebind_snapshot(
    snapshot: AgentSnapshot,
    *,
    subject: AgentSnapshotSubject,
    authority_scope_fingerprint: str,
) -> AgentSnapshot:
    _sha256_hex(authority_scope_fingerprint, "authority_scope_fingerprint")
    components = tuple(
        component.model_copy(
            update={
                "logical": _rebind_logical_ref(
                    component.logical,
                    authority_scope_fingerprint,
                )
            }
        )
        for component in snapshot.components
    )
    body = next(
        component for component in components if component.kind is AgentSnapshotComponentKind.BODY
    )
    rebound_subject = subject.model_copy(update={"body_release": body.logical})
    memory_state = snapshot.memory_state
    if memory_state is not None:
        updates = {
            field_name: (
                None
                if reference is None
                else _rebind_logical_ref(reference, authority_scope_fingerprint)
            )
            for field_name, reference in (
                ("knowledge", memory_state.knowledge),
                ("transcript_evidence", memory_state.transcript_evidence),
                ("artifact_evidence", memory_state.artifact_evidence),
                ("work_context", memory_state.work_context),
                ("recall_policy", memory_state.recall_policy),
                ("admission_policy", memory_state.admission_policy),
                ("context_projection_policy", memory_state.context_projection_policy),
                ("interaction_focus", memory_state.interaction_focus),
                ("recall_receipts", memory_state.recall_receipts),
                ("context_exposures", memory_state.context_exposures),
                ("index_readiness", memory_state.index_readiness),
            )
        }
        rebound_memory = type(memory_state).model_validate(
            memory_state.model_copy(update=updates).model_dump(mode="json")
        )
    else:
        rebound_memory = None
    rebound = AgentSnapshot.model_validate(
        snapshot.model_copy(
            update={
                "subject": rebound_subject,
                "authority_scope_fingerprint": authority_scope_fingerprint,
                "components": components,
                "memory_state": rebound_memory,
                "evaluator": None,
                "promotion_authority": None,
            }
        ).model_dump(mode="json")
    )
    if rebound.snapshot_root != snapshot.snapshot_root:
        raise AgentBundleError("scope_rebinding_changed_snapshot_root")
    return rebound


class AgentBundleCoordinator:
    """Export and import complete authorized AgentSnapshot closures."""

    def __init__(
        self,
        *,
        snapshot_store: AgentSnapshotStore,
        object_store: AgentSnapshotObjectStore,
        secret_redactor: SecretRedactor | None = None,
        materialization_authority: AgentBundleMaterializationAuthority | None = None,
        terminal_authority: AgentSnapshotTerminalAuthority | None = None,
    ) -> None:
        if not isinstance(snapshot_store, AgentSnapshotStore):
            raise TypeError("snapshot_store must be an AgentSnapshotStore.")
        if not isinstance(object_store, AgentSnapshotObjectStore):
            raise TypeError("object_store must be an AgentSnapshotObjectStore.")
        self.snapshot_store = snapshot_store
        self.object_store = object_store
        if secret_redactor is not None and not isinstance(secret_redactor, SecretRedactor):
            raise TypeError("secret_redactor must be a SecretRedactor.")
        self.secret_redactor = secret_redactor
        if materialization_authority is not None and not isinstance(
            materialization_authority,
            AgentBundleMaterializationAuthority,
        ):
            raise TypeError(
                "materialization_authority must be an AgentBundleMaterializationAuthority."
            )
        self.materialization_authority = materialization_authority
        if terminal_authority is not None and not isinstance(
            terminal_authority,
            AgentSnapshotTerminalAuthority,
        ):
            raise TypeError("terminal_authority must be an AgentSnapshotTerminalAuthority.")
        self.terminal_authority = terminal_authority

    async def export(
        self,
        *,
        operation_id: str,
        access: AgentSnapshotAccess,
        profile: AgentSnapshotProfile,
        destination: str | Path,
        mode: AgentBundleMode = AgentBundleMode.FULL,
        destination_inventory: AgentBundleInventory | None = None,
    ) -> AgentBundleExportReceipt:
        operation_id = _clean(operation_id, "operation_id", max_chars=256)
        destination_path = Path(destination)
        if not destination_path.is_absolute():
            raise ValueError("Bundle destination must be absolute.")
        inventory = destination_inventory or AgentBundleInventory()
        if mode is AgentBundleMode.FULL and inventory.object_digests:
            raise ValueError("A full bundle does not consume a destination inventory.")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        async with _async_path_lock(
            destination_path.parent,
            destination_path.name,
            lock_directory_name="cayu-agent-bundle-export-locks",
        ):
            existing_receipt: AgentBundleExportReceipt | None = None
            if destination_path.exists():
                bundle, _ = await asyncio.to_thread(
                    self._read_and_verify_bundle_directory,
                    destination_path,
                )
                snapshot = await self.snapshot_store.get_snapshot(access)
                if (
                    bundle.snapshot_ref != snapshot.ref
                    or bundle.export_binding_id != access.binding_id
                    or bundle.export_authority_scope_fingerprint
                    != access.authority_scope_fingerprint
                    or bundle.destination_inventory_fingerprint != inventory.fingerprint
                    or bundle.profile is not profile
                    or bundle.mode is not mode
                ):
                    raise AgentBundleError("bundle_destination_conflict")
                existing_receipt = AgentBundleExportReceipt.create(
                    operation_id=operation_id,
                    bundle=bundle,
                    destination_fingerprint=_bundle_path_fingerprint(destination_path),
                )
            protection = AgentSnapshotProtection.create(
                operation_id=operation_id,
                access=access,
                kind=AgentSnapshotProtectionKind.EXPORTING,
                owner="agent-bundle-coordinator",
                reason="bundle-export-in-progress",
            )
            await self.snapshot_store.protect_snapshot(protection)
            completed = False
            try:
                if existing_receipt is not None:
                    receipt = existing_receipt
                else:
                    receipt = await self._export_protected(
                        operation_id=operation_id,
                        access=access,
                        profile=profile,
                        destination=destination_path,
                        mode=mode,
                        destination_inventory=inventory,
                    )
                completed = True
                return receipt
            finally:
                if completed:
                    await self.snapshot_store.release_snapshot_protection(
                        operation_id=operation_id,
                        access=access,
                        protection_id=protection.protection_id,
                    )

    async def _export_protected(
        self,
        *,
        operation_id: str,
        access: AgentSnapshotAccess,
        profile: AgentSnapshotProfile,
        destination: Path,
        mode: AgentBundleMode,
        destination_inventory: AgentBundleInventory,
    ) -> AgentBundleExportReceipt:
        snapshot = await self.snapshot_store.get_snapshot(access)
        nodes = await self.snapshot_store.enumerate_snapshot_closure(access)
        objects: dict[str, tuple[AgentBundleObjectRef, bytes | None]] = {}

        portable_snapshot = AgentSnapshot.model_validate(
            snapshot.model_copy(update={"evaluator": None, "promotion_authority": None}).model_dump(
                mode="json"
            )
        )
        snapshot_bytes = _canonical_json(portable_snapshot, "agent_snapshot")
        snapshot_ref = _object_ref(
            snapshot_bytes,
            kind=AgentBundleObjectKind.SNAPSHOT_DOCUMENT,
            schema_id="cayu.agent-snapshot.v3",
        )
        objects[snapshot_ref.digest] = (snapshot_ref, snapshot_bytes)
        for node in nodes:
            content = _canonical_json(node, "agent_snapshot_node")
            reference = AgentBundleObjectRef(
                digest=node.digest,
                kind=AgentBundleObjectKind.SNAPSHOT_NODE,
                schema_id=node.schema_id,
                byte_count=len(content),
            )
            objects[reference.digest] = (reference, content)

        external_bindings: dict[tuple[str, str, str], AgentExternalBindingRequirement] = {}
        materialized_disk_bytes = 0
        session_package: AgentSnapshotComponentPackage | None = None
        for component in snapshot.components:
            if component.completeness is not AgentSnapshotCompleteness.COMPLETE or (
                component.materialization
                in {
                    AgentSnapshotMaterializationCapability.REFERENCE_ONLY,
                    AgentSnapshotMaterializationCapability.UNAVAILABLE,
                }
            ):
                raise AgentBundleError(
                    "component_not_portable",
                    component.kind.value,
                )
            package_bytes = await self.object_store.get_digest(
                component.logical.fingerprint,
                max_bytes=AGENT_BUNDLE_MAX_INDEX_BYTES,
            )
            if package_bytes is None:
                raise AgentBundleError("component_manifest_missing", component.kind.value)
            package = _component_package_from_bytes(package_bytes)
            if (
                package.digest != component.logical.fingerprint
                or package.kind is not component.kind
                or package.provider_id != component.provider_id
                or package.profile is not profile
            ):
                raise AgentBundleError("component_manifest_identity_mismatch", component.kind.value)
            if package.kind is AgentSnapshotComponentKind.SESSION:
                session_package = package
            package_ref = _object_ref(
                package_bytes,
                kind=AgentBundleObjectKind.COMPONENT_MANIFEST,
                schema_id=package.component_schema,
            )
            objects[package_ref.digest] = (package_ref, package_bytes)
            for file in package.files:
                reference = AgentBundleObjectRef(
                    digest=file.digest,
                    kind=AgentBundleObjectKind.COMPONENT_BLOB,
                    schema_id="cayu.agent-snapshot.component-blob.v1",
                    byte_count=file.byte_count,
                )
                if not await self.object_store.verify(reference):
                    raise AgentBundleError("component_blob_missing", component.kind.value)
                existing = objects.get(reference.digest)
                if existing is not None and existing[0] != reference:
                    raise AgentBundleError("bundle_object_metadata_conflict")
                objects[reference.digest] = (reference, None)
                materialized_disk_bytes += file.byte_count
            for binding in package.external_bindings:
                key = (binding.kind.value, binding.name, binding.requirement_fingerprint)
                external_bindings[key] = binding

        if session_package is None:
            raise AgentBundleError("session_disposition_missing")
        if (
            profile is AgentSnapshotProfile.REUSABLE_AGENT
            and session_package.session_disposition
            is not AgentSnapshotSessionDisposition.FRESH_ON_MATERIALIZE
        ):
            raise AgentBundleError("reusable_agent_session_not_fresh")
        if (
            profile is AgentSnapshotProfile.CONTINUING_AGENT
            and session_package.session_disposition
            is not AgentSnapshotSessionDisposition.SAFE_FRONTIER
        ):
            raise AgentBundleError("continuing_agent_frontier_not_restorable")

        closure = tuple(reference for reference, _ in objects.values())
        if self.secret_redactor is not None:
            for reference, content in objects.values():
                if content is not None and self.secret_redactor.contains_secret_bytes(content):
                    raise AgentBundleError("bundle_contains_secret")
                if (
                    content is None
                    and reference.kind is AgentBundleObjectKind.COMPONENT_BLOB
                    and await self.object_store.contains_secret(
                        reference,
                        self.secret_redactor,
                    )
                ):
                    raise AgentBundleError("bundle_contains_secret")
        if len(closure) > AGENT_BUNDLE_MAX_OBJECTS:
            raise AgentBundleError("bundle_object_count_limit_exceeded")
        logical_bytes = sum(reference.byte_count for reference in closure)
        if logical_bytes > AGENT_BUNDLE_MAX_TOTAL_BYTES:
            raise AgentBundleError("bundle_total_size_limit_exceeded")
        inventory = set(destination_inventory.object_digests)
        transferred = tuple(
            sorted(
                digest
                for digest, (reference, _) in objects.items()
                if reference.kind is AgentBundleObjectKind.SNAPSHOT_DOCUMENT
                or digest not in inventory
            )
        )
        incremental_bytes = sum(objects[digest][0].byte_count for digest in transferred)
        shared_bytes = logical_bytes - incremental_bytes
        root_node = next(node for node in nodes if node.digest == snapshot.snapshot_root)
        bundle = AgentBundle.create(
            snapshot_ref=snapshot.ref,
            export_binding_id=access.binding_id,
            export_authority_scope_fingerprint=access.authority_scope_fingerprint,
            destination_inventory_fingerprint=destination_inventory.fingerprint,
            profile=profile,
            mode=mode,
            snapshot_document=snapshot_ref,
            closure=closure,
            transferred_digests=transferred,
            external_bindings=external_bindings.values(),
            size_report=AgentBundleSizeReport(
                root_manifest_bytes=len(_canonical_json(root_node, "agent_snapshot_root")),
                logical_closure_bytes=logical_bytes,
                unique_stored_bytes=incremental_bytes,
                shared_stored_bytes=shared_bytes,
                incremental_transfer_bytes=incremental_bytes,
                materialized_disk_bytes=materialized_disk_bytes,
                unresolved_external_bindings=tuple(
                    sorted(
                        f"{binding.kind.value}:{binding.name}:{binding.requirement_fingerprint}"
                        for binding in external_bindings.values()
                        if binding.required
                    )
                ),
            ),
        )
        await self._write_bundle(
            destination,
            bundle,
            objects,
            operation_id,
        )
        return AgentBundleExportReceipt.create(
            operation_id=operation_id,
            bundle=bundle,
            destination_fingerprint=_bundle_path_fingerprint(destination),
        )

    async def _write_bundle(
        self,
        destination: Path,
        bundle: AgentBundle,
        objects: Mapping[str, tuple[AgentBundleObjectRef, bytes | None]],
        operation_id: str,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        index_bytes = _canonical_json(bundle, "agent_bundle")
        if len(index_bytes) > AGENT_BUNDLE_MAX_INDEX_BYTES:
            raise AgentBundleError("bundle_index_size_limit_exceeded")
        if destination.exists():
            if destination.is_symlink() or not destination.is_dir():
                raise AgentBundleError("bundle_destination_conflict")
            existing_index = destination / AGENT_BUNDLE_INDEX_FILENAME
            if (
                existing_index.is_file()
                and not existing_index.is_symlink()
                and existing_index.read_bytes() == index_bytes
            ):
                return
            raise AgentBundleError("bundle_destination_conflict")
        operation_digest = _content_digest(
            {"operation_id": operation_id, "bundle_id": bundle.bundle_id},
            "bundle_export_operation",
        )
        temporary = destination.parent / f".{destination.name}.{operation_digest}.tmp"
        if temporary.exists():
            if temporary.is_symlink() or not temporary.is_dir():
                raise AgentBundleError("bundle_export_staging_conflict")
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        try:
            (temporary / AGENT_BUNDLE_INDEX_FILENAME).write_bytes(index_bytes)
            for digest in bundle.transferred_digests:
                reference, content = objects[digest]
                target = _safe_object_path(temporary, digest)
                target.parent.mkdir(parents=True, exist_ok=True)
                if content is None:
                    await self.object_store.copy_to(reference, target)
                else:
                    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(content)
                        stream.flush()
                        os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    async def import_bundle(
        self,
        *,
        operation_id: str,
        source: str | Path,
        subject: AgentSnapshotSubject,
        authority_scope_fingerprint: str,
        owner: str,
        retention_class: AgentSnapshotRetentionClass = AgentSnapshotRetentionClass.RELEASE,
    ) -> AgentBundleImportReceipt:
        operation_id = _clean(operation_id, "operation_id", max_chars=256)
        owner = _clean(owner, "owner", max_chars=256)
        source_path = Path(source)
        if not source_path.is_absolute():
            raise ValueError("Bundle source must be absolute.")
        bundle, content_by_digest = await asyncio.to_thread(
            self._read_and_verify_bundle_directory,
            source_path,
        )
        imported: list[str] = []
        reused: list[str] = []
        resolved: dict[str, bytes | Path | None] = {}
        by_digest = {reference.digest: reference for reference in bundle.closure}
        for digest, reference in sorted(by_digest.items()):
            content = content_by_digest.get(digest)
            if content is not None:
                if reference.kind in {
                    AgentBundleObjectKind.COMPONENT_MANIFEST,
                    AgentBundleObjectKind.COMPONENT_BLOB,
                }:
                    if isinstance(content, Path):
                        await self.object_store.put_file(reference, content)
                    else:
                        await self.object_store.put(reference, content)
                imported.append(digest)
                resolved[digest] = content
                continue
            if reference.kind not in {
                AgentBundleObjectKind.COMPONENT_MANIFEST,
                AgentBundleObjectKind.COMPONENT_BLOB,
            }:
                if reference.kind is AgentBundleObjectKind.SNAPSHOT_NODE:
                    reused.append(digest)
                    resolved[digest] = None
                    continue
                raise AgentBundleError("thin_bundle_nonportable_object_omitted", digest)
            if reference.kind is AgentBundleObjectKind.COMPONENT_BLOB:
                if not await self.object_store.verify(reference):
                    raise AgentBundleError("thin_bundle_object_unavailable", digest)
                content = None
            else:
                content = await self.object_store.get(reference)
                if content is None:
                    raise AgentBundleError("thin_bundle_object_unavailable", digest)
            reused.append(digest)
            resolved[digest] = content

        snapshot_content = resolved[bundle.snapshot_document.digest]
        if not isinstance(snapshot_content, bytes):
            raise AgentBundleError("snapshot_document_invalid")
        try:
            source_snapshot = agent_snapshot_from_json(snapshot_content)
        except Exception as error:
            raise AgentBundleError("snapshot_document_invalid") from error
        if source_snapshot.snapshot_root != bundle.snapshot_ref.snapshot_root:
            raise AgentBundleError("bundle_snapshot_root_mismatch")
        if (
            source_snapshot.authority_scope_fingerprint != bundle.export_authority_scope_fingerprint
            or source_snapshot.identity_binding.binding_id != bundle.export_binding_id
        ):
            raise AgentBundleError("bundle_export_scope_mismatch")
        self._verify_snapshot_and_component_closure(bundle, source_snapshot, resolved)
        if self.secret_redactor is not None:
            for digest, reference in sorted(by_digest.items()):
                content = resolved[digest]
                contains_secret = False
                if isinstance(content, Path):
                    if reference.kind in {
                        AgentBundleObjectKind.COMPONENT_MANIFEST,
                        AgentBundleObjectKind.COMPONENT_BLOB,
                    }:
                        contains_secret = await self.object_store.contains_secret(
                            reference,
                            self.secret_redactor,
                        )
                    else:
                        contains_secret = await asyncio.to_thread(
                            _file_contains_secret,
                            content,
                            self.secret_redactor,
                        )
                elif isinstance(content, bytes):
                    contains_secret = self.secret_redactor.contains_secret_bytes(content)
                elif reference.kind is AgentBundleObjectKind.COMPONENT_BLOB:
                    contains_secret = await self.object_store.contains_secret(
                        reference,
                        self.secret_redactor,
                    )
                if contains_secret:
                    raise AgentBundleError("bundle_contains_secret")
        rebound = _rebind_snapshot(
            source_snapshot,
            subject=subject,
            authority_scope_fingerprint=authority_scope_fingerprint,
        )
        binding = AgentSnapshotIdentityBinding.create(
            subject=rebound.subject,
            snapshot=rebound.ref,
            authority_scope_fingerprint=authority_scope_fingerprint,
        )
        access = AgentSnapshotAccess(
            snapshot=rebound.ref,
            binding_id=binding.binding_id,
            authority_scope_fingerprint=authority_scope_fingerprint,
        )
        pin_request = AgentSnapshotPinRequest(
            operation_id=f"{operation_id}:pin",
            access=access,
            owner=owner,
            reason="imported-agent-bundle",
            retention_class=retention_class,
        )
        try:
            _, pin = await self.snapshot_store.put_snapshot_and_pin(
                rebound,
                binding,
                pin_request,
            )
        except NotImplementedError as error:
            raise AgentBundleError("atomic_import_publication_unsupported") from error
        return AgentBundleImportReceipt.create(
            operation_id=operation_id,
            bundle=bundle,
            binding_id=binding.binding_id,
            pin=pin,
            imported_digests=imported,
            reused_digests=reused,
        )

    async def materialize(
        self,
        request: AgentBundleMaterializationRequest,
        *,
        materialization_root: str | Path,
    ) -> AgentBundleMaterializationReceipt:
        if type(request) is not AgentBundleMaterializationRequest:
            raise TypeError("request must be an AgentBundleMaterializationRequest.")
        validated = AgentBundleMaterializationRequest.model_validate(
            request.model_dump(mode="json")
        )
        authority = self.materialization_authority
        if authority is None:
            raise AgentBundleError("materialization_authority_unavailable")
        snapshot = await self.snapshot_store.get_snapshot(validated.access)
        if snapshot.ref != validated.bundle.snapshot_ref:
            raise AgentBundleError("materialization_snapshot_mismatch")
        packages: list[AgentSnapshotComponentPackage] = []
        for component in snapshot.components:
            content = await self.object_store.get_digest(
                component.logical.fingerprint,
                max_bytes=AGENT_BUNDLE_MAX_INDEX_BYTES,
            )
            if content is None:
                raise AgentBundleError("component_manifest_missing", component.kind.value)
            package = _component_package_from_bytes(content)
            if package.profile is not validated.bundle.profile:
                raise AgentBundleError("component_profile_mismatch", component.kind.value)
            packages.append(package)
        session = next(
            (package for package in packages if package.kind is AgentSnapshotComponentKind.SESSION),
            None,
        )
        if session is None:
            raise AgentBundleError("session_disposition_missing")
        if (
            validated.bundle.profile is AgentSnapshotProfile.REUSABLE_AGENT
            and session.session_disposition
            is not AgentSnapshotSessionDisposition.FRESH_ON_MATERIALIZE
        ):
            raise AgentBundleError("reusable_agent_session_not_fresh")
        if (
            validated.bundle.profile is AgentSnapshotProfile.CONTINUING_AGENT
            and session.session_disposition is not AgentSnapshotSessionDisposition.SAFE_FRONTIER
        ):
            raise AgentBundleError("continuing_agent_frontier_not_restorable")
        authorization = await authority.authorize_materialization(validated)
        if type(authorization) is not AgentBundleMaterializationAuthorization:
            raise AgentBundleError("materialization_authorization_invalid")
        authorization = AgentBundleMaterializationAuthorization.model_validate(
            authorization.model_dump(mode="json")
        )
        _validate_materialization_authorization(validated, authorization)
        authorization_content = _canonical_json(
            authorization,
            "bundle_materialization_authorization",
        )
        await self.object_store.put(
            _object_ref(
                authorization_content,
                kind=AgentBundleObjectKind.OPERATION_RECEIPT,
                schema_id="cayu.agent-bundle-materialization-authorization.v1",
            ),
            authorization_content,
        )
        providers = await load_portable_agent_snapshot_component_providers(
            snapshot,
            object_store=self.object_store,
            materialization_root=materialization_root,
            secret_redactor=self.secret_redactor,
        )
        materialization = await AgentSnapshotCoordinator(
            providers,
            store=self.snapshot_store,
        ).materialize(
            AgentSnapshotMaterializationRequest(
                access=validated.access,
                candidate_id=validated.candidate_id,
                trial_id=validated.trial_id,
                state_mode=validated.state_mode,
                state_partition_fingerprint=authorization.state_partition_fingerprint,
            )
        )
        receipt = AgentBundleMaterializationReceipt.create(
            request=validated,
            authorization=authorization,
            materialization=materialization,
        )
        content = _canonical_json(receipt, "bundle_materialization_receipt")
        await self.object_store.put(
            _object_ref(
                content,
                kind=AgentBundleObjectKind.OPERATION_RECEIPT,
                schema_id="cayu.agent-bundle-materialization-receipt.v1",
            ),
            content,
        )
        return receipt

    async def capture_terminal(
        self,
        request: AgentSnapshotTerminalCaptureRequest,
        *,
        providers: Iterable[AgentSnapshotComponentProvider],
    ) -> AgentSnapshotTerminalCaptureReceipt:
        if type(request) is not AgentSnapshotTerminalCaptureRequest:
            raise TypeError("request must be an AgentSnapshotTerminalCaptureRequest.")
        validated = AgentSnapshotTerminalCaptureRequest.model_validate(
            request.model_dump(mode="json")
        )
        terminal_authority = self.terminal_authority
        if terminal_authority is None:
            raise AgentBundleError("terminal_authority_unavailable")
        parent_materialization = await self.snapshot_store.load_materialization(
            validated.parent_materialization_fingerprint
        )
        if parent_materialization is None:
            raise AgentBundleError("parent_materialization_missing")
        if parent_materialization.snapshot_fingerprint != (
            validated.capture.parent_snapshot_fingerprint
        ):
            raise AgentBundleError("terminal_capture_parent_mismatch")
        result = await self.snapshot_store.load_result(validated.parent_result_fingerprint)
        if result is None or result.fingerprint != validated.parent_result_fingerprint:
            raise AgentBundleError("terminal_result_missing")
        trial = await self.snapshot_store.load_trial(result.trial_fingerprint)
        if trial is None or trial.fingerprint != result.trial_fingerprint:
            raise AgentBundleError("terminal_trial_missing")
        if trial.materialization_fingerprint != parent_materialization.fingerprint:
            raise AgentBundleError("terminal_result_materialization_mismatch")
        if result.terminal_disposition is AgentSnapshotTerminalDisposition.OUTCOME_UNKNOWN:
            raise AgentBundleError("terminal_result_outcome_unknown")
        if (
            result.open_operation_ids
            or result.pending_approval_ids
            or result.provider_continuation_ids
        ):
            raise AgentBundleError("terminal_result_frontier_open")
        if (
            validated.profile is AgentSnapshotProfile.CONTINUING_AGENT
            and result.safe_frontier_fingerprint is None
        ):
            raise AgentBundleError("terminal_result_safe_frontier_missing")
        authorization = await terminal_authority.authorize_terminal_capture(
            request=validated,
            materialization=parent_materialization,
            trial=trial,
            result=result,
        )
        if type(authorization) is not AgentSnapshotTerminalAuthorization:
            raise AgentBundleError("terminal_authorization_invalid")
        authorization = AgentSnapshotTerminalAuthorization.model_validate(
            authorization.model_dump(mode="json")
        )
        if (
            authorization.request_fingerprint != validated.request_fingerprint
            or authorization.result_fingerprint != result.fingerprint
            or authorization.runtime_evidence_fingerprint != result.runtime_evidence_fingerprint
        ):
            raise AgentBundleError("terminal_authorization_mismatch")
        authorization_content = _canonical_json(
            authorization,
            "terminal_capture_authorization",
        )
        await self.object_store.put(
            _object_ref(
                authorization_content,
                kind=AgentBundleObjectKind.OPERATION_RECEIPT,
                schema_id="cayu.agent-snapshot-terminal-authorization.v1",
            ),
            authorization_content,
        )
        provider_items = tuple(providers)
        for provider in provider_items:
            if isinstance(provider, PortableAgentSnapshotComponentProvider) and (
                provider.package.profile is not validated.profile
            ):
                raise AgentBundleError("terminal_component_profile_mismatch")
        descendant = await AgentSnapshotCoordinator(
            provider_items,
            store=self.snapshot_store,
        ).capture(validated.capture)
        receipt = AgentSnapshotTerminalCaptureReceipt.create(
            request=validated,
            result=result,
            authorization=authorization,
            descendant_snapshot_ref=descendant.ref,
        )
        content = _canonical_json(receipt, "terminal_capture_receipt")
        await self.object_store.put(
            _object_ref(
                content,
                kind=AgentBundleObjectKind.OPERATION_RECEIPT,
                schema_id="cayu.agent-snapshot-terminal-capture-receipt.v1",
            ),
            content,
        )
        return receipt

    @staticmethod
    def _read_and_verify_bundle_directory(
        source: Path,
    ) -> tuple[AgentBundle, dict[str, bytes | Path]]:
        if source.is_symlink() or not source.is_dir():
            raise AgentBundleError("bundle_source_not_directory")
        index_path = source / AGENT_BUNDLE_INDEX_FILENAME
        if index_path.is_symlink() or not index_path.is_file():
            raise AgentBundleError("bundle_index_missing")
        if index_path.stat().st_size > AGENT_BUNDLE_MAX_INDEX_BYTES:
            raise AgentBundleError("bundle_index_size_limit_exceeded")
        index_bytes = index_path.read_bytes()
        try:
            bundle = AgentBundle.model_validate_json(index_bytes)
        except Exception as error:
            raise AgentBundleError("bundle_index_invalid") from error
        if _canonical_json(bundle, "agent_bundle") != index_bytes:
            raise AgentBundleError("bundle_index_not_canonical")
        expected_paths = {AGENT_BUNDLE_INDEX_FILENAME}
        content_by_digest: dict[str, bytes | Path] = {}
        total_bytes = len(index_bytes)
        by_digest = {reference.digest: reference for reference in bundle.closure}
        for digest in bundle.transferred_digests:
            reference = by_digest[digest]
            path = _safe_object_path(source, digest)
            expected_paths.add(path.relative_to(source).as_posix())
            if path.is_symlink() or not path.is_file():
                raise AgentBundleError("bundle_object_missing", digest)
            size = path.stat().st_size
            if size != reference.byte_count or size > AGENT_BUNDLE_MAX_OBJECT_BYTES:
                raise AgentBundleError("bundle_object_size_mismatch", digest)
            total_bytes += size
            if total_bytes > AGENT_BUNDLE_MAX_TOTAL_BYTES:
                raise AgentBundleError("bundle_total_size_limit_exceeded")
            if reference.kind is AgentBundleObjectKind.COMPONENT_BLOB:
                actual_digest, actual_size = _file_digest_and_size(
                    path,
                    max_bytes=reference.byte_count,
                )
                if actual_digest != digest or actual_size != size:
                    raise AgentBundleError("bundle_object_integrity_mismatch", digest)
                content_by_digest[digest] = path
                continue
            content = path.read_bytes()
            if len(content) != size:
                raise AgentBundleError("bundle_object_integrity_mismatch", digest)
            if reference.kind is AgentBundleObjectKind.SNAPSHOT_NODE:
                try:
                    node = AgentSnapshotNode.model_validate_json(content)
                except Exception as error:
                    raise AgentBundleError("bundle_snapshot_node_invalid", digest) from error
                if node.digest != digest or _canonical_json(node, "agent_snapshot_node") != content:
                    raise AgentBundleError("bundle_object_integrity_mismatch", digest)
            elif _digest(content) != digest:
                raise AgentBundleError("bundle_object_integrity_mismatch", digest)
            content_by_digest[digest] = content
        for root, directory_names, filenames in os.walk(source, followlinks=False):
            root_path = Path(root)
            if root_path.is_symlink():
                raise AgentBundleError("bundle_path_symlink")
            for name in directory_names:
                candidate = root_path / name
                if candidate.is_symlink():
                    raise AgentBundleError("bundle_path_symlink")
            for name in filenames:
                candidate = root_path / name
                if candidate.is_symlink() or not candidate.is_file():
                    raise AgentBundleError("bundle_path_not_regular")
                relative = candidate.relative_to(source).as_posix()
                if relative not in expected_paths:
                    raise AgentBundleError("bundle_contains_extra_object", relative)
        return bundle, content_by_digest

    @staticmethod
    def _verify_snapshot_and_component_closure(
        bundle: AgentBundle,
        snapshot: AgentSnapshot,
        content_by_digest: Mapping[str, bytes | Path | None],
    ) -> None:
        by_digest = {reference.digest: reference for reference in bundle.closure}
        expected_node_digests = {node.digest for node in snapshot.merkle_nodes()}
        bundled_node_digests = {
            reference.digest
            for reference in bundle.closure
            if reference.kind is AgentBundleObjectKind.SNAPSHOT_NODE
        }
        if expected_node_digests != bundled_node_digests:
            raise AgentBundleError("snapshot_node_closure_mismatch")
        for node in snapshot.merkle_nodes():
            reference = by_digest[node.digest]
            expected = _canonical_json(node, "agent_snapshot_node")
            actual = content_by_digest.get(node.digest)
            if actual is None:
                actual = expected
            if reference.byte_count != len(expected) or actual != expected:
                raise AgentBundleError("snapshot_node_content_mismatch", node.digest)
        expected_component_digests = {
            component.logical.fingerprint for component in snapshot.components
        }
        bundled_component_digests = {
            reference.digest
            for reference in bundle.closure
            if reference.kind is AgentBundleObjectKind.COMPONENT_MANIFEST
        }
        if expected_component_digests != bundled_component_digests:
            raise AgentBundleError("component_manifest_closure_mismatch")
        expected_blob_digests: set[str] = set()
        expected_external_bindings: dict[tuple[str, str, str], AgentExternalBindingRequirement] = {}
        for component in snapshot.components:
            content = content_by_digest[component.logical.fingerprint]
            if not isinstance(content, bytes):
                raise AgentBundleError("component_manifest_missing")
            package = _component_package_from_bytes(content)
            if (
                package.kind is not component.kind
                or package.provider_id != component.provider_id
                or package.profile is not bundle.profile
            ):
                raise AgentBundleError("component_manifest_identity_mismatch")
            expected_blob_digests.update(file.digest for file in package.files)
            for binding in package.external_bindings:
                key = (
                    binding.kind.value,
                    binding.name,
                    binding.requirement_fingerprint,
                )
                expected_external_bindings[key] = binding
            for file in package.files:
                reference = by_digest.get(file.digest)
                if (
                    reference is None
                    or reference.kind is not AgentBundleObjectKind.COMPONENT_BLOB
                    or reference.byte_count != file.byte_count
                ):
                    raise AgentBundleError("component_blob_integrity_mismatch")
        bundled_blob_digests = {
            reference.digest
            for reference in bundle.closure
            if reference.kind is AgentBundleObjectKind.COMPONENT_BLOB
        }
        if expected_blob_digests != bundled_blob_digests:
            raise AgentBundleError("component_blob_closure_mismatch")
        expected_binding_tuple = tuple(
            expected_external_bindings[key] for key in sorted(expected_external_bindings)
        )
        if expected_binding_tuple != bundle.external_bindings:
            raise AgentBundleError("external_binding_closure_mismatch")


__all__ = [
    "AGENT_BUNDLE_INDEX_FILENAME",
    "AGENT_BUNDLE_MAX_INDEX_BYTES",
    "AGENT_BUNDLE_MAX_OBJECTS",
    "AGENT_BUNDLE_MAX_OBJECT_BYTES",
    "AGENT_BUNDLE_MAX_TOTAL_BYTES",
    "AGENT_BUNDLE_OBJECT_DIRECTORY",
    "AGENT_BUNDLE_RECORD_TYPE",
    "AGENT_BUNDLE_SCHEMA_VERSION",
    "AgentBundle",
    "AgentBundleCoordinator",
    "AgentBundleError",
    "AgentBundleExportReceipt",
    "AgentBundleImportReceipt",
    "AgentBundleInventory",
    "AgentBundleMaterializationAuthority",
    "AgentBundleMaterializationAuthorization",
    "AgentBundleMaterializationReceipt",
    "AgentBundleMaterializationRequest",
    "AgentBundleMode",
    "AgentBundleObjectKind",
    "AgentBundleObjectRef",
    "AgentBundleSizeReport",
    "AgentExternalBindingKind",
    "AgentExternalBindingRequirement",
    "AgentExternalBindingResolution",
    "AgentMaterializationFreshIdentities",
    "AgentSnapshotComponentFile",
    "AgentSnapshotComponentPackage",
    "AgentSnapshotMaterializationMode",
    "AgentSnapshotObjectStore",
    "AgentSnapshotProfile",
    "AgentSnapshotSessionDisposition",
    "AgentSnapshotTerminalAuthority",
    "AgentSnapshotTerminalAuthorization",
    "AgentSnapshotTerminalCaptureReceipt",
    "AgentSnapshotTerminalCaptureRequest",
    "FileSystemAgentSnapshotObjectStore",
    "PortableAgentSnapshotComponentProvider",
    "agent_snapshot_component_package",
    "load_portable_agent_snapshot_component_providers",
    "store_agent_snapshot_component_package",
]
