"""Identity-bound immutable inputs shared across execution environments."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import posixpath
import re
import shutil
import stat
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._filesystem_lock import cooperative_path_lock
from cayu._task_wait import await_shielded_task_outcome
from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank

IMMUTABLE_INPUT_FORMAT_VERSION = "cayu.immutable-tree.v1"
DEFAULT_IMMUTABLE_INPUT_MAX_FILES = 100_000
DEFAULT_IMMUTABLE_INPUT_MAX_FILE_BYTES = 64 * 1024 * 1024
DEFAULT_IMMUTABLE_INPUT_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_ATTACHMENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_CONTAINER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_STORE_SCHEMA_VERSION = 1
_DOCKER_MOUNT_AUTHORITY = object()


class ImmutableInputProjectionCapability(StrEnum):
    """Truthful environment-input behavior supplied by one adapter."""

    SHARED_READ_ONLY = "shared_read_only"
    MUTABLE_SYNC_BINDING = "mutable_sync_binding"
    BOUNDED_COPY = "mutable_sync_binding"
    WORKSPACE_MATERIALIZATION = "workspace_materialization"
    UNSUPPORTED = "unsupported"


class ImmutableInputProjectionUnsupportedError(RuntimeError):
    """An adapter cannot provide the requested immutable-input guarantees."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = require_durable_clean_nonblank(reason_code, "reason_code")
        super().__init__(f"Immutable input projection is unsupported: {self.reason_code}.")


class ImmutableInputMutationError(RuntimeError):
    """A supposedly immutable materialization no longer matches its identity."""

    def __init__(self, projection_fingerprint: str, reason_code: str) -> None:
        self.projection_fingerprint = _require_sha256(
            projection_fingerprint, "projection_fingerprint"
        )
        self.reason_code = require_durable_clean_nonblank(reason_code, "reason_code")
        super().__init__(
            "Immutable input materialization failed verification: "
            f"{self.reason_code} ({self.projection_fingerprint})."
        )


class ImmutableInputAttachmentStateError(RuntimeError):
    """An attachment identifier was reused after its durable release boundary."""

    def __init__(self, attachment_id: str, state: str) -> None:
        self.attachment_id = _require_attachment_id(attachment_id)
        self.state = require_durable_clean_nonblank(state, "state")
        super().__init__(
            f"Immutable input attachment {self.attachment_id!r} is already {self.state}."
        )


class ImmutableInputAdapterCapability(BaseModel):
    """Content-free declaration of one adapter's projection mechanism."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    adapter: str = Field(max_length=128)
    capability: ImmutableInputProjectionCapability
    mechanism: str | None = Field(default=None, max_length=128)
    read_only_enforced: StrictBool = False
    shared_materialization: StrictBool = False
    durable_recovery: StrictBool = False
    explicit_copy_fallback: StrictBool = False
    reason_code: str | None = Field(default=None, max_length=128)

    @field_validator("adapter")
    @classmethod
    def validate_adapter(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "adapter")

    @model_validator(mode="after")
    def validate_claims(self) -> Self:
        if self.capability is ImmutableInputProjectionCapability.SHARED_READ_ONLY:
            if not (
                self.mechanism
                and self.read_only_enforced
                and self.shared_materialization
                and self.durable_recovery
            ):
                raise ValueError("Shared read-only capability requires all supporting claims.")
            if self.reason_code is not None:
                raise ValueError("Supported immutable input capability cannot carry a reason.")
        elif self.capability is ImmutableInputProjectionCapability.MUTABLE_SYNC_BINDING:
            if not self.explicit_copy_fallback or self.read_only_enforced:
                raise ValueError(
                    "Mutable sync fallback must be explicit and cannot claim a read-only mount."
                )
        elif self.capability is ImmutableInputProjectionCapability.WORKSPACE_MATERIALIZATION:
            if self.read_only_enforced or self.shared_materialization:
                raise ValueError("Workspace materialization cannot claim immutable sharing.")
        elif self.reason_code is None:
            raise ValueError("Unsupported immutable input capability requires a reason code.")
        return self


class ImmutableInputProjection(BaseModel):
    """Exact content and policy identity for one read-only target projection."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    content_root: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    logical_bytes: StrictInt = Field(ge=0, le=DEFAULT_IMMUTABLE_INPUT_MAX_TOTAL_BYTES)
    file_count: StrictInt = Field(ge=0, le=DEFAULT_IMMUTABLE_INPUT_MAX_FILES)
    target_path: str = Field(max_length=4096)
    policy_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    runtime_compatibility_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    authorization_scope_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    format_version: Literal["cayu.immutable-tree.v1"] = IMMUTABLE_INPUT_FORMAT_VERSION
    max_files: StrictInt = Field(default=DEFAULT_IMMUTABLE_INPUT_MAX_FILES, ge=1)
    max_file_bytes: StrictInt = Field(default=DEFAULT_IMMUTABLE_INPUT_MAX_FILE_BYTES, ge=1)
    max_total_bytes: StrictInt = Field(default=DEFAULT_IMMUTABLE_INPUT_MAX_TOTAL_BYTES, ge=1)

    @field_validator(
        "content_root",
        "policy_fingerprint",
        "runtime_compatibility_fingerprint",
        "authorization_scope_fingerprint",
    )
    @classmethod
    def validate_sha256(cls, value: str, info: Any) -> str:
        return _require_sha256(value, info.field_name)

    @field_validator("target_path")
    @classmethod
    def validate_target_path(cls, value: str) -> str:
        target = require_durable_clean_nonblank(value, "target_path")
        if not posixpath.isabs(target):
            raise ValueError("Immutable input target_path must be absolute.")
        normalized = posixpath.normpath(target)
        reserved_roots = ("/proc", "/sys", "/dev", "/tmp", "/workspace")
        if normalized == "/" or any(
            normalized == root or normalized.startswith(root + "/") for root in reserved_roots
        ):
            raise ValueError("Immutable input target_path must be a dedicated bounded path.")
        if "," in normalized:
            raise ValueError("Immutable input target_path must not contain commas.")
        return normalized

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.file_count > self.max_files:
            raise ValueError("Immutable input file_count exceeds max_files.")
        if self.logical_bytes > self.max_total_bytes:
            raise ValueError("Immutable input logical_bytes exceeds max_total_bytes.")
        if self.max_file_bytes > self.max_total_bytes:
            raise ValueError("Immutable input max_file_bytes cannot exceed max_total_bytes.")
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.model_dump(mode="json"), "immutable_input_projection")


@dataclass(frozen=True, slots=True)
class LocalImmutableInput:
    """A local source whose bytes must verify against an exact projection."""

    root: Path
    projection: ImmutableInputProjection

    def __post_init__(self) -> None:
        root = Path(self.root).resolve()
        if not root.is_dir():
            raise ValueError("Local immutable input root must be an existing directory.")
        if not isinstance(self.projection, ImmutableInputProjection):
            raise TypeError("projection must be an ImmutableInputProjection.")
        object.__setattr__(self, "root", root)


@dataclass(frozen=True, slots=True, init=False)
class DockerImmutableInputMount:
    """Opaque manager-issued authority for one exact Docker read-only mount."""

    source_path: str
    target_path: str
    projection_fingerprint: str
    attachment_id: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("Docker immutable mounts must be issued by ImmutableInputStore.")

    @classmethod
    def _issue(
        cls,
        *,
        source_path: str,
        target_path: str,
        projection_fingerprint: str,
        attachment_id: str,
        authority: object,
    ) -> DockerImmutableInputMount:
        if authority is not _DOCKER_MOUNT_AUTHORITY:
            raise TypeError("Docker immutable mounts must be issued by ImmutableInputStore.")
        value = object.__new__(cls)
        object.__setattr__(value, "source_path", source_path)
        object.__setattr__(value, "target_path", target_path)
        object.__setattr__(value, "projection_fingerprint", projection_fingerprint)
        object.__setattr__(value, "attachment_id", attachment_id)
        return value


@dataclass(frozen=True, slots=True, init=False)
class ImmutableInputAttachment:
    """Durable ownership of a verified materialization by one environment."""

    attachment_id: str
    owner_id: str
    projection: ImmutableInputProjection
    materialization_path: Path
    reused: bool

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("Immutable input attachments must be issued by ImmutableInputStore.")

    @classmethod
    def _issue(
        cls,
        *,
        attachment_id: str,
        owner_id: str,
        projection: ImmutableInputProjection,
        materialization_path: Path,
        reused: bool,
        authority: object,
    ) -> ImmutableInputAttachment:
        if authority is not _DOCKER_MOUNT_AUTHORITY:
            raise TypeError("Immutable input attachments must be issued by ImmutableInputStore.")
        value = object.__new__(cls)
        object.__setattr__(value, "attachment_id", attachment_id)
        object.__setattr__(value, "owner_id", owner_id)
        object.__setattr__(value, "projection", projection)
        object.__setattr__(value, "materialization_path", materialization_path)
        object.__setattr__(value, "reused", reused)
        return value

    def docker_mount(self) -> DockerImmutableInputMount:
        return DockerImmutableInputMount._issue(
            source_path=str(self.materialization_path),
            target_path=self.projection.target_path,
            projection_fingerprint=self.projection.fingerprint,
            attachment_id=self.attachment_id,
            authority=_DOCKER_MOUNT_AUTHORITY,
        )


class ImmutableInputDiagnostic(BaseModel):
    """Bounded content-free diagnostics for one shared materialization."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    projection_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    content_root: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    logical_bytes: StrictInt = Field(ge=0)
    physical_bytes: StrictInt = Field(ge=0)
    reference_count: StrictInt = Field(ge=0)
    attachment_count: StrictInt = Field(ge=0)
    reuse_count: StrictInt = Field(ge=0)
    cleanup_state: Literal["retained", "eligible", "collecting", "removed"]
    wait_reason: Literal["none", "materialization_lock"] = "none"


def inspect_local_immutable_input(
    root: str | os.PathLike[str],
    *,
    target_path: str,
    policy_fingerprint: str,
    runtime_compatibility_fingerprint: str,
    authorization_scope_fingerprint: str,
    max_files: int = DEFAULT_IMMUTABLE_INPUT_MAX_FILES,
    max_file_bytes: int = DEFAULT_IMMUTABLE_INPUT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_IMMUTABLE_INPUT_MAX_TOTAL_BYTES,
) -> LocalImmutableInput:
    """Inspect a bounded local tree and return its exact projection identity."""

    source = Path(root).resolve()
    manifest = _tree_manifest(
        source,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
    )
    projection = ImmutableInputProjection(
        content_root=_manifest_root(manifest),
        logical_bytes=_manifest_logical_bytes(manifest),
        file_count=len(manifest),
        target_path=target_path,
        policy_fingerprint=policy_fingerprint,
        runtime_compatibility_fingerprint=runtime_compatibility_fingerprint,
        authorization_scope_fingerprint=authorization_scope_fingerprint,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
    )
    return LocalImmutableInput(root=source, projection=projection)


class ImmutableInputStore:
    """Durable convergent registry and materializer for immutable local trees."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        managed_root = Path(root).resolve()
        managed_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not managed_root.is_dir():
            raise ValueError("Immutable input store root must be a directory.")
        os.chmod(managed_root, 0o700)
        self.root = managed_root
        self._objects = managed_root / "objects"
        self._objects.mkdir(mode=0o700, exist_ok=True)
        os.chmod(self._objects, 0o700)
        self._registry = managed_root / "registry.json"
        self._thread_lock = threading.RLock()

    async def attach(
        self,
        source: LocalImmutableInput,
        *,
        attachment_id: str,
        owner_id: str,
    ) -> ImmutableInputAttachment:
        """Materialize once and durably attach one exact environment owner."""

        task = asyncio.create_task(
            asyncio.to_thread(
                self.attach_sync,
                source,
                attachment_id=attachment_id,
                owner_id=owner_id,
            )
        )
        outcome = await await_shielded_task_outcome(task)
        if outcome.error is not None:
            raise outcome.error
        if outcome.cancellation is not None:
            if outcome.result is not None:
                await asyncio.to_thread(self.release_sync, attachment_id)
            raise outcome.cancellation
        assert isinstance(outcome.result, ImmutableInputAttachment)
        return outcome.result

    def attach_sync(
        self,
        source: LocalImmutableInput,
        *,
        attachment_id: str,
        owner_id: str,
    ) -> ImmutableInputAttachment:
        if not isinstance(source, LocalImmutableInput):
            raise TypeError("source must be a LocalImmutableInput.")
        if self.root.is_relative_to(source.root) or source.root.is_relative_to(self.root):
            raise ValueError("Immutable input source and managed store must not overlap.")
        attachment_id = _require_attachment_id(attachment_id)
        owner_id = require_durable_clean_nonblank(owner_id, "owner_id")
        projection = source.projection
        with self._exclusive_registry() as registry:
            existing_attachment = registry["attachments"].get(attachment_id)
            if existing_attachment is not None:
                if (
                    existing_attachment["owner_id"] != owner_id
                    or existing_attachment["projection_fingerprint"] != projection.fingerprint
                ):
                    raise RuntimeError("Immutable input attachment identity conflicts.")
                if existing_attachment["state"] != "active":
                    raise ImmutableInputAttachmentStateError(
                        attachment_id,
                        str(existing_attachment["state"]),
                    )
                materialization = self._require_materialization(registry, projection)
                self._verify_materialization(Path(materialization["path"]), projection)
                return self._attachment(
                    attachment_id, owner_id, projection, materialization, reused=True
                )

            materialization = registry["materializations"].get(projection.fingerprint)
            reused = materialization is not None
            if materialization is not None and materialization.get("cleanup_state") == "collecting":
                path = self._materialization_path(projection.fingerprint)
                if path.exists():
                    self._verify_materialization(path, projection)
                    materialization["cleanup_state"] = "retained"
                else:
                    del registry["materializations"][projection.fingerprint]
                    materialization = None
                    reused = False
            if materialization is None:
                materialization = self._materialize(source)
                registry["materializations"][projection.fingerprint] = materialization
            else:
                self._require_materialization(registry, projection)
                self._verify_materialization(Path(materialization["path"]), projection)
                materialization["reuse_count"] = int(materialization["reuse_count"]) + 1
            registry["attachments"][attachment_id] = {
                "owner_id": owner_id,
                "projection_fingerprint": projection.fingerprint,
                "state": "active",
            }
            self._write_registry(registry)
            return self._attachment(
                attachment_id, owner_id, projection, materialization, reused=reused
            )

    async def release(self, attachment_id: str) -> None:
        task = asyncio.create_task(asyncio.to_thread(self.release_sync, attachment_id))
        outcome = await await_shielded_task_outcome(task)
        if outcome.error is not None:
            raise outcome.error
        if outcome.cancellation is not None:
            raise outcome.cancellation

    async def mark_container_closing(
        self,
        attachments: tuple[ImmutableInputAttachment, ...],
        *,
        container_id: str,
    ) -> None:
        task = asyncio.create_task(
            asyncio.to_thread(
                self.mark_container_closing_sync,
                attachments,
                container_id=container_id,
            )
        )
        outcome = await await_shielded_task_outcome(task)
        if outcome.error is not None:
            raise outcome.error
        if outcome.cancellation is not None:
            raise outcome.cancellation

    def mark_container_closing_sync(
        self,
        attachments: tuple[ImmutableInputAttachment, ...],
        *,
        container_id: str,
    ) -> None:
        values = tuple(attachments)
        if not values or any(type(value) is not ImmutableInputAttachment for value in values):
            raise TypeError("Container cleanup requires manager-issued immutable attachments.")
        container_id = _require_container_id(container_id)
        with self._exclusive_registry() as registry:
            for attachment in values:
                retained = registry["attachments"].get(attachment.attachment_id)
                if (
                    retained is None
                    or retained.get("owner_id") != attachment.owner_id
                    or retained.get("projection_fingerprint") != attachment.projection.fingerprint
                ):
                    raise RuntimeError("Immutable input cleanup authority changed.")
                state = retained.get("state")
                if state == "closing" and retained.get("container_id") == container_id:
                    continue
                if state != "active" or "container_id" in retained:
                    raise ImmutableInputAttachmentStateError(
                        attachment.attachment_id,
                        str(state),
                    )
            for attachment in values:
                retained = registry["attachments"][attachment.attachment_id]
                retained["state"] = "closing"
                retained["container_id"] = container_id
            self._write_registry(registry)

    async def interrupted_cleanup_container_id(
        self,
        attachment_ids: tuple[str, ...],
    ) -> str | None:
        task = asyncio.create_task(
            asyncio.to_thread(
                self.interrupted_cleanup_container_id_sync,
                attachment_ids,
            )
        )
        outcome = await await_shielded_task_outcome(task)
        if outcome.error is not None:
            raise outcome.error
        if outcome.cancellation is not None:
            raise outcome.cancellation
        assert outcome.result is None or isinstance(outcome.result, str)
        return outcome.result

    def interrupted_cleanup_container_id_sync(
        self,
        attachment_ids: tuple[str, ...],
    ) -> str | None:
        identifiers = tuple(_require_attachment_id(value) for value in attachment_ids)
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError("Immutable cleanup attachment ids must be nonempty and unique.")
        with self._exclusive_registry() as registry:
            states = [registry["attachments"].get(value) for value in identifiers]
            closing = [
                value for value in states if value is not None and value.get("state") == "closing"
            ]
            if not closing:
                return None
            container_ids = {value.get("container_id") for value in closing}
            if len(container_ids) != 1:
                raise RuntimeError("Immutable input cleanup container identity changed.")
            container_id = next(iter(container_ids))
            if type(container_id) is not str:
                raise RuntimeError("Immutable input cleanup lost its container identity.")
            return _require_container_id(container_id)

    async def reconcile_interrupted_container_cleanup(
        self,
        attachment_ids: tuple[str, ...],
        *,
        container_id: str,
        container_exists: bool,
    ) -> bool:
        task = asyncio.create_task(
            asyncio.to_thread(
                self.reconcile_interrupted_container_cleanup_sync,
                attachment_ids,
                container_id=container_id,
                container_exists=container_exists,
            )
        )
        outcome = await await_shielded_task_outcome(task)
        if outcome.error is not None:
            raise outcome.error
        if outcome.cancellation is not None:
            raise outcome.cancellation
        assert type(outcome.result) is bool
        return outcome.result

    def reconcile_interrupted_container_cleanup_sync(
        self,
        attachment_ids: tuple[str, ...],
        *,
        container_id: str,
        container_exists: bool,
    ) -> bool:
        identifiers = tuple(_require_attachment_id(value) for value in attachment_ids)
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError("Immutable cleanup attachment ids must be nonempty and unique.")
        container_id = _require_container_id(container_id)
        if type(container_exists) is not bool:
            raise TypeError("container_exists must be a bool.")
        with self._exclusive_registry() as registry:
            retained = [registry["attachments"].get(value) for value in identifiers]
            if any(value is None for value in retained):
                raise RuntimeError("Immutable input cleanup attachment disappeared.")
            values = [value for value in retained if value is not None]
            closing = [value for value in values if value.get("state") == "closing"]
            if any(value.get("container_id") != container_id for value in closing):
                raise RuntimeError("Immutable input cleanup container identity changed.")
            if container_exists:
                if len(closing) != len(values):
                    raise RuntimeError(
                        "A live container retained partially released immutable inputs."
                    )
                for value in values:
                    value["state"] = "active"
                    value.pop("container_id", None)
                self._write_registry(registry)
                return True
            for value in closing:
                value["state"] = "released"
                value.pop("container_id", None)
            if closing:
                self._write_registry(registry)
            return False

    def release_sync(self, attachment_id: str) -> None:
        attachment_id = _require_attachment_id(attachment_id)
        with self._exclusive_registry() as registry:
            attachment = registry["attachments"].get(attachment_id)
            if attachment is None or attachment["state"] == "released":
                return
            attachment["state"] = "released"
            attachment.pop("container_id", None)
            self._write_registry(registry)

    def inspect(self) -> tuple[ImmutableInputDiagnostic, ...]:
        with self._exclusive_registry() as registry:
            diagnostics: list[ImmutableInputDiagnostic] = []
            for fingerprint, materialization in sorted(registry["materializations"].items()):
                attachments = [
                    item
                    for item in registry["attachments"].values()
                    if item["projection_fingerprint"] == fingerprint
                ]
                active = sum(item["state"] in {"active", "closing"} for item in attachments)
                diagnostics.append(
                    ImmutableInputDiagnostic(
                        projection_fingerprint=fingerprint,
                        content_root=materialization["projection"]["content_root"],
                        logical_bytes=materialization["projection"]["logical_bytes"],
                        physical_bytes=materialization["physical_bytes"],
                        reference_count=active,
                        attachment_count=len(attachments),
                        reuse_count=materialization["reuse_count"],
                        cleanup_state=(
                            "collecting"
                            if materialization.get("cleanup_state") == "collecting"
                            else ("retained" if active else "eligible")
                        ),
                    )
                )
            return tuple(diagnostics)

    def collect(self, projection_fingerprint: str) -> bool:
        """Remove one unreferenced materialization; active identities fail closed."""

        projection_fingerprint = _require_sha256(projection_fingerprint, "projection_fingerprint")
        with self._exclusive_registry() as registry:
            materialization = registry["materializations"].get(projection_fingerprint)
            if materialization is None:
                return False
            if any(
                item["projection_fingerprint"] == projection_fingerprint
                and item["state"] in {"active", "closing"}
                for item in registry["attachments"].values()
            ):
                raise RuntimeError("Cannot collect a referenced immutable input materialization.")
            expected_path = self._materialization_path(projection_fingerprint)
            materialization_path = Path(str(materialization.get("path", "")))
            if materialization_path != expected_path:
                raise ImmutableInputMutationError(
                    projection_fingerprint,
                    "registry_path_conflict",
                )
            materialization["cleanup_state"] = "collecting"
            self._write_registry(registry)
            _remove_tree(materialization_path)
            del registry["materializations"][projection_fingerprint]
            registry["attachments"] = {
                key: item
                for key, item in registry["attachments"].items()
                if item["projection_fingerprint"] != projection_fingerprint
            }
            self._write_registry(registry)
            return True

    def _materialize(self, source: LocalImmutableInput) -> dict[str, Any]:
        projection = source.projection
        manifest = _tree_manifest(
            source.root,
            max_files=projection.max_files,
            max_file_bytes=projection.max_file_bytes,
            max_total_bytes=projection.max_total_bytes,
        )
        if (
            _manifest_root(manifest) != projection.content_root
            or len(manifest) != projection.file_count
            or _manifest_logical_bytes(manifest) != projection.logical_bytes
        ):
            raise ImmutableInputMutationError(projection.fingerprint, "source_identity_drift")
        destination = self._materialization_path(projection.fingerprint)
        if destination.exists():
            self._verify_materialization(destination, projection)
            return {
                "projection": projection.model_dump(mode="json"),
                "path": str(destination),
                "physical_bytes": _physical_tree_bytes(destination),
                "reuse_count": 0,
                "cleanup_state": "retained",
            }
        temporary = Path(tempfile.mkdtemp(prefix=".publishing-", dir=self._objects))
        try:
            for entry in manifest:
                relative = str(entry["path"])
                source_path = source.root / relative
                target_path = temporary / relative
                target_path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                _copy_verified_file(source_path, target_path, entry)
            for directory, directories, _files in os.walk(temporary, topdown=False):
                for name in directories:
                    os.chmod(Path(directory) / name, 0o555)
            self._verify_materialization(temporary, projection)
            os.replace(temporary, destination)
            os.chmod(destination, 0o555)
            _fsync_directory(self._objects)
        except BaseException:
            if temporary.exists():
                _remove_tree(temporary, ignore_errors=True)
            raise
        return {
            "projection": projection.model_dump(mode="json"),
            "path": str(destination),
            "physical_bytes": _physical_tree_bytes(destination),
            "reuse_count": 0,
            "cleanup_state": "retained",
        }

    def _require_materialization(
        self,
        registry: dict[str, Any],
        projection: ImmutableInputProjection,
    ) -> dict[str, Any]:
        value = registry["materializations"].get(projection.fingerprint)
        if value is None or value["projection"] != projection.model_dump(mode="json"):
            raise ImmutableInputMutationError(projection.fingerprint, "registry_identity_conflict")
        expected_path = self._materialization_path(projection.fingerprint)
        if Path(str(value.get("path", ""))) != expected_path:
            raise ImmutableInputMutationError(projection.fingerprint, "registry_path_conflict")
        return value

    def _materialization_path(self, projection_fingerprint: str) -> Path:
        fingerprint = _require_sha256(projection_fingerprint, "projection_fingerprint")
        return self._objects / fingerprint.removeprefix("sha256:")

    def _verify_materialization(
        self,
        path: Path,
        projection: ImmutableInputProjection,
    ) -> None:
        try:
            manifest = _tree_manifest(
                path,
                max_files=projection.max_files,
                max_file_bytes=projection.max_file_bytes,
                max_total_bytes=projection.max_total_bytes,
            )
        except (OSError, ValueError) as exc:
            raise ImmutableInputMutationError(
                projection.fingerprint, "materialization_unreadable"
            ) from exc
        if _manifest_root(manifest) != projection.content_root:
            raise ImmutableInputMutationError(projection.fingerprint, "materialization_drift")

    def _attachment(
        self,
        attachment_id: str,
        owner_id: str,
        projection: ImmutableInputProjection,
        materialization: Mapping[str, Any],
        *,
        reused: bool,
    ) -> ImmutableInputAttachment:
        return ImmutableInputAttachment._issue(
            attachment_id=attachment_id,
            owner_id=owner_id,
            projection=projection,
            materialization_path=Path(str(materialization["path"])),
            reused=reused,
            authority=_DOCKER_MOUNT_AUTHORITY,
        )

    @contextmanager
    def _exclusive_registry(self) -> Iterator[dict[str, Any]]:
        with (
            self._thread_lock,
            cooperative_path_lock(
                self.root,
                "registry",
                lock_directory_name="cayu-immutable-input-locks",
            ),
        ):
            self._recover_orphaned_publications()
            yield self._read_registry()

    def _recover_orphaned_publications(self) -> None:
        removed = False
        for path in sorted(self._objects.iterdir(), key=lambda value: value.name):
            if not path.name.startswith(".publishing-"):
                continue
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError("Immutable input publication staging entry is unsafe.")
            _remove_tree(path)
            removed = True
        if removed:
            _fsync_directory(self._objects)

    def _read_registry(self) -> dict[str, Any]:
        if not self._registry.exists():
            return {
                "schema_version": _STORE_SCHEMA_VERSION,
                "materializations": {},
                "attachments": {},
            }
        try:
            value = json.loads(self._registry.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("Immutable input registry is unreadable.") from exc
        if (
            type(value) is not dict
            or value.get("schema_version") != _STORE_SCHEMA_VERSION
            or type(value.get("materializations")) is not dict
            or type(value.get("attachments")) is not dict
        ):
            raise RuntimeError("Immutable input registry has an unsupported schema.")
        return value

    def _write_registry(self, registry: Mapping[str, Any]) -> None:
        payload = canonical_durable_json_bytes(dict(registry), "immutable_input_registry")
        descriptor, temporary_name = tempfile.mkstemp(prefix=".registry-", dir=self.root)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self._registry)
            _fsync_directory(self.root)
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name)
            raise


def _fingerprint(value: object, field_name: str) -> str:
    return "sha256:" + hashlib.sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()


def _require_sha256(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 identity.")
    return value


def _require_attachment_id(value: str) -> str:
    value = require_durable_clean_nonblank(value, "attachment_id")
    if _ATTACHMENT_PATTERN.fullmatch(value) is None:
        raise ValueError("attachment_id contains unsupported characters or is too long.")
    return value


def _require_container_id(value: str) -> str:
    value = require_durable_clean_nonblank(value, "container_id")
    if _CONTAINER_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("container_id must be a full lowercase Docker container ID.")
    return value


def _tree_manifest(
    root: Path,
    *,
    max_files: int,
    max_file_bytes: int,
    max_total_bytes: int,
) -> list[dict[str, object]]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Immutable input root must be a real directory.")
    entries: list[dict[str, object]] = []
    total = 0
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names.sort()
        file_names.sort()
        directory_path = Path(directory)
        for name in tuple(directory_names):
            path = directory_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ValueError("Immutable inputs cannot contain symbolic links.")
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError("Immutable inputs cannot contain special directory entries.")
        for name in file_names:
            path = directory_path / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Immutable inputs may contain only regular files.")
            if info.st_size > max_file_bytes:
                raise ValueError("Immutable input file exceeds max_file_bytes.")
            total += info.st_size
            if total > max_total_bytes:
                raise ValueError("Immutable input tree exceeds max_total_bytes.")
            relative = path.relative_to(root).as_posix()
            digest, observed = _hash_regular_file(path, info)
            entries.append(
                {
                    "path": relative,
                    "size": observed.st_size,
                    "sha256": "sha256:" + digest,
                    "executable": bool(observed.st_mode & 0o111),
                }
            )
            if len(entries) > max_files:
                raise ValueError("Immutable input tree exceeds max_files.")
    entries.sort(key=lambda item: str(item["path"]))
    return entries


def _manifest_root(manifest: list[dict[str, object]]) -> str:
    return _fingerprint(
        {"format_version": IMMUTABLE_INPUT_FORMAT_VERSION, "files": manifest},
        "immutable_input_manifest",
    )


def _manifest_logical_bytes(manifest: list[dict[str, object]]) -> int:
    total = 0
    for entry in manifest:
        size = entry.get("size")
        if type(size) is not int:
            raise TypeError("Immutable input manifest size must be an integer.")
        total += size
    return total


def _copy_verified_file(source: Path, target: Path, expected: Mapping[str, object]) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    digest = hashlib.sha256()
    written = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("Immutable input source is not a regular file.")
        with (
            os.fdopen(descriptor, "rb", closefd=False) as source_handle,
            open(target, "xb") as target_handle,
        ):
            while chunk := source_handle.read(1024 * 1024):
                digest.update(chunk)
                written += len(chunk)
                target_handle.write(chunk)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        written != expected["size"]
        or "sha256:" + digest.hexdigest() != expected["sha256"]
        or _file_identity(before) != _file_identity(after)
        or bool(after.st_mode & 0o111) is not expected["executable"]
    ):
        raise RuntimeError("Immutable input source changed during materialization.")
    os.chmod(target, 0o555 if expected["executable"] else 0o444)


def _hash_regular_file(path: Path, expected: os.stat_result) -> tuple[str, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    observed_bytes = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (
            before.st_dev,
            before.st_ino,
        ) != (expected.st_dev, expected.st_ino):
            raise ValueError("Immutable input changed during inspection.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while chunk := handle.read(1024 * 1024):
                observed_bytes += len(chunk)
                digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _file_identity(before) != _file_identity(after) or observed_bytes != after.st_size:
        raise ValueError("Immutable input changed during inspection.")
    return digest.hexdigest(), after


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _physical_tree_bytes(path: Path) -> int:
    return sum(
        entry.stat(follow_symlinks=False).st_size
        for entry in path.rglob("*")
        if entry.is_file() and not entry.is_symlink()
    )


def _remove_tree(path: Path, *, ignore_errors: bool = False) -> None:
    try:
        if path.exists():
            for directory, directories, files in os.walk(path, topdown=False):
                for name in files:
                    os.chmod(Path(directory) / name, 0o600, follow_symlinks=False)
                for name in directories:
                    os.chmod(Path(directory) / name, 0o700, follow_symlinks=False)
            os.chmod(path, 0o700, follow_symlinks=False)
            shutil.rmtree(path)
    except OSError:
        if not ignore_errors:
            raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def docker_immutable_input_capability() -> ImmutableInputAdapterCapability:
    """Return Docker's concrete manager-issued bind-mount capability."""

    return ImmutableInputAdapterCapability(
        adapter="docker",
        capability=ImmutableInputProjectionCapability.SHARED_READ_ONLY,
        mechanism="verified_host_bind_mount_v1",
        read_only_enforced=True,
        shared_materialization=True,
        durable_recovery=True,
    )


def require_immutable_input_projection(
    capability: ImmutableInputAdapterCapability,
    *,
    allow_bounded_copy_fallback: bool = False,
) -> ImmutableInputAdapterCapability:
    """Fail closed unless an adapter can project or an explicit copy fallback is allowed."""

    if not isinstance(capability, ImmutableInputAdapterCapability):
        raise TypeError("capability must be ImmutableInputAdapterCapability.")
    if capability.capability is ImmutableInputProjectionCapability.SHARED_READ_ONLY:
        return capability
    if (
        allow_bounded_copy_fallback
        and capability.capability is ImmutableInputProjectionCapability.MUTABLE_SYNC_BINDING
        and capability.explicit_copy_fallback
    ):
        return capability
    raise ImmutableInputProjectionUnsupportedError(
        capability.reason_code or "explicit_bounded_copy_fallback_required"
    )


__all__ = [
    "DEFAULT_IMMUTABLE_INPUT_MAX_FILES",
    "DEFAULT_IMMUTABLE_INPUT_MAX_FILE_BYTES",
    "DEFAULT_IMMUTABLE_INPUT_MAX_TOTAL_BYTES",
    "IMMUTABLE_INPUT_FORMAT_VERSION",
    "DockerImmutableInputMount",
    "ImmutableInputAdapterCapability",
    "ImmutableInputAttachment",
    "ImmutableInputAttachmentStateError",
    "ImmutableInputDiagnostic",
    "ImmutableInputMutationError",
    "ImmutableInputProjection",
    "ImmutableInputProjectionCapability",
    "ImmutableInputProjectionUnsupportedError",
    "ImmutableInputStore",
    "LocalImmutableInput",
    "docker_immutable_input_capability",
    "inspect_local_immutable_input",
    "require_immutable_input_projection",
]
