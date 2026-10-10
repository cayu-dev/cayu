"""Opt-in execution snapshots. Live adapter handles never enter durable records."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import require_durable_clean_nonblank


class ExecutionSnapshotError(RuntimeError):
    """A snapshot cannot safely be published or activated."""


class ExecutionSnapshotConflict(ExecutionSnapshotError):
    """The expected binding, owner, execution position or operation changed."""


class ExecutionSnapshotOutcomeUnknown(ExecutionSnapshotError):
    """Submission may have succeeded. Inspect/reconcile; never blindly replay."""


class ExecutionSnapshotFidelity(StrEnum):
    UNSUPPORTED = "unsupported"
    FILESYSTEM = "filesystem"
    SELECTED_PROCESSES = "selected_processes"
    COMPLETE_SUBSTRATE = "complete_substrate"


class _SnapshotModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    @field_validator("*", mode="after")
    @classmethod
    def clean_text(cls, value):
        if type(value) is str:
            require_durable_clean_nonblank(value, "snapshot field")
            if len(value.encode()) > 512:
                raise ValueError("Snapshot text exceeds its bound.")
        return value


class ExecutionSnapshotPolicy(_SnapshotModel):
    timeout_seconds: StrictInt = Field(default=120, ge=1, le=3600)
    max_artifact_bytes: StrictInt = Field(default=128 * 1024**2, ge=1, le=1024**3)
    max_total_bytes: StrictInt = Field(default=256 * 1024**2, ge=1, le=4 * 1024**3)
    max_files: StrictInt = Field(default=10000, ge=1, le=100000)
    retention_seconds: StrictInt = Field(default=86400, ge=1, le=365 * 86400)
    max_records: StrictInt = Field(default=128, ge=1, le=1024)


class ExecutionSnapshotCapability(_SnapshotModel):
    fidelity: ExecutionSnapshotFidelity = ExecutionSnapshotFidelity.UNSUPPORTED
    adapter: str = "unsupported"
    adapter_version: str = "1"
    snapshot_format: str = "none"
    compatibility_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    requires_managed_launch: bool = False

    @model_validator(mode="after")
    def capable_identity(self):
        if self.fidelity != ExecutionSnapshotFidelity.UNSUPPORTED and (
            self.compatibility_sha256 is None
            or self.snapshot_format == "none"
            or self.adapter == "unsupported"
        ):
            raise ValueError(
                "Snapshot capability requires exact compatibility and format identity."
            )
        return self


class ExecutionSnapshotPosition(_SnapshotModel):
    """Exact durable controller position, including effects and delivered results.

    Restoration never replaces this checkpoint with older controller state.
    An advanced/different checkpoint requires explicit reconciliation instead.
    """

    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transcript_cursor: StrictInt = Field(ge=0)


class ExecutionSnapshotArtifact(_SnapshotModel):
    role: Literal["process", "workspace"]
    artifact_id: str = Field(pattern=r"^art_[0-9a-f]{32}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: StrictInt = Field(ge=1)


class ExecutionSnapshotRecord(_SnapshotModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^esnap_[0-9a-f]{32}$")
    session_id: str
    session_instance_id: str
    environment_name: str
    binding_generation: str
    allocation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_run_epoch: StrictInt = Field(ge=0)
    capability: ExecutionSnapshotCapability
    position: ExecutionSnapshotPosition
    created_at: AwareDatetime
    expires_at: AwareDatetime
    retention: Literal["retained", "deleting", "deleted"] = "retained"
    artifacts: tuple[ExecutionSnapshotArtifact, ...] = Field(min_length=2, max_length=2)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_bytes: StrictInt = Field(ge=1)
    pin_owner: str

    @model_validator(mode="after")
    def complete_components(self):
        if (
            sorted(part.role for part in self.artifacts) != ["process", "workspace"]
            or len({part.artifact_id for part in self.artifacts}) != 2
            or sum(part.size_bytes for part in self.artifacts) != self.total_bytes
            or self.expires_at <= self.created_at
        ):
            raise ValueError("Snapshot manifest component accounting or lifetime is invalid.")
        return self


class ExecutionSnapshotOperation(_SnapshotModel):
    id: str = Field(pattern=r"^esop_[0-9a-f]{32}$")
    kind: Literal["capture", "restore", "delete"]
    state: Literal["intent", "submitted", "unknown", "verified", "succeeded"]
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_epoch: StrictInt = Field(ge=0)
    binding_generation: str
    snapshot_id: str = Field(pattern=r"^esnap_[0-9a-f]{32}$")
    target_generation: str | None = None
    allocation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    position: ExecutionSnapshotPosition | None = None
    created_at: AwareDatetime | None = None


class ExecutionSnapshotSummary(_SnapshotModel):
    id: str
    capability: ExecutionSnapshotCapability
    created_at: AwareDatetime
    expires_at: AwareDatetime
    retention: Literal["retained", "deleting", "deleted"]
    expired: bool
    total_bytes: StrictInt = Field(ge=1)


class ExecutionSnapshotInspection(_SnapshotModel):
    environment_name: str
    binding_generation: str | None
    operations: tuple[ExecutionSnapshotOperation, ...] = Field(max_length=100)
    snapshots: tuple[ExecutionSnapshotSummary, ...] = Field(max_length=100)
    truncated: bool


@dataclass(frozen=True)
class CapturedExecutionSnapshot:
    """Private bounded bytes. Do not log or include in tool/event payloads."""

    process: bytes
    workspace: bytes


SnapshotFence = Callable[[], Awaitable[None]]


class ExecutionSnapshotAdapter(ABC):
    """A capability bound to one allocation and explicitly managed workload.

    Restore must keep the workload inactive until activate() is called. A lost
    acknowledgement is never permission to repeat a non-idempotent mutation.
    All methods must check the supplied fence before each substrate mutation.
    """

    @property
    @abstractmethod
    def capability(self) -> ExecutionSnapshotCapability: ...

    @property
    @abstractmethod
    def allocation_sha256(self) -> str: ...

    @abstractmethod
    async def capture(
        self, operation_id: str, policy: ExecutionSnapshotPolicy, fence: SnapshotFence
    ) -> CapturedExecutionSnapshot: ...

    @abstractmethod
    async def restore(
        self,
        operation_id: str,
        snapshot: CapturedExecutionSnapshot,
        policy: ExecutionSnapshotPolicy,
        fence: SnapshotFence,
    ) -> None: ...

    @abstractmethod
    async def activate(self, operation_id: str, fence: SnapshotFence) -> None: ...

    @abstractmethod
    async def resume_source(self, operation_id: str, fence: SnapshotFence) -> None: ...

    async def preflight_capture(self, policy: ExecutionSnapshotPolicy) -> None:
        """Reject a capture that is certain to fail, without mutating the substrate.

        Runs before the operation is recorded as submitted, so a failure here
        leaves nothing to reconcile.
        """
        return None

    async def recover_capture(self, operation_id, policy, fence) -> CapturedExecutionSnapshot:
        """Read the existing held capture without issuing another checkpoint."""
        raise ExecutionSnapshotOutcomeUnknown("Adapter cannot prove the submitted capture.")

    async def verify_restore(self, operation_id, policy, fence) -> None:
        """Prove that the existing restore is complete and still inactive."""
        raise ExecutionSnapshotOutcomeUnknown("Adapter cannot prove the submitted restoration.")
