"""Source-only deterministic export contracts, not participant mandates.

Values express intent and historical evidence, never live permission. Runtime
entrances must use ``prepare_contract`` before consuming untrusted values and
receive access context separately from requests. Inline payloads remain private.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator

from cayu._validation import MAX_PORTABLE_JSON_INTEGER
from cayu.collaboration._contracts import (
    CollaborationConflict,
    ContractValue,
    ExactLookup,
    ExpectedOperation,
    Generation,
    Identifier,
    InitiatorBinding,
    ObjectRef,
    OperationRef,
    OwnerRef,
)

if TYPE_CHECKING:
    from cayu.sessions.base import TranscriptRecord


SessionExportAction = Literal[
    "initialize", "source", "export", "readback", "expose", "release", "retire"
]
ExportDigest = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
SourceIndex = Annotated[StrictInt, Field(ge=0, le=MAX_PORTABLE_JSON_INTEGER)]


class SessionExportDenied(PermissionError):
    def __init__(self) -> None:
        super().__init__("Session export access is not authorized.")


class SessionExportUnavailable(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Session export is unavailable.")


class SessionExportConflict(CollaborationConflict):
    def __init__(self) -> None:
        super().__init__("Session export conflicts with the expected operation.")


class SessionExportCapacityExceeded(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Session export capacity is exhausted.")


class SessionExportRef(ContractValue):
    session_id: Identifier
    session_instance_id: Identifier
    operation: OperationRef


class SessionExportRequest(ContractValue):
    ref: SessionExportRef
    source_indices: tuple[SourceIndex, ...] = Field(max_length=16)
    audience: OwnerRef
    projector: ObjectRef
    policy: ObjectRef

    @field_validator("source_indices")
    @classmethod
    def canonical_indices(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(value)) != len(value):
            raise ValueError("Source indices must be unique.")
        return tuple(sorted(value))


class SessionExportAccessContext(ContractValue):
    """Authenticated host input passed separately; construction grants no trust."""

    principal: Identifier


class ExportLimits(ContractValue):
    max_exports: StrictInt = Field(ge=1, le=1024)
    max_retained_bytes: StrictInt = Field(ge=1, le=64 * 1024 * 1024)
    max_pending: StrictInt = Field(ge=1, le=64)


class SessionExportNamespace(ContractValue):
    """Stable owner namespace chosen once by the session store transaction."""

    owner: OwnerRef
    session_id: Identifier
    session_instance_id: Identifier
    namespace_incarnation: Identifier
    generation: Literal[1]
    limits: ExportLimits

    @field_validator("generation", mode="before")
    @classmethod
    def strict_generation(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Export namespace generation must be an integer.")
        return value


class SessionExportAuthorization(ContractValue):
    """Historical policy evidence; replay does not renew or authenticate it."""

    issuer: OwnerRef
    principal: Identifier
    policy: ObjectRef
    revision: Generation
    expires_at_ms: Annotated[StrictInt, Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)]


class SessionExportIntent(ContractValue):
    """Complete frozen export inputs, including source rows and output digests."""

    request: SessionExportRequest
    limits: ExportLimits
    source_commitment: ExportDigest
    output_commitment: ExportDigest
    authorization: SessionExportAuthorization

    @model_validator(mode="after")
    def consistent_policy(self) -> SessionExportIntent:
        if self.request.policy != self.authorization.policy:
            raise ValueError("Export authorization policy conflicts with its request.")
        return self


class SessionExportReceipt(ContractValue):
    """Exact historical evidence without payload or current disclosure authority."""

    expected: ExpectedOperation[SessionExportIntent]
    event_id: Identifier

    @model_validator(mode="after")
    def consistent_operation(self) -> SessionExportReceipt:
        if self.expected.operation != self.expected.intent.request.ref.operation:
            raise ValueError("Export receipt operation conflicts with its request.")
        return self


class SessionExportPolicy(ABC):
    @property
    @abstractmethod
    def ref(self) -> ObjectRef:
        """Stable registered implementation/configuration identity."""

    @abstractmethod
    def acquire(
        self,
        context: SessionExportAccessContext,
        *,
        session_id: str,
        session_instance_id: str,
        actions: tuple[SessionExportAction, ...],
        audience: OwnerRef | None = None,
    ) -> AbstractAsyncContextManager[SessionExportAuthorization]:
        """Acquire all requested permissions or raise SessionExportDenied.

        This synchronous factory returns one async guard that independently
        verifies every requested permission without nested guard acquisition.
        Export uses ("readback", "source", "export"); settlement uses
        ("readback", "release") or ("readback", "retire"). These combined guards
        cover concurrent historical reconciliation without nested acquisition.
        Payload reads use ("readback", "expose").
        Initialization uses ("initialize",) with the exact session IDs, before
        an operation reference exists. Empty or unknown actions must be denied.
        Host registration is trusted to supply the guard's serialization
        guarantee; these value types do not establish that guarantee themselves.
        Implementations must serialize revocation with the yielded permission
        for the entire guarded
        operation. Runtime must validate expiry against the authoritative store
        clock at commit and check expiry again at exposure. Runtime owners must
        hold the guard through owned publication/exposure and drain in-flight
        work before releasing it.
        Historical authorization values cannot replace a fresh acquisition.
        """


class SessionExportProjector(ABC):
    @property
    @abstractmethod
    def ref(self) -> ObjectRef:
        """Stable registered implementation/configuration identity."""

    @abstractmethod
    def project(self, source: tuple[TranscriptRecord, ...]) -> dict[str, Any]:
        """Deterministically project approved source fields, without model calls."""

    @abstractmethod
    def validate(
        self,
        source: tuple[TranscriptRecord, ...],
        output: dict[str, Any],
        audience: OwnerRef,
    ) -> bool:
        """Positively approve exact output for source and audience.

        JSON validity alone is insufficient. Runtime must require literal True
        from this registered validator before publication, and independently
        enforce output bounds. Implementations must not make implicit model calls.
        """


class SessionExportSettlementRequest(ContractValue):
    """Public settlement intent, never a caller-supplied acceptance receipt."""

    request: SessionExportRequest
    operation: OperationRef
    mode: Literal["release", "retire"]


class SessionExportAcceptance(ContractValue):
    """Evidence resolved by a registered receiving owner, not caller authority."""

    export_receipt: SessionExportReceipt
    receiving_owner: OwnerRef
    receipt_id: Identifier


class SessionExportAcceptanceReader(ABC):
    @property
    @abstractmethod
    def owner(self) -> OwnerRef:
        """Stable receiving owner whose evidence this reader authenticates."""

    @abstractmethod
    async def lookup(self, receipt: SessionExportReceipt) -> ExactLookup[SessionExportAcceptance]:
        """Resolve exact authenticated acceptance without disclosing payload."""


class SessionExportSettlementReceipt(ContractValue):
    """Historical settlement bound to its own complete initiating identity."""

    request: SessionExportSettlementRequest
    initiator: InitiatorBinding
    acceptance: SessionExportAcceptance | None
    event_id: Identifier

    @model_validator(mode="after")
    def consistent_acceptance(self) -> SessionExportSettlementReceipt:
        if any(
            value is not None
            for value in (
                self.initiator.participant,
                self.initiator.mandate,
                self.initiator.invocation_id,
                self.initiator.interaction_id,
            )
        ):
            raise ValueError("Session export settlement requires source-only identity.")
        if self.request.mode == "release":
            if self.acceptance is None:
                raise ValueError("Export release requires receiving acceptance.")
            if self.acceptance.export_receipt.expected.intent.request != self.request.request:
                raise ValueError("Export acceptance conflicts with its settlement request.")
            if self.acceptance.receiving_owner != self.request.request.audience:
                raise ValueError("Export acceptance conflicts with its audience.")
        elif self.acceptance is not None:
            raise ValueError("Export retirement must not include receiving acceptance.")
        return self


@dataclass(frozen=True)
class SessionExportRegistration:
    """Trusted host registration; callbacks are not serialized contract values."""

    owner: OwnerRef
    policy: SessionExportPolicy
    projectors: tuple[SessionExportProjector, ...]
    limits: ExportLimits
    readers: tuple[SessionExportAcceptanceReader, ...] = ()
