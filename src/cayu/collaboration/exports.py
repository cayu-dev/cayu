"""Source-owned deterministic and reviewed-content export contracts.

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
from cayu.collaboration._permits import PermitCommand, ReceivingSettlementReceipt
from cayu.collaboration._session_export_bounds import initiator_bytes
from cayu.collaboration.mandates import (
    MandateAccessContext,
    MandateResolution,
    MandateResolver,
    ResourceSelectorOwner,
)
from cayu.collaboration.releases import (
    ContentReleaseReader,
    ContentReleaseReceipt,
    ContentReleaseRequest,
)
from cayu.sessions.invocation import SessionInvocation

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
    mode: Literal["deterministic", "reviewed_prose"] = "deterministic"
    release: ContentReleaseRequest | None = None

    @model_validator(mode="after")
    def qualified_mode(self) -> SessionExportRequest:
        if (self.mode == "reviewed_prose") != (self.release is not None):
            raise ValueError("Export mode requires its exact release input.")
        return self

    @field_validator("source_indices")
    @classmethod
    def canonical_indices(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(value)) != len(value):
            raise ValueError("Source indices must be unique.")
        return tuple(sorted(value))


class SessionExportAccessContext(ContractValue):
    """Authenticated host input passed separately; construction grants no trust."""

    principal: Identifier
    mandate: MandateAccessContext | None = None

    @model_validator(mode="after")
    def exact_principal(self) -> SessionExportAccessContext:
        if self.mandate is not None and self.mandate.principal != self.principal:
            raise ValueError("Export and mandate principals conflict.")
        return self


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


class SessionExportRuntimeOrigin(ContractValue):
    """Historical runtime attribution, never a caller-constructible execution grant."""

    session_id: Identifier
    session_instance_id: Identifier
    run_epoch: Annotated[StrictInt, Field(ge=0, le=MAX_PORTABLE_JSON_INTEGER)]
    invocation_schema_version: Literal[1] = 1
    invocation_trust: Literal["server_verified", "host_asserted", "unattributed"]
    invocation_subject: Identifier | None = None
    invocation_tenant: Identifier | None = None
    root_invocation_id: Identifier
    root_session_id: Identifier
    invocation_source: Literal["http_run", "sdk_run", "fork", "subagent", "task", "workflow_step"]
    interaction_id: Identifier
    model_step_id: Identifier
    model_attempt_id: Identifier
    tool_round_id: Identifier
    tool_call_id: Identifier
    tool_name: Identifier
    idempotency_key: Identifier
    effective_arguments_sha256: ExportDigest
    execution_profile_fingerprint: ExportDigest

    @property
    def invocation(self) -> SessionInvocation:
        return SessionInvocation.model_validate(
            {
                "schema_version": self.invocation_schema_version,
                "origin": {
                    "trust": self.invocation_trust,
                    "subject": self.invocation_subject,
                    "tenant": self.invocation_tenant,
                },
                "root_invocation_id": self.root_invocation_id,
                "root_session_id": self.root_session_id,
                "source": self.invocation_source,
            }
        )

    @model_validator(mode="after")
    def valid_invocation(self) -> SessionExportRuntimeOrigin:
        # Reuse the runtime's semantic contract without admitting mutable or
        # foreign model objects into the frozen collaboration value envelope.
        _ = self.invocation
        return self


class SessionExportAuthorization(ContractValue):
    """Historical policy evidence; replay does not renew or authenticate it."""

    issuer: OwnerRef
    principal: Identifier
    policy: ObjectRef
    revision: Generation
    expires_at_ms: Annotated[StrictInt, Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)]
    mandate: MandateResolution | None = None
    runtime: SessionExportRuntimeOrigin | None = None

    @model_validator(mode="after")
    def exact_mandate_principal(self) -> SessionExportAuthorization:
        if self.mandate is not None and (
            self.mandate.principal.principal != self.principal
            or self.mandate.chain.entries[-1].principal != self.principal
            or self.mandate.principal.issuer != self.mandate.chain.entries[-1].issuer
        ):
            raise ValueError("Mandate identity conflicts with the export principal.")
        return self

    def initiating_identity(self) -> InitiatorBinding:
        leaf = None if self.mandate is None else self.mandate.chain.entries[-1]
        participant = None if leaf is None else leaf.participant
        return InitiatorBinding(
            issuer=self.issuer if leaf is None else leaf.issuer,
            principal=self.principal,
            participant=None
            if participant is None
            else ObjectRef(
                owner=participant.owner,
                kind="participant",
                object_id=participant.participant_id,
                incarnation=participant.incarnation,
            ),
            mandate=None if leaf is None else leaf.reference,
            invocation_id=None
            if self.runtime is None
            else self.runtime.invocation.root_invocation_id,
            interaction_id=None if self.runtime is None else self.runtime.interaction_id,
        )


class SessionExportIntent(ContractValue):
    """Complete frozen export inputs, including source rows and output digests."""

    request: SessionExportRequest
    limits: ExportLimits
    source_commitment: ExportDigest
    output_commitment: ExportDigest
    authorization: SessionExportAuthorization
    release_receipt: ContentReleaseReceipt | None = None

    @model_validator(mode="after")
    def consistent_policy(self) -> SessionExportIntent:
        if self.request.policy != self.authorization.policy:
            raise ValueError("Export authorization policy conflicts with its request.")
        if (self.request.mode == "reviewed_prose") != (self.release_receipt is not None):
            raise ValueError("Export mode conflicts with its release evidence.")
        if self.release_receipt is not None:
            expected = self.release_receipt.expected
            if (
                expected.request != self.request.release
                or expected.session_id != self.request.ref.session_id
                or expected.session_instance_id != self.request.ref.session_instance_id
                or expected.source_indices != self.request.source_indices
                or expected.audience != self.request.audience
                or expected.validator != self.request.projector
                or expected.policy != self.request.policy
                or expected.request.source_commitment != self.source_commitment
            ):
                raise ValueError("Exact content release conflicts with the export.")
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


class SessionExportReconciliation(ContractValue):
    """A published result or positively excluded participant preparation.

    Exclusion is not a published export and cannot authorize payload exposure.
    The original operation key remains occupied after either outcome.
    """

    request: SessionExportRequest
    state: Literal["published", "excluded"]
    receipt: SessionExportReceipt | None

    @model_validator(mode="after")
    def exact_result(self) -> SessionExportReconciliation:
        if (self.state == "published") != (self.receipt is not None) or (
            self.receipt is not None and self.receipt.expected.intent.request != self.request
        ):
            raise ValueError("Export reconciliation evidence conflicts.")
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

    def acquire_runtime(
        self,
        context: SessionExportAccessContext,
        *,
        origin: SessionExportRuntimeOrigin,
        session_id: str,
        session_instance_id: str,
        actions: tuple[SessionExportAction, ...],
        audience: OwnerRef | None = None,
    ) -> AbstractAsyncContextManager[SessionExportAuthorization]:
        """Explicit runtime adapter, denied unless the application implements it.

        Cayu derives origin from a live privately bound tool context and checks
        its exact requester session. Implementations authenticate the requested
        principal against that origin and independently authorize the selected
        source and actions. They return policy evidence without runtime fields;
        Cayu attaches the authenticated origin itself. Source-session ownership
        is not implied by this caller attribution.
        """
        raise SessionExportDenied()


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

    async def settlement(
        self,
        receipt: SessionExportReceipt,
        expected: PermitCommand,
    ) -> ExactLookup[ReceivingSettlementReceipt]:
        """Resolve exact receiving settlement; acceptance alone is insufficient."""
        from cayu.collaboration._contracts import ExactUnavailable

        return ExactUnavailable()


class SessionExportSettlementReceipt(ContractValue):
    """Historical settlement bound to its own complete initiating identity."""

    request: SessionExportSettlementRequest
    initiator: InitiatorBinding
    mandate_commitment: ExportDigest | None = None
    acceptance: SessionExportAcceptance | None
    event_id: Identifier

    @model_validator(mode="after")
    def consistent_acceptance(self) -> SessionExportSettlementReceipt:
        initiator_bytes(self.initiator)
        if (self.initiator.mandate is not None) != (self.mandate_commitment is not None):
            raise ValueError("Settlement must bind its complete initiating mandate.")
        if self.initiator.participant is not None and self.initiator.mandate is None:
            raise ValueError("Participant settlement requires mandate attribution.")
        if self.initiator.invocation_id is not None or self.initiator.interaction_id is not None:
            raise ValueError("Host export settlement cannot assert runtime invocation identity.")
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
    release_readers: tuple[ContentReleaseReader, ...] = ()
    mandates: MandateResolver | None = None
    resource_owners: tuple[ResourceSelectorOwner, ...] = ()
