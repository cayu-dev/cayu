"""Qualified local immutable-resource owner for issue #1800.

The owner is deliberately separate from generic artifact writes and from
Docker input projection. It records the complete operation and responsibility
tuple, keeps retention under the artifact store's deletion fence, and exposes
only owner-issued receipts.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import stat
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from contextlib import (
    AbstractAsyncContextManager,
    ExitStack,
    asynccontextmanager,
    contextmanager,
    suppress,
)
from contextvars import copy_context
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Literal, Self, TypeVar, cast
from uuid import uuid4

from pydantic import ConfigDict, Field, StrictInt, model_validator

from cayu._filesystem_lock import cooperative_path_lock
from cayu._task_wait import await_shielded_task_outcome, restore_task_cancellation_requests
from cayu._validation import DURABLE_DOCUMENT_LIMITS, canonical_durable_json_bytes
from cayu.artifacts._input_manifest import FolderInputManifest
from cayu.artifacts.base import ArtifactMetadata, ArtifactReadResult, ArtifactStore
from cayu.artifacts.local import LocalArtifactStore
from cayu.collaboration._contracts import (
    ContractValue,
    ExactConflict,
    ExactLookup,
    ExactMatch,
    ExactNotFound,
    ExactUnavailable,
    ExpectedOperation,
    Generation,
    Identifier,
    ObjectRef,
    OwnerRef,
)
from cayu.collaboration._mandate_validation import (
    MandateInput,
    MandateUse,
    validate_mandate_resolution,
)
from cayu.collaboration._permits import (
    PermitCommand,
    PermitSettlementReader,
    ReceivingSettlementReceipt,
)
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.base import CollaborationStore
from cayu.collaboration.mandates import (
    MandateAccessContext,
    MandateDenied,
    MandateResolver,
    ResourceSelector,
    ResourceSelectorOwner,
)
from cayu.collaboration.participants import CollaborationInitialization
from cayu.vaults.redaction import SecretRedactor

RESOURCE_KIND_ARTIFACT = "artifact"
RESOURCE_KIND_FOLDER = "immutable_folder"
RESOURCE_OPERATION_SCHEMA = 1
RESOURCE_JOURNAL_SCHEMA = 1
RESOURCE_MAX_OPERATIONS = 256
RESOURCE_EVENT_SLOTS_PER_OPERATION = 7
RESOURCE_MAX_EVENTS = RESOURCE_MAX_OPERATIONS * RESOURCE_EVENT_SLOTS_PER_OPERATION
RESOURCE_MAX_MATERIALS = 32
RESOURCE_MAX_RESERVED_BYTES = 4 * 1024**3
RESOURCE_FOREGROUND_TIMEOUT_S = 5.0
RESOURCE_JOURNAL_MAX_BYTES = DURABLE_DOCUMENT_LIMITS.max_bytes
RESOURCE_JOURNAL_MAX_NODES = DURABLE_DOCUMENT_LIMITS.max_nodes
_ResultT = TypeVar("_ResultT")


def _artifact_incarnation(metadata: ArtifactMetadata) -> str:
    # Local publication creates immutable metadata (including created_at).
    # Re-publication under a reused artifact id is a different incarnation.
    return hashlib.sha256(
        canonical_durable_json_bytes(metadata.model_dump(mode="json"), "metadata")
    ).hexdigest()


def _event_usage(journal):
    events = journal["events"]
    if not isinstance(events, list):
        raise ResourceOwnerUnavailable("Invalid resource events.")
    reserved = 0
    for family in ("operations", "transfers"):
        for record in journal[family].values():
            remaining = record.get("event_slots_remaining")
            if (
                type(remaining) is not int
                or not 0 <= remaining <= RESOURCE_EVENT_SLOTS_PER_OPERATION
            ):
                raise ResourceOwnerUnavailable("Invalid resource event reservation.")
            reserved += remaining
    return len(events) + reserved


def _reserve_events(journal):
    if _event_usage(journal) + RESOURCE_EVENT_SLOTS_PER_OPERATION > RESOURCE_MAX_EVENTS:
        raise ResourceOwnerError("Resource event reservation capacity exhausted.")
    return RESOURCE_EVENT_SLOTS_PER_OPERATION


def _append_resource_event(journal, family, digest, stage, **details):
    key = "operation" if family == "operations" else "transfer"
    # Stage evidence is immutable and emitted once. Repeated attempts update the
    # retained snapshot, not an unbounded series of identical stage events.
    if any(event.get(key) == digest and event.get("stage") == stage for event in journal["events"]):
        return
    record = journal[family][digest]
    remaining = record["event_slots_remaining"]
    if type(remaining) is not int or remaining <= 0:
        raise ResourceOwnerUnavailable("Resource event reservation is unavailable.")
    journal["events"].append({key: digest, "stage": stage, **details})
    record["event_slots_remaining"] = 0 if stage == "responsibility_settled" else remaining - 1


def _registered_artifact_store(store: ArtifactStore) -> dict[str, object]:
    if type(store) is not LocalArtifactStore:
        raise ResourceOwnerUnsupported(
            "Mandate receiver requires the qualified local artifact store."
        )
    return {
        "id": store.id,
        "root": str(store.root),
        "root_identity": [str(part) for part in store._root_identity],
    }


def _reserved_bytes(journal: Mapping[str, object]) -> int:
    """Count each durable responsibility once, including pending transfers.

    Pins owned by acquisition and transfer are separate obligations, even for
    identical bytes. Their reservations remain charged until release commits.
    """
    total = 0
    for family, schema in (
        ("operations", ResourceAcquisitionCommand),
        ("transfers", ResourceTransferCommand),
    ):
        records = journal[family]
        if not isinstance(records, dict):
            raise ResourceOwnerUnavailable("Invalid resource accounting records.")
        for key, record in records.items():
            try:
                if not isinstance(record, dict):
                    raise ValueError("Invalid responsibility record")
                record = cast("dict[str, Any]", record)
                command = schema.model_validate_json(record["command"])
                if resource_operation_digest(command) != key:
                    raise ValueError("Operation identity mismatch")
                stage = record["stage"]
                if stage not in {
                    "pending",
                    "uncertain",
                    "owned",
                    "accepted",
                    "releasing",
                    "cleaning",
                    "released",
                }:
                    raise ValueError("Unknown responsibility state")
                if stage == "released" and record.get("responsibility_settled") is True:
                    continue
                total += (
                    command.intent.max_total_bytes
                    if isinstance(command, ResourceAcquisitionCommand)
                    else command.intent.receipt.total_bytes
                )
            except (KeyError, TypeError, ValueError):
                raise ResourceOwnerUnavailable("Invalid resource accounting record.") from None
    return total


def _raise_diagnostic_failure(primary: BaseException, diagnostic: BaseException) -> None:
    """Keep original ordered errors, with new control signals authoritative."""
    if diagnostic.__context__ is primary:
        diagnostic.__context__ = None
    if isinstance(
        diagnostic, (asyncio.CancelledError, KeyboardInterrupt, SystemExit, GeneratorExit)
    ):
        raise diagnostic from primary
    if isinstance(primary, (asyncio.CancelledError, KeyboardInterrupt, SystemExit, GeneratorExit)):
        previous = primary.__cause__
        evidence = [diagnostic] if previous is None else [previous, diagnostic]
        raise primary from BaseExceptionGroup("Resource cleanup failures", evidence)
    raise BaseExceptionGroup(
        "Resource operation and diagnostic failures", [primary, diagnostic]
    ) from None


def _consume_resource_task(task: asyncio.Task[object]) -> None:
    """Consume late task outcomes so bounded foreground waits stay observable."""
    try:
        task.exception()
    except (asyncio.CancelledError, BaseException):
        return


class ResourceOwnerError(RuntimeError):
    """The owner cannot establish or reconcile exact resource responsibility."""


class ResourceOwnerConflict(ResourceOwnerError):
    """A fixed operation key was reused with different expected input."""


class ResourceOwnerUnavailable(ResourceOwnerError):
    """Durable readback or source material is unavailable."""


class ResourceOwnerUnsupported(ResourceOwnerError):
    """The requested adapter or selector mode is not qualified."""


class _PreparationStopped(ResourceOwnerUnavailable):
    """The observer stopped; this is not a new caller cancellation signal."""


@dataclass
class _PreparationWork:
    stopped: bool = False
    observer: asyncio.Task[Any] | None = field(default_factory=asyncio.current_task, repr=False)
    historical_cancellations: int = field(init=False)

    def __post_init__(self) -> None:
        self.historical_cancellations = 0 if self.observer is None else self.observer.cancelling()

    def stop_requested(self) -> bool:
        # A ready worker can resume before the shielded observer processes its
        # wakeup. Honor a newly requested cancellation in that ordering too,
        # without treating previously handled requests as fresh cancellation.
        if self.observer is not None and self.observer.cancelling() > self.historical_cancellations:
            self.stopped = True
        return self.stopped

    def check(self) -> None:
        if self.stop_requested():
            raise _PreparationStopped("Resource preparation observer stopped.")


@dataclass
class _ResourceOutcome(Generic[_ResultT]):
    result: _ResultT | None = None
    error: BaseException | None = None


async def _resource_io(call, *args, **kwargs):
    """Join dispatched filesystem work before releasing its enclosing owner."""

    def run():
        try:
            return _ResourceOutcome(result=call(*args, **kwargs))
        except BaseException as error:
            return _ResourceOutcome(error=error)

    pending = asyncio.get_running_loop().run_in_executor(None, copy_context().run, run)
    return await _resource_outcome(pending)


async def _resource_outcome(task):
    outcome = await await_shielded_task_outcome(task)
    error = outcome.error
    if error is None and outcome.result is not None:
        error = outcome.result.error
    if outcome.cancellation is not None:
        restore_task_cancellation_requests(
            outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
        )
        raise outcome.cancellation from error
    if error is not None:
        raise error
    assert outcome.result is not None
    return outcome.result.result


class ResourcePreparationLease(ContractValue):
    """Runtime-created, receiver-bound authority; not a caller credential."""

    operation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    receiver: OwnerRef
    resolver_generation: StrictInt = Field(ge=0)
    receiver_generation: StrictInt = Field(ge=0)
    revocation_generation: StrictInt = Field(ge=0)
    expires_at_ms: StrictInt = Field(ge=1)
    nonce: str = Field(min_length=16, max_length=128)
    authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    authority_json: str = Field(min_length=2, max_length=65536)
    chain_generations: tuple[tuple[ObjectRef, StrictInt], ...] = Field(max_length=16)


class ResourcePreparationReceipt(ContractValue):
    """Owner-issued bounded preparation evidence, not caller-mintable authority."""

    operation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    owner: OwnerRef
    command_bytes_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    permit: PermitCommand
    lease: ResourcePreparationLease


class ResourceAcquisitionIntent(ContractValue):
    """All decision-bearing resource acquisition inputs."""

    unordered_fields = frozenset({"allowed_operations"})

    selector: ResourceSelector
    allowed_operations: tuple[Literal["read", "transfer", "release"], ...] = Field(
        min_length=1, max_length=3
    )
    isolation_mode: Literal["local_durable"] = "local_durable"
    max_materials: StrictInt = Field(ge=1, le=RESOURCE_MAX_MATERIALS)
    max_manifest_bytes: StrictInt = Field(ge=1024, le=32 * 1024 * 1024)
    max_total_bytes: StrictInt = Field(ge=1, le=4 * 1024**3)
    cleanup_owner: OwnerRef
    policy: ObjectRef

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if "release" not in self.allowed_operations:
            raise ValueError("Resource acquisition must permit mandatory release.")
        if self.selector.resource.revision is None:
            raise ValueError("Resource selector must pin an exact revision.")
        if self.selector.resource.kind not in (RESOURCE_KIND_ARTIFACT, RESOURCE_KIND_FOLDER):
            raise ValueError("Unsupported immutable resource kind.")
        if self.policy.revision is None:
            raise ValueError("Resource policy must be pinned.")
        if self.cleanup_owner.application_scope != self.selector.resource.owner.application_scope:
            raise ValueError("Cleanup owner scope conflicts with resource owner.")
        return self


class ResourceAcquisitionCommand(ExpectedOperation[ResourceAcquisitionIntent]):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    kind: Literal["resource_acquisition"] = "resource_acquisition"
    schema_version: Literal[1] = RESOURCE_OPERATION_SCHEMA
    mode: Literal["acquire"] = "acquire"
    receipt_stage: Literal["acquired"] = "acquired"

    @model_validator(mode="after")
    def exact_owner(self) -> Self:
        if self.source != self.intent.selector.resource.owner:
            raise ValueError("Resource source conflicts with selector owner.")
        if self.destination != self.intent.cleanup_owner:
            raise ValueError("Resource destination must equal cleanup owner.")
        if self.initiator.issuer != self.source:
            raise ValueError("Resource initiator issuer conflicts with source owner.")
        return self


class ResourcePreparationAuthorization(ContractValue):
    """Evidence yielded under a registered authority's revocation guard."""

    command: ResourceAcquisitionCommand | ResourceTransferCommand
    permit: PermitCommand
    lease: ResourcePreparationLease


class ResourcePreparationReader(ABC):
    """Trusted application registration, never selected by operation input.

    Implementations authenticate caller identity, the exact retained permit,
    command, policy, bounds and source revision. Merely validating or echoing
    these values is not authorization. The guard serializes revocation until
    exit, performs no acquisition effects, and must not block the event loop.
    Durable recovery requires re-registration of the same authority.
    """

    @property
    @abstractmethod
    def owner(self) -> OwnerRef: ...

    @abstractmethod
    def acquire(
        self, command: ResourceAcquisitionCommand | ResourceTransferCommand, permit: PermitCommand
    ) -> AbstractAsyncContextManager[ResourcePreparationAuthorization]: ...

    async def revalidate(
        self,
        command: ResourceAcquisitionCommand | ResourceTransferCommand,
        lease: ResourcePreparationLease,
    ) -> None:
        """Observe current authority without authorizing a later effect."""
        async with self.revalidation_guard(command, lease):
            pass

    @abstractmethod
    def revalidation_guard(
        self,
        command: ResourceAcquisitionCommand | ResourceTransferCommand,
        lease: ResourcePreparationLease,
    ) -> AbstractAsyncContextManager[None]:
        """Serialize revocation through dispatch/publication, not effect settlement."""
        ...

    @abstractmethod
    async def authorize_release(self, command: ResourceAcquisitionCommand) -> None:
        """Authenticate current public cleanup authority independently of acquisition expiry."""
        ...

    @abstractmethod
    async def register_responsibility(self, command, permit) -> None: ...

    @abstractmethod
    def transfer_permit(self, command: ResourceTransferCommand) -> PermitCommand:
        """Resolve exact destination responsibility from trusted registration."""
        ...

    @abstractmethod
    async def settle_responsibility(
        self, command, permit, reader: PermitSettlementReader
    ) -> None: ...


class MandateResourcePreparationReader(ResourcePreparationReader):
    """Concrete registered receiver using the shared mandate resolver."""

    def __init__(
        self,
        *,
        owner: ResourceSelectorOwner,
        resolver: MandateResolver,
        context: MandateAccessContext,
        redactor: SecretRedactor,
        registration: ObjectRef,
        policy: ObjectRef,
        artifact_store: ArtifactStore,
        collaboration_store: CollaborationStore,
        initialized: CollaborationInitialization,
        responsibilities: tuple[tuple[ResourceAcquisitionCommand, PermitCommand], ...],
        transfers: tuple[tuple[ResourceTransferCommand, PermitCommand], ...] = (),
    ) -> None:
        if not isinstance(owner, ResourceSelectorOwner) or not isinstance(
            resolver, MandateResolver
        ):
            raise TypeError("Registered resource owner and mandate resolver are required.")
        self._owner, self._resolver = owner, resolver
        self._context = prepare_contract(MandateAccessContext, context, redactor=redactor)
        self._redactor = redactor
        self._registration = prepare_contract(ObjectRef, registration, redactor=redactor)
        self._policy = prepare_contract(ObjectRef, policy, redactor=redactor)
        if (
            registration.owner != owner.owner
            or registration.revision is None
            or policy.revision is None
        ):
            raise ResourceOwnerUnsupported("Pinned receiver registration and policy are required.")
        if not isinstance(collaboration_store, CollaborationStore):
            raise ResourceOwnerUnsupported(
                "Registered collaboration and artifact stores are required."
            )
        self._store_binding = _registered_artifact_store(artifact_store)
        self._collaboration_store = collaboration_store
        self._initialized = prepare_contract(
            CollaborationInitialization, initialized, redactor=redactor
        )
        self._responsibilities = tuple(
            (
                prepare_contract(ResourceAcquisitionCommand, command, redactor=redactor),
                prepare_contract(PermitCommand, permit, redactor=redactor),
            )
            for command, permit in responsibilities
        ) + tuple(
            (
                prepare_contract(ResourceTransferCommand, command, redactor=redactor),
                prepare_contract(PermitCommand, permit, redactor=redactor),
            )
            for command, permit in transfers
        )
        if len(self._responsibilities) > RESOURCE_MAX_OPERATIONS:
            raise ResourceOwnerUnsupported("Preparation registration is too large.")
        if len(
            {resource_operation_digest(command) for command, _ in self._responsibilities}
        ) != len(self._responsibilities):
            raise ResourceOwnerConflict("Preparation registration is ambiguous.")

    def _expected_permit(self, command):
        for expected, permit in self._responsibilities:
            if expected.operation == command.operation:
                if expected != command:
                    raise ResourceOwnerConflict("Registered acquisition conflicts.")
                return permit
        raise ResourceOwnerUnsupported("Acquisition has no registered responsibility.")

    def _authority_json(self):
        return canonical_durable_json_bytes(
            {
                "resolver": self._resolver.ref.model_dump(mode="json"),
                "receiver": self._registration.model_dump(mode="json"),
                "policy": self._policy.model_dump(mode="json"),
                "artifact_store": self._store_binding,
                "context": self._context.model_dump(mode="json"),
                "collaboration": self._initialized.model_dump(mode="json"),
                "capabilities": ["artifact", "immutable_folder", "local_durable", "unpaid"],
                "responsibilities": [
                    [command.model_dump(mode="json"), permit.model_dump(mode="json")]
                    for command, permit in self._responsibilities
                ],
            },
            "resource_authority",
        ).decode()

    def _check_command(self, command):
        self._expected_permit(command)
        if isinstance(command, ResourceTransferCommand):
            # The source initiator is part of the exact host-registered handoff;
            # the receiving mandate separately authenticates destination use.
            if (
                command.destination != self.owner
                or command.intent.receipt.command.intent.policy != self._policy
            ):
                raise ResourceOwnerUnsupported("Transfer conflicts with receiver registration.")
            return
        context = self._context
        initiator = command.initiator
        if (
            command.intent.policy != self._policy
            or initiator.issuer != context.issuer
            or initiator.principal != context.principal
            or initiator.participant != context.participant
            or initiator.mandate != context.mandate
            or initiator.invocation_id is not None
            or initiator.interaction_id is not None
        ):
            raise ResourceOwnerUnsupported(
                "Acquisition initiator or policy conflicts with registration."
            )

    def transfer_permit(self, command):
        self._check_command(command)
        return self._expected_permit(command)

    async def register_responsibility(self, command, permit):
        self._check_command(command)
        if self._expected_permit(command) != permit:
            raise ResourceOwnerConflict("Registered responsibility conflicts.")
        await self._collaboration_store._register_permit(
            self._initialized, permit, redactor=self._redactor
        )

    async def settle_responsibility(self, command, permit, reader):
        if self._expected_permit(command) != permit:
            raise ResourceOwnerConflict("Registered responsibility conflicts.")
        found = await reader.lookup(permit)
        if not isinstance(found, ExactMatch):
            raise ResourceOwnerUnavailable("Cleanup is not settled.")
        settle = (
            self._collaboration_store._exclude_permit
            if found.receipt.proves_exclusion
            else self._collaboration_store._settle_permit
        )
        await settle(self._initialized, permit, reader=reader, redactor=self._redactor)

    async def authorize_release(self, command):
        async with self._resolver.acquire(self._context) as resolution:
            validate_mandate_resolution(
                resolution,
                context=self._context,
                resolver=self._resolver.ref,
                use=MandateUse(
                    audience=self.owner,
                    scope=self.owner.application_scope,
                    actions=("release",),
                    resources=(command.intent.selector,),
                    inputs=(
                        MandateInput(source=command.intent.selector.resource, channel="artifact"),
                    ),
                ),
                now_ms=int(time.time() * 1000),
                resource_owners={self.owner: self._owner},
                redactor=self._redactor,
            )

    @property
    def owner(self) -> OwnerRef:
        return self._owner.owner

    @asynccontextmanager
    async def acquire(self, command, permit):
        self._check_command(command)
        if self._expected_permit(command) != permit:
            raise ResourceOwnerConflict("Registered responsibility conflicts.")
        if (
            permit.source != self.owner
            or permit.destination != self.owner
            or permit.intent.request.target != _preparation_target(command)
        ):
            raise ResourceOwnerUnsupported("Preparation authority identity conflicts.")
        resolution_use = MandateUse(
            audience=self.owner,
            scope=self.owner.application_scope,
            actions=("prepare",),
            resources=(_preparation_selector(command),),
            inputs=(
                MandateInput(source=_preparation_selector(command).resource, channel="artifact"),
            ),
        )
        async with self._resolver.acquire(self._context) as resolution:
            checked = validate_mandate_resolution(
                resolution,
                context=self._context,
                resolver=self._resolver.ref,
                use=resolution_use,
                now_ms=int(time.time() * 1000),
                resource_owners={self.owner: self._owner},
                redactor=self._redactor,
            )
            leaf = checked.chain.entries[-1]
            yield ResourcePreparationAuthorization(
                command=command,
                permit=permit,
                lease=ResourcePreparationLease(
                    operation_digest=resource_operation_digest(command),
                    receiver=self.owner,
                    resolver_generation=checked.principal.resolver.revision or 0,
                    receiver_generation=cast("int", self._registration.revision),
                    revocation_generation=leaf.revocation_generation,
                    expires_at_ms=min(
                        checked.principal.expires_at_ms,
                        *(entry.expires_at_ms for entry in checked.chain.entries),
                    ),
                    nonce=uuid4().hex,
                    authority_sha256=hashlib.sha256(self._authority_json().encode()).hexdigest(),
                    authority_json=self._authority_json(),
                    chain_generations=tuple(
                        (entry.reference, entry.revocation_generation)
                        for entry in checked.chain.entries
                    ),
                ),
            )

    @asynccontextmanager
    async def revalidation_guard(self, command, lease):
        self._check_command(command)
        if (
            lease.authority_json != self._authority_json()
            or lease.authority_sha256 != hashlib.sha256(lease.authority_json.encode()).hexdigest()
        ):
            raise ResourceOwnerUnavailable("Preparation registration changed.")
        if lease.receiver != self.owner or lease.operation_digest != resource_operation_digest(
            command
        ):
            raise ResourceOwnerConflict("Preparation lease identity conflicts.")
        if lease.expires_at_ms <= int(time.time() * 1000):
            raise ResourceOwnerUnavailable("Preparation lease expired.")
        use = MandateUse(
            audience=self.owner,
            scope=self.owner.application_scope,
            actions=("prepare",),
            resources=(_preparation_selector(command),),
            inputs=(
                MandateInput(source=_preparation_selector(command).resource, channel="artifact"),
            ),
        )
        async with self._resolver.acquire(self._context) as resolution:
            checked = validate_mandate_resolution(
                resolution,
                context=self._context,
                resolver=self._resolver.ref,
                use=use,
                now_ms=int(time.time() * 1000),
                resource_owners={self.owner: self._owner},
                redactor=self._redactor,
            )
            if checked.principal.resolver.revision != lease.resolver_generation:
                raise ResourceOwnerUnavailable("Preparation resolver generation changed.")
            if self._registration.revision != lease.receiver_generation:
                raise ResourceOwnerUnavailable("Preparation receiver generation changed.")
            if (
                tuple(
                    (entry.reference, entry.revocation_generation)
                    for entry in checked.chain.entries
                )
                != lease.chain_generations
            ):
                raise ResourceOwnerUnavailable("Preparation lease was revoked.")
            if lease.expires_at_ms <= int(time.time() * 1000):
                raise ResourceOwnerUnavailable("Preparation lease expired.")
            yield


class ResourceAcquisitionReceipt(ContractValue):
    """Immutable owner-issued evidence of retained exact material."""

    command: ResourceAcquisitionCommand
    receipt_id: Identifier
    stage: Literal["pending", "owned", "releasing", "uncertain", "transferred", "released"]
    operation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    material_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=RESOURCE_MAX_MATERIALS)
    content_commitment: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    manifest_commitment: str = Field(pattern=r"^[0-9a-f]{64}$")
    material_count: StrictInt = Field(ge=1, le=RESOURCE_MAX_MATERIALS)
    total_bytes: StrictInt = Field(ge=0, le=4 * 1024**3)
    allowed_operations: tuple[Literal["read", "transfer", "release"], ...] = Field(
        min_length=1, max_length=3
    )

    @model_validator(mode="after")
    def exact_shape(self) -> Self:
        digest = hashlib.sha256(
            canonical_durable_json_bytes(
                self.command.operation.model_dump(mode="json"), "resource_operation_identity"
            )
        ).hexdigest()
        if self.operation_digest != digest:
            raise ValueError("Resource receipt operation identity conflicts.")
        if self.material_count != len(self.material_ids):
            raise ValueError("Resource receipt material count conflicts.")
        if tuple(sorted(set(self.allowed_operations))) != tuple(
            sorted(set(self.command.intent.allowed_operations))
        ):
            raise ValueError("Resource receipt operation allowance conflicts.")
        return self


class ResourceTransferIntent(ContractValue):
    receipt: ResourceAcquisitionReceipt
    destination: OwnerRef
    cleanup_owner: OwnerRef
    acceptance_generation: Generation
    expected_material_commitment: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def exact_transfer(self) -> Self:
        if self.receipt.stage not in ("owned", "transferred"):
            raise ValueError("Only owned resource receipts may transfer.")
        if self.receipt.content_commitment != self.expected_material_commitment:
            raise ValueError("Transfer material commitment conflicts.")
        if self.cleanup_owner != self.receipt.command.intent.cleanup_owner:
            raise ValueError("Transfer cleanup owner conflicts with acquisition.")
        return self


class ResourceTransferCommand(ExpectedOperation[ResourceTransferIntent]):
    kind: Literal["resource_transfer"] = "resource_transfer"
    schema_version: Literal[1] = 1
    mode: Literal["transfer"] = "transfer"
    receipt_stage: Literal["accepted"] = "accepted"

    @model_validator(mode="after")
    def exact_owner(self) -> Self:
        if self.source != self.intent.receipt.command.destination:
            raise ValueError("Transfer source does not own the receipt.")
        if self.destination != self.intent.destination:
            raise ValueError("Transfer destination conflicts with intent.")
        return self


class ResourceTransferReceipt(ContractValue):
    command: ResourceTransferCommand
    receipt_id: Identifier
    stage: Literal["pending", "accepted", "releasing", "uncertain", "released"]
    operation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    destination_pin_owner: Identifier

    @model_validator(mode="after")
    def exact_shape(self) -> Self:
        expected = hashlib.sha256(
            canonical_durable_json_bytes(
                self.command.operation.model_dump(mode="json"), "resource_operation_identity"
            )
        ).hexdigest()
        if self.operation_digest != expected:
            raise ValueError("Transfer receipt operation identity conflicts.")
        return self


def resource_operation_digest(command: ResourceAcquisitionCommand | ResourceTransferCommand) -> str:
    return hashlib.sha256(
        canonical_durable_json_bytes(
            command.operation.model_dump(mode="json"), "resource_operation_identity"
        )
    ).hexdigest()


ResourcePreparationAuthorization.model_rebuild()


def _preparation_selector(command):
    if isinstance(command, ResourceTransferCommand):
        return command.intent.receipt.command.intent.selector
    return command.intent.selector


def _preparation_target(command):
    if isinstance(command, ResourceTransferCommand):
        return ObjectRef(
            owner=command.destination,
            kind="resource_transfer",
            object_id=resource_operation_digest(command),
            incarnation=command.destination.incarnation,
            revision=command.intent.acceptance_generation,
        )
    return command.intent.selector.resource


def _journal_capacity_envelope(value: Mapping[str, object]) -> dict[str, object]:
    """Upper bound all live snapshots and remaining events without double counting.

    Recomputed from durable commands on every publication, including unrelated
    manifest/authorization writes. No reservation can disappear on restart.
    Receipt IDs, digests, stages and diagnostic codes are runtime-bounded here;
    command/member payloads use the exact persisted operation's full size.
    """
    envelope = dict(value)
    raw_events = value["events"]
    if not isinstance(raw_events, list):
        raise ResourceOwnerUnavailable("Invalid resource event history.")
    events = list(raw_events)
    for family in ("operations", "transfers"):
        records = {}
        raw_records = value[family]
        if not isinstance(raw_records, dict):
            raise ResourceOwnerUnavailable("Invalid resource operation history.")
        for digest, record in raw_records.items():
            if not isinstance(digest, str) or not isinstance(record, dict):
                raise ResourceOwnerUnavailable("Invalid resource operation record.")
            record = cast("dict[str, Any]", record)
            if record.get("responsibility_settled") is True:
                records[digest] = record
                continue
            command = json.loads(record["command"])
            receipt = {
                "command": command,
                "receipt_id": "res_" + "0" * 32,
                "stage": "transferred",
                "operation_digest": digest,
            }
            if family == "operations":
                receipt.update(
                    {
                        "material_ids": record["material_ids"],
                        "content_commitment": "sha256:" + "0" * 64,
                        "manifest_commitment": "0" * 64,
                        "material_count": RESOURCE_MAX_MATERIALS,
                        "total_bytes": RESOURCE_MAX_RESERVED_BYTES,
                        "allowed_operations": command["intent"]["allowed_operations"],
                    }
                )
            else:
                receipt["destination_pin_owner"] = "resource-transfer:" + "0" * 64 + ":" + digest
            records[digest] = {
                **record,
                "stage": "releasing",
                "receipt": receipt,
                ("owned_receipt" if family == "operations" else "accepted_receipt"): receipt,
                "acknowledged_material_ids": record["material_ids"],
                # False is the larger JSON boolean. The actual ACK is True.
                "registration_acknowledged": False,
                "responsibility_settled": False,
                # Dynamically created class names may require six-byte JSON escapes.
                "failure_code": "\x01" * 128,
            }
            for _ in range(record["event_slots_remaining"]):
                events.append(
                    {
                        "operation" if family == "operations" else "transfer": digest,
                        "stage": "responsibility_settled",
                        "material_count": RESOURCE_MAX_MATERIALS,
                    }
                )
        envelope[family] = records
    envelope["events"] = events
    return envelope


class _ResourceJournal:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.path = self.root / "resource-owner.json"
        self._lock = threading.RLock()

    @contextmanager
    def locked(self):
        with (
            self._lock,
            cooperative_path_lock(
                self.root, "resource-owner", lock_directory_name="cayu-resource-owner-locks"
            ),
        ):
            value = self._read()
            original = copy.deepcopy(value)
            yield value
            if value != original:
                self._write(value)

    @asynccontextmanager
    async def transaction(self, *, commit_guard=None):
        """Retain a thread-owned transaction through commit or definite abort.

        The loop edits a detached snapshot while its worker holds the journal
        lock. Only the worker reads, publishes, syncs, or releases that lock.
        Guarded publication authorizes commit dispatch, not its slow settlement.
        """
        loop = asyncio.get_running_loop()
        ready = loop.create_future()
        finish = threading.Event()
        commit = False

        class Aborted(Exception):
            pass

        def worker():
            try:
                with self.locked() as value:
                    loop.call_soon_threadsafe(ready.set_result, _ResourceOutcome(result=value))
                    finish.wait()
                    if not commit:
                        raise Aborted()
                return _ResourceOutcome()
            except Aborted as aborted:
                # The lock attaches unlock/close failures to the abort sentinel.
                # Suppress only the sentinel, never its original cleanup evidence.
                return _ResourceOutcome(error=aborted.__cause__)
            except BaseException as error:
                if not finish.is_set():
                    loop.call_soon_threadsafe(ready.set_result, _ResourceOutcome(error=error))
                return _ResourceOutcome(error=error)

        # A retained transaction must not occupy the default executor while an
        # async mandate resolver may need that same executor to authorize commit.
        completed = loop.create_future()

        def run_worker():
            outcome = worker()
            loop.call_soon_threadsafe(completed.set_result, outcome)

        context = copy_context()
        threading.Thread(
            target=context.run, args=(run_worker,), name="cayu-resource-journal"
        ).start()

        primary = None
        try:
            value = await _resource_outcome(ready)
            yield value
            if commit_guard is None:
                commit = True
                self._dispatch_commit(value, finish)
            else:
                async with commit_guard:
                    commit = True
                    self._dispatch_commit(value, finish)
        except BaseException as error:
            primary = error
        finally:
            finish.set()
            try:
                await _resource_outcome(completed)
            except BaseException as error:
                if primary is not None and error is not primary:
                    _raise_diagnostic_failure(primary, error)
                raise
        if primary is not None:
            raise primary

    def _dispatch_commit(self, value, finish):
        """Release the prepared transaction to its retained filesystem worker."""
        finish.set()

    def _read(self) -> dict[str, object]:
        if not self.path.exists():
            return {
                "schema_version": RESOURCE_JOURNAL_SCHEMA,
                "operations": {},
                "transfers": {},
                "manifests": {},
                "authorizations": {},
                "events": [],
            }
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ResourceOwnerUnavailable("Resource owner journal is unavailable.") from exc
        if (
            type(value) is not dict
            or value.get("schema_version") != RESOURCE_JOURNAL_SCHEMA
            or type(value.get("operations")) is not dict
            or type(value.get("transfers")) is not dict
            or type(value.get("manifests")) is not dict
            or type(value.get("authorizations")) is not dict
            or type(value.get("events")) is not list
        ):
            raise ResourceOwnerUnavailable("Resource owner journal schema is invalid.")
        return value

    def bind_identity(self, *, owner: OwnerRef, store: LocalArtifactStore) -> None:
        expected_owner = canonical_durable_json_bytes(
            owner.model_dump(mode="json"), "resource_owner"
        )
        with self.locked() as journal:
            existing_owner = journal.get("owner")
            existing_store = journal.get("artifact_store")
            expected_store = {
                "id": store.id,
                "root": str(store.root),
                "root_identity": [str(part) for part in store._root_identity],
            }
            if existing_owner is None and existing_store is None:
                if any(
                    journal.get(name) for name in ("operations", "transfers", "manifests", "events")
                ):
                    raise ResourceOwnerUnavailable("Resource owner journal identity is missing.")
                journal["owner"] = json.loads(expected_owner)
                journal["artifact_store"] = expected_store
            elif existing_owner != json.loads(expected_owner) or existing_store != expected_store:
                raise ResourceOwnerUnavailable("Resource owner journal identity conflicts.")

    def _write(self, value: Mapping[str, object]) -> None:
        events = value["events"]
        if not isinstance(events, list) or _event_usage(value) > RESOURCE_MAX_EVENTS:
            raise ResourceOwnerError("Resource owner event capacity exhausted.")
        # Every writer must preserve the terminal-state envelope of every live
        # operation. Metadata/manifests/authorizations cannot spend that space.
        canonical_durable_json_bytes(
            _journal_capacity_envelope(value),
            "resource_owner_journal_capacity",
            max_bytes=RESOURCE_JOURNAL_MAX_BYTES,
            max_nodes=RESOURCE_JOURNAL_MAX_NODES,
        )
        data = canonical_durable_json_bytes(dict(value), "resource_owner_journal")
        fd, temp_name = tempfile.mkstemp(prefix=".resource-owner-", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
            directory_fd = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(temp_name)
            raise


class LocalArtifactResourceOwner(ResourceSelectorOwner):
    """Local durable artifact owner; only exact artifact and manifest selectors qualify."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        owner: OwnerRef,
        artifact_store: ArtifactStore,
        redactor: SecretRedactor | None = None,
        preparation_reader: ResourcePreparationReader | None = None,
    ) -> None:
        if type(owner) is not OwnerRef or not isinstance(artifact_store, ArtifactStore):
            raise TypeError("owner and artifact_store must be qualified values.")
        if not isinstance(artifact_store, LocalArtifactStore):
            raise ResourceOwnerUnsupported("Only the local durable artifact owner is qualified.")
        if artifact_store.supports_pins is not True:
            raise ResourceOwnerUnsupported("Artifact store does not provide durable pins.")
        self._owner = owner
        self._store = artifact_store
        self._journal = _ResourceJournal(Path(root))
        self._redactor = redactor or SecretRedactor()
        if preparation_reader is not None and (
            not isinstance(preparation_reader, ResourcePreparationReader)
            or preparation_reader.owner != owner
        ):
            raise ResourceOwnerUnsupported("Preparation authority registration conflicts.")
        if preparation_reader is None:
            raise ResourceOwnerUnsupported("A registered preparation authority is required.")
        self._preparation_reader = preparation_reader
        if isinstance(
            preparation_reader, MandateResourcePreparationReader
        ) and preparation_reader._store_binding != _registered_artifact_store(artifact_store):
            raise ResourceOwnerUnsupported("Artifact store differs from registered authority.")
        self._journal.bind_identity(owner=self.owner, store=artifact_store)
        self._workers: set[asyncio.Task] = set()

    @contextmanager
    def _sync_mutation_guard(self, *others: LocalArtifactResourceOwner):
        """Nonblocking process-shared election, retained across external awaits.

        This is separate from the short journal publication lock. A competing
        retry refuses rather than blocking the event loop or redispatching.
        """
        roots = sorted({owner._journal.root for owner in (self, *others)})
        with ExitStack() as stack:
            for root in roots:
                try:
                    stack.enter_context(
                        cooperative_path_lock(
                            root,
                            "mutation",
                            lock_directory_name="cayu-resource-mutation-locks",
                            blocking=False,
                        )
                    )
                except BlockingIOError:
                    raise ResourceOwnerUnavailable("Resource mutation remains owned.") from None
            yield

    @asynccontextmanager
    async def _mutation_guard(self, *others: LocalArtifactResourceOwner):
        """Retain lock setup and teardown without blocking the event loop."""
        loop = asyncio.get_running_loop()
        ready = loop.create_future()
        completed = loop.create_future()
        finish = threading.Event()
        primary = None
        entered = False

        def worker():
            nonlocal entered
            try:
                with self._sync_mutation_guard(*others):
                    entered = True
                    loop.call_soon_threadsafe(ready.set_result, _ResourceOutcome())
                    finish.wait()
                    if primary is not None:
                        raise primary
                outcome = _ResourceOutcome()
            except BaseException as error:
                outcome = _ResourceOutcome(error=error)
                if not entered:
                    loop.call_soon_threadsafe(ready.set_result, outcome)
            loop.call_soon_threadsafe(completed.set_result, outcome)

        context = copy_context()
        threading.Thread(target=context.run, args=(worker,), name="cayu-resource-mutation").start()
        try:
            await _resource_outcome(ready)
            yield
        except BaseException as error:
            primary = error
        finally:
            finish.set()
            await _resource_outcome(completed)
        if primary is not None:
            raise primary

    async def _observe(
        self, task: asyncio.Task[_ResourceOutcome[_ResultT]], work: _PreparationWork | None = None
    ) -> _ResultT:
        outcome = await await_shielded_task_outcome(
            task,
            timeout_s=RESOURCE_FOREGROUND_TIMEOUT_S,
            timeout_after_cancellation_s=0.0,
        )
        error = outcome.error
        if error is None and outcome.result is not None:
            error = outcome.result.error
        if outcome.cancellation is not None:
            if work is not None:
                work.stopped = True
            restore_task_cancellation_requests(
                outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
            )
            if error is not None:
                raise outcome.cancellation from error
            raise outcome.cancellation
        if error is not None:
            raise error
        if outcome.timed_out:
            if work is not None:
                work.stopped = True
            raise ResourceOwnerUnavailable("Resource settlement remains owned; drain or retry.")
        assert outcome.result is not None
        return cast("_ResultT", outcome.result.result)

    async def _owned(
        self,
        operation: Callable[[], Awaitable[_ResultT]],
        *others: LocalArtifactResourceOwner,
        work: _PreparationWork | None = None,
    ) -> _ResultT:
        # Cleanup may continue after its first effect, but cancellation while
        # acquiring its fence must not start a previously undispatched release.
        entry_work = work if work is not None else _PreparationWork()

        async def run() -> _ResourceOutcome[_ResultT]:
            # Shielded futures may log late exceptions after their observer has
            # cancelled. Carry originals as data until an explicit owner observes
            # them; never send raw exception text to the loop exception handler.
            try:
                async with self._mutation_guard(*others):
                    entry_work.check()
                    for owner in (self, *others):
                        await _resource_io(owner._validate_store_identity)
                    entry_work.check()
                    return _ResourceOutcome(result=await operation())
            except BaseException as error:
                return _ResourceOutcome(error=error)

        task = asyncio.create_task(run())
        self._workers.add(task)
        task.add_done_callback(_consume_resource_task)
        try:
            return await self._observe(task, entry_work)
        finally:
            if task.done():
                self._workers.discard(task)

    async def drain(self) -> None:
        """Observe retained in-process workers without cancelling their effects."""
        for task in tuple(self._workers):
            try:
                await self._observe(task)
            finally:
                if task.done():
                    self._workers.discard(task)

    @property
    def owner(self) -> OwnerRef:
        return self._owner

    def _validate_store_identity(self):
        try:
            current = os.stat(self._store.root, follow_symlinks=False)
        except OSError:
            raise ResourceOwnerUnavailable("Artifact store root is unavailable.") from None
        if (
            not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != self._store._root_identity
        ):
            raise ResourceOwnerUnavailable("Artifact store physical identity changed.")

    async def _unpin_resource(self, material_id, pin_owner):
        try:
            await self._store._release_resource_pin(material_id, owner=pin_owner)
        except FileNotFoundError:
            # Missing material proves absence only in the original physical store.
            await _resource_io(self._validate_store_identity)

    def _pin_owner(self, digest: str, *, transfer: bool = False) -> str:
        receiver = hashlib.sha256(
            canonical_durable_json_bytes(self.owner.model_dump(mode="json"), "pin_receiver")
        ).hexdigest()
        family = "resource-transfer" if transfer else "resource"
        return f"{family}:{receiver}:{digest}"

    def canonicalize(self, selector: ResourceSelector) -> ResourceSelector:
        if type(selector) is not ResourceSelector or selector.resource.owner != self.owner:
            raise ResourceOwnerUnsupported("Selector is not owned by this resource owner.")
        if selector.mode != "exact" or selector.resource.revision is None:
            raise ResourceOwnerUnsupported("Only exact pinned selectors are qualified.")
        if selector.resource.kind not in (RESOURCE_KIND_ARTIFACT, RESOURCE_KIND_FOLDER):
            raise ResourceOwnerUnsupported("Resource kind is not qualified.")
        if selector.resource.kind == RESOURCE_KIND_FOLDER:
            if (
                selector.resource.revision != 1
                or re.fullmatch(r"[0-9a-f]{64}", selector.resource.incarnation) is None
                or selector.resource.object_id != "manifest_" + selector.resource.incarnation[:32]
            ):
                raise ResourceOwnerConflict("Folder manifest identity is not canonical.")
        elif (
            selector.resource.revision != 1
            or re.fullmatch(r"[0-9a-f]{64}", selector.resource.incarnation) is None
        ):
            raise ResourceOwnerConflict("Artifact publication identity is not canonical.")
        return selector

    async def _validate_selection(self, selector: ResourceSelector) -> None:
        """Observe source state inside the retained acquisition worker, not the loop."""
        self.canonicalize(selector)
        if selector.resource.kind == RESOURCE_KIND_FOLDER:
            await _resource_io(self._manifest_for, selector)
        else:
            observed = await _resource_io(
                self.artifact_selector, self._store, self.owner, selector.resource.object_id
            )
            if observed != selector:
                raise ResourceOwnerConflict("Artifact publication identity conflicts.")

    @staticmethod
    def artifact_selector(
        store: LocalArtifactStore, owner: OwnerRef, artifact_id: str
    ) -> ResourceSelector:
        """Observe source identity before host registration; this grants no authority."""
        if not isinstance(store, LocalArtifactStore) or type(owner) is not OwnerRef:
            raise ResourceOwnerUnsupported("Only local immutable artifact selectors are qualified.")
        metadata = store._resource_metadata(artifact_id)
        return ResourceSelector(
            resource=ObjectRef(
                owner=owner,
                kind=RESOURCE_KIND_ARTIFACT,
                object_id=metadata.id,
                incarnation=_artifact_incarnation(metadata),
                revision=1,
            )
        )

    def register_manifest(self, manifest: FolderInputManifest) -> ResourceSelector:
        """Persist trusted folder material and return its exact selector."""
        if type(manifest) is not FolderInputManifest:
            raise TypeError("manifest must be an exact FolderInputManifest.")
        if not manifest.entries:
            raise ResourceOwnerUnsupported("Empty folder manifests are not acquirable resources.")
        if {entry.member.resource.owner for entry in manifest.entries} - {self.owner}:
            raise ResourceOwnerUnsupported("Folder member owner is not this owner.")
        for member in manifest.retained_members:
            self.canonicalize(ResourceSelector(resource=member.resource))
            if (
                member.resource
                != self.artifact_selector(
                    self._store, self.owner, member.resource.object_id
                ).resource
            ):
                raise ResourceOwnerConflict("Folder member publication identity conflicts.")
        encoded = canonical_durable_json_bytes(manifest.model_dump(mode="json"), "input_manifest")
        identity = hashlib.sha256(encoded).hexdigest()
        selector = ResourceSelector(
            resource=ObjectRef(
                owner=self.owner,
                kind=RESOURCE_KIND_FOLDER,
                object_id="manifest_" + identity[:32],
                incarnation=identity,
                revision=1,
            )
        )
        with self._journal.locked() as journal:
            manifests = journal["manifests"]
            assert isinstance(manifests, dict)
            previous = manifests.get(selector.resource.object_id)
            current = manifest.model_dump(mode="json")
            if previous is not None and previous != current:
                raise ResourceOwnerConflict("Manifest selector conflicts with stored material.")
            manifests[selector.resource.object_id] = current
        return selector

    def _manifest_for(self, selector: ResourceSelector) -> FolderInputManifest:
        with self._journal.locked() as journal:
            manifests = journal["manifests"]
            assert isinstance(manifests, dict)
            raw = manifests.get(selector.resource.object_id)
        if raw is None:
            raise ResourceOwnerUnavailable("Exact folder manifest is unavailable.")
        try:
            manifest = FolderInputManifest.model_validate(raw)
        except (TypeError, ValueError) as exc:
            raise ResourceOwnerUnavailable("Exact folder manifest is invalid.") from exc
        encoded = canonical_durable_json_bytes(manifest.model_dump(mode="json"), "input_manifest")
        if (
            hashlib.sha256(encoded).hexdigest() != selector.resource.incarnation
            or selector.resource.revision != 1
        ):
            raise ResourceOwnerConflict("Folder manifest identity changed.")
        return manifest

    def contains(self, parent: ResourceSelector, child: ResourceSelector) -> bool:
        return self.canonicalize(parent) == self.canonicalize(child)

    def _prepare(self, command: ResourceAcquisitionCommand) -> ResourceAcquisitionCommand:
        checked = self._prepare_identity(command)
        self.canonicalize(checked.intent.selector)
        self._validate_bounds(checked)
        return checked

    def _prepare_identity(self, command: ResourceAcquisitionCommand) -> ResourceAcquisitionCommand:
        checked = prepare_contract(ResourceAcquisitionCommand, command, redactor=self._redactor)
        if checked.source != self.owner or checked.intent.cleanup_owner != self.owner:
            raise ResourceOwnerUnsupported("Resource command is not owned by this receiver.")
        return checked

    def _validate_bounds(self, command: ResourceAcquisitionCommand) -> None:
        if command.intent.selector.resource.kind != RESOURCE_KIND_FOLDER:
            return
        manifest = self._manifest_for(command.intent.selector)
        if (
            len(manifest.retained_members) > command.intent.max_materials
            or sum(member.size_bytes for member in manifest.retained_members)
            > command.intent.max_total_bytes
            or len(canonical_durable_json_bytes(manifest.model_dump(mode="json"), "input_manifest"))
            > command.intent.max_manifest_bytes
        ):
            raise ResourceOwnerUnsupported("Folder exceeds deterministic acquisition bounds.")

    async def authorize(
        self, command: ResourceAcquisitionCommand, *, permit: PermitCommand
    ) -> ResourcePreparationReceipt:
        """Persist trusted bounded preparation before any acquisition dispatch."""
        checked = self._prepare_identity(command)
        permit = prepare_contract(PermitCommand, permit, redactor=self._redactor)
        work = _PreparationWork()

        async def admitted():
            await _resource_io(self._prepare, checked)
            await self._validate_selection(checked.intent.selector)
            work.check()
            async with self._preparation_guard(checked, permit) as evidence:
                work.check()
            return await self._authorize(checked, permit, evidence, work=work)

        return await self._owned(admitted, work=work)

    @asynccontextmanager
    async def _preparation_guard(self, command, permit):
        reader = self._preparation_reader
        if reader is None or reader.owner != self.owner:
            raise ResourceOwnerUnsupported("A registered preparation authority is required.")
        async with reader.acquire(command, permit) as raw:
            evidence = prepare_contract(
                ResourcePreparationAuthorization, raw, redactor=self._redactor
            )
            if evidence.command != command or evidence.permit != permit:
                raise ResourceOwnerConflict("Preparation authority returned conflicting evidence.")
            if evidence.lease.operation_digest != resource_operation_digest(command):
                raise ResourceOwnerConflict("Preparation lease operation conflicts.")
            if evidence.lease.receiver != self.owner:
                raise ResourceOwnerConflict("Preparation lease receiver conflicts.")
            if evidence.lease.expires_at_ms <= int(time.time() * 1000):
                raise ResourceOwnerUnavailable("Preparation lease expired.")
            yield evidence

    async def _authorize(self, checked, permit, evidence, *, work):
        request = permit.intent.request
        if (
            permit.source != self.owner
            or permit.destination != self.owner
            or request.target != _preparation_target(checked)
            or request.source_operation != checked.operation
            or request.effect_scope
            != ("transfer" if isinstance(checked, ResourceTransferCommand) else "acquire")
            or request.target.owner != self.owner
            or permit.operation.application_scope != self.owner.application_scope
        ):
            raise ResourceOwnerUnsupported("Preparation responsibility conflicts with acquisition.")
        digest = resource_operation_digest(checked)
        encoded = canonical_durable_json_bytes(checked.model_dump(mode="json"), "resource_command")
        receipt = ResourcePreparationReceipt(
            operation_digest=digest,
            owner=self.owner,
            command_bytes_sha256=hashlib.sha256(encoded).hexdigest(),
            permit=permit,
            lease=evidence.lease,
        )
        async with self._journal.transaction(
            commit_guard=self._preparation_effect_guard(checked, evidence, work)
        ) as journal:
            authorizations = journal["authorizations"]
            assert isinstance(authorizations, dict)
            existing = authorizations.get(digest)
            if existing is not None:
                if not isinstance(existing, dict) or existing.get("command") != encoded.decode():
                    raise ResourceOwnerConflict("Preparation operation key conflicts.")
                stored = ResourcePreparationReceipt.model_validate(existing["receipt"])
                if stored.permit != permit:
                    raise ResourceOwnerConflict("Preparation permit conflicts.")
                if (
                    stored.lease.operation_digest != digest
                    or stored.lease.receiver != self.owner
                    or stored.lease.expires_at_ms <= int(time.time() * 1000)
                ):
                    raise ResourceOwnerUnavailable("Preparation lease is expired or invalid.")
                return stored
            if len(authorizations) >= RESOURCE_MAX_OPERATIONS:
                raise ResourceOwnerError("Preparation capacity exhausted.")
            authorizations[digest] = {
                "command": encoded.decode(),
                "receipt": receipt.model_dump(mode="json"),
            }
        return receipt

    async def _require_authorization(
        self,
        command: ResourceAcquisitionCommand | ResourceTransferCommand,
        preparation: ResourcePreparationReceipt | None,
    ) -> ResourcePreparationReceipt:
        digest = resource_operation_digest(command)
        encoded = canonical_durable_json_bytes(command.model_dump(mode="json"), "resource_command")
        async with self._journal.transaction() as journal:
            record = journal["authorizations"].get(digest)
        if not isinstance(record, dict) or record.get("command") != encoded.decode():
            raise ResourceOwnerUnavailable("Trusted preparation is required.")
        try:
            stored = ResourcePreparationReceipt.model_validate(record["receipt"])
        except (TypeError, ValueError) as exc:
            raise ResourceOwnerUnavailable("Trusted preparation is invalid.") from exc
        if (
            stored.owner != self.owner
            or stored.operation_digest != digest
            or stored.command_bytes_sha256 != hashlib.sha256(encoded).hexdigest()
            or stored.permit.source != self.owner
            or stored.permit.intent.request.target.owner != self.owner
            or stored.lease.operation_digest != digest
            or stored.lease.receiver != self.owner
            or stored.lease.expires_at_ms <= int(time.time() * 1000)
        ):
            raise ResourceOwnerUnavailable("Trusted preparation identity is invalid.")
        if preparation is not None and preparation != stored:
            raise ResourceOwnerConflict("Preparation evidence conflicts with owner state.")
        return stored

    async def _record(
        self,
        digest: str,
        command: ResourceAcquisitionCommand,
        material_ids: tuple[str, ...],
    ) -> dict[str, object] | None:
        encoded = canonical_durable_json_bytes(
            command.model_dump(mode="json"), "resource_command"
        ).decode()
        async with self._journal.transaction() as journal:
            operations = journal["operations"]
            assert isinstance(operations, dict)
            existing = operations.get(digest)
            if existing is not None:
                if not isinstance(existing, dict) or existing.get("command") != encoded:
                    raise ResourceOwnerConflict("Resource operation key conflicts.")
                return existing
            if len(operations) >= RESOURCE_MAX_OPERATIONS:
                raise ResourceOwnerError("Resource operation capacity exhausted.")
            reserved = _reserved_bytes(journal)
            if reserved + command.intent.max_total_bytes > RESOURCE_MAX_RESERVED_BYTES:
                raise ResourceOwnerError("Resource byte reservation capacity exhausted.")
            event_slots = _reserve_events(journal)
            operations[digest] = {
                "command": encoded,
                "stage": "pending",
                "receipt": None,
                "material_ids": list(material_ids),
                "event_slots_remaining": event_slots,
            }
            _append_resource_event(journal, "operations", digest, "pending")
            return None

    async def _set_record(
        self, digest: str, stage: str, receipt: ResourceAcquisitionReceipt, *, commit_guard=None
    ) -> None:
        async with self._journal.transaction(commit_guard=commit_guard) as journal:
            operations = journal["operations"]
            assert isinstance(operations, dict)
            record = operations.get(digest)
            if not isinstance(record, dict):
                raise ResourceOwnerUnavailable("Resource operation record disappeared.")
            if stage == "owned":
                original = record.get("owned_receipt")
                encoded_receipt = receipt.model_dump(mode="json")
                if original is not None and original != encoded_receipt:
                    raise ResourceOwnerConflict("Original resource ownership receipt conflicts.")
                record["owned_receipt"] = encoded_receipt
            record["stage"] = stage
            record["receipt"] = receipt.model_dump(mode="json")
            _append_resource_event(journal, "operations", digest, stage)

    async def acquire(
        self,
        command: ResourceAcquisitionCommand,
        *,
        preparation: ResourcePreparationReceipt | None = None,
    ) -> ResourceAcquisitionReceipt:
        command = self._prepare_identity(command)
        work = _PreparationWork()

        async def admitted():
            await _resource_io(self._prepare, command)
            stored = await self._require_authorization(command, preparation)
            await self._validate_selection(command.intent.selector)
            await self._revalidate_preparation(command, stored, work)
            return await self._acquire_once(command, work)

        return await self._owned(admitted, work=work)

    async def _revalidate_preparation(self, command, stored, work=None):
        if work is not None:
            work.check()
        reader = self._preparation_reader
        if reader is None or reader.owner != self.owner:
            raise ResourceOwnerUnsupported("A registered preparation authority is required.")
        await reader.revalidate(command, stored.lease)
        if work is not None:
            work.check()

    @asynccontextmanager
    async def _preparation_effect_guard(self, command, stored, work: _PreparationWork):
        work.check()
        async with self._preparation_reader.revalidation_guard(command, stored.lease):
            work.check()
            if stored.lease.expires_at_ms <= int(time.time() * 1000):
                raise ResourceOwnerUnavailable("Preparation lease expired.")
            yield

    async def _pin_prepared_resource(self, artifact_id, pin_owner, command, stored, work):
        async def dispatch(start):
            pending = None
            primary = None
            try:
                async with self._preparation_effect_guard(command, stored, work):
                    # start submits to the executor synchronously. Merely
                    # scheduling an async pin coroutine is not dispatch proof.
                    # Observe the executor future itself. A wrapper Task would
                    # also be cancelled by loop shutdown, cancelling its await
                    # without stopping the already-running filesystem thread.
                    pending = start()
            except BaseException as error:
                primary = error
            if pending is not None:
                # A failing guard exit cannot abandon an already submitted pin.
                # The public observer is bounded; this owned settlement is not.
                outcome = await await_shielded_task_outcome(pending)
                error = outcome.error
                if outcome.cancellation is not None:
                    restore_task_cancellation_requests(
                        outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
                    )
                    if error is not None:
                        if primary is not None:
                            error = BaseExceptionGroup(
                                "Resource dispatch failures", [primary, error]
                            )
                        raise outcome.cancellation from error
                    if primary is not None:
                        raise outcome.cancellation from primary
                    raise outcome.cancellation
                if error is not None:
                    if primary is not None:
                        _raise_diagnostic_failure(primary, error)
                    raise error
            if primary is not None:
                raise primary

        await self._store._pin_resource(artifact_id, owner=pin_owner, dispatch=dispatch)

    async def _acquire_once(self, command: ResourceAcquisitionCommand, work: _PreparationWork):
        digest = resource_operation_digest(command)
        selector = command.intent.selector
        planned_material_ids = (
            tuple(
                member.resource.object_id
                for member in (await _resource_io(self._manifest_for, selector)).retained_members
            )
            if selector.resource.kind == RESOURCE_KIND_FOLDER
            else (selector.resource.object_id,)
        )
        existing = await self._record(digest, command, planned_material_ids)
        if existing is not None and existing.get("receipt") is not None:
            existing_receipt = ResourceAcquisitionReceipt.model_validate(existing["receipt"])
            if existing_receipt.stage in {"releasing", "released"}:
                raise ResourceOwnerUnavailable("Resource release remains pending settlement.")
            return existing_receipt
        if existing is not None:
            raise ResourceOwnerUnavailable("Resource acquisition remains pending reconciliation.")
        return await self._acquire(command, digest, work)

    async def _cleanup_stopped(self, command, family, error):
        try:
            digest = resource_operation_digest(command)
            async with self._journal.transaction() as journal:
                record = journal[family].get(digest)
                if record is None:
                    return
                expected = canonical_durable_json_bytes(
                    command.model_dump(mode="json"), "resource_cleanup_command"
                ).decode()
                if record.get("command") != expected:
                    raise ResourceOwnerConflict(
                        "Stopped cleanup command conflicts with retained intent."
                    )
                if record["stage"] not in {"pending", "uncertain", "cleaning"}:
                    return
                material_ids = tuple(record["material_ids"])
            if family == "operations":
                await self._release_pending_materials(digest, material_ids)
            else:
                await self._release_pending_transfer(digest, material_ids)
        except BaseException as cleanup:
            _raise_diagnostic_failure(error, cleanup)

    async def _acquire(
        self, command: ResourceAcquisitionCommand, digest: str, work: _PreparationWork
    ):
        try:
            return await self._acquire_impl(command, digest, work)
        except BaseException as error:
            if work is not None and work.stop_requested():
                await self._cleanup_stopped(command, "operations", error)
            raise

    async def _acquire_impl(
        self, command: ResourceAcquisitionCommand, digest: str, work: _PreparationWork
    ):
        # Also classify already-retained impossible work as permanent during recovery.
        await _resource_io(self._validate_bounds, command)
        stored = await self._require_authorization(command, None)
        await self._preparation_reader.register_responsibility(command, stored.permit)
        async with self._journal.transaction() as journal:
            journal["operations"][digest]["registration_acknowledged"] = True
        await self._revalidate_preparation(command, stored, work)
        selector = command.intent.selector
        if selector.resource.kind == RESOURCE_KIND_FOLDER:
            manifest = await _resource_io(self._manifest_for, selector)
            members = manifest.retained_members
            if len(members) > command.intent.max_materials:
                raise ResourceOwnerUnsupported("Folder material count exceeds command bound.")
            material_ids: list[str] = []
            total_bytes = 0
            try:
                for member in members:
                    artifact_id = member.resource.object_id
                    await self._pin_prepared_resource(
                        artifact_id,
                        self._pin_owner(digest),
                        command,
                        await self._require_authorization(command, None),
                        work,
                    )
                    material_ids.append(artifact_id)
                    if work is not None:
                        work.check()
                    read = await self._store.read_bytes(
                        artifact_id, max_bytes=command.intent.max_total_bytes
                    )
                    if (
                        type(read) is not ArtifactReadResult
                        or read.truncated
                        or read.metadata.id != artifact_id
                        or member.resource.revision != 1
                        or member.resource.incarnation != _artifact_incarnation(read.metadata)
                        or read.total_bytes != member.size_bytes
                        or hashlib.sha256(
                            canonical_durable_json_bytes(
                                read.metadata.model_dump(mode="json"), "metadata"
                            )
                        ).hexdigest()
                        != member.metadata_sha256
                        or hashlib.sha256(read.content).hexdigest() != member.content_sha256
                    ):
                        raise ResourceOwnerUnavailable("Folder member exact readback failed.")
                    total_bytes += len(read.content)
                    if total_bytes > command.intent.max_total_bytes:
                        raise ResourceOwnerUnsupported("Folder total bytes exceeds command bound.")
                manifest_bytes = canonical_durable_json_bytes(
                    manifest.model_dump(mode="json"), "input_manifest"
                )
                if len(manifest_bytes) > command.intent.max_manifest_bytes:
                    raise ResourceOwnerUnsupported("Folder manifest exceeds command bound.")
                receipt = ResourceAcquisitionReceipt(
                    command=command,
                    receipt_id="res_" + uuid4().hex,
                    stage="owned",
                    operation_digest=digest,
                    material_ids=tuple(material_ids),
                    content_commitment="sha256:" + hashlib.sha256(manifest_bytes).hexdigest(),
                    manifest_commitment=hashlib.sha256(manifest_bytes).hexdigest(),
                    material_count=len(material_ids),
                    total_bytes=total_bytes,
                    allowed_operations=command.intent.allowed_operations,
                )
                await self._set_record(
                    digest,
                    "owned",
                    receipt,
                    commit_guard=self._preparation_effect_guard(
                        command, await self._require_authorization(command, None), work
                    ),
                )
                return receipt
            except BaseException as error:
                await self._retain_uncertain(digest, tuple(material_ids), error)
                raise
        artifact_id = selector.resource.object_id
        material_ids = (artifact_id,)
        try:
            # Keep the pin inside the owned settlement region.  Cancellation
            # immediately after pin dispatch must still persist uncertainty.
            await self._pin_prepared_resource(
                artifact_id,
                self._pin_owner(digest),
                command,
                await self._require_authorization(command, None),
                work,
            )
            if work is not None:
                work.check()
            read = await self._store.read_bytes(
                artifact_id, max_bytes=command.intent.max_total_bytes
            )
            if (
                type(read) is not ArtifactReadResult
                or read.truncated
                or read.metadata.id != artifact_id
                or selector.resource.revision != 1
                or selector.resource.incarnation != _artifact_incarnation(read.metadata)
                or read.total_bytes != len(read.content)
            ):
                raise ResourceOwnerUnavailable("Exact artifact readback failed.")
            content = "sha256:" + hashlib.sha256(read.content).hexdigest()
            receipt = ResourceAcquisitionReceipt(
                command=command,
                receipt_id="res_" + uuid4().hex,
                stage="owned",
                operation_digest=digest,
                material_ids=material_ids,
                content_commitment=content,
                manifest_commitment=hashlib.sha256(
                    canonical_durable_json_bytes(
                        {
                            "artifact_id": artifact_id,
                            "metadata": read.metadata.model_dump(mode="json"),
                        },
                        "resource_manifest",
                    )
                ).hexdigest(),
                material_count=1,
                total_bytes=len(read.content),
                allowed_operations=command.intent.allowed_operations,
            )
            await self._set_record(
                digest,
                "owned",
                receipt,
                commit_guard=self._preparation_effect_guard(
                    command, await self._require_authorization(command, None), work
                ),
            )
            return receipt
        except BaseException as error:
            # Pin remains intentionally retained on every post-dispatch failure;
            # drain/reconcile owns release rather than guessing that pin stopped.
            await self._retain_uncertain(digest, material_ids, error)
            raise

    async def _set_uncertain(
        self, digest: str, material_ids: tuple[str, ...], error: BaseException
    ) -> None:
        """Retain partial ownership; exception text is never persisted."""
        async with self._journal.transaction() as journal:
            operations = journal["operations"]
            assert isinstance(operations, dict)
            record = operations.get(digest)
            if not isinstance(record, dict):
                return
            # Publication may have committed before reporting an error. Never
            # erase an acknowledged-by-readback receipt during diagnostics.
            if record.get("stage") == "owned" and record.get("receipt") is not None:
                return
            record["stage"] = "uncertain"
            record["receipt"] = None
            record["acknowledged_material_ids"] = list(material_ids)
            record["failure_code"] = type(error).__name__[:128]
            _append_resource_event(
                journal, "operations", digest, "uncertain", material_count=len(material_ids)
            )

    async def _retain_uncertain(
        self, digest: str, material_ids: tuple[str, ...], error: BaseException
    ) -> None:
        try:
            await self._set_uncertain(digest, material_ids, error)
        except BaseException as diagnostic_error:
            _raise_diagnostic_failure(error, diagnostic_error)

    async def _release_pending_materials(self, digest: str, material_ids: tuple[str, ...]) -> None:
        """Owner-internal cleanup for work whose acquisition authority expired."""
        await self._begin_pending_cleanup("operations", digest, material_ids)
        for material_id in material_ids:
            await self._unpin_resource(material_id, self._pin_owner(digest))
        async with self._journal.transaction() as journal:
            operations = journal["operations"]
            assert isinstance(operations, dict)
            record = operations.get(digest)
            if isinstance(record, dict) and record.get("stage") == "cleaning":
                record["stage"] = "released"
                record["receipt"] = None
                _append_resource_event(journal, "operations", digest, "released")
        await self._settle_responsibility(digest)

    async def _settle_responsibility(self, digest, family="operations"):
        async with self._journal.transaction() as journal:
            record = journal[family][digest]
            if record["stage"] != "released":
                raise ResourceOwnerUnavailable("Resource cleanup has not settled.")
            if record.get("responsibility_settled") is True:
                return
            schema = (
                ResourceAcquisitionCommand if family == "operations" else ResourceTransferCommand
            )
            command = schema.model_validate_json(record["command"])
            preparation = ResourcePreparationReceipt.model_validate(
                journal["authorizations"][digest]["receipt"]
            )
        await self._preparation_reader.settle_responsibility(
            command, preparation.permit, _ResourceSettlementReader(self, digest, family)
        )
        async with self._journal.transaction() as journal:
            journal[family][digest]["responsibility_settled"] = True
            _append_resource_event(journal, family, digest, "responsibility_settled")

    async def _begin_pending_cleanup(
        self, family: str, digest: str, material_ids: tuple[str, ...]
    ) -> None:
        async with self._journal.transaction() as journal:
            record = journal[family][digest]
            if record["stage"] not in {"pending", "uncertain", "cleaning"}:
                raise ResourceOwnerConflict("Pending cleanup state conflicts.")
            if not material_ids or list(material_ids) != record.get("material_ids"):
                raise ResourceOwnerUnavailable("Pending cleanup material identity is unavailable.")
            if record["stage"] != "cleaning":
                record["stage"] = "cleaning"
                _append_resource_event(journal, family, digest, "cleaning")

    async def readback(
        self, command: ResourceAcquisitionCommand
    ) -> ExactLookup[ResourceAcquisitionReceipt]:
        command = prepare_contract(ResourceAcquisitionCommand, command, redactor=self._redactor)
        return await self._owned(lambda: self._readback(command))

    async def _readback(self, command):
        digest = resource_operation_digest(command)
        encoded = canonical_durable_json_bytes(
            command.model_dump(mode="json"), "resource_command"
        ).decode()
        async with self._journal.transaction() as journal:
            operations = journal["operations"]
            assert isinstance(operations, dict)
            record = operations.get(digest)
            if record is None:
                return ExactNotFound()
            if not isinstance(record, dict) or record.get("command") != encoded:
                return ExactConflict()
            if record.get("receipt") is None:
                return ExactUnavailable()
            try:
                raw_receipt = record["receipt"]
                receipt = ResourceAcquisitionReceipt.model_validate(raw_receipt)
            except (TypeError, ValueError):
                return ExactUnavailable()
            if receipt.stage != "owned":
                return ExactUnavailable()
        try:
            self.canonicalize(command.intent.selector)
            stored_authorization = await self._require_authorization(command, None)
            await self._revalidate_preparation(command, stored_authorization)
            await self._verify_receipt_material(receipt)
        except (ResourceOwnerError, OSError, ValueError):
            return ExactUnavailable()
        return ExactMatch[ResourceAcquisitionReceipt](receipt=receipt)

    async def _verify_receipt_material(self, receipt: ResourceAcquisitionReceipt) -> None:
        selector = receipt.command.intent.selector
        if selector.resource.kind == RESOURCE_KIND_FOLDER:
            manifest = await _resource_io(self._manifest_for, selector)
            if (
                tuple(member.resource.object_id for member in manifest.retained_members)
                != receipt.material_ids
            ):
                raise ResourceOwnerConflict("Folder material identity changed.")
            encoded = canonical_durable_json_bytes(
                manifest.model_dump(mode="json"), "input_manifest"
            )
            if "sha256:" + hashlib.sha256(encoded).hexdigest() != receipt.content_commitment:
                raise ResourceOwnerConflict("Folder manifest commitment changed.")
            members = {member.resource.object_id: member for member in manifest.retained_members}
        else:
            members = {receipt.material_ids[0]: None}
        total = 0
        for artifact_id, member in members.items():
            read = await self._store.read_bytes(
                artifact_id, max_bytes=receipt.command.intent.max_total_bytes
            )
            if (
                type(read) is not ArtifactReadResult
                or read.truncated
                or read.metadata.id != artifact_id
            ):
                raise ResourceOwnerUnavailable("Retained material is unavailable.")
            if member is None:
                content = "sha256:" + hashlib.sha256(read.content).hexdigest()
                if content != receipt.content_commitment:
                    raise ResourceOwnerConflict("Artifact content commitment changed.")
                metadata_commitment = hashlib.sha256(
                    canonical_durable_json_bytes(
                        {
                            "artifact_id": artifact_id,
                            "metadata": read.metadata.model_dump(mode="json"),
                        },
                        "resource_manifest",
                    )
                ).hexdigest()
                if metadata_commitment != receipt.manifest_commitment:
                    raise ResourceOwnerConflict("Artifact metadata commitment changed.")
            if member is not None and (
                read.total_bytes != member.size_bytes
                or hashlib.sha256(
                    canonical_durable_json_bytes(read.metadata.model_dump(mode="json"), "metadata")
                ).hexdigest()
                != member.metadata_sha256
                or hashlib.sha256(read.content).hexdigest() != member.content_sha256
            ):
                raise ResourceOwnerConflict("Folder member commitment changed.")
            total += len(read.content)
        if total != receipt.total_bytes:
            raise ResourceOwnerConflict("Resource total bytes changed.")

    async def release(self, receipt: ResourceAcquisitionReceipt) -> None:
        receipt = prepare_contract(ResourceAcquisitionReceipt, receipt, redactor=self._redactor)

        async def admitted():
            await self._preparation_reader.authorize_release(receipt.command)
            await self._release(receipt)

        await self._owned(admitted)

    async def _release(self, receipt: ResourceAcquisitionReceipt) -> None:
        if type(receipt) is not ResourceAcquisitionReceipt:
            raise TypeError("release requires an owner-issued receipt.")
        command = self._prepare_identity(receipt.command)
        digest = resource_operation_digest(command)
        if digest != receipt.operation_digest:
            raise ResourceOwnerConflict("Release receipt operation conflicts.")
        encoded = canonical_durable_json_bytes(
            command.model_dump(mode="json"), "resource_command"
        ).decode()
        async with self._journal.transaction() as journal:
            operations = journal["operations"]
            assert isinstance(operations, dict)
            record = operations.get(digest)
            if not isinstance(record, dict) or record.get("command") != encoded:
                raise ResourceOwnerUnavailable("Release command is not durably owned.")
            raw_receipt = record.get("receipt")
            if not isinstance(raw_receipt, dict):
                raise ResourceOwnerUnavailable("Release receipt is unavailable.")
            try:
                durable_receipt = ResourceAcquisitionReceipt.model_validate(raw_receipt)
                original_receipt = ResourceAcquisitionReceipt.model_validate(
                    record.get("owned_receipt")
                )
            except (TypeError, ValueError) as exc:
                raise ResourceOwnerUnavailable("Release receipt is invalid.") from exc
        if original_receipt.stage != "owned" or receipt != original_receipt:
            raise ResourceOwnerConflict("Release receipt does not match original ownership.")
        if durable_receipt != original_receipt.model_copy(update={"stage": durable_receipt.stage}):
            raise ResourceOwnerUnavailable("Retained release state conflicts with ownership.")
        if durable_receipt.stage == "released":
            await self._settle_responsibility(digest)
            return
        if durable_receipt.stage in {"owned", "releasing"}:
            active_receipt = durable_receipt
        else:
            raise ResourceOwnerConflict("Release receipt does not match durable ownership.")
        if "release" not in active_receipt.allowed_operations:
            raise ResourceOwnerUnsupported("Receipt does not permit release.")
        releasing = active_receipt.model_copy(update={"stage": "releasing"})
        await self._set_record(digest, "releasing", releasing)
        for material_id in active_receipt.material_ids:
            await self._unpin_resource(material_id, self._pin_owner(digest))
        await self._set_record(
            digest,
            "released",
            releasing.model_copy(update={"stage": "released"}),
        )
        await self._settle_responsibility(digest)

    async def accept_transfer(
        self,
        command: ResourceTransferCommand,
        *,
        source_owner: LocalArtifactResourceOwner,
    ) -> ResourceTransferReceipt:
        if not isinstance(source_owner, LocalArtifactResourceOwner):
            raise TypeError("source_owner must be a local resource owner.")
        # Detach before handing input to the worker/cleanup owner. Acquisition
        # and cancellation cleanup must use the same exact prepared command.
        command = prepare_contract(ResourceTransferCommand, command, redactor=self._redactor)
        work = _PreparationWork()
        return await self._owned(
            lambda: self._accept_transfer(command, source_owner=source_owner, work=work),
            source_owner,
            work=work,
        )

    async def _accept_transfer(self, command, *, source_owner, work: _PreparationWork):
        try:
            return await self._accept_transfer_impl(command, source_owner=source_owner, work=work)
        except BaseException as error:
            if work is not None and work.stop_requested():
                await self._cleanup_stopped(command, "transfers", error)
            raise

    async def _accept_transfer_impl(
        self,
        command: ResourceTransferCommand,
        *,
        source_owner: LocalArtifactResourceOwner,
        work: _PreparationWork,
    ) -> ResourceTransferReceipt:
        if work is not None:
            work.check()
        if not isinstance(source_owner, LocalArtifactResourceOwner):
            raise TypeError("source_owner must be a registered local resource owner.")
        if command.destination != self.owner:
            raise ResourceOwnerUnsupported("Transfer destination is not this owner.")
        if command.source != source_owner.owner:
            raise ResourceOwnerUnsupported("Transfer source owner is not the registered receiver.")
        if (
            self._store.id != source_owner._store.id
            or getattr(self._store, "_root_identity", None)
            != getattr(source_owner._store, "_root_identity", None)
            or getattr(self._store, "root", None) != getattr(source_owner._store, "root", None)
        ):
            raise ResourceOwnerUnsupported("Cross-store transfer is not qualified.")
        receipt = command.intent.receipt
        if "transfer" not in receipt.allowed_operations:
            raise ResourceOwnerUnsupported("Receipt does not permit transfer.")
        permit = prepare_contract(
            PermitCommand,
            self._preparation_reader.transfer_permit(command),
            redactor=self._redactor,
        )
        async with self._preparation_guard(command, permit) as evidence:
            if work is not None:
                work.check()
        await self._authorize(command, permit, evidence, work=work)
        stored = await self._require_authorization(command, None)
        await self._revalidate_preparation(command, stored, work)
        transfer_digest = resource_operation_digest(command)
        encoded_command = canonical_durable_json_bytes(
            command.model_dump(mode="json"), "resource_transfer"
        ).decode()
        async with self._journal.transaction() as journal:
            transfers = journal["transfers"]
            assert isinstance(transfers, dict)
            existing = transfers.get(transfer_digest)
            if existing is not None:
                if not isinstance(existing, dict) or existing.get("command") != encoded_command:
                    raise ResourceOwnerConflict("Transfer operation key conflicts.")
                if existing.get("stage") in {"cleaning", "releasing", "released"}:
                    raise ResourceOwnerUnavailable("Transfer retention is being released.")
                raw_existing = existing.get("receipt")
                if isinstance(raw_existing, dict):
                    existing_receipt = ResourceTransferReceipt.model_validate(raw_existing)
                    if existing_receipt.stage == "accepted":
                        return existing_receipt
                elif existing.get("stage") not in {"pending", "uncertain"}:
                    raise ResourceOwnerUnavailable("Transfer record is invalid.")
            elif len(transfers) >= RESOURCE_MAX_OPERATIONS:
                raise ResourceOwnerError("Resource transfer capacity exhausted.")
            reserved = _reserved_bytes(journal)
            additional = receipt.total_bytes if existing is None else 0
            if reserved + additional > RESOURCE_MAX_RESERVED_BYTES:
                raise ResourceOwnerError("Transfer byte reservation capacity exhausted.")
            if existing is None:
                event_slots = _reserve_events(journal)
                transfers[transfer_digest] = {
                    "command": encoded_command,
                    "receipt": None,
                    "stage": "pending",
                    "material_ids": list(receipt.material_ids),
                    "event_slots_remaining": event_slots,
                }
                _append_resource_event(journal, "transfers", transfer_digest, "pending")
        await self._preparation_reader.register_responsibility(command, stored.permit)
        if work is not None:
            work.check()
        source_lookup = await source_owner._readback(command.intent.receipt.command)
        if (
            not isinstance(source_lookup, ExactMatch)
            or source_lookup.receipt != command.intent.receipt
        ):
            raise ResourceOwnerUnavailable("Source owner did not authenticate the exact receipt.")
        destination_pin = self._pin_owner(transfer_digest, transfer=True)
        transfer = ResourceTransferReceipt(
            command=command,
            receipt_id="tr_" + uuid4().hex,
            stage="pending",
            operation_digest=transfer_digest,
            destination_pin_owner=destination_pin,
        )
        pinned: list[str] = []
        try:
            for material_id in receipt.material_ids:
                await self._pin_prepared_resource(
                    material_id, destination_pin, command, stored, work
                )
                pinned.append(material_id)
            accepted = transfer.model_copy(update={"stage": "accepted"})
            async with self._journal.transaction(
                commit_guard=self._preparation_effect_guard(command, stored, work)
            ) as journal:
                transfers = journal["transfers"]
                assert isinstance(transfers, dict)
                record = transfers.get(transfer_digest)
                if not isinstance(record, dict) or record.get("command") != encoded_command:
                    raise ResourceOwnerUnavailable("Transfer record disappeared.")
                record["stage"] = "accepted"
                record["receipt"] = accepted.model_dump(mode="json")
                record["accepted_receipt"] = accepted.model_dump(mode="json")
                _append_resource_event(journal, "transfers", transfer_digest, "accepted")
        except BaseException as primary_error:
            uncertain = transfer.model_copy(update={"stage": "uncertain"})
            try:
                async with self._journal.transaction() as journal:
                    transfers = journal["transfers"]
                    assert isinstance(transfers, dict)
                    record = transfers.get(transfer_digest)
                    if isinstance(record, dict) and record.get("stage") != "accepted":
                        record["stage"] = "uncertain"
                        record["receipt"] = uncertain.model_dump(mode="json")
                        record["acknowledged_material_ids"] = pinned
                        _append_resource_event(journal, "transfers", transfer_digest, "uncertain")
            except BaseException as diagnostic_error:
                _raise_diagnostic_failure(primary_error, diagnostic_error)
            raise
        return accepted

    async def _release_pending_transfer(self, digest: str, material_ids: tuple[str, ...]) -> None:
        await self._begin_pending_cleanup("transfers", digest, material_ids)
        for material_id in material_ids:
            await self._unpin_resource(material_id, self._pin_owner(digest, transfer=True))
        async with self._journal.transaction() as journal:
            transfers = journal["transfers"]
            assert isinstance(transfers, dict)
            record = transfers.get(digest)
            if isinstance(record, dict) and record.get("stage") == "cleaning":
                record["stage"] = "released"
                record["receipt"] = None
                _append_resource_event(journal, "transfers", digest, "released")
        await self._settle_responsibility(digest, "transfers")

    async def read_transfer(
        self, command: ResourceTransferCommand
    ) -> ExactLookup[ResourceTransferReceipt]:
        command = prepare_contract(ResourceTransferCommand, command, redactor=self._redactor)
        return await self._owned(lambda: self._read_transfer(command))

    async def _read_transfer(self, command):
        found = await self._read_transfer_for_cleanup(command)
        if not isinstance(found, ExactMatch):
            return found
        try:
            stored = await self._require_authorization(command, None)
            await self._revalidate_preparation(command, stored)
        except (ResourceOwnerError, OSError, ValueError):
            return ExactUnavailable()
        return found

    async def _read_transfer_for_cleanup(self, command):
        """Exact durable acceptance; internal callers retain both mutation fences.

        This is settlement readback, not authority for new acquisition or public
        historical access. Source release may finish after a grant is revoked.
        """
        digest = resource_operation_digest(command)
        encoded = canonical_durable_json_bytes(
            command.model_dump(mode="json"), "resource_transfer"
        ).decode()
        async with self._journal.transaction() as journal:
            transfers = journal["transfers"]
            assert isinstance(transfers, dict)
            record = transfers.get(digest)
            if record is None:
                return ExactNotFound()
            if not isinstance(record, dict) or record.get("command") != encoded:
                return ExactConflict()
            try:
                receipt = ResourceTransferReceipt.model_validate(record.get("receipt", record))
                if record.get("stage") != "accepted" or receipt.stage != "accepted":
                    return ExactUnavailable()
                return ExactMatch[ResourceTransferReceipt](receipt=receipt)
            except (TypeError, ValueError):
                return ExactUnavailable()

    async def release_transfer(self, receipt: ResourceTransferReceipt) -> None:
        """Release destination retention using its exact durable acceptance."""
        receipt = prepare_contract(ResourceTransferReceipt, receipt, redactor=self._redactor)
        if receipt.command.destination != self.owner:
            raise ResourceOwnerConflict("Transfer release belongs to another owner.")

        async def admitted():
            await self._preparation_reader.authorize_release(receipt.command.intent.receipt.command)
            await self._release_transfer(receipt)

        await self._owned(admitted)

    async def _release_transfer(self, receipt: ResourceTransferReceipt) -> None:
        async with self._journal.transaction() as journal:
            transfers = journal["transfers"]
            assert isinstance(transfers, dict)
            digest = resource_operation_digest(receipt.command)
            record = transfers.get(digest)
            if (
                not isinstance(record, dict)
                or record.get("command")
                != canonical_durable_json_bytes(
                    receipt.command.model_dump(mode="json"), "resource_transfer"
                ).decode()
            ):
                raise ResourceOwnerUnavailable("Transfer acceptance is unavailable.")
            raw_accepted = record.get("accepted_receipt")
            if not isinstance(raw_accepted, dict):
                raise ResourceOwnerUnavailable("Transfer acceptance is unavailable.")
            accepted = ResourceTransferReceipt.model_validate(raw_accepted)
            raw_current = record.get("receipt")
            durable = (
                ResourceTransferReceipt.model_validate(raw_current)
                if isinstance(raw_current, dict)
                else accepted
            )
        if accepted != receipt:
            raise ResourceOwnerConflict("Transfer release receipt conflicts.")
        if durable.stage == "released":
            await self._settle_responsibility(digest, "transfers")
            return
        if durable.stage not in {"accepted", "releasing"}:
            raise ResourceOwnerUnavailable("Transfer has not settled.")
        digest = durable.operation_digest

        async def publish(stage):
            async with self._journal.transaction() as journal:
                record = journal["transfers"][digest]
                record["stage"] = stage
                record["receipt"] = durable.model_copy(update={"stage": stage}).model_dump(
                    mode="json"
                )
                _append_resource_event(journal, "transfers", digest, stage)

        await publish("releasing")
        for material_id in durable.command.intent.receipt.material_ids:
            # A previous release may have succeeded before process loss,
            # allowing deletion. Absent material cannot retain this pin.
            await self._unpin_resource(material_id, durable.destination_pin_owner)
        await publish("released")
        await self._settle_responsibility(digest, "transfers")

    async def reconcile(
        self, *, source_owners: tuple[LocalArtifactResourceOwner, ...] = ()
    ) -> tuple[str, ...]:
        if any(not isinstance(source, LocalArtifactResourceOwner) for source in source_owners):
            raise TypeError("Recovery sources must be local resource owners.")
        if len({source.owner for source in source_owners}) != len(source_owners):
            raise ResourceOwnerConflict("Recovery source registration is ambiguous.")
        work = _PreparationWork()
        return await self._owned(
            lambda: self._reconcile(source_owners, work), *source_owners, work=work
        )

    async def _reconcile(self, source_owners, work: _PreparationWork) -> tuple[str, ...]:
        """Reconcile pending/uncertain acquisitions after restart or lost ACK.

        Reconciliation adopts the original operation identity and command. It
        never creates a successor key and leaves unresolved material fenced.
        """
        pending: list[tuple[str, str, tuple[str, ...], str]] = []
        unsettled = []
        releasing = []
        transfers = []
        async with self._journal.transaction() as journal:
            operations = journal["operations"]
            assert isinstance(operations, dict)
            for digest, record in operations.items():
                if (
                    record.get("stage") == "released"
                    and record.get("responsibility_settled") is not True
                ):
                    unsettled.append(digest)
                if isinstance(record, dict) and record.get("stage") == "releasing":
                    releasing.append(record["owned_receipt"])
                if isinstance(record, dict) and record.get("stage") in {
                    "pending",
                    "uncertain",
                    "cleaning",
                }:
                    command_json = record.get("command")
                    if isinstance(command_json, str):
                        raw_material_ids = record.get("material_ids", ())
                        material_ids = (
                            tuple(raw_material_ids)
                            if isinstance(raw_material_ids, list)
                            and all(isinstance(value, str) for value in raw_material_ids)
                            else ()
                        )
                        pending.append((digest, command_json, material_ids, record["stage"]))
            transfers = [
                dict(record)
                for record in journal["transfers"].values()
                if record.get("stage")
                in {"pending", "uncertain", "releasing", "cleaning", "accepted"}
                or (
                    record.get("stage") == "released"
                    and record.get("responsibility_settled") is not True
                )
            ]
        recovered: list[str] = []
        for digest in unsettled:
            await self._settle_responsibility(digest)
            recovered.append(digest)
        for digest, command_json, material_ids, stage in pending:
            if stage == "cleaning":
                await self._release_pending_materials(digest, material_ids)
                recovered.append(digest)
                continue
            try:
                command = ResourceAcquisitionCommand.model_validate_json(command_json)
                if resource_operation_digest(command) != digest:
                    raise ResourceOwnerConflict("Pending operation identity changed.")
                if work is not None:
                    work.check()
                try:
                    await self._validate_selection(command.intent.selector)
                except (ResourceOwnerConflict, FileNotFoundError) as error:
                    # A positively changed/missing immutable publication cannot
                    # become this exact input on retry. Retire only our pins.
                    raise ResourceOwnerUnavailable(
                        "Retained source identity is unavailable."
                    ) from error
                stored = await self._require_authorization(command, None)
                await self._revalidate_preparation(command, stored, work)
                await self._acquire(command, digest, work)
            except (ResourceOwnerUnavailable, ResourceOwnerUnsupported, MandateDenied) as error:
                if work is not None and work.stop_requested():
                    await self._cleanup_stopped(command, "operations", error)
                else:
                    await self._release_pending_materials(digest, material_ids)
                recovered.append(digest)
                continue
            except (ResourceOwnerError, OSError, ValueError):
                continue
            recovered.append(digest)
        for raw_receipt in releasing:
            receipt = ResourceAcquisitionReceipt.model_validate(raw_receipt)
            await self._release(receipt)
            recovered.append(receipt.operation_digest)
        sources = {source.owner: source for source in source_owners}
        for record in transfers:
            command = ResourceTransferCommand.model_validate_json(record["command"])
            if record["stage"] == "accepted":
                if command.source not in sources:
                    continue
                # Recovery consumes destination-owned durable acceptance, never
                # a caller-shaped receipt or a renewed source release grant.
                await sources[command.source]._release_transferred_source(
                    ResourceTransferReceipt.model_validate(record["accepted_receipt"]),
                    destination_owner=self,
                )
            elif record["stage"] == "released":
                await self._settle_responsibility(resource_operation_digest(command), "transfers")
            elif record["stage"] == "releasing":
                await self._release_transfer(
                    ResourceTransferReceipt.model_validate(record["accepted_receipt"])
                )
            elif record["stage"] != "cleaning" and command.source in sources:
                try:
                    await self._accept_transfer(
                        command, source_owner=sources[command.source], work=work
                    )
                except (ResourceOwnerUnavailable, ResourceOwnerUnsupported, MandateDenied) as error:
                    if work is not None and work.stop_requested():
                        await self._cleanup_stopped(command, "transfers", error)
                    else:
                        await self._release_pending_transfer(
                            resource_operation_digest(command), tuple(record["material_ids"])
                        )
            else:
                raw_material_ids = record.get("material_ids", ())
                material_ids = (
                    tuple(raw_material_ids)
                    if isinstance(raw_material_ids, list)
                    and all(isinstance(value, str) for value in raw_material_ids)
                    else ()
                )
                await self._release_pending_transfer(
                    resource_operation_digest(command), material_ids
                )
            recovered.append(resource_operation_digest(command))
        return tuple(recovered)

    async def release_transferred_source(
        self,
        transfer: ResourceTransferReceipt,
        *,
        destination_owner: LocalArtifactResourceOwner,
    ) -> ResourceAcquisitionReceipt:
        if not isinstance(destination_owner, LocalArtifactResourceOwner):
            raise TypeError("destination_owner must be a local resource owner.")
        transfer = prepare_contract(ResourceTransferReceipt, transfer, redactor=self._redactor)

        async def admitted():
            await self._preparation_reader.authorize_release(
                transfer.command.intent.receipt.command
            )
            return await self._release_transferred_source(
                transfer, destination_owner=destination_owner
            )

        return await self._owned(admitted, destination_owner)

    async def _release_transferred_source(
        self,
        transfer: ResourceTransferReceipt,
        *,
        destination_owner: LocalArtifactResourceOwner,
    ) -> ResourceAcquisitionReceipt:
        if not isinstance(destination_owner, LocalArtifactResourceOwner):
            raise TypeError("destination_owner must be a registered local resource owner.")
        if transfer.stage != "accepted":
            raise ResourceOwnerUnavailable("Destination transfer is not accepted.")
        lookup = await destination_owner._read_transfer_for_cleanup(transfer.command)
        if (
            not isinstance(lookup, ExactMatch)
            or lookup.receipt != transfer
            or lookup.receipt.stage != "accepted"
        ):
            raise ResourceOwnerUnavailable("Destination acceptance was not authenticated.")
        command = transfer.command
        source_receipt = command.intent.receipt
        await self._release(source_receipt)
        return source_receipt.model_copy(update={"stage": "transferred"})


class _ResourceSettlementReader(PermitSettlementReader):
    """Owner-internal evidence, only read under the resource mutation fence."""

    def __init__(self, owner, digest, family="operations"):
        self._resource_owner = owner
        self._digest = digest
        self._family = family

    @property
    def owner(self):
        return self._resource_owner.owner

    async def lookup(self, expected):
        async with self._resource_owner._journal.transaction() as journal:
            record = journal[self._family].get(self._digest)
            authorization = journal["authorizations"].get(self._digest)
            if not isinstance(record, dict) or not isinstance(authorization, dict):
                return ExactUnavailable()
            preparation = ResourcePreparationReceipt.model_validate(authorization["receipt"])
            if preparation.permit != expected or record.get("stage") != "released":
                return ExactUnavailable()
        return ExactMatch[ReceivingSettlementReceipt](
            receipt=ReceivingSettlementReceipt(
                expected=expected,
                receiving_owner=self.owner,
                receipt_id="resource-settled-" + self._digest,
                # Released is a permanent admission tombstone, and is published
                # only after all planned pins have been cleaned under the owner
                # fence. Both facts hold independently of registration ACKs.
                outcome="quiescent",
                admission_excluded=True,
            )
        )


__all__ = [
    "LocalArtifactResourceOwner",
    "MandateResourcePreparationReader",
    "ResourceAcquisitionCommand",
    "ResourceAcquisitionIntent",
    "ResourceAcquisitionReceipt",
    "ResourceOwnerConflict",
    "ResourceOwnerError",
    "ResourceOwnerUnavailable",
    "ResourceOwnerUnsupported",
    "ResourcePreparationAuthorization",
    "ResourcePreparationLease",
    "ResourcePreparationReader",
    "ResourcePreparationReceipt",
    "ResourceTransferCommand",
    "ResourceTransferIntent",
    "ResourceTransferReceipt",
    "resource_operation_digest",
]
