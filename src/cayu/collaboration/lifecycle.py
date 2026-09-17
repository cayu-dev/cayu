"""Namespace and participant lifecycle values; none of these values grant execution."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StrictBool, StrictInt, model_validator

from cayu.collaboration._contracts import (
    ContractValue,
    ExpectedOperation,
    Generation,
    Identifier,
    OperationRef,
    OwnerRef,
)
from cayu.collaboration.participants import (
    CollaborationLimits,
    Counter,
    ParticipantEvent,
    ParticipantRef,
    ParticipantSnapshot,
    VersionOne,
)

LifecycleState = Literal["active", "draining", "disabled", "retired"]
NamespaceState = Literal["open", "sealed", "retired"]


class NamespaceRef(ContractValue):
    owner: OwnerRef
    namespace_incarnation: Identifier
    generation: Generation

    def operation(self, caller_key: str) -> OperationRef:
        return OperationRef(
            application_scope=self.owner.application_scope,
            namespace_incarnation=self.namespace_incarnation,
            generation=self.generation,
            caller_key=caller_key,
        )


class NamespaceSnapshot(ContractValue):
    reference: NamespaceRef
    revision: Generation
    state: NamespaceState
    outstanding_obligations: Counter
    content: Literal["retained", "partial"] = "retained"

    @model_validator(mode="after")
    def retired_is_settled(self) -> NamespaceSnapshot:
        if self.state == "retired" and self.outstanding_obligations:
            raise ValueError("Retired namespace cannot retain unsettled obligations.")
        if self.content == "partial" and self.state != "retired":
            raise ValueError("Only retired namespace content may be pruned.")
        return self


class NamespaceInspection(ContractValue):
    """Current state, distinct from immutable bootstrap and control receipts."""

    current: NamespaceSnapshot
    retired_through: Counter
    pruned_through: Counter
    retention_revision: Generation
    retained_generations: Generation

    @model_validator(mode="after")
    def ordered_floors(self) -> NamespaceInspection:
        if not self.pruned_through <= self.retired_through < self.current.reference.generation:
            raise ValueError("Namespace retention frontiers conflict.")
        if self.current.state == "retired":
            raise ValueError("Current namespace must retain a control authority.")
        return self


class NamespaceSeal(ContractValue):
    operation: OperationRef
    namespace: NamespaceRef
    expected_revision: Generation
    kind: Literal["namespace_seal"] = "namespace_seal"


class NamespaceRotate(ContractValue):
    operation: OperationRef
    namespace: NamespaceRef
    expected_revision: Generation
    kind: Literal["namespace_rotate"] = "namespace_rotate"


class NamespaceRetire(ContractValue):
    """Retire the next contiguous generation from a later control generation."""

    operation: OperationRef
    namespace: NamespaceRef
    expected_revision: Generation
    expected_retired_through: Counter
    kind: Literal["namespace_retire"] = "namespace_retire"


class NamespacePrune(ContractValue):
    """One exact bounded batch; a fresh key is required for a later batch."""

    operation: OperationRef
    namespace: NamespaceRef
    expected_retention_revision: Generation
    max_records: Annotated[StrictInt, Field(ge=1, le=32)] = 32
    kind: Literal["namespace_prune"] = "namespace_prune"


class ParticipantLifecycleChange(ContractValue):
    operation: OperationRef
    participant: ParticipantRef
    expected_lifecycle_revision: Generation
    state: LifecycleState
    control_policy: Literal["settle_registered"] = "settle_registered"
    kind: Literal["participant_lifecycle"] = "participant_lifecycle"


LifecycleRequest = (
    NamespaceSeal | NamespaceRotate | NamespaceRetire | NamespacePrune | ParticipantLifecycleChange
)


class LifecycleIntent(ContractValue):
    request: LifecycleRequest
    limits: CollaborationLimits


class LifecycleCommand(ExpectedOperation[LifecycleIntent]):
    kind: Literal[
        "namespace_seal",
        "namespace_rotate",
        "namespace_retire",
        "namespace_prune",
        "participant_lifecycle",
    ]
    schema_version: VersionOne = 1
    mode: Literal["lifecycle"] = "lifecycle"
    receipt_stage: Literal["committed"] = "committed"

    @model_validator(mode="after")
    def exact_control_authority(self) -> LifecycleCommand:
        request = self.intent.request
        if (
            self.operation != request.operation
            or self.kind != request.kind
            or self.source != self.destination
            or self.initiator.issuer != self.source
            or any(
                value is not None
                for value in (
                    self.initiator.participant,
                    self.initiator.mandate,
                    self.initiator.invocation_id,
                    self.initiator.interaction_id,
                )
            )
        ):
            raise ValueError("Lifecycle control authority conflicts.")
        if isinstance(request, ParticipantLifecycleChange):
            if request.participant.owner != self.source:
                raise ValueError("Participant belongs to another lifecycle owner.")
            return self
        target = request.namespace
        if (
            target.owner != self.source
            or target.namespace_incarnation != self.operation.namespace_incarnation
        ):
            raise ValueError("Namespace belongs to another lifecycle owner.")
        if isinstance(request, (NamespaceSeal, NamespaceRotate)):
            if target.generation != self.operation.generation:
                raise ValueError("Namespace election requires its exact generation.")
        elif target.generation >= self.operation.generation:
            raise ValueError("Namespace maintenance requires a later control generation.")
        if (
            isinstance(request, NamespaceRetire)
            and target.generation != request.expected_retired_through + 1
        ):
            raise ValueError("Namespace retirement cannot skip a generation.")
        return self


class NamespaceRetirementEvidence(ContractValue):
    """Admission rejection evidence, never a replacement for an exact receipt."""

    namespace: NamespaceRef
    retired_through: Generation
    pruned_through: Counter
    content: Literal["retained", "partial", "pruned"]

    @model_validator(mode="after")
    def proven_floor(self) -> NamespaceRetirementEvidence:
        if (
            self.namespace.generation > self.retired_through
            or self.pruned_through > self.retired_through
            or (self.content == "pruned") != (self.namespace.generation <= self.pruned_through)
        ):
            raise ValueError("Retirement evidence does not cover the requested generation.")
        return self


class LifecycleReceipt(ContractValue):
    """Immutable accepted transition, not a claim about later live state."""

    expected: LifecycleCommand
    namespace: NamespaceSnapshot | None = None
    successor: NamespaceSnapshot | None = None
    participant: ParticipantSnapshot | None = None
    event: ParticipantEvent
    removed_records: Counter = 0
    pruned_through: Counter | None = None
    retention_revision: Generation | None = None
    complete: StrictBool | None = None

    @model_validator(mode="after")
    def exact_transition(self) -> LifecycleReceipt:
        request = self.expected.intent.request
        event_types = {
            "namespace_seal": "namespace_sealed",
            "namespace_rotate": "namespace_rotated",
            "namespace_retire": "namespace_retired",
            "namespace_prune": "namespace_pruned",
            "participant_lifecycle": "participant_lifecycle_changed",
        }
        if (
            self.event.operation != request.operation
            or self.event.type != event_types[request.kind]
        ):
            raise ValueError("Lifecycle receipt event conflicts.")
        if isinstance(request, ParticipantLifecycleChange):
            if (
                self.namespace is not None
                or self.successor is not None
                or self.participant is None
                or self.participant.reference != request.participant
                or self.participant.lifecycle_revision != request.expected_lifecycle_revision + 1
                or self.participant.lifecycle != request.state
                or self.participant.control_policy != request.control_policy
                or self.event.participants != (request.participant,)
            ):
                raise ValueError("Participant lifecycle receipt conflicts.")
        else:
            if (
                self.participant is not None
                or self.namespace is None
                or self.namespace.reference != request.namespace
                or self.event.participants
            ):
                raise ValueError("Namespace receipt authority conflicts.")
            if isinstance(request, (NamespaceSeal, NamespaceRotate, NamespaceRetire)):
                state = "retired" if isinstance(request, NamespaceRetire) else "sealed"
                if (
                    self.namespace.state != state
                    or self.namespace.revision != request.expected_revision + 1
                ):
                    raise ValueError("Namespace receipt transition conflicts.")
            if isinstance(request, NamespaceRotate):
                if (
                    self.successor is None
                    or self.successor.reference
                    != NamespaceRef(
                        owner=request.namespace.owner,
                        namespace_incarnation=request.namespace.namespace_incarnation,
                        generation=request.namespace.generation + 1,
                    )
                    or self.successor.state != "open"
                    or self.successor.revision != 1
                    or self.successor.outstanding_obligations
                ):
                    raise ValueError("Namespace rotation successor conflicts.")
            elif self.successor is not None:
                raise ValueError("Only rotation can elect a successor.")
        if not isinstance(request, NamespacePrune) and self.removed_records:
            raise ValueError("Only pruning can remove retained records.")
        if isinstance(request, NamespacePrune):
            if (
                self.namespace is None
                or self.namespace.state != "retired"
                or self.retention_revision != request.expected_retention_revision + 1
                or self.complete is None
                or self.pruned_through != request.namespace.generation - int(not self.complete)
                or self.removed_records > request.max_records
            ):
                raise ValueError("Pruning receipt frontier conflicts.")
        elif any(
            value is not None
            for value in (self.pruned_through, self.retention_revision, self.complete)
        ):
            raise ValueError("Only pruning can publish a retention frontier.")
        return self


class CollaborationNamespaceRetired(RuntimeError):
    """Fresh admission is forbidden; inspect namespace evidence separately."""


class CollaborationHistoryUnavailable(RuntimeError):
    """A retained-history cursor no longer proves complete traversal."""
