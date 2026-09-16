"""Participant administration values. References are identities, not grants."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, StrictInt, model_validator

from cayu.collaboration._contracts import (
    ContractValue,
    ExpectedOperation,
    Generation,
    Identifier,
    OperationRef,
    OwnerRef,
)

Counter = Annotated[StrictInt, Field(ge=0, le=2**53 - 1)]


def _version_one(value: object) -> int:
    if type(value) is not int or value != 1:
        raise ValueError("Expected integer version one.")
    return value


VersionOne = Annotated[Literal[1], BeforeValidator(_version_one)]


class CollaborationNotInitialized(RuntimeError):
    """Explicit collaboration initialization has not completed."""


class CollaborationUnavailable(RuntimeError):
    """Authoritative collaboration evidence or configuration is unavailable."""


class CollaborationCapacityExceeded(RuntimeError):
    """New admission would exceed retained-state capacity."""


class CollaborationLimits(ContractValue):
    """Finite, immutable bootstrap limits; reserved controls cannot be borrowed."""

    participants: Generation
    aliases: Generation
    operations: Generation
    events: Generation
    retained_bytes: Generation
    control_operations: Generation
    control_events: Generation
    control_bytes: Generation
    namespaces: Generation
    generations: Generation
    obligations: Generation

    @model_validator(mode="after")
    def control_reserves_fit(self) -> CollaborationLimits:
        if (
            self.control_operations >= self.operations
            or self.control_events >= self.events
            or self.control_bytes >= self.retained_bytes
        ):
            raise ValueError("Control reserves must leave ordinary admission capacity.")
        return self


class CollaborationBootstrap(ContractValue):
    application_scope: Identifier
    provisioning_scope: Identifier
    owner_name: Identifier
    limits: CollaborationLimits
    version: VersionOne = 1


class CollaborationInitialization(ContractValue):
    binding: CollaborationBootstrap
    owner: OwnerRef
    namespace_incarnation: Identifier
    generation: VersionOne = 1

    @model_validator(mode="after")
    def bound_owner(self) -> CollaborationInitialization:
        if (self.owner.application_scope, self.owner.owner_id) != (
            self.binding.application_scope,
            self.binding.owner_name,
        ):
            raise ValueError("Bootstrap owner does not match its binding.")
        return self

    def operation(self, caller_key: str) -> OperationRef:
        return OperationRef(
            application_scope=self.binding.application_scope,
            namespace_incarnation=self.namespace_incarnation,
            generation=self.generation,
            caller_key=caller_key,
        )


class ParticipantRef(ContractValue):
    owner: OwnerRef
    participant_id: Identifier
    incarnation: Identifier


class ParticipantConfigurationRef(ContractValue):
    """Application-registered version, not executable serialized configuration."""

    name: Identifier
    version: Generation


class ParticipantConfiguration(ContractValue):
    definition: ParticipantConfigurationRef
    routing: ParticipantConfigurationRef
    admission: ParticipantConfigurationRef


class ParticipantSnapshot(ContractValue):
    reference: ParticipantRef
    configuration: ParticipantConfiguration
    configuration_revision: Generation
    lifecycle: Literal["active"] = "active"
    lifecycle_revision: VersionOne = 1
    admission_generation: VersionOne = 1


class ParticipantAlias(ContractValue):
    alias: Identifier
    target: ParticipantRef
    revision: Generation


class ParticipantCreate(ContractValue):
    operation: OperationRef
    configuration: ParticipantConfiguration
    alias: Identifier | None = None
    expected_alias_revision: Counter = 0
    kind: Literal["create"] = "create"

    @model_validator(mode="after")
    def absent_alias(self) -> ParticipantCreate:
        if self.alias is None and self.expected_alias_revision != 0:
            raise ValueError("An absent alias cannot carry a revision expectation.")
        return self


class ParticipantConfigure(ContractValue):
    operation: OperationRef
    participant: ParticipantRef
    expected_configuration_revision: Generation
    configuration: ParticipantConfiguration
    kind: Literal["configure"] = "configure"


class ParticipantAliasChange(ContractValue):
    operation: OperationRef
    alias: Identifier
    expected_alias_revision: Counter
    expected_target: ParticipantRef | None
    target: ParticipantRef | None
    kind: Literal["alias"] = "alias"

    @model_validator(mode="after")
    def changes_binding(self) -> ParticipantAliasChange:
        if self.target == self.expected_target:
            raise ValueError("Alias changes must change the exact target.")
        return self


ParticipantMutation = ParticipantCreate | ParticipantConfigure | ParticipantAliasChange


class ParticipantIntent(ContractValue):
    request: ParticipantMutation
    limits: CollaborationLimits


class ParticipantCommand(ExpectedOperation[ParticipantIntent]):
    """Identity-family controls are closed schema, not caller-authored text."""

    kind: Literal["create", "configure", "alias"]
    schema_version: VersionOne = 1
    mode: Literal["identity"] = "identity"
    receipt_stage: Literal["committed"] = "committed"


class ParticipantEvent(ContractValue):
    id: Identifier
    sequence: Generation
    operation: OperationRef | None
    type: Literal["initialized", "created", "configured", "alias_changed"]
    participants: tuple[ParticipantRef, ...] = Field(max_length=2)

    @model_validator(mode="after")
    def event_shape(self) -> ParticipantEvent:
        if self.type == "initialized":
            if self.operation is not None or self.participants or self.sequence != 1:
                raise ValueError("Initialization event evidence conflicts.")
        elif self.operation is None or not self.participants or self.sequence < 2:
            raise ValueError("Participant event evidence is incomplete.")
        elif any(
            ref.owner.application_scope != self.operation.application_scope
            for ref in self.participants
        ):
            raise ValueError("Participant event scope conflicts.")
        return self


class ParticipantReceipt(ContractValue):
    expected: ParticipantCommand
    participants: tuple[ParticipantSnapshot, ...] = Field(max_length=2)
    alias: ParticipantAlias | None
    alias_revision: Counter
    event: ParticipantEvent

    @model_validator(mode="after")
    def exact_result(self) -> ParticipantReceipt:
        request = self.expected.intent.request
        refs = tuple(p.reference for p in self.participants)
        if (
            self.expected.operation != request.operation
            or self.event.operation != request.operation
            or self.event.participants != refs
            or any(ref.owner != self.expected.source for ref in refs)
            or self.expected.kind != request.kind
        ):
            raise ValueError("Receipt authority conflicts.")
        if isinstance(request, (ParticipantCreate, ParticipantConfigure)):
            if len(self.participants) != 1:
                raise ValueError("Receipt must identify one participant.")
            snapshot = self.participants[0]
            revision = (
                1
                if isinstance(request, ParticipantCreate)
                else request.expected_configuration_revision + 1
            )
            if (
                snapshot.configuration != request.configuration
                or snapshot.configuration_revision != revision
            ):
                raise ValueError("Receipt configuration conflicts.")
            if isinstance(request, ParticipantConfigure):
                if (
                    refs != (request.participant,)
                    or self.alias is not None
                    or self.event.type != "configured"
                ):
                    raise ValueError("Configuration receipt conflicts.")
                return self
            if self.event.type != "created":
                raise ValueError("Creation event conflicts.")
            target = snapshot.reference if request.alias is not None else None
        else:
            if (
                refs
                != tuple(
                    ref for ref in (request.expected_target, request.target) if ref is not None
                )
                or self.event.type != "alias_changed"
            ):
                raise ValueError("Alias receipt participants conflict.")
            target = request.target
        if target is None:
            if self.alias is not None:
                raise ValueError("Receipt unexpectedly binds an alias.")
        elif self.alias is None or (self.alias.alias, self.alias.target, self.alias.revision) != (
            request.alias,
            target,
            self.alias_revision,
        ):
            raise ValueError("Receipt alias conflicts.")
        if request.alias is not None and self.alias_revision != request.expected_alias_revision + 1:
            raise ValueError("Receipt alias revision conflicts.")
        return self


class ParticipantInspection(ContractValue):
    participant: ParticipantSnapshot
    alias_revision: Counter


class ParticipantCursor(ContractValue):
    scope: Identifier
    principal: Identifier
    allowed: tuple[ParticipantRef, ...] | None
    after_id: Identifier


class ParticipantPage(ContractValue):
    participants: tuple[ParticipantSnapshot, ...] = Field(max_length=64)
    next_cursor: ParticipantCursor | None


class ParticipantEventCursor(ContractValue):
    scope: Identifier
    principal: Identifier
    allowed: tuple[ParticipantRef, ...] | None
    after_sequence: Counter


class ParticipantEventPage(ContractValue):
    events: tuple[ParticipantEvent, ...] = Field(max_length=64)
    next_cursor: ParticipantEventCursor | None
