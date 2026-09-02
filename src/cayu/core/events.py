from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from cayu._validation import (
    copy_durable_json_value,
    require_durable_clean_nonblank,
)

_CUSTOM_EVENT_TYPE_RE = re.compile(r"^custom\.[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$")
EVENT_ID_MAX_CHARS = 512


def _empty_runtime_authority() -> frozenset[tuple[str, str]]:
    """Return a precisely typed empty private-authority registry."""

    return frozenset()


def _empty_runtime_nested_authority() -> frozenset[tuple[tuple[str, ...], str]]:
    """Return a precisely typed empty nested-authority registry."""

    return frozenset()


def _validate_custom_event_type(event_type: object) -> str:
    if type(event_type) is not str or _CUSTOM_EVENT_TYPE_RE.fullmatch(event_type) is None:
        raise ValueError(
            "Custom event types must use non-empty dot-separated segments "
            "in the 'custom.' namespace."
        )
    return event_type


def validate_public_custom_event_type(event_type: object) -> str:
    """Return one caller-owned custom event type outside Cayu's namespace."""

    event_type = _validate_custom_event_type(event_type)
    if event_type.startswith("custom.cayu."):
        raise ValueError("The custom.cayu. namespace is reserved for cayu internals.")
    return event_type


class EventType(StrEnum):
    SERVER_MUTATION_ACCEPTED = "server.mutation.accepted"

    SESSION_STARTED = "session.started"
    SESSION_RESUMED = "session.resumed"
    SESSION_COMPLETED = "session.completed"
    SESSION_FAILED = "session.failed"
    SESSION_INTERRUPTED = "session.interrupted"
    SESSION_INTERRUPTION_CASCADE_RETRY_REQUESTED = "session.interruption_cascade_retry_requested"
    SESSION_INTERRUPTION_CASCADE_COMPLETED = "session.interruption_cascade_completed"
    SESSION_INTERRUPTION_CASCADE_FAILED = "session.interruption_cascade_failed"
    SESSION_AWAITING_USER_INPUT = "session.awaiting_user_input"
    SESSION_CHECKPOINTED = "session.checkpointed"
    SESSION_FORKED = "session.forked"
    SESSION_LIMIT_REACHED = "session.limit_reached"
    SESSION_MESSAGE_QUEUED = "session.message.queued"
    SESSION_MESSAGE_DELIVERED = "session.message.delivered"
    SESSION_MODEL_SWITCHED = "session.model.switched"
    SESSION_EXECUTION_PROFILE_DECIDED = "session.execution_profile.decided"
    SESSION_EXECUTION_PROFILE_REJECTED = "session.execution_profile.rejected"
    SESSION_RUN_FENCED = "session.run_fenced"
    TURN_COMPLETED = "turn.completed"

    INTERACTION_STARTED = "interaction.started"
    INTERACTION_RESUMED = "interaction.resumed"
    INTERACTION_PAUSED = "interaction.paused"
    INTERACTION_COMPLETED = "interaction.completed"
    INTERACTION_FAILED = "interaction.failed"
    INTERACTION_INTERRUPTED = "interaction.interrupted"

    BUDGET_CHECKED = "budget.checked"
    BUDGET_LIMIT_REACHED = "budget.limit_reached"
    BUDGET_RESERVED = "budget.reserved"
    BUDGET_RECONCILED = "budget.reconciled"
    BUDGET_RESERVATION_FAILED = "budget.reservation_failed"
    BUDGET_RESERVATION_RELEASED = "budget.reservation_released"

    CREDENTIAL_PROXY_CHECKED = "credential.proxy.checked"
    CREDENTIAL_MODE_SELECTED = "credential.mode.selected"

    EGRESS_GRANT_MINTED = "egress.grant.minted"
    EGRESS_GRANT_REVOKED = "egress.grant.revoked"
    EGRESS_REQUEST_AUTHORIZED = "egress.request.authorized"
    EGRESS_REQUEST_DENIED = "egress.request.denied"
    EGRESS_AUTHORITY_REQUESTED = "egress.authority.requested"
    EGRESS_AUTHORITY_AUTHORIZED = "egress.authority.authorized"
    EGRESS_AUTHORITY_INSTALLING = "egress.authority.installing"
    EGRESS_AUTHORITY_ACTIVATED = "egress.authority.activated"
    EGRESS_AUTHORITY_REFUSED = "egress.authority.refused"
    EGRESS_AUTHORITY_AMBIGUOUS = "egress.authority.ambiguous"

    MCP_MANIFEST_CHECKED = "mcp.manifest.checked"
    MCP_MANIFEST_BLOCKED = "mcp.manifest.blocked"

    TASK_CREATED = "task.created"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_CANCELLED = "task.cancelled"
    TASK_INTERRUPTED_HANDOFF = "task.interrupted_handoff"
    TASK_COMPLETION_RESULT_RESOLVED = "task.completion_result.resolved"

    FORK_GROUP_CREATED = "fork_group.created"
    FORK_GROUP_BRANCHES_RUNNING = "fork_group.branches_running"
    FORK_GROUP_AWAITING_EVALUATION = "fork_group.awaiting_evaluation"
    FORK_GROUP_COMPLETED = "fork_group.completed"
    FORK_GROUP_FAILED = "fork_group.failed"

    MODEL_STARTED = "model.started"
    MODEL_TEXT_DELTA = "model.text.delta"
    MODEL_THINKING_DELTA = "model.thinking.delta"
    MODEL_HOSTED_TOOL_CALL = "model.hosted_tool_call"
    MODEL_CITATION = "model.citation"
    MODEL_COMPLETED = "model.completed"
    MODEL_ERROR = "model.error"
    MODEL_RETRY = "model.retry"
    MODEL_ATTEMPT_DISCARDED = "model.attempt_discarded"
    PROVIDER_OPERATION_STARTING = "provider.operation.starting"
    PROVIDER_OPERATION_STARTED = "provider.operation.started"
    PROVIDER_OPERATION_PROGRESS = "provider.operation.progress"
    PROVIDER_OPERATION_CANCEL_REQUESTED = "provider.operation.cancel_requested"
    PROVIDER_OPERATION_CANCEL_RESOLVED = "provider.operation.cancel_resolved"
    PROVIDER_OPERATION_RECONNECT_SCHEDULED = "provider.operation.reconnect_scheduled"
    PROVIDER_OPERATION_RECONNECT_STARTED = "provider.operation.reconnect_started"
    PROVIDER_OPERATION_RECOVERY_REQUIRED = "provider.operation.recovery_required"
    PROVIDER_OPERATION_RESOLVED = "provider.operation.resolved"
    PROVIDER_OPERATION_RECONCILED = "provider.operation.reconciled"
    REQUEST_FOOTPRINT_RECORDED = "request.footprint.recorded"
    TOOL_EXPOSURE_RECORDED = "tool.exposure.recorded"
    TARGETED_TOOL_GRANT_ISSUED = "tool.grant.issued"
    TARGETED_TOOL_GRANT_REUSED = "tool.grant.reused"
    TARGETED_TOOL_GRANT_RECONSTRUCTED = "tool.grant.reconstructed"
    TARGETED_TOOL_GRANT_EXPIRED = "tool.grant.expired"
    TARGETED_TOOL_GRANT_REVOKED = "tool.grant.revoked"
    TARGETED_TOOL_GRANT_FORK_RESET = "tool.grant.fork_reset"
    TARGETED_TOOL_REFERENCE_CONSUMED = "tool.reference.consumed"
    TARGETED_TOOL_REFERENCE_REJOINED = "tool.reference.rejoined"
    TARGETED_TOOL_REFERENCE_REJECTED = "tool.reference.rejected"

    STRUCTURED_OUTPUT_VALIDATED = "structured_output.validated"
    STRUCTURED_OUTPUT_VALIDATING = "structured_output.validating"
    STRUCTURED_OUTPUT_FAILED = "structured_output.failed"
    STRUCTURED_OUTPUT_RETRY = "structured_output.retry"

    CONTEXT_COMPACTION_STARTED = "context.compaction.started"
    CONTEXT_COMPACTION_COMPLETED = "context.compaction.completed"
    CONTEXT_COMPACTION_FAILED = "context.compaction.failed"
    CONTEXT_COUNTED = "context.counted"
    CONTEXT_COUNT_FAILED = "context.count.failed"
    CONTEXT_COUNT_RECONCILED = "context.count.reconciled"
    CONTEXT_PRESSURE_ESTIMATED = "context.pressure.estimated"
    CONTEXT_PRESSURE_RECONCILED = "context.pressure.reconciled"
    CONTEXT_OVERFLOW_DETECTED = "context.overflow.detected"
    CONTEXT_OVERFLOW_RECOVERING = "context.overflow.recovering"
    CONTEXT_OVERFLOW_FAILED = "context.overflow.failed"

    AUTOMATIC_RECALL_STARTED = "memory.recall.started"
    AUTOMATIC_RECALL_COMPLETED = "memory.recall.completed"
    AUTOMATIC_RECALL_FAILED = "memory.recall.failed"
    AUTOMATIC_RECALL_ADMITTED = "memory.recall.admitted"

    ENVIRONMENT_BINDING_STARTED = "environment.binding.started"
    ENVIRONMENT_BINDING_COMPLETED = "environment.binding.completed"
    ENVIRONMENT_BINDING_FAILED = "environment.binding.failed"
    ENVIRONMENT_BINDING_FINALIZE_STARTED = "environment.binding.finalize_started"
    ENVIRONMENT_BINDING_FINALIZE_COMPLETED = "environment.binding.finalize_completed"
    ENVIRONMENT_BINDING_FINALIZE_FAILED = "environment.binding.finalize_failed"
    ENVIRONMENT_FACTORY_STARTED = "environment.factory.started"
    ENVIRONMENT_FACTORY_COMPLETED = "environment.factory.completed"
    ENVIRONMENT_FACTORY_FAILED = "environment.factory.failed"

    WORKSPACE_REVISION_OBSERVED = "workspace.revision.observed"
    WORKSPACE_MUTATION_RECORDED = "workspace.mutation.recorded"
    WORKSPACE_OBSERVATION_FINALIZED = "workspace.observation.finalized"

    HOOK_STARTED = "hook.started"
    HOOK_COMPLETED = "hook.completed"
    HOOK_FAILED = "hook.failed"

    TOOL_CALL_STARTED = "tool.call.started"
    TOOL_CALL_COMPLETED = "tool.call.completed"
    TOOL_CALL_FAILED = "tool.call.failed"
    TOOL_CALL_BLOCKED = "tool.call.blocked"
    TOOL_CALL_APPROVAL_REQUESTED = "tool.call.approval_requested"
    TOOL_CALL_APPROVED = "tool.call.approved"
    TOOL_CALL_APPROVAL_DENIED = "tool.call.approval_denied"
    TOOL_CALL_APPROVAL_EXPIRED = "tool.call.approval_expired"

    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_STEP_STARTED = "workflow.step.started"
    WORKFLOW_STEP_COMPLETED = "workflow.step.completed"
    WORKFLOW_COMPLETED = "workflow.completed"

    MEMORY_SEARCH = "memory.search"
    RUNNER_EXEC_STARTED = "runner.exec.started"
    RUNNER_EXEC_COMPLETED = "runner.exec.completed"

    RUNTIME_SINK_FAILED = "runtime.sink.failed"
    RUNTIME_INTERACTION_TRANSITION_ACKNOWLEDGEMENT_FAILED = (
        "runtime.interaction_transition.acknowledgement_failed"
    )


class Event(BaseModel):
    """Append-only runtime event.

    Events are the common language between terminal output, dashboard views,
    persistent sessions, webhooks, and hosted-platform adapters.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    # `EventType` is a StrEnum. A known type validates back to the enum member here, but an event
    # deserialized from JSON elsewhere (webhook / SSE / JSONL, or `payload["type"]`) carries a plain
    # str — so ALWAYS compare event types with `==`, never `is` (identity fails on the str form).
    # (`| str` also admits `custom.*` types, which stay plain strings.)
    type: EventType | str
    session_id: str
    interaction_id: str | None = None
    id: str = Field(default_factory=lambda: str(uuid4()), max_length=EVENT_ID_MAX_CHARS)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    agent_name: str | None = None
    environment_name: str | None = None
    workflow_name: str | None = None
    tool_name: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    _id_origin: Literal["caller", "runtime"] = PrivateAttr(default="caller")
    _runtime_generated_id: str | None = PrivateAttr(default=None)
    _runtime_payload_authority: frozenset[tuple[str, str]] = PrivateAttr(
        default_factory=_empty_runtime_authority
    )
    _runtime_nested_payload_authority: frozenset[tuple[tuple[str, ...], str]] = PrivateAttr(
        default_factory=_empty_runtime_nested_authority
    )
    _runtime_envelope_authority: frozenset[tuple[str, str]] = PrivateAttr(
        default_factory=_empty_runtime_authority
    )
    _durable_sequence: int | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        del __context
        self._id_origin = "caller" if "id" in self.model_fields_set else "runtime"
        self._runtime_generated_id = self.id if self._id_origin == "runtime" else None

    def __eq__(self, other: object) -> bool:
        """Compare only public durable fields, never private projection metadata."""

        if type(other) is not Event:
            return NotImplemented
        return (
            self.type,
            self.session_id,
            self.interaction_id,
            self.id,
            self.timestamp,
            self.agent_name,
            self.environment_name,
            self.workflow_name,
            self.tool_name,
            self.payload,
        ) == (
            other.type,
            other.session_id,
            other.interaction_id,
            other.id,
            other.timestamp,
            other.agent_name,
            other.environment_name,
            other.workflow_name,
            other.tool_name,
            other.payload,
        )

    @field_validator("payload", mode="before")
    @classmethod
    def copy_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_value(value, "payload")

    @field_validator("session_id", "id")
    @classmethod
    def validate_nonblank_ids(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator(
        "interaction_id",
        "agent_name",
        "environment_name",
        "workflow_name",
        "tool_name",
    )
    @classmethod
    def validate_optional_nonblank_names(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: EventType | str) -> EventType | str:
        if isinstance(value, EventType):
            return value

        try:
            return EventType(value)
        except ValueError:
            pass

        return _validate_custom_event_type(value)


def copy_event(event: Event) -> Event:
    if type(event) is not Event:
        raise TypeError("Events must be Event instances.")
    copied = Event(
        type=event.type,
        session_id=event.session_id,
        interaction_id=event.interaction_id,
        id=event.id,
        timestamp=event.timestamp,
        agent_name=event.agent_name,
        environment_name=event.environment_name,
        workflow_name=event.workflow_name,
        tool_name=event.tool_name,
        payload=copy_durable_json_value(event.payload, "payload"),
    )
    copied._id_origin = event._id_origin
    copied._runtime_generated_id = event._runtime_generated_id
    copied._runtime_payload_authority = event._runtime_payload_authority
    copied._runtime_nested_payload_authority = event._runtime_nested_payload_authority
    copied._runtime_envelope_authority = event._runtime_envelope_authority
    copied._durable_sequence = event._durable_sequence
    return copied


def event_id_is_runtime_generated(event: Event) -> bool:
    """Return positive in-process provenance for a default-generated event id."""

    if type(event) is not Event:
        raise TypeError("event must be an Event.")
    return (
        event._id_origin == "runtime"
        and event._runtime_generated_id is not None
        and event.id == event._runtime_generated_id
    )


def event_with_runtime_generated_id(event: Event) -> Event:
    """Copy an internally constructed event and attest its exact explicit ID.

    Deserialization and ordinary ``Event(id=...)`` construction intentionally do
    not acquire this in-process provenance. Runtime producers use this helper
    only when they themselves generated a UUID or content-addressed identity.
    """

    if type(event) is not Event:
        raise TypeError("event must be an Event.")
    copied = copy_event(event)
    copied._id_origin = "runtime"
    copied._runtime_generated_id = copied.id
    return copied


def event_with_runtime_payload_authority(
    event: Event,
    *field_names: str,
) -> Event:
    """Copy an event and attest exact runtime-produced top-level payload linkage."""

    if type(event) is not Event:
        raise TypeError("event must be an Event.")
    authority = set(event._runtime_payload_authority)
    for field_name in field_names:
        if type(field_name) is not str or not field_name or not field_name.isidentifier():
            raise ValueError("Runtime payload authority fields must be identifiers.")
        value = event.payload.get(field_name)
        if type(value) is not str or not value.strip():
            raise ValueError(
                f"event.payload.{field_name} must be a non-empty string before attestation."
            )
        authority.add((field_name, value))
    copied = copy_event(event)
    copied._runtime_payload_authority = frozenset(authority)
    return copied


def event_with_runtime_nested_payload_authority(
    event: Event,
    *paths: tuple[str, ...],
) -> Event:
    """Attest exact runtime-produced linkage below a typed payload container."""

    if type(event) is not Event:
        raise TypeError("event must be an Event.")
    authority = set(event._runtime_nested_payload_authority)

    def collect(value: Any, path: tuple[str, ...], offset: int) -> None:
        if offset == len(path):
            if value is None:
                # Typed lists may contain optional authority fields. Absence
                # needs no attestation; present siblings still retain their
                # exact runtime provenance through the wildcard path.
                return
            if type(value) is not str or not value.strip():
                raise ValueError(
                    f"event.payload.{'.'.join(path)} must contain non-empty strings "
                    "before attestation."
                )
            authority.add((path, value))
            return
        segment = path[offset]
        if segment == "*":
            if type(value) is not list:
                raise ValueError(
                    f"event.payload.{'.'.join(path[:offset])} must be a list before attestation."
                )
            for child in value:
                collect(child, path, offset + 1)
            return
        if not segment.isidentifier():
            raise ValueError("Runtime nested payload authority paths must use identifiers or '*'.")
        if type(value) is not dict or segment not in value:
            raise ValueError(
                f"event.payload.{'.'.join(path[: offset + 1])} is missing before attestation."
            )
        collect(value[segment], path, offset + 1)

    for path in paths:
        if type(path) is not tuple or len(path) < 2:
            raise ValueError(
                "Runtime nested payload authority paths require at least two segments."
            )
        collect(event.payload, path, 0)
    copied = copy_event(event)
    copied._runtime_nested_payload_authority = frozenset(authority)
    return copied


def event_with_runtime_envelope_authority(
    event: Event,
    *field_names: str,
) -> Event:
    """Attest exact envelope identities selected by the runtime.

    Ordinary construction and deserialization intentionally do not acquire this
    provenance. Runtime producers may attest only identities that have already
    crossed their caller-input boundary.
    """

    if type(event) is not Event:
        raise TypeError("event must be an Event.")
    authority = set(event._runtime_envelope_authority)
    for field_name in field_names:
        if field_name not in {"session_id", "interaction_id"}:
            raise ValueError("Runtime envelope authority supports session_id and interaction_id.")
        value = getattr(event, field_name)
        if type(value) is not str or not value.strip():
            raise ValueError(f"event.{field_name} must be a non-empty string before attestation.")
        authority.add((field_name, value))
    copied = copy_event(event)
    copied._runtime_envelope_authority = frozenset(authority)
    return copied


def event_envelope_authority_is_runtime_generated(
    event: Event,
    *,
    field_name: str,
    value: str,
) -> bool:
    """Return positive in-process provenance for one exact envelope identity."""

    if type(event) is not Event:
        raise TypeError("event must be an Event.")
    return (
        field_name in {"session_id", "interaction_id"}
        and type(value) is str
        and getattr(event, field_name) == value
        and (field_name, value) in event._runtime_envelope_authority
    )


def event_payload_authority_is_runtime_generated(
    event: Event,
    *,
    field_name: str,
    value: str,
) -> bool:
    """Return positive in-process provenance for one exact payload authority value."""

    if type(event) is not Event:
        raise TypeError("event must be an Event.")
    return (
        type(field_name) is str
        and type(value) is str
        and event.payload.get(field_name) == value
        and (field_name, value) in event._runtime_payload_authority
    )


def event_retains_runtime_payload_authority(
    event: Event,
    *,
    field_name: str,
    value: str,
) -> bool:
    """Return private provenance retained across a redacted payload projection.

    Unlike :func:`event_payload_authority_is_runtime_generated`, this helper does
    not require the public payload to still expose ``value``. It is intended only
    for runtime recovery code that already knows the exact expected authority and
    separately verifies that the observed replacement is Cayu's fixed private
    projection marker.
    """

    if type(event) is not Event:
        raise TypeError("event must be an Event.")
    return (
        type(field_name) is str
        and type(value) is str
        and (field_name, value) in event._runtime_payload_authority
    )


def event_nested_payload_authority_is_runtime_generated(
    event: Event,
    *,
    path: tuple[str, ...],
    value: str,
) -> bool:
    """Return positive provenance for one exact nested payload authority value."""

    if type(event) is not Event:
        raise TypeError("event must be an Event.")

    def contains(candidate: Any, offset: int) -> bool:
        if offset == len(path):
            return type(candidate) is str and candidate == value
        segment = path[offset]
        if segment == "*":
            return type(candidate) is list and any(
                contains(child, offset + 1) for child in candidate
            )
        return (
            type(candidate) is dict
            and segment in candidate
            and contains(candidate[segment], offset + 1)
        )

    return (
        type(path) is tuple
        and type(value) is str
        and contains(event.payload, 0)
        and (path, value) in event._runtime_nested_payload_authority
    )


def event_with_durable_sequence(event: Event, sequence: int) -> Event:
    """Copy an event and attach its private durable sequence for live projection."""

    if type(event) is not Event:
        raise TypeError("event must be an Event.")
    if type(sequence) is not int or sequence < 1:
        raise ValueError("sequence must be an integer greater than or equal to 1.")
    copied = copy_event(event)
    copied._durable_sequence = sequence
    return copied


def event_durable_sequence(event: Event) -> int | None:
    """Return a private attached sequence without making it part of serialization."""

    if type(event) is not Event:
        raise TypeError("event must be an Event.")
    return event._durable_sequence
