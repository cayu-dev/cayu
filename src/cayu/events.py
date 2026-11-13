"""Runtime events, event types, and event identity helpers."""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    field_validator,
    model_validator,
)

from cayu._validation import (
    DURABLE_DOCUMENT_LIMITS,
    DURABLE_EVENT_LIMITS,
    MAX_DURABLE_JSON_INTEGER,
    MIN_DURABLE_JSON_INTEGER,
    copy_durable_record,
    inspect_bounded_durable_json,
    require_durable_clean_nonblank,
)

_CUSTOM_EVENT_TYPE_RE = re.compile(r"^custom\.[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$")
EVENT_ID_MAX_CHARS = 512


class _PayloadIdentity:
    """Keep one stamped object alive and compare identity without reading content."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return type(other) is _PayloadIdentity and self.value is other.value

    def __deepcopy__(self, memo: dict[int, Any]) -> _PayloadIdentity:
        # Same result as the generic slots reconstruction, which dominated
        # deep copies of stamped events.
        return _PayloadIdentity(deepcopy(self.value, memo))


class _PayloadStampMismatch(Exception):
    """Internal signal that an Event payload changed after validation."""


class _CandidateEnvelopeStamp:
    """Compare admitted envelope values without exposing them in private reprs."""

    __slots__ = ("values",)

    def __init__(self, values: tuple[object, ...]) -> None:
        self.values = values

    def __eq__(self, other: object) -> bool:
        return type(other) is _CandidateEnvelopeStamp and self.values == other.values


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
    WORKSPACE_CHECKPOINT_UPDATED = "workspace.checkpoint.updated"
    EXECUTION_SNAPSHOT_UPDATED = "execution.snapshot.updated"
    SERVER_MUTATION_ACCEPTED = "server.mutation.accepted"

    RECOVERY_PLAN_ITEM_EXECUTED = "recovery.plan.item.executed"

    SESSION_STARTED = "session.started"
    SESSION_RESUMED = "session.resumed"
    SESSION_COMPLETED = "session.completed"
    SESSION_FAILED = "session.failed"
    SESSION_INTERRUPTED = "session.interrupted"
    SESSION_DELEGATED_ACTION_UPDATED = "session.delegated_action.updated"
    SESSION_INTERRUPTION_CASCADE_RETRY_REQUESTED = "session.interruption_cascade_retry_requested"
    SESSION_INTERRUPTION_CASCADE_COMPLETED = "session.interruption_cascade_completed"
    SESSION_INTERRUPTION_CASCADE_FAILED = "session.interruption_cascade_failed"
    SESSION_AWAITING_USER_INPUT = "session.awaiting_user_input"
    SESSION_CHECKPOINTED = "session.checkpointed"
    SESSION_FORKED = "session.forked"
    SESSION_EXPORT_PUBLISHED = "session.export.published"
    SESSION_EXPORT_RELEASED = "session.export.released"
    SESSION_EXPORT_RETIRED = "session.export.retired"
    SESSION_LIMIT_REACHED = "session.limit_reached"
    SESSION_MESSAGE_QUEUED = "session.message.queued"
    SESSION_MESSAGE_DELIVERED = "session.message.delivered"
    SESSION_MESSAGE_WITHDRAWN = "session.message.withdrawn"
    SESSION_MESSAGE_QUARANTINED = "session.message.quarantined"
    SESSION_MESSAGE_STALE = "session.message.stale"
    SESSION_MESSAGE_EXPIRED = "session.message.expired"
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

    MODEL_STARTED = "model.started"
    MODEL_TEXT_DELTA = "model.text.delta"
    MODEL_THINKING_DELTA = "model.thinking.delta"
    MODEL_HOSTED_TOOL_CALL = "model.hosted_tool_call"
    MODEL_CITATION = "model.citation"
    MODEL_COMPLETED = "model.completed"
    MODEL_AUXILIARY_ATTEMPT_STARTED = "model.auxiliary.attempt_started"
    MODEL_AUXILIARY_ATTEMPT_SETTLED = "model.auxiliary.attempt_settled"
    MODEL_ERROR = "model.error"
    MODEL_HTTP_CLEANUP = "model.http_cleanup"
    MODEL_RETRY = "model.retry"
    MODEL_ATTEMPT_DISCARDED = "model.attempt_discarded"
    MODEL_FAILOVER_SELECTED = "model.failover.selected"
    MODEL_FAILOVER_EXHAUSTED = "model.failover.exhausted"
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

    KNOWLEDGE_LOOKUP = "knowledge.lookup"

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
    ENVIRONMENT_LIFECYCLE_PROGRESS = "environment.lifecycle.progress"
    ENVIRONMENT_LIFECYCLE_TRANSITION = "environment.lifecycle.transition"
    ENVIRONMENT_DEFERRED = "environment.deferred"
    ENVIRONMENT_MATERIALIZATION_STARTED = "environment.materialization.started"
    ENVIRONMENT_MATERIALIZATION_COMPLETED = "environment.materialization.completed"
    ENVIRONMENT_MATERIALIZATION_FAILED = "environment.materialization.failed"

    WORKSPACE_REVISION_OBSERVED = "workspace.revision.observed"
    WORKSPACE_MUTATION_RECORDED = "workspace.mutation.recorded"
    WORKSPACE_OBSERVATION_FINALIZED = "workspace.observation.finalized"

    HOOK_STARTED = "hook.started"
    HOOK_COMPLETED = "hook.completed"
    HOOK_FAILED = "hook.failed"

    TOOL_CALL_STARTED = "tool.call.started"
    TOOL_CALL_COMPLETED = "tool.call.completed"
    TOOL_CALL_FAILED = "tool.call.failed"
    TOOL_EFFECT_RECONCILIATION_OBSERVED = "tool.effect.reconciliation.observed"
    TOOL_EFFECT_OUTCOME_UNKNOWN = "tool.effect.outcome_unknown"
    TOOL_EFFECT_CLEANUP_OBSERVED = "tool.effect.cleanup.observed"
    TOOL_EFFECT_RECONCILIATION_STARTED = "tool.effect.reconciliation.started"
    TOOL_EFFECT_RECONCILIATION_CONFLICT = "tool.effect.reconciliation.conflict"
    TOOL_EFFECT_RECEIPT_VALIDATED = "tool.effect.receipt.validated"
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
    RUNTIME_EVENT_LOOP_LAGGED = "runtime.event_loop.lagged"
    RUNTIME_INTERACTION_TRANSITION_ACKNOWLEDGEMENT_FAILED = (
        "runtime.interaction_transition.acknowledgement_failed"
    )


SESSION_EXPORT_EVENT_TYPES = frozenset(
    {
        EventType.SESSION_EXPORT_PUBLISHED,
        EventType.SESSION_EXPORT_RELEASED,
        EventType.SESSION_EXPORT_RETIRED,
    }
)
SESSION_EXPORT_EVENT_FIELDS = frozenset({"export_commitment", "output_commitment"})


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
    # Exact bytes admitted by the explicit trusted JSONL history boundary.
    # This is not export mutation authority and is invalid after any edit.
    _trusted_export_history: bytes | None = PrivateAttr(default=None)
    # A content-independent structural stamp lets runtime-owned copies reuse
    # already-validated immutable strings without trusting a caller-mutated
    # payload. ``model_copy(update=...)`` and any nested mutation change the
    # root/container/scalar identity graph and therefore fall back to the full
    # durable validator.
    _payload_validation_stamp: object | None = PrivateAttr(default=None)
    _candidate_envelope_stamp: _CandidateEnvelopeStamp | None = PrivateAttr(default=None)
    # The JSON text a built-in store wrote for this payload after validating it,
    # set only by ``event_from_trusted_store_record``. A copy reuses it only while
    # the payload still serializes to exactly this text, so a mutated payload
    # falls back to the full durable validator like an unstamped one.
    _stored_payload_json: str | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        del __context
        self._id_origin = "caller" if "id" in self.model_fields_set else "runtime"
        self._runtime_generated_id = self.id if self._id_origin == "runtime" else None

    @model_validator(mode="wrap")
    @classmethod
    def validate_durable_envelope(cls, value: Any, handler: Any) -> Event:
        try:
            existing = isinstance(value, Event)
            event = handler(value)
            # Unpublished candidates have the larger document ceiling;
            # store/writer/SSE admission still applies the publication ceiling.
            envelope_stamp = _validate_event_candidate(event)
            if not existing:
                # Only fresh field admission gives us a private detached payload.
                # Revalidating an existing Event skips field validators and can
                # race with caller mutation, so it must never refresh proofs.
                # model_construct bypasses this wrapper and mints neither proof.
                event._payload_validation_stamp = _durable_payload_stamp(event.payload)
                event._candidate_envelope_stamp = envelope_stamp
            return event
        finally:
            value = None

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Event:
        """Deep-copy a stamped payload in one pass instead of copying its stamp too.

        The generic copy also deep-copies ``_payload_validation_stamp``, which
        holds an identity node for every payload value. For a standalone copy
        (``model_copy(deep=True)``) whose stamp still proves the payload is the
        validated tree, copy its containers once and restamp the copy; every
        other field and private attribute still takes the generic path.

        A caller memo means the event is copied as part of a larger structure
        that may also hold its payload or part of it, so that copy, like a
        changed payload, takes the generic path and keeps the shared references.
        """

        if type(self) is not Event or memo is not None:
            return super().__deepcopy__(memo)
        stamp = self._payload_validation_stamp
        try:
            payload = _copy_prevalidated_payload(self.payload, stamp)
        except _PayloadStampMismatch:
            stored = _copy_store_written_payload(self)
            if stored is None:
                return super().__deepcopy__()
            return super().__deepcopy__({id(self.payload): stored})
        # Nothing outside this event refers to the payload or its stamp, and a
        # validated payload is a tree, so mapping just these two is complete.
        return super().__deepcopy__(
            {id(self.payload): payload, id(stamp): _durable_payload_stamp(payload)}
        )

    def _copy_without_payload(self) -> Event:
        """A detached copy with an empty payload, made without reading the payload.

        Every other field and private attribute is copied as ``model_copy(deep=True)``
        copies it; the payload and its validation stamp are replaced, not copied.
        """

        payload: dict[str, Any] = {}
        memo: dict[int, Any] = {id(self.payload): payload}
        stamp = self._payload_validation_stamp
        if stamp is not None:
            memo[id(stamp)] = _durable_payload_stamp(payload)
        return super().__deepcopy__(memo)

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
        return copy_durable_record(value, "payload", limits=DURABLE_DOCUMENT_LIMITS)

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


def validate_session_export_event(event: Event) -> None:
    """Require exact producer-attested metadata before export event publication.

    Producers attest both commitments with ``event_with_runtime_payload_authority``
    only after the export owner has verified the operation. This is projection
    provenance, never a receipt, disclosure grant, or replay permission. Store
    ingestion must enforce this boundary too; ordinary deserialization does not
    restore producer provenance.
    """

    if type(event) is not Event:
        raise TypeError("event must be an Event.")
    if event.type not in SESSION_EXPORT_EVENT_TYPES:
        return
    payload = event.payload
    if (
        type(payload) is not dict
        or any(type(key) is not str for key in payload)
        or set(payload) != SESSION_EXPORT_EVENT_FIELDS
    ):
        raise ValueError("Session export events require exactly two commitment fields.")
    for field_name in SESSION_EXPORT_EVENT_FIELDS:
        value = payload[field_name]
        if not (
            type(value) is str
            and re.fullmatch(r"[0-9a-f]{64}", value) is not None
            and event_payload_authority_is_runtime_generated(
                event, field_name=field_name, value=value
            )
        ):
            raise ValueError("Session export events require runtime-attested SHA-256 commitments.")


def event_durable_envelope(event: Event) -> dict[str, Any]:
    """One explicit envelope, including every durable correlation field."""

    return {
        "id": event.id,
        "type": str(event.type),
        "session_id": event.session_id,
        "interaction_id": event.interaction_id,
        "timestamp": event.timestamp.isoformat(),
        "agent_name": event.agent_name,
        "environment_name": event.environment_name,
        "workflow_name": event.workflow_name,
        "tool_name": event.tool_name,
        "payload": event.payload,
    }


def validate_event_envelope(event: Event) -> None:
    inspect_bounded_durable_json(
        event_durable_envelope(event),
        "event",
        max_bytes=DURABLE_EVENT_LIMITS.max_bytes,
        max_nodes=DURABLE_EVENT_LIMITS.max_nodes,
        max_nesting=DURABLE_EVENT_LIMITS.max_nesting,
        canonical_numbers=False,
    )


def _validate_event_candidate(event: Event) -> _CandidateEnvelopeStamp:
    """Bound private preparation without admitting the candidate for publication."""

    envelope = event_durable_envelope(event)
    inspect_bounded_durable_json(
        envelope,
        "event",
        max_bytes=DURABLE_DOCUMENT_LIMITS.max_bytes,
        max_nodes=DURABLE_DOCUMENT_LIMITS.max_nodes,
        max_nesting=DURABLE_DOCUMENT_LIMITS.max_nesting,
        canonical_numbers=False,
    )
    # Capture the same immutable envelope values that were inspected, including
    # timestamp serialization, rather than recomputing a potentially changed view.
    envelope.pop("payload")
    return _CandidateEnvelopeStamp(tuple((type(value), value) for value in envelope.values()))


def copy_event(event: Event) -> Event:
    if type(event) is not Event:
        raise TypeError("Events must be Event instances.")
    try:
        payload = _copy_prevalidated_payload(
            event.payload,
            event._payload_validation_stamp,
        )
    except _PayloadStampMismatch:
        stored = _copy_store_written_payload(event)
        if stored is not None:
            payload = stored
            payload_is_unchanged = True
            payload_is_admitted = True
        else:
            payload_is_unchanged = False
            # The same admission Event's payload validator applies, so an object
            # result is installed below instead of being copied a second time. A
            # non-object still goes through the constructor and fails its type check.
            payload = copy_durable_record(event.payload, "payload", limits=DURABLE_DOCUMENT_LIMITS)
            payload_is_admitted = type(payload) is dict
    else:
        stored = None
        payload_is_unchanged = True
        payload_is_admitted = True
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
        # Validate the small envelope normally. The payload is installed below
        # only when its structural stamp proves it is the exact graph already
        # accepted by Event's durable validator, or it was just admitted above.
        payload={} if payload_is_admitted else payload,
    )
    if payload_is_admitted:
        copied.payload = payload
        if (
            not payload_is_unchanged
            or event._candidate_envelope_stamp != copied._candidate_envelope_stamp
        ):
            copied._candidate_envelope_stamp = _validate_event_candidate(copied)
        if stored is not None:
            # Still the stored text: keep proving it by content, not by a stamp.
            copied._stored_payload_json = event._stored_payload_json
        else:
            copied._payload_validation_stamp = _durable_payload_stamp(payload)
    copied._id_origin = event._id_origin
    copied._runtime_generated_id = event._runtime_generated_id
    copied._runtime_payload_authority = event._runtime_payload_authority
    copied._runtime_nested_payload_authority = event._runtime_nested_payload_authority
    copied._runtime_envelope_authority = event._runtime_envelope_authority
    copied._durable_sequence = event._durable_sequence
    copied._trusted_export_history = event._trusted_export_history
    return copied


def store_payload_json(payload: Any) -> str:
    """The exact JSON text built-in stores write for an event payload."""

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class _UntrustedStoredPayload(Exception):
    """Internal signal that stored payload text needs the full durable validator."""


def _stored_json_float(text: str) -> float | int:
    value = float(text)
    if not math.isfinite(value):
        raise _UntrustedStoredPayload
    if value.is_integer():
        # The durable validator admits an integral float as the integer it equals.
        if not MIN_DURABLE_JSON_INTEGER <= value <= MAX_DURABLE_JSON_INTEGER:
            raise _UntrustedStoredPayload
        return int(value)
    return value


def _stored_json_int(text: str) -> int:
    value = int(text)
    # Only a token of 19 or more digits can fall outside the 64-bit range.
    if len(text) > 18 and not MIN_DURABLE_JSON_INTEGER <= value <= MAX_DURABLE_JSON_INTEGER:
        raise _UntrustedStoredPayload
    return value


def _stored_json_constant(text: str) -> Any:
    raise _UntrustedStoredPayload


# The hooks see number and constant tokens only, never string contents, so a
# non-finite number, an integral float or an out-of-range integer is caught
# exactly while the rest decodes at C speed (the integer hook adds well under a
# microsecond to a typical stored payload).
_STORED_PAYLOAD_DECODER = json.JSONDecoder(
    parse_float=_stored_json_float,
    parse_int=_stored_json_int,
    parse_constant=_stored_json_constant,
)
_STORED_TEXT_ESCAPE_RE = re.compile(r"\\u(?:0000|[dD][89a-fA-F])")
_TIMESTAMP_ADAPTER: TypeAdapter[datetime] = TypeAdapter(datetime)
_EVENT_TYPES_BY_VALUE: dict[str, EventType] = {member.value: member for member in EventType}


def _decode_stored_payload(text: Any) -> dict[str, Any]:
    """Decode payload text a built-in store wrote, or raise ``_UntrustedStoredPayload``."""

    if type(text) is not str:
        raise _UntrustedStoredPayload
    try:
        encoded_length = len(text.encode("utf-8"))
    except UnicodeEncodeError:
        # A lone surrogate code point in the text itself.
        raise _UntrustedStoredPayload from None
    if encoded_length > DURABLE_DOCUMENT_LIMITS.max_bytes or "\x00" in text:
        raise _UntrustedStoredPayload
    if "\\u" in text and _STORED_TEXT_ESCAPE_RE.search(text) is not None:
        # An escaped NUL or surrogate; valid surrogate pairs also land here and
        # are admitted by the full validator, which this only costs time.
        raise _UntrustedStoredPayload
    try:
        payload = _STORED_PAYLOAD_DECODER.decode(text)
    except (ValueError, RecursionError):
        raise _UntrustedStoredPayload from None
    if type(payload) is not dict:
        raise _UntrustedStoredPayload
    return payload


def _copy_store_written_payload(event: Event) -> dict[str, Any] | None:
    """A detached copy of a payload that still equals the text its store wrote."""

    text = event._stored_payload_json
    if text is None:
        return None
    try:
        if store_payload_json(event.payload) != text:
            return None
    except (TypeError, ValueError, RecursionError):
        return None
    # Decoding the stored text, not copying the live graph, means the copy is
    # always the value the store validated, whatever objects the caller holds.
    try:
        return _decode_stored_payload(text)
    except _UntrustedStoredPayload:
        return None


def _trusted_envelope_text(value: Any, field_name: str, *, optional: bool) -> str | None:
    if value is None and optional:
        return None
    if type(value) is not str:
        raise _UntrustedStoredPayload
    return require_durable_clean_nonblank(value, field_name)


def event_from_trusted_store_record(
    *,
    event_type: Any,
    session_id: Any,
    interaction_id: Any,
    event_id: Any,
    timestamp: Any,
    agent_name: Any,
    environment_name: Any,
    workflow_name: Any,
    tool_name: Any,
    payload_json: Any,
    runtime_payload_authority_candidates: tuple[str, ...] = (),
) -> Event:
    """Rebuild an Event from a record a built-in store wrote, without re-walking it.

    Only built-in store decoders call this, for rows the store's own ingestion
    boundary (``_copy_event_for_session_store``) validated before writing them.
    Trust boundaries (JSONL restore, imports, events from outside the store)
    construct ``Event`` normally and keep full validation.

    Checks that still run on every record:

    - every envelope field: exact types, the event-type registry, durable clean
      non-blank identifiers and names, the id length limit and timestamp parsing,
      with the same validators ``Event`` uses;
    - the payload: JSON syntax, a JSON object, the durable byte limit, no NaN
      or infinity, integers and integral floats inside the durable integer
      range (an integral float becomes the integer it equals, as the validator
      makes it), and no NUL or lone surrogate, raw or escaped;
    - authority: each candidate field the store attested that holds a string is
      attested again, and must be non-blank, exactly as
      ``event_with_runtime_payload_authority`` requires.

    Checks skipped for store-written rows, which the write path enforced and
    only a direct edit of the database could break: the payload's node count
    and its nesting depth. ``cayu storage verify-events`` reruns the full
    validator over stored events.

    A record failing any of these checks goes through the validating
    constructor instead, which raises exactly the error a read raised before.
    The payload carries no identity stamp, which costs more than decoding it.
    Instead the stored text is kept, and a copy reuses it only while the
    payload still serializes to that exact text.
    """

    try:
        payload = _decode_stored_payload(payload_json)
        if type(event_type) is not str:
            raise _UntrustedStoredPayload
        checked_type: EventType | str | None = _EVENT_TYPES_BY_VALUE.get(event_type)
        if checked_type is None:
            checked_type = _validate_custom_event_type(event_type)
        if type(event_id) is not str or len(event_id) > EVENT_ID_MAX_CHARS:
            raise _UntrustedStoredPayload
        fields: dict[str, Any] = {
            "type": checked_type,
            "session_id": _trusted_envelope_text(session_id, "session_id", optional=False),
            "interaction_id": _trusted_envelope_text(
                interaction_id, "interaction_id", optional=True
            ),
            "id": _trusted_envelope_text(event_id, "id", optional=False),
            "timestamp": _TIMESTAMP_ADAPTER.validate_python(timestamp),
            "agent_name": _trusted_envelope_text(agent_name, "agent_name", optional=True),
            "environment_name": _trusted_envelope_text(
                environment_name, "environment_name", optional=True
            ),
            "workflow_name": _trusted_envelope_text(workflow_name, "workflow_name", optional=True),
            "tool_name": _trusted_envelope_text(tool_name, "tool_name", optional=True),
            "payload": payload,
        }
        authority = []
        for field_name in runtime_payload_authority_candidates:
            value = payload.get(field_name)
            if type(value) is not str:
                continue
            if not value.strip():
                raise _UntrustedStoredPayload
            authority.append((field_name, value))
    except (_UntrustedStoredPayload, ValueError, TypeError):
        event = Event(
            type=event_type,
            session_id=session_id,
            interaction_id=interaction_id,
            id=event_id,
            timestamp=timestamp,
            agent_name=agent_name,
            environment_name=environment_name,
            workflow_name=workflow_name,
            tool_name=tool_name,
            payload=json.loads(payload_json) if type(payload_json) is str else payload_json,
        )
        attested = tuple(
            field_name
            for field_name in runtime_payload_authority_candidates
            if type(event.payload.get(field_name)) is str
        )
        return event_with_runtime_payload_authority(event, *attested) if attested else event
    event = Event.model_construct(**fields)
    # The stamp the validating constructor records: every envelope value except
    # the payload, in ``event_durable_envelope`` order.
    rendered_type = str(event.type)
    rendered_timestamp = event.timestamp.isoformat()
    event._candidate_envelope_stamp = _CandidateEnvelopeStamp(
        (
            (str, event.id),
            (str, rendered_type),
            (str, event.session_id),
            (type(event.interaction_id), event.interaction_id),
            (str, rendered_timestamp),
            (type(event.agent_name), event.agent_name),
            (type(event.environment_name), event.environment_name),
            (type(event.workflow_name), event.workflow_name),
            (type(event.tool_name), event.tool_name),
        )
    )
    # Kept even when the text is not this payload's canonical form: a copy then
    # finds the two differ and takes the full validator, which costs only time.
    event._stored_payload_json = payload_json
    if authority:
        event._runtime_payload_authority = frozenset(authority)
    return event


def _durable_payload_stamp(value: Any) -> object:
    """Return a mutation-sensitive identity stamp without reading string data."""

    value_type = type(value)
    if value_type is dict:
        return (
            "object",
            _PayloadIdentity(value),
            tuple(
                (_PayloadIdentity(key), _durable_payload_stamp(item)) for key, item in value.items()
            ),
        )
    if value_type is list:
        return (
            "array",
            _PayloadIdentity(value),
            tuple(_durable_payload_stamp(item) for item in value),
        )
    return (value_type, _PayloadIdentity(value))


def _copy_prevalidated_payload(value: Any, stamp: object) -> Any:
    """Copy and identity-check one validated graph in the same traversal.

    Reading each container into a local immutable snapshot before comparing it
    closes the check-then-copy race: a replacement observed by this pass must
    match the strong object reference retained in ``stamp``. A later mutation
    affects only the caller's graph, never the detached result.
    """

    value_type = type(value)
    if value_type is dict:
        if (
            type(stamp) is not tuple
            or len(stamp) != 3
            or stamp[0] != "object"
            or stamp[1] != _PayloadIdentity(value)
            or type(stamp[2]) is not tuple
        ):
            raise _PayloadStampMismatch
        items = tuple(value.items())
        children = stamp[2]
        if len(items) != len(children):
            raise _PayloadStampMismatch
        copied: dict[str, Any] = {}
        for (key, item), child in zip(items, children, strict=True):
            if type(child) is not tuple or len(child) != 2 or child[0] != _PayloadIdentity(key):
                raise _PayloadStampMismatch
            copied[key] = _copy_prevalidated_payload(item, child[1])
        return copied
    if value_type is list:
        if (
            type(stamp) is not tuple
            or len(stamp) != 3
            or stamp[0] != "array"
            or stamp[1] != _PayloadIdentity(value)
            or type(stamp[2]) is not tuple
        ):
            raise _PayloadStampMismatch
        items = tuple(value)
        children = stamp[2]
        if len(items) != len(children):
            raise _PayloadStampMismatch
        return [
            _copy_prevalidated_payload(item, child)
            for item, child in zip(items, children, strict=True)
        ]
    if (
        type(stamp) is not tuple
        or len(stamp) != 2
        or stamp[0] is not value_type
        or stamp[1] != _PayloadIdentity(value)
    ):
        raise _PayloadStampMismatch
    return value


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


def runtime_payload_authority_paths(
    event: Event,
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Return the payload fields and nested paths a runtime producer attested.

    A durable record that rebuilds this event later re-attests exactly these, so
    the rebuilt event carries the same provenance as the one first published.
    """

    if type(event) is not Event:
        raise TypeError("event must be an Event.")
    fields = sorted({field_name for field_name, _ in event._runtime_payload_authority})
    paths = sorted({path for path, _ in event._runtime_nested_payload_authority})
    return tuple(fields), tuple(paths)


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
