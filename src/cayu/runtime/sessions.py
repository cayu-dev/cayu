from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import heapq
import json
import math
import secrets
import time
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from itertools import islice
from typing import Any, ClassVar, Literal
from uuid import uuid4
from weakref import ReferenceType, ref

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema  # noqa: TC002 - Pydantic needs this at runtime.

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    JsonUtf8SizeCounter,
    canonical_durable_json_bytes,
    compact_json_utf8_size,
    copy_durable_json_object,
    copy_durable_json_value,
    copy_label_map,
    json_utf8_size_within_limit,
    require_durable_json_text,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu._validation import (
    require_durable_nonblank as require_nonblank,
)
from cayu.core.events import (
    EVENT_ID_MAX_CHARS,
    Event,
    EventType,
    copy_event,
    event_with_runtime_envelope_authority,
    event_with_runtime_payload_authority,
)
from cayu.core.messages import (
    Message,
    MessageRole,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
    copy_message,
    detach_message,
)
from cayu.core.thinking import ThinkingConfig
from cayu.core.workflows import WORKFLOW_ATTEMPT_EVENT_TYPE
from cayu.runtime._model_completion_publication import (
    LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
    ModelStepPublicationCheckpoint,
)
from cayu.runtime._terminal_evidence import (
    SESSION_RUN_OPERATION_ID_PAYLOAD_KEY,
    TERMINAL_EVIDENCE_EVENT_TYPES,
    TERMINAL_EVIDENCE_QUERY_LIMIT,
    classify_current_terminal_evidence,
)
from cayu.runtime.aggregates import (
    EXACT_AGGREGATE,
    AggregateAccuracy,
    AggregateAccuracyKind,
    AggregateCount,
    AggregateUsageMetrics,
    BoundedUsagePricingInputAccumulator,
    UsageAggregateBreakdown,
    UsageAggregateGroup,
    UsageAggregateRemainder,
    UsageAggregateTotals,
    UsageRollupStoreResult,
    UsageSessionAggregateBreakdown,
    UsageSessionAggregateGroup,
    UsageSessionAggregateRemainder,
    add_aggregate_usage,
    aggregate_usage_metrics_from_event_payload,
    build_aggregate_usage_metrics,
    normalize_aggregate_event_timestamp,
    project_aggregate_usage_inspection_event,
    require_bounded_usage_session_id,
)
from cayu.runtime.approvals import (
    PendingToolCallApproval,
    ResolutionActor,
    ToolPolicyEvidence,
    copy_resolution_actor,
    resolution_actor_payload,
)
from cayu.runtime.budgets import (
    BudgetLimit,
    BudgetReservationIdentityConflict,
    SessionBudgetInspection,
    copy_request_budget_limits,
    is_budget_inspection_event,
    model_completion_budget_settlements,
    project_budget_inspection_event,
    project_budget_model_attempt_inspection_event,
    session_budget_inspection,
)
from cayu.runtime.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.runtime.execution_units import (
    ModelAttemptIdentity,
    ToolRoundIdentity,
    copy_tool_round_identity,
)
from cayu.runtime.loop_policies import LoopPolicy, validate_loop_policies
from cayu.runtime.public_authority import (
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
    parse_public_authority_alias,
)
from cayu.runtime.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime.stop_policy import RunLimits, copy_run_limits
from cayu.runtime.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputError,
    StructuredOutputSpec,
    StructuredOutputStrategy,
    StructuredOutputValidation,
    copy_structured_output_spec,
)
from cayu.runtime.usage import UsageMetrics


class SessionStatusConflict(ValueError):
    """A session status transition was rejected because the session was not in an
    allowed source status (e.g. resuming a session another worker is already
    running). Subclasses ``ValueError`` so existing ``except ValueError`` handlers
    keep working; callers that need to react specifically (e.g. requeue) catch this.
    """


class SessionRunFenced(RuntimeError):
    """A durable write was rejected because its run no longer owns the session epoch."""


class McpManifestHistoryConflict(RuntimeError):
    """An MCP manifest check could not establish or fence authoritative history."""


class _McpManifestBaselineEvidenceInvalid(ValueError):
    """Stored MCP baseline evidence could not be decoded or validated safely."""


class PersistedEventSideEffectClaimLost(RuntimeError):
    """A side-effect acknowledgement lost ownership to a replacement claim."""


PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES = 4096
# These budgets are nested deliberately. A maximum-size session ID is
# base64-encoded into a list cursor that fits 4 KiB; a maximum-size custom-store
# list cursor is base64-encoded into a recovery cursor that fits 8 KiB.
MAX_SESSION_ID_BYTES = 2048
MAX_SESSION_LIST_CURSOR_BYTES = 4096
MAX_INCOMPLETE_SESSIONS_RECOVERY_CURSOR_BYTES = 8192
_SESSION_CURSOR_VERSION = 1
_NONPORTABLE_PERSISTED_EVENT_SIDE_EFFECT_ERROR = (
    "Persisted event side effect failed with non-portable error details."
)
_TRUNCATED_PERSISTED_EVENT_SIDE_EFFECT_ERROR_SUFFIX = "... [truncated]"


def validate_persisted_event_side_effect_error(
    value: str,
    field_name: str = "error",
) -> str:
    """Validate a portable, bounded side-effect failure description."""

    value = require_nonblank(value, field_name)
    if len(value.encode("utf-8")) > PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES:
        raise ValueError(
            f"`{field_name}` must not exceed "
            f"{PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES} UTF-8 bytes."
        )
    return value


def portable_persisted_event_side_effect_error(value: object) -> str:
    """Project untrusted exception text into the durable handoff contract."""

    if type(value) is not str:
        return _NONPORTABLE_PERSISTED_EVENT_SIDE_EFFECT_ERROR
    try:
        value = require_nonblank(value, "error")
    except ValueError:
        return _NONPORTABLE_PERSISTED_EVENT_SIDE_EFFECT_ERROR

    encoded = value.encode("utf-8")
    if len(encoded) <= PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES:
        return value

    suffix = _TRUNCATED_PERSISTED_EVENT_SIDE_EFFECT_ERROR_SUFFIX
    prefix = encoded[: PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES - len(suffix.encode("utf-8"))]
    while True:
        try:
            return prefix.decode("utf-8") + suffix
        except UnicodeDecodeError:
            prefix = prefix[:-1]


def _require_bounded_session_id(value: str, field_name: str) -> str:
    value = require_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > MAX_SESSION_ID_BYTES:
        raise ValueError(f"`{field_name}` must not exceed {MAX_SESSION_ID_BYTES} UTF-8 bytes.")
    return value


def _require_bounded_session_list_cursor(value: str, field_name: str) -> str:
    value = require_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > MAX_SESSION_LIST_CURSOR_BYTES:
        raise ValueError(
            f"`{field_name}` must not exceed {MAX_SESSION_LIST_CURSOR_BYTES} UTF-8 bytes."
        )
    return value


class SessionQueuedMessagesPending(RuntimeError):
    """Terminalization lost a race to durable queued session input."""


class SessionRuntimePublicationConflict(ValueError):
    """A runtime publication identity conflicts with its durable receipt."""


class SessionModelCompletionStageConflict(SessionRuntimePublicationConflict):
    """A logical model-completion stage conflicts with immutable durable evidence."""


class SessionModelCompletionStageIncomplete(RuntimeError):
    """A logical model step was prepared but has no durable terminal completion."""


_SESSION_RUN_FENCES: ContextVar[dict[str, int] | None] = ContextVar(
    "cayu_session_run_fence",
    default=None,
)
_SESSION_INTERACTION_IDS: ContextVar[dict[str, str] | None] = ContextVar(
    "cayu_session_interaction_ids",
    default=None,
)
_SESSION_INTERACTION_STARTED_AT: ContextVar[dict[str, float] | None] = ContextVar(
    "cayu_session_interaction_started_at",
    default=None,
)
_SESSION_INTERACTION_RECOVERED_ACTIVE_THROUGH: ContextVar[dict[str, datetime] | None] = ContextVar(
    "cayu_session_interaction_recovered_active_through",
    default=None,
)
_SESSION_INVOCATION_INTERACTION_IDS: ContextVar[dict[str, tuple[str, ...]] | None] = ContextVar(
    "cayu_session_invocation_interaction_ids",
    default=None,
)


class _InheritInteractionAttribution:
    __slots__ = ()


INHERIT_INTERACTION = _InheritInteractionAttribution()
InteractionAttribution = str | None | _InheritInteractionAttribution
UNASSOCIATED_RUNTIME_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
        EventType.SESSION_FAILED,
        EventType.SESSION_INTERRUPTED,
    }
)


def _current_session_interaction_id(session_id: str) -> str | None:
    interactions = _SESSION_INTERACTION_IDS.get()
    return None if interactions is None else interactions.get(session_id)


def _current_session_interaction_started_at(session_id: str) -> float | None:
    started = _SESSION_INTERACTION_STARTED_AT.get()
    return None if started is None else started.get(session_id)


def _current_session_interaction_recovered_active_through(
    session_id: str,
) -> datetime | None:
    recovered = _SESSION_INTERACTION_RECOVERED_ACTIVE_THROUGH.get()
    return None if recovered is None else recovered.get(session_id)


def _set_session_interaction_recovered_active_through(
    session_id: str,
    active_through: datetime,
) -> None:
    session_id = require_clean_nonblank(session_id, "session_id")
    if active_through.tzinfo is None or active_through.utcoffset() is None:
        raise ValueError("active_through must be timezone-aware.")
    recovered = dict(_SESSION_INTERACTION_RECOVERED_ACTIVE_THROUGH.get() or {})
    recovered[session_id] = active_through.astimezone(UTC)
    _SESSION_INTERACTION_RECOVERED_ACTIVE_THROUGH.set(recovered)


def _clear_session_interaction_recovered_active_through(session_id: str) -> None:
    recovered = _SESSION_INTERACTION_RECOVERED_ACTIVE_THROUGH.get()
    if recovered is None or session_id not in recovered:
        return
    remaining = dict(recovered)
    remaining.pop(session_id)
    _SESSION_INTERACTION_RECOVERED_ACTIVE_THROUGH.set(remaining or None)


def _current_session_invocation_interaction_ids(session_id: str) -> tuple[str, ...]:
    interactions = _SESSION_INVOCATION_INTERACTION_IDS.get()
    return () if interactions is None else interactions.get(session_id, ())


def resolve_interaction_attribution(
    session_id: str,
    interaction_id: InteractionAttribution,
) -> str | None:
    """Resolve omitted transcript attribution while preserving explicit null."""

    if interaction_id is INHERIT_INTERACTION:
        interaction_id = _current_session_interaction_id(session_id)
    if interaction_id is None:
        return None
    if type(interaction_id) is not str:
        raise TypeError("interaction_id must be a string, None, or INHERIT_INTERACTION.")
    return require_clean_nonblank(interaction_id, "interaction_id")


def attribute_event_to_current_interaction(event: Event) -> Event:
    """Copy an event and explicitly apply the active runtime interaction identity."""

    copied = copy_event(event)
    if copied.interaction_id is None and copied.type not in UNASSOCIATED_RUNTIME_EVENT_TYPES:
        interaction_id = _current_session_interaction_id(copied.session_id)
        if interaction_id is not None:
            copied = copied.model_copy(update={"interaction_id": interaction_id})
    active_interaction_id = _current_session_interaction_id(copied.session_id)
    invocation_interaction_ids = _current_session_invocation_interaction_ids(copied.session_id)
    if active_interaction_id is not None or invocation_interaction_ids:
        fields = ["session_id"]
        if active_interaction_id is not None and copied.interaction_id == active_interaction_id:
            fields.append("interaction_id")
        copied = event_with_runtime_envelope_authority(copied, *fields)
    return copied


def attribute_events_to_current_interaction(events: list[Event]) -> list[Event]:
    if type(events) is not list:
        raise TypeError("Runtime events must be a list.")
    return [attribute_event_to_current_interaction(event) for event in events]


def _activate_session_interaction(session_id: str, interaction_id: str) -> None:
    session_id = require_clean_nonblank(session_id, "session_id")
    interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
    interactions = dict(_SESSION_INTERACTION_IDS.get() or {})
    previous_interaction_id = interactions.get(session_id)
    interactions[session_id] = interaction_id
    _SESSION_INTERACTION_IDS.set(interactions)
    started = dict(_SESSION_INTERACTION_STARTED_AT.get() or {})
    if previous_interaction_id != interaction_id or session_id not in started:
        started[session_id] = time.monotonic()
        _SESSION_INTERACTION_STARTED_AT.set(started)
    invocation_interactions = dict(_SESSION_INVOCATION_INTERACTION_IDS.get() or {})
    seen = invocation_interactions.get(session_id, ())
    if interaction_id not in seen:
        invocation_interactions[session_id] = (*seen, interaction_id)
        _SESSION_INVOCATION_INTERACTION_IDS.set(invocation_interactions)


def _close_session_interaction(session_id: str) -> None:
    """Clear active attribution while retaining invocation-level history."""

    interactions = _SESSION_INTERACTION_IDS.get()
    if interactions is not None and session_id in interactions:
        remaining = dict(interactions)
        remaining.pop(session_id)
        _SESSION_INTERACTION_IDS.set(remaining or None)
    started = _SESSION_INTERACTION_STARTED_AT.get()
    if started is not None and session_id in started:
        remaining_started = dict(started)
        remaining_started.pop(session_id)
        _SESSION_INTERACTION_STARTED_AT.set(remaining_started or None)
    _clear_session_interaction_recovered_active_through(session_id)


def _deactivate_session_interaction(session_id: str) -> None:
    _close_session_interaction(session_id)
    invocation_interactions = _SESSION_INVOCATION_INTERACTION_IDS.get()
    if invocation_interactions is not None and session_id in invocation_interactions:
        remaining_invocations = dict(invocation_interactions)
        remaining_invocations.pop(session_id)
        _SESSION_INVOCATION_INTERACTION_IDS.set(remaining_invocations or None)


class _SessionRunFenceContext:
    """Carry one stream's task-local runtime ownership across task boundaries."""

    def __init__(self) -> None:
        fences = _SESSION_RUN_FENCES.get()
        self._fences = None if fences is None else dict(fences)
        interaction_ids = _SESSION_INTERACTION_IDS.get()
        self._interaction_ids = None if interaction_ids is None else dict(interaction_ids)
        interaction_started_at = _SESSION_INTERACTION_STARTED_AT.get()
        self._interaction_started_at = (
            None if interaction_started_at is None else dict(interaction_started_at)
        )
        recovered_active_through = _SESSION_INTERACTION_RECOVERED_ACTIVE_THROUGH.get()
        self._recovered_active_through = (
            None if recovered_active_through is None else dict(recovered_active_through)
        )
        invocation_interaction_ids = _SESSION_INVOCATION_INTERACTION_IDS.get()
        self._invocation_interaction_ids = (
            None if invocation_interaction_ids is None else dict(invocation_interaction_ids)
        )

    @classmethod
    def current_or_new(cls) -> _SessionRunFenceContext:
        current = _ACTIVE_SESSION_RUN_FENCE_CONTEXT.get()
        task = asyncio.current_task()
        if current is not None and current[1]() is task:
            return current[0]
        return cls()

    @contextmanager
    def activate(self) -> Iterator[None]:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("Session run-fence context requires a running task.")
        current = _ACTIVE_SESSION_RUN_FENCE_CONTEXT.get()
        if current is not None and current[0] is self and current[1]() is task:
            yield
            return
        fences = None if self._fences is None else dict(self._fences)
        fence_token = _SESSION_RUN_FENCES.set(fences)
        interaction_ids = None if self._interaction_ids is None else dict(self._interaction_ids)
        interaction_id_token = _SESSION_INTERACTION_IDS.set(interaction_ids)
        interaction_started_at = (
            None if self._interaction_started_at is None else dict(self._interaction_started_at)
        )
        interaction_started_at_token = _SESSION_INTERACTION_STARTED_AT.set(interaction_started_at)
        recovered_active_through = (
            None if self._recovered_active_through is None else dict(self._recovered_active_through)
        )
        recovered_active_through_token = _SESSION_INTERACTION_RECOVERED_ACTIVE_THROUGH.set(
            recovered_active_through
        )
        invocation_interaction_ids = (
            None
            if self._invocation_interaction_ids is None
            else dict(self._invocation_interaction_ids)
        )
        invocation_interaction_ids_token = _SESSION_INVOCATION_INTERACTION_IDS.set(
            invocation_interaction_ids
        )
        context_token = _ACTIVE_SESSION_RUN_FENCE_CONTEXT.set((self, ref(task)))
        try:
            yield
        finally:
            current = _SESSION_RUN_FENCES.get()
            self._fences = None if current is None else dict(current)
            current_interaction_ids = _SESSION_INTERACTION_IDS.get()
            self._interaction_ids = (
                None if current_interaction_ids is None else dict(current_interaction_ids)
            )
            current_interaction_started_at = _SESSION_INTERACTION_STARTED_AT.get()
            self._interaction_started_at = (
                None
                if current_interaction_started_at is None
                else dict(current_interaction_started_at)
            )
            current_recovered_active_through = _SESSION_INTERACTION_RECOVERED_ACTIVE_THROUGH.get()
            self._recovered_active_through = (
                None
                if current_recovered_active_through is None
                else dict(current_recovered_active_through)
            )
            current_invocation_interaction_ids = _SESSION_INVOCATION_INTERACTION_IDS.get()
            self._invocation_interaction_ids = (
                None
                if current_invocation_interaction_ids is None
                else dict(current_invocation_interaction_ids)
            )
            _ACTIVE_SESSION_RUN_FENCE_CONTEXT.reset(context_token)
            _SESSION_INVOCATION_INTERACTION_IDS.reset(invocation_interaction_ids_token)
            _SESSION_INTERACTION_RECOVERED_ACTIVE_THROUGH.reset(recovered_active_through_token)
            _SESSION_INTERACTION_STARTED_AT.reset(interaction_started_at_token)
            _SESSION_INTERACTION_IDS.reset(interaction_id_token)
            _SESSION_RUN_FENCES.reset(fence_token)


_ACTIVE_SESSION_RUN_FENCE_CONTEXT: ContextVar[
    tuple[_SessionRunFenceContext, ReferenceType[asyncio.Task[Any]]] | None
] = ContextVar(
    "cayu_active_session_run_fence_context",
    default=None,
)


def _current_session_run_epoch(session_id: str) -> int | None:
    fences = _SESSION_RUN_FENCES.get()
    return None if fences is None else fences.get(session_id)


def _activate_session_run_fence(session: Session) -> None:
    fences = dict(_SESSION_RUN_FENCES.get() or {})
    fences[session.id] = session.run_epoch
    _SESSION_RUN_FENCES.set(fences)


def _deactivate_session_run_fence(session_id: str) -> None:
    fences = _SESSION_RUN_FENCES.get()
    if fences is None or session_id not in fences:
        return
    remaining = dict(fences)
    remaining.pop(session_id)
    _SESSION_RUN_FENCES.set(remaining or None)


def _assert_session_run_epoch(session_id: str, session: Session) -> None:
    _assert_session_run_epoch_value(session_id, session.run_epoch)


def _assert_session_run_epoch_value(session_id: str, current_run_epoch: int) -> None:
    expected = _current_session_run_epoch(session_id)
    if expected is not None and current_run_epoch != expected:
        raise SessionRunFenced(
            f"Session run epoch no longer owns {session_id}: expected {expected}, "
            f"current {current_run_epoch}."
        )


class SessionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    INTERRUPTING = "interrupting"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


SESSION_MESSAGE_CONTENT_MAX_BYTES = 65_536
SESSION_MESSAGE_DELIVERY_BATCH_LIMIT = 100
SESSION_RUNTIME_METADATA_KEYS = frozenset({"subagent"})
SESSION_RUNTIME_METADATA_PREFIX = "cayu:"


def is_runtime_owned_session_metadata_key(key: str) -> bool:
    """Return whether a session metadata entry is owned by Cayu's runtime."""

    return key in SESSION_RUNTIME_METADATA_KEYS or key.startswith(SESSION_RUNTIME_METADATA_PREFIX)


def copy_session_user_metadata(replacement: dict[str, Any]) -> dict[str, Any]:
    """Validate and detach a complete user-authored metadata replacement."""

    copied_replacement = copy_durable_json_object(replacement, "metadata")
    for key in copied_replacement:
        if is_runtime_owned_session_metadata_key(key):
            raise ValueError(
                f"Session metadata key {key!r} is runtime-owned and cannot be replaced."
            )
    return copied_replacement


def replace_session_user_metadata(
    current: dict[str, Any],
    replacement: dict[str, Any],
) -> dict[str, Any]:
    """Combine validated user metadata with the current runtime-owned entries.

    Callers perform this merge while holding the store's session lock. Runtime
    metadata participates in recovery and policy enforcement, so it must be read
    and retained in the same transaction that writes the replacement.
    """

    if type(current) is not dict:
        raise TypeError("Current session metadata must be an object.")
    if type(replacement) is not dict:
        raise TypeError("Session metadata replacement must be an object.")
    if any(is_runtime_owned_session_metadata_key(key) for key in replacement):
        raise ValueError("Session user metadata replacement contains a runtime-owned key.")
    runtime_metadata = {
        key: copy_durable_json_value(value, "current_metadata")
        for key, value in current.items()
        if is_runtime_owned_session_metadata_key(key)
    }
    return {**replacement, **runtime_metadata}


class SessionMessageDeliveryMode(StrEnum):
    NEXT_TURN = "next_turn"
    ON_IDLE = "on_idle"


class SessionMessageQueueStatus(StrEnum):
    QUEUED = "queued"
    DELIVERED = "delivered"


class SessionDebugState(StrEnum):
    NEEDS_ATTENTION = "needs_attention"
    SESSION_FAILURE = "session_failure"
    TOOL_ISSUE = "tool_issue"
    INTERRUPTION = "interruption"


def _empty_run_request_authority() -> frozenset[tuple[str, str]]:
    return frozenset()


class RunRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        hide_input_in_errors=True,
    )

    agent_name: str
    messages: list[Message]
    # Optional caller-provided id for a new session. It must be unique.
    session_id: str | None = None
    parent_session_id: str | None = None
    # Durable budget/accounting identity shared by related sessions. Defaults to
    # task_id when present, otherwise session_id. Forks inherit the source value.
    causal_budget_id: str | None = None
    task_id: str | None = None
    task_worker_id: str | None = None
    # Per-run provider override. Resolution order for new sessions:
    # request.provider_name -> agent spec provider_name -> model-pattern route ->
    # app default provider.
    provider_name: str | None = None
    # Per-run model override for new sessions. Resume keeps the stored session
    # model unless ResumeRequest.model is set.
    model: str | None = None
    environment_name: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_steps: StrictInt = Field(default=16, ge=1, le=256)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    loop_policies: SkipJsonSchema[tuple[LoopPolicy, ...]] = Field(
        default_factory=tuple,
        exclude=True,
    )
    _runtime_generated_authority: frozenset[tuple[str, str]] = PrivateAttr(
        default_factory=_empty_run_request_authority
    )

    @field_validator("messages")
    @classmethod
    def copy_messages(cls, value):
        return [detach_message(message) for message in value]

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "metadata")

    @field_validator("labels", mode="before")
    @classmethod
    def copy_request_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels", allow_reserved=False)

    @field_validator("structured_output")
    @classmethod
    def copy_structured_output(
        cls,
        value: StructuredOutputSpec | None,
    ) -> StructuredOutputSpec | None:
        return copy_structured_output_spec(value)

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...]:
        return copy_request_budget_limits(value)

    @field_validator("loop_policies", mode="before")
    @classmethod
    def copy_loop_policies(cls, value) -> tuple[LoopPolicy, ...]:
        return validate_loop_policies(value, field_name="loop_policies")

    @field_validator("agent_name")
    @classmethod
    def validate_nonblank_agent_name(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("session_id")
    @classmethod
    def validate_optional_session_id(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return _require_bounded_session_id(value, info.field_name)

    @field_validator(
        "parent_session_id",
        "causal_budget_id",
        "task_id",
        "task_worker_id",
        "provider_name",
        "model",
        "environment_name",
    )
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_task_worker_handoff(self) -> RunRequest:
        if self.task_worker_id is not None and self.task_id is None:
            raise ValueError("RunRequest.task_worker_id requires task_id.")
        return self


class ResumeRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        hide_input_in_errors=True,
    )

    session_id: str
    messages: list[Message]
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_steps: StrictInt = Field(default=16, ge=1, le=256)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    loop_policies: SkipJsonSchema[tuple[LoopPolicy, ...]] = Field(
        default_factory=tuple,
        exclude=True,
    )

    @field_validator("messages")
    @classmethod
    def copy_messages(cls, value):
        copied_messages = [detach_message(message) for message in value]
        if not copied_messages:
            raise ValueError("ResumeRequest messages cannot be empty.")
        return copied_messages

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "metadata")

    @field_validator("structured_output")
    @classmethod
    def copy_structured_output(
        cls,
        value: StructuredOutputSpec | None,
    ) -> StructuredOutputSpec | None:
        return copy_structured_output_spec(value)

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...]:
        return copy_request_budget_limits(value)

    @field_validator("loop_policies", mode="before")
    @classmethod
    def copy_loop_policies(cls, value) -> tuple[LoopPolicy, ...]:
        return validate_loop_policies(value, field_name="loop_policies")

    @field_validator("session_id", "model")
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class CompactSessionRequest(BaseModel):
    """Request an explicit, application-owned compaction of durable session context."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    session_id: str
    idempotency_key: str = Field(max_length=256)
    expected_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    expected_transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    reason: Literal["application_requested"] = "application_requested"
    instructions: str | None = Field(default=None, max_length=4096)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    requested_by: ResolutionActor | None = None

    @field_validator("session_id", "idempotency_key")
    @classmethod
    def validate_required_strings(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("instructions")
    @classmethod
    def validate_optional_instructions(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, "instructions")

    @field_validator("limits")
    @classmethod
    def copy_limits(cls, value: RunLimits) -> RunLimits:
        return copy_run_limits(value)

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...]:
        return copy_request_budget_limits(value)

    @field_validator("requested_by")
    @classmethod
    def copy_requested_by(cls, value: ResolutionActor | None) -> ResolutionActor | None:
        return copy_resolution_actor(value)

    @model_validator(mode="after")
    def validate_durable_text(self) -> CompactSessionRequest:
        require_durable_json_text(
            self.model_dump(mode="json", exclude={"requested_by"}),
            "CompactSessionRequest",
        )
        if self.requested_by is not None:
            require_durable_json_text(
                self.requested_by.model_dump(mode="json", exclude={"claims"}),
                "CompactSessionRequest.requested_by",
            )
        return self


class EnqueueSessionMessageRequest(BaseModel):
    """Submit durable user steering for an active session."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    session_id: str
    idempotency_key: str = Field(max_length=256)
    content: str = Field(max_length=SESSION_MESSAGE_CONTENT_MAX_BYTES)
    delivery_mode: SessionMessageDeliveryMode
    requested_by: ResolutionActor | None = None

    @field_validator("session_id", "idempotency_key")
    @classmethod
    def validate_required_strings(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        value = require_nonblank(value, "content")
        if len(value.encode("utf-8")) > SESSION_MESSAGE_CONTENT_MAX_BYTES:
            raise ValueError(
                "content exceeds the maximum encoded size of "
                f"{SESSION_MESSAGE_CONTENT_MAX_BYTES} bytes."
            )
        return value

    @field_validator("requested_by")
    @classmethod
    def copy_requested_by(cls, value: ResolutionActor | None) -> ResolutionActor | None:
        return copy_resolution_actor(value)

    @model_validator(mode="after")
    def validate_durable_text(self) -> EnqueueSessionMessageRequest:
        require_durable_json_text(
            self.model_dump(mode="json", exclude={"requested_by"}),
            "EnqueueSessionMessageRequest",
        )
        if self.requested_by is not None:
            require_durable_json_text(
                self.requested_by.model_dump(mode="json", exclude={"claims"}),
                "EnqueueSessionMessageRequest.requested_by",
            )
        return self


class SessionQueuedMessage(BaseModel):
    """One durable queued user message and its delivery state."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    queue_id: str
    session_id: str
    idempotency_key: str
    content: str
    delivery_mode: SessionMessageDeliveryMode
    status: SessionMessageQueueStatus
    ordering_key: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    accepted_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    accepted_transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    accepted_event_id: str
    accepted_at: datetime
    requested_by: ResolutionActor | None = None
    delivered_run_epoch: StrictInt | None = Field(default=None, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    delivered_transcript_cursor: StrictInt | None = Field(
        default=None, ge=0, le=MAX_DURABLE_JSON_INTEGER
    )
    delivered_event_id: str | None = None
    delivered_at: datetime | None = None

    @field_validator(
        "queue_id",
        "session_id",
        "idempotency_key",
        "content",
        "accepted_event_id",
        "delivered_event_id",
    )
    @classmethod
    def validate_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)

    @field_validator("requested_by")
    @classmethod
    def copy_requested_by(cls, value: ResolutionActor | None) -> ResolutionActor | None:
        return copy_resolution_actor(value)


class EnqueueSessionMessageResult(BaseModel):
    """Typed enqueue result, including the durable acceptance event."""

    model_config = ConfigDict(extra="forbid")

    message: SessionQueuedMessage
    event: Event
    replayed: StrictBool = False

    @field_validator("message")
    @classmethod
    def copy_message(cls, value: SessionQueuedMessage) -> SessionQueuedMessage:
        if type(value) is not SessionQueuedMessage:
            raise TypeError("message must be a SessionQueuedMessage.")
        return value.model_copy(deep=True)

    @field_validator("event")
    @classmethod
    def copy_event(cls, value: Event) -> Event:
        return copy_event(value)


class SessionMessageDeliveryBatch(BaseModel):
    """One bounded atomic queue-delivery batch at a fixed eligibility cutoff."""

    model_config = ConfigDict(extra="forbid")

    messages: tuple[SessionQueuedMessage, ...] = Field(default_factory=tuple)
    events: tuple[Event, ...] = Field(default_factory=tuple)
    delivery_id: str
    interaction_id: str | None = None
    eligible_through: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    has_more: StrictBool = False
    replayed: StrictBool = False

    @field_validator("messages", mode="before")
    @classmethod
    def copy_messages(cls, value) -> tuple[SessionQueuedMessage, ...]:
        return tuple(message.model_copy(deep=True) for message in value)

    @field_validator("events", mode="before")
    @classmethod
    def copy_events(cls, value) -> tuple[Event, ...]:
        return tuple(copy_event(event) for event in value)

    @field_validator("delivery_id", "interaction_id")
    @classmethod
    def validate_identifiers(cls, value: str | None, info) -> str | None:
        if value is None:
            if info.field_name == "delivery_id":
                raise ValueError("delivery_id is required.")
            return None
        return require_clean_nonblank(value, info.field_name)


class InteractionTransitionResult(BaseModel):
    """Exact result of an atomic interaction-lifecycle/session transition."""

    model_config = ConfigDict(extra="forbid")

    session: Session
    event: Event
    status_changed: StrictBool
    replayed: StrictBool = False

    @field_validator("session")
    @classmethod
    def copy_session(cls, value: Session) -> Session:
        return copy_session(value)

    @field_validator("event")
    @classmethod
    def copy_event(cls, value: Event) -> Event:
        return copy_event(value)


class _InteractionTransitionReceipt(BaseModel):
    """Immutable store-owned evidence for exact lifecycle-transition replay."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    record_type: Literal["cayu.interaction-transition"] = "cayu.interaction-transition"
    schema_version: Literal[1] = 1
    session: Session
    event: Event
    from_statuses: tuple[SessionStatus, ...]
    to_status: SessionStatus
    only_if_no_queued_messages: StrictBool
    status_changed: StrictBool
    record_digest: str

    @field_validator("session")
    @classmethod
    def copy_session(cls, value: Session) -> Session:
        return copy_session(value)

    @field_validator("event")
    @classmethod
    def copy_event(cls, value: Event) -> Event:
        return copy_event(value)

    @field_validator("record_digest")
    @classmethod
    def validate_record_digest(cls, value: str) -> str:
        _require_raw_sha256_digest(value)
        return value

    @model_validator(mode="after")
    def validate_source_statuses(self) -> _InteractionTransitionReceipt:
        if not self.from_statuses:
            raise ValueError("Interaction-transition receipt requires source statuses.")
        if len(set(self.from_statuses)) != len(self.from_statuses):
            raise ValueError("Interaction-transition receipt source statuses must be unique.")
        if self.session.id != self.event.session_id:
            raise ValueError("Interaction-transition receipt session and event disagree.")
        if self.status_changed:
            if self.session.status is not self.to_status:
                raise ValueError(
                    "Interaction-transition receipt changed status does not match its session."
                )
        elif not self.only_if_no_queued_messages or self.session.status not in self.from_statuses:
            raise ValueError("Interaction-transition receipt unchanged status is inconsistent.")
        return self


class InterruptSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    requested_by: ResolutionActor | None = None

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "metadata")

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("reason")
    @classmethod
    def validate_optional_reason(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("requested_by")
    @classmethod
    def copy_requested_by(cls, value: ResolutionActor | None) -> ResolutionActor | None:
        return copy_resolution_actor(value)


class ForkSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_session_id: str
    session_id: str | None = None
    agent_name: str | None = None
    model: str | None = None
    environment_name: str | None = None
    transcript_cursor: StrictInt | None = Field(default=None, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    copy_checkpoint: StrictBool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id")
    @classmethod
    def validate_optional_destination_session_id(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return _require_bounded_session_id(value, info.field_name)

    @field_validator("source_session_id", "agent_name", "model", "environment_name")
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "metadata")


class SessionIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_name: str
    model: str
    runtime_name: str = "cayu"
    runtime_version: str | None = None

    @field_validator("provider_name", "model", "runtime_name")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("runtime_version")
    @classmethod
    def validate_optional_runtime_version(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class Session(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    # SessionStore implementations may set this from RunRequest.session_id.
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_name: str
    provider_name: str
    model: str
    parent_session_id: str | None = None
    causal_budget_id: str
    runtime_name: str = "cayu"
    runtime_version: str | None = None
    environment_name: str | None = None
    status: SessionStatus = SessionStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_activity_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_epoch: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def default_causal_budget_id(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            session_id = value.get("id")
            if session_id is None:
                session_id = str(uuid4())
                value["id"] = session_id
            if value.get("causal_budget_id") is None and isinstance(session_id, str):
                value["causal_budget_id"] = session_id
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "metadata")

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels")

    @field_validator(
        "agent_name",
        "provider_name",
        "model",
        "causal_budget_id",
        "runtime_name",
    )
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        # The byte ceiling is a creation-boundary contract. Session records may
        # predate it or come from an external store, and must remain loadable so
        # operators can inspect and migrate them.
        return require_clean_nonblank(value, info.field_name)

    @field_validator("parent_session_id", "environment_name", "runtime_version")
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class SessionStateSnapshot(BaseModel):
    """Bounded session state for status polling and control-plane coordination."""

    model_config = ConfigDict(extra="forbid")

    id: str
    status: SessionStatus
    updated_at: datetime
    last_activity_at: datetime

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "id")

    @field_validator("updated_at", "last_activity_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)


_INTERRUPTION_CASCADE_ATTEMPT_ID_MAX_CHARS = 128
_INTERRUPTION_CASCADE_CLAIM_ID_MAX_CHARS = 128
_INTERRUPTION_CASCADE_TIMESTAMP_MAX_CHARS = 64
_INTERRUPTION_CASCADE_GENERATION_MAX_CHARS = 32
_INTERRUPTION_CASCADE_MISSING = object()


def _project_interruption_cascade_marker_fields(
    marker_type: str | None,
    field_types: Mapping[str, str | None],
    field_values: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build the bounded marker view used by control-plane status polling."""

    if marker_type is None or marker_type == "null":
        return None
    if marker_type != "object":
        return {}

    projected: dict[str, Any] = {}

    def required_string(field: str, max_chars: int) -> None:
        value = field_values.get(field)
        if (
            field_types.get(field) == "string"
            and type(value) is str
            and value.strip()
            and len(value) <= max_chars
        ):
            projected[field] = value
        else:
            projected[field] = None

    def optional_string(field: str, max_chars: int) -> None:
        field_type = field_types.get(field)
        if field_type is None or field_type == "null":
            return
        value = field_values.get(field)
        if type(value) is str and field_type == "string" and len(value) <= max_chars:
            projected[field] = value
        else:
            projected[field] = 0

    required_string("attempt_id", _INTERRUPTION_CASCADE_ATTEMPT_ID_MAX_CHARS)
    projected["interrupt_payload"] = (
        {} if field_types.get("interrupt_payload") == "object" else None
    )

    generation_type = field_types.get("generation")
    if generation_type is not None:
        generation = field_values.get("generation")
        try:
            if (
                generation_type not in {"integer", "number"}
                or type(generation) is not str
                or len(generation) > _INTERRUPTION_CASCADE_GENERATION_MAX_CHARS
                or not generation.lstrip("-").isdigit()
            ):
                raise ValueError
            projected["generation"] = int(generation)
        except ValueError:
            projected["generation"] = None

    failure_type = field_types.get("failure_recorded")
    if failure_type is not None:
        failure = field_values.get("failure_recorded")
        if failure_type == "boolean" and type(failure) is bool:
            projected["failure_recorded"] = failure
        else:
            projected["failure_recorded"] = None

    optional_string("claim_id", _INTERRUPTION_CASCADE_CLAIM_ID_MAX_CHARS)
    optional_string("claim_expires_at", _INTERRUPTION_CASCADE_TIMESTAMP_MAX_CHARS)
    optional_string("created_at", _INTERRUPTION_CASCADE_TIMESTAMP_MAX_CHARS)
    return projected


def _project_interruption_cascade_marker(marker: Any) -> dict[str, Any] | None:
    if marker is _INTERRUPTION_CASCADE_MISSING or marker is None:
        return None
    if type(marker) is not dict:
        return {}

    field_types: dict[str, str | None] = {}
    field_values: dict[str, Any] = {}
    for field in (
        "attempt_id",
        "interrupt_payload",
        "generation",
        "failure_recorded",
        "claim_id",
        "claim_expires_at",
        "created_at",
    ):
        if field not in marker:
            field_types[field] = None
            continue
        value = marker[field]
        if value is None:
            field_types[field] = "null"
        elif type(value) is str:
            field_types[field] = "string"
            max_chars = {
                "attempt_id": _INTERRUPTION_CASCADE_ATTEMPT_ID_MAX_CHARS,
                "claim_id": _INTERRUPTION_CASCADE_CLAIM_ID_MAX_CHARS,
                "claim_expires_at": _INTERRUPTION_CASCADE_TIMESTAMP_MAX_CHARS,
                "created_at": _INTERRUPTION_CASCADE_TIMESTAMP_MAX_CHARS,
            }.get(field, _INTERRUPTION_CASCADE_GENERATION_MAX_CHARS)
            field_values[field] = value[: max_chars + 1]
        elif type(value) is bool:
            field_types[field] = "boolean"
            field_values[field] = value
        elif type(value) is int:
            field_types[field] = "integer"
            field_values[field] = str(value)[: _INTERRUPTION_CASCADE_GENERATION_MAX_CHARS + 1]
        elif type(value) is float:
            field_types[field] = "number"
            field_values[field] = str(value)[: _INTERRUPTION_CASCADE_GENERATION_MAX_CHARS + 1]
        elif type(value) is dict:
            field_types[field] = "object"
        elif type(value) is list:
            field_types[field] = "array"
        else:
            field_types[field] = "other"
    return _project_interruption_cascade_marker_fields("object", field_types, field_values)


CheckpointTransform = Callable[
    [Session, dict[str, Any] | None],
    dict[str, Any] | None,
]
CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS = 32


@dataclass(frozen=True)
class CheckpointRootFieldProjection:
    """Bounded opaque evidence for one runtime-selected root field."""

    is_present: bool
    json_type: str | None
    scalar_text: str | None
    scalar_text_truncated: bool = False


CheckpointRootFieldProjectionValidator = Callable[
    [str, CheckpointRootFieldProjection],
    None,
]


@dataclass(frozen=True)
class CheckpointRootFieldGuard:
    """One runtime-owned validator to run inside a store read snapshot."""

    key: str
    validate: CheckpointRootFieldProjectionValidator

    def __post_init__(self) -> None:
        key = require_clean_nonblank(self.key, "checkpoint root field key")
        if any(
            char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
            for char in key
        ):
            raise ValueError(
                "checkpoint root field key must contain only ASCII letters, digits, "
                "and underscores."
            )
        if not callable(self.validate):
            raise TypeError("checkpoint root field validator must be callable.")


def checkpoint_root_field_projection(
    checkpoint: dict[str, Any] | None,
    key: str,
) -> CheckpointRootFieldProjection:
    """Project one root field without copying arbitrary checkpoint contents."""

    if checkpoint is None or key not in checkpoint:
        return CheckpointRootFieldProjection(
            is_present=False,
            json_type=None,
            scalar_text=None,
        )
    value = checkpoint[key]
    if value is None:
        json_type = "null"
        scalar_text = None
    elif type(value) is bool:
        json_type = "boolean"
        scalar_text = None
    elif type(value) is int:
        json_type = "integer"
        scalar_text = str(value)
    elif type(value) is float:
        json_type = "number"
        scalar_text = None
    elif type(value) is str:
        json_type = "string"
        scalar_text = None
    elif type(value) is list:
        json_type = "array"
        scalar_text = None
    elif type(value) is dict:
        json_type = "object"
        scalar_text = None
    else:
        json_type = "other"
        scalar_text = None
    truncated = (
        scalar_text is not None and len(scalar_text) > CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS
    )
    return CheckpointRootFieldProjection(
        is_present=True,
        json_type=json_type,
        scalar_text=(
            None if scalar_text is None else scalar_text[:CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS]
        ),
        scalar_text_truncated=truncated,
    )


def checkpoint_root_field_projection_from_storage(
    *,
    json_type: str | None,
    scalar_text: Any,
) -> CheckpointRootFieldProjection:
    """Normalize bounded root-field evidence projected by a durable store."""

    json_type_aliases = {
        "false": "boolean",
        "real": "number",
        "text": "string",
        "true": "boolean",
    }
    normalized_json_type = (
        None if json_type is None else json_type_aliases.get(json_type, json_type)
    )
    normalized_scalar = None if scalar_text is None else str(scalar_text)
    return CheckpointRootFieldProjection(
        is_present=json_type is not None,
        json_type=normalized_json_type,
        scalar_text=(
            None
            if normalized_scalar is None
            else normalized_scalar[:CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS]
        ),
        scalar_text_truncated=(
            normalized_scalar is not None
            and len(normalized_scalar) > CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS
        ),
    )


ForkTranscriptValidator = Callable[[tuple[Message, ...]], bool]
FORK_TRANSCRIPT_VALIDATION_ERROR = (
    "Fork transcript contains a workload secret and cannot be copied without "
    "changing durable conversation history."
)


def fork_transcript_is_accepted(
    messages: list[Message],
    validator: ForkTranscriptValidator | None,
) -> bool:
    """Require explicit positive validation for the exact transcript being copied."""

    if validator is None:
        return True
    # A validator is an external callback. Give it an isolated projection so
    # mutation cannot alter either the source transcript or the messages that
    # will be committed to the fork.
    validation_messages = tuple(detach_message(message) for message in messages)
    try:
        accepted = validator(validation_messages)
    except Exception:
        return False
    finally:
        validation_messages = ()
    return type(accepted) is bool and accepted


def transform_fork_checkpoint(
    source_session: Session,
    source_checkpoint: dict[str, Any] | None,
    transform: CheckpointTransform,
) -> dict[str, Any] | None:
    """Apply a fork transform without retaining its raw input on failure."""

    try:
        return transform(source_session, source_checkpoint)
    except BaseException:
        if source_checkpoint is not None:
            source_checkpoint.clear()
        raise


RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX = "__cayu_runtime_publication_v1__:"
INTERACTION_TRANSITION_OPERATION_KEY_PREFIX = "__cayu_interaction_transition_v1__:"
RUNTIME_PUBLICATION_RECORD_TYPE = "cayu.runtime-publication"
RUNTIME_PUBLICATION_SCHEMA_VERSION = 2
RUNTIME_PUBLICATION_MAX_CHECKPOINT_OPERATIONS = 128
RUNTIME_PUBLICATION_MAX_TRANSCRIPT_MESSAGES = 512
RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS = 512
RUNTIME_PUBLICATION_MAX_TOOL_CALLS = 256
_PENDING_TOOL_ROUND_CHECKPOINT_KEY = "pending_tool_round"
_TOOL_ROUND_TERMINAL_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    }
)
_TOOL_ROUND_LIFECYCLE_EVENT_TYPES = frozenset(
    {EventType.TOOL_CALL_STARTED, *_TOOL_ROUND_TERMINAL_EVENT_TYPES}
)
_STRUCTURED_OUTPUT_AUXILIARY_SCHEMA_VERSION = 1
_STRUCTURED_OUTPUT_AUXILIARY_KIND = "structured-output-validation"
_STRUCTURED_OUTPUT_AUXILIARY_INTENT_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "step",
        "attempt",
        "valid",
        "retry_scheduled",
        "event_ids",
    }
)
RuntimePublicationKind = Literal[
    "model-step",
    "context-compaction",
    "tool-round",
    "approval-open",
    "approval-close",
]


@dataclass(frozen=True, slots=True)
class _StructuredOutputToolRoundAuxiliary:
    step: int
    attempt: int
    valid: bool
    retry_scheduled: bool
    event_ids: tuple[str, ...]
    session_id: str
    output_present: bool
    output: Any
    errors: tuple[dict[str, Any], ...]


class RuntimePublicationEventReference(BaseModel):
    """Content-bound reference to one already-durable runtime event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(max_length=EVENT_ID_MAX_CHARS)
    event_digest: str

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "event_id")

    @field_validator("event_digest")
    @classmethod
    def validate_event_digest(cls, value: str) -> str:
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("event_digest must be a lowercase SHA-256 digest.")
        return value


def _copy_runtime_publication_event_reference(
    reference: RuntimePublicationEventReference | dict[str, Any],
) -> RuntimePublicationEventReference:
    try:
        if type(reference) is RuntimePublicationEventReference:
            return RuntimePublicationEventReference(
                event_id=reference.event_id,
                event_digest=reference.event_digest,
            )
        if type(reference) is dict:
            return RuntimePublicationEventReference.model_validate(reference)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Runtime publication event reference is malformed.") from exc
    raise TypeError(
        "Runtime publication event references must be "
        "RuntimePublicationEventReference values or objects."
    )


class RuntimePublicationCheckpointOperation(BaseModel):
    """One independently fenced top-level checkpoint mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(max_length=256)
    expected_value_digest: str | None
    action: Literal["set", "delete"]
    value: Any = None

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        return require_clean_nonblank(value, "checkpoint operation key")

    @field_validator("expected_value_digest")
    @classmethod
    def validate_expected_value_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("expected_value_digest must be a lowercase SHA-256 digest.")
        return value

    @field_validator("value", mode="before")
    @classmethod
    def copy_value(cls, value: Any) -> Any:
        return copy_durable_json_value(value, "checkpoint operation value")

    @model_validator(mode="after")
    def validate_action(self) -> RuntimePublicationCheckpointOperation:
        if self.action == "delete":
            if self.expected_value_digest is None:
                raise ValueError("A delete checkpoint operation requires a present-value digest.")
            if self.value is not None:
                raise ValueError("A delete checkpoint operation cannot carry a value.")
        return self


def _copy_runtime_publication_checkpoint_operation(
    operation: RuntimePublicationCheckpointOperation | dict[str, Any],
) -> RuntimePublicationCheckpointOperation:
    try:
        if type(operation) is RuntimePublicationCheckpointOperation:
            return RuntimePublicationCheckpointOperation(
                key=operation.key,
                expected_value_digest=operation.expected_value_digest,
                action=operation.action,
                value=operation.value,
            )
        if type(operation) is dict:
            return RuntimePublicationCheckpointOperation.model_validate(operation)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Runtime publication checkpoint operation is malformed.") from exc
    raise TypeError(
        "Checkpoint operations must be RuntimePublicationCheckpointOperation values or objects."
    )


class RuntimePublicationMutation(BaseModel):
    """Deterministic checkpoint patch committed with a publication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operations: tuple[RuntimePublicationCheckpointOperation, ...] = Field(default_factory=tuple)

    @field_validator("operations", mode="before")
    @classmethod
    def copy_operations(
        cls,
        value,
    ) -> tuple[RuntimePublicationCheckpointOperation, ...]:
        if type(value) not in (list, tuple):
            raise TypeError("operations must be a list or tuple of checkpoint operations.")
        if len(value) > RUNTIME_PUBLICATION_MAX_CHECKPOINT_OPERATIONS:
            raise ValueError(
                "operations cannot contain more than "
                f"{RUNTIME_PUBLICATION_MAX_CHECKPOINT_OPERATIONS} checkpoint mutations."
            )
        copied = tuple(
            sorted(
                (_copy_runtime_publication_checkpoint_operation(operation) for operation in value),
                key=lambda operation: operation.key,
            )
        )
        keys = tuple(operation.key for operation in copied)
        if len(set(keys)) != len(keys):
            raise ValueError("operations cannot mutate the same checkpoint key more than once.")
        return copied


def _copy_runtime_publication_mutation_value(
    mutation: RuntimePublicationMutation | dict[str, Any],
) -> RuntimePublicationMutation:
    try:
        if type(mutation) is RuntimePublicationMutation:
            return RuntimePublicationMutation(
                operations=mutation.operations,
            )
        if type(mutation) is dict:
            return RuntimePublicationMutation.model_validate(mutation)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Runtime publication mutation is malformed.") from exc
    raise TypeError("Runtime publication mutation must be a RuntimePublicationMutation or object.")


class RuntimePublicationRequest(BaseModel):
    """Immutable input bundle for one logical model-step or tool-round publication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    publication_id: str = Field(max_length=256)
    kind: RuntimePublicationKind
    interaction_id: str | None = Field(default=None, max_length=256)
    intent: dict[str, Any]
    mutation: RuntimePublicationMutation
    transcript_messages: tuple[Message, ...]
    events: tuple[Event, ...]
    referenced_events: tuple[RuntimePublicationEventReference, ...] = Field(default_factory=tuple)

    @field_validator("publication_id", "kind")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("interaction_id")
    @classmethod
    def validate_optional_interaction_id(cls, value: str | None) -> str | None:
        return None if value is None else require_clean_nonblank(value, "interaction_id")

    @field_validator("intent", mode="before")
    @classmethod
    def copy_intent(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "intent")

    @field_validator("mutation", mode="before")
    @classmethod
    def copy_mutation(
        cls,
        value: RuntimePublicationMutation | dict[str, Any],
    ) -> RuntimePublicationMutation:
        return _copy_runtime_publication_mutation_value(value)

    @field_validator("transcript_messages", mode="before")
    @classmethod
    def copy_transcript_messages(cls, value) -> tuple[Message, ...]:
        if type(value) not in (list, tuple):
            raise TypeError("transcript_messages must be a list or tuple of Message values.")
        return tuple(detach_message(message) for message in value)

    @field_validator("events", mode="before")
    @classmethod
    def copy_events(cls, value) -> tuple[Event, ...]:
        if type(value) not in (list, tuple):
            raise TypeError("events must be a list or tuple of Event values.")
        copied: list[Event] = []
        seen: set[str] = set()
        for event in value:
            copied_event = copy_event(event)
            if copied_event.id in seen:
                raise ValueError(f"Runtime publication event id is duplicated: {copied_event.id}")
            seen.add(copied_event.id)
            copied.append(copied_event)
        return tuple(copied)

    @field_validator("referenced_events", mode="before")
    @classmethod
    def copy_referenced_events(
        cls,
        value,
    ) -> tuple[RuntimePublicationEventReference, ...]:
        if type(value) not in (list, tuple):
            raise TypeError("referenced_events must be a list or tuple of event references.")
        copied: list[RuntimePublicationEventReference] = []
        seen: set[str] = set()
        for reference in value:
            try:
                copied_reference = _copy_runtime_publication_event_reference(reference)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError("referenced_events contains a malformed reference.") from exc
            if copied_reference.event_id in seen:
                raise ValueError(f"Referenced event id is duplicated: {copied_reference.event_id}")
            seen.add(copied_reference.event_id)
            copied.append(copied_reference)
        return tuple(copied)

    @model_validator(mode="after")
    def validate_batch_limits(self) -> RuntimePublicationRequest:
        if len(self.transcript_messages) > RUNTIME_PUBLICATION_MAX_TRANSCRIPT_MESSAGES:
            raise ValueError(
                "transcript_messages cannot contain more than "
                f"{RUNTIME_PUBLICATION_MAX_TRANSCRIPT_MESSAGES} messages."
            )
        event_binding_count = len(self.events) + len(self.referenced_events)
        if event_binding_count > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS:
            raise ValueError(
                "events and referenced_events cannot contain more than "
                f"{RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS} total event bindings."
            )
        appended_event_ids = {event.id for event in self.events}
        referenced_event_ids = {reference.event_id for reference in self.referenced_events}
        if appended_event_ids & referenced_event_ids:
            raise ValueError("Appended and referenced runtime publication events cannot overlap.")
        return self


class RuntimePublicationReceipt(BaseModel):
    """Store-owned, insert-only proof of one atomic runtime publication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["cayu.runtime-publication"] = RUNTIME_PUBLICATION_RECORD_TYPE
    schema_version: Literal[2] = RUNTIME_PUBLICATION_SCHEMA_VERSION
    session_id: str
    publication_id: str = Field(max_length=256)
    kind: RuntimePublicationKind
    interaction_id: str | None = Field(default=None, max_length=256)
    intent: dict[str, Any]
    request_digest: str
    publication_digest: str
    checkpoint_digest: str
    transcript_digest: str
    events_digest: str
    source_status: SessionStatus
    source_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    transcript_start_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    transcript_end_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    appended_event_ids: tuple[str, ...]
    referenced_events: tuple[RuntimePublicationEventReference, ...]
    published_at: datetime

    @field_validator("session_id", "publication_id", "kind")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("interaction_id")
    @classmethod
    def validate_optional_interaction_id(cls, value: str | None) -> str | None:
        return None if value is None else require_clean_nonblank(value, "interaction_id")

    @field_validator("intent", mode="before")
    @classmethod
    def copy_intent(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "intent")

    @field_validator(
        "request_digest",
        "publication_digest",
        "checkpoint_digest",
        "transcript_digest",
        "events_digest",
    )
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.")
        return value

    @field_validator("appended_event_ids", mode="before")
    @classmethod
    def copy_event_ids(cls, value, info) -> tuple[str, ...]:
        if type(value) not in (list, tuple):
            raise TypeError(f"{info.field_name} must be a list or tuple of event ids.")
        copied = tuple(
            require_clean_nonblank(event_id, f"{info.field_name} item") for event_id in value
        )
        if len(set(copied)) != len(copied):
            raise ValueError(f"{info.field_name} must not contain duplicate event ids.")
        return copied

    @field_validator("referenced_events", mode="before")
    @classmethod
    def copy_referenced_events(
        cls,
        value,
    ) -> tuple[RuntimePublicationEventReference, ...]:
        if type(value) not in (list, tuple):
            raise TypeError("referenced_events must be a list or tuple of event references.")
        copied = tuple(_copy_runtime_publication_event_reference(item) for item in value)
        event_ids = tuple(reference.event_id for reference in copied)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("referenced_events must not contain duplicate event ids.")
        return copied

    @field_validator("published_at")
    @classmethod
    def normalize_published_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("published_at must include a timezone.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_boundaries(self) -> RuntimePublicationReceipt:
        if self.transcript_end_cursor < self.transcript_start_cursor:
            raise ValueError("transcript_end_cursor cannot precede transcript_start_cursor.")
        if (
            self.transcript_end_cursor - self.transcript_start_cursor
            > RUNTIME_PUBLICATION_MAX_TRANSCRIPT_MESSAGES
        ):
            raise ValueError("Runtime publication receipt transcript span is too large.")
        if (
            len(self.appended_event_ids) + len(self.referenced_events)
            > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS
        ):
            raise ValueError("Runtime publication receipt event binding count is too large.")
        referenced_event_ids = {reference.event_id for reference in self.referenced_events}
        if set(self.appended_event_ids) & referenced_event_ids:
            raise ValueError("Appended and referenced event ids must not overlap.")
        return self


class RuntimePublicationResult(BaseModel):
    """Result of a new runtime publication or an exact durable replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session: Session
    receipt: RuntimePublicationReceipt
    replayed: StrictBool


@dataclass(frozen=True)
class _PreparedRuntimePublication:
    session_id: str
    storage_key: str
    request: RuntimePublicationRequest
    request_digest: str
    transcript_payloads: tuple[dict[str, Any], ...]
    event_payloads: tuple[dict[str, Any], ...]
    transcript_digest: str
    events_digest: str
    expected_statuses: frozenset[SessionStatus] | None
    expected_run_epoch: int | None
    expected_transcript_cursor: int | None


MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX = "__cayu_model_completion_stage_v1__:"
MODEL_COMPLETION_STAGE_RECORD_TYPE = "cayu.model-completion-stage"
MODEL_COMPLETION_STAGE_PREPARATION_RECORD_TYPE = "cayu.model-completion-stage.prepared"
MODEL_COMPLETION_STAGE_TERMINAL_RECORD_TYPE = "cayu.model-completion-stage.completed"
MODEL_COMPLETION_ACTIVE_STAGE_RECORD_TYPE = "cayu.model-completion-stage.active"
MODEL_COMPLETION_WINNER_RECORD_TYPE = "cayu.model-completion-stage.winner"
MODEL_COMPLETION_STAGE_ABANDONMENT_RECORD_TYPE = "cayu.model-completion-stage.abandoned"
MODEL_COMPLETION_STAGE_SCHEMA_VERSION = 1
MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY = MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX + "active"
_NON_TURN_MODEL_COMPLETION_CLASSIFICATIONS = frozenset({"failed", "filtered", "invalid", "length"})
_MESSAGELESS_MODEL_COMPLETION_CLASSIFICATIONS = _NON_TURN_MODEL_COMPLETION_CLASSIFICATIONS | {
    "continue"
}
ModelCompletionPurpose = Literal["assistant-turn", "context-compaction"]


class ModelCompletionStageRequest(BaseModel):
    """Immutable identity and accounting link prepared before provider dispatch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_id: str = Field(max_length=256)
    logical_step_id: str = Field(max_length=256)
    dispatch_ordinal: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    purpose: ModelCompletionPurpose = "assistant-turn"
    intent: dict[str, Any]
    reservation_ids: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("stage_id", "logical_step_id")
    @classmethod
    def validate_stage_id(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("intent", mode="before")
    @classmethod
    def copy_intent(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "intent")

    @field_validator("reservation_ids", mode="before")
    @classmethod
    def copy_reservation_ids(cls, value) -> tuple[str, ...]:
        if type(value) not in (list, tuple):
            raise TypeError("reservation_ids must be a list or tuple of reservation ids.")
        copied = tuple(
            require_clean_nonblank(reservation_id, "reservation_ids item")
            for reservation_id in value
        )
        if len(copied) > 32:
            raise ValueError("reservation_ids cannot contain more than 32 ids.")
        if len(set(copied)) != len(copied):
            raise ValueError("reservation_ids must not contain duplicate ids.")
        return copied


class ModelCompletionStage(BaseModel):
    """Verified view assembled from append-only preparation and completion records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["cayu.model-completion-stage"] = MODEL_COMPLETION_STAGE_RECORD_TYPE
    schema_version: Literal[1] = MODEL_COMPLETION_STAGE_SCHEMA_VERSION
    session_id: str
    stage_id: str = Field(max_length=256)
    logical_step_id: str = Field(max_length=256)
    dispatch_ordinal: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    purpose: ModelCompletionPurpose
    state: Literal["in_flight", "completed"]
    intent: dict[str, Any]
    reservation_ids: tuple[str, ...]
    preparation_request_digest: str
    preparation_digest: str
    source_status: SessionStatus
    source_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    source_transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    prepared_at: datetime
    publication: RuntimePublicationRequest | None = None
    publication_material_digest: str | None = None
    completion_digest: str | None = None
    completed_at: datetime | None = None

    @field_validator("session_id", "stage_id", "logical_step_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("intent", mode="before")
    @classmethod
    def copy_intent(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "intent")

    @field_validator("reservation_ids", mode="before")
    @classmethod
    def copy_reservation_ids(cls, value) -> tuple[str, ...]:
        return ModelCompletionStageRequest.copy_reservation_ids(value)

    @field_validator(
        "preparation_request_digest",
        "preparation_digest",
        "publication_material_digest",
        "completion_digest",
    )
    @classmethod
    def validate_digest(cls, value: str | None, info) -> str | None:
        if value is None and info.field_name in {
            "publication_material_digest",
            "completion_digest",
        }:
            return None
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.")
        return value

    @field_validator("prepared_at", "completed_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must include a timezone.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_state(self) -> ModelCompletionStage:
        terminal_fields = (
            self.publication,
            self.publication_material_digest,
            self.completion_digest,
            self.completed_at,
        )
        if self.state == "in_flight" and any(value is not None for value in terminal_fields):
            raise ValueError("An in-flight model-completion stage cannot contain terminal data.")
        if self.state == "completed" and any(value is None for value in terminal_fields):
            raise ValueError("A completed model-completion stage requires all terminal data.")
        if self.completed_at is not None and self.completed_at < self.prepared_at:
            raise ValueError("completed_at cannot precede prepared_at.")
        return self


class ModelCompletionStageResult(BaseModel):
    """Result of preparing or terminally completing one immutable stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: ModelCompletionStage
    replayed: StrictBool
    dispatch_authorized: StrictBool

    @model_validator(mode="after")
    def validate_dispatch_authorization(self) -> ModelCompletionStageResult:
        expected = self.stage.state == "in_flight" and not self.replayed
        if self.dispatch_authorized != expected:
            raise ValueError(
                "dispatch_authorized must be true exactly for a newly inserted "
                "in-flight preparation."
            )
        return self


class ActiveModelCompletionStage(BaseModel):
    """Verified snapshot of the model dispatch currently eligible to publish."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: ModelCompletionStage
    activated_at: datetime
    marker_digest: str

    @field_validator("activated_at")
    @classmethod
    def normalize_activated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("activated_at must include a timezone.")
        return value.astimezone(UTC)

    @field_validator("marker_digest")
    @classmethod
    def validate_marker_digest(cls, value: str) -> str:
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("marker_digest must be a lowercase SHA-256 digest.")
        return value


class ModelCompletionStageAbandonment(BaseModel):
    """Durable proof that an exact pre-dispatch preparation was abandoned."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["cayu.model-completion-stage.abandoned"] = (
        MODEL_COMPLETION_STAGE_ABANDONMENT_RECORD_TYPE
    )
    schema_version: Literal[1] = MODEL_COMPLETION_STAGE_SCHEMA_VERSION
    session_id: str
    stage_id: str = Field(max_length=256)
    logical_step_id: str = Field(max_length=256)
    dispatch_ordinal: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    purpose: ModelCompletionPurpose
    preparation_request_digest: str
    preparation_digest: str
    active_marker_digest: str
    source_status: SessionStatus
    source_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    source_transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    abandoned_at: datetime
    abandonment_digest: str

    @field_validator("session_id", "stage_id", "logical_step_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator(
        "preparation_request_digest",
        "preparation_digest",
        "active_marker_digest",
        "abandonment_digest",
    )
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.")
        return value

    @field_validator("abandoned_at")
    @classmethod
    def normalize_abandoned_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("abandoned_at must include a timezone.")
        return value.astimezone(UTC)


class ModelCompletionStageAbandonmentResult(BaseModel):
    """Result of a new abandonment or an exact durable replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    abandonment: ModelCompletionStageAbandonment
    replayed: StrictBool


class _ModelCompletionStagePreparationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["cayu.model-completion-stage.prepared"]
    schema_version: Literal[1]
    session_id: str
    stage_id: str
    logical_step_id: str
    dispatch_ordinal: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    purpose: ModelCompletionPurpose
    intent: dict[str, Any]
    reservation_ids: tuple[str, ...]
    request_digest: str
    source_status: SessionStatus
    source_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    source_transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    prepared_at: datetime
    record_digest: str


class _ModelCompletionStageTerminalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["cayu.model-completion-stage.completed"]
    schema_version: Literal[1]
    session_id: str
    stage_id: str
    logical_step_id: str
    dispatch_ordinal: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    purpose: ModelCompletionPurpose
    preparation_digest: str
    publication: RuntimePublicationRequest
    publication_material_digest: str
    completed_at: datetime
    record_digest: str


class _ActiveModelCompletionStageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["cayu.model-completion-stage.active"]
    schema_version: Literal[1]
    session_id: str
    stage_id: str
    logical_step_id: str
    dispatch_ordinal: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    purpose: ModelCompletionPurpose
    preparation_digest: str
    source_status: SessionStatus
    source_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    source_transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    activated_at: datetime
    record_digest: str


class _ModelCompletionStageWinnerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["cayu.model-completion-stage.winner"]
    schema_version: Literal[1]
    session_id: str
    stage_id: str
    logical_step_id: str
    dispatch_ordinal: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    purpose: ModelCompletionPurpose
    preparation_digest: str
    completion_digest: str
    publication_material_digest: str
    receipt_request_digest: str
    publication_digest: str
    published_at: datetime
    record_digest: str


class _ModelCompletionStageAbandonmentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["cayu.model-completion-stage.abandoned"]
    schema_version: Literal[1]
    session_id: str
    stage_id: str
    logical_step_id: str
    dispatch_ordinal: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    purpose: ModelCompletionPurpose
    preparation_request_digest: str
    preparation_digest: str
    active_marker_digest: str
    source_status: SessionStatus
    source_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    source_transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    abandoned_at: datetime
    record_digest: str


@dataclass(frozen=True)
class _PreparedModelCompletionStage:
    session_id: str
    preparation_storage_key: str
    terminal_storage_key: str
    abandonment_storage_key: str
    winner_storage_key: str
    publication_storage_key: str
    request: ModelCompletionStageRequest
    request_digest: str
    expected_statuses: frozenset[SessionStatus]
    expected_run_epoch: int
    expected_transcript_cursor: int


@dataclass(frozen=True)
class _PreparedModelCompletionStageTerminal:
    session_id: str
    stage_id: str
    preparation_storage_key: str
    terminal_storage_key: str
    publication: RuntimePublicationRequest
    publication_material_digest: str


@dataclass(frozen=True)
class _PreparedModelCompletionStageAbandonment:
    session_id: str
    stage_id: str
    preparation_storage_key: str
    terminal_storage_key: str
    abandonment_storage_key: str
    preparation_digest: str
    expected_run_epoch: int


@dataclass(frozen=True)
class _ModelCompletionStagePromotionContext:
    stage_id: str
    preparation_storage_key: str
    terminal_storage_key: str
    active_storage_key: str
    winner_storage_key: str
    completion_digest: str


class SessionOperationPublication(BaseModel):
    """One atomic checkpoint/event publication plus terminal operation records."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    checkpoint: dict[str, Any]
    operation_records: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("checkpoint", mode="before")
    @classmethod
    def copy_checkpoint(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "checkpoint")

    @field_validator("operation_records", mode="before")
    @classmethod
    def copy_operation_records(
        cls,
        value: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        copied = copy_durable_json_object(value, "operation_records")
        for key, record in copied.items():
            _reject_reserved_runtime_publication_key(key, "operation_records key")
            if type(record) is not dict:
                raise ValueError("Session operation records must be objects.")
        return copied


SessionOperationTransform = Callable[
    [Session, dict[str, Any] | None, dict[str, Any] | None],
    SessionOperationPublication,
]


_MCP_MANIFEST_BASELINE_MAX_TOOLS = 10_000


def _require_sha256_identifier(value: str, field_name: str) -> str:
    value = require_clean_nonblank(value, field_name)
    if (
        len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{field_name} must be a SHA-256 identifier.")
    return value


def _mcp_manifest_session_ref(session_id: str) -> str:
    session_id = require_clean_nonblank(session_id, "session_id")
    encoded = json.dumps(
        {
            "schema": "cayu.mcp.accepted_session_ref.v1",
            "session_id": session_id,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _mcp_authoritative_manifest_hash(
    *,
    source_manifest_hash: str,
    server_hash: str,
    tools: Iterable[Mapping[str, str]],
    exposed_tools: Iterable[Mapping[str, str]],
) -> str:
    """Bind every durable manifest-evidence dimension into one authority hash."""

    source_manifest_hash = _require_sha256_identifier(
        source_manifest_hash,
        "source_manifest_hash",
    )
    server_hash = _require_sha256_identifier(server_hash, "server_hash")

    def canonical_evidence(
        value: Iterable[Mapping[str, str]],
        field_name: str,
    ) -> list[dict[str, str]]:
        evidence: list[dict[str, str]] = []
        tool_ids: set[str] = set()
        for item in value:
            if not isinstance(item, Mapping) or set(item) != {"tool_id", "contract_hash"}:
                raise ValueError(
                    f"{field_name} entries must contain only tool_id and contract_hash."
                )
            tool_id = _require_sha256_identifier(item["tool_id"], f"{field_name} tool_id")
            contract_hash = _require_sha256_identifier(
                item["contract_hash"],
                f"{field_name} contract_hash",
            )
            if tool_id in tool_ids:
                raise ValueError(f"{field_name} must not contain duplicate tool_id values.")
            tool_ids.add(tool_id)
            evidence.append(
                {
                    "tool_id": tool_id,
                    "contract_hash": contract_hash,
                }
            )
        evidence.sort(key=lambda entry: entry["tool_id"])
        return evidence

    encoded = json.dumps(
        {
            "schema": "cayu.mcp.authorized_manifest.v2",
            "source_manifest_hash": source_manifest_hash,
            "server_hash": server_hash,
            "tools": canonical_evidence(tools, "tools"),
            "exposed_tools": canonical_evidence(exposed_tools, "exposed_tools"),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class McpManifestBaseline(BaseModel):
    """Authoritative accepted MCP manifest for one durable history identity.

    ``tools`` and ``exposed_tools`` contain only fixed-size ``tool_id`` and
    ``contract_hash`` SHA-256 evidence. Raw MCP or Cayu tool names do not cross
    this durable boundary.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    history_key: str
    generation: StrictInt = Field(ge=1)
    manifest_identity: str
    manifest_hash: str
    source_manifest_hash: str
    server_hash: str
    tools: tuple[dict[str, Any], ...] = Field(
        default_factory=tuple,
        max_length=_MCP_MANIFEST_BASELINE_MAX_TOOLS,
    )
    exposed_tools: tuple[dict[str, Any], ...] = Field(
        default_factory=tuple,
        max_length=_MCP_MANIFEST_BASELINE_MAX_TOOLS,
    )
    accepted_session_ref: str
    accepted_event_id: str = Field(max_length=EVENT_ID_MAX_CHARS)
    accepted_at: datetime

    @field_validator(
        "history_key",
        "manifest_identity",
        "manifest_hash",
        "source_manifest_hash",
        "server_hash",
        "accepted_session_ref",
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _require_sha256_identifier(value, info.field_name)

    @field_validator("accepted_event_id")
    @classmethod
    def validate_reference_ids(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("tools", "exposed_tools", mode="before")
    @classmethod
    def copy_tools(cls, value, info) -> tuple[dict[str, Any], ...]:
        field_name = info.field_name
        if not isinstance(value, list | tuple):
            raise TypeError(f"{field_name} must be a list or tuple.")
        if len(value) > _MCP_MANIFEST_BASELINE_MAX_TOOLS:
            raise ValueError(
                f"{field_name} must not contain more than "
                f"{_MCP_MANIFEST_BASELINE_MAX_TOOLS} entries."
            )
        copied = copy_durable_json_value(list(value), field_name)
        if type(copied) is not list or any(type(item) is not dict for item in copied):
            raise TypeError(f"{field_name} must contain JSON objects.")
        tool_ids: set[str] = set()
        for item in copied:
            if set(item) != {"tool_id", "contract_hash"}:
                raise ValueError(
                    f"{field_name} entries must contain only tool_id and contract_hash."
                )
            for item_field_name in ("tool_id", "contract_hash"):
                item[item_field_name] = _require_sha256_identifier(
                    item[item_field_name],
                    f"{info.field_name} {item_field_name}",
                )
            tool_id = item["tool_id"]
            if tool_id in tool_ids:
                raise ValueError(f"{info.field_name} must not contain duplicate tool_id values.")
            tool_ids.add(tool_id)
        return tuple(copied)

    @field_validator("accepted_at")
    @classmethod
    def normalize_accepted_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("accepted_at must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_manifest_evidence(self) -> McpManifestBaseline:
        authoritative_hash = _mcp_authoritative_manifest_hash(
            source_manifest_hash=self.source_manifest_hash,
            server_hash=self.server_hash,
            tools=self.tools,
            exposed_tools=self.exposed_tools,
        )
        if self.manifest_hash != authoritative_hash:
            raise ValueError(
                "manifest_hash does not match the authoritative MCP manifest evidence."
            )
        return self


class McpManifestBaselineLoadResult(BaseModel):
    """Authoritative accepted baselines loaded from one durable store."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baselines: dict[str, McpManifestBaseline] = Field(default_factory=dict)

    @field_validator("baselines", mode="before")
    @classmethod
    def copy_baselines(cls, value) -> dict[str, McpManifestBaseline]:
        if type(value) is not dict:
            raise TypeError("baselines must be a dict.")
        return {
            _require_sha256_identifier(key, "baselines key"): McpManifestBaseline.model_validate(
                baseline.model_dump(mode="python")
                if isinstance(baseline, McpManifestBaseline)
                else baseline
            )
            for key, baseline in value.items()
        }

    @model_validator(mode="after")
    def validate_baseline_keys(self) -> McpManifestBaselineLoadResult:
        for key, baseline in self.baselines.items():
            if baseline.history_key != key:
                raise ValueError("MCP baseline history_key does not match its load-result key.")
        return self


class McpManifestPublicationResult(BaseModel):
    """Result of one atomic manifest-baseline compare-and-publication attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    published: StrictBool
    baselines: dict[str, McpManifestBaseline] = Field(default_factory=dict)

    @field_validator("baselines", mode="before")
    @classmethod
    def copy_baselines(cls, value) -> dict[str, McpManifestBaseline]:
        if type(value) is not dict:
            raise TypeError("baselines must be a dict.")
        return {
            _require_sha256_identifier(
                key,
                "baselines key",
            ): McpManifestBaseline.model_validate(
                baseline.model_dump(mode="python")
                if isinstance(baseline, McpManifestBaseline)
                else baseline
            )
            for key, baseline in value.items()
        }

    @model_validator(mode="after")
    def validate_baseline_keys(self) -> McpManifestPublicationResult:
        for key, baseline in self.baselines.items():
            if baseline.history_key != key:
                raise ValueError(
                    "MCP baseline history_key does not match its publication-result key."
                )
        return self


class SessionOrder(StrEnum):
    CREATED_AT_ASC = "created_at_asc"
    CREATED_AT_DESC = "created_at_desc"
    UPDATED_AT_ASC = "updated_at_asc"
    UPDATED_AT_DESC = "updated_at_desc"
    LAST_ACTIVITY_AT_ASC = "last_activity_at_asc"
    LAST_ACTIVITY_AT_DESC = "last_activity_at_desc"


class EventOrder(StrEnum):
    SEQUENCE_ASC = "sequence_asc"
    SEQUENCE_DESC = "sequence_desc"


class PendingActionKind(StrEnum):
    TOOL_APPROVAL = "tool_approval"
    USER_INPUT = "user_input"
    MANUAL_RECOVERY = "manual_recovery"


class PendingActionIssueCode(StrEnum):
    """Why one pending-action candidate could not be projected safely."""

    SOURCE_TOO_LARGE = "source_too_large"
    SOURCE_TOO_COMPLEX = "source_too_complex"
    SOURCE_INVALID = "source_invalid"


DEFAULT_PENDING_ACTION_RESULT_MAX_BYTES = 2 * 1024 * 1024
MAX_PENDING_ACTION_RESULT_BYTES = 16 * 1024 * 1024
# A model/runtime should never produce hundreds of calls in one tool round. This
# cap is also a storage-safety boundary: SQL stores inspect the count before
# expanding checkpoint call identifiers into rows.
MAX_PENDING_ACTION_TOOL_CALLS = 256
# A normal call contributes one start and one terminal event. Keep enough room
# for repeated crashes and explicit reconciliation while bounding corrupted or
# adversarial evidence independently for every call in the pending round.
MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL = 16


class PendingActionResultTooLarge(RuntimeError):
    """A pending-action page exceeded its caller-selected serialized byte ceiling."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        super().__init__(
            f"Pending-action response exceeds the {max_bytes}-byte result limit. "
            "Request a smaller page or inspect the session directly."
        )


PENDING_ACTION_EVENT_TYPE_VALUES = frozenset(
    {
        "tool.call.approval_requested",
        "session.awaiting_user_input",
        "session.interrupted",
        "session.resumed",
        "session.completed",
        "session.failed",
        "tool.call.started",
        "tool.call.completed",
        "tool.call.failed",
        "tool.call.blocked",
        "tool.call.approval_denied",
    }
)
PENDING_ACTION_BARRIER_EVENT_TYPE_VALUES = frozenset(
    {"session.resumed", "session.completed", "session.failed"}
)


class LabelSelectorOperator(StrEnum):
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    IN = "in"
    NOT_IN = "not_in"


class LabelSelectorRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    operator: LabelSelectorOperator
    values: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        return next(iter(copy_label_map({value: "_"}, "label selector").keys()))

    @field_validator("values", mode="before")
    @classmethod
    def copy_values(cls, value) -> tuple[str, ...]:
        if value is None:
            return ()
        if type(value) is str:
            raise ValueError("`values` must be a sequence of strings.")
        values = tuple(value)
        copied: list[str] = []
        for index, item in enumerate(values):
            if type(item) is not str:
                raise ValueError("`values` must contain only strings.")
            copied_value = next(
                iter(copy_label_map({f"value_{index}": item}, "label selector value").values())
            )
            if copied_value in copied:
                raise ValueError("`values` must not contain duplicates.")
            copied.append(copied_value)
        return tuple(copied)

    @model_validator(mode="after")
    def validate_operator_values(self) -> LabelSelectorRequirement:
        if self.operator in {LabelSelectorOperator.EXISTS, LabelSelectorOperator.NOT_EXISTS}:
            if self.values:
                raise ValueError(f"`{self.operator}` label selector must not include values.")
        elif not self.values:
            raise ValueError(f"`{self.operator}` label selector requires at least one value.")
        return self


def copy_label_selector_requirements(
    value: Any,
    field_name: str = "label_selectors",
) -> tuple[LabelSelectorRequirement, ...]:
    if value is None:
        return ()
    if type(value) is LabelSelectorRequirement:
        return (value.model_copy(deep=True),)
    if type(value) in {str, dict}:
        raise ValueError(f"`{field_name}` must be a sequence of label selector requirements.")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(
            f"`{field_name}` must be a sequence of label selector requirements."
        ) from exc
    return tuple(
        item.model_copy(deep=True)
        if type(item) is LabelSelectorRequirement
        else LabelSelectorRequirement.model_validate(item)
        for item in values
    )


class SessionQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    q: str | None = None
    status: SessionStatus | None = None
    debug_state: SessionDebugState | None = None
    agent_name: str | None = None
    provider_name: str | None = None
    model: str | None = None
    environment_name: str | None = None
    parent_session_id: str | None = None
    causal_budget_id: str | None = None
    last_activity_before: datetime | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    label_selectors: tuple[LabelSelectorRequirement, ...] = Field(default_factory=tuple)
    limit: StrictInt = Field(default=100, ge=1, le=1000)
    offset: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    cursor: str | None = None
    include_total_count: StrictBool = False
    order_by: SessionOrder = SessionOrder.UPDATED_AT_DESC

    @field_validator("cursor")
    @classmethod
    def validate_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_bounded_session_list_cursor(value, "cursor")

    @model_validator(mode="after")
    def reject_cursor_with_offset(self) -> SessionQuery:
        # A keyset cursor and offset are two different paging schemes; combining them
        # would silently ignore the offset, so reject it explicitly.
        if self.cursor is not None and self.offset:
            raise ValueError("cursor and a non-zero offset cannot be combined.")
        return self

    @field_validator(
        "q",
        "agent_name",
        "provider_name",
        "model",
        "environment_name",
        "parent_session_id",
        "causal_budget_id",
    )
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("last_activity_before")
    @classmethod
    def validate_last_activity_before(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("last_activity_before must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("labels", mode="before")
    @classmethod
    def copy_query_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels")

    @field_validator("label_selectors", mode="before")
    @classmethod
    def copy_query_label_selectors(cls, value) -> tuple[LabelSelectorRequirement, ...]:
        return copy_label_selector_requirements(value)


MAX_AGGREGATE_LABEL_FILTERS = 50
MAX_AGGREGATE_LABEL_SELECTORS = 25
MAX_AGGREGATE_LABEL_SELECTOR_VALUES = 100


class SessionAggregateFilter(BaseModel):
    """Current session attributes that may scope a store-native aggregate."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    agent_name: str | None = None
    provider_name: str | None = None
    model: str | None = None
    environment_name: str | None = None
    parent_session_id: str | None = None
    causal_budget_id: str | None = None
    labels: dict[str, str] = Field(
        default_factory=dict,
        max_length=MAX_AGGREGATE_LABEL_FILTERS,
    )
    label_selectors: tuple[LabelSelectorRequirement, ...] = Field(
        default_factory=tuple,
        max_length=MAX_AGGREGATE_LABEL_SELECTORS,
    )

    @field_validator(
        "agent_name",
        "provider_name",
        "model",
        "environment_name",
        "parent_session_id",
        "causal_budget_id",
    )
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("labels", mode="before")
    @classmethod
    def copy_filter_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels")

    @field_validator("label_selectors", mode="before")
    @classmethod
    def copy_filter_label_selectors(cls, value) -> tuple[LabelSelectorRequirement, ...]:
        return copy_label_selector_requirements(value)

    @model_validator(mode="after")
    def validate_selector_value_count(self) -> SessionAggregateFilter:
        value_count = sum(len(selector.values) for selector in self.label_selectors)
        if value_count > MAX_AGGREGATE_LABEL_SELECTOR_VALUES:
            raise ValueError(
                "Aggregate label selectors cannot contain more than "
                f"{MAX_AGGREGATE_LABEL_SELECTOR_VALUES} total values."
            )
        return self


class SessionStatusCounts(BaseModel):
    """Complete current-session counts for every lifecycle status."""

    model_config = ConfigDict(extra="forbid")

    pending: AggregateCount = Field(ge=0)
    running: AggregateCount = Field(ge=0)
    interrupting: AggregateCount = Field(ge=0)
    completed: AggregateCount = Field(ge=0)
    failed: AggregateCount = Field(ge=0)
    interrupted: AggregateCount = Field(ge=0)


class SessionOperationalSnapshot(BaseModel):
    """Exact current session counts captured by one store-local read snapshot."""

    model_config = ConfigDict(extra="forbid")

    as_of: datetime
    total_count: AggregateCount = Field(ge=0)
    counts_by_status: SessionStatusCounts
    accuracy: AggregateAccuracy

    @field_validator("as_of")
    @classmethod
    def normalize_as_of(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_total(self) -> SessionOperationalSnapshot:
        if sum(self.counts_by_status.model_dump().values()) != self.total_count:
            raise ValueError("Session status counts must sum to total_count.")
        return self


MAX_USAGE_ROLLUP_WINDOW = timedelta(days=366)


class UsageRollupQuery(BaseModel):
    """Bounded event-time usage query with explicit current-session filtering."""

    model_config = ConfigDict(extra="forbid")

    start_at: datetime
    end_at: datetime
    sessions: SessionAggregateFilter = Field(default_factory=SessionAggregateFilter)
    group_limit: StrictInt = Field(default=20, ge=1, le=100)
    session_group_limit: StrictInt | None = Field(default=None, ge=1, le=100)
    include_pricing_inputs: StrictBool = False
    pricing_input_limit: StrictInt = Field(default=1000, ge=1, le=5000)

    @field_validator("start_at", "end_at")
    @classmethod
    def normalize_window_timestamp(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_window(self) -> UsageRollupQuery:
        if self.start_at >= self.end_at:
            raise ValueError("Usage rollup start_at must be before end_at.")
        if self.end_at - self.start_at > MAX_USAGE_ROLLUP_WINDOW:
            raise ValueError(
                f"Usage rollup window cannot exceed {MAX_USAGE_ROLLUP_WINDOW.days} days."
            )
        return self


class EventRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    event: Event

    @field_validator("event")
    @classmethod
    def copy_event(cls, value: Event) -> Event:
        return copy_event(value)


class PersistedEventSideEffectStatus(StrEnum):
    PENDING = "pending"
    LEASED = "leased"
    DELIVERED = "delivered"
    FAILED = "failed"
    DEAD_LETTERED = "dead_lettered"


class PersistedEventSideEffectClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    event_id: str
    event_sequence: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    event: Event
    attempt: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    claim_id: str = Field(default_factory=lambda: str(uuid4()))
    lease_expires_at: datetime

    @field_validator("session_id", "event_id", "claim_id")
    @classmethod
    def validate_clean_strings(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("event")
    @classmethod
    def copy_claim_event(cls, value: Event) -> Event:
        return copy_event(value)

    @field_validator("lease_expires_at")
    @classmethod
    def normalize_lease_expires_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("lease_expires_at must be timezone-aware.")
        return value.astimezone(UTC)


class PersistedEventSideEffectDelivery(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    session_id: str
    event_id: str
    event_sequence: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    status: PersistedEventSideEffectStatus
    attempts: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    claim_id: str | None = None
    lease_expires_at: datetime | None = None
    next_attempt_at: datetime | None = None
    last_error: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("session_id", "event_id", "claim_id", "last_error")
    @classmethod
    def validate_optional_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        if info.field_name == "last_error":
            # Stores validate every new write strictly. Normalize legacy rows on
            # reconstruction so an older unbounded diagnostic cannot wedge the
            # side-effect recovery queue after an upgrade.
            return portable_persisted_event_side_effect_error(value)
        return require_clean_nonblank(value, info.field_name)

    @field_validator("lease_expires_at", "next_attempt_at", "updated_at")
    @classmethod
    def normalize_delivery_timestamp(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)


class PendingActionQuery(BaseModel):
    """Bounded query for durable control-plane actions blocking a session."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    session_id: str | None = None
    kind: PendingActionKind | None = None
    agent_name: str | None = None
    environment_name: str | None = None
    q: str | None = None
    cursor: str | None = None
    limit: StrictInt = Field(default=50, ge=1, le=200)
    max_result_bytes: StrictInt = Field(
        default=DEFAULT_PENDING_ACTION_RESULT_MAX_BYTES,
        ge=1024,
        le=MAX_PENDING_ACTION_RESULT_BYTES,
    )

    @field_validator(
        "session_id",
        "agent_name",
        "environment_name",
        "q",
        "cursor",
    )
    @classmethod
    def validate_optional_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class PendingActionSession(BaseModel):
    """Bounded session identity embedded in pending-action query results."""

    model_config = ConfigDict(extra="forbid")

    id: str
    agent_name: str
    provider_name: str
    model: str
    parent_session_id: str | None = None
    causal_budget_id: str
    runtime_name: str
    runtime_version: str | None = None
    environment_name: str | None = None
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    labels: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_session(cls, session: Session) -> PendingActionSession:
        return cls(
            id=session.id,
            agent_name=session.agent_name,
            provider_name=session.provider_name,
            model=session.model,
            parent_session_id=session.parent_session_id,
            causal_budget_id=session.causal_budget_id,
            runtime_name=session.runtime_name,
            runtime_version=session.runtime_version,
            environment_name=session.environment_name,
            status=session.status,
            created_at=session.created_at,
            updated_at=session.updated_at,
            labels=session.labels,
        )

    @field_validator(
        "id",
        "agent_name",
        "provider_name",
        "model",
        "causal_budget_id",
        "runtime_name",
    )
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("parent_session_id", "environment_name", "runtime_version")
    @classmethod
    def validate_optional_nonblank_fields(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("created_at", "updated_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value: dict[str, str]) -> dict[str, str]:
        return copy_label_map(value, "labels")


class PendingActionRecord(BaseModel):
    """One current action derived from a session checkpoint and its source event."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: PendingActionKind
    session: PendingActionSession
    event: EventRecord
    title: str
    detail: str | None = None
    tool_name: str | None = None
    approval_id: str | None = None
    input_id: str | None = None
    round_id: str | None = None
    tool_call_id: str | None = None
    source_linkage: dict[str, str] = Field(default_factory=dict, exclude=True, repr=False)
    policy_evidence: ToolPolicyEvidence | None = None
    question: str | None = None
    options: list[str] = Field(default_factory=list)
    arguments: dict[str, Any] | None = None

    @field_validator("id", "title")
    @classmethod
    def validate_required_strings(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator(
        "detail",
        "tool_name",
        "approval_id",
        "input_id",
        "round_id",
        "tool_call_id",
        "question",
    )
    @classmethod
    def validate_optional_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)

    @field_validator("session")
    @classmethod
    def copy_session(cls, value: PendingActionSession) -> PendingActionSession:
        return value.model_copy(deep=True)

    @field_validator("event")
    @classmethod
    def copy_event_record(cls, value: EventRecord) -> EventRecord:
        return value.model_copy(deep=True)

    @field_validator("source_linkage", mode="before")
    @classmethod
    def copy_source_linkage(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if type(value) is not dict:
            raise TypeError("source_linkage must be a mapping of event fields to strings.")
        allowed = {"approval_id", "input_id", "tool_round_id", "tool_call_id"}
        if not set(value) <= allowed:
            raise ValueError("source_linkage contains an unsupported event field.")
        return {
            field_name: require_nonblank(field_value, f"source_linkage.{field_name}")
            for field_name, field_value in value.items()
        }

    @field_validator("options", mode="before")
    @classmethod
    def copy_options(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if type(value) is not list:
            raise TypeError("options must be a list of strings.")
        return [require_nonblank(item, "options") for item in value]

    @field_validator("arguments", mode="before")
    @classmethod
    def copy_arguments(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        return copy_durable_json_object(value, "arguments")


class PendingActionIssue(BaseModel):
    """Bounded visibility for a candidate that could not be materialized."""

    model_config = ConfigDict(extra="forbid")

    code: PendingActionIssueCode
    session_id: str
    agent_name: str
    status: SessionStatus
    updated_at: datetime
    detail: str

    @field_validator("session_id", "agent_name", "detail")
    @classmethod
    def validate_required_strings(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("updated_at")
    @classmethod
    def normalize_updated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("updated_at must be timezone-aware.")
        return value.astimezone(UTC)

    @classmethod
    def source_too_large(
        cls,
        session: PendingActionSession,
        *,
        max_bytes: int,
    ) -> PendingActionIssue:
        return cls(
            code=PendingActionIssueCode.SOURCE_TOO_LARGE,
            session_id=session.id,
            agent_name=session.agent_name,
            status=session.status,
            updated_at=session.updated_at,
            detail=(
                "Pending-action source data for this session exceeds the "
                f"{max_bytes}-byte inspection limit. Open the session directly "
                "to inspect or resolve it."
            ),
        )

    @classmethod
    def source_too_complex(
        cls,
        session: PendingActionSession,
        *,
        max_tool_calls: int,
    ) -> PendingActionIssue:
        return cls(
            code=PendingActionIssueCode.SOURCE_TOO_COMPLEX,
            session_id=session.id,
            agent_name=session.agent_name,
            status=session.status,
            updated_at=session.updated_at,
            detail=(
                "Pending tool-round state for this session exceeds the "
                f"{max_tool_calls}-call inspection limit. Open the session directly "
                "to inspect or resolve it."
            ),
        )

    @classmethod
    def ledger_too_complex(
        cls,
        session: PendingActionSession,
        *,
        max_events_per_call: int,
    ) -> PendingActionIssue:
        return cls(
            code=PendingActionIssueCode.SOURCE_TOO_COMPLEX,
            session_id=session.id,
            agent_name=session.agent_name,
            status=session.status,
            updated_at=session.updated_at,
            detail=(
                "Pending tool-round evidence for this session exceeds the "
                f"{max_events_per_call}-event per-call inspection limit. "
                "Open the session directly to inspect or resolve it."
            ),
        )

    @classmethod
    def source_invalid(cls, session: PendingActionSession) -> PendingActionIssue:
        return cls(
            code=PendingActionIssueCode.SOURCE_INVALID,
            session_id=session.id,
            agent_name=session.agent_name,
            status=session.status,
            updated_at=session.updated_at,
            detail=(
                "Pending-action state for this session is incomplete or inconsistent. "
                "Inspect the session directly before attempting to resume it."
            ),
        )


class PendingActionListResult(BaseModel):
    """One stable page of pending actions and its candidate continuation cursor."""

    model_config = ConfigDict(extra="forbid")

    actions: list[PendingActionRecord] = Field(default_factory=list)
    issues: list[PendingActionIssue] = Field(default_factory=list)
    next_cursor: str | None = None
    has_more: StrictBool = False
    total_count: StrictInt | None = Field(default=None, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    inspected_candidate_count: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)


def enforce_pending_action_result_size(
    result: PendingActionListResult,
    *,
    max_bytes: int,
) -> PendingActionListResult:
    """Return ``result`` or fail before an oversized API body is serialized."""
    if not json_utf8_size_within_limit(result, max_bytes):
        raise PendingActionResultTooLarge(max_bytes)
    return result


class EventSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    total_events: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    counts_by_type: dict[str, StrictInt] = Field(default_factory=dict)
    latest_event: EventRecord | None = None

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "session_id")


class SerializedRecordSummary(BaseModel):
    """Exact serialized-size totals for one kind of durable session record."""

    model_config = ConfigDict(extra="forbid")

    record_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    total_bytes: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    largest_record_bytes: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)


class SessionInspectionIdentity(BaseModel):
    """Bounded, metadata-free identity used by operator inspection."""

    model_config = ConfigDict(extra="forbid")

    id: str
    agent_name: str
    provider_name: str
    model: str
    parent_session_id: str | None
    causal_budget_id: str
    runtime_name: str
    runtime_version: str | None
    environment_name: str | None
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime
    run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    labels: dict[str, str] = Field(default_factory=dict)
    label_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    labels_truncated: StrictBool = False


class SessionInspectionUsageSummary(BaseModel):
    """Lossless usage totals for bounded operator inspection."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    session_id: str
    model_steps: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    tool_calls: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    provider_names: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    usage: AggregateUsageMetrics = Field(default_factory=build_aggregate_usage_metrics)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("provider_names", "models", mode="before")
    @classmethod
    def copy_string_lists(cls, value: list[str], info) -> list[str]:
        copied = copy_durable_json_value(value, info.field_name)
        if type(copied) is not list:
            raise ValueError(f"{info.field_name} must be a list.")
        result: list[str] = []
        for index, item in enumerate(copied):
            if type(item) is not str:
                raise ValueError(f"{info.field_name}[{index}] must be a string.")
            result.append(require_clean_nonblank(item, f"{info.field_name}[{index}]"))
        return result


class SessionInspectionSummary(BaseModel):
    """Bounded backend-neutral diagnostic overview for one durable session."""

    model_config = ConfigDict(extra="forbid")

    session: SessionInspectionIdentity
    transcript: SerializedRecordSummary
    events: SerializedRecordSummary
    usage: SessionInspectionUsageSummary
    model_calls: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    model_calls_with_usage: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    tool_calls: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    pending_action_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    pending_action_kinds: tuple[PendingActionKind, ...] = ()
    pending_action_issue_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    queued_message_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    delivered_message_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    outstanding_message_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    operation_event_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    terminal_failure_state: Literal["none", "failed", "interrupted"]
    budget: SessionBudgetInspection


_SESSION_INSPECTION_MAX_RECORDS = 100_000
_SESSION_INSPECTION_MAX_RETAINED_EVENT_BYTES = 64 * 1024 * 1024
_SESSION_INSPECTION_PAGE_SIZE = 200
SESSION_INSPECTION_LABEL_LIMIT = 200


def _bounded_session_inspection_labels(
    labels: dict[str, str],
) -> tuple[dict[str, str], int, bool]:
    label_count = len(labels)
    retained_keys = heapq.nsmallest(SESSION_INSPECTION_LABEL_LIMIT, labels)
    return (
        {key: labels[key] for key in retained_keys},
        label_count,
        label_count > len(retained_keys),
    )


def _retain_session_inspection_event(current_bytes: int, event: Event) -> int:
    retained_bytes = current_bytes + compact_json_utf8_size(event.model_dump(mode="json"))
    if retained_bytes > _SESSION_INSPECTION_MAX_RETAINED_EVENT_BYTES:
        raise ValueError(
            "Session inspection exceeds the retained-event safety limit of "
            f"{_SESSION_INSPECTION_MAX_RETAINED_EVENT_BYTES} bytes."
        )
    return retained_bytes


class SessionOutcome(BaseModel):
    """Derived reason for the current session state.

    The outcome is computed from durable events. It is intentionally not stored
    as separate state so event replay remains the source of truth.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: SessionStatus
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)
    retry: dict[str, Any] | None = None
    terminal_event: EventRecord | None = None
    latest_retry_event: EventRecord | None = None

    @field_validator("session_id", "reason")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("details", mode="before")
    @classmethod
    def copy_details(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "details")

    @field_validator("retry", mode="before")
    @classmethod
    def copy_retry(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        return copy_durable_json_object(value, "retry")


class IncompleteSessionRecoveryAction(StrEnum):
    SKIPPED_ACTIVE = "skipped_active"
    SKIPPED_TERMINAL = "skipped_terminal"
    SKIPPED_UNREGISTERED_AGENT = "skipped_unregistered_agent"
    REPAIRED_TERMINAL_EVIDENCE = "repaired_terminal_evidence"
    PENDING_APPROVAL = "pending_approval"
    PENDING_USER_INPUT = "pending_user_input"
    REPAIRED_TOOL_ROUND = "repaired_tool_round"
    INTERRUPTED_ABANDONED = "interrupted_abandoned"
    FINALIZED_INTERRUPT = "finalized_interrupt"
    FAILED = "failed"


class IncompleteSessionRecoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    inactive_before: datetime | None = None
    reason: str = "worker_recovered_incomplete_session"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "reason")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("inactive_before")
    @classmethod
    def validate_inactive_before(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("inactive_before must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "metadata")


class IncompleteSessionsRecoveryRequest(BaseModel):
    """Select and bound one resumable incomplete-session recovery page."""

    model_config = ConfigDict(extra="forbid")

    statuses: set[SessionStatus]
    limit: StrictInt = Field(default=100, ge=1, le=1000)
    inspection_limit: StrictInt = Field(default=1000, ge=1, le=10_000)
    cursor: str | None = None
    inactive_before: datetime | None = None
    reason: str = "worker_recovered_incomplete_session"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("statuses", mode="before")
    @classmethod
    def copy_statuses(cls, value) -> set[SessionStatus]:
        if value is None:
            raise ValueError("statuses is required for batch incomplete-session recovery.")
        if not isinstance(value, (set, list, tuple)):
            raise ValueError("statuses must be a set of SessionStatus values.")
        statuses: set[SessionStatus] = set()
        for status in value:
            if not isinstance(status, SessionStatus):
                status = SessionStatus(status)
            statuses.add(status)
        if not statuses:
            raise ValueError("statuses must not be empty.")
        recoverable_statuses = {
            SessionStatus.PENDING,
            SessionStatus.RUNNING,
            SessionStatus.INTERRUPTING,
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.INTERRUPTED,
        }
        unsupported_statuses = statuses - recoverable_statuses
        if unsupported_statuses:
            unsupported = ", ".join(sorted(status.value for status in unsupported_statuses))
            supported = ", ".join(sorted(status.value for status in recoverable_statuses))
            raise ValueError(
                f"statuses contains unsupported recovery status values: {unsupported}. "
                f"Supported values are: {supported}."
            )
        return statuses

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return require_clean_nonblank(value, "reason")

    @field_validator("cursor")
    @classmethod
    def validate_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = require_clean_nonblank(value, "cursor")
        if len(value.encode("utf-8")) > MAX_INCOMPLETE_SESSIONS_RECOVERY_CURSOR_BYTES:
            raise ValueError(
                f"cursor exceeds its {MAX_INCOMPLETE_SESSIONS_RECOVERY_CURSOR_BYTES}-byte limit."
            )
        return value

    @field_validator("inactive_before")
    @classmethod
    def validate_inactive_before(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("inactive_before must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "metadata")


class IncompleteSessionRecoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    previous_status: SessionStatus
    status: SessionStatus
    actions: tuple[IncompleteSessionRecoveryAction, ...]
    events: tuple[Event, ...] = Field(default_factory=tuple)
    pending_approval_id: str | None = None
    pending_user_input_id: str | None = None
    message: str

    @field_validator("session_id", "message")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("events")
    @classmethod
    def copy_events(cls, value) -> tuple[Event, ...]:
        return tuple(copy_event(event) for event in value)


class IncompleteSessionsRecoveryPage(BaseModel):
    """One bounded batch-recovery page and its continuation position."""

    model_config = ConfigDict(extra="forbid")

    results: tuple[IncompleteSessionRecoveryResult, ...] = Field(
        default_factory=tuple,
        max_length=1000,
    )
    inspected_session_count: StrictInt = Field(
        default=0,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    next_cursor: str | None = None

    @field_validator("results")
    @classmethod
    def copy_results(cls, value) -> tuple[IncompleteSessionRecoveryResult, ...]:
        return tuple(result.model_copy(deep=True) for result in value)

    @field_validator("next_cursor")
    @classmethod
    def validate_next_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = require_clean_nonblank(value, "next_cursor")
        if len(value.encode("utf-8")) > MAX_INCOMPLETE_SESSIONS_RECOVERY_CURSOR_BYTES:
            raise ValueError(
                "next_cursor exceeds its "
                f"{MAX_INCOMPLETE_SESSIONS_RECOVERY_CURSOR_BYTES}-byte limit."
            )
        return value


class EventQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    session_id: str | None = None
    session_ids: tuple[str, ...] = Field(default_factory=tuple)
    event_id: str | None = None
    interaction_id: str | None = None
    causal_budget_id: str | None = None
    event_type: EventType | str | None = None
    event_types: tuple[EventType | str, ...] = Field(default_factory=tuple)
    exclude_event_types: tuple[EventType | str, ...] = Field(default_factory=tuple)
    agent_name: str | None = None
    environment_name: str | None = None
    workflow_name: str | None = None
    workflow_attempt_id: str | None = None
    workflow_step_id: str | None = None
    workflow_attempt_fenced: StrictBool = False
    tool_name: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    after_sequence: StrictInt | None = Field(default=None, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    before_sequence: StrictInt | None = Field(default=None, ge=1, le=MAX_DURABLE_JSON_INTEGER)
    limit: StrictInt = Field(default=100, ge=1, le=5000)
    order_by: EventOrder = EventOrder.SEQUENCE_ASC

    @field_validator(
        "session_id",
        "event_id",
        "interaction_id",
        "causal_budget_id",
        "agent_name",
        "environment_name",
        "workflow_name",
        "workflow_attempt_id",
        "workflow_step_id",
        "tool_name",
    )
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("session_ids", mode="before")
    @classmethod
    def copy_session_ids(cls, value) -> tuple[str, ...]:
        if value is None:
            return ()
        if type(value) is str:
            raise ValueError("`session_ids` must be a sequence of strings.")
        values = tuple(value)
        copied: list[str] = []
        for index, item in enumerate(values):
            clean_item = require_clean_nonblank(item, f"session_ids[{index}]")
            if clean_item in copied:
                raise ValueError("`session_ids` must not contain duplicates.")
            copied.append(clean_item)
        return tuple(copied)

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, value: EventType | str | None) -> EventType | str | None:
        if value is None:
            return None
        if isinstance(value, EventType):
            return value
        return Event(type=value, session_id="query").type

    @field_validator("event_types", "exclude_event_types", mode="before")
    @classmethod
    def copy_event_types(cls, value, info) -> tuple[EventType | str, ...]:
        if value is None:
            return ()
        if type(value) is str:
            raise ValueError(f"`{info.field_name}` must be a sequence of event types.")
        normalized: list[EventType | str] = []
        for item in tuple(value):
            if not isinstance(item, EventType):
                item = Event(type=item, session_id="query").type
            if item in normalized:
                raise ValueError(f"`{info.field_name}` must not contain duplicates.")
            normalized.append(item)
        return tuple(normalized)

    @field_validator("since", "until")
    @classmethod
    def validate_query_timestamp(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_time_range(self) -> EventQuery:
        if self.session_id is not None and self.session_ids:
            raise ValueError("Use either `session_id` or `session_ids`, not both.")
        if self.event_id is not None and self.session_id is None:
            raise ValueError("EventQuery event_id requires session_id.")
        if self.event_type is not None and self.event_types:
            raise ValueError("Use either `event_type` or `event_types`, not both.")
        if self.workflow_attempt_id is not None and self.workflow_step_id is None:
            raise ValueError("EventQuery workflow_attempt_id requires workflow_step_id.")
        if self.workflow_attempt_fenced and self.workflow_step_id is None:
            raise ValueError("EventQuery workflow_attempt_fenced requires workflow_step_id.")
        if self.workflow_attempt_fenced and self.workflow_attempt_id is not None:
            raise ValueError("Use workflow_attempt_id or workflow_attempt_fenced, not both.")
        if (
            self.workflow_step_id is not None
            and self.workflow_attempt_id is None
            and not self.workflow_attempt_fenced
        ):
            raise ValueError(
                "EventQuery workflow_step_id requires an exact or fenced attempt scope."
            )
        if self.workflow_step_id is not None and (
            self.session_id is None
            or self.workflow_name is None
            or (self.event_type is None and not self.event_types)
        ):
            raise ValueError(
                "Workflow-step event queries require session_id, workflow_name, "
                "and an event-type filter."
            )
        if self.since is not None and self.until is not None and self.since >= self.until:
            raise ValueError("EventQuery since must be before until.")
        if (
            self.before_sequence is not None
            and self.session_id is None
            and len(self.session_ids) != 1
        ):
            raise ValueError("EventQuery before_sequence requires exactly one session.")
        if (
            self.after_sequence is not None
            and self.before_sequence is not None
            and self.after_sequence >= self.before_sequence
        ):
            raise ValueError("EventQuery after_sequence must be before before_sequence.")
        return self


class EventQueryResultTooLarge(ValueError):
    """A bounded event query would hydrate more serialized bytes than allowed."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        super().__init__(f"Event query exceeds the {max_bytes}-byte safety limit.")


class TranscriptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    interaction_id: str | None = None
    message: Message

    @field_validator("interaction_id")
    @classmethod
    def validate_interaction_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "interaction_id")

    @field_validator("message")
    @classmethod
    def copy_message(cls, value: Message) -> Message:
        return copy_message(value)


class DeferredInteractionInput(BaseModel):
    """Source messages durably admitted but not yet visible in the transcript."""

    model_config = ConfigDict(extra="forbid")

    interaction_id: str
    source_messages: list[Message]

    @field_validator("interaction_id")
    @classmethod
    def validate_interaction_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "interaction_id")

    @field_validator("source_messages")
    @classmethod
    def copy_source_messages(cls, value: list[Message]) -> list[Message]:
        return [copy_message(message) for message in value]


class TranscriptPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[TranscriptRecord] = Field(default_factory=list)
    total_records: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)


class SessionListResult(BaseModel):
    """One page of a session listing plus its keyset cursor and (optional) total count."""

    model_config = ConfigDict(extra="forbid")

    sessions: list[Session] = Field(default_factory=list)
    next_cursor: str | None = None
    # None unless the query opted in via include_total_count (COUNT is expensive at scale).
    total_count: StrictInt | None = Field(default=None, ge=0, le=MAX_DURABLE_JSON_INTEGER)

    @field_validator("next_cursor")
    @classmethod
    def validate_next_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_bounded_session_list_cursor(value, "next_cursor")


SESSION_TOPOLOGY_MAX_ANCESTOR_DEPTH = 32
SESSION_TOPOLOGY_MAX_EXPANDED_PARENTS = 50
SESSION_TOPOLOGY_DEFAULT_CHILD_LIMIT = 25
SESSION_TOPOLOGY_MAX_CHILD_LIMIT = 100
SESSION_TOPOLOGY_MAX_NODES = 500
SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES = 1024
SESSION_TOPOLOGY_MAX_CURSOR_BYTES = 4096


class SessionTopologyDepthExceeded(ValueError):
    """A focus session has more ancestors than the bounded topology contract."""


class SessionTopologyCycle(ValueError):
    """Durable parent-session records contain a lineage cycle."""


def _bounded_session_topology_text(
    value: str,
    field_name: str,
    *,
    max_bytes: int,
) -> str:
    value = require_clean_nonblank(value, field_name)
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} must contain portable Unicode text.") from exc
    if len(encoded) > max_bytes:
        raise ValueError(f"{field_name} exceeds the session topology byte limit.")
    return value


class SessionTopologyNode(BaseModel):
    """Bounded metadata-free identity for one session topology node."""

    model_config = ConfigDict(extra="forbid")

    id: str
    agent_name: str
    provider_name: str
    model: str
    parent_session_id: str | None
    causal_budget_id: str
    runtime_name: str
    runtime_version: str | None
    environment_name: str | None
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime

    @classmethod
    def from_session(cls, session: Session) -> SessionTopologyNode:
        if type(session) is not Session:
            raise TypeError("Session topology nodes require Session instances.")
        return cls(
            id=session.id,
            agent_name=session.agent_name,
            provider_name=session.provider_name,
            model=session.model,
            parent_session_id=session.parent_session_id,
            causal_budget_id=session.causal_budget_id,
            runtime_name=session.runtime_name,
            runtime_version=session.runtime_version,
            environment_name=session.environment_name,
            status=session.status,
            created_at=session.created_at,
            updated_at=session.updated_at,
            last_activity_at=session.last_activity_at,
        )

    @field_validator(
        "id",
        "agent_name",
        "provider_name",
        "model",
        "causal_budget_id",
        "runtime_name",
    )
    @classmethod
    def validate_required_strings(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("id", "causal_budget_id")
    @classmethod
    def validate_bounded_ids(cls, value: str, info) -> str:
        return _bounded_session_topology_text(
            value,
            info.field_name,
            max_bytes=SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES,
        )

    @field_validator("parent_session_id")
    @classmethod
    def validate_parent_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_session_topology_text(
            value,
            "parent_session_id",
            max_bytes=SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES,
        )

    @field_validator("runtime_version", "environment_name")
    @classmethod
    def validate_optional_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("created_at", "updated_at", "last_activity_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)


class SessionTopologyQuery(BaseModel):
    """One bounded ancestry plus batched child-branch read."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    focus_session_id: str
    expanded_parent_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=SESSION_TOPOLOGY_MAX_EXPANDED_PARENTS,
    )
    child_cursors: dict[str, str] = Field(
        default_factory=dict,
        max_length=SESSION_TOPOLOGY_MAX_EXPANDED_PARENTS,
    )
    ancestor_depth_limit: StrictInt = Field(
        default=SESSION_TOPOLOGY_MAX_ANCESTOR_DEPTH,
        ge=1,
        le=SESSION_TOPOLOGY_MAX_ANCESTOR_DEPTH,
    )
    child_limit: StrictInt = Field(
        default=SESSION_TOPOLOGY_DEFAULT_CHILD_LIMIT,
        ge=1,
        le=SESSION_TOPOLOGY_MAX_CHILD_LIMIT,
    )

    @field_validator("focus_session_id")
    @classmethod
    def validate_focus_session_id(cls, value: str) -> str:
        return _bounded_session_topology_text(
            value,
            "focus_session_id",
            max_bytes=SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES,
        )

    @field_validator("expanded_parent_ids", mode="before")
    @classmethod
    def copy_expanded_parent_ids(cls, value) -> tuple[str, ...]:
        if value is None:
            return ()
        if type(value) is str:
            raise ValueError("expanded_parent_ids must be a sequence of strings.")
        copied: list[str] = []
        for index, item in enumerate(tuple(value)):
            parent_id = _bounded_session_topology_text(
                item,
                f"expanded_parent_ids[{index}]",
                max_bytes=SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES,
            )
            if parent_id in copied:
                raise ValueError("expanded_parent_ids must not contain duplicates.")
            copied.append(parent_id)
        return tuple(copied)

    @field_validator("child_cursors", mode="before")
    @classmethod
    def copy_child_cursors(cls, value) -> dict[str, str]:
        if value is None:
            return {}
        if type(value) is not dict:
            raise ValueError("child_cursors must be an object.")
        copied: dict[str, str] = {}
        for raw_parent_id, raw_cursor in value.items():
            parent_id = _bounded_session_topology_text(
                raw_parent_id,
                "child_cursors key",
                max_bytes=SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES,
            )
            copied[parent_id] = _bounded_session_topology_text(
                raw_cursor,
                f"child_cursors[{parent_id!r}]",
                max_bytes=SESSION_TOPOLOGY_MAX_CURSOR_BYTES,
            )
        return copied

    @model_validator(mode="after")
    def validate_cursor_parents(self) -> SessionTopologyQuery:
        unknown = set(self.child_cursors).difference(self.expanded_parent_ids)
        if unknown:
            raise ValueError("child_cursors keys must also appear in expanded_parent_ids.")
        return self


class SessionTopologyBranch(BaseModel):
    """One independently pageable direct-child branch."""

    model_config = ConfigDict(extra="forbid")

    parent_session_id: str
    children: tuple[SessionTopologyNode, ...] = Field(
        default=(),
        max_length=SESSION_TOPOLOGY_MAX_CHILD_LIMIT,
    )
    next_cursor: str | None = None
    has_more: StrictBool = False

    @field_validator("parent_session_id")
    @classmethod
    def validate_parent_session_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "parent_session_id")

    @field_validator("next_cursor")
    @classmethod
    def validate_next_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "next_cursor")

    @model_validator(mode="after")
    def validate_continuation(self) -> SessionTopologyBranch:
        if self.has_more and self.next_cursor is None:
            raise ValueError("A topology branch with more children requires a cursor.")
        if not self.has_more and self.next_cursor is not None:
            raise ValueError("A complete topology branch cannot expose a cursor.")
        return self


class SessionTopologyStoreResult(BaseModel):
    """Backend-neutral bounded topology projection."""

    model_config = ConfigDict(extra="forbid")

    focus: SessionTopologyNode
    ancestors: tuple[SessionTopologyNode, ...] = Field(
        default=(),
        max_length=SESSION_TOPOLOGY_MAX_ANCESTOR_DEPTH,
    )
    expanded_parents: tuple[SessionTopologyNode, ...] = Field(
        default=(),
        max_length=SESSION_TOPOLOGY_MAX_EXPANDED_PARENTS,
    )
    branches: tuple[SessionTopologyBranch, ...] = Field(
        default=(),
        max_length=SESSION_TOPOLOGY_MAX_EXPANDED_PARENTS,
    )

    @model_validator(mode="after")
    def validate_shape(self) -> SessionTopologyStoreResult:
        ancestor_ids = [node.id for node in self.ancestors]
        if len(set(ancestor_ids)) != len(ancestor_ids) or self.focus.id in ancestor_ids:
            raise ValueError("Topology ancestors must form an acyclic path to the focus.")
        if self.ancestors:
            if self.ancestors[0].parent_session_id is not None:
                raise ValueError("Topology ancestry must start at the durable root session.")
            for parent, child in zip(self.ancestors, self.ancestors[1:], strict=False):
                if child.parent_session_id != parent.id:
                    raise ValueError("Topology ancestors contain a contradictory parent edge.")
            if self.focus.parent_session_id != self.ancestors[-1].id:
                raise ValueError("Topology ancestry does not terminate at the focus parent.")
        elif self.focus.parent_session_id is not None:
            raise ValueError("Topology omitted the focus session's durable ancestors.")
        expanded_ids = [node.id for node in self.expanded_parents]
        if len(set(expanded_ids)) != len(expanded_ids):
            raise ValueError("Topology expanded parents must not contain duplicates.")
        if len(self.branches) != len(self.expanded_parents):
            raise ValueError("Every expanded parent requires exactly one branch.")
        if [branch.parent_session_id for branch in self.branches] != [
            parent.id for parent in self.expanded_parents
        ]:
            raise ValueError("Topology branches must preserve expanded-parent order.")
        for branch in self.branches:
            child_ids = [child.id for child in branch.children]
            if len(set(child_ids)) != len(child_ids):
                raise ValueError("A topology branch must not repeat a child session.")
            if any(
                child.parent_session_id != branch.parent_session_id for child in branch.children
            ):
                raise ValueError("A topology branch contains a contradictory parent edge.")
            if list(branch.children) != sorted(
                branch.children,
                key=lambda child: (child.created_at, child.id),
            ):
                raise ValueError("Topology branch children must use stable creation ordering.")
            if branch.next_cursor is not None:
                cursor_created_at, cursor_id = decode_session_topology_cursor(
                    branch.next_cursor,
                    parent_session_id=branch.parent_session_id,
                )
                if not branch.children or (
                    cursor_created_at,
                    cursor_id,
                ) != (
                    branch.children[-1].created_at,
                    branch.children[-1].id,
                ):
                    raise ValueError(
                        "A topology continuation cursor must identify the last returned child."
                    )
        all_nodes = (
            self.focus,
            *self.ancestors,
            *self.expanded_parents,
            *(child for branch in self.branches for child in branch.children),
        )
        nodes_by_id: dict[str, SessionTopologyNode] = {}
        for node in all_nodes:
            prior = nodes_by_id.setdefault(node.id, node)
            if prior != node:
                raise ValueError("Topology contains contradictory representations of one session.")
        _reject_loaded_session_topology_cycles(nodes_by_id)
        if len(nodes_by_id) > SESSION_TOPOLOGY_MAX_NODES:
            raise ValueError(
                f"Session topology cannot retain more than {SESSION_TOPOLOGY_MAX_NODES} nodes."
            )
        return self


class TranscriptQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    session_id: str
    interaction_id: str | None = None
    role: MessageRole | str | None = None
    offset: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    limit: StrictInt = Field(default=100, ge=1, le=5000)
    # When False, ThinkingPart content is stripped from the returned messages. This is a
    # content view, not a record filter: `total_records` stays the role-matched total, a
    # page may hold fewer than `limit` records when thinking-only turns drop out, and each
    # record keeps its true transcript `index` (so offset pagination is unaffected).
    include_thinking: StrictBool = True

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "session_id")

    @field_validator("interaction_id")
    @classmethod
    def validate_interaction_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "interaction_id")

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: MessageRole | str | None) -> MessageRole | None:
        if value is None:
            return None
        return MessageRole(value)


def _without_thinking_parts(message: Message) -> Message | None:
    kept = [part for part in message.content if type(part) is not ThinkingPart]
    if not kept:
        return None
    return Message(role=message.role, content=tuple(kept))


def filter_transcript_records(
    records: list[TranscriptRecord], *, include_thinking: bool
) -> list[TranscriptRecord]:
    """Apply a `TranscriptQuery.include_thinking` filter to a page of records.

    When ``include_thinking`` is False, ThinkingParts are stripped from each message and
    records whose message is left empty (a thinking-only turn) are dropped. Every
    surviving message on that path is freshly rebuilt through full validation, so it
    shares no payload state with the input records; when True, records pass through
    unchanged and the caller is responsible for any isolation.
    """
    if include_thinking:
        return records
    filtered: list[TranscriptRecord] = []
    for record in records:
        message = _without_thinking_parts(record.message)
        if message is not None:
            filtered.append(
                TranscriptRecord(
                    index=record.index,
                    interaction_id=record.interaction_id,
                    message=message,
                )
            )
    return filtered


def _public_authority_alias_store_key(
    public_alias: str,
    *,
    field_name: str,
    private_value: str | None,
    scope_session_id: str | None,
) -> tuple[str, str, str]:
    parsed = parse_public_authority_alias(public_alias)
    if parsed is None or parsed.field_name != field_name:
        raise ValueError("Public authority alias is malformed or field-mismatched.")
    if field_name == "session_id":
        if scope_session_id is not None:
            raise ValueError("Session aliases must not have a session scope.")
        scope_key = ""
    elif field_name == "interaction_id":
        if scope_session_id is None:
            raise ValueError("Interaction aliases require a private session scope.")
        scope_key = require_nonblank(scope_session_id, "scope_session_id")
    else:
        raise ValueError("field_name must be session_id or interaction_id.")
    if private_value is not None:
        require_nonblank(private_value, "private_value")
    return field_name, scope_key, public_alias


def _authenticated_public_authority_alias_private_value(
    codec: PublicAuthorityAliasCodec | None,
    public_alias: str,
    private_value: str | None,
    *,
    field_name: str,
    scope_session_id: str | None,
) -> str | None:
    """Return a registry value only while its signing key remains accepted."""

    if private_value is None or codec is None:
        return None
    try:
        authenticated = codec.matches(
            public_alias,
            private_value,
            field_name=field_name,
            session_id=scope_session_id,
        )
    except (TypeError, ValueError):
        raise ValueError(
            "Public authority alias registry contains invalid private authority."
        ) from None
    return private_value if authenticated else None


class SessionStore(ABC):
    """Persistent store for sessions and append-only events.

    Implementations own the run-epoch contract. A successful transition to
    ``RUNNING`` claims a new epoch for the current execution context. Runtime
    progress writes made by that context must reject a stale epoch with
    ``SessionRunFenced``. Releasing the claim revokes the durable epoch before
    clearing task-local ownership so inherited child contexts cannot write late.
    """

    # Custom stores must opt into optional capabilities explicitly. Conservative
    # defaults keep discovery truthful for inherited methods that fail closed.
    supports_usage_aggregates: ClassVar[bool] = False
    supports_mcp_manifest_history: ClassVar[bool] = False
    supports_session_topology: ClassVar[bool] = False
    supports_public_authority_aliases: ClassVar[bool] = False

    @property
    def public_authority_alias_codec(self) -> PublicAuthorityAliasCodec | None:
        """Return the immutable codec bound to this store, when supported."""

        return None

    async def register_public_authority_alias(
        self,
        public_alias: str,
        *,
        field_name: str,
        private_value: str,
        scope_session_id: str | None = None,
    ) -> None:
        """Persist one public-to-private authority mapping before exposure.

        This optional capability is deliberately explicit. A runtime must not
        publish a one-way public alias unless its store can resolve that alias
        after restart and from another worker.
        """

        raise NotImplementedError(
            "This SessionStore does not support durable public authority aliases."
        )

    async def resolve_public_authority_alias(
        self,
        public_alias: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> str | None:
        """Resolve one exact indexed alias within its authority scope."""

        raise NotImplementedError(
            "This SessionStore does not support durable public authority aliases."
        )

    async def public_authority_private_value_exists(
        self,
        private_value: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> bool:
        """Return positive indexed evidence that private authority exists."""

        raise NotImplementedError(
            "This SessionStore does not support durable public authority aliases."
        )

    @abstractmethod
    async def create(
        self,
        request: RunRequest,
        *,
        identity: SessionIdentity,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
    ) -> Session:
        """Create a session, optionally admitting its first interaction atomically.

        With admission, create the session directly in ``RUNNING``, claim its
        first run epoch, and persist the start event plus deferred source batch
        in the same transaction. Both optional arguments are supplied together.
        """

    @abstractmethod
    async def create_fork(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        expected_source_run_epoch: int,
    ) -> Session:
        """Create a forked session with copied transcript/checkpoint state."""

    async def create_fork_with_transcript_validation(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        expected_source_run_epoch: int,
        transcript_validator: ForkTranscriptValidator,
    ) -> Session:
        """Create a fork after validating the exact transcript inside the copy boundary.

        Custom stores must opt into this safety capability explicitly. The
        conservative default prevents the runtime from silently falling back to
        a time-of-check/time-of-use transcript validation.
        """

        raise NotImplementedError(
            "This SessionStore does not support atomic fork transcript validation."
        )

    @abstractmethod
    async def load(self, session_id: str) -> Session | None:
        """Load a session by id."""

    @abstractmethod
    async def load_state(self, session_id: str) -> SessionStateSnapshot | None:
        """Load bounded mutable state without labels or unbounded metadata."""

    @abstractmethod
    async def inspect_identity(self, session_id: str) -> SessionInspectionIdentity:
        """Load bounded session identity without materializing session metadata."""

    @abstractmethod
    async def update_status(self, session_id: str, status: SessionStatus) -> Session:
        """Update session status and return the updated session."""

    @abstractmethod
    async def update_model(self, session_id: str, model: str) -> Session:
        """Update the active model for a session and return the updated session."""

    @abstractmethod
    async def transition_status(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
    ) -> Session:
        """Atomically transition status, claiming a new epoch when entering RUNNING."""

    @abstractmethod
    async def transition_status_and_checkpoint(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_transform: CheckpointTransform,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
        continued_interaction_id: str | None = None,
        defer_interaction_source: bool = False,
    ) -> Session:
        """Atomically persist a transition/checkpoint and optional interaction admission."""

    @abstractmethod
    async def transition_status_if_no_queued_messages(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
    ) -> Session:
        """Atomically terminalize only when no durable queued input remains."""

    @abstractmethod
    async def publish_interaction_transition(
        self,
        session_id: str,
        *,
        event: Event,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        only_if_no_queued_messages: bool = False,
    ) -> InteractionTransitionResult:
        """Atomically publish one interaction state and its session transition.

        A completion guarded by ``only_if_no_queued_messages`` always publishes
        the interaction event, but leaves the session status unchanged when
        queued input remains. Repeating the exact event identity reconstructs
        the committed result without re-evaluating that queue boundary.
        """

    @abstractmethod
    async def fence_stalled_run(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        inactive_before: datetime,
    ) -> Session | None:
        """Atomically evict a stale run and claim its newly incremented epoch.

        Returns ``None`` when the session is no longer in one of ``statuses`` or
        has activity newer than ``inactive_before``.
        """
        raise NotImplementedError

    @abstractmethod
    async def fence_run_and_transform_checkpoint(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        checkpoint_transform: CheckpointTransform,
    ) -> Session:
        """Atomically transform a checkpoint and claim a newly incremented epoch.

        The transform must return the replacement checkpoint. An exception or
        ``None`` result aborts both the checkpoint update and the epoch fence.
        This operation is for ownership that must persist a checkpoint lease
        and fence the prior run as one transaction. Recovery that needs only an
        inactivity predicate should continue to use :meth:`fence_stalled_run`.
        """
        raise NotImplementedError

    @abstractmethod
    async def release_run_fence(self, session_id: str) -> None:
        """Revoke this task's epoch after all trailing writes finish."""
        _deactivate_session_run_fence(require_clean_nonblank(session_id, "session_id"))

    @abstractmethod
    async def claim_budget_reservation_identity(
        self,
        *,
        reservation_id: str,
        publication_session_id: str,
        publication_id: str,
    ) -> None:
        """Durably bind one ledger-wide reservation id to an exact session event.

        Claims are idempotent only for the same session/event pair and must not
        be removed when a session is deleted. ``append_events`` atomically marks
        the matching claim published when it stores ``budget.reserved``. The
        claim is a session-scoped mutation: the publication session must exist
        and an active run fence must still own its current epoch.
        """

    async def append_event(self, session_id: str, event: Event) -> None:
        """Append one event to a session."""
        await self.append_events(session_id, [event])

    @abstractmethod
    async def append_events(self, session_id: str, events: list[Event]) -> None:
        """Append events to a session in one durable batch."""

    async def append_workflow_step_started(
        self,
        session_id: str,
        event: Event,
        *,
        workflow_name: str,
        attempt_id: str,
    ) -> bool:
        """Atomically append a step reservation only for the current attempt.

        The attempt-fence check and event insert share one store transaction.
        ``False`` means either a newer attempt won or this exact reservation was
        already published.
        """

        raise NotImplementedError(
            f"{type(self).__name__} does not support atomic workflow step reservation."
        )

    async def load_mcp_manifest_baselines(
        self,
        history_keys: tuple[str, ...],
    ) -> McpManifestBaselineLoadResult:
        """Load authoritative baselines for the requested manifest-history keys."""

        raise NotImplementedError(
            f"{type(self).__name__} must implement durable MCP manifest history."
        )

    async def compare_and_publish_mcp_manifest_checks(
        self,
        session_id: str,
        *,
        expected_generations: dict[str, int | None],
        baseline_updates: dict[str, McpManifestBaseline],
        events: list[Event],
    ) -> McpManifestPublicationResult:
        """Atomically publish decisions and return the exact resulting requested baselines.

        Every accepted ``first_seen`` or ``changed`` event must install the
        matching authoritative baseline in the same publication. Every
        expected history key must have exactly one coherent outcome;
        ``unchanged`` must match its current baseline, and blocked outcomes
        must not update one.
        """

        raise NotImplementedError(
            f"{type(self).__name__} must implement atomic MCP manifest publication."
        )

    @abstractmethod
    async def claim_persisted_event_side_effect(
        self,
        *,
        session_id: str | None = None,
        event_id: str | None = None,
        lease_seconds: float = 300.0,
    ) -> PersistedEventSideEffectClaim | None:
        """Claim the oldest eligible persisted event side-effect handoff."""

    @abstractmethod
    async def get_persisted_event_side_effect_delivery(
        self,
        *,
        session_id: str,
        event_id: str,
    ) -> PersistedEventSideEffectDelivery | None:
        """Load one persisted event side-effect handoff by event identity."""

    @abstractmethod
    async def mark_persisted_event_side_effect_delivered(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> PersistedEventSideEffectDelivery:
        """Mark one claimed event's configured side effects delivered."""

    @abstractmethod
    async def mark_persisted_event_side_effect_failed(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        error: str,
        max_attempts: int,
        retry_delay_seconds: float,
    ) -> PersistedEventSideEffectDelivery:
        """Release one failed claim for retry or dead-letter it."""

    @abstractmethod
    async def list_persisted_event_side_effect_deliveries(
        self,
        *,
        statuses: set[PersistedEventSideEffectStatus] | None = None,
        claimable_only: bool = False,
        after_sequence: int | None = None,
        limit: int = 100,
    ) -> list[PersistedEventSideEffectDelivery]:
        """Inspect persisted side-effect delivery state in event order.

        ``claimable_only`` excludes terminal deliveries and live leases.
        """

    @abstractmethod
    async def enqueue_session_message(
        self,
        request: EnqueueSessionMessageRequest,
    ) -> EnqueueSessionMessageResult:
        """Atomically accept queued input with its causal event, or replay it."""

    @abstractmethod
    async def deliver_queued_session_messages(
        self,
        session_id: str,
        *,
        include_on_idle: bool,
        delivery_id: str | None = None,
        eligible_through: int | None = None,
        limit: int = SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
        interaction_id: str | None = None,
        interaction_started_event: Event | None = None,
    ) -> SessionMessageDeliveryBatch:
        """Atomically append and mark one bounded queue batch delivered.

        Runtime callers supply a stable ``delivery_id`` for acknowledgement-loss
        reconstruction. Omitting it creates a one-shot identity for direct store
        callers and returns that identity in the result.
        """

    @abstractmethod
    async def publish_checkpoint_and_events(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        """Atomically transform a checkpoint and append its causal event batch."""

    @abstractmethod
    async def load_session_operation(
        self,
        session_id: str,
        idempotency_key: str,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> dict[str, Any] | None:
        """Load one terminal durable operation record by caller idempotency key.

        A supplied root guard runs in the same read snapshot before the
        checkpoint-coupled operation record is interpreted.
        """

    @abstractmethod
    async def publish_session_operation(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        """Atomically publish a checkpoint, events, and terminal operation records."""

    async def publish_session_operation_guarded(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform,
        commit_guard: Callable[[], None],
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        """Publish with a guard evaluated at the backend's commit boundary."""

        raise NotImplementedError(
            f"{type(self).__name__} must implement atomic guarded operation publication."
        )

    async def load_runtime_publication_receipt(
        self,
        session_id: str,
        publication_id: str,
    ) -> RuntimePublicationReceipt | None:
        """Load and verify one store-owned runtime publication receipt."""

        session_id = require_clean_nonblank(session_id, "session_id")
        publication_id = require_clean_nonblank(publication_id, "publication_id")
        storage_key = _runtime_publication_storage_key(publication_id)
        record = await self._load_runtime_publication_receipt_record(
            session_id,
            storage_key,
            publication_id,
        )
        if record is None:
            return None
        return _reconstruct_runtime_publication_receipt(
            record,
            storage_key=storage_key,
            session_id=session_id,
            publication_id=publication_id,
        )

    async def publish_runtime_publication(
        self,
        session_id: str,
        *,
        request: RuntimePublicationRequest,
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> RuntimePublicationResult:
        """Atomically publish or exactly replay one logical runtime boundary."""

        prepared = _prepare_runtime_publication(
            session_id,
            request,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )
        return await self._publish_runtime_publication_atomic(prepared)

    async def _load_runtime_publication_receipt_record(
        self,
        session_id: str,
        storage_key: str,
        publication_id: str,
    ) -> dict[str, Any] | None:
        """Backend receipt/material lookup hook; fail closed for custom stores.

        Implementations must verify the committed transcript slice, appended
        event batch, and exact content of every referenced event in the same
        locked or transactional snapshot used to load the receipt.
        """

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable runtime publications."
        )

    async def _publish_runtime_publication_atomic(
        self,
        prepared: _PreparedRuntimePublication,
    ) -> RuntimePublicationResult:
        """Backend atomic publication hook; intentionally non-abstract for compatibility."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable runtime publications."
        )

    async def load_model_completion_stage(
        self,
        session_id: str,
        stage_id: str,
    ) -> ModelCompletionStage | None:
        """Load and verify append-only pre-dispatch and terminal completion evidence."""

        session_id, stage_id, preparation_key, terminal_key = (
            _model_completion_stage_storage_identity(session_id, stage_id)
        )
        preparation_record, terminal_record = await self._load_model_completion_stage_records(
            session_id,
            preparation_key,
            terminal_key,
        )
        return _reconstruct_model_completion_stage(
            preparation_record,
            terminal_record,
            session_id=session_id,
            stage_id=stage_id,
            preparation_storage_key=preparation_key,
            terminal_storage_key=terminal_key,
        )

    async def load_active_model_completion_stage(
        self,
        session_id: str,
    ) -> ActiveModelCompletionStage | None:
        """Load a verified snapshot of the dispatch currently eligible to publish."""

        session_id = require_clean_nonblank(session_id, "session_id")
        (
            active_record,
            preparation_record,
            terminal_record,
        ) = await self._load_active_model_completion_stage_records(session_id)
        return _reconstruct_active_model_completion_stage(
            active_record,
            preparation_record,
            terminal_record,
            session_id=session_id,
        )

    async def prepare_model_completion_stage(
        self,
        session_id: str,
        *,
        request: ModelCompletionStageRequest,
        expected_statuses: set[SessionStatus],
        expected_run_epoch: int,
        expected_transcript_cursor: int,
    ) -> ModelCompletionStageResult:
        """Insert or replay a stage; dispatch only when the result explicitly authorizes it."""

        prepared = _prepare_model_completion_stage(
            session_id,
            request,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )
        return await self._prepare_model_completion_stage_atomic(prepared)

    async def complete_model_completion_stage(
        self,
        session_id: str,
        *,
        stage_id: str,
        publication: RuntimePublicationRequest,
    ) -> ModelCompletionStageResult:
        """Append or exactly replay immutable terminal provider-response material."""

        prepared = _prepare_model_completion_stage_terminal(
            session_id,
            stage_id=stage_id,
            publication=publication,
        )
        return await self._complete_model_completion_stage_atomic(prepared)

    async def abandon_model_completion_stage(
        self,
        session_id: str,
        *,
        stage_id: str,
        preparation_digest: str,
        expected_run_epoch: int,
    ) -> ModelCompletionStageAbandonmentResult:
        """Atomically abandon one exact active stage before provider dispatch.

        An exact replay reconstructs the durable abandonment tombstone. A later
        preparation of the same request receives a new preparation digest, so a
        delayed retry of an older abandonment can never clear it.
        """

        prepared = _prepare_model_completion_stage_abandonment(
            session_id,
            stage_id=stage_id,
            preparation_digest=preparation_digest,
            expected_run_epoch=expected_run_epoch,
        )
        return await self._abandon_model_completion_stage_atomic(prepared)

    async def promote_model_completion_stage(
        self,
        session_id: str,
        *,
        stage_id: str,
        expected_run_epoch: int,
    ) -> RuntimePublicationResult:
        """Atomically verify and publish a terminal stage without a load/publish race."""

        expected_run_epoch = _validate_required_runtime_fence(
            expected_run_epoch,
            "expected_run_epoch",
        )
        session_id, stage_id, preparation_key, terminal_key = (
            _model_completion_stage_storage_identity(session_id, stage_id)
        )
        return await self._promote_model_completion_stage_atomic(
            session_id=session_id,
            stage_id=stage_id,
            preparation_storage_key=preparation_key,
            terminal_storage_key=terminal_key,
            expected_run_epoch=expected_run_epoch,
        )

    async def _load_model_completion_stage_records(
        self,
        session_id: str,
        preparation_storage_key: str,
        terminal_storage_key: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Backend stage lookup hook; intentionally fail closed for custom stores."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion stages."
        )

    async def _load_active_model_completion_stage_records(
        self,
        session_id: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        """Backend snapshot hook for the active marker and its immutable stage."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion stages."
        )

    async def _prepare_model_completion_stage_atomic(
        self,
        prepared: _PreparedModelCompletionStage,
    ) -> ModelCompletionStageResult:
        """Backend hook for an insert-only, fenced preparation record."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion stages."
        )

    async def _complete_model_completion_stage_atomic(
        self,
        prepared: _PreparedModelCompletionStageTerminal,
    ) -> ModelCompletionStageResult:
        """Backend hook for an insert-only terminal completion record."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion stages."
        )

    async def _abandon_model_completion_stage_atomic(
        self,
        prepared: _PreparedModelCompletionStageAbandonment,
    ) -> ModelCompletionStageAbandonmentResult:
        """Backend hook for an exact, fenced pre-dispatch abandonment."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion stages."
        )

    async def _promote_model_completion_stage_atomic(
        self,
        *,
        session_id: str,
        stage_id: str,
        preparation_storage_key: str,
        terminal_storage_key: str,
        expected_run_epoch: int,
    ) -> RuntimePublicationResult:
        """Backend hook that verifies the stage and publishes under one store lock."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion stages."
        )

    @abstractmethod
    async def load_events(self, session_id: str) -> list[Event]:
        """Load all events for a session."""

    async def load_tool_round_lifecycle_events(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
    ) -> list[Event]:
        """Load exact started/terminal candidates for bounded tool-call identities.

        Built-in stores override this with their fixed-size pending-action
        indexes. The paginated fallback preserves correctness for custom stores
        without silently truncating lifecycle evidence.
        """

        session_id = require_clean_nonblank(session_id, "session_id")
        copied_ids = _validate_tool_round_call_ids(
            tool_call_ids,
            "tool_call_ids",
        )
        selected_ids = set(copied_ids)
        lifecycle_events: list[Event] = []
        after_sequence = 0
        while True:
            records = await self.query_events(
                EventQuery(
                    session_id=session_id,
                    event_types=tuple(_TOOL_ROUND_LIFECYCLE_EVENT_TYPES),
                    after_sequence=after_sequence,
                    limit=5000,
                    order_by=EventOrder.SEQUENCE_ASC,
                )
            )
            if not records:
                break
            for record in records:
                if record.event.payload.get("tool_call_id") not in selected_ids:
                    continue
                lifecycle_events.append(copy_event(record.event))
                if len(lifecycle_events) > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS:
                    raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
            after_sequence = records[-1].sequence
            if len(records) < 5000:
                break
        return lifecycle_events

    async def load_tool_round_lifecycle_events_for_round(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
        *,
        tool_round_identity: ToolRoundIdentity,
    ) -> list[Event]:
        """Load bounded lifecycle evidence for one exact logical tool round.

        Built-in stores provide indexed implementations. The default
        implementation pages over durable lifecycle events so reused provider
        call ids cannot hide current-round evidence behind the publication
        binding limit.
        """

        session_id = require_clean_nonblank(session_id, "session_id")
        copied_ids = _validate_tool_round_call_ids(
            tool_call_ids,
            "tool_call_ids",
        )
        tool_round_identity = copy_tool_round_identity(tool_round_identity)
        selected_ids = set(copied_ids)
        lifecycle_events: list[Event] = []
        after_sequence = 0
        while True:
            records = await self.query_events(
                EventQuery(
                    session_id=session_id,
                    event_types=tuple(_TOOL_ROUND_LIFECYCLE_EVENT_TYPES),
                    after_sequence=after_sequence,
                    limit=5000,
                    order_by=EventOrder.SEQUENCE_ASC,
                )
            )
            if not records:
                break
            for record in records:
                event = record.event
                if event.payload.get(
                    "tool_call_id"
                ) not in selected_ids or not _tool_round_event_matches_scope_or_is_ambiguous(
                    event,
                    tool_round_identity,
                ):
                    continue
                lifecycle_events.append(copy_event(event))
                if len(lifecycle_events) > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS:
                    raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
            after_sequence = records[-1].sequence
            if len(records) < 5000:
                break
        return lifecycle_events

    @abstractmethod
    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        """Query stored events with durable sequence cursors."""

    async def query_events_bounded(
        self,
        query: EventQuery,
        *,
        max_bytes: int,
    ) -> list[EventRecord]:
        """Query events only when the backend can enforce bytes before hydration."""

        raise NotImplementedError(
            f"{type(self).__name__} does not implement byte-bounded event queries."
        )

    async def query_latest_interaction_events(
        self,
        session_id: str,
        *,
        before_sequence: int | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        """Return each interaction's latest lifecycle event in descending order.

        Built-in stores implement this as a native keyset query. The explicit
        capability error prevents endpoints from silently degrading to an
        unbounded event-log scan for out-of-tree stores.
        """

        raise NotImplementedError(
            "This SessionStore does not support native interaction pagination."
        )

    @abstractmethod
    async def summarize_events(self, session_id: str) -> EventSummary:
        """Summarize stored events for one session without loading every event."""

    @abstractmethod
    async def summarize_outcome(self, session_id: str) -> SessionOutcome:
        """Derive the current session outcome from durable events."""

    async def inspect_summary(
        self,
        session_id: str,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> SessionInspectionSummary:
        """Return one exact, content-free diagnostic summary.

        The default implementation deliberately composes the public paginated
        query contracts so out-of-tree stores gain the same semantics without a
        backend-specific schema dependency. Implementations may replace it with
        an equivalent bounded native aggregate.
        """

        session_id = require_clean_nonblank(session_id, "session_id")
        identity = await self.inspect_identity(session_id)

        event_count = 0
        event_total_bytes = 0
        event_largest_bytes = 0
        usage_accumulator = _SessionInspectionUsageAccumulator()
        budget_events: list[Event] = []
        budget_model_attempt_terminal_events: list[Event] = []
        retained_event_bytes = 0
        queued_message_count = 0
        delivered_message_count = 0
        operation_event_count = 0
        after_sequence = 0
        while True:
            records = await self.query_events(
                EventQuery(
                    session_id=session_id,
                    after_sequence=after_sequence,
                    limit=_SESSION_INSPECTION_PAGE_SIZE,
                    order_by=EventOrder.SEQUENCE_ASC,
                )
            )
            if not records:
                break
            event_count += len(records)
            if event_count > _SESSION_INSPECTION_MAX_RECORDS:
                raise ValueError(
                    "Session inspection exceeds the "
                    f"{_SESSION_INSPECTION_MAX_RECORDS}-event safety limit."
                )
            for record in records:
                payload_bytes = compact_json_utf8_size(record.event.payload)
                event_total_bytes += payload_bytes
                event_largest_bytes = max(event_largest_bytes, payload_bytes)
                event = record.event
                if event.type in {EventType.MODEL_COMPLETED, EventType.MODEL_ERROR}:
                    model_attempt_event = project_budget_model_attempt_inspection_event(event)
                    retained_event_bytes = _retain_session_inspection_event(
                        retained_event_bytes,
                        model_attempt_event,
                    )
                    budget_model_attempt_terminal_events.append(model_attempt_event)
                if event.type in {EventType.MODEL_COMPLETED, EventType.TOOL_CALL_STARTED}:
                    usage_event, usage_metrics = project_aggregate_usage_inspection_event(event)
                    retained_event_bytes = _retain_session_inspection_event(
                        retained_event_bytes,
                        usage_event,
                    )
                    usage_accumulator.add(event.type, usage_metrics)
                if is_budget_inspection_event(event):
                    budget_event = project_budget_inspection_event(event)
                    retained_event_bytes = _retain_session_inspection_event(
                        retained_event_bytes,
                        budget_event,
                    )
                    budget_events.append(budget_event)
                queued_message_count += event.type == EventType.SESSION_MESSAGE_QUEUED
                delivered_message_count += event.type == EventType.SESSION_MESSAGE_DELIVERED
                operation_event_count += event.type == EventType.SERVER_MUTATION_ACCEPTED
            after_sequence = records[-1].sequence
            if len(records) < _SESSION_INSPECTION_PAGE_SIZE:
                break

        transcript_count = 0
        transcript_total_bytes = 0
        transcript_largest_bytes = 0
        offset = 0
        while True:
            page = await self.query_transcript(
                TranscriptQuery(
                    session_id=session_id,
                    offset=offset,
                    limit=_SESSION_INSPECTION_PAGE_SIZE,
                )
            )
            if offset == 0 and page.total_records > _SESSION_INSPECTION_MAX_RECORDS:
                raise ValueError(
                    "Session inspection exceeds the "
                    f"{_SESSION_INSPECTION_MAX_RECORDS}-message safety limit."
                )
            for record in page.records:
                message_bytes = compact_json_utf8_size(record.message.model_dump(mode="json"))
                transcript_count += 1
                transcript_total_bytes += message_bytes
                transcript_largest_bytes = max(transcript_largest_bytes, message_bytes)
            offset += _SESSION_INSPECTION_PAGE_SIZE
            if offset >= page.total_records:
                break

        pending = await self.query_pending_actions(
            PendingActionQuery(session_id=session_id, limit=200),
            checkpoint_root_guard=checkpoint_root_guard,
        )
        usage, model_calls_with_usage = usage_accumulator.result(session_id)
        budget = session_budget_inspection(
            budget_events,
            model_attempt_terminal_events=budget_model_attempt_terminal_events,
        )
        return SessionInspectionSummary(
            session=identity,
            transcript=SerializedRecordSummary(
                record_count=transcript_count,
                total_bytes=transcript_total_bytes,
                largest_record_bytes=transcript_largest_bytes,
            ),
            events=SerializedRecordSummary(
                record_count=event_count,
                total_bytes=event_total_bytes,
                largest_record_bytes=event_largest_bytes,
            ),
            usage=usage,
            model_calls=usage.model_steps,
            model_calls_with_usage=model_calls_with_usage,
            tool_calls=usage.tool_calls,
            pending_action_count=len(pending.actions),
            pending_action_kinds=tuple(
                sorted({action.kind for action in pending.actions}, key=str)
            ),
            pending_action_issue_count=len(pending.issues),
            queued_message_count=queued_message_count,
            delivered_message_count=delivered_message_count,
            outstanding_message_count=max(
                queued_message_count - delivered_message_count,
                0,
            ),
            operation_event_count=operation_event_count,
            terminal_failure_state=(
                "failed"
                if identity.status is SessionStatus.FAILED
                else "interrupted"
                if identity.status is SessionStatus.INTERRUPTED
                else "none"
            ),
            budget=budget,
        )

    @abstractmethod
    async def list_sessions(self, query: SessionQuery | None = None) -> SessionListResult:
        """List sessions (filtered/sorted/paginated) with a keyset cursor and total count."""

    async def query_session_topology(
        self,
        query: SessionTopologyQuery,
    ) -> SessionTopologyStoreResult:
        """Load bounded ancestry and batched direct-child branches.

        This optional store-native read prevents the control plane from issuing
        one list query per rendered node. Custom stores opt in explicitly through
        ``supports_session_topology``.
        """

        raise NotImplementedError(
            "This SessionStore does not support bounded session topology queries."
        )

    async def aggregate_operational_snapshot(
        self,
        filters: SessionAggregateFilter | None = None,
    ) -> SessionOperationalSnapshot:
        """Count current session states in one store-local read snapshot.

        Default raises ``NotImplementedError`` so adding this control-plane read
        model does not make existing out-of-tree stores uninstantiable.
        """
        raise NotImplementedError(
            "This SessionStore does not support operational aggregate snapshots."
        )

    async def aggregate_usage(self, query: UsageRollupQuery) -> UsageRollupStoreResult:
        """Aggregate bounded event-time activity and usage without loading histories.

        Default raises ``NotImplementedError`` so adding this control-plane read
        model does not make existing out-of-tree stores uninstantiable.
        """
        raise NotImplementedError("This SessionStore does not support usage aggregates.")

    @abstractmethod
    async def list_sessions_with_pending_interruption_cascade(
        self,
        query: SessionQuery | None = None,
    ) -> SessionListResult:
        """List only sessions carrying a pending interruption cascade marker.

        Implementations should filter at the storage layer. This is the
        restart-discovery path, so scanning all historical sessions and loading
        their checkpoints individually is not an acceptable implementation.
        """

    @abstractmethod
    async def query_pending_actions(
        self,
        query: PendingActionQuery | None = None,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> PendingActionListResult:
        """List current checkpoint-backed actions without scanning session histories.

        Implementations must bound candidate and event reads, apply filtering at
        the storage layer where possible, and avoid per-session history queries.
        A supplied root guard runs on bounded opaque evidence in the same read
        snapshot before any pending state is interpreted.
        """

    async def delete_session(self, session_id: str) -> None:
        """Delete a session and cascade to its events, transcript, and checkpoint.

        Raises ``ValueError`` if the session is in-flight (``RUNNING`` or
        ``INTERRUPTING`` — interrupt it first), has incomplete current-run terminal
        evidence, or if one of its accepted budget reservations has no fully
        delivered terminal settlement audit event.
        Idempotent: deleting a session that does not exist is a no-op.

        Default raises ``NotImplementedError`` so out-of-tree stores keep working.
        """
        raise NotImplementedError("This SessionStore does not support delete_session.")

    async def update_labels(self, session_id: str, labels: dict[str, str]) -> Session:
        """Replace a session's labels (full replacement, not a merge) and return it.

        Annotation writes update ``updated_at`` but do not refresh the runtime
        recovery signal in ``last_activity_at``.

        Raises ``KeyError`` if the session does not exist. Default raises
        ``NotImplementedError`` so out-of-tree stores keep working.
        """
        raise NotImplementedError("This SessionStore does not support update_labels.")

    async def update_metadata(self, session_id: str, metadata: dict[str, Any]) -> Session:
        """Replace a session's user metadata and return the updated session.

        This is a full replacement of user-authored metadata, not a merge.
        Runtime-owned entries are retained atomically and cannot be supplied by
        callers. Annotation writes update ``updated_at`` but do not refresh the
        runtime recovery signal in ``last_activity_at``.

        Raises ``KeyError`` if the session does not exist. Default raises
        ``NotImplementedError`` so out-of-tree stores keep working.
        """
        raise NotImplementedError("This SessionStore does not support update_metadata.")

    @abstractmethod
    async def append_transcript_messages(
        self,
        session_id: str,
        messages: list[Message],
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> None:
        """Append provider-neutral transcript messages to a session.

        Stored state must not alias caller-passed messages: implementations
        must detach nested JSON payloads (serialize, or use
        `cayu.core.messages.detach_message`) so a producer mutating a message
        after append cannot rewrite history.
        """

    @abstractmethod
    async def replace_initial_transcript_messages(
        self,
        session_id: str,
        expected_messages: list[Message],
        replacement_messages: list[Message],
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
        checkpoint_transform: CheckpointTransform | None = None,
    ) -> None:
        """Atomically materialize deferred input as the ordered initial transcript.

        The replacement must end with ``expected_messages``.  Any leading
        bootstrap/system messages are stored with null interaction attribution;
        the source suffix is attributed to ``interaction_id``.  No partial or
        externally visible pre-finalization transcript is permitted.
        When supplied, ``checkpoint_transform`` runs inside the same write
        transaction before the initial-transcript authority marker is read.
        """

    @abstractmethod
    async def load_deferred_interaction_input(
        self,
        session_id: str,
    ) -> DeferredInteractionInput | None:
        """Load the private admitted source batch, if one still exists."""

    @abstractmethod
    async def materialize_deferred_interaction_input(
        self,
        session_id: str,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> bool:
        """Materialize deferred source input without a bootstrap prefix.

        Returns ``True`` when a pending admission was materialized and ``False``
        when no matching deferred input exists.  Recovery and setup-failure paths
        use this before making the owning interaction or session terminal.
        """

    @abstractmethod
    async def append_transcript_messages_and_transform_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        checkpoint_transform: CheckpointTransform,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> None:
        """Append transcript messages and atomically transform the current checkpoint.

        The transform must be synchronous and thread-safe; a store may execute
        it on a worker thread while holding its transactional write boundary.
        """

    @abstractmethod
    async def load_transcript(self, session_id: str) -> list[Message]:
        """Load provider-neutral transcript messages for a session.

        Returned messages must not alias stored state: callers own the result
        and may mutate nested payloads without corrupting the transcript.
        """

    @abstractmethod
    async def query_transcript(self, query: TranscriptQuery) -> TranscriptPage:
        """Query provider-neutral transcript messages with stable message indexes.

        Same isolation contract as `load_transcript`.
        """

    @abstractmethod
    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        """Persist a checkpoint for resume/replay."""

    @abstractmethod
    async def transform_checkpoint(
        self,
        session_id: str,
        checkpoint_transform: CheckpointTransform,
    ) -> None:
        """Atomically transform the latest checkpoint for a session.

        The transform runs while the store owns its session/checkpoint write
        boundary and must be synchronous and thread-safe because a store may
        execute it on a worker thread. Returning ``None`` leaves the checkpoint
        unchanged.
        """

    @abstractmethod
    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        """Load the latest checkpoint for a session."""

    @abstractmethod
    async def load_interruption_cascade_marker(
        self,
        session_id: str,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> dict[str, Any] | None:
        """Load a bounded structural projection of the durable cascade marker.

        A supplied root guard runs on bounded opaque evidence in the same read
        snapshot before the marker is interpreted.
        """


@dataclass(frozen=True)
class _PreparedInMemoryEvent:
    record: EventRecord
    event_type: str
    projected_record: EventRecord | None
    retained_lookup_keys: tuple[str, ...]
    history_lookup_keys: tuple[str, ...]
    delivery: PersistedEventSideEffectDelivery | None


@dataclass(frozen=True)
class _PreparedInMemoryEventAppend:
    events: tuple[_PreparedInMemoryEvent, ...]
    next_sequence: int
    budget_reservation_publications: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _PreparedInMemoryCheckpointStore:
    checkpoint: dict[str, Any]
    has_pending_action: bool
    rebuilt_records: dict[str, dict[str, EventRecord]] | None = None
    rebuilt_event_history: dict[str, list[EventRecord]] | None = None
    rebuilt_index_scope: tuple[frozenset[str], str | None, str | None, str | None] | None = None


@dataclass(frozen=True)
class _InMemoryMessageDeliveryRecord:
    session_id: str
    include_on_idle: bool
    requested_eligible_through: int | None
    limit: int
    interaction_id: str | None
    interaction_started_event: Event | None
    batch: SessionMessageDeliveryBatch


class InMemorySessionStore(SessionStore):
    """In-process session store for tests, local development, and examples."""

    supports_usage_aggregates: ClassVar[bool] = True
    supports_mcp_manifest_history: ClassVar[bool] = True
    supports_session_topology: ClassVar[bool] = True
    supports_public_authority_aliases: ClassVar[bool] = True

    def __init__(
        self,
        *,
        public_authority_alias_codec: PublicAuthorityAliasCodec | None = None,
    ) -> None:
        if public_authority_alias_codec is not None and not isinstance(
            public_authority_alias_codec,
            PublicAuthorityAliasCodec,
        ):
            raise TypeError("public_authority_alias_codec must be a PublicAuthorityAliasCodec.")
        if public_authority_alias_codec is None:
            # In-memory stores have no restart or multi-worker key-distribution
            # boundary. Give each store lifetime a cryptographically random key;
            # persistent stores require explicit deployment provisioning.
            encoded_key = (
                base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")
            )
            public_authority_alias_codec = PublicAuthorityAliasCodec(
                PublicAuthorityAliasKeyring(
                    active_key_id="memory",
                    keys={"memory": SecretStr(encoded_key)},
                )
            )
        self._public_authority_alias_codec = public_authority_alias_codec
        self._lock = asyncio.Lock()
        self._sessions: dict[str, Session] = {}
        self._public_authority_aliases: dict[tuple[str, str, str], str] = {}
        # Stable direct-child keys maintained with session lifecycle writes. Topology
        # pages can therefore seek one parent branch without scanning the complete
        # in-memory session registry.
        self._child_session_keys_by_parent: dict[str, list[tuple[datetime, str]]] = {}
        self._events: dict[str, list[Event]] = {}
        self._event_records: list[EventRecord] = []
        # Secondary indexes over ``_event_records`` (same EventRecord objects, kept in
        # sequence-ascending order) so per-session summaries and type/session-scoped
        # queries — including the per-step budget read path — stop scanning the global
        # event list. Keyed by session id and by ``str(event.type)`` respectively.
        self._session_event_records: dict[str, list[EventRecord]] = {}
        self._interaction_event_records: dict[
            str,
            dict[str, list[EventRecord]],
        ] = {}
        self._latest_interaction_event_records: dict[str, dict[str, EventRecord]] = {}
        self._latest_interaction_event_records_by_sequence: dict[
            str,
            list[EventRecord],
        ] = {}
        self._pending_action_event_records: dict[
            str,
            dict[str, dict[str, EventRecord]],
        ] = {}
        # The identity scope represented by each bounded pending-action index.
        # A checkpoint can reuse a tool-call id for a later execution, so lookup
        # keys alone are insufficient to decide whether rebuilding is required.
        self._pending_action_index_scopes: dict[
            str,
            tuple[frozenset[str], str | None, str | None, str | None],
        ] = {}
        # Unlike the bounded pending-action projection above, runtime publication
        # validation must observe *every* lifecycle event for the active tool
        # round so a caller cannot omit a contradictory second terminal event.
        # This history is identifier-scoped and released with the pending marker.
        self._pending_action_event_history: dict[
            str,
            dict[str, list[EventRecord]],
        ] = {}
        self._pending_action_latest_barrier_records: dict[str, EventRecord] = {}
        self._event_records_by_id: dict[tuple[str, str], EventRecord] = {}
        self._type_event_records: dict[str, list[EventRecord]] = {}
        self._workflow_step_event_records: dict[
            tuple[str, str, str, str, str],
            list[EventRecord],
        ] = {}
        self._workflow_fenced_step_event_records: dict[
            tuple[str, str, str, str],
            list[EventRecord],
        ] = {}
        self._workflow_attempt_event_records: dict[
            tuple[str, str],
            list[EventRecord],
        ] = {}
        self._workflow_active_attempt_ids: dict[tuple[str, str], str] = {}
        self._event_ids: dict[str, set[str]] = {}
        # Store-wide ownership registry. It intentionally outlives session
        # deletion: reservation ids are ledger reconciliation keys, not
        # session-local event attributes.
        self._budget_reservation_identities: dict[str, tuple[str, str, bool]] = {}
        self._persisted_event_side_effect_deliveries: dict[
            tuple[str, str], PersistedEventSideEffectDelivery
        ] = {}
        self._next_event_sequence = 1
        self._transcripts: dict[str, list[Message]] = {}
        self._transcript_interaction_ids: dict[str, list[str | None]] = {}
        self._transcript_indices_by_interaction: dict[str, dict[str | None, list[int]]] = {}
        self._deferred_interaction_inputs: dict[str, tuple[str, list[Message]]] = {}
        self._checkpoints: dict[str, dict[str, Any]] = {}
        self._session_operation_records: dict[str, dict[str, dict[str, Any]]] = {}
        self._mcp_manifest_baselines: dict[str, McpManifestBaseline] = {}
        self._pending_action_session_ids: set[str] = set()
        # Retain every accepted message for idempotent replay, but keep queued
        # delivery state in per-session/mode deques. Hot delivery and terminal
        # fencing therefore depend on the pending set, not historical deliveries.
        self._queued_session_messages_by_idempotency: dict[
            str, dict[str, SessionQueuedMessage]
        ] = {}
        self._pending_session_messages: dict[
            tuple[str, SessionMessageDeliveryMode], deque[SessionQueuedMessage]
        ] = {}
        self._session_message_delivery_records: dict[str, _InMemoryMessageDeliveryRecord] = {}
        self._next_session_message_ordering_key = 1

    @property
    def public_authority_alias_codec(self) -> PublicAuthorityAliasCodec:
        return self._public_authority_alias_codec

    def _index_session_parent_unlocked(self, session: Session) -> None:
        parent_session_id = session.parent_session_id
        if parent_session_id is None:
            return
        keys = self._child_session_keys_by_parent.setdefault(parent_session_id, [])
        key = (session.created_at, session.id)
        index = bisect_left(keys, key)
        if index < len(keys) and keys[index] == key:
            raise RuntimeError("Duplicate in-memory session topology index entry.")
        keys.insert(index, key)

    def _remove_session_parent_index_unlocked(self, session: Session) -> None:
        parent_session_id = session.parent_session_id
        if parent_session_id is None:
            return
        keys = self._child_session_keys_by_parent.get(parent_session_id)
        if keys is None:
            raise RuntimeError("Missing in-memory session topology index branch.")
        key = (session.created_at, session.id)
        index = bisect_left(keys, key)
        if index >= len(keys) or keys[index] != key:
            raise RuntimeError("Missing in-memory session topology index entry.")
        keys.pop(index)
        if not keys:
            del self._child_session_keys_by_parent[parent_session_id]

    def _prepare_checkpoint_store_unlocked(
        self,
        session_id: str,
        checkpoint: dict[str, Any],
        *,
        additional_event_records: tuple[EventRecord, ...] = (),
    ) -> _PreparedInMemoryCheckpointStore:
        from cayu.runtime.pending_actions import (
            checkpoint_has_pending_action_candidate,
            pending_action_checkpoint_lookup_ids,
            pending_action_event_has_unknown_round_call,
            pending_action_event_is_from_different_tool_round,
            pending_action_event_lookup_id,
            pending_action_event_retains_history,
            pending_action_evidence_round_from_checkpoint,
            pending_action_lookup_key,
            project_pending_action_event_record,
            retain_pending_action_index_record,
        )

        if not checkpoint_has_pending_action_candidate(checkpoint):
            return _PreparedInMemoryCheckpointStore(
                checkpoint=checkpoint,
                has_pending_action=False,
            )

        lookup_keys = frozenset(
            pending_action_lookup_key(identifier)
            for identifier in pending_action_checkpoint_lookup_ids(checkpoint)
        )
        try:
            pending_round = pending_action_evidence_round_from_checkpoint(checkpoint)
        except (TypeError, ValueError):
            pending_round = None
        index_scope = (
            lookup_keys,
            None if pending_round is None else pending_round.model_step_id,
            None if pending_round is None else pending_round.model_attempt_id,
            None if pending_round is None else pending_round.tool_round_id,
        )
        rebuilt: dict[str, dict[str, EventRecord]] | None = None
        rebuilt_history: dict[str, list[EventRecord]] | None = None
        if self._pending_action_index_scopes.get(session_id) != index_scope:
            # A public checkpoint transform may legitimately reintroduce a
            # previously cleared durable action or reuse a provider call id for
            # a later execution. SQL stores retain the source events, so rebuild
            # both identity-scoped projections from complete durable evidence.
            rebuilt = {}
            rebuilt_history = {}
            round_lookup_key = (
                None
                if pending_round is None
                else pending_action_lookup_key(pending_round.tool_round_id)
            )
            records = (
                *self._session_event_records.get(session_id, ()),
                *additional_event_records,
            )
            for record in records:
                event_type = str(record.event.type)
                if (
                    event_type not in PENDING_ACTION_EVENT_TYPE_VALUES
                    or event_type in PENDING_ACTION_BARRIER_EVENT_TYPE_VALUES
                ):
                    continue
                retain_for_scope = (
                    pending_round is not None
                    and pending_action_event_has_unknown_round_call(
                        record.event,
                        pending_round,
                    )
                )
                lookup_id = pending_action_event_lookup_id(record.event)
                if lookup_id is None and not retain_for_scope:
                    continue
                projected_record = project_pending_action_event_record(record)
                if lookup_id is not None:
                    lookup_key = pending_action_lookup_key(lookup_id)
                    if lookup_key in lookup_keys:
                        rebuilt_history.setdefault(lookup_key, []).append(record)
                        if not (
                            pending_action_event_retains_history(event_type)
                            and (
                                pending_round is None
                                or pending_action_event_is_from_different_tool_round(
                                    record.event,
                                    pending_round,
                                )
                            )
                        ):
                            retain_pending_action_index_record(
                                rebuilt.setdefault(lookup_key, {}),
                                projected_record,
                            )
                if retain_for_scope:
                    assert round_lookup_key is not None
                    retain_pending_action_index_record(
                        rebuilt.setdefault(round_lookup_key, {}),
                        projected_record,
                    )
                    rebuilt_history.setdefault(round_lookup_key, []).append(record)
        return _PreparedInMemoryCheckpointStore(
            checkpoint=checkpoint,
            has_pending_action=True,
            rebuilt_records=rebuilt,
            rebuilt_event_history=rebuilt_history,
            rebuilt_index_scope=index_scope if rebuilt is not None else None,
        )

    def _apply_checkpoint_store_unlocked(
        self,
        session_id: str,
        prepared: _PreparedInMemoryCheckpointStore,
    ) -> None:
        self._checkpoints[session_id] = prepared.checkpoint
        if prepared.has_pending_action:
            if prepared.rebuilt_records is not None:
                self._pending_action_event_records[session_id] = prepared.rebuilt_records
                assert prepared.rebuilt_event_history is not None
                self._pending_action_event_history[session_id] = prepared.rebuilt_event_history
                assert prepared.rebuilt_index_scope is not None
                self._pending_action_index_scopes[session_id] = prepared.rebuilt_index_scope
            self._pending_action_session_ids.add(session_id)
            return

        self._pending_action_session_ids.discard(session_id)
        # Once the checkpoint no longer names a pending action, identifier-
        # scoped event history cannot contribute to a future action. Keep the
        # latest lifecycle barrier, but release the potentially growing maps.
        removed = self._pending_action_event_records.pop(session_id, None)
        self._pending_action_event_history.pop(session_id, None)
        if removed or session_id in self._pending_action_index_scopes:
            self._pending_action_index_scopes[session_id] = (
                frozenset(),
                None,
                None,
                None,
            )

    def _store_checkpoint_unlocked(self, session_id: str, checkpoint: dict[str, Any]) -> None:
        prepared = self._prepare_checkpoint_store_unlocked(session_id, checkpoint)
        self._apply_checkpoint_store_unlocked(session_id, prepared)

    def _extend_transcript_attribution_unlocked(
        self,
        session_id: str,
        interaction_ids: list[str | None],
    ) -> None:
        """Append attribution and maintain a lazily built absolute-index projection."""

        stored = self._transcript_interaction_ids[session_id]
        start = len(stored)
        stored.extend(interaction_ids)
        projection = self._transcript_indices_by_interaction.get(session_id)
        if projection is None:
            return
        for offset, interaction_id in enumerate(interaction_ids):
            projection.setdefault(interaction_id, []).append(start + offset)

    def _transcript_interaction_projection_unlocked(
        self,
        session_id: str,
    ) -> dict[str | None, list[int]]:
        projection = self._transcript_indices_by_interaction.get(session_id)
        if projection is not None:
            return projection
        projection = {}
        for index, interaction_id in enumerate(
            self._transcript_interaction_ids.get(session_id, ())
        ):
            projection.setdefault(interaction_id, []).append(index)
        self._transcript_indices_by_interaction[session_id] = projection
        return projection

    async def register_public_authority_alias(
        self,
        public_alias: str,
        *,
        field_name: str,
        private_value: str,
        scope_session_id: str | None = None,
    ) -> None:
        key = _public_authority_alias_store_key(
            public_alias,
            field_name=field_name,
            private_value=private_value,
            scope_session_id=scope_session_id,
        )
        codec = self.public_authority_alias_codec
        if codec is None or not codec.matches(
            public_alias,
            private_value,
            field_name=field_name,
            session_id=scope_session_id,
        ):
            raise ValueError("Public authority alias lacks valid store-configured provenance.")
        async with self._lock:
            self._register_public_authority_alias_unlocked(key, private_value)

    def _register_public_authority_alias_unlocked(
        self,
        key: tuple[str, str, str],
        private_value: str,
    ) -> None:
        existing = self._public_authority_aliases.get(key)
        if existing is not None and existing != private_value:
            raise ValueError("Public authority alias conflicts with existing private authority.")
        self._public_authority_aliases[key] = private_value

    def _register_private_authority_alias_unlocked(
        self,
        private_value: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> None:
        codec = self.public_authority_alias_codec
        if codec is None:
            return
        for public_alias in codec.aliases(
            private_value,
            field_name=field_name,
            session_id=scope_session_id,
        ):
            key = _public_authority_alias_store_key(
                public_alias,
                field_name=field_name,
                private_value=private_value,
                scope_session_id=scope_session_id,
            )
            self._register_public_authority_alias_unlocked(key, private_value)

    async def resolve_public_authority_alias(
        self,
        public_alias: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> str | None:
        key = _public_authority_alias_store_key(
            public_alias,
            field_name=field_name,
            private_value=None,
            scope_session_id=scope_session_id,
        )
        async with self._lock:
            return _authenticated_public_authority_alias_private_value(
                self.public_authority_alias_codec,
                public_alias,
                self._public_authority_aliases.get(key),
                field_name=field_name,
                scope_session_id=scope_session_id,
            )

    async def public_authority_private_value_exists(
        self,
        private_value: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> bool:
        codec = self.public_authority_alias_codec
        if codec is None:
            raise RuntimeError("Public authority alias codec is unavailable.")
        probe = codec.encode(
            private_value,
            field_name=field_name,
            session_id=scope_session_id,
        )
        field_name, scope_key, _probe = _public_authority_alias_store_key(
            probe,
            field_name=field_name,
            private_value=private_value,
            scope_session_id=scope_session_id,
        )
        async with self._lock:
            return any(
                key_field == field_name and key_scope == scope_key and stored == private_value
                for (key_field, key_scope, _alias), stored in self._public_authority_aliases.items()
            )

    async def create(
        self,
        request: RunRequest,
        *,
        identity: SessionIdentity,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
    ) -> Session:
        if type(request) is not RunRequest:
            raise TypeError("Session creation requires a RunRequest.")
        request = copy_run_request(request)
        identity = copy_session_identity(identity)
        async with self._lock:
            session_id = request.session_id or str(uuid4())
            admission = _copy_optional_interaction_admission(
                session_id,
                interaction_started_event,
                interaction_source_messages,
                defer_transcript=True,
            )
            if session_id in self._sessions:
                raise ValueError(f"Session already exists: {session_id}")
            if request.parent_session_id == session_id:
                raise ValueError("Session cannot be its own parent.")
            if (
                request.parent_session_id is not None
                and request.parent_session_id not in self._sessions
            ):
                raise ValueError(f"Parent session not found: {request.parent_session_id}")

            now = datetime.now(UTC)
            session = Session(
                id=session_id,
                agent_name=request.agent_name,
                provider_name=identity.provider_name,
                model=identity.model,
                parent_session_id=request.parent_session_id,
                causal_budget_id=request.causal_budget_id or request.task_id or session_id,
                runtime_name=identity.runtime_name,
                runtime_version=identity.runtime_version,
                environment_name=request.environment_name,
                status=(SessionStatus.RUNNING if admission is not None else SessionStatus.PENDING),
                created_at=now,
                updated_at=now,
                last_activity_at=now,
                labels=request.labels,
                metadata=deepcopy(request.metadata),
                run_epoch=1 if admission is not None else 0,
            )
            self._register_private_authority_alias_unlocked(
                session.id,
                field_name="session_id",
            )
            self._sessions[session.id] = session
            self._index_session_parent_unlocked(session)
            self._events[session.id] = []
            self._event_ids[session.id] = set()
            self._session_event_records[session.id] = []
            self._interaction_event_records[session.id] = {}
            self._latest_interaction_event_records[session.id] = {}
            self._latest_interaction_event_records_by_sequence[session.id] = []
            self._pending_action_event_records[session.id] = {}
            self._session_operation_records[session.id] = {}
            self._transcripts[session.id] = []
            self._transcript_interaction_ids[session.id] = []
            if admission is not None:
                started_event, source_messages = admission
                interaction_id = started_event.interaction_id
                if interaction_id is None:
                    raise AssertionError("Interaction admission lost its identity.")
                self._sessions[session.id] = self._append_events_unlocked(
                    session,
                    [started_event],
                )
                self._deferred_interaction_inputs[session.id] = (
                    interaction_id,
                    source_messages,
                )
                self._store_checkpoint_unlocked(
                    session.id,
                    _initial_transcript_pending_checkpoint(interaction_id),
                )
                session = self._sessions[session.id]
                _activate_session_run_fence(session)
            return session.model_copy(deep=True)

    async def create_fork(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        expected_source_run_epoch: int,
    ) -> Session:
        return await self._create_fork(
            source_session_id=source_session_id,
            fork=fork,
            source_statuses=source_statuses,
            transcript_cursor=transcript_cursor,
            checkpoint_transform=checkpoint_transform,
            expected_source_run_epoch=expected_source_run_epoch,
            transcript_validator=None,
        )

    async def create_fork_with_transcript_validation(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        expected_source_run_epoch: int,
        transcript_validator: ForkTranscriptValidator,
    ) -> Session:
        return await self._create_fork(
            source_session_id=source_session_id,
            fork=fork,
            source_statuses=source_statuses,
            transcript_cursor=transcript_cursor,
            checkpoint_transform=checkpoint_transform,
            expected_source_run_epoch=expected_source_run_epoch,
            transcript_validator=transcript_validator,
        )

    async def _create_fork(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        expected_source_run_epoch: int,
        transcript_validator: ForkTranscriptValidator | None,
    ) -> Session:
        source_session_id, fork, allowed_statuses, transcript_cursor = (
            _prepare_session_fork_request(
                source_session_id=source_session_id,
                fork=fork,
                source_statuses=source_statuses,
                transcript_cursor=transcript_cursor,
            )
        )
        async with self._lock:
            source_session = _validate_session_fork_source(
                source_session=self._sessions.get(source_session_id),
                source_session_id=source_session_id,
                fork=fork,
                allowed_statuses=allowed_statuses,
                expected_source_run_epoch=expected_source_run_epoch,
            )
            if MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY in self._session_operation_records.get(
                source_session_id, {}
            ):
                raise ValueError(
                    "Cannot fork a session while a model-completion stage is active: "
                    f"{source_session_id}"
                )
            if fork.id in self._sessions:
                raise ValueError(f"Session already exists: {fork.id}")

            source_transcript = self._transcripts.get(source_session_id, [])
            source_transcript_length = len(source_transcript)
            # Fork and source transcripts may share message objects: internal
            # transcript lists never escape except through the detaching
            # load/append/query boundaries, so the cheap share stays invisible.
            if transcript_cursor is None:
                copied_transcript = [copy_message(message) for message in source_transcript]
            else:
                if transcript_cursor > source_transcript_length:
                    source_transcript = []
                    raise ValueError("transcript_cursor is greater than source transcript length.")
                copied_transcript = [
                    copy_message(message) for message in source_transcript[:transcript_cursor]
                ]
            # A cursor may intentionally exclude legacy secret-bearing suffix
            # entries. The source store owns those records; this frame does not.
            source_transcript = []
            if not fork_transcript_is_accepted(copied_transcript, transcript_validator):
                copied_transcript.clear()
                copied_transcript = []
                raise ValueError(FORK_TRANSCRIPT_VALIDATION_ERROR) from None
            copied_checkpoint = None
            if checkpoint_transform is not None:
                stored_checkpoint = self._checkpoints.get(source_session_id)
                checkpoint_input = (
                    None if stored_checkpoint is None else deepcopy(stored_checkpoint)
                )
                stored_checkpoint = None
                copied_checkpoint = transform_fork_checkpoint(
                    source_session.model_copy(deep=True),
                    checkpoint_input,
                    checkpoint_transform,
                )
                checkpoint_input = None
                if copied_checkpoint is not None:
                    copied_checkpoint = copy_durable_json_object(
                        copied_checkpoint,
                        "checkpoint",
                    )

            self._register_private_authority_alias_unlocked(
                fork.id,
                field_name="session_id",
            )
            self._sessions[fork.id] = fork.model_copy(deep=True)
            self._index_session_parent_unlocked(fork)
            self._events[fork.id] = []
            self._event_ids[fork.id] = set()
            self._session_event_records[fork.id] = []
            self._interaction_event_records[fork.id] = {}
            self._latest_interaction_event_records[fork.id] = {}
            self._latest_interaction_event_records_by_sequence[fork.id] = []
            self._pending_action_event_records[fork.id] = {}
            self._session_operation_records[fork.id] = {}
            self._transcripts[fork.id] = copied_transcript
            source_interaction_ids = self._transcript_interaction_ids.get(source_session_id, [])
            copied_count = len(copied_transcript)
            # Historical attribution remains tied to the source session's
            # interactions; new child work receives new child interaction IDs.
            self._transcript_interaction_ids[fork.id] = list(source_interaction_ids[:copied_count])
            for interaction_id in set(self._transcript_interaction_ids[fork.id]):
                if interaction_id is None:
                    continue
                self._register_private_authority_alias_unlocked(
                    interaction_id,
                    field_name="interaction_id",
                    scope_session_id=fork.id,
                )
            if copied_checkpoint is not None:
                self._store_checkpoint_unlocked(fork.id, copied_checkpoint)
            return fork.model_copy(deep=True)

    async def load(self, session_id: str) -> Session | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return session.model_copy(deep=True)

    async def load_state(self, session_id: str) -> SessionStateSnapshot | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return SessionStateSnapshot(
                id=session.id,
                status=session.status,
                updated_at=session.updated_at,
                last_activity_at=session.last_activity_at,
            )

    async def inspect_identity(self, session_id: str) -> SessionInspectionIdentity:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            labels, label_count, labels_truncated = _bounded_session_inspection_labels(
                session.labels
            )
            return SessionInspectionIdentity(
                id=session.id,
                agent_name=session.agent_name,
                provider_name=session.provider_name,
                model=session.model,
                parent_session_id=session.parent_session_id,
                causal_budget_id=session.causal_budget_id,
                runtime_name=session.runtime_name,
                runtime_version=session.runtime_version,
                environment_name=session.environment_name,
                status=session.status,
                created_at=session.created_at,
                updated_at=session.updated_at,
                last_activity_at=session.last_activity_at,
                run_epoch=session.run_epoch,
                labels=labels,
                label_count=label_count,
                labels_truncated=labels_truncated,
            )

    async def update_status(self, session_id: str, status: SessionStatus) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(status, SessionStatus):
            raise ValueError("Session status must be a SessionStatus.")
        # Delegate to the guarded transition machine with every source status
        # allowed: preserves the unconditional (any -> status) setter contract
        # while sharing one write path and its not-found guard.
        return await self.transition_status(
            session_id,
            from_statuses=set(SessionStatus),
            to_status=status,
        )

    async def update_model(self, session_id: str, model: str) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        model = require_clean_nonblank(model, "model")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            now = datetime.now(UTC)
            updated = session.model_copy(
                update={
                    "model": model,
                    "updated_at": now,
                    "last_activity_at": now,
                }
            )
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    async def delete_session(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return  # idempotent: deleting a missing session is a no-op
            if session.status in DELETE_BLOCKED_SESSION_STATUSES:
                raise ValueError(
                    f"Cannot delete a session while it is {session.status}; "
                    f"interrupt it first: {session_id}"
                )
            checkpoint = self._checkpoints.get(session_id)
            deletion_now = datetime.now(UTC)
            active_recovery_claim_id = _active_unexpired_incomplete_recovery_claim_id(
                checkpoint,
                now=deletion_now,
            )
            if active_recovery_claim_id is not None:
                raise ValueError(
                    "Cannot delete a session while incomplete-session recovery claim "
                    f"{active_recovery_claim_id} is active: {session_id}"
                )
            run_operation = _session_run_operation_from_checkpoint(checkpoint)
            if run_operation is not None:
                raise ValueError(
                    "Cannot delete a session while terminal publication "
                    f"{run_operation.operation_id} is incomplete: {session_id}"
                )
            evidence_events: list[Event] = []
            for record in reversed(self._session_event_records.get(session_id, [])):
                if record.event.type not in _TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES:
                    continue
                evidence_events.append(record.event)
                if len(evidence_events) == _TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT:
                    break
            terminal_publication_block = _terminal_publication_delete_block_reason(
                session=session,
                checkpoint=checkpoint,
                evidence_events=evidence_events,
            )
            if terminal_publication_block is not None:
                raise ValueError(
                    f"Cannot delete a session while {terminal_publication_block}: {session_id}"
                )
            active_operation_id = _active_unexpired_session_operation_id(
                checkpoint,
                now=deletion_now,
            )
            if active_operation_id is not None:
                raise ValueError(
                    "Cannot delete a session while durable operation "
                    f"{active_operation_id} is active: {session_id}"
                )
            if MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY in self._session_operation_records.get(
                session_id, {}
            ):
                raise ValueError(
                    "Cannot delete a session while a model-completion stage is active: "
                    f"{session_id}"
                )
            owned_reservation_ids = {
                reservation_id
                for reservation_id, ownership in self._budget_reservation_identities.items()
                if ownership[0] == session_id
            }
            for reservation_id in owned_reservation_ids:
                terminal_events = [
                    event
                    for event in self._events.get(session_id, [])
                    if event.type
                    in {
                        EventType.BUDGET_RECONCILED,
                        EventType.BUDGET_RESERVATION_RELEASED,
                    }
                    and event.payload.get("reservation_id") == reservation_id
                ]
                delivery = (
                    None
                    if len(terminal_events) != 1
                    else self._persisted_event_side_effect_deliveries.get(
                        (session_id, terminal_events[0].id)
                    )
                )
                if (
                    delivery is None
                    or delivery.status is not PersistedEventSideEffectStatus.DELIVERED
                ):
                    raise ValueError(
                        "Cannot delete a session while a budget settlement audit "
                        f"event is pending: {session_id}"
                    )
            self._remove_session_parent_index_unlocked(session)
            self._sessions.pop(session_id, None)
            self._events.pop(session_id, None)
            self._event_ids.pop(session_id, None)
            self._persisted_event_side_effect_deliveries = {
                key: delivery
                for key, delivery in self._persisted_event_side_effect_deliveries.items()
                if key[0] != session_id
            }
            self._session_event_records.pop(session_id, None)
            self._interaction_event_records.pop(session_id, None)
            self._latest_interaction_event_records.pop(session_id, None)
            self._latest_interaction_event_records_by_sequence.pop(session_id, None)
            self._pending_action_event_records.pop(session_id, None)
            self._pending_action_index_scopes.pop(session_id, None)
            self._pending_action_latest_barrier_records.pop(session_id, None)
            self._transcripts.pop(session_id, None)
            self._transcript_interaction_ids.pop(session_id, None)
            self._transcript_indices_by_interaction.pop(session_id, None)
            self._deferred_interaction_inputs.pop(session_id, None)
            self._checkpoints.pop(session_id, None)
            self._session_operation_records.pop(session_id, None)
            self._pending_action_session_ids.discard(session_id)
            self._queued_session_messages_by_idempotency.pop(session_id, None)
            self._session_message_delivery_records = {
                delivery_id: record
                for delivery_id, record in self._session_message_delivery_records.items()
                if record.session_id != session_id
            }
            for delivery_mode in SessionMessageDeliveryMode:
                self._pending_session_messages.pop((session_id, delivery_mode), None)
            self._event_records = [
                record for record in self._event_records if record.event.session_id != session_id
            ]
            self._event_records_by_id = {
                key: record
                for key, record in self._event_records_by_id.items()
                if key[0] != session_id
            }
            self._workflow_step_event_records = {
                key: records
                for key, records in self._workflow_step_event_records.items()
                if key[0] != session_id
            }
            self._workflow_fenced_step_event_records = {
                key: records
                for key, records in self._workflow_fenced_step_event_records.items()
                if key[0] != session_id
            }
            self._workflow_attempt_event_records = {
                key: records
                for key, records in self._workflow_attempt_event_records.items()
                if key[0] != session_id
            }
            self._workflow_active_attempt_ids = {
                key: attempt_id
                for key, attempt_id in self._workflow_active_attempt_ids.items()
                if key[0] != session_id
            }
            for event_type, records in list(self._type_event_records.items()):
                remaining = [record for record in records if record.event.session_id != session_id]
                if remaining:
                    self._type_event_records[event_type] = remaining
                else:
                    del self._type_event_records[event_type]
            # Mirror the SQL FK's ON DELETE SET NULL: children keep loading, parent ref
            # cleared. The branch index makes this proportional to the deleted
            # session's direct children rather than the complete registry.
            child_keys = self._child_session_keys_by_parent.pop(session_id, [])
            for _, child_id in child_keys:
                child = self._sessions.get(child_id)
                if child is None or child.parent_session_id != session_id:
                    raise RuntimeError("Inconsistent in-memory session topology index.")
                self._sessions[child_id] = child.model_copy(update={"parent_session_id": None})

    async def update_labels(self, session_id: str, labels: dict[str, str]) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        new_labels = copy_label_map(labels, "labels", allow_reserved=False)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            now = datetime.now(UTC)
            updated = session.model_copy(update={"labels": new_labels, "updated_at": now})
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    async def update_metadata(self, session_id: str, metadata: dict[str, Any]) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        user_metadata = copy_session_user_metadata(metadata)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            new_metadata = replace_session_user_metadata(session.metadata, user_metadata)
            now = datetime.now(UTC)
            updated = session.model_copy(update={"metadata": new_metadata, "updated_at": now})
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    async def transition_status(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            if session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status transition not allowed: {session.status} -> {to_status}"
                )

            now = datetime.now(UTC)
            updated = session.model_copy(
                update={
                    "status": to_status,
                    "updated_at": now,
                    "last_activity_at": now,
                    "run_epoch": session.run_epoch + (to_status == SessionStatus.RUNNING),
                }
            )
            self._sessions[session_id] = updated
            result = updated.model_copy(deep=True)
            if to_status == SessionStatus.RUNNING:
                _activate_session_run_fence(result)
            return result

    async def transition_status_and_checkpoint(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_transform: CheckpointTransform,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
        continued_interaction_id: str | None = None,
        defer_interaction_source: bool = False,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        admission = _copy_transition_interaction_admission(
            session_id,
            interaction_started_event,
            interaction_source_messages,
            continued_interaction_id=continued_interaction_id,
            defer_interaction_source=defer_interaction_source,
        )
        if admission is not None and to_status is not SessionStatus.RUNNING:
            raise ValueError("Interaction admission requires a transition to running.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            if session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status transition not allowed: {session.status} -> {to_status}"
                )

            current_checkpoint = self._checkpoints.get(session_id)
            transformed_checkpoint = checkpoint_transform(
                session.model_copy(deep=True),
                None if current_checkpoint is None else deepcopy(current_checkpoint),
            )
            if transformed_checkpoint is not None:
                transformed_checkpoint = copy_durable_json_object(
                    transformed_checkpoint,
                    "checkpoint",
                )

            if admission is not None:
                _, interaction_id, _, defer_source = admission
                existing_deferred = self._deferred_interaction_inputs.get(session_id)
                if existing_deferred is not None and (
                    not defer_source or existing_deferred[0] != interaction_id
                ):
                    raise RuntimeError("Session already has deferred interaction input.")

            now = datetime.now(UTC)
            updated = session.model_copy(
                update={
                    "status": to_status,
                    "updated_at": now,
                    "last_activity_at": now,
                    "run_epoch": session.run_epoch + (to_status == SessionStatus.RUNNING),
                }
            )
            prepared_events: _PreparedInMemoryEventAppend | None = None
            if admission is not None and admission[0] is not None:
                prepared_events = self._prepare_event_append_unlocked(
                    updated,
                    [admission[0]],
                )
            prepared_checkpoint = (
                None
                if transformed_checkpoint is None
                else self._prepare_checkpoint_store_unlocked(
                    session_id,
                    transformed_checkpoint,
                    additional_event_records=(
                        ()
                        if prepared_events is None
                        else tuple(event.record for event in prepared_events.events)
                    ),
                )
            )

            self._sessions[session_id] = updated
            if prepared_checkpoint is not None:
                self._apply_checkpoint_store_unlocked(session_id, prepared_checkpoint)
            if admission is not None:
                _, interaction_id, source_messages, defer_source = admission
                if prepared_events is not None:
                    self._sessions[session_id] = self._apply_event_append_unlocked(
                        updated,
                        prepared_events,
                    )
                if defer_source:
                    self._deferred_interaction_inputs[session_id] = (
                        interaction_id,
                        source_messages,
                    )
                else:
                    self._transcripts[session_id].extend(source_messages)
                    self._extend_transcript_attribution_unlocked(
                        session_id, [interaction_id] * len(source_messages)
                    )
                updated = self._sessions[session_id]
            result = updated.model_copy(deep=True)
            if to_status == SessionStatus.RUNNING:
                _activate_session_run_fence(result)
            return result

    async def transition_status_if_no_queued_messages(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            if session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status transition not allowed: {session.status} -> {to_status}"
                )
            if any(
                (session_id, delivery_mode) in self._pending_session_messages
                for delivery_mode in SessionMessageDeliveryMode
            ):
                raise SessionQueuedMessagesPending(
                    f"Session has durable queued messages: {session_id}"
                )
            now = datetime.now(UTC)
            updated = session.model_copy(
                update={
                    "status": to_status,
                    "updated_at": now,
                    "last_activity_at": now,
                    "run_epoch": session.run_epoch + (to_status == SessionStatus.RUNNING),
                }
            )
            self._sessions[session_id] = updated
            result = updated.model_copy(deep=True)
            if to_status == SessionStatus.RUNNING:
                _activate_session_run_fence(result)
            return result

    async def publish_interaction_transition(
        self,
        session_id: str,
        *,
        event: Event,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        only_if_no_queued_messages: bool = False,
    ) -> InteractionTransitionResult:
        (
            session_id,
            copied_event,
            allowed_statuses,
            target_status,
            conditional,
        ) = _prepare_interaction_transition(
            session_id,
            event=event,
            from_statuses=from_statuses,
            to_status=to_status,
            only_if_no_queued_messages=only_if_no_queued_messages,
        )
        receipt_storage_key = _interaction_transition_storage_key(copied_event.id)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            operation_records = self._session_operation_records.setdefault(session_id, {})
            receipt_record = operation_records.get(receipt_storage_key)
            existing = self._event_records_by_id.get((session_id, copied_event.id))
            if receipt_record is not None:
                receipt = _reconstruct_interaction_transition_receipt(
                    receipt_record,
                    event=copied_event,
                    from_statuses=allowed_statuses,
                    to_status=target_status,
                    only_if_no_queued_messages=conditional,
                )
                if existing is not None and existing.event != receipt.event:
                    raise RuntimeError(
                        "Interaction transition receipt conflicts with retained event history."
                    )
                return InteractionTransitionResult(
                    session=receipt.session,
                    event=receipt.event,
                    status_changed=receipt.status_changed,
                    replayed=True,
                )
            if existing is not None:
                raise RuntimeError(
                    "Interaction transition event exists without its immutable receipt."
                )
            if session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status transition not allowed: {session.status} -> {target_status}"
                )
            queued = conditional and any(
                (session_id, delivery_mode) in self._pending_session_messages
                for delivery_mode in SessionMessageDeliveryMode
            )
            now = datetime.now(UTC)
            updated = self._append_events_unlocked(session, [copied_event])
            if not queued:
                updated = updated.model_copy(
                    update={
                        "status": target_status,
                        "updated_at": now,
                        "last_activity_at": now,
                    }
                )
            self._sessions[session_id] = updated
            operation_records[receipt_storage_key] = _interaction_transition_receipt_record(
                session=updated,
                event=copied_event,
                from_statuses=allowed_statuses,
                to_status=target_status,
                only_if_no_queued_messages=conditional,
                status_changed=not queued,
            )
            return InteractionTransitionResult(
                session=updated,
                event=copied_event,
                status_changed=not queued,
            )

    async def fence_stalled_run(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        inactive_before: datetime,
    ) -> Session | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(statuses, "statuses")
        if inactive_before.tzinfo is None or inactive_before.utcoffset() is None:
            raise ValueError("inactive_before must be timezone-aware.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if session.status not in allowed_statuses or session.last_activity_at > inactive_before:
                return None
            fenced = session.model_copy(
                update={
                    "run_epoch": session.run_epoch + 1,
                    "last_activity_at": datetime.now(UTC),
                }
            )
            self._sessions[session_id] = fenced
            result = fenced.model_copy(deep=True)
            _activate_session_run_fence(result)
            return result

    async def fence_run_and_transform_checkpoint(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        checkpoint_transform: CheckpointTransform,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(statuses, "statuses")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if session.status not in allowed_statuses:
                raise SessionStatusConflict(f"Session status cannot be fenced: {session.status}")
            current = self._checkpoints.get(session_id)
            transformed = checkpoint_transform(
                session.model_copy(deep=True),
                None if current is None else deepcopy(current),
            )
            if transformed is None:
                raise ValueError("Fenced checkpoint transform must return a checkpoint.")
            transformed = copy_durable_json_object(transformed, "checkpoint")
            fenced = session.model_copy(
                update={
                    "run_epoch": session.run_epoch + 1,
                    "last_activity_at": datetime.now(UTC),
                }
            )
            self._store_checkpoint_unlocked(session_id, transformed)
            self._sessions[session_id] = fenced
            result = fenced.model_copy(deep=True)
            _activate_session_run_fence(result)
            return result

    async def release_run_fence(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        expected_run_epoch = _current_session_run_epoch(session_id)
        if expected_run_epoch is None:
            _deactivate_session_interaction(session_id)
            return
        try:
            async with self._lock:
                session = self._sessions.get(session_id)
                if session is not None and session.run_epoch == expected_run_epoch:
                    self._sessions[session_id] = session.model_copy(
                        update={"run_epoch": session.run_epoch + 1}
                    )
        finally:
            _deactivate_session_run_fence(session_id)
            _deactivate_session_interaction(session_id)

    async def append_event(self, session_id: str, event: Event) -> None:
        await self.append_events(session_id, [event])

    async def claim_budget_reservation_identity(
        self,
        *,
        reservation_id: str,
        publication_session_id: str,
        publication_id: str,
    ) -> None:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        publication_session_id = require_clean_nonblank(
            publication_session_id,
            "publication_session_id",
        )
        publication_id = require_clean_nonblank(publication_id, "publication_id")
        async with self._lock:
            session = self._sessions.get(publication_session_id)
            if session is None:
                raise KeyError(f"Session not found: {publication_session_id}")
            _assert_session_run_epoch(publication_session_id, session)
            existing = self._budget_reservation_identities.get(reservation_id)
            if existing is None:
                self._budget_reservation_identities[reservation_id] = (
                    publication_session_id,
                    publication_id,
                    False,
                )
                return
            if existing[:2] != (publication_session_id, publication_id):
                raise BudgetReservationIdentityConflict(
                    "Budget ledger reused a reservation identity."
                )

    def _prepare_event_append_unlocked(
        self,
        session: Session,
        events: Iterable[Event],
        *,
        events_are_detached: bool = False,
    ) -> _PreparedInMemoryEventAppend:
        from cayu.runtime.pending_actions import (
            pending_action_event_has_unknown_round_call,
            pending_action_event_is_from_different_tool_round,
            pending_action_event_lookup_id,
            pending_action_event_retains_history,
            pending_action_evidence_round_from_checkpoint,
            pending_action_lookup_key,
            project_pending_action_event_record,
        )

        session_id = session.id
        existing_ids = self._event_ids[session_id]
        event_batch = tuple(events)
        batch_ids: set[str] = set()
        budget_reservation_publications: dict[str, str] = {}
        for event in event_batch:
            if event.id in existing_ids or event.id in batch_ids:
                raise ValueError(f"Event already exists for session {session_id}: {event.id}")
            batch_ids.add(event.id)
            if event.type != EventType.BUDGET_RESERVED:
                continue
            raw_reservation_id = event.payload.get("reservation_id")
            if type(raw_reservation_id) is not str:
                continue
            reservation_id = raw_reservation_id
            existing = self._budget_reservation_identities.get(reservation_id)
            if existing is not None and (
                existing[:2] != (event.session_id, event.id) or existing[2]
            ):
                raise BudgetReservationIdentityConflict(
                    "Budget ledger reused a reservation identity."
                )
            previous_publication_id = budget_reservation_publications.setdefault(
                reservation_id,
                event.id,
            )
            if previous_publication_id != event.id:
                raise BudgetReservationIdentityConflict(
                    "Budget ledger reused a reservation identity."
                )

        try:
            pending_round = pending_action_evidence_round_from_checkpoint(
                self._checkpoints.get(session_id)
            )
        except (TypeError, ValueError):
            pending_round = None
        index_scope = self._pending_action_index_scopes.get(session_id)
        active_lookup_keys = frozenset() if index_scope is None else index_scope[0]
        round_lookup_key = (
            None
            if pending_round is None
            else pending_action_lookup_key(pending_round.tool_round_id)
        )
        prepared: list[_PreparedInMemoryEvent] = []
        next_sequence = self._next_event_sequence
        for event in event_batch:
            stored_event = event if events_are_detached else event.model_copy(deep=True)
            record = EventRecord(sequence=next_sequence, event=stored_event)
            event_type = str(stored_event.type)
            projected_record: EventRecord | None = None
            retention_keys: list[str] = []
            history_keys: list[str] = []
            if event_type in PENDING_ACTION_EVENT_TYPE_VALUES:
                retain_for_scope = (
                    pending_round is not None
                    and pending_action_event_has_unknown_round_call(
                        stored_event,
                        pending_round,
                    )
                )
                lookup_id = pending_action_event_lookup_id(stored_event)
                if lookup_id is not None:
                    lookup_key = pending_action_lookup_key(lookup_id)
                    if lookup_key in active_lookup_keys:
                        history_keys.append(lookup_key)
                        if not (
                            pending_action_event_retains_history(event_type)
                            and (
                                pending_round is None
                                or pending_action_event_is_from_different_tool_round(
                                    stored_event,
                                    pending_round,
                                )
                            )
                        ):
                            retention_keys.append(lookup_key)
                if retain_for_scope:
                    assert round_lookup_key is not None
                    retention_keys.append(round_lookup_key)
                    history_keys.append(round_lookup_key)
                if (
                    event_type in PENDING_ACTION_BARRIER_EVENT_TYPE_VALUES
                    or retention_keys
                    or history_keys
                ):
                    projected_record = project_pending_action_event_record(record)
            delivery = (
                None
                if event_type == str(EventType.RUNTIME_SINK_FAILED)
                else PersistedEventSideEffectDelivery(
                    session_id=session_id,
                    event_id=stored_event.id,
                    event_sequence=record.sequence,
                    status=PersistedEventSideEffectStatus.PENDING,
                )
            )
            prepared.append(
                _PreparedInMemoryEvent(
                    record=record,
                    event_type=event_type,
                    projected_record=projected_record,
                    retained_lookup_keys=tuple(dict.fromkeys(retention_keys)),
                    history_lookup_keys=tuple(dict.fromkeys(history_keys)),
                    delivery=delivery,
                )
            )
            next_sequence += 1
        return _PreparedInMemoryEventAppend(
            events=tuple(prepared),
            next_sequence=next_sequence,
            budget_reservation_publications=tuple(budget_reservation_publications.items()),
        )

    def _apply_event_append_unlocked(
        self,
        session: Session,
        prepared: _PreparedInMemoryEventAppend,
        *,
        activity_at: datetime | None = None,
    ) -> Session:
        from cayu.runtime.pending_actions import retain_pending_action_index_record

        session_id = session.id
        existing_ids = self._event_ids[session_id]
        session_records = self._session_event_records.setdefault(session_id, [])
        from cayu.runtime.interactions import INTERACTION_LIFECYCLE_EVENT_TYPES

        lifecycle_types = {str(event_type) for event_type in INTERACTION_LIFECYCLE_EVENT_TYPES}
        for prepared_event in prepared.events:
            record = prepared_event.record
            event_type = prepared_event.event_type
            projected_record = prepared_event.projected_record
            retained_lookup_keys = prepared_event.retained_lookup_keys
            history_lookup_keys = prepared_event.history_lookup_keys
            stored_event = record.event
            self._events[session_id].append(stored_event)
            self._event_records.append(record)
            self._event_records_by_id[(session_id, stored_event.id)] = record
            session_records.append(record)
            interaction_id = stored_event.interaction_id
            if interaction_id is not None:
                self._register_private_authority_alias_unlocked(
                    interaction_id,
                    field_name="interaction_id",
                    scope_session_id=session_id,
                )
                self._interaction_event_records.setdefault(session_id, {}).setdefault(
                    interaction_id,
                    [],
                ).append(record)
            if stored_event.type == EventType.TURN_COMPLETED:
                nested_interaction_ids = stored_event.payload.get("interaction_ids")
                if type(nested_interaction_ids) is list:
                    for nested_interaction_id in dict.fromkeys(
                        value
                        for value in nested_interaction_ids
                        if type(value) is str and value.strip()
                    ):
                        self._register_private_authority_alias_unlocked(
                            nested_interaction_id,
                            field_name="interaction_id",
                            scope_session_id=session_id,
                        )
            if interaction_id is not None and event_type in lifecycle_types:
                latest_by_id = self._latest_interaction_event_records.setdefault(session_id, {})
                latest_by_sequence = self._latest_interaction_event_records_by_sequence.setdefault(
                    session_id, []
                )
                previous = latest_by_id.get(interaction_id)
                if previous is not None:
                    previous_index = bisect_left(
                        latest_by_sequence,
                        previous.sequence,
                        key=lambda candidate: candidate.sequence,
                    )
                    if (
                        previous_index >= len(latest_by_sequence)
                        or latest_by_sequence[previous_index].sequence != previous.sequence
                    ):
                        raise RuntimeError(
                            "In-memory interaction latest-event indexes are inconsistent."
                        )
                    latest_by_sequence.pop(previous_index)
                latest_by_id[interaction_id] = record
                latest_by_sequence.append(record)
            if projected_record is not None:
                if event_type in PENDING_ACTION_BARRIER_EVENT_TYPE_VALUES:
                    self._pending_action_latest_barrier_records[session_id] = projected_record
                else:
                    by_lookup_id = self._pending_action_event_records.setdefault(session_id, {})
                    event_history = self._pending_action_event_history.setdefault(
                        session_id,
                        {},
                    )
                    for retention_key in retained_lookup_keys:
                        retain_pending_action_index_record(
                            by_lookup_id.setdefault(retention_key, {}),
                            projected_record,
                        )
                    for history_key in history_lookup_keys:
                        event_history.setdefault(history_key, []).append(record)
            self._type_event_records.setdefault(event_type, []).append(record)
            workflow_name = stored_event.workflow_name
            workflow_attempt_id = stored_event.payload.get("attempt_id")
            workflow_step_id = stored_event.payload.get("step_id")
            if (
                stored_event.type == WORKFLOW_ATTEMPT_EVENT_TYPE
                and isinstance(workflow_name, str)
                and workflow_name
                and isinstance(workflow_attempt_id, str)
                and workflow_attempt_id
            ):
                self._workflow_attempt_event_records.setdefault(
                    (session_id, workflow_name),
                    [],
                ).append(record)
                self._workflow_active_attempt_ids[(session_id, workflow_name)] = workflow_attempt_id
            if (
                isinstance(workflow_name, str)
                and workflow_name
                and isinstance(workflow_attempt_id, str)
                and workflow_attempt_id
                and isinstance(workflow_step_id, str)
                and workflow_step_id
            ):
                self._workflow_step_event_records.setdefault(
                    (
                        session_id,
                        workflow_name,
                        workflow_attempt_id,
                        workflow_step_id,
                        event_type,
                    ),
                    [],
                ).append(record)
                if (
                    self._workflow_active_attempt_ids.get((session_id, workflow_name))
                    == workflow_attempt_id
                ):
                    self._workflow_fenced_step_event_records.setdefault(
                        (
                            session_id,
                            workflow_name,
                            workflow_step_id,
                            event_type,
                        ),
                        [],
                    ).append(record)
            existing_ids.add(stored_event.id)
            if prepared_event.delivery is not None:
                self._persisted_event_side_effect_deliveries[(session_id, stored_event.id)] = (
                    prepared_event.delivery
                )
        self._budget_reservation_identities.update(
            {
                reservation_id: (session_id, publication_id, True)
                for reservation_id, publication_id in (prepared.budget_reservation_publications)
            }
        )
        self._next_event_sequence = prepared.next_sequence
        if not prepared.events:
            return session
        return session.model_copy(update={"last_activity_at": activity_at or datetime.now(UTC)})

    def _append_events_unlocked(
        self,
        session: Session,
        events: list[Event],
    ) -> Session:
        prepared = self._prepare_event_append_unlocked(session, events)
        return self._apply_event_append_unlocked(session, prepared)

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        session_id, copied_events = _copy_session_event_batch(session_id, events)

        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            self._sessions[session_id] = self._append_events_unlocked(session, copied_events)

    async def append_workflow_step_started(
        self,
        session_id: str,
        event: Event,
        *,
        workflow_name: str,
        attempt_id: str,
    ) -> bool:
        session_id, copied_event, workflow_name, attempt_id = _copy_workflow_step_reservation(
            session_id,
            event,
            workflow_name=workflow_name,
            attempt_id=attempt_id,
        )
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            if self._workflow_active_attempt_ids.get((session_id, workflow_name)) != attempt_id:
                return False
            if (session_id, copied_event.id) in self._event_records_by_id:
                return False
            self._sessions[session_id] = self._append_events_unlocked(session, [copied_event])
            return True

    async def load_mcp_manifest_baselines(
        self,
        history_keys: tuple[str, ...],
    ) -> McpManifestBaselineLoadResult:
        keys = _validate_mcp_manifest_history_keys(history_keys)
        async with self._lock:
            return McpManifestBaselineLoadResult(
                baselines={
                    key: self._mcp_manifest_baselines[key].model_copy(deep=True)
                    for key in keys
                    if key in self._mcp_manifest_baselines
                },
            )

    async def compare_and_publish_mcp_manifest_checks(
        self,
        session_id: str,
        *,
        expected_generations: dict[str, int | None],
        baseline_updates: dict[str, McpManifestBaseline],
        events: list[Event],
    ) -> McpManifestPublicationResult:
        (
            session_id,
            expected,
            updates,
            copied_events,
        ) = _copy_mcp_manifest_publication(
            session_id,
            expected_generations=expected_generations,
            baseline_updates=baseline_updates,
            events=events,
        )

        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            current = {
                key: self._mcp_manifest_baselines[key]
                for key in expected
                if key in self._mcp_manifest_baselines
            }
            if any(
                expected_generation
                != (None if (baseline := current.get(key)) is None else baseline.generation)
                for key, expected_generation in expected.items()
            ):
                return McpManifestPublicationResult(
                    published=False,
                    baselines={
                        key: baseline.model_copy(deep=True) for key, baseline in current.items()
                    },
                )
            _validate_mcp_manifest_publication_state(
                expected_generations=expected,
                current_baselines=current,
                baseline_updates=updates,
                events=copied_events,
            )
            self._sessions[session_id] = self._append_events_unlocked(session, copied_events)
            for key, baseline in updates.items():
                self._mcp_manifest_baselines[key] = baseline.model_copy(deep=True)
            return McpManifestPublicationResult(
                published=True,
                baselines={
                    key: self._mcp_manifest_baselines[key].model_copy(deep=True)
                    for key in expected
                    if key in self._mcp_manifest_baselines
                },
            )

    async def claim_persisted_event_side_effect(
        self,
        *,
        session_id: str | None = None,
        event_id: str | None = None,
        lease_seconds: float = 300.0,
    ) -> PersistedEventSideEffectClaim | None:
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")
        if event_id is not None:
            event_id = require_clean_nonblank(event_id, "event_id")
        if (session_id is None) != (event_id is None):
            raise ValueError("session_id and event_id must be supplied together.")
        if type(lease_seconds) not in {int, float} or lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than 0.")
        async with self._lock:
            now = datetime.now(UTC)
            deliveries = sorted(
                self._persisted_event_side_effect_deliveries.values(),
                key=lambda delivery: delivery.event_sequence,
            )
            for delivery in deliveries:
                if session_id is not None and (
                    delivery.session_id != session_id or delivery.event_id != event_id
                ):
                    continue
                if delivery.status in {
                    PersistedEventSideEffectStatus.DELIVERED,
                    PersistedEventSideEffectStatus.DEAD_LETTERED,
                }:
                    continue
                if (
                    delivery.status is PersistedEventSideEffectStatus.LEASED
                    and delivery.lease_expires_at is not None
                    and delivery.lease_expires_at > now
                ):
                    continue
                if (
                    delivery.status is PersistedEventSideEffectStatus.FAILED
                    and delivery.next_attempt_at is not None
                    and delivery.next_attempt_at > now
                ):
                    continue
                record = self._event_records_by_id.get((delivery.session_id, delivery.event_id))
                if record is None:
                    raise RuntimeError("Persisted side-effect delivery lost its source event.")
                claim = PersistedEventSideEffectClaim(
                    session_id=delivery.session_id,
                    event_id=delivery.event_id,
                    event_sequence=delivery.event_sequence,
                    event=record.event,
                    attempt=delivery.attempts + 1,
                    lease_expires_at=now + timedelta(seconds=float(lease_seconds)),
                )
                self._persisted_event_side_effect_deliveries[
                    (delivery.session_id, delivery.event_id)
                ] = delivery.model_copy(
                    update={
                        "status": PersistedEventSideEffectStatus.LEASED,
                        "attempts": claim.attempt,
                        "claim_id": claim.claim_id,
                        "lease_expires_at": claim.lease_expires_at,
                        "next_attempt_at": None,
                        "last_error": None,
                        "updated_at": now,
                    },
                    deep=True,
                )
                return claim
            return None

    async def get_persisted_event_side_effect_delivery(
        self,
        *,
        session_id: str,
        event_id: str,
    ) -> PersistedEventSideEffectDelivery | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        event_id = require_clean_nonblank(event_id, "event_id")
        async with self._lock:
            delivery = self._persisted_event_side_effect_deliveries.get((session_id, event_id))
            return None if delivery is None else delivery.model_copy(deep=True)

    async def mark_persisted_event_side_effect_delivered(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> PersistedEventSideEffectDelivery:
        claim = PersistedEventSideEffectClaim.model_validate(claim)
        async with self._lock:
            delivery = self._matching_persisted_event_side_effect_claim_unlocked(claim)
            updated = delivery.model_copy(
                update={
                    "status": PersistedEventSideEffectStatus.DELIVERED,
                    "claim_id": None,
                    "lease_expires_at": None,
                    "next_attempt_at": None,
                    "last_error": None,
                    "updated_at": datetime.now(UTC),
                },
                deep=True,
            )
            self._persisted_event_side_effect_deliveries[(claim.session_id, claim.event_id)] = (
                updated
            )
            return updated.model_copy(deep=True)

    async def mark_persisted_event_side_effect_failed(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        error: str,
        max_attempts: int,
        retry_delay_seconds: float,
    ) -> PersistedEventSideEffectDelivery:
        claim = PersistedEventSideEffectClaim.model_validate(claim)
        error = validate_persisted_event_side_effect_error(error)
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be an integer greater than or equal to 1.")
        if (
            type(retry_delay_seconds) not in {int, float}
            or not math.isfinite(retry_delay_seconds)
            or retry_delay_seconds < 0
        ):
            raise ValueError("retry_delay_seconds must be a finite non-negative number.")
        dead_lettered = claim.attempt >= max_attempts
        async with self._lock:
            now = datetime.now(UTC)
            delivery = self._matching_persisted_event_side_effect_claim_unlocked(claim)
            updated = delivery.model_copy(
                update={
                    "status": (
                        PersistedEventSideEffectStatus.DEAD_LETTERED
                        if dead_lettered
                        else PersistedEventSideEffectStatus.FAILED
                    ),
                    "claim_id": None,
                    "lease_expires_at": None,
                    "next_attempt_at": (
                        None
                        if dead_lettered
                        else now + timedelta(seconds=float(retry_delay_seconds))
                    ),
                    "last_error": error,
                    "updated_at": now,
                },
                deep=True,
            )
            self._persisted_event_side_effect_deliveries[(claim.session_id, claim.event_id)] = (
                updated
            )
            return updated.model_copy(deep=True)

    async def list_persisted_event_side_effect_deliveries(
        self,
        *,
        statuses: set[PersistedEventSideEffectStatus] | None = None,
        claimable_only: bool = False,
        after_sequence: int | None = None,
        limit: int = 100,
    ) -> list[PersistedEventSideEffectDelivery]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000.")
        if type(claimable_only) is not bool:
            raise TypeError("claimable_only must be a bool.")
        if after_sequence is not None and (type(after_sequence) is not int or after_sequence < 0):
            raise ValueError("after_sequence must be a non-negative integer.")
        selected_statuses = (
            None
            if statuses is None
            else {PersistedEventSideEffectStatus(status) for status in statuses}
        )
        async with self._lock:
            now = datetime.now(UTC)
            deliveries = sorted(
                self._persisted_event_side_effect_deliveries.values(),
                key=lambda delivery: delivery.event_sequence,
            )
            return [
                delivery.model_copy(deep=True)
                for delivery in deliveries
                if after_sequence is None or delivery.event_sequence > after_sequence
                if selected_statuses is None or delivery.status in selected_statuses
                if not claimable_only
                or (
                    delivery.status
                    in {
                        PersistedEventSideEffectStatus.PENDING,
                        PersistedEventSideEffectStatus.FAILED,
                    }
                    and (
                        delivery.status is not PersistedEventSideEffectStatus.FAILED
                        or delivery.next_attempt_at is None
                        or delivery.next_attempt_at <= now
                    )
                )
                or (
                    delivery.status is PersistedEventSideEffectStatus.LEASED
                    and delivery.lease_expires_at is not None
                    and delivery.lease_expires_at <= now
                )
            ][:limit]

    def _matching_persisted_event_side_effect_claim_unlocked(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> PersistedEventSideEffectDelivery:
        delivery = self._persisted_event_side_effect_deliveries.get(
            (claim.session_id, claim.event_id)
        )
        if delivery is None:
            raise ValueError("Persisted event side-effect delivery was not found.")
        if (
            delivery.status is not PersistedEventSideEffectStatus.LEASED
            or delivery.claim_id != claim.claim_id
            or delivery.attempts != claim.attempt
        ):
            raise PersistedEventSideEffectClaimLost(
                "Persisted event side-effect claim is no longer active."
            )
        return delivery

    async def enqueue_session_message(
        self,
        request: EnqueueSessionMessageRequest,
    ) -> EnqueueSessionMessageResult:
        request = copy_enqueue_session_message_request(request)
        async with self._lock:
            session = self._sessions.get(request.session_id)
            if session is None:
                raise KeyError(f"Session not found: {request.session_id}")
            messages_by_idempotency = self._queued_session_messages_by_idempotency.get(
                request.session_id
            )
            existing = (
                None
                if messages_by_idempotency is None
                else messages_by_idempotency.get(request.idempotency_key)
            )
            if existing is not None:
                _validate_equivalent_queued_session_message(existing, request)
                record = self._event_records_by_id.get(
                    (request.session_id, existing.accepted_event_id)
                )
                if record is None:
                    raise RuntimeError(
                        "Queued session message is missing its durable acceptance event."
                    )
                return EnqueueSessionMessageResult(
                    message=existing,
                    event=record.event,
                    replayed=True,
                )
            if session.status not in {SessionStatus.PENDING, SessionStatus.RUNNING}:
                raise SessionStatusConflict(
                    "Session messages may be enqueued only while a session is pending or running."
                )
            accepted_at = datetime.now(UTC)
            queue_id = str(uuid4())
            ordering_key = self._next_session_message_ordering_key
            accepted_event = Event(
                type=EventType.SESSION_MESSAGE_QUEUED,
                session_id=session.id,
                agent_name=session.agent_name,
                environment_name=session.environment_name,
                timestamp=accepted_at,
                payload=_queued_session_message_event_payload(
                    queue_id=queue_id,
                    delivery_mode=request.delivery_mode,
                    ordering_key=ordering_key,
                    actor=request.requested_by,
                    run_epoch=session.run_epoch,
                    transcript_cursor=len(self._transcripts.get(session.id, [])),
                ),
            )
            queued_message = SessionQueuedMessage(
                queue_id=queue_id,
                session_id=session.id,
                idempotency_key=request.idempotency_key,
                content=request.content,
                delivery_mode=request.delivery_mode,
                status=SessionMessageQueueStatus.QUEUED,
                ordering_key=ordering_key,
                accepted_run_epoch=session.run_epoch,
                accepted_transcript_cursor=len(self._transcripts.get(session.id, [])),
                accepted_event_id=accepted_event.id,
                accepted_at=accepted_at,
                requested_by=_audit_resolution_actor(request.requested_by),
            )
            self._sessions[session.id] = self._append_events_unlocked(
                session,
                [accepted_event],
            )
            self._queued_session_messages_by_idempotency.setdefault(session.id, {})[
                queued_message.idempotency_key
            ] = queued_message
            pending_key = (session.id, queued_message.delivery_mode)
            self._pending_session_messages.setdefault(pending_key, deque()).append(queued_message)
            self._next_session_message_ordering_key += 1
            return EnqueueSessionMessageResult(
                message=queued_message,
                event=accepted_event,
            )

    async def deliver_queued_session_messages(
        self,
        session_id: str,
        *,
        include_on_idle: bool,
        delivery_id: str | None = None,
        eligible_through: int | None = None,
        limit: int = SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
        interaction_id: str | None = None,
        interaction_started_event: Event | None = None,
    ) -> SessionMessageDeliveryBatch:
        session_id = require_clean_nonblank(session_id, "session_id")
        delivery_id = (
            str(uuid4())
            if delivery_id is None
            else require_clean_nonblank(delivery_id, "delivery_id")
        )
        if type(include_on_idle) is not bool:
            raise TypeError("include_on_idle must be a bool.")
        if interaction_id is not None:
            interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
        interaction_started_event = _copy_queued_interaction_started_event(
            session_id,
            interaction_id,
            interaction_started_event,
        )
        eligible_through = _validate_message_delivery_eligible_through(eligible_through)
        if type(limit) is not int or not 1 <= limit <= SESSION_MESSAGE_DELIVERY_BATCH_LIMIT:
            raise ValueError(f"limit must be between 1 and {SESSION_MESSAGE_DELIVERY_BATCH_LIMIT}.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            existing_delivery = self._session_message_delivery_records.get(delivery_id)
            if existing_delivery is not None:
                _validate_equivalent_message_delivery(
                    existing_delivery,
                    session_id=session_id,
                    include_on_idle=include_on_idle,
                    eligible_through=eligible_through,
                    limit=limit,
                    interaction_id=interaction_id,
                    interaction_started_event=interaction_started_event,
                )
                return existing_delivery.batch.model_copy(
                    update={"replayed": True},
                    deep=True,
                )
            if session.status != SessionStatus.RUNNING:
                raise SessionStatusConflict(
                    "Queued session messages may be delivered only while running."
                )
            boundary = (
                self._next_session_message_ordering_key - 1
                if eligible_through is None
                else eligible_through
            )

            selected_key: tuple[str, SessionMessageDeliveryMode] | None = None
            selected: list[SessionQueuedMessage] = []
            delivery_modes: tuple[SessionMessageDeliveryMode, ...] = (
                SessionMessageDeliveryMode.NEXT_TURN,
            )
            if include_on_idle:
                delivery_modes += (SessionMessageDeliveryMode.ON_IDLE,)
            for delivery_mode in delivery_modes:
                pending_key = (session_id, delivery_mode)
                pending = self._pending_session_messages.get(pending_key)
                if pending is None or pending[0].ordering_key > boundary:
                    continue
                selected_key = pending_key
                for message in pending:
                    if len(selected) >= limit or message.ordering_key > boundary:
                        break
                    selected.append(message)
                break
            if not selected:
                batch = SessionMessageDeliveryBatch(
                    delivery_id=delivery_id,
                    interaction_id=interaction_id,
                    eligible_through=boundary,
                    has_more=False,
                )
                self._session_message_delivery_records[delivery_id] = (
                    _InMemoryMessageDeliveryRecord(
                        session_id=session_id,
                        include_on_idle=include_on_idle,
                        requested_eligible_through=eligible_through,
                        limit=limit,
                        interaction_id=interaction_id,
                        interaction_started_event=interaction_started_event,
                        batch=batch,
                    )
                )
                return batch
            if selected_key is None:
                raise RuntimeError("Queued message selection lost its pending index.")

            transcript_cursor = len(self._transcripts.get(session_id, []))
            delivered_at = datetime.now(UTC)
            updated_messages: list[SessionQueuedMessage] = []
            delivery_events: list[Event] = []
            transcript_messages: list[Message] = []
            for offset, queued_message in enumerate(selected, start=1):
                delivered_cursor = transcript_cursor + offset
                delivery_event = Event(
                    type=EventType.SESSION_MESSAGE_DELIVERED,
                    session_id=session.id,
                    interaction_id=interaction_id,
                    agent_name=session.agent_name,
                    environment_name=session.environment_name,
                    timestamp=delivered_at,
                    payload={
                        **_queued_session_message_event_payload(
                            queue_id=queued_message.queue_id,
                            delivery_mode=queued_message.delivery_mode,
                            ordering_key=queued_message.ordering_key,
                            actor=queued_message.requested_by,
                            run_epoch=session.run_epoch,
                            transcript_cursor=delivered_cursor,
                        ),
                        "accepted_run_epoch": queued_message.accepted_run_epoch,
                        "accepted_transcript_cursor": (queued_message.accepted_transcript_cursor),
                    },
                )
                updated_messages.append(
                    queued_message.model_copy(
                        update={
                            "status": SessionMessageQueueStatus.DELIVERED,
                            "delivered_run_epoch": session.run_epoch,
                            "delivered_transcript_cursor": delivered_cursor,
                            "delivered_event_id": delivery_event.id,
                            "delivered_at": delivered_at,
                        },
                        deep=True,
                    )
                )
                delivery_events.append(delivery_event)
                transcript_messages.append(
                    detach_message(Message.text(MessageRole.USER, queued_message.content))
                )

            persisted_events = [
                *([interaction_started_event] if interaction_started_event is not None else []),
                *delivery_events,
            ]
            updated_session = self._append_events_unlocked(session, persisted_events)
            self._transcripts.setdefault(session_id, []).extend(transcript_messages)
            self._transcript_interaction_ids.setdefault(session_id, [])
            self._extend_transcript_attribution_unlocked(
                session_id, [interaction_id] * len(transcript_messages)
            )
            selected_queue = self._pending_session_messages[selected_key]
            for _ in selected:
                selected_queue.popleft()
            if not selected_queue:
                del self._pending_session_messages[selected_key]
            messages_by_idempotency = self._queued_session_messages_by_idempotency[session_id]
            for updated_message in updated_messages:
                messages_by_idempotency[updated_message.idempotency_key] = updated_message
            self._sessions[session_id] = updated_session

            def has_eligible(delivery_mode: SessionMessageDeliveryMode) -> bool:
                pending = self._pending_session_messages.get((session_id, delivery_mode))
                return pending is not None and pending[0].ordering_key <= boundary

            has_more = has_eligible(SessionMessageDeliveryMode.NEXT_TURN) or (
                include_on_idle and has_eligible(SessionMessageDeliveryMode.ON_IDLE)
            )
            batch = SessionMessageDeliveryBatch(
                messages=tuple(updated_messages),
                events=tuple(persisted_events),
                delivery_id=delivery_id,
                interaction_id=interaction_id,
                eligible_through=boundary,
                has_more=has_more,
            )
            self._session_message_delivery_records[delivery_id] = _InMemoryMessageDeliveryRecord(
                session_id=session_id,
                include_on_idle=include_on_idle,
                requested_eligible_through=eligible_through,
                limit=limit,
                interaction_id=interaction_id,
                interaction_started_event=interaction_started_event,
                batch=batch,
            )
            return batch

    async def publish_checkpoint_and_events(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        session_id, copied_events = _copy_session_event_batch(session_id, events)
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        allowed_statuses = (
            None
            if expected_statuses is None
            else _validate_status_set(expected_statuses, "expected_statuses")
        )
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            if allowed_statuses is not None and session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status is not eligible for checkpoint publication: {session.status}"
                )
            if expected_run_epoch is not None and session.run_epoch != expected_run_epoch:
                raise SessionRunFenced(
                    f"Session source run epoch is stale: expected {expected_run_epoch}, "
                    f"current {session.run_epoch}."
                )
            current_cursor = len(self._transcripts.get(session_id, []))
            if (
                expected_transcript_cursor is not None
                and current_cursor != expected_transcript_cursor
            ):
                raise ValueError(
                    "Session source transcript cursor is stale: expected "
                    f"{expected_transcript_cursor}, current {current_cursor}."
                )
            current = self._checkpoints.get(session_id)
            transformed = checkpoint_transform(
                session.model_copy(deep=True),
                None if current is None else deepcopy(current),
            )
            if transformed is None:
                raise ValueError("Checkpoint transform must return a checkpoint.")
            copied_checkpoint = copy_durable_json_object(transformed, "checkpoint")
            updated = self._append_events_unlocked(session, copied_events)
            self._store_checkpoint_unlocked(session_id, copied_checkpoint)
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    async def load_session_operation(
        self,
        session_id: str,
        idempotency_key: str,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        idempotency_key = _reject_reserved_runtime_publication_key(
            idempotency_key,
            "idempotency_key",
        )
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            if checkpoint_root_guard is not None:
                checkpoint_root_guard.validate(
                    session_id,
                    checkpoint_root_field_projection(
                        self._checkpoints.get(session_id),
                        checkpoint_root_guard.key,
                    ),
                )
            record = self._session_operation_records.get(session_id, {}).get(idempotency_key)
            return None if record is None else copy_durable_json_object(record, "session_operation")

    async def _load_runtime_publication_receipt_record(
        self,
        session_id: str,
        storage_key: str,
        publication_id: str,
    ) -> dict[str, Any] | None:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            record = self._session_operation_records.get(session_id, {}).get(storage_key)
            if record is None:
                return None
            copied = deepcopy(record)
            receipt = _reconstruct_runtime_publication_receipt(
                copied,
                storage_key=storage_key,
                session_id=session_id,
                publication_id=publication_id,
            )
            self._validate_runtime_publication_material_unlocked(receipt)
            return copied

    def _validate_runtime_publication_material_unlocked(
        self,
        receipt: RuntimePublicationReceipt,
    ) -> None:
        transcript = self._transcripts.get(receipt.session_id, [])
        transcript_segment = transcript[
            receipt.transcript_start_cursor : receipt.transcript_end_cursor
        ]
        transcript_interaction_ids = self._transcript_interaction_ids.get(
            receipt.session_id,
            [],
        )[receipt.transcript_start_cursor : receipt.transcript_end_cursor]
        referenced_event_ids = _runtime_publication_referenced_event_ids(receipt.referenced_events)
        requested_event_ids = set((*receipt.appended_event_ids, *referenced_event_ids))
        events_by_id = {}
        for event_id in requested_event_ids:
            record = self._event_records_by_id.get((receipt.session_id, event_id))
            if record is not None:
                events_by_id[event_id] = record.event
        _validate_runtime_publication_durable_material(
            receipt,
            transcript_messages=transcript_segment,
            transcript_interaction_ids=transcript_interaction_ids,
            appended_events=(
                events_by_id[event_id]
                for event_id in receipt.appended_event_ids
                if event_id in events_by_id
            ),
            durable_referenced_events=(
                events_by_id[event_id]
                for event_id in referenced_event_ids
                if event_id in events_by_id
            ),
        )

    async def publish_session_operation(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        return await self._publish_session_operation(
            session_id,
            idempotency_key=idempotency_key,
            operation_transform=operation_transform,
            commit_guard=None,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )

    async def publish_session_operation_guarded(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform,
        commit_guard: Callable[[], None],
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        return await self._publish_session_operation(
            session_id,
            idempotency_key=idempotency_key,
            operation_transform=operation_transform,
            commit_guard=commit_guard,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )

    async def _publish_session_operation(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform,
        commit_guard: Callable[[], None] | None,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None,
        expected_run_epoch: int | None,
        expected_transcript_cursor: int | None,
    ) -> Session:
        session_id, copied_events = _copy_session_event_batch(session_id, events)
        idempotency_key = _reject_reserved_runtime_publication_key(
            idempotency_key,
            "idempotency_key",
        )
        if operation_transform is None:
            raise TypeError("operation_transform is required.")
        allowed_statuses = (
            None
            if expected_statuses is None
            else _validate_status_set(expected_statuses, "expected_statuses")
        )
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            if allowed_statuses is not None and session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status is not eligible for checkpoint publication: {session.status}"
                )
            if expected_run_epoch is not None and session.run_epoch != expected_run_epoch:
                raise SessionRunFenced(
                    f"Session source run epoch is stale: expected {expected_run_epoch}, "
                    f"current {session.run_epoch}."
                )
            current_cursor = len(self._transcripts.get(session_id, []))
            if (
                expected_transcript_cursor is not None
                and current_cursor != expected_transcript_cursor
            ):
                raise ValueError(
                    "Session source transcript cursor is stale: expected "
                    f"{expected_transcript_cursor}, current {current_cursor}."
                )
            current_checkpoint = self._checkpoints.get(session_id)
            current_record = self._session_operation_records.get(session_id, {}).get(
                idempotency_key
            )
            publication = operation_transform(
                session.model_copy(deep=True),
                None if current_checkpoint is None else deepcopy(current_checkpoint),
                None
                if current_record is None
                else copy_durable_json_object(current_record, "session_operation"),
            )
            if type(publication) is not SessionOperationPublication:
                raise TypeError(
                    "Session operation transform must return a SessionOperationPublication."
                )
            copied_checkpoint = copy_durable_json_object(
                publication.checkpoint,
                "checkpoint",
            )
            copied_records = copy_durable_json_object(
                publication.operation_records,
                "operation_records",
            )
            _validate_session_operation_record_keys(copied_records)
            if commit_guard is not None:
                commit_guard()
            updated = self._append_events_unlocked(session, copied_events)
            self._store_checkpoint_unlocked(session_id, copied_checkpoint)
            operation_records = self._session_operation_records.setdefault(session_id, {})
            operation_records.update(copied_records)
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    async def _load_model_completion_stage_records(
        self,
        session_id: str,
        preparation_storage_key: str,
        terminal_storage_key: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            records = self._session_operation_records[session_id]
            preparation = records.get(preparation_storage_key)
            terminal = records.get(terminal_storage_key)
            return (
                None if preparation is None else deepcopy(preparation),
                None if terminal is None else deepcopy(terminal),
            )

    async def _load_active_model_completion_stage_records(
        self,
        session_id: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            records = self._session_operation_records[session_id]
            active_record = records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY)
            if active_record is None:
                return None, None, None
            marker = _reconstruct_active_model_completion_stage_record(
                active_record,
                session_id=session_id,
            )
            _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
                session_id,
                marker.stage_id,
            )
            preparation = records.get(preparation_key)
            terminal = records.get(terminal_key)
            return (
                deepcopy(active_record),
                None if preparation is None else deepcopy(preparation),
                None if terminal is None else deepcopy(terminal),
            )

    async def _prepare_model_completion_stage_atomic(
        self,
        prepared: _PreparedModelCompletionStage,
    ) -> ModelCompletionStageResult:
        session_id = prepared.session_id
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            records = self._session_operation_records[session_id]
            stage = _reconstruct_model_completion_stage(
                records.get(prepared.preparation_storage_key),
                records.get(prepared.terminal_storage_key),
                session_id=session_id,
                stage_id=prepared.request.stage_id,
                preparation_storage_key=prepared.preparation_storage_key,
                terminal_storage_key=prepared.terminal_storage_key,
            )
            _validate_model_completion_stage_repreparation(
                records.get(prepared.abandonment_storage_key),
                prepared,
                source_status=session.status if stage is None else stage.source_status,
            )
            if stage is not None:
                _validate_model_completion_stage_preparation_replay(stage, prepared)
                active = _reconstruct_active_model_completion_stage_from_records(
                    records,
                    session_id=session_id,
                )
                _validate_model_completion_preparation_replay_state(
                    stage,
                    active=active,
                    winner_exists=records.get(prepared.winner_storage_key) is not None,
                    receipt_exists=(records.get(prepared.publication_storage_key) is not None),
                )
                return ModelCompletionStageResult(
                    stage=stage,
                    replayed=True,
                    dispatch_authorized=False,
                )

            active = _reconstruct_active_model_completion_stage_from_records(
                records,
                session_id=session_id,
            )
            _validate_model_completion_active_marker_for_preparation(
                active,
                prepared,
                source_status=session.status,
            )
            if (
                records.get(prepared.winner_storage_key) is not None
                or records.get(prepared.publication_storage_key) is not None
            ):
                raise SessionModelCompletionStageConflict(
                    "The logical model step already has durable publication state."
                )

            _assert_session_run_epoch(session_id, session)
            if session.status not in prepared.expected_statuses:
                raise SessionStatusConflict(
                    "Session status is not eligible for model-completion preparation: "
                    f"{session.status}"
                )
            if session.run_epoch != prepared.expected_run_epoch:
                raise SessionRunFenced(
                    "Session source run epoch is stale: expected "
                    f"{prepared.expected_run_epoch}, current {session.run_epoch}."
                )
            current_cursor = len(self._transcripts.get(session_id, []))
            if current_cursor != prepared.expected_transcript_cursor:
                raise ValueError(
                    "Session source transcript cursor is stale: expected "
                    f"{prepared.expected_transcript_cursor}, current {current_cursor}."
                )

            prepared_at = _next_runtime_publication_timestamp(session)
            preparation_record = _model_completion_stage_preparation_record(
                prepared,
                source_session=session,
                prepared_at=prepared_at,
            )
            stage = _reconstruct_model_completion_stage(
                preparation_record,
                None,
                session_id=session_id,
                stage_id=prepared.request.stage_id,
                preparation_storage_key=prepared.preparation_storage_key,
                terminal_storage_key=prepared.terminal_storage_key,
            )
            assert stage is not None
            active_record = _active_model_completion_stage_record(
                stage,
                activated_at=prepared_at,
            )
            records[prepared.preparation_storage_key] = preparation_record
            records[MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY] = active_record
            self._sessions[session_id] = session.model_copy(
                update={"updated_at": prepared_at, "last_activity_at": prepared_at},
                deep=True,
            )
            return ModelCompletionStageResult(
                stage=stage,
                replayed=False,
                dispatch_authorized=True,
            )

    async def _complete_model_completion_stage_atomic(
        self,
        prepared: _PreparedModelCompletionStageTerminal,
    ) -> ModelCompletionStageResult:
        session_id = prepared.session_id
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            records = self._session_operation_records[session_id]
            stage = _reconstruct_model_completion_stage(
                records.get(prepared.preparation_storage_key),
                records.get(prepared.terminal_storage_key),
                session_id=session_id,
                stage_id=prepared.stage_id,
                preparation_storage_key=prepared.preparation_storage_key,
                terminal_storage_key=prepared.terminal_storage_key,
            )
            if stage is None:
                raise KeyError(f"Model-completion stage not found: {prepared.stage_id}")
            if stage.state == "completed":
                _validate_model_completion_stage_terminal_replay(stage, prepared)
                return ModelCompletionStageResult(
                    stage=stage,
                    replayed=True,
                    dispatch_authorized=False,
                )
            _validate_model_completion_stage_publication(
                prepared.publication,
                session_id=session_id,
                stage=stage,
            )
            if not _runtime_publication_json_equal(prepared.publication.intent, stage.intent):
                raise SessionModelCompletionStageConflict(
                    "The terminal model completion intent conflicts with its preparation."
                )

            completed_at = _next_runtime_publication_timestamp(session)
            terminal_record = _model_completion_stage_terminal_record(
                prepared,
                stage=stage,
                completed_at=completed_at,
            )
            completed_stage = _reconstruct_model_completion_stage(
                records[prepared.preparation_storage_key],
                terminal_record,
                session_id=session_id,
                stage_id=prepared.stage_id,
                preparation_storage_key=prepared.preparation_storage_key,
                terminal_storage_key=prepared.terminal_storage_key,
            )
            assert completed_stage is not None
            advances_last_activity = _model_completion_terminal_advances_last_activity(
                records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                stage=stage,
                current_run_epoch=session.run_epoch,
            )
            records[prepared.terminal_storage_key] = terminal_record
            self._sessions[session_id] = session.model_copy(
                update={
                    "updated_at": completed_at,
                    "last_activity_at": (
                        completed_at if advances_last_activity else session.last_activity_at
                    ),
                },
                deep=True,
            )
            return ModelCompletionStageResult(
                stage=completed_stage,
                replayed=False,
                dispatch_authorized=False,
            )

    async def _abandon_model_completion_stage_atomic(
        self,
        prepared: _PreparedModelCompletionStageAbandonment,
    ) -> ModelCompletionStageAbandonmentResult:
        session_id = prepared.session_id
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            records = self._session_operation_records[session_id]
            stage = _reconstruct_model_completion_stage(
                records.get(prepared.preparation_storage_key),
                records.get(prepared.terminal_storage_key),
                session_id=session_id,
                stage_id=prepared.stage_id,
                preparation_storage_key=prepared.preparation_storage_key,
                terminal_storage_key=prepared.terminal_storage_key,
            )
            if stage is None:
                active_record = records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY)
                if active_record is not None:
                    active_marker = _reconstruct_active_model_completion_stage_record(
                        active_record,
                        session_id=session_id,
                    )
                    if active_marker.stage_id == prepared.stage_id:
                        raise SessionModelCompletionStageConflict(
                            "The active model-completion marker references a missing stage."
                        )
                replayed = _replay_model_completion_stage_abandonment(
                    prepared,
                    records.get(prepared.abandonment_storage_key),
                )
                logical_step_id = replayed.abandonment.logical_step_id
                if (
                    records.get(_model_completion_stage_winner_storage_key(logical_step_id))
                    is not None
                    or records.get(_runtime_publication_storage_key(logical_step_id)) is not None
                ):
                    raise SessionModelCompletionStageConflict(
                        "An abandoned model-completion stage has durable publication state."
                    )
                return replayed

            active = _reconstruct_active_model_completion_stage_from_records(
                records,
                session_id=session_id,
            )
            winner_storage_key = _model_completion_stage_winner_storage_key(stage.logical_step_id)
            publication_storage_key = _runtime_publication_storage_key(stage.logical_step_id)
            _validate_model_completion_stage_for_abandonment(
                session=session,
                stage=stage,
                active=active,
                prepared=prepared,
                abandonment_record=records.get(prepared.abandonment_storage_key),
                winner_exists=records.get(winner_storage_key) is not None,
                receipt_exists=records.get(publication_storage_key) is not None,
            )
            assert active is not None
            abandoned_at = _next_runtime_publication_timestamp(session)
            abandonment_record = _model_completion_stage_abandonment_record(
                stage,
                active=active,
                abandoned_at=abandoned_at,
            )
            abandonment = _reconstruct_model_completion_stage_abandonment(
                abandonment_record,
                session_id=session_id,
                stage_id=prepared.stage_id,
                storage_key=prepared.abandonment_storage_key,
            )

            records[prepared.abandonment_storage_key] = abandonment_record
            del records[prepared.preparation_storage_key]
            del records[MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY]
            self._sessions[session_id] = session.model_copy(
                update={"updated_at": abandoned_at, "last_activity_at": abandoned_at},
                deep=True,
            )
            return ModelCompletionStageAbandonmentResult(
                abandonment=abandonment,
                replayed=False,
            )

    async def _promote_model_completion_stage_atomic(
        self,
        *,
        session_id: str,
        stage_id: str,
        preparation_storage_key: str,
        terminal_storage_key: str,
        expected_run_epoch: int,
    ) -> RuntimePublicationResult:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            records = self._session_operation_records[session_id]
            stage = _reconstruct_model_completion_stage(
                records.get(preparation_storage_key),
                records.get(terminal_storage_key),
                session_id=session_id,
                stage_id=stage_id,
                preparation_storage_key=preparation_storage_key,
                terminal_storage_key=terminal_storage_key,
            )
            if stage is None:
                raise KeyError(f"Model-completion stage not found: {stage_id}")

            publication_storage_key = _runtime_publication_storage_key(stage.logical_step_id)
            winner_storage_key = _model_completion_stage_winner_storage_key(stage.logical_step_id)
            receipt_record = records.get(publication_storage_key)
            if receipt_record is not None:
                active = _reconstruct_active_model_completion_stage_from_records(
                    records,
                    session_id=session_id,
                )
                _validate_model_completion_promotion_replay_active_marker(active, stage)
                receipt = _reconstruct_runtime_publication_receipt(
                    deepcopy(receipt_record),
                    storage_key=publication_storage_key,
                    session_id=session_id,
                    publication_id=stage.logical_step_id,
                )
                self._validate_runtime_publication_material_unlocked(receipt)
                return _replay_promoted_model_completion_stage(
                    session=session,
                    stage=stage,
                    receipt_record=deepcopy(receipt_record),
                    winner_record=deepcopy(records.get(winner_storage_key)),
                )
            if records.get(winner_storage_key) is not None:
                raise SessionModelCompletionStageConflict(
                    "A model-completion winner exists without its runtime publication receipt."
                )
            active = _reconstruct_active_model_completion_stage_from_records(
                records,
                session_id=session_id,
            )
            _validate_model_completion_active_marker_for_promotion(active, stage)
            prepared = _prepare_model_completion_stage_promotion(
                stage,
                expected_run_epoch=expected_run_epoch,
            )
            result = self._publish_runtime_publication_unlocked(
                prepared,
                session=session,
                model_completion_stage=stage,
            )
            return result

    async def _publish_runtime_publication_atomic(
        self,
        prepared: _PreparedRuntimePublication,
    ) -> RuntimePublicationResult:
        async with self._lock:
            session = self._sessions.get(prepared.session_id)
            if session is None:
                raise KeyError(f"Session not found: {prepared.session_id}")
            return self._publish_runtime_publication_unlocked(
                prepared,
                session=session,
            )

    def _publish_runtime_publication_unlocked(
        self,
        prepared: _PreparedRuntimePublication,
        *,
        session: Session,
        model_completion_stage: ModelCompletionStage | None = None,
    ) -> RuntimePublicationResult:
        session_id = prepared.session_id
        request = prepared.request
        operation_records = self._session_operation_records[session_id]
        existing_record = operation_records.get(prepared.storage_key)
        if existing_record is not None:
            receipt = _reconstruct_runtime_publication_receipt(
                deepcopy(existing_record),
                storage_key=prepared.storage_key,
                session_id=session_id,
                publication_id=request.publication_id,
                request_digest=prepared.request_digest,
            )
            _validate_runtime_publication_replay_receipt(receipt, prepared)
            self._validate_runtime_publication_material_unlocked(receipt)
            return RuntimePublicationResult(
                session=session.model_copy(deep=True),
                receipt=receipt,
                replayed=True,
            )

        _assert_session_run_epoch(session_id, session)
        if (
            prepared.expected_statuses is not None
            and session.status not in prepared.expected_statuses
        ):
            raise SessionStatusConflict(
                f"Session status is not eligible for runtime publication: {session.status}"
            )
        if (
            prepared.expected_run_epoch is not None
            and session.run_epoch != prepared.expected_run_epoch
        ):
            raise SessionRunFenced(
                "Session source run epoch is stale: expected "
                f"{prepared.expected_run_epoch}, current {session.run_epoch}."
            )
        transcript_start_cursor = len(self._transcripts.get(session_id, []))
        if (
            prepared.expected_transcript_cursor is not None
            and transcript_start_cursor != prepared.expected_transcript_cursor
        ):
            raise ValueError(
                "Session source transcript cursor is stale: expected "
                f"{prepared.expected_transcript_cursor}, "
                f"current {transcript_start_cursor}."
            )

        appended_event_ids = {event.id for event in request.events}
        referenced_event_ids = set(
            _runtime_publication_referenced_event_ids(request.referenced_events)
        )
        if appended_event_ids & referenced_event_ids:
            raise ValueError("Appended and referenced runtime publication events overlap.")
        durable_events_by_id = {}
        for event_id in referenced_event_ids:
            record = self._event_records_by_id.get((session_id, event_id))
            if record is not None:
                durable_events_by_id[event_id] = record.event
        _validate_runtime_publication_event_references(
            request.referenced_events,
            durable_events_by_id,
            interaction_id=request.interaction_id,
        )
        current_checkpoint = self._checkpoints.get(session_id)
        _validate_tool_round_checkpoint_mutation(request, current_checkpoint)
        durable_tool_events: list[Event] = []
        tool_round_identity = _tool_round_publication_identity(request)
        if tool_round_identity is not None:
            from cayu.runtime.pending_actions import pending_action_lookup_key

            execution_identity, tool_call_ids = tool_round_identity
            event_history = self._pending_action_event_history.get(session_id, {})
            for tool_call_id in tool_call_ids:
                lookup_key = pending_action_lookup_key(tool_call_id)
                durable_tool_events.extend(
                    record.event
                    for record in event_history.get(lookup_key, ())
                    if record.event.type in _TOOL_ROUND_LIFECYCLE_EVENT_TYPES
                    and _tool_round_event_matches_scope_or_is_ambiguous(
                        record.event,
                        execution_identity,
                    )
                )
            if len(durable_tool_events) > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS:
                raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
        _validate_tool_round_publication(
            request,
            durable_events_by_id,
            durable_tool_events=durable_tool_events,
        )
        existing_event_ids = appended_event_ids & self._event_ids[session_id]
        if existing_event_ids:
            raise ValueError(
                f"Event already exists for session {session_id}: {min(existing_event_ids)}"
            )

        checkpoint = _apply_runtime_publication_checkpoint_mutation(
            request.mutation,
            current_checkpoint,
        )
        if prepared.storage_key in operation_records:
            raise SessionRuntimePublicationConflict(
                "Runtime publication receipt appeared while the session lock was held."
            )
        prepared_events = self._prepare_event_append_unlocked(
            session,
            request.events,
            events_are_detached=True,
        )
        prepared_checkpoint = (
            None
            if checkpoint is None or not request.mutation.operations
            else self._prepare_checkpoint_store_unlocked(
                session_id,
                checkpoint,
                additional_event_records=tuple(event.record for event in prepared_events.events),
            )
        )

        published_at = _next_runtime_publication_timestamp(session)
        receipt = _build_runtime_publication_receipt(
            prepared,
            source_session=session,
            checkpoint=checkpoint,
            transcript_start_cursor=transcript_start_cursor,
            published_at=published_at,
        )
        receipt_record = _runtime_publication_receipt_record(receipt)
        winner_record = (
            None
            if model_completion_stage is None
            else _model_completion_stage_winner_record(
                model_completion_stage,
                receipt=receipt,
            )
        )

        updated = self._apply_event_append_unlocked(
            session,
            prepared_events,
            activity_at=published_at,
        ).model_copy(
            update={
                "updated_at": published_at,
                "last_activity_at": published_at,
            }
        )
        self._transcripts[session_id].extend(request.transcript_messages)
        self._extend_transcript_attribution_unlocked(
            session_id, [request.interaction_id] * len(request.transcript_messages)
        )
        if prepared_checkpoint is not None:
            self._apply_checkpoint_store_unlocked(session_id, prepared_checkpoint)
        operation_records[prepared.storage_key] = receipt_record
        if model_completion_stage is not None:
            assert winner_record is not None
            winner_storage_key = _model_completion_stage_winner_storage_key(
                model_completion_stage.logical_step_id
            )
            operation_records[winner_storage_key] = winner_record
            operation_records.pop(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY)
        self._sessions[session_id] = updated
        return RuntimePublicationResult(
            session=updated.model_copy(deep=True),
            receipt=receipt,
            replayed=False,
        )

    async def load_events(self, session_id: str) -> list[Event]:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            return [event.model_copy(deep=True) for event in self._events.get(session_id, [])]

    async def load_tool_round_lifecycle_events(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
    ) -> list[Event]:
        from cayu.runtime.pending_actions import pending_action_lookup_key

        session_id = require_clean_nonblank(session_id, "session_id")
        copied_ids = _validate_tool_round_call_ids(tool_call_ids, "tool_call_ids")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            history = self._pending_action_event_history.get(session_id, {})
            records = [
                record
                for tool_call_id in copied_ids
                for record in history.get(pending_action_lookup_key(tool_call_id), ())
                if record.event.type in _TOOL_ROUND_LIFECYCLE_EVENT_TYPES
            ]
            records.sort(key=lambda record: record.sequence)
            if len(records) > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS:
                raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
            return [copy_event(record.event) for record in records]

    async def load_tool_round_lifecycle_events_for_round(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
        *,
        tool_round_identity: ToolRoundIdentity,
    ) -> list[Event]:
        from cayu.runtime.pending_actions import pending_action_lookup_key

        session_id = require_clean_nonblank(session_id, "session_id")
        copied_ids = _validate_tool_round_call_ids(tool_call_ids, "tool_call_ids")
        tool_round_identity = copy_tool_round_identity(tool_round_identity)
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            history = self._pending_action_event_history.get(session_id, {})
            records = [
                record
                for tool_call_id in copied_ids
                for record in history.get(pending_action_lookup_key(tool_call_id), ())
                if record.event.type in _TOOL_ROUND_LIFECYCLE_EVENT_TYPES
                and _tool_round_event_matches_scope_or_is_ambiguous(
                    record.event,
                    tool_round_identity,
                )
            ]
            records.sort(key=lambda record: record.sequence)
            if len(records) > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS:
                raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
            return [copy_event(record.event) for record in records]

    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        query = copy_event_query(query)
        async with self._lock:
            return self._query_events_unlocked(query, max_bytes=None)

    async def query_events_bounded(
        self,
        query: EventQuery,
        *,
        max_bytes: int,
    ) -> list[EventRecord]:
        query = copy_event_query(query)
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer.")
        async with self._lock:
            return self._query_events_unlocked(query, max_bytes=max_bytes)

    def _query_events_unlocked(
        self,
        query: EventQuery,
        *,
        max_bytes: int | None,
    ) -> list[EventRecord]:
        # `event_type` and `event_types` are mutually exclusive, so they
        # collapse into one filter set (empty = no type filter).
        if query.event_type is not None:
            event_types = frozenset((str(query.event_type),))
        else:
            event_types = frozenset(str(event_type) for event_type in query.event_types)
        excluded_event_types = frozenset(
            str(event_type) for event_type in query.exclude_event_types
        )
        candidates = self._query_candidate_records(query, event_types)
        start = (
            bisect_right(candidates, query.after_sequence, key=lambda record: record.sequence)
            if query.after_sequence is not None
            else 0
        )
        stop = (
            bisect_left(candidates, query.before_sequence, key=lambda record: record.sequence)
            if query.before_sequence is not None
            else len(candidates)
        )
        indexes = (
            range(stop - 1, start - 1, -1)
            if query.order_by == EventOrder.SEQUENCE_DESC
            else range(start, stop)
        )
        size_counter = None if max_bytes is None else JsonUtf8SizeCounter(max_bytes)
        records: list[EventRecord] = []
        for index in indexes:
            record = candidates[index]
            if not _event_record_matches(
                record,
                query,
                event_types,
                excluded_event_types,
            ):
                continue
            if not _event_record_matches_session(record, query, self._sessions):
                continue
            if size_counter is not None and not size_counter.value(record):
                assert max_bytes is not None
                raise EventQueryResultTooLarge(max_bytes)
            records.append(
                EventRecord(
                    sequence=record.sequence,
                    event=record.event,
                )
            )
            if len(records) == query.limit:
                break
        return records

    async def query_latest_interaction_events(
        self,
        session_id: str,
        *,
        before_sequence: int | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        session_id = require_clean_nonblank(session_id, "session_id")
        before_sequence, limit = _validate_interaction_page(before_sequence, limit)
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            records = self._latest_interaction_event_records_by_sequence.get(session_id, [])
            stop = (
                bisect_left(
                    records,
                    before_sequence,
                    key=lambda record: record.sequence,
                )
                if before_sequence is not None
                else len(records)
            )
            start = max(0, stop - limit)
            return [record.model_copy(deep=True) for record in reversed(records[start:stop])]

    def _query_candidate_records(
        self,
        query: EventQuery,
        event_types: frozenset[str],
    ) -> list[EventRecord]:
        """Pick the narrowest index that still covers a query's candidate rows.

        All returned lists stay sequence-ascending so downstream ordering and
        ``after_sequence`` paging behave exactly as a full scan would.
        """
        if query.event_id is not None and query.session_id is not None:
            record = self._event_records_by_id.get((query.session_id, query.event_id))
            return [] if record is None else [record]
        if (
            query.session_id is not None
            and query.workflow_name is not None
            and event_types == frozenset((str(WORKFLOW_ATTEMPT_EVENT_TYPE),))
        ):
            return self._workflow_attempt_event_records.get(
                (query.session_id, query.workflow_name),
                [],
            )
        if query.workflow_step_id is not None:
            assert query.session_id is not None
            assert query.workflow_name is not None
            if len(event_types) == 1:
                event_type = next(iter(event_types))
                if query.workflow_attempt_fenced:
                    return self._workflow_fenced_step_event_records.get(
                        (
                            query.session_id,
                            query.workflow_name,
                            query.workflow_step_id,
                            event_type,
                        ),
                        [],
                    )
                if query.workflow_attempt_id is not None:
                    return self._workflow_step_event_records.get(
                        (
                            query.session_id,
                            query.workflow_name,
                            query.workflow_attempt_id,
                            query.workflow_step_id,
                            event_type,
                        ),
                        [],
                    )
                raise RuntimeError("Workflow-step queries require an attempt scope.")
            merged: list[EventRecord] = []
            for event_type in event_types:
                if query.workflow_attempt_fenced:
                    merged.extend(
                        self._workflow_fenced_step_event_records.get(
                            (
                                query.session_id,
                                query.workflow_name,
                                query.workflow_step_id,
                                event_type,
                            ),
                            [],
                        )
                    )
                elif query.workflow_attempt_id is not None:
                    merged.extend(
                        self._workflow_step_event_records.get(
                            (
                                query.session_id,
                                query.workflow_name,
                                query.workflow_attempt_id,
                                query.workflow_step_id,
                                event_type,
                            ),
                            [],
                        )
                    )
                else:
                    raise RuntimeError("Workflow-step queries require an attempt scope.")
            merged.sort(key=lambda record: record.sequence)
            return merged
        if query.session_id is not None:
            if query.interaction_id is not None:
                return self._interaction_event_records.get(query.session_id, {}).get(
                    query.interaction_id,
                    [],
                )
            return self._session_event_records.get(query.session_id, [])
        if query.session_ids:
            merged: list[EventRecord] = []
            for session_id in query.session_ids:
                if query.interaction_id is None:
                    merged.extend(self._session_event_records.get(session_id, []))
                else:
                    merged.extend(
                        self._interaction_event_records.get(session_id, {}).get(
                            query.interaction_id,
                            [],
                        )
                    )
            merged.sort(key=lambda record: record.sequence)
            return merged
        if event_types:
            if len(event_types) == 1:
                return self._type_event_records.get(next(iter(event_types)), [])
            merged = []
            for event_type in event_types:
                merged.extend(self._type_event_records.get(event_type, []))
            merged.sort(key=lambda record: record.sequence)
            return merged
        return self._event_records

    async def summarize_events(self, session_id: str) -> EventSummary:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")

            records = self._session_event_records.get(session_id, [])
            return event_summary_from_records(session_id, records)

    async def summarize_outcome(self, session_id: str) -> SessionOutcome:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")

            records = self._session_event_records.get(session_id, [])
            return session_outcome_from_records(session, records)

    async def list_sessions(self, query: SessionQuery | None = None) -> SessionListResult:
        return await self._list_sessions(query, pending_interruption_cascade_only=False)

    async def query_session_topology(
        self,
        query: SessionTopologyQuery,
    ) -> SessionTopologyStoreResult:
        if type(query) is not SessionTopologyQuery:
            raise TypeError("Session topology queries must be SessionTopologyQuery instances.")
        query = query.model_copy(deep=True)
        async with self._lock:
            focus_session = self._sessions.get(query.focus_session_id)
            if focus_session is None:
                raise KeyError(f"Session not found: {query.focus_session_id}")

            ancestor_sessions: list[Session] = []
            seen_ancestor_ids = {focus_session.id}
            parent_session_id = focus_session.parent_session_id
            while parent_session_id is not None:
                if parent_session_id in seen_ancestor_ids:
                    raise SessionTopologyCycle(
                        f"Session topology contains a parent cycle at {parent_session_id}."
                    )
                if len(ancestor_sessions) >= query.ancestor_depth_limit:
                    raise SessionTopologyDepthExceeded(
                        f"Session topology exceeds the {query.ancestor_depth_limit}-ancestor limit."
                    )
                parent = self._sessions.get(parent_session_id)
                if parent is None:
                    raise ValueError(
                        f"Session topology references missing parent {parent_session_id}."
                    )
                ancestor_sessions.append(parent)
                seen_ancestor_ids.add(parent.id)
                parent_session_id = parent.parent_session_id
            ancestor_sessions.reverse()

            expanded_sessions: list[Session] = []
            for parent_id in query.expanded_parent_ids:
                parent = self._sessions.get(parent_id)
                if parent is None:
                    raise KeyError(f"Session not found: {parent_id}")
                expanded_sessions.append(parent)

            branch_candidates: list[list[Session]] = []
            for parent in expanded_sessions:
                child_keys = self._child_session_keys_by_parent.get(parent.id, [])
                start_index = 0
                cursor = query.child_cursors.get(parent.id)
                if cursor is not None:
                    cursor_created_at, cursor_id = decode_session_topology_cursor(
                        cursor,
                        parent_session_id=parent.id,
                    )
                    start_index = bisect_right(child_keys, (cursor_created_at, cursor_id))
                page_keys = child_keys[start_index : start_index + query.child_limit + 1]
                candidates: list[Session] = []
                for _, child_id in page_keys:
                    child = self._sessions.get(child_id)
                    if child is None or child.parent_session_id != parent.id:
                        raise RuntimeError("Inconsistent in-memory session topology index.")
                    candidates.append(child)
                branch_candidates.append(candidates)

            return build_session_topology_result(
                focus=SessionTopologyNode.from_session(focus_session),
                ancestors=(
                    SessionTopologyNode.from_session(session) for session in ancestor_sessions
                ),
                expanded_parents=(
                    SessionTopologyNode.from_session(session) for session in expanded_sessions
                ),
                branch_candidates=(
                    (SessionTopologyNode.from_session(session) for session in candidates)
                    for candidates in branch_candidates
                ),
                child_limit=query.child_limit,
            )

    async def aggregate_operational_snapshot(
        self,
        filters: SessionAggregateFilter | None = None,
    ) -> SessionOperationalSnapshot:
        filters = copy_session_aggregate_filter(filters)
        session_query = session_query_from_aggregate_filter(filters)
        async with self._lock:
            as_of = datetime.now(UTC)
            counts = {status: 0 for status in SessionStatus}
            total_count = 0
            for session in self._sessions.values():
                if _session_matches(session, session_query):
                    counts[session.status] += 1
                    total_count += 1
            return SessionOperationalSnapshot(
                as_of=as_of,
                total_count=total_count,
                counts_by_status=SessionStatusCounts.model_validate(counts),
                accuracy=EXACT_AGGREGATE.model_copy(),
            )

    async def aggregate_usage(self, query: UsageRollupQuery) -> UsageRollupStoreResult:
        query = copy_usage_rollup_query(query)
        session_query = session_query_from_aggregate_filter(query.sessions)
        async with self._lock:
            as_of = datetime.now(UTC)
            matching_session_count = 0
            active_session_count = 0
            for session in self._sessions.values():
                if not _session_matches(session, session_query):
                    continue
                matching_session_count += 1
                active_session_count += session.status in {
                    SessionStatus.PENDING,
                    SessionStatus.RUNNING,
                    SessionStatus.INTERRUPTING,
                }

            def matching_session_records() -> Iterable[
                tuple[str, SessionStatus, Iterable[EventRecord]]
            ]:
                for session_id, session in self._sessions.items():
                    if _session_matches(session, session_query):
                        yield (
                            session_id,
                            session.status,
                            self._session_event_records.get(session_id, ()),
                        )

            return _usage_rollup_from_session_records(
                session_records=matching_session_records,
                query=query,
                as_of=as_of,
                matching_session_count=matching_session_count,
                active_session_count=active_session_count,
            )

    async def list_sessions_with_pending_interruption_cascade(
        self,
        query: SessionQuery | None = None,
    ) -> SessionListResult:
        return await self._list_sessions(query, pending_interruption_cascade_only=True)

    async def query_pending_actions(
        self,
        query: PendingActionQuery | None = None,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> PendingActionListResult:
        from cayu.runtime.pending_actions import (
            PENDING_ACTION_CHECKPOINT_KEYS,
            PENDING_ACTION_SESSION_STATUSES,
            pending_action_checkpoint_tool_call_count,
            pending_action_event_projection_bytes,
            pending_action_from_records,
            pending_action_matches_query,
            pending_action_source_is_invalid,
            project_pending_action_checkpoint,
            select_pending_action_indexed_records,
        )

        if query is None:
            query = PendingActionQuery()
        elif type(query) is not PendingActionQuery:
            raise TypeError("Pending-action queries must be PendingActionQuery instances.")
        else:
            query = query.model_copy(deep=True)

        candidate_limit = min(query.limit * 4, 800) + 1
        async with self._lock:
            if query.session_id is None:
                candidate_ids = self._pending_action_session_ids
            elif query.session_id in self._pending_action_session_ids:
                candidate_ids = {query.session_id}
            else:
                candidate_ids = set()
            candidates = [
                session
                for session_id in candidate_ids
                if (session := self._sessions.get(session_id)) is not None
                and session.status in PENDING_ACTION_SESSION_STATUSES
                and (query.agent_name is None or session.agent_name == query.agent_name)
                and (
                    query.environment_name is None
                    or session.environment_name == query.environment_name
                )
            ]
            candidates = _sort_sessions(candidates, SessionOrder.UPDATED_AT_DESC)
            if query.cursor is not None:
                cursor_dt, cursor_id = decode_session_cursor(query.cursor)
                candidates = [
                    session
                    for session in candidates
                    if _session_after_cursor(
                        session,
                        SessionOrder.UPDATED_AT_DESC,
                        cursor_dt,
                        cursor_id,
                    )
                ]

            candidate_window = candidates[:candidate_limit]
            has_more_candidates = len(candidate_window) == candidate_limit
            inspected_candidates = candidate_window[: candidate_limit - 1]
            actions: list[PendingActionRecord] = []
            issues: list[PendingActionIssue] = []
            materialized_source_bytes = 0
            inspected = 0
            more_matching = False
            last_inspected_session: PendingActionSession | None = None
            for session in inspected_candidates:
                previous_last_inspected_session = last_inspected_session
                projected_session = PendingActionSession.from_session(session)
                checkpoint = self._checkpoints.get(session.id)
                if checkpoint_root_guard is not None:
                    checkpoint_root_guard.validate(
                        session.id,
                        checkpoint_root_field_projection(
                            checkpoint,
                            checkpoint_root_guard.key,
                        ),
                    )
                pending_checkpoint_source = (
                    None
                    if checkpoint is None
                    else {
                        key: checkpoint[key]
                        for key in PENDING_ACTION_CHECKPOINT_KEYS
                        if checkpoint.get(key) is not None
                    }
                )
                source_too_complex = (
                    pending_action_checkpoint_tool_call_count(pending_checkpoint_source)
                    > MAX_PENDING_ACTION_TOOL_CALLS
                )
                candidate_size = JsonUtf8SizeCounter(query.max_result_bytes)
                source_fits = (
                    not source_too_complex
                    and candidate_size.value(projected_session)
                    and candidate_size.value(pending_checkpoint_source)
                )
                projected_checkpoint = None
                records: list[EventRecord] = []
                if source_fits:
                    projected_checkpoint = project_pending_action_checkpoint(checkpoint)
                    records, ledger_too_complex = select_pending_action_indexed_records(
                        projected_checkpoint,
                        self._pending_action_event_records.get(session.id, {}),
                        self._pending_action_latest_barrier_records.get(session.id),
                    )
                    source_fits = (
                        not ledger_too_complex
                        and all(
                            (projection_bytes := pending_action_event_projection_bytes(record))
                            is None
                            or projection_bytes <= query.max_result_bytes
                            for record in records
                        )
                        and all(candidate_size.value(record) for record in records)
                    )
                else:
                    ledger_too_complex = False
                candidate_source_bytes = query.max_result_bytes - candidate_size.remaining
                if source_fits and (
                    materialized_source_bytes + candidate_source_bytes > query.max_result_bytes
                ):
                    more_matching = True
                    break

                inspected += 1
                last_inspected_session = projected_session
                if not source_fits:
                    if len(actions) + len(issues) == query.limit:
                        more_matching = True
                        inspected -= 1
                        last_inspected_session = previous_last_inspected_session
                        break
                    issues.append(
                        PendingActionIssue.ledger_too_complex(
                            projected_session,
                            max_events_per_call=MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL,
                        )
                        if ledger_too_complex
                        else PendingActionIssue.source_too_complex(
                            projected_session,
                            max_tool_calls=MAX_PENDING_ACTION_TOOL_CALLS,
                        )
                        if source_too_complex
                        else PendingActionIssue.source_too_large(
                            projected_session,
                            max_bytes=query.max_result_bytes,
                        )
                    )
                    continue

                materialized_source_bytes += candidate_source_bytes
                action = pending_action_from_records(
                    projected_session,
                    records,
                    projected_checkpoint,
                )
                if pending_action_source_is_invalid(
                    projected_session,
                    projected_checkpoint,
                    action,
                    records,
                ):
                    if len(actions) + len(issues) == query.limit:
                        more_matching = True
                        inspected -= 1
                        last_inspected_session = previous_last_inspected_session
                        break
                    issues.append(PendingActionIssue.source_invalid(projected_session))
                    continue
                if action is None:
                    continue
                if query.kind is not None and action.kind != query.kind:
                    continue
                if not pending_action_matches_query(action, query.q):
                    continue
                if len(actions) + len(issues) == query.limit:
                    more_matching = True
                    inspected -= 1
                    last_inspected_session = previous_last_inspected_session
                    break
                actions.append(action)

            has_more = more_matching or has_more_candidates
            cursor_session = last_inspected_session
            next_cursor = (
                encode_session_cursor(cursor_session, SessionOrder.UPDATED_AT_DESC)
                if has_more and cursor_session is not None
                else None
            )
            return enforce_pending_action_result_size(
                PendingActionListResult(
                    actions=actions,
                    issues=issues,
                    next_cursor=next_cursor,
                    has_more=has_more,
                    # A cursor query sees only the suffix after that cursor, so a
                    # final page's local length is not a global total. The first
                    # page can report an exact total only when it exhausted the
                    # candidate set without hitting either bound.
                    total_count=(
                        len(actions) + len(issues)
                        if query.cursor is None and not has_more
                        else None
                    ),
                    inspected_candidate_count=inspected,
                ),
                max_bytes=query.max_result_bytes,
            )

    async def _list_sessions(
        self,
        query: SessionQuery | None,
        *,
        pending_interruption_cascade_only: bool,
    ) -> SessionListResult:
        query = copy_session_query(query)
        base_query = query.model_copy(update={"debug_state": None})
        async with self._lock:
            candidates = (
                (
                    self._sessions[session_id]
                    for session_id, checkpoint in self._checkpoints.items()
                    if "pending_interruption_cascade" in checkpoint and session_id in self._sessions
                )
                if pending_interruption_cascade_only
                else self._sessions.values()
            )
            matching = [
                session
                for session in candidates
                if _session_matches(session, base_query)
                and _session_matches_debug_state(
                    session,
                    self._session_event_records.get(session.id, []),
                    query.debug_state,
                )
            ]
            total = len(matching) if query.include_total_count else None
            ordered = _sort_sessions(matching, query.order_by)
            if query.cursor is not None:
                cursor_dt, cursor_id = decode_session_cursor(query.cursor)
                ordered = [
                    session
                    for session in ordered
                    if _session_after_cursor(session, query.order_by, cursor_dt, cursor_id)
                ]
                window = ordered[: query.limit + 1]
            else:
                window = ordered[query.offset : query.offset + query.limit + 1]
            page = window[: query.limit]
            next_cursor = session_next_cursor(page, len(window) > query.limit, query.order_by)
            return SessionListResult(
                sessions=[session.model_copy(deep=True) for session in page],
                next_cursor=next_cursor,
                total_count=total,
            )

    async def append_transcript_messages(
        self,
        session_id: str,
        messages: list[Message],
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)
        copied_messages = _detach_transcript_messages(messages)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            if not copied_messages:
                return
            if interaction_id is not None:
                self._register_private_authority_alias_unlocked(
                    interaction_id,
                    field_name="interaction_id",
                    scope_session_id=session_id,
                )
            self._transcripts[session_id].extend(copied_messages)
            self._extend_transcript_attribution_unlocked(
                session_id, [interaction_id] * len(copied_messages)
            )
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": datetime.now(UTC)}
            )

    async def replace_initial_transcript_messages(
        self,
        session_id: str,
        expected_messages: list[Message],
        replacement_messages: list[Message],
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
        checkpoint_transform: CheckpointTransform | None = None,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)
        if interaction_id is None:
            raise ValueError("Initial transcript publication requires an interaction identity.")
        expected = _detach_transcript_messages(expected_messages)
        replacement = _detach_transcript_messages(replacement_messages)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            deferred = self._deferred_interaction_inputs.get(session_id)
            if deferred != (interaction_id, expected):
                raise RuntimeError("Deferred interaction input changed before finalization.")
            self._register_private_authority_alias_unlocked(
                interaction_id,
                field_name="interaction_id",
                scope_session_id=session_id,
            )
            if self._transcripts.get(session_id):
                raise RuntimeError("Initial transcript changed before finalization.")
            if len(replacement) < len(expected) or (
                expected and replacement[-len(expected) :] != expected
            ):
                raise RuntimeError("Initial transcript must preserve the admitted source suffix.")
            prefix_count = len(replacement) - len(expected)
            current_checkpoint = self._checkpoints.get(session_id)
            if checkpoint_transform is not None:
                transformed = checkpoint_transform(
                    session.model_copy(deep=True),
                    None if current_checkpoint is None else deepcopy(current_checkpoint),
                )
                if transformed is not None:
                    current_checkpoint = copy_durable_json_object(
                        transformed,
                        "checkpoint",
                    )
            checkpoint = _checkpoint_after_initial_transcript_publication(
                current_checkpoint,
                interaction_id=interaction_id,
            )
            self._transcripts[session_id] = replacement
            self._transcript_interaction_ids[session_id] = [None] * prefix_count + [
                interaction_id
            ] * len(expected)
            self._transcript_indices_by_interaction.pop(session_id, None)
            self._deferred_interaction_inputs.pop(session_id, None)
            if checkpoint is None:
                self._checkpoints.pop(session_id, None)
            else:
                self._store_checkpoint_unlocked(session_id, checkpoint)
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": datetime.now(UTC)}
            )

    async def materialize_deferred_interaction_input(
        self,
        session_id: str,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> bool:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            deferred = self._deferred_interaction_inputs.get(session_id)
            if deferred is None:
                return False
            deferred_interaction_id, messages = deferred
            if deferred_interaction_id != interaction_id:
                raise RuntimeError("Deferred interaction input belongs to another interaction.")
            self._transcripts[session_id].extend(messages)
            self._extend_transcript_attribution_unlocked(
                session_id,
                [interaction_id] * len(messages),
            )
            self._deferred_interaction_inputs.pop(session_id, None)
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": datetime.now(UTC)}
            )
            return True

    async def load_deferred_interaction_input(
        self,
        session_id: str,
    ) -> DeferredInteractionInput | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            deferred = self._deferred_interaction_inputs.get(session_id)
            if deferred is None:
                return None
            interaction_id, messages = deferred
            return DeferredInteractionInput(
                interaction_id=interaction_id,
                source_messages=messages,
            )

    async def append_transcript_messages_and_transform_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        checkpoint_transform: CheckpointTransform,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)
        copied_messages = _detach_transcript_messages(messages)
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            current = self._checkpoints.get(session_id)
            transformed = checkpoint_transform(
                session.model_copy(deep=True),
                None if current is None else deepcopy(current),
            )
            if transformed is None:
                raise ValueError("Checkpoint transform must return a checkpoint.")
            copied_checkpoint = copy_durable_json_object(transformed, "checkpoint")
            if copied_messages:
                if interaction_id is not None:
                    self._register_private_authority_alias_unlocked(
                        interaction_id,
                        field_name="interaction_id",
                        scope_session_id=session_id,
                    )
                self._transcripts[session_id].extend(copied_messages)
                self._extend_transcript_attribution_unlocked(
                    session_id, [interaction_id] * len(copied_messages)
                )
            self._store_checkpoint_unlocked(session_id, copied_checkpoint)
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": datetime.now(UTC)}
            )

    async def load_transcript(self, session_id: str) -> list[Message]:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            return [detach_message(message) for message in self._transcripts.get(session_id, [])]

    async def query_transcript(self, query: TranscriptQuery) -> TranscriptPage:
        query = copy_transcript_query(query)
        async with self._lock:
            if query.session_id not in self._sessions:
                raise KeyError(f"Session not found: {query.session_id}")

            messages = self._transcripts.get(query.session_id, [])
            interaction_ids = self._transcript_interaction_ids.get(query.session_id, [])
            if query.interaction_id is None:
                candidate_indices: Sequence[int] = range(len(messages))
            else:
                candidate_indices = self._transcript_interaction_projection_unlocked(
                    query.session_id
                ).get(query.interaction_id, ())
            if query.role is not None:
                candidate_indices = [
                    index for index in candidate_indices if messages[index].role == query.role
                ]
            total_records = len(candidate_indices)
            page_indices = candidate_indices[query.offset : query.offset + query.limit]
            # Per filter_transcript_records' contract, the include_thinking=False
            # path already isolates — detach only pass-through records.
            records = [
                TranscriptRecord(
                    index=index,
                    interaction_id=interaction_ids[index],
                    message=(
                        detach_message(messages[index])
                        if query.include_thinking
                        else messages[index]
                    ),
                )
                for index in page_indices
            ]
            return TranscriptPage(
                records=filter_transcript_records(records, include_thinking=query.include_thinking),
                total_records=total_records,
            )

    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint state must be a dictionary.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            self._store_checkpoint_unlocked(
                session_id,
                copy_durable_json_object(state, "checkpoint"),
            )
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": datetime.now(UTC)}
            )

    async def transform_checkpoint(
        self,
        session_id: str,
        checkpoint_transform: CheckpointTransform,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            current = self._checkpoints.get(session_id)
            transformed = checkpoint_transform(
                session.model_copy(deep=True),
                None if current is None else deepcopy(current),
            )
            if transformed is None:
                return
            self._store_checkpoint_unlocked(
                session_id,
                copy_durable_json_object(transformed, "checkpoint"),
            )
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": datetime.now(UTC)}
            )

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            checkpoint = self._checkpoints.get(session_id)
            if checkpoint is None:
                return None
            return deepcopy(checkpoint)

    async def load_interruption_cascade_marker(
        self,
        session_id: str,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            checkpoint = self._checkpoints.get(session_id)
            if checkpoint_root_guard is not None:
                checkpoint_root_guard.validate(
                    session_id,
                    checkpoint_root_field_projection(
                        checkpoint,
                        checkpoint_root_guard.key,
                    ),
                )
            marker = (
                _INTERRUPTION_CASCADE_MISSING
                if checkpoint is None or "pending_interruption_cascade" not in checkpoint
                else checkpoint["pending_interruption_cascade"]
            )
            return _project_interruption_cascade_marker(marker)


def event_summary_from_records(
    session_id: str,
    records: list[EventRecord],
) -> EventSummary:
    session_id = require_clean_nonblank(session_id, "session_id")
    counts_by_type: dict[str, int] = {}
    latest_record: EventRecord | None = None
    for record in records:
        if record.event.session_id != session_id:
            continue
        event_type = str(record.event.type)
        counts_by_type[event_type] = counts_by_type.get(event_type, 0) + 1
        if latest_record is None or record.sequence > latest_record.sequence:
            latest_record = record
    return EventSummary(
        session_id=session_id,
        total_events=sum(counts_by_type.values()),
        counts_by_type=counts_by_type,
        latest_event=_copy_event_record(latest_record),
    )


def session_outcome_from_records(
    session: Session,
    records: list[EventRecord],
) -> SessionOutcome:
    session = copy_session(session)
    session_records = [record for record in records if record.event.session_id == session.id]

    latest_lifecycle_sequence = 0
    for record in reversed(session_records):
        if _is_outcome_lifecycle_event(record.event):
            latest_lifecycle_sequence = record.sequence
            break

    terminal_record: EventRecord | None = None
    for record in reversed(session_records):
        if record.sequence <= latest_lifecycle_sequence:
            break
        if _is_outcome_terminal_event(record.event):
            terminal_record = record
            break

    retry_record: EventRecord | None = None
    for record in reversed(session_records):
        if record.sequence <= latest_lifecycle_sequence:
            break
        if record.event.type == EventType.MODEL_RETRY:
            retry_record = record
            break

    return session_outcome(
        session,
        terminal_event=terminal_record,
        latest_retry_event=retry_record,
    )


def session_outcome(
    session: Session,
    *,
    terminal_event: EventRecord | None,
    latest_retry_event: EventRecord | None,
) -> SessionOutcome:
    session = copy_session(session)
    terminal_event = _copy_event_record(terminal_event)
    latest_retry_event = _copy_event_record(latest_retry_event)
    if not _terminal_event_matches_status(session, terminal_event):
        terminal_event = None
    reason, details = _outcome_reason_and_details(session, terminal_event)
    return SessionOutcome(
        session_id=session.id,
        status=session.status,
        reason=reason,
        details=details,
        retry=_retry_details(latest_retry_event),
        terminal_event=terminal_event,
        latest_retry_event=latest_retry_event,
    )


def _is_outcome_terminal_event(event: Event) -> bool:
    return event.type in {
        EventType.SESSION_COMPLETED,
        EventType.SESSION_FAILED,
        EventType.SESSION_INTERRUPTED,
    }


def _is_outcome_lifecycle_event(event: Event) -> bool:
    return event.type in {
        EventType.SESSION_STARTED,
        EventType.SESSION_RESUMED,
    }


def _outcome_reason_and_details(
    session: Session,
    terminal_event: EventRecord | None,
) -> tuple[str, dict[str, Any]]:
    if session.status not in _OUTCOME_TERMINAL_STATUSES:
        return session.status.value, {}
    if terminal_event is None:
        return session.status.value, {}

    event = terminal_event.event
    if event.type != _OUTCOME_EVENT_TYPE_BY_STATUS[session.status]:
        return session.status.value, {}

    payload = event.payload
    if event.type == EventType.SESSION_COMPLETED:
        return "completed", {}
    if event.type == EventType.SESSION_FAILED:
        return "failed", _copy_payload_fields(payload, ("error", "error_type"))
    if event.type == EventType.SESSION_INTERRUPTED:
        reason = _optional_payload_string(payload, "interruption_type") or "interrupted"
        details = _copy_payload_fields(
            payload,
            (
                "interruption_type",
                "reason",
                "limit",
                "maximum",
                "actual",
                "message",
                "error",
                "error_type",
                "manual_recovery_required",
                "tool_call_id",
                "tool_name",
            ),
        )
        return reason, details
    return session.status.value, {}


def _terminal_event_matches_status(
    session: Session,
    terminal_event: EventRecord | None,
) -> bool:
    if terminal_event is None:
        return False
    expected_event_type = _OUTCOME_EVENT_TYPE_BY_STATUS.get(session.status)
    if expected_event_type is None:
        return False
    return terminal_event.event.type == expected_event_type


def _retry_details(latest_retry_event: EventRecord | None) -> dict[str, Any] | None:
    if latest_retry_event is None:
        return None
    return _copy_payload_fields(
        latest_retry_event.event.payload,
        (
            "provider",
            "model",
            "step",
            "attempt",
            "next_attempt",
            "max_attempts",
            "delay_seconds",
            "reason",
            "status_code",
        ),
    )


def _copy_event_record(record: EventRecord | None) -> EventRecord | None:
    if record is None:
        return None
    return EventRecord(sequence=record.sequence, event=record.event)


def _copy_payload_fields(payload: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for field in fields:
        if field in payload and payload[field] is not None:
            copied[field] = copy_durable_json_value(payload[field], field)
    return copied


def _optional_payload_string(payload: dict[str, Any], field: str) -> str | None:
    value = payload.get(field)
    if type(value) is str and value.strip():
        return value
    return None


_OUTCOME_TERMINAL_STATUSES = {
    SessionStatus.COMPLETED,
    SessionStatus.FAILED,
    SessionStatus.INTERRUPTED,
}

_OUTCOME_EVENT_TYPE_BY_STATUS = {
    SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
    SessionStatus.FAILED: EventType.SESSION_FAILED,
    SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
}


def copy_transcript_messages(messages: list[Message]) -> list[Message]:
    if type(messages) is not list:
        raise TypeError("Transcript messages must be a list.")
    return [detach_message(message) for message in messages]


def _detach_transcript_messages(messages: list[Message]) -> list[Message]:
    if type(messages) is not list:
        raise TypeError("Transcript messages must be a list.")
    return [detach_message(message) for message in messages]


def copy_run_request(request: RunRequest) -> RunRequest:
    if type(request) is not RunRequest:
        raise TypeError("Session creation requires a RunRequest.")
    messages = getattr(request, "messages", None)
    if type(messages) is not list:
        raise ValueError("RunRequest messages must be a list.")
    copied = RunRequest(
        agent_name=request.agent_name,
        messages=[detach_message(message) for message in messages],
        session_id=request.session_id,
        parent_session_id=request.parent_session_id,
        causal_budget_id=request.causal_budget_id,
        task_id=request.task_id,
        task_worker_id=request.task_worker_id,
        provider_name=request.provider_name,
        model=request.model,
        environment_name=request.environment_name,
        labels=copy_label_map(request.labels, "labels"),
        metadata=copy_durable_json_object(request.metadata, "metadata"),
        max_steps=request.max_steps,
        limits=copy_run_limits(request.limits),
        budget_limits=copy_request_budget_limits(request.budget_limits),
        retry_policy=copy_retry_policy(request.retry_policy) if request.retry_policy else None,
        structured_output=copy_structured_output_spec(request.structured_output),
        thinking=request.thinking,
        loop_policies=validate_loop_policies(request.loop_policies, field_name="loop_policies"),
    )
    copied._runtime_generated_authority = request._runtime_generated_authority
    return copied


def run_request_with_runtime_generated_authority(
    request: RunRequest,
    *field_names: str,
) -> RunRequest:
    """Attest exact run authority selected by a trusted runtime boundary."""

    if type(request) is not RunRequest:
        raise TypeError("Runtime authority requires a RunRequest.")
    authority = set(request._runtime_generated_authority)
    for field_name in field_names:
        if field_name not in {
            "session_id",
            "task_id",
            "parent_session_id",
            "causal_budget_id",
        }:
            raise ValueError("Unsupported runtime-generated run authority field.")
        value = getattr(request, field_name)
        if type(value) is not str or not value.strip():
            raise ValueError(
                f"RunRequest.{field_name} must be a non-empty string before attestation."
            )
        authority.add((field_name, value))
    copied = copy_run_request(request)
    copied._runtime_generated_authority = frozenset(authority)
    return copied


def run_request_authority_is_runtime_generated(
    request: RunRequest,
    *,
    field_name: str,
    value: str,
) -> bool:
    """Return positive in-process provenance for exact generated run authority."""

    return (
        type(request) is RunRequest
        and type(field_name) is str
        and type(value) is str
        and getattr(request, field_name, None) == value
        and (field_name, value) in request._runtime_generated_authority
    )


def copy_resume_request(request: ResumeRequest) -> ResumeRequest:
    if type(request) is not ResumeRequest:
        raise TypeError("Session resume requires a ResumeRequest.")
    messages = getattr(request, "messages", None)
    if type(messages) is not list:
        raise ValueError("ResumeRequest messages must be a list.")
    return ResumeRequest(
        session_id=request.session_id,
        messages=[detach_message(message) for message in messages],
        model=request.model,
        metadata=copy_durable_json_object(request.metadata, "metadata"),
        max_steps=request.max_steps,
        limits=copy_run_limits(request.limits),
        budget_limits=copy_request_budget_limits(request.budget_limits),
        retry_policy=copy_retry_policy(request.retry_policy) if request.retry_policy else None,
        structured_output=copy_structured_output_spec(request.structured_output),
        thinking=request.thinking,
        loop_policies=validate_loop_policies(request.loop_policies, field_name="loop_policies"),
    )


def copy_compact_session_request(request: CompactSessionRequest) -> CompactSessionRequest:
    if type(request) is not CompactSessionRequest:
        raise TypeError("Session compaction requires a CompactSessionRequest.")
    return CompactSessionRequest(
        session_id=request.session_id,
        idempotency_key=request.idempotency_key,
        expected_run_epoch=request.expected_run_epoch,
        expected_transcript_cursor=request.expected_transcript_cursor,
        reason=request.reason,
        instructions=request.instructions,
        limits=copy_run_limits(request.limits),
        budget_limits=copy_request_budget_limits(request.budget_limits),
        requested_by=copy_resolution_actor(request.requested_by),
    )


def copy_enqueue_session_message_request(
    request: EnqueueSessionMessageRequest,
) -> EnqueueSessionMessageRequest:
    if type(request) is not EnqueueSessionMessageRequest:
        raise TypeError("Queued input requires an EnqueueSessionMessageRequest.")
    return EnqueueSessionMessageRequest(
        session_id=request.session_id,
        idempotency_key=request.idempotency_key,
        content=request.content,
        delivery_mode=request.delivery_mode,
        requested_by=copy_resolution_actor(request.requested_by),
    )


def copy_interrupt_session_request(request: InterruptSessionRequest) -> InterruptSessionRequest:
    if type(request) is not InterruptSessionRequest:
        raise TypeError("Session interruption requires an InterruptSessionRequest.")
    return InterruptSessionRequest(
        session_id=request.session_id,
        reason=request.reason,
        metadata=copy_durable_json_object(request.metadata, "metadata"),
        requested_by=copy_resolution_actor(request.requested_by),
    )


def copy_incomplete_session_recovery_request(
    request: IncompleteSessionRecoveryRequest,
) -> IncompleteSessionRecoveryRequest:
    if type(request) is not IncompleteSessionRecoveryRequest:
        raise TypeError("Incomplete session recovery requires an IncompleteSessionRecoveryRequest.")
    return IncompleteSessionRecoveryRequest(
        session_id=request.session_id,
        inactive_before=request.inactive_before,
        reason=request.reason,
        metadata=copy_durable_json_object(request.metadata, "metadata"),
    )


def copy_incomplete_sessions_recovery_request(
    request: IncompleteSessionsRecoveryRequest,
) -> IncompleteSessionsRecoveryRequest:
    if type(request) is not IncompleteSessionsRecoveryRequest:
        raise TypeError(
            "Incomplete sessions recovery requires an IncompleteSessionsRecoveryRequest."
        )
    return IncompleteSessionsRecoveryRequest(
        statuses=set(request.statuses),
        limit=request.limit,
        inspection_limit=request.inspection_limit,
        cursor=request.cursor,
        inactive_before=request.inactive_before,
        reason=request.reason,
        metadata=copy_durable_json_object(request.metadata, "metadata"),
    )


def copy_fork_session_request(request: ForkSessionRequest) -> ForkSessionRequest:
    if type(request) is not ForkSessionRequest:
        raise TypeError("Session fork requires a ForkSessionRequest.")
    return ForkSessionRequest(
        source_session_id=request.source_session_id,
        session_id=request.session_id,
        agent_name=request.agent_name,
        model=request.model,
        environment_name=request.environment_name,
        transcript_cursor=request.transcript_cursor,
        copy_checkpoint=request.copy_checkpoint,
        metadata=copy_durable_json_object(request.metadata, "metadata"),
    )


def copy_session(session: Session) -> Session:
    if type(session) is not Session:
        raise TypeError("Session copy requires a Session.")
    return Session(
        id=session.id,
        agent_name=session.agent_name,
        provider_name=session.provider_name,
        model=session.model,
        parent_session_id=session.parent_session_id,
        causal_budget_id=session.causal_budget_id,
        runtime_name=session.runtime_name,
        runtime_version=session.runtime_version,
        environment_name=session.environment_name,
        status=session.status,
        created_at=session.created_at,
        updated_at=session.updated_at,
        last_activity_at=session.last_activity_at,
        run_epoch=session.run_epoch,
        labels=copy_label_map(session.labels, "labels"),
        metadata=copy_durable_json_object(session.metadata, "metadata"),
    )


def copy_session_identity(identity: SessionIdentity) -> SessionIdentity:
    if type(identity) is not SessionIdentity:
        raise TypeError("Session creation requires a SessionIdentity.")
    return SessionIdentity(
        provider_name=identity.provider_name,
        model=identity.model,
        runtime_name=identity.runtime_name,
        runtime_version=identity.runtime_version,
    )


def _validate_status_set(
    statuses: set[SessionStatus],
    field_name: str,
) -> set[SessionStatus]:
    if type(statuses) is not set:
        raise TypeError(f"{field_name} must be a set of SessionStatus values.")
    if not statuses:
        raise ValueError(f"{field_name} cannot be empty.")
    for status in statuses:
        if not isinstance(status, SessionStatus):
            raise ValueError(f"{field_name} must contain SessionStatus values.")
    return set(statuses)


def _audit_resolution_actor(actor: ResolutionActor | None) -> ResolutionActor | None:
    if actor is None:
        return None
    return ResolutionActor(
        subject=actor.subject,
        tenant=actor.tenant,
        source=actor.source,
    )


def _queued_session_message_event_payload(
    *,
    queue_id: str,
    delivery_mode: SessionMessageDeliveryMode,
    ordering_key: int,
    actor: ResolutionActor | None,
    run_epoch: int,
    transcript_cursor: int,
) -> dict[str, Any]:
    return {
        "queue_id": queue_id,
        "delivery_mode": str(delivery_mode),
        "ordering_key": ordering_key,
        "actor": resolution_actor_payload(actor),
        "run_epoch": run_epoch,
        "transcript_cursor": transcript_cursor,
    }


def _validate_equivalent_queued_session_message(
    existing: SessionQueuedMessage,
    request: EnqueueSessionMessageRequest,
) -> None:
    if (
        existing.content != request.content
        or existing.delivery_mode != request.delivery_mode
        or resolution_actor_payload(existing.requested_by)
        != resolution_actor_payload(request.requested_by)
    ):
        raise ValueError(
            "Session message idempotency key was already used for a different request."
        )


def _prepare_session_fork_request(
    *,
    source_session_id: str,
    fork: Session,
    source_statuses: set[SessionStatus],
    transcript_cursor: int | None,
) -> tuple[str, Session, set[SessionStatus], int | None]:
    source_session_id = require_clean_nonblank(source_session_id, "source_session_id")
    fork = copy_session(fork)
    allowed_statuses = _validate_status_set(source_statuses, "source_statuses")
    if fork.parent_session_id != source_session_id:
        raise ValueError("Fork parent_session_id must match source_session_id.")
    if transcript_cursor is not None and transcript_cursor < 0:
        raise ValueError("transcript_cursor must be greater than or equal to 0.")
    return source_session_id, fork, allowed_statuses, transcript_cursor


def _validate_session_fork_source(
    *,
    source_session: Session | None,
    source_session_id: str,
    fork: Session,
    allowed_statuses: set[SessionStatus],
    expected_source_run_epoch: int,
) -> Session:
    if source_session is None:
        raise KeyError(f"Session not found: {source_session_id}")
    if source_session.status not in allowed_statuses:
        raise ValueError(f"Source session status is not forkable: {source_session.status}")
    if source_session.run_epoch != expected_source_run_epoch:
        raise ValueError(
            "Source session changed while the fork was being prepared: "
            f"run_epoch {source_session.run_epoch} != {expected_source_run_epoch}"
        )
    if fork.status != source_session.status:
        raise ValueError(
            "Fork status must match source session status: "
            f"{fork.status} != {source_session.status}"
        )
    if fork.provider_name != source_session.provider_name:
        raise ValueError(
            "Fork provider_name must match source session provider_name: "
            f"{fork.provider_name} != {source_session.provider_name}"
        )
    return source_session


def _reject_reserved_runtime_publication_key(value: str, field_name: str) -> str:
    value = require_clean_nonblank(value, field_name)
    if value.startswith(RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX):
        raise ValueError(f"{field_name} cannot use the reserved runtime publication namespace.")
    if value.startswith(INTERACTION_TRANSITION_OPERATION_KEY_PREFIX):
        raise ValueError(f"{field_name} cannot use the reserved interaction-transition namespace.")
    if value.startswith(MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX):
        raise ValueError(f"{field_name} cannot use the reserved model-completion stage namespace.")
    return value


def _interaction_transition_storage_key(event_id: str) -> str:
    event_id = require_clean_nonblank(event_id, "event.id")
    return (
        INTERACTION_TRANSITION_OPERATION_KEY_PREFIX + sha256(event_id.encode("utf-8")).hexdigest()
    )


def _interaction_transition_receipt_record(
    *,
    session: Session,
    event: Event,
    from_statuses: set[SessionStatus],
    to_status: SessionStatus,
    only_if_no_queued_messages: bool,
    status_changed: bool,
) -> dict[str, Any]:
    ordered_from_statuses = tuple(sorted(from_statuses, key=str))
    payload = {
        "record_type": "cayu.interaction-transition",
        "schema_version": 1,
        "session": session.model_dump(mode="json"),
        "event": event.model_dump(mode="json"),
        "from_statuses": [str(status) for status in ordered_from_statuses],
        "to_status": str(to_status),
        "only_if_no_queued_messages": only_if_no_queued_messages,
        "status_changed": status_changed,
    }
    receipt = _InteractionTransitionReceipt(
        session=session,
        event=event,
        from_statuses=ordered_from_statuses,
        to_status=to_status,
        only_if_no_queued_messages=only_if_no_queued_messages,
        status_changed=status_changed,
        record_digest=_canonical_runtime_publication_digest(payload),
    )
    return receipt.model_dump(mode="json")


def _reconstruct_interaction_transition_receipt(
    record: object,
    *,
    event: Event,
    from_statuses: set[SessionStatus],
    to_status: SessionStatus,
    only_if_no_queued_messages: bool,
) -> _InteractionTransitionReceipt:
    try:
        receipt = _InteractionTransitionReceipt.model_validate(record)
        digest_payload = receipt.model_dump(mode="json", exclude={"record_digest"})
        if _canonical_runtime_publication_digest(digest_payload) != receipt.record_digest:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Stored interaction-transition receipt is invalid.") from exc
    if (
        receipt.event != event
        or set(receipt.from_statuses) != from_statuses
        or receipt.to_status is not to_status
        or receipt.only_if_no_queued_messages is not only_if_no_queued_messages
    ):
        raise ValueError(
            "Interaction transition event identity was reused with different data "
            "for the transition."
        )
    return receipt


def _validate_session_operation_record_keys(records: Mapping[str, Any]) -> None:
    for key in records:
        _reject_reserved_runtime_publication_key(key, "operation_records key")


def _runtime_publication_storage_key(publication_id: str) -> str:
    publication_id = require_clean_nonblank(publication_id, "publication_id")
    if len(publication_id) > 256:
        raise ValueError("publication_id must contain at most 256 characters.")
    return (
        RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX
        + sha256(publication_id.encode("utf-8")).hexdigest()
    )


def _model_completion_stage_storage_identity(
    session_id: str,
    stage_id: str,
) -> tuple[str, str, str, str]:
    session_id = require_clean_nonblank(session_id, "session_id")
    stage_id = require_clean_nonblank(stage_id, "stage_id")
    if len(stage_id) > 256:
        raise ValueError("stage_id must contain at most 256 characters.")
    identity_hash = sha256(stage_id.encode("utf-8")).hexdigest()
    return (
        session_id,
        stage_id,
        MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX + "prepared:" + identity_hash,
        MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX + "completed:" + identity_hash,
    )


def _model_completion_stage_winner_storage_key(logical_step_id: str) -> str:
    logical_step_id = require_clean_nonblank(logical_step_id, "logical_step_id")
    if len(logical_step_id) > 256:
        raise ValueError("logical_step_id must contain at most 256 characters.")
    return (
        MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX
        + "winner:"
        + sha256(logical_step_id.encode("utf-8")).hexdigest()
    )


def _model_completion_stage_abandonment_storage_key(stage_id: str) -> str:
    stage_id = require_clean_nonblank(stage_id, "stage_id")
    if len(stage_id) > 256:
        raise ValueError("stage_id must contain at most 256 characters.")
    return (
        MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX
        + "abandoned:"
        + sha256(stage_id.encode("utf-8")).hexdigest()
    )


def _canonical_runtime_publication_digest(value: Any) -> str:
    encoded = canonical_durable_json_bytes(value, "runtime_publication")
    return sha256(encoded).hexdigest()


def runtime_publication_checkpoint_value_digest(value: Any) -> str:
    """Return the canonical digest used to fence one checkpoint value."""

    return _canonical_runtime_publication_digest(value)


def runtime_publication_checkpoint_mutation(
    source_checkpoint: dict[str, Any] | None,
    checkpoint: dict[str, Any] | None,
) -> RuntimePublicationMutation:
    """Build a deterministic top-level CAS patch without fencing unrelated keys."""

    if checkpoint is None:
        if source_checkpoint is not None:
            raise ValueError(
                "Runtime publication checkpoint mutations cannot delete the whole checkpoint."
            )
        return RuntimePublicationMutation()
    source = (
        {}
        if source_checkpoint is None
        else copy_durable_json_object(source_checkpoint, "source_checkpoint")
    )
    target = copy_durable_json_object(checkpoint, "checkpoint")
    operations: list[RuntimePublicationCheckpointOperation] = []
    for key in sorted(source.keys() | target.keys()):
        source_present = key in source
        target_present = key in target
        if (
            source_present
            and target_present
            and _runtime_publication_json_equal(
                source[key],
                target[key],
            )
        ):
            continue
        expected_value_digest = (
            runtime_publication_checkpoint_value_digest(source[key]) if source_present else None
        )
        if target_present:
            operations.append(
                RuntimePublicationCheckpointOperation(
                    key=key,
                    expected_value_digest=expected_value_digest,
                    action="set",
                    value=target[key],
                )
            )
        else:
            operations.append(
                RuntimePublicationCheckpointOperation(
                    key=key,
                    expected_value_digest=expected_value_digest,
                    action="delete",
                )
            )
    return RuntimePublicationMutation(operations=tuple(operations))


def _apply_runtime_publication_checkpoint_mutation(
    mutation: RuntimePublicationMutation,
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if checkpoint is None and not mutation.operations:
        return None
    updated = {} if checkpoint is None else copy_durable_json_object(checkpoint, "checkpoint")
    for operation in mutation.operations:
        present = operation.key in updated
        if operation.expected_value_digest is None:
            matches = not present
        else:
            matches = present and (
                runtime_publication_checkpoint_value_digest(updated[operation.key])
                == operation.expected_value_digest
            )
        if not matches:
            raise SessionRuntimePublicationConflict(
                "A runtime publication checkpoint key changed before publication: "
                f"{operation.key!r}."
            )
        if operation.action == "set":
            updated[operation.key] = copy_durable_json_value(
                operation.value,
                "checkpoint operation value",
            )
        else:
            del updated[operation.key]
    return updated


def _runtime_publication_event_digest(event: Event) -> str:
    copied_event = copy_event(event)
    return _canonical_runtime_publication_digest(copied_event.model_dump(mode="json"))


def runtime_publication_event_reference(event: Event) -> RuntimePublicationEventReference:
    """Return an immutable ID-and-content reference for one exact event value."""

    copied_event = copy_event(event)
    return RuntimePublicationEventReference(
        event_id=copied_event.id,
        event_digest=_canonical_runtime_publication_digest(copied_event.model_dump(mode="json")),
    )


def _runtime_publication_referenced_event_ids(
    references: Iterable[RuntimePublicationEventReference],
) -> tuple[str, ...]:
    return tuple(reference.event_id for reference in references)


def _validate_runtime_publication_event_references(
    references: Iterable[RuntimePublicationEventReference],
    durable_events_by_id: Mapping[str, Event],
    *,
    interaction_id: str | None,
) -> None:
    for reference in references:
        durable_event = durable_events_by_id.get(reference.event_id)
        if durable_event is None:
            raise ValueError(
                "Runtime publication references an event that is not durable for the "
                f"session: {reference.event_id}"
            )
        if _runtime_publication_event_digest(durable_event) != reference.event_digest:
            raise ValueError(
                "Runtime publication event reference does not match durable event content "
                f"for the session: {reference.event_id}"
            )
        if durable_event.interaction_id != interaction_id:
            raise ValueError(
                "Runtime publication event reference belongs to a different interaction "
                f"for the session: {reference.event_id}"
            )


def _validate_tool_round_call_ids(
    value: Any,
    field_name: str,
) -> tuple[str, ...]:
    if type(value) not in (list, tuple):
        raise TypeError(f"{field_name} must be a list or tuple.")
    tool_call_ids = tuple(
        require_clean_nonblank(tool_call_id, f"{field_name} item") for tool_call_id in value
    )
    if not tool_call_ids or len(tool_call_ids) > RUNTIME_PUBLICATION_MAX_TOOL_CALLS:
        raise ValueError(
            f"{field_name} must contain between 1 and {RUNTIME_PUBLICATION_MAX_TOOL_CALLS} values."
        )
    if len(set(tool_call_ids)) != len(tool_call_ids):
        raise ValueError(f"{field_name} cannot repeat tool call ids.")
    return tool_call_ids


def _tool_round_event_matches_scope_or_is_ambiguous(
    event: Event,
    tool_round_identity: ToolRoundIdentity,
) -> bool:
    """Select exact or contradictory evidence while excluding a valid prior round.

    Only a complete valid identity whose round and originating model attempt
    both differ is safely stale. Missing, malformed, or partially conflicting
    identity remains visible so the publication transaction fails closed.
    """

    target = copy_tool_round_identity(tool_round_identity)
    try:
        event_identity = ToolRoundIdentity.model_validate(
            {
                "model_step_id": event.payload.get("model_step_id"),
                "model_attempt_id": event.payload.get("model_attempt_id"),
                "tool_round_id": event.payload.get("tool_round_id"),
            }
        )
    except (TypeError, ValueError):
        return True
    return not (
        event_identity.tool_round_id != target.tool_round_id
        and (
            event_identity.model_step_id,
            event_identity.model_attempt_id,
        )
        != (
            target.model_step_id,
            target.model_attempt_id,
        )
    )


def _tool_round_publication_identity(
    request: RuntimePublicationRequest,
) -> tuple[ToolRoundIdentity, tuple[str, ...]] | None:
    if request.kind != "tool-round":
        return None

    round_id = request.intent.get("round_id")
    raw_tool_call_ids = request.intent.get("tool_call_ids")
    if type(round_id) is not str or not round_id.strip():
        raise ValueError("A tool-round publication requires a nonblank intent.round_id.")
    if request.publication_id != f"tool-round:{round_id}":
        raise ValueError("A tool-round publication_id must be derived from its intent.round_id.")
    model_step_id = request.intent.get("model_step_id")
    model_attempt_id = request.intent.get("model_attempt_id")
    tool_round_id = request.intent.get("tool_round_id")
    if (
        type(model_step_id) is not str
        or type(model_attempt_id) is not str
        or type(tool_round_id) is not str
    ):
        raise ValueError("A tool-round publication requires a complete execution identity.")
    try:
        execution_identity = ToolRoundIdentity(
            model_step_id=model_step_id,
            model_attempt_id=model_attempt_id,
            tool_round_id=tool_round_id,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("A tool-round publication execution identity is malformed.") from exc
    if execution_identity.tool_round_id != round_id:
        raise ValueError(
            "A tool-round publication intent.round_id conflicts with its execution identity."
        )
    if type(raw_tool_call_ids) is not list:
        raise ValueError("A tool-round publication requires intent.tool_call_ids as a list.")
    tool_call_ids = _validate_tool_round_call_ids(
        raw_tool_call_ids,
        "intent.tool_call_ids",
    )
    return execution_identity, tool_call_ids


def _validate_structured_output_auxiliary_integer(
    value: Any,
    field_name: str,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if not 1 <= value <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError(f"{field_name} must be between 1 and {MAX_DURABLE_JSON_INTEGER}.")
    return value


def _validate_durable_structured_output_validation(
    value: Any,
    field_name: str,
) -> StructuredOutputValidation:
    if (
        type(value) is not dict
        or set(value) != {"valid", "output", "errors"}
        or type(value.get("valid")) is not bool
        or type(value.get("errors")) is not list
    ):
        raise ValueError(f"{field_name} must be a structured-output validation object.")
    try:
        validation = StructuredOutputValidation.model_validate(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is malformed.") from exc
    if validation.valid:
        if validation.errors:
            raise ValueError(f"{field_name} cannot contain errors for a valid output.")
    elif validation.output is not None or not validation.errors:
        raise ValueError(f"{field_name} must contain errors and no output for an invalid result.")
    return validation


def _validate_structured_output_tool_round_auxiliary(
    request: RuntimePublicationRequest,
    *,
    round_id: str,
) -> _StructuredOutputToolRoundAuxiliary | None:
    if "auxiliary" not in request.intent:
        if request.events:
            raise ValueError(
                "A tool-round publication cannot append events without a validated "
                "intent.auxiliary contract."
            )
        return None

    raw_auxiliary = request.intent["auxiliary"]
    if type(raw_auxiliary) is not dict:
        raise TypeError("A tool-round intent.auxiliary value must be an object.")
    if set(raw_auxiliary) != _STRUCTURED_OUTPUT_AUXILIARY_INTENT_KEYS:
        raise ValueError(
            "A structured-output tool-round intent.auxiliary value has invalid fields."
        )
    if (
        type(raw_auxiliary["schema_version"]) is not int
        or raw_auxiliary["schema_version"] != _STRUCTURED_OUTPUT_AUXILIARY_SCHEMA_VERSION
    ):
        raise ValueError(
            "A structured-output tool-round intent.auxiliary schema_version must be 1."
        )
    if raw_auxiliary["kind"] != _STRUCTURED_OUTPUT_AUXILIARY_KIND:
        raise ValueError("A structured-output tool-round intent.auxiliary kind is invalid.")
    step = _validate_structured_output_auxiliary_integer(
        raw_auxiliary["step"],
        "intent.auxiliary.step",
    )
    attempt = _validate_structured_output_auxiliary_integer(
        raw_auxiliary["attempt"],
        "intent.auxiliary.attempt",
    )
    valid = raw_auxiliary["valid"]
    retry_scheduled = raw_auxiliary["retry_scheduled"]
    if type(valid) is not bool:
        raise TypeError("intent.auxiliary.valid must be a boolean.")
    if type(retry_scheduled) is not bool:
        raise TypeError("intent.auxiliary.retry_scheduled must be a boolean.")
    if valid and retry_scheduled:
        raise ValueError("A valid structured-output result cannot schedule a retry.")

    raw_event_ids = raw_auxiliary["event_ids"]
    if type(raw_event_ids) is not list:
        raise TypeError("intent.auxiliary.event_ids must be a list.")
    event_ids = tuple(
        require_clean_nonblank(event_id, "intent.auxiliary.event_ids item")
        for event_id in raw_event_ids
    )
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("intent.auxiliary.event_ids cannot contain duplicates.")

    expected_event_types = (
        (
            EventType.STRUCTURED_OUTPUT_VALIDATING,
            EventType.STRUCTURED_OUTPUT_VALIDATED,
        )
        if valid
        else (
            (
                EventType.STRUCTURED_OUTPUT_VALIDATING,
                EventType.STRUCTURED_OUTPUT_FAILED,
                EventType.STRUCTURED_OUTPUT_RETRY,
            )
            if retry_scheduled
            else (
                EventType.STRUCTURED_OUTPUT_VALIDATING,
                EventType.STRUCTURED_OUTPUT_FAILED,
            )
        )
    )
    if event_ids != tuple(event.id for event in request.events):
        raise ValueError(
            "Structured-output auxiliary event ids and order must exactly match "
            "intent.auxiliary.event_ids."
        )
    if tuple(event.type for event in request.events) != expected_event_types:
        raise ValueError("Structured-output auxiliary events have an invalid type sequence.")

    event_session_ids = {event.session_id for event in request.events}
    if len(event_session_ids) != 1:
        raise ValueError("Structured-output auxiliary events must belong to one session.")
    event_session_id = next(iter(event_session_ids))
    first_event = request.events[0]
    expected_name = first_event.payload.get("name")
    if expected_name is not None:
        require_clean_nonblank(expected_name, "structured-output event name")
    expected_max_retries = first_event.payload.get("max_retries")
    if type(expected_max_retries) is not int or not 0 <= expected_max_retries <= 8:
        raise ValueError("Structured-output auxiliary events require a valid max_retries value.")

    outcome_output_present = False
    outcome_output: Any = None
    outcome_errors: tuple[dict[str, Any], ...] | None = None
    for event in request.events:
        if event.tool_name is not None:
            raise ValueError("Structured-output auxiliary events cannot carry a tool identity.")
        if event.payload.get("tool_round_id") != round_id:
            raise ValueError("A structured-output auxiliary event has a conflicting round id.")
        if type(event.payload.get("step")) is not int or event.payload["step"] != step:
            raise ValueError("A structured-output auxiliary event has a conflicting step.")
        if type(event.payload.get("attempt")) is not int or event.payload["attempt"] != attempt:
            raise ValueError("A structured-output auxiliary event has a conflicting attempt.")
        if event.payload.get("name") != expected_name:
            raise ValueError("Structured-output auxiliary events have conflicting names.")
        if (
            type(event.payload.get("max_retries")) is not int
            or event.payload["max_retries"] != expected_max_retries
        ):
            raise ValueError("Structured-output auxiliary events have conflicting max_retries.")
        if event.type == EventType.STRUCTURED_OUTPUT_VALIDATING:
            if event.payload.get("strategy") != StructuredOutputStrategy.TOOL.value:
                raise ValueError("A structured-output validating event must use tool strategy.")
            if any(field in event.payload for field in ("valid", "output", "errors")):
                raise ValueError("A structured-output validating event cannot claim an outcome.")
            continue
        event_valid = event.payload.get("valid")
        if type(event_valid) is not bool or event_valid != valid:
            raise ValueError(
                "A structured-output outcome event conflicts with intent.auxiliary.valid."
            )
        raw_errors = event.payload.get("errors")
        if type(raw_errors) is not list:
            raise ValueError("A structured-output outcome event requires an errors list.")
        normalized_errors: list[dict[str, Any]] = []
        for raw_error in raw_errors:
            try:
                normalized_error = StructuredOutputError.model_validate(raw_error).model_dump(
                    mode="json"
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "A structured-output outcome event contains an invalid error."
                ) from exc
            normalized_errors.append(normalized_error)
        event_errors = tuple(normalized_errors)
        event_output_present = "output" in event.payload
        event_output = (
            copy_durable_json_value(event.payload["output"], "structured_output_event.output")
            if event_output_present
            else None
        )
        if valid:
            if event_errors or not event_output_present:
                raise ValueError("A valid structured-output event requires output and no errors.")
        elif not event_errors or event_output_present:
            raise ValueError("A failed structured-output event requires errors and no output.")
        if outcome_errors is None:
            outcome_output_present = event_output_present
            outcome_output = event_output
            outcome_errors = event_errors
        elif (
            event_output_present != outcome_output_present
            or not _runtime_publication_json_equal(event_output, outcome_output)
            or event_errors != outcome_errors
        ):
            raise ValueError(
                "Structured-output outcome and retry events contain conflicting results."
            )

    assert outcome_errors is not None

    return _StructuredOutputToolRoundAuxiliary(
        step=step,
        attempt=attempt,
        valid=valid,
        retry_scheduled=retry_scheduled,
        event_ids=event_ids,
        session_id=event_session_id,
        output_present=outcome_output_present,
        output=outcome_output,
        errors=outcome_errors,
    )


def _validate_tool_round_publication(
    request: RuntimePublicationRequest,
    durable_events_by_id: Mapping[str, Event],
    *,
    durable_tool_events: Iterable[Event] | None = None,
) -> None:
    identity = _tool_round_publication_identity(request)
    if identity is None:
        return
    execution_identity, tool_call_ids = identity
    round_id = execution_identity.tool_round_id
    auxiliary = _validate_structured_output_tool_round_auxiliary(
        request,
        round_id=round_id,
    )

    result_parts: list[ToolResultPart] = []
    if len(request.transcript_messages) != 1:
        raise ValueError("A tool-round publication must append one grouped tool-result message.")
    for message in request.transcript_messages:
        if message.role != MessageRole.TOOL:
            raise ValueError("A tool-round publication can append only tool-result messages.")
        for part in message.content:
            if not isinstance(part, ToolResultPart):
                raise ValueError(
                    "A tool-round publication transcript can contain only tool results."
                )
            result_parts.append(part)
    if tuple(part.tool_call_id for part in result_parts) != tool_call_ids:
        raise ValueError(
            "Tool-round result order and identities must exactly match intent.tool_call_ids."
        )

    candidate_durable_events = (
        tuple(durable_events_by_id.values())
        if durable_tool_events is None
        else tuple(durable_tool_events)
    )
    scoped_durable_events: list[Event] = []
    for event in candidate_durable_events:
        if event.type not in _TOOL_ROUND_LIFECYCLE_EVENT_TYPES:
            raise ValueError("Tool-round durable evidence may contain only lifecycle events.")
        tool_call_id = event.payload.get("tool_call_id")
        if type(tool_call_id) is not str or tool_call_id not in tool_call_ids:
            raise ValueError("Tool-round durable evidence must identify one intended tool call.")
        try:
            durable_identity = ToolRoundIdentity.model_validate(
                {
                    "model_step_id": event.payload.get("model_step_id"),
                    "model_attempt_id": event.payload.get("model_attempt_id"),
                    "tool_round_id": event.payload.get("tool_round_id"),
                }
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Tool-round durable evidence requires a complete valid execution identity."
            ) from exc
        if durable_identity != execution_identity:
            raise ValueError("Tool-round durable evidence has a conflicting execution identity.")
        scoped_durable_events.append(event)

    scoped_durable_event_ids = tuple(event.id for event in scoped_durable_events)
    referenced_event_ids = tuple(reference.event_id for reference in request.referenced_events)
    if len(set(scoped_durable_event_ids)) != len(scoped_durable_event_ids) or set(
        scoped_durable_event_ids
    ) != set(referenced_event_ids):
        raise SessionRuntimePublicationConflict(
            "A tool-round publication must bind every durable lifecycle event for "
            "its intended tool calls."
        )
    if auxiliary is not None and any(
        event.session_id != auxiliary.session_id for event in scoped_durable_events
    ):
        raise ValueError(
            "Structured-output auxiliary and lifecycle events belong to different sessions."
        )

    material_events = tuple(scoped_durable_events)
    started_by_call: dict[str, Event] = {}
    terminal_by_call: dict[str, Event] = {}
    for event in material_events:
        if event.type != EventType.TOOL_CALL_STARTED and (
            event.type not in _TOOL_ROUND_TERMINAL_EVENT_TYPES
        ):
            raise ValueError(
                "A tool-round publication may bind only started and terminal tool events."
            )
        tool_call_id = event.payload.get("tool_call_id")
        if type(tool_call_id) is not str or tool_call_id not in tool_call_ids:
            raise ValueError("A tool-round publication event must identify one intended tool call.")
        event_round_id = event.payload.get("tool_round_id")
        if event_round_id != round_id:
            raise ValueError("A tool-round publication event has a conflicting round id.")
        destination = (
            started_by_call if event.type == EventType.TOOL_CALL_STARTED else terminal_by_call
        )
        if tool_call_id in destination:
            raise ValueError(
                "A tool-round publication cannot bind duplicate started or terminal events "
                f"for tool call {tool_call_id}."
            )
        destination[tool_call_id] = event

    if set(terminal_by_call) != set(tool_call_ids):
        raise ValueError("A tool-round publication requires exactly one terminal event per result.")
    if not set(started_by_call).issubset(terminal_by_call):
        raise ValueError(
            "Every started tool call in a publication requires a matching terminal event."
        )

    for part in result_parts:
        if (
            part.model_step_id != execution_identity.model_step_id
            or part.model_attempt_id != execution_identity.model_attempt_id
            or part.tool_round_id != execution_identity.tool_round_id
        ):
            raise ValueError("A tool-round transcript result has a conflicting execution identity.")
        terminal = terminal_by_call[part.tool_call_id]
        if terminal.tool_name != part.tool_name:
            raise ValueError(
                "A tool-round terminal event tool name conflicts with its transcript result."
            )
        terminal_idempotency_key = terminal.payload.get("idempotency_key")
        if type(terminal_idempotency_key) is not str or not terminal_idempotency_key:
            raise ValueError("A tool-round terminal event requires a stable idempotency key.")
        started = started_by_call.get(part.tool_call_id)
        if started is not None and (
            started.tool_name != part.tool_name
            or started.payload.get("idempotency_key") != terminal_idempotency_key
        ):
            raise ValueError("A tool-round started event conflicts with its terminal event.")
        expected_result = {
            "content": part.content,
            "structured": part.structured,
            "artifacts": part.artifacts,
            "is_error": part.is_error,
        }
        if not _runtime_publication_json_equal(
            terminal.payload.get("result"),
            expected_result,
        ):
            raise ValueError(
                "A tool-round terminal event result conflicts with its transcript result."
            )
        if (terminal.type == EventType.TOOL_CALL_COMPLETED and part.is_error) or (
            terminal.type != EventType.TOOL_CALL_COMPLETED and not part.is_error
        ):
            raise ValueError(
                "A tool-round terminal event outcome conflicts with its transcript result."
            )

    pending_round_deletions = [
        operation
        for operation in request.mutation.operations
        if operation.key == _PENDING_TOOL_ROUND_CHECKPOINT_KEY and operation.action == "delete"
    ]
    if len(pending_round_deletions) != 1:
        raise ValueError(
            "A tool-round publication must delete exactly one fenced pending round marker."
        )


def _validate_structured_output_tool_round_marker(
    request: RuntimePublicationRequest,
    *,
    marker: dict[str, Any],
    raw_calls: list[dict[str, Any]],
) -> None:
    round_id = request.intent["round_id"]
    auxiliary = _validate_structured_output_tool_round_auxiliary(
        request,
        round_id=round_id,
    )
    structured_calls = [
        call for call in raw_calls if call.get("tool_name") == STRUCTURED_OUTPUT_TOOL_NAME
    ]
    if auxiliary is None:
        if structured_calls:
            raise SessionRuntimePublicationConflict(
                "A pending structured-output tool round requires validated "
                "intent.auxiliary evidence."
            )
        return
    if not structured_calls:
        raise SessionRuntimePublicationConflict(
            "Structured-output auxiliary evidence requires the reserved tool "
            "in the durable pending marker."
        )
    if any(type(call.get("arguments")) is not dict for call in structured_calls):
        raise SessionRuntimePublicationConflict(
            "The reserved structured-output call arguments are malformed."
        )

    raw_spec = marker.get("structured_output")
    if type(raw_spec) is not dict:
        raise SessionRuntimePublicationConflict(
            "Structured-output auxiliary evidence requires durable structured-output config."
        )
    try:
        spec = StructuredOutputSpec.model_validate(raw_spec)
    except (TypeError, ValueError) as exc:
        raise SessionRuntimePublicationConflict(
            "The durable structured-output config is malformed."
        ) from exc
    if spec.strategy != StructuredOutputStrategy.TOOL:
        raise SessionRuntimePublicationConflict(
            "Structured-output tool-round auxiliary evidence requires tool strategy."
        )
    if auxiliary.attempt > spec.max_retries + 1:
        raise SessionRuntimePublicationConflict(
            "Structured-output auxiliary attempt exceeds the durable retry policy."
        )
    if auxiliary.retry_scheduled and auxiliary.attempt > spec.max_retries:
        raise SessionRuntimePublicationConflict(
            "Structured-output auxiliary retry exceeds the durable retry policy."
        )
    raw_validation = marker.get("structured_output_validation")
    try:
        durable_validation = _validate_durable_structured_output_validation(
            raw_validation,
            "pending_tool_round.structured_output_validation",
        )
    except (TypeError, ValueError) as exc:
        raise SessionRuntimePublicationConflict(
            "The durable structured-output validation is malformed."
        ) from exc
    durable_errors = tuple(error.model_dump(mode="json") for error in durable_validation.errors)
    if (
        auxiliary.valid != durable_validation.valid
        or (
            durable_validation.valid
            and (
                durable_errors
                or not auxiliary.output_present
                or not _runtime_publication_json_equal(
                    auxiliary.output,
                    durable_validation.output,
                )
            )
        )
        or (
            not durable_validation.valid
            and (
                durable_validation.output is not None
                or auxiliary.output_present
                or auxiliary.errors != durable_errors
            )
        )
    ):
        raise SessionRuntimePublicationConflict(
            "Structured-output auxiliary outcome conflicts with its authoritative validation."
        )
    marker_model_step = marker.get("model_step")
    if marker_model_step is not None and (
        type(marker_model_step) is not int or marker_model_step != auxiliary.step
    ):
        raise SessionRuntimePublicationConflict(
            "Structured-output auxiliary step conflicts with the durable pending marker."
        )
    marker_structured_output_attempt = marker.get("structured_output_attempt")
    if marker_structured_output_attempt is not None and (
        type(marker_structured_output_attempt) is not int
        or marker_structured_output_attempt != auxiliary.attempt
    ):
        raise SessionRuntimePublicationConflict(
            "Structured-output auxiliary attempt conflicts with the durable pending marker."
        )

    marker_agent_name = marker.get("agent_name")
    if type(marker_agent_name) is not str:
        raise SessionRuntimePublicationConflict(
            "The durable pending tool round agent identity is malformed."
        )
    try:
        marker_agent_name = require_clean_nonblank(
            marker_agent_name,
            "pending tool round agent_name",
        )
    except (TypeError, ValueError) as exc:
        raise SessionRuntimePublicationConflict(
            "The durable pending tool round agent identity is malformed."
        ) from exc
    marker_environment_name = marker.get("environment_name")
    if marker_environment_name is not None:
        try:
            marker_environment_name = require_clean_nonblank(
                marker_environment_name,
                "pending tool round environment_name",
            )
        except (TypeError, ValueError) as exc:
            raise SessionRuntimePublicationConflict(
                "The durable pending tool round environment identity is malformed."
            ) from exc

    for event in request.events:
        if (
            event.agent_name != marker_agent_name
            or event.environment_name != marker_environment_name
        ):
            raise SessionRuntimePublicationConflict(
                "A structured-output auxiliary event conflicts with the durable "
                "pending-round identity."
            )
        if event.payload.get("name") != spec.name:
            raise SessionRuntimePublicationConflict(
                "A structured-output auxiliary event conflicts with the durable output name."
            )
        if (
            type(event.payload.get("max_retries")) is not int
            or event.payload["max_retries"] != spec.max_retries
        ):
            raise SessionRuntimePublicationConflict(
                "A structured-output auxiliary event conflicts with the durable retry policy."
            )

    result_parts = [
        part
        for message in request.transcript_messages
        for part in message.content
        if isinstance(part, ToolResultPart)
    ]
    result_parts_by_id = {part.tool_call_id: part for part in result_parts}
    for structured_call in structured_calls:
        tool_call_id = structured_call.get("tool_call_id")
        if type(tool_call_id) is not str:
            raise SessionRuntimePublicationConflict(
                "The reserved structured-output call identity is malformed."
            )
        result_part = result_parts_by_id.get(tool_call_id)
        if result_part is None or result_part.tool_name != STRUCTURED_OUTPUT_TOOL_NAME:
            raise SessionRuntimePublicationConflict(
                "The reserved structured-output call is not bound to its transcript result."
            )
    if auxiliary.valid:
        if len(structured_calls) != 1 or len(raw_calls) != 1:
            raise SessionRuntimePublicationConflict(
                "A valid structured-output auxiliary outcome requires exactly one "
                "reserved tool call."
            )
        structured_call_id = structured_calls[0]["tool_call_id"]
        if result_parts_by_id[structured_call_id].is_error:
            raise SessionRuntimePublicationConflict(
                "A valid structured-output auxiliary outcome conflicts with its tool result."
            )
        structured_result = result_parts_by_id[structured_call_id].structured
        if (
            type(structured_result) is not dict
            or set(structured_result) != {"output"}
            or not auxiliary.output_present
            or not _runtime_publication_json_equal(
                structured_result["output"],
                auxiliary.output,
            )
        ):
            raise SessionRuntimePublicationConflict(
                "A valid structured-output event conflicts with its grouped tool result."
            )
    elif any(not part.is_error for part in result_parts):
        raise SessionRuntimePublicationConflict(
            "A failed structured-output auxiliary outcome conflicts with its tool results."
        )
    else:
        expected_errors = list(auxiliary.errors)
        for result_part in result_parts:
            structured_result = result_part.structured
            if (
                type(structured_result) is not dict
                or set(structured_result) != {"structured_output_errors"}
                or not _runtime_publication_json_equal(
                    structured_result["structured_output_errors"],
                    expected_errors,
                )
            ):
                raise SessionRuntimePublicationConflict(
                    "A failed structured-output event conflicts with its grouped tool results."
                )


def _validate_tool_round_checkpoint_mutation(
    request: RuntimePublicationRequest,
    checkpoint: dict[str, Any] | None,
) -> None:
    publication_identity = _tool_round_publication_identity(request)
    if publication_identity is None:
        return
    execution_identity, intended_call_ids = publication_identity
    marker = None if checkpoint is None else checkpoint.get(_PENDING_TOOL_ROUND_CHECKPOINT_KEY)
    if type(marker) is not dict:
        raise SessionRuntimePublicationConflict(
            "A tool-round publication requires its durable pending round marker."
        )
    raw_calls = marker.get("tool_calls")
    if type(raw_calls) is not list or any(type(call) is not dict for call in raw_calls):
        raise SessionRuntimePublicationConflict(
            "The durable pending tool round call list is malformed."
        )
    marker_call_ids = tuple(call.get("tool_call_id") for call in raw_calls)
    if (
        marker.get("tool_round_id") != execution_identity.tool_round_id
        or marker_call_ids != intended_call_ids
    ):
        raise SessionRuntimePublicationConflict(
            "The durable pending tool round conflicts with the publication intent."
        )
    for field_name in ("model_step_id", "model_attempt_id", "tool_round_id"):
        if marker.get(field_name) != getattr(execution_identity, field_name):
            raise SessionRuntimePublicationConflict(
                "The durable pending tool-round execution identity conflicts "
                "with the publication intent."
            )
    _validate_structured_output_tool_round_marker(
        request,
        marker=marker,
        raw_calls=raw_calls,
    )
    deletions = [
        operation
        for operation in request.mutation.operations
        if operation.key == _PENDING_TOOL_ROUND_CHECKPOINT_KEY and operation.action == "delete"
    ]
    if len(deletions) != 1:
        raise SessionRuntimePublicationConflict(
            "A tool-round publication must delete exactly one fenced pending round marker."
        )
    deletion = deletions[0]
    if runtime_publication_checkpoint_value_digest(marker) != deletion.expected_value_digest:
        raise SessionRuntimePublicationConflict(
            "The tool-round publication does not fence its exact pending marker."
        )


def _runtime_publication_json_equal(left: Any, right: Any) -> bool:
    """Compare validated JSON material under the durable canonicalization contract."""

    return _canonical_runtime_publication_digest(left) == _canonical_runtime_publication_digest(
        right
    )


def _validate_required_runtime_fence(value: int, field_name: str) -> int:
    validated = _validate_optional_runtime_publication_fence(value, field_name)
    if validated is None:
        raise TypeError(f"{field_name} must be an integer.")
    return validated


def _validate_optional_runtime_publication_fence(
    value: int | None,
    field_name: str,
) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer or None.")
    if value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0.")
    if value > MAX_DURABLE_JSON_INTEGER:
        raise ValueError(f"{field_name} must be at most {MAX_DURABLE_JSON_INTEGER}.")
    return value


def _prepare_model_completion_stage(
    session_id: str,
    request: ModelCompletionStageRequest,
    *,
    expected_statuses: set[SessionStatus],
    expected_run_epoch: int,
    expected_transcript_cursor: int,
) -> _PreparedModelCompletionStage:
    session_id = require_clean_nonblank(session_id, "session_id")
    if type(request) is not ModelCompletionStageRequest:
        raise TypeError("Model completion staging requires a ModelCompletionStageRequest.")
    try:
        copied_request = ModelCompletionStageRequest(
            stage_id=request.stage_id,
            logical_step_id=request.logical_step_id,
            dispatch_ordinal=request.dispatch_ordinal,
            purpose=request.purpose,
            intent=request.intent,
            reservation_ids=request.reservation_ids,
        )
    except AttributeError as exc:
        raise ValueError("Model completion stage request is malformed.") from exc
    allowed_statuses = frozenset(_validate_status_set(expected_statuses, "expected_statuses"))
    expected_run_epoch = _validate_required_runtime_fence(
        expected_run_epoch,
        "expected_run_epoch",
    )
    expected_transcript_cursor = _validate_required_runtime_fence(
        expected_transcript_cursor,
        "expected_transcript_cursor",
    )
    _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
        session_id,
        copied_request.stage_id,
    )
    winner_key = _model_completion_stage_winner_storage_key(copied_request.logical_step_id)
    request_digest = _canonical_runtime_publication_digest(
        {
            "session_id": session_id,
            "stage_id": copied_request.stage_id,
            "logical_step_id": copied_request.logical_step_id,
            "dispatch_ordinal": copied_request.dispatch_ordinal,
            "purpose": copied_request.purpose,
            "intent": copied_request.intent,
            "reservation_ids": list(copied_request.reservation_ids),
            "expected_statuses": sorted(str(status) for status in allowed_statuses),
            "expected_run_epoch": expected_run_epoch,
            "expected_transcript_cursor": expected_transcript_cursor,
        }
    )
    return _PreparedModelCompletionStage(
        session_id=session_id,
        preparation_storage_key=preparation_key,
        terminal_storage_key=terminal_key,
        abandonment_storage_key=_model_completion_stage_abandonment_storage_key(
            copied_request.stage_id
        ),
        winner_storage_key=winner_key,
        publication_storage_key=_runtime_publication_storage_key(copied_request.logical_step_id),
        request=copied_request,
        request_digest=request_digest,
        expected_statuses=allowed_statuses,
        expected_run_epoch=expected_run_epoch,
        expected_transcript_cursor=expected_transcript_cursor,
    )


def _prepare_model_completion_stage_abandonment(
    session_id: str,
    *,
    stage_id: str,
    preparation_digest: str,
    expected_run_epoch: int,
) -> _PreparedModelCompletionStageAbandonment:
    session_id, stage_id, preparation_key, terminal_key = _model_completion_stage_storage_identity(
        session_id, stage_id
    )
    try:
        _require_raw_sha256_digest(preparation_digest)
    except ValueError as exc:
        raise ValueError("preparation_digest must be a lowercase SHA-256 digest.") from exc
    expected_run_epoch = _validate_required_runtime_fence(
        expected_run_epoch,
        "expected_run_epoch",
    )
    return _PreparedModelCompletionStageAbandonment(
        session_id=session_id,
        stage_id=stage_id,
        preparation_storage_key=preparation_key,
        terminal_storage_key=terminal_key,
        abandonment_storage_key=_model_completion_stage_abandonment_storage_key(stage_id),
        preparation_digest=preparation_digest,
        expected_run_epoch=expected_run_epoch,
    )


def _prepare_model_completion_stage_terminal(
    session_id: str,
    *,
    stage_id: str,
    publication: RuntimePublicationRequest,
) -> _PreparedModelCompletionStageTerminal:
    session_id, stage_id, preparation_key, terminal_key = _model_completion_stage_storage_identity(
        session_id, stage_id
    )
    if type(publication) is not RuntimePublicationRequest:
        raise TypeError("publication must be a RuntimePublicationRequest.")
    try:
        copied_publication = RuntimePublicationRequest(
            publication_id=publication.publication_id,
            kind=publication.kind,
            interaction_id=publication.interaction_id,
            intent=publication.intent,
            mutation=publication.mutation,
            transcript_messages=publication.transcript_messages,
            events=publication.events,
            referenced_events=publication.referenced_events,
        )
    except AttributeError as exc:
        raise ValueError("Model completion publication is malformed.") from exc
    publication_payload = copied_publication.model_dump(mode="json")
    return _PreparedModelCompletionStageTerminal(
        session_id=session_id,
        stage_id=stage_id,
        preparation_storage_key=preparation_key,
        terminal_storage_key=terminal_key,
        publication=copied_publication,
        publication_material_digest=_canonical_runtime_publication_digest(publication_payload),
    )


def _validate_model_completion_stage_publication(
    publication: RuntimePublicationRequest,
    *,
    session_id: str,
    stage: ModelCompletionStage | _ModelCompletionStagePreparationRecord,
) -> None:
    if publication.publication_id != stage.logical_step_id:
        raise ValueError("Model completion publication_id must equal logical_step_id.")
    if not _runtime_publication_json_equal(publication.intent, stage.intent):
        raise SessionModelCompletionStageConflict(
            "The terminal model completion intent conflicts with its preparation."
        )
    if stage.purpose == "assistant-turn":
        if publication.kind != "model-step":
            raise ValueError("Assistant-turn publication kind must be 'model-step'.")
        if len(publication.transcript_messages) > 1 or any(
            message.role != MessageRole.ASSISTANT for message in publication.transcript_messages
        ):
            raise ValueError(
                "An assistant-turn publication may contain at most one assistant message."
            )
    else:
        if publication.kind != "context-compaction":
            raise ValueError("Context-compaction publication kind must be 'context-compaction'.")
        if publication.transcript_messages:
            raise ValueError("A context-compaction publication cannot append transcript messages.")
    completed_events = [
        event for event in publication.events if event.type == EventType.MODEL_COMPLETED
    ]
    if len(completed_events) != 1:
        raise ValueError(
            "A model completion publication must append exactly one model.completed event."
        )
    if stage.purpose == "assistant-turn" and (
        len(publication.events) != 1 or publication.referenced_events
    ):
        raise ValueError(
            "An assistant-turn publication must bind exactly its model.completed event."
        )
    completed_event = completed_events[0]
    if stage.purpose == "assistant-turn" and (
        stage.reservation_ids or "budget_settlements" in completed_event.payload
    ):
        model_completion_budget_settlements(
            completed_event,
            reservation_ids=stage.reservation_ids,
        )
    classification = completed_event.payload.get("step_classification")
    if (
        stage.purpose == "assistant-turn"
        and not publication.transcript_messages
        and (
            type(classification) is not dict
            or classification.get("type") not in _MESSAGELESS_MODEL_COMPLETION_CLASSIFICATIONS
        )
    ):
        raise ValueError(
            "A model completion publication without an assistant message requires "
            "a non-turn or continue model.completed step classification."
        )
    if (
        stage.purpose == "assistant-turn"
        and completed_event.payload.get("purpose") == "context_compaction"
    ):
        raise ValueError("An assistant-turn publication cannot carry context-compaction evidence.")
    if (
        stage.purpose == "context-compaction"
        and completed_event.payload.get("purpose") != "context_compaction"
    ):
        raise ValueError(
            "A context-compaction publication requires matching model completion evidence."
        )
    for event in publication.events:
        if event.session_id != session_id:
            raise ValueError("Model completion event session_id does not match target session.")
    appended_event_ids = {event.id for event in publication.events}
    overlap = appended_event_ids & set(
        _runtime_publication_referenced_event_ids(publication.referenced_events)
    )
    if overlap:
        raise ValueError(
            f"Model completion cannot both append and reference event id {min(overlap)}."
        )
    if stage.purpose == "assistant-turn":
        _validate_assistant_model_completion_publication(
            publication,
            stage=stage,
            completed_event=completed_event,
            classification=classification,
        )


def _validate_assistant_model_completion_publication(
    publication: RuntimePublicationRequest,
    *,
    stage: ModelCompletionStage | _ModelCompletionStagePreparationRecord,
    completed_event: Event,
    classification: Any,
) -> None:
    if type(classification) is not dict:
        raise ValueError("An assistant-turn publication requires a model step classification.")
    raw_model_step_id = stage.intent.get("model_step_id")
    raw_model_attempt_id = stage.intent.get("model_attempt_id")
    model_attempt_identity = None
    if raw_model_step_id is not None or raw_model_attempt_id is not None:
        if type(raw_model_step_id) is not str or type(raw_model_attempt_id) is not str:
            raise ValueError(
                "An assistant-turn completion stage has a malformed model-attempt identity."
            )
        try:
            model_attempt_identity = ModelAttemptIdentity(
                model_step_id=raw_model_step_id,
                model_attempt_id=raw_model_attempt_id,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "An assistant-turn completion stage has a malformed model-attempt identity."
            ) from exc
        if (
            model_attempt_identity.model_step_id != stage.logical_step_id
            or completed_event.payload.get("model_step_id") != model_attempt_identity.model_step_id
            or completed_event.payload.get("model_attempt_id")
            != model_attempt_identity.model_attempt_id
        ):
            raise ValueError(
                "The model.completed identity conflicts with its staged model attempt."
            )
    assistant_message = (
        publication.transcript_messages[0] if publication.transcript_messages else None
    )
    tool_calls = (
        tuple(part for part in assistant_message.content if isinstance(part, ToolCallPart))
        if assistant_message is not None
        else ()
    )
    classification_type = classification.get("type")
    if tool_calls and classification_type != "continue":
        raise ValueError(
            "An assistant model completion with tool calls must be classified as continue."
        )

    tool_round_identity = None
    if tool_calls:
        if model_attempt_identity is None:
            raise ValueError("Assistant tool calls require a staged model-attempt identity.")
        raw_identities = {
            (call.model_step_id, call.model_attempt_id, call.tool_round_id) for call in tool_calls
        }
        if len(raw_identities) != 1:
            raise ValueError("Assistant tool calls must share one complete tool-round identity.")
        model_step_id, model_attempt_id, tool_round_id = next(iter(raw_identities))
        if model_step_id is None or model_attempt_id is None or tool_round_id is None:
            raise ValueError("Assistant tool calls must carry a complete tool-round identity.")
        try:
            tool_round_identity = ToolRoundIdentity(
                model_step_id=model_step_id,
                model_attempt_id=model_attempt_id,
                tool_round_id=tool_round_id,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Assistant tool calls must carry a valid tool-round identity."
            ) from exc
        if (
            tool_round_identity.model_step_id != model_attempt_identity.model_step_id
            or tool_round_identity.model_attempt_id != model_attempt_identity.model_attempt_id
        ):
            raise ValueError(
                "Assistant tool-round identity conflicts with its staged model completion."
            )
    expected_tool_round_id = (
        None if tool_round_identity is None else tool_round_identity.tool_round_id
    )
    expected_operation_keys = {LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY}
    if tool_calls:
        expected_operation_keys.add(_PENDING_TOOL_ROUND_CHECKPOINT_KEY)
    operations_by_key = {operation.key: operation for operation in publication.mutation.operations}
    actual_operation_keys = frozenset(operations_by_key)
    if actual_operation_keys not in {
        frozenset(expected_operation_keys),
        frozenset(expected_operation_keys | {CHECKPOINT_SCHEMA_VERSION_KEY}),
    }:
        raise ValueError(
            "An assistant model completion must atomically publish exactly its model-step "
            "pointer, optional pending tool-round marker, and optional root checkpoint "
            "schema stamp."
        )
    schema_operation = operations_by_key.get(CHECKPOINT_SCHEMA_VERSION_KEY)
    if schema_operation is not None and (
        schema_operation.action != "set"
        or schema_operation.expected_value_digest
        != runtime_publication_checkpoint_value_digest(
            CURRENT_CHECKPOINT_SCHEMA_VERSION,
        )
        or schema_operation.value != CURRENT_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError(
            "A model completion root checkpoint schema stamp must fence the current version."
        )

    pointer_operation = operations_by_key[LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY]
    if pointer_operation.action != "set":
        raise ValueError("A model-step publication pointer must use a set operation.")
    try:
        pointer = ModelStepPublicationCheckpoint.model_validate(pointer_operation.value)
    except (TypeError, ValueError) as exc:
        raise ValueError("The model-step publication pointer is malformed.") from exc
    expected_transcript_end = stage.source_transcript_cursor + int(assistant_message is not None)
    expected_pointer = ModelStepPublicationCheckpoint(
        logical_step_id=stage.logical_step_id,
        stage_id=stage.stage_id,
        source_transcript_cursor=stage.source_transcript_cursor,
        transcript_end_cursor=expected_transcript_end,
        completion_event_id=completed_event.id,
        classification=classification,
        assistant_message_published=assistant_message is not None,
        tool_round_id=expected_tool_round_id,
    )
    if pointer != expected_pointer:
        raise ValueError("The model-step publication pointer conflicts with its staged completion.")
    transcript_cursor = completed_event.payload.get("transcript_cursor")
    if type(transcript_cursor) is not int or transcript_cursor != expected_transcript_end:
        raise ValueError(
            "The model.completed transcript cursor conflicts with its staged completion."
        )

    if not tool_calls:
        return
    pending_operation = operations_by_key[_PENDING_TOOL_ROUND_CHECKPOINT_KEY]
    if pending_operation.action != "set" or type(pending_operation.value) is not dict:
        raise ValueError("A model-step pending tool-round marker must use an object set operation.")
    marker = pending_operation.value
    expected_marker_keys = {
        "tool_round_id",
        "model_step_id",
        "model_attempt_id",
        "agent_name",
        "environment_name",
        "task_id",
        "tool_calls",
        "policy_state",
        "policy_context_version",
        "request_metadata",
        "deferred_messages",
        "structured_output",
        "thinking",
        "max_steps",
        "limits",
        "budget_limits",
        "retry_policy",
        "source_model_step_id",
        "source_transcript_cursor",
        "model_step",
        "structured_output_attempt",
        "structured_output_validation",
    }
    if set(marker) != expected_marker_keys:
        raise ValueError("The model-step pending tool-round marker has invalid fields.")
    if tool_round_identity is None:
        raise ValueError("The model-step pending tool round lost its identity.")
    if (
        marker.get("tool_round_id") != tool_round_identity.tool_round_id
        or marker.get("model_step_id") != tool_round_identity.model_step_id
        or marker.get("model_attempt_id") != tool_round_identity.model_attempt_id
    ):
        raise ValueError("The pending tool-round identity conflicts with its model step.")
    if marker.get("source_model_step_id") != tool_round_identity.model_step_id:
        raise ValueError("The pending tool round has a conflicting source model-step identity.")
    if marker.get("source_transcript_cursor") != stage.source_transcript_cursor:
        raise ValueError("The pending tool round has a conflicting source transcript cursor.")
    try:
        require_clean_nonblank(marker.get("agent_name"), "pending tool-round agent_name")
        for field_name in ("environment_name", "task_id"):
            value = marker.get(field_name)
            if value is not None:
                require_clean_nonblank(value, f"pending tool-round {field_name}")
    except (TypeError, ValueError) as exc:
        raise ValueError("The pending tool-round runtime identity is malformed.") from exc
    marker_model_step = marker.get("model_step")
    if type(marker_model_step) is not int or marker_model_step < 1:
        raise ValueError("The pending tool round has an invalid model step.")
    completed_model_step = completed_event.payload.get("step")
    if completed_model_step is not None and (
        type(completed_model_step) is not int or completed_model_step != marker_model_step
    ):
        raise ValueError("The pending tool round has a conflicting model step.")

    raw_calls = marker.get("tool_calls")
    if type(raw_calls) is not list or len(raw_calls) != len(tool_calls):
        raise ValueError("The pending tool-round calls conflict with the assistant message.")
    try:
        pending_calls = tuple(
            PendingToolCallApproval.model_validate(raw_call) for raw_call in raw_calls
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("The pending tool-round call records are malformed.") from exc
    for pending_call, tool_call in zip(pending_calls, tool_calls, strict=True):
        if (
            pending_call.tool_call_id != tool_call.tool_call_id
            or pending_call.tool_name != tool_call.tool_name
            or not _runtime_publication_json_equal(
                pending_call.arguments,
                tool_call.arguments,
            )
        ):
            raise ValueError("The pending tool-round calls conflict with the assistant message.")
        if (
            pending_call.policy_evidence != ToolPolicyEvidence.UNPLANNED
            or pending_call.policy_decision is not None
            or pending_call.reason is not None
            or pending_call.metadata
            or pending_call.active_taint_labels
        ):
            raise ValueError("A newly published pending tool round cannot contain policy outcomes.")

    raw_structured_output = marker.get("structured_output")
    structured_output = None
    if raw_structured_output is not None:
        if type(raw_structured_output) is not dict:
            raise ValueError("The pending tool-round structured-output config is malformed.")
        try:
            structured_output = StructuredOutputSpec.model_validate(raw_structured_output)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "The pending tool-round structured-output config is malformed."
            ) from exc
    has_structured_finalizer = any(
        tool_call.tool_name == STRUCTURED_OUTPUT_TOOL_NAME for tool_call in tool_calls
    )
    structured_output_attempt = marker.get("structured_output_attempt")
    raw_structured_output_validation = marker.get("structured_output_validation")
    if has_structured_finalizer:
        if structured_output is None or structured_output.strategy != StructuredOutputStrategy.TOOL:
            raise ValueError(
                "The reserved structured-output call requires matching durable config."
            )
        if (
            type(structured_output_attempt) is not int
            or not 1 <= structured_output_attempt <= structured_output.max_retries + 1
        ):
            raise ValueError("The pending structured-output round requires a valid attempt number.")
        try:
            _validate_durable_structured_output_validation(
                raw_structured_output_validation,
                "pending_tool_round.structured_output_validation",
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "The pending structured-output round requires authoritative validation."
            ) from exc
    elif structured_output_attempt is not None or raw_structured_output_validation is not None:
        raise ValueError(
            "An ordinary pending tool round cannot carry structured-output attempt evidence."
        )


def _model_completion_stage_preparation_record(
    prepared: _PreparedModelCompletionStage,
    *,
    source_session: Session,
    prepared_at: datetime,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "record_type": MODEL_COMPLETION_STAGE_PREPARATION_RECORD_TYPE,
        "schema_version": MODEL_COMPLETION_STAGE_SCHEMA_VERSION,
        "session_id": prepared.session_id,
        "stage_id": prepared.request.stage_id,
        "logical_step_id": prepared.request.logical_step_id,
        "dispatch_ordinal": prepared.request.dispatch_ordinal,
        "purpose": prepared.request.purpose,
        "intent": prepared.request.intent,
        "reservation_ids": list(prepared.request.reservation_ids),
        "request_digest": prepared.request_digest,
        "source_status": str(source_session.status),
        "source_run_epoch": source_session.run_epoch,
        "source_transcript_cursor": prepared.expected_transcript_cursor,
        "prepared_at": prepared_at.isoformat().replace("+00:00", "Z"),
    }
    payload["record_digest"] = _canonical_runtime_publication_digest(payload)
    return copy_durable_json_object(payload, "model_completion_stage_preparation")


def _model_completion_stage_terminal_record(
    prepared: _PreparedModelCompletionStageTerminal,
    *,
    stage: ModelCompletionStage,
    completed_at: datetime,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "record_type": MODEL_COMPLETION_STAGE_TERMINAL_RECORD_TYPE,
        "schema_version": MODEL_COMPLETION_STAGE_SCHEMA_VERSION,
        "session_id": prepared.session_id,
        "stage_id": prepared.stage_id,
        "logical_step_id": stage.logical_step_id,
        "dispatch_ordinal": stage.dispatch_ordinal,
        "purpose": stage.purpose,
        "preparation_digest": stage.preparation_digest,
        "publication": prepared.publication.model_dump(mode="json"),
        "publication_material_digest": prepared.publication_material_digest,
        "completed_at": completed_at.isoformat().replace("+00:00", "Z"),
    }
    payload["record_digest"] = _canonical_runtime_publication_digest(payload)
    return copy_durable_json_object(payload, "model_completion_stage_terminal")


def _active_model_completion_stage_record(
    stage: ModelCompletionStage,
    *,
    activated_at: datetime,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "record_type": MODEL_COMPLETION_ACTIVE_STAGE_RECORD_TYPE,
        "schema_version": MODEL_COMPLETION_STAGE_SCHEMA_VERSION,
        "session_id": stage.session_id,
        "stage_id": stage.stage_id,
        "logical_step_id": stage.logical_step_id,
        "dispatch_ordinal": stage.dispatch_ordinal,
        "purpose": stage.purpose,
        "preparation_digest": stage.preparation_digest,
        "source_status": str(stage.source_status),
        "source_run_epoch": stage.source_run_epoch,
        "source_transcript_cursor": stage.source_transcript_cursor,
        "activated_at": activated_at.isoformat().replace("+00:00", "Z"),
    }
    payload["record_digest"] = _canonical_runtime_publication_digest(payload)
    return copy_durable_json_object(payload, "active_model_completion_stage")


def _model_completion_stage_abandonment_record(
    stage: ModelCompletionStage,
    *,
    active: ActiveModelCompletionStage,
    abandoned_at: datetime,
) -> dict[str, Any]:
    if stage.state != "in_flight" or active.stage != stage:
        raise SessionModelCompletionStageConflict(
            "Only the exact active in-flight model-completion stage can be abandoned."
        )
    payload: dict[str, Any] = {
        "record_type": MODEL_COMPLETION_STAGE_ABANDONMENT_RECORD_TYPE,
        "schema_version": MODEL_COMPLETION_STAGE_SCHEMA_VERSION,
        "session_id": stage.session_id,
        "stage_id": stage.stage_id,
        "logical_step_id": stage.logical_step_id,
        "dispatch_ordinal": stage.dispatch_ordinal,
        "purpose": stage.purpose,
        "preparation_request_digest": stage.preparation_request_digest,
        "preparation_digest": stage.preparation_digest,
        "active_marker_digest": active.marker_digest,
        "source_status": str(stage.source_status),
        "source_run_epoch": stage.source_run_epoch,
        "source_transcript_cursor": stage.source_transcript_cursor,
        "abandoned_at": abandoned_at.isoformat().replace("+00:00", "Z"),
    }
    payload["record_digest"] = _canonical_runtime_publication_digest(payload)
    return copy_durable_json_object(payload, "model_completion_stage_abandonment")


def _model_completion_stage_winner_record(
    stage: ModelCompletionStage,
    *,
    receipt: RuntimePublicationReceipt,
) -> dict[str, Any]:
    if (
        stage.state != "completed"
        or stage.completion_digest is None
        or stage.publication_material_digest is None
    ):
        raise SessionModelCompletionStageIncomplete(
            f"Model-completion stage is still in flight: {stage.stage_id}"
        )
    payload: dict[str, Any] = {
        "record_type": MODEL_COMPLETION_WINNER_RECORD_TYPE,
        "schema_version": MODEL_COMPLETION_STAGE_SCHEMA_VERSION,
        "session_id": stage.session_id,
        "stage_id": stage.stage_id,
        "logical_step_id": stage.logical_step_id,
        "dispatch_ordinal": stage.dispatch_ordinal,
        "purpose": stage.purpose,
        "preparation_digest": stage.preparation_digest,
        "completion_digest": stage.completion_digest,
        "publication_material_digest": stage.publication_material_digest,
        "receipt_request_digest": receipt.request_digest,
        "publication_digest": receipt.publication_digest,
        "published_at": receipt.published_at.isoformat().replace("+00:00", "Z"),
    }
    payload["record_digest"] = _canonical_runtime_publication_digest(payload)
    return copy_durable_json_object(payload, "model_completion_stage_winner")


def _require_raw_sha256_digest(value: Any) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError


def _reconstruct_active_model_completion_stage_record(
    record: dict[str, Any],
    *,
    session_id: str,
    storage_key: str = MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
) -> _ActiveModelCompletionStageRecord:
    try:
        if type(record) is not dict:
            raise TypeError
        if set(record) != set(_ActiveModelCompletionStageRecord.model_fields):
            raise ValueError
        for field in (
            "record_type",
            "session_id",
            "stage_id",
            "logical_step_id",
            "purpose",
            "preparation_digest",
            "source_status",
            "activated_at",
            "record_digest",
        ):
            if type(record[field]) is not str:
                raise TypeError
        for field in (
            "schema_version",
            "dispatch_ordinal",
            "source_run_epoch",
            "source_transcript_cursor",
        ):
            if type(record[field]) is not int:
                raise TypeError
        _require_raw_sha256_digest(record["preparation_digest"])
        _require_raw_sha256_digest(record["record_digest"])
        reconstructed = _ActiveModelCompletionStageRecord.model_validate(record)
        if reconstructed.record_type != MODEL_COMPLETION_ACTIVE_STAGE_RECORD_TYPE:
            raise ValueError
        if reconstructed.schema_version != MODEL_COMPLETION_STAGE_SCHEMA_VERSION:
            raise ValueError
        if reconstructed.session_id != session_id:
            raise ValueError
        if storage_key != MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY:
            raise ValueError
        digest_payload = reconstructed.model_dump(mode="json", exclude={"record_digest"})
        if _canonical_runtime_publication_digest(digest_payload) != reconstructed.record_digest:
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionModelCompletionStageConflict(
            "The durable active model-completion marker is malformed."
        ) from exc
    return reconstructed


def _reconstruct_model_completion_stage_abandonment(
    record: dict[str, Any],
    *,
    session_id: str,
    stage_id: str,
    storage_key: str,
) -> ModelCompletionStageAbandonment:
    try:
        if type(record) is not dict:
            raise TypeError
        if set(record) != set(_ModelCompletionStageAbandonmentRecord.model_fields):
            raise ValueError
        for field in (
            "record_type",
            "session_id",
            "stage_id",
            "logical_step_id",
            "purpose",
            "preparation_request_digest",
            "preparation_digest",
            "active_marker_digest",
            "source_status",
            "abandoned_at",
            "record_digest",
        ):
            if type(record[field]) is not str:
                raise TypeError
        for field in (
            "schema_version",
            "dispatch_ordinal",
            "source_run_epoch",
            "source_transcript_cursor",
        ):
            if type(record[field]) is not int:
                raise TypeError
        for field in (
            "preparation_request_digest",
            "preparation_digest",
            "active_marker_digest",
            "record_digest",
        ):
            _require_raw_sha256_digest(record[field])
        reconstructed = _ModelCompletionStageAbandonmentRecord.model_validate(record)
        if reconstructed.record_type != MODEL_COMPLETION_STAGE_ABANDONMENT_RECORD_TYPE:
            raise ValueError
        if reconstructed.schema_version != MODEL_COMPLETION_STAGE_SCHEMA_VERSION:
            raise ValueError
        if reconstructed.session_id != session_id or reconstructed.stage_id != stage_id:
            raise ValueError
        if storage_key != _model_completion_stage_abandonment_storage_key(stage_id):
            raise ValueError
        digest_payload = reconstructed.model_dump(mode="json", exclude={"record_digest"})
        if _canonical_runtime_publication_digest(digest_payload) != reconstructed.record_digest:
            raise ValueError
        return ModelCompletionStageAbandonment(
            session_id=reconstructed.session_id,
            stage_id=reconstructed.stage_id,
            logical_step_id=reconstructed.logical_step_id,
            dispatch_ordinal=reconstructed.dispatch_ordinal,
            purpose=reconstructed.purpose,
            preparation_request_digest=reconstructed.preparation_request_digest,
            preparation_digest=reconstructed.preparation_digest,
            active_marker_digest=reconstructed.active_marker_digest,
            source_status=reconstructed.source_status,
            source_run_epoch=reconstructed.source_run_epoch,
            source_transcript_cursor=reconstructed.source_transcript_cursor,
            abandoned_at=reconstructed.abandoned_at,
            abandonment_digest=reconstructed.record_digest,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionModelCompletionStageConflict(
            "The durable model-completion abandonment is malformed or conflicts with its key."
        ) from exc


def _reconstruct_model_completion_stage_winner_record(
    record: dict[str, Any],
    *,
    session_id: str,
    logical_step_id: str,
    storage_key: str,
) -> _ModelCompletionStageWinnerRecord:
    try:
        if type(record) is not dict:
            raise TypeError
        if set(record) != set(_ModelCompletionStageWinnerRecord.model_fields):
            raise ValueError
        for field in (
            "record_type",
            "session_id",
            "stage_id",
            "logical_step_id",
            "purpose",
            "preparation_digest",
            "completion_digest",
            "publication_material_digest",
            "receipt_request_digest",
            "publication_digest",
            "published_at",
            "record_digest",
        ):
            if type(record[field]) is not str:
                raise TypeError
        if type(record["schema_version"]) is not int:
            raise TypeError
        if type(record["dispatch_ordinal"]) is not int:
            raise TypeError
        for field in (
            "preparation_digest",
            "completion_digest",
            "publication_material_digest",
            "receipt_request_digest",
            "publication_digest",
            "record_digest",
        ):
            _require_raw_sha256_digest(record[field])
        reconstructed = _ModelCompletionStageWinnerRecord.model_validate(record)
        if reconstructed.record_type != MODEL_COMPLETION_WINNER_RECORD_TYPE:
            raise ValueError
        if reconstructed.schema_version != MODEL_COMPLETION_STAGE_SCHEMA_VERSION:
            raise ValueError
        if (
            reconstructed.session_id != session_id
            or reconstructed.logical_step_id != logical_step_id
            or storage_key != _model_completion_stage_winner_storage_key(logical_step_id)
        ):
            raise ValueError
        digest_payload = reconstructed.model_dump(mode="json", exclude={"record_digest"})
        if _canonical_runtime_publication_digest(digest_payload) != reconstructed.record_digest:
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionModelCompletionStageConflict(
            "The durable model-completion winner is malformed or conflicts with its key."
        ) from exc
    return reconstructed


def _reconstruct_model_completion_stage_preparation(
    record: dict[str, Any],
    *,
    session_id: str,
    stage_id: str,
    storage_key: str,
) -> _ModelCompletionStagePreparationRecord:
    try:
        if type(record) is not dict:
            raise TypeError
        if set(record) != set(_ModelCompletionStagePreparationRecord.model_fields):
            raise ValueError
        for field in (
            "record_type",
            "session_id",
            "stage_id",
            "logical_step_id",
            "purpose",
            "request_digest",
            "source_status",
            "prepared_at",
            "record_digest",
        ):
            if type(record[field]) is not str:
                raise TypeError
        if type(record["schema_version"]) is not int:
            raise TypeError
        if type(record["dispatch_ordinal"]) is not int:
            raise TypeError
        if type(record["intent"]) is not dict or type(record["reservation_ids"]) is not list:
            raise TypeError
        if type(record["source_run_epoch"]) is not int:
            raise TypeError
        if type(record["source_transcript_cursor"]) is not int:
            raise TypeError
        _require_raw_sha256_digest(record["request_digest"])
        _require_raw_sha256_digest(record["record_digest"])
        reconstructed = _ModelCompletionStagePreparationRecord.model_validate(record)
        if reconstructed.record_type != MODEL_COMPLETION_STAGE_PREPARATION_RECORD_TYPE:
            raise ValueError
        if reconstructed.schema_version != MODEL_COMPLETION_STAGE_SCHEMA_VERSION:
            raise ValueError
        if reconstructed.session_id != session_id or reconstructed.stage_id != stage_id:
            raise ValueError
        _, _, expected_key, _ = _model_completion_stage_storage_identity(session_id, stage_id)
        if storage_key != expected_key:
            raise ValueError
        digest_payload = reconstructed.model_dump(mode="json", exclude={"record_digest"})
        if _canonical_runtime_publication_digest(digest_payload) != reconstructed.record_digest:
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionModelCompletionStageConflict(
            "The durable model-completion preparation is malformed or conflicts with its key."
        ) from exc
    return reconstructed


def _reconstruct_model_completion_stage_terminal(
    record: dict[str, Any],
    *,
    session_id: str,
    stage_id: str,
    storage_key: str,
) -> _ModelCompletionStageTerminalRecord:
    try:
        if type(record) is not dict:
            raise TypeError
        if set(record) != set(_ModelCompletionStageTerminalRecord.model_fields):
            raise ValueError
        for field in (
            "record_type",
            "session_id",
            "stage_id",
            "logical_step_id",
            "purpose",
            "preparation_digest",
            "publication_material_digest",
            "completed_at",
            "record_digest",
        ):
            if type(record[field]) is not str:
                raise TypeError
        if (
            type(record["schema_version"]) is not int
            or type(record["dispatch_ordinal"]) is not int
            or type(record["publication"]) is not dict
        ):
            raise TypeError
        _require_raw_sha256_digest(record["preparation_digest"])
        _require_raw_sha256_digest(record["publication_material_digest"])
        _require_raw_sha256_digest(record["record_digest"])
        publication_record = record["publication"]
        required_publication_fields = set(RuntimePublicationRequest.model_fields) - {
            "interaction_id"
        }
        if not required_publication_fields.issubset(publication_record) or (
            set(publication_record) - set(RuntimePublicationRequest.model_fields)
        ):
            raise ValueError
        raw_transcript = publication_record["transcript_messages"]
        raw_events = publication_record["events"]
        if type(raw_transcript) is not list or type(raw_events) is not list:
            raise TypeError
        publication = RuntimePublicationRequest(
            publication_id=publication_record["publication_id"],
            kind=publication_record["kind"],
            interaction_id=publication_record.get("interaction_id"),
            intent=publication_record["intent"],
            mutation=publication_record["mutation"],
            transcript_messages=tuple(
                Message.model_validate(message) for message in raw_transcript
            ),
            events=tuple(Event.model_validate(event) for event in raw_events),
            referenced_events=publication_record["referenced_events"],
        )
        reconstructed = _ModelCompletionStageTerminalRecord.model_validate(
            {**record, "publication": publication}
        )
        if reconstructed.record_type != MODEL_COMPLETION_STAGE_TERMINAL_RECORD_TYPE:
            raise ValueError
        if reconstructed.schema_version != MODEL_COMPLETION_STAGE_SCHEMA_VERSION:
            raise ValueError
        if reconstructed.session_id != session_id or reconstructed.stage_id != stage_id:
            raise ValueError
        _, _, _, expected_key = _model_completion_stage_storage_identity(session_id, stage_id)
        if storage_key != expected_key:
            raise ValueError
        publication_payload = reconstructed.publication.model_dump(mode="json")
        if (
            _canonical_runtime_publication_digest(publication_payload)
            != reconstructed.publication_material_digest
        ):
            raise ValueError
        digest_payload = reconstructed.model_dump(mode="json", exclude={"record_digest"})
        if _canonical_runtime_publication_digest(digest_payload) != reconstructed.record_digest:
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionModelCompletionStageConflict(
            "The durable terminal model completion is malformed or conflicts with its key."
        ) from exc
    return reconstructed


def _reconstruct_model_completion_stage(
    preparation_record: dict[str, Any] | None,
    terminal_record: dict[str, Any] | None,
    *,
    session_id: str,
    stage_id: str,
    preparation_storage_key: str,
    terminal_storage_key: str,
) -> ModelCompletionStage | None:
    if preparation_record is None:
        if terminal_record is not None:
            raise SessionModelCompletionStageConflict(
                "A terminal model completion exists without its immutable preparation."
            )
        return None
    preparation = _reconstruct_model_completion_stage_preparation(
        preparation_record,
        session_id=session_id,
        stage_id=stage_id,
        storage_key=preparation_storage_key,
    )
    terminal = None
    if terminal_record is not None:
        terminal = _reconstruct_model_completion_stage_terminal(
            terminal_record,
            session_id=session_id,
            stage_id=stage_id,
            storage_key=terminal_storage_key,
        )
        if terminal.preparation_digest != preparation.record_digest:
            raise SessionModelCompletionStageConflict(
                "The terminal model completion does not match its immutable preparation."
            )
        if (
            terminal.logical_step_id != preparation.logical_step_id
            or terminal.dispatch_ordinal != preparation.dispatch_ordinal
            or terminal.purpose != preparation.purpose
        ):
            raise SessionModelCompletionStageConflict(
                "The terminal model completion identity conflicts with its preparation."
            )
        if not _runtime_publication_json_equal(terminal.publication.intent, preparation.intent):
            raise SessionModelCompletionStageConflict(
                "The terminal model completion intent conflicts with its preparation."
            )
        try:
            _validate_model_completion_stage_publication(
                terminal.publication,
                session_id=session_id,
                stage=preparation,
            )
        except (TypeError, ValueError) as exc:
            raise SessionModelCompletionStageConflict(
                "The terminal model completion publication conflicts with its preparation."
            ) from exc

    try:
        return ModelCompletionStage(
            session_id=session_id,
            stage_id=stage_id,
            logical_step_id=preparation.logical_step_id,
            dispatch_ordinal=preparation.dispatch_ordinal,
            purpose=preparation.purpose,
            state="in_flight" if terminal is None else "completed",
            intent=preparation.intent,
            reservation_ids=preparation.reservation_ids,
            preparation_request_digest=preparation.request_digest,
            preparation_digest=preparation.record_digest,
            source_status=preparation.source_status,
            source_run_epoch=preparation.source_run_epoch,
            source_transcript_cursor=preparation.source_transcript_cursor,
            prepared_at=preparation.prepared_at,
            publication=None if terminal is None else terminal.publication,
            publication_material_digest=(
                None if terminal is None else terminal.publication_material_digest
            ),
            completion_digest=None if terminal is None else terminal.record_digest,
            completed_at=None if terminal is None else terminal.completed_at,
        )
    except (TypeError, ValueError) as exc:
        raise SessionModelCompletionStageConflict(
            "The durable model-completion stage is malformed."
        ) from exc


def _reconstruct_active_model_completion_stage(
    active_record: dict[str, Any] | None,
    preparation_record: dict[str, Any] | None,
    terminal_record: dict[str, Any] | None,
    *,
    session_id: str,
) -> ActiveModelCompletionStage | None:
    if active_record is None:
        if preparation_record is not None or terminal_record is not None:
            raise SessionModelCompletionStageConflict(
                "Active model-completion material exists without its marker."
            )
        return None
    marker = _reconstruct_active_model_completion_stage_record(
        active_record,
        session_id=session_id,
    )
    _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
        session_id,
        marker.stage_id,
    )
    stage = _reconstruct_model_completion_stage(
        preparation_record,
        terminal_record,
        session_id=session_id,
        stage_id=marker.stage_id,
        preparation_storage_key=preparation_key,
        terminal_storage_key=terminal_key,
    )
    if stage is None:
        raise SessionModelCompletionStageConflict(
            "The active model-completion marker references a missing stage."
        )
    if (
        stage.logical_step_id != marker.logical_step_id
        or stage.dispatch_ordinal != marker.dispatch_ordinal
        or stage.purpose != marker.purpose
        or stage.preparation_digest != marker.preparation_digest
        or stage.source_status != marker.source_status
        or stage.source_run_epoch != marker.source_run_epoch
        or stage.source_transcript_cursor != marker.source_transcript_cursor
    ):
        raise SessionModelCompletionStageConflict(
            "The active model-completion marker conflicts with its stage."
        )
    try:
        return ActiveModelCompletionStage(
            stage=stage,
            activated_at=marker.activated_at,
            marker_digest=marker.record_digest,
        )
    except (TypeError, ValueError) as exc:
        raise SessionModelCompletionStageConflict(
            "The active model-completion snapshot is malformed."
        ) from exc


def _reconstruct_active_model_completion_stage_from_records(
    records: Mapping[str, dict[str, Any]],
    *,
    session_id: str,
) -> ActiveModelCompletionStage | None:
    active_record = records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY)
    if active_record is None:
        return None
    marker = _reconstruct_active_model_completion_stage_record(
        active_record,
        session_id=session_id,
    )
    _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
        session_id,
        marker.stage_id,
    )
    return _reconstruct_active_model_completion_stage(
        active_record,
        records.get(preparation_key),
        records.get(terminal_key),
        session_id=session_id,
    )


def _validate_model_completion_active_marker_for_preparation(
    active: ActiveModelCompletionStage | None,
    prepared: _PreparedModelCompletionStage,
    *,
    source_status: SessionStatus,
) -> None:
    if active is None:
        return
    current = active.stage
    request = prepared.request
    if current.state == "completed":
        raise SessionModelCompletionStageConflict(
            "A terminal model-completion stage must be published before another dispatch."
        )
    if (
        current.logical_step_id != request.logical_step_id
        or current.purpose != request.purpose
        or current.source_status != source_status
        or current.source_run_epoch != prepared.expected_run_epoch
        or current.source_transcript_cursor != prepared.expected_transcript_cursor
    ):
        raise SessionModelCompletionStageConflict(
            "A different logical model completion is already active for the session."
        )
    if request.dispatch_ordinal <= current.dispatch_ordinal:
        raise SessionModelCompletionStageConflict(
            "A model-completion retry must advance dispatch_ordinal monotonically."
        )


def _validate_model_completion_preparation_replay_state(
    stage: ModelCompletionStage,
    *,
    active: ActiveModelCompletionStage | None,
    winner_exists: bool,
    receipt_exists: bool,
) -> None:
    if winner_exists or receipt_exists:
        raise SessionModelCompletionStageConflict(
            "A prepared model-completion stage already has durable publication state."
        )
    if active is None:
        raise SessionModelCompletionStageConflict(
            "A prepared model-completion stage no longer has an active marker."
        )
    active_stage = active.stage
    if (
        active_stage.stage_id != stage.stage_id
        or active_stage.logical_step_id != stage.logical_step_id
        or active_stage.dispatch_ordinal != stage.dispatch_ordinal
        or active_stage.purpose != stage.purpose
        or active_stage.preparation_digest != stage.preparation_digest
        or active_stage.source_status != stage.source_status
        or active_stage.source_run_epoch != stage.source_run_epoch
        or active_stage.source_transcript_cursor != stage.source_transcript_cursor
    ):
        raise SessionModelCompletionStageConflict(
            "The prepared model-completion stage is no longer the exact active dispatch."
        )


def _validate_model_completion_stage_repreparation(
    abandonment_record: dict[str, Any] | None,
    prepared: _PreparedModelCompletionStage,
    *,
    source_status: SessionStatus,
) -> None:
    if abandonment_record is None:
        return
    abandonment = _reconstruct_model_completion_stage_abandonment(
        abandonment_record,
        session_id=prepared.session_id,
        stage_id=prepared.request.stage_id,
        storage_key=prepared.abandonment_storage_key,
    )
    request = prepared.request
    if (
        abandonment.logical_step_id != request.logical_step_id
        or abandonment.dispatch_ordinal != request.dispatch_ordinal
        or abandonment.purpose != request.purpose
        or abandonment.preparation_request_digest != prepared.request_digest
        or abandonment.source_status != source_status
        or abandonment.source_run_epoch != prepared.expected_run_epoch
        or abandonment.source_transcript_cursor != prepared.expected_transcript_cursor
    ):
        raise SessionModelCompletionStageConflict(
            "An abandoned model-completion stage id can only be reused for its exact "
            "request and durable source boundary."
        )


def _replay_model_completion_stage_abandonment(
    prepared: _PreparedModelCompletionStageAbandonment,
    abandonment_record: dict[str, Any] | None,
) -> ModelCompletionStageAbandonmentResult:
    if abandonment_record is None:
        raise KeyError(f"Model-completion stage not found: {prepared.stage_id}")
    abandonment = _reconstruct_model_completion_stage_abandonment(
        abandonment_record,
        session_id=prepared.session_id,
        stage_id=prepared.stage_id,
        storage_key=prepared.abandonment_storage_key,
    )
    if (
        abandonment.preparation_digest != prepared.preparation_digest
        or abandonment.source_run_epoch != prepared.expected_run_epoch
    ):
        raise SessionModelCompletionStageConflict(
            "The model-completion abandonment conflicts with the durable tombstone."
        )
    return ModelCompletionStageAbandonmentResult(
        abandonment=abandonment,
        replayed=True,
    )


def _validate_model_completion_stage_for_abandonment(
    *,
    session: Session,
    stage: ModelCompletionStage,
    active: ActiveModelCompletionStage | None,
    prepared: _PreparedModelCompletionStageAbandonment,
    abandonment_record: dict[str, Any] | None,
    winner_exists: bool,
    receipt_exists: bool,
) -> None:
    if stage.state != "in_flight":
        raise SessionModelCompletionStageConflict(
            "A terminal model-completion stage cannot be abandoned."
        )
    if stage.preparation_digest != prepared.preparation_digest:
        raise SessionModelCompletionStageConflict(
            "The model-completion abandonment preparation digest is stale."
        )
    if stage.source_run_epoch != prepared.expected_run_epoch:
        raise SessionRunFenced(
            "Model-completion stage run epoch is stale: expected "
            f"{prepared.expected_run_epoch}, prepared {stage.source_run_epoch}."
        )
    _assert_session_run_epoch(prepared.session_id, session)
    if session.run_epoch != prepared.expected_run_epoch:
        raise SessionRunFenced(
            "Session source run epoch is stale: expected "
            f"{prepared.expected_run_epoch}, current {session.run_epoch}."
        )
    if active is None or active.stage != stage:
        raise SessionModelCompletionStageConflict(
            "Only the exact active model-completion stage can be abandoned."
        )
    if winner_exists or receipt_exists:
        raise SessionModelCompletionStageConflict(
            "A model-completion stage with durable publication state cannot be abandoned."
        )
    if abandonment_record is None:
        return
    prior = _reconstruct_model_completion_stage_abandonment(
        abandonment_record,
        session_id=prepared.session_id,
        stage_id=prepared.stage_id,
        storage_key=prepared.abandonment_storage_key,
    )
    if (
        prior.logical_step_id != stage.logical_step_id
        or prior.dispatch_ordinal != stage.dispatch_ordinal
        or prior.purpose != stage.purpose
        or prior.preparation_request_digest != stage.preparation_request_digest
        or prior.source_status != stage.source_status
        or prior.source_run_epoch != stage.source_run_epoch
        or prior.source_transcript_cursor != stage.source_transcript_cursor
    ):
        raise SessionModelCompletionStageConflict(
            "The active model-completion stage conflicts with its prior abandonment."
        )
    if prior.preparation_digest == stage.preparation_digest:
        raise SessionModelCompletionStageConflict(
            "An abandoned model-completion preparation was unexpectedly restored."
        )


def _validate_model_completion_promotion_replay_active_marker(
    active: ActiveModelCompletionStage | None,
    stage: ModelCompletionStage,
) -> None:
    if active is not None and active.stage.logical_step_id == stage.logical_step_id:
        raise SessionModelCompletionStageConflict(
            "A published model-completion winner still has an active marker."
        )


def _model_completion_terminal_advances_last_activity(
    active_record: dict[str, Any] | None,
    *,
    stage: ModelCompletionStage,
    current_run_epoch: int,
) -> bool:
    if active_record is None:
        return False
    try:
        marker = _reconstruct_active_model_completion_stage_record(
            active_record,
            session_id=stage.session_id,
        )
    except SessionModelCompletionStageConflict:
        # Terminal provider evidence is append-only even when unrelated active
        # state is corrupt. A malformed marker can never refresh run liveness.
        return False
    return (
        stage.source_run_epoch == current_run_epoch
        and marker.stage_id == stage.stage_id
        and marker.logical_step_id == stage.logical_step_id
        and marker.dispatch_ordinal == stage.dispatch_ordinal
        and marker.purpose == stage.purpose
        and marker.preparation_digest == stage.preparation_digest
        and marker.source_status == stage.source_status
        and marker.source_run_epoch == stage.source_run_epoch
        and marker.source_transcript_cursor == stage.source_transcript_cursor
    )


def _validate_model_completion_active_marker_for_promotion(
    active: ActiveModelCompletionStage | None,
    stage: ModelCompletionStage,
) -> None:
    if active is None:
        raise SessionModelCompletionStageConflict(
            "The model-completion stage is no longer active and cannot be promoted."
        )
    if (
        active.stage.stage_id != stage.stage_id
        or active.stage.logical_step_id != stage.logical_step_id
        or active.stage.dispatch_ordinal != stage.dispatch_ordinal
        or active.stage.purpose != stage.purpose
        or active.stage.preparation_digest != stage.preparation_digest
    ):
        raise SessionModelCompletionStageConflict(
            "The model-completion stage was superseded before promotion."
        )


def _validate_model_completion_stage_winner(
    winner_record: dict[str, Any] | None,
    *,
    stage: ModelCompletionStage,
    receipt: RuntimePublicationReceipt,
    winner_storage_key: str,
) -> _ModelCompletionStageWinnerRecord:
    if winner_record is None:
        raise SessionModelCompletionStageConflict(
            "A model-completion receipt exists without its durable winner claim."
        )
    winner = _reconstruct_model_completion_stage_winner_record(
        winner_record,
        session_id=stage.session_id,
        logical_step_id=stage.logical_step_id,
        storage_key=winner_storage_key,
    )
    if (
        winner.stage_id != stage.stage_id
        or winner.dispatch_ordinal != stage.dispatch_ordinal
        or winner.purpose != stage.purpose
        or winner.preparation_digest != stage.preparation_digest
        or winner.completion_digest != stage.completion_digest
        or winner.publication_material_digest != stage.publication_material_digest
        or winner.receipt_request_digest != receipt.request_digest
        or winner.publication_digest != receipt.publication_digest
        or winner.published_at != receipt.published_at
    ):
        raise SessionModelCompletionStageConflict(
            "The runtime publication receipt belongs to a different model-completion winner."
        )
    return winner


def _validate_model_completion_stage_preparation_replay(
    stage: ModelCompletionStage,
    prepared: _PreparedModelCompletionStage,
) -> None:
    if stage.preparation_request_digest != prepared.request_digest:
        raise SessionModelCompletionStageConflict(
            "The model-completion stage id was already prepared for a different request."
        )


def _validate_model_completion_stage_terminal_replay(
    stage: ModelCompletionStage,
    prepared: _PreparedModelCompletionStageTerminal,
) -> None:
    if (
        stage.state != "completed"
        or stage.publication_material_digest != prepared.publication_material_digest
    ):
        raise SessionModelCompletionStageConflict(
            "The model-completion stage already has different terminal material."
        )


def _prepare_model_completion_stage_promotion(
    stage: ModelCompletionStage,
    *,
    expected_run_epoch: int,
) -> _PreparedRuntimePublication:
    if stage.state != "completed" or stage.publication is None:
        raise SessionModelCompletionStageIncomplete(
            f"Model-completion stage is still in flight: {stage.stage_id}"
        )
    return _prepare_runtime_publication(
        stage.session_id,
        stage.publication,
        expected_statuses={
            stage.source_status,
            SessionStatus.INTERRUPTING,
            SessionStatus.FAILED,
        },
        expected_run_epoch=expected_run_epoch,
        expected_transcript_cursor=stage.source_transcript_cursor,
    )


def _replay_promoted_model_completion_stage(
    *,
    session: Session,
    stage: ModelCompletionStage,
    receipt_record: dict[str, Any],
    winner_record: dict[str, Any] | None,
) -> RuntimePublicationResult:
    if stage.state != "completed" or stage.publication is None:
        raise SessionModelCompletionStageIncomplete(
            f"Model-completion stage is still in flight: {stage.stage_id}"
        )
    storage_key = _runtime_publication_storage_key(stage.logical_step_id)
    receipt = _reconstruct_runtime_publication_receipt(
        receipt_record,
        storage_key=storage_key,
        session_id=stage.session_id,
        publication_id=stage.logical_step_id,
    )
    if (
        receipt.source_status
        not in {
            stage.source_status,
            SessionStatus.INTERRUPTING,
            SessionStatus.FAILED,
        }
        or receipt.transcript_start_cursor != stage.source_transcript_cursor
    ):
        raise SessionModelCompletionStageConflict(
            "The runtime publication receipt conflicts with its model-completion stage."
        )
    _validate_model_completion_stage_winner(
        winner_record,
        stage=stage,
        receipt=receipt,
        winner_storage_key=_model_completion_stage_winner_storage_key(stage.logical_step_id),
    )
    prepared = _prepare_runtime_publication(
        stage.session_id,
        stage.publication,
        expected_statuses={stage.source_status},
        expected_run_epoch=receipt.source_run_epoch,
        expected_transcript_cursor=stage.source_transcript_cursor,
    )
    receipt = _reconstruct_runtime_publication_receipt(
        receipt_record,
        storage_key=prepared.storage_key,
        session_id=stage.session_id,
        publication_id=stage.logical_step_id,
        request_digest=prepared.request_digest,
    )
    _validate_runtime_publication_replay_receipt(receipt, prepared)
    return RuntimePublicationResult(
        session=session.model_copy(deep=True),
        receipt=receipt,
        replayed=True,
    )


def _prepare_runtime_publication(
    session_id: str,
    request: RuntimePublicationRequest,
    *,
    expected_statuses: set[SessionStatus] | None,
    expected_run_epoch: int | None,
    expected_transcript_cursor: int | None,
) -> _PreparedRuntimePublication:
    session_id = require_clean_nonblank(session_id, "session_id")
    if type(request) is not RuntimePublicationRequest:
        raise TypeError("Runtime publication requires a RuntimePublicationRequest.")
    try:
        copied_request = RuntimePublicationRequest(
            publication_id=request.publication_id,
            kind=request.kind,
            interaction_id=request.interaction_id,
            intent=request.intent,
            mutation=request.mutation,
            transcript_messages=request.transcript_messages,
            events=request.events,
            referenced_events=request.referenced_events,
        )
    except AttributeError as exc:
        raise ValueError("Runtime publication request is malformed.") from exc

    appended_event_ids = {event.id for event in copied_request.events}
    referenced_event_ids = set(
        _runtime_publication_referenced_event_ids(copied_request.referenced_events)
    )
    overlap = appended_event_ids & referenced_event_ids
    if overlap:
        raise ValueError(
            f"Runtime publication cannot both append and reference event id {min(overlap)}."
        )
    for event in copied_request.events:
        if event.session_id != session_id:
            raise ValueError("Runtime publication event session_id does not match target session.")
        if event.interaction_id != copied_request.interaction_id:
            raise ValueError(
                "Runtime publication event interaction_id does not match its transcript "
                "attribution."
            )

    allowed_statuses = (
        None
        if expected_statuses is None
        else frozenset(_validate_status_set(expected_statuses, "expected_statuses"))
    )
    expected_run_epoch = _validate_optional_runtime_publication_fence(
        expected_run_epoch,
        "expected_run_epoch",
    )
    expected_transcript_cursor = _validate_optional_runtime_publication_fence(
        expected_transcript_cursor,
        "expected_transcript_cursor",
    )
    transcript_payloads = tuple(
        message.model_dump(mode="json") for message in copied_request.transcript_messages
    )
    event_payloads = tuple(event.model_dump(mode="json") for event in copied_request.events)
    transcript_digest = _canonical_runtime_publication_digest(list(transcript_payloads))
    events_digest = _canonical_runtime_publication_digest(list(event_payloads))
    request_digest = _canonical_runtime_publication_digest(
        {
            "session_id": session_id,
            "publication_id": copied_request.publication_id,
            "kind": copied_request.kind,
            "interaction_id": copied_request.interaction_id,
            "intent": copied_request.intent,
            "mutation": copied_request.mutation.model_dump(mode="json"),
            "transcript_messages": list(transcript_payloads),
            "events": list(event_payloads),
            "referenced_events": [
                reference.model_dump(mode="json") for reference in copied_request.referenced_events
            ],
        }
    )
    return _PreparedRuntimePublication(
        session_id=session_id,
        storage_key=_runtime_publication_storage_key(copied_request.publication_id),
        request=copied_request,
        request_digest=request_digest,
        transcript_payloads=transcript_payloads,
        event_payloads=event_payloads,
        transcript_digest=transcript_digest,
        events_digest=events_digest,
        expected_statuses=allowed_statuses,
        expected_run_epoch=expected_run_epoch,
        expected_transcript_cursor=expected_transcript_cursor,
    )


def _next_runtime_publication_timestamp(session: Session) -> datetime:
    now = datetime.now(UTC)
    persisted = max(session.updated_at.astimezone(UTC), session.last_activity_at.astimezone(UTC))
    if now <= persisted:
        return persisted + timedelta(microseconds=1)
    return now


def _build_runtime_publication_receipt(
    prepared: _PreparedRuntimePublication,
    *,
    source_session: Session,
    checkpoint: dict[str, Any] | None,
    transcript_start_cursor: int,
    published_at: datetime,
) -> RuntimePublicationReceipt:
    request = prepared.request
    receipt_payload: dict[str, Any] = {
        "record_type": RUNTIME_PUBLICATION_RECORD_TYPE,
        "schema_version": RUNTIME_PUBLICATION_SCHEMA_VERSION,
        "session_id": prepared.session_id,
        "publication_id": request.publication_id,
        "kind": request.kind,
        "interaction_id": request.interaction_id,
        "intent": request.intent,
        "request_digest": prepared.request_digest,
        "checkpoint_digest": _canonical_runtime_publication_digest(checkpoint),
        "transcript_digest": prepared.transcript_digest,
        "events_digest": prepared.events_digest,
        "source_status": str(source_session.status),
        "source_run_epoch": source_session.run_epoch,
        "transcript_start_cursor": transcript_start_cursor,
        "transcript_end_cursor": transcript_start_cursor + len(request.transcript_messages),
        "appended_event_ids": [event.id for event in request.events],
        "referenced_events": [
            reference.model_dump(mode="json") for reference in request.referenced_events
        ],
        "published_at": published_at.isoformat().replace("+00:00", "Z"),
    }
    publication_digest = _canonical_runtime_publication_digest(receipt_payload)
    return RuntimePublicationReceipt(
        **receipt_payload,
        publication_digest=publication_digest,
    )


def _runtime_publication_receipt_record(
    receipt: RuntimePublicationReceipt,
) -> dict[str, Any]:
    return copy_durable_json_object(
        receipt.model_dump(mode="json"),
        "runtime_publication_receipt",
    )


def _reconstruct_runtime_publication_receipt(
    record: dict[str, Any],
    *,
    storage_key: str,
    session_id: str,
    publication_id: str,
    request_digest: str | None = None,
) -> RuntimePublicationReceipt:
    try:
        if type(record) is not dict:
            raise TypeError
        if set(record) != set(RuntimePublicationReceipt.model_fields):
            raise ValueError
        string_fields = (
            "record_type",
            "session_id",
            "publication_id",
            "kind",
            "request_digest",
            "publication_digest",
            "checkpoint_digest",
            "transcript_digest",
            "events_digest",
            "source_status",
            "published_at",
        )
        if any(type(record[field]) is not str for field in string_fields):
            raise TypeError
        if type(record["schema_version"]) is not int:
            raise TypeError
        if type(record["intent"]) is not dict:
            raise TypeError
        if record["interaction_id"] is not None and type(record["interaction_id"]) is not str:
            raise TypeError
        for field in (
            "source_run_epoch",
            "transcript_start_cursor",
            "transcript_end_cursor",
        ):
            if type(record[field]) is not int:
                raise TypeError
        if type(record["appended_event_ids"]) is not list:
            raise TypeError
        if type(record["referenced_events"]) is not list:
            raise TypeError
        receipt = RuntimePublicationReceipt.model_validate(record)
        if receipt.record_type != RUNTIME_PUBLICATION_RECORD_TYPE:
            raise ValueError
        if receipt.schema_version != RUNTIME_PUBLICATION_SCHEMA_VERSION:
            raise ValueError
        if receipt.session_id != session_id or receipt.publication_id != publication_id:
            raise ValueError
        if _runtime_publication_storage_key(receipt.publication_id) != storage_key:
            raise ValueError
        digest_payload = receipt.model_dump(mode="json", exclude={"publication_digest"})
        if _canonical_runtime_publication_digest(digest_payload) != receipt.publication_digest:
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication receipt is malformed or conflicts with its key."
        ) from exc
    if request_digest is not None and receipt.request_digest != request_digest:
        raise SessionRuntimePublicationConflict(
            "The runtime publication id was already used for a different request."
        )
    return receipt


def _validate_runtime_publication_replay_receipt(
    receipt: RuntimePublicationReceipt,
    prepared: _PreparedRuntimePublication,
) -> None:
    request = prepared.request
    if (
        receipt.kind != request.kind
        or receipt.interaction_id != request.interaction_id
        or not _runtime_publication_json_equal(receipt.intent, request.intent)
        or receipt.transcript_digest != prepared.transcript_digest
        or receipt.events_digest != prepared.events_digest
        or receipt.appended_event_ids != tuple(event.id for event in request.events)
        or receipt.referenced_events != request.referenced_events
        or receipt.transcript_end_cursor - receipt.transcript_start_cursor
        != len(request.transcript_messages)
    ):
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication receipt conflicts with its request material."
        )


def _validate_runtime_publication_durable_material(
    receipt: RuntimePublicationReceipt,
    *,
    transcript_messages: Iterable[Message],
    transcript_interaction_ids: Iterable[str | None],
    appended_events: Iterable[Event],
    durable_referenced_events: Iterable[Event],
) -> None:
    """Fail closed when a receipt no longer describes its committed durable material.

    Checkpoints intentionally are not re-read here: later legitimate checkpoint
    transforms replace that mutable snapshot. ``checkpoint_digest`` remains
    commit-time evidence, while transcript/event records are immutable material
    that must continue to match on every load and replay.
    """

    try:
        transcript = tuple(detach_message(message) for message in transcript_messages)
        interaction_ids = tuple(transcript_interaction_ids)
        events = tuple(copy_event(event) for event in appended_events)
        durable_references = tuple(copy_event(event) for event in durable_referenced_events)
    except (AttributeError, KeyError, TypeError, UnicodeError, ValueError) as exc:
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication material is malformed."
        ) from exc
    expected_transcript_count = receipt.transcript_end_cursor - receipt.transcript_start_cursor
    if len(transcript) != expected_transcript_count:
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication transcript segment is missing or truncated."
        )
    if len(interaction_ids) != expected_transcript_count or any(
        interaction_id != receipt.interaction_id for interaction_id in interaction_ids
    ):
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication transcript attribution conflicts with its receipt."
        )
    transcript_digest = _canonical_runtime_publication_digest(
        [message.model_dump(mode="json") for message in transcript]
    )
    if transcript_digest != receipt.transcript_digest:
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication transcript segment conflicts with its receipt."
        )

    if tuple(event.id for event in events) != receipt.appended_event_ids:
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication event batch is missing or reordered."
        )
    events_digest = _canonical_runtime_publication_digest(
        [event.model_dump(mode="json") for event in events]
    )
    if events_digest != receipt.events_digest:
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication event batch conflicts with its receipt."
        )
    if any(event.interaction_id != receipt.interaction_id for event in events):
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication event attribution conflicts with its receipt."
        )

    referenced_event_ids = _runtime_publication_referenced_event_ids(receipt.referenced_events)
    if tuple(event.id for event in durable_references) != referenced_event_ids:
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication references a missing event."
        )
    for reference, durable_event in zip(
        receipt.referenced_events,
        durable_references,
        strict=True,
    ):
        if _runtime_publication_event_digest(durable_event) != reference.event_digest:
            raise SessionRuntimePublicationConflict(
                "The durable runtime publication referenced event content conflicts with "
                "its receipt."
            )
        if durable_event.interaction_id != receipt.interaction_id:
            raise SessionRuntimePublicationConflict(
                "The durable runtime publication referenced event attribution conflicts "
                "with its receipt."
            )


def _copy_session_event_batch(session_id: str, events: list[Event]) -> tuple[str, list[Event]]:
    session_id = require_clean_nonblank(session_id, "session_id")
    if type(events) is not list:
        raise TypeError("Session events must be a list.")

    copied_events: list[Event] = []
    seen_event_ids: set[str] = set()
    for event in events:
        if type(event) is not Event:
            raise TypeError("Session events must be Event instances.")
        copied_event = copy_event(event)
        if copied_event.session_id != session_id:
            raise ValueError("Event session_id does not match target session.")
        if copied_event.id in seen_event_ids:
            raise ValueError(f"Event already exists for session {session_id}: {copied_event.id}")
        seen_event_ids.add(copied_event.id)
        copied_events.append(copied_event)
    return session_id, copied_events


def _copy_workflow_step_reservation(
    session_id: str,
    event: Event,
    *,
    workflow_name: str,
    attempt_id: str,
) -> tuple[str, Event, str, str]:
    session_id, copied_events = _copy_session_event_batch(session_id, [event])
    copied_event = copied_events[0]
    workflow_name = require_clean_nonblank(workflow_name, "workflow_name")
    attempt_id = require_clean_nonblank(attempt_id, "attempt_id")
    if copied_event.type != EventType.WORKFLOW_STEP_STARTED:
        raise ValueError("Workflow step reservation requires a workflow.step.started event.")
    if copied_event.workflow_name != workflow_name:
        raise ValueError("Workflow step reservation event has the wrong workflow_name.")
    if copied_event.payload.get("attempt_id") != attempt_id:
        raise ValueError("Workflow step reservation event has the wrong attempt_id.")
    step_id = copied_event.payload.get("step_id")
    if not isinstance(step_id, str) or not step_id:
        raise ValueError("Workflow step reservation requires a non-empty step_id.")
    return session_id, copied_event, workflow_name, attempt_id


def _validate_mcp_manifest_history_keys(history_keys: tuple[str, ...]) -> tuple[str, ...]:
    if type(history_keys) is not tuple:
        raise TypeError("history_keys must be a tuple.")
    keys = tuple(_require_sha256_identifier(key, "history_keys item") for key in history_keys)
    if len(keys) != len(set(keys)):
        raise ValueError("history_keys must be unique.")
    return keys


def _stored_mcp_manifest_baseline(
    history_key: str,
    generation: int,
    value: Any,
) -> McpManifestBaseline:
    try:
        history_key = _require_sha256_identifier(history_key, "stored MCP history_key")
        if type(generation) is not int or generation < 1:
            raise ValueError("Stored MCP manifest generation must be a positive integer.")
        baseline = McpManifestBaseline.model_validate(value)
        if baseline.history_key != history_key or baseline.generation != generation:
            raise ValueError("Stored MCP manifest baseline columns disagree with its payload.")
        return baseline
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise _McpManifestBaselineEvidenceInvalid(
            "Stored MCP manifest baseline evidence is invalid."
        ) from None


def _stored_mcp_manifest_baseline_json(
    history_key: str,
    generation: int,
    value: Any,
) -> McpManifestBaseline:
    try:
        if not isinstance(value, str):
            raise TypeError("Stored MCP manifest baseline JSON must be text.")
        decoded = json.loads(value)
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise _McpManifestBaselineEvidenceInvalid(
            "Stored MCP manifest baseline evidence is invalid."
        ) from None
    return _stored_mcp_manifest_baseline(history_key, generation, decoded)


def _copy_mcp_manifest_publication(
    session_id: str,
    *,
    expected_generations: dict[str, int | None],
    baseline_updates: dict[str, McpManifestBaseline],
    events: list[Event],
) -> tuple[
    str,
    dict[str, int | None],
    dict[str, McpManifestBaseline],
    list[Event],
]:
    if type(expected_generations) is not dict or not expected_generations:
        raise ValueError("expected_generations must be a non-empty dict.")
    expected: dict[str, int | None] = {}
    for key, generation in expected_generations.items():
        key = _require_sha256_identifier(key, "expected_generations key")
        if generation is not None and (type(generation) is not int or generation < 1):
            raise ValueError("Expected MCP manifest generations must be positive integers or None.")
        expected[key] = generation

    if type(baseline_updates) is not dict:
        raise TypeError("baseline_updates must be a dict.")
    updates: dict[str, McpManifestBaseline] = {}
    for key, baseline in baseline_updates.items():
        key = _require_sha256_identifier(key, "baseline_updates key")
        if key not in expected:
            raise ValueError("MCP baseline update has no matching generation expectation.")
        if type(baseline) is not McpManifestBaseline:
            raise TypeError("baseline_updates must contain McpManifestBaseline values.")
        copied = McpManifestBaseline.model_validate(baseline.model_dump(mode="python"))
        if copied.history_key != key:
            raise ValueError("MCP baseline history_key does not match its update key.")
        expected_generation = expected[key]
        next_generation = 1 if expected_generation is None else expected_generation + 1
        if copied.generation != next_generation:
            raise ValueError("MCP baseline update generation does not follow its expectation.")
        updates[key] = copied

    session_id, copied_events = _copy_session_event_batch(session_id, events)
    if not copied_events:
        raise ValueError("MCP manifest publication requires at least one event.")
    for event in copied_events:
        if event.type not in {
            EventType.MCP_MANIFEST_CHECKED,
            EventType.MCP_MANIFEST_BLOCKED,
        }:
            raise ValueError("MCP manifest publication contains an unrelated event.")
        history_key = event.payload.get("history_key")
        if not isinstance(history_key, str) or history_key not in expected:
            raise ValueError("MCP manifest event has no matching history expectation.")
        if (
            event.type == EventType.MCP_MANIFEST_CHECKED
            and event.payload.get("status") in {"first_seen", "changed"}
            and event.payload.get("outcome") == "accepted"
        ):
            baseline = updates.get(history_key)
            if baseline is None or baseline.accepted_event_id != event.id:
                raise ValueError(
                    "Every accepted MCP manifest transition must have exactly one "
                    "matching baseline update."
                )
    events_by_id = {event.id: event for event in copied_events}
    for history_key, baseline in updates.items():
        accepted_event = events_by_id.get(baseline.accepted_event_id)
        if (
            accepted_event is None
            or accepted_event.type != EventType.MCP_MANIFEST_CHECKED
            or _mcp_manifest_session_ref(accepted_event.session_id) != baseline.accepted_session_ref
            or accepted_event.timestamp != baseline.accepted_at
            or accepted_event.payload.get("history_key") != history_key
            or accepted_event.payload.get("manifest_identity") != baseline.manifest_identity
            or accepted_event.payload.get("manifest_hash") != baseline.manifest_hash
            or accepted_event.payload.get("source_manifest_hash") != baseline.source_manifest_hash
            or accepted_event.payload.get("server_hash") != baseline.server_hash
            or accepted_event.payload.get("status") not in {"first_seen", "changed"}
            or accepted_event.payload.get("outcome") != "accepted"
        ):
            raise ValueError(
                "MCP baseline update must match exactly one accepted manifest-check event."
            )
    return session_id, expected, updates, copied_events


def _validate_mcp_manifest_publication_state(
    *,
    expected_generations: Mapping[str, int | None],
    current_baselines: Mapping[str, McpManifestBaseline],
    baseline_updates: Mapping[str, McpManifestBaseline],
    events: Iterable[Event],
) -> None:
    """Validate one fenced publication against its authoritative current state."""

    events_by_history_key: dict[str, list[Event]] = {
        history_key: [] for history_key in expected_generations
    }
    for event in events:
        history_key = event.payload.get("history_key")
        if not isinstance(history_key, str) or history_key not in events_by_history_key:
            raise ValueError("MCP manifest event has no matching history expectation.")
        events_by_history_key[history_key].append(event)

    if any(len(history_events) != 1 for history_events in events_by_history_key.values()):
        raise ValueError(
            "MCP manifest publication must contain exactly one outcome per history key."
        )

    for history_key, history_events in events_by_history_key.items():
        event = history_events[0]
        status = event.payload.get("status")
        outcome = event.payload.get("outcome")
        current = current_baselines.get(history_key)
        update = baseline_updates.get(history_key)

        if status == "first_seen":
            if current is not None:
                raise ValueError("An MCP first_seen outcome cannot replace an existing baseline.")
        elif status in {"changed", "unchanged"}:
            if current is None:
                raise ValueError(f"An MCP {status} outcome requires an existing baseline.")
        else:
            raise ValueError("MCP manifest publication contains an invalid status.")

        if outcome == "accepted":
            if event.type != EventType.MCP_MANIFEST_CHECKED:
                raise ValueError("An accepted MCP manifest outcome must be a checked event.")
            if status in {"first_seen", "changed"}:
                if update is None:
                    raise ValueError("An accepted MCP manifest transition requires an update.")
            elif update is not None:
                raise ValueError("An unchanged MCP manifest outcome cannot update its baseline.")
        elif outcome == "blocked":
            if event.type != EventType.MCP_MANIFEST_BLOCKED:
                raise ValueError("A blocked MCP manifest outcome must be a blocked event.")
            if update is not None:
                raise ValueError("A blocked MCP manifest outcome cannot update its baseline.")
        elif outcome == "batch_blocked":
            if event.type != EventType.MCP_MANIFEST_CHECKED:
                raise ValueError("A batch-blocked MCP manifest outcome must be a checked event.")
            if update is not None:
                raise ValueError("A batch-blocked MCP manifest outcome cannot update its baseline.")
        else:
            raise ValueError("MCP manifest publication contains an invalid outcome.")

        event_evidence_values: list[str] = []
        for field in (
            "manifest_identity",
            "manifest_hash",
            "source_manifest_hash",
            "server_hash",
        ):
            value = event.payload.get(field)
            if not isinstance(value, str):
                raise ValueError(f"MCP event {field} must be a SHA-256 identifier.")
            event_evidence_values.append(_require_sha256_identifier(value, f"MCP event {field}"))
        event_evidence = tuple(event_evidence_values)
        if current is not None:
            current_evidence = (
                current.manifest_identity,
                current.manifest_hash,
                current.source_manifest_hash,
                current.server_hash,
            )
            if event_evidence[0] != current.manifest_identity:
                raise ValueError(
                    "MCP manifest identity cannot change within one durable history key."
                )
            if status == "unchanged" and event_evidence != current_evidence:
                raise ValueError(
                    "An unchanged MCP manifest event must match the current baseline exactly."
                )
            if status == "changed" and event_evidence == current_evidence:
                raise ValueError(
                    "A changed MCP manifest event must differ from the current baseline."
                )


def _validate_equivalent_message_delivery(
    existing: _InMemoryMessageDeliveryRecord,
    *,
    session_id: str,
    include_on_idle: bool,
    eligible_through: int | None,
    limit: int,
    interaction_id: str | None,
    interaction_started_event: Event | None,
) -> None:
    if (
        existing.session_id != session_id
        or existing.include_on_idle != include_on_idle
        or existing.requested_eligible_through != eligible_through
        or existing.limit != limit
        or existing.interaction_id != interaction_id
        or existing.interaction_started_event != interaction_started_event
    ):
        raise ValueError("delivery_id was already used for a different queue delivery.")


def _validate_message_delivery_eligible_through(value: int | None) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError(
            f"eligible_through must be an integer between 0 and {MAX_DURABLE_JSON_INTEGER}."
        )
    return value


def _copy_queued_interaction_started_event(
    session_id: str,
    interaction_id: str | None,
    event: Event | None,
) -> Event | None:
    if event is None:
        return None
    if interaction_id is None:
        raise ValueError("interaction_id is required with interaction_started_event.")
    copied = copy_event(event)
    if copied.session_id != session_id:
        raise ValueError("interaction_started_event belongs to a different session.")
    if copied.interaction_id != interaction_id:
        raise ValueError("interaction_started_event belongs to a different interaction.")
    if copied.type != EventType.INTERACTION_STARTED:
        raise ValueError("interaction_started_event must be interaction.started.")
    return copied


def _prepare_interaction_transition(
    session_id: str,
    *,
    event: Event,
    from_statuses: set[SessionStatus],
    to_status: SessionStatus,
    only_if_no_queued_messages: bool,
) -> tuple[str, Event, set[SessionStatus], SessionStatus, bool]:
    session_id = require_clean_nonblank(session_id, "session_id")
    if type(event) is not Event:
        raise TypeError("event must be an Event.")
    copied_event = copy_event(event)
    if copied_event.session_id != session_id:
        raise ValueError("Interaction transition event belongs to a different session.")
    if copied_event.interaction_id is None:
        raise ValueError("Interaction transition event must have an interaction_id.")
    statuses_by_event_type: dict[str, set[SessionStatus]] = {
        EventType.INTERACTION_PAUSED: {
            SessionStatus.FAILED,
            SessionStatus.INTERRUPTED,
        },
        EventType.INTERACTION_COMPLETED: {SessionStatus.COMPLETED},
        EventType.INTERACTION_FAILED: {SessionStatus.FAILED},
        EventType.INTERACTION_INTERRUPTED: {SessionStatus.INTERRUPTED},
    }
    expected_statuses = statuses_by_event_type.get(copied_event.type)
    if expected_statuses is None:
        raise ValueError("event must be a terminal or paused interaction lifecycle event.")
    if not isinstance(to_status, SessionStatus):
        raise ValueError("to_status must be a SessionStatus.")
    if to_status not in expected_statuses:
        rendered_statuses = ", ".join(sorted(str(status) for status in expected_statuses))
        raise ValueError(
            f"{copied_event.type} requires session status in "
            f"{{{rendered_statuses}}}, not {to_status}."
        )
    if type(only_if_no_queued_messages) is not bool:
        raise TypeError("only_if_no_queued_messages must be a bool.")
    if only_if_no_queued_messages and copied_event.type != EventType.INTERACTION_COMPLETED:
        raise ValueError("only_if_no_queued_messages is valid only for interaction.completed.")
    return (
        session_id,
        copied_event,
        _validate_status_set(from_statuses, "from_statuses"),
        to_status,
        only_if_no_queued_messages,
    )


def _copy_interaction_admission(
    session_id: str,
    started_event: Event,
    source_messages: list[Message],
    *,
    defer_transcript: bool,
) -> tuple[Event, list[Message]]:
    session_id = require_clean_nonblank(session_id, "session_id")
    if type(defer_transcript) is not bool:
        raise TypeError("defer_transcript must be a bool.")
    if type(started_event) is not Event:
        raise TypeError("started_event must be an Event.")
    copied_event = copy_event(started_event)
    if copied_event.session_id != session_id:
        raise ValueError("started_event belongs to a different session.")
    if copied_event.type != EventType.INTERACTION_STARTED:
        raise ValueError("started_event must be interaction.started.")
    if copied_event.interaction_id is None:
        raise ValueError("started_event must have an interaction_id.")
    copied_messages = _detach_transcript_messages(source_messages)
    return copied_event, copied_messages


def _copy_optional_interaction_admission(
    session_id: str,
    started_event: Event | None,
    source_messages: list[Message] | None,
    *,
    defer_transcript: bool,
) -> tuple[Event, list[Message]] | None:
    if (started_event is None) != (source_messages is None):
        raise ValueError(
            "interaction_started_event and interaction_source_messages must be supplied together."
        )
    if started_event is None or source_messages is None:
        return None
    return _copy_interaction_admission(
        session_id,
        started_event,
        source_messages,
        defer_transcript=defer_transcript,
    )


def _copy_transition_interaction_admission(
    session_id: str,
    started_event: Event | None,
    source_messages: list[Message] | None,
    *,
    continued_interaction_id: str | None,
    defer_interaction_source: bool,
) -> tuple[Event | None, str, list[Message], bool] | None:
    if type(defer_interaction_source) is not bool:
        raise TypeError("defer_interaction_source must be a bool.")
    if started_event is not None and continued_interaction_id is not None:
        raise ValueError(
            "interaction_started_event and continued_interaction_id are mutually exclusive."
        )
    if started_event is None and continued_interaction_id is None:
        if source_messages is not None or defer_interaction_source:
            raise ValueError("Interaction source messages require an interaction identity.")
        return None
    if source_messages is None:
        raise ValueError("interaction_source_messages are required for interaction admission.")
    if started_event is not None:
        event, messages = _copy_interaction_admission(
            session_id,
            started_event,
            source_messages,
            defer_transcript=defer_interaction_source,
        )
        interaction_id = event.interaction_id
        if interaction_id is None:
            raise AssertionError("Interaction admission lost its identity.")
        return event, interaction_id, messages, defer_interaction_source
    if continued_interaction_id is None:
        raise AssertionError("Continued interaction admission lost its identity.")
    interaction_id = require_clean_nonblank(continued_interaction_id, "continued_interaction_id")
    return (
        None,
        interaction_id,
        _detach_transcript_messages(source_messages),
        defer_interaction_source,
    )


def _validate_interaction_page(
    before_sequence: int | None,
    limit: int,
) -> tuple[int | None, int]:
    if before_sequence is not None and (type(before_sequence) is not int or before_sequence < 1):
        raise ValueError("before_sequence must be a positive integer.")
    if type(limit) is not int or limit < 1 or limit > 5000:
        raise ValueError("limit must be between 1 and 5000.")
    return before_sequence, limit


def copy_session_query(query: SessionQuery | None) -> SessionQuery:
    if query is None:
        return SessionQuery()
    if type(query) is not SessionQuery:
        raise TypeError("Session queries must be SessionQuery instances.")
    return SessionQuery(
        q=query.q,
        status=query.status,
        debug_state=query.debug_state,
        agent_name=query.agent_name,
        provider_name=query.provider_name,
        model=query.model,
        environment_name=query.environment_name,
        parent_session_id=query.parent_session_id,
        causal_budget_id=query.causal_budget_id,
        last_activity_before=query.last_activity_before,
        labels=copy_label_map(query.labels, "labels"),
        label_selectors=copy_label_selector_requirements(query.label_selectors),
        limit=query.limit,
        offset=query.offset,
        cursor=query.cursor,
        include_total_count=query.include_total_count,
        order_by=query.order_by,
    )


def copy_session_aggregate_filter(
    filters: SessionAggregateFilter | None,
) -> SessionAggregateFilter:
    if filters is None:
        return SessionAggregateFilter()
    if type(filters) is not SessionAggregateFilter:
        raise TypeError("Session aggregate filters must be SessionAggregateFilter instances.")
    return SessionAggregateFilter.model_validate(filters.model_dump(mode="python"))


def session_query_from_aggregate_filter(filters: SessionAggregateFilter) -> SessionQuery:
    filters = copy_session_aggregate_filter(filters)
    return SessionQuery(
        agent_name=filters.agent_name,
        provider_name=filters.provider_name,
        model=filters.model,
        environment_name=filters.environment_name,
        parent_session_id=filters.parent_session_id,
        causal_budget_id=filters.causal_budget_id,
        labels=filters.labels,
        label_selectors=filters.label_selectors,
    )


def copy_usage_rollup_query(query: UsageRollupQuery) -> UsageRollupQuery:
    if type(query) is not UsageRollupQuery:
        raise TypeError("Usage aggregate queries must be UsageRollupQuery instances.")
    return UsageRollupQuery.model_validate(query.model_dump(mode="python"))


def copy_event_query(query: EventQuery | None) -> EventQuery:
    if query is None:
        return EventQuery()
    if type(query) is not EventQuery:
        raise TypeError("Event queries must be EventQuery instances.")
    return EventQuery(
        session_id=query.session_id,
        session_ids=query.session_ids,
        event_id=query.event_id,
        interaction_id=query.interaction_id,
        causal_budget_id=query.causal_budget_id,
        event_type=query.event_type,
        event_types=query.event_types,
        exclude_event_types=query.exclude_event_types,
        agent_name=query.agent_name,
        environment_name=query.environment_name,
        workflow_name=query.workflow_name,
        workflow_attempt_id=query.workflow_attempt_id,
        workflow_step_id=query.workflow_step_id,
        workflow_attempt_fenced=query.workflow_attempt_fenced,
        tool_name=query.tool_name,
        since=query.since,
        until=query.until,
        after_sequence=query.after_sequence,
        before_sequence=query.before_sequence,
        limit=query.limit,
        order_by=query.order_by,
    )


def copy_transcript_query(query: TranscriptQuery) -> TranscriptQuery:
    if type(query) is not TranscriptQuery:
        raise TypeError("Transcript queries must be TranscriptQuery instances.")
    return TranscriptQuery(
        session_id=query.session_id,
        interaction_id=query.interaction_id,
        role=query.role,
        offset=query.offset,
        limit=query.limit,
        include_thinking=query.include_thinking,
    )


def _session_matches(session: Session, query: SessionQuery) -> bool:
    if query.q is not None and not _session_query_text_matches(session, query.q):
        return False
    if query.status is not None and session.status != query.status:
        return False
    if query.debug_state is not None:
        raise ValueError("SessionQuery debug_state requires event-aware store filtering.")
    if query.agent_name is not None and session.agent_name != query.agent_name:
        return False
    if query.provider_name is not None and session.provider_name != query.provider_name:
        return False
    if query.model is not None and session.model != query.model:
        return False
    if query.parent_session_id is not None and session.parent_session_id != query.parent_session_id:
        return False
    if query.causal_budget_id is not None and session.causal_budget_id != query.causal_budget_id:
        return False
    if (
        query.last_activity_before is not None
        and session.last_activity_at > query.last_activity_before
    ):
        return False
    for key, value in query.labels.items():
        if session.labels.get(key) != value:
            return False
    for selector in query.label_selectors:
        if not _label_selector_matches(session.labels, selector):
            return False
    return not (
        query.environment_name is not None and session.environment_name != query.environment_name
    )


@dataclass
class _UsageAccumulator:
    session_count: int = 0
    model_steps: int = 0
    model_steps_with_usage: int = 0
    usage: AggregateUsageMetrics = dataclass_field(default_factory=build_aggregate_usage_metrics)

    def add(self, metrics: UsageMetrics | None) -> None:
        self.model_steps += 1
        if metrics is None:
            return
        self.model_steps_with_usage += 1
        self.usage = add_aggregate_usage(self.usage, metrics)

    def merge(self, other: _UsageAccumulator) -> None:
        self.session_count += other.session_count
        self.model_steps += other.model_steps
        self.model_steps_with_usage += other.model_steps_with_usage
        self.usage = add_aggregate_usage(self.usage, other.usage)

    def totals(self, *, tool_calls: int = 0) -> UsageAggregateTotals:
        return UsageAggregateTotals(
            session_count=self.session_count,
            model_steps=self.model_steps,
            model_steps_with_usage=self.model_steps_with_usage,
            tool_calls=tool_calls,
            usage=self.usage,
        )


@dataclass
class _SessionInspectionUsageAccumulator:
    """Single-pass inspection fold with native aggregate usage semantics."""

    totals: _UsageAccumulator = dataclass_field(default_factory=_UsageAccumulator)
    provider_names: list[str] = dataclass_field(default_factory=list)
    models: list[str] = dataclass_field(default_factory=list)
    _provider_names_seen: set[str] = dataclass_field(default_factory=set)
    _models_seen: set[str] = dataclass_field(default_factory=set)
    tool_calls: int = 0

    def add(self, event_type: str, metrics: UsageMetrics | None) -> None:
        if event_type == EventType.TOOL_CALL_STARTED:
            self.tool_calls += 1
            return
        if event_type != EventType.MODEL_COMPLETED:
            return
        self.totals.add(metrics)
        if metrics is None:
            return
        if (
            metrics.provider_name is not None
            and metrics.provider_name not in self._provider_names_seen
        ):
            self._provider_names_seen.add(metrics.provider_name)
            self.provider_names.append(metrics.provider_name)
        if metrics.model is not None and metrics.model not in self._models_seen:
            self._models_seen.add(metrics.model)
            self.models.append(metrics.model)

    def result(self, session_id: str) -> tuple[SessionInspectionUsageSummary, int]:
        return (
            SessionInspectionUsageSummary(
                session_id=session_id,
                model_steps=self.totals.model_steps,
                tool_calls=self.tool_calls,
                provider_names=self.provider_names,
                models=self.models,
                usage=self.totals.usage,
            ),
            self.totals.model_steps_with_usage,
        )


_IN_MEMORY_USAGE_GROUP_CANDIDATE_LIMIT = 512
_UsageGroupKey = tuple[str | None, str | None]
_UsageGroupSortKey = tuple[bool, str, bool, str]
_SessionRecordsFactory = Callable[
    [],
    Iterable[tuple[str, SessionStatus, Iterable[EventRecord]]],
]


@dataclass
class _SessionUsageAccumulator:
    session_id: str
    status: SessionStatus
    usage: _UsageAccumulator = dataclass_field(default_factory=_UsageAccumulator)
    tool_calls: int = 0
    has_activity: bool = False

    def add_event(self, event: Event) -> None:
        if event.type == EventType.TOOL_CALL_STARTED:
            self.tool_calls += 1
            self.has_activity = True
            return
        if event.type != EventType.MODEL_COMPLETED:
            return
        self.has_activity = True
        self.usage.add(aggregate_usage_metrics_from_event_payload(event.payload))

    def candidate(self) -> _SessionUsageCandidate:
        self.usage.session_count = int(self.has_activity)
        return _SessionUsageCandidate(
            session_id=self.session_id,
            status=self.status,
            active=self.status
            in {
                SessionStatus.PENDING,
                SessionStatus.RUNNING,
                SessionStatus.INTERRUPTING,
            },
            totals=self.usage.totals(tool_calls=self.tool_calls),
        )


@dataclass(frozen=True)
class _SessionUsageCandidate:
    """Internal group candidate whose identity may remain in the remainder."""

    session_id: str
    status: SessionStatus
    active: bool
    totals: UsageAggregateTotals

    def retained_group(self) -> UsageSessionAggregateGroup:
        # Only identities crossing the public retained-group contract need its
        # independent byte bound. Omitted candidates contribute counters only.
        require_bounded_usage_session_id(self.session_id)
        return UsageSessionAggregateGroup(
            session_id=self.session_id,
            status=self.status.value,
            active=self.active,
            totals=self.totals,
        )


@dataclass
class _SessionUsageRemainderAccumulator:
    group_count: int = 0
    active_session_count: int = 0
    totals: _UsageAccumulator = dataclass_field(default_factory=_UsageAccumulator)
    tool_calls: int = 0

    def add(self, group: _SessionUsageCandidate) -> None:
        self.group_count += 1
        self.active_session_count += int(group.active)
        group_totals = _UsageAccumulator(
            session_count=group.totals.session_count,
            model_steps=group.totals.model_steps,
            model_steps_with_usage=group.totals.model_steps_with_usage,
            usage=group.totals.usage,
        )
        self.totals.merge(group_totals)
        self.tool_calls += group.totals.tool_calls

    def result(self) -> UsageSessionAggregateRemainder | None:
        if not self.group_count:
            return None
        return UsageSessionAggregateRemainder(
            group_count=self.group_count,
            active_session_count=self.active_session_count,
            totals=self.totals.totals(tool_calls=self.tool_calls),
        )


@dataclass
class _InMemoryUsageGroupCandidates:
    """Bounded heavy-hitter candidates for one in-memory breakdown dimension."""

    limit: int = _IN_MEMORY_USAGE_GROUP_CANDIDATE_LIMIT
    _estimates: dict[_UsageGroupKey, tuple[int, int, int]] = dataclass_field(default_factory=dict)
    _heap: list[tuple[int, int, bool, str, bool, str, int, _UsageGroupKey]] = dataclass_field(
        default_factory=list
    )
    _generation: int = 0
    sampled: bool = False

    def observe(self, key: _UsageGroupKey, metrics: UsageMetrics | None) -> None:
        token_weight = 0 if metrics is None else metrics.total_tokens
        current = self._estimates.get(key)
        if current is not None:
            self._record(key, current[0] + token_weight, current[1] + 1)
            return
        if len(self._estimates) < self.limit:
            self._record(key, token_weight, 1)
            return

        self.sampled = True
        minimum_tokens, minimum_steps, minimum_key = self._pop_minimum()
        del self._estimates[minimum_key]
        self._record(
            key,
            minimum_tokens + token_weight,
            minimum_steps + 1,
        )

    @property
    def keys(self) -> tuple[_UsageGroupKey, ...]:
        return tuple(self._estimates)

    def _record(self, key: _UsageGroupKey, tokens: int, steps: int) -> None:
        self._generation += 1
        generation = self._generation
        self._estimates[key] = tokens, steps, generation
        heapq.heappush(
            self._heap,
            (
                tokens,
                steps,
                *_usage_group_identity_sort_key(key),
                generation,
                key,
            ),
        )
        if len(self._heap) > self.limit * 2:
            self._heap = [
                (
                    item_tokens,
                    item_steps,
                    *_usage_group_identity_sort_key(item_key),
                    item_generation,
                    item_key,
                )
                for item_key, (item_tokens, item_steps, item_generation) in (
                    self._estimates.items()
                )
            ]
            heapq.heapify(self._heap)

    def _pop_minimum(self) -> tuple[int, int, _UsageGroupKey]:
        while self._heap:
            tokens, steps, _, _, _, _, generation, key = heapq.heappop(self._heap)
            if self._estimates.get(key) == (tokens, steps, generation):
                return tokens, steps, key
        raise RuntimeError("In-memory usage candidate heap lost its retained groups.")


def _usage_rollup_from_session_records(
    *,
    session_records: _SessionRecordsFactory,
    query: UsageRollupQuery,
    as_of: datetime,
    matching_session_count: int,
    active_session_count: int,
) -> UsageRollupStoreResult:
    totals = _UsageAccumulator()
    pricing = BoundedUsagePricingInputAccumulator(query.pricing_input_limit)
    session_pricing = BoundedUsagePricingInputAccumulator(query.pricing_input_limit)
    provider_candidates = _InMemoryUsageGroupCandidates()
    model_candidates = _InMemoryUsageGroupCandidates()
    retained_session_groups: list[_SessionUsageCandidate] = []
    session_remainder = _SessionUsageRemainderAccumulator()
    activity_session_count = 0
    tool_calls = 0

    for session_id, status, records in session_records():
        session_has_activity = False
        session_accumulator = (
            None
            if query.session_group_limit is None
            else _SessionUsageAccumulator(session_id=session_id, status=status)
        )
        for record in records:
            event = record.event
            event_timestamp = normalize_aggregate_event_timestamp(event.timestamp)
            if event_timestamp < query.start_at or event_timestamp >= query.end_at:
                continue
            if session_accumulator is not None:
                session_accumulator.add_event(event)
            if event.type == EventType.TOOL_CALL_STARTED:
                session_has_activity = True
                tool_calls += 1
                continue
            if event.type != EventType.MODEL_COMPLETED:
                continue

            session_has_activity = True
            metrics = aggregate_usage_metrics_from_event_payload(event.payload)
            totals.add(metrics)
            provider_candidates.observe(
                _usage_group_key(metrics, dimension="provider"),
                metrics,
            )
            model_candidates.observe(
                _usage_group_key(metrics, dimension="model"),
                metrics,
            )

            if not query.include_pricing_inputs or pricing.truncated:
                continue
            pricing.add_payload(
                effective_on=event_timestamp.date(),
                occurrences=1,
                payload=event.payload,
            )
        if session_has_activity:
            activity_session_count += 1
        if session_accumulator is not None:
            assert query.session_group_limit is not None
            _retain_bounded_session_usage_group(
                retained_session_groups,
                session_remainder,
                session_accumulator.candidate(),
                limit=query.session_group_limit,
            )

    pricing_items, pricing_group_count, pricing_accuracy = pricing.result()
    session_breakdown = _in_memory_session_usage_breakdown(
        retained_session_groups,
        session_remainder,
        limit=query.session_group_limit,
    )
    if query.include_pricing_inputs and session_breakdown is not None and session_breakdown.groups:
        visible_session_ids = {group.session_id for group in session_breakdown.groups}
        for session_id, _, records in session_records():
            if session_id not in visible_session_ids:
                continue
            for record in records:
                event = record.event
                if not _aggregate_model_event_is_in_window(event, query):
                    continue
                session_pricing.add_payload(
                    session_id=session_id,
                    effective_on=normalize_aggregate_event_timestamp(event.timestamp).date(),
                    occurrences=1,
                    payload=event.payload,
                )
                if session_pricing.truncated:
                    break
            if session_pricing.truncated:
                break
    (
        session_pricing_items,
        session_pricing_group_count,
        session_pricing_accuracy,
    ) = session_pricing.result()

    totals.session_count = activity_session_count
    return UsageRollupStoreResult(
        as_of=as_of,
        start_at=query.start_at,
        end_at=query.end_at,
        totals=totals.totals(tool_calls=tool_calls),
        provider_breakdown=_bounded_in_memory_usage_breakdown(
            session_records,
            query=query,
            limit=query.group_limit,
            dimension="provider",
            candidates=provider_candidates,
        ),
        model_breakdown=_bounded_in_memory_usage_breakdown(
            session_records,
            query=query,
            limit=query.group_limit,
            dimension="model",
            candidates=model_candidates,
        ),
        session_breakdown=session_breakdown,
        pricing_inputs=pricing_items,
        pricing_inputs_included=query.include_pricing_inputs,
        pricing_input_group_count=pricing_group_count,
        pricing_inputs_accuracy=pricing_accuracy,
        session_pricing_inputs=session_pricing_items,
        session_pricing_inputs_included=(
            query.include_pricing_inputs and query.session_group_limit is not None
        ),
        session_pricing_input_group_count=session_pricing_group_count,
        session_pricing_inputs_accuracy=session_pricing_accuracy,
        active_session_count=active_session_count,
        matching_session_count=matching_session_count,
    )


def _retain_bounded_session_usage_group(
    retained: list[_SessionUsageCandidate],
    remainder: _SessionUsageRemainderAccumulator,
    group: _SessionUsageCandidate,
    *,
    limit: int,
) -> None:
    if len(retained) < limit:
        retained.append(group)
        return
    worst = max(retained, key=_session_usage_group_rank_key)
    if _session_usage_group_rank_key(group) < _session_usage_group_rank_key(worst):
        retained.remove(worst)
        remainder.add(worst)
        retained.append(group)
        return
    remainder.add(group)


def _in_memory_session_usage_breakdown(
    retained: list[_SessionUsageCandidate],
    remainder: _SessionUsageRemainderAccumulator,
    *,
    limit: int | None,
) -> UsageSessionAggregateBreakdown | None:
    if limit is None:
        return None
    groups = tuple(
        candidate.retained_group()
        for candidate in sorted(retained, key=_session_usage_group_rank_key)
    )
    remainder_result = remainder.result()
    return UsageSessionAggregateBreakdown(
        groups=groups,
        remainder=remainder_result,
        accuracy=(
            EXACT_AGGREGATE.model_copy()
            if remainder_result is None
            else AggregateAccuracy(
                kind=AggregateAccuracyKind.TRUNCATED,
                reason="Matching sessions exceed session_group_limit.",
                limit=limit,
            )
        ),
    )


def _session_usage_group_rank_key(
    group: _SessionUsageCandidate,
) -> tuple[int, int, str]:
    return (
        -group.totals.usage.total_tokens,
        -group.totals.model_steps,
        group.session_id,
    )


def _bounded_in_memory_usage_breakdown(
    session_records: _SessionRecordsFactory,
    *,
    query: UsageRollupQuery,
    limit: int,
    dimension: Literal["provider", "model"],
    candidates: _InMemoryUsageGroupCandidates,
) -> UsageAggregateBreakdown:
    accumulators = _accumulate_usage_group_batch(
        session_records,
        query=query,
        dimension=dimension,
        keys=candidates.keys,
    )
    visible_items = sorted(accumulators.items(), key=_usage_group_rank_key)[:limit]

    visible = tuple(
        UsageAggregateGroup(
            provider_name=key[0],
            model=key[1],
            totals=accumulator.totals(),
        )
        for key, accumulator in visible_items
    )
    if candidates.sampled:
        return UsageAggregateBreakdown(
            groups=visible,
            remainder=None,
            accuracy=AggregateAccuracy(
                kind=AggregateAccuracyKind.SAMPLED,
                reason=(
                    f"Distinct {dimension} groups exceed the bounded in-memory "
                    "heavy-hitter candidate limit."
                ),
                limit=candidates.limit,
            ),
        )

    group_count = len(candidates.keys)
    if group_count <= limit:
        return UsageAggregateBreakdown(
            groups=visible,
            remainder=None,
            accuracy=EXACT_AGGREGATE.model_copy(),
        )

    remainder = _accumulate_usage_remainder(
        session_records,
        query=query,
        dimension=dimension,
        visible_keys={(group.provider_name, group.model) for group in visible},
    )
    return UsageAggregateBreakdown(
        groups=visible,
        remainder=UsageAggregateRemainder(
            group_count=group_count - len(visible),
            totals=remainder.totals(),
        ),
        accuracy=AggregateAccuracy(
            kind=AggregateAccuracyKind.TRUNCATED,
            reason=f"Distinct {dimension} groups exceed group_limit.",
            limit=limit,
        ),
    )


def _accumulate_usage_group_batch(
    session_records: _SessionRecordsFactory,
    *,
    query: UsageRollupQuery,
    dimension: Literal["provider", "model"],
    keys: tuple[_UsageGroupKey, ...],
) -> dict[_UsageGroupKey, _UsageAccumulator]:
    accumulators = {key: _UsageAccumulator() for key in keys}
    for _, _, records in session_records():
        seen: set[_UsageGroupKey] = set()
        for record in records:
            event = record.event
            if not _aggregate_model_event_is_in_window(event, query):
                continue
            metrics = aggregate_usage_metrics_from_event_payload(event.payload)
            key = _usage_group_key(metrics, dimension=dimension)
            accumulator = accumulators.get(key)
            if accumulator is None:
                continue
            accumulator.add(metrics)
            seen.add(key)
        for key in seen:
            accumulators[key].session_count += 1
    return accumulators


def _accumulate_usage_remainder(
    session_records: _SessionRecordsFactory,
    *,
    query: UsageRollupQuery,
    dimension: Literal["provider", "model"],
    visible_keys: set[_UsageGroupKey],
) -> _UsageAccumulator:
    remainder = _UsageAccumulator()
    for _, _, records in session_records():
        session_has_remainder = False
        for record in records:
            event = record.event
            if not _aggregate_model_event_is_in_window(event, query):
                continue
            metrics = aggregate_usage_metrics_from_event_payload(event.payload)
            if _usage_group_key(metrics, dimension=dimension) in visible_keys:
                continue
            remainder.add(metrics)
            session_has_remainder = True
        remainder.session_count += session_has_remainder
    return remainder


def _aggregate_model_event_is_in_window(event: Event, query: UsageRollupQuery) -> bool:
    if event.type != EventType.MODEL_COMPLETED:
        return False
    timestamp = normalize_aggregate_event_timestamp(event.timestamp)
    return query.start_at <= timestamp < query.end_at


def _usage_group_key(
    metrics: UsageMetrics | None,
    *,
    dimension: Literal["provider", "model"],
) -> _UsageGroupKey:
    provider_name = None if metrics is None else metrics.provider_name
    model = None if metrics is None or dimension == "provider" else metrics.model
    return provider_name, model


def _usage_group_identity_sort_key(key: _UsageGroupKey) -> _UsageGroupSortKey:
    return key[0] is None, key[0] or "", key[1] is None, key[1] or ""


def _usage_group_rank_key(
    item: tuple[_UsageGroupKey, _UsageAccumulator],
) -> tuple[int, int, bool, str, bool, str]:
    key, accumulator = item
    return (
        -accumulator.usage.total_tokens,
        -accumulator.model_steps,
        *_usage_group_identity_sort_key(key),
    )


def _session_matches_debug_state(
    session: Session,
    records: list[EventRecord],
    debug_state: SessionDebugState | None,
) -> bool:
    if debug_state is None:
        return True
    has_tool_debug_event = any(_is_tool_debug_event(record.event.type) for record in records)
    has_session_failure = session.status == SessionStatus.FAILED
    has_interruption = session.status == SessionStatus.INTERRUPTED
    if debug_state == SessionDebugState.TOOL_ISSUE:
        return has_tool_debug_event
    if debug_state == SessionDebugState.SESSION_FAILURE:
        return has_session_failure
    if debug_state == SessionDebugState.INTERRUPTION:
        return has_interruption
    if debug_state == SessionDebugState.NEEDS_ATTENTION:
        return has_session_failure or has_interruption or has_tool_debug_event
    return False


def _is_tool_debug_event(event_type: EventType | str) -> bool:
    return event_type in {EventType.TOOL_CALL_FAILED, EventType.TOOL_CALL_BLOCKED}


def _session_query_text_matches(session: Session, query: str) -> bool:
    needle = query.casefold()
    haystacks = [
        session.id,
        session.agent_name,
        session.provider_name,
        session.model,
        session.environment_name,
        session.parent_session_id,
        session.causal_budget_id,
        *session.labels.keys(),
        *session.labels.values(),
    ]
    return any(needle in value.casefold() for value in haystacks if type(value) is str and value)


def _label_selector_matches(
    labels: dict[str, str],
    selector: LabelSelectorRequirement,
) -> bool:
    value = labels.get(selector.key)
    if selector.operator == LabelSelectorOperator.EXISTS:
        return value is not None
    if selector.operator == LabelSelectorOperator.NOT_EXISTS:
        return value is None
    if selector.operator == LabelSelectorOperator.IN:
        return value in selector.values
    if selector.operator == LabelSelectorOperator.NOT_IN:
        return value is None or value not in selector.values
    raise ValueError(f"Unsupported label selector operator: {selector.operator}")


def _sort_sessions(sessions: list[Session], order_by: SessionOrder) -> list[Session]:
    if order_by == SessionOrder.CREATED_AT_ASC:
        return sorted(sessions, key=lambda session: (session.created_at, session.id))
    if order_by == SessionOrder.CREATED_AT_DESC:
        return sorted(
            sorted(sessions, key=lambda session: session.id),
            key=lambda session: session.created_at,
            reverse=True,
        )
    if order_by == SessionOrder.UPDATED_AT_ASC:
        return sorted(sessions, key=lambda session: (session.updated_at, session.id))
    if order_by == SessionOrder.UPDATED_AT_DESC:
        return sorted(
            sorted(sessions, key=lambda session: session.id),
            key=lambda session: session.updated_at,
            reverse=True,
        )
    if order_by == SessionOrder.LAST_ACTIVITY_AT_ASC:
        return sorted(sessions, key=lambda session: (session.last_activity_at, session.id))
    return sorted(
        sorted(sessions, key=lambda session: session.id),
        key=lambda session: session.last_activity_at,
        reverse=True,
    )


# Sessions that must be interrupted before they can be deleted (in-flight work).
DELETE_BLOCKED_SESSION_STATUSES = frozenset({SessionStatus.RUNNING, SessionStatus.INTERRUPTING})
_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY = "incomplete_session_recovery_claim"
INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY = "initial_transcript_pending"
_TERMINAL_PUBLICATION_EVENT_TYPE_BY_STATUS = {
    SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
    SessionStatus.FAILED: EventType.SESSION_FAILED,
    SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
}
# Storage implementations import these aliases while collecting the same
# bounded evidence consumed by the shared runtime classifier.
_TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES = TERMINAL_EVIDENCE_EVENT_TYPES
_TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT = TERMINAL_EVIDENCE_QUERY_LIMIT
_TERMINAL_PENDING_ACTION_CHECKPOINT_KEYS = frozenset(
    {
        "pending_tool_approval",
        "pending_user_input",
        _PENDING_TOOL_ROUND_CHECKPOINT_KEY,
    }
)


def _terminal_publication_delete_block_reason(
    *,
    session: Session,
    checkpoint: dict[str, Any] | None,
    evidence_events: Sequence[Event],
) -> str | None:
    """Return why deletion would erase repairable current-run terminal evidence.

    ``evidence_events`` must contain the newest lifecycle/terminal events in
    descending session-sequence order, bounded by
    ``_TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT``.
    """

    expected_event_type = _TERMINAL_PUBLICATION_EVENT_TYPE_BY_STATUS.get(session.status)
    if expected_event_type is None:
        return None
    if len(evidence_events) > _TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT:
        raise ValueError("Terminal publication deletion evidence exceeds its bounded query.")

    if checkpoint is not None and "pending_session_interrupt" in checkpoint:
        return "pending interruption terminal publication is incomplete"

    run_operation = _session_run_operation_from_checkpoint(checkpoint)
    classification = classify_current_terminal_evidence(
        evidence_events=evidence_events,
        expected_event_type=expected_event_type,
        run_operation_id=(None if run_operation is None else run_operation.operation_id),
        interruption_request_id=None,
    )
    terminal_events = classification.events
    if classification.run_operation_conflict:
        return "terminal publication evidence contradicts the pending run operation"

    if any(event.type != expected_event_type for event in terminal_events):
        return "terminal publication evidence contradicts the durable session status"
    if len(terminal_events) > 1:
        return "terminal publication contains duplicate current-run evidence"

    pending_action = checkpoint is not None and any(
        key in checkpoint for key in _TERMINAL_PENDING_ACTION_CHECKPOINT_KEYS
    )
    terminal_event_required = (
        run_operation is not None
        or pending_action
        or classification.latest_lifecycle_event_type != EventType.SESSION_FORKED
    )
    if terminal_event_required and not terminal_events:
        return "terminal publication evidence is incomplete"
    return None


def _initial_transcript_pending_checkpoint(interaction_id: str) -> dict[str, Any]:
    """Create the authority marker held until initial transcript publication."""

    return {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY: {
            "version": 1,
            "interaction_id": require_clean_nonblank(interaction_id, "interaction_id"),
        },
    }


def _initial_transcript_pending_interaction_id(
    checkpoint: dict[str, Any] | None,
) -> str | None:
    if checkpoint is None or INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY not in checkpoint:
        return None
    marker = checkpoint[INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY]
    if type(marker) is not dict or marker.get("version") != 1:
        raise ValueError("Initial transcript authority marker is invalid.")
    interaction_id = marker.get("interaction_id")
    if type(interaction_id) is not str:
        raise ValueError("Initial transcript authority marker is invalid.")
    return require_clean_nonblank(interaction_id, "interaction_id")


def _checkpoint_after_initial_transcript_publication(
    checkpoint: dict[str, Any] | None,
    *,
    interaction_id: str,
) -> dict[str, Any] | None:
    pending_interaction_id = _initial_transcript_pending_interaction_id(checkpoint)
    if pending_interaction_id is None:
        # Deferred admission is also used after session creation, when there is
        # no environment-derived bootstrap authority to protect.
        return checkpoint
    if pending_interaction_id != interaction_id:
        raise RuntimeError("Initial transcript authority changed before finalization.")
    assert checkpoint is not None
    updated = copy_durable_json_object(checkpoint, "checkpoint")
    updated.pop(INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY)
    return updated or None


_SESSION_RUN_OPERATION_CHECKPOINT_KEY = "session_run_operation"
_SESSION_RUN_OPERATION_ID_PAYLOAD_KEY = SESSION_RUN_OPERATION_ID_PAYLOAD_KEY


@dataclass(frozen=True)
class _SessionRunOperation:
    operation_id: str
    run_epoch: int


def _session_run_operation_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> _SessionRunOperation | None:
    """Parse the durable identity of the latest resumed/continued session run."""
    if checkpoint is None or _SESSION_RUN_OPERATION_CHECKPOINT_KEY not in checkpoint:
        return None
    marker = checkpoint[_SESSION_RUN_OPERATION_CHECKPOINT_KEY]
    if type(marker) is not dict or marker.get("version") != 1:
        raise ValueError("Session run operation checkpoint is invalid.")
    operation_id = require_clean_nonblank(
        marker.get("operation_id"),
        "session_run_operation.operation_id",
    )
    run_epoch = marker.get("run_epoch")
    if type(run_epoch) is not int or run_epoch < 1 or run_epoch > MAX_DURABLE_JSON_INTEGER:
        raise ValueError("Session run operation checkpoint run_epoch is invalid.")
    return _SessionRunOperation(
        operation_id=operation_id,
        run_epoch=run_epoch,
    )


def _checkpoint_with_session_run_operation(
    *,
    checkpoint: dict[str, Any] | None,
    current_session: Session,
    operation_id: str,
) -> dict[str, Any]:
    """Record the next running epoch in the same transaction as its status claim."""
    operation_id = require_clean_nonblank(operation_id, "session run operation_id")
    existing_operation = _session_run_operation_from_checkpoint(checkpoint)
    if existing_operation is not None:
        raise RuntimeError(
            "Session has incomplete terminal evidence for its previous run. "
            "Recover the session before starting another run."
        )
    next_run_epoch = current_session.run_epoch + 1
    if next_run_epoch > MAX_DURABLE_JSON_INTEGER:
        raise ValueError("Session run operation run_epoch exceeds the durable integer limit.")
    updated = {} if checkpoint is None else copy_durable_json_object(checkpoint, "checkpoint")
    updated[_SESSION_RUN_OPERATION_CHECKPOINT_KEY] = {
        "version": 1,
        "operation_id": operation_id,
        "run_epoch": next_run_epoch,
    }
    return updated


def _event_with_session_run_operation(
    event: Event,
    operation: _SessionRunOperation,
) -> Event:
    """Bind terminal evidence to the durable run operation that produced it."""
    payload = copy_durable_json_object(event.payload, "event payload")
    existing_operation_id = payload.get(_SESSION_RUN_OPERATION_ID_PAYLOAD_KEY)
    if existing_operation_id not in {None, operation.operation_id}:
        raise ValueError("Terminal event carries a conflicting session run operation identity.")
    payload[_SESSION_RUN_OPERATION_ID_PAYLOAD_KEY] = operation.operation_id
    bound = event.model_copy(update={"payload": payload}, deep=True)
    return event_with_runtime_payload_authority(
        bound,
        _SESSION_RUN_OPERATION_ID_PAYLOAD_KEY,
    )


def _incomplete_recovery_claim_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> tuple[str, datetime] | None:
    """Parse the runtime-owned incomplete-session recovery lease."""
    if checkpoint is None or _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY not in checkpoint:
        return None
    marker = checkpoint[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY]
    if type(marker) is not dict or marker.get("version") != 1:
        raise ValueError("Incomplete-session recovery claim checkpoint is invalid.")
    claim_id = require_clean_nonblank(marker.get("claim_id"), "recovery_claim.claim_id")
    claimed_at_value = require_clean_nonblank(
        marker.get("claimed_at"),
        "recovery_claim.claimed_at",
    )
    expires_at_value = require_clean_nonblank(
        marker.get("claim_expires_at"),
        "recovery_claim.claim_expires_at",
    )
    try:
        claimed_at = datetime.fromisoformat(claimed_at_value)
        expires_at = datetime.fromisoformat(expires_at_value)
    except ValueError as exc:
        raise ValueError("Incomplete-session recovery claim timestamps are invalid.") from exc
    if (
        claimed_at.tzinfo is None
        or claimed_at.utcoffset() is None
        or expires_at.tzinfo is None
        or expires_at.utcoffset() is None
        or expires_at <= claimed_at
    ):
        raise ValueError("Incomplete-session recovery claim timestamps are invalid.")
    return claim_id, expires_at


def _active_unexpired_incomplete_recovery_claim_id(
    checkpoint: dict[str, Any] | None,
    *,
    now: datetime,
) -> str | None:
    """Return the durable recovery owner while its lease remains live."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware.")
    claim = _incomplete_recovery_claim_from_checkpoint(checkpoint)
    if claim is None:
        return None
    claim_id, expires_at = claim
    return claim_id if expires_at.astimezone(UTC) > now.astimezone(UTC) else None


def _active_unexpired_session_operation_id(
    checkpoint: dict[str, Any] | None,
    *,
    now: datetime,
) -> str | None:
    """Return the active durable operation while its claim lease is valid."""

    if checkpoint is None:
        return None
    stored = checkpoint.get("session_operations")
    if stored is None:
        return None
    if type(stored) is not dict:
        raise ValueError("Session operation checkpoint must be an object.")
    active_operation_id = stored.get("active_operation_id")
    if active_operation_id is None:
        return None
    active_operation_id = require_clean_nonblank(
        active_operation_id,
        "active_operation_id",
    )
    records = stored.get("records")
    if type(records) is not dict:
        raise ValueError("Session operation checkpoint records must be an object.")
    active_record = next(
        (
            record
            for record in records.values()
            if type(record) is dict and record.get("operation_id") == active_operation_id
        ),
        None,
    )
    if active_record is None:
        raise ValueError("Active durable session operation record is missing.")
    claim_expires_at = active_record.get("claim_expires_at")
    if type(claim_expires_at) is not str:
        return active_operation_id
    try:
        expiry = datetime.fromisoformat(claim_expires_at)
    except ValueError:
        return active_operation_id
    if expiry.tzinfo is None or expiry.utcoffset() is None:
        return active_operation_id
    return active_operation_id if expiry.astimezone(UTC) > now.astimezone(UTC) else None


_DESCENDING_SESSION_ORDERS = frozenset(
    {
        SessionOrder.CREATED_AT_DESC,
        SessionOrder.UPDATED_AT_DESC,
        SessionOrder.LAST_ACTIVITY_AT_DESC,
    }
)
_CREATED_AT_ORDERS = frozenset({SessionOrder.CREATED_AT_ASC, SessionOrder.CREATED_AT_DESC})
_LAST_ACTIVITY_AT_ORDERS = frozenset(
    {SessionOrder.LAST_ACTIVITY_AT_ASC, SessionOrder.LAST_ACTIVITY_AT_DESC}
)


def session_order_is_descending(order_by: SessionOrder) -> bool:
    return order_by in _DESCENDING_SESSION_ORDERS


def session_sort_column(order_by: SessionOrder) -> str:
    """The session column an order sorts by — the keyset cursor's primary key."""
    if order_by in _CREATED_AT_ORDERS:
        return "created_at"
    if order_by in _LAST_ACTIVITY_AT_ORDERS:
        return "last_activity_at"
    return "updated_at"


def _session_sort_value(
    session: Session | PendingActionSession,
    order_by: SessionOrder,
) -> datetime:
    if order_by in _CREATED_AT_ORDERS:
        return session.created_at
    if order_by in _LAST_ACTIVITY_AT_ORDERS:
        return session.last_activity_at if isinstance(session, Session) else session.updated_at
    return session.updated_at


def encode_session_cursor(
    session: Session | PendingActionSession,
    order_by: SessionOrder,
) -> str:
    """Opaque keyset cursor for the last row of a page: (sort value, session id)."""
    sort_value = _session_sort_value(session, order_by).astimezone(UTC).isoformat()
    session_id = _require_bounded_session_id(session.id, "session.id")
    material = {
        "version": _SESSION_CURSOR_VERSION,
        "sort_value": sort_value,
        "session_id_b64": base64.urlsafe_b64encode(session_id.encode("utf-8")).decode("ascii"),
    }
    encoded = base64.urlsafe_b64encode(
        canonical_durable_json_bytes(material, "session cursor")
    ).decode("ascii")
    return _require_bounded_session_list_cursor(encoded, "session cursor")


def decode_session_cursor(cursor: str) -> tuple[datetime, str]:
    """Decode a cursor to (sort value, session id). Raises ValueError if malformed."""
    try:
        cursor = _require_bounded_session_list_cursor(cursor, "cursor")
        encoded = cursor.encode("ascii")
        raw = base64.b64decode(encoded, altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(raw) != encoded:
            raise ValueError("Non-canonical session cursor.")
        decoded = json.loads(raw.decode("utf-8"))
        if (
            type(decoded) is not dict
            or set(decoded) != {"version", "sort_value", "session_id_b64"}
            or type(decoded["version"]) is not int
            or decoded["version"] != _SESSION_CURSOR_VERSION
            or type(decoded["sort_value"]) is not str
            or type(decoded["session_id_b64"]) is not str
        ):
            raise ValueError("Invalid session cursor material.")
        encoded_session_id = decoded["session_id_b64"].encode("ascii")
        session_id_bytes = base64.b64decode(
            encoded_session_id,
            altchars=b"-_",
            validate=True,
        )
        if base64.urlsafe_b64encode(session_id_bytes) != encoded_session_id:
            raise ValueError("Non-canonical session identifier.")
        session_id = _require_bounded_session_id(
            session_id_bytes.decode("utf-8"),
            "session cursor id",
        )
    except (
        binascii.Error,
        UnicodeError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError("Invalid session cursor.") from exc
    try:
        sort_value = datetime.fromisoformat(decoded["sort_value"])
    except ValueError as exc:
        raise ValueError("Invalid session cursor.") from exc
    # Sort values are always encoded as UTC-aware timestamps; a naive datetime is
    # a malformed/forged cursor that would raise TypeError when later compared
    # against the timezone-aware session timestamps.
    if sort_value.tzinfo is None:
        raise ValueError("Invalid session cursor.")
    return sort_value, session_id


def encode_session_topology_cursor(
    parent_session_id: str,
    node: SessionTopologyNode,
) -> str:
    """Encode a parent-bound direct-child cursor."""

    parent_session_id = _bounded_session_topology_text(
        parent_session_id,
        "parent_session_id",
        max_bytes=SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES,
    )
    if type(node) is not SessionTopologyNode:
        raise TypeError("Session topology cursors require SessionTopologyNode values.")
    raw = json.dumps(
        [
            parent_session_id,
            node.created_at.astimezone(UTC).isoformat(),
            node.id,
        ],
        separators=(",", ":"),
    )
    encoded = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")
    if len(encoded) > SESSION_TOPOLOGY_MAX_CURSOR_BYTES:
        raise ValueError("Session topology cursor exceeds its byte limit.")
    return encoded


def decode_session_topology_cursor(
    cursor: str,
    *,
    parent_session_id: str,
) -> tuple[datetime, str]:
    """Decode a cursor and reject reuse against a different parent branch."""

    parent_session_id = _bounded_session_topology_text(
        parent_session_id,
        "parent_session_id",
        max_bytes=SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES,
    )
    try:
        cursor = _bounded_session_topology_text(
            cursor,
            "cursor",
            max_bytes=SESSION_TOPOLOGY_MAX_CURSOR_BYTES,
        )
        encoded = cursor.encode("ascii")
        raw = base64.b64decode(encoded, altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(raw) != encoded:
            raise ValueError("Non-canonical session topology cursor.")
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ValueError("Invalid session topology cursor.") from exc
    if (
        type(decoded) is not list
        or len(decoded) != 3
        or type(decoded[0]) is not str
        or type(decoded[1]) is not str
        or type(decoded[2]) is not str
        or decoded[0] != parent_session_id
        or not decoded[2]
    ):
        raise ValueError("Invalid session topology cursor.")
    try:
        child_session_id = require_clean_nonblank(decoded[2], "cursor child_session_id")
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid session topology cursor.") from exc
    try:
        created_at = datetime.fromisoformat(decoded[1])
    except ValueError as exc:
        raise ValueError("Invalid session topology cursor.") from exc
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("Invalid session topology cursor.")
    return created_at.astimezone(UTC), child_session_id


def build_session_topology_result(
    *,
    focus: SessionTopologyNode,
    ancestors: Iterable[SessionTopologyNode],
    expanded_parents: Iterable[SessionTopologyNode],
    branch_candidates: Iterable[Iterable[SessionTopologyNode]],
    child_limit: int,
) -> SessionTopologyStoreResult:
    """Apply the shared response-node ceiling without losing branch continuation."""

    if type(focus) is not SessionTopologyNode:
        raise TypeError("focus must be a SessionTopologyNode.")
    if (
        type(child_limit) is not int
        or child_limit < 1
        or child_limit > SESSION_TOPOLOGY_MAX_CHILD_LIMIT
    ):
        raise ValueError("child_limit is outside the session topology bounds.")
    ancestor_nodes = tuple(islice(ancestors, SESSION_TOPOLOGY_MAX_ANCESTOR_DEPTH + 1))
    if len(ancestor_nodes) > SESSION_TOPOLOGY_MAX_ANCESTOR_DEPTH:
        raise ValueError("Session topology exceeds the ancestor-node bound.")
    expanded_nodes = tuple(islice(expanded_parents, SESSION_TOPOLOGY_MAX_EXPANDED_PARENTS + 1))
    if len(expanded_nodes) > SESSION_TOPOLOGY_MAX_EXPANDED_PARENTS:
        raise ValueError("Session topology exceeds the expanded-parent bound.")
    candidate_pages = tuple(
        tuple(islice(page, child_limit + 1))
        for page in islice(branch_candidates, len(expanded_nodes) + 1)
    )
    if len(expanded_nodes) != len(candidate_pages):
        raise ValueError("Every expanded parent requires one candidate page.")

    retained_ids = {
        focus.id,
        *(node.id for node in ancestor_nodes),
        *(node.id for node in expanded_nodes),
    }
    nonempty_after = [
        sum(bool(later) for later in candidate_pages[index + 1 :])
        for index in range(len(candidate_pages))
    ]
    branches: list[SessionTopologyBranch] = []
    for parent, candidates, remaining_nonempty in zip(
        expanded_nodes,
        candidate_pages,
        nonempty_after,
        strict=True,
    ):
        available = SESSION_TOPOLOGY_MAX_NODES - len(retained_ids)
        branch_capacity = max(0, available - remaining_nonempty)
        retained = candidates[: min(child_limit, branch_capacity)]
        if candidates and not retained:
            raise RuntimeError("Session topology node allocation could not retain a branch cursor.")
        retained_ids.update(child.id for child in retained)
        has_more = len(candidates) > len(retained)
        branches.append(
            SessionTopologyBranch(
                parent_session_id=parent.id,
                children=retained,
                next_cursor=(
                    encode_session_topology_cursor(parent.id, retained[-1]) if has_more else None
                ),
                has_more=has_more,
            )
        )
    loaded_nodes = (
        focus,
        *ancestor_nodes,
        *expanded_nodes,
        *(child for branch in branches for child in branch.children),
    )
    loaded_nodes_by_id: dict[str, SessionTopologyNode] = {}
    for node in loaded_nodes:
        loaded_nodes_by_id.setdefault(node.id, node)
    _reject_loaded_session_topology_cycles(loaded_nodes_by_id)
    return SessionTopologyStoreResult(
        focus=focus,
        ancestors=ancestor_nodes,
        expanded_parents=expanded_nodes,
        branches=tuple(branches),
    )


def _reject_loaded_session_topology_cycles(
    nodes_by_id: Mapping[str, SessionTopologyNode],
) -> None:
    """Reject every parent cycle visible in the returned bounded projection."""

    complete: set[str] = set()
    for start_id in nodes_by_id:
        if start_id in complete:
            continue
        path: list[str] = []
        path_positions: dict[str, int] = {}
        current_id: str | None = start_id
        while current_id is not None and current_id in nodes_by_id:
            if current_id in complete:
                break
            if current_id in path_positions:
                raise SessionTopologyCycle(
                    "Session topology contains a cycle among loaded session nodes."
                )
            path_positions[current_id] = len(path)
            path.append(current_id)
            current_id = nodes_by_id[current_id].parent_session_id
        complete.update(path)


def session_next_cursor(page: list[Session], has_more: bool, order_by: SessionOrder) -> str | None:
    """The keyset cursor for the next page: the last row's cursor, or None if no more."""
    return encode_session_cursor(page[-1], order_by) if has_more and page else None


def _session_after_cursor(
    session: Session,
    order_by: SessionOrder,
    cursor_value: datetime,
    cursor_id: str,
) -> bool:
    """Whether ``session`` falls strictly after the cursor under ``order_by`` (id tiebreak ASC)."""
    value = _session_sort_value(session, order_by)
    if value != cursor_value:
        if session_order_is_descending(order_by):
            return value < cursor_value
        return value > cursor_value
    return session.id > cursor_id


def _event_record_matches(
    record: EventRecord,
    query: EventQuery,
    event_types: frozenset[str],
    excluded_event_types: frozenset[str],
) -> bool:
    event = record.event
    if query.after_sequence is not None and record.sequence <= query.after_sequence:
        return False
    if query.before_sequence is not None and record.sequence >= query.before_sequence:
        return False
    if query.session_id is not None and event.session_id != query.session_id:
        return False
    if query.session_ids and event.session_id not in query.session_ids:
        return False
    if query.event_id is not None and event.id != query.event_id:
        return False
    if query.interaction_id is not None and event.interaction_id != query.interaction_id:
        return False
    event_timestamp = event.timestamp.astimezone(UTC)
    if query.since is not None and event_timestamp < query.since:
        return False
    if query.until is not None and event_timestamp >= query.until:
        return False
    if event_types and str(event.type) not in event_types:
        return False
    if str(event.type) in excluded_event_types:
        return False
    if query.agent_name is not None and event.agent_name != query.agent_name:
        return False
    if query.environment_name is not None and event.environment_name != query.environment_name:
        return False
    if query.workflow_name is not None and event.workflow_name != query.workflow_name:
        return False
    if (
        query.workflow_attempt_id is not None
        and event.payload.get("attempt_id") != query.workflow_attempt_id
    ):
        return False
    if (
        query.workflow_step_id is not None
        and event.payload.get("step_id") != query.workflow_step_id
    ):
        return False
    return not (query.tool_name is not None and event.tool_name != query.tool_name)


def _event_record_matches_session(
    record: EventRecord,
    query: EventQuery,
    sessions: dict[str, Session],
) -> bool:
    if query.causal_budget_id is None:
        return True
    session = sessions.get(record.event.session_id)
    return session is not None and session.causal_budget_id == query.causal_budget_id
