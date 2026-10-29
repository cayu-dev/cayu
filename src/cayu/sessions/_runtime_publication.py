"""Runtime-publication values, bounded event references and content identities."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import MAX_DURABLE_JSON_INTEGER, copy_durable_json_object
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import EVENT_ID_MAX_CHARS, Event, copy_event
from cayu.messages import Message, detach_message
from cayu.sessions import _checkpoint_mutations as checkpoint_mutation_rules
from cayu.sessions import _operation_records as operation_record_rules
from cayu.sessions import authority as session_authority_rules
from cayu.sessions import event_delivery as session_event_rules
from cayu.sessions._argument_continuity import ArgumentContinuity
from cayu.sessions._checkpoint_mutations import (
    RUNTIME_PUBLICATION_MAX_CHECKPOINT_OPERATIONS,
    RuntimePublicationMutation,
)
from cayu.sessions._operation_records import (
    RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX,
    RuntimePublicationOperationRecordMutation,
    SessionOperationPublication,
)
from cayu.sessions.records import Session, SessionStatus

RUNTIME_PUBLICATION_RECORD_TYPE = "cayu.runtime-publication"


RUNTIME_PUBLICATION_SCHEMA_VERSION = 2


RUNTIME_PUBLICATION_MAX_TRANSCRIPT_MESSAGES = 512


RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS = 512


RuntimePublicationKind = Literal[
    "model-step",
    "context-compaction",
    "auxiliary-inference",
    "tool-round",
    "approval-open",
    "approval-close",
    "user-input-open",
    "user-input-close",
    "wake-open",
    "wake-close",
    "workspace-observation",
]


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


class RuntimePublicationRequest(BaseModel):
    """Immutable input bundle for one logical model-step or tool-round publication."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    publication_id: str = Field(max_length=256)
    kind: RuntimePublicationKind
    interaction_id: str | None = Field(default=None, max_length=256)
    intent: dict[str, Any]
    mutation: RuntimePublicationMutation
    transcript_messages: tuple[Message, ...]
    events: tuple[Event, ...]
    argument_continuity: ArgumentContinuity | None = Field(default=None, repr=False)
    operation_record_mutations: tuple[RuntimePublicationOperationRecordMutation, ...] = Field(
        default_factory=tuple
    )
    referenced_events: tuple[RuntimePublicationEventReference, ...] = Field(default_factory=tuple)

    @field_validator("publication_id", "kind")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("argument_continuity", mode="before")
    @classmethod
    def copy_argument_continuity(
        cls, value: ArgumentContinuity | dict[str, Any] | None
    ) -> ArgumentContinuity | None:
        if value is None:
            return None
        if type(value) is ArgumentContinuity:
            value = {
                "nonce": value.nonce,
                "profile": value.profile,
                "scope": value.scope,
                "arguments": value.arguments,
            }
        return ArgumentContinuity.model_validate(value)

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
        return checkpoint_mutation_rules._copy_runtime_publication_mutation_value(value)

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
            # Runtime publications are a separate atomic persistence path from
            # append_events(). Apply the same origin-lineage trust boundary here
            # before the request and event digests are derived. The publication
            # preparation path reconstructs this model, so this also sanitizes
            # validator-bypassed model_copy/model_construct inputs before commit.
            copied_event = session_event_rules._copy_event_for_session_store(event)
            if copied_event.id in seen:
                raise ValueError(f"Runtime publication event id is duplicated: {copied_event.id}")
            seen.add(copied_event.id)
            copied.append(copied_event)
        return tuple(copied)

    @field_validator("operation_record_mutations", mode="before")
    @classmethod
    def copy_operation_record_mutations(
        cls,
        value: object,
    ) -> tuple[RuntimePublicationOperationRecordMutation, ...]:
        if type(value) not in (list, tuple):
            raise TypeError("operation_record_mutations must be a list or tuple.")
        mutation_values = cast(
            "operation_record_rules._RuntimePublicationOperationRecordMutationValues", value
        )
        if len(mutation_values) > RUNTIME_PUBLICATION_MAX_CHECKPOINT_OPERATIONS:
            raise ValueError(
                "operation_record_mutations cannot exceed the runtime publication bound."
            )
        copied = tuple(
            sorted(
                (
                    operation_record_rules._copy_runtime_publication_operation_record_mutation(
                        mutation
                    )
                    for mutation in mutation_values
                ),
                key=lambda mutation: mutation.key,
            )
        )
        keys = tuple(mutation.key for mutation in copied)
        if len(keys) != len(set(keys)):
            raise ValueError("operation_record_mutations cannot repeat a key.")
        return copied

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
        event_binding_limit = _runtime_publication_event_limit(self.kind, self.intent)
        if event_binding_count > event_binding_limit:
            raise ValueError(
                "events and referenced_events cannot contain more than "
                f"{event_binding_limit} total event bindings."
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
        if len(self.appended_event_ids) + len(
            self.referenced_events
        ) > _runtime_publication_event_limit(self.kind, self.intent):
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


def _runtime_publication_storage_key(publication_id: str) -> str:
    publication_id = require_clean_nonblank(publication_id, "publication_id")
    if len(publication_id) > 256:
        raise ValueError("publication_id must contain at most 256 characters.")
    return (
        RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX
        + sha256(publication_id.encode("utf-8")).hexdigest()
    )


def _runtime_publication_event_digest(event: Event) -> str:
    copied_event = copy_event(event)
    return session_authority_rules._canonical_runtime_publication_digest(
        copied_event.model_dump(mode="json")
    )


def runtime_publication_event_reference(event: Event) -> RuntimePublicationEventReference:
    """Return an immutable ID-and-content reference for one exact event value."""

    copied_event = copy_event(event)
    return RuntimePublicationEventReference(
        event_id=copied_event.id,
        event_digest=session_authority_rules._canonical_runtime_publication_digest(
            copied_event.model_dump(mode="json")
        ),
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


def _tool_round_lifecycle_event_limit(tool_call_ids: list[str] | tuple[str, ...]) -> int:
    # At most one started and one terminal event per admitted call. Retain the
    # historical allowance for small/legacy lookups; semantic validation still
    # rejects duplicate or contradictory lifecycle evidence.
    return max(RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS, 2 * len(tool_call_ids))


def _runtime_publication_event_limit(kind: str, intent: dict[str, Any]) -> int:
    if kind in {"tool-round", "user-input-close"}:
        call_ids = intent.get("tool_call_ids")
        if type(call_ids) is list or type(call_ids) is tuple:
            return _tool_round_lifecycle_event_limit(call_ids)
    return RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS


class _SessionCommitGuardCancelled(RuntimeError):
    """An owned commit guard stopped without caller-owned cancellation."""


@dataclass(frozen=True, slots=True)
class _OwnedOffThreadSessionCommitGuard:
    callback: Callable[[], None]

    def __call__(self) -> None:
        try:
            self.callback()
        except asyncio.CancelledError as error:
            raise _SessionCommitGuardCancelled(
                "Session commit guard was unexpectedly cancelled."
            ) from error


async def _run_session_commit_guard_owned(
    commit_guard: Callable[[], None],
) -> BaseException | None:
    """Run an explicitly off-thread commit guard without abandoning its mutation."""

    if type(commit_guard) is not _OwnedOffThreadSessionCommitGuard:
        commit_guard()
        return None

    worker = asyncio.create_task(asyncio.to_thread(commit_guard))
    caller_signal: BaseException | None = None
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError as cancellation:
            if caller_signal is None:
                caller_signal = cancellation
            elif isinstance(caller_signal, asyncio.CancelledError):
                caller_signal.add_note(
                    "Additional cancellation arrived while the session commit guard settled."
                )
            else:
                caller_signal = BaseExceptionGroup(
                    "Session commit guard received multiple control signals.",
                    [caller_signal, cancellation],
                )
        except BaseException as signal:
            if worker.done():
                break
            caller_signal = (
                signal
                if caller_signal is None
                else BaseExceptionGroup(
                    "Session commit guard received multiple control signals.",
                    [caller_signal, signal],
                )
            )
    try:
        worker.result()
    except BaseException as guard_error:
        if not isinstance(guard_error, (Exception, asyncio.CancelledError)):
            if caller_signal is not None:
                raise guard_error from caller_signal
            raise
        if caller_signal is not None:
            raise caller_signal from guard_error
        raise
    return caller_signal


SessionOperationInitializer = Callable[
    [Session],
    Mapping[str, dict[str, Any]],
]


SessionOperationTransform = Callable[
    [Session, dict[str, Any] | None, dict[str, Any] | None],
    SessionOperationPublication,
]


StoreTimeSessionOperationTransform = Callable[
    [Session, dict[str, Any] | None, dict[str, Any] | None, datetime],
    SessionOperationPublication,
]


def _prepare_initial_session_operation_records(
    session: Session,
    initializer: SessionOperationInitializer | None,
) -> dict[str, dict[str, Any]]:
    """Run one initializer against a detached session and validate its records."""

    if initializer is None:
        return {}
    raw_records = initializer(session.model_copy(deep=True))
    if not isinstance(raw_records, Mapping):
        raise TypeError("Session operation initializer must return a mapping.")
    if len(raw_records) > RUNTIME_PUBLICATION_MAX_CHECKPOINT_OPERATIONS:
        raise ValueError(
            "Initial session operation records cannot exceed "
            f"{RUNTIME_PUBLICATION_MAX_CHECKPOINT_OPERATIONS} entries."
        )
    copied = copy_durable_json_object(
        dict(raw_records),
        "initial_session_operation_records",
    )
    from cayu.collaboration._session_export_store import OPERATION_PREFIX

    if any(key.startswith(OPERATION_PREFIX) for key in copied):
        raise ValueError("Session export records require their publication owner.")
    operation_record_rules._validate_session_operation_record_keys(copied)
    for record in copied.values():
        if type(record) is not dict:
            raise ValueError("Initial session operation records must be objects.")
    return copied
