"""Crash-safe construction and publication of one durable ordinary tool round.

The helper deliberately accepts only the exact, bounded lifecycle evidence for
the pending round.  Session-history discovery remains a store/runtime concern;
once discovered, this module validates and orders that evidence from the durable
``PendingToolRound`` identity alone.

Structured-output rounds can add auxiliary events through the explicit
``ToolRoundPublicationExtension`` hook.  The ordinary request stays strict, and
an extension may not replace or weaken its transcript, checkpoint mutation, or
durable lifecycle-event bindings.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from cayu._exception_groups import add_exception_note_safely
from cayu._task_wait import (
    await_shielded_task_outcome,
    unexpected_child_cancellation_error,
)
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    copy_durable_json_object,
    require_clean_nonblank,
)
from cayu.core.events import Event, EventType, copy_event
from cayu.runtime import _resume_ledger as resume_ledger
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_argument_publication as tool_argument_publication
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _transcript as transcript_support
from cayu.runtime._durable_subagents import (
    DURABLE_SUBAGENT_SUBMISSION_SEEDS_CHECKPOINT_KEY,
    DURABLE_SUBAGENT_SUBMISSIONS_CHECKPOINT_KEY,
    checkpoint_with_compacted_durable_subagent_submissions,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._tool_round_recovery import (
    PENDING_TOOL_ROUND_CHECKPOINT_KEY,
    PendingToolRound,
    checkpoint_without_pending_tool_round,
    pending_tool_round_from_checkpoint,
    pending_tool_round_identity,
    ready_assistant_publication_message,
)
from cayu.runtime.sessions import (
    RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS,
    RUNTIME_PUBLICATION_MAX_TOOL_CALLS,
    RuntimePublicationRequest,
    RuntimePublicationResult,
    SessionStatus,
    SessionStore,
    runtime_publication_checkpoint_mutation,
    runtime_publication_checkpoint_value_digest,
    runtime_publication_event_reference,
)
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME

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
_TOOL_ROUND_PUBLICATION_SCHEMA_VERSION = 1
_TOOL_ROUND_AUXILIARY_INTENT_KEY = "auxiliary"


@dataclass(frozen=True, slots=True)
class ToolRoundPublicationEvidence:
    """Canonical durable evidence for one complete ordinary tool round."""

    lifecycle_events: tuple[Event, ...]
    terminal_events: tuple[Event, ...]
    outcomes: tuple[runtime_records.ToolCallOutcome, ...]


class ToolRoundPublicationExtension(Protocol):
    """Typed extension point for structured-output auxiliary publication.

    Storage must explicitly support any auxiliary events before an extended
    request is published.  The helper validates that extensions retain every
    ordinary-round invariant and add no lifecycle event of their own.
    """

    def build_request(
        self,
        *,
        ordinary_request: RuntimePublicationRequest,
        pending_round: PendingToolRound,
    ) -> RuntimePublicationRequest: ...


@dataclass(frozen=True, slots=True)
class PreparedToolRoundPublication:
    """Stable request and fences retained for acknowledgement-loss replay."""

    session_id: str
    expected_statuses: frozenset[SessionStatus]
    expected_run_epoch: int
    expected_transcript_cursor: int
    _request: RuntimePublicationRequest
    _request_digest: str

    @property
    def request(self) -> RuntimePublicationRequest:
        """Return a detached inspection copy; publication retains its exact input."""

        return _copy_publication_request(self._request)


def collect_tool_round_publication_evidence(
    *,
    session_id: str,
    pending_round: PendingToolRound,
    durable_events: list[Event] | tuple[Event, ...],
) -> ToolRoundPublicationEvidence:
    """Validate and order an exact bounded set of started and terminal events.

    ``durable_events`` is intentionally not a whole session history.  Callers
    must query the candidate lifecycle events for the pending call identities;
    unrelated events are rejected so accidental filtering gaps fail closed.
    """

    session_id = require_clean_nonblank(session_id, "session_id")
    copied_pending_round = _copy_pending_round(pending_round)
    pending_calls = tuple(copied_pending_round.tool_calls)
    if len(pending_calls) > RUNTIME_PUBLICATION_MAX_TOOL_CALLS:
        raise ValueError(
            "Pending tool round cannot contain more than "
            f"{RUNTIME_PUBLICATION_MAX_TOOL_CALLS} tool calls."
        )
    pending_calls_by_id = {call.tool_call_id: call for call in pending_calls}
    if len(pending_calls_by_id) != len(pending_calls):
        raise ValueError("Pending tool round cannot repeat tool call ids.")

    if type(durable_events) not in (list, tuple):
        raise TypeError("durable_events must be a list or tuple of Event values.")
    maximum_evidence = min(
        RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS,
        len(pending_calls) * 2,
    )
    if len(durable_events) > maximum_evidence:
        raise ValueError(
            "Tool-round durable evidence exceeds the maximum possible started "
            "and terminal event count."
        )

    started_by_call: dict[str, Event] = {}
    terminal_by_call: dict[str, Event] = {}
    seen_event_ids: set[str] = set()
    for raw_event in durable_events:
        event = copy_event(raw_event)
        if event.id in seen_event_ids:
            raise ValueError(f"Tool-round durable event id is duplicated: {event.id}")
        seen_event_ids.add(event.id)
        if event.type not in _TOOL_ROUND_LIFECYCLE_EVENT_TYPES:
            raise ValueError(
                "Tool-round durable evidence may contain only started and terminal events."
            )

        tool_call_id = event.payload.get("tool_call_id")
        if type(tool_call_id) is not str or tool_call_id not in pending_calls_by_id:
            raise ValueError("Tool-round durable evidence must identify one pending tool call.")
        pending_call = pending_calls_by_id[tool_call_id]
        _validate_lifecycle_event_identity(
            event=event,
            session_id=session_id,
            pending_round=copied_pending_round,
            tool_call_id=tool_call_id,
            tool_name=pending_call.tool_name,
        )

        if event.type == EventType.TOOL_CALL_STARTED:
            if tool_call_id in started_by_call:
                raise ValueError(f"Tool call {tool_call_id} has duplicate durable started events.")
            if not tool_argument_publication.started_arguments_match_private_call(
                event.payload,
                private_arguments=pending_call.arguments,
            ):
                raise ValueError(
                    f"Tool call {tool_call_id} started with arguments that conflict "
                    "with its pending round."
                )
            started_by_call[tool_call_id] = event
            continue

        if tool_call_id in terminal_by_call:
            raise ValueError(f"Tool call {tool_call_id} has duplicate durable terminal events.")
        terminal_by_call[tool_call_id] = event

    expected_call_ids = set(pending_calls_by_id)
    if set(terminal_by_call) != expected_call_ids:
        missing = sorted(expected_call_ids - set(terminal_by_call))
        raise ValueError(
            "Tool-round publication requires exactly one terminal event for every "
            f"pending call; missing: {', '.join(missing)}."
        )

    ordered_lifecycle_events: list[Event] = []
    ordered_terminal_events: list[Event] = []
    ordered_outcomes: list[runtime_records.ToolCallOutcome] = []
    for pending_call in pending_calls:
        started = started_by_call.get(pending_call.tool_call_id)
        if started is not None:
            ordered_lifecycle_events.append(started.model_copy(deep=True))
        terminal = terminal_by_call[pending_call.tool_call_id]
        outcome = resume_ledger.tool_call_outcome_from_terminal_event(
            event=terminal,
            pending_tool_call=pending_call,
        )
        _validate_terminal_outcome(terminal=terminal, outcome=outcome)
        copied_terminal = terminal.model_copy(deep=True)
        ordered_lifecycle_events.append(copied_terminal)
        ordered_terminal_events.append(copied_terminal.model_copy(deep=True))
        ordered_outcomes.append(outcome)

    return ToolRoundPublicationEvidence(
        lifecycle_events=tuple(ordered_lifecycle_events),
        terminal_events=tuple(ordered_terminal_events),
        outcomes=tuple(ordered_outcomes),
    )


def build_tool_round_publication_request(
    *,
    session_id: str,
    pending_round: PendingToolRound,
    source_checkpoint: dict[str, Any] | None,
    durable_events: list[Event] | tuple[Event, ...],
    extension: ToolRoundPublicationExtension | None = None,
) -> RuntimePublicationRequest:
    """Build one deterministic request from a durable marker and exact evidence."""

    request, _ = _build_tool_round_publication_request(
        session_id=session_id,
        pending_round=pending_round,
        source_checkpoint=source_checkpoint,
        durable_events=durable_events,
        extension=extension,
    )
    return _copy_publication_request(request)


def prepare_tool_round_publication(
    *,
    session_id: str,
    pending_round: PendingToolRound,
    source_checkpoint: dict[str, Any] | None,
    durable_events: list[Event] | tuple[Event, ...],
    expected_statuses: set[SessionStatus] | frozenset[SessionStatus],
    expected_run_epoch: int,
    expected_transcript_cursor: int,
    extension: ToolRoundPublicationExtension | None = None,
) -> PreparedToolRoundPublication:
    """Retain one exact request and its mandatory first-commit fences."""

    request, _ = _build_tool_round_publication_request(
        session_id=session_id,
        pending_round=pending_round,
        source_checkpoint=source_checkpoint,
        durable_events=durable_events,
        extension=extension,
    )
    copied_statuses = _validate_expected_statuses(expected_statuses)
    expected_run_epoch = _validate_nonnegative_integer(
        expected_run_epoch,
        "expected_run_epoch",
    )
    expected_transcript_cursor = _validate_nonnegative_integer(
        expected_transcript_cursor,
        "expected_transcript_cursor",
    )
    request_digest = _publication_request_digest(session_id, request)
    return PreparedToolRoundPublication(
        session_id=session_id,
        expected_statuses=copied_statuses,
        expected_run_epoch=expected_run_epoch,
        expected_transcript_cursor=expected_transcript_cursor,
        _request=request,
        _request_digest=request_digest,
    )


async def publish_tool_round_publication(
    prepared: PreparedToolRoundPublication,
    *,
    session_store: SessionStore,
    event_writer: RuntimeEventWriter,
) -> RuntimePublicationResult:
    """Publish or replay the retained request, then idempotently fan out its events.

    Caller cancellation is deferred across both durable publication and
    post-commit fan-out.  A store error is not guessed to mean "not committed":
    callers retain ``prepared`` and can safely invoke this function again.
    """

    if type(prepared) is not PreparedToolRoundPublication:
        raise TypeError("prepared must be a PreparedToolRoundPublication.")
    request = prepared._request
    if _publication_request_digest(prepared.session_id, request) != prepared._request_digest:
        raise RuntimeError("Prepared tool-round publication request was mutated.")

    publication_task = asyncio.create_task(
        session_store.publish_runtime_publication(
            prepared.session_id,
            request=request,
            expected_statuses=set(prepared.expected_statuses),
            expected_run_epoch=prepared.expected_run_epoch,
            expected_transcript_cursor=prepared.expected_transcript_cursor,
        )
    )
    publication_outcome = await await_shielded_task_outcome(publication_task)
    cancellation = publication_outcome.cancellation
    publication_error = _classify_child_error(
        publication_outcome.error,
        operation="Tool-round durable publication",
    )
    if publication_error is not None:
        _raise_with_cancellation(
            publication_error,
            cancellation,
            cancellation_note="Tool-round durable publication failed during cancellation.",
        )
    result = publication_outcome.result
    if result is None:
        _raise_with_cancellation(
            RuntimeError("Tool-round durable publication returned no result."),
            cancellation,
            cancellation_note="Tool-round durable publication lost its result during cancellation.",
        )
    assert result is not None
    try:
        _validate_publication_result(
            prepared=prepared,
            request=request,
            result=result,
        )
    except BaseException as exc:
        _raise_with_cancellation(
            exc,
            cancellation,
            cancellation_note="Tool-round publication acknowledgement was invalid.",
        )

    appended_events = list(request.events)
    if appended_events:
        fan_out_task = asyncio.create_task(event_writer.fan_out_persisted(appended_events))
        fan_out_outcome = await await_shielded_task_outcome(
            fan_out_task,
            cancellation=cancellation,
        )
        cancellation = fan_out_outcome.cancellation
        fan_out_error = _classify_child_error(
            fan_out_outcome.error,
            operation="Tool-round persisted-event fan-out",
        )
        if fan_out_error is not None:
            _raise_with_cancellation(
                fan_out_error,
                cancellation,
                cancellation_note="Tool-round event fan-out failed during cancellation.",
            )
        if fan_out_outcome.result is None:
            _raise_with_cancellation(
                RuntimeError("Tool-round persisted-event fan-out returned no result."),
                cancellation,
                cancellation_note="Tool-round event fan-out lost its result during cancellation.",
            )

    if cancellation is not None:
        raise cancellation
    return result


async def publish_tool_round_with_exact_replay(
    prepared: PreparedToolRoundPublication,
    *,
    session_store: SessionStore,
    event_writer: RuntimeEventWriter,
) -> asyncio.CancelledError | None:
    """Publish once, reconciling an ambiguous acknowledgement by exact replay.

    A missing receipt proves that the first attempt did not commit.  A present
    receipt permits replay of the retained request only; rebuilding from current
    state could accidentally acknowledge different terminal evidence.
    """

    try:
        await publish_tool_round_publication(
            prepared,
            session_store=session_store,
            event_writer=event_writer,
        )
        return None
    except (BaseExceptionGroup, Exception, asyncio.CancelledError) as publication_error:
        # ``publish_tool_round_publication`` converts child-only scalar
        # cancellation to an operational error. A scalar cancellation crossing
        # this runtime-owned boundary is therefore authoritative caller
        # cancellation; a cancellation nested in a store-controlled group is
        # not. Preserve fatal child groups after reconciling their possibly
        # committed publication instead of forging caller cancellation from one
        # of their leaves.
        cancellation = (
            publication_error if isinstance(publication_error, asyncio.CancelledError) else None
        )
        fatal_child_group = (
            publication_error
            if isinstance(publication_error, BaseExceptionGroup)
            and not isinstance(publication_error, ExceptionGroup)
            else None
        )

        async def reconcile() -> bool:
            receipt = await session_store.load_runtime_publication_receipt(
                prepared.session_id,
                prepared.request.publication_id,
            )
            if receipt is None:
                return False
            await publish_tool_round_publication(
                prepared,
                session_store=session_store,
                event_writer=event_writer,
            )
            return True

        reconciliation_task = asyncio.create_task(reconcile())
        outcome = await await_shielded_task_outcome(
            reconciliation_task,
            cancellation=cancellation,
        )
        cancellation = outcome.cancellation
        if outcome.error is not None:
            reconciliation_error = outcome.error
            if isinstance(reconciliation_error, asyncio.CancelledError):
                reconciliation_error = unexpected_child_cancellation_error(
                    reconciliation_error,
                    operation="Tool-round publication reconciliation",
                )
            if cancellation is not None:
                add_exception_note_safely(
                    cancellation,
                    "Tool-round publication reconciliation failed: "
                    f"{type(reconciliation_error).__name__}.",
                )
                raise cancellation from reconciliation_error
            add_exception_note_safely(
                reconciliation_error,
                "Tool-round publication reconciliation also failed after "
                f"{type(publication_error).__name__}.",
            )
            raise reconciliation_error from publication_error
        if outcome.result is not True:
            if cancellation is not None:
                add_exception_note_safely(
                    cancellation,
                    (
                        "Tool-round publication failed before a durable receipt could be "
                        "reconstructed."
                    ),
                )
                raise cancellation from publication_error
            raise publication_error
        if cancellation is not None:
            return cancellation
        if fatal_child_group is not None:
            raise fatal_child_group from None
        return None


def _build_tool_round_publication_request(
    *,
    session_id: str,
    pending_round: PendingToolRound,
    source_checkpoint: dict[str, Any] | None,
    durable_events: list[Event] | tuple[Event, ...],
    extension: ToolRoundPublicationExtension | None,
) -> tuple[RuntimePublicationRequest, ToolRoundPublicationEvidence]:
    session_id = require_clean_nonblank(session_id, "session_id")
    copied_pending_round = _copy_pending_round(pending_round)
    is_structured_output_round = any(
        call.tool_name == STRUCTURED_OUTPUT_TOOL_NAME for call in copied_pending_round.tool_calls
    )
    if is_structured_output_round and extension is None:
        raise ValueError("Structured-output tool rounds require an explicit publication extension.")
    if not is_structured_output_round and extension is not None:
        raise ValueError(
            "Tool-round publication extensions are reserved for structured-output rounds."
        )
    copied_source_checkpoint = (
        None
        if source_checkpoint is None
        else copy_durable_json_object(source_checkpoint, "source_checkpoint")
    )
    if copied_source_checkpoint is None:
        raise ValueError("Source checkpoint has no pending tool round.")
    durable_pending_round = pending_tool_round_from_checkpoint(copied_source_checkpoint)
    if durable_pending_round is None:
        raise ValueError("Source checkpoint has no pending tool round.")
    if not _durable_json_equal(
        durable_pending_round.model_dump(mode="json"),
        copied_pending_round.model_dump(mode="json"),
    ):
        raise ValueError("Pending tool round conflicts with the durable source checkpoint marker.")

    evidence = collect_tool_round_publication_evidence(
        session_id=session_id,
        pending_round=copied_pending_round,
        durable_events=durable_events,
    )
    checkpoint = checkpoint_without_pending_tool_round(copied_source_checkpoint)
    checkpoint = checkpoint_with_compacted_durable_subagent_submissions(
        checkpoint,
        tool_round_id=copied_pending_round.tool_round_id,
        committed_handoffs=_committed_durable_subagent_handoffs(evidence),
    )
    mutation = runtime_publication_checkpoint_mutation(
        copied_source_checkpoint,
        checkpoint,
    )
    marker_operations = [
        operation
        for operation in mutation.operations
        if operation.key == PENDING_TOOL_ROUND_CHECKPOINT_KEY
    ]
    allowed_compaction_keys = {
        DURABLE_SUBAGENT_SUBMISSIONS_CHECKPOINT_KEY,
        DURABLE_SUBAGENT_SUBMISSION_SEEDS_CHECKPOINT_KEY,
    }
    unexpected_operations = [
        operation
        for operation in mutation.operations
        if operation.key != PENDING_TOOL_ROUND_CHECKPOINT_KEY
        and operation.key not in allowed_compaction_keys
    ]
    if (
        len(marker_operations) != 1
        or marker_operations[0].action != "delete"
        or unexpected_operations
    ):
        raise AssertionError("Tool-round publication contains an unexpected checkpoint mutation.")

    marker = copied_source_checkpoint[PENDING_TOOL_ROUND_CHECKPOINT_KEY]
    pending_round_digest = runtime_publication_checkpoint_value_digest(marker)
    if marker_operations[0].expected_value_digest != pending_round_digest:
        raise AssertionError("Pending tool-round deletion lost its exact marker fence.")
    tool_round_identity = pending_tool_round_identity(copied_pending_round)
    tool_call_ids = [call.tool_call_id for call in copied_pending_round.tool_calls]
    tool_result_messages = transcript_support.tool_result_messages(
        list(evidence.outcomes),
        tool_round_identity=tool_round_identity,
    )
    if len(tool_result_messages) != 1:
        raise AssertionError("Tool-round transcript construction must return one message.")
    transcript_messages = tuple(tool_result_messages)
    if copied_pending_round.assistant_message_state == "quarantined":
        projected_assistant = transcript_support.assistant_message_with_projected_tool_arguments(
            ready_assistant_publication_message(copied_pending_round),
            evidence.outcomes,
        )
        transcript_messages = (projected_assistant, *tool_result_messages)
    interaction_ids = {event.interaction_id for event in evidence.lifecycle_events}
    if len(interaction_ids) != 1 or None in interaction_ids:
        raise ValueError(
            "Tool-round publication lifecycle evidence requires one interaction identity."
        )
    interaction_id = next(iter(interaction_ids))
    ordinary_request = RuntimePublicationRequest(
        publication_id=f"tool-round:{copied_pending_round.tool_round_id}",
        kind="tool-round",
        interaction_id=interaction_id,
        intent={
            "schema_version": _TOOL_ROUND_PUBLICATION_SCHEMA_VERSION,
            "round_id": copied_pending_round.tool_round_id,
            **tool_round_identity.payload(),
            "tool_call_ids": tool_call_ids,
            "pending_round_digest": pending_round_digest,
        },
        mutation=mutation,
        transcript_messages=transcript_messages,
        events=(),
        referenced_events=tuple(
            runtime_publication_event_reference(event) for event in evidence.lifecycle_events
        ),
    )
    if extension is None:
        return ordinary_request, evidence

    extended_request = extension.build_request(
        ordinary_request=_copy_publication_request(ordinary_request),
        pending_round=_copy_pending_round(copied_pending_round),
    )
    if type(extended_request) is not RuntimePublicationRequest:
        raise TypeError("Tool-round publication extension must return a RuntimePublicationRequest.")
    copied_extended_request = _copy_publication_request(extended_request)
    _validate_extended_request(
        session_id=session_id,
        ordinary_request=ordinary_request,
        extended_request=copied_extended_request,
    )
    return copied_extended_request, evidence


def _committed_durable_subagent_handoffs(
    evidence: ToolRoundPublicationEvidence,
) -> dict[str, tuple[str, str]]:
    """Return exact child/task pairs with positive durable handoff evidence."""

    committed: dict[str, tuple[str, str]] = {}
    for outcome in evidence.outcomes:
        structured = outcome.result.structured
        if not isinstance(structured, Mapping) or structured.get("mode") != "durable":
            continue
        child_session_id = structured.get("child_session_id")
        queue_task_id = structured.get("queue_task_id")
        if (
            type(child_session_id) is str
            and type(queue_task_id) is str
            and (
                structured.get("status") == "queued"
                or (
                    structured.get("recovered") is True
                    and structured.get("recovery_reason")
                    == "pending_tool_round_reattached_subagent"
                )
            )
        ):
            committed[outcome.call.id] = (child_session_id, queue_task_id)
    return committed


def _validate_lifecycle_event_identity(
    *,
    event: Event,
    session_id: str,
    pending_round: PendingToolRound,
    tool_call_id: str,
    tool_name: str,
) -> None:
    if event.session_id != session_id:
        raise ValueError("Tool-round durable event belongs to a different session.")
    identity = pending_tool_round_identity(pending_round)
    if not identity.matches_payload(event.payload):
        raise ValueError("Tool-round durable event has a conflicting execution identity.")
    if event.agent_name != pending_round.agent_name:
        raise ValueError("Tool-round durable event has a conflicting agent identity.")
    if event.environment_name != pending_round.environment_name:
        raise ValueError("Tool-round durable event has a conflicting environment identity.")
    if event.tool_name != tool_name:
        raise ValueError(f"Tool call {tool_call_id} durable event has a conflicting tool name.")
    expected_idempotency_key = tool_execution.tool_idempotency_key(
        session_id=session_id,
        tool_round_id=pending_round.tool_round_id,
        tool_call_id=tool_call_id,
    )
    if event.payload.get("idempotency_key") != expected_idempotency_key:
        raise ValueError(
            f"Tool call {tool_call_id} durable event has a conflicting idempotency key."
        )


def _validate_terminal_outcome(
    *,
    terminal: Event,
    outcome: runtime_records.ToolCallOutcome,
) -> None:
    completed = terminal.type == EventType.TOOL_CALL_COMPLETED
    if completed == outcome.result.is_error:
        raise ValueError(
            f"Tool call {outcome.call.id} terminal event type conflicts with its result."
        )


def _validate_extended_request(
    *,
    session_id: str,
    ordinary_request: RuntimePublicationRequest,
    extended_request: RuntimePublicationRequest,
) -> None:
    if extended_request.publication_id != ordinary_request.publication_id:
        raise ValueError("Tool-round extension cannot replace the publication identity.")
    if extended_request.kind != ordinary_request.kind:
        raise ValueError("Tool-round extension cannot replace the publication kind.")
    if extended_request.interaction_id != ordinary_request.interaction_id:
        raise ValueError("Tool-round extension cannot replace the interaction identity.")
    if not _durable_json_equal(
        extended_request.mutation.model_dump(mode="json"),
        ordinary_request.mutation.model_dump(mode="json"),
    ):
        raise ValueError("Tool-round extension cannot replace the checkpoint mutation.")
    if not _durable_json_equal(
        [message.model_dump(mode="json") for message in extended_request.transcript_messages],
        [message.model_dump(mode="json") for message in ordinary_request.transcript_messages],
    ):
        raise ValueError("Tool-round extension cannot replace the grouped result message.")
    if not _durable_json_equal(
        [reference.model_dump(mode="json") for reference in extended_request.referenced_events],
        [reference.model_dump(mode="json") for reference in ordinary_request.referenced_events],
    ):
        raise ValueError("Tool-round extension cannot replace durable lifecycle evidence.")

    ordinary_intent = ordinary_request.intent
    for key, value in ordinary_intent.items():
        if key not in extended_request.intent or not _durable_json_equal(
            extended_request.intent[key],
            value,
        ):
            raise ValueError(f"Tool-round extension cannot replace intent.{key}.")
    extra_intent_keys = set(extended_request.intent) - set(ordinary_intent)
    if extra_intent_keys - {_TOOL_ROUND_AUXILIARY_INTENT_KEY}:
        raise ValueError("Tool-round extension intent must be namespaced under intent.auxiliary.")
    if (
        _TOOL_ROUND_AUXILIARY_INTENT_KEY in extended_request.intent
        and type(extended_request.intent[_TOOL_ROUND_AUXILIARY_INTENT_KEY]) is not dict
    ):
        raise TypeError("Tool-round extension intent.auxiliary must be an object.")

    for event in extended_request.events:
        if event.session_id != session_id:
            raise ValueError("Tool-round auxiliary event belongs to a different session.")
        if event.type in _TOOL_ROUND_LIFECYCLE_EVENT_TYPES:
            raise ValueError(
                "Tool-round extension cannot append started or terminal lifecycle events."
            )


def _validate_expected_statuses(
    statuses: set[SessionStatus] | frozenset[SessionStatus],
) -> frozenset[SessionStatus]:
    if type(statuses) not in (set, frozenset):
        raise TypeError("expected_statuses must be a set or frozenset of SessionStatus values.")
    if not statuses:
        raise ValueError("expected_statuses cannot be empty.")
    if any(type(status) is not SessionStatus for status in statuses):
        raise TypeError("expected_statuses must contain only SessionStatus values.")
    return frozenset(statuses)


def _validate_nonnegative_integer(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0.")
    if value > MAX_DURABLE_JSON_INTEGER:
        raise ValueError(f"{field_name} must be at most {MAX_DURABLE_JSON_INTEGER}.")
    return value


def _validate_publication_result(
    *,
    prepared: PreparedToolRoundPublication,
    request: RuntimePublicationRequest,
    result: RuntimePublicationResult,
) -> None:
    if type(result) is not RuntimePublicationResult:
        raise TypeError("Session store returned an invalid runtime publication result.")
    receipt = result.receipt
    if result.session.id != prepared.session_id or receipt.session_id != prepared.session_id:
        raise RuntimeError("Tool-round publication acknowledgement has a conflicting session.")
    if receipt.publication_id != request.publication_id or receipt.kind != request.kind:
        raise RuntimeError("Tool-round publication acknowledgement has a conflicting identity.")
    if receipt.interaction_id != request.interaction_id:
        raise RuntimeError(
            "Tool-round publication acknowledgement has a conflicting interaction identity."
        )
    if not _durable_json_equal(receipt.intent, request.intent):
        raise RuntimeError("Tool-round publication acknowledgement has conflicting intent.")
    if (
        receipt.source_status not in prepared.expected_statuses
        or receipt.source_run_epoch != prepared.expected_run_epoch
        or receipt.transcript_start_cursor != prepared.expected_transcript_cursor
    ):
        raise RuntimeError("Tool-round publication acknowledgement has conflicting source fences.")
    if receipt.appended_event_ids != tuple(event.id for event in request.events):
        raise RuntimeError(
            "Tool-round publication acknowledgement has conflicting appended events."
        )
    if not _durable_json_equal(
        [reference.model_dump(mode="json") for reference in receipt.referenced_events],
        [reference.model_dump(mode="json") for reference in request.referenced_events],
    ):
        raise RuntimeError(
            "Tool-round publication acknowledgement has conflicting referenced events."
        )
    expected_message_count = len(request.transcript_messages)
    if receipt.transcript_end_cursor - receipt.transcript_start_cursor != expected_message_count:
        raise RuntimeError(
            "Tool-round publication acknowledgement has a conflicting transcript span."
        )


def _classify_child_error(
    error: BaseException | None,
    *,
    operation: str,
) -> BaseException | None:
    if isinstance(error, asyncio.CancelledError):
        return unexpected_child_cancellation_error(error, operation=operation)
    return error


def _raise_with_cancellation(
    error: BaseException,
    cancellation: asyncio.CancelledError | None,
    *,
    cancellation_note: str,
) -> None:
    if cancellation is not None:
        cancellation.add_note(cancellation_note)
        raise cancellation from error
    raise error


def _copy_pending_round(pending_round: PendingToolRound) -> PendingToolRound:
    if type(pending_round) is not PendingToolRound:
        raise TypeError("pending_round must be a PendingToolRound.")
    return PendingToolRound.model_validate(pending_round.model_dump(mode="json"))


def _copy_publication_request(
    request: RuntimePublicationRequest,
) -> RuntimePublicationRequest:
    return RuntimePublicationRequest(
        publication_id=request.publication_id,
        kind=request.kind,
        interaction_id=request.interaction_id,
        intent=request.intent,
        mutation=request.mutation,
        transcript_messages=request.transcript_messages,
        events=request.events,
        operation_record_mutations=request.operation_record_mutations,
        referenced_events=request.referenced_events,
    )


def _publication_request_digest(
    session_id: str,
    request: RuntimePublicationRequest,
) -> str:
    return runtime_publication_checkpoint_value_digest(
        {
            "session_id": session_id,
            "request": request.model_dump(mode="json"),
        }
    )


def _durable_json_equal(left: Any, right: Any) -> bool:
    try:
        return runtime_publication_checkpoint_value_digest(
            left
        ) == runtime_publication_checkpoint_value_digest(right)
    except (TypeError, ValueError):
        return False
