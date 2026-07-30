"""Crash-safe publication for opening and closing one tool approval.

Approval checkpoints are security authority, so an ambiguous store response
cannot be interpreted as either "committed" or "not committed".  This module
retains the exact request, reconciles it by its insert-only runtime-publication
receipt, and replays only identical material.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from cayu._exception_groups import add_exception_note_safely
from cayu._task_wait import (
    await_shielded_task_outcome,
    unexpected_child_cancellation_error,
)
from cayu._validation import MAX_DURABLE_JSON_INTEGER, copy_durable_json_object
from cayu.core.events import Event
from cayu.core.messages import Message
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime.sessions import (
    RuntimePublicationRequest,
    RuntimePublicationResult,
    SessionStatus,
    SessionStore,
    attribute_events_to_current_interaction,
    runtime_publication_checkpoint_mutation,
    runtime_publication_checkpoint_value_digest,
    runtime_publication_event_reference,
)

ApprovalPublicationKind = Literal["approval-open", "approval-close"]


@dataclass(frozen=True, slots=True)
class PreparedApprovalPublication:
    """One immutable approval publication and its first-commit fences."""

    session_id: str
    expected_statuses: frozenset[SessionStatus]
    expected_run_epoch: int
    expected_transcript_cursor: int
    _request: RuntimePublicationRequest
    _request_digest: str

    @property
    def request(self) -> RuntimePublicationRequest:
        return _copy_request(self._request)


def prepare_approval_publication(
    *,
    session_id: str,
    publication_id: str,
    kind: ApprovalPublicationKind,
    intent: dict[str, Any],
    source_checkpoint: dict[str, Any] | None,
    target_checkpoint: dict[str, Any] | None,
    transcript_messages: list[Message] | tuple[Message, ...] = (),
    events: list[Event] | tuple[Event, ...] = (),
    referenced_events: list[Event] | tuple[Event, ...] = (),
    expected_statuses: set[SessionStatus] | frozenset[SessionStatus],
    expected_run_epoch: int,
    expected_transcript_cursor: int,
) -> PreparedApprovalPublication:
    """Build and retain one exact approval-boundary request."""

    if kind not in {"approval-open", "approval-close"}:
        raise ValueError("Unsupported approval publication kind.")
    if type(expected_statuses) not in (set, frozenset) or not expected_statuses:
        raise ValueError("expected_statuses must be a non-empty set.")
    if any(type(status) is not SessionStatus for status in expected_statuses):
        raise TypeError("expected_statuses must contain SessionStatus values.")
    for value, field_name in (
        (expected_run_epoch, "expected_run_epoch"),
        (expected_transcript_cursor, "expected_transcript_cursor"),
    ):
        if type(value) is not int or not 0 <= value <= MAX_DURABLE_JSON_INTEGER:
            raise ValueError(f"{field_name} must be a non-negative durable integer.")
    copied_source = (
        None
        if source_checkpoint is None
        else copy_durable_json_object(source_checkpoint, "source_checkpoint")
    )
    copied_target = (
        None
        if target_checkpoint is None
        else copy_durable_json_object(target_checkpoint, "target_checkpoint")
    )
    attributed_events = tuple(attribute_events_to_current_interaction(list(events)))
    interaction_ids = {event.interaction_id for event in (*attributed_events, *referenced_events)}
    if len(interaction_ids) != 1 or None in interaction_ids:
        raise ValueError("Approval publication evidence requires one interaction identity.")
    interaction_id = next(iter(interaction_ids))
    request = RuntimePublicationRequest(
        publication_id=publication_id,
        kind=kind,
        interaction_id=interaction_id,
        intent=intent,
        mutation=runtime_publication_checkpoint_mutation(copied_source, copied_target),
        transcript_messages=tuple(transcript_messages),
        events=attributed_events,
        referenced_events=tuple(
            runtime_publication_event_reference(event) for event in referenced_events
        ),
    )
    digest = runtime_publication_checkpoint_value_digest(
        {"session_id": session_id, "request": request.model_dump(mode="json")}
    )
    return PreparedApprovalPublication(
        session_id=session_id,
        expected_statuses=frozenset(expected_statuses),
        expected_run_epoch=expected_run_epoch,
        expected_transcript_cursor=expected_transcript_cursor,
        _request=request,
        _request_digest=digest,
    )


async def publish_approval_with_exact_replay(
    prepared: PreparedApprovalPublication,
    *,
    session_store: SessionStore,
    event_writer: RuntimeEventWriter,
    fan_out: bool = True,
) -> asyncio.CancelledError | None:
    """Publish once and reconcile a lost acknowledgement from the exact receipt."""

    if type(prepared) is not PreparedApprovalPublication:
        raise TypeError("prepared must be a PreparedApprovalPublication.")
    request = prepared._request
    if (
        runtime_publication_checkpoint_value_digest(
            {"session_id": prepared.session_id, "request": request.model_dump(mode="json")}
        )
        != prepared._request_digest
    ):
        raise RuntimeError("Prepared approval publication request was mutated.")

    cancellation: asyncio.CancelledError | None = None
    try:
        cancellation = await _publish(prepared, session_store=session_store)
    except (BaseExceptionGroup, Exception, asyncio.CancelledError) as publication_error:
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
                request.publication_id,
            )
            if receipt is None:
                return False
            await _publish(prepared, session_store=session_store)
            return True

        outcome = await await_shielded_task_outcome(
            asyncio.create_task(reconcile()),
            cancellation=cancellation,
        )
        cancellation = outcome.cancellation
        reconciliation_error = outcome.error
        if isinstance(reconciliation_error, asyncio.CancelledError):
            reconciliation_error = unexpected_child_cancellation_error(
                reconciliation_error,
                operation="Approval publication reconciliation",
            )
        if reconciliation_error is not None:
            if cancellation is not None:
                add_exception_note_safely(
                    cancellation,
                    "Approval publication reconciliation failed.",
                )
                raise cancellation from reconciliation_error
            raise reconciliation_error from publication_error
        if outcome.result is not True:
            if cancellation is not None:
                add_exception_note_safely(
                    cancellation,
                    "Approval publication failed before a durable receipt was reconstructed.",
                )
                raise cancellation from publication_error
            raise publication_error
        if fatal_child_group is not None:
            raise fatal_child_group from None

    if cancellation is not None:
        if not fan_out:
            return cancellation
        raise cancellation
    appended_events = list(request.events)
    if fan_out and appended_events:
        # Fan-out is post-commit and idempotent. Keep it caller-cancellable so
        # interruption can close a durably opened approval even when a sink is
        # unavailable; the persisted side-effect handoff remains retryable.
        await event_writer.fan_out_persisted(appended_events)
    return None


async def _publish(
    prepared: PreparedApprovalPublication,
    *,
    session_store: SessionStore,
) -> asyncio.CancelledError | None:
    outcome = await await_shielded_task_outcome(
        asyncio.create_task(
            session_store.publish_runtime_publication(
                prepared.session_id,
                request=prepared._request,
                expected_statuses=set(prepared.expected_statuses),
                expected_run_epoch=prepared.expected_run_epoch,
                expected_transcript_cursor=prepared.expected_transcript_cursor,
            )
        )
    )
    error = outcome.error
    if isinstance(error, asyncio.CancelledError):
        error = unexpected_child_cancellation_error(
            error,
            operation="Approval durable publication",
        )
    if error is not None:
        if outcome.cancellation is not None:
            outcome.cancellation.add_note(
                "Approval durable publication failed during cancellation."
            )
            raise outcome.cancellation from error
        raise error
    result = outcome.result
    if type(result) is not RuntimePublicationResult:
        raise RuntimeError("Approval durable publication returned an invalid result.")
    receipt = result.receipt
    request = prepared._request
    if (
        result.session.id != prepared.session_id
        or receipt.session_id != prepared.session_id
        or receipt.publication_id != request.publication_id
        or receipt.kind != request.kind
        or receipt.interaction_id != request.interaction_id
        or receipt.intent != request.intent
        or receipt.source_status not in prepared.expected_statuses
        or receipt.source_run_epoch != prepared.expected_run_epoch
        or receipt.transcript_start_cursor != prepared.expected_transcript_cursor
        or receipt.transcript_end_cursor - receipt.transcript_start_cursor
        != len(request.transcript_messages)
        or receipt.appended_event_ids != tuple(event.id for event in request.events)
        or receipt.referenced_events != request.referenced_events
    ):
        raise RuntimeError("Approval publication acknowledgement conflicts with its request.")
    return outcome.cancellation


def _copy_request(request: RuntimePublicationRequest) -> RuntimePublicationRequest:
    return RuntimePublicationRequest(
        publication_id=request.publication_id,
        kind=request.kind,
        interaction_id=request.interaction_id,
        intent=request.intent,
        mutation=request.mutation,
        transcript_messages=request.transcript_messages,
        events=request.events,
        referenced_events=request.referenced_events,
    )
