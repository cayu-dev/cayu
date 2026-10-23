"""Owned task continuation for an explicitly delegated Inbox decision."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncGenerator
from contextlib import suppress
from typing import TYPE_CHECKING
from uuid import uuid4

from cayu._task_wait import (
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    restore_task_cancellation_requests,
)
from cayu._validation import revalidate_model_input
from cayu.approvals.tools import ToolApprovalRequest, copy_tool_approval_request
from cayu.approvals.user_input import (
    UserInputResponse,
    copy_user_input_response,
    user_input_resolution_request_digest,
)
from cayu.events import Event
from cayu.runtime._approval_support import approval_resolution_request_digest
from cayu.runtime._delegated_event_stream import _close_delegated_event_stream
from cayu.runtime.human_attention import HumanAttentionReference, HumanAttentionRequest
from cayu.sessions import PendingActionQuery
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.records import SessionStatus
from cayu.tasks.continuation_claims import TaskContinuationExpectation
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import Task, TaskClaimLost, TaskStatus
from cayu.tasks.worker import TaskHandlerOutcome, _run_recovered_interrupted_task_handler

if TYPE_CHECKING:
    from cayu.applications import CayuApp


async def _require_current_selection(app: CayuApp, reference: HumanAttentionReference) -> None:
    session = await app.session_store.load(reference.session_id)
    if session is None or session.instance_id != reference.session_instance_id:
        raise TaskClaimLost("Selected decision session changed.")
    pending = await app.session_store.query_pending_actions(
        PendingActionQuery(session_id=reference.session_id)
    )
    if pending.issues or pending.has_more:
        raise TaskClaimLost("Selected decision cannot be proved from complete source evidence.")
    if not any(
        selected is not None and selected.reference == reference
        for selected in (
            HumanAttentionRequest.from_pending_action(item) for item in pending.actions
        )
    ):
        raise TaskClaimLost("Selected decision pause changed.")


async def controlled_inbox_continuation(
    app: CayuApp,
    request: ToolApprovalRequest | UserInputResponse,
    reference: HumanAttentionReference,
) -> AsyncGenerator[Event, None]:
    """Resolve an explicitly delegated decision using current runtime authority."""
    request = (
        copy_tool_approval_request(request)
        if isinstance(request, ToolApprovalRequest)
        else copy_user_input_response(request)
    )
    reference = HumanAttentionReference.model_validate(
        revalidate_model_input(reference, HumanAttentionReference)
    )
    if request.task_worker_id is not None or request.task_handoff_id is not None:
        raise ValueError("Controlled continuation cannot borrow supplied worker authority.")
    action_id = (
        request.approval_id if isinstance(request, ToolApprovalRequest) else request.input_id
    )
    if request.session_id != reference.session_id or action_id != reference.action_id:
        raise TaskClaimLost("Decision request does not match its selected pause.")
    await app.require_human_review_resolution_authority(
        request.session_id, request.review_reference
    )
    await _require_current_selection(app, reference)
    linked = (
        []
        if app.task_store is None
        else await app.task_store.list_tasks(
            TaskQuery(session_id=reference.session_id, status=TaskStatus.RUNNING, limit=2)
        )
    )
    if len(linked) > 1:
        raise TaskClaimLost("Session has ambiguous task ownership.")
    if not linked:
        stream = (
            app.resolve_tool_approval(request)
            if isinstance(request, ToolApprovalRequest)
            else app.resolve_user_input(request)
        )
        async with _close_delegated_event_stream(stream) as owned:
            async for event in owned:
                yield event
        return
    task = linked[0]
    if (
        task.session_instance_id != reference.session_instance_id
        or task.worker_id is not None
        or task.lease_expires_at is not None
        or task.interrupted_handoff_id is None
    ):
        raise TaskClaimLost("Task continuation is not available for claiming.")
    decision_digest = (
        approval_resolution_request_digest(request)
        if isinstance(request, ToolApprovalRequest)
        else user_input_resolution_request_digest(request)
    )
    expected = TaskContinuationExpectation(
        task_id=task.id,
        session_id=reference.session_id,
        session_instance_id=reference.session_instance_id,
        prior_handoff_id=task.interrupted_handoff_id,
        operation_sha256=hashlib.sha256(
            json.dumps(
                {"selection": reference.model_dump(), "decision_digest": decision_digest},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    )
    async with _close_delegated_event_stream(
        _owned_inbox_continuation(app, request, reference, expected)
    ) as owned:
        async for event in owned:
            yield event


async def _owned_inbox_continuation(
    app: CayuApp,
    request: ToolApprovalRequest | UserInputResponse,
    reference: HumanAttentionReference,
    expected: TaskContinuationExpectation,
) -> AsyncGenerator[Event, None]:
    """Keep claim, runtime stream, heartbeat and cleanup under one stream owner.

    The HTTP detached mutation driver owns this iterator. Losing its observer does
    not close it; application shutdown does, and joins the task worker's cleanup.
    """
    task_store = app.task_store
    if task_store is None:
        raise NotImplementedError("Controlled continuation requires a task store.")
    events: asyncio.Queue[Event] = asyncio.Queue(maxsize=64)
    handler_errors: list[Exception] = []

    async def handler(owned_app: CayuApp, task: Task, worker_id: str):
        selected = request.model_copy(
            update={"task_worker_id": worker_id, "task_handoff_id": task.interrupted_handoff_id}
        )
        stream = (
            owned_app.resolve_tool_approval(selected)
            if isinstance(selected, ToolApprovalRequest)
            else owned_app.resolve_user_input(selected)
        )
        try:
            await _require_current_selection(owned_app, reference)
            async with _close_delegated_event_stream(stream) as owned_stream:
                async for event in owned_stream:
                    await events.put(event)
        except Exception as error:
            handler_errors.append(error)
            # A rejected decision must not turn a still-interrupted task into a
            # failed task. The normal worker handoff verifies the durable boundary.
            try:
                session = await owned_app.session_store.load(expected.session_id)
            except Exception as read_error:
                combined = ExceptionGroup(
                    "Decision and authoritative state read failed.", [error, read_error]
                )
                handler_errors[-1] = combined
                # Unknown session state is not authority to terminalize the task.
                raise SessionRunFenced("Decision authority could not be revalidated.") from combined
            if (
                session is not None
                and session.instance_id == expected.session_instance_id
                and session.status is SessionStatus.INTERRUPTED
            ):
                return TaskHandlerOutcome.SESSION_INTERRUPTED
            raise
        session = await owned_app.session_store.load(expected.session_id)
        if (
            session is not None
            and session.instance_id == expected.session_instance_id
            and session.status is SessionStatus.INTERRUPTED
        ):
            return TaskHandlerOutcome.SESSION_INTERRUPTED
        return None

    async def run():
        page = await _run_recovered_interrupted_task_handler(
            app,
            task_store,
            handler,
            worker_id=f"inbox-worker:{uuid4()}",
            expected=expected,
            apply_unproductive_backoff=False,
            lease_seconds=300,
            query=None,
            after=None,
        )
        if page.task is None:
            raise TaskClaimLost("Selected task continuation is currently unavailable.")
        if handler_errors:
            raise handler_errors[0]

    worker = asyncio.create_task(capture_awaitable_outcome(lambda: app._run_worker_step(run)))
    receive: asyncio.Task[Event] | None = None
    primary: BaseException | None = None
    try:
        while not worker.done() or not events.empty():
            receive = asyncio.create_task(events.get())
            await asyncio.wait({receive, worker}, return_when=asyncio.FIRST_COMPLETED)
            if receive.done():
                event = receive.result()
                receive = None
                yield event
            else:
                receive.cancel()
                with suppress(asyncio.CancelledError):
                    await receive
                receive = None
        outcome = worker.result()
        if outcome.error is not None:
            raise outcome.error
    except BaseException as error:
        primary = error
        raise
    finally:
        if receive is not None:
            # Queue receive owns no mutation; cancelling it cannot delay joining
            # the worker that owns the lease and runtime cleanup.
            receive.cancel()

        async def discard_remaining_events():
            while True:
                await events.get()

        # The worker deliberately drains dispatched mutations under a durable
        # cancellation fence. Keep its bounded event channel flowing while the
        # observer is gone, otherwise a long cleanup could deadlock on put().
        discard = asyncio.create_task(discard_remaining_events())
        if not worker.done():
            worker.cancel()
        try:
            joined = await await_shielded_task_outcome(worker)
        finally:
            discard.cancel()
        restore_task_cancellation_requests(
            joined.cancellation_requests_consumed, cancellation=joined.cancellation
        )
        failure = joined.error or (None if joined.result is None else joined.result.error)
        if joined.cancellation is not None and primary is None:
            raise joined.cancellation from failure
        if (
            failure is not None
            and failure is not primary
            and not isinstance(failure, asyncio.CancelledError)
        ):
            if primary is None:
                raise failure
            if isinstance(primary, asyncio.CancelledError):
                raise primary from failure
            raise BaseExceptionGroup("Inbox continuation and cleanup failed.", [primary, failure])
