"""Shared fenced stream cleanup for application delegation and recovered execution."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Iterable
from contextlib import asynccontextmanager

from cayu._exception_groups import exception_cause, exception_tree_contains, set_exception_cause
from cayu._task_wait import capture_awaitable_outcome, unexpected_child_cancellation_error
from cayu.core.events import Event
from cayu.runtime.sessions import _SessionRunFenceContext
from cayu.runtime.workspace_observation_recovery import (
    retain_workspace_observation_pending_cancellation_requests,
    workspace_observation_pending_cancellation_requests,
)


class _RunFenceOwnedEventStream:
    """Advance and close one delegated stream under its captured run fences."""

    def __init__(self, stream: AsyncGenerator[Event, None]) -> None:
        self._stream = stream
        self._run_fences = _SessionRunFenceContext.current_or_new()

    def __aiter__(self) -> _RunFenceOwnedEventStream:
        return self

    async def __anext__(self) -> Event:
        with self._run_fences.activate():
            return await anext(self._stream)

    async def aclose(self) -> None:
        with self._run_fences.activate():
            await self._stream.aclose()


def _attach_delegated_failure_causes(
    authoritative_failure: BaseException,
    failures: Iterable[BaseException | None],
    *,
    message: str,
) -> None:
    evidence: list[BaseException] = []
    for failure in (*failures, exception_cause(authoritative_failure)):
        if failure is None or failure is authoritative_failure:
            continue
        if any(candidate is failure for candidate in evidence):
            continue
        evidence.append(failure)
    if not evidence:
        return
    set_exception_cause(
        authoritative_failure,
        evidence[0] if len(evidence) == 1 else BaseExceptionGroup(message, evidence),
    )


async def _close_owned_event_stream_resisting_cancellation(
    owned_stream: _RunFenceOwnedEventStream,
    *,
    cancellation: asyncio.CancelledError | None = None,
) -> tuple[tuple[asyncio.CancelledError, ...], BaseException | None]:
    """Finish delegated cleanup despite cancellation of the awaiting task."""

    cleanup_task = asyncio.create_task(
        capture_awaitable_outcome(owned_stream.aclose),
        name="cayu-delegated-stream-cleanup",
    )
    cancellations = [] if cancellation is None else [cancellation]
    while not cleanup_task.done():
        try:
            await asyncio.wait(
                (cleanup_task,),
                return_when=asyncio.ALL_COMPLETED,
            )
        except asyncio.CancelledError as exc:
            # asyncio.wait raises only when this caller is cancelled. A cancelled
            # cleanup task completes the wait and is inspected through result() below.
            cancellations.append(exc)
            if cleanup_task.cancelled():
                break
            continue

    cleanup_outcome = cleanup_task.result()
    cleanup_failure = cleanup_outcome.error
    if isinstance(cleanup_failure, asyncio.CancelledError):
        cleanup_failure = unexpected_child_cancellation_error(
            cleanup_failure,
            operation="Delegated runtime stream cleanup",
        )
    return tuple(cancellations), cleanup_failure


@asynccontextmanager
async def _close_delegated_event_stream(
    stream: AsyncGenerator[Event, None],
) -> AsyncIterator[_RunFenceOwnedEventStream]:
    """Close a delegated stream synchronously without hiding its exit signal."""

    owned_stream = _RunFenceOwnedEventStream(stream)
    authoritative_failure: BaseException | None = None
    try:
        yield owned_stream
    except BaseException as exc:
        authoritative_failure = exc
        raise
    finally:
        current_task = asyncio.current_task()
        workspace_cancellation_requests = (
            0
            if authoritative_failure is None
            else workspace_observation_pending_cancellation_requests(authoritative_failure)
        )
        cancellation_requests_at_cleanup_entry = (
            0 if current_task is None else current_task.cancelling()
        )
        cancellation_pending_at_cleanup_entry = bool(
            current_task is not None and getattr(current_task, "_must_cancel", False)
        )
        cancellation_requests_to_preserve_at_entry = (
            0
            if workspace_cancellation_requests == 0
            else max(
                workspace_cancellation_requests,
                cancellation_requests_at_cleanup_entry,
            )
        )
        pending_at_cleanup_entry: asyncio.CancelledError | None = None
        # ``Task.cancelling()`` includes historical requests that were already
        # delivered. A real checkpoint is the only positive evidence that one
        # of the entry-time requests is still pending for this cleanup.
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError as exc:
            pending_at_cleanup_entry = exc
        cancellation_requests_after_checkpoint = (
            0 if current_task is None else current_task.cancelling()
        )
        new_checkpoint_cancellation_requests = max(
            cancellation_requests_after_checkpoint - cancellation_requests_at_cleanup_entry,
            0,
        )
        try:
            cancellations, cleanup_failure = await _close_owned_event_stream_resisting_cancellation(
                owned_stream,
                cancellation=pending_at_cleanup_entry,
            )
            new_cancellations = list(cancellations)
            if new_cancellations:
                cancellation_requests_after_cleanup = (
                    0 if current_task is None else current_task.cancelling()
                )
                new_cancellation_requests = max(
                    cancellation_requests_after_cleanup - cancellation_requests_at_cleanup_entry,
                    0,
                )
                authoritative_cancellation_failure = isinstance(
                    authoritative_failure,
                    asyncio.CancelledError,
                ) or (
                    isinstance(authoritative_failure, BaseExceptionGroup)
                    and exception_tree_contains(authoritative_failure, asyncio.CancelledError)
                )
                pending_entry_cancellation_is_historical = (
                    pending_at_cleanup_entry is not None
                    and authoritative_cancellation_failure
                    and workspace_cancellation_requests > 0
                    and cancellation_requests_at_cleanup_entry <= workspace_cancellation_requests
                    and cancellation_pending_at_cleanup_entry
                    and new_checkpoint_cancellation_requests == 0
                )
                historical_group_cancellation = (
                    pending_entry_cancellation_is_historical and new_cancellation_requests == 0
                )
                if pending_entry_cancellation_is_historical:
                    new_cancellations = [
                        candidate
                        for candidate in new_cancellations
                        if candidate is not pending_at_cleanup_entry
                    ]
                if (
                    historical_group_cancellation
                    and authoritative_failure is not None
                    and not new_cancellations
                ):
                    process_control = exception_tree_contains(
                        authoritative_failure,
                        (GeneratorExit, KeyboardInterrupt, SystemExit),
                    )
                    authoritative_failure.add_note(
                        "Delegated runtime stream cleanup observed the already-grouped "
                        "caller cancellation. The authoritative failure group remains "
                        + (
                            "intact with concurrent process control."
                            if process_control
                            else "intact."
                        )
                    )
                    _attach_delegated_failure_causes(
                        authoritative_failure,
                        (cleanup_failure,),
                        message="Delegated runtime stream cleanup and concurrent control causes",
                    )
            if new_cancellations:
                authoritative_process_control = (
                    authoritative_failure is not None
                    and not isinstance(authoritative_failure, GeneratorExit)
                    and (
                        isinstance(authoritative_failure, (KeyboardInterrupt, SystemExit))
                        or (
                            isinstance(authoritative_failure, BaseExceptionGroup)
                            and exception_tree_contains(
                                authoritative_failure,
                                (GeneratorExit, KeyboardInterrupt, SystemExit),
                            )
                        )
                    )
                )
                cleanup_process_control = cleanup_failure is not None and exception_tree_contains(
                    cleanup_failure,
                    (GeneratorExit, KeyboardInterrupt, SystemExit),
                )
                if (
                    workspace_cancellation_requests > 0
                    or authoritative_cancellation_failure
                    or authoritative_process_control
                    or cleanup_process_control
                ):
                    failures: list[BaseException] = []
                    if authoritative_failure is not None:
                        failures.append(authoritative_failure)
                    failures.extend(new_cancellations)
                    if cleanup_failure is not None and all(
                        cleanup_failure is not failure for failure in failures
                    ):
                        failures.append(cleanup_failure)
                    propagated_failure = BaseExceptionGroup(
                        "Delegated runtime stream cleanup received additional failures.",
                        failures,
                    )
                    current_cancellation_requests = (
                        0 if current_task is None else current_task.cancelling()
                    )
                    cancellation_requests_after_entry = max(
                        current_cancellation_requests - cancellation_requests_at_cleanup_entry,
                        0,
                    )
                    pending_entry_request_count = int(
                        pending_at_cleanup_entry is not None
                        and cancellation_pending_at_cleanup_entry
                    )
                    authenticated_cancellation_requests = (
                        cancellation_requests_to_preserve_at_entry
                        + cancellation_requests_after_entry
                        if workspace_cancellation_requests > 0
                        else cancellation_requests_after_entry + pending_entry_request_count
                    )
                    if authenticated_cancellation_requests:
                        retain_workspace_observation_pending_cancellation_requests(
                            propagated_failure,
                            authenticated_cancellation_requests,
                        )
                    raise propagated_failure from None
                if (
                    authoritative_failure is not None
                    and authoritative_failure is not new_cancellations[0]
                ):
                    new_cancellations[0].add_note(
                        "Delegated runtime stream cleanup was cancelled after an earlier "
                        f"{type(authoritative_failure).__name__}."
                    )
                if cleanup_failure is not None and cleanup_failure is not new_cancellations[0]:
                    new_cancellations[0].add_note(
                        "Delegated runtime stream cleanup also failed: "
                        f"{type(cleanup_failure).__name__}."
                    )
                _attach_delegated_failure_causes(
                    new_cancellations[0],
                    (authoritative_failure, cleanup_failure),
                    message="Delegated runtime stream cancellation evidence",
                )
                raise new_cancellations[0]
            if cleanup_failure is not None:
                if authoritative_failure is None or isinstance(
                    authoritative_failure, GeneratorExit
                ):
                    raise cleanup_failure
                authoritative_failure.add_note(
                    "Delegated runtime stream cleanup failed: "
                    f"{type(cleanup_failure).__name__}. "
                    "The original stream failure remains authoritative."
                )
                if cleanup_failure is not authoritative_failure:
                    _attach_delegated_failure_causes(
                        authoritative_failure,
                        (cleanup_failure,),
                        message="Delegated runtime stream cleanup and prior failure causes",
                    )
        finally:
            if current_task is not None:
                cancellation_requests_after_cleanup = current_task.cancelling()
                cancellation_requests_to_preserve = (
                    0
                    if workspace_cancellation_requests == 0
                    else cancellation_requests_to_preserve_at_entry
                    + max(
                        cancellation_requests_after_cleanup
                        - cancellation_requests_at_cleanup_entry,
                        0,
                    )
                )
                for _request in range(
                    max(
                        cancellation_requests_to_preserve - cancellation_requests_after_cleanup,
                        0,
                    )
                ):
                    current_task.cancel()
