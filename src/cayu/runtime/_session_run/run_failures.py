"""How a session run classifies and settles the failures that end it."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from cayu._exception_groups import exception_cause, iter_exception_tree
from cayu._exception_state import pop_exception_state
from cayu._task_wait import capture_awaitable_outcome
from cayu.exceptions import (
    TerminalEventPublicationUncertain,
    _is_runtime_interaction_lifecycle_publication_rejection,
)
from cayu.runtime._durable_subagents import is_durable_subagent_submission_unsettled
from cayu.runtime._session_finalization import (
    _INTERACTION_TRANSITION_RUN_FENCE_ATTRIBUTE,
    _is_interaction_transition_run_fence,
)
from cayu.sessions._invocation_scope import _current_session_interaction_id


async def _settle_setup_interruption(operation: Callable[[], Awaitable[None]]) -> None:
    """Join fresh interruption work without normalizing its caller's requests."""
    cleanup = asyncio.create_task(
        capture_awaitable_outcome(operation), name="cayu-setup-interruption-cleanup"
    )
    cancellation: asyncio.CancelledError | None = None
    while not cleanup.done():
        try:
            await asyncio.wait((cleanup,))
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
    failure = cleanup.result().error
    if cancellation is not None:
        if failure is not None and not isinstance(failure, (Exception, asyncio.CancelledError)):
            raise BaseExceptionGroup(
                "Setup interruption failure and caller cancellation.", [failure, cancellation]
            )
        raise cancellation from failure
    if failure is not None:
        raise failure


def _consume_authenticated_interaction_transition_run_fence(
    error: BaseException,
) -> bool:
    """Consume current transition fence markers without discarding evidence.

    A sibling transition boundary removes the marker from caller cancellation,
    but the exact store failure in its cause remains independently marked.  The
    enclosing run owner must recognize that authenticated child marker before
    deciding whether stale-epoch cleanup is legal.
    """

    authenticated = False
    for candidate in iter_exception_tree(error):
        if not _is_interaction_transition_run_fence(candidate):
            continue
        authenticated = True
        pop_exception_state(
            candidate,
            _INTERACTION_TRANSITION_RUN_FENCE_ATTRIBUTE,
        )
    return authenticated


def _recovery_owns_run_failure(
    error: BaseException,
    *,
    session_id: str,
    runtime_authority: object,
) -> bool:
    """Whether a run body deliberately left its invocation open for recovery.

    Uncertain terminal evidence, an unsettled durable child submission, or a
    lifecycle publication the runtime refused (also when it is only the cause
    of the exit, as after a refused abandonment) leave the session RUNNING. A
    refusal counts only with this runtime's provenance: extension code can
    construct the public exception, and must not keep a session open with it.
    """

    interaction_id = _current_session_interaction_id(session_id)
    pending = [error]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop()
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if isinstance(
            candidate, TerminalEventPublicationUncertain
        ) or _is_runtime_interaction_lifecycle_publication_rejection(
            candidate,
            session_id=session_id,
            interaction_id=interaction_id,
            runtime_authority=runtime_authority,
        ):
            return True
        pending.extend(iter_exception_tree(candidate))
        cause = exception_cause(candidate)
        if cause is not None:
            pending.append(cause)
    return is_durable_subagent_submission_unsettled(error, parent_session_id=session_id)


def _failure_carries_interaction_publication_rejection(
    failure: Exception,
    *,
    session_id: str,
    interaction_id: str | None,
    runtime_authority: object,
) -> bool:
    cause = exception_cause(failure)
    return cause is not None and any(
        _is_runtime_interaction_lifecycle_publication_rejection(
            candidate,
            session_id=session_id,
            interaction_id=interaction_id,
            runtime_authority=runtime_authority,
        )
        for candidate in iter_exception_tree(cause)
    )
