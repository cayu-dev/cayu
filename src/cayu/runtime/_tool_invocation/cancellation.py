"""Preserve caller cancellation while completed tool results are published."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable
from contextlib import suppress
from typing import Any, Never, TypeVar

from cayu._exception_groups import (
    exception_cause,
    iter_exception_tree,
    set_exception_cause,
)
from cayu._validation import (
    copy_json_value,
)
from cayu.runners._cleanup import (
    attach_runner_cancellation_failure,
    sanitize_runner_artifacts,
)
from cayu.runners.base import attach_cancellation_artifacts
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _runtime_records as runtime_records
from cayu.vaults.redaction import SecretRedactor

_PostToolResultT = TypeVar("_PostToolResultT")


def _contains_process_signal(error: BaseException | None) -> bool:
    if error is None:
        return False
    return any(
        isinstance(candidate, (GeneratorExit, KeyboardInterrupt, SystemExit))
        for candidate in iter_exception_tree(error)
    )


def _raise_preserved_post_tool_cancellation(
    cancellation: asyncio.CancelledError | None,
    failure: BaseException,
    *,
    restore_cancellation_requests: int,
) -> Never:
    """Keep an earlier caller cancellation authoritative over terminalization failure."""

    if cancellation is None or type(failure) is GeneratorExit or _contains_process_signal(failure):
        raise failure
    # Terminalization crosses stores, hooks, projections, and other extension
    # seams. Preserve the fact of the secondary failure without retaining its
    # potentially workload-derived message, traceback, or mutable state on the
    # public cancellation object.
    safe_failure = RuntimeError("Tool terminalization failed after caller cancellation.")
    del failure
    prior_cause = exception_cause(cancellation)
    if prior_cause is None:
        cause: BaseException = safe_failure
    else:
        cause = BaseExceptionGroup(
            "Post-tool cancellation and terminalization failures.",
            [prior_cause, safe_failure],
        )
    attach_runner_cancellation_failure(cancellation, cause)
    set_exception_cause(cancellation, cause)
    _raise_restored_post_tool_cancellation(
        cancellation,
        restore_cancellation_requests=restore_cancellation_requests,
        cause=cause,
    )


async def _receive_restored_post_tool_cancellation() -> None:
    """Observe restored control before publishing the durable tool outcome."""

    current_task = asyncio.current_task()
    if current_task is None or not current_task.cancelling():
        return
    # Observation and tool helpers restore requests before raising their owned
    # cancellation. On Python 3.11/3.12, uncancel() cannot rescind the queued
    # injection. Receive it before another publication await, while retaining
    # the original exception, its evidence, and every task cancellation count.
    with suppress(asyncio.CancelledError):
        await asyncio.sleep(0)


def _raise_restored_post_tool_cancellation(
    cancellation: asyncio.CancelledError,
    *,
    restore_cancellation_requests: int,
    cause: BaseException | None = None,
) -> Never:
    """Redeliver owned cancellation without erasing Task.cancelling() evidence."""

    current_task = asyncio.current_task()
    if current_task is not None:
        for _request in range(restore_cancellation_requests):
            current_task.cancel()
    raise cancellation from cause


async def _await_post_tool_operation(
    operation: Awaitable[_PostToolResultT],
    *,
    cancellation: asyncio.CancelledError | None,
    restore_cancellation_requests: int,
) -> _PostToolResultT:
    """Await terminalization without allowing it to replace owned cancellation."""

    if cancellation is None:
        return await operation
    try:
        return await operation
    except BaseException as failure:
        _raise_preserved_post_tool_cancellation(
            cancellation,
            failure,
            restore_cancellation_requests=restore_cancellation_requests,
        )


async def _iterate_post_tool_events(
    events: AsyncIterator[_PostToolResultT],
    *,
    cancellation: asyncio.CancelledError | None,
    restore_cancellation_requests: int,
) -> AsyncIterator[_PostToolResultT]:
    """Iterate terminalization events under the same cancellation authority."""

    if cancellation is None:
        async for event in events:
            yield event
        return
    try:
        async for event in events:
            yield event
    except BaseException as failure:
        _raise_preserved_post_tool_cancellation(
            cancellation,
            failure,
            restore_cancellation_requests=restore_cancellation_requests,
        )


def _transfer_cancellation_evidence(
    target: asyncio.CancelledError,
    sources: list[tuple[asyncio.CancelledError, str | None]],
    *,
    tool_calls: list[runtime_records.ToolCallRequest],
) -> None:
    """Move authenticated cleanup evidence onto one authoritative cancellation."""

    invocation_secrets.transfer_admission_refusals(
        target, [source for source, _tool_call_id in sources]
    )

    artifacts_by_id: dict[str, list[dict[str, Any]]] = {}
    redactors_by_id: dict[str, SecretRedactor] = {}
    unassigned_artifacts: list[dict[str, Any]] = []
    public_artifacts: list[dict[str, Any]] = []
    producer_ids: list[str] = []

    def extend_artifacts(
        destination: list[dict[str, Any]],
        artifacts: list[dict[str, Any]],
    ) -> None:
        copied = copy_json_value(artifacts, "cancellation_artifacts")
        if type(copied) is not list:
            return
        for artifact in copied:
            if type(artifact) is not dict:
                continue
            if artifact not in destination:
                destination.append(artifact)
            for public_artifact in sanitize_runner_artifacts([artifact]):
                if public_artifact not in public_artifacts:
                    public_artifacts.append(public_artifact)

    for source, fallback_tool_call_id in sources:
        source_artifacts_by_id = invocation_secrets.cancellation_artifacts_by_id(source)
        if source_artifacts_by_id is not None:
            for tool_call_id, artifacts in source_artifacts_by_id.items():
                extend_artifacts(
                    artifacts_by_id.setdefault(tool_call_id, []),
                    artifacts,
                )
        source_redactors_by_id = invocation_secrets.cancellation_redactors_by_id(source)
        if source_redactors_by_id is not None:
            redactors_by_id.update(source_redactors_by_id)

        source_artifacts = invocation_secrets.cancellation_artifacts(source)
        producer_id = invocation_secrets.cancellation_tool_call_id(source) or fallback_tool_call_id
        if producer_id is not None and producer_id not in producer_ids:
            producer_ids.append(producer_id)
        if source_artifacts:
            if producer_id is not None:
                extend_artifacts(
                    artifacts_by_id.setdefault(producer_id, []),
                    source_artifacts,
                )
            else:
                extend_artifacts(unassigned_artifacts, source_artifacts)
        source_redactor = invocation_secrets.cancellation_redactor(source)
        if source_redactor is not None and producer_id is not None:
            redactors_by_id[producer_id] = source_redactor

    if unassigned_artifacts and len(tool_calls) == 1:
        extend_artifacts(
            artifacts_by_id.setdefault(tool_calls[0].id, []),
            unassigned_artifacts,
        )
        unassigned_artifacts = []
    if not sources:
        return

    invocation_secrets.initialize_cancellation_evidence(target)
    if len(producer_ids) == 1:
        invocation_secrets.set_cancellation_tool_call_id(
            target,
            producer_ids[0],
        )
    if artifacts_by_id:
        invocation_secrets.set_cancellation_artifacts_by_id(
            target,
            artifacts_by_id,
        )
    if redactors_by_id:
        invocation_secrets.set_cancellation_redactors_by_id(
            target,
            redactors_by_id,
        )
    if public_artifacts:
        attach_cancellation_artifacts(target, public_artifacts)


def _grouped_cancellation_evidence(
    group: BaseExceptionGroup,
    *,
    tool_calls: list[runtime_records.ToolCallRequest],
) -> tuple[
    list[dict[str, Any]] | None,
    dict[str, list[dict[str, Any]]] | None,
    dict[str, SecretRedactor] | None,
]:
    """Project grouped cancellation leaves onto interrupted-round evidence."""

    sources = [
        (
            candidate,
            invocation_secrets.cancellation_tool_call_id(candidate),
        )
        for candidate in iter_exception_tree(group)
        if isinstance(candidate, asyncio.CancelledError)
    ]
    if not sources:
        return None, None, None
    cancellation = asyncio.CancelledError()
    _transfer_cancellation_evidence(
        cancellation,
        sources,
        tool_calls=tool_calls,
    )
    artifacts_by_id = invocation_secrets.cancellation_artifacts_by_id(cancellation)
    redactors_by_id = invocation_secrets.cancellation_redactors_by_id(cancellation)
    artifacts = invocation_secrets.cancellation_artifacts(cancellation)
    if artifacts_by_id is not None:
        artifacts = []
    return (
        artifacts or None,
        artifacts_by_id,
        redactors_by_id,
    )


def retain_cancellation_context(
    cancellation: asyncio.CancelledError,
    *,
    redactor: SecretRedactor,
    tool_call_id: str,
) -> None:
    """Attach the current call identity and redactor before cancellation escapes."""

    invocation_secrets.initialize_cancellation_evidence(cancellation)
    invocation_secrets.set_cancellation_redactor(cancellation, redactor)
    invocation_secrets.set_cancellation_tool_call_id(cancellation, tool_call_id)
