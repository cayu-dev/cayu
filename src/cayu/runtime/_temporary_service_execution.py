"""Bound active service observation without replacing session lifetime authority."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from cayu.deadlines import ExecutionDeadline
from cayu.events import Event
from cayu.runtime._delegated_event_stream import _close_delegated_event_stream


async def drive_temporary_service(
    stream: AsyncGenerator[Event, None], boundary: ExecutionDeadline
) -> None:
    """Cancel the owned runtime task at the earliest applicable boundary.

    This is not the parent session's execution deadline. Ordinary resume must
    retain that immutable lifetime boundary. Nor is expiry an exclusion receipt:
    existing runtime cleanup and native receiving readback alone prove settlement.
    The caller reconciles retained preparations/admissions before entering here,
    so a retry cannot start a new timed execution for an unresolved operation.
    """
    timer = asyncio.timeout(boundary.remaining_seconds())
    try:
        async with timer:
            async with _close_delegated_event_stream(stream) as owned:
                boundary.require_admission("temporary_service")
                async for _ in owned:
                    boundary.require_admission("temporary_service_progress")
            # A provider may suppress cancellation and return a terminal event.
            # Retain that evidence, but do not report timely service completion.
            boundary.require_admission("temporary_service_completion")
    except BaseException as error:
        if timer.expired() or boundary.expired:
            error.__dict__["temporary_service_deadline"] = boundary.inspection()
        raise
