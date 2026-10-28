"""Report a brokered sandbox's place in Cayu Cloud's queue as environment lifecycle progress.

A create with ``queue_seconds`` may wait in Cloud's queue for capacity before
it starts provisioning. ``report_sandbox_queue_position`` reads where it
stands and reports that through the runtime's environment lifecycle reporter,
so an operator reading ``environment.lifecycle.progress`` events can tell a
queued sandbox from a stuck one. ``BrokeredSandboxCodingEnvironmentFactory``
runs it beside ``wait_until_ready``; anything else that creates a sandbox
during a lifecycle operation can do the same.
"""

from __future__ import annotations

import asyncio
import logging
import math

from cayu.environments.lifecycle import (
    EnvironmentLifecyclePhase,
    EnvironmentLifecycleProgressReporter,
    EnvironmentLifecycleProgressStatus,
    current_environment_lifecycle_progress_reporter,
)
from cayu.sandbox_broker.client import SandboxBrokerClient, SandboxBrokerError

__all__ = [
    "DEFAULT_QUEUE_HEARTBEAT_SECONDS",
    "DEFAULT_QUEUE_POLL_INTERVAL_SECONDS",
    "report_sandbox_queue_position",
]

_LOGGER = logging.getLogger(__name__)
#: How often the queue is read: one ``capacity`` request each time.
DEFAULT_QUEUE_POLL_INTERVAL_SECONDS = 10.0
#: The longest a waiting sandbox goes without a report when its position doesn't change.
DEFAULT_QUEUE_HEARTBEAT_SECONDS = 60.0


async def report_sandbox_queue_position(
    broker: SandboxBrokerClient,
    sandbox_id: str,
    *,
    reporter: EnvironmentLifecycleProgressReporter | None = None,
    phase: EnvironmentLifecyclePhase = EnvironmentLifecyclePhase.FACTORY_PROVISIONING,
    poll_interval_s: float = DEFAULT_QUEUE_POLL_INTERVAL_SECONDS,
    heartbeat_s: float = DEFAULT_QUEUE_HEARTBEAT_SECONDS,
) -> None:
    """Report where a queued sandbox stands until it leaves Cloud's queue.

    Each report is an ``advanced`` observation of ``phase`` whose
    ``queued_count`` is the sandbox's queue position: how many places have to
    free up before Cloud admits it, 1 being the next. It is 0 when Cloud says
    a free place is certainly the sandbox's, and left out when Cloud can't
    work the position out or doesn't report positions; the report still says
    the sandbox is waiting. A report goes out when the position changes and at
    least every ``heartbeat_s`` otherwise. These reports use at most half of
    the policy's ``max_progress_events``, so the operation keeps room for its
    own. Lifecycle progress carries only counts, so the limit the sandbox
    waits for (``waiting_for``) goes to the log.

    ``reporter`` defaults to the runtime's current one; with neither, this
    returns at once. It returns once the sandbox isn't waiting any more
    (admitted, failed or ended), and is meant to be cancelled when the caller's
    ``wait_until_ready`` returns. A failed read is skipped: whether the sandbox
    becomes ready is for ``wait_until_ready`` to say. A lifecycle deadline the
    reporter raises propagates.
    """

    for name, value in (("poll_interval_s", poll_interval_s), ("heartbeat_s", heartbeat_s)):
        if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive number of seconds.")
    if not isinstance(phase, EnvironmentLifecyclePhase):
        raise TypeError("phase must be an EnvironmentLifecyclePhase.")
    if reporter is None:
        reporter = current_environment_lifecycle_progress_reporter()
    if reporter is None:
        return
    budget = reporter.policy.max_progress_events // 2
    loop = asyncio.get_running_loop()
    reported_at: float | None = None
    reported_position: int | None = None
    logged: tuple[int | None, str | None] | None = None
    while budget > 0:
        try:
            waiting, position, waiting_for = await _queue_observation(broker, sandbox_id)
        except SandboxBrokerError as error:
            _LOGGER.debug(
                "Reading brokered sandbox %s's queue position failed (%s).",
                sandbox_id,
                type(error).__name__,
            )
        else:
            if not waiting:
                return
            if logged != (position, waiting_for):
                logged = (position, waiting_for)
                _LOGGER.info(
                    "Brokered sandbox %s waits in Cayu Cloud's queue for %s capacity "
                    "(position %s).",
                    sandbox_id,
                    waiting_for or "any",
                    "unknown" if position is None else position,
                )
            now = loop.time()
            if (
                reported_at is None
                or position != reported_position
                or now - reported_at >= heartbeat_s
            ):
                progress = await reporter.report(
                    phase,
                    EnvironmentLifecycleProgressStatus.ADVANCED,
                    queued_count=position,
                )
                if progress is not None:
                    budget -= 1
                    reported_at = now
                    reported_position = position
        await asyncio.sleep(poll_interval_s)


async def _queue_observation(
    broker: SandboxBrokerClient, sandbox_id: str
) -> tuple[bool, int | None, str | None]:
    """Whether the sandbox waits, its position as reported, and the limit it waits for."""

    try:
        request = await broker.queue_position(sandbox_id)
    except SandboxBrokerError as error:
        if type(error) is not SandboxBrokerError:
            # Refused or unavailable: skip this read.
            raise
        # A Cloud without queue positions, or a queue longer than Cloud lists: the
        # record still says whether the sandbox waits.
        record = await broker.get(sandbox_id)
        return record.status == "provisioning" and record.queued_until is not None, None, None
    if request is None:
        return False, None, None
    if request.position is None and request.waiting_for is None:
        # A free place is certainly this sandbox's; the next admission pass takes it.
        return True, 0, None
    return True, request.position, request.waiting_for
