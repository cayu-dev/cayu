"""Retire a retained environment owner once its execution environment is gone for good.

A release that fails or is blocked keeps the session's environment owner. Its
workspace reservation stays held for a finalize retry, and the environment's
mutation fence stays closed until fenced work proves it stopped. A binding can
say when that can never happen: once the environment it bound no longer exists,
no retry can publish anything and nothing in flight can change the bound copy.
The owner is then retired, so the workspace and the registration can be used
again by later sessions in this process.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# How long a watch waits before settling again after an attempt refused to retire.
_RETRY_FIRST_SECONDS = 1.0
_RETRY_MAX_SECONDS = 60.0


def release_is_retained(owner: Any, *, claimed: bool = False) -> bool:
    """Whether a finished cleanup attempt left ``owner`` holding a factory environment.

    The owner must have finished at least one cleanup attempt, have no other
    attempt running (``claimed`` means the caller's own attempt holds it), and
    have its failure evidence durable with no in-memory completion marker left
    to clear: those are the runtime's own obligations, and retiring the owner
    must not skip them.
    """

    if not (owner.cleanup_started and owner.cleanup_finished):
        return False
    if owner.cleanup_settlement_started is not claimed:
        return False
    if (
        not owner.cleanup_release_safe
        or owner.pending_finalize_failure_event is not None
        or owner.pending_completion_marker_clear is not None
    ):
        return False
    registered = owner.registered_environment
    if (
        not registered.factory_backed
        or registered.environment.binding is None
        or registered.bound_workspace is None
    ):
        return False
    return (
        owner.cleanup_settlement_deferred
        or owner.cleanup_requires_finalize_retry
        or registered.workspace_mutation_fence.has_uncertain_mutations()
    )


@dataclass(eq=False)
class _Watch:
    owner: Any
    changed: asyncio.Event
    task: asyncio.Task[None] | None = None
    end: Awaitable[str] | None = None


class RetainedEnvironmentEndWatches:
    """Watch each retained owner's environment and retire the owner when it ends.

    ``observe`` is called whenever a session's cleanup state changes. The first
    time it finds the session's owner retained, it asks the binding for
    ``_retained_release_end`` and waits on it in a task. A reason is recorded on
    the owner as ``environment_end_reason``, and the owner is settled through
    the lifecycle's ordinary cleanup entry (``settle``), which publishes its
    lifecycle progress and calls ``retire`` under its own claim. A cleanup
    attempt that is running is never overtaken.
    """

    def __init__(
        self,
        setups: MutableMapping[str, Any],
        *,
        owner_type: type,
        settle: Callable[[str, Any], Awaitable[None]],
        clear_completion_marker: Callable[[str], Awaitable[None]],
    ) -> None:
        self._setups = setups
        self._owner_type = owner_type
        self._settle = settle
        self._clear_completion_marker = clear_completion_marker
        self._watches: dict[str, _Watch] = {}

    def observe(self, session_id: str) -> None:
        owner = self._setups.get(session_id)
        watch = self._watches.get(session_id)
        if watch is not None:
            stopped = watch.task is not None and watch.task.done()
            if watch.owner is owner and not stopped:
                watch.changed.set()
                return
            # The owner settled or was replaced, or its watch stopped early (for
            # example with the event loop it ran in): start again from scratch.
            del self._watches[session_id]
            self._discard(watch)
        if owner is None or type(owner) is not self._owner_type or not release_is_retained(owner):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        watch = _Watch(owner=owner, changed=asyncio.Event())
        self._watches[session_id] = watch
        registered = owner.registered_environment
        if owner.environment_end_reason is not None:
            # The environment is already known to be gone; only settlement is left.
            end: Awaitable[str] | None = _known(owner.environment_end_reason)
        else:
            try:
                end = registered.environment.binding._retained_release_end(
                    registered.bound_workspace
                )
            except Exception as error:
                logger.warning(
                    "Session %s keeps its environment %r: its binding could not say when "
                    "the environment ends (%s).",
                    session_id,
                    registered.spec.name,
                    type(error).__name__,
                )
                return
        if end is None:
            return
        watch.end = end
        watch.task = loop.create_task(
            self._run(session_id, watch, end),
            name=f"cayu-retained-environment-end-{session_id}",
        )

    @staticmethod
    def _discard(watch: _Watch) -> None:
        task = watch.task
        if task is not None and not task.done() and task is not _current_task():
            task.cancel()
        end = watch.end
        if inspect.iscoroutine(end) and inspect.getcoroutinestate(end) == inspect.CORO_CREATED:
            # Its task never started, so nothing else will close it.
            end.close()

    async def _run(self, session_id: str, watch: _Watch, end: Awaitable[str]) -> None:
        owner = watch.owner
        name = owner.registered_environment.spec.name
        try:
            reason = await end
        except Exception as error:
            logger.warning(
                "Session %s keeps its environment %r: waiting for the environment to end "
                "failed (%s).",
                session_id,
                name,
                type(error).__name__,
            )
            return
        if type(reason) is not str or not reason.strip():
            logger.warning(
                "Session %s keeps its environment %r: its binding ended the wait without a reason.",
                session_id,
                name,
            )
            return
        owner.environment_end_reason = reason
        backoff = _RETRY_FIRST_SECONDS
        while self._watches.get(session_id) is watch and self._setups.get(session_id) is owner:
            watch.changed.clear()
            if not release_is_retained(owner):
                # Another cleanup attempt holds the owner; it signals when done.
                await watch.changed.wait()
                continue
            try:
                await self._settle(session_id, owner)
            except Exception as error:
                logger.warning(
                    "Settling session %s's environment %r after it ended failed (%s).",
                    session_id,
                    name,
                    type(error).__name__,
                )
            if self._setups.get(session_id) is not owner:
                return
            # The attempt refused. Its own state change doesn't count as a reason
            # to try again: wait for another change, or back off and retry.
            watch.changed.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(watch.changed.wait(), backoff)
            backoff = min(backoff * 2, _RETRY_MAX_SECONDS)

    async def retire(self, session_id: str, owner: Any) -> bool:
        """Retire ``owner``, which the caller's cleanup attempt has claimed.

        Clears the session's pending completion, abandons the binding (dropping
        its workspace reservation), retires the environment's mutation fence
        scope and removes the owner. Returns ``False`` and keeps the owner when
        any step refuses; the watch tries again after a backoff.
        """

        registered = owner.registered_environment
        reason = owner.environment_end_reason
        try:
            # The pending completion it would have finished can't be finished
            # now; a later turn of the session must not inherit it.
            await self._clear_completion_marker(session_id)
        except Exception as error:
            logger.warning(
                "Session %s's environment %r is gone (%s), but its pending completion could "
                "not be cleared (%s).",
                session_id,
                registered.spec.name,
                reason,
                type(error).__name__,
            )
            return False
        if self._setups.get(session_id) is not owner:
            return False
        try:
            released = registered.environment.binding.abandon(registered.bound_workspace)
        except Exception as error:
            logger.warning(
                "Session %s's environment %r is gone (%s), but its binding refused to let it "
                "go (%s).",
                session_id,
                registered.spec.name,
                reason,
                type(error).__name__,
            )
            return False
        if released is False:
            logger.warning(
                "Session %s's environment %r is gone (%s), but its binding still owns it.",
                session_id,
                registered.spec.name,
                reason,
            )
            return False
        fenced = registered.workspace_mutation_fence.retire_scope()
        del self._setups[session_id]
        logger.warning(
            "Released session %s's environment %r after its release could not finish: %s. "
            "Its workspace reservation is dropped and %d unsettled workspace mutation(s) no "
            "longer fence it; anything it never published is lost.",
            session_id,
            registered.spec.name,
            reason,
            fenced,
        )
        return True


def _current_task() -> asyncio.Task[Any] | None:
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


async def _known(reason: str) -> str:
    return reason


__all__ = ["RetainedEnvironmentEndWatches", "release_is_retained"]
