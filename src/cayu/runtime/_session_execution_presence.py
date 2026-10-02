"""One bounded heartbeat per local session epoch, independent of model/tool progress."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress
from dataclasses import dataclass
from math import isfinite
from uuid import uuid4

from cayu.sessions._process_liveness import current_process_identity
from cayu.sessions.execution import (
    SessionExecutionConfig,
    bind_execution_progress,
    current_execution_owner_kind,
)

_LOG = logging.getLogger(__name__)
_PROCESS_PID = os.getpid()
_PROCESS_ID = uuid4().hex
_RELEASE_WAIT_SECONDS = 5.0


def process_owner_id():
    global _PROCESS_PID, _PROCESS_ID
    if os.getpid() != _PROCESS_PID:
        _PROCESS_PID, _PROCESS_ID = os.getpid(), uuid4().hex
    return _PROCESS_ID


@dataclass
class _Presence:
    owner: object
    stop: asyncio.Event
    task: asyncio.Task[None]


class SessionExecutionPresence:
    def __init__(self, store, config: SessionExecutionConfig):
        self.store = store
        self.config = config
        self.groups = {}
        self.progress = {}

    def bind(self, session_id):
        self.progress.setdefault(session_id, (0, "publishing"))
        bind_execution_progress(self.progress)

    def discard_unused_progress(self, session_id):
        if not any(key[0] == session_id for key in self.groups):
            self.progress.pop(session_id, None)

    async def ensure(self, session, *, task_id=None):
        if not self.store.supports_session_execution:
            return
        key = session.id, session.instance_id, session.run_epoch
        if key in self.groups:
            return
        self.bind(session.id)
        try:
            owner = await self.store._claim_session_execution(
                session.id,
                token=uuid4().hex,
                owner_id=process_owner_id(),
                process_identity=current_process_identity(),
                owner_kind=current_execution_owner_kind(task_id=task_id),
                owner_label=self.config.owner_label,
                lease_seconds=self.config.lease_seconds,
            )
        except Exception as exc:
            # Presence is observational: a failed claim must not abort the run
            # or recovery that owns the session. The run fence still guards it.
            self.discard_unused_progress(session.id)
            _LOG.warning("Session execution presence claim failed: %s", type(exc).__name__)
            return
        except BaseException:
            self.discard_unused_progress(session.id)
            raise
        if owner is None:
            self.discard_unused_progress(session.id)
            return
        stop = asyncio.Event()
        from cayu.sessions.base import _SESSION_RUN_FENCE_OWNERS

        fence_owner = (_SESSION_RUN_FENCE_OWNERS.get() or {}).get(session.id)
        if fence_owner is not None and fence_owner.run_epoch == session.run_epoch:
            loop = asyncio.get_running_loop()

            def retired():
                # A closed loop already owns cancellation of its heartbeat.
                with suppress(RuntimeError):
                    loop.call_soon_threadsafe(stop.set)

            fence_owner.on_retire(retired)
        task = asyncio.create_task(self._heartbeat(key, owner, stop))
        self.groups[key] = _Presence(owner, stop, task)

    def stop(self, session_id, *, run_epoch):
        for key, group in tuple(self.groups.items()):
            if key[0] == session_id and key[2] == run_epoch:
                group.stop.set()

    async def stop_and_wait(self, session_id, *, run_epoch):
        """Wait boundedly; unfinished release stays owned until actual settlement."""
        tasks = []
        for key, group in tuple(self.groups.items()):
            if key[0] == session_id and key[2] == run_epoch:
                group.stop.set()
                tasks.append(group.task)
        if tasks:
            # Waiting may be cancelled, but cancellation does not prove the
            # store's release stopped. Neither timeout nor caller cancellation
            # cancels the heartbeat or removes its exact entry from groups.
            cancellation = None
            pending = set(tasks)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + _RELEASE_WAIT_SECONDS
            while pending:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    _, pending = await asyncio.wait(pending, timeout=remaining)
                except asyncio.CancelledError as error:
                    if cancellation is None:
                        cancellation = error
            if pending:
                _LOG.warning("Session execution presence release remains pending.")
            failures = []
            for task in tasks:
                if not task.done():
                    continue
                try:
                    task.result()
                except BaseException as error:
                    failures.append(error)
            failure = (
                None
                if not failures
                else failures[0]
                if len(failures) == 1
                else BaseExceptionGroup("Execution presence release failed.", failures)
            )
            if cancellation is not None:
                raise cancellation from failure
            if failure is not None:
                raise failure

    async def drain(self, *, timeout_s: float) -> bool:
        """Observe stopped owners only; never stop live execution or cancel writes."""
        if type(timeout_s) not in {int, float} or not isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be a finite positive number.")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            tasks = {group.task for group in self.groups.values() if group.stop.is_set()}
            if not tasks:
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            done, _ = await asyncio.wait(
                tasks, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                return False
            for task in done:
                task.result()

    async def _heartbeat(self, key, owner, stop):
        generation = -1
        try:
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), self.config.heartbeat_interval_seconds)
                    break
                except TimeoutError:
                    pass
                latest, kind = self.progress.get(owner.session_id, (generation, "publishing"))
                try:
                    renewed = await self.store._renew_session_execution(
                        owner,
                        lease_seconds=self.config.lease_seconds,
                        progress_kind=kind if latest != generation else None,
                    )
                    if renewed is None and not stop.is_set():
                        renewed = await self.store._claim_session_execution(
                            owner.session_id,
                            token=owner.token,
                            owner_id=owner.owner_id,
                            process_identity=owner.process_identity,
                            owner_kind=owner.owner_kind,
                            owner_label=owner.owner_label,
                            lease_seconds=self.config.lease_seconds,
                        )
                except Exception as exc:
                    # Keep trying past lease expiry: once the store answers again,
                    # renewal fails and the epoch-checked reclaim above decides
                    # whether this run still owns the session.
                    _LOG.warning("Session execution heartbeat failed: %s", type(exc).__name__)
                    continue
                if renewed is None:
                    break
                generation, owner = latest, renewed
        finally:
            if stop.is_set():
                try:
                    await self.store._release_session_execution(owner)
                except Exception as exc:
                    _LOG.warning(
                        "Session execution presence release failed: %s", type(exc).__name__
                    )
            self.groups.pop(key, None)
            self.discard_unused_progress(owner.session_id)
