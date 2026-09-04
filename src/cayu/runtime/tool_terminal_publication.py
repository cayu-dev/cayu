"""Bounded cooperative scheduling and metrics for staged tool terminals."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from threading import Lock
from time import process_time
from typing import TypeVar, cast

from cayu._task_wait import await_shielded_task_outcome, restore_task_cancellation_requests

TOOL_TERMINAL_PUBLICATION_SLICE_BYTES = 256 * 1024
TOOL_TERMINAL_PUBLICATION_CAPACITY_BYTES = 4 * 1024 * 1024
TOOL_TERMINAL_PUBLICATION_MAX_OFFLOADS = 4
TOOL_TERMINAL_STAGED_CAPACITY_BYTES = 128 * 1024 * 1024

_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True, slots=True)
class ToolTerminalPublicationMetricsSnapshot:
    """Content-free process-local staged-terminal publication measurements."""

    staged_count: int
    staged_bytes: int
    maximum_staged_count: int
    maximum_staged_bytes: int
    oldest_staged_delay_s: float
    validation_cpu_s: float
    publication_count: int
    publication_lag_total_s: float
    publication_lag_max_s: float
    cooperative_yields: int
    oversized_offloads: int
    configured_slice_bytes: int
    configured_capacity_bytes: int
    configured_max_offloads: int
    configured_staged_capacity_bytes: int
    reserved_round_bytes: int
    maximum_reserved_round_bytes: int
    active_round_reservations: int
    round_reservation_waiters: int
    active_exclusive_rounds: int
    active_publication_bytes: int
    maximum_active_publication_bytes: int


@dataclass(frozen=True, slots=True)
class _Stage:
    payload_bytes: int
    effect_completed_at: datetime


@dataclass(frozen=True, slots=True)
class _RoundReservation:
    maximum_bytes: int | None
    weight: int
    exclusive: bool


class ToolTerminalPublicationGovernor:
    """One fair CPU/capacity domain shared by all sessions in a ``CayuApp``."""

    def __init__(
        self,
        *,
        slice_bytes: int = TOOL_TERMINAL_PUBLICATION_SLICE_BYTES,
        capacity_bytes: int = TOOL_TERMINAL_PUBLICATION_CAPACITY_BYTES,
        max_offloads: int = TOOL_TERMINAL_PUBLICATION_MAX_OFFLOADS,
        staged_capacity_bytes: int = TOOL_TERMINAL_STAGED_CAPACITY_BYTES,
    ) -> None:
        for value, name in (
            (slice_bytes, "slice_bytes"),
            (capacity_bytes, "capacity_bytes"),
            (max_offloads, "max_offloads"),
            (staged_capacity_bytes, "staged_capacity_bytes"),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if slice_bytes > capacity_bytes:
            raise ValueError("slice_bytes must not exceed capacity_bytes.")
        self.slice_bytes = slice_bytes
        self.capacity_bytes = capacity_bytes
        self.max_offloads = max_offloads
        self.staged_capacity_bytes = staged_capacity_bytes
        self._condition = asyncio.Condition()
        self._small_waiters: deque[object] = deque()
        self._oversized_waiters: deque[object] = deque()
        self._active_bytes = 0
        self._maximum_active_bytes = 0
        self._active_offloads = 0
        self._round_waiters: deque[object] = deque()
        self._round_reservations: dict[tuple[str, str], _RoundReservation] = {}
        self._round_staged_bytes: dict[tuple[str, str], int] = {}
        self._stage_round_keys: dict[tuple[str, str], tuple[str, str]] = {}
        self._reserved_round_bytes = 0
        self._maximum_reserved_round_bytes = 0
        self._metrics_lock = Lock()
        self._stages: dict[tuple[str, str], _Stage] = {}
        self._staged_bytes = 0
        self._maximum_staged_count = 0
        self._maximum_staged_bytes = 0
        self._validation_cpu_s = 0.0
        self._publication_count = 0
        self._publication_lag_total_s = 0.0
        self._publication_lag_max_s = 0.0
        self._cooperative_yields = 0
        self._oversized_offloads = 0
        self._notification_tasks: set[asyncio.Task[None]] = set()

    async def reserve_round(
        self,
        *,
        session_id: str,
        tool_round_id: str,
        maximum_bytes: int | None,
    ) -> None:
        """Reserve a deferred round before any of its tool effects execute.

        Declared rounds share the fixed byte budget. A declared round larger
        than the domain, or a round containing an unbounded tool, receives the
        standard exclusive oversize lease so partial sibling stages can never
        deadlock behind unrelated rounds.
        """

        if maximum_bytes is not None and (type(maximum_bytes) is not int or maximum_bytes <= 0):
            raise ValueError("maximum_bytes must be a positive integer or None.")
        key = (session_id, tool_round_id)
        exclusive = maximum_bytes is None or maximum_bytes > self.staged_capacity_bytes
        weight = self.staged_capacity_bytes if maximum_bytes is None else maximum_bytes
        requested = _RoundReservation(
            maximum_bytes=maximum_bytes,
            weight=weight,
            exclusive=exclusive,
        )
        token = object()
        async with self._condition:
            existing = self._round_reservations.get(key)
            if existing is not None:
                if existing != requested:
                    raise RuntimeError("Tool-round capacity reservation conflicts.")
                return
            self._round_waiters.append(token)
            try:
                while True:
                    no_exclusive = not any(
                        reservation.exclusive for reservation in self._round_reservations.values()
                    )
                    admissible = (
                        not self._round_reservations
                        if exclusive
                        else (
                            no_exclusive
                            and self._reserved_round_bytes + weight <= self.staged_capacity_bytes
                        )
                    )
                    if self._round_waiters[0] is token and admissible:
                        self._round_waiters.popleft()
                        self._round_reservations[key] = requested
                        self._reserved_round_bytes += weight
                        self._maximum_reserved_round_bytes = max(
                            self._maximum_reserved_round_bytes,
                            self._reserved_round_bytes,
                        )
                        self._condition.notify_all()
                        return
                    await self._condition.wait()
            except BaseException:
                with suppress(ValueError):
                    self._round_waiters.remove(token)
                self._condition.notify_all()
                raise

    def release_round(self, *, session_id: str, tool_round_id: str) -> None:
        """Release one process-local round lease after its stages drain."""

        reservation = self._round_reservations.pop((session_id, tool_round_id), None)
        if reservation is None:
            return
        self._reserved_round_bytes -= reservation.weight
        self._round_staged_bytes.pop((session_id, tool_round_id), None)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        notification = loop.create_task(self._notify_waiters())
        self._notification_tasks.add(notification)
        notification.add_done_callback(self._notification_tasks.discard)

    async def _notify_waiters(self) -> None:
        async with self._condition:
            self._condition.notify_all()

    def stage(
        self,
        *,
        session_id: str,
        event_id: str,
        payload_bytes: int,
        effect_completed_at: datetime,
        tool_round_id: str | None = None,
    ) -> None:
        """Observe one durable stage idempotently without retaining its content."""

        self._record_stage(
            session_id=session_id,
            event_id=event_id,
            payload_bytes=payload_bytes,
            effect_completed_at=effect_completed_at,
            tool_round_id=tool_round_id,
            reconcile_payload=False,
        )

    def reconcile_stage(
        self,
        *,
        session_id: str,
        event_id: str,
        payload_bytes: int,
        effect_completed_at: datetime,
        tool_round_id: str | None = None,
    ) -> None:
        """Reconcile content-free metrics with an authoritative durable stage."""

        self._record_stage(
            session_id=session_id,
            event_id=event_id,
            payload_bytes=payload_bytes,
            effect_completed_at=effect_completed_at,
            tool_round_id=tool_round_id,
            reconcile_payload=True,
        )

    def _record_stage(
        self,
        *,
        session_id: str,
        event_id: str,
        payload_bytes: int,
        effect_completed_at: datetime,
        tool_round_id: str | None,
        reconcile_payload: bool,
    ) -> None:
        """Record or reconcile one stage under the owning round reservation."""

        if type(payload_bytes) is not int or payload_bytes < 0:
            raise ValueError("payload_bytes must be a non-negative integer.")
        if effect_completed_at.tzinfo is None:
            raise ValueError("effect_completed_at must be timezone-aware.")
        key = (session_id, event_id)
        with self._metrics_lock:
            previous = self._stages.get(key)
            if previous is not None:
                if previous.effect_completed_at != effect_completed_at or (
                    previous.payload_bytes != payload_bytes and not reconcile_payload
                ):
                    raise RuntimeError("Staged terminal metrics conflict with durable identity.")
                previous_payload_bytes = previous.payload_bytes
                if tool_round_id is not None:
                    round_key = (session_id, tool_round_id)
                    previous_round_key = self._stage_round_keys.get(key)
                    if previous_round_key is not None and previous_round_key != round_key:
                        raise RuntimeError(
                            "Staged terminal belongs to a different round reservation."
                        )
                    if previous_round_key is None:
                        reservation = self._round_reservations.get(round_key)
                        if reservation is None:
                            raise RuntimeError("Staged terminal has no owning round reservation.")
                        round_staged_bytes = self._round_staged_bytes.get(round_key, 0)
                        if (
                            reservation.maximum_bytes is not None
                            and round_staged_bytes + payload_bytes > reservation.maximum_bytes
                        ):
                            raise RuntimeError(
                                "Staged terminal exceeds its declared round payload reservation."
                            )
                        self._round_staged_bytes[round_key] = round_staged_bytes + payload_bytes
                        self._stage_round_keys[key] = round_key
                    elif previous_payload_bytes != payload_bytes:
                        reservation = self._round_reservations.get(round_key)
                        if reservation is None:
                            raise RuntimeError("Staged terminal has no owning round reservation.")
                        round_staged_bytes = self._round_staged_bytes.get(round_key, 0)
                        reconciled_round_bytes = (
                            round_staged_bytes - previous_payload_bytes + payload_bytes
                        )
                        if (
                            reservation.maximum_bytes is not None
                            and reconciled_round_bytes > reservation.maximum_bytes
                        ):
                            raise RuntimeError(
                                "Staged terminal exceeds its declared round payload reservation."
                            )
                        self._round_staged_bytes[round_key] = reconciled_round_bytes
                elif previous_payload_bytes != payload_bytes:
                    previous_round_key = self._stage_round_keys.get(key)
                    if previous_round_key is not None:
                        raise RuntimeError(
                            "Staged terminal reconciliation lost its round reservation."
                        )
                if previous_payload_bytes != payload_bytes:
                    self._stages[key] = _Stage(payload_bytes, effect_completed_at)
                    self._staged_bytes += payload_bytes - previous_payload_bytes
                    self._maximum_staged_bytes = max(
                        self._maximum_staged_bytes,
                        self._staged_bytes,
                    )
                return
            if tool_round_id is not None:
                round_key = (session_id, tool_round_id)
                reservation = self._round_reservations.get(round_key)
                if reservation is None:
                    raise RuntimeError("Staged terminal has no owning round reservation.")
                round_staged_bytes = self._round_staged_bytes.get(round_key, 0)
                if (
                    reservation.maximum_bytes is not None
                    and round_staged_bytes + payload_bytes > reservation.maximum_bytes
                ):
                    raise RuntimeError(
                        "Staged terminal exceeds its declared round payload reservation."
                    )
                self._round_staged_bytes[round_key] = round_staged_bytes + payload_bytes
                self._stage_round_keys[key] = round_key
            self._stages[key] = _Stage(payload_bytes, effect_completed_at)
            self._staged_bytes += payload_bytes
            self._maximum_staged_count = max(self._maximum_staged_count, len(self._stages))
            self._maximum_staged_bytes = max(
                self._maximum_staged_bytes,
                self._staged_bytes,
            )

    def published(self, *, session_id: str, event_id: str, published_at: datetime) -> None:
        if published_at.tzinfo is None:
            raise ValueError("published_at must be timezone-aware.")
        with self._metrics_lock:
            key = (session_id, event_id)
            stage = self._stages.pop(key, None)
            if stage is None:
                return
            round_key = self._stage_round_keys.pop(key, None)
            if round_key is not None:
                retained = self._round_staged_bytes.get(round_key, 0) - stage.payload_bytes
                if retained > 0:
                    self._round_staged_bytes[round_key] = retained
                else:
                    self._round_staged_bytes.pop(round_key, None)
            self._staged_bytes -= stage.payload_bytes
            lag = max((published_at - stage.effect_completed_at).total_seconds(), 0.0)
            self._publication_count += 1
            self._publication_lag_total_s += lag
            self._publication_lag_max_s = max(self._publication_lag_max_s, lag)

    def record_validation_cpu(self, seconds: float) -> None:
        if not isfinite(seconds) or seconds < 0:
            return
        with self._metrics_lock:
            self._validation_cpu_s += seconds

    def snapshot(self) -> ToolTerminalPublicationMetricsSnapshot:
        now = datetime.now(UTC)
        with self._metrics_lock:
            oldest = max(
                (
                    max((now - item.effect_completed_at).total_seconds(), 0.0)
                    for item in self._stages.values()
                ),
                default=0.0,
            )
            return ToolTerminalPublicationMetricsSnapshot(
                staged_count=len(self._stages),
                staged_bytes=self._staged_bytes,
                maximum_staged_count=self._maximum_staged_count,
                maximum_staged_bytes=self._maximum_staged_bytes,
                oldest_staged_delay_s=oldest,
                validation_cpu_s=self._validation_cpu_s,
                publication_count=self._publication_count,
                publication_lag_total_s=self._publication_lag_total_s,
                publication_lag_max_s=self._publication_lag_max_s,
                cooperative_yields=self._cooperative_yields,
                oversized_offloads=self._oversized_offloads,
                configured_slice_bytes=self.slice_bytes,
                configured_capacity_bytes=self.capacity_bytes,
                configured_max_offloads=self.max_offloads,
                configured_staged_capacity_bytes=self.staged_capacity_bytes,
                reserved_round_bytes=self._reserved_round_bytes,
                maximum_reserved_round_bytes=self._maximum_reserved_round_bytes,
                active_round_reservations=len(self._round_reservations),
                round_reservation_waiters=len(self._round_waiters),
                active_exclusive_rounds=sum(
                    reservation.exclusive for reservation in self._round_reservations.values()
                ),
                active_publication_bytes=self._active_bytes,
                maximum_active_publication_bytes=self._maximum_active_bytes,
            )

    async def run_cpu(
        self,
        estimated_bytes: int,
        operation: Callable[[], _ResultT],
    ) -> _ResultT:
        """Run one bounded slice fairly; move oversized work off the event loop."""

        if type(estimated_bytes) is not int or estimated_bytes < 0:
            raise ValueError("estimated_bytes must be a non-negative integer.")
        if not callable(operation):
            raise TypeError("operation must be callable.")
        oversized = estimated_bytes > self.slice_bytes
        # One offloaded operation occupies one configured scheduling slice,
        # independent of its total payload size. Its thread remains owned until
        # completion, while the remaining capacity stays available to bounded
        # synchronous work from unrelated sessions.
        weight = self.slice_bytes if oversized else max(estimated_bytes, 1)
        token = object()
        waiters = self._oversized_waiters if oversized else self._small_waiters
        async with self._condition:
            waiters.append(token)
            try:
                while (
                    waiters[0] is not token
                    or self._active_bytes + weight > self.capacity_bytes
                    or (oversized and self._active_offloads >= self.max_offloads)
                ):
                    await self._condition.wait()
                waiters.popleft()
                with self._metrics_lock:
                    self._active_bytes += weight
                    self._maximum_active_bytes = max(
                        self._maximum_active_bytes,
                        self._active_bytes,
                    )
                self._active_offloads += int(oversized)
                self._condition.notify_all()
            except BaseException:
                with suppress(ValueError):
                    waiters.remove(token)
                self._condition.notify_all()
                raise
        try:
            # Always hand the loop to already-ready heartbeats, interrupts, and
            # unrelated sessions before consuming the next CPU slice.
            await asyncio.sleep(0)
            started = process_time()
            if oversized:
                worker = asyncio.create_task(asyncio.to_thread(operation))
                outcome = await await_shielded_task_outcome(worker)
                elapsed = max(process_time() - started, 0.0)
                self.record_validation_cpu(elapsed)
                if outcome.error is not None:
                    raise outcome.error
                if outcome.cancellation is not None:
                    restore_task_cancellation_requests(
                        outcome.cancellation_requests_consumed,
                        cancellation=outcome.cancellation,
                    )
                    raise outcome.cancellation
                result = cast("_ResultT", outcome.result)
            else:
                result = operation()
                self.record_validation_cpu(max(process_time() - started, 0.0))
            with self._metrics_lock:
                self._cooperative_yields += 1
                self._oversized_offloads += int(oversized)
            return result
        finally:
            async with self._condition:
                with self._metrics_lock:
                    self._active_bytes -= weight
                self._active_offloads -= int(oversized)
                self._condition.notify_all()


def utc_now() -> datetime:
    """Small injectable-default companion kept here for metrics tests."""

    return datetime.now(UTC)
