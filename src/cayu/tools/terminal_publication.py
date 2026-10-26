"""Bounded cooperative scheduling and metrics for staged tool terminals."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import isfinite
from threading import Lock
from time import process_time
from typing import TypeVar, cast

from cayu._task_wait import await_shielded_task_outcome, restore_task_cancellation_requests
from cayu.sessions.invocation import SessionInvocation, copy_session_invocation

TOOL_TERMINAL_PUBLICATION_SLICE_BYTES = 256 * 1024
TOOL_TERMINAL_PUBLICATION_CAPACITY_BYTES = 4 * 1024 * 1024
TOOL_TERMINAL_PUBLICATION_MAX_OFFLOADS = 4
TOOL_TERMINAL_STAGED_CAPACITY_BYTES = 512 * 1024 * 1024
TOOL_ROUND_ADMISSION_TIMEOUT_SECONDS = 600.0
TOOL_ROUND_ADMISSION_NOTICE_SECONDS = 60.0
# A capacity wait names at most this many sessions and tool names per list.
_MAX_NAMED_HOLDER_SESSIONS = 4
_MAX_NAMED_TOOLS = 8

logger = logging.getLogger(__name__)

_ResultT = TypeVar("_ResultT")


class ToolRoundAdmissionTimeoutError(RuntimeError):
    """A deferred tool round got no publication capacity before its deadline.

    Raised before any of the round's tools run, so failing the round is safe.
    """


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
    group: tuple[str, ...]
    # False when ``maximum_bytes`` is an admission weight, not a bound every
    # stage was checked against, such as the event-limit ceiling reserved for
    # a tool that declares no bound.
    enforce_maximum: bool = True
    # Names of the round's registered tools that declare no terminal payload
    # bound. They explain a wait and take no part in the reservation itself.
    unbounded_tool_names: tuple[str, ...] = field(default=(), compare=False)


def _require_positive_seconds(value: float, name: str) -> None:
    if type(value) not in (int, float) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number of seconds.")


class ToolTerminalPublicationGovernor:
    """One fair CPU/capacity domain shared by all sessions in a ``CayuApp``."""

    def __init__(
        self,
        *,
        slice_bytes: int = TOOL_TERMINAL_PUBLICATION_SLICE_BYTES,
        capacity_bytes: int = TOOL_TERMINAL_PUBLICATION_CAPACITY_BYTES,
        max_offloads: int = TOOL_TERMINAL_PUBLICATION_MAX_OFFLOADS,
        staged_capacity_bytes: int = TOOL_TERMINAL_STAGED_CAPACITY_BYTES,
        round_admission_timeout_s: float | None = TOOL_ROUND_ADMISSION_TIMEOUT_SECONDS,
        round_admission_notice_s: float = TOOL_ROUND_ADMISSION_NOTICE_SECONDS,
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
        if round_admission_timeout_s is not None:
            _require_positive_seconds(round_admission_timeout_s, "round_admission_timeout_s")
        _require_positive_seconds(round_admission_notice_s, "round_admission_notice_s")
        self.slice_bytes = slice_bytes
        self.capacity_bytes = capacity_bytes
        self.max_offloads = max_offloads
        self.staged_capacity_bytes = staged_capacity_bytes
        # A deferred round waits for this lease before any of its tools run.
        # A round nested under a reservation its family already holds may
        # bypass the queue, so a waiter can wait while that reservation's
        # descendants keep running; the deadline turns that into a failure.
        self.round_admission_timeout_s = (
            None if round_admission_timeout_s is None else float(round_admission_timeout_s)
        )
        self.round_admission_notice_s = float(round_admission_notice_s)
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
        unbounded_tool_names: tuple[str, ...] = (),
        enforce_maximum: bool = True,
    ) -> None:
        """Reserve a deferred round before any of its tool effects execute.

        Bounded rounds share the fixed byte budget. A round larger than the
        domain, or one with no byte bound at all, receives the standard
        exclusive oversize lease so partial sibling stages can never deadlock
        behind unrelated rounds. ``unbounded_tool_names`` names the round's
        tools that declare no bound, so a wait can say why. With
        ``enforce_maximum`` false, ``maximum_bytes`` only weighs admission and
        stages are not refused for exceeding it.
        """

        await self._reserve_round(
            session_id=session_id,
            tool_round_id=tool_round_id,
            maximum_bytes=maximum_bytes,
            group=("round", session_id, tool_round_id),
            unbounded_tool_names=unbounded_tool_names,
            enforce_maximum=enforce_maximum,
        )

    async def _reserve_invocation_round(
        self,
        *,
        session_id: str,
        tool_round_id: str,
        maximum_bytes: int | None,
        invocation: SessionInvocation,
        lineage: tuple[str, ...] | None = None,
        unbounded_tool_names: tuple[str, ...] = (),
        enforce_maximum: bool = True,
    ) -> None:
        """Reserve using provenance resolved by the runtime's session owner.

        This internal entrance does not accept caller-provided parent metadata.
        Related rounds share an exclusive domain, not execution authority.

        ``lineage`` is the requesting session followed by its durable
        ancestors. A related round skips queued unrelated rounds only when one
        of those sessions already holds a reservation in the family, because
        that holder may be awaiting this round. Other related rounds, such as
        siblings whose parent holds nothing, take their turn in the queue. When
        the lineage is unknown, every related round keeps the queue bypass.
        """
        invocation = copy_session_invocation(invocation)
        if lineage is not None and (
            type(lineage) is not tuple
            or not lineage
            or lineage[0] != session_id
            or any(type(item) is not str or not item for item in lineage)
        ):
            raise ValueError("lineage must start with session_id and contain session IDs.")
        await self._reserve_round(
            session_id=session_id,
            tool_round_id=tool_round_id,
            maximum_bytes=maximum_bytes,
            group=("invocation", invocation.root_invocation_id, invocation.root_session_id),
            lineage=None if lineage is None else frozenset(lineage),
            unbounded_tool_names=unbounded_tool_names,
            enforce_maximum=enforce_maximum,
        )

    def _group_usage(self) -> dict[tuple[str, ...], tuple[int, bool]]:
        grouped: dict[tuple[str, ...], list[_RoundReservation]] = {}
        for reservation in self._round_reservations.values():
            grouped.setdefault(reservation.group, []).append(reservation)
        usage = {}
        for group, members in grouped.items():
            weight = (
                self.staged_capacity_bytes
                if any(member.maximum_bytes is None for member in members)
                else sum(member.weight for member in members)
            )
            usage[group] = (
                weight,
                any(member.exclusive for member in members) or weight > self.staged_capacity_bytes,
            )
        return usage

    async def _reserve_round(
        self,
        *,
        session_id: str,
        tool_round_id: str,
        maximum_bytes: int | None,
        group: tuple[str, ...],
        lineage: frozenset[str] | None = None,
        unbounded_tool_names: tuple[str, ...] = (),
        enforce_maximum: bool = True,
    ) -> None:
        if type(enforce_maximum) is not bool:
            raise TypeError("enforce_maximum must be a bool.")
        if maximum_bytes is not None and (type(maximum_bytes) is not int or maximum_bytes <= 0):
            raise ValueError("maximum_bytes must be a positive integer or None.")
        if type(unbounded_tool_names) is not tuple or any(
            type(name) is not str or not name for name in unbounded_tool_names
        ):
            raise ValueError("unbounded_tool_names must be a tuple of tool names.")
        key = (session_id, tool_round_id)
        exclusive = maximum_bytes is None or maximum_bytes > self.staged_capacity_bytes
        weight = self.staged_capacity_bytes if maximum_bytes is None else maximum_bytes
        requested = _RoundReservation(
            maximum_bytes=maximum_bytes,
            weight=weight,
            exclusive=exclusive,
            group=group,
            enforce_maximum=enforce_maximum,
            unbounded_tool_names=tuple(sorted(set(unbounded_tool_names))),
        )
        token = object()
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        deadline = (
            None
            if self.round_admission_timeout_s is None
            else started_at + self.round_admission_timeout_s
        )
        notice_at: float | None = started_at + self.round_admission_notice_s
        async with self._condition:
            existing = self._round_reservations.get(key)
            if existing is not None:
                if existing != requested:
                    raise RuntimeError("Tool-round capacity reservation conflicts.")
                return
            self._round_waiters.append(token)
            try:
                while True:
                    existing = self._round_reservations.get(key)
                    if existing is not None:
                        if existing != requested:
                            raise RuntimeError("Tool-round capacity reservation conflicts.")
                        self._round_waiters.remove(token)
                        self._condition.notify_all()
                        return
                    groups = self._group_usage()
                    previous_group = groups.get(group)
                    related = previous_group is not None
                    prior_weight, prior_exclusive = previous_group or (0, False)
                    members = {
                        holder: reservation
                        for holder, reservation in self._round_reservations.items()
                        if reservation.group == group
                    }
                    group_weight = (
                        self.staged_capacity_bytes
                        if maximum_bytes is None
                        or any(member.maximum_bytes is None for member in members.values())
                        else prior_weight + weight
                    )
                    # A durable invocation family is one capacity domain. Its
                    # aggregate can outgrow the budget just like one oversized
                    # round; admit it exclusively when no other family owns
                    # capacity instead of failing the child its parent awaits.
                    group_exclusive = (
                        prior_exclusive or exclusive or group_weight > self.staged_capacity_bytes
                    )
                    others = [usage for identity, usage in groups.items() if identity != group]
                    admissible = (
                        not others
                        if group_exclusive
                        else (
                            not any(is_exclusive for _, is_exclusive in others)
                            and sum(amount for amount, _ in others) + group_weight
                            <= self.staged_capacity_bytes
                        )
                    )
                    # Only a round that a held family reservation may be
                    # awaiting skips the queue. Letting every family member in
                    # ahead of queued rounds lets overlapping siblings keep the
                    # family's lease alive and starve an unrelated waiter.
                    nested = related and (
                        lineage is None
                        or any(holder_session_id in lineage for holder_session_id, _ in members)
                    )
                    if nested and not admissible:
                        # Waiting could deadlock a parent that already owns a
                        # reservation and is awaiting this child. Refuse before
                        # this round dispatches, without releasing that parent.
                        raise RuntimeError(
                            "Related tool round exceeds available publication capacity."
                        )
                    if (nested or self._round_waiters[0] is token) and admissible:
                        self._round_waiters.remove(token)
                        self._round_reservations[key] = requested
                        self._reserved_round_bytes += group_weight - prior_weight
                        self._maximum_reserved_round_bytes = max(
                            self._maximum_reserved_round_bytes,
                            self._reserved_round_bytes,
                        )
                        self._condition.notify_all()
                        return
                    now = loop.time()
                    if deadline is not None and now >= deadline:
                        # The error reaches this session's durable events, so
                        # only the operator log names the sessions holding the
                        # lease; they may belong to another tenant.
                        logger.warning(
                            "Tool round failed waiting for terminal publication capacity: "
                            "session_id=%s tool_round_id=%s waited_s=%.0f %s",
                            session_id,
                            tool_round_id,
                            now - started_at,
                            self._capacity_holders(requested, groups, name_holders=True),
                        )
                        raise ToolRoundAdmissionTimeoutError(
                            f"Tool round waited {now - started_at:.0f}s for terminal "
                            "publication capacity before running any tool; "
                            f"{self._capacity_holders(requested, groups, name_holders=False)}"
                        )
                    if notice_at is not None and now >= notice_at:
                        notice_at = None
                        logger.warning(
                            "Tool round is still waiting for terminal publication capacity: "
                            "session_id=%s tool_round_id=%s waited_s=%.0f %s",
                            session_id,
                            tool_round_id,
                            now - started_at,
                            self._capacity_holders(requested, groups, name_holders=True),
                        )
                    wake_at = min(
                        (moment for moment in (deadline, notice_at) if moment is not None),
                        default=None,
                    )
                    # On timeout the lock is held again; re-check before failing.
                    with suppress(TimeoutError):
                        async with asyncio.timeout_at(wake_at):
                            await self._condition.wait()
            except BaseException:
                with suppress(ValueError):
                    self._round_waiters.remove(token)
                self._condition.notify_all()
                raise

    def _capacity_holders(
        self,
        waiting: _RoundReservation,
        usage: Mapping[tuple[str, ...], tuple[int, bool]],
        *,
        name_holders: bool,
    ) -> str:
        """Describe, without content, what holds the lease a round waits for.

        Beside the counts it names the tools, the exclusive holders' and the
        waiting round's, that declare no bound, because each call to such a
        tool reserves the largest terminal result Cayu can store. Registered
        tool names belong to the application. Holder session IDs and tool names
        are included only when ``name_holders`` is set, for the operator log.
        """

        others = [value for group, value in usage.items() if group != waiting.group]
        description = (
            f"other_reservation_groups={len(others)} "
            f"exclusive_groups={sum(is_exclusive for _, is_exclusive in others)} "
            f"reserved_round_bytes={self._reserved_round_bytes} "
            f"round_waiters={len(self._round_waiters)}"
        )
        holders = [
            (holder_session_id, reservation)
            for (holder_session_id, _), reservation in self._round_reservations.items()
            if reservation.group != waiting.group and usage[reservation.group][1]
        ]
        holder_tools = {name for _, item in holders for name in item.unbounded_tool_names}
        if holders and name_holders:
            description += (
                " exclusive_holder_sessions="
                f"{_bounded_names({session for session, _ in holders}, _MAX_NAMED_HOLDER_SESSIONS)}"
            )
            description += (
                f" exclusive_holder_unbounded_tools={_bounded_names(holder_tools) or 'none'}"
            )
        if waiting.unbounded_tool_names:
            description += (
                f" waiting_round_unbounded_tools={_bounded_names(waiting.unbounded_tool_names)}"
            )
        if (name_holders and holder_tools) or waiting.unbounded_tool_names:
            description += (
                ". Each call to a tool without ToolSpec.max_terminal_payload_bytes reserves "
                "the largest terminal result Cayu can store; declare the largest result each "
                "named tool returns, or raise ToolExecutionConfig."
                "tool_terminal_staged_capacity_bytes."
            )
        return description

    def release_round(self, *, session_id: str, tool_round_id: str) -> None:
        """Release one process-local round lease after its stages drain."""

        reservation = self._round_reservations.pop((session_id, tool_round_id), None)
        if reservation is None:
            return
        self._reserved_round_bytes = sum(amount for amount, _ in self._group_usage().values())
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
                            and reservation.enforce_maximum
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
                            and reservation.enforce_maximum
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
                    and reservation.enforce_maximum
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
        groups = self._group_usage()
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
                    groups[reservation.group][1]
                    for reservation in self._round_reservations.values()
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


def _bounded_names(names: set[str] | tuple[str, ...], limit: int = _MAX_NAMED_TOOLS) -> str:
    ordered = sorted(set(names))
    listed = ",".join(ordered[:limit])
    if len(ordered) > limit:
        listed += f",+{len(ordered) - limit}_more"
    return listed


def utc_now() -> datetime:
    """Small injectable-default companion kept here for metrics tests."""

    return datetime.now(UTC)
