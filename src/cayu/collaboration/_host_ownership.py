"""Process-local supervision for explicitly started collaboration servicing.

This is not a dispatch authority or a queue. Native role adapters must retain
their exact durable responsibility before starting work here, and reauthenticate
inside the operation. An observation deadline never releases that responsibility.
No task is created by construction.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal

from cayu._exception_groups import exception_cause, failure_control_cause, set_exception_cause
from cayu.collaboration._host_reads import HostReadResult, _completed_result

Role = Literal["execution", "maintenance"]
Phase = Literal["uncertain", "active"]


class HostCapacityExceeded(RuntimeError):
    """A local slot/byte reservation was refused before invoking the adapter."""


def _positive_integer(value: int, maximum: int, name: str) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} is outside its supported bounds.")


def _seconds(value: float) -> None:
    if type(value) not in (int, float) or not 0 < value <= 3600 or not isfinite(value):
        raise ValueError("Observation timeout must be finite and between zero and one hour.")


@dataclass(frozen=True, slots=True)
class HostOwnershipLimits:
    execution_slots: int
    maintenance_slots: int
    retained_operations: int
    retained_bytes: int

    def __post_init__(self) -> None:
        _positive_integer(self.execution_slots, 64, "execution_slots")
        _positive_integer(self.maintenance_slots, 64, "maintenance_slots")
        _positive_integer(self.retained_operations, 128, "retained_operations")
        _positive_integer(self.retained_bytes, 8 * 1024 * 1024, "retained_bytes")
        if self.execution_slots + self.maintenance_slots > self.retained_operations:
            raise ValueError("Retained capacity must cover execution and maintenance slots.")


@dataclass(frozen=True, slots=True)
class HostOperationIdentity:
    """Opaque exact-intent commitment; never serialized access or model content."""

    key: str
    commitment: str

    def __post_init__(self) -> None:
        if type(self.key) is not str or not self.key or len(self.key.encode("utf-8")) > 256:
            raise ValueError("Host operation key must be a bounded nonempty string.")
        if (
            type(self.commitment) is not str
            or len(self.commitment) != 64
            or any(character not in "0123456789abcdef" for character in self.commitment)
        ):
            raise ValueError("Host operation requires a canonical SHA-256 commitment.")


@dataclass(frozen=True, slots=True)
class HostReconciledResult:
    """Native adapter has exact handoff evidence despite an earlier failure.

    This is private plumbing, never caller-supplied settlement authority. The
    original exception must reach the observer after its local slot is released.
    """

    value: Any
    failure: Exception

    def __post_init__(self):
        if not isinstance(self.failure, Exception):
            raise TypeError("Recovered host failures must not demote control signals.")


@dataclass(frozen=True, slots=True)
class HostOwnedOutcome:
    identity: HostOperationIdentity
    value: Any = None
    error: BaseException | None = None
    reconciled: bool = False


@dataclass(frozen=True, slots=True)
class HostOwnedObservation:
    completed: tuple[HostOwnedOutcome, ...]
    active: tuple[HostOperationIdentity, ...]
    uncertain: tuple[HostOperationIdentity, ...]


@dataclass(slots=True)
class _Operation:
    identity: HostOperationIdentity
    role: Role
    reserved_bytes: int
    stop: asyncio.Event
    task: asyncio.Task[HostOwnedOutcome]
    phase: Phase = "uncertain"
    reported: bool = False
    control_reported: bool = False
    outcome: HostOwnedOutcome | None = None
    dispatch_deadline: float | None = None
    window_changed: asyncio.Event = field(default_factory=asyncio.Event)
    reconcile: Callable[[], Coroutine[Any, Any, Any | None]] | None = None
    reconciliation: asyncio.Task[HostReadResult] | None = None


class HostOwnership:
    """Retain tasks, including native supervision, beyond observer lifetime.

    A role operation may return only after it has either positively settled or
    transferred unresolved work to its durable native recovery owner. The stop
    event requests bounded owner-specific stopping; it is not an abort receipt.
    Completed results remain capacity-counted until the native role adapter
    confirms settlement/handoff, not merely because its Python task returned.
    """

    def __init__(self, limits: HostOwnershipLimits) -> None:
        if type(limits) is not HostOwnershipLimits:
            raise TypeError("Host ownership requires exact finite limits.")
        self.limits = limits
        self._operations: dict[str, _Operation] = {}
        self._closing = False
        self._observing = False

    @property
    def closing(self) -> bool:
        return self._closing

    @property
    def pending(self) -> int:
        return len(self._operations)

    def has_slot(self, role: Role) -> bool:
        """Scheduling hint only; start still checks exact intent and byte capacity.

        In-flight and completed-but-unacknowledged operations both occupy slots.
        Avoid querying more candidates when none can be serviced in this role.
        """
        if role not in ("execution", "maintenance"):
            raise ValueError("Unknown host operation role.")
        ceiling = (
            self.limits.execution_slots if role == "execution" else self.limits.maintenance_slots
        )
        return (
            not self._closing
            and len(self._operations) < self.limits.retained_operations
            and sum(operation.role == role for operation in self._operations.values()) < ceiling
        )

    def start(
        self,
        identity: HostOperationIdentity,
        *,
        role: Role,
        reserved_bytes: int,
        action: Callable[[asyncio.Event], Coroutine[Any, Any, Any]],
        reconcile: Callable[[], Coroutine[Any, Any, Any | None]] | None = None,
    ) -> bool:
        """Join an identical local operation or start one under reserved capacity.

        The callable is internal, not a public adapter qualification hook. Its
        coroutine is created inside the retained task, after capacity admission.
        The result is False only when the exact operation was already retained.
        """
        if type(identity) is not HostOperationIdentity:
            raise TypeError("Host operation identity must be validated.")
        if role not in ("execution", "maintenance"):
            raise ValueError("Unknown host operation role.")
        _positive_integer(reserved_bytes, 128 * 1024, "reserved_bytes")
        if not callable(action):
            raise TypeError("Host operation requires an internal callable.")
        if reconcile is not None and not callable(reconcile):
            raise TypeError("Host reconciliation requires an internal read adapter.")
        if self._closing:
            raise RuntimeError("Collaboration host is closing.")
        existing = self._operations.get(identity.key)
        if existing is not None:
            if (
                existing.identity != identity
                or existing.role != role
                or existing.reserved_bytes != reserved_bytes
                or (existing.reconcile is None) != (reconcile is None)
            ):
                raise ValueError("Retained host operation conflicts with expected intent.")
            return False
        if not self.has_slot(role):
            raise HostCapacityExceeded("Collaboration host operation capacity is full.")
        # Partition bytes as well as counts: ordinary execution cannot consume
        # the capacity needed to retain mandatory maintenance outcomes.
        total_slots = self.limits.execution_slots + self.limits.maintenance_slots
        execution_bytes = self.limits.retained_bytes * self.limits.execution_slots // total_slots
        byte_ceiling = (
            execution_bytes if role == "execution" else self.limits.retained_bytes - execution_bytes
        )
        used = sum(
            operation.reserved_bytes
            for operation in self._operations.values()
            if operation.role == role
        )
        if used + reserved_bytes > byte_ceiling:
            raise HostCapacityExceeded("Collaboration host retained-byte capacity is full.")
        stop = asyncio.Event()

        async def own() -> HostOwnedOutcome:
            try:
                # Also support eager task factories: install ownership before
                # the adapter can report native admission or start dispatch.
                await asyncio.sleep(0)
                result = await action(stop)
                if type(result) is HostReconciledResult:
                    return HostOwnedOutcome(
                        identity, value=result.value, error=result.failure, reconciled=True
                    )
                return HostOwnedOutcome(identity, value=result)
            except BaseException as error:
                # The adapter owns cleanup and error ordering. Preserve that
                # exact graph once; do not log, stringify, or flatten it here.
                outcome = HostOwnedOutcome(identity, error=error)
                self._operations[identity.key].outcome = outcome
                if isinstance(error, asyncio.CancelledError):
                    raise
                return outcome

        coroutine = own()
        try:
            task = asyncio.create_task(coroutine, name="cayu-collaboration-host-owned")
        except BaseException:
            coroutine.close()
            raise
        self._operations[identity.key] = _Operation(
            identity, role, reserved_bytes, stop, task, reconcile=reconcile
        )
        return True

    @staticmethod
    def _outcome(operation: _Operation) -> HostOwnedOutcome:
        # Task.result() on a cancelled task need not return the same exception
        # object on repeated reads. Retain the original graph, including when
        # cancellation prevented first entry into the supervision coroutine.
        if operation.outcome is None:
            result = _completed_result(operation.task)
            operation.outcome = (
                result
                if isinstance(result, HostOwnedOutcome)
                else HostOwnedOutcome(operation.identity, error=result.error)
            )
        return operation.outcome

    def _start_reconciliations(self) -> None:
        """One retained exact reconciliation per failed turn during observation.

        The original effect is never called again. Owner readback may finish a
        source acknowledgement of a proven receiving commit, using the turn's
        existing slot/byte reservation even after this observer is cancelled.
        Closing permits this settlement, not new effects or dispatch windows.
        """
        for operation in self._operations.values():
            if operation.reconcile is None or not operation.task.done() or not operation.reported:
                continue
            outcome = self._outcome(operation)
            if outcome.error is None or getattr(outcome, "reconciled", False):
                continue
            if operation.reconciliation is not None:
                if not operation.reconciliation.done():
                    continue
                observed = _completed_result(operation.reconciliation)
                if observed.error is not None or observed.value is not None:
                    # Errors await collection; positive evidence awaits the
                    # normal role-specific acknowledgement. Neither is dropped.
                    continue
                operation.reconciliation = None

            async def read(operation=operation):
                try:
                    await asyncio.sleep(0)
                    assert operation.reconcile is not None
                    return HostReadResult(value=await operation.reconcile())
                except BaseException as error:
                    return HostReadResult(error=error)

            coroutine = read()
            try:
                operation.reconciliation = asyncio.create_task(
                    coroutine, name="cayu-collaboration-host-reconciliation"
                )
            except BaseException:
                coroutine.close()
                raise

    def take_control_failures(self) -> list[BaseException]:
        """Deliver owned control signals once, independently of native settlement."""
        failures = []
        for operation in self._operations.values():
            if not operation.task.done() or operation.control_reported:
                continue
            error = self._outcome(operation).error
            if error is not None and not isinstance(error, Exception):
                operation.control_reported = True
                failures.append(error)
        return failures

    def take_reconciliation_failures(self) -> list[BaseException]:
        """Report each read attempt once without discarding the original fence."""
        failures = []
        for operation in self._operations.values():
            task = operation.reconciliation
            if task is None or not task.done():
                continue
            observed = _completed_result(task)
            if observed.error is None:
                continue
            original = self._outcome(operation).error
            assert original is not None
            current = observed.error
            if isinstance(current, Exception) and not isinstance(original, Exception):
                # This is a new ordinary read failure, not renewed cancellation.
                # Retain historical control evidence only as its cause.
                cause = exception_cause(current)
                set_exception_cause(
                    current,
                    failure_control_cause(
                        [original, *(() if cause is None else (cause,))], current
                    ),
                )
                failure = current
            elif isinstance(current, Exception):
                assert isinstance(original, Exception)
                failure = (
                    current
                    if current is original
                    else ExceptionGroup(
                        "Host effect and reconciliation failed", [original, current]
                    )
                )
            else:
                cause = exception_cause(current)
                set_exception_cause(
                    current,
                    failure_control_cause(
                        [original, *(() if cause is None else (cause,))], current
                    ),
                )
                failure = current
            failures.append(failure)
            operation.reconciliation = None
        return failures

    def _pending_tasks(self):
        return tuple(
            task
            for operation in self._operations.values()
            for task in (operation.task, operation.reconciliation)
            if task is not None
            and (not task.done() or not operation.reported or task is operation.reconciliation)
        )

    def mark_active(self, identity: HostOperationIdentity) -> None:
        """Called only by a role adapter after positive native admission evidence."""
        operation = self._operations.get(identity.key)
        if operation is None or operation.identity != identity or operation.task.done():
            raise ValueError("Host activity does not match a retained operation.")
        operation.phase = "active"

    def renew_dispatch_windows(self, deadline: float) -> None:
        """Renew local scheduling time for retained work during explicit servicing.

        This does not renew native claims, semantic deadlines, or authorization.
        Each receiving operation must authenticate those immediately at dispatch.
        """
        if type(deadline) not in (int, float) or not isfinite(deadline):
            raise ValueError("Host dispatch window must have a finite deadline.")
        if self._closing:
            return
        for operation in self._operations.values():
            if not operation.task.done():
                operation.dispatch_deadline = deadline
                operation.window_changed.set()

    async def wait_for_dispatch_window(
        self, identity: HostOperationIdentity, *, initial_deadline: float
    ) -> bool:
        """Keep prepared work owned until a live pass permits local dispatch.

        A single bounded pass cannot authorize effects indefinitely. After that
        window ends, only a later explicit pass can renew it; shutdown instead
        wakes the adapter to hand off without starting new work.
        """
        operation = self._operations.get(identity.key)
        if (
            operation is None
            or operation.identity != identity
            or operation.task is not asyncio.current_task()
        ):
            raise ValueError("Dispatch observation requires its retained operation owner.")
        if type(initial_deadline) not in (int, float) or not isfinite(initial_deadline):
            raise ValueError("Host dispatch window must have a finite deadline.")
        if operation.dispatch_deadline is None:
            operation.dispatch_deadline = initial_deadline
        while not self._closing and not operation.stop.is_set():
            if asyncio.get_running_loop().time() < operation.dispatch_deadline:
                return True
            operation.window_changed.clear()
            await operation.window_changed.wait()
        return False

    def mark_uncertain(self, identity: HostOperationIdentity) -> None:
        operation = self._operations.get(identity.key)
        if operation is None or operation.identity != identity:
            raise ValueError("Host uncertainty does not match a retained operation.")
        operation.phase = "uncertain"

    def inspect(self) -> HostOwnedObservation:
        return self._snapshot(consume=False)

    def release_settled(self, identity: HostOperationIdentity) -> None:
        """Native adapter confirms exact settlement or durable recovery handoff.

        This private bookkeeping operation is not a receipt validator. The
        adapter must positively reconcile with its owner before invoking it.
        Neither an exception nor task completion is sufficient by itself.
        """
        operation = self._operations.get(identity.key)
        if (
            operation is None
            or operation.identity != identity
            or not operation.task.done()
            or not operation.reported
            or (operation.reconciliation is not None and not operation.reconciliation.done())
        ):
            raise ValueError("Host settlement requires its observed exact operation.")
        del self._operations[identity.key]

    def _snapshot(self, *, consume: bool) -> HostOwnedObservation:
        completed = []
        active = []
        uncertain = []
        for operation in self._operations.values():
            if operation.task.done():
                outcome = self._outcome(operation)
                recovery = operation.reconciliation
                if recovery is not None and recovery.done():
                    recovered = _completed_result(recovery)
                    if recovered.error is None and recovered.value is not None:
                        outcome = HostOwnedOutcome(
                            operation.identity,
                            value=recovered.value,
                            error=outcome.error,
                            reconciled=True,
                        )
                if not operation.reported or not consume:
                    completed.append(outcome)
                if consume:
                    operation.reported = True
                uncertain.append(operation.identity)
            elif operation.phase == "active":
                active.append(operation.identity)
            else:
                uncertain.append(operation.identity)
        return HostOwnedObservation(tuple(completed), tuple(active), tuple(uncertain))

    async def observe(self, timeout_s: float) -> HostOwnedObservation:
        _seconds(timeout_s)
        if self._observing:
            raise RuntimeError("Collaboration host servicing is already active.")
        self._observing = True
        try:
            self._start_reconciliations()
            pending = self._pending_tasks()
            if pending:
                # asyncio.wait never cancels children on timeout or caller
                # cancellation. In particular, no wait_for/gather owns them.
                await asyncio.wait(
                    pending,
                    timeout=timeout_s,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            else:
                await asyncio.sleep(0)
            return self._snapshot(consume=True)
        finally:
            self._observing = False

    def request_close(self) -> None:
        self._closing = True
        for operation in self._operations.values():
            operation.stop.set()
            operation.window_changed.set()

    async def close(self, timeout_s: float) -> HostOwnedObservation:
        _seconds(timeout_s)
        self.request_close()
        # Closing can signal an active observer, but does not race it for
        # completed outcomes. The host facade joins its active pass first.
        if self._observing:
            raise RuntimeError("Join active servicing before collecting shutdown outcomes.")
        self._observing = True
        try:
            self._start_reconciliations()
            pending = self._pending_tasks()
            if pending:
                await asyncio.wait(
                    pending,
                    timeout=timeout_s,
                )
            else:
                await asyncio.sleep(0)
            return self._snapshot(consume=True)
        finally:
            self._observing = False
