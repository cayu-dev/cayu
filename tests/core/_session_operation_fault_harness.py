"""Deterministic fault scheduling for SessionStore operation publications.

This module is repository-private test infrastructure.  It deliberately wraps
one store instance instead of emulating the large SessionStore contract, so the
real backend keeps ownership of validation, transforms, transactions, and CAS.
"""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeAlias, cast

from cayu._exception_groups import (
    exception_cause,
    iter_exception_tree,
    set_exception_cause,
)
from cayu.core import Event, EventType
from cayu.runtime.sessions import (
    Session,
    SessionOperationTransform,
    SessionStatus,
    SessionStore,
    _OwnedOffThreadSessionCommitGuard,
    _reject_reserved_runtime_publication_key,
)

_MAX_SAFE_ID_BYTES = 64
_MAX_SELECTOR_BYTES = 8_192
_MAX_RULES = 64
_MAX_ACTIONS = 256
_DEFAULT_TRACE_LIMIT = 256
_MAX_TRACE_LIMIT = 256
_BARRIER_RELEASE_TIMEOUT_SECONDS = 30.0
_CLEANUP_TIMEOUT_SECONDS = 5.0
_SAFE_ID_PATTERN = re.compile(r"[A-Za-z0-9_.:-]+\Z")
_MISSING = object()


class MatchPolicy(StrEnum):
    DELEGATE = "delegate"
    FAIL = "fail"


class ReleaseDisposition(StrEnum):
    DELEGATE = "delegate"
    COMMIT = "commit"
    RETURN = "return"
    RAISE = "raise"


class CommitEvidence(StrEnum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class PublicationBoundary(StrEnum):
    SESSION_OPERATION = "session_operation"
    EVENT_APPEND = "event_append"


class PublicationFaultActionKind(StrEnum):
    UNMATCHED_DELEGATE = "unmatched_delegate"
    EXHAUSTED_DELEGATE = "exhausted_delegate"
    FAIL_BEFORE_TRANSFORM = "fail_before_transform"
    FAIL_BEFORE_COMMIT = "fail_before_commit"
    COMMIT_THEN_RAISE = "commit_then_raise"
    DELEGATE = "delegate"
    PAUSE_BEFORE_TRANSFORM = "pause_before_transform"
    PAUSE_BEFORE_COMMIT = "pause_before_commit"
    PAUSE_AFTER_COMMIT = "pause_after_commit"
    SCHEDULE_REJECTED = "schedule_rejected"


class PublicationFaultOutcome(StrEnum):
    RETURNED = "returned"
    INJECTED_FAILURE = "injected_failure"
    DELEGATE_FAILURE = "delegate_failure"
    CANCELLED = "cancelled"
    SCHEDULE_REJECTED = "schedule_rejected"


class InjectedSessionOperationPublicationError(ConnectionError):
    """A content-free failure injected at one explicit publication boundary."""


class SessionOperationFaultScheduleError(AssertionError):
    """A bounded publication fault schedule was invalid or not satisfied."""


class SessionOperationFaultCleanupError(RuntimeError):
    """The harness could not restore or drain its instance-local interception."""


@dataclass(frozen=True, slots=True)
class _OwnedDrainOutcome:
    cancellation: asyncio.CancelledError | None = None
    failures: tuple[BaseException, ...] = ()


def _require_exact_positive_int(value: object, field_name: str, *, maximum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0 or value > maximum:
        raise ValueError(f"{field_name} must be between 1 and {maximum}")
    return value


def _require_safe_id(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    if (
        not value
        or len(value.encode("utf-8")) > _MAX_SAFE_ID_BYTES
        or _SAFE_ID_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(
            f"{field_name} must be a non-empty bounded identifier using safe characters"
        )
    return value


def _require_selector_text(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    if not value or len(value.encode("utf-8")) > _MAX_SELECTOR_BYTES:
        raise ValueError(f"{field_name} must be non-empty and bounded")
    return value


class PublicationBarrier:
    """A one-shot barrier usable from async code or a synchronous commit guard."""

    def __init__(self) -> None:
        self._binding_lock = threading.Lock()
        self._thread_release = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._entered: asyncio.Event | None = None
        self._async_release: asyncio.Event | None = None
        self._entry_claimed = False

    def _bind(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._binding_lock:
            if self._loop is not None:
                raise SessionOperationFaultScheduleError(
                    "A publication barrier cannot be installed more than once."
                )
            self._loop = loop
            self._entered = asyncio.Event()
            self._async_release = asyncio.Event()
            if self._thread_release.is_set():
                self._async_release.set()

    def _claim_entry(self) -> None:
        with self._binding_lock:
            if self._entry_claimed:
                raise SessionOperationFaultScheduleError(
                    "A publication barrier cannot be entered more than once."
                )
            self._entry_claimed = True

    def _bound_state(
        self,
    ) -> tuple[asyncio.AbstractEventLoop, asyncio.Event, asyncio.Event]:
        with self._binding_lock:
            loop = self._loop
            entered = self._entered
            async_release = self._async_release
        if loop is None or entered is None or async_release is None:
            raise SessionOperationFaultScheduleError("The publication barrier is not installed.")
        return loop, entered, async_release

    async def _enter_async(self) -> None:
        self._claim_entry()
        loop, entered, async_release = self._bound_state()
        if asyncio.get_running_loop() is not loop:
            raise SessionOperationFaultScheduleError(
                "The publication barrier was entered from a foreign event loop."
            )
        entered.set()
        try:
            await asyncio.wait_for(
                async_release.wait(),
                timeout=_BARRIER_RELEASE_TIMEOUT_SECONDS,
            )
        except TimeoutError as error:
            raise SessionOperationFaultScheduleError(
                "A publication barrier was not released within its bounded lifetime."
            ) from error

    def _enter_sync(self) -> None:
        self._claim_entry()
        loop, entered, _ = self._bound_state()
        loop.call_soon_threadsafe(entered.set)
        if not self._thread_release.wait(_BARRIER_RELEASE_TIMEOUT_SECONDS):
            raise SessionOperationFaultScheduleError(
                "A publication commit barrier was not released within its bounded lifetime."
            )

    async def wait_until_entered(self, *, timeout: float = 5.0) -> None:
        if type(timeout) is not float or timeout <= 0:
            raise TypeError("timeout must be a positive float")
        _, entered, _ = self._bound_state()
        try:
            await asyncio.wait_for(entered.wait(), timeout=timeout)
        except TimeoutError as error:
            raise SessionOperationFaultScheduleError(
                "The publication barrier was not entered before the test watchdog expired."
            ) from error

    def release(self) -> None:
        self._thread_release.set()
        with self._binding_lock:
            loop = self._loop
            async_release = self._async_release
        if loop is not None and async_release is not None and not loop.is_closed():
            loop.call_soon_threadsafe(async_release.set)


@dataclass(frozen=True, slots=True)
class SessionOperationSelector:
    session_id: str | None = None
    idempotency_key: str | None = None
    idempotency_key_prefix: str | None = None
    event_types: frozenset[EventType] = frozenset()
    label: str | None = None

    def __post_init__(self) -> None:
        if self.session_id is not None:
            _require_selector_text(self.session_id, "session_id")
        if self.idempotency_key is not None:
            _require_selector_text(self.idempotency_key, "idempotency_key")
        if self.idempotency_key_prefix is not None:
            _require_selector_text(self.idempotency_key_prefix, "idempotency_key_prefix")
        if self.idempotency_key is not None and self.idempotency_key_prefix is not None:
            raise ValueError("exact operation key and operation key prefix are mutually exclusive")
        if type(self.event_types) is not frozenset:
            raise TypeError("event_types must be a frozenset")
        if any(type(event_type) is not EventType for event_type in self.event_types):
            raise TypeError("event_types must contain only EventType values")
        if self.label is not None:
            _require_safe_id(self.label, "label")
        if not any(
            (
                self.session_id is not None,
                self.idempotency_key is not None,
                self.idempotency_key_prefix is not None,
                bool(self.event_types),
                self.label is not None,
            )
        ):
            raise ValueError("a publication selector must contain at least one criterion")

    def _matches(
        self,
        *,
        session_id: object,
        idempotency_key: object,
        events: object,
        label: str | None,
    ) -> bool:
        if self.session_id is not None and (
            type(session_id) is not str or session_id != self.session_id
        ):
            return False
        if self.idempotency_key is not None and (
            type(idempotency_key) is not str or idempotency_key != self.idempotency_key
        ):
            return False
        if self.idempotency_key_prefix is not None:
            if type(idempotency_key) is not str:
                return False
            if not idempotency_key.startswith(self.idempotency_key_prefix):
                return False
        if self.label is not None and label != self.label:
            return False
        if self.event_types:
            if type(events) is not list or any(type(event) is not Event for event in events):
                raise SessionOperationFaultScheduleError(
                    "An event selector received an invalid event batch."
                )
            event_batch = cast("list[Event]", events)
            if any(type(event.type) is not EventType for event in event_batch):
                raise SessionOperationFaultScheduleError(
                    "An event selector received an invalid event batch."
                )
            actual_types = frozenset(event.type for event in event_batch)
            if not self.event_types.issubset(actual_types):
                return False
        return True


@dataclass(frozen=True, slots=True)
class FailBeforeTransform:
    count: int = 1

    def __post_init__(self) -> None:
        _require_exact_positive_int(self.count, "count", maximum=_MAX_ACTIONS)


@dataclass(frozen=True, slots=True)
class FailBeforeCommit:
    count: int = 1

    def __post_init__(self) -> None:
        _require_exact_positive_int(self.count, "count", maximum=_MAX_ACTIONS)


@dataclass(frozen=True, slots=True)
class CommitThenRaise:
    count: int = 1

    def __post_init__(self) -> None:
        _require_exact_positive_int(self.count, "count", maximum=_MAX_ACTIONS)


@dataclass(frozen=True, slots=True)
class Delegate:
    count: int = 1

    def __post_init__(self) -> None:
        _require_exact_positive_int(self.count, "count", maximum=_MAX_ACTIONS)


@dataclass(frozen=True, slots=True)
class PauseBeforeTransform:
    barrier: PublicationBarrier
    then: ReleaseDisposition = ReleaseDisposition.DELEGATE

    def __post_init__(self) -> None:
        if type(self.barrier) is not PublicationBarrier:
            raise TypeError("barrier must be a PublicationBarrier")
        if type(self.then) is not ReleaseDisposition:
            raise TypeError("then must be a ReleaseDisposition")
        if self.then not in (ReleaseDisposition.DELEGATE, ReleaseDisposition.RAISE):
            raise ValueError("PauseBeforeTransform must delegate or raise after release")


@dataclass(frozen=True, slots=True)
class PauseBeforeCommit:
    barrier: PublicationBarrier
    then: ReleaseDisposition = ReleaseDisposition.COMMIT

    def __post_init__(self) -> None:
        if type(self.barrier) is not PublicationBarrier:
            raise TypeError("barrier must be a PublicationBarrier")
        if type(self.then) is not ReleaseDisposition:
            raise TypeError("then must be a ReleaseDisposition")
        if self.then not in (ReleaseDisposition.COMMIT, ReleaseDisposition.RAISE):
            raise ValueError("PauseBeforeCommit must commit or raise after release")


@dataclass(frozen=True, slots=True)
class PauseAfterCommit:
    barrier: PublicationBarrier
    then: ReleaseDisposition = ReleaseDisposition.RETURN

    def __post_init__(self) -> None:
        if type(self.barrier) is not PublicationBarrier:
            raise TypeError("barrier must be a PublicationBarrier")
        if type(self.then) is not ReleaseDisposition:
            raise TypeError("then must be a ReleaseDisposition")
        if self.then not in (ReleaseDisposition.RETURN, ReleaseDisposition.RAISE):
            raise ValueError("PauseAfterCommit must return or raise after release")


FaultAction: TypeAlias = (
    FailBeforeTransform
    | FailBeforeCommit
    | CommitThenRaise
    | Delegate
    | PauseBeforeTransform
    | PauseBeforeCommit
    | PauseAfterCommit
)


@dataclass(frozen=True, slots=True)
class SessionOperationFaultRule:
    rule_id: str
    selector: SessionOperationSelector
    actions: tuple[FaultAction, ...]
    on_exhausted: MatchPolicy = MatchPolicy.FAIL

    def __post_init__(self) -> None:
        _require_safe_id(self.rule_id, "rule_id")
        if type(self.selector) is not SessionOperationSelector:
            raise TypeError("selector must be a SessionOperationSelector")
        if type(self.actions) is not tuple or not self.actions:
            raise TypeError("actions must be a non-empty tuple")
        if any(not isinstance(action, _FAULT_ACTION_TYPES) for action in self.actions):
            raise TypeError("actions contains an unsupported fault action")
        if type(self.on_exhausted) is not MatchPolicy:
            raise TypeError("on_exhausted must be a MatchPolicy")


@dataclass(frozen=True, slots=True)
class PublicationFaultTrace:
    sequence: int
    matched_rule_id: str | None
    action: PublicationFaultActionKind
    outcome: PublicationFaultOutcome
    action_reached: bool
    transform_started: bool
    transform_completed: bool
    committed: CommitEvidence
    acknowledgement_returned: bool


_FAULT_ACTION_TYPES = (
    FailBeforeTransform,
    FailBeforeCommit,
    CommitThenRaise,
    Delegate,
    PauseBeforeTransform,
    PauseBeforeCommit,
    PauseAfterCommit,
)


def _action_count(action: FaultAction) -> int:
    if isinstance(action, (FailBeforeTransform, FailBeforeCommit, CommitThenRaise, Delegate)):
        return action.count
    return 1


def _action_kind(action: FaultAction) -> PublicationFaultActionKind:
    if isinstance(action, FailBeforeTransform):
        return PublicationFaultActionKind.FAIL_BEFORE_TRANSFORM
    if isinstance(action, FailBeforeCommit):
        return PublicationFaultActionKind.FAIL_BEFORE_COMMIT
    if isinstance(action, CommitThenRaise):
        return PublicationFaultActionKind.COMMIT_THEN_RAISE
    if isinstance(action, Delegate):
        return PublicationFaultActionKind.DELEGATE
    if isinstance(action, PauseBeforeTransform):
        return PublicationFaultActionKind.PAUSE_BEFORE_TRANSFORM
    if isinstance(action, PauseBeforeCommit):
        return PublicationFaultActionKind.PAUSE_BEFORE_COMMIT
    return PublicationFaultActionKind.PAUSE_AFTER_COMMIT


@dataclass(slots=True)
class _ScheduledAction:
    action: FaultAction
    claimed: bool = False
    reached: bool = False


@dataclass(slots=True)
class _RuleState:
    rule: SessionOperationFaultRule
    actions: tuple[_ScheduledAction, ...]
    next_index: int = 0


@dataclass(frozen=True, slots=True)
class _Decision:
    rule_state: _RuleState | None
    scheduled_action: _ScheduledAction | None
    action: FaultAction | None
    kind: PublicationFaultActionKind


class _InvocationEvidence:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.transform_started = False
        self.transform_completed = False
        self.transform_failed = False
        self.action_reached = False
        self.committed = CommitEvidence.UNKNOWN
        self.acknowledgement_returned = False

    def mark_transform_started(self) -> None:
        with self._lock:
            self.transform_started = True

    def mark_transform_completed(self) -> None:
        with self._lock:
            self.transform_completed = True

    def mark_transform_failed(self) -> None:
        with self._lock:
            self.transform_failed = True
            self.committed = CommitEvidence.NO

    def mark_action_reached(self) -> None:
        with self._lock:
            self.action_reached = True

    def mark_committed(self, value: CommitEvidence) -> None:
        with self._lock:
            self.committed = value

    def mark_acknowledged(self) -> None:
        with self._lock:
            self.acknowledgement_returned = True

    def snapshot(self) -> tuple[bool, bool, bool, CommitEvidence, bool]:
        with self._lock:
            return (
                self.action_reached,
                self.transform_started,
                self.transform_completed,
                self.committed,
                self.acknowledgement_returned,
            )


class SessionOperationFaultHarness:
    """Install one bounded operation-publication schedule on one store instance."""

    def __init__(
        self,
        store: SessionStore,
        *,
        rules: Sequence[SessionOperationFaultRule],
        boundary: PublicationBoundary = PublicationBoundary.SESSION_OPERATION,
        on_unmatched: MatchPolicy = MatchPolicy.DELEGATE,
        trace_limit: int = _DEFAULT_TRACE_LIMIT,
    ) -> None:
        if not isinstance(store, SessionStore):
            raise TypeError("store must be a SessionStore")
        copied_rules = tuple(rules)
        if not copied_rules or len(copied_rules) > _MAX_RULES:
            raise ValueError(f"rules must contain between 1 and {_MAX_RULES} entries")
        if any(type(rule) is not SessionOperationFaultRule for rule in copied_rules):
            raise TypeError("rules must contain SessionOperationFaultRule values")
        rule_ids = tuple(rule.rule_id for rule in copied_rules)
        if len(set(rule_ids)) != len(rule_ids):
            raise ValueError("rule identifiers must be unique")
        if type(on_unmatched) is not MatchPolicy:
            raise TypeError("on_unmatched must be a MatchPolicy")
        if type(boundary) is not PublicationBoundary:
            raise TypeError("boundary must be a PublicationBoundary")
        if boundary is PublicationBoundary.EVENT_APPEND and any(
            isinstance(action, (FailBeforeCommit, PauseBeforeCommit))
            for rule in copied_rules
            for action in rule.actions
        ):
            raise ValueError(
                "event_append does not expose a safe between-validation-and-commit seam"
            )
        self._trace_limit = _require_exact_positive_int(
            trace_limit,
            "trace_limit",
            maximum=_MAX_TRACE_LIMIT,
        )
        expanded_total = sum(
            _action_count(action) for rule in copied_rules for action in rule.actions
        )
        if expanded_total > _MAX_ACTIONS:
            raise ValueError(f"the expanded action schedule must not exceed {_MAX_ACTIONS}")
        if self._trace_limit < expanded_total:
            raise ValueError("trace_limit must reserve one entry for every scheduled action")

        states = []
        barriers = []
        barrier_ids: set[int] = set()
        for rule in copied_rules:
            expanded = tuple(
                _ScheduledAction(action)
                for action in rule.actions
                for _ in range(_action_count(action))
            )
            states.append(_RuleState(rule=rule, actions=expanded))
            for scheduled in expanded:
                action = scheduled.action
                if isinstance(
                    action,
                    (PauseBeforeTransform, PauseBeforeCommit, PauseAfterCommit),
                ):
                    barrier_id = id(action.barrier)
                    if barrier_id in barrier_ids:
                        raise ValueError("each pause action must own a distinct barrier")
                    barrier_ids.add(barrier_id)
                    barriers.append(action.barrier)

        self._store = store
        self._boundary = boundary
        self._method_name = (
            "publish_session_operation"
            if boundary is PublicationBoundary.SESSION_OPERATION
            else "append_events"
        )
        self._rule_states = tuple(states)
        self._barriers = tuple(barriers)
        self._on_unmatched = on_unmatched
        self._state_lock = threading.Lock()
        self._label: ContextVar[str | None] = ContextVar(
            "cayu_test_session_operation_fault_label",
            default=None,
        )
        self._installed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._original_instance_value: object = _MISSING
        self._original_publish: Any = None
        self._original_publish_guarded: Any = None
        self._active_count = 0
        self._active_zero: asyncio.Event | None = None
        self._sequence = 0
        self._trace_records: list[PublicationFaultTrace] = []
        self._incidental_trace_limit = self._trace_limit - expanded_total
        self._incidental_trace_records = 0
        self._dropped_trace_entries = 0

    @property
    def trace(self) -> tuple[PublicationFaultTrace, ...]:
        with self._state_lock:
            return tuple(sorted(self._trace_records, key=lambda record: record.sequence))

    @property
    def dropped_trace_entries(self) -> int:
        with self._state_lock:
            return self._dropped_trace_entries

    @contextmanager
    def label(self, name: str) -> Iterator[None]:
        validated = _require_safe_id(name, "label")
        token = self._label.set(validated)
        try:
            yield
        finally:
            self._label.reset(token)

    async def __aenter__(self) -> SessionOperationFaultHarness:
        if self._installed:
            raise SessionOperationFaultScheduleError(
                "The publication fault harness is already installed."
            )
        loop = asyncio.get_running_loop()
        current_publish = getattr(self._store, self._method_name)
        if getattr(current_publish, "__cayu_session_operation_fault_harness__", False):
            raise SessionOperationFaultScheduleError(
                "A publication fault harness is already installed on this store."
            )
        if self._boundary is PublicationBoundary.SESSION_OPERATION and any(
            isinstance(scheduled.action, (FailBeforeCommit, PauseBeforeCommit))
            for state in self._rule_states
            for scheduled in state.actions
        ):
            checker = getattr(
                self._store,
                "_supports_owned_off_thread_session_commit_guard_protocol",
                None,
            )
            if not callable(checker) or checker() is not True:
                raise SessionOperationFaultScheduleError(
                    "The selected store does not own the required commit-guard protocol."
                )

        for barrier in self._barriers:
            barrier._bind(loop)

        store_dict = getattr(self._store, "__dict__", None)
        if type(store_dict) is not dict:
            raise SessionOperationFaultScheduleError(
                "The selected store does not support instance-local interception."
            )
        self._original_instance_value = store_dict.get(
            self._method_name,
            _MISSING,
        )
        self._original_publish = current_publish
        self._original_publish_guarded = (
            self._store.publish_session_operation_guarded
            if self._boundary is PublicationBoundary.SESSION_OPERATION
            else None
        )
        self._loop = loop
        self._active_zero = asyncio.Event()
        self._active_zero.set()

        if self._boundary is PublicationBoundary.SESSION_OPERATION:

            async def intercepted_publish(
                session_id: str,
                *,
                idempotency_key: str,
                operation_transform: SessionOperationTransform,
                events: list[Event],
                expected_statuses: set[SessionStatus] | None = None,
                expected_run_epoch: int | None = None,
                expected_transcript_cursor: int | None = None,
            ) -> Session:
                return await self._intercept(
                    session_id,
                    idempotency_key=idempotency_key,
                    operation_transform=operation_transform,
                    events=events,
                    expected_statuses=expected_statuses,
                    expected_run_epoch=expected_run_epoch,
                    expected_transcript_cursor=expected_transcript_cursor,
                )

            intercepted_publish.__dict__["__cayu_session_operation_fault_harness__"] = True
            self._store.__dict__[self._method_name] = intercepted_publish
        else:

            async def intercepted_append(session_id: str, events: list[Event]) -> None:
                await self._intercept_event_append(session_id, events=events)

            intercepted_append.__dict__["__cayu_session_operation_fault_harness__"] = True
            self._store.__dict__[self._method_name] = intercepted_append
        self._installed = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        del exc_type, traceback
        secondary: list[BaseException] = []
        try:
            self._restore()
        except BaseException as error:
            secondary.append(error)
        for barrier in self._barriers:
            try:
                barrier.release()
            except BaseException as error:
                secondary.append(error)

        drain_outcome = await self._drain_active_owned()
        cancellation = drain_outcome.cancellation
        if isinstance(exc, asyncio.CancelledError):
            if cancellation is not None and cancellation is not exc:
                exc.add_note(
                    "Additional cancellation was delivered while the publication "
                    "fault harness drained active calls."
                )
                for note in getattr(cancellation, "__notes__", ()):
                    exc.add_note(note)
            cancellation = exc
        secondary.extend(drain_outcome.failures)

        unsatisfied = self._unsatisfied_error()
        if unsatisfied is not None:
            if isinstance(exc, (GeneratorExit, KeyboardInterrupt, SystemExit)):
                exc.add_note(str(unsatisfied))
            else:
                secondary.append(unsatisfied)

        if cancellation is not None:
            ordered_failures: list[BaseException] = []
            seen_failure_ids = {id(cancellation)}
            has_new_failure = False
            existing_cause = exception_cause(cancellation)
            if existing_cause is not None:
                ordered_failures.append(existing_cause)
                seen_failure_ids.update(
                    id(candidate) for candidate in iter_exception_tree(existing_cause)
                )
            if exc is not None and exc is not cancellation and id(exc) not in seen_failure_ids:
                ordered_failures.append(exc)
                has_new_failure = True
                seen_failure_ids.update(id(candidate) for candidate in iter_exception_tree(exc))
            for failure in secondary:
                if id(failure) in seen_failure_ids:
                    continue
                ordered_failures.append(failure)
                has_new_failure = True
                seen_failure_ids.update(id(candidate) for candidate in iter_exception_tree(failure))
            if not has_new_failure:
                if exc is cancellation:
                    return False
                raise cancellation from existing_cause
            cause = (
                ordered_failures[0]
                if len(ordered_failures) == 1
                else BaseExceptionGroup(
                    "Publication fault harness body and cleanup failed.",
                    ordered_failures,
                )
            )
            if not set_exception_cause(cancellation, cause):
                raise BaseExceptionGroup(
                    "Publication fault harness body and cleanup failed.",
                    [cancellation, cause],
                ) from None
            raise cancellation from exception_cause(cancellation)
        if exc is None:
            if len(secondary) == 1:
                raise secondary[0]
            if secondary:
                raise BaseExceptionGroup(
                    "Publication fault harness cleanup failed.",
                    secondary,
                )
            return False
        if secondary:
            raise BaseExceptionGroup(
                "Publication fault harness body and cleanup failed.",
                [exc, *secondary],
            )
        return False

    def _restore(self) -> None:
        if not self._installed:
            return
        try:
            if self._original_instance_value is _MISSING:
                self._store.__dict__.pop(self._method_name, None)
            else:
                self._store.__dict__[self._method_name] = self._original_instance_value
        except BaseException as error:
            raise SessionOperationFaultCleanupError(
                "The publication fault harness could not restore the store method."
            ) from error
        finally:
            self._installed = False

    async def _drain_active_owned(self) -> _OwnedDrainOutcome:
        active_zero = self._active_zero
        if active_zero is None or active_zero.is_set():
            return _OwnedDrainOutcome()

        worker = asyncio.create_task(
            asyncio.wait_for(active_zero.wait(), timeout=_CLEANUP_TIMEOUT_SECONDS)
        )
        owner_task = asyncio.current_task()
        observed_cancelling = owner_task.cancelling() if owner_task is not None else 0
        cancellations: list[asyncio.CancelledError] = []
        failures: list[BaseException] = []
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError as cancellation:
                current_cancelling = owner_task.cancelling() if owner_task is not None else 0
                if worker.cancelled() and current_cancelling <= observed_cancelling:
                    break
                observed_cancelling = current_cancelling
                cancellations.append(cancellation)
            except BaseException:
                if worker.done():
                    break
                raise
        try:
            worker.result()
        except asyncio.CancelledError as error:
            cleanup_failure = SessionOperationFaultCleanupError(
                "The publication fault harness drain waiter was cancelled independently."
            )
            cleanup_failure.__cause__ = error
            failures.append(cleanup_failure)
        except TimeoutError as error:
            failures.append(
                SessionOperationFaultCleanupError(
                    "Intercepted publication calls did not settle after barriers were released."
                )
            )
            failures[-1].__cause__ = error
        except BaseException as error:
            failures.append(error)
        cancellation = cancellations[0] if cancellations else None
        if cancellation is not None and len(cancellations) > 1:
            additional = len(cancellations) - 1
            noun = "signal" if additional == 1 else "signals"
            verb = "was" if additional == 1 else "were"
            cancellation.add_note(
                f"{additional} additional cancellation {noun} {verb} "
                "delivered while the publication fault harness drained active calls."
            )
        return _OwnedDrainOutcome(
            cancellation=cancellation,
            failures=tuple(failures),
        )

    def _select(
        self,
        *,
        session_id: object,
        idempotency_key: object,
        events: object,
    ) -> _Decision:
        label = self._label.get()
        with self._state_lock:
            matches = [
                state
                for state in self._rule_states
                if state.rule.selector._matches(
                    session_id=session_id,
                    idempotency_key=idempotency_key,
                    events=events,
                    label=label,
                )
            ]
            if len(matches) > 1:
                raise SessionOperationFaultScheduleError(
                    "Multiple publication fault rules matched one operation."
                )
            if not matches:
                if self._on_unmatched is MatchPolicy.FAIL:
                    raise SessionOperationFaultScheduleError(
                        "An unexpected publication did not match any fault rule."
                    )
                return _Decision(
                    rule_state=None,
                    scheduled_action=None,
                    action=None,
                    kind=PublicationFaultActionKind.UNMATCHED_DELEGATE,
                )
            state = matches[0]
            if state.next_index >= len(state.actions):
                if state.rule.on_exhausted is MatchPolicy.FAIL:
                    raise SessionOperationFaultScheduleError(
                        "A publication matched an exhausted fault rule."
                    )
                return _Decision(
                    rule_state=state,
                    scheduled_action=None,
                    action=None,
                    kind=PublicationFaultActionKind.EXHAUSTED_DELEGATE,
                )
            scheduled = state.actions[state.next_index]
            state.next_index += 1
            scheduled.claimed = True
            return _Decision(
                rule_state=state,
                scheduled_action=scheduled,
                action=scheduled.action,
                kind=_action_kind(scheduled.action),
            )

    def _mark_action_reached(self, decision: _Decision) -> None:
        scheduled = decision.scheduled_action
        if scheduled is not None:
            with self._state_lock:
                scheduled.reached = True

    def _unsatisfied_error(self) -> SessionOperationFaultScheduleError | None:
        with self._state_lock:
            unsatisfied = tuple(
                (state.rule.rule_id, sum(not action.reached for action in state.actions))
                for state in self._rule_states
                if any(not action.reached for action in state.actions)
            )
        if not unsatisfied:
            return None
        summary = ", ".join(f"{rule_id}:{count}" for rule_id, count in unsatisfied)
        return SessionOperationFaultScheduleError(
            f"Publication fault schedule was not satisfied ({summary})."
        )

    def _begin_invocation(self) -> int:
        active_zero = self._active_zero
        if active_zero is None:
            raise SessionOperationFaultScheduleError(
                "The publication fault harness is not installed."
            )
        with self._state_lock:
            self._active_count += 1
            self._sequence += 1
            sequence = self._sequence
        active_zero.clear()
        return sequence

    def _finish_invocation(
        self,
        trace: PublicationFaultTrace,
        *,
        scheduled: bool,
    ) -> None:
        active_zero = self._active_zero
        with self._state_lock:
            if scheduled:
                self._trace_records.append(trace)
            elif self._incidental_trace_records < self._incidental_trace_limit:
                self._trace_records.append(trace)
                self._incidental_trace_records += 1
            else:
                self._dropped_trace_entries += 1
            self._active_count -= 1
            settled = self._active_count == 0
        if settled and active_zero is not None:
            active_zero.set()

    async def _intercept(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None,
        expected_run_epoch: int | None,
        expected_transcript_cursor: int | None,
    ) -> Session:
        if asyncio.get_running_loop() is not self._loop:
            raise SessionOperationFaultScheduleError(
                "A publication was attempted from a foreign event loop."
            )
        sequence = self._begin_invocation()
        evidence = _InvocationEvidence()
        decision = _Decision(
            rule_state=None,
            scheduled_action=None,
            action=None,
            kind=PublicationFaultActionKind.SCHEDULE_REJECTED,
        )
        outcome = PublicationFaultOutcome.SCHEDULE_REJECTED
        try:
            decision = self._select(
                session_id=session_id,
                idempotency_key=idempotency_key,
                events=events,
            )
            action = decision.action
            if isinstance(action, FailBeforeTransform):
                evidence.mark_action_reached()
                evidence.mark_committed(CommitEvidence.NO)
                self._mark_action_reached(decision)
                raise InjectedSessionOperationPublicationError(
                    "Session operation publication failed before transform."
                )
            if isinstance(action, PauseBeforeTransform):
                evidence.mark_action_reached()
                self._mark_action_reached(decision)
                try:
                    await action.barrier._enter_async()
                except BaseException:
                    evidence.mark_committed(CommitEvidence.NO)
                    raise
                if action.then is ReleaseDisposition.RAISE:
                    evidence.mark_committed(CommitEvidence.NO)
                    raise InjectedSessionOperationPublicationError(
                        "Session operation publication failed before transform."
                    )

            def observed_transform(
                current_session: Session,
                checkpoint: dict[str, Any] | None,
                current_record: dict[str, Any] | None,
            ):
                evidence.mark_transform_started()
                try:
                    result = operation_transform(
                        current_session,
                        checkpoint,
                        current_record,
                    )
                except BaseException:
                    evidence.mark_transform_failed()
                    raise
                evidence.mark_transform_completed()
                return result

            effective_transform = (
                observed_transform if callable(operation_transform) else operation_transform
            )
            call_kwargs = {
                "operation_transform": effective_transform,
                "events": events,
                "expected_statuses": expected_statuses,
                "expected_run_epoch": expected_run_epoch,
                "expected_transcript_cursor": expected_transcript_cursor,
            }

            if isinstance(action, (FailBeforeCommit, PauseBeforeCommit)):

                def test_commit_guard() -> None:
                    evidence.mark_action_reached()
                    self._mark_action_reached(decision)
                    if isinstance(action, PauseBeforeCommit):
                        try:
                            action.barrier._enter_sync()
                        except BaseException:
                            evidence.mark_committed(CommitEvidence.NO)
                            raise
                        if action.then is ReleaseDisposition.COMMIT:
                            return
                    evidence.mark_committed(CommitEvidence.NO)
                    raise InjectedSessionOperationPublicationError(
                        "Session operation publication failed before commit."
                    )

                result = await self._original_publish_guarded(
                    session_id,
                    idempotency_key=_reject_reserved_runtime_publication_key(
                        idempotency_key,
                        "idempotency_key",
                    ),
                    commit_guard=_OwnedOffThreadSessionCommitGuard(test_commit_guard),
                    **call_kwargs,
                )
            else:
                if isinstance(action, Delegate):
                    evidence.mark_action_reached()
                    self._mark_action_reached(decision)
                result = await self._original_publish(
                    session_id,
                    idempotency_key=idempotency_key,
                    **call_kwargs,
                )

            evidence.mark_committed(CommitEvidence.YES)
            if isinstance(action, CommitThenRaise):
                evidence.mark_action_reached()
                self._mark_action_reached(decision)
                raise InjectedSessionOperationPublicationError(
                    "Session operation publication acknowledgement was lost after commit."
                )
            if isinstance(action, PauseAfterCommit):
                evidence.mark_action_reached()
                self._mark_action_reached(decision)
                await action.barrier._enter_async()
                if action.then is ReleaseDisposition.RAISE:
                    raise InjectedSessionOperationPublicationError(
                        "Session operation publication acknowledgement was lost after commit."
                    )
            evidence.mark_acknowledged()
            outcome = PublicationFaultOutcome.RETURNED
            return result
        except asyncio.CancelledError:
            outcome = PublicationFaultOutcome.CANCELLED
            raise
        except InjectedSessionOperationPublicationError:
            outcome = PublicationFaultOutcome.INJECTED_FAILURE
            raise
        except SessionOperationFaultScheduleError:
            if decision.kind is PublicationFaultActionKind.SCHEDULE_REJECTED:
                evidence.mark_committed(CommitEvidence.NO)
            outcome = PublicationFaultOutcome.SCHEDULE_REJECTED
            raise
        except BaseException:
            _, transform_started, _, committed, _ = evidence.snapshot()
            if not transform_started and committed is CommitEvidence.UNKNOWN:
                evidence.mark_committed(CommitEvidence.NO)
            outcome = PublicationFaultOutcome.DELEGATE_FAILURE
            raise
        finally:
            (
                action_reached,
                transform_started,
                transform_completed,
                committed,
                acknowledgement_returned,
            ) = evidence.snapshot()
            matched_rule_id = (
                None if decision.rule_state is None else decision.rule_state.rule.rule_id
            )
            self._finish_invocation(
                PublicationFaultTrace(
                    sequence=sequence,
                    matched_rule_id=matched_rule_id,
                    action=decision.kind,
                    outcome=outcome,
                    action_reached=action_reached,
                    transform_started=transform_started,
                    transform_completed=transform_completed,
                    committed=committed,
                    acknowledgement_returned=acknowledgement_returned,
                ),
                scheduled=decision.scheduled_action is not None,
            )

    async def _intercept_event_append(
        self,
        session_id: str,
        *,
        events: list[Event],
    ) -> None:
        """Apply the shared fault vocabulary to a direct event append."""

        if asyncio.get_running_loop() is not self._loop:
            raise SessionOperationFaultScheduleError(
                "An event append was attempted from a foreign event loop."
            )
        sequence = self._begin_invocation()
        evidence = _InvocationEvidence()
        decision = _Decision(
            rule_state=None,
            scheduled_action=None,
            action=None,
            kind=PublicationFaultActionKind.SCHEDULE_REJECTED,
        )
        outcome = PublicationFaultOutcome.SCHEDULE_REJECTED
        try:
            decision = self._select(
                session_id=session_id,
                idempotency_key=None,
                events=events,
            )
            action = decision.action
            if isinstance(action, FailBeforeTransform):
                evidence.mark_action_reached()
                evidence.mark_committed(CommitEvidence.NO)
                self._mark_action_reached(decision)
                raise InjectedSessionOperationPublicationError(
                    "Event publication failed before append."
                )
            if isinstance(action, PauseBeforeTransform):
                evidence.mark_action_reached()
                self._mark_action_reached(decision)
                try:
                    await action.barrier._enter_async()
                except BaseException:
                    evidence.mark_committed(CommitEvidence.NO)
                    raise
                if action.then is ReleaseDisposition.RAISE:
                    evidence.mark_committed(CommitEvidence.NO)
                    raise InjectedSessionOperationPublicationError(
                        "Event publication failed before append."
                    )
            if isinstance(action, Delegate):
                evidence.mark_action_reached()
                self._mark_action_reached(decision)
            await self._original_publish(session_id, events)
            evidence.mark_committed(CommitEvidence.YES)
            if isinstance(action, CommitThenRaise):
                evidence.mark_action_reached()
                self._mark_action_reached(decision)
                raise InjectedSessionOperationPublicationError(
                    "Event publication acknowledgement was lost after commit."
                )
            if isinstance(action, PauseAfterCommit):
                evidence.mark_action_reached()
                self._mark_action_reached(decision)
                await action.barrier._enter_async()
                if action.then is ReleaseDisposition.RAISE:
                    raise InjectedSessionOperationPublicationError(
                        "Event publication acknowledgement was lost after commit."
                    )
            evidence.mark_acknowledged()
            outcome = PublicationFaultOutcome.RETURNED
        except asyncio.CancelledError:
            outcome = PublicationFaultOutcome.CANCELLED
            raise
        except InjectedSessionOperationPublicationError:
            outcome = PublicationFaultOutcome.INJECTED_FAILURE
            raise
        except SessionOperationFaultScheduleError:
            if decision.kind is PublicationFaultActionKind.SCHEDULE_REJECTED:
                evidence.mark_committed(CommitEvidence.NO)
            outcome = PublicationFaultOutcome.SCHEDULE_REJECTED
            raise
        except BaseException:
            outcome = PublicationFaultOutcome.DELEGATE_FAILURE
            raise
        finally:
            (
                action_reached,
                transform_started,
                transform_completed,
                committed,
                acknowledgement_returned,
            ) = evidence.snapshot()
            matched_rule_id = (
                None if decision.rule_state is None else decision.rule_state.rule.rule_id
            )
            self._finish_invocation(
                PublicationFaultTrace(
                    sequence=sequence,
                    matched_rule_id=matched_rule_id,
                    action=decision.kind,
                    outcome=outcome,
                    action_reached=action_reached,
                    transform_started=transform_started,
                    transform_completed=transform_completed,
                    committed=committed,
                    acknowledgement_returned=acknowledgement_returned,
                ),
                scheduled=decision.scheduled_action is not None,
            )
