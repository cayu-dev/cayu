"""Task-local run ownership, interaction attribution and retained settlement context."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any
from weakref import ReferenceType, ref

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import Event, EventType, copy_event, event_with_runtime_envelope_authority
from cayu.sessions._interaction_transitions import (
    InteractionTransitionSpec,
    copy_interaction_transition_spec,
)
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.records import Session

_SESSION_CLEANUP_IDENTITIES: ContextVar[dict[str, tuple[str, int]] | None] = ContextVar(
    "cayu_session_cleanup_identities",
    default=None,
)


_SESSION_RUN_FENCES: ContextVar[dict[str, int] | None] = ContextVar(
    "cayu_session_run_fence",
    default=None,
)


_SESSION_RUN_FENCE_OWNERS: ContextVar[dict[str, _SessionRunFenceOwnership] | None] = ContextVar(
    "cayu_session_run_fence_owners",
    default=None,
)


_SESSION_INTERACTION_IDS: ContextVar[dict[str, str] | None] = ContextVar(
    "cayu_session_interaction_ids",
    default=None,
)


_SESSION_INTERACTION_STARTED_AT: ContextVar[dict[str, float] | None] = ContextVar(
    "cayu_session_interaction_started_at",
    default=None,
)


_SESSION_INTERACTION_RECOVERED_ACTIVE_THROUGH: ContextVar[dict[str, datetime] | None] = ContextVar(
    "cayu_session_interaction_recovered_active_through",
    default=None,
)


_SESSION_INVOCATION_INTERACTION_IDS: ContextVar[dict[str, tuple[str, ...]] | None] = ContextVar(
    "cayu_session_invocation_interaction_ids",
    default=None,
)


_SESSION_SETTLED_INTERACTION_IDS: ContextVar[dict[str, str] | None] = ContextVar(
    "cayu_session_settled_interaction_ids",
    default=None,
)


_SESSION_SETTLED_INTERACTION_TRANSITIONS: ContextVar[dict[str, object] | None] = ContextVar(
    "cayu_session_settled_interaction_transitions",
    default=None,
)


_SESSION_INVOCATION_TERMINAL_EVENTS: ContextVar[dict[str, Event] | None] = ContextVar(
    "cayu_session_invocation_terminal_events",
    default=None,
)


class _InheritInteractionAttribution:
    __slots__ = ()


INHERIT_INTERACTION = _InheritInteractionAttribution()


InteractionAttribution = str | None | _InheritInteractionAttribution


UNASSOCIATED_RUNTIME_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
        EventType.SESSION_FAILED,
        EventType.SESSION_INTERRUPTED,
        EventType.SESSION_EXPORT_PUBLISHED,
        EventType.SESSION_EXPORT_RELEASED,
        EventType.SESSION_EXPORT_RETIRED,
        EventType.HOOK_STARTED,
        EventType.HOOK_COMPLETED,
        EventType.HOOK_FAILED,
        EventType.TASK_INTERRUPTED_HANDOFF,
        EventType.TASK_COMPLETION_RESULT_RESOLVED,
        EventType.RUNTIME_INTERACTION_TRANSITION_ACKNOWLEDGEMENT_FAILED,
    }
)


def _current_session_interaction_id(session_id: str) -> str | None:
    interactions = _SESSION_INTERACTION_IDS.get()
    return None if interactions is None else interactions.get(session_id)


def _current_session_interaction_started_at(session_id: str) -> float | None:
    started = _SESSION_INTERACTION_STARTED_AT.get()
    return None if started is None else started.get(session_id)


def _current_session_interaction_recovered_active_through(
    session_id: str,
) -> datetime | None:
    recovered = _SESSION_INTERACTION_RECOVERED_ACTIVE_THROUGH.get()
    return None if recovered is None else recovered.get(session_id)


def _set_session_interaction_recovered_active_through(
    session_id: str,
    active_through: datetime,
) -> None:
    session_id = require_clean_nonblank(session_id, "session_id")
    if active_through.tzinfo is None or active_through.utcoffset() is None:
        raise ValueError("active_through must be timezone-aware.")
    recovered = dict(_SESSION_INTERACTION_RECOVERED_ACTIVE_THROUGH.get() or {})
    recovered[session_id] = active_through.astimezone(UTC)
    _SESSION_INTERACTION_RECOVERED_ACTIVE_THROUGH.set(recovered)


def _clear_session_interaction_recovered_active_through(session_id: str) -> None:
    recovered = _SESSION_INTERACTION_RECOVERED_ACTIVE_THROUGH.get()
    if recovered is None or session_id not in recovered:
        return
    remaining = dict(recovered)
    remaining.pop(session_id)
    _SESSION_INTERACTION_RECOVERED_ACTIVE_THROUGH.set(remaining or None)


def _current_session_invocation_interaction_ids(session_id: str) -> tuple[str, ...]:
    interactions = _SESSION_INVOCATION_INTERACTION_IDS.get()
    return () if interactions is None else interactions.get(session_id, ())


def _mark_session_interaction_settled(
    session_id: str,
    interaction_id: str,
    transition: object | None = None,
) -> None:
    """Record the exact interaction settled by the current stream invocation."""

    session_id = require_clean_nonblank(session_id, "session_id")
    interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
    settled = dict(_SESSION_SETTLED_INTERACTION_IDS.get() or {})
    settled[session_id] = interaction_id
    _SESSION_SETTLED_INTERACTION_IDS.set(settled)
    if transition is not None:
        if not isinstance(transition, InteractionTransitionSpec):
            raise TypeError("Settled interaction transition has an invalid type.")
        transitions = dict(_SESSION_SETTLED_INTERACTION_TRANSITIONS.get() or {})
        transitions[session_id] = copy_interaction_transition_spec(transition)
        _SESSION_SETTLED_INTERACTION_TRANSITIONS.set(transitions)


def _current_session_invocation_settlement_transition(
    session_id: str,
) -> InteractionTransitionSpec | None:
    transitions = _SESSION_SETTLED_INTERACTION_TRANSITIONS.get()
    if transitions is None:
        return None
    transition = transitions.get(session_id)
    if transition is None:
        return None
    if not isinstance(transition, InteractionTransitionSpec):
        raise RuntimeError("Settled invocation transition authority is malformed.")
    return copy_interaction_transition_spec(transition)


def _mark_session_invocation_terminal_event(event: Event) -> None:
    """Retain exact task-local provenance for one durable session terminal event."""

    if type(event) is not Event or event.type not in {
        EventType.SESSION_COMPLETED,
        EventType.SESSION_FAILED,
        EventType.SESSION_INTERRUPTED,
    }:
        raise TypeError("Invocation terminal evidence must be a session terminal event.")
    retained = dict(_SESSION_INVOCATION_TERMINAL_EVENTS.get() or {})
    retained[event.session_id] = copy_event(event)
    _SESSION_INVOCATION_TERMINAL_EVENTS.set(retained)


def _current_session_invocation_terminal_event(session_id: str) -> Event | None:
    retained = _SESSION_INVOCATION_TERMINAL_EVENTS.get()
    event = None if retained is None else retained.get(session_id)
    return None if event is None else copy_event(event)


def _clear_session_invocation_terminal_event(session_id: str) -> None:
    retained = _SESSION_INVOCATION_TERMINAL_EVENTS.get()
    if retained is None or session_id not in retained:
        return
    remaining = dict(retained)
    remaining.pop(session_id)
    _SESSION_INVOCATION_TERMINAL_EVENTS.set(remaining or None)


def _latest_session_invocation_interaction_is_settled(session_id: str) -> bool:
    """Return whether this invocation durably settled its latest interaction."""

    settled = _SESSION_SETTLED_INTERACTION_IDS.get()
    if settled is None:
        return False
    settled_interaction_id = settled.get(session_id)
    invocation_interaction_ids = _current_session_invocation_interaction_ids(session_id)
    return bool(
        settled_interaction_id is not None
        and invocation_interaction_ids
        and settled_interaction_id == invocation_interaction_ids[-1]
    )


def _clear_session_interaction_settlement(session_id: str) -> None:
    settled = _SESSION_SETTLED_INTERACTION_IDS.get()
    if settled is None or session_id not in settled:
        return
    remaining = dict(settled)
    remaining.pop(session_id)
    _SESSION_SETTLED_INTERACTION_IDS.set(remaining or None)
    transitions = _SESSION_SETTLED_INTERACTION_TRANSITIONS.get()
    if transitions is not None and session_id in transitions:
        retained = dict(transitions)
        retained.pop(session_id)
        _SESSION_SETTLED_INTERACTION_TRANSITIONS.set(retained or None)


def resolve_interaction_attribution(
    session_id: str,
    interaction_id: InteractionAttribution,
) -> str | None:
    """Resolve omitted transcript attribution while preserving explicit null."""

    if interaction_id is INHERIT_INTERACTION:
        interaction_id = _current_session_interaction_id(session_id)
    if interaction_id is None:
        return None
    if type(interaction_id) is not str:
        raise TypeError("interaction_id must be a string, None, or INHERIT_INTERACTION.")
    return require_clean_nonblank(interaction_id, "interaction_id")


def attribute_event_to_current_interaction(event: Event) -> Event:
    """Copy an event and explicitly apply the active runtime interaction identity."""

    copied = copy_event(event)
    if copied.interaction_id is None and copied.type not in UNASSOCIATED_RUNTIME_EVENT_TYPES:
        interaction_id = _current_session_interaction_id(copied.session_id)
        if interaction_id is not None:
            copied = copied.model_copy(update={"interaction_id": interaction_id})
    active_interaction_id = _current_session_interaction_id(copied.session_id)
    invocation_interaction_ids = _current_session_invocation_interaction_ids(copied.session_id)
    if active_interaction_id is not None or invocation_interaction_ids:
        fields = ["session_id"]
        if active_interaction_id is not None and copied.interaction_id == active_interaction_id:
            fields.append("interaction_id")
        copied = event_with_runtime_envelope_authority(copied, *fields)
    return copied


def attribute_events_to_current_interaction(events: list[Event]) -> list[Event]:
    if type(events) is not list:
        raise TypeError("Runtime events must be a list.")
    return [attribute_event_to_current_interaction(event) for event in events]


def _activate_session_interaction(session_id: str, interaction_id: str) -> None:
    session_id = require_clean_nonblank(session_id, "session_id")
    interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
    interactions = dict(_SESSION_INTERACTION_IDS.get() or {})
    previous_interaction_id = interactions.get(session_id)
    interactions[session_id] = interaction_id
    _SESSION_INTERACTION_IDS.set(interactions)
    started = dict(_SESSION_INTERACTION_STARTED_AT.get() or {})
    if previous_interaction_id != interaction_id or session_id not in started:
        started[session_id] = time.monotonic()
        _SESSION_INTERACTION_STARTED_AT.set(started)
    invocation_interactions = dict(_SESSION_INVOCATION_INTERACTION_IDS.get() or {})
    seen = invocation_interactions.get(session_id, ())
    if interaction_id not in seen:
        # A newly admitted invocation may legitimately reuse an open interaction
        # during recovery. Settlement from the previous invocation must not make
        # cancellation of the resumed work look already committed.
        _clear_session_interaction_settlement(session_id)
        invocation_interactions[session_id] = (*seen, interaction_id)
        _SESSION_INVOCATION_INTERACTION_IDS.set(invocation_interactions)


def _close_session_interaction(session_id: str) -> None:
    """Clear active attribution while retaining invocation-level history."""

    interactions = _SESSION_INTERACTION_IDS.get()
    if interactions is not None and session_id in interactions:
        remaining = dict(interactions)
        remaining.pop(session_id)
        _SESSION_INTERACTION_IDS.set(remaining or None)
    started = _SESSION_INTERACTION_STARTED_AT.get()
    if started is not None and session_id in started:
        remaining_started = dict(started)
        remaining_started.pop(session_id)
        _SESSION_INTERACTION_STARTED_AT.set(remaining_started or None)
    _clear_session_interaction_recovered_active_through(session_id)


def _deactivate_session_interaction(session_id: str) -> None:
    _close_session_interaction(session_id)
    _clear_session_interaction_settlement(session_id)
    _clear_session_invocation_terminal_event(session_id)
    invocation_interactions = _SESSION_INVOCATION_INTERACTION_IDS.get()
    if invocation_interactions is not None and session_id in invocation_interactions:
        remaining_invocations = dict(invocation_interactions)
        remaining_invocations.pop(session_id)
        _SESSION_INVOCATION_INTERACTION_IDS.set(remaining_invocations or None)


class _SessionRunFenceContext:
    """Carry one stream's task-local runtime ownership across task boundaries."""

    def __init__(self) -> None:
        self._cleanup_identities = _SESSION_CLEANUP_IDENTITIES.get()
        fences = _SESSION_RUN_FENCES.get()
        self._fences = None if fences is None else dict(fences)
        fence_owners = _SESSION_RUN_FENCE_OWNERS.get()
        self._fence_owners = None if fence_owners is None else dict(fence_owners)
        interaction_ids = _SESSION_INTERACTION_IDS.get()
        self._interaction_ids = None if interaction_ids is None else dict(interaction_ids)
        interaction_started_at = _SESSION_INTERACTION_STARTED_AT.get()
        self._interaction_started_at = (
            None if interaction_started_at is None else dict(interaction_started_at)
        )
        recovered_active_through = _SESSION_INTERACTION_RECOVERED_ACTIVE_THROUGH.get()
        self._recovered_active_through = (
            None if recovered_active_through is None else dict(recovered_active_through)
        )
        invocation_interaction_ids = _SESSION_INVOCATION_INTERACTION_IDS.get()
        self._invocation_interaction_ids = (
            None if invocation_interaction_ids is None else dict(invocation_interaction_ids)
        )
        settled_interaction_ids = _SESSION_SETTLED_INTERACTION_IDS.get()
        self._settled_interaction_ids = (
            None if settled_interaction_ids is None else dict(settled_interaction_ids)
        )
        settled_interaction_transitions = _SESSION_SETTLED_INTERACTION_TRANSITIONS.get()
        self._settled_interaction_transitions = (
            None
            if settled_interaction_transitions is None
            else dict(settled_interaction_transitions)
        )
        terminal_events = _SESSION_INVOCATION_TERMINAL_EVENTS.get()
        self._terminal_events = None if terminal_events is None else dict(terminal_events)

    @classmethod
    def current_or_new(cls) -> _SessionRunFenceContext:
        current = _ACTIVE_SESSION_RUN_FENCE_CONTEXT.get()
        task = asyncio.current_task()
        if current is not None and current[1]() is task:
            return current[0]
        return cls()

    @contextmanager
    def activate(self) -> Iterator[None]:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("Session run-fence context requires a running task.")
        current = _ACTIVE_SESSION_RUN_FENCE_CONTEXT.get()
        if current is not None and current[0] is self and current[1]() is task:
            yield
            return
        fences = None if self._fences is None else dict(self._fences)
        fence_token = _SESSION_RUN_FENCES.set(fences)
        cleanup_identity_token = _SESSION_CLEANUP_IDENTITIES.set(self._cleanup_identities)
        fence_owners = None if self._fence_owners is None else dict(self._fence_owners)
        fence_owner_token = _SESSION_RUN_FENCE_OWNERS.set(fence_owners)
        interaction_ids = None if self._interaction_ids is None else dict(self._interaction_ids)
        interaction_id_token = _SESSION_INTERACTION_IDS.set(interaction_ids)
        interaction_started_at = (
            None if self._interaction_started_at is None else dict(self._interaction_started_at)
        )
        interaction_started_at_token = _SESSION_INTERACTION_STARTED_AT.set(interaction_started_at)
        recovered_active_through = (
            None if self._recovered_active_through is None else dict(self._recovered_active_through)
        )
        recovered_active_through_token = _SESSION_INTERACTION_RECOVERED_ACTIVE_THROUGH.set(
            recovered_active_through
        )
        invocation_interaction_ids = (
            None
            if self._invocation_interaction_ids is None
            else dict(self._invocation_interaction_ids)
        )
        invocation_interaction_ids_token = _SESSION_INVOCATION_INTERACTION_IDS.set(
            invocation_interaction_ids
        )
        settled_interaction_ids = (
            None if self._settled_interaction_ids is None else dict(self._settled_interaction_ids)
        )
        settled_interaction_ids_token = _SESSION_SETTLED_INTERACTION_IDS.set(
            settled_interaction_ids
        )
        settled_interaction_transitions = (
            None
            if self._settled_interaction_transitions is None
            else dict(self._settled_interaction_transitions)
        )
        settled_interaction_transitions_token = _SESSION_SETTLED_INTERACTION_TRANSITIONS.set(
            settled_interaction_transitions
        )
        terminal_events = None if self._terminal_events is None else dict(self._terminal_events)
        terminal_events_token = _SESSION_INVOCATION_TERMINAL_EVENTS.set(terminal_events)
        context_token = _ACTIVE_SESSION_RUN_FENCE_CONTEXT.set((self, ref(task)))
        try:
            yield
        finally:
            self._cleanup_identities = _SESSION_CLEANUP_IDENTITIES.get()
            current = _SESSION_RUN_FENCES.get()
            self._fences = None if current is None else dict(current)
            current_fence_owners = _SESSION_RUN_FENCE_OWNERS.get()
            self._fence_owners = (
                None if current_fence_owners is None else dict(current_fence_owners)
            )
            current_interaction_ids = _SESSION_INTERACTION_IDS.get()
            self._interaction_ids = (
                None if current_interaction_ids is None else dict(current_interaction_ids)
            )
            current_interaction_started_at = _SESSION_INTERACTION_STARTED_AT.get()
            self._interaction_started_at = (
                None
                if current_interaction_started_at is None
                else dict(current_interaction_started_at)
            )
            current_recovered_active_through = _SESSION_INTERACTION_RECOVERED_ACTIVE_THROUGH.get()
            self._recovered_active_through = (
                None
                if current_recovered_active_through is None
                else dict(current_recovered_active_through)
            )
            current_invocation_interaction_ids = _SESSION_INVOCATION_INTERACTION_IDS.get()
            self._invocation_interaction_ids = (
                None
                if current_invocation_interaction_ids is None
                else dict(current_invocation_interaction_ids)
            )
            current_settled_interaction_ids = _SESSION_SETTLED_INTERACTION_IDS.get()
            self._settled_interaction_ids = (
                None
                if current_settled_interaction_ids is None
                else dict(current_settled_interaction_ids)
            )
            current_settled_interaction_transitions = _SESSION_SETTLED_INTERACTION_TRANSITIONS.get()
            self._settled_interaction_transitions = (
                None
                if current_settled_interaction_transitions is None
                else dict(current_settled_interaction_transitions)
            )
            current_terminal_events = _SESSION_INVOCATION_TERMINAL_EVENTS.get()
            self._terminal_events = (
                None if current_terminal_events is None else dict(current_terminal_events)
            )
            _ACTIVE_SESSION_RUN_FENCE_CONTEXT.reset(context_token)
            _SESSION_INVOCATION_TERMINAL_EVENTS.reset(terminal_events_token)
            _SESSION_SETTLED_INTERACTION_TRANSITIONS.reset(settled_interaction_transitions_token)
            _SESSION_SETTLED_INTERACTION_IDS.reset(settled_interaction_ids_token)
            _SESSION_INVOCATION_INTERACTION_IDS.reset(invocation_interaction_ids_token)
            _SESSION_INTERACTION_RECOVERED_ACTIVE_THROUGH.reset(recovered_active_through_token)
            _SESSION_INTERACTION_STARTED_AT.reset(interaction_started_at_token)
            _SESSION_INTERACTION_IDS.reset(interaction_id_token)
            _SESSION_RUN_FENCE_OWNERS.reset(fence_owner_token)
            _SESSION_CLEANUP_IDENTITIES.reset(cleanup_identity_token)
            _SESSION_RUN_FENCES.reset(fence_token)


class _SessionRunFenceOwnership:
    """One transferable, exactly retired process-local run-fence owner."""

    __slots__ = (
        "_retired",
        "_retirement_callbacks",
        "run_epoch",
        "session_id",
        "session_instance_id",
    )

    def __init__(
        self, *, session_id: str, run_epoch: int, session_instance_id: str | None = None
    ) -> None:
        self.session_instance_id = session_instance_id
        self.session_id = session_id
        self.run_epoch = run_epoch
        self._retired = False
        self._retirement_callbacks: list[Callable[[], None]] = []

    def on_retire(self, callback: Callable[[], None]) -> None:
        if self._retired:
            callback()
        else:
            self._retirement_callbacks.append(callback)

    @property
    def retired(self) -> bool:
        return self._retired

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Install this exact owner while an independent supervisor finalizes it."""

        if self._retired:
            raise RuntimeError("A retired session run-fence owner cannot be reactivated.")
        fences = _SESSION_RUN_FENCES.get()
        owners = _SESSION_RUN_FENCE_OWNERS.get()
        current_owner = None if owners is None else owners.get(self.session_id)
        current_epoch = None if fences is None else fences.get(self.session_id)
        if (
            current_owner is self
            and current_epoch == self.run_epoch
            and fences is not None
            and next(reversed(fences)) == self.session_id
        ):
            yield
            return
        updated_fences = dict(fences or {})
        updated_fences.pop(self.session_id, None)
        updated_fences[self.session_id] = self.run_epoch
        fence_token = _SESSION_RUN_FENCES.set(updated_fences)
        updated_owners = dict(owners or {})
        updated_owners[self.session_id] = self
        owner_token = _SESSION_RUN_FENCE_OWNERS.set(updated_owners)
        try:
            yield
        finally:
            _SESSION_RUN_FENCE_OWNERS.reset(owner_token)
            _SESSION_RUN_FENCES.reset(fence_token)

    def retire(self) -> bool:
        """Retire this owner in every copied context and detach the current copy."""

        if self._retired:
            return False
        self._retired = True
        callbacks, self._retirement_callbacks = self._retirement_callbacks, []
        for callback in callbacks:
            callback()
        owners = _SESSION_RUN_FENCE_OWNERS.get()
        fences = _SESSION_RUN_FENCES.get()
        if owners is not None and owners.get(self.session_id) is self:
            remaining_owners = dict(owners)
            remaining_owners.pop(self.session_id)
            _SESSION_RUN_FENCE_OWNERS.set(remaining_owners or None)
            if fences is not None and fences.get(self.session_id) == self.run_epoch:
                remaining_fences = dict(fences)
                remaining_fences.pop(self.session_id)
                _SESSION_RUN_FENCES.set(remaining_fences or None)
        return True


_ACTIVE_SESSION_RUN_FENCE_CONTEXT: ContextVar[
    tuple[_SessionRunFenceContext, ReferenceType[asyncio.Task[Any]]] | None
] = ContextVar(
    "cayu_active_session_run_fence_context",
    default=None,
)


def _current_session_run_epoch(session_id: str) -> int | None:
    fences = _SESSION_RUN_FENCES.get()
    run_epoch = None if fences is None else fences.get(session_id)
    if run_epoch is None:
        return None
    owners = _SESSION_RUN_FENCE_OWNERS.get()
    ownership = None if owners is None else owners.get(session_id)
    if ownership is not None and ownership.retired and ownership.run_epoch == run_epoch:
        assert fences is not None
        assert owners is not None
        remaining_fences = dict(fences)
        remaining_fences.pop(session_id)
        _SESSION_RUN_FENCES.set(remaining_fences or None)
        remaining_owners = dict(owners)
        remaining_owners.pop(session_id)
        _SESSION_RUN_FENCE_OWNERS.set(remaining_owners or None)
        return None
    return run_epoch


def _activate_session_run_fence(session: Session) -> None:
    identities = dict(_SESSION_CLEANUP_IDENTITIES.get() or {})
    identities.pop(session.id, None)
    identities[session.id] = (session.instance_id, session.run_epoch)
    _SESSION_CLEANUP_IDENTITIES.set(identities)
    fences = dict(_SESSION_RUN_FENCES.get() or {})
    fences.pop(session.id, None)
    fences[session.id] = session.run_epoch
    _SESSION_RUN_FENCES.set(fences)
    owners = _SESSION_RUN_FENCE_OWNERS.get()
    if owners is not None and session.id in owners:
        remaining_owners = dict(owners)
        remaining_owners.pop(session.id)
        _SESSION_RUN_FENCE_OWNERS.set(remaining_owners or None)


def _activate_owned_session_run_fence(session: Session) -> _SessionRunFenceOwnership:
    _activate_session_run_fence(session)
    ownership = _SessionRunFenceOwnership(
        session_id=session.id,
        run_epoch=session.run_epoch,
        session_instance_id=session.instance_id,
    )
    owners = dict(_SESSION_RUN_FENCE_OWNERS.get() or {})
    owners[session.id] = ownership
    _SESSION_RUN_FENCE_OWNERS.set(owners)
    return ownership


def _current_recovery_cleanup_identity() -> tuple[str, str, int] | None:
    """Project the innermost admitted identity without granting recovery authority."""
    identities = _SESSION_CLEANUP_IDENTITIES.get() or {}
    fences = _SESSION_RUN_FENCES.get() or {}
    owners = _SESSION_RUN_FENCE_OWNERS.get() or {}
    if not fences:
        return None
    session_id = next(reversed(fences))
    epoch = fences[session_id]
    owner = owners.get(session_id)
    if owner is not None and owner.session_instance_id is not None and owner.run_epoch == epoch:
        return session_id, owner.session_instance_id, epoch
    identity = identities.get(session_id)
    if identity is None or identity[1] != epoch:
        return None
    return session_id, identity[0], epoch


def _deactivate_session_run_fence(session_id: str) -> None:
    identities = dict(_SESSION_CLEANUP_IDENTITIES.get() or {})
    identities.pop(session_id, None)
    _SESSION_CLEANUP_IDENTITIES.set(identities or None)
    fences = _SESSION_RUN_FENCES.get()
    if fences is not None and session_id in fences:
        remaining = dict(fences)
        remaining.pop(session_id)
        _SESSION_RUN_FENCES.set(remaining or None)
    owners = _SESSION_RUN_FENCE_OWNERS.get()
    if owners is not None and session_id in owners:
        remaining_owners = dict(owners)
        remaining_owners.pop(session_id)
        _SESSION_RUN_FENCE_OWNERS.set(remaining_owners or None)


def _assert_session_run_epoch(session_id: str, session: Session) -> None:
    _assert_session_run_epoch_value(session_id, session.run_epoch)


def _assert_session_run_epoch_value(session_id: str, current_run_epoch: int) -> None:
    expected = _current_session_run_epoch(session_id)
    if expected is not None and current_run_epoch != expected:
        raise SessionRunFenced(
            f"Session run epoch no longer owns {session_id}: expected {expected}, "
            f"current {current_run_epoch}."
        )
