"""Every terminal outcome is finished from its durable decision after a crash.

A failed turn and an approval pause commit their terminal transition before
their TURN_COMPLETED and session event are published and before their run fence
is released, as a completed turn does. The fault-injection harness kills the
process at every store write after that transition, over the in-memory and
SQLite stores. A fresh app's recovery must then publish exactly the events the
outcome owed and leave the session able to continue. Closing the event stream
at each event the outcome publishes must finish the same publication.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import pytest
from tests.fault_injection import explore, harness
from tests.fault_injection.harness import (
    APPROVAL_PROMPT,
    FaultKind,
    Injector,
    ManualClock,
    _instrumented_store_type,
)

from cayu.events import EventType
from cayu.messages import Message, TextPart
from cayu.sessions.records import SessionStatus
from cayu.sessions.store import SessionStore
from cayu.storage.sqlite import SQLiteSessionStore

_SQLiteSessionStore = _instrumented_store_type(SQLiteSessionStore)

FAIL_PROMPT = "fault-injection: fail the turn"
_TERMINAL_TRANSITION = "store.settle_session_invocation:enter"
_OWED_INVARIANT = "terminal-outcome-evidence-incomplete"
_SESSION_EVENT = {
    SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
    SessionStatus.FAILED: EventType.SESSION_FAILED,
    SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
}
# The interaction outcomes whose run publishes a TURN_COMPLETED of its own.
_DECIDED_INTERACTION_EVENTS = frozenset(
    {EventType.INTERACTION_COMPLETED, EventType.INTERACTION_FAILED, EventType.INTERACTION_PAUSED}
)
_SETTLED_INTERACTION_EVENTS = _DECIDED_INTERACTION_EVENTS | {EventType.INTERACTION_INTERRUPTED}

SCENARIOS = {
    "failure": explore.Scenario(
        "failure", explore._streamed(lambda app, _: explore.run_turn(app, FAIL_PROMPT))
    ),
    "approval-pause": explore.Scenario(
        "approval-pause", explore._streamed(lambda app, _: explore.run_turn(app, APPROVAL_PROMPT))
    ),
}


async def _owed_evidence_violations(store: SessionStore) -> list[tuple[str, str]]:
    """After the last settled interaction: one TURN_COMPLETED and one session event."""

    session = await store.load(explore.SESSION_ID)
    if session is None or session.status not in _SESSION_EVENT:
        return []
    events = await store.load_events(explore.SESSION_ID)
    settled = [
        index for index, event in enumerate(events) if event.type in _SETTLED_INTERACTION_EVENTS
    ]
    if not settled:
        return []
    tail = [event.type for event in events[settled[-1] + 1 :]]
    turns = tail.count(EventType.TURN_COMPLETED)
    decided = events[settled[-1]].type in _DECIDED_INTERACTION_EVENTS
    if (turns != 1 if decided else turns > 1) or tail.count(_SESSION_EVENT[session.status]) != 1:
        return [(_OWED_INVARIANT, f"status={session.status.value} after={[str(t) for t in tail]}")]
    return []


@pytest.fixture(autouse=True)
def harness_with_outcomes(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    scripted = harness._scripted_response

    def scripted_or_failing(messages: list[Message]):
        last = messages[-1] if messages else None
        prompt = "".join(
            part.text for part in (last.content if last else ()) if type(part) is TextPart
        )
        if prompt == FAIL_PROMPT:
            raise RuntimeError("fault-injection: the provider failed")
        return scripted(messages)

    store_violations = explore.store_violations

    async def with_owed_evidence(store: SessionStore) -> list[tuple[str, str]]:
        return [*await store_violations(store), *await _owed_evidence_violations(store)]

    monkeypatch.setattr(harness, "_scripted_response", scripted_or_failing)
    monkeypatch.setattr(explore, "store_violations", with_owed_evidence)
    yield


@pytest.fixture(params=["memory", "sqlite"])
def backend(request: pytest.FixtureRequest, tmp_path, monkeypatch) -> str:
    if request.param == "sqlite":
        names = itertools.count()

        def session_store(injector: Injector, clock: ManualClock) -> SQLiteSessionStore:
            # Each point runs on a store of its own; the previous point's is done.
            for stale in tmp_path.glob("sessions-*"):
                stale.unlink()
            store = _SQLiteSessionStore(
                tmp_path / f"sessions-{next(names)}.sqlite", ownership_clock=clock
            )
            store.fault_injector = injector
            return store

        monkeypatch.setattr(explore, "session_store", session_store)
    return request.param


def _points_after_terminal_transition(scenario: explore.Scenario) -> list[str]:
    """Every crash point from the terminal transition's commit to the fence release."""

    points = explore.baseline_points(scenario, FaultKind.CRASH)
    transitions = [index for index, point in enumerate(points) if _TERMINAL_TRANSITION in point]
    assert transitions, "the scenario never commits its terminal transition"
    return points[transitions[-1] + 1 :]


def _violations(scenario: explore.Scenario, fault: FaultKind, points: list[str]) -> list[str]:
    found = []
    for point in points:
        result = explore.run_point(scenario, fault, point)
        assert result.reached, point
        found.extend(violation.describe() for violation in result.violations)
    return found


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_recovery_finishes_an_outcome_cut_short_by_a_crash(backend: str, scenario: str) -> None:
    points = _points_after_terminal_transition(SCENARIOS[scenario])

    violations = _violations(SCENARIOS[scenario], FaultKind.CRASH, points)

    assert not violations, "\n".join(violations)


@pytest.mark.parametrize(
    ("scenario", "events"),
    [
        ("failure", ("event:interaction.failed#0", "event:turn.completed#0")),
        ("approval-pause", ("event:interaction.paused#0", "event:turn.completed#0")),
    ],
)
def test_closing_at_an_outcome_event_finishes_its_publication(
    backend: str, scenario: str, events: tuple[str, ...]
) -> None:
    violations = _violations(SCENARIOS[scenario], FaultKind.STREAM_CLOSE, list(events))

    assert not violations, "\n".join(violations)
