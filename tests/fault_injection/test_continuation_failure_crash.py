"""A failed approval continuation must restore its pause with settled, durable evidence.

When an approval continuation fails before its close commits, the coordinator
restores the resumable INTERRUPTED pause with a raw status write and then
publishes SESSION_INTERRUPTED as a second write. The resumed interaction gets
no terminal transition, and a crash between the two writes loses the session
event for good: its ID is random and nothing owes it. Both cases are strict
expected failures until those continuations publish their interaction
transition with what it owes (the terminal-decision ratchet lists the sites).
"""

from __future__ import annotations

import pytest
from tests.fault_injection import explore
from tests.fault_injection.harness import FaultKind
from tests.fault_injection.invariants import Invariant, Stage
from tests.fault_injection.known_failures import FINDINGS

from cayu.events import EventType
from cayu.runtime import _approval_publication
from cayu.sessions.records import SessionStatus

pytestmark = pytest.mark.fault_injection

_SCENARIO = explore.Scenario(
    "approval-close-fails",
    explore._streamed(lambda app, setup: explore.approve(app, explore.pending_approval(setup))),
    setup=explore._turn(explore.APPROVAL_PROMPT),
)
# The SESSION_INTERRUPTED write right after the raw INTERRUPTED status write.
_SESSION_EVENT_WRITE = "store.append_event_and_claim_side_effect:enter#4"
_SESSION_EVENT_MISSING = "terminal-session-event-missing"


@pytest.fixture(autouse=True)
def failing_close(monkeypatch: pytest.MonkeyPatch) -> None:
    publish = _approval_publication.publish_approval_with_exact_replay

    async def fail_close(prepared, **kwargs):
        if prepared._request.kind == "approval-close":
            raise RuntimeError("fault-injection: approval close failed")
        return await publish(prepared, **kwargs)

    monkeypatch.setattr(_approval_publication, "publish_approval_with_exact_replay", fail_close)


@pytest.fixture(autouse=True)
def session_event_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Also require an interrupted session's last resume to end in its session event."""

    store_violations = explore.store_violations

    async def with_session_event(store):
        found = await store_violations(store)
        session = await store.load(explore.SESSION_ID)
        if session is not None and session.status is SessionStatus.INTERRUPTED:
            types = [event.type for event in await store.load_events(explore.SESSION_ID)]
            resumed = max(
                (i for i, kind in enumerate(types) if kind is EventType.SESSION_RESUMED),
                default=-1,
            )
            if EventType.SESSION_INTERRUPTED not in types[resumed + 1 :]:
                found.append((_SESSION_EVENT_MISSING, "no SESSION_INTERRUPTED after the resume"))
        return found

    monkeypatch.setattr(explore, "store_violations", with_session_event)


def _recovered(result: explore.CaseResult, invariant: str) -> list[str]:
    return [
        violation.describe()
        for violation in result.violations
        if violation.stage == Stage.RECOVERED and violation.invariant == invariant
    ]


def test_the_crash_point_follows_the_raw_status_write() -> None:
    baseline = explore._run_isolated(_SCENARIO, None, None)
    points = explore.injection_points(baseline, FaultKind.CRASH)
    assert points[points.index(_SESSION_EVENT_WRITE) - 1] == "store.update_status:enter#0"
    assert not _recovered(baseline, _SESSION_EVENT_MISSING)
    assert explore.run_point(_SCENARIO, FaultKind.CRASH, _SESSION_EVENT_WRITE).reached


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=FINDINGS["continuation-pause-interaction-open"].summary,
)
def test_a_failed_continuation_settles_its_resumed_interaction() -> None:
    result = explore._run_isolated(_SCENARIO, None, None)

    assert not _recovered(result, Invariant.INTERACTION_UNSETTLED)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=FINDINGS["continuation-pause-event-lost"].summary,
)
def test_a_crash_after_the_restored_pause_keeps_its_session_event() -> None:
    result = explore.run_point(_SCENARIO, FaultKind.CRASH, _SESSION_EVENT_WRITE)

    assert not _recovered(result, _SESSION_EVENT_MISSING)
