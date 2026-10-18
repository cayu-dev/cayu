"""Host-enforced reads for enrolled fork controllers, on every built-in session store."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_session_fork_overlays import _controller_toolset, _http
from tests.runtime.runtime_session_support import (
    _pending_tool_approval_from_public_event,
    collect_events,
    collect_tool_approval_events,
)
from tests.runtime.runtime_supervision_support import (
    WINDOW_END,
    WINDOW_START,
    approval_app,
    resolve_approval,
    supervision_window,
)
from tests.runtime.runtime_tool_support import RequireApprovalPolicy, SideEffectTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.actors import ResolutionActor, ResolutionActorSource
from cayu.approvals.tools import ToolApprovalDecision
from cayu.events import EventType
from cayu.execution_profiles import ExecutionProfileAdoptionIntent
from cayu.fork_controllers import ForkController, ForkControllerDenied
from cayu.messages import Message
from cayu.sessions import ForkOverlay, ForkOverlayToolGrant, ForkSessionRequest
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.queries import SessionQuery
from cayu.sessions.requests import RunRequest
from cayu.sessions.supervision_events import SupervisionEventCursorInvalid
from cayu.tools import ToolCapabilityCeiling

__all__ = ["conformance_postgres_dsn", "session_store_case"]

AGENT = "assistant"
OTHER_AGENT = "payroll"
COMPASS = "compass"
AUDITOR = "auditor"
COMPASS_SUBJECT = "svc:compass"
AUDITOR_SUBJECT = "svc:auditor"
BODY = f"{AGENT}@fork-controller:{COMPASS}"
LOOKUP = "mcp__compass__lookup"
# The enrolled agent's own host session, a foreign agent's session, the
# controller's own forks (body and instructions-only), and another controller's
# instructions-only fork of the same host session.
HOST = "host-session"
FOREIGN = "payroll-session"
OWN_BODY_FORK = "compass-body-fork"
OWN_NOTES_FORK = "compass-notes-fork"
OTHER_CONTROLLER_FORK = "auditor-fork"
COMPASS_READABLE = {HOST, OWN_BODY_FORK, OWN_NOTES_FORK}
UNREADABLE = (FOREIGN, OTHER_CONTROLLER_FORK, "no-such-session")


def _app(
    store,
    *,
    compass_agents: tuple[str, ...] = (AGENT, OTHER_AGENT),
    sessions: int = 2,
) -> CayuApp:
    app = approval_app(store, sessions=sessions)
    app.register_agent(
        AgentSpec(name=OTHER_AGENT, model="fake-model"),
        tools=[SideEffectTool()],
        tool_policy=RequireApprovalPolicy(),
    )
    toolset, _server = _controller_toolset()
    app.enroll_fork_controller(
        ForkController(
            name=COMPASS,
            subjects=(COMPASS_SUBJECT,),
            agents=compass_agents,
            mcp_toolsets=(toolset,),
        )
    )
    app.enroll_fork_controller(
        ForkController(name=AUDITOR, subjects=(AUDITOR_SUBJECT,), agents=(AGENT,))
    )
    return app


async def _settled_approval(app: CayuApp, store, session_id: str, agent: str) -> None:
    """Run one session to an approval and deny it: one settled supervision touch."""

    events = await collect_events(
        app,
        RunRequest(
            agent_name=agent,
            session_id=session_id,
            messages=[Message.text("user", "use the tool")],
        ),
    )
    requested = next(e for e in events if e.type == EventType.TOOL_CALL_APPROVAL_REQUESTED)
    pending = await _pending_tool_approval_from_public_event(store, requested)
    await collect_tool_approval_events(
        app, resolve_approval(session_id, pending, ToolApprovalDecision.DENY)
    )


def _notes(controller: str) -> ForkOverlay:
    return ForkOverlay(controller=controller, instructions="Investigate the denial only.")


async def _world(store, **options: Any) -> CayuApp:
    """Build every kind of session a controller read must tell apart."""

    app = _app(store, **options)
    await _settled_approval(app, store, HOST, AGENT)
    await _settled_approval(app, store, FOREIGN, OTHER_AGENT)
    compass = app.fork_controller_access(COMPASS_SUBJECT)
    auditor = app.fork_controller_access(AUDITOR_SUBJECT)
    for stream in (
        compass.fork(
            ForkSessionRequest(
                source_session_id=HOST,
                session_id=OWN_BODY_FORK,
                tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                execution_profile_selection="current_child",
                profile_adoption=ExecutionProfileAdoptionIntent(
                    idempotency_key="compass-body-fork",
                    reason="Compass investigates one denial.",
                    requested_by=ResolutionActor(
                        subject=COMPASS_SUBJECT, source=ResolutionActorSource.REQUEST
                    ),
                ),
                overlay=ForkOverlay(
                    controller=COMPASS, tools=(ForkOverlayToolGrant(tool_name=LOOKUP),)
                ),
            )
        ),
        compass.fork(
            ForkSessionRequest(
                source_session_id=HOST, session_id=OWN_NOTES_FORK, overlay=_notes(COMPASS)
            )
        ),
        auditor.fork(
            ForkSessionRequest(
                source_session_id=HOST,
                session_id=OTHER_CONTROLLER_FORK,
                overlay=_notes(AUDITOR),
            )
        ),
    ):
        async for _event in stream:
            pass
    return app


def _controller_server(app: CayuApp, subject: str = COMPASS_SUBJECT):
    from fastapi import FastAPI

    from cayu.server.fork_controller import create_fork_controller_router

    async def authenticate(_request):
        return subject

    server = FastAPI()
    server.include_router(create_fork_controller_router(app, authenticate=authenticate))
    return server


def _in_store(case, scenario):
    async def run():
        store = await _open_store(case)
        try:
            return await scenario(store)
        finally:
            await _close_store(store)

    return asyncio.run(run())


async def _denied(operation) -> bool:
    try:
        await operation
    except ForkControllerDenied as exc:
        return str(exc) == "The session is not readable by this controller."
    return False


def test_store_declares_fork_controller_read_support(session_store_case):
    async def scenario(store):
        return type(store).__dict__.get("supports_fork_controller_reads")

    assert _in_store(session_store_case, scenario) is True


def test_controller_reads_only_enrolled_agents_and_its_own_forks(session_store_case):
    async def scenario(store):
        app = await _world(store, compass_agents=(AGENT,))
        compass = app.fork_controller_access(COMPASS_SUBJECT)
        listed = await compass.sessions(SessionQuery(include_total_count=True))
        loaded = {sid: (await compass.session(sid)).agent_name for sid in COMPASS_READABLE}
        denied = {sid: await _denied(compass.session(sid)) for sid in UNREADABLE}
        denied_events = {sid: await _denied(compass.events(sid)) for sid in UNREADABLE}
        own_events = await compass.events(OWN_NOTES_FORK)
        everything = await compass.query_events(EventQuery(limit=5000))
        named = await compass.query_events(
            EventQuery(session_ids=(FOREIGN, OTHER_CONTROLLER_FORK, HOST), limit=5000)
        )
        only_foreign = await compass.query_events(EventQuery(session_ids=(FOREIGN,)))
        single_foreign = await compass.query_events(EventQuery(session_id=FOREIGN))
        unforked = await compass.query_events(EventQuery(limit=5000), exclude_forked_sessions=True)
        return (
            listed,
            loaded,
            denied,
            denied_events,
            own_events,
            everything,
            named,
            (
                only_foreign,
                single_foreign,
            ),
            unforked,
        )

    (
        listed,
        loaded,
        denied,
        denied_events,
        own_events,
        everything,
        named,
        foreign_only,
        unforked,
    ) = _in_store(session_store_case, scenario)
    assert {session.id for session in listed.sessions} == COMPASS_READABLE
    assert listed.total_count == len(COMPASS_READABLE)
    assert loaded == {HOST: AGENT, OWN_BODY_FORK: BODY, OWN_NOTES_FORK: AGENT}
    # Absent, foreign-agent and other-controller sessions are denied identically.
    assert all(denied.values()), denied
    assert all(denied_events.values()), denied_events
    assert own_events and {record.event.session_id for record in own_events} == {OWN_NOTES_FORK}
    assert {record.event.session_id for record in everything} == COMPASS_READABLE
    assert {record.event.session_id for record in named} == {HOST}
    assert foreign_only == ([], [])
    # Excluding forks drops this controller's own forks too.
    assert {record.event.session_id for record in unforked} == {HOST}


def test_controller_session_lists_filter_by_creation_time(session_store_case):
    from datetime import datetime

    async def scenario(store):
        app = await _world(store, compass_agents=(AGENT,))
        compass = app.fork_controller_access(COMPASS_SUBJECT)
        cutoff = (await store.load(HOST)).created_at
        sdk = await compass.sessions(SessionQuery(created_after=cutoff, include_total_count=True))
        server = _controller_server(app)
        http = await _http(
            server, "GET", "/fork-controller/sessions", params={"created_after": cutoff.isoformat()}
        )
        naive = await _http(
            server,
            "GET",
            "/fork-controller/sessions",
            params={"created_after": cutoff.replace(tzinfo=None).isoformat()},
        )
        with pytest.raises(ValueError, match="created_after must be timezone-aware"):
            SessionQuery(created_after=datetime(2026, 1, 1))
        return sdk, http, naive

    sdk, http, naive = _in_store(session_store_case, scenario)
    # Strictly after the host session; the read limit still drops the other
    # controller's fork, created after the cutoff too.
    later = {OWN_BODY_FORK, OWN_NOTES_FORK}
    assert {session.id for session in sdk.sessions} == later
    assert sdk.total_count == len(later)
    assert http.status_code == 200
    assert {item["id"] for item in http.json()["sessions"]} == later
    assert http.json()["total_count"] == len(later)
    assert naive.status_code == 422


def test_controller_refuses_authority_bearing_query_fields() -> None:
    from cayu.sessions._access_rules import _query_bounds
    from cayu.sessions._fork_controller_bounds import _ForkControllerReadBounds

    async def scenario():
        store = InMemorySessionStore()
        app = _app(store)
        compass = app.fork_controller_access(COMPASS_SUBJECT)
        refused = []
        for query in (
            EventQuery(causal_budget_id="budget"),
            EventQuery(budget_limit_id="limit"),
            EventQuery(workflow_name="close"),
            EventQuery(
                session_id="s",
                workflow_name="close",
                workflow_attempt_id="attempt",
                workflow_step_id="step",
                event_type="workflow.step.started",
            ),
            EventQuery(model_step_id="step"),
            EventQuery(session_id="s", event_id="event"),
        ):
            with pytest.raises(ValueError, match="cannot set"):
                await compass.query_events(query)
            refused.append(query)
        with pytest.raises(ValueError, match="causal_budget_id"):
            await compass.sessions(SessionQuery(causal_budget_id="budget"))
        # A controller read never replaces another reader's bound.
        token = _query_bounds.set(
            _ForkControllerReadBounds(controller=AUDITOR, agent_names=(AGENT,))
        )
        try:
            for operation in (
                compass.sessions(),
                compass.session(HOST),
                compass.query_events(EventQuery()),
            ):
                with pytest.raises(ForkControllerDenied, match="another read bound"):
                    await operation
        finally:
            _query_bounds.reset(token)
        with pytest.raises(ValueError, match="causal_budget_id"):
            await compass.supervision_events(
                supervision_window(sessions={"causal_budget_id": "budget"})
            )
        return refused

    assert len(asyncio.run(scenario())) == 6


def test_narrowing_enrollment_on_restart_cuts_reads_immediately(session_store_case):
    async def scenario(store):
        first = await _world(store)
        before = await first.fork_controller_access(COMPASS_SUBJECT).sessions()
        # The host restarts with the controller enrolled for one agent only.
        restarted = _app(store, compass_agents=(AGENT,))
        compass = restarted.fork_controller_access(COMPASS_SUBJECT)
        after = await compass.sessions()
        events = await compass.query_events(EventQuery(limit=5000))
        page = await compass.supervision_events(supervision_window())
        return before, after, events, page, await _denied(compass.session(FOREIGN))

    before, after, events, page, foreign_denied = _in_store(session_store_case, scenario)
    assert FOREIGN in {session.id for session in before.sessions}
    assert {session.id for session in after.sessions} == COMPASS_READABLE
    assert FOREIGN not in {record.event.session_id for record in events}
    assert {event.session_id for event in page.events} == {HOST}
    assert foreign_denied


def test_supervision_reads_are_bounded_and_cursors_bound_to_the_controller(session_store_case):
    async def scenario(store):
        app = _app(store)
        await _settled_approval(app, store, HOST, AGENT)
        await _settled_approval(app, store, FOREIGN, OTHER_AGENT)
        compass = app.fork_controller_access(COMPASS_SUBJECT)
        auditor = app.fork_controller_access(AUDITOR_SUBJECT)
        whole = await compass.supervision_events(supervision_window())
        first = await compass.supervision_events(supervision_window(limit=1))
        second = await compass.supervision_events(
            supervision_window(limit=1, cursor=first.next_cursor)
        )
        auditor_page = await auditor.supervision_events(supervision_window())
        control_plane = await store.query_supervision_events(
            supervision_window(limit=1), include_content=False
        )
        rejected = []
        for reader, cursor in (
            (auditor, first.next_cursor),
            (compass, control_plane.next_cursor),
            (compass, (await auditor.supervision_events(supervision_window(limit=1))).next_cursor),
        ):
            try:
                await reader.supervision_events(supervision_window(limit=1, cursor=cursor))
            except SupervisionEventCursorInvalid:
                rejected.append(True)
        return whole, first, second, auditor_page, rejected

    whole, first, second, auditor_page, rejected = _in_store(session_store_case, scenario)
    assert [event.session_id for event in whole.events] == [HOST, FOREIGN]
    assert [event.session_id for event in first.events + second.events] == [HOST, FOREIGN]
    assert [event.session_id for event in auditor_page.events] == [HOST]
    assert rejected == [True, True, True]


class _UndeclaredStore(InMemorySessionStore):
    """A store subclass that never declared fork-controller read support."""


def test_a_store_without_support_fails_closed() -> None:
    async def scenario():
        store = _UndeclaredStore()
        app = _app(store)
        compass = app.fork_controller_access(COMPASS_SUBJECT)
        failures = []
        for operation in (
            compass.sessions(),
            compass.session(HOST),
            compass.events(HOST),
            compass.query_events(EventQuery()),
            compass.supervision_events(supervision_window()),
        ):
            try:
                await operation
            except NotImplementedError:
                failures.append(True)
        server = _controller_server(app)
        statuses = [
            (await _http(server, "GET", "/fork-controller/sessions")).status_code,
            (await _http(server, "GET", f"/fork-controller/sessions/{HOST}")).status_code,
            (await _http(server, "GET", f"/fork-controller/sessions/{HOST}/events")).status_code,
            (await _http(server, "POST", "/fork-controller/events/query", json={})).status_code,
            (
                await _http(
                    server,
                    "POST",
                    "/fork-controller/supervision-events/query",
                    json={"start_at": WINDOW_START.isoformat(), "end_at": WINDOW_END.isoformat()},
                )
            ).status_code,
        ]
        return failures, statuses

    failures, statuses = asyncio.run(scenario())
    assert failures == [True] * 5
    assert statuses == [501] * 5


def test_an_unknown_read_bound_raises(tmp_path) -> None:
    from cayu.sessions._access_rules import _query_bounds, session_read_admitted
    from cayu.storage import SQLiteSessionStore
    from cayu.storage._session_store_sql import session_access_clause
    from cayu.storage._sqlite_session_queries import SQL_DIALECT

    async def scenario():
        failures = []
        for store in (InMemorySessionStore(), SQLiteSessionStore(tmp_path / "s.sqlite")):
            app = _app(store)
            await _settled_approval(app, store, HOST, AGENT)
            session = await store.load(HOST)
            token = _query_bounds.set(object())  # type: ignore[arg-type]
            try:
                await store.query_events(EventQuery())
            except TypeError:
                failures.append(type(store).__name__)
            finally:
                _query_bounds.reset(token)
            close = getattr(store, "close", None)
            if close is not None:
                await close()
        with pytest.raises(TypeError, match="Unknown session read bound"):
            session_read_admitted(object(), session)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Unknown session read bound"):
            session_access_clause(object(), dialect=SQL_DIALECT)  # type: ignore[arg-type]
        return failures

    assert asyncio.run(scenario()) == ["InMemorySessionStore", "SQLiteSessionStore"]


def test_http_read_routes_apply_the_same_limit(session_store_case):
    async def scenario(store):
        app = await _world(store, compass_agents=(AGENT,))
        server = _controller_server(app)
        listed = await _http(server, "GET", "/fork-controller/sessions", params={"limit": 50})
        details = {
            sid: await _http(server, "GET", f"/fork-controller/sessions/{sid}")
            for sid in (*COMPASS_READABLE, *UNREADABLE)
        }
        session_events = {
            sid: await _http(server, "GET", f"/fork-controller/sessions/{sid}/events")
            for sid in (*COMPASS_READABLE, *UNREADABLE)
        }
        paged = await _http(
            server, "GET", f"/fork-controller/sessions/{HOST}/events", params={"limit": 1}
        )
        queried = await _http(
            server,
            "POST",
            "/fork-controller/events/query",
            json={"session_ids": [FOREIGN, OTHER_CONTROLLER_FORK, HOST], "limit": 1000},
        )
        # ``since`` omitted: profile decisions from before any window are readable.
        decisions = await _http(
            server,
            "POST",
            "/fork-controller/events/query",
            json={"event_type": "session.execution_profile.decided", "limit": 1000},
        )
        unforked = await _http(
            server,
            "POST",
            "/fork-controller/events/query",
            json={"exclude_forked_sessions": True, "limit": 1000},
        )
        refused = [
            await _http(server, "POST", "/fork-controller/events/query", json={field: value})
            for field, value in (
                ("causal_budget_id", "budget"),
                ("budget_limit_id", "limit"),
                ("workflow_name", "close"),
                ("include_content", True),
            )
        ]
        too_large = await _http(
            server, "POST", "/fork-controller/events/query", json={"max_bytes": 64}
        )
        too_large_session = await _http(
            server,
            "GET",
            f"/fork-controller/sessions/{HOST}/events",
            params={"max_bytes": 64},
        )
        window = {"start_at": WINDOW_START.isoformat(), "end_at": WINDOW_END.isoformat()}
        supervision = await _http(
            server,
            "POST",
            "/fork-controller/supervision-events/query",
            json={**window, "exclude_forked_sessions": True},
        )
        supervision_too_large = await _http(
            server,
            "POST",
            "/fork-controller/supervision-events/query",
            json={**window, "max_bytes": 1},
        )
        supervision_content = await _http(
            server,
            "POST",
            "/fork-controller/supervision-events/query",
            json={**window, "include_content": True},
        )
        return (
            listed,
            details,
            session_events,
            paged,
            queried,
            decisions,
            unforked,
            refused,
            (too_large, too_large_session, supervision_too_large),
            supervision,
            supervision_content,
        )

    (
        listed,
        details,
        session_events,
        paged,
        queried,
        decisions,
        unforked,
        refused,
        too_large,
        supervision,
        supervision_content,
    ) = _in_store(session_store_case, scenario)
    assert listed.status_code == 200
    assert {item["id"] for item in listed.json()["sessions"]} == COMPASS_READABLE
    assert listed.json()["total_count"] == len(COMPASS_READABLE)
    for sid in COMPASS_READABLE:
        assert details[sid].status_code == 200 and details[sid].json()["id"] == sid
        assert session_events[sid].status_code == 200
        assert session_events[sid].json()["session_id"] == sid
        assert session_events[sid].json()["events"]
    # Absent, foreign and other-controller sessions are indistinguishable.
    for sid in UNREADABLE:
        assert details[sid].status_code == 403
        assert details[sid].json() == {"detail": "Fork controller access denied."}
        assert session_events[sid].status_code == 403
        assert session_events[sid].json() == details[sid].json()
    page = paged.json()
    assert len(page["events"]) == 1 and page["has_more"] is True
    assert page["next_sequence"] == page["events"][0]["sequence"]
    assert queried.status_code == 200
    assert {event["session_id"] for event in queried.json()["events"]} == {HOST}
    assert decisions.status_code == 200
    assert {event["session_id"] for event in decisions.json()["events"]} <= COMPASS_READABLE
    assert {event["type"] for event in decisions.json()["events"]} == {
        "session.execution_profile.decided"
    }
    assert {event["session_id"] for event in unforked.json()["events"]} == {HOST}
    assert [response.status_code for response in refused] == [422] * 4
    assert [response.status_code for response in too_large] == [413] * 3
    assert supervision.status_code == 200
    assert [event["session_id"] for event in supervision.json()["events"]] == [HOST]
    assert supervision_content.status_code == 422


def test_a_malformed_snapshot_id_is_refused_like_a_foreign_one() -> None:
    async def scenario():
        return await _http(
            _controller_server(_app(InMemorySessionStore())),
            "GET",
            "/fork-controller/sources/cayu_event_bad:not-an-alias/snapshot",
        )

    malformed = asyncio.run(scenario())
    assert malformed.status_code == 403
    assert malformed.json() == {"detail": "Fork controller access denied."}


def test_session_event_reads_refuse_invalid_ranges_with_422(session_store_case):
    async def scenario(store):
        app = await _world(store, compass_agents=(AGENT,))
        server = _controller_server(app)
        responses = []
        for params in (
            {"after_sequence": 10, "before_sequence": 5},
            {"after_sequence": 2**64},
            {"before_sequence": 2**64},
        ):
            responses.append(
                await _http(
                    server, "GET", f"/fork-controller/sessions/{HOST}/events", params=params
                )
            )
        return responses

    for response in _in_store(session_store_case, scenario):
        assert response.status_code == 422, response.text
