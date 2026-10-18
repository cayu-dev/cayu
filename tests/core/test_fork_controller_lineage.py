"""Sessions created under a fork stay in its lineage, on every built-in session store.

A controller fork's overlay record names its owner, but a session created under
the fork (a subagent, a delegated run, a grandchild) carries no record. The
runtime stamps a lineage marker on every fork and on every session created
under one, so another controller can neither read nor copy that work, and
``exclude_forked_sessions`` keeps a supervisor's investigations out of the
supervised agent's ledger.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _new_postgres_store,
    _open_store,
    _public_authority_alias_codec,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_fork_controller_reads import (
    AGENT,
    AUDITOR,
    AUDITOR_SUBJECT,
    COMPASS,
    COMPASS_SUBJECT,
    FOREIGN,
    HOST,
    OTHER_CONTROLLER_FORK,
    OWN_BODY_FORK,
    OWN_NOTES_FORK,
    _app,
    _controller_server,
    _notes,
    _world,
)
from tests.core.test_session_fork_overlays import _http
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import (
    _pending_tool_approval_from_public_event,
    collect_events,
    collect_tool_approval_events,
)
from tests.runtime.runtime_supervision_support import (
    WINDOW_END,
    WINDOW_START,
    resolve_approval,
    supervision_window,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ToolApprovalDecision
from cayu.events import EventType
from cayu.fork_controllers import ForkController, ForkControllerDenied
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions import ForkSessionRequest, ResumeRequest
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.queries import SessionQuery
from cayu.sessions.records import Session, SessionIdentity
from cayu.sessions.requests import RunRequest
from cayu.storage import SQLiteSessionStore
from cayu.tools.subagents import SubagentSpec, SubagentTool

__all__ = ["conformance_postgres_dsn", "session_store_case"]

# The durable metadata key is part of the stored contract, so it is spelled out.
FORK_LINEAGE_METADATA_KEY = "cayu:fork_lineage"

HOST_FORK = "host-fork"
AUDITOR_CHILD = "auditor-fork-child"
AUDITOR_GRANDCHILD = "auditor-fork-grandchild"
OWN_CHILD = "compass-fork-child"
OWN_GRANDCHILD = "compass-fork-grandchild"
HOST_CHILD = "host-child"
HOST_FORK_CHILD = "host-fork-child"
# Created in order, each under the parent named beside it.
CHILDREN = (
    (AUDITOR_CHILD, OTHER_CONTROLLER_FORK),
    (AUDITOR_GRANDCHILD, AUDITOR_CHILD),
    (OWN_CHILD, OWN_NOTES_FORK),
    (OWN_GRANDCHILD, OWN_CHILD),
    (HOST_CHILD, HOST),
    (HOST_FORK_CHILD, HOST_FORK),
)
AUDITOR_LINEAGE = {OTHER_CONTROLLER_FORK, AUDITOR_CHILD, AUDITOR_GRANDCHILD}
COMPASS_LINEAGE = {OWN_BODY_FORK, OWN_NOTES_FORK, OWN_CHILD, OWN_GRANDCHILD}
HOST_LINEAGE = {HOST_FORK, HOST_FORK_CHILD}
UNFORKED = {HOST, FOREIGN, HOST_CHILD}
COMPASS_READABLE = UNFORKED | COMPASS_LINEAGE | HOST_LINEAGE
# The auditor is enrolled for the host agent only, so the payroll session is foreign to it.
AUDITOR_READABLE = (UNFORKED - {FOREIGN}) | AUDITOR_LINEAGE | HOST_LINEAGE
EXPECTED_MARKERS = {
    **{sid: {"controller": AUDITOR, "forked": True} for sid in AUDITOR_LINEAGE},
    **{sid: {"controller": COMPASS, "forked": True} for sid in COMPASS_LINEAGE},
    **{sid: {"controller": None, "forked": True} for sid in HOST_LINEAGE},
    **dict.fromkeys(UNFORKED),
    HOST_CHILD: {"controller": None, "forked": False},
}
WINDOW = {"start_at": WINDOW_START.isoformat(), "end_at": WINDOW_END.isoformat()}


async def _loaded(store, session_id: str) -> Session:
    session = await store.load(session_id)
    assert session is not None, session_id
    return session


async def _marker(store, session_id: str):
    return (await _loaded(store, session_id)).metadata.get(FORK_LINEAGE_METADATA_KEY)


async def _settled_child(app: CayuApp, store, session_id: str, parent: str) -> None:
    """Run one session under ``parent`` to an approval and deny it."""

    events = await collect_events(
        app,
        RunRequest(
            agent_name=AGENT,
            session_id=session_id,
            parent_session_id=parent,
            messages=[Message.text("user", "use the tool")],
        ),
    )
    requested = next(e for e in events if e.type == EventType.TOOL_CALL_APPROVAL_REQUESTED)
    pending = await _pending_tool_approval_from_public_event(store, requested)
    await collect_tool_approval_events(
        app, resolve_approval(session_id, pending, ToolApprovalDecision.DENY)
    )


async def _lineage_world(store, *, extra_sessions: int = 0) -> CayuApp:
    app = await _world(store, sessions=2 + len(CHILDREN) + extra_sessions)
    async for _event in app.fork_session(
        ForkSessionRequest(source_session_id=HOST, session_id=HOST_FORK)
    ):
        pass
    for child, parent in CHILDREN:
        await _settled_child(app, store, child, parent)
    return app


async def _denied(operation) -> bool:
    try:
        await operation
    except ForkControllerDenied:
        return True
    return False


async def _stream_denied(stream) -> bool:
    try:
        async for _event in stream:
            pass
    except ForkControllerDenied:
        return True
    return False


async def _observe(app: CayuApp, store) -> dict:
    """Everything one controller can see or do, through the SDK and over HTTP."""

    compass = app.fork_controller_access(COMPASS_SUBJECT)
    auditor = app.fork_controller_access(AUDITOR_SUBJECT)
    server = _controller_server(app)
    seen: dict = {
        "markers": {sid: await _marker(store, sid) for sid in EXPECTED_MARKERS},
        "listed": {s.id for s in (await compass.sessions(SessionQuery(limit=100))).sessions},
        "auditor_listed": {
            s.id for s in (await auditor.sessions(SessionQuery(limit=100))).sessions
        },
        "detail_denied": {sid: await _denied(compass.session(sid)) for sid in AUDITOR_LINEAGE},
        "events_denied": {sid: await _denied(compass.events(sid)) for sid in AUDITOR_LINEAGE},
        "own_child_detail": (await compass.session(OWN_GRANDCHILD)).id,
        "events": {r.event.session_id for r in await compass.query_events(EventQuery(limit=5000))},
        "named_events": await compass.query_events(
            EventQuery(session_ids=(AUDITOR_CHILD, AUDITOR_GRANDCHILD))
        ),
        "unforked_events": {
            r.event.session_id
            for r in await compass.query_events(
                EventQuery(limit=5000), exclude_forked_sessions=True
            )
        },
        "auditor_unforked_events": {
            r.event.session_id
            for r in await auditor.query_events(
                EventQuery(limit=5000), exclude_forked_sessions=True
            )
        },
        "supervision": {
            (e.session_id, e.forked)
            for e in (await compass.supervision_events(supervision_window(limit=100))).events
        },
        "unforked_supervision": {
            e.session_id
            for e in (
                await compass.supervision_events(
                    supervision_window(limit=100, exclude_forked_sessions=True)
                )
            ).events
        },
        "snapshot_denied": {
            sid: await _denied(compass.snapshot_source(sid)) for sid in AUDITOR_LINEAGE
        },
        "own_snapshots": [
            (await compass.snapshot_source(sid)).source_session_id
            for sid in (OWN_GRANDCHILD, HOST_FORK_CHILD, HOST_CHILD)
        ],
        "fork_denied": {
            sid: await _stream_denied(
                compass.fork(
                    ForkSessionRequest(
                        source_session_id=sid, session_id=f"copy-of-{sid}", overlay=_notes(COMPASS)
                    )
                )
            )
            for sid in (AUDITOR_CHILD, AUDITOR_GRANDCHILD)
        },
        "host_fork_denied": await _stream_denied(
            app.fork_session(
                ForkSessionRequest(source_session_id=AUDITOR_CHILD, session_id="host-copy")
            )
        ),
        "copies": [
            await store.load(sid)
            for sid in ("copy-of-" + AUDITOR_CHILD, "copy-of-" + AUDITOR_GRANDCHILD, "host-copy")
        ],
    }
    # The control plane's supervision reads share the same lineage meaning.
    seen["control_plane_supervision"] = {
        (e.session_id, e.forked)
        for e in (await app.query_supervision_events(supervision_window(limit=100))).events
    }
    seen["control_plane_unforked_supervision"] = {
        e.session_id
        for e in (
            await app.query_supervision_events(
                supervision_window(limit=100, exclude_forked_sessions=True)
            )
        ).events
    }
    listed = await _http(server, "GET", "/fork-controller/sessions", params={"limit": 100})
    seen["http_listed"] = {item["id"] for item in listed.json()["sessions"]}
    seen["http_detail"] = {
        sid: (await _http(server, "GET", f"/fork-controller/sessions/{sid}")).status_code
        for sid in (*AUDITOR_LINEAGE, OWN_CHILD, OWN_GRANDCHILD, HOST_CHILD, HOST_FORK_CHILD)
    }
    seen["http_events"] = {
        sid: (await _http(server, "GET", f"/fork-controller/sessions/{sid}/events")).status_code
        for sid in (AUDITOR_CHILD, AUDITOR_GRANDCHILD, OWN_GRANDCHILD)
    }
    seen["http_snapshot"] = {
        sid: (await _http(server, "GET", f"/fork-controller/sources/{sid}/snapshot")).status_code
        for sid in (AUDITOR_CHILD, AUDITOR_GRANDCHILD)
    }
    queried = await _http(
        server,
        "POST",
        "/fork-controller/events/query",
        json={"session_ids": [AUDITOR_CHILD, AUDITOR_GRANDCHILD, HOST_CHILD], "limit": 1000},
    )
    seen["http_named_events"] = {event["session_id"] for event in queried.json()["events"]}
    unforked = await _http(
        server,
        "POST",
        "/fork-controller/events/query",
        json={"exclude_forked_sessions": True, "limit": 1000},
    )
    seen["http_unforked_events"] = {event["session_id"] for event in unforked.json()["events"]}
    supervision = await _http(
        server, "POST", "/fork-controller/supervision-events/query", json={**WINDOW, "limit": 100}
    )
    seen["http_supervision"] = {
        (event["session_id"], event["forked"]) for event in supervision.json()["events"]
    }
    unforked_supervision = await _http(
        server,
        "POST",
        "/fork-controller/supervision-events/query",
        json={**WINDOW, "limit": 100, "exclude_forked_sessions": True},
    )
    seen["http_unforked_supervision"] = {
        event["session_id"] for event in unforked_supervision.json()["events"]
    }
    return seen


def _assert_lineage_contract(seen: dict, *, markers: dict = EXPECTED_MARKERS) -> None:
    # Another controller's fork and everything created under it is invisible.
    assert seen["listed"] == COMPASS_READABLE
    assert seen["auditor_listed"] == AUDITOR_READABLE
    assert all(seen["detail_denied"].values()), seen["detail_denied"]
    assert all(seen["events_denied"].values()), seen["events_denied"]
    assert seen["own_child_detail"] == OWN_GRANDCHILD
    assert seen["events"] == COMPASS_READABLE
    assert seen["named_events"] == []
    # Excluding forks drops every session in a forked lineage, own children included.
    assert seen["unforked_events"] == UNFORKED
    assert seen["auditor_unforked_events"] == UNFORKED - {FOREIGN}
    touched = {sid for sid, _forked in seen["supervision"]}
    assert AUDITOR_LINEAGE.isdisjoint(touched)
    assert {OWN_CHILD, OWN_GRANDCHILD, HOST_CHILD, HOST_FORK_CHILD} <= touched
    assert all(forked == (sid not in UNFORKED) for sid, forked in seen["supervision"])
    assert seen["unforked_supervision"] == UNFORKED
    # Another controller's lineage can be neither snapshotted nor forked, by a
    # controller or by the host; the controller's own and host lineages can.
    assert all(seen["snapshot_denied"].values()), seen["snapshot_denied"]
    assert seen["own_snapshots"] == [OWN_GRANDCHILD, HOST_FORK_CHILD, HOST_CHILD]
    assert all(seen["fork_denied"].values()), seen["fork_denied"]
    assert seen["host_fork_denied"] is True
    assert seen["copies"] == [None, None, None]
    assert seen["http_listed"] == COMPASS_READABLE
    assert seen["http_detail"] == {
        **dict.fromkeys(AUDITOR_LINEAGE, 403),
        **dict.fromkeys((OWN_CHILD, OWN_GRANDCHILD, HOST_CHILD, HOST_FORK_CHILD), 200),
    }
    assert seen["http_events"] == {AUDITOR_CHILD: 403, AUDITOR_GRANDCHILD: 403, OWN_GRANDCHILD: 200}
    assert seen["http_snapshot"] == {AUDITOR_CHILD: 403, AUDITOR_GRANDCHILD: 403}
    assert seen["http_named_events"] == {HOST_CHILD}
    assert seen["http_unforked_events"] == UNFORKED
    assert seen["http_supervision"] == seen["supervision"]
    assert seen["http_unforked_supervision"] == UNFORKED
    assert AUDITOR_LINEAGE & {sid for sid, _forked in seen["control_plane_supervision"]}
    assert all(forked == (sid not in UNFORKED) for sid, forked in seen["control_plane_supervision"])
    assert seen["control_plane_unforked_supervision"] == UNFORKED
    assert seen["markers"] == markers


def test_fork_lineage_bounds_reads_ownership_and_exclusion(session_store_case):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = await _lineage_world(store)
            return await _observe(app, store)
        finally:
            await _close_store(store)

    _assert_lineage_contract(asyncio.run(scenario()))


async def _strip_lineage_markers(store, case) -> None:
    """Store every session as the release before lineage markers wrote it.

    That release already wrote overlay records, ``session.forked`` events,
    fork invocation provenance and parent links, but no lineage marker.
    """

    for session in (await store.list_sessions(SessionQuery(limit=100))).sessions:
        if FORK_LINEAGE_METADATA_KEY in session.metadata:
            metadata = dict(session.metadata)
            del metadata[FORK_LINEAGE_METADATA_KEY]
            await _write_metadata(store, case, session.id, metadata)


# Sessions created after the upgrade under sessions written before it.
AFTER_UPGRADE_AUDITOR_CHILD = "after-upgrade-auditor-child"
AFTER_UPGRADE_HOST_FORK_CHILD = "after-upgrade-host-fork-child"
AFTER_UPGRADE_HOST_FORK = "after-upgrade-host-fork"


def test_sessions_written_before_lineage_markers_keep_their_lineage(session_store_case):
    """An upgraded store settles unmarked forks and descendants through their parents."""

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = await _lineage_world(store, extra_sessions=2)
            await _strip_lineage_markers(store, session_store_case)
            seen = await _observe(app, store)
            await _settled_child(app, store, AFTER_UPGRADE_AUDITOR_CHILD, AUDITOR_GRANDCHILD)
            await _settled_child(app, store, AFTER_UPGRADE_HOST_FORK_CHILD, HOST_FORK_CHILD)
            async for _event in app.fork_session(
                ForkSessionRequest(
                    source_session_id=HOST_FORK_CHILD, session_id=AFTER_UPGRADE_HOST_FORK
                )
            ):
                pass
            compass = app.fork_controller_access(COMPASS_SUBJECT)
            auditor = app.fork_controller_access(AUDITOR_SUBJECT)
            after = {
                "compass_listed": {
                    s.id for s in (await compass.sessions(SessionQuery(limit=100))).sessions
                },
                "auditor_listed": {
                    s.id for s in (await auditor.sessions(SessionQuery(limit=100))).sessions
                },
                "compass_denied": await _denied(compass.session(AFTER_UPGRADE_AUDITOR_CHILD)),
                "auditor_child": (await auditor.session(AFTER_UPGRADE_AUDITOR_CHILD)).id,
                "unforked": {
                    r.event.session_id
                    for r in await compass.query_events(
                        EventQuery(limit=5000), exclude_forked_sessions=True
                    )
                },
                "unforked_supervision": {
                    e.session_id
                    for e in (
                        await app.query_supervision_events(
                            supervision_window(limit=100, exclude_forked_sessions=True)
                        )
                    ).events
                },
                "markers": {
                    sid: await _marker(store, sid)
                    for sid in (
                        AFTER_UPGRADE_AUDITOR_CHILD,
                        AFTER_UPGRADE_HOST_FORK_CHILD,
                        AFTER_UPGRADE_HOST_FORK,
                    )
                },
            }
            # Deleting a legacy fork cuts its unmarked descendants off from the
            # lineage that owned them, so no controller can read them any more.
            await store.delete_session(OTHER_CONTROLLER_FORK)
            orphaned = {
                "compass_denied": await _denied(compass.session(AUDITOR_CHILD)),
                "auditor_denied": await _denied(auditor.session(AUDITOR_CHILD)),
                "auditor_descendants_denied": [
                    await _denied(auditor.session(sid))
                    for sid in (AUDITOR_GRANDCHILD, AFTER_UPGRADE_AUDITOR_CHILD)
                ],
                "parent": (await _loaded(store, AUDITOR_GRANDCHILD)).parent_session_id,
                "unforked": {
                    r.event.session_id
                    for r in await auditor.query_events(
                        EventQuery(limit=5000), exclude_forked_sessions=True
                    )
                },
            }
            return seen, after, orphaned
        finally:
            await _close_store(store)

    seen, after, orphaned = asyncio.run(scenario())
    _assert_lineage_contract(seen, markers=dict.fromkeys(EXPECTED_MARKERS))
    host_lineage = HOST_LINEAGE | {AFTER_UPGRADE_HOST_FORK_CHILD, AFTER_UPGRADE_HOST_FORK}
    assert after["compass_listed"] == COMPASS_READABLE | host_lineage
    assert after["auditor_listed"] == (
        AUDITOR_READABLE | host_lineage | {AFTER_UPGRADE_AUDITOR_CHILD}
    )
    assert after["compass_denied"]
    assert after["auditor_child"] == AFTER_UPGRADE_AUDITOR_CHILD
    assert after["unforked"] == UNFORKED
    assert after["unforked_supervision"] == UNFORKED
    # A child takes its marker from its parent, so one under an unmarked parent
    # is settled the same way; a new fork records the owner it resolved.
    assert after["markers"] == {
        AFTER_UPGRADE_AUDITOR_CHILD: None,
        AFTER_UPGRADE_HOST_FORK_CHILD: None,
        AFTER_UPGRADE_HOST_FORK: {"controller": None, "forked": True},
    }
    assert orphaned["compass_denied"] and orphaned["auditor_denied"]
    # The grandchild still reaches its parent, which no longer settles anything.
    assert orphaned["parent"] == AUDITOR_CHILD
    assert orphaned["auditor_descendants_denied"] == [True, True]
    assert orphaned["unforked"] == UNFORKED - {FOREIGN}


def test_fork_lineage_survives_parent_deletion_and_a_restart(session_store_case):
    store_kind, tmp_path, postgres_dsn = session_store_case

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            await _lineage_world(store)
            # The auditor's fork goes away; the work created under it stays its own.
            await store.delete_session(OTHER_CONTROLLER_FORK)
        finally:
            if store_kind != "memory":
                await _close_store(store)
        codec = _public_authority_alias_codec()
        if store_kind == "sqlite":
            store = SQLiteSessionStore(
                tmp_path / "sessions.sqlite", public_authority_alias_codec=codec
            )
        elif store_kind == "postgres":
            store = _new_postgres_store(postgres_dsn, public_authority_alias_codec=codec)
        try:
            restarted = _app(store)
            compass = restarted.fork_controller_access(COMPASS_SUBJECT)
            return (
                {s.id for s in (await compass.sessions(SessionQuery(limit=100))).sessions},
                {
                    r.event.session_id
                    for r in await compass.query_events(
                        EventQuery(limit=5000), exclude_forked_sessions=True
                    )
                },
                await _denied(compass.session(AUDITOR_CHILD)),
                await _denied(compass.snapshot_source(AUDITOR_CHILD)),
                await store.load(AUDITOR_CHILD),
                {
                    sid: await _marker(store, sid)
                    for sid in EXPECTED_MARKERS
                    if sid != OTHER_CONTROLLER_FORK
                },
            )
        finally:
            await _close_store(store)

    listed, unforked, child_denied, snapshot_denied, child, markers = asyncio.run(scenario())
    assert listed == COMPASS_READABLE
    assert unforked == UNFORKED
    assert child_denied and snapshot_denied
    assert child.parent_session_id is None
    assert markers == {
        sid: marker for sid, marker in EXPECTED_MARKERS.items() if sid != OTHER_CONTROLLER_FORK
    }


@pytest.mark.store_semantics
@pytest.mark.parametrize("removal", ["delete", "detach"])
def test_unforked_child_stays_readable_after_parent_removal(session_store_case, removal):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = await _world(store, sessions=3)
            await _settled_child(app, store, HOST_CHILD, HOST)
            if removal == "delete":
                await store.delete_session(HOST)
            else:
                await store.detach_session_children(
                    HOST,
                    (HOST_CHILD,),
                    closure_receipt={
                        "root_session_id": HOST,
                        "plan_id": "a" * 64,
                        "operation": "detach",
                    },
                )
            child = await _loaded(store, HOST_CHILD)
            assert child.parent_session_id is None
            compass = app.fork_controller_access(COMPASS_SUBJECT)
            assert (await compass.session(HOST_CHILD)).id == HOST_CHILD
            assert HOST_CHILD in {
                s.id for s in (await compass.sessions(SessionQuery(limit=100))).sessions
            }
            assert HOST_CHILD in {
                r.event.session_id
                for r in await compass.query_events(
                    EventQuery(limit=5000), exclude_forked_sessions=True
                )
            }
            supervision = await app.query_supervision_events(supervision_window(limit=100))
            assert (HOST_CHILD, False) in {(e.session_id, e.forked) for e in supervision.events}
            unforked = await app.query_supervision_events(
                supervision_window(limit=100, exclude_forked_sessions=True)
            )
            assert HOST_CHILD in {e.session_id for e in unforked.events}
        finally:
            await _close_store(store)

    asyncio.run(scenario())


def test_a_subagent_of_a_controller_fork_joins_its_lineage(session_store_case):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            provider = FakeProvider(
                [
                    [
                        ModelStreamEvent.text_delta("host done"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                    [
                        ModelStreamEvent.tool_call(
                            id="call_investigate",
                            name="subagent",
                            arguments={"agent": "helper", "task": "investigate the denial"},
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [
                        ModelStreamEvent.text_delta("investigated"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                    [
                        ModelStreamEvent.text_delta("fork done"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                ]
            )
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name=AGENT, model="fake-model"),
                tools=[
                    SubagentTool(
                        app,
                        agents={
                            "helper": SubagentSpec(agent_name="helper", description="Investigate.")
                        },
                    )
                ],
            )
            app.register_agent(AgentSpec(name="helper", model="fake-model"))
            for name, subject in ((AUDITOR, AUDITOR_SUBJECT), (COMPASS, COMPASS_SUBJECT)):
                app.enroll_fork_controller(
                    ForkController(
                        name=name,
                        subjects=(subject,),
                        agents=(AGENT, "helper"),
                        host_tools=("subagent",),
                        allow_model_runs=True,
                    )
                )
            await collect_events(
                app,
                RunRequest(
                    agent_name=AGENT, session_id=HOST, messages=[Message.text("user", "go")]
                ),
            )
            auditor = app.fork_controller_access(AUDITOR_SUBJECT)
            async for _event in auditor.fork(
                ForkSessionRequest(
                    source_session_id=HOST,
                    session_id=OTHER_CONTROLLER_FORK,
                    overlay=_notes(AUDITOR),
                )
            ):
                pass
            async for _event in auditor.resume(
                ResumeRequest(
                    session_id=OTHER_CONTROLLER_FORK,
                    messages=[Message.text("user", "investigate")],
                )
            ):
                pass
            [child] = (
                await store.list_sessions(SessionQuery(parent_session_id=OTHER_CONTROLLER_FORK))
            ).sessions
            compass = app.fork_controller_access(COMPASS_SUBJECT)
            return (
                child,
                await _denied(compass.session(child.id)),
                {s.id for s in (await compass.sessions()).sessions},
                (await auditor.session(child.id)).id,
                {
                    r.event.session_id
                    for r in await auditor.query_events(
                        EventQuery(limit=5000), exclude_forked_sessions=True
                    )
                },
            )
        finally:
            await _close_store(store)

    child, compass_denied, compass_listed, auditor_read, auditor_unforked = asyncio.run(scenario())
    assert child.agent_name == "helper"
    assert compass_denied
    assert compass_listed == {HOST}
    assert child.metadata.get(FORK_LINEAGE_METADATA_KEY) == {"controller": AUDITOR, "forked": True}
    assert auditor_read == child.id
    assert auditor_unforked == {HOST}


def test_callers_cannot_forge_or_strip_a_fork_lineage_marker() -> None:
    from cayu.server import ServerConfig, create_server
    from cayu.sessions.creation import session_metadata_for_creation
    from cayu.tools.subagents import _subagent_arguments

    forged = {FORK_LINEAGE_METADATA_KEY: {"controller": COMPASS, "forked": True}}

    async def scenario():
        store = InMemorySessionStore(public_authority_alias_codec=_public_authority_alias_codec())
        app = await _lineage_world(store)
        refusals = []
        for build in (
            lambda: RunRequest(
                agent_name=AGENT, messages=[Message.text("user", "x")], metadata=forged
            ),
            lambda: ForkSessionRequest(source_session_id=HOST, metadata=forged),
            lambda: session_metadata_for_creation(
                forged, identity=SessionIdentity(provider_name="fake", model="fake-model")
            ),
        ):
            with pytest.raises(ValueError, match="fork-lineage authority"):
                build()
            refusals.append(True)
        # A model cannot seed the marker through subagent arguments.
        _alias, _task, metadata = _subagent_arguments(
            {"agent": "helper", "task": "x", "metadata": {**forged, "note": "kept"}}
        )
        # Callers can neither replace nor strip the marker of an existing session.
        with pytest.raises(ValueError, match="runtime-owned"):
            await store.update_metadata(HOST, forged)
        await store.update_metadata(AUDITOR_CHILD, {"note": "edited"})
        server = create_server(app, config=ServerConfig.local_development())
        patched = await _http(
            server, "PATCH", f"/api/sessions/{HOST}/metadata", json={"metadata": forged}
        )
        ran = await _http(
            server,
            "POST",
            "/api/run",
            json={
                "prompt": "x",
                "agent": AGENT,
                "metadata": {FORK_LINEAGE_METADATA_KEY: {"controller": None, "forked": True}},
            },
        )
        compass = app.fork_controller_access(COMPASS_SUBJECT)
        return (
            refusals,
            metadata,
            (await _loaded(store, AUDITOR_CHILD)).metadata,
            await _marker(store, HOST),
            patched.status_code,
            ran.status_code,
            {
                session.id: session.metadata.get(FORK_LINEAGE_METADATA_KEY)
                for session in (await store.list_sessions(SessionQuery(limit=100))).sessions
                if session.id not in EXPECTED_MARKERS
            },
            await _denied(compass.session(AUDITOR_CHILD)),
        )

    (refusals, metadata, edited, host_marker, patched, ran, run_markers, still_denied) = (
        asyncio.run(scenario())
    )
    assert refusals == [True, True, True]
    assert metadata == {"note": "kept"}
    assert edited[FORK_LINEAGE_METADATA_KEY] == {"controller": AUDITOR, "forked": True}
    assert edited["note"] == "edited"
    assert host_marker is None
    assert patched == 422
    # The HTTP run body has no metadata field, so the forged marker never reaches the run.
    assert ran == 200
    assert list(run_markers.values()) == [None]
    assert still_denied


MALFORMED_MARKERS = (
    "auditor",
    [AUDITOR],
    {"forked": True},
    {"controller": None},
    {"controller": None, "forked": 1},
    {"controller": None, "forked": "true"},
    {"controller": None, "forked": 0},
    {"controller": None, "forked": "false"},
    {"controller": COMPASS, "forked": False},
    {"controller": 7, "forked": True},
    {"controller": AUDITOR, "forked": True},
)
ADMITTED_MARKERS = (
    {"controller": None, "forked": True},
    {"controller": COMPASS, "forked": True},
    {"controller": None, "forked": False},
)


async def _write_marker(store, case, session_id: str, marker) -> None:
    """Rewrite one durable marker below every validator, as storage corruption would."""

    session = await store.load(session_id)
    await _write_metadata(
        store, case, session_id, {**session.metadata, FORK_LINEAGE_METADATA_KEY: marker}
    )


async def _write_metadata(store, case, session_id: str, metadata: dict) -> None:
    """Rewrite one session's durable metadata below every validator."""

    store_kind, tmp_path, postgres_dsn = case
    session = await store.load(session_id)
    if store_kind == "memory":
        store._sessions[session_id] = session.model_copy(update={"metadata": metadata})
    elif store_kind == "sqlite":
        import sqlite3

        with sqlite3.connect(tmp_path / "sessions.sqlite") as connection:
            connection.execute(
                "UPDATE cayu_sessions SET metadata_json = ? WHERE id = ?",
                (json.dumps(metadata), session_id),
            )
    else:
        import psycopg

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            await connection.execute(
                "UPDATE cayu_sessions SET metadata = %s::jsonb WHERE id = %s",
                (json.dumps(metadata), session_id),
            )


@pytest.mark.store_semantics
def test_a_malformed_lineage_marker_is_unreadable_in_every_store(session_store_case):
    markers = (*MALFORMED_MARKERS, *ADMITTED_MARKERS)
    session_ids = [f"marked-{index}" for index in range(len(markers))]
    identity = SessionIdentity(provider_name="fake", model="fake-model")

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            for session_id, marker in zip(session_ids, markers, strict=True):
                await store.create(
                    RunRequest(
                        agent_name=AGENT,
                        session_id=session_id,
                        messages=[Message.text("user", "x")],
                    ),
                    identity=identity,
                )
                await _write_marker(store, session_store_case, session_id, marker)
            compass = app.fork_controller_access(COMPASS_SUBJECT)
            return (
                {s.id for s in (await compass.sessions(SessionQuery(limit=100))).sessions},
                {sid for sid in session_ids if not await _denied(compass.session(sid))},
            )
        finally:
            await _close_store(store)

    listed, loaded = asyncio.run(scenario())
    # By position: ``{"forked": 1} == {"forked": True}`` in Python, never in JSON.
    expected = set(session_ids[len(MALFORMED_MARKERS) :])
    # The store's SQL (or memory predicate) and the Python predicate agree.
    assert listed == loaded == expected
    from cayu.sessions._fork_controller_bounds import lineage_marker_admits

    assert {
        sid
        for sid, marker in zip(session_ids, markers, strict=True)
        if lineage_marker_admits({FORK_LINEAGE_METADATA_KEY: marker}, COMPASS)
    } == expected
