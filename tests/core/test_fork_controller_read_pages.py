"""Fork-controller read pages: complete filters, commit visibility, byte limits, read cost."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_fork_controller_reads import (
    AGENT,
    COMPASS,
    COMPASS_SUBJECT,
    HOST,
    _app,
    _controller_server,
    _notes,
    _settled_approval,
)
from tests.core.test_session_fork_overlays import _http
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_supervision_support import (
    WINDOW_END,
    WINDOW_START,
    supervision_window,
)
from tests.runtime.runtime_tool_support import RequireApprovalPolicy, SideEffectTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import Event, EventType
from cayu.fork_controllers import ForkController
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.releases import ReleaseIdentity
from cayu.sessions import ForkSessionRequest
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest

__all__ = ["conformance_postgres_dsn", "session_store_case"]

RELEASE = "release-1"
WINDOW = {"start_at": WINDOW_START.isoformat(), "end_at": WINDOW_END.isoformat()}
IDENTITY = SessionIdentity(provider_name="fake", model="fake-model")


def _release_app(store) -> CayuApp:
    """One session's approval pause and continuation, recorded on one Release."""

    app = CayuApp(session_store=store, release_identity=ReleaseIdentity(id=RELEASE))
    app.register_provider(
        FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name=AGENT, model="fake-model"),
        tools=[SideEffectTool()],
        tool_policy=RequireApprovalPolicy(),
    )
    app.enroll_fork_controller(
        ForkController(name=COMPASS, subjects=(COMPASS_SUBJECT,), agents=(AGENT,))
    )
    return app


def test_http_supervision_reads_apply_every_query_field(session_store_case):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _release_app(store)
            await _settled_approval(app, store, HOST, AGENT)
            server = _controller_server(app)

            async def query(**fields):
                return await _http(
                    server,
                    "POST",
                    "/fork-controller/supervision-events/query",
                    json={**WINDOW, **fields},
                )

            compass = app.fork_controller_access(COMPASS_SUBJECT)
            filtered_first = await query(limit=1, release_id=RELEASE)
            return (
                await query(),
                await query(release_id="no-such-release"),
                await query(release_id=RELEASE),
                await compass.supervision_events(supervision_window(release_id="no-such-release")),
                await compass.supervision_events(supervision_window(release_id=RELEASE)),
                filtered_first,
                # A cursor issued for a release-filtered query belongs to that query only.
                await query(limit=1, cursor=filtered_first.json()["next_cursor"]),
            )
        finally:
            await _close_store(store)

    plain, other, matching, sdk_other, sdk_matching, first, reused = asyncio.run(scenario())
    assert plain.status_code == 200 and plain.json()["events"]
    assert {event["release_id"] for event in plain.json()["events"]} == {RELEASE}
    assert other.status_code == 200 and other.json()["events"] == []
    assert sdk_other.events == ()
    assert matching.json()["events"] == plain.json()["events"]
    assert [e.supervision_id for e in sdk_matching.events] == [
        e["supervision_id"] for e in plain.json()["events"]
    ]
    assert first.status_code == 200
    assert reused.status_code == 422
    assert reused.json() == {"detail": "Invalid supervision event cursor."}


def test_supervision_requests_map_onto_every_query_field() -> None:
    from cayu.server.contracts import SupervisionEventsRequest
    from cayu.sessions.supervision_events import SupervisionEventQuery

    request_fields = set(SupervisionEventsRequest.model_fields) - {"session_filter"}
    assert request_fields | {"sessions"} == set(SupervisionEventQuery.model_fields)


def _event(session_id: str, delta: str) -> Event:
    return Event(
        type=EventType.MODEL_TEXT_DELTA,
        session_id=session_id,
        agent_name=AGENT,
        payload={"delta": delta},
    )


async def _session_with_deltas(store, session_id: str, deltas: list[str]) -> None:
    await store.create(
        RunRequest(agent_name=AGENT, session_id=session_id, messages=[Message.text("user", "x")]),
        identity=IDENTITY,
    )
    if deltas:
        await store.append_events(session_id, [_event(session_id, delta) for delta in deltas])


def test_the_byte_limit_bounds_only_the_returned_page(session_store_case):
    """A page whose records fit is returned even when the next record would not fit with it."""

    cap = 100_000
    # Each 60 KB delta fits alone; any two together exceed the cap.
    sizes = ["a", "b", "c" * 60_000, "d" * 60_000]

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _session_with_deltas(store, HOST, sizes)
            server = _controller_server(app)
            records = await app.fork_controller_access(COMPASS_SUBJECT).events(HOST, limit=1000)
            # The page ends on the first large delta; the second one would not fit with it.
            limit = next(
                index
                for index, record in enumerate(records)
                if record.event.payload == {"delta": sizes[3]}
            )
            first = await _http(
                server,
                "GET",
                f"/fork-controller/sessions/{HOST}/events",
                params={"limit": limit, "max_bytes": cap},
            )
            walked, cursor, statuses = [], None, []
            while True:
                page = await _http(
                    server,
                    "POST",
                    "/fork-controller/events/query",
                    json={
                        "limit": 1,
                        "max_bytes": cap,
                        **({} if cursor is None else {"after_sequence": cursor}),
                    },
                )
                statuses.append(page.status_code)
                if page.status_code != 200:
                    break
                body = page.json()
                walked.extend(event["payload"].get("delta") for event in body["events"])
                cursor = body["next_sequence"]
                if not body["has_more"]:
                    break
            return limit, first, walked, statuses
        finally:
            await _close_store(store)

    limit, first, walked, statuses = asyncio.run(scenario())
    assert first.status_code == 200
    body = first.json()
    assert len(body["events"]) == limit
    assert body["events"][-1]["payload"]["delta"] == sizes[2]
    assert body["has_more"] is True
    # Paging one record at a time never wedges on two records that only fit apart.
    assert set(statuses) == {200}
    assert [delta for delta in walked if delta is not None] == sizes


def test_cross_session_controller_reads_ascend_from_zero(session_store_case):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _session_with_deltas(store, HOST, ["a", "b"])
            await _session_with_deltas(store, "second-session", ["c"])
            compass = app.fork_controller_access(COMPASS_SUBJECT)
            server = _controller_server(app)
            refused = []
            for query in (
                EventQuery(order_by=EventOrder.SEQUENCE_DESC),
                EventQuery(session_ids=(HOST, "second-session"), order_by=EventOrder.SEQUENCE_DESC),
            ):
                with pytest.raises(ValueError, match="ascending"):
                    await compass.query_events(query)
                refused.append(True)
            http_refused = await _http(
                server, "POST", "/fork-controller/events/query", json={"order_by": "sequence_desc"}
            )
            single_desc = await compass.query_events(
                EventQuery(session_id=HOST, order_by=EventOrder.SEQUENCE_DESC)
            )
            http_single_desc = await _http(
                server,
                "GET",
                f"/fork-controller/sessions/{HOST}/events",
                params={"order_by": "sequence_desc"},
            )
            return refused, http_refused, single_desc, http_single_desc
        finally:
            await _close_store(store)

    refused, http_refused, single_desc, http_single_desc = asyncio.run(scenario())
    assert refused == [True, True]
    assert http_refused.status_code == 422
    sequences = [record.sequence for record in single_desc]
    assert sequences == sorted(sequences, reverse=True) and sequences
    assert http_single_desc.status_code == 200


def test_postgres_first_controller_page_never_passes_an_uncommitted_lower_sequence(
    tmp_path, conformance_postgres_dsn
):
    import psycopg

    case = ("postgres", tmp_path, conformance_postgres_dsn)

    async def scenario():
        store = await _open_store(case)
        try:
            app = _app(store)
            compass = app.fork_controller_access(COMPASS_SUBJECT)
            await _session_with_deltas(store, "early-session", ["first"])
            await _session_with_deltas(store, "late-session", [])
            async with await psycopg.AsyncConnection.connect(conformance_postgres_dsn) as writer:
                async with writer.cursor() as cursor:
                    await cursor.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = current_schema() AND table_name = 'cayu_events' "
                        "AND column_name NOT IN ('sequence', 'insert_xid')"
                    )
                    columns = [row[0] for row in await cursor.fetchall()]
                    copied = ", ".join(
                        {
                            "event_id": "'uncommitted-event'",
                            "session_order": "session_order + 1000",
                            "payload": "jsonb_set(payload, '{delta}', '\"uncommitted\"')",
                            "event": (
                                "jsonb_set(jsonb_set(event, '{id}', '\"uncommitted-event\"'), "
                                "'{payload,delta}', '\"uncommitted\"')"
                            ),
                        }.get(column, column)
                        for column in columns
                    )
                    # An open transaction holds a lower sequence while a later one commits.
                    await cursor.execute(
                        f"INSERT INTO cayu_events ({', '.join(columns)}) "
                        f"SELECT {copied} FROM cayu_events "
                        "WHERE session_id = 'early-session' AND event_type = 'model.text.delta'"
                    )
                    await store.append_event("late-session", _event("late-session", "third"))
                    first_page = await compass.query_events(
                        EventQuery(event_type=EventType.MODEL_TEXT_DELTA)
                    )
                await writer.commit()
            next_page = await compass.query_events(
                EventQuery(
                    event_type=EventType.MODEL_TEXT_DELTA,
                    after_sequence=first_page[-1].sequence,
                )
            )
            return first_page, next_page
        finally:
            await _close_store(store)

    first_page, next_page = asyncio.run(scenario())
    assert [record.event.payload["delta"] for record in first_page] == ["first"]
    assert [record.event.payload["delta"] for record in next_page] == ["uncommitted", "third"]


class _CountingIndex(dict):
    """The memory store's per-type index, counting fork-marker lookups."""

    forked_lookups = 0

    def get(self, key, default=None):
        if key == str(EventType.SESSION_FORKED):
            type(self).forked_lookups += 1
        return super().get(key, default)


def test_memory_exclude_forked_reads_index_fork_markers_once() -> None:
    from tests.core.session_store_conformance.support import _public_authority_alias_codec

    from cayu.sessions.memory import InMemorySessionStore

    async def scenario():
        store = InMemorySessionStore(public_authority_alias_codec=_public_authority_alias_codec())
        app = _app(store)
        await _settled_approval(app, store, HOST, AGENT)
        compass = app.fork_controller_access(COMPASS_SUBJECT)
        for index in range(3):
            async for _event in compass.fork(
                ForkSessionRequest(
                    source_session_id=HOST, session_id=f"fork-{index}", overlay=_notes(COMPASS)
                )
            ):
                pass
        store._type_event_records = _CountingIndex(store._type_event_records)
        records = await compass.query_events(EventQuery(limit=5000), exclude_forked_sessions=True)
        return records, _CountingIndex.forked_lookups

    records, lookups = asyncio.run(scenario())
    assert len(records) > 1 and {record.event.session_id for record in records} == {HOST}
    # One lookup per read, not one per admitted record.
    assert lookups == 1
