"""Public entrance regressions for scoped durable steering (no provider calls)."""

from __future__ import annotations

import asyncio
import base64
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import SecretStr

from cayu import (
    CayuApp,
    EnqueueSessionMessageRequest,
    InMemorySessionStore,
    Message,
    ResolutionActor,
    ResolutionActorSource,
    RunRequest,
    SessionIdentity,
    SessionMessageAccessContext,
    SessionMessageAccessDenied,
    SessionMessageAccessPolicy,
    SessionMessageActionRequest,
    SessionMessageConditions,
    SessionMessageConflict,
    SessionMessageCursor,
    SessionMessageQuery,
    SessionMessageTarget,
)
from cayu.core.events import Event, EventType
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.runtime.sessions import EventQuery, SessionStatus
from cayu.server import AuthContext, ServerConfig, create_server
from cayu.storage import SQLiteSessionStore
from cayu.vaults import SecretRedactor

CONTEXT = SessionMessageAccessContext(subject="alice", tenant="tenant-a")
HEADERS = {"Authorization": "Bearer alice"}
CONTENT = "private steering canary"


class OwnershipPolicy(SessionMessageAccessPolicy):
    """Trusted application table, deliberately independent of session metadata."""

    def __init__(self) -> None:
        self.grants: set[tuple[str, str | None, str, str, str]] = set()
        self.calls: list[tuple[str, str]] = []

    def authorize(self, context, *, session_id, session_instance_id, action):
        self.calls.append((session_id, action))
        return (
            context.subject,
            context.tenant,
            session_id,
            session_instance_id,
            action,
        ) in self.grants


async def authenticate(request: Request) -> AuthContext:
    if request.headers.get("Authorization") != HEADERS["Authorization"]:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return AuthContext(subject=CONTEXT.subject, tenant=CONTEXT.tenant)


def application(*, policy_enabled=True, redactor=None, store_type=InMemorySessionStore):
    store = store_type()
    policy = OwnershipPolicy()
    app = CayuApp(
        session_store=store,
        session_message_access_policy=policy if policy_enabled else None,
        secret_redactor=redactor,
        enable_logging=False,
    )

    async def seed():
        for name in ("target", "source", "foreign"):
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=name,
                    messages=[Message.text("user", "initial private transcript")],
                    metadata={"tenant": "tenant-a"},
                ),
                identity=SessionIdentity(provider_name="never-called", model="never-called"),
            )
            if name != "foreign":
                for action in ("inspect", "enqueue", "source", "withdraw", "quarantine"):
                    policy.grants.add(("alice", "tenant-a", name, session.instance_id, action))

    asyncio.run(seed())
    return app, store, policy


def enqueue_request(**updates: Any):
    values: dict[str, Any] = {
        "session_id": "target",
        "idempotency_key": "message-1",
        "content": CONTENT,
        "delivery_mode": "next_turn",
        **updates,
    }
    return EnqueueSessionMessageRequest(**values)


def client_for(app):
    return TestClient(create_server(app, config=ServerConfig.protected(authenticate)))


def assert_private(response):
    assert "no-store" in response.headers.get("Cache-Control", "")


def sse_events(response):
    return [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]


def cursor_params(cursor, *, limit=2):
    return {"limit": limit, **{f"cursor_{key}": value for key, value in cursor.items()}}


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_http_and_sdk_cursor_preserve_priority_fifo_highwater_and_unreadable_rows(
    tmp_path, backend
):
    database = tmp_path / "cursor.sqlite"
    factory = InMemorySessionStore if backend == "memory" else lambda: SQLiteSessionStore(database)
    app, store, _ = application(store_type=factory)
    client = client_for(app)
    try:
        accepted = {}
        for key, mode in (
            ("idle-a", "on_idle"),
            ("next-a", "next_turn"),
            ("unknown", "next_turn"),
            ("idle-b", "on_idle"),
            ("next-b", "next_turn"),
        ):
            accepted[key] = asyncio.run(
                app.enqueue_session_message(
                    enqueue_request(idempotency_key=key, delivery_mode=mode),
                    context=CONTEXT,
                )
            )
        if backend == "memory":
            object.__setattr__(
                store._queued_session_messages_by_idempotency["target"]["unknown"],
                "delivery_mode",
                "future-mode",
            )
        else:
            with sqlite3.connect(database) as connection:
                connection.execute("PRAGMA ignore_check_constraints = ON")
                connection.execute(
                    "UPDATE cayu_session_message_queue SET delivery_mode = ? WHERE queue_id = ?",
                    ("future-mode", accepted["unknown"].message.queue_id),
                )
        full = asyncio.run(
            app.inspect_session_messages(
                SessionMessageQuery(session_id="target"),
                context=CONTEXT,
            )
        )
        withdraw = next(
            row for row in full.records if row.queue_id == accepted["next-a"].message.queue_id
        )
        action = client.post(
            f"/api/sessions/target/messages/{withdraw.queue_id}/withdraw",
            headers=HEADERS,
            json={
                "session_instance_id": full.session_instance_id,
                "expected_revision": withdraw.revision,
                "idempotency_key": "withdraw-next-a",
            },
        )
        assert action.status_code == 200, action.text
        first = client.get("/api/sessions/target/messages", headers=HEADERS, params={"limit": 2})
        assert first.status_code == 200, first.text
        assert_private(first)
        page = first.json()
        rows = list(page["records"])
        assert [row["queue_id"] for row in rows] == [
            accepted[key].message.queue_id for key in ("next-a", "next-b")
        ]
        assert rows[0]["status"] == "withdrawn"
        cursor = page["next_cursor"]
        assert cursor == {
            "session_instance_id": full.session_instance_id,
            "through_ordering_key": accepted["next-b"].message.ordering_key,
            "after_priority": 0,
            "after_ordering_key": accepted["next-b"].message.ordering_key,
        }
        late = asyncio.run(
            app.enqueue_session_message(
                enqueue_request(idempotency_key="late-next"),
                context=CONTEXT,
            )
        )
        while cursor is not None:
            sdk_page = asyncio.run(
                app.inspect_session_messages(
                    SessionMessageQuery(
                        session_id="target",
                        cursor=SessionMessageCursor.model_validate(cursor),
                        limit=2,
                    ),
                    context=CONTEXT,
                )
            )
            response = client.get(
                "/api/sessions/target/messages", headers=HEADERS, params=cursor_params(cursor)
            )
            assert response.status_code == 200, response.text
            assert_private(response)
            page = response.json()
            assert page == sdk_page.model_dump(mode="json")
            rows.extend(page["records"])
            cursor = page["next_cursor"]
            if cursor is not None:
                assert cursor["through_ordering_key"] == accepted["next-b"].message.ordering_key
        assert [row["queue_id"] for row in rows] == [
            accepted[key].message.queue_id
            for key in ("next-a", "next-b", "idle-a", "idle-b", "unknown")
        ]
        assert rows[-1]["validity"] == "unreadable" and rows[-1]["message"] is None
        fresh = client.get("/api/sessions/target/messages", headers=HEADERS).json()
        assert late.message.queue_id in {row["queue_id"] for row in fresh["records"]}
        assert fresh["next_cursor"] is None
    finally:
        if backend == "sqlite":
            asyncio.run(store.close())


@pytest.mark.parametrize(
    "params",
    [
        {"after_ordering_key": "1"},
        {"cursor_after_priority": "0"},
        {"limit": "101"},
        {"cursor": "private-canary"},
        [("limit", "1"), ("limit", "2")],
        {
            "cursor_session_instance_id": "private-canary",
            "cursor_through_ordering_key": "1",
            "cursor_after_priority": "true",
            "cursor_after_ordering_key": "0",
        },
        {
            "cursor_session_instance_id": "private-canary",
            "cursor_through_ordering_key": "1",
            "cursor_after_priority": "3",
            "cursor_after_ordering_key": "0",
        },
        {
            "cursor_session_instance_id": "private-canary",
            "cursor_through_ordering_key": "1",
            "cursor_after_priority": "0",
            "cursor_after_ordering_key": "2",
        },
    ],
)
def test_http_cursor_rejects_partial_unknown_duplicate_and_invalid_parameters(params):
    app, store, _ = application()
    before = asyncio.run(store.load_events("target"))
    response = client_for(app).get("/api/sessions/target/messages", headers=HEADERS, params=params)
    assert response.status_code == 422, response.text
    assert_private(response)
    assert "private-canary" not in response.text
    assert asyncio.run(store.load_events("target")) == before


def test_http_cursor_never_grants_access_and_recreated_session_rejects_old_cursor():
    app, store, policy = application()
    for key in ("one", "two"):
        asyncio.run(
            app.enqueue_session_message(enqueue_request(idempotency_key=key), context=CONTEXT)
        )
    client = client_for(app)
    first = client.get("/api/sessions/target/messages", headers=HEADERS, params={"limit": 1}).json()
    cursor = first["next_cursor"]
    foreign = client.get(
        "/api/sessions/foreign/messages", headers=HEADERS, params=cursor_params(cursor)
    )
    assert foreign.status_code == 403
    grants = set(policy.grants)
    policy.grants.clear()
    revoked = client.get(
        "/api/sessions/target/messages", headers=HEADERS, params=cursor_params(cursor)
    )
    assert revoked.status_code == 403
    policy.grants = grants
    asyncio.run(store.delete_session("target"))
    replacement = asyncio.run(
        store.create(
            RunRequest(agent_name="assistant", session_id="target", messages=[]),
            identity=SessionIdentity(provider_name="never-called", model="never-called"),
        )
    )
    policy.grants.add(
        (CONTEXT.subject, CONTEXT.tenant, "target", replacement.instance_id, "inspect")
    )
    stale = client.get(
        "/api/sessions/target/messages", headers=HEADERS, params=cursor_params(cursor)
    )
    assert stale.status_code == 409, stale.text
    assert_private(stale)
    fresh = client.get("/api/sessions/target/messages", headers=HEADERS)
    assert fresh.status_code == 200
    assert fresh.json()["records"] == [] and fresh.json()["next_cursor"] is None


def test_sdk_cursor_is_detached_before_authorization_await(monkeypatch):
    app, store, _ = application()

    async def run():
        for key in ("one", "two"):
            await app.enqueue_session_message(enqueue_request(idempotency_key=key), context=CONTEXT)
        first = await app.inspect_session_messages(
            SessionMessageQuery(session_id="target", limit=1),
            context=CONTEXT,
        )
        query = SessionMessageQuery(session_id="target", cursor=first.next_cursor, limit=1)
        entered = asyncio.Event()
        release = asyncio.Event()
        load = store.load

        async def blocked_load(session_id):
            if session_id == "target":
                entered.set()
                await release.wait()
            return await load(session_id)

        monkeypatch.setattr(store, "load", blocked_load)
        task = asyncio.create_task(app.inspect_session_messages(query, context=CONTEXT))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            object.__setattr__(query.cursor, "after_priority", 2)
        finally:
            release.set()
        second = await task
        assert len(second.records) == 1
        assert second.records[0].message.idempotency_key == "two"
        assert second.next_cursor is None

    asyncio.run(run())


def test_http_enqueue_only_cannot_replay_history_but_can_retry_exact_acceptance():
    app, store, policy = application()
    policy.grants = {grant for grant in policy.grants if grant[-1] == "enqueue"}
    canary = "private-model-history-not-a-workload-secret"
    asyncio.run(
        store.append_event(
            "target",
            Event(type=EventType.MODEL_TEXT_DELTA, session_id="target", payload={"delta": canary}),
        )
    )
    client = client_for(app)
    body = {"idempotency_key": "one", "content": CONTENT, "delivery_mode": "next_turn"}
    before = asyncio.run(store.load_events("target"))
    denied = client.post(
        "/api/sessions/target/messages",
        json=body,
        headers={**HEADERS, "Last-Event-ID": "target:"},
    )
    assert denied.status_code == 400, denied.text
    assert_private(denied)
    assert canary not in denied.text
    assert asyncio.run(store.load_events("target")) == before
    assert client.get("/api/sessions/target/messages", headers=HEADERS).status_code == 403
    first = client.post("/api/sessions/target/messages", json=body, headers=HEADERS)
    retry = client.post("/api/sessions/target/messages", json=body, headers=HEADERS)
    assert first.status_code == retry.status_code == 200
    assert_private(first)
    assert_private(retry)
    assert sse_events(first) == sse_events(retry)
    assert len(sse_events(first)) == 1
    assert sse_events(first)[0]["type"] == "session.message.queued"
    assert canary not in first.text + retry.text
    conflict = client.post(
        "/api/sessions/target/messages", json={**body, "content": "changed"}, headers=HEADERS
    )
    assert conflict.status_code == 409
    assert (
        len(
            asyncio.run(
                store.inspect_session_messages(SessionMessageQuery(session_id="target"))
            ).records
        )
        == 1
    )


@pytest.mark.parametrize("prune_scope", [None, "target"])
@pytest.mark.parametrize(
    "outcome", ["queued", "withdraw", "quarantine", "delivered", "stale", "expired"]
)
def test_http_sqlite_pruning_retains_exact_references_and_fresh_app_replay(
    tmp_path, outcome, prune_scope
):
    database = tmp_path / "queue.sqlite"
    app, store, policy = application(store_type=lambda: SQLiteSessionStore(database))
    client = client_for(app)
    try:
        enqueue_body: dict[str, Any] = {
            "idempotency_key": "one",
            "content": CONTENT,
            "delivery_mode": "next_turn",
        }
        if outcome == "expired":
            enqueue_body["conditions"] = {"expires_at": "2000-01-01T00:00:00Z"}
        elif outcome == "stale":
            session = asyncio.run(store.load("target"))
            enqueue_body["conditions"] = {
                "target": {
                    "session_instance_id": session.instance_id,
                    "run_epoch": session.run_epoch,
                    "transcript_cursor": 999,
                }
            }
        accepted = client.post(
            "/api/sessions/target/messages",
            headers=HEADERS,
            json=enqueue_body,
        )
        assert accepted.status_code == 200, accepted.text
        accepted_id = sse_events(accepted)[0]["id"]
        if outcome == "quarantine":
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "UPDATE cayu_session_message_queue SET message_json = ? WHERE session_id = ?",
                    ('{"invalid":"malformed-private-content"}', "target"),
                )
        initial = client.get("/api/sessions/target/messages", headers=HEADERS)
        assert initial.status_code == 200, initial.text
        page = initial.json()
        record = page["records"][0]
        raw = asyncio.run(store.inspect_session_messages(SessionMessageQuery(session_id="target")))
        assert record["revision"] == raw.records[0].revision
        if outcome != "quarantine":
            assert record["message"]["accepted_event_id"] == accepted_id
            assert accepted_id != raw.records[0].message.accepted_event_id
        else:
            assert record["message"] is None
        terminal_id = None
        if outcome in {"delivered", "stale", "expired"}:

            async def deliver():
                await store.update_status("target", SessionStatus.RUNNING)
                batch = await store.deliver_queued_session_messages("target", include_on_idle=True)
                assert len(batch.messages) == (1 if outcome == "delivered" else 0)
                await app._event_writer.fan_out_persisted(list(batch.events))
                return await app._project_emitted_event_for_public_api(batch.events[-1])

            terminal_id = asyncio.run(deliver()).id
        elif outcome in {"withdraw", "quarantine"}:
            path = f"/api/sessions/target/messages/{record['queue_id']}/{outcome}"
            body = {
                "session_instance_id": page["session_instance_id"],
                "idempotency_key": "terminal-one",
                "expected_revision": record["revision"],
            }
            action = client.post(path, json=body, headers=HEADERS)
            assert action.status_code == 200, action.text
            terminal_id = action.json()["event"]["id"]
            assert action.json()["record"]["terminal_event_id"] == terminal_id

        before = client.get("/api/sessions/target/messages", headers=HEADERS)
        assert before.status_code == 200, before.text
        projected = before.json()["records"][0]
        assert projected["terminal_event_id"] == terminal_id
        raw = asyncio.run(store.inspect_session_messages(SessionMessageQuery(session_id="target")))
        assert projected["revision"] == raw.records[0].revision
        if terminal_id is not None:
            assert terminal_id != raw.records[0].terminal_event_id
        if outcome == "delivered":
            assert projected["message"]["delivered_event_id"] == terminal_id

        async def seed_prunable_history():
            for session_id in ("target", "foreign"):
                event = Event(
                    type=EventType.MODEL_TEXT_DELTA,
                    session_id=session_id,
                    payload={"delta": "unrelated private history"},
                )
                await store.append_event(session_id, event)
                await app._event_writer.fan_out_persisted([event])

        asyncio.run(seed_prunable_history())
        queue_events = asyncio.run(
            store.query_events(
                EventQuery(
                    session_id="target",
                    event_types=(
                        EventType.SESSION_MESSAGE_QUEUED,
                        EventType.SESSION_MESSAGE_DELIVERED,
                        EventType.SESSION_MESSAGE_WITHDRAWN,
                        EventType.SESSION_MESSAGE_QUARANTINED,
                        EventType.SESSION_MESSAGE_STALE,
                        EventType.SESSION_MESSAGE_EXPIRED,
                    ),
                )
            )
        )
        assert asyncio.run(
            store.prune_events(
                before=datetime.now(UTC) + timedelta(days=1),
                session_id=prune_scope,
            )
        ) == (2 if prune_scope is None else 1)
        assert asyncio.run(store.query_events(EventQuery(session_id="target"))) == queue_events
        transcript = asyncio.run(store.load_transcript("target"))
        asyncio.run(store.close())
        store = SQLiteSessionStore(database)
        fresh_policy = OwnershipPolicy()
        fresh_policy.grants = set(policy.grants)
        app = CayuApp(
            session_store=store,
            session_message_access_policy=fresh_policy,
            enable_logging=False,
        )
        client = client_for(app)
        after = client.get("/api/sessions/target/messages", headers=HEADERS)
        assert after.status_code == 200, after.text
        assert_private(after)
        assert after.json() == before.json()
        assert "malformed-private-content" not in after.text
        if outcome in {"withdraw", "quarantine"}:
            for _ in range(2):
                replay = client.post(path, json=body, headers=HEADERS)
                assert replay.status_code == 200, replay.text
                assert_private(replay)
                assert replay.json()["replayed"] is True
                assert replay.json()["event"] == action.json()["event"]
                assert replay.json()["record"]["terminal_event_id"] == terminal_id
                assert CONTENT not in str(replay.json()["event"])
        elif outcome == "queued":
            retry = client.post("/api/sessions/target/messages", json=enqueue_body, headers=HEADERS)
            assert retry.status_code == 200, retry.text
            assert sse_events(retry) == sse_events(accepted)
        assert asyncio.run(store.query_events(EventQuery(session_id="target"))) == queue_events
        assert asyncio.run(store.load_transcript("target")) == transcript
        assert (
            asyncio.run(store.inspect_session_messages(SessionMessageQuery(session_id="target")))
            .records[0]
            .revision
            == raw.records[0].revision
        )
    finally:
        asyncio.run(store.close())


def test_http_side_effect_receipt_cannot_replace_missing_queue_event_evidence(monkeypatch):
    app, store, _ = application()
    client = client_for(app)
    accepted = client.post(
        "/api/sessions/target/messages",
        headers=HEADERS,
        json={"idempotency_key": "one", "content": CONTENT, "delivery_mode": "next_turn"},
    )
    assert accepted.status_code == 200
    page = client.get("/api/sessions/target/messages", headers=HEADERS).json()
    record = page["records"][0]
    path = f"/api/sessions/target/messages/{record['queue_id']}/withdraw"
    body = {
        "session_instance_id": page["session_instance_id"],
        "idempotency_key": "withdraw-one",
        "expected_revision": record["revision"],
    }
    first = client.post(path, headers=HEADERS, json=body)
    assert first.status_code == 200, first.text
    raw = asyncio.run(store.inspect_session_messages(SessionMessageQuery(session_id="target")))
    assert (
        asyncio.run(
            store.get_persisted_event_side_effect_delivery(
                session_id="target",
                event_id=raw.records[0].terminal_event_id,
            )
        )
        is not None
    )

    async def history_unavailable(query):
        return []

    monkeypatch.setattr(store, "query_events", history_unavailable)
    after = client.get("/api/sessions/target/messages", headers=HEADERS)
    assert after.status_code == 409, after.text
    assert_private(after)
    assert CONTENT not in after.text
    replay = client.post(path, headers=HEADERS, json=body)
    assert replay.status_code == 409, replay.text
    assert_private(replay)
    assert CONTENT not in replay.text
    assert (
        asyncio.run(store.inspect_session_messages(SessionMessageQuery(session_id="target")))
        .records[0]
        .revision
        == raw.records[0].revision
    )


@pytest.mark.parametrize("history_unavailable", [False, True])
@pytest.mark.parametrize("bad_reference", ["missing", "other_queue", "wrong_kind", "other_session"])
def test_http_event_reference_requires_session_queue_and_event_kind(
    bad_reference, history_unavailable, monkeypatch
):
    app, store, _ = application()
    client = client_for(app)
    for key in ("one", "two"):
        response = client.post(
            "/api/sessions/target/messages",
            headers=HEADERS,
            json={"idempotency_key": key, "content": CONTENT, "delivery_mode": "next_turn"},
        )
        assert response.status_code == 200
    before = client.get("/api/sessions/target/messages", headers=HEADERS)
    assert before.status_code == 200
    stored = store._queued_session_messages_by_idempotency["target"]["one"]
    if bad_reference == "other_queue":
        event_id = store._queued_session_messages_by_idempotency["target"]["two"].accepted_event_id
    elif bad_reference == "other_session":
        source = asyncio.run(
            app.enqueue_session_message(
                enqueue_request(session_id="source"),
                context=CONTEXT,
            )
        )
        event_id = store._queued_session_messages_by_idempotency["source"][
            "message-1"
        ].accepted_event_id
        assert source.event.id != event_id
    elif bad_reference == "wrong_kind":
        event = Event(
            type=EventType.MODEL_TEXT_DELTA,
            session_id="target",
            payload={"queue_id": stored.queue_id, "delta": CONTENT},
        )
        asyncio.run(store.append_event("target", event))
        event_id = event.id
    else:
        event_id = "missing-private-event"
    if bad_reference != "missing":
        assert (
            asyncio.run(
                store.get_persisted_event_side_effect_delivery(
                    session_id="source" if bad_reference == "other_session" else "target",
                    event_id=event_id,
                )
            )
            is not None
        )
    stored.accepted_event_id = event_id
    if history_unavailable:

        async def no_history(query):
            return []

        monkeypatch.setattr(store, "query_events", no_history)
    projected = client.get("/api/sessions/target/messages", headers=HEADERS)
    assert projected.status_code == 200, projected.text
    assert_private(projected)
    damaged = next(row for row in projected.json()["records"] if row["queue_id"] == stored.queue_id)
    assert damaged["validity"] == "unreadable" and damaged["message"] is None
    assert event_id not in projected.text


@pytest.mark.parametrize("field", ["session_id", "id", "type", "queue_id", "duplicate"])
def test_http_marks_record_unreadable_for_conflicting_acceptance_linkage(monkeypatch, field):
    app, store, _ = application()
    asyncio.run(app.enqueue_session_message(enqueue_request(), context=CONTEXT))
    client = client_for(app)
    assert client.get("/api/sessions/target/messages", headers=HEADERS).status_code == 200
    query_events = store.query_events

    async def conflicting_lookup(query):
        records = await query_events(query)
        if not records:
            return records
        record = records[0]
        if field == "duplicate":
            return [record, record.model_copy(deep=True)]
        if field == "queue_id":
            event = record.event.model_copy(
                update={"payload": {**record.event.payload, "queue_id": "wrong-queue"}}, deep=True
            )
        else:
            value = EventType.MODEL_TEXT_DELTA if field == "type" else "wrong-private-identity"
            event = record.event.model_copy(update={field: value}, deep=True)
        return [record.model_copy(update={"event": event}, deep=True)]

    monkeypatch.setattr(store, "query_events", conflicting_lookup)
    projected = client.get("/api/sessions/target/messages", headers=HEADERS)
    assert projected.status_code == 200, projected.text
    assert_private(projected)
    assert projected.json()["records"][0]["validity"] == "unreadable"
    assert projected.json()["records"][0]["message"] is None
    assert CONTENT not in projected.text and "wrong-private-identity" not in projected.text


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("private_source_id", ["source", "cayu_authority_source"])
def test_source_alias_carries_separate_resolution_proof_through_sdk_and_http(
    tmp_path, backend, private_source_id
):
    # A private identity containing a secret is aliasable. An identity equal to
    # an entire secret remains forbidden by the existing authority contract.
    codec = PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={
                "test": SecretStr(
                    base64.urlsafe_b64encode(bytes(range(32))).decode("ascii").rstrip("=")
                )
            },
        )
    )
    factory = (
        InMemorySessionStore
        if backend == "memory"
        else lambda: SQLiteSessionStore(
            tmp_path / "aliases.sqlite", public_authority_alias_codec=codec
        )
    )
    app, store, policy = application(store_type=factory, redactor=SecretRedactor("our"))
    try:
        if private_source_id != "source":
            session = asyncio.run(
                store.create(
                    RunRequest(agent_name="assistant", session_id=private_source_id, messages=[]),
                    identity=SessionIdentity(provider_name="never-called", model="never-called"),
                )
            )
            policy.grants.add(
                (CONTEXT.subject, CONTEXT.tenant, private_source_id, session.instance_id, "source")
            )
        alias = app.project_session_id_for_exposure(private_source_id)
        assert alias != private_source_id and "our" not in alias
        asyncio.run(
            store.register_public_authority_alias(
                alias,
                field_name="session_id",
                private_value=private_source_id,
            )
        )
        observation = asyncio.run(
            app.snapshot_session_message_source(
                alias,
                context=CONTEXT,
                include_transcript_digest=True,
                include_checkpoint_digest=True,
            )
        )
        assert observation.session_id == alias
        raw = observation.model_copy(update={"session_id": private_source_id})
        with pytest.raises(SessionMessageAccessDenied):
            asyncio.run(app.snapshot_session_message_source(private_source_id, context=CONTEXT))
        with pytest.raises(SessionMessageAccessDenied):
            asyncio.run(
                app.enqueue_session_message(
                    enqueue_request(conditions=SessionMessageConditions(source=raw)),
                    context=CONTEXT,
                )
            )
        accepted = asyncio.run(
            app.enqueue_session_message(
                enqueue_request(conditions=SessionMessageConditions(source=observation)),
                context=CONTEXT,
            )
        )
        assert accepted.message.conditions.source == observation
        assert accepted.event.payload["source"]["session_id"] == alias
        client = client_for(app)
        http_snapshot = client.post(
            f"/api/sessions/{alias}/messages/source-snapshot",
            headers=HEADERS,
            json={"include_transcript_digest": True, "include_checkpoint_digest": True},
        )
        assert http_snapshot.status_code == 200, http_snapshot.text
        assert http_snapshot.json() == observation.model_dump(mode="json")
        body = {
            "idempotency_key": "http-alias",
            "content": CONTENT,
            "delivery_mode": "next_turn",
            "conditions": {"source": raw.model_dump(mode="json")},
        }
        denied = client.post("/api/sessions/target/messages", headers=HEADERS, json=body)
        assert denied.status_code == 403, denied.text
        assert_private(denied)
        body["conditions"] = {"source": observation.model_dump(mode="json")}
        http = client.post("/api/sessions/target/messages", headers=HEADERS, json=body)
        assert http.status_code == 200, http.text
        assert sse_events(http)[0]["payload"]["source"]["session_id"] == alias
        page = client.get("/api/sessions/target/messages", headers=HEADERS)
        assert page.status_code == 200, page.text
        for record in page.json()["records"]:
            assert record["message"]["conditions"]["source"] == observation.model_dump(mode="json")
        durable = asyncio.run(
            store.inspect_session_messages(SessionMessageQuery(session_id="target"))
        )
        assert len(durable.records) == 2
        assert all(record.message.conditions.source == raw for record in durable.records)
        assert (private_source_id, "source") in policy.calls
        policy.grants = {grant for grant in policy.grants if grant[-1] != "source"}
        revoked = client.post("/api/sessions/target/messages", headers=HEADERS, json=body)
        assert revoked.status_code == 403
        assert_private(revoked)
    finally:
        if backend == "sqlite":
            asyncio.run(store.close())


def test_sdk_ordinary_enqueue_still_works_without_policy_but_inspection_does_not():
    app, store, _ = application(policy_enabled=False)

    async def run():
        result = await app.enqueue_session_message(enqueue_request())
        assert result.message.content == CONTENT
        with pytest.raises(SessionMessageAccessDenied):
            await app.inspect_session_messages(
                SessionMessageQuery(session_id="target"), context=CONTEXT
            )
        page = await store.inspect_session_messages(SessionMessageQuery(session_id="target"))
        assert len(page.records) == 1

    asyncio.run(run())


def test_sdk_scoped_enqueue_requires_context_and_authorizes_source_and_target():
    app, store, policy = application()

    async def run():
        source = await app.snapshot_session_message_source(
            "source",
            context=CONTEXT,
            include_transcript_digest=True,
            include_checkpoint_digest=True,
        )
        target = await store.load("target")
        conditions = SessionMessageConditions(
            source=source,
            target=SessionMessageTarget(
                session_instance_id=target.instance_id,
                run_epoch=target.run_epoch,
                transcript_cursor=1,
            ),
        )
        with pytest.raises(SessionMessageAccessDenied):
            await app.enqueue_session_message(enqueue_request(conditions=conditions))
        accepted = await app.enqueue_session_message(
            enqueue_request(conditions=conditions), context=CONTEXT
        )
        assert accepted.message.conditions == conditions
        assert ("source", "source") in policy.calls
        assert ("target", "enqueue") in policy.calls
        assert CONTENT not in str(accepted.event.payload)
        assert (await store.load("target")).status == "pending"

    asyncio.run(run())


@pytest.mark.parametrize("foreign_side", ["source", "target"])
def test_sdk_foreign_source_or_target_fails_before_queue_write(foreign_side):
    app, store, _ = application()

    async def run():
        source = await store.snapshot_session_message_source(
            "foreign" if foreign_side == "source" else "source"
        )
        target = "foreign" if foreign_side == "target" else "target"
        with pytest.raises(SessionMessageAccessDenied):
            await app.enqueue_session_message(
                enqueue_request(
                    session_id=target, conditions=SessionMessageConditions(source=source)
                ),
                context=CONTEXT,
            )
        assert not (
            await store.inspect_session_messages(SessionMessageQuery(session_id=target))
        ).records

    asyncio.run(run())


def test_source_snapshot_denied_before_store_reads_protected_content(monkeypatch):
    app, store, _ = application()

    async def forbidden(*args, **kwargs):
        pytest.fail("A denied source reached raw snapshot access")

    monkeypatch.setattr(store, "snapshot_session_message_source", forbidden)
    with pytest.raises(SessionMessageAccessDenied):
        asyncio.run(
            app.snapshot_session_message_source(
                "foreign", context=CONTEXT, include_transcript_digest=True
            )
        )


@pytest.mark.parametrize(
    "path,body",
    [
        (
            "/api/sessions/target/messages",
            {"idempotency_key": "one", "content": CONTENT, "delivery_mode": "next_turn"},
        ),
        ("/api/sessions/target/messages/source-snapshot", {}),
    ],
)
def test_http_requires_policy_even_with_valid_authentication(path, body):
    app, store, _ = application(policy_enabled=False)
    response = client_for(app).post(path, json=body, headers=HEADERS)
    assert response.status_code == 403
    assert_private(response)
    assert CONTENT not in response.text
    assert not asyncio.run(
        store.inspect_session_messages(SessionMessageQuery(session_id="target"))
    ).records


@pytest.mark.parametrize("schema_collision", [None, "status", "withdrawn"])
def test_http_inspect_withdraw_and_replay_are_private_and_attributable(schema_collision):
    app, store, _ = application(
        redactor=None if schema_collision is None else SecretRedactor(schema_collision)
    )
    client = client_for(app)
    response = client.post(
        "/api/sessions/target/messages",
        headers=HEADERS,
        json={"idempotency_key": "one", "content": CONTENT, "delivery_mode": "next_turn"},
    )
    assert response.status_code == 200, response.text
    assert_private(response)
    assert CONTENT not in response.text
    response = client.get("/api/sessions/target/messages", headers=HEADERS)
    assert response.status_code == 200, response.text
    assert_private(response)
    page = response.json()
    record = page["records"][0]
    assert record["message"]["content"] == CONTENT
    assert record["message"]["requested_by"]["source"] == "http_auth"
    body = {
        "session_instance_id": page["session_instance_id"],
        "idempotency_key": "withdraw-one",
        "expected_revision": record["revision"],
    }
    path = f"/api/sessions/target/messages/{record['queue_id']}/withdraw"
    first = client.post(path, json=body, headers=HEADERS)
    assert first.status_code == 200, first.text
    assert_private(first)
    assert first.json()["record"]["status"] == "withdrawn"
    assert first.json()["event"]["payload"]["actor"]["subject"] == "alice"
    assert CONTENT not in str(first.json()["event"])
    replay = client.post(path, json=body, headers=HEADERS)
    assert replay.status_code == 200, replay.text
    assert replay.json()["replayed"] is True
    assert replay.json()["event"] == first.json()["event"]
    assert first.json()["event"]["payload"]["status"] == "withdrawn"
    stored = asyncio.run(store.load_events("target"))
    terminal = [event for event in stored if event.type == "session.message.withdrawn"]
    assert len(terminal) == 1
    assert terminal[0].payload["status"] == "withdrawn"
    assert CONTENT not in str(terminal[0].payload)
    assert asyncio.run(store.load("target")).status == "pending"


@pytest.mark.parametrize(
    "field,value",
    [
        ("context", {"subject": "alice", "tenant": "tenant-a"}),
        ("requested_by", {"subject": "alice", "tenant": "tenant-a", "source": "http_auth"}),
        ("metadata", {"tenant": "tenant-a"}),
    ],
)
def test_http_rejects_context_actor_and_metadata_spoof_without_mutation(field, value):
    app, store, _ = application()
    response = client_for(app).post(
        "/api/sessions/target/messages",
        headers=HEADERS,
        json={
            "idempotency_key": "one",
            "content": CONTENT,
            "delivery_mode": "next_turn",
            field: value,
        },
    )
    assert response.status_code in {400, 422}, response.text
    assert_private(response)
    assert CONTENT not in response.text
    assert not asyncio.run(
        store.inspect_session_messages(SessionMessageQuery(session_id="target"))
    ).records


@pytest.mark.parametrize("session_id", ["foreign", "missing"])
def test_http_inspection_and_snapshot_deny_foreign_and_missing_uniformly(session_id):
    app, _, _ = application()
    client = client_for(app)
    responses = [
        client.get(f"/api/sessions/{session_id}/messages", headers=HEADERS),
        client.post(
            f"/api/sessions/{session_id}/messages/source-snapshot", json={}, headers=HEADERS
        ),
    ]
    for response in responses:
        assert response.status_code == 403
        assert_private(response)
        assert session_id not in response.text


def test_http_source_snapshot_and_conditions_reach_durable_queue():
    app, store, _ = application()
    client = client_for(app)
    response = client.post(
        "/api/sessions/source/messages/source-snapshot",
        headers=HEADERS,
        json={"include_transcript_digest": True, "include_checkpoint_digest": True},
    )
    assert response.status_code == 200, response.text
    assert_private(response)
    source = response.json()
    assert len(source["transcript_sha256"]) == 64
    response = client.post(
        "/api/sessions/target/messages",
        headers=HEADERS,
        json={
            "idempotency_key": "one",
            "content": CONTENT,
            "delivery_mode": "next_turn",
            "conditions": {"source": source},
        },
    )
    assert response.status_code == 200, response.text
    queued = asyncio.run(store.inspect_session_messages(SessionMessageQuery(session_id="target")))
    assert queued.records[0].message.conditions.source.session_id == "source"


def test_sdk_rejects_secret_condition_authority_without_redacting_it():
    app, store, _ = application(redactor=SecretRedactor("secret-instance"))

    async def run():
        with pytest.raises(ValueError):
            await app.enqueue_session_message(
                enqueue_request(
                    conditions=SessionMessageConditions(
                        target=SessionMessageTarget(
                            session_instance_id="secret-instance", run_epoch=0, transcript_cursor=1
                        ),
                    )
                ),
                context=CONTEXT,
            )
        assert not (
            await store.inspect_session_messages(SessionMessageQuery(session_id="target"))
        ).records

    asyncio.run(run())


def test_sdk_actions_reject_foreign_context_without_terminal_mutation():
    app, store, _ = application()

    async def run():
        await app.enqueue_session_message(enqueue_request(), context=CONTEXT)
        page = await app.inspect_session_messages(
            SessionMessageQuery(session_id="target"), context=CONTEXT
        )
        record = page.records[0]
        with pytest.raises(SessionMessageAccessDenied):
            await app.apply_session_message_action(
                SessionMessageActionRequest(
                    session_id="target",
                    session_instance_id=page.session_instance_id,
                    queue_id=record.queue_id,
                    expected_revision=record.revision,
                    action="quarantine",
                    idempotency_key="quarantine-one",
                ),
                context=SessionMessageAccessContext(subject="bob", tenant="tenant-b"),
            )
        assert (
            await store.inspect_session_messages(SessionMessageQuery(session_id="target"))
        ).records[0].status == "queued"

    asyncio.run(run())


@pytest.mark.parametrize("foreign_side", ["source", "target"])
def test_http_derivative_enqueue_authorizes_both_scopes(foreign_side):
    app, store, _ = application()
    source = asyncio.run(
        store.snapshot_session_message_source("foreign" if foreign_side == "source" else "source")
    )
    target = "foreign" if foreign_side == "target" else "target"
    response = client_for(app).post(
        f"/api/sessions/{target}/messages",
        headers=HEADERS,
        json={
            "idempotency_key": "one",
            "content": CONTENT,
            "delivery_mode": "next_turn",
            "conditions": {"source": source.model_dump(mode="json")},
        },
    )
    assert response.status_code == 403, response.text
    assert_private(response)
    assert CONTENT not in response.text
    assert not asyncio.run(
        store.inspect_session_messages(SessionMessageQuery(session_id=target))
    ).records


@pytest.mark.parametrize("action", ["withdraw", "quarantine"])
def test_http_actions_reject_foreign_target_and_actor_spoof(action):
    app, store, _ = application()
    asyncio.run(app.enqueue_session_message(enqueue_request(), context=CONTEXT))
    page = asyncio.run(
        app.inspect_session_messages(SessionMessageQuery(session_id="target"), context=CONTEXT)
    )
    record = page.records[0]
    body = {
        "session_instance_id": page.session_instance_id,
        "idempotency_key": "action-one",
        "expected_revision": record.revision,
    }
    client = client_for(app)
    denied = client.post(
        f"/api/sessions/foreign/messages/{record.queue_id}/{action}", json=body, headers=HEADERS
    )
    assert denied.status_code == 403
    assert_private(denied)
    spoofed = client.post(
        f"/api/sessions/target/messages/{record.queue_id}/{action}",
        headers=HEADERS,
        json={**body, "requested_by": {"subject": "alice", "source": "http_auth"}},
    )
    assert spoofed.status_code == 422
    assert_private(spoofed)
    assert (
        asyncio.run(store.inspect_session_messages(SessionMessageQuery(session_id="target")))
        .records[0]
        .status
        == "queued"
    )


def test_http_quarantine_unreadable_record_never_projects_its_content():
    app, store, _ = application()
    asyncio.run(app.enqueue_session_message(enqueue_request(), context=CONTEXT))
    # External mutation: a row whose required display content no longer parses.
    stored = store._queued_session_messages_by_idempotency["target"]["message-1"]
    stored.content = None
    client = client_for(app)
    response = client.get("/api/sessions/target/messages", headers=HEADERS)
    assert response.status_code == 200, response.text
    page = response.json()
    record = page["records"][0]
    assert record["validity"] == "unreadable"
    assert record["message"] is None
    assert CONTENT not in response.text
    response = client.post(
        f"/api/sessions/target/messages/{record['queue_id']}/quarantine",
        headers=HEADERS,
        json={
            "session_instance_id": page["session_instance_id"],
            "idempotency_key": "quarantine-one",
            "expected_revision": record["revision"],
        },
    )
    assert response.status_code == 200, response.text
    assert_private(response)
    assert response.json()["record"]["status"] == "quarantined"
    assert response.json()["record"]["message"] is None
    assert CONTENT not in response.text


@pytest.mark.parametrize("grant", [None, 1, "yes"])
def test_policy_requires_literal_positive_boolean(grant, monkeypatch):
    app, store, policy = application()
    monkeypatch.setattr(policy, "authorize", lambda *args, **kwargs: grant)
    with pytest.raises(SessionMessageAccessDenied):
        asyncio.run(app.enqueue_session_message(enqueue_request(), context=CONTEXT))
    assert not asyncio.run(
        store.inspect_session_messages(SessionMessageQuery(session_id="target"))
    ).records


def test_http_validation_and_policy_failures_do_not_echo_canaries(monkeypatch, caplog, capsys):
    app, store, policy = application()
    client = client_for(app)
    secret = "unregistered-private-diagnostic-canary"

    def fail_policy(*args, **kwargs):
        raise ValueError(secret)

    monkeypatch.setattr(policy, "authorize", fail_policy)
    response = client.get("/api/sessions/target/messages", headers=HEADERS)
    assert response.status_code == 500
    assert_private(response)
    assert secret not in response.text
    response = client.post(
        "/api/sessions/target/messages/source-snapshot",
        headers=HEADERS,
        json={"include_transcript_digest": secret, "context": {"subject": secret}},
    )
    assert response.status_code == 422
    assert_private(response)
    captured = capsys.readouterr()
    assert secret not in response.text + caplog.text + captured.out + captured.err
    assert not asyncio.run(
        store.inspect_session_messages(SessionMessageQuery(session_id="target"))
    ).records


def test_http_enqueue_replay_rechecks_revoked_permission():
    app, _, policy = application()
    client = client_for(app)
    body = {"idempotency_key": "one", "content": CONTENT, "delivery_mode": "next_turn"}
    first = client.post("/api/sessions/target/messages", headers=HEADERS, json=body)
    assert first.status_code == 200, first.text
    policy.grants.clear()
    denied = client.post(
        "/api/sessions/target/messages",
        headers={**HEADERS, "Last-Event-ID": "target:"},
        json=body,
    )
    assert denied.status_code == 403
    assert_private(denied)
    assert "session.message.queued" not in denied.text


@pytest.mark.parametrize("source", list(ResolutionActorSource))
def test_scoped_sdk_cannot_supply_matching_actor_with_forged_source_or_claims(source):
    app, store, _ = application()

    async def run():
        actor = ResolutionActor(
            subject="alice", tenant="tenant-a", source=source, claims={"role": "admin"}
        )
        with pytest.raises(SessionMessageAccessDenied):
            await app.enqueue_session_message(enqueue_request(requested_by=actor), context=CONTEXT)
        assert not (
            await store.inspect_session_messages(SessionMessageQuery(session_id="target"))
        ).records
        accepted = await app.enqueue_session_message(enqueue_request(), context=CONTEXT)
        assert accepted.message.requested_by == ResolutionActor(
            subject="alice", tenant="tenant-a", source=ResolutionActorSource.REQUEST
        )
        page = await app.inspect_session_messages(
            SessionMessageQuery(session_id="target"), context=CONTEXT
        )
        record = page.records[0]
        with pytest.raises(SessionMessageAccessDenied):
            await app.apply_session_message_action(
                SessionMessageActionRequest(
                    session_id="target",
                    session_instance_id=page.session_instance_id,
                    queue_id=record.queue_id,
                    expected_revision=record.revision,
                    action="withdraw",
                    idempotency_key="withdraw-one",
                    requested_by=actor,
                ),
                context=CONTEXT,
            )
        assert (
            await store.inspect_session_messages(SessionMessageQuery(session_id="target"))
        ).records[0].status == "queued"

    asyncio.run(run())


def test_public_sdk_cannot_forge_runtime_scenario_actor():
    app, store, _ = application(policy_enabled=False)

    async def run():
        before = await store.load_events("target")
        with pytest.raises(ValueError, match="reserved for system actors"):
            await app.enqueue_session_message(
                enqueue_request(
                    requested_by=ResolutionActor(
                        subject="cayu:eval-scenario", source=ResolutionActorSource.SYSTEM
                    )
                )
            )
        assert not (
            await store.inspect_session_messages(SessionMessageQuery(session_id="target"))
        ).records
        assert await store.load_events("target") == before

    asyncio.run(run())


@pytest.mark.parametrize("policy_enabled", [False, True])
def test_scenario_enqueue_retains_scoped_admission_and_runtime_actor(policy_enabled):
    app, store, _ = application(policy_enabled=policy_enabled)

    async def run():
        if policy_enabled:
            with pytest.raises(SessionMessageAccessDenied):
                await app._enqueue_session_message_from_scenario(enqueue_request())
            assert not (
                await store.inspect_session_messages(SessionMessageQuery(session_id="target"))
            ).records
        else:
            accepted = await app._enqueue_session_message_from_scenario(enqueue_request())
            assert accepted.message.requested_by == ResolutionActor(
                subject="cayu:eval-scenario", source=ResolutionActorSource.SYSTEM
            )
            replayed = await app._enqueue_session_message_from_scenario(enqueue_request())
            assert replayed.replayed
            assert replayed.message == accepted.message

        with pytest.raises(SessionMessageAccessDenied):
            await app._enqueue_session_message_from_scenario(
                enqueue_request(
                    requested_by=ResolutionActor(
                        subject="cayu:eval-scenario", source=ResolutionActorSource.SYSTEM
                    )
                )
            )

    asyncio.run(run())


def test_unscoped_sdk_actor_is_re_stamped_as_request_without_claims():
    app, store, _ = application(policy_enabled=False)

    async def run():
        await app.enqueue_session_message(
            enqueue_request(
                requested_by=ResolutionActor(
                    subject="alice",
                    tenant="tenant-a",
                    source=ResolutionActorSource.HTTP_AUTH,
                    claims={"role": "admin"},
                )
            )
        )
        page = await store.inspect_session_messages(SessionMessageQuery(session_id="target"))
        assert page.records[0].message.requested_by == ResolutionActor(
            subject="alice",
            tenant="tenant-a",
            source=ResolutionActorSource.REQUEST,
        )

    asyncio.run(run())


def test_store_read_failure_is_not_permission_denial_and_http_diagnostic_is_private(monkeypatch):
    app, store, _ = application()
    original = OSError("private database connection canary")

    async def failed_load(*args, **kwargs):
        raise original

    monkeypatch.setattr(store, "load", failed_load)
    with pytest.raises(OSError) as raised:
        asyncio.run(
            app.inspect_session_messages(SessionMessageQuery(session_id="target"), context=CONTEXT)
        )
    assert raised.value is original
    response = client_for(app).get("/api/sessions/target/messages", headers=HEADERS)
    assert response.status_code == 500
    assert_private(response)
    assert "database" not in response.text


def test_policy_evaluation_failure_has_distinct_sdk_classification(monkeypatch):
    from cayu.runtime._session_message_coordinator import SessionMessageAuthorizationUnavailable

    app, _, policy = application()
    original = OSError("policy backend unavailable")

    def failed_policy(*args, **kwargs):
        raise original

    monkeypatch.setattr(policy, "authorize", failed_policy)
    with pytest.raises(SessionMessageAuthorizationUnavailable) as raised:
        asyncio.run(
            app.inspect_session_messages(SessionMessageQuery(session_id="target"), context=CONTEXT)
        )
    assert raised.value.__cause__ is original


@pytest.mark.parametrize("recreate", [False, True])
def test_source_deletion_or_recreation_cannot_authorize_old_acceptance_replay(recreate):
    app, store, policy = application()

    async def run():
        source = await app.snapshot_session_message_source("source", context=CONTEXT)
        request = enqueue_request(conditions=SessionMessageConditions(source=source))
        accepted = await app.enqueue_session_message(request, context=CONTEXT)
        await store.delete_session("source")
        if recreate:
            replacement = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="source",
                    messages=[Message.text("user", "replacement")],
                ),
                identity=SessionIdentity(provider_name="never-called", model="never-called"),
            )
            policy.grants.add(("alice", "tenant-a", "source", replacement.instance_id, "source"))
        with pytest.raises(SessionMessageConflict if recreate else SessionMessageAccessDenied):
            await app.enqueue_session_message(request, context=CONTEXT)
        page = await app.inspect_session_messages(
            SessionMessageQuery(session_id="target"), context=CONTEXT
        )
        assert len(page.records) == 1
        assert page.records[0].queue_id == accepted.message.queue_id
        assert page.records[0].message.conditions.source == source

    asyncio.run(run())


def test_http_openapi_queue_errors_match_private_detail_envelope():
    app, _, _ = application()
    schema = create_server(app, config=ServerConfig.protected(authenticate)).openapi()
    paths = schema["paths"]
    for path, method in (
        ("/api/sessions/{session_id}/messages", "get"),
        ("/api/sessions/{session_id}/messages", "post"),
        ("/api/sessions/{session_id}/messages/source-snapshot", "post"),
        ("/api/sessions/{session_id}/messages/{queue_id}/withdraw", "post"),
        ("/api/sessions/{session_id}/messages/{queue_id}/quarantine", "post"),
    ):
        assert paths[path][method]["responses"]["422"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/ApiErrorResponse"
        }


def test_same_identity_does_not_make_sdk_actor_equivalent_to_authenticated_http_actor():
    app, store, _ = application()
    accepted = asyncio.run(app.enqueue_session_message(enqueue_request(), context=CONTEXT))
    assert accepted.message.requested_by.source == "request"
    response = client_for(app).post(
        "/api/sessions/target/messages",
        headers=HEADERS,
        json={"idempotency_key": "message-1", "content": CONTENT, "delivery_mode": "next_turn"},
    )
    assert response.status_code == 409, response.text
    assert_private(response)
    page = asyncio.run(store.inspect_session_messages(SessionMessageQuery(session_id="target")))
    assert len(page.records) == 1
    assert page.records[0].message.requested_by.source == "request"


def test_policy_explicit_typed_denial_remains_permission_denial(monkeypatch):
    app, _, policy = application()

    def deny(*args, **kwargs):
        raise SessionMessageAccessDenied()

    monkeypatch.setattr(policy, "authorize", deny)
    response = client_for(app).get("/api/sessions/target/messages", headers=HEADERS)
    assert response.status_code == 403
    assert_private(response)


@pytest.mark.parametrize("failure,status", [("deny", 403), ("unavailable", 500)])
def test_http_detached_admission_preserves_permission_vs_operational_failure(
    monkeypatch, failure, status
):
    app, store, policy = application()
    authorize = policy.authorize
    calls = 0

    def change_after_preflight(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            if failure == "unavailable":
                raise OSError("unregistered policy canary")
            return False
        return authorize(*args, **kwargs)

    monkeypatch.setattr(policy, "authorize", change_after_preflight)
    response = client_for(app).post(
        "/api/sessions/target/messages",
        headers=HEADERS,
        json={"idempotency_key": "one", "content": CONTENT, "delivery_mode": "next_turn"},
    )
    assert response.status_code == status, response.text
    assert_private(response)
    assert "canary" not in response.text
    assert not asyncio.run(
        store.inspect_session_messages(SessionMessageQuery(session_id="target"))
    ).records


@pytest.mark.parametrize("version", [None, False, True, 0, 2, "1"])
def test_lifecycle_guard_requires_exact_supported_integer_before_conditional_admission(version):
    class UnsupportedStore(InMemorySessionStore):
        session_message_lifecycle_version = version

        async def enqueue_session_message(self, request):
            pytest.fail("Unsupported conditional admission reached the mutating store method")

    app, _, _ = application(store_type=UnsupportedStore)
    with pytest.raises(NotImplementedError):
        asyncio.run(
            app.enqueue_session_message(
                enqueue_request(
                    conditions=SessionMessageConditions(
                        expires_at=datetime(2099, 1, 1, tzinfo=UTC),
                    )
                ),
                context=CONTEXT,
            )
        )


@pytest.mark.parametrize(
    "operation",
    [
        "enqueue_session_message",
        "deliver_queued_session_messages",
        "inspect_session_messages",
        "apply_session_message_action",
        "snapshot_session_message_source",
    ],
)
def test_lifecycle_override_cannot_inherit_another_implementation_attestation(operation):
    class AttestedParent(InMemorySessionStore):
        session_message_lifecycle_version = 1

    async def unsupported(*args, **kwargs):
        pytest.fail("An unattested operation was invoked")

    custom = type("UnattestedOverride", (AttestedParent,), {operation: unsupported})
    app, store, _ = application(store_type=custom)

    async def run():
        with pytest.raises(NotImplementedError):
            if operation in {"enqueue_session_message", "deliver_queued_session_messages"}:
                await app.enqueue_session_message(
                    enqueue_request(
                        conditions=SessionMessageConditions(
                            expires_at=datetime(2099, 1, 1, tzinfo=UTC),
                        )
                    ),
                    context=CONTEXT,
                )
            elif operation == "inspect_session_messages":
                await app.inspect_session_messages(
                    SessionMessageQuery(session_id="target"), context=CONTEXT
                )
            elif operation == "snapshot_session_message_source":
                await app.snapshot_session_message_source("source", context=CONTEXT)
            else:
                session = await store.load("target")
                await app.apply_session_message_action(
                    SessionMessageActionRequest(
                        session_id="target",
                        session_instance_id=session.instance_id,
                        queue_id="queue",
                        idempotency_key="action",
                        expected_revision="0" * 64,
                        action="withdraw",
                    ),
                    context=CONTEXT,
                )

    asyncio.run(run())


def test_lifecycle_http_unsupported_conditions_are_unavailable_not_accepted():
    class UnsupportedStore(InMemorySessionStore):
        session_message_lifecycle_version = None

    app, _, _ = application(store_type=UnsupportedStore)
    response = client_for(app).post(
        "/api/sessions/target/messages",
        headers=HEADERS,
        json={
            "idempotency_key": "one",
            "content": CONTENT,
            "delivery_mode": "next_turn",
            "conditions": {"expires_at": "2099-01-01T00:00:00Z"},
        },
    )
    assert response.status_code == 503, response.text
    assert_private(response)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("entrance", ["sdk", "http"])
@pytest.mark.parametrize("replacement_receipt", [False, True])
def test_scoped_enqueue_fences_recreated_target_before_write_or_replay(
    tmp_path, monkeypatch, backend, entrance, replacement_receipt
):
    factory = (
        InMemorySessionStore
        if backend == "memory"
        else lambda: SQLiteSessionStore(tmp_path / "admission-fence.sqlite")
    )
    app, store, policy = application(store_type=factory)

    async def run():
        transport = ASGITransport(
            app=create_server(app, config=ServerConfig.protected(authenticate))
        )
        async with AsyncClient(transport=transport, base_url="http://test") as client:

            async def admit():
                if entrance == "sdk":
                    return await app.enqueue_session_message(enqueue_request(), context=CONTEXT)
                return await client.post(
                    "/api/sessions/target/messages",
                    headers=HEADERS,
                    json={
                        "idempotency_key": "message-1",
                        "content": CONTENT,
                        "delivery_mode": "next_turn",
                    },
                )

            # Same-incarnation exact retries remain successful and content-free
            # events are not duplicated. No delivery freshness is synthesized.
            first = await admit()
            before_retry = await store.load_events("target")
            retry = await admit()
            if entrance == "http":
                assert first.status_code == retry.status_code == 200
                assert sse_events(first) == sse_events(retry)
            else:
                assert retry.replayed and retry.message.queue_id == first.message.queue_id
            assert await store.load_events("target") == before_retry
            page = await store.inspect_session_messages(SessionMessageQuery(session_id="target"))
            assert len(page.records) == 1
            assert page.records[0].message.conditions == SessionMessageConditions()
            original = await store.load("target")
            entered, release = asyncio.Event(), asyncio.Event()
            engine_enqueue = app._session_engine.enqueue_session_message
            prepared = None

            async def blocked_enqueue(request, **kwargs):
                nonlocal prepared
                assert kwargs["expected_authorized_target_instance_id"] == original.instance_id
                prepared = request
                entered.set()
                await release.wait()
                return await engine_enqueue(request, **kwargs)

            monkeypatch.setattr(app._session_engine, "enqueue_session_message", blocked_enqueue)
            task = asyncio.create_task(admit())
            try:
                await asyncio.wait_for(entered.wait(), timeout=5)
                await store.delete_session("target")
                replacement = await store.create(
                    RunRequest(agent_name="assistant", session_id="target", messages=[]),
                    identity=SessionIdentity(provider_name="never-called", model="never-called"),
                )
                assert replacement.instance_id != original.instance_id
                assert not policy.authorize(
                    CONTEXT,
                    session_id="target",
                    session_instance_id=replacement.instance_id,
                    action="enqueue",
                )
                if replacement_receipt:
                    # Exact same body/actor on B must not make A's stale grant
                    # sufficient for the store's early idempotency return.
                    await store.enqueue_session_message(prepared)
                before = await store.inspect_session_messages(
                    SessionMessageQuery(session_id="target")
                )
                events = await store.load_events("target")
                transcript = await store.load_transcript("target")
            finally:
                release.set()
            if entrance == "sdk":
                with pytest.raises(SessionMessageConflict):
                    await task
            else:
                response = await task
                assert response.status_code == 409, response.text
                assert_private(response)
                assert CONTENT not in response.text
            assert (
                await store.inspect_session_messages(SessionMessageQuery(session_id="target"))
                == before
            )
            assert await store.load_events("target") == events
            assert await store.load_transcript("target") == transcript == []

    try:
        asyncio.run(run())
    finally:
        if backend == "sqlite":
            asyncio.run(store.close())


def test_unscoped_custom_enqueue_needs_no_fence_keyword_but_scoped_requires_v1():
    class OrdinaryStore(InMemorySessionStore):
        session_message_lifecycle_version = None

        async def enqueue_session_message(self, request):
            return await super().enqueue_session_message(request)

    app, _, _ = application(policy_enabled=False, store_type=OrdinaryStore)
    accepted = asyncio.run(app.enqueue_session_message(enqueue_request()))
    assert not accepted.replayed
    scoped, _, _ = application(store_type=OrdinaryStore)
    with pytest.raises(NotImplementedError):
        asyncio.run(scoped.enqueue_session_message(enqueue_request(), context=CONTEXT))
    response = client_for(scoped).post(
        "/api/sessions/target/messages",
        headers=HEADERS,
        json={"idempotency_key": "message-1", "content": CONTENT, "delivery_mode": "next_turn"},
    )
    assert response.status_code == 503, response.text
    assert_private(response)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("entrance", ["sdk", "http"])
@pytest.mark.parametrize("operation", ["inspect", "source"])
def test_protected_read_fences_recreated_session_before_content_access(
    tmp_path, monkeypatch, backend, entrance, operation
):
    factory = (
        InMemorySessionStore
        if backend == "memory"
        else lambda: SQLiteSessionStore(tmp_path / "read-fence.sqlite")
    )
    app, store, policy = application(store_type=factory)

    async def run():
        transport = ASGITransport(
            app=create_server(app, config=ServerConfig.protected(authenticate))
        )
        async with AsyncClient(transport=transport, base_url="http://test") as client:

            async def read():
                if entrance == "http":
                    if operation == "inspect":
                        return await client.get("/api/sessions/target/messages", headers=HEADERS)
                    return await client.post(
                        "/api/sessions/target/messages/source-snapshot",
                        headers=HEADERS,
                        json={"include_transcript_digest": True, "include_checkpoint_digest": True},
                    )
                if operation == "inspect":
                    return await app.inspect_session_messages(
                        SessionMessageQuery(session_id="target"), context=CONTEXT
                    )
                return await app.snapshot_session_message_source(
                    "target",
                    context=CONTEXT,
                    include_transcript_digest=True,
                    include_checkpoint_digest=True,
                )

            initial = await read()
            if entrance == "http":
                assert initial.status_code == 200, initial.text
            original = await store.load("target")
            entered, release = asyncio.Event(), asyncio.Event()
            authorize = app._session_message_coordinator._authorize

            async def blocked_authorize(*args, **kwargs):
                result = await authorize(*args, **kwargs)
                if result[0].instance_id == original.instance_id:
                    entered.set()
                    await release.wait()
                return result

            monkeypatch.setattr(app._session_message_coordinator, "_authorize", blocked_authorize)
            task = asyncio.create_task(read())
            try:
                await asyncio.wait_for(entered.wait(), timeout=5)
                await store.delete_session("target")
                replacement = await store.create(
                    RunRequest(
                        agent_name="assistant",
                        session_id="target",
                        messages=[Message.text("user", CONTENT)],
                    ),
                    identity=SessionIdentity(provider_name="never-called", model="never-called"),
                )
                await store.enqueue_session_message(enqueue_request())
                assert replacement.instance_id != original.instance_id
                assert not policy.authorize(
                    CONTEXT,
                    session_id="target",
                    session_instance_id=replacement.instance_id,
                    action=operation,
                )
                # Observe the real hydration/digest entrance, not just the
                # response postcheck. Positive reauthorization below proves
                # this spy is on the actual protected reader path.
                owner = store
                if operation == "source":
                    reader_name = "_session_message_source_unlocked"
                elif backend == "memory":
                    reader_name = "_inspect_session_message_unlocked"
                else:
                    import cayu.storage.sqlite as sqlite_module

                    owner = sqlite_module
                    reader_name = "_queued_session_message_from_row"
                reader = getattr(owner, reader_name)
                reads = []

                def observed_reader(*args, **kwargs):
                    reads.append("protected-content-read")
                    return reader(*args, **kwargs)

                monkeypatch.setattr(owner, reader_name, observed_reader)
                before_events = await store.load_events("target")
            finally:
                release.set()
            if entrance == "sdk":
                with pytest.raises(SessionMessageConflict):
                    await task
            else:
                denied = await task
                assert denied.status_code == 409, denied.text
                assert_private(denied)
                assert CONTENT not in denied.text
            assert reads == []
            assert await store.load_events("target") == before_events
            policy.grants.add(
                (CONTEXT.subject, CONTEXT.tenant, "target", replacement.instance_id, operation)
            )
            allowed = await read()
            assert reads
            if entrance == "http":
                assert allowed.status_code == 200, allowed.text
                assert_private(allowed)
                value = allowed.json()
                assert value["session_instance_id"] == replacement.instance_id
                if operation == "inspect":
                    assert value["records"][0]["message"]["content"] == CONTENT
            else:
                assert allowed.session_instance_id == replacement.instance_id
                if operation == "inspect":
                    assert allowed.records[0].message.content == CONTENT
            assert await store.load_events("target") == before_events

    try:
        asyncio.run(run())
    finally:
        if backend == "sqlite":
            asyncio.run(store.close())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("entrance", ["sdk", "http"])
@pytest.mark.parametrize("bad_terminal", ["wrong", "ambiguous", "missing"])
def test_corrupt_acceptance_pointer_can_be_inspected_quarantined_and_replayed(
    tmp_path, monkeypatch, backend, entrance, bad_terminal
):
    database = tmp_path / "pointer.sqlite"
    factory = InMemorySessionStore if backend == "memory" else lambda: SQLiteSessionStore(database)
    app, store, _ = application(store_type=factory)
    client = client_for(app)
    asyncio.run(app.enqueue_session_message(enqueue_request(), context=CONTEXT))
    broken = "missing-private-acceptance-pointer"
    if backend == "memory":
        store._queued_session_messages_by_idempotency["target"][
            "message-1"
        ].accepted_event_id = broken
    else:
        with sqlite3.connect(database) as connection:
            connection.execute(
                "UPDATE cayu_session_message_queue SET accepted_event_id = ? WHERE session_id = ?",
                (broken, "target"),
            )

    def inspect():
        if entrance == "sdk":
            return asyncio.run(
                app.inspect_session_messages(
                    SessionMessageQuery(session_id="target"), context=CONTEXT
                )
            ).model_dump(mode="json")
        response = client.get("/api/sessions/target/messages", headers=HEADERS)
        assert response.status_code == 200, response.text
        assert_private(response)
        return response.json()

    try:
        raw = asyncio.run(store.inspect_session_messages(SessionMessageQuery(session_id="target")))
        page = inspect()
        record = page["records"][0]
        assert record["message"] is None and record["validity"] == "unreadable"
        assert record["revision"] == raw.records[0].revision
        request = SessionMessageActionRequest(
            session_id="target",
            session_instance_id=page["session_instance_id"],
            queue_id=record["queue_id"],
            expected_revision=record["revision"],
            idempotency_key="quarantine-pointer",
            action="quarantine",
        )
        path = f"/api/sessions/target/messages/{record['queue_id']}/quarantine"
        body = {
            "session_instance_id": request.session_instance_id,
            "idempotency_key": request.idempotency_key,
            "expected_revision": request.expected_revision,
        }

        def action():
            if entrance == "sdk":
                return asyncio.run(
                    app.apply_session_message_action(request, context=CONTEXT)
                ).model_dump(mode="json")
            response = client.post(path, headers=HEADERS, json=body)
            assert response.status_code == 200, response.text
            assert_private(response)
            return response.json()

        first = action()
        assert first["replayed"] is False
        assert first["record"]["status"] == "quarantined"
        assert first["record"]["message"] is None
        assert first["record"]["validity"] == "unreadable"
        assert first["record"]["terminal_event_id"] == first["event"]["id"]
        events = asyncio.run(store.load_events("target"))
        retry = action()
        assert retry == {**first, "replayed": True}
        assert inspect()["records"][0] == first["record"]
        terminal_raw = asyncio.run(
            store.inspect_session_messages(SessionMessageQuery(session_id="target"))
        )
        assert first["record"]["revision"] == terminal_raw.records[0].revision
        assert terminal_raw.records[0].message.accepted_event_id == broken
        assert asyncio.run(store.load_events("target")) == events
        assert sum(event.type == EventType.SESSION_MESSAGE_QUARANTINED for event in events) == 1
        assert asyncio.run(store.load_transcript("target")) == []
        assert CONTENT not in json.dumps(first) and broken not in json.dumps(first)

        query_events = store.query_events
        terminal_id = terminal_raw.records[0].terminal_event_id

        async def invalid_terminal(query):
            records = await query_events(query)
            if query.event_id != terminal_id:
                return records
            if bad_terminal == "missing":
                return []
            if bad_terminal == "ambiguous":
                return [records[0], records[0].model_copy(deep=True)]
            event = records[0].event.model_copy(
                update={"payload": {**records[0].event.payload, "queue_id": "wrong-queue"}},
                deep=True,
            )
            return [records[0].model_copy(update={"event": event}, deep=True)]

        monkeypatch.setattr(store, "query_events", invalid_terminal)
        if entrance == "sdk":
            with pytest.raises(SessionMessageConflict):
                inspect()
            with pytest.raises(SessionMessageConflict):
                action()
        else:
            for denied in (
                client.get("/api/sessions/target/messages", headers=HEADERS),
                client.post(path, headers=HEADERS, json=body),
            ):
                assert denied.status_code == 409, denied.text
                assert_private(denied)
                assert CONTENT not in denied.text
        assert asyncio.run(store.load_events("target")) == events
    finally:
        if backend == "sqlite":
            asyncio.run(store.close())


@pytest.mark.parametrize("error_type", [OSError, SessionMessageConflict])
def test_acceptance_lookup_failure_is_not_reclassified_as_unreadable(monkeypatch, error_type):
    app, store, _ = application()
    asyncio.run(app.enqueue_session_message(enqueue_request(), context=CONTEXT))
    error = error_type()

    async def failed_lookup(query):
        raise error

    monkeypatch.setattr(store, "query_events", failed_lookup)
    with pytest.raises(error_type) as raised:
        asyncio.run(
            app.inspect_session_messages(SessionMessageQuery(session_id="target"), context=CONTEXT)
        )
    assert raised.value is error
    response = client_for(app).get("/api/sessions/target/messages", headers=HEADERS)
    assert response.status_code == (500 if error_type is OSError else 409)
    assert_private(response)
    assert CONTENT not in response.text


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("entrance", ["sdk", "http"])
@pytest.mark.parametrize("outcome", ["delivered", "withdrawn", "quarantined", "stale", "expired"])
def test_original_enqueue_replays_after_every_terminal_outcome(
    tmp_path, monkeypatch, backend, entrance, outcome
):
    factory = (
        InMemorySessionStore
        if backend == "memory"
        else lambda: SQLiteSessionStore(tmp_path / "terminal-enqueue-replay.sqlite")
    )
    app, store, _ = application(store_type=factory)
    client = client_for(app)
    conditions = SessionMessageConditions()
    if outcome == "expired":
        conditions = SessionMessageConditions(expires_at=datetime(2000, 1, 1, tzinfo=UTC))
    elif outcome == "stale":
        session = asyncio.run(store.load("target"))
        conditions = SessionMessageConditions(
            target=SessionMessageTarget(
                session_instance_id=session.instance_id,
                run_epoch=session.run_epoch,
                transcript_cursor=999,
            )
        )
    request = enqueue_request(conditions=conditions)
    body = request.model_dump(mode="json", exclude={"session_id", "requested_by"})

    def enqueue():
        if entrance == "sdk":
            return asyncio.run(app.enqueue_session_message(request, context=CONTEXT))
        return client.post("/api/sessions/target/messages", headers=HEADERS, json=body)

    try:
        first = enqueue()
        if entrance == "http":
            assert first.status_code == 200, first.text
        page = asyncio.run(
            app.inspect_session_messages(SessionMessageQuery(session_id="target"), context=CONTEXT)
        )
        if outcome in {"withdrawn", "quarantined"}:
            asyncio.run(
                app.apply_session_message_action(
                    SessionMessageActionRequest(
                        session_id="target",
                        session_instance_id=page.session_instance_id,
                        queue_id=page.records[0].queue_id,
                        expected_revision=page.records[0].revision,
                        idempotency_key="terminal",
                        action="withdraw" if outcome == "withdrawn" else "quarantine",
                    ),
                    context=CONTEXT,
                )
            )
        else:

            async def deliver():
                await store.update_status("target", SessionStatus.RUNNING)
                batch = await store.deliver_queued_session_messages("target", include_on_idle=True)
                await app._event_writer.fan_out_persisted(list(batch.events))

            asyncio.run(deliver())
        page = asyncio.run(
            app.inspect_session_messages(SessionMessageQuery(session_id="target"), context=CONTEXT)
        )
        assert page.records[0].status == outcome
        events = asyncio.run(store.load_events("target"))
        transcript = asyncio.run(store.load_transcript("target"))
        assert len(transcript) == (1 if outcome == "delivered" else 0)
        for _ in range(2):
            replay = enqueue()
            if entrance == "sdk":
                assert replay.replayed and replay.event == first.event
                assert replay.message.status == outcome
                assert replay.message == page.records[0].message
                assert replay.message.accepted_event_id == first.event.id
                if outcome == "delivered":
                    assert replay.message.delivered_event_id == page.records[0].terminal_event_id
            else:
                assert replay.status_code == 200, replay.text
                assert_private(replay)
                assert sse_events(replay) == sse_events(first)
            assert asyncio.run(store.load_events("target")) == events
            assert asyncio.run(store.load_transcript("target")) == transcript

        raw = asyncio.run(store.inspect_session_messages(SessionMessageQuery(session_id="target")))
        query_events = store.query_events
        # Both reference families remain exact. A corrupt delivery witness must
        # not become invisible merely because HTTP returns only the acceptance.
        ids = [raw.records[0].message.accepted_event_id]
        if outcome == "delivered":
            ids.append(raw.records[0].message.delivered_event_id)
        for invalid_id in ids:

            async def corrupted_lookup(query, invalid_id=invalid_id):
                records = await query_events(query)
                if query.event_id != invalid_id or not records:
                    return records
                event = records[0].event.model_copy(
                    update={"payload": {**records[0].event.payload, "queue_id": "wrong-queue"}},
                    deep=True,
                )
                return [records[0].model_copy(update={"event": event}, deep=True)]

            monkeypatch.setattr(store, "query_events", corrupted_lookup)
            if entrance == "sdk":
                with pytest.raises(SessionMessageConflict):
                    enqueue()
            else:
                denied = enqueue()
                assert denied.status_code == 409, denied.text
                assert_private(denied)
                assert CONTENT not in denied.text
            assert asyncio.run(store.load_events("target")) == events
            assert asyncio.run(store.load_transcript("target")) == transcript
    finally:
        if backend == "sqlite":
            asyncio.run(store.close())
