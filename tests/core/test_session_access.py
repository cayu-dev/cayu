from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from cayu.sessions.access import (
    ScopedSessionAccess,
    SessionAccessDenied,
    SessionAccessRule,
    SessionAccessScope,
    SessionAccessSelector,
)
from cayu.sessions.base import InMemorySessionStore, RunRequest, SessionIdentity
from cayu.sessions.queries import SessionQuery
from cayu.storage.sqlite import SQLiteSessionStore


def rule(**labels):
    return SessionAccessRule(
        selectors=tuple(
            SessionAccessSelector(key=key, values=(value,)) for key, value in labels.items()
        )
    )


async def conformance(store):
    prefix = uuid4().hex
    for suffix, org, department in [
        ("a", "acme", "finance"),
        ("b", "other", "finance"),
        ("c", "acme", "legal"),
    ]:
        await store.create(
            RunRequest(
                agent_name="shared",
                messages=[],
                session_id=prefix + suffix,
                labels={"organization": org, "department": department},
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
    from cayu.events import Event, EventType
    from cayu.sessions.base import EventQuery

    for suffix in ("a", "b", "c"):
        await store.append_event(
            prefix + suffix,
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id=prefix + suffix,
                payload={
                    "usage_metrics": {
                        "provider_name": "fake",
                        "model": "fake",
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "total_tokens": 12,
                    }
                },
            ),
        )
    acme = rule(organization="acme")
    finance = rule(department="finance")
    current = SessionAccessScope(read=(acme,), update_labels=(acme,))

    async def resolve():
        return current

    handle = ScopedSessionAccess(store, admitted=current, resolve=resolve)
    assert (await handle.load(prefix + "a")).id == prefix + "a"
    for session_id in [prefix + "b", prefix + "absent"]:
        with pytest.raises(SessionAccessDenied, match="not authorized"):
            await handle.load(session_id)
    result = await handle.list_sessions(SessionQuery(q=prefix, limit=1, include_total_count=True))
    assert len(result.sessions) == 1 and result.total_count == 2 and result.next_cursor
    second = await handle.list_sessions(SessionQuery(q=prefix, cursor=result.next_cursor, limit=1))
    assert len(second.sessions) == 1
    assert second.sessions[0].id != result.sessions[0].id
    assert not (await handle.list_sessions(SessionQuery(labels={"organization": "other"}))).sessions
    assert not (
        await handle.list_sessions(
            SessionQuery(
                label_selectors=[{"key": "organization", "operator": "not_in", "values": ["acme"]}]
            )
        )
    ).sessions
    export = await handle.events(
        EventQuery(session_ids=tuple(prefix + suffix for suffix in ("a", "b", "c")))
    )
    assert {event.event.session_id for event in export} == {prefix + "a", prefix + "c"}
    assert not await handle.events(EventQuery(session_id=prefix + "b"))
    usage = await handle.usage(
        EventQuery(session_ids=tuple(prefix + suffix for suffix in ("a", "b", "c")))
    )
    assert usage.summary.model_steps == 2
    foreign_usage = await handle.usage(EventQuery(session_id=prefix + "b"))
    assert foreign_usage.summary.model_steps == 0
    from tests.core.test_cost_accounting import _pricing

    costs = await handle.costs(
        _pricing(), EventQuery(session_ids=tuple(prefix + suffix for suffix in ("a", "b", "c")))
    )
    assert costs.totals.model_steps == 2
    assert (
        await handle.costs(_pricing(), EventQuery(session_id=prefix + "b"))
    ).totals.model_steps == 0

    before = await handle.load(prefix + "a")
    for labels in [{"organization": "other"}, {"department": "finance"}]:
        with pytest.raises(SessionAccessDenied):
            await handle.update_labels(prefix + "a", labels)
        assert (await handle.load(prefix + "a")).labels == before.labels
    changed = await handle.update_labels(prefix + "a", {**before.labels, "note": "review"})
    assert changed.labels["note"] == "review"
    current = SessionAccessScope(read=(finance,))
    assert (await handle.load(prefix + "a")).id == prefix + "a"
    with pytest.raises(SessionAccessDenied):
        await handle.load(prefix + "c")
    with pytest.raises(SessionAccessDenied):
        await handle.update_labels(prefix + "a", before.labels)
    current = SessionAccessScope(read=(SessionAccessRule(allow_all=True),))
    with pytest.raises(SessionAccessDenied):
        await handle.load(prefix + "b")
    current = SessionAccessScope(read=(acme,), create=(acme,), modify=(acme,), delete=(acme,))
    writer = ScopedSessionAccess(store, admitted=current, resolve=resolve)
    own_id = prefix + "new"
    identity = SessionIdentity(provider_name="fake", model="fake")
    await writer.create(
        RunRequest(
            agent_name="shared", messages=[], session_id=own_id, labels={"organization": "acme"}
        ),
        identity=identity,
    )
    with pytest.raises(SessionAccessDenied):
        await writer.create(
            RunRequest(
                agent_name="shared",
                messages=[],
                session_id=prefix + "denied",
                labels={"organization": "other"},
            ),
            identity=identity,
        )
    assert await store.load(prefix + "denied") is None
    with pytest.raises(SessionAccessDenied):
        await writer.create(
            RunRequest(
                agent_name="shared",
                messages=[],
                session_id=prefix + "child",
                parent_session_id=prefix + "b",
                labels={"organization": "acme"},
            ),
            identity=identity,
        )
    assert await store.load(prefix + "child") is None
    await writer.update_metadata(own_id, {"note": "changed"})
    assert (await writer.load(own_id)).metadata["note"] == "changed"
    with pytest.raises(SessionAccessDenied):
        await writer.update_metadata(prefix + "b", {"note": "foreign"})
    for kind in ("events", "transcript"):
        assert not (await writer.read_records(own_id, kind=kind)).records
        with pytest.raises(SessionAccessDenied):
            await writer.read_records(prefix + "b", kind=kind)
    with pytest.raises(SessionAccessDenied):
        await writer.read_records(own_id, kind="checkpoint")
    with pytest.raises(SessionAccessDenied):
        await writer.delete_session(prefix + "b")
    await writer.delete_session(own_id)
    assert await store.load(own_id) is None
    from cayu._resource_access_binding import ResourceExecutionBinding
    from cayu.resource_access import (
        ResourceAccessPolicy,
        current_binding,
        encode_scope,
        execution_access,
        model_data_access,
    )
    from cayu.sessions.base import TranscriptQuery

    current = SessionAccessScope(
        read=(acme,), execute=(acme,), inspect_state=(acme,), modify=(acme,)
    )

    class Policy(ResourceAccessPolicy):
        authority = "session-native-test"

        async def resolve(self, subject):
            return current

    policy = Policy()
    binding = ResourceExecutionBinding(
        authority=policy.authority, subject="alice", admitted_json=encode_scope(current)
    )
    foreign_session = await store.load(prefix + "b")
    async with execution_access(binding, policy, {"organization": "acme"}), model_data_access():
        assert await store.load(prefix + "b") is None
        result = await store.list_sessions(SessionQuery(q=prefix, include_total_count=True))
        assert result.total_count == 2
        for operation in (
            store.summarize_events,
            store.summarize_outcome,
            store.load_events,
            store.load_transcript,
            store.load_checkpoint,
        ):
            with pytest.raises(SessionAccessDenied):
                await operation(prefix + "b")
            await operation(prefix + "a")
        with pytest.raises(SessionAccessDenied):
            await store.query_transcript(TranscriptQuery(session_id=prefix + "b"))
        assert not (await store.query_transcript(TranscriptQuery(session_id=prefix + "a"))).records
        from cayu.sessions.base import EnqueueSessionMessageRequest

        for foreign_id in (prefix + "b", prefix + "missing"):
            with pytest.raises(SessionAccessDenied):
                await store.snapshot_session_message_source(foreign_id)
            with pytest.raises(SessionAccessDenied):
                await store.enqueue_session_message(
                    EnqueueSessionMessageRequest(
                        session_id=foreign_id,
                        idempotency_key="foreign",
                        content="no",
                        delivery_mode="next_turn",
                    )
                )
            with pytest.raises(SessionAccessDenied):
                await store.update_metadata(foreign_id, {"note": "forbidden"})
        from types import SimpleNamespace

        from tests.core.test_peer_content import _delivery_request

        # Peer writes authorize both source and target before replay or mutation.
        own = await store.load(prefix + "a")
        # Operator fixture data is retained before the scoped block below.
        for source, target in ((own, foreign_session), (foreign_session, own)):
            peer = _delivery_request(
                source=source,
                target=target,
                sender=SimpleNamespace(participant_id="sender", incarnation="sender-v1"),
                consumer=SimpleNamespace(participant_id="receiver", incarnation="receiver-v1"),
            )
            with pytest.raises(SessionAccessDenied):
                await store.append_peer_content(peer)
        own_source = await store.snapshot_session_message_source(prefix + "a")
        assert own_source.session_id == prefix + "a"
        with pytest.raises(NotImplementedError, match="operator"):
            await store.load_state(prefix + "b")
        # Recipient planning remains an explicitly authenticated operator
        # surface, including methods inherited from a selection-fence mixin.
        # Denial must happen before interpreting caller data or reading records.
        for operation in (
            store.capture_recipient_continuation,
            store.read_context_view_selection_decision,
        ):
            with pytest.raises(NotImplementedError, match="operator"):
                await operation(None)

    assert current_binding() is None
    shared = SessionAccessRule(allow_all=True)
    current = SessionAccessScope(
        read=(shared,),
        update_labels=(shared,),
        relabel=(shared,),
        protected_label_keys=("organization",),
    )
    administrator = ScopedSessionAccess(store, admitted=current, resolve=resolve)
    await administrator.update_labels(prefix + "a", {"organization": "other"})
    audit = await administrator.read_records(prefix + "a", kind="access_audit")
    assert len(audit.records) == 1
    assert audit.records[0]["old_labels"]["organization"] == "acme"
    assert audit.records[0]["new_labels"] == {"organization": "other"}
    current = SessionAccessScope()
    assert not (await handle.list_sessions()).sessions
    with pytest.raises(SessionAccessDenied):
        await handle.load(prefix + "a")


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_session_access_conformance(backend, tmp_path):
    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "access.db")
        )
        try:
            await conformance(store)
        finally:
            if not isinstance(store, InMemorySessionStore):
                await store.close()

    asyncio.run(run())


def test_postgres_session_access_conformance(postgres_dsn):
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async def run():
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await conformance(store)
        finally:
            if not isinstance(store, InMemorySessionStore):
                await store.close()

    asyncio.run(run())


def test_scope_values_are_detached_and_explicit():
    from dataclasses import FrozenInstanceError

    values = ["acme"]
    selector = SessionAccessSelector(key="organization", values=values)
    values.append("other")
    assert selector.values == ("acme",)
    with pytest.raises(FrozenInstanceError):
        selector.key = "owner"
    with pytest.raises(ValueError):
        SessionAccessRule()
    with pytest.raises(ValueError):
        SessionAccessRule(selectors=(selector,), allow_all=True)
    with pytest.raises(TypeError):
        SessionAccessScope(protected_label_keys="organization")
    with pytest.raises(ValueError):
        SessionAccessSelector(key="organization", values=("a",) * 51)


def test_resolver_outage_and_unattested_subclass_fail_closed():
    async def run():
        scope = SessionAccessScope(read=(SessionAccessRule(allow_all=True),))
        calls = []

        class UnattestedStore(InMemorySessionStore):
            async def load(self, session_id):
                calls.append(session_id)

        async def outage():
            raise RuntimeError("policy unavailable")

        with pytest.raises(NotImplementedError):
            ScopedSessionAccess(UnattestedStore(), admitted=scope, resolve=outage)
        handle = ScopedSessionAccess(InMemorySessionStore(), admitted=scope, resolve=outage)
        with pytest.raises(RuntimeError, match="policy unavailable"):
            await handle.load("anything")
        assert calls == []
        for method in (
            "run",
            "resume",
            "load_checkpoint",
            "load_events",
        ):
            assert not hasattr(handle, method)

    asyncio.run(run())


def test_alternatives_do_not_escape_admitted_scope():
    async def run():
        store = InMemorySessionStore()
        maximum = SessionAccessScope(read=(rule(organization="acme"),))
        current = SessionAccessScope(read=(rule(owner="alice"), rule(department="finance")))

        async def resolve():
            return current

        handle = ScopedSessionAccess(store, admitted=maximum, resolve=resolve)
        for name, labels in [
            ("private", {"organization": "acme", "owner": "alice"}),
            ("group", {"organization": "acme", "department": "finance"}),
            ("excluded", {"organization": "other", "owner": "alice"}),
            ("neither", {"organization": "acme", "owner": "bob"}),
        ]:
            await store.create(
                RunRequest(session_id=name, agent_name="shared", messages=[], labels=labels),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
        result = await handle.list_sessions(SessionQuery(include_total_count=True))
        assert {session.id for session in result.sessions} == {"private", "group"}
        assert result.total_count == 2

    asyncio.run(run())


@pytest.mark.parametrize("operation", ["labels", "metadata", "delete"])
def test_sqlite_relabel_race_checks_inside_write_transaction(tmp_path, operation):
    import sqlite3
    import threading

    async def run():
        path = tmp_path / "race.db"
        store = SQLiteSessionStore(path)
        await store.create(
            RunRequest(
                session_id="session",
                agent_name="shared",
                messages=[],
                labels={"organization": "acme"},
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        scope = SessionAccessScope(
            read=(rule(organization="acme"),),
            update_labels=(rule(organization="acme"),),
            modify=(rule(organization="acme"),),
            delete=(rule(organization="acme"),),
        )

        async def resolve():
            return scope

        handle = ScopedSessionAccess(store, admitted=scope, resolve=resolve)
        competitor = sqlite3.connect(path, check_same_thread=False)
        competitor.execute("BEGIN IMMEDIATE")
        competitor.execute(
            "UPDATE cayu_session_labels SET value = 'other' WHERE session_id = 'session' AND key = 'organization'"
        )
        write_started = threading.Event()
        failures = []

        def trace(sql):
            if sql == "BEGIN IMMEDIATE":
                write_started.set()

        store._connection.set_trace_callback(trace)

        def commit_competitor():
            if not write_started.wait(5):
                failures.append("scoped mutation did not acquire a write transaction")
            competitor.commit()

        thread = threading.Thread(target=commit_competitor)
        thread.start()
        try:
            with pytest.raises(SessionAccessDenied):
                if operation == "labels":
                    await handle.update_labels(
                        "session", {"organization": "acme", "note": "stale-write"}
                    )
                elif operation == "metadata":
                    await handle.update_metadata("session", {"note": "stale-write"})
                else:
                    await handle.delete_session("session")
            assert (await store.load("session")).labels == {"organization": "other"}
        finally:
            thread.join(5)
            competitor.close()
            await store.close()
        assert not thread.is_alive() and not failures

    asyncio.run(run())


def test_access_scope_bounds_fit_organization_sized_policies() -> None:
    from cayu.sessions.access import SessionAccessRule, SessionAccessScope, SessionAccessSelector

    many_values = SessionAccessSelector(key="team", values=tuple(f"t{i}" for i in range(500)))
    with pytest.raises(ValueError, match="at most 500 values"):
        SessionAccessSelector(key="team", values=tuple(f"t{i}" for i in range(501)))
    rules = tuple(
        SessionAccessRule(selectors=(SessionAccessSelector(key="tenant", values=(f"x{i}",)),))
        for i in range(64)
    )
    scope = SessionAccessScope(read=rules, protected_label_keys=tuple(f"k{i}" for i in range(200)))
    assert len(scope.read) == 64
    assert SessionAccessRule(selectors=(many_values,)).matches({"team": "t499"})
    with pytest.raises(ValueError, match="at most 64 SessionAccessRule"):
        SessionAccessScope(read=(*rules, rules[0]))
