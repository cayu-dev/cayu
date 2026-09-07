from __future__ import annotations

import random
from decimal import Decimal

import pytest

from cayu.core import Event, EventType
from cayu.runtime._cost_accounting import CostAccountingReducer, cost_group_key
from cayu.runtime.costs import ModelPrice, PriceBook, estimate_session_cost, session_cost_totals
from cayu.runtime.sessions import EventQuery


def _pricing():
    return PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="openai",
                model="gpt-test",
                input_per_million=Decimal("1.25"),
                output_per_million=Decimal("3.75"),
                web_search_per_thousand=Decimal("10"),
            ),
        )
    )


def _events(seed, count=600):
    rng = random.Random(seed)
    events = []
    for index in range(count):
        hosted = rng.choice([True, False])
        payload = {"provider_name": "openai", "model": "gpt-test"}
        attempt = rng.choice([None, " malformed ", str(rng.randrange(40))])
        if attempt is not None:
            payload["model_attempt_id"] = attempt
        if hosted:
            payload.update(
                tool_type="web_search",
                call_id=f"call-{index}",
                status=rng.choice(["completed", "failed", "started"]),
            )
        elif rng.randrange(4):
            payload["usage_metrics"] = {
                "provider_name": "openai",
                "model": "gpt-test",
                "input_tokens": index,
                "output_tokens": 3,
                "total_tokens": index + 3,
            }
        events.append(
            Event(
                type=EventType.MODEL_HOSTED_TOOL_CALL if hosted else EventType.MODEL_COMPLETED,
                session_id="session",
                payload=payload,
            )
        )
    return events


def _reduce(events, *, additional_events=(), details=True):
    reducer = CostAccountingReducer(
        EventQuery(session_id="session"),
        _pricing(),
        currency="USD",
        details=details,
        additional_events=tuple(additional_events),
    )
    rows = sorted(
        enumerate(events, 1),
        key=lambda row: (
            cost_group_key(row[1]),
            row[1].type != EventType.MODEL_HOSTED_TOOL_CALL,
            row[0],
        ),
    )
    for sequence, event in rows:
        reducer.add(sequence, event)
    return reducer.snapshot()


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("pending", [0, 19])
def test_grouped_cost_reducer_matches_chronological_reference(seed, pending):
    events = _events(seed)
    durable = events[:-pending] if pending else events
    tail = events[-pending:] if pending else []
    actual = _reduce(durable, additional_events=tail)
    expected = estimate_session_cost(session_id="session", events=events, pricing=_pricing())
    assert actual.details == expected
    assert actual.totals == session_cost_totals(expected)
    assert actual.through_sequence == len(durable)
    totals_only = _reduce(durable, additional_events=tail, details=False)
    assert totals_only.details is None
    assert totals_only.totals == actual.totals


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_native_cost_snapshot_matches_reference(backend, tmp_path, request, monkeypatch):
    import asyncio
    from uuid import uuid4

    from cayu.core import Message
    from cayu.runtime.sessions import InMemorySessionStore, RunRequest, SessionIdentity
    from cayu.storage import PostgresSessionStore, SQLiteSessionStore
    from cayu.storage.migrations import SchemaMode

    async def run():
        if backend == "memory":
            store = InMemorySessionStore()
        elif backend == "sqlite":
            store = SQLiteSessionStore(tmp_path / "cost.sqlite")
        else:
            store = PostgresSessionStore(
                request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
            )
        try:
            sessions = [f"cost-{uuid4()}" for _ in range(2)]
            expected = []
            stored_events = {}
            for index, session_id in enumerate(sessions):
                await store.create(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "hi")],
                    ),
                    identity=SessionIdentity(provider_name="openai", model="gpt-test"),
                )
                events = [
                    event.model_copy(update={"session_id": session_id}) for event in _events(index)
                ]
                await store.append_events(session_id, events)
                stored_events[session_id] = events
                expected.append(
                    estimate_session_cost(session_id=session_id, events=events, pricing=_pricing())
                )

            async def forbidden(*args, **kwargs):
                raise AssertionError("Cost snapshot must not load/query complete histories.")

            monkeypatch.setattr(store, "load_events", forbidden)
            monkeypatch.setattr(store, "query_events", forbidden)
            for session_id, summary in zip(sessions, expected, strict=True):
                actual = await store.read_cost_accounting(
                    EventQuery(session_id=session_id), _pricing(), details=True
                )
                assert actual.details == summary
                assert actual.totals == session_cost_totals(summary)
                tail = tuple(
                    event.model_copy(update={"session_id": session_id}) for event in _events(11, 19)
                )
                changed_duplicate = stored_events[session_id][-1].model_copy(deep=True)
                changed_duplicate.payload["model_attempt_id"] = "different-attempt"
                with_pending = await store.read_cost_accounting(
                    EventQuery(session_id=session_id),
                    _pricing(),
                    details=True,
                    additional_events=(changed_duplicate, *tail, *tail),
                )
                assert with_pending.details == estimate_session_cost(
                    session_id=session_id,
                    events=[*stored_events[session_id], *tail],
                    pricing=_pricing(),
                )
            aggregate = await store.read_cost_accounting(
                EventQuery(session_ids=sessions), _pricing(), by_session=True
            )
            assert aggregate.details is None
            assert aggregate.totals.total_cost == sum(item.total_cost for item in expected)
            assert aggregate.totals.model_steps == sum(item.model_steps for item in expected)
            assert {item.session_id: item for item in aggregate.session_totals} == {
                item.session_id: session_cost_totals(item) for item in expected
            }
        finally:
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_incremental_cost_reprices_only_changed_groups_and_expires_windows(
    backend, tmp_path, request, monkeypatch
):
    import asyncio
    from datetime import UTC, datetime, timedelta

    from cayu.core import Message
    from cayu.runtime._cost_accounting_refresh import CostAccountingRead
    from cayu.runtime.sessions import InMemorySessionStore, RunRequest, SessionIdentity

    async def run():
        from uuid import uuid4

        from cayu.storage import PostgresSessionStore, SQLiteSessionStore
        from cayu.storage.migrations import SchemaMode

        if backend == "memory":
            store = InMemorySessionStore()
        elif backend == "sqlite":
            store = SQLiteSessionStore(tmp_path / "refresh.sqlite")
        else:
            store = PostgresSessionStore(
                request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
            )
        session_id = f"refresh-{uuid4()}"
        try:
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "hi")],
                ),
                identity=SessionIdentity(provider_name="openai", model="gpt-test"),
            )
            start = datetime(2026, 1, 1, tzinfo=UTC)
            events = [
                event.model_copy(
                    update={
                        "session_id": session_id,
                        "timestamp": start + timedelta(seconds=index % 30),
                    }
                )
                for index, event in enumerate(_events(7))
            ]
            await store.append_events(session_id, events)
            query = EventQuery(
                session_id=session_id, since=start, until=start + timedelta(seconds=10)
            )
            previous = await store.read_cost_accounting(query, _pricing())
            seen = []
            original_add = CostAccountingRead.add

            def observed(self, sequence, event):
                if self.incremental:
                    seen.append(event.id)
                return original_add(self, sequence, event)

            monkeypatch.setattr(CostAccountingRead, "add", observed)
            unchanged = await store.read_cost_accounting(query, _pricing(), previous=previous)
            assert unchanged.totals == previous.totals
            assert seen == []

            tail = [
                event.model_copy(
                    update={
                        "session_id": session_id,
                        "timestamp": start + timedelta(seconds=index % 30),
                    }
                )
                for index, event in enumerate(_events(8, 19))
            ]
            with_pending = await store.read_cost_accounting(
                query, _pricing(), previous=previous, additional_events=tuple(tail)
            )
            reference = estimate_session_cost(
                session_id=session_id,
                pricing=_pricing(),
                events=[
                    event
                    for event in [*events, *tail]
                    if query.since <= event.timestamp < query.until
                ],
            )
            assert with_pending.totals == session_cost_totals(reference)
            assert with_pending.durable_totals == previous.durable_totals
            await store.append_events(session_id, tail)
            events.extend(tail)
            for since, until in [(0, 10), (3, 15), (12, 23), (0, 30)]:
                query = EventQuery(
                    session_id=session_id,
                    since=start + timedelta(seconds=since),
                    until=start + timedelta(seconds=until),
                )
                refreshed = await store.read_cost_accounting(
                    query, _pricing(), previous=with_pending
                )
                expected = estimate_session_cost(
                    session_id=session_id,
                    pricing=_pricing(),
                    events=[
                        event for event in events if query.since <= event.timestamp < query.until
                    ],
                )
                assert refreshed.totals == session_cost_totals(expected)
                assert refreshed.durable_totals == refreshed.totals
                with_pending = refreshed
        finally:
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(run())


def test_cost_addition_and_removal_preserve_small_charges():
    from cayu.runtime.costs import add_cost_amounts

    total = add_cost_amounts(Decimal("1e100"), Decimal("0.1"))
    assert add_cost_amounts(total, Decimal("-1e100")) == Decimal("0.1")


def _priced_event(session_id, *, attempt=None):
    payload = {
        "usage_metrics": {
            "provider_name": "openai",
            "model": "gpt-test",
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
        }
    }
    if attempt is not None:
        payload["model_attempt_id"] = attempt
    return Event(session_id=session_id, type=EventType.MODEL_COMPLETED, payload=payload)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_cost_cursor_rejects_modified_totals_and_invalidates_after_deletion(
    backend, tmp_path, request
):
    import asyncio
    from uuid import uuid4

    from cayu.core import Message
    from cayu.runtime.sessions import (
        InMemorySessionStore,
        RunRequest,
        SessionIdentity,
        SessionStatus,
    )
    from cayu.storage import PostgresSessionStore, SQLiteSessionStore
    from cayu.storage.migrations import SchemaMode

    async def run():
        if backend == "memory":
            store = InMemorySessionStore()
            other = store
        elif backend == "sqlite":
            store = SQLiteSessionStore(tmp_path / "generation.sqlite")
            other = SQLiteSessionStore(tmp_path / "generation.sqlite")
        else:
            dsn = request.getfixturevalue("postgres_dsn")
            store = PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)
            other = PostgresSessionStore(dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            ids = [f"cursor-{uuid4()}" for _ in range(2)]
            for session_id in ids:
                await store.create(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "hi")],
                    ),
                    identity=SessionIdentity(provider_name="openai", model="gpt-test"),
                )
                await store.append_event(session_id, _priced_event(session_id))
                await store.update_status(session_id, SessionStatus.COMPLETED)
                await store.append_event(
                    session_id, Event(session_id=session_id, type=EventType.SESSION_COMPLETED)
                )
            query = EventQuery(session_ids=tuple(ids))
            initial = await store.read_cost_accounting(query, _pricing())
            assert initial.durable_totals is not None
            forged = initial.model_copy(deep=True)
            forged.durable_totals.total_cost = Decimal(0)
            repaired = await store.read_cost_accounting(query, _pricing(), previous=forged)
            assert repaired.totals == initial.totals
            if other is not store:
                reopened = await other.read_cost_accounting(query, _pricing(), previous=initial)
                assert reopened.totals == initial.totals
                if backend == "postgres":
                    assert reopened.cursor is None and initial.cursor is None
                else:
                    assert reopened.cursor.signature != initial.cursor.signature
            await other.delete_session(ids[1])
            refreshed = await store.read_cost_accounting(query, _pricing(), previous=initial)
            assert refreshed.totals.model_steps == 1
            assert refreshed.totals.total_cost == Decimal("0.00001125")
            if backend == "postgres":
                assert refreshed.cursor is None
            else:
                assert refreshed.cursor.generation > initial.cursor.generation
        finally:
            if other is not store and hasattr(other, "close"):
                await other.close()
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_cost_group_lookup_uses_index_and_keeps_full_attempt_identity(backend, tmp_path, request):
    import asyncio
    from uuid import uuid4

    from cayu.core import Message
    from cayu.runtime._cost_accounting import cost_accounting_query
    from cayu.runtime.sessions import RunRequest, SessionIdentity
    from cayu.storage import PostgresSessionStore, SQLiteSessionStore
    from cayu.storage import _session_store_sql as sql
    from cayu.storage import postgres as postgres_module
    from cayu.storage import sqlite as sqlite_module
    from cayu.storage._cost_accounting_sql import cost_group_lookup_statement
    from cayu.storage.migrations import SchemaMode

    async def run():
        store = (
            SQLiteSessionStore(tmp_path / "index.sqlite")
            if backend == "sqlite"
            else PostgresSessionStore(
                request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
            )
        )
        try:
            session_id = f"cost-plan-{uuid4()}"
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "hi")],
                ),
                identity=SessionIdentity(provider_name="openai", model="gpt-test"),
            )
            events = [
                _priced_event(session_id, attempt=f"attempt-{index}") for index in range(1000)
            ]
            prefix = "p" * 150
            events.extend(
                [
                    _priced_event(session_id, attempt=prefix + "a"),
                    _priced_event(session_id, attempt=prefix + "b"),
                ]
            )
            await store.append_events(session_id, events)
            dialect = (
                sqlite_module._SQL_DIALECT if backend == "sqlite" else postgres_module._SQL_DIALECT
            )
            plan = sql.build_accounting_event_query_sql(
                cost_accounting_query(EventQuery(session_id=session_id)), dialect=dialect
            )
            statement, params = cost_group_lookup_statement(
                columns="cayu_events.sequence",
                plan=plan,
                key=(session_id, True, prefix + "a"),
                postgres=backend == "postgres",
            )
            if backend == "sqlite":

                def inspect(connection):
                    plan_text = str(
                        [
                            tuple(row)
                            for row in connection.execute("EXPLAIN QUERY PLAN " + statement, params)
                        ]
                    )
                    rows = connection.execute(statement, params).fetchall()
                    return plan_text, len(rows)

                plan_text, count = await store._run_read(inspect)
            else:
                async with store._connection() as connection, connection.cursor() as cursor:
                    await cursor.execute("ANALYZE cayu_events")
                    await cursor.execute("EXPLAIN (FORMAT JSON) " + statement, params)
                    plan_text = str((await cursor.fetchone())[0])
                    await cursor.execute(statement, params)
                    count = len(await cursor.fetchall())
            assert "idx_cayu_events_cost_attempt" in plan_text
            assert count == 1
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_cost_snapshot_race_keeps_pending_out_of_next_durable_baseline(
    backend, tmp_path, request, monkeypatch
):
    import asyncio
    from uuid import uuid4

    from cayu.core import Message
    from cayu.runtime._cost_accounting_refresh import CostAccountingRead
    from cayu.runtime.sessions import RunRequest, SessionIdentity
    from cayu.storage import PostgresSessionStore, SQLiteSessionStore
    from cayu.storage.migrations import SchemaMode

    async def run():
        store = (
            SQLiteSessionStore(tmp_path / "cost-race.sqlite")
            if backend == "sqlite"
            else PostgresSessionStore(
                request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
            )
        )
        try:
            session_id = f"cost-race-{uuid4()}"
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "hi")],
                ),
                identity=SessionIdentity(provider_name="openai", model="gpt-test"),
            )
            await store.append_events(
                session_id,
                [_priced_event(session_id, attempt=f"attempt-{index}") for index in range(513)],
            )
            late = Event(
                session_id=session_id,
                type=EventType.MODEL_HOSTED_TOOL_CALL,
                payload={
                    "provider_name": "openai",
                    "model": "gpt-test",
                    "model_attempt_id": "attempt-0",
                    "tool_type": "web_search",
                    "call_id": "late-search",
                    "status": "completed",
                },
            )
            raced = False
            if backend == "sqlite":
                loop = asyncio.get_running_loop()
                original_add = CostAccountingRead.add

                def append_between_fetches(self, sequence, event):
                    nonlocal raced
                    if not raced:
                        raced = True
                        asyncio.run_coroutine_threadsafe(
                            store.append_event(session_id, late), loop
                        ).result(timeout=10)
                    return original_add(self, sequence, event)

                monkeypatch.setattr(CostAccountingRead, "add", append_between_fetches)
            else:
                from psycopg import AsyncServerCursor

                original_fetch = AsyncServerCursor.fetchmany

                async def append_between_fetches(self, size=0):
                    nonlocal raced
                    rows = await original_fetch(self, size)
                    if rows and self.name.startswith("cost_") and not raced:
                        raced = True
                        await store.append_event(session_id, late)
                    return rows

                monkeypatch.setattr(AsyncServerCursor, "fetchmany", append_between_fetches)
            query = EventQuery(session_id=session_id)
            first = await store.read_cost_accounting(query, _pricing(), additional_events=(late,))
            assert raced
            baseline = Decimal(513) * Decimal("0.00001125")
            assert first.durable_totals.total_cost == baseline
            assert first.totals.total_cost == baseline + Decimal("0.01")
            next_snapshot = await store.read_cost_accounting(query, _pricing(), previous=first)
            assert next_snapshot.totals == first.totals
            assert next_snapshot.durable_totals == next_snapshot.totals
            assert next_snapshot.through_sequence > first.through_sequence
            unchanged = await store.read_cost_accounting(query, _pricing(), previous=next_snapshot)
            assert unchanged.totals == next_snapshot.totals
            assert unchanged.through_sequence == next_snapshot.through_sequence
        finally:
            await store.close()

    asyncio.run(run())


def test_detailed_output_limit_stops_before_retaining_historical_line_items():
    from cayu.runtime._cost_accounting import CostAccountingOutputTooLarge

    reducer = CostAccountingReducer(
        EventQuery(session_id="session"),
        _pricing(),
        currency="USD",
        details=True,
        max_detail_bytes=1,
    )
    with pytest.raises(CostAccountingOutputTooLarge):
        reducer.add(1, _priced_event("session"))
    assert reducer._lines == []


def test_budget_memory_store_refresh_and_unsupported_store_fail_closed(monkeypatch):
    import asyncio
    from datetime import UTC, datetime, timedelta

    from cayu.runtime.budgets import BudgetStore, BudgetWindow, InMemoryBudgetStore

    class LegacyStore(BudgetStore):
        async def append_event(self, event):
            pass

        async def load_events_for_budget(self, **kwargs):
            raise AssertionError("Accounting must not fall back to full history.")

    async def run():
        store = InMemoryBudgetStore()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        events = [
            event.model_copy(update={"agent_name": "assistant", "timestamp": now})
            for event in _events(9, 100)
        ]
        for event in events:
            await store.append_event(event)
        monkeypatch.setattr(store, "load_events_for_budget", LegacyStore().load_events_for_budget)
        for scope, key in [("app", None), ("agent", "assistant")]:
            kwargs = dict(
                scope=scope,
                key=key,
                window=BudgetWindow.rolling(seconds=10),
                pricing=_pricing(),
                now=now + timedelta(seconds=1),
            )
            initial = await store.read_cost_for_budget(**kwargs)
            expected = estimate_session_cost(
                session_id="session", events=events, pricing=_pricing()
            )
            assert initial.totals.total_cost == expected.total_cost
            repeated = await store.read_cost_for_budget(**kwargs, previous=initial)
            assert repeated.totals == initial.totals
            expired = await store.read_cost_for_budget(
                **{**kwargs, "now": now + timedelta(seconds=11)}, previous=repeated
            )
            assert expired.totals.total_cost == 0
            assert expired.totals.model_steps == 0
            with pytest.raises(NotImplementedError, match="bounded cost"):
                await LegacyStore().read_cost_for_budget(**kwargs)

    asyncio.run(run())


def test_active_cost_refresh_serializes_and_drops_removed_scopes():
    import asyncio

    from cayu.core import Message
    from cayu.runtime import CayuApp, RunLimits
    from cayu.runtime._run_limits import SessionUsageTracker
    from cayu.runtime.sessions import InMemorySessionStore, RunRequest, SessionIdentity

    async def run():
        store = InMemorySessionStore()
        await store.create(
            RunRequest(
                session_id="active", agent_name="assistant", messages=[Message.text("user", "hi")]
            ),
            identity=SessionIdentity(provider_name="openai", model="gpt-test"),
        )
        first = _priced_event("active")
        await store.append_event("active", first)
        tracker = SessionUsageTracker(store, session_id="active")
        await tracker.retain_cost_scopes("request", {"limit"})
        entered = asyncio.Event()
        release = asyncio.Event()
        previous_reads = []

        async def read(previous):
            previous_reads.append(previous)
            result = await store.read_cost_accounting(
                EventQuery(session_id="active"), _pricing(), previous=previous
            )
            if len(previous_reads) == 1:
                entered.set()
                await release.wait()
            return result

        cold = asyncio.create_task(tracker.cost_snapshot("request", "limit", read))
        await entered.wait()
        tail = _priced_event("active")
        await store.append_event("active", tail)
        refresh = asyncio.create_task(tracker.cost_snapshot("request", "limit", read))
        await asyncio.sleep(0)
        assert len(previous_reads) == 1
        release.set()
        old, current = await asyncio.gather(cold, refresh)
        assert old.totals.model_steps == 1
        assert current.totals.model_steps == 2
        assert previous_reads[0] is None
        assert previous_reads[1].through_sequence == old.through_sequence
        assert (
            current.totals.total_cost
            == session_cost_totals(
                estimate_session_cost(session_id="active", events=[first, tail], pricing=_pricing())
            ).total_cost
        )
        app = CayuApp(session_store=store, enable_logging=False)
        session = await store.load("active")
        await app._run_limit_controller.evaluate_request_limits(
            session=session,
            agent_name="assistant",
            environment_name=None,
            limits=RunLimits(),
            budget_limits=(),
            run_started_at=0,
            usage_tracker=tracker,
        )
        assert not tracker._cost_snapshots
        await tracker.cost_snapshot("request", "limit", read)
        assert previous_reads[-1] is None
        assert not tracker._cost_snapshots

    asyncio.run(run())
