from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest

from cayu.core import Event, EventType, Message
from cayu.runtime.costs import ModelPrice, PriceBook, estimate_causal_budget_cost
from cayu.runtime.sessions import (
    EventQuery,
    InMemorySessionStore,
    RunRequest,
    SessionAggregateFilter,
    SessionIdentity,
    UsageRollupQuery,
)
from cayu.runtime.usage import causal_budget_usage_summary, session_usage_summary
from cayu.storage import SQLiteSessionStore


def test_repeated_session_ids_do_not_duplicate_accounting():
    event = Event(
        type=EventType.MODEL_COMPLETED,
        session_id="one",
        payload={
            "usage_metrics": {
                "provider_name": "fake",
                "model": "fake",
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
            }
        },
    )
    ids = ["one", "empty", "one", "empty"]
    usage = causal_budget_usage_summary(causal_budget_id="budget", session_ids=ids, events=[event])
    cost = estimate_causal_budget_cost(
        causal_budget_id="budget",
        session_ids=ids,
        events=[event],
        pricing=PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="fake",
                    model="fake",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        ),
    )
    assert usage.session_ids == cost.session_ids == ["one", "empty"]
    assert usage.session_count == cost.session_count == 2
    assert len(usage.session_summaries) == len(cost.session_costs) == 2
    assert usage.model_steps == cost.model_steps == 1
    assert cost.total_cost == Decimal("0.000005")
    assert cost.total_cost == sum(row.total_cost for row in cost.session_costs)
    assert cost.unpriced_model_steps == sum(row.unpriced_model_steps for row in cost.session_costs)
    assert ids == ["one", "empty", "one", "empty"]


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_json_key_order_does_not_exhaust_raw_pricing_group_bound(
    backend, tmp_path, request, monkeypatch
):
    from cayu.storage import _postgres_aggregates, _sqlite_aggregates
    from cayu.storage.migrations import SchemaMode

    monkeypatch.setattr(_sqlite_aggregates, "MAX_USAGE_PRICING_RAW_CANDIDATES", 2)
    monkeypatch.setattr(_postgres_aggregates, "MAX_USAGE_PRICING_RAW_CANDIDATES", 2)

    async def run():
        if backend == "memory":
            store = InMemorySessionStore()
        elif backend == "sqlite":
            store = SQLiteSessionStore(tmp_path / "groups.sqlite")
        else:
            from cayu.storage import PostgresSessionStore

            store = PostgresSessionStore(
                request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
            )
        try:
            await store.create(
                RunRequest(
                    session_id="groups",
                    agent_name="canonical-pricing-groups",
                    messages=[Message.text("user", "hi")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            start = datetime(2026, 1, 1, tzinfo=UTC)
            fields = [("input_tokens", 3), ("output_tokens", 2), ("total_tokens", 5)]
            for index in range(3):
                metrics = dict(fields[index:] + fields[:index])
                await store.append_event(
                    "groups",
                    Event(
                        type=EventType.MODEL_COMPLETED,
                        session_id="groups",
                        timestamp=start,
                        payload={"usage_metrics": metrics},
                    ),
                )
            summary = await store.aggregate_usage(
                UsageRollupQuery(
                    sessions=SessionAggregateFilter(agent_name="canonical-pricing-groups"),
                    start_at=start,
                    end_at=start + timedelta(days=1),
                    include_pricing_inputs=True,
                    pricing_input_limit=1,
                )
            )
            assert summary.pricing_inputs_accuracy.kind == "exact"
            assert summary.pricing_input_group_count == 1
            assert summary.pricing_inputs[0].occurrences == 3
            assert summary.totals.usage.total_tokens == 15
        finally:
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(run())


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def accounting_store_factory(request, tmp_path):
    if request.param == "memory":
        return InMemorySessionStore
    if request.param == "sqlite":
        return lambda: SQLiteSessionStore(tmp_path / "accounting.sqlite")
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    dsn = request.getfixturevalue("postgres_dsn")
    return lambda: PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)


def test_usage_snapshot_matches_reference_and_incremental_watermark(
    accounting_store_factory, monkeypatch
):
    from cayu.runtime._usage_accounting import UsageAccountingReducer

    page_sizes = []
    add_page = UsageAccountingReducer.add_page

    def observed(self, records):
        page_sizes.append(len(records))
        return add_page(self, records)

    monkeypatch.setattr(UsageAccountingReducer, "add_page", observed)

    async def run():
        store = accounting_store_factory()
        session_id = f"accounting-{uuid4()}"
        try:
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "hi")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            events = [
                Event(
                    session_id=session_id,
                    agent_name="assistant",
                    type=EventType.MODEL_COMPLETED if index % 3 else EventType.TOOL_CALL_STARTED,
                    payload={
                        "usage_metrics": {
                            "provider_name": "fake",
                            "model": "fake",
                            "input_tokens": index,
                            "output_tokens": 2,
                            "total_tokens": index + 2,
                        }
                    },
                )
                for index in range(1025)
            ]
            for offset in range(0, len(events), 256):
                await store.append_events(session_id, events[offset : offset + 256])
            snapshot = await store.read_usage_accounting(
                EventQuery(session_id=session_id), by_session=True
            )
            assert snapshot.summary == session_usage_summary(session_id, events)
            assert snapshot.session_summaries == (snapshot.summary,)
            assert max(page_sizes) == 256 and sum(page_sizes) == 1025
            scoped = await store.read_usage_accounting(
                EventQuery(session_ids=(session_id, "absent"), agent_name="assistant"),
                by_session=True,
            )
            assert scoped.summary.model_dump(exclude={"session_id"}) == snapshot.summary.model_dump(
                exclude={"session_id"}
            )
            assert scoped.session_summaries == (snapshot.summary,)
            tail = Event(type=EventType.TOOL_CALL_STARTED, session_id=session_id)
            await store.append_event(session_id, tail)
            resumed = await store.read_usage_accounting(
                EventQuery(session_id=session_id, after_sequence=snapshot.through_sequence)
            )
            assert resumed.summary == session_usage_summary(session_id, [tail])
            assert resumed.through_sequence > snapshot.through_sequence
            empty = await store.read_usage_accounting(
                EventQuery(session_id=session_id, after_sequence=resumed.through_sequence)
            )
            assert empty.summary.model_steps == empty.summary.tool_calls == 0
            assert empty.through_sequence == resumed.through_sequence
        finally:
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(run())


def test_turn_usage_tracker_keeps_totals_without_retaining_events(monkeypatch):
    from cayu.runtime._run_limits import SessionUsageTracker

    async def run():
        store = InMemorySessionStore()
        session_id = "tracker"
        await store.create(
            RunRequest(
                session_id=session_id, agent_name="assistant", messages=[Message.text("user", "hi")]
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        await store.append_event(
            session_id, Event(session_id=session_id, type=EventType.TOOL_CALL_STARTED)
        )
        tracker = SessionUsageTracker(store, session_id=session_id)
        await tracker.mark_current_position()
        assert (await tracker.usage_summary()).tool_calls == 0
        for number in range(3):
            await store.append_event(
                session_id, Event(session_id=session_id, type=EventType.TOOL_CALL_STARTED)
            )
            assert (await tracker.usage_summary()).tool_calls == number + 1
            assert (await tracker.usage_summary()).tool_calls == number + 1
        assert not hasattr(tracker, "_events")

        original_read = store.read_usage_accounting

        async def delayed_read(*args, **kwargs):
            await asyncio.sleep(0)
            return await original_read(*args, **kwargs)

        monkeypatch.setattr(store, "read_usage_accounting", delayed_read)
        await store.append_event(
            session_id, Event(session_id=session_id, type=EventType.TOOL_CALL_STARTED)
        )
        snapshots = await asyncio.gather(*(tracker.usage_summary() for _ in range(5)))
        assert all(snapshot.tool_calls == 4 for snapshot in snapshots)

    asyncio.run(run())


def test_usage_read_has_fixed_working_set_for_one_hundred_thousand_events(monkeypatch):
    import gc
    import tracemalloc

    from cayu.runtime import BudgetLimit, CayuApp, RunLimits

    async def run():
        store = InMemorySessionStore()
        await store.create(
            RunRequest(
                session_id="large",
                agent_name="assistant",
                causal_budget_id="large-budget",
                messages=[Message.text("user", "hi")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        for offset in range(0, 100_000, 1000):
            await store.append_events(
                "large",
                [
                    Event(
                        id=f"usage-{index}",
                        session_id="large",
                        agent_name="assistant",
                        type=EventType.MODEL_COMPLETED,
                        payload={
                            "usage_metrics": {
                                "provider_name": "fake",
                                "model": "fake",
                                "input_tokens": 3,
                                "output_tokens": 2,
                                "total_tokens": 5,
                            }
                        },
                    )
                    for index in range(offset, offset + 1000)
                ],
            )

        async def forbidden(*args, **kwargs):
            raise AssertionError(
                "Usage accounting must not load complete history or query-all pages."
            )

        monkeypatch.setattr(store, "load_events", forbidden)
        monkeypatch.setattr(store, "query_events", forbidden)
        # Warm the parser before tracing working memory, excluding durable store
        # storage and the intentionally fixed-size public output from the input.
        await store.read_usage_accounting(EventQuery(session_id="large", before_sequence=1001))
        gc.collect()
        tracemalloc.start()
        try:
            snapshot = await store.read_usage_accounting(EventQuery(session_id="large"))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert snapshot.summary.model_steps == 100_000
        assert snapshot.summary.usage.total_tokens == 500_000
        assert peak < 8 * 1024 * 1024
        app = CayuApp(session_store=store, enable_logging=False)
        assert (await app.get_session_usage("large")).usage.total_tokens == 500_000
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="fake",
                    model="fake",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        )
        await store.read_cost_accounting(
            EventQuery(session_id="large", before_sequence=1001), pricing
        )
        limits = tuple(
            BudgetLimit(scope=scope, key=key, pricing=pricing, max_estimated_cost=Decimal("1"))
            for scope, key in (("app", None), ("agent", "assistant"), ("causal", "large-budget"))
        )
        session = await store.load("large")
        assert session is not None
        tracker = app._run_limit_controller.usage_tracker("large")
        gc.collect()
        tracemalloc.start()
        try:
            result = await app._run_limit_controller.evaluate_request_limits(
                session=session,
                agent_name="assistant",
                environment_name=None,
                limits=RunLimits(),
                budget_limits=limits,
                run_started_at=0,
                usage_tracker=tracker,
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert result.decision is None
        assert result.cost_summary is not None
        assert result.cost_summary.total_cost == Decimal("0.5")
        assert result.cost_summary.model_steps == 100_000
        assert result.cost_summary.priced_model_steps == 100_000
        assert result.cost_summary.unpriced_model_steps == 0
        assert "line_items" not in result.cost_summary.model_dump()
        assert len(tracker._cost_snapshots) == 3
        assert all(
            snapshot.totals.total_cost == Decimal("0.5")
            for snapshot in tracker._cost_snapshots.values()
        )
        assert peak < 8 * 1024 * 1024

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_usage_snapshot_excludes_append_committed_between_fetches(
    backend, tmp_path, request, monkeypatch
):
    from cayu.runtime._usage_accounting import UsageAccountingReducer
    from cayu.storage.migrations import SchemaMode

    async def run():
        if backend == "sqlite":
            store = SQLiteSessionStore(tmp_path / "snapshot.sqlite")
        else:
            from cayu.storage import PostgresSessionStore

            store = PostgresSessionStore(
                request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
            )
        session_id = f"snapshot-{uuid4()}"
        try:
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "hi")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            await store.append_events(
                session_id,
                [
                    Event(session_id=session_id, type=EventType.TOOL_CALL_STARTED)
                    for _ in range(513)
                ],
            )
            tail = Event(session_id=session_id, type=EventType.TOOL_CALL_STARTED)
            appended = False
            if backend == "sqlite":
                loop = asyncio.get_running_loop()
                original_add = UsageAccountingReducer.add_page

                def append_during_read(self, rows):
                    nonlocal appended
                    if not appended:
                        appended = True
                        asyncio.run_coroutine_threadsafe(
                            store.append_event(session_id, tail), loop
                        ).result(timeout=10)
                    return original_add(self, rows)

                monkeypatch.setattr(UsageAccountingReducer, "add_page", append_during_read)
            else:
                from psycopg import AsyncServerCursor

                original_fetch = AsyncServerCursor.fetchmany

                async def append_during_fetch(self, size=0):
                    nonlocal appended
                    rows = await original_fetch(self, size)
                    if self.name.startswith("usage_") and not appended:
                        appended = True
                        await store.append_event(session_id, tail)
                    return rows

                monkeypatch.setattr(AsyncServerCursor, "fetchmany", append_during_fetch)
            snapshot = await store.read_usage_accounting(EventQuery(session_id=session_id))
            assert appended
            assert snapshot.summary.tool_calls == 513
            resumed = await store.read_usage_accounting(
                EventQuery(session_id=session_id, after_sequence=snapshot.through_sequence)
            )
            assert resumed.summary.tool_calls == 1
        finally:
            await store.close()

    asyncio.run(run())


def test_notification_existence_keeps_scope_window_and_exact_identity_without_hydration(
    accounting_store_factory, monkeypatch
):
    async def run():
        store = accounting_store_factory()
        session_id = f"notification-{uuid4()}"
        now = datetime(2026, 1, 2, tzinfo=UTC)
        try:
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "hi")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            await store.append_events(
                session_id,
                [
                    Event(
                        session_id=session_id,
                        agent_name="assistant",
                        type=EventType.BUDGET_LIMIT_REACHED,
                        timestamp=now,
                        payload={"budget_limit_id": 123, "cost_summary": {"large": "x" * 100_000}},
                    ),
                    Event(
                        session_id=session_id,
                        agent_name="assistant",
                        type=EventType.BUDGET_LIMIT_REACHED,
                        timestamp=now - timedelta(days=1),
                        payload={"budget_limit_id": "123"},
                    ),
                    Event(
                        session_id=session_id,
                        agent_name="another",
                        type=EventType.BUDGET_LIMIT_REACHED,
                        timestamp=now,
                        payload={"budget_limit_id": "123"},
                    ),
                ],
            )

            async def forbidden(*args, **kwargs):
                raise AssertionError("Existence must not hydrate events.")

            monkeypatch.setattr(store, "query_events", forbidden)
            monkeypatch.setattr(store, "load_events", forbidden)
            query = EventQuery(
                session_id=session_id,
                event_type=EventType.BUDGET_LIMIT_REACHED,
                budget_limit_id="123",
                agent_name="assistant",
                since=now,
                until=now + timedelta(days=1),
            )
            assert await store.event_exists(query) is False
            await store.append_event(
                session_id,
                Event(
                    session_id=session_id,
                    agent_name="assistant",
                    type=EventType.BUDGET_LIMIT_REACHED,
                    timestamp=now,
                    payload={"budget_limit_id": "123"},
                ),
            )
            assert await store.event_exists(query) is True
            assert (
                await store.event_exists(query.model_copy(update={"budget_limit_id": "different"}))
                is False
            )
        finally:
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(run())


def test_run_limits_merge_inflight_usage_at_the_snapshot_boundary_without_history(monkeypatch):
    import time

    from cayu.runtime import CayuApp, RunLimits

    async def run():
        store = InMemorySessionStore()
        session_id = "inflight"
        session = await store.create(
            RunRequest(
                session_id=session_id, agent_name="assistant", messages=[Message.text("user", "hi")]
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )

        def completion():
            return Event(
                session_id=session_id,
                type=EventType.MODEL_COMPLETED,
                payload={
                    "usage_metrics": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
                },
            )

        first, inflight = completion(), completion()
        await store.append_event(session_id, first)
        app = CayuApp(session_store=store, enable_logging=False)
        controller = app._run_limit_controller
        tracker = controller.usage_tracker(session_id)
        original_read = store.read_usage_accounting
        appended = False

        async def racing_read(query, **kwargs):
            nonlocal appended
            snapshot = await original_read(query, **kwargs)
            if not appended:
                appended = True
                await store.append_event(session_id, inflight)
            return snapshot

        async def forbidden(*args, **kwargs):
            raise AssertionError("Run-limit accounting must not load complete history.")

        monkeypatch.setattr(store, "read_usage_accounting", racing_read)
        monkeypatch.setattr(store, "load_events", forbidden)
        monkeypatch.setattr(store, "query_events", forbidden)
        for maximum in (11, 11, 10):
            result = await controller.evaluate_request_limits(
                session=session,
                agent_name="assistant",
                environment_name=None,
                limits=RunLimits(scope="session", max_total_tokens=maximum),
                budget_limits=(),
                run_started_at=time.monotonic(),
                usage_tracker=tracker,
                additional_usage_events=[first, inflight, inflight],
            )
            assert result.usage_summary.usage.total_tokens == 10
            assert (result.decision is not None) is (maximum == 10)
            assert not hasattr(tracker, "_events")

    asyncio.run(run())
