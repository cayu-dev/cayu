from __future__ import annotations

import asyncio
import random
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.budgets.usage import session_usage_summary
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.runtime._usage_accounting import SessionUsageCache, UsageAccountingReducer
from cayu.sessions.base import InMemorySessionStore, RunRequest
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import SessionIdentity
from cayu.storage import SQLiteSessionStore

_KINDS = (
    "completed",
    "completed_unmeasured",
    "completed_normalization_failed",
    "auxiliary",
    "auxiliary_unmeasured",
    "hosted",
    "tool_call",
    "unrelated",
)
_IDENTITIES = (("openai", "gpt-a"), ("openai", "gpt-b"), ("anthropic", "claude-c"), (None, None))


def _counter(rng: random.Random) -> int:
    # Individual counters stay signed 64-bit; a few maxima overflow the totals.
    return rng.choice([0, 1, rng.randrange(1, 10_000), MAX_DURABLE_JSON_INTEGER])


def _usage(rng: random.Random, provider: str | None, model: str | None) -> dict:
    return {
        "provider_name": provider,
        "model": model,
        "input_tokens": _counter(rng),
        "output_tokens": _counter(rng),
        "total_tokens": _counter(rng),
        "reasoning_output_tokens": _counter(rng),
        "cache": {
            "read_tokens": _counter(rng),
            "write_tokens": _counter(rng),
            "write_5m_tokens": _counter(rng),
            "write_1h_tokens": _counter(rng),
            "cached_input_tokens": _counter(rng),
            "uncached_input_tokens": _counter(rng),
        },
    }


def _event(rng: random.Random, session_id: str, index: int, kind: str | None = None) -> Event:
    kind = kind or rng.choice(_KINDS)
    provider, model = rng.choice(_IDENTITIES)
    # Several provider attempts of one model step share or reuse attempt ids.
    attempt = f"attempt-{rng.randrange(6)}"
    if kind == "tool_call":
        return Event(type=EventType.TOOL_CALL_STARTED, session_id=session_id)
    if kind == "unrelated":
        return Event(type=EventType.MODEL_STARTED, session_id=session_id)
    if kind == "hosted":
        return Event(
            type=EventType.MODEL_HOSTED_TOOL_CALL,
            session_id=session_id,
            payload={
                "provider_name": provider or "openai",
                "model": model or "gpt-a",
                "model_attempt_id": attempt,
                "tool_type": "web_search",
                "call_id": f"call-{index}",
                "status": rng.choice(["completed", "failed", "started"]),
            },
        )
    payload: dict = {"model_attempt_id": attempt}
    if kind in {"completed", "auxiliary"}:
        payload["usage_metrics"] = _usage(rng, provider, model)
    elif kind == "completed_normalization_failed":
        payload["usage_metrics"] = _usage(rng, provider, model)
        payload["usage_normalization_failed"] = True
    if kind.startswith("auxiliary"):
        payload["auxiliary_outcome"] = rng.choice(["completed", "failed", "outcome_unknown"])
        return Event(
            type=EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED,
            session_id=session_id,
            payload=payload,
        )
    return Event(type=EventType.MODEL_COMPLETED, session_id=session_id, payload=payload)


def _batches(seed: int, session_id: str, sizes: list[int]) -> list[list[Event]]:
    rng = random.Random(seed)
    index = 0
    batches = []
    for size in sizes:
        batch = []
        for _ in range(size):
            batch.append(_event(rng, session_id, index))
            index += 1
        batches.append(batch)
    return batches


async def _create(store, session_id: str) -> None:
    await store.create(
        RunRequest(
            session_id=session_id,
            agent_name="assistant",
            messages=[Message.text("user", "hi")],
        ),
        identity=SessionIdentity(provider_name="openai", model="gpt-a"),
    )


async def _full_read(store, session_id: str):
    # Grouped output bypasses the carried-forward cache and folds all history.
    return await store.read_usage_accounting(EventQuery(session_id=session_id), by_session=True)


def _hydrated_rows(monkeypatch) -> list[int]:
    rows: list[int] = []
    add_page = UsageAccountingReducer.add_page

    def observed(self, records):
        rows.append(len(records))
        return add_page(self, records)

    monkeypatch.setattr(UsageAccountingReducer, "add_page", observed)
    return rows


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def usage_store_pair(request, tmp_path):
    """Return a factory for a store and a second instance over the same data."""
    if request.param == "memory":

        def memory():
            store = InMemorySessionStore()
            return store, store

        return memory
    if request.param == "sqlite":
        path = tmp_path / "usage.sqlite"
        return lambda: (SQLiteSessionStore(path), SQLiteSessionStore(path))
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    dsn = request.getfixturevalue("postgres_dsn")
    return lambda: (
        PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE),
        PostgresSessionStore(dsn, schema_mode=SchemaMode.VALIDATE),
    )


async def _close(store, other) -> None:
    if other is not store and hasattr(other, "close"):
        await other.close()
    if hasattr(store, "close"):
        await store.close()


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    sizes=st.lists(st.integers(min_value=0, max_value=40), min_size=1, max_size=8),
    repeat=st.lists(st.booleans(), min_size=8, max_size=8),
)
def test_incremental_usage_matches_full_recompute_over_random_streams(seed, sizes, repeat):
    async def run():
        store = InMemorySessionStore()
        session_id = "property"
        await _create(store, session_id)
        history: list[Event] = []
        for batch, reread in zip(_batches(seed, session_id, sizes), repeat, strict=False):
            if batch:
                await store.append_events(session_id, batch)
            history.extend(batch)
            for _ in range(2 if reread else 1):
                incremental = await store.read_usage_accounting(EventQuery(session_id=session_id))
                full = await _full_read(store, session_id)
                assert incremental.summary == session_usage_summary(session_id, history)
                assert incremental.summary == full.summary
                assert incremental.through_sequence == full.through_sequence
                assert incremental.generation == full.generation

    asyncio.run(run())


@pytest.mark.parametrize("seed", [3, 17])
def test_incremental_usage_matches_full_recompute_on_every_store(usage_store_pair, seed):
    async def run():
        store, other = usage_store_pair()
        session_id = f"usage-{uuid4()}"
        try:
            await _create(store, session_id)
            history: list[Event] = []
            for number, batch in enumerate(_batches(seed, session_id, [300, 0, 1, 25, 7, 90])):
                # Alternate writers so the cache also sees another instance's appends.
                writer = other if number % 2 else store
                if batch:
                    await writer.append_events(session_id, batch)
                history.extend(batch)
                incremental = await store.read_usage_accounting(EventQuery(session_id=session_id))
                full = await _full_read(other, session_id)
                assert incremental.summary == session_usage_summary(session_id, history)
                assert incremental.summary == full.summary
                assert incremental.through_sequence == full.through_sequence
            assert incremental.summary.usage.input_tokens > MAX_DURABLE_JSON_INTEGER
        finally:
            await _close(store, other)

    asyncio.run(run())


def test_repeated_usage_read_hydrates_only_new_events(usage_store_pair, monkeypatch):
    rows = _hydrated_rows(monkeypatch)

    async def run():
        store, other = usage_store_pair()
        session_id = f"bounded-{uuid4()}"
        try:
            await _create(store, session_id)
            rng = random.Random(5)
            history = [_event(rng, session_id, index) for index in range(2000)]
            # A long unrelated tail must not be inspected again on every read.
            history.extend(_event(rng, session_id, 0, "unrelated") for _ in range(300))
            for offset in range(0, len(history), 500):
                await store.append_events(session_id, history[offset : offset + 500])

            cold = await store.read_usage_accounting(EventQuery(session_id=session_id))
            assert sum(rows) > 1000
            rows.clear()
            for _ in range(3):
                warm = await store.read_usage_accounting(EventQuery(session_id=session_id))
                assert warm == cold
            assert sum(rows) == 0

            tail = [_event(rng, session_id, index, "completed") for index in range(7)]
            tail.append(_event(rng, session_id, 0, "unrelated"))
            await other.append_events(session_id, tail)
            history.extend(tail)
            refreshed = await store.read_usage_accounting(EventQuery(session_id=session_id))
            assert sum(rows) == 7
            assert refreshed.summary == session_usage_summary(session_id, history)
            assert refreshed.through_sequence > cold.through_sequence
        finally:
            await _close(store, other)

    asyncio.run(run())


def test_sqlite_repeated_usage_read_work_is_independent_of_history(tmp_path):
    async def vm_steps(store, session_id: str) -> int:
        steps = 0

        def count() -> int:
            nonlocal steps
            steps += 1
            return 0

        await store._run_read(lambda connection: connection.set_progress_handler(count, 1))
        try:
            await store.read_usage_accounting(EventQuery(session_id=session_id))
        finally:
            await store._run_read(lambda connection: connection.set_progress_handler(None, 1))
        return steps

    async def run():
        store = SQLiteSessionStore(tmp_path / "work.sqlite")
        try:
            rng = random.Random(11)
            costs = {}
            for size in (200, 6000):
                session_id = f"history-{size}"
                await _create(store, session_id)
                events = [_event(rng, session_id, index) for index in range(size)]
                for offset in range(0, size, 1000):
                    await store.append_events(session_id, events[offset : offset + 1000])
                await store.read_usage_accounting(EventQuery(session_id=session_id))
                idle = await vm_steps(store, session_id)
                await store.append_events(
                    session_id,
                    [_event(rng, session_id, index, "completed") for index in range(20)],
                )
                after_tail = await vm_steps(store, session_id)
                costs[size] = (idle, after_tail)
            (small_idle, small_tail), (large_idle, large_tail) = costs[200], costs[6000]
            # The same work regardless of a 30x longer history.
            assert large_idle <= small_idle * 1.2
            assert large_tail <= small_tail * 1.2
            assert small_tail > small_idle
            # The measurement itself is sensitive to history: a cold read is not.
            store._session_usage_cache = SessionUsageCache()
            assert await vm_steps(store, "history-6000") > 100 * large_tail
        finally:
            await store.close()

    asyncio.run(run())


def test_deletion_advances_generation_and_invalidates_cached_usage(usage_store_pair, monkeypatch):
    rows = _hydrated_rows(monkeypatch)

    async def run():
        store, other = usage_store_pair()
        kept, removed = f"kept-{uuid4()}", f"removed-{uuid4()}"
        try:
            rng = random.Random(23)
            histories = {}
            for session_id in (kept, removed):
                await _create(store, session_id)
                histories[session_id] = [
                    _event(rng, session_id, index, "completed") for index in range(40)
                ]
                await store.append_events(session_id, histories[session_id])
            before = await store.read_usage_accounting(EventQuery(session_id=kept))
            await store.read_usage_accounting(EventQuery(session_id=kept))
            rows.clear()

            await other.delete_session(removed)
            after = await store.read_usage_accounting(EventQuery(session_id=kept))
            assert after.generation > before.generation
            assert sum(rows) == 40
            assert after.summary == before.summary

            # The session id can be reused; nothing from the deleted history survives.
            await _create(other, removed)
            reused = [_event(rng, removed, index, "completed") for index in range(3)]
            await other.append_events(removed, reused)
            assert (
                await store.read_usage_accounting(EventQuery(session_id=removed))
            ).summary == session_usage_summary(removed, reused)
        finally:
            await _close(store, other)

    asyncio.run(run())


def test_sqlite_event_pruning_invalidates_cached_usage(tmp_path):
    async def run():
        path = tmp_path / "prune.sqlite"
        store, pruner = SQLiteSessionStore(path), SQLiteSessionStore(path)
        try:
            await _create(store, "pruned")
            start = datetime(2026, 1, 1, tzinfo=UTC)
            rng = random.Random(29)
            events = [
                _event(rng, "pruned", index, "completed").model_copy(
                    update={"timestamp": start + timedelta(minutes=index)}
                )
                for index in range(20)
            ]
            await store.append_events("pruned", events)
            before = await store.read_usage_accounting(EventQuery(session_id="pruned"))
            assert before.summary == session_usage_summary("pruned", events)

            # Retention prunes only events whose side effects were delivered.
            for event in events:
                claim = await pruner.claim_persisted_event_side_effect(
                    session_id="pruned", event_id=event.id
                )
                assert claim is not None
                await pruner.mark_persisted_event_side_effect_delivered(claim)
            assert (
                await pruner.prune_events(before=start + timedelta(minutes=12), session_id="pruned")
                == 12
            )
            after = await store.read_usage_accounting(EventQuery(session_id="pruned"))
            assert after.generation > before.generation
            assert after.summary == session_usage_summary("pruned", events[12:])
        finally:
            await pruner.close()
            await store.close()

    asyncio.run(run())


def test_usage_cache_serves_only_plain_whole_session_reads():
    from cayu.runtime._usage_accounting import usage_accounting_query

    def scope(query, **flags):
        flags = {"by_session": False, "by_identity": False, **flags}
        return SessionUsageCache.session_scope(usage_accounting_query(query), **flags)

    assert scope(EventQuery(session_id="one")) == "one"
    assert scope(EventQuery(session_id="one", after_sequence=0)) == "one"
    assert scope(EventQuery(session_id="one", after_sequence=4)) is None
    assert scope(EventQuery(session_id="one", before_sequence=4)) is None
    assert scope(EventQuery(session_id="one", agent_name="assistant")) is None
    assert scope(EventQuery(session_id="one", since=datetime(2026, 1, 1, tzinfo=UTC))) is None
    assert scope(EventQuery(session_ids=("one",))) is None
    assert scope(EventQuery(session_id="one"), by_session=True) is None
    assert scope(EventQuery(session_id="one"), by_identity=True) is None


def test_usage_cache_is_bounded_and_forgets_older_generations():
    async def run():
        store = InMemorySessionStore()
        store._session_usage_cache = SessionUsageCache(max_sessions=2)
        for session_id in ("a", "b", "c"):
            await _create(store, session_id)
            await store.append_event(
                session_id, Event(type=EventType.TOOL_CALL_STARTED, session_id=session_id)
            )
            await store.read_usage_accounting(EventQuery(session_id=session_id))
        assert list(store._session_usage_cache._entries) == ["b", "c"]
        await store.delete_session("a")
        assert (await store.read_usage_accounting(EventQuery(session_id="b"))).summary.tool_calls
        assert list(store._session_usage_cache._entries) == ["b"]

    asyncio.run(run())
