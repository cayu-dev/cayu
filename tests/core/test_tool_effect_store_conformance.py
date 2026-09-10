from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from tests.core.test_tool_effect_state import _event, _intent, _receipt, _terminal

from cayu.runtime import InMemorySessionStore
from cayu.runtime._tool_effect_state import ToolEffectConflict, ToolEffectStateOwner
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore


@asynccontextmanager
async def _stores(backend, path, dsn):
    opened = []
    memory = InMemorySessionStore()

    def open_store():
        if backend == "memory":
            return memory
        store = (
            SQLiteSessionStore(str(path))
            if backend == "sqlite"
            else PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)
        )
        opened.append(store)
        return store

    try:
        yield open_store
    finally:
        for store in reversed(opened):
            await store.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("dispatch_first", [False, True])
@pytest.mark.parametrize("lose_acknowledgement", [False, True])
def test_prepared_settlement_competes_with_dispatch(
    backend, dispatch_first, lose_acknowledgement, tmp_path, request, monkeypatch
):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        async with _stores(backend, tmp_path / "preparation.db", dsn) as open_store:
            store = open_store()
            intent = await _intent(store, session_id=f"prepared-{uuid4().hex}")
            owner = ToolEffectStateOwner(store)
            prepared = await owner.prepare(intent, run_epoch=0)
            event = _event(intent, failed=True)
            event.payload["result"]["structured"] = {
                "recovery_reason": "tool_effect_not_dispatched",
                "executed": False,
                "outcome_unknown": False,
            }
            original_publish = store.publish_session_operation
            lose_once = lose_acknowledgement

            async def publish(*args, **kwargs):
                nonlocal lose_once
                result = await original_publish(*args, **kwargs)
                if lose_once:
                    lose_once = False
                    raise OSError("prepared settlement acknowledgement lost")
                return result

            monkeypatch.setattr(store, "publish_session_operation", publish)

            async def settle():
                return await owner.transition(
                    prepared,
                    state="failed",
                    run_epoch=0,
                    terminal=_terminal(event),
                    events=(event,),
                )

            async def dispatch():
                return await ToolEffectStateOwner(open_store()).transition(
                    prepared, state="executing", run_epoch=0
                )

            # Both contenders hold the identical stale prepared revision. Order
            # their transaction consumption explicitly to prove both winners.
            winner = await (dispatch() if dispatch_first else settle())
            with pytest.raises(ToolEffectConflict):
                await (settle() if dispatch_first else dispatch())
            reconstructed = ToolEffectStateOwner(open_store())
            assert await reconstructed.load(intent) == winner
            if not dispatch_first:
                assert not lose_once
                assert winner.dispatch_id is None and winner.revision == 1
                assert await settle() == winner
                assert [e.id for e in await store.load_events(intent.session_id)] == [event.id]
            else:
                assert winner.dispatch_id is not None and winner.terminal is None
                assert await store.load_events(intent.session_id) == []

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("lose_acknowledgement", [False, True])
def test_receipt_atomic_selection_replay_and_reopen(
    backend, lose_acknowledgement, tmp_path, request, monkeypatch
):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        async with _stores(backend, tmp_path / "receipt.db", dsn) as open_store:
            store = open_store()
            intent = await _intent(store, session_id=f"effect-{uuid4().hex}")
            owner = ToolEffectStateOwner(store)
            executing = await owner.begin(intent, run_epoch=0)
            unknown = await owner.transition(executing, state="outcome_unknown", run_epoch=0)
            uncertainty_events = await store.load_events(intent.session_id)
            assert len(uncertainty_events) == 1
            assert uncertainty_events[0].type.value == "tool.effect.outcome_unknown"
            event = _event(intent)
            terminal = _terminal(event, _receipt(intent))
            publish = store.publish_session_operation
            lose_once = lose_acknowledgement

            async def commit_then_lose(*args, **kwargs):
                nonlocal lose_once
                result = await publish(*args, **kwargs)
                if lose_once:
                    lose_once = False
                    raise OSError("receipt commit acknowledgement lost")
                return result

            monkeypatch.setattr(store, "publish_session_operation", commit_then_lose)
            selected = await owner.transition(
                unknown,
                state="reconciled_completed",
                run_epoch=0,
                terminal=terminal,
                events=(event,),
            )
            assert not lose_once
            # A separate persistent-store instance must reconstruct the complete
            # selected receipt without relying on any process-private provenance.
            reconstructed = open_store()
            owner = ToolEffectStateOwner(reconstructed)
            assert await owner.load(intent) == selected
            assert (
                await owner.transition(
                    unknown,
                    state="reconciled_completed",
                    run_epoch=0,
                    terminal=terminal,
                    events=(event,),
                )
                == selected
            )
            with pytest.raises(ToolEffectConflict):
                await owner.transition(
                    unknown,
                    state="reconciled_completed",
                    run_epoch=0,
                    terminal=terminal.model_copy(
                        update={
                            "receipt": terminal.receipt.model_copy(update={"receipt_id": "other"})
                        }
                    ),
                    events=(event,),
                )
            assert await owner.load(intent) == selected
            assert [item.id for item in await reconstructed.load_events(intent.session_id)] == [
                uncertainty_events[0].id,
                event.id,
            ]

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("receipt_first", [False, True])
def test_normal_and_reconciled_store_selection_has_one_winner(
    backend, receipt_first, tmp_path, request
):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        async with _stores(backend, tmp_path / "race.db", dsn) as open_store:
            first, second = open_store(), open_store()
            intent = await _intent(first, session_id=f"effect-{uuid4().hex}")
            owner = ToolEffectStateOwner(first)
            executing = await owner.begin(intent, run_epoch=0)
            unknown = await owner.transition(executing, state="outcome_unknown", run_epoch=0)
            normal, recovered = (
                _event(intent, event_id="normal"),
                _event(intent, event_id="receipt"),
            )
            gate = asyncio.Event()

            async def select(store, event, receipt):
                await gate.wait()
                return await ToolEffectStateOwner(store).transition(
                    unknown,
                    state="reconciled_completed" if receipt else "completed",
                    run_epoch=0,
                    terminal=_terminal(event, receipt),
                    events=(event,),
                )

            contenders = [(first, normal, None), (second, recovered, _receipt(intent))]
            if receipt_first:
                contenders.reverse()
            tasks = [asyncio.create_task(select(*args)) for args in contenders]
            gate.set()
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            assert sum(isinstance(item, ToolEffectConflict) for item in outcomes) == 1
            events = await first.load_events(intent.session_id)
            assert len(events) == 2
            assert events[0].type.value == "tool.effect.outcome_unknown"
            selected = await owner.load(intent)
            assert selected.terminal.event_id == events[1].id
            assert await ToolEffectStateOwner(second).load(intent) == selected

    asyncio.run(scenario())
