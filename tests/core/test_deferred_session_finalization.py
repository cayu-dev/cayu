"""Cancellation cleanup callbacks remain valid after their handler exits."""

import asyncio

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    CayuConfig,
    EventType,
    Message,
    ModelStreamEvent,
    OperationsConfig,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
)
from cayu.sessions.cleanup import RecoveryCleanupPolicy


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_deferred_cancelled_session_finalization(tmp_path, monkeypatch, backend, request):
    path = tmp_path / "deferred.db"
    if backend == "postgres":
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        dsn = request.getfixturevalue("postgres_dsn")
        store = PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)
    else:
        store = SQLiteSessionStore(path)
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        config=CayuConfig(
            operations=OperationsConfig(
                recovery_cleanup_policy=RecoveryCleanupPolicy(
                    # The blocked callback below forces expiry independently of storage
                    # speed. Give the later database finalization a realistic budget.
                    step_timeout_seconds=1.0,
                    overall_timeout_seconds=5.0,
                )
            )
        ),
    )
    provider = ScriptedModelProvider(
        [[ModelStreamEvent.text_delta("ready"), ModelStreamEvent.completed({})]]
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker", model="test"))
    reached = asyncio.Event()
    release = asyncio.Event()
    close_entered = asyncio.Event()
    original_close = app._session_engine._close_durable_pending_tool_round_after_interrupt

    async def delayed_close(**kwargs):
        close_entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue
        async for event in original_close(**kwargs):
            yield event

    monkeypatch.setattr(
        app._session_engine, "_close_durable_pending_tool_round_after_interrupt", delayed_close
    )
    original_append = store.append_event

    async def pause_model(session_id, event):
        result = await original_append(session_id, event)
        if event.type == EventType.MODEL_TEXT_DELTA:
            reached.set()
            await asyncio.Event().wait()
        return result

    monkeypatch.setattr(store, "append_event", pause_model)

    async def run():
        async for _ in app.run(
            RunRequest(
                agent_name="worker", session_id="deferred", messages=[Message.text("user", "go")]
            )
        ):
            pass

    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(reached.wait(), 10)
        task.cancel("original caller")
        assert task.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        assert close_entered.is_set()
        assert task.cancelled() and task.cancelling() == 1
        assert app.recovery_cleanup_status().retained_tasks > 0
    finally:
        release.set()
        assert await app.drain_recovery_cleanups(timeout_s=10)
    assert app.recovery_cleanup_status().failed_after_timeout == 0
    events = await store.load_events("deferred")
    terminals = [e for e in events if e.type == EventType.SESSION_INTERRUPTED]
    assert len(terminals) == 1
    assert len(provider.requests) == 1
    await store.close()
    reopened = PostgresSessionStore(dsn) if backend == "postgres" else SQLiteSessionStore(path)
    assert await reopened.load_events("deferred") == events
    assert (await reopened.load("deferred")).status.value != "running"
    await reopened.close()
