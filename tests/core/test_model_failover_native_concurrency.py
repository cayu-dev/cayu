"""Native transaction conformance complements public process-loss recovery."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import uuid4

import pytest
from tests.core.test_model_failover_stages import (
    _initial_admission,
    _prepare,
    _StageMemoryStore,
    _StageSQLiteStore,
)

from cayu import EventType
from cayu.sessions.base import SessionModelCompletionStageConflict


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_native_preparation_race_and_content_conflict(tmp_path, backend, request):
    session_id = f"preparation-race-{uuid4().hex}"
    if backend == "postgres":
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        dsn = request.getfixturevalue("postgres_dsn")

        class StagePostgresStore(PostgresSessionStore):
            model_failover_stage_version = 1
            invocation_lifecycle_command_version = 1

    def fresh_store():
        if backend == "postgres":
            return StagePostgresStore(dsn, schema_mode=SchemaMode.CREATE)
        if backend == "sqlite":
            return _StageSQLiteStore(tmp_path / "race.sqlite")
        return _StageMemoryStore()

    async def scenario():
        store = fresh_store()
        try:
            admission, attempt = await _initial_admission(store, session_id=session_id)
            ready = asyncio.Event()

            async def prepare():
                await ready.wait()
                return await _prepare(store, admission, attempt)

            contenders = [asyncio.create_task(prepare()) for _ in range(2)]
            ready.set()
            try:
                results = await asyncio.gather(*contenders)
            finally:
                for contender in contenders:
                    if not contender.done():
                        contender.cancel()
                await asyncio.gather(*contenders, return_exceptions=True)
            assert sum(result.dispatch_authorized for result in results) == 1
            assert sum(result.replayed for result in results) == 1
            assert results[0].stage == results[1].stage
            assert sum(len(result.prepared_events) for result in results) == 1
            stage = results[0].stage
            checkpoint = await store.load_checkpoint(session_id)
            persisted = await store.load_events(session_id)
            assert sum(event.type is EventType.MODEL_FAILOVER_SELECTED for event in persisted) == 1
            if backend != "memory":
                await store.close()
                store = fresh_store()
            changed = replace(
                admission,
                successor=admission.successor.model_copy(update={"request_fingerprint": "f" * 64}),
            )
            with pytest.raises(SessionModelCompletionStageConflict):
                await _prepare(store, changed, attempt)
            assert await store.load_checkpoint(session_id) == checkpoint
            assert await store.load_events(session_id) == persisted
            replay = await _prepare(store, admission, attempt)
            assert replay.replayed and not replay.dispatch_authorized
            assert replay.stage == stage and not replay.prepared_events
            assert (
                await store.load_model_completion_stage_dispatch(session_id, stage.stage_id) is None
            )
        finally:
            if backend != "memory":
                await store.close()

    asyncio.run(scenario())
