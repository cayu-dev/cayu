"""Runtime readers reuse only fresh, owned checkpoint validation."""

from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest
from tests.core.test_pending_tool_round_reader import SnapshotStore
from tests.core.test_tool_round_publication import (
    _created_store,
    _lifecycle_events,
    _pending_round,
    _source_checkpoint,
)

import cayu._validation as validation
from cayu.runtime._checkpoint_store import (
    _RuntimeCheckpointSessionStore,
    runtime_checkpoint_session_store,
)
from cayu.sessions._assistant_tool_round_publication import StagedToolCallTerminal
from cayu.sessions._pending_tool_round_reader import load_pending_tool_round
from cayu.sessions.base import InMemorySessionStore, RunRequest, SessionIdentity
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY as VERSION,
)
from cayu.sessions.checkpoints import (
    CURRENT_CHECKPOINT_SCHEMA_VERSION as CURRENT,
)
from cayu.sessions.checkpoints import (
    CheckpointCompatibilityError,
    _DecodedRuntimeCheckpoint,
    runtime_checkpoint_writer_view,
)
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("size", [0, 1000])
@pytest.mark.parametrize("warm_cache", [False, True])
def test_native_runtime_round_read_admits_once_after_storage(
    backend, size, warm_cache, tmp_path, monkeypatch
):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "checkpoint.sqlite")
        )
        try:
            await store.create(
                RunRequest(agent_name="agent", session_id="session", messages=[]),
                identity=SessionIdentity(provider_name="provider", model="model"),
            )
            source = _source_checkpoint(_pending_round())
            source[VERSION] = CURRENT
            source["history"] = [{"text": "retained"} for _ in range(size)]
            source["probe"] = f"round-{backend}-{size}-{warm_cache}"
            await store.checkpoint("session", source)
            if warm_cache:
                await store.load_checkpoint("session")
            runtime = runtime_checkpoint_session_store(store)
            walk = validation._walk_bounded_durable_json
            visits = []
            reads = []
            original_load = store.load_checkpoint

            def counted(value, field_name, **kwargs):
                if type(value) is dict and "history" in value:
                    visits.append(value)
                return walk(value, field_name, **kwargs)

            async def load(session_id):
                reads.append(session_id)
                return await original_load(session_id)

            with monkeypatch.context() as patch:
                patch.setattr(validation, "_walk_bounded_durable_json", counted)
                patch.setattr(store, "load_checkpoint", load)
                checkpoint, pending = await load_pending_tool_round(runtime, "session")
            assert len(visits) == (2 if backend == "sqlite" and not warm_cache else 1)
            assert reads == ["session"]
            assert pending is not None and checkpoint == source
            pending.tool_calls[0].arguments["query"] = "changed round"
            assert checkpoint["pending_tool_round"]["tool_calls"][0]["arguments"] == {
                "query": "alpha"
            }
            checkpoint["history"].append({"text": "changed checkpoint"})
            assert await store.load_checkpoint("session") == source
            source["pending_tool_round"]["tool_calls"][0]["arguments"]["query"] = "fresh"
            await store.checkpoint("session", source)
            _, fresh = await load_pending_tool_round(runtime, "session")
            assert fresh.tool_calls[0].arguments["query"] == "fresh"
            source.pop("pending_tool_round")
            await store.checkpoint("session", source)
            assert (await load_pending_tool_round(runtime, "session"))[1] is None
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("version", [None, *range(1, CURRENT + 1)])
def test_runtime_round_read_migrates_without_mutating_custom_store(version):
    async def scenario():
        source = _source_checkpoint(_pending_round())
        if version is not None:
            source[VERSION] = version
        before = deepcopy(source)
        store = SnapshotStore(source)
        checkpoint, pending = await load_pending_tool_round(
            runtime_checkpoint_session_store(store), "session"
        )
        assert checkpoint[VERSION] == CURRENT and pending is not None
        assert source == before and checkpoint is not source
        assert store.reads == ["session"]

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid", ["future", "unrelated"])
def test_runtime_round_read_rejects_invalid_complete_checkpoint(invalid):
    async def scenario():
        source = _source_checkpoint(_pending_round())
        source[VERSION] = CURRENT
        if invalid == "future":
            source[VERSION] = CURRENT + 1
            error = CheckpointCompatibilityError
        else:
            source["unrelated"] = object()
            error = validation.DurableValueError
        store = SnapshotStore(source)
        with pytest.raises(error):
            await load_pending_tool_round(runtime_checkpoint_session_store(store), "session")
        assert store.reads == ["session"]

    asyncio.run(scenario())


@pytest.mark.parametrize("consume", [False, True])
@pytest.mark.parametrize("invalid_schema", [False, True])
def test_runtime_round_read_rechecks_context_and_clears_rejected_tracebacks(
    consume, invalid_schema
):
    async def scenario():
        secret = "private-composed-reader-secret"
        source = _source_checkpoint(_pending_round())
        source[VERSION] = CURRENT
        source["pending_tool_round"]["tool_calls"][0]["arguments"]["query"] = secret
        store = SnapshotStore(source)
        runtime = runtime_checkpoint_session_store(store)
        await load_pending_tool_round(runtime, "session", redactor=SecretRedactor())
        if invalid_schema:
            source["pending_tool_round"]["unexpected"] = True
            redactor = SecretRedactor()
        else:
            redactor = SecretRedactor().with_secret(secret)
        before = deepcopy(source)
        with pytest.raises(ValueError, match="cannot be executed") as caught:
            await load_pending_tool_round(
                runtime, "session", redactor=redactor, consume_on_rejection=consume
            )
        assert source == before and store.reads == ["session"] * 2
        traceback = caught.value.__traceback__
        while traceback is not None:
            if traceback.tb_frame.f_globals.get("__name__", "").startswith("cayu."):
                assert not any(
                    secret in repr(value) for value in traceback.tb_frame.f_locals.values()
                )
            traceback = traceback.tb_next

    asyncio.run(scenario())


def test_decoded_snapshot_owns_its_input_and_can_only_be_consumed_once():
    source = {VERSION: CURRENT, "nested": [1]}
    snapshot = _DecodedRuntimeCheckpoint(source, session_id="session")
    source["nested"].append(2)
    assert snapshot.take(session_id="session") == {VERSION: CURRENT, "nested": [1]}
    with pytest.raises(RuntimeError, match="already been consumed"):
        snapshot.take(session_id="session")
    wrong_session = _DecodedRuntimeCheckpoint(source, session_id="other")
    with pytest.raises(ValueError, match="another session"):
        wrong_session.take(session_id="session")
    with pytest.raises(RuntimeError, match="already been consumed"):
        wrong_session.take(session_id="other")


def test_runtime_round_read_rechecks_session_provenance():
    async def scenario():
        session_id = "private-composed-session"
        _, session = await _created_store(InMemorySessionStore(), session_id=session_id)
        pending = _pending_round()
        terminal = _lifecycle_events(pending, session_id=session_id)[1]
        pending.staged_terminals = [StagedToolCallTerminal(tool_call_id="call-a", event=terminal)]
        source = SnapshotStore(_source_checkpoint(pending))
        store = runtime_checkpoint_session_store(source)
        redactor = SecretRedactor().with_secret(session_id)
        _, loaded = await load_pending_tool_round(
            store, session_id, redactor=redactor, runtime_session=session
        )
        assert loaded.staged_terminals[0].event.session_id == session_id
        for context in (None, session.model_copy(update={"id": "different-session"})):
            with pytest.raises(ValueError, match="workload secret"):
                await load_pending_tool_round(
                    store, session_id, redactor=redactor, runtime_session=context
                )
        assert source.reads == [session_id] * 3

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["unvalidated", "wrong_session", "consumed"])
def test_custom_reader_cannot_return_unvalidated_or_reused_admission(kind):
    async def scenario():
        source = {VERSION: CURRENT, "history": [1]}
        admitted = _DecodedRuntimeCheckpoint(
            source, session_id="other" if kind == "wrong_session" else "session"
        )
        if kind == "consumed":
            admitted.take(session_id="session")

        class Store(SnapshotStore):
            async def _load_decoded_runtime_checkpoint(self, session_id):
                return source if kind == "unvalidated" else admitted

        error = {"unvalidated": TypeError, "wrong_session": ValueError, "consumed": RuntimeError}[
            kind
        ]
        with pytest.raises(error):
            await load_pending_tool_round(Store(source), "session")

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_type", [OSError, asyncio.CancelledError])
def test_runtime_round_read_propagates_load_failure_without_retry(failure_type):
    async def scenario():
        calls = []
        failure = failure_type("unavailable")

        class Store:
            async def load_checkpoint(self, session_id):
                calls.append(session_id)
                raise failure

        with pytest.raises(failure_type) as caught:
            await load_pending_tool_round(runtime_checkpoint_session_store(Store()), "session")
        assert caught.value is failure and calls == ["session"]

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["proxy", "subclass"])
def test_runtime_round_read_preserves_public_load_overrides(kind):
    async def scenario():
        source = SnapshotStore(_source_checkpoint(_pending_round()))
        calls = []

        class Override(_RuntimeCheckpointSessionStore):
            async def load_checkpoint(self, session_id):
                calls.append(session_id)
                checkpoint = await super().load_checkpoint(session_id)
                checkpoint["pending_tool_round"]["tool_calls"][0]["arguments"]["query"] = "policy"
                return checkpoint

        class Proxy:
            def __init__(self, store):
                self.store = store

            def __getattr__(self, name):
                return getattr(self.store, name)

            async def load_checkpoint(self, session_id):
                calls.append(session_id)
                checkpoint = await self.store.load_checkpoint(session_id)
                checkpoint["pending_tool_round"]["tool_calls"][0]["arguments"]["query"] = "policy"
                return checkpoint

        store = (
            Override(source)
            if kind == "subclass"
            else Proxy(runtime_checkpoint_session_store(source))
        )
        _, pending = await load_pending_tool_round(store, "session")
        assert pending.tool_calls[0].arguments["query"] == "policy"
        assert calls == ["session"] and source.reads == ["session"]

    asyncio.run(scenario())


@pytest.mark.parametrize("version", [1, CURRENT])
def test_writer_projection_reuses_decoded_document_and_stays_detached(version, monkeypatch):
    source = {VERSION: CURRENT, "history": [{"text": "retained"}] * 1000}
    visits = []
    walk = validation._walk_bounded_durable_json

    def counted(value, field_name, **kwargs):
        if type(value) is dict and "history" in value:
            visits.append(value)
        return walk(value, field_name, **kwargs)

    monkeypatch.setattr(validation, "_walk_bounded_durable_json", counted)
    projected = runtime_checkpoint_writer_view(source, writer_version=version, session_id="session")
    assert len(visits) == 1
    assert projected[VERSION] == version and source[VERSION] == CURRENT
    projected["history"][0]["text"] = "changed"
    assert source["history"][0]["text"] == "retained"


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("warm_cache", [False, True])
def test_unchanged_authority_snapshot_reuses_its_detached_decode(
    backend, warm_cache, tmp_path, monkeypatch
):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "snapshot.sqlite")
        )
        try:
            await store.create(
                RunRequest(agent_name="agent", session_id="session", messages=[]),
                identity=SessionIdentity(provider_name="provider", model="model"),
            )
            source = {VERSION: CURRENT, "history": [{"text": "retained"}] * 1000}
            source["probe"] = f"snapshot-{backend}-{warm_cache}"
            await store.checkpoint("session", source)
            before = await store.load("session")
            if warm_cache:
                await store.load_checkpoint("session")
            visits = []
            walk = validation._walk_bounded_durable_json

            def counted(value, field_name, **kwargs):
                if type(value) is dict and "history" in value:
                    visits.append(value)
                return walk(value, field_name, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(validation, "_walk_bounded_durable_json", counted)
                session, checkpoint = await runtime_checkpoint_session_store(
                    store
                ).load_session_checkpoint_snapshot("session")
            assert len(visits) == (3 if backend == "sqlite" and not warm_cache else 2)
            assert session == before and await store.load("session") == before
            checkpoint["history"][0]["text"] = "changed"
            assert await store.load_checkpoint("session") == source
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
