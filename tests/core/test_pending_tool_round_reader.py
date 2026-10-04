"""Fresh round reads preserve their source snapshot and validation context."""

import asyncio
from copy import deepcopy

import pytest
from tests.core.test_tool_round_publication import (
    _created_store,
    _lifecycle_events,
    _pending_round,
    _source_checkpoint,
)

import cayu._validation as validation
from cayu.runtime._tool_round_recovery import load_pending_tool_round
from cayu.sessions._assistant_tool_round_publication import StagedToolCallTerminal
from cayu.sessions.base import InMemorySessionStore
from cayu.vaults.redaction import SecretRedactor


class SnapshotStore:
    """Expose exact read snapshots, including invalid externally retained data."""

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.reads = []

    async def load_checkpoint(self, session_id):
        self.reads.append(session_id)
        return self.checkpoint


@pytest.mark.parametrize("history_size", [0, 1000])
def test_read_preserves_source_and_admits_checkpoint_once(monkeypatch, history_size):
    async def scenario():
        source = _source_checkpoint(_pending_round())
        source["history"] = [{"text": "retained"}] * history_size
        store = SnapshotStore(source)
        visits = []
        walk = validation._walk_bounded_durable_json

        def counted(value, field_name, **kwargs):
            if field_name == "checkpoint":
                visits.append(value)
            return walk(value, field_name, **kwargs)

        monkeypatch.setattr(validation, "_walk_bounded_durable_json", counted)
        checkpoint, pending = await load_pending_tool_round(store, "session")
        assert checkpoint is source
        assert store.reads == ["session"]
        assert len(visits) == 1 and visits[0] is source
        assert pending is not None
        pending.tool_calls[0].arguments["query"] = "changed result"
        assert source["pending_tool_round"]["tool_calls"][0]["arguments"] == {"query": "alpha"}

    asyncio.run(scenario())


def test_read_is_fresh_after_checkpoint_changes_and_removal():
    async def scenario():
        store = SnapshotStore(None)
        assert await load_pending_tool_round(store, "session") == (None, None)
        source = _source_checkpoint(_pending_round())
        store.checkpoint = source
        _, first = await load_pending_tool_round(store, "session")
        source["pending_tool_round"]["tool_calls"][0]["arguments"]["query"] = "updated"
        checkpoint, second = await load_pending_tool_round(store, "session")
        assert checkpoint is source
        assert first.tool_calls[0].arguments["query"] == "alpha"
        assert second.tool_calls[0].arguments["query"] == "updated"
        del source["pending_tool_round"]
        assert await load_pending_tool_round(store, "session") == (source, None)
        assert store.reads == ["session"] * 4

    asyncio.run(scenario())


@pytest.mark.parametrize("has_round", [False, True])
def test_read_rejects_invalid_unrelated_retained_data(has_round):
    async def scenario():
        source = _source_checkpoint(_pending_round()) if has_round else {}
        source["unrelated"] = {"invalid": object()}
        store = SnapshotStore(source)
        with pytest.raises(validation.DurableValueError):
            await load_pending_tool_round(store, "session")
        assert store.reads == ["session"]

    asyncio.run(scenario())


@pytest.mark.parametrize("consume", [False, True])
@pytest.mark.parametrize("invalid_schema", [False, True])
def test_read_rechecks_redactor_and_preserves_rejection_ownership(consume, invalid_schema):
    async def scenario():
        secret = "private-reader-secret-contents"
        source = _source_checkpoint(_pending_round())
        source["pending_tool_round"]["tool_calls"][0]["arguments"]["query"] = secret
        store = SnapshotStore(source)
        await load_pending_tool_round(store, "session", redactor=SecretRedactor())
        if invalid_schema:
            source["pending_tool_round"]["unexpected"] = True
            redactor = SecretRedactor()
        else:
            redactor = SecretRedactor().with_secret(secret)
        before = deepcopy(source)
        with pytest.raises(ValueError, match="cannot be executed") as caught:
            await load_pending_tool_round(
                store, "session", redactor=redactor, consume_on_rejection=consume
            )
        assert source == ({} if consume else before)
        assert store.reads == ["session"] * 2
        traceback = caught.value.__traceback__
        while traceback is not None:
            if "/src/cayu/" in traceback.tb_frame.f_code.co_filename:
                assert not [
                    name
                    for name, value in traceback.tb_frame.f_locals.items()
                    if secret in repr(value)
                ]
            traceback = traceback.tb_next

    asyncio.run(scenario())


def test_read_rechecks_runtime_session_provenance():
    async def scenario():
        session_id = "private-reader-session-identifier"
        _, session = await _created_store(InMemorySessionStore(), session_id=session_id)
        pending = _pending_round()
        terminal = _lifecycle_events(pending, session_id=session_id)[1]
        pending.staged_terminals = [StagedToolCallTerminal(tool_call_id="call-a", event=terminal)]
        store = SnapshotStore(_source_checkpoint(pending))
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
        assert store.reads == [session_id] * 3

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_type", [OSError, asyncio.CancelledError])
def test_read_propagates_store_failure_without_retry(failure_type):
    async def scenario():
        failure = failure_type("checkpoint unavailable")
        reads = []

        class FailingStore:
            async def load_checkpoint(self, session_id):
                reads.append(session_id)
                raise failure

        with pytest.raises(failure_type) as caught:
            await load_pending_tool_round(FailingStore(), "session")
        assert caught.value is failure
        assert reads == ["session"]

    asyncio.run(scenario())
