from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from tests.core.test_explicit_session_compaction import (
    UsageCompactionProvider,
    _create_profiled_session,
)
from tests.core.test_targeted_tool_grants import _codec

from cayu.core import AgentSpec, EventType, Message
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime import (
    CayuApp,
    CheckpointCompactionContextPolicy,
    CompactSessionRequest,
    InMemorySessionStore,
    ModelCompactor,
    RunRequest,
    SessionIdentity,
    SessionStatus,
)
from cayu.storage import PostgresSessionStore, SQLiteSessionStore
from cayu.storage.migrations import SchemaMode


class AckProvider(UsageCompactionProvider):
    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="tests:compaction-ack", behavior_version="1", implementation_version="1"
        )


def _store(backend, tmp_path, request):
    if backend == "memory":
        return InMemorySessionStore(public_authority_alias_codec=_codec())
    if backend == "sqlite":
        return SQLiteSessionStore(tmp_path / "ack.sqlite", public_authority_alias_codec=_codec())
    return PostgresSessionStore(
        request.getfixturevalue("postgres_dsn"),
        schema_mode=SchemaMode.CREATE,
        public_authority_alias_codec=_codec(),
    )


def _app(store, provider):
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(provider=provider, model="summary-model"),
            max_user_turns=1,
        ),
    )
    return app


async def _setup(store):
    provider = AckProvider()
    app = _app(store, provider)
    session_id = f"ack-{uuid4()}"
    session = await _create_profiled_session(
        app,
        store,
        RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    transcript = [
        Message.text("user", "old"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current"),
    ]
    await store.append_transcript_messages(session.id, transcript)
    session = await store.update_status(session.id, SessionStatus.COMPLETED)
    return (
        app,
        provider,
        CompactSessionRequest(
            session_id=session.id,
            idempotency_key="compact",
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=len(transcript),
        ),
    )


async def _operation(store, command):
    terminal = await store.load_session_operation(command.session_id, command.idempotency_key)
    if terminal is not None:
        return terminal
    checkpoint = await store.load_checkpoint(command.session_id)
    return (
        (checkpoint or {})
        .get("session_operations", {})
        .get("records", {})
        .get(command.idempotency_key)
    )


@asynccontextmanager
async def _fault(store, event_type, *, committed=True, after_commit=None, error=None):
    method = "publish_session_operation_guarded_with_store_time"
    original = getattr(store, method)
    hits = 0

    async def publish(*args, **kwargs):
        nonlocal hits
        if hits or not any(event.type == event_type for event in kwargs["events"]):
            return await original(*args, **kwargs)
        hits += 1
        if committed:
            await original(*args, **kwargs)
            if after_commit is not None:
                await after_commit()
        raise error if error is not None else ConnectionError("lost publication acknowledgement")

    setattr(store, method, publish)
    try:
        yield
        assert hits == 1
    finally:
        delattr(store, method)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize(
    "boundary", [EventType.CONTEXT_COMPACTION_STARTED, EventType.SESSION_CHECKPOINTED]
)
def test_committed_ack_loss_recovers_same_request_and_replays_once(
    backend, boundary, tmp_path, request
):
    async def run():
        store = _store(backend, tmp_path, request)
        try:
            app, provider, command = await _setup(store)
            async with _fault(store, boundary):
                first = [event async for event in app.compact_session(command)]
            assert first[-1].type == EventType.SESSION_CHECKPOINTED
            assert provider.calls == 1
            operation = await _operation(store, command)
            assert operation["status"] == "completed"
            if backend != "memory":
                await store.close()
                store = _store(backend, tmp_path, request)
            restarted_provider = AckProvider()
            restarted_app = _app(store, restarted_provider)
            replay = [event async for event in restarted_app.compact_session(command)]
            assert restarted_provider.calls == 0
            assert [event.id for event in replay] == [event.id for event in first]
            assert provider.calls == 1
            events = await store.load_events(command.session_id)
            assert sum(event.type == EventType.CONTEXT_COMPACTION_STARTED for event in events) == 1
            assert (
                sum(event.type == EventType.CONTEXT_COMPACTION_COMPLETED for event in events) == 1
            )
            assert not any(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in events)
        finally:
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_precommit_initial_failure_keeps_original_error_and_never_dispatches(
    backend, tmp_path, request
):
    async def run():
        store = _store(backend, tmp_path, request)
        try:
            app, provider, command = await _setup(store)
            async with _fault(store, EventType.CONTEXT_COMPACTION_STARTED, committed=False):
                with pytest.raises(ConnectionError):
                    _ = [event async for event in app.compact_session(command)]
            assert provider.calls == 0
            assert (
                await store.load_session_operation(command.session_id, command.idempotency_key)
                is None
            )
            assert not any(
                event.type == EventType.CONTEXT_COMPACTION_STARTED
                for event in await store.load_events(command.session_id)
            )
        finally:
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize(
    "boundary", [EventType.CONTEXT_COMPACTION_STARTED, EventType.SESSION_CHECKPOINTED]
)
def test_cancellation_during_reconciliation_remains_authoritative(
    backend, boundary, tmp_path, request
):
    async def run():
        store = _store(backend, tmp_path, request)
        release = asyncio.Event()
        try:
            app, provider, command = await _setup(store)
            entered = asyncio.Event()
            original = app._event_writer.is_persisted

            async def blocked(event):
                if event.type == boundary:
                    entered.set()
                    await release.wait()
                return await original(event)

            app._event_writer.is_persisted = blocked

            async def collect():
                return [event async for event in app.compact_session(command)]

            async with _fault(store, boundary):
                task = asyncio.create_task(collect())
                await asyncio.wait_for(entered.wait(), 5)
                task.cancel("cancel acknowledgement recovery")
                release.set()
                with pytest.raises(asyncio.CancelledError, match="cancel acknowledgement recovery"):
                    await task
                assert task.cancelled()
            operation = await _operation(store, command)
            initial = boundary == EventType.CONTEXT_COMPACTION_STARTED
            assert operation["status"] == ("running" if initial else "completed")
            assert provider.calls == (0 if initial else 1)
            if backend != "memory":
                await store.close()
                store = _store(backend, tmp_path, request)
            restarted_provider = AckProvider()
            restarted = _app(store, restarted_provider)
            if initial:
                with pytest.raises(RuntimeError, match="already running"):
                    _ = [event async for event in restarted.compact_session(command)]
            else:
                replay = [event async for event in restarted.compact_session(command)]
                assert replay[-1].type == EventType.SESSION_CHECKPOINTED
            assert restarted_provider.calls == 0
            assert not any(
                event.type == EventType.CONTEXT_COMPACTION_FAILED
                for event in await store.load_events(command.session_id)
            )
        finally:
            release.set()
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize(
    "mismatch",
    ["current_attempt_id", "operation_id", "request_digest", "event_ids", "claim_expires_at"],
)
def test_initial_ack_reconciliation_never_adopts_mismatched_claim(
    backend, mismatch, tmp_path, request
):
    async def run():
        store = _store(backend, tmp_path, request)
        try:
            app, provider, command = await _setup(store)

            async def replace_claim():
                def replace(_session, checkpoint):
                    record = checkpoint["session_operations"]["records"][command.idempotency_key]
                    record[mismatch] = (
                        []
                        if mismatch == "event_ids"
                        else "2000-01-01T00:00:00+00:00"
                        if mismatch == "claim_expires_at"
                        else "foreign"
                    )
                    return checkpoint

                await store.transform_checkpoint(command.session_id, replace)

            async with _fault(
                store, EventType.CONTEXT_COMPACTION_STARTED, after_commit=replace_claim
            ):
                with pytest.raises(ConnectionError, match="lost publication acknowledgement"):
                    _ = [event async for event in app.compact_session(command)]
            assert provider.calls == 0
            operation = await _operation(store, command)
            assert operation["status"] == "running"
            assert operation[mismatch] == (
                []
                if mismatch == "event_ids"
                else "2000-01-01T00:00:00+00:00"
                if mismatch == "claim_expires_at"
                else "foreign"
            )
        finally:
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_concurrent_same_key_cannot_adopt_claim_during_ack_recovery(backend, tmp_path, request):
    async def run():
        store = _store(backend, tmp_path, request)
        try:
            app, provider, command = await _setup(store)
            competitor_provider = AckProvider()
            competitor = _app(store, competitor_provider)

            async def compete():
                with pytest.raises(RuntimeError, match="already running"):
                    _ = [event async for event in competitor.compact_session(command)]
                assert competitor_provider.calls == 0

            async with _fault(store, EventType.CONTEXT_COMPACTION_STARTED, after_commit=compete):
                events = [event async for event in app.compact_session(command)]
            assert events[-1].type == EventType.SESSION_CHECKPOINTED
            assert provider.calls == 1
        finally:
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize(
    "boundary", [EventType.CONTEXT_COMPACTION_STARTED, EventType.SESSION_CHECKPOINTED]
)
def test_failed_reconciliation_preserves_original_error(backend, boundary, tmp_path, request):
    async def run():
        store = _store(backend, tmp_path, request)
        try:
            app, provider, command = await _setup(store)
            original = app._event_writer.is_persisted

            async def unavailable(event):
                if event.type == boundary:
                    raise OSError("reconciliation read unavailable")
                return await original(event)

            app._event_writer.is_persisted = unavailable
            async with _fault(store, boundary):
                with pytest.raises(ConnectionError, match="lost publication acknowledgement"):
                    _ = [event async for event in app.compact_session(command)]
            operation = await _operation(store, command)
            initial = boundary == EventType.CONTEXT_COMPACTION_STARTED
            assert operation["status"] == ("running" if initial else "completed")
            assert provider.calls == (0 if initial else 1)
            assert not any(
                event.type == EventType.CONTEXT_COMPACTION_FAILED
                for event in await store.load_events(command.session_id)
            )
        finally:
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(run())


def test_initial_reconciliation_timeout_is_bounded_and_preserves_claim(
    monkeypatch, tmp_path, request
):
    from cayu.runtime import _session_engine

    monkeypatch.setattr(_session_engine, "_SESSION_OPERATION_STORE_WAIT_TIMEOUT_SECONDS", 0.05)

    async def run():
        store = _store("memory", tmp_path, request)
        app, provider, command = await _setup(store)
        cancelled = asyncio.Event()

        async def stalled(_event):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        app._event_writer.is_persisted = stalled
        async with _fault(store, EventType.CONTEXT_COMPACTION_STARTED):
            with pytest.raises(ConnectionError, match="lost publication acknowledgement"):
                async with asyncio.timeout(2):
                    _ = [event async for event in app.compact_session(command)]
        await asyncio.wait_for(cancelled.wait(), 1)
        assert provider.calls == 0
        assert (await _operation(store, command))["status"] == "running"

    asyncio.run(run())


@pytest.mark.parametrize(
    "boundary", [EventType.CONTEXT_COMPACTION_STARTED, EventType.SESSION_CHECKPOINTED]
)
def test_non_exception_interruption_is_not_converted_to_success(boundary, tmp_path, request):
    class Interrupted(BaseException):
        pass

    async def run():
        store = _store("memory", tmp_path, request)
        app, provider, command = await _setup(store)
        interruption = Interrupted("stop")
        async with _fault(store, boundary, error=interruption):
            with pytest.raises(Interrupted) as raised:
                _ = [event async for event in app.compact_session(command)]
        assert raised.value is interruption
        initial = boundary == EventType.CONTEXT_COMPACTION_STARTED
        assert provider.calls == (0 if initial else 1)
        assert (await _operation(store, command))["status"] == (
            "running" if initial else "completed"
        )

    asyncio.run(run())
