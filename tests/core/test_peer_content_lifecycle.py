"""Peer delivery through real tool, human-input and public stream boundaries."""

import asyncio
import traceback
import warnings
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from tests.core.test_participant_identity import CONTEXT, app, create, registration
from tests.core.test_peer_content import (
    QualificationPeerExposurePolicy,
    QualifiedPeerProvider,
    _delivery_request,
    _exposure_id,
)
from tests.core.test_queued_session_messages import BlockingTool
from tests.core.test_session_creation_fence import _collaboration_factory, _store_factory

from cayu.agents import AgentSpec
from cayu.approvals.user_input import UserInputResponse
from cayu.collaboration.exports import ExportLimits, SessionExportRegistration
from cayu.collaboration.peer_content import PeerAppendKey
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent, record_peer_serialization
from cayu.providers.openai import build_openai_payload
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.base import RunRequest
from cayu.sessions.context_views import (
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
)
from cayu.sessions.invocation import InvocationOriginClaim
from cayu.tools.user_input import UserInputTool


class LifecycleProvider(QualifiedPeerProvider):
    def __init__(self, mode):
        super().__init__(events=[])
        self.mode = mode
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.second_started = asyncio.Event()
        self.release_second = asyncio.Event()

    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="tests:peer-lifecycle", behavior_version="1", implementation_version="1"
        )

    async def stream(self, request):
        build_openai_payload(request, stream=True)
        await record_peer_serialization(request)
        self.requests.append(request.model_copy(deep=True))
        number = len(self.requests)
        if number == 1 and self.mode in {"tool", "human"}:
            if self.mode == "human":
                self.started.set()
                await self.release.wait()
            yield ModelStreamEvent.tool_call(
                id="call-peer-gate",
                name="blocking_tool" if self.mode == "tool" else "ask_user",
                arguments={} if self.mode == "tool" else {"question": "Which environment?"},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        if number == 2 and self.mode == "tool":
            self.second_started.set()
            await self.release_second.wait()
        yield ModelStreamEvent.text_delta("completed response")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class LifecyclePolicy(QualificationPeerExposurePolicy):
    def __init__(self):
        super().__init__()
        self.active = 0

    @asynccontextmanager
    async def acquire_peer_exposure(self, context, **kwargs):
        async with super().acquire_peer_exposure(context, **kwargs) as projection:
            self.active += 1
            try:
                yield projection
            finally:
                self.active -= 1


@asynccontextmanager
async def journey(
    backend, tmp_path, request, mode, *, validate_schema=False, policy_factory=LifecyclePolicy
):
    store = _store_factory(backend, tmp_path, request)()
    if validate_schema:
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore
        from cayu.storage.sqlite import SQLiteSessionStore

        if backend == "postgres":
            await store.ensure_schema()
        await store.close()
        store = (
            SQLiteSessionStore(tmp_path / "creation-fence.sqlite", schema_mode=SchemaMode.VALIDATE)
            if backend == "sqlite"
            else PostgresSessionStore(
                request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.VALIDATE
            )
        )
    collaboration = _collaboration_factory(backend, tmp_path, request)()
    policy = policy_factory()
    provider = LifecycleProvider(mode)
    tool = BlockingTool()
    current = app(
        collaboration,
        registration(),
        session_store=store,
        session_exports=SessionExportRegistration(
            owner=policy.ref.owner,
            policy=policy,
            projectors=(),
            limits=ExportLimits(max_exports=8, max_pending=4, max_retained_bytes=65536),
        ),
    )
    current.register_provider(provider, default=True)
    current.register_agent(
        AgentSpec(name="reviewer", model="model"),
        tools=[UserInputTool()] if mode == "human" else [tool] if mode == "tool" else [],
    )
    initialized = await current.initialize_collaboration()
    _, source_record = await create(current, initialized, key="source")
    _, target_record = await create(current, initialized, key="target")
    sender = source_record.participants[0].reference
    consumer = target_record.participants[0].reference

    async def inert(participant, key):
        creation = ParticipantSessionCreationRequest(
            request=RunRequest(
                agent_name="reviewer",
                messages=[Message.text("user", key)],
                invocation_origin=InvocationOriginClaim(subject=CONTEXT.principal),
            ),
            creation_key=key + uuid4().hex,
        )
        session, _ = await current.create_participant_session(
            creation, participant=participant, context=CONTEXT
        )
        return session, creation

    try:
        source, _ = await inert(sender, "source")
        target, creation = await inert(consumer, "target")

        async def delivery():
            loaded = await store.load(target.id)
            snapshot = await store.load_transcript_snapshot(target.id)
            result = _delivery_request(
                source=source, target=loaded, sender=sender, consumer=consumer, suffix=uuid4().hex
            )
            result = result.model_copy(
                update={
                    "attempt_key": result.attempt_key.model_copy(
                        update={"target_transcript_cursor": snapshot.cursor}
                    )
                }
            )
            policy.allowed_receipts.add(result.occurrence.producer_receipt_id)
            return result

        stream = current.execute_participant_session(
            ParticipantSessionExecutionRequest(
                request=creation.request.model_copy(update={"session_id": target.id}),
                session_instance_id=target.instance_id,
                execution_key="execute" + uuid4().hex,
            ),
            participant=consumer,
            context=CONTEXT,
        )
        yield current, store, provider, policy, tool, target, delivery, stream
    finally:
        if backend != "memory":
            await store.close()
            await collaboration.close()


def peer_count(messages):
    return sum(part.type == "peer_content" for message in messages for part in message.content)


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("mode", ["tool", "human"])
async def test_peer_delivery_waits_for_actual_whole_tool_turn(backend, mode, tmp_path, request):
    async with journey(backend, tmp_path, request, mode) as state:
        current, store, provider, policy, tool, target, delivery, stream = state
        events = []

        async def consume():
            async for event in stream:
                events.append(event)

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(
                tool.started.wait() if mode == "tool" else provider.started.wait(), 20
            )
            expected = await delivery()
            accepted = await current.append_peer_content(expected, context=CONTEXT)
            assert accepted.status == ("pending" if mode == "tool" else "appended")
            assert len(provider.requests) == 1
            assert peer_count(await store.load_transcript(target.id)) == 0
            if mode == "tool":
                pending = await current.service_pending_peer_content(target.id, context=CONTEXT)
                assert len(pending) == 1 and pending[0].status == "pending"
                tool.release.set()
                await asyncio.wait_for(provider.second_started.wait(), 20)
                serviced = await current.service_pending_peer_content(target.id, context=CONTEXT)
                assert len(serviced) == 1 and serviced[0].status == "appended"
                provider.release_second.set()
            else:
                provider.release.set()
            await asyncio.wait_for(task, 30)
            if mode == "human":
                awaiting = next(
                    e for e in events if e.type == EventType.SESSION_AWAITING_USER_INPUT
                )
                assert len(provider.requests) == 1
                assert peer_count(await store.load_transcript(target.id)) == 0
                paused_delivery = await delivery()
                checkpoint = await store.load_checkpoint(target.id)
                paused_result = await current.append_peer_content(paused_delivery, context=CONTEXT)
                assert paused_result.status == "excluded"
                assert (await store.read_peer_content(paused_delivery.append_key)) == paused_result
                assert await store.load_checkpoint(target.id) == checkpoint
                assert len(provider.requests) == 1
                events.extend(
                    [
                        event
                        async for event in current.resolve_user_input(
                            UserInputResponse(
                                session_id=target.id,
                                input_id=awaiting.payload["input_id"],
                                answer="production",
                            ),
                            context=CONTEXT,
                        )
                    ]
                )
            assert any(e.type == EventType.SESSION_COMPLETED for e in events)
            assert len(provider.requests) == 3
            assert peer_count(provider.requests[1].messages) == 0
            assert peer_count(provider.requests[2].messages) == 1
            parts = [part for m in provider.requests[2].messages for part in m.content]
            calls = [p for p in parts if p.type == "tool_call"]
            results = [p for p in parts if p.type == "tool_result"]
            assert len(calls) == len(results) == 1
            assert calls[0].tool_call_id == results[0].tool_call_id == "call-peer-gate"
            assert parts.index(calls[0]) < parts.index(results[0])
            assert peer_count(await store.load_transcript(target.id)) == 1
            replay = await current.append_peer_content(expected, context=CONTEXT)
            assert replay.status == "appended" and replay.replayed
            assert len(provider.requests) == 3 and policy.active == 0
        finally:
            tool.release.set()
            provider.release.set()
            provider.release_second.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await stream.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("boundary", [EventType.MODEL_STARTED, EventType.MODEL_TEXT_DELTA])
async def test_public_peer_stream_abandonment_preserves_exposure(
    backend, boundary, tmp_path, request
):
    async with journey(backend, tmp_path, request, "abandon") as state:
        current, store, provider, policy, _, target, delivery, stream = state
        expected = await delivery()
        assert (await current.append_peer_content(expected, context=CONTEXT)).status == "appended"
        try:
            async with asyncio.timeout(30):
                async for event in stream:
                    if event.type == boundary:
                        break
                else:
                    pytest.fail("public stream did not reach abandonment boundary")
            assert policy.active == 1
        finally:
            await stream.aclose()
        assert policy.active == 0
        assert target.id not in current._participant_session_execution_locks
        assert (await store.load(target.id)).status == "interrupted"
        assert len(provider.requests) == (0 if boundary == EventType.MODEL_STARTED else 1)
        exposure = await store.read_peer_content_exposure(
            expected.append_key, _exposure_id(expected, policy.calls[0]["model_attempt_id"])
        )
        assert exposure is not None
        assert exposure.outcome == ("pending" if boundary == EventType.MODEL_STARTED else "exposed")


@pytest.mark.anyio
@pytest.mark.parametrize("entrance", ["append", "exclude", "read"])
async def test_rejected_peer_values_do_not_leak_through_diagnostics(
    entrance, tmp_path, request, caplog, capsys
):
    async with journey("memory", tmp_path, request, "diagnostics") as state:
        current, store, provider, _, _, _, delivery, stream = state
        expected = await delivery()
        canary = "peer-rejected-" + uuid4().hex

        class HostileValue:
            def __repr__(self):
                return canary

            def __str__(self):
                return canary

        # Bypass construction deliberately: every public entrance must revalidate,
        # including safe diagnostics for a malformed value and its valid sibling.
        object.__setattr__(expected.occurrence.payload, "text", HostileValue())
        object.__setattr__(expected.occurrence, "producer_receipt_id", canary)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with pytest.raises(ValueError) as failure:
                if entrance == "append":
                    await current.append_peer_content(expected, context=CONTEXT)
                elif entrance == "exclude":
                    await current.exclude_peer_content(
                        expected, reason="withdrawn", context=CONTEXT
                    )
                else:
                    await current.read_peer_content(
                        expected.append_key, expected=expected, context=CONTEXT
                    )
        diagnostic = "".join(traceback.format_exception(failure.value))
        diagnostic += str(failure.value) + repr(failure.value)
        diagnostic += "".join(str(warning.message) for warning in captured)
        output = capsys.readouterr()
        diagnostic += caplog.text + output.out + output.err
        assert canary not in diagnostic
        assert await store.read_peer_content(expected.append_key) is None
        assert await store.list_pending_peer_content() == ()
        assert provider.requests == []
        await stream.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_peer_readback_revalidates_key_before_diagnostics(
    backend, tmp_path, request, caplog, capsys
):
    async with journey(backend, tmp_path, request, "diagnostics") as state:
        current, store, provider, _, _, _, delivery, stream = state
        expected = await delivery()
        canary = "peer-key-" + uuid4().hex
        malformed = expected.append_key.model_copy(
            update={"collaboration_generation": canary, "occurrence_id": canary}
        )
        for entrance in ("public", "public_expected", "native", "exposure"):
            caplog.clear()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                with pytest.raises(ValueError) as failure:
                    if entrance.startswith("public"):
                        await current.read_peer_content(
                            malformed,
                            context=CONTEXT,
                            expected=expected if entrance == "public_expected" else None,
                        )
                    elif entrance == "native":
                        await store.read_peer_content(malformed)
                    else:
                        await store.read_peer_content_exposure(malformed, "no-exposure")
            diagnostic = "".join(traceback.format_exception(failure.value))
            diagnostic += str(failure.value) + repr(failure.value)
            diagnostic += "".join(str(w.message) for w in captured)
            output = capsys.readouterr()
            assert canary not in diagnostic + caplog.text + output.out + output.err
        assert await store.read_peer_content(expected.append_key) is None
        assert await store.list_pending_peer_content() == ()
        assert provider.requests == []
        # Valid reconstructed keys retain normal not-found semantics.
        copied = PeerAppendKey.model_validate_json(expected.append_key.model_dump_json())
        assert await current.read_peer_content(copied, context=CONTEXT) is None
        await stream.aclose()


@pytest.mark.anyio
async def test_postgres_peer_append_and_deletion_have_consistent_lock_order(
    tmp_path, request, monkeypatch
):
    from psycopg import AsyncCursor

    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async with journey("postgres", tmp_path, request, "race") as state:
        current, store, provider, _, _, target, delivery, stream = state
        expected = await delivery()
        deletion = PostgresSessionStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.VALIDATE
        )
        await deletion.ensure_schema()
        deletion_locked = asyncio.Event()
        append_ready = asyncio.Event()
        allow_append = asyncio.Event()
        append_at_session_lock = asyncio.Event()
        allow_deletion = asyncio.Event()
        original_quiescence = deletion._require_session_erasure_quiescence
        original_execute = AsyncCursor.execute

        async def held_deletion(cursor, session):
            await original_quiescence(cursor, session)
            deletion_locked.set()
            await allow_deletion.wait()

        async def execute(cursor, statement, parameters=None, **kwargs):
            text = str(statement)
            if not append_ready.is_set() and (
                "CREATE TABLE IF NOT EXISTS cayu_peer_content_receipts" in text
                or (
                    "pg_advisory_xact_lock" in text
                    and parameters
                    and str(parameters[0]).startswith("peer-append:")
                )
            ):
                append_ready.set()
                await allow_append.wait()
            if text.startswith(
                "SELECT instance_id, run_epoch, status, agent_name, environment_name "
                "FROM cayu_sessions"
            ):
                append_at_session_lock.set()
            return await original_execute(cursor, statement, parameters, **kwargs)

        monkeypatch.setattr(deletion, "_require_session_erasure_quiescence", held_deletion)
        monkeypatch.setattr(AsyncCursor, "execute", execute)

        async def reached(event, task):
            waiter = asyncio.create_task(event.wait())
            try:
                done, _ = await asyncio.wait((waiter, task), return_when=asyncio.FIRST_COMPLETED)
                if task in done:
                    await task
                    assert event.is_set(), "operation ended before reaching its barrier"
            finally:
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)

        delete_task = None
        append_task = asyncio.create_task(current.append_peer_content(expected, context=CONTEXT))
        try:
            async with asyncio.timeout(30):
                await reached(append_ready, append_task)
                delete_task = asyncio.create_task(deletion.delete_session(target.id))
                await reached(deletion_locked, delete_task)
                allow_append.set()
                await reached(append_at_session_lock, append_task)
                allow_deletion.set()
                await delete_task
                result = await append_task
            assert result.status == "pending" and result.reason == "target_not_created"
            assert await store.load(target.id) is None
            assert await store.read_peer_content(expected.append_key) == result
            settled = await current.exclude_peer_content(
                expected, reason="target_deleted", context=CONTEXT
            )
            assert settled.status == "excluded"
            assert provider.requests == []
        finally:
            allow_append.set()
            allow_deletion.set()
            tasks = [append_task] + ([] if delete_task is None else [delete_task])
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await stream.aclose()
            await deletion.close()


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_validated_peer_runtime_performs_no_schema_mutations(
    backend, tmp_path, request, monkeypatch
):
    import sqlite3

    async with journey(backend, tmp_path, request, "schema", validate_schema=True) as state:
        current, store, provider, policy, _, _, delivery, stream = state
        ddl = []

        if backend == "sqlite":

            def authorize(action, *args):
                if action in {sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_ALTER_TABLE}:
                    ddl.append(action)
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            store._connection.set_authorizer(authorize)
        else:
            from psycopg import AsyncCursor

            original_execute = AsyncCursor.execute

            async def execute(cursor, statement, parameters=None, **kwargs):
                text = str(statement).lstrip().upper()
                if text.startswith(("CREATE ", "ALTER ", "DROP ")):
                    ddl.append(text)
                    raise AssertionError("Validated peer runtime attempted DDL")
                return await original_execute(cursor, statement, parameters, **kwargs)

            monkeypatch.setattr(AsyncCursor, "execute", execute)
        try:
            expected = await delivery()
            appended = await current.append_peer_content(expected, context=CONTEXT)
            assert appended.status == "appended"
            assert (await current.append_peer_content(expected, context=CONTEXT)).replayed
            pending = await delivery()
            pending = pending.model_copy(
                update={
                    "attempt_key": pending.attempt_key.model_copy(
                        update={"target_transcript_cursor": 1}
                    )
                }
            )
            assert (await current.append_peer_content(pending, context=CONTEXT)).status == "pending"
            excluded = await current.exclude_peer_content(
                pending, reason="withdrawn", context=CONTEXT
            )
            assert excluded.status == "excluded"
            assert (
                await current.exclude_peer_content(pending, reason="withdrawn", context=CONTEXT)
            ).replayed
            events = [event async for event in stream]
            assert any(event.type == EventType.SESSION_COMPLETED for event in events)
            assert len(provider.requests) == 1
            exposure = await store.read_peer_content_exposure(
                expected.append_key, _exposure_id(expected, policy.calls[0]["model_attempt_id"])
            )
            assert exposure is not None and exposure.outcome == "exposed"
            assert await current.read_peer_content(expected.append_key, context=CONTEXT) == appended
            assert not ddl
        finally:
            await stream.aclose()
            if backend == "sqlite":
                store._connection.set_authorizer(None)


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_peer_schema_requires_explicit_migration(backend, tmp_path, request):
    import sqlite3

    from cayu.storage.migrations import SchemaMode, SchemaTooOld
    from cayu.storage.postgres import PostgresSessionStore
    from cayu.storage.sqlite import SQLiteSessionStore

    creator = _store_factory(backend, tmp_path, request)()
    if backend == "postgres":
        await creator.ensure_schema()
    await creator.close()
    path = tmp_path / "creation-fence.sqlite"
    if backend == "sqlite":
        with sqlite3.connect(path) as connection:
            connection.execute("DELETE FROM cayu_schema_migrations WHERE revision = 104")
    else:
        import psycopg

        dsn = request.getfixturevalue("postgres_dsn")
        async with await psycopg.AsyncConnection.connect(dsn) as connection:
            await connection.execute("DELETE FROM cayu_schema_migrations WHERE revision = 104")
    try:
        if backend == "sqlite":
            with pytest.raises(SchemaTooOld, match="requires >= 104"):
                SQLiteSessionStore(path, schema_mode=SchemaMode.VALIDATE)
        else:
            validator = PostgresSessionStore(dsn, schema_mode=SchemaMode.VALIDATE)
            try:
                with pytest.raises(SchemaTooOld, match="requires >= 104"):
                    await validator.ensure_schema()
            finally:
                await validator.close()
    finally:
        migrator = (
            SQLiteSessionStore(path, schema_mode=SchemaMode.MIGRATE)
            if backend == "sqlite"
            else PostgresSessionStore(dsn, schema_mode=SchemaMode.MIGRATE)
        )
        try:
            if backend == "postgres":
                await migrator.ensure_schema()
        finally:
            await migrator.close()
