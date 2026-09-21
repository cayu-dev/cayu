from __future__ import annotations

import asyncio
import multiprocessing
import sqlite3

import pytest

from cayu.agents import AgentSpec
from cayu.collaboration.lifecycle import ParticipantLifecycleChange
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.base import InMemorySessionStore, Message, ResumeRequest, RunRequest
from cayu.sessions.context_views import (
    ContextViewLimits,
    ContextViewManifest,
    ContextViewOwnershipRequest,
    ContextViewPublicationRequest,
    ContextViewSelectionRequest,
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
)
from cayu.storage.sqlite import SQLiteSessionStore


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("lifecycle", ["disabled", "retired"])
def test_inactive_participant_cannot_resume_but_can_release(backend, lifecycle, tmp_path, request):
    from tests.core.test_participant_identity import CONTEXT, app, create, registration

    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    def stores():
        if backend == "memory":
            return InMemorySessionStore(), InMemoryCollaborationStore()
        if backend == "sqlite":
            from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore

            return SQLiteSessionStore(tmp_path / "sessions.sqlite"), SQLiteCollaborationStore(
                tmp_path / "participants.sqlite"
            )
        from cayu.storage.collaboration_postgres import PostgresCollaborationStore
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        return PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE), PostgresCollaborationStore(
            dsn, schema_mode=SchemaMode.CREATE
        )

    async def run():
        sessions, collaboration = stores()
        prefix = f"{backend}-{lifecycle}"
        reg = registration()
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
            * 4
        )

        async def application():
            value = app(collaboration, reg, session_store=sessions)
            value.register_provider(provider, default=True)
            value.register_agent(AgentSpec(name="reviewer", model="model"))
            initialized = await value.initialize_collaboration()
            return value, initialized

        try:
            value, initialized = await application()
            _, created = await create(value, initialized)
            participant = created.participants[0].reference
            creation = ParticipantSessionCreationRequest(
                RunRequest(agent_name="reviewer", messages=[Message.text("user", "start")]),
                "create-session",
            )
            session, _ = await value.create_participant_session(
                creation, participant=participant, context=CONTEXT
            )
            assert provider.requests == []
            events = [
                event
                async for event in value.execute_participant_session(
                    ParticipantSessionExecutionRequest(
                        request=creation.request.model_copy(update={"session_id": session.id}),
                        session_instance_id=session.instance_id,
                        execution_key="execute",
                    ),
                    participant=participant,
                    context=CONTEXT,
                )
            ]
            assert any(event.type is EventType.SESSION_COMPLETED for event in events)
            from cayu.runtime._model_completion_publication import (
                model_step_publication_from_checkpoint,
            )

            pointer = model_step_publication_from_checkpoint(
                await sessions.load_checkpoint(session.id)
            )
            assert pointer is not None
            completed = next(
                event
                for event in await sessions.load_events(session.id)
                if event.id == pointer.completion_event_id
            )
            manifest = await value.publish_completed_context_view(
                ContextViewPublicationRequest(
                    source_session_id=session.id,
                    source_session_instance_id=session.instance_id,
                    view_id=f"{prefix}-view",
                    interaction_id=completed.interaction_id,
                    boundary_id=pointer.logical_step_id,
                    projection_schema="whole-turn.v1",
                    publication_key=f"{prefix}-publish",
                ),
                participant=participant,
                context=CONTEXT,
            )
            selected = await value.select_context_view(
                ContextViewSelectionRequest(
                    source_owner=participant.owner,
                    source_session_id=session.id,
                    source_session_instance_id=session.instance_id,
                    selector="latest",
                    projection_schema=manifest.projection_schema,
                    extension_set_commitment=manifest.extension_set_commitment,
                    limits=ContextViewLimits(),
                    selection_key=f"{prefix}-select",
                ),
                participant=participant,
                context=CONTEXT,
            )
            adopted = await value.transition_context_view_ownership(
                ContextViewOwnershipRequest(
                    selection_key=selected.selection_key,
                    view_id=manifest.view_id,
                    pin_commitment=selected.pin_commitment,
                    expected_state="selected",
                    expected_revision=selected.ownership_revision,
                    operation="adopt",
                    current_owner=participant.owner,
                    destination_owner=participant.owner,
                    operation_key=f"{prefix}-adopt",
                ),
                participant=participant,
                destination_participant=participant,
                context=CONTEXT,
            )
            resume = ResumeRequest(session_id=session.id, messages=[Message.text("user", "next")])
            with pytest.raises(PermissionError, match="administration"):
                [event async for event in value.resume(resume)]
            assert len(provider.requests) == 1
            from cayu.collaboration.access import (
                CollaborationAccessContext,
                CollaborationAccessDenied,
            )

            with pytest.raises(CollaborationAccessDenied):
                [
                    event
                    async for event in value.resume(
                        resume, context=CollaborationAccessContext(principal="outsider")
                    )
                ]
            [event async for event in value.resume(resume, context=CONTEXT)]
            assert len(provider.requests) == 2
            # Real cancellation during authority readback must not admit work.
            before_cancel = (
                await sessions.load(session.id),
                await sessions.load_events(session.id),
            )
            original_inspect = collaboration.inspect
            inspected = asyncio.Event()

            async def paused_inspect(*args, **kwargs):
                await original_inspect(*args, **kwargs)
                inspected.set()
                await asyncio.Event().wait()

            async def collect():
                return [event async for event in value.resume(resume, context=CONTEXT)]

            collaboration.inspect = paused_inspect
            caller = asyncio.create_task(collect())
            try:
                await asyncio.wait_for(inspected.wait(), 10)
                caller.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await caller
                assert caller.cancelled() and caller.cancelling() == 1
            finally:
                collaboration.inspect = original_inspect
                if not caller.done():
                    caller.cancel()
                await asyncio.gather(caller, return_exceptions=True)
            assert before_cancel == (
                await sessions.load(session.id),
                await sessions.load_events(session.id),
            )
            assert len(provider.requests) == 2
            await value.change_participant_lifecycle(
                ParticipantLifecycleChange(
                    operation=initialized.operation("deactivate"),
                    participant=participant,
                    expected_lifecycle_revision=1,
                    state=lifecycle,
                ),
                context=CONTEXT,
            )
            if backend != "memory":
                await sessions.close()
                await collaboration.close()
                sessions, collaboration = stores()
                value, _ = await application()
            before = (
                await sessions.load(session.id),
                await sessions.load_checkpoint(session.id),
                await sessions.load_events(session.id),
                await sessions.load_transcript(session.id),
            )
            with pytest.raises(PermissionError, match="active participants"):
                [event async for event in value.resume(resume, context=CONTEXT)]
            assert len(provider.requests) == 2
            assert before == (
                await sessions.load(session.id),
                await sessions.load_checkpoint(session.id),
                await sessions.load_events(session.id),
                await sessions.load_transcript(session.id),
            )
            release = ContextViewOwnershipRequest(
                selection_key=selected.selection_key,
                view_id=manifest.view_id,
                pin_commitment=selected.pin_commitment,
                expected_state=adopted.state,
                expected_revision=adopted.ownership_revision,
                operation="release",
                current_owner=participant.owner,
                operation_key=f"{prefix}-release",
            )
            with pytest.raises(PermissionError, match="active participants"):
                await value.transition_context_view_ownership(
                    release.model_copy(
                        update={
                            "operation": "transfer",
                            "destination_owner": participant.owner,
                            "operation_key": "forbidden-transfer",
                        }
                    ),
                    participant=participant,
                    destination_participant=participant,
                    context=CONTEXT,
                )
            with pytest.raises(CollaborationAccessDenied):
                await value.transition_context_view_ownership(
                    release,
                    participant=participant,
                    context=CollaborationAccessContext(principal="outsider"),
                )
            released = await value.transition_context_view_ownership(
                release,
                participant=participant,
                context=CONTEXT,
            )
            assert released.state == "released"
            assert (
                await value.transition_context_view_ownership(
                    release,
                    participant=participant,
                    context=CONTEXT,
                )
                == released
            )
            assert [
                event.state
                for event in await sessions.read_context_view_lifecycle_events(manifest.view_id)
            ] == ["adopted", "released"]
            await sessions.delete_session(session.id)
            assert await sessions.load(session.id) is None
            # Unbound sessions are not subject to participant administration.
            ordinary = [
                event
                async for event in value.run(
                    RunRequest(
                        agent_name="reviewer",
                        session_id=f"{prefix}-ordinary",
                        messages=[Message.text("user", "start")],
                    )
                )
            ]
            assert any(event.type is EventType.SESSION_COMPLETED for event in ordinary)
            [
                event
                async for event in value.resume(
                    ResumeRequest(
                        session_id=f"{prefix}-ordinary", messages=[Message.text("user", "next")]
                    )
                )
            ]
        finally:
            if hasattr(sessions, "close"):
                await sessions.close()
            await collaboration.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("material", ["complete", "missing", "resource", "wrong_frontier"])
def test_compaction_mutation_requires_independent_material(backend, material, tmp_path, request):
    from datetime import UTC, datetime

    from tests.core.test_context_view_admission import _close, _factory, _new_manifest, _selection
    from tests.core.test_context_views import _replace_manifest

    from cayu.sessions.base import SessionOperationPublication

    factory = _factory(backend, tmp_path, request, [datetime.now(UTC)])

    async def run():
        store = factory()
        try:
            manifest = await _new_manifest(store)
            compaction = '{"input_frontier":0,"retained_output_frontier":1,"retained_suffix_frontier":1,"state":"uncompacted"}'
            if material == "wrong_frontier":
                compaction = compaction.replace('"input_frontier":0', '"input_frontier":1')
            manifest = _replace_manifest(
                manifest,
                compaction_json=None if material == "missing" else compaction,
                resource_references_json='["unqualified"]' if material == "resource" else None,
            )
            await store.publish_context_view(manifest, publication_key=manifest.view_id)
            await store.select_context_view(_selection(manifest, "pin"))
            before = await store.load_checkpoint(manifest.source_session_id)

            async def compact():
                await store.publish_session_operation_guarded_with_store_time(
                    manifest.source_session_id,
                    idempotency_key="compaction-proof",
                    operation_transform=lambda *_: SessionOperationPublication(
                        checkpoint={"compacted": True}
                    ),
                    commit_guard=lambda: None,
                    commit_time_guard=lambda _: None,
                    events=[],
                    context_view_compaction_cursor=1,
                )

            if material == "complete":
                await compact()
                assert (await store.load_checkpoint(manifest.source_session_id))[
                    "compacted"
                ] is True
            else:
                with pytest.raises(ValueError, match="Pinned context-view material"):
                    await compact()
                assert await store.load_checkpoint(manifest.source_session_id) == before
            # Even a complete copy cannot authorize source deletion.
            with pytest.raises(ValueError, match="retention pin"):
                await store.delete_session(manifest.source_session_id)
        finally:
            await _close(store)

    asyncio.run(run())


def _publish_process(path, document, key, quota, gate_sql, ready, start, captured, release, result):
    import cayu.sessions.context_views as contracts

    contracts.CONTEXT_VIEW_MAX_PUBLICATIONS_PER_OWNER = quota

    async def run():
        store = SQLiteSessionStore(path)
        connection = store._connection

        class Cursor:
            def __init__(self, cursor):
                self.cursor = cursor

            def fetchone(self):
                row = self.cursor.fetchone()
                captured.set()
                assert release.wait(30)
                return row

        class Connection:
            def __getattr__(self, name):
                return getattr(connection, name)

            def __enter__(self):
                connection.__enter__()
                return self

            def __exit__(self, *args):
                return connection.__exit__(*args)

            def execute(self, sql, *args):
                cursor = connection.execute(sql, *args)
                return Cursor(cursor) if gate_sql and gate_sql in sql else cursor

        store._connection = Connection()
        ready.set()
        try:
            assert start.wait(30)
            manifest = await store.publish_context_view(
                ContextViewManifest.model_validate_json(document), publication_key=key
            )
            result.put(("ok", manifest.manifest_commitment))
        except Exception as error:
            result.put((type(error).__name__, str(error)))
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("same_key", [True, False])
def test_sqlite_publication_serializes_independent_processes(tmp_path, same_key):
    from tests.core.test_context_views import _manifest_for_store, _replace_manifest

    path = tmp_path / "publication.sqlite"

    async def prepare():
        store = SQLiteSessionStore(path)
        try:
            return await _manifest_for_store(store)
        finally:
            await store.close()

    first = asyncio.run(prepare())
    second = first if same_key else _replace_manifest(first, view_id="second")
    ctx = multiprocessing.get_context("spawn")
    ready_a, ready_b, start_a, start_b, captured, release = [ctx.Event() for _ in range(6)]
    results = ctx.Queue()
    query = "WHERE publication_key" if same_key else "SELECT COUNT(*)"
    processes = [
        ctx.Process(
            target=_publish_process,
            args=(
                str(path),
                first.model_dump_json(),
                "key",
                1,
                query,
                ready_a,
                start_a,
                captured,
                release,
                results,
            ),
        ),
        ctx.Process(
            target=_publish_process,
            args=(
                str(path),
                second.model_dump_json(),
                "key" if same_key else "second",
                1,
                None,
                ready_b,
                start_b,
                captured,
                release,
                results,
            ),
        ),
    ]
    try:
        for process in processes:
            process.start()
        assert ready_a.wait(30) and ready_b.wait(30)
        start_a.set()
        assert captured.wait(30)
        # The initial replay/quota read already owns a cross-process write lock.
        with (
            sqlite3.connect(path, timeout=0) as competing,
            pytest.raises(sqlite3.OperationalError, match="locked"),
        ):
            competing.execute("BEGIN IMMEDIATE")
        start_b.set()
        release.set()
        outcomes = [results.get(timeout=30), results.get(timeout=30)]
        if same_key:
            assert outcomes == [("ok", first.manifest_commitment)] * 2
        else:
            assert sorted(item[0] for item in outcomes) == ["OverflowError", "ok"]
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM cayu_context_views").fetchone()[0] == 1
    finally:
        start_a.set()
        start_b.set()
        release.set()
        for process in processes:
            process.join(30)
            if process.is_alive():
                process.kill()
                process.join()
        results.close()
        results.join_thread()


@pytest.mark.parametrize("lost_ack", [False, True])
def test_sqlite_publication_rollback_and_acknowledgement_loss(tmp_path, lost_ack):
    from tests.core.test_context_views import _manifest_for_store

    async def run():
        store = SQLiteSessionStore(tmp_path / "publication-failure.sqlite")
        manifest = await _manifest_for_store(store)
        connection = store._connection

        class Connection:
            def __getattr__(self, name):
                return getattr(connection, name)

            def __enter__(self):
                connection.__enter__()
                return self

            def __exit__(self, *args):
                result = connection.__exit__(*args)
                if lost_ack and args[0] is None:
                    raise OSError("acknowledgement lost")
                return result

            def execute(self, sql, *args):
                result = connection.execute(sql, *args)
                if not lost_ack and "INSERT INTO cayu_context_views" in sql:
                    raise OSError("write failed before commit")
                return result

        try:
            store._connection = Connection()
            with pytest.raises(OSError):
                await store.publish_context_view(manifest, publication_key="key")
            assert not connection.in_transaction
            assert connection.execute("SELECT COUNT(*) FROM cayu_context_views").fetchone()[
                0
            ] == int(lost_ack)
        finally:
            store._connection = connection
            await store.close()
        reopened = SQLiteSessionStore(tmp_path / "publication-failure.sqlite")
        try:
            assert await reopened.publish_context_view(manifest, publication_key="key") == manifest
            assert (
                reopened._connection.execute("SELECT COUNT(*) FROM cayu_context_views").fetchone()[
                    0
                ]
                == 1
            )
        finally:
            await reopened.close()

    asyncio.run(run())
