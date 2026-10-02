"""Native reader characterization, not producer-registration acceptance coverage."""

from uuid import uuid4

import pytest

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration._native_output import read_native_output
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.sessions._model_completion_publication import model_step_publication_from_checkpoint
from cayu.sessions.base import InMemorySessionStore, RunRequest


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_native_output_reads_exact_public_runtime_publication(backend, tmp_path, request):
    if backend == "memory":
        store = InMemorySessionStore()
    elif backend == "sqlite":
        from cayu.storage.sqlite import SQLiteSessionStore

        store = SQLiteSessionStore(tmp_path / "output.sqlite")
    else:
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        store = PostgresSessionStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )
    try:
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("answer"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="producer", model="scripted"))
        session_id = "output-" + uuid4().hex
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="producer",
                    session_id=session_id,
                    messages=[Message.text("user", "answer")],
                )
            )
        ]
        assert events
        checkpoint = await runtime_checkpoint_session_store(store).load_checkpoint(session_id)
        pointer = model_step_publication_from_checkpoint(checkpoint)
        assert pointer is not None
        stage = await store.load_model_completion_stage(session_id, pointer.stage_id)
        session = await store.load(session_id)
        assert stage is not None and stage.publication is not None and session is not None
        expected = dict(
            session_id=session_id,
            session_instance_id=session.instance_id,
            invocation_id=stage.publication.interaction_id,
            run_epoch=stage.source_run_epoch,
            stage_id=pointer.stage_id,
            source_indices=(pointer.source_transcript_cursor,),
        )
        evidence = await read_native_output(store, **expected)
        assert evidence == await read_native_output(store, **expected)
        assert evidence.publication_id == stage.publication.publication_id
        for field, value in (
            ("session_instance_id", "wrong-incarnation"),
            ("invocation_id", "wrong-invocation"),
            ("run_epoch", stage.source_run_epoch + 1),
            ("run_epoch", True),
            ("source_indices", (True,)),
            ("source_indices", (2**53,)),
            ("stage_id", "absent-stage"),
        ):
            with pytest.raises(CollaborationConflict):
                await read_native_output(store, **(expected | {field: value}))
        assert len(provider.requests) == 1
    finally:
        if backend != "memory":
            await store.close()
