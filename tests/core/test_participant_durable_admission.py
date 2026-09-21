"""Cross-instance durable permit ordering for participant admission."""

import asyncio

import pytest
from tests.core import test_participant_identity as identity_tests
from tests.core.test_collaboration_permits import REDACTOR, Receiver, permit
from tests.core.test_participant_identity import CONTEXT, app, create, registration
from tests.core.test_participant_lifecycle import change

from cayu.agents import AgentSpec
from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.base import InMemorySessionStore, Message, RunRequest
from cayu.sessions.context_views import (
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
)

pytestmark = pytest.mark.anyio
stores = identity_tests.stores


async def test_permit_read_then_disable_rejects_without_mutation(stores):
    first_store, second_store = stores(), stores()
    reg = registration()
    first, second = app(first_store, reg), app(second_store, reg)
    initialized = await first.initialize_collaboration()
    assert await second.initialize_collaboration() == initialized
    _, created = await create(first, initialized)
    participant = created.participants[0].reference
    ready, release = asyncio.Event(), asyncio.Event()

    async def register_after_read():
        inspected = await first.inspect_participant(participant, context=CONTEXT)
        expected = permit(
            initialized,
            participant,
            revision=inspected.participant.lifecycle_revision,
            generation=inspected.participant.admission_generation,
        )
        ready.set()
        await release.wait()
        return await first_store._register_permit(initialized, expected, redactor=REDACTOR)

    task = asyncio.create_task(register_after_read())
    try:
        await asyncio.wait_for(ready.wait(), 10)
        await second.change_participant_lifecycle(
            change(initialized, participant, key="disable", revision=1, state="disabled"),
            context=CONTEXT,
        )
        before = await second.inspect_participant(participant, context=CONTEXT)
        events = await second.list_participant_events(context=CONTEXT)
        release.set()
        with pytest.raises(CollaborationConflict):
            await asyncio.wait_for(task, 10)
        assert await second.inspect_participant(participant, context=CONTEXT) == before
        assert await second.list_participant_events(context=CONTEXT) == events
        assert before.issued_permit_frontier == before.outstanding_obligations == 0
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_registered_permit_remains_valid_after_disable_and_settles(stores):
    first_store, second_store = stores(), stores()
    reg = registration()
    first, second = app(first_store, reg), app(second_store, reg)
    initialized = await first.initialize_collaboration()
    assert await second.initialize_collaboration() == initialized
    _, created = await create(first, initialized)
    participant = created.participants[0].reference
    expected = permit(initialized, participant)
    admitted = await first_store._register_permit(initialized, expected, redactor=REDACTOR)
    disabled = await second.change_participant_lifecycle(
        change(initialized, participant, key="disable", revision=1, state="disabled"),
        context=CONTEXT,
    )
    assert disabled.participant.covered_permit_frontier == admitted.position == 1
    assert await second_store._register_permit(initialized, expected, redactor=REDACTOR) == admitted
    settled = await second_store._settle_permit(
        initialized, expected, reader=Receiver(expected), redactor=REDACTOR
    )
    assert settled.receiving_receipt.outcome == "quiescent"
    assert (
        await first.inspect_participant(participant, context=CONTEXT)
    ).outstanding_obligations == 0


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_public_execution_race_rejects_after_disable_before_permit_registration(
    backend, tmp_path, request
):
    if backend == "memory":
        first_store = second_store = identity_tests.InMemoryCollaborationStore()
    elif backend == "sqlite":
        from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore

        path = tmp_path / "participant-race.sqlite"
        first_store = SQLiteCollaborationStore(path)
        second_store = SQLiteCollaborationStore(path)
    else:
        from cayu.storage.collaboration_postgres import PostgresCollaborationStore
        from cayu.storage.migrations import SchemaMode

        dsn = request.getfixturevalue("postgres_dsn")
        first_store = PostgresCollaborationStore(dsn, schema_mode=SchemaMode.CREATE)
        second_store = PostgresCollaborationStore(dsn, schema_mode=SchemaMode.VALIDATE)
    if backend == "memory":
        session_store = InMemorySessionStore()
    elif backend == "sqlite":
        from cayu.storage.sqlite import SQLiteSessionStore

        session_store = SQLiteSessionStore(tmp_path / "participant-race-sessions.sqlite")
    else:
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        session_store = PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)
    reg = registration()
    first = app(first_store, reg, session_store=session_store)
    second = app(second_store, reg, session_store=session_store)
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    first.register_provider(provider, default=True)
    first.register_agent(AgentSpec(name="reviewer", model="model"))
    initialized = await first.initialize_collaboration()
    assert await second.initialize_collaboration() == initialized
    _, created = await create(first, initialized)
    participant = created.participants[0].reference
    creation = ParticipantSessionCreationRequest(
        creation_key="race-create",
        request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "start")]),
    )
    session, _ = await first.create_participant_session(
        creation, participant=participant, context=CONTEXT
    )
    ready, release = asyncio.Event(), asyncio.Event()
    original = first_store._register_permit

    async def paused_register(*args, **kwargs):
        ready.set()
        await release.wait()
        return await original(*args, **kwargs)

    first_store._register_permit = paused_register
    execution = ParticipantSessionExecutionRequest(
        request=creation.request.model_copy(update={"session_id": session.id}),
        session_instance_id=session.instance_id,
        execution_key="race-execution",
    )

    async def execute():
        return [
            event
            async for event in first.execute_participant_session(
                execution, participant=participant, context=CONTEXT
            )
        ]

    task = asyncio.create_task(execute())
    await asyncio.wait_for(ready.wait(), 10)
    await second.change_participant_lifecycle(
        change(initialized, participant, key="race-disable", revision=1, state="disabled"),
        context=CONTEXT,
    )
    before = await session_store.load(session.id)
    release.set()
    outcome = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(outcome[0], CollaborationConflict)
    assert provider.requests == []
    assert await session_store.load(session.id) == before
    if first_store is not second_store:
        await first_store.close()
        await second_store.close()
    close_session = getattr(session_store, "close", None)
    if close_session is not None:
        await close_session()


async def test_public_execution_permit_registered_before_disable_remains_valid():
    """A durable handoff acquired before disablement may finish its exact admission."""
    collaboration = identity_tests.InMemoryCollaborationStore()
    session_store = InMemorySessionStore()
    reg = registration()
    first = app(collaboration, reg, session_store=session_store)
    second = app(collaboration, reg, session_store=session_store)
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    first.register_provider(provider, default=True)
    first.register_agent(AgentSpec(name="reviewer", model="model"))
    initialized = await first.initialize_collaboration()
    assert await second.initialize_collaboration() == initialized
    _, created = await create(first, initialized, key="permit-before-disable")
    participant = created.participants[0].reference
    creation = ParticipantSessionCreationRequest(
        creation_key="permit-before-disable-create",
        request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "start")]),
    )
    session, _ = await first.create_participant_session(
        creation, participant=participant, context=CONTEXT
    )
    admitted, release = asyncio.Event(), asyncio.Event()
    runtime_store = first._runtime_session_store
    original_apply = runtime_store.apply_invocation_lifecycle_command

    async def paused_apply(command):
        admitted.set()
        await release.wait()
        return await original_apply(command)

    runtime_store.apply_invocation_lifecycle_command = paused_apply
    execution = ParticipantSessionExecutionRequest(
        request=creation.request.model_copy(update={"session_id": session.id}),
        session_instance_id=session.instance_id,
        execution_key="permit-before-disable-run",
    )

    async def consume():
        return [
            event
            async for event in first.execute_participant_session(
                execution, participant=participant, context=CONTEXT
            )
        ]

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(admitted.wait(), 10)
        await second.change_participant_lifecycle(
            change(
                initialized, participant, key="disable-after-permit", revision=1, state="disabled"
            ),
            context=CONTEXT,
        )
        release.set()
        events = await asyncio.wait_for(task, 10)
        assert any(event.type.value == "session.completed" for event in events)
        assert len(provider.requests) == 1
        assert (
            await first.list_participant_obligations(
                participant, context=CONTEXT, pending_only=True
            )
        ).obligations == ()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        runtime_store.apply_invocation_lifecycle_command = original_apply


@pytest.mark.parametrize("interruption", ["cancel", "timeout"])
async def test_public_execution_interruption_after_permit_keeps_pending_responsibility(
    interruption,
):
    collaboration = identity_tests.InMemoryCollaborationStore()
    session_store = InMemorySessionStore()
    reg = registration()
    application = app(collaboration, reg, session_store=session_store)
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    application.register_provider(provider, default=True)
    application.register_agent(AgentSpec(name="reviewer", model="model"))
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized, key="cancel-permit")
    participant = created.participants[0].reference
    creation = ParticipantSessionCreationRequest(
        creation_key="cancel-create",
        request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "start")]),
    )
    session, _ = await application.create_participant_session(
        creation, participant=participant, context=CONTEXT
    )
    entered, release = asyncio.Event(), asyncio.Event()
    runtime_store = application._runtime_session_store
    original_apply = runtime_store.apply_invocation_lifecycle_command

    async def paused_apply(command):
        entered.set()
        await release.wait()
        return await original_apply(command)

    runtime_store.apply_invocation_lifecycle_command = paused_apply
    execution = ParticipantSessionExecutionRequest(
        request=creation.request.model_copy(update={"session_id": session.id}),
        session_instance_id=session.instance_id,
        execution_key="cancel-run",
    )

    async def consume():
        async for _ in application.execute_participant_session(
            execution, participant=participant, context=CONTEXT
        ):
            pass

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(entered.wait(), 10)
        if interruption == "cancel":
            task.cancel("caller interruption")
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(task, 0.01)
        release.set()
        obligations = await application.list_participant_obligations(
            participant, context=CONTEXT, pending_only=True
        )
        assert len(obligations.obligations) == 1
        assert obligations.obligations[0].state == "pending"
        assert provider.requests == []
        runtime_store.apply_invocation_lifecycle_command = original_apply
        replayed = [
            event
            async for event in application.execute_participant_session(
                execution, participant=participant, context=CONTEXT
            )
        ]
        assert any(event.type.value == "session.completed" for event in replayed)
        assert len(provider.requests) == 1
        assert (
            await application.list_participant_obligations(
                participant, context=CONTEXT, pending_only=True
            )
        ).obligations == ()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_public_execution_registration_lost_ack_replays_exact_permit():
    collaboration = identity_tests.InMemoryCollaborationStore()
    session_store = InMemorySessionStore()
    reg = registration()
    application = app(collaboration, reg, session_store=session_store)
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    application.register_provider(provider, default=True)
    application.register_agent(AgentSpec(name="reviewer", model="model"))
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized, key="lost-permit-ack")
    participant = created.participants[0].reference
    creation = ParticipantSessionCreationRequest(
        creation_key="lost-create",
        request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "start")]),
    )
    session, _ = await application.create_participant_session(
        creation, participant=participant, context=CONTEXT
    )
    original = collaboration._register_permit
    raise_ack = True

    async def commit_then_raise(*args, **kwargs):
        nonlocal raise_ack
        result = await original(*args, **kwargs)
        if raise_ack:
            raise_ack = False
            raise ConnectionError("permit acknowledgement lost")
        return result

    collaboration._register_permit = commit_then_raise
    execution = ParticipantSessionExecutionRequest(
        request=creation.request.model_copy(update={"session_id": session.id}),
        session_instance_id=session.instance_id,
        execution_key="lost-run",
    )
    with pytest.raises(CollaborationUnavailable, match="acknowledgement"):
        [
            event
            async for event in application.execute_participant_session(
                execution, participant=participant, context=CONTEXT
            )
        ]
    replayed = [
        event
        async for event in application.execute_participant_session(
            execution, participant=participant, context=CONTEXT
        )
    ]
    assert any(event.type.value == "session.completed" for event in replayed)
    assert len(provider.requests) == 1
    obligations = await application.list_participant_obligations(
        participant, context=CONTEXT, pending_only=True
    )
    assert obligations.obligations == ()
