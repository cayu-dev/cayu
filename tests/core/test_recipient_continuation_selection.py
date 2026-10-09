"""Native CONTINUE identity comes from a real completed participant invocation."""

import asyncio
import threading
from contextlib import asynccontextmanager

import pytest
from tests.core.test_participant_identity import CONTEXT, app, create, registration
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_prepared_admission_public import prepared_scenario

from cayu.agents import AgentSpec
from cayu.collaboration._contracts import ExactMatch
from cayu.collaboration.prepared_admission import RecipientContinuationRequest
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions import ResumeRequest, RunRequest
from cayu.sessions._recipient_continuation import RecipientContinuationSelection
from cayu.sessions.base import InMemorySessionStore
from cayu.sessions.context_views import (
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
)

pytestmark = pytest.mark.anyio


async def prepared_continue_scenario(native_stores, *, queued_turn=False):
    application, resolver, original, provider, session, initialized = await prepared_scenario(
        native_stores,
        provider_events=[
            [ModelStreamEvent.text_delta("completed"), ModelStreamEvent.completed({})],
            [ModelStreamEvent.text_delta("continued"), ModelStreamEvent.completed({})],
        ]
        + (
            [[ModelStreamEvent.text_delta("queued"), ModelStreamEvent.completed({})]]
            if queued_turn
            else []
        ),
    )
    participant = original.prepared.recipient
    creation = ParticipantSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "input")]),
        creation_key="existing-root:" + initialized.owner.application_scope,
    )
    session, _ = await application.create_participant_session(
        creation, participant=participant, context=CONTEXT
    )
    events = [
        event
        async for event in application.execute_participant_session(
            ParticipantSessionExecutionRequest(
                request=creation.request.model_copy(update={"session_id": session.id}),
                session_instance_id=session.instance_id,
                execution_key="continue-preparation-source",
            ),
            participant=participant,
            context=CONTEXT,
        )
    ]
    assert any(event.type.value == "session.completed" for event in events)
    intent = RecipientContinuationRequest(
        participant=participant,
        session_id=session.id,
        session_instance_id=session.instance_id,
    )
    prepared = await application.prepare_recipient_continuation(intent, context=CONTEXT)
    command = original.model_copy(update={"decision": "continue", "prepared": prepared})
    return application, resolver, command, provider, session, initialized


async def test_public_continue_admission_exact_reader_and_inert_control(native_stores):
    from cayu.collaboration.requests import RequestControl

    (
        application,
        resolver,
        command,
        provider,
        session,
        initialized,
    ) = await prepared_continue_scenario(native_stores)
    before = (
        await application.session_store.load(session.id),
        await application.session_store.load_checkpoint(session.id),
        await application.session_store.load_transcript(session.id),
    )
    receipt = await application.admit_collaboration_request(
        command, context=resolver.recipient.context
    )
    assert receipt.state == "admitted"
    assert receipt.admission_permit is not None
    found = await application.collaboration_admission_reader().lookup(
        command, context=resolver.sender.context
    )
    assert isinstance(found, ExactMatch) and found.receipt == receipt
    assert (
        await application.admit_collaboration_request(command, context=resolver.recipient.context)
        == receipt
    )
    controlled = await application.control_collaboration_request(
        RequestControl(
            operation=initialized.operation("cancel-continue-selection"),
            expected=command.expected,
            expected_revision=receipt.revision,
            kind="cancel",
        ),
        context=resolver.sender.context,
    )
    assert controlled is not None
    assert len(provider.requests) == 1
    assert before == (
        await application.session_store.load(session.id),
        await application.session_store.load_checkpoint(session.id),
        await application.session_store.load_transcript(session.id),
    )


@pytest.mark.parametrize("already_admitted", [False, True])
async def test_public_continue_exact_boundary_is_not_silently_reselected(
    native_stores, already_admitted
):
    application, resolver, command, provider, session, _ = await prepared_continue_scenario(
        native_stores
    )
    if already_admitted:
        receipt = await application.admit_collaboration_request(
            command, context=resolver.recipient.context
        )
    events = [
        event
        async for event in application.resume(
            ResumeRequest(session_id=session.id, messages=[Message.text("user", "advance")]),
            context=CONTEXT,
        )
    ]
    assert any(event.type.value == "session.completed" for event in events)
    before = await application.session_store.load_checkpoint(session.id)
    if already_admitted:
        # Historical replay does not select a newer turn or renew launch rights.
        assert (
            await application.admit_collaboration_request(
                command, context=resolver.recipient.context
            )
            == receipt
        )
    else:
        with pytest.raises(ValueError):
            await application.admit_collaboration_request(
                command, context=resolver.recipient.context
            )
    assert len(provider.requests) == 2
    assert before == await application.session_store.load_checkpoint(session.id)


async def test_public_continue_disable_between_selection_and_admission(native_stores):
    from tests.core.test_participant_lifecycle import change

    (
        application,
        resolver,
        command,
        provider,
        session,
        initialized,
    ) = await prepared_continue_scenario(native_stores)
    before = await application.session_store.load_checkpoint(session.id)
    await application.change_participant_lifecycle(
        change(
            initialized,
            command.prepared.recipient,
            key="disable-selected-recipient",
            revision=1,
            state="disabled",
        ),
        context=CONTEXT,
    )
    with pytest.raises((ValueError, PermissionError)):
        await application.admit_collaboration_request(command, context=resolver.recipient.context)
    assert before == await application.session_store.load_checkpoint(session.id)
    assert len(provider.requests) == 1


@pytest.mark.parametrize("signal", ["cancel", "lost_ack"])
async def test_public_continue_lost_ack_reconstructs_without_native_target(
    native_stores, monkeypatch, signal
):
    from cayu.collaboration._contracts import ExactConflict
    from cayu.collaboration.participants import CollaborationUnavailable
    from cayu.collaboration.request_access import RequestRegistration

    application, resolver, command, provider, _, _ = await prepared_continue_scenario(native_stores)
    store = native_stores[0]
    original = store._transaction
    committed, release = asyncio.Event(), asyncio.Event()
    faulted = False
    key = (
        command.operation.namespace_incarnation,
        command.operation.generation,
        command.operation.caller_key,
    )

    @asynccontextmanager
    async def transaction(scope, *, write):
        nonlocal faulted
        interrupt = False
        async with original(scope, write=write) as tx:
            yield tx
            if write and not faulted and await tx.get("operations", key) is not None:
                interrupt = faulted = True
        if interrupt:
            committed.set()
            await release.wait()
            if signal == "lost_ack":
                raise ConnectionError("Admission acknowledgement lost.")

    monkeypatch.setattr(store, "_transaction", transaction)
    caller = asyncio.create_task(
        application.admit_collaboration_request(command, context=resolver.recipient.context)
    )
    try:
        await asyncio.wait_for(committed.wait(), 30)
        if signal == "cancel":
            caller.cancel()
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert caller.cancelled() and caller.cancelling() == 2
        release.set()
        if signal == "lost_ack":
            with pytest.raises(CollaborationUnavailable):
                await caller
        reconstructed = app(
            native_stores[2](),
            application._participant_coordinator._registration,
            collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=300_000),
        )
        await reconstructed.initialize_collaboration()
        found = await reconstructed.collaboration_admission_reader().lookup(
            command, context=resolver.sender.context
        )
        assert isinstance(found, ExactMatch)
        assert found.receipt.command == command
        selection = command.prepared.target.selection
        for field, changed in (
            ("session_id", "different-session"),
            ("session_instance_id", "different-incarnation"),
            ("run_epoch", selection.run_epoch + 1),
            ("participant_binding_sha256", "a" * 64),
            ("checkpoint_sha256", "b" * 64),
            ("release_identity", "different-release"),
            ("release_sha256", "c" * 64),
            ("interaction_id", "different-interaction"),
            ("model_step_id", "different-step"),
            ("completion_event_id", "different-completion"),
            ("transcript_end_cursor", selection.transcript_end_cursor + 1),
        ):
            conflicting = command.model_copy(
                update={
                    "prepared": command.prepared.model_copy(
                        update={
                            "target": command.prepared.target.model_copy(
                                update={"selection": selection.model_copy(update={field: changed})}
                            )
                        }
                    )
                }
            )
            assert isinstance(
                await reconstructed.collaboration_admission_reader().lookup(
                    conflicting, context=resolver.sender.context
                ),
                ExactConflict,
            )
        assert len(provider.requests) == 1
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)


async def completed_participant(
    native_stores, *, provider=None, tools=(), terminal_event="session.completed"
):
    collaboration, sessions, *_ = native_stores
    application = app(collaboration, registration(), session_store=sessions)
    application.register_provider(
        provider
        or ScriptedModelProvider(
            [[ModelStreamEvent.text_delta("completed"), ModelStreamEvent.completed({})]],
            name="provider",
        ),
        default=True,
    )
    application.register_agent(AgentSpec(name="reviewer", model="model"), tools=list(tools))
    initialized = await application.initialize_collaboration()
    _, receipt = await create(application, initialized)
    participant = receipt.participants[0].reference
    creation = ParticipantSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "start")]),
        creation_key="continue-source:" + initialized.owner.application_scope,
    )
    session, _ = await application.create_participant_session(
        creation, participant=participant, context=CONTEXT
    )
    events = [
        event
        async for event in application.execute_participant_session(
            ParticipantSessionExecutionRequest(
                request=creation.request.model_copy(update={"session_id": session.id}),
                session_instance_id=session.instance_id,
                execution_key="initial-execution",
            ),
            participant=participant,
            context=CONTEXT,
        )
    ]
    assert any(event.type.value == terminal_event for event in events)
    return application, session, participant


async def test_selection_cannot_convert_human_pause_into_ordinary_continuation(native_stores):
    from cayu.tools.user_input import UserInputTool

    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.tool_call(
                id="human-question", name="ask_user", arguments={"question": "Continue?"}
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ],
        name="provider",
    )
    application, session, _ = await completed_participant(
        native_stores,
        provider=provider,
        tools=(UserInputTool(),),
        terminal_event="session.awaiting_user_input",
    )
    store = application.session_store
    before = (
        await store.load(session.id),
        await store.load_checkpoint(session.id),
        await store.load_events(session.id),
    )
    with pytest.raises(ValueError):
        await store.capture_recipient_continuation(session.id)
    assert before == (
        await store.load(session.id),
        await store.load_checkpoint(session.id),
        await store.load_events(session.id),
    )
    assert len(provider.requests) == 1


async def test_existing_context_publication_keeps_historical_projection():
    from tests.core.test_context_views import (
        _test_public_context_view_publication_selection_and_readback,
    )

    await _test_public_context_view_publication_selection_and_readback()


async def test_selection_does_not_reinterpret_historical_turn_as_current_idle(native_stores):
    entered, release = asyncio.Event(), asyncio.Event()

    class PausingProvider(ScriptedModelProvider):
        async def stream(self, request):
            if self.requests:
                entered.set()
                await release.wait()
            async for event in super().stream(request):
                yield event

    provider = PausingProvider(
        response_factory=lambda _: [
            ModelStreamEvent.text_delta("completed"),
            ModelStreamEvent.completed({}),
        ],
        name="provider",
    )
    application, session, _ = await completed_participant(native_stores, provider=provider)
    store = application.session_store
    original = await store.capture_recipient_continuation(session.id)

    async def continue_session():
        return [
            event
            async for event in application.resume(
                ResumeRequest(session_id=session.id, messages=[Message.text("user", "next")]),
                context=CONTEXT,
            )
        ]

    caller = asyncio.create_task(continue_session())
    try:
        await asyncio.wait_for(entered.wait(), 20)
        historical = await store.capture_context_view_publication_source(session.id)
        assert historical.completion_event.id == original.completion_event_id
        before = await store.load_checkpoint(session.id)
        with pytest.raises(ValueError, match="quiescent completed turn"):
            await store.capture_recipient_continuation(session.id)
        assert before == await store.load_checkpoint(session.id)
        release.set()
        events = await asyncio.wait_for(caller, 20)
        assert any(event.type.value == "session.completed" for event in events), events
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)
    current = await store.capture_recipient_continuation(session.id)
    assert current.session_instance_id == original.session_instance_id
    assert current.run_epoch > original.run_epoch
    assert current.completion_event_id != original.completion_event_id
    assert current.release_identity != original.release_identity
    assert len(provider.requests) == 2


async def test_native_selection_is_exact_read_only_and_reconstructable(native_stores, tmp_path):
    application, session, participant = await completed_participant(native_stores)
    store = application.session_store
    before = (
        await store.load(session.id),
        await store.load_checkpoint(session.id),
        await store.load_events(session.id),
        await store.load_transcript(session.id),
    )
    selected = await store.capture_recipient_continuation(session.id)
    assert selected.participant == participant
    assert selected.session_instance_id == session.instance_id
    assert selected.run_epoch == before[0].run_epoch
    assert selected == await store.capture_recipient_continuation(session.id)
    assert selected == RecipientContinuationSelection.model_validate_json(
        selected.model_dump_json()
    )
    historical = await store.capture_context_view_publication_source(session.id)
    assert selected.completion_event_id == historical.completion_event.id
    assert selected.transcript_end_cursor == historical.transcript_end_cursor
    assert before == (
        await store.load(session.id),
        await store.load_checkpoint(session.id),
        await store.load_events(session.id),
        await store.load_transcript(session.id),
    )
    backend, address = native_stores[3]
    if backend == "memory":
        return
    if backend == "sqlite":
        from cayu.storage.sqlite import SQLiteSessionStore

        reconstructed = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    else:
        from cayu.storage.postgres import PostgresSessionStore

        reconstructed = PostgresSessionStore(address)
    try:
        assert selected == await reconstructed.capture_recipient_continuation(session.id)
    finally:
        await reconstructed.close()


async def test_unqualified_wrapper_does_not_inherit_selection_capability():
    class Wrapper(InMemorySessionStore):
        pass

    with pytest.raises(NotImplementedError, match="not qualified"):
        await Wrapper().capture_recipient_continuation("source")


@pytest.mark.parametrize("qualified", [False, True])
async def test_public_continue_requires_qualified_delegating_store(native_stores, qualified):
    from cayu.collaboration.participants import CollaborationUnavailable

    original, resolver, command, provider, session, _ = await prepared_continue_scenario(
        native_stores
    )
    source = original.session_store

    class Wrapper(InMemorySessionStore):
        async def capture_recipient_continuation(self, session_id):
            return await source.capture_recipient_continuation(session_id)

        async def load(self, session_id):
            return await source.load(session_id)

    class Qualified(Wrapper):
        recipient_continuation_selection_version = 1

    wrapped = app(
        native_stores[0],
        original._participant_coordinator._registration,
        session_store=Qualified() if qualified else Wrapper(),
        collaboration_requests=original._request_coordinator._registration,
        budget_binding_receiver=original.budget_binding_receiver,
        enable_common_root_budget_binding=True,
    )
    await wrapped.initialize_collaboration()
    intent = RecipientContinuationRequest(
        participant=command.prepared.recipient,
        session_id=session.id,
        session_instance_id=session.instance_id,
    )
    before = await original.inspect_collaboration_request(
        command.expected, context=resolver.recipient.context
    )
    if qualified:
        assert (
            await wrapped.prepare_recipient_continuation(intent, context=CONTEXT)
            == command.prepared
        )
        receipt = await wrapped.admit_collaboration_request(
            command, context=resolver.recipient.context
        )
        assert receipt.state == "admitted"
    else:
        with pytest.raises(NotImplementedError, match="not qualified"):
            await wrapped.prepare_recipient_continuation(intent, context=CONTEXT)
        with pytest.raises(CollaborationUnavailable):
            await wrapped.admit_collaboration_request(command, context=resolver.recipient.context)
        assert (
            await original.inspect_collaboration_request(
                command.expected, context=resolver.recipient.context
            )
            == before
        )
    assert len(provider.requests) == 1
    assert (await source.load(session.id)).status == "completed"


async def test_selection_rejects_input_appended_beyond_completed_frontier(native_stores):
    application, session, _ = await completed_participant(native_stores)
    store = application.session_store
    selected = await store.capture_recipient_continuation(session.id)
    # A native caller can append after release. The old completion remains valid
    # historical material, but no longer identifies the whole current frontier.
    await store.append_transcript_messages(session.id, [Message.text("user", "unconsumed")])
    source = await store.capture_context_view_publication_source(session.id)
    assert source.completion_event.id == selected.completion_event_id
    before = await store.load_transcript(session.id)
    with pytest.raises(ValueError, match="quiescent completed turn"):
        await store.capture_recipient_continuation(session.id)
    assert before == await store.load_transcript(session.id)


@pytest.mark.parametrize("occupancy", ["queue", "closure"])
async def test_public_continue_refuses_native_occupancy_without_admission(
    native_stores, monkeypatch, occupancy
):
    from cayu.collaboration.participants import CollaborationUnavailable
    from cayu.sessions.messaging import EnqueueSessionMessageRequest, SessionMessageDeliveryMode

    application, resolver, command, provider, session, _ = await prepared_continue_scenario(
        native_stores, queued_turn=occupancy == "queue"
    )
    store = application.session_store
    before = await application.inspect_collaboration_request(
        command.expected, context=resolver.recipient.context
    )
    entered, release = asyncio.Event(), asyncio.Event()
    closing = None
    if occupancy == "queue":
        stream = provider.stream

        async def enqueue_during_model_round(model_request):
            # Queues admit only pending/running sessions. Hold the actual running
            # owner with pending input; preparation must not steal its next turn.
            await store.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=session.id,
                    idempotency_key="queued-before-selection",
                    content="Steering for the next turn",
                    delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                )
            )
            entered.set()
            await release.wait()
            async for event in stream(model_request):
                yield event

        monkeypatch.setattr(provider, "stream", enqueue_during_model_round)

        async def continue_with_queue():
            return [
                event
                async for event in application.resume(
                    ResumeRequest(session_id=session.id, messages=[Message.text("user", "next")]),
                    context=CONTEXT,
                )
            ]

        closing = asyncio.create_task(continue_with_queue())
    else:
        claim = store.claim_session_closure_progress

        async def hold_claim(progress):
            await claim(progress)
            entered.set()
            await release.wait()

        monkeypatch.setattr(store, "claim_session_closure_progress", hold_claim)
        closing = asyncio.create_task(application.erase_session_closure(session.id))
    barrier = asyncio.create_task(entered.wait()) if closing is not None else None
    try:
        if closing is not None:
            done, _ = await asyncio.wait(
                (closing, barrier), timeout=30, return_when=asyncio.FIRST_COMPLETED
            )
            if closing in done:
                await closing
                pytest.fail("The native owner did not reach its occupancy barrier")
            assert barrier in done
        snapshot = await store._capture_completed_turn_snapshot(session.id)
        assert snapshot.has_queued_input if occupancy == "queue" else snapshot.has_closure_owner
        intent = RecipientContinuationRequest(
            participant=command.prepared.recipient,
            session_id=session.id,
            session_instance_id=session.instance_id,
        )
        with pytest.raises(ValueError, match="quiescent completed turn"):
            await application.prepare_recipient_continuation(intent, context=CONTEXT)
        with pytest.raises(CollaborationUnavailable):
            await application.admit_collaboration_request(
                command, context=resolver.recipient.context
            )
        assert (
            await application.inspect_collaboration_request(
                command.expected, context=resolver.recipient.context
            )
            == before
        )
        assert len(provider.requests) == 1
    finally:
        release.set()
        if barrier is not None:
            barrier.cancel()
            await asyncio.gather(barrier, return_exceptions=True)
        if closing is not None:
            result = await closing
            if occupancy == "queue":
                assert any(event.type.value == "session.completed" for event in result), result
                assert len(provider.requests) == 3


@pytest.mark.parametrize("native_stores", ["sqlite", "postgres"], indirect=True)
async def test_selection_snapshot_is_coherent_while_another_instance_advances(
    native_stores, tmp_path, monkeypatch
):
    provider = ScriptedModelProvider(
        response_factory=lambda _: [
            ModelStreamEvent.text_delta("completed"),
            ModelStreamEvent.completed({}),
        ],
        name="provider",
    )
    application, session, _ = await completed_participant(native_stores, provider=provider)
    expected = await application.session_store.capture_recipient_continuation(session.id)
    backend, address = native_stores[3]
    entered = asyncio.Event()
    release = threading.Event()
    async_release = asyncio.Event()
    first = True
    if backend == "sqlite":
        from cayu.storage.sqlite import SQLiteSessionStore

        reader = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        original_read = reader._run_read
        loop = asyncio.get_running_loop()

        class ReadConnection:
            def __init__(self, connection):
                self.connection = connection

            def __enter__(self):
                self.connection.__enter__()
                return self

            def __exit__(self, *args):
                return self.connection.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self.connection, name)

            def execute(self, sql, *args):
                nonlocal first
                result = self.connection.execute(sql, *args)
                if first and sql.startswith("SELECT") and "FROM cayu_sessions" in sql:
                    first = False
                    loop.call_soon_threadsafe(entered.set)
                    if not release.wait(30):
                        raise TimeoutError("Native snapshot barrier was not released.")
                return result

        async def held_read(query):
            return await original_read(lambda connection: query(ReadConnection(connection)))

        monkeypatch.setattr(reader, "_run_read", held_read)
    else:
        from cayu.storage.postgres import PostgresSessionStore

        reader = PostgresSessionStore(address)
        original_load = reader._load

        async def held_load(cur, session_id):
            nonlocal first
            result = await original_load(cur, session_id)
            if first:
                first = False
                entered.set()
                await async_release.wait()
            return result

        monkeypatch.setattr(reader, "_load", held_load)
    pending = asyncio.create_task(reader.capture_recipient_continuation(session.id))
    try:
        await asyncio.wait_for(entered.wait(), 20)
        events = [
            event
            async for event in application.resume(
                ResumeRequest(session_id=session.id, messages=[Message.text("user", "advance")]),
                context=CONTEXT,
            )
        ]
        assert any(event.type.value == "session.completed" for event in events)
        release.set()
        async_release.set()
        # Neither a new profile/frontier with an old release, nor an old session
        # with the new checkpoint, may leak from the consistent read transaction.
        assert await asyncio.wait_for(pending, 20) == expected
        current = await reader.capture_recipient_continuation(session.id)
        assert current.run_epoch > expected.run_epoch
        assert current.completion_event_id != expected.completion_event_id
    finally:
        release.set()
        async_release.set()
        await asyncio.gather(pending, return_exceptions=True)
        await reader.close()
