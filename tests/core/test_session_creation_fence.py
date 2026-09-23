from __future__ import annotations

import asyncio
import warnings
from dataclasses import replace
from uuid import uuid4

import pytest
from tests.core.test_participant_identity import CONTEXT, app, create, registration

from cayu.agents import AgentSpec
from cayu.collaboration._contracts import (
    CollaborationConflict,
    CollaborationContractError,
    ExactConflict,
    ExactMatch,
    ExactNotFound,
)
from cayu.collaboration.access import CollaborationAccessDenied
from cayu.collaboration.lifecycle import ParticipantLifecycleChange
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.base import EventQuery, InMemorySessionStore, RunRequest, SessionIdentity
from cayu.sessions.context_views import (
    ContextViewLimits,
    ContextViewOwnershipRequest,
    ContextViewPublicationRequest,
    ContextViewSelectionRequest,
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
    RecipientSessionCreationRequest,
)
from cayu.sessions.creation_fence import (
    _SESSION_CREATION_AUTHORITY,
    SessionCreationConflict,
    SessionCreationExcluded,
)


def _store_factory(backend, tmp_path, request):
    if backend == "memory":
        store = InMemorySessionStore()
        return lambda: store
    if backend == "sqlite":
        from cayu.storage.sqlite import SQLiteSessionStore

        return lambda: SQLiteSessionStore(tmp_path / "creation-fence.sqlite")
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    dsn = request.getfixturevalue("postgres_dsn")
    return lambda: PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)


async def _application(store, collaboration, registration_value, *, batches=()):
    application = app(collaboration, registration_value, session_store=store)
    provider = ScriptedModelProvider(list(batches))
    application.register_provider(provider, default=True)
    application.register_agent(AgentSpec(name="reviewer", model="model", system_prompt="system"))
    initialized = await application.initialize_collaboration()
    return application, initialized, provider


def _collaboration_factory(backend, tmp_path, request):
    if backend == "memory":
        store = InMemoryCollaborationStore()
        return lambda: store
    if backend == "sqlite":
        from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore

        return lambda: SQLiteCollaborationStore(tmp_path / "prepared-collaboration.sqlite")
    from cayu.storage.collaboration_postgres import PostgresCollaborationStore
    from cayu.storage.migrations import SchemaMode

    dsn = request.getfixturevalue("postgres_dsn")
    return lambda: PostgresCollaborationStore(dsn, schema_mode=SchemaMode.CREATE)


async def _fork_selection(application, sessions, participant):
    from cayu.runtime._model_completion_publication import model_step_publication_from_checkpoint

    key = uuid4().hex
    request = ParticipantSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "source")]),
        creation_key="fork-source-" + key,
    )
    session, _ = await application.create_participant_session(
        request, participant=participant, context=CONTEXT
    )
    execution = ParticipantSessionExecutionRequest(
        request=request.request.model_copy(update={"session_id": session.id}),
        session_instance_id=session.instance_id,
        execution_key="fork-execute-" + key,
    )
    async for _ in application.execute_participant_session(
        execution, participant=participant, context=CONTEXT
    ):
        pass
    pointer = model_step_publication_from_checkpoint(await sessions.load_checkpoint(session.id))
    assert pointer is not None
    completion = await sessions.query_events(
        EventQuery(session_id=session.id, event_id=pointer.completion_event_id, limit=1)
    )
    manifest = await application.publish_completed_context_view(
        ContextViewPublicationRequest(
            source_session_id=session.id,
            source_session_instance_id=session.instance_id,
            view_id="fork-view-" + key,
            interaction_id=completion[0].event.interaction_id,
            boundary_id=pointer.logical_step_id,
            projection_schema="whole-turn.v1",
            publication_key="fork-publication-" + key,
        ),
        participant=participant,
        context=CONTEXT,
    )
    selected = await application.select_context_view(
        ContextViewSelectionRequest(
            source_owner=participant.owner,
            source_session_id=session.id,
            source_session_instance_id=session.instance_id,
            selector="latest",
            projection_schema=manifest.projection_schema,
            extension_set_commitment=manifest.extension_set_commitment,
            limits=ContextViewLimits(),
            selection_key="fork-selection-" + key,
        ),
        participant=participant,
        context=CONTEXT,
    )
    return await application.transition_context_view_ownership(
        ContextViewOwnershipRequest(
            selection_key=selected.selection_key,
            view_id=manifest.view_id,
            pin_commitment=selected.pin_commitment,
            expected_state="selected",
            expected_revision=selected.ownership_revision,
            operation="adopt",
            current_owner=participant.owner,
            destination_owner=participant.owner,
            operation_key="fork-adopt-" + key,
        ),
        participant=participant,
        destination_participant=participant,
        context=CONTEXT,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("winner", ["creation", "exclusion"])
@pytest.mark.parametrize("mode", ["fresh", "fork"])
async def test_public_recipient_creation_fence_orders_exact_decisions(
    backend, winner, mode, tmp_path, request, monkeypatch, caplog, capsys
):
    factory = _store_factory(backend, tmp_path, request)
    sessions, competitor = factory(), factory()
    collaboration = InMemoryCollaborationStore()
    application, initialized, provider = await _application(
        sessions,
        collaboration,
        registration(),
        batches=(
            [
                ModelStreamEvent.text_delta("retained history"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        )
        if mode == "fork"
        else (),
    )
    _, participant = await create(application, initialized, key="fence-participant")
    recipient = participant.participants[0].reference
    selection = await _fork_selection(application, sessions, recipient) if mode == "fork" else None
    requests_before = len(provider.requests)
    creation = RecipientSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "input")]),
        creation_key="fence-child-" + uuid4().hex,
        recipient=recipient,
        mode=mode,
        selected_view=selection,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    targets = []
    original = sessions.create_participant_owned_session

    async def paused(*args, **kwargs):
        targets.append(kwargs["creation_target"])
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(sessions, "create_participant_owned_session", paused)
    task = asyncio.create_task(application.create_recipient_session(creation, context=CONTEXT))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        target = targets[0]
        canary = "secret-canary-invalid-target"

        class Opaque:
            def __repr__(self):
                return canary

            def __str__(self):
                return canary

        malformed = target.model_copy(update={"receiving_owner": Opaque(), "creation_key": canary})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(CollaborationContractError) as rejected:
                await competitor.read_session_creation_decision(malformed)
        assert rejected.value.__context__ is None and rejected.value.__cause__ is None
        assert canary not in str(rejected.value) + repr(rejected.value) + caplog.text
        assert all(canary not in str(warning.message) for warning in caught)
        output = capsys.readouterr()
        assert canary not in output.out + output.err
        pending = await competitor.read_session_creation_decision(target)
        assert isinstance(pending, ExactMatch) and pending.receipt.state == "pending"
        page = await competitor.list_pending_session_creations(target.receiving_owner, limit=1)
        assert page.decisions == (pending.receipt,)
        assert page.next_cursor is None
        if winner == "exclusion":
            excluded = await competitor._exclude_session_creation_target(
                target, authority=_SESSION_CREATION_AUTHORITY
            )
            assert excluded.state == "excluded"
            release.set()
            with pytest.raises(SessionCreationExcluded):
                await asyncio.wait_for(task, 10)
            assert await application.lookup_recipient_session(creation, context=CONTEXT) is None
            inspection = await application.inspect_participant(creation.recipient, context=CONTEXT)
            assert inspection.outstanding_obligations == 0
        else:
            release.set()
            session, receipt = await asyncio.wait_for(task, 10)
            decision = await competitor._exclude_session_creation_target(
                target, authority=_SESSION_CREATION_AUTHORITY
            )
            assert decision.state == "created"
            assert (decision.session_id, decision.session_instance_id) == (
                session.id,
                session.instance_id,
            )
            assert await application.create_recipient_session(creation, context=CONTEXT) == (
                session,
                receipt,
            )
        for field in (
            "creation_key",
            "request_commitment",
            "material_commitment",
            "execution_identity_commitment",
            "requested_session_id",
        ):
            changed = target.model_copy(update={field: "conflicting-value"})
            assert isinstance(
                await competitor.read_session_creation_decision(changed), ExactConflict
            )
            with pytest.raises(SessionCreationConflict):
                await competitor._register_session_creation_target(
                    changed, authority=_SESSION_CREATION_AUTHORITY
                )
        assert not (
            await competitor.list_pending_session_creations(target.receiving_owner)
        ).decisions
        assert len(provider.requests) == requests_before
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if backend != "memory":
            await sessions.close()
            await competitor.close()


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_public_cancelled_future_target_survives_restart_and_isolates_public_id(
    backend, tmp_path, request, monkeypatch
):
    factory = _store_factory(backend, tmp_path, request)
    sessions = factory()
    collaboration, config = InMemoryCollaborationStore(), registration()
    application, initialized, provider = await _application(sessions, collaboration, config)
    _, participant = await create(application, initialized, key="cancel-fence-participant")
    public_id = "shared-public-id-" + uuid4().hex
    creation = RecipientSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", session_id=public_id, messages=[]),
        creation_key="cancelled-child-" + uuid4().hex,
        recipient=participant.participants[0].reference,
    )
    entered = asyncio.Event()
    targets = []

    async def blocked(*args, **kwargs):
        targets.append(kwargs["creation_target"])
        entered.set()
        await asyncio.Future()

    with monkeypatch.context() as patch:
        patch.setattr(sessions, "create_participant_owned_session", blocked)
        task = asyncio.create_task(application.create_recipient_session(creation, context=CONTEXT))
        await asyncio.wait_for(entered.wait(), 10)
        task.cancel()
        assert task.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        assert task.cancelled()
    target = targets[0]
    if backend != "memory":
        await sessions.close()
    reopened = factory()
    recovered, _, _ = await _application(reopened, collaboration, config)
    try:
        page = await reopened.list_pending_session_creations(target.receiving_owner)
        assert page.decisions[0].target == target
        await reopened._exclude_session_creation_target(
            target, authority=_SESSION_CREATION_AUTHORITY
        )
        with pytest.raises(SessionCreationExcluded):
            await recovered.create_recipient_session(creation, context=CONTEXT)
        replacement = replace(creation, creation_key="unrelated-child-" + uuid4().hex)
        session, _ = await recovered.create_recipient_session(replacement, context=CONTEXT)
        assert session.id == public_id
        decision = await reopened.read_session_creation_decision(target)
        assert isinstance(decision, ExactMatch) and decision.receipt.state == "excluded"
        other_operation = target.permit.intent.request.source_operation.model_copy(
            update={"caller_key": "never-registered"}
        )
        other_registration = target.permit.intent.request.model_copy(
            update={"source_operation": other_operation}
        )
        other_permit = target.permit.model_copy(
            update={
                "intent": target.permit.intent.model_copy(update={"request": other_registration})
            }
        )
        absent = target.model_copy(update={"permit": other_permit})
        assert isinstance(await reopened.read_session_creation_decision(absent), ExactNotFound)
        assert provider.requests == []
    finally:
        if backend != "memory":
            await reopened.close()


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("winner", ["admission", "disablement"])
async def test_public_creation_responsibility_orders_disablement_and_lost_ack(
    backend, winner, tmp_path, request, monkeypatch
):
    session_factory = _store_factory(backend, tmp_path, request)
    sessions, other_sessions = session_factory(), session_factory()
    if backend == "memory":
        collaboration = other_collaboration = InMemoryCollaborationStore()
    elif backend == "sqlite":
        from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore

        collaboration = SQLiteCollaborationStore(tmp_path / "collaboration.sqlite")
        other_collaboration = SQLiteCollaborationStore(tmp_path / "collaboration.sqlite")
    else:
        from cayu.storage.collaboration_postgres import PostgresCollaborationStore
        from cayu.storage.migrations import SchemaMode

        dsn = request.getfixturevalue("postgres_dsn")
        collaboration = PostgresCollaborationStore(dsn, schema_mode=SchemaMode.CREATE)
        other_collaboration = PostgresCollaborationStore(dsn, schema_mode=SchemaMode.CREATE)
    config = registration()
    application, initialized, provider = await _application(sessions, collaboration, config)
    other, _, other_provider = await _application(other_sessions, other_collaboration, config)
    _, participant_receipt = await create(application, initialized, key="disable-owner")
    participant = participant_receipt.participants[0].reference
    creation = RecipientSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[]),
        creation_key="disable-child-" + uuid4().hex,
        recipient=participant,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    original_create = sessions.create_participant_owned_session
    original_register = collaboration._register_permit

    async def register(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original_register(*args, **kwargs)

    async def commit_then_lose_ack(*args, **kwargs):
        entered.set()
        await release.wait()
        await original_create(*args, **kwargs)
        raise ConnectionError("creation acknowledgement lost")

    if winner == "disablement":
        monkeypatch.setattr(collaboration, "_register_permit", register)
    else:
        monkeypatch.setattr(sessions, "create_participant_owned_session", commit_then_lose_ack)
    task = asyncio.create_task(application.create_recipient_session(creation, context=CONTEXT))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        before = await other.inspect_participant(participant, context=CONTEXT)
        assert before.outstanding_obligations == (1 if winner == "admission" else 0)
        disabled = await other.change_participant_lifecycle(
            ParticipantLifecycleChange(
                operation=initialized.operation("disable-during-creation"),
                participant=participant,
                expected_lifecycle_revision=1,
                state="disabled",
            ),
            context=CONTEXT,
        )
        assert disabled.participant.covered_permit_frontier == before.issued_permit_frontier
        release.set()
        if winner == "disablement":
            with pytest.raises((CollaborationConflict, CollaborationAccessDenied)):
                await asyncio.wait_for(task, 10)
            assert await other.lookup_recipient_session(creation, context=CONTEXT) is None
            page = await other_sessions.list_pending_session_creations(participant.owner)
            assert len(page.decisions) == 1
            assert not page.decisions[0].responsibility_registered
            await other_sessions._exclude_session_creation_target(
                page.decisions[0].target, authority=_SESSION_CREATION_AUTHORITY
            )
        else:
            with pytest.raises(ConnectionError, match="acknowledgement lost"):
                await asyncio.wait_for(task, 10)
            unresolved = await other.inspect_participant(participant, context=CONTEXT)
            assert unresolved.outstanding_obligations == 1
            # Another application reconstructs the admitted authority after the
            # acknowledged lifecycle change; it does not create new authority.
            session, receipt = await other.create_recipient_session(creation, context=CONTEXT)
            assert receipt.participant_receipt.binding.lifecycle_revision == 1
            assert await other.create_recipient_session(creation, context=CONTEXT) == (
                session,
                receipt,
            )
        after = await other.inspect_participant(participant, context=CONTEXT)
        assert after.participant.lifecycle == "disabled"
        assert after.outstanding_obligations == 0
        assert provider.requests == other_provider.requests == []
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if backend != "memory":
            await sessions.close()
            await other_sessions.close()
            await collaboration.close()
            await other_collaboration.close()


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("registered_before_cancel", [False, True])
async def test_prepared_target_recovers_cross_store_registration_gap(
    backend, registered_before_cancel, tmp_path, request, monkeypatch
):
    session_factory = _store_factory(backend, tmp_path, request)
    collaboration_factory = _collaboration_factory(backend, tmp_path, request)
    sessions, collaboration = session_factory(), collaboration_factory()
    config = registration()
    application, initialized, provider = await _application(sessions, collaboration, config)
    _, participant_receipt = await create(application, initialized, key="prepared-owner")
    participant = participant_receipt.participants[0].reference
    creation = RecipientSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "lost input")]),
        creation_key="prepared-child-" + uuid4().hex,
        recipient=participant,
    )
    entered = asyncio.Event()
    original_register = collaboration._register_permit

    async def blocked_registration(*args, **kwargs):
        if registered_before_cancel:
            await original_register(*args, **kwargs)
        entered.set()
        await asyncio.Future()

    with monkeypatch.context() as patch:
        patch.setattr(collaboration, "_register_permit", blocked_registration)
        task = asyncio.create_task(application.create_recipient_session(creation, context=CONTEXT))
        await asyncio.wait_for(entered.wait(), 10)
        task.cancel()
        assert task.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        assert task.cancelled()
    if backend != "memory":
        await sessions.close()
        await collaboration.close()
    reopened, recovered_collaboration = session_factory(), collaboration_factory()
    recovered, recovered_initialization, _ = await _application(
        reopened, recovered_collaboration, config
    )
    try:
        # Recovery needs only the registered owner, not the lost RunRequest.
        pending = await reopened.list_pending_session_creations(participant.owner)
        assert len(pending.decisions) == 1
        decision = pending.decisions[0]
        target = decision.target
        assert not decision.responsibility_registered
        assert target.request_commitment == creation.participant_request.request_commitment
        assert target.material_commitment and target.execution_identity_commitment
        with pytest.raises(PermissionError, match="trusted receiving owner"):
            await reopened._register_session_creation_target(target, authority=object())

        def unexpected_binding(_session):
            raise AssertionError("An unadmitted target reached child preparation.")

        with pytest.raises(PermissionError, match="has not been registered"):
            await reopened.create(
                RunRequest(agent_name="reviewer", messages=[]),
                identity=SessionIdentity(provider_name="provider", model="model"),
                creation_target=target,
                participant_request_commitment=target.request_commitment,
                participant_binding_factory=unexpected_binding,
            )
        inspection = await recovered.inspect_participant(participant, context=CONTEXT)
        assert inspection.outstanding_obligations == int(registered_before_cancel)
        excluded = await reopened._exclude_session_creation_target(
            target, authority=_SESSION_CREATION_AUTHORITY
        )
        assert excluded.state == "excluded"
        # An exact source registration that completes late (or lost its ACK)
        # cannot turn the retained whole-target exclusion back into admission.
        await recovered_collaboration._register_permit(
            recovered_initialization, target.permit, redactor=recovered._secret_redactor
        )
        assert (
            await reopened._register_session_creation_target(
                target, authority=_SESSION_CREATION_AUTHORITY
            )
            == excluded
        )
        from cayu.sessions._recipient_admission import settle_recipient_creation

        await settle_recipient_creation(recovered, target)
        inspection = await recovered.inspect_participant(participant, context=CONTEXT)
        assert inspection.outstanding_obligations == 0
        assert inspection.issued_permit_frontier == 1
        assert await recovered.lookup_recipient_session(creation, context=CONTEXT) is None
        assert provider.requests == []
    finally:
        if backend != "memory":
            await reopened.close()
            await recovered_collaboration.close()


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("entrance", ["create_participant_owned_session", "create"])
async def test_public_recipient_wrapper_cannot_drop_creation_fence(
    backend, entrance, tmp_path, request, monkeypatch
):
    sessions = _store_factory(backend, tmp_path, request)()
    application, initialized, provider = await _application(
        sessions, InMemoryCollaborationStore(), registration()
    )
    _, participant = await create(application, initialized, key="wrapper-owner")
    creation = RecipientSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[]),
        creation_key="wrapper-child-" + uuid4().hex,
        recipient=participant.participants[0].reference,
    )
    original = getattr(sessions, entrance)

    async def dropping_wrapper(*args, **kwargs):
        kwargs.pop("creation_target", None)
        return await original(*args, **kwargs)

    monkeypatch.setattr(sessions, entrance, dropping_wrapper)
    try:
        with pytest.raises(PermissionError, match="requires its exact durable target"):
            await application.create_recipient_session(creation, context=CONTEXT)
        assert await application.lookup_recipient_session(creation, context=CONTEXT) is None
        pending = await sessions.list_pending_session_creations(creation.recipient.owner)
        assert len(pending.decisions) == 1
        assert pending.decisions[0].responsibility_registered
        assert provider.requests == []
    finally:
        if backend != "memory":
            await sessions.close()


class _DeliveryTransactionConsumer:
    """Conformance consumer of the shared transaction, not a peer implementation."""

    def __init__(self, store, backend, memory_records):
        self.store, self.backend, self.memory_records = store, backend, memory_records

    async def decide(self, target, obligation, *, exclude=False):
        from cayu.sessions.creation_fence import decide
        from cayu.storage import _creation_fence

        def result(current, previous):
            if previous is not None:
                return tuple(previous)
            if exclude:
                return ("excluded", None)
            if current.state != "created":
                return ("pending", None)
            return ("appended", current.session_instance_id)

        if self.backend == "memory":
            async with self.store._lock:
                current = self.store._session_creation_decisions[target.key]
                decide(target, current)
                key = (target.key, obligation)
                value = result(current, self.memory_records.get(key))
                if value[0] != "pending":
                    self.memory_records[key] = value
                return value

        ddl = """CREATE TABLE IF NOT EXISTS cayu_test_creation_delivery_decisions (
            target_key TEXT NOT NULL, obligation TEXT NOT NULL,
            state TEXT NOT NULL, instance_id TEXT,
            PRIMARY KEY (target_key, obligation))"""
        if self.backend == "sqlite":

            def transaction(connection):
                connection.execute("BEGIN IMMEDIATE")
                try:
                    connection.execute(ddl)
                    current = _creation_fence.sqlite_read(connection, target)
                    assert current is not None
                    row = connection.execute(
                        "SELECT state, instance_id FROM cayu_test_creation_delivery_decisions "
                        "WHERE target_key = ? AND obligation = ?",
                        (target.key, obligation),
                    ).fetchone()
                    value = result(current, row)
                    if row is None and value[0] != "pending":
                        connection.execute(
                            "INSERT INTO cayu_test_creation_delivery_decisions VALUES (?, ?, ?, ?)",
                            (target.key, obligation, *value),
                        )
                    connection.commit()
                    return value
                except BaseException:
                    connection.rollback()
                    raise

            return await self.store._run_write(transaction)

        await self.store._ensure_ready()
        async with self.store._connection() as connection, connection.cursor() as cur:
            await cur.execute(ddl)
            await _creation_fence.postgres_lock(cur, target)
            current = await _creation_fence.postgres_read(cur, target)
            assert current is not None
            await cur.execute(
                "SELECT state, instance_id FROM cayu_test_creation_delivery_decisions "
                "WHERE target_key = %s AND obligation = %s",
                (target.key, obligation),
            )
            row = await cur.fetchone()
            value = result(current, row)
            if row is None and value[0] != "pending":
                await cur.execute(
                    "INSERT INTO cayu_test_creation_delivery_decisions VALUES (%s, %s, %s, %s)",
                    (target.key, obligation, *value),
                )
            await connection.commit()
            return value


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_shared_creation_transaction_preserves_separate_delivery_exclusion(
    backend, tmp_path, request, monkeypatch
):
    factory = _store_factory(backend, tmp_path, request)
    sessions, reopened = factory(), factory()
    application, initialized, provider = await _application(
        sessions, InMemoryCollaborationStore(), registration()
    )
    _, participant = await create(application, initialized, key="delivery-fence-owner")
    creation = RecipientSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[]),
        creation_key="delivery-fence-child-" + uuid4().hex,
        recipient=participant.participants[0].reference,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    targets = []
    create_child = sessions.create_participant_owned_session

    async def paused(*args, **kwargs):
        targets.append(kwargs["creation_target"])
        entered.set()
        await release.wait()
        return await create_child(*args, **kwargs)

    monkeypatch.setattr(sessions, "create_participant_owned_session", paused)
    task = asyncio.create_task(application.create_recipient_session(creation, context=CONTEXT))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        target = targets[0]
        memory_records = {}
        consumer = _DeliveryTransactionConsumer(sessions, backend, memory_records)
        assert await consumer.decide(target, "withdrawn-interest", exclude=True) == (
            "excluded",
            None,
        )
        pending = await sessions.read_session_creation_decision(target)
        assert isinstance(pending, ExactMatch) and pending.receipt.state == "pending"
        release.set()
        session, _ = await asyncio.wait_for(task, 10)
        recovered = _DeliveryTransactionConsumer(reopened, backend, memory_records)
        assert await recovered.decide(target, "withdrawn-interest") == ("excluded", None)
        assert await recovered.decide(target, "another-interest") == (
            "appended",
            session.instance_id,
        )
        assert await recovered.decide(target, "another-interest", exclude=True) == (
            "appended",
            session.instance_id,
        )
        assert provider.requests == []
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if backend != "memory":
            await sessions.close()
            await reopened.close()
