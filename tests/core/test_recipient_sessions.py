from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace

import pytest
from tests.artifacts.test_resources import (
    LocalArtifactResourceOwner,
    _transfer_command,
    authorized,
    command,
    make_store,
    registered_transfer,
)
from tests.artifacts.test_resources import (
    owner as resource_owner,
)
from tests.core.test_participant_identity import CONTEXT, app, create, registration

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.agents import AgentSpec
from cayu.artifacts import ArtifactScope
from cayu.artifacts.attachments import file_attachment
from cayu.artifacts.resources import ResourceTransferReceipt
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.environments import Environment, EnvironmentSpec
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import FilePart, Message, TextPart, ToolResultPart
from cayu.sessions.base import InMemorySessionStore, RunRequest
from cayu.sessions.context_views import (
    ParticipantSessionCreationRequest,
    RecipientSessionCreationReceipt,
    RecipientSessionCreationRequest,
)


class _BlockingRecipientStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def create_participant_owned_session(self, *args, **kwargs):
        self.started.set()
        await self.release.wait()
        return await super().create_participant_owned_session(*args, **kwargs)


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("character", ["x", "é"])
async def test_recipient_creation_key_byte_boundaries(backend, character, tmp_path, request):
    collaboration = InMemoryCollaborationStore()
    if backend == "memory":
        sessions = InMemorySessionStore()
    elif backend == "sqlite":
        from cayu.storage.sqlite import SQLiteSessionStore

        sessions = SQLiteSessionStore(tmp_path / "key-boundaries.sqlite")
    else:
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        sessions = PostgresSessionStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )
    try:
        application = app(collaboration, registration(), session_store=sessions)
        provider = ScriptedModelProvider([])
        application.register_provider(provider, default=True)
        application.register_agent(AgentSpec(name="reviewer", model="model"))
        initialized = await application.initialize_collaboration()
        _, participant = await create(application, initialized, key="key-boundary-owner")
        run_request = RunRequest(agent_name="reviewer", messages=[Message.text("user", "input")])

        def key_with_bytes(size):
            count, remainder = divmod(size, len(character.encode("utf-8")))
            key = character * count + "x" * remainder
            assert len(key.encode("utf-8")) == size
            return key

        # The underlying participant API retains its independent 256-byte bound.
        assert ParticipantSessionCreationRequest(
            request=run_request, creation_key=key_with_bytes(256)
        ).creation_key == key_with_bytes(256)
        for size in (247, 256, 257):
            with pytest.raises(ValueError, match="at most 246 UTF-8 bytes"):
                RecipientSessionCreationRequest(
                    request=run_request,
                    creation_key=key_with_bytes(size),
                    recipient=participant.participants[0].reference,
                )
        for size in (245, 246):
            creation = RecipientSessionCreationRequest(
                request=run_request,
                creation_key=key_with_bytes(size),
                recipient=participant.participants[0].reference,
            )
            assert len(creation.participant_request.creation_key.encode("utf-8")) == size + 10
            assert await application.lookup_recipient_session(creation, context=CONTEXT) is None
            created = await application.create_recipient_session(creation, context=CONTEXT)
            assert await application.create_recipient_session(creation, context=CONTEXT) == created
            assert await application.lookup_recipient_session(creation, context=CONTEXT) == created
            assert await sessions.load_transcript(created[0].id) == run_request.messages
        assert provider.requests == []
    finally:
        if backend != "memory":
            await sessions.close()
        await collaboration.close()


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_registered_resource_preparation_is_bound_to_exact_recipient(
    backend, tmp_path, request
):
    store, artifact = await asyncio.to_thread(make_store, tmp_path)
    source_ref = resource_owner("participant-bound-source")
    source = LocalArtifactResourceOwner(tmp_path / "source", owner=source_ref, artifact_store=store)
    owned = await authorized(source, command(source_ref, artifact.id, store=store))
    (
        destination,
        transfer,
        _,
        _,
        collaboration,
        initialized,
        participant,
    ) = await registered_transfer(tmp_path, store, artifact, source, owned)
    if backend == "memory":
        sessions = InMemorySessionStore()
    elif backend == "sqlite":
        from cayu.storage.sqlite import SQLiteSessionStore

        sessions = SQLiteSessionStore(tmp_path / "recipient-bound.sqlite")
    else:
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        sessions = PostgresSessionStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )
    application = app(
        collaboration, registration(scope=source_ref.application_scope), session_store=sessions
    )
    provider = ScriptedModelProvider([])
    application.register_provider(provider, default=True)
    application.register_agent(AgentSpec(name="reviewer", model="model", system_prompt="system"))
    await application.initialize_collaboration()
    try:
        accepted = await destination.accept_transfer(transfer, source_owner=source)
        preparation = await destination.read_preparation(transfer)
        assert preparation.permit.intent.request.participant == participant
        _, other = await create(application, initialized, key="other-recipient")
        other_participant = other.participants[0].reference
        assert other_participant.owner == participant.owner
        creation = RecipientSessionCreationRequest(
            request=RunRequest(agent_name="reviewer", messages=[]),
            creation_key="participant-bound-child",
            recipient=participant,
            resource_transfers=(accepted,),
            preparation_receipts=(preparation,),
        )
        wrong = replace(creation, recipient=other_participant)
        with pytest.raises(PermissionError, match="another recipient"):
            await application.create_recipient_session(
                wrong, context=CONTEXT, resource_owner=destination
            )
        assert await application.lookup_recipient_session(wrong, context=CONTEXT) is None
        session, receipt = await application.create_recipient_session(
            creation, context=CONTEXT, resource_owner=destination
        )
        assert receipt.recipient == participant
        assert await application.lookup_recipient_session(creation, context=CONTEXT) == (
            session,
            receipt,
        )
        assert provider.requests == []
    finally:
        if backend != "memory":
            await sessions.close()
        await collaboration.close()


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_recipient_creation_and_lookup_detach_input_before_waiting(
    backend, tmp_path, request, monkeypatch
) -> None:
    if backend == "memory":
        session_store = InMemorySessionStore()
    elif backend == "sqlite":
        from cayu.storage.sqlite import SQLiteSessionStore

        session_store = SQLiteSessionStore(tmp_path / "mutation.sqlite")
    else:
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        session_store = PostgresSessionStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )
    application = app(InMemoryCollaborationStore(), registration(), session_store=session_store)
    provider = ScriptedModelProvider([])
    application.register_provider(provider, default=True)
    application.register_agent(AgentSpec(name="reviewer", model="model", system_prompt="system"))
    initialized = await application.initialize_collaboration()
    _, participant_receipt = await create(application, initialized, key="mutation-owner")
    original = RecipientSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "input A")]),
        creation_key="mutation-child",
        recipient=participant_receipt.participants[0].reference,
    )
    caller_input = replace(original)
    lock = application._participant_session_authority_lock
    waiting = asyncio.Event()

    class ObservedLock:
        async def __aenter__(self):
            waiting.set()
            await lock.acquire()

        async def __aexit__(self, *args):
            lock.release()

    with monkeypatch.context() as patch:
        patch.setattr(application, "_participant_session_authority_lock", ObservedLock())
        await lock.acquire()
        task = asyncio.create_task(
            application.create_recipient_session(caller_input, context=CONTEXT)
        )
        try:
            await asyncio.wait_for(waiting.wait(), 10)
            assert not task.done()
            caller_input.request.messages[:] = [Message.text("user", "input B")]
        finally:
            lock.release()
        session, receipt = await asyncio.wait_for(task, 10)
    assert await session_store.load_transcript(session.id) == original.request.messages
    assert receipt.participant_receipt.recipient_metadata_json == original.metadata_json
    assert receipt.request_commitment == original.participant_request.request_commitment
    assert await application.create_recipient_session(original, context=CONTEXT) == (
        session,
        receipt,
    )
    assert await application.lookup_recipient_session(original, context=CONTEXT) == (
        session,
        receipt,
    )
    with pytest.raises(ValueError, match="conflicts"):
        await application.create_recipient_session(caller_input, context=CONTEXT)

    # Lookup must retain the same expectation across the store read as well.
    caller_lookup = replace(original)
    lookup_started = asyncio.Event()
    lookup_release = asyncio.Event()
    lookup = session_store.lookup_participant_session_creation

    async def blocked_lookup(creation):
        found = await lookup(creation)
        lookup_started.set()
        await lookup_release.wait()
        return found

    with monkeypatch.context() as patch:
        patch.setattr(session_store, "lookup_participant_session_creation", blocked_lookup)
        task = asyncio.create_task(
            application.lookup_recipient_session(caller_lookup, context=CONTEXT)
        )
        try:
            await asyncio.wait_for(lookup_started.wait(), 10)
            caller_lookup.request.messages[:] = [Message.text("user", "input B")]
        finally:
            lookup_release.set()
        assert await asyncio.wait_for(task, 10) == (session, receipt)
    with pytest.raises(ValueError, match="conflicts"):
        await application.lookup_recipient_session(caller_lookup, context=CONTEXT)
    assert provider.requests == []


class _CommitThenRaiseRecipientStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.raise_once = True

    async def create_participant_owned_session(self, *args, **kwargs):
        result = await super().create_participant_owned_session(*args, **kwargs)
        if self.raise_once:
            self.raise_once = False
            raise ConnectionError("acknowledgement lost")
        return result


class _BlockingHandoffOwner(LocalArtifactResourceOwner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handoff_started = asyncio.Event()
        self.handoff_release = asyncio.Event()

    async def release_transferred_source(self, transfer, *, destination_owner):
        self.handoff_started.set()
        await self.handoff_release.wait()
        return await super().release_transferred_source(
            transfer, destination_owner=destination_owner
        )

    async def _release_transferred_source_internal(self, transfer, *, destination_owner):
        self.handoff_started.set()
        await self.handoff_release.wait()
        return await super()._release_transferred_source_internal(
            transfer, destination_owner=destination_owner
        )


class _FailSecondHandoffOwner(LocalArtifactResourceOwner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handoff_count = 0
        self.fail_once = True

    async def _release_transferred_source_internal(self, transfer, *, destination_owner):
        self.handoff_count += 1
        if self.handoff_count == 2 and self.fail_once:
            self.fail_once = False
            raise OSError("second source cleanup failed")
        return await super()._release_transferred_source_internal(
            transfer, destination_owner=destination_owner
        )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_fresh_recipient_creation_is_inert_and_replays_exactly(
    backend: str, tmp_path, request
) -> None:
    collaboration_store = InMemoryCollaborationStore()
    if backend == "memory":
        session_store = InMemorySessionStore()
    elif backend == "sqlite":
        from cayu.storage.sqlite import SQLiteSessionStore

        session_store = SQLiteSessionStore(tmp_path / "recipient.sqlite")
    else:
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        session_store = PostgresSessionStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )
    application = app(collaboration_store, registration(), session_store=session_store)
    provider = ScriptedModelProvider([])
    application.register_provider(provider, default=True)
    application.register_agent(AgentSpec(name="reviewer", model="model", system_prompt="system"))
    initialized = await application.initialize_collaboration()
    _, participant_receipt = await create(application, initialized, key="recipient-owner")
    participant = participant_receipt.participants[0].reference
    creation = RecipientSessionCreationRequest(
        request=RunRequest(
            agent_name="reviewer",
            messages=[Message(role="user", content=(TextPart(text="bounded input"),))],
        ),
        creation_key="recipient-child",
        recipient=participant,
    )

    session, receipt = await application.create_recipient_session(creation, context=CONTEXT)
    replay, replay_receipt = await application.create_recipient_session(creation, context=CONTEXT)

    assert replay.id == session.id
    assert replay_receipt == receipt
    assert receipt.mode == "fresh"
    assert receipt.recipient == participant
    assert await session_store.load_transcript(session.id) == creation.request.messages
    assert await session_store.load_transcript_cursor(session.id) == len(creation.request.messages)
    assert provider.requests == []
    # Recipient provenance cannot be smuggled through the ordinary root
    # creation entrance, even when the caller has the exact derived request.
    with pytest.raises(TypeError, match="Recipient provenance"):
        await application.create_participant_session(
            creation.participant_request,
            participant=participant,
            context=CONTEXT,
        )
    missing_attachment = RecipientSessionCreationRequest(
        request=RunRequest(
            agent_name="reviewer",
            messages=[
                Message(
                    role="user",
                    content=(
                        FilePart(
                            attachment=file_attachment(
                                artifact_id="art_missing",
                                kind="image",
                                filename="missing.png",
                                content_type="image/png",
                                size_bytes=8,
                            )
                        ),
                    ),
                )
            ],
        ),
        creation_key="missing-attachment-child",
        recipient=participant,
    )
    with pytest.raises(PermissionError, match="without exact retained transfers") as missing_error:
        await application.create_recipient_session(missing_attachment, context=CONTEXT)
    assert "art_missing" not in str(missing_error.value)
    assert await application.lookup_recipient_session(missing_attachment, context=CONTEXT) is None
    tool_result_attachment = RecipientSessionCreationRequest(
        request=RunRequest(
            agent_name="reviewer",
            messages=[
                Message(
                    role="tool",
                    content=(
                        ToolResultPart(
                            tool_call_id="call-1",
                            tool_name="read_file",
                            artifacts=[
                                file_attachment(
                                    artifact_id="tool_art_missing",
                                    kind="image",
                                    filename="missing.png",
                                    content_type="image/png",
                                    size_bytes=8,
                                )
                            ],
                        ),
                    ),
                )
            ],
        ),
        creation_key="tool-result-attachment-child",
        recipient=participant,
    )
    with pytest.raises(PermissionError) as tool_result_error:
        await application.create_recipient_session(tool_result_attachment, context=CONTEXT)
    assert "tool_art_missing" not in str(tool_result_error.value)
    assert (
        await application.lookup_recipient_session(tool_result_attachment, context=CONTEXT) is None
    )
    concurrent_creation = RecipientSessionCreationRequest(
        request=creation.request,
        creation_key="concurrent-child",
        recipient=participant,
    )
    concurrent = await asyncio.gather(
        application.create_recipient_session(concurrent_creation, context=CONTEXT),
        application.create_recipient_session(concurrent_creation, context=CONTEXT),
    )
    assert concurrent[0][0].id == concurrent[1][0].id
    assert concurrent[0][1] == concurrent[1][1]
    recovered = await application.lookup_recipient_session(creation, context=CONTEXT)
    assert recovered is not None
    assert recovered[0].id == session.id
    assert recovered[1] == receipt

    conflicting = RecipientSessionCreationRequest(
        request=RunRequest(
            agent_name="reviewer",
            messages=[Message(role="user", content=(TextPart(text="changed"),))],
        ),
        creation_key=creation.creation_key,
        recipient=participant,
    )
    with pytest.raises(ValueError, match="conflicts"):
        await application.create_recipient_session(conflicting, context=CONTEXT)

    with pytest.raises(ValueError, match="selected view"):
        RecipientSessionCreationRequest(
            request=creation.request,
            creation_key="missing-view",
            recipient=participant,
            mode="fork",
        )


@pytest.mark.anyio
async def test_recipient_creation_cancellation_leaves_exact_retry_available() -> None:
    session_store = _BlockingRecipientStore()
    application = app(InMemoryCollaborationStore(), registration(), session_store=session_store)
    provider = ScriptedModelProvider([])
    application.register_provider(provider, default=True)
    application.register_agent(AgentSpec(name="reviewer", model="model", system_prompt="system"))
    initialized = await application.initialize_collaboration()
    _, participant_receipt = await create(application, initialized, key="cancel-owner")
    creation = RecipientSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[]),
        creation_key="cancel-child",
        recipient=participant_receipt.participants[0].reference,
    )
    task = asyncio.create_task(application.create_recipient_session(creation, context=CONTEXT))
    await session_store.started.wait()
    task.cancel()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled() and task.cancelling() == 2
    assert (
        await session_store.lookup_participant_session_creation(creation.participant_request)
        is None
    )
    session_store.release.set()
    session, receipt = await application.create_recipient_session(creation, context=CONTEXT)
    assert receipt.session_id == session.id
    assert provider.requests == []


@pytest.mark.anyio
async def test_recipient_creation_reconciles_commit_then_lost_acknowledgement() -> None:
    session_store = _CommitThenRaiseRecipientStore()
    application = app(InMemoryCollaborationStore(), registration(), session_store=session_store)
    application.register_provider(ScriptedModelProvider([]), default=True)
    application.register_agent(AgentSpec(name="reviewer", model="model", system_prompt="system"))
    initialized = await application.initialize_collaboration()
    _, participant_receipt = await create(application, initialized, key="ack-owner")
    creation = RecipientSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[]),
        creation_key="ack-child",
        recipient=participant_receipt.participants[0].reference,
    )
    with pytest.raises(ConnectionError, match="acknowledgement"):
        await application.create_recipient_session(creation, context=CONTEXT)
    session, receipt = await application.create_recipient_session(creation, context=CONTEXT)
    assert receipt.session_id == session.id
    assert await application.lookup_recipient_session(creation, context=CONTEXT) == (
        session,
        receipt,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_recipient_creation_authenticates_accepted_resource_transfer(
    backend: str, tmp_path, request
) -> None:
    collaboration_store = InMemoryCollaborationStore()
    store, artifact = await asyncio.to_thread(make_store, tmp_path)
    from tests.core.test_builtin_tools import TINY_PNG_BYTES

    from cayu.artifacts import ArtifactScope

    artifact = await store.put_bytes(
        TINY_PNG_BYTES,
        filename="image.png",
        content_type="image/png",
        scope=ArtifactScope.ENVIRONMENT,
        environment_name="resource",
    )
    if backend == "memory":
        session_store = InMemorySessionStore()
    elif backend == "sqlite":
        from cayu.storage.sqlite import SQLiteSessionStore

        session_store = SQLiteSessionStore(tmp_path / "transfer-recipient.sqlite")
    else:
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        session_store = PostgresSessionStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )
    application = app(
        collaboration_store,
        registration(scope="app"),
        session_store=session_store,
    )
    application.register_provider(ScriptedModelProvider([]), default=True)
    application.register_environment(
        Environment(EnvironmentSpec(name="resource"), artifact_store=store), default=True
    )
    application.register_agent(AgentSpec(name="reviewer", model="model", system_prompt="system"))
    initialized = await application.initialize_collaboration()
    _, participant_receipt = await create(application, initialized, key="transfer-owner")
    participant = participant_receipt.participants[0].reference
    source_ref = resource_owner("transfer-source")
    source = LocalArtifactResourceOwner(tmp_path / "source", owner=source_ref, artifact_store=store)
    destination = LocalArtifactResourceOwner(
        tmp_path / "destination",
        owner=participant.owner,
        artifact_store=store,
        participant=participant,
    )
    owned = await authorized(source, command(source_ref, artifact.id, store=store))
    transfer = _transfer_command(owned, source_ref, participant.owner)
    accepted = await destination.accept_transfer(transfer, source_owner=source)
    assert isinstance(accepted, ResourceTransferReceipt)
    creation = RecipientSessionCreationRequest(
        request=RunRequest(
            agent_name="reviewer",
            messages=[
                Message(
                    role="user",
                    content=(
                        FilePart(
                            attachment=file_attachment(
                                artifact_id=artifact.id,
                                kind="image",
                                filename=artifact.filename,
                                content_type="image/png",
                                size_bytes=artifact.size_bytes,
                            )
                        ),
                    ),
                )
            ],
        ),
        creation_key="transfer-child",
        recipient=participant,
        resource_transfers=(accepted,),
        preparation_receipts=(await destination.read_preparation(accepted.command),),
    )
    session, receipt = await application.create_recipient_session(
        creation, context=CONTEXT, resource_owner=destination
    )
    assert receipt.resource_transfer_commitments == (accepted.operation_digest,)
    assert session.id == receipt.session_id
    unrelated_tool_request = creation.request.model_copy(
        update={
            "messages": [
                Message(
                    role="tool",
                    content=(
                        ToolResultPart(
                            tool_call_id="unrelated-call",
                            tool_name="read_file",
                            artifacts=[
                                file_attachment(
                                    artifact_id="unrelated-artifact",
                                    kind="image",
                                    filename="other.png",
                                    content_type="image/png",
                                    size_bytes=8,
                                )
                            ],
                        ),
                    ),
                )
            ]
        },
        deep=True,
    )
    unrelated_tool_creation = RecipientSessionCreationRequest(
        request=unrelated_tool_request,
        creation_key="unrelated-tool-result-child",
        recipient=participant,
        resource_transfers=(accepted,),
        preparation_receipts=(await destination.read_preparation(accepted.command),),
    )
    with pytest.raises(PermissionError) as unrelated_error:
        await application.create_recipient_session(
            unrelated_tool_creation, context=CONTEXT, resource_owner=destination
        )
    assert "unrelated-artifact" not in str(unrelated_error.value)
    assert (
        await application.lookup_recipient_session(unrelated_tool_creation, context=CONTEXT) is None
    )
    forged_material = receipt.model_dump(mode="json", exclude={"receipt_commitment"})
    forged_material["creation_key"] = "forged-creation-key"
    forged_material["receipt_commitment"] = (
        "sha256:"
        + hashlib.sha256(
            canonical_bounded_durable_json_bytes(
                {
                    key: value
                    for key, value in forged_material.items()
                    if key != "receipt_commitment"
                },
                "recipient receipt",
                max_bytes=512 * 1024,
                max_nodes=8192,
                max_nesting=64,
            )
        ).hexdigest()
    )
    forged_receipt = RecipientSessionCreationReceipt.model_validate(forged_material)
    with pytest.raises(PermissionError, match="durable metadata"):
        await application.settle_recipient_resource_handoff(
            forged_receipt,
            resource_owner=destination,
            source_owners=(source,),
            context=CONTEXT,
        )
    with pytest.raises(PermissionError, match="invalid"):
        await application.settle_recipient_resource_handoff(
            receipt.model_copy(update={"session_instance_id": "wrong-incarnation"}),
            resource_owner=destination,
            source_owners=(source,),
            context=CONTEXT,
        )
    assert (await source.readback(owned.command)).receipt == owned
    reopened_source = LocalArtifactResourceOwner(
        tmp_path / "source", owner=source_ref, artifact_store=store
    )
    reopened_destination = LocalArtifactResourceOwner(
        tmp_path / "destination",
        owner=participant.owner,
        artifact_store=store,
        participant=participant,
    )
    await application.settle_recipient_resource_handoff(
        receipt,
        resource_owner=reopened_destination,
        source_owners=(reopened_source,),
        context=CONTEXT,
    )
    recovered = await application.reconcile_recipient_resource_handoff(
        creation,
        resource_owner=destination,
        source_owners=(source,),
        context=CONTEXT,
    )
    assert recovered == receipt
    # Replay after acknowledged release is idempotent, and destination retention
    # still prevents deletion after the source pin has gone.
    await application.settle_recipient_resource_handoff(
        receipt, resource_owner=destination, source_owners=(source,), context=CONTEXT
    )
    with pytest.raises(ValueError):
        await store.delete(artifact.id)


@pytest.mark.anyio
async def test_recipient_handoff_cancellation_preserves_retryable_source_ownership(
    tmp_path,
) -> None:
    store, artifact = await asyncio.to_thread(make_store, tmp_path)
    application = app(
        InMemoryCollaborationStore(),
        registration(scope="app"),
        session_store=InMemorySessionStore(),
    )
    application.register_provider(ScriptedModelProvider([]), default=True)
    application.register_agent(AgentSpec(name="reviewer", model="model", system_prompt="system"))
    initialized = await application.initialize_collaboration()
    _, participant_receipt = await create(application, initialized, key="handoff-cancel-owner")
    participant = participant_receipt.participants[0].reference
    source_ref = resource_owner("handoff-cancel-source")
    source = _BlockingHandoffOwner(tmp_path / "source", owner=source_ref, artifact_store=store)
    destination = LocalArtifactResourceOwner(
        tmp_path / "destination",
        owner=participant.owner,
        artifact_store=store,
        participant=participant,
    )
    owned = await authorized(source, command(source_ref, artifact.id, store=store))
    accepted = await destination.accept_transfer(
        _transfer_command(owned, source_ref, participant.owner), source_owner=source
    )
    creation = RecipientSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[]),
        creation_key="handoff-cancel-child",
        recipient=participant,
        resource_transfers=(accepted,),
        preparation_receipts=(await destination.read_preparation(accepted.command),),
    )
    _, receipt = await application.create_recipient_session(
        creation, context=CONTEXT, resource_owner=destination
    )
    task = asyncio.create_task(
        application.settle_recipient_resource_handoff(
            receipt, resource_owner=destination, source_owners=(source,), context=CONTEXT
        )
    )
    await source.handoff_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    source.handoff_release.set()
    await application.settle_recipient_resource_handoff(
        receipt, resource_owner=destination, source_owners=(source,), context=CONTEXT
    )
    forged = accepted.model_copy(update={"destination_pin_owner": "forged-pin"})
    with pytest.raises(ValueError, match="preparation evidence"):
        await application.create_recipient_session(
            RecipientSessionCreationRequest(
                request=creation.request,
                creation_key="forged-transfer-child",
                recipient=participant,
                resource_transfers=(forged,),
            ),
            context=CONTEXT,
            resource_owner=destination,
        )


@pytest.mark.anyio
async def test_recipient_multi_transfer_partial_cleanup_is_retryable(tmp_path) -> None:
    store, first = await asyncio.to_thread(make_store, tmp_path)
    second = await store.put_bytes(
        b"second",
        artifact_id="art_" + "2" * 32,
        filename="second.txt",
        scope=ArtifactScope.ENVIRONMENT,
        environment_name="resource",
    )
    application = app(
        InMemoryCollaborationStore(),
        registration(scope="app"),
        session_store=InMemorySessionStore(),
    )
    application.register_provider(ScriptedModelProvider([]), default=True)
    application.register_agent(AgentSpec(name="reviewer", model="model", system_prompt="system"))
    initialized = await application.initialize_collaboration()
    _, participant_receipt = await create(application, initialized, key="multi-owner")
    participant = participant_receipt.participants[0].reference
    source_ref = resource_owner("multi-source")
    source = _FailSecondHandoffOwner(tmp_path / "source", owner=source_ref, artifact_store=store)
    destination = LocalArtifactResourceOwner(
        tmp_path / "destination",
        owner=participant.owner,
        artifact_store=store,
        participant=participant,
    )
    first_owned = await authorized(source, command(source_ref, first.id, key="first", store=store))
    second_owned = await authorized(
        source, command(source_ref, second.id, key="second", store=store)
    )
    accepted_items = []
    for index, owned in enumerate((first_owned, second_owned)):
        transfer_command = _transfer_command(owned, source_ref, participant.owner).model_copy(
            update={
                "operation": _transfer_command(
                    owned, source_ref, participant.owner
                ).operation.model_copy(update={"caller_key": f"transfer-{index}"})
            },
            deep=True,
        )
        accepted_items.append(
            await destination.accept_transfer(transfer_command, source_owner=source)
        )
    accepted = tuple(accepted_items)
    creation = RecipientSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[]),
        creation_key="multi-child",
        recipient=participant,
        resource_transfers=accepted,
        preparation_receipts=tuple(
            [await destination.read_preparation(item.command) for item in accepted]
        ),
    )
    _, receipt = await application.create_recipient_session(
        creation, context=CONTEXT, resource_owner=destination
    )
    with pytest.raises(OSError, match="second source cleanup"):
        await application.settle_recipient_resource_handoff(
            receipt, resource_owner=destination, source_owners=(source,), context=CONTEXT
        )
    source.handoff_count = 0
    await application.reconcile_recipient_resource_handoff(
        creation, resource_owner=destination, source_owners=(source,), context=CONTEXT
    )
    assert source.handoff_count == 2
