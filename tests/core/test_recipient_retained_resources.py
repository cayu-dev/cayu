"""Public publication → retained historical resource → inert recipient witness."""

import json
from dataclasses import replace
from uuid import uuid4

import pytest
from tests.artifacts.test_resources import (
    LocalArtifactResourceOwner,
    _transfer_command,
    command,
    preparation_permit,
)
from tests.core.test_builtin_tools import TINY_PNG_BYTES, AttachmentTool
from tests.core.test_participant_identity import CONTEXT, app, create, registration

from cayu.agents import AgentSpec
from cayu.artifacts import ArtifactScope, LocalArtifactStore
from cayu.artifacts.resources import ResourceOwnerConflict, ResourceOwnerUnsupported
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.environments import Environment, EnvironmentSpec
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.messages import FilePart, Message, ToolResultPart
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._model_completion_publication import model_step_publication_from_checkpoint
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.base import EventQuery, InMemorySessionStore, ResumeRequest, RunRequest
from cayu.sessions.context_views import (
    ContextViewLimits,
    ContextViewOwnershipRequest,
    ContextViewPublicationRequest,
    ContextViewSelectionRequest,
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
    RecipientSessionCreationRequest,
)


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("artifact_scope", [ArtifactScope.ENVIRONMENT, ArtifactScope.SESSION])
async def test_fork_retains_source_attachment_with_exact_transfer(
    backend, artifact_scope, tmp_path, request
):
    case_id = uuid4().hex
    if backend == "memory":
        sessions = InMemorySessionStore()
    elif backend == "sqlite":
        from cayu.storage.sqlite import SQLiteSessionStore

        sessions = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    else:
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        sessions = PostgresSessionStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    collaboration = InMemoryCollaborationStore()
    registered = registration()
    application = app(collaboration, registered, session_store=sessions)
    registered_artifacts = LocalArtifactStore(artifacts.root, store_id=artifacts.id)
    environment_spec = EnvironmentSpec(
        name="local",
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="retained-artifact-environment", behavior_version="1", implementation_version="1"
        ),
    )
    application.register_environment(
        Environment(
            environment_spec,
            artifact_store=registered_artifacts,
        ),
        default=True,
    )
    batches = iter(
        [
            [
                ModelStreamEvent.tool_call(name="attach_file", arguments={}, id="attachment-call"),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("completed"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    provider = ScriptedModelProvider(response_factory=lambda _: next(batches))
    application.register_provider(provider, default=True)
    initialized = await application.initialize_collaboration()
    _, source_participants = await create(application, initialized, key="source")
    _, destination_participants = await create(application, initialized, key="recipient")
    source_participant = source_participants.participants[0].reference
    recipient = destination_participants.participants[0].reference
    part = await application.attach_file(
        TINY_PNG_BYTES,
        filename="image.png",
        kind="image",
        scope=artifact_scope,
        session_id=("source-" + case_id if artifact_scope is ArtifactScope.SESSION else None),
    )
    application.register_agent(
        AgentSpec(name="reviewer", model="model", system_prompt="system"),
        tools=[AttachmentTool(part.attachment["artifact_id"], len(TINY_PNG_BYTES))],
    )
    root_request = ParticipantSessionCreationRequest(
        request=RunRequest(
            agent_name="reviewer",
            messages=[Message.text("user", "Attach image")],
            max_steps=1,
            session_id="source-" + case_id,
        ),
        creation_key="source-session-" + case_id,
    )
    source_session, _ = await application.create_participant_session(
        root_request, participant=source_participant, context=CONTEXT
    )
    events = [
        event
        async for event in application.execute_participant_session(
            ParticipantSessionExecutionRequest(
                request=root_request.request.model_copy(update={"session_id": source_session.id}),
                session_instance_id=source_session.instance_id,
                execution_key="first",
            ),
            participant=source_participant,
            context=CONTEXT,
        )
    ]
    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in events)
    from cayu.sessions.context_views import context_view_artifact_ids

    retained = await sessions.load_transcript_snapshot(source_session.id)
    assert context_view_artifact_ids([record.message for record in retained.records]), retained
    pointer = model_step_publication_from_checkpoint(
        await sessions.load_checkpoint(source_session.id)
    )
    completed = (
        await sessions.query_events(
            EventQuery(session_id=source_session.id, event_id=pointer.completion_event_id, limit=1)
        )
    )[0].event
    source_owner = LocalArtifactResourceOwner(
        tmp_path / "source-owner", owner=source_participant.owner, artifact_store=artifacts
    )
    cmd = command(source_participant.owner, part.attachment["artifact_id"], store=artifacts)
    permit = preparation_permit(cmd)
    permit = permit.model_copy(
        update={
            "intent": permit.intent.model_copy(
                update={
                    "request": permit.intent.request.model_copy(
                        update={"participant": source_participant}
                    )
                }
            )
        }
    )
    prepared = await source_owner.authorize(cmd, permit=permit)
    owned = await source_owner.acquire(cmd, preparation=prepared)
    publication = ContextViewPublicationRequest(
        source_session_id=source_session.id,
        source_session_instance_id=source_session.instance_id,
        view_id="retained-resource-view-" + case_id,
        interaction_id=completed.interaction_id,
        boundary_id=pointer.logical_step_id,
        projection_schema="whole-turn.v1",
        publication_key="retained-resource-publication-" + case_id,
    )
    with pytest.raises(ValueError, match="qualified resource"):
        await application.publish_completed_context_view(
            publication, participant=source_participant, context=CONTEXT
        )
    assert await sessions.lookup_context_view_publication(publication.publication_key) is None
    publication = publication.model_copy(update={"resource_receipts": (owned,)})
    # Genuine authority over a same-ID artifact in another physical store does
    # not prove that it supplied this completed source turn.
    wrong_artifacts = LocalArtifactStore(tmp_path / "wrong-source", store_id=artifacts.id)
    await wrong_artifacts.put_bytes(
        TINY_PNG_BYTES[:-1] + bytes([TINY_PNG_BYTES[-1] ^ 1]),
        artifact_id=part.attachment["artifact_id"],
        filename="image.png",
        content_type="image/png",
        scope=artifact_scope,
        session_id=(source_session.id if artifact_scope is ArtifactScope.SESSION else None),
        environment_name="local",
    )
    wrong_owner = LocalArtifactResourceOwner(
        tmp_path / "wrong-source-owner",
        owner=source_participant.owner,
        artifact_store=wrong_artifacts,
    )
    wrong_command = command(
        source_participant.owner, part.attachment["artifact_id"], store=wrong_artifacts
    )
    wrong_permit = preparation_permit(wrong_command)
    wrong_permit = wrong_permit.model_copy(
        update={
            "intent": wrong_permit.intent.model_copy(
                update={
                    "request": wrong_permit.intent.request.model_copy(
                        update={"participant": source_participant}
                    )
                }
            )
        }
    )
    wrong_preparation = await wrong_owner.authorize(wrong_command, permit=wrong_permit)
    wrong_receipt = await wrong_owner.acquire(wrong_command, preparation=wrong_preparation)
    wrong_publication = publication.model_copy(update={"resource_receipts": (wrong_receipt,)})
    with pytest.raises(ResourceOwnerUnsupported, match="store"):
        await application.publish_completed_context_view(
            wrong_publication,
            participant=source_participant,
            context=CONTEXT,
            resource_owner=wrong_owner,
        )
    assert await sessions.lookup_context_view_publication(publication.publication_key) is None
    # Reconfiguration must not authenticate a substituted store merely because
    # its public ID and application-declared behavior version are unchanged.
    replacement = app(collaboration, registered, session_store=sessions)
    replacement.register_environment(
        Environment(environment_spec, artifact_store=wrong_artifacts), default=True
    )
    replacement.register_agent(AgentSpec(name="reviewer", model="model", system_prompt="system"))
    await replacement.initialize_collaboration()
    with pytest.raises(ValueError, match="Historical artifact environment"):
        await replacement.publish_completed_context_view(
            wrong_publication,
            participant=source_participant,
            context=CONTEXT,
            resource_owner=wrong_owner,
        )
    assert await sessions.lookup_context_view_publication(publication.publication_key) is None
    requests_before_publication = len(provider.requests)
    manifest = await application.publish_completed_context_view(
        publication, participant=source_participant, context=CONTEXT, resource_owner=source_owner
    )
    assert len(provider.requests) == requests_before_publication
    # A new application with the same qualified physical identity can publish
    # another exact historical view without executing the source again.
    restored = app(collaboration, registered, session_store=sessions)
    restored.register_environment(
        Environment(
            environment_spec,
            artifact_store=LocalArtifactStore(artifacts.root, store_id=artifacts.id),
        ),
        default=True,
    )
    restored.register_agent(AgentSpec(name="reviewer", model="model", system_prompt="system"))
    await restored.initialize_collaboration()
    await restored.publish_completed_context_view(
        publication.model_copy(
            update={
                "publication_key": publication.publication_key + "-restored",
                "view_id": publication.view_id + "-restored",
            }
        ),
        participant=source_participant,
        context=CONTEXT,
        resource_owner=source_owner,
    )
    selected = await application.select_context_view(
        ContextViewSelectionRequest(
            source_owner=source_participant.owner,
            source_session_id=source_session.id,
            source_session_instance_id=source_session.instance_id,
            selector="exact",
            exact_view_id=manifest.view_id,
            projection_schema=manifest.projection_schema,
            extension_set_commitment=manifest.extension_set_commitment,
            limits=ContextViewLimits(),
            selection_key="resource-selection-" + case_id,
        ),
        participant=source_participant,
        context=CONTEXT,
    )
    adopted = await application.transition_context_view_ownership(
        ContextViewOwnershipRequest(
            selection_key=selected.selection_key,
            view_id=manifest.view_id,
            pin_commitment=selected.pin_commitment,
            expected_state=selected.state,
            expected_revision=selected.ownership_revision,
            operation="adopt",
            current_owner=source_participant.owner,
            destination_owner=recipient.owner,
            operation_key="resource-adopt-" + case_id,
        ),
        participant=source_participant,
        destination_participant=recipient,
        context=CONTEXT,
    )
    destination = LocalArtifactResourceOwner(
        tmp_path / "destination-owner",
        owner=recipient.owner,
        artifact_store=artifacts,
        participant=recipient,
    )
    accepted = await destination.accept_transfer(
        _transfer_command(owned, source_participant.owner, recipient.owner),
        source_owner=source_owner,
    )
    creation = RecipientSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[]),
        creation_key="resource-fork-" + case_id,
        recipient=recipient,
        mode="fork",
        selected_view=adopted,
        resource_transfers=(accepted,),
        preparation_receipts=(await destination.read_preparation(accepted.command),),
    )
    fresh = replace(
        creation,
        mode="fresh",
        selected_view=None,
        creation_key="fresh-resource-" + case_id,
        request=RunRequest(agent_name="reviewer", messages=[Message(role="user", content=(part,))]),
    )
    requests_before_validation = len(provider.requests)
    if artifact_scope is ArtifactScope.SESSION:
        for candidate in (creation, fresh):
            with pytest.raises(ResourceOwnerUnsupported, match="scope"):
                await application.create_recipient_session(
                    candidate, context=CONTEXT, resource_owner=destination
                )
            assert await application.lookup_recipient_session(candidate, context=CONTEXT) is None
        assert len(provider.requests) == requests_before_validation
        if backend != "memory":
            await sessions.close()
        return

    for mode_request in (creation, fresh):
        for tool_result in (False, True):
            for defect in ("size", "content-type", "duplicate"):
                invalid_payload = dict(part.attachment)
                if defect == "size":
                    invalid_payload["size_bytes"] += 1
                elif defect == "content-type":
                    invalid_payload["content_type"] = "image/jpeg"
                else:
                    invalid_payload["filename"] = "conflicting-name.png"
                payloads = (
                    [invalid_payload]
                    if defect != "duplicate"
                    else [part.attachment, invalid_payload]
                )
                messages = (
                    [
                        Message(
                            role="tool",
                            content=(
                                ToolResultPart(
                                    tool_call_id="retained-result",
                                    tool_name="attach_file",
                                    content="attachment",
                                    artifacts=payloads,
                                ),
                            ),
                        )
                    ]
                    if tool_result
                    else [
                        Message(
                            role="user",
                            content=tuple(FilePart(attachment=value) for value in payloads),
                        )
                    ]
                )
                candidate = replace(
                    mode_request,
                    creation_key=f"{mode_request.creation_key}-{tool_result}-{defect}",
                    request=mode_request.request.model_copy(update={"messages": messages}),
                )
                with pytest.raises((ValueError, ResourceOwnerConflict), match="conflict"):
                    await application.create_recipient_session(
                        candidate, context=CONTEXT, resource_owner=destination
                    )
                assert (
                    await application.lookup_recipient_session(candidate, context=CONTEXT) is None
                )
    assert len(provider.requests) == requests_before_validation

    # A real retained receipt is insufficient for an unrelated physical store,
    # including another store containing the same public artifact identifier.
    other = LocalArtifactStore(tmp_path / "other-artifacts", store_id=artifacts.id)
    await other.put_bytes(
        TINY_PNG_BYTES,
        artifact_id=part.attachment["artifact_id"],
        filename="image.png",
        content_type="image/png",
        scope=ArtifactScope.ENVIRONMENT,
        environment_name="local",
    )
    other_application = app(collaboration, registered, session_store=sessions)
    other_provider = ScriptedModelProvider([])
    other_application.register_provider(other_provider, default=True)
    other_application.register_agent(
        AgentSpec(name="reviewer", model="model", system_prompt="system")
    )
    other_application.register_environment(
        Environment(EnvironmentSpec(name="local"), artifact_store=other), default=True
    )
    await other_application.initialize_collaboration()
    application.register_environment(
        Environment(EnvironmentSpec(name="wrong-scope"), artifact_store=registered_artifacts)
    )
    application.register_environment(Environment(EnvironmentSpec(name="no-store")))
    for mode_request in (creation, fresh):
        wrong_store = replace(mode_request, creation_key=mode_request.creation_key + "-wrong-store")
        with pytest.raises(ResourceOwnerUnsupported, match="store"):
            await other_application.create_recipient_session(
                wrong_store, context=CONTEXT, resource_owner=destination
            )
        assert (
            await other_application.lookup_recipient_session(wrong_store, context=CONTEXT) is None
        )
        for environment_name in ("wrong-scope", "no-store"):
            candidate = replace(
                mode_request,
                creation_key=mode_request.creation_key + "-" + environment_name,
                request=mode_request.request.model_copy(
                    update={"environment_name": environment_name}
                ),
            )
            with pytest.raises(ResourceOwnerUnsupported):
                await application.create_recipient_session(
                    candidate, context=CONTEXT, resource_owner=destination
                )
            assert await application.lookup_recipient_session(candidate, context=CONTEXT) is None
    assert len(provider.requests) == requests_before_validation
    assert other_provider.requests == []
    fresh_session, fresh_receipt = await application.create_recipient_session(
        fresh, context=CONTEXT, resource_owner=destination
    )
    assert await application.create_recipient_session(
        fresh, context=CONTEXT, resource_owner=destination
    ) == (fresh_session, fresh_receipt)
    with pytest.raises(PermissionError, match="retained transfers"):
        await application.create_recipient_session(
            replace(creation, resource_transfers=(), preparation_receipts=()), context=CONTEXT
        )
    calls_before = len(provider.requests)
    child, receipt = await application.create_recipient_session(
        creation, context=CONTEXT, resource_owner=destination
    )
    assert len(provider.requests) == calls_before
    assert [
        item.model_dump(mode="json") for item in await sessions.load_transcript(child.id)
    ] == json.loads(manifest.messages_json)
    assert await application.create_recipient_session(
        creation, context=CONTEXT, resource_owner=destination
    ) == (child, receipt)
    advanced = [
        event
        async for event in application.resume(
            ResumeRequest(
                session_id=source_session.id,
                messages=[Message.text("user", "continue")],
                max_steps=1,
            ),
            context=CONTEXT,
        )
    ]
    assert any(event.type == EventType.SESSION_COMPLETED for event in advanced)
    await sessions.validate_context_view_compaction(source_session.id, manifest.transcript_cursor)
    await application.settle_recipient_resource_handoff(
        receipt, resource_owner=destination, source_owners=(source_owner,), context=CONTEXT
    )
    with pytest.raises(ValueError):
        await artifacts.delete(part.attachment["artifact_id"])
    assert await application.lookup_recipient_session(creation, context=CONTEXT) == (child, receipt)
    if backend != "memory":
        await sessions.close()
        if backend == "sqlite":
            reopened = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        else:
            reopened = PostgresSessionStore(request.getfixturevalue("postgres_dsn"))
        recovered = app(collaboration, registered, session_store=reopened)
        await recovered.initialize_collaboration()
        assert await recovered.lookup_recipient_session(creation, context=CONTEXT) == (
            child,
            receipt,
        )
        retained_view = await reopened.read_context_view(
            manifest.view_id, source_session_id=source_session.id
        )
        assert retained_view.view == manifest
        await reopened.close()
