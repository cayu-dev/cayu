"""Authenticated publication, ownership and readback of historical context views."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from hashlib import sha256

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.artifacts.resources import LocalArtifactResourceOwner
from cayu.collaboration._coordinator import ParticipantCoordinator
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.participants import ParticipantRef
from cayu.runtime import _session_request_boundary as session_request_boundary
from cayu.runtime._runtime_records import RegisteredAgentState, RegisteredEnvironment
from cayu.sessions.base import SessionStore
from cayu.sessions.context_views import (
    ContextViewExtensionRecord,
    ContextViewExtensionRegistration,
    ContextViewManifest,
    ContextViewOwnershipRequest,
    ContextViewProjectionSource,
    ContextViewPublicationRequest,
    ContextViewReadback,
    ContextViewSelectionRequest,
    context_view_artifact_ids,
    context_view_attachments,
    json_commitment,
    project_context_view_extensions,
)
from cayu.vaults.redaction import SecretRedactor


@dataclass(frozen=True, slots=True, repr=False)
class _ContextViewPublication:
    """Retain publication inputs without exposing them in resource diagnostics."""

    session_store: SessionStore
    manifest: ContextViewManifest
    publication_key: str

    async def __call__(self) -> ContextViewManifest:
        return await self.session_store.publish_context_view(
            self.manifest, publication_key=self.publication_key
        )


async def publish_completed_context_view(
    request: ContextViewPublicationRequest,
    *,
    session_store: SessionStore,
    participants: ParticipantCoordinator,
    resolve_session: Callable[[str], Awaitable[tuple[str, str | None]]],
    redactor: SecretRedactor,
    extensions: tuple[ContextViewExtensionRegistration, ...] = (),
    project_extensions: Callable[
        [ContextViewProjectionSource], tuple[tuple[ContextViewExtensionRecord, ...], str]
    ]
    | None = None,
    get_registered_environment: Callable[[str | None], RegisteredEnvironment | None] | None = None,
    get_registered_agent: Callable[[str], RegisteredAgentState] | None = None,
    process_identity: str | None = None,
    participant: ParticipantRef,
    context: CollaborationAccessContext,
    resource_owner: LocalArtifactResourceOwner | None = None,
) -> ContextViewManifest:
    """Publish an immutable view from authoritative completed-turn evidence."""

    publish = None
    try:
        if type(request) is not ContextViewPublicationRequest:
            raise TypeError("Context-view publication requires a typed request.")
        from cayu.collaboration._preparation import prepare_contract

        request = prepare_contract(ContextViewPublicationRequest, request, redactor=redactor)
        resource_references_json = (
            None
            if not request.resource_receipts
            else canonical_bounded_durable_json_bytes(
                [item.model_dump(mode="json") for item in request.resource_receipts],
                "context view resources",
                max_bytes=256 * 1024,
                max_nodes=8192,
                max_nesting=64,
            ).decode()
        )
        if request.resource_receipts and not isinstance(resource_owner, LocalArtifactResourceOwner):
            raise PermissionError("Context-view resources require their qualified source owner.")
        if type(participant) is not ParticipantRef:
            raise TypeError("Context-view publication requires a ParticipantRef.")
        if session_store.context_view_version is None:
            raise RuntimeError("The configured SessionStore does not support context views.")
        source_session_id, _ = await resolve_session(request.source_session_id)
        request = request.model_copy(update={"source_session_id": source_session_id}, deep=True)
        inspection = await participants.inspect(
            participant,
            context=context,
            action="administration",
        )
        if inspection.participant.lifecycle != "active":
            raise PermissionError("Only active participants can publish context views.")
        lookup_publication = getattr(session_store, "lookup_context_view_publication", None)
        if lookup_publication is None:
            raise RuntimeError(
                "The configured SessionStore cannot reconcile context-view publication."
            )
        existing_manifest = await lookup_publication(request.publication_key)
        if existing_manifest is not None:
            stored_extension_schema = tuple(
                (extension.extension, extension.schema_version)
                for extension in existing_manifest.extensions
            )
            current_extension_schema = tuple(
                (extension.extension, extension.schema_version) for extension in extensions
            )
            if (
                existing_manifest.participant != participant
                or existing_manifest.source_session_id != request.source_session_id
                or existing_manifest.source_session_instance_id
                != request.source_session_instance_id
                or existing_manifest.view_id != request.view_id
                or existing_manifest.interaction_id != request.interaction_id
                or existing_manifest.boundary_id != request.boundary_id
                or existing_manifest.projection_schema != request.projection_schema
                or stored_extension_schema != current_extension_schema
                or existing_manifest.resource_references_json != resource_references_json
            ):
                raise ValueError("Context-view publication key conflicts with its request.")
            return existing_manifest
        source = await session_store.capture_context_view_publication_source(
            request.source_session_id
        )
        session = source.session
        if session.instance_id != request.source_session_instance_id:
            raise LookupError("The source session incarnation is unavailable.")
        binding = source.binding
        if binding.participant != participant:
            raise PermissionError("The source session is not bound to this participant.")
        for field_name in ("agent_name", "provider_name", "model", "environment_name"):
            session_request_boundary.require_secret_free_session_authority(
                getattr(session, field_name),
                field_name=field_name,
                redactor=redactor,
            )
        pointer = source.pointer
        completion_event = source.completion_event
        assert completion_event.interaction_id is not None
        if request.interaction_id != completion_event.interaction_id:
            raise ValueError(
                "The publication interaction identity conflicts with completion evidence."
            )
        if request.boundary_id != pointer.logical_step_id:
            raise ValueError(
                "The publication boundary identity conflicts with completion evidence."
            )
        records = source.records

        required_artifacts = context_view_artifact_ids([record.message for record in records])
        if not set(required_artifacts) <= {
            item for receipt in request.resource_receipts for item in receipt.material_ids
        }:
            raise ValueError(
                "Context-view publication requires independently qualified resource references."
            )
        messages_json = canonical_bounded_durable_json_bytes(
            [record.message.model_dump(mode="json") for record in records],
            "context view messages",
            max_bytes=8 * 1024 * 1024,
            max_nodes=8192,
            max_nesting=64,
        ).decode("utf-8")
        # Extensions receive only the bounded historical projection source.
        # Passing the executable Session model would expose invocation,
        # metadata, leases, and other private authority to a producer even
        # though none of it is eligible for a historical manifest.
        # No application-owned executable context is published by this baseline
        # projection.  ``null`` is explicit and committed rather than a mutable
        # placeholder that could be mistaken for retained configuration.
        application_context_json = None
        historical_profile = source.execution_profile
        historical_ancestry_json = canonical_bounded_durable_json_bytes(
            {
                "source_session_instance_id": session.instance_id,
                "provider_name": session.provider_name,
                "model": session.model,
                "agent_name": session.agent_name,
                "execution_profile": historical_profile.model_dump(mode="json"),
                "runtime_build_fingerprint": session.runtime_build_fingerprint,
                "creation_definition_json": binding.historical_definition_json,
                "creation_definition_commitment": json_commitment(
                    binding.historical_definition_json
                ),
            },
            "historical ancestry",
            max_bytes=256 * 1024,
            max_nodes=8192,
            max_nesting=64,
        ).decode("utf-8")
        causal_budget_ancestry_json = canonical_bounded_durable_json_bytes(
            {"causal_budget_id": session.causal_budget_id},
            "causal budget ancestry",
            max_bytes=256 * 1024,
            max_nodes=8192,
            max_nesting=64,
        ).decode("utf-8")
        compaction_json = canonical_bounded_durable_json_bytes(
            {
                "state": "uncompacted",
                "input_frontier": pointer.source_transcript_cursor,
                "retained_output_frontier": source.transcript_end_cursor,
                "retained_suffix_frontier": source.transcript_end_cursor,
            },
            "context view compaction relationship",
            max_bytes=256 * 1024,
            max_nodes=8192,
            max_nesting=64,
        ).decode("utf-8")
        extension_source = ContextViewProjectionSource(
            source_session_id=session.id,
            source_session_instance_id=session.instance_id,
            participant_json=canonical_bounded_durable_json_bytes(
                participant.model_dump(mode="json"), "participant", max_bytes=8192, max_nodes=64
            ).decode("utf-8"),
            interaction_id=completion_event.interaction_id,
            boundary_id=pointer.logical_step_id,
            completion_event_id=pointer.completion_event_id,
            source_transcript_cursor=pointer.source_transcript_cursor,
            transcript_cursor=source.transcript_end_cursor,
            messages_json=messages_json,
            compaction_json=compaction_json,
            historical_ancestry_json=historical_ancestry_json,
        )
        extension_records, extension_set_commitment = (
            project_context_view_extensions(extensions, extension_source)
            if project_extensions is None
            else project_extensions(extension_source)
        )
        material = {
            "schema_version": 1,
            "source_owner": participant.owner.model_dump(mode="json"),
            "participant": participant.model_dump(mode="json"),
            "source_session_id": session.id,
            "source_session_instance_id": session.instance_id,
            "view_id": request.view_id,
            "interaction_id": request.interaction_id,
            "boundary_id": request.boundary_id,
            "completion_event_id": pointer.completion_event_id,
            "transcript_cursor": source.transcript_end_cursor,
            "projection_schema": request.projection_schema,
            "extension_set_commitment": extension_set_commitment,
            "messages_json": messages_json,
            "application_context_json": application_context_json,
            "extensions": [record.model_dump(mode="json") for record in extension_records],
            "historical_ancestry_json": historical_ancestry_json,
            "causal_budget_ancestry_json": causal_budget_ancestry_json,
            "resource_references_json": resource_references_json,
            "compaction_json": compaction_json,
            "messages_commitment": json_commitment(messages_json, "messages"),
            "application_context_commitment": json_commitment(
                application_context_json or "null", "application context"
            ),
        }
        material["manifest_commitment"] = (
            "sha256:"
            + sha256(
                canonical_bounded_durable_json_bytes(
                    material,
                    "context view manifest",
                    max_bytes=8 * 1024 * 1024,
                    max_nodes=8192,
                    max_nesting=64,
                )
            ).hexdigest()
        )
        manifest = ContextViewManifest.model_validate(material)

        # The resource owner can retain this operation after caller cancellation.
        # Its representation must not reveal the store or historical contents.
        publish = _ContextViewPublication(
            session_store,
            manifest,
            publication_key=request.publication_key,
        )

        if request.resource_receipts:
            assert resource_owner is not None
            from cayu.runtime._execution_profile_admission import (
                require_historical_artifact_environment,
            )

            if (
                get_registered_environment is None
                or get_registered_agent is None
                or process_identity is None
            ):
                raise ValueError(
                    "Context-view resources require registered environment and agent resolvers "
                    "and a process identity."
                )
            environment = get_registered_environment(session.environment_name)
            require_historical_artifact_environment(
                profile=source.execution_profile,
                registered_environment=environment,
                registered_agent=get_registered_agent(session.agent_name),
                runtime_version=session.runtime_version,
                process_identity=process_identity,
                redactor=redactor,
            )
            assert environment is not None
            return await resource_owner._run_context_view_commit(
                request.resource_receipts,
                participant=participant,
                attachments=context_view_attachments([record.message for record in records]),
                artifact_store=environment.environment.artifact_store,
                environment_name=environment.spec.name,
                session_id=session.id,
                operation=publish,
            )
        return await publish()
    finally:
        # Extension dependencies may have sensitive representations in tracebacks.
        del (
            session_store,
            participants,
            resolve_session,
            redactor,
            extensions,
            project_extensions,
            get_registered_environment,
            get_registered_agent,
            publish,
        )


async def transition_context_view_ownership(
    request: ContextViewOwnershipRequest,
    *,
    session_store: SessionStore,
    participants: ParticipantCoordinator,
    participant: ParticipantRef,
    destination_participant: ParticipantRef | None = None,
    context: CollaborationAccessContext,
):
    """Apply one authenticated, replay-safe context-view ownership transition."""

    try:
        if type(request) is not ContextViewOwnershipRequest:
            raise TypeError("Context-view ownership requires a typed request.")
        if type(participant) is not ParticipantRef:
            raise TypeError("Context-view ownership requires a ParticipantRef.")
        if session_store.context_view_version is None:
            raise RuntimeError("The configured SessionStore does not support context views.")
        inspection = await participants.inspect(
            participant,
            context=context,
            action="administration",
        )
        if inspection.participant.lifecycle != "active" and request.operation != "release":
            raise PermissionError("Only active participants can change context-view ownership.")
        if request.current_owner != participant.owner:
            raise PermissionError("The authenticated participant is not the current owner.")
        request = request.model_copy(update={"current_participant": participant}, deep=True)
        if request.destination_owner is not None:
            if (
                destination_participant is None
                or destination_participant.owner != request.destination_owner
            ):
                raise PermissionError(
                    "The destination participant does not match the transfer owner."
                )
            destination_inspection = await participants.inspect(
                destination_participant,
                context=context,
                action="administration",
            )
            if destination_inspection.participant.lifecycle != "active":
                raise PermissionError("Ownership cannot be transferred to an inactive participant.")
            request = request.model_copy(
                update={"destination_participant": destination_participant}, deep=True
            )
        return await session_store.transition_context_view_ownership(request)
    finally:
        # Extension dependencies may have sensitive representations in tracebacks.
        del session_store, participants


async def select_context_view(
    request: ContextViewSelectionRequest,
    *,
    session_store: SessionStore,
    participants: ParticipantCoordinator,
    resolve_session: Callable[[str], Awaitable[tuple[str, str | None]]],
    participant: ParticipantRef,
    context: CollaborationAccessContext,
):
    try:
        if type(request) is not ContextViewSelectionRequest:
            raise TypeError("Context-view selection requires a typed request.")
        if type(participant) is not ParticipantRef:
            raise TypeError("Context-view selection requires a ParticipantRef.")
        if session_store.context_view_version is None:
            raise RuntimeError("The configured SessionStore does not support context views.")
        source_session_id, _ = await resolve_session(request.source_session_id)
        request = request.model_copy(update={"source_session_id": source_session_id}, deep=True)
        inspection = await participants.inspect(
            participant,
            context=context,
            action="administration",
        )
        if inspection.participant.lifecycle != "active":
            raise PermissionError("Only active participants can select context views.")
        if request.source_owner != participant.owner:
            raise PermissionError("The authenticated participant does not own the source view.")
        binding = await session_store.load_participant_session_binding(request.source_session_id)
        if (
            binding is None
            or binding.participant != participant
            or binding.session_id != request.source_session_id
            or binding.session_instance_id != request.source_session_instance_id
        ):
            raise PermissionError(
                "The source session is not bound to this participant incarnation."
            )
        return await session_store.select_context_view(request)
    finally:
        # Extension dependencies may have sensitive representations in tracebacks.
        del session_store, participants, resolve_session


async def read_context_view(
    view_id: str,
    *,
    session_store: SessionStore,
    participants: ParticipantCoordinator,
    resolve_session: Callable[[str], Awaitable[tuple[str, str | None]]],
    source_session_id: str,
    participant: ParticipantRef,
    context: CollaborationAccessContext,
) -> ContextViewReadback:
    try:
        if type(view_id) is not str or type(source_session_id) is not str:
            raise TypeError("Context-view readback requires string identifiers.")
        if type(participant) is not ParticipantRef:
            raise TypeError("Context-view readback requires a ParticipantRef.")
        if session_store.context_view_version is None:
            raise RuntimeError("The configured SessionStore does not support context views.")
        source_session_id, _ = await resolve_session(source_session_id)
        await participants.inspect(
            participant,
            context=context,
            action="readback",
        )
        readback = await session_store.read_context_view(
            view_id,
            source_session_id=source_session_id,
        )
        # A collaboration OwnerRef is shared by many participants. Historical
        # owner events cannot authenticate a different participant incarnation.
        events = await session_store.read_context_view_lifecycle_events(
            view_id, limit=1, owner_participant=participant
        )
        authorized_participants = {
            readback.view.participant,
            *(event.owner_participant for event in events if event.owner_participant is not None),
        }
        if participant not in authorized_participants:
            raise PermissionError("The authenticated participant does not own the source view.")
        return readback
    finally:
        # Extension dependencies may have sensitive representations in tracebacks.
        del session_store, participants, resolve_session


async def read_context_view_lifecycle_events(
    view_id: str,
    *,
    session_store: SessionStore,
    participants: ParticipantCoordinator,
    resolve_session: Callable[[str], Awaitable[tuple[str, str | None]]],
    source_session_id: str,
    participant: ParticipantRef,
    context: CollaborationAccessContext,
    limit: int = 256,
):
    try:
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("Context-view public event limits must be between 1 and 256.")
        if type(view_id) is not str or type(source_session_id) is not str:
            raise TypeError("Context-view event readback requires string identifiers.")
        if type(participant) is not ParticipantRef:
            raise TypeError("Context-view event readback requires a ParticipantRef.")
        if session_store.context_view_version is None:
            raise RuntimeError("The configured SessionStore does not support context views.")
        source_session_id, _ = await resolve_session(source_session_id)
        await participants.inspect(
            participant,
            context=context,
            action="readback",
        )
        readback = await session_store.read_context_view(
            view_id,
            source_session_id=source_session_id,
        )
        authority_events = await session_store.read_context_view_lifecycle_events(
            view_id, limit=1, owner_participant=participant
        )
        authorized_participants = {
            readback.view.participant,
            *(
                event.owner_participant
                for event in authority_events
                if event.owner_participant is not None
            ),
        }
        if participant not in authorized_participants:
            raise PermissionError("The authenticated participant does not own the view events.")
        events = await session_store.read_context_view_lifecycle_events(view_id, limit=limit)
        if not events:
            raise LookupError("Context-view lifecycle evidence is unavailable.")
        return events
    finally:
        # Extension dependencies may have sensitive representations in tracebacks.
        del session_store, participants, resolve_session
