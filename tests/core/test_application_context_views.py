"""Direct context-view composition and application boundary preservation."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from tests.artifacts.test_resources import (
    LocalArtifactResourceOwner,
    command,
    preparation_permit,
)
from tests.core.test_builtin_tools import TINY_PNG_BYTES

import cayu
from cayu import _application_context_views as views
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.artifacts import ArtifactScope, LocalArtifactStore
from cayu.artifacts.attachments import FileAttachment
from cayu.artifacts.resources import ResourceOwnerUnavailable
from cayu.collaboration._contracts import OwnerRef
from cayu.collaboration._coordinator import ParticipantCoordinator
from cayu.collaboration.access import (
    CollaborationAccessContext,
    CollaborationAccessDenied,
    CollaborationAccessGrant,
    CollaborationAccessPolicy,
    CollaborationRegistration,
)
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.collaboration.participants import (
    CollaborationBootstrap,
    CollaborationLimits,
    ParticipantConfiguration,
    ParticipantConfigurationRef,
    ParticipantCreate,
    ParticipantRef,
)
from cayu.environments import Environment, EnvironmentSpec
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import Event, EventType
from cayu.messages import FilePart
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions._model_completion_publication import (
    LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
    ModelStepPublicationCheckpoint,
)
from cayu.sessions.base import InMemorySessionStore, Message
from cayu.sessions.context_views import (
    ContextViewExtensionProjection,
    ContextViewExtensionRegistration,
    ContextViewLimits,
    ContextViewOwnershipRequest,
    ContextViewPublicationRequest,
    ContextViewSelectionRequest,
    ParticipantSessionCreationRequest,
)
from cayu.sessions.requests import RunRequest
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.vaults.redaction import SecretRedactor

CONTEXT = CollaborationAccessContext(principal="operator")


class ViewPolicy(CollaborationAccessPolicy):
    def authorize(self, context, *, application_scope, action):
        if context.principal != "operator":
            raise CollaborationAccessDenied("denied")
        return CollaborationAccessGrant(application_scope=application_scope, participants=None)


async def completed_source(sessions, *, environment=None, messages=None):
    """Prepare authoritative completed-turn evidence before exercising the component."""
    configuration = ParticipantConfiguration(
        definition=ParticipantConfigurationRef(name="reviewer", version=1),
        routing=ParticipantConfigurationRef(name="route", version=1),
        admission=ParticipantConfigurationRef(name="admission", version=1),
    )
    collaboration = InMemoryCollaborationStore()
    application = CayuApp(
        session_store=sessions,
        collaboration_store=collaboration,
        collaboration=CollaborationRegistration(
            bootstrap=CollaborationBootstrap(
                application_scope=uuid4().hex,
                provisioning_scope="application",
                owner_name="participants",
                limits=CollaborationLimits(
                    participants=8,
                    aliases=8,
                    operations=32,
                    events=64,
                    retained_bytes=1024 * 1024,
                    control_operations=8,
                    control_events=8,
                    control_bytes=65536,
                    namespaces=4,
                    generations=8,
                    obligations=16,
                ),
            ),
            access_policy=ViewPolicy(),
            configurations=(configuration,),
        ),
        enable_logging=False,
    )
    application.register_provider(ScriptedModelProvider([]), default=True)
    if environment is not None:
        application.register_environment(environment, default=True)
    application.register_agent(AgentSpec(name="reviewer", model="model"))
    initialized = await application.initialize_collaboration()
    created = await application.create_participant(
        ParticipantCreate(operation=initialized.operation("create"), configuration=configuration),
        context=CONTEXT,
    )
    participant = created.participants[0].reference
    session, _ = await application.create_participant_session(
        ParticipantSessionCreationRequest(
            request=RunRequest(agent_name="reviewer", messages=[]), creation_key="source"
        ),
        participant=participant,
        context=CONTEXT,
    )
    await sessions.append_event(
        session.id,
        Event(
            id="completed",
            type=EventType.MODEL_COMPLETED,
            session_id=session.id,
            interaction_id="interaction",
            payload={
                "execution_profile_fingerprint": session.metadata["cayu:execution_profile"][
                    "expected"
                ]["fingerprint"]
            },
        ),
    )
    if messages is None:
        messages = [Message.text("assistant", "completed result")]
    pointer = ModelStepPublicationCheckpoint(
        logical_step_id="step",
        stage_id="stage",
        source_transcript_cursor=len(messages) - 1,
        transcript_end_cursor=len(messages),
        completion_event_id="completed",
        classification={"type": "final"},
        assistant_message_published=True,
    )
    await sessions.append_transcript_messages_and_transform_checkpoint(
        session.id,
        messages,
        lambda _session, checkpoint: {
            **(checkpoint or {}),
            LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY: pointer.model_dump(mode="json"),
        },
    )
    return (
        application,
        participant,
        ContextViewPublicationRequest(
            source_session_id=session.id,
            source_session_instance_id=session.instance_id,
            view_id="view",
            interaction_id="interaction",
            boundary_id="step",
            projection_schema="whole-turn.v1",
            publication_key="publish",
        ),
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_component_composes_complete_view_lifecycle(backend, request, sqlite_resources):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def run():
        async with sqlite_resources as resources:
            if backend == "sqlite":
                sessions = resources.own(SQLiteSessionStore(resources.path()))
            elif backend == "postgres":
                sessions = resources.own(PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE))
            else:
                sessions = InMemorySessionStore()
            application, participant, publication = await completed_source(sessions)

            async def resolve(value):
                assert value in {"source-alias", publication.source_session_id}
                return publication.source_session_id, publication.source_session_id

            common = dict(
                session_store=sessions,
                participants=application._participant_coordinator,
                participant=participant,
                context=CONTEXT,
            )
            projected = []

            def project(source):
                projected.append(source)
                return ContextViewExtensionProjection(
                    source_commitment=source.commitment, projection_json='{"note":"history"}'
                )

            try:
                # No application controller, provider, environment or resource resolver is
                # involved in these operations; the app above only prepares the source.
                options = dict(
                    **common,
                    resolve_session=resolve,
                    redactor=SecretRedactor(),
                    extensions=(
                        ContextViewExtensionRegistration(
                            extension="history", schema_version=1, project=project
                        ),
                    ),
                )
                aliased = publication.model_copy(update={"source_session_id": "source-alias"})
                manifest = await views.publish_completed_context_view(aliased, **options)
                assert projected and manifest.resource_references_json is None
                assert await views.publish_completed_context_view(aliased, **options) == manifest
                assert len(projected) == 1  # Replay does not recapture or reproject history.
                selected = await views.select_context_view(
                    ContextViewSelectionRequest(
                        source_owner=participant.owner,
                        source_session_id="source-alias",
                        source_session_instance_id=publication.source_session_instance_id,
                        selector="exact",
                        exact_view_id=manifest.view_id,
                        projection_schema=manifest.projection_schema,
                        extension_set_commitment=manifest.extension_set_commitment,
                        limits=ContextViewLimits(),
                        selection_key="selection",
                    ),
                    resolve_session=resolve,
                    **common,
                )
                assert (
                    await views.read_context_view(
                        manifest.view_id,
                        source_session_id="source-alias",
                        resolve_session=resolve,
                        **common,
                    )
                ).view == manifest
                denied = dict(common, context=CollaborationAccessContext(principal="stranger"))
                with pytest.raises(CollaborationAccessDenied):
                    await views.read_context_view(
                        manifest.view_id,
                        source_session_id="source-alias",
                        resolve_session=resolve,
                        **denied,
                    )
                released = await views.transition_context_view_ownership(
                    ContextViewOwnershipRequest(
                        selection_key=selected.selection_key,
                        view_id=manifest.view_id,
                        pin_commitment=selected.pin_commitment,
                        expected_state=selected.state,
                        expected_revision=selected.ownership_revision,
                        operation="release",
                        current_owner=selected.owner,
                        operation_key="release",
                    ),
                    **common,
                )
                assert released.state == "released"
                events = await views.read_context_view_lifecycle_events(
                    manifest.view_id,
                    source_session_id="source-alias",
                    resolve_session=resolve,
                    **common,
                )
                assert events and events[-1].state == "released"
            finally:
                await application.aclose()

    asyncio.run(run())


def test_component_imports_without_application_or_execution_controllers():
    script = """
import importlib.abc
import sys
blocked = {"cayu.applications", "cayu.runtime._session_engine",
           "cayu.runtime._model_step_executor", "cayu.runtime._recovery_coordinator",
           "cayu.runtime._tool_round_executor"}
class RejectControllers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError(f"Context views imported {fullname}")
sys.meta_path.insert(0, RejectControllers())
from cayu import _application_context_views
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("assembled", [False, True])
@pytest.mark.parametrize("outcome", ["success", "resource_failure", "store_failure"])
def test_resource_publication_settles_safely_after_cancellation(
    assembled, outcome, tmp_path, monkeypatch
):
    secret = "private-retained-view-store"
    monkeypatch.setattr(InMemorySessionStore, "__repr__", lambda self: secret)

    async def run():
        sessions = InMemorySessionStore()
        artifacts = LocalArtifactStore(tmp_path / "artifacts")
        artifact = await artifacts.put_bytes(
            TINY_PNG_BYTES,
            filename="result.png",
            content_type="image/png",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="local",
        )
        attachment = FileAttachment(
            artifact_id=artifact.id,
            kind="image",
            filename=artifact.filename,
            content_type=artifact.content_type,
            size_bytes=artifact.size_bytes,
        )
        environment = Environment(
            EnvironmentSpec(
                name="local",
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name="retained-view-environment",
                    behavior_version="1",
                    implementation_version="1",
                ),
            ),
            artifact_store=artifacts,
        )
        application, participant, publication = await completed_source(
            sessions,
            environment=environment,
            messages=[
                Message(
                    role="user", content=(FilePart(attachment=attachment.model_dump(mode="json")),)
                ),
                Message.text("assistant", "completed result"),
            ],
        )
        owner = LocalArtifactResourceOwner(
            tmp_path / "owner",
            owner=participant.owner,
            artifact_store=artifacts,
        )
        entered = asyncio.Event()
        release = asyncio.Event()
        caller = None
        try:
            acquisition = command(participant.owner, artifact.id, store=artifacts)
            permit = preparation_permit(acquisition)
            permit = permit.model_copy(
                update={
                    "intent": permit.intent.model_copy(
                        update={
                            "request": permit.intent.request.model_copy(
                                update={"participant": participant}
                            )
                        }
                    )
                }
            )
            prepared = await owner.authorize(acquisition, permit=permit)
            receipt = await owner.acquire(acquisition, preparation=prepared)
            publication = publication.model_copy(update={"resource_receipts": (receipt,)})
            original_readback = owner._readback
            original_publish = InMemorySessionStore.publish_context_view
            published = []

            async def readback(command):
                entered.set()
                await release.wait()
                if outcome == "resource_failure":
                    raise ResourceOwnerUnavailable("Receipt readback unavailable")
                return await original_readback(command)

            async def publish(store, manifest, *, publication_key):
                published.append(publication_key)
                if outcome == "store_failure":
                    raise RuntimeError("Publication unavailable")
                return await original_publish(store, manifest, publication_key=publication_key)

            monkeypatch.setattr(owner, "_readback", readback)
            monkeypatch.setattr(InMemorySessionStore, "publish_context_view", publish)
            arguments = dict(participant=participant, context=CONTEXT, resource_owner=owner)
            if assembled:
                target = application.publish_completed_context_view
            else:
                target = views.publish_completed_context_view
                arguments.update(
                    session_store=sessions,
                    participants=application._participant_coordinator,
                    resolve_session=application._resolve_public_session_authority,
                    redactor=application._secret_redactor,
                    get_registered_environment=application._get_registered_environment_for_session,
                    get_registered_agent=application._get_registered_agent,
                    process_identity=application._execution_profile_process_identity,
                )
            caller = asyncio.create_task(target(publication, **arguments))
            await asyncio.wait_for(entered.wait(), 5)
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(caller, 5)
            assert not published
            release.set()
            if outcome == "success":
                await asyncio.wait_for(owner.drain(), 5)
            else:
                error, message = (
                    (ResourceOwnerUnavailable, "Receipt readback unavailable")
                    if outcome == "resource_failure"
                    else (RuntimeError, "Publication unavailable")
                )
                with pytest.raises(error, match=message) as failure:
                    await asyncio.wait_for(owner.drain(), 5)
                tb = failure.value.__traceback__
                while tb:
                    if tb.tb_frame.f_globals.get("__name__", "").startswith("cayu."):
                        assert secret not in repr(tb.tb_frame.f_locals)
                    tb = tb.tb_next
            manifest = await sessions.lookup_context_view_publication(publication.publication_key)
            if outcome == "success":
                assert manifest is not None and manifest.view_id == publication.view_id
            else:
                assert manifest is None
            assert published == (
                [] if outcome == "resource_failure" else [publication.publication_key]
            )
        finally:
            release.set()
            if caller is not None:
                await asyncio.gather(caller, return_exceptions=True)
            try:
                await owner.drain()
            finally:
                await application.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("assembled", [False, True])
@pytest.mark.parametrize(
    "operation",
    [
        "publish_completed_context_view",
        "select_context_view",
        "transition_context_view_ownership",
        "read_context_view",
        "read_context_view_lifecycle_events",
    ],
)
def test_rejected_view_operation_does_not_retain_dependency_diagnostics(assembled, operation):
    secret = "private-view-dependency"

    class UnsupportedStore(InMemorySessionStore):
        context_view_version = None

        def __repr__(self):
            return secret

    async def resolve(value):
        raise AssertionError("Unsupported stores must be rejected before resolution")

    async def run():
        sessions = UnsupportedStore()
        redactor = SecretRedactor(secret)
        participants = ParticipantCoordinator(store=None, registration=None, redactor=redactor)
        participant = ParticipantRef(
            owner=OwnerRef(application_scope="app", owner_id="owner", incarnation="owner-1"),
            participant_id="participant",
            incarnation="participant-1",
        )
        arguments = dict(participant=participant, context=CONTEXT)
        if operation == "publish_completed_context_view":
            value = ContextViewPublicationRequest(
                source_session_id="session",
                source_session_instance_id="instance",
                view_id="view",
                interaction_id="interaction",
                boundary_id="step",
                projection_schema="whole-turn.v1",
                publication_key="publication",
            )
        elif operation == "select_context_view":
            value = ContextViewSelectionRequest(
                source_owner=participant.owner,
                source_session_id="session",
                source_session_instance_id="instance",
                selector="latest",
                projection_schema="whole-turn.v1",
                extension_set_commitment="sha256:" + "0" * 64,
                limits=ContextViewLimits(),
                selection_key="selection",
            )
        elif operation == "transition_context_view_ownership":
            value = ContextViewOwnershipRequest(
                selection_key="selection",
                view_id="view",
                pin_commitment="sha256:" + "0" * 64,
                expected_state="selected",
                expected_revision=1,
                operation="release",
                current_owner=participant.owner,
                operation_key="release",
            )
        else:
            value = "view"
            arguments["source_session_id"] = "session"
        app = None
        if assembled:
            app = CayuApp(session_store=sessions, secret_redactor=redactor, enable_logging=False)
            target = getattr(app, operation)
        else:
            target = getattr(views, operation)
            arguments.update(session_store=sessions, participants=participants)
            if operation != "transition_context_view_ownership":
                arguments["resolve_session"] = resolve
            if operation == "publish_completed_context_view":
                arguments["redactor"] = redactor
        try:
            with pytest.raises(RuntimeError, match="does not support context views") as failure:
                await target(value, **arguments)
            tb = failure.value.__traceback__
            while tb:
                if tb.tb_frame.f_globals.get("__name__", "").startswith("cayu."):
                    assert secret not in repr(tb.tb_frame.f_locals)
                tb = tb.tb_next
        finally:
            if app is not None:
                await app.aclose()

    asyncio.run(run())
