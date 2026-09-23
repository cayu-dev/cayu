from __future__ import annotations

import asyncio
import hashlib
import json
import warnings

import pytest

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.applications import CayuApp
from cayu.artifacts.attachments import file_attachment
from cayu.collaboration._contracts import OwnerRef
from cayu.collaboration.participants import ParticipantRef
from cayu.events import Event, EventType
from cayu.messages import ToolResultPart
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.base import (
    CompactSessionRequest,
    EventQuery,
    InMemorySessionStore,
    Message,
    ResumeRequest,
    RunRequest,
    SessionIdentity,
    SessionRunFenced,
    SessionStatusConflict,
)
from cayu.sessions.context_views import (
    ContextViewExtensionProjection,
    ContextViewExtensionRegistration,
    ContextViewLimits,
    ContextViewManifest,
    ContextViewOwnershipRequest,
    ContextViewProjectionSource,
    ContextViewPublicationRequest,
    ContextViewReadback,
    ContextViewSelectionRequest,
    ParticipantSessionBinding,
    ParticipantSessionCreationReceipt,
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
    RecipientSessionCreationReceipt,
    RecipientSessionCreationRequest,
    project_context_view_extensions,
)
from cayu.storage.sqlite import SQLiteSessionStore


def _manifest() -> ContextViewManifest:
    owner = OwnerRef(application_scope="app", owner_id="session-owner", incarnation="owner-1")
    participant = ParticipantRef(owner=owner, participant_id="participant", incarnation="p-1")
    messages_json = json.dumps(
        [Message.text("assistant", "done").model_dump(mode="json")],
        sort_keys=True,
        separators=(",", ":"),
    )
    extension_set_commitment = "sha256:" + hashlib.sha256(b"[]").hexdigest()
    material = {
        "schema_version": 1,
        "source_owner": owner.model_dump(mode="json"),
        "participant": participant.model_dump(mode="json"),
        "source_session_id": "session",
        "source_session_instance_id": "session-1",
        "view_id": "view-1",
        "interaction_id": "interaction-1",
        "boundary_id": "boundary-1",
        "completion_event_id": "completed-1",
        "transcript_cursor": 1,
        "projection_schema": "whole-turn.v1",
        "extension_set_commitment": extension_set_commitment,
        "messages_json": messages_json,
        "application_context_json": None,
        "extensions": [],
        "historical_ancestry_json": "{}",
        "causal_budget_ancestry_json": "{}",
        "resource_references_json": None,
        "compaction_json": None,
        "messages_commitment": "sha256:" + hashlib.sha256(messages_json.encode()).hexdigest(),
        "application_context_commitment": "sha256:" + hashlib.sha256(b"null").hexdigest(),
    }
    material["manifest_commitment"] = (
        "sha256:"
        + hashlib.sha256(
            canonical_bounded_durable_json_bytes(
                material,
                "context view manifest",
                max_bytes=8 * 1024 * 1024,
                max_nodes=8192,
                max_nesting=64,
            )
        ).hexdigest()
    )
    return ContextViewManifest.model_validate(material)


async def _manifest_for_store(store, session_id="session") -> ContextViewManifest:
    session = await store.load(session_id)
    if session is None:
        session = await store.create(
            RunRequest(agent_name="agent", session_id=session_id, messages=[]),
            identity=SessionIdentity(provider_name="provider", model="model"),
        )
    return _replace_manifest(
        _manifest(), source_session_id=session.id, source_session_instance_id=session.instance_id
    )


def _replace_manifest(manifest, **changes) -> ContextViewManifest:
    material = manifest.model_dump(mode="json")
    material.update(changes)
    material.pop("manifest_commitment")
    material["manifest_commitment"] = (
        "sha256:"
        + hashlib.sha256(
            canonical_bounded_durable_json_bytes(
                material,
                "context view manifest",
                max_bytes=8 * 1024 * 1024,
                max_nodes=8192,
                max_nesting=64,
            )
        ).hexdigest()
    )
    return ContextViewManifest.model_validate(material)


def test_manifest_is_positive_schema_and_historical_only() -> None:
    manifest = _manifest()
    readback = ContextViewReadback(view=manifest)
    assert readback.historical_only is True
    assert [Message.model_validate(item) for item in json.loads(manifest.messages_json)] == [
        Message.text("assistant", "done")
    ]
    assert "provider" not in manifest.messages_json


def test_manifest_commitment_rejects_post_construction_material_change() -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="Message commitment"):
        ContextViewManifest.model_validate(
            {**manifest.model_dump(mode="json"), "messages_json": "[]"}
        )


def test_selection_requires_exact_or_minimum_selector_evidence() -> None:
    owner = OwnerRef(application_scope="app", owner_id="owner", incarnation="owner-1")
    with pytest.raises(ValueError, match="Exact selection"):
        ContextViewSelectionRequest(
            source_owner=owner,
            source_session_id="session",
            source_session_instance_id="session-1",
            selector="exact",
            projection_schema="whole-turn.v1",
            extension_set_commitment="sha256:" + "0" * 64,
            limits=ContextViewLimits(),
            selection_key="selection-1",
        )


def test_ownership_operation_keys_reserve_store_expiry_namespace() -> None:
    with pytest.raises(ValueError, match="reserved expiry namespace"):
        ContextViewOwnershipRequest(
            selection_key="selection-1",
            view_id="view-1",
            pin_commitment="sha256:" + "0" * 64,
            expected_state="selected",
            expected_revision=1,
            operation="adopt",
            current_owner=OwnerRef(
                application_scope="app", owner_id="owner", incarnation="owner-1"
            ),
            destination_owner=OwnerRef(
                application_scope="app", owner_id="owner", incarnation="owner-1"
            ),
            operation_key="expiry:selection-1:1",
        )


def test_context_view_limits_are_finite() -> None:
    limits = ContextViewLimits()
    assert limits.max_views > 0
    with pytest.raises(ValueError):
        ContextViewLimits(max_views=0)


def test_postgres_context_view_admission_lock_order_is_canonical() -> None:
    """Publication/selection shared locks must be acquired owner-first."""

    from cayu.storage.postgres import PostgresSessionStore

    owner = OwnerRef(application_scope="app", owner_id="owner", incarnation="owner-1")
    calls: list[str] = []

    class Cursor:
        async def execute(self, query, params=()):
            assert "pg_advisory_xact_lock" in query
            assert params
            value = params[0]
            if value.startswith("context-view-owner:"):
                calls.append("owner")
            elif value == "context-view-lifecycle":
                calls.append("lifecycle")
            elif value.startswith("context-view-session:"):
                calls.append("session")
            else:  # pragma: no cover - protects the lock-order assertion
                raise AssertionError(value)

    async def run() -> None:
        store = object.__new__(PostgresSessionStore)
        await store._lock_context_view_admission(
            Cursor(), owner=owner, lifecycle=True, session_id="session"
        )

    asyncio.run(run())
    assert calls == ["owner", "lifecycle", "session"]


def test_context_view_operation_keys_are_bounded() -> None:
    with pytest.raises(ValueError, match="512 UTF-8 bytes"):
        ContextViewPublicationRequest(
            source_session_id="session",
            source_session_instance_id="instance",
            view_id="view",
            interaction_id="interaction",
            boundary_id="boundary",
            projection_schema="whole-turn.v1",
            publication_key="k" * 513,
        )
    with pytest.raises(ValueError, match="512 UTF-8 bytes"):
        ContextViewSelectionRequest(
            source_owner=OwnerRef(application_scope="app", owner_id="owner", incarnation="owner-1"),
            source_session_id="session",
            source_session_instance_id="instance",
            selector="latest",
            projection_schema="whole-turn.v1",
            extension_set_commitment="sha256:" + "0" * 64,
            limits=ContextViewLimits(),
            selection_key="k" * 513,
        )


def test_context_view_selection_enforces_view_and_retained_byte_limits() -> None:
    asyncio.run(_test_context_view_selection_enforces_view_and_retained_byte_limits())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_context_view_publication_key_validation_across_backends(
    backend: str, tmp_path, request
) -> None:
    postgres_dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None
    asyncio.run(_test_context_view_publication_key_validation(backend, tmp_path, postgres_dsn))


async def _test_context_view_publication_key_validation(
    backend: str, tmp_path, postgres_dsn: str | None
) -> None:
    if backend == "memory":
        store = InMemorySessionStore()
    elif backend == "sqlite":
        store = SQLiteSessionStore(tmp_path / "publication-key.sqlite")
    elif backend == "postgres":
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        assert postgres_dsn is not None
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
    else:  # pragma: no cover - guarded by the parameter list
        raise AssertionError(backend)
    try:
        manifest = await _manifest_for_store(store, session_id=f"publication-key-{backend}")
        manifest = _replace_manifest(manifest, view_id=f"publication-key-view-{backend}")
        for invalid_key in ("", " \t", "leading ", "control\x00key"):
            with pytest.raises((TypeError, ValueError)):
                await store.publish_context_view(manifest, publication_key=invalid_key)
        published = await store.publish_context_view(
            manifest, publication_key=f"valid-publication-{backend}"
        )
        assert published == manifest
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            await close()


async def _test_context_view_selection_enforces_view_and_retained_byte_limits() -> None:
    from cayu.sessions.context_views import context_view_manifest_bytes

    store = InMemorySessionStore()
    manifest = _manifest()
    manifest = await _manifest_for_store(store)
    await store.publish_context_view(manifest, publication_key="byte-publication")
    size = context_view_manifest_bytes(manifest)
    base = dict(
        source_owner=manifest.source_owner,
        source_session_id=manifest.source_session_id,
        source_session_instance_id=manifest.source_session_instance_id,
        selector="latest",
        projection_schema=manifest.projection_schema,
        extension_set_commitment=manifest.extension_set_commitment,
    )
    with pytest.raises(OverflowError, match="byte limit"):
        await store.select_context_view(
            ContextViewSelectionRequest(
                **base,
                limits=ContextViewLimits(max_view_bytes=size - 1),
                selection_key="byte-too-small",
            )
        )
    await store.select_context_view(
        ContextViewSelectionRequest(
            **base,
            limits=ContextViewLimits(max_view_bytes=size, max_retained_bytes=size),
            selection_key="byte-exact",
        )
    )
    duplicate = await store.select_context_view(
        ContextViewSelectionRequest(
            **base,
            limits=ContextViewLimits(max_view_bytes=size, max_retained_bytes=size),
            selection_key="retained-duplicate",
        )
    )
    assert duplicate.view.view_id == manifest.view_id


def test_extension_registration_requires_bounded_named_producer() -> None:
    registration = ContextViewExtensionRegistration(
        extension="prompt",
        schema_version=1,
        project=lambda value: _extension_projection(value, {"text": value.messages_json}),
    )
    assert registration.extension == "prompt"
    with pytest.raises(ValueError):
        ContextViewExtensionRegistration(extension="", schema_version=1, project=lambda _: None)


def _projection_source() -> ContextViewProjectionSource:
    return ContextViewProjectionSource(
        source_session_id="session",
        source_session_instance_id="instance",
        participant_json="{}",
        interaction_id="interaction",
        boundary_id="boundary",
        completion_event_id="completion",
        source_transcript_cursor=0,
        transcript_cursor=1,
        messages_json="[]",
        compaction_json='{"input_frontier":0}',
        historical_ancestry_json="{}",
    )


def _extension_projection(source, value) -> ContextViewExtensionProjection:
    return ContextViewExtensionProjection(
        source_commitment=source.commitment,
        projection_json=canonical_bounded_durable_json_bytes(
            value, "projection", max_bytes=256 * 1024, max_nodes=8192
        ).decode(),
    )


def test_extension_projection_is_canonical_and_all_or_nothing() -> None:
    registrations = (
        ContextViewExtensionRegistration(
            extension="prompt",
            schema_version=1,
            project=lambda s: _extension_projection(s, {"value": 2, "label": "count"}),
        ),
        ContextViewExtensionRegistration(
            extension="optional", schema_version=1, project=lambda _: None
        ),
    )
    records, commitment = project_context_view_extensions(registrations, _projection_source())
    assert records[0].projection_json == '{"label":"count","value":2}'
    assert records[1].state == "absent"
    assert commitment.startswith("sha256:")
    failing = (
        ContextViewExtensionRegistration(
            extension="ok", schema_version=1, project=lambda s: _extension_projection(s, {})
        ),
        ContextViewExtensionRegistration(
            extension="bad",
            schema_version=1,
            project=lambda s: _extension_projection(s, {"not": object()}),
        ),
    )
    with pytest.raises(ValueError):
        project_context_view_extensions(failing, _projection_source())
    for unsafe in (
        {"resource_reference": "artifact-1"},
        {"lease": "lease-1"},
        {"tool": "runner"},
        {"dispatch_receipt": {"id": "receipt-1"}},
        {"provider_stage": {"stage": "live"}},
        {"human": {"approval": "required"}},
        {"input_ownership": {"owner": "caller"}},
    ):
        with pytest.raises(ValueError, match="unsupported live-authority"):
            project_context_view_extensions(
                (
                    ContextViewExtensionRegistration(
                        extension="unsafe",
                        schema_version=1,
                        project=lambda s, value=unsafe: _extension_projection(s, value),
                    ),
                ),
                _projection_source(),
            )
    records, _ = project_context_view_extensions(
        (
            ContextViewExtensionRegistration(
                extension="profile",
                schema_version=1,
                project=lambda s: _extension_projection(
                    s,
                    {
                        "provider_name": "historical-provider",
                        "model": "historical-model",
                    },
                ),
            ),
        ),
        _projection_source(),
    )
    assert records[0].state == "present"


def test_application_rejects_duplicate_context_extensions() -> None:
    extension = ContextViewExtensionRegistration(
        extension="prompt", schema_version=1, project=lambda _: None
    )
    with pytest.raises(ValueError, match="duplicate"):
        CayuApp(
            enable_logging=False,
            context_view_extensions=(extension, extension),
        )


def test_application_fails_closed_for_incomplete_context_view_store() -> None:
    class IncompleteContextViewStore(InMemorySessionStore):
        context_view_version = None

    extension = ContextViewExtensionRegistration(
        extension="prompt", schema_version=1, project=lambda _: None
    )
    with pytest.raises(RuntimeError, match="does not yet support"):
        CayuApp(
            enable_logging=False,
            session_store=IncompleteContextViewStore(),
            context_view_extensions=(extension,),
        )


def test_creation_request_requires_stable_key_and_run_request() -> None:
    request = RunRequest(agent_name="agent", messages=[])
    intent = ParticipantSessionCreationRequest(request=request, creation_key="create-1")
    assert intent.creation_key == "create-1"
    request.metadata["later"] = "change"
    assert "later" not in intent.request.metadata
    with pytest.raises(ValueError):
        ParticipantSessionCreationRequest(request=request, creation_key="")


def test_public_participant_session_creation_is_inert_and_replayable(
    capsys, caplog, recwarn
) -> None:
    asyncio.run(_test_public_participant_session_creation_is_inert_and_replayable())
    secrets = (
        'quoted-credential-"canary',
        "backslash-credential-\\canary",
        "multiline-credential\ncanary",
    )
    asyncio.run(_assert_historical_prompt_secrets_rejected(secrets))
    captured = capsys.readouterr()
    diagnostics = (
        captured.out + captured.err + caplog.text + "".join(str(item.message) for item in recwarn)
    )
    for secret in secrets:
        assert secret not in diagnostics
        assert json.dumps(secret)[1:-1] not in diagnostics


async def _assert_historical_prompt_secrets_rejected(secrets) -> None:
    import base64

    from pydantic import SecretStr
    from tests.core.test_participant_identity import CONTEXT, app, create, registration

    from cayu.agents import AgentSpec
    from cayu.collaboration.memory import InMemoryCollaborationStore
    from cayu.evals.testing import ScriptedModelProvider
    from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
    from cayu.vaults import SecretRedactor

    codec = PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={
                "test": SecretStr(base64.urlsafe_b64encode(bytes([31]) * 32).decode().rstrip("="))
            },
        )
    )
    for index, secret in enumerate(secrets):
        store = InMemorySessionStore(public_authority_alias_codec=codec)
        application = app(
            InMemoryCollaborationStore(),
            registration(),
            session_store=store,
            secret_redactor=SecretRedactor([secret]),
        )
        provider = ScriptedModelProvider([])
        application.register_provider(provider, default=True)
        application.register_agent(
            AgentSpec(name="reviewer", model="model", system_prompt=f"Private value: {secret}")
        )
        initialized = await application.initialize_collaboration()
        _, receipt = await create(application, initialized, key=f"secret-participant-{index}")
        creation = ParticipantSessionCreationRequest(
            request=RunRequest(
                agent_name="reviewer", session_id=f"secret-session-{index}", messages=[]
            ),
            creation_key=f"secret-creation-{index}",
        )
        with pytest.raises(ValueError, match="contains a workload secret") as failure:
            await application.create_participant_session(
                creation, participant=receipt.participants[0].reference, context=CONTEXT
            )
        assert secret not in str(failure.value) + repr(failure.value)
        assert await store.load(creation.request.session_id) is None
        assert await store.lookup_participant_session_creation(creation) is None
        assert provider.requests == []


def test_public_participant_session_execution_is_authenticated_and_replayable() -> None:
    asyncio.run(_test_public_participant_session_execution_is_authenticated_and_replayable())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_public_participant_context_view_witness_across_backends(
    backend: str, tmp_path, request
) -> None:
    """Exercise the public creation/execution/publication journey on each backend."""

    postgres_dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None
    asyncio.run(_test_public_participant_context_view_witness(backend, tmp_path, postgres_dsn))


async def _test_public_participant_context_view_witness(
    backend: str, tmp_path, postgres_dsn: str | None
) -> None:
    from tests.artifacts.test_resources import (
        LocalArtifactResourceOwner,
        _transfer_command,
        authorized,
        command,
        make_store,
    )
    from tests.core.test_explicit_session_compaction import RecordingCompactor
    from tests.core.test_participant_identity import CONTEXT, app, create, registration

    from cayu.agents import AgentSpec
    from cayu.collaboration.memory import InMemoryCollaborationStore
    from cayu.context import CheckpointCompactionContextPolicy
    from cayu.evals.testing import ScriptedModelProvider

    if backend == "memory":
        session_store = InMemorySessionStore()
        collaboration_store = InMemoryCollaborationStore()
    elif backend == "sqlite":
        from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore

        session_store = SQLiteSessionStore(tmp_path / "witness-sessions.sqlite")
        collaboration_store = SQLiteCollaborationStore(tmp_path / "witness-collaboration.sqlite")
    elif backend == "postgres":
        from cayu.storage.collaboration_postgres import PostgresCollaborationStore
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        assert postgres_dsn is not None
        session_store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        collaboration_store = PostgresCollaborationStore(
            postgres_dsn, schema_mode=SchemaMode.CREATE
        )
    else:  # pragma: no cover - guarded by the parameter list
        raise AssertionError(backend)

    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("first turn"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("continued turn"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("third turn"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    collaboration_registration = registration()
    wrong_boundary = [True]
    unsafe_projection = [None]

    def project_history(source):
        if unsafe_projection[0] is not None:
            return _extension_projection(source, unsafe_projection[0])
        with pytest.raises(ValueError, match="frozen"):
            source.boundary_id = "later-turn"
        result = _extension_projection(
            source,
            {"text": source.messages_json, "items": [{"label": "status", "value": "completed"}]},
        )
        if wrong_boundary[0]:
            return result.model_copy(update={"source_commitment": "sha256:" + "0" * 64})
        return result

    extension = ContextViewExtensionRegistration(
        extension="boundary-history",
        schema_version=1,
        project=project_history,
    )
    extensions = (
        extension,
        ContextViewExtensionRegistration(
            extension="optional-history",
            schema_version=1,
            project=lambda _: None,
        ),
    )
    application = app(
        collaboration_store,
        collaboration_registration,
        session_store=session_store,
        context_view_extensions=extensions,
    )
    application.register_provider(provider, default=True)
    original_spec = AgentSpec(
        name="reviewer", model="model", system_prompt="original historical prompt"
    )
    context_policy = CheckpointCompactionContextPolicy(
        compactor=RecordingCompactor(), max_user_turns=1, compact_after_messages=100
    )
    application.register_agent(original_spec, context_policy=context_policy)
    try:
        initialized = await application.initialize_collaboration()
        _, participant_receipt = await create(application, initialized, key=f"witness-{backend}")
        participant = participant_receipt.participants[0].reference
        creation = ParticipantSessionCreationRequest(
            request=RunRequest(
                agent_name="reviewer",
                messages=[Message.text("user", "start")],
            ),
            creation_key=f"witness-create-{backend}",
        )
        session, _ = await application.create_participant_session(
            creation,
            participant=participant,
            context=CONTEXT,
        )
        assert provider.requests == []

        execution = ParticipantSessionExecutionRequest(
            request=creation.request.model_copy(update={"session_id": session.id}),
            session_instance_id=session.instance_id,
            execution_key=f"witness-execute-{backend}",
        )
        events = [
            event
            async for event in application.execute_participant_session(
                execution,
                participant=participant,
                context=CONTEXT,
            )
        ]
        assert any(event.type is EventType.SESSION_COMPLETED for event in events)
        checkpoint = await session_store.load_checkpoint(session.id)
        from cayu.runtime._model_completion_publication import (
            model_step_publication_from_checkpoint,
        )

        pointer = model_step_publication_from_checkpoint(checkpoint)
        assert pointer is not None
        completion = await session_store.query_events(
            EventQuery(session_id=session.id, event_id=pointer.completion_event_id, limit=1)
        )
        assert len(completion) == 1 and completion[0].event.interaction_id is not None
        publication_request = ContextViewPublicationRequest(
            source_session_id=session.id,
            source_session_instance_id=session.instance_id,
            view_id=f"witness-view-{backend}",
            interaction_id=completion[0].event.interaction_id,
            boundary_id=pointer.logical_step_id,
            projection_schema="whole-turn.v1",
            publication_key=f"witness-publication-{backend}",
        )
        with pytest.raises(ValueError, match="another boundary"):
            await application.publish_completed_context_view(
                publication_request,
                participant=participant,
                context=CONTEXT,
            )
        assert (
            await session_store.lookup_context_view_publication(publication_request.publication_key)
            is None
        )
        wrong_boundary[0] = False
        for unsafe in (
            {"tool_call": {"name": "send_email"}},
            {"provider": {"stage": "active"}},
            {"tool_calls": [{"name": "send_email"}]},
            {"items": [{"tool_call": {"name": "send_email"}}]},
            {"text": {"provider_stage": "active"}},
            {"value": {"lease": "live"}},
            {"dispatch_receipt": {}},
            {"human": {}},
            {"input_ownership": {}},
            {"ticket": {}},
            {"reservation": {}},
            {"allocation": {}},
            {"cleanup": {}},
        ):
            unsafe_projection[0] = unsafe
            with pytest.raises(ValueError, match="unsupported live-authority"):
                await application.publish_completed_context_view(
                    publication_request,
                    participant=participant,
                    context=CONTEXT,
                )
            assert (
                await session_store.lookup_context_view_publication(
                    publication_request.publication_key
                )
                is None
            )
        unsafe_projection[0] = None
        publisher = app(
            collaboration_store,
            collaboration_registration,
            session_store=session_store,
            context_view_extensions=extensions,
        )
        publisher.register_provider(provider, default=True)
        publisher.register_agent(
            AgentSpec(name="reviewer", model="model", system_prompt="replacement prompt")
        )
        await publisher.initialize_collaboration()
        original_snapshot_loader = session_store.capture_context_view_publication_source
        advance_once = True

        async def advancing_snapshot(session_id):
            nonlocal advance_once
            snapshot = await original_snapshot_loader(session_id)
            if advance_once:
                advance_once = False
                advanced = [
                    event
                    async for event in application.resume(
                        ResumeRequest(
                            session_id=session.id,
                            messages=[Message.text("user", "advance during publication")],
                        ),
                        context=CONTEXT,
                    )
                ]
                assert any(event.type is EventType.SESSION_COMPLETED for event in advanced)
            return snapshot

        session_store.capture_context_view_publication_source = advancing_snapshot
        try:
            manifest = await publisher.publish_completed_context_view(
                publication_request,
                participant=participant,
                context=CONTEXT,
            )
        finally:
            session_store.capture_context_view_publication_source = original_snapshot_loader
        assert "first turn" in manifest.extensions[0].projection_json
        assert "continued turn" not in manifest.extensions[0].projection_json
        selection = await application.select_context_view(
            ContextViewSelectionRequest(
                source_owner=participant.owner,
                source_session_id=session.id,
                source_session_instance_id=session.instance_id,
                selector="latest",
                projection_schema=manifest.projection_schema,
                extension_set_commitment=manifest.extension_set_commitment,
                limits=ContextViewLimits(),
                selection_key=f"witness-selection-{backend}",
            ),
            participant=participant,
            context=CONTEXT,
        )
        assert selection.view == manifest
        later_pointer = model_step_publication_from_checkpoint(
            await session_store.load_checkpoint(session.id)
        )
        assert later_pointer is not None
        later_completion = await session_store.query_events(
            EventQuery(
                session_id=session.id,
                event_id=later_pointer.completion_event_id,
                limit=1,
            )
        )
        later_manifest = await application.publish_completed_context_view(
            ContextViewPublicationRequest(
                source_session_id=session.id,
                source_session_instance_id=session.instance_id,
                view_id=f"later-view-{backend}",
                interaction_id=later_completion[0].event.interaction_id,
                boundary_id=later_pointer.logical_step_id,
                projection_schema=manifest.projection_schema,
                publication_key=f"later-publication-{backend}",
            ),
            participant=participant,
            context=CONTEXT,
        )
        assert later_manifest.transcript_cursor > manifest.transcript_cursor
        assert later_manifest.extension_set_commitment == manifest.extension_set_commitment
        assert (
            later_manifest.extensions[0].source_commitment
            != manifest.extensions[0].source_commitment
        )
        assert later_manifest.extensions[1].state == manifest.extensions[1].state == "absent"
        latest_request = ContextViewSelectionRequest(
            source_owner=participant.owner,
            source_session_id=session.id,
            source_session_instance_id=session.instance_id,
            selector="latest",
            projection_schema=manifest.projection_schema,
            extension_set_commitment=manifest.extension_set_commitment,
            limits=ContextViewLimits(),
            selection_key=f"latest-again-{backend}",
        )
        latest = await application.select_context_view(
            latest_request, participant=participant, context=CONTEXT
        )
        minimum = await application.select_context_view(
            latest_request.model_copy(
                update={
                    "selector": "minimum",
                    "minimum_transcript_cursor": later_manifest.transcript_cursor,
                    "selection_key": f"minimum-again-{backend}",
                }
            ),
            participant=participant,
            context=CONTEXT,
        )
        assert latest.view == minimum.view == later_manifest
        assert (
            await application.select_context_view(
                latest_request.model_copy(
                    update={
                        "selection_key": selection.selection_key,
                    }
                ),
                participant=participant,
                context=CONTEXT,
            )
            == selection
        )
        with pytest.raises(LookupError):
            await application.select_context_view(
                latest_request.model_copy(
                    update={
                        "extension_set_commitment": "sha256:" + "0" * 64,
                        "selection_key": f"wrong-schema-{backend}",
                    }
                ),
                participant=participant,
                context=CONTEXT,
            )
        for additional in (latest, minimum):
            await application.transition_context_view_ownership(
                ContextViewOwnershipRequest(
                    selection_key=additional.selection_key,
                    view_id=additional.view.view_id,
                    pin_commitment=additional.pin_commitment,
                    expected_state=additional.state,
                    expected_revision=additional.ownership_revision,
                    operation="release",
                    current_owner=participant.owner,
                    operation_key=f"release-{additional.selection_key}",
                ),
                participant=participant,
                context=CONTEXT,
            )
        await session_store.validate_context_view_compaction(session.id, manifest.transcript_cursor)
        adopted = await application.transition_context_view_ownership(
            ContextViewOwnershipRequest(
                selection_key=selection.selection_key,
                view_id=manifest.view_id,
                pin_commitment=selection.pin_commitment,
                expected_state="selected",
                expected_revision=selection.ownership_revision,
                operation="adopt",
                current_owner=participant.owner,
                destination_owner=participant.owner,
                operation_key=f"witness-adopt-{backend}",
            ),
            participant=participant,
            destination_participant=participant,
            context=CONTEXT,
        )
        provider_requests_before_recipient = len(provider.requests)
        artifact_store, artifact = await asyncio.to_thread(make_store, tmp_path)
        from tests.core.test_builtin_tools import TINY_PNG_BYTES

        from cayu.artifacts import ArtifactScope
        from cayu.environments import Environment, EnvironmentSpec

        artifact = await artifact_store.put_bytes(
            TINY_PNG_BYTES,
            filename="image.png",
            content_type="image/png",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="resource",
        )
        application.register_environment(
            Environment(EnvironmentSpec(name="resource"), artifact_store=artifact_store)
        )
        source_owner_ref = OwnerRef(
            application_scope=participant.owner.application_scope,
            owner_id="fork-resource-source",
            incarnation="source-1",
        )
        resource_source = LocalArtifactResourceOwner(
            tmp_path / "fork-source", owner=source_owner_ref, artifact_store=artifact_store
        )
        resource_destination = LocalArtifactResourceOwner(
            tmp_path / "fork-destination",
            owner=participant.owner,
            artifact_store=artifact_store,
            participant=participant,
        )
        acquired = await authorized(
            resource_source, command(source_owner_ref, artifact.id, store=artifact_store)
        )
        accepted = await resource_destination.accept_transfer(
            _transfer_command(acquired, source_owner_ref, participant.owner),
            source_owner=resource_source,
        )
        recipient_creation = RecipientSessionCreationRequest(
            request=RunRequest(
                agent_name="reviewer",
                environment_name="resource",
                messages=[
                    Message.text("user", "recipient first input"),
                    Message(
                        role="tool",
                        content=(
                            ToolResultPart(
                                tool_call_id="fork-call",
                                tool_name="read_file",
                                artifacts=[
                                    file_attachment(
                                        artifact_id=artifact.id,
                                        kind="image",
                                        filename=artifact.filename,
                                        content_type="image/png",
                                        size_bytes=artifact.size_bytes,
                                    )
                                ],
                            ),
                        ),
                    ),
                ],
            ),
            creation_key=f"recipient-fork-{backend}",
            recipient=participant,
            mode="fork",
            selected_view=adopted,
            resource_transfers=(accepted,),
            preparation_receipts=(await resource_destination.read_preparation(accepted.command),),
        )
        recipient_session, recipient_receipt = await application.create_recipient_session(
            recipient_creation, context=CONTEXT, resource_owner=resource_destination
        )
        assert recipient_receipt.resource_transfer_commitments == (accepted.operation_digest,)
        assert (await resource_source.readback(acquired.command)).receipt == acquired
        forged_material = recipient_receipt.model_dump(mode="json", exclude={"receipt_commitment"})
        forged_material["source_view_commitment"] = "sha256:" + "0" * 64
        forged_material["receipt_commitment"] = (
            "sha256:"
            + hashlib.sha256(
                canonical_bounded_durable_json_bytes(
                    forged_material,
                    "recipient receipt",
                    max_bytes=512 * 1024,
                    max_nodes=8192,
                    max_nesting=64,
                )
            ).hexdigest()
        )
        forged_handoff = RecipientSessionCreationReceipt.model_validate(forged_material)
        with pytest.raises(PermissionError, match="durable metadata"):
            await application.settle_recipient_resource_handoff(
                forged_handoff,
                resource_owner=resource_destination,
                source_owners=(resource_source,),
                context=CONTEXT,
            )
        assert (await resource_source.readback(acquired.command)).receipt == acquired
        assert recipient_receipt.mode == "fork"
        assert recipient_session.id != session.id
        assert len(provider.requests) == provider_requests_before_recipient
        expected_input = [
            *json.loads(manifest.messages_json),
            Message.text("user", "recipient first input").model_dump(mode="json"),
            Message(
                role="tool",
                content=(
                    ToolResultPart(
                        tool_call_id="fork-call",
                        tool_name="read_file",
                        artifacts=[
                            file_attachment(
                                artifact_id=artifact.id,
                                kind="image",
                                filename=artifact.filename,
                                content_type="image/png",
                                size_bytes=artifact.size_bytes,
                            )
                        ],
                    ),
                ),
            ).model_dump(mode="json"),
        ]
        assert [
            message.model_dump(mode="json")
            for message in await session_store.load_transcript(recipient_session.id)
        ] == expected_input
        assert recipient_receipt.participant_receipt.initial_input_commitment == (
            "sha256:"
            + hashlib.sha256(
                canonical_bounded_durable_json_bytes(
                    expected_input,
                    "expected recipient input",
                    max_bytes=8 * 1024 * 1024,
                    max_nodes=8192,
                    max_nesting=64,
                )
            ).hexdigest()
        )
        forged_selection = adopted.model_copy(
            update={"pin_commitment": "sha256:" + "0" * 64}, deep=True
        )
        with pytest.raises(PermissionError, match="ownership"):
            await application.create_recipient_session(
                RecipientSessionCreationRequest(
                    request=recipient_creation.request,
                    creation_key=f"forged-recipient-{backend}",
                    recipient=participant,
                    mode="fork",
                    selected_view=forged_selection,
                    resource_transfers=(accepted,),
                    preparation_receipts=(
                        await resource_destination.read_preparation(accepted.command),
                    ),
                ),
                context=CONTEXT,
                resource_owner=resource_destination,
            )
        # A frozen model can still be bypassed with model_copy(update=...).
        # The public request boundary must reject that malformed nested value
        # before metadata serialization can emit its contents in a warning.
        malformed_selection = adopted.model_copy(
            update={"owner_participant": "secret-canary"}, deep=True
        )
        with warnings.catch_warnings(record=True) as diagnostics:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="valid selected view receipt"):
                RecipientSessionCreationRequest(
                    request=recipient_creation.request,
                    creation_key=f"malformed-recipient-{backend}",
                    recipient=participant,
                    mode="fork",
                    selected_view=malformed_selection,
                )
        assert all("secret-canary" not in str(item.message) for item in diagnostics)
        with pytest.raises(ValueError, match="conflicts"):
            await application.create_recipient_session(
                RecipientSessionCreationRequest(
                    request=RunRequest(
                        agent_name="reviewer", messages=[Message.text("user", "changed input")]
                    ),
                    creation_key=recipient_creation.creation_key,
                    recipient=participant,
                    mode="fork",
                    selected_view=adopted,
                ),
                context=CONTEXT,
            )
        # A later participant must be authorized even when their first evidence
        # is beyond the public 256-event page, and even for a one-event request.
        _, reader_receipt = await create(application, initialized, key=f"witness-reader-{backend}")
        reader = reader_receipt.participants[0].reference
        for index in range(257):
            destination = reader if index == 256 else participant
            adopted = await application.transition_context_view_ownership(
                ContextViewOwnershipRequest(
                    selection_key=selection.selection_key,
                    view_id=manifest.view_id,
                    pin_commitment=selection.pin_commitment,
                    expected_state=adopted.state,
                    expected_revision=adopted.ownership_revision,
                    operation="transfer",
                    current_owner=participant.owner,
                    destination_owner=destination.owner,
                    operation_key=f"witness-transfer-{backend}-{index}",
                ),
                participant=participant,
                destination_participant=destination,
                context=CONTEXT,
            )
        assert (
            await application.read_context_view(
                manifest.view_id,
                source_session_id=session.id,
                participant=reader,
                context=CONTEXT,
            )
        ).view == manifest
        assert (
            len(
                await application.read_context_view_lifecycle_events(
                    manifest.view_id,
                    source_session_id=session.id,
                    participant=reader,
                    context=CONTEXT,
                    limit=1,
                )
            )
            == 1
        )
        continued = [
            event
            async for event in application.resume(
                ResumeRequest(
                    session_id=session.id,
                    messages=[Message.text("user", "continue")],
                ),
                context=CONTEXT,
            )
        ]
        assert any(event.type is EventType.SESSION_COMPLETED for event in continued)
        assert len(provider.requests) == 3
        readback_application = application
        if backend != "memory":
            session_path = tmp_path / "witness-sessions.sqlite"
            collaboration_path = tmp_path / "witness-collaboration.sqlite"
            await session_store.close()
            await collaboration_store.close()
            if backend == "sqlite":
                from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore

                session_store = SQLiteSessionStore(session_path)
                collaboration_store = SQLiteCollaborationStore(collaboration_path)
            else:
                from cayu.storage.collaboration_postgres import PostgresCollaborationStore
                from cayu.storage.migrations import SchemaMode
                from cayu.storage.postgres import PostgresSessionStore

                assert postgres_dsn is not None
                session_store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
                collaboration_store = PostgresCollaborationStore(
                    postgres_dsn, schema_mode=SchemaMode.VALIDATE
                )
            readback_application = app(
                collaboration_store,
                collaboration_registration,
                session_store=session_store,
            )
            await readback_application.initialize_collaboration()
        recovered_recipient = await readback_application.lookup_recipient_session(
            recipient_creation, context=CONTEXT
        )
        assert recovered_recipient is not None
        assert recovered_recipient[0].id == recipient_session.id
        assert recovered_recipient[1] == recipient_receipt
        # Reconstruct both journal owners and finish the handoff only from the
        # durable child. SQL session/collaboration stores were reopened above.
        resource_source = LocalArtifactResourceOwner(
            tmp_path / "fork-source", owner=source_owner_ref, artifact_store=artifact_store
        )
        resource_destination = LocalArtifactResourceOwner(
            tmp_path / "fork-destination",
            owner=participant.owner,
            artifact_store=artifact_store,
            participant=participant,
        )
        assert (
            await readback_application.reconcile_recipient_resource_handoff(
                recipient_creation,
                resource_owner=resource_destination,
                source_owners=(resource_source,),
                context=CONTEXT,
            )
            == recipient_receipt
        )
        assert (
            await readback_application.reconcile_recipient_resource_handoff(
                recipient_creation,
                resource_owner=resource_destination,
                source_owners=(resource_source,),
                context=CONTEXT,
            )
            == recipient_receipt
        )
        assert (await resource_destination.read_transfer(accepted.command)).receipt == accepted
        from cayu.collaboration._contracts import ExactUnavailable

        assert isinstance(await resource_source.readback(acquired.command), ExactUnavailable)
        with pytest.raises(ValueError):
            await artifact_store.delete(artifact.id)
        replayed_child, replayed_receipt = await readback_application.create_recipient_session(
            recipient_creation, context=CONTEXT, resource_owner=resource_destination
        )
        assert replayed_child.id == recipient_session.id
        assert replayed_receipt == recipient_receipt
        assert len(provider.requests) == 3
        assert [
            message.model_dump(mode="json")
            for message in await session_store.load_transcript(recipient_session.id)
        ] == expected_input
        # The pin is still adopted across parent continuation and durable reopen.
        await session_store.validate_context_view_compaction(session.id, manifest.transcript_cursor)
        readback = await readback_application.read_context_view(
            manifest.view_id,
            source_session_id=session.id,
            participant=participant,
            context=CONTEXT,
        )
        assert readback.view == manifest and readback.historical_only is True
        ancestry = json.loads(manifest.historical_ancestry_json)
        definition = json.loads(ancestry["creation_definition_json"])
        assert definition["rendered_system_prompt"] == "original historical prompt"
        assert definition["agent_definition_commitment"].startswith("sha256:")
        with pytest.raises(ValueError, match="retention pin"):
            await session_store.validate_context_view_source_closure(session.id)
        if backend != "memory":
            readback_application.register_provider(provider, default=True)
            readback_application.register_agent(original_spec, context_policy=context_policy)
        current = await session_store.load(session.id)
        snapshot = await session_store.load_transcript_snapshot(session.id)
        assert current is not None
        compacted = [
            event
            async for event in readback_application.compact_session(
                CompactSessionRequest(
                    session_id=session.id,
                    idempotency_key=f"witness-compaction-success-{backend}",
                    expected_run_epoch=current.run_epoch,
                    expected_transcript_cursor=snapshot.cursor,
                ),
                context=CONTEXT,
            )
        ]
        assert any(event.type is EventType.SESSION_CHECKPOINTED for event in compacted)
        if backend != "memory":
            await session_store.close()
            if backend == "sqlite":
                session_store = SQLiteSessionStore(tmp_path / "witness-sessions.sqlite")
            else:
                session_store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
            readback_application = app(
                collaboration_store, collaboration_registration, session_store=session_store
            )
            await readback_application.initialize_collaboration()
        assert (
            await readback_application.read_context_view(
                manifest.view_id,
                source_session_id=session.id,
                participant=participant,
                context=CONTEXT,
            )
        ).view == manifest
        assert (
            await readback_application.read_context_view(
                manifest.view_id,
                source_session_id=session.id,
                participant=reader,
                context=CONTEXT,
            )
        ).view == manifest
        # Compaction and restart did not discharge or replace the live pin.
        with pytest.raises(ValueError, match="retention pin"):
            await session_store.validate_context_view_source_closure(session.id)
        released = await readback_application.transition_context_view_ownership(
            ContextViewOwnershipRequest(
                selection_key=selection.selection_key,
                view_id=manifest.view_id,
                pin_commitment=selection.pin_commitment,
                expected_state=adopted.state,
                expected_revision=adopted.ownership_revision,
                operation="release",
                current_owner=participant.owner,
                operation_key=f"witness-release-{backend}",
            ),
            participant=reader,
            context=CONTEXT,
        )
        assert released.state == "released"
        await session_store.validate_context_view_source_closure(session.id)
        if backend != "memory":
            # Every duplicated field is checked independently after reconstruction.
            lifecycle = await session_store.read_context_view_lifecycle_events(
                manifest.view_id, owner_participant=reader
            )
            original = lifecycle[0].model_dump(mode="json")
            corruptions = [
                {**original, field: replacement}
                for field, replacement in (
                    ("event_id", "different-event"),
                    ("operation_key", "different-operation"),
                    ("selection_key", "different-selection"),
                    ("view_id", "different-view"),
                    ("state", "expired"),
                    ("pin_commitment", "different-pin"),
                    ("ownership_revision", original["ownership_revision"] + 1),
                )
            ] + [
                {**original, "owner": {**original["owner"], field: "different-owner"}}
                for field in ("application_scope", "owner_id", "incarnation")
            ]
            for corrupted in corruptions:
                if backend == "sqlite":
                    session_store._connection.execute(
                        "UPDATE cayu_context_view_lifecycle_events SET event_json = ? WHERE event_id = ?",
                        (json.dumps(corrupted), original["event_id"]),
                    )
                    session_store._connection.commit()
                else:
                    from psycopg.types.json import Jsonb

                    async with session_store._connection() as conn, conn.cursor() as cur:
                        await cur.execute(
                            "UPDATE cayu_context_view_lifecycle_events SET event_json = %s WHERE event_id = %s",
                            (Jsonb(corrupted), original["event_id"]),
                        )
                        await conn.commit()
                with pytest.raises(ValueError, match="lifecycle indexes conflict"):
                    await readback_application.read_context_view(
                        manifest.view_id,
                        source_session_id=session.id,
                        participant=reader,
                        context=CONTEXT,
                    )
        # Pause after the public binding check, before native pin admission.
        native_select = session_store.select_context_view
        binding_checked = asyncio.Event()
        allow_admission = asyncio.Event()

        async def delayed_select(intent):
            binding_checked.set()
            await allow_admission.wait()
            return await native_select(intent)

        session_store.select_context_view = delayed_select
        late_selection = asyncio.create_task(
            readback_application.select_context_view(
                latest_request.model_copy(update={"selection_key": f"deleted-race-{backend}"}),
                participant=participant,
                context=CONTEXT,
            )
        )
        try:
            await asyncio.wait_for(binding_checked.wait(), 10)
            await session_store.delete_session(session.id)
            allow_admission.set()
            with pytest.raises(LookupError, match="source session incarnation"):
                await asyncio.wait_for(late_selection, 10)
            assert await session_store.load(session.id) is None
        finally:
            allow_admission.set()
            await asyncio.gather(late_selection, return_exceptions=True)
            session_store.select_context_view = native_select
    finally:
        close_session = getattr(session_store, "close", None)
        if close_session is not None:
            await close_session()
        close_collaboration = getattr(collaboration_store, "close", None)
        if close_collaboration is not None:
            await close_collaboration()


async def _test_public_participant_session_execution_is_authenticated_and_replayable() -> None:
    from tests.core.test_participant_identity import CONTEXT, app, create, registration

    from cayu.agents import AgentSpec
    from cayu.collaboration.lifecycle import ParticipantLifecycleChange
    from cayu.collaboration.memory import InMemoryCollaborationStore
    from cayu.evals.testing import ScriptedModelProvider

    application = app(InMemoryCollaborationStore(), registration())
    application.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("hello"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("continued"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    application.register_agent(AgentSpec(name="reviewer", model="model"))
    initialized = await application.initialize_collaboration()
    _, participant_receipt = await create(application, initialized, key="execution")
    participant = participant_receipt.participants[0].reference
    _, sibling_receipt = await create(application, initialized, key="execution-sibling")
    sibling = sibling_receipt.participants[0].reference
    creation = ParticipantSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "start")]),
        creation_key="execution-create",
    )
    session, _ = await application.create_participant_session(
        creation, participant=participant, context=CONTEXT
    )
    blocked_creation = ParticipantSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[]),
        creation_key="execution-disabled-create",
    )
    blocked_session, _ = await application.create_participant_session(
        blocked_creation, participant=participant, context=CONTEXT
    )
    await application.change_participant_lifecycle(
        ParticipantLifecycleChange(
            operation=initialized.operation("execution-disable"),
            participant=participant,
            expected_lifecycle_revision=1,
            state="disabled",
        ),
        context=CONTEXT,
    )
    blocked_execution = ParticipantSessionExecutionRequest(
        request=blocked_creation.request.model_copy(update={"session_id": blocked_session.id}),
        session_instance_id=blocked_session.instance_id,
        execution_key="execution-disabled",
    )
    with pytest.raises(PermissionError, match="active participants"):
        [
            event
            async for event in application.execute_participant_session(
                blocked_execution, participant=participant, context=CONTEXT
            )
        ]
    await application.change_participant_lifecycle(
        ParticipantLifecycleChange(
            operation=initialized.operation("execution-reenable"),
            participant=participant,
            expected_lifecycle_revision=2,
            state="active",
        ),
        context=CONTEXT,
    )
    execution = ParticipantSessionExecutionRequest(
        request=creation.request.model_copy(update={"session_id": session.id}),
        session_instance_id=session.instance_id,
        execution_key="execution-1",
    )
    events = [
        event
        async for event in application.execute_participant_session(
            execution, participant=participant, context=CONTEXT
        )
    ]
    assert any(event.type.value == "session.completed" for event in events)
    checkpoint = await application.session_store.load_checkpoint(session.id)
    from cayu.runtime._model_completion_publication import model_step_publication_from_checkpoint

    pointer = model_step_publication_from_checkpoint(checkpoint)
    assert pointer is not None
    completion_event = next(
        event
        for event in await application.session_store.load_events(session.id)
        if event.id == pointer.completion_event_id
    )
    assert completion_event.interaction_id is not None
    manifest = await application.publish_completed_context_view(
        ContextViewPublicationRequest(
            source_session_id=session.id,
            source_session_instance_id=session.instance_id,
            view_id="execution-view",
            interaction_id=completion_event.interaction_id,
            boundary_id=pointer.logical_step_id,
            projection_schema="whole-turn.v1",
            publication_key="execution-publication",
        ),
        participant=participant,
        context=CONTEXT,
    )
    selection = await application.select_context_view(
        ContextViewSelectionRequest(
            source_owner=participant.owner,
            source_session_id=session.id,
            source_session_instance_id=session.instance_id,
            selector="latest",
            projection_schema=manifest.projection_schema,
            extension_set_commitment=manifest.extension_set_commitment,
            limits=ContextViewLimits(),
            selection_key="execution-selection",
        ),
        participant=participant,
        context=CONTEXT,
    )
    assert selection.view == manifest
    readback = await application.read_context_view(
        manifest.view_id,
        source_session_id=session.id,
        participant=participant,
        context=CONTEXT,
    )
    assert readback.view == manifest
    with pytest.raises(LookupError, match="incarnation"):
        wrong = ParticipantSessionExecutionRequest(
            request=execution.request,
            session_instance_id="wrong-instance",
            execution_key="execution-2",
        )
        [
            event
            async for event in application.execute_participant_session(
                wrong, participant=participant, context=CONTEXT
            )
        ]
    with pytest.raises(PermissionError, match="bound to this participant"):
        await anext(
            application.execute_participant_session(execution, participant=sibling, context=CONTEXT)
        )
    replacement_creation = ParticipantSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[]),
        creation_key="execution-replacement-create",
    )
    replacement_session, _ = await application.create_participant_session(
        replacement_creation, participant=participant, context=CONTEXT
    )
    await application.change_participant_lifecycle(
        ParticipantLifecycleChange(
            operation=initialized.operation("execution-retire"),
            participant=participant,
            expected_lifecycle_revision=(
                await application.inspect_participant(participant, context=CONTEXT)
            ).participant.lifecycle_revision,
            state="retired",
        ),
        context=CONTEXT,
    )
    _, replacement_receipt = await create(
        application, initialized, key="execution-replacement-participant"
    )
    replacement = replacement_receipt.participants[0].reference
    replacement_execution = ParticipantSessionExecutionRequest(
        request=replacement_creation.request.model_copy(
            update={"session_id": replacement_session.id}
        ),
        session_instance_id=replacement_session.instance_id,
        execution_key="execution-replacement-run",
    )
    with pytest.raises(PermissionError, match="bound to this participant"):
        await anext(
            application.execute_participant_session(
                replacement_execution, participant=replacement, context=CONTEXT
            )
        )


def test_public_participant_session_execution_reconciles_lost_admission_ack() -> None:
    asyncio.run(_test_public_participant_session_execution_reconciles_lost_admission_ack())


async def _test_public_participant_session_execution_reconciles_lost_admission_ack() -> None:
    from tests.core.test_participant_identity import CONTEXT, app, create, registration

    from cayu.agents import AgentSpec
    from cayu.collaboration.lifecycle import ParticipantLifecycleChange
    from cayu.collaboration.memory import InMemoryCollaborationStore
    from cayu.evals.testing import ScriptedModelProvider

    session_store = InMemorySessionStore()
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    application = app(InMemoryCollaborationStore(), registration(), session_store=session_store)
    runtime_store = application._runtime_session_store
    original_apply = runtime_store.apply_invocation_lifecycle_command
    raise_admission_ack = True

    async def apply_with_lost_ack(command):
        nonlocal raise_admission_ack
        result = await original_apply(command)
        if raise_admission_ack:
            raise_admission_ack = False
            raise ConnectionError("participant admission acknowledgement lost")
        return result

    runtime_store.apply_invocation_lifecycle_command = apply_with_lost_ack
    application.register_provider(provider, default=True)
    application.register_agent(AgentSpec(name="reviewer", model="model"))
    initialized = await application.initialize_collaboration()
    _, participant_receipt = await create(application, initialized, key="execution-lost-ack")
    participant = participant_receipt.participants[0].reference
    creation = ParticipantSessionCreationRequest(
        request=RunRequest(
            agent_name="reviewer",
            messages=[Message.text("user", "start")],
            session_id="requested-participant-session",
        ),
        creation_key="execution-lost-ack-create",
    )
    session, _ = await application.create_participant_session(
        creation, participant=participant, context=CONTEXT
    )
    execution = ParticipantSessionExecutionRequest(
        request=creation.request,
        session_instance_id=session.instance_id,
        execution_key="execution-lost-ack-run",
    )
    with pytest.raises(ConnectionError, match="acknowledgement lost"):
        [
            event
            async for event in application.execute_participant_session(
                execution, participant=participant, context=CONTEXT
            )
        ]
    assert (
        len(
            (
                await application.list_participant_obligations(
                    participant, context=CONTEXT, pending_only=True
                )
            ).obligations
        )
        == 1
    )
    await application.change_participant_lifecycle(
        ParticipantLifecycleChange(
            operation=initialized.operation("execution-lost-ack-disable"),
            participant=participant,
            expected_lifecycle_revision=1,
            state="disabled",
        ),
        context=CONTEXT,
    )
    replayed = [
        event
        async for event in application.execute_participant_session(
            execution, participant=participant, context=CONTEXT
        )
    ]
    assert any(event.type.value == "session.completed" for event in replayed)
    assert len(provider.requests) == 1
    restored = await session_store.load(session.id)
    assert restored is not None and restored.status.value == "completed"


def test_public_participant_session_concurrent_activation_is_fenced() -> None:
    asyncio.run(_test_public_participant_session_concurrent_activation_is_fenced())


async def _test_public_participant_session_concurrent_activation_is_fenced() -> None:
    from tests.core.test_participant_identity import CONTEXT, app, create, registration

    from cayu.agents import AgentSpec
    from cayu.collaboration.memory import InMemoryCollaborationStore
    from cayu.evals.testing import ScriptedModelProvider

    session_store = InMemorySessionStore()
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    application = app(InMemoryCollaborationStore(), registration(), session_store=session_store)
    application.register_provider(provider, default=True)
    application.register_agent(AgentSpec(name="reviewer", model="model"))
    initialized = await application.initialize_collaboration()
    _, participant_receipt = await create(application, initialized, key="execution-race")
    participant = participant_receipt.participants[0].reference
    creation = ParticipantSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "start")]),
        creation_key="execution-race-create",
    )
    session, _ = await application.create_participant_session(
        creation, participant=participant, context=CONTEXT
    )
    execution = ParticipantSessionExecutionRequest(
        request=creation.request.model_copy(update={"session_id": session.id}),
        session_instance_id=session.instance_id,
        execution_key="execution-race-run",
    )

    async def consume() -> list[object]:
        return [
            event
            async for event in application.execute_participant_session(
                execution, participant=participant, context=CONTEXT
            )
        ]

    first, second = await asyncio.gather(consume(), consume(), return_exceptions=True)
    outcomes = (first, second)
    assert sum(isinstance(value, list) for value in outcomes) == 1
    error = next(value for value in outcomes if isinstance(value, BaseException))
    assert isinstance(error, (SessionStatusConflict, SessionRunFenced))
    current = await session_store.load(session.id)
    assert current is not None
    if current.status.value != "completed":
        await consume()
    assert len(provider.requests) == 1
    completed = await session_store.load(session.id)
    assert completed is not None and completed.status.value == "completed"


def test_participant_execution_does_not_block_unrelated_sessions() -> None:
    asyncio.run(_test_participant_execution_does_not_block_unrelated_sessions())


async def _test_participant_execution_does_not_block_unrelated_sessions() -> None:
    from tests.core.test_participant_identity import CONTEXT, app, create, registration

    from cayu.agents import AgentSpec
    from cayu.collaboration.memory import InMemoryCollaborationStore
    from cayu.evals.testing import ScriptedModelProvider

    provider = ScriptedModelProvider([])
    application = app(InMemoryCollaborationStore(), registration())
    application.register_provider(provider, default=True)
    application.register_agent(AgentSpec(name="reviewer", model="model"))
    initialized = await application.initialize_collaboration()
    _, first_receipt = await create(application, initialized, key="execution-isolation-1")
    _, second_receipt = await create(application, initialized, key="execution-isolation-2")
    first_participant = first_receipt.participants[0].reference
    second_participant = second_receipt.participants[0].reference

    async def create_execution(participant, creation_key, execution_key):
        creation = ParticipantSessionCreationRequest(
            request=RunRequest(agent_name="reviewer", messages=[]),
            creation_key=creation_key,
        )
        session, _ = await application.create_participant_session(
            creation, participant=participant, context=CONTEXT
        )
        return ParticipantSessionExecutionRequest(
            request=creation.request.model_copy(update={"session_id": session.id}),
            session_instance_id=session.instance_id,
            execution_key=execution_key,
        )

    first_execution = await create_execution(
        first_participant, "execution-isolation-create-1", "execution-isolation-run-1"
    )
    second_execution = await create_execution(
        second_participant, "execution-isolation-create-2", "execution-isolation-run-2"
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def controlled_run(request, **_kwargs):
        if request.session_id == first_execution.request.session_id:
            started.set()
            await release.wait()
        if False:
            yield None

    application._run_private = controlled_run

    async def consume(execution, participant):
        return [
            event
            async for event in application.execute_participant_session(
                execution, participant=participant, context=CONTEXT
            )
        ]

    first_task = asyncio.create_task(consume(first_execution, first_participant))
    await asyncio.sleep(0)
    if first_task.done():
        first_task.result()
    await asyncio.wait_for(started.wait(), 10)
    second_task = asyncio.create_task(consume(second_execution, second_participant))
    try:
        second_events = await asyncio.wait_for(second_task, 10)
        assert second_events == []
    finally:
        release.set()
        await asyncio.gather(first_task, return_exceptions=True)


def test_public_participant_session_execution_reconstructs_after_sqlite_restart(tmp_path) -> None:
    asyncio.run(
        _test_public_participant_session_execution_reconstructs_after_sqlite_restart(tmp_path)
    )


async def _test_public_participant_session_execution_reconstructs_after_sqlite_restart(
    tmp_path,
) -> None:
    from tests.core.test_participant_identity import CONTEXT, app, create, registration

    from cayu.agents import AgentSpec
    from cayu.evals.testing import ScriptedModelProvider
    from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore

    session_path = tmp_path / "participant-execution.sqlite"
    collaboration_path = tmp_path / "participant-collaboration.sqlite"
    session_store = SQLiteSessionStore(session_path)
    collaboration_store = SQLiteCollaborationStore(collaboration_path)
    collaboration_registration = registration()

    def build(store, participants, provider):
        application = app(participants, collaboration_registration, session_store=store)
        application.register_provider(provider, default=True)
        application.register_agent(AgentSpec(name="reviewer", model="model"))
        return application

    first_provider = ScriptedModelProvider([])
    first = build(session_store, collaboration_store, first_provider)
    initialized = await first.initialize_collaboration()
    _, participant_receipt = await create(first, initialized, key="execution-restart")
    participant = participant_receipt.participants[0].reference
    creation = ParticipantSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "start")]),
        creation_key="execution-restart-create",
    )
    session, _ = await first.create_participant_session(
        creation, participant=participant, context=CONTEXT
    )
    execution = ParticipantSessionExecutionRequest(
        request=creation.request.model_copy(update={"session_id": session.id}),
        session_instance_id=session.instance_id,
        execution_key="execution-restart-run",
    )
    await session_store.close()
    await collaboration_store.close()

    restarted_store = SQLiteSessionStore(session_path)
    restarted_collaboration = SQLiteCollaborationStore(collaboration_path)
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    restarted = build(restarted_store, restarted_collaboration, provider)
    try:
        await restarted.initialize_collaboration()
        events = [
            event
            async for event in restarted.execute_participant_session(
                execution, participant=participant, context=CONTEXT
            )
        ]
        assert any(event.type.value == "session.completed" for event in events)
        assert len(provider.requests) == 1
        restored = await restarted_store.load(session.id)
        assert restored is not None and restored.status.value == "completed"
    finally:
        await restarted_store.close()
        await restarted_collaboration.close()


@pytest.mark.parametrize("after_commit", [False, True])
def test_public_creation_cancellation_replays_after_sqlite_reopen(tmp_path, after_commit):
    asyncio.run(_public_creation_cancellation_replays_after_sqlite_reopen(tmp_path, after_commit))


async def _public_creation_cancellation_replays_after_sqlite_reopen(tmp_path, after_commit):
    from tests.core.test_participant_identity import CONTEXT, app, create, registration

    from cayu.agents import AgentSpec
    from cayu.evals.testing import ScriptedModelProvider
    from cayu.sessions.base import EventQuery
    from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore

    entered = asyncio.Event()
    release = asyncio.Event()
    committed = []

    class PausingStore(SQLiteSessionStore):
        async def create_participant_owned_session(self, *args, **kwargs):
            if not after_commit:
                entered.set()
                await release.wait()
            result = await super().create_participant_owned_session(*args, **kwargs)
            committed.append(result)
            if after_commit:
                entered.set()
                await release.wait()
            return result

    reg = registration()
    session_path = tmp_path / "sessions.sqlite"
    participant_path = tmp_path / "participants.sqlite"
    sessions = PausingStore(session_path)
    participants = SQLiteCollaborationStore(participant_path)

    def application_for(session_store, participant_store):
        application = app(participant_store, reg, session_store=session_store)
        application.register_provider(ScriptedModelProvider([]), default=True)
        application.register_agent(AgentSpec(name="reviewer", model="model"))
        return application

    application = application_for(sessions, participants)
    caller = None
    creation = ParticipantSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[]),
        creation_key="cancelled-creation",
    )
    try:
        initialized = await application.initialize_collaboration()
        _, receipt = await create(application, initialized)
        participant = receipt.participants[0].reference
        caller = asyncio.create_task(
            application.create_participant_session(
                creation, participant=participant, context=CONTEXT
            )
        )
        await asyncio.wait_for(entered.wait(), 5)
        assert bool(committed) is after_commit
        caller.cancel()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert caller.cancelled() and caller.cancelling() == 2
        stored = await sessions.lookup_participant_session_creation(creation)
        assert (stored is not None) is after_commit
        if after_commit:
            assert stored == committed[0]
    finally:
        release.set()
        if caller is not None and not caller.done():
            caller.cancel()
            await asyncio.gather(caller, return_exceptions=True)
        await sessions.close()
        await participants.close()

    sessions = SQLiteSessionStore(session_path)
    participants = SQLiteCollaborationStore(participant_path)
    try:
        restored = application_for(sessions, participants)
        await restored.initialize_collaboration()
        result = await restored.create_participant_session(
            creation, participant=participant, context=CONTEXT
        )
        if after_commit:
            assert result == committed[0]
        assert (
            await restored.create_participant_session(
                creation, participant=participant, context=CONTEXT
            )
            == result
        )
        session, _ = result
        assert session.status.value == "pending"
        assert not await sessions.query_events(EventQuery(session_id=session.id, limit=16))
        assert len((await sessions.list_sessions()).sessions) == 1
    finally:
        await sessions.close()
        await participants.close()


def test_public_context_view_publication_selection_and_readback() -> None:
    asyncio.run(_test_public_context_view_publication_selection_and_readback())


async def _test_public_context_view_publication_selection_and_readback() -> None:
    from tests.core.test_participant_identity import CONTEXT, Policy, app, registration

    from cayu.agents import AgentSpec
    from cayu.collaboration.memory import InMemoryCollaborationStore
    from cayu.evals.testing import ScriptedModelProvider
    from cayu.runtime._model_completion_publication import (
        LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
        ModelStepPublicationCheckpoint,
    )

    class EnabledStore(InMemorySessionStore):
        context_view_version = 1

        fail_publication_once = False

        async def publish_context_view(self, manifest, *, publication_key):
            result = await super().publish_context_view(manifest, publication_key=publication_key)
            if self.fail_publication_once:
                self.fail_publication_once = False
                raise RuntimeError("publication acknowledgement lost")
            return result

    collaboration_store = InMemoryCollaborationStore()
    session_store = EnabledStore()
    extension_inputs: list[object] = []
    extension = ContextViewExtensionRegistration(
        extension="safe-history",
        schema_version=1,
        project=lambda value: extension_inputs.append(value) or None,
    )
    policy = Policy()
    application = app(
        collaboration_store,
        registration(policy=policy),
        session_store=session_store,
        context_view_extensions=(extension,),
    )
    application.register_provider(ScriptedModelProvider([]), default=True)
    application.register_agent(AgentSpec(name="reviewer", model="model"))
    await application.initialize_collaboration()
    initialized = application._participant_coordinator._initialized
    assert initialized is not None
    from cayu.collaboration.participants import ParticipantCreate

    participant = (
        (
            await application.create_participant(
                ParticipantCreate(
                    operation=initialized.operation("context-view-publication-participant"),
                    configuration=registration().configurations[0],
                ),
                context=CONTEXT,
            )
        )
        .participants[0]
        .reference
    )
    creation = ParticipantSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[]),
        creation_key="context-view-publication-create",
    )
    session, _ = await application.create_participant_session(
        creation,
        participant=participant,
        context=CONTEXT,
    )
    await session_store.append_event(
        session.id,
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id=session.id,
            interaction_id="interaction",
            id="completion",
            payload={
                "execution_profile_fingerprint": session.metadata["cayu:execution_profile"][
                    "expected"
                ]["fingerprint"],
            },
        ),
    )
    pointer = ModelStepPublicationCheckpoint(
        logical_step_id="step",
        stage_id="stage",
        source_transcript_cursor=0,
        transcript_end_cursor=1,
        completion_event_id="completion",
        classification={"type": "final"},
        assistant_message_published=True,
    )

    def checkpoint_transform(_session, checkpoint):
        return {
            **(checkpoint or {}),
            LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY: pointer.model_dump(mode="json"),
        }

    await session_store.append_transcript_messages_and_transform_checkpoint(
        session.id,
        [Message.text("assistant", "done")],
        checkpoint_transform,
    )
    publication_request = ContextViewPublicationRequest(
        source_session_id=session.id,
        source_session_instance_id=session.instance_id,
        view_id="public-view",
        interaction_id="interaction",
        boundary_id="step",
        projection_schema="whole-turn.v1",
        publication_key="public-publication",
    )
    with pytest.raises(ValueError, match="identity conflicts"):
        await application.publish_completed_context_view(
            publication_request.model_copy(
                update={"interaction_id": "forged-interaction", "publication_key": "forged"}
            ),
            participant=participant,
            context=CONTEXT,
        )
    session_store.fail_publication_once = True
    with pytest.raises(RuntimeError, match="acknowledgement lost"):
        await application.publish_completed_context_view(
            publication_request,
            participant=participant,
            context=CONTEXT,
        )
    manifest = await application.publish_completed_context_view(
        publication_request, participant=participant, context=CONTEXT
    )
    assert extension_inputs
    assert isinstance(extension_inputs[-1], ContextViewProjectionSource)
    assert extension_inputs[-1].boundary_id == "step"
    assert "invocation" not in ContextViewProjectionSource.model_fields
    await session_store.transform_checkpoint(
        session.id,
        lambda _session, _checkpoint: {},
    )
    assert (
        await application.publish_completed_context_view(
            publication_request,
            participant=participant,
            context=CONTEXT,
        )
        == manifest
    )
    application._context_view_extensions = (
        ContextViewExtensionRegistration(
            extension="safe-history",
            schema_version=2,
            project=lambda _: None,
        ),
    )
    with pytest.raises(ValueError, match="conflicts with its request"):
        await application.publish_completed_context_view(
            publication_request,
            participant=participant,
            context=CONTEXT,
        )
    application._context_view_extensions = (extension,)
    selection = await application.select_context_view(
        ContextViewSelectionRequest(
            source_owner=participant.owner,
            source_session_id=session.id,
            source_session_instance_id=session.instance_id,
            selector="latest",
            projection_schema=manifest.projection_schema,
            extension_set_commitment=manifest.extension_set_commitment,
            limits=ContextViewLimits(),
            selection_key="public-selection",
        ),
        participant=participant,
        context=CONTEXT,
    )
    readback = await application.read_context_view(
        manifest.view_id,
        source_session_id=session.id,
        participant=participant,
        context=CONTEXT,
    )
    assert selection.view == manifest
    assert readback.view == manifest
    assert readback.historical_only is True
    assert manifest.compaction_json is not None

    # Both participants use the same collaboration-store OwnerRef. A grant to
    # the sibling does not authorize disclosure of the source participant's view.
    sibling = (
        (
            await application.create_participant(
                ParticipantCreate(
                    operation=initialized.operation("sibling"),
                    configuration=registration().configurations[0],
                ),
                context=CONTEXT,
            )
        )
        .participants[0]
        .reference
    )
    assert sibling.owner == participant.owner
    assert sibling != participant
    policy.allowed = (sibling,)
    with pytest.raises(PermissionError, match="does not own"):
        await application.read_context_view(
            manifest.view_id,
            source_session_id=session.id,
            participant=sibling,
            context=CONTEXT,
        )
    selection_request = ContextViewSelectionRequest(
        source_owner=sibling.owner,
        source_session_id=session.id,
        source_session_instance_id=session.instance_id,
        selector="latest",
        projection_schema=manifest.projection_schema,
        extension_set_commitment=manifest.extension_set_commitment,
        limits=ContextViewLimits(),
        selection_key="sibling-selection",
    )
    with pytest.raises(PermissionError, match="not bound"):
        await application.select_context_view(
            selection_request, participant=sibling, context=CONTEXT
        )
    policy.allowed = (participant, sibling)
    adopted = await application.transition_context_view_ownership(
        ContextViewOwnershipRequest(
            selection_key=selection.selection_key,
            view_id=manifest.view_id,
            pin_commitment=selection.pin_commitment,
            expected_state="selected",
            expected_revision=selection.ownership_revision,
            operation="adopt",
            current_owner=participant.owner,
            destination_owner=participant.owner,
            operation_key="public-adopt",
        ),
        participant=participant,
        destination_participant=participant,
        context=CONTEXT,
    )
    transferred = await application.transition_context_view_ownership(
        ContextViewOwnershipRequest(
            selection_key=selection.selection_key,
            view_id=manifest.view_id,
            pin_commitment=selection.pin_commitment,
            expected_state="adopted",
            expected_revision=adopted.ownership_revision,
            operation="transfer",
            current_owner=participant.owner,
            destination_owner=sibling.owner,
            operation_key="public-transfer",
        ),
        participant=participant,
        destination_participant=sibling,
        context=CONTEXT,
    )
    assert transferred.state == "transferred"
    transferred_readback = await application.read_context_view(
        manifest.view_id,
        source_session_id=session.id,
        participant=sibling,
        context=CONTEXT,
    )
    assert transferred_readback.view == manifest
    transferred_events = await application.read_context_view_lifecycle_events(
        manifest.view_id,
        source_session_id=session.id,
        participant=sibling,
        context=CONTEXT,
    )
    assert [event.state for event in transferred_events] == ["adopted", "transferred"]
    source_events = await application.read_context_view_lifecycle_events(
        manifest.view_id,
        source_session_id=session.id,
        participant=participant,
        context=CONTEXT,
    )
    assert source_events == transferred_events
    # A rejected attempt must not consume the key or leave an altered receipt.
    policy.allowed = (participant,)
    assert (
        await application.select_context_view(
            selection_request, participant=participant, context=CONTEXT
        )
    ).view == manifest
    from cayu.collaboration.lifecycle import ParticipantLifecycleChange

    await application.change_participant_lifecycle(
        ParticipantLifecycleChange(
            operation=initialized.operation("disable-source"),
            participant=participant,
            expected_lifecycle_revision=1,
            state="disabled",
        ),
        context=CONTEXT,
    )
    historical = await application.read_context_view(
        manifest.view_id,
        source_session_id=session.id,
        participant=participant,
        context=CONTEXT,
    )
    assert historical.view == manifest
    # Disablement revokes new disclosure, not the already committed binding.
    historical_binding = await session_store.load_participant_session_binding(session.id)
    assert historical_binding is not None and historical_binding.participant == participant


async def _test_public_participant_session_creation_is_inert_and_replayable() -> None:
    from tests.core.test_participant_identity import CONTEXT, app, registration

    from cayu.agents import AgentSpec
    from cayu.collaboration.memory import InMemoryCollaborationStore
    from cayu.evals.testing import ScriptedModelProvider
    from cayu.sessions.base import EventQuery

    class EnabledStore(InMemorySessionStore):
        context_view_version = 1

    collaboration_store = InMemoryCollaborationStore()
    session_store = EnabledStore()
    application = app(
        collaboration_store,
        registration(),
        session_store=session_store,
    )
    application.register_provider(ScriptedModelProvider([]), default=True)
    application.register_agent(AgentSpec(name="reviewer", model="model"))
    await application.initialize_collaboration()
    initialized = application._participant_coordinator._initialized
    assert initialized is not None
    from cayu.collaboration.participants import ParticipantCreate

    participant = (
        (
            await application.create_participant(
                ParticipantCreate(
                    operation=initialized.operation("context-view-participant"),
                    configuration=registration().configurations[0],
                ),
                context=CONTEXT,
            )
        )
        .participants[0]
        .reference
    )
    creation = ParticipantSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[]),
        creation_key="context-view-create",
    )
    first_session, first_receipt = await application.create_participant_session(
        creation,
        participant=participant,
        context=CONTEXT,
    )
    replay_session, replay_receipt = await application.create_participant_session(
        creation,
        participant=participant,
        context=CONTEXT,
    )
    assert first_session == replay_session
    assert first_receipt == replay_receipt
    assert first_session.status.value == "pending"
    assert await session_store.load_participant_session_binding(first_session.id) is not None
    assert not await session_store.query_events(EventQuery(session_id=first_session.id, limit=16))
    conflicting_request = ParticipantSessionCreationRequest(
        request=RunRequest(
            agent_name="reviewer",
            messages=[],
            session_id="a-different-requested-session-id",
        ),
        creation_key="context-view-create",
    )
    with pytest.raises(ValueError, match="conflicts with the request"):
        await application.create_participant_session(
            conflicting_request,
            participant=participant,
            context=CONTEXT,
        )


def test_memory_publication_and_selection_are_idempotent() -> None:
    asyncio.run(_test_memory_publication_and_selection_are_idempotent())


def test_postgres_context_view_store_reconstructs_ownership(postgres_dsn: str) -> None:
    asyncio.run(_test_postgres_context_view_store_reconstructs_ownership(postgres_dsn))


def test_postgres_context_view_publication_and_selection_do_not_deadlock(
    postgres_dsn: str,
) -> None:
    asyncio.run(_test_postgres_context_view_publication_and_selection_do_not_deadlock(postgres_dsn))


async def _test_postgres_context_view_publication_and_selection_do_not_deadlock(
    postgres_dsn: str,
) -> None:
    from contextlib import asynccontextmanager

    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    first_lock_acquired = asyncio.Event()
    selection_started = asyncio.Event()
    race_enabled = False

    class BarrierCursor:
        def __init__(self, cursor, role: str):
            self._context = cursor
            self._cursor = None
            self._role = role
            self._first_advisory_lock = True

        async def __aenter__(self):
            self._cursor = await self._context.__aenter__()
            return self

        async def __aexit__(self, *args):
            return await self._context.__aexit__(*args)

        def __getattr__(self, name):
            return getattr(self._cursor, name)

        async def execute(self, query, params=None):
            assert self._cursor is not None
            is_advisory_lock = (
                isinstance(query, str)
                and "pg_advisory_xact_lock" in query
                and params
                and isinstance(params[0], str)
                and params[0].startswith("context-view-")
            )
            if race_enabled and is_advisory_lock and self._first_advisory_lock:
                self._first_advisory_lock = False
                if self._role == "selection":
                    selection_started.set()
                    await first_lock_acquired.wait()
                result = await self._cursor.execute(query, params)
                if self._role == "publication":
                    first_lock_acquired.set()
                    await selection_started.wait()
                return result
            if params is None:
                return await self._cursor.execute(query)
            return await self._cursor.execute(query, params)

    class BarrierConnectionStore(PostgresSessionStore):
        def __init__(self, *args, lock_role: str, **kwargs):
            self._lock_role = lock_role
            super().__init__(*args, **kwargs)

        @asynccontextmanager
        async def _connection(self):
            async with super()._connection() as connection:

                class ConnectionProxy:
                    def cursor(proxy_self, *args, **kwargs):
                        return BarrierCursor(connection.cursor(*args, **kwargs), self._lock_role)

                    def __getattr__(proxy_self, name):
                        return getattr(connection, name)

                yield ConnectionProxy()

    first = BarrierConnectionStore(
        postgres_dsn,
        min_size=1,
        max_size=2,
        schema_mode=SchemaMode.CREATE,
        lock_role="publication",
    )
    second = BarrierConnectionStore(
        postgres_dsn,
        min_size=1,
        max_size=2,
        schema_mode=SchemaMode.VALIDATE,
        lock_role="selection",
    )
    try:
        manifest = await _manifest_for_store(first, session_id="postgres-lock-order-session")
        manifest = _replace_manifest(manifest, view_id="postgres-lock-order-base-view")
        await second.ensure_schema()
        await first.publish_context_view(manifest, publication_key="postgres-lock-order-base")
        next_manifest = _replace_manifest(manifest, view_id="postgres-lock-order-next")
        selection = ContextViewSelectionRequest(
            source_owner=manifest.source_owner,
            source_session_id=manifest.source_session_id,
            source_session_instance_id=manifest.source_session_instance_id,
            selector="exact",
            exact_view_id=manifest.view_id,
            projection_schema=manifest.projection_schema,
            extension_set_commitment=manifest.extension_set_commitment,
            limits=ContextViewLimits(),
            selection_key="postgres-lock-order-selection",
        )
        barrier = asyncio.Barrier(2)

        async def publish() -> ContextViewManifest:
            await barrier.wait()
            return await first.publish_context_view(
                next_manifest, publication_key="postgres-lock-order-next"
            )

        async def select():
            await barrier.wait()
            return await second.select_context_view(selection)

        race_enabled = True
        published, selected = await asyncio.wait_for(
            asyncio.gather(publish(), select()), timeout=15
        )
        assert published == next_manifest
        assert selected.view.view_id == manifest.view_id
    finally:
        await first.close()
        await second.close()


async def _test_postgres_context_view_store_reconstructs_ownership(postgres_dsn: str) -> None:
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    manifest = _manifest()
    store = PostgresSessionStore(
        postgres_dsn, min_size=1, max_size=2, schema_mode=SchemaMode.CREATE
    )
    try:
        manifest = await _manifest_for_store(store)
        await store.publish_context_view(manifest, publication_key="postgres-publication")
        selected = await store.select_context_view(
            ContextViewSelectionRequest(
                source_owner=manifest.source_owner,
                source_session_id=manifest.source_session_id,
                source_session_instance_id=manifest.source_session_instance_id,
                selector="latest",
                projection_schema=manifest.projection_schema,
                extension_set_commitment=manifest.extension_set_commitment,
                limits=ContextViewLimits(),
                selection_key="postgres-selection",
            )
        )
        concurrent_request = ContextViewSelectionRequest(
            source_owner=manifest.source_owner,
            source_session_id=manifest.source_session_id,
            source_session_instance_id=manifest.source_session_instance_id,
            selector="latest",
            projection_schema=manifest.projection_schema,
            extension_set_commitment=manifest.extension_set_commitment,
            limits=ContextViewLimits(),
            selection_key="postgres-concurrent-selection",
        )
        concurrent = await asyncio.gather(
            store.select_context_view(concurrent_request),
            store.select_context_view(concurrent_request),
        )
        assert concurrent[0] == concurrent[1]
        with pytest.raises(ValueError, match="retention pin"):
            await store.validate_context_view_source_closure(manifest.source_session_id)
        adopted = await store.transition_context_view_ownership(
            ContextViewOwnershipRequest(
                selection_key=selected.selection_key,
                view_id=manifest.view_id,
                pin_commitment=selected.pin_commitment,
                expected_state="selected",
                expected_revision=selected.ownership_revision,
                operation="adopt",
                current_owner=manifest.source_owner,
                destination_owner=manifest.source_owner,
                operation_key="postgres-adopt",
            )
        )
        assert adopted.state == "adopted"
        assert adopted.owner_participant == manifest.participant
        await store.close()
        store = PostgresSessionStore(
            postgres_dsn, min_size=1, max_size=2, schema_mode=SchemaMode.VALIDATE
        )
        restored = await store.read_context_view(
            manifest.view_id, source_session_id=manifest.source_session_id
        )
        assert restored.view == manifest
        events = await store.read_context_view_lifecycle_events(manifest.view_id)
        assert [event.state for event in events] == ["adopted"]
    finally:
        await store.close()


async def _test_memory_publication_and_selection_are_idempotent() -> None:
    manifest = _manifest()
    store = InMemorySessionStore()
    manifest = await _manifest_for_store(store)
    first = await store.publish_context_view(manifest, publication_key="publish-1")
    replay = await store.publish_context_view(manifest, publication_key="publish-1")
    assert first == replay
    owner = manifest.source_owner
    selected = await store.select_context_view(
        ContextViewSelectionRequest(
            source_owner=owner,
            source_session_id=manifest.source_session_id,
            source_session_instance_id=manifest.source_session_instance_id,
            selector="latest",
            projection_schema=manifest.projection_schema,
            extension_set_commitment=manifest.extension_set_commitment,
            limits=ContextViewLimits(),
            selection_key="selection-1",
        )
    )
    assert selected.view.view_id == manifest.view_id
    concurrent_request = ContextViewSelectionRequest(
        source_owner=owner,
        source_session_id=manifest.source_session_id,
        source_session_instance_id=manifest.source_session_instance_id,
        selector="latest",
        projection_schema=manifest.projection_schema,
        extension_set_commitment=manifest.extension_set_commitment,
        limits=ContextViewLimits(),
        selection_key="selection-concurrent",
    )
    concurrent = await asyncio.gather(
        store.select_context_view(concurrent_request),
        store.select_context_view(concurrent_request),
    )
    assert concurrent[0] == concurrent[1]
    assert (
        await store.select_context_view(
            ContextViewSelectionRequest(
                source_owner=owner,
                source_session_id=manifest.source_session_id,
                source_session_instance_id=manifest.source_session_instance_id,
                selector="latest",
                projection_schema=manifest.projection_schema,
                extension_set_commitment=manifest.extension_set_commitment,
                limits=ContextViewLimits(),
                selection_key="selection-1",
            )
        )
        == selected
    )
    adopted = await store.transition_context_view_ownership(
        ContextViewOwnershipRequest(
            selection_key=selected.selection_key,
            view_id=selected.view.view_id,
            pin_commitment=selected.pin_commitment,
            expected_state="selected",
            expected_revision=selected.ownership_revision,
            operation="adopt",
            current_owner=owner,
            destination_owner=owner,
            operation_key="adopt-1",
        )
    )
    assert adopted.state == "adopted"
    assert adopted.ownership_revision == 2
    assert (
        await store.transition_context_view_ownership(
            ContextViewOwnershipRequest(
                selection_key=selected.selection_key,
                view_id=selected.view.view_id,
                pin_commitment=selected.pin_commitment,
                expected_state="selected",
                expected_revision=selected.ownership_revision,
                operation="adopt",
                current_owner=owner,
                destination_owner=owner,
                operation_key="adopt-1",
            )
        )
        == adopted
    )
    released = await store.transition_context_view_ownership(
        ContextViewOwnershipRequest(
            selection_key=selected.selection_key,
            view_id=selected.view.view_id,
            pin_commitment=selected.pin_commitment,
            expected_state="adopted",
            expected_revision=adopted.ownership_revision,
            operation="release",
            current_owner=owner,
            operation_key="release-1",
        )
    )
    assert released.state == "released"
    events = await store.read_context_view_lifecycle_events(manifest.view_id)
    assert [event.state for event in events] == ["adopted", "released"]


def test_memory_participant_creation_is_atomic_and_replayable() -> None:
    asyncio.run(_test_memory_participant_creation_is_atomic_and_replayable())


def test_sqlite_participant_creation_is_atomic_and_replayable(tmp_path) -> None:
    asyncio.run(
        _test_memory_participant_creation_is_atomic_and_replayable(
            SQLiteSessionStore(tmp_path / "participant-sessions.sqlite")
        )
    )


def test_sqlite_context_view_publication_selection_and_readback(tmp_path) -> None:
    asyncio.run(_test_sqlite_context_view_publication_selection_and_readback(tmp_path))


def test_sqlite_context_view_publication_reconciles_commit_then_ack_loss(tmp_path) -> None:
    asyncio.run(_test_sqlite_context_view_publication_reconciles_commit_then_ack_loss(tmp_path))


async def _test_sqlite_context_view_publication_reconciles_commit_then_ack_loss(tmp_path) -> None:
    class CommitThenRaiseStore(SQLiteSessionStore):
        raised = False

        async def publish_context_view(self, manifest, *, publication_key):
            result = await super().publish_context_view(manifest, publication_key=publication_key)
            if not self.raised:
                self.raised = True
                raise ConnectionError("acknowledgement lost after publication commit")
            return result

    store = CommitThenRaiseStore(tmp_path / "publication-lost-ack.sqlite")
    try:
        manifest = await _manifest_for_store(store)
        with pytest.raises(ConnectionError, match="acknowledgement lost"):
            await store.publish_context_view(manifest, publication_key="lost-ack-publication")
        assert await store.lookup_context_view_publication("lost-ack-publication") == manifest
        assert (
            await store.publish_context_view(manifest, publication_key="lost-ack-publication")
            == manifest
        )
    finally:
        await store.close()


def test_sqlite_context_view_ownership_reconciles_commit_then_ack_loss(tmp_path) -> None:
    asyncio.run(_test_sqlite_context_view_ownership_reconciles_commit_then_ack_loss(tmp_path))


async def _test_sqlite_context_view_ownership_reconciles_commit_then_ack_loss(tmp_path) -> None:
    class CommitThenRaiseStore(SQLiteSessionStore):
        raised = False

        async def transition_context_view_ownership(self, request):
            result = await super().transition_context_view_ownership(request)
            if not self.raised:
                self.raised = True
                raise ConnectionError("acknowledgement lost after ownership commit")
            return result

    store = CommitThenRaiseStore(tmp_path / "ownership-lost-ack.sqlite")
    try:
        manifest = _manifest()
        manifest = await _manifest_for_store(store)
        await store.publish_context_view(manifest, publication_key="ownership-publication")
        selected = await store.select_context_view(
            ContextViewSelectionRequest(
                source_owner=manifest.source_owner,
                source_session_id=manifest.source_session_id,
                source_session_instance_id=manifest.source_session_instance_id,
                selector="latest",
                projection_schema=manifest.projection_schema,
                extension_set_commitment=manifest.extension_set_commitment,
                limits=ContextViewLimits(),
                selection_key="ownership-lost-ack-selection",
            )
        )
        request = ContextViewOwnershipRequest(
            selection_key=selected.selection_key,
            view_id=manifest.view_id,
            pin_commitment=selected.pin_commitment,
            expected_state="selected",
            expected_revision=selected.ownership_revision,
            operation="adopt",
            current_owner=manifest.source_owner,
            destination_owner=manifest.source_owner,
            operation_key="ownership-lost-ack-operation",
        )
        with pytest.raises(ConnectionError, match="acknowledgement lost"):
            await store.transition_context_view_ownership(request)
        replay = await store.transition_context_view_ownership(request)
        assert replay.state == "adopted"
        assert len(await store.read_context_view_lifecycle_events(manifest.view_id)) == 1
    finally:
        await store.close()


def test_sqlite_context_view_ownership_reconstructs_after_restart(tmp_path) -> None:
    asyncio.run(_test_sqlite_context_view_ownership_reconstructs_after_restart(tmp_path))


def test_sqlite_context_view_expiry_publishes_lifecycle_evidence(tmp_path) -> None:
    asyncio.run(_test_sqlite_context_view_expiry_publishes_lifecycle_evidence(tmp_path))


def test_memory_context_view_expiry_allows_direct_session_deletion() -> None:
    asyncio.run(_test_memory_context_view_expiry_allows_direct_session_deletion())


async def _test_memory_context_view_expiry_allows_direct_session_deletion() -> None:
    from datetime import UTC, datetime, timedelta

    now = datetime(2026, 1, 1, tzinfo=UTC)
    store = InMemorySessionStore(ownership_clock=lambda: now)
    manifest = await _manifest_for_store(store, session_id="memory-expiry-delete")
    await store.publish_context_view(manifest, publication_key="memory-expiry-publication")
    await store.select_context_view(
        ContextViewSelectionRequest(
            source_owner=manifest.source_owner,
            source_session_id=manifest.source_session_id,
            source_session_instance_id=manifest.source_session_instance_id,
            selector="latest",
            projection_schema=manifest.projection_schema,
            extension_set_commitment=manifest.extension_set_commitment,
            limits=ContextViewLimits(max_lifetime_seconds=1),
            selection_key="memory-expiry-selection",
        )
    )
    now += timedelta(seconds=2)
    await store.validate_context_view_source_closure(manifest.source_session_id)
    await store.delete_session(manifest.source_session_id)
    assert await store.load(manifest.source_session_id) is None


def test_memory_context_view_failed_selection_rolls_back_expiry() -> None:
    asyncio.run(_test_memory_context_view_failed_selection_rolls_back_expiry())


async def _test_memory_context_view_failed_selection_rolls_back_expiry() -> None:
    from datetime import UTC, datetime, timedelta

    now = datetime(2026, 1, 1, tzinfo=UTC)
    store = InMemorySessionStore(ownership_clock=lambda: now)
    manifest = await _manifest_for_store(store, session_id="memory-expiry-rollback")
    await store.publish_context_view(manifest, publication_key="memory-expiry-rollback-publication")
    request = ContextViewSelectionRequest(
        source_owner=manifest.source_owner,
        source_session_id=manifest.source_session_id,
        source_session_instance_id=manifest.source_session_instance_id,
        selector="latest",
        projection_schema=manifest.projection_schema,
        extension_set_commitment=manifest.extension_set_commitment,
        limits=ContextViewLimits(max_lifetime_seconds=1),
        selection_key="memory-expiry-rollback-selection",
    )
    await store.select_context_view(request)
    now += timedelta(seconds=2)
    prior_receipt = store._context_selection_receipts[request.selection_key]
    prior_events = store._context_lifecycle_events.copy()
    with pytest.raises(ValueError, match="selection key conflicts"):
        await store.select_context_view(
            request.model_copy(update={"minimum_transcript_cursor": 99})
        )
    assert store._context_selection_receipts[request.selection_key] == prior_receipt
    assert store._context_lifecycle_events == prior_events


async def _test_sqlite_context_view_expiry_publishes_lifecycle_evidence(tmp_path) -> None:
    from datetime import UTC, datetime, timedelta

    from cayu.sessions.context_views import ContextViewSelectionRequest

    now = datetime(2026, 1, 1, tzinfo=UTC)
    path = tmp_path / "context-views-expiry.sqlite"
    store = SQLiteSessionStore(path, ownership_clock=lambda: now)
    manifest = _manifest()
    try:
        manifest = await _manifest_for_store(store)
        await store.publish_context_view(manifest, publication_key="expiry-publication")
        selected = await store.select_context_view(
            ContextViewSelectionRequest(
                source_owner=manifest.source_owner,
                source_session_id=manifest.source_session_id,
                source_session_instance_id=manifest.source_session_instance_id,
                selector="latest",
                projection_schema=manifest.projection_schema,
                extension_set_commitment=manifest.extension_set_commitment,
                limits=ContextViewLimits(max_lifetime_seconds=1),
                selection_key="expiry-selection",
            )
        )
        now = now + timedelta(seconds=2)
        with pytest.raises(LookupError, match="expired"):
            await store.transition_context_view_ownership(
                ContextViewOwnershipRequest(
                    selection_key=selected.selection_key,
                    view_id=selected.view.view_id,
                    pin_commitment=selected.pin_commitment,
                    expected_state="selected",
                    expected_revision=selected.ownership_revision,
                    operation="adopt",
                    current_owner=manifest.source_owner,
                    destination_owner=manifest.source_owner,
                    operation_key="expired-adopt",
                )
            )
        # Expired selected pins are no longer retention owners even before a
        # later selection request performs bounded lifecycle cleanup.
        await store.validate_context_view_source_closure(manifest.source_session_id)
        await store.validate_context_view_compaction(
            manifest.source_session_id, manifest.transcript_cursor
        )
        retry = await store.select_context_view(
            ContextViewSelectionRequest(
                source_owner=manifest.source_owner,
                source_session_id=manifest.source_session_id,
                source_session_instance_id=manifest.source_session_instance_id,
                selector="latest",
                projection_schema=manifest.projection_schema,
                extension_set_commitment=manifest.extension_set_commitment,
                limits=ContextViewLimits(max_lifetime_seconds=1),
                selection_key="expiry-selection-retry",
            )
        )
        assert retry.selection_key == "expiry-selection-retry"
        events = await store.read_context_view_lifecycle_events(manifest.view_id)
        assert len(events) == 1
        assert events[0].state == "expired"
        assert events[0].selection_key == selected.selection_key
        assert events[0].owner_participant == selected.owner_participant
    finally:
        await store.close()


async def _test_sqlite_context_view_ownership_reconstructs_after_restart(tmp_path) -> None:
    from cayu.sessions.context_views import ContextViewSelectionRequest

    path = tmp_path / "context-views-restart.sqlite"
    store = SQLiteSessionStore(path)
    manifest = _manifest()
    try:
        manifest = await _manifest_for_store(store)
        await store.publish_context_view(manifest, publication_key="restart-publication")
        selected = await store.select_context_view(
            ContextViewSelectionRequest(
                source_owner=manifest.source_owner,
                source_session_id=manifest.source_session_id,
                source_session_instance_id=manifest.source_session_instance_id,
                selector="latest",
                projection_schema=manifest.projection_schema,
                extension_set_commitment=manifest.extension_set_commitment,
                limits=ContextViewLimits(),
                selection_key="restart-selection",
            )
        )
        request = ContextViewOwnershipRequest(
            selection_key=selected.selection_key,
            view_id=selected.view.view_id,
            pin_commitment=selected.pin_commitment,
            expected_state="selected",
            expected_revision=selected.ownership_revision,
            operation="adopt",
            current_owner=manifest.source_owner,
            destination_owner=manifest.source_owner,
            operation_key="restart-adopt",
        )
        adopted = await store.transition_context_view_ownership(request)
        assert adopted.state == "adopted"
        assert adopted.owner_participant == manifest.participant
    finally:
        await store.close()

    restored = SQLiteSessionStore(path)
    try:
        assert await restored.transition_context_view_ownership(request) == adopted
        events = await restored.read_context_view_lifecycle_events(manifest.view_id)
        assert [event.operation_key for event in events] == ["restart-adopt"]
        assert (
            await restored.read_context_view(
                manifest.view_id,
                source_session_id=manifest.source_session_id,
            )
        ).view == manifest
    finally:
        await restored.close()


async def _test_sqlite_context_view_publication_selection_and_readback(tmp_path) -> None:
    from cayu.sessions.context_views import ContextViewSelectionRequest

    store = SQLiteSessionStore(tmp_path / "context-views.sqlite")
    try:
        manifest = _manifest()
        manifest = await _manifest_for_store(store)
        first = await store.publish_context_view(manifest, publication_key="publication-1")
        replay = await store.publish_context_view(manifest, publication_key="publication-1")
        assert first == replay
        selected = await store.select_context_view(
            ContextViewSelectionRequest(
                source_owner=manifest.source_owner,
                source_session_id=manifest.source_session_id,
                source_session_instance_id=manifest.source_session_instance_id,
                selector="latest",
                projection_schema=manifest.projection_schema,
                extension_set_commitment=manifest.extension_set_commitment,
                limits=ContextViewLimits(),
                selection_key="selection-1",
            )
        )
        assert selected.view == manifest
        concurrent_request = ContextViewSelectionRequest(
            source_owner=manifest.source_owner,
            source_session_id=manifest.source_session_id,
            source_session_instance_id=manifest.source_session_instance_id,
            selector="latest",
            projection_schema=manifest.projection_schema,
            extension_set_commitment=manifest.extension_set_commitment,
            limits=ContextViewLimits(),
            selection_key="selection-concurrent",
        )
        concurrent = await asyncio.gather(
            store.select_context_view(concurrent_request),
            store.select_context_view(concurrent_request),
        )
        assert concurrent[0] == concurrent[1]
        with pytest.raises(ValueError, match="retention pin"):
            await store.validate_context_view_source_closure(manifest.source_session_id)
        with pytest.raises(ValueError, match="Pinned context-view material"):
            await store.validate_context_view_compaction(
                manifest.source_session_id, manifest.transcript_cursor
            )
        adopted = await store.transition_context_view_ownership(
            ContextViewOwnershipRequest(
                selection_key=selected.selection_key,
                view_id=selected.view.view_id,
                pin_commitment=selected.pin_commitment,
                expected_state="selected",
                expected_revision=selected.ownership_revision,
                operation="adopt",
                current_owner=manifest.source_owner,
                destination_owner=manifest.source_owner,
                operation_key="sqlite-adopt-1",
            )
        )
        assert adopted.state == "adopted"
        assert adopted.ownership_revision == 2
        events = await store.read_context_view_lifecycle_events(manifest.view_id)
        assert events[0].state == "adopted"
        assert (
            await store.select_context_view(
                ContextViewSelectionRequest(
                    source_owner=manifest.source_owner,
                    source_session_id=manifest.source_session_id,
                    source_session_instance_id=manifest.source_session_instance_id,
                    selector="latest",
                    projection_schema=manifest.projection_schema,
                    extension_set_commitment=manifest.extension_set_commitment,
                    limits=ContextViewLimits(),
                    selection_key="selection-1",
                )
            )
            == adopted
        )
        readback = await store.read_context_view(
            manifest.view_id, source_session_id=manifest.source_session_id
        )
        assert readback.view == manifest
    finally:
        await store.close()


def test_postgres_participant_creation_is_atomic_and_replayable(postgres_dsn) -> None:
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async def run():
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            (
                creation,
                first_receipt,
                first_id,
            ) = await _test_memory_participant_creation_is_atomic_and_replayable(
                store, cleanup=False
            )
        finally:
            await store.close()

        # Read-only inspection uses the same reconstruction path.  It must not
        # issue row-locking SELECTs, since PostgreSQL rejects those in a
        # read-only transaction.
        read_only = PostgresSessionStore(
            postgres_dsn, schema_mode=SchemaMode.VALIDATE, read_only=True
        )
        try:
            restored = await read_only.lookup_participant_session_creation(creation)
            assert restored is not None
            assert restored[0].id == first_id
            assert restored[1] == first_receipt
        finally:
            await read_only.close()

        cleanup = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            await cleanup.delete_session(first_id)
        finally:
            await cleanup.close()

    asyncio.run(run())


async def _test_memory_participant_creation_is_atomic_and_replayable(
    store=None, *, cleanup=True
) -> tuple[ParticipantSessionCreationRequest, ParticipantSessionCreationReceipt, str]:
    owner = OwnerRef(application_scope="app", owner_id="owner", incarnation="owner-1")
    participant = ParticipantRef(owner=owner, participant_id="participant", incarnation="p-1")
    creation = ParticipantSessionCreationRequest(
        request=RunRequest(agent_name="agent", messages=[]),
        creation_key="creation-1",
    )
    if store is None:
        store = InMemorySessionStore()

    def build(session):
        binding = ParticipantSessionBinding(
            application_scope="app",
            participant=participant,
            session_id=session.id,
            session_instance_id=session.instance_id,
            lifecycle_revision=1,
            configuration_revision=1,
            admission_generation=1,
            creator_commitment="sha256:" + "1" * 64,
            authorization_commitment="sha256:" + "2" * 64,
            initial_input_commitment="sha256:" + "3" * 64,
            request_commitment=creation.request_commitment,
            execution_profile_commitment="sha256:" + "4" * 64,
            historical_definition_json="{}",
            creation_key=creation.creation_key,
        )
        material = {
            "binding": binding.model_dump(mode="json"),
            "requested_session_id": None,
            "initial_input_commitment": binding.initial_input_commitment,
            "execution_profile_json": "{}",
            "schema_version": 1,
        }
        receipt_commitment = (
            "sha256:"
            + hashlib.sha256(
                canonical_bounded_durable_json_bytes(
                    material,
                    "receipt",
                    max_bytes=512 * 1024,
                    max_nodes=8192,
                    max_nesting=64,
                )
            ).hexdigest()
        )
        return binding, ParticipantSessionCreationReceipt(
            binding=binding,
            requested_session_id=None,
            initial_input_commitment=binding.initial_input_commitment,
            execution_profile_json="{}",
            receipt_commitment=receipt_commitment,
        )

    first, first_receipt = await store.create_participant_owned_session(
        creation,
        resolved_request=creation.request,
        identity=SessionIdentity(provider_name="provider", model="model"),
        binding_factory=build,
    )
    replay, replay_receipt = await store.create_participant_owned_session(
        creation,
        resolved_request=creation.request,
        identity=SessionIdentity(provider_name="provider", model="model"),
        binding_factory=build,
    )
    assert first.id == replay.id
    assert first_receipt == replay_receipt
    assert (
        await store.load_participant_session_binding(first.id)
    ).session_instance_id == first.instance_id
    conflicting = ParticipantSessionCreationRequest(
        request=RunRequest(agent_name="agent", messages=[Message.text("user", "different")]),
        creation_key="creation-1",
    )
    with pytest.raises(ValueError, match="conflicts with the request"):
        await store.create_participant_owned_session(
            conflicting,
            resolved_request=conflicting.request,
            identity=SessionIdentity(provider_name="provider", model="model"),
            binding_factory=build,
        )
    with pytest.raises(ValueError, match="conflicts with the request"):
        await store.lookup_participant_session_creation(conflicting)
    if isinstance(store, SQLiteSessionStore):
        path = store.path
        await store.close()
        store = SQLiteSessionStore(path)
        restored = await store.lookup_participant_session_creation(creation)
        assert restored is not None and restored[1] == first_receipt
        for column, changed in (
            ("creation_key", "different-key"),
            ("session_instance_id", "different-incarnation"),
            ("participant_id", "different-participant"),
        ):
            original = store._connection.execute(
                f"SELECT {column} FROM cayu_participant_session_bindings WHERE session_id = ?",
                (first.id,),
            ).fetchone()[0]
            with store._connection:
                store._connection.execute(
                    f"UPDATE cayu_participant_session_bindings SET {column} = ? WHERE session_id = ?",
                    (changed, first.id),
                )
            for read in (
                store.load_participant_session_binding,
                store.load_participant_session_creation_receipt,
            ):
                with pytest.raises(RuntimeError, match="indexes conflict"):
                    await read(first.id)
            with store._connection:
                store._connection.execute(
                    f"UPDATE cayu_participant_session_bindings SET {column} = ? WHERE session_id = ?",
                    (original, first.id),
                )
    if cleanup:
        await store.delete_session(first.id)
        assert await store.load_participant_session_binding(first.id) is None
        assert await store.lookup_participant_session_creation(creation) is None
    if isinstance(store, SQLiteSessionStore):
        await store.close()
    return creation, first_receipt, first.id
