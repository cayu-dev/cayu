from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from contextlib import asynccontextmanager
from hashlib import sha256

import pytest

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.collaboration._contracts import (
    MAX_DEPTH,
    MAX_ENVELOPE_BYTES,
    MAX_NODES,
    ObjectRef,
    OperationRef,
    OwnerRef,
)
from cayu.collaboration.exports import (
    ExportLimits,
    SessionExportAccessContext,
    SessionExportAuthorization,
    SessionExportDenied,
    SessionExportPolicy,
    SessionExportProjector,
    SessionExportReceipt,
    SessionExportRef,
    SessionExportRegistration,
    SessionExportRequest,
)
from cayu.collaboration.peer_content import (
    PeerAppendKey,
    PeerContentAppendAuthorization,
    PeerContentAppendRequest,
    PeerContentConflict,
    PeerContentExposureRequest,
    PeerContentOccurrence,
    PeerContentPayload,
    PeerDeliveryAttemptKey,
    PeerModelAttemptOrigin,
)
from cayu.evals.testing import ScriptedModelProvider


class QualifiedPeerProvider(ScriptedModelProvider):
    """Qualification double with the real OpenAI portable-input contract."""

    accept_peers = True

    def preflight_portable_messages(self, *, model, messages, tools):
        from cayu.providers.base import ModelProvider
        from cayu.providers.openai import OpenAIProvider

        if not self.accept_peers:
            return ModelProvider.preflight_portable_messages(
                self, model=model, messages=messages, tools=tools
            )
        return OpenAIProvider(api_key="test-key").preflight_portable_messages(
            model=model, messages=messages, tools=tools
        )


class QualificationPeerExposurePolicy(SessionExportPolicy):
    """Explicit test policy used by the public peer-delivery qualification."""

    def __init__(self) -> None:
        self.revoked = False
        self.allow_cleanup = True
        self.allowed_receipts = set()
        self.export_receipts: dict[str, tuple[SessionExportReceipt, str, str]] = {}
        self.tamper_append_authorization = False
        self.calls = []
        self.block = False
        self.close_fails = False
        self.projection = None
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def ref(self):
        return ObjectRef(
            owner=OwnerRef(
                application_scope="peer-qualification",
                owner_id="owner",
                incarnation="owner-incarnation",
            ),
            kind="peer-policy",
            object_id="qualification",
            incarnation="v1",
        )

    @asynccontextmanager
    async def acquire(self, context, **kwargs):
        if context.principal != "operator":
            raise SessionExportDenied()
        del kwargs
        yield SessionExportAuthorization(
            issuer=self.ref.owner,
            principal=context.principal,
            policy=self.ref,
            revision=1,
            expires_at_ms=4102444800000,
        )

    @asynccontextmanager
    async def acquire_peer_append(self, context, *, request, append_key, occurrence):
        del context, append_key
        exported = self.export_receipts.get(occurrence.source_export_receipt_id)
        if (
            self.revoked
            or occurrence.producer_receipt_id not in self.allowed_receipts
            or (self.export_receipts and exported is None)
        ):
            raise SessionExportDenied()
        if exported is not None:
            receipt, payload_sha256, consumer_id = exported
            expected = receipt.expected.intent.request
            if (
                expected.ref.session_id != occurrence.sender_session_id
                or expected.ref.session_instance_id != occurrence.sender_session_instance_id
                or expected.audience.owner_id != consumer_id
                or payload_sha256 != occurrence.payload.content_sha256
            ):
                raise SessionExportDenied()
        authorization = PeerContentAppendAuthorization(
            source_export_receipt_id=occurrence.source_export_receipt_id,
            producer_receipt_id=occurrence.producer_receipt_id,
            source_session_id=occurrence.sender_session_id,
            source_session_instance_id=occurrence.sender_session_instance_id,
            content_sha256=occurrence.payload.content_sha256,
            audience=occurrence.audience,
        )
        if self.tamper_append_authorization:
            authorization = authorization.model_copy(update={"content_sha256": "0" * 64})
        yield authorization

    @asynccontextmanager
    async def acquire_peer_exclusion(self, context, *, request, receipt, reason):
        if (
            context.principal != "operator"
            or not self.allow_cleanup
            or receipt.operation_key != request.operation_key
            or receipt.append_key != request.append_key
            or request.occurrence.producer_receipt_id not in self.allowed_receipts
        ):
            raise SessionExportDenied()
        yield

    @asynccontextmanager
    async def acquire_peer_read(self, context, *, append_key, receipt):
        if self.revoked:
            raise SessionExportDenied()
        if receipt.occurrence is not None:
            async with self.acquire_peer_append(
                context, request=None, append_key=append_key, occurrence=receipt.occurrence
            ):
                yield
        else:
            if context.principal != "operator":
                raise SessionExportDenied()
            yield

    def register_export(
        self, receipt: SessionExportReceipt, *, payload_sha256: str, consumer_id: str
    ) -> None:
        self.export_receipts[receipt.event_id] = (receipt, payload_sha256, consumer_id)

    @asynccontextmanager
    async def acquire_peer_exposure(self, context, **kwargs):
        del context
        if self.revoked:
            raise SessionExportDenied()
        self.calls.append(kwargs)
        if self.block:
            self.entered.set()
            await self.release.wait()
        try:
            yield self.projection or kwargs["occurrence"].payload
        finally:
            if self.close_fails:
                raise RuntimeError("peer receiver cleanup failed")


class PeerExportProjector(SessionExportProjector):
    @property
    def ref(self):
        return ObjectRef(
            owner=OwnerRef(
                application_scope="peer-qualification",
                owner_id="owner",
                incarnation="owner-incarnation",
            ),
            kind="peer-projector",
            object_id="source-text",
            incarnation="v1",
        )

    def project(self, source):
        text = " ".join(
            part.text for record in source for part in record.message.content if part.type == "text"
        )
        return {"text": text}

    def validate(self, source, output, audience):
        return isinstance(output, dict) and isinstance(output.get("text"), str)


def _payload(text="finding") -> PeerContentPayload:
    material = {"text": text, "artifact_commitments": []}
    digest = sha256(
        canonical_bounded_durable_json_bytes(
            material,
            "test.peer_payload",
            max_bytes=MAX_ENVELOPE_BYTES,
            max_nodes=MAX_NODES,
            max_nesting=MAX_DEPTH,
        )
    ).hexdigest()
    return PeerContentPayload(text=text, content_sha256=digest)


def _exposure_id(delivery, model_attempt_id):
    return PeerContentExposureRequest.for_model_attempt(
        append_key=delivery.append_key,
        append_operation_key=delivery.operation_key,
        model_attempt_id=model_attempt_id,
        provider_name="scripted",
        capability_version=1,
    ).exposure_id


def test_runtime_exposure_ids_are_bounded_and_bind_every_identity_field():
    delivery = _request()
    inputs = dict(
        append_key=delivery.append_key,
        append_operation_key="o" * 512,
        model_attempt_id="a" * 512,
        provider_name="p" * 512,
        capability_version=1,
    )
    expected = PeerContentExposureRequest.for_model_attempt(**inputs)
    assert len(expected.operation_key.encode()) <= 512
    assert len(expected.exposure_id.encode()) <= 512
    assert PeerContentExposureRequest.for_model_attempt(**inputs) == expected
    alternatives = dict(
        append_key=delivery.append_key.model_copy(update={"projection_id": "another-projection"}),
        append_operation_key="other-operation",
        model_attempt_id="other-attempt",
        provider_name="other-provider",
        capability_version=2,
    )
    for field, value in alternatives.items():
        changed = PeerContentExposureRequest.for_model_attempt(**(inputs | {field: value}))
        assert changed.operation_key != expected.operation_key
        assert changed.exposure_id != expected.exposure_id


def _request() -> PeerContentAppendRequest:
    payload = _payload()
    occurrence_material = {
        "occurrence_id": "occurrence-1",
        "sender_participant_id": "sender",
        "sender_participant_incarnation": "sender-incarnation",
        "sender_session_id": "source-session",
        "sender_session_instance_id": "source-instance",
        "producer_receipt_id": "producer-receipt",
        "source_export_receipt_id": "export-receipt",
        "payload": payload.model_dump(mode="json"),
        "audience": ["consumer"],
    }
    provenance = sha256(
        canonical_bounded_durable_json_bytes(
            occurrence_material,
            "test.peer_occurrence",
            max_bytes=MAX_ENVELOPE_BYTES,
            max_nodes=MAX_NODES,
            max_nesting=MAX_DEPTH,
        )
    ).hexdigest()
    occurrence = PeerContentOccurrence(
        occurrence_id="occurrence-1",
        sender_participant_id="sender",
        sender_participant_incarnation="sender-incarnation",
        sender_session_id="source-session",
        sender_session_instance_id="source-instance",
        producer_receipt_id="producer-receipt",
        source_export_receipt_id="export-receipt",
        payload=payload,
        audience=("consumer",),
        provenance_sha256=provenance,
    )
    append_key = PeerAppendKey(
        collaboration_namespace="collaboration",
        collaboration_generation=1,
        occurrence_id=occurrence.occurrence_id,
        consumer_id="consumer",
        consumer_participant_incarnation="consumer-incarnation",
        projection_id="peer-text",
        projection_schema="peer-text-v1",
        target_session_id="target-session",
        target_session_instance_id="target-instance",
    )
    attempt_key = PeerDeliveryAttemptKey(
        append_key=append_key,
        interest_id="interest-1",
        attempt_generation=1,
        target_run_epoch=0,
        target_transcript_cursor=0,
        withdrawal_generation=1,
        deadline_at_ms=4102444800000,
    )
    return PeerContentAppendRequest(
        operation_key="append-operation-1",
        append_key=append_key,
        attempt_key=attempt_key,
        occurrence=occurrence,
    )


def _delivery_request(
    *,
    source,
    target,
    sender,
    consumer,
    suffix="public",
    source_export_receipt_id="export-receipt",
    payload=None,
) -> PeerContentAppendRequest:
    base = _request()
    payload = base.occurrence.payload if payload is None else payload
    occurrence_material = {
        "occurrence_id": f"occurrence-{suffix}",
        "sender_participant_id": sender.participant_id,
        "sender_participant_incarnation": sender.incarnation,
        "sender_session_id": source.id,
        "sender_session_instance_id": source.instance_id,
        "producer_receipt_id": f"producer-{suffix}",
        "source_export_receipt_id": source_export_receipt_id,
        "payload": payload.model_dump(mode="json"),
        "audience": [consumer.participant_id],
    }
    provenance = sha256(
        canonical_bounded_durable_json_bytes(
            occurrence_material,
            "test.peer_public_occurrence",
            max_bytes=MAX_ENVELOPE_BYTES,
            max_nodes=MAX_NODES,
            max_nesting=MAX_DEPTH,
        )
    ).hexdigest()
    occurrence = PeerContentOccurrence(
        occurrence_id=f"occurrence-{suffix}",
        sender_participant_id=sender.participant_id,
        sender_participant_incarnation=sender.incarnation,
        sender_session_id=source.id,
        sender_session_instance_id=source.instance_id,
        producer_receipt_id=f"producer-{suffix}",
        source_export_receipt_id=source_export_receipt_id,
        payload=payload,
        audience=(consumer.participant_id,),
        provenance_sha256=provenance,
    )
    append_key = base.append_key.model_copy(
        update={
            "occurrence_id": occurrence.occurrence_id,
            "consumer_id": consumer.participant_id,
            "consumer_participant_incarnation": consumer.incarnation,
            "target_session_id": target.id,
            "target_session_instance_id": target.instance_id,
        }
    )
    attempt_key = base.attempt_key.model_copy(
        update={
            "append_key": append_key,
            "target_run_epoch": target.run_epoch,
            "target_transcript_cursor": 0,
        }
    )
    return PeerContentAppendRequest(
        operation_key=f"append-{suffix}",
        append_key=append_key,
        attempt_key=attempt_key,
        occurrence=occurrence,
    )


def test_registered_receiver_delegates_current_policy_and_fails_closed() -> None:
    from cayu.collaboration._session_export_coordinator import SessionExportCoordinator
    from cayu.collaboration.peer_content import PeerContentExposureItem
    from cayu.sessions.base import InMemorySessionStore
    from cayu.vaults.redaction import SecretRedactor

    request = _request()
    origin = PeerModelAttemptOrigin(
        target_session_id=request.append_key.target_session_id,
        target_session_instance_id=request.append_key.target_session_instance_id,
        run_epoch=1,
        root_invocation_id="root",
        requester_principal="alice",
        interaction_id="interaction",
        model_step_id="step",
        model_attempt_id="attempt",
        append_key=request.append_key,
        provider_name="provider",
        model="model",
        capability_version=1,
        exposure_generation=1,
    )

    async def run() -> None:
        policy = QualificationPeerExposurePolicy()
        policy.allowed_receipts.add("producer-receipt")
        coordinator = SessionExportCoordinator(
            store=InMemorySessionStore(),
            registration=SessionExportRegistration(
                owner=OwnerRef(
                    application_scope="scope", owner_id="owner", incarnation="owner-incarnation"
                ),
                policy=policy,
                projectors=(),
                limits=ExportLimits(max_exports=4, max_pending=2, max_retained_bytes=4096),
            ),
            redactor=SecretRedactor(()),
        )
        item = PeerContentExposureItem(
            origin=origin,
            occurrence=request.occurrence,
            audience=OwnerRef(
                application_scope="scope", owner_id="consumer", incarnation="consumer-incarnation"
            ),
        )
        async with coordinator.acquire_peer_exposures_runtime((item,)) as projections:
            assert projections == (request.occurrence.payload,)
        for field, changed in (
            ("model_attempt_id", "another-attempt"),
            ("model", "another-model"),
            ("root_invocation_id", "another-invocation"),
        ):
            conflicting = item.model_copy(
                update={"origin": origin.model_copy(update={field: changed})}
            )
            with pytest.raises(SessionExportDenied):
                coordinator.acquire_peer_exposures_runtime((item, conflicting))
        with pytest.raises(SessionExportDenied):
            coordinator.acquire_peer_exposures_runtime((item.model_dump(),))
        async with coordinator.acquire_peer_exposure_runtime(
            origin,
            append_key=request.append_key,
            occurrence=request.occurrence,
            audience=OwnerRef(
                application_scope="scope", owner_id="consumer", incarnation="consumer-incarnation"
            ),
            provider_name="provider",
            model="model",
            model_attempt_id="attempt",
            capability_version=1,
        ) as projection:
            assert projection == request.occurrence.payload
        for changed in (
            origin.model_copy(update={"model_attempt_id": "other-attempt"}),
            origin.model_copy(update={"target_session_id": "other-session"}),
            origin.model_copy(update={"target_session_instance_id": "other-instance"}),
        ):
            with pytest.raises(SessionExportDenied):
                coordinator.acquire_peer_exposure_runtime(
                    changed,
                    append_key=request.append_key,
                    occurrence=request.occurrence,
                    audience=OwnerRef(
                        application_scope="scope",
                        owner_id="consumer",
                        incarnation="consumer-incarnation",
                    ),
                    provider_name="provider",
                    model="model",
                    model_attempt_id="attempt",
                    capability_version=1,
                )
        policy.revoked = True
        with pytest.raises(SessionExportDenied):
            async with coordinator.acquire_peer_exposure_runtime(
                origin,
                append_key=request.append_key,
                occurrence=request.occurrence,
                audience=OwnerRef(
                    application_scope="scope",
                    owner_id="consumer",
                    incarnation="consumer-incarnation",
                ),
                provider_name="provider",
                model="model",
                model_attempt_id="attempt",
                capability_version=1,
            ):
                pass

    asyncio.run(run())


def test_peer_content_requires_matching_payload_commitment() -> None:
    request = _request()
    assert (
        request.occurrence.to_message_part(
            append_key=request.append_key,
            projection_id="peer-text",
            operation_key=request.operation_key,
        ).executable
        is False
    )
    with pytest.raises(ValueError):
        PeerContentPayload(text="changed", content_sha256=request.occurrence.payload.content_sha256)


@pytest.mark.parametrize("outcome", ["exposed", "not_exposed"])
def test_public_exposure_settlement_rejects_caller_evidence(outcome):
    from tests.core.test_participant_identity import CONTEXT

    from cayu import CayuApp
    from cayu.collaboration.peer_content import PeerContentUnavailable

    async def run():
        app = CayuApp(enable_logging=False)
        claim = PeerContentExposureRequest(
            operation_key="forged-settlement",
            append_key=_request().append_key,
            exposure_id="forged-exposure",
            model_attempt_id="forged-attempt",
            provider_name="fake",
            capability_version=1,
            outcome=outcome,
            reason="denied" if outcome == "not_exposed" else None,
        )
        with pytest.raises(PeerContentUnavailable, match="runtime owner"):
            await app.expose_peer_content(claim, context=CONTEXT)
        assert (
            await app.session_store.read_peer_content_exposure(claim.append_key, claim.exposure_id)
            is None
        )

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_public_participant_peer_delivery_uses_registered_policy(
    backend, tmp_path, request, monkeypatch
) -> None:
    from tests.core.test_participant_identity import CONTEXT, create, registration

    from cayu import CayuApp
    from cayu.agents import AgentSpec
    from cayu.collaboration.memory import InMemoryCollaborationStore
    from cayu.events import EventType
    from cayu.messages import Message
    from cayu.providers.base import ModelStreamEvent
    from cayu.sessions.base import InMemorySessionStore, RunRequest
    from cayu.sessions.context_views import (
        ParticipantSessionCreationRequest,
        ParticipantSessionExecutionRequest,
    )
    from cayu.sessions.invocation import InvocationOriginClaim
    from cayu.storage import PostgresSessionStore, SQLiteSessionStore
    from cayu.storage.collaboration_postgres import PostgresCollaborationStore
    from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
    from cayu.storage.migrations import SchemaMode

    async def run() -> None:
        from tests.core.test_targeted_tool_grants import _codec

        if backend == "memory":
            session_store = InMemorySessionStore(public_authority_alias_codec=_codec())
            collaboration_store = InMemoryCollaborationStore()
        elif backend == "sqlite":
            session_store = SQLiteSessionStore(
                tmp_path / "peer-session.sqlite", public_authority_alias_codec=_codec()
            )
            collaboration_store = SQLiteCollaborationStore(tmp_path / "peer-collaboration.sqlite")
        else:
            dsn = request.getfixturevalue("postgres_dsn")
            session_store = PostgresSessionStore(
                dsn, schema_mode=SchemaMode.CREATE, public_authority_alias_codec=_codec()
            )
            collaboration_store = PostgresCollaborationStore(dsn, schema_mode=SchemaMode.CREATE)
        peer_policy = QualificationPeerExposurePolicy()
        peer_projector = PeerExportProjector()
        from cayu.context.base import DefaultContextPolicy, UsageTriggeredContextPolicy
        from cayu.context.counting import ContextCountingConfig, ContextCountingMode
        from cayu.runtime.session_message_lifecycle import (
            SessionMessageAccessContext,
            SessionMessageAccessPolicy,
            SessionMessageQuery,
        )

        class QueuePolicy(SessionMessageAccessPolicy):
            def authorize(self, context, **kwargs):
                return context.subject == "queue-reader"

        from cayu.vaults.redaction import SecretRedactor

        secret = "peer-projection-secret-canary"
        application = CayuApp(
            secret_redactor=SecretRedactor(secret),
            context_counting=ContextCountingConfig(mode=ContextCountingMode.OBSERVE),
            session_message_access_policy=QueuePolicy(),
            session_store=session_store,
            collaboration_store=collaboration_store,
            collaboration=registration(),
            session_exports=SessionExportRegistration(
                owner=OwnerRef(
                    application_scope="peer-qualification",
                    owner_id="owner",
                    incarnation="owner-incarnation",
                ),
                policy=peer_policy,
                projectors=(peer_projector,),
                limits=ExportLimits(max_exports=8, max_pending=4, max_retained_bytes=65536),
            ),
            enable_logging=False,
        )
        initialized = await application.initialize_collaboration()
        _, source_receipt = await create(application, initialized, key="source")
        _, target_receipt = await create(application, initialized, key="target")
        source_participant = source_receipt.participants[0].reference
        target_participant = target_receipt.participants[0].reference

        class SerializedProvider(QualifiedPeerProvider):
            phase = None
            entered = asyncio.Event()
            peer_count_calls = 0
            count_calls = 0

            async def count_input_tokens(self, request):
                from cayu.providers.base import has_peer_content

                self.count_calls += 1
                if has_peer_content(request):
                    self.peer_count_calls += 1
                return None

            async def stream(self, request):
                from cayu.providers.base import record_peer_serialization
                from cayu.providers.openai import build_openai_payload

                if any(p.type == "peer_content" for m in request.messages for p in m.content):
                    from cayu.runtime._model_step_executor import _model_request_fingerprint

                    assert secret not in request.model_dump_json()
                    active = await session_store.load_active_model_completion_stage(
                        peer_policy.calls[-1]["origin"].target_session_id
                    )
                    assert active is not None
                    assert active.stage.intent["request_fingerprint"] == _model_request_fingerprint(
                        provider_name=self.name, model_request=request
                    )

                if self.phase == "before_serialization":
                    self.entered.set()
                    await asyncio.Future()
                build_openai_payload(request, stream=True)
                await record_peer_serialization(request)
                if self.phase == "after_serialization":
                    self.entered.set()
                    await asyncio.Future()
                async for event in super().stream(request):
                    yield event

        provider = SerializedProvider(
            [
                [
                    ModelStreamEvent.text_delta("source complete"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("target complete"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        application.register_provider(provider, default=True)
        application.register_agent(
            AgentSpec(name="reviewer", model="model"),
            context_policy=UsageTriggeredContextPolicy(
                triggered_policy=DefaultContextPolicy(),
                trigger_estimated_context_tokens=1,
                verify_estimate_with_provider_count=True,
            ),
        )
        source_request = ParticipantSessionCreationRequest(
            request=RunRequest(
                agent_name="reviewer",
                messages=[Message.text("user", "source")],
                invocation_origin=InvocationOriginClaim(subject=CONTEXT.principal),
            ),
            creation_key="source-session",
        )
        target_request = ParticipantSessionCreationRequest(
            request=RunRequest(
                agent_name="reviewer",
                messages=[Message.text("user", "target")],
                invocation_origin=InvocationOriginClaim(subject=CONTEXT.principal),
            ),
            creation_key="target-session",
        )
        source, _ = await application.create_participant_session(
            source_request, participant=source_participant, context=CONTEXT
        )
        target, _ = await application.create_participant_session(
            target_request, participant=target_participant, context=CONTEXT
        )
        source_execution = ParticipantSessionExecutionRequest(
            request=source_request.request.model_copy(update={"session_id": source.id}),
            session_instance_id=source.instance_id,
            execution_key="source-execute",
        )
        source_events = [
            event
            async for event in application.execute_participant_session(
                source_execution, participant=source_participant, context=CONTEXT
            )
        ]
        assert any(event.type is EventType.SESSION_COMPLETED for event in source_events)
        export_namespace = await application.initialize_session_exports(
            source.id, context=SessionExportAccessContext(principal=CONTEXT.principal)
        )
        export_request = SessionExportRequest(
            ref=SessionExportRef(
                session_id=source.id,
                session_instance_id=source.instance_id,
                operation=OperationRef(
                    application_scope="peer-qualification",
                    namespace_incarnation=export_namespace.namespace_incarnation,
                    generation=export_namespace.generation,
                    caller_key="source-peer-export",
                ),
            ),
            source_indices=(1,),
            audience=OwnerRef(
                application_scope="peer-qualification",
                owner_id=target_participant.participant_id,
                incarnation=target_participant.incarnation,
            ),
            projector=peer_projector.ref,
            policy=peer_policy.ref,
        )
        export_receipt = await application.export_session(
            export_request, context=SessionExportAccessContext(principal=CONTEXT.principal)
        )
        exported_view = await application.read_session_export(
            export_request,
            context=SessionExportAccessContext(principal=CONTEXT.principal),
        )
        exported_payload = PeerContentPayload(
            text=exported_view["text"],
            content_sha256=sha256(
                canonical_bounded_durable_json_bytes(
                    {"text": exported_view["text"], "artifact_commitments": []},
                    "test.peer_payload",
                    max_bytes=MAX_ENVELOPE_BYTES,
                    max_nodes=MAX_NODES,
                    max_nesting=MAX_DEPTH,
                )
            ).hexdigest(),
        )
        peer_policy.register_export(
            export_receipt,
            payload_sha256=exported_payload.content_sha256,
            consumer_id=target_participant.participant_id,
        )
        delivery = _delivery_request(
            source=source,
            target=target,
            sender=source_participant,
            consumer=target_participant,
            source_export_receipt_id=export_receipt.event_id,
            payload=exported_payload,
        )
        peer_policy.allowed_receipts.add("producer-public")
        delivery = delivery.model_copy(update={"operation_key": "k" * 512})
        forged_delivery = delivery.model_copy(update={"operation_key": "append-forged-policy"})
        peer_policy.tamper_append_authorization = True
        with pytest.raises(SessionExportDenied):
            await application.append_peer_content(forged_delivery, context=CONTEXT)
        assert await session_store.read_peer_content(forged_delivery.append_key) is None
        peer_policy.tamper_append_authorization = False
        provider.accept_peers = False
        with pytest.raises(ValueError, match="peer-content support"):
            await application.append_peer_content(delivery, context=CONTEXT)
        assert await session_store.read_peer_content(delivery.append_key) is None
        provider.accept_peers = True
        stale_delivery = delivery.model_copy(
            update={
                "attempt_key": delivery.attempt_key.model_copy(
                    update={
                        "target_transcript_cursor": 1,
                    }
                ),
            }
        )
        pending = await application.append_peer_content(stale_delivery, context=CONTEXT)
        assert pending.status == "pending"
        peer_policy.revoked = True
        with pytest.raises(SessionExportDenied):
            await application.service_pending_peer_content(target.id, context=CONTEXT)
        assert (await session_store.read_peer_content(delivery.append_key)).status == "pending"
        peer_policy.revoked = False
        original_append = application._session_engine.append_peer_content

        async def lose_ack(value, **kwargs):
            await original_append(value, **kwargs)
            raise ConnectionError("committed peer append acknowledgement lost")

        with monkeypatch.context() as patch:
            patch.setattr(application._session_engine, "append_peer_content", lose_ack)
            with pytest.raises(ConnectionError, match="acknowledgement lost"):
                await application.service_pending_peer_content(target.id, context=CONTEXT)
        recovered = [await application.append_peer_content(stale_delivery, context=CONTEXT)]
        assert len(recovered) == 1
        assert recovered[0].status == "appended"
        # Servicing uses a new receiving cursor, never a rewritten caller tuple.
        with pytest.raises(PeerContentConflict):
            await application.append_peer_content(delivery, context=CONTEXT)
        delivery = stale_delivery
        append = await application.append_peer_content(delivery, context=CONTEXT)
        assert append.status == "appended"
        assert (
            await application.read_peer_content(delivery.append_key, context=CONTEXT)
        ).occurrence is not None
        peer_policy.revoked = True
        withheld = await application.read_peer_content(delivery.append_key, context=CONTEXT)
        assert withheld.status == "appended"
        assert withheld.disclosure == "withheld" and withheld.occurrence is None
        assert exported_payload.text not in withheld.model_dump_json()
        inspection = await application.inspect_session_messages(
            SessionMessageQuery(session_id=target.id),
            context=SessionMessageAccessContext(subject="queue-reader"),
        )
        assert inspection.records
        assert inspection.records[0].status == "queued"
        assert inspection.records[0].validity == "unreadable"
        assert inspection.records[0].message is None
        assert exported_payload.text not in inspection.model_dump_json()
        peer_policy.revoked = False
        replayed_append = await application.append_peer_content(delivery, context=CONTEXT)
        assert replayed_append.replayed is True
        from datetime import UTC, datetime

        # Store time, never caller time, decides the deadline at the mutation.
        fixed_now = datetime(2030, 1, 1, tzinfo=UTC)
        now_ms = int(fixed_now.timestamp() * 1000)
        for offset in (-1, 0, 1):
            deadline_target, _ = await application.create_participant_session(
                ParticipantSessionCreationRequest(
                    request=target_request.request, creation_key=f"deadline-target-{offset}"
                ),
                participant=target_participant,
                context=CONTEXT,
            )
            deadline_request = _delivery_request(
                source=source,
                target=deadline_target,
                sender=source_participant,
                consumer=target_participant,
                suffix=f"deadline-{offset}",
                source_export_receipt_id=export_receipt.event_id,
                payload=exported_payload,
            )
            peer_policy.allowed_receipts.add(f"producer-deadline-{offset}")
            deadline_request = deadline_request.model_copy(
                update={
                    "attempt_key": deadline_request.attempt_key.model_copy(
                        update={"deadline_at_ms": now_ms + offset}
                    )
                }
            )
            if backend == "postgres":
                original_clock = session_store._session_store_now

                async def fixed_store_time(_cur):
                    return fixed_now

                session_store._session_store_now = fixed_store_time
            else:
                original_clock = session_store._ownership_clock
                session_store._ownership_clock = lambda: fixed_now
            try:
                deadline_result = await application.append_peer_content(
                    deadline_request, context=CONTEXT
                )
            finally:
                if backend == "postgres":
                    session_store._session_store_now = original_clock
                else:
                    session_store._ownership_clock = original_clock
            assert deadline_result.status == ("appended" if offset > 0 else "excluded")
            if offset <= 0:
                assert deadline_result.reason == "delivery_deadline_expired"
                replacement = deadline_request.model_copy(
                    update={
                        "operation_key": f"deadline-replacement-{offset}",
                        "replaces_operation_key": deadline_request.operation_key,
                        "attempt_key": deadline_request.attempt_key.model_copy(
                            update={
                                "attempt_generation": 2,
                                "deadline_at_ms": 4102444800000,
                            }
                        ),
                    }
                )
                assert (
                    await application.append_peer_content(replacement, context=CONTEXT)
                ).status == "appended"
                old = await application.append_peer_content(deadline_request, context=CONTEXT)
                assert old.replayed and old.status == "excluded"
                changed_deadline = deadline_request.model_copy(
                    update={
                        "attempt_key": deadline_request.attempt_key.model_copy(
                            update={"deadline_at_ms": now_ms + 100}
                        )
                    }
                )
                with pytest.raises(PeerContentConflict):
                    await application.append_peer_content(changed_deadline, context=CONTEXT)

        race_target, _ = await application.create_participant_session(
            ParticipantSessionCreationRequest(
                request=target_request.request, creation_key="replacement-race-target"
            ),
            participant=target_participant,
            context=CONTEXT,
        )
        race_request = _delivery_request(
            source=source,
            target=race_target,
            sender=source_participant,
            consumer=target_participant,
            suffix="replacement-race",
            source_export_receipt_id=export_receipt.event_id,
            payload=exported_payload,
        )
        peer_policy.allowed_receipts.add("producer-replacement-race")
        race_request = race_request.model_copy(
            update={
                "attempt_key": race_request.attempt_key.model_copy(
                    update={"target_transcript_cursor": 1}
                )
            }
        )
        assert (
            await application.append_peer_content(race_request, context=CONTEXT)
        ).status == "pending"
        replacement = race_request.model_copy(
            update={
                "operation_key": "replacement-race-second",
                "replaces_operation_key": race_request.operation_key,
                "attempt_key": race_request.attempt_key.model_copy(
                    update={
                        "attempt_generation": 2,
                        "target_transcript_cursor": 0,
                    }
                ),
            }
        )
        with pytest.raises(PeerContentConflict):
            await application.append_peer_content(replacement, context=CONTEXT)
        results = await asyncio.gather(
            application.exclude_peer_content(race_request, reason="withdrawn", context=CONTEXT),
            application.append_peer_content(replacement, context=CONTEXT),
            return_exceptions=True,
        )
        assert results[0].status == "excluded"
        assert isinstance(results[1], PeerContentConflict) or results[1].status == "appended"
        assert (
            await application.append_peer_content(replacement, context=CONTEXT)
        ).status == "appended"
        assert (
            await application.append_peer_content(race_request, context=CONTEXT)
        ).status == "excluded"
        assert (
            await session_store.read_peer_content(replacement.append_key)
        ).operation_key == replacement.operation_key
        old_readback = await application.read_peer_content(
            race_request.append_key, expected=race_request, context=CONTEXT
        )
        assert old_readback.status == "excluded"
        with pytest.raises(PeerContentConflict):
            await application.read_peer_content(
                race_request.append_key,
                expected=race_request.model_copy(update={"wake_policy": "ordinary_continuation"}),
                context=CONTEXT,
            )

        target_execution = ParticipantSessionExecutionRequest(
            request=target_request.request.model_copy(update={"session_id": target.id}),
            session_instance_id=target.instance_id,
            execution_key="target-execute",
        )
        projected_text = "Authorized projection " + secret
        peer_policy.projection = _payload(projected_text)
        target_events = [
            event
            async for event in application.execute_participant_session(
                target_execution, participant=target_participant, context=CONTEXT
            )
        ]
        assert any(event.type is EventType.SESSION_COMPLETED for event in target_events), [
            (event.type, event.payload) for event in target_events
        ]
        assert peer_policy.calls
        assert any(
            part.type == "peer_content" and part.text.startswith("Authorized projection ")
            for message in provider.requests[-1].messages
            for part in message.content
        )
        peer_policy.projection = None
        assert any(
            any(
                part.type == "peer_content"
                for message in request.messages
                for part in message.content
            )
            for request in provider.requests[1:]
        )
        if backend in {"sqlite", "postgres"}:
            competing_store = (
                SQLiteSessionStore(
                    tmp_path / "peer-session.sqlite", public_authority_alias_codec=_codec()
                )
                if backend == "sqlite"
                else PostgresSessionStore(
                    dsn, schema_mode=SchemaMode.CREATE, public_authority_alias_codec=_codec()
                )
            )
            competing_exposure = PeerContentExposureRequest(
                operation_key="competing-exposure-operation",
                append_key=delivery.append_key,
                exposure_id="competing-exposure-id",
                model_attempt_id="competing-attempt",
                provider_name="scripted",
                capability_version=1,
                outcome="exposed",
            )
            competing_results = await asyncio.gather(
                session_store.record_peer_content_exposure(competing_exposure),
                competing_store.record_peer_content_exposure(competing_exposure),
            )
            assert sum(not result.replayed for result in competing_results) == 1
            assert sum(result.replayed for result in competing_results) == 1
            await competing_store.close()
        interrupted_exposures = []
        for phase in (
            "guard",
            "before_serialization",
            "after_serialization",
            "serialization_ack_loss",
            "cleanup_failure",
        ):
            cancelled_target_request = ParticipantSessionCreationRequest(
                request=RunRequest(
                    agent_name="reviewer",
                    messages=[Message.text("user", "cancelled target")],
                    invocation_origin=InvocationOriginClaim(subject=CONTEXT.principal),
                ),
                creation_key=f"cancelled-target-session-{phase}",
            )
            cancelled_target, _ = await application.create_participant_session(
                cancelled_target_request, participant=target_participant, context=CONTEXT
            )
            cancelled_delivery = _delivery_request(
                source=source,
                target=cancelled_target,
                sender=source_participant,
                consumer=target_participant,
                suffix=f"cancelled-{phase}",
                source_export_receipt_id=export_receipt.event_id,
                payload=exported_payload,
            )
            peer_policy.allowed_receipts.add(f"producer-cancelled-{phase}")
            assert (
                await application.append_peer_content(cancelled_delivery, context=CONTEXT)
            ).status == "appended"
            peer_policy.block = phase == "guard"
            peer_policy.entered = asyncio.Event()
            provider.phase = "before_serialization" if phase == "cleanup_failure" else phase
            peer_policy.close_fails = phase == "cleanup_failure"
            provider.entered = asyncio.Event()
            cancelled_execution = ParticipantSessionExecutionRequest(
                request=cancelled_target_request.request.model_copy(
                    update={"session_id": cancelled_target.id}
                ),
                session_instance_id=cancelled_target.instance_id,
                execution_key="cancelled-target-execute",
            )

            async def consume_cancelled_execution(cancelled_execution=cancelled_execution):
                async for _ in application.execute_participant_session(
                    cancelled_execution, participant=target_participant, context=CONTEXT
                ):
                    pass

            original_record = session_store.record_peer_content_exposure
            if phase == "serialization_ack_loss":

                async def lose_serialization_ack(exposure, original_record=original_record):
                    await original_record(exposure)
                    raise ConnectionError("serialization acknowledgement lost")

                session_store.record_peer_content_exposure = lose_serialization_ack
            cancelled_task = asyncio.create_task(consume_cancelled_execution())
            try:
                if phase == "serialization_ack_loss":
                    await asyncio.wait_for(cancelled_task, 10)
                else:
                    await asyncio.wait_for(
                        (peer_policy.entered if phase == "guard" else provider.entered).wait(), 5
                    )
                    cancelled_task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await cancelled_task
                    assert cancelled_task.cancelled()
                    assert cancelled_task.cancelling() >= 1
            finally:
                session_store.record_peer_content_exposure = original_record
            cancelled_attempt_id = peer_policy.calls[-1]["model_attempt_id"]
            pending_exposure = await session_store.read_peer_content_exposure(
                cancelled_delivery.append_key,
                _exposure_id(cancelled_delivery, cancelled_attempt_id),
            )
            assert pending_exposure is not None
            assert pending_exposure.outcome == (
                "exposed"
                if phase in {"after_serialization", "serialization_ack_loss"}
                else "pending"
            )
            interrupted_exposures.append(pending_exposure)
            peer_policy.block = False
            peer_policy.close_fails = False
            peer_policy.release.set()
        provider.phase = None
        revoked_target_request = ParticipantSessionCreationRequest(
            request=RunRequest(
                agent_name="reviewer",
                messages=[Message.text("user", "revoked target")],
                invocation_origin=InvocationOriginClaim(subject=CONTEXT.principal),
            ),
            creation_key="revoked-target-session",
        )
        revoked_target, _ = await application.create_participant_session(
            revoked_target_request, participant=target_participant, context=CONTEXT
        )
        revoked_delivery = _delivery_request(
            source=source,
            target=revoked_target,
            sender=source_participant,
            consumer=target_participant,
            suffix="revoked",
            source_export_receipt_id=export_receipt.event_id,
            payload=exported_payload,
        )
        peer_policy.allowed_receipts.add("producer-revoked")
        assert (
            await application.append_peer_content(revoked_delivery, context=CONTEXT)
        ).status == "appended"
        peer_policy.revoked = True
        calls_before_revoked_run = len(peer_policy.calls)
        revoked_execution = ParticipantSessionExecutionRequest(
            request=revoked_target_request.request.model_copy(
                update={"session_id": revoked_target.id}
            ),
            session_instance_id=revoked_target.instance_id,
            execution_key="revoked-target-execute",
        )
        revoked_events = [
            event
            async for event in application.execute_participant_session(
                revoked_execution, participant=target_participant, context=CONTEXT
            )
        ]
        assert not any(event.type is EventType.SESSION_COMPLETED for event in revoked_events)
        assert len(peer_policy.calls) == calls_before_revoked_run
        assert provider.peer_count_calls == 0
        assert provider.count_calls > 0

        if backend != "memory":
            completed_exposure = await session_store.read_peer_content_exposure(
                delivery.append_key,
                _exposure_id(delivery, peer_policy.calls[0]["model_attempt_id"]),
            )
            assert completed_exposure is not None
            exposures = [completed_exposure, *interrupted_exposures]
            result = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-m", "tests.recovery.peer_creation_readback_worker"],
                input=json.dumps(
                    {
                        "backend": backend,
                        "location": str(tmp_path / "peer-session.sqlite")
                        if backend == "sqlite"
                        else dsn,
                        "request": delivery.model_dump(mode="json"),
                        "qualification_alias_codec": True,
                        "exposures": [item.model_dump(mode="json") for item in exposures],
                    }
                ),
                text=True,
                capture_output=True,
                timeout=30,
                check=True,
            )
            reconstructed = json.loads(result.stdout)
            assert reconstructed["status"] == "appended"
            assert reconstructed["same_target"]
            assert reconstructed["exact_exposures"] == [True] * len(exposures)

        if backend == "sqlite":
            await session_store.close()
            exposure_id = _exposure_id(delivery, peer_policy.calls[0]["model_attempt_id"])
            reopened_session_store = SQLiteSessionStore(
                tmp_path / "peer-session.sqlite", public_authority_alias_codec=_codec()
            )
            recovered_exposure = await reopened_session_store.read_peer_content_exposure(
                delivery.append_key, exposure_id
            )
            assert recovered_exposure is not None
            assert recovered_exposure.outcome == "exposed"
            original_replay = await reopened_session_store.read_peer_content_attempt(delivery)
            assert original_replay is not None and original_replay.status == "appended"
            for expected in interrupted_exposures:
                reconstructed = await reopened_session_store.read_peer_content_exposure(
                    expected.append_key, expected.exposure_id
                )
                assert reconstructed == expected
            old_attempt = await reopened_session_store.append_peer_content(race_request)
            assert old_attempt.status == "excluded" and old_attempt.replayed
            successor = await reopened_session_store.append_peer_content(replacement)
            assert successor.status == "appended" and successor.replayed
            assert successor.operation_key == replacement.operation_key
            await reopened_session_store.close()
            await collaboration_store.close()
        elif backend == "postgres":
            await session_store.close()
            exposure_id = _exposure_id(delivery, peer_policy.calls[0]["model_attempt_id"])
            reopened_session_store = PostgresSessionStore(
                dsn, schema_mode=SchemaMode.CREATE, public_authority_alias_codec=_codec()
            )
            recovered_exposure = await reopened_session_store.read_peer_content_exposure(
                delivery.append_key, exposure_id
            )
            assert recovered_exposure is not None
            assert recovered_exposure.outcome == "exposed"
            original_replay = await reopened_session_store.read_peer_content_attempt(delivery)
            assert original_replay is not None and original_replay.status == "appended"
            for expected in interrupted_exposures:
                reconstructed = await reopened_session_store.read_peer_content_exposure(
                    expected.append_key, expected.exposure_id
                )
                assert reconstructed == expected
            old_attempt = await reopened_session_store.append_peer_content(race_request)
            assert old_attempt.status == "excluded" and old_attempt.replayed
            successor = await reopened_session_store.append_peer_content(replacement)
            assert successor.status == "appended" and successor.replayed
            assert successor.operation_key == replacement.operation_key
            await reopened_session_store.close()
            await collaboration_store.close()

    asyncio.run(run())


def test_peer_content_rejects_consumer_outside_audience() -> None:
    request = _request()
    data = request.model_dump(mode="json")
    data["append_key"]["consumer_id"] = "other"
    data["attempt_key"]["append_key"]["consumer_id"] = "other"
    with pytest.raises(ValueError):
        PeerContentAppendRequest.model_validate(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("deadline_at_ms", None),
        ("deadline_at_ms", True),
        ("deadline_at_ms", 0),
        ("attempt_generation", True),
        ("attempt_generation", 0),
        ("attempt_generation", 65),
    ],
)
def test_peer_attempt_boundaries_fail_closed(field, value):
    request = _request()
    mutated = request.model_copy(
        update={"attempt_key": request.attempt_key.model_copy(update={field: value})}
    )
    with pytest.raises(ValueError):
        PeerContentAppendRequest.model_validate(mutated)


@pytest.mark.parametrize("generation", [1, 64])
def test_peer_attempt_generation_bounds(generation):
    request = _request()
    material = request.model_dump(mode="python")
    material["attempt_key"]["attempt_generation"] = generation
    assert (
        PeerContentAppendRequest.model_validate(material).attempt_key.attempt_generation
        == generation
    )


@pytest.mark.parametrize("backend", ["memory"])
def test_store_append_replay_and_conflict(tmp_path, backend, request):
    from cayu.collaboration._contracts import OwnerRef
    from cayu.collaboration.participants import ParticipantRef
    from cayu.runtime.session_message_lifecycle import SessionMessageQuery
    from cayu.sessions.base import InMemorySessionStore, RunRequest, SessionIdentity, SessionStatus
    from cayu.sessions.context_views import ParticipantSessionBinding

    async def run():
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(session_id="target-session", agent_name="test", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        store._sessions[session.id] = session.model_copy(update={"status": SessionStatus.RUNNING})
        store._participant_session_bindings["source-session"] = (
            ParticipantSessionBinding.model_construct(
                application_scope="scope",
                participant=ParticipantRef.model_construct(
                    owner=OwnerRef.model_construct(application_scope="scope", owner_id="owner"),
                    participant_id="sender",
                    incarnation="sender-incarnation",
                ),
                session_id="source-session",
                session_instance_id="source-instance",
            )
        )
        store._participant_session_bindings[session.id] = ParticipantSessionBinding.model_construct(
            application_scope="scope",
            participant=ParticipantRef.model_construct(
                owner=OwnerRef.model_construct(application_scope="scope", owner_id="owner"),
                participant_id="consumer",
                incarnation="consumer-incarnation",
            ),
            session_id=session.id,
            session_instance_id=session.instance_id,
        )
        material = _request().model_dump(mode="json")
        material["append_key"]["target_session_instance_id"] = session.instance_id
        material["attempt_key"]["append_key"] = material["append_key"].copy()
        request = PeerContentAppendRequest.model_validate(material)
        target_binding = store._participant_session_bindings.pop(session.id)
        pending = await store.append_peer_content(request)
        assert pending.status == "pending"
        store._participant_session_bindings[session.id] = target_binding
        from cayu.collaboration.peer_content import PeerContentUnavailable

        with pytest.raises(PeerContentUnavailable, match="qualification"):
            await store.append_peer_content(request)
        assert await store.read_peer_content(request.append_key) == pending

        async def invalid_async_qualifier(session):
            raise AssertionError("An async preflight must not run.")

        with pytest.raises(PeerContentUnavailable, match="synchronous"):
            await store.append_peer_content(request, qualify_target=invalid_async_qualifier)
        assert await store.read_peer_content(request.append_key) == pending

        async def native_admit(value, **kwargs):
            return await store.append_peer_content(
                value, qualify_target=lambda session: None, **kwargs
            )

        recovered = await store.retry_pending_peer_content(
            session.id,
            expected_session_instance_id=session.instance_id,
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=0,
            admit=native_admit,  # isolated native conformance, no public disclosure
        )
        assert len(recovered) == 1
        result = recovered[0]
        assert result.status == "appended"
        exposure = await store.record_peer_content_exposure(
            PeerContentExposureRequest(
                operation_key="exposure-operation-1",
                append_key=request.append_key,
                exposure_id="exposure-1",
                model_attempt_id="attempt-1",
                provider_name="fake",
                capability_version=1,
                outcome="exposed",
            )
        )
        assert exposure.outcome == "exposed"
        assert (
            await store.read_peer_content_exposure(request.append_key, "exposure-1")
        ) == exposure
        assert (
            await store.record_peer_content_exposure(
                PeerContentExposureRequest(
                    operation_key="exposure-operation-1",
                    append_key=request.append_key,
                    exposure_id="exposure-1",
                    model_attempt_id="attempt-1",
                    provider_name="fake",
                    capability_version=1,
                    outcome="exposed",
                )
            )
        ).replayed
        with pytest.raises(PeerContentConflict):
            await store.record_peer_content_exposure(
                PeerContentExposureRequest(
                    operation_key="exposure-operation-1",
                    append_key=request.append_key,
                    exposure_id="different-exposure",
                    model_attempt_id="attempt-1",
                    provider_name="fake",
                    capability_version=1,
                    outcome="exposed",
                )
            )
        concurrent_request = PeerContentExposureRequest(
            operation_key="exposure-operation-concurrent",
            append_key=request.append_key,
            exposure_id="exposure-concurrent",
            model_attempt_id="attempt-concurrent",
            provider_name="fake",
            capability_version=1,
            outcome="exposed",
        )
        concurrent = await asyncio.gather(
            *(store.record_peer_content_exposure(concurrent_request) for _ in range(4))
        )
        assert sum(not result.replayed for result in concurrent) == 1
        assert sum(result.replayed for result in concurrent) == 3
        replay = await store.append_peer_content(request)
        assert replay.replayed
        assert replay.model_copy(update={"replayed": False}) == result
        queued = await store.inspect_session_messages(
            SessionMessageQuery(session_id=session.id, limit=10)
        )
        assert queued.records[0].message is not None
        assert queued.records[0].message.message is not None
        assert queued.records[0].message.message.content[0].type == "peer_content"
        await store.deliver_queued_session_messages(session.id, include_on_idle=True)
        transcript = await store.load_transcript(session.id)
        assert len(transcript) == 1
        assert transcript[0].content[0].type == "peer_content"
        material["wake_policy"] = "ordinary_continuation"
        with pytest.raises(PeerContentConflict):
            await store.append_peer_content(PeerContentAppendRequest.model_validate(material))
        assert await store.load_transcript(session.id) == transcript

    asyncio.run(run())


@pytest.mark.parametrize("provider", ["openai", "chat_completions", "anthropic", "bedrock"])
def test_integrated_payload_builders_preserve_peer_attribution(provider):
    from cayu.messages import Message, MessageRole
    from cayu.providers.anthropic import build_anthropic_payload
    from cayu.providers.base import ModelRequest
    from cayu.providers.bedrock import build_bedrock_converse_payload
    from cayu.providers.chat_completions import build_chat_completions_payload
    from cayu.providers.openai import build_openai_payload

    builders = {
        "openai": build_openai_payload,
        "chat_completions": build_chat_completions_payload,
        "anthropic": build_anthropic_payload,
        "bedrock": build_bedrock_converse_payload,
    }
    request = _request()
    part = request.occurrence.to_message_part(
        append_key=request.append_key,
        projection_id="peer-text",
        operation_key=request.operation_key,
    )
    request = ModelRequest(
        model="test-model",
        messages=[Message(role=MessageRole.ASSISTANT, content=(part,))],
    )
    payload = builders[provider](request)
    encoded = json.dumps(payload)
    assert "Peer content from sender" in encoded
    assert "occurrence-1" in encoded
    assert "finding" in encoded
    assert '"role": "user"' not in encoded
    assert '"role": "system"' not in encoded
    assert "tool_result" not in encoded


def test_generic_provider_preflight_rejects_unqualified_peer_content():
    from cayu.messages import Message, MessageRole
    from cayu.providers.base import _preflight_provider_portable_messages

    request = _request()
    part = request.occurrence.to_message_part(
        append_key=request.append_key,
        projection_id="peer-text",
        operation_key=request.operation_key,
    )
    with pytest.raises(ValueError, match="does not declare peer-content support"):
        _preflight_provider_portable_messages(
            model="test-model",
            messages=[Message(role=MessageRole.ASSISTANT, content=(part,))],
            tools=[],
            supports_system_messages=True,
            supports_tool_history=True,
            supports_tool_definitions=True,
            supports_file_attachments=True,
        )


@pytest.mark.parametrize(
    "adapter", ["openai", "chat", "anthropic", "bedrock", "vertex", "subscription"]
)
def test_real_provider_serialization_reports_positive_boundary(adapter):
    """The actual adapters, not just payload helpers, report before transport."""
    from tests.core.test_openai_subscription_provider import StaticSubscriptionAuth
    from tests.core.test_vertex_provider import _provider as vertex_provider

    from cayu.messages import Message
    from cayu.providers.anthropic import AnthropicProvider
    from cayu.providers.base import ModelRequest
    from cayu.providers.bedrock import BedrockProvider
    from cayu.providers.chat_completions import ChatCompletionsProvider
    from cayu.providers.openai import OpenAIProvider
    from cayu.providers.openai_subscription import OpenAISubscriptionProvider

    async def run():
        evidence = []
        payloads = []

        class Transport:
            async def stream_response_events(self, **kwargs):
                payloads.append(kwargs["payload"])
                assert evidence == ["serialized"]
                if False:
                    yield

            stream_chat_completions = stream_response_events
            stream_message_events = stream_response_events

            async def create_response(self, **kwargs):
                payloads.append(kwargs["payload"])
                assert evidence == ["serialized"]
                return {}

            def converse_stream(self, **payload):
                payloads.append(payload)
                assert evidence == ["serialized"]
                return {"stream": iter(())}

        transport = Transport()
        providers = {
            "openai": lambda: OpenAIProvider(api_key="test-key", transport=transport),
            "chat": lambda: ChatCompletionsProvider(api_key="test-key", transport=transport),
            "anthropic": lambda: AnthropicProvider(api_key="test-key", transport=transport),
            "bedrock": lambda: BedrockProvider(client=transport),
            "vertex": lambda: vertex_provider(transport),
            "subscription": lambda: OpenAISubscriptionProvider(
                auth=StaticSubscriptionAuth(), transport=transport
            ),
        }
        provider = providers[adapter]()
        delivery = _request()
        part = delivery.occurrence.to_message_part(
            append_key=delivery.append_key,
            projection_id="peer-text",
            operation_key=delivery.operation_key,
        )
        model_request = ModelRequest(
            model="test-model", messages=[Message(role="assistant", content=(part,))]
        )

        async def serialized(actual):
            assert actual is model_request
            evidence.append("serialized")

        model_request._peer_serialization_observer = serialized
        if adapter in {"openai", "anthropic", "bedrock", "vertex"}:
            from cayu.collaboration.peer_content import PeerContentUnavailable

            with pytest.raises(PeerContentUnavailable, match="token counting"):
                await provider.count_input_tokens(model_request)
            assert payloads == []
            assert evidence == []
        events = provider.runtime_stream(model_request)
        try:
            async for _ in events:
                pass
        finally:
            await events.aclose()
        assert evidence == ["serialized"]
        assert len(payloads) == 1
        assert "Peer content from sender" in json.dumps(payloads[0])
        assert "finding" in json.dumps(payloads[0])
        assert "_peer_serialization_observer" not in model_request.model_dump_json()
        reconstructed = ModelRequest.model_validate_json(model_request.model_dump_json())
        events = provider.runtime_stream(reconstructed)
        try:
            async for _ in events:
                pass
        finally:
            await events.aclose()
        assert len(payloads) == 1  # Identical caller bytes do not restore runtime authority.

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_exclusion_exact_replay_rejects_changed_evidence(tmp_path, backend):
    from cayu.sessions.base import InMemorySessionStore
    from cayu.storage import SQLiteSessionStore

    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "exclusion.sqlite")
        )
        try:
            request = _request()
            receipt = await store.exclude_peer_content(request, reason="withdrawn")
            assert receipt.status == "excluded"
            replay = await store.exclude_peer_content(request, reason="withdrawn")
            assert replay.replayed
            assert replay.model_copy(update={"replayed": False}) == receipt
            with pytest.raises(PeerContentConflict):
                await store.exclude_peer_content(request, reason="different")
            changed = request.model_dump(mode="json")
            changed["operation_key"] = "different-operation"
            changed["wake_policy"] = "ordinary_continuation"
            with pytest.raises(PeerContentConflict):
                await store.exclude_peer_content(
                    PeerContentAppendRequest.model_validate(changed), reason="withdrawn"
                )
            assert await store.read_peer_content(request.append_key) == receipt
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(run())
