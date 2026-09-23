"""Public compaction keeps peer material typed and independently authorized."""

from uuid import uuid4

import pytest
from tests.core.test_participant_identity import CONTEXT, app, create, registration
from tests.core.test_peer_content import (
    QualificationPeerExposurePolicy,
    QualifiedPeerProvider,
    _delivery_request,
)
from tests.core.test_session_creation_fence import _collaboration_factory, _store_factory

from cayu.agents import AgentSpec
from cayu.collaboration.exports import ExportLimits, SessionExportRegistration
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    ModelCompactor,
    PromptCacheCompactor,
    TranscriptDigestCompactor,
)
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent, record_peer_serialization
from cayu.providers.openai import build_openai_payload
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.base import CompactSessionRequest, ResumeRequest, RunRequest
from cayu.sessions.context_views import (
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
)
from cayu.sessions.invocation import InvocationOriginClaim


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("compactor_kind", ["digest", "model", "cache"])
async def test_public_peer_compaction_and_restart(backend, compactor_kind, tmp_path, request):
    factory = _store_factory(backend, tmp_path, request)
    collaboration = _collaboration_factory(backend, tmp_path, request)()
    store = factory()
    policy = QualificationPeerExposurePolicy()
    unique = uuid4().hex
    collaboration_registration = registration()

    def response(request):
        return [
            ModelStreamEvent.text_delta("safe response"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]

    class Provider(QualifiedPeerProvider):
        @property
        def execution_profile_identity(self):
            return ExecutionProfileBehaviorIdentity(
                name="tests:peer-compaction-provider",
                behavior_version="1",
                implementation_version="1",
            )

        async def stream(self, request):
            build_openai_payload(request, stream=True)
            await record_peer_serialization(request)
            async for event in super().stream(request):
                yield event

    provider = Provider(response_factory=response)
    summarizer = Provider(response_factory=response)

    def application():
        compactor = (
            TranscriptDigestCompactor()
            if compactor_kind == "digest"
            else ModelCompactor(provider=summarizer, model="model")
            if compactor_kind == "model"
            else PromptCacheCompactor(provider=summarizer)
        )
        result = app(
            collaboration,
            collaboration_registration,
            session_store=store,
            session_exports=SessionExportRegistration(
                owner=policy.ref.owner,
                policy=policy,
                projectors=(),
                limits=ExportLimits(max_exports=8, max_pending=4, max_retained_bytes=65536),
            ),
        )
        result.register_provider(provider, default=True)
        result.register_agent(
            AgentSpec(name="reviewer", model="model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
                compact_after_messages=100 if compactor_kind == "digest" else 1,
            ),
        )
        return result

    current = application()
    initialized = await current.initialize_collaboration()
    _, source_record = await create(current, initialized, key="source")
    _, target_record = await create(current, initialized, key="target")
    sender = source_record.participants[0].reference
    consumer = target_record.participants[0].reference

    async def inert(participant, key):
        creation = ParticipantSessionCreationRequest(
            request=RunRequest(
                agent_name="reviewer",
                messages=[Message.text("user", key)],
                invocation_origin=InvocationOriginClaim(subject=CONTEXT.principal),
            ),
            creation_key=key + unique,
        )
        session, _ = await current.create_participant_session(
            creation, participant=participant, context=CONTEXT
        )
        return session, creation

    try:
        source, _ = await inert(sender, "source")
        target, creation = await inert(consumer, "target")
        delivery = _delivery_request(
            source=source, target=target, sender=sender, consumer=consumer, suffix=unique
        )
        policy.allowed_receipts.add(delivery.occurrence.producer_receipt_id)
        assert (await current.append_peer_content(delivery, context=CONTEXT)).status == "appended"
        events = [
            event
            async for event in current.execute_participant_session(
                ParticipantSessionExecutionRequest(
                    request=creation.request.model_copy(update={"session_id": target.id}),
                    session_instance_id=target.instance_id,
                    execution_key="execute",
                ),
                participant=consumer,
                context=CONTEXT,
            )
        ]
        assert any(event.type == EventType.SESSION_COMPLETED for event in events), [
            (e.type, e.payload) for e in events
        ]

        async def resume(text):
            return [
                event
                async for event in current.resume(
                    ResumeRequest(session_id=target.id, messages=[Message.text("user", text)]),
                    context=CONTEXT,
                )
            ]

        events = await resume("next turn")

        async def compact(key):
            loaded = await store.load(target.id)
            snapshot = await store.load_transcript_snapshot(target.id)
            return [
                event
                async for event in current.compact_session(
                    CompactSessionRequest(
                        session_id=target.id,
                        idempotency_key=key,
                        expected_run_epoch=loaded.run_epoch,
                        expected_transcript_cursor=snapshot.cursor,
                    ),
                    context=CONTEXT,
                )
            ]

        assert any(event.type == EventType.SESSION_COMPLETED for event in events)
        if compactor_kind == "digest":
            events = await compact("explicit-initial")
        assert any(event.type == EventType.CONTEXT_COMPACTION_COMPLETED for event in events)
        checkpoint = await store.load_checkpoint(target.id)
        assert delivery.occurrence.payload.text not in checkpoint["context_compaction"]["summary"]
        assert (
            sum(
                part.type == "peer_content"
                for message in provider.requests[-1].messages
                for part in message.content
            )
            == 1
        )
        if backend != "memory":
            await store.close()
            store = factory()
            current = application()
            await current.initialize_collaboration()
        events = await resume("after restart")
        assert any(event.type == EventType.SESSION_COMPLETED for event in events)
        assert (
            sum(
                part.type == "peer_content"
                for message in provider.requests[-1].messages
                for part in message.content
            )
            == 1
        )
        if compactor_kind != "digest":
            assert summarizer.requests
        for captured in summarizer.requests:
            assert delivery.occurrence.payload.text not in captured.model_dump_json()
            assert not any(
                part.type == "peer_content"
                for message in captured.messages
                for part in message.content
            )

        policy.revoked = True
        if compactor_kind == "digest":
            compacted = await compact("explicit-revoked")
            assert any(event.type == EventType.SESSION_CHECKPOINTED for event in compacted)
        dispatched = len(provider.requests)
        events = await resume("revoked turn")
        assert not any(event.type == EventType.SESSION_COMPLETED for event in events)
        assert len(provider.requests) == dispatched
        for captured in summarizer.requests:
            assert delivery.occurrence.payload.text not in captured.model_dump_json()
    finally:
        if backend != "memory":
            await store.close()
            await collaboration.close()
