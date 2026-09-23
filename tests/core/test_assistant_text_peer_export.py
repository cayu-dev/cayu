"""Real OpenAI transport -> source-owned visible text -> peer serialization."""

import asyncio
import json
import subprocess
import sys
from contextlib import asynccontextmanager
from hashlib import sha256

import httpx
import pytest
from tests.core.test_participant_identity import CONTEXT, app, create, registration
from tests.core.test_peer_content import QualificationPeerExposurePolicy, _delivery_request
from tests.core.test_session_creation_fence import _collaboration_factory, _store_factory

from cayu.agents import AgentSpec
from cayu.collaboration._contracts import ObjectRef, OperationRef, OwnerRef
from cayu.collaboration._session_export_store import source_digest
from cayu.collaboration.exports import (
    ExportLimits,
    SessionExportAccessContext,
    SessionExportConflict,
    SessionExportDenied,
    SessionExportProjector,
    SessionExportRef,
    SessionExportRegistration,
    SessionExportRequest,
)
from cayu.collaboration.peer_content import PeerContentPayload
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.openai import HttpxOpenAITransport, OpenAIProvider
from cayu.sessions.base import ResumeRequest, RunRequest
from cayu.sessions.context_views import (
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
)
from cayu.sessions.invocation import InvocationOriginClaim

PRIVATE = "private-provider-state-canary"
HIDDEN = "hidden-thinking-canary"
VISIBLE = "Review finding: validate the amount before transfer."
EXPORT_CONTEXT = SessionExportAccessContext(principal=CONTEXT.principal)


class Policy(QualificationPeerExposurePolicy):
    @asynccontextmanager
    async def acquire(self, context, **kwargs):
        if self.revoked:
            raise SessionExportDenied()
        async with super().acquire(context, **kwargs) as grant:
            yield grant

    @asynccontextmanager
    async def acquire_peer_exposure(self, context, **kwargs):
        async with (
            self.acquire_peer_append(
                context,
                request=None,
                append_key=kwargs["append_key"],
                occurrence=kwargs["occurrence"],
            ),
            super().acquire_peer_exposure(context, **kwargs) as payload,
        ):
            yield payload


class TextProjector(SessionExportProjector):
    def __init__(self, policy):
        self.identity = ObjectRef(
            owner=policy.ref.owner, kind="projector", object_id="visible-text", incarnation="v1"
        )
        self.sources = []

    @property
    def ref(self):
        return self.identity

    def project(self, source):
        self.sources.append(source)
        assert all(row.message.role == "assistant" for row in source)
        assert all(part.type == "text" for row in source for part in row.message.content)
        assert PRIVATE not in repr(source) and HIDDEN not in repr(source)
        return {"text": "".join(part.text for row in source for part in row.message.content)}

    def validate(self, source, output, audience):
        assert PRIVATE not in repr(source) and HIDDEN not in repr(source)
        return output == {
            "text": "".join(part.text for row in source for part in row.message.content)
        }


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_real_assistant_export_peer_journey(backend, tmp_path, request):
    store_factory = _store_factory(backend, tmp_path, request)
    collaboration_factory = _collaboration_factory(backend, tmp_path, request)
    store, collaboration = store_factory(), collaboration_factory()
    registered = registration()
    policy = Policy()
    projector = TextProjector(policy)
    payloads = []

    async def handler(request):
        body = json.loads(request.content)
        payloads.append(body)
        first = len(payloads) == 1
        output = [
            {
                "type": "message",
                "id": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": VISIBLE if first else "Acknowledged",
                        "annotations": [],
                    }
                ],
            }
        ]
        if first:
            output.insert(
                0,
                {
                    "type": "reasoning",
                    "id": "reasoning",
                    "encrypted_content": PRIVATE,
                    "summary": [{"type": "summary_text", "text": HIDDEN}],
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "response-source" if first else "response-target",
                "object": "response",
                "model": "gpt-test",
                "status": "completed",
                "output": output,
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = HttpxOpenAITransport()
        transport._client._client = client

        def build():
            current = app(
                collaboration,
                registered,
                session_store=store,
                session_exports=SessionExportRegistration(
                    owner=policy.ref.owner,
                    policy=policy,
                    projectors=(projector,),
                    limits=ExportLimits(max_exports=8, max_pending=4, max_retained_bytes=65536),
                ),
            )
            current.register_provider(
                OpenAIProvider(
                    api_key="deterministic-test-key", streaming=False, transport=transport
                ),
                default=True,
            )
            current.register_agent(AgentSpec(name="reviewer", model="gpt-test"))
            return current

        current = build()
        try:
            initialized = await current.initialize_collaboration()
            _, sender_receipt = await create(current, initialized, key="sender")
            _, consumer_receipt = await create(current, initialized, key="consumer")
            sender = sender_receipt.participants[0].reference
            consumer = consumer_receipt.participants[0].reference

            async def inert(participant, key):
                creation = ParticipantSessionCreationRequest(
                    creation_key=key,
                    request=RunRequest(
                        agent_name="reviewer",
                        messages=[Message.text("user", key)],
                        invocation_origin=InvocationOriginClaim(subject=CONTEXT.principal),
                    ),
                )
                session, _ = await current.create_participant_session(
                    creation, participant=participant, context=CONTEXT
                )
                execution = ParticipantSessionExecutionRequest(
                    request=creation.request.model_copy(update={"session_id": session.id}),
                    session_instance_id=session.instance_id,
                    execution_key=key + "-execute",
                )
                return session, execution

            source, source_execution = await inert(sender, "source")
            target, target_execution = await inert(consumer, "target")
            events = [
                event
                async for event in current.execute_participant_session(
                    source_execution, participant=sender, context=CONTEXT
                )
            ]
            assert any(event.type == EventType.SESSION_COMPLETED for event in events)
            rows = (await store.load_transcript_window(source.id, start_index=0, limit=16)).records
            row = next(
                row
                for row in rows
                if any(part.type == "text" and part.text == VISIBLE for part in row.message.content)
            )
            assert any(part.type == "provider_state" for part in row.message.content)
            assert PRIVATE in row.model_dump_json()
            namespace = await current.initialize_session_exports(source.id, context=EXPORT_CONTEXT)
            export_request = SessionExportRequest(
                ref=SessionExportRef(
                    session_id=source.id,
                    session_instance_id=source.instance_id,
                    operation=OperationRef(
                        application_scope=policy.ref.owner.application_scope,
                        namespace_incarnation=namespace.namespace_incarnation,
                        generation=namespace.generation,
                        caller_key="assistant-export",
                    ),
                ),
                source_indices=(row.index,),
                source_selection="assistant_visible_text_v1",
                audience=OwnerRef(
                    application_scope=policy.ref.owner.application_scope,
                    owner_id=consumer.participant_id,
                    incarnation=consumer.incarnation,
                ),
                projector=projector.ref,
                policy=policy.ref,
            )
            with pytest.raises(SessionExportDenied):
                await current.export_session(
                    export_request.model_copy(update={"source_selection": "whole_records"}),
                    context=EXPORT_CONTEXT,
                )
            receipt = await current.export_session(export_request, context=EXPORT_CONTEXT)
            assert receipt.expected.intent.source_commitment == source_digest(projector.sources[0])
            assert source_digest((row,)) not in receipt.model_dump_json()
            assert projector.sources[0][0].index == row.index
            assert projector.sources[0][0].interaction_id == row.interaction_id
            with pytest.raises(SessionExportConflict):
                await current.export_session(
                    export_request.model_copy(update={"source_selection": "whole_records"}),
                    context=EXPORT_CONTEXT,
                )
            assert await current.read_session_export(export_request, context=EXPORT_CONTEXT) == {
                "text": VISIBLE
            }

            # Reconstruct native owners before replay and before recipient execution.
            if backend != "memory":
                await store.close()
                await collaboration.close()
                restored = await asyncio.to_thread(
                    subprocess.run,
                    [sys.executable, "-m", "tests.recovery.assistant_text_export_readback_worker"],
                    input=json.dumps(
                        {
                            "backend": backend,
                            "location": str(tmp_path / "creation-fence.sqlite")
                            if backend == "sqlite"
                            else request.getfixturevalue("postgres_dsn"),
                            "request": export_request.model_dump(mode="json"),
                        }
                    ),
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                assert restored.returncode == 0, restored.stderr
                assert json.loads(restored.stdout) == {
                    "receipt": receipt.model_dump(mode="json"),
                    "payload": {"text": VISIBLE},
                }
                assert (
                    PRIVATE not in restored.stdout + restored.stderr
                    and HIDDEN not in restored.stdout + restored.stderr
                )
                store, collaboration = store_factory(), collaboration_factory()
            current = build()
            await current.initialize_collaboration()
            assert await current.export_session(export_request, context=EXPORT_CONTEXT) == receipt
            assert len(projector.sources) == 1
            exported = await current.read_session_export(export_request, context=EXPORT_CONTEXT)
            commitment = sha256(
                json.dumps(
                    {"artifact_commitments": [], "text": exported["text"]},
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            payload = PeerContentPayload(text=exported["text"], content_sha256=commitment)
            policy.register_export(
                receipt, payload_sha256=payload.content_sha256, consumer_id=consumer.participant_id
            )
            delivery = _delivery_request(
                source=source,
                target=target,
                sender=sender,
                consumer=consumer,
                source_export_receipt_id=receipt.event_id,
                payload=payload,
            )
            policy.allowed_receipts.add(delivery.occurrence.producer_receipt_id)
            assert (
                await current.append_peer_content(delivery, context=CONTEXT)
            ).status == "appended"
            events = [
                event
                async for event in current.execute_participant_session(
                    target_execution, participant=consumer, context=CONTEXT
                )
            ]
            assert any(event.type == EventType.SESSION_COMPLETED for event in events)
            assert len(payloads) == 2 and VISIBLE in json.dumps(payloads[1])
            for value in (
                receipt.model_dump_json(),
                json.dumps(exported),
                json.dumps(payloads[1]),
                (
                    await current.read_peer_content(delivery.append_key, context=CONTEXT)
                ).model_dump_json(),
            ):
                assert PRIVATE not in value and HIDDEN not in value
            policy.revoked = True
            with pytest.raises(SessionExportDenied):
                await current.read_session_export(export_request, context=EXPORT_CONTEXT)
            events = [
                event
                async for event in current.resume(
                    ResumeRequest(
                        session_id=target.id, messages=[Message.text("user", "continue")]
                    ),
                    context=CONTEXT,
                )
            ]
            assert not any(event.type == EventType.SESSION_COMPLETED for event in events)
            assert len(payloads) == 2
        finally:
            await current.drain_session_exports()
            if backend != "memory":
                await store.close()
                await collaboration.close()
