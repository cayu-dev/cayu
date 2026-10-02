"""Actual participant execution and resume; source/wait setup remains explicit."""

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from tests.core.test_budget_binding import _binding, _limit
from tests.core.test_collaboration_request_foundation import RequestResolver
from tests.core.test_collaboration_waits import wait_for
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_participant_identity import app as collaboration_app
from tests.core.test_participant_identity import stores as stores
from tests.core.test_temporary_continuation_permits import prepared

from cayu.agents import AgentSpec
from cayu.budgets import BudgetReservation
from cayu.collaboration._capabilities import CapabilityDescriptor
from cayu.collaboration._clarification_commands import ClarificationOpenReceipt
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.collaboration.request_access import RequestRegistration
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime._session_continuation import (
    ContinuationConflict,
    ContinuationUnavailable,
    continuation_digest,
)
from cayu.runtime._session_continuation_owner import LATCH_FAMILY, SessionContinuationOwner
from cayu.runtime._temporary_continuation import (
    TemporaryServiceAdmission,
    TemporaryServiceRecord,
    temporary_service_key,
)
from cayu.runtime._temporary_continuation_permits import TemporaryServicePermitAuthority
from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint
from cayu.sessions.base import InMemorySessionStore, ResumeRequest, RunRequest
from cayu.sessions.context_views import (
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
)
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    "registration_failure", (None, "before_commit", "after_commit", "changed_budget", "timeout")
)
@pytest.mark.parametrize("side_session", (False, True))
async def test_bound_participant_service_uses_real_resume_preparation(
    stores, tmp_path, request, monkeypatch, registration_failure, side_session
):
    source = stores()
    if isinstance(source, InMemoryCollaborationStore):
        store = InMemorySessionStore()
    elif isinstance(source, SQLiteCollaborationStore):
        store = SQLiteSessionStore(tmp_path / "runtime.sqlite")
    else:
        store = PostgresSessionStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )
    try:
        scope = uuid4().hex
        budget_binding = _binding(
            application_scope=scope,
            binding_id="service:" + scope,
            limits=(
                _limit().model_copy(
                    update={
                        "reservation": BudgetReservation(
                            max_input_tokens=8192, max_output_tokens=8192
                        )
                    }
                ),
            ),
        )
        application, initial, template = await prepared(
            source,
            open_question=True,
            session_store=store,
            budget_binding=budget_binding,
            service_timeout_ms=120_000 if registration_failure == "timeout" else None,
        )
        async with source._transaction(initial.binding.application_scope, write=False) as tx:
            opening = ClarificationOpenReceipt.model_validate(
                await tx.get(
                    "operations", operation_key(template.dispatch.intent.question.operation)
                )
            )
        resolver = RequestResolver(opening.command.expected.intent.request)
        application = collaboration_app(
            source,
            application._participant_coordinator._registration,
            session_store=store,
            enable_common_root_budget_binding=True,
            budget_binding_receiver=application.budget_binding_receiver,
            collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=300_000),
        )
        await application.initialize_collaboration()
        disclosure_lock = asyncio.Lock()
        guarded_dispatches = []
        timed_dispatches = []
        provider_cancelled = asyncio.Event()

        @asynccontextmanager
        async def admission_guard(dispatch):
            async with disclosure_lock:
                guarded_dispatches.append(dispatch)
                yield dispatch.intent.question.deadline_at_ms

        class GuardedProvider(ScriptedModelProvider):
            async def stream(self, request):
                # Model exposure needs the same non-reentrant policy lock. The
                # registration guard must have ended before this real dispatch.
                async with disclosure_lock:
                    if registration_failure == "timeout" and len(self.requests) == 1 + side_session:
                        timed_dispatches.append(request)
                        try:
                            await asyncio.Future()
                        except asyncio.CancelledError:
                            provider_cancelled.set()
                            raise
                    async for event in super().stream(request):
                        yield event

        provider = GuardedProvider(
            [
                [
                    ModelStreamEvent.text_delta("parent complete"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("clarification answer"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
            + (
                [
                    [
                        ModelStreamEvent.text_delta("side clarification answer"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ]
                if side_session
                else []
            ),
            name="provider",
        )
        application.register_provider(provider, default=True)
        application.register_agent(AgentSpec(name="reviewer", model="model"))
        participant = template.dispatch.intent.question.responder
        creation = ParticipantSessionCreationRequest(
            creation_key="runtime-service-source:" + initial.binding.application_scope,
            request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "start")]),
        )
        session, _ = await application.create_participant_session(
            creation, participant=participant, context=CONTEXT
        )
        owner = SessionContinuationOwner(
            store=store,
            owner=initial.owner,
            receiver=application.collaboration_wait_latch_receiver(),
            receiver_capability=CapabilityDescriptor(
                owner=initial.owner, mutations=(), readbacks=(LATCH_FAMILY,)
            ),
            redactor=SecretRedactor(),
            temporary_permits=TemporaryServicePermitAuthority(
                source, initial, redactor=SecretRedactor(), admission_guard=admission_guard
            ),
        )
        assert owner.temporary_permits is not None
        # This witness observes a complete engine turn, not only one mutation.
        # Keep a finite bound matching its explicit service policy; observation
        # expiry is tested separately and is not evidence of execution failure.
        owner.owners.observation_timeout = (
            180
            if registration_failure == "timeout"
            else template.dispatch.intent.question.policy.service_timeout_ms / 1000
        )
        # Use the real registered wait owner to bind the execution commitment;
        # a directly constructed private handoff cannot confer that authority.
        wait = wait_for(opening.command, initial)
        stream = application.execute_participant_session_to_wait(
            ParticipantSessionExecutionRequest(
                request=creation.request.model_copy(update={"session_id": session.id}),
                session_instance_id=session.instance_id,
                execution_key="parent-execution",
            ),
            wait,
            participant=participant,
            context=CONTEXT,
            wait_context=resolver.context,
        )
        try:
            events = [event async for event in stream]
        finally:
            await stream.aclose()
        assert any(event.type == EventType.SESSION_INTERRUPTED for event in events)
        parked = await store.load_continuation_ticket(
            session.id,
            session_instance_id=session.instance_id,
            registration_key="execution-wait:" + continuation_digest(wait.operation),
        )
        assert parked is not None
        assert parked.ticket.execution_admission_sha256 is not None
        assert parked.ticket.collaboration_wait_sha256 == continuation_digest(wait)
        source_session = session
        if side_session:
            side_creation = ParticipantSessionCreationRequest(
                creation_key=creation.creation_key + ":side", request=creation.request
            )
            session, _ = await application.create_participant_session(
                side_creation, participant=participant, context=CONTEXT
            )
            [
                event
                async for event in application.execute_participant_session(
                    ParticipantSessionExecutionRequest(
                        request=side_creation.request.model_copy(update={"session_id": session.id}),
                        session_instance_id=session.instance_id,
                        execution_key="side-initial-execution",
                    ),
                    participant=participant,
                    context=CONTEXT,
                )
            ]
        initial_requests = 1 + side_session
        checkpoint = await runtime_checkpoint_session_store(store).load_checkpoint(session.id)
        active = active_invocation_execution_profile_from_checkpoint(checkpoint)
        assert active is not None
        binding = await store.load_participant_session_binding(session.id)
        assert binding is not None
        resume = ResumeRequest(
            session_id=session.id, messages=[Message.text("user", "please clarify")]
        )
        before = await store.load(session.id)
        assert before is not None
        before_events = await store.load_events(session.id)
        assert len(provider.requests) == initial_requests
        with pytest.raises(PermissionError):
            [event async for event in application.resume(resume)]
        assert await store.load(session.id) == before
        assert await store.load_events(session.id) == before_events
        assert len(provider.requests) == initial_requests
        intent = template.dispatch.intent.model_copy(
            update={
                "ticket": parked.ticket,
                "mode": "side_session" if side_session else "same_session",
                "target": template.dispatch.intent.target.model_copy(
                    update={"object_id": session.id, "incarnation": session.instance_id}
                ),
                "participant_binding_sha256": continuation_digest(binding),
                "execution_profile_sha256": active.profile.fingerprint,
                "resume_sha256": application._session_engine.work_attempt_source_request_sha256(
                    resume, kind="continuation"
                ),
                "prepared_at_ms": int(datetime.now(UTC).timestamp() * 1000),
            }
        )
        if registration_failure in {"before_commit", "after_commit"}:
            original = TemporaryServicePermitAuthority.register

            async def commit_then_lose_ack(self, candidate):
                if registration_failure == "after_commit":
                    await original(self, candidate)
                raise OSError("service registration acknowledgement lost")

            monkeypatch.setattr(TemporaryServicePermitAuthority, "register", commit_then_lose_ack)
            with pytest.raises(ContinuationUnavailable):
                await owner.service_temporary(
                    application, resume, intent, participant_context=CONTEXT
                )
            monkeypatch.setattr(TemporaryServicePermitAuthority, "register", original)
            registered = await owner.temporary_permits.lookup(intent)
            assert (registered is not None) == (registration_failure == "after_commit")
            prepared_record = TemporaryServiceRecord.model_validate(
                await store.load_session_operation(
                    source_session.id, temporary_service_key(intent.operation)
                )
            )
            assert prepared_record.state == "prepared"
            after_failure = await store.load(session.id)
            assert after_failure is not None
            after_failure_events = await store.load_events(session.id)
            with pytest.raises(ContinuationUnavailable):
                await owner.service_temporary(
                    application, resume, intent, participant_context=CONTEXT
                )
            assert await store.load(session.id) == after_failure
            assert await store.load_events(session.id) == after_failure_events
            assert len(provider.requests) == initial_requests
            excluded = await owner.exclude_temporary(prepared_record.admission)
            assert excluded.state == "excluded"
            assert excluded.settlement is not None
            assert excluded.settlement.admission_excluded
            assert await owner.exclude_temporary(prepared_record.admission) == excluded
            assert (
                await owner.service_temporary(
                    application, resume, intent, participant_context=CONTEXT
                )
                == excluded
            )
            registered = await owner.temporary_permits.lookup(intent)
            if registration_failure == "after_commit":
                assert registered is not None
                assert registered.state == "settled"
            else:
                assert registered is None
            assert len(provider.requests) == initial_requests
            replayed_session = await store.load(session.id)
            assert replayed_session is not None
            assert replayed_session.run_epoch == after_failure.run_epoch
            return
        registered_bindings = []
        if registration_failure == "changed_budget":
            application.budget_binding_receiver.binding = budget_binding.model_copy(
                update={"binding_id": budget_binding.binding_id + ":replacement"}
            )
            register_binding = application.budget_ledger.register_budget_binding

            async def record_registration(**kwargs):
                registered_bindings.append(kwargs)
                return await register_binding(**kwargs)

            monkeypatch.setattr(
                application.budget_ledger, "register_budget_binding", record_registration
            )
        if registration_failure == "timeout":
            with pytest.raises(ContinuationUnavailable):
                await owner.service_temporary(
                    application, resume, intent, participant_context=CONTEXT
                )
            # Public observation can finish while owned runtime cleanup drains.
            await asyncio.wait_for(provider_cancelled.wait(), 60)
            assert len(timed_dispatches) == 1
            retained = TemporaryServiceRecord.model_validate(
                await store.load_session_operation(
                    source_session.id, temporary_service_key(intent.operation)
                )
            )
            assert retained.state == "admitted"
            # Cancellation is not an exclusion. Exact retry reconciles that
            # admission; it may return only if native quiescence is established.
            try:
                replay = await owner.service_temporary(
                    application, resume, intent, participant_context=CONTEXT
                )
            except ContinuationUnavailable:
                pass
            else:
                assert replay.state == "returned"
            assert len(timed_dispatches) == 1
            timed_out_session = await store.load(session.id)
            assert timed_out_session is not None
            assert timed_out_session.execution_deadline == before.execution_deadline
            return
        returned = await owner.service_temporary(
            application, resume, intent, participant_context=CONTEXT
        )
        if registration_failure == "changed_budget":
            assert returned.state == "returned"
            assert returned.released_session_status == "failed"
            assert len(provider.requests) == initial_requests
            assert registered_bindings == []
            failed = [
                event
                for event in await store.load_events(session.id)
                if event.type == EventType.SESSION_FAILED
            ]
            assert len(failed) == 1
            assert failed[0].payload["error_type"] == "BudgetBindingError"
            assert "common-root budget authority conflicts" in failed[0].payload["error"]
            assert (
                await owner.service_temporary(
                    application, resume, intent, participant_context=CONTEXT
                )
                == returned
            )
            assert len(provider.requests) == initial_requests
            return
        if len(provider.requests) != initial_requests + 1:
            pytest.fail(
                "Expected one service dispatch; durable tail:\n"
                + "\n".join(
                    f"{event.type}: {event.payload!r}"
                    for event in (await store.load_events(session.id))[-12:]
                )
            )
        assert len(guarded_dispatches) == 1 and not disclosure_lock.locked()
        retained = TemporaryServiceRecord.model_validate(
            await store.load_session_operation(
                source_session.id, temporary_service_key(intent.operation)
            )
        )
        assert isinstance(retained.admission, TemporaryServiceAdmission)
        returned = await owner.reconcile_temporary(retained.admission)
        assert returned.state == "returned"
        assert returned.intent.ticket == parked.ticket
        from cayu.collaboration._clarification_reply_production import authenticate_reply_production
        from cayu.collaboration._contracts import CollaborationConflict
        from cayu.collaboration._session_export_store import source_digest
        from cayu.sessions._model_completion_publication import (
            model_step_publication_from_checkpoint,
        )

        completed_checkpoint = await runtime_checkpoint_session_store(store).load_checkpoint(
            session.id
        )
        pointer = model_step_publication_from_checkpoint(completed_checkpoint)
        assert pointer is not None and pointer.assistant_message_published
        production = await authenticate_reply_production(
            store,
            returned,
            stage_id=pointer.stage_id,
            source_indices=(pointer.source_transcript_cursor,),
        )
        page = await store.load_transcript_window(
            session.id, start_index=pointer.source_transcript_cursor, limit=1
        )
        visible_rows = [
            row.model_copy(
                update={
                    "message": row.message.model_copy(
                        update={
                            "content": tuple(
                                part for part in row.message.content if part.type == "text"
                            )
                        }
                    )
                }
            )
            for row in page.records
        ]
        assert production.source_commitment == source_digest(visible_rows)
        assert production.invocation_id == intent.invocation_id
        load_stage = store.load_model_completion_stage
        for purpose in ("context-compaction", "auxiliary-inference"):

            async def wrong_purpose(session_id, stage_id, *, purpose=purpose):
                stage = await load_stage(session_id, stage_id)
                assert stage is not None
                return stage.model_copy(update={"purpose": purpose})

            with monkeypatch.context() as patch:
                patch.setattr(store, "load_model_completion_stage", wrong_purpose)
                with pytest.raises(CollaborationConflict):
                    await authenticate_reply_production(
                        store,
                        returned,
                        stage_id=pointer.stage_id,
                        source_indices=(pointer.source_transcript_cursor,),
                    )
        for stage_id, indices in (
            ("missing-stage", (pointer.source_transcript_cursor,)),
            (pointer.stage_id, (0,)),
        ):
            with pytest.raises(CollaborationConflict):
                await authenticate_reply_production(
                    store, returned, stage_id=stage_id, source_indices=indices
                )
        after = await store.load(session.id)
        after_events = await store.load_events(session.id)
        assert (
            await owner.service_temporary(application, resume, intent, participant_context=CONTEXT)
            == returned
        )
        with pytest.raises(ContinuationConflict):
            await owner.service_temporary(
                application,
                resume.model_copy(update={"messages": [Message.text("user", "different input")]}),
                intent,
                participant_context=CONTEXT,
            )
        with pytest.raises(ContinuationConflict):
            await owner.service_temporary(
                application,
                resume,
                intent.model_copy(update={"prepared_at_ms": intent.prepared_at_ms + 1}),
                participant_context=CONTEXT,
            )
        with pytest.raises(ContinuationConflict):
            await owner.service_temporary(
                application,
                resume.model_copy(update={"retry_policy": None}),
                intent,
                participant_context=CONTEXT,
            )
        assert await store.load(session.id) == after
        assert await store.load_events(session.id) == after_events
        assert len(provider.requests) == initial_requests + 1
        if side_session:
            await store.delete_session(session.id)
            assert (
                await owner.service_temporary(
                    application, resume, intent, participant_context=CONTEXT
                )
                == returned
            )
            replacement, _ = await application.create_participant_session(
                ParticipantSessionCreationRequest(
                    creation_key=creation.creation_key + ":replacement",
                    request=creation.request.model_copy(update={"session_id": session.id}),
                ),
                participant=participant,
                context=CONTEXT,
            )
            assert replacement.instance_id != session.instance_id
            replacement_before = await store.load(replacement.id)
            assert (
                await owner.service_temporary(
                    application, resume, intent, participant_context=CONTEXT
                )
                == returned
            )
            assert await store.load(replacement.id) == replacement_before
            assert len(provider.requests) == initial_requests + 1
    finally:
        if not isinstance(store, InMemorySessionStore):
            await store.close()
