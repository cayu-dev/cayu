from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import warnings
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from tests.core.test_completion_result_resolvers import _Resolver
from tests.core.test_completion_verifier_adapters import (
    RecordingVerifier,
    _accepted_decision,
    _contract,
    _rejected_decision,
)
from tests.core.test_provider_operation_offline_recovery import (
    _OfflineOperationAdapter,
    _OfflineOperationProvider,
)
from tests.core.test_verified_work_contracts import (
    _assert_secret_absent_from_cayu_error,
    _claim_completion_verification,
    _RecordingProvider,
    _result_reference,
    _task_result,
    _verifier_profile_fingerprint,
)
from tests.core.verified_worker_fixtures import (
    VerifiedWorkerStoreFactory,
    wait_for_verified_worker_lease_expiry,
)
from tests.core.verified_worker_fixtures import (
    verified_work_postgres_dsn as verified_work_postgres_dsn,
)
from tests.core.verified_worker_fixtures import (
    verified_worker_store_factory as verified_worker_store_factory,
)

from cayu import (
    AgentSpec,
    BudgetLimit,
    BudgetPolicy,
    CayuApp,
    CompletionContinuationPolicy,
    CompletionDecisionCreate,
    CompletionProposalCreate,
    CompletionRejectionAction,
    CompletionVerdict,
    CompletionVerificationClaimRequest,
    CompletionVerifierExecutionError,
    CompletionVerifierExecutionRequest,
    EventType,
    InMemorySessionStore,
    InMemoryTaskStore,
    Message,
    ModelPrice,
    PriceBook,
    RunRequest,
    SessionStatus,
    SQLiteSessionStore,
    SQLiteTaskStore,
    TaskCreate,
    TaskQuery,
    TaskStatus,
    TaskStore,
    WorkCompletionConflict,
)
from cayu._exception_groups import iter_exception_tree
from cayu.core.events import Event
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.tools import Tool, ToolEffect, ToolResult, ToolSpec
from cayu.deadlines import ExecutionDeadlineExceeded
from cayu.environments import Environment, EnvironmentSpec
from cayu.providers import (
    ModelStreamEvent,
    ProviderOperationConnection,
    ProviderOperationSnapshot,
    ProviderOperationStartIdempotencySupport,
    ProviderOperationStatus,
)
from cayu.runtime import EventQuery
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime._recovery_coordinator import RecoveryCoordinator
from cayu.runtime.hooks import RuntimeHook
from cayu.runtime.loop_policies import BeforeStopDecision, LoopPolicy
from cayu.runtime.verified_task_worker import (
    VerifiedTaskHandler,
    VerifiedTaskHandlerReport,
    VerifiedTaskWorker,
    VerifiedTaskWorkerDraining,
)
from cayu.runtime.work_attempt_admission import (
    WorkAttemptAdmissionConflict,
    WorkAttemptExecutionClaimLost,
    WorkAttemptExecutionRequest,
    WorkAttemptProposalRequest,
    WorkAttemptRecoveryRequest,
    WorkAttemptRecoveryRequired,
    WorkAttemptRunRequest,
)
from cayu.vaults import SecretRef, StaticVault

# Process-loss recovery includes schema validation, durable replay, cleanup,
# verification and settlement. This is a harness deadlock guard, not a runtime
# deadline contract; ten seconds cancelled healthy progress on loaded CI hosts.
_RECOVERY_COMPLETION_TIMEOUT_SECONDS = 60


class _ContinueOnceVerifier(RecordingVerifier):
    async def verify(self, request):
        self.requests.append(request)
        return self.decision if request.attempt.ordinal == 1 else _accepted_decision()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_documented_reference_handler_completes_through_public_worker(backend, tmp_path):
    from examples.verified_task_handler import ReferencedResultHandler

    from cayu import VerifiedTaskWorker as PublicWorker

    async def scenario():
        sessions = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.sqlite")
        )
        tasks = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
        )
        try:
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            provider = _RecordingProvider()
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            contract = _contract()
            await tasks.publish_work_contract(contract)
            task = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            verifier = RecordingVerifier(_accepted_decision())
            resolver = _Resolver(_task_result())
            app.register_completion_verifier(contract.verifier, verifier)
            app.register_completion_result_resolver(contract.result_resolver, resolver)
            candidates = []

            async def candidate(context):
                assert context.task.id == task.id
                assert context.contract == contract
                candidates.append(context)
                return _result_reference()

            handler = ReferencedResultHandler("worker", candidate)
            async with PublicWorker(app, handler, worker_id="documented-worker") as worker:
                assert await asyncio.wait_for(worker.run(max_tasks=1), 15) == 1
            final = await tasks.load_task(task.id)
            assert final.status is TaskStatus.COMPLETED
            assert (
                len(provider.requests)
                == len(candidates)
                == len(verifier.requests)
                == len(resolver.requests)
                == 1
            )
            admission = await tasks.load_latest_work_attempt_admission(task.id)
            proposal = await tasks.load_completion_proposal_for_attempt(admission.attempt_id)
            assert proposal.proposal_id == candidates[0].proposal_id
            assert proposal.result == _result_reference()
            receipt = await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
            assert receipt.task == final and receipt.retired_contract_binding
        finally:
            if backend == "sqlite":
                await sessions.close()
                await tasks.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("replacement_lease,cancel_recovery", [(1, False), (5, False), (5, True)])
def test_worker_renews_during_unentered_attempt_recovery(
    backend, replacement_lease, cancel_recovery, tmp_path
):
    from tests.core.test_work_attempt_admission import (
        _BlockFirstSQLiteWorkAttemptSettlementFence,
        _BlockFirstWorkAttemptSettlementFence,
    )

    async def scenario():
        sessions = (
            _BlockFirstWorkAttemptSettlementFence()
            if backend == "memory"
            else _BlockFirstSQLiteWorkAttemptSettlementFence(tmp_path / "sessions.sqlite")
        )
        tasks = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
        )
        running = None
        release_fence = sessions.release_fence
        try:
            source = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            source.register_provider(_RecordingProvider(), default=True)
            source.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            contract = _contract()
            await tasks.publish_work_contract(contract)
            task = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            admission = await source.admit_work_attempt(
                RunRequest(
                    agent_name="worker",
                    task_id=task.id,
                    session_id="unentered-session",
                    messages=[Message.text("user", "Preserve this admitted input.")],
                ),
                execution=WorkAttemptExecutionRequest(
                    admission_id="unentered-admission",
                    claim_id="unentered-claim",
                    attempt_id="unentered-attempt",
                    interaction_id="unentered-interaction",
                    worker_id="original-worker",
                    generation=1,
                    lease_seconds=1,
                ),
            )
            assert admission.execution_entry is None
            await asyncio.sleep(
                max(0, (admission.claim.lease_expires_at - datetime.now(UTC)).total_seconds())
                + 0.05
            )
            replacement = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            provider = _RecordingProvider()
            replacement.register_provider(provider, default=True)
            replacement.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            verifier = RecordingVerifier(_accepted_decision())
            replacement.register_completion_verifier(contract.verifier, verifier)
            replacement.register_completion_result_resolver(
                contract.result_resolver, _Resolver(_task_result())
            )
            handler = _StaticHandler()
            expired_recovery = False
            async with VerifiedTaskWorker(
                replacement,
                handler,
                worker_id="replacement",
                lease_seconds=replacement_lease,
                callback_timeout_seconds=0.5,
            ) as worker:
                running = asyncio.create_task(worker.run(max_tasks=1))
                await asyncio.wait_for(sessions.fence_committed.wait(), 5)
                claimed = await tasks.load_work_attempt_admission(admission.admission_id)
                assert claimed.state.value == "recovering"
                assert claimed.claim.generation == 2
                await asyncio.sleep(
                    max(0, (claimed.claim.lease_expires_at - datetime.now(UTC)).total_seconds())
                    + 0.15
                )
                renewed = await tasks.load_work_attempt_admission(admission.admission_id)
                assert renewed.claim.lease_expires_at > claimed.claim.lease_expires_at
                assert renewed.claim.renewal is not None
                assert not running.done()
                with pytest.raises(WorkAttemptExecutionClaimLost):
                    await source.recover_work_attempt(
                        WorkAttemptRecoveryRequest(
                            admission_id=admission.admission_id,
                            claim_id="competitor",
                            worker_id="competitor",
                            generation=3,
                            lease_seconds=10,
                        )
                    )
                if cancel_recovery:
                    running.cancel("cancel recovery owner")
                    with pytest.raises(asyncio.CancelledError) as cancelled:
                        await running
                    assert cancelled.value.args == ("cancel recovery owner",)
                    assert running.cancelling() == 1 and running.cancelled()
                    assert provider.requests == handler.proposals == verifier.requests == []
                sessions.release_fence.set()
                if cancel_recovery:
                    if worker._running is not None:
                        done, _ = await asyncio.wait({worker._running}, timeout=10)
                        assert done
                    await worker.aclose()
                    expired_recovery = True
                try:
                    if not cancel_recovery:
                        assert await asyncio.wait_for(running, 10) == 1
                except (WorkAttemptExecutionClaimLost, ExceptionGroup) as error:
                    # A one-second lease may expire while the runtime performs
                    # synchronous checkpoint validation. Success is not the
                    # contract once ownership expires: reject stale effects
                    # and leave the same attempt recoverable by its next owner.
                    assert replacement_lease == 1
                    leaves = [
                        item
                        for item in iter_exception_tree(error)
                        if not isinstance(item, BaseExceptionGroup)
                    ]
                    assert leaves and all(
                        isinstance(item, WorkAttemptExecutionClaimLost) for item in leaves
                    )
                    expired_recovery = True
            if expired_recovery:
                stale = await tasks.load_work_attempt_admission(admission.admission_id)
                assert stale.claim.generation == 2
                assert stale.execution_entry is None
                assert provider.requests == []
                assert handler.proposals == verifier.requests == []
                assert (
                    await tasks.load_completion_proposal_for_attempt(admission.attempt_id) is None
                )
                assert (await tasks.load_task(task.id)).status is TaskStatus.RUNNING
                await asyncio.sleep(
                    max(0, (stale.claim.lease_expires_at - datetime.now(UTC)).total_seconds())
                    + 0.05
                )
                if backend == "sqlite":
                    await tasks.close()
                    await sessions.close()
                    tasks = SQLiteTaskStore(tmp_path / "tasks.sqlite")
                    sessions = SQLiteSessionStore(tmp_path / "sessions.sqlite")
                successor = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
                successor.register_provider(provider, default=True)
                successor.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
                successor.register_completion_verifier(contract.verifier, verifier)
                successor.register_completion_result_resolver(
                    contract.result_resolver, _Resolver(_task_result())
                )
                async with VerifiedTaskWorker(
                    successor, handler, worker_id="next-owner"
                ) as next_worker:
                    assert await asyncio.wait_for(next_worker.run(max_tasks=1), 10) == 1
            assert handler.preparations == []
            assert len(handler.proposals) == len(verifier.requests) == 1
            assert len(provider.requests) == 1
            final = await tasks.load_work_attempt_admission(admission.admission_id)
            assert final.run_semantics == admission.run_semantics
            assert (
                final.source_execution_profile_fingerprint
                == admission.source_execution_profile_fingerprint
            )
            assert (await tasks.load_task(task.id)).status is TaskStatus.COMPLETED
        finally:
            release_fence.set()
            if running is not None and not running.done():
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
            if backend == "sqlite":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())


class _StaticHandler(VerifiedTaskHandler):
    def __init__(self):
        self.preparations = []
        self.proposals = []

    async def prepare(self, context):
        self.preparations.append(context)
        return RunRequest(agent_name="worker", messages=[Message.text("user", "Build the result.")])

    async def propose(self, context):
        self.proposals.append(context)
        return VerifiedTaskHandlerReport(
            proposal=CompletionProposalCreate(
                proposal_id=context.proposal_id,
                attempt_id=context.attempt.attempt_id,
                result=_result_reference(),
            )
        )


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("delivery_mode", ["next_turn", "on_idle"])
@pytest.mark.parametrize("entrance", ["sdk", "http", "sibling_before", "sibling_during"])
def test_worker_fences_queued_steering_before_input_or_interaction_publication(
    backend, delivery_mode, entrance, verified_worker_store_factory, monkeypatch
):
    from cayu import EnqueueSessionMessageRequest, SessionMessageQuery
    from cayu.runtime.work_contracts import TaskCompletionDecisionRequired

    async def scenario():
        sessions, tasks = verified_worker_store_factory()
        entered, release = asyncio.Event(), asyncio.Event()
        running = None
        policy = None
        if entrance == "http":
            from tests.server.test_session_message_lifecycle_api import OwnershipPolicy

            policy = OwnershipPolicy()
        try:
            app = CayuApp(
                session_store=sessions,
                task_store=tasks,
                session_message_access_policy=policy,
                enable_logging=False,
            )
            sibling = CayuApp(session_store=sessions, enable_logging=False)
            provider = _RecordingProvider()

            async def stream(request):
                provider.requests.append(request)
                entered.set()
                await release.wait()
                yield ModelStreamEvent.text_delta("Done.")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})

            monkeypatch.setattr(provider, "stream", stream)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            contract = _contract()
            await tasks.publish_work_contract(contract)
            task = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            verifier = RecordingVerifier(_accepted_decision())
            app.register_completion_verifier(contract.verifier, verifier)
            app.register_completion_result_resolver(
                contract.result_resolver, _Resolver(_task_result())
            )
            handler = _StaticHandler()
            content = "Steering must not enter this admitted attempt."

            def request_for(session_id):
                return EnqueueSessionMessageRequest(
                    session_id=session_id,
                    idempotency_key="governed-steering",
                    content=content,
                    delivery_mode=delivery_mode,
                )

            if entrance == "sibling_before":
                execute = app._execute_work_attempt

                async def enqueue_before_execution(request):
                    admission = await tasks.load_work_attempt_admission(request.admission_id)
                    await sibling.enqueue_session_message(request_for(admission.session_id))
                    async for event in execute(request):
                        yield event

                monkeypatch.setattr(app, "_execute_work_attempt", enqueue_before_execution)
            async with VerifiedTaskWorker(app, handler, worker_id="steering-owner") as worker:
                running = asyncio.create_task(worker.run(max_tasks=1))
                await asyncio.wait_for(entered.wait(), 15)
                admission = await tasks.load_latest_work_attempt_admission(task.id)
                session_id = admission.session_id
                before_events = await sessions.query_events(EventQuery(session_id=session_id))
                before_transcript = await sessions.load_transcript(session_id)
                if entrance == "sdk":
                    with pytest.raises(TaskCompletionDecisionRequired):
                        await app.enqueue_session_message(request_for(session_id))
                elif entrance == "http":
                    from httpx import ASGITransport, AsyncClient
                    from tests.server.test_session_message_lifecycle_api import (
                        HEADERS,
                        authenticate,
                    )

                    from cayu.server import ServerConfig, create_server

                    session = await sessions.load(session_id)
                    policy.grants.add(
                        ("alice", "tenant-a", session_id, session.instance_id, "enqueue")
                    )
                    server = create_server(app, config=ServerConfig.protected(authenticate))
                    async with AsyncClient(
                        transport=ASGITransport(app=server), base_url="http://test"
                    ) as client:
                        response = await client.post(
                            f"/api/sessions/{session_id}/messages",
                            headers=HEADERS,
                            json={
                                "idempotency_key": "governed-steering",
                                "content": content,
                                "delivery_mode": delivery_mode,
                            },
                        )
                    assert response.status_code == 409, response.text
                elif entrance == "sibling_during":
                    await sibling.enqueue_session_message(request_for(session_id))
                if entrance in {"sdk", "http"}:
                    assert (
                        await sessions.query_events(EventQuery(session_id=session_id))
                        == before_events
                    )
                    assert await sessions.load_transcript(session_id) == before_transcript
                    assert not (
                        await sessions.inspect_session_messages(
                            SessionMessageQuery(session_id=session_id)
                        )
                    ).records
                release.set()
                assert await asyncio.wait_for(running, 15) == 1
            final = await tasks.load_task(task.id)
            receipt = await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
            assert receipt.task == final
            assert final.worker_id is None and final.lease_expires_at is None
            assert len(provider.requests) == 1
            assert all(
                content not in message.model_dump_json()
                for message in provider.requests[0].messages
            )
            events = [
                record.event
                for record in await sessions.query_events(EventQuery(session_id=session_id))
            ]
            assert sum(event.type is EventType.INTERACTION_STARTED for event in events) == 1
            assert not any(event.type is EventType.SESSION_MESSAGE_DELIVERED for event in events)
            if entrance.startswith("sibling"):
                assert final.status is TaskStatus.NEEDS_ATTENTION
                assert final.status_reason == "work_contract_execution_failed"
                assert not receipt.retired_contract_binding
                assert not handler.proposals and not verifier.requests
                page = await sessions.inspect_session_messages(
                    SessionMessageQuery(session_id=session_id)
                )
                assert len(page.records) == 1 and page.records[0].status.value == "queued"
                assert page.records[0].message.content == content
            else:
                assert final.status is TaskStatus.COMPLETED
                assert receipt.retired_contract_binding
                assert len(handler.proposals) == len(verifier.requests) == 1
        finally:
            release.set()
            if running is not None and not running.done():
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
            if backend != "memory":
                await sessions.close()
                await tasks.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("delivery_mode", ["next_turn", "on_idle"])
def test_worker_retirement_restores_ordinary_queued_steering(
    backend, delivery_mode, verified_worker_store_factory, monkeypatch
):
    from cayu import EnqueueSessionMessageRequest, ResumeRequest, SessionMessageQuery

    async def scenario():
        sessions, tasks = verified_worker_store_factory()
        entered, release = asyncio.Event(), asyncio.Event()
        release.set()
        running = None
        try:
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            provider = _RecordingProvider()

            async def stream(request):
                provider.requests.append(request)
                entered.set()
                await release.wait()
                yield ModelStreamEvent.text_delta("Done.")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})

            monkeypatch.setattr(provider, "stream", stream)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            contract = _contract()
            await tasks.publish_work_contract(contract)
            task = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            verifier = RecordingVerifier(_accepted_decision())
            app.register_completion_verifier(contract.verifier, verifier)
            app.register_completion_result_resolver(
                contract.result_resolver, _Resolver(_task_result())
            )
            handler = _StaticHandler()
            async with VerifiedTaskWorker(app, handler, worker_id="retiring-owner") as worker:
                assert await asyncio.wait_for(worker.run(max_tasks=1), 15) == 1
            completed = await tasks.load_task(task.id)
            admission = await tasks.load_latest_work_attempt_admission(task.id)
            receipt = await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
            assert completed.status is TaskStatus.COMPLETED and receipt.retired_contract_binding
            session_id = admission.session_id
            assert await tasks.load_active_work_contract_task_for_session(session_id) is None
            entered.clear()
            release.clear()

            async def resume():
                return [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id=session_id,
                            messages=[Message.text("user", "Ordinary follow-up.")],
                        )
                    )
                ]

            running = asyncio.create_task(resume())
            await asyncio.wait_for(entered.wait(), 15)
            accepted = await app.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=session_id,
                    idempotency_key="after-retirement",
                    content="Ordinary queued follow-up.",
                    delivery_mode=delivery_mode,
                )
            )
            assert not accepted.replayed
            release.set()
            events = await asyncio.wait_for(running, 15)
            assert any(event.type is EventType.SESSION_COMPLETED for event in events)
            assert len(provider.requests) == 3
            assert any(
                "Ordinary queued follow-up." in message.model_dump_json()
                for message in provider.requests[-1].messages
            )
            page = await sessions.inspect_session_messages(
                SessionMessageQuery(session_id=session_id)
            )
            assert len(page.records) == 1 and page.records[0].status.value == "delivered"
            assert await tasks.load_task(task.id) == completed
            assert len(handler.proposals) == len(verifier.requests) == 1
            assert (
                await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id) == receipt
            )
        finally:
            release.set()
            if running is not None and not running.done():
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
            if backend != "memory":
                await sessions.close()
                await tasks.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("pause_kind", ["user_input", "tool_approval"])
def test_worker_preserves_human_input_pause_without_proposing_completion(
    backend, pause_kind, tmp_path, verified_worker_store_factory
):
    from cayu import AlwaysRequireApprovalToolPolicy
    from cayu.tools.user_input import UserInputTool

    class InputProvider(_RecordingProvider):
        async def stream(self, request):
            self.requests.append(request)
            yield ModelStreamEvent.tool_call(
                id="human-input",
                name="ask_user" if pause_kind == "user_input" else "record_effect",
                arguments={"question": "Which environment?"} if pause_kind == "user_input" else {},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})

    async def scenario():
        sessions, tasks = verified_worker_store_factory()
        try:
            app = CayuApp(
                session_store=sessions,
                task_store=tasks,
                enable_logging=False,
            )
            provider = InputProvider()
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="worker", model="verified-work-test-model"),
                tools=[UserInputTool()]
                if pause_kind == "user_input"
                else [_RecordedExternalEffect(tmp_path / "unapproved-effect")],
                **(
                    {"tool_policy": AlwaysRequireApprovalToolPolicy()}
                    if pause_kind == "tool_approval"
                    else {}
                ),
            )
            contract = _contract()
            await tasks.publish_work_contract(contract)
            task = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            verifier = RecordingVerifier(_accepted_decision())
            app.register_completion_verifier(contract.verifier, verifier)
            app.register_completion_result_resolver(
                contract.result_resolver, _Resolver(_task_result())
            )
            handler = _StaticHandler()
            async with VerifiedTaskWorker(app, handler, worker_id="human-owner") as worker:
                assert await asyncio.wait_for(worker.run(max_tasks=1), 10) == 1
            held = await tasks.load_task(task.id)
            assert held.status is TaskStatus.NEEDS_ATTENTION
            assert held.status_reason == "work_contract_execution_interrupted"
            assert held.work_contract == contract.reference()
            assert held.worker_id is None and held.lease_expires_at is None
            session = await sessions.load(held.session_id)
            assert session.status is SessionStatus.INTERRUPTED
            checkpoint = await sessions.load_checkpoint(session.id)
            assert f"pending_{pause_kind}" in checkpoint
            assert not (tmp_path / "unapproved-effect").exists()
            assert len(provider.requests) == len(handler.preparations) == 1
            assert not handler.proposals and not verifier.requests
            assert await tasks.list_unsettled_work_attempt_admissions(limit=1) == []
        finally:
            if backend != "memory":
                await sessions.close()
                await tasks.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("strategy", ["native", "tool"])
@pytest.mark.parametrize("outcome", ["valid", "invalid", "repair"])
def test_worker_requires_structured_validation_before_proposal(
    backend, strategy, outcome, verified_worker_store_factory
):
    from cayu import StructuredOutputSpec
    from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME

    specification = StructuredOutputSpec(
        strategy=strategy,
        max_retries=1 if outcome == "repair" else 0,
        json_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )

    class StructuredProvider(_RecordingProvider):
        supports_native_structured_output = True

        async def stream(self, request):
            self.requests.append(request)
            valid = outcome == "valid" or (outcome == "repair" and len(self.requests) == 2)
            if strategy == "native":
                yield ModelStreamEvent.text_delta('{"answer":"ok"}' if valid else '{"wrong":true}')
            else:
                yield ModelStreamEvent.tool_call(
                    id=f"structured-{len(self.requests)}",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "ok"} if valid else {"wrong": True}},
                )
            yield ModelStreamEvent.completed(
                {"finish_reason": "stop" if strategy == "native" else "tool_calls"}
            )

    class Handler(_StaticHandler):
        async def prepare(self, context):
            request = await super().prepare(context)
            return request.model_copy(update={"structured_output": specification})

    async def scenario():
        sessions, tasks = verified_worker_store_factory()
        try:
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            provider = StructuredProvider()
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            contract = _contract()
            await tasks.publish_work_contract(contract)
            task = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            verifier = RecordingVerifier(_accepted_decision())
            resolver = _Resolver(_task_result())
            app.register_completion_verifier(contract.verifier, verifier)
            app.register_completion_result_resolver(contract.result_resolver, resolver)
            handler = Handler()
            async with VerifiedTaskWorker(app, handler, worker_id="structured-worker") as worker:
                assert await asyncio.wait_for(worker.run(max_tasks=1), 15) == 1
            final = await tasks.load_task(task.id)
            succeeds = outcome != "invalid"
            assert final.status is (
                TaskStatus.COMPLETED if succeeds else TaskStatus.NEEDS_ATTENTION
            )
            assert len(provider.requests) == (2 if outcome == "repair" else 1)
            assert (
                len(handler.proposals)
                == len(verifier.requests)
                == len(resolver.requests)
                == int(succeeds)
            )
            admission = await tasks.load_latest_work_attempt_admission(task.id)
            assert admission.run_semantics.structured_output == specification
            events = await sessions.query_events(
                EventQuery(
                    session_id=admission.session_id,
                    event_types={
                        EventType.STRUCTURED_OUTPUT_VALIDATED,
                        EventType.STRUCTURED_OUTPUT_FAILED,
                    },
                    limit=10,
                )
            )
            assert sum(
                record.event.type is EventType.STRUCTURED_OUTPUT_VALIDATED for record in events
            ) == int(succeeds)
            assert sum(
                record.event.type is EventType.STRUCTURED_OUTPUT_FAILED for record in events
            ) == int(outcome != "valid")
            if not succeeds:
                assert final.status_reason == "work_contract_execution_failed"
                assert (
                    await tasks.load_completion_proposal_for_attempt(admission.attempt_id) is None
                )
            receipt = await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
            assert receipt.task == final and receipt.retired_contract_binding is succeeds
        finally:
            if backend != "memory":
                await sessions.close()
                await tasks.close()

    asyncio.run(scenario())


class _ReplayStopPolicy(LoopPolicy):
    def __init__(self, action: str):
        self.action = action
        self.requests = []

    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name=f"tests:verified-model-replay:{self.action}",
            behavior_version="1",
            implementation_version="1",
        )

    async def before_stop(self, context):
        result = context.step_result
        self.requests.append(
            (context.step, result.model_step_id, result.model_attempt_id, result.text_content)
        )
        if self.action == "continue" and context.step == 1:
            return BeforeStopDecision.continue_with(Message.text("user", "Perform the follow-up."))
        if self.action == "interrupt":
            return BeforeStopDecision.interrupt("Explicit review is required.")
        if self.action == "fail":
            return BeforeStopDecision.fail("The recovered result did not pass policy.")
        return BeforeStopDecision.complete()


@pytest.mark.parametrize("result_payload", [None, True, [], "invalid"])
def test_tool_settlement_rejects_non_object_terminal_result(result_payload):
    event = Event(
        type=EventType.TOOL_CALL_FAILED,
        session_id="settlement-shape-test",
        tool_name="record_effect",
        payload={"tool_call_id": "record-effect", "result": result_payload},
    )
    assert not RecoveryCoordinator._tool_terminals_are_settled([event], ["record-effect"])


_RECOVERY_SECRET_CANARY = "verified-recovery-dynamic-vault-canary"


def _register_recovery_vault(app):
    app.register_environment(
        Environment(
            EnvironmentSpec(
                name="recovery-vault",
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name="tests:verified-recovery-vault",
                    behavior_version="1",
                    implementation_version="1",
                ),
            ),
            vault=StaticVault({"recovery_key": _RECOVERY_SECRET_CANARY}),
        ),
        default=True,
    )


class _RecordedRecoveryHook(RuntimeHook):
    def __init__(self, counter: Path, mode: str):
        self.counter = counter
        self.mode = mode
        self.entered = asyncio.Event()
        self.cancel_received = asyncio.Event()
        self.release = asyncio.Event()
        self.quiesced = asyncio.Event()

    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name=f"tests:verified-staged-recovery-hook-{self.mode}",
            behavior_version="1",
            implementation_version="1",
        )

    async def after_tool_call(self, context):
        assert _RECOVERY_SECRET_CANARY not in context.result.model_dump_json()
        with self.counter.open("a", encoding="utf-8") as stream:
            stream.write(context.tool_call_id + "\n")
        if self.mode == "fail" and context.tool_call_id == "record-effect-2":
            raise RuntimeError("The recovered observational callback failed.")
        if self.mode == "cancel" and context.tool_call_id == "record-effect-2":
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancel_received.set()
                await self.release.wait()
                raise
            finally:
                self.quiesced.set()


class _ToolReplayProvider(_RecordingProvider):
    def __init__(self, call_count=1):
        super().__init__()
        self.call_count = call_count

    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name=f"tests:verified-tool-publication-provider-{self.call_count}",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request):
        self.requests.append(request)
        if any(message.role.value == "tool" for message in request.messages):
            yield ModelStreamEvent.text_delta("The recorded effect is complete.")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
        else:
            for index in range(self.call_count):
                yield ModelStreamEvent.tool_call(
                    id="record-effect" if index == 0 else f"record-effect-{index + 1}",
                    name="record_effect",
                    arguments={},
                )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


class _RecordedExternalEffect(Tool):
    spec = ToolSpec(
        name="record_effect",
        description="Record one observable non-idempotent effect for crash recovery.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        effect=ToolEffect.EXTERNAL,
        parallel_safe=False,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:verified-tool-publication-effect",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    def __init__(
        self,
        counter: Path,
        *,
        result_error: bool = False,
        execution_error: bool = False,
        stage_terminal: bool = False,
        dynamic_secret: bool = False,
    ):
        super().__init__()
        self.counter = counter
        self.result_error = result_error
        self.execution_error = execution_error
        self.dynamic_secret = dynamic_secret
        self.spec = self.spec.model_copy(
            update={
                "workspace_mutation": stage_terminal,
                "execution_profile_identity": ExecutionProfileBehaviorIdentity(
                    name=f"tests:verified-tool-publication-effect-{result_error}-{execution_error}-{stage_terminal}-{dynamic_secret}",
                    behavior_version="1",
                    implementation_version="1",
                ),
            }
        )

    async def run(self, ctx, args):
        content = "Recorded exactly one effect."
        if self.dynamic_secret:
            resolved = await ctx.vault.resolve(SecretRef(name="recovery_key"))
            content = resolved.value.get_secret_value()
            assert content == _RECOVERY_SECRET_CANARY
        with self.counter.open("a", encoding="utf-8") as stream:
            stream.write("effect\n")
        if self.execution_error:
            raise RuntimeError("External effect happened but its outcome is unknown.")
        return ToolResult(
            content=content,
            structured={"recorded": True},
            is_error=self.result_error,
        )


def _crash_worker_tool_publication(
    directory: str,
    *,
    after_publication: bool,
    crash_before_terminal: bool = False,
    call_count: int = 1,
    result_error: bool = False,
    execution_error: bool = False,
    crash_staged_terminal: bool = False,
    hook_mode: str = "none",
    dynamic_secret: bool = False,
    postgres_dsn: str | None = None,
) -> None:
    async def scenario():
        root = Path(directory)
        factory = VerifiedWorkerStoreFactory(
            "sqlite" if postgres_dsn is None else "postgres", root, postgres_dsn
        )
        sessions, tasks = factory()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
            runtime_hooks=[]
            if hook_mode == "none"
            else [_RecordedRecoveryHook(root / "hooks.txt", hook_mode)],
        )
        if dynamic_secret:
            _register_recovery_vault(app)
        app.register_provider(_ToolReplayProvider(call_count), default=True)
        tool = _RecordedExternalEffect(
            root / "effects.txt",
            result_error=result_error,
            execution_error=execution_error,
            stage_terminal=crash_staged_terminal,
            dynamic_secret=dynamic_secret,
        )
        if crash_before_terminal:
            run = tool.run

            async def lose_effect_result(ctx, args):
                result = await run(ctx, args)
                if tool.counter.read_text(encoding="utf-8").count("effect\n") == call_count:
                    os._exit(76)
                return result

            tool.run = lose_effect_result
        app.register_agent(
            AgentSpec(name="worker", model="verified-work-test-model"),
            tools=[tool],
        )
        contract = _contract()
        await tasks.publish_work_contract(contract)
        await tasks.create_task(
            TaskCreate(
                task_id="tool-publication-crash",
                type="verified",
                work_contract=contract.reference(),
            )
        )
        app.register_completion_verifier(contract.verifier, RecordingVerifier(_accepted_decision()))
        app.register_completion_result_resolver(contract.result_resolver, _Resolver(_task_result()))
        publish = sessions.publish_runtime_publication
        append = sessions.append_events

        async def crash_at_staged_terminal(session_id, events):
            if (
                crash_staged_terminal
                and any(
                    event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
                    for event in events
                )
                and tool.counter.read_text(encoding="utf-8").count("effect\n") == call_count
            ):
                os._exit(77)
            return await append(session_id, events)

        sessions.append_events = crash_at_staged_terminal

        async def crash_at_tool_publication(*args, **kwargs):
            if kwargs["request"].kind != "tool-round":
                return await publish(*args, **kwargs)
            assert (root / "effects.txt").read_text(encoding="utf-8") == "effect\n" * call_count
            if after_publication:
                await publish(*args, **kwargs)
            os._exit(75)

        sessions.publish_runtime_publication = crash_at_tool_publication
        async with VerifiedTaskWorker(
            app,
            _StaticHandler(),
            worker_id="crashed-tool-worker",
            lease_seconds=5,
            callback_timeout_seconds=1,
        ) as worker:
            await worker.run(max_tasks=1)

    asyncio.run(scenario())


@pytest.mark.parametrize("publication_boundary", ["before", "after", "staged"])
@pytest.mark.parametrize("call_count", [1, 2])
@pytest.mark.parametrize("result_error", [False, True], ids=["success", "returned-error"])
def test_worker_recovers_tool_publication_after_process_exit(
    tmp_path,
    publication_boundary,
    call_count,
    result_error,
    hook_mode="none",
    dynamic_secret=False,
    cancellation_requests=1,
    late_publication_failure=False,
    store_factory=None,
):
    after_publication = publication_boundary == "after"
    crash_staged_terminal = publication_boundary == "staged"
    repository = Path(__file__).resolve().parents[2]
    factory = store_factory or VerifiedWorkerStoreFactory("sqlite", tmp_path)
    child_environment = {**os.environ, "PYTHONPATH": str(repository / "src")}
    child_environment.pop("CAYU_TEST_VERIFIED_WORKER_DSN", None)
    if factory.postgres_dsn is not None:
        child_environment["CAYU_TEST_VERIFIED_WORKER_DSN"] = factory.postgres_dsn
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.core.test_verified_task_worker import _crash_worker_tool_publication; "
            "import os, sys; _crash_worker_tool_publication("
            "sys.argv[1], after_publication=sys.argv[2] == 'True', call_count=int(sys.argv[3]), "
            "result_error=sys.argv[4] == 'True', crash_staged_terminal=sys.argv[5] == 'True', "
            "hook_mode=sys.argv[6], dynamic_secret=sys.argv[7] == 'True', "
            "postgres_dsn=os.environ.get('CAYU_TEST_VERIFIED_WORKER_DSN'))",
            str(tmp_path),
            str(after_publication),
            str(call_count),
            str(result_error),
            str(crash_staged_terminal),
            hook_mode,
            str(dynamic_secret),
        ],
        cwd=repository,
        env=child_environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert child.returncode == (77 if crash_staged_terminal else 75), (child.stdout, child.stderr)
    assert _RECOVERY_SECRET_CANARY not in child.stdout + child.stderr

    async def scenario():
        sessions, tasks = factory()
        try:
            admission = await tasks.load_latest_work_attempt_admission("tool-publication-crash")
            assert admission.execution_entry is not None
            [original] = await sessions.query_events(
                EventQuery(session_id=admission.session_id, event_types={EventType.MODEL_COMPLETED})
            )
            round_id = original.event.payload["tool_round_id"]
            assert original.event.payload["step"] == 1
            publication_id = f"tool-round:{round_id}"
            receipt = await sessions.load_runtime_publication_receipt(
                admission.session_id, publication_id
            )
            assert (receipt is not None) is after_publication
            pending = tool_round_recovery.pending_tool_round_from_checkpoint(
                await sessions.load_checkpoint(admission.session_id)
            )
            if after_publication:
                assert pending is None
            else:
                assert pending.tool_round_id == round_id
                call_ids = {"record-effect"}
                if call_count == 2:
                    call_ids.add("record-effect-2")
                lifecycle = await sessions.load_tool_round_lifecycle_events_for_round(
                    admission.session_id,
                    sorted(call_ids),
                    tool_round_identity=tool_round_recovery.pending_tool_round_identity(pending),
                )
                outcomes, _ = tool_round_recovery.recorded_tool_outcomes(
                    events=lifecycle, pending_round=pending
                )
                assert set(outcomes) == call_ids
                assert all(
                    result.result.structured == {"recorded": True} for result in outcomes.values()
                )
                assert all(result.result.is_error is result_error for result in outcomes.values())
                if dynamic_secret:
                    assert pending.assistant_publication.secret_resolution_scope == "dynamic"
                    assert set(pending.assistant_publication.covered_tool_call_ids) == call_ids
                    for staged in pending.staged_terminals:
                        assert _RECOVERY_SECRET_CANARY not in staged.model_dump_json()
                if crash_staged_terminal:
                    terminal_ids = {
                        event.payload["tool_call_id"]
                        for event in lifecycle
                        if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
                    }
                    last_id = "record-effect" if call_count == 1 else "record-effect-2"
                    assert last_id not in terminal_ids
                    assert any(
                        staged.tool_call_id == last_id
                        for staged in tool_round_recovery.staged_terminal_records(pending)
                    )
            assert (tmp_path / "effects.txt").read_text(encoding="utf-8") == "effect\n" * call_count
            await wait_for_verified_worker_lease_expiry(tasks, admission.claim.lease_expires_at)
            recovery_hook = (
                None
                if hook_mode == "none"
                else _RecordedRecoveryHook(tmp_path / "hooks.txt", hook_mode)
            )
            app = CayuApp(
                session_store=sessions,
                task_store=tasks,
                enable_logging=False,
                runtime_hooks=[] if recovery_hook is None else [recovery_hook],
            )
            provider = _ToolReplayProvider(call_count)
            if dynamic_secret:
                _register_recovery_vault(app)
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="worker", model="verified-work-test-model"),
                tools=[
                    _RecordedExternalEffect(
                        tmp_path / "effects.txt",
                        result_error=result_error,
                        stage_terminal=crash_staged_terminal,
                        dynamic_secret=dynamic_secret,
                    )
                ],
            )
            contract = _contract()
            verifier = RecordingVerifier(_accepted_decision())
            app.register_completion_verifier(contract.verifier, verifier)
            app.register_completion_result_resolver(
                contract.result_resolver, _Resolver(_task_result())
            )
            handler = _StaticHandler()
            if hook_mode != "none":
                assert (tmp_path / "hooks.txt").read_text(encoding="utf-8") == "record-effect\n"
            if crash_staged_terminal:
                assert await app._session_engine.has_recoverable_work_attempt_model_result(
                    admission
                )
                if call_count == 2:
                    assert [
                        (item.tool_call_id, item.hooks_state) for item in pending.staged_terminals
                    ] == [("record-effect", "completed"), ("record-effect-2", "pending")]
            if hook_mode == "cancel":
                async with VerifiedTaskWorker(
                    app, handler, worker_id="replacement", callback_timeout_seconds=0.1
                ) as worker:
                    running = asyncio.create_task(worker.run(max_tasks=1))
                    try:
                        await asyncio.wait_for(recovery_hook.entered.wait(), 10)
                        running.cancel("cancel recovered hook")
                        assert running.cancelling() == 1
                        if cancellation_requests == 2:
                            await asyncio.sleep(0)
                            assert not running.done()
                            assert not recovery_hook.quiesced.is_set()
                            # The owned wait temporarily consumes the first request;
                            # both requests must be restored on the final cancelled task.
                            assert running.cancelling() == 0
                            running.cancel("cancel recovered hook while draining")
                            assert running.cancelling() == 1
                        with pytest.raises(asyncio.CancelledError):
                            await asyncio.wait_for(running, 10)
                        assert running.cancelled()
                        assert running.cancelling() == cancellation_requests
                        assert not recovery_hook.quiesced.is_set()
                        assert worker._running is not None and not worker._running.done()
                        with pytest.raises(VerifiedTaskWorkerDraining):
                            await worker.aclose()
                        current = await tasks.load_latest_work_attempt_admission(admission.task_id)
                        stop = asyncio.Event()
                        async with VerifiedTaskWorker(
                            app, _StaticHandler(), worker_id="competitor", poll_interval_s=0.01
                        ) as competitor:
                            discover = competitor._discover_unfinished_attempt
                            scans = 0

                            async def observe_scan():
                                nonlocal scans
                                result = await discover()
                                scans += 1
                                if scans == 3:
                                    stop.set()
                                return result

                            competitor._discover_unfinished_attempt = observe_scan
                            assert await asyncio.wait_for(competitor.run(stop=stop), 10) == 0
                        assert scans >= 3
                        assert (
                            await tasks.load_latest_work_attempt_admission(admission.task_id)
                        ).claim.generation == current.claim.generation
                        if late_publication_failure:
                            append = sessions.append_events

                            async def fail_hook_publication(session_id, events):
                                if any(
                                    event.type is EventType.HOOK_COMPLETED
                                    and event.payload.get("tool_call_id") == "record-effect-2"
                                    for event in events
                                ):
                                    raise ConnectionError("recovered hook publication failed")
                                return await append(session_id, events)

                            sessions.append_events = fail_hook_publication
                        recovery_hook.release.set()
                        done, _ = await asyncio.wait({worker._running}, timeout=10)
                        assert done
                        assert recovery_hook.quiesced.is_set()
                        if late_publication_failure:
                            with pytest.raises(BaseException) as late:
                                await worker.aclose()
                            assert not isinstance(late.value, asyncio.CancelledError)
                            assert (
                                sum(
                                    isinstance(error, ConnectionError)
                                    and str(error)
                                    == "ConnectionError: recovered hook publication failed"
                                    for error in iter_exception_tree(late.value)
                                )
                                == 1
                            )
                            await worker.aclose()
                        else:
                            await worker.aclose()
                        assert worker._running is None
                    finally:
                        recovery_hook.release.set()
                        if not running.done():
                            running.cancel()
                        await asyncio.gather(running, return_exceptions=True)
                        if worker._running is not None:
                            done, _ = await asyncio.wait({worker._running}, timeout=10)
                            assert done
                            await worker.aclose()
                assert provider.requests == []
                assert verifier.requests == []
                assert handler.proposals == []
                assert (await tasks.load_task(admission.task_id)).status is TaskStatus.RUNNING
                assert (
                    await tasks.load_completion_proposal_for_attempt(admission.attempt_id) is None
                )
                assert (tmp_path / "effects.txt").read_text(
                    encoding="utf-8"
                ) == "effect\n" * call_count
                assert (tmp_path / "hooks.txt").read_text(encoding="utf-8").splitlines() == [
                    "record-effect",
                    "record-effect-2",
                ]
                return
            async with VerifiedTaskWorker(app, handler, worker_id="replacement") as worker:
                try:
                    assert (
                        await asyncio.wait_for(
                            worker.run(max_tasks=1), _RECOVERY_COMPLETION_TIMEOUT_SECONDS
                        )
                        == 1
                    )
                except TimeoutError as failure:
                    current = await tasks.load_latest_work_attempt_admission(admission.task_id)
                    current_task = await tasks.load_task(admission.task_id)
                    current_session = await sessions.load(admission.session_id)
                    failure.add_note(
                        "Replacement progress: "
                        f"generation={current.claim.generation}, "
                        f"admission={current.state.value}, task={current_task.status.value}, "
                        f"session={current_session.status.value}, "
                        f"provider_requests={len(provider.requests)}, "
                        f"proposals={len(handler.proposals)}, verifier_requests={len(verifier.requests)}"
                    )
                    raise
            assert (tmp_path / "effects.txt").read_text(encoding="utf-8") == "effect\n" * call_count
            assert handler.preparations == []
            assert len(provider.requests) == 1
            result_parts = [
                part
                for message in provider.requests[0].messages
                if message.role.value == "tool"
                for part in message.content
                if part.type == "tool_result"
            ]
            assert len(result_parts) == call_count
            assert {part.tool_call_id for part in result_parts} == (
                {"record-effect", "record-effect-2"} if call_count == 2 else {"record-effect"}
            )
            for part in result_parts:
                assert part.tool_round_id == round_id
                assert part.model_step_id == original.event.payload["model_step_id"]
                assert part.model_attempt_id == original.event.payload["model_attempt_id"]
                if dynamic_secret and part.tool_call_id == "record-effect-2":
                    assert part.is_error
                    assert part.structured["reason"] == "recovery_hook_secret_scope_unavailable"
                else:
                    assert part.structured == {"recorded": True}
                    assert part.is_error is result_error
                assert _RECOVERY_SECRET_CANARY not in part.model_dump_json()
            completions = await sessions.query_events(
                EventQuery(session_id=admission.session_id, event_types={EventType.MODEL_COMPLETED})
            )
            assert sorted(record.event.payload["step"] for record in completions) == [1, 2]
            assert (await tasks.load_task(admission.task_id)).status is TaskStatus.COMPLETED
            assert (
                await tasks.load_latest_work_attempt_admission(admission.task_id)
            ).attempt_id == (admission.attempt_id)
            assert (
                await sessions.load_runtime_publication_receipt(
                    admission.session_id, publication_id
                )
                is not None
            )
            assert (
                tool_round_recovery.pending_tool_round_from_checkpoint(
                    await sessions.load_checkpoint(admission.session_id)
                )
                is None
            )
            assert len(verifier.requests) == 1
            if hook_mode != "none":
                assert (tmp_path / "hooks.txt").read_text(encoding="utf-8").splitlines() == (
                    ["record-effect"] if dynamic_secret else ["record-effect", "record-effect-2"]
                )
                hook_failures = await sessions.query_events(
                    EventQuery(session_id=admission.session_id, event_types={EventType.HOOK_FAILED})
                )
                assert len(hook_failures) == (1 if hook_mode == "fail" else 0)
            if dynamic_secret:
                for record in await sessions.query_events(
                    EventQuery(session_id=admission.session_id)
                ):
                    assert _RECOVERY_SECRET_CANARY not in record.event.model_dump_json()
                for request in provider.requests:
                    for message in request.messages:
                        assert _RECOVERY_SECRET_CANARY not in message.model_dump_json()
        finally:
            await tasks.close()
            await sessions.close()

    asyncio.run(scenario())


@pytest.mark.postgres_recovery
@pytest.mark.parametrize("backend", ["postgres"])
@pytest.mark.parametrize("publication_boundary", ["before", "after", "staged"])
@pytest.mark.parametrize("call_count", [1, 2])
@pytest.mark.parametrize("result_error", [False, True], ids=["success", "returned-error"])
def test_postgres_worker_recovers_tool_publication_after_process_exit(
    tmp_path,
    backend,
    publication_boundary,
    call_count,
    result_error,
    verified_worker_store_factory,
):
    test_worker_recovers_tool_publication_after_process_exit(
        tmp_path,
        publication_boundary,
        call_count,
        result_error,
        store_factory=verified_worker_store_factory,
    )


@pytest.mark.postgres_recovery
@pytest.mark.parametrize("backend", ["postgres"])
@pytest.mark.parametrize(
    ("hook_mode", "dynamic_secret", "cancellation_requests", "late_publication_failure"),
    [
        ("pass", False, 1, False),
        ("fail", False, 1, False),
        ("pass", True, 1, False),
        ("cancel", False, 1, False),
        ("cancel", False, 2, False),
        ("cancel", False, 1, True),
    ],
    ids=["complete", "fail", "lost-redactor", "cancel", "cancel-twice", "late-publication-failure"],
)
def test_postgres_worker_preserves_recovered_hook_ownership(
    tmp_path,
    backend,
    hook_mode,
    dynamic_secret,
    cancellation_requests,
    late_publication_failure,
    verified_worker_store_factory,
):
    test_worker_recovers_tool_publication_after_process_exit(
        tmp_path,
        "staged",
        2,
        False,
        hook_mode=hook_mode,
        dynamic_secret=dynamic_secret,
        cancellation_requests=cancellation_requests,
        late_publication_failure=late_publication_failure,
        store_factory=verified_worker_store_factory,
    )


@pytest.mark.parametrize("hook_mode", ["pass", "fail"])
def test_worker_recovers_pending_tool_hook_without_repeating_completed_hook(tmp_path, hook_mode):
    test_worker_recovers_tool_publication_after_process_exit(
        tmp_path, "staged", 2, False, hook_mode=hook_mode
    )


def test_worker_skips_pending_hook_after_dynamic_secret_registry_loss(tmp_path):
    test_worker_recovers_tool_publication_after_process_exit(
        tmp_path, "staged", 2, False, hook_mode="pass", dynamic_secret=True
    )


@pytest.mark.parametrize("cancellation_requests", [1, 2])
def test_worker_cancellation_owns_recovered_hook_until_quiescence(tmp_path, cancellation_requests):
    test_worker_recovers_tool_publication_after_process_exit(
        tmp_path,
        "staged",
        2,
        False,
        hook_mode="cancel",
        cancellation_requests=cancellation_requests,
    )


def test_worker_reports_late_recovered_hook_publication_failure_once(tmp_path):
    test_worker_recovers_tool_publication_after_process_exit(
        tmp_path, "staged", 2, False, hook_mode="cancel", late_publication_failure=True
    )


class _WorkerCrashOperationAdapter(_OfflineOperationAdapter):
    def __init__(self, directory, *, crash, start_only=False):
        super().__init__(ProviderOperationStatus.COMPLETED)
        self.directory = Path(directory)
        self.crash = crash
        self.start_only = start_only
        self.recovery_keys = []

    @property
    def start_idempotency_support(self):
        if self.start_only:
            return ProviderOperationStartIdempotencySupport.EXACT
        return super().start_idempotency_support

    async def start(self, request):
        self.start_calls += 1
        self.start_requests.append(request)
        assert self.crash, "replacement must retrieve the original operation, never start again"
        # Exclusive creation is independent external evidence of one dispatch.
        with (self.directory / "background-started").open("x") as marker:
            marker.write(self.state.operation_id)
        if self.start_only:
            with (self.directory / "background-start-key").open("x") as marker:
                marker.write(request.idempotency_key)
            # Provider acceptance happened, but Cayu never received the identity.
            os._exit(78)

        async def events():
            # The runtime has published the returned operation identity before
            # pulling its first event. Do not run any Python cleanup on loss.
            os._exit(78)
            yield  # pragma: no cover - keeps this an async iterator

        return ProviderOperationConnection(
            state=self.state,
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )

    async def recover_start(self, request):
        assert self.start_only and not self.crash
        assert request.idempotency_key == (self.directory / "background-start-key").read_text()
        self.recovery_keys.append(request.idempotency_key)

        async def events():
            if False:
                yield ModelStreamEvent.text_delta("")

        return ProviderOperationConnection(
            state=self.state, status=ProviderOperationStatus.COMPLETED, events=events()
        )


class _WorkerCrashOperationProvider(_OfflineOperationProvider):
    def __init__(self, directory, *, crash=False, start_only=False):
        self.adapter = _WorkerCrashOperationAdapter(directory, crash=crash, start_only=start_only)


def _crash_worker_background_operation(directory, postgres_dsn=None, start_only=False):
    async def scenario():
        sessions, tasks = VerifiedWorkerStoreFactory(
            "sqlite" if postgres_dsn is None else "postgres", Path(directory), postgres_dsn
        )()
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        app.register_provider(
            _WorkerCrashOperationProvider(directory, crash=True, start_only=start_only),
            default=True,
        )
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        contract = _contract()
        await tasks.publish_work_contract(contract)
        await tasks.create_task(
            TaskCreate(
                task_id="background-operation-crash",
                type="verified",
                work_contract=contract.reference(),
            )
        )
        app.register_completion_verifier(contract.verifier, RecordingVerifier(_accepted_decision()))
        app.register_completion_result_resolver(contract.result_resolver, _Resolver(_task_result()))
        async with VerifiedTaskWorker(
            app,
            _StaticHandler(),
            worker_id="crashed-worker",
            lease_seconds=5,
            callback_timeout_seconds=1,
        ) as worker:
            await worker.run(max_tasks=1)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize(
    "pending_first", [False, True], ids=["completed", "pending_then_completed"]
)
def test_worker_recovers_recorded_background_operation_after_process_exit(
    backend,
    pending_first,
    tmp_path,
    verified_worker_store_factory,
    *,
    start_only=False,
    retrieval_failure=False,
    cancel_retrieval=False,
):
    from cayu import VerifiedTaskWorker as PublicWorker
    from cayu.runtime.provider_operations import (
        load_recoverable_provider_operation,
        load_recoverable_provider_operation_start,
    )

    repository = Path(__file__).resolve().parents[2]
    factory = verified_worker_store_factory
    child_environment = {**os.environ, "PYTHONPATH": str(repository / "src")}
    child_environment.pop("CAYU_TEST_VERIFIED_WORKER_DSN", None)
    if factory.postgres_dsn is not None:
        child_environment["CAYU_TEST_VERIFIED_WORKER_DSN"] = factory.postgres_dsn
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.core.test_verified_task_worker import "
            "_crash_worker_background_operation; import os, sys; "
            "_crash_worker_background_operation(sys.argv[1], "
            "os.environ.get('CAYU_TEST_VERIFIED_WORKER_DSN'), sys.argv[2] == 'True')",
            str(tmp_path),
            str(start_only),
        ],
        cwd=repository,
        env=child_environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert child.returncode == 78, (child.stdout, child.stderr)

    async def scenario():
        sessions, tasks = factory()
        try:
            original = await tasks.load_latest_work_attempt_admission("background-operation-crash")
            assert original.execution_entry is not None
            assert original.claim.generation == 1
            active = await sessions.load_active_model_completion_stage(original.session_id)
            assert active is not None and active.stage.state == "in_flight"
            operation = await load_recoverable_provider_operation(sessions, active.stage)
            start = await load_recoverable_provider_operation_start(sessions, active.stage)
            assert start is not None
            if start_only:
                assert operation is None
                assert start.start_id == (tmp_path / "background-start-key").read_text()
                assert start.idempotency_support is ProviderOperationStartIdempotencySupport.EXACT
            else:
                assert operation is not None
                assert (tmp_path / "background-started").read_text() == operation.state.operation_id
            assert (
                await sessions.load_model_completion_stage_dispatch(
                    original.session_id, active.stage.stage_id
                )
                is not None
            )
            await wait_for_verified_worker_lease_expiry(tasks, original.claim.lease_expires_at)
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            provider = _WorkerCrashOperationProvider(tmp_path, start_only=start_only)
            assert (
                tmp_path / "background-started"
            ).read_text() == provider.adapter.state.operation_id
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            contract = _contract()
            verifier = RecordingVerifier(_accepted_decision())
            app.register_completion_verifier(contract.verifier, verifier)
            app.register_completion_result_resolver(
                contract.result_resolver, _Resolver(_task_result())
            )
            handler = _StaticHandler()
            async with PublicWorker(
                app,
                handler,
                worker_id="replacement",
                lease_seconds=5,
                callback_timeout_seconds=1,
            ) as worker:
                if cancel_retrieval:
                    started = asyncio.Event()
                    release = asyncio.Event()
                    settled = asyncio.Event()

                    async def blocked_retrieval(state):
                        provider.adapter.retrieve_calls.append(state)
                        started.set()
                        try:
                            try:
                                await release.wait()
                            except asyncio.CancelledError:
                                # Cancellation of the waiter is not proof that
                                # the adapter's dispatched request has stopped.
                                await release.wait()
                            return ProviderOperationSnapshot(
                                state=state, status=ProviderOperationStatus.IN_PROGRESS
                            )
                        finally:
                            settled.set()

                    provider.adapter.retrieve = blocked_retrieval
                    running = asyncio.create_task(worker.run(max_tasks=1))
                    try:
                        await asyncio.wait_for(started.wait(), 10)
                        running.cancel("cancel worker during provider retrieval")
                        with pytest.raises(
                            asyncio.CancelledError, match="cancel worker during provider retrieval"
                        ):
                            await asyncio.wait_for(running, 5)
                        assert running.cancelling() == 1 and running.cancelled()
                        assert not settled.is_set()
                        with pytest.raises(VerifiedTaskWorkerDraining):
                            await worker.run(max_tasks=1)
                        # A distinct public worker must also observe the live
                        # owner's lease without taking over the same attempt.
                        queried = asyncio.Event()
                        stop_competitor = asyncio.Event()
                        competing = None
                        try:
                            async with PublicWorker(
                                app,
                                _StaticHandler(),
                                worker_id="competing-retrieval-owner",
                                lease_seconds=5,
                                callback_timeout_seconds=1,
                            ) as competitor:
                                step = competitor._step

                                async def observed_step(*args, **kwargs):
                                    result = await step(*args, **kwargs)
                                    queried.set()
                                    return result

                                competitor._step = observed_step
                                competing = asyncio.create_task(
                                    competitor.run(stop=stop_competitor, max_tasks=1)
                                )
                                await asyncio.wait_for(queried.wait(), 5)
                                stop_competitor.set()
                                assert await asyncio.wait_for(competing, 5) == 0
                        finally:
                            stop_competitor.set()
                            if competing is not None:
                                if not competing.done():
                                    competing.cancel()
                                await asyncio.gather(competing, return_exceptions=True)
                        retained = await tasks.load_latest_work_attempt_admission(original.task_id)
                        assert retained.attempt_id == original.attempt_id
                        assert retained.claim.generation == 2
                        assert handler.preparations == handler.proposals == verifier.requests == []
                        assert (
                            await tasks.load_work_attempt_lifecycle_receipt(original.admission_id)
                            is None
                        )
                    finally:
                        release.set()
                        if not running.done():
                            running.cancel()
                        await asyncio.gather(running, return_exceptions=True)
                        await worker.aclose()
                    assert settled.is_set()
                    assert provider.adapter.start_calls == 0
                    assert provider.adapter.retrieve_calls == [provider.adapter.state]
                    assert handler.proposals == verifier.requests == []
                    assert (
                        await tasks.load_work_attempt_lifecycle_receipt(original.admission_id)
                        is None
                    )
                    assert (
                        await tasks.load_task(original.task_id)
                    ).status is not TaskStatus.COMPLETED
                    still_active = await sessions.load_active_model_completion_stage(
                        original.session_id
                    )
                    assert still_active is not None and still_active.stage == active.stage
                    return
                if retrieval_failure:

                    async def unavailable(state):
                        provider.adapter.retrieve_calls.append(state)
                        raise RuntimeError("provider retrieval unavailable")

                    provider.adapter.retrieve = unavailable
                    with pytest.raises(WorkAttemptRecoveryRequired):
                        await asyncio.wait_for(worker.run(max_tasks=1), 15)
                    failed = await tasks.load_latest_work_attempt_admission(original.task_id)
                    assert failed.attempt_id == original.attempt_id
                    assert failed.claim.generation == 2
                    assert provider.adapter.retrieve_calls == [provider.adapter.state]
                    assert provider.adapter.start_calls == 0
                    assert handler.preparations == handler.proposals == verifier.requests == []
                    assert (
                        await tasks.load_task(original.task_id)
                    ).status is not TaskStatus.COMPLETED
                    assert (
                        await tasks.load_completion_proposal_for_attempt(original.attempt_id)
                        is None
                    )
                    assert (
                        await tasks.load_work_attempt_lifecycle_receipt(original.admission_id)
                        is None
                    )
                    still_active = await sessions.load_active_model_completion_stage(
                        original.session_id
                    )
                    assert still_active is not None and still_active.stage == active.stage
                    required = await sessions.query_events(
                        EventQuery(
                            session_id=original.session_id,
                            event_types={EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED},
                        )
                    )
                    assert len(required) == 1
                    assert required[0].event.payload["recovery_reason"] == "unavailable"
                    assert (
                        await sessions.query_events(
                            EventQuery(
                                session_id=original.session_id,
                                event_types={EventType.MODEL_COMPLETED},
                            )
                        )
                        == []
                    )
                    return
                if pending_first:
                    provider.adapter.status = ProviderOperationStatus.IN_PROGRESS
                    with pytest.raises(WorkAttemptRecoveryRequired):
                        await asyncio.wait_for(worker.run(max_tasks=1), 15)
                    pending = await tasks.load_latest_work_attempt_admission(original.task_id)
                    assert pending.attempt_id == original.attempt_id
                    assert pending.claim.generation == 2
                    assert provider.adapter.retrieve_calls == [operation.state]
                    assert provider.adapter.start_calls == 0
                    assert handler.proposals == verifier.requests == []
                    assert (
                        await tasks.load_work_attempt_lifecycle_receipt(original.admission_id)
                        is None
                    )
                    still_active = await sessions.load_active_model_completion_stage(
                        original.session_id
                    )
                    assert still_active is not None and still_active.stage == active.stage
                    await wait_for_verified_worker_lease_expiry(
                        tasks, pending.claim.lease_expires_at
                    )
                    provider.adapter.status = ProviderOperationStatus.COMPLETED
                assert await asyncio.wait_for(worker.run(max_tasks=1), 15) == 1
            current = await tasks.load_latest_work_attempt_admission(original.task_id)
            assert current.attempt_id == original.attempt_id
            assert current.claim.generation == (3 if pending_first else 2)
            assert provider.adapter.start_calls == 0
            assert provider.adapter.recovery_keys == ([start.start_id] if start_only else [])
            assert provider.adapter.retrieve_calls == [provider.adapter.state] * (
                2 if pending_first else 1
            )
            assert handler.preparations == []
            assert len(verifier.requests) == len(handler.proposals) == 1
            final = await tasks.load_task(original.task_id)
            assert final.status is TaskStatus.COMPLETED
            receipt = await tasks.load_work_attempt_lifecycle_receipt(original.admission_id)
            assert receipt.retired_contract_binding
            assert await sessions.load_active_model_completion_stage(original.session_id) is None
            completed = await sessions.query_events(
                EventQuery(session_id=original.session_id, event_types={EventType.MODEL_COMPLETED})
            )
            assert len(completed) == 1
        finally:
            await tasks.close()
            await sessions.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_worker_recovers_background_start_acknowledgement_loss(
    backend, tmp_path, verified_worker_store_factory
):
    test_worker_recovers_recorded_background_operation_after_process_exit(
        backend, False, tmp_path, verified_worker_store_factory, start_only=True
    )


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_worker_fences_unavailable_background_retrieval(
    backend, tmp_path, verified_worker_store_factory
):
    test_worker_recovers_recorded_background_operation_after_process_exit(
        backend, False, tmp_path, verified_worker_store_factory, retrieval_failure=True
    )


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_worker_owns_background_retrieval_after_real_cancellation(
    backend, tmp_path, verified_worker_store_factory
):
    test_worker_recovers_recorded_background_operation_after_process_exit(
        backend, False, tmp_path, verified_worker_store_factory, cancel_retrieval=True
    )


class _StructuredReplayProvider(_RecordingProvider):
    supports_native_structured_output = True

    def __init__(self, outcome):
        super().__init__()
        self.outcome = outcome

    async def stream(self, request):
        self.requests.append(request)
        if self.outcome.startswith("tool_"):
            from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME

            yield ModelStreamEvent.tool_call(
                id="structured-recovery-result",
                name=STRUCTURED_OUTPUT_TOOL_NAME,
                arguments={
                    "output": {"answer": "ok"} if self.outcome == "tool_valid" else {"wrong": True}
                },
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta(
            '{"answer":"ok"}' if self.outcome == "valid" else '{"wrong":true}'
        )
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _StructuredReplayHandler(_StaticHandler):
    def __init__(self, strategy="native", *, max_retries=0):
        super().__init__()
        self.strategy = strategy
        self.max_retries = max_retries

    async def prepare(self, context):
        from cayu import StructuredOutputSpec

        request = await super().prepare(context)
        return request.model_copy(
            update={
                "structured_output": StructuredOutputSpec(
                    strategy=self.strategy,
                    max_retries=self.max_retries,
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                )
            }
        )


def _crash_worker_before_model_promotion(
    directory: str,
    *,
    after_promotion: bool = False,
    terminal_failure: bool = False,
    crash_in_flight: bool = False,
    stop_action: str | None = None,
    structured: str | None = None,
    postgres_dsn: str | None = None,
) -> None:
    """Exit the worker around promotion of a durably completed model stage."""

    async def scenario():
        sessions, tasks = VerifiedWorkerStoreFactory(
            "sqlite" if postgres_dsn is None else "postgres", Path(directory), postgres_dsn
        )()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
            loop_policies=None if stop_action is None else [_ReplayStopPolicy(stop_action)],
        )
        provider = (
            _RecordingProvider() if structured is None else _StructuredReplayProvider(structured)
        )
        if crash_in_flight:
            original_stream = provider.stream

            async def crash_during_stream(request):
                async for event in original_stream(request):
                    os._exit(73)
                    yield event

            provider.stream = crash_during_stream
        elif terminal_failure:
            original_stream = provider.stream

            async def fail_after_completion(request):
                async for event in original_stream(request):
                    yield event
                raise RuntimeError("provider stream failed after completion")

            provider.stream = fail_after_completion
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        contract = _contract()
        await tasks.publish_work_contract(contract)
        await tasks.create_task(
            TaskCreate(
                task_id="model-promotion-crash",
                type="verified",
                work_contract=contract.reference(),
            )
        )
        app.register_completion_verifier(contract.verifier, RecordingVerifier(_accepted_decision()))
        app.register_completion_result_resolver(contract.result_resolver, _Resolver(_task_result()))

        promote = sessions.promote_model_completion_stage

        async def crash_before_promotion(*args, **kwargs):
            if after_promotion:
                await promote(*args, **kwargs)
            # Deliberately bypass Python finally blocks: this is a persistent
            # crash window, not an exception that the running owner can repair.
            os._exit(73)

        sessions.promote_model_completion_stage = crash_before_promotion
        async with VerifiedTaskWorker(
            app,
            _StaticHandler()
            if structured is None
            else _StructuredReplayHandler(
                "tool" if structured.startswith("tool_") else "native",
                max_retries=1 if structured in {"repair", "tool_repair"} else 0,
            ),
            worker_id="crashed-worker",
            lease_seconds=5,
            callback_timeout_seconds=1,
        ) as worker:
            await worker.run(max_tasks=1)

    asyncio.run(scenario())


def _crash_replacement_after_execution_entry(
    directory: str, postgres_dsn: str | None = None
) -> None:
    """Kill a second real owner after recovery and durable execution election."""

    async def scenario():
        sessions, tasks = VerifiedWorkerStoreFactory(
            "sqlite" if postgres_dsn is None else "postgres", Path(directory), postgres_dsn
        )()
        original = await tasks.load_latest_work_attempt_admission("model-promotion-crash")
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        provider = _RecordingProvider()
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        contract = _contract()
        app.register_completion_verifier(contract.verifier, RecordingVerifier(_accepted_decision()))
        app.register_completion_result_resolver(contract.result_resolver, _Resolver(_task_result()))
        handler = _StaticHandler()
        enter = app._session_engine._enter_work_attempt_invocation_context

        async def crash_after_entry(*args, **kwargs):
            context = await enter(*args, **kwargs)
            elected = context.work_attempt.admission
            assert elected.attempt_id == original.attempt_id
            assert elected.execution_entry.request.generation == original.claim.generation + 1
            assert elected.recovery_evidence_sha256 is not None
            assert provider.requests == []
            assert handler.preparations == []
            os._exit(74)

        app._session_engine._enter_work_attempt_invocation_context = crash_after_entry
        async with VerifiedTaskWorker(
            app,
            handler,
            worker_id="second-crashed-worker",
            lease_seconds=5,
            callback_timeout_seconds=1,
        ) as worker:
            await worker.run(max_tasks=1)

    asyncio.run(scenario())


@pytest.mark.parametrize("after_promotion", [False, True])
@pytest.mark.parametrize("terminal_failure", [False, True], ids=["valid_turn", "failed_turn"])
@pytest.mark.parametrize("crash_again", [False, True], ids=["one_crash", "two_crashes"])
def test_worker_recovers_terminal_model_stage_after_process_exit(
    tmp_path,
    after_promotion,
    terminal_failure,
    crash_again,
    structured=None,
    closed_fault=None,
    store_factory=None,
):
    repository = Path(__file__).resolve().parents[2]
    factory = store_factory or VerifiedWorkerStoreFactory("sqlite", tmp_path)
    child_environment = {**os.environ, "PYTHONPATH": str(repository / "src")}
    child_environment.pop("CAYU_TEST_VERIFIED_WORKER_DSN", None)
    if factory.postgres_dsn is not None:
        child_environment["CAYU_TEST_VERIFIED_WORKER_DSN"] = factory.postgres_dsn
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.core.test_verified_task_worker import "
            "_crash_worker_before_model_promotion; "
            "import os, sys; _crash_worker_before_model_promotion("
            "sys.argv[1], after_promotion=sys.argv[2] == 'True', "
            "terminal_failure=sys.argv[3] == 'True', "
            "structured=None if sys.argv[4] == 'None' else sys.argv[4], "
            "postgres_dsn=os.environ.get('CAYU_TEST_VERIFIED_WORKER_DSN'))",
            str(tmp_path),
            str(after_promotion),
            str(terminal_failure),
            str(structured),
        ],
        cwd=repository,
        env=child_environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert child.returncode == 73, (child.stdout, child.stderr)
    if crash_again:
        replacement = subprocess.run(
            [
                sys.executable,
                "-c",
                "from tests.core.test_verified_task_worker import "
                "_crash_replacement_after_execution_entry; "
                "import os, sys; _crash_replacement_after_execution_entry("
                "sys.argv[1], postgres_dsn=os.environ.get('CAYU_TEST_VERIFIED_WORKER_DSN'))",
                str(tmp_path),
            ],
            cwd=repository,
            env=child_environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert replacement.returncode == 74, (replacement.stdout, replacement.stderr)

    async def scenario():
        sessions, tasks = factory()
        try:
            admission = await tasks.load_latest_work_attempt_admission("model-promotion-crash")
            assert admission.execution_entry is not None
            assert admission.claim.generation == (2 if crash_again else 1)
            staged = await sessions.load_active_model_completion_stage(admission.session_id)
            completed_events = await sessions.query_events(
                EventQuery(session_id=admission.session_id, event_types={EventType.MODEL_COMPLETED})
            )
            if after_promotion or crash_again:
                assert staged is None
                assert len(completed_events) == 1
                completion = completed_events[0].event
            else:
                assert staged is not None and staged.stage.state == "completed"
                assert completed_events == []
                completion = staged.stage.publication.events[0]
            assert completion.payload["step_classification"]["type"] == (
                "failed"
                if terminal_failure
                else "continue"
                if structured and structured.startswith("tool_")
                else "final"
            )
            assert await tasks.load_completion_proposal_for_attempt(admission.attempt_id) is None
            await asyncio.sleep(
                max(0, (admission.claim.lease_expires_at - datetime.now(UTC)).total_seconds())
                + 0.05
            )
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            provider = (
                _RecordingProvider()
                if structured is None
                else _StructuredReplayProvider(
                    "valid"
                    if structured == "repair"
                    else "tool_valid"
                    if structured == "tool_repair"
                    else structured
                )
            )
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            contract = _contract()
            verifier = RecordingVerifier(_accepted_decision())
            app.register_completion_verifier(contract.verifier, verifier)
            app.register_completion_result_resolver(
                contract.result_resolver, _Resolver(_task_result())
            )
            handler = _StaticHandler()
            faults_observed = []
            if closed_fault is not None:
                coordinator = app._session_engine._recovery_coordinator
                read_closed = coordinator._load_closed_structured_output_events
                read_receipt = sessions.load_runtime_publication_receipt
                read_events = sessions.query_events

                async def corrupted_receipt(*args, **kwargs):
                    result = await read_receipt(*args, **kwargs)
                    if result is not None and result.kind == "tool-round":
                        result = result.model_copy(deep=True)
                        auxiliary_changes = {
                            "boolean_schema": ("schema_version", True),
                            "receipt_step": ("step", 2),
                            "receipt_attempt": ("attempt", 2),
                            "receipt_valid": ("valid", False),
                            "receipt_retry": ("retry_scheduled", True),
                            "receipt_event_ids": ("event_ids", ["another-event"]),
                            "receipt_auxiliary_kind": ("kind", "another-validation"),
                        }
                        if closed_fault in auxiliary_changes:
                            key, value = auxiliary_changes[closed_fault]
                            result.intent["auxiliary"][key] = value
                        elif closed_fault in {
                            "receipt_model_step",
                            "receipt_model_attempt",
                            "receipt_round",
                            "receipt_calls",
                        }:
                            field = {
                                "receipt_model_step": "model_step_id",
                                "receipt_model_attempt": "model_attempt_id",
                                "receipt_round": "tool_round_id",
                                "receipt_calls": "tool_call_ids",
                            }[closed_fault]
                            result.intent[field] = (
                                ["another-call"] if field == "tool_call_ids" else "another-identity"
                            )
                        else:
                            field = {
                                "receipt_session": "session_id",
                                "receipt_interaction": "interaction_id",
                                "receipt_appended_ids": "appended_event_ids",
                                "receipt_publication": "publication_id",
                                "receipt_digest": "request_digest",
                                "receipt_source_epoch": "source_run_epoch",
                                "receipt_cursor": "transcript_end_cursor",
                            }[closed_fault]
                            result = result.model_copy(
                                update={
                                    field: ("another-event",)
                                    if field == "appended_event_ids"
                                    else "0" * 64
                                    if field == "request_digest"
                                    else getattr(result, field) + 1
                                    if field in {"source_run_epoch", "transcript_end_cursor"}
                                    else "another-identity"
                                }
                            )
                        faults_observed.append(closed_fault)
                    return result

                async def corrupted_events(query):
                    result = await read_events(query)
                    if (
                        query.event_id is not None
                        and result
                        and result[0].event.type is EventType.STRUCTURED_OUTPUT_VALIDATED
                    ):
                        faults_observed.append(closed_fault)
                        if closed_fault == "missing_event":
                            return []
                        result = [record.model_copy(deep=True) for record in result]
                        if closed_fault == "duplicate_event":
                            return [result[0], result[0].model_copy(deep=True)]
                        envelope_fields = {
                            "event_identity": "id",
                            "event_session": "session_id",
                            "event_interaction": "interaction_id",
                            "event_agent": "agent_name",
                            "event_environment": "environment_name",
                        }
                        if closed_fault in envelope_fields:
                            result[0] = result[0].model_copy(
                                update={
                                    "event": result[0].event.model_copy(
                                        update={envelope_fields[closed_fault]: "another-identity"}
                                    )
                                }
                            )
                        else:
                            result[0].event.payload["valid"] = False
                    return result

                async def read_conflicting_closed(*args, **kwargs):
                    # Inject at the new readback boundary only, after real
                    # recovery publication; do not alter durable source bytes.
                    with pytest.MonkeyPatch.context() as patch:
                        if closed_fault == "boolean_schema" or closed_fault.startswith("receipt_"):
                            patch.setattr(
                                sessions, "load_runtime_publication_receipt", corrupted_receipt
                            )
                        else:
                            patch.setattr(sessions, "query_events", corrupted_events)
                        return await read_closed(*args, **kwargs)

                coordinator._load_closed_structured_output_events = read_conflicting_closed
            async with VerifiedTaskWorker(app, handler, worker_id="replacement") as worker:
                assert (
                    await asyncio.wait_for(
                        worker.run(max_tasks=1), _RECOVERY_COMPLETION_TIMEOUT_SECONDS
                    )
                    == 1
                )
            assert handler.preparations == []
            repair = structured in {"repair", "tool_repair"}
            assert len(provider.requests) == int(repair)
            if repair:
                specification = admission.run_semantics.structured_output
                assert specification.max_retries == 1
                assert specification.strategy == (
                    "tool" if structured == "tool_repair" else "native"
                )
            failed = (
                terminal_failure
                or structured in {"invalid", "tool_invalid"}
                or closed_fault is not None
            )
            if closed_fault is not None:
                assert faults_observed == [closed_fault]
            assert (await tasks.load_task(admission.task_id)).status is (
                TaskStatus.NEEDS_ATTENTION if failed else TaskStatus.COMPLETED
            )
            current = await tasks.load_latest_work_attempt_admission(admission.task_id)
            assert current.attempt_id == admission.attempt_id
            assert current.run_semantics == admission.run_semantics
            assert await sessions.load_active_model_completion_stage(admission.session_id) is None
            receipt = await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
            assert receipt.retired_contract_binding is not failed
            assert len(verifier.requests) == len(handler.proposals) == int(not failed)
            if structured is not None:
                records = await sessions.query_events(
                    EventQuery(
                        session_id=admission.session_id,
                        event_types={
                            EventType.STRUCTURED_OUTPUT_VALIDATED,
                            EventType.STRUCTURED_OUTPUT_FAILED,
                        },
                        limit=10,
                    )
                )
                assert [record.event.type for record in records] == (
                    [EventType.STRUCTURED_OUTPUT_FAILED, EventType.STRUCTURED_OUTPUT_VALIDATED]
                    if repair
                    else [
                        EventType.STRUCTURED_OUTPUT_FAILED
                        if structured in {"invalid", "tool_invalid"}
                        else EventType.STRUCTURED_OUTPUT_VALIDATED
                    ]
                )
                if repair:
                    assert [record.event.payload["attempt"] for record in records] == [1, 2]
        finally:
            await tasks.close()
            await sessions.close()

    asyncio.run(scenario())


@pytest.mark.postgres_recovery
@pytest.mark.parametrize("backend", ["postgres"])
@pytest.mark.parametrize("after_promotion", [False, True])
@pytest.mark.parametrize("terminal_failure", [False, True], ids=["valid_turn", "failed_turn"])
@pytest.mark.parametrize("crash_again", [False, True], ids=["one_crash", "two_crashes"])
def test_postgres_worker_recovers_terminal_model_stage_after_process_exit(
    tmp_path, backend, after_promotion, terminal_failure, crash_again, verified_worker_store_factory
):
    test_worker_recovers_terminal_model_stage_after_process_exit(
        tmp_path,
        after_promotion,
        terminal_failure,
        crash_again,
        store_factory=verified_worker_store_factory,
    )


@pytest.mark.parametrize("structured", ["valid", "invalid", "tool_valid", "tool_invalid"])
@pytest.mark.parametrize("after_promotion", [False, True])
def test_worker_validates_recovered_structured_model_result(tmp_path, structured, after_promotion):
    test_worker_recovers_terminal_model_stage_after_process_exit(
        tmp_path, after_promotion, False, False, structured=structured
    )


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("structured", ["repair", "tool_repair"])
@pytest.mark.parametrize("after_promotion", [False, True])
def test_worker_repairs_recovered_structured_model_result(
    tmp_path, backend, structured, after_promotion, verified_worker_store_factory
):
    test_worker_recovers_terminal_model_stage_after_process_exit(
        tmp_path,
        after_promotion,
        False,
        False,
        structured=structured,
        store_factory=verified_worker_store_factory,
    )


@pytest.mark.parametrize(
    "fault",
    [
        "boolean_schema",
        "missing_event",
        "conflicting_event",
        "duplicate_event",
        "event_identity",
        "event_session",
        "event_interaction",
        "event_agent",
        "event_environment",
        "receipt_step",
        "receipt_attempt",
        "receipt_valid",
        "receipt_retry",
        "receipt_event_ids",
        "receipt_auxiliary_kind",
        "receipt_session",
        "receipt_interaction",
        "receipt_appended_ids",
        "receipt_publication",
        "receipt_digest",
        "receipt_source_epoch",
        "receipt_cursor",
        "receipt_model_step",
        "receipt_model_attempt",
        "receipt_round",
        "receipt_calls",
    ],
)
def test_worker_rejects_conflicting_closed_structured_evidence(tmp_path, fault):
    test_worker_recovers_terminal_model_stage_after_process_exit(
        tmp_path, False, False, False, structured="tool_valid", closed_fault=fault
    )


@pytest.mark.parametrize("action", ["complete", "continue", "interrupt", "fail"])
@pytest.mark.parametrize("after_promotion", [False, True])
def test_worker_model_replay_runs_registered_stop_policy(tmp_path, action, after_promotion):
    repository = Path(__file__).resolve().parents[2]
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.core.test_verified_task_worker import "
            "_crash_worker_before_model_promotion; "
            "import sys; _crash_worker_before_model_promotion("
            "sys.argv[1], stop_action=sys.argv[2], after_promotion=sys.argv[3] == 'True')",
            str(tmp_path),
            action,
            str(after_promotion),
        ],
        cwd=repository,
        env={**os.environ, "PYTHONPATH": str(repository / "src")},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert child.returncode == 73, (child.stdout, child.stderr)

    async def scenario():
        sessions = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        tasks = SQLiteTaskStore(tmp_path / "tasks.sqlite")
        try:
            admission = await tasks.load_latest_work_attempt_admission("model-promotion-crash")
            active = await sessions.load_active_model_completion_stage(admission.session_id)
            if active is not None:
                original_completion = active.stage.publication.events[0]
            else:
                [record] = await sessions.query_events(
                    EventQuery(
                        session_id=admission.session_id, event_types={EventType.MODEL_COMPLETED}
                    )
                )
                original_completion = record.event
            await asyncio.sleep(
                max(0, (admission.claim.lease_expires_at - datetime.now(UTC)).total_seconds())
                + 0.05
            )
            policy = _ReplayStopPolicy(action)
            app = CayuApp(
                session_store=sessions,
                task_store=tasks,
                enable_logging=False,
                loop_policies=[policy],
            )
            provider = _RecordingProvider()
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            contract = _contract()
            verifier = RecordingVerifier(_accepted_decision())
            app.register_completion_verifier(contract.verifier, verifier)
            app.register_completion_result_resolver(
                contract.result_resolver, _Resolver(_task_result())
            )
            handler = _StaticHandler()
            async with VerifiedTaskWorker(app, handler, worker_id="replacement") as worker:
                assert (
                    await asyncio.wait_for(
                        worker.run(max_tasks=1), _RECOVERY_COMPLETION_TIMEOUT_SECONDS
                    )
                    == 1
                )
            assert policy.requests[0] == (
                1,
                original_completion.payload["model_step_id"],
                original_completion.payload["model_attempt_id"],
                "verified work test response",
            )
            follow_up = action == "continue"
            accepted = action in {"complete", "continue"}
            assert len(policy.requests) == 1 + int(follow_up)
            assert len(provider.requests) == int(follow_up)
            if follow_up:
                assert policy.requests[1][0] == 2
                assert policy.requests[1][1:3] != policy.requests[0][1:3]
                assert (
                    sum(
                        message.role.value == "assistant"
                        for message in provider.requests[0].messages
                    )
                    == 1
                )
            transcript = await sessions.load_transcript(admission.session_id)
            assert sum(message.role.value == "assistant" for message in transcript) == (
                1 + int(follow_up)
            )
            assert handler.preparations == []
            assert len(handler.proposals) == int(accepted)
            assert len(verifier.requests) == int(accepted)
            assert (await tasks.load_task(admission.task_id)).status is (
                TaskStatus.COMPLETED if accepted else TaskStatus.NEEDS_ATTENTION
            )
            current = await tasks.load_latest_work_attempt_admission(admission.task_id)
            assert current.attempt_id == admission.attempt_id
            receipt = await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
            assert receipt.retired_contract_binding is accepted
        finally:
            await tasks.close()
            await sessions.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("tool_effect", "call_count", "execution_error", "after_publication"),
    [
        (False, 1, False, False),
        (True, 1, False, False),
        (True, 2, False, False),
        (True, 1, True, False),
        (True, 1, True, True),
    ],
    ids=[
        "model",
        "tool",
        "tool-successful-prefix",
        "recorded-unknown-precommit",
        "recorded-unknown-postcommit",
    ],
)
def test_worker_keeps_unknown_model_dispatch_fenced_after_process_exit(
    tmp_path,
    tool_effect,
    call_count,
    execution_error,
    after_publication,
    dynamic_secret=False,
    store_factory=None,
):
    repository = Path(__file__).resolve().parents[2]
    factory = store_factory or VerifiedWorkerStoreFactory("sqlite", tmp_path)
    child_environment = {**os.environ, "PYTHONPATH": str(repository / "src")}
    child_environment.pop("CAYU_TEST_VERIFIED_WORKER_DSN", None)
    if factory.postgres_dsn is not None:
        child_environment["CAYU_TEST_VERIFIED_WORKER_DSN"] = factory.postgres_dsn
    script = (
        "from tests.core.test_verified_task_worker import _crash_worker_tool_publication; "
        "import os, sys; _crash_worker_tool_publication("
        "sys.argv[1], after_publication=sys.argv[4] == 'True', "
        "crash_before_terminal=sys.argv[3] != 'True', execution_error=sys.argv[3] == 'True', "
        "call_count=int(sys.argv[2]), dynamic_secret=sys.argv[5] == 'True', "
        "hook_mode='pass' if sys.argv[5] == 'True' else 'none', "
        "postgres_dsn=os.environ.get('CAYU_TEST_VERIFIED_WORKER_DSN'))"
        if tool_effect
        else "from tests.core.test_verified_task_worker import "
        "_crash_worker_before_model_promotion; "
        "import os, sys; _crash_worker_before_model_promotion(sys.argv[1], crash_in_flight=True, "
        "postgres_dsn=os.environ.get('CAYU_TEST_VERIFIED_WORKER_DSN'))"
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            str(call_count),
            str(execution_error),
            str(after_publication),
            str(dynamic_secret),
        ],
        cwd=repository,
        env=child_environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert child.returncode == (75 if execution_error else 76 if tool_effect else 73), (
        child.stdout,
        child.stderr,
    )
    assert _RECOVERY_SECRET_CANARY not in child.stdout + child.stderr

    async def scenario():
        sessions, tasks = factory()
        try:
            admission = await tasks.load_latest_work_attempt_admission(
                "tool-publication-crash" if tool_effect else "model-promotion-crash"
            )
            active = await sessions.load_active_model_completion_stage(admission.session_id)
            checkpoint = await sessions.load_checkpoint(admission.session_id)
            if execution_error:
                assert active is None
                [terminal] = await sessions.query_events(
                    EventQuery(
                        session_id=admission.session_id, event_types={EventType.TOOL_CALL_FAILED}
                    )
                )
                assert terminal.event.payload["tool_call_id"] == "record-effect"
                assert terminal.event.payload["outcome_unknown"] is True
                assert terminal.event.payload["manual_reconciliation_required"] is True
                assert terminal.event.payload["tool_effect"] == ToolEffect.EXTERNAL.value
                receipt = await sessions.load_runtime_publication_receipt(
                    admission.session_id, f"tool-round:{terminal.event.payload['tool_round_id']}"
                )
                assert (receipt is not None) is after_publication
                assert (tmp_path / "effects.txt").read_text(encoding="utf-8") == "effect\n"
            elif tool_effect:
                assert active is None
                pending = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
                call_ids = {"record-effect"}
                if call_count == 2:
                    call_ids.add("record-effect-2")
                lifecycle = await sessions.load_tool_round_lifecycle_events_for_round(
                    admission.session_id,
                    sorted(call_ids),
                    tool_round_identity=tool_round_recovery.pending_tool_round_identity(pending),
                )
                outcomes, started_ids = tool_round_recovery.recorded_tool_outcomes(
                    events=lifecycle, pending_round=pending
                )
                assert set(outcomes) == ({"record-effect"} if call_count == 2 else set())
                if dynamic_secret:
                    assert pending.assistant_publication.secret_resolution_scope == "dynamic"
                    assert set(pending.assistant_publication.covered_tool_call_ids) == {
                        "record-effect"
                    }
                    assert outcomes["record-effect"].result.is_error
                    assert outcomes["record-effect"].result.structured["outcome_unknown"] is True
                    assert any(
                        staged.event.payload.get("secret_scope_incomplete") is True
                        for staged in tool_round_recovery.staged_terminal_records(pending)
                    )
                elif call_count == 2:
                    assert outcomes["record-effect"].result.structured == {"recorded": True}
                assert started_ids == call_ids
                assert (tmp_path / "effects.txt").read_text(encoding="utf-8") == (
                    "effect\n" * call_count
                )
            else:
                assert active.stage.state == "in_flight"
                assert (
                    await sessions.load_model_completion_stage_dispatch(
                        admission.session_id, active.stage.stage_id
                    )
                    is not None
                )
            await wait_for_verified_worker_lease_expiry(tasks, admission.claim.lease_expires_at)
            app = CayuApp(
                session_store=sessions,
                task_store=tasks,
                enable_logging=False,
                runtime_hooks=[_RecordedRecoveryHook(tmp_path / "hooks.txt", "pass")]
                if dynamic_secret
                else [],
            )
            if dynamic_secret:
                _register_recovery_vault(app)
            provider = _ToolReplayProvider(call_count) if tool_effect else _RecordingProvider()
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="worker", model="verified-work-test-model"),
                tools=[
                    _RecordedExternalEffect(
                        tmp_path / "effects.txt",
                        execution_error=execution_error,
                        dynamic_secret=dynamic_secret,
                    )
                ]
                if tool_effect
                else [],
            )
            contract = _contract()
            verifier = RecordingVerifier(_accepted_decision())
            app.register_completion_verifier(contract.verifier, verifier)
            app.register_completion_result_resolver(
                contract.result_resolver, _Resolver(_task_result())
            )
            handler = _StaticHandler()
            stop = asyncio.Event()
            async with VerifiedTaskWorker(
                app, handler, worker_id="replacement", poll_interval_s=0.01
            ) as worker:
                discover = worker._discover_unfinished_attempt
                scans = 0

                async def observe_scan():
                    nonlocal scans
                    result = await discover()
                    scans += 1
                    if scans == 3:
                        stop.set()
                    return result

                worker._discover_unfinished_attempt = observe_scan
                assert await asyncio.wait_for(worker.run(stop=stop), 10) == 0
                assert scans >= 3
            assert await tasks.load_latest_work_attempt_admission(admission.task_id) == admission
            assert await sessions.load_active_model_completion_stage(admission.session_id) == active
            if tool_effect:
                assert await sessions.load_checkpoint(admission.session_id) == checkpoint
                assert (tmp_path / "effects.txt").read_text(encoding="utf-8") == (
                    "effect\n" * call_count
                )
            assert await tasks.load_completion_proposal_for_attempt(admission.attempt_id) is None
            assert (await tasks.load_task(admission.task_id)).status is TaskStatus.RUNNING
            assert handler.preparations == []
            assert provider.requests == []
            assert verifier.requests == []
            if dynamic_secret:
                assert not (tmp_path / "hooks.txt").exists()
                for record in await sessions.query_events(
                    EventQuery(session_id=admission.session_id)
                ):
                    assert _RECOVERY_SECRET_CANARY not in record.event.model_dump_json()
                for message in await sessions.load_transcript(admission.session_id):
                    assert _RECOVERY_SECRET_CANARY not in message.model_dump_json()
        finally:
            await tasks.close()
            await sessions.close()

    asyncio.run(scenario())


@pytest.mark.postgres_recovery
@pytest.mark.parametrize("backend", ["postgres"])
@pytest.mark.parametrize(
    ("tool_effect", "call_count", "execution_error", "after_publication", "dynamic_secret"),
    [
        (False, 1, False, False, False),
        (True, 1, False, False, False),
        (True, 2, False, False, False),
        (True, 1, True, False, False),
        (True, 1, True, True, False),
        (True, 2, False, False, True),
    ],
    ids=[
        "model",
        "tool",
        "tool-successful-prefix",
        "recorded-unknown-precommit",
        "recorded-unknown-postcommit",
        "incomplete-secret-scope",
    ],
)
def test_postgres_worker_keeps_unknown_dispatch_fenced_after_process_exit(
    tmp_path,
    backend,
    tool_effect,
    call_count,
    execution_error,
    after_publication,
    dynamic_secret,
    verified_worker_store_factory,
):
    test_worker_keeps_unknown_model_dispatch_fenced_after_process_exit(
        tmp_path,
        tool_effect,
        call_count,
        execution_error,
        after_publication,
        dynamic_secret=dynamic_secret,
        store_factory=verified_worker_store_factory,
    )


def test_worker_fences_incomplete_dynamic_secret_scope_after_process_exit(tmp_path):
    test_worker_keeps_unknown_model_dispatch_fenced_after_process_exit(
        tmp_path, True, 2, False, False, dynamic_secret=True
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize(
    "fault",
    ["prepare", "create", "activate", "prepare_delayed", "continue_prepare", "prepare_cancel"],
)
def test_worker_recovers_preparing_source_without_repeating_handler(
    backend, fault, verified_worker_store_factory, monkeypatch
):
    async def scenario():
        sessions, tasks = verified_worker_store_factory()
        provider = _RecordingProvider()
        continuing = fault == "continue_prepare"
        contract = _contract(
            continuation_policy=CompletionContinuationPolicy(
                rejection_action=CompletionRejectionAction.CONTINUE,
                max_attempts=3,
                max_repeated_gap_count=3,
            )
        )
        verifier = (
            _ContinueOnceVerifier(_rejected_decision())
            if continuing
            else RecordingVerifier(_accepted_decision())
        )

        def make_app():
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            app.register_completion_verifier(contract.verifier, verifier)
            app.register_completion_result_resolver(
                contract.result_resolver, _Resolver(_task_result())
            )
            return app

        try:
            await tasks.publish_work_contract(contract)
            task = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            handler = _StaticHandler()
            target, method = (
                (sessions, "create")
                if fault == "create"
                else (
                    tasks,
                    "activate_work_attempt_admission"
                    if fault == "activate"
                    else "prepare_work_attempt_admission",
                )
            )
            original = getattr(type(target), method)

            async def lose_ack(store, request, *args, **kwargs):
                result = await original(store, request, *args, **kwargs)
                if continuing and request.kind == "initial":
                    return result
                raise ConnectionError("owned admission operation committed before acknowledgement")

            with monkeypatch.context() as patch:
                patch.setattr(type(target), method, lose_ack)
                source_app = make_app()
                if fault == "create":

                    async def stop_before_activation(engine, **kwargs):
                        assert kwargs["session"].causal_budget_id == task.id
                        raise ConnectionError("created session reconciled before activation")

                    patch.setattr(
                        type(source_app._session_engine),
                        "_activate_runtime_work_attempt",
                        stop_before_activation,
                    )
                async with VerifiedTaskWorker(
                    source_app,
                    handler,
                    worker_id="original",
                    lease_seconds=5 if continuing else 3,
                    callback_timeout_seconds=0.5,
                ) as worker:
                    with pytest.raises(ConnectionError):
                        await worker.run(max_tasks=1)
            prepared = await tasks.load_latest_work_attempt_admission(task.id)
            assert prepared.state.value == ("active" if fault == "activate" else "preparing")
            assert prepared.source_request is not None
            assert prepared.kind == ("continuation" if continuing else "initial")
            assert (await sessions.load(prepared.session_id) is None) == fault.startswith("prepare")
            assert len(handler.preparations) == 1
            assert (
                len(provider.requests)
                == len(handler.proposals)
                == len(verifier.requests)
                == int(continuing)
            )
            await wait_for_verified_worker_lease_expiry(tasks, prepared.claim.lease_expires_at)
            if backend != "memory":
                await sessions.close()
                await tasks.close()
                sessions, tasks = verified_worker_store_factory()

            class RecoveryHandler(_StaticHandler):
                async def prepare(self, context):
                    raise AssertionError("Recovery must not prepare source again")

            replacement_handler = RecoveryHandler()
            creation_committed, release_creation = asyncio.Event(), asyncio.Event()
            original_create = type(sessions).create

            async def delay_creation(store, request, *args, **kwargs):
                result = await original_create(store, request, *args, **kwargs)
                creation_committed.set()
                await release_creation.wait()
                return result

            async with VerifiedTaskWorker(
                make_app(),
                replacement_handler,
                worker_id="replacement",
                lease_seconds=5,
                callback_timeout_seconds=0.5,
            ) as worker:
                with monkeypatch.context() as patch:
                    if fault in {"prepare_delayed", "prepare_cancel"}:
                        patch.setattr(type(sessions), "create", delay_creation)
                    running = asyncio.create_task(worker.run(max_tasks=1))
                    try:
                        if fault in {"prepare_delayed", "prepare_cancel"}:
                            await asyncio.wait_for(creation_committed.wait(), 10)
                            claimed = await tasks.load_work_attempt_admission(prepared.admission_id)
                            await wait_for_verified_worker_lease_expiry(
                                tasks,
                                claimed.claim.lease_expires_at + timedelta(seconds=0.15),
                            )
                            renewed = await tasks.load_work_attempt_admission(prepared.admission_id)
                            assert renewed.state.value == "preparing"
                            assert renewed.claim.generation == 2
                            assert renewed.claim.lease_expires_at > claimed.claim.lease_expires_at
                            assert not running.done()
                            with pytest.raises(WorkAttemptExecutionClaimLost):
                                await make_app().recover_work_attempt(
                                    WorkAttemptRecoveryRequest(
                                        admission_id=prepared.admission_id,
                                        claim_id="competing-claim",
                                        worker_id="competing",
                                        generation=3,
                                        lease_seconds=10,
                                    )
                                )
                        if fault == "prepare_cancel":
                            running.cancel("cancel preparing owner")
                            with pytest.raises(asyncio.CancelledError) as cancelled:
                                await running
                            assert cancelled.value.args == ("cancel preparing owner",)
                            assert running.cancelling() == 1 and running.cancelled()
                            assert (
                                provider.requests
                                == replacement_handler.proposals
                                == verifier.requests
                                == []
                            )
                            with pytest.raises(WorkAttemptExecutionClaimLost):
                                await make_app().recover_work_attempt(
                                    WorkAttemptRecoveryRequest(
                                        admission_id=prepared.admission_id,
                                        claim_id="competing-after-cancel",
                                        worker_id="competing",
                                        generation=3,
                                        lease_seconds=10,
                                    )
                                )
                            assert (await tasks.load_task(task.id)).status is TaskStatus.RUNNING
                            release_creation.set()
                            await worker.aclose()
                        else:
                            release_creation.set()
                            assert await asyncio.wait_for(running, 20) == 1
                    finally:
                        release_creation.set()
                        if not running.done():
                            running.cancel()
                        await asyncio.gather(running, return_exceptions=True)
            if fault == "prepare_cancel":
                abandoned = await tasks.load_work_attempt_admission(prepared.admission_id)
                await wait_for_verified_worker_lease_expiry(tasks, abandoned.claim.lease_expires_at)
                if backend != "memory":
                    await sessions.close()
                    await tasks.close()
                    sessions, tasks = verified_worker_store_factory()
                async with VerifiedTaskWorker(
                    make_app(),
                    replacement_handler,
                    worker_id="after-cancellation",
                    lease_seconds=10,
                    callback_timeout_seconds=0.5,
                ) as worker:
                    assert await asyncio.wait_for(worker.run(max_tasks=1), 20) == 1
            completed = await tasks.load_task(task.id)
            final = await tasks.load_work_attempt_admission(prepared.admission_id)
            assert completed.status is TaskStatus.COMPLETED
            assert final.claim.generation == (3 if fault == "prepare_cancel" else 2)
            assert final.source_request == prepared.source_request
            assert final.run_semantics == prepared.run_semantics
            assert final.prepare_request_sha256 == prepared.prepare_request_sha256
            assert len(replacement_handler.proposals) == 1
            assert len(provider.requests) == len(verifier.requests) == (2 if continuing else 1)
        finally:
            if backend != "memory":
                await sessions.close()
                await tasks.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize(
    ("outcome", "competition"),
    [
        ("completed", False),
        ("failed", False),
        ("interrupted", False),
        ("user_input", False),
        ("tool_approval", False),
        ("completed", True),
    ],
)
def test_worker_recovers_expired_completed_execution_without_redispatch(
    backend, outcome, competition, tmp_path, verified_worker_store_factory
):
    from cayu import AlwaysRequireApprovalToolPolicy
    from cayu.tools.user_input import UserInputTool

    async def scenario():
        sessions, tasks = verified_worker_store_factory()
        entered = asyncio.Event()
        running = None

        class SourceProvider(_RecordingProvider):
            async def stream(self, request):
                if outcome == "completed":
                    async for chunk in super().stream(request):
                        yield chunk
                elif outcome in {"user_input", "tool_approval"}:
                    self.requests.append(request)
                    yield ModelStreamEvent.tool_call(
                        id="pending-human-input",
                        name="ask_user" if outcome == "user_input" else "record_effect",
                        arguments={"question": "Choose the target environment."}
                        if outcome == "user_input"
                        else {},
                    )
                    yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                else:
                    self.requests.append(request)
                    entered.set()
                    if outcome == "failed":
                        raise RuntimeError("source execution failed")
                    await asyncio.Event().wait()

        try:
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            provider = SourceProvider()
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="worker", model="verified-work-test-model"),
                tools=[UserInputTool()]
                if outcome == "user_input"
                else [_RecordedExternalEffect(tmp_path / "unapproved-recovery-effect")]
                if outcome == "tool_approval"
                else [],
                **(
                    {"tool_policy": AlwaysRequireApprovalToolPolicy()}
                    if outcome == "tool_approval"
                    else {}
                ),
            )
            contract = _contract()
            await tasks.publish_work_contract(contract)
            task = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            admission = await app.admit_work_attempt(
                RunRequest(
                    agent_name="worker",
                    task_id=task.id,
                    session_id="completed-source-session",
                    messages=[Message.text("user", "Build the result.")],
                ),
                execution=WorkAttemptExecutionRequest(
                    admission_id="completed-source-admission",
                    claim_id="source-claim",
                    attempt_id="completed-source-attempt",
                    interaction_id="completed-source-interaction",
                    worker_id="source-worker",
                    generation=1,
                    lease_seconds=5,
                ),
            )

            async def execute():
                async for _ in app._execute_work_attempt(
                    WorkAttemptRunRequest(
                        admission_id=admission.admission_id,
                        claim_id=admission.claim.claim_id,
                        worker_id=admission.claim.worker_id,
                        generation=1,
                        lease_seconds=5,
                    )
                ):
                    pass

            running = asyncio.create_task(execute())
            if outcome == "interrupted":
                await asyncio.wait_for(entered.wait(), 5)
                await wait_for_verified_worker_lease_expiry(tasks, admission.claim.lease_expires_at)
                # A real dispatched provider is still in flight after expiry.
                # Discovery must neither acquire another generation nor stall
                # unrelated queue work while its release remains unproven.
                unrelated = await tasks.create_task(
                    TaskCreate(type="verified", work_contract=contract.reference())
                )
                observer = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
                observer.register_provider(_RecordingProvider(), default=True)
                observer.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
                observer.register_completion_verifier(
                    contract.verifier, RecordingVerifier(_accepted_decision())
                )
                observer.register_completion_result_resolver(
                    contract.result_resolver, _Resolver(_task_result())
                )
                async with VerifiedTaskWorker(
                    observer, _StaticHandler(), worker_id="observer"
                ) as observer_worker:
                    assert await asyncio.wait_for(observer_worker.run(max_tasks=1), 10) == 1
                assert (await tasks.load_task(unrelated.id)).status is TaskStatus.COMPLETED
                assert (
                    await tasks.load_work_attempt_admission(admission.admission_id)
                ).claim.generation == 1
                assert not running.done()
                running.cancel("stop source worker")
                with pytest.raises(asyncio.CancelledError):
                    await running
                assert running.cancelling() == 1 and running.cancelled()
            else:
                await asyncio.wait_for(running, 10)
            admission = await tasks.load_work_attempt_admission(admission.admission_id)
            session = await sessions.load(admission.session_id)
            human_pause = outcome in {"user_input", "tool_approval"}
            terminal_outcome = "interrupted" if human_pause else outcome
            assert session.status is SessionStatus(terminal_outcome)
            original_checkpoint = await sessions.load_checkpoint(session.id)
            if human_pause:
                assert f"pending_{outcome}" in original_checkpoint
            release = await app._session_engine.load_work_attempt_release_evidence(admission)
            assert await tasks.load_completion_proposal_for_attempt(admission.attempt_id) is None
            assert (await tasks.load_task(task.id)).status is TaskStatus.RUNNING
            # Real expiry, not a constructed claim-loss exception or mutated
            # store clock. The completed invocation remains durable throughout.
            await wait_for_verified_worker_lease_expiry(tasks, admission.claim.lease_expires_at)
            if backend != "memory":
                await tasks.close()
                await sessions.close()
                sessions, tasks = verified_worker_store_factory()
            restarted = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            verifier = RecordingVerifier(_accepted_decision())
            resolver = _Resolver(_task_result())
            restarted.register_completion_verifier(contract.verifier, verifier)
            restarted.register_completion_result_resolver(contract.result_resolver, resolver)
            handler = _StaticHandler()
            async with VerifiedTaskWorker(restarted, handler, worker_id="replacement") as worker:
                if not competition:
                    assert await asyncio.wait_for(worker.run(max_tasks=1), 10) == 1
                else:
                    other = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
                    other.register_completion_verifier(contract.verifier, verifier)
                    other.register_completion_result_resolver(contract.result_resolver, resolver)
                    ready, stop = asyncio.Event(), asyncio.Event()
                    arrivals = 0

                    def racing_recovery(original):
                        async def recover(request):
                            nonlocal arrivals
                            arrivals += 1
                            if arrivals == 2:
                                ready.set()
                            await ready.wait()
                            return await original(request)

                        return recover

                    restarted._claim_work_attempt_recovery = racing_recovery(
                        restarted._claim_work_attempt_recovery
                    )
                    other._claim_work_attempt_recovery = racing_recovery(
                        other._claim_work_attempt_recovery
                    )
                    async with VerifiedTaskWorker(
                        other, handler, worker_id="replacement", poll_interval_s=0.01
                    ) as competitor:
                        # Force both app instances to observe the unclaimed
                        # proposal before either can dispatch verification.
                        # Same worker label is deliberately not owner identity.
                        verification_ready = asyncio.Event()
                        verification_requests = []

                        def racing_verification(original):
                            async def prepare(proposal_id, contract):
                                result = await original(proposal_id, contract)
                                request, decision = result
                                if request is not None and decision is None:
                                    verification_requests.append(request)
                                    if len(verification_requests) == 2:
                                        verification_ready.set()
                                    await verification_ready.wait()
                                return result

                            return prepare

                        for item in (worker, competitor):
                            item._verification_request = racing_verification(
                                item._verification_request
                            )
                        owners = [
                            asyncio.create_task(item.run(stop=stop, max_tasks=1))
                            for item in (worker, competitor)
                        ]
                        try:
                            done, _ = await asyncio.wait(
                                owners, timeout=10, return_when=asyncio.FIRST_COMPLETED
                            )
                            assert done
                            assert next(iter(done)).result() == 1
                            stop.set()
                            # The loser may subsequently discover the released
                            # proposal and reconcile the same committed result.
                            # Counts are local handled outcomes, not a global
                            # once-only acknowledgement. Durable effects and
                            # collaborator calls are checked below.
                            counts = sorted(await asyncio.wait_for(asyncio.gather(*owners), 5))
                            assert counts in ([0, 1], [1, 1])
                            assert len(verification_requests) == 2
                            assert len({item.claim_id for item in verification_requests}) == 2
                        finally:
                            stop.set()
                            for operation in owners:
                                if not operation.done():
                                    operation.cancel()
                            await asyncio.gather(*owners, return_exceptions=True)
            final = await tasks.load_task(task.id)
            recovered = await tasks.load_work_attempt_admission(admission.admission_id)
            assert recovered.claim.generation == 2
            assert recovered.execution_entry == admission.execution_entry
            assert recovered.run_semantics == admission.run_semantics
            assert (
                await restarted._session_engine.load_work_attempt_release_evidence(recovered)
                == release
            )
            assert len(provider.requests) == 1
            assert not handler.preparations
            assert (
                len(handler.proposals)
                == len(verifier.requests)
                == (1 if outcome == "completed" else 0)
            )
            # The existing resolver contract is single-flight per app, not
            # across application instances. Concurrent immutable reads may
            # repeat; result publication and final settlement must not.
            assert len(resolver.requests) in (
                {1, 2} if competition else {1 if outcome == "completed" else 0}
            )
            if resolver.requests:
                assert all(value == resolver.requests[0] for value in resolver.requests)
            result_events = await sessions.query_events(
                EventQuery(
                    session_id=admission.session_id,
                    event_types={EventType.TASK_COMPLETION_RESULT_RESOLVED},
                    limit=10,
                )
            )
            assert len(result_events) == (1 if outcome == "completed" else 0)
            receipt = await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
            assert receipt.task == final
            assert final.status is (
                TaskStatus.COMPLETED if outcome == "completed" else TaskStatus.NEEDS_ATTENTION
            )
            if outcome != "completed":
                assert final.status_reason == f"work_contract_execution_{terminal_outcome}"
                assert receipt.request.kind == "runtime_stop"
            if human_pause:
                recovered_checkpoint = await sessions.load_checkpoint(session.id)
                assert (
                    recovered_checkpoint[f"pending_{outcome}"]
                    == original_checkpoint[f"pending_{outcome}"]
                )
                assert not (tmp_path / "unapproved-recovery-effect").exists()
            assert receipt.retired_contract_binding is (outcome == "completed")
            assert await tasks.list_unsettled_work_attempt_admissions() == []
        finally:
            if running is not None and not running.done():
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
            if backend != "memory":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("signal", ["timeout", "cancel", "deadline"])
@pytest.mark.parametrize("late_failure", [False, True])
@pytest.mark.parametrize("context_exit", [False, True])
def test_worker_close_retains_exact_verifier_until_settlement(
    backend, signal, late_failure, context_exit, verified_worker_store_factory, monkeypatch
):
    from cayu.runtime import verified_task_worker as worker_module

    if signal == "timeout":
        monkeypatch.setattr(worker_module, "_VERIFIER_TIMEOUT_SECONDS", 0.05)

    async def scenario():
        sessions, tasks = verified_worker_store_factory()
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        release = asyncio.Event()

        class ResistantVerifier(RecordingVerifier):
            async def verify(self, request):
                self.requests.append(request)
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await release.wait()
                    if late_failure:
                        raise ValueError("late worker verifier cleanup failed") from None
                    return _accepted_decision()

        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        provider = _RecordingProvider()
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        contract = _contract()
        verifier = ResistantVerifier(_accepted_decision())
        resolver = _Resolver(_task_result())
        app.register_completion_verifier(contract.verifier, verifier)
        app.register_completion_result_resolver(contract.result_resolver, resolver)
        await tasks.publish_work_contract(contract)
        task = await tasks.create_task(
            TaskCreate(type="verified", work_contract=contract.reference())
        )
        worker = VerifiedTaskWorker(
            app,
            _StaticHandler(),
            worker_id="retained-verifier",
            callback_timeout_seconds=0.2,
            max_elapsed_seconds=5 if signal == "deadline" else 3600,
        )

        async def invoke():
            if context_exit:
                async with worker:
                    return await worker.run(max_tasks=1)
            return await worker.run(max_tasks=1)

        run = asyncio.create_task(invoke())
        try:
            await asyncio.wait_for(entered.wait(), 10)
            if signal == "cancel":
                run.cancel("stop verified worker")
                with pytest.raises(asyncio.CancelledError) as caught:
                    await run
                assert caught.value.args == ("stop verified worker",)
                assert run.cancelling() == 1 and run.cancelled()
            elif context_exit:
                with pytest.raises(ExceptionGroup) as caught:
                    await run
                leaves = [
                    error
                    for error in iter_exception_tree(caught.value)
                    if not isinstance(error, BaseExceptionGroup)
                ]
                assert [type(error) for error in leaves] == [
                    CompletionVerifierExecutionError,
                    VerifiedTaskWorkerDraining,
                ]
            else:
                with pytest.raises(CompletionVerifierExecutionError, match="bounded execution"):
                    await run
            await asyncio.wait_for(cancelled.wait(), 5)
            assert worker._verification is not None
            if context_exit:
                with pytest.raises(RuntimeError, match="closed"):
                    await worker.run(max_tasks=1)
            else:
                unobserved = worker._verification
                with pytest.raises(ValueError, match="max_tasks"):
                    await worker.run(max_tasks=True)
                assert worker._verification is unobserved
                assert unobserved.observation is None
                with pytest.raises(VerifiedTaskWorkerDraining):
                    await worker.run(max_tasks=1)
            begin = asyncio.get_running_loop().time()
            with pytest.raises(VerifiedTaskWorkerDraining):
                await worker.aclose()
            assert asyncio.get_running_loop().time() - begin < 1.5
            retained = worker._verification
            assert retained is not None
            closing = asyncio.create_task(worker.aclose())
            await asyncio.sleep(0)
            closing.cancel("stop close wait")
            with pytest.raises(asyncio.CancelledError) as caught:
                await closing
            assert caught.value.args == ("stop close wait",)
            assert closing.cancelling() == 1 and closing.cancelled()
            assert worker._verification is retained
            observer = retained.observation
            assert observer is not None
            observer.cancel("interrupt observation only")
            assert observer.cancelling() == 1
            observed = await observer
            assert isinstance(observed.error, asyncio.CancelledError)
            assert observer.cancelling() == 0 and not observer.cancelled()
            with pytest.raises(RuntimeError, match="without caller cancellation"):
                await worker.aclose()
            assert asyncio.current_task().cancelling() == 0
            assert worker._verification is retained
            assert retained.observation is None
            assert len(provider.requests) == len(verifier.requests) == 1
            assert resolver.requests == []
            expected_status = (
                TaskStatus.NEEDS_ATTENTION if signal == "deadline" else TaskStatus.RUNNING
            )
            assert (await tasks.load_task(task.id)).status is expected_status
            admission = await tasks.load_latest_work_attempt_admission(task.id)
            before_cleanup = await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
            if signal == "deadline":
                assert before_cleanup.request.kind == "proposal_deadline_stop"
                assert not before_cleanup.retired_contract_binding
            else:
                assert before_cleanup is None
            if late_failure and context_exit:
                closing = asyncio.create_task(worker.aclose())
                await asyncio.sleep(0)
                observer = retained.observation
                assert observer is not None
                observer.add_done_callback(
                    lambda _done: closing.cancel("close cancellation alongside cleanup failure")
                )
                release.set()
                with pytest.raises(asyncio.CancelledError) as caught:
                    await closing
                assert closing.cancelling() == 1 and closing.cancelled()
                assert caught.value.args == ("close cancellation alongside cleanup failure",)
                assert isinstance(caught.value.__cause__, CompletionVerifierExecutionError)
                assert "late worker verifier cleanup" in str(caught.value.__cause__)
            elif late_failure:
                closing = [asyncio.create_task(worker.aclose()) for _ in range(2)]
                await asyncio.sleep(0)
                release.set()
                results = await asyncio.gather(*closing, return_exceptions=True)
                failures = [result for result in results if result is not None]
                assert len(failures) == 1
                assert isinstance(failures[0], CompletionVerifierExecutionError)
                assert "late worker verifier cleanup" in str(failures[0])
            else:
                release.set()
                await worker.aclose()
            assert worker._verification is None
            await worker.aclose()  # The late failure is not replayed by close.
            admission = await tasks.load_latest_work_attempt_admission(task.id)
            proposal = await tasks.load_completion_proposal_for_attempt(admission.attempt_id)
            assert await tasks.load_completion_decision_for_proposal(proposal.proposal_id) is None
            assert (
                await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
                == before_cleanup
            )
            assert (await tasks.load_task(task.id)).status is expected_status
            assert not app._completion_verifier_coordinator._adapter_tasks
            assert not app._completion_verifier_coordinator._adapter_capacity_reservations
            assert not app._completion_verifier_coordinator._draining_adapter_tasks
        finally:
            release.set()
            if not run.done():
                run.cancel()
                await asyncio.gather(run, return_exceptions=True)
            await worker.aclose()
            if backend != "memory":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("decision_first", [False, True])
def test_worker_concurrent_decision_and_expiry_have_one_durable_winner(
    backend, decision_first, verified_worker_store_factory, monkeypatch
):
    async def scenario():
        sessions, tasks = verified_worker_store_factory()
        contract = _contract()
        provider = _RecordingProvider()
        verifier = RecordingVerifier(_accepted_decision())
        resolver = _Resolver(_task_result())
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        app.register_completion_verifier(contract.verifier, verifier)
        app.register_completion_result_resolver(contract.result_resolver, resolver)
        replacement = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        replacement.register_completion_verifier(contract.verifier, verifier)
        replacement.register_completion_result_resolver(contract.result_resolver, resolver)
        decision_dispatched = asyncio.Event()
        decision_release = asyncio.Event()
        stop_dispatched = asyncio.Event()
        stop_release = asyncio.Event()
        publish = type(tasks).record_completion_decision
        settle = type(tasks).settle_work_attempt_lifecycle

        async def blocked_decision(store, request):
            decision_dispatched.set()
            await decision_release.wait()
            return await publish(store, request)

        async def blocked_stop(store, request):
            if request.kind == "proposal_deadline_stop":
                stop_dispatched.set()
                await stop_release.wait()
            return await settle(store, request)

        monkeypatch.setattr(type(tasks), "record_completion_decision", blocked_decision)
        monkeypatch.setattr(type(tasks), "settle_work_attempt_lifecycle", blocked_stop)
        await tasks.publish_work_contract(contract)
        task = await tasks.create_task(
            TaskCreate(type="verified", work_contract=contract.reference())
        )
        original = VerifiedTaskWorker(
            app, _StaticHandler(), worker_id="original", max_elapsed_seconds=5
        )
        recovered = VerifiedTaskWorker(replacement, _StaticHandler(), worker_id="replacement")
        runs = []
        try:
            first = asyncio.create_task(original.run(max_tasks=1))
            runs.append(first)
            await asyncio.wait_for(decision_dispatched.wait(), 10)
            admission = await tasks.load_latest_work_attempt_admission(task.id)
            await asyncio.sleep(
                max(
                    0,
                    (
                        admission.run_semantics.deadline_expires_at - datetime.now(UTC)
                    ).total_seconds(),
                )
                + 0.05
            )
            second = asyncio.create_task(recovered.run(max_tasks=1))
            runs.append(second)
            await asyncio.wait_for(stop_dispatched.wait(), 5)
            assert not first.done() and not second.done()
            assert await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id) is None
            if decision_first:
                decision_release.set()
                assert await asyncio.wait_for(first, 10) == 1
                stop_release.set()
                assert await asyncio.wait_for(second, 10) == 1
            else:
                stop_release.set()
                assert await asyncio.wait_for(second, 10) == 1
                decision_release.set()
                with pytest.raises(WorkCompletionConflict):
                    await asyncio.wait_for(first, 10)
            await original.aclose()
            await recovered.aclose()
            final = await tasks.load_task(task.id)
            receipt = await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
            proposal = await tasks.load_completion_proposal_for_attempt(admission.attempt_id)
            decision = await tasks.load_completion_decision_for_proposal(proposal.proposal_id)
            assert receipt.task == final
            assert final.status is (
                TaskStatus.COMPLETED if decision_first else TaskStatus.NEEDS_ATTENTION
            )
            assert receipt.request.kind == (
                "decision_application" if decision_first else "proposal_deadline_stop"
            )
            assert receipt.retired_contract_binding is decision_first
            assert (decision is not None) is decision_first
            assert len(provider.requests) == len(verifier.requests) == 1
            assert len(resolver.requests) == int(decision_first)
            assert await tasks.list_unsettled_work_attempt_admissions() == []
        finally:
            decision_release.set()
            stop_release.set()
            for run in runs:
                if not run.done():
                    run.cancel()
            await asyncio.gather(*runs, return_exceptions=True)
            await original.aclose()
            await recovered.aclose()
            if backend != "memory":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_worker_reclaims_only_expired_unadmitted_contract_preparation(backend, tmp_path):
    async def scenario():
        sessions = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "preparation-sessions.sqlite")
        )
        tasks = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "preparation-tasks.sqlite")
        )
        try:
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            app.register_provider(_RecordingProvider(), default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            contract = _contract()
            await tasks.publish_work_contract(contract)
            ordinary = await tasks.create_task(TaskCreate(type="ordinary"))
            task = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            ordinary_claim = await tasks.claim_task(
                "ordinary", TaskQuery(has_work_contract=False), lease_seconds=1
            )
            claimed = await tasks.claim_task(
                "abandoned-preparation", TaskQuery(has_work_contract=True), lease_seconds=1
            )
            assert claimed.id == task.id and ordinary_claim.id == ordinary.id
            await asyncio.sleep(1.05)
            app.register_completion_verifier(
                contract.verifier, RecordingVerifier(_accepted_decision())
            )
            app.register_completion_result_resolver(
                contract.result_resolver, _Resolver(_task_result())
            )
            handler = _StaticHandler()
            async with VerifiedTaskWorker(app, handler, worker_id="replacement") as worker:
                assert await asyncio.wait_for(worker.run(max_tasks=1), 10) == 1
            assert (await tasks.load_task(task.id)).status is TaskStatus.COMPLETED
            assert await tasks.load_task(ordinary.id) == ordinary_claim
            assert len(handler.preparations) == len(handler.proposals) == 1
        finally:
            if backend == "sqlite":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("fault", ["proposal", "decision"])
def test_worker_restart_discovers_published_work_without_redispatch(
    backend, fault, verified_worker_store_factory
):
    async def scenario():
        sessions, tasks = verified_worker_store_factory()
        try:
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            provider = _RecordingProvider()
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            contract = _contract()
            await tasks.publish_work_contract(contract)
            task = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            verifier = RecordingVerifier(_accepted_decision())
            resolver = _Resolver(_task_result())
            app.register_completion_verifier(contract.verifier, verifier)
            app.register_completion_result_resolver(contract.result_resolver, resolver)
            owner = app if fault == "proposal" else app._completion_verifier_coordinator
            method = "submit_work_attempt_proposal" if fault == "proposal" else "_verify"
            original = getattr(owner, method)

            async def lose_reply(*args, **kwargs):
                # The worker uses owned verification, not the public manual
                # wrapper. Lose its reply after the same real durable commit.
                await original(*args, **kwargs)
                raise ConnectionError("committed reply lost")

            setattr(owner, method, lose_reply)
            async with VerifiedTaskWorker(
                app, _StaticHandler(), worker_id="before-restart"
            ) as worker:
                with pytest.raises(ConnectionError, match="committed reply lost"):
                    await worker.run(max_tasks=1)
            admission = await tasks.load_latest_work_attempt_admission(task.id)
            proposal = await tasks.load_completion_proposal_for_attempt(admission.attempt_id)
            assert proposal is not None
            assert await tasks.load_completion_proposal_for_attempt("missing-attempt") is None
            assert len(provider.requests) == 1
            assert (await tasks.load_task(task.id)).status is TaskStatus.RUNNING
            if backend != "memory":
                await tasks.close()
                await sessions.close()
                sessions, tasks = verified_worker_store_factory()
            # No execution provider or agent is registered after restart: this
            # path must consume durable proposal/release evidence, not rerun.
            restarted = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            restarted.register_completion_verifier(contract.verifier, verifier)
            restarted.register_completion_result_resolver(contract.result_resolver, resolver)
            handler = _StaticHandler()
            async with VerifiedTaskWorker(
                restarted, handler, worker_id="after-restart", poll_interval_s=0.01
            ) as worker:
                assert await asyncio.wait_for(worker.run(max_tasks=1), 10) == 1
            final = await tasks.load_task(task.id)
            assert final.status is TaskStatus.COMPLETED
            assert not handler.preparations and not handler.proposals
            assert len(provider.requests) == len(verifier.requests) == len(resolver.requests) == 1
            assert (
                await tasks.load_completion_proposal_for_attempt(admission.attempt_id) == proposal
            )
            receipt = await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
            assert receipt.task == final
            assert receipt.retired_contract_binding
            assert (
                await tasks.load_active_work_contract_task_for_session(admission.session_id) is None
            )
            assert await tasks.list_unsettled_work_attempt_admissions() == []
        finally:
            if backend != "memory":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_worker_discovery_skips_live_verifier_and_handles_unrelated_work(backend, tmp_path):
    async def scenario():
        sessions = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "concurrent-sessions.sqlite")
        )
        tasks = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "concurrent-tasks.sqlite")
        )
        entered, release = asyncio.Event(), asyncio.Event()
        running = None

        class BlockingFirstVerifier(RecordingVerifier):
            async def verify(self, request):
                self.requests.append(request)
                if len(self.requests) == 1:
                    entered.set()
                    await release.wait()
                return self.decision

        try:
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            provider = _RecordingProvider()
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            contract = _contract()
            await tasks.publish_work_contract(contract)
            first = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            verifier = BlockingFirstVerifier(_accepted_decision())
            app.register_completion_verifier(contract.verifier, verifier)
            app.register_completion_result_resolver(
                contract.result_resolver, _Resolver(_task_result())
            )
            async with (
                VerifiedTaskWorker(app, _StaticHandler(), worker_id="first") as worker,
                VerifiedTaskWorker(app, _StaticHandler(), worker_id="second") as competitor,
            ):
                running = asyncio.create_task(worker.run(max_tasks=1))
                await asyncio.wait_for(entered.wait(), 10)
                second = await tasks.create_task(
                    TaskCreate(type="verified", work_contract=contract.reference())
                )
                assert await asyncio.wait_for(competitor.run(max_tasks=1), 10) == 1
                assert not running.done()
                assert (await tasks.load_task(first.id)).status is TaskStatus.RUNNING
                assert (await tasks.load_task(second.id)).status is TaskStatus.COMPLETED
                assert len(verifier.requests) == 2
                release.set()
                assert await asyncio.wait_for(running, 10) == 1
                assert (await tasks.load_task(first.id)).status is TaskStatus.COMPLETED
                assert len(provider.requests) == len(verifier.requests) == 2
        finally:
            release.set()
            if running is not None and not running.done():
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
            if backend == "sqlite":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize(
    "outcome",
    [
        "accepted",
        "continue",
        "interrupt",
        "blocked",
        "needs_review",
        "attempt_limit",
        "repeated_gap",
    ],
)
def test_worker_run_claims_only_contract_queue_and_completes_lifecycle(
    backend, outcome, verified_worker_store_factory
):
    async def scenario():
        sessions, tasks = verified_worker_store_factory()
        try:
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            provider = _RecordingProvider()
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            contract = _contract(
                continuation_policy=CompletionContinuationPolicy(
                    rejection_action=CompletionRejectionAction.INTERRUPT
                    if outcome == "interrupt"
                    else CompletionRejectionAction.CONTINUE,
                    max_attempts=2 if outcome == "attempt_limit" else 10,
                    max_repeated_gap_count=1 if outcome == "repeated_gap" else 10,
                )
            )
            await tasks.publish_work_contract(contract)
            ordinary = await tasks.create_task(TaskCreate(type="ordinary"))
            task = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            decision = _accepted_decision() if outcome == "accepted" else _rejected_decision()
            if outcome in {"blocked", "needs_review"}:
                decision = decision.model_copy(update={"verdict": CompletionVerdict(outcome)})
            verifier = (_ContinueOnceVerifier if outcome == "continue" else RecordingVerifier)(
                decision
            )
            resolver = _Resolver(_task_result())
            app.register_completion_verifier(contract.verifier, verifier)
            app.register_completion_result_resolver(contract.result_resolver, resolver)
            handler = _StaticHandler()
            async with VerifiedTaskWorker(app, handler, worker_id="real-worker") as worker:
                assert await worker.run(max_tasks=1) == 1
            final = await tasks.load_task(task.id)
            expected = {
                "accepted": TaskStatus.COMPLETED,
                "continue": TaskStatus.COMPLETED,
                "interrupt": TaskStatus.PAUSED,
                "blocked": TaskStatus.BLOCKED,
                "needs_review": TaskStatus.NEEDS_ATTENTION,
                "attempt_limit": TaskStatus.NEEDS_ATTENTION,
                "repeated_gap": TaskStatus.NEEDS_ATTENTION,
            }[outcome]
            assert final.status is expected
            if outcome in {"attempt_limit", "repeated_gap"}:
                assert (
                    final.status_reason
                    == {
                        "attempt_limit": "work_contract_attempt_limit",
                        "repeated_gap": "work_contract_repeated_gap_limit",
                    }[outcome]
                )
            assert (await tasks.load_task(ordinary.id)).status is TaskStatus.PENDING
            expected_runs = 2 if outcome in {"continue", "attempt_limit", "repeated_gap"} else 1
            assert len(provider.requests) == expected_runs
            assert len(verifier.requests) == expected_runs
            assert len(handler.preparations) == 1
            assert len(handler.proposals) == expected_runs
            assert len(resolver.requests) == (1 if expected is TaskStatus.COMPLETED else 0)
            admission = await tasks.load_latest_work_attempt_admission(task.id)
            receipt = await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
            assert receipt.task == final
            assert receipt.retired_contract_binding is (expected is TaskStatus.COMPLETED)
            assert (
                await tasks.load_active_work_contract_task_for_session(admission.session_id) is None
            ) is (expected is TaskStatus.COMPLETED)
        finally:
            if backend != "memory":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("change", ["prose", "reason_code", "gap_code"])
def test_worker_repeated_gap_uses_semantic_identity_not_prose(
    backend, change, verified_worker_store_factory
):
    from cayu import VerifiedTaskWorker as PublicWorker

    class ChangingGapVerifier(RecordingVerifier):
        async def verify(self, request):
            self.requests.append(request)
            ordinal = request.attempt.ordinal
            if ordinal == 3:
                return _accepted_decision()
            assert ordinal in {1, 2}
            decision = _rejected_decision()
            summary = "First explanation." if ordinal == 1 else "Different explanation. " * 100
            outcome = decision.criterion_outcomes[0].model_copy(
                update={
                    "summary": summary,
                    "reason_code": "package.unavailable"
                    if ordinal == 2 and change == "reason_code"
                    else "package.missing",
                }
            )
            gap = decision.gaps[0].model_copy(
                update={
                    "summary": summary,
                    "code": "package.unavailable"
                    if ordinal == 2 and change == "gap_code"
                    else "package.missing",
                }
            )
            return decision.model_copy(update={"criterion_outcomes": (outcome,), "gaps": (gap,)})

    async def scenario():
        sessions, tasks = verified_worker_store_factory()
        try:
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            provider = _RecordingProvider()
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            contract = _contract(
                continuation_policy=CompletionContinuationPolicy(
                    rejection_action=CompletionRejectionAction.CONTINUE,
                    max_attempts=4,
                    max_repeated_gap_count=1,
                )
            )
            await tasks.publish_work_contract(contract)
            task = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            verifier = ChangingGapVerifier(_rejected_decision())
            resolver = _Resolver(_task_result())
            app.register_completion_verifier(contract.verifier, verifier)
            app.register_completion_result_resolver(contract.result_resolver, resolver)
            handler = _StaticHandler()
            async with PublicWorker(app, handler, worker_id="semantic-gap-worker") as worker:
                assert await worker.run(max_tasks=1) == 1
            decisions = [
                await tasks.load_completion_decision_for_proposal(request.proposal.proposal_id)
                for request in verifier.requests
            ]
            first, second = decisions[:2]
            assert first.gaps[0].summary != second.gaps[0].summary
            assert first.criterion_outcomes[0].summary != second.criterion_outcomes[0].summary
            assert (first.gap_fingerprint == second.gap_fingerprint) is (change == "prose")
            final = await tasks.load_task(task.id)
            admission = await tasks.load_latest_work_attempt_admission(task.id)
            receipt = await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
            expected_runs = 2 if change == "prose" else 3
            assert len(provider.requests) == len(verifier.requests) == expected_runs
            assert len(handler.preparations) == 1
            assert len(handler.proposals) == expected_runs
            assert receipt.task == final
            if change == "prose":
                assert final.status is TaskStatus.NEEDS_ATTENTION
                assert final.status_reason == "work_contract_repeated_gap_limit"
                assert not receipt.retired_contract_binding
                assert resolver.requests == []
            else:
                assert final.status is TaskStatus.COMPLETED
                assert receipt.retired_contract_binding
                assert len(resolver.requests) == 1
                assert admission.continuation.decision == second
                assert admission.continuation.gap_fingerprint == second.gap_fingerprint
                assert admission.continuation.decision.gaps == second.gaps
            assert (
                await tasks.load_active_work_contract_task_for_session(admission.session_id) is None
            ) is (change != "prose")
        finally:
            if backend != "memory":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "missing",
    [
        "record_work_attempt_execution_stop",
        "enter_work_attempt_execution",
        "load_active_work_contract_task_for_session",
        "load_work_attempt_lifecycle_receipt",
        "load_completion_proposal_for_attempt",
    ],
)
def test_worker_rejects_partial_store_before_claim_or_callback(missing):
    async def scenario():
        partial_type = type(
            "PartialWorkerStore",
            (InMemoryTaskStore,),
            {
                "supports_verified_task_worker": True,
                "verified_work_mutations_are_cancellation_quiescent": True,
                missing: getattr(TaskStore, missing),
            },
        )
        tasks = partial_type()
        contract = _contract()
        await tasks.publish_work_contract(contract)
        task = await tasks.create_task(
            TaskCreate(type="verified", work_contract=contract.reference())
        )
        app = CayuApp(task_store=tasks, session_store=InMemorySessionStore(), enable_logging=False)
        handler = _StaticHandler()
        async with VerifiedTaskWorker(app, handler, worker_id="partial-worker") as worker:
            with pytest.raises(NotImplementedError, match="complete verified-worker"):
                await worker.run(max_tasks=1)
        assert await tasks.load_task(task.id) == task
        assert not handler.preparations
        assert not handler.proposals

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "backend,restart",
    [
        ("memory", False),
        ("sqlite", False),
        ("sqlite", True),
        ("postgres", False),
        ("postgres", True),
    ],
)
@pytest.mark.parametrize("maximum", ["1.00", "1.20", "1.21"])
def test_worker_preserves_priced_causal_budget_across_attempts(
    backend, restart, maximum, verified_worker_store_factory
):
    from cayu import BudgetReservation

    class PricedProvider(_RecordingProvider):
        async def stream(self, request):
            self.requests.append(request)
            yield ModelStreamEvent.text_delta("Candidate result.")
            yield ModelStreamEvent.completed(
                {"finish_reason": "stop", "usage": {"input_tokens": 100, "output_tokens": 0}}
            )

    class Handler(_StaticHandler):
        async def prepare(self, context):
            request = await super().prepare(context)
            limit = BudgetLimit(
                scope="causal",
                key=context.task.id,
                max_estimated_cost=Decimal(maximum),
                pricing=PriceBook(
                    prices=(
                        ModelPrice.fixed(
                            provider_name=_RecordingProvider.name,
                            model="verified-work-test-model",
                            input_per_million=Decimal("6000"),
                            output_per_million=Decimal("0"),
                        ),
                    )
                ),
                reservation=BudgetReservation(max_input_tokens=100, max_output_tokens=0),
            )
            return request.model_copy(update={"budget_limits": (limit,)})

    async def scenario():
        sessions, tasks = verified_worker_store_factory()
        ledger = verified_worker_store_factory.budget_ledger()
        try:
            app = CayuApp(
                session_store=sessions, task_store=tasks, budget_ledger=ledger, enable_logging=False
            )
            provider = PricedProvider()
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            contract = _contract(
                continuation_policy=CompletionContinuationPolicy(
                    rejection_action=CompletionRejectionAction.CONTINUE,
                    max_attempts=3,
                    max_repeated_gap_count=3,
                )
            )
            await tasks.publish_work_contract(contract)
            task = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            verifier = _ContinueOnceVerifier(_rejected_decision())
            app.register_completion_verifier(contract.verifier, verifier)
            resolver = _Resolver(_task_result())
            app.register_completion_result_resolver(contract.result_resolver, resolver)
            handler = Handler()
            if restart:

                async def stop_before_continuation(*args, **kwargs):
                    raise ConnectionError("restart before priced continuation")

                app._continue_verified_task = stop_before_continuation
            async with VerifiedTaskWorker(app, handler, worker_id="priced-worker") as worker:
                if restart:
                    with pytest.raises(Exception) as failure:
                        await asyncio.wait_for(worker.run(max_tasks=1), 20)
                    assert any(
                        isinstance(error, ConnectionError)
                        and "restart before priced continuation" in str(error)
                        for error in iter_exception_tree(failure.value)
                    )
                else:
                    assert await asyncio.wait_for(worker.run(max_tasks=1), 20) == 1
            if restart:
                assert (
                    len(provider.requests) == len(verifier.requests) == len(handler.proposals) == 1
                )
                await sessions.close()
                await tasks.close()
                await ledger.close()
                sessions, tasks = verified_worker_store_factory()
                ledger = verified_worker_store_factory.budget_ledger()
                app = CayuApp(
                    session_store=sessions,
                    task_store=tasks,
                    budget_ledger=ledger,
                    enable_logging=False,
                )
                replacement_provider = PricedProvider()
                replacement_verifier = _ContinueOnceVerifier(_rejected_decision())
                replacement_handler = Handler()
                replacement_resolver = _Resolver(_task_result())
                app.register_provider(replacement_provider, default=True)
                app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
                app.register_completion_verifier(contract.verifier, replacement_verifier)
                app.register_completion_result_resolver(
                    contract.result_resolver, replacement_resolver
                )
                async with VerifiedTaskWorker(
                    app, replacement_handler, worker_id="replacement-priced-worker"
                ) as worker:
                    assert await asyncio.wait_for(worker.run(max_tasks=1), 20) == 1
                assert not replacement_handler.preparations
                provider.requests.extend(replacement_provider.requests)
                verifier.requests.extend(replacement_verifier.requests)
                handler.proposals.extend(replacement_handler.proposals)
                resolver.requests.extend(replacement_resolver.requests)
            final = await tasks.load_task(task.id)
            permitted = maximum == "1.21"
            # Reservation admits equality, but the ordinary post-call budget
            # check interrupts at the cap before a completion proposal.
            assert len(provider.requests) == (1 if maximum == "1.00" else 2)
            assert len(verifier.requests) == len(handler.proposals) == (2 if permitted else 1)
            assert len(handler.preparations) == 1
            assert len(resolver.requests) == int(permitted)
            assert final.status is (
                TaskStatus.COMPLETED if permitted else TaskStatus.NEEDS_ATTENTION
            )
            admission = await tasks.load_latest_work_attempt_admission(task.id)
            assert admission.run_semantics.causal_budget_id == task.id
            assert admission.run_semantics.budget_limits[0].key == task.id
            if not permitted:
                assert final.status_reason == "work_contract_budget_limit"
                assert admission.execution_stop.request.reason == "budget_limit"
                assert (
                    await tasks.load_completion_proposal_for_attempt(admission.attempt_id) is None
                )
            receipt = await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
            assert receipt.task == final
            assert receipt.retired_contract_binding is permitted
        finally:
            if backend != "memory":
                await sessions.close()
                await tasks.close()
                await ledger.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize(
    "budget_source,restart",
    [
        *(
            (source, restart)
            for source in ("request", "app", "deadline")
            for restart in (False, True)
        ),
        ("caller_cancel", False),
    ],
)
def test_worker_retains_source_limit_stop_without_proposal(
    backend, budget_source, restart, verified_worker_store_factory, monkeypatch
):
    limit = BudgetLimit(
        scope="run" if budget_source == "request" else "app",
        max_estimated_cost=Decimal("1"),
        pricing=PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="unregistered-provider",
                    model="unregistered-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        ),
    )

    class Handler(_StaticHandler):
        async def prepare(self, context):
            request = await super().prepare(context)
            return request.model_copy(
                update={
                    "budget_limits": (limit,) if budget_source == "request" else (),
                }
            )

    async def scenario():
        sessions, tasks = verified_worker_store_factory()
        provider_lifecycle = []
        provider_entered = asyncio.Event()
        if budget_source in {"deadline", "caller_cancel"}:

            async def deadline_stream(provider, request):
                provider.requests.append(request)
                provider_lifecycle.append("entered")
                provider_entered.set()
                try:
                    await asyncio.Event().wait()
                    yield None
                finally:
                    await asyncio.sleep(0)
                    provider_lifecycle.append("settled")

            monkeypatch.setattr(_RecordingProvider, "stream", deadline_stream)
        stop_reason = "elapsed_limit" if budget_source == "deadline" else "budget_limit"
        status_reason = "work_contract_" + stop_reason
        try:
            policy = BudgetPolicy(limits=(limit,) if budget_source == "app" else ())
            app = CayuApp(
                task_store=tasks, session_store=sessions, budget_policy=policy, enable_logging=False
            )
            provider = _RecordingProvider()
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            contract = _contract()
            await tasks.publish_work_contract(contract)
            task = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            verifier = RecordingVerifier(_accepted_decision())
            app.register_completion_verifier(contract.verifier, verifier)
            handler = Handler()
            settle = type(tasks).settle_work_attempt_lifecycle
            if restart:

                async def fail_before_settlement(store, request):
                    raise ConnectionError("stop before lifecycle settlement")

                monkeypatch.setattr(
                    type(tasks), "settle_work_attempt_lifecycle", fail_before_settlement
                )
            async with VerifiedTaskWorker(
                app,
                handler,
                worker_id="budget-worker",
                lease_seconds=5,
                callback_timeout_seconds=1,
                max_elapsed_seconds=3 if budget_source == "deadline" else 3600,
            ) as worker:
                if budget_source == "caller_cancel":
                    running = asyncio.create_task(worker.run(max_tasks=1))
                    try:
                        await asyncio.wait_for(provider_entered.wait(), 10)
                        running.cancel("caller cancellation is not expiry")
                        with pytest.raises(asyncio.CancelledError) as cancelled:
                            await running
                        assert cancelled.value.args == ("caller cancellation is not expiry",)
                        assert running.cancelling() == 1 and running.cancelled()
                        if worker._running is not None:
                            done, _ = await asyncio.wait({worker._running}, timeout=10)
                            assert done
                    finally:
                        if not running.done():
                            running.cancel()
                            await asyncio.gather(running, return_exceptions=True)
                elif restart:
                    with pytest.raises(Exception) as failed:
                        await worker.run(max_tasks=1)
                    assert any(
                        "stop before lifecycle settlement" in str(error)
                        for error in iter_exception_tree(failed.value)
                    )
                else:
                    assert await worker.run(max_tasks=1) == 1
            if budget_source == "caller_cancel":
                pending = await tasks.load_latest_work_attempt_admission(task.id)
                assert pending.execution_stop is None
                assert await tasks.load_work_attempt_lifecycle_receipt(pending.admission_id) is None
                assert (await tasks.load_task(task.id)).status is TaskStatus.RUNNING
                assert provider_lifecycle == ["entered", "settled"]
                assert handler.proposals == verifier.requests == []
                return
            if restart:
                monkeypatch.setattr(type(tasks), "settle_work_attempt_lifecycle", settle)
                pending = await tasks.load_latest_work_attempt_admission(task.id)
                assert pending.execution_stop.request.reason == stop_reason
                assert await tasks.load_work_attempt_lifecycle_receipt(pending.admission_id) is None
                assert (await tasks.load_task(task.id)).status is TaskStatus.RUNNING
                if backend != "memory":
                    await tasks.close()
                    await sessions.close()
                    sessions, tasks = verified_worker_store_factory()
                app = CayuApp(
                    task_store=tasks,
                    session_store=sessions,
                    budget_policy=policy,
                    enable_logging=False,
                )
                app.register_provider(provider, default=True)
                app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
                app.register_completion_verifier(contract.verifier, verifier)
                await asyncio.sleep(
                    max(0, (pending.claim.lease_expires_at - datetime.now(UTC)).total_seconds())
                    + 0.05
                )
                async with VerifiedTaskWorker(
                    app,
                    handler,
                    worker_id="replacement-budget-worker",
                    lease_seconds=5,
                    callback_timeout_seconds=1,
                ) as worker:
                    assert await worker.run(max_tasks=1) == 1
            final = await tasks.load_task(task.id)
            assert final.status is TaskStatus.NEEDS_ATTENTION
            assert final.status_reason == status_reason
            admission = await tasks.load_latest_work_attempt_admission(task.id)
            assert admission.execution_stop.request.reason == stop_reason
            assert admission.execution_stop.request.generation == 1
            assert admission.claim.generation == (2 if restart else 1)
            assert len(handler.preparations) == 1
            receipt = await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
            assert receipt.task == final
            assert receipt.request.stop_reason == status_reason
            assert (
                receipt.request.release_evidence.run_epoch
                == admission.execution_entry.request.run_epoch
            )
            assert not receipt.retired_contract_binding
            assert handler.proposals == verifier.requests == []
            if budget_source == "deadline":
                assert len(provider.requests) == 1
                assert provider_lifecycle == ["entered", "settled"]
            else:
                assert provider.requests == []
            assert await tasks.load_completion_proposal_for_attempt(admission.attempt_id) is None
            if backend != "memory":
                await tasks.close()
                await sessions.close()
                sessions, tasks = verified_worker_store_factory()
                assert await tasks.load_latest_work_attempt_admission(task.id) == admission
                assert (
                    await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
                    == receipt
                )
                assert await tasks.load_task(task.id) == final
        finally:
            if backend != "memory":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "fault",
    [
        "delay",
        "restart",
        "reply_loss",
        "reply_loss_conflict",
        "decision_first",
        "live_verifier",
        "live_verifier:reply_loss",
        "live_verifier:cancel_before_stop",
        "live_verifier:cancel_after_stop",
        "preflight_expiry",
        "preflight_expiry:renewal_failure",
        "publication_expiry",
        "publication_expiry:stop_failure",
    ],
)
def test_worker_settles_expired_published_proposal(backend, fault, tmp_path, monkeypatch):
    expiry_reply_loss = fault.endswith(":reply_loss")
    expiry_stop_failure = fault.endswith(":stop_failure")
    expiry_renewal_failure = fault.endswith(":renewal_failure")
    cancel_stop = fault.endswith((":cancel_before_stop", ":cancel_after_stop"))
    commit_before_cancel = fault.endswith(":cancel_after_stop")
    fault = fault.split(":")[0]

    async def scenario():
        sessions = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.sqlite")
        )
        tasks = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
        )
        provider = _RecordingProvider()
        verifier_settled = asyncio.Event()
        stop_dispatched = asyncio.Event()
        stop_cancelled = asyncio.Event()
        release_stop = asyncio.Event()

        class BlockingVerifier(RecordingVerifier):
            async def verify(self, request):
                self.requests.append(request)
                try:
                    await asyncio.Event().wait()
                finally:
                    verifier_settled.set()

        verifier = (
            BlockingVerifier(_accepted_decision())
            if fault == "live_verifier"
            else RecordingVerifier(_accepted_decision())
        )
        handler = _StaticHandler()
        contract = _contract()

        def make_app():
            app = CayuApp(task_store=tasks, session_store=sessions, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            app.register_completion_verifier(contract.verifier, verifier)
            app.register_completion_result_resolver(
                contract.result_resolver, _Resolver(_task_result())
            )
            return app

        app = make_app()
        submit = type(tasks).submit_admitted_completion_proposal
        settle = type(tasks).settle_work_attempt_lifecycle
        renew = type(tasks).renew_completion_verification_claim
        verify_request = VerifiedTaskWorker._verification_request

        async def delayed_submit(store, request):
            proposal = await submit(store, request)
            if fault in {"live_verifier", "preflight_expiry", "publication_expiry"}:
                return proposal
            admission = await store.load_work_attempt_admission(request.admission_id)
            if fault == "decision_first":
                await app._settle_verified_task_decision(
                    admission.admission_id,
                    CompletionVerifierExecutionRequest(
                        proposal_id=proposal.proposal_id,
                        claim_id="early-verifier-claim",
                        decision_id="early-verifier-decision",
                        worker_id="early-verifier",
                    ),
                )
            await asyncio.sleep(
                max(
                    0,
                    (
                        admission.run_semantics.deadline_expires_at - datetime.now(UTC)
                    ).total_seconds(),
                )
                + 0.05
            )
            return proposal

        async def lose_settlement_reply(store, request):
            receipt = await settle(store, request)
            if request.kind == "proposal_deadline_stop":
                if fault == "reply_loss_conflict":
                    raise WorkAttemptAdmissionConflict("expiry settlement reply lost")
                raise ConnectionError("expiry settlement reply lost")
            return receipt

        async def fail_expiry_settlement(store, request):
            if request.kind == "proposal_deadline_stop":
                raise ConnectionError("expiry settlement unavailable")
            return await settle(store, request)

        async def blocked_expiry_settlement(store, request):
            if request.kind != "proposal_deadline_stop":
                return await settle(store, request)
            receipt = await settle(store, request) if commit_before_cancel else None
            stop_dispatched.set()
            try:
                await release_stop.wait()
            except asyncio.CancelledError:
                stop_cancelled.set()
                await release_stop.wait()
            return receipt if receipt is not None else await settle(store, request)

        async def stop_before_verifier(worker, proposal_id, contract):
            raise ConnectionError("restart after proposal publication")

        async def delayed_renew(store, request):
            claim = await renew(store, request)
            proposal = await store.load_completion_proposal(request.proposal_id)
            admission = await store.load_latest_work_attempt_admission(proposal.task_id)
            await asyncio.sleep(
                max(
                    0,
                    (
                        admission.run_semantics.deadline_expires_at - datetime.now(UTC)
                    ).total_seconds(),
                )
                + 0.05
            )
            if expiry_renewal_failure:
                raise ConnectionError("renewal reply lost after deadline")
            return claim

        monkeypatch.setattr(type(tasks), "submit_admitted_completion_proposal", delayed_submit)
        if fault in {"reply_loss", "reply_loss_conflict"} or expiry_reply_loss:
            monkeypatch.setattr(type(tasks), "settle_work_attempt_lifecycle", lose_settlement_reply)
        if expiry_stop_failure:
            monkeypatch.setattr(
                type(tasks), "settle_work_attempt_lifecycle", fail_expiry_settlement
            )
        if cancel_stop:
            monkeypatch.setattr(
                type(tasks), "settle_work_attempt_lifecycle", blocked_expiry_settlement
            )
        if fault == "restart":
            monkeypatch.setattr(VerifiedTaskWorker, "_verification_request", stop_before_verifier)
        if fault == "preflight_expiry":
            monkeypatch.setattr(type(tasks), "renew_completion_verification_claim", delayed_renew)
        if fault == "publication_expiry":
            from cayu.runtime import _completion_verifier_coordinator as verifier_owner

            capture_operation = verifier_owner.capture_task_store_operation
            record_decision = type(tasks).record_completion_decision
            decision_writes = []

            async def track_decision_write(store, request):
                decision_writes.append(request)
                return await record_decision(store, request)

            async def delayed_publication(operation, **kwargs):
                if kwargs.get("operation_name") == "Completion decision publication":
                    admission = await tasks.load_latest_work_attempt_admission(task.id)
                    await asyncio.sleep(
                        max(
                            0,
                            (
                                admission.run_semantics.deadline_expires_at - datetime.now(UTC)
                            ).total_seconds(),
                        )
                        + 0.05
                    )
                return await capture_operation(operation, **kwargs)

            monkeypatch.setattr(verifier_owner, "capture_task_store_operation", delayed_publication)
            monkeypatch.setattr(type(tasks), "record_completion_decision", track_decision_write)
        try:
            await tasks.publish_work_contract(contract)
            task = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            async with VerifiedTaskWorker(
                app,
                handler,
                worker_id="expiry-worker",
                max_elapsed_seconds=5,
                lease_seconds=10,
                callback_timeout_seconds=1,
            ) as worker:
                if fault in {"live_verifier", "preflight_expiry", "publication_expiry"}:
                    expected_error = (
                        ExceptionGroup
                        if expiry_stop_failure
                        else ConnectionError
                        if expiry_renewal_failure
                        else CompletionVerifierExecutionError
                        if fault == "live_verifier"
                        else RuntimeError
                        if fault == "publication_expiry"
                        else ExecutionDeadlineExceeded
                    )
                    if cancel_stop:
                        running = asyncio.create_task(worker.run(max_tasks=1))
                        try:
                            await asyncio.wait_for(stop_dispatched.wait(), 10)
                            pending_admission = await tasks.load_latest_work_attempt_admission(
                                task.id
                            )
                            prior_receipt = await tasks.load_work_attempt_lifecycle_receipt(
                                pending_admission.admission_id
                            )
                            assert (prior_receipt is not None) is commit_before_cancel
                            running.cancel("cancel while expiry receipt is in flight")
                            with pytest.raises(asyncio.CancelledError) as cancelled:
                                await running
                            assert cancelled.value.args == (
                                "cancel while expiry receipt is in flight",
                            )
                            assert running.cancelling() == 1 and running.cancelled()
                            await asyncio.wait_for(stop_cancelled.wait(), 5)
                            assert worker._running is not None and not worker._running.done()
                            with pytest.raises(VerifiedTaskWorkerDraining):
                                await worker.run(max_tasks=1)
                        finally:
                            release_stop.set()
                            if not running.done():
                                running.cancel()
                                await asyncio.gather(running, return_exceptions=True)
                        with pytest.raises(
                            CompletionVerifierExecutionError, match="bounded execution"
                        ):
                            await worker.aclose()
                        await worker.aclose()
                    else:
                        with pytest.raises(expected_error) as expired:
                            await worker.run(max_tasks=1)
                    if expiry_renewal_failure:
                        assert str(expired.value) == "renewal reply lost after deadline"
                    stopped_admission = await tasks.load_latest_work_attempt_admission(task.id)
                    stopped_receipt = await tasks.load_work_attempt_lifecycle_receipt(
                        stopped_admission.admission_id
                    )
                    if expiry_stop_failure:
                        assert stopped_receipt is None
                        assert [type(error) for error in expired.value.exceptions] == [
                            RuntimeError,
                            ConnectionError,
                        ]
                        assert str(expired.value.exceptions[1]) == "expiry settlement unavailable"
                        monkeypatch.setattr(type(tasks), "settle_work_attempt_lifecycle", settle)
                    else:
                        assert stopped_receipt is not None
                        assert stopped_receipt.request.kind == "proposal_deadline_stop"
                        assert stopped_receipt.task.status is TaskStatus.NEEDS_ATTENTION
                        assert stopped_receipt.task.status_reason == "work_contract_elapsed_limit"
                        assert not stopped_receipt.retired_contract_binding
                    if fault == "publication_expiry":
                        # The owned storage boundary detaches non-allowlisted
                        # exceptions. Prove denial before dispatch, independently
                        # of that boundary's sanitized outward error shape.
                        primary = (
                            expired.value.exceptions[0] if expiry_stop_failure else expired.value
                        )
                        assert type(primary) is RuntimeError
                        assert str(primary) == (
                            "ExecutionDeadlineExceeded: Execution deadline expired "
                            "before completion_decision."
                        )
                        assert decision_writes == []
                    if fault == "live_verifier":
                        await asyncio.wait_for(verifier_settled.wait(), 5)
                    coordinator = app._completion_verifier_coordinator
                    if coordinator._adapter_tasks:
                        _, pending = await asyncio.wait(coordinator._adapter_tasks, timeout=5)
                        assert not pending
                    await asyncio.sleep(0)
                    settlements = {
                        draining.settlement_task
                        for draining in coordinator._draining_adapter_tasks.values()
                        if draining.settlement_task is not None
                    }
                    if settlements:
                        _, pending = await asyncio.wait(settlements, timeout=5)
                        assert not pending
                    assert not coordinator._adapter_tasks
                elif fault == "restart":
                    with pytest.raises(ConnectionError, match="restart after proposal"):
                        await worker.run(max_tasks=1)
                else:
                    assert await worker.run(max_tasks=1) == 1
            if fault in {"restart", "live_verifier", "preflight_expiry", "publication_expiry"}:
                monkeypatch.setattr(VerifiedTaskWorker, "_verification_request", verify_request)
                if backend == "sqlite":
                    await sessions.close()
                    await tasks.close()
                    sessions = SQLiteSessionStore(tmp_path / "sessions.sqlite")
                    tasks = SQLiteTaskStore(tmp_path / "tasks.sqlite")
                app = make_app()
                if fault == "restart" or expiry_stop_failure:
                    async with VerifiedTaskWorker(
                        app, handler, worker_id="replacement-expiry-worker"
                    ) as worker:
                        assert await worker.run(max_tasks=1) == 1
                else:
                    # The first worker already settled this expired proposal.
                    # Reopening must preserve that receipt, not rediscover work
                    # which would require a second settlement owner.
                    assert await tasks.list_unsettled_work_attempt_admissions() == []
            final = await tasks.load_task(task.id)
            admission = await tasks.load_latest_work_attempt_admission(task.id)
            proposal = await tasks.load_completion_proposal_for_attempt(admission.attempt_id)
            receipt = await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
            assert receipt.task == final
            assert (
                len(provider.requests) == len(handler.preparations) == len(handler.proposals) == 1
            )
            assert admission.attempt.ordinal == 1
            if fault == "decision_first":
                assert final.status is TaskStatus.COMPLETED
                assert receipt.request.kind == "decision_application"
                assert receipt.retired_contract_binding
                assert len(verifier.requests) == 1
            else:
                assert final.status is TaskStatus.NEEDS_ATTENTION
                assert final.status_reason == "work_contract_elapsed_limit"
                assert receipt.request.kind == "proposal_deadline_stop"
                assert receipt.request.proposal_id == proposal.proposal_id
                assert receipt.request.proposal_request_sha256 == proposal.request_sha256
                assert not receipt.retired_contract_binding
                assert len(verifier.requests) == (
                    1 if fault in {"live_verifier", "publication_expiry"} else 0
                )
                assert (
                    await tasks.load_completion_decision_for_proposal(proposal.proposal_id) is None
                )
                assert (
                    await tasks.load_active_work_contract_task_for_session(admission.session_id)
                    is not None
                )
        finally:
            if backend == "sqlite":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "failure",
    [
        "provider",
        "handler",
        "prose",
        "handler_timeout",
        "elapsed",
        "source_elapsed",
        "settlement_reply",
    ],
)
def test_worker_settles_quiescent_non_success_without_a_proposal(
    backend,
    failure,
    tmp_path,
    monkeypatch,
):
    async def scenario():
        sessions = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.sqlite")
        )
        tasks = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
        )
        calls = []
        original_stream = _RecordingProvider.stream

        async def stream(provider, request):
            calls.append(request)
            if failure == "provider":
                raise RuntimeError("provider execution failed")
            if failure == "source_elapsed":
                await asyncio.Event().wait()
            async for event in original_stream(provider, request):
                yield event

        class Handler(_StaticHandler):
            async def propose(self, context):
                self.proposals.append(context)
                if failure == "prose":
                    return "The task is done."
                if failure == "handler_timeout":
                    await asyncio.Event().wait()
                if failure == "elapsed":
                    await asyncio.sleep(2.1)
                    return VerifiedTaskHandlerReport(
                        proposal=CompletionProposalCreate(
                            proposal_id=context.proposal_id,
                            attempt_id=context.attempt.attempt_id,
                            result=_result_reference(),
                        )
                    )
                raise ValueError("application proposal cannot be produced")

        monkeypatch.setattr(_RecordingProvider, "stream", stream)
        if failure == "settlement_reply":
            settle = type(tasks).settle_work_attempt_lifecycle

            async def lose_reply(store, request):
                await settle(store, request)
                raise ConnectionError("runtime stop reply lost")

            monkeypatch.setattr(type(tasks), "settle_work_attempt_lifecycle", lose_reply)
        try:
            app = CayuApp(task_store=tasks, session_store=sessions, enable_logging=False)
            app.register_provider(_RecordingProvider(), default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            contract = _contract()
            await tasks.publish_work_contract(contract)
            task = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            verifier = RecordingVerifier(_accepted_decision())
            app.register_completion_verifier(contract.verifier, verifier)
            handler = Handler()
            async with VerifiedTaskWorker(
                app,
                handler,
                worker_id="non-success-worker",
                lease_seconds=10,
                callback_timeout_seconds=0.05 if failure == "handler_timeout" else 3.0,
                max_elapsed_seconds=(
                    2 if failure == "elapsed" else 3 if failure == "source_elapsed" else 3600
                ),
            ) as worker:
                assert await worker.run(max_tasks=1) == 1
            final = await tasks.load_task(task.id)
            assert final.status is TaskStatus.NEEDS_ATTENTION
            assert final.status_reason == {
                "provider": "work_contract_execution_failed",
                "elapsed": "work_contract_elapsed_limit",
                "source_elapsed": "work_contract_elapsed_limit",
            }.get(failure, "work_contract_handler_failed")
            admission = await tasks.load_latest_work_attempt_admission(task.id)
            receipt = await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
            assert receipt.task == final
            assert receipt.request.kind == "runtime_stop"
            assert receipt.request.stop_reason == final.status_reason
            assert not receipt.retired_contract_binding
            assert (
                await tasks.load_active_work_contract_task_for_session(admission.session_id)
                is not None
            )
            assert admission.attempt.ordinal == 1
            assert not verifier.requests
            assert calls
            if failure in {"provider", "source_elapsed"}:
                assert not handler.proposals
            for context in handler.proposals:
                assert await tasks.load_completion_proposal(context.proposal_id) is None
        finally:
            if backend == "sqlite":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("first", ["heartbeat", "cancel"])
def test_worker_preserves_callback_cleanup_failure_after_owner_signal(
    backend,
    first,
    tmp_path,
    monkeypatch,
):
    async def scenario():
        sessions = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.sqlite")
        )
        tasks = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
        )
        entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        worker = None
        run = None

        class Handler(_StaticHandler):
            async def prepare(self, context):
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await release.wait()
                    raise ValueError("callback cleanup failed") from None

        heartbeat = type(tasks).heartbeat

        async def fail_heartbeat(store, *args, **kwargs):
            await heartbeat(store, *args, **kwargs)
            raise RuntimeError("heartbeat reply failed")

        if first == "heartbeat":
            monkeypatch.setattr(type(tasks), "heartbeat", fail_heartbeat)
        try:
            contract = _contract()
            await tasks.publish_work_contract(contract)
            task = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            app = CayuApp(task_store=tasks, session_store=sessions, enable_logging=False)
            worker = VerifiedTaskWorker(
                app,
                Handler(),
                worker_id="colliding-worker",
                lease_seconds=3,
                callback_timeout_seconds=2.0 if first == "heartbeat" else 0.05,
            )
            run = asyncio.create_task(worker.run(max_tasks=1))
            await asyncio.wait_for(entered.wait(), 5)
            if first == "cancel":
                run.cancel("stop worker")
                with pytest.raises(asyncio.CancelledError) as caught:
                    await run
                assert caught.value.args == ("stop worker",)
                assert run.cancelling() == 1 and run.cancelled()
            await asyncio.wait_for(cancelled.wait(), 5)
            assert (await tasks.load_task(task.id)).status is TaskStatus.CLAIMED
            release.set()
            if first == "heartbeat":
                with pytest.raises(ExceptionGroup) as caught:
                    await asyncio.wait_for(run, 5)
                leaves = [
                    item
                    for item in iter_exception_tree(caught.value)
                    if not isinstance(item, BaseExceptionGroup)
                ]
                assert [str(item) for item in leaves] == [
                    "heartbeat reply failed",
                    "callback cleanup failed",
                ]
            else:
                done, _ = await asyncio.wait({worker._running}, timeout=5)
                assert done
                with pytest.raises(ValueError, match="callback cleanup failed"):
                    await worker.aclose()
            assert (await tasks.load_task(task.id)).status is TaskStatus.CLAIMED
        finally:
            release.set()
            if run is not None and not run.done():
                run.cancel()
                await asyncio.gather(run, return_exceptions=True)
            if worker is not None:
                await worker.aclose()
            if backend == "sqlite":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("cancel_count", [1, 2])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_worker_close_cancellation_preserves_settled_callback_failure(
    backend, cancel_count, cleanup_fails, tmp_path
):
    async def scenario():
        sessions = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.sqlite")
        )
        tasks = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
        )
        entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        cleanup_error = ValueError("callback cleanup failed")
        worker = run = close = None

        class Handler(_StaticHandler):
            async def prepare(self, context):
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await release.wait()
                    if cleanup_fails:
                        raise cleanup_error from None
                    raise

        try:
            contract = _contract()
            await tasks.publish_work_contract(contract)
            task = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            worker = VerifiedTaskWorker(
                CayuApp(task_store=tasks, session_store=sessions, enable_logging=False),
                Handler(),
                worker_id="cancelled-close-worker",
                lease_seconds=3,
                callback_timeout_seconds=0.05,
            )
            run = asyncio.create_task(worker.run(max_tasks=1))
            await asyncio.wait_for(entered.wait(), 5)
            run.cancel("stop worker")
            with pytest.raises(asyncio.CancelledError):
                await run
            assert run.cancelling() == 1 and run.cancelled()
            await asyncio.wait_for(cancelled.wait(), 5)
            close = asyncio.create_task(worker.aclose())
            await asyncio.sleep(0)
            for _ in range(cancel_count):
                close.cancel("stop close")
                await asyncio.sleep(0)
            release.set()
            with pytest.raises(asyncio.CancelledError) as caught:
                await close
            assert caught.value.args == ("stop close",)
            assert close.cancelling() == cancel_count and close.cancelled()
            if cleanup_fails:
                # The callback boundary deliberately detaches extension errors
                # for credential safety before the shutdown owner observes them.
                cause = caught.value.__cause__
                assert type(cause) is ValueError
                assert cause.args == cleanup_error.args
                assert cause.__cause__ is None
            else:
                assert caught.value.__cause__ is None
            assert worker._running is None
            assert (await tasks.load_task(task.id)).status is TaskStatus.CLAIMED
            # Settlement is consumed exactly once, not replayed as a later failure.
            await worker.aclose()
        finally:
            release.set()
            for pending in (run, close):
                if pending is not None and not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
            if worker is not None:
                await worker.aclose()
            if backend == "sqlite":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("signal", ["timeout", "cancel"])
def test_worker_retains_heartbeat_and_drain_handle_for_late_preparation(
    backend,
    signal,
    tmp_path,
    monkeypatch,
):
    async def scenario():
        sessions = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.sqlite")
        )
        tasks = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
        )
        release = asyncio.Event()
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        renewed = asyncio.Event()
        worker = None
        run = None

        class SlowHandler(_StaticHandler):
            async def prepare(self, context):
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await release.wait()
                return await super().prepare(context)

        heartbeat = type(tasks).heartbeat

        async def observe_heartbeat(store, *args, **kwargs):
            result = await heartbeat(store, *args, **kwargs)
            renewed.set()
            return result

        monkeypatch.setattr(type(tasks), "heartbeat", observe_heartbeat)
        try:
            contract = _contract()
            await tasks.publish_work_contract(contract)
            task = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            handler = SlowHandler()
            worker = VerifiedTaskWorker(
                app,
                handler,
                worker_id="draining-worker",
                lease_seconds=1,
                callback_timeout_seconds=0.05,
            )
            run = asyncio.create_task(worker.run(max_tasks=1))
            await asyncio.wait_for(entered.wait(), 5)
            if signal == "cancel":
                run.cancel()
                assert run.cancelling() == 1
                with pytest.raises(asyncio.CancelledError):
                    await run
                assert run.cancelled()
                with pytest.raises(VerifiedTaskWorkerDraining):
                    await worker.aclose()
            await asyncio.wait_for(cancelled.wait(), 5)
            await asyncio.wait_for(renewed.wait(), 5)
            pending = await tasks.load_task(task.id)
            assert pending.status is TaskStatus.CLAIMED
            assert pending.session_id is None
            assert await tasks.claim_task("competing-worker") is None
            assert not handler.proposals
            assert worker._running is not None and not worker._running.done()
            release.set()
            if signal == "timeout":
                assert await asyncio.wait_for(run, 5) == 1
                final = await tasks.load_task(task.id)
                assert final.status is TaskStatus.NEEDS_ATTENTION
                assert final.status_reason == "work_contract_preparation_timed_out"
            else:
                done, _ = await asyncio.wait({worker._running}, timeout=5)
                assert done
                await worker.aclose()
                assert (await tasks.load_task(task.id)).status is TaskStatus.CLAIMED
            assert (await tasks.load_task(task.id)).session_id is None
            assert not handler.proposals
        finally:
            release.set()
            if run is not None and not run.done():
                run.cancel()
                await asyncio.gather(run, return_exceptions=True)
            if worker is not None:
                await worker.aclose()
            if backend == "sqlite":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    ("outcome", "fault"),
    [
        ("accepted", None),
        ("continue", None),
        ("interrupt", None),
        ("blocked", None),
        ("needs_review", None),
        ("accepted", "owned_start"),
        ("continue", "owned_start"),
        ("interrupt", "owned_start"),
        ("blocked", "owned_start"),
        ("needs_review", "owned_start"),
        ("accepted", "owned_cancel"),
        ("accepted", "owned_timeout"),
        ("accepted", "application"),
        ("accepted", "retirement"),
        ("accepted", "cancel_application"),
        ("accepted", "discovery"),
        ("accepted", "discovery_decision"),
        ("accepted", "discovery_decision_conflict"),
        ("accepted", "discovery_store_decision"),
        ("continue", "successor"),
        ("continue", "cancel_successor"),
    ],
)
def test_worker_decision_phase_composes_real_execution_and_existing_owners(
    backend, outcome, fault, tmp_path, monkeypatch, caplog, capsys
):
    async def scenario():
        sessions = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "worker-sessions.sqlite")
        )
        tasks = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "worker-tasks.sqlite")
        )
        try:
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            provider = _RecordingProvider()
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            contract = _contract(
                continuation_policy=CompletionContinuationPolicy(
                    rejection_action=(
                        CompletionRejectionAction.INTERRUPT
                        if outcome == "interrupt"
                        else CompletionRejectionAction.CONTINUE
                    )
                )
            )
            await tasks.publish_work_contract(contract)
            await tasks.create_task(
                TaskCreate(
                    task_id="verified-worker-task", type="work", work_contract=contract.reference()
                )
            )
            decision = _accepted_decision() if outcome == "accepted" else _rejected_decision()
            if outcome in {"blocked", "needs_review"}:
                decision = decision.model_copy(update={"verdict": CompletionVerdict(outcome)})
            verifier_started = asyncio.Event()
            verifier_cancelled = asyncio.Event()
            verifier_release = asyncio.Event()

            class RetainedVerifier(RecordingVerifier):
                async def verify(self, request):
                    self.requests.append(request)
                    verifier_started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        verifier_cancelled.set()
                        await verifier_release.wait()
                        return decision

            verifier_type = (
                RetainedVerifier
                if fault in {"owned_cancel", "owned_timeout"}
                else _ContinueOnceVerifier
                if outcome == "continue"
                else RecordingVerifier
            )
            verifier = verifier_type(decision)
            resolver = _Resolver(_task_result())
            app.register_completion_verifier(contract.verifier, verifier)
            app.register_completion_result_resolver(contract.result_resolver, resolver)
            admission = await app.admit_work_attempt(
                RunRequest(
                    agent_name="worker",
                    task_id="verified-worker-task",
                    session_id="verified-worker-session",
                    messages=[Message.text("user", "Build the proposed result.")],
                ),
                execution=WorkAttemptExecutionRequest(
                    admission_id="verified-worker-admission",
                    claim_id="verified-worker-execution-claim",
                    attempt_id="verified-worker-attempt",
                    interaction_id="verified-worker-interaction",
                    worker_id="verified-worker",
                    generation=1,
                    lease_seconds=300,
                ),
            )
            if fault == "owned_start":
                with pytest.raises(WorkAttemptRecoveryRequired, match="released proposed attempt"):
                    await app._start_verified_task_decision(
                        admission.admission_id,
                        CompletionVerifierExecutionRequest(
                            proposal_id="verified-worker-proposal",
                            claim_id="verified-worker-verifier-claim",
                            decision_id="verified-worker-decision",
                            worker_id="verified-worker",
                        ),
                    )
                assert verifier.requests == []
                assert not app._completion_verifier_coordinator._adapter_tasks
            async for _ in app._execute_work_attempt(
                WorkAttemptRunRequest(
                    admission_id=admission.admission_id,
                    claim_id=admission.claim.claim_id,
                    worker_id=admission.claim.worker_id,
                    generation=1,
                    lease_seconds=300,
                )
            ):
                pass
            assert len(provider.requests) == 1
            assert (await sessions.load(admission.session_id)).status is SessionStatus.COMPLETED
            assert (await tasks.load_task(admission.task_id)).status is TaskStatus.RUNNING
            await app.submit_work_attempt_proposal(
                WorkAttemptProposalRequest(
                    admission_id=admission.admission_id,
                    claim_id=admission.claim.claim_id,
                    generation=1,
                    proposal=CompletionProposalCreate(
                        proposal_id="verified-worker-proposal",
                        attempt_id=admission.attempt_id,
                        result=_result_reference(),
                        evidence_references=(),
                    ),
                )
            )
            verification = CompletionVerifierExecutionRequest(
                proposal_id="verified-worker-proposal",
                claim_id="verified-worker-verifier-claim",
                decision_id="verified-worker-decision",
                worker_id="verified-worker",
                lease_seconds=9
                if fault
                in {"discovery_decision", "discovery_decision_conflict", "discovery_store_decision"}
                else 300,
                execution_timeout_seconds=7.0
                if fault
                in {"discovery_decision", "discovery_decision_conflict", "discovery_store_decision"}
                else 30.0,
            )
            if fault in {
                "discovery",
                "discovery_decision",
                "discovery_decision_conflict",
                "discovery_store_decision",
            }:
                # These admission/attempt/proposal IDs came from the public
                # admitted seam, not the worker's deterministic name function.
                if fault != "discovery":
                    if fault == "discovery_store_decision":
                        claim_request = CompletionVerificationClaimRequest(
                            claim_id=verification.claim_id,
                            proposal_id=verification.proposal_id,
                            worker_id=verification.worker_id,
                            verifier=contract.verifier,
                            verifier_profile_fingerprint=_verifier_profile_fingerprint(
                                contract.verifier
                            ),
                            lease_seconds=9,
                        )
                        await _claim_completion_verification(tasks, claim_request)
                        await tasks.renew_completion_verification_claim(claim_request)
                        decision = await tasks.record_completion_decision(
                            CompletionDecisionCreate(
                                **_accepted_decision().model_dump(mode="python"),
                                decision_id=verification.decision_id,
                                proposal_id=verification.proposal_id,
                                claim_id=verification.claim_id,
                                worker_id=verification.worker_id,
                                verifier=contract.verifier,
                                verifier_profile_fingerprint=claim_request.verifier_profile_fingerprint,
                            )
                        )
                    else:
                        decision = await app.verify_completion_proposal(verification)
                    claim = await tasks.load_completion_verification_claim(verification.proposal_id)
                    assert claim.lease_seconds == 9
                    assert claim.execution_timeout_seconds == (
                        None if fault == "discovery_store_decision" else 7.0
                    )
                    # Runtime renewal happened before adapter dispatch. The
                    # remaining window is not the original request duration.
                    assert claim.lease_expires_at > claim.claimed_at + timedelta(seconds=9)
                    if backend == "sqlite":
                        await tasks.close()
                        await sessions.close()
                        tasks = SQLiteTaskStore(tmp_path / "worker-tasks.sqlite")
                        sessions = SQLiteSessionStore(tmp_path / "worker-sessions.sqlite")
                        assert (
                            await tasks.load_completion_verification_claim(verification.proposal_id)
                            == claim
                        )
                    assert decision.decision_id == "verified-worker-decision"
                restarted = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
                if fault == "discovery":
                    restarted.register_completion_verifier(contract.verifier, verifier)
                restarted.register_completion_result_resolver(contract.result_resolver, resolver)
                handler = _StaticHandler()
                async with VerifiedTaskWorker(
                    restarted, handler, worker_id="discovery-worker"
                ) as worker:
                    if fault == "discovery_decision_conflict":
                        original_load = type(tasks).load_completion_verification_claim
                        secret = "verifier-claim-private-canary"

                        class HostileDuration:
                            def __repr__(self):
                                return secret

                        bad_duration = 10

                        async def conflicting_claim(store, proposal_id):
                            current = await original_load(store, proposal_id)
                            update = {"lease_seconds": bad_duration}
                            if isinstance(bad_duration, HostileDuration):
                                update["worker_id"] = secret
                            return current.model_copy(update=update)

                        async def assert_diagnostic_safe_rejection():
                            with warnings.catch_warnings(record=True) as captured_warnings:
                                warnings.simplefilter("always")
                                with pytest.raises(RuntimeError) as caught:
                                    await asyncio.wait_for(worker.run(max_tasks=1), 10)
                            _assert_secret_absent_from_cayu_error(caught.value, secret)
                            assert all(
                                secret not in str(item.message) for item in captured_warnings
                            )
                            assert secret not in caplog.text
                            captured = capsys.readouterr()
                            assert secret not in captured.out + captured.err

                        with monkeypatch.context() as patch:
                            patch.setattr(
                                type(tasks), "load_completion_verification_claim", conflicting_claim
                            )
                            with pytest.raises(RuntimeError, match="retained request authority"):
                                await asyncio.wait_for(worker.run(max_tasks=1), 10)
                            bad_duration = HostileDuration()
                            await assert_diagnostic_safe_rejection()
                            patch.setattr(
                                type(tasks), "load_completion_verification_claim", original_load
                            )
                            original_decision_load = type(
                                tasks
                            ).load_completion_decision_for_proposal

                            async def conflicting_decision(store, proposal_id):
                                current = await original_decision_load(store, proposal_id)
                                return current.model_copy(
                                    update={"worker_id": HostileDuration(), "summary": secret}
                                )

                            patch.setattr(
                                type(tasks),
                                "load_completion_decision_for_proposal",
                                conflicting_decision,
                            )
                            await assert_diagnostic_safe_rejection()
                        assert (
                            await tasks.load_task(admission.task_id)
                        ).status is TaskStatus.RUNNING
                        assert resolver.requests == handler.preparations == handler.proposals == []
                        assert len(provider.requests) == len(verifier.requests) == 1
                        assert (
                            await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
                            is None
                        )
                        return
                    assert await asyncio.wait_for(worker.run(max_tasks=1), 10) == 1
                final = await tasks.load_task(admission.task_id)
                assert final.status is TaskStatus.COMPLETED
                assert not handler.preparations and not handler.proposals
                assert len(provider.requests) == len(resolver.requests) == 1
                assert len(verifier.requests) == (0 if fault == "discovery_store_decision" else 1)
                receipt = await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
                assert receipt.task == final
                assert receipt.retired_contract_binding
                assert await tasks.list_unsettled_work_attempt_admissions() == []
                return
            if fault in {"owned_cancel", "owned_timeout"}:
                if fault == "owned_timeout":
                    verification = verification.model_copy(
                        update={"execution_timeout_seconds": 0.05}
                    )
                started = await app._start_verified_task_decision(
                    admission.admission_id, verification
                )
                waiting = asyncio.create_task(started.result())
                try:
                    await asyncio.wait_for(verifier_started.wait(), 5)
                    if fault == "owned_cancel":
                        waiting.cancel("cancel owned decision phase")
                        with pytest.raises(asyncio.CancelledError):
                            await waiting
                        assert waiting.cancelling() == 1
                        assert waiting.cancelled()
                    else:
                        with pytest.raises(
                            CompletionVerifierExecutionError, match="bounded execution"
                        ):
                            await waiting
                    await asyncio.wait_for(verifier_cancelled.wait(), 5)
                    drain_waiter = asyncio.create_task(started.verification.settlement())
                    await asyncio.sleep(0)
                    assert not drain_waiter.done()
                    drain_waiter.cancel("stop observing, keep owned verification")
                    with pytest.raises(asyncio.CancelledError):
                        await drain_waiter
                    assert drain_waiter.cancelling() == 1 and drain_waiter.cancelled()
                    assert (await tasks.load_task(admission.task_id)).status is TaskStatus.RUNNING
                    assert len(verifier.requests) == 1
                    assert resolver.requests == []
                finally:
                    verifier_release.set()
                    if not waiting.done():
                        waiting.cancel()
                        await asyncio.gather(waiting, return_exceptions=True)
                    assert (
                        await asyncio.wait_for(started.verification.settlement(), 5)
                    ).failure is None
                assert (
                    await tasks.load_completion_decision_for_proposal(verification.proposal_id)
                    is None
                )
                assert (
                    await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id) is None
                )
                return
            if fault not in {None, "owned_start"} and outcome == "accepted":
                committed = asyncio.Event()
                with monkeypatch.context() as patch:
                    if fault in {"application", "cancel_application"}:
                        resolve = app.resolve_completion_result

                        async def lose_application_reply(request):
                            await resolve(request)
                            committed.set()
                            if fault == "cancel_application":
                                await asyncio.Event().wait()
                            raise ConnectionError("worker application reply lost")

                        patch.setattr(app, "resolve_completion_result", lose_application_reply)
                    else:
                        settle = type(tasks).settle_work_attempt_lifecycle

                        async def lose_retirement_reply(store, request):
                            await settle(store, request)
                            committed.set()
                            raise ConnectionError("worker retirement reply lost")

                        patch.setattr(
                            type(tasks), "settle_work_attempt_lifecycle", lose_retirement_reply
                        )
                    operation = asyncio.create_task(
                        app._settle_verified_task_decision(admission.admission_id, verification)
                    )
                    if fault == "cancel_application":
                        await asyncio.wait_for(committed.wait(), 10)
                        operation.cancel()
                        assert operation.cancelling() == 1
                        with pytest.raises(asyncio.CancelledError):
                            await operation
                        assert operation.cancelled()
                    else:
                        with pytest.raises(ConnectionError):
                            await operation
                    assert committed.is_set()
                assert (await tasks.load_task(admission.task_id)).status is TaskStatus.COMPLETED
                bound = await tasks.load_active_work_contract_task_for_session(admission.session_id)
                assert (bound is None) is (fault == "retirement")
                # Recovery must use durable verifier/application evidence, not
                # the vanished original collaborator registrations.
                app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            if fault == "owned_start":
                started = await app._start_verified_task_decision(
                    admission.admission_id, verification
                )
                try:
                    result = await started.result()
                finally:
                    settled = await asyncio.wait_for(started.verification.settlement(), 5)
                    assert settled.failure is None
            else:
                result = await app._settle_verified_task_decision(
                    admission.admission_id, verification
                )
            expected_status = {
                "accepted": TaskStatus.COMPLETED,
                "continue": TaskStatus.RUNNING,
                "interrupt": TaskStatus.PAUSED,
                "blocked": TaskStatus.BLOCKED,
                "needs_review": TaskStatus.NEEDS_ATTENTION,
            }[outcome]
            assert result.application.task.status is expected_status
            assert await tasks.load_task(admission.task_id) == result.application.task
            assert len(verifier.requests) == 1
            assert len(resolver.requests) == (1 if outcome == "accepted" else 0)
            if outcome == "continue":
                assert result.settlement is None
                assert (
                    await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id) is None
                )
            else:
                assert result.settlement.task == result.application.task
                assert result.settlement.retired_contract_binding is (outcome == "accepted")
            bound = await tasks.load_active_work_contract_task_for_session(admission.session_id)
            assert (bound is None) is (outcome == "accepted")
            # Fresh app has no registered collaborators. Existing verifier,
            # application and lifecycle receipts must carry the exact replay.
            restarted = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            replay = await restarted._settle_verified_task_decision(
                admission.admission_id, verification
            )
            assert replay == result
            assert len(provider.requests) == 1
            assert len(verifier.requests) == 1
            assert len(resolver.requests) == (1 if outcome == "accepted" else 0)
            if outcome == "continue":

                async def schedule():
                    return await app._continue_verified_task(
                        admission.admission_id,
                        result.decision.decision_id,
                        worker_id="verified-worker",
                        lease_seconds=300,
                    )

                if fault in {"successor", "cancel_successor"}:
                    committed = asyncio.Event()
                    admit = app.admit_work_attempt

                    async def lose_successor_reply(request, *, execution):
                        await admit(request, execution=execution)
                        committed.set()
                        if fault == "cancel_successor":
                            await asyncio.Event().wait()
                        raise ConnectionError("worker successor reply lost")

                    with monkeypatch.context() as patch:
                        patch.setattr(app, "admit_work_attempt", lose_successor_reply)
                        operation = asyncio.create_task(schedule())
                        if fault == "cancel_successor":
                            await asyncio.wait_for(committed.wait(), 10)
                            operation.cancel()
                            assert operation.cancelling() == 1
                            with pytest.raises(asyncio.CancelledError):
                                await operation
                            assert operation.cancelled()
                        else:
                            with pytest.raises(ConnectionError):
                                await operation
                        assert committed.is_set()
                successor = await schedule()
                assert await schedule() == successor
                with pytest.raises(WorkAttemptRecoveryRequired, match="dedicated recovery"):
                    await app._continue_verified_task(
                        admission.admission_id,
                        result.decision.decision_id,
                        worker_id="competing-worker",
                        lease_seconds=300,
                    )
                assert successor.attempt.ordinal == 2
                assert successor.continuation.decision == result.decision
                assert successor.continuation.gaps == result.decision.gaps
                assert (
                    successor.continuation.application_idempotency_key
                    == result.application.idempotency_key
                )
                assert successor.run_semantics == admission.run_semantics
                assert (
                    successor.source_execution_profile_fingerprint
                    == admission.source_execution_profile_fingerprint
                )
                assert (
                    await tasks.load_latest_work_attempt_admission(admission.task_id) == successor
                )
                assert len(provider.requests) == 1
                async for _ in app._execute_work_attempt(
                    WorkAttemptRunRequest(
                        admission_id=successor.admission_id,
                        claim_id=successor.claim.claim_id,
                        worker_id=successor.claim.worker_id,
                        generation=1,
                        lease_seconds=300,
                    )
                ):
                    pass
                assert len(provider.requests) == 2
                await app.submit_work_attempt_proposal(
                    WorkAttemptProposalRequest(
                        admission_id=successor.admission_id,
                        claim_id=successor.claim.claim_id,
                        generation=1,
                        proposal=CompletionProposalCreate(
                            proposal_id="verified-worker-proposal-2",
                            attempt_id=successor.attempt_id,
                            result=_result_reference(),
                        ),
                    )
                )
                completed = await app._settle_verified_task_decision(
                    successor.admission_id,
                    CompletionVerifierExecutionRequest(
                        proposal_id="verified-worker-proposal-2",
                        claim_id="verified-worker-verifier-claim-2",
                        decision_id="verified-worker-decision-2",
                        worker_id="verified-worker",
                    ),
                )
                assert completed.application.task.status is TaskStatus.COMPLETED
                assert completed.settlement.retired_contract_binding
                assert (
                    await tasks.load_active_work_contract_task_for_session(admission.session_id)
                    is None
                )
                assert len(verifier.requests) == 2
                assert len(resolver.requests) == 1
            else:
                before = await tasks.load_task(admission.task_id)
                with pytest.raises(WorkCompletionConflict):
                    await app._continue_verified_task(
                        admission.admission_id,
                        result.decision.decision_id,
                        worker_id="verified-worker",
                        lease_seconds=300,
                    )
                assert await tasks.load_task(admission.task_id) == before
                assert (
                    await tasks.load_latest_work_attempt_admission(admission.task_id)
                ).admission_id == admission.admission_id
                assert len(provider.requests) == 1
        finally:
            if backend == "sqlite":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())
