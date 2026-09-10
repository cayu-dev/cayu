from __future__ import annotations

import asyncio
import copy
import pickle
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.task_invocation_fixtures import task_backed_session_invocation
from tests.core.test_invocation_lifecycle_commands import (
    _create_command,
    _NeverCalledProvider,
    _profile,
)
from tests.core.test_verified_work_contracts import _contract, _RecordingProvider
from tests.core.test_work_attempt_admission import (
    _configured_public_initial_admission,
    _prepare_request,
)

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    InMemorySessionStore,
    InMemoryTaskStore,
    Message,
    SecretRedactor,
    SessionStatus,
    SQLiteSessionStore,
    SQLiteTaskStore,
    TaskCreate,
    TaskStatus,
)
from cayu.runtime._invocation_lifecycle import (
    PreparedInvocationBinding,
    _authenticated_invocation_context,
    released_invocation_evidence,
)
from cayu.runtime._work_attempt_invocation import (
    WorkAttemptInvocationAuthority,
    _authenticated_work_attempt_invocation,
)
from cayu.runtime.execution_profiles import ActiveInvocationExecutionProfile
from cayu.runtime.tool_exposure import ToolCapabilityCeiling
from cayu.runtime.work_attempt_admission import (
    WorkAttemptAdmission,
    WorkAttemptAdmissionActivate,
    WorkAttemptExecutionClaimLost,
    WorkAttemptRecoveryRequest,
    WorkAttemptRecoveryRequired,
    WorkAttemptRunRequest,
)
from cayu.runtime.work_attempt_semantics import WorkAttemptRunSemantics


@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled", "abandoned"])
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_admitted_common_runtime_settles_session_without_accepting_task(
    outcome, backend, monkeypatch, tmp_path, request
):
    pytest_request = request

    async def scenario():
        entered = asyncio.Event()
        ownership_now = [datetime.now(UTC)]
        original_stream = _RecordingProvider.stream

        async def controlled_stream(provider, request):
            entered.set()
            if outcome == "failed":
                raise RuntimeError("injected governed provider failure")
            if outcome == "cancelled":
                await asyncio.Event().wait()
            async for event in original_stream(provider, request):
                yield event

        monkeypatch.setattr(_RecordingProvider, "stream", controlled_stream)
        sessions = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "governed-sessions.sqlite")
        )
        tasks = (
            InMemoryTaskStore(
                clock=lambda: ownership_now[0], ownership_clock=lambda: ownership_now[0]
            )
            if backend == "memory"
            else SQLiteTaskStore(
                tmp_path / "governed-tasks.sqlite",
                clock=lambda: ownership_now[0],
                ownership_clock=lambda: ownership_now[0],
            )
        )
        if backend == "sqlite":
            pytest_request.addfinalizer(lambda: asyncio.run(tasks.close()))
            pytest_request.addfinalizer(lambda: asyncio.run(sessions.close()))
        app, request, execution = await _configured_public_initial_admission(
            prefix="governed-run",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(),
            agent_system_prompt="Preserve this governed system prompt.",
        )
        admission = await app.admit_work_attempt(request, execution=execution)
        run_request = WorkAttemptRunRequest(
            admission_id=admission.admission_id,
            claim_id=admission.claim.claim_id,
            worker_id=admission.claim.worker_id,
            generation=admission.claim.generation,
            lease_seconds=execution.lease_seconds,
        )
        with pytest.raises(TypeError):
            async for _ in app._execute_work_attempt(admission):
                pytest.fail("A caller-provided admission must not dispatch work.")
        foreign_app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        with pytest.raises(WorkAttemptExecutionClaimLost, match="not owned here"):
            async for _ in foreign_app._execute_work_attempt(run_request):
                pytest.fail("A foreign process owner must not dispatch work.")
        assert (
            await tasks.load_work_attempt_admission(admission.admission_id)
        ).execution_entry is None
        deferred_input = await sessions.load_deferred_interaction_input(admission.session_id)

        async def unrelated_deferred_input(_session_id):
            return deferred_input.model_copy(update={"interaction_id": "unrelated-interaction"})

        with monkeypatch.context() as patch:
            patch.setattr(sessions, "load_deferred_interaction_input", unrelated_deferred_input)
            with pytest.raises(RuntimeError, match="another interaction"):
                async for _ in app._execute_work_attempt(run_request):
                    pytest.fail("Conflicting deferred input must fail before execution entry.")
        assert not entered.is_set()
        assert (await tasks.load_work_attempt_admission(admission.admission_id)) == admission
        assert (
            await sessions.load_deferred_interaction_input(admission.session_id) == deferred_input
        )
        session = await sessions.load(admission.session_id)
        context = await app._session_engine._resolve_work_attempt_invocation_context(
            admission,
            session=session,
            checkpoint=await sessions.load_checkpoint(session.id),
        )
        try:
            events = []

            async def consume():
                stream = app._execute_work_attempt(run_request)
                async for event in stream:
                    events.append(event)
                    if outcome == "abandoned" and entered.is_set():
                        await stream.aclose()
                        break

            owner = asyncio.create_task(consume())
            if outcome == "cancelled":
                await asyncio.wait_for(entered.wait(), timeout=3)
                # The provider is actually in flight, not merely admitted.
                # A duplicate runtime entrance must not acquire a second run.
                with pytest.raises(WorkAttemptRecoveryRequired, match="already entered"):
                    await app._session_engine._enter_work_attempt_invocation_context(
                        admission,
                        session=session,
                        checkpoint=await sessions.load_checkpoint(session.id),
                        lease_seconds=execution.lease_seconds,
                    )
                owner.cancel("cancel governed execution")
                with pytest.raises(asyncio.CancelledError):
                    await owner
                assert owner.cancelled()
                assert owner.cancelling() == 1
            elif outcome == "failed":
                # The shared runtime publishes ordinary provider failures as a
                # failed session event; cancellation remains a raised signal.
                await owner
                failures = [event for event in events if event.type is EventType.SESSION_FAILED]
                assert len(failures) == 1
                assert failures[0].payload["error"] == "injected governed provider failure"
            else:
                await owner
        finally:
            if not owner.done():
                owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
        expected = {
            "completed": SessionStatus.COMPLETED,
            "failed": SessionStatus.FAILED,
            "cancelled": SessionStatus.INTERRUPTED,
            "abandoned": SessionStatus.INTERRUPTED,
        }[outcome]
        assert entered.is_set()
        if outcome == "completed":
            assert EventType.SESSION_COMPLETED in [event.type for event in events]
        assert EventType.TASK_COMPLETED not in [event.type for event in events]
        assert EventType.TASK_FAILED not in [event.type for event in events]
        settled_session = await sessions.load(session.id)
        assert settled_session.status is expected
        assert (await sessions.load_transcript(session.id))[0] == Message.text(
            "system", "Preserve this governed system prompt."
        )
        evidence = released_invocation_evidence(
            settled_session,
            await sessions.load_checkpoint(session.id),
            session_id=session.id,
            session_instance_id=session.instance_id,
            active_profile=context.active_profile,
        )
        assert evidence.interaction_id == admission.interaction_id
        assert evidence.profile_fingerprint == admission.source_execution_profile_fingerprint
        durable_admission = await tasks.load_work_attempt_admission(admission.admission_id)
        # A new app needs no provider registration or process-local context to
        # verify the exact released invocation from durable evidence.
        assert (
            await foreign_app._session_engine.load_work_attempt_release_evidence(durable_admission)
            == evidence
        )
        wrong_entry = durable_admission.execution_entry.model_copy(
            update={
                "request": durable_admission.execution_entry.request.model_copy(
                    update={"run_epoch": evidence.run_epoch + 1}
                )
            }
        )
        with pytest.raises(RuntimeError):
            await foreign_app._session_engine.load_work_attempt_release_evidence(
                durable_admission.model_copy(update={"execution_entry": wrong_entry})
            )
        with pytest.raises(WorkAttemptRecoveryRequired, match="already entered"):
            async for _ in app._execute_work_attempt(run_request):
                pytest.fail("A settled invocation cannot be redispatched.")
        assert await tasks.load_work_attempt_admission(admission.admission_id) == durable_admission
        assert (await tasks.load_task(admission.task_id)).status is TaskStatus.RUNNING
        assert await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id) is None
        checkpoint_before = await sessions.load_checkpoint(session.id)
        transcript_before = await sessions.load_transcript(session.id)
        ownership_now[0] = durable_admission.claim.lease_expires_at + timedelta(seconds=1)
        recovery_request = WorkAttemptRecoveryRequest(
            admission_id=admission.admission_id,
            claim_id="replacement-settlement-claim",
            worker_id="replacement-worker",
            generation=2,
            lease_seconds=300,
        )
        activate = tasks.activate_work_attempt_recovery
        acknowledgement_lost = False

        async def lose_activation_acknowledgement(store, request):
            nonlocal acknowledgement_lost
            assert store is tasks
            result = await activate(request)
            if not acknowledgement_lost:
                acknowledgement_lost = True
                raise ConnectionError("injected activation acknowledgement loss")
            return result

        with monkeypatch.context() as patch:
            patch.setattr(
                type(tasks), "activate_work_attempt_recovery", lose_activation_acknowledgement
            )
            with pytest.raises(ConnectionError, match="acknowledgement loss"):
                await foreign_app.recover_work_attempt(recovery_request)
            recovered = await foreign_app.recover_work_attempt(recovery_request)
        assert recovered.claim.generation == 2
        assert recovered.execution_entry == durable_admission.execution_entry
        assert await foreign_app.recover_work_attempt(recovery_request) == recovered
        assert await sessions.load(session.id) == settled_session
        assert await sessions.load_checkpoint(session.id) == checkpoint_before
        assert await sessions.load_transcript(session.id) == transcript_before
        assert (
            await foreign_app._session_engine.load_work_attempt_release_evidence(recovered)
            == evidence
        )
        assert (await tasks.load_task(admission.task_id)).status is TaskStatus.RUNNING

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    ("failure", "readback_conflict"),
    [
        ("acknowledgement", None),
        ("cancellation", None),
        ("acknowledgement", "message"),
        ("acknowledgement", "attribution"),
        ("acknowledgement", "cursor"),
    ],
)
def test_governed_initial_publication_acknowledgement_loss(
    backend, failure, readback_conflict, tmp_path, monkeypatch
):
    async def scenario():
        sessions = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "initial-publication-sessions.sqlite")
        )
        tasks = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "initial-publication-tasks.sqlite")
        )
        try:
            app, request, execution = await _configured_public_initial_admission(
                prefix="initial-publication",
                sessions=sessions,
                tasks=tasks,
                redactor=SecretRedactor(),
                agent_system_prompt="Retain this exact initial system prompt.",
            )
            admission = await app.admit_work_attempt(request, execution=execution)
            deferred = await sessions.load_deferred_interaction_input(admission.session_id)
            expected = deferred.initial_transcript_messages
            publish = type(sessions).replace_initial_transcript_messages
            load_snapshot = type(sessions).load_transcript_snapshot
            publications = []
            publication_committed = asyncio.Event()
            provider_calls = []
            original_stream = _RecordingProvider.stream

            async def record_provider_call(provider, request):
                provider_calls.append(request)
                async for event in original_stream(provider, request):
                    yield event

            async def conflicting_readback(store, session_id):
                result = await load_snapshot(store, session_id)
                if publications and readback_conflict is not None:
                    result = result.model_copy(deep=True)
                    if readback_conflict == "message":
                        result.records[0].message = Message.text("system", "Different prompt.")
                    elif readback_conflict == "attribution":
                        result.records[0].interaction_id = admission.interaction_id
                    else:
                        result.cursor += 1
                return result

            async def publish_then_lose_acknowledgement(store, *args, **kwargs):
                await publish(store, *args, **kwargs)
                publications.append(args[0])
                publication_committed.set()
                if failure == "cancellation":
                    await asyncio.Event().wait()
                raise ConnectionError("initial publication acknowledgement lost")

            with monkeypatch.context() as patch:
                patch.setattr(_RecordingProvider, "stream", record_provider_call)
                patch.setattr(type(sessions), "load_transcript_snapshot", conflicting_readback)
                patch.setattr(
                    type(sessions),
                    "replace_initial_transcript_messages",
                    publish_then_lose_acknowledgement,
                )

                async def consume():
                    return [
                        event
                        async for event in app._execute_work_attempt(
                            WorkAttemptRunRequest(
                                admission_id=admission.admission_id,
                                claim_id=admission.claim.claim_id,
                                worker_id=admission.claim.worker_id,
                                generation=1,
                                lease_seconds=execution.lease_seconds,
                            )
                        )
                    ]

                if failure == "cancellation":
                    owner = asyncio.create_task(consume())
                    try:
                        await asyncio.wait_for(publication_committed.wait(), timeout=5)
                        owner.cancel("cancel after initial publication commit")
                        assert owner.cancelling() == 1
                        with pytest.raises(asyncio.CancelledError):
                            await owner
                        assert owner.cancelled()
                        assert owner.cancelling() == 1
                    finally:
                        if not owner.done():
                            owner.cancel()
                            await asyncio.gather(owner, return_exceptions=True)
                elif readback_conflict is None:
                    events = await consume()
                else:
                    with pytest.raises(RuntimeError, match="conflicts with admitted input"):
                        await consume()
            assert publications == [admission.session_id]
            assert provider_calls == []
            assert await sessions.load_transcript(admission.session_id) == expected
            assert await sessions.load_deferred_interaction_input(admission.session_id) is None
            assert "initial_transcript_pending" not in await sessions.load_checkpoint(
                admission.session_id
            )
            assert (await tasks.load_task(admission.task_id)).status is TaskStatus.RUNNING
            if readback_conflict is not None:
                assert (
                    await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id) is None
                )
                return
            if failure == "cancellation":
                assert (
                    await sessions.load(admission.session_id)
                ).status is SessionStatus.INTERRUPTED
            else:
                failures = [event for event in events if event.type is EventType.SESSION_FAILED]
                assert len(failures) == 1
                assert failures[0].payload["error"] == "initial publication acknowledgement lost"
                assert (await sessions.load(admission.session_id)).status is SessionStatus.FAILED
            entered = await tasks.load_work_attempt_admission(admission.admission_id)
            await app._session_engine.load_work_attempt_release_evidence(entered)
        finally:
            if backend == "sqlite":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_work_attempt_recovers_entry_before_model_dispatch(backend, tmp_path, monkeypatch):
    async def scenario():
        now = [datetime.now(UTC)]
        sessions = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "entry-crash-sessions.sqlite")
        )
        tasks = (
            InMemoryTaskStore(clock=lambda: now[0], ownership_clock=lambda: now[0])
            if backend == "memory"
            else SQLiteTaskStore(
                tmp_path / "entry-crash-tasks.sqlite",
                clock=lambda: now[0],
                ownership_clock=lambda: now[0],
            )
        )
        try:
            app, request, execution = await _configured_public_initial_admission(
                prefix="entry-crash", sessions=sessions, tasks=tasks, redactor=SecretRedactor()
            )
            admission = await app.admit_work_attempt(request, execution=execution)
            run_request = WorkAttemptRunRequest(
                admission_id=admission.admission_id,
                claim_id=admission.claim.claim_id,
                worker_id=admission.claim.worker_id,
                generation=1,
                lease_seconds=execution.lease_seconds,
            )

            async def fail_before_loop(**_kwargs):
                raise RuntimeError("injected failure before model dispatch")

            with monkeypatch.context() as patch:
                patch.setattr(
                    app._session_engine, "_reconstruct_targeted_tool_grants", fail_before_loop
                )
                with pytest.raises(RuntimeError, match="before model dispatch"):
                    async for _ in app._execute_work_attempt(run_request):
                        pass
            entered = await tasks.load_work_attempt_admission(admission.admission_id)
            assert entered.execution_entry is not None
            assert (await tasks.load_task(admission.task_id)).status is TaskStatus.RUNNING
            with pytest.raises(WorkAttemptRecoveryRequired, match="already entered"):
                async for _ in app._execute_work_attempt(run_request):
                    pytest.fail("Uncertain setup must not permit same-generation redispatch.")
            now[0] = entered.claim.lease_expires_at + timedelta(seconds=1)
            replacement = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            provider = _RecordingProvider()
            replacement.register_provider(provider, default=True)
            replacement.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            coordinator = replacement._session_engine._recovery_coordinator
            factory = coordinator._reconstruct_invocation_context
            recovery_contexts = []

            def record_recovery_context(**kwargs):
                context = factory(**kwargs)
                recovery_contexts.append(context)
                return context

            with monkeypatch.context() as patch:
                patch.setattr(
                    coordinator, "_reconstruct_invocation_context", record_recovery_context
                )
                recovered = await replacement.recover_work_attempt(
                    WorkAttemptRecoveryRequest(
                        admission_id=admission.admission_id,
                        claim_id="entry-crash-replacement",
                        worker_id="replacement-worker",
                        generation=2,
                        lease_seconds=300,
                    )
                )
            assert recovery_contexts
            for context in recovery_contexts:
                assert context.work_attempt.admission.claim.generation == 2
                assert context.work_attempt.admission.execution_entry == entered.execution_entry
                # Characterize the sibling workspace guard under the actual
                # reconstructed recovery context; full workspace acceptance
                # still requires its real sync/finalization fault regression.
                guarded_task = await coordinator._require_governed_completion_task(
                    session=await sessions.load(admission.session_id),
                    marker={"task_id": admission.task_id},
                    invocation_context=context,
                )
                assert guarded_task.status is TaskStatus.RUNNING
                guarded_task.metadata["caller-mutation"] = True
                assert "caller-mutation" not in (await tasks.load_task(admission.task_id)).metadata
                for marker in ({}, {"task_id": None}, {"task_id": "unrelated-task"}):
                    with pytest.raises(RuntimeError, match="conflicting task authority"):
                        await coordinator._require_governed_completion_task(
                            session=await sessions.load(admission.session_id),
                            marker=marker,
                            invocation_context=context,
                        )
            assert not provider.requests
            assert recovered.execution_entry == entered.execution_entry
            async for _ in replacement._execute_work_attempt(
                WorkAttemptRunRequest(
                    admission_id=recovered.admission_id,
                    claim_id=recovered.claim.claim_id,
                    worker_id=recovered.claim.worker_id,
                    generation=2,
                    lease_seconds=300,
                )
            ):
                pass
            assert len(provider.requests) == 1
            assert (await sessions.load(admission.session_id)).status is SessionStatus.COMPLETED
            assert (await tasks.load_task(admission.task_id)).status is TaskStatus.RUNNING
        finally:
            if backend == "sqlite":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())


def test_work_attempt_context_retains_exact_authority_without_exporting_provenance():
    async def scenario():
        profile = _profile()
        store = InMemoryTaskStore()
        contract = _contract()
        await store.publish_work_contract(contract)
        task = await store.create_task(
            TaskCreate(task_id="context-task", type="work", work_contract=contract.reference())
        )
        session_invocation = await task_backed_session_invocation(store, task.id, "context-session")
        request = _prepare_request(
            task_id=task.id, session_id="context-session", session_invocation=session_invocation
        ).model_copy(
            update={
                "source_execution_profile_fingerprint": profile.fingerprint,
                "run_semantics": WorkAttemptRunSemantics(max_steps=7),
            }
        )
        prepared = await store.prepare_work_attempt_admission(request)
        with pytest.raises(ValueError, match="executable authority"):
            _authenticated_work_attempt_invocation(prepared)
        admission = await store.activate_work_attempt_admission(
            WorkAttemptAdmissionActivate(
                admission_id=prepared.admission_id,
                claim_id=prepared.claim.claim_id,
                prepare_request_sha256=prepared.prepare_request_sha256,
                session_evidence_sha256="1" * 64,
            )
        )
        # Durable reconstruction is data, not an independently usable authority.
        restored = WorkAttemptAdmission.model_validate_json(admission.model_dump_json())
        with pytest.raises(TypeError, match="runtime owner"):
            WorkAttemptInvocationAuthority(restored)
        authority = _authenticated_work_attempt_invocation(restored)
        assert authority.admission == admission
        restored.run_semantics.request_metadata["caller"] = True
        authority.admission.run_semantics.request_metadata["returned"] = True
        assert authority.admission.run_semantics.request_metadata == {}
        assert copy.copy(authority) is authority
        assert copy.deepcopy(authority) is authority
        with pytest.raises(FrozenInstanceError):
            authority._admission_json = "{}"
        with pytest.raises(TypeError, match="no serialization form"):
            pickle.dumps(authority)

        app = CayuApp(enable_logging=False)
        provider = _NeverCalledProvider()
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        active = ActiveInvocationExecutionProfile(
            session_id=admission.session_id,
            interaction_id=admission.interaction_id,
            run_epoch=1,
            profile=profile,
        )
        arguments = dict(
            active_profile=active,
            binding=PreparedInvocationBinding(
                session_id=admission.session_id,
                session_instance_id=session_invocation.session_instance_id,
                interaction_id=admission.interaction_id,
                run_epoch=1,
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
                runtime_name="cayu",
                runtime_version="test",
                runtime_build_provenance=profile.runtime_build_provenance,
                environment_name=None,
            ),
            validated_profile=active.profile,
            registered_agent=app._agents["assistant"],
            registered_provider=app._providers[provider.name],
            registered_environment=None,
            runtime_hooks=app._runtime_hooks,
            loop_policies=app._loop_policies,
            request_loop_policies=(),
            budget_policy=app.budget_policy,
            tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("original_tool",)),
            recovery_claim_id="recovery-claim",
        )
        with pytest.raises(TypeError, match="authenticated work-attempt"):
            _authenticated_invocation_context(**arguments, work_attempt=admission)
        context = _authenticated_invocation_context(**arguments, work_attempt=authority)
        assert context.work_attempt is authority
        assert context.without_recovery_claim().work_attempt is authority
        sessions = InMemorySessionStore()
        created = await sessions.apply_invocation_lifecycle_command(
            _create_command(
                session_id=admission.session_id,
                session_instance_id=session_invocation.session_instance_id,
                interaction_id=admission.interaction_id,
                profile=profile,
            )
        )
        admitted = context.with_admitted_session(created.session)
        assert admitted.work_attempt is authority
        rebound = admitted.with_rebound_session(
            created.session.model_copy(update={"run_epoch": 2}),
            active_profile=active.model_copy(update={"run_epoch": 2}),
        )
        assert rebound.work_attempt is authority
        with pytest.raises(ValueError, match="Work-attempt authority conflicts"):
            admitted.with_queued_interaction(
                created.session,
                active_profile=active.model_copy(
                    update={"interaction_id": "unrelated-interaction"}
                ),
            )

    asyncio.run(scenario())
