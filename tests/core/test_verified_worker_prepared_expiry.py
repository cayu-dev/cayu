"""Expired prepared admission recovery through the public worker."""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from tests.core.test_completion_result_resolvers import _Resolver
from tests.core.test_completion_verifier_adapters import _contract, _rejected_decision
from tests.core.test_verified_task_worker import _ContinueOnceVerifier, _StaticHandler
from tests.core.test_verified_work_contracts import _RecordingProvider, _task_result
from tests.core.test_work_attempt_lifecycle import _active_settlement_fixture
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
    CayuApp,
    CompletionContinuationPolicy,
    CompletionRejectionAction,
    Message,
    RunRequest,
    TaskCreate,
    TaskStatus,
    VerifiedTaskWorker,
)
from cayu.runtime.sessions import copy_run_request
from cayu.runtime.work_attempt_admission import (
    WorkAttemptAdmissionState,
    WorkAttemptExecutionEntryDisposition,
    WorkAttemptExecutionEntryRequest,
    require_work_attempt_execution_entry_result,
)


def _crash_prepared_expiry(
    directory, continuing, session_mutated, postgres_dsn=None, recovery_entry=False
):
    async def scenario():
        sessions, tasks = VerifiedWorkerStoreFactory(
            "postgres" if postgres_dsn else "sqlite", Path(directory), postgres_dsn
        )()
        contract = _contract(
            continuation_policy=CompletionContinuationPolicy(
                rejection_action=CompletionRejectionAction.CONTINUE,
                max_attempts=3,
                max_repeated_gap_count=3,
            )
        )
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        provider = _RecordingProvider()
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        verifier = _ContinueOnceVerifier(_rejected_decision())
        app.register_completion_verifier(contract.verifier, verifier)
        app.register_completion_result_resolver(contract.result_resolver, _Resolver(_task_result()))
        handler = _StaticHandler()
        prepare = type(tasks).prepare_work_attempt_admission
        activate = type(app._session_engine)._activate_runtime_work_attempt
        enter = type(tasks).enter_work_attempt_execution

        async def crash_entered(store, request):
            result = await enter(store, request)
            assert result.admission.execution_stop.request.reason == "elapsed_limit"
            assert handler.preparations == handler.proposals == []
            assert provider.requests == verifier.requests == []
            os._exit(83)

        def crash_if_target(admission):
            if (admission.kind == "continuation") is continuing:
                assert len(handler.preparations) == 1
                assert (
                    len(handler.proposals)
                    == len(provider.requests)
                    == len(verifier.requests)
                    == int(continuing)
                )
                os._exit(82)

        async def crash_prepared(store, request):
            result = await prepare(store, request)
            crash_if_target(result)
            return result

        async def crash_created(engine, **kwargs):
            crash_if_target(kwargs["admission"])
            return await activate(engine, **kwargs)

        if recovery_entry:
            type(tasks).enter_work_attempt_execution = crash_entered
        elif session_mutated:
            type(app._session_engine)._activate_runtime_work_attempt = crash_created
        else:
            type(tasks).prepare_work_attempt_admission = crash_prepared
        async with VerifiedTaskWorker(
            app,
            handler,
            worker_id="prepared-expiry-child",
            lease_seconds=5,
            callback_timeout_seconds=1,
            max_elapsed_seconds=8,
        ) as worker:
            await worker.run(max_tasks=1)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("continuing", [False, True])
@pytest.mark.parametrize("session_mutated", [False, True])
def test_worker_settles_expired_prepared_admission(
    backend,
    continuing,
    session_mutated,
    verified_worker_store_factory,
    monkeypatch,
    probe_creation=False,
    process_loss=False,
    second_crash=False,
):
    async def scenario():
        sessions, tasks = verified_worker_store_factory()
        contract = _contract(
            continuation_policy=CompletionContinuationPolicy(
                rejection_action=CompletionRejectionAction.CONTINUE,
                max_attempts=3,
                max_repeated_gap_count=3,
            )
        )
        provider = _RecordingProvider()
        verifier = _ContinueOnceVerifier(_rejected_decision())

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
            original_app = make_app()
            handler = _StaticHandler()
            prepare = type(tasks).prepare_work_attempt_admission
            activate = type(original_app._session_engine)._activate_runtime_work_attempt

            def target(admission):
                return (admission.kind == "continuation") is continuing

            async def lose_preparation_reply(store, request):
                result = await prepare(store, request)
                if target(result):
                    raise ConnectionError("prepared admission committed")
                return result

            async def stop_before_activation(engine, **kwargs):
                if target(kwargs["admission"]):
                    raise ConnectionError("session mutation committed before activation")
                return await activate(engine, **kwargs)

            async def run_crash_child(recovery_entry=False):
                repository = Path(__file__).resolve().parents[2]
                environment = {**os.environ, "PYTHONPATH": str(repository / "src")}
                environment.pop("CAYU_TEST_VERIFIED_WORKER_DSN", None)
                if verified_worker_store_factory.postgres_dsn is not None:
                    environment["CAYU_TEST_VERIFIED_WORKER_DSN"] = (
                        verified_worker_store_factory.postgres_dsn
                    )
                child = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    "from tests.core.test_verified_worker_prepared_expiry import _crash_prepared_expiry; "
                    "import os, sys; _crash_prepared_expiry(sys.argv[1], sys.argv[2] == 'True', "
                    "sys.argv[3] == 'True', os.environ.get('CAYU_TEST_VERIFIED_WORKER_DSN'), "
                    "sys.argv[4] == 'True')",
                    str(verified_worker_store_factory.directory),
                    str(continuing),
                    str(session_mutated),
                    str(recovery_entry),
                    cwd=repository,
                    env=environment,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(child.communicate(), 60)
                    assert child.returncode == (83 if recovery_entry else 82), (stdout, stderr)
                finally:
                    if child.returncode is None:
                        child.kill()
                        await child.communicate()

            if process_loss:
                await tasks.close()
                await sessions.close()
                await run_crash_child()
                sessions, tasks = verified_worker_store_factory()
            with monkeypatch.context() as faults:
                if session_mutated:
                    faults.setattr(
                        type(original_app._session_engine),
                        "_activate_runtime_work_attempt",
                        stop_before_activation,
                    )
                else:
                    faults.setattr(
                        type(tasks), "prepare_work_attempt_admission", lose_preparation_reply
                    )
                async with VerifiedTaskWorker(
                    original_app,
                    handler,
                    worker_id="prepared-expiry-original",
                    lease_seconds=5,
                    callback_timeout_seconds=1,
                    max_elapsed_seconds=8,
                ) as worker:
                    if not process_loss:
                        with pytest.raises(Exception):
                            await worker.run(max_tasks=1)
            prepared = await tasks.load_latest_work_attempt_admission(task.id)
            assert prepared.state is WorkAttemptAdmissionState.PREPARING
            assert prepared.execution_entry is None
            assert target(prepared)
            session = await sessions.load(prepared.session_id)
            assert (session is not None) is (session_mutated or continuing)
            assert len(handler.preparations) == int(not process_loss)
            assert (
                len(provider.requests)
                == len(handler.proposals)
                == len(verifier.requests)
                == int(continuing and not process_loss)
            )
            await wait_for_verified_worker_lease_expiry(tasks, prepared.claim.lease_expires_at)
            await asyncio.sleep(
                max(
                    0,
                    (
                        prepared.run_semantics.deadline_expires_at - datetime.now(UTC)
                    ).total_seconds(),
                )
                + 0.05
            )
            if backend != "memory":
                await tasks.close()
                await sessions.close()
                if second_crash:
                    await run_crash_child(recovery_entry=True)
                sessions, tasks = verified_worker_store_factory()
            stopped = None
            if second_crash:
                stopped = await tasks.load_latest_work_attempt_admission(task.id)
                assert stopped.state is WorkAttemptAdmissionState.ACTIVE
                assert stopped.execution_entry is not None
                assert stopped.execution_stop.request.reason == "elapsed_limit"
                assert stopped.claim.generation == prepared.claim.generation + 1
                replay = await tasks.enter_work_attempt_execution(stopped.execution_entry.request)
                assert replay.admission == stopped
                await wait_for_verified_worker_lease_expiry(tasks, stopped.claim.lease_expires_at)
            replacement = make_app()
            replacement_handler = _StaticHandler()
            settle = type(tasks).settle_work_attempt_lifecycle
            lost_reply = False
            probes = []
            create = type(sessions).create

            async def inspect_creation(store, request, **kwargs):
                assert request.execution_deadline.expired
                for variant in (
                    "raw",
                    "serialized",
                    "deep",
                    "messages",
                    "private",
                    "deadline",
                    "identity",
                ):
                    candidate = copy_run_request(request)
                    arguments = dict(kwargs)
                    if variant == "raw":
                        candidate._runtime_work_attempt_creation = None
                    elif variant == "serialized":
                        pass  # Reconstruction itself may reject lost private task authority.
                    elif variant == "deep":
                        candidate = request.model_copy(deep=True)
                    elif variant == "messages":
                        candidate.messages.append(Message.text("user", "prepared-expiry-canary"))
                    elif variant == "private":
                        candidate._runtime_initial_transcript_authority = None
                    elif variant == "deadline":
                        candidate.execution_deadline = candidate.execution_deadline.model_copy(
                            update={
                                "expires_at": candidate.execution_deadline.expires_at
                                + timedelta(seconds=60)
                            }
                        )
                    else:
                        arguments["identity"] = kwargs["identity"].model_copy(
                            update={"model": "conflicting-model"}
                        )
                    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
                        if variant == "serialized":
                            candidate = RunRequest.model_validate(request.model_dump(mode="json"))
                        await create(store, candidate, **arguments)
                    assert await store.load(request.session_id) is None
                    probes.append(variant)
                # The maintained copier preserves all private authorities. Generic
                # deep copying loses other existing request tokens and fails closed.
                return await create(store, copy_run_request(request), **kwargs)

            async def lose_settlement_reply(store, request):
                nonlocal lost_reply
                result = await settle(store, request)
                if not lost_reply:
                    lost_reply = True
                    raise ConnectionError("expiry receipt acknowledgement lost")
                return result

            with monkeypatch.context() as faults:
                if probe_creation:
                    faults.setattr(type(sessions), "create", inspect_creation)
                if session_mutated:
                    faults.setattr(
                        type(tasks), "settle_work_attempt_lifecycle", lose_settlement_reply
                    )
                async with VerifiedTaskWorker(
                    replacement,
                    replacement_handler,
                    worker_id="prepared-expiry-replacement",
                    lease_seconds=5,
                    callback_timeout_seconds=1,
                ) as worker:
                    assert await asyncio.wait_for(worker.run(max_tasks=1), 60) == 1
            final = await tasks.load_task(task.id)
            admission = await tasks.load_latest_work_attempt_admission(task.id)
            receipt = await tasks.load_work_attempt_lifecycle_receipt(prepared.admission_id)
            assert final.status is TaskStatus.NEEDS_ATTENTION
            assert final.status_reason == "work_contract_elapsed_limit"
            assert receipt.task == final and not receipt.retired_contract_binding
            assert admission.admission_id == prepared.admission_id
            assert admission.run_semantics == prepared.run_semantics
            assert admission.execution_stop.request.reason == "elapsed_limit"
            if stopped is not None:
                assert admission.execution_stop == stopped.execution_stop
                assert admission.execution_entry == stopped.execution_entry
            assert (
                await sessions.load(admission.session_id)
            ).execution_deadline.model_dump() == prepared.run_semantics.deadline.model_dump()
            assert (
                len(provider.requests)
                == len(verifier.requests)
                == int(continuing and not process_loss)
            )
            assert probes == (
                ["raw", "serialized", "deep", "messages", "private", "deadline", "identity"]
                if probe_creation
                else []
            )
            assert replacement_handler.preparations == replacement_handler.proposals == []
            assert await tasks.load_completion_proposal_for_attempt(prepared.attempt_id) is None
            assert lost_reply is session_mutated
            if backend != "memory":
                await tasks.close()
                await sessions.close()
                sessions, tasks = verified_worker_store_factory()
            assert await tasks.settle_work_attempt_lifecycle(receipt.request) == receipt
            assert await tasks.load_task(task.id) == final
        finally:
            if backend != "memory":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_prepared_expiry_creation_requires_exact_private_authority(
    backend, verified_worker_store_factory, monkeypatch, capsys, caplog, recwarn
):
    test_worker_settles_expired_prepared_admission(
        backend, False, False, verified_worker_store_factory, monkeypatch, probe_creation=True
    )
    captured = capsys.readouterr()
    assert "prepared-expiry-canary" not in captured.out + captured.err + caplog.text
    assert all("prepared-expiry-canary" not in str(warning.message) for warning in recwarn)


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("continuing", [False, True])
@pytest.mark.parametrize("session_mutated", [False, True])
def test_prepared_expiry_after_real_process_exit(
    backend, continuing, session_mutated, verified_worker_store_factory, monkeypatch
):
    test_worker_settles_expired_prepared_admission(
        backend,
        continuing,
        session_mutated,
        verified_worker_store_factory,
        monkeypatch,
        process_loss=True,
    )


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("continuing", [False, True])
def test_prepared_expiry_survives_second_exit_after_execution_entry(
    backend, continuing, verified_worker_store_factory, monkeypatch
):
    test_worker_settles_expired_prepared_admission(
        backend,
        continuing,
        False,
        verified_worker_store_factory,
        monkeypatch,
        process_loss=True,
        second_crash=True,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_expired_execution_entry_atomically_publishes_exact_stop(
    backend, verified_worker_store_factory
):
    async def scenario():
        sessions, tasks = verified_worker_store_factory()
        try:
            admission, _ = await _active_settlement_fixture(
                tasks, deadline=datetime.now(UTC) - timedelta(seconds=1)
            )
            claim = admission.claim
            request = WorkAttemptExecutionEntryRequest(
                admission_id=admission.admission_id,
                prepare_request_sha256=admission.prepare_request_sha256,
                claim_id=claim.claim_id,
                worker_id=claim.worker_id,
                execution_owner_id=claim.execution_owner_id,
                generation=claim.generation,
                run_epoch=1,
            )
            results = await asyncio.gather(
                *(tasks.enter_work_attempt_execution(request) for _ in range(3))
            )
            entered = [
                result
                for result in results
                if result.disposition is WorkAttemptExecutionEntryDisposition.ENTERED
            ]
            assert len(entered) == 1
            result = entered[0]
            stop = result.admission.execution_stop
            assert stop.request.reason == "elapsed_limit"
            assert stop.request.execution_entry == result.admission.execution_entry
            assert stop.recorded_at == result.admission.execution_entry.entered_at
            for observed in results:
                assert (
                    require_work_attempt_execution_entry_result(observed, admission, request)
                    == observed
                )
                assert observed.admission == result.admission
            for changed in (
                None,
                stop.model_copy(
                    update={"request": stop.request.model_copy(update={"reason": "budget_limit"})}
                ),
                stop.model_copy(update={"recorded_at": stop.recorded_at + timedelta(seconds=1)}),
            ):
                invalid = result.model_copy(
                    update={
                        "admission": result.admission.model_copy(update={"execution_stop": changed})
                    }
                )
                with pytest.raises(RuntimeError, match="deadline-stop authority"):
                    require_work_attempt_execution_entry_result(invalid, admission, request)
            if backend != "memory":
                await tasks.close()
                await sessions.close()
                sessions, tasks = verified_worker_store_factory()
            replay = await tasks.enter_work_attempt_execution(request)
            assert replay.disposition is WorkAttemptExecutionEntryDisposition.ALREADY_ENTERED
            assert replay.admission == result.admission
        finally:
            if backend != "memory":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())
