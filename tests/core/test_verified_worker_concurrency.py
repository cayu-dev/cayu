"""A rejected verified attempt waits for its key without stopping the worker."""

import asyncio

import pytest
from tests.core.test_completion_result_resolvers import _Resolver
from tests.core.test_completion_verifier_adapters import (
    RecordingVerifier,
    _accepted_decision,
    _contract,
    _rejected_decision,
)
from tests.core.test_verified_task_worker import _StaticHandler
from tests.core.test_verified_work_contracts import _RecordingProvider, _task_result
from tests.core.verified_worker_fixtures import (
    verified_work_postgres_dsn as verified_work_postgres_dsn,
)
from tests.core.verified_worker_fixtures import (
    verified_worker_store_factory as verified_worker_store_factory,
)

from cayu import AgentSpec, CayuApp
from cayu.runtime.verified_task_worker import VerifiedTaskWorker
from cayu.tasks import TaskCreate, TaskQuery, TaskStatus
from cayu.tasks.admission import WorkAttemptAdmissionState
from cayu.tasks.contracts import CompletionContinuationPolicy, CompletionRejectionAction


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_busy_key_defers_verified_continuation_and_worker_services_unrelated_task(
    backend, verified_worker_store_factory
):
    async def scenario():
        sessions, tasks = verified_worker_store_factory()
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        try:
            provider = _RecordingProvider()
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
            original = await tasks.create_task(
                TaskCreate(
                    type="verified",
                    concurrency_key="shared",
                    work_contract=contract.reference(),
                )
            )
            unrelated = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            peer = await tasks.create_task(TaskCreate(type="peer", concurrency_key="shared"))
            peer_claim = None

            class Verifier(RecordingVerifier):
                async def verify(self, request):
                    nonlocal peer_claim
                    self.requests.append(request)
                    if request.attempt.task_id == original.id and request.attempt.ordinal == 1:
                        peer_claim = await tasks.claim_task("peer-owner", TaskQuery(type="peer"))
                        assert peer_claim is not None and peer_claim.id == peer.id
                        return _rejected_decision()
                    return _accepted_decision()

            verifier = Verifier(_accepted_decision())
            app.register_completion_verifier(contract.verifier, verifier)
            app.register_completion_result_resolver(
                contract.result_resolver, _Resolver(_task_result())
            )
            handler = _StaticHandler()
            async with VerifiedTaskWorker(
                app, handler, worker_id="worker", poll_interval_s=0.01
            ) as worker:
                assert await asyncio.wait_for(worker.run(max_tasks=1), 20) == 1
                assert (await tasks.load_task(unrelated.id)).status is TaskStatus.COMPLETED
                waiting = await tasks.load_task(original.id)
                assert waiting.status is TaskStatus.RUNNING and waiting.worker_id is None
                predecessor = await tasks.load_latest_work_attempt_admission(original.id)
                assert predecessor.state is WorkAttemptAdmissionState.RELEASED
                assert len(provider.requests) == 2
                assert len(verifier.requests) == 2
                await tasks.release_task(
                    peer_claim.id,
                    peer_claim.worker_id,
                    lease_expires_at=peer_claim.lease_expires_at,
                )
                assert await asyncio.wait_for(worker.run(max_tasks=1), 20) == 1
            assert (await tasks.load_task(original.id)).status is TaskStatus.COMPLETED
            successor = await tasks.load_latest_work_attempt_admission(original.id)
            assert successor.continuation.prior_admission_id == predecessor.admission_id
            configuration = (await sessions.load_checkpoint(successor.session_id))[
                "invocation_configuration"
            ]
            assert configuration["max_steps"] == successor.run_semantics.max_steps
            assert configuration["limits"] == successor.run_semantics.limits.model_dump(mode="json")
            assert len(provider.requests) == len(verifier.requests) == 3
            assert len(handler.preparations) == 2
        finally:
            await app.aclose()
            if backend != "memory":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())
