"""Application-lifetime completion ownership preserves independent worker settlements."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_completion_result_resolvers import _Resolver
from tests.core.test_completion_verifier_adapters import (
    RecordingVerifier,
    _accepted_decision,
    _contract,
)
from tests.core.test_verified_task_worker import _StaticHandler
from tests.core.test_verified_work_contracts import _RecordingProvider, _task_result
from tests.core.verified_worker_fixtures import (
    verified_work_postgres_dsn as verified_work_postgres_dsn,
)
from tests.core.verified_worker_fixtures import (
    verified_worker_store_factory as verified_worker_store_factory,
)

from cayu import (
    AgentSpec,
    CayuApp,
    CompletionDecisionApplicationRequest,
    CompletionResultResolutionRequest,
    CompletionVerifierExecutionRequest,
    TaskCreate,
    TaskStatus,
    VerifiedTaskWorker,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_concurrent_worker_settlements_share_registration_and_preserve_public_replay(
    backend, verified_worker_store_factory, monkeypatch
):
    async def scenario():
        sessions, tasks = verified_worker_store_factory()
        both_entered = asyncio.Event()

        class ConcurrentVerifier(RecordingVerifier):
            entered = 0

            async def verify(self, request):
                self.entered += 1
                if self.entered == 2:
                    both_entered.set()
                await asyncio.wait_for(both_entered.wait(), 15)
                return await super().verify(request)

        try:
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            owner = app._verified_completion
            provider = _RecordingProvider()
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            contract = _contract()
            await tasks.publish_work_contract(contract)
            created = [
                await tasks.create_task(
                    TaskCreate(type="verified", work_contract=contract.reference())
                )
                for _ in range(2)
            ]
            verifier = ConcurrentVerifier(_accepted_decision())
            resolver = _Resolver(_task_result())
            app.register_completion_verifier(contract.verifier, verifier)
            app.register_completion_result_resolver(contract.result_resolver, resolver)

            async def application_round_trip(*args, **kwargs):
                raise AssertionError("Worker settlement must call its phase owners directly.")

            with monkeypatch.context() as patch:
                for method in (
                    "verify_completion_proposal",
                    "apply_completion_decision",
                    "resolve_completion_result",
                ):
                    patch.setattr(app, method, application_round_trip)
                async with (
                    VerifiedTaskWorker(app, _StaticHandler(), worker_id="first") as first,
                    VerifiedTaskWorker(app, _StaticHandler(), worker_id="second") as second,
                ):
                    assert first._completion is second._completion is owner
                    assert await asyncio.wait_for(
                        asyncio.gather(first.run(max_tasks=1), second.run(max_tasks=1)), 30
                    ) == [1, 1]

            for task in created:
                final = await tasks.load_task(task.id)
                assert final.status is TaskStatus.COMPLETED
                admission = await tasks.load_latest_work_attempt_admission(task.id)
                proposal = await tasks.load_completion_proposal_for_attempt(admission.attempt_id)
                decision = await tasks.load_completion_decision_for_proposal(proposal.proposal_id)
                receipt = await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
                assert receipt.task == final and receipt.retired_contract_binding
                assert receipt.request.decision_id == decision.decision_id
                assert (
                    await app.verify_completion_proposal(
                        CompletionVerifierExecutionRequest(
                            proposal_id=proposal.proposal_id,
                            claim_id=decision.claim_id,
                            decision_id=decision.decision_id,
                            worker_id=decision.worker_id,
                        )
                    )
                    == decision
                )
                assert (
                    await app.resolve_completion_result(
                        CompletionResultResolutionRequest(
                            task_id=task.id,
                            decision_id=decision.decision_id,
                            idempotency_key=receipt.request.application_idempotency_key,
                        )
                    )
                    == final
                )
                assert (
                    await app.apply_completion_decision(
                        CompletionDecisionApplicationRequest(
                            task_id=task.id,
                            decision_id=decision.decision_id,
                            idempotency_key=receipt.request.application_idempotency_key,
                            result=final.result,
                            result_reference=proposal.result,
                        )
                    )
                    == final
                )
            assert app._verified_completion is owner
            assert len(provider.requests) == len(verifier.requests) == len(resolver.requests) == 2
        finally:
            if backend != "memory":
                await sessions.close()
                await tasks.close()

    asyncio.run(scenario())
