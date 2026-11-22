"""A fresh governed attempt must not replay the prior attempt's structured tail."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_verified_task_worker import _StructuredReplayProvider
from tests.core.test_verified_work_contracts import (
    _artifact_evidence,
    _claim_completion_verification,
    _contract,
    _rejected_decision,
    _result_reference,
    _verifier_profile_fingerprint,
)

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest
from cayu.context.structured_output import StructuredOutputSpec
from cayu.sessions import InMemorySessionStore
from cayu.storage.sqlite import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks.admission import (
    WorkAttemptExecutionRequest,
    WorkAttemptProposalRequest,
    WorkAttemptRunRequest,
)
from cayu.tasks.contracts import (
    CompletionDecisionApplicationRequest,
    CompletionProposalCreate,
    CompletionVerificationClaimRequest,
)
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_rejected_structured_attempt_can_execute_its_fresh_continuation(tmp_path, backend):
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
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        provider = _StructuredReplayProvider("tool_valid")
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        contract = _contract(contract_id="structured-continuation-contract")
        await tasks.publish_work_contract(contract)
        task = await tasks.create_task(
            TaskCreate(
                task_id="structured-continuation-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )

        async def execute(admission):
            return [
                event
                async for event in app._execute_work_attempt(
                    WorkAttemptRunRequest(
                        admission_id=admission.admission_id,
                        claim_id=admission.claim.claim_id,
                        worker_id=admission.claim.worker_id,
                        generation=admission.claim.generation,
                        lease_seconds=300,
                    )
                )
            ]

        try:
            first = await app.admit_work_attempt(
                RunRequest(
                    agent_name="worker",
                    task_id=task.id,
                    session_id="structured-continuation-session",
                    messages=[Message.text("user", "Produce the first result.")],
                    structured_output=StructuredOutputSpec(
                        strategy="tool",
                        json_schema={
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                    ),
                ),
                execution=WorkAttemptExecutionRequest(
                    admission_id="admission-1",
                    claim_id="claim-1",
                    attempt_id="attempt-1",
                    interaction_id="interaction-1",
                    worker_id="worker-1",
                    generation=1,
                    lease_seconds=300,
                ),
            )
            first_events = await execute(first)
            assert first_events[-1].type is EventType.SESSION_COMPLETED
            assert len(provider.requests) == 1
            proposal = await app.submit_work_attempt_proposal(
                WorkAttemptProposalRequest(
                    admission_id=first.admission_id,
                    claim_id=first.claim.claim_id,
                    generation=first.claim.generation,
                    proposal=CompletionProposalCreate(
                        proposal_id="proposal-1",
                        attempt_id=first.attempt_id,
                        result=_result_reference(),
                        evidence_references=(_artifact_evidence(),),
                    ),
                )
            )
            verifier_claim = await _claim_completion_verification(
                tasks,
                CompletionVerificationClaimRequest(
                    claim_id="verifier-claim",
                    proposal_id=proposal.proposal_id,
                    worker_id="verifier-worker",
                    verifier=contract.verifier,
                    verifier_profile_fingerprint=_verifier_profile_fingerprint(contract.verifier),
                ),
            )
            decision = await tasks.record_completion_decision(
                _rejected_decision(
                    proposal_id=proposal.proposal_id,
                    claim_id=verifier_claim.claim_id,
                    worker_id=verifier_claim.worker_id,
                    decision_id="rejected-decision",
                )
            )
            await tasks.apply_completion_decision(
                CompletionDecisionApplicationRequest(
                    task_id=task.id,
                    decision_id=decision.decision_id,
                    idempotency_key="apply-rejection",
                )
            )
            second = await app.admit_work_attempt(
                ResumeRequest(
                    session_id=first.session_id,
                    messages=[Message.text("user", "Address the verifier gaps.")],
                ),
                execution=WorkAttemptExecutionRequest(
                    admission_id="admission-2",
                    claim_id="claim-2",
                    attempt_id="attempt-2",
                    interaction_id="interaction-2",
                    worker_id="worker-2",
                    task_id=task.id,
                    predecessor_admission_id=first.admission_id,
                    generation=1,
                    lease_seconds=300,
                ),
            )
            second_events = await execute(second)
            assert second_events[-1].type is EventType.SESSION_COMPLETED, (
                second_events[-1].payload.get("error"),
                len(provider.requests),
            )
            assert len(provider.requests) == 2
        finally:
            await app.aclose()
            if backend == "sqlite":
                await tasks.close()
                await sessions.close()

    asyncio.run(scenario())
