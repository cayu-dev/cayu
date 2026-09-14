"""Memory cleanup follows task ownership, never cross-namespace string matches."""

import asyncio

from tests.core.session_closure_conformance import create_closure_session
from tests.core.task_invocation_fixtures import unattributed_session_invocation_binding
from tests.core.test_verified_work_contracts import (
    _accepted_decision,
    _approval_evidence,
    _artifact_evidence,
    _claim_completion_verification,
    _contract,
    _digest,
    _result_reference,
    _task_result,
    _verifier_profile_fingerprint,
)

from cayu import CayuApp, InMemoryTaskStore, TaskCreate
from cayu.tasks.base import TaskTopologyQuery
from cayu.tasks.contracts import (
    CompletionDecisionApplicationRequest,
    CompletionProposalCreate,
    CompletionVerificationClaimRequest,
    WorkAttemptCreate,
)


def test_public_memory_closure_preserves_other_namespaces():
    class ApplicationTaskStore(InMemoryTaskStore):
        def __init__(self):
            super().__init__()
            self.application_records = {
                "shared-id": {"private": "application-owned"},
                ("application", "shared-id"): "also retained",
            }

    async def scenario():
        tasks = ApplicationTaskStore()
        app = CayuApp(task_store=tasks, enable_logging=False)
        for session_id in ("root", "shared-id"):
            await create_closure_session(app.session_store, session_id)
        for task_id, session_id in (("shared-id", "root"), ("other-task", "shared-id")):
            await tasks.create_task(TaskCreate(task_id=task_id, type="test", session_id=session_id))
            await tasks.complete_task(task_id, {})
        contract = await tasks.publish_work_contract(_contract(contract_id="shared-id"))
        application_records = dict(tasks.application_records)
        query = TaskTopologyQuery(linked_session_ids=("shared-id",))
        before = await tasks.query_task_topology(query)
        assert before.session_branches

        result = await app.erase_session_closure("root")
        assert result.complete
        assert await tasks.load_task("shared-id") is None
        assert await tasks.load_task("other-task") is not None
        assert (await tasks.query_task_topology(query)).session_branches == before.session_branches
        assert await tasks.load_work_contract(contract.reference()) == contract
        assert tasks.application_records == application_records
        replay = await app.erase_session_closure("root")
        assert replay.complete and replay.already_absent
        assert (await tasks.query_task_topology(query)).session_branches == before.session_branches
        assert (await app.erase_session_closure("shared-id")).complete
        assert await tasks.load_task("other-task") is None
        assert tasks.application_records == application_records
        assert await tasks.load_work_contract(contract.reference()) == contract

    asyncio.run(scenario())


def test_memory_closure_removes_owned_verified_work_dependencies():
    async def scenario():
        tasks = InMemoryTaskStore()
        app = CayuApp(task_store=tasks, enable_logging=False)
        await create_closure_session(app.session_store, "root")
        contract = await tasks.publish_work_contract(_contract())
        task = await tasks.create_running_task(
            TaskCreate(
                task_id="selected",
                type="verified",
                session_id="root",
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding("root"),
        )
        attempt = await tasks.begin_work_attempt(
            WorkAttemptCreate(
                attempt_id="attempt",
                task_id=task.id,
                session_id="root",
                contract=contract.reference(),
                execution_profile_fingerprint=_digest("profile"),
            )
        )
        proposal = await tasks.submit_completion_proposal(
            CompletionProposalCreate(
                proposal_id="proposal",
                attempt_id=attempt.attempt_id,
                result=_result_reference("2"),
                evidence_references=(_artifact_evidence(), _approval_evidence()),
            )
        )
        claim = await _claim_completion_verification(
            tasks,
            CompletionVerificationClaimRequest(
                claim_id="claim",
                proposal_id=proposal.proposal_id,
                worker_id="verifier",
                verifier=contract.verifier,
                verifier_profile_fingerprint=_verifier_profile_fingerprint(contract.verifier),
            ),
        )
        decision = await tasks.record_completion_decision(
            _accepted_decision(
                proposal_id=proposal.proposal_id,
                claim_id=claim.claim_id,
                worker_id=claim.worker_id,
            )
        )
        await tasks.apply_completion_decision(
            CompletionDecisionApplicationRequest(
                task_id=task.id,
                decision_id=decision.decision_id,
                idempotency_key="apply",
                result=_task_result("2"),
                result_reference=proposal.result,
            )
        )
        assert (await app.erase_session_closure("root")).complete
        assert await tasks.load_work_attempt(attempt.attempt_id) is None
        assert await tasks.load_completion_proposal(proposal.proposal_id) is None
        assert await tasks.load_completion_verifier_profile(proposal.proposal_id) is None
        assert await tasks.load_completion_verification_claim(proposal.proposal_id) is None
        assert await tasks.load_completion_decision(decision.decision_id) is None
        assert await tasks.load_completion_decision_application_receipt(task.id, "apply") is None
        assert await tasks.load_active_work_contract_task_for_session("root") is None
        assert tasks._verification_claims_by_id == {}
        assert await tasks.load_work_contract(contract.reference()) == contract

    asyncio.run(scenario())
