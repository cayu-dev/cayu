"""Session closure removes every Cayu-owned verified-work record of its tasks."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest
from tests.core.session_closure_conformance import create_closure_session
from tests.core.task_invocation_fixtures import unattributed_session_invocation_binding
from tests.core.test_completion_result_resolvers import _Resolver
from tests.core.test_completion_verifier_adapters import (
    RecordingVerifier,
    _accepted_decision,
    _contract,
    _digest,
)
from tests.core.test_verified_work_contracts import (
    _RecordingProvider,
    _result_reference,
    _task_result,
)

from cayu import AgentSpec, CayuApp
from cayu.sessions.base import InMemorySessionStore
from cayu.storage.migrations import SchemaMode
from cayu.storage.sqlite import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks.base import InMemoryTaskStore
from cayu.tasks.contracts import (
    CompletionDecisionApplicationRequest,
    CompletionProposalCreate,
    CompletionResultReference,
    WorkAttemptCreate,
    WorkContract,
    completion_result_sha256,
)
from cayu.tasks.creation import TaskCreate
from cayu.tasks.records import TaskStatus
from cayu.verification.completion_verifiers import CompletionVerifierExecutionRequest

BACKENDS = ["memory", "sqlite", "postgres"]


def _stores(backend: str, tmp_path: Path, request: pytest.FixtureRequest):
    if backend == "memory":
        return InMemorySessionStore(), InMemoryTaskStore()
    if backend == "sqlite":
        return (
            SQLiteSessionStore(tmp_path / "closure.sqlite"),
            SQLiteTaskStore(tmp_path / "closure.sqlite"),
        )
    from cayu.storage.postgres import PostgresSessionStore, PostgresTaskStore

    dsn = request.getfixturevalue("postgres_dsn")
    return (
        PostgresSessionStore(dsn, schema_mode=SchemaMode.MIGRATE),
        PostgresTaskStore(dsn, schema_mode=SchemaMode.MIGRATE),
    )


async def _close(sessions, tasks) -> None:
    for store in (tasks, sessions):
        close = getattr(store, "close", None)
        if close is not None:
            await close()


async def _assert_verified_work_erased(tasks, *, task_id: str, proposal_id: str) -> None:
    assert await tasks.load_task(task_id) is None
    assert await tasks.load_completion_proposal(proposal_id) is None
    assert await tasks.load_completion_verification_claim(proposal_id) is None
    assert await tasks.load_completion_verifier_profile(proposal_id) is None
    assert await tasks.load_completion_decision_for_proposal(proposal_id) is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_closure_erases_a_verified_and_applied_contract_task(backend, tmp_path, request):
    async def scenario():
        sessions, tasks = _stores(backend, tmp_path, request)
        suffix = uuid4().hex[:12]
        try:
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            session_id = f"closure-root-{suffix}"
            await create_closure_session(sessions, session_id)
            contract = await tasks.publish_work_contract(_contract())
            task = await tasks.create_running_task(
                TaskCreate(
                    task_id=f"closure-task-{suffix}",
                    type="verified",
                    session_id=session_id,
                    work_contract=contract.reference(),
                ),
                session_invocation=unattributed_session_invocation_binding(session_id),
            )
            attempt = await tasks.begin_work_attempt(
                WorkAttemptCreate(
                    attempt_id=f"closure-attempt-{suffix}",
                    task_id=task.id,
                    session_id=session_id,
                    contract=contract.reference(),
                    execution_profile_fingerprint=_digest("worker-profile"),
                )
            )
            proposal = await tasks.submit_completion_proposal(
                CompletionProposalCreate(
                    proposal_id=f"closure-proposal-{suffix}",
                    attempt_id=attempt.attempt_id,
                    result=_result_reference(),
                )
            )
            app.register_completion_verifier(
                contract.verifier, RecordingVerifier(_accepted_decision())
            )
            decision = await app.verify_completion_proposal(
                CompletionVerifierExecutionRequest(
                    proposal_id=proposal.proposal_id,
                    claim_id=f"closure-claim-{suffix}",
                    decision_id=f"closure-decision-{suffix}",
                    worker_id="closure-verifier",
                )
            )
            completed = await app.apply_completion_decision(
                CompletionDecisionApplicationRequest(
                    task_id=task.id,
                    decision_id=decision.decision_id,
                    idempotency_key=f"closure-apply-{suffix}",
                    result=_task_result(),
                    result_reference=proposal.result,
                )
            )
            assert completed.status is TaskStatus.COMPLETED

            report = await app.erase_session_closure(session_id)
            assert report.complete, report.error
            await _assert_verified_work_erased(
                tasks, task_id=task.id, proposal_id=proposal.proposal_id
            )
            assert await tasks.load_work_contract(contract.reference()) == contract
        finally:
            await _close(sessions, tasks)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_closure_erases_a_task_completed_by_the_verified_worker(backend, tmp_path, request):
    from examples.verified_task_handler import ReferencedResultHandler

    from cayu.runtime.verified_task_worker import VerifiedTaskWorker

    async def scenario():
        sessions, tasks = _stores(backend, tmp_path, request)
        suffix = uuid4().hex[:12]
        try:
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            app.register_provider(_RecordingProvider(), default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            contract = await tasks.publish_work_contract(_contract())
            task = await tasks.create_task(
                TaskCreate(
                    task_id=f"worker-closure-task-{suffix}",
                    type="verified",
                    work_contract=contract.reference(),
                )
            )
            app.register_completion_verifier(
                contract.verifier, RecordingVerifier(_accepted_decision())
            )
            app.register_completion_result_resolver(
                contract.result_resolver, _Resolver(_task_result())
            )

            async def candidate(context):
                return _result_reference()

            handler = ReferencedResultHandler("worker", candidate)
            async with VerifiedTaskWorker(app, handler, worker_id="closure-worker") as worker:
                assert await asyncio.wait_for(worker.run(max_tasks=1), 30) == 1
            final = await tasks.load_task(task.id)
            assert final is not None and final.status is TaskStatus.COMPLETED
            admission = await tasks.load_latest_work_attempt_admission(task.id)
            assert admission is not None
            proposal = await tasks.load_completion_proposal_for_attempt(admission.attempt_id)
            assert proposal is not None
            assert final.session_id is not None

            report = await app.erase_session_closure(final.session_id)
            assert report.complete, report.error
            await _assert_verified_work_erased(
                tasks, task_id=task.id, proposal_id=proposal.proposal_id
            )
            assert await tasks.load_work_attempt_admission(admission.admission_id) is None
        finally:
            await _close(sessions, tasks)

    asyncio.run(scenario())


def test_deletion_plan_covers_owned_tables_children_first(tmp_path):
    import sqlite3

    from cayu.storage._session_closure_sql import (
        TASK_CLOSURE_DELETION_PLAN,
        TASK_CLOSURE_DEPENDENCIES,
        task_closure_deletion_steps,
    )

    direct = {(table, column) for table, column, parent in TASK_CLOSURE_DELETION_PLAN if not parent}
    assert direct == TASK_CLOSURE_DEPENDENCIES
    for _table, _column, parent in TASK_CLOSURE_DELETION_PLAN:
        assert parent is None or (parent, "task_id") in TASK_CLOSURE_DEPENDENCIES
    assert task_closure_deletion_steps(set(TASK_CLOSURE_DEPENDENCIES)) == (
        TASK_CLOSURE_DELETION_PLAN
    )

    path = tmp_path / "schema.sqlite"
    asyncio.run(SQLiteTaskStore(path).close())
    connection = sqlite3.connect(path)
    try:
        order = [table for table, _column, _parent in TASK_CLOSURE_DELETION_PLAN]
        planned = set(order)
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name GLOB 'cayu_*'"
            )
        ]
        for table in tables:
            for foreign_key in connection.execute(f"PRAGMA foreign_key_list('{table}')"):
                parent, on_delete = foreign_key[2], foreign_key[6]
                if parent not in planned or on_delete == "CASCADE":
                    continue
                # Every restrictive child of a planned table is itself deleted
                # by the plan, before its parent.
                assert table in planned, (table, parent)
                assert order.index(table) < order.index(parent), (table, parent)
    finally:
        connection.close()


async def _verify_and_apply(
    app: CayuApp,
    tasks,
    contract: WorkContract,
    *,
    session_id: str,
    suffix: str,
) -> tuple[str, str]:
    """Run one contracted task through verification and decision application."""

    await create_closure_session(app.session_store, session_id)
    await tasks.publish_work_contract(contract)
    task = await tasks.create_running_task(
        TaskCreate(
            task_id=f"ledger-task-{suffix}",
            type="verified",
            session_id=session_id,
            work_contract=contract.reference(),
        ),
        session_invocation=unattributed_session_invocation_binding(session_id),
    )
    attempt = await tasks.begin_work_attempt(
        WorkAttemptCreate(
            attempt_id=f"ledger-attempt-{suffix}",
            task_id=task.id,
            session_id=session_id,
            contract=contract.reference(),
            execution_profile_fingerprint=_digest("worker-profile"),
        )
    )
    result = {"candidate": suffix}
    proposal = await tasks.submit_completion_proposal(
        CompletionProposalCreate(
            proposal_id=f"ledger-proposal-{suffix}",
            attempt_id=attempt.attempt_id,
            result=CompletionResultReference(
                kind="task.result",
                reference_id=f"ledger-result-{suffix}",
                digest=completion_result_sha256(result),
            ),
        )
    )
    decision = await app.verify_completion_proposal(
        CompletionVerifierExecutionRequest(
            proposal_id=proposal.proposal_id,
            claim_id=f"ledger-claim-{suffix}",
            decision_id=f"ledger-decision-{suffix}",
            worker_id="ledger-verifier",
            lease_seconds=60,
            execution_timeout_seconds=30.0,
        )
    )
    completed = await app.apply_completion_decision(
        CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision.decision_id,
            idempotency_key=f"ledger-apply-{suffix}",
            result=result,
            result_reference=proposal.result,
        )
    )
    assert completed.status is TaskStatus.COMPLETED
    return task.id, proposal.proposal_id


@pytest.mark.parametrize("backend", BACKENDS)
def test_closure_erases_provider_verifier_dispatches(backend, tmp_path, request):
    from tests.core.test_provider_completion_verifiers import (
        CountingProvider,
        Judge,
        _answer,
        _completion,
    )
    from tests.core.test_provider_completion_verifiers import _contract as _provider_contract
    from tests.core.test_provider_completion_verifiers import _reference as _provider_reference

    async def scenario():
        sessions, tasks = _stores(backend, tmp_path, request)
        suffix = uuid4().hex[:12]
        try:
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            app.register_provider(CountingProvider(_completion(_answer())))
            app.register_completion_verifier(_provider_reference(), Judge())
            session_id = f"ledger-session-{suffix}"
            task_id, proposal_id = await _verify_and_apply(
                app, tasks, _provider_contract(suffix), session_id=session_id, suffix=suffix
            )
            assert len(await tasks.list_completion_verifier_dispatches(proposal_id)) == 1

            report = await app.erase_session_closure(session_id)
            assert report.complete, report.error
            assert await tasks.list_completion_verifier_dispatches(proposal_id) == ()
            await _assert_verified_work_erased(tasks, task_id=task_id, proposal_id=proposal_id)
            if isinstance(tasks, InMemoryTaskStore):
                assert tasks._completion_verifier_dispatch_proposals == {}
        finally:
            await _close(sessions, tasks)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", BACKENDS)
def test_closure_erases_completion_evaluation_runs(backend, tmp_path, request):
    from tests.core.test_completion_evaluations import (
        Bench,
        Gate,
        _evaluator_reference,
        _score,
        _verifier_reference,
    )
    from tests.core.test_completion_evaluations import _contract as _evaluation_contract

    async def scenario():
        sessions, tasks = _stores(backend, tmp_path, request)
        suffix = uuid4().hex[:12]
        try:
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            app.register_completion_verifier(_verifier_reference(), Gate())
            app.register_completion_evaluator(_evaluator_reference(), Bench(_score(0.9)))
            session_id = f"ledger-session-{suffix}"
            task_id, proposal_id = await _verify_and_apply(
                app, tasks, _evaluation_contract(suffix), session_id=session_id, suffix=suffix
            )
            assert len(await tasks.list_completion_evaluation_runs(proposal_id)) == 1

            report = await app.erase_session_closure(session_id)
            assert report.complete, report.error
            assert await tasks.list_completion_evaluation_runs(proposal_id) == ()
            await _assert_verified_work_erased(tasks, task_id=task_id, proposal_id=proposal_id)
            if isinstance(tasks, InMemoryTaskStore):
                assert tasks._completion_evaluation_run_proposals == {}
        finally:
            await _close(sessions, tasks)

    asyncio.run(scenario())
