"""Failover remains part of exact verified-worker source and execution state."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_completion_result_resolvers import _Resolver
from tests.core.test_completion_verifier_adapters import (
    RecordingVerifier,
    _accepted_decision,
    _contract,
)
from tests.core.test_model_failover_recovery import _RecoveryProvider
from tests.core.test_verified_task_worker import _StaticHandler
from tests.core.test_verified_work_contracts import _task_result
from tests.core.verified_worker_fixtures import (
    verified_work_postgres_dsn as verified_work_postgres_dsn,
)
from tests.core.verified_worker_fixtures import (
    verified_worker_store_factory as verified_worker_store_factory,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import EventType
from cayu.runtime.retry_policy import RetryPolicy
from cayu.runtime.verified_task_worker import VerifiedTaskWorker
from cayu.sessions.base import EventQuery, ModelFailoverPolicy, ModelTarget
from cayu.tasks.base import TaskCreate, TaskStatus


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_public_verified_worker_executes_and_retains_failover(
    backend, verified_worker_store_factory
):
    policy = ModelFailoverPolicy(
        fallbacks=(ModelTarget(provider_name="backup", model="large"),), max_total_attempts=2
    )

    class Handler(_StaticHandler):
        async def prepare(self, context):
            request = await super().prepare(context)
            return request.model_copy(
                update={"failover": policy, "retry_policy": RetryPolicy(max_attempts=1)}
            )

    async def scenario():
        sessions, tasks = verified_worker_store_factory()
        try:
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            primary, backup = _RecoveryProvider("primary"), _RecoveryProvider("backup")
            app.register_provider(primary, default=True)
            app.register_provider(backup)
            app.register_agent(AgentSpec(name="worker", model="small"))
            contract = _contract()
            await tasks.publish_work_contract(contract)
            task = await tasks.create_task(
                TaskCreate(type="verified", work_contract=contract.reference())
            )
            verifier, resolver = RecordingVerifier(_accepted_decision()), _Resolver(_task_result())
            app.register_completion_verifier(contract.verifier, verifier)
            app.register_completion_result_resolver(contract.result_resolver, resolver)
            handler = Handler()
            async with VerifiedTaskWorker(app, handler, worker_id="failover-worker") as worker:
                assert await asyncio.wait_for(worker.run(max_tasks=1), 60) == 1
            final = await tasks.load_task(task.id)
            assert final.status is TaskStatus.COMPLETED
            assert len(primary.requests) == len(backup.requests) == 1
            assert len(handler.proposals) == len(verifier.requests) == len(resolver.requests) == 1
            admission = await tasks.load_latest_work_attempt_admission(task.id)
            assert admission is not None and admission.source_request is not None
            assert admission.run_semantics is not None
            assert admission.run_semantics.failover == policy
            assert admission.source_request.request["failover"] == policy.model_dump(mode="json")
            assert "failover" in admission.source_request.fields_set
            session = await sessions.load(admission.session_id)
            assert session is not None and session.provider_name == "primary"
            checkpoint = await sessions.load_checkpoint(admission.session_id)
            assert checkpoint["model_failover"]["candidate_index"] == 1
            events = await sessions.query_events(
                EventQuery(session_id=admission.session_id, limit=100)
            )
            assert (
                sum(record.event.type is EventType.MODEL_FAILOVER_SELECTED for record in events)
                == 2
            )
            if backend != "memory":
                await sessions.close()
                await tasks.close()
                sessions, tasks = verified_worker_store_factory()
                assert await tasks.load_latest_work_attempt_admission(task.id) == admission
                assert await sessions.load_checkpoint(admission.session_id) == checkpoint
                assert await tasks.load_task(task.id) == final
        finally:
            if backend != "memory":
                await sessions.close()
                await tasks.close()

    asyncio.run(scenario())
