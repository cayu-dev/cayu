"""Verified workers leave queued work recoverable when their application shuts down."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_completion_result_resolvers import _Resolver
from tests.core.test_completion_verifier_adapters import (
    RecordingVerifier,
    _accepted_decision,
    _contract,
)
from tests.core.test_verified_work_contracts import (
    _RecordingProvider,
    _result_reference,
    _task_result,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.runtime.verified_task_worker import VerifiedTaskWorker
from cayu.sessions.base import InMemorySessionStore
from cayu.tasks.base import InMemoryTaskStore, TaskCreate, TaskStatus


async def _app_with_task():
    from examples.verified_task_handler import ReferencedResultHandler

    sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
    app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
    provider = _RecordingProvider()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
    contract = _contract()
    await tasks.publish_work_contract(contract)
    task = await tasks.create_task(TaskCreate(type="verified", work_contract=contract.reference()))
    app.register_completion_verifier(contract.verifier, RecordingVerifier(_accepted_decision()))
    app.register_completion_result_resolver(contract.result_resolver, _Resolver(_task_result()))

    async def candidate(_context):
        return _result_reference()

    return app, tasks, task, provider, ReferencedResultHandler("worker", candidate)


def test_a_worker_on_a_closing_app_claims_nothing() -> None:
    async def scenario() -> None:
        app, tasks, task, provider, handler = await _app_with_task()
        app.seal_admissions()
        async with VerifiedTaskWorker(app, handler, worker_id="worker") as worker:
            assert await asyncio.wait_for(worker.run(max_tasks=1), 15) == 0
        current = await tasks.load_task(task.id)
        assert current is not None and current.status is TaskStatus.PENDING
        assert provider.requests == []

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["prepare", "execute"])
def test_shutdown_mid_attempt_records_no_terminal_outcome(phase: str) -> None:
    async def scenario() -> None:
        app, tasks, task, provider, handler = await _app_with_task()
        reached: list[str] = []
        if phase == "prepare":
            prepare = handler.prepare

            async def sealing_prepare(context):
                # Shutdown begins after the claim, before admission.
                reached.append(phase)
                app.seal_admissions()
                return await prepare(context)

            handler.prepare = sealing_prepare
        else:

            def sealing_execute(request):
                # Shutdown begins after admission, before execution starts.
                reached.append(phase)
                app.seal_admissions()
                return CayuApp._execute_work_attempt(app, request)

            app._execute_work_attempt = sealing_execute
        async with VerifiedTaskWorker(app, handler, worker_id="worker") as worker:
            assert await asyncio.wait_for(worker.run(max_tasks=1), 15) == 0
        assert reached == [phase]
        current = await tasks.load_task(task.id)
        assert current is not None
        assert current.status not in {TaskStatus.FAILED, TaskStatus.COMPLETED}
        assert provider.requests == []

    asyncio.run(scenario())


def test_a_worker_owned_verifier_failure_is_reported_only_by_the_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.core.test_verified_task_worker import _StaticHandler

    from cayu.verification import verified_task_worker as worker_module

    monkeypatch.setattr(worker_module, "_VERIFIER_TIMEOUT_SECONDS", 0.05)

    class LateFailingVerifier(RecordingVerifier):
        def __init__(self) -> None:
            super().__init__(_accepted_decision())
            self.cancelled = asyncio.Event()
            self.release = asyncio.Event()

        async def verify(self, request):
            del request
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                await self.release.wait()
                raise ValueError("late worker verifier failure") from None
            raise AssertionError("unreachable")

    async def scenario() -> None:
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        app.register_provider(_RecordingProvider(), default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        contract = _contract()
        verifier = LateFailingVerifier()
        app.register_completion_verifier(contract.verifier, verifier)
        app.register_completion_result_resolver(contract.result_resolver, _Resolver(_task_result()))
        await tasks.publish_work_contract(contract)
        await tasks.create_task(TaskCreate(type="verified", work_contract=contract.reference()))
        worker = VerifiedTaskWorker(
            app, _StaticHandler(), worker_id="w", callback_timeout_seconds=0.2
        )
        with pytest.raises(Exception):
            await worker.run(max_tasks=1)
        await asyncio.wait_for(verifier.cancelled.wait(), 5)
        verifier.release.set()

        # The worker holds this drain, so the app waits for it but does not report it.
        assert await app.drain_verified_completions(timeout_s=5) is True
        with pytest.raises(Exception, match="late worker verifier failure"):
            await worker.aclose()

    asyncio.run(scenario())
