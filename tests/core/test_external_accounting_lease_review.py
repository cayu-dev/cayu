"""Gateway receipts and worker identity survive independent observation/lease changes."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import httpx
import pytest
from fastapi import FastAPI
from tests.core.test_external_agents import ScriptedAdapter, pricing, request
from tests.core.test_external_lifecycle_review import _launch, _noop, _Result, _sdk

from cayu import CayuApp
from cayu.budgets import BudgetLimit
from cayu.events import EventType
from cayu.external_agents import ExternalAgentError, ExternalAgentEvent, run_external_agent
from cayu.external_agents.base import ExternalAgentBudgetExceeded
from cayu.external_agents.claude import ClaudeAgentAdapter
from cayu.external_agents.server import ExternalAgentGateway, ExternalModelRoute
from cayu.sessions import InterruptSessionRequest
from cayu.sessions.records import SessionStatus
from cayu.storage import SQLiteTaskStore
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskQuery, TaskStatus, run_task_worker


def test_empty_claude_output_schema_uses_structured_result(monkeypatch):
    async def scenario():
        async def receive(options):
            assert options.output_format == {"type": "json_schema", "schema": {}}
            yield _Result()

        _sdk(monkeypatch, receive, _noop)
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        try:
            result = await run_external_agent(
                app,
                request(),
                ClaudeAgentAdapter(model="claude", launch=_launch),
                prompt="review",
                output_schema={},
            )
            assert result.output == {"answer": 42}
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("outcome", ["complete", "fail", "interrupt"])
def test_external_settlement_keeps_original_managed_lease(monkeypatch, tmp_path, backend, outcome):
    async def scenario():
        app = CayuApp(
            task_store=InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.db"),
            enable_logging=False,
        )
        task = await app.create_task(TaskCreate(type="external"))
        generations = []
        attached = asyncio.Event()
        first_renewal = asyncio.Event()
        second_renewal = asyncio.Event()
        heartbeat = app.task_store.heartbeat

        async def renew(*args, **kwargs):
            current = await heartbeat(*args, **kwargs)
            generations.append(current.lease_expires_at)
            if len(generations) == 2:
                first_renewal.set()
            if len(generations) == 3:
                second_renewal.set()
            return current

        monkeypatch.setattr(app.task_store, "heartbeat", renew)
        sessions = []
        failures = []

        class Adapter(ScriptedAdapter):
            async def stream(self, context):
                sessions.append(context.session_id)
                attached.set()
                await second_renewal.wait()
                if outcome == "fail":
                    raise RuntimeError("vendor failed")
                if outcome == "interrupt":
                    await asyncio.Event().wait()
                yield ExternalAgentEvent("result", {"output": "done"}, "result")

        async def handle(current_app, claimed, worker_id):
            await first_renewal.wait()
            try:
                await run_external_agent(
                    current_app,
                    request(
                        task_id=claimed.id,
                        task_worker_id=worker_id,
                        task_lease_expires_at=claimed.lease_expires_at,
                    ),
                    Adapter([]),
                    prompt="review",
                )
            except ExternalAgentError as exc:
                failures.append(str(exc))
                if outcome != "fail":
                    raise

        worker = asyncio.create_task(
            run_task_worker(
                app,
                app.task_store,
                handle,
                worker_id="worker",
                lease_seconds=3,
                max_tasks=1,
                poll_interval_s=0.01,
            )
        )
        try:
            await asyncio.wait_for(attached.wait(), 10)
            intermediate = await app.task_store.load_task(task.id)
            assert intermediate.lease_expires_at == generations[1]
            await asyncio.wait_for(second_renewal.wait(), 10)
            if outcome == "interrupt":
                async for _ in app.interrupt_session(
                    InterruptSessionRequest(session_id=sessions[0])
                ):
                    pass
            await asyncio.wait_for(worker, 10)
            assert len(set(generations)) >= 3
            settled = await app.task_store.load_task(task.id)
            assert (
                settled.status
                is {
                    "complete": TaskStatus.COMPLETED,
                    "fail": TaskStatus.FAILED,
                    "interrupt": TaskStatus.CANCELLED,
                }[outcome]
            )
            assert (await app.session_store.load(sessions[0])).status is {
                "complete": SessionStatus.COMPLETED,
                "fail": SessionStatus.FAILED,
                "interrupt": SessionStatus.INTERRUPTED,
            }[outcome]
            assert (
                failures
                == {
                    "complete": [],
                    "fail": ["External harness failed (RuntimeError)."],
                    "interrupt": ["External session was interrupted by its operator."],
                }[outcome]
            )
        finally:
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("quota", ["identities", "bytes"])
def test_admitted_gateway_receipt_survives_sdk_observation_quota(quota):
    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        recorders = []
        calls = []

        async def upstream_response(upstream_request):
            calls.append(upstream_request)
            recorder = recorders[0]
            if quota == "identities":
                recorder.seen.update({f"sdk-{index}": "digest" for index in range(100_000)})
            else:
                recorder.observed_bytes = 64 * 1024 * 1024
            return httpx.Response(
                200,
                json={
                    "model": "test",
                    "usage": {"input_tokens": 2_000_000, "output_tokens": 0},
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream_response)) as upstream:
            gateway = ExternalAgentGateway(
                client=upstream,
                public_base_url="http://gateway",
                routes={"model": ExternalModelRoute("https://model.test", {}, ("/responses",))},
            )
            surface = FastAPI()
            surface.include_router(gateway.router)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=surface)) as guest:

                class Adapter(ScriptedAdapter):
                    async def stream(self, context):
                        recorders.append(context.check_budget.__self__)
                        async with gateway.grant(
                            context, route="model", models=frozenset({"test"})
                        ) as grant:
                            for expected in (200, 403):
                                response = await guest.post(
                                    grant.model_base_url + "/responses",
                                    headers={"Authorization": f"Bearer {grant.token}"},
                                    json={"model": "test"},
                                )
                                assert response.status_code == expected
                        yield ExternalAgentEvent("result", {"output": "done"}, "result")

                with pytest.raises(ExternalAgentError):
                    await run_external_agent(app, request(), Adapter([]), prompt="review")
        [task] = await app.task_store.list_tasks(TaskQuery())
        assert task.status is TaskStatus.FAILED
        cost = await app.get_session_cost(task.session_id, pricing=pricing())
        assert cost.total_cost == Decimal(2) and cost.model_steps == 1
        assert len(calls) == 1
        usage = [
            event
            for event in await app.session_store.load_events(task.session_id)
            if event.type is EventType.MODEL_COMPLETED
        ]
        assert len(usage) == 1 and usage[0].payload["usage"]["input_tokens"] == 2_000_000
        next_adapter = ScriptedAdapter([ExternalAgentEvent("result", {"output": "next"}, "result")])
        with pytest.raises(ExternalAgentBudgetExceeded):
            await run_external_agent(
                app,
                request(budget_limits=(BudgetLimit.usd("1", scope="app", pricing=pricing()),)),
                next_adapter,
                prompt="next",
            )
        assert next_adapter.consumed == 0
        await app.aclose()

    asyncio.run(scenario())
