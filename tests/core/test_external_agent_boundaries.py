from __future__ import annotations

import asyncio
from contextlib import suppress
from decimal import Decimal
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from tests.core.test_external_agents import ScriptedAdapter, request

from cayu import CayuApp
from cayu.budgets import BudgetLimit, ModelPrice, PriceBook
from cayu.deadlines import ExecutionDeadline
from cayu.events import EventType
from cayu.external_agents import (
    ExternalAgentError,
    ExternalAgentEvent,
    recover_external_agent_run,
    run_external_agent,
)
from cayu.external_agents.environment import docker_agent_launch
from cayu.external_agents.server import ExternalAgentGateway, ExternalModelRoute
from cayu.external_agents.tools import CayuToolBridge, ExternalPermissionTool, external_tool_scope
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import run_request_with_task_invocation
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskQuery, TaskStatus
from cayu.tools.policy import AllowAllToolPolicy


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_setup_crash_before_task_attachment_recovers_exact_invocation(backend, tmp_path):
    from cayu.storage import SQLiteSessionStore, SQLiteTaskStore

    async def scenario():
        app = CayuApp(
            session_store=SQLiteSessionStore(tmp_path / "sessions.db")
            if backend == "sqlite"
            else None,
            task_store=SQLiteTaskStore(tmp_path / "tasks.db")
            if backend == "sqlite"
            else InMemoryTaskStore(),
            enable_logging=False,
        )
        session_id = str(uuid4())
        task = await app.create_task(TaskCreate(type="external_agent", session_id=session_id))
        bound = run_request_with_task_invocation(
            request(task_id=task.id, session_id=session_id),
            await app.task_store.load_invocation_snapshot(task.id),
        )
        session = await app.session_store.create(
            bound,
            identity=SessionIdentity(
                provider_name="test",
                model="test",
                runtime_name="external_agent",
            ),
        )
        assert task.session_instance_id is None
        receipts = []

        async def stopped(current):
            receipts.append(current.instance_id)

        await recover_external_agent_run(
            app, session.id, quiesce=stopped, evidence_id="setup-crash"
        )
        assert (await app.task_store.load_task(task.id)).status is TaskStatus.CANCELLED
        assert (await app.session_store.load(session.id)).status is SessionStatus.INTERRUPTED
        assert receipts == [session.instance_id]
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("revoke", [False, True])
def test_external_worker_requires_session_access_and_rechecks_revocation(revoke):
    from tests.core.test_resource_execution_access import Policy, scope

    from cayu.resource_access import ResourceAccessGrant
    from cayu.sessions._access_rules import SessionAccessScope
    from cayu.tasks.worker import run_task_worker

    async def scenario():
        policy = Policy()
        policy.current = ResourceAccessGrant(
            tasks=scope("acme"), sessions=scope("acme") if revoke else SessionAccessScope()
        )
        app = CayuApp(
            task_store=InMemoryTaskStore(), resource_access_policy=policy, enable_logging=False
        )
        task = await (await app.access("alice")).tasks.create(
            TaskCreate(type="external_agent"),
            labels={"organization": "acme"},
        )
        entered, closed = asyncio.Event(), asyncio.Event()

        class Adapter(ScriptedAdapter):
            async def stream(self, context):
                entered.set()
                try:
                    await asyncio.Event().wait()
                    yield ExternalAgentEvent("result", {"output": "unreachable"}, "final")
                finally:
                    closed.set()

        async def handler(app, task, worker_id):
            await run_external_agent(
                app,
                request(
                    task_id=task.id,
                    task_worker_id=worker_id,
                    task_lease_expires_at=task.lease_expires_at,
                ),
                Adapter([]),
                prompt="test",
            )

        running = asyncio.create_task(
            run_task_worker(app, app.task_store, handler, worker_id="worker", max_tasks=1)
        )
        if revoke:
            await asyncio.wait_for(entered.wait(), 5)
            policy.current = ResourceAccessGrant(tasks=scope("acme"))
        await asyncio.wait_for(running, 5)
        assert entered.is_set() is revoke
        assert closed.is_set() is revoke
        latest = await app.task_store.load_task(task.id)
        assert latest.status is not TaskStatus.COMPLETED
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("permanent", [False, True])
@pytest.mark.parametrize("acknowledgement_lost", [False, True])
def test_failed_receipt_write_fences_all_capabilities_and_prevents_success(
    permanent, acknowledgement_lost, monkeypatch
):
    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        emit = app.emit_event
        attempts = 0

        async def failing(event):
            nonlocal attempts
            if event.type is EventType.MODEL_COMPLETED:
                attempts += 1
                if permanent or attempts == 1:
                    if acknowledgement_lost:
                        await emit(event)
                    raise OSError("usage publication unavailable")
            return await emit(event)

        monkeypatch.setattr(app, "emit_event", failing)
        calls, statuses = [], []
        price = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="scripted-external",
                    model="test",
                    input_per_million=Decimal(1),
                    output_per_million=Decimal(1),
                ),
            )
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: (
                    calls.append(req)
                    or httpx.Response(
                        200,
                        json={
                            "model": "test",
                            "usage": {"input_tokens": 2_000_000, "output_tokens": 0},
                        },
                    )
                )
            )
        ) as upstream:
            gateway = ExternalAgentGateway(
                client=upstream,
                public_base_url="http://gateway",
                routes={"model": ExternalModelRoute("https://model.test", {}, ("/responses",))},
            )
            surface = FastAPI()
            surface.include_router(gateway.router)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=surface, raise_app_exceptions=False)
            ) as guest:

                class Adapter(ScriptedAdapter):
                    async def stream(self, context):
                        async with (
                            gateway.grant(
                                context, route="model", models=frozenset({"test"})
                            ) as first,
                            gateway.grant(
                                context, route="model", models=frozenset({"test"})
                            ) as second,
                        ):
                            for grant in (first, second, first):
                                response = await guest.post(
                                    grant.model_base_url + "/responses",
                                    headers={"Authorization": f"Bearer {grant.token}"},
                                    json={"model": "test"},
                                )
                                statuses.append(response.status_code)
                        yield ExternalAgentEvent("result", {"output": "done"}, "final")

                with pytest.raises(ExternalAgentError):
                    await run_external_agent(
                        app,
                        request(budget_limits=(BudgetLimit.usd("1", pricing=price),)),
                        Adapter([]),
                        prompt="test",
                    )
        assert statuses == [500, 403, 403]
        assert len(calls) == 1
        (task,) = await app.task_store.list_tasks(TaskQuery())
        assert task.status is not TaskStatus.COMPLETED
        assert (await app.session_store.load(task.session_id)).status is not SessionStatus.COMPLETED
        events = await app.session_store.load_events(task.session_id)
        assert sum(event.type is EventType.MODEL_COMPLETED for event in events) == (
            0 if permanent and not acknowledgement_lost else 1
        )
        await app.aclose()

    asyncio.run(scenario())


def test_mcp_side_effects_stop_after_model_exhausts_parent_budget():
    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        executed = []

        class Effect(ExternalPermissionTool):
            async def run(self, ctx, args):
                executed.append(args)
                return await super().run(ctx, args)

        bridge = CayuToolBridge(
            app, name="bridge", tools=[Effect("effect")], policy=AllowAllToolPolicy()
        )
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="scripted-external",
                    model="test",
                    input_per_million=Decimal(1),
                    output_per_million=Decimal(1),
                ),
            )
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    200, json={"usage": {"input_tokens": 2_000_000, "output_tokens": 0}}
                )
            )
        ) as upstream:
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
                        async with (
                            external_tool_scope(
                                bridge,
                                session_id=context.session_id,
                                execution_deadline=context.execution_deadline,
                            ) as endpoint,
                            gateway.grant(
                                context, route="model", models=frozenset({"test"}), tools=endpoint
                            ) as grant,
                        ):
                            headers = {"Authorization": f"Bearer {grant.token}"}
                            response = await guest.post(
                                grant.model_base_url + "/responses",
                                headers=headers,
                                json={"model": "test"},
                            )
                            assert response.status_code == 200
                            for _ in range(3):
                                response = await guest.post(
                                    grant.mcp_url,
                                    headers=headers,
                                    json={
                                        "jsonrpc": "2.0",
                                        "id": 1,
                                        "method": "tools/call",
                                        "params": {"name": "effect", "arguments": {}},
                                    },
                                )
                                assert response.status_code == 403
                        yield ExternalAgentEvent("result", {"output": "done"}, "final")

                with pytest.raises(ExternalAgentError):
                    await run_external_agent(
                        app,
                        request(budget_limits=(BudgetLimit.usd("1", pricing=pricing),)),
                        Adapter([]),
                        prompt="test",
                    )
        assert not executed
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("allow", [False, True])
def test_mcp_discovery_applies_manifest_policy_before_exposure(allow):
    from tests.core.test_mcp import _fake_toolset

    from cayu.runtime import McpManifestPolicy, McpManifestPolicyAction
    from cayu.runtime.mcp_manifest_policy import McpManifestPolicyError

    async def scenario():
        toolset = _fake_toolset()
        app = CayuApp(
            task_store=InMemoryTaskStore(),
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(
                on_first_seen=McpManifestPolicyAction.ALLOW
                if allow
                else McpManifestPolicyAction.BLOCK
            ),
        )
        bridge = CayuToolBridge(app, name="mcp", tools=toolset.tools, policy=AllowAllToolPolicy())

        class Adapter(ScriptedAdapter):
            async def stream(self, context):
                async with external_tool_scope(
                    bridge,
                    session_id=context.session_id,
                    execution_deadline=context.execution_deadline,
                ) as endpoint:
                    if allow:
                        listed = await endpoint.dispatch("tools/list", {})
                        assert listed["tools"][0]["description"] == toolset.tools[0].description
                    else:
                        with pytest.raises(McpManifestPolicyError):
                            await endpoint.dispatch("tools/list", {})
                yield ExternalAgentEvent("result", {"output": "done"}, "final")

        result = await run_external_agent(app, request(), Adapter([]), prompt="test")
        events = await app.session_store.load_events(result.session_id)
        assert any(str(event.type).startswith("mcp.manifest") for event in events)
        assert toolset.session.calls == []
        await app.aclose()

    asyncio.run(scenario())


def test_cleanup_failure_retains_live_task_until_supervisor_quiescence(tmp_path, monkeypatch):
    import cayu.external_agents.environment as environment
    from cayu.external_agents._cleanup import close_external_guest

    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        attempted = asyncio.Event()
        removed = False

        class Runner:
            container_id = "immutable-container"
            docker_path = "/mock/docker"

            async def close(self):
                attempted.set()
                if not removed:
                    raise RuntimeError("docker rm unavailable")

        async def create(*args, **kwargs):
            return Runner()

        async def internal(*args):
            pass

        monkeypatch.setattr(environment.DockerRunner, "create", create)
        monkeypatch.setattr(environment, "_require_internal_network", internal)
        monkeypatch.setattr(environment.shutil, "which", lambda _: "/mock/docker")

        class Adapter(ScriptedAdapter):
            async def stream(self, context):
                async with docker_agent_launch(
                    image="test", cli="/cli", workspace=tmp_path, network="isolated", environment={}
                ) as launch:
                    try:
                        yield ExternalAgentEvent("result", {"output": "done"}, "final")
                    finally:
                        await close_external_guest(launch.close)

        running = asyncio.create_task(
            run_external_agent(app, request(), Adapter([]), prompt="test")
        )
        await asyncio.wait_for(attempted.wait(), 5)
        await asyncio.sleep(0.1)
        (task,) = await app.task_store.list_tasks(TaskQuery())
        assert task.status is TaskStatus.RUNNING
        assert not running.done()
        session = await app.session_store.load(task.session_id)
        assert session.status is SessionStatus.RUNNING
        assert (
            session.id,
            session.instance_id,
            session.run_epoch,
        ) in app._session_control.execution_presence.groups
        receipts = []

        async def quiesce(current):
            nonlocal removed
            receipts.append(current.instance_id)
            removed = True

        await asyncio.wait_for(
            recover_external_agent_run(
                app, session.id, quiesce=quiesce, evidence_id="container-removed"
            ),
            5,
        )
        with suppress(asyncio.CancelledError, ExternalAgentError):
            await running
        assert receipts == [session.instance_id]
        assert (await app.task_store.load_task(task.id)).status is TaskStatus.CANCELLED
        assert (await app.session_store.load(session.id)).status is SessionStatus.INTERRUPTED
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.process
def test_bridge_keeps_real_process_isolation_boundary():
    import sys

    if sys.platform != "linux":
        pytest.skip("process containment requires Linux")
    from tests.core.test_process_isolated_tools import _tool

    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        bridge = CayuToolBridge(app, name="isolated", tools=[_tool()], policy=AllowAllToolPolicy())

        class Adapter(ScriptedAdapter):
            async def stream(self, context):
                result = await bridge.call(
                    "isolated_fixture",
                    {"text": "hello"},
                    parent_session_id=context.session_id,
                    execution_deadline=context.execution_deadline,
                    call_id="isolated-1",
                )
                assert not result.is_error, result
                assert result.content == "hello"
                assert result.structured["environment_marker"] == "declared"
                yield ExternalAgentEvent("result", {"output": "done"}, "final")

        await run_external_agent(app, request(), Adapter([]), prompt="test")
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["function_call", "custom_tool_call"])
def test_gateway_accepts_namespaced_client_tool_followup(kind):
    async def scenario():
        calls = []
        tool_call = {
            "type": kind,
            "id": "fc-1",
            "call_id": "call-1",
            "namespace": "functions",
            "name": "read",
            ("arguments" if kind == "function_call" else "input"): "{}",
        }

        def reply(req):
            calls.append(req)
            return httpx.Response(
                200, json={"output": [tool_call], "usage": {"input_tokens": 1, "output_tokens": 1}}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as upstream:
            from cayu.external_agents.base import ExternalAgentContext

            gateway = ExternalAgentGateway(
                client=upstream,
                public_base_url="http://gateway",
                routes={"model": ExternalModelRoute("https://model.test", {}, ("/responses",))},
            )
            surface = FastAPI()
            surface.include_router(gateway.router)
            context = ExternalAgentContext("run", "prompt", ExecutionDeadline.after(10))
            async with (
                httpx.AsyncClient(transport=httpx.ASGITransport(app=surface)) as guest,
                gateway.grant(context, route="model", models=frozenset({"test"})) as grant,
            ):
                headers = {"Authorization": f"Bearer {grant.token}"}
                first = await guest.post(
                    grant.model_base_url + "/responses",
                    headers=headers,
                    json={"model": "test", "input": "read"},
                )
                followup = [
                    *first.json()["output"],
                    {
                        "type": "function_call_output"
                        if kind == "function_call"
                        else "custom_tool_call_output",
                        "call_id": "call-1",
                        "output": "done",
                    },
                ]
                second = await guest.post(
                    grant.model_base_url + "/responses",
                    headers=headers,
                    json={"model": "test", "input": followup},
                )
                assert second.status_code == 200
                assert len(calls) == 2

    asyncio.run(scenario())
