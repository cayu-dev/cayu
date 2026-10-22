"""Service capabilities never expose credentials or outlive the owning run."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal

import httpx
import pytest
from fastapi import FastAPI

from cayu.deadlines import ExecutionDeadline
from cayu.external_agents._model_protocol import validate_model_request
from cayu.external_agents.base import ExternalAgentContext, ExternalAgentError
from cayu.external_agents.server import ExternalAgentGateway, ExternalModelRoute


def test_gateway_enforces_route_model_credentials_and_revocation():
    async def scenario():
        seen = []
        secrets = []

        def upstream(request):
            seen.append(request)
            return httpx.Response(200, content=b"prefix service-secret suffix")

        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
            gateway = ExternalAgentGateway(
                client=client,
                public_base_url="http://gateway",
                routes={
                    "openai": ExternalModelRoute(
                        "https://upstream.test",
                        {"Authorization": "Bearer service-secret"},
                        ("/responses",),
                    )
                },
            )
            app = FastAPI()
            app.include_router(gateway.router)
            context = ExternalAgentContext(
                "run", "prompt", ExecutionDeadline.after(10), protect_secret=secrets.append
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://gateway"
            ) as guest:
                async with gateway.grant(
                    context, route="openai", models=frozenset({"model"})
                ) as grant:
                    headers = {"Authorization": f"Bearer {grant.token}", "x-api-key": "attacker"}
                    response = await guest.post(
                        grant.model_base_url + "/responses",
                        headers=headers,
                        json={"model": "model"},
                    )
                    assert response.status_code == 200
                    assert "service-secret" not in response.text
                    assert seen[0].headers["authorization"] == "Bearer service-secret"
                    assert "x-api-key" not in seen[0].headers
                    for path, model in [
                        ("/responses", "foreign"),
                        ("/secrets", "model"),
                        ("/responses", []),
                    ]:
                        denied = await guest.post(
                            grant.model_base_url + path, headers=headers, json={"model": model}
                        )
                        assert denied.status_code == 403
                    assert len(seen) == 1
                assert secrets == [grant.token]
                denied = await guest.post(
                    grant.model_base_url + "/responses", headers=headers, json={"model": "model"}
                )
                assert denied.status_code == 401

    asyncio.run(scenario())


def test_gateway_scope_exit_joins_idle_upstream_stream():
    async def scenario():
        entered, closed = asyncio.Event(), asyncio.Event()
        receipts = []
        active = True

        async def record(event):
            receipts.append(event)

        async def check_budget():
            if not active:
                raise ExternalAgentError("External session execution authority ended.")

        class Stalled(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                await asyncio.Event().wait()
                yield b"unreachable"

            async def aclose(self):
                await asyncio.sleep(0)
                closed.set()

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Stalled()))
        ) as client:
            gateway = ExternalAgentGateway(
                client=client,
                public_base_url="http://gateway",
                routes={
                    "model": ExternalModelRoute("https://upstream.test", {}, ("/responses",)),
                },
            )
            app = FastAPI()
            app.include_router(gateway.router)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as guest:
                async with gateway.grant(
                    ExternalAgentContext(
                        "run",
                        "prompt",
                        ExecutionDeadline.after(10),
                        check_budget=check_budget,
                        record_model_usage=record,
                    ),
                    route="model",
                    models=frozenset({"model"}),
                ) as grant:
                    request = asyncio.create_task(
                        guest.post(
                            grant.model_base_url + "/responses",
                            headers={"Authorization": f"Bearer {grant.token}"},
                            json={"model": "model"},
                        )
                    )
                    await asyncio.wait_for(entered.wait(), 5)
                    active = False
                assert closed.is_set()
                assert request.done()
                result = await asyncio.gather(request, return_exceptions=True)
                assert isinstance(result[0], asyncio.CancelledError)
                assert len(receipts) == 1 and receipts[0].payload == {}
                assert not gateway.grants

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "path,payload",
    [
        ("responses", {"tools": [{"type": "mcp", "server_url": "https://attacker.invalid"}]}),
        ("responses", {"tools": [{"type": "web_search"}]}),
        (
            "responses",
            {
                "tools": [
                    {"type": "namespace", "name": "unsafe", "tools": [{"type": "code_interpreter"}]}
                ]
            },
        ),
        (
            "responses",
            {
                "input": [
                    {
                        "type": "additional_tools",
                        "role": "developer",
                        "tools": [{"type": "web_search"}],
                    }
                ]
            },
        ),
        ("responses", {"reasoning": {"context": "stored"}}),
        ("responses", {"previous_response_id": "other-response"}),
        ("responses", {"background": True}),
        ("responses", {"input": [{"type": "item_reference", "id": "other-item"}]}),
        (
            "responses",
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": "https://attacker.invalid/private"}
                        ],
                    }
                ]
            },
        ),
        (
            "responses",
            {
                "input": [
                    {"role": "user", "content": [{"type": "input_file", "file_id": "other-file"}]}
                ]
            },
        ),
        ("v1/messages", {"mcp_servers": [{"url": "https://attacker.invalid"}]}),
        ("v1/messages", {"tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}]}),
        ("v1/messages", {"tools": [{"type": "computer_20250124", "name": "computer"}]}),
        (
            "v1/messages",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "url", "url": "https://attacker.invalid"},
                            }
                        ],
                    }
                ]
            },
        ),
        ("v1/messages", {"container": "previous-container"}),
        (
            "v1/messages",
            {
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "data": "AA=="}}
                        ],
                    }
                ]
            },
        ),
    ],
)
def test_gateway_refuses_provider_execution_fetches_and_stored_references(path, payload):
    async def scenario():
        calls = []
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: calls.append(request) or httpx.Response(200, json={})
            )
        ) as upstream:
            gateway = ExternalAgentGateway(
                client=upstream,
                public_base_url="http://gateway",
                routes={
                    "model": ExternalModelRoute("https://model.test", {}, (f"/{path}",)),
                },
            )
            surface = FastAPI()
            surface.include_router(gateway.router)
            async with (
                httpx.AsyncClient(transport=httpx.ASGITransport(app=surface)) as guest,
                gateway.grant(
                    ExternalAgentContext("run", "test", ExecutionDeadline.after(10)),
                    route="model",
                    models=frozenset({"model"}),
                ) as grant,
            ):
                result = await guest.post(
                    f"{grant.model_base_url}/{path}",
                    headers={"Authorization": f"Bearer {grant.token}"},
                    json={"model": "model", **payload},
                )
                assert result.status_code == 400
                assert not calls

    asyncio.run(scenario())


def test_gateway_beta_headers_are_owned_by_service_configuration():
    async def scenario():
        seen = []
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: seen.append(request) or httpx.Response(200, json={})
            )
        ) as upstream:
            for configured in ({}, {"anthropic-beta": "trusted-feature"}):
                gateway = ExternalAgentGateway(
                    client=upstream,
                    public_base_url="http://gateway",
                    routes={
                        "model": ExternalModelRoute(
                            "https://model.test", configured, ("/v1/messages",)
                        ),
                    },
                )
                surface = FastAPI()
                surface.include_router(gateway.router)
                async with (
                    httpx.AsyncClient(transport=httpx.ASGITransport(app=surface)) as guest,
                    gateway.grant(
                        ExternalAgentContext("run", "test", ExecutionDeadline.after(10)),
                        route="model",
                        models=frozenset({"model"}),
                    ) as grant,
                ):
                    result = await guest.post(
                        f"{grant.model_base_url}/v1/messages",
                        headers={
                            "Authorization": f"Bearer {grant.token}",
                            "anthropic-beta": "mcp-connector-2025-04-04",
                        },
                        json={
                            "model": "model",
                            "tools": [{"name": "local", "input_schema": {"type": "object"}}],
                        },
                    )
                    assert result.status_code == 200
                    assert seen[-1].headers.get("anthropic-beta") == configured.get(
                        "anthropic-beta"
                    )

    asyncio.run(scenario())


@pytest.mark.parametrize("protocol", ["messages", "responses"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("budget", ["1", "10", "unknown"])
def test_gateway_owns_usage_and_serializes_direct_calls_across_capabilities(
    protocol, streaming, budget
):
    from cayu import CayuApp, RunRequest
    from cayu.budgets import BudgetLimit, ModelPrice, PriceBook
    from cayu.external_agents import (
        ExternalAgentBudgetExceeded,
        ExternalAgentEvent,
        run_external_agent,
    )
    from cayu.tasks import InMemoryTaskStore, TaskQuery

    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="test",
                    model="model",
                    input_per_million=Decimal(1),
                    output_per_million=Decimal(1),
                ),
            )
        )
        calls, statuses = [], []
        entered, release = asyncio.Event(), asyncio.Event()
        path = "/v1/messages" if protocol == "messages" else "/responses"

        async def upstream_response(request):
            calls.append(request)
            entered.set()
            await release.wait()
            usage = {} if budget == "unknown" else {"input_tokens": 2_000_000, "output_tokens": 0}
            if not streaming:
                return httpx.Response(200, json={"model": "model", "usage": usage})
            events = (
                [
                    {
                        "type": "message_start",
                        "message": {
                            "id": "same-provider-id",
                            "model": "model",
                            "usage": {"input_tokens": 2_000_000, "output_tokens": 1},
                        },
                    },
                    {
                        "type": "message_delta",
                        "usage": {} if budget == "unknown" else {"output_tokens": 0},
                    },
                    {"type": "message_stop"},
                ]
                if protocol == "messages"
                else [
                    {"type": "response.completed", "response": {"model": "model", "usage": usage}}
                ]
            )
            return httpx.Response(
                200,
                content="".join(f"data: {json.dumps(event)}\n\n" for event in events),
                headers={"content-type": "text/event-stream"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream_response)) as upstream:
            gateway = ExternalAgentGateway(
                client=upstream,
                public_base_url="http://gateway",
                routes={"model": ExternalModelRoute("https://model.test", {}, (path,))},
            )
            surface = FastAPI()
            surface.include_router(gateway.router)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=surface)) as guest:

                class Adapter:
                    name, provider_name, model = "direct-calls", "test", "model"

                    async def stream(self, context):
                        async with (
                            gateway.grant(
                                context, route="model", models=frozenset({"model"})
                            ) as first,
                            gateway.grant(
                                context, route="model", models=frozenset({"model"})
                            ) as second,
                        ):
                            pending = [
                                asyncio.create_task(
                                    guest.post(
                                        grant.model_base_url + path,
                                        headers={"Authorization": f"Bearer {grant.token}"},
                                        json={"model": "model", "stream": streaming},
                                    )
                                )
                                for grant in (first, second, first)
                            ]
                            await asyncio.wait_for(entered.wait(), 5)
                            await asyncio.sleep(0)
                            assert len(calls) == 1
                            release.set()
                            statuses.extend(
                                response.status_code for response in await asyncio.gather(*pending)
                            )
                        # Guest counters must neither hide nor double charge calls.
                        yield ExternalAgentEvent(
                            "usage",
                            {"input_tokens": 100_000_000, "output_tokens": 0},
                            "untrusted-sdk",
                        )
                        yield ExternalAgentEvent("result", {"output": "done"}, "result")

                run = run_external_agent(
                    app,
                    RunRequest(
                        agent_name="test",
                        messages=[],
                        execution_deadline=ExecutionDeadline.after(20),
                        budget_limits=(
                            BudgetLimit.usd(
                                "1" if budget == "unknown" else budget, pricing=pricing
                            ),
                        ),
                    ),
                    Adapter(),
                    prompt="test",
                )
                if budget in {"1", "unknown"}:
                    with pytest.raises(ExternalAgentBudgetExceeded):
                        await run
                    assert statuses == [200, 403, 403]
                else:
                    await run
                    assert statuses == [200, 200, 200]
        (task,) = await app.task_store.list_tasks(TaskQuery())
        cost = await app.get_session_cost(task.session_id, pricing=pricing)
        assert cost.total_cost == Decimal({"1": 2, "10": 6, "unknown": 0}[budget])
        assert cost.model_steps == len(calls)
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [429, 529])
@pytest.mark.parametrize("protocol", ["messages", "responses"])
def test_completed_upstream_rejection_keeps_budget_available_for_retry(status, protocol):
    from cayu import CayuApp, RunRequest
    from cayu.budgets import BudgetLimit, ModelPrice, PriceBook
    from cayu.external_agents import ExternalAgentEvent, run_external_agent
    from cayu.tasks import InMemoryTaskStore

    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        path = "/v1/messages" if protocol == "messages" else "/responses"
        calls, statuses = [], []
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="test",
                    model="model",
                    input_per_million=Decimal(1),
                    output_per_million=Decimal(1),
                ),
            )
        )

        def reply(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(status, json={"error": {"message": "retry later"}})
            return httpx.Response(
                200,
                json={
                    "model": "model",
                    "usage": {
                        "input_tokens": 1_000_000,
                        "output_tokens": 0,
                    },
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as upstream:
            gateway = ExternalAgentGateway(
                client=upstream,
                public_base_url="http://gateway",
                routes={"model": ExternalModelRoute("https://model.test", {}, (path,))},
            )
            surface = FastAPI()
            surface.include_router(gateway.router)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=surface)) as guest:

                class Adapter:
                    name, provider_name, model = "retry", "test", "model"

                    async def stream(self, context):
                        async with gateway.grant(
                            context, route="model", models=frozenset({"model"})
                        ) as grant:
                            for _ in range(2):
                                response = await guest.post(
                                    grant.model_base_url + path,
                                    headers={"Authorization": f"Bearer {grant.token}"},
                                    json={"model": "model"},
                                )
                                statuses.append(response.status_code)
                        yield ExternalAgentEvent("result", {"output": "done"}, "result")

                result = await run_external_agent(
                    app,
                    RunRequest(
                        agent_name="test",
                        messages=[],
                        execution_deadline=ExecutionDeadline.after(20),
                        budget_limits=(BudgetLimit.usd("100", pricing=pricing),),
                    ),
                    Adapter(),
                    prompt="test",
                )
        assert statuses == [status, 200] and len(calls) == 2
        cost = await app.get_session_cost(result.session_id, pricing=pricing)
        assert cost.total_cost == Decimal(1)
        assert cost.model_steps == 2
        await app.aclose()

    asyncio.run(scenario())


def test_mcp_transport_runs_native_tool_and_rejects_revoked_scope():
    from cayu import CayuApp, RunRequest
    from cayu.external_agents import CayuToolBridge, ExternalAgentEvent, run_external_agent
    from cayu.external_agents.tools import external_tool_scope
    from cayu.tasks import InMemoryTaskStore
    from cayu.tools import Tool, ToolEffect, ToolResult, ToolSpec
    from cayu.tools.policy import AllowAllToolPolicy

    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        calls = []

        class Echo(Tool):
            spec = ToolSpec(name="echo", effect=ToolEffect.NONE)

            async def run(self, ctx, args):
                calls.append(ctx.session_id)
                return ToolResult(content="native result")

        bridge = CayuToolBridge(app, name="echo", tools=[Echo()], policy=AllowAllToolPolicy())
        async with httpx.AsyncClient() as upstream:
            gateway = ExternalAgentGateway(
                client=upstream,
                public_base_url="http://gateway",
                routes={"model": ExternalModelRoute("https://unused.test", {}, ("/responses",))},
            )
            surface = FastAPI()
            surface.include_router(gateway.router)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=surface)) as guest:

                class Adapter:
                    name, provider_name, model = "mcp", "test", "test"

                    async def stream(self, context):
                        async with external_tool_scope(
                            bridge,
                            session_id=context.session_id,
                            execution_deadline=context.execution_deadline,
                        ) as endpoint:
                            async with gateway.grant(
                                context, route="model", models=frozenset({"test"}), tools=endpoint
                            ) as grant:
                                headers = {"Authorization": f"Bearer {grant.token}"}
                                for method, params in (
                                    (
                                        "initialize",
                                        {
                                            "protocolVersion": "2025-03-26",
                                            "capabilities": {},
                                            "clientInfo": {"name": "test", "version": "1"},
                                        },
                                    ),
                                    ("tools/list", {}),
                                    ("tools/call", {"name": "echo", "arguments": {}}),
                                ):
                                    response = await guest.post(
                                        grant.mcp_url,
                                        headers=headers,
                                        json={
                                            "jsonrpc": "2.0",
                                            "id": 1,
                                            "method": method,
                                            "params": params,
                                        },
                                    )
                                    assert response.status_code == 200
                                    assert "result" in response.json(), response.text
                                assert (
                                    response.json()["result"]["content"][0]["text"]
                                    == "native result"
                                )
                                refused = await guest.post(
                                    grant.mcp_url,
                                    headers=headers,
                                    json={
                                        "jsonrpc": "2.0",
                                        "id": 2,
                                        "method": "tools/call",
                                        "params": {
                                            "name": "echo",
                                            "arguments": {"capability": grant.token},
                                        },
                                    },
                                )
                                assert refused.status_code == 400
                                assert grant.token not in refused.text
                                assert len(calls) == 1
                                assert (
                                    await app.session_store.load(calls[0])
                                ).parent_session_id == context.session_id
                            response = await guest.post(grant.mcp_url, headers=headers, json={})
                            assert response.status_code == 401
                        yield ExternalAgentEvent("result", {"output": "done"}, "result")

                await run_external_agent(
                    app,
                    RunRequest(
                        agent_name="mcp",
                        messages=[],
                        execution_deadline=ExecutionDeadline.after(20),
                    ),
                    Adapter(),
                    prompt="test",
                )
        await app.aclose()

    asyncio.run(scenario())


def test_gateway_accepts_the_pinned_cli_request_shapes():
    # Claude Code adds system-role messages with their own effort setting, and
    # Codex declares its client tools as an input item.
    validate_model_request(
        {
            "model": "model",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "env", "cache_control": {"type": "ephemeral"}}
                    ],
                    "output_config": {"effort": "medium"},
                },
            ],
        },
        "messages",
    )
    validate_model_request(
        {
            "model": "model",
            "reasoning": {"effort": "medium", "context": "all_turns"},
            "input": [
                {
                    "type": "additional_tools",
                    "id": "at_1",
                    "role": "developer",
                    "tools": [
                        {
                            "type": "namespace",
                            "name": "functions",
                            "description": "",
                            "tools": [
                                {
                                    "type": "custom",
                                    "name": "exec",
                                    "format": {
                                        "type": "grammar",
                                        "syntax": "lark",
                                        "definition": "start: /.+/",
                                    },
                                },
                                {"type": "function", "name": "wait", "strict": False},
                            ],
                        }
                    ],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
            ],
        },
        "responses",
    )
