"""External protocol continuations retain native results and exact observation identity."""

from __future__ import annotations

import asyncio
from copy import deepcopy

import httpx
import pytest
from fastapi import FastAPI
from tests.core.test_external_agents import ScriptedAdapter, request

from cayu import CayuApp
from cayu.deadlines import ExecutionDeadline
from cayu.external_agents import ExternalAgentError, ExternalAgentEvent, run_external_agent
from cayu.external_agents.base import ExternalAgentContext
from cayu.external_agents.server import ExternalAgentGateway, ExternalModelRoute
from cayu.external_agents.tools import CayuToolBridge
from cayu.sessions import SessionStatus
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks import InMemoryTaskStore, TaskStatus
from cayu.tools import Tool, ToolResult, ToolSpec
from cayu.tools.policy import AllowAllToolPolicy
from cayu.vaults.redaction import SecretRedactor


def _inline_document(kind):
    if kind == "text":
        source = {"type": "text", "media_type": "text/plain", "data": "The grass is green."}
        citation = {"type": "char_location", "start_char_index": 0, "end_char_index": 19}
    else:
        source = {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0xLjc="}
        citation = {"type": "page_location", "start_page_number": 1, "end_page_number": 2}
    return (
        {"type": "document", "source": source, "citations": {"enabled": True}},
        {
            **citation,
            "document_index": 0,
            "document_title": None,
            "file_id": None,
            "cited_text": "The grass is green.",
        },
    )


@pytest.mark.parametrize("kind", ["text", "pdf"])
def test_gateway_roundtrips_inline_document_citations(kind):
    async def scenario():
        document, citation = _inline_document(kind)
        text = {"type": "text", "text": "Green.", "citations": [citation]}
        upstream_calls = []

        def upstream(request):
            upstream_calls.append(request)
            return httpx.Response(
                200,
                json={"content": [text], "usage": {"input_tokens": 10, "output_tokens": 1}},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
            gateway = ExternalAgentGateway(
                client=client,
                public_base_url="http://gateway",
                routes={"model": ExternalModelRoute("https://upstream.test", {}, ("/messages",))},
            )
            surface = FastAPI()
            surface.include_router(gateway.router)
            async with (
                httpx.AsyncClient(transport=httpx.ASGITransport(app=surface)) as guest,
                gateway.grant(
                    ExternalAgentContext("run", "test", ExecutionDeadline.after(20)),
                    route="model",
                    models=frozenset({"model"}),
                ) as grant,
            ):
                headers = {"Authorization": f"Bearer {grant.token}"}
                body = {"model": "model", "messages": [{"role": "user", "content": [document]}]}
                first = await guest.post(
                    grant.model_base_url + "/messages", headers=headers, json=body
                )
                assert first.status_code == 200
                body["messages"].extend(
                    [
                        {"role": "assistant", "content": first.json()["content"]},
                        {"role": "user", "content": "Why?"},
                    ]
                )
                followup = await guest.post(
                    grant.model_base_url + "/messages", headers=headers, json=body
                )
                assert followup.status_code == 200, followup.text
                assert len(upstream_calls) == 2
                for invalid in [
                    {**citation, "document_index": 1},
                    {**citation, "document_index": True},
                    {**citation, "file_id": "stored-file"},
                    {**citation, "url": "https://private.invalid"},
                    {"type": "web_search_result_location", "url": "https://private.invalid"},
                    {**citation, "start_char_index" if kind == "text" else "start_page_number": -1},
                ]:
                    malformed = deepcopy(body)
                    malformed["messages"][1]["content"][0]["citations"] = [invalid]
                    denied = await guest.post(
                        grant.model_base_url + "/messages", headers=headers, json=malformed
                    )
                    assert denied.status_code == 400
                assert len(upstream_calls) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("placement", ["request", "content", "tool"])
def test_gateway_rejects_one_hour_cache_writes_before_dispatch(placement):
    async def scenario():
        calls = []

        def upstream(request):
            calls.append(request)
            return httpx.Response(200, json={"usage": {"input_tokens": 1, "output_tokens": 1}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
            gateway = ExternalAgentGateway(
                client=client,
                public_base_url="http://gateway",
                routes={"model": ExternalModelRoute("https://upstream.test", {}, ("/messages",))},
            )
            surface = FastAPI()
            surface.include_router(gateway.router)
            async with (
                httpx.AsyncClient(transport=httpx.ASGITransport(app=surface)) as guest,
                gateway.grant(
                    ExternalAgentContext("run", "test", ExecutionDeadline.after(20)),
                    route="model",
                    models=frozenset({"model"}),
                ) as grant,
            ):
                cache = {"type": "ephemeral", "ttl": "1h"}
                body = {"model": "model"}
                if placement == "request":
                    body["cache_control"] = cache
                elif placement == "content":
                    body["messages"] = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "x", "cache_control": cache}],
                        }
                    ]
                else:
                    body["tools"] = [
                        {
                            "name": "local",
                            "input_schema": {"type": "object"},
                            "cache_control": cache,
                        }
                    ]
                headers = {"Authorization": f"Bearer {grant.token}"}
                denied = await guest.post(
                    grant.model_base_url + "/messages", headers=headers, json=body
                )
                assert denied.status_code == 400 and not calls
                cache["ttl"] = "5m"
                accepted = await guest.post(
                    grant.model_base_url + "/messages", headers=headers, json=body
                )
                assert accepted.status_code == 200 and len(calls) == 1

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("replay", ["identical", "conflicting", "distinct"])
def test_result_replay_preserves_success_and_rejects_conflicts(backend, replay, tmp_path):
    async def scenario():
        app = CayuApp(
            session_store=None
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.db"),
            task_store=InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.db"),
            enable_logging=False,
        )
        first = ExternalAgentEvent("result", {"output": "done"}, "result")
        second = ExternalAgentEvent(
            "result",
            {"output": "changed" if replay == "conflicting" else "done"},
            "other" if replay == "distinct" else "result",
        )
        try:

            class Adapter(ScriptedAdapter):
                async def stream(self, context):
                    self.session_id = context.session_id
                    async for observation in super().stream(context):
                        yield observation

            adapter = Adapter([first, second])
            if replay == "identical":
                result = await run_external_agent(app, request(), adapter, prompt="test")
                assert result.output == "done"
                assert (
                    await app.task_store.load_task(result.task_id)
                ).status is TaskStatus.COMPLETED
            else:
                with pytest.raises(ExternalAgentError):
                    await run_external_agent(app, request(), adapter, prompt="test")
            session = await app.session_store.load(adapter.session_id)
            assert session.status is (
                SessionStatus.COMPLETED if replay == "identical" else SessionStatus.FAILED
            )
            events = await app.session_store.load_events(session.id)
            assert sum(str(event.type) == "custom.external_agent.result" for event in events) == (
                2 if replay == "distinct" else 1
            )
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_native_bridge_returns_published_error_diagnostics():
    async def scenario():
        app = CayuApp(
            task_store=InMemoryTaskStore(),
            enable_logging=False,
            secret_redactor=SecretRedactor("private-diagnostic"),
        )

        class ErrorTool(Tool):
            spec = ToolSpec(name="diagnose")

            async def run(self, context, arguments):
                return ToolResult(
                    content="Retry with max_bytes >= 4. private-diagnostic",
                    structured={"minimum_max_bytes": 4, "secret": "private-diagnostic"},
                    is_error=True,
                )

        bridge = CayuToolBridge(
            app, name="diagnostics", tools=[ErrorTool()], policy=AllowAllToolPolicy()
        )

        class Adapter(ScriptedAdapter):
            async def stream(self, context):
                result = await bridge.call(
                    "diagnose",
                    {},
                    parent_session_id=context.session_id,
                    execution_deadline=context.execution_deadline,
                    call_id="diagnostic",
                )
                assert result.is_error
                assert "max_bytes >= 4" in result.content
                assert result.structured["minimum_max_bytes"] == 4
                assert "private-diagnostic" not in result.model_dump_json()
                yield ExternalAgentEvent("result", {"output": "diagnosed"}, "result")

        try:
            result = await run_external_agent(app, request(), Adapter([]), prompt="test")
            assert result.output == "diagnosed"
        finally:
            await app.aclose()

    asyncio.run(scenario())
