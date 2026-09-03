"""Run a deterministic OpenAI hosted Tool Search vertical without an API call."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

from cayu import (
    AgentSpec,
    CayuApp,
    InMemorySessionStore,
    Message,
    OpenAIProvider,
    RunRequest,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from cayu.runtime.tool_discovery import (
    TOOL_DISCOVERY_VIEW_OPERATION_KEY,
    ToolDiscoveryViewState,
)


class RememberKnowledge(Tool):
    spec = ToolSpec(
        name="remember_knowledge",
        description="Save one reviewed fact as durable application knowledge.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"fact": {"type": "string"}},
            "required": ["fact"],
        },
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        del ctx
        self.calls.append(dict(args))
        return ToolResult(content=f"remembered: {args['fact']}")


class FixtureOpenAITransport:
    """Return one hosted selection/call followed by one final response."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        schema = RememberKnowledge.spec.input_schema
        self._batches = [
            self._completed(
                "resp_hosted_call",
                [
                    {
                        "type": "tool_search_call",
                        "execution": "server",
                        "call_id": None,
                        "status": "completed",
                        "arguments": {"paths": ["remember_knowledge"]},
                    },
                    {
                        "type": "tool_search_output",
                        "execution": "server",
                        "call_id": None,
                        "status": "completed",
                        "tools": [
                            {
                                "type": "function",
                                "name": "remember_knowledge",
                                "description": RememberKnowledge.spec.description,
                                "parameters": schema,
                                "strict": False,
                                "defer_loading": True,
                                "output_schema": None,
                            }
                        ],
                    },
                    {
                        "type": "function_call",
                        "call_id": "remember_hosted_fixture_1",
                        "name": "remember_knowledge",
                        "arguments": ('{"fact":"Hosted Tool Search keeps Cayu authority atomic."}'),
                        "status": "completed",
                    },
                ],
            ),
            self._completed(
                "resp_hosted_done",
                [
                    {
                        "type": "message",
                        "id": "msg_hosted_fixture_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Knowledge saved."}],
                    }
                ],
            ),
        ]

    @staticmethod
    def _completed(response_id: str, output: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "model": "gpt-fixture",
                    "status": "completed",
                    "output": output,
                    "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                },
            }
        ]

    async def create_response(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        del url, headers, payload, timeout_s
        raise AssertionError("The fixture does not perform non-streaming requests.")

    async def stream_response_events(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        transport_idle_timeout_s: float,
        protocol_idle_timeout_s: float,
        semantic_progress_timeout_s: float,
        absolute_stream_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        del (
            url,
            headers,
            timeout_s,
            transport_idle_timeout_s,
            protocol_idle_timeout_s,
            semantic_progress_timeout_s,
            absolute_stream_timeout_s,
        )
        self.requests.append(dict(payload))
        if not self._batches:
            raise AssertionError("The fixture received an unexpected provider request.")
        for event in self._batches.pop(0):
            yield event


async def run_scenario() -> dict[str, Any]:
    transport = FixtureOpenAITransport()
    remembered = RememberKnowledge()
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        OpenAIProvider(
            api_key="credential-free-fixture",
            hosted_tool_search_models=("gpt-fixture",),
            transport=transport,
        ),
        default=True,
    )
    session_id = "openai-hosted-tool-search-example"
    app.register_agent(
        AgentSpec(name="assistant", model="gpt-fixture"),
        tools=(remembered,),
        tool_discovery_mode="openai_tool_search_hosted",
    )

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "Find a tool and save the reviewed lesson.")],
            )
        )
    ]
    if events[-1].type != "session.completed":
        raise RuntimeError(
            f"The hosted Tool Search fixture did not complete: {[event.type for event in events]}"
        )
    expected_arguments = {"fact": "Hosted Tool Search keeps Cayu authority atomic."}
    if remembered.calls != [expected_arguments]:
        raise RuntimeError("The hosted-loaded function did not execute exactly once through Cayu.")
    if len(transport.requests) != 2:
        raise RuntimeError("The fixture expected exactly two model requests.")
    expected_tools = [
        {
            "type": "function",
            "name": "remember_knowledge",
            "description": RememberKnowledge.spec.description,
            "parameters": RememberKnowledge.spec.input_schema,
            "strict": False,
            "defer_loading": True,
        },
        {"type": "tool_search", "execution": "server"},
    ]
    if any(request.get("tools") != expected_tools for request in transport.requests):
        raise RuntimeError("The adapter did not preserve the exact hosted Tool Search surface.")
    if any(request.get("parallel_tool_calls") is not False for request in transport.requests):
        raise RuntimeError("Hosted Tool Search did not disable parallel tool calls.")
    replay_types = [item.get("type") for item in transport.requests[1]["input"]]
    if replay_types != [
        None,
        "tool_search_call",
        "tool_search_output",
        "function_call",
        "function_call_output",
    ]:
        raise RuntimeError("The second request did not replay exact hosted selection evidence.")
    view = ToolDiscoveryViewState.model_validate(
        await store.load_session_operation(session_id, TOOL_DISCOVERY_VIEW_OPERATION_KEY)
    )
    if view.revision != 1 or [grant.tool_name for grant in view.grants] != ["remember_knowledge"]:
        raise RuntimeError("The hosted selection did not become one durable branch-local grant.")

    return {
        "schema_version": 1,
        "session_completed": True,
        "provider_requests": len(transport.requests),
        "deferred_candidate_names": ["remember_knowledge"],
        "loaded_tool_names": ["remember_knowledge"],
        "executed_tool_names": ["remember_knowledge"],
        "discovery_view_revision": view.revision,
        "parallel_tool_calls": False,
        "api_key_required": False,
    }


def main() -> None:
    print(json.dumps(asyncio.run(run_scenario()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
