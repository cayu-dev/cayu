"""Run a deterministic OpenAI client Tool Search vertical without an API call."""

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
    """Return one search, one loaded-function call, and one final response."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._batches = [
            self._completed(
                "resp_search",
                [
                    {
                        "type": "tool_search_call",
                        "id": "ts_fixture_1",
                        "call_id": "search_fixture_1",
                        "execution": "client",
                        "arguments": {"query": "remember reviewed knowledge", "limit": 1},
                        "status": "completed",
                    }
                ],
            ),
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_fixture_1",
                        "call_id": "remember_fixture_1",
                        "name": "remember_knowledge",
                        "arguments": "",
                        "status": "in_progress",
                    },
                },
                {
                    "type": "response.function_call_arguments.done",
                    "output_index": 0,
                    "item_id": "fc_fixture_1",
                    "name": "remember_knowledge",
                    "arguments": '{"fact":"Client Tool Search keeps Cayu authority."}',
                },
                *self._completed(
                    "resp_call",
                    [
                        {
                            "type": "function_call",
                            "id": "fc_fixture_1",
                            "call_id": "remember_fixture_1",
                            "name": "remember_knowledge",
                            "arguments": ('{"fact":"Client Tool Search keeps Cayu authority."}'),
                            "status": "completed",
                        }
                    ],
                ),
            ],
            self._completed(
                "resp_done",
                [
                    {
                        "type": "message",
                        "id": "msg_fixture_1",
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
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    app.register_provider(
        OpenAIProvider(
            api_key="credential-free-fixture",
            client_tool_search_models=("gpt-fixture",),
            transport=transport,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="gpt-fixture"),
        tools=(remembered,),
        tool_discovery_mode="openai_tool_search_client",
    )

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="openai-client-tool-search-example",
                messages=[Message.text("user", "Find a tool and save the reviewed lesson.")],
            )
        )
    ]
    if events[-1].type != "session.completed":
        raise RuntimeError(
            f"The Tool Search fixture did not complete: {[event.type for event in events]}"
        )
    if remembered.calls != [{"fact": "Client Tool Search keeps Cayu authority."}]:
        raise RuntimeError("The loaded function did not execute through Cayu.")
    if len(transport.requests) != 3:
        raise RuntimeError("The fixture expected exactly three model requests.")
    if any(
        tool.get("name") == "remember_knowledge"
        for request in transport.requests
        for tool in request.get("tools", [])
    ):
        raise RuntimeError("The application function leaked into the top-level tool array.")
    loaded_output = next(
        item for item in transport.requests[1]["input"] if item.get("type") == "tool_search_output"
    )
    if [tool["name"] for tool in loaded_output["tools"]] != ["remember_knowledge"]:
        raise RuntimeError("Tool Search did not load the canonical registered function.")

    return {
        "schema_version": 1,
        "session_completed": True,
        "provider_requests": len(transport.requests),
        "top_level_tool_counts": [len(request.get("tools", [])) for request in transport.requests],
        "loaded_tool_names": ["remember_knowledge"],
        "executed_tool_names": ["remember_knowledge"],
        "api_key_required": False,
    }


def main() -> None:
    print(json.dumps(asyncio.run(run_scenario()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
