"""The reserved final request cannot force a tool withdrawn by finishing."""

from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest
from tests.core.test_run_limit_finish import ReadTool, answer, spec

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.budgets.run_limits import RunLimits
from cayu.context.structured_output import STRUCTURED_OUTPUT_TOOL_NAME
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.providers.anthropic import build_anthropic_payload
from cayu.providers.chat_completions import build_chat_completions_payload
from cayu.providers.openai import build_openai_payload

_CHOICES = [
    ("openai", {"tool_choice": {"type": "function", "name": "read"}}),
    (
        "openai",
        {
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "function", "name": "read"}],
            }
        },
    ),
    ("openai", {"tool_choice": "required"}),
    ("compatible", {"tool_choice": {"type": "function", "function": {"name": "read"}}}),
    ("compatible", {"function_call": {"name": "read"}}),
]


@pytest.mark.parametrize(
    "strategy,protocol,controls",
    [
        (strategy, protocol, controls)
        for strategy in ("native", "tool")
        for protocol, controls in _CHOICES
    ]
    + [
        ("tool", "anthropic", {"tool_choice": {"type": "tool", "name": "read"}}),
        ("tool", "anthropic", {"tool_choice": {"type": "any"}}),
    ],
)
def test_finish_request_reconciles_provider_tool_choices(strategy, protocol, controls):
    original = {protocol: {**deepcopy(controls), "temperature": 0.2}}
    agent = AgentSpec(name="assistant", model="fake", provider_options=original)
    tool = ReadTool()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(id="read-1", name="read", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            answer(strategy),
        ],
        supports_native_structured_output=True,
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(agent, tools=[tool])

    async def scenario():
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        messages=[Message.text("user", "Find the answer")],
                        max_steps=2,
                        limits=RunLimits(on_limit="finish"),
                        structured_output=spec(strategy),
                    )
                )
            ]
            assert any(event.type is EventType.STRUCTURED_OUTPUT_VALIDATED for event in events)
            first, final = provider.requests
            assert first.options[protocol] == original[protocol]
            assert agent.provider_options == original
            assert final.options[protocol] == {"temperature": 0.2}
            expected = [] if strategy == "native" else [STRUCTURED_OUTPUT_TOOL_NAME]
            assert [tool["name"] for tool in final.tools] == expected
            assert final.options["callable_tool_names"] == expected
            if protocol == "openai":
                payload = build_openai_payload(final)
                assert payload["tool_choice"] == (
                    "none"
                    if strategy == "native"
                    else {
                        "type": "allowed_tools",
                        "mode": "auto",
                        "tools": [{"type": "function", "name": STRUCTURED_OUTPUT_TOOL_NAME}],
                    }
                )
            elif protocol == "compatible":
                payload = build_chat_completions_payload(final, options_key=protocol)
                assert "tool_choice" not in payload and "function_call" not in payload
            else:
                payload = build_anthropic_payload(final)
                assert "tool_choice" not in payload
            assert payload["temperature"] == 0.2
            assert tool.calls == 1
        finally:
            await app.aclose()

    asyncio.run(scenario())
