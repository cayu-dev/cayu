"""Scripted tests can assert what a real model sees, not what the tool returned."""

from __future__ import annotations

import asyncio

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    ExecutionProfileBehaviorIdentity,
    InMemorySessionStore,
    Message,
    ModelRequest,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    StaticToolExposurePolicy,
    StaticToolPolicy,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
    model_facing_text,
    model_facing_tool_result,
    run_to_completion,
)
from cayu.messages import PeerContentPart, ToolCallPart, ToolResultPart


class _AuditCsv(Tool):
    spec = ToolSpec(
        name="audit_csv",
        description="Audit one CSV file.",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        effect=ToolEffect.NONE,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests.audit_csv", behavior_version="1", implementation_version="1"
        ),
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        # The mistake from #1972: findings only in structured, a bare summary in content.
        return ToolResult(content="CSV audit complete", structured={"blank_rows": 3})


def test_model_facing_tool_result_is_the_content_only() -> None:
    result = ToolResult(content="2 rows missing ids", structured={"rows": [4, 9]})

    assert model_facing_tool_result(result) == "2 rows missing ids"
    with pytest.raises(TypeError):
        model_facing_tool_result({"content": "x"})  # type: ignore[arg-type]


def test_model_facing_text_renders_calls_results_and_hides_structured_data() -> None:
    request = ModelRequest(
        model="scripted",
        messages=[
            Message.text("user", "Audit data.csv"),
            Message(
                role="assistant",
                content=(
                    ToolCallPart(
                        tool_call_id="c1", tool_name="audit_csv", arguments={"path": "data.csv"}
                    ),
                ),
            ),
            Message(
                role="tool",
                content=(
                    ToolResultPart(
                        tool_call_id="c1",
                        tool_name="audit_csv",
                        content="CSV audit complete",
                        structured={"blank_rows": 3},
                    ),
                ),
            ),
        ],
    )

    assert model_facing_text(request).splitlines() == [
        "user: Audit data.csv",
        'assistant: call audit_csv {"path": "data.csv"}',
        "tool audit_csv: CSV audit complete",
    ]


def test_model_facing_text_shows_peer_content_with_its_attribution() -> None:
    peer = PeerContentPart(
        text="The reviewer found 2 blank rows.",
        sender_participant_id="reviewer",
        sender_participant_incarnation="inc-1",
        sender_session_id="session-2",
        sender_session_instance_id="instance-2",
        occurrence_id="occurrence-7",
        append_key_json="{}",
        operation_key="operation-1",
        projection_id="projection-1",
        provenance_sha256="0" * 64,
    )
    request = ModelRequest(
        model="scripted",
        messages=[Message(role="assistant", content=(peer,))],
    )

    assert model_facing_text(request) == (
        "assistant: [Peer content from reviewer; occurrence occurrence-7]\n"
        "The reviewer found 2 blank rows."
    )


def test_scripted_run_exposes_findings_the_model_never_received() -> None:
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="c1", name="audit_csv", arguments={"path": "data.csv"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("Done."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="auditor", model="scripted"),
        tools=[_AuditCsv()],
        tool_exposure_policy=StaticToolExposurePolicy(profile_id="audit", tools=("audit_csv",)),
        tool_policy=StaticToolPolicy(),
    )

    outcome = asyncio.run(
        run_to_completion(
            app, RunRequest(agent_name="auditor", messages=[Message.text("user", "Audit")])
        )
    )

    assert outcome.ok
    seen = model_facing_text(provider.requests[1])
    assert "tool audit_csv: CSV audit complete" in seen
    # The structured findings exist, but the real model would never read them.
    assert "blank_rows" not in seen
