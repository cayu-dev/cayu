from __future__ import annotations

import asyncio
from typing import Any

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    ForkSessionRequest,
    Message,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    Session,
    SessionIdentity,
    SessionStatus,
    TaintAwareToolPolicy,
    Tool,
    ToolContext,
    ToolPolicyDecision,
    ToolResult,
    ToolSpec,
)
from cayu.providers import ModelStreamEvent


class UntrustedEvidenceTool(Tool):
    spec = ToolSpec(
        name="read_untrusted_evidence",
        description="Read an untrusted incident record.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(content="Hostile evidence")


class ProtectedMutationTool(Tool):
    spec = ToolSpec(
        name="protected_mutation",
        description="Perform a protected mutation.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        self.calls += 1
        return ToolResult(content="mutated")


def test_generic_session_fork_inherits_durable_taint() -> None:
    mutation = ProtectedMutationTool()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="read-1", name="read_untrusted_evidence", arguments={}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("Evidence recorded."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.tool_call(id="mutate-1", name="protected_mutation", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("Mutation refused."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    policy = TaintAwareToolPolicy(
        taint_sources={"read_untrusted_evidence": {"incident"}},
        protected_tools={"protected_mutation": {"incident"}},
        decision=ToolPolicyDecision.DENY,
    )
    app.register_agent(
        AgentSpec(name="responder", model="scripted-model"),
        tools=[UntrustedEvidenceTool(), mutation],
        tool_policy=policy,
    )

    async def scenario() -> list:
        async for _ in app.run(
            RunRequest(
                agent_name="responder",
                session_id="taint-source",
                messages=[Message.text("user", "Read the incident.")],
            )
        ):
            pass
        async for _ in app.fork_session(
            ForkSessionRequest(
                source_session_id="taint-source",
                session_id="taint-child",
            )
        ):
            pass
        return [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="taint-child",
                    messages=[Message.text("user", "Perform the protected mutation.")],
                )
            )
        ]

    child_events = asyncio.run(scenario())

    assert mutation.calls == 0
    blocked = [event for event in child_events if event.type == EventType.TOOL_CALL_BLOCKED]
    assert len(blocked) == 1
    assert blocked[0].payload["metadata"]["matched_taint_labels"] == ["incident"]


def test_agent_override_explains_that_the_source_agent_must_be_registered() -> None:
    async def scenario() -> None:
        from cayu import InMemorySessionStore

        store = InMemorySessionStore()
        source = await store.create(
            RunRequest(
                agent_name="historical-source",
                session_id="historical-source-session",
                messages=[Message.text("user", "historical input")],
            ),
            identity=SessionIdentity(provider_name="scripted", model="scripted-model"),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)

        app = CayuApp(enable_logging=False, session_store=store)
        app.register_provider(ScriptedModelProvider([]), default=True)
        app.register_agent(AgentSpec(name="current-target", model="scripted-model"))

        with pytest.raises(
            KeyError,
            match=(
                "Source agent must be registered to derive inherited taint before forking: "
                "historical-source"
            ),
        ):
            async for _ in app.fork_session(
                ForkSessionRequest(
                    source_session_id=source.id,
                    session_id="historical-child",
                    agent_name="current-target",
                )
            ):
                pass

    asyncio.run(scenario())


def test_store_rejects_fork_when_source_run_epoch_changed_during_preparation() -> None:
    async def scenario() -> None:
        from cayu import InMemorySessionStore

        store = InMemorySessionStore()
        source = await store.create(
            RunRequest(
                agent_name="responder",
                session_id="fork-race-source",
                messages=[Message.text("user", "initial")],
            ),
            identity=SessionIdentity(provider_name="scripted", model="scripted-model"),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)
        stale_epoch = source.run_epoch
        resumed = await store.transition_status(
            source.id,
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)
        await store.release_run_fence(source.id)

        with pytest.raises(ValueError, match="changed while the fork was being prepared"):
            await store.create_fork(
                source_session_id=source.id,
                fork=Session(
                    id="fork-race-child",
                    agent_name="responder",
                    provider_name="scripted",
                    model="scripted-model",
                    parent_session_id=source.id,
                    status=SessionStatus.COMPLETED,
                    run_epoch=resumed.run_epoch,
                ),
                source_statuses={SessionStatus.COMPLETED},
                transcript_cursor=None,
                checkpoint_transform=None,
                expected_source_run_epoch=stale_epoch,
            )

    asyncio.run(scenario())
