from __future__ import annotations

import asyncio

import pytest
from tests.core.test_tool_round_execution_identities import _SequencedProvider

from cayu import (
    AgentSpec,
    CayuApp,
    ExecutionProfileBehaviorIdentity,
    Message,
    ResumeRequest,
    RunRequest,
    Tool,
    ToolResult,
    ToolSpec,
)
from cayu.core import EventType, ToolResultPart
from cayu.environments import Environment, EnvironmentSpec
from cayu.providers import ModelStreamEvent
from cayu.runners import ExecResult, Runner
from cayu.runtime import InMemorySessionStore
from cayu.runtime._tool_effect_state import ToolEffectStateOwner
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.command_policy import CommandPolicy, CommandPolicyDecision, CommandPolicyResult
from cayu.tools.commands import ExecCommandTool


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize(
    "decision", [CommandPolicyDecision.DENY, CommandPolicyDecision.REQUIRE_COMMAND_APPROVAL]
)
def test_command_refusal_settles_effect_without_uncertainty(backend, recover, decision, tmp_path):
    async def scenario():
        identity = ExecutionProfileBehaviorIdentity(
            name="test:command-refusal", behavior_version="1", implementation_version="1"
        )

        class Provider(_SequencedProvider):
            @property
            def execution_profile_identity(self):
                return identity

        class CloseFailure:
            invocation_lifecycle_command_version = 1
            armed = recover

            async def publish_runtime_publication(self, session_id, *, request, **kwargs):
                if self.armed and request.kind == "tool-round":
                    self.armed = False
                    raise RuntimeError("refused round close unavailable")
                return await super().publish_runtime_publication(
                    session_id, request=request, **kwargs
                )

        class Memory(CloseFailure, InMemorySessionStore):
            invocation_lifecycle_command_version = 1

        class SQLite(CloseFailure, SQLiteSessionStore):
            invocation_lifecycle_command_version = 1

        class NeverCalledRunner(Runner):
            calls = 0

            @property
            def execution_profile_identity(self):
                return identity

            async def exec(self, command, **kwargs):
                self.calls += 1
                return ExecResult(stdout="unexpected")

        class RefusalPolicy(CommandPolicy):
            @property
            def execution_profile_identity(self):
                return identity

            async def evaluate(self, ctx, request):
                return CommandPolicyResult(decision=decision, reason="Not authorized.")

        store = Memory() if backend == "memory" else SQLite(tmp_path / "refusal.sqlite")
        runner = NeverCalledRunner()
        provider = Provider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call", name="exec_command", arguments={"argv": ["git", "push"]}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("Refused safely."),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )

        def make_app():
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_environment(
                Environment(
                    EnvironmentSpec(name="commands", execution_profile_identity=identity),
                    runner=runner,
                ),
                default=True,
            )
            app.register_agent(
                AgentSpec(name="agent", model="test"),
                tools=[ExecCommandTool(policy=RefusalPolicy())],
            )
            return app

        try:
            app = make_app()
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="refusal",
                        agent_name="agent",
                        messages=[Message.text("user", "push")],
                    )
                )
            ]
            assert events[-1].type is (
                EventType.SESSION_FAILED if recover else EventType.SESSION_COMPLETED
            )
            durable = await store.load_events("refusal")
            [blocked] = [event for event in durable if event.type is EventType.TOOL_CALL_BLOCKED]
            session = await store.load("refusal")
            record = await ToolEffectStateOwner(store).resolve_call(
                session, tool_round_id=blocked.payload["tool_round_id"], tool_call_id="call"
            )
            assert record.state == "failed"
            assert record.dispatch_id is not None
            assert record.terminal.event_id == blocked.id
            assert record.terminal.receipt is None
            assert runner.calls == 0
            assert not any(event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN for event in durable)
            if recover:
                if backend == "sqlite":
                    await store.close()
                    store = SQLite(tmp_path / "refusal.sqlite")
                    store.armed = False
                app = make_app()
                resumed = [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id="refusal", messages=[Message.text("user", "continue")]
                        )
                    )
                ]
                assert resumed[-1].type is EventType.SESSION_COMPLETED
            final = await store.load_events("refusal")
            assert [event.id for event in final if event.type is EventType.TOOL_CALL_BLOCKED] == [
                blocked.id
            ]
            assert not any(event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN for event in final)
            assert await ToolEffectStateOwner(store).load(record.intent) == record
            transcript = await store.load_transcript("refusal")
            results = [
                part
                for message in transcript
                for part in message.content
                if isinstance(part, ToolResultPart)
            ]
            assert len(results) == 1 and results[0].is_error
            assert runner.calls == 0
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


def test_raw_refusal_shaped_result_does_not_acquire_policy_authority():
    class Lookalike(Tool):
        spec = ToolSpec(name="lookalike")

        async def run(self, ctx, args):
            return ToolResult(
                content="Command denied by policy.",
                is_error=True,
                structured={"denied_by": "command_policy", "decision": "deny"},
            )

    async def scenario():
        store = InMemorySessionStore()
        provider = _SequencedProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="call", name="lookalike", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("Done."),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="agent", model="test"), tools=[Lookalike()])
        events = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="raw-refusal",
                    agent_name="agent",
                    messages=[Message.text("user", "run")],
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        durable = await store.load_events("raw-refusal")
        assert not any(event.type is EventType.TOOL_CALL_BLOCKED for event in durable)
        [failed] = [event for event in durable if event.type is EventType.TOOL_CALL_FAILED]
        assert "denied_by" not in failed.payload
        session = await store.load("raw-refusal")
        record = await ToolEffectStateOwner(store).resolve_call(
            session, tool_round_id=failed.payload["tool_round_id"], tool_call_id="call"
        )
        assert record.state == "failed" and record.terminal.event_id == failed.id

    asyncio.run(scenario())
