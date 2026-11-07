"""Recovering a switched session's lost process must use the model-facing view."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from tests.core.test_session_model_switch import (
    _app,
    _EchoTool,
    _FailingToolRoundPublicationSQLiteStore,
    _NamedProvider,
    _profiled_source_identity,
)
from tests.core.test_tool_completion import CrashStore, FinalTool
from tests.core.test_tool_round_publication_failure_matrix import _SimulatedProcessLoss

from cayu import EventType, Message, ResumeRequest, RunRequest, SessionStatus
from cayu.messages import MessageRole, ProviderStatePart, TextPart, ThinkingPart
from cayu.providers import ModelStreamEvent
from cayu.runtime import ToolCompletionPolicy
from cayu.sessions import IncompleteSessionRecoveryRequest, InMemorySessionStore, ModelTarget
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools import ToolCapabilityCeiling


@pytest.fixture(params=["memory", "sqlite"])
def crash_store(request, tmp_path):
    backend_type = InMemorySessionStore if request.param == "memory" else SQLiteSessionStore

    @asynccontextmanager
    async def open_store(boundary):
        class FaultStore(CrashStore, backend_type):
            invocation_lifecycle_command_version = 1
            session_access_version = 1

        if request.param == "memory":
            store = FaultStore(boundary=boundary)
        else:
            store = FaultStore(tmp_path / "switch-recovery.sqlite", boundary=boundary)
        try:
            yield store
        finally:
            if request.param == "sqlite":
                await store.close()

    return open_store


@pytest.mark.parametrize("boundary", ["tool-event", "tool-publication"])
@pytest.mark.parametrize("opaque_only", [False, True], ids=["text-bearing", "opaque-only"])
def test_switched_session_recovers_recorded_final_tool(crash_store, boundary, opaque_only):
    session_id = f"switch-recovery-{boundary}"

    async def scenario():
        async with crash_store(boundary) as store:
            first = [] if opaque_only else [ModelStreamEvent.text_delta("first answer")]
            first.append(
                ModelStreamEvent.completed(
                    {"provider_state": [{"provider": "source", "state": {"id": "response-1"}}]}
                )
            )
            target = _NamedProvider(
                "target",
                [
                    [
                        ModelStreamEvent.tool_call(name="ask_customer", arguments={}, id="c1"),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ]
                ],
            )
            tool = FinalTool()
            app, _ = _app(_NamedProvider("source", [first]), target, store=store, tool=tool)
            initial = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "Help")],
                        tool_completion=ToolCompletionPolicy(tool_names=["ask_customer"]),
                    )
                )
            ]
            assert initial[-1].type is EventType.SESSION_COMPLETED
            with pytest.raises(_SimulatedProcessLoss):
                [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id=session_id,
                            messages=[Message.text("user", "Ask on the target")],
                            target=ModelTarget(provider_name="target", model="target-model"),
                            tool_completion=ToolCompletionPolicy(tool_names=["ask_customer"]),
                        )
                    )
                ]
            assert store.crashed and tool.calls == 1

            recovered_target = _NamedProvider("target", [])
            recovered_tool = FinalTool()
            reconstructed, _ = _app(
                _NamedProvider("source", []), recovered_target, store=store, tool=recovered_tool
            )
            recovered = await reconstructed.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id, inactive_for_seconds=0)
            )
            assert recovered.status is SessionStatus.COMPLETED, recovered
            assert recovered_tool.calls == 0
            assert recovered_target.requests == []
            completion = [
                event
                for event in await store.load_events(session_id)
                if event.type is EventType.SESSION_COMPLETED
            ]
            assert completion[-1].payload["reason"] == "host_rendered_tool"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "layout", ["shell-first", "shell-last", "shell-then-reasoned-answer", "reasoned-answer-last"]
)
def test_pending_tool_recovery_keeps_every_switched_row_portable(tmp_path, layout):
    async def scenario():
        store = _FailingToolRoundPublicationSQLiteStore(tmp_path / "shell-recovery.sqlite")
        tool = _EchoTool()
        target = _NamedProvider(
            "target",
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call-after-shell", name="echo", arguments={"value": "portable"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.text_delta("recovered"), ModelStreamEvent.completed()],
            ],
        )
        app, _ = _app(_NamedProvider("source", []), target, store=store, tool=tool)
        session_id = "switch-shell-recovery"
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[],
                tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("echo",)),
            ),
            identity=_profiled_source_identity(tool=tool),
        )
        shell = Message(
            role=MessageRole.ASSISTANT, content=(ThinkingPart(text="source-only reasoning"),)
        )
        answer = Message.text("assistant", "portable answer")
        # A real answer usually carries its own reasoning and provider state.
        reasoned = Message(
            role=MessageRole.ASSISTANT,
            content=(
                ThinkingPart(text="answer reasoning"),
                TextPart(text="portable answer"),
                ProviderStatePart(provider="source", state={"id": "response-1"}),
            ),
        )
        rows = {
            "shell-first": (shell, answer),
            "shell-last": (answer, shell),
            "shell-then-reasoned-answer": (shell, reasoned),
            "reasoned-answer-last": (answer, reasoned),
        }[layout]
        await store.append_transcript_messages(session_id, [Message.text("user", "opening"), *rows])
        await store.update_status(session_id, SessionStatus.COMPLETED)
        failed = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "switch and call")],
                    target=ModelTarget(provider_name="target", model="target-model"),
                )
            )
        ]
        assert failed[-1].type is EventType.SESSION_FAILED
        recovered = [
            event
            async for event in app.resume(
                ResumeRequest(session_id=session_id, messages=[Message.text("user", "continue")])
            )
        ]
        assert recovered[-1].type is EventType.SESSION_COMPLETED
        assert len(target.requests) == 2
        assert all(
            type(part) not in {ProviderStatePart, ThinkingPart}
            for request in target.requests
            for message in request.messages
            for part in message.content
        )
        # The recovered history is the same model view plus the closed round.
        first = target.requests[0].messages
        assert target.requests[1].messages[: len(first)] == first
        await store.close()

    asyncio.run(scenario())


def test_finish_cutoff_after_a_switch_keeps_the_model_facing_history(monkeypatch):
    from cayu.runtime._run_limits import RunLimitGate
    from cayu.runtime.stop_policy import StopDecision, StopLimit

    cutoffs = []

    def finish_now(self):
        if cutoffs:
            return None
        cutoffs.append(True)
        return StopDecision(
            limit=StopLimit.ELAPSED_SECONDS,
            maximum=60,
            actual=59,
            message="Finishing before the elapsed-time limit.",
        )

    monkeypatch.setattr(RunLimitGate, "tool_cutoff_decision", finish_now)

    async def scenario():
        source = _NamedProvider(
            "source",
            [
                [
                    ModelStreamEvent.text_delta("first answer"),
                    ModelStreamEvent.completed(
                        {"provider_state": [{"provider": "source", "state": {"id": "r1"}}]}
                    ),
                ]
            ],
        )
        target = _NamedProvider(
            "target",
            [
                [
                    ModelStreamEvent.tool_call(id="c1", name="echo", arguments={"value": "x"}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.text_delta("finished"), ModelStreamEvent.completed()],
            ],
        )
        app, _ = _app(source, target)
        session_id = "switch-finish-cutoff"
        try:
            [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "opening")],
                    )
                )
            ]
            resumed = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "switch and call")],
                        target=ModelTarget(provider_name="target", model="target-model"),
                    )
                )
            ]
            assert resumed[-1].type is EventType.SESSION_COMPLETED, resumed[-1].payload
            assert cutoffs and len(target.requests) == 2
            assert all(
                type(part) not in {ProviderStatePart, ThinkingPart}
                for request in target.requests
                for message in request.messages
                for part in message.content
            )
            first = target.requests[0].messages
            assert target.requests[1].messages[: len(first)] == first
        finally:
            await app.aclose()

    asyncio.run(scenario())
