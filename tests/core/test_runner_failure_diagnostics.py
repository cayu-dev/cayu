from __future__ import annotations

import asyncio
import contextlib
import errno
import json

import pytest
from tests.core._workload_secret_support import FakeProvider, collect_resume_events

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    Event,
    EventType,
    ExecCommand,
    Message,
    ResumeRequest,
    RunRequest,
    SQLiteSessionStore,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from cayu.providers import ModelStreamEvent
from cayu.runners import Runner
from cayu.runners._diagnostics import tag_runner_failure_phase


def verify_durable_failure(tmp_path, runner, command, number, phase, *, abandon_stream=False):
    class RunTool(Tool):
        spec = ToolSpec(name="run_command", description="Run.", input_schema={"type": "object"})

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            await ctx.runner.exec(command)
            return ToolResult(content="unexpected")

    store_path = tmp_path / "sessions.sqlite"
    app = CayuApp(session_store=SQLiteSessionStore(store_path), enable_logging=False)
    app.register_provider(
        FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="call-failure", name="run_command", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    app.register_environment(
        Environment(EnvironmentSpec(name="runner"), runner=runner), default=True
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[RunTool()])

    async def execute():
        captured = []
        async with contextlib.aclosing(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="failure-evidence",
                    messages=[Message.text("user", "run")],
                )
            )
        ) as stream:
            async for event in stream:
                captured.append(event)
                if abandon_stream and event.type is EventType.RUNNER_EXEC_STARTED:
                    break
        return captured

    events = asyncio.run(execute())
    # A fresh store instance reads durable evidence; JSON export/import retains it too.
    replay = asyncio.run(SQLiteSessionStore(store_path).load_events("failure-evidence"))
    exported = [Event.model_validate_json(event.model_dump_json()) for event in replay]
    sources = (replay, exported) if abandon_stream else (events, replay, exported)
    for source in sources:
        started = next(e for e in source if e.type == EventType.RUNNER_EXEC_STARTED)
        completed = next(e for e in source if e.type == EventType.RUNNER_EXEC_COMPLETED)
        unknown = next(e for e in source if e.type == EventType.TOOL_EFFECT_OUTCOME_UNKNOWN)
        assert started.payload["execution_id"] == completed.payload["execution_id"]
        assert completed.payload["errno"] == number
        assert completed.payload["errno_code"] == errno.errorcode[number]
        assert completed.payload["execution_phase"] == phase
        assert completed.payload["error_type"] == "OSError"
        assert unknown.payload["state"] == "outcome_unknown"
        assert unknown.payload["failure_evidence"]["classification"] == "failure"
        assert not any(
            e.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED} for e in source
        )
        serialized = json.dumps([e.model_dump(mode="json") for e in source])
        assert "secret-failure-canary" not in serialized
        # Runner diagnostics remain on their own durable event. Unknown tool
        # effects deliberately have no synthetic result or result artifacts.
        assert "result" not in unknown.payload

    assert events[-1].type is (
        EventType.RUNNER_EXEC_STARTED if abandon_stream else EventType.SESSION_INTERRUPTED
    )
    checkpoint = asyncio.run(app.session_store.load_checkpoint("failure-evidence"))
    assert checkpoint is not None and "pending_tool_round" in checkpoint
    resumed = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="failure-evidence",
                messages=[Message.text("user", "continue")],
            ),
        )
    )
    assert resumed[-1].type is EventType.SESSION_INTERRUPTED
    durable = asyncio.run(app.session_store.load_events("failure-evidence"))
    for kind in (
        EventType.RUNNER_EXEC_STARTED,
        EventType.RUNNER_EXEC_COMPLETED,
        EventType.TOOL_EFFECT_OUTCOME_UNKNOWN,
        EventType.MODEL_STARTED,
    ):
        assert sum(event.type is kind for event in durable) == 1
    assert not any(
        event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        for event in durable
    )


@pytest.mark.parametrize(
    "number,phase", [(errno.EMFILE, "launch"), (errno.ENOSPC, "stream_handling")]
)
@pytest.mark.parametrize("abandon_stream", [False, True], ids=["consume", "aclose"])
def test_runner_failure_durable_tool_evidence(tmp_path, number, phase, abandon_stream):
    class FailingRunner(Runner):
        isolation = "docker"
        calls = 0

        async def exec(self, command, **kwargs):
            self.calls += 1
            error = OSError(number, "secret-failure-canary")
            tag_runner_failure_phase(error, phase)
            raise error

    runner = FailingRunner()
    verify_durable_failure(
        tmp_path,
        runner,
        ExecCommand.process("unused"),
        number,
        phase,
        abandon_stream=abandon_stream,
    )
    assert runner.calls == 1
