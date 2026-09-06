from __future__ import annotations

import asyncio
import errno
import json

import pytest
from tests.core._workload_secret_support import FakeProvider, collect_events

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    Event,
    EventType,
    ExecCommand,
    Message,
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


def verify_durable_failure(tmp_path, runner, command, number, phase):
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
    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="failure-evidence",
                messages=[Message.text("user", "run")],
            ),
        )
    )
    # A fresh store instance reads durable evidence; JSON export/import retains it too.
    replay = asyncio.run(SQLiteSessionStore(store_path).load_events("failure-evidence"))
    exported = [Event.model_validate_json(event.model_dump_json()) for event in replay]
    for source in (events, replay, exported):
        started = next(e for e in source if e.type == EventType.RUNNER_EXEC_STARTED)
        completed = next(e for e in source if e.type == EventType.RUNNER_EXEC_COMPLETED)
        failed = next(e for e in source if e.type == EventType.TOOL_CALL_FAILED)
        assert started.payload["execution_id"] == completed.payload["execution_id"]
        assert completed.payload["errno"] == number
        assert completed.payload["errno_code"] == errno.errorcode[number]
        assert completed.payload["execution_phase"] == phase
        assert failed.payload["outcome_unknown"] is True
        assert failed.payload["manual_reconciliation_required"] is True
        serialized = json.dumps([e.model_dump(mode="json") for e in source])
        assert "secret-failure-canary" not in serialized
        diagnostics = []

        def visit(value, diagnostics=diagnostics):
            if type(value) is dict:
                if value.get("type") == "cayu.runner_execution_error.v1":
                    diagnostics.append(value)
                for child in value.values():
                    visit(child)
            elif type(value) is list:
                for child in value:
                    visit(child)

        visit([e.model_dump(mode="json") for e in source])
        assert diagnostics
        for diagnostic in diagnostics:
            assert diagnostic["errno"] == number
            assert diagnostic["execution_phase"] == phase
            assert diagnostic["timed_out"] is False
            assert diagnostic["cancelled"] is False


@pytest.mark.parametrize(
    "number,phase", [(errno.EMFILE, "launch"), (errno.ENOSPC, "stream_handling")]
)
def test_runner_failure_durable_tool_evidence(tmp_path, number, phase):
    class FailingRunner(Runner):
        isolation = "docker"
        calls = 0

        async def exec(self, command, **kwargs):
            self.calls += 1
            error = OSError(number, "secret-failure-canary")
            tag_runner_failure_phase(error, phase)
            raise error

    runner = FailingRunner()
    verify_durable_failure(tmp_path, runner, ExecCommand.process("unused"), number, phase)
    assert runner.calls == 1
