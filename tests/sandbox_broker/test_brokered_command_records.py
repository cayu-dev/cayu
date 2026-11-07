"""A brokered command's id and broker times reach its runner event and its tool call."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from tests.core.test_human_review import identity
from tests.core.test_workspace_mutation_receipts import _SingleToolProvider
from tests.sandbox_broker.fake_broker import FakeBroker

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.cli.session import _runner_commands_cell
from cayu.environments import EnvironmentSpec
from cayu.events import Event
from cayu.runners import ExecCommand, ExecResult, RunnerCommandRecord
from cayu.runtime._event_projection import project_persisted_runtime_event
from cayu.sandbox_broker import BrokeredSandboxCodingEnvironmentFactory, BrokeredSandboxRunner
from cayu.server._session_call_routes import _public_tool_call
from cayu.server.contracts import SessionToolCallsResponse
from cayu.sessions._call_inspection import (
    InspectionWindow,
    inspect_tool_calls,
    runner_inspection_record,
    tool_call_rows,
)
from cayu.sessions.base import EventRecord
from cayu.storage import SQLiteSessionStore
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.vaults import SecretRedactor
from cayu.workspaces import LocalWorkspace


async def _ready_runner(broker: FakeBroker) -> BrokeredSandboxRunner:
    client = broker.client()
    created = await client.create(
        idempotency_key=f"k-{len(broker.sandboxes)}",
        image="coding-python",
        egress_mode="none",
        lease_seconds=60,
    )
    sandbox = await client.wait_until_ready(created.id, timeout_s=5, poll_interval_s=0)
    return BrokeredSandboxRunner(client, sandbox, poll_wait_seconds=1)


def test_a_brokered_result_carries_the_brokers_command_record(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario() -> ExecResult:
        runner = await _ready_runner(broker)
        try:
            return await runner.exec(ExecCommand.process("true"))
        finally:
            await runner.close()

    result = asyncio.run(scenario())

    (sandbox,) = broker.sandboxes.values()
    (command_id,) = sandbox.commands
    assert result.runner_command == RunnerCommandRecord(
        id=command_id,
        accepted_at=datetime(2026, 10, 18, 12, 0, tzinfo=UTC),
        started_at=datetime(2026, 10, 18, 12, 0, 0, 250_000, tzinfo=UTC),
        finished_at=datetime(2026, 10, 18, 12, 0, 1, tzinfo=UTC),
    )
    # The record stays in process; a serialized result is what it was.
    assert "runner_command" not in result.model_dump()
    assert ExecResult.model_validate(result.model_dump()).runner_command is None


def _session(tmp_path: Path, broker: FakeBroker) -> SQLiteSessionStore:
    source = tmp_path / "source"
    source.mkdir()
    (source / "calc.py").write_text("x = 1\n")

    class RunTool(Tool):
        spec = ToolSpec(name="run_twice", parallel_safe=False)

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.runner is not None
            await ctx.runner.exec(ExecCommand.process("true"))
            await ctx.runner.exec(ExecCommand.process("echo", "ok"))
            return ToolResult(content="ran")

    store_path = tmp_path / "sessions.sqlite"

    async def scenario() -> list[Event]:
        store = SQLiteSessionStore(store_path)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            _SingleToolProvider(tool_name="run_twice", arguments={}), default=True
        )
        app.register_environment_factory(
            EnvironmentSpec(name="sandbox", execution_profile_identity=identity("brokered")),
            BrokeredSandboxCodingEnvironmentFactory(
                broker=broker.client(),
                source_workspace=LocalWorkspace(source, workspace_id="project"),
                egress="none",
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="coder", model="scripted-model"), tools=[RunTool()])
        try:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="coding",
                        agent_name="coder",
                        messages=[Message.text("user", "run")],
                    )
                )
            ]
        finally:
            await store.close()

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.SESSION_COMPLETED, events[-1].payload
    return SQLiteSessionStore(store_path)


def test_runner_events_and_tool_calls_name_the_brokers_commands(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    store = _session(tmp_path, broker)
    (sandbox,) = broker.sandboxes.values()

    async def read() -> tuple[list[Event], dict]:
        try:
            events = await store.load_events("coding")
            calls = await inspect_tool_calls(
                store, "coding", InspectionWindow(), offset=0, limit=10
            )
            return events, calls
        finally:
            await store.close()

    events, calls = asyncio.run(read())

    completed = [
        event
        for event in events
        if event.type == EventType.RUNNER_EXEC_COMPLETED and event.tool_name == "run_twice"
    ]
    assert len(completed) == 2
    ids = [event.payload["runner_command"]["id"] for event in completed]
    # Each event names a command the broker ran, so it can be found in its records.
    assert set(ids) <= set(sandbox.commands)
    assert completed[0].payload["runner_command"] == {
        "id": ids[0],
        "accepted_at": "2026-10-18T12:00:00+00:00",
        "started_at": "2026-10-18T12:00:00.250000+00:00",
        "finished_at": "2026-10-18T12:00:01+00:00",
    }

    # Public event projections, as `cayu session events` shows them, keep the record.
    public = project_persisted_runtime_event(completed[0], sequence=1, redactor=SecretRedactor())
    assert public.payload["runner_command"] == completed[0].payload["runner_command"]

    (call,) = [row for row in calls["calls"] if row["tool"] == "run_twice"]
    commands = call["runner_commands"]
    assert [command["id"] for command in commands] == ids
    assert all(type(command["adapter"]) is str for command in commands)
    assert all(type(command["runtime_ms"]) is int for command in commands)
    assert {
        (command["backend_ms"], command["queue_ms"], command["run_ms"]) for command in commands
    } == {(1000, 250, 750)}
    cell = _runner_commands_cell(call)
    assert cell is not None and cell.startswith(f"{ids[0]} runtime=")
    assert "backend=1000ms queue=250ms run=750ms" in cell
    # The server's tool-call contract carries the same rows.
    response = SessionToolCallsResponse.model_validate(
        {"session_id": "coding", **calls, "calls": [_public_tool_call(call, "coding", None)]}
    )
    assert response.calls[0].runner_commands[0].id == ids[0]


def test_a_command_without_a_backend_record_still_shows_its_runtime() -> None:
    def record(sequence: int, event_type: EventType, **payload: object) -> EventRecord:
        return EventRecord(
            sequence=sequence,
            event=Event(
                type=event_type,
                session_id="s",
                tool_name="exec_command",
                payload={
                    "tool_call_id": "call-1",
                    "model_step_id": "mstep_" + "1" * 32,
                    "model_attempt_id": "matt_" + "2" * 32,
                    "tool_round_id": "tround_" + "3" * 32,
                    **payload,
                },
            ),
        )

    runner = runner_inspection_record(
        record(2, EventType.RUNNER_EXEC_COMPLETED, adapter="LocalRunner", duration_ms=12)
    )
    (row,) = tool_call_rows(
        [
            record(1, EventType.TOOL_CALL_STARTED, arguments={}),
            record(3, EventType.TOOL_CALL_COMPLETED, result={"content": "ok"}),
        ],
        [runner],
    )

    (command,) = row["runner_commands"]
    assert command["id"] is None and command["adapter"] == "LocalRunner"
    assert command["runtime_ms"] == 12
    assert (command["backend_ms"], command["queue_ms"], command["run_ms"]) == (None, None, None)
    assert _runner_commands_cell(row) == "LocalRunner runtime=12ms"
    # A call without runner commands keeps an empty list.
    (plain,) = tool_call_rows([record(1, EventType.TOOL_CALL_STARTED, arguments={})])
    assert plain["runner_commands"] == [] and _runner_commands_cell(plain) is None
