"""Concurrent sessions share terminal publication capacity unless a tool declares no bound."""

from __future__ import annotations

import asyncio
import logging
import sys

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import Environment, EnvironmentSpec
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import MessageRole, TextPart, ToolCallPart
from cayu.providers import ModelStreamEvent
from cayu.runners import LocalRunner
from cayu.sessions.memory import InMemorySessionStore
from cayu.tools import ExecCommandTool, ProcessCommandPolicy, WriteFileTool
from cayu.tools.base import Tool, ToolEffect, ToolResult, ToolSpec
from cayu.tools.subagents import SubagentSpec, SubagentTool
from cayu.tools.terminal_publication import (
    ToolRoundAdmissionTimeoutError,
    ToolTerminalPublicationGovernor,
)
from cayu.workspaces import LocalWorkspace

_SESSIONS = 4
_REGISTRY_LOGGER = "cayu._application_agent_registry"
_GOVERNOR_LOGGER = "cayu.tools.terminal_publication"


class _HeldRoundsGovernor(ToolTerminalPublicationGovernor):
    """Hold each admitted round until the first ``hold`` admissions have settled.

    With an admission deadline of a microsecond, a round that has to wait for
    capacity fails at once, so ``timed_out`` counts the rounds that waited.
    """

    def __init__(self, *, hold: int) -> None:
        super().__init__(round_admission_timeout_s=1e-6)
        self._hold = hold
        self.admitted = 0
        self.timed_out = 0
        self.settled = asyncio.Event()

    async def _reserve_round(self, **arguments) -> None:
        try:
            await super()._reserve_round(**arguments)
        except ToolRoundAdmissionTimeoutError:
            self.timed_out += 1
            raise
        else:
            self.admitted += 1
        finally:
            if self.admitted + self.timed_out >= self._hold:
                self.settled.set()
        await self.settled.wait()


class _UnboundedTool(Tool):
    """An application tool that declares no terminal bound and waits to be released."""

    def __init__(self, release: asyncio.Event) -> None:
        super().__init__(
            ToolSpec(
                name="slow_report",
                description="Return one report.",
                effect=ToolEffect.EXTERNAL,
                input_schema={"type": "object", "additionalProperties": False},
            )
        )
        self.entered = asyncio.Event()
        self._release = release

    async def run(self, ctx, args):
        self.entered.set()
        await self._release.wait()
        return ToolResult(content="report")


def _label(request) -> str:
    first = request.messages[0]
    return next(part.text for part in first.content if isinstance(part, TextPart))


def _rounds_so_far(request) -> int:
    return sum(
        1
        for message in request.messages
        if message.role is MessageRole.ASSISTANT
        and any(isinstance(part, ToolCallPart) for part in message.content)
    )


def _coding_rounds(request) -> list[ModelStreamEvent]:
    """Each session writes its file and runs a command, then reads the file back."""

    label, workspace = _label(request).split(" ", 1)
    rounds = _rounds_so_far(request)
    if label == "holder":
        if rounds == 0:
            return [
                ModelStreamEvent.tool_call(id="holder-report", name="slow_report", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
    elif rounds == 0:
        return [
            ModelStreamEvent.tool_call(
                id=f"{label}-run",
                name="exec_command",
                arguments={
                    "argv": [sys.executable, "-c", f"print('{label}')"],
                    "cwd": workspace,
                    "timeout_s": 30,
                },
            ),
            ModelStreamEvent.tool_call(
                id=f"{label}-write",
                name="write_file",
                arguments={
                    "path": f"{label}.txt",
                    "content": f"written by {label}\n",
                    "mode": "create",
                },
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    elif rounds == 1:
        return [
            ModelStreamEvent.tool_call(
                id=f"{label}-read",
                name="exec_command",
                arguments={
                    "argv": [sys.executable, "-c", f"print(open('{label}.txt').read())"],
                    "cwd": workspace,
                    "timeout_s": 30,
                },
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    return [
        ModelStreamEvent.text_delta("done"),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


def _app(tmp_path, governor, *, extra_tools=()) -> CayuApp:
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    app._tool_round_executor.invocation.terminals.governor = governor
    app.register_provider(ScriptedModelProvider(response_factory=_coding_rounds), default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            runner=LocalRunner(tmp_path),
            workspace=LocalWorkspace(tmp_path),
        ),
        default=True,
    )
    policy = ProcessCommandPolicy(
        allowed_executables={sys.executable},
        allowed_cwds={str(tmp_path.resolve())},
        max_timeout_s=30,
    )
    app.register_agent(
        AgentSpec(name="coder", model="scripted-model"),
        tools=[ExecCommandTool(policy=policy), WriteFileTool(), *extra_tools],
    )
    return app


async def _run(app: CayuApp, label: str, workspace):
    return [
        event
        async for event in app.run(
            RunRequest(
                agent_name="coder",
                session_id=label,
                messages=[Message.text("user", f"{label} {workspace.resolve()}")],
            )
        )
    ]


def test_concurrent_coding_sessions_share_capacity_without_waiting(tmp_path, caplog):
    async def scenario():
        # Every session's first round keeps its lease until all of them hold
        # one, and a round that had to wait for capacity fails at once.
        governor = _HeldRoundsGovernor(hold=_SESSIONS)
        app = _app(tmp_path, governor)
        labels = [f"session-{index}" for index in range(_SESSIONS)]

        runs = await asyncio.wait_for(
            asyncio.gather(*(_run(app, label, tmp_path) for label in labels)), timeout=60
        )

        assert governor.timed_out == 0
        for label, events in zip(labels, runs, strict=True):
            assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
            terminals = [
                event.type
                for event in events
                if event.type
                in {
                    EventType.TOOL_CALL_COMPLETED,
                    EventType.TOOL_CALL_FAILED,
                    EventType.TOOL_CALL_BLOCKED,
                }
            ]
            assert terminals == [EventType.TOOL_CALL_COMPLETED] * 3
            assert (tmp_path / f"{label}.txt").read_text() == f"written by {label}\n"
        snapshot = governor.snapshot()
        assert governor.admitted == 2 * _SESSIONS
        # Every session's first round held its lease at the same time.
        assert snapshot.maximum_reserved_round_bytes >= _SESSIONS * 1024 * 1024
        assert snapshot.maximum_reserved_round_bytes < snapshot.configured_staged_capacity_bytes
        assert snapshot.active_round_reservations == 0
        assert snapshot.active_exclusive_rounds == 0
        assert snapshot.round_reservation_waiters == 0

    with caplog.at_level(logging.WARNING):
        asyncio.run(scenario())
    assert not [record for record in caplog.records if record.name == _REGISTRY_LOGGER]
    assert not [record for record in caplog.records if record.name == _GOVERNOR_LOGGER]


def test_an_undeclared_tool_holds_every_other_session_until_its_round_drains(tmp_path, caplog):
    async def scenario():
        governor = ToolTerminalPublicationGovernor(
            round_admission_timeout_s=None, round_admission_notice_s=0.05
        )
        release = asyncio.Event()
        unbounded = _UnboundedTool(release)
        app = _app(tmp_path, governor, extra_tools=[unbounded])

        holder = asyncio.create_task(_run(app, "holder", tmp_path))
        await asyncio.wait_for(unbounded.entered.wait(), timeout=30)
        assert governor.snapshot().active_exclusive_rounds == 1

        labels = [f"session-{index}" for index in range(2)]
        others = [asyncio.create_task(_run(app, label, tmp_path)) for label in labels]
        for _ in range(2000):
            if governor.snapshot().round_reservation_waiters == len(labels):
                break
            await asyncio.sleep(0.005)
        assert governor.snapshot().round_reservation_waiters == len(labels)
        # Declared rounds of other sessions wait without running any tool.
        assert not list(tmp_path.glob("session-*.txt"))
        for _ in range(2000):
            if any("exclusive_holder_sessions=holder" in r.getMessage() for r in caplog.records):
                break
            await asyncio.sleep(0.005)

        release.set()
        runs = await asyncio.wait_for(asyncio.gather(holder, *others), timeout=60)
        assert all(events[-1].type is EventType.SESSION_COMPLETED for events in runs)
        assert governor.snapshot().active_round_reservations == 0

    with caplog.at_level(logging.WARNING):
        asyncio.run(scenario())
    notices = [
        record.getMessage()
        for record in caplog.records
        if record.name == _GOVERNOR_LOGGER and "still waiting" in record.getMessage()
    ]
    assert notices
    assert all("exclusive_holder_sessions=holder" in notice for notice in notices)
    assert all("exclusive_holder_unbounded_tools=slow_report" in notice for notice in notices)
    assert all("waiting_round_unbounded_tools" not in notice for notice in notices)
    assert all("ToolSpec.max_terminal_payload_bytes" in notice for notice in notices)
    warnings = [record.getMessage() for record in caplog.records if record.name == _REGISTRY_LOGGER]
    assert len(warnings) == 1
    assert warnings[0].startswith("Tool slow_report declares no max_terminal_payload_bytes.")


def test_registration_warns_once_per_undeclared_application_tool(caplog):
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    release = asyncio.Event()
    with caplog.at_level(logging.WARNING, logger=_REGISTRY_LOGGER):
        app.register_agent(
            AgentSpec(name="first", model="scripted-model"),
            tools=[
                _UnboundedTool(release),
                ExecCommandTool(),
                WriteFileTool(),
                # A foreground child keeps its exclusive lease on purpose.
                SubagentTool(app, agents={"child": SubagentSpec(agent_name="second")}),
            ],
        )
        app.register_agent(
            AgentSpec(name="second", model="scripted-model"), tools=[_UnboundedTool(release)]
        )

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert "Tool slow_report declares no max_terminal_payload_bytes" in messages[0]
    assert "other sessions' tool rounds wait for them" in messages[0]
    assert "ToolSpec(max_terminal_payload_bytes=...)" in messages[0]
    assert "between 65536 and 2031616 bytes" in messages[0]
