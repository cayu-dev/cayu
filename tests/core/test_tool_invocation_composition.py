"""Invocation parts remain usable independently and across execution entrances."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from contextlib import aclosing
from pathlib import Path

import pytest
from tests.core.test_durable_tool_round import _app
from tests.core.test_tool_round_continuation_backends import _pause, _resolve, _runtime

import cayu
from cayu import EventType, InMemorySessionStore, Message, RunRequest
from cayu.tools.base import ToolEffect


@pytest.mark.parametrize(
    "first_import",
    [
        "cayu.runtime._tool_invocation.invocation",
        "cayu.workspaces.checkpoint_lifecycle",
        "cayu.runtime._tool_execution",
    ],
)
def test_invocation_parts_work_without_round_or_application_owners(first_import):
    script = """
import asyncio
import importlib
import importlib.abc
import sys

blocked = {
    "cayu.applications", "cayu.runtime._tool_round_executor",
    "cayu.runtime._session_engine", "cayu.runtime._recovery_coordinator",
}
class RejectOrchestration(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError(f"Invocation part imported {fullname}")
sys.meta_path.insert(0, RejectOrchestration())
importlib.import_module(sys.argv[1])
from cayu.runtime._tool_invocation.invocation import ToolInvocation
from cayu.runtime._tool_invocation.evidence import InvocationPublication
from cayu.runtime._tool_invocation.admission import ToolInvocationAdmission
from cayu.runtime._tool_invocation.resources import ToolInvocationResources
from cayu.runtime._tool_invocation.terminal import ToolTerminalPublisher
from cayu.runtime._paused_tool_round import PausedToolRound, ApprovalRoundPause, UserInputRoundPause
from cayu.runtime._continuation_environment import ContinuationEnvironment
from cayu.runtime._manual_recovery_publication import ManualRecoveryPublication
from cayu.runtime._assistant_model_publication import AssistantModelPublication
from cayu.runtime._execution_profile_continuation import ExecutionProfileContinuation
from cayu.runtime._model_completion_recovery import ModelCompletionRecovery
from cayu.runtime._user_input_recovery_evidence import UserInputRecoveryEvidence
from cayu.runtime._runtime_records import ToolCallRequest
from cayu.runtime._tool_execution import run_tool
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.vaults.redaction import SecretRedactor

class Echo(Tool):
    spec = ToolSpec(name="echo", description="Echo a value.",
        input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
        effect=ToolEffect.NONE)
    async def run(self, ctx, args):
        return ToolResult(content=args["value"])

async def scenario():
    records = []
    async def observe(call_id, snapshot):
        records.append((call_id, snapshot))
    publication = InvocationPublication(
        tool_call=ToolCallRequest(id="echo-1", name="echo", arguments={"value": "ok"}),
        redactor=SecretRedactor(), observer=observe)
    snapshot = await publication.static_scope()
    await publication.record(snapshot)
    assert records == [("echo-1", snapshot)]
    outcome = await run_tool(tool=Echo(), effect=ToolEffect.NONE,
        ctx=ToolContext(session_id="independent"), arguments={"value": "ok"},
        redactor=SecretRedactor)
    assert outcome.result.content == "ok" and not outcome.result.is_error
asyncio.run(scenario())
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, first_import],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("entrance", ["ordinary", "approve", "answer"])
def test_live_and_resumed_calls_use_the_composed_invocation(monkeypatch, entrance):
    async def scenario():
        if entrance == "ordinary":
            app, store, calls = _app(1, effect=ToolEffect.NONE)
        else:
            store = InMemorySessionStore()
            app, _provider, tool = _runtime(store, entrance, include_denied=False)
            calls = tool.calls
        owner = app._tool_round_executor.invocation
        execute = owner.execute
        observed = []

        async def record_invocation(**kwargs):
            observed.append(kwargs["tool_call"].id)
            async with aclosing(execute(**kwargs)) as stream:
                async for item in stream:
                    yield item

        monkeypatch.setattr(owner, "execute", record_invocation)
        session_id = f"invocation-{entrance}"
        if entrance == "ordinary":
            stream = app.run(
                RunRequest(
                    session_id=session_id,
                    agent_name="worker",
                    messages=[Message.text("user", "go")],
                )
            )
        else:
            request = await _pause(app, entrance, session_id)
            assert observed == []
            stream = _resolve(app, request)
        async with aclosing(stream):
            events = [event async for event in stream]
        assert events[-1].type is EventType.SESSION_COMPLETED
        expected = {
            "ordinary": ["call-0"],
            "approve": ["pause", "allowed"],
            "answer": ["allowed"],
        }[entrance]
        assert observed == expected
        assert len(calls) == len(expected)
        terminals = [
            event
            for event in await store.load_events(session_id)
            if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        ]
        terminal_ids = [event.payload["tool_call_id"] for event in terminals]
        assert set(expected) <= set(terminal_ids)
        assert len(terminal_ids) == len(set(terminal_ids))
        assert not owner.detached_environment_work()
        metrics = app.tool_terminal_publication_status()
        assert metrics.active_round_reservations == metrics.staged_count == 0

    asyncio.run(scenario())
