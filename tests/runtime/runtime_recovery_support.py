"""Recovery support."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import Any

from tests.runtime.runtime_provider_support import FakeProvider, VersionedFakeProvider
from tests.runtime.runtime_session_support import collect_events
from tests.runtime.runtime_tool_support import SideEffectTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.run_limits import RunLimits
from cayu.environments.base import Environment
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import RunRequest
from cayu.vaults.redaction import SecretRedactor


class FailingTerminalToolEventStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, *, ownership_clock=None) -> None:
        super().__init__(ownership_clock=ownership_clock)
        self.failed_terminal_once = False
        self.release_calls = 0

    async def release_run_fence(self, session_id: str) -> None:
        self.release_calls += 1
        await super().release_run_fence(session_id)

    async def release_session_invocation(self, command):
        self.release_calls += 1
        return await super().release_session_invocation(command)

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        if not self.failed_terminal_once and any(
            event.type == EventType.TOOL_CALL_COMPLETED for event in events
        ):
            self.failed_terminal_once = True
            raise RuntimeError("terminal tool event unavailable")
        await super().append_events(session_id, events)


class FailNextRunFenceReleaseStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.fail_next_release = False

    async def release_run_fence(self, session_id: str) -> None:
        if self.fail_next_release:
            self.fail_next_release = False
            try:
                raise OSError("run fence database connection lost")
            except OSError as cause:
                raise RuntimeError("run fence release failed") from cause
        await super().release_run_fence(session_id)

    async def release_session_invocation(self, command):
        if self.fail_next_release:
            self.fail_next_release = False
            try:
                raise OSError("run fence database connection lost")
            except OSError as cause:
                raise RuntimeError("run fence release failed") from cause
        return await super().release_session_invocation(command)


class FailingOrdinaryToolResultCloseStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.failed_tool_round_close_once = False

    async def publish_runtime_publication(
        self,
        session_id: str,
        *,
        request,
        expected_statuses=None,
        expected_run_epoch=None,
        expected_transcript_cursor=None,
    ):
        if not self.failed_tool_round_close_once and request.kind == "tool-round":
            self.failed_tool_round_close_once = True
            raise RuntimeError("ordinary tool round close unavailable")
        return await super().publish_runtime_publication(
            session_id,
            request=request,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )


class FailingAfterPendingToolRoundCheckpointStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(
        self,
        *,
        ownership_clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(ownership_clock=ownership_clock)
        self.failed_pending_tool_round_once = False

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        checkpoint = await super().load_checkpoint(session_id)
        if (
            not self.failed_pending_tool_round_once
            and checkpoint is not None
            and "pending_tool_round" in checkpoint
        ):
            self.failed_pending_tool_round_once = True
            raise RuntimeError("pending tool round checkpoint persisted before crash")
        return checkpoint


class FailingSecondTerminalToolEventStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.completed_tool_events = 0

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        for event in events:
            if event.type == EventType.TOOL_CALL_COMPLETED:
                self.completed_tool_events += 1
                if self.completed_tool_events == 2:
                    raise RuntimeError("second terminal tool event unavailable")
        await super().append_events(session_id, events)


def _crashed_tool_round_app(
    session_id: str,
    store: FailingTerminalToolEventStore | None = None,
    *,
    clock: Callable[[], datetime] | None = None,
    secret_redactor: SecretRedactor | None = None,
    environment: Environment | None = None,
    provider_capture: list[FakeProvider] | None = None,
    limits: RunLimits | None = None,
    continuation_events: list[ModelStreamEvent] | None = None,
    tool: SideEffectTool | None = None,
) -> tuple[CayuApp, FailingTerminalToolEventStore, SideEffectTool, dict]:
    """Fail terminal-event publication after the only tool has returned.

    Returns a FAILED session with an intact pending round. For EXTERNAL tools
    the effect record and staged terminal are already selected; this is a
    publication-repair fixture, not an ambiguous external-effect fixture. A
    supplied replay-safe tool retains the ordinary manual-recovery contract.
    """
    store = store if store is not None else FailingTerminalToolEventStore()
    tool = SideEffectTool() if tool is None else tool
    provider = VersionedFakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            (
                [
                    ModelStreamEvent.text_delta("recovered"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
                if continuation_events is None
                else continuation_events
            ),
        ]
    )
    if provider_capture is not None:
        provider_capture.append(provider)
    app = CayuApp(
        session_store=store,
        clock=clock,
        secret_redactor=secret_redactor,
    )
    app.register_provider(provider, default=True)
    if environment is not None:
        app.register_environment(environment, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
                limits=RunLimits() if limits is None else limits,
            ),
        )
    )
    assert initial_events[-1].type == EventType.SESSION_FAILED
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint is not None and "pending_tool_round" in checkpoint
    assert tool.calls == [{}]
    return app, store, tool, checkpoint
