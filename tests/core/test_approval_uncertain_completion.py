"""An uncertain terminal append must not turn committed approval work into failure."""

import asyncio

import pytest
from tests.core.test_user_input import _ScriptedProvider
from tests.runtime.runtime_tool_support import PureRecordingTool, RequireApprovalPolicy

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ToolApprovalRequest
from cayu.events import EventType
from cayu.exceptions import TerminalEventPublicationUncertain
from cayu.messages import Message
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import RunRequest


class _LostCompletionAcknowledgement(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self):
        super().__init__()
        self.completed_event_id = None
        self.failed_readback = False

    async def append_events(self, session_id, events):
        await super().append_events(session_id, events)
        for event in events:
            if event.type is EventType.SESSION_COMPLETED and self.completed_event_id is None:
                self.completed_event_id = event.id
                raise OSError("completion append acknowledgement lost")

    async def query_events(self, query):
        if (
            self.completed_event_id is not None
            and query.event_id == self.completed_event_id
            and not self.failed_readback
        ):
            self.failed_readback = True
            raise OSError("completion readback unavailable")
        return await super().query_events(query)


@pytest.mark.parametrize("approval", [False, True])
def test_uncertain_completion_keeps_committed_session_outcome(approval):
    async def scenario():
        store = _LostCompletionAcknowledgement()
        tool = PureRecordingTool()
        provider = _ScriptedProvider([("work", "side_effect", {})])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            tool_policy=RequireApprovalPolicy() if approval else None,
        )
        request = RunRequest(
            session_id="uncertain-completion",
            agent_name="assistant",
            messages=[Message.text("user", "do the work")],
        )
        stream = app.run(request)
        if approval:
            paused = [event async for event in stream]
            gate = next(e for e in paused if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)
            stream = app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id=request.session_id,
                    approval_id=gate.payload["approval_id"],
                    tool_round_id=gate.payload["tool_round_id"],
                    tool_call_id=gate.payload["tool_call_id"],
                    decision="approve",
                )
            )
        failure = None
        try:
            _ = [event async for event in stream]
        except Exception as error:
            failure = error
        assert store.completed_event_id is not None and store.failed_readback
        persisted = await store.load_events(request.session_id)
        assert any(e.id == store.completed_event_id for e in persisted)
        current = await store.load(request.session_id)
        assert current.status is SessionStatus.COMPLETED, (current.status, failure)
        assert not any(e.type is EventType.SESSION_FAILED for e in persisted)
        assert isinstance(failure, TerminalEventPublicationUncertain), failure
        assert tool.calls == [{}] and len(provider.requests) == 2

    asyncio.run(scenario())
