import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_tool_support import ExpiringApprovalPolicy, SideEffectTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.pending_action_contracts import PendingActionQuery
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import TaskStatus
from cayu.tasks.worker import TaskHandlerOutcome, run_task_worker


def test_expired_approval_preserves_elected_worker_handoff():
    asyncio.run(_scenario())


async def _scenario():
    now = datetime(2026, 7, 9, 12, 0, tzinfo=UTC)
    tasks = InMemoryTaskStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("denial handled"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(task_store=tasks, clock=lambda: now, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=ExpiringApprovalPolicy(expires_in_seconds=60),
    )

    async def pause(runtime, task, worker_id):
        events = [
            event
            async for event in runtime.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="expiry-session",
                    task_id=task.id,
                    task_worker_id=worker_id,
                    task_lease_expires_at=task.lease_expires_at,
                    messages=[Message.text("user", "Use the tool")],
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_INTERRUPTED
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    await tasks.create_task(TaskCreate(task_id="expiry-task", type="job"))
    assert (
        await run_task_worker(
            app,
            tasks,
            pause,
            worker_id="first-worker",
            query=TaskQuery(type="job"),
            max_tasks=1,
            reclaim=False,
        )
        == 1
    )
    now += timedelta(minutes=2)
    elected = (
        await tasks.claim_interrupted_task_continuation(
            "next-worker",
            TaskQuery(type="job"),
            handoff_id=str(uuid4()),
        )
    ).task
    assert elected is not None and elected.interrupted_handoff_id is not None
    approval = (
        await app.session_store.query_pending_actions(
            PendingActionQuery(session_id="expiry-session")
        )
    ).actions[0]
    events = [
        event
        async for event in app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id="expiry-session",
                task_worker_id="next-worker",
                task_handoff_id=elected.interrupted_handoff_id,
                approval_id=approval.approval_id,
                tool_round_id=approval.round_id,
                tool_call_id=approval.tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
            )
        )
    ]
    assert any(event.type is EventType.TOOL_CALL_APPROVAL_EXPIRED for event in events)
    assert tool.calls == []
    assert (await tasks.load_task("expiry-task")).status is TaskStatus.COMPLETED
    assert (await app.session_store.load("expiry-session")).status is SessionStatus.COMPLETED
