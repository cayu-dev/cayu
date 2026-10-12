"""Task dispatch contracts."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta

import pytest
from tests.core._execution_profile_fixtures import (
    profiled_session_identity,
    run_request_with_registered_tool_ceiling,
)
from tests.core.task_invocation_fixtures import task_backed_session_invocation
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import (
    _expected_failure_evidence,
    collect_dispatch_events,
    collect_events,
    collect_fork_events,
)

import cayu.runtime.execution_profiles as execution_profiles_module
import cayu.sessions.base as sessions_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.observability.hooks import RuntimeHook, RuntimeHookContext
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._continuation_task_failure import runtime_task_failure_identity_from_task
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY
from cayu.runtime.execution_profiles import EXECUTION_PROFILE_FINGERPRINT_FIELD
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import EventRecord, Session, SessionStatus
from cayu.sessions.requests import ForkSessionRequest, RunRequest
from cayu.tasks.creation import TaskCreate
from cayu.tasks.dispatch import Dispatcher, DispatchHandle, DispatchRequest, DispatchStatus
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskClaimLost, TaskStatus
from cayu.tasks.terminalization import TaskTerminalizationRequest


async def submit_dispatch(app: CayuApp, request: DispatchRequest) -> DispatchHandle:
    return await app.dispatch(request)


def test_cayu_app_dispatch_returns_inline_handle():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("dispatch answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_dispatch_handle",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    handle = asyncio.run(
        submit_dispatch(
            app,
            DispatchRequest(
                session_id="sess_dispatch_handle",
                dispatch_id="dispatch_handle_1",
                messages=[Message.text("user", "run dispatched work")],
            ),
        )
    )

    assert handle == DispatchHandle(
        dispatch_id="dispatch_handle_1",
        session_id="sess_dispatch_handle",
        backend="inline",
        status=DispatchStatus.COMPLETED,
        metadata={"events": 10},
    )
    assert [message.content[0].text for message in provider.requests[1].messages] == [
        "first request",
        "first answer",
        "run dispatched work",
    ]
    session = asyncio.run(store.load("sess_dispatch_handle"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_cayu_app_dispatches_forked_session_with_task_linkage():
    store = InMemorySessionStore()
    tasks = InMemoryTaskStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("fork dispatch answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, task_store=tasks)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_dispatch_fork_source",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    asyncio.run(
        collect_fork_events(
            app,
            ForkSessionRequest(
                source_session_id="sess_dispatch_fork_source",
                session_id="sess_dispatch_fork_child",
            ),
        )
    )
    task = asyncio.run(
        app.create_task(
            TaskCreate(
                type="follow_up",
                session_id="sess_dispatch_fork_child",
                assigned_agent_name="assistant",
                input={"objective": "summarize follow-up"},
            )
        )
    )

    dispatch_events = asyncio.run(
        collect_dispatch_events(
            app,
            DispatchRequest(
                session_id="sess_dispatch_fork_child",
                dispatch_id="dispatch_fork_1",
                task_id=task.id,
                messages=[Message.text("user", "run the forked follow-up")],
            ),
        )
    )

    assert [event.type for event in dispatch_events] == [
        EventType.SESSION_RESUMED,
        EventType.TASK_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TASK_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert dispatch_events[0].payload == {
        "agent_name": "assistant",
        "appended_messages": 1,
        "dispatch_id": PRIVATE_EVENT_AUTHORITY,
        "task_id": PRIVATE_EVENT_AUTHORITY,
        "parent_session_id": PRIVATE_EVENT_AUTHORITY,
        "run_epoch": 1,
    }
    completed_task = asyncio.run(tasks.load_task(task.id))
    assert completed_task is not None
    assert completed_task.status == TaskStatus.COMPLETED
    assert completed_task.session_id == "sess_dispatch_fork_child"
    fork = asyncio.run(store.load("sess_dispatch_fork_child"))
    assert fork is not None
    assert fork.parent_session_id == "sess_dispatch_fork_source"
    assert fork.status == SessionStatus.COMPLETED


def test_cayu_app_dispatch_rejects_running_session():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_running_session() -> None:
        await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id="sess_dispatch_running",
                messages=[Message.text("user", "hi")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=app,
            ),
        )
        await store.update_status("sess_dispatch_running", SessionStatus.RUNNING)

    asyncio.run(create_running_session())

    with pytest.raises(ValueError, match="execution is still in progress"):
        asyncio.run(
            collect_dispatch_events(
                app,
                DispatchRequest(
                    session_id="sess_dispatch_running",
                    messages=[Message.text("user", "continue")],
                ),
            )
        )


def test_cayu_app_dispatch_requires_task_store_for_task_linkage():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_dispatch_no_task_store",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    with pytest.raises(RuntimeError, match="task_store is required"):
        asyncio.run(
            collect_dispatch_events(
                app,
                DispatchRequest(
                    session_id="sess_dispatch_no_task_store",
                    task_id="task_missing_store",
                    messages=[Message.text("user", "continue")],
                ),
            )
        )
    session = asyncio.run(store.load("sess_dispatch_no_task_store"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_cayu_app_dispatch_uses_configured_dispatcher():
    class RecordingDispatcher(Dispatcher):
        def __init__(self) -> None:
            self.requests: list[DispatchRequest] = []

        async def submit(self, runtime, request: DispatchRequest) -> DispatchHandle:
            self.requests.append(request)
            return DispatchHandle(
                dispatch_id=request.dispatch_id,
                session_id=request.session_id,
                backend="recording",
                status=DispatchStatus.SUBMITTED,
                metadata={"queued": True},
            )

    dispatcher = RecordingDispatcher()
    app = CayuApp(dispatcher=dispatcher)

    handle = asyncio.run(
        submit_dispatch(
            app,
            DispatchRequest(
                session_id="sess_custom_dispatcher",
                dispatch_id="dispatch_custom_1",
                messages=[Message.text("user", "queued")],
            ),
        )
    )

    assert handle == DispatchHandle(
        dispatch_id="dispatch_custom_1",
        session_id="sess_custom_dispatcher",
        backend="recording",
        status=DispatchStatus.SUBMITTED,
        metadata={"queued": True},
    )
    assert [request.dispatch_id for request in dispatcher.requests] == ["dispatch_custom_1"]


def test_cayu_app_dispatch_rejects_mismatched_dispatch_handle():
    class MismatchedDispatcher(Dispatcher):
        async def submit(self, runtime, request: DispatchRequest) -> DispatchHandle:
            return DispatchHandle(
                dispatch_id="other_dispatch",
                session_id=request.session_id,
                backend="mismatched",
                status=DispatchStatus.SUBMITTED,
            )

    app = CayuApp(dispatcher=MismatchedDispatcher())

    with pytest.raises(ValueError, match="wrong request fields: dispatch_id"):
        asyncio.run(
            submit_dispatch(
                app,
                DispatchRequest(
                    session_id="sess_custom_dispatcher",
                    dispatch_id="dispatch_custom_1",
                    messages=[Message.text("user", "queued")],
                ),
            )
        )


def test_cayu_app_runtime_hook_can_fork_and_dispatch_followup_work():
    class RecordingDispatcher(Dispatcher):
        def __init__(self) -> None:
            self.requests: list[DispatchRequest] = []

        async def submit(self, runtime, request: DispatchRequest) -> DispatchHandle:
            self.requests.append(request)
            return DispatchHandle(
                dispatch_id=request.dispatch_id,
                session_id=request.session_id,
                task_id=request.task_id,
                backend="recording",
                status=DispatchStatus.SUBMITTED,
                metadata={"queued": True},
            )

    class FollowupHook(RuntimeHook):
        def __init__(self) -> None:
            self.session_status: SessionStatus | None = None
            self.task_id: str | None = None
            self.handle: DispatchHandle | None = None
            self.actions: list[dict] = []

        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            source_session = context.session
            if source_session.metadata.get("purpose") == "knowledge_extraction":
                return
            self.session_status = source_session.status
            child_session_id = f"{source_session.id}_knowledge"
            await context.fork_session(
                ForkSessionRequest(
                    source_session_id=source_session.id,
                    session_id=child_session_id,
                    metadata={"purpose": "knowledge_extraction"},
                )
            )
            task = await context.create_task(
                TaskCreate(
                    type="knowledge_extraction",
                    session_id=child_session_id,
                    assigned_agent_name=source_session.agent_name,
                    input={"source_session_id": source_session.id},
                )
            )
            self.task_id = task.id
            self.handle = await context.dispatch(
                DispatchRequest(
                    session_id=child_session_id,
                    dispatch_id="dispatch_knowledge_1",
                    task_id=task.id,
                    messages=[Message.text("user", "Extract implementation knowledge.")],
                )
            )
            self.actions = context.actions

    store = InMemorySessionStore()
    tasks = InMemoryTaskStore()
    dispatcher = RecordingDispatcher()
    hook = FollowupHook()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("main task done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(
        session_store=store,
        task_store=tasks,
        dispatcher=dispatcher,
        runtime_hooks=[hook],
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="builder", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="builder",
                session_id="sess_hook_source",
                messages=[Message.text("user", "Build the feature.")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
        EventType.HOOK_STARTED,
        EventType.HOOK_COMPLETED,
    ]
    assert hook.session_status == SessionStatus.COMPLETED
    assert hook.handle == DispatchHandle(
        dispatch_id="dispatch_knowledge_1",
        session_id="sess_hook_source_knowledge",
        task_id=hook.task_id,
        backend="recording",
        status=DispatchStatus.SUBMITTED,
        metadata={"queued": True},
    )
    assert [request.session_id for request in dispatcher.requests] == ["sess_hook_source_knowledge"]
    assert [action["type"] for action in hook.actions] == [
        "fork_session",
        "create_task",
        "dispatch",
    ]

    child = asyncio.run(store.load("sess_hook_source_knowledge"))
    assert child is not None
    assert child.parent_session_id == "sess_hook_source"
    assert child.status == SessionStatus.COMPLETED
    assert sessions_module.session_user_metadata(child.metadata) == {
        "purpose": "knowledge_extraction"
    }
    task = asyncio.run(tasks.load_task(hook.task_id))
    assert task is not None
    assert task.status == TaskStatus.PENDING
    assert task.session_id == "sess_hook_source_knowledge"
    assert events[-1].payload["hook_name"] == "FollowupHook"
    assert events[-1].payload["phase"] == "after_session_completed"
    assert [action["type"] for action in events[-1].payload["actions"]] == [
        "fork_session",
        "create_task",
        "dispatch",
    ]


def test_cayu_app_links_successful_run_to_task():
    session_store = InMemorySessionStore()
    task_store = InMemoryTaskStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )

    async def run_task_session() -> tuple[list[Event], object]:
        await task_store.create_task(
            TaskCreate(
                task_id="task_runtime_success",
                type="respond",
                assigned_agent_name="assistant",
            )
        )
        app = CayuApp(session_store=session_store, task_store=task_store)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_task_success",
                task_id="task_runtime_success",
                messages=[Message.text("user", "hi")],
            ),
        )
        task = await task_store.load_task("task_runtime_success")
        return events, task

    events, task = asyncio.run(run_task_session())

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.TASK_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TASK_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert task is not None
    assert task.status == TaskStatus.COMPLETED
    assert task.session_id == "sess_task_success"
    assert task.result == {
        "session_id": "sess_task_success",
        "agent_name": "assistant",
        "environment_name": None,
    }
    assert events[1].payload["task_id"] == PRIVATE_EVENT_AUTHORITY
    assert events[1].payload["task_status"] == "running"
    assert events[5].payload["task_status"] == "completed"
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_cayu_app_links_claimed_task_to_successful_run():
    session_store = InMemorySessionStore()
    task_store = InMemoryTaskStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )

    async def run_task_session() -> tuple[list[Event], object]:
        await task_store.create_task(
            TaskCreate(
                task_id="task_runtime_claimed_success",
                type="respond",
                assigned_agent_name="assistant",
            )
        )
        claimed = await task_store.claim_task("worker_a", lease_seconds=300)
        assert claimed is not None
        assert claimed.worker_id == "worker_a"
        assert claimed.session_id is None

        app = CayuApp(session_store=session_store, task_store=task_store)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_task_claimed_success",
                task_id="task_runtime_claimed_success",
                task_worker_id="worker_a",
                task_lease_expires_at=claimed.lease_expires_at,
                messages=[Message.text("user", "hi")],
            ),
        )
        task = await task_store.load_task("task_runtime_claimed_success")
        return events, task

    events, task = asyncio.run(run_task_session())

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.TASK_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TASK_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert task is not None
    assert task.status == TaskStatus.COMPLETED
    assert task.session_id == "sess_task_claimed_success"
    assert task.worker_id is None
    assert task.lease_expires_at is None
    assert events[1].payload["task_id"] == PRIVATE_EVENT_AUTHORITY


def test_cayu_app_rejects_stale_same_worker_lease_before_provider_dispatch() -> None:
    ownership_time = [datetime(2026, 1, 1, tzinfo=UTC)]
    session_store = InMemorySessionStore()
    task_store = InMemoryTaskStore(ownership_clock=lambda: ownership_time[0])
    provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])

    async def run_task_session() -> None:
        await task_store.create_task(
            TaskCreate(task_id="task_runtime_stale_same_worker", type="respond")
        )
        stale = await task_store.claim_task("shared-worker", lease_seconds=10)
        assert stale is not None
        assert stale.lease_expires_at is not None
        ownership_time[0] += timedelta(seconds=10)
        assert [task.id for task in await task_store.reclaim_expired()] == [stale.id]
        successor = await task_store.claim_task("shared-worker", lease_seconds=300)
        assert successor is not None
        assert successor.lease_expires_at != stale.lease_expires_at

        app = CayuApp(session_store=session_store, task_store=task_store)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_task_stale_same_worker",
                task_id=stale.id,
                task_worker_id="shared-worker",
                task_lease_expires_at=stale.lease_expires_at,
                messages=[Message.text("user", "hi")],
            ),
        )
        assert EventType.TASK_STARTED not in {event.type for event in events}
        assert events[-1].type is EventType.SESSION_FAILED
        assert events[-1].payload["error_type"] == TaskClaimLost.__name__
        assert provider.requests == []
        assert await task_store.load_task(stale.id) == successor

    asyncio.run(run_task_session())


def test_cayu_app_reconciles_claimed_task_completion_acknowledgement_loss():
    class CommitThenRaiseTaskStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.terminalize_calls = 0

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            self.terminalize_calls += 1
            await super().terminalize_task(request)
            raise ConnectionError("acknowledgement lost")

    session_store = InMemorySessionStore()
    task_store = CommitThenRaiseTaskStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )

    async def run_task_session() -> tuple[list[Event], object]:
        await task_store.create_task(TaskCreate(task_id="task_ack_loss", type="respond"))
        claimed = await task_store.claim_task("worker_a", lease_seconds=300)
        assert claimed is not None
        app = CayuApp(session_store=session_store, task_store=task_store)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_task_ack_loss",
                task_id="task_ack_loss",
                task_worker_id="worker_a",
                task_lease_expires_at=claimed.lease_expires_at,
                messages=[Message.text("user", "hi")],
            ),
        )
        return events, await task_store.load_task("task_ack_loss")

    events, task = asyncio.run(run_task_session())

    assert task is not None
    assert task.status is TaskStatus.COMPLETED
    assert task_store.terminalize_calls == 1
    assert sum(event.type is EventType.TASK_COMPLETED for event in events) == 1


def test_cayu_app_fails_task_when_run_fails():
    session_store = InMemorySessionStore()
    task_store = InMemoryTaskStore()
    provider = FakeProvider([ModelStreamEvent.error("provider down")])

    async def run_task_session() -> tuple[list[Event], object, object]:
        await task_store.create_task(TaskCreate(task_id="task_runtime_failure", type="respond"))
        app = CayuApp(session_store=session_store, task_store=task_store)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_task_failure",
                task_id="task_runtime_failure",
                messages=[Message.text("user", "hi")],
            ),
        )
        task = await task_store.load_task("task_runtime_failure")
        session = await session_store.load("sess_task_failure")
        return events, task, session

    events, task, session = asyncio.run(run_task_session())

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.TASK_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.TASK_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert task is not None
    assert task.status == TaskStatus.FAILED
    assert task.session_id == "sess_task_failure"
    assert session is not None
    assert task.error is not None
    task_error = dict(task.error)
    task_error.pop("runtime_task_failure")
    assert task_error == {
        "message": "provider down",
        "type": "RuntimeError",
        "session_id": "sess_task_failure",
    }
    failure_identity = runtime_task_failure_identity_from_task(
        task,
        session_id=session.id,
        session_instance_id=session.instance_id,
    )
    assert failure_identity is not None
    assert session.status == SessionStatus.FAILED
    assert events[-1].interaction_id is None
    assert events[-1].payload == {
        "failure_evidence": _expected_failure_evidence(
            "sess_task_failure", "RuntimeError", "ModelAttemptFailed"
        ),
        "error": "provider down",
        "error_type": "RuntimeError",
        "runtime_task_failure_id": failure_identity.failure_id,
        EXECUTION_PROFILE_FINGERPRINT_FIELD: failure_identity.execution_profile_fingerprint,
    }


def test_cayu_app_reconciles_claimed_task_failure_acknowledgement_loss():
    class CommitThenRaiseTaskStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.terminalize_calls = 0

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            self.terminalize_calls += 1
            await super().terminalize_task(request)
            raise ConnectionError("acknowledgement lost")

    session_store = InMemorySessionStore()
    task_store = CommitThenRaiseTaskStore()
    provider = FakeProvider([ModelStreamEvent.error("provider down")])

    async def run_task_session() -> tuple[list[Event], object]:
        await task_store.create_task(TaskCreate(task_id="task_failure_ack", type="respond"))
        claimed = await task_store.claim_task("worker_a", lease_seconds=300)
        assert claimed is not None
        app = CayuApp(session_store=session_store, task_store=task_store)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_task_failure_ack",
                task_id="task_failure_ack",
                task_worker_id="worker_a",
                task_lease_expires_at=claimed.lease_expires_at,
                messages=[Message.text("user", "hi")],
            ),
        )
        return events, await task_store.load_task("task_failure_ack")

    events, task = asyncio.run(run_task_session())

    assert task is not None
    assert task.status is TaskStatus.FAILED
    assert task_store.terminalize_calls == 1
    assert sum(event.type is EventType.TASK_FAILED for event in events) == 1


def test_cayu_app_live_stream_includes_failed_interaction_terminal_before_turn_completed():
    session_id = "sess_live_failed_interaction"
    store = InMemorySessionStore()
    provider = FakeProvider([ModelStreamEvent.error("provider down")])
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run() -> tuple[list[Event], list[EventRecord]]:
        live = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hi")],
                )
            )
        ]
        return live, await store.query_events(EventQuery(session_id=session_id))

    live, durable_records = asyncio.run(run())
    live_failed = [event for event in live if event.type == EventType.INTERACTION_FAILED]
    projected_failed = [
        app.project_event_record_for_exposure(record).event
        for record in durable_records
        if record.event.type == EventType.INTERACTION_FAILED
    ]

    assert live_failed == projected_failed
    assert len(live_failed) == 1
    live_types = [event.type for event in live]
    assert live_types.index(EventType.INTERACTION_FAILED) < live_types.index(
        EventType.TURN_COMPLETED
    )


def _assert_unattached_task_failure_payload(
    event: Event,
    session: Session,
    expected_diagnostic: dict[str, str],
) -> None:
    terminal_payload = dict(event.payload)
    runtime_task_failure_id = terminal_payload.pop("runtime_task_failure_id")
    execution_profile_fingerprint = terminal_payload.pop(EXECUTION_PROFILE_FINGERPRINT_FIELD)
    assert terminal_payload.pop("failure_evidence") == _expected_failure_evidence(
        session.id, expected_diagnostic["error_type"], run_epoch=1
    )
    assert terminal_payload == expected_diagnostic
    assert runtime_task_failure_id.startswith("runtime-task-failure:v2:")
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        runtime_task_failure_id.removeprefix("runtime-task-failure:v2:"),
    )
    assert execution_profile_fingerprint == (
        execution_profiles_module.execution_profile_from_session_metadata(
            session.metadata
        ).fingerprint
    )


def test_cayu_app_fails_session_clearly_when_task_store_is_missing():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_missing_task_store",
                task_id="task_without_store",
                messages=[Message.text("user", "hi")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_missing_task_store"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert session is not None
    _assert_unattached_task_failure_payload(
        events[-1],
        session,
        {
            "error": "task_store is required when RunRequest.task_id is set.",
            "error_type": "RuntimeError",
        },
    )
    assert session.status == SessionStatus.FAILED
    assert provider.requests == []


def test_cayu_app_does_not_fail_task_it_could_not_start():
    session_store = InMemorySessionStore()
    task_store = InMemoryTaskStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )

    async def run_task_session() -> tuple[list[Event], object, object]:
        await task_store.create_task(TaskCreate(task_id="task_claimed_elsewhere", type="respond"))
        await task_store.start_task(
            "task_claimed_elsewhere",
            session_id="other_session",
            session_invocation=await task_backed_session_invocation(
                task_store,
                "task_claimed_elsewhere",
                "other_session",
            ),
        )
        app = CayuApp(session_store=session_store, task_store=task_store)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_task_claim_conflict",
                task_id="task_claimed_elsewhere",
                messages=[Message.text("user", "hi")],
            ),
        )
        task = await task_store.load_task("task_claimed_elsewhere")
        session = await session_store.load("sess_task_claim_conflict")
        return events, task, session

    events, task, session = asyncio.run(run_task_session())

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert task is not None
    assert task.status == TaskStatus.RUNNING
    assert task.session_id == "other_session"
    assert session is not None
    _assert_unattached_task_failure_payload(
        events[-1],
        session,
        {
            "error": ("Task task_claimed_elsewhere cannot transition to running from running"),
            "error_type": "ValueError",
        },
    )
    assert provider.requests == []
