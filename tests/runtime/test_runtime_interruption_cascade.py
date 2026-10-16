"""Interruption cascade contracts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from tests.core.session_store_conformance.support import (
    _collect_events,
    _SimulatedProcessLoss,
    _UserInputRecoveryProvider,
)
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import _private_record_for_public_event, collect_events

import cayu.runtime._session_control as session_control_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ResolutionActor, ResolutionActorSource
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY
from cayu.runtime._interruption_coordinator import (
    _background_cascade_interruption,
    suppress_interruption_cascade,
)
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.queries import SessionQuery
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import InterruptSessionRequest, RunRequest
from cayu.tools.subagents import (
    SubagentExecutionMode,
    SubagentResultTool,
    SubagentSpec,
    SubagentTool,
)
from cayu.tools.user_input import UserInputTool
from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor


@pytest.mark.parametrize("delay_parent_finalization", [False, True])
def test_interrupting_parent_interrupts_running_background_subagents(
    monkeypatch,
    delay_parent_finalization,
):
    active_wait_attempts = session_control_module.ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS
    active_wait_interval_s = session_control_module.ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S
    if delay_parent_finalization:
        monkeypatch.setattr(session_control_module, "ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS", 2)
        monkeypatch.setattr(session_control_module, "ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S", 0)

    class BackgroundInterruptProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.child_model_started = asyncio.Event()
            self.parent_second_step_started = asyncio.Event()
            self.release_parent_after_cancel = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            first_text = request.messages[0].content[0].text
            if first_text == "parent task" and len(request.messages) == 1:
                yield ModelStreamEvent.tool_call(
                    id="call_background_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "background review task",
                    },
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            if first_text == "background review task":
                self.child_model_started.set()
                await asyncio.Event().wait()
                return
            if first_text == "parent task" and request.messages[-1].role == "tool":
                self.parent_second_step_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    if not delay_parent_finalization:
                        raise
                    await self.release_parent_after_cancel.wait()
                    yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return
            raise AssertionError("Unexpected background interrupt provider request.")

    async def run():
        store = InMemorySessionStore()
        provider = BackgroundInterruptProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="fake-model"),
            tools=[
                SubagentTool(
                    app,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            mode=SubagentExecutionMode.BACKGROUND,
                        )
                    },
                ),
                SubagentResultTool(store),
            ],
        )
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

        parent_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="parent",
                    session_id="sess_background_parent_interrupt",
                    messages=[Message.text("user", "parent task")],
                ),
            )
        )
        await asyncio.wait_for(provider.child_model_started.wait(), timeout=10)
        await asyncio.wait_for(provider.parent_second_step_started.wait(), timeout=10)
        interrupt_request = InterruptSessionRequest(
            session_id="sess_background_parent_interrupt",
            reason="stop parent and background children",
            requested_by=ResolutionActor(
                subject="operator-42",
                source=ResolutionActorSource.REQUEST,
            ),
        )
        if delay_parent_finalization:
            with pytest.raises(TimeoutError, match="interruption is still finalizing"):
                _ = [event async for event in app.interrupt_session(interrupt_request)]
            children_before_parent_release = (
                await store.list_sessions(
                    SessionQuery(parent_session_id="sess_background_parent_interrupt")
                )
            ).sessions
            assert children_before_parent_release[0].status == SessionStatus.INTERRUPTING
            monkeypatch.setattr(
                session_control_module,
                "ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS",
                active_wait_attempts,
            )
            monkeypatch.setattr(
                session_control_module,
                "ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S",
                active_wait_interval_s,
            )
            provider.release_parent_after_cancel.set()
            interrupt_events = []
        else:
            interrupt_events = [event async for event in app.interrupt_session(interrupt_request)]
        parent_events = await asyncio.wait_for(parent_task, timeout=10)
        assert await app.drain_background_interruptions(timeout_s=1) is True
        child_sessions = (
            await store.list_sessions(
                SessionQuery(parent_session_id="sess_background_parent_interrupt")
            )
        ).sessions
        child_events = await store.load_events(child_sessions[0].id)
        return interrupt_events, parent_events, child_sessions, child_events

    interrupt_events, parent_events, child_sessions, child_events = asyncio.run(run())

    if delay_parent_finalization:
        assert interrupt_events == []
    else:
        assert interrupt_events[-1].type == EventType.SESSION_INTERRUPTED
    assert parent_events[-1].type == EventType.SESSION_INTERRUPTED
    assert len(child_sessions) == 1
    assert child_sessions[0].status == SessionStatus.INTERRUPTED
    child_interrupted = [
        event for event in child_events if event.type == EventType.SESSION_INTERRUPTED
    ]
    assert len(child_interrupted) == 1
    assert child_interrupted[0].payload["metadata"]["source"] == (
        "background_subagent_parent_interrupt"
    )
    assert child_interrupted[0].payload["interruption_origin"] == "background_cascade"
    assert "interruption_origin" not in parent_events[-1].payload
    assert child_interrupted[0].payload["requested_by"] == {
        "subject": "operator-42",
        "tenant": None,
        "source": "request",
    }


def test_retried_parent_interruption_cascades_original_durable_provenance():
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor("-"),
            enable_logging=False,
        )
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="parent", model="fakemodel"))
        app.register_agent(AgentSpec(name="reviewer", model="fakemodel"))
        identity = SessionIdentity(provider_name="fake", model="fakemodel")

        await store.create(
            RunRequest(
                agent_name="parent",
                session_id="sess_interrupted_parent_retry",
                messages=[Message.text("user", "parent task")],
            ),
            identity=identity,
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_interrupted_parent_retry_child",
                parent_session_id="sess_interrupted_parent_retry",
                messages=[Message.text("user", "background task")],
                metadata={"subagent": {"mode": "background"}},
            ),
            identity=identity,
        )
        await store.update_status(
            "sess_interrupted_parent_retry",
            SessionStatus.INTERRUPTED,
        )
        durable_payload = {
            "reason": "original operator stop",
            "metadata": {"ticket": "incident-a"},
            "requested_by": {
                "subject": "operator-a",
                "tenant": "tenant-7",
                "source": "http_auth",
            },
            "interruption_type": "operator_requested",
        }
        await store.append_event(
            "sess_interrupted_parent_retry",
            Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id="sess_interrupted_parent_retry",
                agent_name="parent",
                payload=durable_payload,
            ),
        )
        await store.checkpoint(
            "sess_interrupted_parent_retry",
            {
                "pending_interruption_cascade": {
                    "attempt_id": "retry-durable-provenance",
                    "interrupt_payload": durable_payload,
                    "created_at": datetime.now(UTC).isoformat(),
                    "failure_recorded": True,
                }
            },
        )

        repeated_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_interrupted_parent_retry",
                    reason="retry from another operator",
                    metadata={"ticket": "incident-b"},
                    requested_by=ResolutionActor(
                        subject="operator-b",
                        tenant="tenant-7",
                        source=ResolutionActorSource.HTTP_AUTH,
                    ),
                )
            )
        ]
        assert await app.drain_background_interruptions(timeout_s=1) is True
        child_events = await store.load_events("sess_interrupted_parent_retry_child")
        parent_records = await store.query_events(
            EventQuery(session_id="sess_interrupted_parent_retry")
        )
        return repeated_events, child_events, parent_records

    repeated_events, child_events, parent_records = asyncio.run(run())

    assert repeated_events[0].payload["requested_by"]["subject"] == (f"operator{REDACTED_SECRET}a")
    assert repeated_events[1].type == EventType.SESSION_INTERRUPTION_CASCADE_RETRY_REQUESTED
    assert repeated_events[1].payload["retry_request_id"] == PRIVATE_EVENT_AUTHORITY
    retry_request_id = _private_record_for_public_event(
        parent_records,
        repeated_events[1],
    ).event.payload["retry_request_id"]
    assert UUID(retry_request_id)
    assert repeated_events[1].payload["retry_reason"] == "retry from another operator"
    assert repeated_events[1].payload["retry_metadata"] == {"ticket": f"incident{REDACTED_SECRET}b"}
    assert repeated_events[1].payload["retry_requested_by"]["subject"] == (
        f"operator{REDACTED_SECRET}b"
    )
    child_interrupted = [
        event for event in child_events if event.type == EventType.SESSION_INTERRUPTED
    ]
    assert len(child_interrupted) == 1
    assert child_interrupted[0].payload["interruption_origin"] == "background_cascade"
    assert child_interrupted[0].payload["reason"] == "original operator stop"
    assert child_interrupted[0].payload["metadata"]["parent_metadata"] == {
        "ticket": f"incident{REDACTED_SECRET}a"
    }
    assert child_interrupted[0].payload["requested_by"] == {
        "subject": f"operator{REDACTED_SECRET}a",
        "tenant": f"tenant{REDACTED_SECRET}7",
        "source": "http_auth",
    }
    completed = next(
        record.event
        for record in parent_records
        if record.event.type == EventType.SESSION_INTERRUPTION_CASCADE_COMPLETED
    )
    assert completed.payload["retry_request_id"] == retry_request_id
    assert completed.payload["retry_reason"] == "retry from another operator"
    assert completed.payload["retry_requested_by"]["subject"] == (f"operator{REDACTED_SECRET}b")


def test_interrupted_session_cascade_retry_uses_durable_identity_when_agent_is_unregistered():
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        session_id = "sess_retry_removed_agent"
        durable_payload = {
            "reason": "original operator stop",
            "metadata": {},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        await store.create(
            RunRequest(
                agent_name="removed-agent",
                session_id=session_id,
                messages=[Message.text("user", "historical work")],
            ),
            identity=SessionIdentity(provider_name="removed-provider", model="old-model"),
        )
        await store.update_status(session_id, SessionStatus.INTERRUPTED)
        interrupted = Event(
            type=EventType.SESSION_INTERRUPTED,
            session_id=session_id,
            agent_name="removed-agent",
            payload=durable_payload,
        )
        await store.append_event(session_id, interrupted)
        await store.checkpoint(
            session_id,
            {
                "pending_interruption_cascade": {
                    "attempt_id": "removed-agent-retry",
                    "interrupt_payload": durable_payload,
                    "created_at": datetime.now(UTC).isoformat(),
                    "failure_recorded": True,
                }
            },
        )
        retry_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="retry after deployment",
                    requested_by=ResolutionActor(
                        subject="operator-b",
                        source=ResolutionActorSource.REQUEST,
                    ),
                )
            )
        ]
        assert await app.drain_background_interruptions(timeout_s=1) is True
        return retry_events, await store.query_events(EventQuery(session_id=session_id))

    retry_events, stored_records = asyncio.run(run())

    assert [event.type for event in retry_events] == [
        EventType.SESSION_INTERRUPTED,
        EventType.SESSION_INTERRUPTION_CASCADE_RETRY_REQUESTED,
    ]
    assert retry_events[1].agent_name == "removed-agent"
    assert retry_events[1].payload["retry_requested_by"]["subject"] == "operator-b"
    assert retry_events[1].payload["retry_request_id"] == PRIVATE_EVENT_AUTHORITY
    durable_retry = _private_record_for_public_event(stored_records, retry_events[1]).event
    completed = next(
        record.event
        for record in stored_records
        if record.event.type == EventType.SESSION_INTERRUPTION_CASCADE_COMPLETED
    )
    assert completed.agent_name == "removed-agent"
    assert completed.payload["retry_request_id"] == durable_retry.payload["retry_request_id"]


def test_finalizing_background_child_does_not_fail_parent_cascade(monkeypatch):
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id="sess_parent_with_finalizing_child",
                messages=[Message.text("user", "parent task")],
            ),
            identity=identity,
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_finalizing_background_child",
                parent_session_id="sess_parent_with_finalizing_child",
                messages=[Message.text("user", "background task")],
                metadata={"subagent": {"mode": "background"}},
            ),
            identity=identity,
        )
        captured_requests = []

        async def finalizing_interrupt_session(request):
            captured_requests.append(request)
            raise TimeoutError(f"Session interruption is still finalizing: {request.session_id}")
            yield

        monkeypatch.setattr(
            app._background_interruption_coordinator,
            "_interrupt_session",
            finalizing_interrupt_session,
        )
        await app._background_interruption_coordinator.run_cascade(
            parent_session_id="sess_parent_with_finalizing_child",
            interrupt_payload={
                "reason": "operator stop",
                "metadata": {},
                "requested_by": {
                    "subject": "operator-a",
                    "tenant": None,
                    "source": "http_auth",
                },
                "interruption_type": "operator_requested",
            },
        )
        return captured_requests

    captured_requests = asyncio.run(run())

    assert len(captured_requests) == 1
    assert captured_requests[0].session_id == "sess_finalizing_background_child"
    assert captured_requests[0].requested_by.subject == "operator-a"


def test_background_interruption_reconciles_child_already_interrupting_elsewhere():
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="parent", model="fake-model"))
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        parent_id = "sess_reconcile_interrupting_parent"
        child_id = "sess_reconcile_interrupting_child"
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_id,
                messages=[Message.text("user", "parent")],
            ),
            identity=identity,
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id=child_id,
                parent_session_id=parent_id,
                messages=[Message.text("user", "child")],
                metadata={"subagent": {"mode": "background"}},
            ),
            identity=identity,
        )
        await store.update_status(child_id, SessionStatus.INTERRUPTING)

        async def finish_other_owner():
            await asyncio.sleep(0.01)
            await store.update_status(child_id, SessionStatus.INTERRUPTED)
            await store.append_event(
                child_id,
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=child_id,
                    agent_name="reviewer",
                    payload={"interruption_type": "operator_requested"},
                ),
            )

        other_owner = asyncio.create_task(finish_other_owner())
        await app._background_interruption_coordinator.run_cascade(
            parent_session_id=parent_id,
            interrupt_payload={
                "reason": "operator stop",
                "metadata": {},
                "requested_by": None,
                "interruption_type": "operator_requested",
            },
        )
        await other_owner
        return (
            await store.load(child_id),
            await store.load_checkpoint(parent_id),
            await store.load_events(parent_id),
        )

    child, parent_checkpoint, parent_events = asyncio.run(run())

    assert child.status == SessionStatus.INTERRUPTED
    assert all(
        event.type != EventType.SESSION_INTERRUPTION_CASCADE_FAILED for event in parent_events
    )
    assert parent_checkpoint == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}


def test_background_interruption_cascade_isolates_child_completion_race(monkeypatch):
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="parent", model="fake-model"))
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id="sess_parent_child_completion_race",
                messages=[Message.text("user", "parent task")],
            ),
            identity=identity,
        )
        for child_id in ("sess_child_completes", "sess_child_interrupts"):
            await store.create(
                RunRequest(
                    agent_name="reviewer",
                    session_id=child_id,
                    parent_session_id="sess_parent_child_completion_race",
                    messages=[Message.text("user", "background task")],
                    metadata={"subagent": {"mode": "background"}},
                ),
                identity=identity,
            )

        original_interrupt_session = app.interrupt_session

        async def racing_interrupt_session(request):
            if request.session_id == "sess_child_completes":
                await store.update_status(request.session_id, SessionStatus.COMPLETED)
            async for event in original_interrupt_session(request):
                yield event

        monkeypatch.setattr(
            app._background_interruption_coordinator, "_interrupt_session", racing_interrupt_session
        )
        await app._background_interruption_coordinator.run_cascade(
            parent_session_id="sess_parent_child_completion_race",
            interrupt_payload={
                "reason": "operator stop",
                "metadata": {},
                "requested_by": None,
                "interruption_type": "operator_requested",
            },
        )
        return (
            await store.load("sess_child_completes"),
            await store.load("sess_child_interrupts"),
        )

    completed_child, interrupted_child = asyncio.run(run())

    assert completed_child.status == SessionStatus.COMPLETED
    assert interrupted_child.status == SessionStatus.INTERRUPTED


def test_background_interruption_cascade_persists_partial_failure(monkeypatch):
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        parent_session_id = "sess_parent_partial_interrupt_failure"
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_session_id,
                messages=[Message.text("user", "parent task")],
            ),
            identity=identity,
        )
        for child_id in ("sess_child_interrupt_fails", "sess_child_interrupt_succeeds"):
            await store.create(
                RunRequest(
                    agent_name="reviewer",
                    session_id=child_id,
                    parent_session_id=parent_session_id,
                    messages=[Message.text("user", "background task")],
                    metadata={"subagent": {"mode": "background"}},
                ),
                identity=identity,
            )

        captured_child_ids = []

        async def partially_failing_interrupt_session(request):
            captured_child_ids.append(request.session_id)
            if request.session_id == "sess_child_interrupt_fails":
                raise RuntimeError("sensitive provider detail")
            yield Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id=request.session_id,
                agent_name="reviewer",
            )

        monkeypatch.setattr(
            app._background_interruption_coordinator,
            "_interrupt_session",
            partially_failing_interrupt_session,
        )
        await app._background_interruption_coordinator.run_cascade(
            parent_session_id=parent_session_id,
            interrupt_payload={
                "reason": "operator stop",
                "metadata": {},
                "requested_by": None,
                "interruption_type": "operator_requested",
            },
        )
        return captured_child_ids, await store.load_events(parent_session_id)

    captured_child_ids, parent_events = asyncio.run(run())

    assert set(captured_child_ids) == {
        "sess_child_interrupt_fails",
        "sess_child_interrupt_succeeds",
    }
    failure_event = parent_events[-1]
    assert failure_event.type == EventType.SESSION_INTERRUPTION_CASCADE_FAILED
    assert UUID(failure_event.payload["attempt_id"])
    assert {key: value for key, value in failure_event.payload.items() if key != "attempt_id"} == {
        "interruption_type": "operator_requested",
        "generation": 1,
        "failure_count": 1,
        "failures": [
            {
                "scope": "child",
                "session_id": "sess_child_interrupt_fails",
                "status": "pending",
                "reason": "interruption_request_failed",
                "error_type": "RuntimeError",
            }
        ],
        "failures_truncated": False,
    }
    assert "sensitive provider detail" not in str(failure_event.payload)


def test_background_interruption_cascade_success_resolves_prior_failure(monkeypatch):
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        parent_session_id = "sess_parent_cascade_retry"
        child_session_id = "sess_child_cascade_retry"
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_session_id,
                messages=[Message.text("user", "parent task")],
            ),
            identity=identity,
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id=child_session_id,
                parent_session_id=parent_session_id,
                messages=[Message.text("user", "background task")],
                metadata={"subagent": {"mode": "background"}},
            ),
            identity=identity,
        )
        attempts = 0

        async def retrying_interrupt_session(request):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary failure")
            yield Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id=request.session_id,
                agent_name="reviewer",
            )

        payload = {
            "reason": "operator stop",
            "metadata": {},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        monkeypatch.setattr(
            app._background_interruption_coordinator,
            "_interrupt_session",
            retrying_interrupt_session,
        )
        await app._background_interruption_coordinator.run_cascade(
            parent_session_id=parent_session_id,
            interrupt_payload=payload,
        )
        failed_checkpoint = await store.load_checkpoint(parent_session_id)
        await app._background_interruption_coordinator.run_cascade(
            parent_session_id=parent_session_id,
            interrupt_payload=payload,
        )
        return (
            failed_checkpoint,
            await store.load_checkpoint(parent_session_id),
            await store.load_events(parent_session_id),
        )

    failed_checkpoint, completed_checkpoint, parent_events = asyncio.run(run())

    assert "pending_interruption_cascade" in failed_checkpoint
    assert "pending_interruption_cascade" not in completed_checkpoint
    assert [event.type for event in parent_events] == [
        EventType.SESSION_INTERRUPTION_CASCADE_FAILED,
        EventType.SESSION_INTERRUPTION_CASCADE_COMPLETED,
    ]
    assert parent_events[0].payload["attempt_id"] == parent_events[1].payload["attempt_id"]
    assert [event.payload["generation"] for event in parent_events] == [1, 2]


def test_background_interruption_cascade_retains_marker_while_child_is_interrupting(
    monkeypatch,
):
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        parent_id = "sess_parent_child_still_interrupting"
        child_id = "sess_child_still_interrupting"
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_id,
                messages=[Message.text("user", "parent")],
            ),
            identity=identity,
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id=child_id,
                parent_session_id=parent_id,
                messages=[Message.text("user", "child")],
                metadata={"subagent": {"mode": "background"}},
            ),
            identity=identity,
        )

        async def still_finalizing_interrupt(request):
            await store.update_status(request.session_id, SessionStatus.INTERRUPTING)
            raise TimeoutError("still finalizing")
            yield

        payload = {
            "reason": "operator stop",
            "metadata": {},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        monkeypatch.setattr(
            app._background_interruption_coordinator,
            "_interrupt_session",
            still_finalizing_interrupt,
        )
        await app._background_interruption_coordinator.run_cascade(
            parent_session_id=parent_id,
            interrupt_payload=payload,
        )
        failed_checkpoint = await store.load_checkpoint(parent_id)
        failed_events = await store.load_events(parent_id)

        await store.update_status(child_id, SessionStatus.INTERRUPTED)
        await app._background_interruption_coordinator.run_cascade(
            parent_session_id=parent_id,
            interrupt_payload=payload,
        )
        return (
            failed_checkpoint,
            await store.load_checkpoint(parent_id),
            failed_events,
            await store.load_events(parent_id),
        )

    failed_checkpoint, completed_checkpoint, failed_events, completed_events = asyncio.run(run())

    assert failed_checkpoint["pending_interruption_cascade"]["failure_recorded"] is True
    assert "pending_interruption_cascade" not in completed_checkpoint
    assert failed_events[-1].type == EventType.SESSION_INTERRUPTION_CASCADE_FAILED
    assert failed_events[-1].payload["failures"][0]["status"] == "interrupting"
    assert completed_events[-1].type == EventType.SESSION_INTERRUPTION_CASCADE_COMPLETED


def test_interrupt_waits_for_actual_terminal_event_before_checkpoint_cleanup(monkeypatch):
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_interrupt_terminal_prefix",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

        cascade_payloads = []
        checkpoint_before_terminal = []

        async def capture_cascade(*, parent_session_id, interrupt_payload, create_if_missing=False):
            assert parent_session_id == "sess_interrupt_terminal_prefix"
            assert create_if_missing is False
            cascade_payloads.append(interrupt_payload)

        async def terminal_stream(**kwargs):
            yield await app._event_writer.emit(
                Event(
                    type=EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
                    session_id="sess_interrupt_terminal_prefix",
                    agent_name="assistant",
                    payload={"outcome": "interrupted"},
                )
            )
            checkpoint_before_terminal.append(
                await store.load_checkpoint("sess_interrupt_terminal_prefix")
            )
            yield await app._event_writer.emit(kwargs["event"])

        monkeypatch.setattr(
            app._background_interruption_coordinator, "run_cascade", capture_cascade
        )
        monkeypatch.setattr(
            app._terminal_event_publication,
            "emit",
            terminal_stream,
        )
        events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_interrupt_terminal_prefix",
                    reason="operator stop",
                )
            )
        ]
        return (
            events,
            cascade_payloads,
            checkpoint_before_terminal,
            await store.load_checkpoint("sess_interrupt_terminal_prefix"),
        )

    events, cascade_payloads, checkpoint_before_terminal, final_checkpoint = asyncio.run(run())

    assert [event.type for event in events] == [
        EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert checkpoint_before_terminal[0]["pending_session_interrupt"]["reason"] == "operator stop"
    assert "pending_session_interrupt" not in final_checkpoint
    assert "pending_interruption_cascade" in final_checkpoint
    assert len(cascade_payloads) == 1
    assert all(payload["interruption_type"] == "operator_requested" for payload in cascade_payloads)


def test_interrupt_stream_does_not_wait_for_background_cascade(monkeypatch):
    async def run():
        store = InMemorySessionStore()
        fixed_now = datetime(2026, 2, 3, 4, 5, 6, tzinfo=UTC)
        app = CayuApp(session_store=store, enable_logging=False, clock=lambda: fixed_now)
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_interrupt_cancelled_request_cascade",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        cascade_started = asyncio.Event()
        release_cascade = asyncio.Event()

        async def delayed_cascade(**_kwargs):
            cascade_started.set()
            await release_cascade.wait()

        monkeypatch.setattr(
            app._background_interruption_coordinator, "run_cascade", delayed_cascade
        )
        event_stream = app.interrupt_session(
            InterruptSessionRequest(
                session_id="sess_interrupt_cancelled_request_cascade",
                reason="operator stop",
            )
        )
        request_task = asyncio.create_task(anext(event_stream))
        await asyncio.wait_for(cascade_started.wait(), timeout=10)
        interrupted_event = await asyncio.wait_for(request_task, timeout=1)
        assert interrupted_event.type == EventType.SESSION_INTERRUPTED
        assert len(app._background_interruption_coordinator._tasks) == 1
        await event_stream.aclose()
        assert len(app._background_interruption_coordinator._tasks) == 1
        checkpoint = await store.load_checkpoint("sess_interrupt_cancelled_request_cascade")
        release_cascade.set()
        drained = await app.drain_background_interruptions(timeout_s=1)
        return drained, app._background_interruption_coordinator._tasks, checkpoint, fixed_now

    drained, remaining_tasks, checkpoint, fixed_now = asyncio.run(run())

    assert drained is True
    assert remaining_tasks == set()
    assert checkpoint["pending_interruption_cascade"]["created_at"] == fixed_now.isoformat()


def test_interrupt_does_not_start_or_clear_cascade_before_terminal_event(monkeypatch):
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_cascade_after_terminal"
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        terminal_started = asyncio.Event()
        release_terminal = asyncio.Event()

        async def delayed_terminal_stream(**kwargs):
            terminal_started.set()
            await release_terminal.wait()
            yield await app._event_writer.emit(kwargs["event"])

        monkeypatch.setattr(
            app._terminal_event_publication,
            "emit",
            delayed_terminal_stream,
        )
        stream = app.interrupt_session(
            InterruptSessionRequest(session_id=session_id, reason="operator stop")
        )
        first_event_task = asyncio.create_task(anext(stream))
        await asyncio.wait_for(terminal_started.wait(), timeout=10)
        checkpoint_before_terminal = await store.load_checkpoint(session_id)
        tasks_before_terminal = len(app._background_interruption_coordinator._tasks)

        release_terminal.set()
        first_event = await asyncio.wait_for(first_event_task, timeout=1)
        await stream.aclose()
        assert await app.drain_background_interruptions(timeout_s=1) is True
        return (
            first_event,
            checkpoint_before_terminal,
            tasks_before_terminal,
            await store.load_checkpoint(session_id),
        )

    first_event, checkpoint_before_terminal, tasks_before_terminal, final_checkpoint = asyncio.run(
        run()
    )

    assert first_event.type == EventType.SESSION_INTERRUPTED
    assert "pending_interruption_cascade" in checkpoint_before_terminal
    assert tasks_before_terminal == 0
    assert final_checkpoint == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}


def test_pending_interruption_cascade_resumes_from_durable_checkpoint():
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="parent", model="fake-model"))
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        parent_id = "sess_durable_cascade_parent"
        child_id = "sess_durable_cascade_child"
        payload = {
            "reason": "operator stop",
            "metadata": {"ticket": "incident-7"},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_id,
                messages=[Message.text("user", "parent task")],
            ),
            identity=identity,
        )
        await store.update_status(parent_id, SessionStatus.INTERRUPTED)
        await store.checkpoint(
            parent_id,
            {
                "pending_interruption_cascade": {
                    "attempt_id": "cascade-attempt-7",
                    "interrupt_payload": payload,
                }
            },
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id=child_id,
                parent_session_id=parent_id,
                messages=[Message.text("user", "background task")],
                metadata={"subagent": {"mode": "background"}},
            ),
            identity=identity,
        )

        scheduled = await app.resume_pending_interruption_cascades(
            interrupting_inactive_for_seconds=0
        )
        drained = await app.drain_background_interruptions(timeout_s=1)
        return (
            scheduled,
            drained,
            await store.load(child_id),
            await store.load_checkpoint(parent_id),
            await store.load_events(parent_id),
        )

    scheduled, drained, child, checkpoint, parent_events = asyncio.run(run())

    assert scheduled == 1
    assert drained is True
    assert child.status == SessionStatus.INTERRUPTED
    assert "pending_interruption_cascade" not in checkpoint
    assert parent_events == []


def test_pending_interruption_recovery_discovers_only_indexed_markers(monkeypatch):
    class RecoveryDiscoveryStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.discovery_queries: list[SessionQuery] = []
            self.loaded_checkpoint_ids: list[str] = []

        async def list_sessions(self, query=None):
            raise AssertionError("Recovery must not scan the general session history.")

        async def list_sessions_with_pending_interruption_cascade(self, query=None):
            copied_query = SessionQuery() if query is None else query.model_copy(deep=True)
            self.discovery_queries.append(copied_query)
            return await super().list_sessions_with_pending_interruption_cascade(query)

        async def load_checkpoint(self, session_id):
            self.loaded_checkpoint_ids.append(session_id)
            return await super().load_checkpoint(session_id)

    async def run():
        store = RecoveryDiscoveryStore()
        app = CayuApp(session_store=store, enable_logging=False)
        payload = {
            "reason": "operator stop",
            "metadata": {},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        for index in range(5):
            session_id = f"sess_historical_interrupted_{index}"
            await store.create(
                RunRequest(
                    agent_name="parent",
                    session_id=session_id,
                    messages=[Message.text("user", "historical")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await store.update_status(session_id, SessionStatus.INTERRUPTED)
        parent_id = "sess_indexed_interruption_cascade"
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_id,
                messages=[Message.text("user", "parent")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(parent_id, SessionStatus.INTERRUPTED)
        await store.checkpoint(
            parent_id,
            {
                "pending_interruption_cascade": {
                    "attempt_id": "indexed-recovery-attempt",
                    "interrupt_payload": payload,
                }
            },
        )
        scheduled_parent_ids: list[str] = []

        def record_schedule(**kwargs):
            scheduled_parent_ids.append(kwargs["parent_session_id"])
            return None

        monkeypatch.setattr(
            app._session_finalization,
            "schedule_background_interruption_cascade",
            record_schedule,
        )
        scheduled = await app.resume_pending_interruption_cascades(
            interrupting_inactive_for_seconds=300
        )
        return (
            scheduled,
            scheduled_parent_ids,
            store.discovery_queries,
            store.loaded_checkpoint_ids,
        )

    scheduled, scheduled_parent_ids, discovery_queries, loaded_checkpoint_ids = asyncio.run(run())

    assert scheduled == 1
    assert scheduled_parent_ids == ["sess_indexed_interruption_cascade"]
    assert [query.status for query in discovery_queries] == [
        SessionStatus.INTERRUPTING,
        SessionStatus.INTERRUPTED,
    ]
    assert discovery_queries[0].inactive_for_seconds == 300
    assert discovery_queries[1].inactive_for_seconds is None
    assert loaded_checkpoint_ids == ["sess_indexed_interruption_cascade"]


def test_pending_interruption_cascade_finalizes_interrupting_parent_after_restart():
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="parent", model="fake-model"))
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        parent_id = "sess_restart_interrupting_parent"
        child_id = "sess_restart_interrupting_child"
        payload = {
            "reason": "operator stop",
            "metadata": {"ticket": "incident-8"},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_id,
                messages=[Message.text("user", "parent task")],
            ),
            identity=identity,
        )
        await store.update_status(parent_id, SessionStatus.INTERRUPTING)
        await store.checkpoint(
            parent_id,
            {
                "pending_session_interrupt": payload,
                "pending_interruption_cascade": {
                    "attempt_id": "cascade-attempt-restart",
                    "interrupt_payload": payload,
                },
            },
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id=child_id,
                parent_session_id=parent_id,
                messages=[Message.text("user", "background task")],
                metadata={"subagent": {"mode": "background"}},
            ),
            identity=identity,
        )

        scheduled = await app.resume_pending_interruption_cascades(
            interrupting_inactive_for_seconds=0
        )
        drained = await app.drain_background_interruptions(timeout_s=1)
        return (
            scheduled,
            drained,
            await store.load(parent_id),
            await store.load(child_id),
            await store.load_checkpoint(parent_id),
            await store.load_events(parent_id),
        )

    scheduled, drained, parent, child, checkpoint, parent_events = asyncio.run(run())

    assert scheduled == 1
    assert drained is True
    assert parent.status == SessionStatus.INTERRUPTED
    assert child.status == SessionStatus.INTERRUPTED
    assert "pending_session_interrupt" not in checkpoint
    assert "pending_interruption_cascade" not in checkpoint
    assert [event.type for event in parent_events] == [
        EventType.SESSION_RUN_FENCED,
        EventType.SESSION_INTERRUPTED,
    ]


def test_pending_interruption_restart_admits_parent_once_across_status_scans(monkeypatch):
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="parent", model="fake-model"))
        parent_id = "sess_restart_single_admission"
        payload = {
            "reason": "operator stop",
            "metadata": {},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_id,
                messages=[Message.text("user", "parent task")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(parent_id, SessionStatus.INTERRUPTING)
        await store.checkpoint(
            parent_id,
            {
                "pending_session_interrupt": payload,
                "pending_interruption_cascade": {
                    "attempt_id": "cascade-attempt-single-admission",
                    "interrupt_payload": payload,
                },
            },
        )
        attempts = 0

        async def record_attempt(**_kwargs):
            nonlocal attempts
            attempts += 1

        monkeypatch.setattr(app._background_interruption_coordinator, "run_cascade", record_attempt)
        scheduled = await app.resume_pending_interruption_cascades(
            interrupting_inactive_for_seconds=0
        )
        assert await app.drain_background_interruptions(timeout_s=1) is True
        return scheduled, attempts

    scheduled, attempts = asyncio.run(run())

    assert scheduled == 1
    assert attempts == 1


def test_pending_interruption_restart_does_not_fence_fresh_remote_owner():
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="parent", model="fake-model"))
        parent_id = "sess_restart_fresh_remote_owner"
        payload = {
            "reason": "operator stop",
            "metadata": {},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_id,
                messages=[Message.text("user", "parent task")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(parent_id, SessionStatus.INTERRUPTING)
        await store.checkpoint(
            parent_id,
            {
                "pending_session_interrupt": payload,
                "pending_interruption_cascade": {
                    "attempt_id": "cascade-attempt-fresh-owner",
                    "interrupt_payload": payload,
                },
            },
        )

        scheduled = await app.resume_pending_interruption_cascades(
            interrupting_inactive_for_seconds=1
        )
        return (
            scheduled,
            await store.load(parent_id),
            await store.load_checkpoint(parent_id),
            await store.load_events(parent_id),
        )

    scheduled, parent, checkpoint, events = asyncio.run(run())

    assert scheduled == 0
    assert parent.status == SessionStatus.INTERRUPTING
    assert "pending_interruption_cascade" in checkpoint
    assert events == []


def test_background_interruption_does_not_cross_fork_lineage():
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        parent_id = "sess_cascade_fork_source"
        fork_id = "sess_cascade_independent_fork"
        fork_background_id = "sess_cascade_fork_background"
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_id,
                messages=[Message.text("user", "source")],
            ),
            identity=identity,
        )
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=fork_id,
                parent_session_id=parent_id,
                messages=[Message.text("user", "independent fork")],
            ),
            identity=identity,
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id=fork_background_id,
                parent_session_id=fork_id,
                messages=[Message.text("user", "fork background")],
                metadata={"subagent": {"mode": "background"}},
            ),
            identity=identity,
        )

        await app._background_interruption_coordinator.run_cascade(
            parent_session_id=parent_id,
            interrupt_payload={
                "reason": "operator stop",
                "metadata": {},
                "requested_by": None,
                "interruption_type": "operator_requested",
            },
        )
        return await store.load(fork_background_id)

    fork_background = asyncio.run(run())

    assert fork_background.status == SessionStatus.PENDING


def test_interruption_cascade_generation_rejects_stale_worker_results():
    async def run():
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        store = InMemorySessionStore(ownership_clock=lambda: now[0])
        app = CayuApp(session_store=store, enable_logging=False, clock=lambda: now[0])
        session_id = "sess_cascade_generation_claim"
        payload = {
            "reason": "operator stop",
            "metadata": {},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=session_id,
                messages=[Message.text("user", "parent")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.checkpoint(
            session_id,
            {
                "pending_interruption_cascade": {
                    "attempt_id": "cascade-generation-attempt",
                    "interrupt_payload": payload,
                }
            },
        )

        coordinator = app._background_interruption_coordinator
        first = await coordinator.claim_pending_interruption_cascade(session_id, payload)
        first_claim_status = await app.interruption_cascade_status(session_id)
        overlapping = await coordinator.claim_pending_interruption_cascade(session_id, payload)
        now[0] += timedelta(seconds=20)
        renewed = await coordinator.renew_pending_interruption_cascade_claim(
            session_id,
            first["attempt_id"],
            first["generation"],
            first["claim_id"],
        )
        now[0] += timedelta(seconds=15)
        protected_by_heartbeat = await coordinator.claim_pending_interruption_cascade(
            session_id,
            payload,
        )
        now[0] += timedelta(seconds=16)
        expired_renewal = await coordinator.renew_pending_interruption_cascade_claim(
            session_id,
            first["attempt_id"],
            first["generation"],
            first["claim_id"],
        )
        expired_failure_recorded = await coordinator.mark_pending_interruption_cascade_failed(
            session_id,
            first["attempt_id"],
            first["generation"],
            first["claim_id"],
        )
        expired_completion = await coordinator.complete_pending_interruption_cascade(
            session_id,
            first["attempt_id"],
            first["generation"],
            first["claim_id"],
        )
        await coordinator.release_pending_interruption_cascade_claim(
            session_id,
            first["attempt_id"],
            first["generation"],
            first["claim_id"],
        )
        second = await coordinator.claim_pending_interruption_cascade(session_id, payload)
        stale_failure_recorded = await coordinator.mark_pending_interruption_cascade_failed(
            session_id,
            first["attempt_id"],
            first["generation"],
            first["claim_id"],
        )
        current_failure_recorded = await coordinator.mark_pending_interruption_cascade_failed(
            session_id,
            second["attempt_id"],
            second["generation"],
            second["claim_id"],
        )
        failed_status = await app.interruption_cascade_status(session_id)
        stale_completion = await coordinator.complete_pending_interruption_cascade(
            session_id,
            first["attempt_id"],
            first["generation"],
            first["claim_id"],
        )
        failed_generation_completion = await coordinator.complete_pending_interruption_cascade(
            session_id,
            second["attempt_id"],
            second["generation"],
            second["claim_id"],
        )
        third = await coordinator.claim_pending_interruption_cascade(session_id, payload)
        retry_status = await app.interruption_cascade_status(session_id)
        current_completion = await coordinator.complete_pending_interruption_cascade(
            session_id,
            third["attempt_id"],
            third["generation"],
            third["claim_id"],
        )
        return (
            first,
            first_claim_status,
            overlapping,
            renewed,
            protected_by_heartbeat,
            expired_renewal,
            expired_failure_recorded,
            expired_completion,
            second,
            third,
            retry_status,
            stale_failure_recorded,
            current_failure_recorded,
            failed_status,
            stale_completion,
            failed_generation_completion,
            current_completion,
            await store.load_checkpoint(session_id),
        )

    (
        first,
        first_claim_status,
        overlapping,
        renewed,
        protected_by_heartbeat,
        expired_renewal,
        expired_failure_recorded,
        expired_completion,
        second,
        third,
        retry_status,
        stale_failure_recorded,
        current_failure_recorded,
        failed_status,
        stale_completion,
        failed_generation_completion,
        current_completion,
        checkpoint,
    ) = asyncio.run(run())

    assert first["generation"] == 1
    assert first_claim_status == "pending"
    assert overlapping is None
    assert renewed is True
    assert protected_by_heartbeat is None
    assert expired_renewal is False
    assert expired_failure_recorded is False
    assert expired_completion == (False, False)
    assert second["generation"] == 2
    assert third["generation"] == 3
    assert failed_status == "failed"
    assert retry_status == "pending"
    assert stale_failure_recorded is False
    assert current_failure_recorded is True
    assert stale_completion == (False, False)
    assert failed_generation_completion == (False, False)
    assert current_completion == (True, True)
    assert checkpoint == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}


def test_interruption_cascade_lease_uses_session_store_time() -> None:
    async def run() -> None:
        store_now = [datetime(2026, 1, 1, tzinfo=UTC)]
        worker_now = [datetime(2100, 1, 1, tzinfo=UTC)]
        store = InMemorySessionStore(ownership_clock=lambda: store_now[0])
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: worker_now[0],
        )
        session_id = "sess_cascade_store_time_claim"
        payload = {
            "reason": "operator stop",
            "metadata": {},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=session_id,
                messages=[Message.text("user", "parent")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.checkpoint(
            session_id,
            {
                "pending_interruption_cascade": {
                    "attempt_id": "cascade-store-time-attempt",
                    "interrupt_payload": payload,
                }
            },
        )

        coordinator = app._background_interruption_coordinator
        first = await coordinator.claim_pending_interruption_cascade(session_id, payload)
        assert first is not None
        assert first["created_at"] == store_now[0].isoformat()
        assert await app.interruption_cascade_status(session_id) == "pending"

        worker_now[0] += timedelta(days=3650)
        assert await coordinator.claim_pending_interruption_cascade(session_id, payload) is None
        assert await app.interruption_cascade_status(session_id) == "pending"

        store_now[0] += timedelta(minutes=5)
        replacement = await coordinator.claim_pending_interruption_cascade(session_id, payload)
        assert replacement is not None
        assert replacement["generation"] == first["generation"] + 1
        assert datetime.fromisoformat(replacement["claim_expires_at"]) > store_now[0]

    asyncio.run(run())


def test_interruption_cascade_completion_publish_failure_keeps_retryable_marker(monkeypatch):
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        session_id = "sess_cascade_completion_publish_failure"
        payload = {
            "reason": "operator stop",
            "metadata": {},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=session_id,
                messages=[Message.text("user", "parent")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.checkpoint(
            session_id,
            {
                "pending_interruption_cascade": {
                    "attempt_id": "cascade-completion-publish-attempt",
                    "interrupt_payload": payload,
                    "failure_recorded": True,
                }
            },
        )
        original_emit = app._event_writer.emit

        async def fail_completion_event(event):
            if event.type == EventType.SESSION_INTERRUPTION_CASCADE_COMPLETED:
                raise RuntimeError("event store unavailable")
            return await original_emit(event)

        monkeypatch.setattr(app._event_writer, "emit", fail_completion_event)
        await app._background_interruption_coordinator.run_cascade(
            parent_session_id=session_id,
            interrupt_payload=payload,
        )
        return (
            await store.load_checkpoint(session_id),
            await app.interruption_cascade_status(session_id),
            await store.load_events(session_id),
        )

    checkpoint, status, events = asyncio.run(run())

    marker = checkpoint["pending_interruption_cascade"]
    assert marker["failure_recorded"] is True
    assert "claim_id" not in marker
    assert "claim_expires_at" not in marker
    assert status == "failed"
    assert all(event.type != EventType.SESSION_INTERRUPTION_CASCADE_COMPLETED for event in events)


def test_interruption_cascade_completion_clear_failure_is_durably_reported(monkeypatch):
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        session_id = "sess_cascade_completion_clear_failure"
        payload = {
            "reason": "operator stop",
            "metadata": {},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=session_id,
                messages=[Message.text("user", "parent")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.checkpoint(
            session_id,
            {
                "pending_interruption_cascade": {
                    "attempt_id": "cascade-completion-clear-attempt",
                    "interrupt_payload": payload,
                    "failure_recorded": True,
                }
            },
        )

        async def fail_checkpoint_clear(*_args):
            raise RuntimeError("checkpoint store unavailable")

        monkeypatch.setattr(
            app._background_interruption_coordinator,
            "complete_pending_interruption_cascade",
            fail_checkpoint_clear,
        )
        await app._background_interruption_coordinator.run_cascade(
            parent_session_id=session_id,
            interrupt_payload=payload,
        )
        return (
            await store.load_checkpoint(session_id),
            await app.interruption_cascade_status(session_id),
            await store.load_events(session_id),
        )

    checkpoint, status, events = asyncio.run(run())

    marker = checkpoint["pending_interruption_cascade"]
    assert marker["failure_recorded"] is True
    assert "claim_id" not in marker
    assert status == "failed"
    assert [event.type for event in events] == [
        EventType.SESSION_INTERRUPTION_CASCADE_COMPLETED,
        EventType.SESSION_INTERRUPTION_CASCADE_FAILED,
    ]
    failure = events[-1].payload["failures"][0]
    assert failure == {
        "scope": "parent",
        "session_id": "sess_cascade_completion_clear_failure",
        "reason": "completion_checkpoint_clear_failed",
        "error_type": "RuntimeError",
    }


def test_operator_metadata_cannot_claim_a_background_cascade_origin():
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="reviewer", model="fakemodel"))
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_forged_cascade_origin",
                messages=[Message.text("user", "task")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fakemodel"),
        )
        _ = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_forged_cascade_origin",
                    reason="operator stop",
                    metadata={
                        "source": "background_subagent_parent_interrupt",
                        "interruption_origin": "background_cascade",
                    },
                )
            )
        ]
        return await store.load_events("sess_forged_cascade_origin")

    events = asyncio.run(run())

    interrupted = [event for event in events if event.type == EventType.SESSION_INTERRUPTED]
    assert len(interrupted) == 1
    assert "interruption_origin" not in interrupted[0].payload
    assert interrupted[0].payload["interruption_type"] == "operator_requested"


@pytest.mark.parametrize("cascade_retry", [False, True], ids=["operator", "cascade"])
def test_adopting_a_retained_input_interrupt_keeps_its_operator_identity(cascade_retry):
    async def run():
        store = InMemorySessionStore()
        session_id = "sess_adopted_input_interrupt"

        def make_app() -> CayuApp:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UserInputRecoveryProvider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"), tools=[UserInputTool()]
            )
            return app

        await _collect_events(
            make_app().run(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "ask a question")],
                )
            )
        )
        original = store.transition_status_and_checkpoint
        injected = False

        async def lose_acknowledgement(*args, **kwargs):
            nonlocal injected
            result = await original(*args, **kwargs)
            if kwargs.get("to_status") is SessionStatus.INTERRUPTING and not injected:
                injected = True
                raise _SimulatedProcessLoss("acknowledgement lost")
            return result

        store.transition_status_and_checkpoint = lose_acknowledgement
        request = InterruptSessionRequest(session_id=session_id, reason="operator stop")
        with suppress_interruption_cascade(), pytest.raises(_SimulatedProcessLoss):
            await _collect_events(make_app().interrupt_session(request))
        store.transition_status_and_checkpoint = original
        if cascade_retry:
            with suppress_interruption_cascade(), _background_cascade_interruption():
                return await _collect_events(make_app().interrupt_session(request))
        with suppress_interruption_cascade():
            return await _collect_events(make_app().interrupt_session(request))

    events = asyncio.run(run())

    assert events[-1].type is EventType.SESSION_INTERRUPTED
    assert "interruption_origin" not in events[-1].payload
