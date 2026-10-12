"""Resume contracts."""

from __future__ import annotations

import asyncio
from uuid import UUID

import pytest
from tests.core._execution_profile_fixtures import (
    profiled_session_identity,
    run_request_with_registered_tool_ceiling,
)
from tests.runtime.runtime_provider_support import FakeProvider, VersionedFakeProvider
from tests.runtime.runtime_session_support import (
    _tool_round_identity,
    checkpoint_without_active_invocation_profile,
    collect_events,
    collect_resume_events,
)
from tests.runtime.runtime_tool_support import EchoTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.context.base import ContextPolicy, ContextRequest
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions._model_failover import ModelTarget
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import EventRecord, Session, SessionStatus
from cayu.sessions.requests import ResumeRequest, RunRequest


class OtherProvider(FakeProvider):
    name = "other"


def test_cayu_app_resumes_completed_session_from_stored_transcript():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
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
                session_id="sess_resume",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_resume",
                messages=[Message.text("user", "second request")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[0].payload == {"agent_name": "assistant", "appended_messages": 1, "run_epoch": 3}
    assert [message.content[0].text for message in provider.requests[1].messages] == [
        "first request",
        "first answer",
        "second request",
    ]

    transcript = asyncio.run(store.load_transcript("sess_resume"))
    assert [message.content[0].text for message in transcript] == [
        "first request",
        "first answer",
        "second request",
        "second answer",
    ]
    session = asyncio.run(store.load("sess_resume"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_cayu_app_rejects_completed_session_with_pending_tool_round():
    from cayu.runtime import _runtime_records as runtime_records
    from cayu.runtime import _tool_round_recovery as tool_round_recovery

    store = InMemorySessionStore()
    provider = FakeProvider([])
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    checkpoint, pending_round = tool_round_recovery.checkpoint_with_pending_tool_round(
        None,
        agent_name="assistant",
        environment_name=None,
        task_id=None,
        tool_calls=[
            runtime_records.ToolCallRequest(
                id="completed_pending_call",
                name="side_effect",
                arguments={},
            )
        ],
        policy_outcomes=None,
        structured_output=None,
        tool_round_identity=_tool_round_identity(),
    )

    async def seed() -> None:
        await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id="completed_pending_round",
                messages=[Message.text("user", "hello")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=app,
            ),
        )
        await store.checkpoint("completed_pending_round", checkpoint)
        await store.update_status("completed_pending_round", SessionStatus.COMPLETED)

    asyncio.run(seed())

    with pytest.raises(
        RuntimeError,
        match="Completed session has an inconsistent pending tool round",
    ):
        asyncio.run(
            collect_resume_events(
                app,
                ResumeRequest(
                    session_id="completed_pending_round",
                    messages=[Message.text("user", "continue")],
                ),
            )
        )

    session = asyncio.run(store.load("completed_pending_round"))
    assert session is not None and session.status == SessionStatus.COMPLETED
    assert asyncio.run(store.load_checkpoint("completed_pending_round")) == checkpoint
    assert pending_round.tool_round_id == checkpoint["pending_tool_round"]["tool_round_id"]


def test_cayu_app_resume_uses_stored_provider_and_model_identity():
    class RecordingPolicy(ContextPolicy):
        def __init__(self) -> None:
            self.requests: list[ContextRequest] = []

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return ExecutionProfileBehaviorIdentity(
                name="tests:runtime:stored-model-recording-context",
                behavior_version="1",
                implementation_version="1",
            )

        async def build(self, request: ContextRequest) -> list[Message]:
            self.requests.append(request)
            return request.messages

    store = InMemorySessionStore()
    initial_provider = VersionedFakeProvider(
        [
            ModelStreamEvent.text_delta("first answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    initial_app = CayuApp(session_store=store)
    initial_app.register_provider(initial_provider, default=True)
    initial_app.register_agent(
        AgentSpec(name="assistant", model="stored-model"),
        context_policy=RecordingPolicy(),
    )

    asyncio.run(
        collect_events(
            initial_app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_stored_identity",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    resume_provider = VersionedFakeProvider(
        [
            ModelStreamEvent.text_delta("second answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    default_provider = OtherProvider(
        [
            ModelStreamEvent.text_delta("wrong provider"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    resumed_app = CayuApp(session_store=store)
    resumed_app.register_provider(resume_provider)
    resumed_app.register_provider(default_provider, default=True)
    context_policy = RecordingPolicy()
    resumed_app.register_agent(
        AgentSpec(name="assistant", model="new-default-model"),
        context_policy=context_policy,
    )

    asyncio.run(
        collect_resume_events(
            resumed_app,
            ResumeRequest(
                session_id="sess_stored_identity",
                messages=[Message.text("user", "second request")],
            ),
        )
    )

    assert len(resume_provider.requests) == 1
    assert resume_provider.requests[0].model == "stored-model"
    assert [request.agent.model for request in context_policy.requests] == ["stored-model"]
    assert default_provider.requests == []


def test_cayu_app_resume_model_updates_session_active_model():
    class RecordingPolicy(ContextPolicy):
        def __init__(self) -> None:
            self.requests: list[ContextRequest] = []

        async def build(self, request: ContextRequest) -> list[Message]:
            self.requests.append(request)
            return request.messages

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("third answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    context_policy = RecordingPolicy()
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="stored-model"),
        context_policy=context_policy,
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_model_update",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    second_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_model_update",
                messages=[Message.text("user", "second request")],
                target=ModelTarget(provider_name="fake", model="upgraded-model"),
            ),
        )
    )

    session_after_update = asyncio.run(store.load("sess_model_update"))
    assert session_after_update is not None
    assert session_after_update.model == "upgraded-model"
    assert second_events[0].type == EventType.SESSION_EXECUTION_PROFILE_DECIDED
    assert second_events[1].type == EventType.SESSION_MODEL_SWITCHED
    model_started = next(event for event in second_events if event.type == EventType.MODEL_STARTED)
    assert model_started.payload["model"] == "upgraded-model"

    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_model_update",
                messages=[Message.text("user", "third request")],
            ),
        )
    )

    assert [request.model for request in provider.requests] == [
        "stored-model",
        "upgraded-model",
        "upgraded-model",
    ]
    assert [request.agent.model for request in context_policy.requests] == [
        "stored-model",
        "upgraded-model",
        "upgraded-model",
    ]


def test_cayu_app_resume_releases_run_fence_when_setup_is_cancelled():
    class CancellingTranscriptStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        cancel_load = False

        async def load_transcript(self, session_id: str) -> list[Message]:
            if self.cancel_load:
                raise asyncio.CancelledError()
            return await super().load_transcript(session_id)

    store = CancellingTranscriptStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def setup_and_resume() -> tuple[
        Session,
        list[Message],
        list[EventRecord],
        list[Event],
    ]:
        await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id="sess_resume_setup_cancel",
                messages=[Message.text("user", "hi")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=app,
            ),
        )
        await store.update_status("sess_resume_setup_cancel", SessionStatus.INTERRUPTED)
        store.cancel_load = True
        with pytest.raises(asyncio.CancelledError):
            await collect_resume_events(
                app,
                ResumeRequest(
                    session_id="sess_resume_setup_cancel",
                    messages=[Message.text("user", "continue")],
                ),
            )
        session = await store.load("sess_resume_setup_cancel")
        assert session is not None
        store.cancel_load = False
        transcript = await store.load_transcript(session.id)
        interactions = await store.query_events(
            EventQuery(
                session_id=session.id,
                event_type=EventType.INTERACTION_STARTED,
            )
        )
        session = await store.update_metadata(
            "sess_resume_setup_cancel",
            {"fence_released_after_cancel": True},
        )
        events = await store.load_events("sess_resume_setup_cancel")
        return session, transcript, interactions, events

    session, transcript, interactions, events = asyncio.run(setup_and_resume())

    assert session.status == SessionStatus.INTERRUPTED
    # Repairing the missing terminal event and attempting the resumed run are
    # separate fenced operations; each acquires and releases its own run epoch.
    assert session.run_epoch == 4
    assert transcript == [Message.text("user", "continue")]
    assert len(interactions) == 1
    assert interactions[0].event.interaction_id is not None
    assert session.metadata["fence_released_after_cancel"] is True
    interrupted = [event for event in events if event.type == EventType.SESSION_INTERRUPTED]
    assert len(interrupted) == 2
    assert interrupted[0].payload["terminal_evidence_repaired"] is True
    assert interrupted[1].payload["abandoned"] is True


def test_cayu_app_resume_rejects_active_sessions():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("should not run"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def setup_running_session() -> None:
        await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id="sess_running",
                messages=[Message.text("user", "hi")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        await store.update_status("sess_running", SessionStatus.RUNNING)
        await store.append_transcript_messages(
            "sess_running",
            [Message.text("user", "hi")],
        )

    asyncio.run(setup_running_session())

    with pytest.raises(ValueError, match="execution is still in progress"):
        asyncio.run(
            collect_resume_events(
                app,
                ResumeRequest(
                    session_id="sess_running",
                    messages=[Message.text("user", "continue")],
                ),
            )
        )

    assert provider.requests == []
    session = asyncio.run(store.load("sess_running"))
    assert session is not None
    assert session.status == SessionStatus.RUNNING


def test_cayu_app_resume_marks_session_failed_when_transcript_load_fails():
    class BrokenTranscriptStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def load_transcript(self, session_id: str) -> list[Message]:
            raise RuntimeError("transcript unavailable")

    store = BrokenTranscriptStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("should not run"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def setup_completed_session() -> None:
        await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id="sess_broken_transcript",
                messages=[Message.text("user", "hi")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=app,
            ),
        )
        await store.update_status("sess_broken_transcript", SessionStatus.COMPLETED)

    asyncio.run(setup_completed_session())

    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_broken_transcript",
                messages=[Message.text("user", "continue")],
            ),
        )
    )

    assert provider.requests == []
    assert [event.type for event in events] == [EventType.SESSION_FAILED]
    assert events[0].payload["session_run_operation_id"] == PRIVATE_EVENT_AUTHORITY
    assert {
        key: value for key, value in events[0].payload.items() if key != "session_run_operation_id"
    } == {
        "error": "transcript unavailable",
        "error_type": "RuntimeError",
    }
    failed_records = asyncio.run(
        store.query_events(
            EventQuery(
                session_id="sess_broken_transcript",
                event_type=EventType.SESSION_FAILED,
            )
        )
    )
    assert len(failed_records) == 1
    operation_id = failed_records[0].event.payload["session_run_operation_id"]
    assert str(UUID(operation_id)) == operation_id
    session = asyncio.run(store.load("sess_broken_transcript"))
    assert session is not None
    assert session.status == SessionStatus.FAILED
    assert checkpoint_without_active_invocation_profile(
        asyncio.run(store.load_checkpoint("sess_broken_transcript"))
    ) == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}


def test_cayu_app_resume_uses_context_policy_and_preserves_full_transcript():
    class LastMessagePolicy(ContextPolicy):
        def __init__(self) -> None:
            self.seen_messages: list[list[str]] = []

        async def build(self, request: ContextRequest) -> list[Message]:
            self.seen_messages.append([message.content[0].text for message in request.messages])
            return [request.messages[-1]]

    store = InMemorySessionStore()
    policy = LastMessagePolicy()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_resume_context",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_resume_context",
                messages=[Message.text("user", "second request")],
                metadata={"source": "resume-test"},
            ),
        )
    )

    assert policy.seen_messages[-1] == [
        "first request",
        "first answer",
        "second request",
    ]
    assert [message.content[0].text for message in provider.requests[1].messages] == [
        "second request"
    ]
    transcript = asyncio.run(store.load_transcript("sess_resume_context"))
    assert [message.content[0].text for message in transcript] == [
        "first request",
        "first answer",
        "second request",
        "second answer",
    ]


def test_cayu_app_resume_continues_tool_rounds():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("ready"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "from resumed tool"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_resume_tool",
                messages=[Message.text("user", "first")],
            ),
        )
    )

    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_resume_tool",
                messages=[Message.text("user", "use tool")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert provider.requests[2].messages[-1].role == "tool"
    assert provider.requests[2].messages[-1].content[0].content == "from resumed tool"
    transcript = asyncio.run(store.load_transcript("sess_resume_tool"))
    assert [message.role for message in transcript] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
