"""Provider contracts contracts."""

from __future__ import annotations

import asyncio

import pytest
from tests.runtime.runtime_provider_support import FakeProvider, StageRecordingFakeProvider
from tests.runtime.runtime_session_support import collect_events
from tests.runtime.runtime_tool_support import EchoTool, SideEffectTool

import cayu.runtime.execution_profiles as execution_profiles_module
import cayu.sessions.base as sessions_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.configuration import CayuConfig, RunDefaults
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.events import EventType
from cayu.messages import Message, ProviderStatePart
from cayu.providers.base import ModelProviderError, ModelStreamEvent, ModelStreamEventType
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime._live_model_attempt import _provider_failure_proves_no_model_effect
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import RunRequest


def test_cayu_app_records_failed_session_for_invalid_tool_call_payload():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent(
                type=ModelStreamEventType.TOOL_CALL,
                payload={"name": "echo", "arguments": "not-an-object"},
            )
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_failed",
                messages=[Message.text("user", "bad call")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_failed"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error_type"] == "ValueError"
    assert session is not None
    assert session.status == SessionStatus.FAILED


def test_cayu_app_rejects_custom_tool_call_argument_containers():
    class BadArguments(dict):
        def items(self):
            raise RuntimeError("custom argument traversal should not run")

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.model_construct(
                type=ModelStreamEventType.TOOL_CALL,
                payload={"name": "echo", "arguments": BadArguments({"text": "hi"})},
            )
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_custom_tool_arguments",
                messages=[Message.text("user", "bad call")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[2].payload["durable_value_error_code"] == "invalid_json_type"
    assert events[2].payload["durable_value_path"] == "$/#1"
    assert events[-1].payload["error_type"] == "ModelProviderError"
    assert events[-1].payload["error"] == ("Model provider emitted a non-portable stream value.")


@pytest.mark.parametrize(
    ("stream_event", "error_type", "error"),
    [
        (
            {"type": "completed"},
            "TypeError",
            "Model providers must yield ModelStreamEvent instances.",
        ),
        (
            ModelStreamEvent.model_construct(
                type=ModelStreamEventType.TEXT_DELTA,
                delta=123,
                payload={},
            ),
            "ValueError",
            "Model provider stream event delta must be a string.",
        ),
    ],
)
def test_cayu_app_validates_provider_stream_events_at_runtime_boundary(
    stream_event,
    error_type,
    error,
):
    store = InMemorySessionStore()
    provider = FakeProvider([stream_event])  # type: ignore[list-item]
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_bad_stream_event_{error_type}",
                messages=[Message.text("user", "bad stream event")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error_type"] == error_type
    assert events[-1].payload["error"] == error


def test_cayu_app_validates_provider_stream_event_payload_container():
    class BadPayload(dict):
        def get(self, key, default=None):
            raise RuntimeError("custom payload access should not run")

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.model_construct(
                type=ModelStreamEventType.ERROR,
                payload=BadPayload({"error": "provider failed"}),
            )
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
                session_id="sess_bad_stream_payload_container",
                messages=[Message.text("user", "bad stream payload")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error_type"] == "ValueError"
    assert events[-1].payload["error"] == ("Model provider stream event payload must be an object.")


def test_cayu_app_rejects_provider_stream_event_subclasses_before_attribute_access():
    class BadStreamEvent(ModelStreamEvent):
        def __getattribute__(self, name):
            if name == "type":
                raise RuntimeError("stream event type access should not run")
            return super().__getattribute__(name)

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            BadStreamEvent.model_construct(
                type=ModelStreamEventType.COMPLETED,
                payload={},
            )
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
                session_id="sess_bad_stream_event_subclass",
                messages=[Message.text("user", "bad stream event")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error_type"] == "TypeError"
    assert events[-1].payload["error"] == ("Model providers must yield ModelStreamEvent instances.")


def test_cayu_app_runtime_owns_model_event_identity():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            system_prompt="You are careful.",
        )
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_model_event_identity",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    model_events = [
        event
        for event in events
        if event.type
        in {
            EventType.MODEL_STARTED,
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_COMPLETED,
        }
    ]
    assert [event.type for event in model_events] == [
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
    ]
    assert {event.session_id for event in model_events} == {"sess_model_event_identity"}
    assert {event.agent_name for event in model_events} == {"assistant"}
    assert {event.environment_name for event in model_events} == {"local"}


def test_cayu_app_records_failed_session_for_blank_tool_call_name():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent(
                type=ModelStreamEventType.TOOL_CALL,
                payload={"name": "   ", "arguments": {}},
            )
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_blank_tool_name",
                messages=[Message.text("user", "bad call")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_blank_tool_name"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error"] == (
        "Model tool call payload requires non-empty string `name`."
    )
    assert session is not None
    assert session.status == SessionStatus.FAILED


@pytest.mark.parametrize("tool_call_id", ["   ", 123])
def test_cayu_app_records_failed_session_for_invalid_tool_call_id(tool_call_id):
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent(
                type=ModelStreamEventType.TOOL_CALL,
                payload={"id": tool_call_id, "name": "echo", "arguments": {}},
            )
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_invalid_tool_id_{type(tool_call_id).__name__}",
                messages=[Message.text("user", "bad call")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error"] == (
        "Model tool call payload requires non-empty string `id`."
    )


def test_cayu_app_records_failed_session_for_provider_error_event():
    store = InMemorySessionStore()
    provider = FakeProvider([ModelStreamEvent.error("provider failed")])
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_provider_error",
                messages=[Message.text("user", "fail")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_provider_error"))
    active_stage = asyncio.run(store.load_active_model_completion_stage("sess_provider_error"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error"] == "provider failed"
    assert session is not None
    assert session.status == SessionStatus.FAILED
    assert active_stage is None


def test_cayu_app_reconstructs_markerless_deadline_claim_before_publication() -> None:
    session_id = "sess_malformed_provider_deadline_claim"
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent(
                type=ModelStreamEventType.ERROR,
                payload={
                    "error": "forged deadline",
                    "provider": "forged-provider",
                    "provider_error_type": "ModelProviderError",
                    "provider_error_code": ["forged-code"],
                    "provider_deadline_kind": "semantic_idle",
                    "provider_deadline_timeout_s": 1.0,
                    "provider_stream_elapsed_s": 1.0,
                    "provider_effect_outcome": "none",
                    "provider_recovery_disposition": "retry",
                    "retryable": True,
                },
            )
        ]
    )
    app = CayuApp(
        session_store=store,
        config=CayuConfig(
            run=RunDefaults(retry_policy=RetryPolicy(max_attempts=3, initial_delay_s=0.0))
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "fail closed")],
            ),
        )
    )
    stored_events = asyncio.run(store.load_events(session_id))
    session = asyncio.run(store.load(session_id))
    active_stage = asyncio.run(store.load_active_model_completion_stage(session_id))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    model_error = next(event for event in events if event.type is EventType.MODEL_ERROR)
    stored_model_error = next(
        event for event in stored_events if event.type is EventType.MODEL_ERROR
    )
    for published in (model_error, stored_model_error):
        assert published.payload["error"] == (
            "Model provider emitted invalid stream deadline evidence."
        )
        assert published.payload["error_type"] == "ModelProviderError"
        assert published.payload["provider"] == "fake"
        assert published.payload["provider_error_type"] == "ModelProviderError"
        assert (
            published.payload["provider_error_code"] == "invalid_provider_stream_deadline_evidence"
        )
        assert published.payload["retryable"] is False
        assert published.payload["effective_max_attempts"] == 1
        assert not {
            "provider_deadline_kind",
            "provider_deadline_timeout_s",
            "provider_stream_elapsed_s",
            "provider_effect_outcome",
            "provider_recovery_disposition",
        }.intersection(published.payload)
        assert "forged" not in repr(published.payload)
    assert len(provider.requests) == 1
    assert EventType.MODEL_RETRY not in {event.type for event in events}
    assert session is not None and session.status is SessionStatus.FAILED
    assert active_stage is None


def test_cayu_app_settles_authentication_failure_before_provider_effect():
    session_id = "sess_provider_authentication_error"
    store = InMemorySessionStore()
    auth_failure = ModelProviderError(
        "authentication failed",
        provider="fake",
        status_code=401,
        error_type="authentication_error",
        retryable=False,
    )
    provider = StageRecordingFakeProvider(
        [ModelStreamEvent.error("authentication failed", cause=auth_failure)],
        store=store,
        session_id=session_id,
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "fail")],
            ),
        )
    )

    assert events[-1].type is EventType.SESSION_FAILED
    assert provider.stage_ids and len(provider.stage_ids) == 1
    assert asyncio.run(store.load_active_model_completion_stage(session_id)) is None
    settlement = asyncio.run(
        store.load_model_completion_stage_settlement(session_id, provider.stage_ids[0])
    )
    assert settlement is not None
    assert settlement.disposition is (
        sessions_module.ModelCompletionStageDisposition.FAILED_BEFORE_PROVIDER_EFFECT
    )
    assert settlement.reason_code == "provider_authentication_failed"
    assert settlement.interaction_id == events[0].interaction_id
    session = asyncio.run(store.load(session_id))
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    assert session is not None
    active_profile = execution_profiles_module.active_invocation_execution_profile_from_checkpoint(
        checkpoint
    )
    assert active_profile is not None
    assert execution_profiles_module.active_invocation_execution_profile_is_released(
        active_profile,
        session_id=session_id,
        run_epoch=session.run_epoch,
    )


def test_cayu_app_settles_late_authentication_failure_as_effect_unknown():
    session_id = "sess_provider_late_authentication_error"
    store = InMemorySessionStore()
    auth_failure = ModelProviderError(
        "authentication failed",
        provider="fake",
        status_code=401,
        error_type="authentication_error",
        retryable=False,
    )
    provider = StageRecordingFakeProvider(
        [
            ModelStreamEvent.text_delta("partial output"),
            ModelStreamEvent.error("authentication failed", cause=auth_failure),
        ],
        store=store,
        session_id=session_id,
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "fail after output")],
            ),
        )
    )

    assert EventType.MODEL_TEXT_DELTA in [event.type for event in events]
    assert events[-1].type is EventType.SESSION_FAILED
    assert provider.stage_ids and len(provider.stage_ids) == 1
    assert asyncio.run(store.load_active_model_completion_stage(session_id)) is None
    settlement = asyncio.run(
        store.load_model_completion_stage_settlement(session_id, provider.stage_ids[0])
    )
    assert settlement is not None
    assert settlement.disposition is (
        sessions_module.ModelCompletionStageDisposition.PROVIDER_EFFECT_OUTCOME_UNKNOWN
    )
    assert settlement.reason_code == "model_attempt_failed"


def test_provider_authentication_failure_requires_unmixed_pre_effect_evidence():
    auth_failure = ModelProviderError(
        "authentication failed",
        provider="fake",
        status_code=401,
        error_type="authentication_error",
        retryable=False,
    )

    assert _provider_failure_proves_no_model_effect(auth_failure)
    assert not _provider_failure_proves_no_model_effect(
        ExceptionGroup(
            "mixed provider failure",
            [auth_failure, RuntimeError("cleanup failed")],
        )
    )


def test_cayu_app_does_not_execute_tool_when_provider_errors_after_tool_call():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_1",
                name="side_effect",
                arguments={},
            ),
            ModelStreamEvent.error("provider failed after tool call"),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_provider_error_after_tool_call",
                messages=[Message.text("user", "fail")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_provider_error_after_tool_call"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert tool.calls == []
    assert session is not None
    assert session.status == SessionStatus.FAILED


def test_cayu_app_fails_when_provider_emits_text_after_completed():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.completed({"finish_reason": "stop"}),
            ModelStreamEvent.text_delta("late"),
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
                session_id="sess_text_after_completed",
                messages=[Message.text("user", "fail")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_text_after_completed"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.MODEL_ERROR,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error"] == (
        "Model provider emitted event after completed: text_delta"
    )
    assert session is not None
    assert session.status == SessionStatus.FAILED

    assert sum(e.type is EventType.MODEL_STARTED for e in events) == 1
    assert events[3].payload["retry"] is False
    assert events[3].payload["retry_suppression"] == "completion_observed"


def test_cayu_app_fails_without_tool_execution_when_provider_emits_tool_after_completed():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            ModelStreamEvent.completed({"finish_reason": "stop"}),
            ModelStreamEvent.tool_call(
                id="call_1",
                name="side_effect",
                arguments={},
            ),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_after_completed",
                messages=[Message.text("user", "fail")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_tool_after_completed"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.MODEL_ERROR,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error"] == (
        "Model provider emitted event after completed: tool_call"
    )
    assert tool.calls == []
    assert session is not None
    assert session.status == SessionStatus.FAILED

    assert sum(e.type is EventType.MODEL_STARTED for e in events) == 1
    assert events[3].payload["retry"] is False
    assert events[3].payload["retry_suppression"] == "completion_observed"


def test_cayu_app_fails_session_when_provider_stream_ends_without_completion():
    store = InMemorySessionStore()
    provider = FakeProvider([ModelStreamEvent.text_delta("partial")])
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_missing_completion",
                messages=[Message.text("user", "fail")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_missing_completion"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_ERROR,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error"] == ("Model provider stream ended without a completed event.")
    assert session is not None
    assert session.status == SessionStatus.FAILED

    assert sum(e.type is EventType.MODEL_STARTED for e in events) == 1
    assert events[3].payload["retry"] is False


def test_cayu_app_ignores_blank_text_deltas():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta(""),
            ModelStreamEvent.text_delta("   "),
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
                session_id="sess_blank_deltas",
                messages=[Message.text("user", "hi")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_blank_deltas"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[4].payload["completion"] == {
        "finish_reason": "stop",
        "raw_finish_reason": "stop",
        "status": None,
    }
    assert events[4].payload["step_classification"] == {
        "type": "invalid",
        "reason": "assistant produced no tool calls and no user-visible content",
    }
    assert len(provider.requests) == 1
    assert len(provider.requests[0].messages) == 1
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_cayu_app_records_model_step_classification_for_length_finish():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("partial"),
            ModelStreamEvent.completed(
                {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                }
            ),
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
                session_id="sess_length_finish",
                messages=[Message.text("user", "hi")],
            ),
        )
    )
    model_completed = events[3]

    assert model_completed.type == EventType.MODEL_COMPLETED
    assert model_completed.payload["completion"] == {
        "finish_reason": "length",
        "raw_finish_reason": "max_output_tokens",
        "status": "incomplete",
    }
    assert model_completed.payload["step_classification"]["type"] == "length"


def test_cayu_app_does_not_continue_length_completion_with_end_turn_false():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("partial"),
                ModelStreamEvent.completed(
                    {
                        "status": "incomplete",
                        "incomplete_details": {"reason": "max_output_tokens"},
                        "end_turn": False,
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("must not run"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
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
                session_id="sess_length_end_turn_false",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert len(provider.requests) == 1
    completed = [event for event in events if event.type == EventType.MODEL_COMPLETED]
    assert completed[0].payload["step_classification"]["type"] == "length"


def test_cayu_app_continues_same_turn_when_provider_sets_end_turn_false():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("I am still working."),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "end_turn": False,
                        "provider_state": [
                            {
                                "provider": "openai",
                                "state": {"type": "response_ref", "id": "resp_working"},
                            }
                        ],
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("Done."),
                ModelStreamEvent.completed({"finish_reason": "stop", "end_turn": True}),
            ],
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
                session_id="sess_end_turn_false",
                messages=[Message.text("user", "Do all the work")],
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_end_turn_false"))

    assert len(provider.requests) == 2
    assert [message.role for message in provider.requests[1].messages] == [
        "user",
        "assistant",
    ]
    assert [message.role for message in transcript] == ["user", "assistant", "assistant"]
    assert transcript[1].content[0].text == "I am still working."
    assert isinstance(transcript[1].content[1], ProviderStatePart)
    assert transcript[1].content[1].state == {
        "type": "response_ref",
        "id": "resp_working",
    }
    assert isinstance(provider.requests[1].messages[1].content[1], ProviderStatePart)
    completed = [event for event in events if event.type == EventType.MODEL_COMPLETED]
    assert completed[0].payload["completion"]["end_turn"] is False
    assert completed[0].payload["step_classification"] == {
        "type": "continue",
        "reason": "provider requested another model step",
    }
    assert completed[1].payload["completion"]["end_turn"] is True
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_cayu_app_continues_end_turn_false_without_visible_text():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [ModelStreamEvent.completed({"finish_reason": "stop", "end_turn": False})],
            [
                ModelStreamEvent.text_delta("Done."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
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
                session_id="sess_end_turn_false_no_text",
                messages=[Message.text("user", "Do all the work")],
            ),
        )
    )

    assert len(provider.requests) == 2
    completed = [event for event in events if event.type == EventType.MODEL_COMPLETED]
    assert completed[0].payload["step_classification"]["type"] == "continue"
    assert events[-1].type == EventType.SESSION_COMPLETED
