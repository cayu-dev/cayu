"""Structured output contracts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from tests.runtime.runtime_provider_support import FakeProvider, NativeStructuredOutputFakeProvider
from tests.runtime.runtime_session_support import (
    collect_events,
    collect_resume_events,
    collect_tool_approval_events,
    collect_tool_approval_recovery_events,
)
from tests.runtime.runtime_tool_support import EchoTool, RequireApprovalPolicy, SideEffectTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import (
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
)
from cayu.budgets.run_limits import RunLimits
from cayu.context.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    NativeStructuredOutputUnsupported,
    StructuredOutputSpec,
)
from cayu.events import EventType
from cayu.messages import Message, ToolResultPart
from cayu.providers.base import ModelStreamEvent, NativeStructuredOutputSchemaInvalid
from cayu.providers.openai import OpenAIProvider
from cayu.runtime._session_preparation import _require_native_structured_output_support
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileMismatchError,
)
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.sessions.transcript_queries import TranscriptQuery


class SchemaRejectingProvider(NativeStructuredOutputFakeProvider):
    def preflight_native_structured_output_schema(self, json_schema: dict[str, Any]) -> None:
        raise NativeStructuredOutputSchemaInvalid("$: rejected by provider preflight.")


def test_cayu_app_requires_structured_output_final_tool_call():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta('{"answer":"ok"}'),
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
                session_id="sess_structured_output_requires_tool",
                messages=[Message.text("user", "answer with json")],
                structured_output=StructuredOutputSpec(
                    name="answer",
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    max_retries=0,
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_structured_output_requires_tool"))
    session = asyncio.run(store.load("sess_structured_output_requires_tool"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[5].payload["valid"] is False
    assert events[5].payload["errors"][0]["message"] == (
        f"Final structured output must be submitted with the `{STRUCTURED_OUTPUT_TOOL_NAME}` tool."
    )
    assert events[-1].payload["error"] == (
        "Structured output validation failed after 1 attempt(s)."
    )
    assert session is not None
    assert session.status == SessionStatus.FAILED
    assert provider.requests[0].options["structured_output"]["name"] == "answer"
    assert provider.requests[0].options["structured_output"]["schema"]["required"] == ["answer"]
    assert [message.role for message in transcript] == ["user", "assistant"]
    assert transcript[-1].content[0].text == '{"answer":"ok"}'


def test_cayu_app_accepts_structured_output_final_tool_call():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_final",
                name=STRUCTURED_OUTPUT_TOOL_NAME,
                arguments={"output": {"answer": "ok"}},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
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
                session_id="sess_structured_output_tool_valid",
                messages=[Message.text("user", "answer with structured output")],
                structured_output=StructuredOutputSpec(
                    name="answer",
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_structured_output_tool_valid"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[5].payload["output"] == {"answer": "ok"}
    assert provider.requests[0].tools == [
        {
            "name": STRUCTURED_OUTPUT_TOOL_NAME,
            "description": (
                "Submit the final structured output for this run. Use this only when the "
                "final answer is ready. The value must be provided in the `output` field."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "output": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    }
                },
                "required": ["output"],
                "additionalProperties": False,
            },
        }
    ]
    system_messages = [
        message for message in provider.requests[0].messages if message.role == "system"
    ]
    assert len(system_messages) == 1
    assert STRUCTURED_OUTPUT_TOOL_NAME in system_messages[0].content[0].text
    assert [message.role for message in transcript] == ["user", "assistant", "tool"]
    assert transcript[-1].content[0].tool_name == STRUCTURED_OUTPUT_TOOL_NAME
    assert transcript[-1].content[0].content == "Structured output accepted."
    assert transcript[-1].content[0].structured == {"output": {"answer": "ok"}}
    assert transcript[-1].content[0].is_error is False


def test_cayu_app_redacts_structured_output_tool_result_before_transcript():
    from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor

    secret_value = "sk-live-structured-output-secret"
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_final",
                name=STRUCTURED_OUTPUT_TOOL_NAME,
                arguments={"output": {"answer": secret_value}},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret_value),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_structured_output_tool_redacted",
                messages=[Message.text("user", "answer with structured output")],
                structured_output=StructuredOutputSpec(
                    name="answer",
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_structured_output_tool_redacted"))

    assert events[3].type == EventType.TOOL_CALL_COMPLETED
    assert events[4].type == EventType.STRUCTURED_OUTPUT_VALIDATING
    assert events[5].type == EventType.STRUCTURED_OUTPUT_VALIDATED
    assert events[5].payload["output"] == {"answer": REDACTED_SECRET}
    assert [message.role for message in transcript] == ["user", "assistant", "tool"]
    tool_part = transcript[-1].content[0]
    assert isinstance(tool_part, ToolResultPart)
    assert tool_part.structured == {"output": {"answer": REDACTED_SECRET}}
    assert secret_value not in repr(tool_part)


def test_cayu_app_retries_invalid_structured_output_final_tool_call():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_final_invalid",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"wrong": "value"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_final_valid",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "fixed"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
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
                session_id="sess_structured_output_tool_retry",
                messages=[Message.text("user", "answer with structured output")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    max_retries=1,
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_structured_output_tool_retry"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.STRUCTURED_OUTPUT_RETRY,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[5].payload["errors"][0]["path"] == "$"
    assert events[6].payload["attempt"] == 1
    assert events[11].payload["output"] == {"answer": "fixed"}
    assert [message.role for message in transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    invalid_result = transcript[2].content[0]
    assert invalid_result.tool_name == STRUCTURED_OUTPUT_TOOL_NAME
    assert invalid_result.is_error is True
    assert "Structured output rejected" in invalid_result.content
    assert STRUCTURED_OUTPUT_TOOL_NAME in invalid_result.content
    assert "plain text" in invalid_result.content
    assert provider.requests[1].messages[-1].role == "tool"
    assert provider.requests[1].messages[-1].content[0].is_error is True


def test_cayu_app_redacts_structured_output_tool_validation_errors():
    from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor

    secret_value = "structured-output-error-secret"
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_final_invalid",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": secret_value}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_final_valid",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "ok"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret_value),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_structured_output_tool_error_redaction",
                messages=[Message.text("user", "answer with structured output")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"enum": ["ok"]}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    max_retries=1,
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_structured_output_tool_error_redaction"))

    failed_event = events[5]
    retry_event = events[6]
    assert failed_event.type == EventType.STRUCTURED_OUTPUT_FAILED
    assert retry_event.type == EventType.STRUCTURED_OUTPUT_RETRY
    assert secret_value not in str(failed_event.payload)
    assert secret_value not in str(retry_event.payload)
    assert REDACTED_SECRET in failed_event.payload["errors"][0]["message"]

    invalid_result = transcript[2].content[0]
    assert isinstance(invalid_result, ToolResultPart)
    assert invalid_result.tool_name == STRUCTURED_OUTPUT_TOOL_NAME
    assert invalid_result.is_error is True
    assert secret_value not in str(invalid_result.model_dump(mode="json"))
    assert REDACTED_SECRET in invalid_result.content
    assert REDACTED_SECRET in invalid_result.structured["structured_output_errors"][0]["message"]
    assert secret_value not in str(provider.requests[1].messages[-1].model_dump(mode="json"))


def test_cayu_app_uses_custom_repair_prompt_for_invalid_structured_output_tool_call():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_final_invalid",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"wrong": "value"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_final_valid",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "fixed"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
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
                session_id="sess_structured_output_tool_custom_repair",
                messages=[Message.text("user", "answer with structured output")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    max_retries=1,
                    repair_prompt="Call the finalizer tool again with the corrected object.",
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_structured_output_tool_custom_repair"))

    invalid_result = transcript[2].content[0]
    assert invalid_result.is_error is True
    assert "Structured output rejected" in invalid_result.content
    assert "Call the finalizer tool again with the corrected object." in invalid_result.content


def test_cayu_app_rejects_mixed_structured_output_tool_round_without_side_effects():
    store = InMemorySessionStore()
    side_effect_tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_side_effect",
                    name="side_effect",
                    arguments={"value": "do not run"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_final_mixed",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "too early"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_final_valid",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "fixed"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[side_effect_tool],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_structured_output_tool_mixed",
                messages=[Message.text("user", "use tools and answer")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                    max_retries=1,
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_structured_output_tool_mixed"))

    assert side_effect_tool.calls == []
    assert EventType.TOOL_CALL_STARTED not in [event.type for event in events]
    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_FAILED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.STRUCTURED_OUTPUT_RETRY,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[6].payload["errors"][0]["message"] == (
        "Call the structured-output tool by itself, not in the same tool round as other tools."
    )
    mixed_tool_results = transcript[2].content
    assert [part.tool_name for part in mixed_tool_results] == [
        "side_effect",
        STRUCTURED_OUTPUT_TOOL_NAME,
    ]
    assert all(part.is_error for part in mixed_tool_results)


def test_cayu_app_does_not_count_structured_output_tool_against_tool_call_limit():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_echo",
                    name="echo",
                    arguments={"text": "from tool"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_final",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "done"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
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
                session_id="sess_structured_output_tool_limit",
                messages=[Message.text("user", "use one tool and answer")],
                limits=RunLimits(max_tool_calls=1),
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                ),
            ),
        )
    )

    assert EventType.SESSION_LIMIT_REACHED not in [event.type for event in events]
    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[9].payload["output"] == {"answer": "done"}


def test_cayu_app_validates_native_structured_output_final_text():
    store = InMemorySessionStore()
    provider = NativeStructuredOutputFakeProvider(
        [
            ModelStreamEvent.text_delta('{"answer":"ok"}'),
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
                session_id="sess_structured_output_native_valid",
                messages=[Message.text("user", "answer with structured output")],
                structured_output=StructuredOutputSpec(
                    name="answer",
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    strategy="native",
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_structured_output_native_valid"))
    footprint = next(
        event.payload
        for event in asyncio.run(store.load_events("sess_structured_output_native_valid"))
        if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[5].payload["output"] == {"answer": "ok"}
    assert provider.requests[0].options["structured_output"]["strategy"] == "native"
    assert provider.requests[0].tools == []
    assert [message.role for message in provider.requests[0].messages] == ["user"]
    assert [message.role for message in transcript] == ["user", "assistant"]
    pressure = footprint["context_pressure"]
    assert pressure["estimated_structured_output_input_tokens"] > 0
    assert pressure["estimated_request_overhead_input_tokens"] == (
        pressure["estimated_tool_schema_input_tokens"]
        + pressure["estimated_structured_output_input_tokens"]
        + pressure["estimated_request_options_input_tokens"]
    )
    assert (
        footprint["component_tokens"]["total_input_tokens"]
        == pressure["estimated_context_input_tokens"]
    )


def test_cayu_app_retries_invalid_native_structured_output_final_text():
    store = InMemorySessionStore()
    provider = NativeStructuredOutputFakeProvider(
        [
            [
                ModelStreamEvent.text_delta('{"wrong":"value"}'),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta('{"answer":"fixed"}'),
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
                session_id="sess_structured_output_native_retry",
                messages=[Message.text("user", "answer with structured output")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    max_retries=1,
                    strategy="native",
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_structured_output_native_retry"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.STRUCTURED_OUTPUT_RETRY,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[5].payload["errors"][0]["path"] == "$"
    assert events[11].payload["output"] == {"answer": "fixed"}
    repair_message = transcript[2].content[0].text
    assert "Return only valid JSON" in repair_message
    assert STRUCTURED_OUTPUT_TOOL_NAME not in repair_message


def test_cayu_app_redacts_native_structured_output_repair_prompt_errors():
    from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor

    secret_value = "native-structured-output-error-secret"
    store = InMemorySessionStore()
    provider = NativeStructuredOutputFakeProvider(
        [
            [
                ModelStreamEvent.text_delta(f'{{"answer":"{secret_value}"}}'),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta('{"answer":"ok"}'),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret_value),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_native_structured_output_error_redaction",
                messages=[Message.text("user", "answer with structured output")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"enum": ["ok"]}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    max_retries=1,
                    strategy="native",
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_native_structured_output_error_redaction"))

    failed_event = events[5]
    retry_event = events[6]
    assert failed_event.type == EventType.STRUCTURED_OUTPUT_FAILED
    assert retry_event.type == EventType.STRUCTURED_OUTPUT_RETRY
    assert secret_value not in str(failed_event.payload)
    assert secret_value not in str(retry_event.payload)
    assert REDACTED_SECRET in failed_event.payload["errors"][0]["message"]

    repair_message = transcript[2].content[0].text
    assert "Return only valid JSON" in repair_message
    assert secret_value not in repair_message
    assert REDACTED_SECRET in repair_message
    assert secret_value not in str(provider.requests[1].messages[-1].model_dump(mode="json"))


def _native_answer_spec() -> StructuredOutputSpec:
    return StructuredOutputSpec(
        json_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
        strategy="native",
    )


def test_cayu_app_rejects_native_structured_output_for_unsupported_provider():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta('{"answer":"ok"}'),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    with pytest.raises(NativeStructuredOutputUnsupported, match="provider: fake") as excinfo:
        asyncio.run(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_structured_output_native_unsupported",
                    messages=[Message.text("user", "answer with structured output")],
                    structured_output=_native_answer_spec(),
                ),
            )
        )

    # Still a ValueError, so existing handlers (server 4xx mapping) keep working.
    assert isinstance(excinfo.value, ValueError)
    # Preflight contract: no session persisted, no model request made.
    assert asyncio.run(store.load("sess_structured_output_native_unsupported")) is None
    assert provider.requests == []


def test_cayu_app_rejects_native_structured_output_on_model_routed_provider():
    # Model-pattern routing can select a non-native provider purely by model
    # name, even when the default provider WOULD support native mode.
    class NativeDefaultProvider(NativeStructuredOutputFakeProvider):
        name = "native-default"

    store = InMemorySessionStore()
    routed = FakeProvider(
        [
            ModelStreamEvent.text_delta('{"answer":"ok"}'),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(NativeDefaultProvider([]), default=True)
    app.register_provider(routed, model_patterns=["fake-*"])
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    with pytest.raises(NativeStructuredOutputUnsupported, match="provider: fake"):
        asyncio.run(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_structured_output_native_routed",
                    messages=[Message.text("user", "answer with structured output")],
                    structured_output=_native_answer_spec(),
                ),
            )
        )

    assert asyncio.run(store.load("sess_structured_output_native_routed")) is None
    assert routed.requests == []


def test_cayu_app_resume_rejects_new_native_structured_output_as_profile_drift():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
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
                session_id="sess_native_resume",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    with pytest.raises(ExecutionProfileMismatchError) as caught:
        asyncio.run(
            collect_resume_events(
                app,
                ResumeRequest(
                    session_id="sess_native_resume",
                    messages=[Message.text("user", "again")],
                    structured_output=_native_answer_spec(),
                ),
            )
        )
    assert caught.value.changed_component_classes == (
        ExecutionProfileComponentClass.STRUCTURED_OUTPUT,
    )

    # Checked before the status transition: the session stays resumable.
    session = asyncio.run(store.load("sess_native_resume"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_cayu_app_rejects_native_structured_output_on_tool_approval():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_native_tool_approval",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_event = next(
        event for event in interrupt_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    approval_id = approval_event.payload["approval"]["approval_id"]

    # A paused invocation cannot adopt a new output contract implicitly.
    with pytest.raises(ExecutionProfileMismatchError) as caught:
        asyncio.run(
            collect_tool_approval_events(
                app,
                ToolApprovalRequest(
                    session_id="sess_native_tool_approval",
                    approval_id=approval_id,
                    tool_round_id=approval_event.payload["tool_round_id"],
                    tool_call_id=approval_event.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                    structured_output=_native_answer_spec(),
                ),
            )
        )
    assert caught.value.changed_component_classes == (
        ExecutionProfileComponentClass.STRUCTURED_OUTPUT,
    )

    session = asyncio.run(store.load("sess_native_tool_approval"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_cayu_app_rejects_native_structured_output_on_tool_approval_recovery():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_native_approval_recovery",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_event = next(
        event for event in interrupt_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    approval_id = approval_event.payload["approval"]["approval_id"]

    with pytest.raises(ExecutionProfileMismatchError) as caught:
        asyncio.run(
            collect_tool_approval_recovery_events(
                app,
                ToolApprovalRecoveryRequest(
                    session_id="sess_native_approval_recovery",
                    approval_id=approval_id,
                    tool_round_id=approval_event.payload["tool_round_id"],
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="side effect completed externally",
                    structured_output=_native_answer_spec(),
                ),
            )
        )
    assert caught.value.changed_component_classes == (
        ExecutionProfileComponentClass.STRUCTURED_OUTPUT,
    )

    session = asyncio.run(store.load("sess_native_approval_recovery"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_require_native_structured_output_support_helper():
    app = CayuApp(session_store=InMemorySessionStore())
    app.register_provider(FakeProvider([]), default=True)
    registered_provider = app._get_registered_provider("fake")

    # No spec and TOOL specs pass; only NATIVE on a non-native provider raises.
    _require_native_structured_output_support(None, registered_provider=registered_provider)
    _require_native_structured_output_support(
        StructuredOutputSpec(json_schema={"type": "object"}),
        registered_provider=registered_provider,
    )
    with pytest.raises(NativeStructuredOutputUnsupported):
        _require_native_structured_output_support(
            _native_answer_spec(), registered_provider=registered_provider
        )


def test_cayu_app_rejects_invalid_native_structured_output_schema_before_run():
    store = InMemorySessionStore()
    provider = SchemaRejectingProvider([])
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    with pytest.raises(
        NativeStructuredOutputSchemaInvalid, match="rejected by provider preflight"
    ) as excinfo:
        asyncio.run(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_native_schema_invalid",
                    messages=[Message.text("user", "answer with structured output")],
                    structured_output=_native_answer_spec(),
                ),
            )
        )

    # Still a ValueError, so existing handlers (server 4xx mapping) keep working.
    assert isinstance(excinfo.value, ValueError)
    # Preflight contract: no session persisted, no model request made.
    assert asyncio.run(store.load("sess_native_schema_invalid")) is None
    assert provider.requests == []


def test_cayu_app_rejects_openai_native_schema_before_first_model_call():
    # The issue-#151 acceptance path end-to-end: a schema OpenAI strict mode
    # always refuses (no `additionalProperties: false`) fails before any
    # session is created and before the transport is touched.
    class UnreachableTransport:
        def stream_response_events(self, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            raise AssertionError("preflight must reject before any model request")

        async def create_response(self, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError("preflight must reject before any model request")

    store = InMemorySessionStore()
    provider = OpenAIProvider(api_key="test-key", transport=UnreachableTransport())
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-test"))

    with pytest.raises(
        NativeStructuredOutputSchemaInvalid,
        match=r"\$: .*additionalProperties: false",
    ):
        asyncio.run(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_openai_native_schema_invalid",
                    messages=[Message.text("user", "answer with structured output")],
                    structured_output=_native_answer_spec(),
                ),
            )
        )

    assert asyncio.run(store.load("sess_openai_native_schema_invalid")) is None


def test_cayu_app_rejects_invalid_native_structured_output_schema_on_tool_approval():
    # The resolver restores the effective spec from the pending checkpoint and
    # reruns provider preflight before transitioning the session.
    class ApprovalSchemaProvider(NativeStructuredOutputFakeProvider):
        def __init__(self, events: list[ModelStreamEvent] | list[list[ModelStreamEvent]]) -> None:
            super().__init__(events)
            self.reject_schema = False

        def preflight_native_structured_output_schema(
            self,
            json_schema: dict[str, Any],
        ) -> None:
            del json_schema
            if self.reject_schema:
                raise NativeStructuredOutputSchemaInvalid("$: rejected by provider preflight.")

    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = ApprovalSchemaProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_native_schema_tool_approval",
                messages=[Message.text("user", "use the tool")],
                structured_output=_native_answer_spec(),
            ),
        )
    )
    approval_event = next(
        event for event in interrupt_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    approval_id = approval_event.payload["approval"]["approval_id"]

    provider.reject_schema = True
    with pytest.raises(NativeStructuredOutputSchemaInvalid, match="rejected by provider preflight"):
        asyncio.run(
            collect_tool_approval_events(
                app,
                ToolApprovalRequest(
                    session_id="sess_native_schema_tool_approval",
                    approval_id=approval_id,
                    tool_round_id=approval_event.payload["tool_round_id"],
                    tool_call_id=approval_event.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                ),
            )
        )

    session = asyncio.run(store.load("sess_native_schema_tool_approval"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_cayu_app_retries_structured_output_with_durable_repair_prompt():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("not json"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_final_fixed",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "fixed"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
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
                session_id="sess_structured_output_retry",
                messages=[Message.text("user", "answer with json")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    max_retries=1,
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_structured_output_retry"))
    interaction_record = asyncio.run(
        store.query_latest_interaction_events("sess_structured_output_retry", limit=1)
    )[0]

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.STRUCTURED_OUTPUT_RETRY,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[5].payload["errors"][0]["message"] == (
        f"Final structured output must be submitted with the `{STRUCTURED_OUTPUT_TOOL_NAME}` tool."
    )
    assert events[6].payload["attempt"] == 1
    assert events[11].payload["attempt"] == 2
    assert len(provider.requests) == 2
    assert [message.role for message in transcript] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
    ]
    repair_message = transcript[2].content[0].text
    assert STRUCTURED_OUTPUT_TOOL_NAME in repair_message
    assert "plain text" in repair_message
    assert "Validation errors:" in repair_message
    assert provider.requests[1].messages[-1].role == "user"
    assert provider.requests[1].messages[-1].content[0].text == repair_message
    from cayu.sessions.interactions import InteractionSummaryEvidence

    evidence = InteractionSummaryEvidence.model_validate(interaction_record.event.payload)
    assert evidence.model_step_count == 2
    transcript_records = asyncio.run(
        store.query_transcript(
            TranscriptQuery(
                session_id="sess_structured_output_retry",
                interaction_id=interaction_record.event.interaction_id,
            )
        )
    )
    assert len(transcript_records.records) == len(transcript)


def test_cayu_app_fails_structured_output_after_retries_exhausted():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("not json"),
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
                session_id="sess_structured_output_failed",
                messages=[Message.text("user", "answer with json")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                    max_retries=0,
                ),
            ),
        )
    )
    session = asyncio.run(store.load("sess_structured_output_failed"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error"] == (
        "Structured output validation failed after 1 attempt(s)."
    )
    assert session is not None
    assert session.status == SessionStatus.FAILED


def test_cayu_app_does_not_write_structured_output_repair_without_remaining_step():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("not json"),
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
                session_id="sess_structured_output_no_step_for_repair",
                messages=[Message.text("user", "answer with json")],
                max_steps=1,
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                    max_retries=1,
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_structured_output_no_step_for_repair"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert EventType.STRUCTURED_OUTPUT_RETRY not in [event.type for event in events]
    assert events[-1].payload["error"] == (
        "Structured output validation failed after 1 attempt(s): "
        "maximum model steps reached before repair."
    )
    assert [message.role for message in transcript] == ["user", "assistant"]
    assert transcript[-1].content[0].text == "not json"


def test_cayu_app_validates_structured_output_only_after_tool_round_finishes():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "from tool"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_final",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "from tool"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
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
                session_id="sess_structured_output_after_tool",
                messages=[Message.text("user", "use tool then answer with json")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    }
                ),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 2
    assert events[9].payload["output"] == {"answer": "from tool"}


def test_cayu_app_validates_native_structured_output_only_after_tool_round_finishes():
    store = InMemorySessionStore()
    provider = NativeStructuredOutputFakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "from tool"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta('{"answer":"from tool"}'),
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

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_native_structured_output_after_tool",
                messages=[Message.text("user", "use tool then answer with json")],
                structured_output=_native_answer_spec(),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 2
    assert provider.requests[0].options["structured_output"]["strategy"] == "native"
    assert provider.requests[1].options["structured_output"]["strategy"] == "native"
    assert events[9].payload["output"] == {"answer": "from tool"}
