"""Focused issue #529 model durability boundary tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from tests.core.test_runtime import (
    CountingProvider,
    FakeProvider,
    UsageDialectMutatingProvider,
    collect_events,
)

from cayu._validation import MAX_DURABLE_JSON_INTEGER, DurableValueError
from cayu.core import (
    AgentSpec,
    EventType,
    Message,
    ToolCallPart,
)
from cayu.providers import (
    InputTokenCountConfidence,
    InputTokenCountMethod,
    InputTokenCountResult,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
    UsageDialect,
)
from cayu.runtime import (
    BillingIdentity,
    BudgetWindow,
    CayuApp,
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    ContextCompactor,
    ContextCountingConfig,
    ContextCountingMode,
    EventQuery,
    InMemoryBudgetStore,
    InMemorySessionStore,
    RetryPolicy,
    RunRequest,
)
from cayu.runtime._event_projection import public_event_sequence


def test_context_counting_oversized_result_fails_at_boundary_without_blocking_model():
    forged_count = InputTokenCountResult.model_construct(
        input_tokens=MAX_DURABLE_JSON_INTEGER + 1,
        method=InputTokenCountMethod.OFFICIAL,
        confidence=InputTokenCountConfidence.HIGH,
        components={},
        metadata={},
    )
    provider = CountingProvider(
        [ModelStreamEvent.completed({"finish_reason": "stop"})],
        count_result=forged_count,
    )
    app = CayuApp(
        context_counting=ContextCountingConfig(mode=ContextCountingMode.OBSERVE),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_context_counting_oversized",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    event_types = [event.type for event in events]
    assert EventType.CONTEXT_COUNT_FAILED in event_types
    assert EventType.CONTEXT_COUNTED not in event_types
    assert EventType.MODEL_COMPLETED in event_types
    assert len(provider.requests) == 1


@pytest.mark.parametrize(
    "count_error",
    [
        ModelProviderError(
            "counter\x00workload-secret-value",
            provider="fake",
            retryable=True,
        ),
        TimeoutError("counter\x00workload-secret-value"),
    ],
    ids=["typed", "generic"],
)
def test_context_counting_non_portable_provider_error_is_safe_and_non_blocking(
    count_error: Exception,
) -> None:
    provider = CountingProvider(
        [ModelStreamEvent.completed({"finish_reason": "stop"})],
        count_error=count_error,
    )
    app = CayuApp(
        context_counting=ContextCountingConfig(mode=ContextCountingMode.OBSERVE),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_context_counting_nonportable_failure",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    failed = next(event for event in events if event.type == EventType.CONTEXT_COUNT_FAILED)
    assert failed.payload["error"] == ("Model provider emitted a non-portable error value.")
    assert failed.payload["durable_value_error_code"] == "nul_character"
    assert EventType.MODEL_COMPLETED in [event.type for event in events]
    rendered = json.dumps([event.model_dump(mode="json") for event in events])
    assert "workload-secret-value" not in rendered


def test_cayu_app_rejects_non_portable_request_billing_error_before_dispatch():
    secret = "billing\x00workload-secret-value"

    class NonPortableRequestBillingProvider(FakeProvider):
        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity:
            del request
            raise RuntimeError(secret)

    provider = NonPortableRequestBillingProvider([ModelStreamEvent.completed({})])
    app = CayuApp(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_nonportable_request_billing_error",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert provider.requests == []
    model_error = next(event for event in events if event.type == EventType.MODEL_ERROR)
    assert model_error.payload["stage"] == "billing_identity_for_request"
    assert model_error.payload["provider_error_code"] == "invalid_model_provider_error"
    assert model_error.payload["retryable"] is False
    rendered = json.dumps([event.model_dump(mode="json") for event in events])
    assert "workload-secret-value" not in rendered
    assert events[-1].type == EventType.SESSION_FAILED


@pytest.mark.parametrize(
    "invalid_location",
    ["top-level", "usage", "model-and-usage"],
)
def test_cayu_app_terminalizes_non_portable_completion_with_fail_closed_usage(
    invalid_location: str,
):
    secret_key = "workload-secret-key"
    usage_payload: dict[str, Any] = {"input_tokens": 30, "output_tokens": 4}
    completion_payload: dict[str, Any] = {
        "model": "fake-model",
        "usage": usage_payload,
    }
    if invalid_location == "top-level":
        completion_payload[secret_key] = float("nan")
    elif invalid_location == "usage":
        usage_payload[secret_key] = "invalid\x00value"
    else:
        completion_payload["model"] = "invalid\x00model"
        usage_payload[secret_key] = "invalid\x00value"

    invalid_completion = ModelStreamEvent.model_construct(
        type=ModelStreamEventType.COMPLETED,
        delta="",
        payload=completion_payload,
        completion=None,
    )
    provider = FakeProvider([invalid_completion])
    budget_store = InMemoryBudgetStore()
    app = CayuApp(
        budget_store=budget_store,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_invalid_model_completion_{invalid_location}",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert EventType.MODEL_RETRY not in [event.type for event in events]
    assert EventType.MODEL_ERROR not in [event.type for event in events]
    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    if invalid_location == "top-level":
        assert completed.payload["usage_metrics"]["input_tokens"] == 30
        assert completed.payload["usage_metrics"]["output_tokens"] == 4
        assert completed.payload["usage"] == {"input_tokens": 30, "output_tokens": 4}
        assert "usage_normalization_failed" not in completed.payload
    else:
        assert "usage" not in completed.payload
        assert "usage_metrics" not in completed.payload
        assert completed.payload["usage_normalization_failed"] is True
        assert completed.payload["usage_unavailable_reason"] == (
            "invalid model completion usage telemetry"
        )
    assert completed.payload["model"] == "fake-model"
    assert completed.payload["completion_outcome"] == "invalid_metadata"
    assert completed.payload["completion_error"]["provider_error_code"] == (
        "invalid_model_completion_value"
    )
    rendered = json.dumps(
        [event.model_dump(mode="json") for event in events],
        allow_nan=False,
    )
    assert secret_key not in rendered
    assert "invalid\u0000value" not in rendered
    usage = asyncio.run(app.get_session_usage(f"sess_invalid_model_completion_{invalid_location}"))
    assert usage.model_steps == 1
    assert usage.usage.total_tokens == (34 if invalid_location == "top-level" else 0)
    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    sequence = public_event_sequence(completed.id)
    assert sequence is not None
    durable_records = asyncio.run(
        app.session_store.query_events(
            EventQuery(session_id=f"sess_invalid_model_completion_{invalid_location}")
        )
    )
    durable_completed = next(
        record.event for record in durable_records if record.sequence == sequence
    )
    assert [event.id for event in budget_events] == [durable_completed.id]
    assert events[-1].type == EventType.SESSION_FAILED


@pytest.mark.parametrize(
    "provider_state",
    [
        {},
        ["not-an-object"],
        [{"provider": 7, "state": {}}],
        [{"provider": "fake", "state": []}],
    ],
    ids=["not-a-list", "part-not-an-object", "provider-not-text", "state-not-an-object"],
)
def test_cayu_app_preserves_completed_usage_when_provider_state_is_structurally_invalid(
    provider_state: object,
) -> None:
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": "fake-model",
                    "usage": {
                        "input_tokens": 7,
                        "output_tokens": 3,
                        "total_tokens": 10,
                    },
                    "provider_state": provider_state,
                }
            )
        ]
    )
    app = CayuApp(
        session_store=store,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    session_id = "sess_invalid_provider_state"
    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert EventType.MODEL_RETRY not in [event.type for event in events]
    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["usage_metrics"]["input_tokens"] == 7
    assert completed.payload["usage_metrics"]["output_tokens"] == 3
    assert completed.payload["usage_metrics"]["total_tokens"] == 10
    assert completed.payload["completion_outcome"] == "invalid_transcript_state"
    assert completed.payload["completion_error"]["provider_error_code"] == (
        "invalid_model_completion_transcript"
    )
    assert completed.payload["transcript_cursor"] == 1
    assert "provider_state" not in completed.payload

    usage = asyncio.run(app.get_session_usage(session_id))
    assert usage.model_steps == 1
    assert usage.usage.input_tokens == 7
    assert usage.usage.output_tokens == 3
    assert usage.usage.total_tokens == 10
    transcript = asyncio.run(store.load_transcript(session_id))
    assert transcript == [Message.text("user", "hello")]
    assert events[-1].type == EventType.SESSION_FAILED


def test_cayu_app_does_not_invoke_hostile_completion_key_equality_or_redispatch():
    rejected_key = "provider-owned-secret-key"

    class HostileKey(str):
        __hash__ = str.__hash__

        def __eq__(self, other: object) -> bool:
            del other
            raise TimeoutError("provider-owned key equality must not run")

    completion_payload: dict[Any, Any] = {
        "model": "fake-model",
        "usage": {"input_tokens": 30, "output_tokens": 4},
    }
    completion_payload[HostileKey(rejected_key)] = 1
    invalid_completion = ModelStreamEvent.model_construct(
        type=ModelStreamEventType.COMPLETED,
        delta="",
        payload=completion_payload,
        completion=None,
    )
    provider = FakeProvider([invalid_completion])
    app = CayuApp(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_completion_key_projection",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert EventType.MODEL_RETRY not in [event.type for event in events]
    assert EventType.MODEL_ERROR not in [event.type for event in events]
    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["usage_metrics"]["input_tokens"] == 30
    assert completed.payload["usage_metrics"]["output_tokens"] == 4
    assert completed.payload["completion_outcome"] == "invalid_metadata"
    assert events[-1].type == EventType.SESSION_FAILED
    rendered = json.dumps([event.model_dump(mode="json") for event in events])
    assert rejected_key not in rendered


def test_cayu_app_uses_registration_time_usage_dialect_after_provider_mutation():
    provider = UsageDialectMutatingProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": "fake-model",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 5,
                    },
                }
            )
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_registered_usage_dialect",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert provider.usage_dialect == UsageDialect.GENERIC
    assert completed.payload["usage_metrics"]["input_tokens"] == 15
    assert completed.payload["usage_metrics"]["total_tokens"] == 16


def test_cayu_app_rejects_provider_owned_usage_dialect_string_subclasses():
    class HostileDialect(str):
        def strip(self, *args, **kwargs):
            raise AssertionError("provider-owned string methods must not be invoked")

        def lower(self):
            raise AssertionError("provider-owned string methods must not be invoked")

    provider = FakeProvider([])
    provider.usage_dialect = HostileDialect("anthropic")  # type: ignore[assignment]

    with pytest.raises(TypeError, match="provider.usage_dialect"):
        CayuApp().register_provider(provider)


def test_cayu_app_does_not_redispatch_when_derived_usage_exceeds_portable_range():
    provider = FakeProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": "fake-model",
                    "usage": {
                        "input_tokens": MAX_DURABLE_JSON_INTEGER,
                        "output_tokens": MAX_DURABLE_JSON_INTEGER,
                    },
                }
            )
        ]
    )
    app = CayuApp(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_derived_usage_out_of_range",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert EventType.MODEL_RETRY not in [event.type for event in events]
    assert EventType.MODEL_ERROR not in [event.type for event in events]
    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["rejected_usage_evidence"] == {
        "input_tokens": MAX_DURABLE_JSON_INTEGER,
        "output_tokens": MAX_DURABLE_JSON_INTEGER,
    }
    assert completed.payload["usage_metrics_rejected"] is True
    assert "usage" not in completed.payload
    assert "usage_metrics" not in completed.payload
    assert completed.payload["completion_outcome"] == "invalid_metadata"
    assert completed.payload["completion_error"]["durable_value_error_code"] == (
        "integer_out_of_range"
    )
    assert events[-1].type == EventType.SESSION_FAILED


@pytest.mark.parametrize(
    ("case", "usage_payload", "usage_dialect", "expected_evidence"),
    [
        (
            "total",
            {
                "input_tokens": MAX_DURABLE_JSON_INTEGER,
                "output_tokens": MAX_DURABLE_JSON_INTEGER,
                "workload-secret-key": "invalid\x00value",
            },
            UsageDialect.AUTO,
            {
                "input_tokens": MAX_DURABLE_JSON_INTEGER,
                "output_tokens": MAX_DURABLE_JSON_INTEGER,
            },
        ),
        (
            "anthropic-cache",
            {
                "input_tokens": MAX_DURABLE_JSON_INTEGER,
                "output_tokens": 0,
                "cache_read_input_tokens": 1,
                "workload-secret-key": "invalid\x00value",
            },
            UsageDialect.AUTO,
            {
                "input_tokens": MAX_DURABLE_JSON_INTEGER,
                "output_tokens": 0,
                "cache_read_input_tokens": 1,
            },
        ),
        (
            "cache-details",
            {
                "input_tokens": 1,
                "cache_details": [
                    {"input_tokens": MAX_DURABLE_JSON_INTEGER, "ttl": "5m"},
                    {"input_tokens": 1, "ttl": "5m"},
                ],
                "workload-secret-key": "invalid\x00value",
            },
            UsageDialect.ANTHROPIC,
            {
                "input_tokens": 1,
                "cache_details": [
                    {"input_tokens": MAX_DURABLE_JSON_INTEGER, "ttl": "5m"},
                    {"input_tokens": 1, "ttl": "5m"},
                ],
            },
        ),
    ],
)
def test_cayu_app_retains_safe_usage_evidence_when_malformed_usage_aggregate_overflows(
    case: str,
    usage_payload: dict[str, Any],
    usage_dialect: UsageDialect,
    expected_evidence: dict[str, Any],
) -> None:
    invalid_completion = ModelStreamEvent.model_construct(
        type=ModelStreamEventType.COMPLETED,
        delta="",
        payload={"model": "fake-model", "usage": usage_payload},
        completion=None,
    )
    provider = FakeProvider([invalid_completion])
    provider.usage_dialect = usage_dialect
    app = CayuApp(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_malformed_usage_{case}",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert EventType.MODEL_RETRY not in [event.type for event in events]
    assert EventType.MODEL_ERROR not in [event.type for event in events]
    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["rejected_usage_evidence"] == expected_evidence
    assert completed.payload["usage_metrics_rejected"] is True
    assert "usage" not in completed.payload
    assert "usage_metrics" not in completed.payload
    assert completed.payload["completion_outcome"] == "invalid_metadata"
    rendered = json.dumps([event.model_dump(mode="json") for event in events])
    assert "workload-secret-key" not in rendered
    assert "invalid\\u0000value" not in rendered
    assert events[-1].type == EventType.SESSION_FAILED


def test_cayu_app_terminalizes_nonportable_model_stream_values_without_retry_or_leak(
    caplog: pytest.LogCaptureFixture,
):
    invalid_delta = "timeout\x00workload-secret-value"
    secret_key = "workload-secret-key"
    secret_value = "workload-secret-payload-value"

    class NonPortableStreamProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) != 1:
                raise AssertionError("Non-portable stream values must not be retried.")
            yield ModelStreamEvent(
                type=ModelStreamEventType.TEXT_DELTA,
                delta=invalid_delta,
                payload={secret_key: secret_value},
            )

    store = InMemorySessionStore()
    provider = NonPortableStreamProvider()
    app = CayuApp(
        session_store=store,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_nonportable_model_stream",
                messages=[Message.text("user", "hi")],
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
    assert len(provider.requests) == 1
    assert EventType.MODEL_RETRY not in [event.type for event in events]
    assert EventType.MODEL_ATTEMPT_DISCARDED not in [event.type for event in events]
    assert events[2].payload == {
        "error": "Model provider emitted a non-portable stream value.",
        "error_type": "ModelProviderError",
        "stage": "model_stream_validation",
        "durable_value_error_code": "nul_character",
        "durable_value_path": "$",
        "provider": "fake",
        "provider_error_type": "DurableValueError",
        "provider_error_code": "invalid_model_stream_value",
        "retryable": False,
        "step": 1,
        "attempt": 1,
        "max_attempts": 2,
        "model_step_id": events[1].payload["model_step_id"],
        "model_attempt_id": events[1].payload["model_attempt_id"],
    }
    rendered = json.dumps(
        [event.model_dump(mode="json") for event in events],
        ensure_ascii=True,
    )
    assert "timeout" not in rendered
    assert secret_key not in rendered
    assert secret_value not in rendered
    assert "timeout" not in caplog.text
    assert secret_key not in caplog.text
    assert secret_value not in caplog.text


@pytest.mark.parametrize(
    "raised_error",
    [
        ModelProviderError(
            "timeout\x00workload-secret-value",
            provider="fake",
            status_code=503,
            retryable=True,
        ),
        TimeoutError("timeout\x00workload-secret-value"),
    ],
    ids=["typed", "generic"],
)
def test_cayu_app_terminalizes_nonportable_raised_provider_error_without_retry_or_leak(
    caplog: pytest.LogCaptureFixture,
    raised_error: Exception,
):
    class NonPortableErrorProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            raise raised_error
            yield

    provider = NonPortableErrorProvider()
    app = CayuApp(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_nonportable_raised_provider_error",
                messages=[Message.text("user", "hi")],
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
    assert len(provider.requests) == 1
    assert events[2].payload == {
        "error": "Model provider emitted a non-portable error value.",
        "error_type": "ModelProviderError",
        "stage": "model_stream_validation",
        "durable_value_error_code": "nul_character",
        "durable_value_path": "$",
        "provider": "fake",
        "provider_error_type": "DurableValueError",
        "provider_error_code": "invalid_model_provider_error",
        "retryable": False,
        "step": 1,
        "attempt": 1,
        "max_attempts": 2,
        "model_step_id": events[1].payload["model_step_id"],
        "model_attempt_id": events[1].payload["model_attempt_id"],
    }
    assert EventType.MODEL_RETRY not in [event.type for event in events]
    assert EventType.MODEL_ATTEMPT_DISCARDED not in [event.type for event in events]
    rendered = json.dumps([event.model_dump(mode="json") for event in events])
    assert "timeout" not in rendered
    assert "workload-secret-value" not in rendered
    assert "timeout" not in caplog.text
    assert "workload-secret-value" not in caplog.text


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "expected_error_code"),
    [
        ("error_code", "transient\x00workload-secret-value", "nul_character"),
        ("retry_after_s", float("nan"), "non_finite_number"),
    ],
    ids=["structured-string", "retry-after-nan"],
)
def test_cayu_app_validates_raised_provider_error_control_fields_before_retry(
    field_name: str,
    invalid_value: object,
    expected_error_code: str,
):
    class MutatedProviderErrorProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.error = ModelProviderError(
                "provider overloaded",
                provider=self.name,
                status_code=503,
                retryable=True,
            )
            setattr(self.error, field_name, invalid_value)

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            raise self.error
            yield

    provider = MutatedProviderErrorProvider()
    app = CayuApp(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_nonportable_provider_error_{field_name}",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert EventType.MODEL_RETRY not in [event.type for event in events]
    model_error = next(event for event in events if event.type == EventType.MODEL_ERROR)
    assert model_error.payload["durable_value_error_code"] == expected_error_code
    assert model_error.payload["provider_error_code"] == "invalid_model_provider_error"
    rendered = json.dumps(
        [event.model_dump(mode="json") for event in events],
        allow_nan=False,
    )
    assert "workload-secret-value" not in rendered


def test_cayu_app_uses_one_detached_provider_error_snapshot_per_attempt(
    caplog: pytest.LogCaptureFixture,
):
    class FlippingError(ModelProviderError):
        def __init__(self) -> None:
            super().__init__(
                "provider overloaded",
                provider="fake",
                status_code=503,
                retryable=True,
            )
            self.render_count = 0

        def __str__(self) -> str:
            self.render_count += 1
            if self.render_count == 1:
                return "provider overloaded"
            return "timeout\x00workload-secret-value"

    class FlippingErrorProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.error = FlippingError()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            raise self.error
            yield

    provider = FlippingErrorProvider()
    app = CayuApp(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_flipping_provider_error",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert len(provider.requests) == 2
    model_errors = [event for event in events if event.type == EventType.MODEL_ERROR]
    assert model_errors[0].payload["error"] == "provider overloaded"
    assert model_errors[1].payload["error"] == (
        "Model provider emitted a non-portable error value."
    )
    assert [event.type for event in events].count(EventType.MODEL_RETRY) == 1
    assert [event.type for event in events].count(EventType.MODEL_ATTEMPT_DISCARDED) == 1
    rendered = json.dumps([event.model_dump(mode="json") for event in events])
    assert "timeout" not in rendered
    assert "workload-secret-value" not in rendered
    assert "ValidationError" not in rendered
    assert "timeout" not in caplog.text
    assert "workload-secret-value" not in caplog.text


def test_cayu_app_does_not_publish_forged_durable_value_diagnostics():
    class ForgedDiagnosticProvider(ModelProvider):
        name = "fake"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            error = DurableValueError(
                "nul_character",
                "payload",
                path="$/workload-secret-path",
            )
            error.code = "workload-secret-code"
            raise error
            yield

    app = CayuApp(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
    app.register_provider(ForgedDiagnosticProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_forged_durable_diagnostic",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    error = next(event for event in events if event.type == EventType.MODEL_ERROR)
    assert error.payload["durable_value_error_code"] == "invalid_json_type"
    assert error.payload["durable_value_path"] == "$"
    assert "workload-secret" not in json.dumps(error.payload)


def test_cayu_app_detaches_nested_message_payloads_between_provider_attempts():
    class MutatingTimeoutProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            call = request.messages[0].content[0]
            assert isinstance(call, ToolCallPart)
            if len(self.requests) == 1:
                call.arguments["nested"]["value"] = "provider mutation"
                raise TimeoutError("stream idle timeout")
            assert call.arguments == {"nested": {"value": "original"}}
            yield ModelStreamEvent.text_delta("ok")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    provider = MutatingTimeoutProvider()
    app = CayuApp(
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_detached_provider_attempts",
                messages=[
                    Message.tool_call(
                        tool_call_id="call_1",
                        tool_name="echo",
                        arguments={"nested": {"value": "original"}},
                    ),
                    Message.tool_result(
                        tool_call_id="call_1",
                        tool_name="echo",
                        content="done",
                    ),
                    Message.text("user", "continue"),
                ],
            ),
        )
    )

    assert len(provider.requests) == 2
    assert provider.requests[0].messages[0].content[0].arguments == {
        "nested": {"value": "provider mutation"}
    }
    assert provider.requests[1].messages[0].content[0].arguments == {
        "nested": {"value": "original"}
    }
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_cayu_app_rejects_hostile_exact_result_field_keys_without_lookup() -> None:
    class HostileFieldKey(str):
        equality_calls = 0

        def __hash__(self) -> int:
            return str.__hash__(self)

        def __eq__(self, other: object) -> bool:
            del other
            type(self).equality_calls += 1
            raise AssertionError("result field-key equality must not execute")

    result = CompactionResult(summary="summary", covered_message_count=2)
    fields = dict(object.__getattribute__(result, "__dict__"))
    summary = fields.pop("summary")
    fields[HostileFieldKey("summary")] = summary
    object.__setattr__(result, "__dict__", fields)

    class HostileFieldResultCompactor(ContextCompactor):
        def provider_budget_identity(self, _session) -> None:
            return None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            del request
            return result

    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp()
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=HostileFieldResultCompactor(),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_hostile_compaction_result_fields",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert HostileFieldKey.equality_calls == 0
    assert all(event.type != EventType.MODEL_COMPLETED for event in events)
    failed = next(event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED)
    assert failed.payload["error_type"] == "TypeError"
    assert events[-1].type == EventType.SESSION_FAILED
    assert "must return CompactionResult" in events[-1].payload["error"]
    assert runtime_provider.requests == []
