"""Focused issue #529 compaction durability boundary tests."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator
from decimal import Decimal

import pytest
from pydantic import ValidationError
from tests.core.test_runtime import (
    CompactionIdentityMutatingProvider,
    CompletedThenRetryableCompactionProvider,
    FakeProvider,
    FlakyCompactionProvider,
    UsageDialectMutatingProvider,
    _test_session,
    collect_events,
    compaction_price_book,
)

from cayu._validation import MAX_DURABLE_JSON_INTEGER, DurableValueError
from cayu.core import (
    AgentSpec,
    EventType,
    Message,
    MessageRole,
    TextPart,
)
from cayu.providers import (
    ModelContextOverflowError,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    UsageDialect,
)
from cayu.runtime import (
    BillingIdentity,
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    CayuApp,
    CheckpointCompactionContextPolicy,
    CompactionPrompt,
    CompactionRequest,
    CompactionResult,
    ContextCompactor,
    ModelCompactor,
    PromptCacheCompactor,
    RequestFootprintConfig,
    RetryPolicy,
    RunRequest,
    Session,
)


@pytest.mark.parametrize("compactor_kind", ["model", "prompt_cache"])
def test_provider_backed_compactors_keep_construction_time_usage_dialect(
    compactor_kind: str,
):
    provider = UsageDialectMutatingProvider(
        [
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 5,
                    },
                }
            ),
        ]
    )
    context = [Message.text("user", "old request")]
    if compactor_kind == "model":
        compactor: ContextCompactor = ModelCompactor(
            provider=provider,
            model="summary-model",
        )
        request = CompactionRequest(
            session=_test_session(),
            agent=AgentSpec(name="assistant", model="fake-model"),
            messages=context,
        )
    else:
        compactor = PromptCacheCompactor(provider=provider, model="summary-model")
        request = CompactionRequest(
            session=_test_session(),
            agent=AgentSpec(name="assistant", model="fake-model"),
            messages=context,
            context_messages=context,
            cache_prefix_request=ModelRequest(model="fake-model", messages=context),
        )

    result = asyncio.run(compactor.compact(request))

    assert provider.usage_dialect == UsageDialect.GENERIC
    assert result.model_completed_payloads[0]["usage_metrics"]["input_tokens"] == 15
    assert result.model_completed_payloads[0]["usage_metrics"]["total_tokens"] == 16


@pytest.mark.parametrize("compactor_kind", ["model", "prompt_cache"])
def test_provider_backed_compactors_keep_construction_time_provider_identity(
    compactor_kind: str,
) -> None:
    provider = CompactionIdentityMutatingProvider(
        [
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 5,
                    },
                }
            ),
        ]
    )
    context = [Message.text("user", "old request")]
    session = Session(
        id="sess_compactor_identity_snapshot",
        agent_name="assistant",
        provider_name="gateway",
        model="fake-model",
    )
    if compactor_kind == "model":
        compactor: ContextCompactor = ModelCompactor(
            provider=provider,
            model="summary-model",
        )
        request = CompactionRequest(
            session=session,
            agent=AgentSpec(name="assistant", model="fake-model"),
            messages=context,
        )
    else:
        compactor = PromptCacheCompactor(provider=provider, model="summary-model")
        request = CompactionRequest(
            session=session,
            agent=AgentSpec(name="assistant", model="fake-model"),
            messages=context,
            context_messages=context,
            cache_prefix_request=ModelRequest(model="fake-model", messages=context),
        )

    result = asyncio.run(compactor.compact(request))

    assert len(provider.requests) == 1
    assert provider.name == "poisoned\x00provider"
    assert provider.billing_provider_name == "poisoned\ud800billing"
    completed = result.model_completed_payloads[0]
    assert completed["provider_name"] == "billco"
    assert completed["usage_metrics"]["provider_name"] == "billco"
    assert completed["usage_metrics"]["input_tokens"] == 15
    assert completed["usage_metrics"]["total_tokens"] == 16


@pytest.mark.parametrize("compactor_kind", ["model", "prompt_cache"])
def test_provider_backed_compactors_freeze_invocation_identity_across_dispatch(
    compactor_kind: str,
) -> None:
    class CompactorMutatingProvider(ModelProvider):
        name = "gateway"
        billing_provider_name = "billco"
        usage_dialect = UsageDialect.ANTHROPIC

        def __init__(self) -> None:
            self.compactor: ModelCompactor | PromptCacheCompactor | None = None
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            assert self.compactor is not None
            self.compactor.model = "poisoned-model"
            self.compactor._compactor_name = "poisoned\x00compactor"
            self.compactor._provider_snapshot = object()  # type: ignore[assignment]
            self.compactor._usage_dialect = UsageDialect.GENERIC
            self.compactor.options = {"poisoned": True}
            if isinstance(self.compactor, PromptCacheCompactor):
                self.compactor.compaction_instruction = "poisoned\x00instruction"
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 5,
                    },
                }
            )

    provider = CompactorMutatingProvider()
    messages = [Message.text("user", "old request")]
    session = Session(
        id="sess_compactor_invocation_identity",
        agent_name="assistant",
        provider_name="gateway",
        model="fake-model",
    )
    if compactor_kind == "model":
        compactor: ModelCompactor | PromptCacheCompactor = ModelCompactor(
            provider=provider,
            model="summary-model",
        )
        request = CompactionRequest(
            session=session,
            agent=AgentSpec(name="assistant", model="fake-model"),
            messages=messages,
        )
        expected_model = "summary-model"
        expected_compactor = "ModelCompactor"
    else:
        compactor = PromptCacheCompactor(provider=provider)
        request = CompactionRequest(
            session=session,
            agent=AgentSpec(name="assistant", model="fake-model"),
            messages=messages,
            context_messages=messages,
            cache_prefix_request=ModelRequest(model="fake-model", messages=messages),
        )
        expected_model = "fake-model"
        expected_compactor = "PromptCacheCompactor"
    provider.compactor = compactor

    result = asyncio.run(compactor.compact(request))

    assert [request.model for request in provider.requests] == [expected_model]
    completed = result.model_completed_payloads[0]
    assert completed["provider_name"] == "billco"
    assert completed["requested_model"] == expected_model
    assert completed["compactor"] == expected_compactor
    assert completed["usage_metrics"]["input_tokens"] == 15
    assert completed["usage_metrics"]["total_tokens"] == 16


@pytest.mark.parametrize("compactor_type", [ModelCompactor, PromptCacheCompactor])
@pytest.mark.parametrize(
    ("attribute", "invalid_value", "expected_code"),
    [
        ("name", "gateway\x00invalid", "nul_character"),
        ("billing_provider_name", "billco\ud800invalid", "unicode_surrogate"),
    ],
)
def test_provider_backed_compactors_reject_nonportable_identity_at_construction(
    compactor_type: type[ModelCompactor] | type[PromptCacheCompactor],
    attribute: str,
    invalid_value: str,
    expected_code: str,
) -> None:
    provider = FakeProvider([ModelStreamEvent.completed({})])
    setattr(provider, attribute, invalid_value)

    with pytest.raises(DurableValueError) as exc_info:
        compactor_type(provider=provider, model="summary-model")

    assert exc_info.value.code == expected_code
    assert provider.requests == []


@pytest.mark.parametrize("compactor_type", [ModelCompactor, PromptCacheCompactor])
def test_provider_backed_compactors_do_not_execute_hostile_identity_methods(
    compactor_type: type[ModelCompactor] | type[PromptCacheCompactor],
) -> None:
    class HostileIdentity(str):
        def __bool__(self) -> bool:
            raise AssertionError("identity truthiness must not execute")

        def __eq__(self, other: object) -> bool:
            del other
            raise AssertionError("identity equality must not execute")

        def strip(self, chars: str | None = None) -> str:
            del chars
            raise AssertionError("identity stripping must not execute")

    provider = FakeProvider([ModelStreamEvent.completed({})])
    provider.billing_provider_name = HostileIdentity("billco")

    with pytest.raises(DurableValueError) as exc_info:
        compactor_type(provider=provider, model="summary-model")

    assert exc_info.value.code == "invalid_text_type"
    assert provider.requests == []


def test_model_compactor_retries_transient_generic_provider_errors():
    provider = FlakyCompactionProvider(
        failures=1,
        error=TimeoutError("stream idle timeout"),
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
    )

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old request")],
            )
        )
    )

    assert result.summary == "recovered summary"
    assert len(provider.requests) == 2


def test_model_compactor_detaches_billing_and_retry_requests_from_logical_request():
    class MutatingProvider(ModelProvider):
        name = "mutating-compactor"

        def __init__(self) -> None:
            self.billing_request: ModelRequest | None = None
            self.requests: list[ModelRequest] = []
            self.attempt_inputs: list[tuple[str, str, str]] = []

        @staticmethod
        def _user_text(request: ModelRequest) -> str:
            part = request.messages[-1].content[0]
            assert type(part) is TextPart
            return part.text

        @staticmethod
        def _mutate(request: ModelRequest, marker: str) -> None:
            nested = request.options["nested"]
            assert type(nested) is dict
            nested["value"] = marker
            request.messages[-1] = Message.text(MessageRole.USER, marker)
            request.model = marker

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            self.billing_request = request
            self._mutate(request, "billing-hook-mutation")
            return None

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            nested = request.options["nested"]
            assert type(nested) is dict
            self.attempt_inputs.append((request.model, nested["value"], self._user_text(request)))
            if len(self.requests) == 1:
                self._mutate(request, "first-attempt-mutation")
                raise ModelProviderError(
                    "retry this attempt",
                    provider=self.name,
                    status_code=503,
                    retryable=True,
                )
            yield ModelStreamEvent.text_delta("isolated summary")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    provider = MutatingProvider()
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        options={"nested": {"value": "original"}},
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
    )

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old request")],
            )
        )
    )

    assert result.summary == "isolated summary"
    assert provider.billing_request is not None
    assert len(provider.requests) == 2
    assert provider.billing_request is not provider.requests[0]
    assert provider.requests[0] is not provider.requests[1]
    assert provider.billing_request.options["nested"] is not provider.requests[0].options["nested"]
    assert provider.requests[0].options["nested"] is not provider.requests[1].options["nested"]
    first_prompt = provider.attempt_inputs[0][2]
    assert provider.attempt_inputs == [
        ("summary-model", "original", first_prompt),
        ("summary-model", "original", first_prompt),
    ]
    assert provider.billing_request.model == "billing-hook-mutation"
    assert provider.requests[0].model == "first-attempt-mutation"
    assert provider.requests[1].model == "summary-model"
    assert "billing-hook-mutation" not in first_prompt
    assert "first-attempt-mutation" not in first_prompt


def test_cayu_app_does_not_redispatch_compaction_when_derived_usage_overflows():
    compactor_provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("completed summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "summary-model",
                        "usage": {
                            "input_tokens": MAX_DURABLE_JSON_INTEGER,
                            "output_tokens": MAX_DURABLE_JSON_INTEGER,
                        },
                        "usage_metrics": {"input_tokens": 1, "total_tokens": 1},
                        "usage_metrics_rejected": False,
                        "rejected_usage_evidence": {"spoofed": True},
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("must not retry"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
            ],
        ]
    )
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp()
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
                retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_derived_usage_overflow",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []
    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["rejected_usage_evidence"] == {
        "input_tokens": MAX_DURABLE_JSON_INTEGER,
        "output_tokens": MAX_DURABLE_JSON_INTEGER,
    }
    assert completed.payload["usage_metrics_rejected"] is True
    assert "usage" not in completed.payload
    assert "usage_metrics" not in completed.payload
    assert completed.payload["compaction_outcome"] == "invalid_completion_metadata"
    failed = next(event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED)
    assert failed.payload["error_type"] == "DurableValueError"
    assert "error" not in failed.payload
    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].payload == {
        "error": "Operation failed with a non-portable diagnostic.",
        "error_type": "DurableValueError",
        "durable_value_error_code": "integer_out_of_range",
        "durable_value_error_path": "$",
    }


def test_model_compactor_rejects_non_portable_provider_error_without_retry_or_cause():
    secret = "timeout\x00workload-secret-value"
    provider = FlakyCompactionProvider(
        failures=3,
        error=ModelProviderError(
            secret,
            provider="flaky",
            status_code=503,
            retryable=True,
        ),
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        retry_policy=RetryPolicy(max_attempts=3, initial_delay_s=0.0),
    )

    with pytest.raises(ModelProviderError) as exc_info:
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )

    assert len(provider.requests) == 1
    assert str(exc_info.value) == "Model provider emitted a non-portable error value."
    assert exc_info.value.retryable is False
    assert exc_info.value.error_code == "invalid_model_provider_error"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "timeout" not in repr(exc_info.value)
    assert "workload-secret-value" not in repr(exc_info.value)


def test_model_compactor_rejects_non_portable_generic_error_without_retry_or_cause():
    provider = FlakyCompactionProvider(
        failures=3,
        error=TimeoutError("timeout\x00workload-secret-value"),
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        retry_policy=RetryPolicy(max_attempts=3, initial_delay_s=0.0),
    )

    with pytest.raises(ModelProviderError) as exc_info:
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )

    assert len(provider.requests) == 1
    assert str(exc_info.value) == "Model provider emitted a non-portable error value."
    assert exc_info.value.retryable is False
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "workload-secret-value" not in repr(exc_info.value)


def test_model_compactor_detaches_provider_exception_group_children():
    secret = "workload-secret-value"
    original = ExceptionGroup(
        "provider failures",
        [RuntimeError(secret), ValueError("invalid response")],
    )
    provider = FlakyCompactionProvider(failures=1, error=original)
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
    )

    with pytest.raises(ExceptionGroup) as exc_info:
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )

    detached = exc_info.value
    assert detached is not original
    assert detached.__cause__ is None
    assert detached.__context__ is None
    assert len(detached.exceptions) == 1
    assert str(detached.exceptions[0]) == "Provider exception details were detached."
    assert secret not in repr(detached)


def test_model_compactor_snapshots_generic_provider_error_once_per_attempt():
    class FlippingTimeoutError(TimeoutError):
        def __init__(self) -> None:
            super().__init__("stream idle timeout")
            self.render_count = 0

        def __str__(self) -> str:
            self.render_count += 1
            if self.render_count == 1:
                return "stream idle timeout"
            return "timeout\x00workload-secret-value"

    error = FlippingTimeoutError()
    provider = FlakyCompactionProvider(failures=2, error=error)
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
    )

    with pytest.raises(ModelProviderError) as exc_info:
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )

    assert len(provider.requests) == 2
    assert error.render_count == 2
    assert str(exc_info.value) == "Model provider emitted a non-portable error value."
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "workload-secret-value" not in repr(exc_info.value)


def test_model_compactor_uses_one_detached_provider_error_snapshot_per_attempt():
    class FlippingError(ModelProviderError):
        def __init__(self) -> None:
            super().__init__(
                "provider overloaded",
                provider="flaky",
                status_code=503,
                retryable=True,
            )
            self.render_count = 0

        def __str__(self) -> str:
            self.render_count += 1
            if self.render_count == 1:
                return "provider overloaded"
            return "timeout\x00workload-secret-value"

    provider = FlakyCompactionProvider(failures=2, error=FlippingError())
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
    )

    with pytest.raises(ModelProviderError) as exc_info:
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )

    assert len(provider.requests) == 2
    assert str(exc_info.value) == "Model provider emitted a non-portable error value."
    assert exc_info.value.retryable is False
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "timeout" not in repr(exc_info.value)
    assert "workload-secret-value" not in repr(exc_info.value)


def test_model_compactor_rejects_non_portable_request_billing_error_before_dispatch():
    class NonPortableRequestBillingProvider(FakeProvider):
        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity:
            del request
            raise RuntimeError("billing\x00workload-secret-value")

    provider = NonPortableRequestBillingProvider([ModelStreamEvent.completed({})])
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        retry_policy=RetryPolicy(max_attempts=3, initial_delay_s=0.0),
    )

    with pytest.raises(ModelProviderError) as exc_info:
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )

    assert provider.requests == []
    assert str(exc_info.value) == "Model provider emitted a non-portable error value."
    assert exc_info.value.retryable is False
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "workload-secret-value" not in repr(exc_info.value)


def test_hierarchical_model_compactor_freezes_configuration_for_every_dispatch() -> None:
    class ConfigurationMutatingProvider(ModelProvider):
        name = "gateway"
        billing_provider_name = "billco"
        usage_dialect = UsageDialect.ANTHROPIC

        def __init__(self) -> None:
            self.compactor: ModelCompactor | None = None
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            assert self.compactor is not None
            if len(self.requests) == 1:
                self.compactor.model = "poisoned-model"
                self.compactor._compactor_name = "poisoned\x00compactor"
                self.compactor._provider_snapshot = object()  # type: ignore[assignment]
                self.compactor._usage_dialect = UsageDialect.GENERIC
                self.compactor.max_input_chars = None
                self.compactor.max_hierarchy_calls = 2
            yield ModelStreamEvent.text_delta("short summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {
                        "input_tokens": 4,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 2,
                    },
                }
            )

    provider = ConfigurationMutatingProvider()
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        max_input_chars=1000,
        max_hierarchy_calls=64,
    )
    provider.compactor = compactor

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=Session(
                    id="sess_hierarchy_invocation_identity",
                    agent_name="assistant",
                    provider_name="gateway",
                    model="fake-model",
                ),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "x" * 5000)],
                existing_summary="retain this summary",
            )
        )
    )

    assert len(provider.requests) > 1
    assert all(request.model == "summary-model" for request in provider.requests)
    assert result.metadata["compactor"] == "ModelCompactor"
    assert result.metadata["provider"] == "gateway"
    assert result.metadata["model"] == "summary-model"
    assert result.metadata["max_input_chars"] == 1000
    assert all(payload["provider_name"] == "billco" for payload in result.model_completed_payloads)
    assert all(
        payload["usage_metrics"]["input_tokens"] == 6 for payload in result.model_completed_payloads
    )


@pytest.mark.parametrize("mutation", ["prompt", "coverage"])
def test_model_compactor_revalidates_mutated_custom_prompt_before_dispatch(
    mutation: str,
) -> None:
    provider = FakeProvider([ModelStreamEvent.completed({})])
    prompt = CompactionPrompt(prompt="valid prompt", covered_message_count=1)
    if mutation == "prompt":
        prompt.prompt = "invalid\x00prompt"
    else:
        prompt.covered_message_count = 0
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        prompt_builder=lambda _request: prompt,
    )

    with pytest.raises(ValidationError):
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )

    assert provider.requests == []


def test_model_compactor_rejects_prompt_subclass_before_attribute_access() -> None:
    class HostileCompactionPrompt(CompactionPrompt):
        def __getattribute__(self, name: str):
            if name in {"prompt", "covered_message_count"}:
                raise AssertionError("subclass attributes must not be accessed")
            return super().__getattribute__(name)

    provider = FakeProvider([ModelStreamEvent.completed({})])
    prompt = HostileCompactionPrompt(prompt="valid prompt", covered_message_count=1)
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        prompt_builder=lambda _request: prompt,
    )

    with pytest.raises(TypeError, match="must return CompactionPrompt"):
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )

    assert provider.requests == []


def test_model_compactor_rejects_hostile_exact_prompt_field_keys_without_lookup() -> None:
    class HostileFieldKey(str):
        equality_calls = 0

        def __hash__(self) -> int:
            return str.__hash__(self)

        def __eq__(self, other: object) -> bool:
            del other
            type(self).equality_calls += 1
            raise AssertionError("prompt field-key equality must not execute")

    provider = FakeProvider([ModelStreamEvent.completed({})])
    prompt = CompactionPrompt(prompt="valid prompt", covered_message_count=1)
    object.__setattr__(
        prompt,
        "__dict__",
        {
            HostileFieldKey("prompt"): "valid prompt",
            "covered_message_count": 1,
        },
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        prompt_builder=lambda _request: prompt,
    )

    with pytest.raises(TypeError, match="must return CompactionPrompt"):
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )

    assert HostileFieldKey.equality_calls == 0
    assert provider.requests == []


def test_compaction_contracts_reject_nonportable_values_without_echoing_input():
    secret = "workload-secret"
    request_values = {
        "session": _test_session(),
        "agent": AgentSpec(name="assistant", model="fake-model"),
        "messages": [Message.text("user", "portable")],
    }
    invalid_factories = (
        lambda: CompactionRequest(**request_values, metadata={"value": math.nan}),
        lambda: CompactionRequest(**request_values, instructions=f"{secret}\x00"),
        lambda: CompactionPrompt(
            prompt=f"{secret}\ud800",
            covered_message_count=1,
        ),
        lambda: CompactionPrompt(
            prompt="portable",
            covered_message_count=MAX_DURABLE_JSON_INTEGER + 1,
        ),
        lambda: CompactionResult(
            summary="portable",
            covered_message_count=MAX_DURABLE_JSON_INTEGER + 1,
        ),
        lambda: CompactionResult(
            summary="portable",
            covered_message_count=0,
            source_chunk_count=MAX_DURABLE_JSON_INTEGER + 1,
        ),
        lambda: CompactionResult(
            summary="portable",
            covered_message_count=0,
            metadata={"value": MAX_DURABLE_JSON_INTEGER + 1},
        ),
        lambda: CompactionResult(
            summary="portable",
            covered_message_count=0,
            model_completed_payloads=[{"value": f"{secret}\x00"}],
        ),
    )

    for factory in invalid_factories:
        with pytest.raises((DurableValueError, ValidationError)) as exc_info:
            factory()
        assert secret not in str(exc_info.value)


def test_prompt_cache_bounded_fallback_uses_frozen_provider_identity() -> None:
    class IdentityMutatingOverflowProvider(ModelProvider):
        name = "gateway"
        billing_provider_name = "billco"
        usage_dialect = UsageDialect.ANTHROPIC

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                self.name = "poisoned\x00provider"
                self.billing_provider_name = "poisoned\ud800billing"
                self.usage_dialect = UsageDialect.GENERIC
                raise ModelContextOverflowError(
                    "context too large",
                    provider="gateway",
                    status_code=400,
                    error_code="context_length_exceeded",
                )
            yield ModelStreamEvent.text_delta("bounded summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 5,
                    },
                }
            )

    provider = IdentityMutatingOverflowProvider()
    compactor = PromptCacheCompactor(provider=provider)
    context = [Message.text("user", "cached prefix")]
    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=Session(
                    id="sess_prompt_cache_identity_overflow",
                    agent_name="assistant",
                    provider_name="gateway",
                    model="fake-model",
                ),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "bounded transcript")],
                context_messages=context,
                cache_prefix_request=ModelRequest(
                    model="fake-model",
                    messages=context,
                ),
            )
        )
    )

    assert len(provider.requests) == 2
    assert result.model_completed_payloads[0]["compaction_outcome"] == "context_overflow"
    assert result.model_completed_payloads[0]["provider_name"] == "billco"
    completed = result.model_completed_payloads[1]
    assert completed["provider_name"] == "billco"
    assert completed["usage_metrics"]["provider_name"] == "billco"
    assert completed["usage_metrics"]["input_tokens"] == 15
    assert completed["usage_metrics"]["total_tokens"] == 16


def test_automatic_compaction_retry_after_completion_reconciles_each_attempt() -> None:
    class PricedCompletedThenRetryableProvider(CompletedThenRetryableCompactionProvider):
        billing_provider_name = "fake"

    compactor_provider = PricedCompletedThenRetryableProvider()
    runtime_provider = FakeProvider(
        [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}})]
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=compaction_price_book(),
                    reservation=BudgetReservation(
                        max_input_tokens=10,
                        max_output_tokens=10,
                    ),
                ),
            )
        )
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
                retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_completed_retry_budget",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(compactor_provider.requests) == 2
    compaction_completions = [
        event
        for event in events
        if event.type == EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    ]
    assert [event.payload["usage_metrics"]["input_tokens"] for event in compaction_completions] == [
        1,
        2,
    ]
    reconciliations = [
        event
        for event in events
        if event.type == EventType.BUDGET_RECONCILED
        and event.payload.get("reason") == "automatic context compaction model completed"
    ]
    assert [Decimal(event.payload["actual_amount"]) for event in reconciliations] == [
        Decimal("0.000011"),
        Decimal("0.000012"),
    ]
    assert not any(
        event.type == EventType.BUDGET_RECONCILED
        and "uncertain" in str(event.payload.get("reason", ""))
        for event in events
    )


def test_cayu_app_rejects_wrapper_rewrite_of_runtime_completion_evidence() -> None:
    class RewritingWrapperCompactor(ContextCompactor):
        def __init__(self, provider: ModelProvider) -> None:
            self.provider = provider

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            inner = await ModelCompactor(
                provider=self.provider,
                model="summary-model",
            ).compact(request)
            original = inner.model_completed_payloads[0]
            rewritten = {
                **original,
                "usage_metrics": {
                    **original["usage_metrics"],
                    "input_tokens": 999,
                    "total_tokens": 1001,
                },
            }
            return CompactionResult(
                summary=inner.summary,
                covered_message_count=inner.covered_message_count,
                model_completed_payloads=[rewritten],
            )

    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("valid inner summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                }
            ),
        ]
    )
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(request_footprint=RequestFootprintConfig(enabled=False))
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=RewritingWrapperCompactor(compactor_provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_evidence_rewrite",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    completions = [event for event in events if event.type == EventType.MODEL_COMPLETED]
    assert len(completions) == 1
    assert completions[0].payload["usage_metrics"]["input_tokens"] == 10
    assert completions[0].payload["usage_metrics"]["total_tokens"] == 12
    failed = next(event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED)
    assert failed.payload["error_type"] == "ValueError"
    assert events[-1].type == EventType.SESSION_FAILED
    assert "cannot rewrite runtime-owned completion evidence" in events[-1].payload["error"]
    assert runtime_provider.requests == []


def test_cayu_app_rejects_forged_completion_without_observed_dispatch() -> None:
    completion_payload = {
        "model": "custom-model",
        "provider_name": "custom",
        "requested_model": "custom-model",
        "purpose": "context_compaction",
        "compactor": "ForgedResultCompactor",
    }

    class ForgedResultCompactor(ContextCompactor):
        def provider_budget_identity(self, _session: Session) -> None:
            return None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            return CompactionResult.model_construct(
                summary="invalid\x00summary",
                covered_message_count=len(request.messages),
                model_completed_payloads=[completion_payload],
            )

    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp()
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ForgedResultCompactor(),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_forged_compaction_result",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert (
        events[-1].payload["error"]
        == "Compaction completion was observed outside its provider dispatch."
    )
    assert runtime_provider.requests == []


def test_automatic_compaction_rejects_declared_opaque_provider_before_dispatch() -> None:
    class OpaqueProviderCompactor(ContextCompactor):
        def __init__(self, provider: ModelProvider) -> None:
            self.provider = provider
            self.compact_calls = 0

        def provider_budget_identity(self, _session: Session) -> tuple[str, str]:
            return "fake", "summary-model"

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.compact_calls += 1
            async for _event in self.provider.stream(
                ModelRequest(model="summary-model", messages=request.messages)
            ):
                pass
            return CompactionResult(
                summary="opaque summary",
                covered_message_count=len(request.messages),
            )

    compactor_provider = FakeProvider([ModelStreamEvent.completed({})])
    compactor = OpaqueProviderCompactor(compactor_provider)
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(enable_logging=False)
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_opaque_compactor_request_footprint",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert compactor.compact_calls == 0
    assert compactor_provider.requests == []
    assert runtime_provider.requests == []
    assert all(event.type != EventType.REQUEST_FOOTPRINT_RECORDED for event in events)
    assert events[-1].type == EventType.SESSION_FAILED
    assert "cannot observe each provider dispatch independently" in events[-1].payload["error"]


def test_automatic_compaction_rejects_missing_provider_identity_before_dispatch() -> None:
    class UndeclaredProviderCompactor(ContextCompactor):
        def __init__(self, provider: ModelProvider) -> None:
            self.provider = provider
            self.compact_calls = 0

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.compact_calls += 1
            async for _event in self.provider.stream(
                ModelRequest(model="summary-model", messages=request.messages)
            ):
                pass
            return CompactionResult(
                summary="opaque summary",
                covered_message_count=len(request.messages),
            )

    compactor_provider = FakeProvider([ModelStreamEvent.completed({})])
    compactor = UndeclaredProviderCompactor(compactor_provider)
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(enable_logging=False)
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_undeclared_compactor_request_footprint",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert compactor.compact_calls == 0
    assert compactor_provider.requests == []
    assert runtime_provider.requests == []
    assert all(event.type != EventType.REQUEST_FOOTPRINT_RECORDED for event in events)
    assert events[-1].type == EventType.SESSION_FAILED
    assert "explicitly declare provider_budget_identity" in events[-1].payload["error"]


def test_automatic_compaction_rejects_second_completion_for_one_provider_dispatch():
    class DuplicatingWrapperCompactor(ContextCompactor):
        def __init__(self, provider: ModelProvider) -> None:
            self.provider = provider

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            result = await ModelCompactor(
                provider=self.provider,
                model="summary-model",
            ).compact(request)
            forged = dict(result.model_completed_payloads[0])
            forged.pop("compaction_attempt_id", None)
            return result.model_copy(
                update={
                    "model_completed_payloads": [
                        *result.model_completed_payloads,
                        forged,
                    ]
                }
            )

    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 8, "output_tokens": 2},
                }
            ),
        ]
    )
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(
        request_footprint=RequestFootprintConfig(enabled=False),
        enable_logging=False,
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=DuplicatingWrapperCompactor(compactor_provider),
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_automatic_duplicate_dispatch_completion",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(compactor_provider.requests) == 1
    assert not any(
        event.type == EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
        for event in events
    )
    assert events[-1].type == EventType.SESSION_FAILED
    assert (
        events[-1].payload["error"]
        == "Compaction provider dispatch produced conflicting completion identities."
    )
    assert runtime_provider.requests == []


def test_cayu_app_merges_returned_compaction_evidence_atomically() -> None:
    completion_payload = {
        "model": "custom-model",
        "provider_name": "custom",
        "requested_model": "custom-model",
        "purpose": "context_compaction",
        "compactor": "MalformedEvidenceCompactor",
    }

    class MalformedEvidenceCompactor(ContextCompactor):
        def provider_budget_identity(self, _session: Session) -> None:
            return None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            return CompactionResult.model_construct(
                summary="summary",
                covered_message_count=len(request.messages),
                model_completed_payloads=[completion_payload, "not-an-object"],
            )

    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp()
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=MalformedEvidenceCompactor(),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_atomic_compaction_evidence_merge",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert all(event.type != EventType.MODEL_COMPLETED for event in events)
    failed = next(event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED)
    assert failed.payload["error_type"] == "TypeError"
    assert events[-1].type == EventType.SESSION_FAILED
    assert "must be a list of objects" in events[-1].payload["error"]
    assert runtime_provider.requests == []


def test_cayu_app_rejects_compaction_result_subclass_before_attribute_access() -> None:
    class HostileCompactionResult(CompactionResult):
        def __getattribute__(self, name: str):
            if name in {"model_completed_payloads", "summary", "covered_message_count"}:
                raise AssertionError("subclass attributes must not be accessed")
            return super().__getattribute__(name)

    result = HostileCompactionResult(summary="summary", covered_message_count=2)

    class SubclassResultCompactor(ContextCompactor):
        def provider_budget_identity(self, _session: Session) -> None:
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
            compactor=SubclassResultCompactor(),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_result_subclass",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert all(event.type != EventType.MODEL_COMPLETED for event in events)
    failed = next(event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED)
    assert failed.payload["error_type"] == "TypeError"
    assert events[-1].type == EventType.SESSION_FAILED
    assert "must return CompactionResult" in events[-1].payload["error"]
    assert runtime_provider.requests == []


def test_cayu_app_rejects_non_portable_compaction_error_before_retry_or_publication():
    secret = "timeout\x00workload-secret-value"
    provider_error = ModelProviderError(
        secret,
        provider="fake",
        retryable=True,
    )
    invalid_event = ModelStreamEvent.error(secret, cause=provider_error)
    compactor_provider = FakeProvider([[invalid_event], [invalid_event]])
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp()
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
                retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_unsafe_compaction_error",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []
    attempt, failed, terminal = events[3], events[4], events[-1]
    assert attempt.payload["compaction_outcome"] == "unfinished_stream"
    assert attempt.payload["error_type"] == "DurableValueError"
    assert "usage_metrics" not in attempt.payload
    assert failed.payload["error_type"] == "DurableValueError"
    assert "error" not in failed.payload
    assert terminal.payload == {
        "error": "Operation failed with a non-portable diagnostic.",
        "error_type": "DurableValueError",
        "durable_value_error_code": "nul_character",
        "durable_value_error_path": "$/#0",
    }
    rendered = json.dumps(
        [event.model_dump(mode="json") for event in events],
        allow_nan=False,
    )
    assert "timeout" not in rendered
    assert "workload-secret-value" not in rendered


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
def test_cayu_app_rejects_non_portable_raised_compaction_error_without_retry_or_leak(
    caplog: pytest.LogCaptureFixture,
    raised_error: Exception,
):
    class NonPortableCompactionErrorProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            raise raised_error
            yield

    compactor_provider = NonPortableCompactionErrorProvider()
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp()
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
                retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_unsafe_raised_compaction_error",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []
    attempt, failed, terminal = events[3], events[4], events[-1]
    assert attempt.payload["compaction_outcome"] == "provider_error"
    assert attempt.payload["error_type"] == "ModelProviderError"
    assert failed.payload["error_type"] == "ModelProviderError"
    assert "error" not in failed.payload
    assert terminal.payload["error"] == "Model provider emitted a non-portable error value."
    rendered = json.dumps([event.model_dump(mode="json") for event in events])
    assert "timeout" not in rendered
    assert "workload-secret-value" not in rendered
    assert "timeout" not in caplog.text
    assert "workload-secret-value" not in caplog.text


def test_cayu_app_does_not_publish_forged_compaction_durable_value_diagnostics(
    caplog: pytest.LogCaptureFixture,
):
    class ForgedDiagnosticProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.error = DurableValueError(
                "nul_character",
                "workload-secret-field",
                path="$/#0",
            )
            self.error.code = "workload-secret-code"
            self.error.field_name = "workload-secret-mutated-field"
            self.error.path = "$/workload-secret-path"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            raise self.error
            yield

    compactor_provider = ForgedDiagnosticProvider()
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp()
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
                retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_forged_compaction_durable_diagnostic",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []
    failed = next(event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED)
    assert failed.payload["error_type"] == "DurableValueError"
    assert "error" not in failed.payload
    terminal = events[-1]
    assert terminal.type == EventType.SESSION_FAILED
    assert terminal.payload == {
        "error": "Operation failed with a non-portable diagnostic.",
        "error_type": "DurableValueError",
        "durable_value_error_code": "invalid_json_type",
        "durable_value_error_path": "$",
    }
    rendered = json.dumps([event.model_dump(mode="json") for event in events])
    assert "workload-secret" not in rendered
    assert "workload-secret" not in caplog.text
