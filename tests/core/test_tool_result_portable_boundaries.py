"""Focused issue #529 tool durability boundary tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any

import pytest
from tests.core.test_runtime import (
    FailingOrdinaryToolResultCloseStore,
    FakeProvider,
    _test_session,
    collect_events,
    collect_resume_events,
    compaction_price_book,
)

import cayu.runtime._tool_results as tool_results_module
from cayu._validation import MAX_DURABLE_JSON_INTEGER, DurableValueError
from cayu.core import (
    AgentSpec,
    Event,
    EventType,
    Message,
)
from cayu.core.tools import (
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.environments import (
    Environment,
    EnvironmentSpec,
)
from cayu.providers import (
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
    UsageDialect,
)
from cayu.proxies import CredentialProxy, PassthroughProxy, ProxyAuthorizationResult
from cayu.runtime import (
    AfterToolCallDecision,
    BillingIdentity,
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    BudgetWindow,
    CayuApp,
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    InMemoryBudgetStore,
    InMemorySessionStore,
    InterruptSessionRequest,
    ModelCompactor,
    ResumeRequest,
    RetryPolicy,
    RunRequest,
    RuntimeHook,
    ToolCallHookContext,
)
from cayu.vaults import ResolvedSecret, SecretRef, StaticVault

_HOSTILE_DURABLE_ERROR_SECRET = "workload-secret-durable-error-accessor"


class _HostileDurableValueError(DurableValueError):
    def __init__(self) -> None:
        object.__setattr__(self, "_armed", False)
        super().__init__("nul_character", "provider value", path="$/#0")
        object.__setattr__(self, "_armed", True)

    def __getattribute__(self, name: str):
        if name in {"code", "path"} and object.__getattribute__(self, "_armed"):
            raise RuntimeError(_HOSTILE_DURABLE_ERROR_SECRET)
        return object.__getattribute__(self, name)

    def __str__(self) -> str:
        raise RuntimeError(_HOSTILE_DURABLE_ERROR_SECRET)


def test_cayu_app_rejects_nonportable_proxy_result_before_tool_external_effect() -> None:
    class NonPortableResultProxy(CredentialProxy):
        async def resolve(
            self,
            ref: SecretRef,
            *,
            scope: dict[str, Any] | None = None,
        ) -> ResolvedSecret:
            raise AssertionError("resolve should not be called")

        async def authorize_request(
            self,
            *,
            destination: str,
            credential: SecretRef | None = None,
            action: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> ProxyAuthorizationResult:
            del destination, credential, action, metadata
            return ProxyAuthorizationResult.model_construct(
                allowed=True,
                reason=None,
                metadata={"workload-secret": "invalid\x00metadata"},
            )

    class AuthorizeThenEffectTool(Tool):
        spec = ToolSpec(
            name="authorize_then_effect",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.EXTERNAL,
        )

        def __init__(self) -> None:
            self.external_effects = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.proxy is not None
            await ctx.proxy.authorize_request(
                destination="https://example.com/effect",
                action="create",
                metadata={"request": "portable"},
            )
            self.external_effects += 1
            return ToolResult(content="effect completed")

    tool = AuthorizeThenEffectTool()
    provider = _portable_tool_boundary_provider(
        tool.spec.name,
        "call_nonportable_proxy_result",
    )
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            proxy=NonPortableResultProxy(),
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_nonportable_proxy_result",
                messages=[Message.text("user", "perform the effect")],
            ),
        )
    )

    assert tool.external_effects == 0
    assert EventType.CREDENTIAL_PROXY_CHECKED not in [event.type for event in events]
    terminal = next(event for event in events if event.type == EventType.TOOL_CALL_FAILED)
    assert terminal.payload["terminal_outcome"] == "tool_execution_error"
    assert terminal.payload["tool_effect"] == "external"
    assert terminal.payload["outcome_unknown"] is True
    assert terminal.payload["manual_reconciliation_required"] is True
    assert terminal.payload["durable_value_error_code"] == "nul_character"
    rendered = json.dumps([event.model_dump(mode="json") for event in events])
    assert "workload-secret" not in rendered
    assert "invalid\\u0000metadata" not in rendered
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_cayu_app_isolates_completion_payload_from_billing_hook_mutation() -> None:
    identity = BillingIdentity(
        provider_name="fake",
        resource_id="fake-model",
        request_evidence={"route": "primary"},
    )

    class MutatingCompletionBillingProvider(FakeProvider):
        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity:
            del request
            return identity

        def billing_identity_for_completion(
            self,
            current_identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity | None:
            assert current_identity == identity
            assert payload["usage"] == {"input_tokens": 3, "output_tokens": 1}
            payload["usage"]["input_tokens"] = 0
            payload["mutated\x00field"] = float("nan")
            return identity

    provider = MutatingCompletionBillingProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": "fake-model",
                    "usage": {"input_tokens": 3, "output_tokens": 1},
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
                session_id="sess_completion_billing_payload_mutation",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["usage"] == {"input_tokens": 3, "output_tokens": 1}
    assert completed.payload["usage_metrics"]["total_tokens"] == 4
    assert "mutated\x00field" not in completed.payload
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_cayu_app_isolates_billing_identity_from_completion_hook_mutation() -> None:
    identity = BillingIdentity(
        provider_name="fake",
        resource_id="fake-model",
        request_evidence={"route": "primary"},
    )

    class MutatingCompletionBillingProvider(FakeProvider):
        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity:
            del request
            return identity

        def billing_identity_for_completion(
            self,
            current_identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity | None:
            del payload
            assert current_identity == identity
            assert current_identity is not identity
            # Frozen models are an API guard, not a trust boundary: hostile
            # provider code can deliberately bypass their normal setter.
            object.__setattr__(current_identity, "resource_id", "hook-mutated-model")
            return identity

    provider = MutatingCompletionBillingProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": "fake-model",
                    "usage": {"input_tokens": 3, "output_tokens": 1},
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
                session_id="sess_completion_billing_identity_mutation",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["billing_identity"] == identity.model_dump(mode="json")
    assert completed.payload["usage_metrics"]["total_tokens"] == 4
    assert identity.resource_id == "fake-model"
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_cayu_app_accounts_for_completion_before_non_portable_billing_hook_failure():
    secret = "billing\x00workload-secret-value"

    class NonPortableCompletionBillingProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.closed = False

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            try:
                yield ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {"input_tokens": 30, "output_tokens": 4},
                    }
                )
            finally:
                self.closed = True

        def billing_identity_for_completion(
            self,
            identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity | None:
            del identity, payload
            raise RuntimeError(secret)

    provider = NonPortableCompletionBillingProvider()
    app = CayuApp(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_nonportable_completion_billing_error",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert provider.closed is True
    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["usage_metrics"]["input_tokens"] == 30
    assert completed.payload["usage_metrics"]["output_tokens"] == 4
    assert completed.payload["completion_outcome"] == "billing_identity_resolution_failed"
    assert completed.payload["completion_error"]["provider_error_code"] == (
        "invalid_model_provider_error"
    )
    assert EventType.MODEL_RETRY not in [event.type for event in events]
    assert EventType.MODEL_ERROR not in [event.type for event in events]
    usage = asyncio.run(app.get_session_usage("sess_nonportable_completion_billing_error"))
    assert usage.model_steps == 1
    assert usage.usage.total_tokens == 34
    rendered = json.dumps([event.model_dump(mode="json") for event in events])
    assert "workload-secret-value" not in rendered
    assert events[-1].type == EventType.SESSION_FAILED


def test_cayu_app_does_not_redispatch_valid_completion_after_retryable_billing_hook_error():
    requested_identity = BillingIdentity(
        provider_name="fake",
        resource_id="fake-model",
        request_evidence={"route": "primary"},
    )

    class FailingCompletionBillingProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.hook_calls = 0

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity:
            del request
            return requested_identity

        def billing_identity_for_completion(
            self,
            identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity | None:
            assert identity == requested_identity
            assert payload["usage"]["input_tokens"] == 30
            self.hook_calls += 1
            raise ModelProviderError(
                "completion identity temporarily unavailable",
                provider=self.name,
                status_code=503,
                retryable=True,
            )

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            yield ModelStreamEvent.completed(
                {
                    "model": "fake-model",
                    "usage": {"input_tokens": 30, "output_tokens": 4},
                }
            )

    provider = FailingCompletionBillingProvider()
    app = CayuApp(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_terminal_completion_billing_failure",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert provider.hook_calls == 1
    assert EventType.MODEL_RETRY not in [event.type for event in events]
    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["usage_metrics"]["input_tokens"] == 30
    assert completed.payload["usage_metrics"]["output_tokens"] == 4
    assert completed.payload["billing_identity"] == requested_identity.model_dump(mode="json")
    assert completed.payload["completion_outcome"] == "billing_identity_resolution_failed"
    usage = asyncio.run(app.get_session_usage("sess_terminal_completion_billing_failure"))
    assert usage.model_steps == 1
    assert usage.usage.total_tokens == 34
    assert events[-1].type == EventType.SESSION_FAILED


@pytest.mark.parametrize("phase", ["request", "completion"])
def test_cayu_app_does_not_publish_forged_billing_hook_diagnostics(phase: str):
    secret_field = "workload-secret-field"

    def forged_error() -> DurableValueError:
        error = DurableValueError("nul_character", secret_field, path="$/#0")
        error.code = "workload-secret-code"
        error.path = "$/workload-secret-path"
        return error

    class ForgedBillingErrorProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            del request
            if phase == "request":
                raise forged_error()
            return None

        def billing_identity_for_completion(
            self,
            identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity | None:
            del identity, payload
            if phase == "completion":
                raise forged_error()
            return None

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            yield ModelStreamEvent.completed({"usage": {"input_tokens": 3, "output_tokens": 1}})

    provider = ForgedBillingErrorProvider()
    app = CayuApp(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_forged_billing_error_{phase}",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert len(provider.requests) == (0 if phase == "request" else 1)
    assert EventType.MODEL_RETRY not in [event.type for event in events]
    rendered = json.dumps([event.model_dump(mode="json") for event in events])
    assert secret_field not in rendered
    assert "workload-secret-code" not in rendered
    assert "workload-secret-path" not in rendered
    assert events[-1].payload["error"] == ("Model provider emitted a non-portable error value.")
    if phase == "completion":
        completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
        assert completed.payload["usage_metrics"]["total_tokens"] == 4


@pytest.mark.parametrize("phase", ["request", "completion"])
def test_cayu_app_terminalizes_hostile_durable_billing_hook_error(phase: str):
    class HostileBillingErrorProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            del request
            if phase == "request":
                raise _HostileDurableValueError()
            return None

        def billing_identity_for_completion(
            self,
            identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity | None:
            del identity, payload
            if phase == "completion":
                raise _HostileDurableValueError()
            return None

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            yield ModelStreamEvent.completed({"usage": {"input_tokens": 3, "output_tokens": 1}})

    provider = HostileBillingErrorProvider()
    app = CayuApp(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_hostile_durable_billing_error_{phase}",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert len(provider.requests) == (0 if phase == "request" else 1)
    assert EventType.MODEL_RETRY not in [event.type for event in events]
    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].payload["error"] == ("Model provider emitted a non-portable error value.")
    if phase == "request":
        error_payload = next(
            event.payload for event in events if event.type == EventType.MODEL_ERROR
        )
    else:
        completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
        assert completed.payload["usage_metrics"]["total_tokens"] == 4
        error_payload = completed.payload["completion_error"]
    assert error_payload["provider_error_code"] == "invalid_model_provider_error"
    rendered = json.dumps([event.model_dump(mode="json") for event in events])
    assert _HOSTILE_DURABLE_ERROR_SECRET not in rendered


def test_after_tool_call_hook_can_modify_no_effect_execution_failure_before_persistence():
    class FailingNoEffectTool(Tool):
        spec = ToolSpec(
            name="failing_no_effect",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.NONE,
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            raise RuntimeError("ordinary failure")

    class FailureModifierHook(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> AfterToolCallDecision:
            assert context.result.is_error is True
            return AfterToolCallDecision(
                action="modify",
                modified_result=ToolResult(content="rewritten failure", is_error=True),
            )

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_failing_no_effect",
                    name="failing_no_effect",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[FailingNoEffectTool()],
        runtime_hooks=[FailureModifierHook()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_modifiable_no_effect_failure",
                messages=[Message.text("user", "run")],
            ),
        )
    )

    terminal = next(event for event in events if event.type == EventType.TOOL_CALL_FAILED)
    assert terminal.payload["result"]["content"] == "rewritten failure"
    assert terminal.payload["terminal_outcome"] == "tool_execution_error"
    assert terminal.payload["tool_effect"] == "none"
    assert terminal.payload["result"]["structured"] is None
    relevant_types = [
        event.type
        for event in events
        if event.type
        in {
            EventType.HOOK_STARTED,
            EventType.HOOK_COMPLETED,
            EventType.TOOL_CALL_FAILED,
        }
    ]
    assert relevant_types == [
        EventType.HOOK_STARTED,
        EventType.HOOK_COMPLETED,
        EventType.TOOL_CALL_FAILED,
    ]
    assert provider.requests[1].messages[-1].content[0].content == "rewritten failure"


def test_cayu_app_prioritizes_usage_overflow_over_completion_hook_failure():
    original = ModelProviderError(
        "completion hook must not replace overflow",
        provider="fake",
        status_code=503,
        retryable=True,
    )

    class OverflowAndHookFailureProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__(
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
                            }
                        ),
                    ],
                    [
                        ModelStreamEvent.text_delta("must not retry"),
                        ModelStreamEvent.completed({"usage": {"input_tokens": 1}}),
                    ],
                ]
            )
            self.completion_hook_calls = 0

        def billing_identity_for_completion(
            self,
            identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity | None:
            del identity, payload
            self.completion_hook_calls += 1
            raise original

    store = InMemorySessionStore()
    compactor_provider = OverflowAndHookFailureProvider()
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(session_store=store)
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

    session_id = "sess_compaction_overflow_and_hook_failure"
    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(compactor_provider.requests) == 1
    assert compactor_provider.completion_hook_calls == 1
    assert runtime_provider.requests == []
    completions = [event for event in events if event.type == EventType.MODEL_COMPLETED]
    assert len(completions) == 1
    completion = completions[0]
    assert completion.payload["compaction_outcome"] == "invalid_completion_metadata"
    assert completion.payload["usage_metrics_rejected"] is True
    assert "usage" not in completion.payload
    assert "usage_metrics" not in completion.payload
    failed = next(event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED)
    assert failed.payload["error_type"] == "DurableValueError"
    stored_completions = [
        event
        for event in asyncio.run(store.load_events(session_id))
        if event.type == EventType.MODEL_COMPLETED
    ]
    assert [event.id for event in stored_completions] == [completion.id]
    usage = asyncio.run(app.get_session_usage(session_id))
    assert usage.model_steps == 1


def test_model_compactor_detaches_terminal_completion_hook_failure_without_retry():
    original = ModelProviderError(
        "completion identity transient",
        provider="fake",
        status_code=503,
        retryable=True,
        response_body="workload-secret-response-body",
    )

    class CompletionHookFailureProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__(
                [
                    ModelStreamEvent.text_delta("completed summary"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 30, "output_tokens": 4}}),
                ]
            )
            self.completion_hook_calls = 0

        def billing_identity_for_completion(
            self,
            identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity | None:
            del identity, payload
            self.completion_hook_calls += 1
            raise original

    provider = CompletionHookFailureProvider()
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

    detached = exc_info.value
    assert len(provider.requests) == 1
    assert provider.completion_hook_calls == 1
    assert detached is not original
    assert str(detached) == "Model provider billing identity resolution failed"
    assert detached.status_code == 503
    assert detached.error_type == "BillingIdentityResolutionError"
    assert detached.error_code == "billing_identity_resolution_failed"
    assert detached.retryable is True
    assert detached.response_body is None
    assert detached.__cause__ is None
    assert detached.__context__ is None
    assert "workload-secret-response-body" not in repr(detached)


def test_model_compactor_isolates_request_identity_from_completion_hook_mutation():
    requested_identity = BillingIdentity(
        provider_name="billing-isolation",
        resource_id="summary-model",
        request_evidence={"route": {"region": "original"}},
    )

    class MutatingCompletionBillingProvider(FakeProvider):
        name = "billing-isolation"

        def __init__(self) -> None:
            super().__init__(
                [
                    ModelStreamEvent.text_delta("completed summary"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 4}}),
                ]
            )
            self.completion_identity: BillingIdentity | None = None

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity:
            assert request.model == "summary-model"
            return requested_identity

        def billing_identity_for_completion(
            self,
            identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity:
            assert identity is not None
            assert payload["usage"] == {"input_tokens": 4}
            self.completion_identity = identity
            object.__setattr__(
                identity,
                "request_evidence",
                {"route": {"region": "completion-hook-mutation"}},
            )
            return requested_identity

    provider = MutatingCompletionBillingProvider()
    result = asyncio.run(
        ModelCompactor(provider=provider, model="summary-model").compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old request")],
            )
        )
    )

    assert result.summary == "completed summary"
    assert provider.completion_identity is not None
    assert provider.completion_identity is not requested_identity
    assert provider.completion_identity.request_evidence == {
        "route": {"region": "completion-hook-mutation"}
    }
    assert requested_identity.request_evidence == {"route": {"region": "original"}}
    assert result.model_completed_payloads[0]["billing_identity"] == (
        requested_identity.model_dump(mode="json")
    )


def test_automatic_compaction_budget_uses_identity_frozen_before_billing_hook() -> None:
    class BillingHookMutatingProvider(FakeProvider):
        name = "gateway"
        billing_provider_name = "fake"
        usage_dialect = UsageDialect.ANTHROPIC

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            assert request.model == "summary-model"
            self.name = "poisoned\x00provider"
            self.billing_provider_name = "poisoned\ud800billing"
            self.usage_dialect = UsageDialect.GENERIC
            return None

    compactor_provider = BillingHookMutatingProvider(
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
                        max_input_tokens=20,
                        max_output_tokens=5,
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
                session_id="sess_compaction_frozen_budget_identity",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(compactor_provider.requests) == 1
    assert len(runtime_provider.requests) == 1
    completion = next(
        event
        for event in events
        if event.type == EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    )
    assert completion.payload["provider_name"] == "fake"
    assert completion.payload["usage_metrics"]["provider_name"] == "fake"
    assert completion.payload["usage_metrics"]["input_tokens"] == 15
    reconciliation = next(
        event
        for event in events
        if event.type == EventType.BUDGET_RECONCILED
        and event.payload.get("reason") == "automatic context compaction model completed"
    )
    assert Decimal(reconciliation.payload["actual_amount"]) == Decimal("0.000025")
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_cayu_app_does_not_redispatch_after_valid_completion_billing_hook_fails():
    class FailingCompletionBillingProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__(
                [
                    [
                        ModelStreamEvent.text_delta("summary-one"),
                        ModelStreamEvent.completed(
                            {
                                "model": "summary-model",
                                "usage": {"input_tokens": 30, "output_tokens": 4},
                            }
                        ),
                    ],
                    [
                        ModelStreamEvent.text_delta("must not retry"),
                        ModelStreamEvent.completed(
                            {"usage": {"input_tokens": 1, "output_tokens": 1}}
                        ),
                    ],
                ]
            )
            self.completion_hook_calls = 0

        def billing_identity_for_completion(
            self,
            identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity | None:
            del identity, payload
            self.completion_hook_calls += 1
            raise ModelProviderError(
                "completion identity transient",
                provider=self.name,
                status_code=503,
                retryable=True,
            )

    store = InMemorySessionStore()
    budget_store = InMemoryBudgetStore()
    compactor_provider = FailingCompletionBillingProvider()
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(
        session_store=store,
        budget_store=budget_store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=compaction_price_book(),
                    reservation=BudgetReservation(
                        max_input_tokens=100,
                        max_output_tokens=100,
                    ),
                ),
            )
        ),
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
                session_id="sess_valid_completion_billing_failure",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(compactor_provider.requests) == 1
    assert compactor_provider.completion_hook_calls == 1
    assert runtime_provider.requests == []
    completed_events = [event for event in events if event.type == EventType.MODEL_COMPLETED]
    assert len(completed_events) == 1
    completed = completed_events[0]
    assert completed.payload["usage_metrics"]["input_tokens"] == 30
    assert completed.payload["usage_metrics"]["output_tokens"] == 4
    assert completed.payload["compaction_outcome"] == "completion_observation_failed"
    failed = next(event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED)
    assert failed.payload == {
        "bounded_input": True,
        "checkpoint": "context_compaction",
        "chunk_count": 1,
        "chunk_mode": "failed",
        "compactor": "ModelCompactor",
        "compacted_transcript_cursor": 0,
        "compaction_failed": True,
        "coverage_mode": "failed",
        "previous_compacted_transcript_cursor": 0,
        "newly_compacted_message_count": 0,
        "recent_message_count": 1,
        "represented_message_count": 0,
        "represented_source_end": 0,
        "represented_source_start": 0,
        "requested_source_end": 2,
        "requested_source_start": 0,
        "error_type": "ModelProviderError",
        "model_step_id": completed.payload["model_step_id"],
    }
    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].payload["error"] == "Model provider billing identity resolution failed"
    reconciliation = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
    assert Decimal(reconciliation.payload["actual_amount"]) == Decimal("0.000070")
    usage = asyncio.run(app.get_session_usage("sess_valid_completion_billing_failure"))
    assert usage.model_steps == 1
    assert usage.usage.total_tokens == 34
    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    assert [event.id for event in budget_events] == [completed.id]


def test_cayu_app_does_not_retry_invalid_completion_when_billing_hook_fails():
    requested_identity = BillingIdentity(
        provider_name="fake",
        resource_id="summary-model",
    )

    class FailingCompletionBillingProvider(FakeProvider):
        def __init__(self) -> None:
            invalid_completion = ModelStreamEvent.model_construct(
                type=ModelStreamEventType.COMPLETED,
                delta="",
                payload={
                    "model": "summary-model",
                    "usage": {"input_tokens": 30, "output_tokens": 4},
                    "invalid": float("nan"),
                },
                completion=None,
            )
            super().__init__(
                [
                    [ModelStreamEvent.text_delta("summary"), invalid_completion],
                    [
                        ModelStreamEvent.text_delta("must not retry"),
                        ModelStreamEvent.completed(
                            {"usage": {"input_tokens": 1, "output_tokens": 1}}
                        ),
                    ],
                ]
            )
            self.completion_hook_calls = 0

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity:
            assert request.model == "summary-model"
            return requested_identity

        def billing_identity_for_completion(
            self,
            identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity | None:
            assert identity == requested_identity
            del payload
            self.completion_hook_calls += 1
            raise ModelProviderError(
                "completion billing unavailable",
                provider=self.name,
                status_code=503,
                retryable=True,
            )

    store = InMemorySessionStore()
    budget_store = InMemoryBudgetStore()
    compactor_provider = FailingCompletionBillingProvider()
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(session_store=store, budget_store=budget_store)
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
                session_id="sess_invalid_completion_billing_failure",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(compactor_provider.requests) == 1
    assert compactor_provider.completion_hook_calls == 1
    assert runtime_provider.requests == []
    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["usage_metrics"]["input_tokens"] == 30
    assert completed.payload["usage_metrics"]["output_tokens"] == 4
    assert completed.payload["billing_identity"] == requested_identity.model_dump(mode="json")
    assert "billing_identity" not in completed.payload["usage_metrics"]
    assert completed.payload["compaction_outcome"] == "invalid_completion_metadata"
    failed = next(event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED)
    assert failed.payload["error_type"] == "DurableValueError"
    usage = asyncio.run(app.get_session_usage("sess_invalid_completion_billing_failure"))
    assert usage.model_steps == 1
    assert usage.usage.total_tokens == 34
    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    assert [event.id for event in budget_events] == [completed.id]
    assert budget_events[0].payload["billing_identity"] == requested_identity.model_dump(
        mode="json"
    )


def _portable_tool_boundary_provider(tool_name: str, call_id: str) -> FakeProvider:
    return FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id=call_id, name=tool_name, arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("continued"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )


def test_portable_tool_result_evidence_caps_invalid_object_key_scans() -> None:
    value: dict[Any, Any] = {index: "ignored" for index in range(4096)}
    value["late_portable"] = "must not be scanned"

    evidence = tool_results_module.portable_result_evidence(value)

    assert evidence.included is True
    assert evidence.incomplete is True
    assert evidence.value == {}


def test_portable_tool_result_evidence_omits_receipts_beyond_scan_limit() -> None:
    value = {f"ordinary_{index}": index for index in range(4097)}
    value["receipt_id"] = "receipt-after-scan-limit"

    evidence = tool_results_module.portable_result_evidence(value)

    assert evidence.included is True
    assert evidence.incomplete is True
    assert "receipt_id" not in evidence.value


def test_portable_tool_result_evidence_prioritizes_receipt_without_hostile_key_lookup() -> None:
    class HostileKey(str):
        armed = False
        equality_calls = 0

        def __hash__(self) -> int:
            return hash("receipt_id")

        def __eq__(self, other: object) -> bool:
            del other
            type(self).equality_calls += 1
            if type(self).armed:
                raise AssertionError("provider-owned key equality must not run")
            return False

    value: dict[Any, Any] = {HostileKey("provider_owned_key"): "ignored"}
    value.update(
        {
            **{f"ordinary_{index}": index for index in range(300)},
            "receipt_id": "receipt-within-scan-limit",
            "invalid": float("nan"),
        }
    )
    HostileKey.equality_calls = 0
    HostileKey.armed = True

    evidence = tool_results_module.portable_result_evidence(value)

    assert HostileKey.equality_calls == 0
    assert evidence.included is True
    assert evidence.incomplete is True
    assert evidence.value["receipt_id"] == "receipt-within-scan-limit"


def test_external_invalid_tool_output_ignores_hostile_field_keys_and_replays_terminal_evidence():
    class HostileKey(str):
        armed = False

        def __hash__(self) -> int:
            return hash("structured")

        def __eq__(self, other: object) -> bool:
            del other
            if type(self).armed:
                raise TimeoutError("tool-owned key equality must not run")
            return False

    class HostileResultTool(Tool):
        spec = ToolSpec(
            name="hostile_result_fields",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.EXTERNAL,
        )

        def __init__(self) -> None:
            self.external_effects = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            self.external_effects += 1
            fields: dict[Any, Any] = {HostileKey("provider_owned_key"): "ignored"}
            fields.update(
                {
                    "content": "charged",
                    "structured": {
                        "receipt_id": "receipt-hostile-fields",
                        "invalid": float("nan"),
                    },
                    "artifacts": [],
                    "is_error": False,
                }
            )
            forged = ToolResult.model_construct()
            object.__setattr__(forged, "__dict__", fields)
            HostileKey.armed = True
            return forged

    session_id = "sess_hostile_result_fields"
    store = FailingOrdinaryToolResultCloseStore()
    tool = HostileResultTool()
    provider = _portable_tool_boundary_provider(tool.spec.name, "call_hostile_result_fields")
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "charge")],
            ),
        )
    )

    assert tool.external_effects == 1
    terminal = next(event for event in initial_events if event.type == EventType.TOOL_CALL_FAILED)
    assert terminal.payload["terminal_outcome"] == "invalid_tool_output"
    assert terminal.payload["manual_reconciliation_required"] is True
    evidence = terminal.payload["result"]["structured"]["portable_result_evidence"]
    assert evidence["structured"]["receipt_id"] == "receipt-hostile-fields"
    assert initial_events[-1].type == EventType.SESSION_FAILED

    resumed_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue")],
            ),
        )
    )

    assert resumed_events[-1].type == EventType.SESSION_COMPLETED
    assert tool.external_effects == 1
    stored_events = asyncio.run(store.load_events(session_id))
    assert sum(event.type == EventType.TOOL_CALL_STARTED for event in stored_events) == 1
    assert sum(event.type == EventType.TOOL_CALL_FAILED for event in stored_events) == 1


def test_external_invalid_tool_output_preserves_priority_receipt_across_replay():
    class InvalidExternalReceiptTool(Tool):
        spec = ToolSpec(
            name="invalid_external_receipt",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.EXTERNAL,
        )

        def __init__(self) -> None:
            self.calls = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            self.calls += 1
            structured: dict[str, Any] = {f"junk_{index}": index for index in range(400)}
            # Reconciliation evidence is intentionally behind more ordinary
            # siblings than the salvage node limit.
            structured["receipt_id"] = "receipt-123"
            structured["auxiliary"] = float("nan")
            return ToolResult.model_construct(
                content="charged",
                structured=structured,
                artifacts=[],
                is_error=False,
            )

    session_id = "sess_invalid_external_receipt_replay"
    call_id = "call_invalid_external_receipt"
    store = FailingOrdinaryToolResultCloseStore()
    provider = _portable_tool_boundary_provider("invalid_external_receipt", call_id)
    tool = InvalidExternalReceiptTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "charge")],
            ),
        )
    )

    assert initial_events[-1].type == EventType.SESSION_FAILED
    assert tool.calls == 1
    started = next(event for event in initial_events if event.type == EventType.TOOL_CALL_STARTED)
    failed = next(event for event in initial_events if event.type == EventType.TOOL_CALL_FAILED)
    assert failed.payload["idempotency_key"] == started.payload["idempotency_key"]
    assert failed.payload["terminal_outcome"] == "invalid_tool_output"
    assert failed.payload["tool_effect"] == "external"
    assert failed.payload["outcome_unknown"] is True
    assert failed.payload["manual_reconciliation_required"] is True
    structured = failed.payload["result"]["structured"]
    assert structured["tool_effect"] == failed.payload["tool_effect"]
    assert structured["outcome_unknown"] == failed.payload["outcome_unknown"]
    assert structured["manual_reconciliation_required"] is True
    assert structured["portable_result_evidence"]["structured"]["receipt_id"] == ("receipt-123")
    assert structured["portable_result_evidence_incomplete"] is True
    json.dumps(failed.model_dump(mode="json"), ensure_ascii=False, allow_nan=False)

    resumed_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue")],
            ),
        )
    )

    assert resumed_events[-1].type == EventType.SESSION_COMPLETED
    assert tool.calls == 1
    stored_events = asyncio.run(store.load_events(session_id))
    assert sum(event.type == EventType.TOOL_CALL_STARTED for event in stored_events) == 1
    assert sum(event.type == EventType.TOOL_CALL_FAILED for event in stored_events) == 1
    transcript = asyncio.run(store.load_transcript(session_id))
    tool_part = next(message for message in transcript if message.role == "tool").content[0]
    assert tool_part.structured["portable_result_evidence"]["structured"]["receipt_id"] == (
        "receipt-123"
    )
    replay_tool_part = next(
        message for message in provider.requests[1].messages if message.role == "tool"
    ).content[0]
    assert replay_tool_part.structured == tool_part.structured


def test_external_tool_nonportable_exception_persists_manual_terminal_outcome():
    class NonPortableExceptionTool(Tool):
        spec = ToolSpec(
            name="nonportable_exception",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.EXTERNAL,
        )

        def __init__(self) -> None:
            self.calls = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            self.calls += 1
            raise RuntimeError("charged receipt-456\x00workload-secret")

    from cayu.vaults import SecretRedactor

    session_id = "sess_nonportable_external_exception"
    tool = NonPortableExceptionTool()
    provider = _portable_tool_boundary_provider(tool.spec.name, "call_nonportable_exception")
    app = CayuApp(
        session_store=InMemorySessionStore(),
        secret_redactor=SecretRedactor("workload-secret"),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "charge")],
            ),
        )
    )

    assert tool.calls == 1
    failed = [event for event in events if event.type == EventType.TOOL_CALL_FAILED]
    assert len(failed) == 1
    terminal = failed[0]
    assert terminal.payload["terminal_outcome"] == "tool_execution_error"
    assert terminal.payload["tool_effect"] == "external"
    assert terminal.payload["outcome_unknown"] is True
    assert terminal.payload["manual_reconciliation_required"] is True
    assert terminal.payload["durable_value_error_code"] == "nul_character"
    assert terminal.payload["result"]["structured"]["tool_effect"] == "external"
    assert events[-1].type == EventType.SESSION_COMPLETED
    rendered = json.dumps(
        [event.model_dump(mode="json") for event in events],
        ensure_ascii=False,
        allow_nan=False,
    )
    assert "workload-secret" not in rendered
    assert "receipt-456" not in rendered


def test_external_tool_hostile_exception_rendering_replays_without_reexecution() -> None:
    class HostileDiagnosticError(RuntimeError):
        def __str__(self) -> str:
            raise KeyboardInterrupt("exception rendering must not escape terminalization")

    class HostileExceptionTool(Tool):
        spec = ToolSpec(
            name="hostile_exception",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.EXTERNAL,
        )

        def __init__(self) -> None:
            self.effects = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            self.effects += 1
            raise HostileDiagnosticError()

    session_id = "sess_hostile_external_exception"
    store = FailingOrdinaryToolResultCloseStore()
    tool = HostileExceptionTool()
    provider = _portable_tool_boundary_provider(tool.spec.name, "call_hostile_exception")
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "charge")],
            ),
        )
    )

    assert initial_events[-1].type == EventType.SESSION_FAILED
    terminal = next(event for event in initial_events if event.type == EventType.TOOL_CALL_FAILED)
    assert terminal.payload["terminal_outcome"] == "tool_execution_error"
    assert terminal.payload["manual_reconciliation_required"] is True
    assert terminal.payload["result"]["content"] == (
        "HostileDiagnosticError: tool execution failed"
    )
    assert tool.effects == 1

    resumed_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue")],
            ),
        )
    )

    assert resumed_events[-1].type == EventType.SESSION_COMPLETED
    assert tool.effects == 1
    stored_events = asyncio.run(store.load_events(session_id))
    assert sum(event.type == EventType.TOOL_CALL_STARTED for event in stored_events) == 1
    assert sum(event.type == EventType.TOOL_CALL_FAILED for event in stored_events) == 1


def test_external_tool_hostile_durable_error_persists_manual_terminal_outcome():
    class HostileDurableErrorTool(Tool):
        spec = ToolSpec(
            name="hostile_durable_error",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.EXTERNAL,
        )

        def __init__(self) -> None:
            self.effects = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            self.effects += 1
            raise _HostileDurableValueError()

    session_id = "sess_hostile_durable_external_exception"
    store = FailingOrdinaryToolResultCloseStore()
    tool = HostileDurableErrorTool()
    provider = _portable_tool_boundary_provider(tool.spec.name, "call_hostile_durable_error")
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "charge")],
            ),
        )
    )

    assert initial_events[-1].type == EventType.SESSION_FAILED
    terminal = next(event for event in initial_events if event.type == EventType.TOOL_CALL_FAILED)
    assert terminal.payload["terminal_outcome"] == "tool_execution_error"
    assert terminal.payload["tool_effect"] == "external"
    assert terminal.payload["outcome_unknown"] is True
    assert terminal.payload["manual_reconciliation_required"] is True
    assert terminal.payload["durable_value_error_code"] == "invalid_json_type"
    assert terminal.payload["durable_value_error_path"] == "$"
    assert _HOSTILE_DURABLE_ERROR_SECRET not in json.dumps(
        [event.model_dump(mode="json") for event in initial_events]
    )
    assert tool.effects == 1

    resumed_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue")],
            ),
        )
    )

    assert resumed_events[-1].type == EventType.SESSION_COMPLETED
    assert tool.effects == 1
    stored_events = asyncio.run(store.load_events(session_id))
    assert sum(event.type == EventType.TOOL_CALL_STARTED for event in stored_events) == 1
    assert sum(event.type == EventType.TOOL_CALL_FAILED for event in stored_events) == 1


def test_invalid_tool_output_uses_registered_effect_after_tool_mutates_live_spec():
    class MutatingExternalTool(Tool):
        spec = ToolSpec(
            name="mutating_external_effect",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.EXTERNAL,
        )

        def __init__(self) -> None:
            self.external_effects = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            self.external_effects += 1
            self.spec = ToolSpec(
                name="mutating_external_effect",
                input_schema={"type": "object", "properties": {}},
                effect=ToolEffect.NONE,
            )
            return ToolResult.model_construct(
                content="effect completed",
                structured={"receipt_id": "receipt-registered-effect", "invalid": float("nan")},
                artifacts=[],
                is_error=False,
            )

    tool = MutatingExternalTool()
    provider = _portable_tool_boundary_provider(
        tool.spec.name,
        "call_registered_external_effect",
    )
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_registered_external_effect",
                messages=[Message.text("user", "perform the external effect")],
            ),
        )
    )

    assert tool.external_effects == 1
    assert tool.spec.effect is ToolEffect.NONE
    started = next(event for event in events if event.type == EventType.TOOL_CALL_STARTED)
    terminal = next(event for event in events if event.type == EventType.TOOL_CALL_FAILED)
    assert started.payload["effect"] == "external"
    assert terminal.payload["terminal_outcome"] == "invalid_tool_output"
    assert terminal.payload["tool_effect"] == "external"
    assert terminal.payload["outcome_unknown"] is True
    assert terminal.payload["manual_reconciliation_required"] is True
    assert terminal.payload["result"]["structured"]["tool_effect"] == "external"
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_nonportable_after_tool_hook_failure_keeps_original_terminal_result():
    class ReceiptTool(Tool):
        spec = ToolSpec(
            name="hook_receipt",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.EXTERNAL,
        )

        def __init__(self) -> None:
            self.calls = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            self.calls += 1
            return ToolResult(content="charged", structured={"receipt_id": "receipt-original"})

    class NonPortableHook(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            del context
            raise RuntimeError("hook failed\x00workload-secret")

    from cayu.vaults import SecretRedactor

    tool = ReceiptTool()
    provider = _portable_tool_boundary_provider(tool.spec.name, "call_hook_receipt")
    app = CayuApp(
        session_store=InMemorySessionStore(),
        secret_redactor=SecretRedactor("workload-secret"),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        runtime_hooks=[NonPortableHook()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_nonportable_after_tool_hook",
                messages=[Message.text("user", "charge")],
            ),
        )
    )

    assert tool.calls == 1
    hook_failed = next(event for event in events if event.type == EventType.HOOK_FAILED)
    assert hook_failed.payload["error"] == ("Runtime hook failed with a non-portable diagnostic.")
    assert hook_failed.payload["durable_value_error_code"] == "nul_character"
    terminal = [event for event in events if event.type == EventType.TOOL_CALL_COMPLETED]
    assert len(terminal) == 1
    assert terminal[0].payload["result"]["structured"] == {"receipt_id": "receipt-original"}
    assert "terminal_outcome" not in terminal[0].payload
    assert events[-1].type == EventType.SESSION_COMPLETED
    rendered = json.dumps([event.model_dump(mode="json") for event in events])
    assert "workload-secret" not in rendered


def test_forged_invalid_after_tool_modification_is_failed_closed():
    class ReceiptTool(Tool):
        spec = ToolSpec(
            name="forged_hook_receipt",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.NONE,
        )

        def __init__(self) -> None:
            self.calls = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            self.calls += 1
            return ToolResult(content="original", structured={"receipt_id": "receipt-original"})

    class ForgedModificationHook(RuntimeHook):
        async def after_tool_call(
            self,
            context: ToolCallHookContext,
        ) -> AfterToolCallDecision:
            del context
            forged = ToolResult.model_construct(
                content="forged",
                structured={"receipt_id": "receipt-forged", "auxiliary": float("nan")},
                artifacts=[],
                is_error=False,
            )
            return AfterToolCallDecision.model_construct(
                action="modify",
                modified_result=forged,
            )

    tool = ReceiptTool()
    provider = _portable_tool_boundary_provider(tool.spec.name, "call_forged_hook_receipt")
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        runtime_hooks=[ForgedModificationHook()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_forged_after_tool_modification",
                messages=[Message.text("user", "run")],
            ),
        )
    )

    assert tool.calls == 1
    hook_failed = next(event for event in events if event.type == EventType.HOOK_FAILED)
    assert hook_failed.payload["durable_value_error_code"] == "non_finite_number"
    terminal = [event for event in events if event.type == EventType.TOOL_CALL_COMPLETED]
    assert len(terminal) == 1
    assert terminal[0].payload["result"]["content"] == "original"
    assert terminal[0].payload["result"]["structured"] == {"receipt_id": "receipt-original"}
    json.dumps(terminal[0].model_dump(mode="json"), allow_nan=False)
    assert provider.requests[1].messages[-1].content[0].structured == {
        "receipt_id": "receipt-original"
    }
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_invalid_tool_output_evidence_is_redacted_before_hooks_and_publication():
    from cayu.vaults import REDACTED_SECRET, SecretRedactor

    secret = "receipt-workload-secret"
    observed: list[dict[str, Any] | None] = []

    class InvalidSecretReceiptTool(Tool):
        spec = ToolSpec(
            name="invalid_secret_receipt",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.EXTERNAL,
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            return ToolResult.model_construct(
                content="charged",
                structured={"receipt_id": secret, "auxiliary": float("nan")},
                artifacts=[],
                is_error=False,
            )

    class EvidenceObserver(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            observed.append(context.result.structured)

    store = InMemorySessionStore()
    provider = _portable_tool_boundary_provider(
        "invalid_secret_receipt",
        "call_invalid_secret_receipt",
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[InvalidSecretReceiptTool()],
        runtime_hooks=[EvidenceObserver()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_invalid_secret_receipt",
                messages=[Message.text("user", "charge")],
            ),
        )
    )

    assert len(observed) == 1
    assert observed[0]["portable_result_evidence"]["structured"]["receipt_id"] == (REDACTED_SECRET)
    terminal = next(event for event in events if event.type == EventType.TOOL_CALL_FAILED)
    assert terminal.payload["result"]["structured"] == observed[0]
    transcript = asyncio.run(store.load_transcript("sess_invalid_secret_receipt"))
    tool_part = next(message for message in transcript if message.role == "tool").content[0]
    assert tool_part.structured == observed[0]
    assert provider.requests[1].messages[-1].content[0].structured == observed[0]
    rendered = json.dumps(
        [event.model_dump(mode="json") for event in events],
        ensure_ascii=False,
        allow_nan=False,
    )
    assert secret not in rendered
    assert REDACTED_SECRET in rendered
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_terminal_tool_diagnostics_and_evidence_remain_bounded_after_expanding_redaction():
    from cayu.vaults import REDACTED_SECRET, SecretRedactor

    secret = "z"
    redactor = SecretRedactor(secret)
    redacted_diagnostic = tool_results_module.exception_diagnostic(
        RuntimeError(secret * 4096),
        redactor=redactor,
    )
    assert len(redacted_diagnostic.message.encode("utf-8")) <= (
        tool_results_module._MAX_DIAGNOSTIC_UTF8_BYTES
    )
    assert secret not in redacted_diagnostic.message
    assert REDACTED_SECRET in redacted_diagnostic.message

    class ExpandingEvidenceTool(Tool):
        spec = ToolSpec(
            name="expanding_evidence",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.EXTERNAL,
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            return ToolResult.model_construct(
                content=secret * 4096,
                structured={
                    "receipt_id": secret * 4096,
                    "auxiliary": float("nan"),
                },
                artifacts=[],
                is_error=False,
            )

    provider = _portable_tool_boundary_provider(
        "expanding_evidence",
        "call_expanding_evidence",
    )
    app = CayuApp(secret_redactor=redactor, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ExpandingEvidenceTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_expanding_terminal_evidence",
                messages=[Message.text("user", "run")],
            ),
        )
    )

    terminal = next(event for event in events if event.type == EventType.TOOL_CALL_FAILED)
    structured = terminal.payload["result"]["structured"]
    assert "portable_result_evidence" in structured
    assert structured["portable_result_evidence_incomplete"] is True
    evidence_bytes = len(
        json.dumps(
            structured["portable_result_evidence"],
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert evidence_bytes <= tool_results_module._MAX_PORTABLE_EVIDENCE_UTF8_BYTES
    assert secret not in json.dumps(terminal.model_dump(mode="json"), ensure_ascii=False)


def test_terminal_diagnostic_and_evidence_redact_secret_crossing_byte_boundaries() -> None:
    from cayu.vaults import REDACTED_SECRET, SecretRedactor

    secret = "boundary-secret-canary"
    redactor = SecretRedactor(secret)
    diagnostic_prefix = "d" * (
        tool_results_module._MAX_DIAGNOSTIC_UTF8_BYTES - len(secret.encode("utf-8")) // 2
    )
    diagnostic = tool_results_module.exception_diagnostic(
        RuntimeError(diagnostic_prefix + secret),
        redactor=redactor,
    )

    assert secret not in diagnostic.message
    assert secret[: len(secret) // 2] not in diagnostic.message
    assert len(diagnostic.message.encode("utf-8")) <= (
        tool_results_module._MAX_DIAGNOSTIC_UTF8_BYTES
    )

    evidence_prefix = "e" * (
        tool_results_module._MAX_PORTABLE_EVIDENCE_UTF8_BYTES
        - len(secret.encode("utf-8")) // 2
        - 32
    )
    evidence = tool_results_module.portable_result_evidence(
        {"receipt_id": evidence_prefix + secret},
        redactor=redactor,
    )
    rendered = json.dumps(
        evidence.value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )

    assert evidence.included is True
    assert secret not in rendered
    assert secret[: len(secret) // 2] not in rendered
    assert REDACTED_SECRET in rendered
    assert len(rendered.encode("utf-8")) <= (tool_results_module._MAX_PORTABLE_EVIDENCE_UTF8_BYTES)


def test_exception_type_name_is_redacted_before_its_byte_bound() -> None:
    from cayu.runtime._diagnostics import MAX_DIAGNOSTIC_TYPE_UTF8_BYTES
    from cayu.vaults import REDACTED_SECRET, SecretRedactor

    secret = "exception-type-boundary-secret"
    exception_type = type(
        ("x" * 105) + secret + ("y" * 100),
        (RuntimeError,),
        {},
    )

    diagnostic = tool_results_module.exception_diagnostic(
        exception_type("safe message"),
        redactor=SecretRedactor(secret),
    )

    assert secret not in diagnostic.error_type
    assert secret[:8] not in diagnostic.error_type
    assert REDACTED_SECRET[:8] in diagnostic.error_type
    assert len(diagnostic.error_type.encode("utf-8")) <= (MAX_DIAGNOSTIC_TYPE_UTF8_BYTES)


def test_external_invalid_tool_output_is_durable_before_blocking_after_hook() -> None:
    class InvalidExternalTool(Tool):
        spec = ToolSpec(
            name="invalid_external_before_hook",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.EXTERNAL,
        )

        def __init__(self) -> None:
            self.external_effects = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            self.external_effects += 1
            return ToolResult.model_construct(
                content="charged",
                structured={"receipt_id": "receipt-before-hook", "invalid": float("nan")},
                artifacts=[],
                is_error=False,
            )

    class BlockingAfterToolHook(RuntimeHook):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            del context
            self.started.set()
            await self.release.wait()

    async def scenario() -> None:
        session_id = "sess_invalid_external_before_hook"
        store = InMemorySessionStore()
        tool = InvalidExternalTool()
        hook = BlockingAfterToolHook()
        provider = _portable_tool_boundary_provider(
            tool.spec.name,
            "call_invalid_external_before_hook",
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            runtime_hooks=[hook],
        )

        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "charge")],
                ),
            )
        )
        await asyncio.wait_for(hook.started.wait(), timeout=5)

        stored_while_hook_blocked = await store.load_events(session_id)
        stored_types = [event.type for event in stored_while_hook_blocked]
        terminal = next(
            event for event in stored_while_hook_blocked if event.type == EventType.TOOL_CALL_FAILED
        )
        assert tool.external_effects == 1
        assert terminal.payload["terminal_outcome"] == "invalid_tool_output"
        assert terminal.payload["tool_effect"] == "external"
        assert terminal.payload["outcome_unknown"] is True
        assert terminal.payload["manual_reconciliation_required"] is True
        assert stored_types.index(EventType.TOOL_CALL_FAILED) < stored_types.index(
            EventType.HOOK_STARTED
        )

        hook.release.set()
        events = await asyncio.wait_for(run_task, timeout=5)
        assert tool.external_effects == 1
        assert sum(event.type == EventType.TOOL_CALL_FAILED for event in events) == 1
        assert events[-1].type == EventType.SESSION_COMPLETED

    asyncio.run(scenario())


def test_external_invalid_tool_output_survives_swallowed_interruption_without_reexecution():
    class SwallowingExternalTool(Tool):
        spec = ToolSpec(
            name="swallowing_external_tool",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.EXTERNAL,
        )

        def __init__(self) -> None:
            self.started: asyncio.Event | None = None
            self.external_effects = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            if self.started is None:
                raise AssertionError("Tool test event was not initialized.")
            self.external_effects += 1
            self.started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                return ToolResult.model_construct(
                    content="charged",
                    structured={
                        "receipt_id": "receipt-after-cancellation",
                        "invalid": float("nan"),
                    },
                    artifacts=[],
                    is_error=False,
                )
            raise AssertionError("Tool should have been cancelled.")

    async def scenario():
        session_id = "sess_swallowed_external_interruption"
        store = InMemorySessionStore()
        tool = SwallowingExternalTool()
        tool.started = asyncio.Event()
        provider = _portable_tool_boundary_provider(
            tool.spec.name,
            "call_swallowed_external_interruption",
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "charge")],
                ),
            )
        )
        await tool.started.wait()
        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(session_id=session_id, reason="operator stop")
            )
        ]
        run_events = await run_task
        stored_after_interrupt = await store.load_events(session_id)
        resumed_events = await collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue")],
            ),
        )
        return tool, interrupt_events, run_events, stored_after_interrupt, resumed_events

    tool, interrupt_events, run_events, stored_events, resumed_events = asyncio.run(scenario())

    assert tool.external_effects == 1
    assert interrupt_events[-1].type == EventType.SESSION_INTERRUPTED
    assert run_events[-1].type == EventType.SESSION_INTERRUPTED
    terminal_events = [event for event in stored_events if event.type == EventType.TOOL_CALL_FAILED]
    assert len(terminal_events) == 1
    terminal = terminal_events[0]
    assert terminal.payload["terminal_outcome"] == "invalid_tool_output"
    assert terminal.payload["manual_reconciliation_required"] is True
    evidence = terminal.payload["result"]["structured"]["portable_result_evidence"]
    assert evidence["structured"]["receipt_id"] == "receipt-after-cancellation"
    assert resumed_events[-1].type == EventType.SESSION_COMPLETED
    assert tool.external_effects == 1


def test_external_invalid_tool_output_precedes_proxy_telemetry_failure_and_replays_once():
    class FailProxyTelemetryOnceStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def append_events(self, session_id: str, events: list[Event]) -> None:
            if not self.failed and any(
                event.type == EventType.CREDENTIAL_PROXY_CHECKED for event in events
            ):
                self.failed = True
                raise RuntimeError("proxy telemetry unavailable")
            await super().append_events(session_id, events)

    class InvalidProxyExternalTool(Tool):
        spec = ToolSpec(
            name="invalid_proxy_external_tool",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.EXTERNAL,
        )

        def __init__(self) -> None:
            self.external_effects = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            if ctx.proxy is None:
                raise AssertionError("Proxy was not configured.")
            await ctx.proxy.authorize_request(
                destination="https://api.example.test/charge",
                credential=SecretRef(name="service_key"),
                action="charge",
            )
            self.external_effects += 1
            return ToolResult.model_construct(
                content="charged",
                structured={"receipt_id": "receipt-before-proxy-failure", "invalid": float("nan")},
                artifacts=[],
                is_error=False,
            )

    session_id = "sess_terminal_before_proxy_failure"
    store = FailProxyTelemetryOnceStore()
    tool = InvalidProxyExternalTool()
    provider = _portable_tool_boundary_provider(
        tool.spec.name,
        "call_terminal_before_proxy_failure",
    )
    vault = StaticVault({"service_key": "proxy-secret"})
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            vault=vault,
            proxy=PassthroughProxy(vault),
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "charge")],
            ),
        )
    )

    assert initial_events[-1].type == EventType.SESSION_FAILED
    terminal = next(event for event in initial_events if event.type == EventType.TOOL_CALL_FAILED)
    assert terminal.payload["terminal_outcome"] == "invalid_tool_output"
    assert terminal.payload["manual_reconciliation_required"] is True
    assert tool.external_effects == 1

    resumed_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue")],
            ),
        )
    )

    assert resumed_events[-1].type == EventType.SESSION_COMPLETED
    assert tool.external_effects == 1
    stored_events = asyncio.run(store.load_events(session_id))
    assert sum(event.type == EventType.TOOL_CALL_STARTED for event in stored_events) == 1
    assert sum(event.type == EventType.TOOL_CALL_FAILED for event in stored_events) == 1


def test_terminal_tool_controls_survive_matching_secret_redaction():
    from cayu.vaults import REDACTED_SECRET, SecretRedactor

    error_path = "$/#1"
    control_values = [
        "external",
        "invalid_tool_output",
        "non_finite_number",
        error_path,
    ]
    observed: list[dict[str, Any] | None] = []

    class InvalidControlValueTool(Tool):
        spec = ToolSpec(
            name="invalid_control_value",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.EXTERNAL,
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            return ToolResult.model_construct(
                content="charged",
                structured={"receipt_id": "external", "auxiliary": float("nan")},
                artifacts=[],
                is_error=False,
            )

    class ControlObserver(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            observed.append(context.result.structured)

    store = InMemorySessionStore()
    provider = _portable_tool_boundary_provider(
        "invalid_control_value",
        "call_invalid_control_value",
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(control_values),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[InvalidControlValueTool()],
        runtime_hooks=[ControlObserver()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_terminal_control_redaction",
                messages=[Message.text("user", "charge")],
            ),
        )
    )

    terminal = next(event for event in events if event.type == EventType.TOOL_CALL_FAILED)
    expected_controls = {
        "terminal_outcome": "invalid_tool_output",
        "tool_effect": "external",
        "outcome_unknown": True,
        "manual_reconciliation_required": True,
        "durable_value_error_code": "non_finite_number",
        "durable_value_error_path": error_path,
    }
    assert {key: terminal.payload[key] for key in expected_controls} == expected_controls
    structured = terminal.payload["result"]["structured"]
    assert {key: structured[key] for key in expected_controls} == expected_controls
    assert structured["portable_result_evidence"]["structured"]["receipt_id"] == (REDACTED_SECRET)
    assert observed == [structured]
    transcript = asyncio.run(store.load_transcript("sess_terminal_control_redaction"))
    tool_part = next(message for message in transcript if message.role == "tool").content[0]
    assert tool_part.structured == structured
    assert provider.requests[1].messages[-1].content[0].structured == structured
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_secret_bearing_provider_call_id_fails_closed_before_tool_execution() -> None:
    from cayu.vaults import SecretRedactor

    call_id = "call_secret_identity"

    class InvalidExternalTool(Tool):
        spec = ToolSpec(
            name="invalid_external_secret_identity",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.EXTERNAL,
        )

        def __init__(self) -> None:
            self.effects = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            self.effects += 1
            return ToolResult.model_construct(
                content="charged",
                structured={"receipt_id": "receipt", "bad": float("nan")},
                artifacts=[],
                is_error=False,
            )

    session_id = "sess_secret_call_identity_replay"
    store = FailingOrdinaryToolResultCloseStore()
    tool = InvalidExternalTool()
    provider = _portable_tool_boundary_provider(tool.spec.name, call_id)
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(call_id),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "charge")],
            ),
        )
    )

    assert initial_events[-1].type == EventType.SESSION_FAILED
    assert all(event.type != EventType.TOOL_CALL_STARTED for event in initial_events)
    assert all(event.type != EventType.TOOL_CALL_FAILED for event in initial_events)
    assert call_id not in repr([event.model_dump(mode="json") for event in initial_events])
    assert tool.effects == 0
    completed = next(event for event in initial_events if event.type is EventType.MODEL_COMPLETED)
    assert completed.payload["step_classification"]["type"] == "failed"

    resumed_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue")],
            ),
        )
    )

    active_stage = asyncio.run(store.load_active_model_completion_stage(session_id))
    assert active_stage is None
    assert resumed_events[-1].type is EventType.SESSION_COMPLETED
    assert len(provider.requests) == 2
    assert tool.effects == 0
    stored_events = asyncio.run(store.load_events(session_id))
    stored_started = [event for event in stored_events if event.type == EventType.TOOL_CALL_STARTED]
    stored_terminal = [event for event in stored_events if event.type == EventType.TOOL_CALL_FAILED]
    assert stored_started == []
    assert stored_terminal == []
    transcript = asyncio.run(store.load_transcript(session_id))
    assert all(message.role != "tool" for message in transcript)
    assert call_id not in repr([message.model_dump(mode="json") for message in transcript])
