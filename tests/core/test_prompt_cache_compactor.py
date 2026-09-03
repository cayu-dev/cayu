from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import pytest
from tests._session_provenance import session_fixture

from cayu import (
    AgentSpec,
    CayuApp,
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    ContextPressureOverhead,
    ContextRequest,
    ContextUsageState,
    Environment,
    EnvironmentSpec,
    Event,
    EventType,
    IncompleteSessionRecoveryRequest,
    LocalArtifactStore,
    Message,
    ModelCompactor,
    ModelCompletionManualRecoveryRequest,
    ModelCompletionManualRecoveryRequired,
    ModelTarget,
    PromptCacheCompactor,
    ResumeRequest,
    RetryPolicy,
    RunRequest,
    SessionStatus,
    SQLiteSessionStore,
    TextPart,
    ThinkingConfig,
)
from cayu.artifacts import RESOLVED_FILE_ATTACHMENTS_OPTION
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import (
    ModelContextOverflowError,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamDeadlineError,
    ModelStreamEvent,
    ProviderDeadlineKind,
    ProviderProgressKind,
    ProviderStreamDeadlineEvidence,
    ProviderStreamDeadlines,
)
from cayu.runtime import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    InMemoryBudgetLedger,
    InMemorySessionStore,
)
from cayu.runtime.context import ContextBuildError
from cayu.runtime.costs import ModelPrice, PriceBook
from cayu.storage import SQLiteBudgetLedger


class RecordingProvider(ModelProvider):
    name = "recording"

    def __init__(self, events: list[ModelStreamEvent]) -> None:
        self.events = events
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self.events:
            yield event


class SequencedProvider(ModelProvider):
    name = "sequenced"

    def __init__(self, responses: list[list[ModelStreamEvent]]) -> None:
        self.responses = responses
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self.responses[len(self.requests) - 1]:
            yield event


class RetryOnceProvider(ModelProvider):
    name = "retry-once"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise ModelProviderError(
                "provider overloaded",
                provider=self.name,
                status_code=529,
                retryable=True,
            )
        yield ModelStreamEvent.text_delta("recovered summary")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class ToolCallFailureProvider(ModelProvider):
    name = "tool-call-failure"

    def __init__(self, failure_kind: str) -> None:
        self.failure_kind = failure_kind
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ModelStreamEvent.tool_call(id="call_1", name="inspect_report", arguments={})
            if self.failure_kind == "event":
                yield ModelStreamEvent.error("stream failed after tool call")
                return
            if self.failure_kind == "post_completion_exception":
                yield ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_use",
                        "usage": {"input_tokens": 100, "output_tokens": 10},
                    }
                )
            raise RuntimeError("transport failed after tool call")
        yield ModelStreamEvent.text_delta("bounded summary")
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {"input_tokens": 20, "output_tokens": 5},
            }
        )


class ToolCallDeadlineProvider(ModelProvider):
    name = "tool-call-deadline"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    @property
    def stream_deadlines(self) -> ProviderStreamDeadlines:
        return ProviderStreamDeadlines(
            transport_idle_timeout_s=1,
            protocol_idle_timeout_s=1,
            semantic_progress_timeout_s=0.01,
            absolute_stream_timeout_s=1,
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        yield ModelStreamEvent.tool_call(
            id="call_1",
            name="inspect_report",
            arguments={},
        )
        await asyncio.Event().wait()
        yield ModelStreamEvent.completed({})  # pragma: no cover


class RestartStableRecordingProvider(RecordingProvider):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="test.recording-provider",
            behavior_version="1",
            implementation_version="1",
        )


class RestartStableToolCallDeadlineProvider(ToolCallDeadlineProvider):
    @property
    def billing_provider_name(self) -> str:
        return "tool-call-deadline-billing"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="test.compaction-deadline-provider",
            behavior_version="1",
            implementation_version="1",
        )


class RestartStablePromptCacheCompactor(PromptCacheCompactor):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="test.prompt-cache-compactor",
            behavior_version="1",
            implementation_version="1",
        )


class RejectContextCompactionReceiptStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.rejected_receipts = 0

    async def mark_model_completion_stage_dispatched(self, session_id, *, stage):
        if stage.purpose == "context-compaction":
            self.rejected_receipts += 1
            raise RuntimeError("context-compaction receipt rejected before commit")
        return await super().mark_model_completion_stage_dispatched(
            session_id,
            stage=stage,
        )


class RejectPreProviderReleaseLedger(InMemoryBudgetLedger):
    def __init__(self) -> None:
        super().__init__(reservation_ttl_seconds=None)
        self.reject_releases = True
        self.release_attempts = 0

    async def release_pre_provider_dispatch(self, **kwargs):
        self.release_attempts += 1
        if self.reject_releases:
            raise RuntimeError("pre-provider compaction release rejected")
        return await super().release_pre_provider_dispatch(**kwargs)


_CONTEXT_COMPACTION_RECEIPT_EXIT_CODE = 86
_OVERFLOW_COMPACTION_DISPATCH_EXIT_CODE = 87


class ExitBeforeContextCompactionReceiptStore(SQLiteSessionStore):
    invocation_lifecycle_command_version = 1

    async def mark_model_completion_stage_dispatched(self, session_id, *, stage):
        if stage.purpose == "context-compaction":
            os._exit(_CONTEXT_COMPACTION_RECEIPT_EXIT_CODE)
        return await super().mark_model_completion_stage_dispatched(
            session_id,
            stage=stage,
        )


class MarkerToolCallDeadlineProvider(RestartStableToolCallDeadlineProvider):
    def __init__(self, marker: str) -> None:
        super().__init__()
        self._marker = Path(marker)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self._marker.write_text("provider entered", encoding="utf-8")
        async for event in super().stream(request):
            yield event


class ExitDuringOverflowCompactionProvider(RestartStableToolCallDeadlineProvider):
    def __init__(self, marker: str) -> None:
        super().__init__()
        self._marker = Path(marker)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        self._marker.write_text("provider entered", encoding="utf-8")
        os._exit(_OVERFLOW_COMPACTION_DISPATCH_EXIT_CODE)
        yield ModelStreamEvent.completed({})  # pragma: no cover


class OverflowThenSuccessProvider(RestartStableRecordingProvider):
    def __init__(self) -> None:
        super().__init__([])

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            overflow = ModelContextOverflowError(
                "context too large",
                provider=self.name,
                status_code=400,
                error_code="context_length_exceeded",
            )
            yield ModelStreamEvent.error(str(overflow), cause=overflow)
            return
        yield ModelStreamEvent.text_delta("answer after compaction")
        yield ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}})


class SuccessfulOverflowCompactionProvider(RestartStableToolCallDeadlineProvider):
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        yield ModelStreamEvent.text_delta("bounded summary")
        yield ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}})


def _restart_compaction_budget_policy(
    *,
    assistant_model: str,
    compactor_model: str,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("1"),
                pricing=PriceBook(
                    prices=(
                        ModelPrice.fixed(
                            provider_name="recording",
                            model=assistant_model,
                            input_per_million=Decimal("1"),
                            output_per_million=Decimal("1"),
                        ),
                        ModelPrice.fixed(
                            provider_name="tool-call-deadline-billing",
                            model=compactor_model,
                            input_per_million=Decimal("1"),
                            output_per_million=Decimal("1"),
                        ),
                    )
                ),
                reservation=BudgetReservation(
                    max_input_tokens=1_000,
                    max_output_tokens=1_000,
                ),
            ),
        )
    )


def _resume_until_context_compaction_receipt_exit(
    database: str,
    budget_database: str,
    provider_marker: str,
) -> None:
    async def run() -> None:
        assistant_model = "assistant-model"
        compactor_model = "compactor-model"
        store = ExitBeforeContextCompactionReceiptStore(database)
        budget_ledger = SQLiteBudgetLedger(budget_database)
        main_provider = RestartStableRecordingProvider(
            [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}})]
        )
        compactor_provider = MarkerToolCallDeadlineProvider(provider_marker)
        app = CayuApp(
            session_store=store,
            budget_policy=_restart_compaction_budget_policy(
                assistant_model=assistant_model,
                compactor_model=compactor_model,
            ),
            budget_ledger=budget_ledger,
            enable_logging=False,
        )
        app.register_provider(main_provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model=assistant_model),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RestartStablePromptCacheCompactor(
                    provider=compactor_provider,
                    model=compactor_model,
                ),
                max_user_turns=1,
                compact_after_messages=2,
            ),
        )
        async for _event in app.resume(
            ResumeRequest(
                session_id="app-compaction-receipt-process-loss",
                messages=[Message.text("user", "second request")],
            )
        ):
            pass

    asyncio.run(run())


def _run_until_overflow_compaction_dispatch_exit(
    database: str,
    budget_database: str,
    provider_marker: str,
) -> None:
    async def run() -> None:
        assistant_model = "assistant-model"
        compactor_model = "compactor-model"
        overflow = ModelContextOverflowError(
            "context too large",
            provider="recording",
            status_code=400,
            error_code="context_length_exceeded",
        )
        main_provider = RestartStableRecordingProvider(
            [ModelStreamEvent.error(str(overflow), cause=overflow)]
        )
        compactor_provider = ExitDuringOverflowCompactionProvider(provider_marker)
        app = CayuApp(
            session_store=SQLiteSessionStore(database),
            budget_policy=_restart_compaction_budget_policy(
                assistant_model=assistant_model,
                compactor_model=compactor_model,
            ),
            budget_ledger=SQLiteBudgetLedger(budget_database),
            enable_logging=False,
        )
        app.register_provider(main_provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model=assistant_model),
            context_overflow_policy=CheckpointCompactionContextPolicy(
                compactor=RestartStablePromptCacheCompactor(
                    provider=compactor_provider,
                    model=compactor_model,
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        async for _event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="overflow-compaction-dispatch-process-loss",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "current request"),
                ],
            )
        ):
            pass

    asyncio.run(run())


class ToolThenBoundedProviderErrorProvider(ModelProvider):
    name = "tool-then-bounded-error"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.bounded_error = ModelProviderError(
            "bounded provider unavailable",
            provider=self.name,
            status_code=503,
            error_code="service_unavailable",
            retryable=False,
            retry_after_s=2.5,
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ModelStreamEvent.tool_call(
                id="call_1",
                name="inspect_report",
                arguments={},
            )
            return
        raise self.bounded_error


class InspectReportTool(Tool):
    spec = ToolSpec(
        name="inspect_report",
        description="Inspect a report.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content="inspected")


async def collect_events(stream) -> list:
    return [event async for event in stream]


def _exception_group_leaves(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in _exception_group_leaves(child)]
    return [error]


def test_prompt_cache_compactor_extends_the_exact_model_request_prefix() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("cache-aware summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[
            Message.text("system", "You are careful."),
            Message.text("user", "Inspect the attached report with the registered tool."),
        ],
        tools=[
            {
                "name": "inspect_report",
                "description": "Inspect a report.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        options={
            "anthropic": {
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
                "max_tokens": 4096,
            },
            "thinking": {"enabled": True, "effort": "high"},
            "structured_output": {"strategy": "native"},
            RESOLVED_FILE_ATTACHMENTS_OPTION: {
                "report": {
                    "artifact_id": "report",
                    "kind": "document",
                    "filename": "report.pdf",
                    "content_type": "application/pdf",
                    "data_base64": "JVBERg==",
                    "metadata": {},
                }
            },
        },
    )
    compactor = PromptCacheCompactor(
        provider=provider,
        options={"anthropic": {"max_tokens": 512}},
    )

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=session_fixture(
                    id="prompt-cache-prefix",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=[Message.text("user", "newly compactable")],
                context_messages=cached_request.messages,
                cache_prefix_request=cached_request,
            )
        )
    )

    sent = provider.requests[0]
    assert result.summary == "cache-aware summary"
    assert sent.model == cached_request.model
    assert sent.tools == cached_request.tools
    assert sent.messages[:-1] == cached_request.messages
    assert sent.options["thinking"] == {"enabled": True, "effort": "high"}
    assert (
        sent.options[RESOLVED_FILE_ATTACHMENTS_OPTION]
        == cached_request.options[RESOLVED_FILE_ATTACHMENTS_OPTION]
    )
    assert sent.options["structured_output"] is None
    assert sent.options["anthropic"] == {
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
        "max_tokens": 512,
    }


def test_prompt_cache_compactor_uses_bounded_cross_model_fallback_for_override() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("cross-model summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "cached context")],
    )

    result = asyncio.run(
        PromptCacheCompactor(
            provider=provider,
            model="different-model",
        ).compact(
            CompactionRequest(
                session=session_fixture(
                    id="prompt-cache-model-mismatch",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=[Message.text("user", "newly compactable context")],
                context_messages=cached_request.messages,
                cache_prefix_request=cached_request,
            )
        )
    )

    assert result.summary == "cross-model summary"
    assert result.metadata["compactor"] == "ModelCompactor"
    assert provider.requests[0].model == "different-model"
    assert provider.requests[0].tools == []
    assert "newly compactable context" in provider.requests[0].messages[1].content[0].text


def test_prompt_cache_compactor_uses_bounded_fallback_when_session_model_changed() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("current-model summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    cached_request = ModelRequest(
        model="old-model",
        messages=[Message.text("user", "full cached context")],
        tools=[
            {
                "name": "inspect_report",
                "description": "Inspect a report.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    )

    result = asyncio.run(
        PromptCacheCompactor(provider=provider).compact(
            CompactionRequest(
                session=session_fixture(
                    id="prompt-cache-session-model-changed",
                    agent_name="assistant",
                    provider_name="recording",
                    model="new-model",
                ),
                agent=AgentSpec(name="assistant", model="new-model"),
                messages=[Message.text("user", "newly compactable context")],
                context_messages=cached_request.messages,
                cache_prefix_request=cached_request,
            )
        )
    )

    assert result.summary == "current-model summary"
    assert result.metadata["compactor"] == "ModelCompactor"
    assert provider.requests[0].model == "new-model"
    assert provider.requests[0].tools == []
    assert "full cached context" not in provider.requests[0].messages[1].content[0].text


def test_prompt_cache_compactor_requires_model_for_cross_provider_compaction() -> None:
    provider = RecordingProvider([])
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "full cached context")],
    )

    with pytest.raises(
        ValueError,
        match="model is required when the compactor provider differs",
    ):
        asyncio.run(
            PromptCacheCompactor(provider=provider).compact(
                CompactionRequest(
                    session=session_fixture(
                        id="prompt-cache-provider-model-required",
                        agent_name="assistant",
                        provider_name="original-provider",
                        model="claude-sonnet-4-6",
                    ),
                    agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                    messages=[Message.text("user", "newly compactable context")],
                    context_messages=cached_request.messages,
                    cache_prefix_request=cached_request,
                    force_bounded_compaction=True,
                )
            )
        )

    assert provider.requests == []


def test_prompt_cache_compactor_uses_explicit_model_for_cross_provider_compaction() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("cross-provider summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "full cached context")],
        tools=[
            {
                "name": "inspect_report",
                "description": "Inspect a report.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        options={
            RESOLVED_FILE_ATTACHMENTS_OPTION: {
                "report": {
                    "artifact_id": "report",
                    "kind": "document",
                    "filename": "report.pdf",
                    "content_type": "application/pdf",
                    "data_base64": "JVBERg==",
                    "metadata": {},
                }
            }
        },
    )

    result = asyncio.run(
        PromptCacheCompactor(provider=provider, model="gpt-4.1-mini").compact(
            CompactionRequest(
                session=session_fixture(
                    id="prompt-cache-provider-mismatch",
                    agent_name="assistant",
                    provider_name="original-provider",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=[Message.text("user", "newly compactable context")],
                context_messages=cached_request.messages,
                cache_prefix_request=cached_request,
            )
        )
    )

    assert result.summary == "cross-provider summary"
    assert result.metadata["compactor"] == "ModelCompactor"
    assert provider.requests[0].model == "gpt-4.1-mini"
    assert provider.requests[0].tools == []
    assert RESOLVED_FILE_ATTACHMENTS_OPTION not in provider.requests[0].options
    assert "full cached context" not in provider.requests[0].messages[1].content[0].text
    assert "newly compactable context" in provider.requests[0].messages[1].content[0].text


def test_prompt_cache_compactor_uses_bounded_fallback_for_tool_structured_output() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("plain summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[
            Message.text("system", "You are careful."),
            Message.text(
                "system",
                "Call `__cayu_submit_structured_output` with the final answer.",
            ),
            Message.text("user", "return a report"),
        ],
        tools=[
            {
                "name": "__cayu_submit_structured_output",
                "description": "Submit structured output.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        options={"structured_output": {"strategy": "tool"}},
    )

    result = asyncio.run(
        PromptCacheCompactor(provider=provider).compact(
            CompactionRequest(
                session=session_fixture(
                    id="prompt-cache-tool-structured-output",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=[Message.text("user", "newly compactable context")],
                context_messages=cached_request.messages,
                cache_prefix_request=cached_request,
            )
        )
    )

    assert result.summary == "plain summary"
    assert result.metadata["compactor"] == "ModelCompactor"
    assert provider.requests[0].tools == []
    assert [message.role for message in provider.requests[0].messages] == ["system", "user"]
    assert "__cayu_submit_structured_output" not in str(provider.requests[0].model_dump())


def test_prompt_cache_compactor_degrades_exact_tool_call_to_bounded_input() -> None:
    provider = SequencedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="inspect_report",
                    arguments={},
                ),
                ModelStreamEvent.completed(
                    {
                        "model": "claude-sonnet-4-6-20260601",
                        "finish_reason": "tool_use",
                        "usage": {"input_tokens": 100, "output_tokens": 10},
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("bounded summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "claude-sonnet-4-6-20260601",
                        "finish_reason": "stop",
                        "usage": {"input_tokens": 20, "output_tokens": 5},
                    }
                ),
            ],
        ]
    )
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "full cached context")],
        tools=[
            {
                "name": "inspect_report",
                "description": "Inspect a report.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    )

    result = asyncio.run(
        PromptCacheCompactor(provider=provider).compact(
            CompactionRequest(
                session=session_fixture(
                    id="prompt-cache-tool-call-degradation",
                    agent_name="assistant",
                    provider_name="sequenced",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=[Message.text("user", "newly compactable context")],
                context_messages=cached_request.messages,
                cache_prefix_request=cached_request,
            )
        )
    )

    assert result.summary == "bounded summary"
    assert len(provider.requests) == 2
    assert provider.requests[0].tools == cached_request.tools
    assert provider.requests[1].tools == []
    assert "full cached context" not in str(provider.requests[1].model_dump())
    assert result.metadata["prompt_cache_exact_attempt"] == "rejected_tool_call"
    assert [payload["compactor"] for payload in result.model_completed_payloads] == [
        "PromptCacheCompactor",
        "ModelCompactor",
    ]
    assert [
        payload.get("usage_metrics", {}).get("input_tokens")
        for payload in result.model_completed_payloads
    ] == [100, 20]


@pytest.mark.parametrize("failure_kind", ["event", "exception"])
def test_prompt_cache_compactor_degrades_when_exact_tool_call_stream_fails(
    failure_kind: str,
) -> None:
    provider = ToolCallFailureProvider(failure_kind)
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "full cached context")],
        tools=[
            {
                "name": "inspect_report",
                "description": "Inspect a report.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    )

    result = asyncio.run(
        PromptCacheCompactor(provider=provider).compact(
            CompactionRequest(
                session=session_fixture(
                    id=f"prompt-cache-tool-call-{failure_kind}",
                    agent_name="assistant",
                    provider_name=provider.name,
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=[Message.text("user", "newly compactable context")],
                context_messages=cached_request.messages,
                cache_prefix_request=cached_request,
            )
        )
    )

    assert result.summary == "bounded summary"
    assert len(provider.requests) == 2
    assert provider.requests[1].tools == []
    assert result.model_completed_payloads[0]["compaction_outcome"] == ("rejected_tool_call")
    assert result.model_completed_payloads[0]["usage_unavailable_reason"] == (
        "compaction tool-call attempt ended without provider completion usage"
    )
    assert "usage_metrics" not in result.model_completed_payloads[0]


def test_prompt_cache_compactor_does_not_degrade_typed_deadline_after_tool_call() -> None:
    provider = ToolCallDeadlineProvider()
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "full cached context")],
        tools=[
            {
                "name": "inspect_report",
                "description": "Inspect a report.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    )

    with pytest.raises(ModelStreamDeadlineError) as captured:
        asyncio.run(
            PromptCacheCompactor(provider=provider).compact(
                CompactionRequest(
                    session=session_fixture(
                        id="prompt-cache-tool-call-deadline",
                        agent_name="assistant",
                        provider_name=provider.name,
                        model="claude-sonnet-4-6",
                    ),
                    agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                    messages=[Message.text("user", "newly compactable context")],
                    context_messages=cached_request.messages,
                    cache_prefix_request=cached_request,
                )
            )
        )

    assert len(provider.requests) == 1
    evidence = captured.value.deadline_evidence
    assert evidence.deadline_kind is ProviderDeadlineKind.SEMANTIC_IDLE
    assert evidence.last_progress_kind is ProviderProgressKind.TOOL_CALL


def test_model_compactor_detaches_provider_failure_after_tool_call() -> None:
    secret = "transport\x00workload-secret-value"

    class SecretToolFailureProvider(ModelProvider):
        name = "secret-tool-failure"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            yield ModelStreamEvent.tool_call(
                id="call_1",
                name="inspect_report",
                arguments={},
            )
            raise RuntimeError(secret)

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(
            ModelCompactor(
                provider=SecretToolFailureProvider(),
                model="summary-model",
            ).compact(
                CompactionRequest(
                    session=session_fixture(
                        id="model-compactor-secret-tool-failure",
                        agent_name="assistant",
                        provider_name="secret-tool-failure",
                        model="summary-model",
                    ),
                    agent=AgentSpec(name="assistant", model="summary-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )

    failure = exc_info.value
    assert type(failure).__name__ == "_CompactionToolCallError"
    assert failure.__cause__ is None
    assert failure.__context__ is None
    assert "workload-secret-value" not in repr(failure)


def test_prompt_cache_compactor_preserves_bounded_provider_error_after_tool_degradation() -> None:
    provider = ToolThenBoundedProviderErrorProvider()
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "full cached context")],
        tools=[
            {
                "name": "inspect_report",
                "description": "Inspect a report.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    )

    with pytest.raises(ModelProviderError) as exc_info:
        asyncio.run(
            PromptCacheCompactor(provider=provider).compact(
                CompactionRequest(
                    session=session_fixture(
                        id="prompt-cache-tool-call-bounded-provider-error",
                        agent_name="assistant",
                        provider_name=provider.name,
                        model="claude-sonnet-4-6",
                    ),
                    agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                    messages=[Message.text("user", "newly compactable context")],
                    context_messages=cached_request.messages,
                    cache_prefix_request=cached_request,
                )
            )
        )

    assert len(provider.requests) == 2
    assert provider.requests[1].tools == []
    detached = exc_info.value
    assert detached is not provider.bounded_error
    assert str(detached) == str(provider.bounded_error)
    assert detached.error_payload_fields() == provider.bounded_error.error_payload_fields()
    assert detached.response_body is None
    assert detached.__cause__ is None
    assert detached.__suppress_context__ is True


def test_prompt_cache_compaction_failure_telemetry_is_invocation_scoped() -> None:
    provider = ToolThenBoundedProviderErrorProvider()
    policy = CheckpointCompactionContextPolicy(
        compactor=PromptCacheCompactor(provider=provider),
        max_user_turns=1,
        compact_after_messages=2,
    )
    messages = [
        Message.text("user", "old request"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current request"),
    ]

    async def build_cache_prefix_request(context_messages: list[Message]) -> ModelRequest:
        return ModelRequest(
            model="claude-sonnet-4-6",
            messages=context_messages,
            tools=[
                {
                    "name": "inspect_report",
                    "description": "Inspect a report.",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        )

    def context_request(*, force_bounded_compaction: bool) -> ContextRequest:
        return ContextRequest(
            session=session_fixture(
                id="prompt-cache-failure-telemetry-scope",
                agent_name="assistant",
                provider_name=provider.name,
                model="claude-sonnet-4-6",
            ),
            agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
            messages=messages,
            step=1,
            context_usage=ContextUsageState(
                last_transcript_cursor=2,
                last_provider_name=provider.name,
                last_requested_model="claude-sonnet-4-6",
            ),
            build_cache_prefix_request=build_cache_prefix_request,
            force_bounded_compaction=force_bounded_compaction,
        )

    with pytest.raises(ContextBuildError) as exact_failure:
        asyncio.run(
            policy.build_with_checkpoint(
                context_request(force_bounded_compaction=False),
                checkpoint=None,
            )
        )

    assert len(provider.requests) == 2
    exact_cause = exact_failure.value.cause
    assert isinstance(exact_cause, ModelProviderError)
    assert exact_cause is not provider.bounded_error
    assert str(exact_cause) == str(provider.bounded_error)
    assert exact_cause.error_payload_fields() == provider.bounded_error.error_payload_fields()
    assert exact_cause.__cause__ is None
    assert exact_cause.__suppress_context__ is True
    assert [telemetry.event_type for telemetry in exact_failure.value.compaction_telemetry] == [
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
    ]
    exact_attempts = [
        telemetry
        for telemetry in exact_failure.value.compaction_telemetry
        if telemetry.event_type == EventType.MODEL_COMPLETED
    ]
    assert [item.payload["compaction_outcome"] for item in exact_attempts] == [
        "rejected_tool_call",
        "provider_error",
    ]
    exact_completion = exact_attempts[0]
    assert exact_completion.payload["compaction_outcome"] == "rejected_tool_call"
    assert exact_completion.payload["usage_unavailable_reason"] == (
        "compaction tool-call attempt ended without provider completion usage"
    )
    assert "usage_metrics" not in exact_completion.payload

    with pytest.raises(ContextBuildError) as bounded_only_failure:
        asyncio.run(
            policy.build_with_checkpoint(
                context_request(force_bounded_compaction=True),
                checkpoint=None,
            )
        )

    assert len(provider.requests) == 3
    bounded_cause = bounded_only_failure.value.cause
    assert isinstance(bounded_cause, ModelProviderError)
    assert bounded_cause is not provider.bounded_error
    assert str(bounded_cause) == str(provider.bounded_error)
    assert bounded_cause.error_payload_fields() == provider.bounded_error.error_payload_fields()
    assert bounded_cause.__cause__ is None
    assert bounded_cause.__context__ is None
    assert [
        telemetry.event_type for telemetry in bounded_only_failure.value.compaction_telemetry
    ] == [
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
    ]
    bounded_attempt = bounded_only_failure.value.compaction_telemetry[1]
    assert bounded_attempt.payload["compaction_outcome"] == "provider_error"
    assert bounded_attempt.payload["usage_unavailable_reason"] == (
        "compaction provider dispatch failed without completion usage"
    )


def test_prompt_cache_compactor_retains_usage_before_post_completion_stream_failure() -> None:
    provider = ToolCallFailureProvider("post_completion_exception")
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "full cached context")],
        tools=[
            {
                "name": "inspect_report",
                "description": "Inspect a report.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    )

    result = asyncio.run(
        PromptCacheCompactor(provider=provider).compact(
            CompactionRequest(
                session=session_fixture(
                    id="prompt-cache-tool-call-post-completion-error",
                    agent_name="assistant",
                    provider_name=provider.name,
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=[Message.text("user", "newly compactable context")],
                context_messages=cached_request.messages,
                cache_prefix_request=cached_request,
            )
        )
    )

    assert result.summary == "bounded summary"
    assert len(provider.requests) == 2
    assert result.model_completed_payloads[0]["compaction_outcome"] == ("rejected_tool_call")
    assert result.model_completed_payloads[0]["usage_metrics"]["input_tokens"] == 100
    assert "usage_unavailable_reason" not in result.model_completed_payloads[0]


def test_prompt_cache_compactor_falls_back_without_an_exact_request() -> None:
    provider = RecordingProvider([])

    result = asyncio.run(
        PromptCacheCompactor(provider=provider).compact(
            CompactionRequest(
                session=session_fixture(
                    id="prompt-cache-unavailable",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=[Message.text("user", "newly compactable context")],
                context_messages=[Message.text("user", "not a real provider request")],
            )
        )
    )

    assert result.metadata["compactor"] == "TranscriptDigestCompactor"
    assert provider.requests == []


def test_prompt_cache_compactor_accounts_for_usage_and_ignores_thinking() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.thinking("internal compaction reasoning"),
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed(
                {
                    "model": "claude-sonnet-4-6-20260601",
                    "usage": {"input_tokens": 100, "output_tokens": 10},
                }
            ),
        ]
    )
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "long cached context")],
    )

    result = asyncio.run(
        PromptCacheCompactor(provider=provider).compact(
            CompactionRequest(
                session=session_fixture(
                    id="prompt-cache-usage",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=[Message.text("user", "long cached context")],
                context_messages=cached_request.messages,
                cache_prefix_request=cached_request,
            )
        )
    )

    assert result.summary == "summary"
    assert result.model_completed_payloads == [
        {
            "model": "claude-sonnet-4-6-20260601",
            "usage": {"input_tokens": 100, "output_tokens": 10},
            "provider_name": "recording",
            "requested_model": "claude-sonnet-4-6",
            "purpose": "context_compaction",
            "compactor": "PromptCacheCompactor",
            "usage_metrics": {
                "provider_name": "recording",
                "requested_model": "claude-sonnet-4-6",
                "model": "claude-sonnet-4-6-20260601",
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
                "reasoning_output_tokens": 0,
                "cache": {
                    "read_tokens": 0,
                    "write_tokens": 0,
                    "cached_input_tokens": 0,
                    "uncached_input_tokens": 100,
                },
            },
        }
    ]


def test_prompt_cache_compactor_retries_structured_provider_errors() -> None:
    provider = RetryOnceProvider()
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "long cached context")],
    )

    result = asyncio.run(
        PromptCacheCompactor(
            provider=provider,
            retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
        ).compact(
            CompactionRequest(
                session=session_fixture(
                    id="prompt-cache-retry",
                    agent_name="assistant",
                    provider_name=provider.name,
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=[Message.text("user", "long cached context")],
                context_messages=cached_request.messages,
                cache_prefix_request=cached_request,
            )
        )
    )

    assert result.summary == "recovered summary"
    assert len(provider.requests) == 2


def test_checkpoint_policy_builds_the_cache_prefix_with_runtime_request_shape() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    policy = CheckpointCompactionContextPolicy(
        compactor=PromptCacheCompactor(provider=provider),
        max_user_turns=1,
        compact_after_messages=2,
    )
    messages = [
        Message.text("system", "You are careful."),
        Message.text("user", "old request"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current request"),
    ]

    async def build_cache_prefix_request(context_messages: list[Message]) -> ModelRequest:
        return ModelRequest(
            model="claude-sonnet-4-6",
            messages=context_messages,
            tools=[
                {
                    "name": "inspect_report",
                    "description": "Inspect a report.",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            options={"thinking": {"enabled": True, "effort": "high"}},
        )

    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=session_fixture(
                    id="checkpoint-cache-prefix",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=messages,
                step=1,
                context_usage=ContextUsageState(
                    last_transcript_cursor=3,
                    last_provider_name="recording",
                    last_requested_model="claude-sonnet-4-6",
                ),
                build_cache_prefix_request=build_cache_prefix_request,
            ),
            checkpoint=None,
        )
    )

    assert result.checkpoint is not None
    assert provider.requests[0].tools[0]["name"] == "inspect_report"
    assert provider.requests[0].options["thinking"] == {
        "enabled": True,
        "effort": "high",
    }
    assert provider.requests[0].messages[:-1] == messages


def test_checkpoint_policy_reports_start_when_cache_prefix_build_fails() -> None:
    provider = RecordingProvider([])
    policy = CheckpointCompactionContextPolicy(
        compactor=PromptCacheCompactor(provider=provider),
        max_user_turns=1,
        compact_after_messages=2,
    )

    async def failing_builder(context_messages: list[Message]) -> ModelRequest:
        assert context_messages
        raise RuntimeError("cache prefix construction failed")

    with pytest.raises(ContextBuildError, match="cache prefix construction failed") as exc_info:
        asyncio.run(
            policy.build_with_checkpoint(
                ContextRequest(
                    session=session_fixture(
                        id="checkpoint-cache-prefix-failure",
                        agent_name="assistant",
                        provider_name="recording",
                        model="claude-sonnet-4-6",
                    ),
                    agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                    messages=[
                        Message.text("user", "old request"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current request"),
                    ],
                    step=1,
                    context_usage=ContextUsageState(
                        last_transcript_cursor=2,
                        last_provider_name="recording",
                        last_requested_model="claude-sonnet-4-6",
                    ),
                    build_cache_prefix_request=failing_builder,
                ),
                checkpoint=None,
            )
        )

    assert [telemetry.event_type for telemetry in exc_info.value.compaction_telemetry] == [
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_FAILED,
    ]
    assert all(
        "bounded_input" not in telemetry.payload
        for telemetry in exc_info.value.compaction_telemetry
    )
    assert provider.requests == []


def test_prompt_cache_digest_exhaustion_can_progress_on_bounded_followup() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("provider summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    policy = CheckpointCompactionContextPolicy(
        compactor=PromptCacheCompactor(provider=provider),
        max_user_turns=1,
        compact_after_messages=1,
    )
    messages = [
        Message.text("user", "oversized " + "x" * 10_000),
        Message.text("user", "current"),
    ]

    first = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=session_fixture(
                    id="prompt-cache-digest-exhaustion",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=messages,
                step=1,
            ),
            checkpoint=None,
        )
    )

    assert first.checkpoint is not None
    assert first.checkpoint["context_compaction"]["compacted_transcript_cursor"] == 0
    assert first.checkpoint["context_compaction"]["progress"]["exhausted"] is True
    assert provider.requests == []

    second = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=session_fixture(
                    id="prompt-cache-digest-exhaustion",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=messages,
                step=2,
            ),
            checkpoint=first.checkpoint,
        )
    )

    assert second.checkpoint is not None
    assert second.checkpoint["context_compaction"]["compacted_transcript_cursor"] == 1
    assert second.checkpoint["context_compaction"]["summary"] == "provider summary"
    assert "progress" not in second.checkpoint["context_compaction"]
    assert len(provider.requests) == 1


@pytest.mark.parametrize("last_transcript_cursor", [None, 0, 3])
def test_prompt_cache_digest_exhaustion_uses_fallback_key_without_valid_usage_cursor(
    last_transcript_cursor: int | None,
) -> None:
    provider = RecordingProvider([])
    policy = CheckpointCompactionContextPolicy(
        compactor=PromptCacheCompactor(provider=provider),
        max_user_turns=1,
        compact_after_messages=1,
    )
    messages = [
        Message.text("user", "oversized " + "x" * 10_000),
        Message.text("user", "current"),
    ]

    async def unexpected_cache_prefix_builder(_messages: list[Message]) -> ModelRequest:
        raise AssertionError("an invalid usage cursor cannot reconstruct a cache prefix")

    request = ContextRequest(
        session=session_fixture(
            id="prompt-cache-missing-usage-cursor",
            agent_name="assistant",
            provider_name="recording",
            model="claude-sonnet-4-6",
        ),
        agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
        messages=messages,
        step=1,
        context_usage=ContextUsageState(
            last_transcript_cursor=last_transcript_cursor,
            last_provider_name="recording",
            last_requested_model="claude-sonnet-4-6",
        ),
        build_cache_prefix_request=unexpected_cache_prefix_builder,
    )
    first = asyncio.run(policy.build_with_checkpoint(request, checkpoint=None))

    assert first.checkpoint is not None
    compacted = first.checkpoint["context_compaction"]
    assert compacted["compacted_transcript_cursor"] == 0
    assert compacted["progress"]["exhausted"] is True
    assert compacted["progress"]["key"].startswith("transcript-digest:v2:")
    assert provider.requests == []


def test_checkpoint_policy_skips_cache_request_after_provider_identity_changes() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("bounded summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    policy = CheckpointCompactionContextPolicy(
        compactor=PromptCacheCompactor(provider=provider),
        max_user_turns=1,
        compact_after_messages=2,
    )
    messages = [
        Message.text("user", "old request"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current request"),
    ]

    async def unexpected_builder(context_messages: list[Message]) -> ModelRequest:
        raise AssertionError("a different provider cannot reuse the prior request cache")

    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=session_fixture(
                    id="checkpoint-provider-changed",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=messages,
                step=1,
                context_usage=ContextUsageState(
                    last_transcript_cursor=2,
                    last_provider_name="previous-provider",
                    last_requested_model="claude-sonnet-4-6",
                ),
                build_cache_prefix_request=unexpected_builder,
            ),
            checkpoint=None,
        )
    )

    assert result.checkpoint is not None
    assert result.checkpoint["context_compaction"]["metadata"]["compactor"] == "ModelCompactor"
    assert len(provider.requests) == 1
    assert provider.requests[0].tools == []


def test_checkpoint_policy_skips_cache_request_after_requested_model_changes() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("bounded summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    policy = CheckpointCompactionContextPolicy(
        compactor=PromptCacheCompactor(provider=provider),
        max_user_turns=1,
        compact_after_messages=2,
    )
    messages = [
        Message.text("user", "old request"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current request"),
    ]

    async def unexpected_builder(context_messages: list[Message]) -> ModelRequest:
        raise AssertionError("a different requested model cannot reuse the prior request cache")

    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=session_fixture(
                    id="checkpoint-requested-model-changed",
                    agent_name="assistant",
                    provider_name="recording",
                    model="new-model",
                ),
                agent=AgentSpec(name="assistant", model="new-model"),
                messages=messages,
                step=1,
                context_usage=ContextUsageState(
                    last_transcript_cursor=2,
                    last_provider_name="recording",
                    last_requested_model="old-model",
                ),
                build_cache_prefix_request=unexpected_builder,
            ),
            checkpoint=None,
        )
    )

    assert result.checkpoint is not None
    assert result.checkpoint["context_compaction"]["metadata"]["compactor"] == "ModelCompactor"
    assert len(provider.requests) == 1
    assert provider.requests[0].model == "new-model"
    assert provider.requests[0].tools == []


def test_checkpoint_policy_skips_exact_builder_for_tool_structured_output() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("bounded summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    policy = CheckpointCompactionContextPolicy(
        compactor=PromptCacheCompactor(provider=provider),
        max_user_turns=1,
        compact_after_messages=2,
    )
    messages = [
        Message.text("user", "old request"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current request"),
    ]

    async def unexpected_builder(context_messages: list[Message]) -> ModelRequest:
        raise AssertionError("tool structured output must take the bounded path directly")

    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=session_fixture(
                    id="checkpoint-tool-structured-output",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=messages,
                step=1,
                context_usage=ContextUsageState(
                    last_transcript_cursor=2,
                    last_provider_name="recording",
                    last_requested_model="claude-sonnet-4-6",
                ),
                pressure_overhead=ContextPressureOverhead(
                    structured_output_instruction="Call the reserved output tool."
                ),
                build_cache_prefix_request=unexpected_builder,
            ),
            checkpoint=None,
        )
    )

    assert result.checkpoint is not None
    assert result.checkpoint["context_compaction"]["metadata"]["compactor"] == "ModelCompactor"
    assert len(provider.requests) == 1
    assert provider.requests[0].tools == []


def test_checkpoint_policy_keeps_the_initial_prefix_until_it_can_compact() -> None:
    policy = CheckpointCompactionContextPolicy(
        max_user_turns=1,
        compact_after_messages=4,
    )
    messages = [
        Message.text("system", "You are careful."),
        Message.text("user", "old request"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current request"),
    ]

    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=session_fixture(
                    id="checkpoint-warm-prefix",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=messages,
                step=1,
            ),
            checkpoint=None,
        )
    )

    assert result.checkpoint is None
    assert result.messages == messages


def test_checkpoint_policy_falls_back_without_a_completed_request_cursor() -> None:
    provider = RecordingProvider([])
    policy = CheckpointCompactionContextPolicy(
        compactor=PromptCacheCompactor(provider=provider),
        max_user_turns=1,
        compact_after_messages=2,
    )
    messages = [
        Message.text("user", "old request"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current request"),
    ]

    async def unexpected_builder(context_messages: list[Message]) -> ModelRequest:
        raise AssertionError("no completed request exists to rebuild")

    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=session_fixture(
                    id="checkpoint-cache-unavailable",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=messages,
                step=1,
                build_cache_prefix_request=unexpected_builder,
            ),
            checkpoint=None,
        )
    )

    assert result.checkpoint is not None
    assert result.checkpoint["context_compaction"]["metadata"]["compactor"] == (
        "TranscriptDigestCompactor"
    )
    assert provider.requests == []


def test_checkpoint_policy_skips_cache_request_for_cross_model_override() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("cross-model summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    policy = CheckpointCompactionContextPolicy(
        compactor=PromptCacheCompactor(provider=provider, model="different-model"),
        max_user_turns=1,
        compact_after_messages=2,
    )
    messages = [
        Message.text("user", "old request"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current request"),
    ]

    async def unexpected_builder(context_messages: list[Message]) -> ModelRequest:
        raise AssertionError("cross-model compaction cannot reuse the request cache")

    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=session_fixture(
                    id="checkpoint-cross-model",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=messages,
                step=1,
                context_usage=ContextUsageState(last_transcript_cursor=2),
                build_cache_prefix_request=unexpected_builder,
            ),
            checkpoint=None,
        )
    )

    assert result.checkpoint is not None
    assert result.checkpoint["context_compaction"]["metadata"]["compactor"] == "ModelCompactor"
    assert provider.requests[0].model == "different-model"


def test_checkpoint_policy_does_not_build_a_model_request_for_digest_compaction() -> None:
    policy = CheckpointCompactionContextPolicy(
        max_user_turns=1,
        compact_after_messages=2,
    )
    messages = [
        Message.text("user", "old request"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current request"),
    ]

    async def unexpected_builder(context_messages: list[Message]) -> ModelRequest:
        raise AssertionError("digest compaction must not build or resolve a provider request")

    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=session_fixture(
                    id="checkpoint-digest-lazy",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=messages,
                step=1,
                build_cache_prefix_request=unexpected_builder,
            ),
            checkpoint=None,
        )
    )

    assert result.checkpoint is not None
    assert result.checkpoint["context_compaction"]["metadata"]["compactor"] == (
        "TranscriptDigestCompactor"
    )


def test_cayu_app_uses_cache_prefix_then_bounded_delta_and_accounts_for_both() -> None:
    provider = SequencedProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
            [
                ModelStreamEvent.text_delta("compacted summary"),
                ModelStreamEvent.completed(
                    {
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 10,
                            "cache_read_input_tokens": 80,
                        }
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 11, "output_tokens": 2}}),
            ],
            [
                ModelStreamEvent.text_delta("updated compacted summary"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 50, "output_tokens": 5}}),
            ],
            [
                ModelStreamEvent.text_delta("third answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 12, "output_tokens": 3}}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="claude-sonnet-4-6", system_prompt="Be careful."),
        tools=[InspectReportTool()],
        context_policy=CheckpointCompactionContextPolicy(
            compactor=PromptCacheCompactor(provider=provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )
    thinking = ThinkingConfig(effort="high")

    asyncio.run(
        collect_events(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="app-cache-prefix",
                    messages=[Message.text("user", "first request")],
                    thinking=thinking,
                )
            )
        )
    )
    first_resume_events = asyncio.run(
        collect_events(
            app.resume(
                ResumeRequest(
                    session_id="app-cache-prefix",
                    messages=[Message.text("user", "second request")],
                    thinking=thinking,
                )
            )
        )
    )
    second_resume_events = asyncio.run(
        collect_events(
            app.resume(
                ResumeRequest(
                    session_id="app-cache-prefix",
                    messages=[Message.text("user", "third request")],
                    thinking=thinking,
                )
            )
        )
    )

    initial_request, cached_compaction, second_request, delta_compaction, final_request = (
        provider.requests
    )
    assert cached_compaction.messages[: len(initial_request.messages)] == initial_request.messages
    assert cached_compaction.tools == initial_request.tools
    assert cached_compaction.options["thinking"] == initial_request.options["thinking"]
    assert (
        second_request.messages[1]
        .content[0]
        .text.startswith("Previous session context summary:\ncompacted summary")
    )
    assert delta_compaction.tools == []
    assert [message.role for message in delta_compaction.messages] == ["system", "user"]
    delta_prompt = delta_compaction.messages[1].content[0].text
    assert "Existing summary:\ncompacted summary" in delta_prompt
    assert "user: second request" in delta_prompt
    assert "assistant: second answer" in delta_prompt
    assert "user: first request" not in delta_prompt
    assert (
        final_request.messages[1]
        .content[0]
        .text.startswith("Previous session context summary:\nupdated compacted summary")
    )

    compaction_events = [
        event
        for event in first_resume_events + second_resume_events
        if event.type == EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    ]
    assert [event.payload["compactor"] for event in compaction_events] == [
        "PromptCacheCompactor",
        "ModelCompactor",
    ]
    usage = asyncio.run(app.get_session_usage("app-cache-prefix"))
    assert usage.model_steps == 5
    assert usage.usage.input_tokens == 263
    assert usage.usage.output_tokens == 22
    assert usage.usage.cache.read_tokens == 80
    assert usage.usage.cache.uncached_input_tokens == 183


def test_cayu_app_releases_compaction_budget_when_stage_receipt_is_rejected() -> None:
    assistant_model = "assistant-model"
    compactor_model = "compactor-model"
    store = RejectContextCompactionReceiptStore()
    budget_ledger = InMemoryBudgetLedger(reservation_ttl_seconds=None)
    main_provider = RestartStableRecordingProvider(
        [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}})]
    )
    compactor_provider = RestartStableToolCallDeadlineProvider()
    app = CayuApp(
        session_store=store,
        budget_policy=_restart_compaction_budget_policy(
            assistant_model=assistant_model,
            compactor_model=compactor_model,
        ),
        budget_ledger=budget_ledger,
        enable_logging=False,
    )
    app.register_provider(main_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model=assistant_model),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=RestartStablePromptCacheCompactor(
                provider=compactor_provider,
                model=compactor_model,
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )
    session_id = "app-compaction-receipt-rejected"

    events = asyncio.run(
        collect_events(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[
                        Message.text("user", "old request"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current request"),
                    ],
                )
            )
        )
    )

    assert store.rejected_receipts == 1
    assert compactor_provider.requests == []
    assert main_provider.requests == []
    assert asyncio.run(store.load_active_model_completion_stage(session_id)) is None
    records = tuple(budget_ledger._records.values())
    assert len(records) == 1
    [record] = records
    assert record.provider_name == compactor_provider.billing_provider_name
    assert record.model == compactor_model
    assert record.dispatch_id == record.model_attempt_id
    assert record.status == "released"
    assert EventType.BUDGET_RESERVATION_RELEASED in {event.type for event in events}
    assert EventType.BUDGET_RECONCILED not in {event.type for event in events}
    assert events[-1].type is EventType.SESSION_FAILED
    assert events[-1].payload["error"] == ("context-compaction receipt rejected before commit")


def test_cayu_app_retains_receiptless_compaction_stage_when_budget_release_fails() -> None:
    assistant_model = "assistant-model"
    compactor_model = "compactor-model"
    store = RejectContextCompactionReceiptStore()
    budget_ledger = RejectPreProviderReleaseLedger()
    main_provider = RestartStableRecordingProvider(
        [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}})]
    )
    compactor_provider = RestartStableToolCallDeadlineProvider()
    app = CayuApp(
        session_store=store,
        budget_policy=_restart_compaction_budget_policy(
            assistant_model=assistant_model,
            compactor_model=compactor_model,
        ),
        budget_ledger=budget_ledger,
        enable_logging=False,
    )
    app.register_provider(main_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model=assistant_model),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=RestartStablePromptCacheCompactor(
                provider=compactor_provider,
                model=compactor_model,
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )
    session_id = "app-compaction-release-rejected"

    with pytest.raises(RuntimeError, match="pre-provider compaction release rejected"):
        asyncio.run(
            collect_events(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[
                            Message.text("user", "old request"),
                            Message.text("assistant", "old answer"),
                            Message.text("user", "current request"),
                        ],
                    )
                )
            )
        )

    assert store.rejected_receipts == 1
    assert budget_ledger.release_attempts >= 2
    assert compactor_provider.requests == []
    assert main_provider.requests == []
    active = asyncio.run(store.load_active_model_completion_stage(session_id))
    assert active is not None
    assert active.stage.purpose == "context-compaction"
    assert (
        asyncio.run(
            store.load_model_completion_stage_dispatch(
                session_id,
                active.stage.stage_id,
            )
        )
        is None
    )
    records = tuple(budget_ledger._records.values())
    assert len(records) == 1
    [record] = records
    assert record.dispatch_id == active.stage.intent["model_attempt_id"]
    assert record.status == "active"

    budget_ledger.reject_releases = False
    asyncio.run(
        app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_for_seconds=0,
            )
        )
    )
    assert asyncio.run(store.load_active_model_completion_stage(session_id)) is None
    released = asyncio.run(budget_ledger.load_reservation(record.reservation_id))
    assert released is not None
    assert released.status == "released"


def test_receiptless_compaction_process_loss_releases_exact_budget_without_redispatch(
    tmp_path,
) -> None:
    database = tmp_path / "compaction-receipt-process-loss.sqlite"
    budget_database = tmp_path / "compaction-receipt-process-loss-budget.sqlite"
    provider_marker = tmp_path / "provider-entered"
    assistant_model = "assistant-model"
    compactor_model = "compactor-model"
    budget_policy = _restart_compaction_budget_policy(
        assistant_model=assistant_model,
        compactor_model=compactor_model,
    )
    store = SQLiteSessionStore(database)
    budget_ledger = SQLiteBudgetLedger(budget_database)
    main_provider = RestartStableRecordingProvider(
        [
            ModelStreamEvent.text_delta("first answer"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
        ]
    )
    app = CayuApp(
        session_store=store,
        budget_policy=budget_policy,
        budget_ledger=budget_ledger,
        enable_logging=False,
    )
    app.register_provider(main_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model=assistant_model),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=RestartStablePromptCacheCompactor(
                provider=RestartStableToolCallDeadlineProvider(),
                model=compactor_model,
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )
    session_id = "app-compaction-receipt-process-loss"
    asyncio.run(
        collect_events(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first request")],
                )
            )
        )
    )
    asyncio.run(store.close())
    asyncio.run(budget_ledger.close())

    repository_root = Path(__file__).resolve().parents[2]
    child_script = (
        "from tests.core.test_prompt_cache_compactor import "
        "_resume_until_context_compaction_receipt_exit as run; "
        f"run({str(database)!r}, {str(budget_database)!r}, {str(provider_marker)!r})"
    )
    child_environment = os.environ.copy()
    existing_python_path = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(repository_root / "src"), existing_python_path) if path
    )
    child = subprocess.run(
        [sys.executable, "-c", child_script],
        cwd=repository_root,
        env=child_environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert child.returncode == _CONTEXT_COMPACTION_RECEIPT_EXIT_CODE, child.stderr
    assert not provider_marker.exists()

    reopened_store = SQLiteSessionStore(database)
    reopened_budget_ledger = SQLiteBudgetLedger(budget_database)
    active = asyncio.run(reopened_store.load_active_model_completion_stage(session_id))
    assert active is not None
    assert active.stage.purpose == "context-compaction"
    assert (
        asyncio.run(
            reopened_store.load_model_completion_stage_dispatch(
                session_id,
                active.stage.stage_id,
            )
        )
        is None
    )
    assert len(active.stage.reservation_ids) == 1
    reservation_id = active.stage.reservation_ids[0]
    reservation = asyncio.run(reopened_budget_ledger.load_reservation(reservation_id))
    assert reservation is not None
    assert reservation.status == "active"
    assert reservation.dispatch_id == active.stage.intent["model_attempt_id"]

    restarted_main = RestartStableRecordingProvider(
        [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}})]
    )
    restarted_compactor = RestartStableToolCallDeadlineProvider()
    restarted = CayuApp(
        session_store=reopened_store,
        budget_policy=budget_policy,
        budget_ledger=reopened_budget_ledger,
        enable_logging=False,
    )
    restarted.register_provider(restarted_main, default=True)
    restarted.register_agent(
        AgentSpec(name="assistant", model=assistant_model),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=RestartStablePromptCacheCompactor(
                provider=restarted_compactor,
                model=compactor_model,
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )
    asyncio.run(
        restarted.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_for_seconds=0,
            )
        )
    )

    assert restarted_main.requests == []
    assert restarted_compactor.requests == []
    assert not provider_marker.exists()
    assert asyncio.run(reopened_store.load_active_model_completion_stage(session_id)) is None
    released = asyncio.run(reopened_budget_ledger.load_reservation(reservation_id))
    assert released is not None
    assert released.status == "released"
    assert released.dispatch_id == active.stage.intent["model_attempt_id"]
    asyncio.run(reopened_store.close())
    asyncio.run(reopened_budget_ledger.close())


def test_overflow_compaction_settles_borrowed_stage_budget_during_live_success() -> None:
    assistant_model = "assistant-model"
    compactor_model = "compactor-model"
    session_id = "overflow-compaction-live-budget"
    store = InMemorySessionStore()
    budget_ledger = InMemoryBudgetLedger(reservation_ttl_seconds=None)
    main_provider = OverflowThenSuccessProvider()
    compactor_provider = SuccessfulOverflowCompactionProvider()
    app = CayuApp(
        session_store=store,
        budget_policy=_restart_compaction_budget_policy(
            assistant_model=assistant_model,
            compactor_model=compactor_model,
        ),
        budget_ledger=budget_ledger,
        enable_logging=False,
    )
    app.register_provider(main_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model=assistant_model),
        context_overflow_policy=CheckpointCompactionContextPolicy(
            compactor=RestartStablePromptCacheCompactor(
                provider=compactor_provider,
                model=compactor_model,
            ),
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )

    events = asyncio.run(
        collect_events(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[
                        Message.text("user", "old request"),
                        Message.text("user", "current request"),
                    ],
                )
            )
        )
    )

    session = asyncio.run(store.load(session_id))
    assert session is not None and session.status is SessionStatus.COMPLETED
    assert len(main_provider.requests) == 2
    assert len(compactor_provider.requests) == 1
    companion_records = [
        record
        for record in store._session_operation_records[session_id].values()
        if record.get("record_type") == "cayu.borrowed-automatic-compaction-budget"
    ]
    assert len(companion_records) == 1
    [companion] = companion_records
    assert companion["status"] == "settled"
    assert len(companion["reservation_ids"]) == 1
    compactor_reservation = asyncio.run(
        budget_ledger.load_reservation(companion["reservation_ids"][0])
    )
    assert compactor_reservation is not None
    assert compactor_reservation.status == "reconciled"
    assert compactor_reservation.provider_name == compactor_provider.billing_provider_name
    assert compactor_reservation.model == compactor_model
    assert EventType.BUDGET_RECONCILED in {event.type for event in events}
    assert asyncio.run(store.load_active_model_completion_stage(session_id)) is None


def test_overflow_compaction_process_loss_recovers_borrowed_stage_budget_without_redispatch(
    tmp_path,
) -> None:
    database = tmp_path / "overflow-compaction-dispatch-process-loss.sqlite"
    budget_database = tmp_path / "overflow-compaction-dispatch-process-loss-budget.sqlite"
    provider_marker = tmp_path / "overflow-compactor-entered"
    repository_root = Path(__file__).resolve().parents[2]
    child_script = (
        "from tests.core.test_prompt_cache_compactor import "
        "_run_until_overflow_compaction_dispatch_exit as run; "
        f"run({str(database)!r}, {str(budget_database)!r}, {str(provider_marker)!r})"
    )
    child_environment = os.environ.copy()
    existing_python_path = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(repository_root / "src"), existing_python_path) if path
    )
    child = subprocess.run(
        [sys.executable, "-c", child_script],
        cwd=repository_root,
        env=child_environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert child.returncode == _OVERFLOW_COMPACTION_DISPATCH_EXIT_CODE, child.stderr
    assert provider_marker.read_text(encoding="utf-8") == "provider entered"

    session_id = "overflow-compaction-dispatch-process-loss"
    assistant_model = "assistant-model"
    compactor_model = "compactor-model"
    store = SQLiteSessionStore(database)
    budget_ledger = SQLiteBudgetLedger(budget_database)
    active = asyncio.run(store.load_active_model_completion_stage(session_id))
    assert active is not None
    assert active.stage.purpose == "assistant-turn"
    assert (
        asyncio.run(
            store.load_model_completion_stage_dispatch(
                session_id,
                active.stage.stage_id,
            )
        )
        is not None
    )
    durable_events = asyncio.run(store.load_events(session_id))
    compactor_reservations = [
        event
        for event in durable_events
        if event.type is EventType.BUDGET_RESERVED
        and event.payload.get("provider_name") == "tool-call-deadline-billing"
        and event.payload.get("model") == compactor_model
    ]
    assert len(compactor_reservations) == 1
    compactor_reservation_id = compactor_reservations[0].payload["reservation_id"]
    assert compactor_reservation_id not in active.stage.reservation_ids
    compactor_reservation = asyncio.run(budget_ledger.load_reservation(compactor_reservation_id))
    assert compactor_reservation is not None
    assert compactor_reservation.status == "active"
    assert (
        compactor_reservation.dispatch_id == compactor_reservations[0].payload["model_attempt_id"]
    )
    companion_key = (
        "cayu:borrowed-automatic-compaction-budget:v1:"
        + sha256(active.stage.stage_id.encode("utf-8")).hexdigest()
    )
    companion = asyncio.run(store.load_session_operation(session_id, companion_key))
    assert companion is not None
    assert companion["status"] == "pending"
    assert companion["reservation_ids"] == [compactor_reservation_id]

    restarted_main = RestartStableRecordingProvider(
        [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}})]
    )
    restarted_compactor = RestartStableToolCallDeadlineProvider()
    restarted = CayuApp(
        session_store=store,
        budget_policy=_restart_compaction_budget_policy(
            assistant_model=assistant_model,
            compactor_model=compactor_model,
        ),
        budget_ledger=budget_ledger,
        enable_logging=False,
    )
    restarted.register_provider(restarted_main, default=True)
    restarted.register_agent(
        AgentSpec(name="assistant", model=assistant_model),
        context_overflow_policy=CheckpointCompactionContextPolicy(
            compactor=RestartStablePromptCacheCompactor(
                provider=restarted_compactor,
                model=compactor_model,
            ),
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )
    with pytest.raises(ModelCompletionManualRecoveryRequired):
        asyncio.run(
            restarted.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_for_seconds=0,
                )
            )
        )

    assert restarted_main.requests == []
    assert restarted_compactor.requests == []
    recovered_compactor_reservation = asyncio.run(
        budget_ledger.load_reservation(compactor_reservation_id)
    )
    assert recovered_compactor_reservation is not None
    assert recovered_compactor_reservation.status == "reconciled"
    assert recovered_compactor_reservation.actual_amount == (
        recovered_compactor_reservation.reserved_amount
    )
    recovered_companion = asyncio.run(store.load_session_operation(session_id, companion_key))
    assert recovered_companion is not None
    assert recovered_companion["status"] == "pending"
    recovered_session = asyncio.run(store.load(session_id))
    assert recovered_session is not None
    settlement = asyncio.run(
        restarted.recover_model_completion_stage(
            ModelCompletionManualRecoveryRequest(
                session_id=session_id,
                stage_id=active.stage.stage_id,
                expected_run_epoch=recovered_session.run_epoch,
                terminal_status=SessionStatus.FAILED,
            )
        )
    )
    assert settlement.settlement.reason_code == "operator_outcome_unknown"
    assert asyncio.run(store.load_active_model_completion_stage(session_id)) is None
    settled_companion = asyncio.run(store.load_session_operation(session_id, companion_key))
    assert settled_companion is not None
    assert settled_companion["status"] == "settled"
    replayed = asyncio.run(
        restarted.recover_model_completion_stage(
            ModelCompletionManualRecoveryRequest(
                session_id=session_id,
                stage_id=active.stage.stage_id,
                expected_run_epoch=recovered_session.run_epoch,
                terminal_status=SessionStatus.FAILED,
            )
        )
    )
    assert replayed.replayed is True
    assert replayed.settlement == settlement.settlement
    assert restarted_main.requests == []
    assert restarted_compactor.requests == []
    asyncio.run(store.close())
    asyncio.run(budget_ledger.close())


def test_cayu_app_compaction_deadline_is_durable_and_not_replayed_after_restart(
    tmp_path,
) -> None:
    database = tmp_path / "compaction-deadline.sqlite"
    budget_database = tmp_path / "compaction-deadline-budget.sqlite"
    assistant_model = "assistant-model"
    compactor_model = "compactor-model"
    budget_policy = BudgetPolicy(
        limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("1"),
                pricing=PriceBook(
                    prices=(
                        ModelPrice.fixed(
                            provider_name="recording",
                            model=assistant_model,
                            input_per_million=Decimal("1"),
                            output_per_million=Decimal("1"),
                        ),
                        ModelPrice.fixed(
                            provider_name="tool-call-deadline-billing",
                            model=compactor_model,
                            input_per_million=Decimal("1"),
                            output_per_million=Decimal("1"),
                        ),
                    )
                ),
                reservation=BudgetReservation(
                    max_input_tokens=1_000,
                    max_output_tokens=1_000,
                ),
            ),
        )
    )
    budget_ledger = SQLiteBudgetLedger(budget_database)
    store = SQLiteSessionStore(database)
    main_provider = RestartStableRecordingProvider(
        [
            ModelStreamEvent.text_delta("first answer"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
        ]
    )
    compactor_provider = RestartStableToolCallDeadlineProvider()
    policy = CheckpointCompactionContextPolicy(
        compactor=RestartStablePromptCacheCompactor(
            provider=compactor_provider,
            model=compactor_model,
        ),
        max_user_turns=1,
        compact_after_messages=2,
    )
    app = CayuApp(
        session_store=store,
        budget_policy=budget_policy,
        budget_ledger=budget_ledger,
        enable_logging=False,
    )
    app.register_provider(main_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model=assistant_model),
        context_policy=policy,
    )
    session_id = "app-compaction-deadline"

    asyncio.run(
        collect_events(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first request")],
                )
            )
        )
    )

    async def resume_until_deadline() -> tuple[list, ModelStreamDeadlineError]:
        observed = []
        with pytest.raises(ModelStreamDeadlineError) as captured:
            async for event in app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "second request")],
                )
            ):
                observed.append(event)
        return observed, captured.value

    events, failure = asyncio.run(resume_until_deadline())
    assert len(main_provider.requests) == 1
    assert len(compactor_provider.requests) == 1
    assert failure.deadline_evidence.deadline_kind is ProviderDeadlineKind.SEMANTIC_IDLE
    compaction_started = [
        event
        for event in events
        if event.type is EventType.MODEL_STARTED
        and event.payload.get("purpose") == "context_compaction"
    ]
    compaction_completed = [
        event
        for event in events
        if event.type is EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    ]
    deadline_events = [
        event
        for event in events
        if event.type is EventType.MODEL_ERROR
        and event.payload.get("stage") == "context_compaction_stream"
    ]
    assert len(compaction_started) == len(compaction_completed) == len(deadline_events) == 1
    [model_started] = compaction_started
    [deadline_event] = deadline_events
    assert deadline_event.payload["provider"] == compactor_provider.name
    assert deadline_event.payload["provider_error_type"] == "ModelStreamDeadlineError"
    assert deadline_event.payload["provider_deadline_kind"] == "semantic_idle"
    assert deadline_event.payload["provider_last_progress_kind"] == "tool_call"
    assert deadline_event.payload["provider_effect_outcome"] == "unknown"
    assert deadline_event.payload["provider_recovery_disposition"] == ("manual_settlement_required")
    assert deadline_event.payload["model_step_id"] == model_started.payload["model_step_id"]
    assert deadline_event.payload["model_attempt_id"] == model_started.payload["model_attempt_id"]
    assert EventType.MODEL_RETRY not in {event.type for event in events}

    active = asyncio.run(store.load_active_model_completion_stage(session_id))
    session = asyncio.run(store.load(session_id))
    durable_events = asyncio.run(store.load_events(session_id))
    durable_started = [
        event
        for event in durable_events
        if event.type is EventType.MODEL_STARTED
        and event.payload.get("purpose") == "context_compaction"
    ]
    assert len(durable_started) == 1
    assert active is not None
    assert active.stage.state == "in_flight"
    assert active.stage.purpose == "context-compaction"
    assert active.stage.intent["provider_name"] == compactor_provider.name
    assert active.stage.intent["pricing_provider_name"] == (
        compactor_provider.billing_provider_name
    )
    assert active.stage.intent["requested_model"] == compactor_model
    assert len(active.stage.reservation_ids) == 1
    reservation_id = active.stage.reservation_ids[0]
    reservation = asyncio.run(budget_ledger.load_reservation(reservation_id))
    assert reservation is not None
    assert reservation.dispatch_id == active.stage.intent["model_attempt_id"]
    assert reservation.model_step_id == active.stage.logical_step_id
    assert reservation.model_attempt_id == active.stage.intent["model_attempt_id"]
    assert reservation.provider_name == compactor_provider.billing_provider_name
    assert reservation.model == compactor_model
    assert reservation.status == "reconciled"
    assert active.stage.logical_step_id == durable_started[0].payload["model_step_id"]
    assert active.stage.intent["model_attempt_id"] == durable_started[0].payload["model_attempt_id"]
    assert deadline_event.payload["model_completion_stage_id"] == active.stage.stage_id
    assert session is not None and session.status is SessionStatus.RUNNING
    assert (
        asyncio.run(store.load_model_completion_stage_dispatch(session_id, active.stage.stage_id))
        is not None
    )
    durable_deadlines = [
        event
        for event in durable_events
        if event.type is EventType.MODEL_ERROR
        and event.payload.get("stage") == "context_compaction_stream"
    ]
    assert len(durable_deadlines) == 1
    assert (
        sum(
            event.type is EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
            for event in durable_events
        )
        == 1
    )
    assert durable_deadlines[0].payload["model_step_id"] == active.stage.logical_step_id
    assert (
        durable_deadlines[0].payload["model_attempt_id"] == active.stage.intent["model_attempt_id"]
    )
    assert durable_deadlines[0].payload["provider_deadline_kind"] == "semantic_idle"

    asyncio.run(store.close())
    asyncio.run(budget_ledger.close())
    reopened_store = SQLiteSessionStore(database)
    reopened_budget_ledger = SQLiteBudgetLedger(budget_database)
    restarted_main = RestartStableRecordingProvider(
        [
            ModelStreamEvent.text_delta("must not dispatch"),
            ModelStreamEvent.completed({}),
        ]
    )
    restarted_compactor = RestartStableToolCallDeadlineProvider()
    restarted = CayuApp(
        session_store=reopened_store,
        budget_policy=budget_policy,
        budget_ledger=reopened_budget_ledger,
        enable_logging=False,
    )
    restarted.register_provider(restarted_main, default=True)
    restarted.register_agent(
        AgentSpec(name="assistant", model=assistant_model),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=RestartStablePromptCacheCompactor(
                provider=restarted_compactor,
                model=compactor_model,
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )
    with pytest.raises(ModelCompletionManualRecoveryRequired):
        asyncio.run(
            restarted.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_for_seconds=0,
                )
            )
        )
    assert len(compactor_provider.requests) == 1
    assert restarted_compactor.requests == []
    assert restarted_main.requests == []
    recovered_session = asyncio.run(reopened_store.load(session_id))
    assert recovered_session is not None
    manual_recovery_request = ModelCompletionManualRecoveryRequest(
        session_id=session_id,
        stage_id=active.stage.stage_id,
        expected_run_epoch=recovered_session.run_epoch,
        terminal_status=SessionStatus.FAILED,
    )

    settlement = asyncio.run(restarted.recover_model_completion_stage(manual_recovery_request))
    assert settlement.settlement.reason_code == "operator_outcome_unknown"
    assert asyncio.run(reopened_store.load_active_model_completion_stage(session_id)) is None
    replayed = asyncio.run(restarted.recover_model_completion_stage(manual_recovery_request))
    assert replayed.replayed is True and replayed.settlement == settlement.settlement
    assert len(compactor_provider.requests) == 1
    assert restarted_compactor.requests == []
    asyncio.run(reopened_store.close())
    asyncio.run(reopened_budget_ledger.close())


def test_cayu_app_compaction_rejects_provider_exact_reattachment_claim() -> None:
    class ExactReattachmentClaimProvider(ModelProvider):
        name = "exact-reattachment-claim"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return ExecutionProfileBehaviorIdentity(
                name="test.exact-reattachment-claim-provider",
                behavior_version="1",
                implementation_version="1",
            )

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            raise ModelStreamDeadlineError(
                provider=self.name,
                evidence=ProviderStreamDeadlineEvidence(
                    deadline_kind=ProviderDeadlineKind.SEMANTIC_IDLE,
                    configured_timeout_s=0.01,
                    elapsed_s=0.02,
                    last_progress_kind=None,
                    last_progress_elapsed_s=None,
                    last_progress_at=None,
                ),
                recovery_disposition="reattach_exact_operation",
            )
            yield ModelStreamEvent.completed({})  # pragma: no cover

    async def run() -> None:
        session_id = "app-compaction-provider-exact-reattachment-claim"
        store = InMemorySessionStore()
        main_provider = RestartStableRecordingProvider([ModelStreamEvent.completed({})])
        compactor_provider = ExactReattachmentClaimProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(main_provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="assistant-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RestartStablePromptCacheCompactor(
                    provider=compactor_provider,
                    model="compactor-model",
                ),
                max_user_turns=1,
                compact_after_messages=2,
            ),
        )

        observed: list[Event] = []
        with pytest.raises(ModelStreamDeadlineError) as captured:
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                )
            ):
                observed.append(event)

        assert captured.value.recovery_disposition == "manual_settlement_required"
        assert len(compactor_provider.requests) == 1
        assert main_provider.requests == []
        assert EventType.MODEL_RETRY not in {event.type for event in observed}
        deadline_events = [
            event
            for event in await store.load_events(session_id)
            if event.type is EventType.MODEL_ERROR
            and event.payload.get("stage") == "context_compaction_stream"
        ]
        assert len(deadline_events) == 1
        assert deadline_events[0].payload["provider_recovery_disposition"] == (
            "manual_settlement_required"
        )
        active = await store.load_active_model_completion_stage(session_id)
        assert active is not None
        assert active.stage.state == "in_flight"
        assert active.stage.purpose == "context-compaction"
        with pytest.raises(ModelCompletionManualRecoveryRequired):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_for_seconds=0,
                )
            )
        assert len(compactor_provider.requests) == 1
        assert main_provider.requests == []

    asyncio.run(run())


def test_compaction_deadline_survives_context_failure_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        session_id = "app-compaction-deadline-context-persistence-failure"
        store = InMemorySessionStore()
        main_provider = RestartStableRecordingProvider(
            [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}})]
        )
        compactor_provider = RestartStableToolCallDeadlineProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(main_provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="assistant-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RestartStablePromptCacheCompactor(
                    provider=compactor_provider,
                    model="compactor-model",
                ),
                max_user_turns=1,
                compact_after_messages=2,
            ),
        )

        persistence_failure = RuntimeError("context failure persistence rejected")
        original_emit_many = app._event_writer.emit_many

        async def fail_context_failure_batch(
            emitted_session_id: str,
            events: list[Event],
        ) -> list[Event]:
            if any(event.type is EventType.CONTEXT_COMPACTION_FAILED for event in events):
                raise persistence_failure
            return await original_emit_many(emitted_session_id, events)

        monkeypatch.setattr(app._event_writer, "emit_many", fail_context_failure_batch)
        with pytest.raises(ExceptionGroup) as captured:
            await collect_events(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[
                            Message.text("user", "old"),
                            Message.text("assistant", "old answer"),
                            Message.text("user", "current"),
                        ],
                    )
                )
            )

        leaves = _exception_group_leaves(captured.value)
        deadlines = [leaf for leaf in leaves if isinstance(leaf, ModelStreamDeadlineError)]
        assert len(deadlines) == 1
        assert sum(leaf is persistence_failure for leaf in leaves) == 1
        assert len(leaves) == 2
        assert len(compactor_provider.requests) == 1
        assert main_provider.requests == []

        active = await store.load_active_model_completion_stage(session_id)
        session = await store.load(session_id)
        durable_events = await store.load_events(session_id)
        durable_deadlines = [
            event
            for event in durable_events
            if event.type is EventType.MODEL_ERROR
            and event.payload.get("stage") == "context_compaction_stream"
        ]
        assert len(durable_deadlines) == 1
        assert active is not None
        assert active.stage.state == "in_flight"
        assert active.stage.purpose == "context-compaction"
        assert durable_deadlines[0].payload["model_completion_stage_id"] == active.stage.stage_id
        assert session is not None and session.status is SessionStatus.RUNNING
        assert EventType.SESSION_FAILED not in {event.type for event in durable_events}

        with pytest.raises(ModelCompletionManualRecoveryRequired):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_for_seconds=0,
                )
            )
        assert len(compactor_provider.requests) == 1
        assert main_provider.requests == []

    asyncio.run(run())


def test_overflow_compaction_deadline_survives_failed_overflow_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        session_id = "overflow-compaction-deadline-diagnostic-failure"
        store = InMemorySessionStore()
        overflow = ModelContextOverflowError(
            "context too large",
            provider="recording",
            status_code=400,
            error_code="context_length_exceeded",
        )
        main_provider = RestartStableRecordingProvider(
            [ModelStreamEvent.error(str(overflow), cause=overflow)]
        )
        compactor_provider = RestartStableToolCallDeadlineProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(main_provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="assistant-model"),
            context_overflow_policy=CheckpointCompactionContextPolicy(
                compactor=RestartStablePromptCacheCompactor(
                    provider=compactor_provider,
                    model="compactor-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )

        publication_failure = RuntimeError("context overflow diagnostic rejected")
        original_emit = app._event_writer.emit

        async def fail_overflow_diagnostic(event: Event) -> Event:
            if event.type is EventType.CONTEXT_OVERFLOW_FAILED:
                raise publication_failure
            return await original_emit(event)

        monkeypatch.setattr(app._event_writer, "emit", fail_overflow_diagnostic)
        with pytest.raises(ExceptionGroup) as captured:
            await collect_events(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[
                            Message.text("user", "old"),
                            Message.text("user", "current"),
                        ],
                    )
                )
            )

        leaves = _exception_group_leaves(captured.value)
        deadlines = [leaf for leaf in leaves if isinstance(leaf, ModelStreamDeadlineError)]
        assert len(deadlines) == 1
        assert sum(leaf is publication_failure for leaf in leaves) == 1
        assert len(leaves) == 2
        assert len(main_provider.requests) == 1
        assert len(compactor_provider.requests) == 1

        active = await store.load_active_model_completion_stage(session_id)
        session = await store.load(session_id)
        durable_events = await store.load_events(session_id)
        durable_deadlines = [
            event
            for event in durable_events
            if event.type is EventType.MODEL_ERROR
            and event.payload.get("stage") == "context_compaction_stream"
        ]
        assert len(durable_deadlines) == 1
        assert active is not None
        assert active.stage.state == "in_flight"
        assert active.stage.purpose == "assistant-turn"
        assert durable_deadlines[0].payload["model_completion_stage_id"] == active.stage.stage_id
        assert session is not None and session.status is SessionStatus.RUNNING
        assert EventType.SESSION_FAILED not in {event.type for event in durable_events}
        assert EventType.CONTEXT_OVERFLOW_FAILED not in {event.type for event in durable_events}

        with pytest.raises(ModelCompletionManualRecoveryRequired):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_for_seconds=0,
                )
            )
        assert len(main_provider.requests) == 1
        assert len(compactor_provider.requests) == 1

    asyncio.run(run())


def test_cayu_app_resume_model_override_cannot_reuse_previous_model_cache() -> None:
    provider = SequencedProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
            [
                ModelStreamEvent.text_delta("bounded summary"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 8, "output_tokens": 2}}),
            ],
            [
                ModelStreamEvent.text_delta("new model answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 6, "output_tokens": 2}}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="old-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=PromptCacheCompactor(provider=provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    asyncio.run(
        collect_events(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="app-cache-model-override",
                    messages=[Message.text("user", "first request")],
                )
            )
        )
    )
    resume_events = asyncio.run(
        collect_events(
            app.resume(
                ResumeRequest(
                    session_id="app-cache-model-override",
                    target=ModelTarget(provider_name="sequenced", model="new-model"),
                    messages=[Message.text("user", "second request")],
                )
            )
        )
    )

    assert [request.model for request in provider.requests] == [
        "old-model",
        "new-model",
        "new-model",
    ]
    assert provider.requests[1].tools == []
    assert [message.role for message in provider.requests[1].messages] == ["system", "user"]
    assert any(
        event.type == EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
        for event in resume_events
    )
    compaction_completed = [
        event for event in resume_events if event.type == EventType.CONTEXT_COMPACTION_COMPLETED
    ]
    assert len(compaction_completed) == 1
    assert compaction_completed[0].payload["compactor"] == "PromptCacheCompactor"
    assert compaction_completed[0].payload["chunk_mode"] == "single_request"
    assert resume_events[-1].type == EventType.SESSION_COMPLETED


def test_cayu_app_preserves_resolved_attachment_bytes_in_cached_prefix(tmp_path) -> None:
    provider = SequencedProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {"input_tokens": 10, "output_tokens": 2},
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {"input_tokens": 20, "output_tokens": 2},
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("compacted summary"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("third answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            artifact_store=LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts"),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="claude-sonnet-4-6"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=PromptCacheCompactor(provider=provider),
            max_user_turns=1,
            compact_after_messages=3,
        ),
    )

    async def run_three_turns() -> tuple[str, str]:
        from io import BytesIO

        from PIL import Image

        buffer = BytesIO()
        Image.new("RGB", (1, 1), "white").save(buffer, format="PNG")
        old_part = await app.attach_file(
            buffer.getvalue(),
            filename="old-report.png",
            kind="image",
            session_id="app-cache-attachment",
        )
        await collect_events(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="app-cache-attachment",
                    messages=[
                        Message(
                            role="user",
                            content=[TextPart(text="inspect the old report"), old_part],
                        )
                    ],
                )
            )
        )
        current_part = await app.attach_file(
            buffer.getvalue(),
            filename="current-report.png",
            kind="image",
            session_id="app-cache-attachment",
        )
        await collect_events(
            app.resume(
                ResumeRequest(
                    session_id="app-cache-attachment",
                    messages=[
                        Message(
                            role="user",
                            content=[TextPart(text="inspect the current report"), current_part],
                        )
                    ],
                )
            )
        )
        await collect_events(
            app.resume(
                ResumeRequest(
                    session_id="app-cache-attachment",
                    messages=[Message.text("user", "compare the findings")],
                )
            )
        )
        return (
            old_part.attachment["artifact_id"],
            current_part.attachment["artifact_id"],
        )

    old_artifact_id, current_artifact_id = asyncio.run(run_three_turns())

    initial_request, warm_request, compaction_request, final_request = provider.requests
    assert set(initial_request.options[RESOLVED_FILE_ATTACHMENTS_OPTION]) == {old_artifact_id}
    warm_resolved = warm_request.options[RESOLVED_FILE_ATTACHMENTS_OPTION]
    assert set(warm_resolved) == {current_artifact_id}
    assert compaction_request.messages[: len(warm_request.messages)] == warm_request.messages
    assert compaction_request.options[RESOLVED_FILE_ATTACHMENTS_OPTION] == warm_resolved
    assert old_artifact_id not in compaction_request.options[RESOLVED_FILE_ATTACHMENTS_OPTION]
    assert final_request.options[RESOLVED_FILE_ATTACHMENTS_OPTION] == {}


def test_prompt_cache_compactor_uses_bounded_incremental_compaction_after_checkpoint() -> None:
    compaction_instruction = (
        "Preserve the mandatory retention token across every compaction. Return only a summary."
    )
    provider = SequencedProvider(
        [
            [
                ModelStreamEvent.text_delta("first summary"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("updated summary"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    compactor = PromptCacheCompactor(
        provider=provider,
        compaction_instruction=compaction_instruction,
    )
    session = session_fixture(
        id="repeated-compaction",
        agent_name="assistant",
        provider_name="sequenced",
        model="claude-sonnet-4-6",
    )
    agent = AgentSpec(name="assistant", model="claude-sonnet-4-6")
    first_cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "first full cached context")],
        tools=[
            {
                "name": "inspect_report",
                "description": "Inspect a report.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    )

    first = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=session,
                agent=agent,
                messages=[Message.text("user", "first full cached context")],
                context_messages=first_cached_request.messages,
                cache_prefix_request=first_cached_request,
            )
        )
    )
    second = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=session,
                agent=agent,
                messages=[Message.text("user", "new work since the checkpoint")],
                existing_summary=first.summary,
                context_messages=[
                    Message.text("user", "first full cached context"),
                    Message.text("assistant", "first answer"),
                    Message.text("user", "new work since the checkpoint"),
                ],
                cache_prefix_request=ModelRequest(
                    model="claude-sonnet-4-6",
                    messages=[Message.text("user", "an ever-growing raw transcript")],
                    tools=first_cached_request.tools,
                ),
            )
        )
    )

    incremental_request = provider.requests[1]
    assert first.metadata["compactor"] == "PromptCacheCompactor"
    assert second.metadata["compactor"] == "ModelCompactor"
    assert incremental_request.tools == []
    assert [message.role for message in incremental_request.messages] == ["system", "user"]
    assert incremental_request.messages[0].content[0].text == compaction_instruction
    incremental_prompt = incremental_request.messages[1].content[0].text
    assert "Existing summary:\nfirst summary" in incremental_prompt
    assert "user: new work since the checkpoint" in incremental_prompt
    assert "an ever-growing raw transcript" not in incremental_prompt
