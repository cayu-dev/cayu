"""Application registration, admission and runtime identity contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from collections.abc import AsyncIterator
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import SecretStr, ValidationError
from tests._session_provenance import fixture_session_invocation
from tests.core._execution_profile_fixtures import (
    interrupt_and_release_test_invocation,
    profiled_session_identity,
    run_request_with_registered_tool_ceiling,
)
from tests.core._session_operation_fault_harness import (
    CommitEvidence,
    FailBeforeTransform,
    MatchPolicy,
    PublicationBoundary,
    PublicationFaultActionKind,
    PublicationFaultOutcome,
    SessionOperationFaultHarness,
    SessionOperationFaultRule,
    SessionOperationSelector,
)
from tests.provider_traceback_assertions import is_cayu_source_filename
from tests.runtime.runtime_compaction_support import (
    _AUTOMATIC_COMPACTION_PROCESS_EXIT_CODE,
    CompletedThenRetryableCompactionProvider,
    FlakyCompactionProvider,
    ProcessLossAutomaticCompactionStore,
    RecordingCompactor,
    _expected_compaction_failure,
    compaction_price_book,
)
from tests.runtime.runtime_environment_support import RecordingWorkspaceBinding
from tests.runtime.runtime_provider_support import (
    ContextOverflowProvider,
    CountingProvider,
    FakeProvider,
)
from tests.runtime.runtime_session_support import (
    _AllowForkProfileAdoption,
    _expected_failure_evidence,
    _private_events_for_public_events,
    _private_record_for_public_event,
    _test_session,
    _without_interaction_lifecycle,
    _without_request_evidence,
    assert_only_model_step_publication_checkpoint,
    checkpoint_without_active_invocation_profile,
    checkpoint_without_model_step_publication,
    collect_events,
    collect_resume_events,
)
from tests.runtime.runtime_session_support import (
    request_footprint_session_store as request_footprint_session_store,
)
from tests.runtime.runtime_tool_support import EchoTool, RequireApprovalPolicy, SideEffectTool

import cayu
import cayu.budgets.base as budgets_module
import cayu.context.base as runtime_context_module
import cayu.execution_units as execution_units_module
import cayu.runtime._run_limits as run_limits_module
import cayu.runtime._tool_invocation.admission as tool_admission_module
import cayu.runtime._tool_round_executor as tool_round_executor_module
from cayu._exception_groups import exception_cause, iter_exception_tree
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ResolutionActor, ResolutionActorSource
from cayu.artifacts.attachments import RESOLVED_FILE_ATTACHMENTS_OPTION, file_attachment
from cayu.budgets.base import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    BudgetWindow,
    InMemoryBudgetLedger,
    InMemoryBudgetStore,
)
from cayu.budgets.billing import BillingIdentity
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.budgets.run_limits import RunLimits
from cayu.configuration import CayuConfig, ToolExecutionConfig
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    CompactionPrompt,
    CompactionRequest,
    CompactionResult,
    ContextBuildError,
    ContextBuildResult,
    ContextCompactor,
    ContextPolicy,
    ContextPressureEstimate,
    ContextPressureOverhead,
    ContextRequest,
    ContextUsageState,
    DefaultContextPolicy,
    MessageWindowContextPolicy,
    ModelCompactor,
    ObservedDeltaContextEstimator,
    PromptCacheCompactor,
    RecentTurnsContextPolicy,
    RuntimeManagedContextPolicy,
    TranscriptDigestCompactor,
    UsageTriggeredContextPolicy,
)
from cayu.context.footprints import RequestFootprintConfig
from cayu.context.structured_output import STRUCTURED_OUTPUT_TOOL_NAME
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.events import Event, EventType
from cayu.messages import FilePart, Message, MessageRole, ProviderStatePart, TextPart
from cayu.observability.events import InMemoryEventSink
from cayu.observability.hooks import RuntimeHook
from cayu.providers.base import (
    InputTokenCountConfidence,
    InputTokenCountMethod,
    InputTokenCountResult,
    ModelContextOverflowError,
    ModelContextPressureProfile,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
)
from cayu.providers.bedrock import bedrock_billing_identity, completed_bedrock_billing_identity
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime._compaction import automatic as _automatic_compaction
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY, public_event_id
from cayu.runtime._model_completion_contracts import ModelCompletionManualRecoveryRequired
from cayu.runtime._model_errors import (
    _BillingIdentityResolutionCancelled,
    detach_billing_identity_cancellation_group,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.execution_profiles import ExecutionProfileAdoptionIntent
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.queries import SessionQuery
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.storage.budget_ledger import SQLiteBudgetLedger
from cayu.storage.memory import InMemoryKnowledgeStore, KnowledgeAccessScope, KnowledgeEntry
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.tools.user_input import UserInputTool
from cayu.vaults.redaction import SecretRedactor


class _TestKnowledgeStore(InMemoryKnowledgeStore):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("access_scope", KnowledgeAccessScope.privileged())
        super().__init__(*args, **kwargs)


class MessageCountOverflowProvider(ModelProvider):
    """Overflows full history but accepts a bounded two-message rebuild."""

    name = "message-count-overflow"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(request.messages) >= 3:
            raise ModelContextOverflowError(
                "context too large",
                provider=self.name,
                status_code=400,
                error_code="context_length_exceeded",
            )
        yield ModelStreamEvent.text_delta("bounded response")
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        )


class EventFlattenedOverflowProvider(ModelProvider):
    """Contract-violating provider: flattens overflow into an error event.

    `stream()` must raise `ModelContextOverflowError`, but a provider that
    wraps every failure with `ModelStreamEvent.error(..., cause=exc)` still
    carries the typed overflow identity in the payload. The runtime must route
    that into context-overflow recovery instead of generic retries.
    """

    name = "overflow-event"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ModelStreamEvent.error(
                "context too large",
                cause=ModelContextOverflowError(
                    "context too large",
                    provider=self.name,
                    status_code=400,
                    error_code="context_length_exceeded",
                ),
            )
            return
        yield ModelStreamEvent.text_delta("recovered")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class FailingCompactor(ContextCompactor):
    def provider_budget_identity(self, _session: Session) -> None:
        return None

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        raise RuntimeError("compaction unavailable")


@pytest.mark.parametrize(
    "secret_values",
    [
        ("_",),
        ("0",),
    ],
)
def test_complete_run_preserves_all_runtime_generated_identity_authority(
    secret_values: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CollisionProvider(FakeProvider):
        name = "xyz"

    monkeypatch.setattr(execution_units_module, "uuid4", lambda: UUID(int=0))
    provider = CollisionProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(
        secret_redactor=SecretRedactor(secret_values),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="xyz", model="xyz"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="xyz",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert len(provider.requests) == 1, [(event.type, event.payload) for event in events]
    assert events[-1].type == EventType.SESSION_COMPLETED, [
        (event.type, event.payload) for event in events
    ]
    assert EventType.MODEL_STARTED in {event.type for event in events}
    assert EventType.MODEL_COMPLETED in {event.type for event in events}


def test_budget_flow_preserves_runtime_generated_identity_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CollisionProvider(FakeProvider):
        name = "xyz"

    monkeypatch.setattr(execution_units_module, "uuid4", lambda: UUID(int=0))
    monkeypatch.setattr(budgets_module, "uuid4", lambda: UUID(int=0))
    limit = BudgetLimit(
        scope="app",
        max_estimated_cost=Decimal("1"),
        pricing=PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="xyz",
                    model="xyz",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        ),
        reservation=BudgetReservation(
            max_input_tokens=10,
            max_output_tokens=10,
        ),
    )
    policy = BudgetPolicy(limits=(limit,))
    effective_limit = budgets_module.budget_limits_for_session(
        policy=policy,
        agent_name="xyz",
        causal_budget_id="causal",
    )[0]
    provider = CollisionProvider(
        [
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            )
        ]
    )
    app = CayuApp(
        budget_policy=policy,
        budget_ledger=InMemoryBudgetLedger(),
        secret_redactor=SecretRedactor(
            (
                "0000000000000000",
                effective_limit.budget_limit_id[-16:],
            )
        ),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="xyz", model="xyz"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="xyz",
                session_id="session",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    event_types = {event.type for event in events}
    assert len(provider.requests) == 1, [(event.type, event.payload) for event in events]
    assert events[-1].type == EventType.SESSION_COMPLETED, [
        (events[-1].type, events[-1].payload),
        [event.type for event in events],
    ]
    assert EventType.BUDGET_CHECKED in event_types
    assert EventType.BUDGET_RESERVED in event_types
    assert EventType.BUDGET_RECONCILED in event_types


def test_approval_flow_preserves_runtime_generated_approval_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approval_uuid = UUID(int=0)
    monkeypatch.setattr(tool_admission_module, "uuid4", lambda: approval_uuid)
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call",
                name="side_effect",
                arguments={},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp(
        secret_redactor=SecretRedactor("0000000000000000"),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool()],
        tool_policy=RequireApprovalPolicy(),
    )

    async def run() -> list[Event]:
        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="session",
                messages=[Message.text("user", "run tool")],
            ),
        )
        approval = pending_approval_reader.pending_approval_from_checkpoint(
            await app.session_store.load_checkpoint("session")
        )
        assert approval is not None
        assert approval.approval_id == str(approval_uuid)
        return events

    events = asyncio.run(run())

    event_types = {event.type for event in events}
    assert len(provider.requests) == 1
    assert EventType.TOOL_CALL_APPROVAL_REQUESTED in event_types
    assert events[-1].type == EventType.SESSION_INTERRUPTED, [
        (events[-1].type, events[-1].payload),
        [event.type for event in events],
    ]


def test_user_input_flow_preserves_runtime_generated_input_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool_round_executor_module, "uuid4", lambda: UUID(int=0))
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call",
                name="ask_user",
                arguments={"question": "Continue?"},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp(
        secret_redactor=SecretRedactor("0000000000000000"),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UserInputTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="session",
                messages=[Message.text("user", "ask me")],
            ),
        )
    )

    event_types = {event.type for event in events}
    assert len(provider.requests) == 1
    assert EventType.SESSION_AWAITING_USER_INPUT in event_types
    assert events[-1].type == EventType.SESSION_INTERRUPTED, [
        (events[-1].type, events[-1].payload),
        [event.type for event in events],
    ]


def test_cayu_app_rejects_scope_that_differs_from_bound_store() -> None:
    store = InMemoryKnowledgeStore(access_scope=KnowledgeAccessScope.for_namespace("project-a"))
    with pytest.raises(ValueError, match="must match the scope bound"):
        CayuApp(
            knowledge_store=store,
            knowledge_access_scope=KnowledgeAccessScope.for_namespace("project-b"),
        )


def test_cayu_app_passes_environment_knowledge_store_to_tools() -> None:
    class KnowledgeStoreCheckTool(Tool):
        spec = ToolSpec(
            name="check_knowledge_store",
            description="Check whether the tool context has a knowledge store.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            entry = await ctx.knowledge_store.get_entry("entry_1")
            return ToolResult(
                content="knowledge store available",
                structured={
                    "has_knowledge_store": ctx.knowledge_store is not None,
                    "entry_text": entry.text if entry is not None else None,
                },
            )

    knowledge_store = _TestKnowledgeStore(
        [KnowledgeEntry(id="entry_1", text="Runtime knowledge is available.")]
    )
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_knowledge",
                    name="check_knowledge_store",
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
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            knowledge_store=knowledge_store,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[KnowledgeStoreCheckTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_knowledge_store_context",
                messages=[Message.text("user", "check knowledge")],
            ),
        )
    )

    tool_event = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    result_payload = tool_event.payload["result"]
    assert result_payload["structured"] == {
        "has_knowledge_store": True,
        "entry_text": "Runtime knowledge is available.",
    }


def test_registered_tool_uses_public_schema_override_as_authority():
    class SchemaOverrideTool(Tool):
        spec = ToolSpec(name="lookup", input_schema={"type": "object"})

        @property
        def schema(self) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": {"invoice_id": {"type": "string"}},
                "required": ["invoice_id"],
                "additionalProperties": False,
            }

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            return ToolResult(content=args["invoice_id"])

    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SchemaOverrideTool()],
    )

    assert app.describe().agents[0].tools[0].input_schema == {
        "type": "object",
        "properties": {"invoice_id": {"type": "string"}},
        "required": ("invoice_id",),
        "additionalProperties": False,
    }


def test_register_agent_rejects_synchronous_tool_run():
    class SyncTool(Tool):
        spec = ToolSpec(name="sync")

        def run(self, ctx: ToolContext, args: dict) -> ToolResult:  # type: ignore[override]
            return ToolResult(content="not awaitable")

    app = CayuApp(enable_logging=False)

    with pytest.raises(TypeError, match=r"SyncTool\.run must be declared with `async def`"):
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[SyncTool()],
        )


def test_cayu_app_rejects_invalid_runtime_dependencies():
    class StoreLike:
        pass

    class TaskStoreLike:
        pass

    class DispatcherLike:
        pass

    class HookLike:
        pass

    class BlankNameHook(RuntimeHook):
        @property
        def name(self) -> str:
            return " "

    class SinkLike:
        async def emit(self, event):
            pass

    with pytest.raises(TypeError, match="SessionStore"):
        CayuApp(session_store=StoreLike())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="TaskStore"):
        CayuApp(task_store=TaskStoreLike())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Dispatcher"):
        CayuApp(dispatcher=DispatcherLike())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="RuntimeHook"):
        CayuApp(runtime_hooks=[HookLike()])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="runtime_hook.name"):
        CayuApp(runtime_hooks=[BlankNameHook()])

    app = CayuApp()
    with pytest.raises(TypeError, match="RuntimeHook"):
        app.register_agent(
            AgentSpec(name="invalid_hook_agent", model="fake-model"),
            runtime_hooks=[HookLike()],  # type: ignore[list-item]
        )

    with pytest.raises(ValueError, match="runtime_hook.name"):
        app.register_agent(
            AgentSpec(name="blank_hook_agent", model="fake-model"),
            runtime_hooks=[BlankNameHook()],
        )

    class ReservedStructuredOutputTool(Tool):
        spec = ToolSpec(
            name=STRUCTURED_OUTPUT_TOOL_NAME,
            description="Reserved.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            return ToolResult(content="reserved")

    with pytest.raises(ValueError, match="reserved by the Cayu framework"):
        app.register_agent(
            AgentSpec(name="reserved_tool_agent", model="fake-model"),
            tools=[ReservedStructuredOutputTool()],
        )

    with pytest.raises(TypeError, match="EventSink"):
        CayuApp(event_sinks=[SinkLike()])  # type: ignore[list-item]

    with pytest.raises(TypeError, match="event_sinks"):
        CayuApp(event_sinks=False)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="event_sinks"):
        CayuApp(event_sinks=0)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="event_sinks"):
        CayuApp(event_sinks="")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="secret_redactor"):
        CayuApp(secret_redactor="not a redactor")  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="max_file_attachment_bytes"):
        CayuApp(
            config=CayuConfig(tool_execution=ToolExecutionConfig(max_file_attachment_bytes=1.5))
        )  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_file_attachment_bytes"):
        CayuApp(config=CayuConfig(tool_execution=ToolExecutionConfig(max_file_attachment_bytes=0)))

    with pytest.raises(ValueError, match="max_total_file_attachment_bytes"):
        CayuApp(
            config=CayuConfig(tool_execution=ToolExecutionConfig(max_total_file_attachment_bytes=0))
        )

    with pytest.raises(ValueError, match="max_file_attachments_per_request"):
        CayuApp(
            config=CayuConfig(
                tool_execution=ToolExecutionConfig(max_file_attachments_per_request=0)
            )
        )


def test_cayu_app_preserves_falsey_session_store_instance():
    class FalseySessionStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __bool__(self):
            return False

    store = FalseySessionStore()
    app = CayuApp(session_store=store)

    assert app.session_store is store


def test_cayu_app_run_rejects_invalid_request_type():
    app = CayuApp()

    async def run_invalid_request():
        return [event async for event in app.run({"agent_name": "assistant"})]  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="RunRequest"):
        asyncio.run(run_invalid_request())


def test_cayu_app_run_revalidates_constructed_run_request():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(
        FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    request = RunRequest.model_construct(
        agent_name="assistant",
        session_id="sess_invalid_request",
        messages=[Message.text("user", "hi")],
        metadata={},
        max_steps="2",
    )

    async def run_invalid_request():
        return [event async for event in app.run(request)]

    with pytest.raises(TypeError, match="max_steps must be an integer"):
        asyncio.run(run_invalid_request())

    assert asyncio.run(store.load("sess_invalid_request")) is None


def test_cayu_app_run_revalidates_constructed_request_messages():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(
        FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    request = RunRequest.model_construct(
        agent_name="assistant",
        session_id="sess_invalid_message",
        messages=[
            Message.model_construct(
                role="user",
                content=[TextPart.model_construct(text=" ")],
            )
        ],
        metadata={},
        max_steps=1,
    )

    async def run_invalid_request():
        return [event async for event in app.run(request)]

    with pytest.raises(ValidationError):
        asyncio.run(run_invalid_request())

    assert asyncio.run(store.load("sess_invalid_message")) is None


def test_cayu_app_runs_text_only_session_and_persists_events():
    store = InMemorySessionStore()
    sink = InMemoryEventSink()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store, event_sinks=[sink])
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_text",
                messages=[Message.text("user", "hi")],
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
    ]
    assert events[2].payload["delta"] == "hello"
    for key in (
        "step",
        "attempt",
        "max_attempts",
        "model_step_id",
        "model_attempt_id",
    ):
        assert events[2].payload[key] == events[1].payload[key]
        assert events[3].payload[key] == events[1].payload[key]
    assert provider.requests[0].model == "fake-model"
    assert provider.requests[0].messages[0].content[0].text == "hi"
    assert provider.requests[0].tools == []

    records = asyncio.run(store.query_events(EventQuery(session_id="sess_text")))
    projected = [
        app._project_persisted_event_record_for_exposure(record).event for record in records
    ]
    assert _without_interaction_lifecycle(sink.events) == _without_interaction_lifecycle(projected)
    session = asyncio.run(store.load("sess_text"))

    assert _without_request_evidence(_without_interaction_lifecycle(projected)) == events
    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    assert session.provider_name == "fake"
    assert session.model == "fake-model"
    assert session.runtime_name == "cayu"


@pytest.mark.parametrize("with_reservation", [False, True])
def test_automatic_compaction_uses_its_bedrock_billing_identity(
    with_reservation: bool,
) -> None:
    requested_identity = bedrock_billing_identity(
        invoked_model="global.anthropic.claude-sonnet-4-6",
        source_region="us-east-1",
        resource_type="inference_profile",
        profile_scope="global",
        requested_service_tier="reserved",
    )
    completed_identity = completed_bedrock_billing_identity(
        requested_identity,
        effective_service_tier="default",
    )

    class ContextualCompactionProvider(FakeProvider):
        name = "bedrock"

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity:
            assert request.model == requested_identity.resource_id
            return requested_identity

        def billing_identity_for_completion(
            self,
            identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity | None:
            assert identity == requested_identity
            assert payload["bedrock_service_tier"] == "default"
            return completed_identity

    compactor_provider = ContextualCompactionProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed(
                {
                    "model": requested_identity.resource_id,
                    "usage": {"input_tokens": 1_000_000, "output_tokens": 0},
                    "bedrock_service_tier": "default",
                }
            ),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": "fake-model",
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                }
            )
        ]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="fake",
                model="fake-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
            ModelPrice.fixed(
                provider_name="bedrock",
                model=requested_identity.resource_id,
                match="exact",
                input_per_million=Decimal("2"),
                output_per_million=Decimal("1"),
                pricing_context={
                    "source_region": ("us-east-1",),
                    "service_tier": ("reserved",),
                },
            ),
            ModelPrice.fixed(
                provider_name="bedrock",
                model=requested_identity.resource_id,
                match="exact",
                input_per_million=Decimal("3"),
                output_per_million=Decimal("1"),
                pricing_context={
                    "source_region": ("us-east-1",),
                    "service_tier": ("default",),
                },
            ),
        )
    )
    ledger = InMemoryBudgetLedger()
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("20"),
                    pricing=pricing,
                    reservation=(
                        BudgetReservation(
                            max_input_tokens=1_000_000,
                            max_output_tokens=0,
                        )
                        if with_reservation
                        else None
                    ),
                ),
            )
        ),
        budget_ledger=ledger,
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model=requested_identity.resource_id,
                options={"bedrock": {"serviceTier": {"type": "reserved"}}},
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
                session_id=f"sess_bedrock_compaction_identity_{with_reservation}",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    completed = next(
        event
        for event in events
        if event.type == EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    )
    assert completed.payload["billing_identity"] == completed_identity.model_dump(mode="json")
    assert "billing_identity" not in completed.payload["usage_metrics"]
    assert events[-1].type == EventType.SESSION_COMPLETED
    if with_reservation:
        reserved = next(
            event
            for event in events
            if event.type == EventType.BUDGET_RESERVED
            and event.payload["model"] == requested_identity.resource_id
        )
        reconciled = next(
            event
            for event in events
            if event.type == EventType.BUDGET_RECONCILED
            and event.payload["reservation_id"] == reserved.payload["reservation_id"]
        )
        assert Decimal(reserved.payload["requested"]) == Decimal("3")
        assert reserved.payload["billing_identity"] == requested_identity.model_dump(mode="json")
        assert Decimal(reconciled.payload["actual_amount"]) == Decimal("3")
        assert reconciled.payload["billing_identity"] == completed.payload["billing_identity"]
        assert (
            next(
                record
                for record in ledger._records.values()
                if record.model == requested_identity.resource_id
            ).billing_identity
            == completed_identity
        )
    else:
        assert EventType.BUDGET_RESERVED not in [event.type for event in events]
        assert EventType.BUDGET_RECONCILED not in [event.type for event in events]
        assert ledger._records == {}


@pytest.mark.parametrize("scope", ["app", "run"])
def test_automatic_contextual_budget_counts_prior_hierarchy_dispatches(scope) -> None:
    requested_identity = bedrock_billing_identity(
        invoked_model="global.anthropic.claude-sonnet-4-6",
        source_region="us-east-1",
        resource_type="inference_profile",
        profile_scope="global",
        requested_service_tier="default",
    )
    completed_identity = completed_bedrock_billing_identity(
        requested_identity,
        effective_service_tier="default",
    )

    class ContextualHierarchyProvider(FakeProvider):
        name = "bedrock-adapter"
        billing_provider_name = "bedrock"

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity:
            return requested_identity

        def billing_identity_for_completion(
            self,
            identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity | None:
            assert identity == requested_identity
            return completed_identity

    completed_batch = [
        ModelStreamEvent.text_delta("short summary"),
        ModelStreamEvent.completed(
            {
                "model": requested_identity.resource_id,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            }
        ),
    ]
    compactor_provider = ContextualHierarchyProvider([completed_batch for _index in range(8)])
    runtime_provider = FakeProvider(
        [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}})]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="fake",
                model="fake-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
            ModelPrice.fixed(
                provider_name="bedrock",
                model=requested_identity.resource_id,
                match="exact",
                input_per_million=Decimal("3"),
                output_per_million=Decimal("1"),
                pricing_context={
                    "source_region": ("us-east-1",),
                    "service_tier": ("default",),
                },
            ),
        )
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("0.000004"),
                    pricing=pricing,
                ),
            )
            if scope == "app"
            else ()
        )
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model=requested_identity.resource_id,
                max_input_chars=1000,
            ),
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_automatic_contextual_hierarchy_budget",
                budget_limits=(
                    BudgetLimit(
                        scope="run", max_estimated_cost=Decimal("0.000004"), pricing=pricing
                    ),
                )
                if scope == "run"
                else (),
                messages=[
                    Message.text("user", "oversized " + "x" * 5000),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(compactor_provider.requests) == 2
    assert runtime_provider.requests == []
    completions = [
        event
        for event in events
        if event.type == EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    ]
    assert len(completions) == 2
    assert EventType.BUDGET_RESERVATION_FAILED in {event.type for event in events}
    assert events[-1].type == EventType.SESSION_INTERRUPTED


def test_automatic_hierarchy_enforces_run_limit_before_every_dispatch() -> None:
    compactor_provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("short summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "summary-model",
                        "usage": {
                            "input_tokens": 7,
                            "output_tokens": 3,
                            "total_tokens": 10,
                        },
                    }
                ),
            ]
        ]
    )
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(enable_logging=False)
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
                max_input_chars=1000,
            ),
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_automatic_hierarchy_run_limit",
                messages=[
                    Message.text("user", "oversized " + "x" * 5000),
                    Message.text("user", "current"),
                ],
                limits=RunLimits(max_total_tokens=10),
            ),
        )
    )

    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []
    assert (
        sum(
            event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
            for event in events
        )
        == 1
    )
    assert EventType.SESSION_LIMIT_REACHED in {event.type for event in events}
    assert events[-1].type == EventType.SESSION_INTERRUPTED


def test_automatic_hierarchy_enforces_static_budget_before_every_dispatch() -> None:
    compactor_provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("short summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "summary-model",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 0,
                            "total_tokens": 10,
                        },
                    }
                ),
            ]
        ]
    )
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(
        enable_logging=False,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("0.000010"),
                    pricing=compaction_price_book(),
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
                max_input_chars=1000,
            ),
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_automatic_hierarchy_static_budget",
                messages=[
                    Message.text("user", "oversized " + "x" * 5000),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []
    assert (
        sum(
            event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
            for event in events
        )
        == 1
    )
    assert EventType.BUDGET_LIMIT_REACHED in {event.type for event in events}
    assert events[-1].type == EventType.SESSION_INTERRUPTED


def test_automatic_compaction_request_budget_prices_compactor_model_before_dispatch() -> None:
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("must not be dispatched"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                }
            ),
        ]
    )
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    session_only_pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="fake",
                model="fake-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
        )
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
            ),
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_request_budget_model_preflight",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
                budget_limits=(
                    BudgetLimit(
                        scope="run",
                        max_estimated_cost=Decimal("1"),
                        pricing=session_only_pricing,
                    ),
                ),
            ),
        )
    )

    assert compactor_provider.requests == []
    assert runtime_provider.requests == []
    limit_event = next(event for event in events if event.type == EventType.SESSION_LIMIT_REACHED)
    assert limit_event.payload["message"] == (
        "Budget cannot be verified because fake/summary-model: no matching model pricing."
    )
    assert EventType.MODEL_COMPLETED not in {event.type for event in events}
    assert events[-1].type == EventType.SESSION_INTERRUPTED


@pytest.mark.parametrize("stage", ["request", "completion"])
@pytest.mark.parametrize("typed", [False, True], ids=["plain", "typed"])
def test_model_compactor_drops_custom_billing_hook_credentials(
    stage: str,
    typed: bool,
) -> None:
    canary = f"provider-compaction-billing-{stage}-{typed}-canary-0123456789"

    def credential_failure() -> Exception:
        if typed:
            error: Exception = ModelProviderError(
                f"credential resolver failed near {canary}",
                provider=canary,
                status_code=401,
                error_type=canary,
                error_code=canary,
                request_id=canary,
                retryable=False,
                response_body=canary,
            )
        else:
            error = RuntimeError(f"credential resolver failed near {canary}")
        error.headers = {"Authorization": canary}  # type: ignore[attr-defined]
        error.add_note(f"resolver retained {canary}")
        return error

    class FailingCompactionProvider(FakeProvider):
        def __repr__(self) -> str:
            return f"FailingCompactionProvider(api_key={canary!r})"

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            if stage == "request":
                raise credential_failure()
            return None

        def billing_identity_for_completion(
            self,
            identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity | None:
            if stage == "completion":
                raise credential_failure()
            return identity

    provider = FailingCompactionProvider(
        [
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
        ]
    )
    compactor = ModelCompactor(provider=provider, model="summary-model")

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

    retained = str(exc_info.value) + repr(exc_info.value) + repr(vars(exc_info.value))
    assert canary not in retained
    assert str(exc_info.value) == "Model provider billing identity resolution failed"
    assert exc_info.value.provider == "fake"
    assert exc_info.value.error_type == "BillingIdentityResolutionError"
    assert exc_info.value.error_code == "billing_identity_resolution_failed"
    captured = traceback.TracebackException.from_exception(
        exc_info.value,
        capture_locals=True,
    )
    cayu_frames = [frame for frame in captured.stack if is_cayu_source_filename(frame.filename)]
    leaking_locals = [
        (frame.name, name)
        for frame in cayu_frames
        for name, value in (frame.locals or {}).items()
        if canary in value
    ]
    assert leaking_locals == []
    assert all(not frame.name.startswith("billing_identity_for_") for frame in captured.stack)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize("stage", ["request", "completion"])
def test_model_compactor_drops_custom_billing_hook_cancellation_credentials(
    stage: str,
) -> None:
    canary = f"provider-compaction-billing-cancel-{stage}-canary-0123456789"

    class CancellingCompactionProvider(FakeProvider):
        def __repr__(self) -> str:
            return f"CancellingCompactionProvider(api_key={canary!r})"

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            if stage == "request":
                raise asyncio.CancelledError(f"cancelled near {canary}")
            return None

        def billing_identity_for_completion(
            self,
            identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity | None:
            if stage == "completion":
                raise asyncio.CancelledError(f"cancelled near {canary}")
            return identity

    provider = CancellingCompactionProvider(
        [
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
        ]
    )

    with pytest.raises(ModelProviderError) as exc_info:
        asyncio.run(
            ModelCompactor(provider=provider, model="summary-model").compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )

    retained = str(exc_info.value) + repr(exc_info.value) + repr(vars(exc_info.value))
    assert str(exc_info.value) == "Model provider billing identity resolution failed"
    assert exc_info.value.provider == "fake"
    assert exc_info.value.error_type == "BillingIdentityResolutionError"
    assert exc_info.value.error_code == "billing_identity_resolution_failed"
    assert exc_info.value.retryable is False
    assert canary not in retained
    captured = traceback.TracebackException.from_exception(
        exc_info.value,
        capture_locals=True,
    )
    cayu_frames = [frame for frame in captured.stack if is_cayu_source_filename(frame.filename)]
    leaking_locals = [
        (frame.name, name)
        for frame in cayu_frames
        for name, value in (frame.locals or {}).items()
        if canary in value
    ]
    assert leaking_locals == []
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize("stage", ["request", "completion"])
def test_cayu_app_drops_automatic_compaction_billing_cancellation_credentials(
    stage: str,
) -> None:
    canary = f"automatic-compaction-billing-cancel-{stage}-canary-0123456789"

    class CancellingAutomaticCompactionProvider(FakeProvider):
        def __repr__(self) -> str:
            return f"CancellingAutomaticCompactionProvider(api_key={canary!r})"

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            if stage == "request":
                raise asyncio.CancelledError(f"cancelled near {canary}")
            return None

        def billing_identity_for_completion(
            self,
            identity: BillingIdentity | None,
            payload: dict[str, Any],
        ) -> BillingIdentity | None:
            if stage == "completion":
                raise asyncio.CancelledError(f"cancelled near {canary}")
            return identity

    provider = CancellingAutomaticCompactionProvider(
        [
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(provider=provider, model="summary-model"),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_automatic_compaction_billing_cancel_{stage}",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    retained = json.dumps([event.model_dump(mode="json") for event in events], sort_keys=True)
    assert canary not in retained
    assert events[-1].type is EventType.SESSION_FAILED
    assert EventType.SESSION_INTERRUPTED not in {event.type for event in events}
    expected_terminal: dict[str, Any] = {
        "failure_evidence": _expected_failure_evidence(
            f"sess_automatic_compaction_billing_cancel_{stage}",
            "ModelProviderError",
            "ContextBuildError",
        ),
        "error": "Model provider billing identity resolution failed",
        "error_type": "ModelProviderError",
    }
    if stage == "completion":
        failure = next(
            event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED
        )
        expected_disposition = {
            "phase": "provider_dispatch",
            "reason": "internal_failed",
            "retryable": False,
            "provider_dispatch_disposition": "unknown",
            "recovery_action": "reconcile_completion",
        }
        assert {key: failure.payload[key] for key in expected_disposition} == expected_disposition
        assert type(failure.payload["elapsed_ms"]) is int
        expected_terminal["compaction_failure"] = {
            **expected_disposition,
            "elapsed_ms": failure.payload["elapsed_ms"],
        }
    assert events[-1].payload == expected_terminal


def test_cayu_app_detaches_grouped_compaction_billing_cancellation_credentials() -> None:
    provider_canary = "automatic-compaction-grouped-cancel-canary-0123456789"
    cleanup_marker = "automatic compaction workspace cleanup failed"

    class CancellingCompactionProvider(FakeProvider):
        def __repr__(self) -> str:
            return f"CancellingCompactionProvider(api_key={provider_canary!r})"

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            raise asyncio.CancelledError(f"cancelled near {provider_canary}")

    provider = CancellingCompactionProvider([])
    binding = RecordingWorkspaceBinding(
        fail_finalize=True,
        finalize_error=RuntimeError(cleanup_marker),
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), binding=binding),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(provider=provider, model="summary-model"),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_automatic_compaction_grouped_billing_cancel",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    retained = json.dumps([event.model_dump(mode="json") for event in events], sort_keys=True)
    assert provider_canary not in retained
    assert cleanup_marker in retained
    assert events[-1].type is EventType.SESSION_FAILED
    assert EventType.SESSION_INTERRUPTED not in {event.type for event in events}
    assert events[-1].payload["error"] == "Model provider billing identity resolution failed"
    assert events[-1].payload["error_type"] == "ModelProviderError"


@pytest.mark.parametrize("grouped", [False, True])
def test_cayu_app_detaches_billing_cancellation_before_fallible_interruption_checks(
    grouped: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = f"billing-cancellation-transition-failure-{grouped}-canary-0123456789"

    class CredentialBearingProvider(FakeProvider):
        def __repr__(self) -> str:
            return f"CredentialBearingProvider(api_key={canary!r})"

    provider = CredentialBearingProvider([])
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    private_cancellation_raised = False

    async def failing_session(**kwargs: Any):
        nonlocal private_cancellation_raised
        del kwargs
        private_cancellation_raised = True
        cancellation = _BillingIdentityResolutionCancelled(f"cancelled near {canary}")
        if grouped:
            raise BaseExceptionGroup(
                "credential-bearing group",
                [cancellation, RuntimeError(f"cleanup failed near {canary}")],
            )
        raise cancellation
        yield

    app._session_engine._run_session = failing_session  # type: ignore[method-assign]

    if grouped:

        async def unexpected_load(_store: object, _session_id: str) -> Session | None:
            if private_cancellation_raised:
                raise AssertionError("billing detachment must not perform an unauthenticated load")
            return None

        monkeypatch.setattr(type(app.session_store), "load", unexpected_load)
    else:

        async def unexpected_interrupt_requested(_session_id: str) -> bool:
            raise AssertionError(
                "billing detachment must not perform an unauthenticated interruption check"
            )

        app._session_control.interrupt_requested = (  # type: ignore[method-assign]
            unexpected_interrupt_requested
        )

    expected_failure = BaseExceptionGroup if grouped else asyncio.CancelledError
    with pytest.raises(expected_failure) as exc_info:
        asyncio.run(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=f"sess_billing_cancellation_transition_failure_{grouped}",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )

    retained = str(exc_info.value) + repr(exc_info.value) + repr(vars(exc_info.value))
    assert canary not in retained
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    captured = traceback.TracebackException.from_exception(
        exc_info.value,
        capture_locals=True,
    )
    cayu_frames = [frame for frame in captured.stack if is_cayu_source_filename(frame.filename)]
    assert canary not in repr([(frame.name, frame.locals) for frame in cayu_frames])


@pytest.mark.parametrize("signal_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_detached_billing_cancellation_group_preserves_fatal_signal_types(
    signal_type: type[BaseException],
) -> None:
    canary = f"concurrent-{signal_type.__name__}-canary-0123456789"
    original = BaseExceptionGroup(
        "outer group",
        [
            _BillingIdentityResolutionCancelled(f"cancelled near {canary}"),
            BaseExceptionGroup(
                "nested group",
                [
                    signal_type(f"fatal signal near {canary}"),
                ],
            ),
        ],
    )

    detached = detach_billing_identity_cancellation_group(original)

    assert detached is not None
    assert isinstance(detached.exceptions[0], asyncio.CancelledError)
    nested = detached.exceptions[1]
    assert isinstance(nested, BaseExceptionGroup)
    fatal = next(child for child in nested.exceptions if isinstance(child, signal_type))
    assert canary not in (str(fatal) + repr(fatal) + repr(vars(fatal)))
    assert fatal.__traceback__ is None
    assert fatal.__cause__ is None
    assert fatal.__context__ is None


def test_detached_billing_cancellation_group_preserves_integer_system_exit_code() -> None:
    original = BaseExceptionGroup(
        "outer group",
        [
            _BillingIdentityResolutionCancelled("caller cancellation"),
            SystemExit(17),
        ],
    )

    detached = detach_billing_identity_cancellation_group(original)

    assert detached is not None
    fatal = next(child for child in detached.exceptions if isinstance(child, SystemExit))
    assert fatal.code == 17


@pytest.mark.parametrize("limit_source", ["policy", "request"])
@pytest.mark.parametrize(
    ("resolved_identity", "session_suffix"),
    [
        pytest.param(
            bedrock_billing_identity(
                invoked_model="global.anthropic.claude-sonnet-4-6",
                source_region="eu-west-1",
                resource_type="inference_profile",
                profile_scope="global",
            ),
            "wrong_region",
            id="mismatched-context",
        ),
        pytest.param(None, "missing_identity", id="missing-identity"),
    ],
)
def test_automatic_compaction_strict_contextual_budget_rejects_before_dispatch(
    limit_source: str,
    resolved_identity: BillingIdentity | None,
    session_suffix: str,
) -> None:
    invoked_model = "global.anthropic.claude-sonnet-4-6"

    class ContextualCompactionProvider(FakeProvider):
        name = "bedrock-adapter"
        billing_provider_name = "bedrock"

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            assert request.model == invoked_model
            return resolved_identity

    compactor_provider = ContextualCompactionProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed(
                {
                    "model": invoked_model,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                }
            ),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": "fake-model",
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                }
            )
        ]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="fake",
                model="fake-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
            ModelPrice.fixed(
                provider_name="bedrock",
                model=invoked_model,
                match="exact",
                input_per_million=Decimal("3"),
                output_per_million=Decimal("15"),
                pricing_context={
                    "source_region": ("us-east-1",),
                    "service_tier": ("default",),
                },
            ),
        )
    )
    limit = BudgetLimit(
        scope="app" if limit_source == "policy" else "run",
        max_estimated_cost=Decimal("20"),
        pricing=pricing,
    )
    app = CayuApp(
        budget_policy=(BudgetPolicy(limits=(limit,)) if limit_source == "policy" else None)
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model=invoked_model,
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
                session_id=(f"sess_automatic_compaction_{limit_source}_{session_suffix}"),
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
                budget_limits=(limit,) if limit_source == "request" else (),
            ),
        )
    )

    assert compactor_provider.requests == []
    assert runtime_provider.requests == []
    failures = [event for event in events if event.type == EventType.BUDGET_RESERVATION_FAILED]
    assert len(failures) == 1
    assert "no matching model pricing" in failures[0].payload["message"]
    assert EventType.MODEL_COMPLETED not in [event.type for event in events]
    assert events[-1].type == EventType.SESSION_INTERRUPTED


def test_session_query_validates_label_selector_requirements():
    query = SessionQuery(
        label_selectors=[
            {"key": "project", "operator": "in", "values": ["ap_q2", "research"]},
            {"key": "workflow", "operator": "exists"},
        ]
    )

    assert [selector.key for selector in query.label_selectors] == ["project", "workflow"]

    with pytest.raises(ValidationError):
        SessionQuery(label_selectors={"key": "project", "operator": "exists"})


def test_cayu_app_uses_registered_provider_name_after_provider_mutation():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    provider.name = "mutated"  # type: ignore[assignment]

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_provider_name_snapshot",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert events[1].type == EventType.MODEL_STARTED
    assert events[1].payload["provider"] == "fake"


@pytest.mark.parametrize(
    ("publication_type", "expected_compactor_calls"),
    [
        (EventType.CONTEXT_COMPACTION_STARTED, 0),
        (EventType.MODEL_COMPLETED, 1),
    ],
)
@pytest.mark.parametrize("release_reconciliation", [True, False])
def test_automatic_compaction_cancellation_during_publication_reconciliation_propagates(
    publication_type: EventType,
    expected_compactor_calls: int,
    release_reconciliation: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingCompactionProvider(ModelProvider):
        name = "reconciliation-cancellation-provider"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            self.calls += 1
            yield ModelStreamEvent.text_delta("partial")
            raise ModelProviderError(
                "provider failed",
                provider=self.name,
                retryable=True,
            )

    async def run() -> None:
        store = InMemorySessionStore()
        compaction_provider = FailingCompactionProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="summary-model",
                    retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        reconciliation_started = asyncio.Event()
        allow_reconciliation = asyncio.Event()
        original_emit_many = app._event_writer.emit_many
        original_complete_stage = store.complete_model_completion_stage
        original_is_persisted = app._event_writer.is_persisted

        async def fail_start_publication(session_id: str, events: list[Event]):
            if any(event.type == EventType.CONTEXT_COMPACTION_STARTED for event in events):
                raise RuntimeError("publication acknowledgement failed")
            return await original_emit_many(session_id, events)

        async def fail_completion_persistence(session_id: str, **kwargs):
            result = await original_complete_stage(session_id, **kwargs)
            if result.stage.purpose == "context-compaction":
                raise RuntimeError("publication acknowledgement failed")
            return result

        async def block_target_reconciliation(event: Event) -> bool:
            if event.type == publication_type:
                reconciliation_started.set()
                await allow_reconciliation.wait()
                return False
            return await original_is_persisted(event)

        if publication_type == EventType.CONTEXT_COMPACTION_STARTED:
            monkeypatch.setattr(app._event_writer, "emit_many", fail_start_publication)
        else:
            monkeypatch.setattr(
                store, "complete_model_completion_stage", fail_completion_persistence
            )
        monkeypatch.setattr(app._event_writer, "is_persisted", block_target_reconciliation)
        if not release_reconciliation:
            # Bound the blocked read after cancellation without shortening
            # ordinary publication or reconciliation during setup.
            monkeypatch.setattr(
                _automatic_compaction,
                "_CONTEXT_EVENT_STORE_WAIT_AFTER_CANCELLATION_TIMEOUT_S",
                0.01,
            )
        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=f"sess_reconciliation_cancel_{publication_type.value}",
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                ),
            )
        )
        await asyncio.wait_for(reconciliation_started.wait(), timeout=10)
        task.cancel("cancel during compaction publication reconciliation")
        assert task.cancelling() == 1
        if release_reconciliation:
            allow_reconciliation.set()
        # Observe completion without injecting a second cancellation during
        # termination telemetry. The short reconciliation deadline above is
        # the behavior under test; this guard allows loaded-runner cleanup.
        done, _pending = await asyncio.wait({task}, timeout=30)
        assert task in done
        with pytest.raises(
            asyncio.CancelledError,
            match="cancel during compaction publication reconciliation",
        ):
            await task

        assert task.cancelled()
        assert compaction_provider.calls == expected_compactor_calls

    asyncio.run(run())


@pytest.mark.parametrize("publication_delay", [0, 0.05])
def test_automatic_compaction_cancellation_does_not_wait_for_stalled_completion_sink(
    monkeypatch: pytest.MonkeyPatch,
    publication_delay: float,
) -> None:
    class CompletionStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def complete_model_completion_stage(self, session_id: str, **kwargs):
            # Exercise publication slower than the old 10 ms setup deadline.
            await asyncio.sleep(publication_delay)
            return await super().complete_model_completion_stage(session_id, **kwargs)

    class BlockingCompletionSink(InMemoryEventSink):
        def __init__(self) -> None:
            super().__init__()
            self.blocked = asyncio.Event()
            self.release = asyncio.Event()
            self.cancellation_observed = asyncio.Event()
            self.completion_delivery_calls = 0

        async def emit(self, event: Event) -> None:
            if (
                event.type == EventType.MODEL_COMPLETED
                and event.payload.get("purpose") == "context_compaction"
            ):
                self.completion_delivery_calls += 1
                if self.completion_delivery_calls == 1:
                    self.blocked.set()
                    while not self.release.is_set():
                        try:
                            await self.release.wait()
                        except asyncio.CancelledError:
                            # Model a sink whose in-flight delivery cannot be
                            # interrupted even though Cayu's caller must be.
                            self.cancellation_observed.set()
            await super().emit(event)

    class CompletionProvider(ModelProvider):
        name = "automatic-stalled-completion-sink-provider"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

    async def run() -> None:
        # Shorten only cancellation draining, not ordinary durable publication.
        monkeypatch.setattr(
            _automatic_compaction,
            "_CONTEXT_EVENT_STORE_WAIT_AFTER_CANCELLATION_TIMEOUT_S",
            0.01,
        )
        store = CompletionStore()
        sink = BlockingCompletionSink()
        app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=CompletionProvider(),
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )

        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_automatic_stalled_completion_sink",
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                ),
            )
        )
        try:
            # Reach the cancellation boundary using the normal publication timeout.
            # A 10 ms timeout during setup can abort compaction on a busy CI worker.
            await asyncio.wait_for(sink.blocked.wait(), timeout=10)
            durable = await store.load_events("sess_automatic_stalled_completion_sink")
            completions = [
                event
                for event in durable
                if event.type == EventType.MODEL_COMPLETED
                and event.payload.get("purpose") == "context_compaction"
            ]
            assert len(completions) == 1

            task.cancel("cancel automatic compaction with stalled completion sink")
            assert task.cancelling() == 1
            await asyncio.wait_for(sink.cancellation_observed.wait(), timeout=1)
            done, _pending = await asyncio.wait({task}, timeout=1)
            # Prove caller cancellation completes before the non-cancellable sink.
            assert task in done
            with pytest.raises(
                asyncio.CancelledError,
                match="cancel automatic compaction with stalled completion sink",
            ):
                await task
            assert task.cancelled()

            sink.release.set()
            for _ in range(100):
                delivered = [
                    event
                    for event in sink.events
                    if event.type == EventType.MODEL_COMPLETED
                    and event.payload.get("purpose") == "context_compaction"
                ]
                if delivered:
                    break
                await asyncio.sleep(0.01)
            completion_record = next(
                record
                for record in await store.query_events(
                    EventQuery(session_id="sess_automatic_stalled_completion_sink")
                )
                if record.event.id == completions[0].id
            )
            assert [event.id for event in delivered] == [
                public_event_id(completion_record.sequence)
            ]
            assert sink.completion_delivery_calls == 1
        finally:
            sink.release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_automatic_compaction_late_completion_commit_keeps_one_accounting_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DelayedCompletionStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.blocked = asyncio.Event()
            self.release = asyncio.Event()
            self.blocked_once = False
            self.cancellations = 0

        async def complete_model_completion_stage(self, session_id: str, **kwargs):
            publication = kwargs["publication"]
            if not self.blocked_once and any(
                event.type == EventType.MODEL_COMPLETED
                and event.payload.get("purpose") == "context_compaction"
                for event in publication.events
            ):
                self.blocked_once = True
                self.blocked.set()
                while not self.release.is_set():
                    try:
                        await self.release.wait()
                    except asyncio.CancelledError:
                        # Match a SQLite worker whose physical commit continues
                        # after cancellation of the awaiting task.
                        self.cancellations += 1
            return await super().complete_model_completion_stage(session_id, **kwargs)

    class CompletionProvider(ModelProvider):
        name = "automatic-late-completion-provider"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

    async def run() -> None:
        monkeypatch.setattr(
            _automatic_compaction,
            "_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S",
            0.01,
        )
        store = DelayedCompletionStore()
        sink = InMemoryEventSink()
        compaction_provider = CompletionProvider()
        app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )

        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_automatic_late_completion_commit",
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                ),
            )
        )
        try:
            await asyncio.wait_for(store.blocked.wait(), timeout=5)
            await asyncio.sleep(0.05)
            assert not task.done()
        finally:
            store.release.set()
        events = await asyncio.wait_for(task, timeout=5)
        assert events[-1].type == EventType.SESSION_FAILED
        for _ in range(100):
            durable = await store.load_events("sess_automatic_late_completion_commit")
            completions = [
                event
                for event in durable
                if event.type == EventType.MODEL_COMPLETED
                and event.payload.get("purpose") == "context_compaction"
            ]
            if completions:
                break
            await asyncio.sleep(0.01)

        assert len(completions) == 1
        sink_completions = [
            event
            for event in sink.events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        completion_records = [
            record
            for record in await store.query_events(
                EventQuery(session_id="sess_automatic_late_completion_commit")
            )
            if record.event.id == completions[0].id
        ]
        assert [event.id for event in sink_completions] == [
            public_event_id(completion_records[0].sequence)
        ]

    asyncio.run(run())


def test_automatic_compaction_late_start_commit_reuses_original_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DelayedStartStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.blocked = asyncio.Event()
            self.release = asyncio.Event()
            self.committed = asyncio.Event()
            self.blocked_once = False
            self.original_event_id: str | None = None
            self.cancellations = 0

        async def append_events(self, session_id: str, events: list[Event]) -> None:
            start = next(
                (event for event in events if event.type == EventType.CONTEXT_COMPACTION_STARTED),
                None,
            )
            if not self.blocked_once and start is not None:
                self.blocked_once = True
                self.original_event_id = start.id
                self.blocked.set()
                while not self.release.is_set():
                    try:
                        await self.release.wait()
                    except asyncio.CancelledError:
                        # Match a physical store write that cannot be cancelled.
                        self.cancellations += 1
            await super().append_events(session_id, events)
            if start is not None and start.id == self.original_event_id:
                self.committed.set()

    class CountingCompactionProvider(ModelProvider):
        name = "automatic-late-start-provider"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.calls += 1
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed({"model": request.model})

    async def run() -> None:
        monkeypatch.setattr(
            _automatic_compaction,
            "_AUTOMATIC_COMPACTION_PREDISPATCH_EVENT_STORE_WAIT_TIMEOUT_S",
            0.01,
        )
        store = DelayedStartStore()
        sink = InMemoryEventSink()
        compaction_provider = CountingCompactionProvider()
        app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        original_is_persisted = app._event_writer.is_persisted
        stale_start_checks = 0

        async def release_after_stale_start_check(event: Event) -> bool:
            nonlocal stale_start_checks
            persisted = await original_is_persisted(event)
            if (
                stale_start_checks < 1
                and event.type == EventType.CONTEXT_COMPACTION_STARTED
                and not persisted
            ):
                stale_start_checks += 1
                store.release.set()
                await store.committed.wait()
                # The reconciliation read is stale; the original physical
                # write commits immediately before the exact replay.
                return False
            return persisted

        app._event_writer.is_persisted = release_after_stale_start_check
        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_automatic_late_start_commit",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )

        assert events[-1].type == EventType.SESSION_COMPLETED
        assert stale_start_checks == 1
        assert store.committed.is_set()
        assert store.cancellations >= 1
        assert compaction_provider.calls == 1
        durable = await store.load_events("sess_automatic_late_start_commit")
        starts = [event for event in durable if event.type == EventType.CONTEXT_COMPACTION_STARTED]
        assert [event.id for event in starts] == [store.original_event_id]
        assert EventType.CONTEXT_COMPACTION_FAILED not in {event.type for event in durable}
        sink_starts = [
            event for event in sink.events if event.type == EventType.CONTEXT_COMPACTION_STARTED
        ]
        start_records = [
            record
            for record in await store.query_events(
                EventQuery(session_id="sess_automatic_late_start_commit")
            )
            if record.event.id == store.original_event_id
        ]
        assert [event.id for event in sink_starts] == [public_event_id(start_records[0].sequence)]

    asyncio.run(run())


def test_automatic_compaction_retries_start_publication_with_causal_event() -> None:
    class CountingCompactionProvider(ModelProvider):
        name = "automatic-failed-start-provider"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.calls += 1
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed({"model": request.model})

    async def run() -> None:
        store = InMemorySessionStore()
        compaction_provider = CountingCompactionProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        original_emit_many = app._event_writer.emit_many
        rejected_start_id: str | None = None

        async def reject_first_start(session_id: str, events: list[Event]) -> list[Event]:
            nonlocal rejected_start_id
            start = next(
                (event for event in events if event.type == EventType.CONTEXT_COMPACTION_STARTED),
                None,
            )
            if rejected_start_id is None and start is not None:
                rejected_start_id = start.id
                raise ConnectionError("start publication failed before commit")
            return await original_emit_many(session_id, events)

        app._event_writer.emit_many = reject_first_start
        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_automatic_failed_start_publication",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )

        assert events[-1].type == EventType.SESSION_COMPLETED
        assert compaction_provider.calls == 1
        durable = await store.load_events("sess_automatic_failed_start_publication")
        starts = [event for event in durable if event.type == EventType.CONTEXT_COMPACTION_STARTED]
        failures = [event for event in durable if event.type == EventType.CONTEXT_COMPACTION_FAILED]
        assert [event.id for event in starts] == [rejected_start_id]
        assert failures == []

    asyncio.run(run())


def test_automatic_compaction_exhausted_start_publication_is_typed_and_resumable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CountingCompactionProvider(ModelProvider):
        name = "automatic-exhausted-start-provider"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            self.calls += 1
            yield ModelStreamEvent.completed({})

    async def run() -> None:
        session_id = "sess_automatic_exhausted_start_publication"
        store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        compaction_provider = CountingCompactionProvider()
        await task_store.create_task(
            TaskCreate(
                task_id="task_automatic_exhausted_start_publication",
                type="compaction-recovery",
                assigned_agent_name="assistant",
            )
        )
        app = CayuApp(
            session_store=store,
            task_store=task_store,
            enable_logging=False,
        )
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        attempts = 0
        original_emit_many = app._event_writer.emit_many

        async def reject_start(session_id: str, events: list[Event]) -> list[Event]:
            nonlocal attempts
            if all(event.type == EventType.CONTEXT_COMPACTION_STARTED for event in events):
                attempts += 1
                raise TimeoutError("start publication timed out before commit")
            return await original_emit_many(session_id, events)

        monkeypatch.setattr(
            _automatic_compaction,
            "_AUTOMATIC_COMPACTION_PREDISPATCH_PUBLICATION_ATTEMPTS",
            2,
        )
        monkeypatch.setattr(app._event_writer, "emit_many", reject_start)
        returned = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                task_id="task_automatic_exhausted_start_publication",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )

        assert attempts == 2
        assert compaction_provider.calls == 0
        failure = next(
            event for event in returned if event.type == EventType.CONTEXT_COMPACTION_FAILED
        )
        expected = {
            "phase": "start_publication",
            "reason": "publication_timeout",
            "retryable": True,
            "provider_dispatch_disposition": "not_dispatched",
            "recovery_action": "resume_session",
        }
        assert {key: failure.payload[key] for key in expected} == expected
        assert type(failure.payload["elapsed_ms"]) is int
        terminal = returned[-1]
        assert terminal.type == EventType.SESSION_FAILED
        assert terminal.payload["compaction_failure"] == {
            **expected,
            "elapsed_ms": failure.payload["elapsed_ms"],
        }
        task = await task_store.load_task("task_automatic_exhausted_start_publication")
        assert task is not None
        assert task.status == TaskStatus.FAILED
        assert task.error is not None
        assert task.error["compaction_failure"] == terminal.payload["compaction_failure"]

    asyncio.run(run())


@pytest.mark.parametrize(
    "publication_type",
    [EventType.REQUEST_FOOTPRINT_RECORDED, EventType.MODEL_STARTED],
)
def test_automatic_compaction_retries_predispatch_publication_without_duplicate_provider_call(
    publication_type: EventType,
    request_footprint_session_store,
) -> None:
    class CompactionProvider(ModelProvider):
        name = "predispatch-publication-compactor"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.calls += 1
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed({"model": request.model})

    async def run() -> None:
        session_id = f"sess_automatic_{publication_type.value.replace('.', '_')}"
        compaction_provider = CompactionProvider()
        app = CayuApp(
            session_store=request_footprint_session_store,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="compaction-publication-key",
                fingerprint_key=SecretStr("p" * 32),
            ),
            enable_logging=False,
        )
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        rule = SessionOperationFaultRule(
            rule_id="compaction-predispatch",
            selector=SessionOperationSelector(
                session_id=session_id,
                event_types=frozenset({publication_type}),
            ),
            actions=(FailBeforeTransform(),),
            on_exhausted=MatchPolicy.DELEGATE,
        )
        async with SessionOperationFaultHarness(
            request_footprint_session_store,
            rules=(rule,),
            boundary=PublicationBoundary.EVENT_APPEND,
        ) as faults:
            returned = await collect_events(
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

        assert returned[-1].type == EventType.SESSION_COMPLETED
        assert compaction_provider.calls == 1
        injected = [
            trace for trace in faults.trace if trace.matched_rule_id == "compaction-predispatch"
        ]
        assert len(injected) >= 2
        assert injected[0].action is PublicationFaultActionKind.FAIL_BEFORE_TRANSFORM
        assert injected[0].outcome is PublicationFaultOutcome.INJECTED_FAILURE
        assert injected[0].committed is CommitEvidence.NO
        assert injected[1].action is PublicationFaultActionKind.EXHAUSTED_DELEGATE
        assert injected[1].outcome is PublicationFaultOutcome.RETURNED
        assert injected[1].committed is CommitEvidence.YES
        durable = await request_footprint_session_store.load_events(session_id)
        target_events = [event for event in durable if event.type == publication_type]
        if publication_type == EventType.REQUEST_FOOTPRINT_RECORDED:
            target_events = [
                event
                for event in target_events
                if event.payload.get("request_variant") == "context_compaction"
            ]
        else:
            target_events = [
                event
                for event in target_events
                if event.payload.get("purpose") == "context_compaction"
            ]
        assert len(target_events) == 1

    asyncio.run(run())


def test_automatic_hierarchy_exhausted_next_start_requires_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CountingHierarchyProvider(ModelProvider):
        name = "automatic-hierarchy-publication-provider"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            yield ModelStreamEvent.text_delta(f"summary-{len(self.requests)}")
            yield ModelStreamEvent.completed({"model": request.model})

    async def run() -> None:
        session_id = "sess_automatic_hierarchy_next_start_failure"
        store = InMemorySessionStore()
        compaction_provider = CountingHierarchyProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        original_persist_exact_replay = app._event_writer.persist_exact_replay
        model_start_publications = 0

        async def fail_second_compactor_start(event: Event) -> Event:
            nonlocal model_start_publications
            if (
                event.type == EventType.MODEL_STARTED
                and event.payload.get("purpose") == "context_compaction"
            ):
                model_start_publications += 1
                if model_start_publications >= 2:
                    raise TimeoutError("next hierarchy model-start publication unavailable")
            return await original_persist_exact_replay(event)

        monkeypatch.setattr(
            _automatic_compaction,
            "_AUTOMATIC_COMPACTION_PREDISPATCH_PUBLICATION_ATTEMPTS",
            3,
        )
        monkeypatch.setattr(
            app._event_writer,
            "persist_exact_replay",
            fail_second_compactor_start,
        )
        returned = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[
                    Message.text("user", "oversized " + "x" * 5000),
                    Message.text("user", "current"),
                ],
            ),
        )
        durable = await store.load_events(session_id)

        assert returned[-1].type == EventType.SESSION_FAILED
        assert len(compaction_provider.requests) == 1
        assert (
            sum(
                event.type == EventType.MODEL_COMPLETED
                and event.payload.get("purpose") == "context_compaction"
                for event in durable
            )
            == 1
        )
        failure = next(
            event for event in durable if event.type == EventType.CONTEXT_COMPACTION_FAILED
        )
        expected = {
            "phase": "model_start_publication",
            "reason": "publication_timeout",
            "retryable": False,
            "provider_dispatch_disposition": "dispatched",
            "recovery_action": "reconcile_completion",
        }
        assert {key: failure.payload[key] for key in expected} == expected
        assert {
            key: returned[-1].payload["compaction_failure"][key] for key in expected
        } == expected

    asyncio.run(run())


def test_cayu_app_context_policy_can_trim_model_facing_messages():
    class LastMessagePolicy(ContextPolicy):
        def __init__(self) -> None:
            self.seen: list[ContextRequest] = []

        async def build(self, request: ContextRequest) -> list[Message]:
            self.seen.append(request)
            # Messages are frozen; policies cannot corrupt the transcript.
            with pytest.raises(ValidationError):
                request.messages[0].content[0].text = "mutated"  # type: ignore[misc]
            return [request.messages[-1]]

    policy = LastMessagePolicy()
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("trimmed"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_trim_context",
                messages=[
                    Message.text("user", "old context"),
                    Message.text("user", "current request"),
                ],
                metadata={"source": "test"},
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert len(policy.seen) == 1
    assert policy.seen[0].session.id == "sess_trim_context"
    assert policy.seen[0].agent.name == "assistant"
    assert policy.seen[0].step == 1
    assert policy.seen[0].metadata == {"source": "test"}
    assert [message.content[0].text for message in provider.requests[0].messages] == [
        "current request"
    ]

    transcript = asyncio.run(store.load_transcript("sess_trim_context"))
    assert [message.content[0].text for message in transcript] == [
        "old context",
        "current request",
        "trimmed",
    ]


def test_cayu_app_context_policy_can_replace_tool_results_for_model_only():
    class CompactToolResultPolicy(ContextPolicy):
        async def build(self, request: ContextRequest) -> list[Message]:
            compacted: list[Message] = []
            for message in request.messages:
                if message.role == "tool":
                    compacted.append(
                        Message.tool_result(
                            tool_call_id=message.content[0].tool_call_id,
                            tool_name=message.content[0].tool_name,
                            content="[tool result compacted]",
                            structured={"compacted": True},
                        )
                    )
                else:
                    compacted.append(message)
            return compacted

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "large tool output"},
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
        context_policy=CompactToolResultPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_tool_result",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    provider_tool_result = provider.requests[1].messages[-1].content[0]
    assert provider_tool_result.content == "[tool result compacted]"
    assert provider_tool_result.structured == {"compacted": True}

    transcript = asyncio.run(store.load_transcript("sess_compact_tool_result"))
    stored_tool_result = transcript[2].content[0]
    assert stored_tool_result.content == "large tool output"
    assert stored_tool_result.structured == {
        "agent": "assistant",
        "echoed": "large tool output",
    }


def test_cayu_app_fails_cleanly_when_context_policy_returns_invalid_output():
    class BadPolicy(ContextPolicy):
        async def build(self, request: ContextRequest):
            return tuple(request.messages)

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("should not run"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=BadPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_bad_context_policy",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert provider.requests == []
    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert (
        events[-1].payload["error"]
        == "ContextPolicy.build() must return a list of Message instances."
    )
    transcript = asyncio.run(store.load_transcript("sess_bad_context_policy"))
    assert [message.content[0].text for message in transcript] == ["hi"]


def test_cayu_app_rejects_context_policy_that_cuts_through_tool_round():
    class OrphanToolResultPolicy(ContextPolicy):
        async def build(self, request: ContextRequest) -> list[Message]:
            return [request.messages[-1]]

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "large tool output"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("should not run"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
        context_policy=OrphanToolResultPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_orphan_tool_result",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert "tool results without preceding assistant tool calls" in events[-1].payload["error"]


def test_builtin_context_policies_trim_recent_history_safely():
    messages = [
        Message.text("system", "You are careful."),
        Message.text("user", "old"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current"),
    ]

    turns_policy = RecentTurnsContextPolicy(max_user_turns=1)
    messages_policy = MessageWindowContextPolicy(max_messages=2)

    async def build_contexts() -> tuple[list[Message], list[Message]]:
        request = ContextRequest(
            session=_test_session(),
            agent=AgentSpec(name="assistant", model="fake-model"),
            messages=messages,
            step=1,
        )
        return (
            await turns_policy.build(request),
            await messages_policy.build(request),
        )

    turns_context, messages_context = asyncio.run(build_contexts())

    assert [message.role for message in turns_context] == ["system", "user"]
    assert [message.content[0].text for message in turns_context] == [
        "You are careful.",
        "current",
    ]
    assert [message.role for message in messages_context] == ["system", "user"]
    assert [message.content[0].text for message in messages_context] == [
        "You are careful.",
        "current",
    ]


def test_model_compactor_ignores_thinking_events():
    # A reasoning-capable compaction provider may emit THINKING events; compaction must
    # ignore them (the summary is built from text) rather than crash on an unknown type.
    provider = FakeProvider(
        [
            ModelStreamEvent.thinking("internal reasoning", provider_state={"signature": "S"}),
            ModelStreamEvent.text_delta("summary "),
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"usage": {"output_tokens": 2}}),
        ]
    )
    compactor = ModelCompactor(provider=provider, model="summary-model")

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old request")],
            )
        )
    )

    assert result.summary == "summary done"


def test_model_compactor_summarizes_context_with_provider():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("summary "),
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"usage": {"output_tokens": 2}}),
        ]
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        options={"anthropic": {"max_tokens": 512}},
    )

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[
                    Message.text("user", "old request"),
                    Message.text("assistant", "old answer"),
                ],
                existing_summary="previous context",
            )
        )
    )

    assert result.summary == "summary done"
    assert result.metadata == {
        "compactor": "ModelCompactor",
        "provider": "fake",
        "model": "summary-model",
        "input_truncated": False,
        "max_input_chars": 120000,
        "completed": {"usage": {"output_tokens": 2}},
    }
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.model == "summary-model"
    assert request.tools == []
    assert request.options == {"anthropic": {"max_tokens": 512}}
    assert [message.role for message in request.messages] == ["system", "user"]
    prompt = request.messages[1].content[0].text
    assert "Existing summary:\nprevious context" in prompt
    assert "user: old request" in prompt
    assert "assistant: old answer" in prompt


def test_model_compactor_reports_model_completed_payload_with_usage_metrics():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model-2026-01-01",
                    "usage": {"input_tokens": 40, "output_tokens": 8},
                    "provider_state": {"opaque": True},
                }
            ),
        ]
    )
    compactor = ModelCompactor(provider=provider, model="summary-model")

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old request")],
            )
        )
    )

    assert result.model_completed_payloads == [
        {
            "model": "summary-model-2026-01-01",
            "usage": {"input_tokens": 40, "output_tokens": 8},
            "provider_name": "fake",
            "requested_model": "summary-model",
            "purpose": "context_compaction",
            "compactor": "ModelCompactor",
            "usage_metrics": {
                "provider_name": "fake",
                "requested_model": "summary-model",
                "model": "summary-model-2026-01-01",
                "input_tokens": 40,
                "output_tokens": 8,
                "total_tokens": 48,
                "reasoning_output_tokens": 0,
                "cache": {
                    "read_tokens": 0,
                    "write_tokens": 0,
                    "cached_input_tokens": 0,
                    "uncached_input_tokens": 40,
                },
            },
        }
    ]


def test_model_compactor_retries_transient_provider_errors():
    provider = FlakyCompactionProvider(
        failures=1,
        error=ModelProviderError(
            "provider overloaded",
            provider="flaky",
            status_code=503,
            retryable=True,
        ),
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


def test_model_compactor_retries_transient_provider_error_events():
    typed_error = ModelProviderError(
        "provider overloaded",
        provider="fake",
        status_code=529,
        retryable=True,
        retry_after_s=0.0,
    )
    provider = FakeProvider(
        [
            [ModelStreamEvent.error("provider overloaded", cause=typed_error)],
            [
                ModelStreamEvent.text_delta("recovered summary"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
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


def test_model_compactor_reports_every_completion_when_retry_follows_completed():
    provider = CompletedThenRetryableCompactionProvider()
    result = asyncio.run(
        ModelCompactor(
            provider=provider,
            model="summary-model",
            retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
        ).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old request")],
            )
        )
    )

    assert result.summary == "summary 2"
    assert len(provider.requests) == 2
    assert [
        payload["usage_metrics"]["input_tokens"] for payload in result.model_completed_payloads
    ] == [1, 2]
    assert all(
        "compaction_attempt_id" not in payload for payload in result.model_completed_payloads
    )


def test_prompt_cache_compactor_reports_every_completion_when_retry_follows_completed():
    provider = CompletedThenRetryableCompactionProvider()
    context_messages = [
        Message.text("user", "cached request"),
        Message.text("assistant", "cached answer"),
    ]
    result = asyncio.run(
        PromptCacheCompactor(
            provider=provider,
            retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
        ).compact(
            CompactionRequest(
                session=Session(
                    id="sess_prompt_cache_retry_after_completed",
                    agent_name="assistant",
                    provider_name=provider.name,
                    model="fake-model",
                    invocation=fixture_session_invocation(
                        "sess_prompt_cache_retry_after_completed"
                    ),
                ),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old request")],
                context_messages=context_messages,
                cache_prefix_request=ModelRequest(
                    model="fake-model",
                    messages=context_messages,
                ),
            )
        )
    )

    assert result.summary == "summary 2"
    assert len(provider.requests) == 2
    assert [
        payload["usage_metrics"]["input_tokens"] for payload in result.model_completed_payloads
    ] == [1, 2]
    assert all(
        "compaction_attempt_id" not in payload for payload in result.model_completed_payloads
    )


def test_runtime_compaction_outer_ledger_does_not_duplicate_retry_completions():
    compaction_provider = CompletedThenRetryableCompactionProvider()
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    budget_store = InMemoryBudgetStore()
    app = CayuApp(budget_store=budget_store)
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compaction_provider,
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
                session_id="sess_compaction_completed_retry",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    completions = [
        event
        for event in events
        if event.type == EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    ]
    assert len(compaction_provider.requests) == 2
    assert [event.payload["usage_metrics"]["input_tokens"] for event in completions] == [1, 2]
    assert all("compaction_attempt_id" not in event.payload for event in completions)
    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    session_records = asyncio.run(
        app.session_store.query_events(
            EventQuery(session_id="sess_compaction_completed_retry", limit=100)
        )
    )
    assert [
        event.id for event in budget_events if event.payload.get("purpose") == "context_compaction"
    ] == [
        _private_record_for_public_event(session_records, event).event.id for event in completions
    ]


def test_model_compactor_can_disable_retries_explicitly():
    provider = FlakyCompactionProvider(
        failures=1,
        error=ModelProviderError(
            "provider overloaded",
            provider="flaky",
            status_code=503,
            retryable=True,
        ),
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        retry_policy=RetryPolicy(max_attempts=1),
    )

    with pytest.raises(ModelProviderError, match="provider overloaded"):
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


def test_model_compactor_does_not_retry_non_retryable_provider_error_events():
    typed_error = ModelProviderError(
        "request timed out",
        provider="fake",
        status_code=503,
        retryable=False,
        retry_after_s=2.5,
    )
    provider = FakeProvider([[ModelStreamEvent.error("request timed out", cause=typed_error)]])
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        retry_policy=RetryPolicy(max_attempts=3, initial_delay_s=0.0),
    )

    with pytest.raises(ModelProviderError, match="request timed out") as exc_info:
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.retryable is False
    assert exc_info.value.retry_after_s == 2.5
    assert len(provider.requests) == 1


def test_model_compactor_keeps_provider_only_error_events_untyped():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent(
                    type=ModelStreamEventType.ERROR,
                    payload={"error": "provider unavailable", "provider": "fake"},
                )
            ],
            [
                ModelStreamEvent.text_delta("should not retry"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
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


def test_model_compactor_does_not_retry_terminal_provider_errors():
    provider = FlakyCompactionProvider(
        failures=1,
        error=ModelProviderError(
            "invalid request",
            provider="flaky",
            status_code=400,
            retryable=False,
        ),
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        retry_policy=RetryPolicy(max_attempts=3, initial_delay_s=0.0),
    )

    with pytest.raises(ModelProviderError, match="invalid request"):
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


def test_transcript_digest_compactor_preserves_previous_summary_when_clipping():
    # A large batch of newly compacted messages must not clip the accumulated
    # previous summary out of the front of the combined text.
    compactor = TranscriptDigestCompactor(max_summary_chars=400)
    previous_summary = "decision: ship plan A"

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[
                    Message.text("user", f"filler {index} " + "x" * 40) for index in range(30)
                ],
                existing_summary=previous_summary,
            )
        )
    )

    assert len(result.summary) <= 400
    assert f"Previous summary:\n{previous_summary}" in result.summary
    assert result.covered_message_count is not None
    assert 0 < result.covered_message_count < 30
    assert result.summary.endswith(f"filler {result.covered_message_count - 1} " + "x" * 40)


def test_transcript_digest_compactor_budgets_both_oversized_sections():
    compactor = TranscriptDigestCompactor(max_summary_chars=400)
    previous_summary = ("earlier decision " * 40).strip()

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "latest work " * 50)],
                existing_summary=previous_summary,
            )
        )
    )

    assert result.summary == previous_summary
    assert result.covered_message_count == 0
    assert result.progress_exhausted is True
    assert result.progress_key == compactor._progress_key()
    assert result.metadata["progress_reason"] == "existing_summary_exceeds_limit"


def test_transcript_digest_compactor_keeps_small_previous_summary_unclipped():
    compactor = TranscriptDigestCompactor()
    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[
                    Message.text("user", "old request"),
                    Message.text("assistant", "old answer"),
                ],
                existing_summary="previous context",
            )
        )
    )
    assert result.summary == (
        "Previous summary:\nprevious context\n\n"
        "Newly compacted transcript:\nuser: old request\nassistant: old answer"
    )
    assert compactor.max_summary_chars == 8000


def test_context_policy_receives_previous_actual_context_usage_on_next_call():
    class UsageAwarePolicy(ContextPolicy):
        def __init__(self) -> None:
            self.requests: list[ContextRequest] = []

        async def build(self, request: ContextRequest) -> list[Message]:
            self.requests.append(request)
            if (
                request.context_usage.last_input_tokens is not None
                and request.context_usage.last_input_tokens >= 50
            ):
                return [request.messages[-1]]
            return request.messages

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 100, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 20, "output_tokens": 2}}),
            ],
            [
                ModelStreamEvent.text_delta("third answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 1}}),
            ],
        ]
    )
    policy = UsageAwarePolicy()
    app = CayuApp()
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
                session_id="usage_context_policy",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_context_policy",
                messages=[Message.text("user", "second request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_context_policy",
                messages=[Message.text("user", "third request")],
            ),
        )
    )

    assert len(policy.requests) == 3
    assert policy.requests[0].context_usage.last_input_tokens is None
    assert policy.requests[1].context_usage.last_input_tokens == 100
    assert policy.requests[1].context_usage.last_output_tokens == 4
    assert policy.requests[1].context_usage.last_total_tokens == 104
    assert policy.requests[1].context_usage.last_provider_name == "fake"
    assert policy.requests[1].context_usage.last_requested_model == "fake-model"
    assert policy.requests[1].context_usage.last_model == "fake-model"
    assert policy.requests[2].context_usage.last_input_tokens == 20
    assert len(provider.requests[0].messages) == 1
    assert len(provider.requests[1].messages) == 1
    assert provider.requests[1].messages[0].content[0].text == "second request"
    assert len(provider.requests[2].messages) > 1
    assert provider.requests[2].messages[-1].content[0].text == "third request"


def test_observed_delta_context_estimator_estimates_only_messages_after_cursor():
    estimator = ObservedDeltaContextEstimator(chars_per_token=4, json_chars_per_token=2)
    usage = ContextUsageState(
        last_input_tokens=40,
        last_transcript_cursor=2,
        input_coverage=runtime_context_module.context_input_coverage(
            [
                Message.text("user", "already counted"),
                Message.text("assistant", "already counted answer"),
            ],
            transcript_cursor=2,
        ),
    )
    messages = [
        Message.text("user", "already counted"),
        Message.text("assistant", "already counted answer"),
        Message.text("user", "abcd" * 5),
        Message.tool_result(
            tool_call_id="call_1",
            tool_name="lookup",
            content="xy" * 5,
            structured={"ok": True},
        ),
    ]

    estimate = estimator.estimate(usage=usage, messages=messages)

    assert isinstance(estimate, ContextPressureEstimate)
    assert estimate.method == "observed_plus_estimated_delta"
    assert estimate.confidence == "estimated"
    assert estimate.observed_context_input_tokens == 40
    assert estimate.estimated_delta_input_tokens == 8
    assert estimate.estimated_context_input_tokens == 48
    assert estimate.reserved_output_tokens == 0
    assert estimate.estimated_context_window_tokens == 48
    assert estimate.anchor_transcript_cursor == 2
    assert estimate.current_transcript_cursor == 4
    assert estimate.estimated_message_count == 2


def test_observed_delta_context_estimator_can_estimate_full_request_overhead():
    estimator = ObservedDeltaContextEstimator(chars_per_token=4, json_chars_per_token=2)
    usage = ContextUsageState(last_input_tokens=40, last_transcript_cursor=2)
    messages = [Message.text("user", "abcd" * 5)]
    overhead = ContextPressureOverhead(
        tools=[
            {
                "name": "lookup",
                "description": "x" * 40,
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ],
        structured_output_instruction="return structured output",
        request_options={
            "step": 2,
            "structured_output": {"name": "answer"},
            "openai": {"tool_choice": "auto"},
        },
    )

    estimate = estimator.estimate_full_request(
        usage=usage,
        messages=messages,
        overhead=overhead,
    )

    assert estimate.method == "local_full_request_estimate"
    assert estimate.observed_context_input_tokens == 40
    assert estimate.estimated_message_input_tokens == 5
    assert estimate.estimated_tool_schema_input_tokens > 0
    assert estimate.estimated_structured_output_input_tokens > 0
    assert estimate.estimated_request_options_input_tokens > 0
    assert estimate.estimated_context_input_tokens == (
        estimate.estimated_message_input_tokens
        + estimate.estimated_tool_schema_input_tokens
        + estimate.estimated_structured_output_input_tokens
        + estimate.estimated_request_options_input_tokens
    )
    assert estimate.estimated_context_window_tokens == estimate.estimated_context_input_tokens


def test_observed_delta_context_estimator_counts_only_overhead_delta_after_anchor():
    estimator = ObservedDeltaContextEstimator(chars_per_token=4, json_chars_per_token=2)
    overhead = ContextPressureOverhead(
        tools=[
            {
                "name": "lookup",
                "description": "x" * 80,
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ],
        request_options={"openai": {"tool_choice": "auto"}},
    )
    overhead_tokens = estimator.estimate_full_request(
        usage=ContextUsageState(),
        messages=[Message.text("user", "seed")],
        overhead=overhead,
    ).estimated_request_overhead_input_tokens
    usage = ContextUsageState(
        last_input_tokens=100,
        last_transcript_cursor=1,
        input_coverage=runtime_context_module.context_input_coverage(
            [Message.text("user", "already counted")],
            transcript_cursor=1,
        ),
        last_context_overhead_input_tokens=overhead_tokens,
    )
    messages = [
        Message.text("user", "already counted"),
        Message.text("user", "abcd" * 5),
    ]

    stable = estimator.estimate_anchored_request(
        usage=usage,
        messages=messages,
        overhead=overhead,
    )
    increased = estimator.estimate_anchored_request(
        usage=usage.model_copy(update={"last_context_overhead_input_tokens": overhead_tokens - 5}),
        messages=messages,
        overhead=overhead,
    )
    decreased = estimator.estimate_anchored_request(
        usage=usage.model_copy(update={"last_context_overhead_input_tokens": overhead_tokens + 5}),
        messages=messages,
        overhead=overhead,
    )

    assert stable.estimated_message_input_tokens == 5
    assert stable.estimated_request_overhead_input_tokens == overhead_tokens
    assert stable.previous_request_overhead_input_tokens == overhead_tokens
    assert stable.estimated_request_overhead_delta_tokens == 0
    assert stable.estimated_delta_input_tokens == 5
    assert stable.estimated_context_input_tokens == 105

    assert increased.estimated_request_overhead_delta_tokens == 5
    assert increased.estimated_context_input_tokens == 110
    assert decreased.estimated_request_overhead_delta_tokens == -5
    assert decreased.estimated_context_input_tokens == 100


def test_observed_delta_context_estimator_uses_provider_specific_image_floor():
    estimator = ObservedDeltaContextEstimator(chars_per_token=5)
    from cayu.messages import FilePart

    user_image_message = Message(
        role="user",
        content=[
            TextPart(text="look"),
            FilePart(
                attachment=file_attachment(
                    artifact_id="img_user",
                    kind="image",
                    filename="user.png",
                    content_type="image/png",
                    size_bytes=68,
                )
            ),
        ],
    )
    image_message = Message.tool_result(
        tool_call_id="call_image",
        tool_name="image_tool",
        content="image result",
        artifacts=[
            file_attachment(
                artifact_id="img_1",
                kind="image",
                filename="pixel.png",
                content_type="image/png",
                size_bytes=68,
            )
        ],
    )
    document_message = Message.tool_result(
        tool_call_id="call_document",
        tool_name="document_tool",
        content="document result",
        artifacts=[
            file_attachment(
                artifact_id="doc_1",
                kind="document",
                filename="tiny.pdf",
                content_type="application/pdf",
                size_bytes=120,
            )
        ],
    )

    generic_tokens = estimator.estimate_full_request(
        usage=ContextUsageState(),
        messages=[user_image_message, image_message, document_message],
        overhead=ContextPressureOverhead(image_min_tokens=32),
    )
    anthropic_tokens = estimator.estimate_full_request(
        usage=ContextUsageState(),
        messages=[user_image_message, image_message, document_message],
        overhead=ContextPressureOverhead(
            image_min_tokens=100,
            document_min_tokens=1800,
        ),
    )

    assert generic_tokens.estimated_attachment_input_tokens < 200
    assert anthropic_tokens.estimated_attachment_input_tokens >= 2000
    assert (
        anthropic_tokens.estimated_attachment_input_tokens
        > generic_tokens.estimated_attachment_input_tokens
    )


def test_observed_delta_context_estimator_uses_adaptive_text_density():
    prose = (
        "Remote sandbox Git authentication should use a brokered Git HTTP proxy. "
        "Credentials stay outside the sandbox boundary. " * 100
    )
    dense_json = (
        '{"path":"src/cayu/context/base.py","line":123,"value":"x_y-z.abc/def","ok":true}\n' * 100
    )
    pretty_json = (
        '{\n  "path": "src/cayu/context/base.py",\n  "line": 123,\n'
        '  "value": "x_y-z.abc/def",\n  "ok": true\n}\n' * 100
    )
    logs = (
        "2026-07-02T12:34:56Z INFO session=sess_123 step=4 "
        "tool=search_knowledge latency_ms=42 status=ok input_tokens=14947\n" * 100
    )
    csv = (
        "timestamp,session_id,agent,model,input_tokens,output_tokens,status\n"
        "2026-07-02T12:34:56Z,sess_123,assistant,gpt-5.5,14947,7,ok\n" * 100
    )
    code = (
        "def estimate_context_pressure(request):\n"
        "    if request.count_input_tokens is None:\n"
        "        return estimate\n"
        "    return request.count_input_tokens(messages)\n" * 100
    )
    markdown = (
        "# Heading\n\n- `code/path.py`: explanation with punctuation, numbers 12345, "
        'and JSON {"ok": true}.\n' * 100
    )
    estimator = ObservedDeltaContextEstimator(chars_per_token=5)

    prose_tokens = estimator.estimate_message_tokens(Message.text("user", prose))
    dense_json_tokens = estimator.estimate_message_tokens(Message.text("user", dense_json))
    pretty_json_tokens = estimator.estimate_message_tokens(Message.text("user", pretty_json))
    logs_tokens = estimator.estimate_message_tokens(Message.text("user", logs))
    csv_tokens = estimator.estimate_message_tokens(Message.text("user", csv))
    code_tokens = estimator.estimate_message_tokens(Message.text("user", code))
    markdown_tokens = estimator.estimate_message_tokens(Message.text("user", markdown))

    assert prose_tokens == math.ceil(len(prose) / 5)
    assert dense_json_tokens == math.ceil(len(dense_json) / 3)
    assert pretty_json_tokens == math.ceil(len(pretty_json) / 3)
    assert logs_tokens == math.ceil(len(logs) / 2.75)
    assert csv_tokens == math.ceil(len(csv) / 2.5)
    assert code_tokens == math.ceil(len(code) / 5)
    assert markdown_tokens == math.ceil(len(markdown) / 3.75)


def test_observed_delta_context_estimator_uses_tool_schema_density_override():
    estimator = ObservedDeltaContextEstimator(chars_per_token=5, json_chars_per_token=5)
    tools = [
        {
            "name": "heavy",
            "description": "Search many records. " * 200,
            "input_schema": {
                "type": "object",
                "properties": {
                    f"field_{index}": {
                        "type": "string",
                        "description": "Filter value " * 20,
                    }
                    for index in range(20)
                },
            },
        }
    ]

    default_tokens = estimator.estimate_tool_schema_tokens(tools)
    openai_profile_tokens = estimator.estimate_tool_schema_tokens(
        tools,
        chars_per_token=6,
    )

    assert default_tokens > openai_profile_tokens
    assert openai_profile_tokens == math.ceil(
        len(json.dumps(tools[0], ensure_ascii=False, separators=(",", ":"), sort_keys=True)) / 6
    )


def test_context_policy_receives_estimated_context_pressure_on_next_call():
    class PressureAwarePolicy(ContextPolicy):
        def __init__(self) -> None:
            self.requests: list[ContextRequest] = []

        async def build(self, request: ContextRequest) -> list[Message]:
            self.requests.append(request)
            return request.messages

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 40, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
        ]
    )
    policy = PressureAwarePolicy()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_pressure_estimate",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="context_pressure_estimate",
                messages=[Message.text("user", "abcd" * 4)],
            ),
        )
    )

    first_completed = next(
        event for event in first_events if event.type == EventType.MODEL_COMPLETED
    )
    assert first_completed.payload["transcript_cursor"] == 2
    assert len(policy.requests) == 2
    pressure = policy.requests[1].context_usage.input_pressure
    assert pressure is not None
    assert pressure.observed_context_input_tokens == 40
    assert pressure.estimated_delta_input_tokens == 7
    assert pressure.estimated_context_input_tokens == 47
    assert pressure.estimated_context_window_tokens == 47
    assert pressure.anchor_transcript_cursor == 1
    assert pressure.current_transcript_cursor == 3
    assert pressure.estimated_message_count == 2


def test_context_policy_input_pressure_uses_provider_profile_image_floor():
    class ImageFloorProvider(FakeProvider):
        @property
        def context_pressure_profile(self) -> ModelContextPressureProfile:
            return ModelContextPressureProfile(
                image_min_tokens=100,
                document_min_tokens=1800,
            )

    class PressureAwarePolicy(ContextPolicy):
        def __init__(self) -> None:
            self.requests: list[ContextRequest] = []

        async def build(self, request: ContextRequest) -> list[Message]:
            self.requests.append(request)
            return request.messages

    provider = ImageFloorProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 40, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 50, "output_tokens": 2}}),
            ],
        ]
    )
    policy = PressureAwarePolicy()
    app = CayuApp()
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
                session_id="context_pressure_provider_profile",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="context_pressure_provider_profile",
                messages=[
                    Message.tool_result(
                        tool_call_id="call_image",
                        tool_name="image_tool",
                        content="image result",
                        artifacts=[
                            file_attachment(
                                artifact_id="img_1",
                                kind="image",
                                filename="pixel.png",
                                content_type="image/png",
                                size_bytes=68,
                            )
                        ],
                    ),
                    Message.tool_result(
                        tool_call_id="call_document",
                        tool_name="document_tool",
                        content="document result",
                        artifacts=[
                            file_attachment(
                                artifact_id="doc_1",
                                kind="document",
                                filename="tiny.pdf",
                                content_type="application/pdf",
                                size_bytes=120,
                            )
                        ],
                    ),
                ],
            ),
        )
    )

    assert len(policy.requests) == 2
    pressure = policy.requests[1].context_usage.input_pressure
    assert pressure is not None
    assert pressure.estimated_attachment_input_tokens >= 1900


def test_usage_triggered_context_policy_can_trigger_on_estimated_context_pressure():
    class LargeSchemaTool(Tool):
        spec = ToolSpec(
            name="large_schema",
            description="x" * 200,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "y" * 200,
                    }
                },
            },
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            return ToolResult(content="")

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 40, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=45,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[LargeSchemaTool()],
        context_policy=policy,
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_estimated_pressure",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_estimated_pressure",
                messages=[Message.text("user", "abcd" * 5)],
            ),
        )
    )

    assert len(provider.requests[0].messages) == 1
    assert len(provider.requests[1].messages) == 1
    assert provider.requests[1].messages[0].content[0].text == "abcd" * 5
    checkpoint_events = [
        event for event in first_events if event.type == EventType.SESSION_CHECKPOINTED
    ]
    assert len(checkpoint_events) == 1
    payload = checkpoint_events[0].payload
    assert payload["checkpoint"] == "usage_triggered_context"
    assert payload["min_input_tokens"] is None
    assert payload["trigger_estimated_context_tokens"] == 45
    assert payload["reserved_output_tokens"] == 0
    assert payload["min_total_tokens"] is None
    assert payload["last_input_tokens"] is None
    assert payload["last_total_tokens"] is None
    assert payload["last_transcript_cursor"] is None
    assert payload["estimated_context_input_tokens"] >= 45
    assert payload["estimated_context_window_tokens"] >= 45
    assert payload["estimated_delta_input_tokens"] == payload["estimated_context_input_tokens"]
    assert not [event for event in second_events if event.type == EventType.SESSION_CHECKPOINTED]


def test_usage_triggered_context_policy_does_not_double_count_stable_overhead():
    class LargeSchemaTool(Tool):
        spec = ToolSpec(
            name="large_schema",
            description="x" * 1000,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "y" * 1000,
                    }
                },
            },
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            return ToolResult(content="")

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 1000, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 1010, "output_tokens": 2}}),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=1250,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[LargeSchemaTool()],
        context_policy=policy,
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_stable_overhead",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_stable_overhead",
                messages=[Message.text("user", "abcd" * 5)],
            ),
        )
    )

    first_completed = next(
        event for event in first_events if event.type == EventType.MODEL_COMPLETED
    )
    assert (
        first_completed.payload["context_pressure"]["estimated_request_overhead_input_tokens"] > 300
    )
    assert len(provider.requests[1].messages) == 3
    assert not [event for event in second_events if event.type == EventType.SESSION_CHECKPOINTED]


def test_usage_triggered_context_policy_full_estimates_same_length_projection():
    class ReplacingPolicy(ContextPolicy):
        async def build(self, request: ContextRequest) -> list[Message]:
            return [Message.text("user", "x" * 1000) for _ in request.messages]

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 100, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 1010, "output_tokens": 2}}),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=ReplacingPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=450,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_same_length_projection",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_same_length_projection",
                messages=[Message.text("user", "second request")],
            ),
        )
    )

    assert not [event for event in first_events if event.type == EventType.SESSION_CHECKPOINTED]
    assert len(provider.requests[0].messages) == 1
    assert provider.requests[0].messages[0].content[0].text == "x" * 1000
    assert len(provider.requests[1].messages) == 1
    assert provider.requests[1].messages[0].content[0].text == "second request"
    checkpoint_event = next(
        event for event in second_events if event.type == EventType.SESSION_CHECKPOINTED
    )
    assert checkpoint_event.payload["estimated_context_input_tokens"] >= 300


def test_usage_triggered_context_policy_estimates_base_policy_model_facing_context():
    class InjectingPolicy(ContextPolicy):
        async def build(self, request: ContextRequest) -> list[Message]:
            return [
                *request.messages,
                Message.text("user", "injected context " + ("x" * 400)),
            ]

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 40, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=InjectingPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=80,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_estimated_policy_context",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_estimated_policy_context",
                messages=[Message.text("user", "second request")],
            ),
        )
    )

    assert len(provider.requests[0].messages) == 1
    assert len(provider.requests[1].messages) == 1
    assert provider.requests[1].messages[0].content[0].text == "second request"
    checkpoint_event = next(
        event for event in first_events if event.type == EventType.SESSION_CHECKPOINTED
    )
    assert checkpoint_event.payload["estimated_context_input_tokens"] >= 80
    assert not [event for event in second_events if event.type == EventType.SESSION_CHECKPOINTED]


def test_usage_triggered_context_policy_includes_reserved_output_in_estimated_trigger():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 20, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=94,
        reserved_output_tokens=90,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_reserved_output",
                messages=[Message.text("user", "abcd" * 5)],
            ),
        )
    )

    assert len(provider.requests[0].messages) == 1
    checkpoint_event = next(
        event for event in first_events if event.type == EventType.SESSION_CHECKPOINTED
    )
    assert checkpoint_event.payload["trigger_estimated_context_tokens"] == 94
    assert checkpoint_event.payload["reserved_output_tokens"] == 90
    assert checkpoint_event.payload["estimated_context_input_tokens"] < 100
    assert checkpoint_event.payload["estimated_context_window_tokens"] >= 94


def test_usage_triggered_context_policy_provider_count_can_prevent_estimated_false_positive():
    class LargeSchemaTool(Tool):
        spec = ToolSpec(
            name="large_schema",
            description="x" * 400,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "y" * 400,
                    }
                },
            },
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            return ToolResult(content="")

    provider = CountingProvider(
        [
            ModelStreamEvent.text_delta("answer"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
        ],
        count_result=InputTokenCountResult(
            input_tokens=10,
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
        ),
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=45,
        verify_estimate_with_provider_count=True,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[LargeSchemaTool()],
        context_policy=policy,
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_provider_count_false_positive",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    assert len(provider.count_requests) == 1
    assert provider.count_requests[0].tools
    assert not [event for event in events if event.type == EventType.SESSION_CHECKPOINTED]


def test_usage_triggered_context_policy_provider_count_is_not_called_when_far_from_threshold():
    provider = CountingProvider(
        [
            ModelStreamEvent.text_delta("answer"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
        ],
        count_result=InputTokenCountResult(
            input_tokens=10,
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
        ),
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=10_000,
        verify_estimate_with_provider_count=True,
    )
    app = CayuApp()
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
                session_id="usage_triggered_provider_count_far",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    assert provider.count_requests == []


def test_usage_triggered_context_policy_large_delta_can_request_provider_count():
    provider = CountingProvider(
        [
            ModelStreamEvent.text_delta("answer"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
        ],
        count_result=InputTokenCountResult(
            input_tokens=12_000,
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
        ),
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=10_000,
        verify_estimate_with_provider_count=True,
        provider_count_min_delta_tokens=20,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_provider_count_large_delta",
                messages=[Message.text("user", "x" * 200)],
            ),
        )
    )

    assert len(provider.count_requests) == 1
    checkpoint_event = next(
        event for event in events if event.type == EventType.SESSION_CHECKPOINTED
    )
    assert checkpoint_event.payload["provider_count_input_tokens"] == 12_000
    assert checkpoint_event.payload["provider_count_context_window_tokens"] == 12_000


def test_usage_triggered_context_policy_stays_triggered_after_previous_actual_usage():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 100, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 20, "output_tokens": 2}}),
            ],
            [
                ModelStreamEvent.text_delta("third answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 1}}),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        min_input_tokens=50,
    )
    app = CayuApp()
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
                session_id="usage_triggered_policy",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_policy",
                messages=[Message.text("user", "second request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_policy",
                messages=[Message.text("user", "third request")],
            ),
        )
    )

    assert len(provider.requests[0].messages) == 1
    assert len(provider.requests[1].messages) == 1
    assert provider.requests[1].messages[0].content[0].text == "second request"
    assert len(provider.requests[2].messages) == 1
    assert provider.requests[2].messages[0].content[0].text == "third request"
    checkpoint_events = [
        event for event in second_events if event.type == EventType.SESSION_CHECKPOINTED
    ]
    assert len(checkpoint_events) == 1
    second_model_started = next(
        event for event in second_events if event.type == EventType.MODEL_STARTED
    )
    assert checkpoint_events[0].payload == {
        "checkpoint": "usage_triggered_context",
        "execution_profile_fingerprint": second_model_started.payload[
            "execution_profile_fingerprint"
        ],
        "min_input_tokens": 50,
        "min_total_tokens": None,
        "last_input_tokens": 100,
        "last_total_tokens": 104,
        "model_step_id": second_model_started.payload["model_step_id"],
    }
    checkpoint = asyncio.run(app.session_store.load_checkpoint("usage_triggered_policy"))
    assert checkpoint_without_model_step_publication(checkpoint) == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "usage_triggered_context": {
            "version": 1,
            "min_input_tokens": 50,
            "min_total_tokens": None,
            "last_input_tokens": 100,
            "last_total_tokens": 104,
        },
    }


def test_usage_triggered_context_policy_preserves_marker_with_checkpoint_policy():
    class ReplacingCheckpointPolicy(RuntimeManagedContextPolicy):
        def __init__(self) -> None:
            self.calls = 0

        async def build_with_checkpoint(
            self,
            request: ContextRequest,
            *,
            checkpoint: dict[str, Any] | None,
        ) -> ContextBuildResult:
            self.calls += 1
            return ContextBuildResult(
                messages=[request.messages[-1]],
                checkpoint={"triggered_policy": {"calls": self.calls}},
                checkpoint_event_payload={
                    "checkpoint": "triggered_policy",
                    "calls": self.calls,
                },
            )

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 100, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 20, "output_tokens": 2}}),
            ],
            [
                ModelStreamEvent.text_delta("third answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 1}}),
            ],
        ]
    )
    triggered_policy = ReplacingCheckpointPolicy()
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=triggered_policy,
        min_input_tokens=50,
    )
    app = CayuApp()
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
                session_id="usage_triggered_policy_checkpoint_merge",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_policy_checkpoint_merge",
                messages=[Message.text("user", "second request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_policy_checkpoint_merge",
                messages=[Message.text("user", "third request")],
            ),
        )
    )

    assert triggered_policy.calls == 2
    assert len(provider.requests[1].messages) == 1
    assert len(provider.requests[2].messages) == 1
    checkpoint = asyncio.run(
        app.session_store.load_checkpoint("usage_triggered_policy_checkpoint_merge")
    )
    assert checkpoint_without_model_step_publication(checkpoint) == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "triggered_policy": {"calls": 2},
        "usage_triggered_context": {
            "version": 1,
            "min_input_tokens": 50,
            "min_total_tokens": None,
            "last_input_tokens": 100,
            "last_total_tokens": 104,
        },
    }


def test_usage_triggered_context_policy_can_be_last_call_only():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 100, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 20, "output_tokens": 2}}),
            ],
            [
                ModelStreamEvent.text_delta("third answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 1}}),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        min_input_tokens=50,
        sticky=False,
    )
    app = CayuApp()
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
                session_id="usage_triggered_policy_last_call_only",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_policy_last_call_only",
                messages=[Message.text("user", "second request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_policy_last_call_only",
                messages=[Message.text("user", "third request")],
            ),
        )
    )

    assert len(provider.requests[0].messages) == 1
    assert len(provider.requests[1].messages) == 1
    assert provider.requests[1].messages[0].content[0].text == "second request"
    assert len(provider.requests[2].messages) > 1
    assert provider.requests[2].messages[-1].content[0].text == "third request"
    checkpoint = asyncio.run(
        app.session_store.load_checkpoint("usage_triggered_policy_last_call_only")
    )
    assert_only_model_step_publication_checkpoint(checkpoint)


def test_usage_triggered_context_policy_supports_total_token_threshold():
    policy = UsageTriggeredContextPolicy(
        base_policy=MessageWindowContextPolicy(max_messages=3),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        min_total_tokens=100,
    )
    messages = [
        Message.text("system", "You are careful."),
        Message.text("user", "old"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current"),
    ]

    async def build_context() -> list[Message]:
        return await policy.build(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=2,
                context_usage=ContextUsageState(last_total_tokens=100),
            )
        )

    context = asyncio.run(build_context())

    assert [message.role for message in context] == ["system", "user"]
    assert [message.content[0].text for message in context] == ["You are careful.", "current"]


def test_usage_triggered_context_policy_requires_a_threshold():
    with pytest.raises(ValueError, match="At least one usage threshold"):
        UsageTriggeredContextPolicy(
            triggered_policy=MessageWindowContextPolicy(max_messages=1),
        )


def test_usage_triggered_context_policy_rejects_runtime_managed_estimated_base_policy():
    with pytest.raises(ValueError, match="side-effect-free base_policy"):
        UsageTriggeredContextPolicy(
            base_policy=CheckpointCompactionContextPolicy(),
            triggered_policy=MessageWindowContextPolicy(max_messages=1),
            trigger_estimated_context_tokens=100,
        )


def test_usage_triggered_context_policy_requires_bool_sticky():
    with pytest.raises(TypeError, match="sticky must be a bool"):
        UsageTriggeredContextPolicy(
            triggered_policy=MessageWindowContextPolicy(max_messages=1),
            min_input_tokens=1,
            sticky=1,
        )


def test_context_overflow_policy_rebuilds_context_and_retries_once():
    provider = ContextOverflowProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=RecentTurnsContextPolicy(max_user_turns=1),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_overflow_recovery",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
            ),
        )
    )

    assert [request.messages[0].content[0].text for request in provider.requests] == [
        "old request",
        "new request",
    ]
    assert len(provider.requests[0].messages) == 2
    assert len(provider.requests[1].messages) == 1
    assert [
        event.type
        for event in events
        if event.type
        in {
            EventType.CONTEXT_OVERFLOW_DETECTED,
            EventType.CONTEXT_OVERFLOW_RECOVERING,
            EventType.CONTEXT_OVERFLOW_FAILED,
            EventType.SESSION_COMPLETED,
        }
    ] == [
        EventType.CONTEXT_OVERFLOW_DETECTED,
        EventType.CONTEXT_OVERFLOW_RECOVERING,
        EventType.SESSION_COMPLETED,
    ]
    first_started = next(event for event in events if event.type == EventType.MODEL_STARTED)
    model_error = next(event for event in events if event.type == EventType.MODEL_ERROR)
    assert model_error.payload == {
        "error": "context too large",
        "error_type": "ModelContextOverflowError",
        "provider_name": "overflow",
        "requested_model": "fake-model",
        "stage": "provider_dispatch",
        "context_overflow": True,
        "provider": "overflow",
        "status_code": 400,
        "provider_error_code": "context_length_exceeded",
        "retryable": False,
        "step": 1,
        "attempt": 1,
        "max_attempts": 5,
        "model_step_id": first_started.payload["model_step_id"],
        "model_attempt_id": first_started.payload["model_attempt_id"],
        "execution_profile_fingerprint": first_started.payload["execution_profile_fingerprint"],
    }
    detected = next(event for event in events if event.type == EventType.CONTEXT_OVERFLOW_DETECTED)
    assert detected.payload == {
        "step": 1,
        "phase": "initial",
        "error": "context too large",
        "error_type": "ModelContextOverflowError",
        "provider": "overflow",
        "original_message_count": 2,
        "status_code": 400,
        "provider_error_code": "context_length_exceeded",
        "model_step_id": first_started.payload["model_step_id"],
        "model_attempt_id": first_started.payload["model_attempt_id"],
        "execution_profile_fingerprint": first_started.payload["execution_profile_fingerprint"],
    }
    recovering = next(
        event for event in events if event.type == EventType.CONTEXT_OVERFLOW_RECOVERING
    )
    assert recovering.payload == {
        "step": 1,
        "original_message_count": 2,
        "recovery_message_count": 1,
        "policy": "RecentTurnsContextPolicy",
        "model_step_id": first_started.payload["model_step_id"],
        "model_attempt_id": first_started.payload["model_attempt_id"],
        "execution_profile_fingerprint": first_started.payload["execution_profile_fingerprint"],
    }
    transcript = asyncio.run(app.session_store.load_transcript("context_overflow_recovery"))
    assert [message.content[0].text for message in transcript if message.role == "user"] == [
        "old request",
        "new request",
    ]


def test_context_overflow_recovery_from_error_stream_event():
    provider = EventFlattenedOverflowProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=RecentTurnsContextPolicy(max_user_turns=1),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_overflow_event_recovery",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
            ),
        )
    )

    assert len(provider.requests) == 2
    assert len(provider.requests[1].messages) == 1
    # The typed overflow is rehydrated before the generic retry path. It still
    # receives one terminal model.error so durable budget evidence can join the
    # dispatched attempt exactly.
    assert [event.type for event in events].count(EventType.MODEL_ERROR) == 1
    assert [
        event.type
        for event in events
        if event.type
        in {
            EventType.CONTEXT_OVERFLOW_DETECTED,
            EventType.CONTEXT_OVERFLOW_RECOVERING,
            EventType.CONTEXT_OVERFLOW_FAILED,
            EventType.SESSION_COMPLETED,
        }
    ] == [
        EventType.CONTEXT_OVERFLOW_DETECTED,
        EventType.CONTEXT_OVERFLOW_RECOVERING,
        EventType.SESSION_COMPLETED,
    ]
    first_started = next(event for event in events if event.type == EventType.MODEL_STARTED)
    model_error = next(event for event in events if event.type == EventType.MODEL_ERROR)
    assert {
        "model_step_id": model_error.payload["model_step_id"],
        "model_attempt_id": model_error.payload["model_attempt_id"],
    } == {
        "model_step_id": first_started.payload["model_step_id"],
        "model_attempt_id": first_started.payload["model_attempt_id"],
    }
    assert model_error.payload["stage"] == "provider_dispatch"
    assert model_error.payload["context_overflow"] is True
    assert model_error.payload["retryable"] is False
    detected = next(event for event in events if event.type == EventType.CONTEXT_OVERFLOW_DETECTED)
    assert detected.payload == {
        "step": 1,
        "phase": "initial",
        "error": "context too large",
        "error_type": "ModelContextOverflowError",
        "provider": "overflow-event",
        "original_message_count": 2,
        "status_code": 400,
        "provider_error_code": "context_length_exceeded",
        "model_step_id": first_started.payload["model_step_id"],
        "model_attempt_id": first_started.payload["model_attempt_id"],
        "execution_profile_fingerprint": first_started.payload["execution_profile_fingerprint"],
    }


def test_context_overflow_error_stream_event_without_policy_fails_typed():
    provider = EventFlattenedOverflowProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_overflow_event_no_policy",
                messages=[Message.text("user", "too much")],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert [event.type for event in events].count(EventType.MODEL_ERROR) == 1
    failed = events[-1]
    assert failed.type == EventType.SESSION_FAILED
    assert failed.payload["error"] == "context too large"
    assert failed.payload["error_type"] == "ModelContextOverflowError"


def test_context_overflow_policy_can_checkpoint_compaction_before_retry():
    provider = ContextOverflowProvider(
        success_events=[
            ModelStreamEvent.text_delta("recovered"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}}),
        ]
    )
    compactor = RecordingCompactor()
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("10"),
                    pricing=PriceBook(
                        prices=(
                            ModelPrice.fixed(
                                provider_name="overflow",
                                model="fake-model",
                                input_per_million=Decimal("1"),
                                output_per_million=Decimal("10"),
                            ),
                        )
                    ),
                ),
            )
        )
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_overflow_compaction_recovery",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
            ),
        )
    )

    assert len(compactor.requests) == 1
    assert [message.content[0].text for message in compactor.requests[0].messages] == [
        "old request"
    ]
    assert len(provider.requests) == 2
    assert len(provider.requests[1].messages) == 2
    assert (
        provider.requests[1]
        .messages[0]
        .content[0]
        .text.startswith("Previous session context summary:")
    )
    assert provider.requests[1].messages[1].content[0].text == "new request"
    assert EventType.SESSION_CHECKPOINTED in [event.type for event in events]
    # Deterministic compaction has no provider spend, so it must not add a
    # third budget boundary between the initial check and post-model check.
    assert [event.type for event in events].count(EventType.BUDGET_CHECKED) == 2
    checkpoint = asyncio.run(
        app.session_store.load_checkpoint("context_overflow_compaction_recovery")
    )
    assert checkpoint_without_model_step_publication(checkpoint) == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "context_compaction": {
            "version": 2,
            "summary": "old request",
            "compacted_transcript_cursor": 1,
            "metadata": {"request_count": 1},
        },
    }


@pytest.mark.parametrize("scope", ["run", "session"])
def test_context_overflow_model_compaction_stops_before_recovery_at_token_limit(
    scope: Literal["run", "session"],
) -> None:
    store = InMemorySessionStore()
    runtime_provider = ContextOverflowProvider()
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 16, "output_tokens": 4},
                }
            ),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
            ),
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )
    session_id = f"context_overflow_compaction_{scope}_token_limit"

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
                limits=RunLimits(max_total_tokens=15, scope=scope),
            ),
        )
    )

    assert len(runtime_provider.requests) == 1
    assert len(compactor_provider.requests) == 1
    assert EventType.CONTEXT_OVERFLOW_RECOVERING not in [event.type for event in events]
    assert [
        event.type
        for event in events
        if event.type
        in {
            EventType.CONTEXT_OVERFLOW_DETECTED,
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.CONTEXT_COMPACTION_COMPLETED,
            EventType.SESSION_CHECKPOINTED,
            EventType.SESSION_LIMIT_REACHED,
            EventType.TURN_COMPLETED,
            EventType.SESSION_INTERRUPTED,
        }
    ] == [
        EventType.CONTEXT_OVERFLOW_DETECTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    limit_event = next(event for event in events if event.type == EventType.SESSION_LIMIT_REACHED)
    assert limit_event.payload["limit"] == "total_tokens"
    assert limit_event.payload["actual"] == 20
    assert limit_event.payload["maximum"] == 15
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint is not None
    assert checkpoint["context_compaction"]["summary"] == "model summary"
    usage = asyncio.run(app.get_session_usage(session_id))
    assert usage.model_steps == 1
    assert usage.usage.total_tokens == 20


def test_context_overflow_hierarchy_enforces_run_limit_before_every_dispatch() -> None:
    store = InMemorySessionStore()
    runtime_provider = ContextOverflowProvider()
    compactor_provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("short summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "summary-model",
                        "usage": {
                            "input_tokens": 7,
                            "output_tokens": 3,
                            "total_tokens": 10,
                        },
                    }
                ),
            ]
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
                max_input_chars=1000,
            ),
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_overflow_hierarchy_run_limit",
                messages=[
                    Message.text("user", "oversized " + "x" * 5000),
                    Message.text("user", "current"),
                ],
                limits=RunLimits(max_total_tokens=10),
            ),
        )
    )

    assert len(runtime_provider.requests) == 1
    assert len(compactor_provider.requests) == 1
    event_types = [event.type for event in events]
    assert EventType.CONTEXT_OVERFLOW_RECOVERING not in event_types
    assert event_types.index(EventType.CONTEXT_COMPACTION_FAILED) < event_types.index(
        EventType.SESSION_LIMIT_REACHED
    )
    assert events[-1].type == EventType.SESSION_INTERRUPTED
    checkpoint = asyncio.run(store.load_checkpoint("context_overflow_hierarchy_run_limit"))
    assert checkpoint is None or "context_compaction" not in checkpoint


def test_context_overflow_model_compaction_settles_dispatch_before_budget_gate() -> None:
    store = InMemorySessionStore()
    runtime_provider = ContextOverflowProvider()
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 20, "output_tokens": 0},
                }
            ),
        ]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="overflow",
                model="fake-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("10"),
            ),
            ModelPrice.fixed(
                provider_name="fake",
                model="summary-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("10"),
            ),
        )
    )
    app = CayuApp(
        session_store=store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("0.00002"),
                    pricing=pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=10,
                        max_output_tokens=0,
                    ),
                ),
            )
        ),
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
            ),
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_overflow_compaction_budget_limit",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
            ),
        )
    )

    event_types = [event.type for event in events]
    assert len(runtime_provider.requests) == 1
    assert len(compactor_provider.requests) == 1
    assert event_types.count(EventType.BUDGET_RESERVED) == 2
    assert EventType.CONTEXT_OVERFLOW_RECOVERING not in event_types
    checkpoint_index = event_types.index(EventType.SESSION_CHECKPOINTED)
    reconciliations = [
        (index, event)
        for index, event in enumerate(events)
        if event.type == EventType.BUDGET_RECONCILED
    ]
    compaction_reconciliation_index, compaction_reconciliation = reconciliations[0]
    dispatch_reconciliation_index, dispatch_reconciliation = reconciliations[1]
    reached_index = event_types.index(EventType.BUDGET_LIMIT_REACHED)
    assert compaction_reconciliation_index < checkpoint_index
    assert checkpoint_index < dispatch_reconciliation_index < reached_index
    assert Decimal(compaction_reconciliation.payload["actual_amount"]) == Decimal("0.00002")
    assert compaction_reconciliation.payload["reason"] == (
        "automatic context compaction model completed"
    )
    assert Decimal(dispatch_reconciliation.payload["actual_amount"]) == Decimal("0.00001")
    assert (
        dispatch_reconciliation.payload["reason"]
        == "provider usage unknown after dispatch; charged reserved amount"
    )
    reached = events[reached_index]
    assert Decimal(reached.payload["actual"]) == Decimal("0.00002")
    assert reached.payload["scope"] == "app"
    assert events[-1].type == EventType.SESSION_INTERRUPTED
    cost = asyncio.run(app.get_session_cost("context_overflow_compaction_budget_limit", pricing))
    assert cost.model_steps == 1
    assert cost.total_cost == Decimal("0.00002")


def test_context_overflow_compaction_reservation_denial_settles_before_terminal() -> None:
    runtime_provider = ContextOverflowProvider()
    compactor_provider = FakeProvider([ModelStreamEvent.completed({})])
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="overflow",
                model="fake-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("10"),
            ),
            ModelPrice.fixed(
                provider_name="fake",
                model="summary-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("10"),
            ),
        )
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("0.000019"),
                    pricing=pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=10,
                        max_output_tokens=0,
                    ),
                ),
            )
        )
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
            ),
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_overflow_compaction_reservation_denied",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
            ),
        )
    )

    event_types = [event.type for event in events]
    assert len(runtime_provider.requests) == 1
    assert compactor_provider.requests == []
    assert event_types.index(EventType.BUDGET_RECONCILED) < event_types.index(
        EventType.BUDGET_LIMIT_REACHED
    )
    assert events[-1].type == EventType.SESSION_INTERRUPTED


def test_prompt_cache_compactor_bounds_context_overflow_recovery() -> None:
    provider = MessageCountOverflowProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=CheckpointCompactionContextPolicy(
            compactor=PromptCacheCompactor(provider=provider),
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="prompt_cache_overflow_recovery",
                messages=[Message.text("user", "old request")],
            ),
        )
    )
    assert first_events[-1].type == EventType.SESSION_COMPLETED

    recovery_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="prompt_cache_overflow_recovery",
                messages=[Message.text("user", "new request")],
            ),
        )
    )

    assert recovery_events[-1].type == EventType.SESSION_COMPLETED
    assert [len(request.messages) for request in provider.requests] == [1, 3, 2, 2]
    completed = [
        event for event in recovery_events if event.type == EventType.CONTEXT_COMPACTION_COMPLETED
    ]
    assert len(completed) == 1
    assert completed[0].payload["chunk_mode"] == "single_request"


def test_context_overflow_without_policy_fails_without_retry():
    provider = ContextOverflowProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_overflow_no_policy",
                messages=[Message.text("user", "too much")],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert EventType.CONTEXT_OVERFLOW_DETECTED not in [event.type for event in events]
    assert [event.type for event in events].count(EventType.MODEL_ERROR) == 1
    failed = events[-1]
    assert failed.type == EventType.SESSION_FAILED
    assert failed.payload["error"] == "context too large"
    assert failed.payload["error_type"] == "ModelContextOverflowError"


def test_context_overflow_policy_fails_cleanly_when_recovery_overflows():
    provider = ContextOverflowProvider(overflow_requests=2)
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=MessageWindowContextPolicy(max_messages=1),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_overflow_recovery_failed",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
            ),
        )
    )

    assert len(provider.requests) == 2
    model_errors = [event for event in events if event.type == EventType.MODEL_ERROR]
    assert len(model_errors) == 2
    private_events = asyncio.run(
        _private_events_for_public_events(
            app.session_store,
            "context_overflow_recovery_failed",
            events,
        )
    )
    private_model_errors = [
        event for event in private_events if event.type == EventType.MODEL_ERROR
    ]
    assert (
        private_model_errors[0].payload["model_step_id"]
        == private_model_errors[1].payload["model_step_id"]
    )
    assert (
        private_model_errors[0].payload["model_attempt_id"]
        != private_model_errors[1].payload["model_attempt_id"]
    )
    assert [
        event.type
        for event in events
        if event.type
        in {
            EventType.CONTEXT_OVERFLOW_DETECTED,
            EventType.CONTEXT_OVERFLOW_RECOVERING,
            EventType.CONTEXT_OVERFLOW_FAILED,
        }
    ] == [
        EventType.CONTEXT_OVERFLOW_DETECTED,
        EventType.CONTEXT_OVERFLOW_RECOVERING,
        EventType.CONTEXT_OVERFLOW_FAILED,
    ]
    recovery_started = [event for event in private_events if event.type == EventType.MODEL_STARTED][
        -1
    ]
    failed = next(
        event for event in private_events if event.type == EventType.CONTEXT_OVERFLOW_FAILED
    )
    assert failed.payload == {
        "step": 1,
        "phase": "recovery",
        "error": "context too large",
        "error_type": "ModelContextOverflowError",
        "provider": "overflow",
        "original_message_count": 2,
        "status_code": 400,
        "provider_error_code": "context_length_exceeded",
        "recovery_message_count": 1,
        "model_step_id": recovery_started.payload["model_step_id"],
        "model_attempt_id": recovery_started.payload["model_attempt_id"],
        "execution_profile_fingerprint": recovery_started.payload["execution_profile_fingerprint"],
    }
    assert events[-1].type == EventType.SESSION_FAILED


def test_register_agent_rejects_invalid_context_overflow_policy():
    app = CayuApp()
    with pytest.raises(TypeError, match="context_overflow_policy must be a ContextPolicy"):
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_overflow_policy=object(),  # type: ignore[arg-type]
        )


def test_register_agent_rejects_invalid_targeted_tool_mode():
    app = CayuApp()
    with pytest.raises(TypeError, match="targeted_tool_mode must be a TargetedToolMode"):
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            targeted_tool_mode=1,  # type: ignore[arg-type]
        )


def test_register_agent_rejects_targeted_tools_without_store_authority():
    class UnsupportedTargetedStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        supports_public_authority_aliases = False
        supports_targeted_tool_grants = False

    app = CayuApp(session_store=UnsupportedTargetedStore())
    with pytest.raises(RuntimeError, match="durable targeted-grant state"):
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            targeted_tool_mode="call_tool",
        )


def test_model_compactor_rejects_tool_calls_from_compaction_model():
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_1",
                name="echo",
                arguments={"text": "bad"},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    compactor = ModelCompactor(provider=provider, model="summary-model")

    with pytest.raises(RuntimeError, match="must not call tools"):
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )


def test_model_compactor_fails_on_provider_error():
    provider = FakeProvider([ModelStreamEvent.error("provider unavailable")])
    compactor = ModelCompactor(provider=provider, model="summary-model")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )


def test_model_compactor_bounds_large_compaction_input():
    completed_batch = [
        ModelStreamEvent.text_delta("bounded summary"),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]
    provider = FakeProvider([completed_batch for _index in range(16)])
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        max_input_chars=1000,
    )

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "x" * 5000)],
                existing_summary="important previous summary",
            )
        )
    )

    assert len(provider.requests) > 1
    assert all(len(request.messages[1].content[0].text) <= 1000 for request in provider.requests)
    assert "x" * 100 in provider.requests[0].messages[1].content[0].text
    assert any(
        "important previous summary" in request.messages[1].content[0].text
        for request in provider.requests
    )
    assert result.covered_message_count == 1
    assert result.source_chunk_count > 1
    assert result.source_chunk_mode == "hierarchical_atomic_unit"
    assert result.bounded_input is True
    assert result.metadata["input_truncated"] is True
    assert result.metadata["max_input_chars"] == 1000


def test_checkpoint_policy_rejects_unmarked_zero_coverage_result():
    class ZeroCoverageCompactor(ContextCompactor):
        async def compact(self, request: CompactionRequest) -> CompactionResult:
            del request
            return CompactionResult(
                summary="summary without represented source",
                covered_message_count=0,
            )

    policy = CheckpointCompactionContextPolicy(
        compactor=ZeroCoverageCompactor(),
        max_user_turns=1,
        compact_after_messages=1,
    )

    with pytest.raises(
        ContextBuildError,
        match="zero coverage must report progress_exhausted=true",
    ) as exc_info:
        asyncio.run(
            policy.build_with_checkpoint(
                ContextRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                    step=1,
                    force_compaction=True,
                ),
                checkpoint=None,
            )
        )

    assert isinstance(exc_info.value.cause, ValueError)
    assert exc_info.value.checkpoint is None
    assert [item.event_type for item in exc_info.value.compaction_telemetry] == [
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_FAILED,
    ]


def test_checkpoint_policy_rejects_zero_coverage_summary_replacement():
    class ReplacingExhaustedCompactor(ContextCompactor):
        def _progress_key(self) -> str:
            return "replace-exhausted:v1"

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            assert request.existing_summary == "OLD_HISTORY_FACT"
            return CompactionResult(
                summary="replacement that omits the old history",
                covered_message_count=0,
                source_chunk_count=0,
                progress_exhausted=True,
                progress_key="replace-exhausted:v1",
            )

    policy = CheckpointCompactionContextPolicy(
        compactor=ReplacingExhaustedCompactor(),
        max_user_turns=1,
        compact_after_messages=1,
    )
    checkpoint = {
        "context_compaction": {
            "version": 2,
            "summary": "OLD_HISTORY_FACT",
            "compacted_transcript_cursor": 2,
            "metadata": {"source": "seeded"},
        }
    }

    with pytest.raises(
        ContextBuildError,
        match="zero-coverage compaction must preserve the existing summary",
    ) as exc_info:
        asyncio.run(
            policy.build_with_checkpoint(
                ContextRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[
                        Message.text("user", "already covered request"),
                        Message.text("assistant", "already covered answer"),
                        Message.text("user", "new compactable request"),
                        Message.text("assistant", "new compactable answer"),
                        Message.text("user", "current"),
                    ],
                    step=1,
                    force_compaction=True,
                ),
                checkpoint=checkpoint,
            )
        )

    assert isinstance(exc_info.value.cause, ValueError)
    assert checkpoint["context_compaction"] == {
        "version": 2,
        "summary": "OLD_HISTORY_FACT",
        "compacted_transcript_cursor": 2,
        "metadata": {"source": "seeded"},
    }


@pytest.mark.parametrize(
    "represented_existing_summary_sha256",
    [None, "0" * 64],
    ids=["missing", "wrong"],
)
def test_checkpoint_policy_rejects_unbound_existing_summary_replacement(
    represented_existing_summary_sha256,
):
    class UnboundReplacementCompactor(ContextCompactor):
        async def compact(self, request: CompactionRequest) -> CompactionResult:
            assert request.existing_summary == "OLD_HISTORY_FACT"
            return CompactionResult(
                summary="new context without a prior-summary binding",
                covered_message_count=len(request.messages),
                represented_existing_summary_sha256=(represented_existing_summary_sha256),
            )

    policy = CheckpointCompactionContextPolicy(
        compactor=UnboundReplacementCompactor(),
        max_user_turns=1,
        compact_after_messages=1,
    )

    with pytest.raises(
        ContextBuildError,
        match="must bind represented_existing_summary_sha256",
    ) as exc_info:
        asyncio.run(
            policy.build_with_checkpoint(
                ContextRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[
                        Message.text("user", "already covered request"),
                        Message.text("assistant", "already covered answer"),
                        Message.text("user", "new compactable request"),
                        Message.text("assistant", "new compactable answer"),
                        Message.text("user", "current"),
                    ],
                    step=1,
                    force_compaction=True,
                ),
                checkpoint={
                    "context_compaction": {
                        "version": 2,
                        "summary": "OLD_HISTORY_FACT",
                        "compacted_transcript_cursor": 2,
                        "metadata": {"source": "seeded"},
                    }
                },
            )
        )

    assert isinstance(exc_info.value.cause, ValueError)


def test_model_compactor_bounds_by_complete_message_prefix_and_reports_coverage():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("bounded summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        max_input_chars=1000,
    )
    messages = [
        Message.text("user", "OLDEST_MUST_SURVIVE " + "x" * 400),
        Message.text("assistant", "LATEST_COMPACTABLE " + "x" * 5000),
    ]

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
            )
        )
    )

    prompt = provider.requests[0].messages[1].content[0].text
    assert result.covered_message_count == 1
    assert "OLDEST_MUST_SURVIVE" in prompt
    assert "LATEST_COMPACTABLE" not in prompt


@pytest.mark.parametrize(
    "messages",
    [
        [Message.text("user", "café 😀"), Message.text("assistant", "résumé")],
        [
            Message.tool_call(
                tool_call_id="call_1",
                tool_name="inspect",
                arguments={"path": "report.txt"},
            ),
            Message.tool_result(
                tool_call_id="call_1",
                tool_name="inspect",
                content="result",
            ),
        ],
    ],
    ids=["unicode", "tool_round"],
)
def test_digest_compactor_preserves_message_boundaries(messages):
    result = asyncio.run(
        TranscriptDigestCompactor(max_summary_chars=800).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
            )
        )
    )
    assert result.covered_message_count == len(messages)
    assert result.summary


def test_digest_compactor_covers_provider_state_and_attachment_messages():
    messages = [
        Message(
            role=MessageRole.ASSISTANT,
            content=(ProviderStatePart(provider="fake", state={"opaque": True}),),
        ),
        Message(
            role=MessageRole.USER,
            content=(
                FilePart(
                    attachment=file_attachment(
                        artifact_id="artifact-coverage",
                        kind="image",
                        filename="coverage.png",
                        content_type="image/png",
                        size_bytes=5,
                    )
                ),
            ),
        ),
    ]
    result = asyncio.run(
        TranscriptDigestCompactor(max_summary_chars=800).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
            )
        )
    )
    assert result.covered_message_count == 2
    assert "provider_state" in result.summary
    assert "coverage.png" in result.summary
    assert "artifact-coverage" in result.summary


def test_digest_compactor_represents_tool_result_structured_data_and_artifacts():
    result = asyncio.run(
        TranscriptDigestCompactor(max_summary_chars=1200).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[
                    Message.tool_call(
                        tool_call_id="call_rich_result",
                        tool_name="inspect",
                        arguments={"path": "report.txt"},
                    ),
                    Message.tool_result(
                        tool_call_id="call_rich_result",
                        tool_name="inspect",
                        content="visible result",
                        structured={"decision": "STRUCTURED_MARKER"},
                        artifacts=[{"artifact_id": "TOOL_ARTIFACT_MARKER"}],
                    ),
                ],
            )
        )
    )
    assert result.covered_message_count == 2
    assert "STRUCTURED_MARKER" in result.summary
    assert "TOOL_ARTIFACT_MARKER" in result.summary


def test_digest_compactor_keeps_oversized_uncovered_source_in_every_projection():
    policy = CheckpointCompactionContextPolicy(
        compactor=TranscriptDigestCompactor(max_summary_chars=200),
        max_user_turns=1,
        compact_after_messages=1,
    )
    messages = [
        Message.text("user", "DIGEST_OLDEST_SURVIVES " + "x" * 1000),
        Message.text("user", "current"),
    ]

    first = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=1,
                force_compaction=True,
            ),
            checkpoint=None,
        )
    )
    assert first.checkpoint is not None
    assert first.checkpoint["context_compaction"]["compacted_transcript_cursor"] == 0
    assert "summary" not in first.checkpoint["context_compaction"]
    assert first.checkpoint["context_compaction"]["no_progress"] is True
    assert first.checkpoint["context_compaction"]["progress"] == {
        "exhausted": True,
        "key": policy.compactor._progress_key(),
    }
    first_projection = json.dumps([message.model_dump(mode="json") for message in first.messages])
    assert first_projection.count("DIGEST_OLDEST_SURVIVES") == 1

    second = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=1,
                force_compaction=True,
            ),
            checkpoint=first.checkpoint,
        )
    )
    assert second.checkpoint is not None
    assert second.checkpoint["context_compaction"]["compacted_transcript_cursor"] == 0
    second_projection = json.dumps([message.model_dump(mode="json") for message in second.messages])
    assert second_projection.count("DIGEST_OLDEST_SURVIVES") == 1

    automatic = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=2,
            ),
            checkpoint=second.checkpoint,
        )
    )
    assert automatic.checkpoint is None
    assert automatic.compaction_telemetry == []

    changed_policy = CheckpointCompactionContextPolicy(
        compactor=TranscriptDigestCompactor(max_summary_chars=300),
        max_user_turns=1,
        compact_after_messages=1,
    )
    changed = asyncio.run(
        changed_policy.build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=3,
            ),
            checkpoint=second.checkpoint,
        )
    )
    assert changed.checkpoint is not None
    assert (
        changed.checkpoint["context_compaction"]["progress"]["key"]
        != second.checkpoint["context_compaction"]["progress"]["key"]
    )
    assert any(
        item.event_type == EventType.CONTEXT_COMPACTION_STARTED
        for item in changed.compaction_telemetry
    )


def test_digest_compactor_preserves_oversized_seeded_summary_and_stops_retriggering() -> None:
    policy = CheckpointCompactionContextPolicy(
        compactor=TranscriptDigestCompactor(max_summary_chars=200),
        max_user_turns=1,
        compact_after_messages=1,
    )
    previous_summary = "PRIOR_SUMMARY_FACT " * 20
    checkpoint = {
        "context_compaction": {
            "version": 2,
            "summary": previous_summary,
            "compacted_transcript_cursor": 2,
            "metadata": {"source": "larger-compactor"},
        }
    }
    messages = [
        Message.text("user", "already covered"),
        Message.text("assistant", "already covered answer"),
        Message.text("user", "NEW_UNCOVERED_FACT"),
        Message.text("assistant", "new answer"),
        Message.text("user", "current"),
    ]

    first = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=1,
                force_compaction=True,
            ),
            checkpoint=checkpoint,
        )
    )

    assert first.checkpoint is not None
    compacted = first.checkpoint["context_compaction"]
    assert compacted["summary"] == previous_summary
    assert compacted["compacted_transcript_cursor"] == 2
    assert compacted["progress"] == {
        "exhausted": True,
        "key": policy.compactor._progress_key(),
    }
    projection = json.dumps([message.model_dump(mode="json") for message in first.messages])
    assert projection.count("PRIOR_SUMMARY_FACT") == 20
    assert projection.count("NEW_UNCOVERED_FACT") == 1

    second = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=2,
            ),
            checkpoint=first.checkpoint,
        )
    )
    assert second.checkpoint is None
    assert second.compaction_telemetry == []
    second_projection = json.dumps([message.model_dump(mode="json") for message in second.messages])
    assert second_projection.count("PRIOR_SUMMARY_FACT") == 20
    assert second_projection.count("NEW_UNCOVERED_FACT") == 1


def test_digest_exhaustion_does_not_suppress_replacement_subclass() -> None:
    class ProgressCapableDigestCompactor(TranscriptDigestCompactor):
        def __init__(self) -> None:
            super().__init__(max_summary_chars=200)
            self.requests: list[CompactionRequest] = []

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.requests.append(request)
            return CompactionResult(
                summary="replacement summary",
                covered_message_count=len(request.messages),
                represented_existing_summary_sha256=(
                    hashlib.sha256(request.existing_summary.encode("utf-8")).hexdigest()
                    if request.existing_summary is not None
                    else None
                ),
                source_chunk_mode="custom",
                bounded_input=True,
            )

    messages = [
        Message.text("user", "REPLACEMENT_SOURCE " + "x" * 1000),
        Message.text("user", "current"),
    ]
    exhausted_policy = CheckpointCompactionContextPolicy(
        compactor=TranscriptDigestCompactor(max_summary_chars=200),
        max_user_turns=1,
        compact_after_messages=1,
    )
    exhausted = asyncio.run(
        exhausted_policy.build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=1,
            ),
            checkpoint=None,
        )
    )
    assert exhausted.checkpoint is not None
    exhausted_context = exhausted.checkpoint["context_compaction"]
    assert exhausted_context["progress"]["exhausted"] is True

    replacement = ProgressCapableDigestCompactor()
    replacement_policy = CheckpointCompactionContextPolicy(
        compactor=replacement,
        max_user_turns=1,
        compact_after_messages=1,
    )
    progressed = asyncio.run(
        replacement_policy.build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=2,
            ),
            checkpoint=exhausted.checkpoint,
        )
    )

    assert len(replacement.requests) == 1
    assert progressed.checkpoint is not None
    compacted = progressed.checkpoint["context_compaction"]
    assert compacted["summary"] == "replacement summary"
    assert compacted["compacted_transcript_cursor"] == 1
    assert "progress" not in compacted


def test_digest_no_progress_modes_survive_public_event_sanitization() -> None:
    async def run() -> None:
        session_id = "sess_digest_no_progress_public_telemetry"
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=TranscriptDigestCompactor(max_summary_chars=200),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[
                    Message.text("user", "oversized " + "x" * 1000),
                    Message.text("user", "current"),
                ],
            ),
        )

        completed = next(
            event for event in events if event.type == EventType.CONTEXT_COMPACTION_COMPLETED
        )
        assert completed.payload["coverage_mode"] == "no_progress"
        assert completed.payload["chunk_mode"] == "digest_capacity_exhausted"
        assert completed.payload["represented_message_count"] == 0
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 0

    asyncio.run(run())


def test_digest_no_progress_marker_is_discarded_with_invalid_checkpoint() -> None:
    compactor = TranscriptDigestCompactor(max_summary_chars=400)
    policy = CheckpointCompactionContextPolicy(
        compactor=compactor,
        max_user_turns=1,
        compact_after_messages=1,
    )
    messages = [
        Message.text("user", "SOURCE_MUST_BE_REBUILT"),
        Message.text("user", "current"),
    ]
    invalid_checkpoint = {
        "context_compaction": {
            "version": 2,
            "summary": "stale summary",
            "compacted_transcript_cursor": 99,
            "metadata": {},
            "progress": {
                "exhausted": True,
                "key": compactor._progress_key(),
            },
        }
    }

    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=1,
            ),
            checkpoint=invalid_checkpoint,
        )
    )

    assert result.checkpoint is not None
    assert result.checkpoint["context_compaction"]["compacted_transcript_cursor"] == 1
    assert "SOURCE_MUST_BE_REBUILT" in result.checkpoint["context_compaction"]["summary"]
    assert any(
        item.event_type == EventType.CONTEXT_COMPACTION_STARTED
        for item in result.compaction_telemetry
    )


def test_runtime_progress_state_does_not_overwrite_custom_compactor_metadata() -> None:
    class MetadataCompactor(ContextCompactor):
        async def compact(self, request: CompactionRequest) -> CompactionResult:
            return CompactionResult(
                summary="custom summary",
                covered_message_count=len(request.messages),
                metadata={
                    "progress_exhausted": "custom value",
                    "progress_key": {"owner": "custom compactor"},
                },
            )

    result = asyncio.run(
        CheckpointCompactionContextPolicy(
            compactor=MetadataCompactor(),
            max_user_turns=1,
            compact_after_messages=1,
        ).build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[
                    Message.text("user", "old"),
                    Message.text("user", "current"),
                ],
                step=1,
                force_compaction=True,
            ),
            checkpoint=None,
        )
    )

    assert result.checkpoint is not None
    compacted = result.checkpoint["context_compaction"]
    assert compacted["metadata"] == {
        "progress_exhausted": "custom value",
        "progress_key": {"owner": "custom compactor"},
    }
    assert "progress" not in compacted


def test_digest_compactor_does_not_acknowledge_half_of_tool_round():
    result = asyncio.run(
        TranscriptDigestCompactor(max_summary_chars=240).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[
                    Message.tool_call(
                        tool_call_id="call_digest_atomic",
                        tool_name="inspect",
                        arguments={"query": "x" * 160},
                    ),
                    Message.tool_result(
                        tool_call_id="call_digest_atomic",
                        tool_name="inspect",
                        content="result",
                    ),
                ],
            )
        )
    )
    assert result.covered_message_count == 0


def test_digest_compactor_builds_each_source_message_once(monkeypatch):
    messages = [
        Message.text(
            "user" if index % 2 == 0 else "assistant",
            f"message-{index:04d}",
        )
        for index in range(4096)
    ]
    original = runtime_context_module._message_digest
    digest_calls = 0

    def counted_message_digest(message):
        nonlocal digest_calls
        digest_calls += 1
        return original(message)

    monkeypatch.setattr(runtime_context_module, "_message_digest", counted_message_digest)
    result = asyncio.run(
        TranscriptDigestCompactor(max_summary_chars=1_000_000).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
            )
        )
    )

    assert result.covered_message_count == len(messages)
    assert digest_calls == len(messages)


def test_compaction_result_rejects_missing_or_negative_coverage():
    with pytest.raises(ValueError):
        CompactionResult(summary="summary")
    with pytest.raises(ValueError):
        CompactionResult(summary="summary", covered_message_count=-1)
    with pytest.raises(ValueError):
        CompactionPrompt(prompt="custom prompt", covered_message_count=0)
    with pytest.raises(ValueError):
        CompactionResult(
            summary="summary",
            covered_message_count=0,
            source_chunk_count=-1,
        )
    with pytest.raises(ValueError):
        CompactionResult(
            summary="summary",
            covered_message_count=0,
            source_chunk_mode=" ",
        )
    for invalid_sha256 in ("abc", "A" * 64, "g" * 64):
        with pytest.raises(ValueError, match="lowercase SHA-256"):
            CompactionResult(
                summary="summary",
                covered_message_count=1,
                represented_existing_summary_sha256=invalid_sha256,
            )
    for invalid_mode in ("private\nmode", "x" * 513):
        with pytest.raises(ValueError):
            CompactionResult(
                summary="summary",
                covered_message_count=0,
                source_chunk_mode=invalid_mode,
            )
    for invalid_progress_key in ("private\nkey", "x" * 513):
        with pytest.raises(ValueError):
            CompactionResult(
                summary="summary",
                covered_message_count=0,
                progress_exhausted=True,
                progress_key=invalid_progress_key,
            )
    with pytest.raises(ValueError, match="requires progress_key"):
        CompactionResult(
            summary="summary",
            covered_message_count=0,
            progress_exhausted=True,
        )
    with pytest.raises(ValueError, match="zero coverage"):
        CompactionResult(
            summary="summary",
            covered_message_count=1,
            progress_exhausted=True,
            progress_key="test:v1",
        )
    for invalid_count in (True, "1", 1.0):
        with pytest.raises(ValueError):
            CompactionResult(
                summary="summary",
                covered_message_count=invalid_count,
            )
        with pytest.raises(ValueError):
            CompactionPrompt(
                prompt="custom prompt",
                covered_message_count=invalid_count,
            )


@pytest.mark.parametrize("second_size", [100, 600, 1200, 5000])
def test_model_compactor_coverage_never_crosses_a_message_boundary(second_size):
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    messages = [
        Message.text("user", "first-marker"),
        Message.text("assistant", "second-marker " + "x" * second_size),
    ]
    result = asyncio.run(
        ModelCompactor(
            provider=provider,
            model="summary-model",
            max_input_chars=1000,
        ).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
            )
        )
    )
    prompt = provider.requests[0].messages[-1].content[0].text
    assert result.covered_message_count in {1, 2}
    assert "first-marker" in prompt
    assert ("second-marker" in prompt) is (result.covered_message_count == 2)


@settings(max_examples=40, deadline=None)
@given(
    source_shapes=st.lists(
        st.tuples(
            st.one_of(
                st.sampled_from([0, 1, 700, 799, 800, 801, 999, 1000, 1001, 2000, 4000]),
                st.integers(min_value=0, max_value=4000),
            ),
            st.booleans(),
        ),
        min_size=1,
        max_size=5,
    )
)
def test_model_compaction_property_preserves_every_covered_source_marker(
    source_shapes: list[tuple[int, bool]],
) -> None:
    markers = [f"CAYU_COVERAGE_MARKER_{index:02d}" for index in range(len(source_shapes))]

    class MarkerPreservingProvider(ModelProvider):
        name = "property-marker-preserving"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            prompt = request.messages[-1].content[0].text
            represented = [marker for marker in markers if marker in prompt]
            yield ModelStreamEvent.text_delta(" ".join(represented) or "bounded fragment")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    source_messages = [
        Message.text(
            MessageRole.USER if index % 2 == 0 else MessageRole.ASSISTANT,
            marker + " " + (("😀é" * ((size + 1) // 2))[:size] if unicode_filler else "x" * size),
        )
        for index, (marker, (size, unicode_filler)) in enumerate(
            zip(
                markers,
                source_shapes,
                strict=True,
            )
        )
    ]
    messages = [*source_messages, Message.text(MessageRole.USER, "current turn")]
    original_messages = [message.model_copy(deep=True) for message in messages]
    policy = CheckpointCompactionContextPolicy(
        compactor=ModelCompactor(
            provider=MarkerPreservingProvider(),
            model="summary-model",
            max_input_chars=1000,
        ),
        max_user_turns=1,
        compact_after_messages=1,
    )

    checkpoint = None
    previous_cursor = 0
    for _attempt in range(len(source_messages)):
        result = asyncio.run(
            policy.build_with_checkpoint(
                ContextRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=messages,
                    step=1,
                    force_compaction=True,
                ),
                checkpoint=checkpoint,
            )
        )
        assert result.checkpoint is not None
        checkpoint = result.checkpoint
        compaction_checkpoint = checkpoint["context_compaction"]
        cursor = compaction_checkpoint["compacted_transcript_cursor"]
        assert previous_cursor < cursor <= len(source_messages)
        assert compaction_checkpoint["summary"].split().count(markers[cursor - 1]) == 1

        projection = json.dumps(
            [message.model_dump(mode="json") for message in result.messages],
            ensure_ascii=False,
        )
        summary = compaction_checkpoint["summary"]
        for index, marker in enumerate(markers):
            assert projection.count(marker) == 1
            assert summary.split().count(marker) == (1 if index < cursor else 0)

        assert messages == original_messages
        previous_cursor = cursor
        if cursor == len(source_messages):
            break

    assert previous_cursor == len(source_messages)


def test_bounded_model_compactor_finds_largest_prefix_with_logarithmic_rebuilds(
    monkeypatch,
):
    messages = [
        Message.text(
            "user" if index % 2 == 0 else "assistant",
            f"message-{index:04d} " + "x" * 20,
        )
        for index in range(1024)
    ]
    request = CompactionRequest(
        session=_test_session(),
        agent=AgentSpec(name="assistant", model="fake-model"),
        messages=messages,
    )
    original = runtime_context_module._default_compaction_prompt_parts
    rebuild_count = 0

    def counted_prompt_parts(candidate):
        nonlocal rebuild_count
        rebuild_count += 1
        return original(candidate)

    monkeypatch.setattr(
        runtime_context_module,
        "_default_compaction_prompt_parts",
        counted_prompt_parts,
    )
    prompt, truncated, covered = runtime_context_module._bounded_default_compaction_prompt(
        request,
        max_chars=4000,
    )

    assert prompt is not None
    assert truncated is True
    assert 0 < covered < len(messages)
    assert len(prompt) <= 4000
    assert rebuild_count <= 12


def test_model_compactor_never_splits_tool_call_from_result_at_input_bound():
    completed_batch = [
        ModelStreamEvent.text_delta("summary"),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]
    tool_round = [
        Message.tool_call(
            tool_call_id="call_atomic",
            tool_name="inspect",
            arguments={"query": "TOOL_CALL_MARKER " + "x" * 1500},
        ),
        Message.tool_result(
            tool_call_id="call_atomic",
            tool_name="inspect",
            content="TOOL_RESULT_MARKER",
        ),
    ]

    prefix_provider = FakeProvider(completed_batch)
    prefix_result = asyncio.run(
        ModelCompactor(
            provider=prefix_provider,
            model="summary-model",
            max_input_chars=1000,
        ).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "safe-prefix"), *tool_round],
            )
        )
    )
    prefix_prompt = prefix_provider.requests[0].messages[-1].content[0].text
    assert prefix_result.covered_message_count == 1
    assert "safe-prefix" in prefix_prompt
    assert "TOOL_CALL_MARKER" not in prefix_prompt
    assert "TOOL_RESULT_MARKER" not in prefix_prompt

    hierarchy_provider = FakeProvider([completed_batch for _index in range(16)])
    hierarchy_result = asyncio.run(
        ModelCompactor(
            provider=hierarchy_provider,
            model="summary-model",
            max_input_chars=1000,
        ).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=tool_round,
            )
        )
    )
    hierarchy_prompts = [
        request.messages[-1].content[0].text for request in hierarchy_provider.requests
    ]
    assert hierarchy_result.covered_message_count == 2
    assert hierarchy_result.source_chunk_mode == "hierarchical_atomic_unit"
    assert any("TOOL_CALL_MARKER" in prompt for prompt in hierarchy_prompts)
    assert any("TOOL_RESULT_MARKER" in prompt for prompt in hierarchy_prompts)


@pytest.mark.parametrize(
    ("message", "marker"),
    [
        (
            Message(
                role=MessageRole.USER,
                content=(
                    TextPart(text="UNICODE_BOUNDARY_😀 " + "x" * 2000),
                    FilePart(
                        attachment=file_attachment(
                            artifact_id="HIERARCHY_FILE_MARKER",
                            kind="image",
                            filename="hierarchy.png",
                            content_type="image/png",
                            size_bytes=5,
                        )
                    ),
                ),
            ),
            "HIERARCHY_FILE_MARKER",
        ),
        (
            Message(
                role=MessageRole.ASSISTANT,
                content=(
                    TextPart(text="x" * 2000),
                    ProviderStatePart(
                        provider="fake",
                        state={"opaque": "HIERARCHY_PROVIDER_STATE"},
                    ),
                ),
            ),
            "provider_state provider=fake",
        ),
    ],
    ids=["file_and_unicode", "provider_state"],
)
def test_hierarchical_compaction_represents_boundary_message_parts(message, marker):
    completed_batch = [
        ModelStreamEvent.text_delta("summary"),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]
    provider = FakeProvider([completed_batch for _index in range(16)])
    result = asyncio.run(
        ModelCompactor(
            provider=provider,
            model="summary-model",
            max_input_chars=1000,
        ).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[message],
            )
        )
    )
    prompts = [request.messages[-1].content[0].text for request in provider.requests]
    assert result.covered_message_count == 1
    assert result.source_chunk_mode == "hierarchical_atomic_unit"
    assert any(marker in prompt for prompt in prompts)
    if message.role == MessageRole.USER:
        assert any("UNICODE_BOUNDARY_😀" in prompt for prompt in prompts)


def test_checkpoint_policy_rejects_compactor_coverage_that_splits_tool_round():
    class SplitToolRoundCompactor(ContextCompactor):
        async def compact(self, request: CompactionRequest) -> CompactionResult:
            return CompactionResult(summary="invalid split", covered_message_count=2)

    policy = CheckpointCompactionContextPolicy(
        compactor=SplitToolRoundCompactor(),
        max_user_turns=1,
        compact_after_messages=1,
    )
    messages = [
        Message.text("user", "old"),
        Message.tool_call(
            tool_call_id="call_split",
            tool_name="inspect",
            arguments={"path": "report.txt"},
        ),
        Message.tool_result(
            tool_call_id="call_split",
            tool_name="inspect",
            content="done",
        ),
        Message.text("user", "current"),
    ]
    with pytest.raises(ContextBuildError, match="splits an assistant/tool round") as exc_info:
        asyncio.run(
            policy.build_with_checkpoint(
                ContextRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=messages,
                    step=1,
                    force_compaction=True,
                ),
                checkpoint=None,
            )
        )
    assert isinstance(exc_info.value.cause, ValueError)
    assert exc_info.value.compaction_telemetry[-1].event_type == (
        EventType.CONTEXT_COMPACTION_FAILED
    )
    assert exc_info.value.compaction_telemetry[-1].payload["represented_source_end"] == 0


def test_checkpoint_policy_discards_legacy_cursor_that_splits_tool_round():
    class CaptureCompactor(ContextCompactor):
        def __init__(self) -> None:
            self.request: CompactionRequest | None = None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.request = request
            return CompactionResult(
                summary="replacement summary",
                covered_message_count=len(request.messages),
            )

    compactor = CaptureCompactor()
    policy = CheckpointCompactionContextPolicy(
        compactor=compactor,
        max_user_turns=1,
        compact_after_messages=1,
    )
    messages = [
        Message.text("user", "old"),
        Message.tool_call(
            tool_call_id="call_legacy_split",
            tool_name="inspect",
            arguments={"path": "report.txt"},
        ),
        Message.tool_result(
            tool_call_id="call_legacy_split",
            tool_name="inspect",
            content="done",
        ),
        Message.text("user", "current"),
    ]
    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=1,
                force_compaction=True,
            ),
            checkpoint={
                "context_compaction": {
                    "version": 2,
                    "summary": "legacy invalid summary",
                    "compacted_transcript_cursor": 2,
                    "metadata": {},
                }
            },
        )
    )
    assert compactor.request is not None
    assert compactor.request.existing_summary is None
    assert compactor.request.messages == messages[:3]
    assert result.checkpoint is not None
    assert result.checkpoint["context_compaction"]["compacted_transcript_cursor"] == 3


def test_checkpoint_policy_invalidates_version_one_checkpoint_with_valid_boundary():
    class CaptureCompactor(ContextCompactor):
        def __init__(self) -> None:
            self.request: CompactionRequest | None = None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.request = request
            return CompactionResult(
                summary="replacement summary",
                covered_message_count=len(request.messages),
            )

    compactor = CaptureCompactor()
    policy = CheckpointCompactionContextPolicy(
        compactor=compactor,
        max_user_turns=1,
        compact_after_messages=1,
    )
    messages = [
        Message.text("user", "legacy source may have been clipped"),
        Message.text("assistant", "legacy answer"),
        Message.text("user", "current"),
    ]
    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=1,
                force_compaction=True,
            ),
            checkpoint={
                "context_compaction": {
                    "version": 1,
                    "summary": "legacy potentially incomplete summary",
                    "compacted_transcript_cursor": 2,
                    "metadata": {},
                }
            },
        )
    )

    assert compactor.request is not None
    assert compactor.request.existing_summary is None
    assert compactor.request.messages == messages[:2]
    assert result.checkpoint is not None
    assert result.checkpoint["context_compaction"]["version"] == 2
    assert result.checkpoint["context_compaction"]["compacted_transcript_cursor"] == 2


def test_hierarchical_compaction_cancellation_never_returns_partial_coverage():
    class CancelSecondHierarchyCallProvider(ModelProvider):
        name = "cancel-hierarchy"

        def __init__(self) -> None:
            self.calls = 0
            self.second_started = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.calls += 1
            if self.calls == 2:
                self.second_started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            yield ModelStreamEvent.text_delta("first fragment summary")
            yield ModelStreamEvent.completed({"usage": {"input_tokens": 5, "output_tokens": 2}})

    async def run() -> None:
        provider = CancelSecondHierarchyCallProvider()
        task = asyncio.create_task(
            ModelCompactor(
                provider=provider,
                model="summary-model",
                max_input_chars=1000,
            ).compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "oversized " + "😀" * 3000)],
                )
            )
        )
        await provider.second_started.wait()
        task.cancel("cancel hierarchy")
        with pytest.raises(asyncio.CancelledError, match="cancel hierarchy"):
            await task
        assert provider.calls == 2

    asyncio.run(run())


def test_automatic_hierarchical_cancellation_persists_completed_chunk_usage():
    class CancelSecondHierarchyCallProvider(ModelProvider):
        name = "cancel-hierarchy-accounting"

        def __init__(self) -> None:
            self.calls = 0
            self.second_started = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.calls += 1
            if self.calls == 2:
                self.second_started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            yield ModelStreamEvent.text_delta("first fragment summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                }
            )

    async def run() -> None:
        session_id = "sess_hierarchy_cancelled_after_completion"
        store = InMemorySessionStore()
        compactor_provider = CancelSecondHierarchyCallProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compactor_provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[
                        Message.text("user", "oversized " + "x" * 5000),
                        Message.text("user", "current"),
                    ],
                ),
            )
        )
        await asyncio.wait_for(compactor_provider.second_started.wait(), timeout=10)
        task.cancel("cancel hierarchy after first completion")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert exc_info.value.args == ("cancel hierarchy after first completion",)
        events = await store.load_events(session_id)
        compaction_completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(compaction_completions) == 2
        assert compaction_completions[0].payload["usage_metrics"]["input_tokens"] == 5
        assert compaction_completions[0].payload["usage_metrics"]["output_tokens"] == 2
        assert compaction_completions[1].payload["compaction_outcome"] == "cancelled"
        assert compaction_completions[1].payload["usage_unavailable_reason"]
        failures = [event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED]
        assert len(failures) == 1
        assert failures[0].payload["chunk_count"] == 2
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is None or "context_compaction" not in checkpoint

    asyncio.run(run())


def test_automatic_hierarchical_cancellation_persists_each_completed_chunk():
    class CancelThirdHierarchyCallProvider(ModelProvider):
        name = "cancel-third-hierarchy-accounting"

        def __init__(self) -> None:
            self.calls = 0
            self.third_started = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.calls += 1
            if self.calls == 3:
                self.third_started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            yield ModelStreamEvent.text_delta(f"fragment summary {self.calls}")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {
                        "input_tokens": self.calls,
                        "output_tokens": 1,
                    },
                }
            )

    async def run() -> None:
        session_id = "sess_hierarchy_cancelled_batch_persistence"
        store = InMemorySessionStore()
        budget_store = InMemoryBudgetStore()
        compactor_provider = CancelThirdHierarchyCallProvider()
        app = CayuApp(
            session_store=store,
            budget_store=budget_store,
            enable_logging=False,
        )
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compactor_provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[
                        Message.text("user", "oversized " + "x" * 5000),
                        Message.text("user", "current"),
                    ],
                ),
            )
        )
        await asyncio.wait_for(compactor_provider.third_started.wait(), timeout=10)
        task.cancel("cancel hierarchy after two completions")
        assert task.cancelling() == 1
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert exc_info.value.args == ("cancel hierarchy after two completions",)
        assert task.cancelled()
        events = await store.load_events(session_id)
        compaction_completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(compaction_completions) == 3
        assert [
            event.payload["usage_metrics"]["input_tokens"] for event in compaction_completions[:2]
        ] == [1, 2]
        assert compaction_completions[2].payload["compaction_outcome"] == "cancelled"
        assert compaction_completions[2].payload["usage_unavailable_reason"]
        failures = [event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED]
        assert len(failures) == 1
        assert failures[0].payload["chunk_count"] == 3

    asyncio.run(run())


def test_automatic_compaction_stream_abandonment_preserves_success_accounting():
    async def run() -> None:
        session_id = "sess_compaction_success_stream_abandonment"
        store = InMemorySessionStore()
        compaction_provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "summary-model",
                        "usage": {"input_tokens": 5, "output_tokens": 2},
                    }
                ),
            ]
        )
        main_provider = FakeProvider([ModelStreamEvent.completed({})])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(main_provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        stream = app.run(
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
        while (await anext(stream)).type != EventType.CONTEXT_COMPACTION_STARTED:
            pass
        await stream.aclose()

        assert main_provider.requests == []
        events = await store.load_events(session_id)
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 7
        assert sum(event.type == EventType.CONTEXT_COMPACTION_COMPLETED for event in events) == 1
        assert sum(event.type == EventType.SESSION_CHECKPOINTED for event in events) == 1
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 2

    asyncio.run(run())


def test_automatic_compaction_stream_abandonment_after_budget_event_keeps_outcome():
    async def run() -> None:
        session_id = "sess_compaction_budget_stream_abandonment"
        store = InMemorySessionStore()
        compaction_provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "summary-model",
                        "usage": {"input_tokens": 5, "output_tokens": 2},
                    }
                ),
            ]
        )
        main_provider = FakeProvider([ModelStreamEvent.completed({})])
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=compaction_price_book(),
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=0,
                        ),
                    ),
                )
            ),
        )
        app.register_provider(main_provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        stream = app.run(
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
        while (await anext(stream)).type != EventType.BUDGET_RESERVED:
            pass
        await stream.aclose()

        assert len(compaction_provider.requests) == 1
        assert main_provider.requests == []
        events = await store.load_events(session_id)
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 7
        assert sum(event.type == EventType.BUDGET_RECONCILED for event in events) == 1
        assert sum(event.type == EventType.CONTEXT_COMPACTION_COMPLETED for event in events) == 1
        assert sum(event.type == EventType.SESSION_CHECKPOINTED for event in events) == 1
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 2

    asyncio.run(run())


def test_automatic_compaction_stream_abandonment_preserves_failure_accounting():
    class FailSecondHierarchyCallProvider(ModelProvider):
        name = "fail-second-hierarchy-call"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("second hierarchy call failed")
            yield ModelStreamEvent.text_delta("first fragment summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                }
            )

    async def run() -> None:
        session_id = "sess_compaction_failure_stream_abandonment"
        store = InMemorySessionStore()
        compaction_provider = FailSecondHierarchyCallProvider()
        main_provider = FakeProvider([ModelStreamEvent.completed({})])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(main_provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[
                    Message.text("user", "oversized " + "x" * 5000),
                    Message.text("user", "current"),
                ],
            )
        )
        while (await anext(stream)).type != EventType.CONTEXT_COMPACTION_STARTED:
            pass
        await stream.aclose()

        assert main_provider.requests == []
        events = await store.load_events(session_id)
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 2
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 7
        assert completions[1].payload["compaction_outcome"] == "unfinished_stream"
        assert completions[1].payload["usage_unavailable_reason"]
        assert sum(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in events) == 1
        assert EventType.SESSION_CHECKPOINTED not in {event.type for event in events}
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is None or "context_compaction" not in checkpoint

    asyncio.run(run())


def test_compaction_provider_cancellation_preserves_historical_task_cancellation() -> None:
    class CancellingProvider(ModelProvider):
        name = "provider-cancellation"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            yield ModelStreamEvent.text_delta("partial summary")
            raise asyncio.CancelledError("provider cancellation")

    async def run() -> None:
        current = asyncio.current_task()
        assert current is not None
        current.cancel("historical cancellation")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)
        try:
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await ModelCompactor(
                    provider=CancellingProvider(),
                    model="summary-model",
                ).compact(
                    CompactionRequest(
                        session=_test_session(),
                        agent=AgentSpec(name="assistant", model="fake-model"),
                        messages=[Message.text("user", "old request")],
                    )
                )
            assert exc_info.value.args == ("provider cancellation",)
            assert current.cancelling() == 1
        finally:
            current.uncancel()

    asyncio.run(run())


def test_automatic_compaction_termination_cleanup_preserves_historical_cancellation() -> None:
    class CancellingProvider(ModelProvider):
        name = "automatic-provider-cancellation"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            yield ModelStreamEvent.text_delta("partial summary")
            raise asyncio.CancelledError("provider cancellation")

    async def run() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=CancellingProvider(),
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        current = asyncio.current_task()
        assert current is not None
        current.cancel("historical cancellation")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)
        assert current.cancelling() == 1
        try:
            with pytest.raises(asyncio.CancelledError, match="provider cancellation"):
                await collect_events(
                    app,
                    RunRequest(
                        agent_name="assistant",
                        session_id="sess_termination_historical_cancellation",
                        messages=[
                            Message.text("user", "old"),
                            Message.text("assistant", "old answer"),
                            Message.text("user", "current"),
                        ],
                    ),
                )
            assert current.cancelling() == 1
        finally:
            current.uncancel()

        events = await store.load_events("sess_termination_historical_cancellation")
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert completions[0].payload["compaction_outcome"] == "cancelled"

    asyncio.run(run())


def test_overflow_compaction_termination_cleanup_preserves_historical_cancellation() -> None:
    class CancellingProvider(ModelProvider):
        name = "overflow-provider-cancellation"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            yield ModelStreamEvent.text_delta("partial summary")
            raise asyncio.CancelledError("overflow compactor cancellation")

    async def run() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(ContextOverflowProvider(), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_overflow_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=CancellingProvider(),
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        current = asyncio.current_task()
        assert current is not None
        current.cancel("historical cancellation")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)
        assert current.cancelling() == 1
        try:
            with pytest.raises(
                asyncio.CancelledError,
                match="overflow compactor cancellation",
            ):
                await collect_events(
                    app,
                    RunRequest(
                        agent_name="assistant",
                        session_id="sess_overflow_historical_cancellation",
                        messages=[
                            Message.text("user", "old"),
                            Message.text("user", "current"),
                        ],
                    ),
                )
            assert current.cancelling() == 1
        finally:
            current.uncancel()

        events = await store.load_events("sess_overflow_historical_cancellation")
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert completions[0].payload["compaction_outcome"] == "cancelled"

    asyncio.run(run())


def test_automatic_compaction_ignores_handled_historical_task_cancellation():
    async def run() -> None:
        session_id = "sess_compaction_after_historical_cancellation"
        store = InMemorySessionStore()
        compaction_provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "summary-model",
                        "usage": {"input_tokens": 5, "output_tokens": 2},
                    }
                ),
            ]
        )
        main_provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("answer"),
                ModelStreamEvent.completed({}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(main_provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )

        current = asyncio.current_task()
        assert current is not None
        current.cancel("historical cancellation")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await asyncio.sleep(0)
        assert exc_info.value.args == ("historical cancellation",)
        assert current.cancelling() == 1
        try:
            events = await collect_events(
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
            assert current.cancelling() == 1
        finally:
            current.uncancel()

        assert events[-1].type == EventType.SESSION_COMPLETED
        assert EventType.CONTEXT_COMPACTION_FAILED not in {event.type for event in events}
        assert len(compaction_provider.requests) == 1
        assert len(main_provider.requests) == 1

    asyncio.run(run())


def test_automatic_compaction_real_cancellation_during_start_publication_propagates():
    class BlockingCompactionStartStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.start_publication_started = asyncio.Event()
            self.allow_start_publication = asyncio.Event()

        async def append_events(self, session_id: str, events: list[Event]) -> None:
            if any(event.type == EventType.CONTEXT_COMPACTION_STARTED for event in events):
                self.start_publication_started.set()
                await self.allow_start_publication.wait()
            await super().append_events(session_id, events)

    async def run() -> None:
        session_id = "sess_cancel_compaction_start_publication"
        store = BlockingCompactionStartStore()
        compaction_provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("summary"),
                ModelStreamEvent.completed({}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        task = asyncio.create_task(
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
        await asyncio.wait_for(store.start_publication_started.wait(), timeout=10)
        task.cancel("cancel during compaction start publication")
        await asyncio.sleep(0)
        assert not task.done()
        store.allow_start_publication.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert exc_info.value.args == ("cancel during compaction start publication",)
        assert task.cancelled()
        assert task.cancelling() == 0
        assert compaction_provider.requests == []
        durable = await store.load_events(session_id)
        assert sum(event.type == EventType.CONTEXT_COMPACTION_STARTED for event in durable) == 1
        assert sum(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in durable) == 1

    asyncio.run(run())


def test_automatic_compaction_cancellation_during_outcome_persistence_is_lossless():
    class BlockingContextBatchStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.context_batch_started = asyncio.Event()
            self.allow_context_batch = asyncio.Event()

        async def complete_model_completion_stage(self, session_id: str, **kwargs):
            publication = kwargs["publication"]
            if any(
                event.type == EventType.MODEL_COMPLETED
                and event.payload.get("purpose") == "context_compaction"
                for event in publication.events
            ):
                self.context_batch_started.set()
                await self.allow_context_batch.wait()
            return await super().complete_model_completion_stage(session_id, **kwargs)

    async def run() -> None:
        session_id = "sess_compaction_outcome_persistence_cancellation"
        store = BlockingContextBatchStore()
        compaction_provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "summary-model",
                        "usage": {"input_tokens": 5, "output_tokens": 2},
                    }
                ),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        task = asyncio.create_task(
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
        await asyncio.wait_for(store.context_batch_started.wait(), timeout=10)
        task.cancel("cancel during context outcome persistence")
        assert task.cancelling() == 1
        store.allow_context_batch.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert exc_info.value.args == ("cancel during context outcome persistence",)
        assert task.cancelled()
        events = await store.load_events(session_id)
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 7
        assert sum(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in events) == 1
        assert EventType.CONTEXT_COMPACTION_COMPLETED not in {event.type for event in events}
        assert EventType.SESSION_CHECKPOINTED not in {event.type for event in events}
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is None or "context_compaction" not in checkpoint

    asyncio.run(run())


@pytest.mark.parametrize("failure_point", ["before_commit", "after_commit"])
def test_automatic_compaction_checkpoint_and_evidence_publish_atomically(
    failure_point: str,
) -> None:
    class FailingAtomicCheckpointStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def publish_checkpoint_and_events(self, session_id: str, **kwargs):
            events = kwargs["events"]
            is_compaction_checkpoint = any(
                event.type == EventType.SESSION_CHECKPOINTED for event in events
            )
            if not is_compaction_checkpoint or self.failed:
                return await super().publish_checkpoint_and_events(session_id, **kwargs)
            self.failed = True
            if failure_point == "before_commit":
                raise ConnectionError("compaction checkpoint rejected before commit")
            await super().publish_checkpoint_and_events(session_id, **kwargs)
            raise ConnectionError("compaction checkpoint acknowledgement lost")

    async def run() -> None:
        session_id = f"sess_atomic_compaction_checkpoint_{failure_point}"
        store = FailingAtomicCheckpointStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=FakeProvider(
                        [
                            ModelStreamEvent.text_delta("summary"),
                            ModelStreamEvent.completed(
                                {
                                    "model": "summary-model",
                                    "usage": {"input_tokens": 5, "output_tokens": 2},
                                }
                            ),
                        ]
                    ),
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )

        returned = await collect_events(
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
        durable = await store.load_events(session_id)
        checkpoint = await store.load_checkpoint(session_id)
        completed_count = sum(
            event.type == EventType.CONTEXT_COMPACTION_COMPLETED for event in durable
        )
        checkpointed_count = sum(event.type == EventType.SESSION_CHECKPOINTED for event in durable)

        if failure_point == "after_commit":
            assert returned[-1].type == EventType.SESSION_COMPLETED
            assert completed_count == 1
            assert checkpointed_count == 1
            assert checkpoint is not None
            assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 2
        else:
            assert returned[-1].type == EventType.SESSION_FAILED
            assert completed_count == 0
            assert checkpointed_count == 0
            assert checkpoint is None or "context_compaction" not in checkpoint

        # Provider usage is committed before the atomic checkpoint publication
        # attempt and is never duplicated by either failure branch.
        completions = [
            event
            for event in durable
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1

    asyncio.run(run())


def test_automatic_compaction_lost_checkpoint_ack_reconciles_effective_transform() -> None:
    class RuntimeStateChangingLostAckStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def publish_checkpoint_and_events(self, session_id: str, **kwargs):
            events = kwargs["events"]
            is_compaction_checkpoint = any(
                event.type == EventType.SESSION_CHECKPOINTED for event in events
            )
            if not is_compaction_checkpoint or self.failed:
                return await super().publish_checkpoint_and_events(session_id, **kwargs)
            self.failed = True

            def clear_runtime_owned_state(
                _session: Session,
                checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any]:
                updated = {} if checkpoint is None else dict(checkpoint)
                updated.pop("session_operations", None)
                return updated

            # Simulate a concurrent runtime-owned checkpoint update after the
            # context build read its source checkpoint but before the atomic
            # compaction publication applies its preserving transform.
            await super().transform_checkpoint(session_id, clear_runtime_owned_state)
            await super().publish_checkpoint_and_events(session_id, **kwargs)
            raise ConnectionError("compaction checkpoint acknowledgement lost")

    async def run() -> None:
        session_id = "sess_compaction_effective_transform_reconciliation"
        store = RuntimeStateChangingLostAckStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            FakeProvider(
                [
                    ModelStreamEvent.text_delta("final answer"),
                    ModelStreamEvent.completed({}),
                ]
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id=session_id,
                messages=[],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=app,
            ),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(session_id, transcript)

        def seed_runtime_owned_state(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            updated = {} if checkpoint is None else dict(checkpoint)
            updated["session_operations"] = {
                "version": 1,
                "active_operation_id": None,
                "records": {},
            }
            return updated

        await store.transform_checkpoint(session_id, seed_runtime_owned_state)
        await store.update_status(session_id, SessionStatus.COMPLETED)

        events = await collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "next")],
            ),
        )

        assert events[-1].type == EventType.SESSION_COMPLETED
        assert sum(event.type == EventType.SESSION_CHECKPOINTED for event in events) == 1
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "session_operations" not in checkpoint
        assert checkpoint["context_compaction"]["compacted_transcript_cursor"] > 0

    asyncio.run(run())


def test_automatic_compaction_internal_persistence_cancellation_is_not_caller_cancel() -> None:
    class InternallyCancelledCheckpointStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def publish_checkpoint_and_events(self, session_id: str, **kwargs):
            del session_id, kwargs
            raise asyncio.CancelledError("storage pool cancelled its write")

    async def run() -> None:
        session_id = "sess_compaction_internal_persistence_cancel"
        store = InternallyCancelledCheckpointStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )

        task = asyncio.create_task(
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
        events = await task

        assert not task.cancelled()
        assert events[-1].type == EventType.SESSION_FAILED
        assert events[-1].payload["error_type"] == "RuntimeError"
        assert events[-1].payload["error"] == (
            "Context outcome persistence was cancelled without caller cancellation."
        )
        assert task.exception() is None
        assert checkpoint_without_active_invocation_profile(
            await store.load_checkpoint(session_id)
        ) == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}

    asyncio.run(run())


def test_automatic_compaction_lost_completion_ack_fails_closed_without_retry() -> None:
    class LostCompletionAcknowledgementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def promote_model_completion_stage(
            self,
            session_id: str,
            *,
            stage_id: str,
            expected_run_epoch: int,
        ):
            result = await super().promote_model_completion_stage(
                session_id,
                stage_id=stage_id,
                expected_run_epoch=expected_run_epoch,
            )
            if not self.failed:
                self.failed = True
                raise ConnectionError("compaction completion acknowledgement lost")
            return result

    class RetryableFailureProvider(ModelProvider):
        name = "retryable-compaction"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            raise ModelProviderError(
                "provider overloaded",
                provider=self.name,
                status_code=503,
                retryable=True,
            )
            yield

    async def run() -> None:
        session_id = "sess_compaction_completion_lost_ack"
        store = LostCompletionAcknowledgementStore()
        provider = RetryableFailureProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )

        returned = await collect_events(
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
        durable = await store.load_events(session_id)

        assert len(provider.requests) == 1
        completions = [
            event
            for event in durable
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert completions[0].payload["compaction_outcome"] == "provider_error"
        failures = [event for event in durable if event.type == EventType.CONTEXT_COMPACTION_FAILED]
        assert len(failures) == 1
        expected = {
            "phase": "completion_publication",
            "reason": "publication_failed",
            "retryable": False,
            "provider_dispatch_disposition": "unknown",
            "recovery_action": "reconcile_completion",
        }
        assert {key: failures[0].payload[key] for key in expected} == expected
        assert returned[-1].type == EventType.SESSION_FAILED
        assert {
            key: returned[-1].payload["compaction_failure"][key] for key in expected
        } == expected

    asyncio.run(run())


def test_automatic_compaction_rejects_unattributed_custom_telemetry_without_leaking():
    class AdversarialAutomaticCompactor(ContextCompactor):
        def provider_budget_identity(self, _session: Session) -> None:
            return None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            identity = BillingIdentity(
                provider_name="adversarial-compactor",
                resource_id="summary-model-v1",
            )
            return CompactionResult(
                summary="private summary",
                covered_message_count=len(request.messages),
                source_chunk_mode="vendor-private-mode",
                metadata={
                    "instructions": "metadata instruction secret",
                    "provider_state": {"secret": "metadata provider secret"},
                },
                model_completed_payloads=[
                    {
                        "provider_name": "adversarial-compactor",
                        "requested_model": "summary-model",
                        "model": "summary-model-v1",
                        "billing_identity": identity.model_dump(mode="json"),
                        "usage_metrics": {
                            "input_tokens": 8,
                            "output_tokens": 2,
                            "total_tokens": 10,
                            "provider_state": {"secret": "nested metrics secret"},
                            "arbitrary": "q" * 100_000,
                        },
                        "summary": "model event summary secret",
                        "instructions": "model event instruction secret",
                        "provider_state": {"secret": "opaque continuation secret"},
                        "arbitrary": "p" * 100_000,
                    }
                ],
            )

    async def run() -> None:
        session_id = "sess_automatic_adversarial_compaction_telemetry"
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=AdversarialAutomaticCompactor(),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        await collect_events(
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
        events = await store.load_events(session_id)
        assert not any(
            event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
            for event in events
        )
        assert not any(event.type == EventType.CONTEXT_COMPACTION_COMPLETED for event in events)
        assert events[-1].type == EventType.SESSION_FAILED
        assert (
            events[-1].payload["error"]
            == "Compaction completion was observed outside its provider dispatch."
        )
        serialized = json.dumps(
            [event.model_dump(mode="json") for event in events],
            ensure_ascii=False,
        )
        for secret in (
            "metadata instruction secret",
            "metadata provider secret",
            "model event summary secret",
            "model event instruction secret",
            "opaque continuation secret",
            "nested metrics secret",
            "p" * 1000,
            "q" * 1000,
            "vendor-private-mode",
        ):
            assert secret not in serialized

    asyncio.run(run())


def test_compaction_telemetry_rejects_conflicting_billing_identities():
    event_identity = BillingIdentity(
        provider_name="event-provider",
        resource_id="event-model",
    )
    metrics_identity = BillingIdentity(
        provider_name="metrics-provider",
        resource_id="metrics-model",
    )
    telemetry = runtime_context_module.ContextCompactionTelemetry(
        event_type=EventType.MODEL_COMPLETED,
        payload={
            "billing_identity": event_identity.model_dump(mode="json"),
            "usage_metrics": {
                "billing_identity": metrics_identity.model_dump(mode="json"),
                "input_tokens": 8,
                "output_tokens": 2,
                "total_tokens": 10,
            },
        },
    )

    with pytest.raises(ValueError, match="billing identities do not match"):
        runtime_context_module.sanitize_context_compaction_telemetry(telemetry)


@pytest.mark.parametrize(
    "payload",
    [
        {"billing_identity": "not-an-object"},
        {
            "usage_metrics": {
                "billing_identity": {"provider_name": "missing-resource-id"},
                "input_tokens": 1,
            }
        },
    ],
)
def test_compaction_telemetry_rejects_malformed_billing_identity(payload):
    telemetry = runtime_context_module.ContextCompactionTelemetry(
        event_type=EventType.MODEL_COMPLETED,
        payload=payload,
    )

    with pytest.raises(ValueError, match="billing identity"):
        runtime_context_module.sanitize_context_compaction_telemetry(telemetry)


def test_compaction_telemetry_preserves_exact_maximum_usage_counter():
    maximum = runtime_context_module._COMPACTION_EVENT_INTEGER_MAX
    sanitized = runtime_context_module.sanitize_context_compaction_telemetry(
        runtime_context_module.ContextCompactionTelemetry(
            event_type=EventType.MODEL_COMPLETED,
            payload={
                "provider_name": "provider",
                "model": "model",
                "usage_metrics": {"input_tokens": maximum},
            },
        )
    )

    assert sanitized.payload["usage_metrics"]["input_tokens"] == maximum
    assert "usage_unavailable_reason" not in sanitized.payload


def test_compaction_telemetry_does_not_turn_empty_metrics_into_zero_usage():
    sanitized = runtime_context_module.sanitize_context_compaction_telemetry(
        runtime_context_module.ContextCompactionTelemetry(
            event_type=EventType.MODEL_COMPLETED,
            payload={
                "provider_name": "provider",
                "model": "model",
                "usage_metrics": {},
            },
        )
    )

    assert "usage_metrics" not in sanitized.payload


def test_compaction_telemetry_preserves_identity_from_empty_metrics():
    identity = BillingIdentity(provider_name="provider", resource_id="model")
    sanitized = runtime_context_module.sanitize_context_compaction_telemetry(
        runtime_context_module.ContextCompactionTelemetry(
            event_type=EventType.MODEL_COMPLETED,
            payload={
                "usage_metrics": {
                    "billing_identity": identity.model_dump(mode="json"),
                },
            },
        )
    )

    assert "usage_metrics" not in sanitized.payload
    assert sanitized.payload["billing_identity"] == identity.model_dump(mode="json")


def test_compaction_telemetry_preserves_provider_model_from_empty_metrics():
    sanitized = runtime_context_module.sanitize_context_compaction_telemetry(
        runtime_context_module.ContextCompactionTelemetry(
            event_type=EventType.MODEL_COMPLETED,
            payload={
                "usage_metrics": {
                    "provider_name": "nested-provider",
                    "requested_model": "requested-model",
                    "model": "served-model",
                },
            },
        )
    )

    assert "usage_metrics" not in sanitized.payload
    assert sanitized.payload["provider_name"] == "nested-provider"
    assert sanitized.payload["requested_model"] == "requested-model"
    assert sanitized.payload["model"] == "served-model"


def test_compaction_telemetry_rejects_conflicting_identity_with_empty_metrics():
    event_identity = BillingIdentity(provider_name="event-provider", resource_id="event-model")
    metrics_identity = BillingIdentity(
        provider_name="metrics-provider",
        resource_id="metrics-model",
    )
    telemetry = runtime_context_module.ContextCompactionTelemetry(
        event_type=EventType.MODEL_COMPLETED,
        payload={
            "billing_identity": event_identity.model_dump(mode="json"),
            "usage_metrics": {
                "billing_identity": metrics_identity.model_dump(mode="json"),
            },
        },
    )

    with pytest.raises(ValueError, match="billing identities do not match"):
        runtime_context_module.sanitize_context_compaction_telemetry(telemetry)


@pytest.mark.parametrize(
    "usage_payload",
    [
        {"usage_metrics": {"input_tokens": 2**63}},
        {"usage_metrics": {"input_tokens": -1}},
        {"usage_metrics": {"input_tokens": True}},
        {"usage_metrics": {"cache": {"read_tokens": "1"}}},
        {"usage_metrics_rejected": True},
        {"usage": {"input_tokens": 2**63}},
        {
            "usage": {
                "cache_creation": {
                    "ephemeral_5m": 2**63 - 1,
                    "ephemeral_1h": 1,
                }
            }
        },
    ],
)
def test_compaction_telemetry_marks_present_invalid_usage_unavailable(usage_payload):
    sanitized = runtime_context_module.sanitize_context_compaction_telemetry(
        runtime_context_module.ContextCompactionTelemetry(
            event_type=EventType.MODEL_COMPLETED,
            payload={
                "provider_name": "provider",
                "model": "model",
                **usage_payload,
            },
        )
    )

    assert "usage" not in sanitized.payload
    assert "usage_metrics" not in sanitized.payload
    assert sanitized.payload["usage_unavailable_reason"] == ("invalid compaction usage telemetry")


def test_automatic_compaction_rejects_conflicting_billing_identity_before_checkpoint():
    event_identity = BillingIdentity(
        provider_name="event-provider",
        resource_id="event-model",
    )
    metrics_identity = BillingIdentity(
        provider_name="metrics-provider",
        resource_id="metrics-model",
    )

    class ConflictingIdentityCompactor(ContextCompactor):
        def provider_budget_identity(self, _session: Session) -> None:
            return None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            return CompactionResult(
                summary="must not be checkpointed",
                covered_message_count=len(request.messages),
                model_completed_payloads=[
                    {
                        "billing_identity": event_identity.model_dump(mode="json"),
                        "usage_metrics": {
                            "billing_identity": metrics_identity.model_dump(mode="json"),
                            "input_tokens": 8,
                            "output_tokens": 2,
                            "total_tokens": 10,
                        },
                    }
                ],
            )

    async def run() -> None:
        session_id = "sess_conflicting_compaction_billing_identity"
        store = InMemorySessionStore()
        provider = FakeProvider([ModelStreamEvent.completed({})])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ConflictingIdentityCompactor(),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )

        events = await collect_events(
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

        assert provider.requests == []
        assert EventType.MODEL_COMPLETED not in {event.type for event in events}
        assert EventType.CONTEXT_COMPACTION_COMPLETED not in {event.type for event in events}
        assert events[-1].type == EventType.SESSION_FAILED
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is None or "context_compaction" not in checkpoint

    asyncio.run(run())


def test_hierarchical_compaction_call_limit_fails_before_unbounded_dispatch():
    provider = FakeProvider([])
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        max_input_chars=1000,
        max_hierarchy_calls=2,
    )
    with pytest.raises(ValueError, match="max_hierarchy_calls before dispatch"):
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "oversized " + "x" * 5000)],
                )
            )
        )
    assert provider.requests == []


def test_hierarchical_compaction_retry_consumes_the_actual_dispatch_limit():
    class RetryThenCompleteProvider(ModelProvider):
        name = "retry-then-complete"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                raise ModelProviderError(
                    "retry first hierarchy dispatch",
                    provider=self.name,
                    status_code=503,
                    retryable=True,
                )
            yield ModelStreamEvent.text_delta("first fragment summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                }
            )

    provider = RetryThenCompleteProvider()
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        max_input_chars=1000,
        max_hierarchy_calls=3,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
    )

    with pytest.raises(ValueError, match="exceeded max_hierarchy_calls"):
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "oversized " + "x" * 1100)],
                )
            )
        )

    # The retry is a real provider dispatch. The final logical hierarchy call is
    # rejected before it can become an uncounted fourth request.
    assert len(provider.requests) == 3


def test_hierarchical_compaction_preserves_instructions_within_every_bounded_prompt():
    completed_batch = [
        ModelStreamEvent.text_delta("short partial summary"),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]
    provider = FakeProvider([completed_batch for _index in range(16)])
    instruction = "INSTRUCTION_MARKER: retain exact identifiers."
    result = asyncio.run(
        ModelCompactor(
            provider=provider,
            model="summary-model",
            max_input_chars=1000,
        ).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "oversized " + "x" * 5000)],
                instructions=instruction,
            )
        )
    )
    prompts = [request.messages[-1].content[0].text for request in provider.requests]
    assert result.covered_message_count == 1
    assert len(prompts) > 1
    assert all(instruction in prompt for prompt in prompts)
    assert all(len(prompt) <= 1000 for prompt in prompts)


def test_hierarchical_compaction_fails_closed_when_instructions_leave_no_source_capacity():
    provider = FakeProvider([])
    with pytest.raises(
        ValueError,
        match="max_input_chars is too small for hierarchical compaction",
    ):
        asyncio.run(
            ModelCompactor(
                provider=provider,
                model="summary-model",
                max_input_chars=1000,
            ).compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "oversized " + "x" * 5000)],
                    instructions="i" * 1000,
                )
            )
        )
    assert provider.requests == []


def test_hierarchical_compaction_validates_merge_capacity_before_dispatch():
    provider = FakeProvider([])
    with pytest.raises(
        ValueError,
        match="max_input_chars is too small for hierarchical assembly",
    ):
        asyncio.run(
            ModelCompactor(
                provider=provider,
                model="summary-model",
                max_input_chars=1000,
            ).compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "x" * 100)],
                    instructions="i" * 700,
                )
            )
        )
    assert provider.requests == []


def test_hierarchical_compaction_preflights_existing_summary_call_capacity():
    provider = FakeProvider([])
    with pytest.raises(
        ValueError,
        match="max_hierarchy_calls before dispatch",
    ):
        asyncio.run(
            ModelCompactor(
                provider=provider,
                model="summary-model",
                max_input_chars=1000,
                max_hierarchy_calls=2,
            ).compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "small new source")],
                    existing_summary="prior " + "x" * 5000,
                )
            )
        )
    assert provider.requests == []


def test_hierarchical_compaction_preflights_every_existing_summary_merge_level():
    provider = FakeProvider([])
    with pytest.raises(
        ValueError,
        match="max_hierarchy_calls before dispatch",
    ):
        asyncio.run(
            ModelCompactor(
                provider=provider,
                model="summary-model",
                max_input_chars=1000,
                # The first merge level needs 70 calls. One source call plus
                # one assumed final call used to make 72 appear sufficient,
                # even though the optimistic second level still needs two
                # calls before the actual final merge.
                max_hierarchy_calls=72,
            ).compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "small new source")],
                    existing_summary="prior " + "x" * 50_000,
                )
            )
        )
    assert provider.requests == []


def test_hierarchical_compaction_merges_existing_summary_exactly_once():
    class MarkerPreservingProvider(ModelProvider):
        name = "marker-preserving"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            prompt = request.messages[-1].content[0].text
            markers = [
                marker for marker in ("PRIOR_RANGE_MARKER", "NEW_RANGE_MARKER") if marker in prompt
            ]
            yield ModelStreamEvent.text_delta(" ".join(markers) or "fragment")
            yield ModelStreamEvent.completed({})

    provider = MarkerPreservingProvider()
    result = asyncio.run(
        ModelCompactor(
            provider=provider,
            model="summary-model",
            max_input_chars=1000,
        ).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[
                    Message.text("user", "NEW_RANGE_MARKER " + "x" * 4000),
                    Message.text("assistant", "unacknowledged suffix"),
                ],
                existing_summary="PRIOR_RANGE_MARKER",
            )
        )
    )
    assert result.covered_message_count == 1
    assert result.summary.count("PRIOR_RANGE_MARKER") == 1
    assert result.summary.count("NEW_RANGE_MARKER") == 1
    assert all(
        "unacknowledged suffix" not in request.messages[-1].content[0].text
        for request in provider.requests
    )


def test_hierarchical_compaction_allows_initial_expansion_then_current_pass_shrinkage():
    class ExpandThenShrinkProvider(ModelProvider):
        name = "expand-then-shrink"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.calls += 1
            prompt = request.messages[-1].content[0].text
            if prompt.startswith("Summarize this ordered fragment"):
                summary = "L" * 1500
            elif "L" * 100 in prompt:
                summary = "M" * 600
            elif "M" * 100 in prompt:
                summary = "N" * 300
            elif "N" * 100 in prompt:
                summary = "reduced"
            else:
                summary = "final"
            yield ModelStreamEvent.text_delta(summary)
            yield ModelStreamEvent.completed({})

    provider = ExpandThenShrinkProvider()
    result = asyncio.run(
        ModelCompactor(
            provider=provider,
            model="summary-model",
            max_input_chars=1000,
        ).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "oversized " + "x" * 1500)],
            )
        )
    )

    assert result.covered_message_count == 1
    assert result.summary == "final"
    assert provider.calls > result.source_chunk_count


def test_hierarchical_failure_reports_completed_chunks_without_checkpoint():
    class FailSecondHierarchyCallProvider(ModelProvider):
        name = "fail-hierarchy"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.calls += 1
            if self.calls == 2:
                yield ModelStreamEvent.error("second hierarchy fragment failed")
                return
            yield ModelStreamEvent.text_delta("first fragment summary")
            yield ModelStreamEvent.completed({"usage": {"input_tokens": 5, "output_tokens": 2}})

    provider = FailSecondHierarchyCallProvider()
    policy = CheckpointCompactionContextPolicy(
        compactor=ModelCompactor(
            provider=provider,
            model="summary-model",
            max_input_chars=1000,
        ),
        max_user_turns=1,
        compact_after_messages=1,
    )
    with pytest.raises(ContextBuildError, match="second hierarchy fragment failed") as exc_info:
        asyncio.run(
            policy.build_with_checkpoint(
                ContextRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[
                        Message.text("user", "oversized " + "x" * 5000),
                        Message.text("user", "current"),
                    ],
                    step=1,
                    force_compaction=True,
                ),
                checkpoint=None,
            )
        )
    event_types = [item.event_type for item in exc_info.value.compaction_telemetry]
    assert event_types == [
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
    ]
    assert exc_info.value.compaction_telemetry[0].payload["bounded_input"] is True
    uncertain = exc_info.value.compaction_telemetry[2].payload
    assert uncertain["compaction_outcome"] == "unfinished_stream"
    assert uncertain["usage_unavailable_reason"]
    failure = exc_info.value.compaction_telemetry[-1].payload
    assert failure["represented_message_count"] == 0
    assert failure["chunk_count"] == 2
    assert failure["chunk_mode"] == "failed"
    assert failure["bounded_input"] is True


def test_model_compactor_uses_bounded_default_prompt():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("default summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        max_input_chars=1000,
    )

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "uncovered marker " + "x" * 2000),
                ],
            )
        )
    )

    assert result.summary == "default summary"
    assert result.covered_message_count == 1
    prompt = provider.requests[0].messages[1].content[0].text
    assert "Transcript to compact:\nuser: old request" in prompt
    assert "uncovered marker" not in prompt


def test_prompt_cache_compactor_sends_context_messages_as_prefix():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("session cache "),
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed({"usage": {"output_tokens": 3}}),
        ]
    )
    compactor = PromptCacheCompactor(provider=provider)

    context = [
        Message.text("system", "You are helpful."),
        Message.text("user", "first request"),
        Message.text("assistant", "first answer"),
        Message.text("user", "second request"),
        Message.text("assistant", "second answer"),
    ]
    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old request")],
                context_messages=context,
                cache_prefix_request=ModelRequest(model="fake-model", messages=context),
            )
        )
    )

    assert result.summary == "session cache summary"
    assert result.metadata["compactor"] == "PromptCacheCompactor"
    assert result.metadata["provider"] == "fake"
    assert result.metadata["model"] == "fake-model"
    assert result.metadata["prompt_cache_compaction"] is True
    assert result.metadata["context_message_count"] == 5
    assert result.metadata["attachment_results_preserved"] == 0

    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.model == "fake-model"
    assert request.tools == []
    assert len(request.messages) == len(context) + 1
    for i, original in enumerate(context):
        assert request.messages[i].role == original.role
        assert request.messages[i].content[0].text == original.content[0].text
    instruction_message = request.messages[-1]
    assert instruction_message.role == MessageRole.USER
    assert "Summarize the conversation above" in instruction_message.content[0].text


def test_prompt_cache_compactor_falls_back_when_context_messages_empty():
    provider = FakeProvider([ModelStreamEvent.text_delta("should not be called")])
    compactor = PromptCacheCompactor(provider=provider)

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old request")],
            )
        )
    )

    assert len(provider.requests) == 0
    assert "old request" in result.summary


def test_prompt_cache_compactor_uses_session_model_by_default():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    compactor = PromptCacheCompactor(provider=provider)
    session = Session(
        id="sess_model_test",
        agent_name="assistant",
        provider_name="fake",
        model="claude-sonnet-4-20260601",
        invocation=fixture_session_invocation("sess_model_test"),
    )

    asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=session,
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-20260601"),
                messages=[Message.text("user", "old")],
                context_messages=[Message.text("user", "hello")],
                cache_prefix_request=ModelRequest(
                    model="claude-sonnet-4-20260601",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
    )

    assert provider.requests[0].model == "claude-sonnet-4-20260601"


def test_prompt_cache_compactor_uses_model_override():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    compactor = PromptCacheCompactor(provider=provider, model="override-model")

    asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old")],
                context_messages=[Message.text("user", "hello")],
                cache_prefix_request=ModelRequest(
                    model="fake-model",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
    )

    assert provider.requests[0].model == "override-model"


def test_prompt_cache_compactor_marks_unpriced_exact_tool_call_attempt():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_1", name="echo", arguments={"text": "bad"}),
            ],
            [
                ModelStreamEvent.text_delta("bounded summary"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {"input_tokens": 20, "output_tokens": 5},
                    }
                ),
            ],
        ]
    )
    compactor = PromptCacheCompactor(provider=provider)

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old")],
                context_messages=[Message.text("user", "hello")],
                cache_prefix_request=ModelRequest(
                    model="fake-model",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
    )

    assert result.summary == "bounded summary"
    assert len(provider.requests) == 2
    assert result.metadata["prompt_cache_exact_attempt"] == "rejected_tool_call"
    assert result.model_completed_payloads[0]["compaction_outcome"] == ("rejected_tool_call")
    assert result.model_completed_payloads[0]["usage_unavailable_reason"] == (
        "compaction tool-call attempt ended without provider completion usage"
    )
    assert "usage_metrics" not in result.model_completed_payloads[0]
    assert result.model_completed_payloads[1]["usage_metrics"]["input_tokens"] == 20


def test_prompt_cache_compactor_bounds_exact_context_overflow_once():
    provider = ContextOverflowProvider(
        success_events=[
            ModelStreamEvent.text_delta("bounded summary"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 20, "output_tokens": 5},
                }
            ),
        ]
    )
    compactor = PromptCacheCompactor(provider=provider)
    exact_tool = {
        "name": "exact_only_tool",
        "description": "Only present on the cached request.",
        "input_schema": {"type": "object", "properties": {}},
    }
    session = Session(
        id="sess_prompt_cache_overflow",
        agent_name="assistant",
        provider_name=provider.name,
        model="fake-model",
        invocation=fixture_session_invocation("sess_prompt_cache_overflow"),
    )

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=session,
                agent=AgentSpec(
                    name="assistant",
                    model="fake-model",
                    provider_options={
                        "bounded_only": {"enabled": True},
                        RESOLVED_FILE_ATTACHMENTS_OPTION: {"remove": True},
                    },
                ),
                messages=[Message.text("user", "bounded transcript only")],
                context_messages=[Message.text("user", "context projection")],
                cache_prefix_request=ModelRequest(
                    model="fake-model",
                    messages=[Message.text("user", "cached prefix only")],
                    tools=[exact_tool],
                    options={
                        "exact_only": {"enabled": True},
                        RESOLVED_FILE_ATTACHMENTS_OPTION: {"cached": True},
                    },
                ),
            )
        )
    )

    assert result.summary == "bounded summary"
    assert result.metadata["prompt_cache_exact_attempt"] == "context_overflow"
    assert len(provider.requests) == 2
    exact_request, bounded_request = provider.requests
    assert exact_request.tools == [exact_tool]
    assert exact_request.options["exact_only"] == {"enabled": True}
    assert RESOLVED_FILE_ATTACHMENTS_OPTION in exact_request.options
    assert bounded_request.tools == []
    assert bounded_request.options == {"bounded_only": {"enabled": True}}
    bounded_text = "\n".join(
        part.text
        for message in bounded_request.messages
        for part in message.content
        if isinstance(part, TextPart)
    )
    assert "bounded transcript only" in bounded_text
    assert "cached prefix only" not in bounded_text

    exact_attempt = result.model_completed_payloads[0]
    assert exact_attempt["compaction_outcome"] == "context_overflow"
    assert exact_attempt["context_overflow"] is True
    assert exact_attempt["error_type"] == "ModelContextOverflowError"
    assert exact_attempt["provider"] == provider.name
    assert exact_attempt["status_code"] == 400
    assert exact_attempt["provider_error_code"] == "context_length_exceeded"
    assert exact_attempt["retryable"] is False
    assert exact_attempt["usage_unavailable_reason"] == (
        "exact prompt-cache compaction overflowed without provider completion usage"
    )
    assert "usage" not in exact_attempt
    assert "usage_metrics" not in exact_attempt
    assert result.model_completed_payloads[1]["usage_metrics"]["input_tokens"] == 20


def test_prompt_cache_compactor_surfaces_bounded_failure_after_exact_overflow():
    provider = ContextOverflowProvider(
        success_events=[ModelStreamEvent.error("bounded compaction failed")]
    )
    compactor = PromptCacheCompactor(provider=provider)
    session = Session(
        id="sess_prompt_cache_bounded_failure",
        agent_name="assistant",
        provider_name=provider.name,
        model="fake-model",
        invocation=fixture_session_invocation("sess_prompt_cache_bounded_failure"),
    )

    with pytest.raises(RuntimeError, match="bounded compaction failed") as exc_info:
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=session,
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "bounded transcript")],
                    context_messages=[Message.text("user", "context projection")],
                    cache_prefix_request=ModelRequest(
                        model="fake-model",
                        messages=[Message.text("user", "cached prefix")],
                    ),
                )
            )
        )

    assert len(provider.requests) == 2
    assert type(exc_info.value) is RuntimeError
    assert str(exc_info.value) == "bounded compaction failed"


def test_prompt_cache_compactor_does_not_loop_when_bounded_attempt_also_overflows():
    provider = ContextOverflowProvider(overflow_requests=2)
    compactor = PromptCacheCompactor(provider=provider)
    session = Session(
        id="sess_prompt_cache_double_overflow",
        agent_name="assistant",
        provider_name=provider.name,
        model="fake-model",
        invocation=fixture_session_invocation("sess_prompt_cache_double_overflow"),
    )

    with pytest.raises(RuntimeError, match="context too large") as exc_info:
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=session,
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "bounded transcript")],
                    context_messages=[Message.text("user", "context projection")],
                    cache_prefix_request=ModelRequest(
                        model="fake-model",
                        messages=[Message.text("user", "cached prefix")],
                    ),
                )
            )
        )

    assert len(provider.requests) == 2
    assert isinstance(exc_info.value, ModelContextOverflowError)
    assert exc_info.value.error_code == "context_length_exceeded"


def test_prompt_cache_policy_deduplicates_exact_overflow_and_completes_with_bounded_result():
    overflow = ModelContextOverflowError(
        "context too large",
        provider="fake",
        status_code=400,
        error_code="context_length_exceeded",
    )
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
            [ModelStreamEvent.error(str(overflow), cause=overflow)],
            [
                ModelStreamEvent.text_delta("bounded summary"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 20, "output_tokens": 5}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 8, "output_tokens": 2}}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=PromptCacheCompactor(provider=provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_prompt_cache_policy_overflow_success",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    assert first_events[-1].type == EventType.SESSION_COMPLETED

    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_prompt_cache_policy_overflow_success",
                messages=[Message.text("user", "second request")],
            ),
        )
    )

    assert len(provider.requests) == 4
    exact_request, bounded_request = provider.requests[1:3]
    exact_text = "\n".join(
        part.text
        for message in exact_request.messages
        for part in message.content
        if isinstance(part, TextPart)
    )
    assert "first request" in exact_text
    assert "first answer" in exact_text
    assert len(exact_request.messages) > 2
    assert [message.role for message in bounded_request.messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
    ]
    assert bounded_request.tools == []

    compaction_attempts = [
        event
        for event in resume_events
        if event.type == EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    ]
    # The exact attempt is both recorded before fallback and returned by the compactor.
    # Its preserved internal attempt id must replace, rather than duplicate, that event.
    assert len(compaction_attempts) == 2
    assert (
        sum(
            event.payload.get("compaction_outcome") == "context_overflow"
            for event in compaction_attempts
        )
        == 1
    )
    assert compaction_attempts[0].payload["compaction_outcome"] == "context_overflow"
    assert "compaction_attempt_id" not in compaction_attempts[0].payload
    assert "usage_metrics" not in compaction_attempts[0].payload
    assert compaction_attempts[1].payload["usage_metrics"]["input_tokens"] == 20
    assert "compaction_attempt_id" not in compaction_attempts[1].payload
    completed = next(
        event for event in resume_events if event.type == EventType.CONTEXT_COMPACTION_COMPLETED
    )
    assert completed.payload["chunk_mode"] == "single_request"
    assert EventType.CONTEXT_COMPACTION_FAILED not in [event.type for event in resume_events]
    assert resume_events[-1].type == EventType.SESSION_COMPLETED
    usage = asyncio.run(app.get_session_usage("sess_prompt_cache_policy_overflow_success"))
    assert usage.model_steps == 4
    assert usage.usage.input_tokens == 38
    assert usage.usage.output_tokens == 9


def test_prompt_cache_compactor_records_exact_overflow_when_bounded_attempt_fails():
    overflow = ModelContextOverflowError(
        "context too large",
        provider="fake",
        status_code=400,
        error_code="context_length_exceeded",
    )
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
            [ModelStreamEvent.error(str(overflow), cause=overflow)],
            [ModelStreamEvent.error("bounded compaction failed")],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=PromptCacheCompactor(provider=provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_prompt_cache_failed_fallback_telemetry",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    assert first_events[-1].type == EventType.SESSION_COMPLETED

    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_prompt_cache_failed_fallback_telemetry",
                messages=[Message.text("user", "second request")],
            ),
        )
    )

    assert len(provider.requests) == 3
    compaction_events = [
        event
        for event in resume_events
        if event.type
        in {
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.CONTEXT_COMPACTION_FAILED,
        }
    ]
    assert [event.type for event in compaction_events] == [
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
    ]
    assert "bounded_input" not in compaction_events[0].payload
    exact_attempt = compaction_events[1]
    assert exact_attempt.payload["purpose"] == "context_compaction"
    assert exact_attempt.payload["compaction_outcome"] == "context_overflow"
    assert exact_attempt.payload["usage_unavailable_reason"] == (
        "exact prompt-cache compaction overflowed without provider completion usage"
    )
    assert "usage_metrics" not in exact_attempt.payload
    bounded_attempt = compaction_events[2]
    assert bounded_attempt.payload["compaction_outcome"] == "unfinished_stream"
    assert bounded_attempt.payload["usage_unavailable_reason"]
    assert "usage_metrics" not in bounded_attempt.payload
    failed_compaction = compaction_events[3]
    assert failed_compaction.payload["error_type"] == "RuntimeError"
    assert "error" not in failed_compaction.payload
    assert failed_compaction.payload["error_type"] == "RuntimeError"
    assert resume_events[-1].type == EventType.SESSION_FAILED
    assert resume_events[-1].payload["session_run_operation_id"] == PRIVATE_EVENT_AUTHORITY
    terminal_records = asyncio.run(
        app.session_store.query_events(
            EventQuery(
                session_id="sess_prompt_cache_failed_fallback_telemetry",
                event_type=EventType.SESSION_FAILED,
            )
        )
    )
    terminal_operation_id = terminal_records[-1].event.payload["session_run_operation_id"]
    assert str(UUID(terminal_operation_id)) == terminal_operation_id
    assert {
        key: value
        for key, value in resume_events[-1].payload.items()
        if key != "session_run_operation_id"
    } == {
        "compaction_failure": _expected_compaction_failure(
            failed_compaction,
            reason="provider_failed",
        ),
        "failure_evidence": _expected_failure_evidence(
            "sess_prompt_cache_failed_fallback_telemetry",
            "RuntimeError",
            "ContextBuildError",
            run_epoch=3,
        ),
        "error": "bounded compaction failed",
        "error_type": "RuntimeError",
    }
    usage = asyncio.run(app.get_session_usage("sess_prompt_cache_failed_fallback_telemetry"))
    assert usage.model_steps == 3
    cost = asyncio.run(
        app.get_session_cost(
            "sess_prompt_cache_failed_fallback_telemetry",
            PriceBook(
                prices=(
                    ModelPrice.fixed(
                        provider_name="fake",
                        model="fake-model",
                        input_per_million=Decimal("1"),
                        output_per_million=Decimal("10"),
                    ),
                )
            ),
        )
    )
    assert cost.model_steps == 3
    assert cost.priced_model_steps == 1
    assert cost.unpriced_model_steps == 2
    assert cost.line_items[0].priced is True
    for line_item in cost.line_items[1:]:
        assert line_item.priced is False
        assert line_item.total_cost == Decimal("0")
        assert line_item.missing_pricing_reason == (
            "model.completed event has no token usage metrics"
        )


def test_prompt_cache_compactor_preserves_attempt_order_after_recovered_invalid_summary():
    class RecoveringPromptCacheCompactor(PromptCacheCompactor):
        async def compact(self, request: CompactionRequest) -> CompactionResult:
            try:
                await ModelCompactor(
                    provider=self.provider,
                    model=request.session.model,
                ).compact(request)
            except ValueError:
                return await super().compact(request)
            raise AssertionError("the first compaction unexpectedly returned a valid summary")

    overflow = ModelContextOverflowError(
        "context too large",
        provider="fake",
        status_code=400,
        error_code="context_length_exceeded",
    )
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
            [ModelStreamEvent.completed({"usage": {"input_tokens": 15, "output_tokens": 0}})],
            [ModelStreamEvent.error(str(overflow), cause=overflow)],
            [
                ModelStreamEvent.text_delta(" \n\t"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 20, "output_tokens": 1}}),
            ],
        ]
    )
    app = CayuApp(request_footprint=RequestFootprintConfig(enabled=False))
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=RecoveringPromptCacheCompactor(provider=provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_prompt_cache_invalid_bounded_summary",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    assert first_events[-1].type == EventType.SESSION_COMPLETED

    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_prompt_cache_invalid_bounded_summary",
                messages=[Message.text("user", "second request")],
            ),
        )
    )

    compaction_events = [
        event
        for event in resume_events
        if event.type
        in {
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.CONTEXT_COMPACTION_FAILED,
        }
    ]
    assert [event.type for event in compaction_events] == [
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.MODEL_COMPLETED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
    ]
    recovered_attempt, exact_attempt, bounded_attempt = compaction_events[1:4]
    assert recovered_attempt.payload["compaction_outcome"] == "invalid_summary"
    assert recovered_attempt.payload["usage_metrics"]["input_tokens"] == 15
    assert exact_attempt.payload["compaction_outcome"] == "context_overflow"
    assert "usage_metrics" not in exact_attempt.payload
    assert bounded_attempt.payload["compaction_outcome"] == "invalid_summary"
    assert bounded_attempt.payload["usage_metrics"]["input_tokens"] == 20
    assert bounded_attempt.payload["usage_metrics"]["output_tokens"] == 1
    assert compaction_events[-1].payload["error_type"] == "ValueError"
    assert "error" not in compaction_events[-1].payload
    assert resume_events[-1].type == EventType.SESSION_FAILED


def test_prompt_cache_compactor_fails_on_provider_error():
    provider = FakeProvider([ModelStreamEvent.error("provider unavailable")])
    compactor = PromptCacheCompactor(provider=provider)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old")],
                    context_messages=[Message.text("user", "hello")],
                    cache_prefix_request=ModelRequest(
                        model="fake-model",
                        messages=[Message.text("user", "hello")],
                    ),
                )
            )
        )

    assert len(provider.requests) == 1


def test_prompt_cache_compactor_custom_instruction():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("custom done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    compactor = PromptCacheCompactor(
        provider=provider,
        compaction_instruction="Make it short.",
    )

    asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old")],
                context_messages=[Message.text("user", "hello")],
                cache_prefix_request=ModelRequest(
                    model="fake-model",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
    )

    instruction = provider.requests[0].messages[-1].content[0].text
    assert instruction == "Make it short."


def test_prompt_cache_compactor_custom_fallback():
    provider = FakeProvider([ModelStreamEvent.text_delta("should not be called")])
    recording = RecordingCompactor()
    compactor = PromptCacheCompactor(
        provider=provider,
        fallback_compactor=recording,
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

    assert len(provider.requests) == 0
    assert len(recording.requests) == 1
    assert "old request" in result.summary


def test_prompt_cache_compactor_merges_cached_request_options_with_overrides():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    compactor = PromptCacheCompactor(
        provider=provider,
        options={"anthropic": {"max_tokens": 512}, "custom": {"enabled": True}},
    )

    asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(
                    name="assistant",
                    model="fake-model",
                    provider_options={
                        "anthropic": {
                            "cache_control": {"type": "ephemeral", "ttl": "5m"},
                            "max_tokens": 2048,
                        },
                        "openai": {"prompt_cache_key": "agent-cache"},
                    },
                ),
                messages=[Message.text("user", "old")],
                context_messages=[Message.text("user", "hello")],
                cache_prefix_request=ModelRequest(
                    model="fake-model",
                    messages=[Message.text("user", "hello")],
                    options={
                        "anthropic": {
                            "cache_control": {"type": "ephemeral", "ttl": "5m"},
                            "max_tokens": 2048,
                        },
                        "openai": {"prompt_cache_key": "agent-cache"},
                    },
                ),
            )
        )
    )

    assert provider.requests[0].options == {
        "anthropic": {
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
            "max_tokens": 512,
        },
        "openai": {"prompt_cache_key": "agent-cache"},
        "custom": {"enabled": True},
    }


def test_prompt_cache_compactor_rejects_invalid_provider():
    with pytest.raises(TypeError, match="provider must be a ModelProvider"):
        PromptCacheCompactor(provider="not a provider")


def test_prompt_cache_compactor_rejects_invalid_fallback():
    provider = FakeProvider([])
    with pytest.raises(TypeError, match="fallback_compactor must be a ContextCompactor"):
        PromptCacheCompactor(provider=provider, fallback_compactor="not a compactor")


def test_checkpoint_compaction_passes_context_messages_to_compactor():
    store = InMemorySessionStore()
    compactor = RecordingCompactor()
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
        AgentSpec(
            name="assistant",
            model="fake-model",
            system_prompt="You are careful.",
        ),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_ctx_messages",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    assert any(event.type == EventType.SESSION_COMPLETED for event in first_events)

    second_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_ctx_messages",
                messages=[Message.text("user", "second request")],
            ),
        )
    )
    assert any(event.type == EventType.SESSION_COMPLETED for event in second_events)

    assert len(compactor.requests) >= 1
    last_request = compactor.requests[-1]
    assert len(last_request.context_messages) > 0
    context_roles = [m.role for m in last_request.context_messages]
    assert MessageRole.SYSTEM in context_roles
    assert MessageRole.USER in context_roles


def test_cayu_app_checkpoint_compacts_model_context_without_rewriting_transcript():
    store = InMemorySessionStore()
    compactor = RecordingCompactor()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            system_prompt="You are careful.",
        ),
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
                session_id="sess_compaction",
                messages=[
                    Message.text("user", "old one"),
                    Message.text("assistant", "old answer one"),
                    Message.text("user", "old two"),
                    Message.text("assistant", "old answer two"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[1].payload == {
        "checkpoint": "context_compaction",
        "execution_profile_fingerprint": events[4].payload["execution_profile_fingerprint"],
        "compactor": "RecordingCompactor",
        "compacted_transcript_cursor": 5,
        "previous_compacted_transcript_cursor": 1,
        "newly_compacted_message_count": 4,
        "recent_message_count": 1,
        "requested_source_start": 1,
        "requested_source_end": 5,
        "represented_source_start": 1,
        "represented_source_end": 1,
        "represented_message_count": 0,
        "coverage_mode": "pending",
        "chunk_count": 0,
        "model_step_id": events[4].payload["model_step_id"],
    }
    assert events[2].payload == {
        "checkpoint": "context_compaction",
        "execution_profile_fingerprint": events[4].payload["execution_profile_fingerprint"],
        "compactor": "RecordingCompactor",
        "compacted_transcript_cursor": 5,
        "previous_compacted_transcript_cursor": 1,
        "newly_compacted_message_count": 4,
        "recent_message_count": 1,
        "summary_chars": len("old one|old answer one|old two|old answer two"),
        "requested_source_start": 1,
        "requested_source_end": 5,
        "represented_source_start": 1,
        "represented_source_end": 5,
        "represented_message_count": 4,
        "coverage_mode": "full",
        "chunk_count": 1,
        "chunk_mode": "single_request",
        "bounded_input": False,
        "compaction_failed": False,
        "model_step_id": events[4].payload["model_step_id"],
    }
    assert "summary" not in events[2].payload
    assert events[3].payload == {
        "checkpoint": "context_compaction",
        "execution_profile_fingerprint": events[4].payload["execution_profile_fingerprint"],
        "compacted_transcript_cursor": 5,
        "previous_compacted_transcript_cursor": 1,
        "newly_compacted_message_count": 4,
        "recent_message_count": 1,
        "model_step_id": events[4].payload["model_step_id"],
    }
    assert len(compactor.requests) == 1
    assert [message.content[0].text for message in compactor.requests[0].messages] == [
        "old one",
        "old answer one",
        "old two",
        "old answer two",
    ]

    provider_context = provider.requests[0].messages
    assert [message.role for message in provider_context] == [
        "system",
        "user",
        "user",
    ]
    assert provider_context[1].content[0].text == (
        "Previous session context summary:\nold one|old answer one|old two|old answer two"
    )
    assert provider_context[2].content[0].text == "current"

    transcript = asyncio.run(store.load_transcript("sess_compaction"))
    assert [message.content[0].text for message in transcript] == [
        "You are careful.",
        "old one",
        "old answer one",
        "old two",
        "old answer two",
        "current",
        "final answer",
    ]
    checkpoint = asyncio.run(store.load_checkpoint("sess_compaction"))
    assert checkpoint_without_model_step_publication(checkpoint) == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "context_compaction": {
            "version": 2,
            "summary": "old one|old answer one|old two|old answer two",
            "compacted_transcript_cursor": 5,
            "metadata": {"request_count": 1},
        },
    }


def test_cayu_app_checkpoint_compaction_can_use_model_compactor():
    store = InMemorySessionStore()
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
                options={"anthropic": {"max_tokens": 512}},
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
                session_id="sess_model_compaction",
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
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(compactor_provider.requests) == 1
    assert compactor_provider.requests[0].model == "summary-model"
    assert compactor_provider.requests[0].tools == []
    assert compactor_provider.requests[0].options == {"anthropic": {"max_tokens": 512}}
    assert events[1].payload == {
        "checkpoint": "context_compaction",
        "execution_profile_fingerprint": events[6].payload["execution_profile_fingerprint"],
        "compactor": "ModelCompactor",
        "compacted_transcript_cursor": 2,
        "previous_compacted_transcript_cursor": 0,
        "newly_compacted_message_count": 2,
        "recent_message_count": 1,
        "requested_source_start": 0,
        "requested_source_end": 2,
        "represented_source_start": 0,
        "represented_source_end": 0,
        "represented_message_count": 0,
        "coverage_mode": "pending",
        "chunk_count": 0,
        "bounded_input": True,
        "model_step_id": events[6].payload["model_step_id"],
    }
    parent_started = asyncio.run(
        _private_events_for_public_events(store, "sess_model_compaction", [events[6]])
    )[0]
    assert events[3].payload == {
        "parent_model_step_id": parent_started.payload["model_step_id"],
        "finish_reason": "stop",
        "model": "summary-model",
        "provider_name": "fake",
        "requested_model": "summary-model",
        "purpose": "context_compaction",
        "compactor": "ModelCompactor",
        "model_step_id": events[6].payload["model_step_id"],
        "model_attempt_id": events[3].payload["model_attempt_id"],
        "execution_profile_fingerprint": events[6].payload["execution_profile_fingerprint"],
    }
    assert events[4].payload == {
        "checkpoint": "context_compaction",
        "execution_profile_fingerprint": events[6].payload["execution_profile_fingerprint"],
        "compactor": "ModelCompactor",
        "compacted_transcript_cursor": 2,
        "previous_compacted_transcript_cursor": 0,
        "newly_compacted_message_count": 2,
        "recent_message_count": 1,
        "summary_chars": len("model summary"),
        "requested_source_start": 0,
        "requested_source_end": 2,
        "represented_source_start": 0,
        "represented_source_end": 2,
        "represented_message_count": 2,
        "coverage_mode": "full",
        "chunk_count": 1,
        "chunk_mode": "single_request",
        "bounded_input": False,
        "compaction_failed": False,
        "model_step_id": events[6].payload["model_step_id"],
    }
    assert "model summary" not in str(events[4].payload)

    provider_context = runtime_provider.requests[0].messages
    assert [message.role for message in provider_context] == ["user", "user"]
    assert provider_context[0].content[0].text == (
        "Previous session context summary:\nmodel summary"
    )
    assert provider_context[1].content[0].text == "current"

    checkpoint = asyncio.run(store.load_checkpoint("sess_model_compaction"))
    private_compaction_completion = asyncio.run(
        _private_events_for_public_events(store, "sess_model_compaction", [events[3]])
    )[0]
    assert checkpoint_without_model_step_publication(checkpoint) == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "context_compaction": {
            "version": 2,
            "summary": "model summary",
            "compacted_transcript_cursor": 2,
            "metadata": {
                "compactor": "ModelCompactor",
                "provider": "fake",
                "model": "summary-model",
                "input_truncated": False,
                "max_input_chars": 120000,
                "completed": {
                    "finish_reason": "stop",
                    "model_step_id": private_compaction_completion.payload["model_step_id"],
                    "model_attempt_id": private_compaction_completion.payload["model_attempt_id"],
                },
            },
        },
    }


def test_cayu_app_counts_compaction_model_spend_in_session_usage():
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 40, "output_tokens": 8},
                }
            ),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
        ]
    )
    app = CayuApp()
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
                session_id="sess_compaction_usage",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    compaction_completed = [
        event
        for event in events
        if event.type == EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    ]
    assert len(compaction_completed) == 1
    assert compaction_completed[0].payload["usage_metrics"]["input_tokens"] == 40
    assert compaction_completed[0].payload["usage_metrics"]["model"] == "summary-model"
    assert compaction_completed[0].payload["usage_metrics"]["requested_model"] == "summary-model"

    summary = asyncio.run(app.get_session_usage("sess_compaction_usage"))
    assert summary.model_steps == 2
    assert summary.provider_names == ["fake"]
    assert summary.models == ["summary-model", "fake-model"]
    assert summary.usage.input_tokens == 50
    assert summary.usage.output_tokens == 10
    assert summary.usage.total_tokens == 60


@pytest.mark.parametrize("scope", ["run", "session"])
def test_cayu_app_stops_before_main_model_when_compaction_reaches_token_limit(
    scope: Literal["run", "session"],
):
    store = InMemorySessionStore()
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 16, "output_tokens": 4},
                }
            ),
        ]
    )
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(session_store=store)
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
    session_id = f"sess_compaction_{scope}_token_limit"

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
                limits=RunLimits(max_total_tokens=15, scope=scope),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[6].payload["limit"] == "total_tokens"
    assert events[6].payload["actual"] == 20
    assert events[6].payload["maximum"] == 15
    assert events[7].payload["token_usage"]["total_tokens"] == 20
    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint is not None
    assert checkpoint["context_compaction"]["summary"] == "model summary"
    usage = asyncio.run(app.get_session_usage(session_id))
    assert usage.model_steps == 1
    assert usage.usage.total_tokens == 20


def test_resume_after_compaction_limit_stop_uses_latest_agent_completion_for_context() -> None:
    store = InMemorySessionStore()
    compactor_provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first model summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "summary-model",
                        "usage": {
                            "input_tokens": 16,
                            "output_tokens": 4,
                            "total_tokens": 20,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("second model summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "summary-model",
                        "usage": {
                            "input_tokens": 16,
                            "output_tokens": 4,
                            "total_tokens": 20,
                        },
                    }
                ),
            ],
        ]
    )
    runtime_provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed(
                    {
                        "usage": {
                            "input_tokens": 8,
                            "output_tokens": 2,
                            "total_tokens": 10,
                        }
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("resumed answer"),
                ModelStreamEvent.completed(
                    {
                        "usage": {
                            "input_tokens": 8,
                            "output_tokens": 2,
                            "total_tokens": 10,
                        }
                    }
                ),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        min_total_tokens=15,
        sticky=False,
    )
    app = CayuApp(
        session_store=store,
        execution_profile_policy=_AllowForkProfileAdoption(),
        enable_logging=False,
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )
    session_id = "sess_resume_after_compaction_limit_stop"
    limits = RunLimits(max_total_tokens=15, scope="run")

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "first request")],
                limits=limits,
            ),
        )
    )
    stopped_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "second request")],
                limits=limits,
            ),
        )
    )

    assert first_events[-1].type == EventType.SESSION_COMPLETED
    assert stopped_events[-1].type == EventType.SESSION_INTERRUPTED
    assert any(event.type == EventType.SESSION_LIMIT_REACHED for event in stopped_events)
    assert len(runtime_provider.requests) == 1
    assert len(compactor_provider.requests) == 1

    resumed_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "third request")],
                profile_adoption=ExecutionProfileAdoptionIntent(
                    idempotency_key="resume-after-compaction-limit-stop",
                    reason="The operator authorizes a fresh run without the prior run limit.",
                    requested_by=ResolutionActor(
                        subject="test-operator",
                        source=ResolutionActorSource.REQUEST,
                    ),
                ),
            ),
        )
    )

    assert resumed_events[-1].type == EventType.SESSION_COMPLETED
    assert len(compactor_provider.requests) == 2
    assert len(runtime_provider.requests) == 2
    resumed_projection = json.dumps(
        [message.model_dump(mode="json") for message in runtime_provider.requests[-1].messages]
    )
    assert "second model summary" in resumed_projection
    assert "third request" in resumed_projection


def test_cayu_app_stops_before_main_model_when_compaction_reaches_app_budget():
    store = InMemorySessionStore()
    pricing = compaction_price_book()
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 20, "output_tokens": 0},
                }
            ),
        ]
    )
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(
        session_store=store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("0.00002"),
                    pricing=pricing,
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
                session_id="sess_compaction_app_budget",
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
        EventType.BUDGET_CHECKED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert Decimal(events[8].payload["actual"]) == Decimal("0.00002")
    assert events[9].payload["scope"] == "app"
    assert events[10].payload["limit"] == "estimated_cost"
    assert runtime_provider.requests == []
    checkpoint = asyncio.run(store.load_checkpoint("sess_compaction_app_budget"))
    assert checkpoint is not None
    cost = asyncio.run(app.get_session_cost("sess_compaction_app_budget", pricing))
    assert cost.model_steps == 1
    assert cost.priced_model_steps == 1
    assert cost.total_cost == Decimal("0.00002")


@pytest.mark.parametrize(
    ("maximum", "main_provider_calls", "terminal_event"),
    [
        pytest.param(
            "0.000029",
            0,
            EventType.SESSION_INTERRUPTED,
            id="combined-reservation-above-cap",
        ),
        pytest.param(
            "0.000030",
            1,
            EventType.SESSION_INTERRUPTED,
            id="combined-reservation-at-cap",
        ),
        pytest.param(
            "0.000031",
            1,
            EventType.SESSION_COMPLETED,
            id="combined-reservation-below-cap",
        ),
    ],
)
def test_cayu_app_compaction_spend_participates_in_strict_main_admission(
    maximum: str,
    main_provider_calls: int,
    terminal_event: EventType,
) -> None:
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 20, "output_tokens": 0},
                }
            ),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 0}}),
        ]
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal(maximum),
                    pricing=compaction_price_book(),
                    reservation=BudgetReservation(
                        max_input_tokens=10,
                        max_output_tokens=0,
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
                session_id=f"sess_compaction_strict_admission_{maximum}",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(compactor_provider.requests) == 1
    assert len(runtime_provider.requests) == main_provider_calls
    reconciliations = [event for event in events if event.type == EventType.BUDGET_RECONCILED]
    assert Decimal(reconciliations[0].payload["actual_amount"]) == Decimal("0.00002")
    if main_provider_calls == 0:
        failure = next(
            event for event in events if event.type == EventType.BUDGET_RESERVATION_FAILED
        )
        assert Decimal(failure.payload["actual"]) == Decimal("0.00003")
        assert Decimal(failure.payload["requested"]) == Decimal("0.00001")
    else:
        assert [Decimal(event.payload["actual_amount"]) for event in reconciliations] == [
            Decimal("0.00002"),
            Decimal("0.00001"),
        ]
    assert events[-1].type == terminal_event


def test_cayu_app_rejects_automatic_compaction_before_provider_dispatch() -> None:
    compactor_provider = FakeProvider([ModelStreamEvent.completed({})])
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("0.000009"),
                    pricing=compaction_price_book(),
                    reservation=BudgetReservation(
                        max_input_tokens=10,
                        max_output_tokens=0,
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
                session_id="sess_compaction_reservation_rejected",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert compactor_provider.requests == []
    assert runtime_provider.requests == []
    assert EventType.REQUEST_FOOTPRINT_RECORDED not in [event.type for event in events]
    assert EventType.MODEL_STARTED not in [event.type for event in events]
    assert EventType.BUDGET_RESERVATION_FAILED in [event.type for event in events]
    assert EventType.BUDGET_RESERVED not in [event.type for event in events]
    assert EventType.BUDGET_RECONCILED not in [event.type for event in events]
    assert events[-1].type == EventType.SESSION_INTERRUPTED


def test_cayu_app_automatic_compaction_failure_charges_reserved_amount() -> None:
    compactor_provider = FakeProvider([ModelStreamEvent.error("compactor unavailable")])
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=compaction_price_book(),
                    reservation=BudgetReservation(
                        max_input_tokens=10,
                        max_output_tokens=0,
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
                session_id="sess_compaction_provider_failure_budget",
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
    reconciliation = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
    assert Decimal(reconciliation.payload["actual_amount"]) == Decimal("0.00001")
    assert reconciliation.payload["reason"] == (
        "automatic context compaction dispatch has uncertain usage; charged reserved amount"
    )
    assert events[-1].type == EventType.SESSION_FAILED


def test_automatic_compaction_predispatch_failure_does_not_charge_budget(monkeypatch) -> None:
    compactor_provider = FakeProvider([ModelStreamEvent.completed({})])
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])

    def fail_before_request(request: CompactionRequest, **kwargs) -> str:
        del request, kwargs
        raise ValueError("prompt construction failed")

    monkeypatch.setattr("cayu.context.base._bounded_default_compaction_prompt", fail_before_request)

    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=compaction_price_book(),
                    reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
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
                session_id="sess_compaction_predispatch_failure_budget",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert compactor_provider.requests == []
    assert runtime_provider.requests == []
    assert EventType.BUDGET_RESERVED not in [event.type for event in events]
    assert EventType.BUDGET_RECONCILED not in [event.type for event in events]
    assert EventType.BUDGET_RESERVATION_RELEASED not in [event.type for event in events]
    assert events[-1].type == EventType.SESSION_FAILED


def test_automatic_compaction_supports_non_expiring_custom_ledger() -> None:
    class NonExpiringLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=None)

        async def heartbeat(self, *, reservation_id: str) -> bool:
            del reservation_id
            raise AssertionError("non-expiring ledgers must not be heartbeated")

    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed({"model": "summary-model", "usage": {"input_tokens": 1}}),
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
                    reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
                ),
            )
        ),
        budget_ledger=NonExpiringLedger(),
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
                session_id="sess_non_expiring_compaction_budget_ledger",
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
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_automatic_compaction_cancellation_releases_partial_reservation_setup() -> None:
    class BlockingSecondReservationLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=None)
            self.reserve_calls = 0
            self.second_reservation_started = asyncio.Event()

        async def reserve(self, **kwargs):
            self.reserve_calls += 1
            if self.reserve_calls == 2:
                self.second_reservation_started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            return await super().reserve(**kwargs)

    async def run() -> None:
        session_id = "sess_compaction_partial_reservation_cancelled"
        store = InMemorySessionStore()
        ledger = BlockingSecondReservationLedger()
        compactor_provider = FakeProvider([ModelStreamEvent.completed({})])
        runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
        reservation = BudgetReservation(max_input_tokens=10, max_output_tokens=0)
        pricing = compaction_price_book()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        reservation=reservation,
                    ),
                    BudgetLimit(
                        scope="agent",
                        key="assistant",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        reservation=reservation,
                    ),
                )
            ),
            budget_ledger=ledger,
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

        task = asyncio.create_task(
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
        await asyncio.wait_for(ledger.second_reservation_started.wait(), timeout=10)
        task.cancel("cancel reservation setup")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        records = tuple(ledger._records.values())
        assert exc_info.value.args == ("cancel reservation setup",)
        assert len(records) == 1
        assert records[0].status == "released"
        assert records[0].reason == "reservation setup cancelled"
        assert compactor_provider.requests == []
        assert runtime_provider.requests == []
        stored_events = await store.load_events(session_id)
        reserved = [event for event in stored_events if event.type == EventType.BUDGET_RESERVED]
        assert len(reserved) == 1
        assert reserved[0].payload["reservation_id"] == records[0].reservation_id
        released = [
            event for event in stored_events if event.type == EventType.BUDGET_RESERVATION_RELEASED
        ]
        assert len(released) == 1
        assert released[0].payload["reservation_id"] == records[0].reservation_id
        assert released[0].payload["reason"] == "reservation setup cancelled"

    asyncio.run(run())


def test_automatic_compaction_cancellation_while_persisting_reservation_releases_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        session_id = "sess_compaction_reservation_publish_cancelled"
        store = InMemorySessionStore()
        ledger = InMemoryBudgetLedger(reservation_ttl_seconds=None)
        compactor_provider = FakeProvider([ModelStreamEvent.completed({})])
        runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=compaction_price_book(),
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=0,
                        ),
                    ),
                )
            ),
            budget_ledger=ledger,
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
        reservation_publish_started = asyncio.Event()
        original_emit = app._event_writer.emit
        block_reservation_publish = True

        async def block_first_reservation_publish(event: Event) -> Event:
            nonlocal block_reservation_publish
            if block_reservation_publish and event.type == EventType.BUDGET_RESERVED:
                block_reservation_publish = False
                reservation_publish_started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            return await original_emit(event)

        monkeypatch.setattr(app._event_writer, "emit", block_first_reservation_publish)
        task = asyncio.create_task(
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
        await asyncio.wait_for(reservation_publish_started.wait(), timeout=10)
        task.cancel("cancel reservation event persistence")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        records = tuple(ledger._records.values())
        assert exc_info.value.args == ("cancel reservation event persistence",)
        assert len(records) == 1
        assert records[0].status == "released"
        assert records[0].reason == "reservation setup cancelled"
        assert compactor_provider.requests == []
        assert runtime_provider.requests == []
        stored_events = await store.load_events(session_id)
        assert EventType.BUDGET_RESERVED not in [event.type for event in stored_events]
        released = [
            event for event in stored_events if event.type == EventType.BUDGET_RESERVATION_RELEASED
        ]
        assert len(released) == 1
        assert released[0].payload["reservation_id"] == records[0].reservation_id
        assert released[0].payload["reason"] == "reservation setup cancelled"

    asyncio.run(run())


def test_automatic_compaction_cancellation_waits_for_reconciliation() -> None:
    class BlockingReconciliationLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=None)
            self.reconciliation_started = asyncio.Event()
            self.allow_reconciliation = asyncio.Event()

        async def reconcile(self, **kwargs):
            self.reconciliation_started.set()
            await self.allow_reconciliation.wait()
            return await super().reconcile(**kwargs)

    async def run() -> None:
        session_id = "sess_compaction_reconciliation_cancelled"
        store = InMemorySessionStore()
        ledger = BlockingReconciliationLedger()
        compactor_provider = FakeProvider(
            [ModelStreamEvent.completed({"usage": {"input_tokens": 1}})]
        )
        runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=compaction_price_book(),
                        reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
                    ),
                )
            ),
            budget_ledger=ledger,
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

        task = asyncio.create_task(
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
        await asyncio.wait_for(ledger.reconciliation_started.wait(), timeout=10)
        task.cancel("cancel during compaction reconciliation")
        await asyncio.sleep(0)
        assert not task.done()
        task.cancel("repeated cancellation must not abandon settlement")
        await asyncio.sleep(0)
        assert not task.done()
        ledger.allow_reconciliation.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert exc_info.value.args == ("cancel during compaction reconciliation",)
        records = tuple(ledger._records.values())
        assert len(records) == 1
        assert records[0].status == "reconciled"
        assert records[0].actual_amount == Decimal("0.000001")
        stored_events = await store.load_events(session_id)
        reconciled = [event for event in stored_events if event.type == EventType.BUDGET_RECONCILED]
        assert len(reconciled) == 1
        assert reconciled[0].payload["reservation_id"] == records[0].reservation_id
        assert len(compactor_provider.requests) == 1
        assert runtime_provider.requests == []

    asyncio.run(run())


def test_automatic_compaction_preserves_provider_cancellation_during_settlement() -> None:
    class BlockingCompactionProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.started = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            self.started.set()
            await asyncio.Event().wait()
            yield ModelStreamEvent.completed({})

    class BlockingReconciliationLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=None)
            self.reconciliation_started = asyncio.Event()
            self.allow_reconciliation = asyncio.Event()

        async def reconcile(self, **kwargs):
            self.reconciliation_started.set()
            await self.allow_reconciliation.wait()
            return await super().reconcile(**kwargs)

    async def run() -> None:
        session_id = "sess_compaction_provider_and_settlement_cancelled"
        store = InMemorySessionStore()
        ledger = BlockingReconciliationLedger()
        compactor_provider = BlockingCompactionProvider()
        runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=compaction_price_book(),
                        reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
                    ),
                )
            ),
            budget_ledger=ledger,
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

        task = asyncio.create_task(
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
        await asyncio.wait_for(compactor_provider.started.wait(), timeout=10)
        task.cancel("first cancellation during provider execution")
        await asyncio.wait_for(ledger.reconciliation_started.wait(), timeout=10)
        task.cancel("later cancellation during settlement")
        await asyncio.sleep(0)
        assert not task.done()
        ledger.allow_reconciliation.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert exc_info.value.args == ("first cancellation during provider execution",)
        records = tuple(ledger._records.values())
        assert len(records) == 1
        assert records[0].status == "reconciled"
        assert records[0].actual_amount == records[0].reserved_amount
        stored_events = await store.load_events(session_id)
        reconciled = [event for event in stored_events if event.type == EventType.BUDGET_RECONCILED]
        assert len(reconciled) == 1
        assert reconciled[0].payload["reservation_id"] == records[0].reservation_id
        assert len(compactor_provider.requests) == 1
        assert runtime_provider.requests == []

    asyncio.run(run())


def test_automatic_compaction_cancellation_waits_for_reconciliation_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        session_id = "sess_compaction_reconciliation_publish_cancelled"
        store = InMemorySessionStore()
        ledger = InMemoryBudgetLedger(reservation_ttl_seconds=None)
        compactor_provider = FakeProvider(
            [ModelStreamEvent.completed({"usage": {"input_tokens": 1}})]
        )
        runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=compaction_price_book(),
                        reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
                    ),
                )
            ),
            budget_ledger=ledger,
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
        reconciliation_publish_started = asyncio.Event()
        allow_reconciliation_publish = asyncio.Event()
        original_persist_exact_replay = app._event_writer.persist_exact_replay

        async def block_reconciliation_publish(event: Event) -> Event:
            if event.type == EventType.BUDGET_RECONCILED:
                reconciliation_publish_started.set()
                await allow_reconciliation_publish.wait()
            return await original_persist_exact_replay(event)

        monkeypatch.setattr(
            app._event_writer,
            "persist_exact_replay",
            block_reconciliation_publish,
        )
        task = asyncio.create_task(
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
        await asyncio.wait_for(reconciliation_publish_started.wait(), timeout=10)
        records = tuple(ledger._records.values())
        assert len(records) == 1
        assert records[0].status == "reconciled"
        task.cancel("cancel reconciliation event persistence")
        await asyncio.sleep(0)
        assert not task.done()
        allow_reconciliation_publish.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert exc_info.value.args == ("cancel reconciliation event persistence",)
        stored_events = await store.load_events(session_id)
        reconciled = [event for event in stored_events if event.type == EventType.BUDGET_RECONCILED]
        assert len(reconciled) == 1
        assert reconciled[0].payload["reservation_id"] == records[0].reservation_id
        assert len(compactor_provider.requests) == 1
        assert runtime_provider.requests == []

    asyncio.run(run())


def test_automatic_compaction_cancellation_stays_authoritative_if_settlement_fails() -> None:
    class BlockingFailingReconciliationLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=None)
            self.reconciliation_started = asyncio.Event()
            self.allow_failure = asyncio.Event()

        async def reconcile(self, **kwargs):
            del kwargs
            self.reconciliation_started.set()
            await self.allow_failure.wait()
            raise RuntimeError("budget ledger unavailable")

    async def run() -> None:
        ledger = BlockingFailingReconciliationLedger()
        compactor_provider = FakeProvider([ModelStreamEvent.completed({})])
        runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
        app = CayuApp(
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=compaction_price_book(),
                        reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
                    ),
                )
            ),
            budget_ledger=ledger,
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

        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_compaction_cancelled_settlement_failure",
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                ),
            )
        )
        await asyncio.wait_for(ledger.reconciliation_started.wait(), timeout=10)
        task.cancel("cancel while settlement fails")
        ledger.allow_failure.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert exc_info.value.args == ("cancel while settlement fails",)
        assert exc_info.value.__notes__ == [
            "Automatic compaction budget settlement also failed: "
            "RuntimeError: budget ledger unavailable",
            "Continuation recovery cleanup failed during cancelled session "
            "finalization: ModelCompletionBudgetSettlementPending. The original "
            "failure remains authoritative.",
        ]
        cleanup_failure = exception_cause(exc_info.value)
        assert isinstance(cleanup_failure, BaseExceptionGroup)
        settlement_failure = exception_cause(cleanup_failure)
        assert type(settlement_failure) is RuntimeError
        assert str(settlement_failure) == "budget ledger unavailable"
        cleanup_failures = tuple(iter_exception_tree(cleanup_failure))
        assert (
            sum(
                isinstance(
                    failure,
                    run_limits_module.ModelCompletionBudgetSettlementPending,
                )
                for failure in cleanup_failures
            )
            == 1
        )
        assert len(compactor_provider.requests) == 1
        assert runtime_provider.requests == []
        active = await app.session_store.load_active_model_completion_stage(
            "sess_compaction_cancelled_settlement_failure"
        )
        assert active is not None
        assert active.stage.state == "completed"
        records = tuple(ledger._records.values())
        assert [record.status for record in records] == ["active"]

    asyncio.run(run())


def test_automatic_compaction_settles_other_limits_after_one_reconciliation_fails() -> None:
    class FailMiddleReconciliationLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=None)
            self.reconcile_calls = 0

        async def reconcile(self, **kwargs):
            self.reconcile_calls += 1
            if self.reconcile_calls == 2:
                raise RuntimeError("middle reconciliation failed")
            return await super().reconcile(**kwargs)

    session_id = "sess_compaction_partial_settlement_failure"
    ledger = FailMiddleReconciliationLedger()
    reservation = BudgetReservation(max_input_tokens=10, max_output_tokens=0)
    pricing = compaction_price_book()
    compactor_provider = FakeProvider([ModelStreamEvent.completed({"usage": {"input_tokens": 1}})])
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=pricing,
                    reservation=reservation,
                ),
                BudgetLimit(
                    scope="agent",
                    key="assistant",
                    max_estimated_cost=Decimal("1"),
                    pricing=pricing,
                    reservation=reservation,
                ),
                BudgetLimit(
                    scope="causal",
                    key=session_id,
                    max_estimated_cost=Decimal("1"),
                    pricing=pricing,
                    reservation=reservation,
                ),
            )
        ),
        budget_ledger=ledger,
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

    async def run():
        events = []
        with pytest.raises(RuntimeError, match="middle reconciliation failed") as exc_info:
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
                events.append(event)
        return (
            events,
            exc_info.value,
            await app.session_store.load(session_id),
            await app.session_store.load_active_model_completion_stage(session_id),
        )

    events, failure, session, active = asyncio.run(run())

    records = tuple(ledger._records.values())
    assert ledger.reconcile_calls == 3
    assert [record.status for record in records] == ["reconciled", "active", "reconciled"]
    assert active is not None
    assert active.stage.state == "completed"
    assert set(active.stage.reservation_ids) == {record.reservation_id for record in records}
    assert session is not None and session.status == SessionStatus.RUNNING
    assert EventType.SESSION_FAILED not in [event.type for event in events]
    assert failure.__notes__ == [
        "Automatic compaction budget settlement failed for reservation "
        f"{records[1].reservation_id}."
    ]
    failure_cause = exception_cause(failure)
    assert isinstance(failure_cause, BaseExceptionGroup)
    cause_failures = tuple(iter_exception_tree(failure_cause))
    assert (
        sum(
            isinstance(
                item,
                run_limits_module.ModelCompletionBudgetSettlementPending,
            )
            for item in cause_failures
        )
        == 1
    )

    reconciled = [event for event in events if event.type == EventType.BUDGET_RECONCILED]
    private_reconciled = asyncio.run(
        _private_events_for_public_events(app.session_store, session_id, reconciled)
    )
    assert [event.payload["reservation_id"] for event in private_reconciled] == [
        records[0].reservation_id,
        records[2].reservation_id,
    ]
    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []

    with pytest.raises(
        ModelCompletionManualRecoveryRequired,
        match="promoted without a durable context checkpoint",
    ):
        asyncio.run(
            app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_for_seconds=0,
                )
            )
        )

    recovered_records = tuple(ledger._records.values())
    assert ledger.reconcile_calls == 4
    assert [record.status for record in recovered_records] == [
        "reconciled",
        "reconciled",
        "reconciled",
    ]
    assert recovered_records[1].reason == (run_limits_module.UNKNOWN_POST_DISPATCH_BUDGET_REASON)
    assert asyncio.run(app.session_store.load_active_model_completion_stage(session_id)) is None
    recovered_events = asyncio.run(app.session_store.load_events(session_id))
    recovered_completions = [
        event
        for event in recovered_events
        if event.type is EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    ]
    assert len(recovered_completions) == 1
    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []


@pytest.mark.parametrize(
    ("maximum", "compactor_calls", "runtime_calls", "terminal_event"),
    [
        pytest.param(
            "0.000019",
            1,
            0,
            EventType.SESSION_INTERRUPTED,
            id="retry-reservation-denied",
        ),
        pytest.param(
            "1",
            1,
            0,
            EventType.SESSION_INTERRUPTED,
            id="retry-uncertain-attempt-fails-closed",
        ),
    ],
)
def test_automatic_compaction_retry_requires_an_independent_reservation(
    maximum: str,
    compactor_calls: int,
    runtime_calls: int,
    terminal_event: EventType,
) -> None:
    class InheritedModelCompactor(ModelCompactor):
        pass

    class RetryCompactionProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                raise ModelProviderError(
                    "provider overloaded",
                    provider=self.name,
                    status_code=503,
                    retryable=True,
                )
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed(
                {"model": "summary-model", "usage": {"input_tokens": 1}}
            )

    compactor_provider = RetryCompactionProvider()
    runtime_provider = FakeProvider(
        [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}})]
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal(maximum),
                    pricing=compaction_price_book(),
                    reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
                ),
            )
        )
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=InheritedModelCompactor(
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
                session_id=f"sess_compaction_retry_budget_{maximum}",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(compactor_provider.requests) == compactor_calls
    assert len(runtime_provider.requests) == runtime_calls
    reconciliations = [event for event in events if event.type == EventType.BUDGET_RECONCILED]
    assert Decimal(reconciliations[0].payload["actual_amount"]) == Decimal("0.00001")
    failure = next(event for event in events if event.type == EventType.BUDGET_RESERVATION_FAILED)
    assert Decimal(failure.payload["actual"]) == Decimal("0")
    assert failure.payload["message"] == (
        "Budget cannot be verified because 1 model step(s) have missing or invalid completion usage."
    )
    assert events[-1].type == terminal_event


def test_automatic_compaction_does_not_retry_after_settlement_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingReconcileLedger(InMemoryBudgetLedger):
        async def reconcile(
            self,
            *,
            reservation_id: str,
            actual_amount: Decimal,
            reason: str | None = None,
            occurred_at: datetime | None = None,
            **kwargs,
        ):
            del reservation_id, actual_amount, reason, occurred_at, kwargs
            raise RuntimeError("ledger reconciliation failed")

    class RetryableFailureProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            raise ModelProviderError(
                "provider overloaded",
                provider=self.name,
                status_code=503,
                retryable=True,
            )
            yield

    compactor_provider = RetryableFailureProvider()
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    ledger = FailingReconcileLedger()
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=compaction_price_book(),
                    reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
                ),
            )
        ),
        budget_ledger=ledger,
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

    session_id = "sess_compaction_settlement_failure_no_retry"

    async def run():
        events = []
        with pytest.raises(ModelProviderError, match="provider overloaded") as exc_info:
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
                events.append(event)
        return (
            events,
            exc_info.value,
            await app.session_store.load(session_id),
            await app.session_store.load_active_model_completion_stage(session_id),
        )

    events, failure, session, active = asyncio.run(run())

    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []
    assert EventType.SESSION_FAILED not in [event.type for event in events]
    assert session is not None and session.status == SessionStatus.RUNNING
    assert failure.__notes__ == [
        "Automatic compaction budget settlement also failed: "
        "RuntimeError: ledger reconciliation failed"
    ]
    failure_cause = exception_cause(failure)
    assert isinstance(failure_cause, BaseExceptionGroup)
    cause_failures = tuple(iter_exception_tree(failure_cause))
    assert (
        sum(
            type(item) is RuntimeError and str(item) == "ledger reconciliation failed"
            for item in cause_failures
        )
        == 1
    )
    assert (
        sum(
            isinstance(
                item,
                run_limits_module.ModelCompletionBudgetSettlementPending,
            )
            for item in cause_failures
        )
        == 1
    )
    assert active is not None and active.stage.state == "completed"

    assert active.stage.publication is not None
    [staged_error] = [
        event for event in active.stage.publication.events if event.type is EventType.MODEL_ERROR
    ]
    assert staged_error.payload["status_code"] == 503
    assert staged_error.payload["provider_retryable"] is True
    assert staged_error.payload["retry"] is False
    assert staged_error.payload["retry_disposition"] == "suppressed"
    assert staged_error.payload["retry_suppression"] == "automatic_retry_disabled"

    with pytest.raises(RuntimeError, match="ledger reconciliation failed"):
        asyncio.run(
            app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_for_seconds=0,
                )
            )
        )

    recovered_active = asyncio.run(app.session_store.load_active_model_completion_stage(session_id))
    assert recovered_active == active
    assert [record.status for record in ledger._records.values()] == ["active"]
    recovered_events = asyncio.run(app.session_store.load_events(session_id))
    assert not any(
        event.type is EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
        for event in recovered_events
    )
    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []

    monkeypatch.setattr(ledger, "reconcile", InMemoryBudgetLedger.reconcile.__get__(ledger))
    with pytest.raises(ModelCompletionManualRecoveryRequired, match="without a durable context"):
        asyncio.run(
            app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id, inactive_for_seconds=0)
            )
        )
    recovered_events = asyncio.run(app.session_store.load_events(session_id))
    [recovered_error] = [event for event in recovered_events if event.type is EventType.MODEL_ERROR]
    assert recovered_error.id == staged_error.id
    assert recovered_error.payload["status_code"] == 503
    assert [record.status for record in ledger._records.values()] == ["reconciled"]
    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []


def test_prompt_cache_compaction_does_not_fallback_after_settlement_failure() -> None:
    class FailSecondReconcileLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__()
            self.reconcile_calls = 0

        async def reconcile(
            self,
            *,
            reservation_id: str,
            actual_amount: Decimal,
            reason: str | None = None,
            occurred_at: datetime | None = None,
            **kwargs,
        ):
            self.reconcile_calls += 1
            if self.reconcile_calls == 2:
                raise RuntimeError("ledger reconciliation failed")
            return await super().reconcile(
                reservation_id=reservation_id,
                actual_amount=actual_amount,
                reason=reason,
                occurred_at=occurred_at,
                **kwargs,
            )

    overflow = ModelContextOverflowError(
        "context too large",
        provider="fake",
        status_code=400,
        error_code="context_length_exceeded",
    )
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}}),
            ],
            [ModelStreamEvent.error(str(overflow), cause=overflow)],
        ]
    )
    ledger = FailSecondReconcileLedger()
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=compaction_price_book(),
                    reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
                ),
            )
        ),
        budget_ledger=ledger,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=PromptCacheCompactor(provider=provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )
    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_prompt_cache_settlement_failure",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    assert first_events[-1].type == EventType.SESSION_COMPLETED

    with pytest.raises(ModelContextOverflowError, match="context too large") as exc_info:
        asyncio.run(
            collect_resume_events(
                app,
                ResumeRequest(
                    session_id="sess_prompt_cache_settlement_failure",
                    messages=[Message.text("user", "second")],
                ),
            )
        )

    assert len(provider.requests) == 2
    assert ledger.reconcile_calls == 2
    failure = exc_info.value
    assert failure.__notes__ == [
        "Automatic compaction budget settlement also failed: "
        "RuntimeError: ledger reconciliation failed"
    ]
    failure_cause = exception_cause(failure)
    assert isinstance(failure_cause, BaseExceptionGroup)
    cause_failures = tuple(iter_exception_tree(failure_cause))
    assert (
        sum(
            type(item) is RuntimeError and str(item) == "ledger reconciliation failed"
            for item in cause_failures
        )
        == 1
    )
    assert (
        sum(
            isinstance(
                item,
                run_limits_module.ModelCompletionBudgetSettlementPending,
            )
            for item in cause_failures
        )
        == 1
    )
    active = asyncio.run(
        app.session_store.load_active_model_completion_stage("sess_prompt_cache_settlement_failure")
    )
    assert active is not None and active.stage.state == "completed"


def test_inherited_prompt_cache_compactor_fails_closed_before_unpriced_fallback() -> None:
    class InheritedPromptCacheCompactor(PromptCacheCompactor):
        pass

    overflow = ModelContextOverflowError(
        "context too large",
        provider="fake",
        status_code=400,
        error_code="context_length_exceeded",
    )
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed(
                    {
                        "model": "fake-model",
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    }
                ),
            ],
            [ModelStreamEvent.error(str(overflow), cause=overflow)],
            [
                ModelStreamEvent.text_delta("bounded summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "fake-model",
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    }
                ),
            ],
        ]
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("0.000019"),
                    pricing=compaction_price_book(),
                    reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
                ),
            )
        )
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=InheritedPromptCacheCompactor(provider=provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_inherited_prompt_cache_budget",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    assert first_events[-1].type == EventType.SESSION_COMPLETED

    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_inherited_prompt_cache_budget",
                messages=[Message.text("user", "second")],
            ),
        )
    )

    assert len(provider.requests) == 2
    assert [event.type for event in resume_events].count(EventType.BUDGET_RESERVED) == 1
    failures = [
        event for event in resume_events if event.type == EventType.BUDGET_RESERVATION_FAILED
    ]
    assert len(failures) == 1, [event.type for event in resume_events]
    failure = failures[0]
    assert Decimal(failure.payload["actual"]) == Decimal("0.000001")
    assert failure.payload["message"] == (
        "Budget cannot be verified because 1 model step(s) have missing or invalid completion usage."
    )
    assert resume_events[-1].type == EventType.SESSION_INTERRUPTED


@pytest.mark.parametrize("override_target", ["prompt_once", "oversized_atomic_unit"])
def test_automatic_compaction_rejects_model_compactor_helper_overrides_before_dispatch(
    override_target: str,
) -> None:
    class PromptOnceOverrideCompactor(ModelCompactor):
        async def _compact_prompt_once(self, *args, **kwargs) -> CompactionResult:
            del args, kwargs
            raise AssertionError("overridden prompt helper must not execute")

    class OversizedOverrideCompactor(ModelCompactor):
        async def _compact_oversized_atomic_unit(
            self,
            request: CompactionRequest,
        ) -> CompactionResult:
            del request
            raise AssertionError("overridden hierarchy helper must not execute")

    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("must not dispatch"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 1}}),
        ]
    )
    compactor_type = (
        PromptOnceOverrideCompactor
        if override_target == "prompt_once"
        else OversizedOverrideCompactor
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=compactor_type(
                provider=provider,
                model="summary-model",
                max_input_chars=1000,
            ),
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )
    old_message = "oversized " + "x" * 5000 if override_target == "oversized_atomic_unit" else "old"

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_compaction_helper_override_{override_target}",
                messages=[
                    Message.text("user", old_message),
                    Message.text("user", "current"),
                ],
                limits=RunLimits(max_total_tokens=100),
            ),
        )
    )

    assert provider.requests == []
    assert events[-1].type == EventType.SESSION_FAILED
    assert "cannot safely run opaque provider-backed compactor" in events[-1].payload["error"]


def test_automatic_compaction_rejects_provider_backed_deterministic_identity_claim() -> None:
    class MisdeclaredModelCompactor(ModelCompactor):
        def provider_budget_identity(self, session: Session) -> None:
            del session
            return None

    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("must not dispatch"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 1}}),
        ]
    )
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(enable_logging=False)
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=MisdeclaredModelCompactor(
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
                session_id="sess_compaction_misdeclared_deterministic_identity",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
                limits=RunLimits(max_total_tokens=10),
            ),
        )
    )

    assert compactor_provider.requests == []
    assert runtime_provider.requests == []
    assert events[-1].type == EventType.SESSION_FAILED
    assert "cannot declare a deterministic budget identity" in events[-1].payload["error"]


@pytest.mark.parametrize(
    ("reservation", "expected_compactor_calls", "expected_terminal_event"),
    [
        pytest.param(
            BudgetReservation(max_input_tokens=10, max_output_tokens=0),
            0,
            EventType.SESSION_FAILED,
            id="strict-reservation-fails-closed",
        ),
        pytest.param(
            None,
            0,
            EventType.SESSION_FAILED,
            id="event-backed-budget-also-fails-closed",
        ),
    ],
)
def test_automatic_compaction_opaque_multi_dispatch_budget_boundary(
    reservation: BudgetReservation | None,
    expected_compactor_calls: int,
    expected_terminal_event: EventType,
) -> None:
    class OpaqueMultiDispatchCompactor(ContextCompactor):
        def __init__(self, provider: ModelProvider) -> None:
            self.provider = provider
            self.calls = 0

        def provider_budget_identity(self, session: Session) -> tuple[str, str]:
            del session
            return "fake", "summary-model"

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.calls += 1
            model_request = ModelRequest(
                model="summary-model",
                messages=[Message.text("user", "summarize")],
            )
            for _ in range(2):
                async for _event in self.provider.stream(model_request):
                    pass
            return CompactionResult(
                summary="summary",
                covered_message_count=len(request.messages),
                model_completed_payloads=[
                    {
                        "provider_name": "fake",
                        "requested_model": "summary-model",
                        "model": "summary-model",
                        "purpose": "context_compaction",
                        "usage_metrics": {
                            "provider_name": "fake",
                            "requested_model": "summary-model",
                            "model": "summary-model",
                            "input_tokens": 10,
                            "output_tokens": 0,
                            "total_tokens": 10,
                        },
                    },
                    {
                        "provider_name": "fake",
                        "requested_model": "summary-model",
                        "model": "summary-model",
                        "purpose": "context_compaction",
                        "usage_metrics": {
                            "provider_name": "fake",
                            "requested_model": "summary-model",
                            "model": "summary-model",
                            "input_tokens": 10,
                            "output_tokens": 0,
                            "total_tokens": 10,
                        },
                    },
                ],
            )

    compactor_provider = FakeProvider(
        [
            [ModelStreamEvent.completed({"usage": {"input_tokens": 10}})],
            [ModelStreamEvent.completed({"usage": {"input_tokens": 10}})],
        ]
    )
    compactor = OpaqueMultiDispatchCompactor(compactor_provider)
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("0.000019"),
                    pricing=compaction_price_book(),
                    reservation=reservation,
                ),
            )
        )
    )
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
                session_id=f"sess_compaction_opaque_budget_{reservation is not None}",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert compactor.calls == expected_compactor_calls
    assert len(compactor_provider.requests) == expected_compactor_calls * 2
    assert runtime_provider.requests == []
    assert events[-1].type == expected_terminal_event
    assert EventType.BUDGET_RESERVED not in [event.type for event in events]
    assert EventType.MODEL_COMPLETED not in [event.type for event in events]
    assert "cannot safely run opaque provider-backed compactor" in events[-1].payload["error"]


def test_deterministic_automatic_compaction_does_not_consume_strict_reservation() -> None:
    class DeterministicRecordingCompactor(RecordingCompactor):
        def provider_budget_identity(self, session: Session) -> None:
            del session
            return None

    compactor = DeterministicRecordingCompactor()
    runtime_provider = FakeProvider(
        [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}})]
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("0.00002"),
                    pricing=compaction_price_book(),
                    reservation=BudgetReservation(
                        max_input_tokens=10,
                        max_output_tokens=0,
                    ),
                ),
            )
        )
    )
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
                session_id="sess_deterministic_compaction_strict_budget",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(compactor.requests) == 1
    assert len(runtime_provider.requests) == 1
    assert [event.type for event in events].count(EventType.BUDGET_RESERVED) == 1
    assert [event.type for event in events].count(EventType.BUDGET_RECONCILED) == 1
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_prompt_cache_deterministic_fallback_does_not_consume_compaction_reservation() -> None:
    provider = FakeProvider(
        [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}})]
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("0.00002"),
                    pricing=compaction_price_book(),
                    reservation=BudgetReservation(
                        max_input_tokens=10,
                        max_output_tokens=0,
                    ),
                ),
            )
        )
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=PromptCacheCompactor(provider=provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_prompt_cache_deterministic_budget_fallback",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert [event.type for event in events].count(EventType.BUDGET_RESERVED) == 1
    assert [event.type for event in events].count(EventType.BUDGET_RECONCILED) == 1
    compaction_completed = next(
        event for event in events if event.type == EventType.CONTEXT_COMPACTION_COMPLETED
    )
    assert compaction_completed.payload["chunk_mode"] == "digest_prefix"
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_automatic_compaction_reconciliation_survives_sqlite_ledger_reopen(
    tmp_path,
) -> None:
    ledger_path = tmp_path / "automatic-compaction-budget.sqlite"
    pricing = compaction_price_book()
    limit = BudgetLimit(
        scope="app",
        max_estimated_cost=Decimal("0.000029"),
        pricing=pricing,
        reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
    )
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 20, "output_tokens": 0},
                }
            ),
        ]
    )
    first_runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    first_ledger = SQLiteBudgetLedger(ledger_path)
    first_app = CayuApp(
        budget_policy=BudgetPolicy(limits=(limit,)),
        budget_ledger=first_ledger,
    )
    first_app.register_provider(first_runtime_provider, default=True)
    first_app.register_agent(
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

    asyncio.run(
        collect_events(
            first_app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_sqlite_first",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )
    asyncio.run(first_ledger.close())

    second_runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    second_ledger = SQLiteBudgetLedger(ledger_path)
    second_app = CayuApp(
        budget_policy=BudgetPolicy(limits=(limit,)),
        budget_ledger=second_ledger,
    )
    second_app.register_provider(second_runtime_provider, default=True)
    second_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    second_events = asyncio.run(
        collect_events(
            second_app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_sqlite_second",
                messages=[Message.text("user", "current")],
            ),
        )
    )
    asyncio.run(second_ledger.close())

    assert first_runtime_provider.requests == []
    assert second_runtime_provider.requests == []
    failure = next(
        event for event in second_events if event.type == EventType.BUDGET_RESERVATION_FAILED
    )
    assert Decimal(failure.payload["actual"]) == Decimal("0.00003")
    assert second_events[-1].type == EventType.SESSION_INTERRUPTED


def test_cayu_app_compaction_request_budget_notify_is_nonblocking_and_deduplicated():
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 20, "output_tokens": 0},
                }
            ),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}}),
        ]
    )
    app = CayuApp()
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
                session_id="sess_compaction_request_budget_notify",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
                budget_limits=(
                    BudgetLimit(
                        scope="run",
                        max_estimated_cost=Decimal("0.00002"),
                        pricing=compaction_price_book(),
                        action="notify",
                    ),
                ),
            ),
        )
    )

    notify_events = [event for event in events if event.type == EventType.BUDGET_LIMIT_REACHED]
    assert len(notify_events) == 1
    assert notify_events[0].payload["action"] == "notify"
    assert Decimal(notify_events[0].payload["actual"]) == Decimal("0.00002")
    assert len(compactor_provider.requests) == 1
    assert len(runtime_provider.requests) == 1
    assert events[-1].type == EventType.SESSION_COMPLETED
    usage = asyncio.run(app.get_session_usage("sess_compaction_request_budget_notify"))
    assert usage.model_steps == 2
    assert usage.usage.total_tokens == 21


def test_cayu_app_non_model_compaction_does_not_add_budget_boundary():
    compactor = RecordingCompactor()
    provider = FakeProvider(
        [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}})]
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("10"),
                    pricing=compaction_price_book(),
                ),
            )
        )
    )
    app.register_provider(provider, default=True)
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
                session_id="sess_non_model_compaction_budget_boundaries",
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
        EventType.BUDGET_CHECKED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.BUDGET_CHECKED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(compactor.requests) == 1
    assert len(provider.requests) == 1


def test_cayu_app_model_compaction_policy_without_compaction_keeps_normal_boundaries():
    compactor_provider = FakeProvider([ModelStreamEvent.completed({})])
    runtime_provider = FakeProvider(
        [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 0}})]
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("10"),
                    pricing=compaction_price_book(),
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
                session_id="sess_no_compaction_budget_boundaries",
                messages=[Message.text("user", "current")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.BUDGET_CHECKED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert compactor_provider.requests == []
    assert len(runtime_provider.requests) == 1


def test_cayu_app_keeps_invalid_summary_spend_before_rejecting_custom_recovery():
    class RecoveringCompactor(ContextCompactor):
        def __init__(self, provider: ModelProvider) -> None:
            self.provider = provider

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            try:
                await ModelCompactor(
                    provider=self.provider,
                    model="invalid-summary-model",
                ).compact(request)
            except ValueError:
                return CompactionResult(
                    summary="recovered summary",
                    covered_message_count=len(request.messages),
                    model_completed_payloads=[
                        {
                            "model": "fallback-summary-model",
                            "provider_name": "custom",
                            "requested_model": "fallback-summary-model",
                            "purpose": "context_compaction",
                            "compactor": type(self).__name__,
                            "usage_metrics": {
                                "provider_name": "custom",
                                "requested_model": "fallback-summary-model",
                                "model": "fallback-summary-model",
                                "input_tokens": 40,
                                "output_tokens": 2,
                                "total_tokens": 42,
                                "reasoning_output_tokens": 0,
                                "cache": {
                                    "read_tokens": 0,
                                    "write_tokens": 0,
                                    "cached_input_tokens": 0,
                                    "uncached_input_tokens": 40,
                                },
                            },
                        }
                    ],
                )
            raise AssertionError("the first compaction unexpectedly returned a valid summary")

    store = InMemorySessionStore()
    budget_store = InMemoryBudgetStore()
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.completed(
                {
                    "model": "invalid-summary-model",
                    "usage": {"input_tokens": 100, "output_tokens": 0},
                }
            )
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 3, "output_tokens": 1}}),
        ]
    )
    app = CayuApp(
        session_store=store,
        budget_store=budget_store,
        request_footprint=RequestFootprintConfig(enabled=False),
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=RecoveringCompactor(compactor_provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_recovered_invalid_compaction_summary",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    durable_events = asyncio.run(store.load_events("sess_recovered_invalid_compaction_summary"))
    completed_events = [
        event for event in durable_events if event.type == EventType.MODEL_COMPLETED
    ]
    assert len(completed_events) == 1
    invalid_attempt = completed_events[0]
    assert invalid_attempt.payload["compaction_outcome"] == "invalid_summary"
    assert invalid_attempt.payload["usage_metrics"]["input_tokens"] == 100
    assert "compaction_attempt_id" not in invalid_attempt.payload
    assert invalid_attempt.payload["model_step_id"].startswith("mstep_")
    assert invalid_attempt.payload["model_attempt_id"].startswith("matt_")
    assert events[-1].type == EventType.SESSION_FAILED
    assert (
        events[-1].payload["error"]
        == "Compaction completion was observed outside its provider dispatch."
    )
    assert runtime_provider.requests == []

    usage = asyncio.run(app.get_session_usage("sess_recovered_invalid_compaction_summary"))
    assert usage.model_steps == 1
    assert usage.usage.input_tokens == 100
    assert usage.usage.output_tokens == 0
    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    assert [event.id for event in budget_events] == [invalid_attempt.id]


def test_cayu_app_keeps_successful_inner_compaction_spend_when_wrapper_fails():
    class FailingWrapperCompactor(ContextCompactor):
        def __init__(self, provider: ModelProvider) -> None:
            self.provider = provider

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            await ModelCompactor(provider=self.provider, model="summary-model").compact(request)
            raise RuntimeError("wrapper post-processing failed")

    store = InMemorySessionStore()
    budget_store = InMemoryBudgetStore()
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
    app = CayuApp(
        session_store=store,
        budget_store=budget_store,
        request_footprint=RequestFootprintConfig(enabled=False),
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=FailingWrapperCompactor(compactor_provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_failed_compaction_wrapper",
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
    completed = events[3]
    assert completed.payload["usage_metrics"]["input_tokens"] == 10
    assert completed.payload["usage_metrics"]["output_tokens"] == 2
    assert events[4].payload["error_type"] == "RuntimeError"
    assert "error" not in events[4].payload
    assert runtime_provider.requests == []
    usage = asyncio.run(app.get_session_usage("sess_failed_compaction_wrapper"))
    assert usage.model_steps == 1
    assert usage.usage.total_tokens == 12
    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    private_completed = asyncio.run(
        _private_events_for_public_events(
            app.session_store,
            "sess_failed_compaction_wrapper",
            [completed],
        )
    )[0]
    assert [event.id for event in budget_events] == [private_completed.id]


def test_cayu_app_keeps_compaction_completion_when_stream_fails_after_completed():
    class CompletedThenFailingProvider(ModelProvider):
        name = "fake"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            yield ModelStreamEvent.text_delta("completed summary")
            yield ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 14, "output_tokens": 2},
                }
            )
            raise RuntimeError("stream teardown failed")

    budget_store = InMemoryBudgetStore()
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(budget_store=budget_store)
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=CompletedThenFailingProvider(),
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
                session_id="sess_compaction_stream_teardown_failure",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    completed_events = [event for event in events if event.type == EventType.MODEL_COMPLETED]
    assert len(completed_events) == 1
    completed = completed_events[0]
    assert completed.payload["usage_metrics"]["input_tokens"] == 14
    assert completed.payload["usage_metrics"]["output_tokens"] == 2
    assert "compaction_attempt_id" not in completed.payload
    failed = next(event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED)
    assert failed.payload["error_type"] == "RuntimeError"
    assert "error" not in failed.payload
    assert runtime_provider.requests == []
    usage = asyncio.run(app.get_session_usage("sess_compaction_stream_teardown_failure"))
    assert usage.model_steps == 1
    assert usage.usage.total_tokens == 16
    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    private_completed = asyncio.run(
        _private_events_for_public_events(
            app.session_store,
            "sess_compaction_stream_teardown_failure",
            [completed],
        )
    )[0]
    assert [event.id for event in budget_events] == [private_completed.id]


def test_cayu_app_preserves_provider_order_when_wrapper_recovers_earlier_draft():
    class DraftRecoveringCompactor(ContextCompactor):
        def __init__(self, provider: ModelProvider) -> None:
            self.provider = provider

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            draft = await ModelCompactor(
                provider=self.provider,
                model="draft-model",
            ).compact(request)
            try:
                await ModelCompactor(
                    provider=self.provider,
                    model="refinement-model",
                ).compact(request)
            except ValueError:
                return draft
            raise AssertionError("the refinement unexpectedly returned a valid summary")

    budget_store = InMemoryBudgetStore()
    compactor_provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("usable draft"),
                ModelStreamEvent.completed(
                    {
                        "model": "draft-model",
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta(" \n\t"),
                ModelStreamEvent.completed(
                    {
                        "model": "refinement-model",
                        "usage": {"input_tokens": 20, "output_tokens": 2},
                    }
                ),
            ],
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 3, "output_tokens": 1}}),
        ]
    )
    app = CayuApp(
        budget_store=budget_store,
        request_footprint=RequestFootprintConfig(enabled=False),
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=DraftRecoveringCompactor(compactor_provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_recovered_compaction_draft",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    completed_events = [event for event in events if event.type == EventType.MODEL_COMPLETED]
    assert len(completed_events) == 3
    draft, refinement, runtime_completed = completed_events
    assert draft.payload["requested_model"] == "draft-model"
    assert draft.payload["usage_metrics"]["input_tokens"] == 10
    assert refinement.payload["requested_model"] == "refinement-model"
    assert refinement.payload["compaction_outcome"] == "invalid_summary"
    assert refinement.payload["usage_metrics"]["input_tokens"] == 20
    assert runtime_completed.payload["usage_metrics"]["input_tokens"] == 3
    assert events[-1].type == EventType.SESSION_COMPLETED
    usage = asyncio.run(app.get_session_usage("sess_recovered_compaction_draft"))
    assert usage.model_steps == 3
    assert usage.usage.input_tokens == 33
    assert usage.usage.output_tokens == 4
    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    private_completed_events = asyncio.run(
        _private_events_for_public_events(
            app.session_store,
            "sess_recovered_compaction_draft",
            completed_events,
        )
    )
    assert [event.id for event in budget_events] == [event.id for event in private_completed_events]


def test_cayu_app_rejects_returned_only_completion_and_keeps_anchored_inner_call():
    class ReturnedFirstCompactor(ContextCompactor):
        def __init__(self, provider: ModelProvider) -> None:
            self.provider = provider

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            inner = await ModelCompactor(
                provider=self.provider,
                model="inner-model",
            ).compact(request)
            return CompactionResult(
                summary=inner.summary,
                covered_message_count=inner.covered_message_count,
                model_completed_payloads=[
                    {
                        "model": "custom-model",
                        "provider_name": "custom",
                        "requested_model": "custom-model",
                        "purpose": "context_compaction",
                        "compactor": type(self).__name__,
                        "usage_metrics": {
                            "provider_name": "custom",
                            "requested_model": "custom-model",
                            "model": "custom-model",
                            "input_tokens": 10,
                            "output_tokens": 1,
                            "total_tokens": 11,
                            "reasoning_output_tokens": 0,
                            "cache": {
                                "read_tokens": 0,
                                "write_tokens": 0,
                                "cached_input_tokens": 0,
                                "uncached_input_tokens": 10,
                            },
                        },
                    },
                    *inner.model_completed_payloads,
                ],
            )

    budget_store = InMemoryBudgetStore()
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("inner summary"),
            ModelStreamEvent.completed(
                {
                    "model": "inner-model",
                    "usage": {"input_tokens": 20, "output_tokens": 2},
                }
            ),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 3, "output_tokens": 1}}),
        ]
    )
    app = CayuApp(
        budget_store=budget_store,
        request_footprint=RequestFootprintConfig(enabled=False),
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ReturnedFirstCompactor(compactor_provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_returned_completion_before_inner",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    durable_events = asyncio.run(
        app.session_store.load_events("sess_returned_completion_before_inner")
    )
    completed_events = [
        event for event in durable_events if event.type == EventType.MODEL_COMPLETED
    ]
    assert len(completed_events) == 1
    inner = completed_events[0]
    assert inner.payload["requested_model"] == "inner-model"
    assert inner.payload["usage_metrics"]["input_tokens"] == 20
    assert inner.payload["model_step_id"].startswith("mstep_")
    assert inner.payload["model_attempt_id"].startswith("matt_")
    assert events[-1].type == EventType.SESSION_FAILED
    assert (
        events[-1].payload["error"]
        == "Compaction completion was observed outside its provider dispatch."
    )
    assert runtime_provider.requests == []
    usage = asyncio.run(app.get_session_usage("sess_returned_completion_before_inner"))
    assert usage.model_steps == 1
    assert usage.usage.input_tokens == 20
    assert usage.usage.output_tokens == 2
    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    assert [event.id for event in budget_events] == [inner.id]


@pytest.mark.parametrize(
    ("invalid_value", "invalid_location", "expected_error_code", "expected_error_path"),
    [
        ("timeout\x00workload-secret-value", "top-level", "nul_character", "$/#2"),
        (float("nan"), "top-level", "non_finite_number", "$/#2"),
        ("timeout\x00workload-secret-value", "usage", "nul_character", "$/#1/#2"),
        (
            "timeout\x00workload-secret-value",
            "model-and-usage",
            "nul_character",
            "$/#0",
        ),
    ],
    ids=["nul", "nan", "nested-usage-aux", "invalid-model-and-nested-usage-aux"],
)
def test_cayu_app_retains_only_authoritative_usage_when_completion_metadata_is_not_durable(
    invalid_value: object,
    invalid_location: str,
    expected_error_code: str,
    expected_error_path: str,
):
    store = InMemorySessionStore()
    budget_store = InMemoryBudgetStore()
    secret_key = "workload-secret-key"
    usage = {"input_tokens": 30, "output_tokens": 4}
    payload: dict[str, Any] = {
        "model": "summary-model",
        "usage": usage,
    }
    if invalid_location == "usage":
        usage[secret_key] = invalid_value
    elif invalid_location == "model-and-usage":
        payload["model"] = invalid_value
        usage[secret_key] = invalid_value
    else:
        payload[secret_key] = invalid_value
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("valid summary"),
            ModelStreamEvent.model_construct(
                type=ModelStreamEventType.COMPLETED,
                delta="",
                payload=payload,
                completion=None,
            ),
        ]
    )
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(session_store=store, budget_store=budget_store)
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
                session_id="sess_unsafe_compaction_metadata",
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
    completed = events[3]
    assert completed.payload["provider_name"] == "fake"
    assert completed.payload["requested_model"] == "summary-model"
    assert completed.payload["model"] == "summary-model"
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
            "invalid compaction usage telemetry"
        )
    assert "provider_note" not in completed.payload
    assert completed.payload["compaction_outcome"] == "invalid_completion_metadata"
    assert secret_key not in completed.payload
    assert "compaction_attempt_id" not in completed.payload
    assert events[4].payload["error_type"] == "DurableValueError"
    assert "error" not in events[4].payload
    assert events[-1].payload == {
        "compaction_failure": _expected_compaction_failure(events[4]),
        "failure_evidence": _expected_failure_evidence(
            "sess_unsafe_compaction_metadata", "DurableValueError", "ContextBuildError"
        ),
        "error": "Operation failed with a non-portable diagnostic.",
        "error_type": "DurableValueError",
        "durable_value_error_code": expected_error_code,
        "durable_value_error_path": expected_error_path,
    }
    rendered = json.dumps(
        [event.model_dump(mode="json") for event in events],
        allow_nan=False,
    )
    assert secret_key not in rendered
    assert "timeout" not in rendered
    assert "workload-secret-value" not in rendered
    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []
    usage = asyncio.run(app.get_session_usage("sess_unsafe_compaction_metadata"))
    assert usage.model_steps == 1
    assert usage.usage.total_tokens == (34 if invalid_location == "top-level" else 0)
    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    private_completed = asyncio.run(
        _private_events_for_public_events(
            store,
            "sess_unsafe_compaction_metadata",
            [completed],
        )
    )[0]
    assert [event.id for event in budget_events] == [private_completed.id]


@pytest.mark.parametrize(
    ("summary_events", "expected_error", "expected_error_type", "expected_error_code"),
    [
        ([], "`summary` cannot be blank.", "ValueError", None),
        (
            [ModelStreamEvent.text_delta(" \n\t")],
            "`summary` cannot be blank.",
            "ValueError",
            None,
        ),
        (
            [ModelStreamEvent.text_delta("invalid\x00summary")],
            "`summary` must not contain NUL characters. [code=nul_character; path=$]",
            "DurableValueError",
            "nul_character",
        ),
        (
            [ModelStreamEvent.text_delta("invalid\ud800summary")],
            "`summary` must not contain Unicode surrogate code points. "
            "[code=unicode_surrogate; path=$]",
            "DurableValueError",
            "unicode_surrogate",
        ),
    ],
    ids=["blank", "whitespace", "nul", "surrogate"],
)
def test_cayu_app_counts_completed_compaction_with_invalid_summary(
    summary_events: list[ModelStreamEvent],
    expected_error: str,
    expected_error_type: str,
    expected_error_code: str | None,
):
    store = InMemorySessionStore()
    task_store = InMemoryTaskStore()
    asyncio.run(
        task_store.create_task(
            TaskCreate(
                task_id="task_invalid_compaction_summary",
                type="compaction-recovery",
                assigned_agent_name="assistant",
            )
        )
    )
    budget_store = InMemoryBudgetStore()
    compactor_provider = FakeProvider(
        [
            *summary_events,
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 100, "output_tokens": 0},
                }
            ),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("unused"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(
        session_store=store,
        task_store=task_store,
        budget_store=budget_store,
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
                session_id="sess_invalid_compaction_summary",
                causal_budget_id="job_invalid_compaction_summary",
                task_id="task_invalid_compaction_summary",
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
        EventType.TASK_STARTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
        EventType.TASK_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    completed = events[4]
    assert completed.payload["purpose"] == "context_compaction"
    assert completed.payload["compactor"] == "ModelCompactor"
    assert completed.payload["compaction_outcome"] == "invalid_summary"
    assert completed.payload["provider_name"] == "fake"
    assert completed.payload["requested_model"] == "summary-model"
    assert completed.payload["model"] == "summary-model"
    assert completed.payload["usage"] == {"input_tokens": 100, "output_tokens": 0}
    assert completed.payload["usage_metrics"] == {
        "provider_name": "fake",
        "requested_model": "summary-model",
        "model": "summary-model",
        "input_tokens": 100,
        "output_tokens": 0,
        "total_tokens": 100,
        "reasoning_output_tokens": 0,
        "cache": {
            "read_tokens": 0,
            "write_tokens": 0,
            "cached_input_tokens": 0,
            "uncached_input_tokens": 100,
        },
    }
    failure = events[5]
    assert failure.payload["error_type"] == expected_error_type
    assert "error" not in failure.payload
    expected_disposition = {
        "phase": "provider_dispatch",
        "reason": "internal_failed",
        "retryable": False,
        "provider_dispatch_disposition": "dispatched",
        "recovery_action": "reconcile_completion",
    }
    assert {key: failure.payload[key] for key in expected_disposition} == expected_disposition
    assert type(failure.payload["elapsed_ms"]) is int
    compaction_failure = {
        **expected_disposition,
        "elapsed_ms": failure.payload["elapsed_ms"],
    }
    assert events[7].payload["step_count"] == 1
    assert events[7].payload["token_usage"]["input_tokens"] == 100
    terminal = events[8]
    assert terminal.payload["error_type"] == expected_error_type
    assert terminal.payload["compaction_failure"] == compaction_failure
    assert type(terminal.payload["execution_profile_fingerprint"]) is str
    assert type(terminal.payload["runtime_task_failure_id"]) is str
    if expected_error_type == "DurableValueError":
        assert expected_error_code is not None
        assert terminal.payload["error"] == "Operation failed with a non-portable diagnostic."
        assert terminal.payload["durable_value_error_code"] == expected_error_code
        assert terminal.payload["durable_value_error_path"] == "$"
    else:
        assert terminal.payload["error"] == expected_error
    task = asyncio.run(task_store.load_task("task_invalid_compaction_summary"))
    assert task is not None
    assert task.status == TaskStatus.FAILED
    assert task.error is not None
    assert task.error["compaction_failure"] == compaction_failure
    assert runtime_provider.requests == []
    assert checkpoint_without_active_invocation_profile(
        asyncio.run(store.load_checkpoint("sess_invalid_compaction_summary"))
    ) == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}

    usage = asyncio.run(app.get_session_usage("sess_invalid_compaction_summary"))
    assert usage.model_steps == 1
    assert usage.usage.input_tokens == 100
    assert usage.usage.total_tokens == 100
    causal_usage = asyncio.run(app.get_causal_budget_usage("job_invalid_compaction_summary"))
    assert causal_usage.model_steps == 1
    assert causal_usage.usage.input_tokens == 100

    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="fake",
                model="summary-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("10"),
            ),
        )
    )
    cost = asyncio.run(
        app.get_session_cost(
            "sess_invalid_compaction_summary",
            pricing,
        )
    )
    assert cost.model_steps == 1
    assert cost.priced_model_steps == 1
    assert cost.total_cost == Decimal("0.0001")
    causal_cost = asyncio.run(app.get_causal_budget_cost("job_invalid_compaction_summary", pricing))
    assert causal_cost.model_steps == 1
    assert causal_cost.priced_model_steps == 1
    assert causal_cost.total_cost == Decimal("0.0001")

    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    private_completed = asyncio.run(
        _private_events_for_public_events(
            store,
            "sess_invalid_compaction_summary",
            [completed],
        )
    )[0]
    assert [event.id for event in budget_events] == [private_completed.id]


def test_cayu_app_counts_completed_compaction_rejected_for_tool_call():
    store = InMemorySessionStore()
    budget_store = InMemoryBudgetStore()
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_compaction_tool",
                name="unexpected_tool",
                arguments={},
            ),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "finish_reason": "tool_calls",
                    "usage": {"input_tokens": 80, "output_tokens": 6},
                }
            ),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("unused"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store, budget_store=budget_store)
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
                session_id="sess_rejected_compaction_tool_call",
                causal_budget_id="job_rejected_compaction_tool_call",
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
    completed = events[3]
    assert completed.payload["purpose"] == "context_compaction"
    assert completed.payload["compactor"] == "ModelCompactor"
    assert completed.payload["compaction_outcome"] == "rejected_tool_call"
    assert completed.payload["finish_reason"] == "tool_calls"
    assert completed.payload["usage_metrics"]["input_tokens"] == 80
    assert completed.payload["usage_metrics"]["output_tokens"] == 6
    assert "compaction_attempt_id" not in completed.payload
    assert events[4].payload["error_type"] == "_CompactionToolCallError"
    assert "error" not in events[4].payload
    assert events[5].payload["step_count"] == 1
    assert events[5].payload["token_usage"]["total_tokens"] == 86
    assert events[6].payload["error"] == "Compaction model must not call tools."
    assert runtime_provider.requests == []
    assert checkpoint_without_active_invocation_profile(
        asyncio.run(store.load_checkpoint("sess_rejected_compaction_tool_call"))
    ) == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}

    usage = asyncio.run(app.get_session_usage("sess_rejected_compaction_tool_call"))
    assert usage.model_steps == 1
    assert usage.usage.input_tokens == 80
    assert usage.usage.output_tokens == 6
    causal_usage = asyncio.run(app.get_causal_budget_usage("job_rejected_compaction_tool_call"))
    assert causal_usage.model_steps == 1
    assert causal_usage.usage.total_tokens == 86

    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="fake",
                model="summary-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("10"),
            ),
        )
    )
    cost = asyncio.run(app.get_session_cost("sess_rejected_compaction_tool_call", pricing))
    assert cost.model_steps == 1
    assert cost.priced_model_steps == 1
    assert cost.total_cost == Decimal("0.00014")
    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    private_completed = asyncio.run(
        _private_events_for_public_events(
            store,
            "sess_rejected_compaction_tool_call",
            [completed],
        )
    )[0]
    assert [event.id for event in budget_events] == [private_completed.id]


@pytest.mark.parametrize(
    ("compactor_events", "expected_error", "expected_outcome"),
    [
        (
            [ModelStreamEvent.text_delta("partial summary")],
            "Compaction model stream ended without a completed event.",
            "unfinished_stream",
        ),
        (
            [
                ModelStreamEvent.tool_call(
                    id="call_compaction_tool",
                    name="unexpected_tool",
                    arguments={},
                )
            ],
            "Compaction model must not call tools.",
            "rejected_tool_call",
        ),
    ],
    ids=["partial-text", "tool-call"],
)
def test_cayu_app_does_not_invent_usage_for_unfinished_compaction_stream(
    compactor_events: list[ModelStreamEvent],
    expected_error: str,
    expected_outcome: str,
):
    store = InMemorySessionStore()
    budget_store = InMemoryBudgetStore()
    compactor_provider = FakeProvider(compactor_events)
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("unused"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store, budget_store=budget_store)
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
                session_id="sess_unfinished_compaction",
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
    uncertain = events[3]
    assert uncertain.payload["compaction_outcome"] == expected_outcome
    assert uncertain.payload["usage_unavailable_reason"]
    assert "usage_metrics" not in uncertain.payload
    assert events[4].payload["error_type"] in {"RuntimeError", "_CompactionToolCallError"}
    assert "error" not in events[4].payload
    assert events[5].payload["step_count"] == 1
    assert events[6].payload["error"] == expected_error
    assert runtime_provider.requests == []
    assert checkpoint_without_active_invocation_profile(
        asyncio.run(store.load_checkpoint("sess_unfinished_compaction"))
    ) == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}

    usage = asyncio.run(app.get_session_usage("sess_unfinished_compaction"))
    assert usage.model_steps == 1
    assert usage.usage.total_tokens == 0
    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    private_uncertain = asyncio.run(
        _private_events_for_public_events(
            store,
            "sess_unfinished_compaction",
            [uncertain],
        )
    )[0]
    assert [event.id for event in budget_events] == [private_uncertain.id]


def test_unfinished_automatic_compaction_fails_closed_after_restart() -> None:
    class UnfinishedThenSuccessfulProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                yield ModelStreamEvent.text_delta("partial summary")
                return
            yield ModelStreamEvent.text_delta("must not run")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

    async def run() -> None:
        store = InMemorySessionStore()
        budget_store = InMemoryBudgetStore()
        compactor_provider = UnfinishedThenSuccessfulProvider()
        policy = BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=compaction_price_book(),
                ),
            )
        )

        def new_app() -> CayuApp:
            app = CayuApp(
                session_store=store,
                budget_store=budget_store,
                budget_policy=policy,
                enable_logging=False,
            )
            app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
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
            return app

        first_events = await collect_events(
            new_app(),
            RunRequest(
                agent_name="assistant",
                session_id="sess_unfinished_compaction_first_process",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
        assert first_events[-1].type == EventType.SESSION_FAILED
        uncertain = next(
            event
            for event in first_events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        )
        assert uncertain.payload["compaction_outcome"] == "unfinished_stream"
        assert uncertain.payload["usage_unavailable_reason"]

        restart_events = await collect_events(
            new_app(),
            RunRequest(
                agent_name="assistant",
                session_id="sess_unfinished_compaction_second_process",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )

        assert len(compactor_provider.requests) == 1
        assert any(event.type == EventType.BUDGET_LIMIT_REACHED for event in restart_events)
        assert not any(
            event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
            for event in restart_events
        )
        durable_budget_events = await budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
        private_uncertain = (
            await _private_events_for_public_events(
                store,
                "sess_unfinished_compaction_first_process",
                [uncertain],
            )
        )[0]
        assert [event.id for event in durable_budget_events] == [private_uncertain.id]

    asyncio.run(run())


def test_automatic_compaction_real_cancellation_records_uncertain_dispatch() -> None:
    class BlockingPartialProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.partial_emitted = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            yield ModelStreamEvent.text_delta("partial summary")
            self.partial_emitted.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    async def run() -> None:
        session_id = "sess_cancelled_compaction_dispatch"
        store = InMemorySessionStore()
        budget_store = InMemoryBudgetStore()
        compactor_provider = BlockingPartialProvider()
        app = CayuApp(
            session_store=store,
            budget_store=budget_store,
            enable_logging=False,
        )
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
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

        task = asyncio.create_task(
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
        await asyncio.wait_for(compactor_provider.partial_emitted.wait(), timeout=5)
        task.cancel("cancel compaction after provider dispatch")
        assert task.cancelling() == 1
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert exc_info.value.args == ("cancel compaction after provider dispatch",)
        assert task.cancelled()
        assert compactor_provider.cancelled.is_set()
        events = await store.load_events(session_id)
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert completions[0].payload["compaction_outcome"] == "cancelled"
        assert completions[0].payload["usage_unavailable_reason"]
        assert sum(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in events) == 1
        assert checkpoint_without_active_invocation_profile(
            await store.load_checkpoint(session_id)
        ) == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}
        usage = await app.get_session_usage(session_id)
        assert usage.model_steps == 1
        assert usage.usage.total_tokens == 0
        durable_budget_events = await budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
        assert [event.id for event in durable_budget_events] == [completions[0].id]

    asyncio.run(run())


def test_automatic_compaction_preserves_later_cancellation_during_termination_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingPartialProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.partial_emitted = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            yield ModelStreamEvent.text_delta("partial summary")
            self.partial_emitted.set()
            await asyncio.Event().wait()

    async def run() -> None:
        session_id = "sess_cancelled_compaction_termination_persistence"
        store = InMemorySessionStore()
        compactor_provider = BlockingPartialProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
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

        termination_persistence_started = asyncio.Event()
        allow_termination_persistence = asyncio.Event()
        original_emit_many = app._event_writer.emit_many

        async def block_termination_persistence(
            emitted_session_id: str,
            events: list[Event],
        ) -> list[Event]:
            if any(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in events):
                termination_persistence_started.set()
                await allow_termination_persistence.wait()
            return await original_emit_many(emitted_session_id, events)

        monkeypatch.setattr(app._event_writer, "emit_many", block_termination_persistence)
        task = asyncio.create_task(
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
        await asyncio.wait_for(compactor_provider.partial_emitted.wait(), timeout=5)
        task.cancel("primary cancellation during provider dispatch")
        await asyncio.wait_for(termination_persistence_started.wait(), timeout=10)

        task.cancel("later cancellation during termination persistence")
        await asyncio.sleep(0)
        assert not task.done()
        allow_termination_persistence.set()
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await task

        def leaves(error: BaseException) -> list[BaseException]:
            if isinstance(error, BaseExceptionGroup):
                return [leaf for nested in error.exceptions for leaf in leaves(nested)]
            return [error]

        cancellations = [
            leaf for leaf in leaves(exc_info.value) if isinstance(leaf, asyncio.CancelledError)
        ]
        assert [cancellation.args for cancellation in cancellations] == [
            ("primary cancellation during provider dispatch",),
            ("later cancellation during termination persistence",),
        ]
        assert len(leaves(exc_info.value)) == 2
        assert task.cancelling() == 0
        assert not task.cancelled()

        events = await store.load_events(session_id)
        assert (
            sum(
                event.type == EventType.MODEL_COMPLETED
                and event.payload.get("purpose") == "context_compaction"
                for event in events
            )
            == 1
        )
        assert sum(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in events) == 1
        assert len(compactor_provider.requests) == 1

    asyncio.run(run())


def test_cayu_app_emits_compaction_failed_event_before_session_failure():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("unused"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=FailingCompactor(),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_failed",
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
        EventType.CONTEXT_COMPACTION_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[2].payload == {
        "checkpoint": "context_compaction",
        "execution_profile_fingerprint": events[1].payload["execution_profile_fingerprint"],
        "compactor": "FailingCompactor",
        "compacted_transcript_cursor": 0,
        "previous_compacted_transcript_cursor": 0,
        "newly_compacted_message_count": 0,
        "recent_message_count": 1,
        "error_type": "RuntimeError",
        "requested_source_start": 0,
        "requested_source_end": 2,
        "represented_source_start": 0,
        "represented_source_end": 0,
        "represented_message_count": 0,
        "coverage_mode": "failed",
        "chunk_count": 0,
        "chunk_mode": "failed",
        "compaction_failed": True,
        "model_step_id": events[1].payload["model_step_id"],
    }
    assert events[4].payload == {
        "failure_evidence": _expected_failure_evidence(
            "sess_compaction_failed", "RuntimeError", "ContextBuildError"
        ),
        "error": "compaction unavailable",
        "error_type": "RuntimeError",
    }
    assert provider.requests == []
    assert checkpoint_without_active_invocation_profile(
        asyncio.run(store.load_checkpoint("sess_compaction_failed"))
    ) == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}


def test_cayu_app_emits_compaction_events_before_checkpoint_failure():
    class BrokenCheckpointStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def publish_checkpoint_and_events(self, session_id, **kwargs):
            del session_id, kwargs
            raise RuntimeError("checkpoint unavailable")

    store = BrokenCheckpointStore()
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("unused"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
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
                session_id="sess_checkpoint_failure_after_compaction",
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
    assert EventType.CONTEXT_COMPACTION_COMPLETED not in {event.type for event in events}
    assert EventType.SESSION_CHECKPOINTED not in {event.type for event in events}
    expected_disposition = {
        "phase": "checkpoint_installation",
        "reason": "checkpoint_failed",
        "retryable": False,
        "provider_dispatch_disposition": "dispatched",
        "recovery_action": "fail_closed",
    }
    failure = events[4]
    assert {key: failure.payload[key] for key in expected_disposition} == expected_disposition
    assert type(failure.payload["elapsed_ms"]) is int
    assert failure.payload["compaction_failed"] is True
    assert events[6].payload["error"] == "checkpoint unavailable"
    assert events[6].payload["error_type"] == "RuntimeError"
    assert {
        key: events[6].payload["compaction_failure"][key] for key in expected_disposition
    } == expected_disposition
    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []


def test_cayu_app_checkpoint_compaction_ignores_cursor_without_valid_summary():
    store = InMemorySessionStore()
    compactor = RecordingCompactor()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    session = asyncio.run(
        store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id="sess_bad_checkpoint_pair",
                messages=[],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=app,
            ),
        )
    )
    asyncio.run(store.update_status(session.id, SessionStatus.COMPLETED))
    asyncio.run(
        store.append_transcript_messages(
            session.id,
            [
                Message.text("user", "old"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current"),
            ],
        )
    )
    asyncio.run(
        store.checkpoint(
            session.id,
            {
                "context_compaction": {
                    "version": 2,
                    "summary": 123,
                    "compacted_transcript_cursor": 2,
                    "metadata": {},
                }
            },
        )
    )

    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session.id,
                messages=[Message.text("user", "follow up")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert [message.content[0].text for message in compactor.requests[0].messages] == [
        "old",
        "old answer",
        "current",
    ]
    assert compactor.requests[0].existing_summary is None
    assert events[3].payload == {
        "checkpoint": "context_compaction",
        "execution_profile_fingerprint": events[4].payload["execution_profile_fingerprint"],
        "compacted_transcript_cursor": 3,
        "previous_compacted_transcript_cursor": 0,
        "newly_compacted_message_count": 3,
        "recent_message_count": 1,
        "model_step_id": events[4].payload["model_step_id"],
    }


def test_cayu_app_checkpoint_compaction_ignores_summary_without_valid_cursor():
    store = InMemorySessionStore()
    compactor = RecordingCompactor()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    session = asyncio.run(
        store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id="sess_bad_checkpoint_cursor",
                messages=[],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=app,
            ),
        )
    )
    asyncio.run(store.update_status(session.id, SessionStatus.COMPLETED))
    asyncio.run(
        store.append_transcript_messages(
            session.id,
            [
                Message.text("user", "old"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current"),
            ],
        )
    )
    asyncio.run(
        store.checkpoint(
            session.id,
            {
                "context_compaction": {
                    "version": 2,
                    "summary": "stale summary",
                    "compacted_transcript_cursor": "bad",
                    "metadata": {},
                }
            },
        )
    )

    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session.id,
                messages=[Message.text("user", "follow up")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert [message.content[0].text for message in compactor.requests[0].messages] == [
        "old",
        "old answer",
        "current",
    ]
    assert compactor.requests[0].existing_summary is None
    assert events[3].payload == {
        "checkpoint": "context_compaction",
        "execution_profile_fingerprint": events[4].payload["execution_profile_fingerprint"],
        "compacted_transcript_cursor": 3,
        "previous_compacted_transcript_cursor": 0,
        "newly_compacted_message_count": 3,
        "recent_message_count": 1,
        "model_step_id": events[4].payload["model_step_id"],
    }


def test_cayu_app_resume_uses_checkpointed_compaction_summary():
    store = InMemorySessionStore()
    compactor = RecordingCompactor()
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
        context_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_resume_compaction",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )
    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_resume_compaction",
                messages=[Message.text("user", "follow up")],
            ),
        )
    )

    assert [event.type for event in resume_events] == [
        EventType.SESSION_RESUMED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(compactor.requests) == 2
    assert [message.content[0].text for message in compactor.requests[1].messages] == [
        "current",
        "first answer",
    ]
    assert compactor.requests[1].existing_summary == "old|old answer"

    provider_context = provider.requests[1].messages
    assert [message.role for message in provider_context] == ["user", "user"]
    assert provider_context[0].content[0].text == (
        "Previous session context summary:\nold|old answer|current|first answer"
    )
    assert provider_context[1].content[0].text == "follow up"

    transcript = asyncio.run(store.load_transcript("sess_resume_compaction"))
    assert [message.content[0].text for message in transcript] == [
        "old",
        "old answer",
        "current",
        "first answer",
        "follow up",
        "second answer",
    ]


def test_cayu_app_freezes_tool_declarations_at_registration():
    store = InMemorySessionStore()
    tool = EchoTool()
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
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    tool.spec = ToolSpec(
        name="mutated",
        description="Mutated.",
        input_schema={"type": "object", "properties": {}},
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_frozen_tool",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert provider.requests[0].tools == [
        {
            "name": "echo",
            "description": "Echo text.",
            "input_schema": EchoTool.spec.input_schema,
        }
    ]
    assert events[3].type == EventType.TOOL_CALL_STARTED
    assert events[3].tool_name == "echo"
    assert events[4].type == EventType.TOOL_CALL_COMPLETED


_AUTOMATIC_COMPACTION_PROCESS_SESSION_ID = "automatic-compaction-process-loss"


class ProcessLossSuccessfulCompactionProvider(ModelProvider):
    name = "fake"

    def __init__(self, marker: str) -> None:
        self._marker = Path(marker)

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="test.process-loss-successful-compaction-provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        prior = int(self._marker.read_text(encoding="utf-8")) if self._marker.exists() else 0
        self._marker.write_text(str(prior + 1), encoding="utf-8")
        yield ModelStreamEvent.text_delta("durable summary")
        yield ModelStreamEvent.completed(
            {
                "model": request.model,
                "usage": {"input_tokens": 3, "output_tokens": 2},
            }
        )


def _automatic_compaction_process_budget_policy() -> BudgetPolicy:
    return BudgetPolicy(
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


def _run_until_automatic_compaction_process_exit(
    database: str,
    ledger_database: str,
    marker: str,
    crash_point: str,
) -> None:
    async def run() -> None:
        store = ProcessLossAutomaticCompactionStore(
            database,
            crash_point=crash_point,
        )
        provider = ProcessLossSuccessfulCompactionProvider(marker)
        ledger = SQLiteBudgetLedger(ledger_database)
        app = CayuApp(
            session_store=store,
            budget_ledger=ledger,
            budget_policy=_automatic_compaction_process_budget_policy(),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=_AUTOMATIC_COMPACTION_PROCESS_SESSION_ID,
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            )
        ):
            pass
        raise AssertionError("the child did not reach the compaction crash point")

    asyncio.run(run())


@pytest.mark.parametrize("crash_point", ["completed", "promoted"])
def test_successful_automatic_compaction_process_loss_cannot_redispatch(
    tmp_path: Path,
    crash_point: str,
) -> None:
    database = tmp_path / f"automatic-compaction-{crash_point}.sqlite"
    ledger_database = tmp_path / f"automatic-compaction-{crash_point}-budget.sqlite"
    marker = tmp_path / f"automatic-compaction-{crash_point}-provider-count"
    repository_root = Path(__file__).resolve().parents[2]
    child_script = (
        "from tests.core.test_runtime import "
        "_run_until_automatic_compaction_process_exit as run; "
        f"run({str(database)!r}, {str(ledger_database)!r}, "
        f"{str(marker)!r}, {crash_point!r})"
    )
    child_environment = os.environ.copy()
    existing_python_path = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (
            str(Path(cayu.__file__).resolve().parent.parent),
            existing_python_path,
        )
        if path
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

    assert child.returncode == _AUTOMATIC_COMPACTION_PROCESS_EXIT_CODE, child.stderr
    assert marker.read_text(encoding="utf-8") == "1"

    store = SQLiteSessionStore(database)
    active = asyncio.run(
        store.load_active_model_completion_stage(_AUTOMATIC_COMPACTION_PROCESS_SESSION_ID)
    )
    durable_events = asyncio.run(store.load_events(_AUTOMATIC_COMPACTION_PROCESS_SESSION_ID))
    completions = [
        event
        for event in durable_events
        if event.type is EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    ]
    assert not any(event.type is EventType.SESSION_CHECKPOINTED for event in durable_events)
    if crash_point == "completed":
        assert active is not None
        assert active.stage.state == "completed"
        assert completions == []
    else:
        assert active is None
        assert len(completions) == 1
        assert type(completions[0].payload.get("parent_model_step_id")) is str

    restarted_provider = ProcessLossSuccessfulCompactionProvider(str(marker))
    ledger = SQLiteBudgetLedger(ledger_database)
    restarted = CayuApp(
        session_store=store,
        budget_ledger=ledger,
        budget_policy=_automatic_compaction_process_budget_policy(),
        enable_logging=False,
    )
    restarted.register_provider(restarted_provider, default=True)
    restarted.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=restarted_provider,
                model="summary-model",
            ),
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )

    recovery_failure: ModelCompletionManualRecoveryRequired | None = None
    try:
        asyncio.run(
            restarted.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=_AUTOMATIC_COMPACTION_PROCESS_SESSION_ID,
                    inactive_for_seconds=0,
                )
            )
        )
    except ModelCompletionManualRecoveryRequired as failure:
        recovery_failure = failure
    if crash_point == "completed":
        assert recovery_failure is not None

    recovered_events = asyncio.run(store.load_events(_AUTOMATIC_COMPACTION_PROCESS_SESSION_ID))
    recovered_completions = [
        event
        for event in recovered_events
        if event.type is EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    ]
    assert len(recovered_completions) == 1
    assert type(recovered_completions[0].payload.get("parent_model_step_id")) is str
    assert not any(event.type is EventType.SESSION_CHECKPOINTED for event in recovered_events)
    completed_model_step_id = recovered_completions[0].payload["model_step_id"]
    assert type(completed_model_step_id) is str
    completed_stage = asyncio.run(
        store.load_model_completion_stage(
            _AUTOMATIC_COMPACTION_PROCESS_SESSION_ID,
            f"{completed_model_step_id}:dispatch:0",
        )
    )
    assert completed_stage is not None
    assert len(completed_stage.reservation_ids) == 1
    reservation = asyncio.run(ledger.load_reservation(completed_stage.reservation_ids[0]))
    assert reservation is not None
    assert reservation.status == "reconciled"
    assert reservation.actual_amount == Decimal("0.000023")

    if recovery_failure is not None:
        asyncio.run(
            interrupt_and_release_test_invocation(
                store,
                _AUTOMATIC_COMPACTION_PROCESS_SESSION_ID,
            )
        )
    blocked_events = asyncio.run(
        collect_resume_events(
            restarted,
            ResumeRequest(
                session_id=_AUTOMATIC_COMPACTION_PROCESS_SESSION_ID,
                messages=[Message.text("user", "retry after process loss")],
            ),
        )
    )
    assert marker.read_text(encoding="utf-8") == "1"
    assert blocked_events[-1].type is EventType.SESSION_FAILED
    assert "no later durable context checkpoint" in blocked_events[-1].payload["error"]
    assert not any(
        event.type is EventType.MODEL_STARTED
        and event.payload.get("purpose") == "context_compaction"
        for event in blocked_events
    )
    asyncio.run(ledger.close())
    asyncio.run(store.close())
