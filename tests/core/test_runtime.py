"""Application registration, admission and runtime identity contracts."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import (
    _without_interaction_lifecycle,
    _without_request_evidence,
    collect_events,
)
from tests.runtime.runtime_tool_support import EchoTool, RequireApprovalPolicy, SideEffectTool

import cayu.budgets.base as budgets_module
import cayu.execution_units as execution_units_module
import cayu.runtime._tool_invocation.admission as tool_admission_module
import cayu.runtime._tool_round_executor as tool_round_executor_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetReservation, InMemoryBudgetLedger
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.configuration import CayuConfig, ToolExecutionConfig
from cayu.context.structured_output import STRUCTURED_OUTPUT_TOOL_NAME
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.events import Event, EventType
from cayu.messages import Message, TextPart
from cayu.observability.events import InMemoryEventSink
from cayu.observability.hooks import RuntimeHook
from cayu.providers.base import ModelStreamEvent
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.queries import SessionQuery
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.storage.memory import InMemoryKnowledgeStore, KnowledgeAccessScope, KnowledgeEntry
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.tools.user_input import UserInputTool
from cayu.vaults.redaction import SecretRedactor


class _TestKnowledgeStore(InMemoryKnowledgeStore):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("access_scope", KnowledgeAccessScope.privileged())
        super().__init__(*args, **kwargs)


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
