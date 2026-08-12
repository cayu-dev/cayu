from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from cayu import (
    AgentSpec,
    BeforeToolCallHookContext,
    CayuApp,
    Event,
    EventType,
    InMemorySessionStore,
    Message,
    ModelStreamEvent,
    RunRequest,
    RuntimeHook,
    RuntimeHookContext,
    RuntimeHookPhase,
    ScriptedModelProvider,
    Session,
    SessionStatus,
    Tool,
    ToolCallHookContext,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
    WorkflowSpec,
)
from cayu.core.events import validate_public_custom_event_type
from cayu.runtime import EventQuery, InMemoryEventSink
from cayu.workflows import WorkflowBase

_RESERVED_EVENT_TYPE = "custom.cayu.probe"
_RESERVED_ERROR = "The custom.cayu. namespace is reserved for cayu internals."


class _ProbeTool(Tool):
    spec = ToolSpec(
        name="namespace_probe",
        effect=ToolEffect.NONE,
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        del ctx, args
        return ToolResult(content="ok")


class _ReservedNamespaceHook(RuntimeHook):
    async def before_tool_call(self, context: BeforeToolCallHookContext) -> None:
        await context.emit_custom_event(_RESERVED_EVENT_TYPE)

    async def after_tool_call(self, context: ToolCallHookContext) -> None:
        await context.emit_custom_event(_RESERVED_EVENT_TYPE)

    async def after_session_completed(self, context: RuntimeHookContext) -> None:
        await context.emit_custom_event(_RESERVED_EVENT_TYPE)


class _NamespaceWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="namespace-workflow")

    async def run(self, session_id: str):
        del session_id
        if False:
            yield


class _TrackingHookRuntime:
    def __init__(self) -> None:
        self.emit_called = False

    async def emit_hook_event(
        self,
        *,
        session_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        del session_id, event_type, payload
        self.emit_called = True
        raise AssertionError("Reserved event reached the hook runtime.")


async def _collect(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


@pytest.mark.parametrize(
    "event_type",
    [
        "custom.cayu",
        "custom.cayuse.probe",
        "custom.app.event",
    ],
)
def test_public_custom_event_namespace_accepts_nonreserved_boundaries(
    event_type: str,
) -> None:
    assert validate_public_custom_event_type(event_type) == event_type


@pytest.mark.parametrize(
    "event_type",
    [
        _RESERVED_EVENT_TYPE,
        "custom.cayu.workflow.attempt",
    ],
)
def test_public_custom_event_namespace_rejects_cayu_descendants(event_type: str) -> None:
    with pytest.raises(ValueError, match="reserved for cayu internals"):
        validate_public_custom_event_type(event_type)


@pytest.mark.parametrize(
    "event_type",
    [
        "custom.cayu.",
        "custom",
        "workflow.event",
        " custom.app.event",
        True,
        None,
    ],
)
def test_public_custom_event_namespace_rejects_malformed_values(event_type: object) -> None:
    with pytest.raises(ValueError, match="Custom event types must use"):
        validate_public_custom_event_type(event_type)


def test_reserved_hook_event_fails_all_hook_phases_without_persistence() -> None:
    store = InMemorySessionStore()
    sink = InMemoryEventSink()
    app = CayuApp(
        session_store=store,
        event_sinks=[sink],
        runtime_hooks=[_ReservedNamespaceHook()],
        enable_logging=False,
    )
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_namespace_probe",
                        name="namespace_probe",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        tools=[_ProbeTool()],
    )
    request = RunRequest(
        agent_name="assistant",
        session_id="sess_reserved_hook_namespace",
        messages=[Message.text("user", "run the probe")],
    )

    streamed = asyncio.run(_collect(app, request))
    stored = asyncio.run(store.load_events(request.session_id))
    session = asyncio.run(store.load(request.session_id))

    assert session is not None
    assert session.status is SessionStatus.COMPLETED
    assert _RESERVED_EVENT_TYPE not in {str(event.type) for event in streamed}
    assert _RESERVED_EVENT_TYPE not in {str(event.type) for event in stored}
    assert _RESERVED_EVENT_TYPE not in {str(event.type) for event in sink.events}
    failures = [event for event in stored if event.type == EventType.HOOK_FAILED]
    assert len(failures) == 3
    assert {event.payload["phase"] for event in failures} == {
        "before_tool_call",
        "after_tool_call",
        "after_session_completed",
    }
    assert all(event.payload["error"] == _RESERVED_ERROR for event in failures)
    assert all(event.payload["actions"] == [] for event in failures)


def test_hook_context_rejects_reserved_event_before_custom_runtime_delegation() -> None:
    runtime = _TrackingHookRuntime()
    session = Session(
        id="sess_custom_hook_runtime",
        agent_name="assistant",
        provider_name="custom",
        model="custom-model",
    )
    context = RuntimeHookContext(
        runtime=cast("Any", runtime),
        hook_name="reserved-namespace-hook",
        phase=RuntimeHookPhase.AFTER_SESSION_COMPLETED,
        session=session,
        terminal_event=Event(
            type=EventType.SESSION_COMPLETED,
            session_id=session.id,
        ),
    )

    with pytest.raises(ValueError, match="reserved for cayu internals"):
        asyncio.run(context.emit_custom_event(_RESERVED_EVENT_TYPE))

    assert runtime.emit_called is False
    assert context.actions == []


def test_workflow_rejects_reserved_custom_event_before_fence_or_journal_mutation() -> None:
    app = CayuApp(enable_logging=False)
    context = _NamespaceWorkflow(app).context("wf_reserved_custom_namespace")

    with pytest.raises(ValueError, match="reserved for cayu internals"):
        asyncio.run(context.emit_custom_event(_RESERVED_EVENT_TYPE))

    records = asyncio.run(app.session_store.query_events(EventQuery(session_id=context.session_id)))
    assert records == []


@pytest.mark.parametrize(
    "event_type",
    [
        "custom.cayu",
        "custom.cayuse.probe",
        "custom.app.event",
    ],
)
def test_workflow_persists_nonreserved_custom_event_boundaries(event_type: str) -> None:
    app = CayuApp(enable_logging=False)
    context = _NamespaceWorkflow(app).context(f"wf_{event_type.replace('.', '_')}")

    emitted = asyncio.run(context.emit_custom_event(event_type))
    records = asyncio.run(app.session_store.query_events(EventQuery(session_id=context.session_id)))

    assert emitted.type == event_type
    assert [str(record.event.type) for record in records][-1] == event_type


def test_runtime_batch_rejects_reserved_custom_event_before_persistence() -> None:
    app = CayuApp(enable_logging=False)
    session_id = "sess_reserved_batch_namespace"

    with pytest.raises(ValueError, match="reserved for cayu internals"):
        asyncio.run(
            app.emit_events(
                session_id,
                [
                    Event(type="custom.app.allowed", session_id=session_id),
                    Event(type=_RESERVED_EVENT_TYPE, session_id=session_id),
                ],
            )
        )

    records = asyncio.run(app.session_store.query_events(EventQuery(session_id=session_id)))
    assert records == []
