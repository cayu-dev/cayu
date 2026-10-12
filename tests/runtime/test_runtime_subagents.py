"""Subagents contracts."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator

import pytest
from tests._session_provenance import fixture_session_invocation
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_recovery_support import FailNextRunFenceReleaseStore
from tests.runtime.runtime_session_support import (
    _taint_aware_policy,
    collect_dispatch_events,
    collect_events,
    collect_tool_approval_events,
    collect_user_input_events,
)
from tests.runtime.runtime_tool_support import _ProtectedEmailTool, _TaintSourceTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.approvals.user_input import UserInputResponse
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY
from cayu.sessions.invocation import InvocationOriginTrust, SessionExecutionSource
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.queries import SessionQuery
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.requests import InterruptSessionRequest, RunRequest
from cayu.tasks.dispatch import DispatchRequest
from cayu.tools.policy import TAINT_LABELS_METADATA_KEY, TaintAwareToolPolicy, ToolPolicyDecision
from cayu.tools.subagents import (
    SubagentExecutionMode,
    SubagentResultTool,
    SubagentSpec,
    SubagentTool,
)
from cayu.tools.user_input import UserInputTool


def test_cayu_app_dispatches_existing_session_inline():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("dispatch answer"),
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
                session_id="sess_dispatch_source",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    dispatch_events = asyncio.run(
        collect_dispatch_events(
            app,
            DispatchRequest(
                session_id="sess_dispatch_source",
                dispatch_id="dispatch_1",
                messages=[Message.text("user", "run dispatched work")],
            ),
        )
    )

    assert [event.type for event in dispatch_events] == [
        EventType.SESSION_RESUMED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert dispatch_events[0].payload["dispatch_id"] == PRIVATE_EVENT_AUTHORITY
    assert dispatch_events[0].payload["appended_messages"] == 1
    assert [message.content[0].text for message in provider.requests[1].messages] == [
        "first request",
        "first answer",
        "run dispatched work",
    ]
    session = asyncio.run(store.load("sess_dispatch_source"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_dispatch_inline_cleanup_failure_has_one_forwarding_owner() -> None:
    async def scenario() -> None:
        session_id = "sess_dispatch_cleanup_failure"
        store = FailNextRunFenceReleaseStore()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.text_delta("first answer"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("unused dispatch answer"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "first request")],
            ),
        )

        stream = app.dispatch_inline(
            DispatchRequest(
                session_id=session_id,
                dispatch_id="dispatch_cleanup_failure",
                messages=[Message.text("user", "run dispatched work")],
            )
        )
        interaction_event = await anext(stream)
        assert interaction_event.type == EventType.INTERACTION_STARTED
        first_event = await anext(stream)
        assert first_event.type == EventType.SESSION_RESUMED

        loop = asyncio.get_running_loop()
        reported_contexts: list[dict[str, object]] = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: reported_contexts.append(context))
        try:
            store.fail_next_release = True
            injected = RuntimeError("dispatch consumer rejected event")
            with pytest.raises(RuntimeError, match="dispatch consumer rejected event") as raised:
                await stream.athrow(injected)
            assert raised.value is injected
            cleanup_failure = raised.value.__cause__
            assert isinstance(cleanup_failure, RuntimeError)
            assert str(cleanup_failure) == "run fence release failed"
            assert isinstance(cleanup_failure.__cause__, OSError)
            await asyncio.sleep(0)
            assert reported_contexts == []
        finally:
            loop.set_exception_handler(previous_handler)

        session = await store.load(session_id)
        assert session is not None and session.status == SessionStatus.INTERRUPTED
        assert app._session_control.has_active_tasks(session_id) is False

    asyncio.run(scenario())


def test_subagent_tool_runs_child_session_with_parent_and_causal_linkage():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "Review only the authentication changes.",
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("auth review complete"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("parent received review"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    subagent_tool = SubagentTool(
        app,
        agents={
            "reviewer": SubagentSpec(
                agent_name="reviewer",
                description="Review delegated work.",
            )
        },
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="fake-model"),
        tools=[subagent_tool],
    )
    app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_subagent_parent",
                causal_budget_id="job_subagent",
                messages=[Message.text("user", "Implement and review auth.")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    child_sessions = asyncio.run(
        store.list_sessions(
            SessionQuery(
                parent_session_id="sess_subagent_parent",
            )
        )
    ).sessions
    assert len(child_sessions) == 1
    child = child_sessions[0]
    assert child.agent_name == "reviewer"
    assert child.parent_session_id == "sess_subagent_parent"
    assert child.causal_budget_id == "job_subagent"
    assert child.status == SessionStatus.COMPLETED
    parent = asyncio.run(store.load("sess_subagent_parent"))
    assert parent is not None
    assert child.invocation.origin == parent.invocation.origin
    assert child.invocation.root_session_id == parent.id
    assert child.invocation.source is SessionExecutionSource.SUBAGENT
    subagent_meta = child.metadata["subagent"]
    assert subagent_meta["agent"] == "reviewer"
    assert subagent_meta["agent_name"] == "reviewer"
    assert subagent_meta["context_mode"] == "task_only"
    assert subagent_meta["mode"] == "foreground"
    assert subagent_meta["parent_session_id"] == "sess_subagent_parent"
    # Durable parent->child linkage stamped at spawn (AGT-02): the child records the spawning
    # tool_call_id and idempotency_key so recovery can re-attach it if the parent crashes before
    # TOOL_CALL_COMPLETED. idempotency_key is a runtime-computed hash, so assert shape not value.
    assert subagent_meta["tool_call_id"] == "call_subagent"
    assert isinstance(subagent_meta["idempotency_key"], str) and subagent_meta["idempotency_key"]
    matcher_arguments = {
        "agent": "reviewer",
        "task": "Review only the authentication changes.",
    }
    assert subagent_tool.matches_recoverable_child(
        child,
        parent_invocation=parent.invocation,
        parent_session_id=parent.id,
        causal_budget_id=parent.causal_budget_id,
        environment_name=parent.environment_name,
        tool_call_id=subagent_meta["tool_call_id"],
        idempotency_key=subagent_meta["idempotency_key"],
        arguments=matcher_arguments,
        require_fingerprint=True,
    )
    unrelated = fixture_session_invocation("unrelated-root")
    contradictory_origin = child.invocation.origin.model_copy(
        update={
            "trust": InvocationOriginTrust.HOST_ASSERTED,
            "subject": "unrelated-user",
        }
    )
    contradictory_invocations = (
        child.invocation.model_copy(update={"origin": contradictory_origin}),
        child.invocation.model_copy(update={"root_invocation_id": unrelated.root_invocation_id}),
        child.invocation.model_copy(update={"root_session_id": unrelated.root_session_id}),
        child.invocation.model_copy(update={"source": SessionExecutionSource.SDK_RUN}),
    )
    for invocation in contradictory_invocations:
        contradictory = child.model_copy(update={"invocation": invocation})
        assert not subagent_tool.matches_recoverable_child(
            contradictory,
            parent_invocation=parent.invocation,
            parent_session_id=parent.id,
            causal_budget_id=parent.causal_budget_id,
            environment_name=parent.environment_name,
            tool_call_id=subagent_meta["tool_call_id"],
            idempotency_key=subagent_meta["idempotency_key"],
            arguments=matcher_arguments,
            require_fingerprint=True,
        )

    child_transcript = asyncio.run(store.load_transcript(child.id))
    assert [message.role for message in child_transcript] == ["user", "assistant"]
    assert child_transcript[0].content[0].text == "Review only the authentication changes."
    assert child_transcript[1].content[0].text == "auth review complete"

    parent_transcript = asyncio.run(store.load_transcript("sess_subagent_parent"))
    assert [message.role for message in parent_transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    tool_result = parent_transcript[2].content[0]
    assert tool_result.content == "auth review complete"
    assert tool_result.structured is not None
    assert tool_result.structured["child_session_id"] == child.id
    assert tool_result.structured["parent_session_id"] == "sess_subagent_parent"
    assert tool_result.structured["causal_budget_id"] == "job_subagent"

    assert [request.messages[0].content[0].text for request in provider.requests] == [
        "Implement and review auth.",
        "Review only the authentication changes.",
        "Implement and review auth.",
    ]
    assert provider.requests[2].messages[-1].role == "tool"


def _build_taint_app(
    provider: ModelProvider,
) -> tuple[CayuApp, InMemorySessionStore, _ProtectedEmailTool]:
    """Wire a parent (subagent + taint source) and a reviewer (protected send_email), both under the
    same TaintAwareToolPolicy, sharing one provider whose event batches are consumed in call order."""
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    subagent_tool = SubagentTool(
        app,
        agents={
            "reviewer": SubagentSpec(
                agent_name="reviewer",
                description="Review delegated work.",
            )
        },
    )
    email_tool = _ProtectedEmailTool()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="fake-model"),
        tools=[subagent_tool, _TaintSourceTool()],
        tool_policy=_taint_aware_policy(),
    )
    app.register_agent(
        AgentSpec(name="reviewer", model="fake-model"),
        tools=[email_tool],
        tool_policy=_taint_aware_policy(),
    )
    return app, store, email_tool


def test_subagent_inherits_parent_taint_and_gates_protected_tool():
    # Call order across the shared parent+child provider is deterministic: parent read_web, parent
    # subagent, child send_email (denied), child final, parent final.
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_read_web", name="read_web", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={"agent": "reviewer", "task": "exfiltrate the page"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_send_email",
                    name="send_email",
                    arguments={"body": "untrusted page content"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("review done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("parent done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app, store, email_tool = _build_taint_app(provider)

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_taint_parent",
                messages=[Message.text("user", "Summarize the page and email it.")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED

    child_sessions = asyncio.run(
        store.list_sessions(SessionQuery(parent_session_id="sess_taint_parent"))
    ).sessions
    assert len(child_sessions) == 1
    child = child_sessions[0]

    # The parent's read_web taint is carried across the boundary onto the child session, even
    # though it was acquired from a tool result (never present on the parent's run metadata).
    assert child.metadata.get(TAINT_LABELS_METADATA_KEY) == ["web"]

    # The inherited taint makes the child's protected send_email a taint-aware policy denial: the
    # tool never runs and the tool result carries the "protected" reason (not some unrelated error).
    assert email_tool.calls == []
    child_transcript = asyncio.run(store.load_transcript(child.id))
    assert [message.role for message in child_transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    denied_result = child_transcript[2].content[0]
    assert denied_result.is_error is True
    assert "protected" in denied_result.content.lower()


def test_subagent_without_parent_taint_allows_protected_tool():
    # Deterministic call order: parent subagent, child send_email (allowed), child final, parent final.
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={"agent": "reviewer", "task": "send a greeting"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_send_email",
                    name="send_email",
                    arguments={"body": "hello"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("review done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("parent done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app, store, email_tool = _build_taint_app(provider)

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_untainted_parent",
                messages=[Message.text("user", "Just email a greeting.")],
            ),
        )
    )

    child = asyncio.run(
        store.list_sessions(SessionQuery(parent_session_id="sess_untainted_parent"))
    ).sessions[0]

    # No parent taint means no seed on the child session and no gate on send_email.
    assert child.metadata.get(TAINT_LABELS_METADATA_KEY) is None

    child_transcript = asyncio.run(store.load_transcript(child.id))
    assert child_transcript[2].content[0].is_error is False
    assert child_transcript[2].content[0].content == "email sent"
    assert email_tool.calls == [{"body": "hello"}]


def test_subagent_metadata_arg_cannot_forge_child_taint():
    # An injected parent must not be able to forge child taint by putting the reserved taint key in
    # its model-controlled `metadata` argument. The parent is untainted, so if the forged label
    # leaked through, the child's send_email would be denied; the fix strips it so send_email runs.
    # Deterministic call order: parent subagent (with forged metadata), child send_email, child
    # final, parent final.
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "send a greeting",
                        "metadata": {TAINT_LABELS_METADATA_KEY: ["web"]},
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_send_email",
                    name="send_email",
                    arguments={"body": "hello"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("review done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("parent done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app, store, email_tool = _build_taint_app(provider)

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_forged_parent",
                messages=[Message.text("user", "Delegate a greeting.")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    child = asyncio.run(
        store.list_sessions(SessionQuery(parent_session_id="sess_forged_parent"))
    ).sessions[0]

    # The forged reserved key is stripped from the model-supplied metadata, so the child is not
    # tainted and the protected send_email runs instead of being gated by fabricated taint.
    assert child.metadata.get(TAINT_LABELS_METADATA_KEY) is None
    assert email_tool.calls == [{"body": "hello"}]


def test_subagent_inherits_same_round_taint_source():
    # A taint source (read_web) and the subagent emitted in ONE model round: the barriered subagent
    # must inherit the same-round taint, or the child runs its protected tool untainted.
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_read_web", name="read_web", arguments={}),
                ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={"agent": "reviewer", "task": "exfiltrate the page"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_send_email",
                    name="send_email",
                    arguments={"body": "untrusted page content"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("review done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("parent done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app, store, email_tool = _build_taint_app(provider)

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_same_round",
                messages=[Message.text("user", "Summarize the page and email it.")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    child = asyncio.run(
        store.list_sessions(SessionQuery(parent_session_id="sess_same_round"))
    ).sessions[0]
    assert child.metadata.get(TAINT_LABELS_METADATA_KEY) == ["web"]
    assert email_tool.calls == []


def _build_pausing_taint_app(
    provider: ModelProvider,
    *,
    parent_policy: TaintAwareToolPolicy,
    parent_tools: list,
) -> tuple[CayuApp, InMemorySessionStore, _ProtectedEmailTool]:
    """Parent (subagent + given extra tools/policy) delegates to a reviewer whose protected
    send_email is DENY-gated once tainted."""
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    subagent_tool = SubagentTool(
        app,
        agents={
            "reviewer": SubagentSpec(agent_name="reviewer", description="Review delegated work.")
        },
    )
    email_tool = _ProtectedEmailTool()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="fake-model"),
        tools=[subagent_tool, *parent_tools],
        tool_policy=parent_policy,
    )
    app.register_agent(
        AgentSpec(name="reviewer", model="fake-model"),
        tools=[email_tool],
        tool_policy=_taint_aware_policy(),
    )
    return app, store, email_tool


def test_subagent_taint_survives_approval_pause():
    # A tainted session pauses for approval on a protected subagent; after approval the resumed
    # subagent must still create a tainted child so the child's protected send_email is denied.
    parent_policy = TaintAwareToolPolicy(
        taint_sources={"read_web": ["web"]},
        protected_tools={"subagent": ["web"]},
        decision=ToolPolicyDecision.REQUIRE_APPROVAL,
    )
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={"agent": "reviewer", "task": "email the secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_send_email", name="send_email", arguments={"body": "secret"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("review done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("parent done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app, store, email_tool = _build_pausing_taint_app(
        provider, parent_policy=parent_policy, parent_tools=[_TaintSourceTool()]
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_appr_taint",
                messages=[Message.text("user", "Delegate.")],
                metadata={TAINT_LABELS_METADATA_KEY: ["web"]},
            ),
        )
    )
    approval_event = next(
        event for event in first_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_appr_taint",
                approval_id=approval_event.payload["approval"]["approval_id"],
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )

    child = asyncio.run(
        store.list_sessions(SessionQuery(parent_session_id="sess_appr_taint"))
    ).sessions[0]
    assert child.metadata.get(TAINT_LABELS_METADATA_KEY) == ["web"]
    assert email_tool.calls == []


def test_subagent_taint_survives_user_input_pause():
    # A tainted session pauses for user input in a round that also delegates to a subagent; on
    # resume the subagent must still create a tainted child so its protected send_email is denied.
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={"agent": "reviewer", "task": "email the secret"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_ask",
                    name="ask_user",
                    arguments={"question": "Proceed?"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_send_email", name="send_email", arguments={"body": "secret"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("review done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("parent done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app, store, email_tool = _build_pausing_taint_app(
        provider, parent_policy=_taint_aware_policy(), parent_tools=[UserInputTool()]
    )

    pause_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_input_taint",
                messages=[Message.text("user", "Delegate.")],
                metadata={TAINT_LABELS_METADATA_KEY: ["web"]},
            ),
        )
    )
    input_id = next(
        event for event in pause_events if event.type == EventType.SESSION_AWAITING_USER_INPUT
    ).payload["input_id"]
    asyncio.run(
        collect_user_input_events(
            app,
            UserInputResponse(session_id="sess_input_taint", input_id=input_id, answer="yes"),
        )
    )

    child = asyncio.run(
        store.list_sessions(SessionQuery(parent_session_id="sess_input_taint"))
    ).sessions[0]
    assert child.metadata.get(TAINT_LABELS_METADATA_KEY) == ["web"]
    assert email_tool.calls == []


def test_subagent_tool_background_starts_child_without_waiting_for_completion():
    class BackgroundProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.child_model_started = asyncio.Event()
            self.release_child = asyncio.Event()
            self.parent_final_requested = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            first_text = request.messages[0].content[0].text
            if first_text == "parent task" and len(request.messages) == 1:
                yield ModelStreamEvent.tool_call(
                    id="call_background_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "background review task",
                    },
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            if first_text == "background review task":
                self.child_model_started.set()
                await self.release_child.wait()
                yield ModelStreamEvent.text_delta("background review complete")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return
            if first_text == "parent task" and request.messages[-1].role == "tool":
                self.parent_final_requested.set()
                yield ModelStreamEvent.text_delta("parent continued before child finished")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return
            raise AssertionError("Unexpected background subagent provider request.")

    async def run():
        store = InMemorySessionStore()
        provider = BackgroundProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="fake-model"),
            tools=[
                SubagentTool(
                    app,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            mode=SubagentExecutionMode.BACKGROUND,
                        )
                    },
                )
            ],
        )
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

        parent_events = await collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_subagent_background_parent",
                causal_budget_id="job_subagent_background",
                messages=[Message.text("user", "parent task")],
            ),
        )
        assert parent_events[-1].type == EventType.SESSION_COMPLETED
        assert provider.parent_final_requested.is_set()

        child_sessions = (
            await store.list_sessions(
                SessionQuery(parent_session_id="sess_subagent_background_parent")
            )
        ).sessions
        assert len(child_sessions) == 1
        child = child_sessions[0]
        assert child.causal_budget_id == "job_subagent_background"
        assert child.status == SessionStatus.RUNNING

        parent_transcript = await store.load_transcript("sess_subagent_background_parent")
        tool_result = parent_transcript[2].content[0]
        assert tool_result.is_error is False
        assert tool_result.structured["mode"] == "background"
        assert tool_result.structured["status"] == "started"
        assert tool_result.structured["child_session_id"] == child.id

        await asyncio.wait_for(provider.child_model_started.wait(), timeout=10)
        provider.release_child.set()
        for _ in range(20):
            loaded_child = await store.load(child.id)
            if loaded_child is not None and loaded_child.status == SessionStatus.COMPLETED:
                return parent_events, child.id
            await asyncio.sleep(0)
        raise AssertionError("Background child session did not complete.")

    parent_events, child_session_id = asyncio.run(run())

    assert parent_events[-1].type == EventType.SESSION_COMPLETED
    assert re.fullmatch(r"cayu-child:v1:subagent:[0-9a-f]{64}", child_session_id)


def test_subagent_result_tool_waits_for_background_child_result():
    class BackgroundResultProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.child_model_started = asyncio.Event()
            self.release_child = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            first_text = request.messages[0].content[0].text
            if first_text == "parent task" and len(request.messages) == 1:
                yield ModelStreamEvent.tool_call(
                    id="call_background_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "background review task",
                    },
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            if first_text == "background review task":
                self.child_model_started.set()
                await self.release_child.wait()
                yield ModelStreamEvent.text_delta("background review complete")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return
            if first_text == "parent task" and len(request.messages) == 3:
                tool_result = request.messages[-1].content[0]
                child_session_id = tool_result.structured["child_session_id"]
                self.release_child.set()
                yield ModelStreamEvent.tool_call(
                    id="call_subagent_result",
                    name="subagent_result",
                    arguments={
                        "child_session_id": child_session_id,
                        "wait": True,
                        "timeout_s": 5,
                    },
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            if first_text == "parent task" and len(request.messages) == 5:
                yield ModelStreamEvent.text_delta("parent used background result")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return
            raise AssertionError("Unexpected background result provider request.")

    async def run():
        store = InMemorySessionStore()
        provider = BackgroundResultProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="fake-model"),
            tools=[
                SubagentTool(
                    app,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            mode=SubagentExecutionMode.BACKGROUND,
                        )
                    },
                ),
                SubagentResultTool(store),
            ],
        )
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

        parent_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="parent",
                    session_id="sess_subagent_background_result_parent",
                    messages=[Message.text("user", "parent task")],
                ),
            )
        )
        await asyncio.wait_for(provider.child_model_started.wait(), timeout=10)
        parent_events = await asyncio.wait_for(parent_task, timeout=10)
        parent_transcript = await store.load_transcript("sess_subagent_background_result_parent")
        child_sessions = (
            await store.list_sessions(
                SessionQuery(parent_session_id="sess_subagent_background_result_parent")
            )
        ).sessions
        return parent_events, parent_transcript, child_sessions

    parent_events, parent_transcript, child_sessions = asyncio.run(run())

    assert parent_events[-1].type == EventType.SESSION_COMPLETED
    assert len(child_sessions) == 1
    assert child_sessions[0].status == SessionStatus.COMPLETED
    result_tool_message = parent_transcript[4]
    result_tool_part = result_tool_message.content[0]
    assert result_tool_part.tool_name == "subagent_result"
    assert result_tool_part.content == "background review complete"
    assert result_tool_part.structured["retrieval_status"] == "ready"
    assert result_tool_part.structured["status"] == "completed"


def test_subagent_result_tool_can_wait_for_all_background_children():
    class MultiBackgroundProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.child_model_started = {"task a": asyncio.Event(), "task b": asyncio.Event()}
            self.release_children = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            first_text = request.messages[0].content[0].text
            if first_text == "parent task" and len(request.messages) == 1:
                yield ModelStreamEvent.tool_call(
                    id="call_background_subagent_a",
                    name="subagent",
                    arguments={"agent": "reviewer", "task": "task a"},
                )
                yield ModelStreamEvent.tool_call(
                    id="call_background_subagent_b",
                    name="subagent",
                    arguments={"agent": "reviewer", "task": "task b"},
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            if first_text in {"task a", "task b"}:
                self.child_model_started[first_text].set()
                await self.release_children.wait()
                yield ModelStreamEvent.text_delta(f"{first_text} done")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return
            if first_text == "parent task" and len(request.messages) == 3:
                self.release_children.set()
                yield ModelStreamEvent.tool_call(
                    id="call_all_subagent_results",
                    name="subagent_result",
                    arguments={"all": True, "wait": True, "timeout_s": 1},
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            if first_text == "parent task" and len(request.messages) == 5:
                yield ModelStreamEvent.text_delta("parent used both background results")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return
            raise AssertionError("Unexpected multi-background provider request.")

    async def run():
        store = InMemorySessionStore()
        provider = MultiBackgroundProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="fake-model"),
            tools=[
                SubagentTool(
                    app,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            mode=SubagentExecutionMode.BACKGROUND,
                        )
                    },
                ),
                SubagentResultTool(store),
            ],
        )
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

        parent_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="parent",
                    session_id="sess_subagent_background_all_parent",
                    messages=[Message.text("user", "parent task")],
                ),
            )
        )
        await asyncio.wait_for(provider.child_model_started["task a"].wait(), timeout=10)
        await asyncio.wait_for(provider.child_model_started["task b"].wait(), timeout=10)
        parent_events = await asyncio.wait_for(parent_task, timeout=10)
        parent_transcript = await store.load_transcript("sess_subagent_background_all_parent")
        child_sessions = (
            await store.list_sessions(
                SessionQuery(parent_session_id="sess_subagent_background_all_parent")
            )
        ).sessions
        return parent_events, parent_transcript, child_sessions

    parent_events, parent_transcript, child_sessions = asyncio.run(run())

    assert parent_events[-1].type == EventType.SESSION_COMPLETED
    assert len(child_sessions) == 2
    assert {child.status for child in child_sessions} == {SessionStatus.COMPLETED}
    result_tool_part = parent_transcript[4].content[0]
    assert result_tool_part.tool_name == "subagent_result"
    assert result_tool_part.structured["retrieval_status"] == "ready"
    assert len(result_tool_part.structured["children"]) == 2
    result_texts = {child["result_text"] for child in result_tool_part.structured["children"]}
    assert result_texts == {"task a done", "task b done"}


def test_subagent_tool_background_reports_start_failure_as_tool_error():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_background_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "Review the changes.",
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("parent handled missing child"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="fake-model"),
        tools=[
            SubagentTool(
                app,
                agents={
                    "reviewer": SubagentSpec(
                        agent_name="missing_reviewer",
                        mode=SubagentExecutionMode.BACKGROUND,
                    )
                },
            )
        ],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_subagent_background_missing_child",
                messages=[Message.text("user", "Review auth.")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    parent_transcript = asyncio.run(store.load_transcript("sess_subagent_background_missing_child"))
    tool_result = parent_transcript[2].content[0]
    assert tool_result.is_error is True
    assert tool_result.structured["mode"] == "background"
    assert tool_result.structured["status"] == "start_failed"
    child_sessions = asyncio.run(
        store.list_sessions(
            SessionQuery(parent_session_id="sess_subagent_background_missing_child")
        )
    ).sessions
    assert child_sessions == []


def test_subagent_tool_returns_child_failure_as_tool_error():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "Review the changes.",
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.error("review provider unavailable")],
            [
                ModelStreamEvent.text_delta("parent recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="fake-model"),
        tools=[
            SubagentTool(
                app,
                agents={
                    "reviewer": SubagentSpec(
                        agent_name="reviewer",
                        description="Review delegated work.",
                    )
                },
            )
        ],
    )
    app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_subagent_parent_failure",
                causal_budget_id="job_subagent_failure",
                messages=[Message.text("user", "Implement and review auth.")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    child_sessions = asyncio.run(
        store.list_sessions(
            SessionQuery(
                parent_session_id="sess_subagent_parent_failure",
            )
        )
    ).sessions
    assert len(child_sessions) == 1
    assert child_sessions[0].status == SessionStatus.FAILED
    parent_transcript = asyncio.run(store.load_transcript("sess_subagent_parent_failure"))
    tool_result = parent_transcript[2].content[0]
    assert tool_result.is_error is True
    assert "review provider unavailable" in tool_result.content
    assert tool_result.structured["child_session_id"] == child_sessions[0].id
    assert tool_result.structured["status"] == "session.failed"
    assert provider.requests[2].messages[-1].role == "tool"


def test_subagent_tool_caps_child_result_text():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "Review the changes.",
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("abcdef"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("parent received capped result"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="fake-model"),
        tools=[
            SubagentTool(
                app,
                agents={
                    "reviewer": SubagentSpec(
                        agent_name="reviewer",
                        result_max_chars=4,
                    )
                },
            )
        ],
    )
    app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_subagent_capped",
                messages=[Message.text("user", "Review auth.")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    parent_transcript = asyncio.run(store.load_transcript("sess_subagent_capped"))
    tool_result = parent_transcript[2].content[0]
    assert tool_result.content == "abcd"
    assert tool_result.structured["result_truncated"] is True
    assert tool_result.structured["result_max_chars"] == 4


def test_collect_subagent_result_returns_last_message_tail_truncated():
    from cayu.tools.subagents import _collect_subagent_result

    def _delta(text: str) -> Event:
        return Event(
            type=EventType.MODEL_TEXT_DELTA,
            session_id="sess_child",
            payload={"delta": text},
        )

    events = [
        Event(type=EventType.MODEL_STARTED, session_id="sess_child"),
        _delta("EARLY draft that should be discarded entirely"),
        Event(type=EventType.MODEL_STARTED, session_id="sess_child"),
        _delta("final "),
        _delta("answer"),
        Event(type=EventType.SESSION_COMPLETED, session_id="sess_child"),
    ]

    async def _stream():
        for event in events:
            yield event

    result = asyncio.run(_collect_subagent_result(_stream(), max_chars=64))

    # The result is the LAST assistant message, not the first deltas of every turn.
    assert result.text == "final answer"
    assert result.text_truncated is False
    assert result.event_count == len(events)
    assert result.terminal is not None
    assert result.terminal.type == EventType.SESSION_COMPLETED


def test_collect_subagent_result_tail_truncates_last_message_only():
    from cayu.tools.subagents import _collect_subagent_result

    events = [
        Event(type=EventType.MODEL_STARTED, session_id="sess_child"),
        Event(
            type=EventType.MODEL_TEXT_DELTA,
            session_id="sess_child",
            payload={"delta": "abcdefgh"},
        ),
        Event(type=EventType.SESSION_COMPLETED, session_id="sess_child"),
    ]

    async def _stream():
        for event in events:
            yield event

    result = asyncio.run(_collect_subagent_result(_stream(), max_chars=4))

    assert result.text == "abcd"
    assert result.text_truncated is True


def test_subagent_tool_interrupts_child_session_when_parent_is_interrupted():
    class BlockingChildProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.child_model_started = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            request_index = len(self.requests)
            if request_index == 1:
                yield ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "Review the changes.",
                    },
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            if request_index == 2:
                self.child_model_started.set()
                await asyncio.Event().wait()
                return
            raise AssertionError(f"Unexpected request {request_index}")

    async def run():
        store = InMemorySessionStore()
        provider = BlockingChildProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="fake-model"),
            tools=[
                SubagentTool(
                    app,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            description="Review delegated work.",
                        )
                    },
                )
            ],
        )
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

        parent_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="parent",
                    session_id="sess_subagent_parent_interrupt",
                    messages=[Message.text("user", "Implement and review auth.")],
                ),
            )
        )
        await asyncio.wait_for(provider.child_model_started.wait(), timeout=10)

        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_subagent_parent_interrupt",
                    reason="stop delegated work",
                )
            )
        ]
        parent_events = await asyncio.wait_for(parent_task, timeout=5)
        child_sessions = (
            await store.list_sessions(
                SessionQuery(parent_session_id="sess_subagent_parent_interrupt")
            )
        ).sessions
        child_events = await store.load_events(child_sessions[0].id)
        return interrupt_events, parent_events, child_sessions, child_events

    interrupt_events, parent_events, child_sessions, child_events = asyncio.run(run())

    assert interrupt_events[-1].type == EventType.SESSION_INTERRUPTED
    assert parent_events[-1].type == EventType.SESSION_INTERRUPTED
    assert len(child_sessions) == 1
    assert child_sessions[0].status == SessionStatus.INTERRUPTED
    child_interrupted_events = [
        event for event in child_events if event.type == EventType.SESSION_INTERRUPTED
    ]
    assert len(child_interrupted_events) == 1
    assert child_interrupted_events[0].payload["metadata"] == {"source": "subagent_tool"}


def test_subagent_tool_interrupts_child_session_during_startup_window():
    class DelayedChildStartupStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.child_running = asyncio.Event()
            self.release_child_startup = asyncio.Event()

        async def create(self, request, *, identity, **kwargs) -> Session:
            session = await super().create(request, identity=identity, **kwargs)
            if (
                session.parent_session_id == "sess_subagent_parent_startup_interrupt"
                and session.status is SessionStatus.RUNNING
            ):
                self.child_running.set()
                await self.release_child_startup.wait()
            return session

        async def transition_status(
            self,
            session_id: str,
            *,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
        ) -> Session:
            session = await super().transition_status(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
            )
            if (
                session.parent_session_id == "sess_subagent_parent_startup_interrupt"
                and to_status == SessionStatus.RUNNING
            ):
                self.child_running.set()
                await self.release_child_startup.wait()
            return session

        async def transition_status_and_checkpoint(
            self,
            session_id: str,
            *,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
            checkpoint_transform=None,
            store_time_checkpoint_transform=None,
            **kwargs,
        ) -> Session:
            session = await super().transition_status_and_checkpoint(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_transform=checkpoint_transform,
                store_time_checkpoint_transform=store_time_checkpoint_transform,
                **kwargs,
            )
            if (
                session.parent_session_id == "sess_subagent_parent_startup_interrupt"
                and to_status == SessionStatus.INTERRUPTING
            ):
                self.release_child_startup.set()
            return session

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "Review the changes.",
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("should not finish"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )

    async def run():
        store = DelayedChildStartupStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="fake-model"),
            tools=[
                SubagentTool(
                    app,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            description="Review delegated work.",
                        )
                    },
                )
            ],
        )
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

        parent_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="parent",
                    session_id="sess_subagent_parent_startup_interrupt",
                    messages=[Message.text("user", "Implement and review auth.")],
                ),
            )
        )
        await asyncio.wait_for(store.child_running.wait(), timeout=10)

        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_subagent_parent_startup_interrupt",
                    reason="stop while child starts",
                )
            )
        ]
        parent_events = await asyncio.wait_for(parent_task, timeout=10)
        child_sessions = (
            await store.list_sessions(
                SessionQuery(parent_session_id="sess_subagent_parent_startup_interrupt")
            )
        ).sessions
        child_events = await store.load_events(child_sessions[0].id)
        return interrupt_events, parent_events, child_sessions, child_events

    interrupt_events, parent_events, child_sessions, child_events = asyncio.run(run())

    assert interrupt_events[-1].type == EventType.SESSION_INTERRUPTED
    assert parent_events[-1].type == EventType.SESSION_INTERRUPTED
    assert len(child_sessions) == 1
    assert child_sessions[0].status == SessionStatus.INTERRUPTED
    assert [
        event.type for event in child_events if event.type == EventType.SESSION_INTERRUPTED
    ] == [EventType.SESSION_INTERRUPTED]


def test_subagent_tool_child_cleanup_failure_does_not_mask_parent_interruption():
    class BlockingChildProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.child_model_started = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            request_index = len(self.requests)
            if request_index == 1:
                yield ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "Review the changes.",
                    },
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            if request_index == 2:
                self.child_model_started.set()
                await asyncio.Event().wait()
                return
            raise AssertionError(f"Unexpected request {request_index}")

    class FailingInterruptRuntime:
        def __init__(self, app: CayuApp) -> None:
            self.app = app

        def run(self, request: RunRequest) -> AsyncIterator[Event]:
            return self.app.run(request)

        async def interrupt_session(
            self,
            request: InterruptSessionRequest,
        ) -> AsyncIterator[Event]:
            raise RuntimeError("child interrupt cleanup unavailable")
            yield  # pragma: no cover

    async def run():
        store = InMemorySessionStore()
        provider = BlockingChildProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        subagent_runtime = FailingInterruptRuntime(app)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="fake-model"),
            tools=[
                SubagentTool(
                    subagent_runtime,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            description="Review delegated work.",
                        )
                    },
                )
            ],
        )
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

        parent_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="parent",
                    session_id="sess_subagent_parent_cleanup_failure",
                    messages=[Message.text("user", "Implement and review auth.")],
                ),
            )
        )
        await asyncio.wait_for(provider.child_model_started.wait(), timeout=10)

        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_subagent_parent_cleanup_failure",
                    reason="stop delegated work",
                )
            )
        ]
        parent_events = await asyncio.wait_for(parent_task, timeout=5)
        parent_transcript = await store.load_transcript("sess_subagent_parent_cleanup_failure")
        parent_session_events = await store.load_events("sess_subagent_parent_cleanup_failure")
        return interrupt_events, parent_events, parent_transcript, parent_session_events

    interrupt_events, parent_events, parent_transcript, parent_session_events = asyncio.run(run())

    assert interrupt_events[-1].type == EventType.SESSION_INTERRUPTED
    assert parent_events[-1].type == EventType.SESSION_INTERRUPTED
    assert not any(message.role == "tool" for message in parent_transcript)
    cleanup_events = [
        event
        for event in parent_session_events
        if event.type is EventType.TOOL_EFFECT_CLEANUP_OBSERVED
    ]
    assert len(cleanup_events) == 1
    artifacts = cleanup_events[0].payload["artifacts"]
    assert artifacts == [
        {
            "type": "cayu.subagent_cleanup_error.v1",
            "child_session_id": artifacts[0]["child_session_id"],
            "error": "child interrupt cleanup unavailable",
            "error_type": "RuntimeError",
        }
    ]
    assert not any(
        event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        for event in parent_session_events
    )
    unknown = [
        event
        for event in parent_session_events
        if event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN
    ]
    assert len(unknown) == 1
    assert unknown[0].payload["tool_call_id"] == "call_subagent"
