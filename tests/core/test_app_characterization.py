"""Ordered event-sequence characterization goldens for ``CayuApp``.

These tests pin the *complete, ordered* list of runtime event types emitted by
``CayuApp`` for a handful of core flows (a full multi-tool-round run, the two
durable pause/resume flows, and the run-limit / budget interruption paths). They
exist to protect the CayuApp decomposition (``src/cayu/runtime/app.py`` ->
single-concern collaborators): the whole point of asserting the *entire* ordered
sequence — rather than windows, membership, or ``any()`` — is that any future
extraction that reorders, drops, or inserts an emission makes these fail loudly.

If one of these sequences changes, that is a *behavior change*, not a test to
update. Treat a diff here as a runtime-contract regression until proven
otherwise (see ``docs/runtime-contracts.md`` and the decomposition plan's
non-negotiable invariants), and only re-baseline after confirming the new order
is intended.

Black-box by construction: only the public ``cayu`` API plus a test-local
scripted provider are used. Nothing is imported from ``cayu.runtime.app``. The
expected sequences were derived from observed current behavior and cross-checked
against the documented ordering guarantees (``context.counted`` before
``model.started``; ``turn.completed`` before every terminal session event).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal

from cayu.core import AgentSpec, Event, EventType, Message
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import (
    InputTokenCountConfidence,
    InputTokenCountMethod,
    InputTokenCountResult,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
)
from cayu.runtime import (
    BudgetLimit,
    BudgetWindow,
    CayuApp,
    ContextCountingConfig,
    ContextCountingMode,
    InMemorySessionStore,
    ModelPrice,
    PriceBook,
    RunLimits,
    RunRequest,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
    UserInputResponse,
)
from cayu.tools.user_input import UserInputTool


class ScriptedProvider(ModelProvider):
    """Deterministic provider that replays one scripted event batch per request.

    Modeled on ``tests/core/test_runtime.py``'s ``FakeProvider`` but defined
    locally so these goldens never depend on another test module. When
    ``count_result`` is set it also answers ``count_input_tokens`` so the
    observe-mode context-counting path can be exercised.
    """

    name = "fake"

    def __init__(
        self,
        batches: list[list[ModelStreamEvent]],
        *,
        count_result: InputTokenCountResult | None = None,
    ) -> None:
        self._batches = batches
        self._count_result = count_result
        self.requests: list[ModelRequest] = []
        self.count_requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        index = len(self.requests) - 1
        if index >= len(self._batches):
            raise AssertionError(f"No scripted provider batch for request {index}")
        for event in self._batches[index]:
            yield event

    async def count_input_tokens(self, request: ModelRequest) -> InputTokenCountResult | None:
        self.count_requests.append(request)
        return self._count_result


class RecordingTool(Tool):
    """Minimal side-effecting tool that records each call's arguments."""

    spec = ToolSpec(
        name="side_effect",
        description="Record execution.",
        input_schema={"type": "object", "properties": {}},
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(args)
        return ToolResult(content="recorded")


class RequireApprovalPolicy(ToolPolicy):
    """Force every tool call through the durable approval pause."""

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
            reason=f"Approval required for {request.tool_name}.",
        )


def _completed(
    usage: dict | None = None,
    *,
    finish_reason: str = "stop",
) -> ModelStreamEvent:
    payload: dict = {"finish_reason": finish_reason, "model": "fake-model"}
    if usage is not None:
        payload["usage"] = usage
    return ModelStreamEvent.completed(payload)


async def _collect(events: AsyncIterator[Event]) -> list[Event]:
    return [event async for event in events]


def _types(events: list[Event]) -> list[EventType]:
    return [event.type for event in events]


def test_g1_full_multi_tool_round_run() -> None:
    """G1: a full run with one real tool round, observe-mode context counting on.

    Context counting is enabled here on purpose: it makes the ``context.*``
    emissions concrete so the golden pins their order relative to ``model.*``
    (invariant: ``context.counted`` is emitted before ``model.started``, and the
    reconciliation events follow ``model.completed``).
    """
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
                _completed(
                    {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
                    finish_reason="tool_calls",
                ),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                _completed({"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}),
            ],
        ],
        count_result=InputTokenCountResult(
            input_tokens=12,
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
        ),
    )
    app = CayuApp(
        session_store=InMemorySessionStore(),
        context_counting=ContextCountingConfig(mode=ContextCountingMode.OBSERVE),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[RecordingTool()])

    events = asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="g1_full_run",
                    messages=[Message.text("user", "use the tool")],
                )
            )
        )
    )

    assert _types(events) == [
        EventType.SESSION_STARTED,
        EventType.CONTEXT_PRESSURE_ESTIMATED,
        EventType.CONTEXT_COUNTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_PRESSURE_RECONCILED,
        EventType.CONTEXT_COUNT_RECONCILED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.CONTEXT_PRESSURE_ESTIMATED,
        EventType.CONTEXT_COUNTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_PRESSURE_RECONCILED,
        EventType.CONTEXT_COUNT_RECONCILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]


def test_g2a_user_input_pause_then_resume() -> None:
    """G2a: ask_user pause stream, then the resume stream through completion."""
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="ask_user",
                    arguments={"question": "Which env?", "options": ["dev", "prod"]},
                ),
                _completed(finish_reason="tool_calls"),
            ],
            [
                ModelStreamEvent.text_delta("Deploying to prod."),
                _completed(),
            ],
        ]
    )
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[UserInputTool()])

    pause_events = asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="g2a_user_input",
                    messages=[Message.text("user", "go")],
                )
            )
        )
    )

    assert _types(pause_events) == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.SESSION_AWAITING_USER_INPUT,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    interrupted = pause_events[-1]
    assert interrupted.payload["interruption_type"] == "user_input_required"

    input_id = next(
        event for event in pause_events if event.type == EventType.SESSION_AWAITING_USER_INPUT
    ).payload["input_id"]

    resume_events = asyncio.run(
        _collect(
            app.resolve_user_input(
                UserInputResponse(
                    session_id="g2a_user_input",
                    input_id=input_id,
                    answer="prod",
                )
            )
        )
    )

    assert _types(resume_events) == [
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]


def test_g2b_tool_approval_pause_then_resume() -> None:
    """G2b: approval pause stream, then the approved resume stream through completion."""
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
                _completed(finish_reason="tool_calls"),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                _completed(),
            ],
        ]
    )
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[RecordingTool()],
        tool_policy=RequireApprovalPolicy(),
    )

    pause_events = asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="g2b_tool_approval",
                    messages=[Message.text("user", "use the tool")],
                )
            )
        )
    )

    assert _types(pause_events) == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.TOOL_CALL_APPROVAL_REQUESTED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    interrupted = pause_events[-1]
    assert interrupted.payload["interruption_type"] == "tool_approval_required"

    approval_id = next(
        event for event in pause_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    ).payload["approval"]["approval_id"]

    resume_events = asyncio.run(
        _collect(
            app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="g2b_tool_approval",
                    approval_id=approval_id,
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        )
    )

    assert _types(resume_events) == [
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]


def test_g3a_run_limit_trip() -> None:
    """G3a: a run-level ``max_tool_calls`` limit that trips on the second round.

    The first tool round runs to completion; the model then requests a second
    tool call that would exceed ``max_tool_calls=1``. The limit is reached, the
    queued call is failed (never executed), the turn is summarized, and the
    session interrupts. ``turn.completed`` precedes the terminal
    ``session.interrupted`` per contract.
    """
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={"step": 1}),
                _completed({"input_tokens": 1, "output_tokens": 1}, finish_reason="tool_calls"),
            ],
            [
                ModelStreamEvent.tool_call(id="call_2", name="side_effect", arguments={"step": 2}),
                _completed({"input_tokens": 1, "output_tokens": 1}, finish_reason="tool_calls"),
            ],
        ]
    )
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    app.register_provider(provider, default=True)
    tool = RecordingTool()
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    events = asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="g3a_run_limit",
                    messages=[Message.text("user", "do it")],
                    limits=RunLimits(max_tool_calls=1),
                )
            )
        )
    )

    assert _types(events) == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TOOL_CALL_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[-4].payload["limit"] == "tool_calls"
    assert events[-1].payload["interruption_type"] == "limit_reached"
    # The over-limit second call is never executed.
    assert tool.calls == [{"step": 1}]


def test_g3b_budget_trip() -> None:
    """G3b: a run-level ``BudgetLimit`` priced to be exceeded by the first step.

    The first model step reports usage whose priced cost exceeds the budget, so
    the queued tool call is failed before any side effect, the turn is
    summarized, and the session interrupts.
    """
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
                _completed(
                    {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100},
                    finish_reason="tool_calls",
                ),
            ]
        ]
    )
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    app.register_provider(provider, default=True)
    tool = RecordingTool()
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    budget = BudgetLimit(
        max_estimated_cost=Decimal("0.002"),
        window=BudgetWindow.all_time(),
        pricing=PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="fake",
                    model="fake-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("10"),
                ),
            )
        ),
        scope="session",
    )

    events = asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="g3b_budget",
                    messages=[Message.text("user", "do it")],
                    budget_limits=(budget,),
                )
            )
        )
    )

    assert _types(events) == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TOOL_CALL_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[3].payload["limit"] == "estimated_cost"
    assert events[-1].payload["interruption_type"] == "limit_reached"
    # The tool call is failed before it can run.
    assert tool.calls == []
