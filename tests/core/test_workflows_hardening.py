"""Regression tests for the adversarial-review hardening of the workflow batteries.

Hardening guarantees the smoke/behavioral suites did not reach:

* The journal anchor session must be parked terminal, including after a
  crash-window left an existing anchor ``PENDING``.
* A workflow journal must not adopt or mutate a foreign session with the same id.
* An interrupted child session is a step failure, not a completed workflow step.
* Child steps carry workflow lineage and share the workflow causal budget.
* Two ``gated_loop`` calls in one workflow must not cross-skip: each is namespaced
  so an item completed in the first loop is not skipped in the second.
* Loop namespaces must not depend on execution order: only one automatic name per
  run, and duplicate loop names (including an explicit ``loop0``) are rejected.
* A newer attempt on the same run id fences out an older in-flight context
  (``WorkflowSupersededError``) instead of both double-running steps.
* A parallel branch that raises ``CancelledError`` on its own fails that branch
  only; it does not cancel healthy siblings.
* Resuming onto an already-started child does not journal a second
  ``workflow.step.started`` for the same step.
* ``step_id`` may not use the reserved ``gated-loop:`` namespace.
* ``emit_custom_event`` rejects the internal ``custom.cayu.`` namespace so user
  events cannot forge or mask attempt-fence markers.
* ``WorkflowSupersededError`` propagates out of ``parallel()`` instead of being
  downgraded to a skippable ``StepFailure``.
* ``app.emit_events`` accepts only public ``workflow.``/``custom.`` events, so
  runtime events and internal ``custom.cayu.`` markers cannot bypass their
  owning paths.
* Terminal/custom workflow events check the attempt fence before journaling.
* Concurrent first-step attempts reserve one durable child run before execution.
* Structured-output typed edges return the validated raw value, not redacted logs.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any

import pytest

import cayu
import cayu.workflows as workflows
from cayu import AgentSpec, CayuApp, EventType, ScriptedModelProvider, WorkflowSpec
from cayu.core.events import Event
from cayu.core.messages import Message
from cayu.core.thinking import ThinkingConfig
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.environments import Environment, EnvironmentSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    BudgetLimit,
    BudgetWindow,
    EventQuery,
    IncompleteSessionsRecoveryRequest,
    InMemoryEventSink,
    ModelPricing,
    PricingCatalog,
    RetryPolicy,
    RunLimits,
    RunRequest,
    SessionIdentity,
    SessionStatus,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.vaults import REDACTED_SECRET, SecretRedactor
from cayu.workflows import (
    WORKFLOW_ATTEMPT_EVENT_TYPE,
    WORKFLOW_JOURNAL_MODEL,
    WORKFLOW_JOURNAL_PROVIDER,
    EventStoreJournal,
    JournalFactory,
    StepError,
    StepFailure,
    StepResult,
    StepRunOptions,
    WorkflowBase,
    WorkflowContext,
    WorkflowJournal,
    WorkflowJournalContext,
    WorkflowSupersededError,
    gated_loop,
    parallel,
    pipeline,
    step,
)


async def _passing_gate(item, result):
    return True


async def _drain(workflow, session_id):
    return [event async for event in workflow.run(session_id)]


def _workflow_event(
    session_id: str,
    step_id: str = "s1",
    *,
    attempt_id: str = "attempt",
) -> Event:
    return Event(
        type=EventType.WORKFLOW_STEP_STARTED,
        session_id=session_id,
        workflow_name="wf",
        payload={"step_id": step_id, "attempt_id": attempt_id},
    )


def _submit(output: dict[str, Any]) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(
            id="call_out",
            name=STRUCTURED_OUTPUT_TOOL_NAME,
            arguments={"output": output},
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def _register_scripted_assistant(
    app: CayuApp,
    batches,
    *,
    provider_name: str = "scripted",
) -> ScriptedModelProvider:
    provider = ScriptedModelProvider(batches, name=provider_name)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
    return provider


def _scripted_assistant_app(
    batches,
    *,
    provider_name: str = "scripted",
) -> tuple[CayuApp, ScriptedModelProvider]:
    app = CayuApp(enable_logging=False)
    provider = _register_scripted_assistant(app, batches, provider_name=provider_name)
    return app, provider


class SideEffectTool(Tool):
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
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
            reason=f"Approval required for {request.tool_name}.",
            metadata={"scope": "human"},
        )


def _budget_limit(max_estimated_cost: str = "1.00") -> BudgetLimit:
    return BudgetLimit(
        max_estimated_cost=Decimal(max_estimated_cost),
        window=BudgetWindow.all_time(),
        pricing=PricingCatalog(
            prices=(
                ModelPricing(
                    provider_name="scripted",
                    model="scripted-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("10"),
                ),
            )
        ),
    )


class RecordingApp(CayuApp):
    def __init__(self):
        super().__init__(enable_logging=False)
        self.run_requests: list[RunRequest] = []

    async def run(self, request: RunRequest):
        self.run_requests.append(request)
        assert request.session_id is not None
        yield Event(type=EventType.MODEL_STARTED, session_id=request.session_id)
        yield Event(
            type=EventType.MODEL_TEXT_DELTA,
            session_id=request.session_id,
            payload={"delta": "done"},
        )
        yield Event(type=EventType.MODEL_COMPLETED, session_id=request.session_id)
        yield Event(type=EventType.SESSION_COMPLETED, session_id=request.session_id)


class BlockingProvider(ModelProvider):
    name = "blocking"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.entered = asyncio.Event()
        self.closed = False

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        self.entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.closed = True
        yield ModelStreamEvent.completed({})


class ControlledProvider(ModelProvider):
    name = "controlled"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        self.entered.set()
        await self.release.wait()
        yield ModelStreamEvent.text_delta("old")
        yield ModelStreamEvent.completed({})


class MemoryJournal:
    def __init__(self):
        self.events: list[Event] = []

    async def append(self, event: Event) -> None:
        self.events.append(event)

    async def append_current_attempt(self, event: Event, *, attempt_id: str) -> bool:
        if self._latest_attempt_id() != attempt_id:
            return False
        self.events.append(event)
        return True

    async def append_step_started(self, event: Event, *, attempt_id: str) -> bool:
        return await self.append_current_attempt(event, attempt_id=attempt_id)

    async def completed_step_ids(self, *, attempt_id: str) -> set[str]:
        latest_attempt, _sequence = self._latest_attempt()
        if latest_attempt != attempt_id:
            return set()
        completed: set[str] = set()
        active_attempt_id: str | None = None
        for event in self.events:
            if event.type == WORKFLOW_ATTEMPT_EVENT_TYPE:
                active_attempt_id = self._event_attempt_id(event)
                continue
            if event.type != EventType.WORKFLOW_STEP_COMPLETED:
                continue
            if self._event_attempt_id(event) == active_attempt_id:
                completed.add(event.payload["step_id"])
        return completed

    async def latest_step_child_session_id(
        self,
        *,
        step_id: str,
        event_type: EventType,
    ) -> str | None:
        latest: str | None = None
        for event in self.events:
            if event.type == event_type and event.payload.get("step_id") == step_id:
                child_session_id = event.payload.get("child_session_id")
                if isinstance(child_session_id, str) and child_session_id:
                    latest = child_session_id
        return latest

    async def step_replay_ids(
        self,
        *,
        step_id: str,
        attempt_id: str,
    ) -> tuple[str | None, str | None]:
        latest_attempt, _sequence = self._latest_attempt()
        if latest_attempt != attempt_id:
            return None, None
        completed: str | None = None
        started: str | None = None
        active_attempt_id: str | None = None
        for event in self.events:
            if event.type == WORKFLOW_ATTEMPT_EVENT_TYPE:
                active_attempt_id = self._event_attempt_id(event)
                continue
            if event.payload.get("step_id") != step_id:
                continue
            if self._event_attempt_id(event) != active_attempt_id:
                continue
            child_session_id = event.payload.get("child_session_id")
            if not (isinstance(child_session_id, str) and child_session_id):
                continue
            if event.type == EventType.WORKFLOW_STEP_COMPLETED:
                completed = child_session_id
            elif event.type == EventType.WORKFLOW_STEP_STARTED:
                started = child_session_id
        return completed, started

    async def latest_attempt_id(self) -> str | None:
        latest, _sequence = self._latest_attempt()
        return latest

    def _latest_attempt_id(self) -> str | None:
        latest, _sequence = self._latest_attempt()
        return latest

    def _latest_attempt(self) -> tuple[str | None, int]:
        latest: str | None = None
        sequence = 0
        for index, event in enumerate(self.events, start=1):
            if event.type == WORKFLOW_ATTEMPT_EVENT_TYPE:
                attempt_id = event.payload.get("attempt_id")
                if isinstance(attempt_id, str) and attempt_id:
                    latest = attempt_id
                    sequence = index
        return latest, sequence

    def _event_attempt_id(self, event: Event) -> str:
        event_attempt = event.payload.get("attempt_id")
        if not isinstance(event_attempt, str) or not event_attempt:
            raise ValueError("Workflow journal events require a non-empty attempt_id payload.")
        return event_attempt


class BlockingCurrentAttemptJournal(MemoryJournal):
    def __init__(self, *, blocked_event_type: EventType | str):
        super().__init__()
        self.blocked_event_type = str(blocked_event_type)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self._blocked = False

    async def append_current_attempt(self, event: Event, *, attempt_id: str) -> bool:
        if str(event.type) == self.blocked_event_type and not self._blocked:
            self._blocked = True
            self.entered.set()
            await self.release.wait()
        return await super().append_current_attempt(event, attempt_id=attempt_id)


class TwoLoopWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="two-loops")

    def __init__(self, app):
        super().__init__(app)
        self.calls: list[str] = []

    async def run(self, session_id):
        ctx = self.context(session_id)
        yield await ctx.start()

        async def do(item):
            self.calls.append(item)
            return StepResult(step_id=f"do-{item}", session_id=f"{session_id}:do-{item}")

        # Both loops process the SAME item key. Pre-fix, the second loop skipped
        # "item" because the first journaled "gated-loop:item"; explicit per-loop
        # names keep them independent (and order-independent across resume).
        async for event in gated_loop(
            ctx, ["item"], do=do, gate=_passing_gate, key=str, name="first"
        ):
            yield event
        async for event in gated_loop(
            ctx, ["item"], do=do, gate=_passing_gate, key=str, name="second"
        ):
            yield event

        yield await ctx.completed()


def test_two_gated_loops_do_not_cross_skip():
    app = CayuApp(enable_logging=False)
    workflow = TwoLoopWorkflow(app)

    asyncio.run(_drain(workflow, "wf-two-loops"))

    # Both loops ran their item — no cross-skip.
    assert workflow.calls == ["item", "item"]

    # …recorded under distinct per-loop namespaces.
    async def load_completed() -> set[str]:
        journal = EventStoreJournal(app.session_store, "wf-two-loops", "two-loops")
        attempt_id = await journal.latest_attempt_id()
        assert attempt_id is not None
        return await journal.completed_step_ids(attempt_id=attempt_id)

    journaled = asyncio.run(load_completed())
    assert {"gated-loop:first:item", "gated-loop:second:item"} <= journaled


class TinyWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="tiny")

    async def run(self, session_id):
        ctx = self.context(session_id)
        yield await ctx.start()
        yield await ctx.completed()


def test_workflow_anchor_is_recovery_safe():
    app = CayuApp(enable_logging=False)
    asyncio.run(_drain(TinyWorkflow(app), "wf-anchor"))

    # The journal anchor is parked terminal (COMPLETED), never left PENDING.
    session = asyncio.run(app.session_store.load("wf-anchor"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED

    # The incomplete-session recovery sweep only accepts non-terminal statuses
    # (PENDING/RUNNING/INTERRUPTING), so a COMPLETED anchor is categorically
    # outside what it can even query. Pre-fix the anchor was PENDING and this
    # sweep found it, then raised KeyError on its unregistered agent_name; now it
    # finds nothing and returns cleanly.
    async def sweep():
        return await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(statuses={SessionStatus.PENDING})
        )

    results = asyncio.run(sweep())  # must not raise
    assert isinstance(results, list)
    records = asyncio.run(app.session_store.query_events(EventQuery(session_id="wf-anchor")))
    assert records
    assert all(str(record.event.type).startswith(("workflow.", "custom.")) for record in records)


def test_workflow_anchor_pending_crash_window_is_healed_on_append():
    app = CayuApp(enable_logging=False)
    store = app.session_store

    async def run():
        await store.create(
            RunRequest(
                agent_name="wf",
                session_id="wf-crash-window",
                messages=[],
                metadata={"cayu.workflow": "wf"},
            ),
            identity=SessionIdentity(
                provider_name=WORKFLOW_JOURNAL_PROVIDER,
                model=WORKFLOW_JOURNAL_MODEL,
            ),
        )

        await EventStoreJournal(store, "wf-crash-window", "wf").append(
            _workflow_event("wf-crash-window")
        )
        session = await store.load("wf-crash-window")
        assert session is not None
        assert session.status == SessionStatus.COMPLETED

        return await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(statuses={SessionStatus.PENDING})
        )

    results = asyncio.run(run())  # must not raise
    assert isinstance(results, list)


def test_workflow_journal_refuses_foreign_session_without_mutating_it():
    app = CayuApp(enable_logging=False)
    store = app.session_store

    async def run():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="wf-foreign",
                messages=[Message.text("user", "real run")],
            ),
            identity=SessionIdentity(provider_name="scripted", model="scripted-model"),
        )

        with pytest.raises(ValueError, match="not a workflow journal anchor"):
            await EventStoreJournal(store, "wf-foreign", "wf").append(_workflow_event("wf-foreign"))

        session = await store.load("wf-foreign")
        assert session is not None
        assert session.status == SessionStatus.PENDING
        records = await store.query_events(EventQuery(session_id="wf-foreign"))
        assert all(not str(record.event.type).startswith("workflow.") for record in records)

    asyncio.run(run())


def test_workflow_journal_refuses_mismatched_workflow_anchor():
    app = CayuApp(enable_logging=False)
    store = app.session_store

    async def run():
        await store.create(
            RunRequest(
                agent_name="first-wf",
                session_id="wf-mismatch",
                messages=[],
                metadata={"cayu.workflow": "first-wf"},
            ),
            identity=SessionIdentity(
                provider_name=WORKFLOW_JOURNAL_PROVIDER,
                model=WORKFLOW_JOURNAL_MODEL,
            ),
        )

        with pytest.raises(ValueError, match="different workflow journal"):
            await EventStoreJournal(store, "wf-mismatch", "second-wf").append(
                Event(
                    type=EventType.WORKFLOW_STARTED,
                    session_id="wf-mismatch",
                    workflow_name="second-wf",
                    payload={"attempt_id": "attempt"},
                )
            )

    asyncio.run(run())


def test_workflow_journal_rejects_non_workflow_event_namespace():
    app = CayuApp(enable_logging=False)
    ctx = TinyWorkflow(app).context("wf-event-namespace")

    with pytest.raises(ValueError, match="workflow\\. or custom\\."):
        ctx.event(EventType.SESSION_FAILED)

    async def append_bad_event():
        await EventStoreJournal(app.session_store, "wf-event-namespace", "tiny").append(
            Event(
                type=EventType.SESSION_FAILED,
                session_id="wf-event-namespace",
                workflow_name="tiny",
                payload={"attempt_id": "attempt"},
            )
        )

    with pytest.raises(ValueError, match="workflow\\. or custom\\."):
        asyncio.run(append_bad_event())


def test_custom_journal_factory_receives_runtime_event_emitter():
    sink = InMemoryEventSink()
    app = CayuApp(enable_logging=False, event_sinks=[sink])
    contexts: list[WorkflowJournalContext] = []

    def journal_factory(context: WorkflowJournalContext) -> WorkflowJournal:
        contexts.append(context)
        return EventStoreJournal(
            context.session_store,
            context.session_id,
            context.workflow_name,
            event_emitter=context.emit_events,
        )

    ctx = TinyWorkflow(app, journal_factory=journal_factory).context("wf-custom-emitter")

    asyncio.run(ctx.emit_custom_event("custom.workflow.factory.emitted"))

    assert contexts
    assert contexts[0].session_id == "wf-custom-emitter"
    assert contexts[0].workflow_name == "tiny"
    assert "custom.workflow.factory.emitted" in [event.type for event in sink.events]


def test_custom_journal_runtime_event_emitter_rejects_runtime_namespace():
    app = CayuApp(enable_logging=False)
    contexts: list[WorkflowJournalContext] = []

    def journal_factory(context: WorkflowJournalContext) -> WorkflowJournal:
        contexts.append(context)
        return EventStoreJournal(
            context.session_store,
            context.session_id,
            context.workflow_name,
            event_emitter=context.emit_events,
        )

    TinyWorkflow(app, journal_factory=journal_factory).context("wf-custom-runtime-event")

    async def emit_runtime_event():
        await contexts[0].emit_events(
            [
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id="wf-custom-runtime-event",
                )
            ]
        )

    with pytest.raises(ValueError, match="workflow. or custom."):
        asyncio.run(emit_runtime_event())


def test_custom_journal_runtime_event_emitter_allows_cayu_attempt_marker():
    sink = InMemoryEventSink()
    app = CayuApp(enable_logging=False, event_sinks=[sink])

    def journal_factory(context: WorkflowJournalContext) -> WorkflowJournal:
        return EventStoreJournal(
            context.session_store,
            context.session_id,
            context.workflow_name,
            event_emitter=context.emit_events,
        )

    ctx = TinyWorkflow(app, journal_factory=journal_factory).context("wf-custom-reserved-event")

    asyncio.run(ctx.start())

    assert [event.type for event in sink.events] == [
        WORKFLOW_ATTEMPT_EVENT_TYPE,
        EventType.WORKFLOW_STARTED,
    ]


def test_workflow_journal_completed_steps_are_filtered_by_workflow_name():
    app = CayuApp(enable_logging=False)
    store = app.session_store

    async def run():
        attempt_id = "attempt-filter"
        await store.create(
            RunRequest(
                agent_name="wf",
                session_id="wf-filter",
                messages=[],
                metadata={"cayu.workflow": "wf"},
            ),
            identity=SessionIdentity(
                provider_name=WORKFLOW_JOURNAL_PROVIDER,
                model=WORKFLOW_JOURNAL_MODEL,
            ),
        )
        await store.append_events(
            "wf-filter",
            [
                Event(
                    type=WORKFLOW_ATTEMPT_EVENT_TYPE,
                    session_id="wf-filter",
                    workflow_name="wf",
                    payload={"attempt_id": attempt_id},
                ),
                Event(
                    type=EventType.WORKFLOW_STEP_COMPLETED,
                    session_id="wf-filter",
                    workflow_name="wf",
                    payload={"step_id": "own", "attempt_id": attempt_id},
                ),
                Event(
                    type=EventType.WORKFLOW_STEP_COMPLETED,
                    session_id="wf-filter",
                    workflow_name="other",
                    payload={"step_id": "foreign", "attempt_id": attempt_id},
                ),
            ],
        )
        return await EventStoreJournal(store, "wf-filter", "wf").completed_step_ids(
            attempt_id=attempt_id
        )

    assert asyncio.run(run()) == {"own"}


def test_workflow_journal_completed_steps_pages_past_event_query_limit():
    app = CayuApp(enable_logging=False)
    store = app.session_store

    async def run():
        attempt_id = "attempt-many"
        await store.create(
            RunRequest(
                agent_name="wf",
                session_id="wf-many",
                messages=[],
                metadata={"cayu.workflow": "wf"},
            ),
            identity=SessionIdentity(
                provider_name=WORKFLOW_JOURNAL_PROVIDER,
                model=WORKFLOW_JOURNAL_MODEL,
            ),
        )
        await store.append_events(
            "wf-many",
            [
                Event(
                    type=WORKFLOW_ATTEMPT_EVENT_TYPE,
                    session_id="wf-many",
                    workflow_name="wf",
                    payload={"attempt_id": attempt_id},
                ),
            ]
            + [
                Event(
                    type=EventType.WORKFLOW_STEP_COMPLETED,
                    session_id="wf-many",
                    workflow_name="wf",
                    payload={"step_id": f"s{index}", "attempt_id": attempt_id},
                )
                for index in range(5001)
            ],
        )
        return await EventStoreJournal(store, "wf-many", "wf").completed_step_ids(
            attempt_id=attempt_id
        )

    completed = asyncio.run(run())
    assert len(completed) == 5001
    assert {"s0", "s5000"} <= completed


def test_workflow_journal_latest_child_session_pages_past_event_query_limit():
    app = CayuApp(enable_logging=False)
    store = app.session_store

    async def run():
        attempt_id = "attempt-many-child"
        await store.create(
            RunRequest(
                agent_name="wf",
                session_id="wf-many-child-lookups",
                messages=[],
                metadata={"cayu.workflow": "wf"},
            ),
            identity=SessionIdentity(
                provider_name=WORKFLOW_JOURNAL_PROVIDER,
                model=WORKFLOW_JOURNAL_MODEL,
            ),
        )
        await store.append_events(
            "wf-many-child-lookups",
            [
                Event(
                    type=WORKFLOW_ATTEMPT_EVENT_TYPE,
                    session_id="wf-many-child-lookups",
                    workflow_name="wf",
                    payload={"attempt_id": attempt_id},
                ),
                Event(
                    type=EventType.WORKFLOW_STEP_COMPLETED,
                    session_id="wf-many-child-lookups",
                    workflow_name="wf",
                    payload={
                        "step_id": "target",
                        "child_session_id": "child-old",
                        "attempt_id": attempt_id,
                    },
                ),
            ]
            + [
                Event(
                    type=EventType.WORKFLOW_STEP_COMPLETED,
                    session_id="wf-many-child-lookups",
                    workflow_name="wf",
                    payload={
                        "step_id": f"other-{index}",
                        "child_session_id": f"child-{index}",
                        "attempt_id": attempt_id,
                    },
                )
                for index in range(5000)
            ]
            + [
                Event(
                    type=EventType.WORKFLOW_STEP_COMPLETED,
                    session_id="wf-many-child-lookups",
                    workflow_name="wf",
                    payload={
                        "step_id": "target",
                        "child_session_id": "child-target",
                        "attempt_id": attempt_id,
                    },
                )
            ],
        )
        return await EventStoreJournal(
            store,
            "wf-many-child-lookups",
            "wf",
        ).latest_step_child_session_id(
            step_id="target",
            event_type=EventType.WORKFLOW_STEP_COMPLETED,
        )

    assert asyncio.run(run()) == "child-target"


def test_step_interrupted_child_raises_and_leaves_step_unjournaled():
    app = CayuApp(enable_logging=False)
    tool = SideEffectTool()
    app.register_provider(
        ScriptedModelProvider(
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )
    ctx = TinyWorkflow(app).context("wf-interrupted")

    with pytest.raises(StepError) as excinfo:
        asyncio.run(
            step(
                ctx,
                agent="assistant",
                step_id="s1",
                prompt="use the tool",
                session_id="child-needs-approval",
            )
        )

    assert excinfo.value.step_id == "s1"
    assert excinfo.value.session_id == "child-needs-approval"
    assert "interrupted" in str(excinfo.value)
    assert "tool_approval_required" in str(excinfo.value)
    assert tool.calls == []
    assert "s1" not in asyncio.run(ctx.journal.completed_step_ids(attempt_id=ctx.attempt_id))


def test_step_rerun_while_child_interrupted_reuses_child_without_rerun():
    app = CayuApp(enable_logging=False)
    tool = SideEffectTool()
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_1",
                name="side_effect",
                arguments={"value": "secret"},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )
    workflow = TinyWorkflow(app)

    with pytest.raises(StepError) as first:
        asyncio.run(
            step(
                workflow.context("wf-interrupted-rerun"),
                agent="assistant",
                step_id="s1",
                prompt="use the tool",
            )
        )
    child_session_id = first.value.session_id

    with pytest.raises(StepError) as second:
        asyncio.run(
            step(
                workflow.context("wf-interrupted-rerun"),
                agent="assistant",
                step_id="s1",
                prompt="use the tool again",
            )
        )

    assert second.value.session_id == child_session_id
    assert "interrupted" in str(second.value)
    assert len(provider.requests) == 1
    assert tool.calls == []


def test_step_cancellation_finalizes_started_child_before_replay():
    app = CayuApp(enable_logging=False)
    provider = BlockingProvider()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
    workflow = TinyWorkflow(app)

    async def cancel_running_step():
        ctx = workflow.context("wf-cancel-running")
        task = asyncio.create_task(step(ctx, agent="assistant", step_id="s1", prompt="wait"))
        await asyncio.wait_for(provider.entered.wait(), timeout=1)
        _, child_session_id = await ctx.journal.step_replay_ids(
            step_id="s1",
            attempt_id=ctx.attempt_id,
        )
        assert child_session_id is not None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        child = await app.session_store.load(child_session_id)
        return child_session_id, child

    child_session_id, child = asyncio.run(cancel_running_step())

    assert provider.closed is True
    assert child is not None
    assert child.status != SessionStatus.RUNNING

    with pytest.raises(StepError) as replay:
        asyncio.run(
            step(
                workflow.context("wf-cancel-running"),
                agent="assistant",
                step_id="s1",
                prompt="wait again",
            )
        )
    assert replay.value.session_id == child_session_id


def test_step_reuses_resolved_interrupted_child_on_rerun():
    app = CayuApp(enable_logging=False)
    tool = SideEffectTool()
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="side_effect",
                        arguments={"value": "secret"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.text_delta("approved"), ModelStreamEvent.completed({})],
            ]
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )
    ctx = TinyWorkflow(app).context("wf-resolved-interrupt")

    with pytest.raises(StepError) as excinfo:
        asyncio.run(step(ctx, agent="assistant", step_id="s1", prompt="use the tool"))
    child_session_id = excinfo.value.session_id
    assert child_session_id is not None

    async def approve_child():
        records = await app.session_store.query_events(
            EventQuery(
                session_id=child_session_id,
                event_type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
            )
        )
        approval = records[-1].event.payload["approval"]
        return [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id=child_session_id,
                    approval_id=approval["approval_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]

    approved_events = asyncio.run(approve_child())
    assert approved_events[-1].type == EventType.SESSION_COMPLETED

    resume_ctx = TinyWorkflow(app).context("wf-resolved-interrupt")
    result = asyncio.run(step(resume_ctx, agent="assistant", step_id="s1", prompt="use the tool"))

    assert result.session_id == child_session_id
    assert result.text == "approved"
    assert tool.calls == [{"value": "secret"}]
    assert "s1" in asyncio.run(
        resume_ctx.journal.completed_step_ids(attempt_id=resume_ctx.attempt_id)
    )


def test_step_child_session_records_workflow_lineage():
    app, _provider = _scripted_assistant_app(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    ctx = TinyWorkflow(app).context("wf-lineage")

    result = asyncio.run(
        step(
            ctx,
            agent="assistant",
            step_id="s1",
            prompt="go",
            session_id="child-lineage",
        )
    )

    child = asyncio.run(app.session_store.load(result.session_id))
    assert child is not None
    assert child.parent_session_id == "wf-lineage"
    assert child.causal_budget_id == "wf-lineage"


def test_step_child_session_inherits_anchor_causal_budget_id():
    app, _provider = _scripted_assistant_app(
        [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]
    )

    async def seed_anchor():
        await app.session_store.create(
            RunRequest(
                agent_name="tiny",
                session_id="wf-shared-budget",
                messages=[],
                metadata={"cayu.workflow": "tiny"},
                causal_budget_id="job-workflow",
            ),
            identity=SessionIdentity(
                provider_name=WORKFLOW_JOURNAL_PROVIDER,
                model=WORKFLOW_JOURNAL_MODEL,
            ),
        )

    asyncio.run(seed_anchor())
    ctx = TinyWorkflow(app).context("wf-shared-budget")

    result = asyncio.run(step(ctx, agent="assistant", step_id="s1", prompt="go"))

    child = asyncio.run(app.session_store.load(result.session_id))
    assert child is not None
    assert child.parent_session_id == "wf-shared-budget"
    assert child.causal_budget_id == "job-workflow"


def test_step_no_anchor_custom_journal_keeps_budget_link_without_parent():
    app, _provider = _scripted_assistant_app(
        [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]
    )
    journal = MemoryJournal()

    def journal_factory(context: WorkflowJournalContext) -> WorkflowJournal:
        return journal

    ctx = TinyWorkflow(app, journal_factory=journal_factory).context("wf-memory")
    result = asyncio.run(step(ctx, agent="assistant", step_id="s1", prompt="go"))

    child = asyncio.run(app.session_store.load(result.session_id))
    assert child is not None
    assert child.parent_session_id is None
    assert child.causal_budget_id == "wf-memory"
    assert [event.type for event in journal.events] == [
        WORKFLOW_ATTEMPT_EVENT_TYPE,
        EventType.WORKFLOW_STEP_STARTED,
        EventType.WORKFLOW_STEP_COMPLETED,
    ]


def test_step_custom_journal_ignores_foreign_session_with_workflow_id_for_lineage():
    app, _provider = _scripted_assistant_app(
        [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]
    )
    journal = MemoryJournal()

    async def seed_foreign_session():
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="wf-memory-foreign",
                messages=[Message.text("user", "foreign")],
                causal_budget_id="foreign-budget",
            ),
            identity=SessionIdentity(provider_name="scripted", model="scripted-model"),
        )

    asyncio.run(seed_foreign_session())

    def journal_factory(context: WorkflowJournalContext) -> WorkflowJournal:
        return journal

    ctx = TinyWorkflow(app, journal_factory=journal_factory).context("wf-memory-foreign")
    result = asyncio.run(step(ctx, agent="assistant", step_id="s1", prompt="go"))

    child = asyncio.run(app.session_store.load(result.session_id))
    assert child is not None
    assert child.parent_session_id is None
    assert child.causal_budget_id == "wf-memory-foreign"


def test_step_resume_uses_custom_journal_for_completed_child_lookup():
    app, provider = _scripted_assistant_app(
        [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]
    )
    journal = MemoryJournal()

    def journal_factory(context: WorkflowJournalContext) -> WorkflowJournal:
        return journal

    workflow = TinyWorkflow(app, journal_factory=journal_factory)
    first = asyncio.run(
        step(
            workflow.context("wf-memory-resume"),
            agent="assistant",
            step_id="s1",
            prompt="go",
        )
    )
    second = asyncio.run(
        step(
            workflow.context("wf-memory-resume"),
            agent="assistant",
            step_id="s1",
            prompt="go again",
        )
    )

    assert second.session_id == first.session_id
    assert second.text == "done"
    assert len(provider.requests) == 1


def test_step_rejects_duplicate_step_id_within_one_context():
    app, provider = _scripted_assistant_app(
        [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]
    )
    ctx = TinyWorkflow(app).context("wf-duplicate-step-id")

    asyncio.run(step(ctx, agent="assistant", step_id="s1", prompt="go"))

    with pytest.raises(ValueError, match="Duplicate step_id"):
        asyncio.run(step(ctx, agent="assistant", step_id="s1", prompt="go again"))
    assert len(provider.requests) == 1


def test_step_resume_reuses_completed_default_journal_child_without_rerun():
    app, provider = _scripted_assistant_app(
        [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]
    )
    workflow = TinyWorkflow(app)

    first = asyncio.run(
        step(workflow.context("wf-default-resume"), agent="assistant", step_id="s1", prompt="go")
    )
    second = asyncio.run(
        step(
            workflow.context("wf-default-resume"),
            agent="assistant",
            step_id="s1",
            prompt="go again",
        )
    )

    assert second.session_id == first.session_id
    assert second.text == "done"
    assert len(provider.requests) == 1


def test_parallel_duplicate_step_id_is_programmer_error_not_stepfailure():
    app = CayuApp(enable_logging=False)
    provider = BlockingProvider()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
    ctx = TinyWorkflow(app).context("wf-parallel-duplicate-step")

    async def run_duplicate_steps():
        return await parallel(
            [
                step(ctx, agent="assistant", step_id="same", prompt="first"),
                step(ctx, agent="assistant", step_id="same", prompt="second"),
            ]
        )

    with pytest.raises(ValueError, match="Duplicate step_id"):
        asyncio.run(run_duplicate_steps())

    assert "same" not in asyncio.run(ctx.journal.completed_step_ids(attempt_id=ctx.attempt_id))
    assert provider.requests == []
    assert provider.closed is False


def test_step_replay_recovers_stale_running_child_before_reuse():
    app, provider = _scripted_assistant_app(
        [ModelStreamEvent.text_delta("unexpected"), ModelStreamEvent.completed({})]
    )
    workflow = TinyWorkflow(app)

    async def seed_started_running_child():
        ctx = workflow.context("wf-running-child-replay")
        await ctx.journal.append(ctx.event(WORKFLOW_ATTEMPT_EVENT_TYPE))
        await ctx.journal.append(
            ctx.event(
                EventType.WORKFLOW_STEP_STARTED,
                agent_name="assistant",
                payload={
                    "step_id": "s1",
                    "agent": "assistant",
                    "child_session_id": "child-running-replay",
                },
            )
        )
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="child-running-replay",
                messages=[Message.text("user", "stale")],
            ),
            identity=SessionIdentity(provider_name="scripted", model="scripted-model"),
        )
        await app.session_store.update_status("child-running-replay", SessionStatus.RUNNING)

    asyncio.run(seed_started_running_child())

    with pytest.raises(StepError) as replay:
        asyncio.run(
            step(
                workflow.context("wf-running-child-replay"),
                agent="assistant",
                step_id="s1",
                prompt="retry",
            )
        )

    child = asyncio.run(app.session_store.load("child-running-replay"))
    assert replay.value.session_id == "child-running-replay"
    assert child is not None
    assert child.status == SessionStatus.INTERRUPTED
    assert provider.requests == []


def test_failed_step_is_not_completed_and_resume_uses_fresh_child():
    app, provider = _scripted_assistant_app(
        [
            [ModelStreamEvent.error("kaboom"), ModelStreamEvent.completed({})],
            [ModelStreamEvent.text_delta("recovered"), ModelStreamEvent.completed({})],
        ]
    )
    workflow = TinyWorkflow(app)

    with pytest.raises(StepError) as excinfo:
        asyncio.run(
            step(
                workflow.context("wf-failed-rerun"),
                agent="assistant",
                step_id="s1",
                prompt="go",
            )
        )
    failed_child_id = excinfo.value.session_id
    assert failed_child_id is not None

    async def completed_after_failed_step() -> set[str]:
        journal = EventStoreJournal(app.session_store, "wf-failed-rerun", "tiny")
        attempt_id = await journal.latest_attempt_id()
        assert attempt_id is not None
        return await journal.completed_step_ids(attempt_id=attempt_id)

    assert "s1" not in asyncio.run(completed_after_failed_step())

    result = asyncio.run(
        step(
            workflow.context("wf-failed-rerun"),
            agent="assistant",
            step_id="s1",
            prompt="go again",
        )
    )

    assert result.session_id != failed_child_id
    assert result.text == "recovered"
    assert len(provider.requests) == 2


def test_gated_loop_rejects_duplicate_item_keys_before_resume_can_skip_work():
    app = CayuApp(enable_logging=False)
    ctx = TinyWorkflow(app).context("wf-duplicate-keys")

    async def do(item):
        return StepResult(step_id=f"do-{item}", session_id=f"child-{item}")

    async def run():
        async for _event in gated_loop(
            ctx, ["a", "b"], do=do, gate=_passing_gate, key=lambda _: "x"
        ):
            pass

    with pytest.raises(ValueError, match="Duplicate gated_loop key"):
        asyncio.run(run())


def test_gated_loop_resume_uses_stable_keys_when_items_reorder():
    app = CayuApp(enable_logging=False)
    ctx1 = TinyWorkflow(app).context("wf-reordered")
    calls1: list[str] = []

    async def do1(item):
        calls1.append(item)
        if item == "beta":
            raise RuntimeError("crash")
        return StepResult(step_id=f"do-{item}", session_id=f"child-{item}")

    async def first_run():
        async for _event in gated_loop(
            ctx1,
            ["alpha", "beta", "gamma"],
            do=do1,
            gate=_passing_gate,
            key=str,
        ):
            pass

    with pytest.raises(RuntimeError):
        asyncio.run(first_run())

    ctx2 = TinyWorkflow(app).context("wf-reordered")
    calls2: list[str] = []

    async def do2(item):
        calls2.append(item)
        return StepResult(step_id=f"do-{item}", session_id=f"child-{item}")

    async def second_run():
        async for _event in gated_loop(
            ctx2,
            ["gamma", "beta", "alpha"],
            do=do2,
            gate=_passing_gate,
            key=str,
        ):
            pass

    asyncio.run(second_run())

    assert calls1 == ["alpha", "beta"]
    assert calls2 == ["gamma", "beta"]


def test_step_run_options_defensively_copy_mutable_fields():
    metadata = {"nested": {"value": 1}}
    labels = {"team": "runtime"}
    limits = RunLimits(max_total_tokens=100)
    limit = _budget_limit()

    opts = StepRunOptions(
        metadata=metadata,
        labels=labels,
        limits=limits,
        budget_limits=(limit,),
    )
    metadata["nested"]["value"] = 2
    labels["team"] = "mutated"
    limits.max_total_tokens = 200
    limit.currency = "EUR"

    assert opts.metadata == {"nested": {"value": 1}}
    assert opts.labels == {"team": "runtime"}
    assert opts.limits.max_total_tokens == 100
    assert opts.budget_limits[0].currency == "USD"
    assert opts.budget_limits[0] is not limit


def test_step_forwards_run_options_and_preserves_owned_lineage():
    app = RecordingApp()
    _register_scripted_assistant(
        app,
        [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})],
        provider_name="scripted-alt",
    )
    app.register_environment(Environment(EnvironmentSpec(name="docker")))
    retry_policy = RetryPolicy(max_attempts=2)
    thinking = ThinkingConfig(effort="low")
    limit = _budget_limit()
    ctx = TinyWorkflow(app).context("wf-options")

    result = asyncio.run(
        step(
            ctx,
            agent="assistant",
            step_id="s1",
            prompt="go",
            run_options=StepRunOptions(
                provider_name="scripted-alt",
                environment_name="docker",
                labels={"project": "workflow"},
                metadata={"purpose": "test"},
                max_steps=7,
                limits=RunLimits(max_total_tokens=100),
                budget_limits=(limit,),
                retry_policy=retry_policy,
                thinking=thinking,
                task_id="task-1",
                task_worker_id="worker-1",
            ),
        )
    )

    request = app.run_requests[-1]
    assert request.session_id == result.session_id
    assert request.provider_name == "scripted-alt"
    assert request.environment_name == "docker"
    assert request.labels == {"project": "workflow"}
    assert request.metadata == {"purpose": "test"}
    assert request.max_steps == 7
    assert request.limits.max_total_tokens == 100
    assert request.budget_limits == (limit,)
    assert request.budget_limits[0] is not limit
    assert request.retry_policy is retry_policy
    assert request.thinking is thinking
    assert request.task_id == "task-1"
    assert request.task_worker_id == "worker-1"
    assert request.parent_session_id == "wf-options"
    assert request.causal_budget_id == "wf-options"


def test_workflow_context_emit_custom_event_journals_and_returns_event():
    sink = InMemoryEventSink()
    app = CayuApp(enable_logging=False, event_sinks=[sink])
    ctx = TinyWorkflow(app).context("wf-custom-event")

    event = asyncio.run(
        ctx.emit_custom_event(
            "custom.workflow.gate.completed",
            payload={"gate": "pytest", "passed": True},
            agent_name="assistant",
        )
    )

    assert event.type == "custom.workflow.gate.completed"
    assert event.workflow_name == "tiny"
    assert event.agent_name == "assistant"
    records = asyncio.run(app.session_store.query_events(EventQuery(session_id="wf-custom-event")))
    assert "custom.workflow.gate.completed" in [record.event.type for record in records]
    assert "custom.workflow.gate.completed" in [event.type for event in sink.events]


def test_workflow_context_emit_custom_event_rejects_non_custom_names():
    app = CayuApp(enable_logging=False)
    ctx = TinyWorkflow(app).context("wf-custom-event-reject")

    with pytest.raises(ValueError, match="custom\\."):
        asyncio.run(ctx.emit_custom_event("workflow.not_custom"))


def test_workflow_exports_keep_root_package_focused():
    assert cayu.WorkflowContext is WorkflowContext
    assert cayu.StepRunOptions is StepRunOptions
    assert cayu.StepResult is StepResult
    assert cayu.StepFailure is StepFailure
    assert cayu.WorkflowBase is WorkflowBase
    assert cayu.gated_loop is gated_loop
    assert cayu.parallel is parallel
    assert cayu.pipeline is pipeline
    assert cayu.step is step

    assert workflows.WORKFLOW_JOURNAL_MODEL == WORKFLOW_JOURNAL_MODEL
    assert workflows.WORKFLOW_JOURNAL_PROVIDER == WORKFLOW_JOURNAL_PROVIDER
    assert workflows.WORKFLOW_ATTEMPT_EVENT_TYPE == WORKFLOW_ATTEMPT_EVENT_TYPE
    assert workflows.JournalFactory is JournalFactory
    assert workflows.WorkflowJournal is WorkflowJournal
    assert workflows.WorkflowJournalContext is WorkflowJournalContext
    assert workflows.EventStoreJournal is EventStoreJournal

    assert not hasattr(cayu, "WORKFLOW_JOURNAL_MODEL")
    assert not hasattr(cayu, "WORKFLOW_JOURNAL_PROVIDER")
    assert not hasattr(cayu, "WORKFLOW_ATTEMPT_EVENT_TYPE")
    assert not hasattr(cayu, "JournalFactory")
    assert not hasattr(cayu, "WorkflowJournal")
    assert not hasattr(cayu, "WorkflowJournalContext")
    assert not hasattr(cayu, "EventStoreJournal")


async def _static_do(item):
    return StepResult(step_id=f"do-{item}", session_id=f"session:{item}")


def test_second_auto_named_gated_loop_requires_explicit_name():
    app = CayuApp(enable_logging=False)
    ctx = TinyWorkflow(app).context("wf-second-auto-name")

    async def run():
        async for _ in gated_loop(ctx, ["a"], do=_static_do, gate=_passing_gate, key=str):
            pass
        with pytest.raises(ValueError, match="automatic"):
            async for _ in gated_loop(ctx, ["b"], do=_static_do, gate=_passing_gate, key=str):
                pass

    asyncio.run(run())


def test_gated_loop_rejects_duplicate_names_including_the_auto_namespace():
    app = CayuApp(enable_logging=False)
    ctx = TinyWorkflow(app).context("wf-duplicate-loop-name")

    async def run():
        # An explicit name squatting the automatic namespace collides with a
        # later auto-named loop instead of silently sharing its journal.
        async for _ in gated_loop(
            ctx, ["a"], do=_static_do, gate=_passing_gate, key=str, name="loop0"
        ):
            pass
        with pytest.raises(ValueError, match="Duplicate gated_loop name"):
            async for _ in gated_loop(ctx, ["b"], do=_static_do, gate=_passing_gate, key=str):
                pass

    asyncio.run(run())


def test_newer_attempt_supersedes_older_workflow_context():
    app, provider = _scripted_assistant_app(
        [
            [ModelStreamEvent.text_delta("one"), ModelStreamEvent.completed({})],
            [ModelStreamEvent.text_delta("two"), ModelStreamEvent.completed({})],
        ]
    )
    workflow = TinyWorkflow(app)

    async def run():
        first_attempt = workflow.context("wf-fence")
        await step(first_attempt, agent="assistant", step_id="s1", prompt="go")

        second_attempt = workflow.context("wf-fence")
        await step(second_attempt, agent="assistant", step_id="s2", prompt="go")

        with pytest.raises(WorkflowSupersededError):
            await step(first_attempt, agent="assistant", step_id="s3", prompt="go")

    asyncio.run(run())
    # The fenced-out step never reached the model.
    assert len(provider.requests) == 2


def test_parallel_branch_raising_cancelled_error_fails_only_that_branch():
    async def leaked_cancel():
        raise asyncio.CancelledError()

    async def healthy():
        return StepResult(step_id="healthy", session_id="session:healthy")

    result = asyncio.run(parallel([leaked_cancel(), healthy()]))

    assert [success.step_id for success in result.successes] == ["healthy"]
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert failure.error == "step raised CancelledError without being cancelled"
    assert failure.error_type == "CancelledError"


def test_step_resume_onto_started_child_journals_single_started_event():
    app, _provider = _scripted_assistant_app(
        [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]
    )
    workflow = TinyWorkflow(app)
    session_id = "wf-single-started"
    child_session_id = f"{session_id}:s1:prior001"

    async def seed_prior_started():
        # A prior attempt journaled STARTED for a child that never got created
        # (crash before the child run) — resume must reuse it, not re-journal it.
        journal = EventStoreJournal(app.session_store, session_id, "tiny")
        await journal.append(
            Event(
                type=WORKFLOW_ATTEMPT_EVENT_TYPE,
                session_id=session_id,
                workflow_name="tiny",
                payload={"attempt_id": "prior-attempt"},
            )
        )
        await journal.append(
            Event(
                type=EventType.WORKFLOW_STEP_STARTED,
                session_id=session_id,
                workflow_name="tiny",
                payload={
                    "step_id": "s1",
                    "agent": "assistant",
                    "child_session_id": child_session_id,
                    "attempt_id": "prior-attempt",
                },
            )
        )

    asyncio.run(seed_prior_started())
    result = asyncio.run(
        step(workflow.context(session_id), agent="assistant", step_id="s1", prompt="go")
    )

    assert result.session_id == child_session_id
    records = asyncio.run(
        app.session_store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.WORKFLOW_STEP_STARTED,
                limit=5000,
            )
        )
    )
    started_for_step = [record for record in records if record.event.payload.get("step_id") == "s1"]
    assert len(started_for_step) == 1


def test_step_rejects_reserved_gated_loop_step_id_prefix():
    app = CayuApp(enable_logging=False)
    ctx = TinyWorkflow(app).context("wf-reserved-prefix")

    with pytest.raises(ValueError, match="reserved"):
        step(ctx, agent="assistant", step_id="gated-loop:loop0:item", prompt="go")


def test_emit_custom_event_rejects_reserved_cayu_namespace():
    app = CayuApp(enable_logging=False)
    ctx = TinyWorkflow(app).context("wf-reserved-custom-namespace")

    async def run():
        with pytest.raises(ValueError, match="reserved for cayu internals"):
            await ctx.emit_custom_event(
                "custom.cayu.workflow.attempt", payload={"attempt_id": "forged"}
            )
        event = await ctx.emit_custom_event("custom.myapp.thing", payload={"ok": True})
        assert event.type == "custom.myapp.thing"

    asyncio.run(run())


class _LoopCrashError(RuntimeError):
    pass


def test_sqlite_crash_resume_replays_prefix_without_model_calls(tmp_path):
    db_path = tmp_path / "wf.db"
    run_id = "wf-sqlite-crash-resume"
    hook_log: list[str] = []
    plan_schema = {
        "type": "object",
        "properties": {"notes": {"type": "string"}},
        "required": ["notes"],
        "additionalProperties": False,
    }

    class Maintenance(WorkflowBase):
        spec = WorkflowSpec(name="sqlite-resume")

        def __init__(self, app, *, crash_on: str | None):
            super().__init__(app)
            self.crash_on = crash_on

        async def run(self, session_id):
            ctx = self.context(session_id)
            yield await ctx.start()

            findings = await parallel(
                [
                    step(ctx, agent="assistant", step_id="audit-a", prompt="a"),
                    step(ctx, agent="assistant", step_id="audit-b", prompt="b"),
                ]
            )
            assert findings.ok

            plan = await pipeline(
                findings,
                [
                    lambda prev: step(
                        ctx,
                        agent="assistant",
                        step_id="plan",
                        prompt=f"plan from {len(prev.successes)}",
                        schema=plan_schema,
                    )
                ],
            )
            assert plan.output == {"notes": "the-plan"}

            async def do(item):
                result = await step(ctx, agent="assistant", step_id=f"fix-{item}", prompt=item)
                if item == self.crash_on:
                    raise _LoopCrashError(item)
                return result

            async def gate(item, result):
                return "good" in result.text

            async def on_pass(item, result, outcome):
                hook_log.append(f"commit {item}")

            async def on_fail(item, result, outcome):
                hook_log.append(f"revert {item}")

            async for event in gated_loop(
                ctx,
                ["one", "two", "three"],
                do=do,
                gate=gate,
                on_pass=on_pass,
                on_fail=on_fail,
                key=str,
                name="fixes",
            ):
                yield event

            yield await ctx.completed()

    def build_app(batches):
        app = CayuApp(enable_logging=False, session_store=SQLiteSessionStore(db_path))
        provider = ScriptedModelProvider(batches)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
        return app, provider

    def _text(content):
        return [ModelStreamEvent.text_delta(content), ModelStreamEvent.completed({})]

    plan_submit = [
        ModelStreamEvent.tool_call(
            id="call-plan",
            name=STRUCTURED_OUTPUT_TOOL_NAME,
            arguments={"output": {"notes": "the-plan"}},
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]

    app_a, provider_a = build_app(
        [
            _text("audit"),
            _text("audit"),
            plan_submit,
            _text("good one"),
            _text("bad two"),
            _text("good three"),
        ]
    )
    with pytest.raises(_LoopCrashError):
        asyncio.run(_drain(Maintenance(app_a, crash_on="three"), run_id))
    assert hook_log == ["commit one", "revert two"]
    assert len(provider_a.requests) == 6
    # A crashed process's connection is gone; drop ours before "restarting".
    asyncio.run(app_a.session_store.close())

    # A fresh app instance over the same sqlite file simulates a new process.
    # Zero scripted batches: any model call raises, so a clean completion
    # proves the entire journaled prefix replayed instead of re-running.
    app_b, provider_b = build_app([])
    events = asyncio.run(_drain(Maintenance(app_b, crash_on=None), run_id))
    assert str(events[-1].type) == "workflow.completed"
    assert provider_b.requests == []
    assert hook_log == ["commit one", "revert two", "commit three"]

    records = asyncio.run(
        app_b.session_store.query_events(
            EventQuery(
                session_id=run_id,
                event_type=EventType.WORKFLOW_STEP_STARTED,
                limit=5000,
            )
        )
    )
    step_started = [
        record.event.payload["step_id"]
        for record in records
        if not str(record.event.payload["step_id"]).startswith("gated-loop:")
    ]
    assert sorted(step_started) == [
        "audit-a",
        "audit-b",
        "fix-one",
        "fix-three",
        "fix-two",
        "plan",
    ]

    # The crashed item retried: two STARTED attempts with distinct attempt_ids,
    # and the COMPLETED pairs with the second (resume) attempt.
    item_started = [
        record.event
        for record in records
        if record.event.payload["step_id"] == "gated-loop:fixes:three"
    ]
    assert len(item_started) == 2
    first_attempt, second_attempt = (event.payload["attempt_id"] for event in item_started)
    assert first_attempt != second_attempt
    completed = asyncio.run(
        app_b.session_store.query_events(
            EventQuery(
                session_id=run_id,
                event_type=EventType.WORKFLOW_STEP_COMPLETED,
                limit=5000,
            )
        )
    )
    item_completed = [
        record.event
        for record in completed
        if record.event.payload["step_id"] == "gated-loop:fixes:three"
    ]
    assert [event.payload["attempt_id"] for event in item_completed] == [second_attempt]
    asyncio.run(app_b.session_store.close())


def test_parallel_propagates_workflow_superseded_error():
    app, provider = _scripted_assistant_app(
        [
            [ModelStreamEvent.text_delta("one"), ModelStreamEvent.completed({})],
            [ModelStreamEvent.text_delta("two"), ModelStreamEvent.completed({})],
        ]
    )
    workflow = TinyWorkflow(app)

    async def run():
        first_attempt = workflow.context("wf-fence-parallel")
        await step(first_attempt, agent="assistant", step_id="s1", prompt="go")
        second_attempt = workflow.context("wf-fence-parallel")
        await step(second_attempt, agent="assistant", step_id="s2", prompt="go")

        # A superseded fence inside a parallel branch must stop the fan-out,
        # not surface as an ordinary StepFailure a caller could skip past.
        with pytest.raises(WorkflowSupersededError):
            await parallel([step(first_attempt, agent="assistant", step_id="s3", prompt="go")])

    asyncio.run(run())
    assert len(provider.requests) == 2


def test_terminal_and_custom_events_check_attempt_fence():
    app = CayuApp(enable_logging=False)
    workflow = TinyWorkflow(app)

    async def run():
        first_attempt = workflow.context("wf-terminal-fence")
        await first_attempt.start()
        second_attempt = workflow.context("wf-terminal-fence")
        await second_attempt.start()

        with pytest.raises(WorkflowSupersededError):
            await first_attempt.emit_custom_event("custom.myapp.late", payload={"ok": False})
        with pytest.raises(WorkflowSupersededError):
            await first_attempt.completed()

        await second_attempt.emit_custom_event("custom.myapp.current", payload={"ok": True})
        await second_attempt.completed()

    asyncio.run(run())
    records = asyncio.run(
        app.session_store.query_events(EventQuery(session_id="wf-terminal-fence", limit=5000))
    )
    event_types = [str(record.event.type) for record in records]
    assert "custom.myapp.late" not in event_types
    assert "custom.myapp.current" in event_types
    assert event_types.count(str(EventType.WORKFLOW_COMPLETED)) == 1


def test_stale_custom_event_cannot_commit_after_newer_attempt_takes_over():
    app = CayuApp(enable_logging=False)
    journal = BlockingCurrentAttemptJournal(blocked_event_type="custom.myapp.late")

    def journal_factory(context: WorkflowJournalContext) -> WorkflowJournal:
        return journal

    workflow = TinyWorkflow(app, journal_factory=journal_factory)

    async def run():
        first_attempt = workflow.context("wf-custom-race")
        await first_attempt.start()
        late = asyncio.create_task(
            first_attempt.emit_custom_event("custom.myapp.late", payload={"ok": False})
        )
        await asyncio.wait_for(journal.entered.wait(), timeout=1)

        second_attempt = workflow.context("wf-custom-race")
        await second_attempt.start()
        journal.release.set()

        with pytest.raises(WorkflowSupersededError):
            await late

        await second_attempt.emit_custom_event("custom.myapp.current", payload={"ok": True})

    asyncio.run(run())
    event_types = [str(event.type) for event in journal.events]
    assert "custom.myapp.late" not in event_types
    assert "custom.myapp.current" in event_types


def test_stale_completed_event_cannot_commit_after_newer_attempt_takes_over():
    app = CayuApp(enable_logging=False)
    journal = BlockingCurrentAttemptJournal(blocked_event_type=EventType.WORKFLOW_COMPLETED)

    def journal_factory(context: WorkflowJournalContext) -> WorkflowJournal:
        return journal

    workflow = TinyWorkflow(app, journal_factory=journal_factory)

    async def run():
        first_attempt = workflow.context("wf-completed-race")
        await first_attempt.start()
        stale_completed = asyncio.create_task(first_attempt.completed())
        await asyncio.wait_for(journal.entered.wait(), timeout=1)

        second_attempt = workflow.context("wf-completed-race")
        await second_attempt.start()
        journal.release.set()

        with pytest.raises(WorkflowSupersededError):
            await stale_completed

        await second_attempt.completed()

    asyncio.run(run())
    completed = [event for event in journal.events if event.type == EventType.WORKFLOW_COMPLETED]
    assert len(completed) == 1
    assert completed[0].payload["workflow"] == "tiny"


def test_superseded_in_flight_step_does_not_journal_completion():
    app = CayuApp(enable_logging=False)
    provider = ControlledProvider()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
    workflow = TinyWorkflow(app)

    async def run():
        first_attempt = workflow.context("wf-stale-step-completion")
        old_step = asyncio.create_task(
            step(first_attempt, agent="assistant", step_id="s1", prompt="go")
        )
        await asyncio.wait_for(provider.entered.wait(), timeout=1)

        second_attempt = workflow.context("wf-stale-step-completion")
        await second_attempt.start()
        provider.release.set()

        with pytest.raises(WorkflowSupersededError):
            await old_step

        records = await app.session_store.query_events(
            EventQuery(
                session_id="wf-stale-step-completion",
                event_type=EventType.WORKFLOW_STEP_COMPLETED,
                limit=100,
            )
        )
        assert records == []

    asyncio.run(run())


def test_superseded_in_flight_gated_loop_item_does_not_journal_completion():
    app = CayuApp(enable_logging=False)
    workflow = TinyWorkflow(app)
    entered = asyncio.Event()
    release = asyncio.Event()
    do_calls: list[str] = []

    async def do(item):
        do_calls.append(item)
        entered.set()
        await release.wait()
        return StepResult(step_id="manual", session_id="manual-child")

    async def run():
        first_attempt = workflow.context("wf-stale-loop-completion")
        old_loop = asyncio.create_task(
            _collect_gated_loop(first_attempt, ["item"], do=do, name="items")
        )
        await asyncio.wait_for(entered.wait(), timeout=1)

        second_attempt = workflow.context("wf-stale-loop-completion")
        await second_attempt.start()
        release.set()

        with pytest.raises(WorkflowSupersededError):
            await old_loop

        records = await app.session_store.query_events(
            EventQuery(
                session_id="wf-stale-loop-completion",
                event_type=EventType.WORKFLOW_STEP_COMPLETED,
                limit=100,
            )
        )
        assert records == []

    asyncio.run(run())
    assert do_calls == ["item"]


async def _collect_gated_loop(ctx, items, *, do, name):
    return [
        event
        async for event in gated_loop(
            ctx,
            items,
            do=do,
            gate=_passing_gate,
            key=str,
            name=name,
        )
    ]


def test_workflow_replay_ignores_stale_completions_after_current_attempt():
    app = CayuApp(enable_logging=False)
    store = app.session_store
    previous_attempt = "previous"
    intermediate_attempt = "intermediate"
    current_attempt = "current"

    async def run():
        await store.create(
            RunRequest(
                agent_name="wf",
                session_id="wf-replay-attempt-prefix",
                messages=[],
                metadata={"cayu.workflow": "wf"},
            ),
            identity=SessionIdentity(
                provider_name=WORKFLOW_JOURNAL_PROVIDER,
                model=WORKFLOW_JOURNAL_MODEL,
            ),
        )
        await store.append_events(
            "wf-replay-attempt-prefix",
            [
                Event(
                    type=WORKFLOW_ATTEMPT_EVENT_TYPE,
                    session_id="wf-replay-attempt-prefix",
                    workflow_name="wf",
                    payload={"attempt_id": previous_attempt},
                ),
                Event(
                    type=EventType.WORKFLOW_STEP_COMPLETED,
                    session_id="wf-replay-attempt-prefix",
                    workflow_name="wf",
                    payload={
                        "step_id": "valid-prefix",
                        "child_session_id": "child-prefix",
                        "attempt_id": previous_attempt,
                    },
                ),
                Event(
                    type=WORKFLOW_ATTEMPT_EVENT_TYPE,
                    session_id="wf-replay-attempt-prefix",
                    workflow_name="wf",
                    payload={"attempt_id": intermediate_attempt},
                ),
                Event(
                    type=EventType.WORKFLOW_STEP_COMPLETED,
                    session_id="wf-replay-attempt-prefix",
                    workflow_name="wf",
                    payload={
                        "step_id": "stale-before-current",
                        "child_session_id": "child-stale-before-current",
                        "attempt_id": previous_attempt,
                    },
                ),
                Event(
                    type=WORKFLOW_ATTEMPT_EVENT_TYPE,
                    session_id="wf-replay-attempt-prefix",
                    workflow_name="wf",
                    payload={"attempt_id": current_attempt},
                ),
                Event(
                    type=EventType.WORKFLOW_STEP_COMPLETED,
                    session_id="wf-replay-attempt-prefix",
                    workflow_name="wf",
                    payload={
                        "step_id": "stale-after-current",
                        "child_session_id": "child-stale",
                        "attempt_id": previous_attempt,
                    },
                ),
            ],
        )
        return await EventStoreJournal(
            store,
            "wf-replay-attempt-prefix",
            "wf",
        ).completed_step_ids(attempt_id=current_attempt)

    assert asyncio.run(run()) == {"valid-prefix"}


def test_concurrent_first_step_is_durably_reserved_before_child_run():
    app, provider = _scripted_assistant_app(
        [
            [ModelStreamEvent.text_delta("one"), ModelStreamEvent.completed({})],
            [ModelStreamEvent.text_delta("two"), ModelStreamEvent.completed({})],
        ]
    )
    both_ready = asyncio.Event()
    ready_count = 0

    class DelayedStartedJournal(EventStoreJournal):
        async def _append_events(self, events: list[Event]) -> None:
            nonlocal ready_count
            if any(event.type == EventType.WORKFLOW_STEP_STARTED for event in events):
                ready_count += 1
                if ready_count == 2:
                    both_ready.set()
                await both_ready.wait()
            await super()._append_events(events)

    def journal_factory(context: WorkflowJournalContext) -> WorkflowJournal:
        return DelayedStartedJournal(
            context.session_store,
            context.session_id,
            context.workflow_name,
            event_emitter=context.emit_events,
        )

    workflow = TinyWorkflow(app, journal_factory=journal_factory)

    async def run():
        first_attempt = workflow.context("wf-first-step-claim")
        second_attempt = workflow.context("wf-first-step-claim")
        outcomes = await asyncio.gather(
            step(first_attempt, agent="assistant", step_id="s1", prompt="go"),
            step(second_attempt, agent="assistant", step_id="s1", prompt="go"),
            return_exceptions=True,
        )
        results = [outcome for outcome in outcomes if isinstance(outcome, StepResult)]
        superseded = [
            outcome for outcome in outcomes if isinstance(outcome, WorkflowSupersededError)
        ]
        assert len(results) >= 1
        assert len(results) + len(superseded) == 2
        assert len({result.session_id for result in results}) == 1

    asyncio.run(run())
    assert len(provider.requests) == 1


def test_step_structured_output_returns_unredacted_typed_edge():
    secret = "sk-live-workflow-output-secret"
    app = CayuApp(
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    _register_scripted_assistant(app, [_submit({"token": secret})])
    schema = {
        "type": "object",
        "properties": {"token": {"type": "string"}},
        "required": ["token"],
        "additionalProperties": False,
    }
    ctx = TinyWorkflow(app).context("wf-structured-output-raw")

    result = asyncio.run(
        step(ctx, agent="assistant", step_id="structured", prompt="go", schema=schema)
    )

    assert result.output == {"token": secret}
    events = asyncio.run(app.session_store.load_events(result.session_id))
    validated = [event for event in events if event.type == EventType.STRUCTURED_OUTPUT_VALIDATED]
    assert validated[-1].payload["output"] == {"token": REDACTED_SECRET}


def test_emit_events_rejects_runtime_namespace_events():
    app = CayuApp(enable_logging=False)

    async def run():
        with pytest.raises(ValueError, match="workflow. or custom."):
            await app.emit_events(
                "wf-emit-guard",
                [Event(type=EventType.MODEL_COMPLETED, session_id="wf-emit-guard")],
            )
        with pytest.raises(ValueError, match="custom\\.cayu\\. namespace is reserved"):
            await app.emit_events(
                "wf-emit-guard",
                [
                    Event(
                        type=WORKFLOW_ATTEMPT_EVENT_TYPE,
                        session_id="wf-emit-guard",
                        workflow_name="tiny",
                        payload={"attempt_id": "forged"},
                    )
                ],
            )

    asyncio.run(run())
