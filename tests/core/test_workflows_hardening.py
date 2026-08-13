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
from types import SimpleNamespace
from typing import Any

import pytest
from tests._session_provenance import fixture_session_invocation

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
    InMemorySessionStore,
    InvocationOriginTrust,
    ModelPrice,
    PriceBook,
    RetryPolicy,
    RunLimits,
    RunRequest,
    SessionExecutionSource,
    SessionIdentity,
    SessionStatus,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.runtime._workflow_structured_output_handoff import (
    WorkflowStructuredOutputHandoff,
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
    WorkflowJournalReplayEvidence,
    WorkflowStepCompletionSnapshot,
    WorkflowSupersededError,
    canonical_workflow_step_completion_ids,
    gated_loop,
    parallel,
    pipeline,
    step,
)
from cayu.workflows._step_identity import (
    gated_loop_step_id,
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
        pricing=PriceBook(
            prices=(
                ModelPrice.fixed(
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
            if self._event_attempt_id(event) != active_attempt_id:
                continue
            step_id = event.payload.get("step_id")
            if isinstance(step_id, str) and step_id:
                completed.add(step_id)
        return completed

    async def completed_step_snapshot(
        self,
        *,
        attempt_id: str,
    ) -> WorkflowStepCompletionSnapshot:
        latest_attempt, _sequence = self._latest_attempt()
        attempt_marker = next(
            (
                event
                for event in reversed(self.events)
                if event.type == WORKFLOW_ATTEMPT_EVENT_TYPE
                and self._event_attempt_id(event) == attempt_id
            ),
            None,
        )
        if latest_attempt != attempt_id:
            return canonical_workflow_step_completion_ids(
                (),
                session_id=(
                    attempt_marker.session_id
                    if attempt_marker is not None
                    else "empty-workflow-run"
                ),
                workflow_name=(
                    attempt_marker.workflow_name
                    if attempt_marker is not None and attempt_marker.workflow_name is not None
                    else "empty-workflow"
                ),
                attempt_id=attempt_id,
            )
        completed: list[Event] = []
        active_attempt_id: str | None = None
        for event in self.events:
            if event.type == WORKFLOW_ATTEMPT_EVENT_TYPE:
                active_attempt_id = self._event_attempt_id(event)
                continue
            if event.type != EventType.WORKFLOW_STEP_COMPLETED:
                continue
            if self._event_attempt_id(event) == active_attempt_id:
                completed.append(event)
        if not completed:
            assert attempt_marker is not None
            assert attempt_marker.workflow_name is not None
            return canonical_workflow_step_completion_ids(
                (),
                session_id=attempt_marker.session_id,
                workflow_name=attempt_marker.workflow_name,
                attempt_id=attempt_id,
            )
        workflow_name = completed[0].workflow_name
        assert workflow_name is not None
        return canonical_workflow_step_completion_ids(
            completed,
            session_id=completed[0].session_id,
            workflow_name=workflow_name,
            attempt_id=attempt_id,
        )

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


class PreviousContractMemoryJournal(MemoryJournal):
    """Custom journal implementing Cayu's former raw-ID replay contract."""

    completed_step_snapshot = None  # type: ignore[assignment]


class NaiveRawSnapshotJournal(MemoryJournal):
    """Migration attempt that incorrectly wraps the former raw-ID result."""

    async def completed_step_snapshot(
        self,
        *,
        attempt_id: str,
    ) -> WorkflowStepCompletionSnapshot:
        return WorkflowStepCompletionSnapshot(
            step_ids=frozenset(await self.completed_step_ids(attempt_id=attempt_id))
        )


class FixedCompletionEvidenceJournal(MemoryJournal):
    def __init__(self) -> None:
        super().__init__()
        self.completion_evidence: Any = ()
        self.expected_session_id = ""
        self.expected_workflow_name = ""
        self.snapshot_override: Any = None

    async def completed_step_snapshot(
        self,
        *,
        attempt_id: str,
    ) -> WorkflowStepCompletionSnapshot:
        if self.snapshot_override is not None:
            return self.snapshot_override
        return canonical_workflow_step_completion_ids(
            self.completion_evidence,
            session_id=self.expected_session_id,
            workflow_name=self.expected_workflow_name,
            attempt_id=attempt_id,
        )


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
        # "item" because the first journaled a gated-loop step; explicit per-loop
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
    assert {
        gated_loop_step_id("first", "item"),
        gated_loop_step_id("second", "item"),
    } <= journaled


def test_gated_loop_identity_is_injective_across_delimiter_boundaries():
    app = CayuApp(enable_logging=False)
    ctx = TinyWorkflow(app).context("wf-delimiter-identity")
    calls: list[str] = []

    async def do(item: str) -> StepResult:
        calls.append(item)
        return StepResult(step_id=f"do-{item}", session_id=f"child-{len(calls)}")

    async def run() -> None:
        async for _event in gated_loop(
            ctx,
            ["b:c"],
            do=do,
            gate=_passing_gate,
            key=str,
            name="a",
        ):
            pass
        async for _event in gated_loop(
            ctx,
            ["c"],
            do=do,
            gate=_passing_gate,
            key=str,
            name="a:b",
        ):
            pass

    asyncio.run(run())

    assert calls == ["b:c", "c"]

    records = asyncio.run(
        app.session_store.query_events(
            EventQuery(
                session_id="wf-delimiter-identity",
                event_type=EventType.WORKFLOW_STEP_COMPLETED,
            )
        )
    )
    loop_events = [
        record.event for record in records if record.event.payload.get("kind") == "gated_loop"
    ]
    assert len(loop_events) == 2
    assert len({event.payload["step_id"] for event in loop_events}) == 2
    assert {(event.payload["loop_name"], event.payload["item_key"]) for event in loop_events} == {
        ("a", "b:c"),
        ("a:b", "c"),
    }


def test_gated_loop_replays_legacy_identity_without_cross_skipping_collision() -> None:
    app = CayuApp(enable_logging=False)
    first = TinyWorkflow(app).context("wf-legacy-delimiter-identity")

    async def seed_legacy_completion() -> None:
        await first.journal.append(first.event(WORKFLOW_ATTEMPT_EVENT_TYPE))
        appended = await first.journal.append_current_attempt(
            first.event(
                EventType.WORKFLOW_STEP_COMPLETED,
                payload={
                    "step_id": "gated-loop:a:b:c",
                    "item_key": "b:c",
                    "kind": "gated_loop",
                    "passed": True,
                    "outcome": "pass",
                    "child_session_id": "legacy-child",
                },
            ),
            attempt_id=first.attempt_id,
        )
        assert appended is True

    asyncio.run(seed_legacy_completion())

    resumed = TinyWorkflow(app).context("wf-legacy-delimiter-identity")
    calls: list[str] = []

    async def do(item: str) -> StepResult:
        calls.append(item)
        return StepResult(step_id=f"do-{item}", session_id=f"child-{item}")

    async def run() -> None:
        async for _event in gated_loop(
            resumed,
            ["b:c"],
            do=do,
            gate=_passing_gate,
            key=str,
            name="a",
        ):
            pass
        async for _event in gated_loop(
            resumed,
            ["c"],
            do=do,
            gate=_passing_gate,
            key=str,
            name="a:b",
        ):
            pass

    asyncio.run(run())

    assert calls == ["c"]


@pytest.mark.parametrize("journal_kind", ["event-store", "custom"])
def test_authentic_legacy_v2_shaped_id_does_not_complete_unrelated_modern_item(
    journal_kind: str,
) -> None:
    app = CayuApp(enable_logging=False)
    run_id = "wf-legacy-modern-prefix-collision"
    if journal_kind == "custom":
        custom_journal = MemoryJournal()

        def journal_factory(_context: WorkflowJournalContext) -> WorkflowJournal:
            return custom_journal

        workflow = TinyWorkflow(app, journal_factory=journal_factory)
    else:
        workflow = TinyWorkflow(app)
    first = workflow.context(run_id)
    unrelated_modern_id = gated_loop_step_id("orders", "42")
    legacy_item_key = unrelated_modern_id.removeprefix("gated-loop:v2:")

    async def seed_legacy_completion() -> None:
        await first.journal.append(first.event(WORKFLOW_ATTEMPT_EVENT_TYPE))
        appended = await first.journal.append_current_attempt(
            first.event(
                EventType.WORKFLOW_STEP_COMPLETED,
                payload={
                    "step_id": unrelated_modern_id,
                    "item_key": legacy_item_key,
                    "kind": "gated_loop",
                    "passed": True,
                    "outcome": "pass",
                    "child_session_id": "legacy-child",
                },
            ),
            attempt_id=first.attempt_id,
        )
        assert appended is True

    asyncio.run(seed_legacy_completion())

    resumed = workflow.context(run_id)
    legacy_calls: list[str] = []
    modern_calls: list[str] = []

    async def legacy_do(item: str) -> StepResult:
        legacy_calls.append(item)
        return StepResult(step_id="legacy-do", session_id="legacy-child-rerun")

    async def modern_do(item: str) -> StepResult:
        modern_calls.append(item)
        return StepResult(step_id="modern-do", session_id="modern-child")

    async def run() -> None:
        async for _event in gated_loop(
            resumed,
            [legacy_item_key],
            do=legacy_do,
            gate=_passing_gate,
            key=str,
            name="v2",
        ):
            pass
        async for _event in gated_loop(
            resumed,
            ["42"],
            do=modern_do,
            gate=_passing_gate,
            key=str,
            name="orders",
        ):
            pass

    asyncio.run(run())

    assert legacy_calls == []
    assert modern_calls == ["42"]


def test_previous_contract_custom_journal_cannot_authorize_ambiguous_gated_loop_id() -> None:
    app = CayuApp(enable_logging=False)
    journal = PreviousContractMemoryJournal()

    def journal_factory(_context: WorkflowJournalContext) -> WorkflowJournal:
        return journal

    assert isinstance(journal, WorkflowJournal)
    assert not isinstance(journal, WorkflowJournalReplayEvidence)

    workflow = TinyWorkflow(app, journal_factory=journal_factory)
    first = workflow.context("wf-previous-contract-collision")
    colliding_modern_id = gated_loop_step_id("orders", "42")
    legacy_item_key = colliding_modern_id.removeprefix("gated-loop:v2:")

    async def seed_legacy_completion() -> None:
        await first.journal.append(first.event(WORKFLOW_ATTEMPT_EVENT_TYPE))
        appended = await first.journal.append_current_attempt(
            first.event(
                EventType.WORKFLOW_STEP_COMPLETED,
                payload={
                    "step_id": colliding_modern_id,
                    "item_key": legacy_item_key,
                    "kind": "gated_loop",
                    "passed": True,
                    "outcome": "pass",
                    "child_session_id": "legacy-child",
                },
            ),
            attempt_id=first.attempt_id,
        )
        assert appended is True
        assert await journal.completed_step_ids(attempt_id=first.attempt_id) == {
            colliding_modern_id
        }

    asyncio.run(seed_legacy_completion())
    event_count_before_replay = len(journal.events)
    resumed = workflow.context("wf-previous-contract-collision")
    calls: list[str] = []

    async def do(item: str) -> StepResult:
        calls.append(item)
        return StepResult(step_id="do", session_id="child")

    async def run() -> None:
        async for _event in gated_loop(
            resumed,
            ["42"],
            do=do,
            gate=_passing_gate,
            key=str,
            name="orders",
        ):
            pass

    with pytest.raises(TypeError, match="WorkflowJournalReplayEvidence"):
        asyncio.run(run())

    assert calls == []
    assert len(journal.events) == event_count_before_replay

    naive_journal = NaiveRawSnapshotJournal()
    naive_journal.events = list(journal.events)
    resumed.journal = naive_journal
    with pytest.raises(TypeError, match="canonical_workflow_step_completion_ids"):
        asyncio.run(run())

    assert calls == []

    migrated_journal = MemoryJournal()
    migrated_journal.events = list(naive_journal.events)
    resumed.journal = migrated_journal
    asyncio.run(run())

    assert calls == ["42"]


@pytest.mark.parametrize(
    "evidence_update,expected_error",
    [
        pytest.param("list", TypeError, id="mutable-result"),
        pytest.param("forged", TypeError, id="forged-snapshot"),
        pytest.param("mutated", TypeError, id="mutated-snapshot"),
        pytest.param("other-scope", TypeError, id="transplanted-snapshot"),
        pytest.param("other-attempt", TypeError, id="wrong-attempt-snapshot"),
        pytest.param({"type": EventType.WORKFLOW_STEP_STARTED}, ValueError, id="wrong-type"),
        pytest.param({"session_id": "other-run"}, ValueError, id="wrong-run"),
        pytest.param({"workflow_name": "other-workflow"}, ValueError, id="wrong-workflow"),
        pytest.param({"payload": {"attempt_id": ""}}, ValueError, id="blank-attempt"),
    ],
)
def test_gated_loop_revalidates_custom_journal_completion_evidence_before_callback(
    evidence_update: str | dict[str, Any],
    expected_error: type[Exception],
) -> None:
    app = CayuApp(enable_logging=False)
    journal = FixedCompletionEvidenceJournal()

    def journal_factory(_context: WorkflowJournalContext) -> WorkflowJournal:
        return journal

    ctx = TinyWorkflow(app, journal_factory=journal_factory).context(
        "wf-custom-completion-evidence"
    )
    journal.expected_session_id = ctx.session_id
    journal.expected_workflow_name = ctx.workflow_name
    valid_event = ctx.event(
        EventType.WORKFLOW_STEP_COMPLETED,
        payload={
            "step_id": gated_loop_step_id("orders", "42"),
            "step_id_version": 2,
            "kind": "gated_loop",
            "loop_name": "orders",
            "item_key": "42",
            "child_session_id": "child",
        },
    )
    if evidence_update == "list":
        journal.snapshot_override = [gated_loop_step_id("orders", "42")]
    elif evidence_update == "forged":
        forged = object.__new__(WorkflowStepCompletionSnapshot)
        object.__setattr__(
            forged,
            "_step_ids",
            frozenset({gated_loop_step_id("orders", "42")}),
        )
        journal.snapshot_override = forged
    elif evidence_update == "mutated":
        mutated = canonical_workflow_step_completion_ids(
            (valid_event,),
            session_id=ctx.session_id,
            workflow_name=ctx.workflow_name,
            attempt_id=ctx.attempt_id,
        )
        object.__setattr__(
            mutated,
            "_step_ids",
            frozenset({gated_loop_step_id("other", "item")}),
        )
        journal.snapshot_override = mutated
    elif evidence_update == "other-scope":
        other_event = valid_event.model_copy(update={"session_id": "other-run"})
        journal.snapshot_override = canonical_workflow_step_completion_ids(
            (other_event,),
            session_id="other-run",
            workflow_name=ctx.workflow_name,
            attempt_id=ctx.attempt_id,
        )
    elif evidence_update == "other-attempt":
        journal.snapshot_override = canonical_workflow_step_completion_ids(
            (valid_event,),
            session_id=ctx.session_id,
            workflow_name=ctx.workflow_name,
            attempt_id="other-attempt",
        )
    else:
        journal.completion_evidence = (valid_event.model_copy(update=evidence_update),)
    calls: list[str] = []

    async def do(item: str) -> StepResult:
        calls.append(item)
        return StepResult(step_id="do", session_id="child")

    async def run() -> None:
        async for _event in gated_loop(
            ctx,
            ["42"],
            do=do,
            gate=_passing_gate,
            key=str,
            name="orders",
        ):
            pass

    with pytest.raises(expected_error):
        asyncio.run(run())

    assert calls == []


@pytest.mark.parametrize(
    "legacy_evidence",
    [
        pytest.param({}, id="missing-kind"),
        pytest.param({"kind": "ordinary"}, id="wrong-kind"),
        pytest.param({"kind": ""}, id="blank-kind"),
        pytest.param({"kind": True}, id="boolean-kind"),
        pytest.param({"kind": "gated_loop", "item_key": None}, id="missing-item-key"),
        pytest.param({"kind": "gated_loop", "item_key": True}, id="boolean-item-key"),
        pytest.param({"kind": "gated_loop", "item_key": 1}, id="numeric-item-key"),
        pytest.param({"kind": "gated_loop", "item_key": []}, id="list-item-key"),
        pytest.param({"kind": "gated_loop", "item_key": {}}, id="object-item-key"),
        pytest.param({"kind": "gated_loop", "item_key": ""}, id="empty-item-key"),
        pytest.param({"kind": "gated_loop", "item_key": " "}, id="blank-item-key"),
        pytest.param(
            {"kind": "gated_loop", "item_key": " item"},
            id="leading-whitespace-item-key",
        ),
        pytest.param(
            {"kind": "gated_loop", "item_key": "item "},
            id="trailing-whitespace-item-key",
        ),
        pytest.param(
            {"kind": "gated_loop", "step_id_version": None},
            id="version-field-present",
        ),
    ],
)
def test_gated_loop_does_not_replay_lookalike_without_authentic_legacy_evidence(
    legacy_evidence: dict[str, Any],
) -> None:
    app = CayuApp(enable_logging=False)
    run_id = "wf-legacy-lookalike"
    attempt_id = "legacy-attempt"
    journal = EventStoreJournal(app.session_store, run_id, "tiny")
    unrelated_modern_id = gated_loop_step_id("orders", "42")
    legacy_item_key = unrelated_modern_id.removeprefix("gated-loop:v2:")

    async def seed_lookalike_completion() -> None:
        await journal.append(
            Event(
                type=WORKFLOW_ATTEMPT_EVENT_TYPE,
                session_id=run_id,
                workflow_name="tiny",
                payload={"attempt_id": attempt_id},
            )
        )
        payload: dict[str, Any] = {
            "step_id": unrelated_modern_id,
            "item_key": legacy_item_key,
            "passed": True,
            "outcome": "pass",
            "child_session_id": "lookalike-child",
            "attempt_id": attempt_id,
            **legacy_evidence,
        }
        appended = await journal.append_current_attempt(
            Event(
                type=EventType.WORKFLOW_STEP_COMPLETED,
                session_id=run_id,
                workflow_name="tiny",
                payload=payload,
            ),
            attempt_id=attempt_id,
        )
        assert appended is True

    asyncio.run(seed_lookalike_completion())

    resumed = TinyWorkflow(app).context(run_id)
    calls: list[str] = []

    async def do(item: str) -> StepResult:
        calls.append(item)
        return StepResult(step_id=f"do-{item}", session_id=f"child-{item}")

    async def run() -> None:
        async for _event in gated_loop(
            resumed,
            ["42"],
            do=do,
            gate=_passing_gate,
            key=str,
            name="orders",
        ):
            pass

    asyncio.run(run())

    assert calls == ["42"]


def test_sqlite_replay_migrates_only_authentic_legacy_gated_loop_evidence(
    tmp_path,
) -> None:
    db_path = tmp_path / "legacy-gated-loop.db"
    run_id = "wf-sqlite-legacy-gated-loop"
    attempt_id = "legacy-attempt"
    first_app = CayuApp(enable_logging=False, session_store=SQLiteSessionStore(db_path))
    journal = EventStoreJournal(first_app.session_store, run_id, "tiny")

    async def seed_legacy_completions() -> None:
        await journal.append(
            Event(
                type=WORKFLOW_ATTEMPT_EVENT_TYPE,
                session_id=run_id,
                workflow_name="tiny",
                payload={"attempt_id": attempt_id},
            )
        )
        for item_key, kind in (
            ("ambiguous-before", None),
            ("authentic", "gated_loop"),
            ("ambiguous-after", "ordinary"),
        ):
            payload: dict[str, Any] = {
                "step_id": f"gated-loop:loop:{item_key}",
                "item_key": item_key,
                "passed": True,
                "outcome": "pass",
                "child_session_id": f"legacy-child-{item_key}",
                "attempt_id": attempt_id,
            }
            if kind is not None:
                payload["kind"] = kind
            appended = await journal.append_current_attempt(
                Event(
                    type=EventType.WORKFLOW_STEP_COMPLETED,
                    session_id=run_id,
                    workflow_name="tiny",
                    payload=payload,
                ),
                attempt_id=attempt_id,
            )
            assert appended is True

    asyncio.run(seed_legacy_completions())
    asyncio.run(first_app.session_store.close())

    resumed_app = CayuApp(enable_logging=False, session_store=SQLiteSessionStore(db_path))
    resumed = TinyWorkflow(resumed_app).context(run_id)
    calls: list[str] = []

    async def do(item: str) -> StepResult:
        calls.append(item)
        return StepResult(step_id=f"do-{item}", session_id=f"child-{item}")

    async def run() -> None:
        async for _event in gated_loop(
            resumed,
            ["ambiguous-before", "authentic", "ambiguous-after"],
            do=do,
            gate=_passing_gate,
            key=str,
            name="loop",
        ):
            pass

    asyncio.run(run())

    assert calls == ["ambiguous-before", "ambiguous-after"]
    asyncio.run(resumed_app.session_store.close())


@pytest.mark.parametrize(
    ("loop_name", "item_key"),
    [
        ("v2", "item"),
        ("v2:name", "item"),
        ("v2", "a" * 64),
    ],
)
def test_gated_loop_replays_unversioned_legacy_identity_that_looks_like_v2(
    loop_name: str,
    item_key: str,
) -> None:
    app = CayuApp(enable_logging=False)
    first = TinyWorkflow(app).context("wf-legacy-v2-shaped-identity")

    async def seed_legacy_completion() -> None:
        await first.journal.append(first.event(WORKFLOW_ATTEMPT_EVENT_TYPE))
        appended = await first.journal.append_current_attempt(
            first.event(
                EventType.WORKFLOW_STEP_COMPLETED,
                payload={
                    "step_id": f"gated-loop:{loop_name}:{item_key}",
                    "item_key": item_key,
                    "kind": "gated_loop",
                    "passed": True,
                    "outcome": "pass",
                    "child_session_id": "legacy-child",
                },
            ),
            attempt_id=first.attempt_id,
        )
        assert appended is True

    asyncio.run(seed_legacy_completion())

    resumed = TinyWorkflow(app).context("wf-legacy-v2-shaped-identity")
    calls: list[str] = []

    async def do(item: str) -> StepResult:
        calls.append(item)
        return StepResult(step_id="unexpected", session_id="unexpected-child")

    async def run() -> None:
        async for _event in gated_loop(
            resumed,
            [item_key],
            do=do,
            gate=_passing_gate,
            key=str,
            name=loop_name,
        ):
            pass

    asyncio.run(run())

    assert calls == []


def test_step_replay_ids_returns_a_bounded_number_of_records_for_many_steps() -> None:
    class CountingStore(InMemorySessionStore):
        returned_event_records = 0
        candidate_event_records = 0

        async def query_events(self, query=None):
            records = await super().query_events(query)
            self.returned_event_records += len(records)
            return records

        def _query_candidate_records(self, query, event_types):
            records = super()._query_candidate_records(query, event_types)
            self.candidate_event_records += len(records)
            return records

    store = CountingStore()
    app = CayuApp(session_store=store, enable_logging=False)
    ctx = TinyWorkflow(app).context("wf-bounded-step-replay")

    async def seed_and_lookup() -> list[tuple[str | None, str | None]]:
        await ctx.journal.append(ctx.event(WORKFLOW_ATTEMPT_EVENT_TYPE))
        step_ids = [f"step-{index}" for index in range(1000)]
        await store.append_events(
            ctx.session_id,
            [
                Event(
                    id=f"completion-{index}",
                    type=EventType.WORKFLOW_STEP_COMPLETED,
                    session_id=ctx.session_id,
                    workflow_name=ctx.workflow_name,
                    payload={
                        "attempt_id": ctx.attempt_id,
                        "step_id": step_id,
                        "child_session_id": f"child-{index}",
                    },
                )
                for index, step_id in enumerate(step_ids)
            ],
        )
        store.returned_event_records = 0
        store.candidate_event_records = 0
        results = [
            await ctx.journal.step_replay_ids(
                step_id=step_id,
                attempt_id=ctx.attempt_id,
            )
            for step_id in step_ids
        ]
        assert store.returned_event_records <= (2 * len(step_ids)) + 1
        assert store.candidate_event_records <= (2 * len(step_ids)) + 1
        return results

    replay_ids = asyncio.run(seed_and_lookup())

    assert replay_ids == [(f"child-{index}", None) for index in range(1000)]


class TinyWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="tiny")

    async def run(self, session_id):
        ctx = self.context(session_id)
        yield await ctx.start()
        yield await ctx.completed()


@pytest.mark.parametrize("invalid_name", ["bad\x00name", "bad\ud800name"])
@pytest.mark.parametrize("bypass", ["model_copy", "mutation"])
def test_workflow_base_revalidates_spec_before_journal_factory(
    invalid_name: str,
    bypass: str,
):
    spec = WorkflowSpec(name="valid")
    if bypass == "model_copy":
        spec = spec.model_copy(update={"name": invalid_name})
    else:
        spec.name = invalid_name
    factory_calls: list[WorkflowJournalContext] = []

    def journal_factory(context: WorkflowJournalContext) -> WorkflowJournal:
        factory_calls.append(context)
        return MemoryJournal()

    with pytest.raises(ValueError):
        workflow = TinyWorkflow(
            CayuApp(enable_logging=False),
            spec=spec,
            journal_factory=journal_factory,
        )
        workflow.context("wf-invalid-spec")

    assert factory_calls == []


@pytest.mark.parametrize("invalid_name", ["bad\x00name", "bad\ud800name"])
def test_workflow_context_revalidates_direct_spec(invalid_name: str):
    spec = WorkflowSpec(name="valid").model_copy(update={"name": invalid_name})

    with pytest.raises(ValueError):
        WorkflowContext(
            app=CayuApp(enable_logging=False),
            spec=spec,
            session_id="wf-invalid-context",
            journal=MemoryJournal(),
        )


@pytest.mark.parametrize("invalid_name", ["bad\x00name", "bad\ud800name"])
def test_workflow_context_revalidates_owned_spec_before_journal_factory(
    invalid_name: str,
):
    factory_calls: list[WorkflowJournalContext] = []

    def journal_factory(context: WorkflowJournalContext) -> WorkflowJournal:
        factory_calls.append(context)
        return MemoryJournal()

    workflow = TinyWorkflow(
        CayuApp(enable_logging=False),
        spec=WorkflowSpec(name="valid"),
        journal_factory=journal_factory,
    )
    workflow.spec.name = invalid_name

    with pytest.raises(ValueError):
        workflow.context("wf-mutated-owned-spec")

    assert factory_calls == []


def test_workflow_context_uses_detached_stable_unicode_name_snapshot():
    workflow_name = "Zażółć_日本語_😀"
    source_spec = WorkflowSpec(name=workflow_name)
    factory_contexts: list[WorkflowJournalContext] = []

    def journal_factory(context: WorkflowJournalContext) -> WorkflowJournal:
        factory_contexts.append(context)
        return MemoryJournal()

    workflow = TinyWorkflow(
        CayuApp(enable_logging=False),
        spec=source_spec,
        journal_factory=journal_factory,
    )
    source_spec.name = "caller\x00mutation"
    context = workflow.context("wf-unicode-spec")
    context.spec.name = "context\x00mutation"

    assert factory_contexts[0].workflow_name == workflow_name
    assert context.workflow_name == workflow_name
    assert context.event(EventType.WORKFLOW_STARTED).workflow_name == workflow_name


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

    page = asyncio.run(sweep())  # must not raise
    assert page.results == ()
    assert page.next_cursor is None
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

    page = asyncio.run(run())  # must not raise
    assert page.results == ()
    assert page.next_cursor is None


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
            step_event_reserver=context.reserve_step_started,
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
            step_event_reserver=context.reserve_step_started,
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
            step_event_reserver=context.reserve_step_started,
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


def test_workflow_journal_completion_snapshot_retains_only_canonical_identities():
    app = CayuApp(enable_logging=False)
    store = app.session_store

    async def run() -> WorkflowStepCompletionSnapshot:
        attempt_id = "attempt-compact"
        await store.create(
            RunRequest(
                agent_name="wf",
                session_id="wf-compact-completions",
                messages=[],
                metadata={"cayu.workflow": "wf"},
            ),
            identity=SessionIdentity(
                provider_name=WORKFLOW_JOURNAL_PROVIDER,
                model=WORKFLOW_JOURNAL_MODEL,
            ),
        )
        await store.append_events(
            "wf-compact-completions",
            [
                Event(
                    type=WORKFLOW_ATTEMPT_EVENT_TYPE,
                    session_id="wf-compact-completions",
                    workflow_name="wf",
                    payload={"attempt_id": attempt_id},
                ),
                Event(
                    type=EventType.WORKFLOW_STEP_COMPLETED,
                    session_id="wf-compact-completions",
                    workflow_name="wf",
                    payload={
                        "step_id": "s1",
                        "attempt_id": attempt_id,
                        "detail": "x" * 1_000_000,
                    },
                ),
                Event(
                    type=EventType.WORKFLOW_STEP_COMPLETED,
                    session_id="wf-compact-completions",
                    workflow_name="wf",
                    payload={
                        "step_id": "s1",
                        "attempt_id": attempt_id,
                        "detail": "y" * 1_000_000,
                    },
                ),
            ],
        )
        return await EventStoreJournal(
            store,
            "wf-compact-completions",
            "wf",
        ).completed_step_snapshot(attempt_id=attempt_id)

    snapshot = asyncio.run(run())

    assert type(snapshot) is WorkflowStepCompletionSnapshot
    assert snapshot.step_ids == frozenset({"s1"})
    assert not hasattr(snapshot, "events")


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
        active_stage = await app.session_store.load_active_model_completion_stage(child_session_id)
        events = await app.session_store.load_events(child_session_id)
        return child_session_id, child, active_stage, events

    child_session_id, child, active_stage, events = asyncio.run(cancel_running_step())

    assert provider.closed is True
    assert child is not None
    assert child.status == SessionStatus.INTERRUPTED
    assert active_stage is not None
    assert active_stage.stage.state == "in_flight"
    interrupted = [event for event in events if event.type == EventType.SESSION_INTERRUPTED]
    assert len(interrupted) == 1
    assert interrupted[0].payload["abandoned"] is True

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
    assert len(provider.requests) == 1


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
                    tool_round_id=approval["tool_round_id"],
                    tool_call_id=approval["tool_call_id"],
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
    anchor = asyncio.run(app.session_store.load("wf-lineage"))
    assert anchor is not None
    assert child.invocation.origin == anchor.invocation.origin
    assert child.invocation.root_session_id == anchor.id
    assert child.invocation.source is SessionExecutionSource.WORKFLOW_STEP


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
    assert child.invocation.root_session_id == child.id
    assert child.invocation.source is SessionExecutionSource.WORKFLOW_STEP
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
    assert workflows.WorkflowJournalReplayEvidence is WorkflowJournalReplayEvidence
    assert workflows.WorkflowStepCompletionSnapshot is WorkflowStepCompletionSnapshot
    assert callable(workflows.canonical_workflow_step_completion_ids)
    assert callable(workflows.copy_workflow_step_completion_snapshot)
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
        if record.event.payload["step_id"] == gated_loop_step_id("fixes", "three")
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
        if record.event.payload["step_id"] == gated_loop_step_id("fixes", "three")
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
    workflow = TinyWorkflow(app)

    async def run():
        first_attempt = workflow.context("wf-first-step-claim")
        second_attempt = workflow.context("wf-first-step-claim")
        both_prepared = asyncio.Event()
        prepared_count = 0

        def gate_anchor(context: WorkflowContext) -> None:
            original_anchor = context._workflow_anchor

            async def gated_anchor():
                nonlocal prepared_count
                anchor = await original_anchor()
                prepared_count += 1
                if prepared_count == 2:
                    both_prepared.set()
                await both_prepared.wait()
                return anchor

            context._workflow_anchor = gated_anchor  # type: ignore[method-assign]

        gate_anchor(first_attempt)
        gate_anchor(second_attempt)
        outcomes = await asyncio.gather(
            step(first_attempt, agent="assistant", step_id="s1", prompt="go"),
            step(second_attempt, agent="assistant", step_id="s1", prompt="go"),
            return_exceptions=True,
        )
        results = [outcome for outcome in outcomes if isinstance(outcome, StepResult)]
        superseded = [
            outcome for outcome in outcomes if isinstance(outcome, WorkflowSupersededError)
        ]
        assert len(results) == 1
        assert len(superseded) == 1
        assert len(provider.requests) == 1

    asyncio.run(run())
    assert len(provider.requests) == 1


def test_step_reservation_cannot_commit_after_attempt_takeover() -> None:
    class DelayedReservationStore(InMemorySessionStore):
        stale_entered = asyncio.Event()
        release_stale = asyncio.Event()

        async def append_workflow_step_started(
            self,
            session_id,
            event,
            *,
            workflow_name,
            attempt_id,
        ):
            if attempt_id == stale_attempt_id:
                self.stale_entered.set()
                await self.release_stale.wait()
            return await super().append_workflow_step_started(
                session_id,
                event,
                workflow_name=workflow_name,
                attempt_id=attempt_id,
            )

    store = DelayedReservationStore()
    app = CayuApp(session_store=store, enable_logging=False)
    workflow = TinyWorkflow(app)
    stale = workflow.context("wf-attempt-reservation-race")
    stale_attempt_id = stale.attempt_id

    async def run() -> None:
        await stale.start()
        stale_task = asyncio.create_task(
            stale.journal.append_step_started(
                stale.event(
                    EventType.WORKFLOW_STEP_STARTED,
                    payload={"step_id": "step-a"},
                ),
                attempt_id=stale.attempt_id,
            )
        )
        await store.stale_entered.wait()

        current = workflow.context("wf-attempt-reservation-race")
        await current.start()
        current_reserved = await current.journal.append_step_started(
            current.event(
                EventType.WORKFLOW_STEP_STARTED,
                payload={"step_id": "step-a"},
            ),
            attempt_id=current.attempt_id,
        )
        store.release_stale.set()

        assert current_reserved is True
        assert await stale_task is False
        records = await store.query_events(
            EventQuery(
                session_id=stale.session_id,
                event_type=EventType.WORKFLOW_STEP_STARTED,
                limit=10,
            )
        )
        assert len(records) == 1
        assert records[0].event.payload["attempt_id"] == current.attempt_id

    asyncio.run(run())


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
    transcript = asyncio.run(app.session_store.load_transcript(result.session_id))
    assert secret not in str([message.model_dump(mode="json") for message in transcript])


def test_runtime_owned_workflow_linkage_survives_short_secret_collisions(monkeypatch):
    app = CayuApp(
        secret_redactor=SecretRedactor(("8", "-")),
        enable_logging=False,
    )
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({}),
            ]
        ],
        name="scripted",
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="bot", model="model"))
    monkeypatch.setattr(
        "cayu.workflows.workflow.uuid4",
        lambda: SimpleNamespace(hex="8" * 32),
    )
    monkeypatch.setattr(
        "cayu.runtime._child_session_identity.uuid4",
        lambda: SimpleNamespace(hex="8" * 32),
    )
    ctx = TinyWorkflow(app).context("workflow")

    result = asyncio.run(step(ctx, agent="bot", step_id="step", prompt="go"))

    assert result.session_id.startswith("cayu-child:v1:workflow-step:")
    assert len(result.session_id.removeprefix("cayu-child:v1:workflow-step:")) == 64
    records = asyncio.run(app.session_store.query_events(EventQuery(session_id="workflow")))
    workflow_events = [record.event for record in records]
    assert any(event.type == WORKFLOW_ATTEMPT_EVENT_TYPE for event in workflow_events)
    assert any(event.type == EventType.WORKFLOW_STEP_STARTED for event in workflow_events)
    assert any(event.type == EventType.WORKFLOW_STEP_COMPLETED for event in workflow_events)
    assert len(provider.requests) == 1


def test_caller_owned_workflow_child_linkage_remains_secret_rejected():
    app = CayuApp(
        secret_redactor=SecretRedactor("secret"),
        enable_logging=False,
    )
    provider = ScriptedModelProvider([], name="scripted")
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="bot", model="model"))
    ctx = TinyWorkflow(app).context("workflow")

    with pytest.raises(ValueError, match="child_session_id contains a workload secret"):
        asyncio.run(
            step(
                ctx,
                agent="bot",
                step_id="step",
                prompt="go",
                session_id="caller-secret-child",
            )
        )
    assert provider.requests == []


def test_generated_workflow_child_collision_fails_closed_without_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, provider = _scripted_assistant_app(
        [
            ModelStreamEvent.text_delta("foreign result"),
            ModelStreamEvent.completed({}),
        ]
    )

    async def run() -> tuple[object, object, list[Message], list[Message], list[Event]]:
        async for _event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="collision",
                messages=[Message.text("user", "foreign")],
            )
        ):
            pass
        before = await app.session_store.load("collision")
        before_transcript = await app.session_store.load_transcript("collision")
        monkeypatch.setattr(
            "cayu.workflows.workflow.generate_child_session_id",
            lambda **_kwargs: "collision",
        )

        with pytest.raises(StepError, match="generated child session identity collision"):
            await step(
                TinyWorkflow(app).context("different-parent"),
                agent="assistant",
                step_id="new-step",
                prompt="go",
            )

        after = await app.session_store.load("collision")
        after_transcript = await app.session_store.load_transcript("collision")
        records = await app.session_store.query_events(
            EventQuery(session_id="different-parent", limit=20)
        )
        return (
            before,
            after,
            before_transcript,
            after_transcript,
            [record.event for record in records],
        )

    before, after, before_transcript, after_transcript, workflow_events = asyncio.run(run())

    assert before == after
    assert before_transcript == after_transcript
    assert len(provider.requests) == 1
    assert all(
        event.type not in {EventType.WORKFLOW_STEP_STARTED, EventType.WORKFLOW_STEP_COMPLETED}
        for event in workflow_events
    )


def test_caller_cannot_forge_workflow_session_create_claim() -> None:
    app, provider = _scripted_assistant_app(
        [ModelStreamEvent.text_delta("unexpected"), ModelStreamEvent.completed({})]
    )

    async def run() -> object:
        with pytest.raises(ValueError, match="Session create claim metadata is runtime-owned"):
            async for _event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="forged-workflow-child",
                    messages=[Message.text("user", "go")],
                    metadata={
                        "cayu:session_create_claim": {
                            "record_type": "cayu.session-create-claim",
                            "schema_version": 1,
                            "claim_id": "forged",
                            "request_sha256": "0" * 64,
                        }
                    },
                )
            ):
                pass
        return await app.session_store.load("forged-workflow-child")

    assert asyncio.run(run()) is None
    assert provider.requests == []


def test_generated_workflow_child_collision_during_create_never_attaches_or_reads_foreign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, provider = _scripted_assistant_app(
        [ModelStreamEvent.text_delta("unexpected"), ModelStreamEvent.completed({})]
    )
    store = app.session_store
    original_create = store.create
    original_load = store.load
    foreign_created = False
    foreign_reads = 0

    async def create_with_foreign_winner(request: RunRequest, **kwargs: Any):
        nonlocal foreign_created
        if request.session_id == "collision" and not foreign_created:
            foreign_created = True
            await original_create(
                request.model_copy(
                    update={
                        "parent_session_id": None,
                        "causal_budget_id": "foreign-budget",
                        "messages": [Message.text("user", "foreign")],
                        "metadata": {"owner": "foreign"},
                    },
                    deep=True,
                ),
                identity=kwargs["identity"],
            )
        return await original_create(request, **kwargs)

    async def load_with_foreign_read_probe(session_id: str):
        nonlocal foreign_reads
        if foreign_created and session_id == "collision":
            foreign_reads += 1
        return await original_load(session_id)

    async def run() -> tuple[object, list[Event]]:
        monkeypatch.setattr(store, "create", create_with_foreign_winner)
        monkeypatch.setattr(store, "load", load_with_foreign_read_probe)
        monkeypatch.setattr(
            "cayu.workflows.workflow.generate_child_session_id",
            lambda **_kwargs: "collision",
        )

        with pytest.raises(StepError, match="Session already exists: collision"):
            await step(
                TinyWorkflow(app).context("different-parent"),
                agent="assistant",
                step_id="new-step",
                prompt="go",
            )

        foreign = await original_load("collision")
        records = await store.query_events(EventQuery(session_id="different-parent", limit=20))
        return foreign, [record.event for record in records]

    foreign, workflow_events = asyncio.run(run())

    assert foreign is not None
    assert foreign.metadata["owner"] == "foreign"
    assert foreign.parent_session_id is None
    assert foreign.causal_budget_id == "foreign-budget"
    assert foreign_reads == 1
    assert provider.requests == []
    assert all(
        event.type not in {EventType.WORKFLOW_STEP_STARTED, EventType.WORKFLOW_STEP_COMPLETED}
        for event in workflow_events
    )


def test_generated_workflow_child_cancelled_after_create_is_attached_and_recovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, provider = _scripted_assistant_app(
        [ModelStreamEvent.text_delta("unexpected"), ModelStreamEvent.completed({})]
    )
    store = app.session_store
    original_create = store.create
    create_committed = asyncio.Event()
    release_create = asyncio.Event()

    async def create_then_lose_acknowledgement(request: RunRequest, **kwargs: Any):
        created = await original_create(request, **kwargs)
        if request.session_id == "generated-after-create-cancellation":
            await store.update_metadata(request.session_id, {"observer": "legitimate"})
            await store.update_labels(request.session_id, {"phase": "created"})
            create_committed.set()
            await release_create.wait()
        return created

    async def run() -> tuple[object, list[Event], list[Event], str | None]:
        monkeypatch.setattr(store, "create", create_then_lose_acknowledgement)
        monkeypatch.setattr(
            "cayu.workflows.workflow.generate_child_session_id",
            lambda **_kwargs: "generated-after-create-cancellation",
        )
        workflow = TinyWorkflow(app)
        ctx = workflow.context("workflow-parent")
        task = asyncio.create_task(
            step(
                ctx,
                agent="assistant",
                step_id="cancelled-step",
                prompt="go",
            )
        )
        await asyncio.wait_for(create_committed.wait(), timeout=1)
        task.cancel("cancel after durable child create")
        with pytest.raises(asyncio.CancelledError, match="cancel after durable child create"):
            await task
        release_create.set()

        child = await store.load("generated-after-create-cancellation")
        child_events = await store.load_events("generated-after-create-cancellation")
        workflow_records = await store.query_events(
            EventQuery(session_id="workflow-parent", limit=20)
        )
        with pytest.raises(StepError) as replay:
            await step(
                workflow.context("workflow-parent"),
                agent="assistant",
                step_id="cancelled-step",
                prompt="go again",
            )
        return (
            child,
            child_events,
            [record.event for record in workflow_records],
            replay.value.session_id,
        )

    child, child_events, workflow_events, replay_session_id = asyncio.run(run())

    assert child is not None
    assert child.status == SessionStatus.INTERRUPTED
    assert child.metadata["observer"] == "legitimate"
    assert child.labels == {"phase": "created"}
    assert replay_session_id == "generated-after-create-cancellation"
    assert len(provider.requests) == 0
    started = [event for event in workflow_events if event.type == EventType.WORKFLOW_STEP_STARTED]
    assert len(started) == 1
    assert started[0].payload["child_session_id"] == "generated-after-create-cancellation"
    assert all(event.type != EventType.WORKFLOW_STEP_COMPLETED for event in workflow_events)
    interrupted = [event for event in child_events if event.type == EventType.SESSION_INTERRUPTED]
    assert len(interrupted) == 1


def test_generated_workflow_child_create_acknowledgement_loss_reuses_owned_identity() -> None:
    app, provider = _scripted_assistant_app(
        [ModelStreamEvent.text_delta("unexpected"), ModelStreamEvent.completed({})]
    )
    store = app.session_store
    original_create = store.create
    created_child_id: str | None = None

    async def create_then_lose_acknowledgement(request: RunRequest, **kwargs: Any):
        nonlocal created_child_id
        created = await original_create(request, **kwargs)
        if request.session_id is not None and request.session_id.startswith(
            "cayu-child:v1:workflow-step:"
        ):
            created_child_id = request.session_id
            raise OSError("ack lost after durable workflow child create")
        return created

    async def run() -> tuple[str, object, list[Event]]:
        store.create = create_then_lose_acknowledgement  # type: ignore[method-assign]
        workflow = TinyWorkflow(app)
        with pytest.raises(StepError, match="ack lost after durable workflow child create"):
            await step(
                workflow.context("workflow-parent"),
                agent="assistant",
                step_id="durable-step",
                prompt="go",
            )
        store.create = original_create  # type: ignore[method-assign]
        assert created_child_id is not None
        with pytest.raises(StepError) as replay:
            await step(
                workflow.context("workflow-parent"),
                agent="assistant",
                step_id="durable-step",
                prompt="go",
            )
        child = await store.load(created_child_id)
        records = await store.query_events(EventQuery(session_id="workflow-parent", limit=20))
        assert replay.value.session_id == created_child_id
        return created_child_id, child, [record.event for record in records]

    child_session_id, child, workflow_events = asyncio.run(run())

    assert child is not None
    assert child.id == child_session_id
    assert child.status == SessionStatus.INTERRUPTED
    assert provider.requests == []
    started_ids = [
        event.payload["child_session_id"]
        for event in workflow_events
        if event.type == EventType.WORKFLOW_STEP_STARTED
    ]
    assert started_ids == [child_session_id]


def test_stale_workflow_attempt_does_not_recover_exact_create_winner() -> None:
    app, provider = _scripted_assistant_app(
        [ModelStreamEvent.text_delta("unexpected"), ModelStreamEvent.completed({})]
    )
    store = app.session_store
    original_create = store.create
    created_child_id: str | None = None

    async def create_then_lose_acknowledgement(request: RunRequest, **kwargs: Any):
        nonlocal created_child_id
        created = await original_create(request, **kwargs)
        if request.session_id is not None and request.session_id.startswith(
            "cayu-child:v1:workflow-step:"
        ):
            created_child_id = request.session_id
            raise OSError("ack lost after durable workflow child create")
        return created

    async def run() -> tuple[object, list[Event]]:
        store.create = create_then_lose_acknowledgement  # type: ignore[method-assign]
        ctx = TinyWorkflow(app).context("workflow-parent")

        async def lose_reservation(_event: Event, *, attempt_id: str) -> bool:
            del attempt_id
            return False

        ctx.journal.append_step_started = lose_reservation  # type: ignore[method-assign]
        with pytest.raises(StepError) as failed:
            await step(
                ctx,
                agent="assistant",
                step_id="durable-step",
                prompt="go",
            )
        assert any(
            "WorkflowSupersededError" in note
            for note in getattr(failed.value.__cause__, "__notes__", ())
        )
        assert created_child_id is not None
        child = await store.load(created_child_id)
        events = await store.load_events(created_child_id)
        return child, events

    child, child_events = asyncio.run(run())

    assert child is not None
    assert child.status == SessionStatus.RUNNING
    assert all(event.type != EventType.SESSION_INTERRUPTED for event in child_events)
    assert provider.requests == []


def test_generated_workflow_child_identity_survives_process_reconstruction() -> None:
    class SimulatedProcessLoss(BaseException):
        pass

    app, provider = _scripted_assistant_app(
        [ModelStreamEvent.text_delta("unexpected"), ModelStreamEvent.completed({})]
    )
    store = app.session_store
    original_create = store.create
    created_child_id: str | None = None

    async def create_then_process_exits(request: RunRequest, **kwargs: Any):
        nonlocal created_child_id
        created = await original_create(request, **kwargs)
        if request.session_id is not None and request.session_id.startswith(
            "cayu-child:v1:workflow-step:"
        ):
            created_child_id = request.session_id
            raise SimulatedProcessLoss
        return created

    async def run() -> tuple[str, object, list[Event]]:
        store.create = create_then_process_exits  # type: ignore[method-assign]
        with pytest.raises(SimulatedProcessLoss):
            await step(
                TinyWorkflow(app).context("workflow-parent"),
                agent="assistant",
                step_id="durable-step",
                prompt="go",
            )
        store.create = original_create  # type: ignore[method-assign]
        assert created_child_id is not None
        unacknowledged = await store.load(created_child_id)
        assert unacknowledged is not None
        assert unacknowledged.status == SessionStatus.RUNNING
        with pytest.raises(StepError, match="generated child session identity collision"):
            await step(
                TinyWorkflow(app).context("workflow-parent"),
                agent="assistant",
                step_id="durable-step",
                prompt="contradictory retry input",
            )
        still_unacknowledged = await store.load(created_child_id)
        assert still_unacknowledged is not None
        assert still_unacknowledged.status == SessionStatus.RUNNING
        with pytest.raises(StepError) as reconstructed:
            await step(
                TinyWorkflow(app).context("workflow-parent"),
                agent="assistant",
                step_id="durable-step",
                prompt="go",
            )
        child = await store.load(created_child_id)
        records = await store.query_events(EventQuery(session_id="workflow-parent", limit=20))
        assert reconstructed.value.session_id == created_child_id
        return created_child_id, child, [record.event for record in records]

    child_session_id, child, workflow_events = asyncio.run(run())

    assert child is not None
    assert child.id == child_session_id
    assert child.status == SessionStatus.INTERRUPTED
    assert provider.requests == []
    started_ids = [
        event.payload["child_session_id"]
        for event in workflow_events
        if event.type == EventType.WORKFLOW_STEP_STARTED
    ]
    assert started_ids == [child_session_id]


def test_generated_workflow_child_reconstruction_rejects_contradictory_invocation() -> None:
    class SimulatedProcessLoss(BaseException):
        pass

    app, provider = _scripted_assistant_app(
        [ModelStreamEvent.text_delta("unexpected"), ModelStreamEvent.completed({})]
    )
    store = app.session_store
    assert isinstance(store, InMemorySessionStore)
    original_create = store.create
    created_child_id: str | None = None

    async def create_then_process_exits(request: RunRequest, **kwargs: Any):
        nonlocal created_child_id
        created = await original_create(request, **kwargs)
        if request.session_id is not None and request.session_id.startswith(
            "cayu-child:v1:workflow-step:"
        ):
            created_child_id = request.session_id
            raise SimulatedProcessLoss
        return created

    async def run() -> object:
        store.create = create_then_process_exits  # type: ignore[method-assign]
        with pytest.raises(SimulatedProcessLoss):
            await step(
                TinyWorkflow(app).context("workflow-parent"),
                agent="assistant",
                step_id="durable-step",
                prompt="go",
            )
        store.create = original_create  # type: ignore[method-assign]
        assert created_child_id is not None
        child = await store.load(created_child_id)
        assert child is not None
        unrelated = fixture_session_invocation("unrelated-root")
        contradictory_origin = child.invocation.origin.model_copy(
            update={
                "trust": InvocationOriginTrust.HOST_ASSERTED,
                "subject": "unrelated-user",
            }
        )
        contradictory_invocations = (
            child.invocation.model_copy(update={"origin": contradictory_origin}),
            child.invocation.model_copy(
                update={"root_invocation_id": unrelated.root_invocation_id}
            ),
            child.invocation.model_copy(update={"root_session_id": unrelated.root_session_id}),
            child.invocation.model_copy(update={"source": SessionExecutionSource.SDK_RUN}),
        )
        for invocation in contradictory_invocations:
            store._sessions[created_child_id] = child.model_copy(update={"invocation": invocation})
            with pytest.raises(StepError, match="generated child session identity collision"):
                await step(
                    TinyWorkflow(app).context("workflow-parent"),
                    agent="assistant",
                    step_id="durable-step",
                    prompt="go",
                )
        return await store.load(created_child_id)

    contradictory = asyncio.run(run())

    assert contradictory is not None
    assert contradictory.invocation.source is SessionExecutionSource.SDK_RUN
    assert contradictory.status == SessionStatus.RUNNING
    assert provider.requests == []


def test_generated_workflow_child_cancellation_does_not_settle_foreign_create_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, provider = _scripted_assistant_app(
        [ModelStreamEvent.text_delta("unexpected"), ModelStreamEvent.completed({})]
    )
    store = app.session_store
    original_create = store.create
    foreign_create_committed = asyncio.Event()
    release_create = asyncio.Event()

    async def create_with_foreign_winner_then_block(request: RunRequest, **kwargs: Any):
        if request.session_id == "foreign-create-winner":
            foreign_messages = [Message.text("user", "foreign")]
            await original_create(
                request.model_copy(
                    update={
                        "messages": foreign_messages,
                    },
                    deep=True,
                ),
                identity=kwargs["identity"],
                interaction_started_event=kwargs["interaction_started_event"],
                interaction_source_messages=foreign_messages,
            )
            foreign_create_committed.set()
            await release_create.wait()
        return await original_create(request, **kwargs)

    async def run() -> tuple[object, list[Event], list[Event]]:
        monkeypatch.setattr(store, "create", create_with_foreign_winner_then_block)
        monkeypatch.setattr(
            "cayu.workflows.workflow.generate_child_session_id",
            lambda **_kwargs: "foreign-create-winner",
        )
        task = asyncio.create_task(
            step(
                TinyWorkflow(app).context("workflow-parent"),
                agent="assistant",
                step_id="cancelled-step",
                prompt="go",
            )
        )
        await asyncio.wait_for(foreign_create_committed.wait(), timeout=1)
        task.cancel("cancel after foreign child create")
        with pytest.raises(asyncio.CancelledError, match="cancel after foreign child create"):
            await task
        release_create.set()
        foreign = await original_create.__self__.load("foreign-create-winner")
        foreign_events = await store.load_events("foreign-create-winner")
        workflow_records = await store.query_events(
            EventQuery(session_id="workflow-parent", limit=20)
        )
        return foreign, foreign_events, [record.event for record in workflow_records]

    foreign, foreign_events, workflow_events = asyncio.run(run())

    assert foreign is not None
    assert foreign.status == SessionStatus.RUNNING
    assert foreign.parent_session_id == "workflow-parent"
    assert all(event.type != EventType.SESSION_INTERRUPTED for event in foreign_events)
    assert provider.requests == []
    assert all(
        event.type not in {EventType.WORKFLOW_STEP_STARTED, EventType.WORKFLOW_STEP_COMPLETED}
        for event in workflow_events
    )


def test_workflow_structured_output_handoff_preserves_json_null_and_lifecycle():
    handoff = WorkflowStructuredOutputHandoff()

    handoff.prepare("child")
    with pytest.raises(RuntimeError, match="capture is already active"):
        handoff.prepare("child")
    handoff.record("child", None)
    assert handoff.take("child") == (True, None)
    assert handoff.take("child") == (False, None)

    handoff.prepare("child")
    handoff.record("child", {"attempt": 1})
    handoff.record("child", {"attempt": 2})
    assert handoff.take("child") == (True, {"attempt": 2})

    handoff.prepare("child")
    handoff.discard("child")
    assert handoff.take("child") == (False, None)


def test_step_structured_output_replay_fails_closed_on_redaction_marker():
    app = CayuApp(enable_logging=False)
    store = app.session_store
    child_session_id = "wf-structured-output-replay-child"
    schema = {
        "type": "object",
        "properties": {"token": {"type": "string"}},
        "required": ["token"],
        "additionalProperties": False,
    }
    ctx = TinyWorkflow(app).context("wf-structured-output-replay")

    async def run():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=child_session_id,
                messages=[Message.text("user", "already completed")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_event(
            child_session_id,
            Event(
                type=EventType.STRUCTURED_OUTPUT_VALIDATED,
                session_id=child_session_id,
                payload={"output": {"token": REDACTED_SECRET}},
            ),
        )
        await store.update_status(child_session_id, SessionStatus.COMPLETED)
        return await step(
            ctx,
            agent="assistant",
            step_id="structured",
            prompt="go",
            schema=schema,
            session_id=child_session_id,
        )

    with pytest.raises(StepError, match="cannot be reconstructed safely"):
        asyncio.run(run())


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
