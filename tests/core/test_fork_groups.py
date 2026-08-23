from __future__ import annotations

import asyncio
import json
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from cayu import (
    AgentSpec,
    BudgetLimit,
    BudgetReservation,
    CayuApp,
    Event,
    EventType,
    ExecutionProfileAdoptionIntent,
    ExecutionProfileMismatchError,
    ForkGroupArtifactReference,
    ForkGroupAttemptKind,
    ForkGroupBranchSpec,
    ForkGroupBranchStatus,
    ForkGroupCheckpointSelector,
    ForkGroupConflict,
    ForkGroupDisposition,
    ForkGroupEvaluatorSpec,
    ForkGroupExecutionMode,
    ForkGroupFailureCode,
    ForkGroupFailureMode,
    ForkGroupFailurePolicy,
    ForkGroupGate,
    ForkGroupGateDecision,
    ForkGroupGateRequest,
    ForkGroupGateResult,
    ForkGroupGateSelection,
    ForkGroupReplacementPlanner,
    ForkGroupReplacementPlannerRequest,
    ForkGroupReplacementPlannerSelection,
    ForkGroupReplacementSpec,
    ForkGroupRequest,
    ForkGroupState,
    ForkSessionRequest,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryResult,
    InMemoryBudgetLedger,
    InMemorySessionStore,
    InMemoryTaskStore,
    Message,
    ModelPrice,
    OpenAIWebSearch,
    PriceBook,
    ResolutionActor,
    ResolutionActorSource,
    RunRequest,
    SQLiteSessionStore,
    SQLiteTaskStore,
    StructuredOutputSpec,
    TaskQuery,
    TaskStatus,
    TaskStoreDispatcher,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
    session_fork_profile_relationship,
)
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.messages import TextPart
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import EventQuery, SessionStatus, SessionUsageSummary
from cayu.runtime import _session_control as session_control_runtime
from cayu.runtime import fork_groups as fork_group_runtime
from cayu.runtime.execution_profiles import execution_profile_baseline_from_session_metadata
from cayu.vaults import SecretRedactor


class _ForbiddenEvaluatorTool(Tool):
    spec = ToolSpec(
        name="forbidden_evaluator_tool",
        description="Must never be exposed to the fork-group evaluator.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        del ctx, args
        raise AssertionError("Fork-group evaluator inherited application tool authority.")


class _ConfiguredGate(ForkGroupGate):
    def __init__(self, *, failed_branch_id: str | None = None) -> None:
        self.failed_branch_id = failed_branch_id
        self.requests: list[ForkGroupGateRequest] = []

    @property
    def identity(self) -> str:
        return "tests.configured-gate.v1"

    async def evaluate(self, request: ForkGroupGateRequest) -> ForkGroupGateDecision:
        self.requests.append(request)
        passed = request.branch.branch_id != self.failed_branch_id
        return ForkGroupGateDecision(
            passed=passed,
            summary="deterministic tests passed" if passed else "deterministic tests failed",
        )


class _AttemptGate(ForkGroupGate):
    def __init__(self, *rejected: tuple[str, int]) -> None:
        self.rejected = frozenset(rejected)
        self.requests: list[ForkGroupGateRequest] = []

    @property
    def identity(self) -> str:
        return "tests.attempt-gate.v1"

    async def evaluate(self, request: ForkGroupGateRequest) -> ForkGroupGateDecision:
        self.requests.append(request)
        passed = (request.branch.branch_id, request.branch.attempt_index) not in self.rejected
        return ForkGroupGateDecision(
            passed=passed,
            summary="attempt eligible" if passed else "attempt rejected",
        )


class _SecretFailingGate(ForkGroupGate):
    def __init__(self, secret: str) -> None:
        self.secret = secret

    @property
    def identity(self) -> str:
        return "tests.secret-failing-gate.v1"

    async def evaluate(self, request: ForkGroupGateRequest) -> ForkGroupGateDecision:
        del request
        raise RuntimeError(f"gate rejected {self.secret}")


class _ConfiguredReplacementPlanner(ForkGroupReplacementPlanner):
    def __init__(
        self,
        *,
        message_suffix: str = "",
        artifact_references: tuple[ForkGroupArtifactReference, ...] = (),
    ) -> None:
        self.requests: list[ForkGroupReplacementPlannerRequest] = []
        self.message_suffix = message_suffix
        self.artifact_references = artifact_references

    @property
    def identity(self) -> str:
        return "tests.configured-replacement-planner.v1"

    async def plan(
        self,
        request: ForkGroupReplacementPlannerRequest,
    ) -> ForkGroupReplacementSpec:
        self.requests.append(request)
        return ForkGroupReplacementSpec(
            messages=(
                Message.text(
                    "user",
                    (
                        f"replacement {request.branch_id} attempt {request.attempt_index}"
                        f"{self.message_suffix}"
                    ),
                ),
            ),
            structured_output=_candidate_output(f"replacement-{request.branch_id}"),
            artifact_references=self.artifact_references,
        )


class _SecretFailingReplacementPlanner(ForkGroupReplacementPlanner):
    def __init__(self, secret: str) -> None:
        self.secret = secret

    @property
    def identity(self) -> str:
        return "tests.secret-failing-replacement-planner.v1"

    async def plan(
        self,
        request: ForkGroupReplacementPlannerRequest,
    ) -> ForkGroupReplacementSpec:
        del request
        raise RuntimeError(f"replacement planning rejected {self.secret}")


class _MissingAgentReplacementPlanner(ForkGroupReplacementPlanner):
    @property
    def identity(self) -> str:
        return "tests.missing-agent-replacement-planner.v1"

    async def plan(
        self,
        request: ForkGroupReplacementPlannerRequest,
    ) -> ForkGroupReplacementSpec:
        del request
        return ForkGroupReplacementSpec(
            agent_name="missing-replacement-agent",
            profile_adoption=ExecutionProfileAdoptionIntent(
                idempotency_key="missing-replacement-agent-profile",
                reason="Exercise rejected replacement profile preparation.",
                requested_by=ResolutionActor(
                    subject="tests",
                    source=ResolutionActorSource.REQUEST,
                ),
            ),
            messages=(Message.text("user", "replacement with missing agent"),),
            structured_output=_candidate_output("missing-agent-replacement"),
        )


class _MixedProfileReplacementPlanner(_MissingAgentReplacementPlanner):
    @property
    def identity(self) -> str:
        return "tests.mixed-profile-replacement-planner.v1"

    async def plan(
        self,
        request: ForkGroupReplacementPlannerRequest,
    ) -> ForkGroupReplacementSpec:
        if request.branch_id == "beta":
            return await super().plan(request)
        return ForkGroupReplacementSpec(
            messages=(Message.text("user", f"replacement {request.branch_id}"),),
            structured_output=_candidate_output(f"replacement-{request.branch_id}"),
        )


class _ForkGroupProvider(ModelProvider):
    name = "fork-group-fake"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:fork-group-provider",
            behavior_version="1",
            implementation_version="1",
        )

    def __init__(
        self,
        *,
        fail_branch: str | None = None,
        fail_evaluator: bool = False,
        invalid_judgment: str | None = None,
        replacement_failures: int = 0,
        branch_callback: Callable[[], Awaitable[None]] | None = None,
        replacement_callback: Callable[[], Awaitable[None]] | None = None,
        evaluator_callback: Callable[[], Awaitable[None]] | None = None,
        branch_failure_message: str | None = None,
        branch_delay: float = 0,
    ) -> None:
        self.requests: list[ModelRequest] = []
        self.evaluator_tools: tuple[str, ...] | None = None
        self.evaluator_hosted_tools: tuple[OpenAIWebSearch, ...] | None = None
        self.evaluator_evidence: dict[str, Any] | None = None
        self.evaluator_calls = 0
        self.fail_branch = fail_branch
        self.fail_evaluator = fail_evaluator
        self.invalid_judgment = invalid_judgment
        self.replacement_failures = replacement_failures
        self.replacement_calls = 0
        self.branch_callback = branch_callback
        self.replacement_callback = replacement_callback
        self.evaluator_callback = evaluator_callback
        self.branch_failure_message = branch_failure_message
        self.branch_delay = branch_delay
        self.callback_called = False
        self.replacement_callback_called = False
        self.active_branches = 0
        self.max_active_branches = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        user_text = "\n".join(
            part.text
            for message in request.messages
            if message.role == "user"
            for part in message.content
            if type(part) is TextPart
        )
        structured_tool = next(
            (tool["name"] for tool in request.tools if tool["name"].startswith("__cayu")),
            None,
        )
        if "cayu.fork-group-evidence.v1" in user_text or "cayu.fork-group-evidence.v2" in user_text:
            self.evaluator_calls += 1
            if self.evaluator_callback is not None:
                await self.evaluator_callback()
            self.evaluator_tools = tuple(tool["name"] for tool in request.tools)
            self.evaluator_hosted_tools = request.hosted_tools
            if self.fail_evaluator:
                yield ModelStreamEvent.error("evaluator failed")
                return
            evidence = json.loads(user_text)
            self.evaluator_evidence = evidence
            branch_identities = [
                (branch["branch_id"], branch.get("attempt_id")) for branch in evidence["branches"]
            ]

            def disposition(
                branch_id: str,
                attempt_id: str | None,
                value: str,
                reason: str,
            ) -> dict[str, str]:
                del attempt_id
                item = {
                    "branch_id": branch_id,
                    "disposition": value,
                    "reason": reason,
                }
                return item

            if self.invalid_judgment == "select-excluded":
                excluded = evidence["excluded_attempts"][0]
                output = {
                    "dispositions": [
                        disposition(
                            excluded["branch_id"],
                            excluded["attempt_id"],
                            "selected",
                            "invalid excluded selection",
                        ),
                        disposition(
                            branch_identities[0][0],
                            branch_identities[0][1],
                            "rejected",
                            "invalid excluded selection",
                        ),
                    ]
                }
            elif self.invalid_judgment == "duplicate":
                output = {
                    "dispositions": [
                        disposition(
                            branch_identities[0][0],
                            branch_identities[0][1],
                            "selected",
                            "duplicate judgment",
                        ),
                        disposition(
                            branch_identities[0][0],
                            branch_identities[0][1],
                            "rejected",
                            "duplicate judgment",
                        ),
                    ]
                }
            elif self.invalid_judgment == "multiple-selected":
                output = {
                    "dispositions": [
                        disposition(
                            branch_id,
                            attempt_id,
                            "selected",
                            "invalid judgment",
                        )
                        for branch_id, attempt_id in branch_identities
                    ]
                }
            else:
                output = {
                    "dispositions": [
                        disposition(
                            branch_id,
                            attempt_id,
                            "selected" if index == 0 else "rejected",
                            "deterministic test judgment",
                        )
                        for index, (branch_id, attempt_id) in enumerate(branch_identities)
                    ]
                }
        elif "replacement beta" in user_text:
            self.replacement_calls += 1
            if self.replacement_calls <= self.replacement_failures:
                yield ModelStreamEvent.error("replacement beta failed")
                return
            await self._enter_branch(replacement=True)
            try:
                output = {"candidate": "beta", "score": 3}
            finally:
                self.active_branches -= 1
        elif "candidate alpha" in user_text:
            if self.fail_branch == "alpha":
                yield ModelStreamEvent.error(self.branch_failure_message or "alpha failed")
                return
            await self._enter_branch()
            try:
                output = {"candidate": "alpha", "score": 2}
            finally:
                self.active_branches -= 1
        elif "candidate beta" in user_text:
            if self.fail_branch == "beta":
                yield ModelStreamEvent.error(self.branch_failure_message or "beta failed")
                return
            await self._enter_branch()
            try:
                output = {"candidate": "beta", "score": 1}
            finally:
                self.active_branches -= 1
        else:
            yield ModelStreamEvent.text_delta("source ready")
            yield ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                }
            )
            return
        if structured_tool is None:
            yield ModelStreamEvent.text_delta(json.dumps(output, sort_keys=True))
            yield ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 7, "output_tokens": 3},
                }
            )
            return
        yield ModelStreamEvent.tool_call(
            id=f"call-{len(self.requests)}",
            name=structured_tool,
            arguments={"output": output},
        )
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "tool_calls",
                "usage": {"input_tokens": 7, "output_tokens": 3},
            }
        )

    async def _enter_branch(self, *, replacement: bool = False) -> None:
        self.active_branches += 1
        self.max_active_branches = max(self.max_active_branches, self.active_branches)
        if (
            replacement
            and self.replacement_callback is not None
            and not self.replacement_callback_called
        ):
            self.replacement_callback_called = True
            await self.replacement_callback()
        elif self.branch_callback is not None and not self.callback_called:
            self.callback_called = True
            await self.branch_callback()
        if self.branch_delay:
            await asyncio.sleep(self.branch_delay)


def _candidate_output(name: str) -> StructuredOutputSpec:
    return StructuredOutputSpec(
        name=name,
        json_schema={
            "type": "object",
            "properties": {
                "candidate": {"type": "string", "enum": ["alpha", "beta"]},
                "score": {"type": "integer"},
            },
            "required": ["candidate", "score"],
            "additionalProperties": False,
        },
    )


async def _source(app: CayuApp) -> None:
    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="source",
                session_id="fork-group-source",
                causal_budget_id="fork-group-budget",
                messages=[Message.text("user", "prepare source")],
            )
        )
    ]
    assert events[-1].type == "session.completed"


def _request(
    *,
    group_id: str = "group-success",
    max_parallelism: int = 2,
    gates: tuple[ForkGroupGateSelection, ...] = (),
    extra_alpha: bool = False,
) -> ForkGroupRequest:
    branches = [
        ForkGroupBranchSpec(
            branch_id="alpha",
            session_id=f"{group_id}-alpha",
            messages=(Message.text("user", "candidate alpha"),),
            structured_output=_candidate_output("candidate-alpha"),
        ),
        ForkGroupBranchSpec(
            branch_id="beta",
            session_id=f"{group_id}-beta",
            messages=(Message.text("user", "candidate beta"),),
            structured_output=_candidate_output("candidate-beta"),
        ),
    ]
    if extra_alpha:
        branches.append(
            ForkGroupBranchSpec(
                branch_id="alpha-second",
                session_id=f"{group_id}-alpha-second",
                messages=(Message.text("user", "candidate alpha second"),),
                structured_output=_candidate_output("candidate-alpha-second"),
            )
        )
    return ForkGroupRequest(
        group_id=group_id,
        source_session_id="fork-group-source",
        source_checkpoint=ForkGroupCheckpointSelector(),
        causal_budget_id="fork-group-budget",
        max_parallelism=max_parallelism,
        branches=tuple(branches),
        gates=gates,
        evaluator=ForkGroupEvaluatorSpec(
            session_id=f"{group_id}-evaluator",
            agent_name="evaluator",
        ),
    )


def _app(provider: _ForkGroupProvider, *, session_store: Any | None = None) -> CayuApp:
    app = CayuApp(session_store=session_store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="source", model="fake-model"))
    app.register_agent(AgentSpec(name="evaluator", model="fake-model"))
    return app


def _task_app(
    provider: _ForkGroupProvider,
    *,
    session_store: Any,
    task_store: Any,
) -> tuple[CayuApp, TaskStoreDispatcher]:
    dispatcher = TaskStoreDispatcher(task_store)
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        dispatcher=dispatcher,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="source", model="fake-model"))
    app.register_agent(AgentSpec(name="evaluator", model="fake-model"))
    return app, dispatcher


class _FailOnceLoadTaskStore(InMemoryTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self) -> None:
        super().__init__()
        self.fail_next_load = False

    async def load_task(self, task_id: str) -> Any:
        if self.fail_next_load:
            self.fail_next_load = False
            raise ConnectionError("transient task-store read")
        return await super().load_task(task_id)


class _FailOnceLoadSessionStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail_session_id: str | None = None

    async def load(self, session_id: str) -> Any:
        if session_id == self.fail_session_id:
            self.fail_session_id = None
            raise ConnectionError("transient session-store read")
        return await super().load(session_id)


class _TerminalizingCancelTaskStore(InMemoryTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self) -> None:
        super().__init__()
        self.terminalize_before_cancel = False

    async def cancel_task(
        self,
        task_id: str,
        error: dict[str, Any] | None = None,
    ) -> Any:
        if self.terminalize_before_cancel:
            self.terminalize_before_cancel = False
            await self.fail_task(task_id, {"status": "failed"})
        return await super().cancel_task(task_id, error)


async def assert_task_backed_fork_group_store_conformance(
    session_store: Any,
    task_store: Any,
) -> None:
    """Exercise queue recovery and exact replay on one session/task store pair."""

    provider = _ForkGroupProvider(fail_branch="beta")

    def configured_app() -> tuple[CayuApp, TaskStoreDispatcher]:
        app, dispatcher = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        gate = app.register_fork_group_gate("tests", _ConfiguredGate())
        planner = app.register_fork_group_replacement_planner(
            "tests",
            _ConfiguredReplacementPlanner(),
        )
        assert gate == ForkGroupGateSelection(
            gate_id="tests",
            gate_identity="tests.configured-gate.v1",
        )
        assert planner == ForkGroupReplacementPlannerSelection(
            planner_id="tests",
            planner_identity="tests.configured-replacement-planner.v1",
        )
        return app, dispatcher

    app, dispatcher = configured_app()
    await _source(app)
    request = _request(
        group_id="group-task-store-conformance",
        max_parallelism=1,
        gates=(
            ForkGroupGateSelection(
                gate_id="tests",
                gate_identity="tests.configured-gate.v1",
            ),
        ),
    ).model_copy(
        update={
            "execution_mode": ForkGroupExecutionMode.TASK_DISPATCH,
            "failure_policy": ForkGroupFailurePolicy(
                mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                minimum_viable_branches=2,
                max_replacement_attempts=1,
                replacement_parallelism=1,
                replacement_planner=ForkGroupReplacementPlannerSelection(
                    planner_id="tests",
                    planner_identity="tests.configured-replacement-planner.v1",
                ),
            ),
        }
    )
    peer, _ = configured_app()
    first, second = await asyncio.gather(
        app.run_fork_group(request),
        peer.run_fork_group(request),
    )
    assert len(await task_store.list_tasks()) == 1
    result = first if first.dispatch_attempts else second
    first_task_id = result.dispatch_attempts[0].queue_task_id
    first_task = await task_store.load_task(first_task_id)
    assert first_task is not None
    assert first_task.type == dispatcher.fork_group_task_type
    assert result.dispatch_attempts[0].queue_task_type == dispatcher.fork_group_task_type
    assert (
        await task_store.claim_task(
            "pre-v3-worker",
            TaskQuery(type=dispatcher.task_type),
            lease_seconds=30,
        )
        is None
    )
    await task_store.pause_task(first_task_id, reason="conformance pause boundary")
    paused = await peer.inspect_fork_group(request.source_session_id, request.group_id)
    assert paused is not None
    assert paused.dispatch_attempts[0].task_status is TaskStatus.PAUSED
    assert len(await task_store.list_tasks()) == 1
    await task_store.resume_task(first_task_id)

    for worker_index in range(8):
        if result.state in {ForkGroupState.COMPLETED, ForkGroupState.FAILED}:
            break
        handle = await dispatcher.process_next(
            app,
            worker_id=f"store-worker-{worker_index}",
        )
        assert handle is not None
        result = await app.run_fork_group(request)
    assert result.state is ForkGroupState.COMPLETED
    assert [attempt.kind for attempt in result.dispatch_attempts] == [
        ForkGroupAttemptKind.BRANCH,
        ForkGroupAttemptKind.BRANCH,
        ForkGroupAttemptKind.REPLACEMENT,
        ForkGroupAttemptKind.EVALUATOR,
    ]
    assert provider.replacement_calls == 1
    assert provider.evaluator_calls == 1
    request_count = len(provider.requests)

    fresh_app, fresh_dispatcher = configured_app()
    replay = await fresh_app.run_fork_group(request)
    assert replay.state is ForkGroupState.COMPLETED and replay.replayed is True
    assert len(provider.requests) == request_count
    assert await fresh_dispatcher.process_next(fresh_app, worker_id="store-idle") is None


async def assert_viable_fork_group_store_conformance(session_store: Any) -> None:
    """Exercise replacement, exhaustion, concurrency, and replay on one store."""

    def configure(
        provider: _ForkGroupProvider,
    ) -> tuple[CayuApp, _ConfiguredReplacementPlanner]:
        app = _app(provider, session_store=session_store)
        app.register_fork_group_gate("tests", _ConfiguredGate())
        planner = _ConfiguredReplacementPlanner()
        app.register_fork_group_replacement_planner("tests", planner)
        return app, planner

    def viable_request(group_id: str, *, max_replacements: int) -> ForkGroupRequest:
        return _request(
            group_id=group_id,
            gates=(
                ForkGroupGateSelection(
                    gate_id="tests",
                    gate_identity="tests.configured-gate.v1",
                ),
            ),
        ).model_copy(
            update={
                "failure_policy": ForkGroupFailurePolicy(
                    mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                    minimum_viable_branches=2,
                    max_replacement_attempts=max_replacements,
                    replacement_parallelism=1,
                    replacement_planner=ForkGroupReplacementPlannerSelection(
                        planner_id="tests",
                        planner_identity="tests.configured-replacement-planner.v1",
                    ),
                )
            }
        )

    successful_provider = _ForkGroupProvider(
        fail_branch="beta",
        replacement_failures=1,
    )
    successful_app, successful_planner = configure(successful_provider)
    await _source(successful_app)
    successful_request = viable_request(
        "group-store-conformance-success",
        max_replacements=2,
    )
    successful = await successful_app.run_fork_group(successful_request)
    request_count = len(successful_provider.requests)
    replay = await successful_app.run_fork_group(successful_request)

    assert successful.state is ForkGroupState.COMPLETED
    assert len(successful.branches) == 4
    assert [item.attempt_index for item in successful.branches if item.branch_id == "beta"] == [
        0,
        1,
        2,
    ]
    assert len(successful_planner.requests) == 2
    assert replay.state is ForkGroupState.COMPLETED and replay.replayed is True
    assert replay.branches == successful.branches
    assert len(successful_provider.requests) == request_count

    exhausted_provider = _ForkGroupProvider(
        fail_branch="beta",
        replacement_failures=2,
    )
    exhausted_app, exhausted_planner = configure(exhausted_provider)
    exhausted = await exhausted_app.run_fork_group(
        viable_request("group-store-conformance-exhausted", max_replacements=2)
    )

    assert exhausted.state is ForkGroupState.FAILED
    assert exhausted.failure is not None
    assert exhausted.failure.code is ForkGroupFailureCode.REPLACEMENTS_EXHAUSTED
    assert len(exhausted.branches) == 4
    assert len(exhausted_planner.requests) == 2
    assert exhausted.dispositions == ()

    concurrent_provider = _ForkGroupProvider(fail_branch="beta", branch_delay=0.1)
    first_app, first_planner = configure(concurrent_provider)
    second_app, second_planner = configure(concurrent_provider)
    concurrent_request = viable_request(
        "group-store-conformance-concurrent",
        max_replacements=1,
    )
    await asyncio.gather(
        first_app.run_fork_group(concurrent_request),
        second_app.run_fork_group(concurrent_request),
    )
    converged = await first_app.run_fork_group(concurrent_request)

    assert converged.state is ForkGroupState.COMPLETED
    assert concurrent_provider.replacement_calls == 1
    assert concurrent_provider.evaluator_calls == 1
    assert len(first_planner.requests) + len(second_planner.requests) == 1


@pytest.mark.parametrize(
    "store_factory",
    [
        pytest.param(lambda _path: InMemorySessionStore(), id="in-memory"),
        pytest.param(lambda path: SQLiteSessionStore(path), id="sqlite"),
    ],
)
def test_viable_fork_group_store_conformance(tmp_path, store_factory) -> None:
    async def run() -> None:
        store = store_factory(tmp_path / "fork-groups.sqlite")
        try:
            await assert_viable_fork_group_store_conformance(store)
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                await close()

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["in-memory", "sqlite"])
def test_task_backed_fork_group_store_conformance(tmp_path, store_kind: str) -> None:
    async def run() -> None:
        if store_kind == "in-memory":
            session_store = InMemorySessionStore()
            task_store = InMemoryTaskStore()
        else:
            session_store = SQLiteSessionStore(tmp_path / "fork-group-sessions.sqlite")
            task_store = SQLiteTaskStore(tmp_path / "fork-group-tasks.sqlite")
        try:
            await assert_task_backed_fork_group_store_conformance(
                session_store,
                task_store,
            )
        finally:
            for store in (task_store, session_store):
                close = getattr(store, "close", None)
                if close is not None:
                    await close()

    asyncio.run(run())


def test_task_backed_transient_task_store_read_leaves_group_recoverable() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        session_store = InMemorySessionStore()
        task_store = _FailOnceLoadTaskStore()
        app, _ = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        await _source(app)
        request = _request(group_id="group-task-transient-read").model_copy(
            update={"execution_mode": ForkGroupExecutionMode.TASK_DISPATCH}
        )

        result = await app.run_fork_group(request)
        assert result.state is ForkGroupState.BRANCHES_RUNNING
        task_store.fail_next_load = True

        with pytest.raises(ConnectionError, match="task store.*unavailable"):
            await app.run_fork_group(request)

        inspected = await app.inspect_fork_group(
            request.source_session_id,
            request.group_id,
        )
        assert inspected is not None
        assert inspected.state is ForkGroupState.BRANCHES_RUNNING
        assert inspected.failure is None
        assert all(
            attempt.task_status is TaskStatus.PENDING for attempt in inspected.dispatch_attempts
        )

    asyncio.run(run())


def test_task_backed_transient_session_store_read_leaves_group_recoverable() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        session_store = _FailOnceLoadSessionStore()
        task_store = InMemoryTaskStore()
        app, dispatcher = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        await _source(app)
        request = _request(
            group_id="group-task-transient-session-read",
            max_parallelism=1,
        ).model_copy(update={"execution_mode": ForkGroupExecutionMode.TASK_DISPATCH})

        result = await app.run_fork_group(request)
        assert result.state is ForkGroupState.BRANCHES_RUNNING
        assert len(result.dispatch_attempts) == 1
        assert await dispatcher.process_next(app, worker_id="worker-alpha") is not None
        session_store.fail_session_id = result.dispatch_attempts[0].session_id

        with pytest.raises(ConnectionError, match="session store.*unavailable"):
            await app.run_fork_group(request)

        inspected = await app.inspect_fork_group(
            request.source_session_id,
            request.group_id,
        )
        assert inspected is not None
        assert inspected.state is ForkGroupState.BRANCHES_RUNNING
        assert inspected.failure is None
        assert inspected.branches == ()

        recovered = await app.run_fork_group(request)
        assert recovered.state is ForkGroupState.BRANCHES_RUNNING
        assert len(recovered.branches) == 1
        assert recovered.branches[0].status is ForkGroupBranchStatus.COMPLETED
        assert len(recovered.dispatch_attempts) == 2

    asyncio.run(run())


def test_task_backed_fork_group_authenticates_idempotency_linkage() -> None:
    async def run() -> None:
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        app, _ = _task_app(
            _ForkGroupProvider(),
            session_store=session_store,
            task_store=task_store,
        )
        await _source(app)
        request = _request(group_id="group-task-idempotency-link").model_copy(
            update={"execution_mode": ForkGroupExecutionMode.TASK_DISPATCH}
        )

        await app.run_fork_group(request)
        durable = await session_store.load_session_operation(
            request.source_session_id,
            fork_group_runtime._storage_key(request.group_id),
        )
        assert durable is not None
        record = fork_group_runtime._ForkGroupRecord.model_validate(durable)
        link = record.result.dispatch_attempts[0]
        conflicting_link = link.model_copy(
            update={"idempotency_key": "conflicting-idempotency-authority"}
        )
        conflicting_record = record.model_copy(
            update={
                "result": record.result.model_copy(
                    update={
                        "dispatch_attempts": (
                            conflicting_link,
                            *record.result.dispatch_attempts[1:],
                        )
                    },
                    deep=True,
                )
            },
            deep=True,
        )

        with pytest.raises(ValidationError, match="idempotency authority"):
            fork_group_runtime._ForkGroupRecord.model_validate(
                conflicting_record.model_dump(mode="json", warnings=False)
            )

        task = await task_store.load_task(link.queue_task_id)
        assert task is not None
        envelope = fork_group_runtime._existing_queued_dispatch_envelope(
            task,
            task_type=link.queue_task_type,
        )
        assert envelope is not None
        assert not fork_group_runtime._envelope_matches_dispatch_link(
            envelope,
            conflicting_link,
        )

    asyncio.run(run())


def test_terminal_task_cancellation_accepts_a_concurrent_terminal_winner() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider(fail_branch="alpha")
        session_store = InMemorySessionStore()
        task_store = _TerminalizingCancelTaskStore()
        app, dispatcher = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        await _source(app)
        request = _request(group_id="group-task-terminal-cancel-race").model_copy(
            update={"execution_mode": ForkGroupExecutionMode.TASK_DISPATCH}
        )

        result = await app.run_fork_group(request)
        assert result.state is ForkGroupState.BRANCHES_RUNNING
        assert await dispatcher.process_next(app, worker_id="worker-alpha") is not None
        task_store.terminalize_before_cancel = True

        result = await app.run_fork_group(request)

        assert result.state is ForkGroupState.FAILED
        pending_sibling = next(
            attempt for attempt in result.dispatch_attempts if attempt.branch_id == "beta"
        )
        assert pending_sibling.task_status is TaskStatus.FAILED

    asyncio.run(run())


def test_terminal_group_fences_pending_tasks_before_terminal_publication(monkeypatch) -> None:
    class _SimulatedProcessLoss(BaseException):
        pass

    async def run() -> None:
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        app, dispatcher = _task_app(
            _ForkGroupProvider(fail_branch="alpha"),
            session_store=session_store,
            task_store=task_store,
        )
        await _source(app)
        request = _request(group_id="group-task-terminal-claim-fence").model_copy(
            update={"execution_mode": ForkGroupExecutionMode.TASK_DISPATCH}
        )

        scheduled = await app.run_fork_group(request)
        beta_link = next(
            attempt for attempt in scheduled.dispatch_attempts if attempt.branch_id == "beta"
        )
        assert await dispatcher.process_next(app, worker_id="worker-alpha") is not None

        async def lose_process_after_terminal_publication(*args, **kwargs):
            del args, kwargs
            raise _SimulatedProcessLoss

        monkeypatch.setattr(
            fork_group_runtime,
            "_public_group_result",
            lose_process_after_terminal_publication,
        )
        with pytest.raises(_SimulatedProcessLoss):
            await app.run_fork_group(request)

        durable = await session_store.load_session_operation(
            request.source_session_id,
            fork_group_runtime._storage_key(request.group_id),
        )
        assert durable is not None
        record = fork_group_runtime._ForkGroupRecord.model_validate(durable)
        assert record.result.state is ForkGroupState.FAILED
        beta_task = await task_store.load_task(beta_link.queue_task_id)
        assert beta_task is not None
        assert beta_task.status is TaskStatus.CANCELLED
        assert (
            await task_store.claim_task(
                "post-terminal-worker",
                TaskQuery(type=beta_link.queue_task_type),
            )
            is None
        )

    asyncio.run(run())


def test_terminal_group_interrupts_a_live_task_backed_sibling_before_returning() -> None:
    async def run() -> None:
        branch_started = asyncio.Event()
        release_branch = asyncio.Event()

        async def block_alpha() -> None:
            branch_started.set()
            await release_branch.wait()

        provider = _ForkGroupProvider(
            fail_branch="beta",
            branch_callback=block_alpha,
        )
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        app, dispatcher = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        await _source(app)
        request = _request(group_id="group-task-live-sibling-cancellation").model_copy(
            update={"execution_mode": ForkGroupExecutionMode.TASK_DISPATCH}
        )

        scheduled = await app.run_fork_group(request)
        alpha_worker = asyncio.create_task(dispatcher.process_next(app, worker_id="worker-alpha"))
        try:
            await asyncio.wait_for(branch_started.wait(), timeout=5)
            assert await dispatcher.process_next(app, worker_id="worker-beta") is not None

            failed = await asyncio.wait_for(app.run_fork_group(request), timeout=5)
            assert failed.state is ForkGroupState.FAILED
            assert failed.failure is not None
            assert failed.failure.code is ForkGroupFailureCode.BRANCH_FAILED

            alpha_link = next(
                attempt for attempt in scheduled.dispatch_attempts if attempt.branch_id == "alpha"
            )
            alpha_session = await session_store.load(alpha_link.session_id)
            assert alpha_session is not None
            assert alpha_session.status is SessionStatus.INTERRUPTED
            assert not app._session_control.has_active_tasks(alpha_link.session_id)

            alpha_handle = await asyncio.wait_for(alpha_worker, timeout=5)
            assert alpha_handle is not None
            alpha_task = await task_store.load_task(alpha_link.queue_task_id)
            assert alpha_task is not None
            assert alpha_task.status is TaskStatus.COMPLETED

            release_branch.set()
            await asyncio.sleep(0)
            reloaded = await session_store.load(alpha_link.session_id)
            assert reloaded is not None
            assert reloaded.status is SessionStatus.INTERRUPTED
        finally:
            release_branch.set()
            if not alpha_worker.done():
                alpha_worker.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await alpha_worker

    asyncio.run(run())


def test_terminal_group_waits_for_remote_sibling_quiescence(monkeypatch) -> None:
    async def run() -> None:
        branch_started = asyncio.Event()
        release_branch = asyncio.Event()

        async def block_alpha() -> None:
            branch_started.set()
            await release_branch.wait()

        provider = _ForkGroupProvider(
            fail_branch="beta",
            branch_callback=block_alpha,
        )
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        coordinator, coordinator_dispatcher = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        worker, worker_dispatcher = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        await _source(coordinator)
        request = _request(group_id="group-task-remote-sibling-cancellation").model_copy(
            update={"execution_mode": ForkGroupExecutionMode.TASK_DISPATCH}
        )

        scheduled = await coordinator.run_fork_group(request)
        alpha_link = next(
            attempt for attempt in scheduled.dispatch_attempts if attempt.branch_id == "alpha"
        )
        alpha_worker = asyncio.create_task(
            worker_dispatcher.process_next(worker, worker_id="remote-alpha")
        )
        try:
            await asyncio.wait_for(branch_started.wait(), timeout=5)
            assert (
                await coordinator_dispatcher.process_next(
                    coordinator,
                    worker_id="coordinator-beta",
                )
                is not None
            )

            pending = await asyncio.wait_for(coordinator.run_fork_group(request), timeout=5)
            assert pending.state is ForkGroupState.BRANCHES_RUNNING
            alpha_session = await session_store.load(alpha_link.session_id)
            assert alpha_session is not None
            assert alpha_session.status is SessionStatus.INTERRUPTING
            alpha_task = await task_store.load_task(alpha_link.queue_task_id)
            assert alpha_task is not None
            assert alpha_task.status is TaskStatus.CLAIMED

            release_branch.set()
            alpha_handle = await asyncio.wait_for(alpha_worker, timeout=5)
            assert alpha_handle is not None
            assert alpha_handle.status.value == "interrupted"

            failed = await asyncio.wait_for(coordinator.run_fork_group(request), timeout=5)
            assert failed.state is ForkGroupState.FAILED
            reloaded = await session_store.load(alpha_link.session_id)
            assert reloaded is not None
            assert reloaded.status is SessionStatus.INTERRUPTED
            terminal_task = await task_store.load_task(alpha_link.queue_task_id)
            assert terminal_task is not None
            assert terminal_task.status is TaskStatus.COMPLETED
        finally:
            release_branch.set()
            if not alpha_worker.done():
                alpha_worker.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await alpha_worker

    monkeypatch.setattr(
        session_control_runtime,
        "ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS",
        2,
    )
    monkeypatch.setattr(
        session_control_runtime,
        "ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S",
        0.001,
    )
    asyncio.run(run())


def test_terminal_group_inspection_survives_child_session_deletion() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        app, dispatcher = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        await _source(app)
        request = _request(group_id="group-task-child-deleted").model_copy(
            update={"execution_mode": ForkGroupExecutionMode.TASK_DISPATCH}
        )

        result = await app.run_fork_group(request)
        for worker_index in range(8):
            if result.state is ForkGroupState.COMPLETED:
                break
            assert (
                await dispatcher.process_next(
                    app,
                    worker_id=f"worker-{worker_index}",
                )
                is not None
            )
            result = await app.run_fork_group(request)
        assert result.state is ForkGroupState.COMPLETED

        deleted_session_id = result.branches[0].session_id
        await session_store.delete_session(deleted_session_id)

        inspected = await app.inspect_fork_group(
            request.source_session_id,
            request.group_id,
        )
        assert inspected is not None
        assert inspected.state is ForkGroupState.COMPLETED
        deleted_attempt = next(
            attempt
            for attempt in inspected.dispatch_attempts
            if attempt.session_id == deleted_session_id
        )
        assert deleted_attempt.task_status is TaskStatus.COMPLETED
        assert deleted_attempt.run_epoch is None
        replayed = await app.run_fork_group(request)
        assert replayed.state is ForkGroupState.COMPLETED
        assert replayed.replayed is True

    asyncio.run(run())


def test_run_fork_group_selects_one_tool_free_evaluated_branch() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="source", model="fake-model"))
        app.register_agent(
            AgentSpec(
                name="evaluator",
                model="fake-model",
                system_prompt="Select exactly one fork-group branch.",
                workflow_tool_names=("forbidden_evaluator_tool",),
            ),
            tools=[_ForbiddenEvaluatorTool()],
            hosted_tools=[OpenAIWebSearch()],
        )

        source_events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="source",
                    session_id="fork-group-source",
                    causal_budget_id="fork-group-budget",
                    messages=[Message.text("user", "prepare source")],
                )
            )
        ]
        assert source_events[-1].type == "session.completed"

        request = ForkGroupRequest(
            group_id="group-success",
            source_session_id="fork-group-source",
            source_checkpoint=ForkGroupCheckpointSelector(),
            causal_budget_id="fork-group-budget",
            max_parallelism=2,
            branches=(
                ForkGroupBranchSpec(
                    branch_id="alpha",
                    session_id="fork-group-alpha",
                    messages=(Message.text("user", "candidate alpha"),),
                    structured_output=_candidate_output("candidate-alpha"),
                ),
                ForkGroupBranchSpec(
                    branch_id="beta",
                    session_id="fork-group-beta",
                    messages=(Message.text("user", "candidate beta"),),
                    structured_output=_candidate_output("candidate-beta"),
                ),
            ),
            evaluator=ForkGroupEvaluatorSpec(
                session_id="fork-group-evaluator",
                agent_name="evaluator",
            ),
        )

        result = await app.run_fork_group(request)

        assert result.state is ForkGroupState.COMPLETED
        assert result.source.source_session_id == "fork-group-source"
        assert result.source.causal_budget_id == "fork-group-budget"
        assert {branch.branch_id for branch in result.branches} == {"alpha", "beta"}
        assert all(branch.status == "completed" for branch in result.branches)
        assert all(branch.usage.model_steps == 1 for branch in result.branches)
        assert [(item.branch_id, item.disposition) for item in result.dispositions] == [
            ("alpha", ForkGroupDisposition.SELECTED),
            ("beta", ForkGroupDisposition.REJECTED),
        ]
        assert provider.evaluator_tools is not None
        assert "forbidden_evaluator_tool" not in provider.evaluator_tools
        assert provider.evaluator_hosted_tools == ()
        assert provider.evaluator_evidence is not None
        assert set(provider.evaluator_evidence) == {"schema", "group_id", "source", "branches"}
        assert "prepare source" not in json.dumps(provider.evaluator_evidence)
        assert "candidate alpha" not in json.dumps(provider.evaluator_evidence)
        assert [item["branch_id"] for item in provider.evaluator_evidence["branches"]] == [
            "alpha",
            "beta",
        ]
        with pytest.raises(ValidationError, match="frozen"):
            result.dispositions[0].disposition = ForkGroupDisposition.ARCHIVED

    asyncio.run(run())


def test_fork_group_branch_admits_only_its_frozen_initial_invocation() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        app = _app(provider)
        await _source(app)
        request = _request(group_id="group-frozen-initial-invocation")
        coordinator = app._fork_group_coordinator
        prepared = fork_group_runtime._prepare_request(
            coordinator,
            request,
            source_session_id=request.source_session_id,
        )
        record = await fork_group_runtime._create_record(coordinator, prepared)
        branch = prepared.branches[0]
        assert (
            await fork_group_runtime._prepare_branch_fork(
                coordinator,
                prepared,
                record.result.source,
                branch,
            )
            is None
        )
        frozen = fork_group_runtime._branch_resume_request(prepared, branch)
        provider_request_count = len(provider.requests)

        with pytest.raises(ExecutionProfileMismatchError):
            _ = [
                event
                async for event in app.resume(
                    frozen.model_copy(update={"max_steps": frozen.max_steps + 1})
                )
            ]
        assert len(provider.requests) == provider_request_count

        events = [event async for event in app.resume(frozen)]
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(provider.requests) == provider_request_count + 1

    asyncio.run(run())


def test_run_fork_group_redacts_workload_input_before_durable_identity() -> None:
    async def run() -> None:
        secret = "fork-group-secret-canary"
        provider = _ForkGroupProvider()
        app = CayuApp(
            enable_logging=False,
            secret_redactor=SecretRedactor(secret),
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="source", model="fake-model"))
        app.register_agent(AgentSpec(name="evaluator", model="fake-model"))
        await _source(app)
        request = _request(group_id="group-secret-boundary")
        alpha, beta = request.branches
        request = request.model_copy(
            update={
                "branches": (
                    alpha.model_copy(
                        update={
                            "messages": (Message.text("user", f"candidate alpha {secret}"),),
                            "artifact_references": (
                                ForkGroupArtifactReference(
                                    artifact_id="alpha-report",
                                    description=f"validated report {secret}",
                                ),
                            ),
                        },
                        deep=True,
                    ),
                    beta,
                )
            },
            deep=True,
        )

        result = await app.run_fork_group(request)

        assert result.state is ForkGroupState.COMPLETED
        assert secret not in json.dumps(provider.evaluator_evidence)
        assert all(
            secret not in json.dumps(item.model_dump(mode="json")) for item in provider.requests
        )
        checkpoint = await app.session_store.load_checkpoint("fork-group-source")
        assert secret not in json.dumps(checkpoint)

    asyncio.run(run())


def test_runtime_owned_source_snapshot_is_not_redacted_as_workload_metadata() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        app = CayuApp(
            enable_logging=False,
            secret_redactor=SecretRedactor("completed"),
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="source", model="fake-model"))
        app.register_agent(AgentSpec(name="evaluator", model="fake-model"))
        await _source(app)

        result = await app.run_fork_group(_request(group_id="group-runtime-source-authority"))

        assert result.state is ForkGroupState.COMPLETED
        for branch_id in ("alpha", "beta"):
            child = await app.session_store.load(f"group-runtime-source-authority-{branch_id}")
            assert child is not None
            assert child.metadata["cayu:fork_group_source_snapshot"]["status"] == "completed"

    asyncio.run(run())


def test_public_fork_request_cannot_forge_fork_group_source_authority() -> None:
    with pytest.raises(ValidationError, match="runtime-owned fork-group source authority"):
        ForkSessionRequest(
            source_session_id="source",
            metadata={"cayu:fork_group_source_snapshot": {"status": "completed"}},
        )


def test_run_fork_group_replays_terminal_result_and_rejects_conflicts() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        app = _app(provider)
        await _source(app)
        request = _request(group_id="group-replay")

        first = await app.run_fork_group(request)
        request_count = len(provider.requests)
        replay = await app.run_fork_group(request)

        assert first.state is ForkGroupState.COMPLETED
        assert first.replayed is False
        assert replay.state is ForkGroupState.COMPLETED
        assert replay.replayed is True
        assert replay.dispositions == first.dispositions
        assert len(provider.requests) == request_count
        records = await app.session_store.query_events(
            EventQuery(
                session_id="fork-group-source",
                event_types=(
                    "fork_group.created",
                    "fork_group.branches_running",
                    "fork_group.awaiting_evaluation",
                    "fork_group.completed",
                    "fork_group.failed",
                ),
            )
        )
        assert [record.event.type for record in records] == [
            "fork_group.created",
            "fork_group.branches_running",
            "fork_group.awaiting_evaluation",
            "fork_group.completed",
        ]
        terminal_payload = records[-1].event.payload
        assert terminal_payload["schema_version"] == 3
        assert terminal_payload["selected_branch_id"] == "alpha"
        assert terminal_payload["selected_attempt_id"] == first.branches[0].attempt_id
        assert terminal_payload["dispositions"] == [
            {
                "branch_id": "alpha",
                "attempt_id": first.branches[0].attempt_id,
                "disposition": "selected",
                "reason": "deterministic test judgment",
            },
            {
                "branch_id": "beta",
                "attempt_id": first.branches[1].attempt_id,
                "disposition": "rejected",
                "reason": "deterministic test judgment",
            },
        ]
        assert terminal_payload["branches"] == [
            {
                "branch_id": "alpha",
                "attempt_id": first.branches[0].attempt_id,
                "attempt_request_sha256": first.branches[0].attempt_request_sha256,
                "attempt_index": 0,
                "replaced_attempt_id": None,
                "superseded_by_attempt_id": None,
                "session_id": "group-replay-alpha",
                "status": "completed",
                "eligible": True,
            },
            {
                "branch_id": "beta",
                "attempt_id": first.branches[1].attempt_id,
                "attempt_request_sha256": first.branches[1].attempt_request_sha256,
                "attempt_index": 0,
                "replaced_attempt_id": None,
                "superseded_by_attempt_id": None,
                "session_id": "group-replay-beta",
                "status": "completed",
                "eligible": True,
            },
        ]

        with pytest.raises(ForkGroupConflict, match="different request"):
            await app.run_fork_group(request.model_copy(update={"max_parallelism": 1}))
        assert len(provider.requests) == request_count

        stored = await app.session_store.load_session_operation(
            request.source_session_id,
            fork_group_runtime._storage_key(request.group_id),
        )
        assert stored is not None
        assert stored["schema_version"] == 3
        old_schema = json.loads(json.dumps(stored))
        old_schema["schema_version"] = 2
        with pytest.raises(ValidationError, match="Unsupported fork-group operation record"):
            fork_group_runtime._ForkGroupRecord.model_validate(old_schema)

    asyncio.run(run())


def test_task_backed_fork_group_queues_bounded_attempts_and_replays_once() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        app, dispatcher = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        await _source(app)
        request = _request(
            group_id="group-task-dispatch",
            max_parallelism=1,
        ).model_copy(update={"execution_mode": ForkGroupExecutionMode.TASK_DISPATCH})
        source_request_count = len(provider.requests)

        first = await app.run_fork_group(request)

        assert first.state is ForkGroupState.BRANCHES_RUNNING
        assert len(first.dispatch_attempts) == 1
        assert first.dispatch_attempts[0].kind is ForkGroupAttemptKind.BRANCH
        assert first.dispatch_attempts[0].task_status is TaskStatus.PENDING
        first_child = await session_store.load(first.dispatch_attempts[0].session_id)
        assert first_child is not None and first_child.run_epoch == 0
        assert len(provider.requests) == source_request_count
        task_count = len(await task_store.list_tasks())
        inspected = await app.inspect_fork_group(request.source_session_id, request.group_id)
        assert inspected is not None
        assert inspected.execution_mode is ForkGroupExecutionMode.TASK_DISPATCH
        assert inspected.state is ForkGroupState.BRANCHES_RUNNING
        assert inspected.replayed is True
        assert inspected.dispatch_attempts == first.dispatch_attempts
        assert len(await task_store.list_tasks()) == task_count
        assert len(provider.requests) == source_request_count
        assert await app.inspect_fork_group(request.source_session_id, "missing-group") is None

        first_handle = await dispatcher.process_next(app, worker_id="worker-one")
        assert first_handle is not None
        second = await app.run_fork_group(request)
        assert second.state is ForkGroupState.BRANCHES_RUNNING
        assert len(second.dispatch_attempts) == 2
        assert (
            sum(attempt.task_status is TaskStatus.PENDING for attempt in second.dispatch_attempts)
            == 1
        )

        second_handle = await dispatcher.process_next(app, worker_id="worker-two")
        assert second_handle is not None
        awaiting = await app.run_fork_group(request)
        assert awaiting.state is ForkGroupState.AWAITING_EVALUATION
        assert [attempt.kind for attempt in awaiting.dispatch_attempts] == [
            ForkGroupAttemptKind.BRANCH,
            ForkGroupAttemptKind.BRANCH,
            ForkGroupAttemptKind.EVALUATOR,
        ]
        assert awaiting.dispatch_attempts[-1].task_status is TaskStatus.PENDING

        evaluator_handle = await dispatcher.process_next(app, worker_id="worker-three")
        assert evaluator_handle is not None
        completed = await app.run_fork_group(request)
        request_count = len(provider.requests)
        replay = await app.run_fork_group(request)

        assert completed.state is ForkGroupState.COMPLETED
        assert replay.state is ForkGroupState.COMPLETED and replay.replayed is True
        assert len(replay.dispatch_attempts) == 3
        assert all(
            attempt.task_status is TaskStatus.COMPLETED for attempt in replay.dispatch_attempts
        )
        assert len(provider.requests) == request_count
        assert provider.evaluator_calls == 1
        assert await dispatcher.process_next(app, worker_id="worker-four") is None

    asyncio.run(run())


def test_task_backed_fork_group_recovers_across_fresh_coordinators_and_workers() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()

        producer, _ = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        await _source(producer)
        request = _request(
            group_id="group-task-restart",
            max_parallelism=1,
        ).model_copy(update={"execution_mode": ForkGroupExecutionMode.TASK_DISPATCH})
        first = await producer.run_fork_group(request)
        assert first.state is ForkGroupState.BRANCHES_RUNNING

        branch_worker_one, branch_dispatcher_one = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        assert (
            await branch_dispatcher_one.process_next(
                branch_worker_one,
                worker_id="fresh-branch-one",
            )
            is not None
        )

        coordinator_two, _ = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        second = await coordinator_two.run_fork_group(request)
        assert len(second.dispatch_attempts) == 2

        branch_worker_two, branch_dispatcher_two = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        assert (
            await branch_dispatcher_two.process_next(
                branch_worker_two,
                worker_id="fresh-branch-two",
            )
            is not None
        )

        coordinator_three, _ = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        awaiting = await coordinator_three.run_fork_group(request)
        assert awaiting.state is ForkGroupState.AWAITING_EVALUATION

        evaluator_worker, evaluator_dispatcher = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        assert (
            await evaluator_dispatcher.process_next(
                evaluator_worker,
                worker_id="fresh-evaluator",
            )
            is not None
        )

        final_coordinator, final_dispatcher = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        completed = await final_coordinator.run_fork_group(request)
        assert completed.state is ForkGroupState.COMPLETED
        assert len(completed.dispatch_attempts) == 3
        assert provider.evaluator_calls == 1
        assert len(provider.requests) == 4
        assert (
            await final_dispatcher.process_next(
                final_coordinator,
                worker_id="fresh-idle",
            )
            is None
        )

    asyncio.run(run())


def test_terminal_task_group_reads_do_not_require_historical_evaluator_registration() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        producer, dispatcher = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        await _source(producer)
        request = _request(
            group_id="group-task-terminal-without-evaluator",
            max_parallelism=1,
        ).model_copy(update={"execution_mode": ForkGroupExecutionMode.TASK_DISPATCH})

        result = await producer.run_fork_group(request)
        for worker_index in range(8):
            if result.state is ForkGroupState.COMPLETED:
                break
            assert (
                await dispatcher.process_next(
                    producer,
                    worker_id=f"terminal-worker-{worker_index}",
                )
                is not None
            )
            result = await producer.run_fork_group(request)
        assert result.state is ForkGroupState.COMPLETED
        request_count = len(provider.requests)

        fresh_dispatcher = TaskStoreDispatcher(task_store)
        fresh = CayuApp(
            session_store=session_store,
            task_store=task_store,
            dispatcher=fresh_dispatcher,
            enable_logging=False,
        )
        fresh.register_provider(provider, default=True)
        fresh.register_agent(AgentSpec(name="source", model="fake-model"))

        inspected = await fresh.inspect_fork_group(
            request.source_session_id,
            request.group_id,
        )
        replayed = await fresh.run_fork_group(request)

        assert inspected is not None
        assert inspected.state is ForkGroupState.COMPLETED
        assert inspected.replayed is True
        assert replayed.state is ForkGroupState.COMPLETED
        assert replayed.replayed is True
        assert len(provider.requests) == request_count

    asyncio.run(run())


def test_task_backed_fork_group_dispatches_replacement_and_evaluator_attempts() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider(fail_branch="beta")
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        app, dispatcher = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        gate = app.register_fork_group_gate("tests", _ConfiguredGate())
        planner = app.register_fork_group_replacement_planner(
            "tests",
            _ConfiguredReplacementPlanner(),
        )
        await _source(app)
        request = _request(
            group_id="group-task-replacement",
            max_parallelism=2,
            gates=(gate,),
        ).model_copy(
            update={
                "execution_mode": ForkGroupExecutionMode.TASK_DISPATCH,
                "failure_policy": ForkGroupFailurePolicy(
                    mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                    minimum_viable_branches=2,
                    max_replacement_attempts=1,
                    replacement_parallelism=1,
                    replacement_planner=planner,
                ),
            }
        )

        result = await app.run_fork_group(request)
        for worker_index in range(8):
            if result.state in {ForkGroupState.COMPLETED, ForkGroupState.FAILED}:
                break
            handle = await dispatcher.process_next(
                app,
                worker_id=f"replacement-worker-{worker_index}",
            )
            assert handle is not None
            result = await app.run_fork_group(request)

        assert result.state is ForkGroupState.COMPLETED
        assert [attempt.kind for attempt in result.dispatch_attempts] == [
            ForkGroupAttemptKind.BRANCH,
            ForkGroupAttemptKind.BRANCH,
            ForkGroupAttemptKind.REPLACEMENT,
            ForkGroupAttemptKind.EVALUATOR,
        ]
        assert len(result.branches) == 3
        assert result.branches[1].superseded_by_attempt_id == result.branches[2].attempt_id
        assert result.branches[2].eligible is True
        assert provider.replacement_calls == 1
        assert provider.evaluator_calls == 1

    asyncio.run(run())


def test_task_backed_fork_group_worker_fails_closed_on_profile_drift() -> None:
    class DriftedProvider(_ForkGroupProvider):
        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return ExecutionProfileBehaviorIdentity(
                name="tests:fork-group-provider",
                behavior_version="2",
                implementation_version="2",
            )

    async def run() -> None:
        producer_provider = _ForkGroupProvider()
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        producer, _ = _task_app(
            producer_provider,
            session_store=session_store,
            task_store=task_store,
        )
        await _source(producer)
        request = _request(
            group_id="group-task-profile-drift",
            max_parallelism=2,
        ).model_copy(update={"execution_mode": ForkGroupExecutionMode.TASK_DISPATCH})
        queued = await producer.run_fork_group(request)
        assert queued.dispatch_attempts[0].task_status is TaskStatus.PENDING
        assert len(queued.dispatch_attempts) == 2

        worker_provider = DriftedProvider()
        worker, worker_dispatcher = _task_app(
            worker_provider,
            session_store=session_store,
            task_store=task_store,
        )
        handle = await worker_dispatcher.process_next(
            worker,
            worker_id="profile-drift-worker",
        )
        assert handle is not None
        assert worker_provider.requests == []

        failed = await producer.run_fork_group(request)
        assert failed.state is ForkGroupState.FAILED
        assert failed.failure is not None
        assert failed.failure.code is ForkGroupFailureCode.BRANCH_FAILED
        assert [attempt.task_status for attempt in failed.dispatch_attempts] == [
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ]
        assert "profile" in failed.failure.message.lower()

    asyncio.run(run())


def test_task_backed_fork_group_persists_link_before_task_is_claimable() -> None:
    class LinkObservingTaskStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self, session_store: InMemorySessionStore) -> None:
            super().__init__()
            self.session_store = session_store
            self.linked_task_ids: list[str] = []

        async def create_task(self, request):
            raw = await self.session_store.load_session_operation(
                "fork-group-source",
                fork_group_runtime._storage_key("group-task-link-order"),
            )
            assert raw is not None
            record = fork_group_runtime._ForkGroupRecord.model_validate(raw)
            assert request.task_id is not None
            assert request.task_id in {
                attempt.queue_task_id for attempt in record.result.dispatch_attempts
            }
            self.linked_task_ids.append(request.task_id)
            return await super().create_task(request)

    async def run() -> None:
        provider = _ForkGroupProvider()
        session_store = InMemorySessionStore()
        task_store = LinkObservingTaskStore(session_store)
        app, _ = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        await _source(app)
        request = _request(
            group_id="group-task-link-order",
            max_parallelism=1,
        ).model_copy(update={"execution_mode": ForkGroupExecutionMode.TASK_DISPATCH})

        queued = await app.run_fork_group(request)

        assert queued.state is ForkGroupState.BRANCHES_RUNNING
        assert task_store.linked_task_ids == [queued.dispatch_attempts[0].queue_task_id]
        assert queued.dispatch_attempts[0].task_status is TaskStatus.PENDING

    asyncio.run(run())


def test_task_backed_fork_group_worker_rejects_missing_group_authority() -> None:
    class AuthorityHidingSessionStore(InMemorySessionStore):
        hide_group = False

        async def load_session_operation(
            self,
            session_id: str,
            idempotency_key: str,
            **kwargs,
        ):
            if self.hide_group and idempotency_key == fork_group_runtime._storage_key(
                "group-task-missing-authority"
            ):
                return None
            return await super().load_session_operation(
                session_id,
                idempotency_key,
                **kwargs,
            )

    async def run() -> None:
        provider = _ForkGroupProvider()
        session_store = AuthorityHidingSessionStore()
        task_store = InMemoryTaskStore()
        app, dispatcher = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        await _source(app)
        request = _request(
            group_id="group-task-missing-authority",
            max_parallelism=1,
        ).model_copy(update={"execution_mode": ForkGroupExecutionMode.TASK_DISPATCH})
        queued = await app.run_fork_group(request)
        source_request_count = len(provider.requests)

        session_store.hide_group = True
        handle = await dispatcher.process_next(app, worker_id="missing-authority-worker")
        session_store.hide_group = False

        assert handle is not None
        assert len(provider.requests) == source_request_count
        task = await task_store.load_task(queued.dispatch_attempts[0].queue_task_id)
        assert task is not None and task.status is TaskStatus.FAILED

    asyncio.run(run())


def test_task_backed_fork_group_reclaim_fences_stale_worker_without_duplicate_run() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(task_store, lease_seconds=1)
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="source", model="fake-model"))
        app.register_agent(AgentSpec(name="evaluator", model="fake-model"))
        await _source(app)
        request = _request(
            group_id="group-task-stale-lease",
            max_parallelism=1,
        ).model_copy(update={"execution_mode": ForkGroupExecutionMode.TASK_DISPATCH})
        queued = await app.run_fork_group(request)
        task_id = queued.dispatch_attempts[0].queue_task_id
        stale_claim = await task_store.claim_task(
            "stale-worker",
            TaskQuery(type=dispatcher.fork_group_task_type),
            lease_seconds=1,
        )
        assert stale_claim is not None and stale_claim.id == task_id

        await asyncio.sleep(1.05)
        reclaimed = await task_store.reclaim_expired(
            query=TaskQuery(type=dispatcher.fork_group_task_type)
        )
        assert [task.id for task in reclaimed] == [task_id]
        replacement_claim = await task_store.claim_task(
            "replacement-worker",
            TaskQuery(type=dispatcher.fork_group_task_type),
            lease_seconds=1,
        )
        assert replacement_claim is not None and replacement_claim.id == task_id
        with pytest.raises(ValueError, match="does not own"):
            await task_store.complete_task(
                task_id,
                {"status": "completed"},
                worker_id="stale-worker",
            )
        await task_store.release_task(task_id, "replacement-worker")

        result = queued
        for worker_index in range(6):
            handle = await dispatcher.process_next(
                app,
                worker_id=f"settlement-worker-{worker_index}",
            )
            assert handle is not None
            result = await app.run_fork_group(request)
            if result.state is ForkGroupState.COMPLETED:
                break

        assert result.state is ForkGroupState.COMPLETED
        assert provider.evaluator_calls == 1
        assert len(provider.requests) == 4
        assert all(
            attempt.task_status is TaskStatus.COMPLETED for attempt in result.dispatch_attempts
        )

    asyncio.run(run())


def test_run_fork_group_concurrent_apps_share_one_durable_execution_owner() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = _ForkGroupProvider(branch_delay=0.1)

        def shared_app() -> CayuApp:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="source", model="fake-model"))
            app.register_agent(AgentSpec(name="evaluator", model="fake-model"))
            return app

        first_app = shared_app()
        second_app = shared_app()
        await _source(first_app)
        request = _request(group_id="group-concurrent-apps")

        first, second = await asyncio.gather(
            first_app.run_fork_group(request),
            second_app.run_fork_group(request),
        )
        completed = await first_app.run_fork_group(request)

        assert {first.state, second.state} <= {
            ForkGroupState.CREATED,
            ForkGroupState.BRANCHES_RUNNING,
            ForkGroupState.AWAITING_EVALUATION,
            ForkGroupState.COMPLETED,
        }
        assert completed.state is ForkGroupState.COMPLETED
        assert completed.failure is None
        assert provider.evaluator_calls == 1
        assert (
            sum(
                "candidate alpha" in str(message.content)
                or "candidate beta" in str(message.content)
                for model_request in provider.requests
                for message in model_request.messages
            )
            == 2
        )
        for session_id in (
            "group-concurrent-apps-alpha",
            "group-concurrent-apps-beta",
        ):
            child = await store.load(session_id)
            assert child is not None
            assert child.status is SessionStatus.COMPLETED

    asyncio.run(run())


def test_run_fork_group_reconciles_lost_execution_claim_acknowledgement() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        app = _app(provider)
        await _source(app)
        request = _request(group_id="group-claim-ack-loss")
        original_publish = app.session_store.publish_session_operation
        acknowledgement_lost = False

        async def publish_then_lose_claim_ack(session_id: str, **kwargs):
            nonlocal acknowledgement_lost
            published = await original_publish(session_id, **kwargs)
            durable = await app.session_store.load_session_operation(
                request.source_session_id,
                fork_group_runtime._storage_key(request.group_id),
            )
            if (
                not acknowledgement_lost
                and not kwargs.get("events")
                and durable is not None
                and durable.get("execution_claim") is not None
            ):
                acknowledgement_lost = True
                raise ConnectionError("simulated execution-claim acknowledgement loss")
            return published

        app.session_store.publish_session_operation = (  # ty: ignore[invalid-assignment]
            publish_then_lose_claim_ack
        )

        result = await app.run_fork_group(request)

        assert acknowledgement_lost is True
        assert result.state is ForkGroupState.COMPLETED
        assert result.failure is None
        assert provider.evaluator_calls == 1

    asyncio.run(run())


def test_run_fork_group_enforces_exact_source_checkpoint_selector() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        app = _app(provider)
        await _source(app)
        source = await app.session_store.load("fork-group-source")
        assert source is not None
        profile = execution_profile_baseline_from_session_metadata(source.metadata)
        assert profile is not None
        cursor = await app.session_store.load_transcript_cursor(source.id)
        selector = ForkGroupCheckpointSelector(
            expected_run_epoch=source.run_epoch,
            expected_transcript_cursor=cursor,
            expected_profile_fingerprint=profile.fingerprint,
        )

        result = await app.run_fork_group(
            _request(group_id="group-exact-selector").model_copy(
                update={"source_checkpoint": selector},
                deep=True,
            )
        )

        assert result.state is ForkGroupState.COMPLETED
        assert result.source.run_epoch == selector.expected_run_epoch
        assert result.source.transcript_cursor == selector.expected_transcript_cursor
        assert result.source.execution_profile_fingerprint == selector.expected_profile_fingerprint
        mismatches = (
            ForkGroupCheckpointSelector(expected_run_epoch=source.run_epoch + 1),
            ForkGroupCheckpointSelector(expected_transcript_cursor=cursor + 1),
            ForkGroupCheckpointSelector(expected_profile_fingerprint="0" * 64),
        )
        for index, mismatch in enumerate(mismatches):
            with pytest.raises(ValueError, match="Fork-group"):
                await app.run_fork_group(
                    _request(group_id=f"group-selector-mismatch-{index}").model_copy(
                        update={"source_checkpoint": mismatch},
                        deep=True,
                    )
                )

    asyncio.run(run())


def test_run_fork_group_preserves_successful_sibling_and_skips_evaluator_on_failure() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider(fail_branch="beta")
        app = _app(provider)
        await _source(app)

        result = await app.run_fork_group(_request(group_id="group-branch-failure"))

        assert result.state is ForkGroupState.FAILED
        assert result.failure is not None
        assert result.failure.code is ForkGroupFailureCode.BRANCH_FAILED
        assert result.failure.branch_id == "beta"
        assert [branch.status for branch in result.branches] == ["completed", "failed"]
        alpha = await app.session_store.load("group-branch-failure-alpha")
        beta = await app.session_store.load("group-branch-failure-beta")
        assert alpha is not None and alpha.status is SessionStatus.COMPLETED
        assert beta is not None and beta.status is SessionStatus.FAILED
        assert provider.evaluator_calls == 0
        assert await app.session_store.load("group-branch-failure-evaluator") is None

    asyncio.run(run())


def test_run_fork_group_replaces_failure_and_evaluates_viable_siblings() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider(fail_branch="beta")
        app = _app(provider)
        await _source(app)
        gate = _ConfiguredGate()
        gate_selection = app.register_fork_group_gate("tests", gate)
        planner = _ConfiguredReplacementPlanner()
        planner_selection = app.register_fork_group_replacement_planner(
            "tests",
            planner,
        )
        request = _request(
            group_id="group-viable-replacement",
            gates=(gate_selection,),
        ).model_copy(
            update={
                "failure_policy": ForkGroupFailurePolicy(
                    mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                    minimum_viable_branches=2,
                    max_replacement_attempts=1,
                    replacement_parallelism=1,
                    replacement_planner=planner_selection,
                )
            }
        )

        result = await app.run_fork_group(request)

        assert result.state is ForkGroupState.COMPLETED
        assert result.failure is None
        assert len(result.branches) == 3
        alpha, failed_beta, replacement_beta = result.branches
        assert alpha.branch_id == "alpha"
        assert alpha.attempt_index == 0
        assert alpha.eligible is True
        assert failed_beta.branch_id == "beta"
        assert failed_beta.attempt_index == 0
        assert failed_beta.status == "failed"
        assert failed_beta.eligible is False
        assert failed_beta.superseded_by_attempt_id == replacement_beta.attempt_id
        assert replacement_beta.branch_id == "beta"
        assert replacement_beta.attempt_index == 1
        assert replacement_beta.replaced_attempt_id == failed_beta.attempt_id
        assert replacement_beta.eligible is True
        assert replacement_beta.session_id != failed_beta.session_id
        assert [request.branch_id for request in planner.requests] == ["beta"]
        assert planner.requests[0].attempt_index == 1
        assert planner.requests[0].replaced_attempt.attempt_id == failed_beta.attempt_id
        assert planner.requests[0].idempotency_key == replacement_beta.attempt_id
        assert provider.evaluator_evidence is not None
        assert provider.evaluator_evidence["schema"] == "cayu.fork-group-evidence.v2"
        assert [branch["attempt_id"] for branch in provider.evaluator_evidence["branches"]] == [
            alpha.attempt_id,
            replacement_beta.attempt_id,
        ]
        assert provider.evaluator_evidence["excluded_attempts"] == [
            {
                "attempt_id": failed_beta.attempt_id,
                "attempt_index": 0,
                "branch_id": "beta",
                "replaced_attempt_id": None,
                "session_id": failed_beta.session_id,
                "status": "failed",
                "superseded_by_attempt_id": replacement_beta.attempt_id,
                "gate_results": [],
                "error": failed_beta.error,
            }
        ]
        assert {item.attempt_id for item in result.dispositions} == {
            alpha.attempt_id,
            replacement_beta.attempt_id,
        }

    asyncio.run(run())


def test_viable_replacement_does_not_infer_budget_exhaustion_from_error_text() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider(
            fail_branch="beta",
            branch_failure_message="provider budget formatter failed",
        )
        app = _app(provider)
        await _source(app)
        gate_selection = app.register_fork_group_gate("tests", _ConfiguredGate())
        planner_selection = app.register_fork_group_replacement_planner(
            "tests",
            _ConfiguredReplacementPlanner(),
        )
        request = _request(
            group_id="group-non-budget-wording",
            gates=(gate_selection,),
        ).model_copy(
            update={
                "failure_policy": ForkGroupFailurePolicy(
                    mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                    minimum_viable_branches=2,
                    max_replacement_attempts=1,
                    replacement_parallelism=1,
                    replacement_planner=planner_selection,
                )
            }
        )

        result = await app.run_fork_group(request)

        assert result.state is ForkGroupState.COMPLETED
        assert result.failure is None
        assert result.branches[1].failure_code is ForkGroupFailureCode.BRANCH_FAILED
        assert result.branches[2].eligible is True
        assert provider.replacement_calls == 1
        assert provider.evaluator_calls == 1

    asyncio.run(run())


def test_evaluate_viable_policy_requires_bounded_replacement_and_gates() -> None:
    planner = ForkGroupReplacementPlannerSelection(
        planner_id="tests",
        planner_identity="tests.configured-replacement-planner.v1",
    )
    policy = ForkGroupFailurePolicy(
        mode=ForkGroupFailureMode.EVALUATE_VIABLE,
        minimum_viable_branches=2,
        max_replacement_attempts=1,
        replacement_parallelism=1,
        replacement_planner=planner,
    )

    assert ForkGroupFailurePolicy().mode is ForkGroupFailureMode.FAIL_GROUP
    with pytest.raises(ValidationError, match="deterministic gates"):
        ForkGroupRequest.model_validate(
            _request().model_dump(mode="python") | {"failure_policy": policy}
        )
    with pytest.raises(ValidationError, match="cannot exceed candidate slots"):
        ForkGroupRequest.model_validate(
            _request(
                gates=(ForkGroupGateSelection(gate_id="gate", gate_identity="v1"),)
            ).model_dump(mode="python")
            | {"failure_policy": policy.model_copy(update={"minimum_viable_branches": 3})}
        )
    with pytest.raises(ValidationError, match="cannot exceed max_parallelism"):
        ForkGroupRequest.model_validate(
            _request(
                max_parallelism=1,
                gates=(ForkGroupGateSelection(gate_id="gate", gate_identity="v1"),),
            ).model_dump(mode="python")
            | {"failure_policy": policy.model_copy(update={"replacement_parallelism": 2})}
        )
    with pytest.raises(ValidationError, match="fail-group policy"):
        ForkGroupFailurePolicy(max_replacement_attempts=1)


def test_attempt_identity_is_scoped_to_source_authority() -> None:
    request = _request(group_id="group-shared-name")
    other_source = request.model_copy(update={"source_session_id": "other-source"})
    other_checkpoint = request.model_copy(
        update={
            "source_checkpoint": ForkGroupCheckpointSelector(expected_run_epoch=99),
        }
    )

    attempt_id = fork_group_runtime._attempt_id(request, "beta", 1)

    assert attempt_id != fork_group_runtime._attempt_id(other_source, "beta", 1)
    assert attempt_id != fork_group_runtime._attempt_id(other_checkpoint, "beta", 1)


def test_run_fork_group_records_multiple_replacements_and_limit_exhaustion() -> None:
    async def run() -> None:
        for group_id, failures, expected_state in (
            ("group-multiple-replacements", 1, ForkGroupState.COMPLETED),
            ("group-replacements-exhausted", 2, ForkGroupState.FAILED),
        ):
            provider = _ForkGroupProvider(
                fail_branch="beta",
                replacement_failures=failures,
            )
            app = _app(provider)
            await _source(app)
            gate_selection = app.register_fork_group_gate("tests", _ConfiguredGate())
            planner = _ConfiguredReplacementPlanner()
            planner_selection = app.register_fork_group_replacement_planner(
                "tests",
                planner,
            )
            request = _request(group_id=group_id, gates=(gate_selection,)).model_copy(
                update={
                    "failure_policy": ForkGroupFailurePolicy(
                        mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                        minimum_viable_branches=2,
                        max_replacement_attempts=2,
                        replacement_parallelism=1,
                        replacement_planner=planner_selection,
                    )
                }
            )

            result = await app.run_fork_group(request)

            assert result.state is expected_state
            assert len(result.branches) == 4
            beta_attempts = [attempt for attempt in result.branches if attempt.branch_id == "beta"]
            assert [attempt.attempt_index for attempt in beta_attempts] == [0, 1, 2]
            assert [attempt.replaced_attempt_id for attempt in beta_attempts] == [
                None,
                beta_attempts[0].attempt_id,
                beta_attempts[1].attempt_id,
            ]
            assert [attempt.superseded_by_attempt_id for attempt in beta_attempts] == [
                beta_attempts[1].attempt_id,
                beta_attempts[2].attempt_id,
                None,
            ]
            assert provider.replacement_calls == 2
            assert [item.attempt_index for item in planner.requests] == [1, 2]
            if expected_state is ForkGroupState.COMPLETED:
                assert beta_attempts[-1].eligible is True
                assert result.failure is None
                assert provider.evaluator_calls == 1
            else:
                assert all(not attempt.eligible for attempt in beta_attempts)
                assert result.failure is not None
                assert result.failure.code is ForkGroupFailureCode.REPLACEMENTS_EXHAUSTED
                assert result.dispositions == ()
                assert provider.evaluator_calls == 0

    asyncio.run(run())


def test_run_fork_group_replaces_deterministically_ineligible_attempt() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        app = _app(provider)
        await _source(app)
        gate = _AttemptGate(("beta", 0))
        gate_selection = app.register_fork_group_gate("attempt-gate", gate)
        planner = _ConfiguredReplacementPlanner()
        planner_selection = app.register_fork_group_replacement_planner(
            "tests",
            planner,
        )
        request = _request(
            group_id="group-replace-gate-rejection",
            gates=(gate_selection,),
        ).model_copy(
            update={
                "failure_policy": ForkGroupFailurePolicy(
                    mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                    minimum_viable_branches=2,
                    max_replacement_attempts=1,
                    replacement_parallelism=1,
                    replacement_planner=planner_selection,
                )
            }
        )

        result = await app.run_fork_group(request)

        assert result.state is ForkGroupState.COMPLETED
        _, rejected, replacement = result.branches
        assert rejected.status.value == "completed"
        assert rejected.eligible is False
        assert rejected.gate_results[0].passed is False
        assert rejected.superseded_by_attempt_id == replacement.attempt_id
        assert replacement.eligible is True
        assert replacement.gate_results[0].passed is True
        assert [(item.branch.branch_id, item.branch.attempt_index) for item in gate.requests] == [
            ("alpha", 0),
            ("beta", 0),
            ("beta", 1),
        ]
        assert provider.evaluator_evidence is not None
        assert (
            provider.evaluator_evidence["excluded_attempts"][0]["gate_results"][0]["passed"]
            is False
        )

    asyncio.run(run())


def test_evaluator_cannot_select_an_excluded_attempt() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider(
            fail_branch="beta",
            invalid_judgment="select-excluded",
        )
        app = _app(provider)
        await _source(app)
        gate_selection = app.register_fork_group_gate("tests", _ConfiguredGate())
        planner_selection = app.register_fork_group_replacement_planner(
            "tests",
            _ConfiguredReplacementPlanner(),
        )
        request = _request(
            group_id="group-reject-excluded-selection",
            gates=(gate_selection,),
            extra_alpha=True,
        ).model_copy(
            update={
                "failure_policy": ForkGroupFailurePolicy(
                    mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                    minimum_viable_branches=2,
                    max_replacement_attempts=1,
                    replacement_parallelism=1,
                    replacement_planner=planner_selection,
                )
            }
        )

        result = await app.run_fork_group(request)

        assert result.state is ForkGroupState.FAILED
        assert result.failure is not None
        assert result.failure.code is ForkGroupFailureCode.EVALUATOR_FAILED
        assert result.dispositions == ()
        assert len(result.branches) == 3
        assert any(attempt.status.value == "failed" for attempt in result.branches)
        assert all(attempt.attempt_index == 0 for attempt in result.branches)

    asyncio.run(run())


def test_evaluator_schema_exposes_only_eligible_attempts() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider(fail_branch="beta")
        app = _app(provider)
        await _source(app)
        gate_selection = app.register_fork_group_gate("tests", _ConfiguredGate())
        planner_selection = app.register_fork_group_replacement_planner(
            "tests",
            _ConfiguredReplacementPlanner(),
        )
        request = _request(
            group_id="group-eligible-evaluator-schema",
            gates=(gate_selection,),
        ).model_copy(
            update={
                "failure_policy": ForkGroupFailurePolicy(
                    mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                    minimum_viable_branches=1,
                    max_replacement_attempts=1,
                    replacement_parallelism=1,
                    replacement_planner=planner_selection,
                )
            }
        )

        result = await app.run_fork_group(request)

        assert result.state is ForkGroupState.COMPLETED
        evaluator_request = next(
            model_request
            for model_request in provider.requests
            if any(
                type(part) is TextPart and "cayu.fork-group-evidence.v2" in part.text
                for message in model_request.messages
                for part in message.content
            )
        )
        structured_tool = next(
            tool for tool in evaluator_request.tools if tool["name"].startswith("__cayu")
        )
        dispositions = structured_tool["input_schema"]["properties"]["output"]["properties"][
            "dispositions"
        ]
        assert dispositions["minItems"] == 1
        assert dispositions["maxItems"] == 1
        assert dispositions["items"]["properties"]["branch_id"]["enum"] == ["alpha"]

    asyncio.run(run())


@pytest.mark.parametrize("planner_state", ["missing", "changed"])
def test_viable_replacement_fails_closed_on_planner_authority(
    planner_state: str,
) -> None:
    async def run() -> None:
        provider = _ForkGroupProvider(fail_branch="beta")
        app = _app(provider)
        await _source(app)
        gate_selection = app.register_fork_group_gate("tests", _ConfiguredGate())
        planner_selection = ForkGroupReplacementPlannerSelection(
            planner_id="tests",
            planner_identity="tests.configured-replacement-planner.v1",
        )
        if planner_state == "changed":
            app.register_fork_group_replacement_planner(
                "tests",
                _ConfiguredReplacementPlanner(),
            )
            planner_selection = planner_selection.model_copy(
                update={"planner_identity": "tests.configured-replacement-planner.v2"}
            )
        request = _request(
            group_id=f"group-planner-authority-{planner_state}",
            gates=(gate_selection,),
        ).model_copy(
            update={
                "failure_policy": ForkGroupFailurePolicy(
                    mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                    minimum_viable_branches=2,
                    max_replacement_attempts=1,
                    replacement_parallelism=1,
                    replacement_planner=planner_selection,
                )
            }
        )

        result = await app.run_fork_group(request)

        assert result.state is ForkGroupState.FAILED
        assert result.failure is not None
        assert result.failure.code is ForkGroupFailureCode.REPLACEMENT_FAILED
        assert len(result.branches) == 2
        assert result.dispositions == ()
        assert provider.evaluator_calls == 0

    asyncio.run(run())


def test_failed_replacement_fork_is_published_as_a_durable_attempt() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider(fail_branch="beta")
        app = _app(provider)
        await _source(app)
        gate_selection = app.register_fork_group_gate("tests", _ConfiguredGate())
        planner = _ConfiguredReplacementPlanner()
        planner_selection = app.register_fork_group_replacement_planner("tests", planner)
        request = _request(
            group_id="group-replacement-fork-failure",
            gates=(gate_selection,),
        ).model_copy(
            update={
                "failure_policy": ForkGroupFailurePolicy(
                    mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                    minimum_viable_branches=2,
                    max_replacement_attempts=1,
                    replacement_parallelism=1,
                    replacement_planner=planner_selection,
                )
            }
        )
        original_fork_session = app.fork_session

        async def fail_replacement_fork(
            fork_request: ForkSessionRequest,
        ) -> AsyncIterator[Event]:
            if (fork_request.session_id or "").startswith("fork-replacement:"):
                raise RuntimeError("replacement child creation failed")
            async for event in original_fork_session(fork_request):
                yield event

        app.fork_session = fail_replacement_fork  # ty: ignore[invalid-assignment]
        app._fork_group_coordinator._fork_session_callback = fail_replacement_fork

        result = await app.run_fork_group(request)

        assert result.state is ForkGroupState.FAILED
        assert result.failure is not None
        assert result.failure.code is ForkGroupFailureCode.REPLACEMENT_FAILED
        assert len(result.branches) == 3
        failed_beta, failed_replacement = result.branches[1:]
        assert failed_beta.superseded_by_attempt_id == failed_replacement.attempt_id
        assert failed_replacement.replaced_attempt_id == failed_beta.attempt_id
        assert failed_replacement.status.value == "failed"
        assert failed_replacement.failure_code is ForkGroupFailureCode.BRANCH_FAILED
        assert failed_replacement.eligible is False
        assert (
            failed_replacement.execution_profile_fingerprint
            == result.source.execution_profile_fingerprint
        )
        assert len(planner.requests) == 1
        replacement_session_id = fork_group_runtime._replacement_session_id(
            planner.requests[0].attempt_id
        )
        assert failed_replacement.session_id == replacement_session_id
        assert await app.session_store.load(replacement_session_id) is None

    asyncio.run(run())


def test_failed_replacement_attempt_survives_owner_loss_before_execute_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = _ForkGroupProvider(fail_branch="beta")
        app = _app(provider, session_store=store)
        await _source(app)
        gate_selection = app.register_fork_group_gate("tests", _ConfiguredGate())
        planner_selection = app.register_fork_group_replacement_planner(
            "tests",
            _ConfiguredReplacementPlanner(),
        )
        request = _request(
            group_id="group-replacement-owner-loss",
            gates=(gate_selection,),
        ).model_copy(
            update={
                "failure_policy": ForkGroupFailurePolicy(
                    mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                    minimum_viable_branches=2,
                    max_replacement_attempts=1,
                    replacement_parallelism=1,
                    replacement_planner=planner_selection,
                )
            }
        )
        original_fork_session = app.fork_session

        async def fail_replacement_fork(
            fork_request: ForkSessionRequest,
        ) -> AsyncIterator[Event]:
            if (fork_request.session_id or "").startswith("fork-replacement:"):
                raise RuntimeError("replacement child creation failed")
            async for event in original_fork_session(fork_request):
                yield event

        app.fork_session = fail_replacement_fork  # ty: ignore[invalid-assignment]
        app._fork_group_coordinator._fork_session_callback = fail_replacement_fork
        original_run_viable_attempts = fork_group_runtime._run_viable_attempts

        async def lose_owner_after_viable_attempts(*args: Any, **kwargs: Any) -> Any:
            outcome = await original_run_viable_attempts(*args, **kwargs)
            assert outcome[2] is not None
            raise asyncio.CancelledError

        monkeypatch.setattr(
            fork_group_runtime,
            "_run_viable_attempts",
            lose_owner_after_viable_attempts,
        )

        with pytest.raises(asyncio.CancelledError):
            await app.run_fork_group(request)

        durable = await store.load_session_operation(
            request.source_session_id,
            fork_group_runtime._storage_key(request.group_id),
        )
        assert durable is not None
        assert durable["result"]["state"] == ForkGroupState.FAILED.value
        assert len(durable["result"]["branches"]) == 3
        assert (
            durable["result"]["branches"][-1]["execution_profile_fingerprint"]
            == (durable["result"]["source"]["execution_profile_fingerprint"])
        )

    asyncio.run(run())


def test_rejected_replacement_profile_fails_before_attempt_admission() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider(fail_branch="beta")
        app = _app(provider)
        await _source(app)
        gate_selection = app.register_fork_group_gate("tests", _ConfiguredGate())
        planner_selection = app.register_fork_group_replacement_planner(
            "missing-agent",
            _MissingAgentReplacementPlanner(),
        )
        request = _request(
            group_id="group-replacement-profile-rejected",
            gates=(gate_selection,),
        ).model_copy(
            update={
                "failure_policy": ForkGroupFailurePolicy(
                    mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                    minimum_viable_branches=2,
                    max_replacement_attempts=1,
                    replacement_parallelism=1,
                    replacement_planner=planner_selection,
                )
            }
        )

        result = await app.run_fork_group(request)

        assert result.state is ForkGroupState.FAILED
        assert result.failure is not None
        assert result.failure.code is ForkGroupFailureCode.REPLACEMENT_FAILED
        assert result.failure.branch_id == "beta"
        assert len(result.branches) == 2
        assert all(branch.attempt_index == 0 for branch in result.branches)
        durable = await app.session_store.load_session_operation(
            request.source_session_id,
            fork_group_runtime._storage_key(request.group_id),
        )
        assert durable is not None
        assert len(durable["result"]["branches"]) == 2

    asyncio.run(run())


def test_task_backed_rejected_replacement_profile_fails_before_attempt_admission() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider(fail_branch="beta")
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        app, dispatcher = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        await _source(app)
        gate_selection = app.register_fork_group_gate("tests", _ConfiguredGate())
        planner_selection = app.register_fork_group_replacement_planner(
            "missing-agent",
            _MissingAgentReplacementPlanner(),
        )
        request = _request(
            group_id="group-task-replacement-profile-rejected",
            max_parallelism=2,
            gates=(gate_selection,),
        ).model_copy(
            update={
                "execution_mode": ForkGroupExecutionMode.TASK_DISPATCH,
                "failure_policy": ForkGroupFailurePolicy(
                    mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                    minimum_viable_branches=2,
                    max_replacement_attempts=1,
                    replacement_parallelism=1,
                    replacement_planner=planner_selection,
                ),
            }
        )

        result = await app.run_fork_group(request)
        assert result.state is ForkGroupState.BRANCHES_RUNNING
        assert await dispatcher.process_next(app, worker_id="worker-alpha") is not None
        assert await dispatcher.process_next(app, worker_id="worker-beta") is not None

        result = await app.run_fork_group(request)

        assert result.state is ForkGroupState.FAILED
        assert result.failure is not None
        assert result.failure.code is ForkGroupFailureCode.REPLACEMENT_FAILED
        assert result.failure.branch_id == "beta"
        assert len(result.branches) == 2
        assert all(branch.attempt_index == 0 for branch in result.branches)
        inspected = await app.inspect_fork_group(
            request.source_session_id,
            request.group_id,
        )
        assert inspected is not None
        assert inspected.state is ForkGroupState.FAILED
        durable = await app.session_store.load_session_operation(
            request.source_session_id,
            fork_group_runtime._storage_key(request.group_id),
        )
        assert durable is not None
        assert len(durable["result"]["branches"]) == 2

    asyncio.run(run())


@pytest.mark.parametrize("lose_owner_after_attempt_publication", [False, True])
def test_task_backed_failed_replacement_fork_is_published_as_terminal_attempt(
    monkeypatch: pytest.MonkeyPatch,
    lose_owner_after_attempt_publication: bool,
) -> None:
    async def run() -> None:
        provider = _ForkGroupProvider(fail_branch="beta")
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        app, dispatcher = _task_app(
            provider,
            session_store=session_store,
            task_store=task_store,
        )
        await _source(app)
        gate_selection = app.register_fork_group_gate("tests", _ConfiguredGate())
        planner = _ConfiguredReplacementPlanner()
        planner_selection = app.register_fork_group_replacement_planner("tests", planner)
        request = _request(
            group_id="group-task-replacement-fork-failure",
            max_parallelism=2,
            gates=(gate_selection,),
        ).model_copy(
            update={
                "execution_mode": ForkGroupExecutionMode.TASK_DISPATCH,
                "failure_policy": ForkGroupFailurePolicy(
                    mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                    minimum_viable_branches=2,
                    max_replacement_attempts=1,
                    replacement_parallelism=1,
                    replacement_planner=planner_selection,
                ),
            }
        )
        original_fork_session = app.fork_session

        async def fail_replacement_fork(
            fork_request: ForkSessionRequest,
        ) -> AsyncIterator[Event]:
            if (fork_request.session_id or "").startswith("fork-replacement:"):
                raise RuntimeError("replacement child creation failed")
            async for event in original_fork_session(fork_request):
                yield event

        app.fork_session = fail_replacement_fork  # ty: ignore[invalid-assignment]
        app._fork_group_coordinator._fork_session_callback = fail_replacement_fork

        result = await app.run_fork_group(request)
        assert result.state is ForkGroupState.BRANCHES_RUNNING
        assert await dispatcher.process_next(app, worker_id="worker-alpha") is not None
        assert await dispatcher.process_next(app, worker_id="worker-beta") is not None

        if lose_owner_after_attempt_publication:
            original_publish = fork_group_runtime._publish_task_backed_branches
            owner_lost = False

            async def publish_then_lose_owner(*args: Any, **kwargs: Any) -> Any:
                nonlocal owner_lost
                published = await original_publish(*args, **kwargs)
                if not owner_lost and any(
                    branch.attempt_index > 0 for branch in published.result.branches
                ):
                    owner_lost = True
                    raise asyncio.CancelledError
                return published

            monkeypatch.setattr(
                fork_group_runtime,
                "_publish_task_backed_branches",
                publish_then_lose_owner,
            )
            with pytest.raises(asyncio.CancelledError):
                await app.run_fork_group(request)
            durable = await session_store.load_session_operation(
                request.source_session_id,
                fork_group_runtime._storage_key(request.group_id),
            )
            assert durable is not None
            assert durable["result"]["state"] == ForkGroupState.BRANCHES_RUNNING.value
            assert len(durable["result"]["branches"]) == 3
            assert len(planner.requests) == 1

        result = await app.run_fork_group(request)

        assert result.state is ForkGroupState.FAILED
        assert result.failure is not None
        assert result.failure.code is ForkGroupFailureCode.REPLACEMENT_FAILED
        assert result.failure.branch_id == "beta"
        assert len(result.branches) == 3
        failed_beta, failed_replacement = result.branches[1:]
        assert failed_beta.superseded_by_attempt_id == failed_replacement.attempt_id
        assert failed_replacement.replaced_attempt_id == failed_beta.attempt_id
        assert failed_replacement.status is ForkGroupBranchStatus.FAILED
        assert failed_replacement.execution_profile_fingerprint == (
            result.source.execution_profile_fingerprint
        )
        assert len(planner.requests) == 1
        assert await session_store.load(failed_replacement.session_id) is None

        replayed = await app.run_fork_group(request)
        assert replayed.state is ForkGroupState.FAILED
        assert replayed.replayed is True
        assert len(planner.requests) == 1
        assert provider.evaluator_calls == 0

    asyncio.run(run())


def test_rejected_replacement_profile_preserves_concurrent_authoritative_sibling() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        app = _app(provider)
        await _source(app)
        gate = _AttemptGate(("beta", 0), ("alpha-second", 0))
        gate_selection = app.register_fork_group_gate("attempt-gate", gate)
        planner_selection = app.register_fork_group_replacement_planner(
            "mixed-profile",
            _MixedProfileReplacementPlanner(),
        )
        request = _request(
            group_id="group-mixed-replacement-profile",
            gates=(gate_selection,),
            extra_alpha=True,
        ).model_copy(
            update={
                "failure_policy": ForkGroupFailurePolicy(
                    mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                    minimum_viable_branches=3,
                    max_replacement_attempts=2,
                    replacement_parallelism=2,
                    replacement_planner=planner_selection,
                )
            }
        )

        result = await app.run_fork_group(request)

        assert result.state is ForkGroupState.FAILED
        assert result.failure is not None
        assert result.failure.code is ForkGroupFailureCode.REPLACEMENT_FAILED
        assert result.failure.branch_id == "beta"
        assert len(result.branches) == 4
        beta = next(branch for branch in result.branches if branch.branch_id == "beta")
        alpha_second_attempts = [
            branch for branch in result.branches if branch.branch_id == "alpha-second"
        ]
        assert beta.superseded_by_attempt_id is None
        assert [branch.attempt_index for branch in alpha_second_attempts] == [0, 1]
        assert (
            alpha_second_attempts[0].superseded_by_attempt_id == alpha_second_attempts[1].attempt_id
        )
        assert alpha_second_attempts[1].execution_profile_fingerprint is not None
        replacement_child = await app.session_store.load(alpha_second_attempts[1].session_id)
        assert replacement_child is not None
        durable = await app.session_store.load_session_operation(
            request.source_session_id,
            fork_group_runtime._storage_key(request.group_id),
        )
        assert durable is not None
        assert len(durable["result"]["branches"]) == 4

    asyncio.run(run())


def test_replacement_planner_exception_is_redacted_before_durable_failure() -> None:
    async def run() -> None:
        secret = "fork-group-replacement-secret-canary"
        provider = _ForkGroupProvider(fail_branch="beta")
        app = CayuApp(
            enable_logging=False,
            secret_redactor=SecretRedactor(secret),
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="source", model="fake-model"))
        app.register_agent(AgentSpec(name="evaluator", model="fake-model"))
        await _source(app)
        gate_selection = app.register_fork_group_gate("tests", _ConfiguredGate())
        planner_selection = app.register_fork_group_replacement_planner(
            "secret-planner",
            _SecretFailingReplacementPlanner(secret),
        )
        request = _request(
            group_id="group-secret-replacement-failure",
            gates=(gate_selection,),
        ).model_copy(
            update={
                "failure_policy": ForkGroupFailurePolicy(
                    mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                    minimum_viable_branches=2,
                    max_replacement_attempts=1,
                    replacement_parallelism=1,
                    replacement_planner=planner_selection,
                )
            }
        )

        result = await app.run_fork_group(request)
        durable = await app.session_store.load_session_operation(
            request.source_session_id,
            fork_group_runtime._storage_key(request.group_id),
        )

        assert result.state is ForkGroupState.FAILED
        assert result.failure is not None
        assert result.failure.code is ForkGroupFailureCode.REPLACEMENT_FAILED
        assert secret not in result.failure.message
        assert secret not in json.dumps(durable)

    asyncio.run(run())


def test_viable_replacement_concurrent_recovery_and_terminal_replay_converge() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = _ForkGroupProvider(fail_branch="beta", branch_delay=0.1)

        def configured_app() -> tuple[CayuApp, _ConfiguredReplacementPlanner]:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="source", model="fake-model"))
            app.register_agent(AgentSpec(name="evaluator", model="fake-model"))
            app.register_fork_group_gate("tests", _ConfiguredGate())
            planner = _ConfiguredReplacementPlanner()
            app.register_fork_group_replacement_planner("tests", planner)
            return app, planner

        first_app, first_planner = configured_app()
        second_app, second_planner = configured_app()
        await _source(first_app)
        request = _request(
            group_id="group-concurrent-viable-recovery",
            gates=(
                ForkGroupGateSelection(
                    gate_id="tests",
                    gate_identity="tests.configured-gate.v1",
                ),
            ),
        ).model_copy(
            update={
                "failure_policy": ForkGroupFailurePolicy(
                    mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                    minimum_viable_branches=2,
                    max_replacement_attempts=1,
                    replacement_parallelism=1,
                    replacement_planner=ForkGroupReplacementPlannerSelection(
                        planner_id="tests",
                        planner_identity="tests.configured-replacement-planner.v1",
                    ),
                )
            }
        )

        first, second = await asyncio.gather(
            first_app.run_fork_group(request),
            second_app.run_fork_group(request),
        )
        completed = await second_app.run_fork_group(request)

        assert {first.state, second.state} <= {
            ForkGroupState.CREATED,
            ForkGroupState.BRANCHES_RUNNING,
            ForkGroupState.AWAITING_EVALUATION,
            ForkGroupState.COMPLETED,
        }
        assert completed.state is ForkGroupState.COMPLETED
        assert provider.replacement_calls == 1
        assert provider.evaluator_calls == 1
        assert len(first_planner.requests) + len(second_planner.requests) == 1
        replacement = next(attempt for attempt in completed.branches if attempt.attempt_index == 1)
        child = await store.load(replacement.session_id)
        assert child is not None and child.status is SessionStatus.COMPLETED

        replay_provider = _ForkGroupProvider(fail_branch="alpha", fail_evaluator=True)
        replay_app = CayuApp(session_store=store, enable_logging=False)
        replay_app.register_provider(replay_provider, default=True)
        replay_app.register_agent(AgentSpec(name="source", model="fake-model"))
        replay_app.register_agent(AgentSpec(name="evaluator", model="fake-model"))
        replay = await replay_app.run_fork_group(request)

        assert replay.state is ForkGroupState.COMPLETED
        assert replay.replayed is True
        assert replay.branches == completed.branches
        assert replay.dispositions == completed.dispositions
        assert replay_provider.requests == []

    asyncio.run(run())


def test_viable_replacement_recovers_after_owner_loss_mid_replacement() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        replacement_started = asyncio.Event()
        hold_replacement = asyncio.Event()

        async def block_replacement() -> None:
            replacement_started.set()
            await hold_replacement.wait()

        def configured_app(provider: _ForkGroupProvider) -> CayuApp:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="source", model="fake-model"))
            app.register_agent(AgentSpec(name="evaluator", model="fake-model"))
            app.register_fork_group_gate("tests", _ConfiguredGate())
            app.register_fork_group_replacement_planner(
                "tests",
                _ConfiguredReplacementPlanner(),
            )
            return app

        first_provider = _ForkGroupProvider(
            fail_branch="beta",
            replacement_callback=block_replacement,
        )
        first_app = configured_app(first_provider)
        await _source(first_app)
        request = _request(
            group_id="group-owner-loss-mid-replacement",
            gates=(
                ForkGroupGateSelection(
                    gate_id="tests",
                    gate_identity="tests.configured-gate.v1",
                ),
            ),
        ).model_copy(
            update={
                "failure_policy": ForkGroupFailurePolicy(
                    mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                    minimum_viable_branches=2,
                    max_replacement_attempts=2,
                    replacement_parallelism=1,
                    replacement_planner=ForkGroupReplacementPlannerSelection(
                        planner_id="tests",
                        planner_identity="tests.configured-replacement-planner.v1",
                    ),
                )
            }
        )

        abandoned = asyncio.create_task(first_app.run_fork_group(request))
        await asyncio.wait_for(replacement_started.wait(), timeout=5)
        abandoned.cancel()
        with pytest.raises(asyncio.CancelledError):
            await abandoned

        fresh_provider = _ForkGroupProvider(fail_branch="beta")
        fresh_app = configured_app(fresh_provider)
        recovered = await fresh_app.run_fork_group(request)

        assert recovered.state is ForkGroupState.COMPLETED
        beta_attempts = [attempt for attempt in recovered.branches if attempt.branch_id == "beta"]
        assert [attempt.attempt_index for attempt in beta_attempts] == [0, 1, 2]
        assert beta_attempts[0].superseded_by_attempt_id == beta_attempts[1].attempt_id
        assert beta_attempts[1].superseded_by_attempt_id == beta_attempts[2].attempt_id
        assert beta_attempts[2].eligible is True
        assert all(
            attempt.execution_profile_fingerprint is not None for attempt in beta_attempts[1:]
        )
        assert fresh_provider.replacement_calls == 1
        assert fresh_provider.evaluator_calls == 1
        for attempt in beta_attempts[1:]:
            child = await store.load(attempt.session_id)
            assert child is not None

    asyncio.run(run())


def test_viable_replacement_replay_rejects_changed_planner_output() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        replacement_gated = asyncio.Event()
        hold_gate = asyncio.Event()

        class _BlockingReplacementGate(_ConfiguredGate):
            async def evaluate(self, request: ForkGroupGateRequest) -> ForkGroupGateDecision:
                decision = await super().evaluate(request)
                if request.branch.attempt_index == 1:
                    replacement_gated.set()
                    await hold_gate.wait()
                return decision

        def configured_app(
            provider: _ForkGroupProvider,
            planner: _ConfiguredReplacementPlanner,
            gate: ForkGroupGate,
        ) -> CayuApp:
            app = _app(provider, session_store=store)
            app.register_fork_group_gate("tests", gate)
            app.register_fork_group_replacement_planner("tests", planner)
            return app

        first_app = configured_app(
            _ForkGroupProvider(fail_branch="beta"),
            _ConfiguredReplacementPlanner(),
            _BlockingReplacementGate(),
        )
        await _source(first_app)
        request = _request(
            group_id="group-changed-replacement-replay",
            gates=(
                ForkGroupGateSelection(
                    gate_id="tests",
                    gate_identity="tests.configured-gate.v1",
                ),
            ),
        ).model_copy(
            update={
                "failure_policy": ForkGroupFailurePolicy(
                    mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                    minimum_viable_branches=2,
                    max_replacement_attempts=1,
                    replacement_parallelism=1,
                    replacement_planner=ForkGroupReplacementPlannerSelection(
                        planner_id="tests",
                        planner_identity="tests.configured-replacement-planner.v1",
                    ),
                )
            }
        )

        abandoned = asyncio.create_task(first_app.run_fork_group(request))
        await asyncio.wait_for(replacement_gated.wait(), timeout=5)
        abandoned.cancel()
        with pytest.raises(asyncio.CancelledError):
            await abandoned

        changed_app = configured_app(
            _ForkGroupProvider(fail_branch="beta"),
            _ConfiguredReplacementPlanner(
                message_suffix=" changed",
                artifact_references=(
                    ForkGroupArtifactReference(artifact_id="changed-planner-artifact"),
                ),
            ),
            _ConfiguredGate(),
        )
        with pytest.raises(ForkGroupConflict, match="conflicts with the fork-group request"):
            await changed_app.run_fork_group(request)

    asyncio.run(run())


def test_run_fork_group_fails_closed_on_deterministic_gate_failure() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        app = _app(provider)
        await _source(app)
        gate = _ConfiguredGate(failed_branch_id="beta")
        gate_selection = app.register_fork_group_gate("tests", gate)
        request = _request(
            group_id="group-gate-failure",
            gates=(gate_selection,),
        )

        result = await app.run_fork_group(request)

        assert result.state is ForkGroupState.FAILED
        assert result.failure is not None
        assert result.failure.code is ForkGroupFailureCode.GATE_FAILED
        assert result.failure.branch_id == "beta"
        assert all(branch.status == "completed" for branch in result.branches)
        assert [request.branch.branch_id for request in gate.requests] == ["alpha", "beta"]
        assert result.branches[0].gate_results == (
            ForkGroupGateResult(
                gate_id="tests",
                passed=True,
                summary="deterministic tests passed",
            ),
        )
        assert result.branches[1].gate_results[0].passed is False
        assert provider.evaluator_calls == 0

    asyncio.run(run())


def test_run_fork_group_fails_closed_when_gate_identity_changes() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        app = _app(provider)
        await _source(app)
        selection = app.register_fork_group_gate("tests", _ConfiguredGate())
        request = _request(
            group_id="group-gate-identity-mismatch",
            gates=(selection.model_copy(update={"gate_identity": "tests.configured-gate.v2"}),),
        )

        result = await app.run_fork_group(request)

        assert result.state is ForkGroupState.FAILED
        assert result.failure is not None
        assert result.failure.code is ForkGroupFailureCode.GATE_FAILED
        assert result.failure.branch_id == "alpha"
        assert provider.evaluator_calls == 0

    asyncio.run(run())


def test_gate_exception_is_redacted_before_result_and_durable_record() -> None:
    async def run() -> None:
        secret = "fork-group-gate-secret-canary"
        provider = _ForkGroupProvider()
        app = CayuApp(
            enable_logging=False,
            secret_redactor=SecretRedactor(secret),
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="source", model="fake-model"))
        app.register_agent(AgentSpec(name="evaluator", model="fake-model"))
        await _source(app)
        selection = app.register_fork_group_gate(
            "secret-failure",
            _SecretFailingGate(secret),
        )
        request = _request(
            group_id="group-secret-gate-failure",
            gates=(selection,),
        )

        result = await app.run_fork_group(request)
        durable = await app.session_store.load_session_operation(
            request.source_session_id,
            fork_group_runtime._storage_key(request.group_id),
        )

        assert result.state is ForkGroupState.FAILED
        assert result.failure is not None
        assert result.failure.code is ForkGroupFailureCode.GATE_FAILED
        assert secret not in result.failure.message
        assert secret not in json.dumps(durable)

    asyncio.run(run())


def test_branch_extension_exception_is_redacted_before_result_and_durable_record() -> None:
    async def run() -> None:
        secret = "fork-group-branch-extension-secret-canary"
        provider = _ForkGroupProvider()
        app = CayuApp(
            enable_logging=False,
            secret_redactor=SecretRedactor(secret),
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="source", model="fake-model"))
        app.register_agent(AgentSpec(name="evaluator", model="fake-model"))
        await _source(app)

        async def failing_usage(session_id: str) -> SessionUsageSummary:
            del session_id
            raise RuntimeError(f"usage extension failed {secret}")

        app._fork_group_coordinator._get_session_usage_callback = failing_usage
        request = _request(group_id="group-secret-branch-extension-failure")

        result = await app.run_fork_group(request)
        durable = await app.session_store.load_session_operation(
            request.source_session_id,
            fork_group_runtime._storage_key(request.group_id),
        )

        assert result.state is ForkGroupState.FAILED
        assert result.failure is not None
        assert result.failure.code is ForkGroupFailureCode.BRANCH_FAILED
        assert all(branch.error is not None for branch in result.branches)
        assert secret not in json.dumps(result.model_dump(mode="json"))
        assert secret not in json.dumps(durable)

    asyncio.run(run())


def test_fork_group_copy_suppresses_mutated_request_serializer_diagnostics(
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    class HostileGroupId:
        def __repr__(self) -> str:
            return "MUTATED_FORK_GROUP_SECRET_CANARY"

    async def run() -> None:
        provider = _ForkGroupProvider()
        app = _app(provider)
        await _source(app)
        request = _request(group_id="group-mutated-copy")
        object.__setattr__(request, "group_id", HostileGroupId())

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with pytest.raises(ValidationError):
                await app.run_fork_group(request)

        output = capsys.readouterr()
        rendered = repr(captured) + output.out + output.err + caplog.text
        assert "MUTATED_FORK_GROUP_SECRET_CANARY" not in rendered

    asyncio.run(run())


@pytest.mark.parametrize("invalid_judgment", ["duplicate", "multiple-selected"])
def test_run_fork_group_rejects_invalid_evaluator_judgment(
    invalid_judgment: str,
) -> None:
    async def run() -> None:
        provider = _ForkGroupProvider(invalid_judgment=invalid_judgment)
        app = _app(provider)
        await _source(app)

        result = await app.run_fork_group(_request(group_id=f"group-invalid-{invalid_judgment}"))

        assert result.state is ForkGroupState.FAILED
        assert result.failure is not None
        assert result.failure.code is ForkGroupFailureCode.JUDGMENT_INVALID
        assert result.dispositions == ()
        assert all(branch.status == "completed" for branch in result.branches)
        assert provider.evaluator_calls == 1

    asyncio.run(run())


def test_run_fork_group_preserves_branches_when_evaluator_fails() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider(fail_evaluator=True)
        app = _app(provider)
        await _source(app)

        result = await app.run_fork_group(_request(group_id="group-evaluator-failure"))

        assert result.state is ForkGroupState.FAILED
        assert result.failure is not None
        assert result.failure.code is ForkGroupFailureCode.EVALUATOR_FAILED
        assert all(branch.status == "completed" for branch in result.branches)
        assert result.dispositions == ()
        assert provider.evaluator_calls == 1

    asyncio.run(run())


def test_fork_group_request_rejects_unbounded_or_ambiguous_branches() -> None:
    valid = _request(group_id="group-validation")

    with pytest.raises(ValidationError, match="at least 2"):
        ForkGroupRequest.model_validate(
            {**valid.model_dump(mode="python"), "branches": valid.branches[:1]}
        )
    with pytest.raises(ValidationError, match="branch_ids must be unique"):
        ForkGroupRequest.model_validate(
            {
                **valid.model_dump(mode="python"),
                "branches": (
                    valid.branches[0],
                    valid.branches[1].model_copy(update={"branch_id": "alpha"}),
                ),
            }
        )
    with pytest.raises(ValidationError, match="structured_output or artifact_references"):
        ForkGroupBranchSpec(
            branch_id="no-evidence",
            session_id="no-evidence-session",
            messages=(Message.text("user", "run"),),
        )


def test_run_fork_group_keeps_prepared_siblings_on_one_snapshot_during_source_mutation() -> None:
    async def run() -> None:
        app: CayuApp

        async def mutate_source() -> None:
            await app.session_store.transition_status(
                "fork-group-source",
                from_statuses={SessionStatus.COMPLETED},
                to_status=SessionStatus.RUNNING,
            )
            await app.session_store.transition_status(
                "fork-group-source",
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.COMPLETED,
            )

        provider = _ForkGroupProvider(branch_callback=mutate_source)
        app = _app(provider)
        await _source(app)

        result = await app.run_fork_group(
            _request(group_id="group-source-mutation", max_parallelism=1)
        )

        assert result.state is ForkGroupState.COMPLETED
        assert result.failure is None
        assert [branch.status for branch in result.branches] == ["completed", "completed"]
        source = await app.session_store.load("fork-group-source")
        children = [
            await app.session_store.load("group-source-mutation-alpha"),
            await app.session_store.load("group-source-mutation-beta"),
        ]
        assert source is not None and source.run_epoch > result.source.run_epoch
        assert all(child is not None for child in children)
        relationships = [
            session_fork_profile_relationship(child) for child in children if child is not None
        ]
        assert all(
            relationship is not None
            and relationship.source_run_epoch == result.source.run_epoch
            and relationship.source_profile.fingerprint
            == result.source.execution_profile_fingerprint
            for relationship in relationships
        )
        assert provider.evaluator_calls == 1

    asyncio.run(run())


def test_run_fork_group_fails_closed_if_source_changes_while_siblings_are_prepared() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        app = _app(provider)
        await _source(app)
        original_fork_session = app.fork_session
        fork_count = 0

        async def fork_then_mutate(
            request: ForkSessionRequest,
        ) -> AsyncIterator[Event]:
            nonlocal fork_count
            async for event in original_fork_session(request):
                yield event
            fork_count += 1
            if fork_count == 1:
                await app.session_store.transition_status(
                    "fork-group-source",
                    from_statuses={SessionStatus.COMPLETED},
                    to_status=SessionStatus.RUNNING,
                )
                await app.session_store.transition_status(
                    "fork-group-source",
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.COMPLETED,
                )

        app.fork_session = fork_then_mutate  # ty: ignore[invalid-assignment]
        app._fork_group_coordinator._fork_session_callback = fork_then_mutate
        result = await app.run_fork_group(_request(group_id="group-source-fork-race"))

        assert result.state is ForkGroupState.FAILED
        assert result.failure is not None
        assert result.failure.code is ForkGroupFailureCode.SOURCE_CHANGED
        assert [branch.status for branch in result.branches] == ["interrupted", "failed"]
        assert await app.session_store.load("group-source-fork-race-alpha") is not None
        assert await app.session_store.load("group-source-fork-race-beta") is None
        assert len(provider.requests) == 1
        assert provider.evaluator_calls == 0

    asyncio.run(run())


def test_run_fork_group_fails_closed_on_same_epoch_checkpoint_mutation() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        app = _app(provider)
        await _source(app)
        original_fork_session = app.fork_session
        fork_count = 0

        async def fork_then_mutate_checkpoint(
            request: ForkSessionRequest,
        ) -> AsyncIterator[Event]:
            nonlocal fork_count
            async for event in original_fork_session(request):
                yield event
            fork_count += 1
            if fork_count == 1:
                checkpoint = await app.session_store.load_checkpoint("fork-group-source")
                mutated = {} if checkpoint is None else checkpoint
                mutated["same_epoch_semantic_mutation"] = {"candidate": "different"}
                await app.session_store.checkpoint("fork-group-source", mutated)

        app.fork_session = fork_then_mutate_checkpoint  # ty: ignore[invalid-assignment]
        app._fork_group_coordinator._fork_session_callback = fork_then_mutate_checkpoint
        result = await app.run_fork_group(_request(group_id="group-source-checkpoint-race"))

        assert result.state is ForkGroupState.FAILED
        assert result.failure is not None
        assert result.failure.code is ForkGroupFailureCode.SOURCE_CHANGED
        assert [branch.status for branch in result.branches] == ["interrupted", "failed"]
        source = await app.session_store.load("fork-group-source")
        assert source is not None and source.run_epoch == result.source.run_epoch
        assert await app.session_store.load("group-source-checkpoint-race-alpha") is not None
        assert await app.session_store.load("group-source-checkpoint-race-beta") is None
        assert len(provider.requests) == 1
        assert provider.evaluator_calls == 0

    asyncio.run(run())


def test_run_fork_group_rejects_existing_child_from_older_same_epoch_snapshot() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        app = _app(provider)
        await _source(app)
        request = _request(group_id="group-existing-stale-child")
        coordinator = app._fork_group_coordinator
        prepared = fork_group_runtime._prepare_request(
            coordinator,
            request,
            source_session_id=request.source_session_id,
        )
        source = await app.session_store.load(request.source_session_id)
        transcript = await app.session_store.load_transcript_snapshot(request.source_session_id)
        checkpoint = await app.session_store.load_checkpoint(request.source_session_id)
        assert source is not None
        profile = execution_profile_baseline_from_session_metadata(source.metadata)
        assert profile is not None
        old_snapshot = fork_group_runtime.ForkGroupSourceSnapshot(
            source_session_id=source.id,
            status=source.status,
            run_epoch=source.run_epoch,
            transcript_cursor=transcript.cursor,
            transcript_sha256=fork_group_runtime.session_input_messages_sha256(
                [record.message for record in transcript.records]
            ),
            checkpoint_sha256=fork_group_runtime.fork_source_checkpoint_sha256(checkpoint),
            execution_profile_fingerprint=profile.fingerprint,
            causal_budget_id=source.causal_budget_id,
        )
        fork_status, fork_error, fork_failure_code, _ = await fork_group_runtime._fork_exact_source(
            coordinator,
            fork_group_runtime._branch_fork_request(
                prepared,
                prepared.branches[0],
            ),
            source=old_snapshot,
        )
        assert fork_status is SessionStatus.COMPLETED
        assert fork_error is None
        assert fork_failure_code is None
        mutated = {} if checkpoint is None else checkpoint
        mutated["same_epoch_semantic_mutation"] = {"candidate": "newer"}
        await app.session_store.checkpoint(request.source_session_id, mutated)

        result = await app.run_fork_group(request)

        assert result.state is ForkGroupState.FAILED
        assert result.failure is not None
        assert result.failure.code is ForkGroupFailureCode.SOURCE_CHANGED
        assert result.failure.branch_id == "alpha"
        assert [branch.status for branch in result.branches] == ["failed", "interrupted"]
        assert await app.session_store.load(f"{request.group_id}-alpha") is not None
        assert await app.session_store.load(f"{request.group_id}-beta") is not None
        assert len(provider.requests) == 1
        assert provider.evaluator_calls == 0

    asyncio.run(run())


def test_stale_coordinator_cannot_replace_terminal_fork_group_result() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        app = _app(provider)
        await _source(app)
        request = _request(group_id="group-stale-publication")
        result = await app.run_fork_group(request)
        assert result.state is ForkGroupState.COMPLETED
        coordinator = app._fork_group_coordinator

        terminal = await fork_group_runtime._load_record(
            coordinator,
            request.source_session_id,
            request,
        )
        assert terminal is not None
        stale = terminal.model_copy(
            update={
                "revision": terminal.revision - 1,
                "result": terminal.result.model_copy(
                    update={
                        "state": ForkGroupState.AWAITING_EVALUATION,
                        "dispositions": (),
                        "failure": None,
                    },
                    deep=True,
                ),
            },
            deep=True,
        )
        attempted_failure = fork_group_runtime._result_with(
            stale,
            state=ForkGroupState.FAILED,
            failure=fork_group_runtime._failure(
                ForkGroupFailureCode.INTERNAL_ERROR,
                "stale coordinator failure",
            ),
        )

        authoritative = await fork_group_runtime._publish_record(
            coordinator,
            request.source_session_id,
            attempted_failure,
            EventType.FORK_GROUP_FAILED,
            expected_record=stale,
        )

        assert authoritative == terminal
        records = await app.session_store.query_events(
            EventQuery(
                session_id=request.source_session_id,
                event_types=("fork_group.completed", "fork_group.failed"),
            )
        )
        assert [record.event.type for record in records] == ["fork_group.completed"]

    asyncio.run(run())


def test_run_fork_group_reconciles_lost_terminal_publication_acknowledgement() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        app = _app(provider)
        await _source(app)
        request = _request(group_id="group-terminal-ack-loss")
        original_publish = app.session_store.publish_session_operation
        acknowledgement_lost = False

        async def publish_then_lose_terminal_ack(session_id: str, **kwargs):
            nonlocal acknowledgement_lost
            published = await original_publish(session_id, **kwargs)
            events = kwargs.get("events", [])
            if (
                not acknowledgement_lost
                and events
                and events[0].type is EventType.FORK_GROUP_COMPLETED
            ):
                acknowledgement_lost = True
                raise RuntimeError("simulated terminal acknowledgement loss")
            return published

        app.session_store.publish_session_operation = (  # ty: ignore[invalid-assignment]
            publish_then_lose_terminal_ack
        )
        completed = await app.run_fork_group(request)

        assert acknowledgement_lost is True
        assert completed.state is ForkGroupState.COMPLETED
        assert completed.failure is None
        request_count = len(provider.requests)
        replay = await app.run_fork_group(request)
        assert replay.replayed is True
        assert replay.state is ForkGroupState.COMPLETED
        assert len(provider.requests) == request_count
        records = await app.session_store.query_events(
            EventQuery(
                session_id=request.source_session_id,
                event_types=(EventType.FORK_GROUP_COMPLETED,),
            )
        )
        assert len(records) == 1

    asyncio.run(run())


def test_run_fork_group_retries_precommit_terminal_publication() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        app = _app(provider)
        await _source(app)
        request = _request(group_id="group-terminal-precommit-failure")
        original_publish = app.session_store.publish_session_operation
        publication_failed = False

        async def fail_before_terminal_publication(session_id: str, **kwargs):
            nonlocal publication_failed
            events = kwargs.get("events", [])
            if (
                not publication_failed
                and events
                and events[0].type is EventType.FORK_GROUP_COMPLETED
            ):
                publication_failed = True
                raise ConnectionError("simulated pre-commit terminal publication failure")
            return await original_publish(session_id, **kwargs)

        app.session_store.publish_session_operation = (  # ty: ignore[invalid-assignment]
            fail_before_terminal_publication
        )
        completed = await app.run_fork_group(request)

        assert publication_failed is True
        assert completed.state is ForkGroupState.COMPLETED
        assert completed.failure is None
        assert len(completed.dispositions) == 2
        request_count = len(provider.requests)

        replay = await app.run_fork_group(request)

        assert replay.replayed is True
        assert replay.dispositions == completed.dispositions
        assert len(provider.requests) == request_count

    asyncio.run(run())


def test_viable_group_retains_completed_attempts_when_awaiting_publication_fails() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        app = _app(provider)
        await _source(app)
        gate_selection = app.register_fork_group_gate("tests", _ConfiguredGate())
        planner_selection = app.register_fork_group_replacement_planner(
            "tests",
            _ConfiguredReplacementPlanner(),
        )
        request = _request(
            group_id="group-awaiting-publication-failure",
            gates=(gate_selection,),
        ).model_copy(
            update={
                "failure_policy": ForkGroupFailurePolicy(
                    mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                    minimum_viable_branches=1,
                    max_replacement_attempts=1,
                    replacement_parallelism=1,
                    replacement_planner=planner_selection,
                )
            }
        )
        original_publish = app.session_store.publish_session_operation
        publication_failed = False

        async def fail_before_awaiting_publication(session_id: str, **kwargs):
            nonlocal publication_failed
            events = kwargs.get("events", [])
            if (
                not publication_failed
                and events
                and events[0].type is EventType.FORK_GROUP_AWAITING_EVALUATION
            ):
                publication_failed = True
                raise ConnectionError("simulated pre-commit awaiting publication failure")
            return await original_publish(session_id, **kwargs)

        app.session_store.publish_session_operation = (  # ty: ignore[invalid-assignment]
            fail_before_awaiting_publication
        )
        failed = await app.run_fork_group(request)

        assert publication_failed is True
        assert failed.state is ForkGroupState.FAILED
        assert failed.failure is not None
        assert failed.failure.code is ForkGroupFailureCode.INTERNAL_ERROR
        assert "ConnectionError" in failed.failure.message
        assert [branch.branch_id for branch in failed.branches] == ["alpha", "beta"]
        assert all(branch.status is ForkGroupBranchStatus.COMPLETED for branch in failed.branches)
        request_count = len(provider.requests)

        replay = await app.run_fork_group(request)

        assert replay.replayed is True
        assert replay.state is ForkGroupState.FAILED
        assert replay.branches == failed.branches
        assert len(provider.requests) == request_count

    asyncio.run(run())


def test_stale_publication_failure_does_not_terminalize_successor_claim() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        current_time = [datetime(2026, 1, 1, tzinfo=UTC)]
        successor_evaluator_started = asyncio.Event()
        release_successor = asyncio.Event()

        async def block_successor_evaluator() -> None:
            successor_evaluator_started.set()
            await release_successor.wait()

        def shared_app(provider: _ForkGroupProvider) -> CayuApp:
            app = CayuApp(
                session_store=store,
                enable_logging=False,
                clock=lambda: current_time[0],
            )
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="source", model="fake-model"))
            app.register_agent(AgentSpec(name="evaluator", model="fake-model"))
            return app

        first_app = shared_app(_ForkGroupProvider())
        second_app = shared_app(_ForkGroupProvider(evaluator_callback=block_successor_evaluator))
        await _source(first_app)
        request = _request(group_id="group-stale-publication-failure")
        original_publish = store.publish_session_operation
        publication_failed = False
        successor_task: asyncio.Task[Any] | None = None

        async def fail_after_successor_claims(session_id: str, **kwargs):
            nonlocal publication_failed, successor_task
            events = kwargs.get("events", [])
            if (
                not publication_failed
                and events
                and events[0].type is EventType.FORK_GROUP_AWAITING_EVALUATION
            ):
                publication_failed = True
                current_time[0] += timedelta(hours=1)
                successor_task = asyncio.create_task(second_app.run_fork_group(request))
                await asyncio.wait_for(successor_evaluator_started.wait(), timeout=5)
                raise ConnectionError("simulated stale-owner publication failure")
            return await original_publish(session_id, **kwargs)

        store.publish_session_operation = (  # ty: ignore[invalid-assignment]
            fail_after_successor_claims
        )
        stale_result = await first_app.run_fork_group(request)
        durable_during_successor = await store.load_session_operation(
            request.source_session_id,
            fork_group_runtime._storage_key(request.group_id),
        )

        assert stale_result.state is ForkGroupState.AWAITING_EVALUATION
        assert durable_during_successor is not None
        assert durable_during_successor["result"]["state"] == "awaiting-evaluation"
        assert durable_during_successor["execution_claim"] is not None

        release_successor.set()
        assert successor_task is not None
        successor_result = await successor_task
        replay = await first_app.run_fork_group(request)

        assert successor_result.state is ForkGroupState.COMPLETED
        assert replay.state is ForkGroupState.COMPLETED
        assert replay.replayed is True

    asyncio.run(run())


def test_gate_failure_survives_failed_terminal_publication() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        app = _app(provider)
        await _source(app)
        gate_selection = app.register_fork_group_gate(
            "tests",
            _ConfiguredGate(failed_branch_id="beta"),
        )
        request = _request(
            group_id="group-gate-failure-publication-failure",
            gates=(gate_selection,),
        )
        original_publish = app.session_store.publish_session_operation
        publication_failed = False

        async def fail_before_terminal_publication(session_id: str, **kwargs):
            nonlocal publication_failed
            events = kwargs.get("events", [])
            if not publication_failed and events and events[0].type is EventType.FORK_GROUP_FAILED:
                publication_failed = True
                raise ConnectionError("simulated pre-commit failed publication")
            return await original_publish(session_id, **kwargs)

        app.session_store.publish_session_operation = (  # ty: ignore[invalid-assignment]
            fail_before_terminal_publication
        )
        failed = await app.run_fork_group(request)

        assert publication_failed is True
        assert failed.state is ForkGroupState.FAILED
        assert failed.failure is not None
        assert failed.failure.code is ForkGroupFailureCode.GATE_FAILED
        assert failed.failure.branch_id == "beta"
        assert [branch.branch_id for branch in failed.branches] == ["alpha", "beta"]
        assert failed.branches[1].gate_results[0].passed is False

        replay = await app.run_fork_group(request)

        assert replay.replayed is True
        assert replay.failure == failed.failure
        assert replay.branches == failed.branches

    asyncio.run(run())


def test_run_fork_group_recovers_nonterminal_branch_before_classification() -> None:
    async def run() -> None:
        initial_provider = _ForkGroupProvider()
        initial_app = _app(initial_provider)
        await _source(initial_app)
        request = _request(group_id="group-process-loss")
        coordinator = initial_app._fork_group_coordinator
        prepared = fork_group_runtime._prepare_request(
            coordinator,
            request,
            source_session_id=request.source_session_id,
        )
        record = await fork_group_runtime._create_record(coordinator, prepared)
        record = await fork_group_runtime._publish_record(
            coordinator,
            request.source_session_id,
            fork_group_runtime._result_with(
                record,
                state=ForkGroupState.BRANCHES_RUNNING,
            ),
            EventType.FORK_GROUP_BRANCHES_RUNNING,
            expected_record=record,
        )
        for branch in prepared.branches:
            outcome = await fork_group_runtime._prepare_branch_fork(
                coordinator,
                prepared,
                record.result.source,
                branch,
            )
            assert outcome is None
        await initial_app.session_store.transition_status(
            "group-process-loss-alpha",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
        )

        recovery_provider = _ForkGroupProvider()
        recovered_app = CayuApp(
            session_store=initial_app.session_store,
            enable_logging=False,
        )
        recovered_app.register_provider(recovery_provider, default=True)
        recovered_app.register_agent(AgentSpec(name="source", model="fake-model"))
        recovered_app.register_agent(AgentSpec(name="evaluator", model="fake-model"))
        recovered_session_ids: list[str] = []
        original_recover = recovered_app.recover_incomplete_session

        async def capture_recovery(recovery_request):
            recovered_session_ids.append(recovery_request.session_id)
            return await original_recover(recovery_request)

        recovered_app.recover_incomplete_session = capture_recovery  # ty: ignore[invalid-assignment]
        recovered_app._fork_group_coordinator._recover_incomplete_session_callback = (
            capture_recovery
        )
        result = await recovered_app.run_fork_group(request)

        assert result.state is ForkGroupState.FAILED
        assert result.failure is not None
        assert result.failure.code is ForkGroupFailureCode.BRANCH_INVALID
        assert result.failure.branch_id == "alpha"
        assert recovered_session_ids == ["group-process-loss-alpha"]
        assert [branch.status for branch in result.branches] == ["invalid", "completed"]
        assert recovery_provider.evaluator_calls == 0

        replay_request_count = len(recovery_provider.requests)
        replay = await recovered_app.run_fork_group(request)
        assert replay.replayed is True
        assert replay.state is ForkGroupState.FAILED
        assert len(recovery_provider.requests) == replay_request_count

    asyncio.run(run())


def test_run_fork_group_keeps_group_nonterminal_while_branch_provider_is_pending() -> None:
    async def run() -> None:
        initial_provider = _ForkGroupProvider()
        initial_app = _app(initial_provider)
        await _source(initial_app)
        request = _request(group_id="group-pending-branch")
        coordinator = initial_app._fork_group_coordinator
        prepared = fork_group_runtime._prepare_request(
            coordinator,
            request,
            source_session_id=request.source_session_id,
        )
        record = await fork_group_runtime._create_record(coordinator, prepared)
        record = await fork_group_runtime._publish_record(
            coordinator,
            request.source_session_id,
            fork_group_runtime._result_with(
                record,
                state=ForkGroupState.BRANCHES_RUNNING,
            ),
            EventType.FORK_GROUP_BRANCHES_RUNNING,
            expected_record=record,
        )
        for branch in prepared.branches:
            assert (
                await fork_group_runtime._prepare_branch_fork(
                    coordinator,
                    prepared,
                    record.result.source,
                    branch,
                )
                is None
            )
        await initial_app.session_store.transition_status(
            "group-pending-branch-alpha",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
        )

        recovery_provider = _ForkGroupProvider()
        recovered_app = CayuApp(
            session_store=initial_app.session_store,
            enable_logging=False,
        )
        recovered_app.register_provider(recovery_provider, default=True)
        recovered_app.register_agent(AgentSpec(name="source", model="fake-model"))
        recovered_app.register_agent(AgentSpec(name="evaluator", model="fake-model"))
        recovered_session_ids: list[str] = []

        async def keep_provider_pending(recovery_request):
            recovered_session_ids.append(recovery_request.session_id)
            return IncompleteSessionRecoveryResult(
                session_id=recovery_request.session_id,
                previous_status=SessionStatus.RUNNING,
                status=SessionStatus.RUNNING,
                actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                message="Provider operation remains pending.",
            )

        recovered_app.recover_incomplete_session = (  # ty: ignore[invalid-assignment]
            keep_provider_pending
        )
        recovered_app._fork_group_coordinator._recover_incomplete_session_callback = (
            keep_provider_pending
        )
        pending = await recovered_app.run_fork_group(request)

        assert pending.state is ForkGroupState.BRANCHES_RUNNING
        assert pending.failure is None
        assert pending.dispositions == ()
        assert recovered_session_ids == ["group-pending-branch-alpha"]
        assert recovery_provider.evaluator_calls == 0
        request_count = len(recovery_provider.requests)

        replay = await recovered_app.run_fork_group(request)
        assert replay.state is ForkGroupState.BRANCHES_RUNNING
        assert replay.replayed is True
        assert replay.failure is None
        assert recovered_session_ids == [
            "group-pending-branch-alpha",
            "group-pending-branch-alpha",
        ]
        assert len(recovery_provider.requests) == request_count
        records = await recovered_app.session_store.query_events(
            EventQuery(
                session_id=request.source_session_id,
                event_types=(EventType.FORK_GROUP_FAILED,),
            )
        )
        assert records == []

    asyncio.run(run())


def test_run_fork_group_keeps_group_nonterminal_while_evaluator_provider_is_pending() -> None:
    async def run() -> None:
        initial_provider = _ForkGroupProvider()
        initial_app = _app(initial_provider)
        await _source(initial_app)
        request = _request(group_id="group-evaluator-process-loss")
        coordinator = initial_app._fork_group_coordinator
        prepared = fork_group_runtime._prepare_request(
            coordinator,
            request,
            source_session_id=request.source_session_id,
        )
        record = await fork_group_runtime._create_record(coordinator, prepared)
        record = await fork_group_runtime._publish_record(
            coordinator,
            request.source_session_id,
            fork_group_runtime._result_with(
                record,
                state=ForkGroupState.BRANCHES_RUNNING,
            ),
            EventType.FORK_GROUP_BRANCHES_RUNNING,
            expected_record=record,
        )
        for branch in prepared.branches:
            assert (
                await fork_group_runtime._prepare_branch_fork(
                    coordinator,
                    prepared,
                    record.result.source,
                    branch,
                )
                is None
            )
        branch_results = tuple(
            [
                await fork_group_runtime._run_branch(
                    coordinator,
                    prepared,
                    record.result.source,
                    branch,
                )
                for branch in prepared.branches
            ]
        )
        awaiting_record = fork_group_runtime._result_with(
            record,
            state=ForkGroupState.AWAITING_EVALUATION,
            branches=branch_results,
            evaluator_session_id=prepared.evaluator.session_id,
        )
        awaiting_record = await fork_group_runtime._bind_tool_free_evaluator_authority(
            coordinator,
            awaiting_record,
        )
        awaiting = await fork_group_runtime._publish_record(
            coordinator,
            request.source_session_id,
            awaiting_record,
            EventType.FORK_GROUP_AWAITING_EVALUATION,
            expected_record=record,
        )
        synthetic_name = fork_group_runtime._synthetic_evaluator_name(awaiting)
        evaluator_events = [
            event
            async for event in initial_app.run(
                fork_group_runtime._evaluator_run_request(
                    awaiting,
                    synthetic_agent_name=synthetic_name,
                    branch_identities=tuple(
                        fork_group_runtime._ForkGroupAttemptIdentity(
                            branch_id=branch.branch_id,
                            attempt_id=branch.attempt_id,
                        )
                        for branch in awaiting.result.branches
                    ),
                )
            )
        ]
        assert evaluator_events[-1].type is EventType.SESSION_COMPLETED
        await initial_app.session_store.transition_status(
            prepared.evaluator.session_id,
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
        )

        recovery_provider = _ForkGroupProvider()
        recovered_app = CayuApp(
            session_store=initial_app.session_store,
            enable_logging=False,
        )
        recovered_app.register_provider(recovery_provider, default=True)
        recovered_app.register_agent(AgentSpec(name="source", model="fake-model"))
        recovered_app.register_agent(AgentSpec(name="evaluator", model="fake-model"))
        recovered_session_ids: list[str] = []

        async def keep_provider_pending(recovery_request):
            recovered_session_ids.append(recovery_request.session_id)
            return IncompleteSessionRecoveryResult(
                session_id=recovery_request.session_id,
                previous_status=SessionStatus.RUNNING,
                status=SessionStatus.RUNNING,
                actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                message="Provider operation remains pending.",
            )

        recovered_app.recover_incomplete_session = (  # ty: ignore[invalid-assignment]
            keep_provider_pending
        )
        recovered_app._fork_group_coordinator._recover_incomplete_session_callback = (
            keep_provider_pending
        )
        result = await recovered_app.run_fork_group(request)

        assert result.state is ForkGroupState.AWAITING_EVALUATION
        assert result.failure is None
        assert result.dispositions == ()
        assert recovered_session_ids == [prepared.evaluator.session_id]
        assert all(branch.status == "completed" for branch in result.branches)
        assert recovery_provider.requests == []

        replay = await recovered_app.run_fork_group(request)
        assert replay.state is ForkGroupState.AWAITING_EVALUATION
        assert replay.replayed is True
        assert replay.failure is None
        assert recovered_session_ids == [
            prepared.evaluator.session_id,
            prepared.evaluator.session_id,
        ]
        assert recovery_provider.requests == []

    asyncio.run(run())


def test_run_fork_group_rejects_changed_evaluator_profile_after_restart() -> None:
    async def run() -> None:
        initial_provider = _ForkGroupProvider()
        initial_app = _app(initial_provider)
        await _source(initial_app)
        request = _request(group_id="group-evaluator-profile-drift")
        coordinator = initial_app._fork_group_coordinator
        prepared = fork_group_runtime._prepare_request(
            coordinator,
            request,
            source_session_id=request.source_session_id,
        )
        record = await fork_group_runtime._create_record(coordinator, prepared)
        record = await fork_group_runtime._publish_record(
            coordinator,
            request.source_session_id,
            fork_group_runtime._result_with(
                record,
                state=ForkGroupState.BRANCHES_RUNNING,
            ),
            EventType.FORK_GROUP_BRANCHES_RUNNING,
            expected_record=record,
        )
        for branch in prepared.branches:
            assert (
                await fork_group_runtime._prepare_branch_fork(
                    coordinator,
                    prepared,
                    record.result.source,
                    branch,
                )
                is None
            )
        branch_results = tuple(
            [
                await fork_group_runtime._run_branch(
                    coordinator,
                    prepared,
                    record.result.source,
                    branch,
                )
                for branch in prepared.branches
            ]
        )
        awaiting_record = fork_group_runtime._result_with(
            record,
            state=ForkGroupState.AWAITING_EVALUATION,
            branches=branch_results,
            evaluator_session_id=prepared.evaluator.session_id,
        )
        awaiting_record = await fork_group_runtime._bind_tool_free_evaluator_authority(
            coordinator,
            awaiting_record,
        )
        awaiting = await fork_group_runtime._publish_record(
            coordinator,
            request.source_session_id,
            awaiting_record,
            EventType.FORK_GROUP_AWAITING_EVALUATION,
            expected_record=record,
        )
        assert awaiting.result.state is ForkGroupState.AWAITING_EVALUATION

        changed_provider = _ForkGroupProvider()
        changed_app = CayuApp(
            session_store=initial_app.session_store,
            enable_logging=False,
        )
        changed_app.register_provider(changed_provider, default=True)
        changed_app.register_agent(AgentSpec(name="source", model="fake-model"))
        changed_app.register_agent(AgentSpec(name="evaluator", model="changed-model"))

        with pytest.raises(ForkGroupConflict, match="Evaluator execution profile changed"):
            await changed_app.run_fork_group(request)

        durable = await fork_group_runtime._load_record(
            changed_app._fork_group_coordinator,
            request.source_session_id,
            prepared,
        )
        assert durable is not None
        assert durable.result.state is ForkGroupState.AWAITING_EVALUATION
        assert changed_provider.evaluator_calls == 0

    asyncio.run(run())


def test_run_fork_group_enforces_bounded_parallelism() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider(branch_delay=0.03)
        app = _app(provider)
        await _source(app)

        result = await app.run_fork_group(
            _request(
                group_id="group-parallelism",
                max_parallelism=2,
                extra_alpha=True,
            )
        )

        assert result.state is ForkGroupState.COMPLETED
        assert len(result.branches) == 3
        assert provider.max_active_branches == 2

    asyncio.run(run())


def test_run_fork_group_reports_shared_causal_budget_exhaustion() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        app = CayuApp(enable_logging=False, budget_ledger=InMemoryBudgetLedger())
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="source", model="fake-model"))
        app.register_agent(AgentSpec(name="evaluator", model="fake-model"))
        await _source(app)
        gate_selection = app.register_fork_group_gate("tests", _ConfiguredGate())
        planner_selection = app.register_fork_group_replacement_planner(
            "tests",
            _ConfiguredReplacementPlanner(),
        )
        limit = BudgetLimit(
            scope="causal",
            key="fork-group-budget",
            max_estimated_cost=Decimal("0.00007"),
            pricing=PriceBook(
                prices=(
                    ModelPrice.fixed(
                        provider_name="fork-group-fake",
                        model="fake-model",
                        input_per_million=Decimal("1"),
                        output_per_million=Decimal("10"),
                    ),
                )
            ),
            reservation=BudgetReservation(
                max_input_tokens=1,
                max_output_tokens=1,
            ),
        )
        request = _request(
            group_id="group-budget",
            max_parallelism=1,
            gates=(gate_selection,),
        )
        request = request.model_copy(
            update={
                "branches": tuple(
                    branch.model_copy(
                        update={
                            "budget_limits": (limit,),
                            "structured_output": None,
                            "artifact_references": (
                                ForkGroupArtifactReference(
                                    artifact_id=f"artifact-{branch.branch_id}"
                                ),
                            ),
                        },
                        deep=True,
                    )
                    for branch in request.branches
                ),
                "failure_policy": ForkGroupFailurePolicy(
                    mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                    minimum_viable_branches=2,
                    max_replacement_attempts=1,
                    replacement_parallelism=1,
                    replacement_planner=planner_selection,
                ),
            },
            deep=True,
        )

        result = await app.run_fork_group(request)

        assert result.state is ForkGroupState.FAILED
        assert result.failure is not None
        assert result.failure.code is ForkGroupFailureCode.BUDGET_EXHAUSTED
        assert result.failure.branch_id == "beta"
        assert result.branches[0].status == "completed"
        assert result.branches[1].status == "interrupted"
        assert result.branches[1].failure_code is ForkGroupFailureCode.BUDGET_EXHAUSTED
        assert provider.replacement_calls == 0
        assert provider.evaluator_calls == 0

    asyncio.run(run())


def test_task_backed_fork_group_reports_shared_causal_budget_exhaustion() -> None:
    async def run() -> None:
        provider = _ForkGroupProvider()
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(task_store)
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            dispatcher=dispatcher,
            enable_logging=False,
            budget_ledger=InMemoryBudgetLedger(),
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="source", model="fake-model"))
        app.register_agent(AgentSpec(name="evaluator", model="fake-model"))
        await _source(app)
        gate_selection = app.register_fork_group_gate("tests", _ConfiguredGate())
        planner_selection = app.register_fork_group_replacement_planner(
            "tests",
            _ConfiguredReplacementPlanner(),
        )
        limit = BudgetLimit(
            scope="causal",
            key="fork-group-budget",
            max_estimated_cost=Decimal("0.00007"),
            pricing=PriceBook(
                prices=(
                    ModelPrice.fixed(
                        provider_name="fork-group-fake",
                        model="fake-model",
                        input_per_million=Decimal("1"),
                        output_per_million=Decimal("10"),
                    ),
                )
            ),
            reservation=BudgetReservation(
                max_input_tokens=1,
                max_output_tokens=1,
            ),
        )
        request = _request(
            group_id="group-task-budget",
            max_parallelism=1,
            gates=(gate_selection,),
        )
        request = request.model_copy(
            update={
                "execution_mode": ForkGroupExecutionMode.TASK_DISPATCH,
                "branches": tuple(
                    branch.model_copy(
                        update={
                            "budget_limits": (limit,),
                            "structured_output": None,
                            "artifact_references": (
                                ForkGroupArtifactReference(
                                    artifact_id=f"artifact-{branch.branch_id}"
                                ),
                            ),
                        },
                        deep=True,
                    )
                    for branch in request.branches
                ),
                "failure_policy": ForkGroupFailurePolicy(
                    mode=ForkGroupFailureMode.EVALUATE_VIABLE,
                    minimum_viable_branches=2,
                    max_replacement_attempts=1,
                    replacement_parallelism=1,
                    replacement_planner=planner_selection,
                ),
            },
            deep=True,
        )

        result = await app.run_fork_group(request)
        assert result.state is ForkGroupState.BRANCHES_RUNNING
        assert await dispatcher.process_next(app, worker_id="worker-alpha") is not None
        result = await app.run_fork_group(request)
        assert result.state is ForkGroupState.BRANCHES_RUNNING
        assert await dispatcher.process_next(app, worker_id="worker-beta") is not None

        result = await app.run_fork_group(request)

        assert result.state is ForkGroupState.FAILED
        assert result.failure is not None
        assert result.failure.code is ForkGroupFailureCode.BUDGET_EXHAUSTED
        assert result.failure.branch_id == "beta"
        assert result.branches[0].status is ForkGroupBranchStatus.COMPLETED
        assert result.branches[1].status is ForkGroupBranchStatus.INTERRUPTED
        assert result.branches[1].failure_code is ForkGroupFailureCode.BUDGET_EXHAUSTED
        assert provider.replacement_calls == 0
        assert provider.evaluator_calls == 0

    asyncio.run(run())
