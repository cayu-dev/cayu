from __future__ import annotations

import asyncio
import json
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable
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
    ExecutionProfileMismatchError,
    ForkGroupArtifactReference,
    ForkGroupBranchSpec,
    ForkGroupCheckpointSelector,
    ForkGroupConflict,
    ForkGroupDisposition,
    ForkGroupEvaluatorSpec,
    ForkGroupFailureCode,
    ForkGroupGate,
    ForkGroupGateDecision,
    ForkGroupGateRequest,
    ForkGroupGateResult,
    ForkGroupGateSelection,
    ForkGroupRequest,
    ForkGroupState,
    ForkSessionRequest,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryResult,
    InMemoryBudgetLedger,
    InMemorySessionStore,
    Message,
    ModelPrice,
    PriceBook,
    RunRequest,
    StructuredOutputSpec,
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


class _SecretFailingGate(ForkGroupGate):
    def __init__(self, secret: str) -> None:
        self.secret = secret

    @property
    def identity(self) -> str:
        return "tests.secret-failing-gate.v1"

    async def evaluate(self, request: ForkGroupGateRequest) -> ForkGroupGateDecision:
        del request
        raise RuntimeError(f"gate rejected {self.secret}")


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
        branch_callback: Callable[[], Awaitable[None]] | None = None,
        branch_delay: float = 0,
    ) -> None:
        self.requests: list[ModelRequest] = []
        self.evaluator_tools: tuple[str, ...] | None = None
        self.evaluator_evidence: dict[str, Any] | None = None
        self.evaluator_calls = 0
        self.fail_branch = fail_branch
        self.fail_evaluator = fail_evaluator
        self.invalid_judgment = invalid_judgment
        self.branch_callback = branch_callback
        self.branch_delay = branch_delay
        self.callback_called = False
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
        if "cayu.fork-group-evidence.v1" in user_text:
            self.evaluator_calls += 1
            self.evaluator_tools = tuple(tool["name"] for tool in request.tools)
            if self.fail_evaluator:
                yield ModelStreamEvent.error("evaluator failed")
                return
            evidence = json.loads(user_text)
            self.evaluator_evidence = evidence
            branch_ids = [branch["branch_id"] for branch in evidence["branches"]]
            if self.invalid_judgment == "duplicate":
                output = {
                    "dispositions": [
                        {
                            "branch_id": branch_ids[0],
                            "disposition": "selected",
                            "reason": "duplicate judgment",
                        },
                        {
                            "branch_id": branch_ids[0],
                            "disposition": "rejected",
                            "reason": "duplicate judgment",
                        },
                    ]
                }
            elif self.invalid_judgment == "multiple-selected":
                output = {
                    "dispositions": [
                        {
                            "branch_id": branch_id,
                            "disposition": "selected",
                            "reason": "invalid judgment",
                        }
                        for branch_id in branch_ids
                    ]
                }
            else:
                output = {
                    "dispositions": [
                        {
                            "branch_id": branch_id,
                            "disposition": "selected" if index == 0 else "rejected",
                            "reason": "deterministic test judgment",
                        }
                        for index, branch_id in enumerate(branch_ids)
                    ]
                }
        elif "candidate alpha" in user_text:
            if self.fail_branch == "alpha":
                yield ModelStreamEvent.error("alpha failed")
                return
            await self._enter_branch()
            try:
                output = {"candidate": "alpha", "score": 2}
            finally:
                self.active_branches -= 1
        elif "candidate beta" in user_text:
            if self.fail_branch == "beta":
                yield ModelStreamEvent.error("beta failed")
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

    async def _enter_branch(self) -> None:
        self.active_branches += 1
        self.max_active_branches = max(self.max_active_branches, self.active_branches)
        if self.branch_callback is not None and not self.callback_called:
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


def _app(provider: _ForkGroupProvider) -> CayuApp:
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="source", model="fake-model"))
    app.register_agent(AgentSpec(name="evaluator", model="fake-model"))
    return app


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
        assert terminal_payload["schema_version"] == 1
        assert terminal_payload["selected_branch_id"] == "alpha"
        assert terminal_payload["dispositions"] == [
            {
                "branch_id": "alpha",
                "disposition": "selected",
                "reason": "deterministic test judgment",
            },
            {
                "branch_id": "beta",
                "disposition": "rejected",
                "reason": "deterministic test judgment",
            },
        ]
        assert terminal_payload["branches"] == [
            {
                "branch_id": "alpha",
                "session_id": "group-replay-alpha",
                "status": "completed",
            },
            {
                "branch_id": "beta",
                "session_id": "group-replay-beta",
                "status": "completed",
            },
        ]

        with pytest.raises(ForkGroupConflict, match="different request"):
            await app.run_fork_group(request.model_copy(update={"max_parallelism": 1}))
        assert len(provider.requests) == request_count

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
        fork_status, fork_error = await fork_group_runtime._fork_exact_source(
            coordinator,
            fork_group_runtime._branch_fork_request(
                prepared,
                prepared.branches[0],
            ),
            source=old_snapshot,
        )
        assert fork_status is SessionStatus.COMPLETED
        assert fork_error is None
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
        awaiting = await fork_group_runtime._publish_record(
            coordinator,
            request.source_session_id,
            fork_group_runtime._result_with(
                record,
                state=ForkGroupState.AWAITING_EVALUATION,
                branches=branch_results,
                evaluator_session_id=prepared.evaluator.session_id,
            ),
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
                    branch_ids=tuple(branch.branch_id for branch in awaiting.result.branches),
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
        awaiting = await fork_group_runtime._publish_record(
            coordinator,
            request.source_session_id,
            fork_group_runtime._result_with(
                record,
                state=ForkGroupState.AWAITING_EVALUATION,
                branches=branch_results,
                evaluator_session_id=prepared.evaluator.session_id,
            ),
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
        request = _request(group_id="group-budget", max_parallelism=1)
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
                )
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
        assert provider.evaluator_calls == 0

    asyncio.run(run())
