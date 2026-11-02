"""What sharing a Lambda MicroVM across workflow steps does today.

Both Lambda MicroVM factories allocate a process-external resource and declare
an allocation scope, so they may only be created through ``create_recoverable``
under a durable intent. The shared-allocation path
(``cayu.runtime._shared_environment_allocations``) creates such a factory's
allocation that way, with the intent kept in the workflow run's shared record,
never with a plain ``create()``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from tests.core.test_human_review import identity
from tests.environments.test_aws_lambda_microvm_factory import (
    REGION,
    LambdaWorld,
    WritePlan,
    call,
    done,
)
from tests.runners.lambda_microvm_harness import ClientTokenLambdaModel

from cayu import AgentSpec, EventType
from cayu.egress import (
    ApprovedEgressDestination,
    HttpEgressPolicy,
    VirtualEgressEnvironmentFactory,
)
from cayu.egress.aws_lambda_microvm_adapter import LambdaMicroVMEgressAdapter
from cayu.egress.proxy_exposure import VpcTaskProxyExposure
from cayu.environments import (
    AllocationSharingScope,
    EnvironmentFactoryOperation,
    EnvironmentSpec,
)
from cayu.runtime._shared_allocation_intents import SharedAllocationIntentContext
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.workflows import StepRunOptions, WorkflowBase, WorkflowSpec, step

_SHARED = StepRunOptions(allocation_scope=AllocationSharingScope.PARENT)


def _workflow(agent: str):
    class _OneStep(WorkflowBase):
        spec = WorkflowSpec(name=f"{agent}-step")

        async def run(self, session_id: str):
            ctx = self.context(session_id)
            yield await ctx.start()
            await step(ctx, agent=agent, step_id="plan", prompt="plan", run_options=_SHARED)
            yield await ctx.completed()

    return _OneStep


def _chain(error: BaseException) -> str:
    text = []
    seen: BaseException | None = error
    while seen is not None:
        text.append(repr(seen))
        seen = seen.__cause__ or seen.__context__
    return " ".join(text)


def _run_refused(world: LambdaWorld, app, agent: str) -> str:
    async def scenario() -> str:
        try:
            with pytest.raises(Exception) as raised:
                async for _ in _workflow(agent)(app).run("run-shared"):
                    pass
            return _chain(raised.value)
        finally:
            await world.close()

    return asyncio.run(scenario())


def test_a_shared_lambda_microvm_is_created_recoverably(tmp_path: Path) -> None:
    world = LambdaWorld(tmp_path)
    app = world.app(call("write_plan"), done(), agents={"planner": [WritePlan()]})
    plain_creates: list[object] = []
    factory = app._get_registered_environment("sandbox").factory
    original = factory.create

    async def create(request):
        plain_creates.append(request.operation)
        return await original(request)

    epochs: list[int | None] = []
    original_recoverable = factory.create_recoverable

    async def create_recoverable(request, allocation):
        epochs.append(request.ownership_epoch)
        return await original_recoverable(request, allocation)

    cast("Any", factory).create = create
    cast("Any", factory).create_recoverable = create_recoverable

    async def scenario():
        try:
            return [event async for event in _workflow("planner")(app).run("run-shared")]
        finally:
            await world.close()

    events = asyncio.run(scenario())

    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    # One MicroVM for the scope, created through create_recoverable: the
    # factory's plain create never saw a CREATE.
    (microvm,) = world.model.created_ids()
    assert all(operation is not EnvironmentFactoryOperation.CREATE for operation in plain_creates)
    # The request names the record's claim generation, so the sidecar's owner
    # claim is fenced by it.
    assert epochs == [1]
    assert world.state(microvm) == "TERMINATED"


def test_the_virtual_egress_lambda_factory_shares_through_create_recoverable(
    tmp_path: Path,
) -> None:
    model = ClientTokenLambdaModel()
    factory = VirtualEgressEnvironmentFactory(
        policies={
            "receiver": HttpEgressPolicy(
                name="receiver",
                allowed_hosts=["receiver.example"],
                allowed_endpoints=[("POST", "/v1/actions")],
            )
        },
        approved_destinations=[
            ApprovedEgressDestination(destination="receiver.example", policy_name="receiver")
        ],
        adapter=LambdaMicroVMEgressAdapter(
            region_name=REGION,
            egress_network_connector_arn="arn:aws:lambda:us-east-1:123:network-connector:nc-1",
            exposure=VpcTaskProxyExposure("10.0.1.20"),
            client=model,
        ),
        image=ClientTokenLambdaModel.image_arn,
    )
    world = LambdaWorld(tmp_path)
    app = world.app(call("write_plan"), done(), agents={})
    app.register_environment_factory(
        EnvironmentSpec(name="egress-sandbox"), factory, allocation_scope="parent"
    )
    app.register_agent(
        AgentSpec(name="egress-planner", model="scripted"),
        tools=[WritePlan()],
        environment_name="egress-sandbox",
    )

    plain_creates: list[object] = []
    recoverable: list[tuple[Any, Any]] = []

    async def create(request):
        plain_creates.append(request.operation)
        raise AssertionError("a shared MicroVM is never created through plain create()")

    async def create_recoverable(request, allocation):
        recoverable.append((request, allocation))
        # Stop before the MicroVM is started: what matters here is that the
        # shared path hands the factory its intent, not the guest's readiness.
        raise RuntimeError("stopped before the provider call")

    cast("Any", factory).create = create
    cast("Any", factory).create_recoverable = create_recoverable

    text = _run_refused(world, app, "egress-planner")

    # Its egress authority follows the allocation, so it isn't refused for
    # that, and its creation goes through create_recoverable under the
    # intent kept in the run's shared record.
    assert "stopped before the provider call" in text
    assert "egress authority" not in text
    assert plain_creates == []
    ((request, allocation),) = recoverable
    assert isinstance(allocation, SharedAllocationIntentContext)
    assert request.operation is EnvironmentFactoryOperation.CREATE
    assert request.ownership_epoch == 1
    # Its grants' audit events belong on the run that owns the scope.
    assert request.event_session_id == "run-shared"
    assert model.run_calls == []


class _ReadPlan(Tool):
    spec = ToolSpec(
        name="read_plan",
        parallel_safe=False,
        workspace_mutation=False,
        execution_profile_identity=identity("read-plan"),
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        assert ctx.workspace is not None
        await ctx.workspace.read_bytes("plan.md")
        return ToolResult(content="read the plan")


class _PlanThenSolve(WorkflowBase):
    spec = WorkflowSpec(name="plan-then-solve")

    def __init__(self, app, *, solve: bool, complete: bool = True) -> None:
        super().__init__(app)
        self._solve = solve
        self._complete = complete

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        await step(ctx, agent="planner", step_id="plan", prompt="plan", run_options=_SHARED)
        if self._solve:
            await step(ctx, agent="solver", step_id="solve", prompt="solve", run_options=_SHARED)
            if self._complete:
                yield await ctx.completed()


def _checkpoint_records(checkpoint: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = checkpoint.get("shared_environment_allocations") or {}
    return raw.get("records") or {}, raw.get("orphans") or {}


async def _interrupt_after_plan(world: LambdaWorld) -> dict[str, Any]:
    app = world.app(call("write_plan"), done(), agents={"planner": [WritePlan()]})
    async for _ in _PlanThenSolve(app, solve=False).run("run-interrupted"):
        pass
    # What a cancelled WorkflowBase.execute does: the owner releases with
    # outcome "interrupted", and the binding only suspends the MicroVM.
    await app._environment_lifecycle.shared_allocations.end_owner(
        "run-interrupted", outcome="interrupted"
    )
    return await app._runtime_session_store.load_checkpoint("run-interrupted") or {}


def test_an_interrupted_shared_lambda_microvm_stays_suspended_for_the_run_to_resume(
    tmp_path: Path,
) -> None:
    world = LambdaWorld(tmp_path)

    async def scenario():
        try:
            interrupted = await _interrupt_after_plan(world)
            # A second app on the same stores resumes the run: the solve step
            # reconnects to the suspended MicroVM and reads the planner's file.
            app = world.app(
                call("read_plan"),
                done(),
                agents={"planner": [WritePlan()], "solver": [_ReadPlan()]},
            )
            async for _ in _PlanThenSolve(app, solve=True).run("run-interrupted"):
                pass
            finished = await app._runtime_session_store.load_checkpoint("run-interrupted") or {}
            return interrupted, finished
        finally:
            await world.close()

    interrupted, finished = asyncio.run(scenario())
    (microvm,) = world.model.created_ids()
    records, orphans = _checkpoint_records(interrupted)
    (record,) = records.values()
    # The interrupted release kept the MicroVM and its record for the resume,
    # with the binding state that release left.
    assert record["kept_for_resume"] is True
    assert record["binding_state"]["microvm_id"] == microvm
    assert orphans == {}
    # The resume reconnected to that MicroVM rather than creating another,
    # and the completed run then terminated it and forgot its record.
    assert world.model.created_ids() == [microvm]
    assert world.state(microvm) == "TERMINATED"
    assert _checkpoint_records(finished) == ({}, {})


def test_a_kept_allocation_that_cant_be_reconnected_is_orphaned_and_the_resume_starts_afresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cayu.environments.aws_lambda_microvm import LambdaMicroVMEnvironmentFactory

    world = LambdaWorld(tmp_path)
    create = LambdaMicroVMEnvironmentFactory.create

    async def unreachable_reconnect(self, request):
        if request.operation is EnvironmentFactoryOperation.RECONNECT:
            raise RuntimeError("control plane can't reach the suspended MicroVM")
        return await create(self, request)

    async def not_proven_gone(self, request):
        return False

    async def scenario():
        try:
            await _interrupt_after_plan(world)
            monkeypatch.setattr(LambdaMicroVMEnvironmentFactory, "create", unreachable_reconnect)
            monkeypatch.setattr(
                LambdaMicroVMEnvironmentFactory, "is_allocation_disposed", not_proven_gone
            )
            app = world.app(
                call("read_plan"),
                done(),
                agents={"planner": [WritePlan()], "solver": [_ReadPlan()]},
            )
            async for _ in _PlanThenSolve(app, solve=True).run("run-interrupted"):
                pass
            return await app._runtime_session_store.load_checkpoint("run-interrupted") or {}
        finally:
            await world.close()

    finished = asyncio.run(scenario())
    first, second = world.model.created_ids()
    # The resume didn't fail on the unreachable MicroVM: it ran on a new one,
    # which its completion terminated, and the old one is still owned, as an
    # orphan, so recovery or a hand release can dispose of it.
    assert world.state(second) == "TERMINATED"
    assert world.state(first) == "SUSPENDED"
    records, orphans = _checkpoint_records(finished)
    assert records == {}
    (orphan,) = orphans.values()
    assert orphan["reconnect_metadata"]
    assert "kept_for_resume" not in orphan


def test_a_resumed_allocation_isnt_orphaned_by_a_later_transient_reconnect_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cayu.environments.aws_lambda_microvm import LambdaMicroVMEnvironmentFactory

    world = LambdaWorld(tmp_path)
    create = LambdaMicroVMEnvironmentFactory.create

    async def unreachable_reconnect(self, request):
        if request.operation is EnvironmentFactoryOperation.RECONNECT:
            raise RuntimeError("control plane throttled the reconnect")
        return await create(self, request)

    async def not_proven_gone(self, request):
        return False

    async def scenario():
        try:
            await _interrupt_after_plan(world)
            # The resume reconnects and its solve step works in the MicroVM;
            # then this process "crashes" before the run completes.
            resumed = world.app(
                call("read_plan"),
                done(),
                agents={"planner": [WritePlan()], "solver": [_ReadPlan()]},
            )
            async for _ in _PlanThenSolve(resumed, solve=True, complete=False).run(
                "run-interrupted"
            ):
                pass
            after_resume = (
                await resumed._runtime_session_store.load_checkpoint("run-interrupted") or {}
            )
            # The next resume's reconnect fails transiently, without proof the
            # MicroVM is gone: the record must stay for a retry.
            monkeypatch.setattr(LambdaMicroVMEnvironmentFactory, "create", unreachable_reconnect)
            monkeypatch.setattr(
                LambdaMicroVMEnvironmentFactory, "is_allocation_disposed", not_proven_gone
            )
            retry = world.app(
                done(),
                agents={"planner": [WritePlan()], "solver": [_ReadPlan()]},
            )
            with pytest.raises(Exception):
                async for _ in _PlanThenSolve(retry, solve=True).run("run-interrupted"):
                    pass
            after_failure = (
                await retry._runtime_session_store.load_checkpoint("run-interrupted") or {}
            )
            return after_resume, after_failure
        finally:
            await world.close()

    after_resume, after_failure = asyncio.run(scenario())
    (microvm,) = world.model.created_ids()
    (resumed,) = _checkpoint_records(after_resume)[0].values()
    # The flag covered only the first reconnect after the interrupted release.
    assert "kept_for_resume" not in resumed
    records, orphans = _checkpoint_records(after_failure)
    (kept,) = records.values()
    assert kept["allocation_session_id"] == resumed["allocation_session_id"]
    assert orphans == {}
    assert world.model.created_ids() == [microvm]
