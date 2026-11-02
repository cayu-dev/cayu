"""Workflow steps sharing one Lambda MicroVM, across crashes, takeovers and recovery.

Each "process" is its own CayuApp on its own connection to one SQLite file;
the fake control plane and the per-MicroVM sidecars outlive every app, as AWS
does. A crash is what process exit leaves: the app's in-memory hold on the
allocation is dropped without touching the MicroVM.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest
from tests.core.test_human_review import identity
from tests.egress.test_aws_lambda_microvm_recoverable_allocation import (
    _FakeProxyServer,
    _skip_guest_preflight,
)
from tests.environments.test_aws_lambda_microvm_factory import (
    REGION,
    LambdaWorld,
    WritePlan,
    call,
    done,
)
from tests.runners.lambda_microvm_harness import ClientTokenLambdaModel

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.egress import (
    ApprovedEgressDestination,
    HttpEgressPolicy,
    SandboxCodingEnvironmentFactory,
    VirtualEgressEnvironmentFactory,
)
from cayu.egress.aws_lambda_microvm_adapter import LambdaMicroVMEgressAdapter
from cayu.egress.proxy_exposure import VpcTaskProxyExposure
from cayu.environments import AllocationSharingScope, EnvironmentFactoryOperation, EnvironmentSpec
from cayu.runners import ExecCommand, LambdaMicroVMOwnershipSuperseded, LambdaMicroVMRunner
from cayu.runtime._shared_allocation_intents import SharedAllocationIntentContext
from cayu.runtime._shared_allocation_records import SHARED_ALLOCATIONS_CHECKPOINT_KEY
from cayu.sessions.records import SessionStatus
from cayu.sessions.recovery import (
    IncompleteSessionRecoveryAction,
    IncompleteSessionsRecoveryRequest,
)
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.workflows import StepRunOptions, WorkflowBase, WorkflowSpec, command, step
from cayu.workspaces import LocalWorkspace

# Guest preflight needs a real MicroVM; the egress-mediated case replaces it,
# as the virtual-egress Lambda tests do. It touches only the egress adapter.
__all__ = ["_skip_guest_preflight"]

_SHARED = StepRunOptions(allocation_scope=AllocationSharingScope.PARENT)
_SOLUTION = b"def add(a, b):\n    return a + b\n"
_CHECK = "from solution import add; assert add(2, 3) == 5; print('ok')"


class SolveFromPlan(Tool):
    spec = ToolSpec(
        name="solve",
        parallel_safe=False,
        workspace_mutation=True,
        execution_profile_identity=identity("solve-from-plan"),
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        assert ctx.workspace is not None
        plan = (await ctx.workspace.read_bytes("plan.md")).content
        await ctx.workspace.create_bytes("solution.py", b"def add(a, b):\n    " + plan)
        return ToolResult(content="solved")


_AGENTS: dict[str, list[Tool]] = {"planner": [WritePlan()], "solver": [SolveFromPlan()]}


class _Crash(Exception):
    """The process died here."""


class _Handover(WorkflowBase):
    spec = WorkflowSpec(name="lambda-handover")

    def __init__(
        self,
        app,
        *,
        steps=True,
        check=False,
        crash_after_plan=False,
        after_plan=None,
        before_completion=None,
    ):
        super().__init__(app)
        self._steps = steps
        self._check = check
        self._crash_after_plan = crash_after_plan
        self._after_plan = after_plan
        self._before_completion = before_completion
        self.check_result = None

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        if self._steps:
            await step(ctx, agent="planner", step_id="plan", prompt="plan", run_options=_SHARED)
            if self._after_plan is not None:
                await self._after_plan()
            if self._crash_after_plan:
                raise _Crash()
            await step(ctx, agent="solver", step_id="solve", prompt="solve", run_options=_SHARED)
        if self._check:
            self.check_result = await command(
                ctx,
                step_id="tests",
                argv=["python3", "-c", _CHECK],
                allocation_scope=AllocationSharingScope.PARENT,
            )
        if self._before_completion is not None:
            await self._before_completion(ctx)
        yield await ctx.completed()


def _crash(app: CayuApp) -> list[LambdaMicroVMRunner]:
    """Drop what the app holds in memory without touching the MicroVM; return its runners."""

    registry = app._environment_lifecycle.shared_allocations
    entries = (*registry._live.values(), *registry._closing)
    runners = [entry.runner for entry in entries]
    for entry in entries:
        registry._forget(entry)
    registry._live.clear()
    registry._closing.clear()
    return [runner for runner in runners if isinstance(runner, LambdaMicroVMRunner)]


async def _records(app: CayuApp, run_id: str) -> dict:
    checkpoint = await app.session_store.load_checkpoint(run_id) or {}
    return checkpoint.get(SHARED_ALLOCATIONS_CHECKPOINT_KEY) or {}


async def _journal_completion_then_crash(ctx) -> None:
    # The newest attempt journals its completion, then the process dies
    # before it released anything.
    await ctx._append_current_attempt_event(
        ctx.event(EventType.WORKFLOW_COMPLETED, payload={"workflow": ctx.workflow_name})
    )
    raise _Crash()


def test_two_steps_and_a_command_share_one_microvm(tmp_path: Path) -> None:
    world = LambdaWorld(tmp_path)
    workflows: list[_Handover] = []

    async def scenario():
        try:
            app = world.app(call("write_plan"), done(), call("solve"), done(), agents=_AGENTS)
            workflows.append(_Handover(app, check=True))
            events = [event async for event in workflows[0].run("run-1")]
            return events, await _records(app, "run-1")
        finally:
            await world.close()

    events, records = asyncio.run(scenario())
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    # One MicroVM for both steps and the command: the solver read the planner's
    # file from its disk, and the command imported the solver's.
    (microvm_id,) = world.model.created_ids()
    assert (world.disk / "solution.py").read_bytes() == _SOLUTION
    check = workflows[0].check_result
    assert check is not None and check.exit_code == 0 and check.stdout.strip() == "ok"
    (submission,) = world.model.run_calls
    assert submission["clientToken"].startswith("cayu-")
    # The run's completion terminated it, and its record is gone.
    assert world.state(microvm_id) == "TERMINATED"
    assert records == {}


def test_a_resumed_run_reconnects_with_a_newer_epoch_and_fences_the_crashed_owner(
    tmp_path: Path,
) -> None:
    world = LambdaWorld(tmp_path)
    seen: dict[str, Any] = {}

    async def scenario():
        try:
            first = world.app(call("write_plan"), done(), agents=_AGENTS)
            with pytest.raises(_Crash):
                async for _ in _Handover(first, crash_after_plan=True).run("run-1"):
                    pass
            (stale_runner,) = _crash(first)
            (record,) = (await _records(first, "run-1"))["records"].values()
            seen["record"] = record

            async def check_fencing(ctx) -> None:
                # The crashed process's runner lost the MicroVM to the resumed one.
                with pytest.raises((LambdaMicroVMOwnershipSuperseded, RuntimeError)):
                    await stale_runner.exec(ExecCommand.process("true"))
                # An owner that decided to reattach under the crashed attempt's
                # claim arrives late; its epoch is older than the resumed run's.
                with pytest.raises(LambdaMicroVMOwnershipSuperseded):
                    late = await LambdaMicroVMRunner.from_existing(
                        record["reconnect_metadata"]["microvm_id"],
                        region_name=REGION,
                        client=world.model,
                        endpoint_transport=world.sidecars,
                        close_action="none",
                        poll_interval_s=0,
                        default_cwd=str(world.disk),
                        owner_epoch=record["claim_generation"],
                    )
                    await late.exec(ExecCommand.process("true"))

            second = world.app(call("solve"), done(), agents=_AGENTS)
            events = [
                event
                async for event in _Handover(second, before_completion=check_fencing).run("run-1")
            ]
            return events, await _records(second, "run-1")
        finally:
            await world.close()

    events, records = asyncio.run(scenario())
    # The resumed run kept the MicroVM through both fencing attempts, and its
    # own completion terminated it.
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    record = seen["record"]
    assert record["state"] == "live" and record["binding_state"] is not None
    # The durable identity names the MicroVM and its owner, and holds no credential.
    assert set(record["reconnect_metadata"]) == {
        "microvm_id",
        "endpoint",
        "region",
        "image_identifier",
        "image_version",
        "session_id",
        "environment_name",
    }
    assert "model-token" not in json.dumps(record)
    # The resumed run reconnected to the same MicroVM instead of creating one,
    # and the solver read the plan the crashed run's planner left on its disk.
    assert world.model.created_ids() == [record["reconnect_metadata"]["microvm_id"]]
    assert (world.disk / "solution.py").read_bytes() == _SOLUTION
    assert world.state(record["reconnect_metadata"]["microvm_id"]) == "TERMINATED"
    assert records == {}


def test_a_create_whose_acknowledgement_was_lost_is_adopted_by_the_resumed_run(
    tmp_path: Path,
) -> None:
    world = LambdaWorld(tmp_path)
    # AWS creates the MicroVM, but no acknowledgement ever reaches the first
    # process, through all its retries.
    world.model.lose_acknowledgements = 4

    async def scenario():
        try:
            first = world.app(call("write_plan"), done(), agents=_AGENTS)
            with pytest.raises(Exception):
                async for _ in _Handover(first).run("run-1"):
                    pass
            _crash(first)
            (pending,) = (await _records(first, "run-1"))["records"].values()
            second = world.app(call("write_plan"), done(), call("solve"), done(), agents=_AGENTS)
            events = [event async for event in _Handover(second).run("run-1")]
            return pending, events
        finally:
            await world.close()

    pending, events = asyncio.run(scenario())
    # The intent was dispatched and its identity never acknowledged.
    assert pending["state"] == "creating"
    assert pending["allocation_intent"]["state"] == "dispatched"
    assert pending["allocation_intent"].get("reconnect_metadata") is None
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    # Replaying the intent's client token adopted the one MicroVM AWS made.
    (microvm_id,) = world.model.created_ids()
    tokens = {submission["clientToken"] for submission in world.model.run_calls}
    assert len(tokens) == 1 and len(world.model.run_calls) > 1
    assert world.state(microvm_id) == "TERMINATED"


def test_recovery_reaps_an_ended_runs_unacknowledged_and_live_microvms(tmp_path: Path) -> None:
    world = LambdaWorld(tmp_path)

    async def scenario():
        try:
            # run-live: both steps ran on one MicroVM; the run journaled its
            # completion and died before releasing it.
            live = world.app(call("write_plan"), done(), call("solve"), done(), agents=_AGENTS)
            with pytest.raises(_Crash):
                workflow = _Handover(live, before_completion=_journal_completion_then_crash)
                async for _ in workflow.run("run-live"):
                    pass
            _crash(live)
            # run-lost: the create's acknowledgement was lost, then the run
            # journaled its completion and died.
            world.model.lose_acknowledgements = 4
            lost = world.app(call("write_plan"), done(), agents=_AGENTS)
            with pytest.raises(Exception):
                async for _ in _Handover(lost).run("run-lost"):
                    pass
            _crash(lost)
            ended = world.app(agents=_AGENTS)
            with pytest.raises(_Crash):
                workflow = _Handover(
                    ended, steps=False, before_completion=_journal_completion_then_crash
                )
                async for _ in workflow.run("run-lost"):
                    pass
            _crash(ended)
            assert len(world.model.live_ids()) == 2
            recovering = world.app(agents=_AGENTS)
            page = await recovering.recover_incomplete_sessions(
                IncompleteSessionsRecoveryRequest(statuses={SessionStatus.COMPLETED})
            )
            return (
                page,
                await _records(recovering, "run-live"),
                await _records(recovering, "run-lost"),
            )
        finally:
            await world.close()

    page, live_records, lost_records = asyncio.run(scenario())
    actions = {result.session_id: result.actions for result in page.results}
    assert IncompleteSessionRecoveryAction.REAPED_ALLOCATION in actions["run-live"]
    assert IncompleteSessionRecoveryAction.REAPED_ALLOCATION in actions["run-lost"]
    # Both MicroVMs are gone, proven by readback, and nothing new was created.
    assert len(world.model.created_ids()) == 2
    assert world.model.live_ids() == []
    assert live_records == {} and lost_records == {}


def test_an_agent_session_cant_own_a_shared_microvm_and_leaves_none_behind(
    tmp_path: Path,
) -> None:
    # The binding owns the MicroVM's termination through durable finalization,
    # and an agent session publishes its success before its scope is
    # released, so only a workflow run (which releases before journaling
    # completion) may own a shared MicroVM.
    world = LambdaWorld(tmp_path)

    async def scenario():
        try:
            app = world.app(call("write_plan"), done(), agents=_AGENTS, allocation_scope="parent")
            return [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="solo",
                        agent_name="planner",
                        messages=[Message.text("user", "plan")],
                    )
                )
            ]
        finally:
            await world.close()

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.SESSION_FAILED
    assert "only a workflow run can own" in json.dumps(events[-1].payload)
    # The MicroVM made before the refusal was discarded, not leaked.
    assert world.model.live_ids() == []


def test_steps_share_one_egress_mediated_microvm_and_copy_back_once(tmp_path: Path) -> None:
    world = LambdaWorld(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    adapter = LambdaMicroVMEgressAdapter(
        region_name=REGION,
        egress_network_connector_arn="arn:aws:lambda:us-east-1:123:network-connector:nc-1",
        exposure=VpcTaskProxyExposure("10.0.1.20"),
        client=world.model,
        endpoint_transport_factory=lambda: world.sidecars,
        proxy_server_factory=cast("Any", _FakeProxyServer),
        runner_options={"poll_interval_s": 0, "default_cwd": str(world.disk)},
    )
    factory = SandboxCodingEnvironmentFactory(
        adapter=adapter,
        image=ClientTokenLambdaModel.image_arn,
        source_workspace=LocalWorkspace(source, workspace_id="project"),
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
        execution_profile_identity=identity("lambda-egress-sandbox"),
    )

    async def scenario():
        app = world.app(call("write_plan"), done(), call("solve"), done(), agents={})
        app.register_environment_factory(
            EnvironmentSpec(
                name="egress-sandbox", execution_profile_identity=identity("egress-sandbox")
            ),
            factory,
            allocation_scope="parent",
        )
        for name, tools in _AGENTS.items():
            app.register_agent(
                AgentSpec(name=name, model="scripted"),
                tools=tools,
                environment_name="egress-sandbox",
            )
        try:
            return [event async for event in _Handover(app).run("run-egress")]
        finally:
            await world.close()

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    # One recoverable allocation under a client token for both steps.
    (microvm_id,) = world.model.created_ids()
    (submission,) = world.model.run_calls
    assert submission["clientToken"].startswith("cayu-")
    # The run's completion copied both steps' files back once and terminated it.
    assert (source / "plan.md").read_bytes() == b"return a + b\n"
    assert (source / "solution.py").read_bytes() == _SOLUTION
    assert world.state(microvm_id) == "TERMINATED"
    # The egress adapter's sidecar claim was fenced by the record's claim
    # generation, as the plain factory's is.
    (sidecar,) = world.sidecars.by_endpoint.values()
    assert sidecar.owner_fence._highest_owner_epoch == 1


def test_a_hand_discard_reaps_an_unacknowledged_microvm(tmp_path: Path) -> None:
    world = LambdaWorld(tmp_path)
    # AWS creates the MicroVM, but no acknowledgement reaches the run, which
    # then crashes and is never resumed.
    world.model.lose_acknowledgements = 4

    async def scenario():
        try:
            crashed = world.app(call("write_plan"), done(), agents=_AGENTS)
            with pytest.raises(Exception):
                async for _ in _Handover(crashed).run("run-1"):
                    pass
            _crash(crashed)
            assert len(world.model.live_ids()) == 1
            operator = world.app(agents=_AGENTS)
            page = await operator.list_shared_allocations(workflow_run_id="run-1")
            (allocation,) = page.allocations
            release = await operator.release_shared_allocation(
                allocation.allocation_id, mode="discard", force=True
            )
            return allocation, release, await _records(operator, "run-1")
        finally:
            await world.close()

    allocation, release, records = asyncio.run(scenario())
    assert allocation.state == "creating" and not allocation.reconnectable
    assert release.allocation.allocation_id == allocation.allocation_id
    # The intent's client token found the MicroVM, which was terminated; none
    # was created, and the record is gone.
    assert len(world.model.created_ids()) == 1
    assert world.model.live_ids() == []
    assert len({submission["clientToken"] for submission in world.model.run_calls}) == 1
    assert records == {}


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
    registered = app._get_registered_environment("sandbox")
    assert registered is not None and registered.factory is not None
    factory = registered.factory
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
