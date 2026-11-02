"""What sharing a Lambda MicroVM across workflow steps does today.

The shared-allocation path (``cayu.runtime._shared_environment_allocations``)
creates a scope's allocation with the factory's plain ``create()``. Both Lambda
MicroVM factories allocate a process-external resource and declare an
allocation scope, so they may only be created through ``create_recoverable``
under a durable intent. Until the shared path routes such factories through
``create_recoverable``, sharing either of them is refused before any AWS call,
never satisfied with a non-recoverable create. The egress-mediated factory is
also refused for its egress authority.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.environments.test_aws_lambda_microvm_factory import (
    REGION,
    LambdaWorld,
    WritePlan,
    call,
    done,
)
from tests.runners.lambda_microvm_harness import ClientTokenLambdaModel

from cayu import AgentSpec
from cayu.egress import (
    ApprovedEgressDestination,
    HttpEgressPolicy,
    VirtualEgressEnvironmentFactory,
)
from cayu.egress.aws_lambda_microvm_adapter import LambdaMicroVMEgressAdapter
from cayu.egress.proxy_exposure import VpcTaskProxyExposure
from cayu.environments import AllocationSharingScope, EnvironmentSpec
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


def test_sharing_the_lambda_factory_is_refused_before_any_provider_call(tmp_path: Path) -> None:
    world = LambdaWorld(tmp_path)
    app = world.app(call("write_plan"), done(), agents={"planner": [WritePlan()]})

    text = _run_refused(world, app, "planner")

    assert "create_recoverable" in text
    assert world.model.run_calls == []
    assert world.model.created_ids() == []


def test_the_virtual_egress_lambda_factory_is_refused_for_its_egress_authority(
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

    text = _run_refused(world, app, "egress-planner")

    assert "egress authority" in text
    assert model.run_calls == []
