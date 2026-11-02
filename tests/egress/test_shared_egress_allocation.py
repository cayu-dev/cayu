"""Workflow steps share one virtual-egress sandbox and its allocation-scoped grants."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("cryptography")

from cayu import AgentSpec, CayuApp, EventType
from cayu.credentials import CredentialMode
from cayu.egress.adapter import EgressBinding, RunnerFinalizationResult, SandboxEgressAdapter
from cayu.egress.docker_adapter import GUEST_CA_PATH
from cayu.egress.policy import HttpEgressPolicy
from cayu.egress.runtime import (
    VIRTUAL_EGRESS_EVENT_TYPES,
    VirtualCredentialSpec,
    VirtualEgressEnvironmentFactory,
)
from cayu.environments import EnvironmentSpec
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runners import LocalRunner, Runner
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.storage import InMemoryKnowledgeStore, KnowledgeAccessScope
from cayu.tools import Tool, ToolResult, ToolSpec
from cayu.vaults.base import SecretRef
from cayu.vaults.static import StaticVault
from cayu.workflows import StepError, WorkflowBase, WorkflowSpec, step

_POLICY = "provider-example"
_REAL_SECRET = "sk_test_51SharedEgressRealSecret"


class _Adapter(SandboxEgressAdapter):
    """Records every allocation it prepares; runs commands on a local runner."""

    process_external_allocation = False

    def __init__(self, root: Path, *, runner_kind: str = "docker") -> None:
        self.runner_kind = runner_kind
        self.root = root
        self.prepared: list[dict[str, Any]] = []
        self.runners: list[Runner] = []
        self.overlays: list[dict[str, str]] = []
        self.torn_down = 0

    async def prepare(self, *, session_id, grants, broker):  # type: ignore[no-untyped-def]
        self.prepared.append({"session_id": session_id, "grants": tuple(grants)})

        async def teardown() -> None:
            self.torn_down += 1

        return EgressBinding(
            env={"HTTPS_PROXY": "http://cayu-egress:8080"},
            ca_cert_pem=b"-----BEGIN CERTIFICATE-----\n",
            runner_kind=self.runner_kind,
            network="net",
            sidecar="car",
            guest_ca_path=GUEST_CA_PATH,
            teardown=teardown,
        )

    async def create_runner(self, request):  # type: ignore[no-untyped-def]
        self.overlays.append(dict(request.env_overlay))
        runner = LocalRunner(self.root, credential_mode=CredentialMode.VIRTUAL_EGRESS)
        self.runners.append(runner)
        return runner

    async def finalize_runner(self, runner: Runner, *, outcome: str | None):
        del outcome
        await runner.close()
        return RunnerFinalizationResult(workspace_mutations_quiescent=True)


class _ProcessExternalAdapter(_Adapter):
    """Stands in for a Lambda MicroVM: creation needs a durable allocation intent."""

    process_external_allocation = True


class _PerSessionAuthorityFactory(VirtualEgressEnvironmentFactory):
    @property
    def egress_authority_follows_allocation(self) -> bool:
        return False


def _text(delta: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.text_delta(delta),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


def _app(
    adapter: _Adapter,
    *,
    factory_type: type = VirtualEgressEnvironmentFactory,
    script: list | None = None,
    tools: list | None = None,
    **factory_options: Any,
) -> CayuApp:
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(script or [_text("planned"), _text("solved")]), default=True
    )
    factory = factory_type(
        adapter=adapter,
        resolver=StaticVault({"stripe_test_key": _REAL_SECRET}),
        policies={
            _POLICY: HttpEgressPolicy(
                name=_POLICY,
                allowed_hosts=["api.stripe.com"],
                allowed_endpoints=[("POST", "/v1/customers")],
            )
        },
        credentials=[
            VirtualCredentialSpec(
                env_name="STRIPE_SECRET_KEY",
                secret=SecretRef(name="stripe_test_key"),
                destination="api.stripe.com",
                policy_name=_POLICY,
            )
        ],
        event_emitter=app.scoped_event_emitter(event_types=VIRTUAL_EGRESS_EVENT_TYPES),
        **factory_options,
    )
    app.register_environment_factory(
        EnvironmentSpec(name="egress"), factory, default=True, allocation_scope="parent"
    )
    for name in ("planner", "solver"):
        app.register_agent(AgentSpec(name=name, model="scripted"), tools=tools or [])
    return app


class _PlanThenSolve(WorkflowBase):
    spec = WorkflowSpec(name="plan-then-solve")

    def __init__(self, app, *, observe=None) -> None:
        super().__init__(app)
        self._observe = observe or (lambda label: None)

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        await step(ctx, agent="planner", step_id="plan", prompt="plan")
        self._observe("after_plan")
        await step(ctx, agent="solver", step_id="solve", prompt="solve")
        yield await ctx.completed()


def test_steps_share_one_egress_sandbox_and_its_grants_are_audited_on_the_run(tmp_path):
    adapter = _Adapter(tmp_path)
    app = _app(adapter)
    torn_down_after_plan: list[int] = []

    def observe(label: str) -> None:
        torn_down_after_plan.append(adapter.torn_down)

    async def scenario():
        events = [event async for event in _PlanThenSolve(app, observe=observe).run("run-egress")]
        return events, await app.session_store.load_events("run-egress")

    events, run_events = asyncio.run(scenario())

    assert events[-1].type == EventType.WORKFLOW_COMPLETED
    # One broker, one grant set, one runner for both steps. The grant came from
    # the factory's fixed credential, which each step's own create() would
    # have minted too.
    (prepared,) = adapter.prepared
    assert prepared["session_id"].startswith("run-egress.shared-")
    (grant,) = prepared["grants"]
    assert grant.env_name == "STRIPE_SECRET_KEY"
    assert grant.destination == "api.stripe.com"
    assert len(adapter.runners) == 1
    assert _REAL_SECRET not in str(adapter.overlays)
    # The planner ending didn't revoke the grant or tear the sandbox down;
    # the run's completion did, once.
    assert torn_down_after_plan == [0]
    assert adapter.torn_down == 1
    # The allocation's identity isn't a session, so its grant events are
    # recorded on the run that owns the scope rather than dropped.
    types = [event.type for event in run_events]
    assert types.count(EventType.CREDENTIAL_MODE_SELECTED) == 1
    assert types.count(EventType.EGRESS_GRANT_MINTED) == 1
    assert types.count(EventType.EGRESS_GRANT_REVOKED) == 1
    minted = next(event for event in run_events if event.type == EventType.EGRESS_GRANT_MINTED)
    assert minted.payload["grant_id"] == grant.grant_id
    assert _REAL_SECRET not in str([event.payload for event in run_events])


def test_an_egress_authority_that_belongs_to_each_session_is_refused(tmp_path):
    adapter = _Adapter(tmp_path)
    app = _app(adapter, factory_type=_PerSessionAuthorityFactory)

    async def scenario():
        async for _event in _PlanThenSolve(app).run("run-per-session"):
            pass

    with pytest.raises(StepError) as raised:
        asyncio.run(scenario())
    message = str(raised.value)
    assert "egress authority belongs to each session" in message
    assert "can't tell which session sent a request" in message
    assert "allocation_scope='session'" in message
    assert adapter.prepared == []


def test_a_shared_process_external_egress_sandbox_needs_create_or_lookup_recovery(tmp_path):
    # A shared allocation of a process-external sandbox is created through
    # create_recoverable, under an intent kept in the run's shared record. An
    # adapter that can't look a dispatched creation up again can't make that
    # safe, so sharing it is refused before anything is allocated.
    adapter = _ProcessExternalAdapter(tmp_path, runner_kind="lambda-microvm")
    app = _app(adapter)

    async def scenario():
        async for _event in _PlanThenSolve(app).run("run-microvm"):
            pass

    with pytest.raises(StepError) as raised:
        asyncio.run(scenario())
    message = str(raised.value)
    assert "durable create-or-lookup recovery" in message
    assert "lambda-microvm" in message
    assert adapter.prepared == []


class _KnowledgeProbe(Tool):
    spec = ToolSpec(
        name="probe",
        input_schema={"type": "object", "properties": {}},
        max_terminal_payload_bytes=1024,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:shared-egress:probe", behavior_version="1", implementation_version="1"
        ),
    )

    def __init__(self, seen: list[tuple[str, list[str]]]) -> None:
        super().__init__()
        self._seen = seen

    async def run(self, ctx, args):
        self._seen.append((ctx.agent_name, list(ctx.knowledge_access_scope.allowed_namespaces)))
        return ToolResult(content="probed")


def _probe_call(call_id: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(id=call_id, name="probe", arguments={}),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def test_each_step_on_a_shared_egress_sandbox_gets_its_own_knowledge_scope(tmp_path):
    adapter = _Adapter(tmp_path)
    seen: list[tuple[str, list[str]]] = []
    scoped: list[str] = []

    def scope_for(request):
        scoped.append(request.session_id)
        owner = "scope" if ".shared-" in request.session_id else "own"
        return KnowledgeAccessScope.for_namespace(f"ns-{request.agent_name}-{owner}")

    app = _app(
        adapter,
        script=[_probe_call("p1"), _text("planned"), _probe_call("p2"), _text("solved")],
        tools=[_KnowledgeProbe(seen)],
        knowledge_store=InMemoryKnowledgeStore(),
        knowledge_access_scope_factory=scope_for,
    )

    async def scenario():
        return [event async for event in _PlanThenSolve(app).run("run-knowledge")]

    events = asyncio.run(scenario())

    assert events[-1].type == EventType.WORKFLOW_COMPLETED
    assert len(adapter.prepared) == 1
    # Each step reads knowledge under the scope its own request maps to; the
    # scope the allocation was created with reaches neither.
    assert seen == [("planner", ["ns-planner-own"]), ("solver", ["ns-solver-own"])]
    assert scoped[0].startswith("run-knowledge.shared-")
