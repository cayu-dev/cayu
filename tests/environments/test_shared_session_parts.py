"""Sessions sharing one allocation each get their own stores and MCP servers."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cayu import AgentSpec, CayuApp, EventType
from cayu.artifacts.local import LocalArtifactStore
from cayu.environments import (
    DeferredMaterialization,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    NativeBinding,
    SharedSessionParts,
)
from cayu.evals import ScriptedModelProvider
from cayu.mcp import McpServerSpec
from cayu.providers import ModelStreamEvent
from cayu.runners import LocalRunner
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.storage import InMemoryKnowledgeStore, KnowledgeAccessScope
from cayu.tools import Tool, ToolResult, ToolSpec
from cayu.vaults.static import StaticVault
from cayu.workflows import StepError, WorkflowBase, WorkflowSpec, step
from cayu.workspaces import LocalWorkspace


def _text(delta: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.text_delta(delta),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


def _probe_call(call_id: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(id=call_id, name="probe", arguments={}),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


async def _text_files(workspace) -> list[str]:
    listed = await workspace.list("*.txt")
    return sorted(str(path) for path in listed.paths)


class _Probe(Tool):
    """Record what one session's tools can reach, and leave a trace in each part."""

    spec = ToolSpec(
        name="probe",
        input_schema={"type": "object", "properties": {}},
        max_terminal_payload_bytes=1024,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:shared-session-parts:probe",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    def __init__(self, seen: list[dict]) -> None:
        super().__init__()
        self._seen = seen

    async def run(self, ctx, args):
        agent = ctx.agent_name
        await ctx.artifact_store.put_bytes(
            f"from {agent}".encode(), filename=f"{agent}.txt", session_id=ctx.session_id
        )
        await ctx.workspace.write_bytes(f"{agent}.txt", agent.encode())
        files = await _text_files(ctx.workspace)
        self._seen.append(
            {
                "agent": agent,
                "artifact_store": ctx.artifact_store_id,
                "namespaces": list(ctx.knowledge_access_scope.allowed_namespaces),
                "mcp": [server.name for server in ctx.mcp_servers],
                "workspace_files": files,
            }
        )
        return ToolResult(content="probed")


class _BaseFactory(EnvironmentFactory):
    def __init__(self, root: Path) -> None:
        self.root = root
        (root / "work").mkdir(parents=True, exist_ok=True)
        self.requests: list[EnvironmentFactoryRequest] = []
        self.finalized: list[str | None] = []

    @property
    def secret_resolution_scope(self):
        return "static"

    def _binding(self) -> NativeBinding:
        finalized = self.finalized

        class _Binding(NativeBinding):
            async def finalize(self, bound, *, outcome=None, metadata=None):
                finalized.append(outcome)
                return await super().finalize(bound, outcome=outcome, metadata=metadata)

        return _Binding()

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.requests.append(request)
        return EnvironmentFactoryResult(
            environment=self._environment(request),
            release=self._release,
        )

    def _environment(self, request: EnvironmentFactoryRequest) -> Environment:
        return Environment(
            EnvironmentSpec(name=request.environment_name),
            workspace=LocalWorkspace(self.root / "work", workspace_id="shared-work"),
            runner=LocalRunner(self.root / "work"),
            binding=self._binding(),
        )

    async def _release(self, action: EnvironmentFactoryReleaseAction) -> None:
        del action


class _PerSessionFactory(_BaseFactory):
    """Gives every request its own artifact store, knowledge scope and MCP server."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.knowledge = InMemoryKnowledgeStore()
        self.stores: dict[str, LocalArtifactStore] = {}
        self.part_requests: list[EnvironmentFactoryRequest] = []

    def _store(self, name: str) -> LocalArtifactStore:
        if name not in self.stores:
            self.stores[name] = LocalArtifactStore(
                self.root / f"artifacts-{name}", store_id=f"store-{name}"
            )
        return self.stores[name]

    def _environment(self, request: EnvironmentFactoryRequest) -> Environment:
        # What create() gives the request it is called with: on a shared
        # allocation that is the scope's own request, which no session is.
        base = super()._environment(request)
        return Environment(
            base.spec,
            workspace=base.workspace,
            runner=base.runner,
            binding=base.binding,
            artifact_store=self._store("creator"),
            knowledge_store=self.knowledge,
            knowledge_access_scope=KnowledgeAccessScope.for_namespace("private:creator"),
            mcp_servers=[McpServerSpec(name="mcp-creator", url="https://creator.invalid/mcp")],
        )

    @property
    def attaches_session_parts(self) -> bool:
        return True

    async def attach_session_parts(self, request, shared) -> SharedSessionParts:
        assert isinstance(shared, EnvironmentFactoryResult)
        self.part_requests.append(request)
        agent = request.agent_name
        return SharedSessionParts(
            artifact_store=self._store(agent),
            knowledge_store=self.knowledge,
            knowledge_access_scope=KnowledgeAccessScope.for_namespace(f"private:{agent}"),
            mcp_servers=(McpServerSpec(name=f"mcp-{agent}", url=f"https://{agent}.invalid/mcp"),),
        )


class _PlanThenSolve(WorkflowBase):
    spec = WorkflowSpec(name="plan-then-solve")

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        await step(ctx, agent="planner", step_id="plan", prompt="plan")
        await step(ctx, agent="solver", step_id="solve", prompt="solve")
        yield await ctx.completed()


def _app(factory: EnvironmentFactory, seen: list[dict], *, probes: bool = True) -> CayuApp:
    script = (
        [_probe_call("plan-probe"), _text("planned"), _probe_call("solve-probe"), _text("solved")]
        if probes
        else [_text("planned"), _text("solved")]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider(script), default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="shared"), factory, default=True, allocation_scope="parent"
    )
    tools = [_Probe(seen)] if probes else []
    for name in ("planner", "solver"):
        app.register_agent(AgentSpec(name=name, model="scripted"), tools=tools)
    return app


async def _run(app: CayuApp, run_id: str) -> list:
    return [event async for event in _PlanThenSolve(app).run(run_id)]


def test_each_session_gets_its_own_stores_and_mcp_servers_on_one_allocation(tmp_path):
    factory = _PerSessionFactory(tmp_path)
    seen: list[dict] = []
    events = asyncio.run(_run(_app(factory, seen), "run-parts"))

    assert events[-1].type == EventType.WORKFLOW_COMPLETED
    # One allocation: the solver sees the planner's file in the shared workspace.
    assert len(factory.requests) == 1
    assert factory.requests[0].session_id.startswith("run-parts.shared-")
    assert seen == [
        {
            "agent": "planner",
            "artifact_store": "store-planner",
            "namespaces": ["private:planner"],
            "mcp": ["mcp-planner"],
            "workspace_files": ["planner.txt"],
        },
        {
            "agent": "solver",
            "artifact_store": "store-solver",
            "namespaces": ["private:solver"],
            "mcp": ["mcp-solver"],
            "workspace_files": ["planner.txt", "solver.txt"],
        },
    ]
    # Each session's parts came from its own request, never the scope's.
    assert [request.agent_name for request in factory.part_requests] == ["planner", "solver"]
    assert all(
        ".shared-" not in request.session_id and request.parent_session_id == "run-parts"
        for request in factory.part_requests
    )
    assert factory.finalized == ["completed"]

    async def listed(name: str) -> list[str]:
        result = await factory.stores[name].list()
        return sorted(item.filename for item in result.artifacts)

    async def listings():
        return {name: await listed(name) for name in ("planner", "solver", "creator")}

    # Neither session could write into the other's store, and the creating
    # request's store reached nobody.
    assert asyncio.run(listings()) == {
        "planner": ["planner.txt"],
        "solver": ["solver.txt"],
        "creator": [],
    }


class _FixedPartsWithMcpFactory(_BaseFactory):
    def _environment(self, request):
        base = super()._environment(request)
        return Environment(
            base.spec,
            workspace=base.workspace,
            runner=base.runner,
            binding=base.binding,
            mcp_servers=[McpServerSpec(name="mcp-fixed", url="https://fixed.invalid/mcp")],
        )


def test_mcp_servers_without_per_session_parts_are_refused_with_the_way_out(tmp_path):
    factory = _FixedPartsWithMcpFactory(tmp_path)
    with pytest.raises(StepError) as raised:
        asyncio.run(_run(_app(factory, [], probes=False), "run-mcp"))

    failure = str(raised.value)
    assert "it exposes MCP servers" in failure
    assert "attach_session_parts" in failure
    assert "allocation_scope='session'" in failure


class _VaultFactory(_PerSessionFactory):
    def _environment(self, request):
        base = super()._environment(request)
        return Environment(
            base.spec,
            workspace=base.workspace,
            runner=base.runner,
            binding=base.binding,
            vault=StaticVault({"token": "secret"}),
        )

    @property
    def secret_resolution_scope(self):
        return "dynamic"


def test_secrets_are_still_refused_when_the_factory_attaches_session_parts(tmp_path):
    factory = _VaultFactory(tmp_path)
    with pytest.raises(StepError) as raised:
        asyncio.run(_run(_app(factory, [], probes=False), "run-vault"))

    failure = str(raised.value)
    assert "secret resolution scope is dynamic" in failure
    assert "vault or credential proxy" in failure
    assert "every other session's commands could read it" in failure
    assert factory.part_requests == []


class _WrongPartsFactory(_PerSessionFactory):
    async def attach_session_parts(self, request, shared):
        return {"artifact_store": self._store(request.agent_name)}


def test_attach_session_parts_must_return_shared_session_parts(tmp_path):
    with pytest.raises(StepError, match="must return SharedSessionParts"):
        asyncio.run(_run(_app(_WrongPartsFactory(tmp_path), [], probes=False), "run-bad"))


def test_the_default_hook_refuses():
    class Plain(_BaseFactory):
        pass

    factory = Plain.__new__(Plain)
    assert factory.attaches_session_parts is False
    assert factory.egress_authority_follows_allocation is False
    request = EnvironmentFactoryRequest(session_id="s", agent_name="a", environment_name="e")
    shared = EnvironmentFactoryResult(environment=Environment(EnvironmentSpec(name="e")))
    with pytest.raises(RuntimeError, match="no per-session parts"):
        asyncio.run(factory.attach_session_parts(request, shared))


def test_shared_session_parts_validates_its_fields():
    with pytest.raises(TypeError, match="ArtifactStore"):
        SharedSessionParts(artifact_store=object())  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="McpServerSpec"):
        SharedSessionParts(mcp_servers="mcp")  # ty: ignore[invalid-argument-type]
    parts = SharedSessionParts(mcp_servers=(McpServerSpec(name="m", url="https://m.invalid"),))
    assert isinstance(parts.mcp_servers, tuple)


class _DeferredFactory(_BaseFactory):
    """A deferred factory: create() makes nothing, the first runner use does."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.materializations: list[str] = []
        self.closed = 0

    @property
    def deferred_materialization(self) -> bool:
        return True

    def _environment(self, request):
        factory = self

        class _ClosingRunner(LocalRunner):
            async def close(self) -> None:
                factory.closed += 1
                await super().close()

        async def materialize(mode):
            factory.materializations.append(mode)
            return _ClosingRunner(factory.root / "work"), factory._binding()

        materialization = DeferredMaterialization(
            materialize,
            default_cwd=str(self.root / "work"),
            placeholder_workspace=lambda runner: LocalWorkspace(
                self.root / "work", workspace_id="deferred-work"
            ),
        )
        return Environment(
            EnvironmentSpec(name=request.environment_name),
            workspace=LocalWorkspace(self.root / "work", workspace_id="deferred-source"),
            runner=materialization.runner,
            binding=materialization.binding,
        )


class _PlainProbe(_Probe):
    async def run(self, ctx, args):
        agent = ctx.agent_name
        await ctx.workspace.write_bytes(f"{agent}.txt", agent.encode())
        files = await _text_files(ctx.workspace)
        self._seen.append({"agent": agent, "workspace_files": files})
        return ToolResult(content="probed")


def test_a_deferred_factory_materializes_once_for_the_whole_scope(tmp_path):
    factory = _DeferredFactory(tmp_path)
    seen: list[dict] = []
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                _probe_call("plan-probe"),
                _text("planned"),
                _probe_call("solve-probe"),
                _text("solved"),
            ]
        ),
        default=True,
    )
    app.register_environment_factory(
        EnvironmentSpec(name="shared"), factory, default=True, allocation_scope="parent"
    )
    for name in ("planner", "solver"):
        app.register_agent(AgentSpec(name=name, model="scripted"), tools=[_PlainProbe(seen)])

    events = asyncio.run(_run(app, "run-deferred"))

    assert events[-1].type == EventType.WORKFLOW_COMPLETED
    assert len(factory.requests) == 1
    # Materialized once, when the scope's allocation was created, not again
    # by either session's first tool call.
    assert factory.materializations == ["use"]
    assert seen == [
        {"agent": "planner", "workspace_files": ["planner.txt"]},
        {"agent": "solver", "workspace_files": ["planner.txt", "solver.txt"]},
    ]
    assert factory.finalized == ["completed"]


def test_reaping_a_recorded_allocation_records_its_events_on_the_owner(tmp_path):
    from cayu.runtime._shared_allocation_records import SharedAllocationRecord
    from cayu.runtime._shared_environment_allocations import _dispose_recorded

    class Reconnecting(_BaseFactory):
        discarded = 0

        async def _release(self, action):
            if action is EnvironmentFactoryReleaseAction.DISCARD:
                type(self).discarded += 1

    factory = Reconnecting(tmp_path)
    record = SharedAllocationRecord(
        allocation_session_id="run-reap.shared-0123456789abcdef",
        scope="parent",
        scope_id="run-reap",
        owner_session_id="run-reap",
        environment_name="shared",
        agent_name="planner",
        attempt_id="attempt-1",
        claim_generation=1,
        state="live",
        binding_required=False,
        reconnect_metadata={"sandbox": "s"},
    )

    assert asyncio.run(_dispose_recorded(factory, record)) is True
    (request,) = factory.requests
    # The reaper's RECONNECT names the allocation, and the run that owned it as
    # the session its events (a broker's grant revocations, say) belong on.
    assert request.session_id == record.allocation_session_id
    assert request.event_session == "run-reap"
    assert Reconnecting.discarded == 1
