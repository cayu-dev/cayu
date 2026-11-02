"""A crash during a recoverable factory's create of a shared allocation is recovered or reaped.

A factory that declares an ``allocation_scope`` creates a shared allocation
through ``create_recoverable``, with its allocation intent kept in the shared
record. Each "process" is a separate CayuApp on one SQLite file; a crash is
what process exit leaves. The fakes create only recoverably: their plain
``create`` refuses a CREATE, as Lambda MicroVM's process-external adapter does.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from tests.core.test_human_review import identity
from tests.environments.test_durable_shared_allocations import (
    _crash,
    _keyring,
    _PlanThenSolve,
    _records,
    _SandboxBinding,
    _Sandboxes,
    _SolveIfPlanned,
)
from tests.sandbox_broker.test_brokered_shared_allocation import _call, _done, _WritePlan

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import (
    Environment,
    EnvironmentAllocationScope,
    EnvironmentAllocationState,
    EnvironmentAllocationUnsupportedError,
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
)
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runtime.public_authority import PublicAuthorityAliasCodec
from cayu.sessions.records import SessionStatus
from cayu.sessions.recovery import (
    IncompleteSessionRecoveryAction,
    IncompleteSessionsRecoveryRequest,
)
from cayu.storage import SQLiteSessionStore
from cayu.tools import SubagentTool
from cayu.workflows import WorkflowBase, WorkflowSpec
from cayu.workspaces import LocalWorkspace


async def _drain(stream) -> list:
    return [event async for event in stream]


class _Crash(Exception):
    """The process died here."""


class _RecoverableFactory(EnvironmentFactory):
    """Creates a sandbox only through ``create_recoverable``.

    The sandbox's name comes from the allocation intent, so a dispatched
    creation can be found again. ``reconnectable=True`` adopts a sandbox a
    crashed creation left and reconnects by name; ``reconnectable=False``
    publishes no reconnect metadata and can't adopt a dispatched creation it
    didn't finish, but ``reap_allocation`` can remove it by its intent.
    """

    def __init__(
        self,
        source: Path,
        sandboxes: _Sandboxes,
        *,
        reconnectable: bool,
        crash_after_dispatch: list[bool] | None = None,
    ) -> None:
        self._source = source
        self._sandboxes = sandboxes
        self._reconnectable = reconnectable
        self._crash_after_dispatch = crash_after_dispatch or [False]
        self.calls: list[tuple[str, str]] = []
        # Operations that reached the plain ``create``; a CREATE there is refused.
        self.plain_creates: list[EnvironmentFactoryOperation] = []

    @property
    def secret_resolution_scope(self):
        return "static"

    def allocation_scope(self, request: EnvironmentFactoryRequest):
        if request.operation is not EnvironmentFactoryOperation.CREATE:
            return None
        return EnvironmentAllocationScope(provider="fake-sandbox", adapter_generation="v1")

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self._sandboxes.requests.append(request)
        self.plain_creates.append(request.operation)
        if request.operation is EnvironmentFactoryOperation.CREATE:
            raise EnvironmentAllocationUnsupportedError("This sandbox is created only recoverably.")
        if not self._reconnectable:
            raise RuntimeError("This sandbox can't be reconnected.")
        name = request.reconnect_metadata["sandbox"]
        if not self._sandboxes.path(name).is_dir():
            raise RuntimeError("the sandbox is gone")
        return self._result(request, name)

    async def create_recoverable(self, request, allocation) -> EnvironmentFactoryResult:
        self._sandboxes.requests.append(request)
        name = allocation.intent.allocation_id[-16:]
        if allocation.state is EnvironmentAllocationState.UNPREPARED:
            await allocation.prepare({"sandbox": name})
        if allocation.state is EnvironmentAllocationState.PREPARED:
            await allocation.mark_dispatched()
            self.calls.append(("create", name))
            self._sandboxes.path(name).mkdir(parents=True)
            if self._crash_after_dispatch[0]:
                self._crash_after_dispatch[0] = False
                raise _Crash()
        elif allocation.state is EnvironmentAllocationState.DISPATCHED:
            if not self._reconnectable:
                raise RuntimeError("can't tell whether the interrupted creation finished")
            assert self._sandboxes.path(name).is_dir()
            self.calls.append(("adopt", name))
        if allocation.state is EnvironmentAllocationState.DISPATCHED:
            await allocation.acknowledge({"sandbox": name})
        return self._result(request, name)

    async def reap_allocation(self, request, allocation) -> None:
        name = allocation.intent.provider_metadata["sandbox"]
        if allocation.state is EnvironmentAllocationState.REAPED:
            return
        await allocation.mark_reaping()
        self.calls.append(("reap", name))
        self._sandboxes.dispose(name)
        await allocation.mark_reaped()

    async def is_allocation_disposed(self, request: EnvironmentFactoryRequest) -> bool:
        return not self._sandboxes.path(request.reconnect_metadata["sandbox"]).exists()

    def _result(self, request: EnvironmentFactoryRequest, name: str) -> EnvironmentFactoryResult:
        sandboxes = self._sandboxes

        async def release(action: EnvironmentFactoryReleaseAction) -> None:
            if action is EnvironmentFactoryReleaseAction.DISCARD:
                sandboxes.dispose(name)

        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                workspace=LocalWorkspace(self._source, workspace_id="project"),
                binding=_SandboxBinding(sandboxes, name),
            ),
            reconnect_metadata={"sandbox": name} if self._reconnectable else {},
            release=release,
        )


class _World:
    def __init__(self, tmp_path: Path, *, reconnectable: bool, crash: bool = False) -> None:
        self.source = tmp_path / "source"
        self.source.mkdir()
        (self.source / "calc.py").write_bytes(b"def add(a, b):\n    return a - b\n")
        self.sandboxes = _Sandboxes(tmp_path / "sandboxes")
        self.database = tmp_path / "sessions.sqlite"
        self.stores: list[SQLiteSessionStore] = []
        self.reconnectable = reconnectable
        self.crash = [crash]
        self.factories: list[_RecoverableFactory] = []

    def app(self, *turns) -> CayuApp:
        keyring = _keyring()
        store = SQLiteSessionStore(
            self.database, public_authority_alias_codec=PublicAuthorityAliasCodec(keyring)
        )
        self.stores.append(store)
        app = CayuApp(
            session_store=store, enable_logging=False, public_authority_alias_keyring=keyring
        )
        app.register_provider(ScriptedModelProvider(list(turns)), default=True)
        factory = _RecoverableFactory(
            self.source,
            self.sandboxes,
            reconnectable=self.reconnectable,
            crash_after_dispatch=self.crash,
        )
        self.factories.append(factory)
        app.register_environment_factory(
            EnvironmentSpec(name="sandbox", execution_profile_identity=identity("sandboxes")),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="planner", model="scripted"), tools=[_WritePlan()])
        app.register_agent(AgentSpec(name="solver", model="scripted"), tools=[_SolveIfPlanned()])
        return app

    def calls(self) -> list[tuple[str, str]]:
        return [call for factory in self.factories for call in factory.calls]

    async def close(self) -> None:
        for store in self.stores:
            await store.close()


async def _crash_mid_create(world: _World, run_id: str) -> CayuApp:
    """The first step's factory dispatches its create, and the process dies there."""

    first = world.app()
    with pytest.raises(Exception):
        await _drain(_PlanThenSolve(first).run(run_id))
    _crash(first)
    (record,) = (await _records(first, run_id))["records"].values()
    assert record["state"] == "creating"
    assert record["allocation_intent"]["state"] == "dispatched"
    return first


def test_a_resumed_run_recovers_the_creation_a_crash_interrupted(tmp_path):
    world = _World(tmp_path, reconnectable=True, crash=True)

    async def scenario():
        try:
            await _crash_mid_create(world, "run-recover")
            second = world.app(_call("write_plan"), _done(), _call("solve"), _done())
            events = await _drain(_PlanThenSolve(second).run("run-recover"))
            return events, await _records(second, "run-recover")
        finally:
            await world.close()

    events, records = asyncio.run(scenario())
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    # The resumed run adopted the sandbox the crashed creation dispatched,
    # under the same identity; nothing else was created.
    (created, name), (adopted, same) = world.calls()
    assert (created, adopted) == ("create", "adopt") and name == same
    assert len({request.session_id for request in world.sandboxes.requests}) == 1
    # No plain CREATE ever reached the factory.
    assert all(not factory.plain_creates for factory in world.factories)
    assert world.sandboxes.finalizations == ["completed"]
    assert world.sandboxes.disposed == [name]
    assert (world.source / "solution.py").read_bytes() == b"def add(a, b):\n    return a + b\n"
    assert records == {}


def test_a_resumed_run_reaps_a_creation_it_cannot_recover_and_starts_over(tmp_path):
    world = _World(tmp_path, reconnectable=False, crash=True)

    async def scenario():
        try:
            await _crash_mid_create(world, "run-reap")
            second = world.app(_call("write_plan"), _done(), _call("solve"), _done())
            events = await _drain(_PlanThenSolve(second).run("run-reap"))
            return events, await _records(second, "run-reap")
        finally:
            await world.close()

    events, records = asyncio.run(scenario())
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    # The interrupted creation couldn't be adopted, so it was reaped through
    # its intent, and a fresh sandbox under a new identity replaced it.
    (created, old), (reaped, gone), (fresh, new) = world.calls()
    assert (created, reaped, fresh) == ("create", "reap", "create")
    assert old == gone and new != old
    assert len({request.session_id for request in world.sandboxes.requests}) == 2
    assert world.sandboxes.disposed == [old, new]
    assert world.sandboxes.finalizations == ["completed"]
    assert (world.source / "solution.py").exists()
    assert records == {}


@pytest.mark.parametrize("reconnectable", [True, False])
def test_recovery_reaps_an_ended_runs_interrupted_creation(tmp_path, reconnectable):
    world = _World(tmp_path, reconnectable=reconnectable, crash=True)

    async def scenario():
        try:
            first = await _crash_mid_create(world, "run-ended")
            # Another attempt journals the run's completion and stops, so no
            # attempt will come back for the interrupted creation.
            ended = world.app()

            class Complete(WorkflowBase):
                spec = WorkflowSpec(name="plan-then-solve")

                async def run(self, session_id):
                    ctx = self.context(session_id)
                    yield await ctx.start()
                    await ctx._append_current_attempt_event(
                        ctx.event(EventType.WORKFLOW_COMPLETED, payload={"workflow": "x"})
                    )

            await _drain(Complete(ended).run("run-ended"))
            assert len((await _records(first, "run-ended"))["records"]) == 1
            recovering = world.app()
            page = await recovering.recover_incomplete_sessions(
                IncompleteSessionsRecoveryRequest(statuses={SessionStatus.COMPLETED})
            )
            return page, await _records(recovering, "run-ended")
        finally:
            await world.close()

    page, records = asyncio.run(scenario())
    actions = {result.session_id: result.actions for result in page.results}
    assert IncompleteSessionRecoveryAction.REAPED_ALLOCATION in actions["run-ended"]
    (created, name), (reaped, same) = world.calls()
    assert (created, reaped) == ("create", "reap") and name == same
    assert world.sandboxes.disposed == [name]
    assert records == {}


def test_recovery_reaps_an_unreachable_orphan_through_its_intent(tmp_path):
    world = _World(tmp_path, reconnectable=False)

    async def scenario():
        try:
            first = world.app(_call("write_plan"), _done())
            with pytest.raises(Exception):
                await _drain(_PlanThenSolve(first, crash_after_plan=True).run("run-orphan"))
            _crash(first)
            # The resumed run can't reconnect, so it starts a new sandbox and
            # leaves the old one as an orphan whose intent says how to remove it.
            second = world.app(_call("write_plan"), _done(), _call("solve"), _done())
            await _drain(_PlanThenSolve(second).run("run-orphan"))
            (orphan,) = (await _records(second, "run-orphan"))["orphans"].values()
            assert orphan["allocation_intent"]["state"] == "acknowledged"
            page = await second.recover_incomplete_sessions(
                IncompleteSessionsRecoveryRequest(statuses={SessionStatus.COMPLETED})
            )
            return page, await _records(second, "run-orphan")
        finally:
            await world.close()

    page, records = asyncio.run(scenario())
    actions = {result.session_id: result.actions for result in page.results}
    assert IncompleteSessionRecoveryAction.REAPED_ALLOCATION in actions["run-orphan"]
    (first, old), (second, new), (reaped, gone) = world.calls()
    assert (first, second, reaped) == ("create", "create", "reap") and gone == old
    assert sorted(world.sandboxes.disposed) == sorted([old, new])
    assert records == {}


def test_an_agent_shares_a_recoverable_only_sandbox_with_its_subagent(tmp_path):
    """With a store that ends with the process the scope is still recorded, for its intent."""

    sandboxes = _Sandboxes(tmp_path / "sandboxes")
    source = tmp_path / "source"
    source.mkdir()
    factory = _RecoverableFactory(source, sandboxes, reconnectable=True)
    app = CayuApp(enable_logging=False)
    lead = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="spawn", name="subagent", arguments={"agent": "helper", "task": "help"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            _done(),
        ],
        name="lead-provider",
    )
    helper = ScriptedModelProvider([_done()], name="helper-provider")
    app.register_provider(lead, default=True)
    app.register_provider(helper)
    # No copy-back binding: an agent session can own only such a scope.
    original = factory._result

    def without_binding(request, name):
        result = original(request, name)
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                workspace=LocalWorkspace(sandboxes.path(name), workspace_id=f"sandbox-{name}"),
            ),
            reconnect_metadata=result.reconnect_metadata,
            release=result.release,
        )

    cast("Any", factory)._result = without_binding
    app.register_environment_factory(
        EnvironmentSpec(name="sandbox"), factory, default=True, allocation_scope="parent"
    )
    app.register_agent(AgentSpec(name="helper", model="scripted", provider_name=helper.name))
    app.register_agent(
        AgentSpec(name="lead", model="scripted", provider_name=lead.name),
        tools=[SubagentTool(app, agents={"helper": "helper"})],
    )

    async def scenario():
        return await _drain(
            app.run(
                RunRequest(
                    session_id="lead-recoverable",
                    agent_name="lead",
                    messages=[Message.text("user", "work with your helper")],
                )
            )
        )

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.SESSION_COMPLETED, events[-1].payload
    assert helper.requests, "the subagent ran"
    # One recoverable creation for the lead and its helper, no plain CREATE,
    # released at the lead's terminal event.
    ((created, name),) = factory.calls
    assert created == "create"
    assert factory.plain_creates == []
    assert len(sandboxes.requests) == 1
    assert sandboxes.disposed == [name]
    assert not sandboxes.path(name).exists()
    assert not app._environment_lifecycle.shared_allocations.live_keys()


def test_a_hand_release_discards_an_interrupted_creation_through_its_intent(tmp_path):
    world = _World(tmp_path, reconnectable=False, crash=True)

    async def scenario():
        try:
            first = await _crash_mid_create(world, "run-release")
            (record,) = (await _records(first, "run-release"))["records"].values()
            operator = world.app()
            released = await operator.release_shared_allocation(
                record["allocation_session_id"], mode="discard", force=True
            )
            return released, await _records(operator, "run-release")
        finally:
            await world.close()

    released, records = asyncio.run(scenario())
    assert released.mode.value == "discard"
    # The creation never returned, but its intent is enough to remove it.
    (created, name), (reaped, same) = world.calls()
    assert (created, reaped) == ("create", "reap") and name == same
    assert world.sandboxes.disposed == [name]
    assert records == {}


def test_a_resumed_run_adopts_a_creation_an_older_cayu_left_mid_plain_create(tmp_path, monkeypatch):
    """A record left mid-create with no intent came from plain create(); retry that first."""

    from hashlib import sha256

    from cayu.runtime._shared_allocation_records import (
        SharedAllocationRecords,
        shared_allocation_record_id,
    )
    from cayu.workflows import WorkflowBase, WorkflowSpec

    world = _World(tmp_path, reconnectable=True)

    async def plain_create_by_name(factory, request):
        # This factory's older version created through plain create(), naming
        # the sandbox after the request's identity, so a retry adopts it.
        factory._sandboxes.requests.append(request)
        factory.plain_creates.append(request.operation)
        if request.operation is EnvironmentFactoryOperation.RECONNECT:
            name = request.reconnect_metadata["sandbox"]
        else:
            name = sha256(request.session_id.encode()).hexdigest()[:16]
            factory.calls.append(
                ("adopt" if factory._sandboxes.path(name).is_dir() else "create", name)
            )
            factory._sandboxes.path(name).mkdir(parents=True, exist_ok=True)
        return factory._result(request, name)

    monkeypatch.setattr(_RecoverableFactory, "create", plain_create_by_name)

    class Start(WorkflowBase):
        spec = WorkflowSpec(name="plan-then-solve")

        async def run(self, session_id):
            yield await self.context(session_id).start()

    async def scenario():
        try:
            first = world.app()
            async for _ in Start(first).run("run-legacy"):
                pass
            # What the crashed process left: a record still creating, with no
            # allocation intent, and the sandbox its plain create() made.
            claim = await SharedAllocationRecords(first._runtime_session_store).claim(
                record_id=shared_allocation_record_id("parent", "run-legacy", "sandbox"),
                scope="parent",
                scope_id="run-legacy",
                owner_session_id="run-legacy",
                environment_name="sandbox",
                agent_name="planner",
                attempt_id="attempt-legacy",
                attempt_sequence=0,
            )
            assert claim is not None
            legacy = sha256(claim.record.allocation_session_id.encode()).hexdigest()[:16]
            world.sandboxes.path(legacy).mkdir(parents=True)
            second = world.app(_call("write_plan"), _done(), _call("solve"), _done())
            events = await _drain(_PlanThenSolve(second).run("run-legacy"))
            return legacy, events, await _records(second, "run-legacy")
        finally:
            await world.close()

    legacy, events, records = asyncio.run(scenario())
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    # The resumed run adopted the sandbox the interrupted plain create made,
    # rather than creating another one under a new intent and leaking it.
    assert world.calls() == [("adopt", legacy)]
    assert world.sandboxes.disposed == [legacy]
    assert not any(world.sandboxes.root.iterdir())
    assert records == {}
