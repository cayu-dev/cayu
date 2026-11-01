"""A workflow run's shared allocation survives a crash and is reconnected in another process.

Each "process" is a separate CayuApp on its own connection to one SQLite file.
A crash is what process exit leaves: the app's in-memory state is gone (its
hold on the allocation is dropped without touching it), and only the store and
the sandbox remain.
"""

from __future__ import annotations

import asyncio
import base64
import shutil
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import SecretStr
from tests.core.test_human_review import identity
from tests.sandbox_broker.test_brokered_shared_allocation import _call, _done, _WritePlan

from cayu import AgentSpec, CayuApp, EventType
from cayu.environments import (
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    SyncBinding,
)
from cayu.evals import ScriptedModelProvider
from cayu.runtime._shared_allocation_records import (
    SHARED_ALLOCATIONS_CHECKPOINT_KEY,
    SharedAllocationClaimLost,
    SharedAllocationRecords,
)
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions.records import SessionStatus
from cayu.sessions.recovery import (
    IncompleteSessionRecoveryAction,
    IncompleteSessionsRecoveryRequest,
)
from cayu.storage import SQLiteSessionStore
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.workflows import StepRunOptions, WorkflowBase, WorkflowSpec, step
from cayu.workspaces import LocalWorkspace

_SHARED = StepRunOptions(allocation_scope="parent")


@dataclass
class _Sandboxes:
    """Sandboxes that outlive every app, like Docker containers outlive a process."""

    root: Path
    requests: list[EnvironmentFactoryRequest] = field(default_factory=list)
    copy_ins: list[str] = field(default_factory=list)
    finalizations: list[str | None] = field(default_factory=list)
    disposed: list[str] = field(default_factory=list)

    def path(self, name: str) -> Path:
        return self.root / name

    def dispose(self, name: str) -> None:
        if self.path(name).exists():
            shutil.rmtree(self.path(name))
            self.disposed.append(name)


class _SandboxBinding(SyncBinding):
    def __init__(self, sandboxes: _Sandboxes, name: str) -> None:
        super().__init__(
            target_workspace=LocalWorkspace(sandboxes.path(name), workspace_id=f"sandbox-{name}"),
            source_conflict_policy="require_revision",
            max_file_bytes=1 << 20,
            max_total_bytes=1 << 24,
        )
        self._sandboxes = sandboxes
        self._name = name

    async def bind(self, workspace, runner, *, session_id, **options):
        self._sandboxes.copy_ins.append(session_id)
        return await super().bind(workspace, runner, session_id=session_id, **options)

    async def finalize(self, bound, *, outcome=None, metadata=None):
        snapshot = await super().finalize(bound, outcome=outcome, metadata=metadata)
        self._sandboxes.finalizations.append(outcome)
        self._sandboxes.dispose(self._name)
        return snapshot


class _ReconnectingFactory(EnvironmentFactory):
    """Allocates a sandbox directory per request identity and reconnects to it by name."""

    def __init__(self, source: Path, sandboxes: _Sandboxes) -> None:
        self._source = source
        self._sandboxes = sandboxes

    @property
    def secret_resolution_scope(self):
        return "static"

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self._sandboxes.requests.append(request)
        if request.operation is EnvironmentFactoryOperation.RECONNECT:
            name = request.reconnect_metadata["sandbox"]
            if not self._sandboxes.path(name).is_dir():
                raise RuntimeError("the sandbox is gone")
        else:
            name = sha256(request.session_id.encode()).hexdigest()[:16]
            self._sandboxes.path(name).mkdir(parents=True, exist_ok=True)
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
            reconnect_metadata={"sandbox": name},
            release=release,
        )

    async def is_allocation_disposed(self, request: EnvironmentFactoryRequest) -> bool:
        return not self._sandboxes.path(request.reconnect_metadata["sandbox"]).exists()


class _SolveIfPlanned(Tool):
    spec = ToolSpec(name="solve", parallel_safe=False, workspace_mutation=True)

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        assert ctx.workspace is not None
        plan = (await ctx.workspace.read_bytes("plan.md")).content
        assert plan.startswith(b"replace a - b")
        source = (await ctx.workspace.read_bytes("calc.py")).content
        await ctx.workspace.create_bytes("solution.py", source.replace(b"a - b", b"a + b"))
        return ToolResult(content="solved")


class _Review(Tool):
    spec = ToolSpec(name="review", parallel_safe=False, workspace_mutation=True)

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        assert ctx.workspace is not None
        solution = (await ctx.workspace.read_bytes("solution.py")).content
        assert b"a + b" in solution
        await ctx.workspace.create_bytes("review.md", b"looks right\n")
        return ToolResult(content="reviewed")


class _Crash(Exception):
    """The process died here."""


class _PlanThenSolve(WorkflowBase):
    spec = WorkflowSpec(name="plan-then-solve")

    def __init__(self, app, *, crash_after_plan=False, before_completion=None) -> None:
        super().__init__(app)
        self._crash_after_plan = crash_after_plan
        self._before_completion = before_completion

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        await step(ctx, agent="planner", step_id="plan", prompt="plan", run_options=_SHARED)
        if self._crash_after_plan:
            raise _Crash()
        await step(ctx, agent="solver", step_id="solve", prompt="solve", run_options=_SHARED)
        if self._before_completion is not None:
            await self._before_completion(ctx)
        yield await ctx.completed()


def _keyring() -> PublicAuthorityAliasKeyring:
    return PublicAuthorityAliasKeyring(
        active_key_id="test",
        keys={"test": SecretStr(base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("="))},
    )


class _World:
    """One source checkout, one sandbox host and one session database, shared by every app."""

    def __init__(self, tmp_path: Path) -> None:
        self.source = tmp_path / "source"
        self.source.mkdir()
        (self.source / "calc.py").write_bytes(b"def add(a, b):\n    return a - b\n")
        self.sandboxes = _Sandboxes(tmp_path / "sandboxes")
        self.database = tmp_path / "sessions.sqlite"
        self.stores: list[SQLiteSessionStore] = []

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
        app.register_environment_factory(
            EnvironmentSpec(name="sandbox", execution_profile_identity=identity("sandboxes")),
            _ReconnectingFactory(self.source, self.sandboxes),
            default=True,
        )
        app.register_agent(AgentSpec(name="planner", model="scripted"), tools=[_WritePlan()])
        app.register_agent(AgentSpec(name="solver", model="scripted"), tools=[_SolveIfPlanned()])
        app.register_agent(AgentSpec(name="reviewer", model="scripted"), tools=[_Review()])
        return app

    async def close(self) -> None:
        for store in self.stores:
            await store.close()


def _crash(app: CayuApp) -> None:
    """Drop everything the app holds in memory, without touching any allocation."""

    registry = app._environment_lifecycle.shared_allocations
    for entry in (*registry._live.values(), *registry._closing):
        registry._forget(entry)
    registry._live.clear()
    registry._closing.clear()


async def _records(app: CayuApp, run_id: str) -> dict:
    checkpoint = await app.session_store.load_checkpoint(run_id) or {}
    return checkpoint.get(SHARED_ALLOCATIONS_CHECKPOINT_KEY) or {}


async def _crash_after_plan(world: _World, run_id: str) -> CayuApp:
    first = world.app(_call("write_plan"), _done())
    with pytest.raises(_Crash):
        async for _ in _PlanThenSolve(first, crash_after_plan=True).run(run_id):
            pass
    _crash(first)
    return first


def test_a_resumed_run_reconnects_to_the_crashed_runs_allocation(tmp_path):
    world = _World(tmp_path)

    async def scenario():
        try:
            first = await _crash_after_plan(world, "run-1")
            (record,) = (await _records(first, "run-1"))["records"].values()
            assert record["state"] == "live" and record["binding_state"] is not None
            # The plan exists only in the sandbox.
            assert not (world.source / "plan.md").exists()
            second = world.app(_call("solve"), _done())
            events = [event async for event in _PlanThenSolve(second).run("run-1")]
            return events, await _records(second, "run-1")
        finally:
            await world.close()

    events, records = asyncio.run(scenario())
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    created, reconnected = world.sandboxes.requests
    assert created.operation is EnvironmentFactoryOperation.CREATE
    assert reconnected.operation is EnvironmentFactoryOperation.RECONNECT
    # The same allocation, under the identity recorded before it was created.
    assert reconnected.session_id == created.session_id
    assert created.session_id.startswith("run-1.shared-")
    # Copied in once, by the crashed process; the resumed one rebuilt its binding.
    assert world.sandboxes.copy_ins == [created.session_id]
    # The solver read the plan the crashed process's planner left in the sandbox,
    # and the run's completion copied both back and disposed of the sandbox.
    assert (world.source / "plan.md").read_bytes() == b"replace a - b with a + b\n"
    assert (world.source / "solution.py").read_bytes() == b"def add(a, b):\n    return a + b\n"
    assert world.sandboxes.finalizations == ["completed"]
    assert len(world.sandboxes.disposed) == 1
    assert records == {}


def test_a_durable_store_never_labels_the_allocation_for_the_process_reaper(tmp_path):
    world = _World(tmp_path)

    async def scenario():
        try:
            await _crash_after_plan(world, "run-label")
            second = world.app(_call("solve"), _done())
            async for _ in _PlanThenSolve(second).run("run-label"):
                pass
        finally:
            await world.close()

    asyncio.run(scenario())
    # Neither the creation nor the reconnect names a process-local owner, so
    # Docker coding never labels the container and
    # reap_orphaned_docker_coding_containers never lists it: the crashed
    # process's exit can't remove what the resumed run adopted.
    assert [request.operation for request in world.sandboxes.requests] == [
        EnvironmentFactoryOperation.CREATE,
        EnvironmentFactoryOperation.RECONNECT,
    ]
    assert all(request.process_local_owner is None for request in world.sandboxes.requests)


def test_only_the_newest_of_two_resuming_attempts_owns_the_allocation(tmp_path):
    world = _World(tmp_path)

    async def scenario():
        try:
            await _crash_after_plan(world, "run-race")
            solved = asyncio.Event()
            go_on = asyncio.Event()

            async def pause(ctx):
                solved.set()
                await go_on.wait()
                raise RuntimeError("the older attempt fails after the newer one took over")

            older = world.app(_call("solve"), _done())
            older_run = asyncio.ensure_future(
                _drain(_PlanThenSolve(older, before_completion=pause).execute("run-race"))
            )
            await asyncio.wait_for(solved.wait(), 30)
            # A newer attempt resumes the same run in a third process. Both steps
            # are already journaled, so it attaches to nothing; its completion
            # takes the recorded allocation over, reconnects and releases it.
            newer = world.app()
            events = [event async for event in _PlanThenSolve(newer).run("run-race")]
            assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
            assert world.sandboxes.finalizations == ["completed"]
            # The older attempt fails and releases what it holds: the claim is
            # gone, so it lets go without finalizing or disposing of anything.
            go_on.set()
            with pytest.raises(RuntimeError, match="older attempt fails"):
                await older_run
            assert await older.close_idle_environment_resources(timeout_s=5) is True
        finally:
            await world.close()

    asyncio.run(scenario())
    operations = [request.operation for request in world.sandboxes.requests]
    assert operations == [
        EnvironmentFactoryOperation.CREATE,
        EnvironmentFactoryOperation.RECONNECT,
        EnvironmentFactoryOperation.RECONNECT,
    ]
    assert len({request.session_id for request in world.sandboxes.requests}) == 1
    assert world.sandboxes.copy_ins == [world.sandboxes.requests[0].session_id]
    assert world.sandboxes.finalizations == ["completed"]
    assert len(world.sandboxes.disposed) == 1
    assert (world.source / "solution.py").exists()


async def _drain(stream):
    async for _ in stream:
        pass


def test_recovery_reaps_an_ended_runs_allocation_and_keeps_a_resumable_one(tmp_path):
    world = _World(tmp_path)

    async def journal_completion_then_crash(ctx):
        # The run's newest attempt journals its completion, and the process
        # dies before it released the allocation.
        await ctx._append_current_attempt_event(
            ctx.event(EventType.WORKFLOW_COMPLETED, payload={"workflow": ctx.workflow_name})
        )
        raise _Crash()

    async def scenario():
        try:
            ended = world.app(_call("write_plan"), _done(), _call("solve"), _done())
            with pytest.raises(_Crash):
                workflow = _PlanThenSolve(ended, before_completion=journal_completion_then_crash)
                async for _ in workflow.run("run-ended"):
                    pass
            _crash(ended)
            await _crash_after_plan(world, "run-resumable")
            assert len(world.sandboxes.requests) == 2
            ended_sandbox, resumable_sandbox = (
                request.session_id for request in world.sandboxes.requests
            )
            recovering = world.app()
            page = await recovering.recover_incomplete_sessions(
                IncompleteSessionsRecoveryRequest(statuses={SessionStatus.COMPLETED})
            )
            return (
                page,
                await _records(recovering, "run-ended"),
                await _records(recovering, "run-resumable"),
                ended_sandbox,
                resumable_sandbox,
            )
        finally:
            await world.close()

    page, ended_records, resumable_records, ended_sandbox, resumable_sandbox = asyncio.run(
        scenario()
    )
    actions = {result.session_id: result.actions for result in page.results}
    assert IncompleteSessionRecoveryAction.REAPED_ALLOCATION in actions["run-ended"]
    assert IncompleteSessionRecoveryAction.REAPED_ALLOCATION not in actions.get("run-resumable", ())
    # The ended run's sandbox was reconnected and discarded, without a copy-back.
    assert world.sandboxes.requests[-1].operation is EnvironmentFactoryOperation.RECONNECT
    assert world.sandboxes.requests[-1].session_id == ended_sandbox
    assert world.sandboxes.finalizations == []
    assert world.sandboxes.disposed == [sha256(ended_sandbox.encode()).hexdigest()[:16]]
    assert ended_records == {}
    # The resumable run keeps its sandbox and record for its resume.
    assert world.sandboxes.path(sha256(resumable_sandbox.encode()).hexdigest()[:16]).is_dir()
    assert len(resumable_records["records"]) == 1


def test_a_process_whose_attempt_was_taken_over_reclaims_before_reusing_its_sandbox(tmp_path):
    """Attempt A1 creates, B takes over in another process, then A2 runs back in A1's process."""

    world = _World(tmp_path)

    class PlanSolveReview(WorkflowBase):
        spec = WorkflowSpec(name="plan-then-solve")

        def __init__(self, app, *, pause_after=None) -> None:
            super().__init__(app)
            self._pause_after = pause_after or {}

        async def run(self, session_id):
            ctx = self.context(session_id)
            yield await ctx.start()
            for agent, step_id in (
                ("planner", "plan"),
                ("solver", "solve"),
                ("reviewer", "review"),
            ):
                await step(ctx, agent=agent, step_id=step_id, prompt=step_id, run_options=_SHARED)
                pause = self._pause_after.get(step_id)
                if pause is not None:
                    await pause()
            yield await ctx.completed()

    def gate():
        reached, release = asyncio.Event(), asyncio.Event()

        async def pause():
            reached.set()
            await release.wait()
            raise RuntimeError("this attempt fails after it was taken over")

        return reached, release, pause

    async def scenario():
        try:
            first = world.app(_call("write_plan"), _done(), _call("review"), _done())
            a1_reached, _a1_release, a1_pause = gate()
            a1 = asyncio.ensure_future(
                _drain(PlanSolveReview(first, pause_after={"plan": a1_pause}).execute("run-pp"))
            )
            await asyncio.wait_for(a1_reached.wait(), 30)
            # B resumes the run in a second process and takes the sandbox over.
            second = world.app(_call("solve"), _done())
            b_reached, b_release, b_pause = gate()
            b = asyncio.ensure_future(
                _drain(PlanSolveReview(second, pause_after={"solve": b_pause}).execute("run-pp"))
            )
            await asyncio.wait_for(b_reached.wait(), 30)
            # A2 starts in the first process, which still holds A1's sandbox.
            # Its review step reuses that sandbox only after claiming the record
            # back from B, and sees B's solution there.
            events = [event async for event in PlanSolveReview(first).run("run-pp")]
            assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
            b_release.set()
            with pytest.raises(RuntimeError, match="taken over"):
                await b
            a1.cancel()
            with pytest.raises(asyncio.CancelledError):
                await a1
            assert await first.close_idle_environment_resources(timeout_s=5) is True
            assert await second.close_idle_environment_resources(timeout_s=5) is True
            return await _records(first, "run-pp")
        finally:
            await world.close()

    records = asyncio.run(scenario())
    # A1 created it, B reconnected to it, and A2 reused A1's live handle.
    assert [request.operation for request in world.sandboxes.requests] == [
        EnvironmentFactoryOperation.CREATE,
        EnvironmentFactoryOperation.RECONNECT,
    ]
    assert len(world.sandboxes.copy_ins) == 1
    # Only A2, the newest attempt, finalized: one copy-back with everyone's files.
    assert world.sandboxes.finalizations == ["completed"]
    assert len(world.sandboxes.disposed) == 1
    for name in ("plan.md", "solution.py", "review.md"):
        assert (world.source / name).exists(), name
    assert records == {}


def test_two_attempts_claiming_one_record_never_both_hold_it(tmp_path):
    world = _World(tmp_path)

    class Start(WorkflowBase):
        spec = WorkflowSpec(name="plan-then-solve")

        async def run(self, session_id):
            yield await self.context(session_id).start()

    async def scenario():
        try:
            first, second = world.app(), world.app()
            async for _ in Start(first).run("run-claim"):
                pass
            one = SharedAllocationRecords(first._runtime_session_store)
            other = SharedAllocationRecords(second._runtime_session_store)
            fields = {
                "record_id": "r",
                "scope": "parent",
                "scope_id": "run-claim",
                "owner_session_id": "run-claim",
                "environment_name": "sandbox",
                "agent_name": "planner",
            }
            claimed = await one.claim(**fields, attempt_id="attempt-a", attempt_sequence=1)
            assert claimed is not None and claimed.previous is None
            held = await one.update(
                "r", claimed.record, state="created", reconnect_metadata={"sandbox": "x"}
            )
            # A newer attempt takes the record over; it keeps the identity
            # because a created allocation can be reconnected.
            taken = await other.claim(**fields, attempt_id="attempt-b", attempt_sequence=2)
            assert taken is not None
            assert taken.record.allocation_session_id == held.allocation_session_id
            assert taken.record.claim_generation == held.claim_generation + 1
            with pytest.raises(SharedAllocationClaimLost):
                await one.claim(**fields, attempt_id="attempt-a", attempt_sequence=1)
            # The older attempt finds out at its next write, and can neither
            # dispose of nor forget the allocation.
            with pytest.raises(SharedAllocationClaimLost):
                await one.update("r", held, state="live")
            assert not await one.holds("run-claim", held.allocation_session_id, {"attempt-a"})
            assert not await one.remove("run-claim", held.allocation_session_id, {"attempt-a"})
            assert await other.holds("run-claim", held.allocation_session_id, {"attempt-b"})
            # A claimant that turns out superseded after claiming puts the
            # previous claim back.
            stale = await one.claim(**fields, attempt_id="attempt-c", attempt_sequence=3)
            assert stale is not None
            await one.restore("r", stale)
            assert await other.holds("run-claim", held.allocation_session_id, {"attempt-b"})
            assert not await one.holds("run-claim", held.allocation_session_id, {"attempt-c"})
        finally:
            await world.close()

    asyncio.run(scenario())


def test_completion_retries_a_transient_reconnect_failure(tmp_path, monkeypatch):
    world = _World(tmp_path)

    async def stop(ctx):
        raise _Crash()

    async def scenario():
        try:
            first = world.app(_call("write_plan"), _done(), _call("solve"), _done())
            with pytest.raises(_Crash):
                await _drain(_PlanThenSolve(first, before_completion=stop).run("retry-release"))
            _crash(first)
            original = _ReconnectingFactory.create
            failed = False

            async def flaky(factory, request):
                nonlocal failed
                if not failed and request.operation is EnvironmentFactoryOperation.RECONNECT:
                    failed = True
                    raise RuntimeError("temporary reconnect failure")
                return await original(factory, request)

            monkeypatch.setattr(_ReconnectingFactory, "create", flaky)

            async def retry(ctx):
                with pytest.raises(RuntimeError, match="temporary reconnect failure"):
                    await ctx.completed()

            resumed = world.app()
            await _drain(_PlanThenSolve(resumed, before_completion=retry).run("retry-release"))
            assert failed
            assert world.sandboxes.finalizations == ["completed"]
            assert not await _records(resumed, "retry-release")
            assert (world.source / "plan.md").exists()
        finally:
            await world.close()

    asyncio.run(scenario())


def test_recovery_finishes_checkpointed_disposal_without_reconnecting(tmp_path, monkeypatch):
    from cayu.environments._finalization_disposal import checkpoint_finalization_disposal

    world = _World(tmp_path)
    recovered = []

    async def interrupted_disposal(binding, bound, *, outcome=None, metadata=None):
        await SyncBinding.finalize(binding, bound, outcome=outcome, metadata=metadata)
        await checkpoint_finalization_disposal({"sandbox": binding._name})
        binding._sandboxes.dispose(binding._name)
        raise _Crash()

    async def recover(factory, request, state):
        recovered.append(state["sandbox"])
        factory._sandboxes.dispose(state["sandbox"])

    async def never_prove_disposed(factory, request):
        return False

    monkeypatch.setattr(_SandboxBinding, "finalize", interrupted_disposal)
    monkeypatch.setattr(_ReconnectingFactory, "recover_finalization_disposal", recover)
    monkeypatch.setattr(_ReconnectingFactory, "is_allocation_disposed", never_prove_disposed)

    async def scenario():
        try:
            first = world.app(_call("write_plan"), _done(), _call("solve"), _done())
            with pytest.raises(_Crash):
                await _drain(_PlanThenSolve(first).run("disposal-retry"))
            _crash(first)
            [record] = (await _records(first, "disposal-retry"))["records"].values()
            assert record["disposal_state"] is not None
            resumed = world.app()
            await _drain(_PlanThenSolve(resumed).run("disposal-retry"))
            assert recovered == [record["reconnect_metadata"]["sandbox"]]
            assert len(world.sandboxes.requests) == 1
            assert not await _records(resumed, "disposal-retry")
        finally:
            await world.close()

    asyncio.run(scenario())


def test_a_live_allocation_without_a_binding_remains_reconnectable():
    from cayu.runtime._shared_allocation_records import SharedAllocationRecord

    record = SharedAllocationRecord(
        allocation_session_id="allocation",
        scope="parent",
        scope_id="run",
        owner_session_id="run",
        environment_name="sandbox",
        agent_name="agent",
        attempt_id="attempt",
        claim_generation=1,
        state="live",
        reconnect_metadata={"sandbox": "existing"},
        binding_required=False,
    )
    assert SharedAllocationRecord.from_payload(record.payload()).reconnectable


def test_recovery_keeps_unconfirmed_disposal_records(tmp_path):
    from cayu.runtime._shared_allocation_records import SharedAllocationRecord
    from cayu.runtime._shared_environment_allocations import _dispose_recorded

    world = _World(tmp_path)
    record = SharedAllocationRecord(
        allocation_session_id="allocation",
        scope="parent",
        scope_id="run",
        owner_session_id="run",
        environment_name="sandbox",
        agent_name="agent",
        attempt_id="attempt",
        claim_generation=1,
        state="live",
        result_metadata={"resource": "still-running"},
    )
    assert (
        asyncio.run(_dispose_recorded(_ReconnectingFactory(world.source, world.sandboxes), record))
        is False
    )


def test_a_resumed_run_reuses_an_allocation_without_a_binding(tmp_path, monkeypatch):
    from dataclasses import replace

    world = _World(tmp_path)
    original = _ReconnectingFactory.create

    async def without_binding(factory, request):
        result = await original(factory, request)
        name = result.reconnect_metadata["sandbox"]
        if request.operation is EnvironmentFactoryOperation.CREATE:
            shutil.copytree(factory._source, factory._sandboxes.path(name), dirs_exist_ok=True)
        return replace(
            result,
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                workspace=LocalWorkspace(factory._sandboxes.path(name), workspace_id="sandbox"),
            ),
        )

    monkeypatch.setattr(_ReconnectingFactory, "create", without_binding)

    async def scenario():
        try:
            first = await _crash_after_plan(world, "unbound-run")
            [record] = (await _records(first, "unbound-run"))["records"].values()
            assert record["binding_required"] is False
            resumed = world.app(_call("solve"), _done())
            await _drain(_PlanThenSolve(resumed).run("unbound-run"))
            assert len(world.sandboxes.requests) == 2
            assert world.sandboxes.requests[-1].operation is EnvironmentFactoryOperation.RECONNECT
            assert not await _records(resumed, "unbound-run")
        finally:
            await world.close()

    asyncio.run(scenario())


def test_a_command_after_replayed_steps_reconnects_to_the_durable_allocation(tmp_path, monkeypatch):
    import sys
    from dataclasses import replace

    from cayu.runners import LocalRunner
    from cayu.workflows import command

    world = _World(tmp_path)
    original = _ReconnectingFactory.create

    async def with_runner(factory, request):
        result = await original(factory, request)
        name = result.reconnect_metadata["sandbox"]
        return replace(
            result,
            environment=Environment(
                result.environment.spec,
                workspace=result.environment.workspace,
                binding=result.environment.binding,
                runner=LocalRunner(factory._sandboxes.path(name)),
            ),
        )

    monkeypatch.setattr(_ReconnectingFactory, "create", with_runner)

    async def stop(ctx):
        raise _Crash()

    async def inspect(ctx):
        result = await command(
            ctx,
            step_id="inspect",
            environment_name="sandbox",
            allocation_scope="parent",
            argv=[
                sys.executable,
                "-c",
                "from pathlib import Path; assert b'a + b' in Path('solution.py').read_bytes()",
            ],
        )
        assert result.passed

    async def scenario():
        try:
            first = world.app(_call("write_plan"), _done(), _call("solve"), _done())
            with pytest.raises(_Crash):
                await _drain(_PlanThenSolve(first, before_completion=stop).run("command-resume"))
            _crash(first)
            resumed = world.app()
            await _drain(_PlanThenSolve(resumed, before_completion=inspect).run("command-resume"))
            assert len(world.sandboxes.requests) == 2
            assert world.sandboxes.requests[-1].operation is EnvironmentFactoryOperation.RECONNECT
            assert world.sandboxes.finalizations == ["completed"]
            assert b"a + b" in (world.source / "solution.py").read_bytes()
            assert not await _records(resumed, "command-resume")
        finally:
            await world.close()

    asyncio.run(scenario())
