"""An agent session's shared sandbox survives a crash and is reconnected in another process.

The agent owns an ``allocation_scope="parent"`` scope: it and its subagents
work in one sandbox, which its run releases at its terminal event. Each
"process" is a separate CayuApp on its own connection to one SQLite file. A
crash is what process exit leaves: the app never released the scope, its
in-memory hold on the sandbox is gone, and only the store and the sandbox
remain.
"""

from __future__ import annotations

import asyncio
import base64
import shutil
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import SecretStr
from tests.core.test_human_review import identity

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest
from cayu.environments import (
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
)
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runtime._shared_allocation_records import (
    SHARED_ALLOCATIONS_CHECKPOINT_KEY,
    SharedAllocationClaimLost,
    SharedAllocationRecords,
)
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.recovery import (
    IncompleteSessionRecoveryAction,
    IncompleteSessionsRecoveryRequest,
)
from cayu.storage import SQLiteSessionStore
from cayu.tools import SubagentTool
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.tools.policy import AllowAllToolPolicy
from cayu.workspaces import LocalWorkspace


@dataclass
class _Sandboxes:
    """Sandboxes that outlive every app, like containers outlive a process."""

    root: Path
    requests: list[EnvironmentFactoryRequest] = field(default_factory=list)
    disposed: list[str] = field(default_factory=list)

    def path(self, name: str) -> Path:
        return self.root / name

    def dispose(self, name: str) -> None:
        if self.path(name).exists():
            shutil.rmtree(self.path(name))
            self.disposed.append(name)


class _SandboxFactory(EnvironmentFactory):
    """One sandbox directory per request identity, reconnected by name.

    The sandbox is the session's workspace itself (no copy-back binding), the
    kind of allocation an agent session may own a shared scope of.
    """

    def __init__(self, sandboxes: _Sandboxes) -> None:
        self._sandboxes = sandboxes

    @property
    def secret_resolution_scope(self):
        return "static"

    @property
    def execution_profile_identity(self):
        return identity("sandbox-factory")

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
                workspace=LocalWorkspace(sandboxes.path(name), workspace_id=f"sandbox-{name}"),
            ),
            reconnect_metadata={"sandbox": name},
            release=release,
        )

    async def is_allocation_disposed(self, request: EnvironmentFactoryRequest) -> bool:
        return not self._sandboxes.path(request.reconnect_metadata["sandbox"]).exists()


class _WriteNotes(Tool):
    spec = ToolSpec(
        name="write_notes",
        parallel_safe=False,
        workspace_mutation=True,
        execution_profile_identity=identity("write-notes"),
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        assert ctx.workspace is not None
        await ctx.workspace.create_bytes("notes.md", b"the helper was here\n")
        return ToolResult(content="noted")


class _ReadNotes(Tool):
    spec = ToolSpec(
        name="read_notes",
        parallel_safe=False,
        workspace_mutation=False,
        execution_profile_identity=identity("read-notes"),
    )

    def __init__(self, seen: list[bytes]) -> None:
        self._seen = seen

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        assert ctx.workspace is not None
        self._seen.append((await ctx.workspace.read_bytes("notes.md")).content)
        return ToolResult(content="read")


class _AllowAll(AllowAllToolPolicy):
    """Allow every tool, with an identity that stays the same in every process."""

    @property
    def execution_profile_identity(self):
        return identity("allow-all")


def _call(tool: str, **arguments) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(id=f"call-{tool}", name=tool, arguments=arguments),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def _done() -> list[ModelStreamEvent]:
    return [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]


class _Provider(ScriptedModelProvider):
    """A scripted provider that, with ``stall``, stalls after its script until cancelled."""

    execution_profile_identity = identity("scripted-provider")

    def __init__(self, turns, *, name: str, stall: bool = False) -> None:
        super().__init__(turns, name=name)
        self.stalled = asyncio.Event()
        self._remaining = len(turns)
        self._stall = stall

    async def stream(self, request):
        if self._stall and self._remaining == 0:
            self.stalled.set()
            await asyncio.Event().wait()
        self._remaining -= 1
        async for event in super().stream(request):
            yield event


def _keyring() -> PublicAuthorityAliasKeyring:
    return PublicAuthorityAliasKeyring(
        active_key_id="test",
        keys={"test": SecretStr(base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("="))},
    )


class _World:
    """One sandbox host and one session database, shared by every app."""

    def __init__(self, tmp_path: Path) -> None:
        self.sandboxes = _Sandboxes(tmp_path / "sandboxes")
        self.database = tmp_path / "sessions.sqlite"
        self.stores: list[SQLiteSessionStore] = []
        self.seen: list[bytes] = []
        self.leads: dict[int, _Provider] = {}

    def app(self, lead_turns=(), helper_turns=(), *, stall=False) -> CayuApp:
        keyring = _keyring()
        store = SQLiteSessionStore(
            self.database, public_authority_alias_codec=PublicAuthorityAliasCodec(keyring)
        )
        self.stores.append(store)
        app = CayuApp(
            session_store=store, enable_logging=False, public_authority_alias_keyring=keyring
        )
        lead = _Provider(list(lead_turns), name="lead-provider", stall=stall)
        helper = _Provider(list(helper_turns), name="helper-provider")
        app.register_provider(lead, default=True)
        app.register_provider(helper)
        app.register_environment_factory(
            EnvironmentSpec(name="sandbox", execution_profile_identity=identity("sandboxes")),
            _SandboxFactory(self.sandboxes),
            default=True,
            allocation_scope="parent",
        )
        app.register_agent(
            AgentSpec(name="helper", model="scripted", provider_name=helper.name),
            tools=[_WriteNotes()],
            tool_policy=_AllowAll(),
        )
        app.register_agent(
            AgentSpec(name="lead", model="scripted", provider_name=lead.name),
            tools=[
                SubagentTool(
                    app,
                    agents={"helper": "helper"},
                    execution_profile_identity=identity("subagent"),
                ),
                _ReadNotes(self.seen),
            ],
            tool_policy=_AllowAll(),
        )
        self.leads[id(app)] = lead
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


async def _records(app: CayuApp, session_id: str) -> dict:
    checkpoint = await app._runtime_session_store.load_checkpoint(session_id) or {}
    return checkpoint.get(SHARED_ALLOCATIONS_CHECKPOINT_KEY) or {}


async def _drain(stream) -> list:
    return [event async for event in stream]


async def _crash_after_the_helper_wrote(
    world: _World, session_id: str, *, forget: bool = True
) -> CayuApp:
    """Run the lead until its helper wrote notes.md, then let the process die.

    The process dies before the lead's run releases its scope: the release
    hook never runs, so the shared sandbox and its record are left behind.
    With ``forget=False`` the app keeps its in-memory hold, like a process
    that stalled instead of dying.
    """

    first = world.app(
        [_call("subagent", agent="helper", task="take notes")],
        [_call("write_notes"), _done()],
        stall=True,
    )

    async def died_before_releasing(session, event) -> None:
        del session, event

    cast("Any", first._environment_lifecycle)._end_shared_allocation_owner = died_before_releasing
    run = asyncio.ensure_future(
        _drain(
            first.run(
                RunRequest(
                    session_id=session_id,
                    agent_name="lead",
                    messages=[Message.text("user", "work with your helper")],
                )
            )
        )
    )
    await asyncio.wait_for(world.leads[id(first)].stalled.wait(), 30)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    if forget:
        _crash(first)
    return first


def test_a_resumed_agent_reconnects_to_its_crashed_runs_sandbox(tmp_path):
    world = _World(tmp_path)

    async def scenario():
        try:
            first = await _crash_after_the_helper_wrote(world, "lead-1")
            stored = await first.session_store.load("lead-1")
            assert stored is not None
            (record,) = (await _records(first, "lead-1"))["records"].values()
            assert record["state"] == "live"
            # Claimed by the crashed run's epoch of the session.
            assert record["attempt_id"].startswith(f"run:{stored.instance_id}:")
            assert record["attempt_sequence"] <= stored.run_epoch
            second = world.app([_call("read_notes"), _done()])
            events = await _drain(
                second.resume(
                    ResumeRequest(session_id="lead-1", messages=[Message.text("user", "go on")])
                )
            )
            return stored, events, await _records(second, "lead-1")
        finally:
            await world.close()

    stored, events, records = asyncio.run(scenario())
    assert stored.status is SessionStatus.INTERRUPTED, stored.status
    assert events[-1].type == EventType.SESSION_COMPLETED, events[-1].payload
    created, reconnected = world.sandboxes.requests
    assert created.operation is EnvironmentFactoryOperation.CREATE
    assert reconnected.operation is EnvironmentFactoryOperation.RECONNECT
    # The same sandbox, under the identity recorded before it was created, and
    # the helper's file from the crashed run is still in it.
    assert reconnected.session_id == created.session_id
    assert created.session_id.startswith("lead-1.shared-")
    # Each request names the record's claim generation, which the takeover raised.
    assert created.ownership_epoch is not None and reconnected.ownership_epoch is not None
    assert reconnected.ownership_epoch > created.ownership_epoch
    assert world.seen == [b"the helper was here\n"]
    # A durable store's requests never carry a process-local owner, so a
    # Docker reaper can't remove what the resumed run adopted.
    assert all(request.process_local_owner is None for request in world.sandboxes.requests)
    # The resumed run released the sandbox at its terminal event.
    assert world.sandboxes.disposed == [sha256(created.session_id.encode()).hexdigest()[:16]]
    assert records == {}


def test_a_stale_holder_lets_go_once_a_newer_run_took_the_sandbox_over(tmp_path):
    world = _World(tmp_path)

    async def scenario():
        try:
            # The first process stalled rather than died: it still holds the
            # sandbox, and its release runs late, after a second process
            # resumed the session and took the sandbox over.
            first = await _crash_after_the_helper_wrote(world, "lead-stale", forget=False)
            first_registry = first._environment_lifecycle.shared_allocations
            assert first_registry.live_keys()
            second = world.app([_call("read_notes")], stall=True)
            resumed = asyncio.ensure_future(
                _drain(
                    second.resume(
                        ResumeRequest(
                            session_id="lead-stale", messages=[Message.text("user", "go on")]
                        )
                    )
                )
            )
            await asyncio.wait_for(world.leads[id(second)].stalled.wait(), 30)
            stale = await first.session_store.load("lead-stale")
            assert stale is not None
            (entry,) = first_registry._live.values()
            held_before = set(entry.attempts)
            # A session of the stale run attaches (a subagent whose spawn was
            # in flight, say): it claims as the run this process executes, not
            # as whatever run the store holds now, so it doesn't adopt the
            # newer run's claim.
            assert await first_registry._reclaim(entry)
            assert entry.attempts == held_before
            # Nor can the stale process create or claim anything as its old run.
            claimant = await first_registry._agent_claimant("lead-stale")
            assert claimant is not None and claimant.attempt_id in held_before
            with pytest.raises(SharedAllocationClaimLost):
                await claimant.fence()
            # The first process's late release, for its own run of the session.
            await first_registry.end_owner("lead-stale", outcome="interrupted", run_epoch=1)
            sandbox = sha256(world.sandboxes.requests[0].session_id.encode()).hexdigest()[:16]
            # It let go without disposing of what the newer run uses.
            assert world.sandboxes.path(sandbox).is_dir()
            assert world.sandboxes.disposed == []
            assert not first_registry.live_keys()
            (record,) = (await _records(first, "lead-stale"))["records"].values()
            assert record["attempt_id"] == f"run:{stale.instance_id}:{stale.run_epoch}"
            # A claim for the stale run is refused outright.
            records = SharedAllocationRecords(first._runtime_session_store)
            with pytest.raises(SharedAllocationClaimLost):
                await records.claim(
                    record_id=next(iter((await _records(first, "lead-stale"))["records"])),
                    scope=record["scope"],
                    scope_id=record["scope_id"],
                    owner_session_id="lead-stale",
                    environment_name="sandbox",
                    agent_name="lead",
                    attempt_id="run:stale:1",
                    attempt_sequence=1,
                )
            resumed.cancel()
            with pytest.raises(asyncio.CancelledError):
                await resumed
            return sandbox
        finally:
            await world.close()

    sandbox = asyncio.run(scenario())
    assert world.seen == [b"the helper was here\n"]
    # Only the newer run, the owner, finalized: its interruption released it.
    assert world.sandboxes.disposed == [sandbox]


def test_only_one_of_two_racing_resumes_owns_the_sandbox(tmp_path):
    world = _World(tmp_path)

    async def scenario():
        try:
            await _crash_after_the_helper_wrote(world, "lead-race")
            apps = [world.app([_call("read_notes"), _done()]) for _ in range(2)]
            outcomes = await asyncio.gather(
                *(
                    _drain(
                        app.resume(
                            ResumeRequest(
                                session_id="lead-race", messages=[Message.text("user", "go on")]
                            )
                        )
                    )
                    for app in apps
                ),
                return_exceptions=True,
            )
            return outcomes, await _records(apps[0], "lead-race")
        finally:
            await world.close()

    outcomes, records = asyncio.run(scenario())
    completed = [
        outcome
        for outcome in outcomes
        if isinstance(outcome, list) and outcome[-1].type == EventType.SESSION_COMPLETED
    ]
    assert len(completed) == 1, outcomes
    # One reconnect, by the resume that won; the other never reached the sandbox.
    operations = [request.operation for request in world.sandboxes.requests]
    assert operations == [EnvironmentFactoryOperation.CREATE, EnvironmentFactoryOperation.RECONNECT]
    assert world.seen == [b"the helper was here\n"]
    assert len(world.sandboxes.disposed) == 1
    assert records == {}


def test_recovery_keeps_an_interrupted_agents_sandbox_and_reaps_it_once_closed(tmp_path):
    world = _World(tmp_path)

    async def scenario():
        try:
            await _crash_after_the_helper_wrote(world, "lead-close")
            recovering = world.app()
            page = await recovering.recover_incomplete_sessions(
                IncompleteSessionsRecoveryRequest(
                    statuses={SessionStatus.INTERRUPTED, SessionStatus.COMPLETED}
                )
            )
            kept = await _records(recovering, "lead-close")
            closed = await recovering.close_session("lead-close", reason="nobody will continue")
            return page, kept, closed, await _records(recovering, "lead-close")
        finally:
            await world.close()

    page, kept, closed, records = asyncio.run(scenario())
    created = world.sandboxes.requests[0]
    sandbox = sha256(created.session_id.encode()).hexdigest()[:16]
    # Interrupted, the session can still continue, so its sandbox stays.
    assert all(
        IncompleteSessionRecoveryAction.REAPED_ALLOCATION not in result.actions
        for result in page.results
    )
    assert len(kept["records"]) == 1
    # Closed, it can't, so the close reaps it: reconnected and discarded.
    assert IncompleteSessionRecoveryAction.REAPED_ALLOCATION in closed.actions, closed
    assert world.sandboxes.requests[-1].operation is EnvironmentFactoryOperation.RECONNECT
    assert world.sandboxes.disposed == [sandbox]
    assert records == {}


def test_recovery_reaps_the_sandbox_of_an_agent_whose_run_ended_unreleased(tmp_path):
    world = _World(tmp_path)

    async def scenario():
        try:
            first = world.app(
                [_call("subagent", agent="helper", task="take notes"), _done()],
                [_call("write_notes"), _done()],
            )

            async def died_before_releasing(session, event) -> None:
                del session, event

            cast(
                "Any", first._environment_lifecycle
            )._end_shared_allocation_owner = died_before_releasing
            events = await _drain(
                first.run(
                    RunRequest(
                        session_id="lead-ended",
                        agent_name="lead",
                        messages=[Message.text("user", "work with your helper")],
                    )
                )
            )
            assert events[-1].type == EventType.SESSION_COMPLETED
            _crash(first)
            assert len((await _records(first, "lead-ended"))["records"]) == 1
            recovering = world.app()
            page = await recovering.recover_incomplete_sessions(
                IncompleteSessionsRecoveryRequest(statuses={SessionStatus.COMPLETED})
            )
            return page, await _records(recovering, "lead-ended")
        finally:
            await world.close()

    page, records = asyncio.run(scenario())
    actions = {result.session_id: result.actions for result in page.results}
    assert IncompleteSessionRecoveryAction.REAPED_ALLOCATION in actions["lead-ended"]
    assert len(world.sandboxes.disposed) == 1
    assert records == {}


def test_recovery_reads_the_owner_after_the_snapshot_so_a_racing_resume_keeps_its_sandbox(
    tmp_path,
):
    world = _World(tmp_path)

    async def scenario():
        try:
            first = world.app(
                [_call("subagent", agent="helper", task="take notes"), _done()],
                [_call("write_notes"), _done()],
            )

            async def died_before_releasing(session, event) -> None:
                del session, event

            cast(
                "Any", first._environment_lifecycle
            )._end_shared_allocation_owner = died_before_releasing
            await _drain(
                first.run(
                    RunRequest(
                        session_id="lead-race-reap",
                        agent_name="lead",
                        messages=[Message.text("user", "work with your helper")],
                    )
                )
            )
            _crash(first)
            recovering = world.app()
            registry = recovering._environment_lifecycle.shared_allocations
            records = registry._records
            assert records is not None
            original_load = records.load
            raced = []

            async def load_while_a_resume_claims(owner_session_id):
                # Recovery has seen the session completed; before it reads the
                # records, another process resumes the session and claims them.
                if not raced:
                    raced.append(True)
                    await recovering.session_store.update_status(
                        owner_session_id, SessionStatus.RUNNING
                    )
                    snapshot = await original_load(owner_session_id)
                    assert snapshot is not None
                    ((record_id, record),) = snapshot.records.items()
                    await records.claim(
                        record_id=record_id,
                        scope=record.scope,
                        scope_id=record.scope_id,
                        owner_session_id=owner_session_id,
                        environment_name=record.environment_name,
                        agent_name=record.agent_name,
                        attempt_id="run:resumed:9",
                        attempt_sequence=9,
                    )
                return await original_load(owner_session_id)

            cast("Any", records).load = load_while_a_resume_claims
            reaped = await registry.reap_recorded("lead-race-reap")
            return reaped, await _records(recovering, "lead-race-reap")
        finally:
            await world.close()

    reaped, records = asyncio.run(scenario())
    # The resumed run's sandbox and its claim are left alone.
    assert reaped == 0
    assert world.sandboxes.disposed == []
    ((record),) = records["records"].values()
    assert record["attempt_id"] == "run:resumed:9"


def test_a_fork_never_claims_its_sources_shared_allocation(tmp_path):
    from cayu.runtime._fork_source_snapshot import strip_source_owned_fork_checkpoint_state

    world = _World(tmp_path)

    async def scenario():
        try:
            first = await _crash_after_the_helper_wrote(world, "lead-source")
            store = first._runtime_session_store
            checkpoint = await store.load_checkpoint("lead-source") or {}
            source_records = checkpoint[SHARED_ALLOCATIONS_CHECKPOINT_KEY]
            # A fork's checkpoint is stripped of the source's records...
            copied = dict(checkpoint)
            strip_source_owned_fork_checkpoint_state(copied)
            assert SHARED_ALLOCATIONS_CHECKPOINT_KEY not in copied
            # ...and a record the fork's checkpoint still carries (copied before
            # the key was stripped) is never claimed as the fork's: the claim
            # gives the fork a fresh allocation of its own.
            await store.create(
                RunRequest(session_id="lead-fork", agent_name="lead", messages=[]),
                identity=SessionIdentity(provider_name="lead-provider", model="scripted"),
            )
            (record_id, source_record), *_ = source_records["records"].items()

            def inherit(session, current):
                del session
                return {**(current or {}), SHARED_ALLOCATIONS_CHECKPOINT_KEY: source_records}

            await store.transform_checkpoint("lead-fork", inherit)
            claim = await SharedAllocationRecords(store).claim(
                record_id=record_id,
                scope=source_record["scope"],
                scope_id=source_record["scope_id"],
                owner_session_id="lead-fork",
                environment_name="sandbox",
                agent_name="helper",
                attempt_id="run:fork:1",
                attempt_sequence=1,
            )
            return source_record, claim
        finally:
            await world.close()

    source_record, claim = asyncio.run(scenario())
    assert claim is not None and claim.previous is None
    assert claim.record.owner_session_id == "lead-fork"
    assert claim.record.allocation_session_id != source_record["allocation_session_id"]
    assert claim.record.allocation_session_id.startswith("lead-fork.shared-")


def test_a_resumed_agent_releases_its_crashed_runs_sandbox_without_using_it_again(tmp_path):
    world = _World(tmp_path)

    async def scenario():
        try:
            await _crash_after_the_helper_wrote(world, "lead-quiet")
            # The resumed lead finishes without its helper or any workspace tool.
            second = world.app([_done()])
            events = await _drain(
                second.resume(
                    ResumeRequest(
                        session_id="lead-quiet", messages=[Message.text("user", "wrap up")]
                    )
                )
            )
            return events, await _records(second, "lead-quiet")
        finally:
            await world.close()

    events, records = asyncio.run(scenario())
    assert events[-1].type == EventType.SESSION_COMPLETED, events[-1].payload
    created, reconnected = world.sandboxes.requests
    assert created.operation is EnvironmentFactoryOperation.CREATE
    # The lead's own environment is the shared scope it owns, so its resumed
    # run reattaches to the recorded sandbox when it sets up, even with no
    # tool using it, and releases it at its terminal event.
    assert reconnected.operation is EnvironmentFactoryOperation.RECONNECT
    assert reconnected.session_id == created.session_id
    assert world.sandboxes.disposed == [sha256(created.session_id.encode()).hexdigest()[:16]]
    assert records == {}
