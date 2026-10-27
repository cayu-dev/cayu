"""Workers reach the app's stores, drop environments safely, and forks rebuild per-job ones."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest
from cayu._application_environment_maintenance import (
    _checkpoint_owes_environment_cleanup,
    ensure_session_environment,
)
from cayu.artifacts import LocalArtifactStore
from cayu.environments import Environment, EnvironmentSpec
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
)
from cayu.environments.rebuild import (
    EnvironmentRebuild,
    EnvironmentRebuildMismatch,
    EnvironmentRebuildRecord,
    EnvironmentRebuildRequest,
)
from cayu.evals import ScriptedModelProvider
from cayu.events import Event
from cayu.exceptions import EnvironmentInUse
from cayu.fork_controllers import ForkAgentAvailability, ForkController
from cayu.knowledge.scopes import KnowledgeAccessScope
from cayu.providers import ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest
from cayu.sessions.requests import ForkSessionRequest
from cayu.storage import SQLiteSessionStore
from cayu.storage.knowledge_memory import InMemoryKnowledgeStore
from cayu.tasks import InMemoryTaskStore
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.vaults import SecretRedactor

SLOT = "run-1-slot-0"


class CloneFactory(EnvironmentFactory):
    """A per-job factory over one clone revision that can be rebuilt elsewhere."""

    def __init__(self, revision: str = "abc123", *, gate: asyncio.Event | None = None) -> None:
        self.revision = revision
        self.gate = gate
        self.started = asyncio.Event()
        self.created: list[str] = []

    @property
    def rebuild_record(self) -> EnvironmentRebuildRecord:
        return EnvironmentRebuildRecord(
            "scout.clone", {"source": "https://example.com/org/repo", "revision": self.revision}
        )

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        # Sessions resumed in another process keep their profile only with a
        # stable declaration, which a rebuildable factory needs anyway.
        return ExecutionProfileBehaviorIdentity(
            name="scout.clone", behavior_version="1", implementation_version=self.revision
        )

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        self.created.append(request.session_id)
        return EnvironmentFactoryResult(
            environment=Environment(EnvironmentSpec(name=request.environment_name))
        )


def _slot_spec(name: str = SLOT) -> EnvironmentSpec:
    return EnvironmentSpec(
        name=name,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="scout.slot", behavior_version="1", implementation_version="1"
        ),
    )


def _app(store, **options) -> CayuApp:
    app = CayuApp(session_store=store, enable_logging=False, **options)
    app.register_provider(
        ScriptedModelProvider([[ModelStreamEvent.completed()]] * 8, name="fake"), default=True
    )
    app.register_agent(AgentSpec(name="hunter", model="small"))
    return app


async def _run(app: CayuApp, *, environment_name: str = SLOT) -> list[Event]:
    return [
        event
        async for event in app.run(
            RunRequest(
                agent_name="hunter",
                environment_name=environment_name,
                messages=[Message.text("user", "hunt")],
            )
        )
    ]


def test_task_and_knowledge_stores_are_read_only_app_properties() -> None:
    tasks = InMemoryTaskStore()
    scope = KnowledgeAccessScope.for_namespace("scout")
    knowledge = InMemoryKnowledgeStore()
    app = CayuApp(
        task_store=tasks,
        knowledge_store=knowledge,
        knowledge_access_scope=scope,
        enable_logging=False,
    )
    bare = CayuApp(enable_logging=False)

    assert app.task_store is tasks
    assert app.knowledge_store is knowledge
    assert bare.task_store is None and bare.knowledge_store is None
    assert isinstance(CayuApp.task_store, property)
    assert isinstance(CayuApp.knowledge_store, property)
    with pytest.raises(AttributeError):
        app.task_store = InMemoryTaskStore()  # type: ignore[misc]
    with pytest.raises(AttributeError):
        app.knowledge_store = None  # type: ignore[misc]


def test_unregister_drops_a_finished_environment_and_a_later_resume_fails_closed() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        app = _app(store)
        app.register_environment_factory(EnvironmentSpec(name=SLOT), CloneFactory())
        events = await _run(app)
        assert events[-1].type is EventType.SESSION_COMPLETED
        session_id = events[-1].session_id

        await app.unregister_environment(SLOT)

        assert SLOT not in app.list_environments()
        with pytest.raises(KeyError, match="not registered"):
            await app.unregister_environment(SLOT)
        # Like a process that never registered the name: refused before any work.
        before = await store.load(session_id)
        with pytest.raises(KeyError, match="Environment not registered"):
            async for _ in app.resume(
                ResumeRequest(session_id=session_id, messages=[Message.text("user", "more")])
            ):
                pass
        after = await store.load(session_id)
        assert after is not None and before is not None
        assert after.status is SessionStatus.COMPLETED
        assert after.run_epoch == before.run_epoch
        # The name can be registered again, for example for the next job.
        app.register_environment_factory(EnvironmentSpec(name=SLOT), CloneFactory("def456"))
        await app.aclose()

    asyncio.run(scenario())


def test_unregister_refuses_the_default_environments() -> None:
    async def scenario() -> None:
        app = _app(InMemorySessionStore())
        app.register_environment(Environment(EnvironmentSpec(name="main")), default=True)
        app.register_environment(Environment(EnvironmentSpec(name="review")))
        app.register_agent(AgentSpec(name="reviewer", model="small"), environment_name="review")

        with pytest.raises(EnvironmentInUse) as default:
            await app.unregister_environment("main")
        with pytest.raises(EnvironmentInUse) as agent_default:
            await app.unregister_environment("review")

        assert default.value.reason == "default_environment"
        assert agent_default.value.reason == "agent_default"
        assert app.list_environments() == ("main", "review")
        await app.aclose()

    asyncio.run(scenario())


def test_unregister_refuses_while_a_session_is_bound_here_or_unfinished_elsewhere(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        database = tmp_path / "sessions.sqlite"
        gate = asyncio.Event()
        factory = CloneFactory(gate=gate)
        worker = _app(SQLiteSessionStore(database))
        worker.register_environment_factory(EnvironmentSpec(name=SLOT), factory)
        # Another process of the same application, with the same name registered.
        other = _app(SQLiteSessionStore(database))
        other.register_environment_factory(EnvironmentSpec(name=SLOT), CloneFactory())

        run = asyncio.create_task(_run(worker))
        await factory.started.wait()
        with pytest.raises(EnvironmentInUse) as here:
            await worker.unregister_environment(SLOT)
        with pytest.raises(EnvironmentInUse) as elsewhere:
            await other.unregister_environment(SLOT)
        gate.set()
        events = await run

        assert here.value.reason == "in_process_work"
        assert elsewhere.value.reason == "unfinished_session"
        assert SLOT in worker.list_environments() and SLOT in other.list_environments()
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert await worker.drain_environment_cleanups()
        await worker.unregister_environment(SLOT)
        await other.unregister_environment(SLOT)
        await worker.aclose()
        await other.aclose()

    asyncio.run(scenario())


def test_terminal_sessions_block_unregistering_only_while_they_owe_cleanup() -> None:
    name = SLOT
    assert not _checkpoint_owes_environment_cleanup(None, name)
    assert not _checkpoint_owes_environment_cleanup(
        {"environment_factory_reconnect": {name: {"id": "kept for reconnect"}}}, name
    )
    assert _checkpoint_owes_environment_cleanup(
        {"pending_completion_finalization": {"environment_name": name}}, name
    )
    assert not _checkpoint_owes_environment_cleanup(
        {"pending_completion_finalization": {"environment_name": "other"}}, name
    )
    assert _checkpoint_owes_environment_cleanup(
        {"environment_factory_pending_disposals": {name: {"state": "pending"}}}, name
    )


def test_unregister_consults_the_lifecycle_census_of_pending_allocations(monkeypatch) -> None:
    # Unreaped intents and published receipts whose setup failed before the
    # first transcript are the lifecycle's census; a FAILED session can hold
    # a receipt with nothing else pending, and its factory must stay.
    async def scenario() -> None:
        app = _app(InMemorySessionStore())
        app.register_environment_factory(EnvironmentSpec(name=SLOT), CloneFactory())
        await _run(app)
        lifecycle = app._environment_lifecycle
        seen: list[str] = []

        async def pending(session):
            seen.append(session.id)
            return (SLOT,)

        monkeypatch.setattr(lifecycle, "pending_allocation_names", pending)
        with pytest.raises(EnvironmentInUse) as refused:
            await app.unregister_environment(SLOT)
        assert refused.value.reason == "pending_cleanup"
        assert len(seen) == 1 and SLOT in app.list_environments()

        async def corrupt(session):
            raise ValueError("Environment allocation intents must be an object.")

        monkeypatch.setattr(lifecycle, "pending_allocation_names", corrupt)
        with pytest.raises(EnvironmentInUse):
            await app.unregister_environment(SLOT)
        await app.aclose()

    asyncio.run(scenario())


def test_unregister_rechecks_an_idle_release_that_started_during_the_scan(monkeypatch) -> None:
    class SpareFactory(CloneFactory):
        def __init__(self) -> None:
            super().__init__()
            self.release_gate = asyncio.Event()

        async def release_idle_resources(self) -> None:
            await self.release_gate.wait()

    async def scenario() -> None:
        store = InMemorySessionStore()
        app = _app(store)
        factory = SpareFactory()
        app.register_environment_factory(EnvironmentSpec(name=SLOT), factory)
        list_sessions = store.list_sessions

        async def list_while_draining(query=None):
            # A drain starts the factory's release while the scan is suspended.
            assert not await app.drain_environment_cleanups(timeout_s=0.05)
            return await list_sessions(query)

        monkeypatch.setattr(store, "list_sessions", list_while_draining)
        with pytest.raises(EnvironmentInUse) as refused:
            await app.unregister_environment(SLOT)
        assert refused.value.reason == "in_process_work"
        assert SLOT in app.list_environments()
        monkeypatch.setattr(store, "list_sessions", list_sessions)
        factory.release_gate.set()
        assert await app.drain_environment_cleanups()
        await app.unregister_environment(SLOT)
        await app.aclose()

    asyncio.run(scenario())


def test_unregister_keeps_the_artifact_store_for_session_deletion(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(InMemorySessionStore())
        artifacts = LocalArtifactStore(tmp_path / "artifacts")
        app.register_environment(Environment(EnvironmentSpec(name=SLOT), artifact_store=artifacts))

        await app.unregister_environment(SLOT)

        assert app.list_environments() == ()
        assert app._environment_registry.registered_artifact_stores() == (artifacts,)
        await app.aclose()

    asyncio.run(scenario())


def test_a_serving_process_rebuilds_a_per_job_environment_to_fork_and_resume(
    tmp_path: Path,
) -> None:
    database = tmp_path / "sessions.sqlite"
    rebuilt: list[EnvironmentRebuildRequest] = []

    async def rebuild(request: EnvironmentRebuildRequest) -> EnvironmentRebuild:
        rebuilt.append(request)
        return EnvironmentRebuild(
            spec=_slot_spec(request.environment_name),
            factory=CloneFactory(request.record.payload["revision"]),
        )

    def serving() -> CayuApp:
        app = _app(SQLiteSessionStore(database))
        app.register_environment_rebuilder("scout.clone", rebuild)
        return app

    async def scenario() -> tuple[str, str, str]:
        worker = _app(SQLiteSessionStore(database))
        worker.register_environment_factory(_slot_spec(), CloneFactory())
        events = await _run(worker)
        source_id = events[-1].session_id
        await worker.unregister_environment(SLOT)
        await worker.aclose()

        first = serving()
        fork = [
            e async for e in first.fork_session(ForkSessionRequest(source_session_id=source_id))
        ]
        fork_id = fork[-1].session_id
        assert SLOT in first.list_environments()
        await first.aclose()

        # A fresh serving process reads the record persisted on the unrun fork.
        second = serving()
        refork = [
            e async for e in second.fork_session(ForkSessionRequest(source_session_id=fork_id))
        ]
        resumed = [
            e
            async for e in second.resume(
                ResumeRequest(session_id=fork_id, messages=[Message.text("user", "dig")])
            )
        ]
        assert resumed[-1].type is EventType.SESSION_COMPLETED
        completed = await second.session_store.query_events(
            EventQuery(session_id=fork_id, event_type=EventType.ENVIRONMENT_FACTORY_COMPLETED)
        )
        assert completed[-1].event.payload["environment_rebuild"] == {
            "kind": "scout.clone",
            "payload": {"source": "https://example.com/org/repo", "revision": "abc123"},
        }
        await second.aclose()
        return source_id, fork_id, refork[-1].session_id

    source_id, fork_id, _ = asyncio.run(scenario())
    assert [(r.environment_name, r.session_id) for r in rebuilt] == [
        (SLOT, source_id),
        (SLOT, fork_id),
    ]
    assert rebuilt[0].record == CloneFactory().rebuild_record
    assert fork_id != source_id


def test_a_rebuild_must_recreate_the_same_environment() -> None:
    async def scenario(result: EnvironmentRebuild) -> CayuApp:
        store = InMemorySessionStore()
        worker = _app(store)
        worker.register_environment_factory(EnvironmentSpec(name=SLOT), CloneFactory())
        source_id = (await _run(worker))[-1].session_id
        await worker.aclose()
        serving = _app(store)

        async def rebuild(_request: EnvironmentRebuildRequest) -> EnvironmentRebuild:
            return result

        serving.register_environment_rebuilder("scout.clone", rebuild)
        with pytest.raises(ValueError):
            async for _ in serving.fork_session(ForkSessionRequest(source_session_id=source_id)):
                pass
        assert serving.list_environments() == ()
        return serving

    for result in (
        EnvironmentRebuild(spec=EnvironmentSpec(name="other"), factory=CloneFactory()),
        EnvironmentRebuild(spec=EnvironmentSpec(name=SLOT), factory=CloneFactory("other-rev")),
    ):
        asyncio.run(scenario(result))


def _rebuilding_app(store, built: list[str]) -> CayuApp:
    async def rebuild(request: EnvironmentRebuildRequest) -> EnvironmentRebuild:
        built.append(request.record.payload["revision"])
        return EnvironmentRebuild(
            spec=_slot_spec(request.environment_name),
            factory=CloneFactory(request.record.payload["revision"]),
        )

    app = _app(store)
    app.register_environment_rebuilder("scout.clone", rebuild)
    return app


def test_a_reused_name_never_runs_a_session_in_another_records_environment(
    tmp_path: Path,
) -> None:
    database = tmp_path / "sessions.sqlite"
    built: list[str] = []

    async def scenario() -> None:
        # One slot name, reused for two jobs over different clones.
        worker = _app(SQLiteSessionStore(database))
        worker.register_environment_factory(_slot_spec(), CloneFactory("job-a"))
        job_a = (await _run(worker))[-1].session_id
        await worker.unregister_environment(SLOT)
        job_b_factory = CloneFactory("job-b")
        worker.register_environment_factory(_slot_spec(), job_b_factory)
        job_b = (await _run(worker))[-1].session_id

        # The worker itself refuses to continue job A in job B's clone.
        before = await worker.session_store.load(job_a)
        with pytest.raises(EnvironmentRebuildMismatch) as in_worker:
            async for _ in worker.resume(
                ResumeRequest(session_id=job_a, messages=[Message.text("user", "more")])
            ):
                pass
        after = await worker.session_store.load(job_a)
        assert in_worker.value.session_id == job_a
        assert before is not None and after is not None and after.run_epoch == before.run_epoch
        assert job_b_factory.created == [job_b]
        await worker.aclose()

        serving = _rebuilding_app(SQLiteSessionStore(database), built)
        fork_a = [
            e async for e in serving.fork_session(ForkSessionRequest(source_session_id=job_a))
        ][-1].session_id
        # The name now holds job A's rebuilt clone: job B must not reuse it.
        with pytest.raises(EnvironmentRebuildMismatch):
            async for _ in serving.fork_session(ForkSessionRequest(source_session_id=job_b)):
                pass
        with pytest.raises(EnvironmentRebuildMismatch):
            async for _ in serving.resume(
                ResumeRequest(session_id=job_b, messages=[Message.text("user", "dig")])
            ):
                pass
        resumed = [
            e
            async for e in serving.resume(
                ResumeRequest(session_id=fork_a, messages=[Message.text("user", "dig")])
            )
        ]
        assert resumed[-1].type is EventType.SESSION_COMPLETED
        await serving.aclose()

    asyncio.run(scenario())
    assert built == ["job-a"]


def test_a_factory_that_failed_or_crashed_can_still_be_rebuilt() -> None:
    class FailingFactory(CloneFactory):
        async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
            raise RuntimeError("allocation lost")

    async def scenario() -> list[str]:
        store = InMemorySessionStore()
        worker = _app(store)
        worker.register_environment_factory(_slot_spec(), FailingFactory())
        events = await _run(worker)
        assert events[-1].type is EventType.SESSION_FAILED
        session_id = events[-1].session_id
        completed = await store.query_events(
            EventQuery(session_id=session_id, event_type=EventType.ENVIRONMENT_FACTORY_COMPLETED)
        )
        assert completed == []
        await worker.aclose()

        built: list[str] = []
        serving = _rebuilding_app(store, built)
        await ensure_session_environment(serving, session_id)
        assert SLOT in serving.list_environments()
        await serving.aclose()
        return built

    assert asyncio.run(scenario()) == ["abc123"]


def test_every_incomplete_session_recovery_path_rebuilds_first() -> None:
    async def scenario() -> list[str]:
        store = InMemorySessionStore()
        worker = _app(store)
        worker.register_environment_factory(_slot_spec(), CloneFactory())
        session_id = (await _run(worker))[-1].session_id
        await worker.aclose()

        built: list[str] = []
        serving = _rebuilding_app(store, built)
        session = await store.load(session_id)
        assert session is not None
        # The shared boundary under single, batch, task and work-attempt recovery.
        await serving._incomplete_recovery.recover_incomplete_session_scoped(
            session=session, inactive_for_seconds=None, reason="test", metadata={}
        )
        assert SLOT in serving.list_environments()
        await serving.aclose()
        return built

    assert asyncio.run(scenario()) == ["abc123"]


class DeferredCloneFactory(CloneFactory):
    """A clone whose crash leftovers recovery releases through the factory."""

    def __init__(self, revision: str = "abc123") -> None:
        super().__init__(revision)
        self.released: list[tuple[str, dict]] = []

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        # One clone implementation for every job; the record names the clone.
        return ExecutionProfileBehaviorIdentity(
            name="scout.clone", behavior_version="1", implementation_version="1"
        )

    @property
    def deferred_materialization(self) -> bool:
        return True

    async def release_deferred_materialization(self, request: EnvironmentFactoryRequest) -> bool:
        self.released.append((request.session_id, request.reconnect_metadata))
        return True

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.created.append(request.session_id)
        return EnvironmentFactoryResult(
            environment=Environment(EnvironmentSpec(name=request.environment_name)),
            reconnect_metadata={"clone": self.revision},
        )


class HangingTool(Tool):
    spec = ToolSpec(
        name="hang",
        description="Never returns, standing in for a worker that crashed mid-round.",
        input_schema={"type": "object"},
        effect=ToolEffect.NONE,
        max_terminal_payload_bytes=1024,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:hang", behavior_version="1", implementation_version="1"
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def _crashing_app(store) -> tuple[CayuApp, HangingTool]:
    tool = HangingTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="call-1", name="hang", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.completed()],
            ],
            name="fake",
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="hunter", model="small"), tools=[tool])
    return app, tool


@pytest.mark.parametrize("rebuilder", [True, False])
def test_a_crashed_session_in_a_reused_slot_settles_without_touching_the_new_job(
    tmp_path: Path, rebuilder: bool
) -> None:
    database = tmp_path / "sessions.sqlite"

    async def scenario() -> None:
        # Job A crashes mid-round in the slot (its worker never finishes).
        crashed, tool = _crashing_app(SQLiteSessionStore(database))
        job_a_factory = DeferredCloneFactory("job-a")
        crashed.register_environment_factory(_slot_spec(), job_a_factory)
        events: list[Event] = []

        async def run_job_a() -> None:
            async for event in crashed.run(
                RunRequest(
                    agent_name="hunter",
                    environment_name=SLOT,
                    messages=[Message.text("user", "hunt")],
                )
            ):
                events.append(event)

        job_a_run = asyncio.create_task(run_job_a())
        await tool.started.wait()
        job_a = events[0].session_id

        # A restarted worker registers the slot for job B and recovers job A.
        worker, _ = _crashing_app(SQLiteSessionStore(database))
        job_b_factory = DeferredCloneFactory("job-b")
        worker.register_environment_factory(_slot_spec(), job_b_factory)
        rebuilt: list[DeferredCloneFactory] = []
        if rebuilder:

            async def rebuild(request: EnvironmentRebuildRequest) -> EnvironmentRebuild:
                factory = DeferredCloneFactory(request.record.payload["revision"])
                rebuilt.append(factory)
                return EnvironmentRebuild(spec=_slot_spec(), factory=factory)

            worker.register_environment_rebuilder("scout.clone", rebuild)

        result = await worker.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=job_a)
        )

        assert result.status is SessionStatus.INTERRUPTED
        # Job B's factory never saw job A's allocation, and the slot still holds B.
        assert job_b_factory.created == [] and job_b_factory.released == []
        assert worker.get_environment_factory(SLOT) is job_b_factory
        if rebuilder:
            # Job A's own environment, rebuilt separately, released its leftover.
            assert [factory.released for factory in rebuilt] == [[(job_a, {"clone": "job-a"})]]
        else:
            assert rebuilt == []
        # Job A no longer pins the slot: it can never run under job B's factory.
        await worker.unregister_environment(SLOT)

        job_a_run.cancel()
        with suppress(BaseException):
            await job_a_run
        await worker.aclose()
        with suppress(BaseException):
            await crashed.aclose()

    asyncio.run(scenario())


async def _crash_job_a_in_reused_slot(database: Path, factory_type=None, artifact_store=None):
    """Job A crashes mid-round in the slot; a restarted worker registers it for job B."""

    factory_type = factory_type or DeferredCloneFactory
    crashed, tool = _crashing_app(SQLiteSessionStore(database))
    crashed.register_environment_factory(
        _slot_spec(), factory_type("job-a"), artifact_store=artifact_store
    )
    events: list[Event] = []

    async def run_job_a() -> None:
        async for event in crashed.run(
            RunRequest(
                agent_name="hunter", environment_name=SLOT, messages=[Message.text("user", "go")]
            )
        ):
            events.append(event)

    job_a_run = asyncio.create_task(run_job_a())
    await tool.started.wait()
    worker, _ = _crashing_app(SQLiteSessionStore(database))
    job_b_factory = factory_type("job-b")
    worker.register_environment_factory(_slot_spec(), job_b_factory, artifact_store=artifact_store)

    async def stop() -> None:
        job_a_run.cancel()
        with suppress(BaseException):
            await job_a_run
        await worker.aclose()
        with suppress(BaseException):
            await crashed.aclose()

    return worker, job_b_factory, events[0].session_id, stop


def test_recovery_planning_reports_a_rebuild_without_calling_the_rebuilder(
    tmp_path: Path,
) -> None:
    from cayu.sessions.recovery import RecoveryPlanRequest, RecoveryPlanSelection

    async def scenario() -> None:
        worker, job_b_factory, job_a, stop = await _crash_job_a_in_reused_slot(
            tmp_path / "sessions.sqlite"
        )
        calls: list[EnvironmentRebuildRequest] = []

        async def rebuild(request: EnvironmentRebuildRequest) -> EnvironmentRebuild:
            calls.append(request)
            raise AssertionError("planning must not rebuild")

        worker.register_environment_rebuilder("scout.clone", rebuild)
        session = await worker.session_store.load(job_a)
        assert session is not None

        preflight = await worker._incomplete_recovery.preflight_incomplete_session(
            session=session, inactive_for_seconds=None
        )
        plan = await worker.plan_recovery(
            RecoveryPlanRequest(selection=RecoveryPlanSelection(session_ids=(job_a,)))
        )

        del preflight
        assert calls == []
        assert [item.registration.reason_code for item in plan.items] == [
            "environment_rebuild_required:scout.clone"
        ]
        # Nothing was created or registered, and job B's factory saw nothing.
        assert worker.list_environments() == (SLOT,)
        assert worker.get_environment_factory(SLOT) is job_b_factory
        assert dict(worker._environment_registry.retired_scoped_registrations) == {}
        assert job_b_factory.created == [] and job_b_factory.released == []
        loaded = await worker.session_store.load(job_a)
        assert loaded is not None and loaded.status is SessionStatus.RUNNING
        await stop()

    asyncio.run(scenario())


def test_a_failing_rebuilder_falls_back_to_settling_the_session(tmp_path: Path) -> None:
    async def scenario() -> None:
        worker, job_b_factory, job_a, stop = await _crash_job_a_in_reused_slot(
            tmp_path / "sessions.sqlite"
        )

        async def rebuild(request: EnvironmentRebuildRequest) -> EnvironmentRebuild:
            raise ValueError("Unknown clone source.")

        worker.register_environment_rebuilder("scout.clone", rebuild)
        result = await worker.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=job_a)
        )
        assert result.status is SessionStatus.INTERRUPTED
        assert job_b_factory.created == [] and job_b_factory.released == []
        await stop()

    asyncio.run(scenario())


def test_a_recovery_rebuilt_factory_has_its_idle_resources_closed(tmp_path: Path) -> None:
    class PooledClone(DeferredCloneFactory):
        def __init__(self, revision: str) -> None:
            super().__init__(revision)
            self.closed = 0

        async def close_idle_resources(self) -> None:
            self.closed += 1

    async def scenario() -> None:
        worker, _job_b_factory, job_a, stop = await _crash_job_a_in_reused_slot(
            tmp_path / "sessions.sqlite"
        )
        rebuilt: list[PooledClone] = []

        async def rebuild(request: EnvironmentRebuildRequest) -> EnvironmentRebuild:
            factory = PooledClone(request.record.payload["revision"])
            rebuilt.append(factory)
            return EnvironmentRebuild(spec=_slot_spec(), factory=factory)

        worker.register_environment_rebuilder("scout.clone", rebuild)
        await worker.recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id=job_a))
        registry = worker._environment_registry
        assert len(registry.retired_scoped_registrations) == 1

        assert await worker.drain_environment_cleanups()

        assert [factory.closed for factory in rebuilt] == [1]
        assert dict(registry.retired_scoped_registrations) == {}
        await stop()

    asyncio.run(scenario())


def test_the_stand_in_keeps_naming_the_sessions_own_record(tmp_path: Path) -> None:
    from cayu._application_environment_maintenance import SessionEnvironments
    from cayu.runtime._environment_lifecycle import _environment_factory_base_payload

    async def scenario() -> None:
        worker, job_b_factory, job_a, stop = await _crash_job_a_in_reused_slot(
            tmp_path / "sessions.sqlite"
        )
        session = await worker.session_store.load(job_a)
        assert session is not None
        job_a_record = DeferredCloneFactory("job-a").rebuild_record

        # No rebuilder: recovery resolves the slot to the call-refusing stand-in.
        async with SessionEnvironments(worker).recovery_scope(session):
            registered = worker._environment_registry.get(SLOT)
            assert registered is not None and registered.factory is not job_b_factory
            # A refused factory call records the session's own record, so it never
            # supersedes which environment the session belongs to.
            payload = _environment_factory_base_payload(
                session=session, registered_environment=registered
            )
            assert payload["environment_rebuild"] == job_a_record.to_payload()
            request = EnvironmentFactoryRequest(
                session_id=job_a, agent_name="hunter", environment_name=SLOT
            )
            with pytest.raises(EnvironmentRebuildMismatch):
                await registered.factory.create(request)
        assert worker.get_environment_factory(SLOT) is job_b_factory
        await stop()

    asyncio.run(scenario())


class RevisionedClone(DeferredCloneFactory):
    """A clone whose declared implementation differs per job."""

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="scout.clone", behavior_version="1", implementation_version=self.revision
        )


def test_planning_defers_environment_compatibility_to_the_rebuild(tmp_path: Path) -> None:
    from cayu.sessions.recovery import (
        RecoveryBlockerCode,
        RecoveryPlanAction,
        RecoveryPlanRequest,
        RecoveryPlanSelection,
        RecoveryRegistrationStatus,
    )

    async def scenario() -> None:
        worker, job_b_factory, job_a, stop = await _crash_job_a_in_reused_slot(
            tmp_path / "sessions.sqlite", RevisionedClone
        )
        rebuilt: list[RevisionedClone] = []

        async def rebuild(request: EnvironmentRebuildRequest) -> EnvironmentRebuild:
            factory = RevisionedClone(request.record.payload["revision"])
            rebuilt.append(factory)
            return EnvironmentRebuild(spec=_slot_spec(), factory=factory)

        worker.register_environment_rebuilder("scout.clone", rebuild)
        plan = await worker.plan_recovery(
            RecoveryPlanRequest(selection=RecoveryPlanSelection(session_ids=(job_a,)))
        )
        (item,) = plan.items
        # Job B's identity is not job A's; the rebuild, not planning, decides.
        assert item.registration.status is RecoveryRegistrationStatus.READY
        assert item.registration.reason_code == "environment_rebuild_required:scout.clone"
        # Only the crash's own tool decision remains, as for any crashed round.
        assert RecoveryBlockerCode.REGISTRATION_INCOMPATIBLE not in {
            blocker.code for blocker in item.blockers
        }
        assert item.allowed_actions != (RecoveryPlanAction.LEAVE_INTACT,)
        assert rebuilt == []

        result = await worker.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=job_a)
        )
        assert result.status is SessionStatus.INTERRUPTED
        assert [factory.released for factory in rebuilt] == [[(job_a, {"clone": "job-a"})]]
        assert job_b_factory.released == []
        await stop()

    asyncio.run(scenario())


def test_an_unreadable_record_never_hands_the_allocation_to_the_slots_factory(
    tmp_path: Path, monkeypatch
) -> None:
    async def scenario() -> None:
        worker, job_b_factory, job_a, stop = await _crash_job_a_in_reused_slot(
            tmp_path / "sessions.sqlite"
        )

        async def rebuild(request: EnvironmentRebuildRequest) -> EnvironmentRebuild:
            return EnvironmentRebuild(
                spec=_slot_spec(), factory=DeferredCloneFactory(request.record.payload["revision"])
            )

        worker.register_environment_rebuilder("scout.clone", rebuild)
        store = worker.session_store
        query_events = store.query_events
        failures = [OSError("event store unavailable")]

        async def flaky_query_events(query=None):
            if (
                query is not None
                and EventType.ENVIRONMENT_FACTORY_STARTED in query.event_types
                and failures
            ):
                raise failures.pop()
            return await query_events(query)

        monkeypatch.setattr(store, "query_events", flaky_query_events)
        with pytest.raises(EnvironmentRebuildMismatch) as refused:
            await worker.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=job_a)
            )
        assert refused.value.verified is False
        assert job_b_factory.released == [] and job_b_factory.created == []
        loaded = await store.load(job_a)
        assert loaded is not None and loaded.status is SessionStatus.RUNNING

        # Once the store answers, the same recovery settles the session.
        result = await worker.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=job_a)
        )
        assert result.status is SessionStatus.INTERRUPTED
        assert job_b_factory.released == []
        await stop()

    asyncio.run(scenario())


def test_rebuild_records_are_validated_at_registration() -> None:
    redactor = SecretRedactor("s3cr3t-token-value")

    class LeakyFactory(CloneFactory):
        @property
        def rebuild_record(self) -> EnvironmentRebuildRecord:
            return EnvironmentRebuildRecord("scout.clone", {"token": "s3cr3t-token-value"})

    class WrongTypeFactory(CloneFactory):
        @property
        def rebuild_record(self):  # type: ignore[override]
            return {"kind": "scout.clone", "payload": {}}

    app = CayuApp(secret_redactor=redactor, enable_logging=False)
    with pytest.raises(ValueError, match="secret"):
        app.register_environment_factory(EnvironmentSpec(name="leaky"), LeakyFactory())
    with pytest.raises(TypeError):
        app.register_environment_factory(EnvironmentSpec(name="wrong"), WrongTypeFactory())
    with pytest.raises(ValueError):
        EnvironmentRebuildRecord("", {})
    with pytest.raises(ValueError):
        EnvironmentRebuildRecord("scout.clone", {"blob": "x" * (17 * 1024)})
    with pytest.raises(ValueError):
        EnvironmentRebuildRecord.from_payload({"kind": "scout.clone"})
    with pytest.raises(ValueError, match="already registered"):

        async def rebuild(_request):
            raise AssertionError

        app.register_environment_rebuilder("scout.clone", rebuild)
        app.register_environment_rebuilder("scout.clone", rebuild)
    assert app.list_environments() == ()


def test_enrollment_reports_which_agents_this_process_can_fork() -> None:
    from cayu.providers.hosted import OpenAIWebSearch

    def host(*, rebuilder: bool) -> CayuApp:
        app = _app(InMemorySessionStore())
        app.register_environment(Environment(EnvironmentSpec(name="main")))
        app.register_agent(AgentSpec(name="local", model="small"), environment_name="main")
        app.register_agent(AgentSpec(name="slots", model="small"), environment_name=SLOT)
        app.register_agent(
            AgentSpec(name="searcher", model="small"), hosted_tools=(OpenAIWebSearch(),)
        )
        if rebuilder:

            async def rebuild(_request):
                raise AssertionError

            app.register_environment_rebuilder("scout.clone", rebuild)
        app.enroll_fork_controller(
            ForkController(
                name="compass",
                subjects=("svc:compass",),
                agents=("hunter", "local", "slots", "searcher"),
            )
        )
        return app

    plain = host(rebuilder=False).fork_controller_access("svc:compass").enrollment()
    rebuilding = host(rebuilder=True).fork_controller_access("svc:compass").enrollment()

    assert plain.agent_availability == (
        ForkAgentAvailability(agent="hunter", forkable=True),
        ForkAgentAvailability(agent="local", forkable=True),
        ForkAgentAvailability(agent="slots", forkable=False, reason="environment_not_registered"),
        ForkAgentAvailability(agent="searcher", forkable=False, reason="hosted_tools_not_allowed"),
    )
    assert plain.rebuildable_environment_kinds == ()
    assert rebuilding.agent_availability[2] == ForkAgentAvailability(agent="slots", forkable=True)
    assert rebuilding.rebuildable_environment_kinds == ("scout.clone",)


@pytest.mark.parametrize("other_registration", [False, True])
def test_unrun_fork_keeps_its_record_after_source_deletion(tmp_path, other_registration):
    from cayu.runtime.session_closure import SessionClosureChildPolicy, SessionClosurePolicy

    async def scenario():
        database = tmp_path / "fork.sqlite"
        worker = _app(SQLiteSessionStore(database))
        worker.register_environment_factory(_slot_spec(), DeferredCloneFactory("job-a"))
        source = (await _run(worker))[-1].session_id
        fork = [e async for e in worker.fork_session(ForkSessionRequest(source_session_id=source))]
        child = fork[-1].session_id
        report = await worker.erase_session_closure(
            source, policy=SessionClosurePolicy(child_policy=SessionClosureChildPolicy.DETACH)
        )
        assert report.complete
        assert await worker.session_store.load(source) is None
        await worker.aclose()
        serving = _app(SQLiteSessionStore(database))
        factories = []

        async def rebuild(request):
            factory = DeferredCloneFactory(request.record.payload["revision"])
            factories.append(factory)
            return EnvironmentRebuild(spec=_slot_spec(), factory=factory)

        serving.register_environment_rebuilder("scout.clone", rebuild)
        other = DeferredCloneFactory("job-b")
        if other_registration:
            serving.register_environment_factory(_slot_spec(), other)
            before = await serving.session_store.load(child)
            with pytest.raises(EnvironmentRebuildMismatch):
                [
                    e
                    async for e in serving.resume(
                        ResumeRequest(session_id=child, messages=[Message.text("user", "more")])
                    )
                ]
            assert await serving.session_store.load(child) == before
            assert other.created == []
        else:
            refork = [
                e async for e in serving.fork_session(ForkSessionRequest(source_session_id=child))
            ]
            assert refork[-1].type is EventType.SESSION_FORKED
            resumed = [
                e
                async for e in serving.resume(
                    ResumeRequest(session_id=child, messages=[Message.text("user", "more")])
                )
            ]
            assert resumed[-1].type is EventType.SESSION_COMPLETED
            assert [f.revision for f in factories] == ["job-a"]
        await serving.aclose()

    asyncio.run(scenario())


def test_repeated_rebuilds_reuse_the_registered_artifact_store(tmp_path):
    async def scenario():
        store = InMemorySessionStore()
        worker = _app(store)
        worker.register_environment_factory(_slot_spec(), CloneFactory())
        source = (await _run(worker))[-1].session_id
        serving = _app(store)
        stores = []

        async def rebuild(request):
            artifact_store = LocalArtifactStore(tmp_path / "artifacts")
            stores.append(artifact_store)
            return EnvironmentRebuild(
                spec=_slot_spec(), factory=CloneFactory(), artifact_store=artifact_store
            )

        serving.register_environment_rebuilder("scout.clone", rebuild)
        for index in range(2):
            [
                e
                async for e in serving.fork_session(
                    ForkSessionRequest(source_session_id=source, session_id=f"rebuilt-fork-{index}")
                )
            ]
            await serving.unregister_environment(SLOT)
        assert len(stores) == 2 and stores[0] is not stores[1]
        assert serving._environment_registry.registered_artifact_stores() == (stores[0],)
        await serving.aclose()
        await worker.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("other_registration", [False, True])
def test_paused_continuation_checks_its_rebuild_record_before_mutation(
    tmp_path, other_registration
):
    from cayu.approvals import UserInputResponse
    from cayu.tools import UserInputTool

    def app(store, *, pause=False):
        result = CayuApp(session_store=store, enable_logging=False)
        rounds = [[ModelStreamEvent.completed()]]
        if pause:
            rounds.insert(
                0,
                [
                    ModelStreamEvent.tool_call(
                        id="ask", name="ask_user", arguments={"question": "Continue?"}
                    ),
                    ModelStreamEvent.completed(),
                ],
            )
        result.register_provider(ScriptedModelProvider(rounds, name="fake"), default=True)
        result.register_agent(AgentSpec(name="hunter", model="small"), tools=[UserInputTool()])
        return result

    async def scenario():
        database = tmp_path / "paused.sqlite"
        worker = app(SQLiteSessionStore(database), pause=True)
        worker.register_environment_factory(_slot_spec(), DeferredCloneFactory("job-b"))
        events = await _run(worker)
        assert events[-1].type is EventType.SESSION_INTERRUPTED
        awaiting = next(e for e in events if e.type is EventType.SESSION_AWAITING_USER_INPUT)
        await worker.aclose()
        serving = app(SQLiteSessionStore(database))
        other = DeferredCloneFactory("job-a")
        rebuilt = []

        async def rebuild(request):
            factory = DeferredCloneFactory(request.record.payload["revision"])
            rebuilt.append(factory)
            return EnvironmentRebuild(spec=_slot_spec(), factory=factory)

        serving.register_environment_rebuilder("scout.clone", rebuild)
        response = UserInputResponse(
            session_id=awaiting.session_id, input_id=awaiting.payload["input_id"], answer="yes"
        )
        if other_registration:
            serving.register_environment_factory(_slot_spec(), other)
            before = await serving.session_store.load(awaiting.session_id)
            checkpoint = await serving.session_store.load_checkpoint(awaiting.session_id)
            with pytest.raises(EnvironmentRebuildMismatch):
                [e async for e in serving.resolve_user_input(response)]
            assert await serving.session_store.load(awaiting.session_id) == before
            assert await serving.session_store.load_checkpoint(awaiting.session_id) == checkpoint
            assert other.created == [] and rebuilt == []
        else:
            continued = [e async for e in serving.resolve_user_input(response)]
            assert continued[-1].type is EventType.SESSION_COMPLETED
            assert [f.revision for f in rebuilt] == ["job-b"]
        await serving.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid_registration", [False, True])
def test_recovery_handles_repeated_artifact_stores_and_registration_failure(
    tmp_path, invalid_registration
):
    class InvalidFactory(DeferredCloneFactory):
        @property
        def secret_resolution_scope(self):
            return "invalid"

    async def scenario():
        database = tmp_path / "recovery.sqlite"
        original_store = (
            None if invalid_registration else LocalArtifactStore(tmp_path / "artifacts")
        )
        _, _, first, stop_first = await _crash_job_a_in_reused_slot(
            database, artifact_store=original_store
        )
        worker, other, second, stop_second = await _crash_job_a_in_reused_slot(
            database, artifact_store=original_store
        )
        factories = []
        stores = []

        async def rebuild(request):
            factory_type = InvalidFactory if invalid_registration else DeferredCloneFactory
            factory = factory_type(request.record.payload["revision"])
            factories.append(factory)
            store = LocalArtifactStore(tmp_path / "artifacts")
            stores.append(store)
            return EnvironmentRebuild(spec=_slot_spec(), factory=factory, artifact_store=store)

        worker.register_environment_rebuilder("scout.clone", rebuild)
        try:
            for session_id in (first, second):
                recovered = await worker.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=session_id)
                )
                assert recovered.status is SessionStatus.INTERRUPTED
            assert other.created == [] and other.released == []
            assert len(factories) == 2
            if invalid_registration:
                assert all(not f.released for f in factories)
                assert worker._environment_registry.registered_artifact_stores() == ()
            else:
                assert [f.released for f in factories] == [
                    [(first, {"clone": "job-a"})],
                    [(second, {"clone": "job-a"})],
                ]
                assert stores[0] is not stores[1]
                assert worker._environment_registry.registered_artifact_stores() == (
                    original_store,
                )
        finally:
            await stop_second()
            await stop_first()

    asyncio.run(scenario())


@pytest.mark.parametrize("second_rebuild_fails", [False, True])
def test_concurrent_recoveries_never_use_another_jobs_rebuilt_factory(
    tmp_path: Path, monkeypatch, caplog, second_rebuild_fails: bool
) -> None:
    from contextlib import asynccontextmanager

    async def scenario() -> None:
        database = tmp_path / "concurrent.sqlite"
        crashed_apps = []
        runs = []
        sessions = []
        serving, _ = _crashing_app(SQLiteSessionStore(database))
        first_build_started = asyncio.Event()
        second_waiting = asyncio.Event()
        finish_first_build = asyncio.Event()
        factories: dict[str, DeferredCloneFactory] = {}
        recoveries = []

        async def run_job(app, events):
            async for event in app.run(
                RunRequest(
                    agent_name="hunter",
                    environment_name=SLOT,
                    messages=[Message.text("user", "go")],
                )
            ):
                events.append(event)

        async def rebuild(request: EnvironmentRebuildRequest) -> EnvironmentRebuild:
            revision = request.record.payload["revision"]
            if revision == "job-a":
                first_build_started.set()
                await finish_first_build.wait()
            elif second_rebuild_fails:
                raise ValueError("Job B cannot be rebuilt here.")
            factory = DeferredCloneFactory(revision)
            factories[revision] = factory
            return EnvironmentRebuild(spec=_slot_spec(), factory=factory)

        registry = serving._environment_registry
        original_lock = registry.rebuild_lock

        @asynccontextmanager
        async def observed_lock(name):
            lock = original_lock(name)
            if lock.locked():
                second_waiting.set()
            async with lock:
                yield

        monkeypatch.setattr(registry, "rebuild_lock", observed_lock)
        serving.register_environment_rebuilder("scout.clone", rebuild)
        try:
            for revision in ("job-a", "job-b"):
                crashed, tool = _crashing_app(SQLiteSessionStore(database))
                crashed_apps.append(crashed)
                crashed.register_environment_factory(_slot_spec(), DeferredCloneFactory(revision))
                events: list[Event] = []
                runs.append(asyncio.create_task(run_job(crashed, events)))
                await tool.started.wait()
                sessions.append(events[0].session_id)

            recoveries.append(
                asyncio.create_task(
                    serving.recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(session_id=sessions[0])
                    )
                )
            )
            await asyncio.wait_for(first_build_started.wait(), timeout=10)
            recoveries.append(
                asyncio.create_task(
                    serving.recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(session_id=sessions[1])
                    )
                )
            )
            # Both recoveries saw an unregistered slot. The second now waits
            # behind A's rebuild while A still holds the per-name lock.
            await asyncio.wait_for(second_waiting.wait(), timeout=10)
            finish_first_build.set()
            results = await asyncio.gather(*recoveries)
            assert [r.status for r in results] == [SessionStatus.INTERRUPTED] * 2
            assert factories["job-a"].released == [(sessions[0], {"clone": "job-a"})]
            assert serving.get_environment_factory(SLOT) is factories["job-a"]
            # Waiting for another record's rebuild is expected contention,
            # not a failed rebuild that should emit an exception warning.
            assert not any(
                record.exc_info and isinstance(record.exc_info[1], EnvironmentRebuildMismatch)
                for record in caplog.records
            )
            if second_rebuild_fails:
                assert "job-b" not in factories
                checkpoint = await serving.session_store.load_checkpoint(sessions[1])
                assert checkpoint is not None
                assert checkpoint["environment_factory_reconnect"][SLOT] == {"clone": "job-b"}
                assert checkpoint["environment_factory_allocation_owner"][SLOT] == sessions[1]
            else:
                assert factories["job-b"].released == [(sessions[1], {"clone": "job-b"})]
        finally:
            finish_first_build.set()
            for task in (*recoveries, *runs):
                task.cancel()
            await asyncio.gather(*recoveries, *runs, return_exceptions=True)
            await serving.aclose()
            for app in crashed_apps:
                with suppress(BaseException):
                    await app.aclose()

    asyncio.run(scenario())
